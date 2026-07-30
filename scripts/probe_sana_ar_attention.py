#!/usr/bin/env python
"""Capture SANA AR action-query modality flow and video linear/flash branch norms."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from probe_sana_action_denoising import (  # noqa: E402
    parse_sigmas,
    predict_velocity,
    to_device,
)
from sana_wam.model.ar.attention_diagnostics import ARAttentionCapture  # noqa: E402
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


DEFAULT_RUN = Path(
    "/home/zch/wuji-openwam-dev/sandbox/"
    "sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25"
)
DEFAULT_PARENT = ROOT / "logs" / "sana_principles_audit_20260722"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", default="checkpoint_step_12000.safetensors")
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PARENT / "action_sigma_probe")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PARENT / "ar_attention_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--sigmas", default="1.0,0.9,0.5")
    parser.add_argument("--video-sigma", type=float, default=0.5)
    parser.add_argument(
        "--attention-eps",
        type=float,
        default=None,
        help="Optional counterfactual denominator epsilon; default keeps checkpoint value.",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def annotate(rows: list[dict[str, Any]], start: int, *, pair: int, sigma: float) -> None:
    for row in rows[start:]:
        row["pair_index"] = pair
        row["sigma"] = sigma
        if "chunk" in row:
            row["phase"] = "bootstrap" if row["chunk"] == 0 else "continuation"


def mean_aggregate(
    rows: list[dict[str, Any]],
    key_names: tuple[str, ...],
    value_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        key = tuple(row[name] for name in key_names)
        counts[key] += 1
        for name in value_names:
            buckets[key][name] += float(row[name])
    result = []
    for key in sorted(buckets):
        count = counts[key]
        result.append(
            {
                **dict(zip(key_names, key)),
                "record_count": count,
                **{name: buckets[key][name] / count for name in value_names},
            }
        )
    return result


def median_aggregate(
    rows: list[dict[str, Any]],
    key_names: tuple[str, ...],
    value_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = tuple(row[name] for name in key_names)
        for name in value_names:
            buckets[key][name].append(float(row[name]))
    return [
        {
            **dict(zip(key_names, key)),
            "record_count": len(next(iter(values.values()))),
            **{name: statistics.median(items) for name, items in values.items()},
        }
        for key, values in sorted(buckets.items())
    ]


def markdown(report: dict[str, Any]) -> str:
    attention = report["attention_aggregates"]
    video = report["video_branch_medians"]
    query_groups = report["query_group_medians"]

    def attention_row(sigma, layer, phase, component):
        return next(
            row
            for row in attention
            if row["sigma"] == sigma
            and row["layer"] == layer
            and row["phase"] == phase
            and row["component"] == component
        )

    def video_row(sigma, layer, scope="all"):
        return next(
            row
            for row in video
            if row["sigma"] == sigma and row["layer"] == layer and row["scope"] == scope
        )

    def query_row(sigma, layer, modality, copy):
        return next(
            row
            for row in query_groups
            if row["sigma"] == sigma
            and row["layer"] == layer
            and row["modality"] == modality
            and row["copy"] == copy
        )

    lines = [
        "# SANA AR attention-flow probe",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Inputs: {report['pair_count'] * 2} cached expert clips, sigmas `{report['sigmas']}`.",
        f"- AR denominator epsilon: `{report['attention_eps']}`.",
        "- The capture calls the production attention first and returns its tensor unchanged; diagnostics run afterward on detached tensors.",
        f"- Output equivalence: `{report['output_equivalence']['status']}` "
        f"(max action delta {report['output_equivalence']['action_max_abs_delta']:.3g}).",
        "",
        "## Mean modality flow across layers",
        "",
        "Denominator fraction is the component's share of the non-negative linear-attention normalizer. Contribution/output is the RMS of that component's normalized numerator divided by final attention-output RMS.",
        "",
        "| sigma | phase | component | denominator fraction | contribution/output RMS | cosine with output |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    summary = mean_aggregate(
        report["attention_records"],
        ("sigma", "phase", "component"),
        (
            "denominator_fraction_mean",
            "contribution_over_output_rms",
            "contribution_output_cosine",
        ),
    )
    for row in summary:
        lines.append(
            f"| {row['sigma']:.3f} | {row['phase']} | {row['component']} | "
            f"{row['denominator_fraction_mean']:.6f} | "
            f"{row['contribution_over_output_rms']:.6f} | "
            f"{row['contribution_output_cosine']:.6f} |"
        )

    endpoint = report["sigmas"][0]
    lines.extend(
        [
            "",
            f"## Layerwise action-query denominator at sigma={endpoint:.3f}",
            "",
            "| layer | boot current video | boot noisy action | cont current video | cont prior video | cont action history | cont noisy action |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in range(20):
        boot_video = attention_row(endpoint, layer, "bootstrap", "current_clean_video")
        boot_self = attention_row(endpoint, layer, "bootstrap", "within_frame_noisy_action")
        cont_current = attention_row(endpoint, layer, "continuation", "current_clean_video")
        cont_prior = attention_row(endpoint, layer, "continuation", "prior_clean_video")
        cont_history = attention_row(endpoint, layer, "continuation", "clean_action_history")
        cont_self = attention_row(endpoint, layer, "continuation", "within_frame_noisy_action")
        lines.append(
            f"| {layer} | {boot_video['denominator_fraction_mean']:.6f} | "
            f"{boot_self['denominator_fraction_mean']:.6f} | "
            f"{cont_current['denominator_fraction_mean']:.6f} | "
            f"{cont_prior['denominator_fraction_mean']:.6f} | "
            f"{cont_history['denominator_fraction_mean']:.6f} | "
            f"{cont_self['denominator_fraction_mean']:.6f} |"
        )

    lines.extend(
        [
            "",
            f"## Linear-attention denominator stability at sigma={endpoint:.3f}",
            "",
            "Medians across pairs and chunks; p0.1% is computed within each query group before aggregation.",
            "",
            "| layer | video-noisy denom p0.1% | video-noisy zero frac | video-noisy output RMS | video-noisy output max | action-noisy denom p0.1% | action-noisy output RMS |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in range(20):
        video_query = query_row(endpoint, layer, "video", "noisy")
        action_query = query_row(endpoint, layer, "action", "noisy")
        lines.append(
            f"| {layer} | {video_query['denominator_p001']:.3e} | "
            f"{video_query['denominator_zero_fraction']:.6f} | "
            f"{video_query['output_rms']:.3e} | {video_query['output_abs_max']:.3e} | "
            f"{action_query['denominator_p001']:.3e} | {action_query['output_rms']:.3e} |"
        )

    lines.extend(
        [
            "",
            f"## Video linear vs window-flash branch at sigma={endpoint:.3f}",
            "",
            "Values are medians across four expert pairs. `unchanged` is the exact BF16 element fraction unchanged by the whole block.",
            "",
            "| layer | scale | input residual RMS | gated linear RMS | gated flash RMS | full block delta RMS | delta/input | unchanged |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in range(20):
        row = video_row(endpoint, layer)
        lines.append(
            f"| {layer} | {row['learnable_fa_scale']:.6g} | "
            f"{row['block_input_residual_rms']:.3e} | {row['gated_linear_rms']:.3e} | "
            f"{row['gated_flash_rms']:.3e} | {row['full_block_delta_rms']:.3e} | "
            f"{row['full_block_delta_over_input_rms']:.3e} | "
            f"{row['full_block_unchanged_fraction']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sigmas = parse_sigmas(args.sigmas)
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    if args.attention_eps is not None:
        if args.attention_eps <= 0:
            raise ValueError("attention-eps must be positive")
        driver.eps = float(args.attention_eps)
    device = torch.device(args.device)
    capture = ARAttentionCapture(driver, capture_video_branches=True)
    equivalence: dict[str, Any] | None = None

    with capture:
        for pair_index in range(args.pairs):
            prepared_path = args.prepared_dir / f"prepared_pair_{pair_index:02d}.pt"
            if not prepared_path.is_file():
                raise FileNotFoundError(
                    f"missing {prepared_path}; run probe_sana_action_denoising.py first"
                )
            saved = torch.load(prepared_path, map_location="cpu", weights_only=False)
            payload = to_device(saved["tensors"], device, architecture.dtype)
            actions = payload["actions"]
            chunks = payload["input_latents"].shape[2] // int(
                architecture._ar_frame_chunk_size
            )
            tokens_per_chunk = actions.shape[1] // chunks
            proprio_per_chunk = payload["proprio_seq"][
                :, [chunk * tokens_per_chunk for chunk in range(chunks)]
            ]
            generator = torch.Generator(device=device).manual_seed(
                args.seed + pair_index * 1009
            )
            action_noise = torch.randn(
                actions.shape,
                generator=generator,
                device=device,
                dtype=architecture.dtype,
            )
            video_noise = torch.randn(
                payload["input_latents"].shape,
                generator=generator,
                device=device,
                dtype=architecture.dtype,
            )

            for sigma in sigmas:
                baseline = None
                if equivalence is None:
                    # Temporarily disable capture so the comparison is against the
                    # unwrapped production path on exactly the same tensors.
                    capture.__exit__(None, None, None)
                    baseline = predict_velocity(
                        architecture,
                        payload,
                        condition="correct",
                        sigma=sigma,
                        video_sigma=args.video_sigma,
                        action_noise=action_noise,
                        video_noise=video_noise,
                        proprio_per_chunk=proprio_per_chunk,
                    )
                    capture.__enter__()

                attention_start = len(capture.records)
                query_group_start = len(capture.query_group_records)
                video_start = len(capture.video_branch_records)
                prediction, target, x0_hat = predict_velocity(
                    architecture,
                    payload,
                    condition="correct",
                    sigma=sigma,
                    video_sigma=args.video_sigma,
                    action_noise=action_noise,
                    video_noise=video_noise,
                    proprio_per_chunk=proprio_per_chunk,
                )
                annotate(capture.records, attention_start, pair=pair_index, sigma=sigma)
                annotate(
                    capture.query_group_records,
                    query_group_start,
                    pair=pair_index,
                    sigma=sigma,
                )
                annotate(
                    capture.video_branch_records,
                    video_start,
                    pair=pair_index,
                    sigma=sigma,
                )
                if baseline is not None:
                    baseline_prediction, baseline_target, baseline_x0 = baseline
                    equivalence = {
                        "status": (
                            "PASS"
                            if torch.equal(prediction, baseline_prediction)
                            and torch.equal(target, baseline_target)
                            and torch.equal(x0_hat, baseline_x0)
                            else "FAIL"
                        ),
                        "action_tensor_equal": torch.equal(
                            prediction, baseline_prediction
                        ),
                        "target_tensor_equal": torch.equal(target, baseline_target),
                        "x0_tensor_equal": torch.equal(x0_hat, baseline_x0),
                        "action_max_abs_delta": float(
                            (prediction - baseline_prediction).abs().max().item()
                        ),
                    }
                    if equivalence["status"] != "PASS":
                        raise RuntimeError(f"diagnostic capture changed output: {equivalence}")
                print(
                    f"pair={pair_index} sigma={sigma:.3f} "
                    f"attention_records={len(capture.records) - attention_start} "
                    f"query_groups={len(capture.query_group_records) - query_group_start} "
                    f"video_records={len(capture.video_branch_records) - video_start}",
                    flush=True,
                )
            del payload, action_noise, video_noise
            torch.cuda.empty_cache()

    if equivalence is None:
        raise RuntimeError("no equivalence check ran")
    attention_values = (
        "key_tokens",
        "numerator_rms",
        "denominator_mean",
        "denominator_fraction_mean",
        "normalized_contribution_rms",
        "contribution_over_output_rms",
        "contribution_output_cosine",
        "output_rms",
        "reconstruction_rmse",
    )
    video_values = (
        "learnable_fa_scale",
        "linear_projected_rms",
        "flash_raw_rms",
        "flash_scaled_rms",
        "flash_scaled_over_linear_rms",
        "linear_flash_cosine",
        "combined_rms",
        "gated_linear_rms",
        "gated_flash_rms",
        "gated_flash_over_linear_rms",
        "gated_linear_flash_cosine",
        "block_input_residual_rms",
        "full_block_delta_rms",
        "full_block_delta_over_input_rms",
        "full_block_output_rms",
        "full_block_unchanged_fraction",
        "block_input_abs_median",
        "full_block_delta_abs_median",
    )
    query_group_values = (
        "query_tokens",
        "query_phi_zero_fraction",
        "denominator_zero_fraction",
        "denominator_below_1e_6_fraction",
        "denominator_min",
        "denominator_p001",
        "denominator_p01",
        "denominator_median",
        "denominator_mean",
        "output_rms",
        "output_abs_p01",
        "output_abs_median",
        "output_abs_max",
    )
    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "pair_count": args.pairs,
        "sigmas": sigmas,
        "video_sigma": args.video_sigma,
        "attention_eps": float(driver.eps),
        "output_equivalence": equivalence,
        "total_seconds": time.perf_counter() - started,
        "attention_records": capture.records,
        "attention_aggregates": mean_aggregate(
            capture.records,
            ("sigma", "layer", "phase", "component"),
            attention_values,
        ),
        "video_branch_records": capture.video_branch_records,
        "video_branch_aggregates": mean_aggregate(
            capture.video_branch_records,
            ("sigma", "layer", "scope"),
            video_values,
        ),
        "video_branch_medians": median_aggregate(
            capture.video_branch_records,
            ("sigma", "layer", "scope"),
            video_values,
        ),
        "query_group_records": capture.query_group_records,
        "query_group_medians": median_aggregate(
            capture.query_group_records,
            ("sigma", "layer", "modality", "copy"),
            query_group_values,
        ),
    }
    json_path = args.output_dir / "ar_attention_probe.json"
    markdown_path = args.output_dir / "ar_attention_probe.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
