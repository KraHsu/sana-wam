#!/usr/bin/env python
"""Compare Phase-1 and selected video weights on the cache-teacher action path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from probe_sana_action_denoising import parse_sigmas, to_device  # noqa: E402
from probe_sana_action_weight_overlay import (  # noqa: E402
    GROUP_PATTERNS,
    apply_groups,
    cache_group_tensors,
    matching_keys,
)
from probe_sana_ar_decomposition import (  # noqa: E402
    build_expert_history,
    query_action,
)
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--overlay-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--sigmas", default="1.0,0.9,0.5")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def bootstrap_mean_ci(
    values: np.ndarray, *, rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Expanded cache-teacher action gate",
        "",
        f"- Contexts: `{report['context_count']}`",
        f"- Gate: `{report['gate']}`",
        "- Criterion: fail only when a paired-context 95% bootstrap interval "
        "shows a strictly positive selected-minus-Phase-1 MSE degradation.",
        "",
        "| sigma | Phase-1 RMSE | selected RMSE | MSE delta | paired 95% CI | result |",
        "| ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["metrics"]:
        lines.append(
            f"| {row['sigma']:.3f} | {row['phase1_rmse']:.6f} | "
            f"{row['selected_rmse']:.6f} | {row['mse_delta']:+.6f} | "
            f"[{row['mse_delta_ci95_low']:+.6f}, "
            f"{row['mse_delta_ci95_high']:+.6f}] | {row['result']} |"
        )
    integrated = report["integrated"]
    lines.extend(
        [
            "",
            f"Integrated equal-sigma MSE delta: `{integrated['mse_delta']:+.6f}` "
            f"(95% CI `[{integrated['mse_delta_ci95_low']:+.6f}, "
            f"{integrated['mse_delta_ci95_high']:+.6f}]`).",
        ]
    )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sigmas = parse_sigmas(args.sigmas)
    if args.pairs <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("pairs and bootstrap samples must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.base_run_dir), device=args.device, ckpt_name=args.base_checkpoint
    )
    architecture.eval()
    device = torch.device(args.device)
    driver = architecture._mot_driver or architecture.build_mot_driver()
    model_state = architecture.state_dict(keep_vars=True)
    state_keys = list(model_state)
    group_keys = {
        group: matching_keys(state_keys, pattern)
        for group, pattern in GROUP_PATTERNS.items()
    }
    base_checkpoint = args.base_run_dir / args.base_checkpoint
    base_tensors = cache_group_tensors(base_checkpoint, group_keys)
    overlay_tensors = cache_group_tensors(args.overlay_checkpoint, group_keys)
    prepared = [
        torch.load(
            args.prepared_dir / f"prepared_pair_{pair_index:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for pair_index in range(args.pairs)
    ]
    conditions = {
        "phase1_video": (),
        "selected_video": tuple(GROUP_PATTERNS),
    }
    context_metrics: dict[str, dict[float, dict[int, float]]] = {
        name: {sigma: {} for sigma in sigmas} for name in conditions
    }

    for condition_name, enabled_groups in conditions.items():
        apply_groups(model_state, base_tensors, overlay_tensors, enabled_groups)
        for pair_index, saved in enumerate(prepared):
            payload = to_device(saved["tensors"], device, architecture.dtype)
            metadata = saved["samples"]
            actions = payload["actions"]
            latent_frames = payload["input_latents"].shape[2]
            chunks = latent_frames // int(architecture._ar_frame_chunk_size)
            tokens_per_chunk = actions.shape[1] // chunks
            video_chunks = list(
                payload["input_latents"].split(
                    int(architecture._ar_frame_chunk_size), dim=2
                )
            )
            action_chunks = list(actions.split(tokens_per_chunk, dim=1))
            proprio_chunks = [
                payload["proprio_seq"][:, chunk * tokens_per_chunk]
                for chunk in range(chunks)
            ]
            context, context_mask = architecture._rollout_step_context(
                payload["context"], payload["seq_lens"], payload["proprio_state"]
            )
            generator = torch.Generator(device=device).manual_seed(
                args.seed + pair_index * 1009
            )
            action_noise = torch.randn(
                actions.shape,
                generator=generator,
                device=device,
                dtype=architecture.dtype,
            )
            square_sums = {
                sigma: torch.zeros(actions.shape[0], dtype=torch.float64)
                for sigma in sigmas
            }
            counts = {
                sigma: torch.zeros(actions.shape[0], dtype=torch.int64)
                for sigma in sigmas
            }
            for chunk in range(chunks):
                video_proprio, action_proprio = architecture._rollout_proprio_deltas(
                    proprio_chunks[chunk]
                )
                teacher_cache = build_expert_history(
                    architecture,
                    driver,
                    video_chunks,
                    action_chunks,
                    proprio_chunks,
                    stop_chunk=chunk,
                    context=context,
                    context_mask=context_mask,
                )
                architecture._ingest_clean_video(
                    driver,
                    teacher_cache,
                    video_chunks[chunk],
                    frame_id=2 * chunk,
                    context=context,
                    context_mask=context_mask,
                    v_proprio=video_proprio,
                )
                start = chunk * tokens_per_chunk
                end = start + tokens_per_chunk
                action_chunk = action_chunks[chunk]
                noise_chunk = action_noise[:, start:end]
                valid = torch.ones_like(action_chunk, dtype=torch.bool)
                if "action_is_pad" in payload:
                    valid &= (~payload["action_is_pad"][:, start:end]).unsqueeze(-1)
                for sigma in sigmas:
                    noisy_action = (
                        (1.0 - sigma) * action_chunk + sigma * noise_chunk
                    )
                    prediction = query_action(
                        architecture,
                        driver,
                        teacher_cache,
                        noisy_action,
                        sigma=sigma,
                        frame_id=2 * chunk + 1,
                        context=context,
                        context_mask=context_mask,
                        action_proprio=action_proprio,
                    )
                    target = noise_chunk.float() - action_chunk.float()
                    error = (prediction - target).double().square()
                    for context_index in range(actions.shape[0]):
                        mask = valid[context_index]
                        square_sums[sigma][context_index] += error[context_index][
                            mask
                        ].sum().cpu()
                        counts[sigma][context_index] += int(mask.sum())
            for context_index, item in enumerate(metadata):
                episode = int(item["episode_index"])
                for sigma in sigmas:
                    context_metrics[condition_name][sigma][episode] = float(
                        square_sums[sigma][context_index]
                        / counts[sigma][context_index]
                    )
            print(
                f"{condition_name} pair {pair_index + 1}/{args.pairs} complete",
                flush=True,
            )

    rng = np.random.default_rng(args.seed)
    metrics = []
    all_differences = []
    sigma_results = []
    for sigma in sigmas:
        base = context_metrics["phase1_video"][sigma]
        selected = context_metrics["selected_video"][sigma]
        if base.keys() != selected.keys():
            raise RuntimeError("condition context identities differ")
        episodes = sorted(base)
        differences = np.asarray(
            [selected[episode] - base[episode] for episode in episodes]
        )
        low, high = bootstrap_mean_ci(
            differences, rng=rng, samples=args.bootstrap_samples
        )
        base_mse = float(np.mean(list(base.values())))
        selected_mse = float(np.mean(list(selected.values())))
        result = "FAIL" if low > 0.0 else "PASS"
        sigma_results.append(result)
        all_differences.append(differences)
        metrics.append(
            {
                "sigma": sigma,
                "phase1_mse": base_mse,
                "phase1_rmse": base_mse**0.5,
                "selected_mse": selected_mse,
                "selected_rmse": selected_mse**0.5,
                "mse_delta": float(differences.mean()),
                "mse_delta_ci95_low": low,
                "mse_delta_ci95_high": high,
                "result": result,
            }
        )
    integrated_differences = np.stack(all_differences).mean(axis=0)
    integrated_low, integrated_high = bootstrap_mean_ci(
        integrated_differences, rng=rng, samples=args.bootstrap_samples
    )
    integrated_result = "FAIL" if integrated_low > 0.0 else "PASS"
    gate = (
        "PASS"
        if all(result == "PASS" for result in sigma_results)
        and integrated_result == "PASS"
        else "FAIL"
    )
    report = {
        "base_checkpoint": str(base_checkpoint),
        "overlay_checkpoint": str(args.overlay_checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "pair_count": args.pairs,
        "context_count": args.pairs * 2,
        "seed": args.seed,
        "sigmas": sigmas,
        "bootstrap_samples": args.bootstrap_samples,
        "metrics": metrics,
        "integrated": {
            "mse_delta": float(integrated_differences.mean()),
            "mse_delta_ci95_low": integrated_low,
            "mse_delta_ci95_high": integrated_high,
            "result": integrated_result,
        },
        "gate": gate,
        "context_metrics": {
            condition: {
                str(sigma): values for sigma, values in by_sigma.items()
            }
            for condition, by_sigma in context_metrics.items()
        },
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "action_cache_gate.json"
    markdown_path = args.output_dir / "action_cache_gate.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
