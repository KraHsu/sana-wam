#!/usr/bin/env python
"""Localize video-to-action coupling with in-memory checkpoint overlays."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from probe_sana_action_denoising import (  # noqa: E402
    parse_sigmas,
    predict_velocity,
    scope_masks,
    to_device,
)
from probe_sana_ar_decomposition import (  # noqa: E402
    build_expert_history,
    query_action,
)
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


GROUP_PATTERNS = {
    "linear_attention": "video_backbone.dit.blocks.*.attn.*",
    "window_flash": "video_backbone.dit.blocks.*.flash_attn_additional.*",
    "temporal_conv": "video_backbone.dit.blocks.*.mlp.t_conv.*",
}
CONDITIONS = (
    ("phase1_video", ()),
    ("linear_attention", ("linear_attention",)),
    ("window_flash", ("window_flash",)),
    ("temporal_conv", ("temporal_conv",)),
    ("linear_attention+window_flash", ("linear_attention", "window_flash")),
    ("linear_attention+temporal_conv", ("linear_attention", "temporal_conv")),
    ("window_flash+temporal_conv", ("window_flash", "temporal_conv")),
    (
        "all_video_groups",
        ("linear_attention", "window_flash", "temporal_conv"),
    ),
)
REPORTED_SCOPES = (
    "all",
    "bootstrap",
    "continuation",
    "chunk_0",
    "chunk_1",
    "chunk_2",
    "chunk_3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--overlay-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--sigmas", default="1.0,0.9,0.5")
    parser.add_argument("--video-sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


@dataclass
class ErrorSums:
    count: int = 0
    velocity_error_sq: float = 0.0
    x0_error_sq: float = 0.0
    velocity_delta_sq: float = 0.0
    x0_delta_sq: float = 0.0

    def add(
        self,
        mask: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        x0_hat: torch.Tensor,
        actions: torch.Tensor,
        baseline_prediction: torch.Tensor,
        baseline_x0: torch.Tensor,
    ) -> None:
        count = int(mask.sum())
        if count == 0:
            return
        self.count += count

        def square_sum(value: torch.Tensor) -> float:
            return float(value[mask].double().square().sum().item())

        self.velocity_error_sq += square_sum(prediction - target)
        self.x0_error_sq += square_sum(x0_hat - actions.float())
        self.velocity_delta_sq += square_sum(prediction - baseline_prediction)
        self.x0_delta_sq += square_sum(x0_hat - baseline_x0)

    def final(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0}

        def rms(total: float) -> float:
            return (total / self.count) ** 0.5

        return {
            "count": self.count,
            "velocity_rmse": rms(self.velocity_error_sq),
            "x0_rmse": rms(self.x0_error_sq),
            "velocity_delta_from_phase1_video_rms": rms(self.velocity_delta_sq),
            "x0_delta_from_phase1_video_rms": rms(self.x0_delta_sq),
        }


def matching_keys(keys: list[str], pattern: str) -> list[str]:
    return sorted(key for key in keys if fnmatch.fnmatchcase(key, pattern))


def cache_group_tensors(
    path: Path,
    group_keys: dict[str, list[str]],
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
        for group, keys in group_keys.items():
            missing = sorted(set(keys) - checkpoint_keys)
            if missing:
                raise KeyError(f"{path} is missing {group} keys: {missing[:5]}")
            result[group] = {
                key: checkpoint.get_tensor(key).clone() for key in keys
            }
    return result


@torch.no_grad()
def apply_groups(
    model_state: dict[str, torch.Tensor],
    base_tensors: dict[str, dict[str, torch.Tensor]],
    overlay_tensors: dict[str, dict[str, torch.Tensor]],
    enabled_groups: tuple[str, ...],
) -> None:
    enabled = set(enabled_groups)
    for group in GROUP_PATTERNS:
        source = overlay_tensors[group] if group in enabled else base_tensors[group]
        for key, value in source.items():
            destination = model_state[key]
            destination.copy_(value.to(device=destination.device))


def markdown(report: dict[str, Any]) -> str:
    sigmas = report["sigmas"]
    rows = {
        (item["condition"], item["sigma"]): item
        for item in report["metrics"]
        if item["scope"] == "all"
    }
    lines = [
        "# SANA video-weight overlay action probe",
        "",
        "Every condition uses Phase-1 action weights and the chunkwise temporal "
        "execution mode. Only the named video parameter groups are overlaid from "
        "chunkwise step 1000 in memory.",
        "",
        "| Condition | "
        + " | ".join(f"sigma {sigma:g} RMSE" for sigma in sigmas)
        + " | sigma 1 delta from base output |",
        "| --- | " + " | ".join("---:" for _ in sigmas) + " | ---: |",
    ]
    for condition in report["conditions"]:
        name = condition["name"]
        values = [rows[(name, sigma)]["velocity_rmse"] for sigma in sigmas]
        delta = rows[(name, sigmas[0])][
            "velocity_delta_from_phase1_video_rms"
        ]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{value:.6f}" for value in values)
            + f" | {delta:.6f} |"
        )
    cache_rows = {
        (item["condition"], item["sigma"]): item
        for item in report["cache_metrics"]
        if item["scope"] == "all"
    }
    lines.extend(
        [
            "",
            "## Cache-teacher action path",
            "",
            "The cache path ingests exact expert video/action history and the "
            "current clean expert video chunk before querying the current noisy "
            "action chunk.",
            "",
            "| Condition | "
            + " | ".join(f"sigma {sigma:g} RMSE" for sigma in sigmas)
            + " | sigma 1 delta from base output |",
            "| --- | " + " | ".join("---:" for _ in sigmas) + " | ---: |",
        ]
    )
    for condition in report["conditions"]:
        name = condition["name"]
        values = [cache_rows[(name, sigma)]["velocity_rmse"] for sigma in sigmas]
        delta = cache_rows[(name, sigmas[0])][
            "velocity_delta_from_phase1_video_rms"
        ]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{value:.6f}" for value in values)
            + f" | {delta:.6f} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sigmas = parse_sigmas(args.sigmas)
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    _, architecture = load_from_checkpoint_dir(
        str(args.base_run_dir),
        device=args.device,
        ckpt_name=args.base_checkpoint,
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
    seen: set[str] = set()
    for group, keys in group_keys.items():
        if not keys:
            raise RuntimeError(f"video overlay group {group!r} matched no tensors")
        overlap = seen.intersection(keys)
        if overlap:
            raise RuntimeError(f"video overlay groups overlap: {sorted(overlap)[:5]}")
        seen.update(keys)

    base_checkpoint = args.base_run_dir / args.base_checkpoint
    base_tensors = cache_group_tensors(base_checkpoint, group_keys)
    overlay_tensors = cache_group_tensors(args.overlay_checkpoint, group_keys)
    group_contract = {}
    for group, keys in group_keys.items():
        parameter_count = sum(model_state[key].numel() for key in keys)
        changed = sum(
            int(torch.count_nonzero(base_tensors[group][key] != overlay_tensors[group][key]))
            for key in keys
        )
        group_contract[group] = {
            "pattern": GROUP_PATTERNS[group],
            "tensor_count": len(keys),
            "parameter_count": parameter_count,
            "changed_element_count": changed,
            "changed_fraction": changed / parameter_count,
        }

    prepared = []
    for pair_index in range(args.pairs):
        path = args.prepared_dir / f"prepared_pair_{pair_index:02d}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        prepared.append(saved)

    baseline_outputs: dict[
        tuple[int, float], tuple[torch.Tensor, torch.Tensor]
    ] = {}
    baseline_cache_outputs: dict[
        tuple[int, int, float], tuple[torch.Tensor, torch.Tensor]
    ] = {}
    metrics = []
    cache_metrics = []
    condition_seconds = {}
    for condition_name, enabled_groups in CONDITIONS:
        condition_started = time.perf_counter()
        apply_groups(model_state, base_tensors, overlay_tensors, enabled_groups)
        accumulators: dict[tuple[float, str], ErrorSums] = defaultdict(ErrorSums)
        cache_accumulators: dict[tuple[float, str], ErrorSums] = defaultdict(
            ErrorSums
        )
        for pair_index, saved in enumerate(prepared):
            payload = to_device(saved["tensors"], device, architecture.dtype)
            metadata = saved["samples"]
            actions = payload["actions"]
            latent_frames = payload["input_latents"].shape[2]
            chunks = latent_frames // int(architecture._ar_frame_chunk_size)
            tokens_per_chunk = actions.shape[1] // chunks
            chunk_indices = [chunk * tokens_per_chunk for chunk in range(chunks)]
            proprio_per_chunk = payload["proprio_seq"][:, chunk_indices]
            all_masks = scope_masks(payload, metadata, chunks)
            masks = {scope: all_masks[scope] for scope in REPORTED_SCOPES}
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
                payload["context"],
                payload["seq_lens"],
                payload["proprio_state"],
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
            video_noise = torch.randn(
                payload["input_latents"].shape,
                generator=generator,
                device=device,
                dtype=architecture.dtype,
            )
            for sigma in sigmas:
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
                cache_key = (pair_index, sigma)
                if condition_name == "phase1_video":
                    baseline_outputs[cache_key] = (
                        prediction.cpu(),
                        x0_hat.cpu(),
                    )
                baseline_prediction_cpu, baseline_x0_cpu = baseline_outputs[
                    cache_key
                ]
                baseline_prediction = baseline_prediction_cpu.to(device)
                baseline_x0 = baseline_x0_cpu.to(device)
                for scope, mask in masks.items():
                    accumulators[(sigma, scope)].add(
                        mask,
                        prediction,
                        target,
                        x0_hat,
                        actions,
                        baseline_prediction,
                        baseline_x0,
                    )

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
                    x0_hat = noisy_action.float() - sigma * prediction
                    cache_key = (pair_index, chunk, sigma)
                    if condition_name == "phase1_video":
                        baseline_cache_outputs[cache_key] = (
                            prediction.cpu(),
                            x0_hat.cpu(),
                        )
                    baseline_prediction_cpu, baseline_x0_cpu = (
                        baseline_cache_outputs[cache_key]
                    )
                    baseline_prediction = baseline_prediction_cpu.to(device)
                    baseline_x0 = baseline_x0_cpu.to(device)
                    phase = "bootstrap" if chunk == 0 else "continuation"
                    for scope in ("all", phase, f"chunk_{chunk}"):
                        cache_accumulators[(sigma, scope)].add(
                            valid,
                            prediction,
                            target,
                            x0_hat,
                            action_chunk,
                            baseline_prediction,
                            baseline_x0,
                        )
            del payload, action_noise, video_noise

        for (sigma, scope), sums in sorted(accumulators.items()):
            metrics.append(
                {
                    "condition": condition_name,
                    "enabled_groups": list(enabled_groups),
                    "sigma": sigma,
                    "scope": scope,
                    **sums.final(),
                }
            )
        for (sigma, scope), sums in sorted(cache_accumulators.items()):
            cache_metrics.append(
                {
                    "condition": condition_name,
                    "enabled_groups": list(enabled_groups),
                    "sigma": sigma,
                    "scope": scope,
                    **sums.final(),
                }
            )
        condition_seconds[condition_name] = time.perf_counter() - condition_started
        all_row = next(
            item
            for item in metrics
            if item["condition"] == condition_name
            and item["sigma"] == sigmas[0]
            and item["scope"] == "all"
        )
        cache_all_row = next(
            item
            for item in cache_metrics
            if item["condition"] == condition_name
            and item["sigma"] == sigmas[0]
            and item["scope"] == "all"
        )
        print(
            f"condition={condition_name} sigma1_rmse={all_row['velocity_rmse']:.6f} "
            f"cache_sigma1_rmse={cache_all_row['velocity_rmse']:.6f} "
            f"delta={all_row['velocity_delta_from_phase1_video_rms']:.6f} "
            f"seconds={condition_seconds[condition_name]:.1f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    report = {
        "base_checkpoint": str(base_checkpoint),
        "overlay_checkpoint": str(args.overlay_checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "seed": args.seed,
        "sigmas": sigmas,
        "video_sigma": args.video_sigma,
        "pair_count": args.pairs,
        "conditions": [
            {"name": name, "enabled_groups": list(groups)}
            for name, groups in CONDITIONS
        ],
        "group_contract": group_contract,
        "condition_seconds": condition_seconds,
        "metrics": metrics,
        "cache_metrics": cache_metrics,
        "cache_contract": {
            "clean_expert_video_action_history": True,
            "clean_current_expert_video": True,
            "current_action_is_noised": True,
            "closed_loop_simulation": False,
        },
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "action_weight_overlay.json"
    markdown_path = args.output_dir / "action_weight_overlay.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
