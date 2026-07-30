#!/usr/bin/env python
"""Trace same-noise SANA Euler trajectories and isolate BF16 accumulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT if (ROOT / "src").is_dir() else ROOT.parent / "phase4"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Sana"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from probe_sana_action_denoising import to_device  # noqa: E402
from probe_sana_ar_decomposition import build_expert_history  # noqa: E402
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


MODE_NAMES = {
    "native": "bf16_forward_bf16_accumulation",
    "fp32_accum": "bf16_forward_fp32_accumulation",
    "fp32_accum_fp32_time": "bf16_forward_fp32_accumulation_fp32_time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair-index", type=int, required=True)
    parser.add_argument("--context-index", type=int, required=True)
    parser.add_argument("--expected-episode", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--steps", default="20,50,100,200")
    parser.add_argument("--common-progress", type=int, default=10)
    parser.add_argument("--modes", default="native,fp32_accum")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def rms(value: torch.Tensor) -> float:
    return float(value.double().square().mean().sqrt().item())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.double().flatten()
    right = right.double().flatten()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right).div(denominator).item())


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_stats(value: torch.Tensor) -> dict[str, float | int]:
    flat = value.detach().float().flatten()
    quantiles = torch.quantile(flat.abs(), torch.tensor([0.99], device=flat.device))
    return {
        "rms": rms(flat),
        "abs_p99": float(quantiles[0].item()),
        "abs_max": float(flat.abs().max().item()),
        "nonfinite_count": int((~torch.isfinite(flat)).sum().item()),
    }


def bf16_ulp(value: torch.Tensor) -> torch.Tensor:
    absolute = value.detach().float().abs()
    normal = torch.pow(2.0, torch.floor(torch.log2(absolute.clamp_min(2.0**-126))) - 7.0)
    return torch.where(absolute == 0.0, torch.full_like(absolute, 2.0**-133), normal)


def cache_signature(cache) -> list[list[tuple[int, bool, int, int, float, float]]]:
    result = []
    for entries in cache.snapshot():
        layer = []
        for entry in entries:
            layer.append(
                (
                    int(entry["frame_id"]),
                    bool(entry["is_pred"]),
                    int(entry["S"].data_ptr()),
                    int(entry["z"].data_ptr()),
                    float(entry["S"].double().sum().item()),
                    float(entry["z"].double().sum().item()),
                )
            )
        result.append(layer)
    return result


@dataclass
class TraceResult:
    mode: str
    steps: int
    repeat: int
    endpoint_full: torch.Tensor
    endpoint_target: torch.Tensor
    common: dict[int, dict[str, torch.Tensor | float]]
    step_metrics: list[dict[str, Any]]
    noise_sha256: str
    cache_unchanged: bool
    requested_timesteps: list[float]
    cast_timesteps: list[float]


def prepare_velocity(
    architecture,
    driver,
    cache,
    *,
    x_model: torch.Tensor,
    t_value: float,
    time_dtype: torch.dtype,
    frame_id: int,
    rope_index: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    frame_proprio: torch.Tensor | None,
) -> torch.Tensor:
    batch, frames = x_model.shape[0], x_model.shape[2]
    extra = (
        {}
        if frame_proprio is None
        else {"frame_proprio_emb": frame_proprio}
    )
    timestep = torch.full(
        (batch,), t_value, device=x_model.device, dtype=time_dtype
    )
    frame_timesteps = torch.full(
        (batch, frames), t_value, device=x_model.device, dtype=time_dtype
    )
    state = architecture.video_backbone.prepare(
        latents=x_model,
        timestep=timestep,
        frame_timesteps=frame_timesteps,
        rope_frame_index=rope_index,
        context=context,
        context_mask=context_mask,
        **extra,
    )
    driver.run_ar_chunk_through_backbone(
        architecture.video_backbone,
        state,
        cache,
        frame_id,
        store_clean=False,
    )
    return architecture.video_backbone.finalize(state)


@torch.inference_mode()
def trace_one(
    architecture,
    driver,
    *,
    payload: dict[str, torch.Tensor],
    pair_index: int,
    context_index: int,
    chunk: int,
    steps: int,
    common_progress: int,
    mode: str,
    repeat: int,
    seed: int,
) -> TraceResult:
    actions = payload["actions"]
    latents = payload["input_latents"]
    frame_chunk_size = int(architecture._ar_frame_chunk_size)
    chunks = latents.shape[2] // frame_chunk_size
    tokens_per_chunk = actions.shape[1] // chunks
    video_chunks = list(latents.split(frame_chunk_size, dim=2))
    action_chunks = list(actions.split(tokens_per_chunk, dim=1))
    proprio_chunks = [
        payload["proprio_seq"][:, index * tokens_per_chunk]
        for index in range(chunks)
    ]
    context, context_mask = architecture._rollout_step_context(
        payload["context"], payload["seq_lens"], payload["proprio_state"]
    )
    cache = build_expert_history(
        architecture,
        driver,
        video_chunks,
        action_chunks,
        proprio_chunks,
        stop_chunk=chunk,
        context=context,
        context_mask=context_mask,
    )
    cache_before = cache_signature(cache)
    video_proprio, _ = architecture._rollout_proprio_deltas(proprio_chunks[chunk])
    batch, frames = video_chunks[chunk].shape[0], video_chunks[chunk].shape[2]
    frame_proprio = video_proprio[:, None, :].expand(batch, frames, -1)
    frame_id = 2 * chunk
    start_frame = (frame_id // 2) * frames
    rope_index = torch.arange(
        start_frame, start_frame + frames, device=video_chunks[chunk].device
    )

    architecture.video_backbone.scheduler.set_timesteps(steps)
    sigmas = [
        float(value)
        for value in architecture.video_backbone.scheduler.sigmas.tolist()
    ] + [0.0]
    timesteps = [
        float(value)
        for value in architecture.video_backbone.scheduler.timesteps.tolist()
    ]
    generator = torch.Generator(device=video_chunks[chunk].device).manual_seed(
        seed + pair_index * 1009 + chunk * 100_003
    )
    noise = torch.randn(
        video_chunks[chunk].shape,
        generator=generator,
        device=video_chunks[chunk].device,
        dtype=architecture.dtype,
    )
    noise_sha = tensor_sha256(noise)
    native = mode == "native"
    fp32_time = mode == "fp32_accum_fp32_time"
    master = noise.clone() if native else noise.float()
    time_dtype = torch.float32 if fp32_time else architecture.dtype
    common_indices = {
        progress * steps // common_progress: progress
        for progress in range(common_progress + 1)
    }
    common: dict[int, dict[str, torch.Tensor | float]] = {}
    step_metrics = []
    requested_timesteps = []
    cast_timesteps = []

    for index in range(steps):
        model_state = master if native else master.to(architecture.dtype)
        t_value = timesteps[index]
        if index in common_indices:
            progress = common_indices[index]
            common[progress] = {
                "sigma": sigmas[index],
                "state_master": master[context_index].detach().float().cpu(),
                "state_model": model_state[context_index].detach().float().cpu(),
            }
        velocity = prepare_velocity(
            architecture,
            driver,
            cache,
            x_model=model_state,
            t_value=t_value,
            time_dtype=time_dtype,
            frame_id=frame_id,
            rope_index=rope_index,
            context=context,
            context_mask=context_mask,
            frame_proprio=frame_proprio,
        )
        if index in common_indices:
            common[common_indices[index]]["velocity"] = (
                velocity[context_index].detach().float().cpu()
            )
        delta_sigma = sigmas[index + 1] - sigmas[index]
        ideal_update = velocity.float() * delta_sigma
        if native:
            next_master = master + velocity * delta_sigma
        else:
            next_master = master + ideal_update
        next_model_state = (
            next_master if native else next_master.to(architecture.dtype)
        )
        target_state = model_state[context_index].float()
        target_ideal = ideal_update[context_index]
        target_realized_master = (
            next_master[context_index].float() - master[context_index].float()
        )
        target_realized_model = (
            next_model_state[context_index].float() - target_state
        )
        ulp = bf16_ulp(target_state)
        timestep_input = torch.tensor(t_value, dtype=time_dtype)
        norm_factor = float(
            architecture.video_backbone._dit.timestep_norm_scale_factor
        )
        if norm_factor != 1.0:
            effective_timestep = float(
                (timestep_input.float() / norm_factor).item()
            )
        else:
            effective_timestep = float(timestep_input.long().float().item())
        requested_timesteps.append(t_value)
        cast_timesteps.append(effective_timestep)
        step_metrics.append(
            {
                "index": index,
                "sigma": sigmas[index],
                "next_sigma": sigmas[index + 1],
                "delta_sigma": delta_sigma,
                "requested_timestep": t_value,
                "model_timestep": effective_timestep,
                "state": tensor_stats(target_state),
                "velocity": tensor_stats(velocity[context_index]),
                "ideal_fp32_update": tensor_stats(target_ideal),
                "realized_master_update": tensor_stats(target_realized_master),
                "realized_model_visible_update": tensor_stats(target_realized_model),
                "master_rounding_residual_rms": rms(
                    target_realized_master - target_ideal
                ),
                "model_visible_rounding_residual_rms": rms(
                    target_realized_model - target_ideal
                ),
                "model_visible_zero_update_fraction": float(
                    (target_realized_model == 0.0).float().mean().item()
                ),
                "ideal_below_half_ulp_fraction": float(
                    (target_ideal.abs() < 0.5 * ulp).float().mean().item()
                ),
                "ideal_below_one_ulp_fraction": float(
                    (target_ideal.abs() < ulp).float().mean().item()
                ),
                "state_quantization_rms": rms(
                    master[context_index].float() - target_state
                ),
            }
        )
        master = next_master

    final_model_state = master if native else master.to(architecture.dtype)
    common[common_progress] = {
        "sigma": 0.0,
        "state_master": master[context_index].detach().float().cpu(),
        "state_model": final_model_state[context_index].detach().float().cpu(),
    }
    cache_after = cache_signature(cache)
    return TraceResult(
        mode=mode,
        steps=steps,
        repeat=repeat,
        endpoint_full=master.detach().float().cpu(),
        endpoint_target=master[context_index].detach().float().cpu(),
        common=common,
        step_metrics=step_metrics,
        noise_sha256=noise_sha,
        cache_unchanged=cache_before == cache_after,
        requested_timesteps=requested_timesteps,
        cast_timesteps=cast_timesteps,
    )


def find_context(reference: dict[str, Any], episode: int) -> dict[str, Any]:
    matches = [
        row for row in reference["context_metrics"] if row["episode_index"] == episode
    ]
    if len(matches) != 1:
        raise RuntimeError("reference context is missing or duplicated")
    return matches[0]


def solution(row: dict[str, Any], steps: int) -> dict[str, Any]:
    matches = [item for item in row["solutions"] if item["steps"] == steps]
    if len(matches) != 1:
        raise RuntimeError(f"reference step {steps} is missing or duplicated")
    return matches[0]


def endpoint_metrics(
    results: dict[tuple[str, int, int], TraceResult],
    *,
    modes: list[str],
    steps: list[int],
    target: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints = []
    pairwise = []
    precision = []
    for mode in modes:
        for step_count in steps:
            result = results[(mode, step_count, 0)]
            endpoints.append(
                {
                    "mode": MODE_NAMES[mode],
                    "steps": step_count,
                    "paired_expert_rmse": rms(result.endpoint_target - target),
                    "endpoint_rms": rms(result.endpoint_target),
                    "endpoint_abs_max": float(result.endpoint_target.abs().max().item()),
                }
            )
        for left_index, left in enumerate(steps):
            for right in steps[left_index + 1 :]:
                pairwise.append(
                    {
                        "mode": MODE_NAMES[mode],
                        "left_steps": left,
                        "right_steps": right,
                        "endpoint_delta_rms": rms(
                            results[(mode, left, 0)].endpoint_target
                            - results[(mode, right, 0)].endpoint_target
                        ),
                    }
                )
    if "native" in modes and "fp32_accum" in modes:
        for step_count in steps:
            precision.append(
                {
                    "steps": step_count,
                    "native_vs_fp32_accum_endpoint_rms": rms(
                        results[("native", step_count, 0)].endpoint_target
                        - results[("fp32_accum", step_count, 0)].endpoint_target
                    ),
                }
            )
    return endpoints, pairwise, precision


def common_sigma_metrics(
    results: dict[tuple[str, int, int], TraceResult],
    *,
    modes: list[str],
    steps: list[int],
    common_progress: int,
) -> list[dict[str, Any]]:
    finest = max(steps)
    rows = []
    for mode in modes:
        reference = results[(mode, finest, 0)]
        for coarse in steps:
            if coarse == finest:
                continue
            candidate = results[(mode, coarse, 0)]
            for progress in range(common_progress + 1):
                left = candidate.common[progress]
                right = reference.common[progress]
                state_delta = left["state_master"] - right["state_master"]
                model_delta = left["state_model"] - right["state_model"]
                row = {
                    "mode": MODE_NAMES[mode],
                    "coarse_steps": coarse,
                    "fine_steps": finest,
                    "progress_numerator": progress,
                    "progress_denominator": common_progress,
                    "sigma": float(right["sigma"]),
                    "state_master_delta_rms": rms(state_delta),
                    "state_model_delta_rms": rms(model_delta),
                    "state_model_delta_abs_p99": float(
                        torch.quantile(model_delta.abs().flatten(), 0.99).item()
                    ),
                    "state_model_delta_abs_max": float(
                        model_delta.abs().max().item()
                    ),
                }
                if progress < common_progress:
                    velocity_delta = left["velocity"] - right["velocity"]
                    state_rms = rms(model_delta)
                    velocity_rms = rms(velocity_delta)
                    row.update(
                        {
                            "velocity_delta_rms": velocity_rms,
                            "velocity_delta_abs_max": float(
                                velocity_delta.abs().max().item()
                            ),
                            "secant_lipschitz": (
                                None if state_rms == 0.0 else velocity_rms / state_rms
                            ),
                            "signed_secant_slope": (
                                None
                                if state_rms == 0.0
                                else float(
                                    torch.dot(
                                        velocity_delta.double().flatten(),
                                        model_delta.double().flatten(),
                                    ).item()
                                    / model_delta.double().square().sum().item()
                                )
                            ),
                            "delta_velocity_state_cosine": cosine(
                                velocity_delta, model_delta
                            ),
                        }
                    )
                rows.append(row)
    return rows


def first_divergence(
    common_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in common_rows:
        key = (row["mode"], row["coarse_steps"], row["fine_steps"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda item: item["progress_numerator"])
        first_above = next(
            (
                row
                for row in rows
                if row["state_model_delta_rms"] > 0.05
            ),
            None,
        )
        first_jump = None
        for previous, current in zip(rows, rows[1:]):
            floor = 1e-12
            ratio = current["state_model_delta_rms"] / max(
                previous["state_model_delta_rms"], floor
            )
            if current["state_model_delta_rms"] > 0.05 and ratio >= 4.0:
                first_jump = {"row": current, "growth_ratio": ratio}
                break
        result.append(
            {
                "mode": key[0],
                "coarse_steps": key[1],
                "fine_steps": key[2],
                "first_state_delta_above_0_05": first_above,
                "first_fourfold_material_jump": first_jump,
            }
        )
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SANA Euler trajectory trace",
        "",
        f"- Task: `{report['task']}`",
        f"- Episode: `{report['episode_index']}`",
        f"- Pair/item: `{report['pair_index']}/{report['context_index']}`",
        f"- Validation: `{report['validation']['result']}`",
        "",
        "## Endpoints",
        "",
        "| Mode | Steps | Expert RMSE | Endpoint RMS |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["endpoints"]:
        lines.append(
            f"| {row['mode']} | {row['steps']} | "
            f"{row['paired_expert_rmse']:.6f} | {row['endpoint_rms']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Native versus FP32 accumulation",
            "",
            "| Steps | Endpoint delta RMS |",
            "| ---: | ---: |",
        ]
    )
    for row in report["precision_endpoint_deltas"]:
        lines.append(
            f"| {row['steps']} | "
            f"{row['native_vs_fp32_accum_endpoint_rms']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The full JSON contains every Euler update, common-sigma state and "
            "velocity secant, timestep quantization, ULP-loss fraction, and "
            "repeatability check.",
        ]
    )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    steps = [int(value) for value in args.steps.split(",")]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if steps != sorted(set(steps)) or len(steps) < 3:
        raise ValueError("steps must contain at least three increasing unique values")
    if any(step % args.common_progress for step in steps):
        raise ValueError("every step count must be divisible by common-progress")
    if any(mode not in MODE_NAMES for mode in modes):
        raise ValueError(f"unsupported mode in {modes}")
    if "native" not in modes:
        raise ValueError("native mode is required for production equivalence")
    if args.repeat < 1:
        raise ValueError("repeat must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    device = torch.device(args.device)
    saved = torch.load(
        args.prepared_dir / f"prepared_pair_{args.pair_index:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not 0 <= args.context_index < len(saved["samples"]):
        raise IndexError("context-index is outside the prepared pair")
    metadata = saved["samples"][args.context_index]
    if int(metadata["episode_index"]) != args.expected_episode:
        raise RuntimeError("registered episode does not match the prepared pair")
    payload = to_device(saved["tensors"], device, architecture.dtype)
    frame_chunk_size = int(architecture._ar_frame_chunk_size)
    chunks = payload["input_latents"].shape[2] // frame_chunk_size
    target = payload["input_latents"].split(frame_chunk_size, dim=2)[args.chunk][
        args.context_index
    ].float().cpu()

    results: dict[tuple[str, int, int], TraceResult] = {}
    for repeat in range(args.repeat):
        ordered_steps = steps if repeat % 2 == 0 else list(reversed(steps))
        for mode in modes:
            for step_count in ordered_steps:
                result = trace_one(
                    architecture,
                    driver,
                    payload=payload,
                    pair_index=args.pair_index,
                    context_index=args.context_index,
                    chunk=args.chunk,
                    steps=step_count,
                    common_progress=args.common_progress,
                    mode=mode,
                    repeat=repeat,
                    seed=args.seed,
                )
                results[(mode, step_count, repeat)] = result
                print(
                    f"repeat={repeat} mode={mode} steps={step_count} complete",
                    flush=True,
                )

    noise_hashes = sorted({result.noise_sha256 for result in results.values()})
    repeatability = []
    for mode in modes:
        for step_count in steps:
            baseline = results[(mode, step_count, 0)].endpoint_full
            for repeat in range(1, args.repeat):
                candidate = results[(mode, step_count, repeat)].endpoint_full
                repeatability.append(
                    {
                        "mode": MODE_NAMES[mode],
                        "steps": step_count,
                        "repeat": repeat,
                        "tensor_equal": torch.equal(baseline, candidate),
                        "delta_rms": rms(baseline - candidate),
                    }
                )

    production_generator = torch.Generator(device=device).manual_seed(
        args.seed + args.pair_index * 1009 + args.chunk * 100_003
    )
    actions = payload["actions"]
    latents = payload["input_latents"]
    video_chunks = list(latents.split(frame_chunk_size, dim=2))
    tokens_per_chunk = actions.shape[1] // chunks
    action_chunks = list(actions.split(tokens_per_chunk, dim=1))
    proprio_chunks = [
        payload["proprio_seq"][:, index * tokens_per_chunk]
        for index in range(chunks)
    ]
    context, context_mask = architecture._rollout_step_context(
        payload["context"], payload["seq_lens"], payload["proprio_state"]
    )
    production_cache = build_expert_history(
        architecture,
        driver,
        video_chunks,
        action_chunks,
        proprio_chunks,
        stop_chunk=args.chunk,
        context=context,
        context_mask=context_mask,
    )
    video_proprio, _ = architecture._rollout_proprio_deltas(
        proprio_chunks[args.chunk]
    )
    production_steps = min(steps)
    architecture.video_backbone.scheduler.set_timesteps(production_steps)
    production_sigmas = [
        float(value)
        for value in architecture.video_backbone.scheduler.sigmas.tolist()
    ] + [0.0]
    production_output = architecture._denoise_video_chunk(
        driver,
        production_cache,
        frame_id=2 * args.chunk,
        like=video_chunks[args.chunk],
        v_sigmas=production_sigmas,
        v_ts=architecture.video_backbone.scheduler.timesteps,
        context=context,
        context_mask=context_mask,
        gen=production_generator,
        v_proprio=video_proprio,
    ).float().cpu()
    native_manual = results[("native", production_steps, 0)].endpoint_full
    production_validation = {
        "steps": production_steps,
        "tensor_equal": torch.equal(production_output, native_manual),
        "delta_rms": rms(production_output - native_manual),
        "max_abs_delta": float((production_output - native_manual).abs().max().item()),
    }

    endpoints, pairwise, precision = endpoint_metrics(
        results, modes=modes, steps=steps, target=target
    )
    common_rows = common_sigma_metrics(
        results,
        modes=modes,
        steps=steps,
        common_progress=args.common_progress,
    )
    schedule_alignment = []
    finest = max(steps)
    finest_result = results[(modes[0], finest, 0)]
    for coarse in steps[:-1]:
        candidate = results[(modes[0], coarse, 0)]
        sigma_delta = max(
            abs(
                float(candidate.common[index]["sigma"])
                - float(finest_result.common[index]["sigma"])
            )
            for index in range(args.common_progress + 1)
        )
        schedule_alignment.append(
            {
                "coarse_steps": coarse,
                "fine_steps": finest,
                "max_common_sigma_abs_delta": sigma_delta,
            }
        )

    reference_validation = None
    if args.reference_json is not None:
        reference = json.loads(args.reference_json.read_text())
        row = find_context(reference, args.expected_episode)
        checks = []
        for step_count in (20, 50, 100):
            if step_count not in steps:
                continue
            actual = next(
                item
                for item in endpoints
                if item["mode"] == MODE_NAMES["native"]
                and item["steps"] == step_count
            )["paired_expert_rmse"]
            expected = solution(row, step_count)["paired_expert_rmse"]
            checks.append(
                {
                    "metric": f"paired_expert_rmse_{step_count}",
                    "expected": expected,
                    "actual": actual,
                    "abs_delta": abs(actual - expected),
                }
            )
        for left, right in ((20, 50), (20, 100), (50, 100)):
            if left not in steps or right not in steps:
                continue
            actual = next(
                item
                for item in pairwise
                if item["mode"] == MODE_NAMES["native"]
                and item["left_steps"] == left
                and item["right_steps"] == right
            )["endpoint_delta_rms"]
            if right == 100:
                expected = solution(row, left)["delta_from_finest_rms"]
            else:
                expected = next(
                    item["output_delta_rms"]
                    for item in row["adjacent_deltas"]
                    if item["coarse_steps"] == left and item["fine_steps"] == right
                )
            checks.append(
                {
                    "metric": f"endpoint_delta_{left}_{right}",
                    "expected": expected,
                    "actual": actual,
                    "abs_delta": abs(actual - expected),
                }
            )
        reference_validation = {
            "path": str(args.reference_json),
            "checks": checks,
            "max_abs_delta": max(item["abs_delta"] for item in checks),
        }

    trace_summaries = []
    step_metric_rows = []
    for (mode, step_count, repeat), result in sorted(results.items()):
        if repeat != 0:
            continue
        cast_unique = len(set(result.cast_timesteps))
        trace_summaries.append(
            {
                "mode": MODE_NAMES[mode],
                "steps": step_count,
                "noise_sha256": result.noise_sha256,
                "cache_unchanged": result.cache_unchanged,
                "unique_requested_timestep_count": len(
                    set(result.requested_timesteps)
                ),
                "unique_model_timestep_count": cast_unique,
                "duplicate_model_timestep_fraction": 1.0
                - cast_unique / len(result.cast_timesteps),
                "mean_model_visible_zero_update_fraction": float(
                    sum(
                        item["model_visible_zero_update_fraction"]
                        for item in result.step_metrics
                    )
                    / len(result.step_metrics)
                ),
                "mean_ideal_below_half_ulp_fraction": float(
                    sum(
                        item["ideal_below_half_ulp_fraction"]
                        for item in result.step_metrics
                    )
                    / len(result.step_metrics)
                ),
                "mean_ideal_below_one_ulp_fraction": float(
                    sum(
                        item["ideal_below_one_ulp_fraction"]
                        for item in result.step_metrics
                    )
                    / len(result.step_metrics)
                ),
            }
        )
        for item in result.step_metrics:
            step_metric_rows.append(
                {
                    "mode": MODE_NAMES[mode],
                    "steps": step_count,
                    **item,
                }
            )

    validation_pass = (
        len(noise_hashes) == 1
        and production_validation["tensor_equal"]
        and all(item["tensor_equal"] for item in repeatability)
        and all(result.cache_unchanged for result in results.values())
        and all(
            item["max_common_sigma_abs_delta"] <= 1e-7
            for item in schedule_alignment
        )
        and (
            reference_validation is None
            or reference_validation["max_abs_delta"] <= 1e-6
        )
    )
    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "pair_index": args.pair_index,
        "context_index": args.context_index,
        "episode_index": int(metadata["episode_index"]),
        "task": args.prepared_dir.name.removesuffix("_contexts_seed20260724"),
        "start_frame": int(metadata["start_frame"]),
        "episode_path": metadata.get("episode_path"),
        "chunk": args.chunk,
        "step_counts": steps,
        "common_progress": args.common_progress,
        "modes": [MODE_NAMES[mode] for mode in modes],
        "seed": args.seed,
        "noise_sha256": noise_hashes[0] if len(noise_hashes) == 1 else noise_hashes,
        "validation": {
            "result": "PASS" if validation_pass else "FAIL",
            "production_loop": production_validation,
            "repeatability": repeatability,
            "all_cache_unchanged": all(
                result.cache_unchanged for result in results.values()
            ),
            "schedule_alignment": schedule_alignment,
            "reference": reference_validation,
        },
        "endpoints": endpoints,
        "pairwise_endpoint_deltas": pairwise,
        "precision_endpoint_deltas": precision,
        "trace_summaries": trace_summaries,
        "common_sigma_metrics": common_rows,
        "first_divergence": first_divergence(common_rows),
        "step_metrics": step_metric_rows,
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "video_euler_trace.json"
    markdown_path = args.output_dir / "video_euler_trace.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))
    if not validation_pass:
        raise RuntimeError("Euler trace validation failed; inspect the emitted report")


if __name__ == "__main__":
    main()
