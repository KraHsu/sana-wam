#!/usr/bin/env python
"""Separate video field bias from local error amplification along Euler rollout."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from probe_sana_action_denoising import to_device  # noqa: E402
from probe_sana_ar_decomposition import build_expert_history  # noqa: E402
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--perturbation-rms", default="0.025,0.05,0.1")
    parser.add_argument("--transverse-directions", type=int, default=2)
    parser.add_argument("--transverse-rms", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


@dataclass
class RMS:
    square_sum: float = 0.0
    count: int = 0

    def add(self, value: torch.Tensor) -> None:
        value = value.double()
        self.square_sum += float(value.square().sum().item())
        self.count += value.numel()

    def result(self) -> float | None:
        if self.count == 0:
            return None
        return (self.square_sum / self.count) ** 0.5


@dataclass
class Ratio:
    numerator_square_sum: float = 0.0
    denominator_square_sum: float = 0.0

    def add(self, numerator: torch.Tensor, denominator: torch.Tensor) -> None:
        self.numerator_square_sum += float(
            numerator.double().square().sum().item()
        )
        self.denominator_square_sum += float(
            denominator.double().square().sum().item()
        )

    def result(self) -> float | None:
        if self.denominator_square_sum == 0.0:
            return None
        return (self.numerator_square_sum / self.denominator_square_sum) ** 0.5


@dataclass
class Projection:
    dot_sum: float = 0.0
    denominator_square_sum: float = 0.0

    def add(self, value: torch.Tensor, direction: torch.Tensor) -> None:
        self.dot_sum += float((value.double() * direction.double()).sum().item())
        self.denominator_square_sum += float(
            direction.double().square().sum().item()
        )

    def result(self) -> float | None:
        if self.denominator_square_sum == 0.0:
            return None
        return self.dot_sum / self.denominator_square_sum


def _unit_rms(value: torch.Tensor) -> torch.Tensor | None:
    rms = value.float().square().mean().sqrt()
    if not bool(torch.isfinite(rms)) or float(rms) <= 1.0e-6:
        return None
    return value.float() / rms


def _orthonormal_basis(values: list[torch.Tensor]) -> list[torch.Tensor]:
    basis = []
    for value in values:
        vector = value.float().flatten()
        for previous in basis:
            vector = vector - torch.dot(vector, previous) * previous
        norm = vector.norm()
        if bool(torch.isfinite(norm)) and float(norm) > 1.0e-12:
            basis.append(vector / norm)
    return basis


def _transverse_direction(
    like: torch.Tensor,
    *,
    generator: torch.Generator,
    excluded: list[torch.Tensor],
) -> torch.Tensor:
    basis = _orthonormal_basis(excluded)
    random = torch.randint(
        0,
        2,
        like.shape,
        generator=generator,
        device=like.device,
        dtype=torch.int8,
    ).float()
    vector = (random * 2.0 - 1.0).flatten()
    for previous in basis:
        vector = vector - torch.dot(vector, previous) * previous
    norm = vector.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 1.0e-12:
        raise RuntimeError("transverse projection produced a zero direction")
    vector = vector / norm * (vector.numel() ** 0.5)
    return vector.reshape_as(like)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Video trajectory error/Jacobian probe",
        "",
        "The exact recurrence is decomposed around the matched expert path:",
        "",
        "```text",
        "e_next = e + delta_sigma * field_bias + delta_sigma * off_path_response",
        "```",
        "",
        "`secant gain` removes field bias and measures amplification of the "
        "existing rollout error. The ideal endpoint-correcting response is "
        "`e / sigma`, whose Euler gain is `next_sigma / sigma`.",
        "",
        "| step | sigma | error | field bias | target error | bias slope | "
        "response slope | required response slope | secant gain | ideal gain | next error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["metrics"]:
        bias_slope = row["field_bias_projection_on_error"]
        response_slope = row["response_projection_on_error"]
        required_slope = row["required_response_projection_on_error"]
        secant_gain = row["secant_euler_gain"]
        lines.append(
            f"| {row['step']} | {row['sigma']:.6f} | "
            f"{row['state_error_rms']:.6f} | "
            f"{row['on_path_field_bias_rms']:.6f} | "
            f"{row['model_state_target_error_rms']:.6f} | "
            f"{'-' if bias_slope is None else f'{bias_slope:.6f}'} | "
            f"{'-' if response_slope is None else f'{response_slope:.6f}'} | "
            f"{'-' if required_slope is None else f'{required_slope:.6f}'} | "
            f"{'-' if secant_gain is None else f'{secant_gain:.6f}'} | "
            f"{row['ideal_euler_gain']:.6f} | "
            f"{row['next_error_rms']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Centered finite-difference Euler gains",
            "",
            "Actual-error directions are evaluated at every requested perturbation "
            "RMS. Transverse directions are orthogonal to both rollout error and "
            "the expert conditional velocity.",
            "",
            "| step | sigma | actual-direction gains | transverse gain | "
            "local response slope | linearization residual |",
            "| ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    primary = str(report["transverse_rms"])
    for row in report["metrics"]:
        gains = ", ".join(
            f"h={key}: {value:.6f}"
            for key, value in row["local_actual_euler_gain"].items()
            if value is not None
        )
        transverse = row["local_transverse_euler_gain"]
        slope = row["local_actual_response_projection"].get(primary)
        residual = row["local_actual_linearization_residual_rms"].get(primary)
        lines.append(
            f"| {row['step']} | {row['sigma']:.6f} | {gains or '-'} | "
            f"{'-' if transverse is None else f'{transverse:.6f}'} | "
            f"{'-' if slope is None else f'{slope:.6f}'} | "
            f"{'-' if residual is None else f'{residual:.6f}'} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    perturbation_rms = [float(value) for value in args.perturbation_rms.split(",")]
    if args.steps < 2:
        raise ValueError("steps must be at least 2")
    if args.chunk <= 0:
        raise ValueError("chunk must be a continuation chunk")
    if not perturbation_rms or any(value <= 0 for value in perturbation_rms):
        raise ValueError("perturbation RMS values must be positive")
    if args.transverse_directions <= 0 or args.transverse_rms <= 0:
        raise ValueError("transverse settings must be positive")
    if args.transverse_rms not in perturbation_rms:
        raise ValueError("transverse RMS must be included in perturbation RMS values")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    device = torch.device(args.device)
    architecture.video_backbone.scheduler.set_timesteps(args.steps)
    sigmas = [
        float(value)
        for value in architecture.video_backbone.scheduler.sigmas.tolist()
    ] + [0.0]
    timesteps = architecture.video_backbone.scheduler.timesteps

    accumulators = []
    for _ in range(args.steps):
        accumulators.append(
            {
                "state_error": RMS(),
                "field_bias": RMS(),
                "response": RMS(),
                "velocity_error": RMS(),
                "model_state_target_error": RMS(),
                "next_error": RMS(),
                "next_error_recurrence": RMS(),
                "recurrence_residual": RMS(),
                "next_without_bias": RMS(),
                "next_without_response": RMS(),
                "next_with_ideal_response": RMS(),
                "next_no_bias_ideal_response": RMS(),
                "field_bias_projection": Projection(),
                "response_projection": Projection(),
                "total_velocity_error_projection": Projection(),
                "secant_gain": Ratio(),
                "local_actual_gain": {
                    value: Ratio() for value in perturbation_rms
                },
                "local_actual_projection": {
                    value: Projection() for value in perturbation_rms
                },
                "local_linearization_residual": {
                    value: RMS() for value in perturbation_rms
                },
                "local_transverse_gain": Ratio(),
                "local_transverse_projection": Projection(),
            }
        )

    for pair_index in range(args.pairs):
        saved = torch.load(
            args.prepared_dir / f"prepared_pair_{pair_index:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        payload = to_device(saved["tensors"], device, architecture.dtype)
        actions = payload["actions"]
        latents = payload["input_latents"]
        frame_chunk_size = int(architecture._ar_frame_chunk_size)
        chunks = latents.shape[2] // frame_chunk_size
        if args.chunk >= chunks:
            raise ValueError(f"chunk {args.chunk} is outside 0..{chunks - 1}")
        tokens_per_chunk = actions.shape[1] // chunks
        video_chunks = list(latents.split(frame_chunk_size, dim=2))
        action_chunks = list(actions.split(tokens_per_chunk, dim=1))
        proprio_chunks = [
            payload["proprio_seq"][:, chunk * tokens_per_chunk]
            for chunk in range(chunks)
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
            stop_chunk=args.chunk,
            context=context,
            context_mask=context_mask,
        )
        clean = video_chunks[args.chunk]
        video_proprio, _ = architecture._rollout_proprio_deltas(
            proprio_chunks[args.chunk]
        )
        batch, n_frames = clean.shape[0], clean.shape[2]
        start_frame = args.chunk * n_frames
        rope_index = torch.arange(
            start_frame, start_frame + n_frames, device=device
        )
        frame_proprio = (
            None
            if video_proprio is None
            else video_proprio[:, None, :].expand(batch, n_frames, -1)
        )
        extra = (
            {} if frame_proprio is None else {"frame_proprio_emb": frame_proprio}
        )

        def forward_video(value: torch.Tensor, timestep: float) -> torch.Tensor:
            frame_timesteps = torch.full(
                (batch, n_frames),
                timestep,
                device=device,
                dtype=architecture.dtype,
            )
            video_state = architecture.video_backbone.prepare(
                latents=value,
                timestep=torch.full(
                    (batch,),
                    timestep,
                    device=device,
                    dtype=architecture.dtype,
                ),
                frame_timesteps=frame_timesteps,
                rope_frame_index=rope_index,
                context=context,
                context_mask=context_mask,
                **extra,
            )
            driver.run_ar_chunk_through_backbone(
                architecture.video_backbone,
                video_state,
                cache,
                frame_id=2 * args.chunk,
                store_clean=False,
            )
            return architecture.video_backbone.finalize(video_state)

        generator = torch.Generator(device=device).manual_seed(
            args.seed + pair_index * 1009 + args.chunk * 100_003
        )
        initial_noise = torch.randn(
            clean.shape,
            generator=generator,
            device=device,
            dtype=architecture.dtype,
        )
        expert_velocity = initial_noise.float() - clean.float()
        state = initial_noise.clone()
        transverse_generator = torch.Generator(device=device).manual_seed(
            args.seed + pair_index * 10_007 + args.chunk * 1_000_003 + 17
        )

        for index in range(args.steps):
            sigma = sigmas[index]
            next_sigma = sigmas[index + 1]
            delta_sigma = next_sigma - sigma
            timestep = float(timesteps[index])
            reference = (clean.float() + sigma * expert_velocity).to(
                architecture.dtype
            )
            next_reference = (clean.float() + next_sigma * expert_velocity).to(
                architecture.dtype
            )
            state_velocity_raw = forward_video(state, timestep)
            reference_velocity_raw = forward_video(reference, timestep)
            state_velocity = state_velocity_raw.float()
            reference_velocity = reference_velocity_raw.float()
            error = state.float() - reference.float()
            field_bias = reference_velocity - expert_velocity
            response = state_velocity - reference_velocity
            ideal_response = error / sigma
            next_state = state + delta_sigma * state_velocity_raw
            next_error = next_state.float() - next_reference.float()
            recurrence = error + delta_sigma * (field_bias + response)
            sums = accumulators[index]
            sums["state_error"].add(error)
            sums["field_bias"].add(field_bias)
            sums["response"].add(response)
            sums["velocity_error"].add(field_bias + response)
            sums["model_state_target_error"].add(
                field_bias + response - ideal_response
            )
            sums["next_error"].add(next_error)
            sums["next_error_recurrence"].add(recurrence)
            sums["recurrence_residual"].add(next_error - recurrence)
            sums["next_without_bias"].add(error + delta_sigma * response)
            sums["next_without_response"].add(
                error + delta_sigma * field_bias
            )
            sums["next_with_ideal_response"].add(
                error + delta_sigma * (field_bias + ideal_response)
            )
            sums["next_no_bias_ideal_response"].add(
                error + delta_sigma * ideal_response
            )

            error_direction = _unit_rms(error)
            if error_direction is not None:
                sums["field_bias_projection"].add(field_bias, error)
                sums["response_projection"].add(response, error)
                sums["total_velocity_error_projection"].add(
                    field_bias + response, error
                )
                sums["secant_gain"].add(
                    error + delta_sigma * response, error
                )
                error_rms = error.float().square().mean().sqrt()
                for perturbation in perturbation_rms:
                    plus = (reference.float() + perturbation * error_direction).to(
                        architecture.dtype
                    )
                    minus = (
                        reference.float() - perturbation * error_direction
                    ).to(architecture.dtype)
                    centered_response = (
                        forward_video(plus, timestep).float()
                        - forward_video(minus, timestep).float()
                    ) / (2.0 * perturbation)
                    local_next = (
                        error_direction + delta_sigma * centered_response
                    )
                    sums["local_actual_gain"][perturbation].add(
                        local_next, error_direction
                    )
                    sums["local_actual_projection"][perturbation].add(
                        centered_response, error_direction
                    )
                    sums["local_linearization_residual"][perturbation].add(
                        response - centered_response * error_rms
                    )

            excluded = [expert_velocity]
            if error_direction is not None:
                excluded.append(error)
            for _ in range(args.transverse_directions):
                direction = _transverse_direction(
                    reference,
                    generator=transverse_generator,
                    excluded=excluded,
                )
                plus = (
                    reference.float() + args.transverse_rms * direction
                ).to(architecture.dtype)
                minus = (
                    reference.float() - args.transverse_rms * direction
                ).to(architecture.dtype)
                centered_response = (
                    forward_video(plus, timestep).float()
                    - forward_video(minus, timestep).float()
                ) / (2.0 * args.transverse_rms)
                local_next = direction + delta_sigma * centered_response
                sums["local_transverse_gain"].add(local_next, direction)
                sums["local_transverse_projection"].add(
                    centered_response, direction
                )
            state = next_state

    metrics = []
    for index, sums in enumerate(accumulators):
        sigma = sigmas[index]
        next_sigma = sigmas[index + 1]
        metrics.append(
            {
                "step": index,
                "sigma": sigma,
                "next_sigma": next_sigma,
                "timestep": float(timesteps[index]),
                "state_error_rms": sums["state_error"].result(),
                "on_path_field_bias_rms": sums["field_bias"].result(),
                "off_path_response_rms": sums["response"].result(),
                "model_state_velocity_error_rms": sums[
                    "velocity_error"
                ].result(),
                "model_state_target_error_rms": sums[
                    "model_state_target_error"
                ].result(),
                "field_bias_projection_on_error": sums[
                    "field_bias_projection"
                ].result(),
                "response_projection_on_error": sums[
                    "response_projection"
                ].result(),
                "required_response_projection_on_error": (
                    None
                    if sums["field_bias_projection"].result() is None
                    else 1.0 / sigma
                    - sums["field_bias_projection"].result()
                ),
                "total_velocity_error_projection_on_error": sums[
                    "total_velocity_error_projection"
                ].result(),
                "ideal_response_slope": 1.0 / sigma,
                "secant_euler_gain": sums["secant_gain"].result(),
                "ideal_euler_gain": next_sigma / sigma,
                "next_error_rms": sums["next_error"].result(),
                "recurrence_predicted_next_error_rms": sums[
                    "next_error_recurrence"
                ].result(),
                "recurrence_residual_rms": sums[
                    "recurrence_residual"
                ].result(),
                "next_error_without_field_bias_rms": sums[
                    "next_without_bias"
                ].result(),
                "next_error_without_off_path_response_rms": sums[
                    "next_without_response"
                ].result(),
                "next_error_with_radial_response_and_current_bias_rms": sums[
                    "next_with_ideal_response"
                ].result(),
                "ideal_endpoint_next_error_rms": sums[
                    "next_no_bias_ideal_response"
                ].result(),
                "local_actual_euler_gain": {
                    str(value): sums["local_actual_gain"][value].result()
                    for value in perturbation_rms
                },
                "local_actual_response_projection": {
                    str(value): sums["local_actual_projection"][value].result()
                    for value in perturbation_rms
                },
                "local_actual_linearization_residual_rms": {
                    str(value): sums["local_linearization_residual"][
                        value
                    ].result()
                    for value in perturbation_rms
                },
                "local_transverse_euler_gain": sums[
                    "local_transverse_gain"
                ].result(),
                "local_transverse_response_projection": sums[
                    "local_transverse_projection"
                ].result(),
            }
        )

    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "pair_count": args.pairs,
        "chunk": args.chunk,
        "steps": args.steps,
        "seed": args.seed,
        "perturbation_rms": perturbation_rms,
        "transverse_directions": args.transverse_directions,
        "transverse_rms": args.transverse_rms,
        "metrics": metrics,
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "video_error_jacobian.json"
    markdown_path = args.output_dir / "video_error_jacobian.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
