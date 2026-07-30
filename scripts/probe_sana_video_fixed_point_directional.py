#!/usr/bin/env python
"""Probe trajectory-aligned SANA video-field sensitivity at a fixed point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
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
from probe_sana_video_euler_trace import (  # noqa: E402
    cache_signature,
    cosine,
    prepare_velocity,
    rms,
    tensor_sha256,
    tensor_stats,
)
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


COARSE_STEPS = 50
FINE_STEPS = 200
PROGRESS_DENOMINATOR = 10
BF16_ARM = "bf16"
FP32_ARM = "fp32_arithmetic_on_bf16_values"


@dataclass(frozen=True)
class RegisteredCase:
    case_id: str
    role: str
    task: str
    pair_index: int
    context_index: int
    episode_index: int
    start_frame: int
    default_progress: int
    matched_case_id: str
    expected_endpoint_delta_rms: float
    expected_state_master_delta_rms: float
    expected_state_model_delta_rms: float
    expected_velocity_delta_rms: float


REGISTERED_CASES = {
    case.case_id: case
    for case in (
        RegisteredCase(
            case_id="lift_pot_outlier",
            role="outlier",
            task="lift_pot",
            pair_index=3,
            context_index=1,
            episode_index=23,
            start_frame=73,
            default_progress=1,
            matched_case_id="lift_pot_median_control",
            expected_endpoint_delta_rms=0.18890876307875124,
            expected_state_master_delta_rms=0.0031339447225910157,
            expected_state_model_delta_rms=0.0036086044498117036,
            expected_velocity_delta_rms=0.26853762520419283,
        ),
        RegisteredCase(
            case_id="lift_pot_median_control",
            role="within_task_median_control",
            task="lift_pot",
            pair_index=1,
            context_index=0,
            episode_index=19,
            start_frame=24,
            default_progress=1,
            matched_case_id="lift_pot_outlier",
            expected_endpoint_delta_rms=0.016667720638959087,
            expected_state_master_delta_rms=0.0006243105070931487,
            expected_state_model_delta_rms=0.0014072662395780897,
            expected_velocity_delta_rms=0.008683489112703745,
        ),
        RegisteredCase(
            case_id="rotate_qrcode_outlier",
            role="outlier",
            task="rotate_qrcode",
            pair_index=5,
            context_index=1,
            episode_index=17,
            start_frame=29,
            default_progress=3,
            matched_case_id="rotate_qrcode_median_control",
            expected_endpoint_delta_rms=0.33050840626960415,
            expected_state_master_delta_rms=0.010280554238702546,
            expected_state_model_delta_rms=0.01045398331911301,
            expected_velocity_delta_rms=0.29367502676068374,
        ),
        RegisteredCase(
            case_id="rotate_qrcode_median_control",
            role="within_task_median_control",
            task="rotate_qrcode",
            pair_index=14,
            context_index=0,
            episode_index=45,
            start_frame=50,
            default_progress=3,
            matched_case_id="rotate_qrcode_outlier",
            expected_endpoint_delta_rms=0.020653891044850905,
            expected_state_master_delta_rms=0.0012517174082476786,
            expected_state_model_delta_rms=0.0019413897146388324,
            expected_velocity_delta_rms=0.013581064322831295,
        ),
        RegisteredCase(
            case_id="click_alarmclock_outlier",
            role="outlier",
            task="click_alarmclock",
            pair_index=21,
            context_index=0,
            episode_index=31,
            start_frame=68,
            default_progress=1,
            matched_case_id="click_alarmclock_median_control",
            expected_endpoint_delta_rms=1.0034365553969657,
            expected_state_master_delta_rms=0.005966269378500535,
            expected_state_model_delta_rms=0.006305014117401041,
            expected_velocity_delta_rms=0.4665227375936605,
        ),
        RegisteredCase(
            case_id="click_alarmclock_median_control",
            role="within_task_median_control",
            task="click_alarmclock",
            pair_index=18,
            context_index=0,
            episode_index=22,
            start_frame=9,
            default_progress=1,
            matched_case_id="click_alarmclock_outlier",
            expected_endpoint_delta_rms=0.022234087409992418,
            expected_state_master_delta_rms=0.0009904009901568463,
            expected_state_model_delta_rms=0.0017888882173287043,
            expected_velocity_delta_rms=0.01260866848337273,
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case-id", choices=sorted(REGISTERED_CASES), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--progress-numerator", type=int)
    parser.add_argument("--epsilons", default="0.005,0.01,0.02,0.05")
    parser.add_argument(
        "--alphas", default="-0.25,0.0,0.25,0.5,0.75,1.0,1.25"
    )
    parser.add_argument("--trajectory-repeats", type=int, default=2)
    parser.add_argument("--forward-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--validation-atol", type=float, default=1.0e-6)
    parser.add_argument(
        "--attempt-fp32",
        action="store_true",
        help=(
            "Try the unchanged video path with BF16 numerical inputs, weights, "
            "and cache promoted to FP32. Unsupported kernels are reported."
        ),
    )
    parser.add_argument(
        "--require-fp32",
        action="store_true",
        help="Treat an unsupported FP32 arm as a validation failure.",
    )
    return parser.parse_args()


def parse_float_list(raw: str, *, label: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or not all(torch.isfinite(torch.tensor(values))):
        raise ValueError(f"{label} must contain finite values")
    return values


def changed_fraction(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left != right).float().mean().item())


def cache_sha256(cache) -> str:
    """Hash cache metadata and every stored tensor value."""
    digest = hashlib.sha256()
    for layer_id, entries in enumerate(cache.snapshot()):
        for entry in entries:
            digest.update(
                (
                    f"{layer_id}:{int(entry['frame_id'])}:"
                    f"{int(bool(entry['is_pred']))}"
                ).encode()
            )
            for field in ("S", "z"):
                value = entry[field]
                digest.update(
                    f"{field}:{tuple(value.shape)}:{value.dtype}".encode()
                )
                digest.update(bytes.fromhex(tensor_sha256(value)))
    return digest.hexdigest()


def cuda_autocast_enabled() -> bool:
    try:
        return bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        return bool(torch.is_autocast_enabled())


def fp32_backend_state() -> dict[str, Any]:
    return {
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_autocast_enabled": cuda_autocast_enabled(),
    }


def configure_strict_fp32_backend() -> tuple[dict[str, Any], dict[str, Any]]:
    before = fp32_backend_state()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    after = fp32_backend_state()
    if (
        after["cuda_matmul_allow_tf32"]
        or after["cudnn_allow_tf32"]
        or after["float32_matmul_precision"] != "highest"
        or after["cuda_autocast_enabled"]
    ):
        raise RuntimeError(f"strict FP32 backend configuration failed: {after}")
    return before, after


def fp32_error_status(error: Exception) -> str:
    """Only known dtype/kernel rejections count as unsupported."""
    message = str(error).lower()
    unsupported_markers = (
        "not implemented for",
        "unsupported dtype",
        "only supports fp16 and bf16",
        "only supports float16 and bfloat16",
        "expected scalar type half",
        "expected scalar type bfloat16",
    )
    if isinstance(error, NotImplementedError) or any(
        marker in message for marker in unsupported_markers
    ):
        return "unsupported"
    return "failed"


def finite_secant(
    state_delta: torch.Tensor, velocity_delta: torch.Tensor
) -> dict[str, float | None]:
    state_rms = rms(state_delta)
    velocity_rms = rms(velocity_delta)
    if state_rms == 0.0:
        return {
            "state_delta_rms": 0.0,
            "velocity_delta_rms": velocity_rms,
            "gain": None,
            "signed_slope": None,
            "cosine": None,
        }
    state_double = state_delta.double().flatten()
    velocity_double = velocity_delta.double().flatten()
    signed_slope = float(
        torch.dot(velocity_double, state_double)
        .div(state_double.square().sum())
        .item()
    )
    return {
        "state_delta_rms": state_rms,
        "velocity_delta_rms": velocity_rms,
        "gain": velocity_rms / state_rms,
        "signed_slope": signed_slope,
        "cosine": cosine(velocity_delta, state_delta),
    }


def effective_timestep(
    architecture, *, requested: float, dtype: torch.dtype
) -> float:
    timestep_input = torch.tensor(requested, dtype=dtype)
    norm_factor = float(
        architecture.video_backbone._dit.timestep_norm_scale_factor
    )
    if norm_factor != 1.0:
        return float((timestep_input.float() / norm_factor).item())
    return float(timestep_input.long().float().item())


@dataclass
class RuntimeInputs:
    payload: dict[str, torch.Tensor]
    cache: Any
    context: torch.Tensor
    context_mask: torch.Tensor
    frame_proprio: torch.Tensor | None
    frame_id: int
    rope_index: torch.Tensor


def build_runtime_inputs(
    architecture,
    driver,
    *,
    payload: dict[str, torch.Tensor],
    chunk: int,
) -> RuntimeInputs:
    actions = payload["actions"]
    latents = payload["input_latents"]
    frame_chunk_size = int(architecture._ar_frame_chunk_size)
    chunks = latents.shape[2] // frame_chunk_size
    if not 0 <= chunk < chunks:
        raise ValueError(f"chunk {chunk} is outside 0..{chunks - 1}")
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
    video_proprio, _ = architecture._rollout_proprio_deltas(
        proprio_chunks[chunk]
    )
    batch, frames = video_chunks[chunk].shape[0], video_chunks[chunk].shape[2]
    frame_proprio = (
        None
        if video_proprio is None
        else video_proprio[:, None, :].expand(batch, frames, -1)
    )
    frame_id = 2 * chunk
    start_frame = (frame_id // 2) * frames
    rope_index = torch.arange(
        start_frame, start_frame + frames, device=latents.device
    )
    return RuntimeInputs(
        payload=payload,
        cache=cache,
        context=context,
        context_mask=context_mask,
        frame_proprio=frame_proprio,
        frame_id=frame_id,
        rope_index=rope_index,
    )


@dataclass
class CapturedTrajectory:
    steps: int
    repeat: int
    capture_index: int
    sigma: float
    requested_timestep: float
    effective_timestep: float
    noise_sha256: str
    capture_master: torch.Tensor
    capture_model: torch.Tensor
    capture_velocity: torch.Tensor
    endpoint_target: torch.Tensor


@torch.inference_mode()
def capture_fp32_accumulation_trajectory(
    architecture,
    driver,
    runtime: RuntimeInputs,
    *,
    steps: int,
    progress: int,
    pair_index: int,
    chunk: int,
    repeat: int,
    seed: int,
    deployment_dtype: torch.dtype,
) -> CapturedTrajectory:
    architecture.video_backbone.scheduler.set_timesteps(steps)
    sigmas = [
        float(value)
        for value in architecture.video_backbone.scheduler.sigmas.tolist()
    ] + [0.0]
    timesteps = [
        float(value)
        for value in architecture.video_backbone.scheduler.timesteps.tolist()
    ]
    capture_index = progress * steps // PROGRESS_DENOMINATOR
    if not 0 <= capture_index < steps:
        raise ValueError("the fixed point must precede the endpoint")
    generator = torch.Generator(
        device=runtime.payload["input_latents"].device
    ).manual_seed(seed + pair_index * 1009 + chunk * 100_003)
    frame_chunk_size = int(architecture._ar_frame_chunk_size)
    like = runtime.payload["input_latents"].split(frame_chunk_size, dim=2)[
        chunk
    ]
    noise = torch.randn(
        like.shape,
        generator=generator,
        device=like.device,
        dtype=deployment_dtype,
    )
    master = noise.float()
    capture_master = None
    capture_model = None
    capture_velocity = None

    for index in range(steps):
        model_state = master.to(deployment_dtype)
        velocity = prepare_velocity(
            architecture,
            driver,
            runtime.cache,
            x_model=model_state,
            t_value=timesteps[index],
            time_dtype=deployment_dtype,
            frame_id=runtime.frame_id,
            rope_index=runtime.rope_index,
            context=runtime.context,
            context_mask=runtime.context_mask,
            frame_proprio=runtime.frame_proprio,
        )
        if index == capture_index:
            capture_master = master.detach().clone()
            capture_model = model_state.detach().clone()
            capture_velocity = velocity.detach().clone()
        master = master + velocity.float() * (sigmas[index + 1] - sigmas[index])

    if (
        capture_master is None
        or capture_model is None
        or capture_velocity is None
    ):
        raise RuntimeError("fixed point was not captured")
    return CapturedTrajectory(
        steps=steps,
        repeat=repeat,
        capture_index=capture_index,
        sigma=sigmas[capture_index],
        requested_timestep=timesteps[capture_index],
        effective_timestep=effective_timestep(
            architecture,
            requested=timesteps[capture_index],
            dtype=deployment_dtype,
        ),
        noise_sha256=tensor_sha256(noise),
        capture_master=capture_master,
        capture_model=capture_model,
        capture_velocity=capture_velocity,
        endpoint_target=master.detach().float().cpu(),
    )


def trajectory_repeatability(
    baseline: CapturedTrajectory, candidate: CapturedTrajectory
) -> dict[str, Any]:
    fields = (
        "capture_master",
        "capture_model",
        "capture_velocity",
        "endpoint_target",
    )
    checks = []
    for field in fields:
        left = getattr(baseline, field)
        right = getattr(candidate, field)
        checks.append(
            {
                "field": field,
                "tensor_equal": torch.equal(left, right),
                "delta_rms": rms(left.float() - right.float()),
            }
        )
    return {
        "steps": baseline.steps,
        "repeat": candidate.repeat,
        "noise_sha256_equal": (
            baseline.noise_sha256 == candidate.noise_sha256
        ),
        "checks": checks,
        "result": "PASS"
        if all(item["tensor_equal"] for item in checks)
        and baseline.noise_sha256 == candidate.noise_sha256
        else "FAIL",
    }


def deterministic_orthogonal(direction: torch.Tensor) -> torch.Tensor:
    base = direction.detach().float().flatten()
    base_norm_squared = torch.dot(base.double(), base.double())
    if float(base_norm_squared) == 0.0:
        raise RuntimeError("trajectory direction is zero")
    shifts = (1, max(1, base.numel() // 3), max(1, base.numel() // 2))
    for shift in shifts:
        candidate = torch.roll(base, shifts=shift).double()
        candidate = candidate - (
            torch.dot(candidate, base.double()) / base_norm_squared
        ) * base.double()
        candidate_rms = candidate.square().mean().sqrt()
        if bool(torch.isfinite(candidate_rms)) and float(candidate_rms) > 1.0e-12:
            result = (candidate / candidate_rms).float().reshape_as(direction)
            result = result - (
                torch.dot(result.flatten(), direction.flatten())
                / torch.dot(direction.flatten(), direction.flatten())
            ) * direction
            return result / result.float().square().mean().sqrt()
    raise RuntimeError("could not construct a deterministic orthogonal control")


@torch.inference_mode()
def repeated_fixed_forward(
    architecture,
    driver,
    runtime: RuntimeInputs,
    *,
    model_input: torch.Tensor,
    t_value: float,
    time_dtype: torch.dtype,
    repeats: int,
    arm: str,
    label: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    baseline = None
    output_dtype = None
    checks = []
    for repeat in range(repeats):
        raw_velocity = prepare_velocity(
            architecture,
            driver,
            runtime.cache,
            x_model=model_input,
            t_value=t_value,
            time_dtype=time_dtype,
            frame_id=runtime.frame_id,
            rope_index=runtime.rope_index,
            context=runtime.context,
            context_mask=runtime.context_mask,
            frame_proprio=runtime.frame_proprio,
        ).detach()
        velocity = raw_velocity.float()
        if baseline is None:
            baseline = velocity.clone()
            output_dtype = str(raw_velocity.dtype)
            continue
        checks.append(
            {
                "repeat": repeat,
                "output_dtype_equal": str(raw_velocity.dtype) == output_dtype,
                "tensor_equal": torch.equal(baseline, velocity),
                "delta_rms": rms(baseline - velocity),
            }
        )
    if baseline is None:
        raise RuntimeError("fixed forward produced no result")
    return baseline, {
        "arm": arm,
        "label": label,
        "output_dtype": output_dtype,
        "checks": checks,
        "result": "PASS"
        if all(
            item["tensor_equal"] and item["output_dtype_equal"]
            for item in checks
        )
        else "FAIL",
    }


def promoted_timestep_input(
    architecture, *, effective: float
) -> float:
    norm_factor = float(
        architecture.video_backbone._dit.timestep_norm_scale_factor
    )
    return effective if norm_factor == 1.0 else effective * norm_factor


def scan_fixed_arm(
    architecture,
    driver,
    runtime: RuntimeInputs,
    *,
    arm: str,
    deployment_dtype: torch.dtype,
    baseline_master: torch.Tensor,
    trajectory_direction: torch.Tensor,
    orthogonal_direction: torch.Tensor,
    context_index: int,
    requested_timestep: float,
    deployed_effective_timestep: float,
    epsilons: list[float],
    alphas: list[float],
    forward_repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if arm not in {BF16_ARM, FP32_ARM}:
        raise ValueError(f"unsupported arm {arm}")
    baseline_target = baseline_master[context_index]
    target_trajectory_direction = trajectory_direction[context_index]
    raw_direction_rms = rms(target_trajectory_direction)
    if raw_direction_rms == 0.0:
        raise RuntimeError("50-step and 200-step fixed states are equal")
    trajectory_unit = trajectory_direction / raw_direction_rms
    directions = {
        "trajectory_50_minus_200": trajectory_unit,
        "deterministic_orthogonal": orthogonal_direction,
    }
    target_nominal_directions = {
        name: value[context_index].float() for name, value in directions.items()
    }
    repeatability = []

    def model_input(nominal: torch.Tensor) -> torch.Tensor:
        quantized = nominal.to(deployment_dtype)
        return quantized if arm == BF16_ARM else quantized.float()

    if arm == BF16_ARM:
        t_value = requested_timestep
        time_dtype = deployment_dtype
    else:
        t_value = promoted_timestep_input(
            architecture, effective=deployed_effective_timestep
        )
        time_dtype = torch.float32

    point_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def evaluate(
        label: str, nominal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if label in point_cache:
            return point_cache[label]
        value = model_input(nominal)
        velocity, repeat = repeated_fixed_forward(
            architecture,
            driver,
            runtime,
            model_input=value,
            t_value=t_value,
            time_dtype=time_dtype,
            repeats=forward_repeats,
            arm=arm,
            label=label,
        )
        repeatability.append(repeat)
        result = (value.detach().float(), velocity)
        point_cache[label] = result
        return result

    baseline_input_full, baseline_velocity_full = evaluate(
        "alpha_0", baseline_master
    )
    baseline_input = baseline_input_full[context_index]
    baseline_velocity = baseline_velocity_full[context_index]
    line_by_alpha = {}
    for alpha in sorted(set(alphas + [0.0, 1.0])):
        label = f"alpha_{alpha:.12g}"
        if alpha == 0.0:
            input_full = baseline_input_full
            velocity_full = baseline_velocity_full
        else:
            input_full, velocity_full = evaluate(
                label, baseline_master + alpha * trajectory_direction
            )
        input_target = input_full[context_index]
        velocity = velocity_full[context_index]
        state_delta = input_target - baseline_input
        velocity_delta = velocity - baseline_velocity
        line_by_alpha[alpha] = {
            "alpha": alpha,
            "nominal_delta_from_alpha_0_rms": abs(alpha) * raw_direction_rms,
            "actual_delta_from_alpha_0_rms": rms(state_delta),
            "changed_from_alpha_0_fraction": changed_fraction(
                input_target, baseline_input
            ),
            "input_quantization_residual_rms": rms(
                input_target
                - (
                    baseline_target
                    + alpha * target_trajectory_direction
                ).float()
            ),
            "input_sha256": tensor_sha256(input_target),
            "velocity": tensor_stats(velocity),
            "velocity_sha256": tensor_sha256(velocity),
            "secant_from_alpha_0": finite_secant(
                state_delta, velocity_delta
            ),
        }

    central = []
    actual_chords: dict[tuple[str, float], torch.Tensor] = {}
    for direction_name, direction in directions.items():
        for epsilon in epsilons:
            plus_nominal = baseline_master + epsilon * direction
            minus_nominal = baseline_master - epsilon * direction
            plus_input_full, plus_velocity_full = evaluate(
                f"central_{direction_name}_{epsilon:.12g}_plus",
                plus_nominal,
            )
            minus_input_full, minus_velocity_full = evaluate(
                f"central_{direction_name}_{epsilon:.12g}_minus",
                minus_nominal,
            )
            plus_input = plus_input_full[context_index]
            minus_input = minus_input_full[context_index]
            plus_velocity = plus_velocity_full[context_index]
            minus_velocity = minus_velocity_full[context_index]
            plus_delta = plus_input - baseline_input
            minus_delta = minus_input - baseline_input
            central_state_delta = plus_input - minus_input
            central_velocity_delta = plus_velocity - minus_velocity
            actual_chords[(direction_name, epsilon)] = central_state_delta
            central.append(
                {
                    "direction": direction_name,
                    "nominal_epsilon_rms": epsilon,
                    "plus_actual_perturbation_rms": rms(plus_delta),
                    "minus_actual_perturbation_rms": rms(minus_delta),
                    "plus_changed_fraction": changed_fraction(
                        plus_input, baseline_input
                    ),
                    "minus_changed_fraction": changed_fraction(
                        minus_input, baseline_input
                    ),
                    "central_changed_fraction": changed_fraction(
                        plus_input, minus_input
                    ),
                    "central_half_input_delta_rms": 0.5
                    * rms(central_state_delta),
                    "central_quantization_asymmetry_rms": rms(
                        plus_delta + minus_delta
                    ),
                    "actual_chord_cosine_to_nominal_trajectory": cosine(
                        central_state_delta,
                        target_nominal_directions["trajectory_50_minus_200"],
                    ),
                    "actual_chord_cosine_to_nominal_orthogonal": cosine(
                        central_state_delta,
                        target_nominal_directions["deterministic_orthogonal"],
                    ),
                    "central_secant": finite_secant(
                        central_state_delta, central_velocity_delta
                    ),
                }
            )

    for row in central:
        epsilon = row["nominal_epsilon_rms"]
        row["actual_chord_cosine_to_same_epsilon_trajectory_chord"] = cosine(
            actual_chords[(row["direction"], epsilon)],
            actual_chords[("trajectory_50_minus_200", epsilon)],
        )

    alpha_zero_input, alpha_zero_velocity = point_cache["alpha_0"]
    alpha_one_input, alpha_one_velocity = point_cache["alpha_1"]

    result = {
        "arm": arm,
        "status": "supported",
        "input_semantics": (
            "deployed_bf16_inputs"
            if arm == BF16_ARM
            else "strict_fp32_arithmetic_on_deployed_bf16_numerical_values"
        ),
        "requested_timestep_input": t_value,
        "time_dtype": str(time_dtype),
        "actual_effective_timestep": effective_timestep(
            architecture, requested=t_value, dtype=time_dtype
        ),
        "observed_output_dtypes": sorted(
            {item["output_dtype"] for item in repeatability}
        ),
        "baseline_input": tensor_stats(baseline_input),
        "baseline_velocity": tensor_stats(baseline_velocity),
        "full_endpoint_hashes": {
            "alpha_0_input": tensor_sha256(alpha_zero_input),
            "alpha_1_input": tensor_sha256(alpha_one_input),
            "alpha_0_velocity": tensor_sha256(alpha_zero_velocity),
            "alpha_1_velocity": tensor_sha256(alpha_one_velocity),
        },
        "line_scan": [line_by_alpha[alpha] for alpha in alphas],
        "central_secants": central,
        "repeatability": repeatability,
    }
    return result, repeatability


def promote_cache_without_reencoding(cache):
    snapshot = []
    for layer in cache.snapshot():
        promoted_layer = []
        for entry in layer:
            promoted = entry.copy()
            promoted["S"] = entry["S"].detach().float()
            promoted["z"] = entry["z"].detach().float()
            promoted_layer.append(promoted)
        snapshot.append(promoted_layer)
    promoted_cache = type(cache)(cache.num_layers, cache.window)
    promoted_cache.restore_snapshot(snapshot)
    return promoted_cache


def promote_runtime_without_reencoding(runtime: RuntimeInputs) -> RuntimeInputs:
    return RuntimeInputs(
        payload=runtime.payload,
        cache=promote_cache_without_reencoding(runtime.cache),
        context=runtime.context.detach().float(),
        context_mask=runtime.context_mask,
        frame_proprio=(
            None
            if runtime.frame_proprio is None
            else runtime.frame_proprio.detach().float()
        ),
        frame_id=runtime.frame_id,
        rope_index=runtime.rope_index,
    )


def non_fp32_floating_state(architecture) -> list[dict[str, str]]:
    result = []
    for name, value in architecture.video_backbone.named_parameters():
        if value.is_floating_point() and value.dtype != torch.float32:
            result.append(
                {"kind": "parameter", "name": name, "dtype": str(value.dtype)}
            )
    for name, value in architecture.video_backbone.named_buffers():
        if value.is_floating_point() and value.dtype != torch.float32:
            result.append(
                {"kind": "buffer", "name": name, "dtype": str(value.dtype)}
            )
    return result


def markdown(report: dict[str, Any]) -> str:
    validation = report["validation"]
    identity = report["trajectory_identity"]
    lines = [
        "# SANA fixed-point directional precision probe",
        "",
        f"- Case: `{report['case']['case_id']}`",
        f"- Pair/item: `{report['case']['pair_index']}/{report['case']['context_index']}`",
        f"- Common progress: `{report['fixed_point']['progress_numerator']}/10`",
        f"- Validation: `{validation['result']}`",
        f"- 50-vs-200 endpoint RMS: `{identity['endpoint_delta_rms']:.9f}`",
        "",
        "## Precision arms",
        "",
    ]
    for arm_name in (BF16_ARM, FP32_ARM):
        arm = report["arms"].get(arm_name)
        if arm is not None:
            lines.append(f"- `{arm_name}`: `{arm['status']}`")
    lines.extend(
        [
            "",
            "## Central secants",
            "",
            "| Arm | Direction | Epsilon | Actual half chord | Gain | Signed slope | Cosine |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm_name, arm in report["arms"].items():
        if arm.get("status") != "supported":
            continue
        for row in arm["central_secants"]:
            secant = row["central_secant"]
            gain = secant["gain"]
            slope = secant["signed_slope"]
            similarity = secant["cosine"]
            lines.append(
                f"| {arm_name} | {row['direction']} | "
                f"{row['nominal_epsilon_rms']:.6f} | "
                f"{row['central_half_input_delta_rms']:.6f} | "
                f"{'-' if gain is None else f'{gain:.6f}'} | "
                f"{'-' if slope is None else f'{slope:.6f}'} | "
                f"{'-' if similarity is None else f'{similarity:.6f}'} |"
            )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    case = REGISTERED_CASES[args.case_id]
    progress = (
        case.default_progress
        if args.progress_numerator is None
        else args.progress_numerator
    )
    epsilons = parse_float_list(args.epsilons, label="epsilons")
    alphas = parse_float_list(args.alphas, label="alphas")
    if not 0 <= progress < PROGRESS_DENOMINATOR:
        raise ValueError("progress-numerator must be in 0..9")
    if any(value <= 0.0 for value in epsilons):
        raise ValueError("epsilons must be positive")
    required_epsilons = {0.005, 0.01, 0.02, 0.05}
    if not required_epsilons.issubset(set(epsilons)):
        raise ValueError(
            "epsilons must include 0.005,0.01,0.02,0.05"
        )
    if 0.0 not in alphas or 1.0 not in alphas:
        raise ValueError("alphas must include 0 and 1")
    if args.trajectory_repeats < 2 or args.forward_repeats < 2:
        raise ValueError("both repeat counts must be at least two")
    if args.validation_atol <= 0.0:
        raise ValueError("validation-atol must be positive")
    if args.require_fp32 and not args.attempt_fp32:
        raise ValueError("require-fp32 requires attempt-fp32")
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
    deployment_dtype = architecture.dtype
    if deployment_dtype != torch.bfloat16:
        raise RuntimeError(
            f"registered deployment arm requires BF16, got {deployment_dtype}"
        )
    saved = torch.load(
        args.prepared_dir / f"prepared_pair_{case.pair_index:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not 0 <= case.context_index < len(saved["samples"]):
        raise IndexError("registered context is outside the prepared pair")
    metadata = saved["samples"][case.context_index]
    prepared_task = args.prepared_dir.name.removesuffix(
        "_contexts_seed20260724"
    )
    identity_checks = {
        "task": {
            "expected": case.task,
            "actual": prepared_task,
            "pass": prepared_task == case.task,
        },
        "episode_index": {
            "expected": case.episode_index,
            "actual": int(metadata["episode_index"]),
            "pass": int(metadata["episode_index"]) == case.episode_index,
        },
        "start_frame": {
            "expected": case.start_frame,
            "actual": int(metadata["start_frame"]),
            "pass": int(metadata["start_frame"]) == case.start_frame,
        },
    }
    if not all(item["pass"] for item in identity_checks.values()):
        raise RuntimeError(f"registered case identity mismatch: {identity_checks}")
    payload = to_device(saved["tensors"], device, deployment_dtype)
    runtime = build_runtime_inputs(
        architecture, driver, payload=payload, chunk=args.chunk
    )
    cache_before = cache_signature(runtime.cache)
    cache_hash_before = cache_sha256(runtime.cache)

    trajectories: dict[tuple[int, int], CapturedTrajectory] = {}
    for repeat in range(args.trajectory_repeats):
        order = (
            (COARSE_STEPS, FINE_STEPS)
            if repeat % 2 == 0
            else (FINE_STEPS, COARSE_STEPS)
        )
        for steps in order:
            trajectories[(steps, repeat)] = (
                capture_fp32_accumulation_trajectory(
                    architecture,
                    driver,
                    runtime,
                    steps=steps,
                    progress=progress,
                    pair_index=case.pair_index,
                    chunk=args.chunk,
                    repeat=repeat,
                    seed=args.seed,
                    deployment_dtype=deployment_dtype,
                )
            )
            print(
                f"repeat={repeat} steps={steps} trajectory complete",
                flush=True,
            )

    coarse = trajectories[(COARSE_STEPS, 0)]
    fine = trajectories[(FINE_STEPS, 0)]
    target = case.context_index
    trajectory_direction = coarse.capture_master - fine.capture_master
    target_trajectory_direction = trajectory_direction[target]
    trajectory_unit = target_trajectory_direction / rms(
        target_trajectory_direction
    )
    target_orthogonal_direction = deterministic_orthogonal(trajectory_unit)
    orthogonal_direction = torch.zeros_like(trajectory_direction)
    orthogonal_direction[target] = target_orthogonal_direction
    direction_validation = {
        "trajectory_unit_rms": rms(trajectory_unit),
        "orthogonal_unit_rms": rms(target_orthogonal_direction),
        "cosine": cosine(trajectory_unit, target_orthogonal_direction),
        "full_batch_cosine": cosine(
            trajectory_direction, orthogonal_direction
        ),
        "trajectory_sha256": tensor_sha256(trajectory_unit),
        "orthogonal_sha256": tensor_sha256(target_orthogonal_direction),
        "batch_semantics": (
            "The trajectory arm moves the full registered two-item batch from "
            "the 200-step state toward the 50-step state. The orthogonal arm "
            "perturbs only the registered item."
        ),
    }
    endpoint_delta = rms(
        coarse.endpoint_target[target] - fine.endpoint_target[target]
    )
    state_master_delta = rms(target_trajectory_direction)
    state_model_delta = rms(
        coarse.capture_model[target].float()
        - fine.capture_model[target].float()
    )
    velocity_delta = rms(
        coarse.capture_velocity[target].float()
        - fine.capture_velocity[target].float()
    )
    default_progress = progress == case.default_progress
    expected_checks = {
        "endpoint_delta_rms": {
            "expected": case.expected_endpoint_delta_rms,
            "actual": endpoint_delta,
            "abs_delta": abs(endpoint_delta - case.expected_endpoint_delta_rms),
            "applicable": True,
        },
        "state_master_delta_rms": {
            "expected": case.expected_state_master_delta_rms,
            "actual": state_master_delta,
            "abs_delta": abs(
                state_master_delta - case.expected_state_master_delta_rms
            ),
            "applicable": default_progress,
        },
        "state_model_delta_rms": {
            "expected": case.expected_state_model_delta_rms,
            "actual": state_model_delta,
            "abs_delta": abs(
                state_model_delta - case.expected_state_model_delta_rms
            ),
            "applicable": default_progress,
        },
        "velocity_delta_rms": {
            "expected": case.expected_velocity_delta_rms,
            "actual": velocity_delta,
            "abs_delta": abs(
                velocity_delta - case.expected_velocity_delta_rms
            ),
            "applicable": default_progress,
        },
    }
    for check in expected_checks.values():
        check["pass"] = (
            not check["applicable"]
            or check["abs_delta"] <= args.validation_atol
        )
    schedule_alignment = {
        "sigma_50": coarse.sigma,
        "sigma_200": fine.sigma,
        "sigma_abs_delta": abs(coarse.sigma - fine.sigma),
        "requested_timestep_50": coarse.requested_timestep,
        "requested_timestep_200": fine.requested_timestep,
        "requested_timestep_abs_delta": abs(
            coarse.requested_timestep - fine.requested_timestep
        ),
        "effective_timestep_50": coarse.effective_timestep,
        "effective_timestep_200": fine.effective_timestep,
        "effective_timestep_equal": (
            coarse.effective_timestep == fine.effective_timestep
        ),
    }
    trajectory_repeats = []
    for steps in (COARSE_STEPS, FINE_STEPS):
        for repeat in range(1, args.trajectory_repeats):
            trajectory_repeats.append(
                trajectory_repeatability(
                    trajectories[(steps, 0)], trajectories[(steps, repeat)]
                )
            )

    arms: dict[str, dict[str, Any]] = {}
    bf16_arm, bf16_forward_repeats = scan_fixed_arm(
        architecture,
        driver,
        runtime,
        arm=BF16_ARM,
        deployment_dtype=deployment_dtype,
        baseline_master=fine.capture_master,
        trajectory_direction=trajectory_direction,
        orthogonal_direction=orthogonal_direction,
        context_index=target,
        requested_timestep=fine.requested_timestep,
        deployed_effective_timestep=fine.effective_timestep,
        epsilons=epsilons,
        alphas=alphas,
        forward_repeats=args.forward_repeats,
    )
    alpha_zero_velocity = next(
        row
        for row in bf16_arm["line_scan"]
        if row["alpha"] == 0.0
    )
    alpha_one_velocity = next(
        row
        for row in bf16_arm["line_scan"]
        if row["alpha"] == 1.0
    )
    fine_velocity_stats = tensor_stats(fine.capture_velocity[target])
    coarse_velocity_stats = tensor_stats(coarse.capture_velocity[target])
    full_hashes = bf16_arm["full_endpoint_hashes"]
    endpoint_velocity_reproduction = {
        "alpha_0_input_tensor_equal": (
            alpha_zero_velocity["input_sha256"]
            == tensor_sha256(fine.capture_model[target].float())
        ),
        "alpha_1_input_tensor_equal": (
            alpha_one_velocity["input_sha256"]
            == tensor_sha256(coarse.capture_model[target].float())
        ),
        "alpha_0_tensor_equal": (
            alpha_zero_velocity["velocity_sha256"]
            == tensor_sha256(fine.capture_velocity[target].float())
        ),
        "alpha_1_tensor_equal": (
            alpha_one_velocity["velocity_sha256"]
            == tensor_sha256(coarse.capture_velocity[target].float())
        ),
        "alpha_0_velocity_stats_match": (
            alpha_zero_velocity["velocity"] == fine_velocity_stats
        ),
        "alpha_1_velocity_stats_match": (
            alpha_one_velocity["velocity"] == coarse_velocity_stats
        ),
        "alpha_0_full_input_tensor_equal": (
            full_hashes["alpha_0_input"]
            == tensor_sha256(fine.capture_model.float())
        ),
        "alpha_1_full_input_tensor_equal": (
            full_hashes["alpha_1_input"]
            == tensor_sha256(coarse.capture_model.float())
        ),
        "alpha_0_full_velocity_tensor_equal": (
            full_hashes["alpha_0_velocity"]
            == tensor_sha256(fine.capture_velocity.float())
        ),
        "alpha_1_full_velocity_tensor_equal": (
            full_hashes["alpha_1_velocity"]
            == tensor_sha256(coarse.capture_velocity.float())
        ),
        "note": (
            "Alpha 0 and alpha 1 use the original full 200-step and 50-step "
            "two-item states, respectively."
        ),
    }
    arms[BF16_ARM] = bf16_arm
    cache_after_bf16 = cache_signature(runtime.cache)
    cache_hash_after_bf16 = cache_sha256(runtime.cache)

    fp32_validation: dict[str, Any] = {
        "requested": args.attempt_fp32,
        "status": "not_requested",
    }
    fp32_forward_repeats: list[dict[str, Any]] = []
    arms[FP32_ARM] = {
        "arm": FP32_ARM,
        "status": "not_requested",
    }
    if args.attempt_fp32:
        stage = "promote_cache_without_reencoding"
        try:
            fp32_runtime = promote_runtime_without_reencoding(runtime)
            fp32_cache_before = cache_signature(fp32_runtime.cache)
            fp32_cache_hash_before = cache_sha256(fp32_runtime.cache)
            stage = "configure_strict_fp32_backend"
            backend_before, backend_strict = configure_strict_fp32_backend()
            stage = "promote_model_in_place"
            architecture.float()
            architecture.eval()
            remaining = non_fp32_floating_state(architecture)
            if remaining:
                raise RuntimeError(
                    "video backbone retains non-FP32 floating state: "
                    f"{remaining[:8]}"
                )
            stage = "fixed_forward_unchanged_kernel_path"
            fp32_arm, fp32_forward_repeats = scan_fixed_arm(
                architecture,
                driver,
                fp32_runtime,
                arm=FP32_ARM,
                deployment_dtype=deployment_dtype,
                baseline_master=fine.capture_master,
                trajectory_direction=trajectory_direction,
                orthogonal_direction=orthogonal_direction,
                context_index=target,
                requested_timestep=fine.requested_timestep,
                deployed_effective_timestep=fine.effective_timestep,
                epsilons=epsilons,
                alphas=alphas,
                forward_repeats=args.forward_repeats,
            )
            if fp32_arm["observed_output_dtypes"] != [str(torch.float32)]:
                raise RuntimeError(
                    "unchanged forward path did not produce FP32 output: "
                    f"{fp32_arm['observed_output_dtypes']}"
                )
            fp32_cache_after = cache_signature(fp32_runtime.cache)
            fp32_cache_hash_after = cache_sha256(fp32_runtime.cache)
            fp32_arm["cache_unchanged"] = (
                fp32_cache_before == fp32_cache_after
                and fp32_cache_hash_before == fp32_cache_hash_after
            )
            fp32_arm["promotion"] = {
                "weights": "checkpoint BF16 numerical values promoted to FP32",
                "inputs": "each deployed BF16 input promoted to FP32",
                "cache": "stored BF16 S/z promoted to FP32 without re-encoding",
                "context": "stored BF16 context/proprio promoted to FP32",
                "arithmetic": "strict FP32 with TF32 disabled",
                "kernel_policy": "unchanged production path; no kernel substitution",
            }
            arms[FP32_ARM] = fp32_arm
            fp32_validation = {
                "requested": True,
                "status": "supported",
                "cache_unchanged": fp32_arm["cache_unchanged"],
                "non_fp32_floating_state": remaining,
                "backend_before": backend_before,
                "backend_strict": backend_strict,
            }
        except Exception as error:  # noqa: BLE001
            error_status = fp32_error_status(error)
            arms[FP32_ARM] = {
                "arm": FP32_ARM,
                "status": error_status,
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "kernel_policy": (
                    "No alternate kernel was selected after the production "
                    "path rejected FP32."
                ),
            }
            fp32_validation = {
                "requested": True,
                "status": error_status,
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
            }

    all_forward_repeats = bf16_forward_repeats + fp32_forward_repeats
    bf16_required_pass = (
        all(item["pass"] for item in identity_checks.values())
        and all(item["pass"] for item in expected_checks.values())
        and schedule_alignment["sigma_abs_delta"] <= 1.0e-7
        and schedule_alignment["requested_timestep_abs_delta"] <= 1.0e-4
        and schedule_alignment["effective_timestep_equal"]
        and coarse.noise_sha256 == fine.noise_sha256
        and all(item["result"] == "PASS" for item in trajectory_repeats)
        and all(
            item["result"] == "PASS" for item in bf16_forward_repeats
        )
        and bf16_arm["observed_output_dtypes"] == [str(deployment_dtype)]
        and bf16_arm["actual_effective_timestep"]
        == fine.effective_timestep
        and cache_before == cache_after_bf16
        and cache_hash_before == cache_hash_after_bf16
        and endpoint_velocity_reproduction["alpha_0_input_tensor_equal"]
        and endpoint_velocity_reproduction["alpha_1_input_tensor_equal"]
        and endpoint_velocity_reproduction["alpha_0_tensor_equal"]
        and endpoint_velocity_reproduction["alpha_1_tensor_equal"]
        and endpoint_velocity_reproduction[
            "alpha_0_full_input_tensor_equal"
        ]
        and endpoint_velocity_reproduction[
            "alpha_1_full_input_tensor_equal"
        ]
        and endpoint_velocity_reproduction[
            "alpha_0_full_velocity_tensor_equal"
        ]
        and endpoint_velocity_reproduction[
            "alpha_1_full_velocity_tensor_equal"
        ]
        and abs(direction_validation["trajectory_unit_rms"] - 1.0)
        <= 1.0e-6
        and abs(direction_validation["orthogonal_unit_rms"] - 1.0)
        <= 1.0e-6
        and (
            direction_validation["cosine"] is not None
            and abs(direction_validation["cosine"]) <= 1.0e-6
        )
        and (
            direction_validation["full_batch_cosine"] is not None
            and abs(direction_validation["full_batch_cosine"]) <= 1.0e-6
        )
    )
    if fp32_validation["status"] == "supported":
        strict_backend = fp32_validation["backend_strict"]
        fp32_pass = (
            fp32_validation["status"] == "supported"
            and bool(fp32_validation["cache_unchanged"])
            and not strict_backend["cuda_matmul_allow_tf32"]
            and not strict_backend["cudnn_allow_tf32"]
            and strict_backend["float32_matmul_precision"] == "highest"
            and not strict_backend["cuda_autocast_enabled"]
            and arms[FP32_ARM]["actual_effective_timestep"]
            == fine.effective_timestep
            and all(
                item["result"] == "PASS"
                for item in fp32_forward_repeats
            )
        )
    else:
        fp32_pass = not args.require_fp32
    validation_pass = bf16_required_pass and fp32_pass
    report = {
        "case": asdict(case),
        "matched_case": asdict(REGISTERED_CASES[case.matched_case_id]),
        "checkpoint": str(args.run_dir / args.checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "fixed_point": {
            "coarse_steps": COARSE_STEPS,
            "fine_steps": FINE_STEPS,
            "progress_numerator": progress,
            "progress_denominator": PROGRESS_DENOMINATOR,
            "used_registered_default_progress": default_progress,
            "sigma": fine.sigma,
            "requested_timestep": fine.requested_timestep,
            "deployed_effective_timestep": fine.effective_timestep,
        },
        "seed": args.seed,
        "epsilons": epsilons,
        "alphas": alphas,
        "trajectory_identity": {
            "endpoint_delta_rms": endpoint_delta,
            "state_master_delta_rms": state_master_delta,
            "state_model_delta_rms": state_model_delta,
            "velocity_delta_rms": velocity_delta,
            "deployed_common_point_secant": finite_secant(
                coarse.capture_model[target].float()
                - fine.capture_model[target].float(),
                coarse.capture_velocity[target].float()
                - fine.capture_velocity[target].float(),
            ),
            "noise_sha256_50": coarse.noise_sha256,
            "noise_sha256_200": fine.noise_sha256,
        },
        "direction_validation": direction_validation,
        "schedule_alignment": schedule_alignment,
        "arms": arms,
        "validation": {
            "result": "PASS" if validation_pass else "FAIL",
            "bf16_required_result": (
                "PASS" if bf16_required_pass else "FAIL"
            ),
            "fp32_requirement_result": "PASS" if fp32_pass else "FAIL",
            "case_identity": identity_checks,
            "registered_metric_reproduction": expected_checks,
            "trajectory_repeatability": trajectory_repeats,
            "fixed_forward_repeatability": all_forward_repeats,
            "bf16_cache_unchanged": (
                cache_before == cache_after_bf16
                and cache_hash_before == cache_hash_after_bf16
            ),
            "bf16_cache_sha256_before": cache_hash_before,
            "bf16_cache_sha256_after": cache_hash_after_bf16,
            "endpoint_velocity_reproduction": endpoint_velocity_reproduction,
            "fp32": fp32_validation,
        },
        "limitations": [
            (
                "The trajectory direction is defined from FP32 master states "
                "whose velocities were produced by the deployed BF16 forward."
            ),
            (
                "Both precision arms use the same BF16-quantized point values; "
                "the strict-FP32 arm isolates forward arithmetic, not input "
                "quantization."
            ),
            (
                "The strict-FP32 arm promotes checkpoint weights, context, "
                "proprioception, and stored history-cache values from their "
                "deployed BF16 numerical values. It does not recover FP32-source "
                "weights or re-encode history in FP32."
            ),
            (
                "The deterministic orthogonal direction is a numerical control, "
                "not an on-manifold video direction. Actual post-quantization "
                "chord cosines are reported for every epsilon."
            ),
            (
                "Finite secants at four scales are local controlled effects, not "
                "a global Lipschitz bound or a closed-loop performance measure."
            ),
        ],
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "video_fixed_point_directional.json"
    markdown_path = args.output_dir / "video_fixed_point_directional.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))
    if not validation_pass:
        raise RuntimeError(
            "fixed-point directional validation failed; inspect emitted report"
        )


if __name__ == "__main__":
    main()
