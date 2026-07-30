"""Local reverse-Euler expansion measurements for clean interpolation states."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class LocalExpansionResult:
    loss: Tensor
    rate: Tensor
    expansion: Tensor
    chord_rms: Tensor
    next_chord_rms: Tensor


@dataclass(frozen=True)
class Phase6ExpansionSpec:
    global_step: int
    action_sigma: float
    epsilon_rms: float
    direction: str
    expansion_noise_seed: int
    expansion_direction_seed: int


_PHASE6_ROW_KEYS = {
    "action_sigma",
    "cycle",
    "domain_seeds",
    "global_step",
    "identity",
    "identity_sha256",
    "plan_sha256",
    "position_in_cycle",
}
_PHASE6_IDENTITY_KEYS = {
    "dataset_index",
    "episode_index",
    "episode_path",
    "prompt",
    "source_dataset",
    "source_kind",
    "source_variant",
    "start_frame",
    "task_name",
}
_PHASE6_SEED_DOMAINS = {
    "video-noise",
    "action-noise",
    "expansion-noise",
    "expansion-direction",
    "reference-query",
    "prompt-choice",
}
_PHASE6_ACTION_SIGMAS = (1.0, 0.9, 0.5)
_PHASE6_EPSILON_RMS = (0.005, 0.01, 0.02)
_PHASE6_DIRECTIONS = (
    "flow_target_tangent",
    "deterministic_rademacher_orthogonal",
)


def _require_plain_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def phase6_expansion_spec(
    phase6_plan_rows: Any, *, batch_size: int
) -> Phase6ExpansionSpec:
    """Validate one frozen plan row and derive the registered E schedule."""
    if batch_size != 1:
        raise ValueError("Phase-6 local expansion requires batch size exactly 1")
    if type(phase6_plan_rows) is not tuple or len(phase6_plan_rows) != 1:
        raise ValueError("phase6_plan_rows must be a one-element tuple[dict]")
    row = phase6_plan_rows[0]
    if type(row) is not dict or set(row) != _PHASE6_ROW_KEYS:
        raise ValueError("Phase-6 plan row keys do not match the frozen schema")
    _require_sha256(row["plan_sha256"], "plan_sha256")
    _require_sha256(row["identity_sha256"], "identity_sha256")

    global_step = _require_plain_int(row["global_step"], "global_step", minimum=1)
    if global_step > 504:
        raise ValueError("global_step must be within the frozen 504-row plan")
    cycle = _require_plain_int(row["cycle"], "cycle")
    position = _require_plain_int(row["position_in_cycle"], "position_in_cycle")
    if cycle >= 12 or position >= 42:
        raise ValueError("cycle/position lie outside the frozen 12 x 42 plan")
    if global_step != cycle * 42 + position + 1:
        raise ValueError("global_step, cycle, and position_in_cycle are inconsistent")

    action_sigma = row["action_sigma"]
    if type(action_sigma) is not float or action_sigma not in _PHASE6_ACTION_SIGMAS:
        raise ValueError("action_sigma is not one of the frozen Phase-6 values")

    identity = row["identity"]
    if not isinstance(identity, Mapping) or set(identity) != _PHASE6_IDENTITY_KEYS:
        raise ValueError("Phase-6 plan identity keys do not match the frozen schema")
    for key in ("task_name", "episode_path", "prompt"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ValueError(f"identity.{key} must be a non-empty string")
    for key in ("dataset_index", "episode_index", "start_frame"):
        _require_plain_int(identity[key], f"identity.{key}")
    if (
        identity["source_dataset"],
        identity["source_variant"],
        identity["source_kind"],
    ) != ("RoboTwin", "clean_50", "ordinary_expert"):
        raise ValueError("Phase-6 expansion accepts ordinary clean_50 rows only")

    seeds = row["domain_seeds"]
    if not isinstance(seeds, Mapping) or set(seeds) != _PHASE6_SEED_DOMAINS:
        raise ValueError("Phase-6 row seed domains do not match the frozen schema")
    for domain, seed in seeds.items():
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError(f"seed {domain!r} must be an unsigned 64-bit integer")

    schedule_index = global_step - 1
    return Phase6ExpansionSpec(
        global_step=global_step,
        action_sigma=action_sigma,
        epsilon_rms=_PHASE6_EPSILON_RMS[schedule_index % 3],
        direction=_PHASE6_DIRECTIONS[schedule_index % 2],
        expansion_noise_seed=seeds["expansion-noise"],
        expansion_direction_seed=seeds["expansion-direction"],
    )


def _broadcast_video_mask(mask: Tensor, value: Tensor) -> Tensor:
    if value.ndim != 5:
        raise ValueError(f"expected video tensor [B,C,T,H,W], got {tuple(value.shape)}")
    if mask.shape != (value.shape[0], value.shape[2]):
        raise ValueError(
            f"mask must be [B={value.shape[0]},T={value.shape[2]}], got {tuple(mask.shape)}"
        )
    return mask.detach().to(
        device=value.device, dtype=torch.float32
    )[:, None, :, None, None]


def current_video_chunk_mask(
    video_is_pad: Tensor | None,
    *,
    batch_size: int,
    num_frames: int,
    frame_chunk_size: int,
    bootstrap_clean_prefix: bool,
    device: torch.device,
) -> Tensor:
    """Select the last non-padded chunk and exclude bootstrap/padded frames."""
    if batch_size != 1:
        raise ValueError("Phase-6 current-chunk selection requires batch size 1")
    if frame_chunk_size <= 0 or num_frames % frame_chunk_size != 0:
        raise ValueError("frame_chunk_size must divide num_frames")
    if video_is_pad is None:
        padded = torch.zeros(
            batch_size, num_frames, device=device, dtype=torch.bool
        )
    else:
        if video_is_pad.dtype != torch.bool:
            raise TypeError("video_is_pad must be boolean for Phase-6 expansion")
        if tuple(video_is_pad.shape) != (batch_size, num_frames):
            raise ValueError(
                "video_is_pad must exactly match the latent frame shape "
                f"({batch_size}, {num_frames})"
            )
        padded = video_is_pad.detach().to(device=device)
    valid = ~padded
    if bootstrap_clean_prefix:
        valid = valid.clone()
        valid[:, 0] = False
    valid_by_chunk = valid.reshape(batch_size, -1, frame_chunk_size).any(dim=2)
    selected = torch.zeros_like(valid)
    for batch_index in range(batch_size):
        candidates = valid_by_chunk[batch_index].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("no non-bootstrap, non-padded expansion chunk exists")
        chunk_index = int(candidates[-1].item())
        lo = chunk_index * frame_chunk_size
        hi = lo + frame_chunk_size
        selected[batch_index, lo:hi] = valid[batch_index, lo:hi]
    return selected.detach()


def fixed_50_step_shifted_sigmas(
    flow_shift: float, *, device: torch.device
) -> Tensor:
    """Return the frozen shifted 50-step deployment grid including zero."""
    if not math.isfinite(flow_shift) or flow_shift <= 0:
        raise ValueError("flow_shift must be finite and positive")
    unshifted = torch.linspace(1.0, 0.0, 51, device=device, dtype=torch.float32)
    return float(flow_shift) * unshifted / (
        1.0 + (float(flow_shift) - 1.0) * unshifted
    )


def masked_video_rms(value: Tensor, mask: Tensor) -> Tensor:
    """Per-sample RMS over valid video elements."""
    expanded = _broadcast_video_mask(mask, value)
    elements_per_frame = value.shape[1] * value.shape[3] * value.shape[4]
    denominator = expanded.sum(dim=(1, 2, 3, 4)) * elements_per_frame
    if bool((denominator <= 0).any()):
        raise ValueError("every sample needs at least one valid expansion frame")
    # Storage padding may contain NaN/Inf; multiplying by a zero mask would
    # preserve those values. Remove padding before squaring and reducing.
    masked = value.float().masked_fill(~expanded.bool(), 0)
    square_sum = masked.square().sum(dim=(1, 2, 3, 4))
    return torch.sqrt(square_sum / denominator)


def normalize_video_direction(direction: Tensor, mask: Tensor) -> Tensor:
    expanded = _broadcast_video_mask(mask, direction)
    masked = direction.detach().float().masked_fill(~expanded.bool(), 0)
    norm = masked_video_rms(masked, mask)
    if bool((norm <= 0).any()) or not bool(torch.isfinite(norm).all()):
        raise ValueError("expansion direction must have finite nonzero masked RMS")
    return masked / norm[:, None, None, None, None]


def rademacher_orthogonal_direction(
    tangent: Tensor,
    mask: Tensor,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Deterministic masked Rademacher direction orthogonal to ``tangent``."""
    tangent_unit = normalize_video_direction(tangent, mask)
    expanded = _broadcast_video_mask(mask, tangent)
    random = torch.randint(
        0,
        2,
        tangent.shape,
        generator=generator,
        device=tangent.device,
        dtype=torch.int64,
    ).float()
    random = random.mul_(2).sub_(1).masked_fill(~expanded.bool(), 0)
    reduce_dims = (1, 2, 3, 4)
    coefficient = (random * tangent_unit).sum(
        dim=reduce_dims
    ) / tangent_unit.square().sum(dim=reduce_dims)
    orthogonal = random - coefficient[:, None, None, None, None] * tangent_unit
    return normalize_video_direction(orthogonal, mask)


def quantized_one_sided_chord(
    center: Tensor,
    direction: Tensor,
    mask: Tensor,
    *,
    epsilon_rms: float,
    model_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return model-visible center/plus states and their detached FP32 chord."""
    if not math.isfinite(epsilon_rms) or epsilon_rms <= 0:
        raise ValueError(f"epsilon_rms must be finite and positive, got {epsilon_rms}")
    if center.shape != direction.shape:
        raise ValueError("center and direction must have identical shapes")
    if not torch.empty((), dtype=model_dtype).is_floating_point():
        raise TypeError("model_dtype must be floating point")
    unit = normalize_video_direction(direction, mask).detach()
    expanded = _broadcast_video_mask(mask, center)
    center_visible = center.detach().to(dtype=model_dtype)
    plus_fp32 = center_visible.float() + float(epsilon_rms) * unit * expanded
    plus_visible = plus_fp32.to(dtype=model_dtype)
    chord = (plus_visible.float() - center_visible.float()).detach()
    chord_rms = masked_video_rms(chord, mask)
    if bool((chord_rms <= 0).any()) or not bool(torch.isfinite(chord_rms).all()):
        raise RuntimeError(
            "model-dtype expansion chord collapsed to zero or non-finite"
        )
    return center_visible, plus_visible, chord


def local_reverse_euler_expansion_loss(
    *,
    chord: Tensor,
    velocity_center: Tensor,
    velocity_plus: Tensor,
    mask: Tensor,
    delta_sigma: Tensor | float,
    maximum_unpenalized_rate: float = 8.0,
) -> LocalExpansionResult:
    """Measure and penalize expansion of one reverse-Euler finite secant."""
    if chord.shape != velocity_center.shape or chord.shape != velocity_plus.shape:
        raise ValueError("chord and both velocity tensors must have identical shapes")
    if not math.isfinite(maximum_unpenalized_rate) or maximum_unpenalized_rate <= 0:
        raise ValueError("maximum_unpenalized_rate must be finite and positive")

    batch = chord.shape[0]
    h = (
        torch.as_tensor(delta_sigma, device=chord.device, dtype=torch.float32)
        .detach()
        .reshape(-1)
    )
    if h.numel() == 1:
        h = h.expand(batch)
    if h.shape != (batch,) or not bool(torch.isfinite(h).all()) or bool((h >= 0).any()):
        raise ValueError(
            "delta_sigma must provide one finite negative value per sample"
        )

    chord = chord.detach().float()
    delta_velocity = velocity_plus.float() - velocity_center.float()
    next_chord = chord + h[:, None, None, None, None] * delta_velocity
    chord_rms = masked_video_rms(chord, mask).detach()
    next_chord_rms = masked_video_rms(next_chord, mask)
    expansion = next_chord_rms / chord_rms
    rate = torch.log(expansion.clamp_min(torch.finfo(torch.float32).tiny)) / h.abs()
    scaled_excess = torch.relu(
        (rate - float(maximum_unpenalized_rate)) / float(maximum_unpenalized_rate)
    )
    loss = scaled_excess.square().mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("local expansion loss is non-finite")
    return LocalExpansionResult(
        loss=loss,
        rate=rate,
        expansion=expansion,
        chord_rms=chord_rms,
        next_chord_rms=next_chord_rms,
    )


def nearest_reverse_delta_sigma(
    sigma: Tensor,
    deployment_sigmas: Sequence[float] | Tensor,
) -> Tensor:
    """Map each sigma to the nearest interval of a fixed descending schedule."""
    schedule = torch.as_tensor(
        deployment_sigmas, device=sigma.device, dtype=torch.float32
    ).flatten()
    if schedule.numel() < 2 or not bool(torch.isfinite(schedule).all()):
        raise ValueError("deployment_sigmas must contain at least two finite values")
    deltas = schedule[1:] - schedule[:-1]
    if bool((deltas >= 0).any()):
        raise ValueError("deployment_sigmas must be strictly descending")
    flat_sigma = sigma.float().reshape(-1)
    indices = (flat_sigma[:, None] - schedule[:-1][None, :]).abs().argmin(dim=1)
    return deltas[indices].reshape(sigma.shape)


__all__ = [
    "LocalExpansionResult",
    "Phase6ExpansionSpec",
    "current_video_chunk_mask",
    "fixed_50_step_shifted_sigmas",
    "local_reverse_euler_expansion_loss",
    "masked_video_rms",
    "nearest_reverse_delta_sigma",
    "normalize_video_direction",
    "phase6_expansion_spec",
    "quantized_one_sided_chord",
    "rademacher_orthogonal_direction",
]
