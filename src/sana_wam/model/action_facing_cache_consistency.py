"""Source-level consistency loss for SANA's action-facing video cache."""

from __future__ import annotations

import math

import torch
from torch import Tensor


AFCC_EPSILON = 1.0e-6


def validate_afcc_weight(value) -> float:
    """Return the fixed binary AFCC treatment weight."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "action_facing_cache_consistency_weight must be exactly 0 or 1"
        )
    weight = float(value)
    if not math.isfinite(weight) or weight not in {0.0, 1.0}:
        raise ValueError(
            "action_facing_cache_consistency_weight must be exactly 0 or 1"
        )
    return weight


def validate_afcc_reference(
    reference: Tensor,
    *,
    batch_size: int,
    num_layers: int,
    num_chunks: int,
    num_heads: int,
    action_tokens_per_chunk: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Validate the complete detached ``[B,L,C,H,T,D]`` teacher tensor."""

    if not isinstance(reference, Tensor):
        raise TypeError("phase1_action_facing_cache_reference must be a tensor")
    expected = (
        batch_size,
        num_layers,
        num_chunks,
        num_heads,
        action_tokens_per_chunk,
        head_dim,
    )
    if tuple(reference.shape) != expected:
        raise ValueError(
            "phase1_action_facing_cache_reference must have shape "
            f"{expected}, got {tuple(reference.shape)}"
        )
    if reference.layout != torch.strided:
        raise TypeError("AFCC reference must be a strided tensor")
    if reference.dtype != dtype:
        raise ValueError(
            f"AFCC reference dtype must be {dtype}, got {reference.dtype}"
        )
    if reference.device != device:
        raise ValueError(
            f"AFCC reference must be on {device}, got {reference.device}"
        )
    if reference.requires_grad or reference.grad_fn is not None:
        raise ValueError("AFCC reference must be detached")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("AFCC reference contains non-finite values")
    return reference.detach()


def normalized_action_facing_numerator_loss(
    student: Tensor,
    reference: Tensor,
    valid_query_mask: Tensor,
    *,
    epsilon: float = AFCC_EPSILON,
) -> Tensor:
    """Compute one equally weighted layer/chunk AFCC term.

    ``student`` and ``reference`` are the production numerator tensors with
    shape ``[B,H,T,D]``. The loss is accumulated in FP32, but both numerator
    operands retain the production dtype and reduction order that created them.
    """

    if not isinstance(student, Tensor) or not isinstance(reference, Tensor):
        raise TypeError("AFCC student and reference must be tensors")
    if student.ndim != 4 or reference.shape != student.shape:
        raise ValueError(
            "AFCC student/reference must share shape [B,H,T,D]; got "
            f"{tuple(student.shape)} and {tuple(reference.shape)}"
        )
    if student.dtype != reference.dtype or student.device != reference.device:
        raise ValueError("AFCC student/reference dtype and device must match")
    if reference.requires_grad or reference.grad_fn is not None:
        raise ValueError("AFCC reference must be detached")
    if (
        not isinstance(valid_query_mask, Tensor)
        or valid_query_mask.dtype != torch.bool
        or valid_query_mask.layout != torch.strided
        or tuple(valid_query_mask.shape) != (student.shape[0], student.shape[2])
    ):
        raise ValueError("valid_query_mask must be boolean [B,T]")
    if valid_query_mask.device != student.device:
        raise ValueError("valid_query_mask must share the student device")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("AFCC epsilon must be finite and positive")

    valid_per_sample = valid_query_mask.sum(dim=1)
    if bool((valid_per_sample == 0).any()):
        raise ValueError("every AFCC sample/chunk needs one valid action query")

    mask = valid_query_mask[:, None, :, None]
    student_fp32 = student.float().masked_fill(~mask, 0.0)
    reference_fp32 = reference.detach().float().masked_fill(~mask, 0.0)
    element_count = (
        valid_per_sample.to(torch.float32)
        * float(student.shape[1] * student.shape[3])
    )
    squared_error = (student_fp32 - reference_fp32).square().sum(
        dim=(1, 2, 3)
    )
    reference_energy = reference_fp32.square().sum(dim=(1, 2, 3))
    mse = squared_error / element_count
    reference_rms_squared = reference_energy / element_count
    return (mse / (reference_rms_squared + float(epsilon))).mean()


def action_facing_cache_consistency_loss(
    student: Tensor,
    reference: Tensor,
    valid_query_mask: Tensor,
    *,
    epsilon: float = AFCC_EPSILON,
) -> Tensor:
    """Average normalized numerator error over all non-empty layer/chunks.

    The numerator tensors use ``[B,L,C,H,T,D]`` and the query mask uses
    ``[B,C,T]``. Empty padded chunks remain captured for the 20-by-4 coverage
    contract but do not define a loss term.
    """

    if not isinstance(student, Tensor) or not isinstance(reference, Tensor):
        raise TypeError("AFCC student and reference must be tensors")
    if student.ndim != 6 or reference.shape != student.shape:
        raise ValueError(
            "AFCC student/reference must share shape [B,L,C,H,T,D]; got "
            f"{tuple(student.shape)} and {tuple(reference.shape)}"
        )
    if student.dtype != reference.dtype or student.device != reference.device:
        raise ValueError("AFCC student/reference dtype and device must match")
    if reference.requires_grad or reference.grad_fn is not None:
        raise ValueError("AFCC reference must be detached")
    expected_mask_shape = (student.shape[0], student.shape[2], student.shape[4])
    if (
        not isinstance(valid_query_mask, Tensor)
        or valid_query_mask.dtype != torch.bool
        or valid_query_mask.layout != torch.strided
        or tuple(valid_query_mask.shape) != expected_mask_shape
    ):
        raise ValueError(
            f"valid_query_mask must be boolean {expected_mask_shape}"
        )
    if valid_query_mask.device != student.device:
        raise ValueError("valid_query_mask must share the student device")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("AFCC epsilon must be finite and positive")

    valid_count = valid_query_mask.sum(dim=2)
    active = valid_count > 0
    if bool((~active).all(dim=1).any()):
        raise ValueError("every AFCC sample needs one non-empty action chunk")
    mask = valid_query_mask[:, None, :, None, :, None]
    student_fp32 = student.float().masked_fill(~mask, 0.0)
    reference_fp32 = reference.detach().float().masked_fill(~mask, 0.0)
    element_count = (
        valid_count[:, None, :].to(torch.float32)
        * float(student.shape[3] * student.shape[5])
    )
    safe_count = element_count.clamp_min(1.0)
    squared_error = (student_fp32 - reference_fp32).square().sum(
        dim=(3, 4, 5)
    )
    reference_energy = reference_fp32.square().sum(dim=(3, 4, 5))
    normalized = (squared_error / safe_count) / (
        reference_energy / safe_count + float(epsilon)
    )
    active_interfaces = active[:, None, :].expand(
        student.shape[0], student.shape[1], student.shape[2]
    )
    normalized = normalized.masked_fill(~active_interfaces, 0.0)
    return normalized.sum() / active_interfaces.sum().to(torch.float32)


__all__ = [
    "AFCC_EPSILON",
    "action_facing_cache_consistency_loss",
    "normalized_action_facing_numerator_loss",
    "validate_afcc_reference",
    "validate_afcc_weight",
]
