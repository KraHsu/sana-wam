"""One-sided action-risk loss with adapter-only gradient injection."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn


def unweighted_pad_masked_action_mse(
    prediction: Tensor,
    target: Tensor,
    action_is_pad: Tensor | None = None,
) -> Tensor:
    """Return one unweighted action-velocity MSE scalar per sample."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and target must share shape [batch, tokens, action_dim]"
        )
    if action_is_pad is None:
        per_token = (prediction.float() - target.float()).square().mean(dim=2)
        result = per_token.mean(dim=1)
    else:
        if action_is_pad.dtype != torch.bool or tuple(action_is_pad.shape) != tuple(
            prediction.shape[:2]
        ):
            raise ValueError(
                "action_is_pad must be boolean with shape [batch, tokens]"
            )
        pad = action_is_pad.detach().to(device=prediction.device)
        keep = (~pad).to(dtype=torch.float32)
        denominator = keep.sum(dim=1)
        if bool((denominator <= 0).any()):
            raise ValueError("every sample needs one non-padded action token")
        # Multiplication by zero does not suppress NaN/Inf in padded storage.
        pad_3d = pad[:, :, None]
        per_token = (
            prediction.float().masked_fill(pad_3d, 0)
            - target.float().masked_fill(pad_3d, 0)
        ).square().mean(dim=2)
        per_token = per_token.masked_fill(pad, 0)
        result = per_token.sum(dim=1) / denominator
    if not bool(torch.isfinite(result).all()):
        raise ValueError("unweighted action error is non-finite")
    return result


def one_sided_relative_excess_loss(
    student_error: Tensor,
    reference_error: Tensor,
    *,
    relative_epsilon: float = 1.0e-4,
) -> Tensor:
    """Penalize per-sample harm while assigning improvements exactly zero."""
    if student_error.shape != reference_error.shape:
        raise ValueError("student and reference errors must have identical shapes")
    if student_error.numel() == 0:
        raise ValueError("action non-regression needs at least one sample")
    if not math.isfinite(relative_epsilon) or relative_epsilon <= 0:
        raise ValueError("relative_epsilon must be finite and positive")
    if not bool(torch.isfinite(student_error).all()):
        raise ValueError("student action error is non-finite")
    if not bool(torch.isfinite(reference_error).all()) or bool(
        (reference_error < 0).any()
    ):
        raise ValueError("reference action error must be finite and non-negative")

    relative_excess = (student_error.float() - reference_error.float()) / (
        reference_error.float() + float(relative_epsilon)
    )
    return torch.relu(relative_excess).square().mean()


def adapter_only_loss_surrogate(
    loss: Tensor,
    adapter: nn.Module | Iterable[nn.Parameter],
) -> Tensor:
    """Return the same scalar value with gradients only for adapter parameters."""
    if loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
        raise ValueError("adapter-only loss must be a finite scalar")
    parameters = tuple(
        adapter.parameters() if isinstance(adapter, nn.Module) else adapter
    )
    if not parameters:
        raise ValueError("adapter-only loss received no parameters")
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("every adapter parameter must require gradients")

    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )
    injection = None
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("action non-regression produced a non-finite gradient")
        term = (parameter * gradient.detach()).sum()
        injection = term if injection is None else injection + term
    return loss.detach() + (injection - injection.detach())


__all__ = [
    "adapter_only_loss_surrogate",
    "one_sided_relative_excess_loss",
    "unweighted_pad_masked_action_mse",
]
