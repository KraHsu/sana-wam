"""Torch-only, model-free CACH action-to-video reduction contract.

This module does not import or construct SANA.  It turns one fixed-capacity
layout chunk into the exact ``[B, K, A]`` additive condition and an explicit
``[B, K]`` latent-valid mask.  Padded action slots are never selected and
padded latent slots are exact zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from sana_wam.model.action_chunk_layout import (
    ChunkLayout,
    LayoutContractError,
    LayoutReasonCode,
)


def _validate_action_tensor(
    value: object,
    *,
    name: str,
    batch: int,
    tokens: int,
    action_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch Tensor")
    if value.ndim != 3:
        raise ValueError(f"{name} must have exact rank 3")
    expected = (batch, tokens, action_dim)
    if tuple(value.shape) != expected:
        raise ValueError(
            f"{name} shape must be exactly {expected}, got {tuple(value.shape)}"
        )
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype must be exactly {dtype}, got {value.dtype}")
    if value.device != device:
        raise ValueError(
            f"{name} device must be exactly {device}, got {value.device}"
        )
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return value


@dataclass(frozen=True)
class ActionConditionBatch:
    """One fixed-``K`` action condition plus its explicit valid-latent mask."""

    condition: Tensor
    latent_valid_mask: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.condition, Tensor) or self.condition.ndim != 3:
            raise TypeError("condition must be a rank-three torch Tensor")
        if (
            not isinstance(self.latent_valid_mask, Tensor)
            or self.latent_valid_mask.dtype is not torch.bool
            or self.latent_valid_mask.ndim != 2
        ):
            raise TypeError("latent_valid_mask must be a boolean rank-two Tensor")
        if tuple(self.condition.shape[:2]) != tuple(
            self.latent_valid_mask.shape
        ):
            raise ValueError("condition and latent_valid_mask batch/time axes differ")
        if self.condition.device != self.latent_valid_mask.device:
            raise ValueError("condition and latent_valid_mask devices differ")


def reduce_end_of_bin_action_condition(
    noisy_actions: Tensor,
    *,
    committed_actions: Optional[Tensor],
    chunk: ChunkLayout,
    no_action_slot: Tensor,
) -> ActionConditionBatch:
    """Map one layout chunk to fixed ``[B, K, A]`` conditioning.

    ``noisy_actions`` carries the full registered chunk capacity.  Its
    ``chunk.action_valid_mask`` is a contiguous prefix, so padded tail actions
    remain explicit but are never selected.  Each valid non-anchor latent uses
    the last valid command in its registered action span.  Bootstrap latent 0
    uses the model-owned ``NO_ACTION`` slot.  Padded latent conditions are exact
    zero and must be consumed together with ``latent_valid_mask``.

    Continuation history is shape-bound here.  Canonical committed-action
    value identity and a reviewed history-summary operator remain separate
    admission blockers.
    """

    if not isinstance(chunk, ChunkLayout):
        raise TypeError("chunk must be a ChunkLayout")
    if not isinstance(noisy_actions, Tensor):
        raise TypeError("noisy_actions must be a torch Tensor")
    if noisy_actions.ndim != 3:
        raise ValueError("noisy_actions must have exact rank 3")
    if not isinstance(no_action_slot, Tensor):
        raise TypeError("no_action_slot must be a torch Tensor")
    if no_action_slot.ndim != 1:
        raise ValueError("no_action_slot must have exact shape (A,)")

    batch, token_capacity, action_dim = noisy_actions.shape
    if token_capacity != chunk.action_slot_capacity:
        raise ValueError(
            "noisy_actions must contain the layout's fixed action-slot capacity"
        )
    if no_action_slot.shape[0] != action_dim:
        raise ValueError("no_action_slot width differs from noisy action width")
    if no_action_slot.dtype != noisy_actions.dtype:
        raise TypeError("no_action_slot and noisy_actions dtype differ")
    if no_action_slot.device != noisy_actions.device:
        raise ValueError("no_action_slot and noisy_actions device differ")
    if not bool(torch.isfinite(noisy_actions.detach()).all()):
        raise ValueError("noisy_actions contains NaN or Inf")
    if not bool(torch.isfinite(no_action_slot.detach()).all()):
        raise ValueError("no_action_slot contains NaN or Inf")

    expected_committed = chunk.action_start
    if committed_actions is None:
        if expected_committed != 0:
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_GAP,
                "continuation chunk is missing its committed action prefix",
            )
    else:
        _validate_action_tensor(
            committed_actions,
            name="committed_actions",
            batch=batch,
            tokens=expected_committed,
            action_dim=action_dim,
            dtype=noisy_actions.dtype,
            device=noisy_actions.device,
        )

    chunk_size = len(chunk.latent_valid_mask)
    condition = torch.zeros(
        (batch, chunk_size, action_dim),
        dtype=noisy_actions.dtype,
        device=noisy_actions.device,
    )
    latent_valid_mask = torch.tensor(
        chunk.latent_valid_mask,
        dtype=torch.bool,
        device=noisy_actions.device,
    ).view(1, chunk_size).expand(batch, chunk_size).clone()

    for local_latent, span in enumerate(chunk.latent_action_spans):
        if span.anchor_no_action_slot:
            if not chunk.is_bootstrap or local_latent != 0:
                raise LayoutContractError(
                    LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
                    "the no-action slot appeared outside bootstrap latent 0",
                )
            condition[:, local_latent, :] = no_action_slot
            continue
        if span.action_count <= 0:
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_GAP,
                "a non-anchor latent has no valid action command",
            )
        local_action = span.action_end - chunk.action_start - 1
        if local_action < 0 or local_action >= chunk.valid_action_count:
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_GAP,
                "latent action span escapes the current valid action interval",
            )
        if not chunk.action_valid_mask[local_action]:
            raise LayoutContractError(
                LayoutReasonCode.PAD_MARKED_VALID,
                "a latent action span selected a padded action slot",
            )
        condition[:, local_latent, :] = noisy_actions[:, local_action, :]

    if bool(condition.masked_select(~latent_valid_mask.unsqueeze(-1)).count_nonzero()):
        raise RuntimeError("padded latent action conditions must remain exact zero")
    return ActionConditionBatch(
        condition=condition,
        latent_valid_mask=latent_valid_mask,
    )


__all__ = [
    "ActionConditionBatch",
    "reduce_end_of_bin_action_condition",
]
