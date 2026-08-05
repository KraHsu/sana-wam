"""Fixed-K prefix compaction for CACH Stage-2 mini-model admission.

The vendor cached/Triton GDN does not accept ``frame_valid_mask``.  CACH keeps
the public layout fixed-width and fail-closed, then compacts the already
validated common prefix before the vendor call.  Model outputs and bridge
tokens are restored to fixed width with exact-zero padding afterwards.

This module is torch-only.  It does not construct a model or access a dataset,
checkpoint, GPU, run root, or filesystem artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FixedKPrefixPlan:
    """One batch-uniform, non-empty contiguous-prefix compaction plan."""

    batch_size: int
    fixed_slots: int
    valid_slots: int

    def __post_init__(self) -> None:
        for name in ("batch_size", "fixed_slots", "valid_slots"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive plain integer")
        if self.valid_slots > self.fixed_slots:
            raise ValueError("valid_slots cannot exceed fixed_slots")

    @classmethod
    def from_mask(cls, frame_valid_mask: Tensor) -> "FixedKPrefixPlan":
        if not isinstance(frame_valid_mask, Tensor):
            raise TypeError("frame_valid_mask must be a torch Tensor")
        if frame_valid_mask.dtype is not torch.bool or frame_valid_mask.ndim != 2:
            raise TypeError("frame_valid_mask must be exact boolean [B,K]")
        batch_size, fixed_slots = frame_valid_mask.shape
        if batch_size <= 0 or fixed_slots <= 0:
            raise ValueError("frame_valid_mask must have non-empty B and K axes")
        if fixed_slots > 1 and bool(
            ((~frame_valid_mask[:, :-1]) & frame_valid_mask[:, 1:]).any()
        ):
            raise ValueError("frame_valid_mask must be a contiguous prefix")
        counts = frame_valid_mask.long().sum(dim=1)
        if bool((counts <= 0).any()):
            raise ValueError("every CACH chunk must contain a valid latent prefix")
        if not bool((counts == counts[0]).all()):
            raise ValueError(
                "frame_valid_mask must have one batch-uniform prefix length"
            )
        return cls(
            batch_size=int(batch_size),
            fixed_slots=int(fixed_slots),
            valid_slots=int(counts[0].item()),
        )

    @property
    def padding_slots(self) -> int:
        return self.fixed_slots - self.valid_slots

    def compact_video(self, value: Tensor) -> Tensor:
        """Validate ``[B,C,K,...]`` and return its valid temporal prefix."""

        if not isinstance(value, Tensor) or value.ndim < 3:
            raise TypeError("video must be a rank-three-or-greater torch Tensor")
        if value.shape[0] != self.batch_size or value.shape[2] != self.fixed_slots:
            raise ValueError("video batch/fixed-K axes differ from the prefix plan")
        if self.padding_slots:
            padding = value[:, :, self.valid_slots :]
            if bool(padding.detach().count_nonzero()):
                raise ValueError("padded video slots must be exact zero")
        return value[:, :, : self.valid_slots]

    def compact_condition(self, value: Tensor) -> Tensor:
        """Validate ``[B,K,A]`` and return its valid temporal prefix."""

        if not isinstance(value, Tensor) or value.ndim != 3:
            raise TypeError("condition must be a rank-three torch Tensor")
        if value.shape[:2] != (self.batch_size, self.fixed_slots):
            raise ValueError("condition batch/fixed-K axes differ from the prefix plan")
        if self.padding_slots:
            padding = value[:, self.valid_slots :]
            if bool(padding.detach().count_nonzero()):
                raise ValueError("padded condition slots must be exact zero")
        return value[:, : self.valid_slots]

    def compact_frame_index(self, value: Tensor | None) -> Tensor | None:
        if value is None:
            return None
        if not isinstance(value, Tensor) or value.ndim not in {1, 2}:
            raise TypeError("frame_index must be a rank-one or rank-two Tensor")
        if value.shape[-1] != self.fixed_slots:
            raise ValueError("frame_index fixed-K axis differs from the prefix plan")
        if value.ndim == 2 and value.shape[0] not in {1, self.batch_size}:
            raise ValueError("frame_index batch axis differs from the prefix plan")
        return value[..., : self.valid_slots]

    def restore_video(self, value: Tensor) -> Tensor:
        """Restore ``[B,C,valid,...]`` to fixed K using exact zeros."""

        if not isinstance(value, Tensor) or value.ndim < 3:
            raise TypeError("vendor video output must be rank three or greater")
        if value.shape[0] != self.batch_size or value.shape[2] != self.valid_slots:
            raise ValueError("vendor video output differs from compacted geometry")
        if not self.padding_slots:
            return value
        padding_shape = list(value.shape)
        padding_shape[2] = self.padding_slots
        return torch.cat([value, value.new_zeros(padding_shape)], dim=2)

    def restore_token_sequence(self, value: Tensor) -> Tensor:
        """Restore ``[B,valid*S,C]`` bridge tokens to ``[B,K*S,C]``."""

        if not isinstance(value, Tensor) or value.ndim != 3:
            raise TypeError("vendor bridge output must be exact [B,N,C]")
        if value.shape[0] != self.batch_size:
            raise ValueError("vendor bridge batch differs from compacted geometry")
        if value.shape[1] % self.valid_slots != 0:
            raise ValueError("vendor bridge tokens do not divide by valid frames")
        if not self.padding_slots:
            return value
        spatial_tokens = value.shape[1] // self.valid_slots
        reshaped = value.reshape(
            self.batch_size,
            self.valid_slots,
            spatial_tokens,
            value.shape[2],
        )
        padding = value.new_zeros(
            self.batch_size,
            self.padding_slots,
            spatial_tokens,
            value.shape[2],
        )
        return torch.cat([reshaped, padding], dim=1).reshape(
            self.batch_size,
            self.fixed_slots * spatial_tokens,
            value.shape[2],
        )

    def compact_token_sequence(self, value: Tensor) -> Tensor:
        """Compact exact-zero ``[B,K*S,C]`` tokens to ``[B,valid*S,C]``."""

        if not isinstance(value, Tensor) or value.ndim != 3:
            raise TypeError("fixed token sequence must be exact [B,N,C]")
        if value.shape[0] != self.batch_size:
            raise ValueError("token sequence batch differs from the prefix plan")
        if value.shape[1] % self.fixed_slots != 0:
            raise ValueError("token sequence does not divide by fixed slots")
        spatial_tokens = value.shape[1] // self.fixed_slots
        reshaped = value.reshape(
            self.batch_size,
            self.fixed_slots,
            spatial_tokens,
            value.shape[2],
        )
        if self.padding_slots and bool(
            reshaped[:, self.valid_slots :].detach().count_nonzero()
        ):
            raise ValueError("padded token-sequence slots must be exact zero")
        return reshaped[:, : self.valid_slots].reshape(
            self.batch_size,
            self.valid_slots * spatial_tokens,
            value.shape[2],
        )


__all__ = ["FixedKPrefixPlan"]
