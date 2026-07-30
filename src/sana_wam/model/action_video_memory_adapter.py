"""Action-only read adapter for cached clean-video linear-attention state.

The canonical AR cache remains the single source of truth.  This module builds
an action-facing *delta* for the even-frame (video) part of a query window:

    S_action = P_V.T @ S_video @ P_K
    z_action = z_video @ D_z

where ``P_K`` and ``P_V`` are per-layer/per-head low-rank residual maps and
``D_z`` is a bounded positive diagonal map.  Returning deltas lets callers add
them to the canonical mixed video/action window without changing its reduction
order, which is required for identity-init endpoint parity.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ActionVideoMemoryAdapter(nn.Module):
    """Per-layer/head residual projection of video ``(S, z)`` cache state.

    One low-rank factor in each numerator projection is initialized to zero, and
    the normalizer logits start at zero.  Consequently ``forward_delta`` returns
    exact zeros at initialization while the zero factors and normalizer logits
    still receive first-step gradients.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        rank: int = 8,
        z_scale_min: float = 0.5,
        z_scale_max: float = 2.0,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        dimensions = {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "rank": rank,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if rank > head_dim:
            raise ValueError(f"rank ({rank}) cannot exceed head_dim ({head_dim})")
        if not (
            math.isfinite(z_scale_min)
            and math.isfinite(z_scale_max)
            and 0.0 < z_scale_min < 1.0 < z_scale_max
        ):
            raise ValueError(
                "z scale bounds must be finite and satisfy "
                f"0 < min < 1 < max; got [{z_scale_min}, {z_scale_max}]"
            )
        if isinstance(init_seed, bool) or not isinstance(init_seed, int):
            raise ValueError(f"init_seed must be an integer, got {init_seed!r}")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rank = rank
        self.z_scale_min = float(z_scale_min)
        self.z_scale_max = float(z_scale_max)
        self.init_seed = init_seed

        shape = (num_layers, num_heads, head_dim, rank)
        # Token-row convention: P = I + left @ right.T.
        self.k_left = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.k_right = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.v_left = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.v_right = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.z_logits = nn.Parameter(
            torch.zeros(num_layers, num_heads, head_dim, dtype=torch.float32)
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Deterministic rank-independent identity initialization."""
        generator = torch.Generator(device="cpu").manual_seed(self.init_seed)
        std = self.head_dim**-0.5
        self.k_left.copy_(
            torch.randn(self.k_left.shape, generator=generator) * std
        )
        self.v_left.copy_(
            torch.randn(self.v_left.shape, generator=generator) * std
        )
        self.k_right.zero_()
        self.v_right.zero_()
        self.z_logits.zero_()

    def _validate_inputs(
        self, layer_id: int, S_video: Tensor, z_video: Tensor
    ) -> None:
        if isinstance(layer_id, bool) or not isinstance(layer_id, int):
            raise TypeError(f"layer_id must be an integer, got {layer_id!r}")
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(
                f"layer_id {layer_id} outside [0, {self.num_layers})"
            )
        expected_s_tail = (self.num_heads, self.head_dim, self.head_dim)
        expected_z_tail = (self.num_heads, 1, self.head_dim)
        if S_video.ndim != 4 or tuple(S_video.shape[1:]) != expected_s_tail:
            raise ValueError(
                "S_video must have shape (B, H, d, d) with "
                f"(H,d)=({self.num_heads},{self.head_dim}); got "
                f"{tuple(S_video.shape)}"
            )
        if z_video.ndim != 4 or tuple(z_video.shape[1:]) != expected_z_tail:
            raise ValueError(
                "z_video must have shape (B, H, 1, d) with "
                f"(H,d)=({self.num_heads},{self.head_dim}); got "
                f"{tuple(z_video.shape)}"
            )
        if S_video.shape[0] != z_video.shape[0]:
            raise ValueError("S_video and z_video batch dimensions differ")
        if S_video.device != z_video.device:
            raise ValueError("S_video and z_video must be on the same device")
        if S_video.dtype != z_video.dtype:
            raise ValueError("S_video and z_video must have the same dtype")
        if not S_video.is_floating_point() or not z_video.is_floating_point():
            raise TypeError("S_video and z_video must be floating-point tensors")

    def normalizer_scale(
        self, layer_id: int, *, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        """Return the bounded positive ``(H, d)`` normalizer scale."""
        logits = self.z_logits[layer_id].to(device=device, dtype=dtype)
        unit = torch.tanh(logits)
        log_min = math.log(self.z_scale_min)
        log_max = math.log(self.z_scale_max)
        # The piecewise log map is exactly zero at identity and approaches the
        # configured bounds without ever making a denominator feature negative.
        log_scale = torch.where(
            unit >= 0,
            unit * log_max,
            (-unit) * log_min,
        )
        return torch.exp(log_scale)

    def forward_delta(
        self, layer_id: int, S_video: Tensor, z_video: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return action-facing ``(delta_S, delta_z)`` for video history."""
        self._validate_inputs(layer_id, S_video, z_video)
        dtype, device = S_video.dtype, S_video.device
        k_left = self.k_left[layer_id].to(device=device, dtype=dtype)
        k_right = self.k_right[layer_id].to(device=device, dtype=dtype)
        v_left = self.v_left[layer_id].to(device=device, dtype=dtype)
        v_right = self.v_right[layer_id].to(device=device, dtype=dtype)

        # S @ P_K, with P_K = I + k_left @ k_right.T.
        right_bottleneck = torch.einsum("bhij,hjr->bhir", S_video, k_left)
        right_update = torch.einsum(
            "bhir,hkr->bhik", right_bottleneck, k_right
        )
        after_k = S_video + right_update

        # P_V.T @ after_k, with P_V = I + v_left @ v_right.T.
        left_bottleneck = torch.einsum("hjr,bhjk->bhrk", v_left, after_k)
        left_update = torch.einsum(
            "hir,bhrk->bhik", v_right, left_bottleneck
        )
        adapted_s = after_k + left_update

        scale = self.normalizer_scale(layer_id, dtype=dtype, device=device)
        adapted_z = z_video * scale.unsqueeze(0).unsqueeze(2)
        return adapted_s - S_video, adapted_z - z_video

    def forward(
        self, layer_id: int, S_video: Tensor, z_video: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the projected video state rather than its residual."""
        delta_s, delta_z = self.forward_delta(layer_id, S_video, z_video)
        return S_video + delta_s, z_video + delta_z


__all__ = ["ActionVideoMemoryAdapter"]
