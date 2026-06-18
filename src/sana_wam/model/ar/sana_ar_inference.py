"""Inference-time primitives for SANA autoregressive rollout.

At training time the clean conditioning history lives *in the sequence* (the
``v_clean`` / ``a_clean`` copies) and the structured kernel
(:func:`sana_ar_linear_attn._ar_chunked_linear_attn`) sums it via a windowed
per-frame prefix. At inference we don't re-run the whole history every step —
instead the per-frame clean dual-track states ``(S, z)`` are **cached** (the
linear-attention analogue of a softmax KV cache) and only the current noisy chunk
is run through the DiT.

Two primitives:

- :class:`ARLinearStateCache` — a per-layer, frame-indexed store of clean
  ``(S = Σ v⊗tilde_k, z = Σ phi_k)`` states with a sliding window and a
  predicted/confirmed (`is_pred`) flag. Mirrors LingBot's KV cache predicted/real
  replacement (``model.py`` ``clear_pred_cache`` / ``update_cache``): a chunk
  predicted this step is added with ``is_pred=True``; when the real observation
  arrives, :meth:`clear_pred` drops it and the confirmed state is written.
- :func:`ar_inference_attn` — one noisy chunk's dual-track linear attention,
  combining the cached windowed-clean state with the chunk's own within-frame
  noise block. This is exactly the noisy-row computation of
  ``_ar_chunked_linear_attn`` with the clean history supplied externally (proven
  equivalent in ``tests/test_sana_ar_inference.py``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

__all__ = ["ARLinearStateCache", "ar_inference_attn"]


class ARLinearStateCache:
    """Per-layer, frame-indexed cache of clean dual-track ``(S, z)`` states.

    One instance covers all DiT layers (``num_layers``). Entries are keyed by
    ``frame_id`` (modality-parity chunk index, video ``2c`` / action ``2c+1``)
    so a single windowed lookup serves both modalities and respects the
    asymmetric video↔action visibility for free.
    """

    def __init__(self, num_layers: int, window: int):
        self.num_layers = int(num_layers)
        self.window = int(window)
        # entries[layer] = list of dicts {frame_id, S, z, is_pred}
        self._entries: List[List[Dict]] = [[] for _ in range(self.num_layers)]

    def reset(self) -> None:
        self._entries = [[] for _ in range(self.num_layers)]

    def clear_pred(self) -> None:
        """Drop all predicted (not-yet-confirmed) entries across every layer.

        Called when a real observation arrives, before writing the confirmed
        state for the same frames (LingBot stage 3: real obs replaces predicted).
        """
        for layer in range(self.num_layers):
            self._entries[layer] = [e for e in self._entries[layer] if not e["is_pred"]]

    def update(self, layer_id: int, frame_id: int, S: Tensor, z: Tensor, *, is_pred: bool) -> None:
        """Write the clean ``(S, z)`` state for ``frame_id`` at ``layer_id``.

        Evicts entries older than the sliding window relative to ``frame_id`` to
        keep the cache bounded (LingBot's ring buffer).
        """
        entries = self._entries[layer_id]
        entries.append({"frame_id": int(frame_id), "S": S, "z": z, "is_pred": bool(is_pred)})
        cutoff = int(frame_id) - self.window
        self._entries[layer_id] = [e for e in entries if e["frame_id"] >= cutoff]

    def windowed_state(
        self, layer_id: int, query_frame: int, *, hi_inclusive: int, window: Optional[int] = None
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Sum cached ``(S, z)`` over frames in ``[query_frame - window, hi_inclusive]``.

        ``hi_inclusive = query_frame - 1`` gives the strict-causal history a noisy
        query sees. ``window`` defaults to the cache's sliding-window width;
        override it to decouple lookup width from the eviction width. Returns
        ``None`` if no cached frame falls in the window.
        """
        win = self.window if window is None else int(window)
        lo = query_frame - win
        S_sum: Optional[Tensor] = None
        z_sum: Optional[Tensor] = None
        for e in self._entries[layer_id]:
            fid = e["frame_id"]
            if lo <= fid <= hi_inclusive:
                S_sum = e["S"] if S_sum is None else S_sum + e["S"]
                z_sum = e["z"] if z_sum is None else z_sum + e["z"]
        if S_sum is None:
            return None
        return S_sum, z_sum


def clean_state_from_tokens(tilde_k: Tensor, v: Tensor, phi_k: Tensor) -> Tuple[Tensor, Tensor]:
    """Dual-track clean state for a set of (single-frame) clean key tokens.

    ``S = Σ_j v_j ⊗ tilde_k_j`` ``(B, H, d, d)``; ``z = Σ_j phi_k_j`` ``(B, H, 1, d)``.
    All inputs ``(B, H, Nk, d)``.
    """
    S = v.transpose(-1, -2) @ tilde_k  # (B, H, d, d)
    z = phi_k.sum(dim=-2, keepdim=True)  # (B, H, 1, d)
    return S, z


def ar_inference_attn(
    tilde_q: Tensor,
    phi_q: Tensor,
    win_state: Optional[Tuple[Tensor, Tensor]],
    tilde_k_self: Tensor,
    phi_k_self: Tensor,
    v_self: Tensor,
    eps: float = 1e-15,
) -> Tensor:
    """One noisy chunk's dual-track linear attention against cache + own block.

    Parameters
    ----------
    tilde_q, phi_q
        Noisy chunk queries ``(B, H, Nq, d)`` (rotated / unrotated ReLU tracks).
    win_state
        ``(S_win, z_win)`` windowed cached clean state, or ``None`` (no history).
    tilde_k_self, phi_k_self, v_self
        The chunk's own noisy keys/values ``(B, H, Nk, d)`` — the within-frame
        ``noise2noise`` block.
    """
    if win_state is not None:
        S_win, z_win = win_state
        num = tilde_q @ S_win.transpose(-1, -2)  # (B, H, Nq, d)
        denom = phi_q @ z_win.transpose(-1, -2)  # (B, H, Nq, 1)
    else:
        num = torch.zeros_like(tilde_q)
        denom = torch.zeros(*tilde_q.shape[:-1], 1, dtype=tilde_q.dtype, device=tilde_q.device)

    a = tilde_q @ tilde_k_self.transpose(-1, -2)  # (B, H, Nq, Nk)
    bb = phi_q @ phi_k_self.transpose(-1, -2)  # (B, H, Nq, Nk)
    num = num + a @ v_self
    denom = denom + bb.sum(dim=-1, keepdim=True)
    return num / (denom + eps)
