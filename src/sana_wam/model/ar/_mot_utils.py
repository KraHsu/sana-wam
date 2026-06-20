"""Shared helpers for MoT (Mixture-of-Transformers) driver implementations.

Centralizes utilities shared by the MoT driver implementations so they don't
drift on equivalent computations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sana_wam.model.video_backbone.adapter import BlockLoopState


def compute_video_tokens_per_frame(vstate: "BlockLoopState", driver_name: str) -> int:
    """Derive video tokens-per-frame from the spatial dims populated on ``vstate``.

    Used by every MoT driver to build the v↔v block of the joint attention mask.
    The video backbone's ``prepare()`` must populate ``h`` and ``w`` on the
    ``BlockLoopState``; if not, raise a clear error attributable to the
    calling driver via ``driver_name``.

    Special-token aware: when the encoder declares
    ``spec.has_special_tokens=True``, the per-frame
    block in the DiT self-attn sequence is
    ``[special | patches]`` of length
    ``tokens_per_frame_special + tokens_per_frame_patch``. The MoT mask
    (``build_video_to_video_mask``) must see this *total* per-frame count
    so its ``first_frame_causal`` / ``per_frame_causal`` modes mask the
    correct rows; using ``h * w`` alone would silently mask only the
    patches portion of frame 0 and leave the 17 special tokens of frame 0
    attendable from later frames.
    """
    h = int(getattr(vstate, "h", 0))
    w = int(getattr(vstate, "w", 0))
    if h <= 0 or w <= 0:
        raise ValueError(
            f"{driver_name}: cannot derive video_tokens_per_frame from vstate "
            f"(h={h}, w={w}). The video backbone's prepare() must populate h/w."
        )
    tokens_per_frame_special = int(getattr(vstate, "tokens_per_frame_special", 0))
    if tokens_per_frame_special > 0:
        return h * w + tokens_per_frame_special
    return h * w


__all__ = ["compute_video_tokens_per_frame"]
