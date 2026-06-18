"""Regression: absolute-offset video RoPE must fail loud on table overflow.

Long-horizon AR rollout feeds growing absolute frame offsets into
``_rope_freqs_with_frame_index`` (``arange(c*fcs, ...)``). Once an offset exceeds
the SANA rope module's precomputed ``freqs`` table, a bare ``ft[idx]`` gather
raises an opaque (CUDA-side) ``IndexError``. This pins the friendly, actionable
``ValueError`` — symmetric to the action backbone's ``_get_rope_freqs_at`` — so a
horizon misconfiguration is diagnosable rather than a cryptic crash.

Uses a tiny fake rope module (only ``attention_head_dim`` + ``freqs`` are read),
so no SANA dependency.
"""

from __future__ import annotations

import pytest
import torch

# blocks_split lives under the ``video_backbone`` package whose ``__init__``
# pulls the full SANA/wan stack (modelscope etc.); like every other video-RoPE
# test it is gated on that import succeeding. The function under test has no SANA
# dependency itself — only the package import chain does — so it runs wherever
# SANA is installed (the cluster) and skips on CPU-only CI.
try:
    from sana_wam.model.video_backbone.sana.blocks_split import _rope_freqs_with_frame_index

    _IMPORTABLE = True
except Exception:  # pragma: no cover - exercised only where SANA is absent
    _IMPORTABLE = False

pytestmark = pytest.mark.skipif(
    not _IMPORTABLE, reason="sana_wam.model.video_backbone.sana not importable (SANA/modelscope absent)"
)


class _FakeRope:
    """Minimal stand-in for SANA's ``WanRotaryPosEmbed``.

    ``head_dim=12`` ⇒ ``freqs`` columns split ``[t_w=2, hw=2, hw=2]`` (the
    frame/height/width bands ``_rope_freqs_with_frame_index`` consumes).
    """

    attention_head_dim = 12

    def __init__(self, max_seq_len: int):
        self.freqs = torch.randn(max_seq_len, 6, dtype=torch.cfloat)


def test_in_bounds_frame_index_ok():
    rope = _FakeRope(max_seq_len=8)
    h = w = 2
    idx = torch.arange(4)  # within table
    out = _rope_freqs_with_frame_index(rope, idx, h, w, torch.device("cpu"))
    assert out.shape == (1, 1, idx.shape[0] * h * w, 6)
    assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()


def test_frame_index_overflow_raises_actionable_error():
    rope = _FakeRope(max_seq_len=4)
    # absolute offsets a long rollout would reach: 2..5, exceeds table length 4.
    idx = torch.arange(2, 6)
    with pytest.raises(ValueError, match="exceeds the precomputed SANA video RoPE table"):
        _rope_freqs_with_frame_index(rope, idx, 2, 2, torch.device("cpu"))


def test_negative_frame_index_raises():
    rope = _FakeRope(max_seq_len=4)
    with pytest.raises(ValueError, match="non-negative"):
        _rope_freqs_with_frame_index(rope, torch.tensor([-1, 0, 1]), 2, 2, torch.device("cpu"))


def test_boundary_index_equal_to_length_raises():
    """Index == table length is out of range (0-based gather)."""
    rope = _FakeRope(max_seq_len=4)
    with pytest.raises(ValueError, match="exceeds the precomputed SANA video RoPE table"):
        _rope_freqs_with_frame_index(rope, torch.tensor([0, 1, 4]), 2, 2, torch.device("cpu"))
