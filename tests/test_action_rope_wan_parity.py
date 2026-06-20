"""Pin the action backbone's 1D RoPE to upstream Wan's rotary convention.

``components.precompute_freqs_cis_1d`` / ``rope_apply_1d`` must match Wan's
``rope_params`` / ``rope_apply`` (``../Sana/diffusion/model/wan/model.py``,
upstream Sana-wm's ``pos_embed_type: wan_rope``) so the fork's action stream
shares the same positional phase the video backbone (Wan-derived) uses.

The convention is reproduced standalone here (no submodule needed): both sides
use ``theta=10000``, ``torch.polar`` for freqs, and a ``view_as_complex`` on a
float64 ``(..., -1, 2)`` reshape for the rotation.
"""

from __future__ import annotations

import torch

from sana_wam.model.action_backbone.components import precompute_freqs_cis_1d, rope_apply_1d


def _wan_rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Verbatim transcription of Wan ``rope_params`` (model.py:125)."""
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _wan_rope_apply_1d(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Wan ``rope_apply`` reduced to the 1D (B, H, S, D) case (model.py:135)."""
    b, h, s, d = x.shape
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(b, h, s, d // 2, 2))
    out = torch.view_as_real(x_c * freqs.view(1, 1, s, d // 2)).flatten(-2)
    return out.to(x.dtype)


def test_precompute_freqs_matches_wan():
    head_dim, max_len = 64, 32
    got = precompute_freqs_cis_1d(head_dim, max_len=max_len, theta=10000.0)
    ref = _wan_rope_params(max_len, head_dim, theta=10000.0)
    assert got.shape == ref.shape == (max_len, head_dim // 2)
    torch.testing.assert_close(got, ref)


def test_rope_apply_matches_wan():
    torch.manual_seed(0)
    b, h, s, d = 2, 4, 16, 64
    x = torch.randn(b, h, s, d, dtype=torch.float32)
    freqs = precompute_freqs_cis_1d(d, max_len=s, theta=10000.0)

    got = rope_apply_1d(x, freqs)
    ref = _wan_rope_apply_1d(x, freqs)

    assert got.shape == x.shape
    torch.testing.assert_close(got, ref)


def test_rope_apply_preserves_norm():
    """A rotation is norm-preserving: |rope(x)| == |x| per (head, position)."""
    torch.manual_seed(1)
    b, h, s, d = 1, 2, 8, 32
    x = torch.randn(b, h, s, d, dtype=torch.float32)
    freqs = precompute_freqs_cis_1d(d, max_len=s)
    out = rope_apply_1d(x, freqs)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)
