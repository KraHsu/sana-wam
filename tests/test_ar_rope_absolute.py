"""Absolute-offset RoPE for the AR path.

`_rope_freqs_with_frame_index` reuses the SANA rope module's precomputed freqs
and only swaps the frame-dimension slice for explicit indices. Pins:
- with frame_index = arange(f) it is byte-identical to WanRotaryPosEmbed.forward
  (backward compatibility / no drift for the non-AR path), and
- the duplicated [arange(T), arange(T)] layout gives a chunk's two copies the
  SAME phase (the whole point: noisy and clean copies of a frame must align).

Gated on third_party/Sana (needs a real WanRotaryPosEmbed).
"""

from __future__ import annotations

import pytest
import torch


def _sana_importable() -> bool:
    try:
        import diffusion.model.nets.sana_multi_scale_video  # noqa: F401
    except Exception:
        return False
    return True


requires_sana = pytest.mark.skipif(not _sana_importable(), reason="third_party/Sana not importable")


@requires_sana
def test_arange_matches_wan_rope():
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone
    from sana_wam.model.video_backbone.sana.blocks_split import _rope_freqs_with_frame_index

    vb = SanaVideoBackbone.from_mini_config(depth=1, hidden_size=128, num_heads=4, linear_head_dim=32, h=8, w=8)
    rope = vb._dit.rope  # WanRotaryPosEmbed
    f, h, w = 4, 4, 4
    ref = rope((f, h, w), torch.device("cpu"))
    got = _rope_freqs_with_frame_index(rope, torch.arange(f), h, w, torch.device("cpu"))
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref)


@requires_sana
def test_duplicated_halves_share_phase():
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone
    from sana_wam.model.video_backbone.sana.blocks_split import _rope_freqs_with_frame_index

    vb = SanaVideoBackbone.from_mini_config(depth=1, hidden_size=128, num_heads=4, linear_head_dim=32, h=8, w=8)
    rope = vb._dit.rope
    T, h, w = 2, 4, 4
    idx = torch.cat([torch.arange(T), torch.arange(T)])  # [0,1,0,1]
    freqs = _rope_freqs_with_frame_index(rope, idx, h, w, torch.device("cpu"))
    tokens_per_half = T * h * w
    noisy = freqs[:, :, :tokens_per_half]
    clean = freqs[:, :, tokens_per_half:]
    # noisy chunk and clean chunk occupy the same frame positions -> identical phase.
    torch.testing.assert_close(noisy, clean)

    # And a naive contiguous [0..2T-1] would NOT share phase (sanity on the bug we fix).
    contiguous = _rope_freqs_with_frame_index(rope, torch.arange(2 * T), h, w, torch.device("cpu"))
    c_noisy = contiguous[:, :, :tokens_per_half]
    c_clean = contiguous[:, :, tokens_per_half:]
    assert not torch.allclose(c_noisy, c_clean)
