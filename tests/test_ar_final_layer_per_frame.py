"""F2: per-frame timestep must reach the SANA ``final_layer`` readout, not just
the block AdaLN.

The AR loss samples an independent diffusion timestep per chunk. The block-level
modulation already honours that (3D ``t0`` path), but the ``final_layer`` readout
was modulated by the per-sample mean — so a frame at sigma≈1 and one at sigma≈0
shared one readout scale/shift. ``blocks_split.prepare`` now builds a 4D ``t``
(``(B, 1, F, D)``) when ``frame_timesteps`` is given, routing ``final_layer``
through its upstream ``forward_frame_aware`` path.

This pins:
1. **Safety** — with a *uniform* per-frame timestep the new path is byte-identical
   to the per-sample ``forward`` (``forward_frame_aware`` collapses to the same
   shift/scale per frame), so no pretrained behaviour drifts.
2. **The fix** — distinct per-chunk timesteps change the readout, and the change
   is *frame-local*: only the frames whose timestep changed move; the others are
   bit-stable. (Block attention is skipped here so ``final_layer`` is isolated.)

Gated on third_party/Sana (needs a real ``T2IFinalLayer``).
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


def _mini_backbone():
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone

    return SanaVideoBackbone.from_mini_config(depth=2, hidden_size=128, num_heads=4, linear_head_dim=32, h=8, w=8)


@requires_sana
def test_uniform_frame_timesteps_match_per_sample():
    """Uniform per-frame t ⇒ final-layer output identical to the per-sample path."""
    torch.manual_seed(0)
    vb = _mini_backbone()
    B, fcs, in_ch = 1, 4, vb._dit.in_channels
    L = 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    latents = torch.randn(B, in_ch, fcs, 8, 8)
    ts = 137  # integer: survives the long() norm-scale discipline

    state_ps = vb.prepare(latents=latents, timestep=torch.full((B,), float(ts)), context=context, seq_lens=seq_lens)
    state_pf = vb.prepare(
        latents=latents,
        timestep=torch.full((B,), float(ts)),
        frame_timesteps=torch.full((B, fcs), float(ts)),
        context=context,
        seq_lens=seq_lens,
    )
    out_ps = vb.finalize(state_ps)
    out_pf = vb.finalize(state_pf)
    assert out_pf.shape == out_ps.shape == (B, in_ch, fcs, 8, 8)
    torch.testing.assert_close(out_pf, out_ps)


@requires_sana
def test_per_chunk_final_modulation_is_real_and_frame_local():
    """Changing one chunk's timestep moves only that chunk's readout frames."""
    torch.manual_seed(1)
    vb = _mini_backbone()
    B, fcs, in_ch = 1, 4, vb._dit.in_channels  # 4 frames = 2 chunks of 2
    L = 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    latents = torch.randn(B, in_ch, fcs, 8, 8)
    a, b = 100, 700  # distinct integer timesteps

    def _finalize(frame_ts):
        st = vb.prepare(
            latents=latents,
            timestep=torch.full((B,), float(a)),
            frame_timesteps=torch.tensor(frame_ts, dtype=torch.float32).view(B, fcs),
            context=context,
            seq_lens=seq_lens,
        )
        return vb.finalize(st)

    out_ref = _finalize([a, a, a, a])  # all chunk t = a
    out_mix = _finalize([a, a, b, b])  # chunk 0 = a, chunk 1 = b

    # Chunk 0 (frames 0-1) timestep unchanged ⇒ readout bit-stable.
    torch.testing.assert_close(out_mix[:, :, :2], out_ref[:, :, :2])
    # Chunk 1 (frames 2-3) timestep changed a→b ⇒ readout must move.
    assert not torch.allclose(out_mix[:, :, 2:], out_ref[:, :, 2:], atol=1e-5)
