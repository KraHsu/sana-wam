"""Phase 5 (engine primitive): single-chunk cache-aware DiT pass.

Exercises SanaARMoTJointDriver.run_ar_chunk_through_backbone on a mini SANA video
backbone: ingesting a clean chunk populates the linear-state cache, and a later
chunk's output then depends on that cached history (closed-loop feedback flows
through the cache). CPU-only; gated on third_party/Sana.
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
def test_single_chunk_pass_uses_cache_history():
    from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
    from sana_wam.model.architecture import DualSystemARArchitecture
    from sana_wam.model.ar.sana_ar_inference import ARLinearStateCache
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone

    torch.manual_seed(0)
    vb = SanaVideoBackbone.from_mini_config(depth=2, hidden_size=128, num_heads=4, linear_head_dim=32, h=8, w=8)
    ab = ActionDiT(
        action_dim=8, dim=64, ffn_dim=128, num_heads=vb.num_heads, num_layers=vb.num_layers,
        video_dim=vb.dim, bridge_layers=tuple(range(vb.num_layers)), variant="joint_self_attn",
        attn_head_dim=vb.head_dim, text_dim=vb.context_dim, attn_kernel="linear_relu",
    )
    arch = DualSystemARArchitecture(cfg=None)
    arch.video_backbone, arch.action_backbone = vb, ab
    arch._mot_driver_kwargs = {"attention_mask_mode": "joint", "video_attention_mask_mode": "first_frame_causal", "mot_checkpoint_mixed_attn": False}
    driver = arch.build_mot_driver()

    B, fcs = 1, 2
    in_ch = vb._dit.in_channels
    L = 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)

    def _prep(latent, ts):
        return vb.prepare(latents=latent, timestep=torch.full((B,), float(ts)), context=context, seq_lens=seq_lens)

    # 1. Ingest a clean chunk at frame 0 -> populate the cache.
    cache = ARLinearStateCache(num_layers=vb.num_layers, window=99)
    clean_latent = torch.randn(B, in_ch, fcs, 8, 8)
    driver.run_ar_chunk_through_backbone(vb, _prep(clean_latent, 0.0), cache, frame_id=0, store_clean=True)
    assert cache.windowed_state(0, 2, hi_inclusive=1) is not None  # frame 0 visible from frame 2

    # 2. Run a later chunk (frame 2) with the populated cache vs an empty cache.
    chunk_latent = torch.randn(B, in_ch, fcs, 8, 8)
    s_hist = driver.run_ar_chunk_through_backbone(vb, _prep(chunk_latent, 0.5), cache, frame_id=2, store_clean=False)
    out_hist = vb.finalize(s_hist)

    empty = ARLinearStateCache(num_layers=vb.num_layers, window=99)
    s_empty = driver.run_ar_chunk_through_backbone(vb, _prep(chunk_latent, 0.5), empty, frame_id=2, store_clean=False)
    out_empty = vb.finalize(s_empty)

    assert torch.isfinite(out_hist).all() and torch.isfinite(out_empty).all()
    assert out_hist.shape == out_empty.shape == (B, in_ch, fcs, 8, 8)
    # Cached clean history must change the prediction.
    assert not torch.allclose(out_hist, out_empty, atol=1e-5)
