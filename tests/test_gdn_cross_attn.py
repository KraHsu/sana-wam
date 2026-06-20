"""End-to-end: GDN video backbone + cross-attention action bridge.

GPU/Triton-gated (GDN's ChunkCausalGDNTriton kernels need CUDA + triton 3.5.1).
Pins the Phase-3 coupling: a GDN video backbone runs natively, its per-layer
features feed the action stream via cross-attention bridges
(``ActionDiT(variant="joint_cross_attn")``), and the non-AR dual flow-matching
loss produces finite video + action terms with finite gradients on both streams.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

_GPU = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="GDN ChunkCausalGDNTriton needs CUDA")


@requires_gpu
def test_gdn_cross_attn_forward_and_loss():
    import sana_wam.model.video_backbone.sana as _s  # noqa: F401 — sys.path + mmcv shim
    from sana_wam.model.cross_attn import DualSystemCrossAttnArchitecture
    from sana_wam.model.video_backbone.sana.adapter import SanaVideoBackbone

    dev, dt = torch.device("cuda"), torch.bfloat16
    vb = SanaVideoBackbone.from_mini_config(
        depth=4, hidden_size=224, num_heads=2, linear_head_dim=112,
        f=3, h=8, w=8, device="cuda", dtype=dt, attn_kernel="gdn", chunk_size=3,
    )
    assert vb.attn_kernel == "gdn"

    arch = DualSystemCrossAttnArchitecture(cfg=None)
    arch.video_backbone = vb
    arch._cross_cfg = {
        "action_dim": 20, "dim": 128, "num_heads": 2, "attn_head_dim": 64,
        "bridge_layers": [0, 1, 2, 3], "text_dim": 64, "ffn_dim": 256,
    }
    arch._device, arch._dtype = dev, dt
    arch.build_action_backbone()
    arch.to(dt).cuda()

    assert arch._bridge_layers == (0, 1, 2, 3)
    assert arch.action_backbone.num_layers == 4

    vb.scheduler.set_timesteps(4, training=True)
    arch.action_backbone.scheduler.set_timesteps(4, training=True)

    B = 1
    inputs = dict(
        input_latents=torch.randn(B, 16, 3, 8, 8, device=dev, dtype=dt),
        actions=torch.randn(B, 6, 20, device=dev, dtype=dt),
        context=torch.randn(B, 8, 64, device=dev, dtype=dt),
        seq_lens=torch.full((B,), 8, dtype=torch.long, device=dev),
    )
    out = arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **inputs)

    assert torch.isfinite(out["loss"]).item()
    assert torch.isfinite(out["loss_video"]).item()
    assert torch.isfinite(out["loss_action"]).item()
    assert float(out["loss_action"]) > 0.0  # action stream actually contributed

    out["loss"].backward()
    # both streams received finite grads
    vb_grads = [p.grad for p in vb.parameters() if p.requires_grad and p.grad is not None]
    ab_grads = [p.grad for p in arch.action_backbone.parameters() if p.requires_grad and p.grad is not None]
    assert vb_grads and all(torch.isfinite(g).all() for g in vb_grads)
    assert ab_grads and all(torch.isfinite(g).all() for g in ab_grads)


@requires_gpu
def test_gdn_cross_attn_video_only_forward():
    """noisy_actions=None → video-only forward returns (video_pred, None)."""
    import sana_wam.model.video_backbone.sana as _s  # noqa: F401
    from sana_wam.model.cross_attn import DualSystemCrossAttnArchitecture
    from sana_wam.model.video_backbone.sana.adapter import SanaVideoBackbone

    dev, dt = torch.device("cuda"), torch.bfloat16
    vb = SanaVideoBackbone.from_mini_config(
        depth=2, hidden_size=224, num_heads=2, linear_head_dim=112,
        f=3, h=8, w=8, device="cuda", dtype=dt, attn_kernel="gdn", chunk_size=3,
    )
    arch = DualSystemCrossAttnArchitecture(cfg=None)
    arch.video_backbone = vb
    arch._cross_cfg = {"action_dim": 20, "dim": 128, "num_heads": 2, "attn_head_dim": 64,
                       "bridge_layers": [0, 1], "text_dim": 64, "ffn_dim": 256}
    arch._device, arch._dtype = dev, dt
    arch.build_action_backbone()
    arch.to(dt).cuda()

    B = 1
    v_pred, a_pred = arch.forward(
        None, None,
        latents=torch.randn(B, 16, 3, 8, 8, device=dev, dtype=dt),
        timestep=torch.tensor([500.0], device=dev),
        context=torch.randn(B, 8, 64, device=dev, dtype=dt),
        seq_lens=torch.full((B,), 8, dtype=torch.long, device=dev),
    )
    assert a_pred is None
    assert v_pred.shape == (B, 16, 3, 8, 8)
