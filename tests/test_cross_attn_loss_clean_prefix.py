"""compute_loss variable clean-prefix (growing-history training).

GPU-gated (GDN backbone). Verifies that per-sample ``num_clean_prefix_frames``
makes the first k_b latent frames clean (sigma=0 ⇒ noisy == clean) and excluded
from the video loss, generalizing the frame-0-only TI2V clean-prefix. Builds the
same mini GDN backbone + cross-attn ActionDiT as test_cross_attn_engine.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

_GPU = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="GDN ChunkCausalGDNTriton needs CUDA")


def _build_arch(dev, dt):
    import sana_wam.model.video_backbone.sana as _s  # noqa: F401
    from sana_wam.model.cross_attn import DualSystemCrossAttnArchitecture
    from sana_wam.model.video_backbone.sana.adapter import SanaVideoBackbone

    vb = SanaVideoBackbone.from_mini_config(
        depth=2, hidden_size=224, num_heads=2, linear_head_dim=112,
        f=9, h=8, w=8, device="cuda", dtype=dt, attn_kernel="gdn", chunk_size=3,
    )
    vb._use_first_frame_cond = True  # TI2V clean-prefix path
    arch = DualSystemCrossAttnArchitecture(cfg=None)
    arch.video_backbone = vb
    arch._cross_cfg = {"action_dim": 20, "dim": 128, "num_heads": 2, "attn_head_dim": 64,
                       "bridge_layers": [0, 1], "text_dim": 64, "ffn_dim": 256}
    arch._device, arch._dtype = dev, dt
    arch.build_action_backbone()
    arch.to(dt).cuda()
    # compute_loss reads vb/ab scheduler.timesteps; populate them (training grid).
    vb.scheduler.set_timesteps(4, training=True)
    arch.action_backbone.scheduler.set_timesteps(4, training=True)
    return arch


@requires_gpu
def test_compute_loss_action_clean_prefix_masks_past_actions():
    """num_clean_prefix_actions excludes the observed-past action steps [0, k) from
    the action loss (symmetric with the video clean prefix)."""
    dev, dt = torch.device("cuda"), torch.bfloat16
    arch = _build_arch(dev, dt)

    B, C, T, Hl, Wl = 2, 16, 9, 8, 8
    atok = 8
    clean_video = torch.randn(B, C, T, Hl, Wl, device=dev, dtype=dt)
    actions = torch.randn(B, atok, 20, device=dev, dtype=dt)
    ctx = torch.randn(B, 8, 64, device=dev, dtype=dt)
    seq = torch.full((B,), 8, dtype=torch.long, device=dev)
    # sample 0: no action prefix (k=0); sample 1: first 3 action steps observed (k=3)
    ncpa = torch.tensor([0, 3], dtype=torch.long, device=dev)

    captured = {}
    orig_dual = type(arch)._dual_mse

    def spy_dual(*a, **kw):
        captured["action_is_pad"] = kw.get("action_is_pad")
        return orig_dual(*a, **kw)

    arch._dual_mse = spy_dual

    inputs = {
        "input_latents": clean_video,
        "context": ctx,
        "seq_lens": seq,
        "num_clean_prefix_actions": ncpa,
    }
    out = arch.compute_loss(actions=actions, lambda_video=1.0, lambda_action=1.0, **inputs)
    assert torch.isfinite(out["loss"]).all()

    aip = captured["action_is_pad"]
    assert aip is not None and tuple(aip.shape) == (B, atok)
    # sample 0: nothing masked; sample 1: steps [0,3) masked, [3,atok) kept.
    assert not aip[0].any()
    assert aip[1, :3].all()
    assert not aip[1, 3:].any()


@requires_gpu
def test_compute_loss_without_action_prefix_field_supervises_all_actions():
    dev, dt = torch.device("cuda"), torch.bfloat16
    arch = _build_arch(dev, dt)
    B = 1
    clean_video = torch.randn(B, 16, 9, 8, 8, device=dev, dtype=dt)
    actions = torch.randn(B, 8, 20, device=dev, dtype=dt)
    ctx = torch.randn(B, 8, 64, device=dev, dtype=dt)
    seq = torch.full((B,), 8, dtype=torch.long, device=dev)

    captured = {}
    orig_dual = type(arch)._dual_mse
    arch._dual_mse = lambda *a, **kw: (captured.update(action_is_pad=kw.get("action_is_pad")) or orig_dual(*a, **kw))

    inputs = {"input_latents": clean_video, "context": ctx, "seq_lens": seq}
    arch.compute_loss(actions=actions, lambda_video=1.0, lambda_action=1.0, **inputs)
    # No clean-prefix field ⇒ no action masking (legacy: supervise the whole trajectory).
    assert captured["action_is_pad"] is None


@requires_gpu
def test_compute_loss_per_sample_clean_prefix_masks():
    dev, dt = torch.device("cuda"), torch.bfloat16
    arch = _build_arch(dev, dt)

    B, C, T, Hl, Wl = 2, 16, 9, 8, 8
    atok = 8
    clean_video = torch.randn(B, C, T, Hl, Wl, device=dev, dtype=dt)
    actions = torch.randn(B, atok, 20, device=dev, dtype=dt)
    ctx = torch.randn(B, 8, 64, device=dev, dtype=dt)
    seq = torch.full((B,), 8, dtype=torch.long, device=dev)
    # sample 0: frame-0-only prefix (k=1); sample 1: 2 clean chunks (k=6)
    ncp = torch.tensor([1, 6], dtype=torch.long, device=dev)

    captured = {}
    orig_forward = arch.forward

    def spy_forward(noisy_actions, a_ts, *, latents=None, **kw):
        captured["noisy_video"] = latents.detach().clone()
        return orig_forward(noisy_actions, a_ts, latents=latents, **kw)

    arch.forward = spy_forward

    inputs = {
        "input_latents": clean_video,
        "context": ctx,
        "seq_lens": seq,
        "num_clean_prefix_frames": ncp,
    }
    out = arch.compute_loss(actions=actions, lambda_video=1.0, lambda_action=1.0, **inputs)

    # finite dual loss
    assert torch.isfinite(out["loss"]).all()

    # clean prefix frames: noisy == clean (sigma=0). Compare on the captured noisy clip.
    noisy = captured["noisy_video"].float()
    cv = clean_video.float()
    # sample 0: only frame 0 clean
    assert torch.allclose(noisy[0, :, :1], cv[0, :, :1], atol=1e-2)
    assert not torch.allclose(noisy[0, :, 1:], cv[0, :, 1:], atol=1e-2)
    # sample 1: frames [0,6) clean, [6,T) noised
    assert torch.allclose(noisy[1, :, :6], cv[1, :, :6], atol=1e-2)
    assert not torch.allclose(noisy[1, :, 6:], cv[1, :, 6:], atol=1e-2)


@requires_gpu
def test_compute_loss_without_prefix_field_is_frame0_only():
    dev, dt = torch.device("cuda"), torch.bfloat16
    arch = _build_arch(dev, dt)

    B, C, T, Hl, Wl = 1, 16, 9, 8, 8
    clean_video = torch.randn(B, C, T, Hl, Wl, device=dev, dtype=dt)
    actions = torch.randn(B, 8, 20, device=dev, dtype=dt)
    ctx = torch.randn(B, 8, 64, device=dev, dtype=dt)
    seq = torch.full((B,), 8, dtype=torch.long, device=dev)

    captured = {}
    orig_forward = arch.forward

    def spy_forward(noisy_actions, a_ts, *, latents=None, **kw):
        captured["noisy_video"] = latents.detach().clone()
        return orig_forward(noisy_actions, a_ts, latents=latents, **kw)

    arch.forward = spy_forward
    inputs = {"input_latents": clean_video, "context": ctx, "seq_lens": seq}
    out = arch.compute_loss(actions=actions, lambda_video=1.0, lambda_action=1.0, **inputs)
    assert torch.isfinite(out["loss"]).all()
    noisy = captured["noisy_video"].float()
    cv = clean_video.float()
    # frame 0 clean, rest noised (legacy behavior preserved)
    assert torch.allclose(noisy[0, :, :1], cv[0, :, :1], atol=1e-2)
    assert not torch.allclose(noisy[0, :, 1:], cv[0, :, 1:], atol=1e-2)
