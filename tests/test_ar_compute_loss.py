"""DualSystemARArchitecture.compute_loss — finite loss + gradient flow.

Builds a mini SANA video backbone + ActionDiT + AR driver, runs one AR
flow-matching loss step over a tiny batch, and asserts: the loss dict is finite,
and backward produces finite gradients on BOTH the video and action backbones
(i.e. the duplicated-sequence forward is differentiable end-to-end). CPU-only;
gated on third_party/Sana.
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


def _build_arch():
    from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
    from sana_wam.model.architecture import DualSystemARArchitecture
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone

    vb = SanaVideoBackbone.from_mini_config(depth=2, hidden_size=128, num_heads=4, linear_head_dim=32, h=8, w=8)
    ab = ActionDiT(
        action_dim=8,
        dim=64,
        ffn_dim=128,
        num_heads=vb.num_heads,
        num_layers=vb.num_layers,
        video_dim=vb.dim,
        bridge_layers=tuple(range(vb.num_layers)),
        variant="joint_self_attn",
        attn_head_dim=vb.head_dim,
        text_dim=vb.context_dim,
        attn_kernel="linear_relu",
    )
    arch = DualSystemARArchitecture(cfg=None)
    arch.video_backbone = vb
    arch.action_backbone = ab
    arch._ar_frame_chunk_size = 1
    arch._ar_attn_window = 72
    arch._ar_noisy_cond_prob = 0.5
    arch._ar_cond_max_ratio = 0.3
    arch._mot_driver_kwargs = {
        "attention_mask_mode": "joint",
        "video_attention_mask_mode": "first_frame_causal",
        "mot_checkpoint_mixed_attn": False,
    }
    arch.build_mot_driver()
    # The trainer calls this to set the top-level device/dtype authority
    # (compute_loss reads self.device/self.dtype). Mini backbones are CPU/fp32.
    arch.set_dtype_device(torch.float32, torch.device("cpu"))
    vb.scheduler.set_timesteps(num_inference_steps=1000, training=True)
    ab.scheduler.set_timesteps(num_inference_steps=1000, training=True)
    return arch, vb, ab


@requires_sana
def test_ar_compute_loss_finite_and_grads():
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()

    B, T = 2, 2  # 2 latent frames, frame_chunk_size=1 -> 2 chunks
    Ta = 4  # 2 action tokens per chunk
    in_ch = vb._dit.in_channels
    Hl = Wl = 8
    L = 4

    batch = dict(
        input_latents=torch.randn(B, in_ch, T, Hl, Wl),
        actions=torch.randn(B, Ta, ab.action_dim),
        context=torch.randn(B, L, vb.context_dim),
        seq_lens=torch.full((B,), L, dtype=torch.long),
    )

    out = arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)
    assert torch.isfinite(out["loss"]).all()
    assert torch.isfinite(out["loss_video"]).all()
    assert torch.isfinite(out["loss_action"]).all()
    assert out["loss"].item() > 0

    out["loss"].backward()

    def _has_finite_grad(module) -> bool:
        found = False
        for p in module.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    return False
                found = True
        return found

    assert _has_finite_grad(vb._dit), "no finite grad reached the video backbone"
    assert _has_finite_grad(ab), "no finite grad reached the action backbone"


@requires_sana
def test_ar_compute_loss_clean_copy_is_deterministic_at_zero_prob():
    """With noisy_cond_prob=0 the clean copy is exactly t=0 (no cond noise)."""
    torch.manual_seed(1)
    arch, vb, ab = _build_arch()
    arch._ar_noisy_cond_prob = 0.0
    clean = torch.randn(2, vb._dit.in_channels, 2, 8, 8)
    copy, ts = arch._make_clean_copy(clean, vb.scheduler, 2, 2, clean.device, clean.dtype, frames=True)
    torch.testing.assert_close(copy, clean)
    assert torch.count_nonzero(ts) == 0
