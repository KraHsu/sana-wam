"""Phase 3 (forward): DualSystemARArchitecture end-to-end forward smoke.

Wires a mini SANA video backbone + mini ActionDiT + AR MoT driver and runs one
forward over a duplicated ``[noisy ++ clean]`` sequence, checking the noisy-copy
video/action predictions have the right shapes and are finite. CPU-only; gated on
``third_party/Sana`` being importable.
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
def test_ar_forward_shapes_and_finite():
    from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
    from sana_wam.model.architecture import DualSystemARArchitecture
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone

    torch.manual_seed(0)
    vb = SanaVideoBackbone.from_mini_config(depth=2, hidden_size=128, num_heads=4, linear_head_dim=32, h=8, w=8)
    assert vb.attn_kernel == "linear_relu"
    text_dim = vb.context_dim
    num_heads, head_dim = vb.num_heads, vb.head_dim

    ab = ActionDiT(
        action_dim=8,
        dim=64,
        ffn_dim=128,
        num_heads=num_heads,
        num_layers=vb.num_layers,
        video_dim=vb.dim,
        bridge_layers=tuple(range(vb.num_layers)),
        variant="joint_self_attn",
        attn_head_dim=head_dim,
        text_dim=text_dim,
        attn_kernel="linear_relu",
    )

    arch = DualSystemARArchitecture(cfg=None)
    arch.video_backbone = vb
    arch.action_backbone = ab
    arch._mot_driver_kwargs = {
        "attention_mask_mode": "joint",
        "video_attention_mask_mode": "first_frame_causal",
        "mot_checkpoint_mixed_attn": False,
    }
    arch.build_mot_driver()

    B = 2
    frame_chunk_size = 1
    num_chunks = 2
    T = frame_chunk_size * num_chunks  # latent frames per copy
    action_tokens_per_chunk = 2
    Ta = action_tokens_per_chunk * num_chunks
    in_ch = vb._dit.in_channels
    Hl = Wl = 8  # latent spatial (patchified to 4x4)

    noisy_latents = torch.randn(B, in_ch, T, Hl, Wl)
    clean_latents = torch.randn(B, in_ch, T, Hl, Wl)
    noisy_actions = torch.randn(B, Ta, ab.action_dim)
    clean_actions = torch.randn(B, Ta, ab.action_dim)
    L = 4
    context = torch.randn(B, L, text_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)

    v_pred, a_pred = arch.forward(
        noisy_actions,
        None,
        latents=noisy_latents,
        ar_clean_latents=clean_latents,
        ar_clean_actions=clean_actions,
        ar_video_frame_timesteps=torch.full((B, T), 500.0),
        ar_action_token_timesteps=torch.full((B, Ta), 500.0),
        ar_frame_chunk_size=frame_chunk_size,
        ar_attn_window=72,
        context=context,
        seq_lens=seq_lens,
    )

    assert v_pred.shape == (B, in_ch, T, Hl, Wl)
    assert a_pred.shape == (B, Ta, ab.action_dim)
    assert torch.isfinite(v_pred).all()
    assert torch.isfinite(a_pred).all()
