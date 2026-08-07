"""Closed-loop AR rollout orchestrator.

Drives DualSystemARArchitecture.ar_rollout over a sequence of observation latents
on a mini SANA video + ActionDiT, checking: each step yields a finite action
chunk of the right shape, and the rollout is closed-loop — changing an early
observation changes later action predictions (history flows through the
linear-state cache). CPU-only; gated on third_party/Sana.
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
        action_dim=8, dim=64, ffn_dim=128, num_heads=vb.num_heads, num_layers=vb.num_layers,
        video_dim=vb.dim, bridge_layers=tuple(range(vb.num_layers)), variant="joint_self_attn",
        attn_head_dim=vb.head_dim, text_dim=vb.context_dim, attn_kernel="linear_relu",
    )
    arch = DualSystemARArchitecture(cfg=None)
    arch.video_backbone, arch.action_backbone = vb, ab
    arch._ar_frame_chunk_size = 2
    arch._ar_attn_window = 72
    arch._mot_driver_kwargs = {"attention_mask_mode": "joint", "video_attention_mask_mode": "first_frame_causal", "mot_checkpoint_mixed_attn": False}
    arch.build_mot_driver()
    arch.set_dtype_device(torch.float32, torch.device("cpu"))
    return arch, vb, ab


@requires_sana
def test_ar_rollout_runs_and_is_closed_loop():
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()

    B, fcs = 1, 2
    in_ch = vb._dit.in_channels
    H = W = 8
    L = 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    n_steps = 3
    a_tokens = 2

    g = torch.Generator().manual_seed(1)
    obs_seq = [torch.randn(B, in_ch, fcs, H, W, generator=g) for _ in range(n_steps)]

    actions = arch.ar_rollout(
        obs_seq, context=context, seq_lens=seq_lens, frame_chunk_size=fcs, attn_window=72,
        video_steps=2, action_steps=2, action_tokens_per_chunk=a_tokens, seed=7,
    )

    assert len(actions) == n_steps
    for a in actions:
        assert a.shape == (B, a_tokens, ab.action_dim)
        assert torch.isfinite(a).all()

    # Closed-loop: change the FIRST observation only; later actions must change
    # (frame-0 confirmed state stays in the window and feeds later chunks).
    obs_seq_b = [o.clone() for o in obs_seq]
    obs_seq_b[0] = torch.randn(B, in_ch, fcs, H, W, generator=torch.Generator().manual_seed(99))
    actions_b = arch.ar_rollout(
        obs_seq_b, context=context, seq_lens=seq_lens, frame_chunk_size=fcs, attn_window=72,
        video_steps=2, action_steps=2, action_tokens_per_chunk=a_tokens, seed=7,
    )

    # Step 0 ingests the differing obs -> its action already differs.
    assert not torch.allclose(actions[0], actions_b[0], atol=1e-5)
    # Step 1 also differs: frame-0 confirmed state (from obs[0]) is in-window.
    assert not torch.allclose(actions[1], actions_b[1], atol=1e-5)


@requires_sana
def test_ar_rollout_threads_proprio_end_to_end():
    """F4: with use_proprioception ON, proprio reaches the backbones' cross-attn —
    a different proprio state changes the rolled-out actions, and a missing one
    raises."""
    import torch.nn as nn

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    S = 7
    arch._use_proprioception_context = True
    arch.proprio_dim = S
    arch.proprio_encoder = nn.Linear(S, vb.context_dim)
    arch.set_dtype_device(torch.float32, torch.device("cpu"))

    B, fcs, in_ch, H, W, L = 1, 2, vb._dit.in_channels, 8, 8, 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    g = torch.Generator().manual_seed(4)
    obs = [torch.randn(B, in_ch, fcs, H, W, generator=g) for _ in range(2)]

    kw = dict(context=context, seq_lens=seq_lens, frame_chunk_size=fcs, attn_window=72,
              video_steps=2, action_steps=2, action_tokens_per_chunk=2, seed=3)

    p1 = torch.randn(B, S, generator=torch.Generator().manual_seed(10))
    p2 = torch.randn(B, S, generator=torch.Generator().manual_seed(20))
    a1 = arch.ar_rollout(obs, proprio_states=p1, **kw)
    a2 = arch.ar_rollout(obs, proprio_states=p2, **kw)
    assert torch.isfinite(a1[0]).all()
    # Different proprio -> different actions (proprio flows through cross-attn).
    assert not torch.allclose(a1[0], a2[0], atol=1e-5)

    # Enabled but missing -> raises (parity with training's strict requirement).
    with pytest.raises(ValueError, match="requires `proprio_state`"):
        arch.ar_rollout(obs, proprio_states=None, **kw)


@requires_sana
def test_ar_rollout_window_isolates_distant_past():
    """With window=1, a change at step 0 must NOT affect step 2's action (frame 0
    falls outside the tight sliding window by then)."""
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    B, fcs, in_ch, H, W, L = 1, 2, vb._dit.in_channels, 8, 8, 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    g = torch.Generator().manual_seed(2)
    obs = [torch.randn(B, in_ch, fcs, H, W, generator=g) for _ in range(3)]
    obs_b = [o.clone() for o in obs]
    obs_b[0] = torch.randn(B, in_ch, fcs, H, W, generator=torch.Generator().manual_seed(123))

    kw = dict(context=context, seq_lens=seq_lens, frame_chunk_size=fcs, attn_window=1,
              video_steps=2, action_steps=2, action_tokens_per_chunk=2, seed=5)
    a = arch.ar_rollout(obs, **kw)
    a_b = arch.ar_rollout(obs_b, **kw)
    # window=1: at step 2 (action frame 5) only frames >= 4 are visible; frame 0
    # (obs[0]) is long evicted -> step-2 action identical.
    torch.testing.assert_close(a[2], a_b[2], atol=1e-6, rtol=1e-5)
