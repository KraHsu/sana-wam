"""Phase 1: per-token (AR) timestep conditioning degenerates to per-sample.

The AR path lets each action token carry its own diffusion timestep so the
duplicated ``[noisy | clean]`` sequence (and different chunks within the noisy
copy) modulate independently. When every token shares the same timestep, the
per-token path must reproduce the original per-sample AdaLN output bit-for-bit —
otherwise the AR variant would silently perturb the pretrained behaviour.
"""

from __future__ import annotations

import torch

from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
from sana_wam.model.base import ActionState


def _make_dit(attn_kernel: str = "linear_relu") -> ActionDiT:
    return ActionDiT(
        action_dim=8,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=64,
        bridge_layers=(0, 1),
        variant="joint_self_attn",
        attn_head_dim=16,
        attn_kernel=attn_kernel,
    )


def _inputs(dit: ActionDiT, *, B: int = 2, T: int = 6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(B, T, dit.action_dim, generator=g)
    context = torch.randn(B, 3, dit.text_dim, generator=g)
    return actions, context


def test_per_token_tmod_matches_per_sample():
    dit = _make_dit().eval()
    B, T = 2, 6
    actions, context = _inputs(dit, B=B, T=T)
    ts = 0.37

    # Per-sample (original) path.
    astate_ps = dit.prepare_state(actions, torch.full((B,), ts), context=context)
    # Per-token path with every token at the same timestep.
    token_ts = torch.full((B, T), ts)
    astate_pt = dit.prepare_state(actions, torch.full((B,), ts), context=context, token_timesteps=token_ts)

    assert astate_ps.payload.t_mod.dim() == 3
    assert astate_pt.payload.t_mod.dim() == 4

    for layer_id in range(dit.num_layers):
        q0, k0, v0, ps0 = dit.pre_attn_at_layer(layer_id, _clone_state(astate_ps))
        q1, k1, v1, ps1 = dit.pre_attn_at_layer(layer_id, _clone_state(astate_pt))
        torch.testing.assert_close(q0, q1, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(k0, k1, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(v0, v1, atol=1e-6, rtol=1e-5)
        # Post-attn must also agree (gate/FFN modulation broadcasts identically).
        attn_out = torch.randn_like(v0)
        out0 = dit.post_attn_at_layer(layer_id, _clone_state(astate_ps), attn_out, ps0).payload.x_action
        out1 = dit.post_attn_at_layer(layer_id, _clone_state(astate_pt), attn_out, ps1).payload.x_action
        torch.testing.assert_close(out0, out1, atol=1e-6, rtol=1e-5)


def test_per_token_tmod_differs_when_timesteps_differ():
    """Sanity: distinct per-token timesteps actually change the modulation."""
    dit = _make_dit().eval()
    B, T = 1, 6
    actions, context = _inputs(dit, B=B, T=T, seed=3)
    token_ts = torch.linspace(0.1, 0.9, T).unsqueeze(0)  # (1, T) all different
    astate_pt = dit.prepare_state(actions, torch.full((B,), 0.5), context=context, token_timesteps=token_ts)
    astate_ps = dit.prepare_state(actions, torch.full((B,), 0.5), context=context)
    q_pt, *_ = dit.pre_attn_at_layer(0, astate_pt)
    q_ps, *_ = dit.pre_attn_at_layer(0, astate_ps)
    assert not torch.allclose(q_pt, q_ps, atol=1e-4)


def test_framed_rope_arange_matches_default():
    """rope_positions=arange(T) reproduces the default 1D-RoPE pre-attn output."""
    dit = _make_dit().eval()
    B, T = 2, 6
    actions, context = _inputs(dit, B=B, T=T, seed=5)
    ts = torch.full((B,), 0.4)

    astate_def = dit.prepare_state(actions, ts, context=context)
    astate_pos = dit.prepare_state(actions, ts, context=context, rope_positions=torch.arange(T))

    for layer_id in range(dit.num_layers):
        q0, k0, v0, _ = dit.pre_attn_at_layer(layer_id, _clone_state(astate_def))
        q1, k1, v1, _ = dit.pre_attn_at_layer(layer_id, _clone_state(astate_pos))
        torch.testing.assert_close(q0, q1, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(k0, k1, atol=1e-6, rtol=1e-5)


def test_framed_rope_changes_phase_for_frame_positions():
    """Frame-aligned positions (e.g. parity 2c+1 repeated per chunk) change Q/K
    relative to plain token-index RoPE — confirming the position actually drives
    the rotary phase."""
    dit = _make_dit().eval()
    B, T = 1, 6
    actions, context = _inputs(dit, B=B, T=T, seed=9)
    ts = torch.full((B,), 0.4)
    # 3 chunks × 2 action tokens/chunk → frame_id parity 1,1,3,3,5,5.
    frame_pos = torch.tensor([1, 1, 3, 3, 5, 5])
    astate_def = dit.prepare_state(actions, ts, context=context)
    astate_fr = dit.prepare_state(actions, ts, context=context, rope_positions=frame_pos)
    q0, k0, _, _ = dit.pre_attn_at_layer(0, astate_def)
    q1, k1, _, _ = dit.pre_attn_at_layer(0, astate_fr)
    assert not torch.allclose(q0, q1, atol=1e-4)
    # Tokens sharing a frame position get the same rotary phase: the two tokens
    # of chunk 0 (positions 1,1) must be rotated identically, so their pre-RoPE
    # equality (same modulation here, distinct content) is preserved structurally.
    # (Sanity: phase is position-driven, not token-index-driven.)


def _clone_state(astate: ActionState) -> ActionState:
    """pre/post_attn mutate ``payload.x_action`` in place — give each call a fresh copy."""
    p = astate.payload
    from sana_wam.model.action_backbone.joint_action_dit import ActionDiTState

    new_payload = ActionDiTState(
        x_action=p.x_action.clone(),
        t_mod=p.t_mod,
        t_embed=p.t_embed,
        action_freqs=p.action_freqs,
        context=p.context,
        context_mask=p.context_mask,
        frame_ids=p.frame_ids,
    )
    return ActionState(action_latents=astate.action_latents, timestep=astate.timestep, payload=new_payload)
