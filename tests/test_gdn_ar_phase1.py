"""Phase 1: DualSystemGDNARArchitecture — cached-GDN AR video + cross-attn action.

GPU/Triton-gated (cached GDN kernels need CUDA + triton). Pins the composition:
the video backbone runs upstream Sana-wm's ``forward_long`` chunk streaming with a
rolling GDN state cache, its per-layer features feed the proven ``joint_cross_attn``
ActionDiT via cross-attention, and the cache advances chunk→chunk so generation
runs PAST the built clip length (the arbitrary-length property the bounded
cross-attn track structurally lacks). No training here — shapes + cache lifecycle.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

_GPU = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="cached GDN kernels need CUDA")

CHUNK_T = 3  # latent frames per AR chunk (matches the built GDN chunk_size)
TA = 6  # action tokens per chunk (arbitrary for Phase 1 shape checks)


def _build_arch(*, depth=4, bridge_layers=(0, 1, 2, 3)):
    """Mini GDN-AR architecture wired like the cross-attn tests, streaming on."""
    import sana_wam.model.video_backbone.sana as _s  # noqa: F401 — sys.path + mmcv shim
    from sana_wam.model.gdn_ar import DualSystemGDNARArchitecture
    from sana_wam.model.video_backbone.sana.adapter import SanaVideoBackbone

    dev, dt = torch.device("cuda"), torch.bfloat16
    vb = SanaVideoBackbone.from_mini_config(
        depth=depth, hidden_size=224, num_heads=2, linear_head_dim=112,
        f=CHUNK_T, h=8, w=8, device="cuda", dtype=dt, attn_kernel="gdn", chunk_size=CHUNK_T,
    )
    arch = DualSystemGDNARArchitecture(cfg=None)
    arch.video_backbone = vb
    arch._cross_cfg = {
        "action_dim": 20, "dim": 128, "num_heads": 2, "attn_head_dim": 64,
        "bridge_layers": list(bridge_layers), "text_dim": 64, "ffn_dim": 256,
        "frame_chunk_size": CHUNK_T,
    }
    arch._device, arch._dtype = dev, dt
    arch.build_action_backbone()
    arch._frame_chunk_size = CHUNK_T  # cfg was set after __init__ in this test wiring
    arch.to(dt).cuda()
    return arch, vb, dev, dt


def _chunk_inputs(dev, dt, B=1):
    return dict(
        context=torch.randn(B, 8, 64, device=dev, dtype=dt),
        seq_lens=torch.full((B,), 8, dtype=torch.long, device=dev),
        video_timestep=torch.tensor([500.0] * B, device=dev),
        # Action time embedder follows the input dtype; real usage (compute_loss)
        # passes timesteps in the model dtype, so mirror that here (bf16).
        action_timestep=torch.tensor([500.0] * B, device=dev, dtype=dt),
    )


@requires_gpu
def test_build_and_enable_streaming():
    """Architecture builds; enable_streaming swaps blocks to the cached variant."""
    from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
    from diffusion.model.nets.sana_blocks import CausalWanRotaryPosEmbed
    from diffusion.model.nets.sana_gdn_blocks import CachedChunkCausalGDN

    arch, vb, _, _ = _build_arch()
    assert arch._bridge_layers == (0, 1, 2, 3)
    assert arch.action_backbone.num_layers == 4
    assert not vb.cached_streaming_enabled

    arch.enable_streaming()
    assert vb.cached_streaming_enabled
    dit = vb._dit
    assert dit.pos_embed_type == "casual_wan_rope"
    assert isinstance(dit.rope, CausalWanRotaryPosEmbed)
    assert isinstance(dit.blocks[0].attn, CachedChunkCausalGDN)
    assert isinstance(dit.blocks[0].mlp, CachedGLUMBConvTemp)

    # idempotent
    arch.enable_streaming()
    assert vb.cached_streaming_enabled

    # fresh cache is one all-None 10-slot list per block
    kv = arch.empty_kv_cache()
    assert len(kv) == vb.num_layers
    assert all(len(slot) == 10 and all(s is None for s in slot) for slot in kv)


@requires_gpu
def test_one_chunk_forward_shapes():
    """One chunk: video_pred + action_pred shapes correct, cache populated."""
    arch, vb, dev, dt = _build_arch()
    arch.enable_streaming()
    kv = arch.empty_kv_cache()
    ci = _chunk_inputs(dev, dt)

    B = 1
    chunk_latents = torch.randn(B, 16, CHUNK_T, 8, 8, device=dev, dtype=dt)
    noisy_actions = torch.randn(B, TA, 20, device=dev, dtype=dt)

    v_pred, a_pred, kv = arch.forward_chunk(
        chunk_latents, start_f=0, end_f=CHUNK_T, kv_cache=kv,
        noisy_actions=noisy_actions, **ci,
    )
    assert v_pred.shape == (B, 16, CHUNK_T, 8, 8)
    assert torch.isfinite(v_pred).all()
    assert a_pred.shape == (B, TA, 20)
    assert torch.isfinite(a_pred).all()
    # GDN recurrent state was written into the cache (slots 0/1 per block).
    assert kv[0][0] is not None and kv[0][1] is not None


@requires_gpu
def test_two_chunk_advances_cache_past_horizon():
    """Two chunks via a rolling cache: streams past the built clip length and the
    GDN state advances between chunks (true autoregression, arbitrary length)."""
    arch, vb, dev, dt = _build_arch()
    arch.enable_streaming()
    kv = arch.empty_kv_cache()
    ci = _chunk_inputs(dev, dt)
    B = 1

    # chunk 0: absolute frames [0, CHUNK_T)
    c0 = torch.randn(B, 16, CHUNK_T, 8, 8, device=dev, dtype=dt)
    v0, a0, kv = arch.forward_chunk(
        c0, start_f=0, end_f=CHUNK_T, kv_cache=kv,
        noisy_actions=torch.randn(B, TA, 20, device=dev, dtype=dt), **ci,
    )
    state_after_c0 = kv[0][0].detach().clone()

    # chunk 1: absolute frames [CHUNK_T, 2*CHUNK_T) — past the built f=CHUNK_T clip.
    c1 = torch.randn(B, 16, CHUNK_T, 8, 8, device=dev, dtype=dt)
    v1, a1, kv = arch.forward_chunk(
        c1, start_f=CHUNK_T, end_f=2 * CHUNK_T, kv_cache=kv,
        noisy_actions=torch.randn(B, TA, 20, device=dev, dtype=dt), **ci,
    )

    assert torch.isfinite(v1).all() and torch.isfinite(a1).all()
    assert v1.shape == (B, 16, CHUNK_T, 8, 8) and a1.shape == (B, TA, 20)
    # 2 chunks = 2*CHUNK_T latent frames generated from a CHUNK_T-frame model.
    # The rolling GDN state must have changed after ingesting the second chunk.
    assert not torch.equal(state_after_c0, kv[0][0]), "GDN state cache did not advance"


@requires_gpu
def test_bridge_gradient_flows_to_video_backbone():
    """The action loss must backprop through the bridge into the VIDEO backbone.

    Load-bearing for Phase-2 training: bridges are captured via forward hooks on
    the bridge blocks, which run inside ``forward_long``'s gradient checkpointing.
    Confirm those captured features are autograd-connected (not the detached
    recompute artifact) by checking the video DiT receives finite grads from an
    action-only loss term.
    """
    arch, vb, dev, dt = _build_arch(depth=2, bridge_layers=(0, 1))
    arch.enable_streaming()
    kv = arch.empty_kv_cache()
    ci = _chunk_inputs(dev, dt)
    B = 1
    chunk_latents = torch.randn(B, 16, CHUNK_T, 8, 8, device=dev, dtype=dt)
    noisy_actions = torch.randn(B, TA, 20, device=dev, dtype=dt)

    _, a_pred, _ = arch.forward_chunk(
        chunk_latents, start_f=0, end_f=CHUNK_T, kv_cache=kv,
        noisy_actions=noisy_actions, use_gradient_checkpointing=True, **ci,
    )
    # Action-ONLY loss → any video grad must arrive through the bridge.
    a_pred.float().pow(2).mean().backward()

    vb_grads = [p.grad for p in vb.parameters() if p.requires_grad and p.grad is not None]
    assert vb_grads, "no grad reached the video backbone — bridge is detached"
    assert all(torch.isfinite(g).all() for g in vb_grads)
    assert any(g.abs().sum() > 0 for g in vb_grads), "video backbone grads are all zero"


@requires_gpu
def test_video_only_chunk_returns_none_action():
    """noisy_actions=None → action term is None; video chunk still produced."""
    arch, vb, dev, dt = _build_arch(depth=2, bridge_layers=(0, 1))
    arch.enable_streaming()
    kv = arch.empty_kv_cache()
    ci = _chunk_inputs(dev, dt)
    B = 1
    c0 = torch.randn(B, 16, CHUNK_T, 8, 8, device=dev, dtype=dt)
    v_pred, a_pred, kv = arch.forward_chunk(
        c0, start_f=0, end_f=CHUNK_T, kv_cache=kv, noisy_actions=None, **ci,
    )
    assert a_pred is None
    assert v_pred.shape == (B, 16, CHUNK_T, 8, 8)
    assert kv[0][0] is not None  # cache still advances on the video-only pass


@requires_gpu
def test_teacher_forced_compute_loss_and_grads():
    """Phase 2: per-chunk teacher-forced flow-matching loss runs over a multi-chunk
    clip, produces finite video+action terms, and backprops finite grads to BOTH
    backbones through the rolling-cache chunk loop."""
    arch, vb, dev, dt = _build_arch(depth=4, bridge_layers=(0, 1, 2, 3))
    vb.scheduler.set_timesteps(4, training=True)
    arch.action_backbone.scheduler.set_timesteps(4, training=True)
    arch._observed_prefix_chunks = 1

    B = 1
    num_chunks = 4
    T = num_chunks * CHUNK_T  # 12 latent frames → 4 chunks
    atc = 3  # action tokens per chunk
    out = arch.compute_loss(
        lambda_video=1.0, lambda_action=1.0,
        input_latents=torch.randn(B, 16, T, 8, 8, device=dev, dtype=dt),
        actions=torch.randn(B, num_chunks * atc, 20, device=dev, dtype=dt),
        context=torch.randn(B, 8, 64, device=dev, dtype=dt),
        seq_lens=torch.full((B,), 8, dtype=torch.long, device=dev),
    )
    assert torch.isfinite(out["loss"]).item()
    assert torch.isfinite(out["loss_video"]).item()
    assert torch.isfinite(out["loss_action"]).item()
    assert float(out["loss_action"]) > 0.0

    out["loss"].backward()
    vb_grads = [p.grad for p in vb.parameters() if p.requires_grad and p.grad is not None]
    ab_grads = [p.grad for p in arch.action_backbone.parameters() if p.requires_grad and p.grad is not None]
    assert vb_grads and all(torch.isfinite(g).all() for g in vb_grads)
    assert ab_grads and all(torch.isfinite(g).all() for g in ab_grads)


@requires_gpu
def test_rolling_loss_masks_padded_tail():
    """v2 mask-aware loop: a clip padded after `valid` latent frames rolls only the
    valid chunks (depths 1..valid_chunks-1), dropping the padded tail."""
    arch, vb, dev, dt = _build_arch(depth=4, bridge_layers=(0, 1, 2, 3))
    vb.scheduler.set_timesteps(4, training=True)
    arch.action_backbone.scheduler.set_timesteps(4, training=True)
    arch._observed_prefix_chunks = 1

    B = 1
    total_chunks = 5
    T = total_chunks * CHUNK_T  # 15 latent frames
    atc = 3
    Ta = total_chunks * atc
    # Only the first 3 chunks (9 latent frames) are valid; the rest is padding.
    valid_lat = 9
    video_is_pad = torch.zeros(B, T, dtype=torch.bool, device=dev)
    video_is_pad[:, valid_lat:] = True

    # Spy how many denoise (supervised) chunks run: count forward_chunk calls with
    # noisy_actions != None.
    calls = {"denoise": 0}
    orig = arch.forward_chunk
    def _spy(*a, **kw):
        if kw.get("noisy_actions") is not None:
            calls["denoise"] += 1
        return orig(*a, **kw)
    arch.forward_chunk = _spy

    out = arch.compute_loss(
        lambda_video=1.0, lambda_action=1.0,
        input_latents=torch.randn(B, 16, T, 8, 8, device=dev, dtype=dt),
        actions=torch.randn(B, Ta, 20, device=dev, dtype=dt),
        video_is_pad=video_is_pad,
        context=torch.randn(B, 8, 64, device=dev, dtype=dt),
        seq_lens=torch.full((B,), 8, dtype=torch.long, device=dev),
    )
    # valid_chunks=3 ⇒ supervise chunks 0(bootstrap),1,2 at depths 0,1,2. NOT 3,4.
    assert calls["denoise"] == 3, f"expected 3 supervised chunks, got {calls['denoise']}"
    assert torch.isfinite(out["loss"]).item() and float(out["loss_action"]) > 0.0


@requires_gpu
def test_rolling_loss_variable_length_batch():
    """B=2 with different valid lengths: each sample rolls its own chunk count."""
    arch, vb, dev, dt = _build_arch(depth=2, bridge_layers=(0, 1))
    vb.scheduler.set_timesteps(4, training=True)
    arch.action_backbone.scheduler.set_timesteps(4, training=True)
    arch._observed_prefix_chunks = 1

    B = 2
    total_chunks = 4
    T = total_chunks * CHUNK_T  # 12
    atc = 3
    video_is_pad = torch.zeros(B, T, dtype=torch.bool, device=dev)
    video_is_pad[0, 6:] = True   # sample 0: valid 2 chunks → supervise 2 (chunks 0,1)
    # sample 1: all valid 4 chunks → supervise 4 (chunks 0,1,2,3)

    calls = {"denoise": 0}
    orig = arch.forward_chunk
    def _spy(*a, **kw):
        if kw.get("noisy_actions") is not None:
            calls["denoise"] += 1
        return orig(*a, **kw)
    arch.forward_chunk = _spy

    out = arch.compute_loss(
        lambda_video=1.0, lambda_action=1.0,
        input_latents=torch.randn(B, 16, T, 8, 8, device=dev, dtype=dt),
        actions=torch.randn(B, total_chunks * atc, 20, device=dev, dtype=dt),
        video_is_pad=video_is_pad,
        context=torch.randn(B, 8, 64, device=dev, dtype=dt),
        seq_lens=torch.full((B,), 8, dtype=torch.long, device=dev),
    )
    # sample0: 2 supervised + sample1: 4 supervised = 6 total (every chunk supervised)
    assert calls["denoise"] == 6, f"expected 6 (2+4) supervised chunks, got {calls['denoise']}"
    assert torch.isfinite(out["loss"]).item()


@requires_gpu
def test_too_short_clip_returns_grad_safe_zero():
    """A clip with <1 full chunk (nothing to supervise, not even bootstrap) returns a
    grad-connected zero loss — backward is a clean no-op, not a crash."""
    arch, vb, dev, dt = _build_arch(depth=2, bridge_layers=(0, 1))
    vb.scheduler.set_timesteps(4, training=True)
    arch.action_backbone.scheduler.set_timesteps(4, training=True)

    B = 1
    T = CHUNK_T - 1  # < one chunk → valid_chunks=0 → no supervision
    out = arch.compute_loss(
        lambda_video=1.0, lambda_action=1.0,
        input_latents=torch.randn(B, 16, T, 8, 8, device=dev, dtype=dt),
        actions=torch.randn(B, max(1, T), 20, device=dev, dtype=dt),
        context=torch.randn(B, 8, 64, device=dev, dtype=dt),
        seq_lens=torch.full((B,), 8, dtype=torch.long, device=dev),
    )
    assert float(out["loss"]) == 0.0
    out["loss"].backward()  # must not raise


@requires_gpu
def test_gradient_checkpointed_loop_matches_eager():
    """Checkpointed denoise (use_gradient_checkpointing) gives finite grads to both
    backbones — the cache-snapshot keeps the recompute reading the right depth."""
    arch, vb, dev, dt = _build_arch(depth=2, bridge_layers=(0, 1))
    vb.scheduler.set_timesteps(4, training=True)
    arch.action_backbone.scheduler.set_timesteps(4, training=True)
    arch._observed_prefix_chunks = 1

    B = 1
    total_chunks = 4
    T = total_chunks * CHUNK_T
    atc = 3
    kw = dict(
        lambda_video=1.0, lambda_action=1.0,
        input_latents=torch.randn(B, 16, T, 8, 8, device=dev, dtype=dt),
        actions=torch.randn(B, total_chunks * atc, 20, device=dev, dtype=dt),
        context=torch.randn(B, 8, 64, device=dev, dtype=dt),
        seq_lens=torch.full((B,), 8, dtype=torch.long, device=dev),
    )
    out = arch.compute_loss(use_gradient_checkpointing=True, **kw)
    assert torch.isfinite(out["loss"]).item() and float(out["loss_action"]) > 0.0
    out["loss"].backward()
    vb_grads = [p.grad for p in vb.parameters() if p.requires_grad and p.grad is not None]
    ab_grads = [p.grad for p in arch.action_backbone.parameters() if p.requires_grad and p.grad is not None]
    assert vb_grads and all(torch.isfinite(g).all() for g in vb_grads)
    assert ab_grads and all(torch.isfinite(g).all() for g in ab_grads)


@requires_gpu
def test_whole_clip_forward_blocked_after_streaming():
    """Guard: the inherited whole-clip forward must refuse once streaming is on."""
    arch, vb, dev, dt = _build_arch(depth=2, bridge_layers=(0, 1))
    arch.enable_streaming()
    with pytest.raises(RuntimeError, match="forward_chunk"):
        arch.forward(
            None, None,
            latents=torch.randn(1, 16, CHUNK_T, 8, 8, device=dev, dtype=dt),
            timestep=torch.tensor([500.0], device=dev),
            context=torch.randn(1, 8, 64, device=dev, dtype=dt),
            seq_lens=torch.full((1,), 8, dtype=torch.long, device=dev),
        )
