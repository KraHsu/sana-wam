"""Gate 1: ARInferenceEngine per-step core ≡ ar_rollout over the same obs sequence.

The deploy engine drives the AR rollout one step per ``generate`` call, holding
the linear-state KV cache + monotonic RNG across calls. This proves that doing so
is numerically identical to a single batch ``ar_rollout`` over the same realized
observation latents — i.e. the engine introduces no train/inference shift beyond
what ``ar_rollout`` already encodes. CPU-only, mini SANA + ActionDiT; gated on
third_party/Sana like the rollout-engine tests.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

# Reuse the mini-arch builder + SANA gate from the rollout-engine test.
from tests.test_ar_rollout_engine import _build_arch, requires_sana


def _engine_cfg(video_steps, action_steps, seed, tokens):
    return OmegaConf.create(
        {
            "inference": {
                "video_steps": video_steps,
                "action_steps": action_steps,
                "seed": seed,
                "ar_action_tokens_per_chunk": tokens,
                "video_num_frames": 13,  # unused by the latent-driven core; satisfies init
            }
        }
    )


def _make_engine(arch, video_steps, action_steps, seed, tokens):
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    return ARInferenceEngine(cfg=_engine_cfg(video_steps, action_steps, seed, tokens), architecture=arch)


def _make_engine_with_cadence(arch, *, num_frames, video_stride, video_num_frames):
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    cfg = OmegaConf.create(
        {
            "inference": {
                "video_steps": 2,
                "action_steps": 2,
                "seed": 0,
                "ar_action_tokens_per_chunk": 2,
                "num_frames": num_frames,
                "video_num_frames": video_num_frames,
            },
            "dataloader": {"video_stride": video_stride},
        }
    )
    return ARInferenceEngine(cfg=cfg, architecture=arch)


@requires_sana
def test_build_obs_clip_matches_training_cadence():
    """The obs clip must be rebuilt from per-step obs_history by subsampling the last
    num_frames frames at video_stride — exactly the dataset's
    range(0, num_frames, video_stride) — so deploy latents match training motion.
    Uses sentinel 'image' values (ints) since _build_obs_clip only slices the list."""
    arch, vb, ab = _build_arch()
    eng = _make_engine_with_cadence(arch, num_frames=49, video_stride=4, video_num_frames=13)

    # 60 per-step frames; engine should take the last 49 and subsample by 4.
    hist = [{"image": i} for i in range(60)]
    clip = eng._build_obs_clip({"obs_history": hist})
    # last 49 = indices 11..59; [::4] over that window -> 11,15,...,59 (13 frames, newest 59 incl.)
    assert clip == list(range(11, 60, 4))
    assert len(clip) == 13
    assert clip[-1] == 59  # newest frame is always included

    # Episode start: fewer frames than the raw window -> left-pad with the oldest.
    short = [{"image": i} for i in range(5)]
    clip_s = eng._build_obs_clip({"obs_history": short})
    assert len(clip_s) == 13
    assert clip_s[-1] == 4  # newest still last
    assert clip_s[0] == 0  # padded with oldest

    # repeat_frame mode: zero-motion clip of the newest frame.
    eng._obs_chunk_mode = "repeat_frame"
    clip_r = eng._build_obs_clip({"obs_history": hist})
    assert clip_r == [59] * 13


@requires_sana
def test_obs_latent_band_auto_leading_at_step0_trailing_after():
    """F3 fix: 'auto' band takes the LEADING latents at step 0 (training chunk 0,
    incl. causal first-frame) and TRAILING after — matching training's rope-band
    assignment. Uses a fake latent tensor where frame i is filled with value i."""
    arch, vb, ab = _build_arch()
    eng = _make_engine_with_cadence(arch, num_frames=49, video_stride=4, video_num_frames=13)
    eng._obs_latent_band = "auto"
    fcs = eng._fcs
    t_lat = 4
    latents = torch.stack([torch.full((1, 3, 8, 8), float(i)) for i in range(t_lat)], dim=2)  # (1,3,4,8,8)

    eng._step_c = 0
    band0 = eng._select_obs_band(latents)
    assert band0.shape[2] == fcs
    assert float(band0[0, 0, 0, 0, 0]) == 0.0 and float(band0[0, 0, -1, 0, 0]) == float(fcs - 1)  # leading [0,1]

    eng._step_c = 5
    band5 = eng._select_obs_band(latents)
    assert float(band5[0, 0, 0, 0, 0]) == float(t_lat - fcs) and float(band5[0, 0, -1, 0, 0]) == float(t_lat - 1)  # trailing [2,3]


@requires_sana
def test_engine_step_matches_ar_rollout():
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()

    B, fcs, in_ch, H, W, L = 1, 2, vb._dit.in_channels, 8, 8, 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    n_steps, a_tokens = 3, 2

    g = torch.Generator().manual_seed(1)
    obs_seq = [torch.randn(B, in_ch, fcs, H, W, generator=g) for _ in range(n_steps)]

    ref = arch.ar_rollout(
        obs_seq, context=context, seq_lens=seq_lens, frame_chunk_size=fcs, attn_window=72,
        video_steps=2, action_steps=2, action_tokens_per_chunk=a_tokens, seed=7,
    )

    engine = _make_engine(arch, video_steps=2, action_steps=2, seed=7, tokens=a_tokens)
    for c, obs in enumerate(obs_seq):
        a = engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)
        assert a.shape == (B, a_tokens, ab.action_dim)
        torch.testing.assert_close(a, ref[c], atol=1e-6, rtol=1e-5)


@requires_sana
def test_engine_reset_is_reproducible():
    """reset() must restore a fresh rollout: a second pass is bit-identical."""
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    B, fcs, in_ch, H, W, L = 1, 2, vb._dit.in_channels, 8, 8, 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    g = torch.Generator().manual_seed(2)
    obs_seq = [torch.randn(B, in_ch, fcs, H, W, generator=g) for _ in range(3)]

    engine = _make_engine(arch, video_steps=2, action_steps=2, seed=5, tokens=2)
    first = [engine._step_with_obs_latent(o, context, seq_lens, proprio=None).clone() for o in obs_seq]
    engine.reset()
    second = [engine._step_with_obs_latent(o, context, seq_lens, proprio=None).clone() for o in obs_seq]
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@requires_sana
def test_engine_step_matches_ar_rollout_with_proprio():
    """F4 path: engine threads proprio identically to ar_rollout (per-step states)."""
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
    obs_seq = [torch.randn(B, in_ch, fcs, H, W, generator=g) for _ in range(2)]
    proprios = [
        torch.randn(B, S, generator=torch.Generator().manual_seed(10)),
        torch.randn(B, S, generator=torch.Generator().manual_seed(11)),
    ]

    ref = arch.ar_rollout(
        obs_seq, context=context, seq_lens=seq_lens, proprio_states=proprios,
        frame_chunk_size=fcs, attn_window=72, video_steps=2, action_steps=2,
        action_tokens_per_chunk=2, seed=3,
    )

    engine = _make_engine(arch, video_steps=2, action_steps=2, seed=3, tokens=2)
    for c, obs in enumerate(obs_seq):
        a = engine._step_with_obs_latent(obs, context, seq_lens, proprio=proprios[c])
        torch.testing.assert_close(a, ref[c], atol=1e-6, rtol=1e-5)


@requires_sana
def test_engine_matches_ar_rollout_bootstrap_mode():
    """F3 bootstrap: with ar_bootstrap_clean_prefix on, each step emits the CURRENT
    chunk's action (frame 2c+1) from the real obs. The engine must stay numerically
    identical to ar_rollout in this mode, and differ from the steady-state mode."""
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    arch._ar_bootstrap_clean_prefix = True

    B, fcs, in_ch, H, W, L = 1, 2, vb._dit.in_channels, 8, 8, 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    obs_seq = [torch.randn(B, in_ch, fcs, H, W, generator=g) for _ in range(3)]

    ref = arch.ar_rollout(
        obs_seq, context=context, seq_lens=seq_lens, frame_chunk_size=fcs, attn_window=72,
        video_steps=2, action_steps=2, action_tokens_per_chunk=2, seed=7,
    )
    engine = _make_engine(arch, video_steps=2, action_steps=2, seed=7, tokens=2)
    for c, obs in enumerate(obs_seq):
        a = engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)
        torch.testing.assert_close(a, ref[c], atol=1e-6, rtol=1e-5)

    # Bootstrap mode (act on current real obs) differs from steady-state (act on predicted video).
    arch._ar_bootstrap_clean_prefix = False
    ref_steady = arch.ar_rollout(
        obs_seq, context=context, seq_lens=seq_lens, frame_chunk_size=fcs, attn_window=72,
        video_steps=2, action_steps=2, action_tokens_per_chunk=2, seed=7,
    )
    assert not torch.allclose(ref[0], ref_steady[0], atol=1e-5)


@requires_sana
def test_engine_matches_ar_rollout_with_per_chunk_proprio():
    """F4 deploy: with per-chunk proprio AdaLN active (non-zero encoders), the engine's
    per-step core stays numerically identical to ar_rollout — the deploy path threads
    the same proprio deltas as the rollout."""
    import torch.nn as nn

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    S = 7
    arch._use_proprioception_context = True
    arch.proprio_dim = S
    arch.context_dim = vb.context_dim
    arch.proprio_encoder = nn.Linear(S, vb.context_dim)
    arch._proprio_per_chunk = True
    arch.proprio_video_embed = None
    arch.proprio_action_embed = None
    arch._maybe_init_per_chunk_proprio()
    for m in (arch.proprio_video_embed, arch.proprio_action_embed):
        nn.init.normal_(m.weight, std=0.3)
        nn.init.normal_(m.bias, std=0.3)
    arch.set_dtype_device(torch.float32, torch.device("cpu"))

    B, fcs, in_ch, H, W, L = 1, 2, vb._dit.in_channels, 8, 8, 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    g = torch.Generator().manual_seed(4)
    obs_seq = [torch.randn(B, in_ch, fcs, H, W, generator=g) for _ in range(2)]
    proprios = [
        torch.randn(B, S, generator=torch.Generator().manual_seed(10)),
        torch.randn(B, S, generator=torch.Generator().manual_seed(11)),
    ]

    ref = arch.ar_rollout(
        obs_seq, context=context, seq_lens=seq_lens, proprio_states=proprios,
        frame_chunk_size=fcs, attn_window=72, video_steps=2, action_steps=2,
        action_tokens_per_chunk=2, seed=3,
    )
    engine = _make_engine(arch, video_steps=2, action_steps=2, seed=3, tokens=2)
    for c, obs in enumerate(obs_seq):
        a = engine._step_with_obs_latent(obs, context, seq_lens, proprio=proprios[c])
        torch.testing.assert_close(a, ref[c], atol=1e-6, rtol=1e-5)
