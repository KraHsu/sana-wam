"""Gate 1: ARInferenceEngine per-step core ≡ ar_rollout over the same obs sequence.

The deploy engine drives the AR rollout one step per ``generate`` call, holding
the linear-state KV cache + monotonic RNG across calls. This proves that doing so
is numerically identical to a single batch ``ar_rollout`` over the same realized
observation latents — i.e. the engine introduces no train/inference shift beyond
what ``ar_rollout`` already encodes. CPU-only, mini SANA + ActionDiT; gated on
third_party/Sana like the rollout-engine tests.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
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

    return ARInferenceEngine(
        cfg=_engine_cfg(video_steps, action_steps, seed, tokens), architecture=arch
    )


def test_action_tokens_derive_from_explicit_training_horizon() -> None:
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    engine = object.__new__(ARInferenceEngine)
    engine._video_num_frames = 5
    engine._temporal_compression = 4
    engine._fcs = 2
    cfg = OmegaConf.create(
        {
            "inference": {"num_frames": 17},
            "dataloader": {"num_frames": 17, "action_horizon": 28},
        }
    )
    assert engine._resolve_action_tokens_per_chunk(cfg) == 28


@requires_sana
def test_action_scheduler_consumes_explicit_deploy_shift() -> None:
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    arch, _vb, _ab = _build_arch()
    cfg = _engine_cfg(video_steps=2, action_steps=3, seed=0, tokens=2)
    cfg.inference.shift = 3.0
    engine = ARInferenceEngine(cfg=cfg, architecture=arch)
    u = 2.0 / 3.0
    expected = 3.0 * u / (1.0 + 2.0 * u)
    assert engine._action_scheduler_shift == 3.0
    assert engine._a_sigmas[1] == pytest.approx(expected)
    assert engine._action_scheduler_sigmas_sha256 == hashlib.sha256(
        np.asarray(engine._a_sigmas, dtype="<f8").tobytes(order="C")
    ).hexdigest()


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
    eng = _make_engine_with_cadence(
        arch, num_frames=49, video_stride=4, video_num_frames=13
    )

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


def test_history113_ablation_restores_full_training_cadence():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    engine = object.__new__(ARInferenceEngine)
    engine._obs_chunk_mode = "rolling_buffer"
    engine._raw_num_frames = 113
    engine._video_stride = 4
    engine._video_num_frames = 29

    full = engine._build_obs_clip({"obs_history": [{"image": i} for i in range(113)]})
    assert full == list(range(0, 113, 4))

    short = engine._build_obs_clip(
        {"obs_history": [{"image": i} for i in range(103, 113)]}
    )
    assert short.count(103) == 26
    assert short[-3:] == [104, 108, 112]


def test_rolling_context_proprio_matches_video_window_start() -> None:
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    engine = object.__new__(ARInferenceEngine)
    engine.architecture = SimpleNamespace(
        uses_proprioception=True,
        normalize_deploy_proprio=lambda value: np.asarray(value, dtype=np.float32),
    )
    engine._proprio_mode = "per_step"
    engine._raw_num_frames = 113
    engine._device = torch.device("cpu")
    engine._dtype = torch.float32

    history = [
        {"image": index, "state": np.asarray([index], dtype=np.float32)}
        for index in range(141)
    ]
    conditions = {
        "obs_history": history,
        "proprio_state": np.asarray([140], dtype=np.float32),
    }
    current = engine._prep_proprio(conditions["proprio_state"])
    context = engine._prep_window_context_proprio(conditions, fallback=current)
    assert context.item() == 28.0
    assert current.item() == 140.0
    assert engine._recent_frames(conditions)[0] == 28

    short = {
        "obs_history": history[:29],
        "proprio_state": np.asarray([28], dtype=np.float32),
    }
    current_short = engine._prep_proprio(short["proprio_state"])
    context_short = engine._prep_window_context_proprio(
        short, fallback=current_short
    )
    assert context_short.item() == 0.0
    assert current_short.item() == 28.0


def test_context_proprio_rejects_history_tail_mismatch() -> None:
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    engine = object.__new__(ARInferenceEngine)
    engine.architecture = SimpleNamespace(
        uses_proprioception=True,
        normalize_deploy_proprio=lambda value: np.asarray(value, dtype=np.float32),
    )
    engine._proprio_mode = "per_step"
    engine._raw_num_frames = 113
    engine._device = torch.device("cpu")
    engine._dtype = torch.float32
    conditions = {
        "obs_history": [{"image": 0, "state": np.asarray([1], dtype=np.float32)}],
        "proprio_state": np.asarray([2], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="differs from obs_history tail"):
        engine._prep_window_context_proprio(
            conditions, fallback=torch.tensor([[2.0]])
        )


def test_causal_single_chunk_uses_current_proprio_for_both_paths() -> None:
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    engine = object.__new__(ARInferenceEngine)
    engine.architecture = SimpleNamespace(uses_proprioception=True)
    engine._proprio_mode = "per_step"
    engine._reset_cache_each_generation = True
    current = torch.tensor([[9.0]])
    conditions = {
        "obs_history": [
            {"image": index, "state": np.asarray([index], dtype=np.float32)}
            for index in range(29)
        ],
        "proprio_state": np.asarray([28], dtype=np.float32),
    }
    assert engine._prep_window_context_proprio(conditions, fallback=current) is current


def test_fresh_single_chunk_boundary_rebases_cache_but_not_episode_index() -> None:
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    class Cache:
        resets = 0

        def reset(self):
            self.resets += 1

    engine = object.__new__(ARInferenceEngine)
    engine._reset_cache_each_generation = True
    engine._episode_generation_index = 3
    engine._step_c = 1
    engine._cache = Cache()
    engine._pending_action_feedback = {
        "action_tokens": 28,
        "generated_at_policy_step": 56,
    }
    engine._last_feedback = {"status": "predicted"}
    engine._begin_fresh_single_chunk_generation(
        {"executed_steps_since_generate": 28, "policy_step": 84}
    )
    assert engine._cache.resets == 1
    assert engine._step_c == 0
    assert engine._episode_generation_index == 3
    assert engine._pending_action_feedback is None
    assert engine._last_feedback == {
        "status": "fresh_cache",
        "reason": "causal_single_chunk_boundary",
    }


@requires_sana
def test_action_horizon_rope_has_unique_slots_and_shared_noisy_clean_phases() -> None:
    arch, _vb, _ab = _build_arch()
    arch._ar_action_horizon_rope = True
    frame_ids = torch.cat(
        [torch.ones(28, dtype=torch.long), torch.ones(28, dtype=torch.long)]
    )
    positions = arch._action_rope_positions(
        frame_ids, action_tokens_per_chunk=28
    )
    assert positions[:28].tolist() == list(range(28))
    assert positions[28:].tolist() == list(range(28))
    assert torch.equal(positions[:28], positions[28:])
    assert not torch.equal(positions, frame_ids)


@requires_sana
def test_obs_latent_band_auto_leading_at_step0_trailing_after():
    """F3 fix: 'auto' band takes the LEADING latents at step 0 (training chunk 0,
    incl. causal first-frame) and TRAILING after — matching training's rope-band
    assignment. Uses a fake latent tensor where frame i is filled with value i."""
    arch, vb, ab = _build_arch()
    eng = _make_engine_with_cadence(
        arch, num_frames=49, video_stride=4, video_num_frames=13
    )
    eng._obs_latent_band = "auto"
    fcs = eng._fcs
    t_lat = 4
    latents = torch.stack(
        [torch.full((1, 3, 8, 8), float(i)) for i in range(t_lat)], dim=2
    )  # (1,3,4,8,8)

    eng._step_c = 0
    band0 = eng._select_obs_band(latents)
    assert band0.shape[2] == fcs
    assert float(band0[0, 0, 0, 0, 0]) == 0.0 and float(band0[0, 0, -1, 0, 0]) == float(
        fcs - 1
    )  # leading [0,1]

    eng._step_c = 5
    band5 = eng._select_obs_band(latents)
    assert float(band5[0, 0, 0, 0, 0]) == float(t_lat - fcs) and float(
        band5[0, 0, -1, 0, 0]
    ) == float(t_lat - 1)  # trailing [2,3]


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
        obs_seq,
        context=context,
        seq_lens=seq_lens,
        frame_chunk_size=fcs,
        attn_window=72,
        video_steps=2,
        action_steps=2,
        action_tokens_per_chunk=a_tokens,
        seed=7,
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
    first = [
        engine._step_with_obs_latent(o, context, seq_lens, proprio=None).clone()
        for o in obs_seq
    ]
    engine.reset()
    second = [
        engine._step_with_obs_latent(o, context, seq_lens, proprio=None).clone()
        for o in obs_seq
    ]
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@requires_sana
def test_engine_step_matches_ar_rollout_with_proprio():
    """Engine and rollout route window-context and current AdaLN proprio identically."""
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
    context_proprios = [proprios[0], proprios[0]]

    ref = arch.ar_rollout(
        obs_seq,
        context=context,
        seq_lens=seq_lens,
        proprio_states=proprios,
        context_proprio_states=context_proprios,
        frame_chunk_size=fcs,
        attn_window=72,
        video_steps=2,
        action_steps=2,
        action_tokens_per_chunk=2,
        seed=3,
    )

    engine = _make_engine(arch, video_steps=2, action_steps=2, seed=3, tokens=2)
    for c, obs in enumerate(obs_seq):
        a = engine._step_with_obs_latent(
            obs,
            context,
            seq_lens,
            proprio=proprios[c],
            context_proprio=context_proprios[c],
        )
        torch.testing.assert_close(a, ref[c], atol=1e-6, rtol=1e-5)


@requires_sana
def test_engine_matches_ar_rollout_bootstrap_mode():
    """The closed-loop observed-action alignment is independent of video bootstrap."""
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    arch._ar_bootstrap_clean_prefix = True

    B, fcs, in_ch, H, W, L = 1, 2, vb._dit.in_channels, 8, 8, 4
    context = torch.randn(B, L, vb.context_dim)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    obs_seq = [torch.randn(B, in_ch, fcs, H, W, generator=g) for _ in range(3)]

    ref = arch.ar_rollout(
        obs_seq,
        context=context,
        seq_lens=seq_lens,
        frame_chunk_size=fcs,
        attn_window=72,
        video_steps=2,
        action_steps=2,
        action_tokens_per_chunk=2,
        seed=7,
    )
    engine = _make_engine(arch, video_steps=2, action_steps=2, seed=7, tokens=2)
    for c, obs in enumerate(obs_seq):
        a = engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)
        torch.testing.assert_close(a, ref[c], atol=1e-6, rtol=1e-5)

    # Video bootstrap is a training/video-loss choice; closed-loop actions always
    # use the aligned realized observation.
    arch._ar_bootstrap_clean_prefix = False
    ref_steady = arch.ar_rollout(
        obs_seq,
        context=context,
        seq_lens=seq_lens,
        frame_chunk_size=fcs,
        attn_window=72,
        video_steps=2,
        action_steps=2,
        action_tokens_per_chunk=2,
        seed=7,
    )
    for aligned, steady in zip(ref, ref_steady):
        torch.testing.assert_close(aligned, steady, atol=0, rtol=0)


@requires_sana
def test_closed_loop_uses_observed_video_and_contiguous_action_frames():
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    arch._ar_bootstrap_clean_prefix = True
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    obs_seq = [
        torch.randn(1, vb._dit.in_channels, 2, 8, 8, generator=g) for _ in range(3)
    ]

    video_calls = []
    action_calls = []
    original_video = arch._denoise_video_chunk
    original_action = arch._denoise_action_chunk

    def video_spy(*args, **kwargs):
        video_calls.append(kwargs.get("frame_id"))
        return original_video(*args, **kwargs)

    def action_spy(*args, **kwargs):
        action_calls.append(kwargs.get("frame_id"))
        return original_action(*args, **kwargs)

    arch._denoise_video_chunk = video_spy
    arch._denoise_action_chunk = action_spy
    try:
        arch.ar_rollout(
            obs_seq,
            context=context,
            seq_lens=seq_lens,
            frame_chunk_size=2,
            attn_window=72,
            video_steps=2,
            action_steps=2,
            action_tokens_per_chunk=2,
            seed=7,
        )
    finally:
        arch._denoise_video_chunk = original_video
        arch._denoise_action_chunk = original_action

    assert video_calls == []
    assert action_calls == [1, 3, 5]


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
        obs_seq,
        context=context,
        seq_lens=seq_lens,
        proprio_states=proprios,
        frame_chunk_size=fcs,
        attn_window=72,
        video_steps=2,
        action_steps=2,
        action_tokens_per_chunk=2,
        seed=3,
    )
    engine = _make_engine(arch, video_steps=2, action_steps=2, seed=3, tokens=2)
    for c, obs in enumerate(obs_seq):
        a = engine._step_with_obs_latent(obs, context, seq_lens, proprio=proprios[c])
        torch.testing.assert_close(a, ref[c], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "bootstrap,expected_order",
    [
        (True, [("action", 1), ("obs", 2)]),
        (False, [("action", 1), ("obs", 2)]),
    ],
)
@requires_sana
def test_measured_feedback_replaces_previous_action_frame_in_causal_order(
    bootstrap, expected_order
):
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    arch._ar_bootstrap_clean_prefix = bootstrap
    cfg = OmegaConf.create(
        {
            "inference": {
                "video_steps": 2,
                "action_steps": 2,
                "seed": 7,
                "cache_feedback_mode": "measured",
                "cache_feedback_fallback": "error",
                "ar_action_tokens_per_chunk": 2,
                "video_num_frames": 13,
            }
        }
    )
    engine = ARInferenceEngine(cfg=cfg, architecture=arch)
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)

    action_frame = engine._pending_action_feedback["frame_id"]
    old_state = next(
        entry["S"].clone()
        for entry in engine._cache._entries[0]
        if entry["frame_id"] == action_frame
    )
    conditions = {
        "executed_steps_since_generate": 2,
        "achieved_state_history": [
            np.full(ab.action_dim, -0.25, dtype=np.float32),
            np.full(ab.action_dim, 0.75, dtype=np.float32),
        ],
    }
    plan, status = engine._prepare_action_cache_feedback(conditions)
    assert status["status"] == "prepared"

    events = []
    original_ingest = arch._ingest_clean_video
    original_commit = engine._commit_action_cache_feedback

    def ingest_spy(*args, **kwargs):
        events.append(("obs", kwargs["frame_id"]))
        return original_ingest(*args, **kwargs)

    def commit_spy(feedback_plan):
        events.append(("action", feedback_plan["frame_id"]))
        return original_commit(feedback_plan)

    arch._ingest_clean_video = ingest_spy
    engine._commit_action_cache_feedback = commit_spy
    try:
        engine._step_with_obs_latent(
            obs, context, seq_lens, proprio=None, feedback_plan=plan
        )
    finally:
        arch._ingest_clean_video = original_ingest
        engine._commit_action_cache_feedback = original_commit

    assert events[:2] == expected_order
    assert engine._step_c == 2
    assert engine.runtime_info["cache_feedback_commits"] == 1
    assert engine.runtime_info["cache_feedback_fallbacks"] == 0
    for entries in engine._cache._entries:
        assert sum(entry["frame_id"] == action_frame for entry in entries) == 1
    new_state = next(
        entry["S"]
        for entry in engine._cache._entries[0]
        if entry["frame_id"] == action_frame
    )
    assert not torch.allclose(old_state, new_state)


@pytest.mark.parametrize(
    "bootstrap,expected_order",
    [
        (True, [("action", 1), ("obs", 2)]),
        (False, [("action", 1), ("obs", 2)]),
    ],
)
@requires_sana
def test_reencode_predicted_rebuilds_previous_chunk_in_causal_order(
    bootstrap, expected_order
):
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    arch._ar_bootstrap_clean_prefix = bootstrap
    cfg = OmegaConf.create(
        {
            "inference": {
                "video_steps": 2,
                "action_steps": 2,
                "seed": 7,
                "cache_feedback_mode": "reencode_predicted",
                "cache_feedback_fallback": "error",
                "ar_action_tokens_per_chunk": 2,
                "video_num_frames": 13,
            }
        }
    )
    engine = ARInferenceEngine(cfg=cfg, architecture=arch)
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)

    pending = engine._pending_action_feedback
    source_raw = pending["predicted_actions"].copy()
    source_normalized = pending["predicted_actions_normalized"].copy()
    action_frame = pending["frame_id"]
    old_states = [
        next(
            entry["S"].clone() for entry in entries if entry["frame_id"] == action_frame
        )
        for entries in engine._cache._entries
    ]
    plan, status = engine._prepare_action_cache_feedback(
        {"executed_steps_since_generate": 2}
    )
    assert status == {
        "status": "prepared",
        "reason": "previous_predicted_chunk",
        "source_generation_index": 0,
    }

    events = []
    original_ingest = arch._ingest_clean_video
    original_commit = engine._commit_action_cache_feedback

    def ingest_spy(*args, **kwargs):
        events.append(("obs", kwargs["frame_id"]))
        return original_ingest(*args, **kwargs)

    def commit_spy(feedback_plan):
        events.append(("action", feedback_plan["frame_id"]))
        return original_commit(feedback_plan)

    arch._ingest_clean_video = ingest_spy
    engine._commit_action_cache_feedback = commit_spy
    try:
        engine._step_with_obs_latent(
            obs, context, seq_lens, proprio=None, feedback_plan=plan
        )
    finally:
        arch._ingest_clean_video = original_ingest
        engine._commit_action_cache_feedback = original_commit

    assert events[:2] == expected_order
    feedback = engine._last_feedback
    assert feedback["status"] == "reencoded_predicted"
    assert feedback["reason"] == "previous_predicted_chunk"
    assert feedback["source_generation_index"] == 0
    np.testing.assert_array_equal(feedback["actions"], source_raw)
    np.testing.assert_array_equal(feedback["actions_normalized"], source_normalized)
    assert engine.runtime_info["cache_feedback_commits"] == 1
    assert engine.runtime_info["cache_feedback_fallbacks"] == 0
    assert engine.runtime_info["last_cache_feedback"] == {
        "status": "reencoded_predicted",
        "reason": "previous_predicted_chunk",
        "frame_id": action_frame,
        "source_generation_index": 0,
    }

    new_states = [
        next(entry["S"] for entry in entries if entry["frame_id"] == action_frame)
        for entries in engine._cache._entries
    ]
    for old, new in zip(old_states, new_states):
        torch.testing.assert_close(new, old, atol=0, rtol=0)


@requires_sana
def test_reencode_predicted_reuses_direct_normalized_model_output():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    class TrackingNormalizer:
        def __init__(self):
            self.normalize_calls = 0

        def normalize(self, _actions):
            self.normalize_calls += 1
            raise AssertionError(
                "predicted cache feedback must not renormalize raw actions"
            )

        @staticmethod
        def unnormalize(actions):
            return actions * np.float32(3.0) + np.float32(17.0)

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    normalizer = TrackingNormalizer()
    arch.action_normalizer = normalizer
    engine = ARInferenceEngine(
        cfg=OmegaConf.create(
            {
                "inference": {
                    "video_steps": 2,
                    "action_steps": 2,
                    "seed": 7,
                    "cache_feedback_mode": "reencode_predicted",
                    "cache_feedback_fallback": "error",
                    "ar_action_tokens_per_chunk": 2,
                    "video_num_frames": 13,
                }
            }
        ),
        architecture=arch,
    )
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    predicted = engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)
    direct_normalized = predicted.squeeze(0).float().numpy()
    pending = engine._pending_action_feedback
    np.testing.assert_array_equal(
        pending["predicted_actions_normalized"], direct_normalized
    )
    np.testing.assert_array_equal(
        pending["predicted_actions"],
        direct_normalized * np.float32(3.0) + np.float32(17.0),
    )

    plan, _ = engine._prepare_action_cache_feedback(
        {"executed_steps_since_generate": 2}
    )
    assert normalizer.normalize_calls == 0
    np.testing.assert_array_equal(plan["actions_normalized"], direct_normalized)
    assert not np.array_equal(plan["actions"], plan["actions_normalized"])


@pytest.mark.parametrize("seed", [7, None], ids=["private_rng", "ambient_global_rng"])
@pytest.mark.parametrize("feedback_mode", ["measured", "reencode_predicted"])
@requires_sana
def test_successful_feedback_commit_does_not_consume_rng(seed, feedback_mode):
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    arch.eval()
    engine = ARInferenceEngine(
        cfg=OmegaConf.create(
            {
                "inference": {
                    "video_steps": 2,
                    "action_steps": 2,
                    "seed": seed,
                    "cache_feedback_mode": feedback_mode,
                    "cache_feedback_fallback": "error",
                    "ar_action_tokens_per_chunk": 2,
                    "video_num_frames": 13,
                }
            }
        ),
        architecture=arch,
    )
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)
    plan, _ = engine._prepare_action_cache_feedback(
        {
            "executed_steps_since_generate": 2,
            "achieved_state_history": [
                np.zeros(ab.action_dim, dtype=np.float32),
                np.ones(ab.action_dim, dtype=np.float32),
            ],
        }
    )
    rng_before = (
        engine._gen.get_state().clone()
        if engine._gen is not None
        else torch.get_rng_state().clone()
    )

    engine._commit_action_cache_feedback(plan)

    rng_after = (
        engine._gen.get_state() if engine._gen is not None else torch.get_rng_state()
    )
    torch.testing.assert_close(rng_after, rng_before, atol=0, rtol=0)


@requires_sana
def test_reencode_commit_uses_source_generation_context_and_proprio():
    import torch.nn as nn

    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    arch._ar_bootstrap_clean_prefix = True
    proprio_dim = 5
    arch._proprio_per_chunk = True
    arch.proprio_video_embed = nn.Linear(proprio_dim, vb.dim)
    arch.proprio_action_embed = nn.Linear(proprio_dim, ab.dim)
    arch.eval()
    engine = ARInferenceEngine(
        cfg=OmegaConf.create(
            {
                "inference": {
                    "video_steps": 2,
                    "action_steps": 2,
                    "seed": 7,
                    "cache_feedback_mode": "reencode_predicted",
                    "cache_feedback_fallback": "error",
                    "ar_action_tokens_per_chunk": 2,
                    "video_num_frames": 13,
                }
            }
        ),
        architecture=arch,
    )
    source_context = torch.randn(1, 4, vb.context_dim)
    current_context = source_context + 10.0
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    source_proprio = torch.full((1, proprio_dim), -0.5)
    current_proprio = torch.full((1, proprio_dim), 0.75)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    engine._step_with_obs_latent(obs, source_context, seq_lens, proprio=source_proprio)
    pending = engine._pending_action_feedback
    source_step_context = pending["context"]
    source_action_proprio = pending["a_proprio"]
    plan, _ = engine._prepare_action_cache_feedback(
        {"executed_steps_since_generate": 2}
    )
    _, current_action_proprio = arch._rollout_proprio_deltas(current_proprio)

    prepare_calls = []
    original_prepare = ab.prepare_state

    def prepare_spy(*args, **kwargs):
        prepare_calls.append((kwargs.get("context"), kwargs.get("token_proprio_emb")))
        return original_prepare(*args, **kwargs)

    ab.prepare_state = prepare_spy
    try:
        engine._step_with_obs_latent(
            obs,
            current_context,
            seq_lens,
            proprio=current_proprio,
            feedback_plan=plan,
        )
    finally:
        ab.prepare_state = original_prepare

    committed_context, committed_proprio = prepare_calls[0]
    assert committed_context is source_step_context
    torch.testing.assert_close(
        committed_proprio, source_action_proprio[:, None, :], atol=0, rtol=0
    )
    assert not torch.equal(committed_context, current_context)
    assert not torch.equal(committed_proprio, current_action_proprio[:, None, :])


@requires_sana
def test_measured_feedback_strict_fallback_is_pre_mutation():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    cfg = OmegaConf.create(
        {
            "inference": {
                "video_steps": 2,
                "action_steps": 2,
                "seed": 7,
                "cache_feedback_mode": "measured",
                "cache_feedback_fallback": "error",
                "ar_action_tokens_per_chunk": 2,
                "video_num_frames": 13,
            }
        }
    )
    engine = ARInferenceEngine(cfg=cfg, architecture=arch)
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)
    cache_before = engine._cache.info()
    rng_before = engine._gen.get_state().clone()

    with pytest.raises(RuntimeError, match="cadence_mismatch"):
        engine._prepare_action_cache_feedback(
            {
                "executed_steps_since_generate": 1,
                "achieved_state_history": [np.zeros(ab.action_dim, dtype=np.float32)],
            }
        )
    assert engine._cache.info() == cache_before
    torch.testing.assert_close(engine._gen.get_state(), rng_before, atol=0, rtol=0)
    assert engine._step_c == 1
    assert engine.runtime_info["cache_feedback_fallbacks"] == 0

    with pytest.raises(RuntimeError, match="insufficient_history"):
        engine._prepare_action_cache_feedback(
            {
                "executed_steps_since_generate": 2,
                "achieved_state_history": [],
            }
        )
    assert engine.runtime_info["cache_feedback_fallbacks"] == 0

    engine._pending_action_feedback["generated_at_policy_step"] = 0
    with pytest.raises(RuntimeError, match="policy_step_mismatch"):
        engine._prepare_action_cache_feedback(
            {
                "executed_steps_since_generate": 2,
                "policy_step": 1,
                "achieved_state_history": [
                    np.zeros(ab.action_dim, dtype=np.float32),
                    np.ones(ab.action_dim, dtype=np.float32),
                ],
            }
        )

    engine._cache_feedback_fallback = "predicted"
    with pytest.raises(RuntimeError, match="invalid AR feedback cadence"):
        engine._prepare_action_cache_feedback(
            {
                "executed_steps_since_generate": 1,
                "achieved_state_history": [np.zeros(ab.action_dim, dtype=np.float32)],
            }
        )


@requires_sana
def test_generate_failure_does_not_publish_prepared_feedback():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, _ = _build_arch()
    engine = ARInferenceEngine(
        cfg=OmegaConf.create(
            {
                "inference": {
                    "video_steps": 2,
                    "action_steps": 2,
                    "seed": 7,
                    "cache_feedback_mode": "reencode_predicted",
                    "cache_feedback_fallback": "error",
                    "ar_action_tokens_per_chunk": 2,
                    "video_num_frames": 13,
                }
            }
        ),
        architecture=arch,
    )
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)
    previous_feedback = {"status": "sentinel", "reason": "pre_generate"}
    engine._last_feedback = previous_feedback

    def fail_build(_conditions):
        raise RuntimeError("injected observation failure")

    engine._build_obs_chunk = fail_build
    with pytest.raises(RuntimeError, match="observation"):
        engine.generate({"executed_steps_since_generate": 2})

    assert engine._last_feedback is previous_feedback
    assert engine.runtime_info["cache_feedback_commits"] == 0
    assert engine.runtime_info["cache_feedback_fallbacks"] == 0


@requires_sana
def test_generate_failure_rolls_back_prepared_fallback_counter():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, _ = _build_arch()
    engine = ARInferenceEngine(
        cfg=OmegaConf.create(
            {
                "inference": {
                    "video_steps": 2,
                    "action_steps": 2,
                    "seed": 7,
                    "cache_feedback_mode": "measured",
                    "cache_feedback_fallback": "predicted",
                    "ar_action_tokens_per_chunk": 2,
                    "video_num_frames": 13,
                }
            }
        ),
        architecture=arch,
    )
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)

    def fail_build(_conditions):
        raise RuntimeError("injected observation failure")

    engine._build_obs_chunk = fail_build
    with pytest.raises(RuntimeError, match="observation"):
        engine.generate(
            {"executed_steps_since_generate": 2, "achieved_state_history": []}
        )

    assert engine.runtime_info["cache_feedback_fallbacks"] == 0


@pytest.mark.parametrize("seed", [7, None], ids=["private_rng", "ambient_global_rng"])
@pytest.mark.parametrize("feedback_mode", ["measured", "reencode_predicted"])
@requires_sana
def test_episode_start_failure_restores_feedback_mode_state(seed, feedback_mode):
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, _ = _build_arch()
    arch._ar_bootstrap_clean_prefix = True
    engine = ARInferenceEngine(
        cfg=OmegaConf.create(
            {
                "inference": {
                    "video_steps": 2,
                    "action_steps": 2,
                    "seed": seed,
                    "cache_feedback_mode": feedback_mode,
                    "cache_feedback_fallback": "error",
                    "ar_action_tokens_per_chunk": 2,
                    "video_num_frames": 13,
                }
            }
        ),
        architecture=arch,
    )
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    rng_before = (
        engine._gen.get_state().clone()
        if engine._gen is not None
        else torch.get_rng_state().clone()
    )
    cache_before = engine._cache.info()
    previous_feedback = engine._last_feedback
    original_action = arch._denoise_action_chunk

    def fail_action(*args, **kwargs):
        torch.randn(1, generator=kwargs["gen"])
        raise RuntimeError("injected episode-start denoise failure")

    arch._denoise_action_chunk = fail_action
    try:
        with pytest.raises(RuntimeError, match="episode-start"):
            engine._step_with_obs_latent(
                obs, context, seq_lens, proprio=None, feedback_plan=None
            )
    finally:
        arch._denoise_action_chunk = original_action

    assert engine._cache.info() == cache_before
    assert engine._pending_action_feedback is None
    assert engine._step_c == 0
    assert engine._last_feedback is previous_feedback
    assert engine.runtime_info["cache_feedback_commits"] == 0
    assert engine.runtime_info["cache_feedback_fallbacks"] == 0
    rng_after = (
        engine._gen.get_state() if engine._gen is not None else torch.get_rng_state()
    )
    torch.testing.assert_close(rng_after, rng_before, atol=0, rtol=0)


@pytest.mark.parametrize("seed", [7, None], ids=["private_rng", "ambient_global_rng"])
@pytest.mark.parametrize("feedback_mode", ["measured", "reencode_predicted"])
@requires_sana
def test_steady_feedback_commit_failure_restores_pre_step_cache(seed, feedback_mode):
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    arch._ar_bootstrap_clean_prefix = False
    cfg = OmegaConf.create(
        {
            "inference": {
                "video_steps": 2,
                "action_steps": 2,
                "seed": seed,
                "cache_feedback_mode": feedback_mode,
                "cache_feedback_fallback": "error",
                "ar_action_tokens_per_chunk": 2,
                "video_num_frames": 13,
            }
        }
    )
    engine = ARInferenceEngine(cfg=cfg, architecture=arch)
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)
    plan, _ = engine._prepare_action_cache_feedback(
        {
            "executed_steps_since_generate": 2,
            "achieved_state_history": [
                np.zeros(ab.action_dim, dtype=np.float32),
                np.ones(ab.action_dim, dtype=np.float32),
            ],
        }
    )
    before = engine._cache.snapshot()
    original_commit = engine._commit_action_cache_feedback

    def fail_commit(_plan):
        raise RuntimeError("injected commit failure")

    engine._commit_action_cache_feedback = fail_commit
    try:
        with pytest.raises(RuntimeError, match="injected"):
            engine._step_with_obs_latent(
                obs, context, seq_lens, proprio=None, feedback_plan=plan
            )
    finally:
        engine._commit_action_cache_feedback = original_commit

    assert engine._step_c == 1
    for restored, expected in zip(engine._cache._entries, before):
        assert [entry["frame_id"] for entry in restored] == [
            entry["frame_id"] for entry in expected
        ]
        assert [entry["is_pred"] for entry in restored] == [
            entry["is_pred"] for entry in expected
        ]
        for restored_entry, expected_entry in zip(restored, expected):
            torch.testing.assert_close(restored_entry["S"], expected_entry["S"])
            torch.testing.assert_close(restored_entry["z"], expected_entry["z"])

    rng_before = (
        engine._gen.get_state().clone()
        if engine._gen is not None
        else torch.get_rng_state().clone()
    )
    pending_before = engine._pending_action_feedback
    original_action = arch._denoise_action_chunk

    def fail_action(*args, **kwargs):
        torch.randn(1, generator=kwargs["gen"])
        raise RuntimeError("injected denoise failure")

    arch._denoise_action_chunk = fail_action
    try:
        with pytest.raises(RuntimeError, match="denoise"):
            engine._step_with_obs_latent(
                obs, context, seq_lens, proprio=None, feedback_plan=plan
            )
    finally:
        arch._denoise_action_chunk = original_action

    assert engine._step_c == 1
    assert engine._pending_action_feedback is pending_before
    assert engine.runtime_info["cache_feedback_commits"] == 0
    rng_after = (
        engine._gen.get_state() if engine._gen is not None else torch.get_rng_state()
    )
    torch.testing.assert_close(rng_after, rng_before, atol=0, rtol=0)
    for restored, expected in zip(engine._cache._entries, before):
        assert [entry["frame_id"] for entry in restored] == [
            entry["frame_id"] for entry in expected
        ]
        for restored_entry, expected_entry in zip(restored, expected):
            torch.testing.assert_close(restored_entry["S"], expected_entry["S"])
            torch.testing.assert_close(restored_entry["z"], expected_entry["z"])


def test_reencode_predicted_telemetry_uses_distinct_chunk_fields(tmp_path):
    from sana_wam.deploy.telemetry import TelemetryRecorder

    recorder = TelemetryRecorder(
        OmegaConf.create(
            {
                "telemetry": {
                    "enabled": True,
                    "output_dir": str(tmp_path),
                    "include_chunks": True,
                    "flush_every": 1,
                }
            }
        )
    )
    recorder.start_episode({"episode_key": "episode-0"}, {})
    predicted = np.arange(40, dtype=np.float32).reshape(2, 20)
    previous = predicted + np.float32(100.0)
    previous_normalized = predicted / np.float32(7.0)
    recorder.record_step(
        request_id=1,
        state=np.zeros(20, dtype=np.float32),
        action=predicted[0],
        latency_ms=1.0,
        policy_info={
            "generated": True,
            "generation_index": 1,
            "predicted_actions": predicted,
            "predicted_actions_normalized": predicted / np.float32(3.0),
            "cache_feedback": {
                "status": "reencoded_predicted",
                "reason": "previous_predicted_chunk",
                "frame_id": 1,
                "source_generation_index": 0,
                "actions": previous,
                "actions_normalized": previous_normalized,
            },
        },
        runtime_info={"cache_feedback_commits": 1},
    )
    recorder.close()

    step = json.loads((tmp_path / "steps.jsonl").read_text().strip())
    feedback = step["policy"]["cache_feedback"]
    assert feedback == {
        "status": "reencoded_predicted",
        "reason": "previous_predicted_chunk",
        "frame_id": 1,
        "source_generation_index": 0,
    }
    with np.load(tmp_path / step["chunk_npz"], allow_pickle=False) as chunk:
        assert "measured_feedback_actions" not in chunk.files
        assert "measured_feedback_actions_normalized" not in chunk.files
        np.testing.assert_array_equal(chunk["reencoded_predicted_actions"], previous)
        np.testing.assert_array_equal(
            chunk["reencoded_predicted_actions_normalized"], previous_normalized
        )
