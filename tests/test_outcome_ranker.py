from __future__ import annotations

import hashlib
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from sana_wam.deploy.outcome_ranker import (
    ACTION_DIM,
    ACTION_TOKENS,
    FEATURE_DIM,
    OutcomeActionRanker,
    summary_action_features,
)


def _write_ranker(path: Path, **overrides) -> str:
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int16),
        "feature_mean": np.zeros(FEATURE_DIM, dtype=np.float32),
        "feature_scale": np.ones(FEATURE_DIM, dtype=np.float32),
        "weight": np.linspace(-1.0, 1.0, FEATURE_DIM, dtype=np.float32),
        "action_tokens": np.asarray(ACTION_TOKENS, dtype=np.int16),
        "action_dim": np.asarray(ACTION_DIM, dtype=np.int16),
        "feature_dim": np.asarray(FEATURE_DIM, dtype=np.int16),
    }
    arrays.update(overrides)
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summary_action_features_match_training_order():
    actions = np.arange(ACTION_TOKENS * ACTION_DIM, dtype=np.float32).reshape(
        ACTION_TOKENS, ACTION_DIM
    )
    delta = np.diff(actions, axis=0)
    expected = np.concatenate(
        (
            actions[0],
            actions[-1],
            actions.mean(axis=0),
            actions.std(axis=0),
            actions.min(axis=0),
            actions.max(axis=0),
            actions[-1] - actions[0],
            np.abs(delta).sum(axis=0),
            np.abs(delta).max(axis=0),
        )
    ).astype(np.float64)
    actual = summary_action_features(actions)
    assert actual.shape == (FEATURE_DIM,)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_ranker_load_and_score(tmp_path):
    path = tmp_path / "ranker.npz"
    sha256 = _write_ranker(path)
    ranker = OutcomeActionRanker.load(path, sha256)
    actions = np.linspace(
        -0.5, 0.75, ACTION_TOKENS * ACTION_DIM, dtype=np.float32
    ).reshape(ACTION_TOKENS, ACTION_DIM)
    expected = summary_action_features(actions) @ ranker.weight
    assert ranker.score(actions) == pytest.approx(float(expected))
    assert ranker.identity["sha256"] == sha256
    assert not ranker.weight.flags.writeable


def test_ranker_fails_closed_on_sha_schema_and_arrays(tmp_path):
    path = tmp_path / "ranker.npz"
    sha256 = _write_ranker(path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        OutcomeActionRanker.load(path, "0" * 64)

    sha256 = _write_ranker(path, schema_version=np.asarray(2, dtype=np.int16))
    with pytest.raises(ValueError, match="unsupported ranker schema_version"):
        OutcomeActionRanker.load(path, sha256)

    bad_scale = np.ones(FEATURE_DIM, dtype=np.float32)
    bad_scale[3] = 0
    sha256 = _write_ranker(path, feature_scale=bad_scale)
    with pytest.raises(ValueError, match="strictly positive"):
        OutcomeActionRanker.load(path, sha256)

    bad_weight = np.ones(FEATURE_DIM, dtype=np.float32)
    bad_weight[4] = np.nan
    sha256 = _write_ranker(path, weight=bad_weight)
    with pytest.raises(ValueError, match="non-finite"):
        OutcomeActionRanker.load(path, sha256)


class _FakeCache:
    def __init__(self):
        self.value = 0.0

    def snapshot(self):
        return {"value": self.value}

    def restore_snapshot(self, snapshot):
        self.value = snapshot["value"]


class _RecordingRanker:
    sha256 = "a" * 64

    def __init__(self):
        self.actions = []

    def score(self, actions):
        self.actions.append(actions.copy())
        return float(actions[0, 0])


def test_generation_zero_branch_restores_selected_cache_feedback_and_rng():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    base_seed = 12345
    engine = object.__new__(ARInferenceEngine)
    engine._device = torch.device("cpu")
    engine._cache = _FakeCache()
    engine._step_c = 0
    engine._pending_action_feedback = None
    engine._last_feedback = {"status": "none"}
    engine._feedback_commits = 0
    engine._feedback_fallbacks = 0
    engine._latched_proprio = np.asarray([7.0], dtype=np.float32)
    engine._gen = torch.Generator().manual_seed(base_seed)
    engine._current_noise_seed = base_seed
    engine._generation_zero_rerank_enabled = True
    engine._generation_zero_rerank_candidate_count = 5
    engine._generation_zero_rerank_count = 0
    engine._last_generation_zero_rerank = None
    engine._outcome_ranker = _RecordingRanker()

    def fake_step(self, *_args, **_kwargs):
        value = float(torch.rand((), generator=self._gen))
        normalized = np.full((ACTION_TOKENS, ACTION_DIM), value, dtype=np.float32)
        self._cache.value += value
        self._pending_action_feedback = {
            "predicted_actions_normalized": normalized.copy(),
            "predicted_actions": normalized.copy(),
            "frame_id": 1,
        }
        self._last_feedback = {"status": "candidate", "value": value}
        self._feedback_commits += 1
        self._feedback_fallbacks += 2
        self._latched_proprio[0] = value
        self._step_c += 1
        return torch.from_numpy(normalized).unsqueeze(0)

    engine._step_with_obs_latent = types.MethodType(fake_step, engine)
    telemetry = engine._rerank_generation_zero(
        None, None, None, None, feedback_plan=None
    )

    seeds = [base_seed] + [
        engine.derive_rerank_candidate_seed(base_seed, index)
        for index in range(1, 5)
    ]
    expected_values = [
        float(torch.rand((), generator=torch.Generator().manual_seed(seed)))
        for seed in seeds
    ]
    selected_index = int(np.argmax(expected_values))
    selected_value = expected_values[selected_index]
    assert telemetry["candidate_seeds"] == seeds
    np.testing.assert_allclose(telemetry["candidate_scores"], expected_values)
    assert len(telemetry["candidate_actions_normalized_sha256"]) == 5
    assert all(
        len(value) == 64
        for value in telemetry["candidate_actions_normalized_sha256"]
    )
    assert telemetry["selected_candidate_index"] == selected_index
    assert telemetry["selected_candidate_seed"] == seeds[selected_index]
    assert engine._step_c == 1
    assert engine._cache.value == pytest.approx(selected_value)
    assert engine._pending_action_feedback["predicted_actions_normalized"][0, 0] == pytest.approx(
        selected_value
    )
    assert engine._feedback_commits == 1
    assert engine._feedback_fallbacks == 2
    assert engine._latched_proprio[0] == pytest.approx(selected_value)

    selected_generator = torch.Generator().manual_seed(seeds[selected_index])
    torch.rand((), generator=selected_generator)
    expected_next = torch.rand((), generator=selected_generator)
    actual_next = torch.rand((), generator=engine._gen)
    torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="exactly once"):
        engine._rerank_generation_zero(None, None, None, None, feedback_plan=None)


def test_common_future_noise_is_candidate_independent_after_generation_zero():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    pair_key = "robotwin/adjust_bottle/demo_clean/scene-42"
    expected_seed = ARInferenceEngine.derive_common_future_noise_seed(
        20260722, pair_key, 1
    )
    samples = []
    for candidate_seed in (11, 99):
        engine = object.__new__(ARInferenceEngine)
        engine._device = torch.device("cpu")
        engine._step_c = 0
        engine._current_noise_seed = candidate_seed
        engine._generation_noise_schedule = "common_future"
        engine._common_future_noise_base_seed = 20260722
        engine._noise_pair_key = pair_key
        engine._gen = torch.Generator().manual_seed(candidate_seed)
        engine._last_generation_noise_seed = None

        assert engine._prepare_generation_rng(0) == candidate_seed
        assert float(torch.rand((), generator=engine._gen)) == pytest.approx(
            float(torch.rand((), generator=torch.Generator().manual_seed(candidate_seed)))
        )
        engine._step_c = 1
        assert engine._prepare_generation_rng(1) == expected_seed
        samples.append(float(torch.rand((), generator=engine._gen)))

    assert samples[0] == samples[1]
    assert ARInferenceEngine.derive_common_future_noise_seed(
        20260722, pair_key, 2
    ) != expected_seed
