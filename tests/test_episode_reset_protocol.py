from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from benchmarks.utils import client
from sana_wam.deploy.policy_server import PolicyServer, ResetConflictError


class _ResetEngine:
    def __init__(self):
        self.reset_contexts = []
        self._seed = None

    def generate(self, conditions):
        return {"actions": np.zeros((1, 2), dtype=np.float32)}

    def reset(self, episode_context=None):
        context = dict(episode_context or {})
        self.reset_contexts.append(context)
        self._seed = context.get("model_noise_seed")

    @property
    def runtime_info(self):
        return {
            "episode_noise_mode": "paired",
            "current_model_noise_seed": self._seed,
        }


def _server(engine=None):
    cfg = OmegaConf.create(
        {
            "policy": {
                "history_len": 10,
                "execute_horizon": None,
                "temporal_ensemble": False,
            },
            "telemetry": {"enabled": False},
        }
    )
    return PolicyServer(engine or _ResetEngine(), cfg)


def test_named_reset_is_idempotent_and_conflicting_seed_is_rejected():
    engine = _ResetEngine()
    server = _server(engine)
    payload = {"episode_key": "adjust_bottle/100001", "model_noise_seed": 17}

    first = server.reset(payload)
    duplicate = server.reset(payload)
    assert first["duplicate"] is False
    assert duplicate["duplicate"] is True
    assert len(engine.reset_contexts) == 1

    with pytest.raises(ResetConflictError, match="already active"):
        server.reset({**payload, "model_noise_seed": 18})
    assert len(engine.reset_contexts) == 1

    with pytest.raises(ResetConflictError, match="noise_pair_key"):
        server.reset({**payload, "noise_pair_key": "different-scene"})


def test_legacy_unnamed_reset_remains_non_idempotent():
    engine = _ResetEngine()
    server = _server(engine)
    server.reset({})
    server.reset({})
    assert len(engine.reset_contexts) == 2


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"episode_key": ""}, "must not be empty"),
        ({"model_noise_seed": True}, "must be an integer"),
        ({"model_noise_seed": 1.5}, "must be an integer"),
        ({"model_noise_seed": -1}, "must be in"),
        ({"model_noise_seed": 2**63}, "must be in"),
        ({"episode_key": "a", "episode_id": "b"}, "disagree"),
    ],
)
def test_invalid_reset_payload_is_rejected(payload, match):
    with pytest.raises(ValueError, match=match):
        PolicyServer._normalize_reset_context(payload)


def test_reset_aliases_are_normalized():
    assert PolicyServer._normalize_reset_context(
        {"episode_id": "scene-1", "model_seed": 0}
    ) == {"episode_key": "scene-1", "model_noise_seed": 0}


def test_empty_reset_body_keeps_legacy_behavior():
    assert PolicyServer._parse_reset_body("") == {}
    assert PolicyServer._parse_reset_body("  \n") == {}
    assert PolicyServer._parse_reset_body('{"episode_key":"scene-1"}') == {
        "episode_key": "scene-1"
    }


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_client_reset_does_not_coerce_invalid_seed(seed):
    with pytest.raises(ValueError, match="must be an integer"):
        client.reset("http://unused", model_noise_seed=seed)
