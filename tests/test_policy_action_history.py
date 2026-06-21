"""WAMPolicy tracks executed actions and feeds them back as action_history.

The cross-attn streaming engine (Design B) pins the executed past actions as a
clean action prefix, so WAMPolicy must accumulate the actions it sends to the env
and pass them via conditions['action_history']. Pure plumbing — no GPU/model.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sana_wam.deploy.policy import WAMPolicy


class _FakeEngine:
    """Returns a fixed action chunk and records the last conditions it saw."""

    def __init__(self, chunk):
        self.chunk = chunk
        self.last_conditions = None

    def generate(self, conditions):
        self.last_conditions = conditions
        return {"actions": self.chunk.copy()}

    def reset(self):
        pass


def _policy(engine):
    cfg = SimpleNamespace(
        history_len=10, execute_horizon=None, temporal_ensemble=False, ensemble_decay=0.5
    )
    return WAMPolicy(engine=engine, cfg=cfg, async_config=None)


def test_policy_accumulates_and_feeds_back_executed_actions():
    chunk = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    engine = _FakeEngine(chunk)
    policy = _policy(engine)

    # Greedy: drain the 3-action chunk one per step.
    a0 = policy.predict_action({})
    a1 = policy.predict_action({})
    a2 = policy.predict_action({})
    assert np.allclose(a0, [0, 0]) and np.allclose(a1, [1, 1]) and np.allclose(a2, [2, 2])

    # All three executed actions are now in the history, oldest→newest.
    assert len(policy.action_history) == 3
    assert np.allclose(np.stack(list(policy.action_history)), chunk)

    # The 4th step regenerates and must pass the accumulated executed history.
    policy.predict_action({})
    fed = engine.last_conditions["action_history"]
    assert len(fed) == 3
    assert np.allclose(np.stack(fed), chunk)


def test_policy_reset_clears_action_history():
    engine = _FakeEngine(np.zeros((2, 2), dtype=np.float32))
    policy = _policy(engine)
    policy.predict_action({})
    assert len(policy.action_history) == 1
    policy.reset()
    assert len(policy.action_history) == 0
