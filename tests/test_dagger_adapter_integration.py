from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from benchmarks.robotwin import sana_wam2robotwin_interface as adapter


def _observation() -> dict:
    camera = np.zeros((4, 5, 3), dtype=np.uint8)
    return {
        "observation": {
            "head_camera": {"rgb": camera},
            "left_camera": {"rgb": camera},
            "right_camera": {"rgb": camera},
        },
        "endpose": {
            "left_endpose": np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            "left_gripper": 1.0,
            "right_endpose": np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            "right_gripper": 1.0,
        },
    }


class _Env:
    task_name = "adjust_bottle"
    step_lim = 400
    eval_success = False

    def __init__(self, step=112):
        self.take_action_cnt = step
        self.actions = []

    def get_instruction(self):
        return "pick up the bottle"

    def take_action(self, action, action_type):
        self.actions.append((np.asarray(action), action_type))


class _Model:
    _send_state = True
    _action_type = "ee"
    _telemetry = SimpleNamespace(enabled=False)

    def __init__(self, dagger):
        self._dagger = dagger
        self._last_interaction = {}

    def step(self, example, step=0):
        del example
        self._last_interaction = {
            "camera_jpegs": {name: b"jpeg" for name in ("head", "left", "right")},
            "response": {
                "policy": {
                    "policy_step": step,
                    "generation_index": step // 28,
                    "chunk_offset": step % 28,
                    "buffer_remaining": 27 - step % 28,
                    "generated": step % 28 == 0,
                    "chunk_len": 28,
                }
            },
        }
        return np.zeros(20, dtype=np.float32)


def test_live_takeover_withholds_anchor_learner_action():
    calls = []

    class Recorder:
        enabled = True

        def process_step(self, **kwargs):
            calls.append(kwargs)
            kwargs["task_env"].eval_success = True
            return SimpleNamespace(consumed=True, success=True)

    env = _Env(step=112)
    model = _Model(Recorder())

    adapter.eval(env, model, _observation())

    assert env.actions == []
    assert env.eval_success is True
    assert len(calls) == 1
    assert calls[0]["env_step"] == 112
    assert calls[0]["policy_info"]["generated"] is True
    assert np.asarray(calls[0]["pre_state"]).shape == (20,)


def test_disabled_takeover_preserves_normal_eef_execution():
    env = _Env(step=7)
    model = _Model(SimpleNamespace(enabled=False))

    adapter.eval(env, model, _observation())

    assert len(env.actions) == 1
    action, action_type = env.actions[0]
    assert action.shape == (16,)
    assert action_type == "ee"
