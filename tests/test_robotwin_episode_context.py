from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.robotwin import sana_wam2robotwin_interface as adapter
from benchmarks.robotwin.eval_policy_wrapper import (
    _install_env_trace_hooks,
    _load_prompt_manifest,
    _prompt_sha256,
)


def _manifest_episode(
    *,
    ordinal=0,
    episode_index=None,
    task_name="adjust_bottle",
    task_config="demo_clean",
    environment_seed=100123,
    prompt="place the green bottle upright",
):
    if episode_index is None:
        episode_index = ordinal
    return {
        "ordinal": ordinal,
        "episode_index": episode_index,
        "task_name": task_name,
        "task_config": task_config,
        "environment_seed": environment_seed,
        "noise_pair_key": (
            f"robotwin/{task_name}/{task_config}/scene-{environment_seed}"
        ),
        "prompt": prompt,
        "prompt_sha256": _prompt_sha256(prompt),
    }


def _write_prompt_manifest(tmp_path, episodes):
    path = tmp_path / "prompt_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "robotwin_prompt_replay",
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )
    return path


class _InstructionEnv:
    task_name = "adjust_bottle"

    def setup_demo(self, **kwargs):
        return kwargs

    def set_instruction(self, instruction=None):
        self.instruction = instruction
        return instruction

    def play_once(self):
        return None


def _trace_instruction_env(prompt_manifest, reset_model=lambda model: None):
    env = _InstructionEnv()
    module = SimpleNamespace(
        class_decorator=lambda task_name: env,
        eval_function_decorator=lambda policy_name, name: reset_model,
        UnStableError=RuntimeError,
    )
    _install_env_trace_hooks(module, prompt_manifest=prompt_manifest)
    return module, module.class_decorator("adjust_bottle")


def test_valid_prompt_manifest_replaces_robotwin_instruction(tmp_path):
    expected_prompt = "move the bottle beside the plate"
    path = _write_prompt_manifest(
        tmp_path,
        [_manifest_episode(prompt=expected_prompt)],
    )
    manifest = _load_prompt_manifest(str(path))
    _, env = _trace_instruction_env(manifest)

    env.setup_demo(
        seed=100123,
        now_ep_num=0,
        task_name="adjust_bottle",
        task_config="demo_clean",
    )
    selected = env.set_instruction(instruction="an unseeded RoboTwin instruction")

    assert selected == expected_prompt
    assert env.instruction == expected_prompt


def test_prompt_manifest_missing_accepted_scene_is_fatal(tmp_path):
    path = _write_prompt_manifest(tmp_path, [_manifest_episode()])
    manifest = _load_prompt_manifest(str(path))
    _, env = _trace_instruction_env(manifest)
    env.setup_demo(
        seed=100124,
        now_ep_num=0,
        task_name="adjust_bottle",
        task_config="demo_clean",
    )

    with pytest.raises(RuntimeError, match="no entry for accepted scene"):
        env.set_instruction(instruction="upstream instruction")


def test_prompt_manifest_rejects_bad_prompt_hash(tmp_path):
    episode = _manifest_episode()
    episode["prompt_sha256"] = "0" * 64
    path = _write_prompt_manifest(tmp_path, [episode])

    with pytest.raises(ValueError, match="prompt SHA-256 mismatch"):
        _load_prompt_manifest(str(path))


@pytest.mark.parametrize("payload", [None, []])
def test_prompt_manifest_rejects_non_object_root(tmp_path, payload):
    path = tmp_path / "prompt_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        _load_prompt_manifest(str(path))


def test_prompt_manifest_rejects_duplicate_ordinal(tmp_path):
    path = _write_prompt_manifest(
        tmp_path,
        [
            _manifest_episode(ordinal=0, environment_seed=100123),
            _manifest_episode(ordinal=0, environment_seed=100124),
        ],
    )

    with pytest.raises(ValueError, match="non-contiguous prompt-manifest ordinal"):
        _load_prompt_manifest(str(path))


def test_prompt_manifest_rejects_nonincreasing_environment_seeds(tmp_path):
    path = _write_prompt_manifest(
        tmp_path,
        [
            _manifest_episode(ordinal=0, environment_seed=100124),
            _manifest_episode(ordinal=1, environment_seed=100123),
        ],
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        _load_prompt_manifest(str(path))


def test_prompt_manifest_rejects_episode_index_ordinal_mismatch(tmp_path):
    path = _write_prompt_manifest(
        tmp_path,
        [_manifest_episode(ordinal=0, episode_index=1)],
    )

    with pytest.raises(ValueError, match="prompt-manifest episode_index mismatch"):
        _load_prompt_manifest(str(path))


def test_prompt_manifest_rejects_runtime_episode_ordinal_mismatch(tmp_path):
    path = _write_prompt_manifest(tmp_path, [_manifest_episode()])
    manifest = _load_prompt_manifest(str(path))
    _, env = _trace_instruction_env(manifest)
    env.setup_demo(
        seed=100123,
        now_ep_num=1,
        task_name="adjust_bottle",
        task_config="demo_clean",
    )

    with pytest.raises(RuntimeError, match="accepted-scene order mismatch"):
        env.set_instruction(instruction="upstream instruction")


def test_prompt_manifest_metadata_reaches_model_reset(tmp_path):
    expected_prompt = "move the bottle beside the plate"
    path = _write_prompt_manifest(
        tmp_path,
        [_manifest_episode(prompt=expected_prompt)],
    )
    manifest = _load_prompt_manifest(str(path))

    def reset_model(model):
        model.context_seen_by_reset = dict(model.context)
        model.reset_calls += 1

    module, env = _trace_instruction_env(manifest, reset_model=reset_model)
    env.setup_demo(
        seed=100123,
        now_ep_num=0,
        task_name="adjust_bottle",
        task_config="demo_clean",
    )
    env.set_instruction(instruction="an unseeded RoboTwin instruction")

    class Model:
        reset_calls = 0
        context = None

        def set_episode_context(self, **context):
            self.context = context

    model = Model()
    module.eval_function_decorator("policy", "reset_model")(model)

    assert model.reset_calls == 1
    assert model.context_seen_by_reset == {
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "environment_seed": 100123,
        "episode_index": 0,
        "prompt": expected_prompt,
        "prompt_sha256": _prompt_sha256(expected_prompt),
        "prompt_source": "manifest",
        "prompt_manifest_ordinal": 0,
        "prompt_manifest_path": str(path),
        "prompt_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_wrapper_passes_exact_setup_seed_to_model_reset():
    class Env:
        task_name = "adjust_bottle"

        def setup_demo(self, **kwargs):
            return kwargs

        def play_once(self):
            return None

    env = Env()

    def reset_model(model):
        model.reset_calls += 1

    module = SimpleNamespace(
        class_decorator=lambda task_name: env,
        eval_function_decorator=lambda policy_name, name: reset_model,
        UnStableError=RuntimeError,
    )
    _install_env_trace_hooks(module)
    traced_env = module.class_decorator("adjust_bottle")
    traced_env.setup_demo(
        seed=100123,
        now_ep_num=7,
        task_name="adjust_bottle",
        task_config="demo_clean",
    )

    class Model:
        reset_calls = 0
        context = None

        def set_episode_context(self, **context):
            self.context = context

    model = Model()
    module.eval_function_decorator("policy", "reset_model")(model)
    assert model.reset_calls == 1
    assert model.context == {
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "environment_seed": 100123,
        "episode_index": 7,
    }


def test_first_prompt_after_boundary_does_not_send_second_reset(monkeypatch):
    resets = []
    monkeypatch.setattr(
        adapter.client,
        "reset",
        lambda server, **kwargs: resets.append(kwargs) or {"status": "ok"},
    )
    monkeypatch.setattr(adapter.client, "encode_numpy_b64", lambda image: "jpeg")
    monkeypatch.setattr(adapter.client, "build_payload", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        adapter.client,
        "post",
        lambda *args, **kwargs: {"action": [0.0, 0.0], "step": 1, "latency_ms": 1.0},
    )

    model = adapter.ModelClient.__new__(adapter.ModelClient)
    model._episode = -1
    model._step = 0
    model._debug = False
    model._debug_dir = "unused"
    model._episode_context = {
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "environment_seed": 100001,
        "episode_index": 0,
    }
    model._episode_noise_mode = "ambient"
    model._episode_key = None
    model._noise_pair_key = None
    model._prompt_revision = 0
    model._awaiting_first_prompt = False
    model._reset_session_id = "test-session"
    model._pending_reset = None
    model._task_description = ""
    model._server = "http://server"
    model._send_state = True
    model._state_dim = 2
    model._request_timeout = 1
    model._action_indices = None
    model._telemetry = SimpleNamespace(enabled=False, start_episode=lambda *args: None)

    model.reset(task_description="")
    model.step(
        {
            "cams": {
                "head": np.zeros((2, 2, 3), dtype=np.uint8),
                "left": None,
                "right": None,
            },
            "lang": "adjust the bottle",
            "state": np.zeros(2, dtype=np.float32),
        }
    )
    assert len(resets) == 1
    assert model._task_description == "adjust the bottle"


def test_strict_manifest_prompt_change_is_fatal_without_second_reset(monkeypatch):
    resets = []
    posts = []
    monkeypatch.setattr(
        adapter.client,
        "reset",
        lambda server, **kwargs: resets.append(kwargs) or {"status": "ok"},
    )
    monkeypatch.setattr(adapter.client, "encode_numpy_b64", lambda image: "jpeg")
    monkeypatch.setattr(adapter.client, "build_payload", lambda **kwargs: kwargs)

    def post(*args, **kwargs):
        posts.append((args, kwargs))
        return {"action": [0.0, 0.0], "step": 1, "latency_ms": 1.0}

    monkeypatch.setattr(adapter.client, "post", post)

    expected_prompt = "adjust the bottle"
    model = adapter.ModelClient.__new__(adapter.ModelClient)
    model._episode = -1
    model._step = 0
    model._debug = False
    model._debug_dir = "unused"
    model._episode_context = {
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "environment_seed": 100001,
        "episode_index": 0,
        "prompt": expected_prompt,
        "prompt_sha256": _prompt_sha256(expected_prompt),
        "prompt_source": "manifest",
        "prompt_manifest_ordinal": 0,
        "prompt_manifest_path": "/tmp/prompt_manifest.json",
        "prompt_manifest_sha256": "a" * 64,
    }
    model._episode_noise_mode = "paired"
    model._episode_key = None
    model._noise_pair_key = None
    model._prompt_revision = 0
    model._awaiting_first_prompt = False
    model._reset_session_id = "test-session"
    model._pending_reset = None
    model._task_description = ""
    model._server = "http://server"
    model._send_state = True
    model._state_dim = 2
    model._request_timeout = 1
    model._action_indices = None
    model._telemetry = SimpleNamespace(enabled=False, start_episode=lambda *args: None)

    observation = {
        "cams": {
            "head": np.zeros((2, 2, 3), dtype=np.uint8),
            "left": None,
            "right": None,
        },
        "lang": expected_prompt,
        "state": np.zeros(2, dtype=np.float32),
    }
    model.reset(task_description="")
    model.step(observation)

    changed_observation = dict(observation, lang="a different instruction")
    with pytest.raises(RuntimeError, match="strict manifest replay"):
        model.step(changed_observation)

    assert len(resets) == 1
    assert len(posts) == 1
    assert model._prompt_revision == 0
    assert "/prompt-" not in model._episode_key


def test_boundary_reset_retry_reuses_pending_episode_key(monkeypatch):
    calls = []

    def flaky_reset(server, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("lost response")
        return {"status": "ok", "duplicate": True}

    monkeypatch.setattr(adapter.client, "reset", flaky_reset)
    model = adapter.ModelClient.__new__(adapter.ModelClient)
    model._episode = -1
    model._step = 9
    model._debug = False
    model._episode_context = {
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "environment_seed": 100001,
        "episode_index": 0,
    }
    model._episode_noise_mode = "paired"
    model._episode_key = None
    model._noise_pair_key = None
    model._prompt_revision = 0
    model._awaiting_first_prompt = False
    model._reset_session_id = "test-session"
    model._pending_reset = None
    model._task_description = "old"
    model._server = "http://server"
    model._telemetry = SimpleNamespace(start_episode=lambda *args: None)

    try:
        model.reset("")
    except TimeoutError:
        pass
    else:
        raise AssertionError("first reset should simulate a lost response")
    assert model._episode == -1
    assert model._step == 9

    model.reset("")
    assert model._episode == 0
    assert model._step == 0
    assert calls[0]["episode_key"] == calls[1]["episode_key"]
    assert calls[0]["metadata"]["noise_pair_key"] == (
        "robotwin/adjust_bottle/demo_clean/scene-100001"
    )
