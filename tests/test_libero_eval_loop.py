from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from benchmarks.libero import eval_policy


class FakePolicy:
    def __init__(self, *, fail_on_predict: bool = False) -> None:
        self.fail_on_predict = fail_on_predict
        self.request_count = 0
        self.reset_calls: list[dict] = []
        self.predict_observations: list[dict] = []

    def reset_episode(self, **kwargs) -> str:
        self.reset_calls.append(dict(kwargs))
        self.request_count = 0
        return (
            f"libero/run-fake/{kwargs['suite']}/task-{kwargs['task_id']}/"
            f"trial-{kwargs['trial_index']}/init-{kwargs['init_state_index']}"
        )

    def predict(self, observation, prompt) -> np.ndarray:
        if self.fail_on_predict:
            raise RuntimeError("synthetic policy failure")
        self.predict_observations.append(dict(observation))
        self.request_count += 1
        return np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


class FakeEnv:
    def __init__(self, success_after_steps: int = 8) -> None:
        self.success_after_steps = success_after_steps
        self.step_count = 0
        self.actions: list[np.ndarray] = []
        self.reset_count = 0
        self.init_states: list[object] = []
        self.seeds: list[int] = []
        self.closed = False

    def seed(self, value: int) -> None:
        self.seeds.append(value)

    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        self.actions = []
        return {"step": self.step_count}

    def set_init_state(self, init_state):
        self.init_states.append(init_state)
        return {"step": self.step_count, "init_state": init_state}

    def step(self, action):
        action_array = np.asarray(action, dtype=np.float32)
        assert action_array.shape == (7,)
        self.actions.append(action_array.copy())
        self.step_count += 1
        success = self.step_count >= self.success_after_steps
        return {"step": self.step_count}, 0.0, success, {}

    def check_success(self) -> bool:
        return self.step_count >= self.success_after_steps

    def close(self) -> None:
        self.closed = True


class FakeBenchmark:
    def __init__(self) -> None:
        self.tasks = [
            SimpleNamespace(name="task_zero", language="do task zero"),
            SimpleNamespace(name="task_one", language="do task one"),
        ]
        self.init_states = {
            0: ["zero-a", "zero-b", "zero-c"],
            1: ["one-a", "one-b", "one-c"],
        }

    def get_task(self, task_id):
        return self.tasks[task_id]

    def get_task_init_states(self, task_id):
        return self.init_states[task_id]


def test_eval_module_only_imports_libero_lazily() -> None:
    source = Path(eval_policy.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    eager_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            eager_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            eager_roots.add(node.module.split(".", 1)[0])
    assert not {"libero", "robosuite", "mujoco", "torch"} & eager_roots


def test_run_episode_settles_then_requests_one_action_per_policy_step() -> None:
    env = FakeEnv(success_after_steps=8)
    policy = FakePolicy()
    result = eval_policy.run_episode(
        env=env,
        policy=policy,
        suite="libero_spatial",
        task_id=3,
        task_name="synthetic_task",
        prompt="perform the synthetic task",
        trial_index=0,
        init_state_index=4,
        init_state="state-four",
        step_budget=20,
        dummy_action=eval_policy.DEFAULT_SETTLE_ACTION,
        settle_steps=5,
    )

    assert result.success is True
    assert result.policy_steps == 3
    assert result.episode_key == (
        "libero/run-fake/libero_spatial/task-3/trial-0/init-4"
    )
    assert env.reset_count == 1
    assert env.init_states == ["state-four"]
    assert len(env.actions) == 8
    for action in env.actions[:5]:
        np.testing.assert_array_equal(
            action, np.asarray(eval_policy.DEFAULT_SETTLE_ACTION, dtype=np.float32)
        )
    for action in env.actions[5:]:
        assert action[0] == pytest.approx(0.1)
    assert len(policy.reset_calls) == 1
    assert len(policy.predict_observations) == result.policy_steps
    assert policy.request_count == result.policy_steps


def test_run_episode_stops_at_budget_without_hidden_retry() -> None:
    env = FakeEnv(success_after_steps=999)
    policy = FakePolicy()
    result = eval_policy.run_episode(
        env=env,
        policy=policy,
        suite="libero_goal",
        task_id=0,
        task_name="never_succeeds",
        prompt="do the task",
        trial_index=0,
        init_state_index=0,
        init_state="fixed",
        step_budget=4,
        dummy_action=eval_policy.DEFAULT_SETTLE_ACTION,
    )
    assert result.success is False
    assert result.policy_steps == 4
    assert policy.request_count == 4
    assert len(env.actions) == 5 + 4


def test_evaluate_benchmark_uses_fixed_init_states_and_closes_each_env() -> None:
    benchmark = FakeBenchmark()
    policy = FakePolicy()
    environments: list[FakeEnv] = []
    observed = []

    def env_factory(task_id, step_budget):
        assert step_budget == 220
        env = FakeEnv(success_after_steps=6)
        environments.append(env)
        return env

    results = eval_policy.evaluate_benchmark(
        benchmark=benchmark,
        task_ids=(0, 1),
        num_trials_per_task=2,
        suite="libero_spatial",
        seed=0,
        settle_steps=5,
        dummy_action=eval_policy.DEFAULT_SETTLE_ACTION,
        step_budgets=eval_policy.SUITE_STEP_BUDGETS,
        env_factory=env_factory,
        policy=policy,
        on_result=observed.append,
    )

    assert len(results) == 4
    assert observed == results
    assert [result.init_state_index for result in results] == [0, 1, 0, 1]
    assert [result.success for result in results] == [True, True, True, True]
    assert [env.seeds for env in environments] == [[0], [0]]
    assert [env.init_states for env in environments] == [
        ["zero-a", "zero-b"],
        ["one-a", "one-b"],
    ]
    assert all(env.closed for env in environments)
    assert len(policy.reset_calls) == 4


def test_evaluate_benchmark_closes_environment_on_policy_failure() -> None:
    benchmark = FakeBenchmark()
    policy = FakePolicy(fail_on_predict=True)
    env = FakeEnv(success_after_steps=999)
    with pytest.raises(RuntimeError, match="synthetic policy failure"):
        eval_policy.evaluate_benchmark(
            benchmark=benchmark,
            task_ids=(0,),
            num_trials_per_task=1,
            suite="libero_spatial",
            seed=0,
            settle_steps=5,
            dummy_action=eval_policy.DEFAULT_SETTLE_ACTION,
            step_budgets=eval_policy.SUITE_STEP_BUDGETS,
            env_factory=lambda _task_id, _step_budget: env,
            policy=policy,
        )
    assert env.closed is True


def test_evaluate_benchmark_refuses_to_repeat_fixed_init_states() -> None:
    benchmark = FakeBenchmark()
    env_factory_called = False

    def env_factory(_task_id, _step_budget):
        nonlocal env_factory_called
        env_factory_called = True
        return FakeEnv()

    with pytest.raises(ValueError, match="repeated states"):
        eval_policy.evaluate_benchmark(
            benchmark=benchmark,
            task_ids=(0,),
            num_trials_per_task=4,
            suite="libero_spatial",
            seed=0,
            settle_steps=5,
            dummy_action=eval_policy.DEFAULT_SETTLE_ACTION,
            step_budgets=eval_policy.SUITE_STEP_BUDGETS,
            env_factory=env_factory,
            policy=FakePolicy(),
        )
    assert env_factory_called is False


def test_summarize_results_counts_every_episode() -> None:
    base = dict(
        suite="libero_object",
        task_name="task",
        prompt="prompt",
        episode_key="key",
        policy_steps=7,
        step_budget=280,
        elapsed_seconds=0.1,
    )
    results = [
        eval_policy.EpisodeResult(
            **base,
            task_id=0,
            trial_index=0,
            init_state_index=0,
            success=True,
        ),
        eval_policy.EpisodeResult(
            **base,
            task_id=0,
            trial_index=1,
            init_state_index=1,
            success=False,
        ),
    ]
    summary = eval_policy.summarize_results(results)
    assert summary["episodes"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["per_task"][0]["episodes"] == 2
    assert summary["per_task"][0]["success_rate"] == 0.5


def test_policy_config_is_the_frozen_benchmark_contract(tmp_path) -> None:
    config_path = Path("benchmarks/libero/policy_config.yml")
    config = eval_policy.load_and_validate_config(config_path)
    assert config["libero_git_sha"] == eval_policy.PINNED_LIBERO_GIT_SHA
    assert config["suite_max_steps"] == eval_policy.SUITE_STEP_BUDGETS
    assert config["state_dim"] == 8
    assert config["action_dim"] == 7
    assert config["dummy_action"] == list(eval_policy.DEFAULT_SETTLE_ACTION)
    assert config["model_noise_base_seed"] == 2026080601
    assert config["expected_server_contract"]["benchmark"] == "libero"

    drifted = copy.deepcopy(config)
    drifted["action_dim"] = 14
    drifted_path = tmp_path / "drifted.yml"
    drifted_path.write_text(yaml.safe_dump(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="action_dim"):
        eval_policy.load_and_validate_config(drifted_path)

    drifted = copy.deepcopy(config)
    drifted["dummy_action"][-1] = 0.0
    drifted_path = tmp_path / "closed-gripper-settle.yml"
    drifted_path.write_text(yaml.safe_dump(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="keep the LIBERO gripper open"):
        eval_policy.load_and_validate_config(drifted_path)


@pytest.mark.parametrize(
    ("raw", "num_tasks", "expected"),
    [("all", 3, (0, 1, 2)), ("0", 10, (0,)), ("9", 10, (9,))],
)
def test_parse_task_ids(raw, num_tasks, expected) -> None:
    assert eval_policy._parse_task_ids(raw, num_tasks) == expected


@pytest.mark.parametrize("raw", ["-1", "10", "01", "libero_90"])
def test_parse_task_ids_rejects_invalid_values(raw) -> None:
    with pytest.raises(ValueError):
        eval_policy._parse_task_ids(raw, 10)
