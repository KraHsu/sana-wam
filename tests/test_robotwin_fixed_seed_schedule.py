from __future__ import annotations

import ast
import inspect
import textwrap
from types import ModuleType
from types import SimpleNamespace

import pytest

from benchmarks.robotwin import eval_policy_wrapper as wrapper
from benchmarks.robotwin.eval_policy_wrapper import (
    _ManifestReplayProtocolAbort,
    _ManifestReplaySeed,
    _ManifestSeedReplayState,
    _install_env_trace_hooks,
    _install_manifest_seed_schedule,
    _install_seed_offset,
    _install_test_num_override,
    _prompt_sha256,
    _validate_manifest_seed_loop_contract,
)


UnStableError = TimeoutError


def _manifest(seeds=(100001, 100005, 100012, 100020, 100021, 100030)):
    entries = {}
    schedule = []
    for ordinal, seed in enumerate(seeds):
        prompt = f"instruction for seed {seed}"
        key = ("adjust_bottle", "demo_clean", seed)
        entry = {
            "ordinal": ordinal,
            "task_name": key[0],
            "task_config": key[1],
            "environment_seed": seed,
            "prompt": prompt,
            "prompt_sha256": _prompt_sha256(prompt),
            "noise_pair_key": f"robotwin/{key[0]}/{key[1]}/scene-{seed}",
        }
        entries[key] = entry
        schedule.append(entry)
    return {
        "path": "/tmp/prompt_manifest.json",
        "sha256": "a" * 64,
        "entries": entries,
        "schedule": tuple(schedule),
    }


class _Env:
    task_name = "adjust_bottle"

    def __init__(self, *, admission_results=None, play_error=None):
        self.admission_results = iter(admission_results or [])
        self.play_error = play_error
        self.plan_success = True
        self.setup_calls = []
        self.instructions = []

    def setup_demo(self, **kwargs):
        self.setup_calls.append((kwargs["now_ep_num"], kwargs["seed"]))
        assert type(kwargs["seed"]) is int

    def play_once(self):
        if self.play_error is not None:
            raise self.play_error

    def close_env(self):
        return None

    def check_success(self):
        return next(self.admission_results, True)

    def set_instruction(self, instruction=None):
        self.instructions.append(instruction)


def _upstream_eval_policy(
    task_name,
    TASK_ENV,
    args,
    model,
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None,
):
    del task_name, model, video_size, instruction_type
    now_seed = st_seed
    now_id = 0
    accepted = 0
    expert_check = True
    while accepted < test_num:
        if expert_check:
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=now_id,
                    seed=now_seed,
                    is_test=True,
                    **args,
                )
                TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                continue
            except Exception:
                TASK_ENV.close_env()
                now_seed += 1
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            accepted += 1
        else:
            now_seed += 1
            continue

        TASK_ENV.setup_demo(
            now_ep_num=now_id,
            seed=now_seed,
            is_test=True,
            **args,
        )
        TASK_ENV.set_instruction(instruction="upstream random prompt")
        now_id += 1
        TASK_ENV.close_env()
        now_seed += 1
    return now_seed, accepted


def _module_with_env(env):
    return SimpleNamespace(
        class_decorator=lambda task_name: env,
        eval_function_decorator=lambda policy_name, name: lambda model: None,
        eval_policy=_upstream_eval_policy,
        UnStableError=RuntimeError,
    )


def _run(module, env, manifest, *, test_num):
    _install_env_trace_hooks(module, prompt_manifest=manifest)
    _install_manifest_seed_schedule(module, manifest)
    traced_env = module.class_decorator("adjust_bottle")
    return module.eval_policy(
        "adjust_bottle",
        traced_env,
        {"task_name": "adjust_bottle", "task_config": "demo_clean"},
        object(),
        100000,
        test_num=test_num,
    )


def test_manifest_schedule_uses_exact_noncontiguous_seed_prefix():
    manifest = _manifest()
    env = _Env()
    module = _module_with_env(env)

    _, accepted = _run(module, env, manifest, test_num=5)

    assert accepted == 5
    assert env.setup_calls == [
        (0, 100001),
        (0, 100001),
        (1, 100005),
        (1, 100005),
        (2, 100012),
        (2, 100012),
        (3, 100020),
        (3, 100020),
        (4, 100021),
        (4, 100021),
    ]
    assert env.instructions == [
        manifest["schedule"][index]["prompt"] for index in range(5)
    ]


def test_manifest_schedule_consumes_final_episode_without_extra_seed():
    manifest = _manifest()
    env = _Env()
    module = _module_with_env(env)

    next_seed, accepted = _run(module, env, manifest, test_num=6)

    assert accepted == 6
    assert next_seed == 100031
    assert [seed for _, seed in env.setup_calls] == [
        seed
        for seed in (100001, 100005, 100012, 100020, 100021, 100030)
        for _ in range(2)
    ]


def test_unpaired_seed_offset_adjusts_start_within_selected_block(monkeypatch):
    calls = []
    module = SimpleNamespace(
        eval_policy=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setenv("ROBOTWIN_ENV_SEED_OFFSET", "7")

    _install_seed_offset(module, prompt_manifest=None)
    module.eval_policy("task", object(), {}, object(), 200000, test_num=2)

    assert calls[0][0][4] == 200007


def test_seed_offset_rejects_exact_manifest_combination(monkeypatch):
    module = SimpleNamespace(eval_policy=lambda *args, **kwargs: None)
    monkeypatch.setenv("ROBOTWIN_ENV_SEED_OFFSET", "7")

    with pytest.raises(ValueError, match="exact prompt manifest"):
        _install_seed_offset(module, prompt_manifest=_manifest())


def test_fixed_seed_admission_failure_never_scans_replacement():
    manifest = _manifest()
    env = _Env(admission_results=[False])
    module = _module_with_env(env)

    with pytest.raises(RuntimeError, match="refusing to scan a replacement scene"):
        _run(module, env, manifest, test_num=1)

    assert env.setup_calls == [(0, 100001)]


def test_fixed_seed_play_exception_never_scans_replacement():
    manifest = _manifest()
    env = _Env(play_error=RuntimeError("planner failed"))
    module = _module_with_env(env)

    with pytest.raises(RuntimeError, match="refusing to scan a replacement scene"):
        _run(module, env, manifest, test_num=1)

    assert env.setup_calls == [(0, 100001)]


def test_fixed_seed_setup_order_change_is_fatal():
    manifest = _manifest()
    state = _ManifestSeedReplayState(manifest["schedule"])
    token = _ManifestReplaySeed(state, 0)
    state.begin_setup(
        token,
        task_name="adjust_bottle",
        task_config="demo_clean",
        environment_seed=100001,
        episode_index=0,
    )
    state.finish_setup(token)

    with pytest.raises(RuntimeError, match="setup mismatch"):
        state.begin_setup(
            token,
            task_name="adjust_bottle",
            task_config="demo_clean",
            environment_seed=100001,
            episode_index=1,
        )


@pytest.mark.parametrize("increment", [True, 1.0, 2, -1])
def test_fixed_seed_rejects_non_unit_integer_arithmetic(increment):
    manifest = _manifest()
    state = _ManifestSeedReplayState(manifest["schedule"])
    token = _ManifestReplaySeed(state, 0)

    with pytest.raises(RuntimeError, match="only supports unit seed advancement"):
        token + increment


def test_active_replay_rejects_plain_seed_token():
    manifest = _manifest()
    env = _Env()
    module = _module_with_env(env)
    _install_env_trace_hooks(module, prompt_manifest=manifest)
    traced_env = module.class_decorator("adjust_bottle")
    manifest["_active_seed_replay_state"] = _ManifestSeedReplayState(
        manifest["schedule"]
    )

    with pytest.raises(_ManifestReplayProtocolAbort, match="lost its scheduling token"):
        traced_env.setup_demo(
            now_ep_num=0,
            seed=100001,
            task_name="adjust_bottle",
            task_config="demo_clean",
        )


def test_manifest_schedule_rejects_test_num_beyond_manifest():
    manifest = _manifest()
    env = _Env()
    module = _module_with_env(env)
    _install_manifest_seed_schedule(module, manifest)

    with pytest.raises(ValueError, match="manifest_episodes=6"):
        module.eval_policy(None, env, {}, None, 100000, test_num=7)


def test_manifest_schedule_rejects_changed_upstream_seed_loop():
    def changed_eval_policy(task_name, env, args, model, st_seed, test_num=100):
        del task_name, env, args, model, test_num
        now_seed = st_seed
        now_seed = int(now_seed) + 1
        return now_seed, 0

    module = SimpleNamespace(eval_policy=changed_eval_policy)

    with pytest.raises(RuntimeError, match="seed-loop contract changed"):
        _install_manifest_seed_schedule(module, _manifest())


def test_manifest_schedule_rejects_removed_admission_predicate(monkeypatch):
    tree = ast.parse(textwrap.dedent(inspect.getsource(_upstream_eval_policy)))
    admission_if = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "plan_success"
            for child in ast.walk(node.test)
        )
    )
    admission_if.test = ast.Constant(value=True)
    changed_source = ast.unparse(ast.fix_missing_locations(tree))
    monkeypatch.setattr(wrapper.inspect, "getsource", lambda function: changed_source)

    with pytest.raises(RuntimeError, match="seed-loop contract changed"):
        _validate_manifest_seed_loop_contract(_upstream_eval_policy)


def test_test_num_override_without_manifest_keeps_legacy_flow(monkeypatch):
    observed = []
    module = SimpleNamespace(
        eval_policy=lambda *args, **kwargs: observed.append(kwargs["test_num"])
    )
    _install_manifest_seed_schedule(module, None)
    monkeypatch.setenv("ROBOTWIN_TEST_NUM", "5")
    _install_test_num_override(module)

    module.eval_policy(None, None, None, None, 100000, test_num=100)

    assert observed == [5]


def test_test_num_override_replaces_positional_argument(monkeypatch):
    observed = []
    module = SimpleNamespace(
        eval_policy=lambda *args, **kwargs: observed.append((args[5], kwargs))
    )
    monkeypatch.setenv("ROBOTWIN_TEST_NUM", "5")
    _install_test_num_override(module)

    module.eval_policy(None, None, None, None, 100000, 100)

    assert observed == [(5, {})]


@pytest.mark.parametrize("test_num", [1, 30])
def test_test_num_override_updates_main_result_denominator(monkeypatch, test_num):
    main_source = """\
def main(usr_args):
    st_seed = 100000
    test_num = 100
    st_seed, suc_num = eval_policy(None, None, None, None, st_seed, test_num=test_num)
    return suc_num / test_num
"""
    module = ModuleType("fake_robotwin_eval_policy")
    module.eval_policy = lambda task, env, args, model, seed, test_num=100: (
        seed,
        test_num,
    )
    exec(compile(main_source, "fake_robotwin_eval_policy.py", "exec"), module.__dict__)
    monkeypatch.setattr(
        wrapper.inspect,
        "getsourcelines",
        lambda function: (main_source.splitlines(keepends=True), 1),
    )
    monkeypatch.setattr(
        wrapper.inspect,
        "getsourcefile",
        lambda function: "fake_robotwin_eval_policy.py",
    )
    monkeypatch.setenv("ROBOTWIN_TEST_NUM", str(test_num))

    _install_test_num_override(module)

    assert module.main({}) == 1.0
