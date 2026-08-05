#!/usr/bin/env python3
"""Run RoboTwin's eval_policy.py without its fragile render self-test."""

from __future__ import annotations

import ast
import importlib.util
import hashlib
import inspect
import json
import os
import sys
import tempfile
import textwrap
import traceback
import types


PROMPT_MANIFEST_SCHEMA_VERSION = 1


def _reject_reserved_stage0_cli_config() -> None:
    """Reject reserved CACH markers before root creation or CUDA prewarm."""

    config_paths = []
    index = 1
    while index < len(sys.argv):
        value = sys.argv[index]
        if value == "--config":
            if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
                raise RuntimeError("--config requires exactly one path")
            config_paths.append(sys.argv[index + 1])
            index += 2
            continue
        if value.startswith("--config="):
            config_paths.append(value.split("=", 1)[1])
        index += 1
    if len(config_paths) != 1 or not config_paths[0]:
        raise RuntimeError(
            "eval_policy_wrapper requires exactly one unambiguous --config path"
        )
    config_path = config_paths[0]
    import yaml

    with open(config_path, "r", encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    if not isinstance(base_config, dict):
        raise RuntimeError("RoboTwin policy config must be a mapping")
    reserved = tuple(
        marker
        for marker in ("cach_stage0", "cach_stage1")
        if marker in base_config
    )
    if reserved:
        raise RuntimeError(
            "benchmarks/robotwin/eval_policy_wrapper.py: CACH configs "
            f"{reserved!r} are non-executable"
        )
    model = base_config.get("model", {})
    architecture = model.get("architecture", {}) if isinstance(model, dict) else {}
    direct_variant = base_config.get("variant")
    nested_variant = (
        architecture.get("variant")
        if isinstance(architecture, dict)
        else None
    )
    if (
        direct_variant == "cach_sana_wam_v0"
        or nested_variant == "cach_sana_wam_v0"
    ):
        raise RuntimeError(
            "benchmarks/robotwin/eval_policy_wrapper.py: CACH evaluation "
            "is not authorized"
        )


class _ManifestSeedReplayState:
    """Advance RoboTwin only after one fixed scene reaches policy rollout."""

    def __init__(self, schedule: tuple[dict, ...]):
        self.schedule = schedule
        self.next_ordinal = 0
        self.setup_successes: dict[int, int] = {}
        self.instruction_successes: set[int] = set()

    def begin_setup(
        self,
        token: _ManifestReplaySeed,
        *,
        task_name,
        task_config,
        environment_seed,
        episode_index,
    ) -> None:
        ordinal = token.ordinal
        if ordinal != self.next_ordinal:
            raise RuntimeError(
                "fixed-seed replay order mismatch: "
                f"token_ordinal={ordinal}, expected_ordinal={self.next_ordinal}"
            )
        expected = self.schedule[ordinal]
        actual = _prompt_manifest_key(task_name, task_config, environment_seed)
        expected_key = (
            expected["task_name"],
            expected["task_config"],
            expected["environment_seed"],
        )
        if actual != expected_key or episode_index != ordinal:
            raise RuntimeError(
                "fixed-seed replay setup mismatch: "
                f"ordinal={ordinal}, actual_scene={actual!r}, "
                f"expected_scene={expected_key!r}, episode_index={episode_index!r}"
            )
        setup_count = self.setup_successes.get(ordinal, 0)
        if setup_count >= 2:
            raise RuntimeError(
                "fixed-seed replay called setup_demo too many times: "
                f"ordinal={ordinal}, seed={int(token)}"
            )

    def finish_setup(self, token: _ManifestReplaySeed) -> None:
        ordinal = token.ordinal
        self.setup_successes[ordinal] = self.setup_successes.get(ordinal, 0) + 1

    def finish_instruction(self, ordinal) -> None:
        if ordinal != self.next_ordinal:
            raise RuntimeError(
                "fixed-seed replay instruction order mismatch: "
                f"ordinal={ordinal!r}, expected_ordinal={self.next_ordinal}"
            )
        if self.setup_successes.get(ordinal, 0) != 2:
            raise RuntimeError(
                "fixed-seed replay selected an instruction before both setup calls: "
                f"ordinal={ordinal!r}"
            )
        if ordinal in self.instruction_successes:
            raise RuntimeError(
                f"fixed-seed replay selected multiple instructions: ordinal={ordinal}"
            )
        self.instruction_successes.add(ordinal)

    def advance(self, token: _ManifestReplaySeed, increment):
        if type(increment) is not int or increment != 1:
            raise RuntimeError(
                "fixed-seed replay only supports unit seed advancement, "
                f"got {increment!r}"
            )
        ordinal = token.ordinal
        setup_count = self.setup_successes.get(ordinal, 0)
        instruction_ready = ordinal in self.instruction_successes
        if ordinal != self.next_ordinal or setup_count != 2 or not instruction_ready:
            expected = self.schedule[ordinal]
            raise RuntimeError(
                "fixed-seed expert admission failed; refusing to scan a "
                "replacement scene: "
                f"ordinal={ordinal}, seed={expected['environment_seed']}, "
                f"completed_setup_calls={setup_count}/2, "
                f"instruction_ready={instruction_ready}"
            )
        self.next_ordinal += 1
        if self.next_ordinal == len(self.schedule):
            return int(token) + 1
        return _ManifestReplaySeed(self, self.next_ordinal)


class _ManifestReplaySeed(int):
    """Integer-compatible cursor over a prompt manifest's non-contiguous seeds."""

    def __new__(cls, state: _ManifestSeedReplayState, ordinal: int):
        value = state.schedule[ordinal]["environment_seed"]
        instance = int.__new__(cls, value)
        instance.state = state
        instance.ordinal = ordinal
        return instance

    def __add__(self, increment):
        return self.state.advance(self, increment)

    def __iadd__(self, increment):
        return self.state.advance(self, increment)


class _ManifestReplayProtocolAbort(BaseException):
    """Escape RoboTwin's broad admission exception handler on contract drift."""


def _prompt_manifest_key(
    task_name, task_config, environment_seed
) -> tuple[str, str, int]:
    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError(f"invalid prompt-manifest task_name: {task_name!r}")
    if not isinstance(task_config, str) or not task_config.strip():
        raise ValueError(f"invalid prompt-manifest task_config: {task_config!r}")
    if isinstance(environment_seed, bool) or not isinstance(environment_seed, int):
        raise ValueError(
            f"invalid prompt-manifest environment_seed: {environment_seed!r}"
        )
    return task_name, task_config, environment_seed


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_prompt_manifest(path: str | None) -> dict | None:
    """Load a fail-closed scene-to-prompt manifest before RoboTwin starts."""
    if not path:
        return None
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"RoboTwin prompt manifest not found: {resolved}")
    with open(resolved, "rb") as stream:
        raw = stream.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid RoboTwin prompt manifest {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"RoboTwin prompt manifest must be a JSON object: {resolved}")
    if payload.get("schema_version") != PROMPT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported prompt-manifest schema in {resolved}: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("kind") != "robotwin_prompt_replay":
        raise ValueError(f"invalid prompt-manifest kind in {resolved}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"prompt manifest has no episodes: {resolved}")

    entries = {}
    schedule = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"prompt manifest episode {index} is not an object")
        if episode.get("ordinal") != index:
            raise ValueError(
                f"non-contiguous prompt-manifest ordinal at episode {index}"
            )
        if episode.get("episode_index") != index:
            raise ValueError(
                f"prompt-manifest episode_index mismatch at episode {index}"
            )
        key = _prompt_manifest_key(
            episode.get("task_name"),
            episode.get("task_config"),
            episode.get("environment_seed"),
        )
        if key in entries:
            raise ValueError(f"duplicate prompt-manifest scene identity: {key!r}")
        if schedule:
            previous = schedule[-1]
            previous_task = (
                previous["task_name"],
                previous["task_config"],
            )
            if key[:2] != previous_task:
                raise ValueError(
                    "prompt manifest mixes task identities: "
                    f"{key[:2]!r} != {previous_task!r}"
                )
            if key[2] <= previous["environment_seed"]:
                raise ValueError(
                    "prompt-manifest environment seeds must be strictly increasing: "
                    f"{key[2]} after {previous['environment_seed']}"
                )
        prompt = episode.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"empty prompt for prompt-manifest scene {key!r}")
        digest = _prompt_sha256(prompt)
        if episode.get("prompt_sha256") != digest:
            raise ValueError(f"prompt SHA-256 mismatch for scene {key!r}")
        expected_pair_key = f"robotwin/{key[0]}/{key[1]}/scene-{key[2]}"
        if episode.get("noise_pair_key") != expected_pair_key:
            raise ValueError(
                f"noise_pair_key mismatch for scene {key!r}: "
                f"{episode.get('noise_pair_key')!r} != {expected_pair_key!r}"
            )
        entry = {
            "ordinal": index,
            "task_name": key[0],
            "task_config": key[1],
            "environment_seed": key[2],
            "prompt": prompt,
            "prompt_sha256": digest,
            "noise_pair_key": expected_pair_key,
        }
        entries[key] = entry
        schedule.append(entry)
    return {
        "path": resolved,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "entries": entries,
        "schedule": tuple(schedule),
    }


def _load_robotwin_eval_module(robotwin_path: str):
    script_path = os.path.join(robotwin_path, "script", "eval_policy.py")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"RoboTwin eval script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("robotwin_eval_policy", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load RoboTwin eval module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_runtime_root(robotwin_path: str) -> str:
    runtime_root = os.environ.get("ROBOTWIN_RUNTIME_ROOT", "")
    if runtime_root:
        runtime_root = os.path.abspath(runtime_root)
        os.makedirs(runtime_root, exist_ok=True)
    elif os.access(robotwin_path, os.W_OK):
        return robotwin_path
    else:
        runtime_root = tempfile.mkdtemp(
            prefix="robotwin_runtime.", dir=os.environ.get("TMPDIR", "/tmp")
        )

    for name in os.listdir(robotwin_path):
        if name == "eval_result":
            continue
        src = os.path.join(robotwin_path, name)
        dst = os.path.join(runtime_root, name)
        if os.path.lexists(dst):
            continue
        os.symlink(src, dst)

    os.makedirs(os.path.join(runtime_root, "eval_result"), exist_ok=True)
    print(f"[eval_policy_wrapper] runtime_root={runtime_root}")
    return runtime_root


def _prewarm_cuda_for_curobo() -> None:
    """Initialize CUDA/Curobo before RoboTwin imports SAPIEN.

    On this container, importing ``sapien`` first can poison CUDA discovery
    for the rest of the process (torch then reports Error 304). Prewarming
    torch.cuda and importing curobo up front keeps the later RoboTwin import
    chain on the healthy path.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            print(
                "[eval_policy_wrapper] torch.cuda.is_available() is false before RoboTwin import"
            )
            return

        _ = torch.cuda.device_count()
        _ = torch.zeros(1, device="cuda")

        from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401

        print("[eval_policy_wrapper] prewarmed CUDA/Curobo before SAPIEN import")
    except Exception as exc:
        print(
            f"[eval_policy_wrapper] CUDA/Curobo prewarm failed: {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()


def _patch_warp_torch_namespace() -> None:
    """Provide the legacy ``warp.torch`` namespace expected by this cuRobo fork.

    Newer Warp versions expose Torch interop as top-level functions
    (``warp.device_from_torch`` etc.) but no longer ship a ``warp.torch``
    submodule. This RoboTwin-pinned cuRobo still calls ``wp.torch.*`` once in
    ``world_mesh.py``. Recreate that namespace as a thin compatibility alias.
    """
    try:
        import warp as wp
    except Exception:
        return

    if hasattr(wp, "torch"):
        return

    interop = types.SimpleNamespace(
        from_torch=getattr(wp, "from_torch", None),
        to_torch=getattr(wp, "to_torch", None),
        dtype_from_torch=getattr(wp, "dtype_from_torch", None),
        dtype_to_torch=getattr(wp, "dtype_to_torch", None),
        device_from_torch=getattr(wp, "device_from_torch", None),
        device_to_torch=getattr(wp, "device_to_torch", None),
        stream_from_torch=getattr(wp, "stream_from_torch", None),
        stream_to_torch=getattr(wp, "stream_to_torch", None),
    )
    if interop.device_from_torch is not None:
        wp.torch = interop
        print("[eval_policy_wrapper] patched warp.torch compatibility namespace")


def _install_env_trace_hooks(module, prompt_manifest: dict | None = None) -> None:
    orig_class_decorator = module.class_decorator
    orig_eval_function_decorator = module.eval_function_decorator
    unstable_error = getattr(module, "UnStableError", None)
    episode_context = {}

    def _wrap_method(env, method_name: str) -> None:
        method = getattr(env, method_name, None)
        if not callable(method):
            return

        def wrapped(*args, **kwargs):
            replay_seed = None
            active_replay_state = (
                prompt_manifest.get("_active_seed_replay_state")
                if prompt_manifest is not None
                else None
            )
            if method_name == "setup_demo":
                replay_seed = kwargs.get("seed")
                if active_replay_state is not None and (
                    not isinstance(replay_seed, _ManifestReplaySeed)
                    or replay_seed.state is not active_replay_state
                ):
                    raise _ManifestReplayProtocolAbort(
                        "fixed-seed replay lost its scheduling token before "
                        f"setup_demo: seed={replay_seed!r}"
                    )
                task_name = kwargs.get("task_name", getattr(env, "task_name", None))
                task_config = kwargs.get("task_config")
                episode_index = kwargs.get("now_ep_num")
                if isinstance(replay_seed, _ManifestReplaySeed):
                    try:
                        replay_seed.state.begin_setup(
                            replay_seed,
                            task_name=task_name,
                            task_config=task_config,
                            environment_seed=int(replay_seed),
                            episode_index=episode_index,
                        )
                    except RuntimeError as exc:
                        raise _ManifestReplayProtocolAbort(str(exc)) from exc
                    # Keep the scheduling token in eval_policy; simulator code gets
                    # an ordinary int and cannot accidentally advance the cursor.
                    kwargs = dict(kwargs)
                    kwargs["seed"] = int(replay_seed)
                episode_context.clear()
                episode_context.update(
                    {
                        "task_name": task_name,
                        "task_config": task_config,
                        "environment_seed": int(replay_seed)
                        if replay_seed is not None
                        else kwargs.get("seed"),
                        "episode_index": episode_index,
                    }
                )
            elif method_name == "set_instruction":
                instruction = args[0] if args else kwargs.get("instruction")
                selected = instruction
                prompt_source = "robotwin"
                if prompt_manifest is not None:
                    key = _prompt_manifest_key(
                        episode_context.get("task_name"),
                        episode_context.get("task_config"),
                        episode_context.get("environment_seed"),
                    )
                    entry = prompt_manifest["entries"].get(key)
                    if entry is None:
                        raise RuntimeError(
                            f"prompt manifest has no entry for accepted scene {key!r}"
                        )
                    episode_index = episode_context.get("episode_index")
                    if episode_index != entry["ordinal"]:
                        raise RuntimeError(
                            "prompt manifest accepted-scene order mismatch: "
                            f"episode_index={episode_index!r}, "
                            f"manifest_ordinal={entry['ordinal']!r}, scene={key!r}"
                        )
                    selected = entry["prompt"]
                    prompt_source = "manifest"
                    episode_context.update(
                        {
                            "prompt_manifest_ordinal": entry["ordinal"],
                            "prompt_manifest_path": prompt_manifest["path"],
                            "prompt_manifest_sha256": prompt_manifest["sha256"],
                        }
                    )
                if not isinstance(selected, str) or not selected.strip():
                    raise RuntimeError(
                        f"accepted scene has an empty instruction: {episode_context!r}"
                    )
                episode_context.update(
                    {
                        "prompt": selected,
                        "prompt_sha256": _prompt_sha256(selected),
                        "prompt_source": prompt_source,
                    }
                )
                if args:
                    args = (selected, *args[1:])
                else:
                    kwargs["instruction"] = selected
            try:
                result = method(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - diagnostic path
                if unstable_error is not None and isinstance(exc, unstable_error):
                    raise
                print(
                    f"[eval_policy_wrapper] exception in {env.__class__.__name__}.{method_name}"
                )
                traceback.print_exc()
                raise
            if isinstance(replay_seed, _ManifestReplaySeed):
                replay_seed.state.finish_setup(replay_seed)
            elif method_name == "set_instruction" and active_replay_state is not None:
                active_replay_state.finish_instruction(
                    episode_context.get("episode_index")
                )
            return result

        setattr(env, method_name, wrapped)

    def traced_class_decorator(task_name):
        env = orig_class_decorator(task_name)
        for method_name in ("setup_demo", "play_once", "set_instruction"):
            _wrap_method(env, method_name)
        return env

    def traced_eval_function_decorator(policy_name, model_name):
        function = orig_eval_function_decorator(policy_name, model_name)
        if model_name != "reset_model":
            return function

        def reset_with_episode_context(model):
            setter = getattr(model, "set_episode_context", None)
            if callable(setter):
                setter(**episode_context)
            return function(model)

        return reset_with_episode_context

    module.class_decorator = traced_class_decorator
    module.eval_function_decorator = traced_eval_function_decorator


def _install_manifest_seed_schedule(module, prompt_manifest: dict | None) -> None:
    if prompt_manifest is None:
        return

    schedule = prompt_manifest["schedule"]
    orig_eval_policy = module.eval_policy
    _validate_manifest_seed_loop_contract(orig_eval_policy)

    def fixed_seed_eval_policy(*args, **kwargs):
        if "test_num" in kwargs:
            test_num = kwargs["test_num"]
        elif len(args) > 5:
            test_num = args[5]
        else:
            test_num = 100
        if isinstance(test_num, bool) or not isinstance(test_num, int):
            raise ValueError(
                f"fixed-seed replay test_num must be an integer, got {test_num!r}"
            )
        if test_num <= 0 or test_num > len(schedule):
            raise ValueError(
                "fixed-seed replay requires 1 <= test_num <= manifest episodes, "
                f"got test_num={test_num}, manifest_episodes={len(schedule)}"
            )

        state = _ManifestSeedReplayState(schedule)
        replay_seed = _ManifestReplaySeed(state, 0)
        if len(args) > 4:
            args = list(args)
            upstream_start_seed = args[4]
            args[4] = replay_seed
            args = tuple(args)
        elif "st_seed" in kwargs:
            upstream_start_seed = kwargs["st_seed"]
            kwargs = dict(kwargs)
            kwargs["st_seed"] = replay_seed
        else:
            raise TypeError("fixed-seed replay eval_policy call has no st_seed")

        print(
            "[eval_policy_wrapper] fixed environment-seed schedule "
            f"episodes={test_num} first_seed={int(replay_seed)} "
            f"upstream_start_seed={upstream_start_seed}"
        )
        if "_active_seed_replay_state" in prompt_manifest:
            raise RuntimeError("fixed-seed replay is already active")
        prompt_manifest["_active_seed_replay_state"] = state
        try:
            result = orig_eval_policy(*args, **kwargs)
            if state.next_ordinal != test_num:
                raise RuntimeError(
                    "fixed-seed replay ended before consuming its exact prefix: "
                    f"consumed={state.next_ordinal}, expected={test_num}"
                )
            return result
        finally:
            prompt_manifest.pop("_active_seed_replay_state", None)

    module.eval_policy = fixed_seed_eval_policy


def _install_seed_offset(module, prompt_manifest: dict | None) -> None:
    raw = os.environ.get("ROBOTWIN_ENV_SEED_OFFSET", "").strip()
    if not raw:
        return
    try:
        offset = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"ROBOTWIN_ENV_SEED_OFFSET must be an integer, got {raw!r}"
        ) from exc
    if offset < 0 or offset >= 100000:
        raise ValueError(
            f"ROBOTWIN_ENV_SEED_OFFSET must satisfy 0 <= offset < 100000, got {offset}"
        )
    if offset == 0:
        return
    if prompt_manifest is not None:
        raise ValueError(
            "ROBOTWIN_ENV_SEED_OFFSET cannot be combined with an exact prompt manifest"
        )

    orig_eval_policy = module.eval_policy

    def offset_eval_policy(*args, **kwargs):
        if len(args) > 4:
            args = list(args)
            upstream_start_seed = args[4]
            if type(upstream_start_seed) is not int:
                raise TypeError("seed-offset eval_policy st_seed must be an integer")
            args[4] = upstream_start_seed + offset
            adjusted_start_seed = args[4]
            args = tuple(args)
        elif "st_seed" in kwargs:
            kwargs = dict(kwargs)
            upstream_start_seed = kwargs["st_seed"]
            if type(upstream_start_seed) is not int:
                raise TypeError("seed-offset eval_policy st_seed must be an integer")
            kwargs["st_seed"] = upstream_start_seed + offset
            adjusted_start_seed = kwargs["st_seed"]
        else:
            raise TypeError("seed-offset eval_policy call has no st_seed")
        if upstream_start_seed // 100000 != adjusted_start_seed // 100000:
            raise ValueError(
                "ROBOTWIN_ENV_SEED_OFFSET crosses the selected 100xxx seed block: "
                f"{upstream_start_seed} -> {adjusted_start_seed}"
            )
        print(
            "[eval_policy_wrapper] offset environment-seed start "
            f"base={upstream_start_seed} offset={offset} "
            f"adjusted={adjusted_start_seed}"
        )
        return orig_eval_policy(*args, **kwargs)

    module.eval_policy = offset_eval_policy


def _validate_manifest_seed_loop_contract(eval_policy) -> None:
    """Fail if upstream no longer exposes the seed-token control points."""
    try:
        source = textwrap.dedent(inspect.getsource(eval_policy))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeError(
            "cannot inspect RoboTwin eval_policy for fixed-seed replay"
        ) from exc

    seed_initializers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "now_seed"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "st_seed"
    ]
    seed_advances = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "now_seed"
    ]
    valid_advances = [
        node
        for node in seed_advances
        if isinstance(node.op, ast.Add)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 1
    ]
    scheduled_setup_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "setup_demo"
        and any(
            keyword.arg == "seed"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "now_seed"
            for keyword in node.keywords
        )
    ]
    scheduled_setup_calls.sort(key=lambda node: node.lineno)
    seed_stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "now_seed"
        and isinstance(node.ctx, ast.Store)
    ]
    parent = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }

    def ancestors(node):
        while node in parent:
            node = parent[node]
            yield node

    setup_tries = [
        next(
            (ancestor for ancestor in ancestors(call) if isinstance(ancestor, ast.Try)),
            None,
        )
        for call in scheduled_setup_calls
    ]
    admission_try = next((node for node in setup_tries if node is not None), None)
    handler_names = (
        [
            handler.type.id if isinstance(handler.type, ast.Name) else None
            for handler in admission_try.handlers
        ]
        if admission_try is not None
        else []
    )
    expert_guarded = admission_try is not None and any(
        isinstance(ancestor, ast.If)
        and isinstance(ancestor.test, ast.Name)
        and ancestor.test.id == "expert_check"
        for ancestor in ancestors(admission_try)
    )
    expected_admission_test = ast.parse(
        "(not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success())",
        mode="eval",
    ).body
    admission_ifs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.dump(node.test, include_attributes=False)
        == ast.dump(expected_admission_test, include_attributes=False)
    ]
    ordered_admission = (
        admission_try is not None
        and len(admission_ifs) == 1
        and len(scheduled_setup_calls) == 2
        and scheduled_setup_calls[0].lineno < admission_ifs[0].lineno
        and admission_ifs[0].end_lineno < scheduled_setup_calls[1].lineno
        and setup_tries == [admission_try, None]
    )
    unsafe_handlers = [
        handler
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler)
        and (
            handler.type is None
            or (
                isinstance(handler.type, ast.Name)
                and handler.type.id == "BaseException"
            )
        )
    ]
    if (
        len(seed_initializers) != 1
        or len(seed_advances) != 4
        or len(valid_advances) != len(seed_advances)
        or len(seed_stores) != 5
        or len(scheduled_setup_calls) != 2
        or handler_names != ["UnStableError", "Exception"]
        or not expert_guarded
        or not ordered_admission
        or len(unsafe_handlers) != 0
    ):
        raise RuntimeError(
            "RoboTwin eval_policy seed-loop contract changed; refusing "
            "fixed-seed replay: "
            f"initializers={len(seed_initializers)}, "
            f"advances={len(seed_advances)}, "
            f"valid_unit_advances={len(valid_advances)}, "
            f"seed_stores={len(seed_stores)}, "
            f"scheduled_setup_calls={len(scheduled_setup_calls)}, "
            f"handlers={handler_names!r}, expert_guarded={expert_guarded}, "
            f"ordered_admission={ordered_admission}, "
            f"unsafe_handlers={len(unsafe_handlers)}"
        )


def _install_test_num_override(module) -> None:
    value = os.environ.get("ROBOTWIN_TEST_NUM", "").strip()
    if not value:
        return
    try:
        test_num = int(value)
    except ValueError as exc:
        raise ValueError(
            f"ROBOTWIN_TEST_NUM must be an integer, got {value!r}"
        ) from exc
    if test_num <= 0:
        raise ValueError(f"ROBOTWIN_TEST_NUM must be > 0, got {test_num}")

    orig_eval_policy = module.eval_policy

    def capped_eval_policy(*args, **kwargs):
        if len(args) > 5:
            args = list(args)
            args[5] = test_num
            args = tuple(args)
            kwargs = dict(kwargs)
            kwargs.pop("test_num", None)
        else:
            kwargs = dict(kwargs)
            kwargs["test_num"] = test_num
        print(f"[eval_policy_wrapper] overriding RoboTwin eval test_num={test_num}")
        return orig_eval_policy(*args, **kwargs)

    module.eval_policy = capped_eval_policy
    _install_main_test_num_override(module, test_num)


def _install_main_test_num_override(module, test_num: int) -> None:
    """Keep RoboTwin's result-file denominator aligned with the eval cap."""
    main_function = getattr(module, "main", None)
    if not callable(main_function):
        return
    try:
        source_lines, start_line = inspect.getsourcelines(main_function)
        tree = ast.parse(textwrap.dedent("".join(source_lines)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeError(
            "cannot inspect RoboTwin main for test_num override"
        ) from exc

    function_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == main_function.__name__
    ]
    assignments = (
        [
            node
            for node in ast.walk(function_defs[0])
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "test_num"
            and isinstance(node.value, ast.Constant)
            and node.value.value == 100
        ]
        if len(function_defs) == 1
        else []
    )
    if len(assignments) != 1:
        raise RuntimeError(
            "RoboTwin main test_num contract changed; refusing an ambiguous "
            f"override: function_defs={len(function_defs)}, "
            f"assignments={len(assignments)}"
        )

    assignments[0].value = ast.copy_location(
        ast.Constant(value=test_num), assignments[0].value
    )
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, start_line - 1)
    filename = inspect.getsourcefile(main_function) or "<robotwin_eval_policy>"
    namespace = main_function.__globals__
    namespace["eval_policy"] = module.eval_policy
    exec(compile(tree, filename, "exec"), namespace)
    module.main = namespace[main_function.__name__]
    print(f"[eval_policy_wrapper] overriding RoboTwin main test_num={test_num}")


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_robot_planner_fallbacks(robotwin_path: str) -> None:
    import envs

    robot_dir = os.path.join(robotwin_path, "envs", "robot")
    planner_path = os.path.join(robot_dir, "planner.py")
    robot_path = os.path.join(robot_dir, "robot.py")

    robot_pkg = types.ModuleType("envs.robot")
    robot_pkg.__file__ = os.path.join(robot_dir, "__init__.py")
    robot_pkg.__package__ = "envs.robot"
    robot_pkg.__path__ = [robot_dir]
    sys.modules["envs.robot"] = robot_pkg
    setattr(envs, "robot", robot_pkg)

    planner_mod = _load_module("envs.robot.planner", planner_path)
    if not hasattr(planner_mod, "CuroboPlanner"):
        planner_mod.CuroboPlanner = type("CuroboPlanner", (), {})
    curobo_cls = planner_mod.CuroboPlanner
    mplib_cls = getattr(planner_mod, "MplibPlanner", None)
    if mplib_cls is None:
        return

    if not hasattr(mplib_cls, "plan_batch"):

        def plan_batch(
            self, now_qpos, target_pose_list, constraint_pose=None, arms_tag=None
        ):
            statuses = []
            positions = []
            velocities = []
            for pose in target_pose_list:
                result = self.plan_path(now_qpos, pose, arms_tag=arms_tag, log=False)
                success = result.get("status") == "Success"
                statuses.append("Success" if success else "Failure")
                positions.append(result.get("position") if success else None)
                velocities.append(result.get("velocity") if success else None)
            return {"status": statuses, "position": positions, "velocity": velocities}

        mplib_cls.plan_batch = plan_batch

    robot_mod = _load_module("envs.robot._robot_impl", robot_path)

    orig_init_robot = robot_mod.Robot._init_robot_
    orig_set_planner = robot_mod.Robot.set_planner

    def safe_init_robot(self, scene, need_topp=False, **kwargs):
        self.left_planner = None
        self.right_planner = None
        self.left_conn = None
        self.right_conn = None
        self.left_proc = None
        self.right_proc = None
        self.communication_flag = False
        return orig_init_robot(self, scene, need_topp, **kwargs)

    def safe_set_planner(self, scene=None):
        try:
            return orig_set_planner(self, scene=scene)
        except Exception as exc:
            print(
                f"[eval_policy_wrapper] planner fallback engaged: {type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

            self.communication_flag = False
            self.left_conn = None
            self.right_conn = None
            self.left_proc = None
            self.right_proc = None
            self.left_planner = mplib_cls(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type
                if self.left_planner_type != "curobo"
                else "mplib_RRT",
                scene,
            )
            self.right_planner = mplib_cls(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type
                if self.right_planner_type != "curobo"
                else "mplib_RRT",
                scene,
            )
            if self.need_topp:
                self.left_mplib_planner = self.left_planner
                self.right_mplib_planner = self.right_planner

    def safe_reset(self, scene, need_topp=False, **kwargs):
        self._init_robot_(scene, need_topp, **kwargs)

        if getattr(self, "communication_flag", False):
            if getattr(self, "left_conn", None):
                self.left_conn.send({"cmd": "reset"})
                _ = self.left_conn.recv()
            if getattr(self, "right_conn", None):
                self.right_conn.send({"cmd": "reset"})
                _ = self.right_conn.recv()
        else:
            left_planner = getattr(self, "left_planner", None)
            right_planner = getattr(self, "right_planner", None)
            curobo_ready = (
                left_planner is not None
                and right_planner is not None
                and isinstance(left_planner, curobo_cls)
                and isinstance(right_planner, curobo_cls)
            )
            if not curobo_ready:
                self.set_planner(scene=scene)

        self.init_joints()

    robot_mod.Robot._init_robot_ = safe_init_robot
    robot_mod.Robot.set_planner = safe_set_planner
    robot_mod.Robot.reset = safe_reset

    robot_pkg.Robot = robot_mod.Robot
    robot_pkg.CuroboPlanner = planner_mod.CuroboPlanner
    robot_pkg.MplibPlanner = planner_mod.MplibPlanner
    robot_pkg.planner = planner_mod
    robot_pkg.robot = robot_mod


def main() -> int:
    _reject_reserved_stage0_cli_config()
    robotwin_path = os.environ.get("ROBOTWIN_PATH")
    if not robotwin_path:
        raise SystemExit("ROBOTWIN_PATH must be set")

    prompt_manifest = _load_prompt_manifest(os.environ.get("ROBOTWIN_PROMPT_MANIFEST"))
    if prompt_manifest is not None:
        print(
            "[eval_policy_wrapper] prompt manifest "
            f"sha256={prompt_manifest['sha256']} entries={len(prompt_manifest['entries'])}"
        )

    runtime_root = _prepare_runtime_root(robotwin_path)
    os.chdir(runtime_root)
    if robotwin_path not in sys.path:
        sys.path.insert(0, robotwin_path)

    _prewarm_cuda_for_curobo()
    _patch_warp_torch_namespace()
    module = _load_robotwin_eval_module(robotwin_path)
    if os.environ.get("ROBOTWIN_ENABLE_PLANNER_FALLBACK", "") == "1":
        _install_robot_planner_fallbacks(robotwin_path)
    _install_env_trace_hooks(module, prompt_manifest=prompt_manifest)
    _install_manifest_seed_schedule(module, prompt_manifest)
    _install_seed_offset(module, prompt_manifest)
    _install_test_num_override(module)
    usr_args = module.parse_args_and_config()
    module.main(usr_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
