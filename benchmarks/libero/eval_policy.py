#!/usr/bin/env python3
"""Deterministic LIBERO closed-loop runner for an external sana-wam server.

LIBERO is imported only inside :func:`main`.  This keeps the repository's main
Python 3.12 environment independent from the older robosuite/MuJoCo simulator
stack and lets the pure rollout logic run in lightweight unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.libero.sana_wam2libero_interface import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    SUPPORTED_EVAL_SUITES,
    LiberoPolicyClient,
)

CONFIG_SCHEMA = "sana-wam-libero-benchmark-v1"
PINNED_LIBERO_GIT_SHA = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SUITE_STEP_BUDGETS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
DEFAULT_SETTLE_STEPS = 5
DEFAULT_SETTLE_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


@dataclass(frozen=True)
class EpisodeResult:
    suite: str
    task_id: int
    task_name: str
    prompt: str
    trial_index: int
    init_state_index: int
    episode_key: str
    success: bool
    policy_steps: int
    step_budget: int
    elapsed_seconds: float


def _plain_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def load_and_validate_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the benchmark config and fail closed on semantic drift."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"LIBERO policy config must be a mapping: {config_path}")
    config = dict(loaded)

    expected_scalars = {
        "schema_version": CONFIG_SCHEMA,
        "libero_git_sha": PINNED_LIBERO_GIT_SHA,
        "task_order_index": 0,
        "seed": 0,
        "camera_height": 256,
        "camera_width": 256,
        "state_dim": STATE_DIM,
        "state_mode": "libero_eef_axis_angle_gripper",
        "action_dim": ACTION_DIM,
        "rotate_images_180": True,
        "training_video_rotation_degrees": 0,
        "action_representation": "relative_delta_pose",
        "one_server_request_per_env_step": True,
        "client_action_queue": False,
        "num_steps_wait": DEFAULT_SETTLE_STEPS,
        "default_num_trials_per_task": 20,
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            raise ValueError(
                f"LIBERO config contract mismatch for {key}: "
                f"expected {expected!r}, got {config.get(key)!r}"
            )
    if config.get("send_state") is not True:
        raise ValueError("LIBERO benchmark v1 requires send_state: true")
    if config.get("state_layout") != [
        "eef_x",
        "eef_y",
        "eef_z",
        "eef_axis_angle_x",
        "eef_axis_angle_y",
        "eef_axis_angle_z",
        "gripper_qpos_0",
        "gripper_qpos_1",
    ]:
        raise ValueError("state_layout does not match the frozen 8D contract")
    if config.get("action_layout") != [
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_axis_angle_x",
        "delta_axis_angle_y",
        "delta_axis_angle_z",
        "open_gripper",
    ]:
        raise ValueError("action_layout does not match the frozen 7D contract")
    if config.get("suite_max_steps") != SUITE_STEP_BUDGETS:
        raise ValueError(
            "suite_max_steps must exactly match the frozen project budgets: "
            f"{SUITE_STEP_BUDGETS}"
        )
    if config.get("dummy_action") != list(DEFAULT_SETTLE_ACTION):
        raise ValueError(
            "dummy_action must keep the LIBERO gripper open during settling: "
            f"expected {list(DEFAULT_SETTLE_ACTION)!r}"
        )
    if config.get("dataset_only_suites") != ["libero_90"]:
        raise ValueError("dataset_only_suites must be exactly ['libero_90']")
    if config.get("camera_mapping") != {
        "agentview_image": "head_camera",
        "robot0_eye_in_hand_image": "left_wrist_camera",
        "missing": "right_wrist_camera",
    }:
        raise ValueError("camera_mapping does not match the frozen two-view layout")
    if config.get("gripper") != {
        "model_open": 1.0,
        "model_closed": 0.0,
        "threshold": 0.5,
        "env_open": -1.0,
        "env_closed": 1.0,
    }:
        raise ValueError("gripper convention does not match the frozen contract")
    expected_server_contract = config.get("expected_server_contract")
    if not isinstance(expected_server_contract, dict):
        raise ValueError("expected_server_contract must be a mapping")
    expected_contract_fields = {
        "schema_version": "sana-wam-libero-policy-v1",
        "benchmark": "libero",
        "dataloader_type": "libero",
        "action_mode": "libero_relative_eef",
        "state_mode": "libero_eef_axis_angle_gripper",
        "normalize_required": True,
        "state_dim": STATE_DIM,
        "state_layout": config.get("state_layout"),
        "action_dim": ACTION_DIM,
        "action_layout": config.get("action_layout"),
        "action_representation": "relative_delta_pose",
        "action_output_space": "denormalized_checkpoint_units",
        "normalization": {
            "action": {
                "active": True,
                "mode": "min-max",
                "stats_key": "libero_relative_eef",
                "dim": ACTION_DIM,
            },
            "state": {
                "active": True,
                "mode": "min-max",
                "stats_key": "libero_eef_axis_angle_gripper",
                "dim": STATE_DIM,
            },
        },
        "gripper": config.get("gripper"),
        "multiview": True,
        "camera_layout": [
            "head_camera",
            "left_wrist_camera",
            "right_wrist_camera",
        ],
        "camera_mapping": config.get("camera_mapping"),
        "missing_right_camera_fill": "black",
        "training_video_rotation_degrees": 0,
        "simulator_video_rotation_degrees": 180,
        "training_dataset_names": [
            "libero_spatial_no_noops_1.0.0_lerobot",
            "libero_object_no_noops_1.0.0_lerobot",
            "libero_goal_no_noops_1.0.0_lerobot",
            "libero_10_no_noops_1.0.0_lerobot",
        ],
        "excluded_training_episodes": ["libero_goal_no_noops_1.0.0_lerobot:82"],
    }
    if expected_server_contract != expected_contract_fields:
        raise ValueError(
            "expected_server_contract does not match the frozen LIBERO policy contract"
        )
    model_noise_base_seed = config.get("model_noise_base_seed")
    _plain_int(model_noise_base_seed, "model_noise_base_seed")
    if model_noise_base_seed >= 1 << 63:
        raise ValueError("model_noise_base_seed must be below 2^63")

    _plain_int(config.get("camera_height"), "camera_height", minimum=1)
    _plain_int(config.get("camera_width"), "camera_width", minimum=1)
    _plain_int(
        config.get("default_num_trials_per_task"),
        "default_num_trials_per_task",
        minimum=1,
    )
    port = _plain_int(config.get("http_port"), "http_port", minimum=1)
    if port > 65535:
        raise ValueError(f"http_port must be <= 65535, got {port}")
    if not isinstance(config.get("host"), str) or not config["host"]:
        raise ValueError("host must be a non-empty string")
    timeout = config.get("request_timeout")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise ValueError("request_timeout must be positive")
    return config


def _step_environment(env: Any, action: np.ndarray) -> tuple[Mapping[str, Any], bool]:
    transition = env.step(action)
    if not isinstance(transition, tuple) or len(transition) != 4:
        raise RuntimeError(
            "pinned LIBERO/robosuite env.step must return (obs, reward, done, info)"
        )
    observation, _reward, done, _info = transition
    if not isinstance(observation, Mapping):
        raise RuntimeError("LIBERO env.step returned a non-mapping observation")
    check_success = getattr(env, "check_success", None)
    success = bool(done)
    if callable(check_success):
        success = success or bool(check_success())
    return observation, success


def run_episode(
    *,
    env: Any,
    policy: LiberoPolicyClient,
    suite: str,
    task_id: int,
    task_name: str,
    prompt: str,
    trial_index: int,
    init_state_index: int,
    init_state: Any,
    step_budget: int,
    dummy_action: Sequence[float],
    settle_steps: int = DEFAULT_SETTLE_STEPS,
) -> EpisodeResult:
    """Run one fixed-init episode with exactly one policy request per step."""

    if suite not in SUPPORTED_EVAL_SUITES:
        raise ValueError(f"unsupported LIBERO evaluation suite: {suite!r}")
    _plain_int(task_id, "task_id")
    _plain_int(trial_index, "trial_index")
    _plain_int(init_state_index, "init_state_index")
    _plain_int(step_budget, "step_budget", minimum=1)
    _plain_int(settle_steps, "settle_steps")

    settle_action = np.asarray(dummy_action, dtype=np.float32)
    if settle_action.shape != (ACTION_DIM,) or not np.isfinite(settle_action).all():
        raise ValueError("dummy_action must be a finite 7D action")
    expected_settle_action = np.asarray(DEFAULT_SETTLE_ACTION, dtype=np.float32)
    if not np.array_equal(settle_action, expected_settle_action):
        raise ValueError(
            "dummy_action must keep the LIBERO gripper open during settling: "
            f"expected {expected_settle_action.tolist()!r}"
        )

    env.reset()
    observation = env.set_init_state(init_state)
    if not isinstance(observation, Mapping):
        raise RuntimeError("LIBERO set_init_state returned a non-mapping observation")

    for _ in range(settle_steps):
        observation, _ = _step_environment(env, settle_action)

    episode_key = policy.reset_episode(
        suite=suite,
        task_id=task_id,
        trial_index=trial_index,
        init_state_index=init_state_index,
        task_name=task_name,
        prompt=prompt,
    )
    started = time.monotonic()
    success = bool(getattr(env, "check_success", lambda: False)())
    policy_steps = 0

    while not success and policy_steps < step_budget:
        action = policy.predict(observation, prompt)
        observation, success = _step_environment(env, action)
        policy_steps += 1

    request_count = getattr(policy, "request_count", policy_steps)
    if request_count != policy_steps:
        raise RuntimeError(
            "LIBERO client request accounting drift: "
            f"requests={request_count}, policy_steps={policy_steps}"
        )
    return EpisodeResult(
        suite=suite,
        task_id=task_id,
        task_name=task_name,
        prompt=prompt,
        trial_index=trial_index,
        init_state_index=init_state_index,
        episode_key=episode_key,
        success=success,
        policy_steps=policy_steps,
        step_budget=step_budget,
        elapsed_seconds=round(time.monotonic() - started, 6),
    )


def evaluate_benchmark(
    *,
    benchmark: Any,
    task_ids: Sequence[int],
    num_trials_per_task: int,
    suite: str,
    seed: int,
    settle_steps: int,
    dummy_action: Sequence[float],
    step_budgets: Mapping[str, int],
    env_factory: Callable[[int, int], Any],
    policy: LiberoPolicyClient,
    on_result: Callable[[EpisodeResult], None] | None = None,
) -> list[EpisodeResult]:
    """Evaluate task IDs serially and always close each simulator environment."""

    _plain_int(num_trials_per_task, "num_trials_per_task", minimum=1)
    _plain_int(seed, "seed")
    if suite not in SUPPORTED_EVAL_SUITES:
        raise ValueError(f"unsupported LIBERO evaluation suite: {suite!r}")
    step_budget = _plain_int(step_budgets[suite], "step_budget", minimum=1)
    results: list[EpisodeResult] = []

    for task_id in task_ids:
        _plain_int(task_id, "task_id")
        task = benchmark.get_task(task_id)
        init_states = benchmark.get_task_init_states(task_id)
        if len(init_states) <= 0:
            raise RuntimeError(f"LIBERO task {task_id} exposes no fixed init states")
        if num_trials_per_task > len(init_states):
            raise ValueError(
                f"requested {num_trials_per_task} trials for LIBERO task {task_id}, "
                f"but only {len(init_states)} fixed init states are available; "
                "refusing to count repeated states as new trials"
            )
        task_name = str(task.name)
        prompt = str(task.language)
        if not task_name or not prompt.strip():
            raise RuntimeError(f"invalid LIBERO task identity at index {task_id}")

        env = env_factory(task_id, step_budget)
        try:
            seed_method = getattr(env, "seed", None)
            if callable(seed_method):
                seed_method(seed)
            for trial_index in range(num_trials_per_task):
                init_index = trial_index
                result = run_episode(
                    env=env,
                    policy=policy,
                    suite=suite,
                    task_id=task_id,
                    task_name=task_name,
                    prompt=prompt,
                    trial_index=trial_index,
                    init_state_index=init_index,
                    init_state=init_states[init_index],
                    step_budget=step_budget,
                    dummy_action=dummy_action,
                    settle_steps=settle_steps,
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
        finally:
            env.close()
    return results


def summarize_results(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Return per-task and overall success accounting without hidden retries."""

    per_task: dict[str, dict[str, Any]] = {}
    for result in results:
        key = f"{result.task_id}:{result.task_name}"
        record = per_task.setdefault(
            key,
            {
                "task_id": result.task_id,
                "task_name": result.task_name,
                "episodes": 0,
                "successes": 0,
            },
        )
        record["episodes"] += 1
        record["successes"] += int(result.success)
    for record in per_task.values():
        record["success_rate"] = record["successes"] / record["episodes"]

    episodes = len(results)
    successes = sum(int(result.success) for result in results)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "per_task": [per_task[key] for key in sorted(per_task)],
    }


def _parse_task_ids(raw: str, num_tasks: int) -> tuple[int, ...]:
    if raw == "all":
        return tuple(range(num_tasks))
    try:
        task_id = int(raw)
    except ValueError as exc:
        raise ValueError(f"task-id must be 'all' or an integer, got {raw!r}") from exc
    if str(task_id) != raw or not 0 <= task_id < num_tasks:
        raise ValueError(f"task-id must be in [0, {num_tasks - 1}], got {raw!r}")
    return (task_id,)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_external_libero(checkout: Path, expected_sha: str) -> dict[str, Any]:
    if not checkout.is_dir() or not (checkout / "libero").is_dir():
        raise FileNotFoundError(f"LIBERO_PATH is not a LIBERO checkout: {checkout}")
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_sha = completed.stdout.strip()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"LIBERO checkout mismatch: expected {expected_sha}, got {actual_sha}"
        )
    worktree_status = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if worktree_status.strip():
        raise RuntimeError(
            "LIBERO checkout is dirty despite the pinned HEAD: "
            f"{worktree_status.strip()}"
        )
    config_root = os.environ.get("LIBERO_CONFIG_PATH")
    if not config_root:
        raise RuntimeError("LIBERO_CONFIG_PATH must be explicitly set before import")
    config_path = Path(config_root).expanduser().resolve() / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"LIBERO config does not exist: {config_path}")
    config_bytes = config_path.read_bytes()
    loaded = yaml.safe_load(config_bytes)
    if not isinstance(loaded, dict):
        raise ValueError(f"LIBERO simulator config must be a mapping: {config_path}")
    expected_checkout_paths = {
        "benchmark_root": checkout / "libero" / "libero",
        "bddl_files": checkout / "libero" / "libero" / "bddl_files",
        "init_states": checkout / "libero" / "libero" / "init_files",
        "assets": checkout / "libero" / "libero" / "assets",
    }
    resolved_paths: dict[str, str] = {}
    for key, expected_path in expected_checkout_paths.items():
        raw_path = loaded.get(key)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"LIBERO simulator config is missing path {key!r}")
        resolved = Path(raw_path).expanduser().resolve()
        expected = expected_path.resolve()
        if resolved != expected or not resolved.is_dir():
            raise RuntimeError(
                f"LIBERO simulator config {key} mismatch: "
                f"expected {expected}, got {resolved}"
            )
        resolved_paths[key] = str(resolved)
    datasets = loaded.get("datasets")
    if not isinstance(datasets, str) or not Path(datasets).expanduser().is_dir():
        raise RuntimeError("LIBERO simulator config datasets path does not exist")
    resolved_paths["datasets"] = str(Path(datasets).expanduser().resolve())
    return {
        "git_sha": actual_sha,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "resolved_paths": resolved_paths,
    }


def _simulator_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for distribution in ("libero", "robosuite", "mujoco", "torch", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _create_output_root(
    base: Path, suite: str, task_selection: str, run_nonce: str
) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = base / f"{suite}-{task_selection}-{timestamp}-{run_nonce[:12]}"
    run_root.mkdir(mode=0o755, exist_ok=False)
    return run_root


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-suite-name", required=True)
    parser.add_argument("--task-id", default="all")
    parser.add_argument("--num-trials-per-task", type=int)
    parser.add_argument("--task-order-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--host")
    parser.add_argument("--http-port", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_and_validate_config(args.config)
    suite = args.task_suite_name
    if suite not in SUPPORTED_EVAL_SUITES:
        raise ValueError(
            f"unsupported closed-loop suite {suite!r}; "
            f"choose one of {sorted(SUPPORTED_EVAL_SUITES)}"
        )

    task_order_index = (
        config["task_order_index"]
        if args.task_order_index is None
        else args.task_order_index
    )
    seed = config["seed"] if args.seed is None else args.seed
    camera_height = (
        config["camera_height"] if args.camera_height is None else args.camera_height
    )
    camera_width = (
        config["camera_width"] if args.camera_width is None else args.camera_width
    )
    for name, actual in (
        ("task_order_index", task_order_index),
        ("seed", seed),
        ("camera_height", camera_height),
        ("camera_width", camera_width),
    ):
        if actual != config[name]:
            raise ValueError(
                f"CLI override for {name} changes frozen identity: "
                f"{actual!r} != {config[name]!r}"
            )

    num_trials = (
        config["default_num_trials_per_task"]
        if args.num_trials_per_task is None
        else args.num_trials_per_task
    )
    _plain_int(num_trials, "num_trials_per_task", minimum=1)
    host = config["host"] if args.host is None else args.host
    http_port = config["http_port"] if args.http_port is None else args.http_port
    _plain_int(http_port, "http_port", minimum=1)
    if http_port > 65535:
        raise ValueError("http_port must be <= 65535")

    libero_path_text = os.environ.get("LIBERO_PATH")
    if not libero_path_text:
        raise RuntimeError("LIBERO_PATH must be explicitly set")
    libero_path = Path(libero_path_text).expanduser().resolve()
    external_identity = _verify_external_libero(libero_path, config["libero_git_sha"])

    # Lazy import: these packages must come from LIBERO_PYTHON, not sana-wam's env.
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_factory = get_benchmark_dict().get(suite)
    if benchmark_factory is None:
        raise RuntimeError(f"pinned LIBERO checkout does not expose suite {suite!r}")
    benchmark = benchmark_factory(task_order_index=task_order_index)
    task_ids = _parse_task_ids(args.task_id, benchmark.get_num_tasks())
    selected_task_assets = []
    init_states_root = Path(external_identity["resolved_paths"]["init_states"])
    for task_id in task_ids:
        task = benchmark.get_task(task_id)
        bddl_path = Path(benchmark.get_task_bddl_file_path(task_id)).resolve()
        init_states_path = (
            init_states_root / task.problem_folder / task.init_states_file
        ).resolve()
        if not bddl_path.is_file() or not init_states_path.is_file():
            raise FileNotFoundError(
                f"missing LIBERO task assets for task {task_id}: "
                f"bddl={bddl_path}, init_states={init_states_path}"
            )
        selected_task_assets.append(
            {
                "task_id": task_id,
                "task_name": str(task.name),
                "bddl_path": str(bddl_path),
                "bddl_sha256": _sha256_file(bddl_path),
                "init_states_path": str(init_states_path),
                "init_states_sha256": _sha256_file(init_states_path),
            }
        )

    run_nonce = uuid.uuid4().hex
    policy = LiberoPolicyClient(
        host=host,
        http_port=http_port,
        request_timeout=float(config["request_timeout"]),
        send_state=True,
        expected_server_contract=config["expected_server_contract"],
        model_noise_base_seed=config["model_noise_base_seed"],
        run_nonce=run_nonce,
    )
    run_root = _create_output_root(
        Path(args.output_dir), suite, args.task_id, run_nonce
    )
    events_path = run_root / "episodes.jsonl"
    run_identity = {
        "schema": CONFIG_SCHEMA,
        "run_nonce": run_nonce,
        "suite": suite,
        "task_ids": list(task_ids),
        "num_trials_per_task": num_trials,
        "task_order_index": task_order_index,
        "seed": seed,
        "camera_height": camera_height,
        "camera_width": camera_width,
        "settle_steps": config["num_steps_wait"],
        "dummy_action": config["dummy_action"],
        "model_noise_base_seed": config["model_noise_base_seed"],
        "suite_step_budget": config["suite_max_steps"][suite],
        "libero_git_sha": external_identity["git_sha"],
        "libero_path": str(libero_path),
        "libero_config_path": external_identity["config_path"],
        "libero_config_sha256": external_identity["config_sha256"],
        "libero_resolved_paths": external_identity["resolved_paths"],
        "selected_task_assets": selected_task_assets,
        "simulator_versions": _simulator_versions(),
        "policy_server": f"http://{host}:{http_port}",
        "policy_server_info": policy.server_info,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_root / "run_identity.json", run_identity)

    def env_factory(task_id: int, step_budget: int) -> Any:
        return OffScreenRenderEnv(
            bddl_file_name=benchmark.get_task_bddl_file_path(task_id),
            camera_heights=camera_height,
            camera_widths=camera_width,
            render_gpu_device_id=args.render_gpu_device_id,
            horizon=step_budget + config["num_steps_wait"] + 1,
        )

    expected_episodes = len(task_ids) * num_trials
    results: list[EpisodeResult] = []
    try:
        with events_path.open("x", encoding="utf-8") as event_stream:

            def record_result(result: EpisodeResult) -> None:
                event_stream.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                event_stream.flush()
                os.fsync(event_stream.fileno())
                print(
                    f"[{result.suite}] task={result.task_id} "
                    f"trial={result.trial_index} success={result.success} "
                    f"steps={result.policy_steps}/{result.step_budget}",
                    flush=True,
                )

            results = evaluate_benchmark(
                benchmark=benchmark,
                task_ids=task_ids,
                num_trials_per_task=num_trials,
                suite=suite,
                seed=seed,
                settle_steps=config["num_steps_wait"],
                dummy_action=config["dummy_action"],
                step_budgets=config["suite_max_steps"],
                env_factory=env_factory,
                policy=policy,
                on_result=record_result,
            )
        if len(results) != expected_episodes:
            raise RuntimeError(
                f"incomplete LIBERO run: expected {expected_episodes} episodes, "
                f"recorded {len(results)}"
            )
        summary = {
            "terminal_status": "COMPLETE",
            "expected_episodes": expected_episodes,
            "run_identity": run_identity,
            **summarize_results(results),
        }
        _write_json(run_root / "summary.json", summary)
        _write_json(
            run_root / "TERMINAL.json",
            {
                "status": "COMPLETE",
                "expected_episodes": expected_episodes,
                "observed_episodes": len(results),
            },
        )
    except BaseException as exc:
        terminal_path = run_root / "TERMINAL.json"
        if not terminal_path.exists():
            try:
                _write_json(
                    terminal_path,
                    {
                        "status": "FAILED",
                        "expected_episodes": expected_episodes,
                        "observed_episodes": len(results),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
        raise
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"results: {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
