#!/usr/bin/env python3
"""Pinned LIBERO simulator worker for the T16 paired closed-loop smoke.

The worker runs in the isolated Python 3.10 simulator environment.  Its stdout
is an NDJSON protocol channel; simulator/library chatter is redirected to
stderr.  It never imports or mutates the sana-wam training environment.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib.metadata
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.libero.sana_wam2libero_interface import (  # noqa: E402
    ACTION_DIM,
    extract_libero_cameras,
    extract_libero_state,
    policy_action_to_libero,
)
from benchmarks.utils.client import encode_numpy_b64  # noqa: E402


PROTOCOL_SCHEMA = "sana-wam-libero-t16-sim-worker-v1"
PINNED_LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
PINNED_INTERPRETER_SHA256 = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
PINNED_INTERPRETER_PATH = Path("/home/zch/workspace/starVLA/.venv-libero/bin/python")
PINNED_INTERPRETER_TARGET = Path("/usr/bin/python3.10")
PINNED_VERSIONS = {
    "Pillow": "12.2.0",
    "bddl": "1.0.1",
    "gym": "0.25.2",
    "mujoco": "3.2.3",
    "numpy": "1.24.4",
    "robosuite": "1.4.0",
    "torch": "2.11.0",
}
PINNED_LIBERO_FILES = {
    "libero/libero/__init__.py": (
        "0f3263e04e58acef0a2184a8b73639292fe9ded83e87ff5d786606ca75bcfb94"
    ),
    "libero/libero/benchmark/__init__.py": (
        "4ce6edbb00b3280692f2e2b2e8a14bf6e2d5878965b0484a386c0355a867ecc5"
    ),
    "libero/libero/envs/bddl_base_domain.py": (
        "055836670ac5fcac7e033fc86404718fd53854996e5776b1ff68b646f7bbcc9d"
    ),
    "libero/libero/envs/env_wrapper.py": (
        "a782fb76c9792268d28979474fe72849e1e98ada49c8e997a65359a8d6b6acd0"
    ),
}
PINNED_MODEL_PROMPT = (
    "pick up the black bowl on the cookie box and place it on the plate"
)
PINNED_BDDL_LANGUAGE = (
    "pick the akita black bowl on the cookies box and place it on the plate"
)
PINNED_BDDL_SHA256 = "3d4ccf070c3d9883ae0676f2d888588f98c696ddad71b7694f47c379fc99ef36"
PINNED_INIT_SHA256 = "0627f5f5ce3ef23be546571012be8ef603d93bcb4032bc80feb34937ba580140"
PINNED_INIT_INDEX = 17
PINNED_INIT_COUNT = 50
PINNED_ENV_SEED = 0
PINNED_SETTLE_STEPS = 5
PINNED_CAMERA_SIZE = 256
PINNED_MAX_POLICY_STEPS = 32
PINNED_MAX_PROTOCOL_LINE_BYTES = 32 * 1024 * 1024
PINNED_DIAGNOSTIC_OBJECTS = (
    "akita_black_bowl_1",
    "cookies_1",
    "plate_1",
)
_ARM_ORDER = ("pre", "post")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    mode = os.lstat(path).st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    return path.resolve()


def _git_identity(checkout: Path) -> dict[str, str]:
    if checkout.is_symlink() or not checkout.is_dir():
        raise ValueError(f"LIBERO checkout is invalid: {checkout}")
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_text = subprocess.run(
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
    if head != PINNED_LIBERO_COMMIT or status_text.strip():
        raise RuntimeError(
            f"LIBERO source identity differs: head={head!r} status={status_text!r}"
        )
    return {"commit": head, "status": status_text}


def _versions() -> dict[str, str]:
    observed = {}
    for name, expected in PINNED_VERSIONS.items():
        value = importlib.metadata.version(name)
        if name == "torch":
            value = value.split("+")[0]
        if value != expected:
            raise RuntimeError(
                f"simulator distribution {name} differs: {value!r} != {expected!r}"
            )
        observed[name] = importlib.metadata.version(name)
    return observed


def _module_origin(module: Any, checkout: Path, label: str) -> str:
    raw = getattr(module, "__file__", None)
    if raw is None:
        raise RuntimeError(f"{label} has no concrete module origin")
    origin = Path(raw).resolve()
    expected_root = checkout.resolve()
    if expected_root not in origin.parents:
        raise RuntimeError(
            f"{label} resolved outside pinned checkout: {origin} not under {expected_root}"
        )
    return str(origin)


def _all_libero_module_origins(checkout: Path) -> dict[str, Any]:
    origins: dict[str, Any] = {}
    expected_root = checkout.resolve()
    for name, module in sorted(sys.modules.items()):
        if name != "libero" and not name.startswith("libero."):
            continue
        raw = getattr(module, "__file__", None)
        if raw is not None:
            origins[name] = _module_origin(module, checkout, name)
            continue
        namespace_paths = [
            Path(path).resolve() for path in getattr(module, "__path__", ())
        ]
        if not namespace_paths or any(
            expected_root not in path.parents and path != expected_root
            for path in namespace_paths
        ):
            raise RuntimeError(f"{name} namespace resolved outside pinned checkout")
        origins[name] = {"namespace_paths": [str(path) for path in namespace_paths]}
    if not origins:
        raise RuntimeError("no imported LIBERO modules were available for origin audit")
    return origins


def _observation_payload(observation: Mapping[str, Any], env: Any) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise TypeError("LIBERO observation must be a mapping")
    cameras = extract_libero_cameras(observation)
    state = extract_libero_state(observation)
    head = cameras["head"]
    left = cameras["left"]
    sim_state = np.asarray(env.get_sim_state(), dtype=np.float64)
    if sim_state.ndim != 1 or not np.isfinite(sim_state).all():
        raise RuntimeError("MuJoCo simulator state must be one finite vector")
    if state.shape != (8,) or not np.isfinite(state).all():
        raise RuntimeError("LIBERO project state must be finite shape (8,)")
    objects = {}
    for name in PINNED_DIAGNOSTIC_OBJECTS:
        position = np.asarray(observation.get(f"{name}_pos"), dtype=np.float64)
        quaternion = np.asarray(observation.get(f"{name}_quat"), dtype=np.float64)
        if (
            position.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
        ):
            raise RuntimeError(f"LIBERO diagnostic object state differs for {name}")
        objects[name] = {
            "position": position.tolist(),
            "position_sha256": _sha256_array(position),
            "quaternion_xyzw": quaternion.tolist(),
            "quaternion_sha256": _sha256_array(quaternion),
        }
    encoded_head = encode_numpy_b64(head)
    encoded_left = encode_numpy_b64(left)
    return {
        "images": {
            "head_camera": encoded_head,
            "left_wrist_camera": encoded_left,
            "right_wrist_camera": None,
        },
        "jpeg_sha256": {
            "head_camera": hashlib.sha256(base64.b64decode(encoded_head)).hexdigest(),
            "left_wrist_camera": hashlib.sha256(
                base64.b64decode(encoded_left)
            ).hexdigest(),
        },
        "observation_sha256": {
            "head_camera": _sha256_array(head),
            "left_wrist_camera": _sha256_array(left),
            "state": _sha256_array(state),
        },
        "objects": objects,
        "sim_state": sim_state.tolist(),
        "sim_state_sha256": _sha256_array(sim_state),
        "state": state.tolist(),
    }


def _step_environment(
    env: Any, action: np.ndarray
) -> tuple[Mapping[str, Any], float, bool, dict]:
    transition = env.step(action)
    if not isinstance(transition, tuple) or len(transition) != 4:
        raise RuntimeError(
            "pinned LIBERO env.step must return (observation,reward,done,info)"
        )
    observation, reward, done, info = transition
    if not isinstance(observation, Mapping):
        raise RuntimeError("LIBERO env.step returned a non-mapping observation")
    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise RuntimeError("LIBERO env.step returned non-finite reward")
    info_keys = sorted(str(key) for key in info) if isinstance(info, Mapping) else []
    return observation, reward_value, bool(done), {"keys": info_keys}


def _send(payload: dict[str, Any]) -> None:
    record = {"schema_version": PROTOCOL_SCHEMA, **payload}
    sys.stdout.write(
        json.dumps(
            record,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    sys.stdout.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-path", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bddl", required=True, type=Path)
    parser.add_argument("--init-file", required=True, type=Path)
    parser.add_argument("--render-gpu-device-id", required=True, type=int)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-worker-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    worker_path = Path(__file__).resolve()
    if _sha256_file(worker_path) != args.expected_worker_sha256.lower():
        raise RuntimeError("simulator worker source SHA256 differs")
    interpreter_link = Path(sys.executable)
    if interpreter_link != PINNED_INTERPRETER_PATH:
        raise RuntimeError(f"simulator interpreter path differs: {interpreter_link}")
    interpreter = interpreter_link.resolve()
    if (
        interpreter != PINNED_INTERPRETER_TARGET
        or not interpreter.is_file()
        or _sha256_file(interpreter) != PINNED_INTERPRETER_SHA256
    ):
        raise RuntimeError("simulator interpreter SHA256 differs")
    checkout = args.libero_path.expanduser().resolve()
    source_identity = _git_identity(checkout)
    source_files = {}
    for relative_path, expected_sha256 in sorted(PINNED_LIBERO_FILES.items()):
        source_file = _regular_file(checkout / relative_path, relative_path)
        observed_sha256 = _sha256_file(source_file)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"pinned LIBERO source file differs: {relative_path}")
        source_files[relative_path] = observed_sha256
    config = _regular_file(args.config.expanduser(), "LIBERO config")
    if _sha256_file(config) != args.expected_config_sha256.lower():
        raise RuntimeError("LIBERO config SHA256 differs")
    configured_root = os.environ.get("LIBERO_CONFIG_PATH")
    if configured_root is None or Path(configured_root).resolve() != config.parent:
        raise RuntimeError("LIBERO_CONFIG_PATH does not bind the pinned config parent")
    bddl = _regular_file(args.bddl.expanduser(), "T16 BDDL")
    init_file = _regular_file(args.init_file.expanduser(), "T16 init file")
    if _sha256_file(bddl) != PINNED_BDDL_SHA256:
        raise RuntimeError("T16 BDDL SHA256 differs")
    if _sha256_file(init_file) != PINNED_INIT_SHA256:
        raise RuntimeError("T16 init-state file SHA256 differs")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("T16 requires MUJOCO_GL=egl")
    expected_render_device = str(args.render_gpu_device_id)
    if args.render_gpu_device_id < 0 or (
        os.environ.get("CUDA_VISIBLE_DEVICES") != expected_render_device
        or os.environ.get("MUJOCO_EGL_DEVICE_ID") != expected_render_device
        or os.environ.get("PYOPENGL_PLATFORM") != "egl"
    ):
        raise RuntimeError(
            "T16 simulator CUDA/EGL identity must equal the physical render GPU"
        )
    versions = _versions()

    # All simulator imports and calls are redirected so stdout remains a pure
    # protocol channel even when robosuite emits import-time warnings.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from libero.libero import envs as libero_envs
        from libero.libero.envs import OffScreenRenderEnv

        critical_module_origins = {
            "libero.envs": _module_origin(libero_envs, checkout, "libero.envs"),
            "libero.env_wrapper": _module_origin(
                sys.modules["libero.libero.envs.env_wrapper"],
                checkout,
                "libero.env_wrapper",
            ),
        }
        if torch.__version__ != "2.11.0+cu130":
            raise RuntimeError(
                f"T16 simulator torch build differs: {torch.__version__!r}"
            )
        init_states = torch.load(init_file, map_location="cpu", weights_only=False)
        init_states = np.asarray(init_states, dtype=np.float64)
        if init_states.ndim != 2 or init_states.shape[0] != PINNED_INIT_COUNT:
            raise RuntimeError(
                f"T16 init-state table shape differs: {init_states.shape!r}"
            )
        init_state = init_states[PINNED_INIT_INDEX].copy()
        if init_state.ndim != 1 or not np.isfinite(init_state).all():
            raise RuntimeError("T16 selected init state must be one finite vector")
        np.random.seed(PINNED_ENV_SEED)
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=PINNED_CAMERA_SIZE,
            camera_widths=PINNED_CAMERA_SIZE,
            render_gpu_device_id=args.render_gpu_device_id,
            horizon=PINNED_MAX_POLICY_STEPS + PINNED_SETTLE_STEPS + 1,
        )
        env.seed(PINNED_ENV_SEED)
        instruction = env.language_instruction
        if isinstance(instruction, (list, tuple)):
            if len(instruction) != 1:
                raise RuntimeError(
                    f"LIBERO language instruction count differs: {instruction!r}"
                )
            instruction = instruction[0]
        if str(instruction).strip() != PINNED_BDDL_LANGUAGE:
            raise RuntimeError(f"LIBERO language instruction differs: {instruction!r}")
        module_origins = _all_libero_module_origins(checkout)

    source_after_import = _git_identity(checkout)
    if source_after_import != source_identity:
        raise RuntimeError("LIBERO checkout changed during worker construction")
    _send(
        {
            "event": "ready",
            "identity": {
                "bddl_sha256": PINNED_BDDL_SHA256,
                "bddl_language": PINNED_BDDL_LANGUAGE,
                "config_path": str(config),
                "config_sha256": _sha256_file(config),
                "init_file_sha256": PINNED_INIT_SHA256,
                "init_index": PINNED_INIT_INDEX,
                "init_state_dtype": str(init_state.dtype),
                "init_state_shape": list(init_state.shape),
                "init_state_sha256": _sha256_array(init_state),
                "interpreter": str(interpreter_link),
                "interpreter_target": str(interpreter),
                "interpreter_sha256": PINNED_INTERPRETER_SHA256,
                "libero_commit": source_identity["commit"],
                "critical_module_origins": critical_module_origins,
                "module_origins": module_origins,
                "render_gpu": {
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "mujoco_egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"],
                    "render_gpu_device_id": args.render_gpu_device_id,
                },
                "task": PINNED_MODEL_PROMPT,
                "source_files": source_files,
                "versions": versions,
                "worker_sha256": args.expected_worker_sha256.lower(),
            },
            "seq": 0,
            "status": "ok",
        }
    )

    next_arm_index = 0
    active_arm: str | None = None
    expected_step = 0
    terminal = False
    arm_complete = True
    reset_identity: dict[str, Any] | None = None
    next_seq = 1
    try:
        for raw_line in sys.stdin:
            try:
                if len(raw_line.encode("utf-8")) > PINNED_MAX_PROTOCOL_LINE_BYTES:
                    raise RuntimeError(
                        "worker request exceeds the fixed line-size bound"
                    )
                request = json.loads(raw_line)
                if not isinstance(request, dict):
                    raise TypeError("worker request must be a JSON object")
                if request.get("seq") != next_seq:
                    raise RuntimeError(
                        f"worker request seq differs: {request.get('seq')!r} != {next_seq}"
                    )
                command = request.get("command")
                if command == "reset":
                    if not arm_complete:
                        raise RuntimeError(
                            "worker reset requested before prior arm completed"
                        )
                    if next_arm_index >= len(_ARM_ORDER):
                        raise RuntimeError("worker received more than two arm resets")
                    arm = request.get("arm")
                    if arm != _ARM_ORDER[next_arm_index]:
                        raise RuntimeError(
                            f"worker arm order differs: {arm!r} != {_ARM_ORDER[next_arm_index]!r}"
                        )
                    with contextlib.redirect_stdout(sys.stderr):
                        env.seed(PINNED_ENV_SEED)
                        env.reset()
                        observation = env.set_init_state(init_state.copy())
                        zero = np.zeros(ACTION_DIM, dtype=np.float32)
                        for _ in range(PINNED_SETTLE_STEPS):
                            observation, _reward, _done, _info = _step_environment(
                                env, zero
                            )
                        success = bool(env.check_success())
                    observed = _observation_payload(observation, env)
                    identity = {
                        "jpeg_sha256": observed["jpeg_sha256"],
                        "objects": observed["objects"],
                        "observation_sha256": observed["observation_sha256"],
                        "sim_state_sha256": observed["sim_state_sha256"],
                    }
                    if reset_identity is None:
                        reset_identity = identity
                    elif identity != reset_identity:
                        raise RuntimeError(
                            f"paired reset identity differs: {identity!r} != {reset_identity!r}"
                        )
                    active_arm = arm
                    expected_step = 0
                    terminal = success
                    arm_complete = success
                    next_arm_index += 1
                    response = {
                        "arm": arm,
                        "event": "reset",
                        "observation": observed,
                        "seq": next_seq,
                        "settle_steps": PINNED_SETTLE_STEPS,
                        "status": "ok",
                        "success": success,
                    }
                elif command == "step":
                    arm = request.get("arm")
                    step = request.get("step")
                    if arm != active_arm or step != expected_step:
                        raise RuntimeError(
                            f"worker step identity differs: arm={arm!r} step={step!r} "
                            f"expected_arm={active_arm!r} expected_step={expected_step}"
                        )
                    if terminal:
                        raise RuntimeError(
                            "worker step requested after terminal transition"
                        )
                    if not 0 <= expected_step < PINNED_MAX_POLICY_STEPS:
                        raise RuntimeError("worker step exceeds the T16 fixed budget")
                    policy_action = np.asarray(request.get("action"), dtype=np.float32)
                    env_action = policy_action_to_libero(policy_action)
                    with contextlib.redirect_stdout(sys.stderr):
                        observation, reward, done, info = _step_environment(
                            env, env_action
                        )
                        success = bool(env.check_success())
                    observed = _observation_payload(observation, env)
                    response = {
                        "arm": arm,
                        "done": bool(done),
                        "env_action": env_action.tolist(),
                        "event": "step",
                        "info": info,
                        "observation": observed,
                        "policy_action": policy_action.tolist(),
                        "reward": reward,
                        "seq": next_seq,
                        "status": "ok",
                        "step": expected_step,
                        "success": success,
                    }
                    expected_step += 1
                    terminal = bool(done) or success
                    arm_complete = terminal or expected_step == PINNED_MAX_POLICY_STEPS
                elif command == "close":
                    if next_arm_index != len(_ARM_ORDER) or not arm_complete:
                        raise RuntimeError(
                            "worker close requested before both arms completed"
                        )
                    source_at_close = _git_identity(checkout)
                    if source_at_close != source_identity:
                        raise RuntimeError(
                            "LIBERO checkout changed during worker lifetime"
                        )
                    for relative_path, expected_sha256 in source_files.items():
                        if _sha256_file(checkout / relative_path) != expected_sha256:
                            raise RuntimeError(
                                f"LIBERO source changed during worker lifetime: {relative_path}"
                            )
                    response = {
                        "event": "closed",
                        "seq": next_seq,
                        "status": "ok",
                    }
                    _send(response)
                    return 0
                else:
                    raise ValueError(f"unsupported worker command: {command!r}")
                _send(response)
                next_seq += 1
            except BaseException as error:
                _send(
                    {
                        "error": str(error),
                        "error_type": type(error).__name__,
                        "event": "error",
                        "seq": next_seq,
                        "status": "error",
                    }
                )
                return 1
        raise RuntimeError("simulator worker stdin closed without explicit close")
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
