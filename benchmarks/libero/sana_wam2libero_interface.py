"""LIBERO-to-sana-wam policy-server adapter.

This module intentionally has no dependency on LIBERO, robosuite, MuJoCo, or
torch.  It runs inside the dedicated LIBERO simulator process and communicates
with an already-running sana-wam policy server over the shared HTTP protocol.

The adapter is strict by design: it accepts the project-specific 8D LIBERO
proprioception layout and a 7D LIBERO-native policy action.  Existing 14D/20D
RoboTwin checkpoints are not converted, sliced, padded, or otherwise reused.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

from benchmarks.utils import client

SUPPORTED_EVAL_SUITES = frozenset(
    {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
)
STATE_DIM = 8
ACTION_DIM = 7


def quaternion_xyzw_to_axis_angle(quaternion: Any) -> np.ndarray:
    """Convert one finite xyzw quaternion with LIBERO's training-time formula.

    This intentionally mirrors the robosuite helper copied by the H200 GR00T
    LIBERO conversion: clip only the scalar component and do *not* canonicalize
    the quaternion sign.  Numerically equivalent rotations are not necessarily
    equivalent model inputs, so online evaluation must match data conversion.
    """

    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"robot0_eef_quat must have shape (4,), got {quat.shape}")
    if not np.all(np.isfinite(quat)):
        raise ValueError("robot0_eef_quat contains non-finite values")

    if float(np.linalg.norm(quat)) <= 1e-12:
        raise ValueError("robot0_eef_quat has zero norm")
    scalar = float(np.clip(quat[3], -1.0, 1.0))
    denominator = float(np.sqrt(1.0 - scalar * scalar))
    if np.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(scalar) / denominator)).astype(np.float32)


def _finite_vector(observation: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    if key not in observation:
        available = ", ".join(sorted(str(name) for name in observation))
        raise KeyError(f"LIBERO observation is missing {key!r}; available: {available}")
    value = np.asarray(observation[key], dtype=np.float32)
    if value.shape != (size,):
        raise ValueError(f"{key} must have shape ({size},), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} contains non-finite values")
    return value


def extract_libero_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Build the frozen 8D project proprioception from a LIBERO observation.

    Layout: ``eef_xyz(3) + eef_axis_angle(3) + gripper_qpos(2)``.
    """

    if not isinstance(observation, Mapping):
        raise TypeError("LIBERO observation must be a mapping")
    eef_position = _finite_vector(observation, "robot0_eef_pos", 3)
    eef_quaternion = _finite_vector(observation, "robot0_eef_quat", 4)
    gripper = _finite_vector(observation, "robot0_gripper_qpos", 2)
    state = np.concatenate(
        [
            eef_position,
            quaternion_xyzw_to_axis_angle(eef_quaternion),
            gripper,
        ]
    ).astype(np.float32, copy=False)
    if state.shape != (STATE_DIM,):  # defensive guard for contract drift
        raise RuntimeError(f"internal LIBERO state layout drifted to {state.shape}")
    return state


def _orient_image(observation: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in observation:
        available = ", ".join(sorted(str(name) for name in observation))
        raise KeyError(f"LIBERO observation is missing {key!r}; available: {available}")
    image = np.asarray(observation[key])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key} must be an HxWx3 RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{key} must have dtype uint8, got {image.dtype}")
    # LIBERO camera arrays are upside down relative to the project dataset.
    # Copy to remove negative strides before Pillow/JPEG encoding.
    return np.ascontiguousarray(image[::-1, ::-1, :])


def extract_libero_cameras(
    observation: Mapping[str, Any],
) -> dict[str, np.ndarray | None]:
    """Map and orient the two LIBERO views to sana-wam's fixed camera slots."""

    if not isinstance(observation, Mapping):
        raise TypeError("LIBERO observation must be a mapping")
    return {
        "head": _orient_image(observation, "agentview_image"),
        "left": _orient_image(observation, "robot0_eye_in_hand_image"),
        "right": None,
    }


def policy_action_to_libero(action: Any) -> np.ndarray:
    """Validate and convert a 7D policy action to LIBERO's controller action.

    The six motion coordinates pass through without clipping.  The final model
    coordinate is trained against ``open=1, closed=0`` and is thresholded to
    LIBERO's ``open=-1, closed=+1`` convention.  Diffusion predictions are not
    range bounded, so finite scores outside ``[0, 1]`` are thresholded directly
    rather than rejected or clamped.
    """

    policy_action = np.asarray(action, dtype=np.float32)
    if policy_action.shape != (ACTION_DIM,):
        raise ValueError(
            "LIBERO requires an exact 7D policy action; "
            f"got shape {policy_action.shape}. RoboTwin 14D/20D checkpoints "
            "are incompatible and must not be sliced or padded."
        )
    if not np.all(np.isfinite(policy_action)):
        raise ValueError("policy action contains non-finite values")

    model_open = float(policy_action[-1])
    env_action = policy_action.copy()
    env_action[-1] = -1.0 if model_open > 0.5 else 1.0
    return env_action


def derive_model_noise_seed(
    *,
    base_seed: int,
    suite: str,
    task_id: int,
    trial_index: int,
    init_state_index: int,
) -> int:
    """Derive a stable signed-63-bit diffusion seed for one benchmark episode."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("model noise base seed must be an integer")
    if not 0 <= base_seed < 1 << 63:
        raise ValueError("model noise base seed must be in [0, 2^63)")
    if suite not in SUPPORTED_EVAL_SUITES:
        raise ValueError(f"unsupported LIBERO evaluation suite: {suite!r}")
    for name, value in (
        ("task_id", task_id),
        ("trial_index", trial_index),
        ("init_state_index", init_state_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    material = (
        f"sana-wam-libero-noise-v1\n{base_seed}\n{suite}\n{task_id}\n"
        f"{trial_index}\n{init_state_index}\n"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def validate_server_contract(
    info: Mapping[str, Any], expected_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the client to a LIBERO-native checkpoint before simulator startup."""

    if not isinstance(info, Mapping) or info.get("model") != "sana-wam":
        raise RuntimeError(f"unexpected policy-server /info response: {info!r}")
    runtime = info.get("inference_runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("policy-server /info is missing inference_runtime")
    identity = runtime.get("deployment_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("policy-server /info is missing deployment_identity")
    expected = dict(expected_contract)
    if identity.get("benchmark_contract") != expected:
        raise RuntimeError(
            "policy server is not bound to the frozen LIBERO checkpoint contract"
        )

    top_level_expectations = {
        "dataloader_type": expected.get("dataloader_type"),
        "action_mode": expected.get("action_mode"),
        "state_mode": expected.get("state_mode"),
        "action_dim": expected.get("action_dim"),
        "state_dim": expected.get("state_dim"),
        "multiview": expected.get("multiview"),
        "camera_layout": expected.get("camera_layout"),
    }
    for key, wanted in top_level_expectations.items():
        if identity.get(key) != wanted:
            raise RuntimeError(
                f"policy-server deployment mismatch for {key}: "
                f"expected {wanted!r}, got {identity.get(key)!r}"
            )
    normalize_mode = identity.get("normalize_mode")
    if expected.get("normalize_required") and (
        normalize_mode is None
        or str(normalize_mode).strip().lower() in {"", "none", "null"}
    ):
        raise RuntimeError("LIBERO checkpoint requires active action normalization")
    expected_normalizers = expected.get("normalization")
    actual_normalizers = identity.get("normalizers")
    if not isinstance(expected_normalizers, Mapping) or not isinstance(
        actual_normalizers, Mapping
    ):
        raise RuntimeError("LIBERO checkpoint normalizer identity is missing")
    for stream in ("action", "state"):
        wanted = expected_normalizers.get(stream)
        actual = actual_normalizers.get(stream)
        if not isinstance(wanted, Mapping) or not isinstance(actual, Mapping):
            raise RuntimeError(f"LIBERO {stream} normalizer identity is missing")
        for key, wanted_value in (
            ("active", wanted.get("active")),
            ("configured_mode", wanted.get("mode")),
            ("dim", wanted.get("dim")),
        ):
            if actual.get(key) != wanted_value:
                raise RuntimeError(
                    f"LIBERO {stream} normalizer mismatch for {key}: "
                    f"expected {wanted_value!r}, got {actual.get(key)!r}"
                )
    if actual_normalizers["state"].get("explicit") is not True:
        raise RuntimeError("LIBERO state normalizer must be independently configured")
    noise_mode = identity.get("episode_noise_mode")
    if noise_mode not in {"ambient", "paired"}:
        raise RuntimeError(
            f"unsupported policy-server episode noise mode: {noise_mode!r}"
        )
    return dict(identity)


class LiberoPolicyClient:
    """One-observation-per-step HTTP client for a LIBERO-native checkpoint."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        http_port: int = 8848,
        request_timeout: float = 300.0,
        send_state: bool = True,
        expected_server_contract: Mapping[str, Any],
        model_noise_base_seed: int,
        run_nonce: str | None = None,
        health_timeout: float = 300.0,
        health_poll_interval: float = 2.0,
    ) -> None:
        if not isinstance(http_port, int) or isinstance(http_port, bool):
            raise TypeError("http_port must be an integer")
        if not 1 <= http_port <= 65535:
            raise ValueError(f"http_port must be in 1..65535, got {http_port}")
        if request_timeout <= 0.0:
            raise ValueError("request_timeout must be positive")
        if health_timeout <= 0.0 or health_poll_interval <= 0.0:
            raise ValueError("health timeout and poll interval must be positive")

        self.server = f"http://{host}:{http_port}"
        self.request_timeout = float(request_timeout)
        self.send_state = bool(send_state)
        if not isinstance(expected_server_contract, Mapping):
            raise TypeError("expected_server_contract must be a mapping")
        self.expected_server_contract = dict(expected_server_contract)
        if isinstance(model_noise_base_seed, bool) or not isinstance(
            model_noise_base_seed, int
        ):
            raise TypeError("model_noise_base_seed must be an integer")
        if not 0 <= model_noise_base_seed < 1 << 63:
            raise ValueError("model_noise_base_seed must be in [0, 2^63)")
        self.model_noise_base_seed = model_noise_base_seed
        self.run_nonce = uuid.uuid4().hex if run_nonce is None else str(run_nonce)
        if not self.run_nonce or not all(
            character.isalnum() or character in {"-", "_"}
            for character in self.run_nonce
        ):
            raise ValueError("run_nonce must contain only alphanumeric, '-' or '_'")
        self.request_count = 0
        self._episode_key: str | None = None
        self._model_noise_seed: int | None = None

        self._wait_until_healthy(health_timeout, health_poll_interval)
        info = client.get(self.server, "/info", timeout=min(10.0, request_timeout))
        self.deployment_identity = validate_server_contract(
            info, self.expected_server_contract
        )
        self.server_info = dict(info)

    def _wait_until_healthy(self, timeout_s: float, poll_interval: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                health = client.get(
                    self.server, "/health", timeout=min(10.0, self.request_timeout)
                )
                if isinstance(health, Mapping) and health.get("status") == "healthy":
                    return
            except Exception as exc:  # server may still be starting
                last_error = exc
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
        raise RuntimeError(
            f"sana-wam server was not healthy within {timeout_s}s at "
            f"{self.server}; last error: {last_error}"
        )

    def reset_episode(
        self,
        *,
        suite: str,
        task_id: int,
        trial_index: int,
        init_state_index: int,
        task_name: str,
        prompt: str,
    ) -> str:
        """Reset all server-side chunk/cache state for one fixed LIBERO episode."""

        if suite not in SUPPORTED_EVAL_SUITES:
            raise ValueError(f"unsupported LIBERO evaluation suite: {suite!r}")
        for name, value in (
            ("task_id", task_id),
            ("trial_index", trial_index),
            ("init_state_index", init_state_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )
        if not isinstance(task_name, str) or not task_name:
            raise ValueError("task_name must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        episode_key = (
            f"libero/run-{self.run_nonce}/{suite}/task-{task_id}/"
            f"trial-{trial_index}/init-{init_state_index}"
        )
        noise_pair_key = (
            f"libero/{suite}/task-{task_id}/trial-{trial_index}/init-{init_state_index}"
        )
        model_noise_seed = derive_model_noise_seed(
            base_seed=self.model_noise_base_seed,
            suite=suite,
            task_id=task_id,
            trial_index=trial_index,
            init_state_index=init_state_index,
        )
        metadata = {
            "benchmark": "libero",
            "run_nonce": self.run_nonce,
            "suite": suite,
            "task_id": task_id,
            "trial_index": trial_index,
            "task_name": task_name,
            "init_state_index": init_state_index,
            "prompt": prompt,
            "noise_pair_key": noise_pair_key,
        }
        response = client.reset(
            self.server,
            timeout=min(30.0, self.request_timeout),
            episode_key=episode_key,
            model_noise_seed=model_noise_seed,
            metadata=metadata,
        )
        if not isinstance(response, Mapping) or response.get("status") != "ok":
            raise RuntimeError(f"policy-server reset failed: {response!r}")
        response_key = response.get("episode_key")
        if response_key != episode_key:
            raise RuntimeError(
                f"policy-server reset key mismatch: {response_key!r} != {episode_key!r}"
            )
        if response.get("duplicate") is not False:
            raise RuntimeError(
                "policy-server returned a duplicate reset; refusing stale episode state"
            )
        if response.get("model_noise_seed") != model_noise_seed:
            raise RuntimeError(
                "policy-server model noise seed mismatch: "
                f"{response.get('model_noise_seed')!r} != {model_noise_seed}"
            )
        self._episode_key = episode_key
        self._model_noise_seed = model_noise_seed
        self.request_count = 0
        return episode_key

    def predict(self, observation: Mapping[str, Any], prompt: str) -> np.ndarray:
        """Send the current LIBERO observation and return one controller action."""

        if self._episode_key is None:
            raise RuntimeError("reset_episode must succeed before the first prediction")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        cameras = extract_libero_cameras(observation)
        state = extract_libero_state(observation) if self.send_state else None
        payload = client.build_payload(
            head=client.encode_numpy_b64(cameras["head"]),
            left_wrist=client.encode_numpy_b64(cameras["left"]),
            right_wrist=None,
            prompt=prompt,
            state=None if state is None else state.tolist(),
        )
        response = client.post(
            self.server, "/predict", payload, timeout=self.request_timeout
        )
        if not isinstance(response, Mapping) or "action" not in response:
            raise RuntimeError(f"invalid policy-server /predict response: {response!r}")
        expected_step = self.request_count + 1
        response_step = response.get("step")
        if (
            isinstance(response_step, bool)
            or not isinstance(response_step, int)
            or response_step != expected_step
        ):
            raise RuntimeError(
                f"policy-server step mismatch: expected {expected_step}, "
                f"got {response_step!r}"
            )
        if response.get("episode_key") != self._episode_key:
            raise RuntimeError(
                "policy-server episode identity changed during LIBERO rollout"
            )
        if response.get("model_noise_seed") != self._model_noise_seed:
            raise RuntimeError(
                "policy-server model noise seed changed during LIBERO rollout"
            )
        action = policy_action_to_libero(response["action"])
        self.request_count = response_step
        return action
