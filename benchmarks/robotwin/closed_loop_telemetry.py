"""Client-side RoboTwin command/achievement telemetry."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

import numpy as np

TELEMETRY_SCHEMA_VERSION = 1


def _array(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _grippers(value):
    array = _array(value)
    if array is None:
        return None
    indices = {20: (9, 19), 16: (7, 15), 14: (6, 13)}.get(array.size)
    return None if indices is None else [float(array[index]) for index in indices]


def _action_space(value):
    array = _array(value)
    if array is None:
        return None
    return {20: "eef20", 16: "ee16", 14: "qpos14"}.get(
        array.size, f"unknown{array.size}"
    )


class ClientTelemetryRecorder:
    def __init__(self, *, enabled: bool, output_dir: str, flush_every: int = 32):
        env_dir = os.environ.get("ROBOTWIN_TELEMETRY_DIR")
        self.enabled = bool(enabled or env_dir)
        self._handle = None
        self._episode_handle = None
        self._line_count = 0
        self._context = {}
        self._episode_open = False
        self._step_count = 0
        self._last_success = False
        if not self.enabled:
            return
        root = Path(env_dir or output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._handle = (root / "steps.jsonl").open("a", encoding="utf-8")
        self._episode_handle = (root / "episodes.jsonl").open("a", encoding="utf-8")
        self._flush_every = max(1, int(flush_every))
        self.run_id = os.environ.get("SANA_WAM_TELEMETRY_RUN_ID", "default")
        atexit.register(self.close)

    def _write(self, handle, payload):
        handle.write(json.dumps(_jsonable(payload), separators=(",", ":")) + "\n")
        self._line_count += 1
        if self._line_count >= self._flush_every:
            self.flush()

    def start_episode(self, context: dict, reset_response: dict) -> None:
        if not self.enabled:
            return
        self.end_episode(reason="next_episode")
        self._context = dict(context)
        self._episode_open = True
        self._step_count = 0
        self._last_success = False
        self._write(
            self._episode_handle,
            {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "producer": "robotwin_client",
                "event": "episode_start",
                "run_id": self.run_id,
                "context": context,
                "reset": reset_response,
            },
        )

    def end_episode(self, *, reason: str) -> None:
        if not self.enabled or not self._episode_open:
            return
        self._write(
            self._episode_handle,
            {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "producer": "robotwin_client",
                "event": "episode_end",
                "run_id": self.run_id,
                "episode_key": self._context.get("episode_key"),
                "environment_seed": self._context.get("environment_seed"),
                "reason": reason,
                "step_count": self._step_count,
                "success": self._last_success,
            },
        )
        self._episode_open = False

    def record_step(
        self,
        *,
        env_step: int,
        prompt: str,
        pre_state,
        server_action,
        policy_action,
        sent_action,
        post_state,
        response: dict,
        success: bool,
        post_state_error: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        pre = _array(pre_state)
        predicted = _array(server_action)
        post = _array(post_state)
        returned_vs_post_error = None
        if predicted is not None and post is not None and predicted.shape == post.shape:
            delta = post - predicted
            returned_vs_post_error = {
                "l2": float(np.linalg.norm(delta)),
                "max_abs": float(np.abs(delta).max()),
            }
            if delta.size == 20:
                returned_vs_post_error.update(
                    {
                        "left_xyz_l2": float(np.linalg.norm(delta[0:3])),
                        "right_xyz_l2": float(np.linalg.norm(delta[10:13])),
                        "gripper_abs": [float(abs(delta[9])), float(abs(delta[19]))],
                    }
                )
        self._write(
            self._handle,
            {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "producer": "robotwin_client",
                "run_id": self.run_id,
                "episode_key": self._context.get("episode_key"),
                "environment_seed": self._context.get("environment_seed"),
                "env_step": int(env_step),
                "prompt": prompt,
                "pre_proprio": pre,
                "pre_action_space": _action_space(pre),
                "pre_gripper": _grippers(pre),
                "server_action": predicted,
                "server_action_space": _action_space(predicted),
                "server_action_gripper": _grippers(predicted),
                "policy_action": _array(policy_action),
                "sent_action": _array(sent_action),
                "sent_action_space": _action_space(sent_action),
                "sent_action_gripper": _grippers(sent_action),
                "post_proprio": post,
                "post_action_space": _action_space(post),
                "post_gripper": _grippers(post),
                "returned_vs_post_error": returned_vs_post_error,
                "post_state_error": post_state_error,
                "server_step": response.get("step"),
                "latency_ms": response.get("latency_ms"),
                "success": bool(success),
            },
        )
        self._step_count += 1
        self._last_success = bool(success)

    def flush(self) -> None:
        for handle in (self._handle, self._episode_handle):
            if handle is not None:
                handle.flush()
        self._line_count = 0

    def close(self) -> None:
        if not self.enabled:
            return
        self.end_episode(reason="shutdown")
        self.flush()
        for name in ("_handle", "_episode_handle"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)
