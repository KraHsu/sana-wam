"""Low-overhead structured telemetry for closed-loop deployment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf

TELEMETRY_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _gripper_values(values) -> Optional[list[float]]:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    indices = {20: (9, 19), 14: (6, 13)}.get(array.size)
    if indices is None:
        return None
    return [float(array[index]) for index in indices]


def _action_space(values) -> Optional[str]:
    if values is None:
        return None
    size = np.asarray(values).size
    return {20: "eef20", 14: "qpos14"}.get(size, f"unknown{size}")


class TelemetryRecorder:
    """Append server events to JSONL and store full generated chunks as NPZ."""

    def __init__(self, cfg) -> None:
        self.enabled = bool(OmegaConf.select(cfg, "telemetry.enabled", default=False))
        self._steps_file = None
        self._episodes_file = None
        self._lines_since_flush = 0
        self._episode_number = -1
        self._episode_key: Optional[str] = None
        self._episode_token: Optional[str] = None
        self._episode_open = False
        self._previous_action = None
        if not self.enabled:
            return

        configured_dir = OmegaConf.select(
            cfg, "telemetry.output_dir", default="./telemetry/server"
        )
        output_dir = os.environ.get("SANA_WAM_TELEMETRY_DIR", str(configured_dir))
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_dir = self.output_dir / "chunks"
        self._include_chunks = bool(
            OmegaConf.select(cfg, "telemetry.include_chunks", default=True)
        )
        if self._include_chunks:
            self._chunks_dir.mkdir(parents=True, exist_ok=True)
        self._flush_every = max(
            1, int(OmegaConf.select(cfg, "telemetry.flush_every", default=32) or 32)
        )
        self.run_id = str(
            os.environ.get(
                "SANA_WAM_TELEMETRY_RUN_ID",
                OmegaConf.select(cfg, "telemetry.run_id", default="default"),
            )
        )
        self._episodes_file = (self.output_dir / "episodes.jsonl").open(
            "a", encoding="utf-8"
        )
        self._steps_file = (self.output_dir / "steps.jsonl").open("a", encoding="utf-8")

    def _write(self, handle, payload: dict) -> None:
        handle.write(json.dumps(_jsonable(payload), separators=(",", ":")) + "\n")
        self._lines_since_flush += 1
        if self._lines_since_flush >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        for handle in (self._episodes_file, self._steps_file):
            if handle is not None:
                handle.flush()
        self._lines_since_flush = 0

    def start_episode(self, context: dict, runtime_info: dict) -> None:
        if not self.enabled:
            return
        self._episode_number += 1
        self._episode_key = str(
            context.get("episode_key") or f"legacy-{self._episode_number}"
        )
        digest = hashlib.sha256(self._episode_key.encode()).hexdigest()[:10]
        self._episode_token = f"ep{self._episode_number:04d}_{digest}"
        self._episode_open = True
        self._previous_action = None
        self._write(
            self._episodes_file,
            {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "producer": "server",
                "event": "episode_start",
                "run_id": self.run_id,
                "episode_number": self._episode_number,
                "episode_key": self._episode_key,
                "episode_token": self._episode_token,
                "context": context,
                "runtime": runtime_info,
            },
        )

    def end_episode(self, runtime_info: dict, *, reason: str) -> None:
        if not self.enabled or not self._episode_open:
            return
        self._write(
            self._episodes_file,
            {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "producer": "server",
                "event": "episode_end",
                "run_id": self.run_id,
                "episode_number": self._episode_number,
                "episode_key": self._episode_key,
                "episode_token": self._episode_token,
                "reason": reason,
                "runtime": runtime_info,
            },
        )
        self._episode_open = False

    def record_step(
        self,
        *,
        request_id: int,
        state,
        action,
        latency_ms: float,
        policy_info: dict,
        runtime_info: dict,
    ) -> None:
        if not self.enabled:
            return
        state_array = (
            None if state is None else np.asarray(state, dtype=np.float32).reshape(-1)
        )
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        generation = dict(policy_info)
        chunk_arrays = {}
        for key in (
            "predicted_actions",
            "predicted_actions_normalized",
        ):
            if key in generation:
                chunk_arrays[key] = np.asarray(generation.pop(key), dtype=np.float32)
        feedback = dict(generation.get("cache_feedback") or {})
        feedback_status = feedback.get("status")
        feedback_prefix = (
            "reencoded_predicted"
            if feedback_status == "reencoded_predicted"
            else "measured_feedback"
        )
        for source, target in (
            ("actions", f"{feedback_prefix}_actions"),
            ("actions_normalized", f"{feedback_prefix}_actions_normalized"),
        ):
            if source in feedback:
                chunk_arrays[target] = np.asarray(
                    feedback.pop(source), dtype=np.float32
                )
        if feedback:
            generation["cache_feedback"] = feedback

        chunk_path = None
        if self._include_chunks and chunk_arrays:
            chunk_arrays["schema_version"] = np.asarray(
                TELEMETRY_SCHEMA_VERSION, dtype=np.int16
            )
            generation_index = int(policy_info.get("generation_index", -1))
            filename = f"{self._episode_token}_chunk{generation_index:06d}.npz"
            path = self._chunks_dir / filename
            temporary = path.with_name(f".{filename}.{os.getpid()}.tmp")
            try:
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **chunk_arrays)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            chunk_path = str(path.relative_to(self.output_dir))

        tracking_error = None
        if (
            state_array is not None
            and self._previous_action is not None
            and state_array.shape == self._previous_action.shape
        ):
            tracking_error = state_array - self._previous_action
        payload = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "producer": "server",
            "run_id": self.run_id,
            "episode_number": self._episode_number,
            "episode_key": self._episode_key,
            "request_id": int(request_id),
            "observed_proprio": state_array,
            "observed_action_space": _action_space(state_array),
            "observed_gripper": _gripper_values(state_array),
            "previous_returned_action": self._previous_action,
            "previous_tracking_error": tracking_error,
            "returned_action": action_array,
            "returned_action_space": _action_space(action_array),
            "returned_gripper": _gripper_values(action_array),
            "latency_ms": float(latency_ms),
            "policy": generation,
            "runtime": runtime_info,
            "chunk_npz": chunk_path,
        }
        self._write(self._steps_file, payload)
        self._previous_action = action_array.copy()

    def close(self) -> None:
        if not self.enabled:
            return
        self.flush()
        for handle in (self._episodes_file, self._steps_file):
            if handle is not None:
                handle.close()
        self._episodes_file = None
        self._steps_file = None
