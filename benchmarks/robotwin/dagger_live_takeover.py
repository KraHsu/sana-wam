"""Strict live-scene DAgger recorder for the RoboTwin ``adjust_bottle`` task.

The module is inert unless ``ROBOTWIN_DAGGER_PLAN`` points at a validated JSON
plan.  An enabled recorder keeps the exact 113 observations seen by the policy,
withholds a selected generation-boundary action, and lets RoboTwin's scripted
expert take over in the same live scene.  Artifacts remain in physical EEF20
units; normalization belongs to the training loader and is pinned by the
baseline ``action_stats.npy`` digest in the plan.

The adapter-facing API is intentionally small::

    recorder = LiveTakeoverRecorder.from_env(server_info)
    recorder.start_episode(context)
    outcome = recorder.process_step(...)
    recorder.close()

``process_step`` is called after ``/predict`` and before ``TASK_ENV.take_action``.
When it returns a non-None outcome, ``consumed`` is true and the learner action
must not be applied.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Mapping

import numpy as np


PLAN_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
PLAN_KIND = "robotwin_live_takeover_plan"
SOURCE_KIND = "fresh_on_policy_history113_dagger"
HISTORY_LEN = 113
ACTION_TOKENS_PER_CHUNK = 28
ACTION_DIM = 20
VIDEO_STRIDE = 4
TEMPORAL_COMPRESSION = 4
AR_FRAME_CHUNK_SIZE = 2
VIDEO_FRAMES = 29
EXPERT_SAVE_FREQ = 15
LABEL_GRIPPER_CLOSED_THRESHOLD = 0.5
LABEL_BOTTLE_LIFT_THRESHOLD_M = 0.05
ANCHOR_TRIGGER_KIND = "active_eef_bottle_proximity"
TRAIN_SEED_MIN = 200_000
TRAIN_SEED_MAX_EXCLUSIVE = 300_000
ENVIRONMENT_SEED_INDEX = 1
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 320
CAMERA_LAYOUT = ("head_camera", "left_camera", "right_camera")
UPSTREAM_STEP_LIMIT = 400
CAMERA_NAMES = ("head", "left", "right")
BASELINE_CHECKPOINT_SHA256 = (
    "aea6b58c9107aeeeffba561f39ffedfc3f8fd8420a4abfcab9038c8cdd4a129d"
)
BASELINE_ACTION_STATS_SHA256 = (
    "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"
)
_ANCHOR_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LiveTakeoverPlanError(ValueError):
    """Raised when an enabled collection plan or deployment is not exact."""


@dataclass(frozen=True)
class TakeoverOutcome:
    """Result returned when the recorder consumes the current learner action."""

    consumed: bool
    success: bool
    status: str
    anchor_id: str
    sample_path: str | None = None


def _plain_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiveTakeoverPlanError(f"{label} must be a plain integer, got {value!r}")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiveTakeoverPlanError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    text = _require_string(value, label)
    if _SHA256_RE.fullmatch(text) is None:
        raise LiveTakeoverPlanError(f"{label} must be lowercase SHA-256 hex")
    return text


def _expect_keys(
    value: Any,
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> dict:
    if not isinstance(value, dict):
        raise LiveTakeoverPlanError(f"{label} must be a JSON object")
    optional = optional or set()
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise LiveTakeoverPlanError(f"{label} missing required fields: {missing}")
    if unknown:
        raise LiveTakeoverPlanError(f"{label} has unknown fields: {unknown}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, relative_to: Path | None = None) -> dict:
    resolved = path.resolve()
    stored_path = (
        str(resolved.relative_to(relative_to.resolve()))
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": stored_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _physical_eef20(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (ACTION_DIM,):
        raise ValueError(f"{label} must have shape ({ACTION_DIM},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array.copy()


def _rgb_jpeg(value: Any, label: str) -> bytes:
    import io

    from PIL import Image

    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"{label} must be an HxWx3 uint8 RGB image")
    stream = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(stream, format="JPEG", quality=95)
    return stream.getvalue()


def _jpeg_bytes(value: Any, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{label} must be JPEG bytes")
    payload = bytes(value)
    if (
        len(payload) < 4
        or not payload.startswith(b"\xff\xd8")
        or not payload.endswith(b"\xff\xd9")
    ):
        raise ValueError(f"{label} is not a complete JPEG payload")
    try:
        import cv2

        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ValueError(f"{label} JPEG validation failed: {exc}") from exc
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(f"{label} is not a decodable three-channel JPEG")
    return payload


def _normalize_policy_info(value: Any, *, env_step: int) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("policy_info must be a mapping")
    required = {
        "policy_step",
        "generation_index",
        "chunk_offset",
        "buffer_remaining",
        "generated",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"policy_info missing fields: {missing}")

    policy_step = _plain_int(value["policy_step"], "policy_info.policy_step")
    generation = _plain_int(value["generation_index"], "policy_info.generation_index")
    chunk_offset = _plain_int(value["chunk_offset"], "policy_info.chunk_offset")
    buffer_remaining = _plain_int(
        value["buffer_remaining"], "policy_info.buffer_remaining"
    )
    generated = value["generated"]
    if not isinstance(generated, bool):
        raise ValueError("policy_info.generated must be boolean")
    if policy_step != env_step:
        raise ValueError(
            f"policy_step mismatch: policy={policy_step}, environment={env_step}"
        )
    expected_generation, expected_offset = divmod(env_step, ACTION_TOKENS_PER_CHUNK)
    if generation != expected_generation or chunk_offset != expected_offset:
        raise ValueError(
            "non-contiguous policy generation metadata: "
            f"got generation={generation} offset={chunk_offset}, "
            f"expected generation={expected_generation} offset={expected_offset}"
        )
    if generated != (chunk_offset == 0):
        raise ValueError(
            f"generated={generated} is inconsistent with chunk_offset={chunk_offset}"
        )
    expected_remaining = ACTION_TOKENS_PER_CHUNK - chunk_offset - 1
    if buffer_remaining != expected_remaining:
        raise ValueError(
            f"buffer_remaining={buffer_remaining}, expected {expected_remaining}"
        )
    chunk_len = value.get("chunk_len")
    if generated:
        if _plain_int(chunk_len, "policy_info.chunk_len") != ACTION_TOKENS_PER_CHUNK:
            raise ValueError(
                f"policy_info.chunk_len must be {ACTION_TOKENS_PER_CHUNK} at a boundary"
            )
    elif chunk_len is not None and (
        _plain_int(chunk_len, "policy_info.chunk_len") != ACTION_TOKENS_PER_CHUNK
    ):
        raise ValueError(f"policy_info.chunk_len must be {ACTION_TOKENS_PER_CHUNK}")

    return {
        "policy_step": policy_step,
        "generation_index": generation,
        "chunk_offset": chunk_offset,
        "buffer_remaining": buffer_remaining,
        "generated": generated,
        "chunk_len": ACTION_TOKENS_PER_CHUNK,
    }


class LiveTakeoverRecorder:
    """Collect one strict live-takeover sample for each plan entry."""

    def __init__(self) -> None:
        self._enabled = False
        self._closed = False
        self._plan: dict | None = None
        self._plan_path: Path | None = None
        self._plan_sha256: str | None = None
        self._output_dir: Path | None = None
        self._server_info: dict = {}
        self._step_limits_artifact: dict = {}
        self._entries: list[dict] = []
        self._entries_by_seed: dict[int, dict] = {}
        self._results: dict[str, dict] = {}
        self._active_entry: dict | None = None
        self._episode_context: dict = {}
        self._history: deque[dict] = deque(maxlen=HISTORY_LEN)
        self._anchor_trigger_observations: list[dict] = []
        self._next_env_step = 0

    @classmethod
    def from_env(cls, server_info: Mapping[str, Any] | None) -> "LiveTakeoverRecorder":
        """Build a recorder, or a dependency-free no-op when the plan is unset."""
        recorder = cls()
        raw_path = os.environ.get("ROBOTWIN_DAGGER_PLAN", "").strip()
        if not raw_path:
            return recorder

        plan_path = Path(raw_path).expanduser().resolve()
        if not plan_path.is_file():
            raise LiveTakeoverPlanError(
                f"ROBOTWIN_DAGGER_PLAN is not a file: {plan_path}"
            )
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LiveTakeoverPlanError(
                f"cannot read DAgger plan {plan_path}: {exc}"
            ) from exc

        recorder._configure(plan_path, plan, server_info)
        return recorder

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _configure(
        self,
        plan_path: Path,
        raw_plan: Any,
        server_info: Mapping[str, Any] | None,
    ) -> None:
        required = {
            "schema_version",
            "kind",
            "source_kind",
            "output_dir",
            "task_name",
            "task_config",
            "policy_history_len",
            "action_tokens_per_chunk",
            "video_stride",
            "temporal_compression",
            "ar_frame_chunk_size",
            "expert_save_freq",
            "cache_feedback_mode",
            "checkpoint_sha256",
            "deploy_config_sha256",
            "deploy_overrides_sha256",
            "prompt_manifest_sha256",
            "episode_noise_mode",
            "episode_noise_base_seed",
            "environment_seed_index",
            "action_stats_path",
            "action_stats_sha256",
            "step_limit_source",
            "repo_step_limit_override_count",
            "upstream_step_limit",
            "episodes",
        }
        plan = _expect_keys(
            raw_plan,
            required=required,
            optional={"metadata", "anchor_trigger", "full_expert_suffix"},
            label="DAgger plan",
        )
        if _plain_int(plan["schema_version"], "schema_version") != PLAN_SCHEMA_VERSION:
            raise LiveTakeoverPlanError(f"schema_version must be {PLAN_SCHEMA_VERSION}")
        if plan["kind"] != PLAN_KIND or plan["source_kind"] != SOURCE_KIND:
            raise LiveTakeoverPlanError(
                f"plan kind/source_kind must be {PLAN_KIND!r}/{SOURCE_KIND!r}"
            )
        if plan["task_name"] != "adjust_bottle":
            raise LiveTakeoverPlanError(
                "live expert currently supports adjust_bottle only"
            )
        _require_string(plan["task_config"], "task_config")
        exact_ints = {
            "policy_history_len": HISTORY_LEN,
            "action_tokens_per_chunk": ACTION_TOKENS_PER_CHUNK,
            "video_stride": VIDEO_STRIDE,
            "temporal_compression": TEMPORAL_COMPRESSION,
            "ar_frame_chunk_size": AR_FRAME_CHUNK_SIZE,
            "expert_save_freq": EXPERT_SAVE_FREQ,
            "repo_step_limit_override_count": 0,
            "environment_seed_index": ENVIRONMENT_SEED_INDEX,
            "upstream_step_limit": UPSTREAM_STEP_LIMIT,
        }
        for field, expected in exact_ints.items():
            actual = _plain_int(plan[field], field)
            if actual != expected:
                raise LiveTakeoverPlanError(f"{field} must be {expected}, got {actual}")
        if plan["step_limit_source"] != "robotwin_upstream_default":
            raise LiveTakeoverPlanError(
                "step_limit_source must be 'robotwin_upstream_default'"
            )
        cache_feedback_mode = _require_string(
            plan["cache_feedback_mode"], "cache_feedback_mode"
        )
        if cache_feedback_mode != "predicted":
            raise LiveTakeoverPlanError(
                "fresh history DAgger requires cache_feedback_mode='predicted'"
            )
        expected_checkpoint_sha = _require_sha256(
            plan["checkpoint_sha256"], "checkpoint_sha256"
        )
        if expected_checkpoint_sha != BASELINE_CHECKPOINT_SHA256:
            raise LiveTakeoverPlanError(
                "checkpoint_sha256 must be the fixed baseline SHA-256 "
                f"{BASELINE_CHECKPOINT_SHA256}"
            )
        expected_deploy_config_sha = _require_sha256(
            plan["deploy_config_sha256"], "deploy_config_sha256"
        )
        expected_deploy_overrides_sha = _require_sha256(
            plan["deploy_overrides_sha256"], "deploy_overrides_sha256"
        )
        _require_sha256(plan["prompt_manifest_sha256"], "prompt_manifest_sha256")
        noise_mode = _require_string(plan["episode_noise_mode"], "episode_noise_mode")
        if noise_mode != "paired":
            raise LiveTakeoverPlanError("episode_noise_mode must be 'paired'")
        noise_base_seed = _plain_int(
            plan["episode_noise_base_seed"], "episode_noise_base_seed"
        )
        if noise_base_seed < 0:
            raise LiveTakeoverPlanError("episode_noise_base_seed must be non-negative")

        stats_path = Path(plan["action_stats_path"]).expanduser()
        if not stats_path.is_absolute():
            stats_path = plan_path.parent / stats_path
        stats_path = stats_path.resolve()
        expected_stats_sha = _require_sha256(
            plan["action_stats_sha256"], "action_stats_sha256"
        )
        if expected_stats_sha != BASELINE_ACTION_STATS_SHA256:
            raise LiveTakeoverPlanError(
                "action_stats_sha256 must be the fixed baseline SHA-256 "
                f"{BASELINE_ACTION_STATS_SHA256}"
            )
        if not stats_path.is_file():
            raise LiveTakeoverPlanError(
                f"action_stats_path is not a file: {stats_path}"
            )
        actual_stats_sha = _sha256_file(stats_path)
        if actual_stats_sha != expected_stats_sha:
            raise LiveTakeoverPlanError(
                "action_stats SHA-256 mismatch: "
                f"expected {expected_stats_sha}, got {actual_stats_sha}"
            )

        anchor_trigger = self._validate_anchor_trigger(plan.get("anchor_trigger"), plan)
        full_expert_suffix = self._validate_full_expert_suffix(
            plan.get("full_expert_suffix")
        )
        plan = dict(plan)
        if anchor_trigger is not None:
            plan["anchor_trigger"] = anchor_trigger
        if full_expert_suffix is not None:
            plan["full_expert_suffix"] = full_expert_suffix

        normalized_server_info = self._validate_server_info(
            server_info,
            checkpoint_sha256=expected_checkpoint_sha,
            deploy_config_sha256=expected_deploy_config_sha,
            deploy_overrides_sha256=expected_deploy_overrides_sha,
            cache_feedback_mode=cache_feedback_mode,
            episode_noise_mode=noise_mode,
            episode_noise_base_seed=noise_base_seed,
        )
        entries = self._validate_entries(plan["episodes"], plan)
        step_limits_artifact = self._validate_step_limit_overrides()

        output_dir = Path(
            _require_string(plan["output_dir"], "output_dir")
        ).expanduser()
        if not output_dir.is_absolute():
            output_dir = plan_path.parent / output_dir
        output_dir = output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise LiveTakeoverPlanError(
                f"output_dir must be absent or empty for an immutable run: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "samples").mkdir()

        plan = dict(plan)
        plan["action_stats_path"] = str(stats_path)
        self._enabled = True
        self._plan = plan
        self._plan_path = plan_path
        self._plan_sha256 = _sha256_file(plan_path)
        self._output_dir = output_dir
        self._server_info = normalized_server_info
        self._step_limits_artifact = step_limits_artifact
        self._entries = entries
        self._entries_by_seed = {entry["environment_seed"]: entry for entry in entries}

    @staticmethod
    def _validate_full_expert_suffix(raw_config: Any) -> dict | None:
        if raw_config is None:
            return None
        config = _expect_keys(
            raw_config,
            required={"enabled"},
            label="full_expert_suffix",
        )
        if config["enabled"] is not True:
            raise LiveTakeoverPlanError(
                "full_expert_suffix.enabled must be true when configured"
            )
        return {"enabled": True}

    @staticmethod
    def _validate_anchor_trigger(
        raw_trigger: Any, plan: Mapping[str, Any]
    ) -> dict | None:
        if raw_trigger is None:
            return None
        trigger = _expect_keys(
            raw_trigger,
            required={
                "kind",
                "min_env_step",
                "max_env_step",
                "distance_m_at_most",
            },
            label="anchor_trigger",
        )
        if trigger["kind"] != ANCHOR_TRIGGER_KIND:
            raise LiveTakeoverPlanError(
                f"anchor_trigger.kind must be {ANCHOR_TRIGGER_KIND!r}"
            )
        minimum = _plain_int(trigger["min_env_step"], "anchor_trigger.min_env_step")
        maximum = _plain_int(trigger["max_env_step"], "anchor_trigger.max_env_step")
        upstream_limit = _plain_int(plan["upstream_step_limit"], "upstream_step_limit")
        if (
            minimum < HISTORY_LEN - 1
            or minimum % ACTION_TOKENS_PER_CHUNK != 0
            or maximum % ACTION_TOKENS_PER_CHUNK != 0
            or maximum < minimum
            or maximum >= upstream_limit
        ):
            raise LiveTakeoverPlanError(
                "anchor_trigger min/max steps must be ordered generation boundaries "
                f"from {HISTORY_LEN - 1} through {upstream_limit - 1}"
            )
        distance = trigger["distance_m_at_most"]
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not np.isfinite(distance)
            or not 0.0 < float(distance) <= 0.5
        ):
            raise LiveTakeoverPlanError(
                "anchor_trigger.distance_m_at_most must be finite in (0, 0.5]"
            )
        return {
            "kind": ANCHOR_TRIGGER_KIND,
            "min_env_step": minimum,
            "max_env_step": maximum,
            "distance_m_at_most": float(distance),
        }

    @staticmethod
    def _validate_step_limit_overrides() -> dict:
        configured = os.environ.get("ROBOTWIN_STEP_LIMITS_PATH", "").strip()
        if configured:
            raise LiveTakeoverPlanError(
                "ROBOTWIN_STEP_LIMITS_PATH must be unset for upstream-default collection"
            )
        path = Path(__file__).resolve().with_name("step_limits.yml")
        if not path.is_file():
            raise LiveTakeoverPlanError(
                f"repository step-limit file is missing: {path}"
            )
        try:
            import yaml

            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise LiveTakeoverPlanError(
                f"cannot validate repository step-limit file {path}: {exc}"
            ) from exc
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise LiveTakeoverPlanError(
                f"repository step-limit file must be a mapping: {path}"
            )
        if parsed:
            raise LiveTakeoverPlanError(
                "repository step-limit overrides must have zero active mappings, "
                f"found {sorted(str(key) for key in parsed)}"
            )
        return _artifact(path)

    @staticmethod
    def _validate_server_info(
        server_info: Mapping[str, Any] | None,
        *,
        checkpoint_sha256: str,
        deploy_config_sha256: str,
        deploy_overrides_sha256: str,
        cache_feedback_mode: str,
        episode_noise_mode: str,
        episode_noise_base_seed: int,
    ) -> dict:
        if not isinstance(server_info, Mapping):
            raise LiveTakeoverPlanError("enabled DAgger collection requires /info data")
        runtime = server_info.get("inference_runtime")
        if not isinstance(runtime, Mapping):
            raise LiveTakeoverPlanError("/info.inference_runtime is missing")
        identity = runtime.get("deployment_identity")
        if not isinstance(identity, Mapping):
            raise LiveTakeoverPlanError(
                "/info.inference_runtime.deployment_identity is missing"
            )
        exact = {
            "history_len": HISTORY_LEN,
            "num_frames": HISTORY_LEN,
            "video_num_frames": VIDEO_FRAMES,
            "video_stride": VIDEO_STRIDE,
            "action_tokens_per_chunk": ACTION_TOKENS_PER_CHUNK,
            "ar_frame_chunk_size": AR_FRAME_CHUNK_SIZE,
        }
        for field, expected in exact.items():
            if (
                _plain_int(identity.get(field), f"deployment_identity.{field}")
                != expected
            ):
                raise LiveTakeoverPlanError(
                    f"deployment_identity.{field} must be {expected}"
                )
        if identity.get("action_mode") != "eef":
            raise LiveTakeoverPlanError("deployment_identity.action_mode must be 'eef'")
        if identity.get("checkpoint_sha256") != checkpoint_sha256:
            raise LiveTakeoverPlanError(
                "deployment checkpoint SHA-256 does not match plan"
            )
        if identity.get("deploy_config_sha256") != deploy_config_sha256:
            raise LiveTakeoverPlanError("deployment config SHA-256 does not match plan")
        if identity.get("deploy_overrides_sha256") != deploy_overrides_sha256:
            raise LiveTakeoverPlanError(
                "deployment overrides SHA-256 does not match plan"
            )
        if identity.get("cache_feedback_mode") != cache_feedback_mode:
            raise LiveTakeoverPlanError(
                "deployment cache_feedback_mode does not match plan"
            )
        if runtime.get("cache_feedback_mode") != cache_feedback_mode:
            raise LiveTakeoverPlanError(
                "runtime cache_feedback_mode does not match plan"
            )
        if identity.get("episode_noise_mode") != episode_noise_mode:
            raise LiveTakeoverPlanError(
                "deployment episode_noise_mode does not match plan"
            )
        if (
            _plain_int(
                identity.get("episode_noise_base_seed"),
                "deployment_identity.episode_noise_base_seed",
            )
            != episode_noise_base_seed
        ):
            raise LiveTakeoverPlanError(
                "deployment episode_noise_base_seed does not match plan"
            )
        if runtime.get("episode_noise_mode") != episode_noise_mode:
            raise LiveTakeoverPlanError(
                "runtime episode_noise_mode does not match plan"
            )
        if (
            _plain_int(
                runtime.get("episode_noise_base_seed"),
                "inference_runtime.episode_noise_base_seed",
            )
            != episode_noise_base_seed
        ):
            raise LiveTakeoverPlanError(
                "runtime episode_noise_base_seed does not match plan"
            )

        policy_config = server_info.get("policy_config")
        if (
            not isinstance(policy_config, Mapping)
            or policy_config.get("execute_horizon") is not None
        ):
            raise LiveTakeoverPlanError(
                "strict collection requires policy.execute_horizon=null"
            )
        async_info = server_info.get("async_inference")
        if (
            not isinstance(async_info, Mapping)
            or async_info.get("enabled") is not False
        ):
            raise LiveTakeoverPlanError(
                "strict collection requires async inference disabled"
            )
        return json.loads(json.dumps(server_info))

    @staticmethod
    def _validate_entries(raw_entries: Any, plan: Mapping[str, Any]) -> list[dict]:
        if not isinstance(raw_entries, list) or not raw_entries:
            raise LiveTakeoverPlanError("episodes must be a non-empty list")
        required = {
            "anchor_id",
            "episode_index",
            "environment_seed",
            "prompt",
            "prompt_sha256",
            "anchor_env_step",
            "generation_index",
        }
        control_fields = {
            "control_success",
            "control_step_count",
            "control_anchor_observed",
        }
        entries: list[dict] = []
        seen_ids: set[str] = set()
        seen_seeds: set[int] = set()
        seen_episode_indices: set[int] = set()
        for ordinal, raw in enumerate(raw_entries):
            entry = _expect_keys(
                raw,
                required=required,
                optional={"metadata", *control_fields},
                label=f"episodes[{ordinal}]",
            )
            present_control_fields = control_fields.intersection(entry)
            if present_control_fields and present_control_fields != control_fields:
                raise LiveTakeoverPlanError(
                    f"episodes[{ordinal}] control outcome fields must be all-or-none"
                )
            anchor_id = _require_string(
                entry["anchor_id"], f"episodes[{ordinal}].anchor_id"
            )
            if _ANCHOR_ID_RE.fullmatch(anchor_id) is None:
                raise LiveTakeoverPlanError(
                    f"episodes[{ordinal}].anchor_id contains unsafe characters"
                )
            if anchor_id in seen_ids:
                raise LiveTakeoverPlanError(f"duplicate anchor_id {anchor_id!r}")
            episode_index = _plain_int(
                entry["episode_index"], f"episodes[{ordinal}].episode_index"
            )
            seed = _plain_int(
                entry["environment_seed"], f"episodes[{ordinal}].environment_seed"
            )
            if not TRAIN_SEED_MIN <= seed < TRAIN_SEED_MAX_EXCLUSIVE:
                raise LiveTakeoverPlanError(
                    f"environment_seed {seed} is outside disjoint training range "
                    f"[{TRAIN_SEED_MIN}, {TRAIN_SEED_MAX_EXCLUSIVE})"
                )
            if seed // 100_000 - 1 != plan["environment_seed_index"]:
                raise LiveTakeoverPlanError(
                    f"environment_seed {seed} does not belong to environment_seed_index="
                    f"{plan['environment_seed_index']}"
                )
            if seed in seen_seeds:
                raise LiveTakeoverPlanError(
                    "one live scene can contain only one takeover anchor; "
                    f"duplicate environment_seed {seed}"
                )
            if episode_index in seen_episode_indices:
                raise LiveTakeoverPlanError(f"duplicate episode_index {episode_index}")
            prompt = _require_string(entry["prompt"], f"episodes[{ordinal}].prompt")
            prompt_sha = _require_sha256(
                entry["prompt_sha256"], f"episodes[{ordinal}].prompt_sha256"
            )
            actual_prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if prompt_sha != actual_prompt_sha:
                raise LiveTakeoverPlanError(
                    f"episodes[{ordinal}] prompt SHA-256 mismatch"
                )
            anchor_step = _plain_int(
                entry["anchor_env_step"], f"episodes[{ordinal}].anchor_env_step"
            )
            generation = _plain_int(
                entry["generation_index"], f"episodes[{ordinal}].generation_index"
            )
            if anchor_step < HISTORY_LEN - 1:
                raise LiveTakeoverPlanError(
                    f"anchor_env_step must be >= {HISTORY_LEN - 1} (no left padding)"
                )
            if anchor_step != generation * ACTION_TOKENS_PER_CHUNK:
                raise LiveTakeoverPlanError(
                    "anchor must be an exact generation boundary: "
                    f"env_step={anchor_step}, generation={generation}"
                )
            anchor_trigger = plan.get("anchor_trigger")
            if (
                anchor_trigger is not None
                and anchor_step != anchor_trigger["max_env_step"]
            ):
                raise LiveTakeoverPlanError(
                    "triggered plan entries must use anchor_env_step as the "
                    "anchor_trigger.max_env_step deadline"
                )
            control_outcome = {}
            if present_control_fields:
                control_success = entry["control_success"]
                anchor_observed = entry["control_anchor_observed"]
                if not isinstance(control_success, bool) or not isinstance(
                    anchor_observed, bool
                ):
                    raise LiveTakeoverPlanError(
                        f"episodes[{ordinal}] control outcome flags must be boolean"
                    )
                control_step_count = _plain_int(
                    entry["control_step_count"],
                    f"episodes[{ordinal}].control_step_count",
                )
                if not 1 <= control_step_count <= plan["upstream_step_limit"]:
                    raise LiveTakeoverPlanError(
                        f"episodes[{ordinal}].control_step_count is outside RoboTwin "
                        "default limits"
                    )
                if anchor_observed != (control_step_count > anchor_step):
                    raise LiveTakeoverPlanError(
                        f"episodes[{ordinal}] control anchor observation is inconsistent"
                    )
                if not anchor_observed and not control_success:
                    raise LiveTakeoverPlanError(
                        f"episodes[{ordinal}] failed control ended before its anchor"
                    )
                control_outcome = {
                    "control_success": control_success,
                    "control_step_count": control_step_count,
                    "control_anchor_observed": anchor_observed,
                }
            normalized = dict(entry)
            normalized.update(
                {
                    "ordinal": ordinal,
                    "anchor_id": anchor_id,
                    "episode_index": episode_index,
                    "environment_seed": seed,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha,
                    "anchor_env_step": anchor_step,
                    "generation_index": generation,
                    "task_name": plan["task_name"],
                    "task_config": plan["task_config"],
                    **control_outcome,
                }
            )
            entries.append(normalized)
            seen_ids.add(anchor_id)
            seen_seeds.add(seed)
            seen_episode_indices.add(episode_index)
        return entries

    def start_episode(self, context: Mapping[str, Any]) -> None:
        """Bind subsequent steps to one exact planned scene and prompt."""
        if not self._enabled:
            return
        if self._closed:
            raise RuntimeError("LiveTakeoverRecorder is closed")
        if self._active_entry is not None and (
            self._active_entry["anchor_id"] not in self._results
        ):
            self._record_exception_result(
                self._active_entry,
                "episode_replaced_before_anchor",
                observed_env_step=self._next_env_step - 1,
            )
        if not isinstance(context, Mapping):
            raise LiveTakeoverPlanError("episode context must be a mapping")
        seed = _plain_int(context.get("environment_seed"), "context.environment_seed")
        entry = self._entries_by_seed.get(seed)
        if entry is None:
            raise LiveTakeoverPlanError(
                f"accepted scene seed {seed} has no live-takeover plan entry"
            )
        expected = {
            "task_name": entry["task_name"],
            "task_config": entry["task_config"],
            "episode_index": entry["episode_index"],
            "prompt": entry["prompt"],
            "prompt_sha256": entry["prompt_sha256"],
            "prompt_manifest_sha256": self._plan["prompt_manifest_sha256"],
            "noise_pair_key": (
                f"robotwin/{entry['task_name']}/{entry['task_config']}/"
                f"scene-{entry['environment_seed']}"
            ),
        }
        for field, expected_value in expected.items():
            if context.get(field) != expected_value:
                raise LiveTakeoverPlanError(
                    f"context.{field}={context.get(field)!r}, expected {expected_value!r}"
                )
        episode_key = _require_string(context.get("episode_key"), "context.episode_key")
        if entry["anchor_id"] in self._results:
            raise LiveTakeoverPlanError(
                f"anchor {entry['anchor_id']!r} already has a result"
            )
        self._active_entry = entry
        self._episode_context = dict(context)
        self._episode_context["episode_key"] = episode_key
        self._history.clear()
        self._anchor_trigger_observations.clear()
        self._next_env_step = 0

    def process_step(
        self,
        task_env: Any,
        env_step: int,
        prompt: str,
        camera_jpegs: Mapping[str, Any],
        pre_state: Any,
        policy_action: Any,
        policy_info: Mapping[str, Any],
        extract_state: Callable[[Any], Any],
    ) -> TakeoverOutcome | None:
        """Record one pre-action step and run the selected same-scene takeover."""
        if not self._enabled:
            return None
        if self._closed:
            raise RuntimeError("LiveTakeoverRecorder is closed")
        entry = self._active_entry
        if entry is None:
            raise LiveTakeoverPlanError("start_episode() must precede process_step()")
        if entry["anchor_id"] in self._results:
            return None

        try:
            step = _plain_int(env_step, "env_step")
            if step != self._next_env_step:
                raise ValueError(
                    f"environment step gap: got {step}, expected {self._next_env_step}"
                )
            if getattr(task_env, "task_name", entry["task_name"]) != entry["task_name"]:
                raise ValueError("TASK_ENV.task_name does not match takeover plan")
            if getattr(task_env, "step_lim", None) != self._plan["upstream_step_limit"]:
                raise ValueError(
                    "TASK_ENV.step_lim does not match the pinned RoboTwin upstream limit: "
                    f"{getattr(task_env, 'step_lim', None)!r} != "
                    f"{self._plan['upstream_step_limit']}"
                )
            if prompt != entry["prompt"]:
                raise ValueError("prompt changed during strict live-takeover replay")
            if not isinstance(camera_jpegs, Mapping):
                raise ValueError("camera_jpegs must be a mapping")
            cameras = {
                name: _jpeg_bytes(camera_jpegs.get(name), f"camera_jpegs.{name}")
                for name in CAMERA_NAMES
            }
            state = _physical_eef20(pre_state, "pre_state")
            action = _physical_eef20(policy_action, "policy_action")
            normalized_policy = _normalize_policy_info(policy_info, env_step=step)

            if self._history:
                self._history[-1]["applied"] = True
            self._history.append(
                {
                    "env_step": step,
                    "prompt": prompt,
                    "cameras": cameras,
                    "pre_state": state,
                    "policy_action": action,
                    "policy": normalized_policy,
                    "applied": False,
                }
            )
            self._next_env_step += 1

            anchor_trigger = self._plan.get("anchor_trigger")
            if anchor_trigger is not None:
                return self._process_proximity_trigger(
                    task_env,
                    entry,
                    step,
                    state,
                    normalized_policy,
                    extract_state,
                )

            anchor_step = entry["anchor_env_step"]
            if step < anchor_step:
                return None
            if step != anchor_step:
                raise ValueError(
                    f"planned anchor {anchor_step} was not consumed before step {step}"
                )
            return self._consume_anchor(task_env, entry, state, extract_state)
        except Exception as exc:
            result = self._record_exception_result(
                entry,
                f"{type(exc).__name__}: {exc}",
                observed_env_step=env_step if isinstance(env_step, int) else None,
            )
            self._terminate_episode(task_env, success=False)
            return TakeoverOutcome(
                consumed=True,
                success=False,
                status="exception",
                anchor_id=entry["anchor_id"],
                sample_path=(result.get("sample") or {})
                .get("artifact", {})
                .get("path"),
            )

    def _process_proximity_trigger(
        self,
        task_env: Any,
        entry: Mapping[str, Any],
        step: int,
        state: np.ndarray,
        policy: Mapping[str, Any],
        extract_state: Callable[[Any], Any],
    ) -> TakeoverOutcome | None:
        trigger = self._plan["anchor_trigger"]
        minimum = trigger["min_env_step"]
        maximum = trigger["max_env_step"]
        if step < minimum:
            return None
        if step > maximum:
            raise ValueError(
                f"proximity trigger deadline {maximum} passed without a result"
            )
        if policy["chunk_offset"] != 0:
            return None

        active_arm = "right" if int(getattr(task_env, "qpose_tag", 0)) == 1 else "left"
        eef_position = (
            state[10:13].astype(np.float64)
            if active_arm == "right"
            else state[0:3].astype(np.float64)
        )
        bottle_position = self._bottle_position(task_env)
        distance = float(np.linalg.norm(eef_position - bottle_position))
        observation = {
            "env_step": step,
            "generation_index": policy["generation_index"],
            "active_arm": active_arm,
            "distance_m": distance,
            "eef_position": eef_position.tolist(),
            "bottle_position": bottle_position.tolist(),
        }
        self._anchor_trigger_observations.append(observation)

        if distance <= trigger["distance_m_at_most"]:
            runtime_entry = dict(entry)
            runtime_entry.update(
                {
                    "planned_anchor_env_step": entry["anchor_env_step"],
                    "anchor_env_step": step,
                    "generation_index": policy["generation_index"],
                    "anchor_trigger_observation": observation,
                }
            )
            return self._consume_anchor(task_env, runtime_entry, state, extract_state)
        if step == maximum:
            return self._record_anchor_not_triggered(
                task_env, entry, active_arm=active_arm
            )
        return None

    def _record_anchor_not_triggered(
        self,
        task_env: Any,
        entry: Mapping[str, Any],
        *,
        active_arm: str,
    ) -> TakeoverOutcome:
        distances = [
            observation["distance_m"]
            for observation in self._anchor_trigger_observations
        ]
        result = self._base_result(entry)
        result.update(
            {
                "status": "anchor_not_triggered",
                "anchor_env_step": None,
                "generation_index": None,
                "label_valid": False,
                "planner_failed": False,
                "expert_failed": False,
                "exception": None,
                "observed_env_step": self._next_env_step - 1,
                "expert_capture_calls": 0,
                "expert_state_count": 0,
                "expert_label_count": 0,
                "active_arm": active_arm,
                "label_gripper_closed": False,
                "label_first_close_index": None,
                "label_bottle_lift_delta": None,
                "label_semantic_valid": False,
                "anchor_triggered": False,
                "anchor_trigger_candidate_count": len(distances),
                "anchor_trigger_min_distance_m": min(distances),
                "anchor_trigger_distance_m": None,
                "sample": None,
            }
        )
        self._append_result(result)
        self._terminate_episode(task_env, success=False)
        return TakeoverOutcome(
            consumed=True,
            success=False,
            status="anchor_not_triggered",
            anchor_id=entry["anchor_id"],
        )

    def _validate_anchor_history(self, entry: Mapping[str, Any]) -> list[dict]:
        history = list(self._history)
        if len(history) != HISTORY_LEN:
            raise ValueError(
                f"anchor requires exactly {HISTORY_LEN} observations, got {len(history)}"
            )
        anchor_step = entry["anchor_env_step"]
        expected_steps = list(range(anchor_step - HISTORY_LEN + 1, anchor_step + 1))
        actual_steps = [row["env_step"] for row in history]
        if actual_steps != expected_steps:
            raise ValueError("113-step history is not contiguous")

        generation = entry["generation_index"]
        expected_generations = [
            generation - 4 + index // ACTION_TOKENS_PER_CHUNK
            for index in range(HISTORY_LEN - 1)
        ] + [generation]
        expected_offsets = [
            index % ACTION_TOKENS_PER_CHUNK for index in range(HISTORY_LEN - 1)
        ] + [0]
        actual_generations = [row["policy"]["generation_index"] for row in history]
        actual_offsets = [row["policy"]["chunk_offset"] for row in history]
        if actual_generations != expected_generations:
            raise ValueError("history does not contain generations G-4 through G")
        if actual_offsets != expected_offsets:
            raise ValueError("history does not contain four full chunk offset cycles")
        expected_applied = [True] * (HISTORY_LEN - 1) + [False]
        if [row["applied"] for row in history] != expected_applied:
            raise ValueError("history applied mask must be 112 true values then false")
        if history[-1]["policy"]["generated"] is not True:
            raise ValueError("anchor is not an exact generated=true boundary")

        prefix = history[ACTION_TOKENS_PER_CHUNK : HISTORY_LEN - 1]
        if len(prefix) != 3 * ACTION_TOKENS_PER_CHUNK:
            raise ValueError("learner prefix is not exactly three complete chunks")
        for index, row in enumerate(prefix):
            expected_generation = generation - 3 + index // ACTION_TOKENS_PER_CHUNK
            expected_offset = index % ACTION_TOKENS_PER_CHUNK
            if (
                row["policy"]["generation_index"] != expected_generation
                or row["policy"]["chunk_offset"] != expected_offset
                or not row["applied"]
            ):
                raise ValueError("learner prefix generation metadata is incomplete")
        return history

    def _consume_anchor(
        self,
        task_env: Any,
        entry: Mapping[str, Any],
        anchor_state: np.ndarray,
        extract_state: Callable[[Any], Any],
    ) -> TakeoverOutcome:
        history = self._validate_anchor_history(entry)
        expert = self._run_expert(task_env, anchor_state, extract_state)
        status = expert["status"]
        sample = None
        full_suffix_sample = None
        exception = expert.get("exception")
        if expert["expert_macro_success"] and self._plan.get("full_expert_suffix"):
            try:
                full_suffix_sample = self._write_full_expert_suffix(
                    entry,
                    history,
                    expert["full_states"],
                    expert["full_camera_jpegs"],
                    expert["active_arm"],
                )
            except Exception as exc:
                status = "exception"
                exception = f"{type(exc).__name__}: {exc}"
        if status == "success":
            try:
                sample = self._write_sample(
                    entry,
                    history,
                    expert["states"],
                    expert["bottle_positions"],
                    expert["active_arm"],
                    expert,
                )
            except Exception as exc:
                status = "exception"
                exception = f"{type(exc).__name__}: {exc}"

        result = self._base_result(entry)
        trigger_observation = entry.get("anchor_trigger_observation")
        if trigger_observation is not None:
            distances = [
                observation["distance_m"]
                for observation in self._anchor_trigger_observations
            ]
            result.update(
                {
                    "anchor_triggered": True,
                    "anchor_trigger_candidate_count": len(distances),
                    "anchor_trigger_min_distance_m": min(distances),
                    "anchor_trigger_distance_m": trigger_observation["distance_m"],
                    "anchor_trigger_eef_position": trigger_observation["eef_position"],
                    "anchor_trigger_bottle_position": trigger_observation[
                        "bottle_position"
                    ],
                }
            )
        result.update(
            {
                "status": status,
                "label_valid": status == "success",
                "planner_failed": status == "planner_failed",
                "expert_failed": status == "expert_failed",
                "exception": exception if status == "exception" else None,
                "expert_capture_calls": expert["capture_calls"],
                "expert_state_count": len(expert["states"]),
                "expert_label_count": max(
                    0, min(ACTION_TOKENS_PER_CHUNK, len(expert["states"]) - 1)
                ),
                "active_arm": expert["active_arm"],
                "label_gripper_closed": expert["label_gripper_closed"],
                "label_first_close_index": expert["label_first_close_index"],
                "label_bottle_lift_delta": expert["label_bottle_lift_delta"],
                "label_semantic_valid": expert["label_semantic_valid"],
                "full_expert_suffix": full_suffix_sample,
                "sample": sample,
            }
        )
        self._append_result(result)
        success = status == "success"
        self._terminate_episode(task_env, success=success)
        return TakeoverOutcome(
            consumed=True,
            success=success,
            status=status,
            anchor_id=entry["anchor_id"],
            sample_path=(sample or {}).get("artifact", {}).get("path"),
        )

    @staticmethod
    def _bottle_name(task_env: Any) -> str | None:
        bottle = getattr(task_env, "bottle", None)
        for candidate in (getattr(bottle, "actor", None), bottle):
            getter = getattr(candidate, "get_name", None)
            if callable(getter):
                return str(getter())
        return None

    @staticmethod
    def _bottle_position(task_env: Any) -> np.ndarray:
        bottle = getattr(task_env, "bottle", None)
        for candidate in (getattr(bottle, "actor", None), bottle):
            getter = getattr(candidate, "get_pose", None)
            if not callable(getter):
                continue
            position = np.asarray(getattr(getter(), "p", None), dtype=np.float64)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("expert bottle position must be a finite xyz vector")
            return position.copy()
        raise ValueError("expert task does not expose bottle.get_pose().p")

    @staticmethod
    def _label_semantics(
        states: list[np.ndarray],
        bottle_positions: list[np.ndarray],
        active_arm: str,
    ) -> dict:
        if len(states) != len(bottle_positions):
            raise ValueError("expert state and bottle-position captures are misaligned")
        label_count = max(0, min(ACTION_TOKENS_PER_CHUNK, len(states) - 1))
        gripper_index = 9 if active_arm == "left" else 19
        first_close_index = None
        if label_count:
            grippers = np.asarray(
                [states[index][gripper_index] for index in range(1, label_count + 1)],
                dtype=np.float64,
            )
            closed = np.flatnonzero(grippers < LABEL_GRIPPER_CLOSED_THRESHOLD)
            if closed.size:
                first_close_index = int(closed[0])
            anchor_z = float(bottle_positions[0][2])
            max_label_z = max(
                float(position[2]) for position in bottle_positions[1 : label_count + 1]
            )
            lift_delta = max_label_z - anchor_z
        else:
            lift_delta = None
        gripper_closed = first_close_index is not None
        semantic_valid = bool(
            label_count == ACTION_TOKENS_PER_CHUNK
            and gripper_closed
            and lift_delta is not None
            and lift_delta >= LABEL_BOTTLE_LIFT_THRESHOLD_M
        )
        return {
            "label_gripper_closed": gripper_closed,
            "label_first_close_index": first_close_index,
            "label_bottle_lift_delta": lift_delta,
            "label_semantic_valid": semantic_valid,
        }

    @classmethod
    def _is_bottle_grasped(cls, task_env: Any, arm: str) -> bool:
        explicit = getattr(task_env, "dagger_bottle_grasped", None)
        if isinstance(explicit, bool):
            return explicit
        robot = task_env.robot
        if arm == "left":
            closed = float(robot.get_left_gripper_val()) < 0.5
            ee = np.asarray(robot.get_left_ee_pose()[:3], dtype=np.float64)
        else:
            closed = float(robot.get_right_gripper_val()) < 0.5
            ee = np.asarray(robot.get_right_ee_pose()[:3], dtype=np.float64)
        bottle = task_env.bottle
        bottle_pose = bottle.get_pose()
        bottle_position = np.asarray(bottle_pose.p, dtype=np.float64)
        near = float(np.linalg.norm(bottle_position - ee)) < 0.12
        table_height = float(getattr(task_env, "table_height", 0.74))
        lifted = float(bottle_position[2]) > table_height + 0.06
        bottle_name = cls._bottle_name(task_env)
        contacts = (
            task_env.get_gripper_actor_contact_position(bottle_name)
            if bottle_name is not None
            else []
        )
        return bool(closed and (len(contacts) > 0 or (near and lifted)))

    def _run_expert(
        self,
        task_env: Any,
        anchor_state: np.ndarray,
        extract_state: Callable[[Any], Any],
    ) -> dict:
        states: list[np.ndarray] = []
        bottle_positions: list[np.ndarray] = []
        full_states: list[np.ndarray] = []
        full_camera_jpegs: list[dict[str, bytes]] = []
        capture_full_suffix = bool(self._plan.get("full_expert_suffix"))
        capture_calls = 0
        original_marker = object()
        instance_dict = getattr(task_env, "__dict__", {})
        original_instance = instance_dict.get("_take_picture", original_marker)

        def capture(_task_env: Any) -> None:
            nonlocal capture_calls
            capture_calls += 1
            state = _physical_eef20(
                extract_state(_task_env), "expert physical EEF20 state"
            )
            bottle_position = self._bottle_position(_task_env)
            if capture_full_suffix:
                observation = _task_env.get_obs().get("observation")
                if not isinstance(observation, Mapping):
                    raise ValueError("expert observation mapping is unavailable")
                camera_jpegs = {
                    name: _rgb_jpeg(
                        observation[f"{name}_camera"]["rgb"],
                        f"expert {name} camera",
                    )
                    for name in CAMERA_NAMES
                }
                full_states.append(state)
                full_camera_jpegs.append(camera_jpegs)
            if len(states) < ACTION_TOKENS_PER_CHUNK + 1:
                states.append(state)
                bottle_positions.append(bottle_position)

        status = "exception"
        exception: str | None = None
        active_arm = "right" if int(getattr(task_env, "qpose_tag", 0)) == 1 else "left"
        try:
            if (
                _plain_int(getattr(task_env, "save_freq", None), "task_env.save_freq")
                != EXPERT_SAVE_FREQ
            ):
                raise ValueError(
                    f"task_env.save_freq must remain RoboTwin default {EXPERT_SAVE_FREQ}"
                )
            task_env._take_picture = MethodType(capture, task_env)
            if not bool(getattr(task_env, "plan_success", False)):
                status = "planner_failed"
            elif bool(task_env.check_success()):
                status = "expert_failed"
                exception = "anchor_scene_already_successful"
            else:
                arm = active_arm
                target_pose = (
                    task_env.right_target_pose
                    if arm == "right"
                    else task_env.left_target_pose
                )
                planner_ok = True
                if not self._is_bottle_grasped(task_env, arm):
                    planner_ok = bool(
                        task_env.move(
                            task_env.grasp_actor(
                                task_env.bottle, arm_tag=arm, pre_grasp_dis=0.1
                            )
                        )
                    ) and bool(task_env.plan_success)
                if planner_ok:
                    planner_ok = bool(
                        task_env.move(
                            task_env.move_by_displacement(
                                arm_tag=arm, z=0.1, move_axis="arm"
                            )
                        )
                    ) and bool(task_env.plan_success)
                if planner_ok:
                    planner_ok = bool(
                        task_env.move(
                            task_env.place_actor(
                                task_env.bottle,
                                target_pose=target_pose,
                                arm_tag=arm,
                                functional_point_id=0,
                                pre_dis=0.0,
                                is_open=False,
                            )
                        )
                    ) and bool(task_env.plan_success)
                if not planner_ok:
                    status = "planner_failed"
                elif not bool(task_env.check_success()):
                    status = "expert_failed"
                else:
                    status = "success"
        except AssertionError as exc:
            if str(exc) == "target_pose cannot be None for move action.":
                status = "planner_failed"
                exception = None
            else:
                status = "exception"
                exception = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            status = "exception"
            exception = f"{type(exc).__name__}: {exc}"
        finally:
            if original_instance is original_marker:
                try:
                    delattr(task_env, "_take_picture")
                except AttributeError:
                    pass
            else:
                task_env._take_picture = original_instance

        if status == "success":
            if not states:
                status = "exception"
                exception = "expert produced no dense samples"
            else:
                max_abs = float(np.max(np.abs(states[0] - anchor_state)))
                if max_abs > 1e-5:
                    status = "exception"
                    exception = (
                        "expert first state does not match anchor: "
                        f"max_abs={max_abs:.8g}"
                    )
                elif len(states) < 2:
                    status = "exception"
                    exception = "expert produced no future correction label"

        expert_macro_success = status == "success"
        semantics = self._label_semantics(states, bottle_positions, active_arm)
        if status == "success" and not semantics["label_semantic_valid"]:
            status = "label_incomplete"
        if capture_full_suffix and expert_macro_success:
            if len(full_states) < 2 or len(full_states) != len(full_camera_jpegs):
                status = "exception"
                exception = "full expert suffix capture is incomplete"
                expert_macro_success = False
            elif not np.allclose(full_states[0], anchor_state, rtol=1e-5, atol=1e-5):
                status = "exception"
                exception = "full expert suffix does not start at the learner anchor"
                expert_macro_success = False
        return {
            "status": status,
            "exception": exception,
            "states": states,
            "bottle_positions": bottle_positions,
            "full_states": full_states,
            "full_camera_jpegs": full_camera_jpegs,
            "expert_macro_success": expert_macro_success,
            "capture_calls": capture_calls,
            "active_arm": active_arm,
            **semantics,
        }

    def _write_full_expert_suffix(
        self,
        entry: Mapping[str, Any],
        history: list[dict],
        expert_states: list[np.ndarray],
        expert_camera_jpegs: list[dict[str, bytes]],
        active_arm: str,
    ) -> dict:
        import h5py

        from benchmarks.utils.action_conversion import rot6d_to_quat_xyzw

        if self._output_dir is None:
            raise RuntimeError("recorder output directory is unavailable")
        if len(expert_states) != len(expert_camera_jpegs) or len(expert_states) < 2:
            raise ValueError("full expert suffix state/camera capture is incomplete")

        # Store only the successful expert suffix. Learner history remains audit
        # metadata and must not become a supervised failure trajectory.
        combined_states = np.stack(expert_states, axis=0).astype(np.float32, copy=False)
        combined_cameras = {
            camera: [frame[camera] for frame in expert_camera_jpegs]
            for camera in CAMERA_NAMES
        }
        left_quat = np.stack([rot6d_to_quat_xyzw(row[3:9]) for row in combined_states])
        right_quat = np.stack(
            [rot6d_to_quat_xyzw(row[13:19]) for row in combined_states]
        )
        endpose = {
            "left_endpose": np.concatenate(
                [combined_states[:, 0:3], left_quat], axis=1
            ).astype(np.float32),
            "left_gripper": combined_states[:, 9].astype(np.float32),
            "right_endpose": np.concatenate(
                [combined_states[:, 10:13], right_quat], axis=1
            ).astype(np.float32),
            "right_gripper": combined_states[:, 19].astype(np.float32),
        }

        suffix_root = self._output_dir / "full_suffix_dataset"
        variant_root = (
            suffix_root
            / "dataset"
            / "adjust_bottle"
            / "aloha-agilex_full_expert_suffix"
        )
        data_dir = variant_root / "data"
        instructions_dir = variant_root / "instructions"
        data_dir.mkdir(parents=True, exist_ok=True)
        instructions_dir.mkdir(exist_ok=True)
        episode_index = entry["ordinal"]
        final_path = data_dir / f"episode{episode_index}.hdf5"
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite suffix episode {final_path}")
        temporary = final_path.with_name(
            f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with h5py.File(temporary, "w") as handle:
                handle.attrs["schema_version"] = ARTIFACT_SCHEMA_VERSION
                handle.attrs["source_kind"] = "full_expert_suffix"
                handle.attrs["source_anchor_id"] = entry["anchor_id"]
                handle.attrs["environment_seed"] = entry["environment_seed"]
                handle.attrs["learner_frame_count"] = len(history)
                handle.attrs["expert_frame_count"] = len(expert_states)
                handle.attrs["success"] = True

                observation = handle.create_group("observation")
                jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
                for camera in CAMERA_NAMES:
                    group = observation.create_group(f"{camera}_camera")
                    dataset = group.create_dataset(
                        "rgb", shape=(len(combined_states),), dtype=jpeg_dtype
                    )
                    for index, payload in enumerate(combined_cameras[camera]):
                        dataset[index] = np.frombuffer(payload, dtype=np.uint8)
                endpose_group = handle.create_group("endpose")
                for key, values in endpose.items():
                    endpose_group.create_dataset(key, data=values)
                handle.flush()
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, final_path)
        finally:
            temporary.unlink(missing_ok=True)

        prompt = entry["prompt"]
        instruction_path = instructions_dir / f"episode{episode_index}.json"
        _atomic_write_bytes(
            instruction_path,
            _json_bytes({"seen": [prompt], "unseen": [prompt]}),
        )
        scene_info_path = variant_root / "scene_info.json"
        scene_info = {}
        if scene_info_path.exists():
            scene_info = json.loads(scene_info_path.read_text(encoding="utf-8"))
        scene_info[f"episode_{episode_index}"] = {
            "info": {
                "active_arm": active_arm,
                "source_kind": "full_expert_suffix",
                "environment_seed": entry["environment_seed"],
                "anchor_env_step": entry["anchor_env_step"],
            }
        }
        _atomic_write_bytes(scene_info_path, _json_bytes(scene_info))
        return {
            "anchor_id": entry["anchor_id"],
            "episode_index": episode_index,
            "environment_seed": entry["environment_seed"],
            "anchor_env_step": entry["anchor_env_step"],
            "prompt": prompt,
            "active_arm": active_arm,
            "learner_frame_count": len(history),
            "expert_frame_count": len(expert_states),
            "frame_count": len(combined_states),
            "artifact": _artifact(final_path, relative_to=suffix_root),
        }

    def _write_sample(
        self,
        entry: Mapping[str, Any],
        history: list[dict],
        expert_states: list[np.ndarray],
        expert_bottle_positions: list[np.ndarray],
        active_arm: str,
        label_semantics: Mapping[str, Any],
    ) -> dict:
        import h5py

        if self._output_dir is None:
            raise RuntimeError("recorder output directory is unavailable")
        expert = np.stack(expert_states, axis=0).astype(np.float32, copy=False)
        bottle_positions = np.stack(expert_bottle_positions, axis=0).astype(
            np.float64, copy=False
        )
        if bottle_positions.shape != (expert.shape[0], 3):
            raise ValueError(
                f"invalid expert bottle-position shape {bottle_positions.shape}"
            )
        if not bool(label_semantics["label_semantic_valid"]):
            raise ValueError("refusing to write semantically incomplete labels")
        labels = expert[1 : ACTION_TOKENS_PER_CHUNK + 1]
        label_count = labels.shape[0]
        if not 1 <= label_count <= ACTION_TOKENS_PER_CHUNK:
            raise ValueError(f"invalid expert label count {label_count}")

        prefix_rows = history[ACTION_TOKENS_PER_CHUNK : HISTORY_LEN - 1]
        learner_prefix = np.stack(
            [row["policy_action"] for row in prefix_rows], axis=0
        ).astype(np.float32, copy=False)
        if learner_prefix.shape != (3 * ACTION_TOKENS_PER_CHUNK, ACTION_DIM):
            raise ValueError(f"invalid learner prefix shape {learner_prefix.shape}")
        correction_pad_value = labels[-1] if label_count else history[-1]["pre_state"]
        correction = np.repeat(
            correction_pad_value[None, :], ACTION_TOKENS_PER_CHUNK, axis=0
        ).astype(np.float32)
        correction[:label_count] = labels
        action = np.concatenate([learner_prefix, correction], axis=0).astype(
            np.float32, copy=False
        )
        action_mask = np.zeros(4 * ACTION_TOKENS_PER_CHUNK, dtype=np.bool_)
        action_mask[
            3 * ACTION_TOKENS_PER_CHUNK : 3 * ACTION_TOKENS_PER_CHUNK + label_count
        ] = True

        boundary_states = np.stack(
            [history[index]["pre_state"] for index in (28, 56, 84, 112)],
            axis=0,
        ).astype(np.float32, copy=False)
        proprio_seq = np.empty((HISTORY_LEN, ACTION_DIM), dtype=np.float32)
        boundaries = (0, 28, 56, 84, HISTORY_LEN)
        for chunk_index in range(4):
            proprio_seq[boundaries[chunk_index] : boundaries[chunk_index + 1]] = (
                boundary_states[chunk_index]
            )
        proprio = history[-1]["pre_state"][None, :].astype(np.float32, copy=False)

        final_path = self._output_dir / "samples" / f"{entry['anchor_id']}.hdf5"
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite sample {final_path}")
        temporary = final_path.with_name(
            f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with h5py.File(temporary, "w") as handle:
                handle.attrs["schema_version"] = ARTIFACT_SCHEMA_VERSION
                handle.attrs["source_kind"] = SOURCE_KIND
                handle.attrs["action_space"] = "physical_eef20"
                handle.attrs["label_valid"] = True
                handle.attrs["anchor_id"] = entry["anchor_id"]
                handle.attrs["environment_seed"] = entry["environment_seed"]
                handle.attrs["anchor_env_step"] = entry["anchor_env_step"]
                handle.attrs["generation_index"] = entry["generation_index"]
                trigger_observation = entry.get("anchor_trigger_observation")
                if trigger_observation is not None:
                    handle.attrs["anchor_mode"] = ANCHOR_TRIGGER_KIND
                    handle.attrs["planned_anchor_env_step"] = entry[
                        "planned_anchor_env_step"
                    ]
                    handle.attrs["anchor_trigger_distance_m"] = trigger_observation[
                        "distance_m"
                    ]
                handle.attrs["episode_key"] = self._episode_context["episode_key"]
                handle.attrs["prompt"] = entry["prompt"]
                handle.attrs["prompt_sha256"] = entry["prompt_sha256"]
                handle.attrs["expert_label_count"] = label_count
                handle.attrs["active_arm"] = active_arm
                handle.attrs["label_gripper_closed"] = bool(
                    label_semantics["label_gripper_closed"]
                )
                handle.attrs["label_first_close_index"] = int(
                    label_semantics["label_first_close_index"]
                )
                handle.attrs["label_bottle_lift_delta"] = float(
                    label_semantics["label_bottle_lift_delta"]
                )
                handle.attrs["label_semantic_valid"] = True
                handle.attrs["step_limit_source"] = "robotwin_upstream_default"
                handle.attrs["upstream_step_limit"] = self._plan["upstream_step_limit"]
                handle.attrs["repo_step_limit_override_count"] = 0

                history_group = handle.create_group("history")
                history_group.create_dataset(
                    "env_step",
                    data=np.asarray(
                        [row["env_step"] for row in history], dtype=np.int64
                    ),
                )
                history_group.create_dataset(
                    "generation_index",
                    data=np.asarray(
                        [row["policy"]["generation_index"] for row in history],
                        dtype=np.int64,
                    ),
                )
                history_group.create_dataset(
                    "chunk_offset",
                    data=np.asarray(
                        [row["policy"]["chunk_offset"] for row in history],
                        dtype=np.int16,
                    ),
                )
                history_group.create_dataset(
                    "generated",
                    data=np.asarray(
                        [row["policy"]["generated"] for row in history],
                        dtype=np.bool_,
                    ),
                )
                history_group.create_dataset(
                    "applied",
                    data=np.asarray(
                        [row["applied"] for row in history], dtype=np.bool_
                    ),
                )
                history_group.create_dataset(
                    "pre_state",
                    data=np.stack([row["pre_state"] for row in history]),
                    dtype=np.float32,
                )
                history_group.create_dataset(
                    "policy_action",
                    data=np.stack([row["policy_action"] for row in history]),
                    dtype=np.float32,
                )
                camera_group = history_group.create_group("camera_jpeg")
                jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
                for camera_name in CAMERA_NAMES:
                    dataset = camera_group.create_dataset(
                        camera_name, shape=(HISTORY_LEN,), dtype=jpeg_dtype
                    )
                    for index, row in enumerate(history):
                        dataset[index] = np.frombuffer(
                            row["cameras"][camera_name], dtype=np.uint8
                        )

                expert_group = handle.create_group("expert")
                expert_group.create_dataset("states", data=expert, dtype=np.float32)
                expert_group.create_dataset(
                    "bottle_positions", data=bottle_positions, dtype=np.float64
                )
                handle.create_dataset("action", data=action, dtype=np.float32)
                handle.create_dataset("action_mask", data=action_mask, dtype=np.bool_)
                handle.create_dataset("proprio", data=proprio, dtype=np.float32)
                handle.create_dataset("proprio_seq", data=proprio_seq, dtype=np.float32)
                handle.flush()
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, final_path)
        finally:
            temporary.unlink(missing_ok=True)

        artifact = _artifact(final_path, relative_to=self._output_dir)
        sample = {
            "anchor_id": entry["anchor_id"],
            "environment_seed": entry["environment_seed"],
            "anchor_env_step": entry["anchor_env_step"],
            "prompt": entry["prompt"],
            "prompt_sha256": entry["prompt_sha256"],
            "episode_key": self._episode_context["episode_key"],
            "label_valid": True,
            "active_arm": active_arm,
            "label_gripper_closed": bool(label_semantics["label_gripper_closed"]),
            "label_first_close_index": int(label_semantics["label_first_close_index"]),
            "label_bottle_lift_delta": float(
                label_semantics["label_bottle_lift_delta"]
            ),
            "label_semantic_valid": True,
            "action_space": "physical_eef20",
            "artifact": artifact,
        }
        trigger_observation = entry.get("anchor_trigger_observation")
        if trigger_observation is not None:
            sample.update(
                {
                    "anchor_mode": ANCHOR_TRIGGER_KIND,
                    "planned_anchor_env_step": entry["planned_anchor_env_step"],
                    "anchor_trigger_distance_m": trigger_observation["distance_m"],
                }
            )
        return sample

    def _base_result(self, entry: Mapping[str, Any]) -> dict:
        episode_key = None
        if (
            self._active_entry is not None
            and self._active_entry["anchor_id"] == entry["anchor_id"]
        ):
            episode_key = self._episode_context.get("episode_key")
        result = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "anchor_id": entry["anchor_id"],
            "ordinal": entry["ordinal"],
            "episode_index": entry["episode_index"],
            "environment_seed": entry["environment_seed"],
            "anchor_env_step": entry["anchor_env_step"],
            "generation_index": entry["generation_index"],
            "task_name": entry["task_name"],
            "task_config": entry["task_config"],
            "prompt": entry["prompt"],
            "prompt_sha256": entry["prompt_sha256"],
            "episode_key": episode_key,
        }
        anchor_trigger = self._plan.get("anchor_trigger")
        if anchor_trigger is not None:
            result.update(
                {
                    "anchor_mode": ANCHOR_TRIGGER_KIND,
                    "planned_anchor_env_step": entry.get(
                        "planned_anchor_env_step",
                        anchor_trigger["max_env_step"],
                    ),
                    "anchor_trigger_config": anchor_trigger,
                }
            )
        for field in (
            "control_success",
            "control_step_count",
            "control_anchor_observed",
        ):
            if field in entry:
                result[field] = entry[field]
        return result

    def _record_exception_result(
        self,
        entry: Mapping[str, Any],
        message: str,
        *,
        observed_env_step: int | None,
    ) -> dict:
        existing = self._results.get(entry["anchor_id"])
        if existing is not None:
            return existing
        result = self._base_result(entry)
        result.update(
            {
                "status": "exception",
                "label_valid": False,
                "planner_failed": False,
                "expert_failed": False,
                "exception": message,
                "observed_env_step": observed_env_step,
                "expert_capture_calls": 0,
                "expert_state_count": 0,
                "expert_label_count": 0,
                "active_arm": None,
                "label_gripper_closed": False,
                "label_first_close_index": None,
                "label_bottle_lift_delta": None,
                "label_semantic_valid": False,
                "full_expert_suffix": None,
                "sample": None,
            }
        )
        self._append_result(result)
        return result

    def _append_result(self, result: dict) -> None:
        anchor_id = result["anchor_id"]
        if anchor_id in self._results:
            raise RuntimeError(f"anchor {anchor_id!r} already has a result")
        self._results[anchor_id] = result
        self._write_results_ledger()

    def _write_results_ledger(self) -> None:
        if self._output_dir is None:
            raise RuntimeError("recorder output directory is unavailable")
        ordered = [
            self._results[entry["anchor_id"]]
            for entry in self._entries
            if entry["anchor_id"] in self._results
        ]
        payload = b"".join(
            (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            for row in ordered
        )
        _atomic_write_bytes(self._output_dir / "results.jsonl", payload)

    @staticmethod
    def _terminate_episode(task_env: Any, *, success: bool) -> None:
        if success:
            task_env.eval_success = True
            return
        step_limit = getattr(task_env, "step_lim", None)
        if isinstance(step_limit, int) and not isinstance(step_limit, bool):
            task_env.take_action_cnt = step_limit

    def close(self) -> None:
        """Finalize missing-anchor results and write the immutable manifest + SHA."""
        if not self._enabled or self._closed:
            return
        assert self._plan is not None
        assert self._plan_path is not None
        assert self._plan_sha256 is not None
        assert self._output_dir is not None
        for entry in self._entries:
            if entry["anchor_id"] not in self._results:
                self._record_exception_result(
                    entry,
                    "anchor_not_observed_before_close",
                    observed_env_step=None,
                )
        if len(self._results) != len(self._entries):
            raise RuntimeError("every selected anchor must have exactly one result")
        if _sha256_file(self._plan_path) != self._plan_sha256:
            raise RuntimeError("DAgger plan changed while collection was running")

        ledger_path = self._output_dir / "results.jsonl"
        samples = [
            self._results[entry["anchor_id"]]["sample"]
            for entry in self._entries
            if self._results[entry["anchor_id"]].get("sample") is not None
        ]
        for sample in samples:
            artifact = sample["artifact"]
            path = self._output_dir / artifact["path"]
            actual = _artifact(path, relative_to=self._output_dir)
            if actual != artifact:
                raise RuntimeError(
                    f"sample artifact changed before manifest finalization: {path}"
                )
        stats_path = Path(self._plan["action_stats_path"])
        stats_artifact = _artifact(stats_path)
        if stats_artifact["sha256"] != self._plan["action_stats_sha256"]:
            raise RuntimeError("baseline action_stats changed during collection")

        full_suffix_samples = [
            self._results[entry["anchor_id"]]["full_expert_suffix"]
            for entry in self._entries
            if self._results[entry["anchor_id"]].get("full_expert_suffix") is not None
        ]
        full_suffix_manifest_artifact = None
        if self._plan.get("full_expert_suffix") is not None:
            suffix_root = self._output_dir / "full_suffix_dataset"
            suffix_manifest = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "kind": "robotwin_full_expert_suffix_dataset",
                "complete": True,
                "source_takeover_plan_sha256": self._plan_sha256,
                "source_checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
                "sample_count": len(full_suffix_samples),
                "samples": full_suffix_samples,
                "dataset": {
                    "dataset_dir": str(suffix_root / "dataset"),
                    "task_name": "adjust_bottle",
                    "robot": "aloha-agilex",
                    "variant": "full_expert_suffix",
                },
            }
            suffix_manifest_path = suffix_root / "dataset_manifest.json"
            _atomic_write_bytes(suffix_manifest_path, _json_bytes(suffix_manifest))
            suffix_digest = _sha256_file(suffix_manifest_path)
            _atomic_write_bytes(
                suffix_root / "dataset_manifest.sha256",
                f"{suffix_digest}  dataset_manifest.json\n".encode("ascii"),
            )
            full_suffix_manifest_artifact = _artifact(
                suffix_manifest_path, relative_to=self._output_dir
            )

        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_kind": SOURCE_KIND,
            "complete": True,
            "action_space": "physical_eef20",
            "plan_artifact": _artifact(self._plan_path),
            "results_artifact": _artifact(ledger_path, relative_to=self._output_dir),
            "action_stats_artifact": stats_artifact,
            "repo_step_limits_artifact": self._step_limits_artifact,
            "deployment_identity": self._server_info["inference_runtime"][
                "deployment_identity"
            ],
            "provenance": {
                "policy_history_len": HISTORY_LEN,
                "num_frames": HISTORY_LEN,
                "video_stride": VIDEO_STRIDE,
                "temporal_compression": TEMPORAL_COMPRESSION,
                "causal_temporal": True,
                "ar_frame_chunk_size": AR_FRAME_CHUNK_SIZE,
                "action_tokens_per_chunk": ACTION_TOKENS_PER_CHUNK,
                "action_dim": ACTION_DIM,
                "action_mode": "eef",
                "normalize_mode": "min-max",
                "height": IMAGE_HEIGHT,
                "width": IMAGE_WIDTH,
                "multiview": True,
                "camera_layout": list(CAMERA_LAYOUT),
                "checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
                "action_stats_sha256": BASELINE_ACTION_STATS_SHA256,
                "cache_feedback_mode": "predicted",
                "environment_seed_index": ENVIRONMENT_SEED_INDEX,
                "step_limit_source": "robotwin_upstream_default",
                "repo_step_limit_override_count": 0,
                "deploy_config_sha256": self._plan["deploy_config_sha256"],
                "deploy_overrides_sha256": self._plan["deploy_overrides_sha256"],
                "prompt_manifest_sha256": self._plan["prompt_manifest_sha256"],
                "takeover_plan_sha256": self._plan_sha256,
                "episode_noise_mode": self._plan["episode_noise_mode"],
                "episode_noise_base_seed": self._plan["episode_noise_base_seed"],
                "upstream_step_limit": self._plan["upstream_step_limit"],
                "expert_save_freq": EXPERT_SAVE_FREQ,
                "anchor_trigger": self._plan.get("anchor_trigger"),
                "full_expert_suffix": self._plan.get("full_expert_suffix"),
            },
            "anchor_count": len(self._entries),
            "valid_sample_count": len(samples),
            "full_expert_suffix_count": len(full_suffix_samples),
            "full_expert_suffix_manifest_artifact": full_suffix_manifest_artifact,
            "anchors": [self._results[entry["anchor_id"]] for entry in self._entries],
            "samples": samples,
        }
        manifest_path = self._output_dir / "dataset_manifest.json"
        manifest_bytes = _json_bytes(manifest)
        _atomic_write_bytes(manifest_path, manifest_bytes)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        _atomic_write_bytes(
            self._output_dir / "dataset_manifest.sha256",
            f"{digest}  dataset_manifest.json\n".encode("ascii"),
        )
        self._closed = True


__all__ = [
    "LiveTakeoverPlanError",
    "LiveTakeoverRecorder",
    "TakeoverOutcome",
]
