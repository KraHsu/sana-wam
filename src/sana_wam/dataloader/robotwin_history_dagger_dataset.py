"""Strict one-anchor-per-sample loader for fresh RoboTwin DAgger data."""

from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from sana_wam.dataloader.base_dataset import BaseActionDataset
from sana_wam.dataloader.transforms.multiview import (
    DEFAULT_MULTIVIEW_CAMERA_LAYOUT,
    assemble_multiview_layout,
    format_prompt_for_inference,
)
from sana_wam.dataloader.transforms.normalize import (
    YAML_TO_NORM_MODE,
    ActionNormalizer,
    load_mode_stats,
)

SOURCE_KIND = "fresh_on_policy_history113_dagger"
SCHEMA_VERSION = 1
BASELINE_CHECKPOINT_SHA256 = (
    "aea6b58c9107aeeeffba561f39ffedfc3f8fd8420a4abfcab9038c8cdd4a129d"
)
BASELINE_ACTION_STATS_SHA256 = (
    "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"
)

HISTORY_LEN = 113
VIDEO_STRIDE = 4
NUM_VIDEO_FRAMES = 29
TEMPORAL_COMPRESSION = 4
AR_FRAME_CHUNK_SIZE = 2
NUM_LATENT_FRAMES = 8
NUM_CHUNKS = 4
ACTION_TOKENS_PER_CHUNK = 28
ACTION_DIM = 20
NUM_ACTION_TOKENS = NUM_CHUNKS * ACTION_TOKENS_PER_CHUNK
CONDITION_ACTION_TOKENS = (NUM_CHUNKS - 1) * ACTION_TOKENS_PER_CHUNK

_CAMERA_KEYS = ("head", "left", "right")
_VIDEO_SAMPLE_INDICES = tuple(range(0, HISTORY_LEN, VIDEO_STRIDE))
_BOUNDARY_STATE_INDICES = (28, 56, 84, 112)
_SHA256_HEX_LENGTH = 64


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        value = config.get(key, default)
    else:
        try:
            value = getattr(config, key)
        except (AttributeError, KeyError):
            value = default
    return default if value is None else value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_manifest_artifact(
    descriptor: Any,
    *,
    manifest_root: Path,
    field: str,
    allow_absolute: bool,
) -> Path:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{field} must be an artifact descriptor")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{field}.path must be non-empty")
    path = Path(raw_path)
    if path.is_absolute():
        if not allow_absolute:
            raise ValueError(f"{field}.path must be relative")
        resolved = path.resolve()
    else:
        resolved = (manifest_root / path).resolve()
        try:
            resolved.relative_to(manifest_root.resolve())
        except ValueError as exc:
            raise ValueError(f"{field}.path escapes the manifest directory") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} is not a file: {resolved}")
    expected_size = _require_int(
        descriptor.get("size_bytes"), field=f"{field}.size_bytes", minimum=1
    )
    expected_sha = _require_sha256(descriptor.get("sha256"), field=f"{field}.sha256")
    if resolved.stat().st_size != expected_size:
        raise ValueError(f"{field} size mismatch: {resolved}")
    if _sha256_file(resolved) != expected_sha:
        raise ValueError(f"{field} SHA256 mismatch: {resolved}")
    return resolved


def _required_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{field} must be a non-empty filesystem path")
    return Path(value).expanduser().resolve()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    if value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return value


def _require_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    return result


def _require_exact(mapping: dict, field: str, expected: Any, *, location: str) -> None:
    if field not in mapping:
        raise ValueError(f"{location} is missing required field {field!r}")
    if mapping[field] != expected:
        raise ValueError(
            f"{location}.{field} must be {expected!r}, got {mapping[field]!r}"
        )


def _jpeg_bytes(value: Any, *, field: str) -> bytes:
    if isinstance(value, bytes):
        raw = value
    else:
        array = np.asarray(value)
        if array.ndim != 1 or array.dtype != np.uint8:
            raise ValueError(f"{field} must contain a one-dimensional uint8 JPEG")
        raw = array.tobytes()
    if len(raw) < 4 or not raw.startswith(b"\xff\xd8"):
        raise ValueError(f"{field} is not a non-empty JPEG payload")
    return raw


def _verify_jpeg(value: Any, *, field: str) -> None:
    raw = _jpeg_bytes(value, field=field)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"{field} is not a decodable JPEG: {exc}") from exc


def _decode_jpeg(value: Any, *, field: str) -> Image.Image:
    raw = _jpeg_bytes(value, field=field)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB").copy()
    except Exception as exc:
        raise ValueError(f"{field} is not a decodable JPEG: {exc}") from exc


@dataclass(frozen=True)
class _ManifestSample:
    anchor_id: str
    episode_key: str
    environment_seed: int
    anchor_env_step: int
    prompt: str
    artifact_path: Path
    artifact_size: int
    artifact_sha256: str
    label_valid: bool
    active_arm: str


@dataclass(frozen=True)
class _ValidatedSample:
    manifest: _ManifestSample
    expert_steps: int
    artifact_fingerprint: tuple[int, int, int, int, int]


def _file_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class RoboTwinHistoryDaggerDataset(BaseActionDataset):
    """Manifest-driven history-113 DAgger dataset.

    Each valid anchor contributes exactly one sample. The first three action
    chunks are learner commands used only as causal context; the final chunk is
    a live-scene expert correction and is the only action-loss region.
    """

    @classmethod
    def from_config(cls, config: Any, split: str = "train"):
        camera_layout = list(
            _get(config, "camera_layout", DEFAULT_MULTIVIEW_CAMERA_LAYOUT)
        )
        return cls(
            source_manifest_path=_get(config, "source_manifest_path"),
            source_manifest_sha256=_get(config, "source_manifest_sha256"),
            action_stats_path=_get(config, "action_stats_path"),
            action_stats_sha256=_get(config, "action_stats_sha256"),
            expected_policy_history_len=int(
                _get(config, "expected_policy_history_len", HISTORY_LEN)
            ),
            action_mode=str(_get(config, "action_mode", "eef")),
            normalize_mode=str(_get(config, "normalize_mode", "min-max")),
            num_frames=int(_get(config, "num_frames", HISTORY_LEN)),
            video_stride=int(_get(config, "video_stride", VIDEO_STRIDE)),
            temporal_compression=int(
                _get(config, "temporal_compression", TEMPORAL_COMPRESSION)
            ),
            causal_temporal=bool(_get(config, "causal_temporal", True)),
            height=int(_get(config, "height", 384)),
            width=int(_get(config, "width", 320)),
            multiview=bool(_get(config, "multiview", True)),
            camera_layout=camera_layout,
            target_camera=str(_get(config, "target_camera", "head_camera")),
            filter_static_segments=bool(_get(config, "filter_static_segments", False)),
            delta_action=bool(_get(config, "delta_action", False)),
            repeat=int(_get(config, "repeat", 1)),
            split=split,
            val_ratio=float(_get(config, "val_ratio", 0.0)),
            seed=int(_get(config, "seed", 42)),
        )

    def __init__(
        self,
        *,
        source_manifest_path: str | Path,
        source_manifest_sha256: str,
        action_stats_path: str | Path,
        action_stats_sha256: str,
        expected_policy_history_len: int = HISTORY_LEN,
        action_mode: str = "eef",
        normalize_mode: str = "min-max",
        num_frames: int = HISTORY_LEN,
        video_stride: int = VIDEO_STRIDE,
        temporal_compression: int = TEMPORAL_COMPRESSION,
        causal_temporal: bool = True,
        height: int = 384,
        width: int = 320,
        multiview: bool = True,
        camera_layout: list[str] | tuple[str, ...] = DEFAULT_MULTIVIEW_CAMERA_LAYOUT,
        target_camera: str = "head_camera",
        filter_static_segments: bool = False,
        delta_action: bool = False,
        repeat: int = 1,
        split: str = "train",
        val_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._validate_config_contract(
            expected_policy_history_len=expected_policy_history_len,
            action_mode=action_mode,
            normalize_mode=normalize_mode,
            num_frames=num_frames,
            video_stride=video_stride,
            temporal_compression=temporal_compression,
            causal_temporal=causal_temporal,
            height=height,
            width=width,
            multiview=multiview,
            camera_layout=camera_layout,
            target_camera=target_camera,
            filter_static_segments=filter_static_segments,
            delta_action=delta_action,
            repeat=repeat,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
        )

        self.action_mode = action_mode
        self.normalize_mode = normalize_mode
        self.num_frames = num_frames
        self.num_action_steps = NUM_ACTION_TOKENS
        self.video_stride = video_stride
        self.num_video_frames = NUM_VIDEO_FRAMES
        self.temporal_compression = temporal_compression
        self.causal_temporal = causal_temporal
        self.height = height
        self.width = width
        self.multiview = multiview
        self.camera_layout = list(camera_layout)
        self.cameras = list(camera_layout)
        self.target_camera = target_camera
        self.filter_static_segments = False
        self.delta_action = False
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed

        manifest_path = _required_path(
            source_manifest_path, field="source_manifest_path"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"source manifest is not a file: {manifest_path}")
        expected_manifest_sha = _require_sha256(
            source_manifest_sha256, field="source_manifest_sha256"
        )
        actual_manifest_sha = _sha256_file(manifest_path)
        if actual_manifest_sha != expected_manifest_sha:
            raise ValueError(
                "source manifest SHA256 mismatch: "
                f"expected {expected_manifest_sha}, got {actual_manifest_sha}"
            )
        self.source_manifest_path = str(manifest_path)
        self.source_manifest_sha256 = actual_manifest_sha

        stats_path = _required_path(action_stats_path, field="action_stats_path")
        if not stats_path.is_file():
            raise FileNotFoundError(f"action_stats_path is not a file: {stats_path}")
        expected_stats_sha = _require_sha256(
            action_stats_sha256, field="action_stats_sha256"
        )
        if expected_stats_sha != BASELINE_ACTION_STATS_SHA256:
            raise ValueError(
                "history DAgger must use the fixed baseline action stats SHA256 "
                f"{BASELINE_ACTION_STATS_SHA256}, got {expected_stats_sha}"
            )
        actual_stats_sha = _sha256_file(stats_path)
        if actual_stats_sha != expected_stats_sha:
            raise ValueError(
                "action stats SHA256 mismatch: "
                f"expected {expected_stats_sha}, got {actual_stats_sha}"
            )
        mode_stats = load_mode_stats(str(stats_path), action_mode)
        self._validate_action_stats(mode_stats)
        self.action_stats_path = str(stats_path)
        self.action_stats_sha256 = actual_stats_sha
        self._mode_stats = mode_stats
        self._action_normalizer = ActionNormalizer(
            mode=YAML_TO_NORM_MODE[normalize_mode], stats=mode_stats
        )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"failed to parse source manifest {manifest_path}: {exc}"
            ) from exc
        records = self._validate_manifest(manifest, manifest_path.parent)
        valid = [record for record in records if record.label_valid]
        if not valid:
            raise ValueError("source manifest contains no label_valid anchors")

        validated = [self._validate_artifact(record) for record in valid]
        selected_episode_keys = self._select_episode_keys(validated)
        self._samples = [
            sample
            for sample in validated
            if sample.manifest.episode_key in selected_episode_keys
        ]
        if not self._samples:
            raise ValueError(
                f"no label_valid anchors selected for split={split!r}; "
                "adjust val_ratio or provide more source episodes"
            )
        self.episode_keys = frozenset(selected_episode_keys)

    @staticmethod
    def _validate_config_contract(**values: Any) -> None:
        expected = {
            "expected_policy_history_len": HISTORY_LEN,
            "action_mode": "eef",
            "normalize_mode": "min-max",
            "num_frames": HISTORY_LEN,
            "video_stride": VIDEO_STRIDE,
            "temporal_compression": TEMPORAL_COMPRESSION,
            "causal_temporal": True,
            "multiview": True,
            "camera_layout": list(DEFAULT_MULTIVIEW_CAMERA_LAYOUT),
            "target_camera": "head_camera",
            "filter_static_segments": False,
            "delta_action": False,
            "repeat": 1,
        }
        for field, wanted in expected.items():
            got = values[field]
            if field == "camera_layout":
                got = list(got)
            if got != wanted:
                raise ValueError(
                    f"history DAgger config field {field} must be {wanted!r}, got {got!r}"
                )
        for field in ("height", "width"):
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer, got {value!r}")
            if value % 32 != 0:
                raise ValueError(f"{field} must be divisible by 32, got {value}")
        if values["split"] not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {values['split']!r}")
        val_ratio = values["val_ratio"]
        if not isinstance(val_ratio, (int, float)) or not 0.0 <= val_ratio <= 1.0:
            raise ValueError(f"val_ratio must be in [0, 1], got {val_ratio!r}")
        if isinstance(values["seed"], bool) or not isinstance(values["seed"], int):
            raise ValueError(f"seed must be an integer, got {values['seed']!r}")

    @staticmethod
    def _validate_action_stats(mode_stats: Any) -> None:
        if not isinstance(mode_stats, dict):
            raise ValueError("action stats file has no 'eef' stats mapping")
        for field in ("mean", "std", "min", "max"):
            if field not in mode_stats:
                raise ValueError(f"eef action stats are missing {field!r}")
            values = np.asarray(mode_stats[field])
            if values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
                raise ValueError(
                    f"eef action stats {field!r} must have finite shape ({ACTION_DIM},), "
                    f"got {values.shape}"
                )
        if np.any(np.asarray(mode_stats["max"]) <= np.asarray(mode_stats["min"])):
            raise ValueError("eef action stats require max > min in every dimension")

    def _validate_manifest(
        self, manifest: Any, manifest_root: Path
    ) -> list[_ManifestSample]:
        if not isinstance(manifest, dict):
            raise ValueError("source manifest root must be an object")
        _require_exact(manifest, "schema_version", SCHEMA_VERSION, location="manifest")
        _require_exact(manifest, "source_kind", SOURCE_KIND, location="manifest")
        _require_exact(manifest, "complete", True, location="manifest")

        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("manifest.provenance must be an object")
        expected_provenance = {
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
            "height": self.height,
            "width": self.width,
            "multiview": True,
            "camera_layout": list(DEFAULT_MULTIVIEW_CAMERA_LAYOUT),
            "checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
            "action_stats_sha256": BASELINE_ACTION_STATS_SHA256,
            "cache_feedback_mode": "predicted",
            "environment_seed_index": 1,
            "step_limit_source": "robotwin_upstream_default",
            "repo_step_limit_override_count": 0,
        }
        for field, expected in expected_provenance.items():
            _require_exact(provenance, field, expected, location="manifest.provenance")
        for field in (
            "deploy_config_sha256",
            "deploy_overrides_sha256",
            "prompt_manifest_sha256",
            "takeover_plan_sha256",
        ):
            _require_sha256(provenance.get(field), field=f"manifest.provenance.{field}")
        noise_mode = provenance.get("episode_noise_mode")
        if not isinstance(noise_mode, str) or not noise_mode:
            raise ValueError("manifest.provenance.episode_noise_mode must be non-empty")
        _require_int(
            provenance.get("episode_noise_base_seed"),
            field="manifest.provenance.episode_noise_base_seed",
            minimum=0,
        )

        raw_samples = manifest.get("samples")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError("manifest.samples must be a non-empty list")
        records: list[_ManifestSample] = []
        anchor_ids: set[str] = set()
        episode_anchors: set[tuple[str, str]] = set()
        artifact_paths: set[Path] = set()
        for index, raw in enumerate(raw_samples):
            location = f"manifest.samples[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{location} must be an object")
            anchor_id = raw.get("anchor_id")
            episode_key = raw.get("episode_key")
            prompt = raw.get("prompt")
            if not isinstance(anchor_id, str) or not anchor_id:
                raise ValueError(f"{location}.anchor_id must be non-empty")
            if not isinstance(episode_key, str) or not episode_key:
                raise ValueError(f"{location}.episode_key must be non-empty")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{location}.prompt must be non-empty")
            if anchor_id in anchor_ids:
                raise ValueError(
                    f"duplicate anchor_id in source manifest: {anchor_id!r}"
                )
            pair = (episode_key, anchor_id)
            if pair in episode_anchors:
                raise ValueError(
                    f"duplicate episode/anchor pair in source manifest: {pair!r}"
                )
            anchor_ids.add(anchor_id)
            episode_anchors.add(pair)

            environment_seed = _require_int(
                raw.get("environment_seed"),
                field=f"{location}.environment_seed",
                minimum=0,
            )
            if not 200_000 <= environment_seed < 300_000:
                raise ValueError(
                    f"{location}.environment_seed must be in disjoint 200xxx range, "
                    f"got {environment_seed}"
                )
            anchor_env_step = _require_int(
                raw.get("anchor_env_step"),
                field=f"{location}.anchor_env_step",
                minimum=HISTORY_LEN - 1,
            )
            label_valid = raw.get("label_valid")
            if not isinstance(label_valid, (bool, np.bool_)):
                raise ValueError(f"{location}.label_valid must be boolean")
            prompt_sha = _require_sha256(
                raw.get("prompt_sha256"), field=f"{location}.prompt_sha256"
            )
            if prompt_sha != _sha256_text(prompt):
                raise ValueError(f"{location}.prompt_sha256 does not match prompt")

            artifact = raw.get("artifact")
            if not isinstance(artifact, dict):
                raise ValueError(f"{location}.artifact must be an object")
            relative_path = artifact.get("path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"{location}.artifact.path must be non-empty")
            relative = Path(relative_path)
            if relative.is_absolute():
                raise ValueError(f"{location}.artifact.path must be relative")
            artifact_path = (manifest_root / relative).resolve()
            try:
                artifact_path.relative_to(manifest_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"{location}.artifact.path escapes the manifest directory"
                ) from exc
            if artifact_path in artifact_paths:
                raise ValueError(
                    f"multiple anchors reference the same artifact: {artifact_path}"
                )
            artifact_paths.add(artifact_path)
            artifact_size = _require_int(
                artifact.get("size_bytes"),
                field=f"{location}.artifact.size_bytes",
                minimum=1,
            )
            artifact_sha = _require_sha256(
                artifact.get("sha256"), field=f"{location}.artifact.sha256"
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"anchor artifact is not a file: {artifact_path}"
                )
            actual_size = artifact_path.stat().st_size
            if actual_size != artifact_size:
                raise ValueError(
                    f"artifact size mismatch for {artifact_path}: "
                    f"expected {artifact_size}, got {actual_size}"
                )
            actual_sha = _sha256_file(artifact_path)
            if actual_sha != artifact_sha:
                raise ValueError(
                    f"artifact SHA256 mismatch for {artifact_path}: "
                    f"expected {artifact_sha}, got {actual_sha}"
                )
            active_arm = raw.get("active_arm", "both")
            if not isinstance(active_arm, str) or not active_arm:
                raise ValueError(f"{location}.active_arm must be non-empty")
            records.append(
                _ManifestSample(
                    anchor_id=anchor_id,
                    episode_key=episode_key,
                    environment_seed=environment_seed,
                    anchor_env_step=anchor_env_step,
                    prompt=prompt,
                    artifact_path=artifact_path,
                    artifact_size=artifact_size,
                    artifact_sha256=artifact_sha,
                    label_valid=bool(label_valid),
                    active_arm=active_arm,
                )
            )
        self._validate_collection_ledger(
            manifest,
            manifest_root=manifest_root,
            provenance=provenance,
            raw_samples=raw_samples,
        )
        return records

    @staticmethod
    def _validate_collection_ledger(
        manifest: dict,
        *,
        manifest_root: Path,
        provenance: dict,
        raw_samples: list[dict],
    ) -> None:
        plan_path = _validate_manifest_artifact(
            manifest.get("plan_artifact"),
            manifest_root=manifest_root,
            field="manifest.plan_artifact",
            allow_absolute=True,
        )
        plan_sha = manifest["plan_artifact"]["sha256"]
        if plan_sha != provenance["takeover_plan_sha256"]:
            raise ValueError("manifest plan artifact SHA256 does not match provenance")
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to parse takeover plan: {exc}") from exc
        if not isinstance(plan, dict):
            raise ValueError("takeover plan root must be an object")
        plan_contract = {
            "kind": "robotwin_live_takeover_plan",
            "source_kind": SOURCE_KIND,
            "checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
            "action_stats_sha256": BASELINE_ACTION_STATS_SHA256,
            "deploy_config_sha256": provenance["deploy_config_sha256"],
            "deploy_overrides_sha256": provenance["deploy_overrides_sha256"],
            "prompt_manifest_sha256": provenance["prompt_manifest_sha256"],
            "cache_feedback_mode": "predicted",
            "episode_noise_mode": provenance["episode_noise_mode"],
            "episode_noise_base_seed": provenance["episode_noise_base_seed"],
            "environment_seed_index": 1,
            "step_limit_source": "robotwin_upstream_default",
            "repo_step_limit_override_count": 0,
        }
        for field, expected in plan_contract.items():
            if plan.get(field) != expected:
                raise ValueError(
                    f"takeover plan field {field} must be {expected!r}, "
                    f"got {plan.get(field)!r}"
                )
        plan_episodes = plan.get("episodes")
        if not isinstance(plan_episodes, list) or not plan_episodes:
            raise ValueError("takeover plan episodes must be a non-empty list")

        results_path = _validate_manifest_artifact(
            manifest.get("results_artifact"),
            manifest_root=manifest_root,
            field="manifest.results_artifact",
            allow_absolute=False,
        )
        try:
            ledger = [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to parse results ledger: {exc}") from exc
        anchors = manifest.get("anchors")
        if not isinstance(anchors, list):
            raise ValueError("manifest.anchors must be a list")
        if ledger != anchors:
            raise ValueError("results ledger does not exactly match manifest.anchors")
        if manifest.get("anchor_count") != len(plan_episodes):
            raise ValueError("manifest.anchor_count does not match takeover plan")
        if len(anchors) != len(plan_episodes):
            raise ValueError("every selected anchor must have exactly one result")

        expected_ids = [episode.get("anchor_id") for episode in plan_episodes]
        actual_ids = [anchor.get("anchor_id") for anchor in anchors]
        if any(
            not isinstance(anchor_id, str) or not anchor_id
            for anchor_id in expected_ids
        ):
            raise ValueError("takeover plan contains an invalid anchor_id")
        if len(set(expected_ids)) != len(expected_ids) or actual_ids != expected_ids:
            raise ValueError(
                "anchor result identities do not exactly match the takeover plan"
            )

        valid_samples = []
        identity_fields = (
            "anchor_id",
            "episode_index",
            "environment_seed",
            "anchor_env_step",
            "generation_index",
            "prompt",
            "prompt_sha256",
        )
        for index, (planned, result) in enumerate(zip(plan_episodes, anchors)):
            if not isinstance(result, dict):
                raise ValueError(f"manifest.anchors[{index}] must be an object")
            for field in identity_fields:
                if result.get(field) != planned.get(field):
                    raise ValueError(
                        f"manifest.anchors[{index}].{field} does not match plan"
                    )
            label_valid = result.get("label_valid")
            sample = result.get("sample")
            if not isinstance(label_valid, bool):
                raise ValueError(
                    f"manifest.anchors[{index}].label_valid must be boolean"
                )
            if label_valid != (
                result.get("status") == "success" and sample is not None
            ):
                raise ValueError(
                    f"manifest.anchors[{index}] has inconsistent result/sample status"
                )
            if label_valid:
                if not isinstance(sample, dict):
                    raise ValueError(
                        f"manifest.anchors[{index}].sample must be an object"
                    )
                valid_samples.append(sample)
            elif sample is not None:
                raise ValueError(
                    f"manifest.anchors[{index}] invalid label must not carry a sample"
                )

        if manifest.get("valid_sample_count") != len(valid_samples):
            raise ValueError("manifest.valid_sample_count is inconsistent")
        if raw_samples != valid_samples:
            raise ValueError(
                "manifest.samples must contain every and only label-valid anchor sample"
            )

    def _validate_artifact(self, record: _ManifestSample) -> _ValidatedSample:
        try:
            artifact = h5py.File(record.artifact_path, "r")
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"failed to open anchor artifact {record.artifact_path}: {exc}"
            ) from exc
        with artifact:
            required = {
                "history/env_step": (HISTORY_LEN,),
                "history/pre_state": (HISTORY_LEN, ACTION_DIM),
                "history/policy_action": (HISTORY_LEN, ACTION_DIM),
                "history/applied": (HISTORY_LEN,),
                "history/generation_index": (HISTORY_LEN,),
                "history/chunk_offset": (HISTORY_LEN,),
                "history/generated": (HISTORY_LEN,),
            }
            for key, shape in required.items():
                if key not in artifact:
                    raise ValueError(
                        f"artifact {record.anchor_id!r} is missing {key!r}"
                    )
                if artifact[key].shape != shape:
                    raise ValueError(
                        f"artifact {record.anchor_id!r} {key} must have shape {shape}, "
                        f"got {artifact[key].shape}"
                    )

            env_step = np.asarray(artifact["history/env_step"][...])
            pre_state = np.asarray(artifact["history/pre_state"][...], dtype=np.float32)
            policy_action = np.asarray(
                artifact["history/policy_action"][...], dtype=np.float32
            )
            action_applied = np.asarray(artifact["history/applied"][...])
            generation = np.asarray(artifact["history/generation_index"][...])
            chunk_offset = np.asarray(artifact["history/chunk_offset"][...])
            generated = np.asarray(artifact["history/generated"][...])

            for key, values in (
                ("history/env_step", env_step),
                ("history/generation_index", generation),
                ("history/chunk_offset", chunk_offset),
            ):
                if values.dtype.kind not in "iu":
                    raise ValueError(
                        f"artifact {record.anchor_id!r} {key} must be integer"
                    )
            for key, values in (
                ("history/applied", action_applied),
                ("history/generated", generated),
            ):
                if values.dtype.kind != "b":
                    raise ValueError(
                        f"artifact {record.anchor_id!r} {key} must be boolean"
                    )
            if not np.isfinite(pre_state).all() or not np.isfinite(policy_action).all():
                raise ValueError(
                    f"artifact {record.anchor_id!r} history state/action contains non-finite values"
                )
            if not np.all(np.diff(env_step.astype(np.int64)) == 1):
                raise ValueError(
                    f"artifact {record.anchor_id!r} history/env_step has a gap"
                )
            if int(env_step[-1]) != record.anchor_env_step:
                raise ValueError(
                    f"artifact {record.anchor_id!r} anchor env step mismatch: "
                    f"{int(env_step[-1])} != {record.anchor_env_step}"
                )
            expected_clock_offsets = env_step.astype(np.int64) % ACTION_TOKENS_PER_CHUNK
            expected_clock_generations = (
                env_step.astype(np.int64) // ACTION_TOKENS_PER_CHUNK
            )
            if not np.array_equal(
                chunk_offset.astype(np.int64), expected_clock_offsets
            ) or not np.array_equal(
                generation.astype(np.int64), expected_clock_generations
            ):
                raise ValueError(
                    f"artifact {record.anchor_id!r} generation metadata is not "
                    "phase-locked to history/env_step"
                )

            expected_offsets = np.concatenate(
                [
                    np.tile(np.arange(ACTION_TOKENS_PER_CHUNK), NUM_CHUNKS),
                    np.asarray([0]),
                ]
            )
            if not np.array_equal(chunk_offset.astype(np.int64), expected_offsets):
                raise ValueError(
                    f"artifact {record.anchor_id!r} does not contain four complete "
                    "28-step learner chunks followed by an anchor boundary"
                )
            anchor_generation = int(generation[-1])
            expected_generations = np.concatenate(
                [
                    np.repeat(
                        anchor_generation - NUM_CHUNKS + index,
                        ACTION_TOKENS_PER_CHUNK,
                    )
                    for index in range(NUM_CHUNKS)
                ]
                + [np.asarray([anchor_generation])]
            )
            if not np.array_equal(generation.astype(np.int64), expected_generations):
                raise ValueError(
                    f"artifact {record.anchor_id!r} generation history is not consecutive"
                )
            expected_generated = expected_offsets == 0
            if not np.array_equal(generated.astype(bool), expected_generated):
                raise ValueError(
                    f"artifact {record.anchor_id!r} generated flags do not match chunk boundaries"
                )
            if not bool(action_applied[:-1].all()) or bool(action_applied[-1]):
                raise ValueError(
                    f"artifact {record.anchor_id!r} requires all history actions applied "
                    "and the anchor learner action not applied"
                )

            for camera in _CAMERA_KEYS:
                key = f"history/camera_jpeg/{camera}"
                if key not in artifact:
                    raise ValueError(
                        f"artifact {record.anchor_id!r} is missing {key!r}"
                    )
                dataset = artifact[key]
                if dataset.shape != (HISTORY_LEN,):
                    raise ValueError(
                        f"artifact {record.anchor_id!r} {key} must have shape "
                        f"({HISTORY_LEN},), got {dataset.shape}"
                    )
                for frame_index in range(HISTORY_LEN):
                    _verify_jpeg(
                        dataset[frame_index],
                        field=f"artifact {record.anchor_id!r} {key}[{frame_index}]",
                    )

            if "expert/states" not in artifact:
                raise ValueError(
                    f"artifact {record.anchor_id!r} is missing 'expert/states'"
                )
            expert_states = np.asarray(artifact["expert/states"][...], dtype=np.float32)
            if (
                expert_states.ndim != 2
                or expert_states.shape[1] != ACTION_DIM
                or not 2 <= expert_states.shape[0] <= ACTION_TOKENS_PER_CHUNK + 1
            ):
                raise ValueError(
                    f"artifact {record.anchor_id!r} expert/states must have shape "
                    f"(1+n, {ACTION_DIM}) with 1 <= n <= {ACTION_TOKENS_PER_CHUNK}, "
                    f"got {expert_states.shape}"
                )
            if not np.isfinite(expert_states).all():
                raise ValueError(
                    f"artifact {record.anchor_id!r} expert/states contains non-finite values"
                )
            if not np.allclose(expert_states[0], pre_state[-1], rtol=1e-5, atol=1e-4):
                raise ValueError(
                    f"artifact {record.anchor_id!r} expert state 0 does not match anchor state"
                )

            expert_steps = int(expert_states.shape[0] - 1)
            expert_targets = expert_states[1:]
            expert_chunk = np.repeat(
                expert_targets[-1:], ACTION_TOKENS_PER_CHUNK, axis=0
            )
            expert_chunk[:expert_steps] = expert_targets
            expected_action = np.concatenate(
                [policy_action[28:112], expert_chunk], axis=0
            ).astype(np.float32)
            expected_action_mask = np.zeros(NUM_ACTION_TOKENS, dtype=bool)
            expected_action_mask[
                CONDITION_ACTION_TOKENS : CONDITION_ACTION_TOKENS + expert_steps
            ] = True
            expected_proprio = pre_state[-1:].astype(np.float32)
            boundary_states = pre_state[list(_BOUNDARY_STATE_INDICES)]
            expected_proprio_seq = np.empty((HISTORY_LEN, ACTION_DIM), dtype=np.float32)
            boundaries = (0, 28, 56, 84, HISTORY_LEN)
            for chunk_index in range(NUM_CHUNKS):
                expected_proprio_seq[
                    boundaries[chunk_index] : boundaries[chunk_index + 1]
                ] = boundary_states[chunk_index]

            expected_root = {
                "action": expected_action,
                "action_mask": expected_action_mask,
                "proprio": expected_proprio,
                "proprio_seq": expected_proprio_seq,
            }
            for key, expected_values in expected_root.items():
                if key not in artifact:
                    raise ValueError(
                        f"artifact {record.anchor_id!r} is missing root dataset {key!r}"
                    )
                actual_values = np.asarray(artifact[key][...])
                if actual_values.shape != expected_values.shape:
                    raise ValueError(
                        f"artifact {record.anchor_id!r} root {key} must have shape "
                        f"{expected_values.shape}, got {actual_values.shape}"
                    )
                if key == "action_mask":
                    if actual_values.dtype.kind != "b":
                        raise ValueError(
                            f"artifact {record.anchor_id!r} root action_mask must be boolean"
                        )
                    matches = np.array_equal(actual_values, expected_values)
                else:
                    if not np.issubdtype(actual_values.dtype, np.floating):
                        raise ValueError(
                            f"artifact {record.anchor_id!r} root {key} must be floating point"
                        )
                    matches = bool(
                        np.isfinite(actual_values).all()
                        and np.allclose(
                            actual_values,
                            expected_values,
                            rtol=1e-6,
                            atol=1e-6,
                        )
                    )
                if not matches:
                    raise ValueError(
                        f"artifact {record.anchor_id!r} root {key} does not match "
                        "the arrays independently reconstructed from history/expert data"
                    )
        return _ValidatedSample(
            manifest=record,
            expert_steps=expert_steps,
            artifact_fingerprint=_file_fingerprint(record.artifact_path),
        )

    def _select_episode_keys(self, samples: list[_ValidatedSample]) -> set[str]:
        episode_keys = sorted({sample.manifest.episode_key for sample in samples})
        shuffled = list(episode_keys)
        random.Random(self.seed).shuffle(shuffled)
        if self.val_ratio <= 0.0:
            n_val = 0
        elif self.val_ratio >= 1.0:
            n_val = len(shuffled)
        else:
            n_val = max(1, int(len(shuffled) * self.val_ratio))
        val_keys = set(shuffled[:n_val])
        return val_keys if self.split == "val" else set(shuffled[n_val:])

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def action_stats(self) -> dict:
        return {
            key: np.asarray(value).copy() for key, value in self._mode_stats.items()
        }

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return self._action_normalizer.unnormalize(np.asarray(action))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        validated = self._samples[index]
        record = validated.manifest
        if _file_fingerprint(record.artifact_path) != validated.artifact_fingerprint:
            raise RuntimeError(
                f"immutable anchor artifact changed after dataset initialization: "
                f"{record.artifact_path}"
            )
        with h5py.File(record.artifact_path, "r") as artifact:
            pre_state = np.asarray(artifact["history/pre_state"][...], dtype=np.float32)
            policy_action = np.asarray(
                artifact["history/policy_action"][...], dtype=np.float32
            )
            expert_states = np.asarray(artifact["expert/states"][...], dtype=np.float32)

            sampled_video: list[Image.Image] = []
            for frame_index in _VIDEO_SAMPLE_INDICES:
                decoded = {
                    camera: _decode_jpeg(
                        artifact[f"history/camera_jpeg/{camera}"][frame_index],
                        field=(
                            f"artifact {record.anchor_id!r} history/camera_jpeg/"
                            f"{camera}[{frame_index}]"
                        ),
                    )
                    for camera in _CAMERA_KEYS
                }
                frames_by_layout = {
                    self.camera_layout[0]: decoded["head"],
                    self.camera_layout[1]: decoded["left"],
                    self.camera_layout[2]: decoded["right"],
                }
                sampled_video.append(
                    assemble_multiview_layout(
                        frames_by_layout,
                        self.camera_layout,
                        self.height,
                        self.width,
                    )
                )

        prefix = policy_action[28:112]
        expert_targets = expert_states[1:]
        expert_chunk = np.repeat(expert_targets[-1:], ACTION_TOKENS_PER_CHUNK, axis=0)
        expert_chunk[: validated.expert_steps] = expert_targets
        action_physical = np.concatenate([prefix, expert_chunk], axis=0)
        action = self._action_normalizer.normalize(action_physical)

        anchor_proprio = self._action_normalizer.normalize(pre_state[-1:])
        boundary_states = self._action_normalizer.normalize(
            pre_state[list(_BOUNDARY_STATE_INDICES)]
        )
        proprio_seq = np.empty((HISTORY_LEN, ACTION_DIM), dtype=np.float32)
        boundaries = (0, 28, 56, 84, HISTORY_LEN)
        for chunk_index in range(NUM_CHUNKS):
            proprio_seq[boundaries[chunk_index] : boundaries[chunk_index + 1]] = (
                boundary_states[chunk_index]
            )

        action_mask = torch.zeros(NUM_ACTION_TOKENS, dtype=torch.bool)
        action_mask[
            CONDITION_ACTION_TOKENS : CONDITION_ACTION_TOKENS + validated.expert_steps
        ] = True
        return {
            "video": sampled_video,
            "first_frame_image": [sampled_video[0]],
            "video_mask": torch.ones(NUM_VIDEO_FRAMES, dtype=torch.bool),
            "action": torch.from_numpy(action),
            "action_mask": action_mask,
            "proprio": torch.from_numpy(anchor_proprio),
            "proprio_mask": torch.ones(1, dtype=torch.bool),
            "proprio_seq": torch.from_numpy(proprio_seq),
            "num_clean_prefix_latent": 0,
            "num_clean_prefix_actions": CONDITION_ACTION_TOKENS,
            "prompt": format_prompt_for_inference(record.prompt),
            "episode_index": index,
            "episode_key": record.episode_key,
            "episode_path": str(record.artifact_path),
            "environment_seed": record.environment_seed,
            "anchor_id": record.anchor_id,
            "anchor_env_step": record.anchor_env_step,
            "start_frame": record.anchor_env_step - (HISTORY_LEN - 1),
            "end_frame": record.anchor_env_step + 1,
            "episode_length": HISTORY_LEN,
            "task_name": "adjust_bottle",
            "active_arm": record.active_arm,
            "source_kind": SOURCE_KIND,
        }


__all__ = [
    "BASELINE_ACTION_STATS_SHA256",
    "BASELINE_CHECKPOINT_SHA256",
    "RoboTwinHistoryDaggerDataset",
    "SOURCE_KIND",
]
