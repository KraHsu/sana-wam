"""Manifest-bound episode-origin view for CACH training rows.

The legacy RoboTwin loader remains unchanged.  This module wraps an already
constructed dataset with an immutable row-order manifest, rejects any
non-episode-origin or clean-prefix sample, builds its registered
``ChunkActionLayout``, selects only causal per-chunk proprio states, and removes
the full state sequence before the sample can reach an architecture.

No filesystem, HDF5, CUDA, model, or training operation is performed here.
Production construction accepts only externally verified timebase proofs.
Synthetic fixtures must use the explicitly test-named classmethod.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Mapping, Sequence

from sana_wam.cach.config import ACTION_ORDER
from sana_wam.model.action_chunk_layout import (
    ChunkActionLayout,
    ChunkActionLayoutSpec,
    LayoutContractError,
    LayoutReasonCode,
    RowTimebaseProof,
    build_chunk_action_layout,
    build_synthetic_chunk_action_layout_for_tests,
    canonical_proprio_row_sha256,
    canonical_sha256,
)


_SYNTHETIC_DATASET_MARKER = "CACH_SYNTHETIC_EPISODE_DATASET_TEST_ONLY_V1"
_ROW_SCHEMA = "cach.episode_origin_manifest_row.v1"
_FIRST_FRAME_BINDING_SCHEMA = "cach.first_frame_video0_binding.v1"

# Exact legacy row schema accepted by this boundary.  Source-only fields are
# recognized so they can be validated or deliberately removed, but are never
# copied into the model-facing sample.
_MODEL_FACING_PASSTHROUGH_KEYS = frozenset(
    {
        "action",
        "action_mask",
        "active_arm",
        "end_frame",
        "episode_index",
        "episode_length",
        "episode_path",
        "first_frame_image",
        "prompt",
        "start_frame",
        "task_name",
        "video",
        "video_mask",
    }
)
_SOURCE_ONLY_KEYS = frozenset(
    {
        "_is_static",
        "num_clean_prefix_actions",
        "num_clean_prefix_latent",
        "proprio",
        "proprio_mask",
        "proprio_seq",
    }
)
_LEGACY_SAMPLE_KEY_ALLOWLIST = (
    _MODEL_FACING_PASSTHROUGH_KEYS | _SOURCE_ONLY_KEYS
)
_STATE_LIKE_KEY_TOKENS = (
    "future",
    "joint",
    "observation",
    "proprio",
    "qpos",
    "qvel",
    "robot_state",
    "state",
)


def _require_plain_int(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a plain integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _as_plain_sequence(value: Any, name: str) -> tuple[Any, ...]:
    """Read a CPU tensor/ndarray/list without importing either array package."""

    device = getattr(value, "device", None)
    if device is not None and str(getattr(device, "type", device)) != "cpu":
        raise ValueError(f"{name} must remain on CPU at the dataset boundary")
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite sequence") from exc


def _plain_bool_mask(value: Any, name: str) -> tuple[bool, ...]:
    items = _as_plain_sequence(value, name)
    if any(type(item) is not bool for item in items):
        raise TypeError(f"{name} must contain booleans only")
    return items


def _leading_length(value: Any, name: str) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            if len(shape) < 1:
                raise ValueError(f"{name} must have a leading time dimension")
            return int(shape[0])
        except TypeError as exc:
            raise ValueError(f"{name} has an invalid shape") from exc
    try:
        return len(value)
    except TypeError as exc:
        raise TypeError(f"{name} must expose a leading time dimension") from exc


def _sequence_item(value: Any, index: int, name: str) -> Any:
    device = getattr(value, "device", None)
    if device is not None and str(getattr(device, "type", device)) != "cpu":
        raise ValueError(f"{name} must remain on CPU at the dataset boundary")
    try:
        return value[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise TypeError(f"{name} must support positional indexing") from exc


def _digest_part(hasher: Any, tag: str, payload: bytes = b"") -> None:
    """Append one unambiguous length-delimited item to a content digest."""

    encoded_tag = tag.encode("utf-8")
    hasher.update(len(encoded_tag).to_bytes(8, "big"))
    hasher.update(encoded_tag)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _array_bytes(value: Any, name: str) -> bytes | None:
    """Return canonical logical-order bytes for an ndarray/tensor-like value."""

    candidate = value
    detached = getattr(candidate, "detach", None)
    if callable(detached):
        candidate = detached()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    contiguous = getattr(candidate, "contiguous", None)
    if callable(contiguous):
        candidate = contiguous()
    numpy = getattr(candidate, "numpy", None)
    if callable(numpy):
        try:
            candidate = numpy()
        except (RuntimeError, TypeError):
            return None
    tobytes = getattr(candidate, "tobytes", None)
    if not callable(tobytes):
        return None
    try:
        raw = tobytes(order="C")
    except TypeError:
        raw = tobytes()
    if not isinstance(raw, bytes):
        try:
            raw = bytes(raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name}.tobytes() must return bytes") from exc
    return raw


def _update_exact_content_digest(
    hasher: Any,
    value: Any,
    name: str,
) -> None:
    """Hash exact CPU payload content without importing torch or NumPy."""

    device = getattr(value, "device", None)
    if device is not None and str(getattr(device, "type", device)) != "cpu":
        raise ValueError(f"{name} must remain on CPU at the dataset boundary")

    if value is None:
        _digest_part(hasher, "none")
        return
    if type(value) is bool:
        _digest_part(hasher, "bool", b"1" if value else b"0")
        return
    if isinstance(value, int) and not isinstance(value, bool):
        _digest_part(hasher, "int", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        _digest_part(hasher, "float64", struct.pack(">d", value))
        return
    if isinstance(value, str):
        _digest_part(hasher, "str", value.encode("utf-8"))
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        _digest_part(hasher, "bytes", bytes(value))
        return

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            canonical_shape = tuple(int(item) for item in shape)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} has a non-canonical shape") from exc
        dtype_name = str(dtype)
        if dtype_name.lower() in {"object", "object_", "numpy.object_"}:
            raise TypeError(f"{name} object dtype is not digestible")
        _digest_part(hasher, "array")
        _digest_part(
            hasher,
            "shape",
            ",".join(str(item) for item in canonical_shape).encode("ascii"),
        )
        _digest_part(hasher, "dtype", dtype_name.encode("utf-8"))
        raw = _array_bytes(value, name)
        if raw is not None:
            _digest_part(hasher, "logical-c-bytes", raw)
            return
        tolist = getattr(value, "tolist", None)
        if not callable(tolist):
            raise TypeError(
                f"{name} array/tensor payload cannot expose exact CPU content"
            )
        _update_exact_content_digest(hasher, tolist(), f"{name}.tolist()")
        return

    mode = getattr(value, "mode", None)
    size = getattr(value, "size", None)
    tobytes = getattr(value, "tobytes", None)
    if isinstance(mode, str) and size is not None and callable(tobytes):
        try:
            canonical_size = tuple(int(item) for item in size)
            raw = tobytes()
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} has invalid image content") from exc
        if not isinstance(raw, bytes):
            raise TypeError(f"{name}.tobytes() must return bytes")
        _digest_part(hasher, "image")
        _digest_part(hasher, "mode", mode.encode("utf-8"))
        _digest_part(
            hasher,
            "size",
            ",".join(str(item) for item in canonical_size).encode("ascii"),
        )
        _digest_part(hasher, "pixels", raw)
        return

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} image mapping keys must be strings")
        _digest_part(hasher, "mapping")
        for key in sorted(value):
            _digest_part(hasher, "key", key.encode("utf-8"))
            _update_exact_content_digest(
                hasher,
                value[key],
                f"{name}.{key}",
            )
        _digest_part(hasher, "mapping-end")
        return

    if isinstance(value, (list, tuple)):
        _digest_part(hasher, "list" if isinstance(value, list) else "tuple")
        for item_index, item in enumerate(value):
            _update_exact_content_digest(
                hasher,
                item,
                f"{name}[{item_index}]",
            )
        _digest_part(hasher, "sequence-end")
        return

    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is value:
            raise TypeError(f"{name} scalar item() returned itself")
        _update_exact_content_digest(hasher, scalar, f"{name}.item()")
        return

    raise TypeError(
        f"{name} contains unsupported exact-content type "
        f"{type(value).__name__}"
    )


def _exact_content_sha256(value: Any, name: str) -> str:
    hasher = hashlib.sha256()
    _digest_part(hasher, "cach.exact_cpu_content.v1")
    _update_exact_content_digest(hasher, value, name)
    return hasher.hexdigest()


def _validate_first_frame_binding(
    sample: Mapping[str, Any],
    layout: ChunkActionLayout,
) -> tuple[str, str]:
    """Bind the sole pinned first-frame reference to exact ``video[0]`` bytes."""

    if "first_frame_image" not in sample:
        raise LayoutContractError(
            LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
            "first_frame_pinned requires first_frame_image",
        )
    if "video" not in sample:
        raise LayoutContractError(
            LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
            "first_frame_pinned requires video",
        )
    if _leading_length(sample["first_frame_image"], "first_frame_image") != 1:
        raise LayoutContractError(
            LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
            "first_frame_image must contain exactly one pinned frame",
        )
    if _leading_length(sample["video"], "video") < 1:
        raise LayoutContractError(
            LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
            "video must contain raw frame 0",
        )

    pinned = _sequence_item(
        sample["first_frame_image"],
        0,
        "first_frame_image",
    )
    video_zero = _sequence_item(sample["video"], 0, "video")
    pinned_digest = _exact_content_sha256(pinned, "first_frame_image[0]")
    video_zero_digest = _exact_content_sha256(video_zero, "video[0]")
    if pinned_digest != video_zero_digest:
        raise LayoutContractError(
            LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
            "first_frame_image[0] is not exact content-equal to video[0]",
            context={
                "first_frame_content_sha256": pinned_digest,
                "video_zero_content_sha256": video_zero_digest,
            },
        )

    binding_digest = canonical_sha256(
        {
            "content_sha256": pinned_digest,
            "layout_instance_digest": layout.layout_instance_digest,
            "raw_frame_index": 0,
            "schema": _FIRST_FRAME_BINDING_SCHEMA,
            "video_buffer_index": 0,
        }
    )
    return pinned_digest, binding_digest


def _select_rows(value: Any, indices: tuple[int, ...], name: str) -> Any:
    """Preserve tensor/ndarray type where possible, otherwise return a tuple."""

    device = getattr(value, "device", None)
    if device is not None and str(getattr(device, "type", device)) != "cpu":
        raise ValueError(f"{name} must remain on CPU at the dataset boundary")
    try:
        return value[list(indices)]
    except (IndexError, KeyError, TypeError):
        try:
            return tuple(value[index] for index in indices)
        except (IndexError, KeyError, TypeError) as exc:
            raise LayoutContractError(
                LayoutReasonCode.PROPRIO_BOUNDARY_MISMATCH,
                "cannot select registered per-chunk proprio indices",
            ) from exc


def _validate_numeric_matrix(
    value: Any,
    *,
    name: str,
    expected_rows: int,
    expected_width: int,
) -> None:
    """Validate an exact CPU float matrix without importing torch or NumPy."""

    rows = _as_plain_sequence(value, name)
    if len(rows) != expected_rows:
        raise ValueError(
            f"{name} row count must be {expected_rows}, got {len(rows)}"
        )
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            observed_shape = tuple(int(item) for item in shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} has a non-canonical shape") from exc
        if observed_shape != (expected_rows, expected_width):
            raise ValueError(
                f"{name} shape must be {(expected_rows, expected_width)}, "
                f"got {observed_shape}"
            )
    dtype = getattr(value, "dtype", None)
    if dtype is not None and str(dtype) not in {"float32", "torch.float32"}:
        raise TypeError(f"{name} dtype must be exact float32, got {dtype}")
    for row_index, row in enumerate(rows):
        columns = _as_plain_sequence(row, f"{name}[{row_index}]")
        if len(columns) != expected_width:
            raise ValueError(
                f"{name}[{row_index}] width must be {expected_width}"
            )
        for column_index, item in enumerate(columns):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(
                    f"{name}[{row_index}][{column_index}] must be numeric"
                )
            if not math.isfinite(float(item)):
                raise ValueError(
                    f"{name}[{row_index}][{column_index}] is non-finite"
                )


def validate_cach_episode_origin_config(config: Mapping[str, Any]) -> None:
    """Reject every legacy default that could change the registered row set.

    This validator is intended to run *before* constructing
    ``MultiTaskRoboTwinDataset``.  That ordering matters because the legacy
    constructor can otherwise auto-compute normalization statistics and write
    them beside the dataset.
    """

    if not isinstance(config, Mapping):
        raise TypeError("CACH dataloader config must be a mapping")

    required_keys = {
        "action_dim",
        "action_mode",
        "action_order",
        "action_representation",
        "action_stats_path",
        "allow_random_row_substitution",
        "causal_temporal",
        "dataset_manifest",
        "delta_action",
        "episode_origin_only",
        "filter_static_segments",
        "growing_history",
        "normalize_mode",
        "num_frames",
        "reject_nonzero_row_start",
        "repeat",
        "row_rate_manifest",
        "seed",
        "temporal_compression",
        "val_ratio",
        "variant",
        "video_stride",
        "window_stride",
    }
    missing = sorted(required_keys - set(config))
    if missing:
        raise ValueError(
            "CACH dataloader config must make legacy defaults explicit; "
            f"missing={missing}"
        )
    if config["episode_origin_only"] is not True:
        raise ValueError("episode_origin_only must be the boolean true")
    if config["reject_nonzero_row_start"] is not True:
        raise ValueError("reject_nonzero_row_start must be the boolean true")
    if config["growing_history"] is not False:
        raise ValueError("growing_history must be the boolean false")
    if config["filter_static_segments"] is not False:
        raise ValueError("filter_static_segments must be the boolean false")
    if config["allow_random_row_substitution"] is not False:
        raise ValueError(
            "allow_random_row_substitution must be the boolean false"
        )
    if config["causal_temporal"] is not True:
        raise LayoutContractError(
            LayoutReasonCode.VAE_CONTRACT_MISMATCH,
            "CACH v0 requires a causal temporal VAE",
        )
    if config["action_mode"] != "eef":
        raise ValueError("CACH-A requires action_mode='eef'")
    if config["delta_action"] is not False:
        raise ValueError("CACH-A requires delta_action=false")
    if config["action_dim"] != 20:
        raise ValueError("CACH-A requires action_dim=20")
    if config["action_representation"] != (
        "absolute_eef_target_xyz_rot6d_gripper"
    ):
        raise ValueError("CACH-A action representation differs")
    try:
        observed_action_order = tuple(config["action_order"])
    except TypeError as exc:
        raise TypeError("action_order must be an exact sequence") from exc
    if observed_action_order != ACTION_ORDER:
        raise ValueError("CACH-A action order differs")

    repeat = config["repeat"]
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat != 1:
        raise ValueError("repeat must be the plain integer 1")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an explicit plain integer")
    val_ratio = config["val_ratio"]
    if (
        isinstance(val_ratio, bool)
        or not isinstance(val_ratio, (int, float))
        or float(val_ratio) != 0.0
    ):
        raise ValueError(
            "val_ratio must be explicitly 0; the external manifest owns the split"
        )
    window_stride = config["window_stride"]
    _require_plain_int(window_stride, "window_stride", minimum=1)
    if window_stride != 1:
        raise ValueError("window_stride must be exactly 1")

    num_frames = _require_plain_int(
        config["num_frames"],
        "num_frames",
        minimum=2,
    )
    video_stride = _require_plain_int(
        config["video_stride"],
        "video_stride",
        minimum=1,
    )
    temporal_compression = _require_plain_int(
        config["temporal_compression"],
        "temporal_compression",
        minimum=1,
    )
    if (num_frames - 1) % video_stride != 0:
        raise LayoutContractError(
            LayoutReasonCode.LAYOUT_GRID_MISMATCH,
            "num_frames endpoint is not on the video-stride grid",
        )
    num_video_frames = ((num_frames - 1) // video_stride) + 1
    if (num_video_frames - 1) % temporal_compression != 0:
        raise LayoutContractError(
            LayoutReasonCode.LAYOUT_GRID_MISMATCH,
            "num_frames endpoint is not on the causal-VAE temporal grid",
        )

    variant = config["variant"]
    _require_nonempty_string(variant, "variant")
    for name in (
        "dataset_manifest",
        "row_rate_manifest",
        "action_stats_path",
        "normalize_mode",
    ):
        _require_nonempty_string(config[name], name)
    if config["normalize_mode"].lower() in {"none", "null"}:
        raise ValueError(
            "normalize_mode must name the registered normalization transform"
        )
    task_name = config.get("task_name")
    train_tasks = config.get("train_tasks")
    has_task_name = (
        isinstance(task_name, str)
        and bool(task_name)
        and task_name.strip() == task_name
    )
    has_train_tasks = (
        isinstance(train_tasks, Sequence)
        and not isinstance(train_tasks, (str, bytes, bytearray))
        and bool(train_tasks)
        and all(
            isinstance(task, str) and bool(task) and task.strip() == task
            for task in train_tasks
        )
    )
    if has_task_name == has_train_tasks:
        raise ValueError(
            "exactly one explicit task_name or non-empty train_tasks cohort is required"
        )
    for cache_key in ("text_embedding_cache_dir", "vae_cache_dir"):
        if config.get(cache_key) is not None:
            raise ValueError(
                f"{cache_key} must remain null until its source manifest is registered"
            )


@dataclass(frozen=True)
class EpisodeOriginManifestRow:
    """One immutable external row-order entry."""

    dataset_index: int
    episode_index: int
    episode_path: str
    task_name: str
    source_row_digest: str
    episode_id_digest: str
    row_start_raw_index: int
    row_end_raw_index: int
    timebase_manifest_digest: str
    timestamp_verification_receipt_digest: str
    schema: str = _ROW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _ROW_SCHEMA:
            raise ValueError(f"unsupported episode-origin row schema: {self.schema!r}")
        _require_plain_int(self.dataset_index, "dataset_index", minimum=0)
        _require_plain_int(self.episode_index, "episode_index", minimum=0)
        _require_nonempty_string(self.episode_path, "episode_path")
        _require_nonempty_string(self.task_name, "task_name")
        _require_sha256(self.source_row_digest, "source_row_digest")
        _require_sha256(self.episode_id_digest, "episode_id_digest")
        _require_plain_int(
            self.row_start_raw_index,
            "row_start_raw_index",
            minimum=0,
        )
        if self.row_start_raw_index != 0:
            raise LayoutContractError(
                LayoutReasonCode.NONZERO_EPISODE_ROW_START,
                "episode-origin manifest rows must start at raw index 0",
            )
        _require_plain_int(
            self.row_end_raw_index,
            "row_end_raw_index",
            minimum=1,
        )
        if self.row_end_raw_index <= self.row_start_raw_index:
            raise ValueError("row_end_raw_index must exceed row_start_raw_index")
        _require_sha256(
            self.timebase_manifest_digest,
            "timebase_manifest_digest",
        )
        _require_sha256(
            self.timestamp_verification_receipt_digest,
            "timestamp_verification_receipt_digest",
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EpisodeOriginManifestRow":
        if not isinstance(value, Mapping):
            raise TypeError("episode-origin row must be a mapping")
        expected = {
            "dataset_index",
            "episode_id_digest",
            "episode_index",
            "episode_path",
            "row_end_raw_index",
            "row_start_raw_index",
            "schema",
            "source_row_digest",
            "task_name",
            "timebase_manifest_digest",
            "timestamp_verification_receipt_digest",
        }
        if set(value) != expected:
            raise ValueError(
                "episode-origin manifest row keys changed: "
                f"missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        return cls(**dict(value))

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset_index": self.dataset_index,
            "episode_id_digest": self.episode_id_digest,
            "episode_index": self.episode_index,
            "episode_path": self.episode_path,
            "row_end_raw_index": self.row_end_raw_index,
            "row_start_raw_index": self.row_start_raw_index,
            "schema": self.schema,
            "source_row_digest": self.source_row_digest,
            "task_name": self.task_name,
            "timebase_manifest_digest": self.timebase_manifest_digest,
            "timestamp_verification_receipt_digest": (
                self.timestamp_verification_receipt_digest
            ),
        }

    @property
    def row_manifest_entry_digest(self) -> str:
        return canonical_sha256(self.to_payload())


def _validate_zero_legacy_prefix(sample: Mapping[str, Any]) -> None:
    for key in ("num_clean_prefix_latent", "num_clean_prefix_actions"):
        if key not in sample:
            continue
        value = sample[key]
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise LayoutContractError(
                LayoutReasonCode.LEGACY_CLEAN_PREFIX_NONZERO,
                f"{key} must be the plain integer 0 under strict_zero",
            )


def _validate_legacy_sample_keys(sample: Mapping[str, Any]) -> None:
    non_string_keys = tuple(
        repr(key) for key in sample if not isinstance(key, str)
    )
    if non_string_keys:
        raise TypeError(
            "legacy sample keys must be strings; "
            f"observed={non_string_keys}"
        )
    unexpected = tuple(sorted(set(sample) - _LEGACY_SAMPLE_KEY_ALLOWLIST))
    if not unexpected:
        return
    state_like = tuple(
        key
        for key in unexpected
        if any(
            token in key.lower().replace("-", "_")
            for token in _STATE_LIKE_KEY_TOKENS
        )
    )
    if state_like:
        raise LayoutContractError(
            LayoutReasonCode.FUTURE_PROPRIO_VISIBLE,
            "unknown state-like source fields are forbidden at the "
            f"model boundary: {state_like}",
        )
    raise ValueError(
        "legacy sample keys differ from the exact CACH allowlist: "
        f"unexpected={unexpected}"
    )


def _validate_legacy_dataset_boundary(dataset: Any) -> None:
    """Reject legacy settings that can silently change a manifest-bound row."""

    children = getattr(dataset, "_sub_datasets", None)
    leaves = tuple(children) if children else (dataset,)
    if not leaves:
        raise ValueError("episode-origin dataset has no legacy source leaves")
    for index, leaf in enumerate(leaves):
        if getattr(leaf, "growing_history", None) is not False:
            raise LayoutContractError(
                LayoutReasonCode.LEGACY_CLEAN_PREFIX_NONZERO,
                f"legacy source leaf {index} must set growing_history=False",
            )
        if getattr(leaf, "_filter_static_segments", None) is not False:
            raise ValueError(
                "episode-origin source must disable filter_static_segments "
                "before materialization"
            )
        repeat = getattr(leaf, "repeat", None)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat != 1:
            raise ValueError("episode-origin source leaves require repeat=1")
        if getattr(leaf, "action_mode", None) != "eef":
            raise ValueError("CACH-A episode-origin source requires action_mode='eef'")
        if getattr(leaf, "delta_action", None) is not False:
            raise ValueError("CACH-A episode-origin source requires delta_action=False")
        if getattr(leaf, "action_dim", None) != 20:
            raise ValueError("CACH-A episode-origin source requires action_dim=20")


def select_layout_proprio(
    sample: Mapping[str, Any],
    layout: ChunkActionLayout,
) -> dict[str, Any]:
    """Return a sanitized model-facing sample with only causal chunk states."""

    if not isinstance(sample, Mapping):
        raise TypeError("legacy sample must be a mapping")
    if not isinstance(layout, ChunkActionLayout):
        raise TypeError("layout must be a ChunkActionLayout")
    _validate_legacy_sample_keys(sample)
    _validate_zero_legacy_prefix(sample)
    if "proprio_seq" not in sample:
        raise LayoutContractError(
            LayoutReasonCode.PROPRIO_BOUNDARY_MISMATCH,
            "legacy source sample has no state sequence for boundary selection",
        )
    proprio_seq = sample["proprio_seq"]
    sequence_length = _leading_length(proprio_seq, "proprio_seq")
    _validate_numeric_matrix(
        proprio_seq,
        name="proprio_seq",
        expected_rows=sequence_length,
        expected_width=20,
    )
    indices = layout.proprio_raw_indices
    if any(index >= layout.valid_raw_count for index in indices):
        raise LayoutContractError(
            LayoutReasonCode.FUTURE_PROPRIO_VISIBLE,
            "layout selected a state beyond the valid observed row",
        )
    if any(index >= sequence_length for index in indices):
        raise LayoutContractError(
            LayoutReasonCode.PROPRIO_BOUNDARY_MISMATCH,
            "state sequence is shorter than a registered proprio boundary",
        )
    selected = _select_rows(proprio_seq, indices, "proprio_seq")
    proprio_value_sha256s = tuple(
        canonical_proprio_row_sha256(
            _sequence_item(
                selected,
                chunk_index,
                "proprio_per_chunk",
            )
        )
        for chunk_index in range(len(indices))
    )
    first_frame_content_sha256, first_frame_binding_digest = (
        _validate_first_frame_binding(sample, layout)
    )

    result = {
        key: sample[key]
        for key in sorted(_MODEL_FACING_PASSTHROUGH_KEYS)
        if key in sample
    }
    result["proprio_per_chunk"] = selected
    result["proprio_per_chunk_mask"] = tuple(True for _ in indices)
    result["proprio_per_chunk_raw_indices"] = indices
    result["proprio_per_chunk_timestamps"] = tuple(
        chunk.proprio_timestamp for chunk in layout.chunks
    )
    result["proprio_per_chunk_receipt_digests"] = tuple(
        chunk.proprio_source_receipt_digest for chunk in layout.chunks
    )
    result["proprio_per_chunk_value_sha256s"] = proprio_value_sha256s
    result["proprio_per_chunk_bindings"] = (
        layout.bind_selected_proprio_digests(proprio_value_sha256s)
    )
    result["action_chunk_layout"] = layout
    result["layout_spec_sha256"] = layout.layout_spec_sha256
    result["layout_instance_digest"] = layout.layout_instance_digest
    result["first_frame_binding_schema"] = _FIRST_FRAME_BINDING_SCHEMA
    result["first_frame_content_sha256"] = first_frame_content_sha256
    result["first_frame_binding_digest"] = first_frame_binding_digest

    if set(result) & _SOURCE_ONLY_KEYS:
        raise LayoutContractError(
            LayoutReasonCode.FUTURE_PROPRIO_VISIBLE,
            "a source-only field survived model-facing sanitization",
        )
    return result


class CachEpisodeOriginDataset:
    """Read-only, manifest-ordered CACH view over a legacy RoboTwin dataset."""

    def __init__(
        self,
        dataset: Any,
        rows: Sequence[EpisodeOriginManifestRow],
        *,
        layout_spec: ChunkActionLayoutSpec,
        timebase_proofs: Mapping[str, RowTimebaseProof],
        dataset_manifest_sha256: str,
        row_order_manifest_sha256: str,
        _synthetic_test_marker: str | None = None,
    ) -> None:
        if _synthetic_test_marker is None:
            synthetic_test = False
        elif _synthetic_test_marker == _SYNTHETIC_DATASET_MARKER:
            synthetic_test = True
        else:
            raise ValueError("invalid synthetic dataset marker")
        if not isinstance(layout_spec, ChunkActionLayoutSpec):
            raise TypeError("layout_spec must be a ChunkActionLayoutSpec")
        if synthetic_test != layout_spec.synthetic_test_only:
            raise LayoutContractError(
                LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
                "dataset mode and layout spec synthetic status differ",
            )
        _require_sha256(dataset_manifest_sha256, "dataset_manifest_sha256")
        _require_sha256(
            row_order_manifest_sha256,
            "row_order_manifest_sha256",
        )
        if not isinstance(timebase_proofs, Mapping):
            raise TypeError("timebase_proofs must be a mapping")

        immutable_rows = tuple(rows)
        if not immutable_rows:
            raise ValueError("episode-origin manifest must contain at least one row")
        if any(not isinstance(row, EpisodeOriginManifestRow) for row in immutable_rows):
            raise TypeError("rows must contain EpisodeOriginManifestRow values only")
        dataset_indices = tuple(row.dataset_index for row in immutable_rows)
        if len(set(dataset_indices)) != len(dataset_indices):
            raise ValueError("episode-origin manifest reuses a dataset index")
        episode_ids = tuple(row.episode_id_digest for row in immutable_rows)
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("episode-origin manifest reuses an episode identity")

        proofs: dict[str, RowTimebaseProof] = {}
        for row in immutable_rows:
            proof = timebase_proofs.get(row.source_row_digest)
            if not isinstance(proof, RowTimebaseProof):
                raise LayoutContractError(
                    LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
                    "manifest row has no matching RowTimebaseProof",
                    context={"source_row_digest": row.source_row_digest},
                )
            if proof.source_row_digest != row.source_row_digest:
                raise LayoutContractError(
                    LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
                    "timebase proof binds a different source row",
                )
            if proof.episode_id_digest != row.episode_id_digest:
                raise LayoutContractError(
                    LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
                    "timebase proof binds a different episode",
                )
            if proof.timebase_manifest_digest != row.timebase_manifest_digest:
                raise LayoutContractError(
                    LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
                    "timebase manifest digest differs from the row manifest",
                )
            if (
                proof.verification_receipt_digest
                != row.timestamp_verification_receipt_digest
            ):
                raise LayoutContractError(
                    LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
                    "timebase verification receipt differs from the row manifest",
                )
            if synthetic_test:
                proof.validate_for_synthetic_test()
            else:
                proof.validate_for_production()
            proofs[row.source_row_digest] = proof

        _validate_legacy_dataset_boundary(dataset)
        try:
            source_length = len(dataset)
        except TypeError as exc:
            raise TypeError("legacy dataset must expose __len__") from exc
        if any(row.dataset_index >= source_length for row in immutable_rows):
            raise ValueError("episode-origin manifest references an invalid dataset index")

        self._dataset = dataset
        self._rows = immutable_rows
        self._layout_spec = layout_spec
        self._proofs = proofs
        self._synthetic_test = synthetic_test
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.row_order_manifest_sha256 = row_order_manifest_sha256
        self.episode_origin_manifest_digest = canonical_sha256(
            {
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "layout_spec_sha256": layout_spec.layout_spec_sha256,
                "row_order_manifest_sha256": row_order_manifest_sha256,
                "rows": tuple(row.to_payload() for row in immutable_rows),
                "synthetic_test": synthetic_test,
            }
        )

    @classmethod
    def for_synthetic_tests(
        cls,
        dataset: Any,
        rows: Sequence[EpisodeOriginManifestRow],
        *,
        layout_spec: ChunkActionLayoutSpec,
        timebase_proofs: Mapping[str, RowTimebaseProof],
    ) -> "CachEpisodeOriginDataset":
        digest = hashlib.sha256(
            b"cach.synthetic.episode-origin-manifest.v1"
        ).hexdigest()
        return cls(
            dataset,
            rows,
            layout_spec=layout_spec,
            timebase_proofs=timebase_proofs,
            dataset_manifest_sha256=digest,
            row_order_manifest_sha256=hashlib.sha256(
                b"cach.synthetic.row-order-manifest.v1"
            ).hexdigest(),
            _synthetic_test_marker=_SYNTHETIC_DATASET_MARKER,
        )

    @property
    def action_dim(self) -> Any:
        return self._dataset.action_dim

    @property
    def action_stats(self) -> Any:
        return self._dataset.action_stats

    @property
    def action_stats_path(self) -> Any:
        return getattr(self._dataset, "action_stats_path", None)

    @property
    def task_names(self) -> Any:
        return getattr(self._dataset, "task_names", None)

    @property
    def rows(self) -> tuple[EpisodeOriginManifestRow, ...]:
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("episode-origin dataset index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        row = self._rows[index]
        sample = self._dataset[row.dataset_index]
        if not isinstance(sample, Mapping):
            raise TypeError("legacy dataset sample must be a mapping")
        observed_identity = {
            "episode_index": sample.get("episode_index"),
            "episode_path": sample.get("episode_path"),
            "start_frame": sample.get("start_frame"),
            "end_frame": sample.get("end_frame"),
            "task_name": sample.get("task_name"),
        }
        expected_identity = {
            "episode_index": row.episode_index,
            "episode_path": row.episode_path,
            "start_frame": row.row_start_raw_index,
            "end_frame": row.row_end_raw_index,
            "task_name": row.task_name,
        }
        if observed_identity != expected_identity:
            if sample.get("start_frame") != 0:
                raise LayoutContractError(
                    LayoutReasonCode.NONZERO_EPISODE_ROW_START,
                    "legacy materializer returned a non-origin row",
                    context={"observed_start": sample.get("start_frame")},
                )
            raise ValueError(
                "legacy sample identity differs from its episode-origin manifest row"
            )
        _validate_zero_legacy_prefix(sample)
        if "first_frame_image" not in sample:
            raise LayoutContractError(
                LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
                "first_frame_pinned requires first_frame_image",
            )

        required_payloads = {
            "action",
            "action_mask",
            "first_frame_image",
            "proprio_seq",
            "video",
            "video_mask",
        }
        missing_payloads = sorted(required_payloads - set(sample))
        if missing_payloads:
            raise ValueError(
                f"legacy sample is missing required payloads: {missing_payloads}"
            )
        action_mask = _plain_bool_mask(sample["action_mask"], "action_mask")
        video_mask = _plain_bool_mask(sample["video_mask"], "video_mask")
        if _leading_length(sample["action"], "action") != len(action_mask):
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_GAP,
                "action payload length differs from action_mask",
            )
        if _leading_length(sample["video"], "video") != len(video_mask):
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_GRID_MISMATCH,
                "video payload length differs from video_mask",
            )
        total_raw_count = len(action_mask) + 1
        expected_video_buffer_count = (
            (total_raw_count - 1) // self._layout_spec.video_stride
        ) + 1
        if len(video_mask) != expected_video_buffer_count:
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_GRID_MISMATCH,
                "video buffer/mask length differs from registered video_stride",
            )
        _validate_numeric_matrix(
            sample["action"],
            name="action",
            expected_rows=len(action_mask),
            expected_width=self._layout_spec.action_dim,
        )
        _validate_numeric_matrix(
            sample["proprio_seq"],
            name="proprio_seq",
            expected_rows=total_raw_count,
            expected_width=self._layout_spec.action_dim,
        )
        valid_action_count = sum(action_mask)
        valid_raw_count = valid_action_count + 1
        if valid_raw_count != row.row_end_raw_index:
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_GAP,
                "row manifest end does not match the contiguous valid action prefix",
            )

        proof = self._proofs[row.source_row_digest]
        builder = (
            build_synthetic_chunk_action_layout_for_tests
            if self._synthetic_test
            else build_chunk_action_layout
        )
        layout = builder(
            spec=self._layout_spec,
            proof=proof,
            row_start_raw_index=row.row_start_raw_index,
            valid_raw_count=valid_raw_count,
            video_valid_mask=video_mask,
            action_valid_mask=action_mask,
        )
        if layout.source_row_digest != row.source_row_digest:
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_INSTANCE_DIGEST_MISMATCH,
                "layout source-row digest differs from the manifest row",
            )
        if layout.episode_id_digest != row.episode_id_digest:
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_INSTANCE_DIGEST_MISMATCH,
                "layout episode digest differs from the manifest row",
            )

        sanitized = select_layout_proprio(sample, layout)
        sanitized["dataset_manifest_sha256"] = self.dataset_manifest_sha256
        sanitized["row_order_manifest_sha256"] = (
            self.row_order_manifest_sha256
        )
        sanitized["episode_origin_manifest_digest"] = (
            self.episode_origin_manifest_digest
        )
        sanitized["episode_origin_row_digest"] = row.row_manifest_entry_digest
        return sanitized


__all__ = [
    "CachEpisodeOriginDataset",
    "EpisodeOriginManifestRow",
    "select_layout_proprio",
    "validate_cach_episode_origin_config",
]
