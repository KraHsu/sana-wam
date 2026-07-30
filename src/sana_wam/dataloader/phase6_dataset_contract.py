"""Byte-level dataset contract for the frozen Phase-6 task plan.

The task-plan v1 identity intentionally contains only sampling metadata.  This
second, independently pinned artifact binds every planned row to its logical
window length, episode bytes, instruction source, and preprocessing geometry.
The module is torch-free so builders and CPU audit tests can use it directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
import re
from typing import Any

from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    EXPANSION_SUPPORT_KEYS,
    ExpansionSupportError,
    expansion_support_contract,
    expansion_support_for_child_window,
    validate_expansion_support,
)
from sana_wam.dataloader.phase6_padding_semantics import (
    PADDING_SEMANTICS_AMENDMENT_SHA256,
    PADDING_SEMANTICS_CONTRACT_SHA256,
    PaddingSemanticsError,
    padding_semantics_contract,
    validate_padding_eligibility_dependency,
    validate_padding_semantics_contract,
)
from sana_wam.dataloader.task_sample_plan import PlanValidationError


DATASET_CONTRACT_SCHEMA_VERSION = "sana-phase6-dataset-contract-v2"
INSTRUCTION_FALLBACK_SCHEMA_VERSION = "sana-phase6-instruction-fallback-descriptor-v1"
EXPECTED_ROW_COUNT = 504
EXPECTED_TASK_COUNT = 42

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "plan_sha256",
        "identity_sha256",
        "action_stats_sha256",
        "expansion_eligibility_amendment_sha256",
        "expansion_support_contract",
        "expansion_support_contract_sha256",
        "padding_semantics_amendment_sha256",
        "padding_semantics_contract",
        "padding_semantics_contract_sha256",
        "preprocessing_contract",
        "rows",
        "episodes",
    }
)
ROW_KEYS = frozenset(
    {
        "global_step",
        "dataset_index",
        "task_name",
        "episode_index",
        "episode_path",
        "start_frame",
        "window_logical_length",
        "episode_sha256",
        "instruction_source_sha256",
        *EXPANSION_SUPPORT_KEYS,
    }
)
EPISODE_KEYS = frozenset(
    {
        "task_name",
        "episode_index",
        "episode_path",
        "resolved_path",
        "size_bytes",
        "sha256",
        "instruction_source",
    }
)
INSTRUCTION_SOURCE_KEYS = frozenset(
    {
        "kind",
        "path",
        "resolved_path",
        "size_bytes",
        "sha256",
        "fallback_descriptor",
    }
)
FALLBACK_DESCRIPTOR_KEYS = frozenset(
    {
        "schema_version",
        "task_name",
        "episode_basename",
        "base_prompt",
    }
)
PREPROCESSING_KEYS = frozenset(
    {
        "action_mode",
        "causal_temporal",
        "delta_action",
        "filter_static_segments",
        "growing_history",
        "height",
        "multiview",
        "normalize_mode",
        "num_frames",
        "repeat",
        "robot",
        "split",
        "temporal_compression",
        "text_embedding_dropout",
        "val_ratio",
        "variant",
        "video_stride",
        "width",
        "window_stride",
        "camera_layout",
        "target_camera",
        "action_stats_path",
        "action_stats_sha256",
        "vae_type",
        "raw_window_len",
        "num_video_frames",
        "video_sample_indices",
        "history_min_frames",
        "history_stride",
        "gdn_chunk_size",
        "static_segment_threshold",
        "max_static_retry",
        "text_embedding_cache_dir",
        "vae_cache_dir",
        "dataset_dir",
        "seed",
        "num_val_samples",
        "backbone",
        "robotwin_dataset_source_sha256",
        "preprocessing_source_manifest",
        "preprocessing_source_manifest_sha256",
        "canonical_train_tasks",
        "canonical_train_tasks_sha256",
        "canonical_train_task_order_sha256",
    }
)
PREPROCESSING_SOURCE_IDS = tuple(
    sorted(
        (
            "sana_wam.dataloader.robotwin_dataset",
            "sana_wam.dataloader.transforms.multiview",
            "sana_wam.dataloader.transforms.normalize",
            "sana_wam.dataloader.transforms.rotation",
        )
    )
)
PREPROCESSING_SOURCE_ENTRY_KEYS = frozenset(
    {"source_id", "path", "resolved_path", "size_bytes", "sha256"}
)


class DatasetContractError(PlanValidationError):
    """Raised when the second Phase-6 artifact or runtime source drifts."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise DatasetContractError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc
    return payload + b"\n"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(data: bytes) -> Any:
    if not isinstance(data, bytes):
        raise DatasetContractError("dataset-contract artifact must be bytes")
    try:
        return json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                DatasetContractError(f"non-finite JSON value: {value}")
            ),
        )
    except DatasetContractError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError) as exc:
        raise DatasetContractError(f"invalid dataset-contract JSON: {exc}") from exc


def _require_exact_keys(value: Any, expected: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise DatasetContractError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != set(expected):
        raise DatasetContractError(
            f"{name} keys differ; missing={sorted(set(expected) - actual)!r}, "
            f"extra={sorted(actual - set(expected))!r}"
        )


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DatasetContractError(f"{name} must be a lowercase SHA256 digest")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetContractError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DatasetContractError(f"{name} must be an integer >= {minimum}")
    return value


def _require_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise DatasetContractError(f"{name} must be a finite JSON float")
    if minimum is not None and value < minimum:
        raise DatasetContractError(f"{name} must be >= {minimum}")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise DatasetContractError(f"{name} must be a JSON boolean")
    return value


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _resolved_path(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(_absolute_path(path))


def _hash_stable_file(
    path: str,
    *,
    file_hasher: Callable[[str], str],
) -> tuple[str, int, str]:
    """Hash a stable target and bind both symlink spelling and real target."""

    absolute = _absolute_path(path)
    if not os.path.isfile(absolute):
        raise DatasetContractError(f"contract source is not a file: {absolute}")
    resolved_before = _resolved_path(absolute)
    stat_before = os.stat(absolute)
    digest = file_hasher(absolute)
    _require_sha256(digest, f"SHA256 for {absolute}")
    stat_after = os.stat(absolute)
    resolved_after = _resolved_path(absolute)
    stable_before = (
        resolved_before,
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
    )
    stable_after = (
        resolved_after,
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
    )
    if stable_before != stable_after:
        raise DatasetContractError(f"contract source changed while hashing: {absolute}")
    return resolved_before, stat_before.st_size, digest


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        value = config.get(key, default)
    elif hasattr(config, "get"):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    return default if value is None and default is not None else value


def _config_int(config: Any, key: str, default: int) -> int:
    value = _config_get(config, key, default)
    if type(value) is not int:
        raise DatasetContractError(f"dataloader.{key} must be an integer")
    return value


def _config_float(config: Any, key: str, default: float) -> float:
    value = _config_get(config, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetContractError(f"dataloader.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DatasetContractError(f"dataloader.{key} must be finite")
    return result


def _task_list_sha256(tasks: Sequence[str]) -> str:
    payload = "".join(f"{task}\n" for task in sorted(tasks)).encode("utf-8")
    return sha256(payload).hexdigest()


def _task_order_sha256(tasks: Sequence[str]) -> str:
    return sha256(_canonical_json_bytes(list(tasks))).hexdigest()


def _children(dataset: Any) -> list[Any]:
    children = getattr(dataset, "_sub_datasets", None)
    if not isinstance(children, list) or len(children) != EXPECTED_TASK_COUNT:
        raise DatasetContractError(
            f"Phase-6 runtime must expose exactly {EXPECTED_TASK_COUNT} task children"
        )
    return children


def _uniform(children: Sequence[Any], attribute: str) -> Any:
    values = [getattr(child, attribute, object()) for child in children]
    if any(type(value) is object for value in values):
        raise DatasetContractError(
            f"runtime child is missing preprocessing attribute {attribute!r}"
        )
    first = values[0]
    if any(value != first for value in values[1:]):
        raise DatasetContractError(
            f"runtime preprocessing attribute {attribute!r} differs across tasks"
        )
    return first


def _normalize_optional_config_path(value: Any, name: str) -> None:
    if value not in (None, ""):
        raise DatasetContractError(f"Phase-6 requires {name}=null")
    return None


def _collect_preprocessing_source_manifest(
    source_paths: Mapping[str, str | os.PathLike[str]],
    *,
    file_hasher: Callable[[str], str],
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(source_paths, Mapping):
        raise DatasetContractError("preprocessing_source_paths must be a mapping")
    if set(source_paths) != set(PREPROCESSING_SOURCE_IDS):
        raise DatasetContractError(
            "preprocessing source IDs differ from the frozen four-file manifest"
        )
    manifest = []
    for source_id in PREPROCESSING_SOURCE_IDS:
        path = _absolute_path(source_paths[source_id])
        resolved, size_bytes, digest = _hash_stable_file(
            path,
            file_hasher=file_hasher,
        )
        manifest.append(
            {
                "source_id": source_id,
                "path": path,
                "resolved_path": resolved,
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )
    return manifest, sha256(_canonical_json_bytes(manifest)).hexdigest()


def collect_preprocessing_contract(
    dataset: Any,
    dataloader_config: Any,
    *,
    action_stats_sha256: str,
    preprocessing_source_paths: Mapping[str, str | os.PathLike[str]],
    train_tasks: Sequence[str],
    file_hasher: Callable[[str], str] = sha256_file,
) -> dict[str, Any]:
    """Observe the exact preprocessing state used by a constructed dataset."""

    action_stats_digest = _require_sha256(action_stats_sha256, "action_stats_sha256")
    if not isinstance(train_tasks, Sequence) or isinstance(train_tasks, (str, bytes)):
        raise DatasetContractError("train_tasks must be an ordered sequence")
    tasks = tuple(train_tasks)
    if len(tasks) != EXPECTED_TASK_COUNT or any(
        not isinstance(task, str) or not task for task in tasks
    ):
        raise DatasetContractError("train_tasks must contain 42 non-empty strings")
    runtime_tasks = getattr(dataset, "task_names", None)
    runtime_tasks = runtime_tasks() if callable(runtime_tasks) else runtime_tasks
    if tuple(runtime_tasks or ()) != tasks:
        raise DatasetContractError("runtime task cohort/order differs from train_tasks")

    children = _children(dataset)
    source_manifest, source_manifest_digest = _collect_preprocessing_source_manifest(
        preprocessing_source_paths,
        file_hasher=file_hasher,
    )
    source_digests = {entry["source_id"]: entry["sha256"] for entry in source_manifest}
    action_stats_path = getattr(dataset, "action_stats_path", None)
    if not action_stats_path:
        raise DatasetContractError("runtime dataset has no action_stats_path")

    camera_layout = _uniform(children, "camera_layout")
    if camera_layout is not None:
        if not isinstance(camera_layout, (list, tuple)):
            raise DatasetContractError(
                "runtime camera_layout must be a sequence or null"
            )
        camera_layout = list(camera_layout)

    text_cache = _normalize_optional_config_path(
        _config_get(dataloader_config, "text_embedding_cache_dir", None),
        "text_embedding_cache_dir",
    )
    vae_cache = _normalize_optional_config_path(
        _config_get(dataloader_config, "vae_cache_dir", None),
        "vae_cache_dir",
    )
    if any(
        getattr(child, "_text_embedding_transform", None) is not None
        for child in children
    ):
        raise DatasetContractError("Phase-6 text embedding cache must be disabled")
    if any(
        getattr(child, "_vae_latent_transform", None) is not None for child in children
    ):
        raise DatasetContractError("Phase-6 VAE cache must be disabled")

    dataset_dir = _config_get(dataloader_config, "dataset_dir", None)
    if not dataset_dir:
        raise DatasetContractError("dataloader.dataset_dir must be configured")
    normalize_mode = _uniform(children, "normalize_mode")
    if normalize_mode is not None and not isinstance(normalize_mode, str):
        raise DatasetContractError("runtime normalize_mode must be a string or null")

    value = {
        "action_mode": _uniform(children, "action_mode"),
        "causal_temporal": _uniform(children, "causal_temporal"),
        "delta_action": _uniform(children, "delta_action"),
        "filter_static_segments": _uniform(children, "_filter_static_segments"),
        "growing_history": _uniform(children, "growing_history"),
        "height": _uniform(children, "height"),
        "multiview": _uniform(children, "multiview"),
        "normalize_mode": normalize_mode,
        "num_frames": _uniform(children, "num_frames"),
        "repeat": _uniform(children, "repeat"),
        "robot": _uniform(children, "robot"),
        "split": _uniform(children, "split"),
        "temporal_compression": _uniform(children, "temporal_compression"),
        "text_embedding_dropout": _config_float(
            dataloader_config, "text_embedding_dropout", 0.0
        ),
        "val_ratio": _config_float(dataloader_config, "val_ratio", 0.0),
        "variant": getattr(dataset, "variant", None),
        "video_stride": _uniform(children, "video_stride"),
        "width": _uniform(children, "width"),
        "window_stride": _uniform(children, "window_stride"),
        "camera_layout": camera_layout,
        "target_camera": _uniform(children, "target_camera"),
        "action_stats_path": _resolved_path(action_stats_path),
        "action_stats_sha256": action_stats_digest,
        "vae_type": _uniform(children, "vae_type"),
        "raw_window_len": _uniform(children, "_raw_window_len"),
        "num_video_frames": _uniform(children, "num_video_frames"),
        "video_sample_indices": list(_uniform(children, "_video_sample_indices")),
        "history_min_frames": _uniform(children, "history_min_frames"),
        "history_stride": _uniform(children, "history_stride"),
        "gdn_chunk_size": _uniform(children, "gdn_chunk_size"),
        "static_segment_threshold": _uniform(children, "_static_segment_threshold"),
        "max_static_retry": _uniform(children, "_max_static_retry"),
        "text_embedding_cache_dir": text_cache,
        "vae_cache_dir": vae_cache,
        "dataset_dir": _resolved_path(dataset_dir),
        "seed": _config_int(dataloader_config, "seed", 42),
        "num_val_samples": _config_int(dataloader_config, "num_val_samples", 4),
        "backbone": _config_get(dataloader_config, "backbone", None),
        "robotwin_dataset_source_sha256": source_digests[
            "sana_wam.dataloader.robotwin_dataset"
        ],
        "preprocessing_source_manifest": source_manifest,
        "preprocessing_source_manifest_sha256": source_manifest_digest,
        "canonical_train_tasks": list(tasks),
        "canonical_train_tasks_sha256": _task_list_sha256(tasks),
        "canonical_train_task_order_sha256": _task_order_sha256(tasks),
    }
    _validate_preprocessing_contract(value)
    return value


def _resolve_runtime_window(dataset: Any, dataset_index: int) -> dict[str, Any]:
    if type(dataset_index) is not int or dataset_index < 0:
        raise DatasetContractError("dataset_index must be a non-negative integer")
    try:
        child_index, local_index = dataset.resolve_global_index(dataset_index)
        children = _children(dataset)
        child = children[child_index]
        episode_index, start_frame, logical_length = child._resolve_local_index(
            local_index
        )
        episode_path = child._episode_files[episode_index]
        support = expansion_support_for_child_window(
            child,
            episode_index=episode_index,
            start_frame=start_frame,
            window_logical_length=logical_length,
        )
    except Exception as exc:
        raise DatasetContractError(
            f"cannot resolve runtime window for dataset index {dataset_index}: {exc}"
        ) from exc
    return {
        "dataset_index": dataset_index,
        "task_name": child.task_name,
        "episode_index": episode_index,
        "episode_path": os.fspath(episode_path),
        "start_frame": start_frame,
        "window_logical_length": logical_length,
        **support,
        "child": child,
    }


def runtime_window_row(
    dataset: Any,
    *,
    global_step: int,
    dataset_index: int,
) -> dict[str, Any]:
    """Return the cheap structural subset rechecked on every planned access."""

    _require_int(global_step, "global_step", minimum=1)
    observed = _resolve_runtime_window(dataset, dataset_index)
    observed.pop("child")
    return {"global_step": global_step, **observed}


def _instruction_key(episode_path: str) -> str:
    basename = os.path.basename(episode_path)
    episode_number = basename.replace("episode", "").replace(".hdf5", "")
    return f"episode{episode_number}.json"


def _fallback_instruction_descriptor(
    child: Any,
    *,
    episode_path: str,
) -> dict[str, Any]:
    task_name = getattr(child, "prompt_task_name", None)
    _require_string(task_name, "fallback prompt task_name")
    basename = os.path.basename(episode_path)
    descriptor = {
        "schema_version": INSTRUCTION_FALLBACK_SCHEMA_VERSION,
        "task_name": task_name,
        "episode_basename": basename,
        "base_prompt": f"The bimanual robot is performing a {task_name} task.",
    }
    payload = _canonical_json_bytes(descriptor)
    fallback_path = f"fallback://robotwin/{task_name}/{_instruction_key(episode_path)}"
    return {
        "kind": "deterministic_fallback",
        "path": fallback_path,
        "resolved_path": None,
        "size_bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
        "fallback_descriptor": descriptor,
    }


def _instruction_source_descriptor(
    child: Any,
    *,
    episode_path: str,
    file_hasher: Callable[[str], str],
    file_cache: dict[str, tuple[str, int, str]],
) -> dict[str, Any]:
    instructions = getattr(child, "_instructions", None)
    if not isinstance(instructions, Mapping):
        raise DatasetContractError("runtime child instructions must be a mapping")
    key = _instruction_key(episode_path)
    if key not in instructions:
        return _fallback_instruction_descriptor(child, episode_path=episode_path)

    data_root = _require_string(getattr(child, "data_root", None), "data_root")
    path = _absolute_path(os.path.join(os.path.dirname(data_root), "instructions", key))
    if path not in file_cache:
        file_cache[path] = _hash_stable_file(path, file_hasher=file_hasher)
    resolved, size_bytes, digest = file_cache[path]
    return {
        "kind": "json_file",
        "path": path,
        "resolved_path": resolved,
        "size_bytes": size_bytes,
        "sha256": digest,
        "fallback_descriptor": None,
    }


def _episode_descriptor(
    child: Any,
    *,
    task_name: str,
    episode_index: int,
    episode_path: str,
    file_hasher: Callable[[str], str],
    file_cache: dict[str, tuple[str, int, str]],
) -> dict[str, Any]:
    path = _absolute_path(episode_path)
    if path not in file_cache:
        file_cache[path] = _hash_stable_file(path, file_hasher=file_hasher)
    resolved, size_bytes, digest = file_cache[path]
    instruction = _instruction_source_descriptor(
        child,
        episode_path=episode_path,
        file_hasher=file_hasher,
        file_cache=file_cache,
    )
    return {
        "task_name": task_name,
        "episode_index": episode_index,
        "episode_path": episode_path,
        "resolved_path": resolved,
        "size_bytes": size_bytes,
        "sha256": digest,
        "instruction_source": instruction,
    }


def _validate_instruction_source(value: Any, name: str) -> None:
    _require_exact_keys(value, INSTRUCTION_SOURCE_KEYS, name)
    kind = value["kind"]
    if kind not in ("json_file", "deterministic_fallback"):
        raise DatasetContractError(f"{name}.kind is unsupported")
    _require_string(value["path"], f"{name}.path")
    _require_int(value["size_bytes"], f"{name}.size_bytes")
    _require_sha256(value["sha256"], f"{name}.sha256")
    if kind == "json_file":
        _require_string(value["resolved_path"], f"{name}.resolved_path")
        if value["fallback_descriptor"] is not None:
            raise DatasetContractError(
                f"{name}.fallback_descriptor must be null for json_file"
            )
    else:
        if value["resolved_path"] is not None:
            raise DatasetContractError(
                f"{name}.resolved_path must be null for fallback"
            )
        descriptor = value["fallback_descriptor"]
        _require_exact_keys(
            descriptor, FALLBACK_DESCRIPTOR_KEYS, f"{name}.fallback_descriptor"
        )
        if descriptor["schema_version"] != INSTRUCTION_FALLBACK_SCHEMA_VERSION:
            raise DatasetContractError(f"{name} fallback schema differs")
        for key in ("task_name", "episode_basename", "base_prompt"):
            _require_string(descriptor[key], f"{name}.fallback_descriptor.{key}")
        payload = _canonical_json_bytes(descriptor)
        if (
            len(payload) != value["size_bytes"]
            or sha256(payload).hexdigest() != value["sha256"]
        ):
            raise DatasetContractError(
                f"{name} fallback size/hash differs from descriptor"
            )


def _validate_preprocessing_contract(value: Any) -> None:
    _require_exact_keys(value, PREPROCESSING_KEYS, "preprocessing_contract")
    for key in (
        "action_mode",
        "robot",
        "split",
        "variant",
        "target_camera",
        "action_stats_path",
        "vae_type",
        "dataset_dir",
    ):
        _require_string(value[key], f"preprocessing_contract.{key}")
    if value["normalize_mode"] is not None:
        _require_string(
            value["normalize_mode"], "preprocessing_contract.normalize_mode"
        )
    if value["backbone"] is not None:
        _require_string(value["backbone"], "preprocessing_contract.backbone")
    for key in (
        "causal_temporal",
        "delta_action",
        "filter_static_segments",
        "growing_history",
        "multiview",
    ):
        _require_bool(value[key], f"preprocessing_contract.{key}")
    for key in (
        "height",
        "num_frames",
        "repeat",
        "temporal_compression",
        "video_stride",
        "width",
        "window_stride",
        "raw_window_len",
        "num_video_frames",
        "history_min_frames",
        "history_stride",
        "gdn_chunk_size",
        "max_static_retry",
    ):
        _require_int(value[key], f"preprocessing_contract.{key}", minimum=1)
    for key in ("seed", "num_val_samples"):
        _require_int(value[key], f"preprocessing_contract.{key}")
    for key in (
        "text_embedding_dropout",
        "val_ratio",
        "static_segment_threshold",
    ):
        _require_float(value[key], f"preprocessing_contract.{key}", minimum=0.0)
    for key in ("text_embedding_cache_dir", "vae_cache_dir"):
        if value[key] is not None:
            raise DatasetContractError(f"preprocessing_contract.{key} must be null")
    _require_sha256(
        value["action_stats_sha256"],
        "preprocessing_contract.action_stats_sha256",
    )
    _require_sha256(
        value["robotwin_dataset_source_sha256"],
        "preprocessing_contract.robotwin_dataset_source_sha256",
    )
    source_manifest = value["preprocessing_source_manifest"]
    if not isinstance(source_manifest, list) or len(source_manifest) != len(
        PREPROCESSING_SOURCE_IDS
    ):
        raise DatasetContractError(
            "preprocessing_source_manifest must contain the frozen four files"
        )
    observed_source_ids = []
    for index, entry in enumerate(source_manifest):
        name = f"preprocessing_contract.preprocessing_source_manifest[{index}]"
        _require_exact_keys(entry, PREPROCESSING_SOURCE_ENTRY_KEYS, name)
        observed_source_ids.append(
            _require_string(entry["source_id"], f"{name}.source_id")
        )
        _require_string(entry["path"], f"{name}.path")
        _require_string(entry["resolved_path"], f"{name}.resolved_path")
        _require_int(entry["size_bytes"], f"{name}.size_bytes")
        _require_sha256(entry["sha256"], f"{name}.sha256")
    if tuple(observed_source_ids) != PREPROCESSING_SOURCE_IDS:
        raise DatasetContractError(
            "preprocessing source manifest IDs/order differ from the frozen contract"
        )
    source_manifest_digest = _require_sha256(
        value["preprocessing_source_manifest_sha256"],
        "preprocessing_contract.preprocessing_source_manifest_sha256",
    )
    if (
        sha256(_canonical_json_bytes(source_manifest)).hexdigest()
        != source_manifest_digest
    ):
        raise DatasetContractError("preprocessing source-manifest SHA mismatch")
    robotwin_entry = next(
        entry
        for entry in source_manifest
        if entry["source_id"] == "sana_wam.dataloader.robotwin_dataset"
    )
    if robotwin_entry["sha256"] != value["robotwin_dataset_source_sha256"]:
        raise DatasetContractError("robotwin source SHA differs from source manifest")
    _require_sha256(
        value["canonical_train_tasks_sha256"],
        "preprocessing_contract.canonical_train_tasks_sha256",
    )
    _require_sha256(
        value["canonical_train_task_order_sha256"],
        "preprocessing_contract.canonical_train_task_order_sha256",
    )
    tasks = value["canonical_train_tasks"]
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_TASK_COUNT:
        raise DatasetContractError("canonical_train_tasks must be a 42-item list")
    if len(set(tasks)) != len(tasks) or any(
        not isinstance(task, str) or not task for task in tasks
    ):
        raise DatasetContractError("canonical_train_tasks must be unique strings")
    if _task_list_sha256(tasks) != value["canonical_train_tasks_sha256"]:
        raise DatasetContractError("canonical train-task set hash mismatch")
    if _task_order_sha256(tasks) != value["canonical_train_task_order_sha256"]:
        raise DatasetContractError("canonical train-task order hash mismatch")
    layout = value["camera_layout"]
    if value["multiview"]:
        if (
            not isinstance(layout, list)
            or len(layout) != 3
            or any(not isinstance(camera, str) or not camera for camera in layout)
        ):
            raise DatasetContractError("multiview camera_layout must contain 3 cameras")
    elif layout is not None:
        raise DatasetContractError("single-view camera_layout must be null")
    indices = value["video_sample_indices"]
    if not isinstance(indices, list) or any(
        type(index) is not int or index < 0 for index in indices
    ):
        raise DatasetContractError("video_sample_indices must be non-negative integers")
    expected_indices = list(range(0, value["num_frames"], value["video_stride"]))
    if indices != expected_indices or len(indices) != value["num_video_frames"]:
        raise DatasetContractError("video sampling geometry is internally inconsistent")
    temporal_count = value["num_video_frames"]
    temporal_compression = value["temporal_compression"]
    if value["causal_temporal"]:
        temporal_valid = (temporal_count - 1) % temporal_compression == 0
    else:
        temporal_valid = temporal_count % temporal_compression == 0
    if not temporal_valid:
        raise DatasetContractError("video/VAE temporal geometry is inconsistent")
    if value["raw_window_len"] != value["num_frames"]:
        raise DatasetContractError(
            "ordinary Phase-6 raw_window_len must equal num_frames"
        )
    if value["action_stats_sha256"] == "0" * 64:
        raise DatasetContractError("action-stats digest cannot be all zero")
    if value["split"] != "train" or value["variant"] != "clean_50":
        raise DatasetContractError("Phase-6 requires train/clean_50 source data")
    if value["repeat"] != 1 or value["growing_history"]:
        raise DatasetContractError("Phase-6 requires repeat=1 ordinary windows")
    if value["filter_static_segments"]:
        raise DatasetContractError("Phase-6 static-segment retry must be disabled")
    if value["text_embedding_dropout"] != 0.0:
        raise DatasetContractError("Phase-6 text embedding dropout must be zero")
    if value["val_ratio"] > 1.0:
        raise DatasetContractError("preprocessing_contract.val_ratio must be <= 1.0")


def _validate_artifact_dict(value: Any) -> None:
    _require_exact_keys(value, TOP_LEVEL_KEYS, "dataset contract")
    if value["schema_version"] != DATASET_CONTRACT_SCHEMA_VERSION:
        raise DatasetContractError("dataset-contract schema_version differs")
    for key in ("plan_sha256", "identity_sha256", "action_stats_sha256"):
        _require_sha256(value[key], key)
    _require_sha256(
        value["expansion_support_contract_sha256"],
        "expansion_support_contract_sha256",
    )
    _require_sha256(
        value["expansion_eligibility_amendment_sha256"],
        "expansion_eligibility_amendment_sha256",
    )
    _require_sha256(
        value["padding_semantics_contract_sha256"],
        "padding_semantics_contract_sha256",
    )
    _require_sha256(
        value["padding_semantics_amendment_sha256"],
        "padding_semantics_amendment_sha256",
    )
    support_contract = value["expansion_support_contract"]
    try:
        support_contract_sha256 = sha256(
            _canonical_json_bytes(support_contract)
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise DatasetContractError(
            "dataset expansion-support contract is not canonical JSON data"
        ) from exc
    if (
        support_contract_sha256 != EXPANSION_SUPPORT_CONTRACT_SHA256
        or support_contract != expansion_support_contract()
        or value["expansion_support_contract_sha256"]
        != EXPANSION_SUPPORT_CONTRACT_SHA256
    ):
        raise DatasetContractError("dataset expansion-support contract differs")
    if (
        value["expansion_eligibility_amendment_sha256"]
        != EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
    ):
        raise DatasetContractError("dataset eligibility-amendment SHA256 differs")
    try:
        validate_padding_semantics_contract(value["padding_semantics_contract"])
    except PaddingSemanticsError as exc:
        raise DatasetContractError(
            "dataset padding-semantics contract differs"
        ) from exc
    if (
        value["padding_semantics_contract_sha256"]
        != PADDING_SEMANTICS_CONTRACT_SHA256
    ):
        raise DatasetContractError("dataset padding-semantics contract differs")
    if (
        value["padding_semantics_amendment_sha256"]
        != PADDING_SEMANTICS_AMENDMENT_SHA256
    ):
        raise DatasetContractError("dataset padding-semantics amendment SHA256 differs")
    try:
        validate_padding_eligibility_dependency(
            expansion_eligibility_amendment_sha256=value[
                "expansion_eligibility_amendment_sha256"
            ],
            expansion_support_contract_sha256=value[
                "expansion_support_contract_sha256"
            ],
        )
    except PaddingSemanticsError as exc:
        raise DatasetContractError(
            "dataset padding/eligibility provenance dependency differs"
        ) from exc
    _validate_preprocessing_contract(value["preprocessing_contract"])
    if (
        value["preprocessing_contract"]["action_stats_sha256"]
        != value["action_stats_sha256"]
    ):
        raise DatasetContractError("top-level/preprocessing action-stats SHA mismatch")

    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROW_COUNT:
        raise DatasetContractError(
            f"rows must contain exactly {EXPECTED_ROW_COUNT} entries"
        )
    seen_indices: set[int] = set()
    for offset, row in enumerate(rows, start=1):
        name = f"rows[{offset - 1}]"
        _require_exact_keys(row, ROW_KEYS, name)
        if _require_int(row["global_step"], f"{name}.global_step", minimum=1) != offset:
            raise DatasetContractError(f"{name}.global_step is not ordered")
        dataset_index = _require_int(row["dataset_index"], f"{name}.dataset_index")
        if dataset_index in seen_indices:
            raise DatasetContractError(f"plan reuses dataset index {dataset_index}")
        seen_indices.add(dataset_index)
        _require_string(row["task_name"], f"{name}.task_name")
        _require_int(row["episode_index"], f"{name}.episode_index")
        _require_string(row["episode_path"], f"{name}.episode_path")
        _require_int(row["start_frame"], f"{name}.start_frame")
        _require_int(
            row["window_logical_length"],
            f"{name}.window_logical_length",
            minimum=2,
        )
        if row["window_logical_length"] > value["preprocessing_contract"]["num_frames"]:
            raise DatasetContractError(
                f"{name}.window_logical_length exceeds preprocessing num_frames"
            )
        _require_sha256(row["episode_sha256"], f"{name}.episode_sha256")
        _require_sha256(
            row["instruction_source_sha256"],
            f"{name}.instruction_source_sha256",
        )
        try:
            support = validate_expansion_support(
                {key: row[key] for key in EXPANSION_SUPPORT_KEYS},
                require_eligible=True,
            )
        except ExpansionSupportError as exc:
            raise DatasetContractError(
                f"{name} has invalid structural expansion support: {exc}"
            ) from exc
        if support["actual_valid_raw_frames"] > row["window_logical_length"]:
            raise DatasetContractError(
                f"{name}.actual_valid_raw_frames exceeds its logical window"
            )

    episodes = value["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise DatasetContractError("episodes must be a non-empty list")
    episode_map: dict[str, Mapping[str, Any]] = {}
    for offset, episode in enumerate(episodes):
        name = f"episodes[{offset}]"
        _require_exact_keys(episode, EPISODE_KEYS, name)
        task_name = _require_string(episode["task_name"], f"{name}.task_name")
        episode_index = _require_int(episode["episode_index"], f"{name}.episode_index")
        episode_path = _require_string(episode["episode_path"], f"{name}.episode_path")
        _require_string(episode["resolved_path"], f"{name}.resolved_path")
        _require_int(episode["size_bytes"], f"{name}.size_bytes")
        _require_sha256(episode["sha256"], f"{name}.sha256")
        _validate_instruction_source(
            episode["instruction_source"], f"{name}.instruction_source"
        )
        if episode_path in episode_map:
            raise DatasetContractError(f"duplicate episode descriptor: {episode_path}")
        episode_map[episode_path] = episode
        if (
            not task_name or episode_index < 0
        ):  # keeps narrowed values live for type checkers
            raise AssertionError("unreachable")
    if [episode["episode_path"] for episode in episodes] != sorted(episode_map):
        raise DatasetContractError("episodes must be sorted by exact episode_path")

    referenced: set[str] = set()
    for row in rows:
        episode = episode_map.get(row["episode_path"])
        if episode is None:
            raise DatasetContractError("row references an absent episode descriptor")
        referenced.add(row["episode_path"])
        if (
            row["task_name"] != episode["task_name"]
            or row["episode_index"] != episode["episode_index"]
            or row["episode_sha256"] != episode["sha256"]
            or row["instruction_source_sha256"]
            != episode["instruction_source"]["sha256"]
        ):
            raise DatasetContractError("row and episode descriptors disagree")
    if referenced != set(episode_map):
        raise DatasetContractError("episodes contains an unreferenced descriptor")


class Phase6DatasetContract:
    """Validated immutable-by-copy view of one canonical artifact."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        copied = deepcopy(dict(value))
        _validate_artifact_dict(copied)
        self._value = copied
        self._rows_by_dataset_index = {
            row["dataset_index"]: deepcopy(row) for row in copied["rows"]
        }

    @property
    def schema_version(self) -> str:
        return self._value["schema_version"]

    @property
    def plan_sha256(self) -> str:
        return self._value["plan_sha256"]

    @property
    def identity_sha256(self) -> str:
        return self._value["identity_sha256"]

    @property
    def action_stats_sha256(self) -> str:
        return self._value["action_stats_sha256"]

    @property
    def expansion_support_contract(self) -> dict[str, Any]:
        return deepcopy(self._value["expansion_support_contract"])

    @property
    def expansion_support_contract_sha256(self) -> str:
        return self._value["expansion_support_contract_sha256"]

    @property
    def expansion_eligibility_amendment_sha256(self) -> str:
        return self._value["expansion_eligibility_amendment_sha256"]

    @property
    def padding_semantics_contract(self) -> dict[str, str]:
        return deepcopy(self._value["padding_semantics_contract"])

    @property
    def padding_semantics_contract_sha256(self) -> str:
        return self._value["padding_semantics_contract_sha256"]

    @property
    def padding_semantics_amendment_sha256(self) -> str:
        return self._value["padding_semantics_amendment_sha256"]

    @property
    def preprocessing_contract(self) -> dict[str, Any]:
        return deepcopy(self._value["preprocessing_contract"])

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._value["rows"]))

    @property
    def episodes(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._value["episodes"]))

    @property
    def artifact_sha256(self) -> str:
        return sha256(self.to_artifact_bytes()).hexdigest()

    def row_for_dataset_index(self, dataset_index: int) -> dict[str, Any]:
        try:
            return deepcopy(self._rows_by_dataset_index[dataset_index])
        except (KeyError, TypeError) as exc:
            raise DatasetContractError(
                f"dataset index {dataset_index!r} is absent from the dataset contract"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._value)

    def to_artifact_bytes(self) -> bytes:
        return _canonical_json_bytes(self._value)

    @classmethod
    def from_artifact_bytes(
        cls,
        data: bytes,
        *,
        expected_artifact_sha256: str | None = None,
        expected_plan_sha256: str | None = None,
        expected_identity_sha256: str | None = None,
        expected_action_stats_sha256: str | None = None,
    ) -> "Phase6DatasetContract":
        if expected_artifact_sha256 is not None:
            expected = _require_sha256(
                expected_artifact_sha256, "expected_artifact_sha256"
            )
            actual = sha256(data).hexdigest()
            if actual != expected:
                raise DatasetContractError(
                    f"dataset-contract artifact SHA mismatch: expected {expected}, got {actual}"
                )
        value = _strict_json_loads(data)
        contract = cls(value)
        if data != contract.to_artifact_bytes():
            raise DatasetContractError(
                "dataset-contract artifact is not canonical JSON"
            )
        for observed, expected, name in (
            (contract.plan_sha256, expected_plan_sha256, "plan_sha256"),
            (contract.identity_sha256, expected_identity_sha256, "identity_sha256"),
            (
                contract.action_stats_sha256,
                expected_action_stats_sha256,
                "action_stats_sha256",
            ),
        ):
            if expected is not None and observed != _require_sha256(
                expected, f"expected_{name}"
            ):
                raise DatasetContractError(
                    f"embedded {name} mismatch: expected {expected}, got {observed}"
                )
        return contract

    @classmethod
    def build(
        cls,
        dataset: Any,
        plan: Any,
        *,
        dataloader_config: Any,
        action_stats_sha256: str,
        preprocessing_source_paths: Mapping[str, str | os.PathLike[str]],
        train_tasks: Sequence[str],
        file_hasher: Callable[[str], str] = sha256_file,
    ) -> "Phase6DatasetContract":
        preprocessing = collect_preprocessing_contract(
            dataset,
            dataloader_config,
            action_stats_sha256=action_stats_sha256,
            preprocessing_source_paths=preprocessing_source_paths,
            train_tasks=train_tasks,
            file_hasher=file_hasher,
        )
        plan_digest = _require_sha256(
            getattr(plan, "plan_sha256", None), "plan.plan_sha256"
        )
        identity_digest = _require_sha256(
            getattr(plan, "identity_sha256", None), "plan.identity_sha256"
        )
        plan_rows = getattr(plan, "rows", None)
        if (
            not isinstance(plan_rows, (list, tuple))
            or len(plan_rows) != EXPECTED_ROW_COUNT
        ):
            raise DatasetContractError("plan must expose exactly 504 ordered rows")

        observed_rows: list[dict[str, Any]] = []
        episode_context: dict[str, tuple[Any, str, int]] = {}
        for expected_step, plan_row in enumerate(plan_rows, start=1):
            identity = getattr(plan_row, "identity", None)
            if (
                getattr(plan_row, "global_step", None) != expected_step
                or identity is None
            ):
                raise DatasetContractError("plan rows are not ordered global steps")
            runtime = _resolve_runtime_window(dataset, identity.dataset_index)
            exact = {
                "task_name": identity.task_name,
                "episode_index": identity.episode_index,
                "episode_path": identity.episode_path,
                "start_frame": identity.start_frame,
            }
            if any(runtime[key] != expected for key, expected in exact.items()):
                raise DatasetContractError(
                    f"plan/runtime structural mismatch at global step {expected_step}"
                )
            context = (runtime["child"], identity.task_name, identity.episode_index)
            prior = episode_context.get(identity.episode_path)
            if prior is not None and (prior[1] != context[1] or prior[2] != context[2]):
                raise DatasetContractError(
                    "one episode path maps to conflicting identities"
                )
            episode_context[identity.episode_path] = context
            observed_rows.append(
                {
                    "global_step": expected_step,
                    "dataset_index": identity.dataset_index,
                    "task_name": identity.task_name,
                    "episode_index": identity.episode_index,
                    "episode_path": identity.episode_path,
                    "start_frame": identity.start_frame,
                    "window_logical_length": runtime["window_logical_length"],
                    **{key: runtime[key] for key in EXPANSION_SUPPORT_KEYS},
                }
            )

        file_cache: dict[str, tuple[str, int, str]] = {}
        episodes: list[dict[str, Any]] = []
        for episode_path in sorted(episode_context):
            child, task_name, episode_index = episode_context[episode_path]
            episodes.append(
                _episode_descriptor(
                    child,
                    task_name=task_name,
                    episode_index=episode_index,
                    episode_path=episode_path,
                    file_hasher=file_hasher,
                    file_cache=file_cache,
                )
            )
        episode_map = {episode["episode_path"]: episode for episode in episodes}
        rows = []
        for observed in observed_rows:
            episode = episode_map[observed["episode_path"]]
            rows.append(
                {
                    **observed,
                    "episode_sha256": episode["sha256"],
                    "instruction_source_sha256": episode["instruction_source"][
                        "sha256"
                    ],
                }
            )
        return cls(
            {
                "schema_version": DATASET_CONTRACT_SCHEMA_VERSION,
                "plan_sha256": plan_digest,
                "identity_sha256": identity_digest,
                "action_stats_sha256": action_stats_sha256,
                "expansion_eligibility_amendment_sha256": (
                    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
                ),
                "expansion_support_contract": expansion_support_contract(),
                "expansion_support_contract_sha256": (
                    EXPANSION_SUPPORT_CONTRACT_SHA256
                ),
                "padding_semantics_amendment_sha256": (
                    PADDING_SEMANTICS_AMENDMENT_SHA256
                ),
                "padding_semantics_contract": padding_semantics_contract(),
                "padding_semantics_contract_sha256": (
                    PADDING_SEMANTICS_CONTRACT_SHA256
                ),
                "preprocessing_contract": preprocessing,
                "rows": rows,
                "episodes": episodes,
            }
        )

    def validate_runtime(
        self,
        dataset: Any,
        plan: Any,
        *,
        dataloader_config: Any,
        action_stats_sha256: str,
        preprocessing_source_paths: Mapping[str, str | os.PathLike[str]],
        train_tasks: Sequence[str],
        file_hasher: Callable[[str], str] = sha256_file,
    ) -> None:
        """Rehash each selected source once and validate all 504 row bindings."""

        if self.plan_sha256 != getattr(plan, "plan_sha256", None):
            raise DatasetContractError("dataset contract does not bind the loaded plan")
        if self.identity_sha256 != getattr(plan, "identity_sha256", None):
            raise DatasetContractError(
                "dataset contract does not bind loaded identities"
            )
        if self.action_stats_sha256 != _require_sha256(
            action_stats_sha256, "action_stats_sha256"
        ):
            raise DatasetContractError("dataset contract action-stats SHA mismatch")
        observed_preprocessing = collect_preprocessing_contract(
            dataset,
            dataloader_config,
            action_stats_sha256=action_stats_sha256,
            preprocessing_source_paths=preprocessing_source_paths,
            train_tasks=train_tasks,
            file_hasher=file_hasher,
        )
        if observed_preprocessing != self._value["preprocessing_contract"]:
            raise DatasetContractError(
                "runtime preprocessing contract differs from artifact"
            )

        plan_rows = getattr(plan, "rows", None)
        if (
            not isinstance(plan_rows, (list, tuple))
            or len(plan_rows) != EXPECTED_ROW_COUNT
        ):
            raise DatasetContractError("loaded plan does not contain exactly 504 rows")
        episode_context: dict[str, tuple[Any, str, int]] = {}
        for artifact_row, plan_row in zip(self._value["rows"], plan_rows):
            identity = getattr(plan_row, "identity", None)
            plan_projection = {
                "global_step": getattr(plan_row, "global_step", None),
                "dataset_index": getattr(identity, "dataset_index", None),
                "task_name": getattr(identity, "task_name", None),
                "episode_index": getattr(identity, "episode_index", None),
                "episode_path": getattr(identity, "episode_path", None),
                "start_frame": getattr(identity, "start_frame", None),
            }
            for key, expected in plan_projection.items():
                if artifact_row[key] != expected:
                    raise DatasetContractError(
                        f"dataset contract/plan mismatch at global step {artifact_row['global_step']}"
                    )
            runtime = _resolve_runtime_window(dataset, artifact_row["dataset_index"])
            runtime_projection = {
                "dataset_index": runtime["dataset_index"],
                "task_name": runtime["task_name"],
                "episode_index": runtime["episode_index"],
                "episode_path": runtime["episode_path"],
                "start_frame": runtime["start_frame"],
                "window_logical_length": runtime["window_logical_length"],
                **{key: runtime[key] for key in EXPANSION_SUPPORT_KEYS},
            }
            if any(
                artifact_row[key] != value for key, value in runtime_projection.items()
            ):
                raise DatasetContractError(
                    f"runtime row drift at global step {artifact_row['global_step']}"
                )
            episode_context.setdefault(
                artifact_row["episode_path"],
                (runtime["child"], runtime["task_name"], runtime["episode_index"]),
            )

        file_cache: dict[str, tuple[str, int, str]] = {}
        observed_episodes = []
        for expected_episode in self._value["episodes"]:
            context = episode_context.get(expected_episode["episode_path"])
            if context is None:
                raise DatasetContractError("episode descriptor has no runtime row")
            child, task_name, episode_index = context
            observed_episodes.append(
                _episode_descriptor(
                    child,
                    task_name=task_name,
                    episode_index=episode_index,
                    episode_path=expected_episode["episode_path"],
                    file_hasher=file_hasher,
                    file_cache=file_cache,
                )
            )
        if observed_episodes != self._value["episodes"]:
            raise DatasetContractError(
                "episode/instruction path, size, target, or SHA differs from artifact"
            )
