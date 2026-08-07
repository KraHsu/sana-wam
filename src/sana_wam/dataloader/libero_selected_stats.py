"""Exact exclusion-aware normalization statistics for selected LIBERO rows."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from sana_wam.dataloader.libero_selection import (
    LIBERO_SPLIT_ALGORITHM,
    LiberoEpisode,
    canonical_libero_roots,
    read_libero_episode_arrays,
    select_libero_episodes,
)
from sana_wam.dataloader.libero_stats import (
    LIBERO_ACTION_DIM,
    LIBERO_ACTION_MODE,
    LIBERO_POPULATION_MANIFEST_SCHEMA_VERSION,
    LIBERO_SELECTED_STATS_SCHEMA_VERSION,
    LIBERO_STATE_DIM,
    LIBERO_STATE_MODE,
    sha256_file,
    validate_libero_stats_payload,
)


LIBERO_SELECTION_CONTRACT_SCHEMA_VERSION = (
    "sana-wam-libero-selected-row-selection-contract-v2"
)
_METADATA_FILES = (
    "meta/info.json",
    "meta/stats_gr00t.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
)
_SHA256_LENGTH = 64
_STAT_FIELDS = ("mean", "std", "min", "max", "q01", "q99")


def canonical_json_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def population_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(manifest))
    value.pop("population_manifest_sha256", None)
    return canonical_json_sha256(value)


def _config_value(config: Any, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, path, default=default)
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return value


def build_selection_contract(
    dataset_roots: Sequence[str | Path],
    *,
    split: str,
    val_ratio: float,
    seed: int,
    excluded_episodes: Sequence[str],
) -> dict[str, Any]:
    # Canonicalization rejects the unresolved path itself when it is a
    # symlink, before resolve() could erase that provenance distinction.
    resolved = canonical_libero_roots(dataset_roots)
    if split not in {"train", "val"}:
        raise ValueError("selected-row stats split must be train or val")
    if (
        isinstance(val_ratio, bool)
        or not isinstance(val_ratio, (int, float))
        or not 0.0 <= float(val_ratio) <= 1.0
    ):
        raise ValueError("selected-row stats val_ratio must be in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("selected-row stats seed must be an integer")
    exclusions = [str(value) for value in excluded_episodes]
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("selected-row stats exclusions contain duplicates")
    return {
        "action_mode": LIBERO_ACTION_MODE,
        "action_row_policy": "unique selected episode rows [0,L-1)",
        "alignment": "observation_t_to_action_t",
        "dataset_roots": [
            {"dataset": root.name, "resolved_root": str(root)} for root in resolved
        ],
        "excluded_episodes": sorted(exclusions),
        "schema_version": LIBERO_SELECTION_CONTRACT_SCHEMA_VERSION,
        "seed": seed,
        "split": split,
        "split_algorithm": LIBERO_SPLIT_ALGORITHM,
        "state_mode": LIBERO_STATE_MODE,
        "state_row_policy": "unique selected episode rows [0,L)",
        "val_ratio": float(val_ratio),
    }


def selection_contract_from_config(config: Any) -> dict[str, Any]:
    roots = _config_value(config, "dataloader.dataset_roots", None)
    if roots is None:
        single = _config_value(config, "dataloader.dataset_root", None)
        roots = [] if single is None else [single]
    if isinstance(roots, (str, Path)):
        roots = [roots]
    return build_selection_contract(
        roots,
        split=str(_config_value(config, "dataloader.split", "train")),
        val_ratio=_config_value(config, "dataloader.val_ratio", 0.0),
        seed=_config_value(config, "dataloader.seed", 42),
        excluded_episodes=list(
            _config_value(config, "dataloader.excluded_episodes", [])
        ),
    )


def _selection_arguments(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_roots": [row["resolved_root"] for row in contract["dataset_roots"]],
        "split": contract["split"],
        "val_ratio": contract["val_ratio"],
        "seed": contract["seed"],
        "excluded_episodes": contract["excluded_episodes"],
        "require_all_exclusions": True,
    }


def _file_pin(path: Path, *, relative_path: str) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _episode_pin(episode: LiberoEpisode) -> dict[str, Any]:
    path = episode.data_path()
    relative = path.relative_to(episode.root).as_posix()
    return {
        "action_rows": episode.length - 1,
        "episode_index": episode.episode_index,
        "length": episode.length,
        "parquet": _file_pin(path, relative_path=relative),
        "state_rows": episode.length,
        "task": episode.task,
        "task_index": episode.task_index,
    }


def _excluded_episode_pin(episode: LiberoEpisode) -> dict[str, Any]:
    return {
        "episode_index": episode.episode_index,
        "length": episode.length,
        "reason": "configured complete-episode exclusion",
        "task": episode.task,
        "task_index": episode.task_index,
    }


def _assert_window_union_complete(
    episodes: Sequence[LiberoEpisode],
    *,
    action_horizon: int,
    window_stride: int,
    require_full_action_horizon: bool,
    include_terminal_full_horizon: bool,
) -> None:
    for episode in episodes:
        covered = np.zeros(episode.length - 1, dtype=np.bool_)
        if require_full_action_horizon:
            terminal_start = episode.length - (action_horizon + 1)
            starts = (
                []
                if terminal_start < 0
                else list(range(0, terminal_start + 1, window_stride))
            )
            if (
                include_terminal_full_horizon
                and starts
                and starts[-1] != terminal_start
            ):
                starts.append(terminal_start)
        else:
            starts = list(range(0, episode.length - 1, window_stride))
        for start in starts:
            actual_actions = min(action_horizon, episode.length - 1 - start)
            covered[start : start + actual_actions] = True
        if not bool(covered.all()):
            missing = np.flatnonzero(~covered).tolist()
            raise ValueError(
                "LIBERO window configuration does not supervise every selected "
                f"action row for {episode.identity}: {missing[:8]!r}"
            )


def validate_libero_window_coverage_for_config(config: Any) -> None:
    """Check training-window coverage without making it population identity."""

    num_frames = _config_value(config, "dataloader.num_frames", 33)
    action_horizon = _config_value(
        config, "dataloader.action_horizon", num_frames - 1
    )
    window_stride = _config_value(config, "dataloader.window_stride", 1)
    require_full = _config_value(
        config, "dataloader.require_full_action_horizon", False
    )
    include_terminal = _config_value(
        config, "dataloader.include_terminal_full_horizon", False
    )
    if (
        isinstance(num_frames, bool)
        or not isinstance(num_frames, int)
        or isinstance(window_stride, bool)
        or not isinstance(window_stride, int)
        or isinstance(action_horizon, bool)
        or not isinstance(action_horizon, int)
        or num_frames < 2
        or action_horizon <= 0
        or window_stride <= 0
        or not isinstance(require_full, bool)
        or not isinstance(include_terminal, bool)
        or (include_terminal and not require_full)
    ):
        raise ValueError("LIBERO training window fields are invalid")
    contract = selection_contract_from_config(config)
    selection = select_libero_episodes(**_selection_arguments(contract))
    _assert_window_union_complete(
        selection.selected,
        action_horizon=action_horizon,
        window_stride=window_stride,
        require_full_action_horizon=require_full,
        include_terminal_full_horizon=include_terminal,
    )


def build_population_manifest(
    config: Any,
) -> tuple[dict[str, Any], tuple[LiberoEpisode, ...]]:
    contract = selection_contract_from_config(config)
    selection = select_libero_episodes(**_selection_arguments(contract))
    selected_by_dataset: dict[str, list[LiberoEpisode]] = {
        root.name: [] for root in selection.roots
    }
    excluded_by_dataset: dict[str, list[LiberoEpisode]] = {
        root.name: [] for root in selection.roots
    }
    for episode in selection.selected:
        selected_by_dataset[episode.dataset].append(episode)
    for episode in selection.excluded:
        excluded_by_dataset[episode.dataset].append(episode)

    sources = []
    for root in selection.roots:
        selected = sorted(
            selected_by_dataset[root.name], key=lambda episode: episode.episode_index
        )
        excluded = sorted(
            excluded_by_dataset[root.name], key=lambda episode: episode.episode_index
        )
        metadata = {
            relative: _file_pin(root / relative, relative_path=relative)
            for relative in _METADATA_FILES
        }
        source_state_rows = sum(episode.length for episode in selected)
        source_action_rows = sum(episode.length - 1 for episode in selected)
        sources.append(
            {
                "counts": {
                    "action_rows": source_action_rows,
                    "excluded_episodes": len(excluded),
                    "selected_episodes": len(selected),
                    "state_rows": source_state_rows,
                },
                "dataset": root.name,
                "excluded_episodes": [
                    _excluded_episode_pin(episode) for episode in excluded
                ],
                "metadata": metadata,
                "resolved_root": str(root),
                "selected_episodes": [_episode_pin(episode) for episode in selected],
            }
        )

    selected_episode_count = len(selection.selected)
    excluded_episode_count = len(selection.excluded)
    state_row_count = sum(episode.length for episode in selection.selected)
    action_row_count = sum(episode.length - 1 for episode in selection.selected)
    if state_row_count - action_row_count != selected_episode_count:
        raise RuntimeError("selected-row state/action count equation differs")
    manifest = {
        "counts": {
            "action_rows": action_row_count,
            "excluded_episodes": excluded_episode_count,
            "selected_episodes": selected_episode_count,
            "source_episodes": selection.source_episode_count,
            "source_state_rows": selection.source_state_row_count,
            "state_rows": state_row_count,
            "suites": len(selection.roots),
        },
        "schema_version": LIBERO_POPULATION_MANIFEST_SCHEMA_VERSION,
        "selection_contract": contract,
        "selection_contract_sha256": canonical_json_sha256(contract),
        "sources": sources,
    }
    manifest["population_manifest_sha256"] = population_manifest_sha256(manifest)
    return manifest, selection.selected


def _statistics(rows: np.ndarray, *, dim: int) -> dict[str, np.ndarray]:
    if rows.ndim != 2 or rows.shape[1] != dim or rows.shape[0] <= 0:
        raise ValueError(f"selected LIBERO rows have invalid shape: {rows.shape}")
    values = rows.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("selected LIBERO rows contain non-finite values")
    quantiles = np.quantile(values, [0.01, 0.99], axis=0, method="linear")
    return {
        "mean": values.mean(axis=0).astype(np.float32),
        "std": np.maximum(values.std(axis=0, ddof=0), 1.0e-3).astype(np.float32),
        "min": values.min(axis=0).astype(np.float32),
        "max": values.max(axis=0).astype(np.float32),
        "q01": quantiles[0].astype(np.float32),
        "q99": quantiles[1].astype(np.float32),
    }


def build_libero_selected_row_stats(config: Any) -> dict[str, Any]:
    manifest, episodes = build_population_manifest(config)
    expected_hashes = {
        f"{source['dataset']}:{episode['episode_index']}": episode["parquet"]["sha256"]
        for source in manifest["sources"]
        for episode in source["selected_episodes"]
    }
    state_blocks = []
    action_blocks = []
    for episode in episodes:
        before = expected_hashes[episode.identity]
        states, actions, _task_indices = read_libero_episode_arrays(episode)
        after = sha256_file(episode.data_path())
        if after != before:
            raise RuntimeError(
                f"LIBERO Parquet changed while building stats: {episode.data_path()}"
            )
        state_blocks.append(states)
        action_blocks.append(actions[:-1])
    states = np.concatenate(state_blocks, axis=0)
    actions = np.concatenate(action_blocks, axis=0)
    counts = manifest["counts"]
    if states.shape != (counts["state_rows"], LIBERO_STATE_DIM):
        raise RuntimeError("selected LIBERO state row count differs")
    if actions.shape != (counts["action_rows"], LIBERO_ACTION_DIM):
        raise RuntimeError("selected LIBERO action row count differs")
    payload = {
        LIBERO_ACTION_MODE: _statistics(actions, dim=LIBERO_ACTION_DIM),
        LIBERO_STATE_MODE: _statistics(states, dim=LIBERO_STATE_DIM),
        "num_action_rows": int(actions.shape[0]),
        "num_timesteps": int(states.shape[0]),
        "population_manifest": manifest,
        "schema_version": LIBERO_SELECTED_STATS_SCHEMA_VERSION,
    }
    validate_libero_selected_stats_payload(payload)
    return payload


def _is_plain_int(value: Any, *, minimum: int = 0) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_file_pin(pin: Any, *, expected_relative_path: str | None = None) -> None:
    if not isinstance(pin, Mapping) or set(pin) != {
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("LIBERO population file pin structure differs")
    relative = pin.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("LIBERO population file pin path differs")
    relative_path = PurePosixPath(relative)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or str(relative_path) != relative
        or (expected_relative_path is not None and relative != expected_relative_path)
    ):
        raise ValueError("LIBERO population file pin path differs")
    if not _is_sha256(pin.get("sha256")) or not _is_plain_int(
        pin.get("size_bytes"), minimum=1
    ):
        raise ValueError("LIBERO population file pin digest/size differs")


def _validate_selection_contract(contract: Any) -> list[dict[str, str]]:
    expected_keys = {
        "action_mode",
        "action_row_policy",
        "alignment",
        "dataset_roots",
        "excluded_episodes",
        "schema_version",
        "seed",
        "split",
        "split_algorithm",
        "state_mode",
        "state_row_policy",
        "val_ratio",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_keys:
        raise ValueError("LIBERO selection contract structure differs")
    exact = {
        "action_mode": LIBERO_ACTION_MODE,
        "action_row_policy": "unique selected episode rows [0,L-1)",
        "alignment": "observation_t_to_action_t",
        "schema_version": LIBERO_SELECTION_CONTRACT_SCHEMA_VERSION,
        "split_algorithm": LIBERO_SPLIT_ALGORITHM,
        "state_mode": LIBERO_STATE_MODE,
        "state_row_policy": "unique selected episode rows [0,L)",
    }
    if any(contract.get(key) != value for key, value in exact.items()):
        raise ValueError("LIBERO selection contract semantic identity differs")
    if contract.get("split") not in {"train", "val"}:
        raise ValueError("LIBERO selection contract split differs")
    val_ratio = contract.get("val_ratio")
    if (
        isinstance(val_ratio, bool)
        or not isinstance(val_ratio, float)
        or not 0.0 <= val_ratio <= 1.0
        or isinstance(contract.get("seed"), bool)
        or not isinstance(contract.get("seed"), int)
    ):
        raise ValueError("LIBERO selection contract split fields differ")

    roots = contract.get("dataset_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("LIBERO selection contract roots differ")
    normalized_roots: list[dict[str, str]] = []
    for row in roots:
        if not isinstance(row, Mapping) or set(row) != {
            "dataset",
            "resolved_root",
        }:
            raise ValueError("LIBERO selection contract root structure differs")
        dataset = row.get("dataset")
        resolved_root = row.get("resolved_root")
        if (
            not isinstance(dataset, str)
            or not dataset
            or not isinstance(resolved_root, str)
            or not PurePosixPath(resolved_root).is_absolute()
            or PurePosixPath(resolved_root).name != dataset
            or str(PurePosixPath(resolved_root)) != resolved_root
        ):
            raise ValueError("LIBERO selection contract root identity differs")
        normalized_roots.append({"dataset": dataset, "resolved_root": resolved_root})
    root_names = [row["dataset"] for row in normalized_roots]
    if root_names != sorted(root_names) or len(root_names) != len(set(root_names)):
        raise ValueError("LIBERO selection contract roots are not uniquely sorted")

    exclusions = contract.get("excluded_episodes")
    if (
        not isinstance(exclusions, list)
        or exclusions != sorted(exclusions)
        or len(exclusions) != len(set(exclusions))
    ):
        raise ValueError("LIBERO selection contract exclusions differ")
    for exclusion in exclusions:
        if not isinstance(exclusion, str):
            raise ValueError("LIBERO selection contract exclusion identity differs")
        try:
            dataset, raw_index = exclusion.rsplit(":", 1)
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "LIBERO selection contract exclusion identity differs"
            ) from exc
        if not dataset or index < 0 or raw_index != str(index):
            raise ValueError("LIBERO selection contract exclusion identity differs")
    return normalized_roots


def validate_libero_selected_stats_payload(
    payload: Mapping[str, Any],
    *,
    expected_selection_contract: Mapping[str, Any] | None = None,
) -> None:
    validate_libero_stats_payload(payload)
    if payload.get("schema_version") != LIBERO_SELECTED_STATS_SCHEMA_VERSION:
        raise ValueError("formal LIBERO stats require selected-row schema v2")
    if set(payload) != {
        LIBERO_ACTION_MODE,
        LIBERO_STATE_MODE,
        "num_action_rows",
        "num_timesteps",
        "population_manifest",
        "schema_version",
    }:
        raise ValueError("selected-row stats top-level structure differs")
    for key, dim in (
        (LIBERO_ACTION_MODE, LIBERO_ACTION_DIM),
        (LIBERO_STATE_MODE, LIBERO_STATE_DIM),
    ):
        stats = payload[key]
        if set(stats) != set(_STAT_FIELDS):
            raise ValueError(f"LIBERO selected stats fields differ for {key}")
        for field in _STAT_FIELDS:
            value = stats[field]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != (dim,)
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"LIBERO selected stats {key}.{field} must be finite FP32 "
                    f"shape ({dim},)"
                )
    manifest = payload.get("population_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("selected-row stats have no population_manifest")
    manifest = dict(manifest)
    if set(manifest) != {
        "counts",
        "population_manifest_sha256",
        "schema_version",
        "selection_contract",
        "selection_contract_sha256",
        "sources",
    }:
        raise ValueError("LIBERO population manifest structure differs")
    if manifest.get("schema_version") != LIBERO_POPULATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("LIBERO population manifest schema differs")
    observed_digest = manifest.get("population_manifest_sha256")
    if (
        not _is_sha256(observed_digest)
        or population_manifest_sha256(manifest) != observed_digest
    ):
        raise ValueError("LIBERO population manifest SHA256 differs")
    contract = manifest.get("selection_contract")
    contract_roots = _validate_selection_contract(contract)
    if manifest.get("selection_contract_sha256") != canonical_json_sha256(contract):
        raise ValueError("LIBERO selection contract SHA256 differs")
    if expected_selection_contract is not None and dict(contract) != dict(
        expected_selection_contract
    ):
        raise ValueError("LIBERO stats selection contract differs from config")

    counts = manifest.get("counts")
    sources = manifest.get("sources")
    if not isinstance(counts, Mapping) or not isinstance(sources, list) or not sources:
        raise ValueError("LIBERO population manifest counts/sources are invalid")
    integer_counts = (
        "action_rows",
        "excluded_episodes",
        "selected_episodes",
        "source_episodes",
        "source_state_rows",
        "state_rows",
        "suites",
    )
    if set(counts) != set(integer_counts) or any(
        not _is_plain_int(counts.get(key)) for key in integer_counts
    ):
        raise ValueError("LIBERO population manifest has invalid counts")
    if counts["selected_episodes"] <= 0 or counts["suites"] != len(sources):
        raise ValueError("LIBERO population manifest selected counts differ")
    if counts["state_rows"] - counts["action_rows"] != counts["selected_episodes"]:
        raise ValueError("LIBERO population state/action count equation differs")
    if payload.get("num_timesteps") != counts["state_rows"]:
        raise ValueError("LIBERO selected state row count differs")
    if payload.get("num_action_rows") != counts["action_rows"]:
        raise ValueError("LIBERO selected action row count differs")
    if not _is_plain_int(payload.get("num_timesteps"), minimum=1) or not _is_plain_int(
        payload.get("num_action_rows"), minimum=1
    ):
        raise ValueError("LIBERO selected top-level row counts are invalid")

    if any(not isinstance(source, Mapping) for source in sources):
        raise ValueError("LIBERO population source structure differs")
    source_names = [source.get("dataset") for source in sources]
    if (
        any(not isinstance(name, str) or not name for name in source_names)
        or source_names != sorted(source_names)
        or len(source_names) != len(set(source_names))
    ):
        raise ValueError("LIBERO population sources are not uniquely sorted")
    contract_root_by_name = {
        row["dataset"]: row["resolved_root"] for row in contract_roots
    }
    if source_names != list(contract_root_by_name):
        raise ValueError("LIBERO population sources differ from selection roots")
    selected_keys = []
    summed_state_rows = 0
    summed_action_rows = 0
    summed_exclusions = 0
    summed_excluded_state_rows = 0
    observed_exclusion_keys = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {
            "counts",
            "dataset",
            "excluded_episodes",
            "metadata",
            "resolved_root",
            "selected_episodes",
        }:
            raise ValueError("LIBERO population source structure differs")
        if source.get("resolved_root") != contract_root_by_name[source["dataset"]]:
            raise ValueError("LIBERO population source root differs")
        selected = source.get("selected_episodes")
        excluded = source.get("excluded_episodes")
        metadata = source.get("metadata")
        source_counts = source.get("counts")
        if (
            not isinstance(selected, list)
            or not isinstance(excluded, list)
            or not isinstance(metadata, Mapping)
            or set(metadata) != set(_METADATA_FILES)
            or not isinstance(source_counts, Mapping)
            or set(source_counts)
            != {"action_rows", "excluded_episodes", "selected_episodes", "state_rows"}
            or any(not _is_plain_int(value) for value in source_counts.values())
        ):
            raise ValueError("LIBERO population source structure differs")
        for relative in _METADATA_FILES:
            _validate_file_pin(metadata[relative], expected_relative_path=relative)
        indices = [episode.get("episode_index") for episode in selected]
        if (
            any(not _is_plain_int(index) for index in indices)
            or indices != sorted(indices)
            or len(indices) != len(set(indices))
        ):
            raise ValueError("LIBERO selected episodes are not uniquely sorted")
        source_state_rows = 0
        source_action_rows = 0
        for episode in selected:
            if not isinstance(episode, Mapping) or set(episode) != {
                "action_rows",
                "episode_index",
                "length",
                "parquet",
                "state_rows",
                "task",
                "task_index",
            }:
                raise ValueError("LIBERO selected episode pin is invalid")
            length = episode.get("length")
            parquet = episode.get("parquet")
            if (
                not _is_plain_int(length, minimum=2)
                or not _is_plain_int(episode.get("task_index"))
                or not isinstance(episode.get("task"), str)
                or not episode["task"]
                or episode.get("state_rows") != length
                or episode.get("action_rows") != length - 1
            ):
                raise ValueError("LIBERO selected episode pin is invalid")
            _validate_file_pin(parquet)
            selected_keys.append((source["dataset"], episode["episode_index"]))
            summed_state_rows += length
            summed_action_rows += length - 1
            source_state_rows += length
            source_action_rows += length - 1
        excluded_indices = [episode.get("episode_index") for episode in excluded]
        if (
            any(not _is_plain_int(index) for index in excluded_indices)
            or excluded_indices != sorted(excluded_indices)
            or len(excluded_indices) != len(set(excluded_indices))
            or set(indices).intersection(excluded_indices)
        ):
            raise ValueError("LIBERO excluded episodes are not uniquely sorted")
        source_excluded_state_rows = 0
        for episode in excluded:
            if not isinstance(episode, Mapping) or set(episode) != {
                "episode_index",
                "length",
                "reason",
                "task",
                "task_index",
            }:
                raise ValueError("LIBERO excluded episode pin is invalid")
            length = episode.get("length")
            if (
                not _is_plain_int(length, minimum=2)
                or not _is_plain_int(episode.get("task_index"))
                or not isinstance(episode.get("task"), str)
                or not episode["task"]
                or episode.get("reason") != "configured complete-episode exclusion"
            ):
                raise ValueError("LIBERO excluded episode pin is invalid")
            source_excluded_state_rows += length
            observed_exclusion_keys.append(
                f"{source['dataset']}:{episode['episode_index']}"
            )
        expected_source_counts = {
            "action_rows": source_action_rows,
            "excluded_episodes": len(excluded),
            "selected_episodes": len(selected),
            "state_rows": source_state_rows,
        }
        if dict(source_counts) != expected_source_counts:
            raise ValueError("LIBERO population source counts differ")
        summed_exclusions += len(excluded)
        summed_excluded_state_rows += source_excluded_state_rows
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("LIBERO selected episode identities are duplicated")
    if sorted(observed_exclusion_keys) != contract["excluded_episodes"]:
        raise ValueError("LIBERO excluded episode pins differ from selection contract")
    if (
        summed_state_rows != counts["state_rows"]
        or summed_action_rows != counts["action_rows"]
        or len(selected_keys) != counts["selected_episodes"]
        or summed_exclusions != counts["excluded_episodes"]
    ):
        raise ValueError("LIBERO population per-source counts differ")
    if (
        counts["source_episodes"] < counts["selected_episodes"] + summed_exclusions
        or counts["source_state_rows"]
        < counts["state_rows"] + summed_excluded_state_rows
    ):
        raise ValueError("LIBERO population source totals differ")
    if (
        contract["split"] == "train"
        and contract["val_ratio"] == 0.0
        and (
            counts["source_episodes"] != counts["selected_episodes"] + summed_exclusions
            or counts["source_state_rows"]
            != counts["state_rows"] + summed_excluded_state_rows
        )
    ):
        raise ValueError("LIBERO full-train population totals differ")

    for key in (LIBERO_ACTION_MODE, LIBERO_STATE_MODE):
        stats = payload[key]
        minimum = np.asarray(stats["min"], dtype=np.float64)
        maximum = np.asarray(stats["max"], dtype=np.float64)
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        if not (
            np.all(minimum <= q01)
            and np.all(q01 <= q99)
            and np.all(q99 <= maximum)
            and np.all(minimum <= mean)
            and np.all(mean <= maximum)
            and np.all(std >= 1.0e-3)
        ):
            raise ValueError(f"LIBERO selected stats ordering differs for {key}")


def validate_libero_selected_stats_for_config(
    payload: Mapping[str, Any],
    config: Any,
    *,
    verify_live_sources: bool,
    verify_numeric_sources: bool = True,
) -> None:
    contract = selection_contract_from_config(config)
    validate_libero_selected_stats_payload(
        payload, expected_selection_contract=contract
    )
    if verify_live_sources:
        observed = payload["population_manifest"]
        if verify_numeric_sources:
            expected_payload = build_libero_selected_row_stats(config)
            expected_manifest = expected_payload["population_manifest"]
        else:
            expected_manifest, _episodes = build_population_manifest(config)
            expected_payload = None
        if observed != expected_manifest:
            raise ValueError(
                "LIBERO population manifest differs from live selected sources"
            )
        if not verify_numeric_sources:
            return
        assert expected_payload is not None
        if (
            payload["num_timesteps"] != expected_payload["num_timesteps"]
            or payload["num_action_rows"] != expected_payload["num_action_rows"]
        ):
            raise ValueError("LIBERO selected row counts differ from live sources")
        for key in (LIBERO_ACTION_MODE, LIBERO_STATE_MODE):
            for field in _STAT_FIELDS:
                if not np.array_equal(
                    payload[key][field], expected_payload[key][field]
                ):
                    raise ValueError(
                        "LIBERO selected numeric statistics differ from live "
                        f"sources: {key}.{field}"
                    )


def _atomic_write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    candidate = path.expanduser()
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite LIBERO selected-row stats: {candidate}"
        )
    parent = candidate.parent.resolve()
    if candidate.parent.is_symlink() or not parent.is_dir():
        raise ValueError(
            f"LIBERO selected-row stats parent must be a directory: {parent}"
        )
    resolved = parent / candidate.name
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{candidate.name}.tmp-",
            dir=parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            np.save(stream, dict(payload), allow_pickle=True)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.link(temporary_name, resolved)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def write_libero_selected_row_stats(
    path: str | Path,
    config: Any,
    *,
    expected_population_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    payload = build_libero_selected_row_stats(config)
    manifest = payload["population_manifest"]
    population_counts = dict(manifest["counts"])
    if expected_population_counts is not None and population_counts != dict(
        expected_population_counts
    ):
        raise ValueError(
            "LIBERO selected population counts differ before artifact publication"
        )
    output = Path(path)
    _atomic_write_payload(output, payload)
    return {
        "action_rows": payload["num_action_rows"],
        "artifact_sha256": sha256_file(output),
        "population_counts": population_counts,
        "population_manifest_sha256": manifest["population_manifest_sha256"],
        "selection_contract_sha256": manifest["selection_contract_sha256"],
        "selected_episodes": manifest["counts"]["selected_episodes"],
        "state_rows": payload["num_timesteps"],
    }


__all__ = [
    "LIBERO_SELECTION_CONTRACT_SCHEMA_VERSION",
    "build_libero_selected_row_stats",
    "build_population_manifest",
    "build_selection_contract",
    "canonical_json_sha256",
    "population_manifest_sha256",
    "selection_contract_from_config",
    "validate_libero_selected_stats_for_config",
    "validate_libero_selected_stats_payload",
    "validate_libero_window_coverage_for_config",
    "write_libero_selected_row_stats",
]
