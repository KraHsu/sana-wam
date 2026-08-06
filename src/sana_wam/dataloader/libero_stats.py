"""LIBERO LeRobot normalization statistics built from pinned metadata.

The H200 LIBERO assets already carry one ``meta/stats_gr00t.json`` per suite.
This module converts those JSON statistics into sana-wam's existing
``action_stats.npy`` checkpoint contract without reading videos or importing
LIBERO / LeRobot.  Multiple suites are pooled exactly for mean/std/min/max.
The pooled q01/q99 values are conservative suite envelopes; production configs
currently use min-max normalization, so those percentile fields are retained
only for schema completeness.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


LIBERO_ACTION_MODE = "libero_relative_eef"
LIBERO_STATE_MODE = "libero_eef_axis_angle_gripper"
LIBERO_ACTION_DIM = 7
LIBERO_STATE_DIM = 8
LIBERO_STATS_SCHEMA_VERSION = "sana-wam-libero-stats-v1"
LIBERO_SELECTED_STATS_SCHEMA_VERSION = "sana-wam-libero-selected-row-stats-v2"
LIBERO_POPULATION_MANIFEST_SCHEMA_VERSION = (
    "sana-wam-libero-selected-row-population-manifest-v2"
)

_SOURCE_METADATA_FILES = (
    "meta/info.json",
    "meta/stats_gr00t.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
)
_STAT_FIELDS = ("mean", "std", "min", "max", "q01", "q99")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one regular file."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"LIBERO source must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(
            f"LIBERO source must be a regular non-symlink file: {resolved}"
        )
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read LIBERO metadata JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"LIBERO metadata JSON must contain an object: {path}")
    return value


def _canonical_roots(dataset_roots: Iterable[str | Path]) -> tuple[Path, ...]:
    candidates = tuple(Path(root).expanduser() for root in dataset_roots)
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"LIBERO dataset root must not be a symlink: {candidate}")
    roots = tuple(candidate.resolve() for candidate in candidates)
    if not roots:
        raise ValueError("LIBERO stats require at least one dataset root")
    if len({root.name for root in roots}) != len(roots):
        raise ValueError("LIBERO dataset root basenames must be unique")
    for root in roots:
        if not root.is_dir():
            raise ValueError(
                f"LIBERO dataset root must be a non-symlink directory: {root}"
            )
    return tuple(sorted(roots, key=lambda root: root.name))


def build_source_manifest(dataset_roots: Iterable[str | Path]) -> dict[str, Any]:
    """Describe the exact metadata inputs used to create a stats artifact."""

    sources = []
    for root in _canonical_roots(dataset_roots):
        files = {}
        for relative in _SOURCE_METADATA_FILES:
            path = root / relative
            files[relative] = {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        sources.append(
            {
                "dataset": root.name,
                "resolved_root": str(root),
                "metadata": files,
            }
        )
    return {
        "schema_version": "sana-wam-libero-stats-source-manifest-v1",
        "sources": sources,
    }


def _stat_vector(
    statistics: Mapping[str, Any], field: str, *, dim: int, label: str
) -> np.ndarray:
    try:
        value = np.asarray(statistics[field], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{field} is missing or non-numeric") from exc
    if value.shape != (dim,) or not np.isfinite(value).all():
        raise ValueError(
            f"{label}.{field} must be finite shape ({dim},), got {value.shape}"
        )
    return value


def _load_suite_stats(root: Path) -> tuple[int, dict[str, dict[str, np.ndarray]]]:
    info = _load_json(root / "meta/info.json")
    frame_count = info.get("total_frames")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise ValueError(f"{root.name} meta/info.json has invalid total_frames")

    payload = _load_json(root / "meta/stats_gr00t.json")
    raw_statistics = payload.get("statistics")
    if not isinstance(raw_statistics, Mapping):
        raise ValueError(f"{root.name} stats_gr00t.json has no statistics object")

    result: dict[str, dict[str, np.ndarray]] = {}
    for source_key, target_key, dim in (
        ("action", LIBERO_ACTION_MODE, LIBERO_ACTION_DIM),
        ("observation.state", LIBERO_STATE_MODE, LIBERO_STATE_DIM),
    ):
        raw = raw_statistics.get(source_key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{root.name} statistics has no {source_key!r}")
        result[target_key] = {
            field: _stat_vector(
                raw,
                field,
                dim=dim,
                label=f"{root.name}:{source_key}",
            )
            for field in _STAT_FIELDS
        }
    return frame_count, result


def _pool_mode(
    weighted: Sequence[tuple[int, Mapping[str, np.ndarray]]], *, dim: int
) -> dict[str, np.ndarray]:
    total = sum(count for count, _stats in weighted)
    if total <= 0:
        raise ValueError("cannot pool empty LIBERO statistics")

    mean = sum(count * stats["mean"] for count, stats in weighted) / total
    second_moment = (
        sum(
            count * (np.square(stats["std"]) + np.square(stats["mean"]))
            for count, stats in weighted
        )
        / total
    )
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    pooled = {
        "mean": mean,
        "std": np.maximum(np.sqrt(variance), 1e-3),
        "min": np.minimum.reduce([stats["min"] for _count, stats in weighted]),
        "max": np.maximum.reduce([stats["max"] for _count, stats in weighted]),
        "q01": np.minimum.reduce([stats["q01"] for _count, stats in weighted]),
        "q99": np.maximum.reduce([stats["q99"] for _count, stats in weighted]),
    }
    for field, vector in pooled.items():
        if vector.shape != (dim,) or not np.isfinite(vector).all():
            raise ValueError(f"pooled LIBERO {field} is invalid")
    return {field: vector.astype(np.float32) for field, vector in pooled.items()}


def build_libero_stats_from_metadata(
    dataset_roots: Iterable[str | Path],
) -> dict[str, Any]:
    """Create the nested stats payload consumed by training and deployment."""

    roots = _canonical_roots(dataset_roots)
    suites = [(root, *_load_suite_stats(root)) for root in roots]
    total_frames = sum(frame_count for _root, frame_count, _stats in suites)
    if not math.isfinite(float(total_frames)) or total_frames <= 0:
        raise ValueError("LIBERO total frame count is invalid")
    return {
        LIBERO_ACTION_MODE: _pool_mode(
            [
                (frame_count, stats[LIBERO_ACTION_MODE])
                for _root, frame_count, stats in suites
            ],
            dim=LIBERO_ACTION_DIM,
        ),
        LIBERO_STATE_MODE: _pool_mode(
            [
                (frame_count, stats[LIBERO_STATE_MODE])
                for _root, frame_count, stats in suites
            ],
            dim=LIBERO_STATE_DIM,
        ),
        "num_timesteps": total_frames,
        "schema_version": LIBERO_STATS_SCHEMA_VERSION,
        "source_manifest": build_source_manifest(roots),
    }


def validate_libero_stats_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed on malformed or dimensionally incompatible stats."""

    schema_version = payload.get("schema_version")
    if schema_version not in {
        LIBERO_STATS_SCHEMA_VERSION,
        LIBERO_SELECTED_STATS_SCHEMA_VERSION,
    }:
        raise ValueError("LIBERO stats schema_version differs")
    for key, dim in (
        (LIBERO_ACTION_MODE, LIBERO_ACTION_DIM),
        (LIBERO_STATE_MODE, LIBERO_STATE_DIM),
    ):
        mode_stats = payload.get(key)
        if not isinstance(mode_stats, Mapping):
            raise ValueError(f"LIBERO stats payload has no {key!r} mapping")
        for field in _STAT_FIELDS:
            _stat_vector(mode_stats, field, dim=dim, label=key)
    manifest_key = (
        "source_manifest"
        if schema_version == LIBERO_STATS_SCHEMA_VERSION
        else "population_manifest"
    )
    if not isinstance(payload.get(manifest_key), Mapping):
        raise ValueError(f"LIBERO stats payload has no {manifest_key}")


def load_libero_stats(path: str | Path) -> dict[str, Any]:
    """Load and validate one sana-wam LIBERO ``action_stats.npy`` file."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"LIBERO stats path must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"LIBERO stats path is not a regular file: {resolved}")
    try:
        payload = np.load(resolved, allow_pickle=True).item()
    except Exception as exc:
        raise ValueError(f"cannot load LIBERO stats file {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("LIBERO stats file must contain a mapping")
    payload = dict(payload)
    validate_libero_stats_payload(payload)
    return payload


def write_libero_stats(path: str | Path, dataset_roots: Iterable[str | Path]) -> str:
    """Atomically materialize stats and return the resulting SHA-256."""

    candidate = Path(path).expanduser()
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite LIBERO stats artifact: {candidate}"
        )
    parent = candidate.parent.resolve()
    if not parent.is_dir() or candidate.parent.is_symlink():
        raise ValueError(
            f"LIBERO stats parent must be a non-symlink directory: {parent}"
        )
    resolved = parent / candidate.name
    payload = build_libero_stats_from_metadata(dataset_roots)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{candidate.name}.tmp-",
            dir=parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            np.save(stream, payload, allow_pickle=True)
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
    return sha256_file(resolved)


__all__ = [
    "LIBERO_ACTION_DIM",
    "LIBERO_ACTION_MODE",
    "LIBERO_STATE_DIM",
    "LIBERO_STATE_MODE",
    "LIBERO_SELECTED_STATS_SCHEMA_VERSION",
    "LIBERO_POPULATION_MANIFEST_SCHEMA_VERSION",
    "LIBERO_STATS_SCHEMA_VERSION",
    "build_libero_stats_from_metadata",
    "build_source_manifest",
    "load_libero_stats",
    "sha256_file",
    "validate_libero_stats_payload",
    "write_libero_stats",
]
