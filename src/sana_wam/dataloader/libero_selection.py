"""Shared LIBERO episode selection and Parquet row-identity contract."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sana_wam.dataloader.libero_stats import (
    LIBERO_ACTION_DIM,
    LIBERO_STATE_DIM,
)


LIBERO_SPLIT_ALGORITHM = (
    "exclude-before-split+python-random-v1:"
    'random.Random("sana-wam-libero-split-v1:{seed}:{dataset}")'
)


@dataclass(frozen=True)
class LiberoEpisode:
    dataset: str
    root: Path
    episode_index: int
    length: int
    task: str
    task_index: int
    chunks_size: int
    data_path_template: str
    video_path_template: str

    @property
    def episode_chunk(self) -> int:
        return self.episode_index // self.chunks_size

    def data_path(self) -> Path:
        return self.root / self.data_path_template.format(
            episode_chunk=self.episode_chunk,
            episode_index=self.episode_index,
        )

    def video_path(self, video_key: str) -> Path:
        return self.root / self.video_path_template.format(
            episode_chunk=self.episode_chunk,
            episode_index=self.episode_index,
            video_key=video_key,
        )

    @property
    def identity(self) -> str:
        return f"{self.dataset}:{self.episode_index}"


@dataclass(frozen=True)
class LiberoSelection:
    roots: tuple[Path, ...]
    eligible: tuple[LiberoEpisode, ...]
    selected: tuple[LiberoEpisode, ...]
    excluded: tuple[LiberoEpisode, ...]
    source_episode_count: int
    source_state_row_count: int


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read LIBERO JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"LIBERO JSON must contain an object: {path}")
    return value


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read LIBERO JSONL {path}: {exc}") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid LIBERO JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"LIBERO JSONL row must be an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def canonical_libero_roots(
    dataset_roots: Sequence[str | Path],
) -> tuple[Path, ...]:
    candidates = tuple(Path(root).expanduser() for root in dataset_roots)
    if not candidates:
        raise ValueError("LIBERO dataset_roots must not be empty")
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"LIBERO dataset root must not be a symlink: {candidate}")
    roots = tuple(
        sorted((candidate.resolve() for candidate in candidates), key=lambda p: p.name)
    )
    if len({root.name for root in roots}) != len(roots):
        raise ValueError("LIBERO dataset root basenames must be unique")
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"invalid LIBERO dataset root: {root}")
    return roots


def _validate_exclusions(excluded_episodes: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in excluded_episodes)
    if len(values) != len(set(values)):
        raise ValueError("LIBERO excluded_episodes contains duplicates")
    for value in values:
        try:
            dataset, raw_index = value.rsplit(":", 1)
            index = int(raw_index)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid LIBERO excluded episode key: {value!r}") from exc
        if not dataset or index < 0 or raw_index != str(index):
            raise ValueError(f"invalid LIBERO excluded episode key: {value!r}")
    return values


def _root_episodes(root: Path) -> tuple[list[LiberoEpisode], int]:
    info = load_json(root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{root.name} is not a LeRobot v2.1 dataset")
    if info.get("fps") != 20:
        raise ValueError(f"{root.name} fps must be exactly 20")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError(f"{root.name} has no features mapping")
    expected_features = {
        "observation.state": ("float32", [LIBERO_STATE_DIM]),
        "action": ("float32", [LIBERO_ACTION_DIM]),
        "observation.images.image": ("video", [256, 256, 3]),
        "observation.images.wrist_image": ("video", [256, 256, 3]),
    }
    for key, (dtype, shape) in expected_features.items():
        feature = features.get(key)
        if not isinstance(feature, Mapping):
            raise ValueError(f"{root.name} is missing feature {key!r}")
        if feature.get("dtype") != dtype or feature.get("shape") != shape:
            raise ValueError(f"{root.name} feature {key!r} has incompatible schema")

    chunks_size = info.get("chunks_size")
    if (
        isinstance(chunks_size, bool)
        or not isinstance(chunks_size, int)
        or chunks_size <= 0
    ):
        raise ValueError(f"{root.name} has invalid chunks_size")
    data_template = info.get("data_path")
    video_template = info.get("video_path")
    if not isinstance(data_template, str) or not isinstance(video_template, str):
        raise ValueError(f"{root.name} has invalid data/video path templates")

    task_to_index: dict[str, int] = {}
    observed_task_indices: set[int] = set()
    for row in read_json_lines(root / "meta/tasks.jsonl"):
        task = row.get("task")
        task_index = row.get("task_index")
        if (
            not isinstance(task, str)
            or not task
            or isinstance(task_index, bool)
            or not isinstance(task_index, int)
            or task_index < 0
            or task in task_to_index
            or task_index in observed_task_indices
        ):
            raise ValueError(f"{root.name} contains an invalid or duplicate task row")
        task_to_index[task] = task_index
        observed_task_indices.add(task_index)

    episode_rows = read_json_lines(root / "meta/episodes.jsonl")
    total_episodes = info.get("total_episodes")
    if (
        isinstance(total_episodes, bool)
        or not isinstance(total_episodes, int)
        or len(episode_rows) != total_episodes
    ):
        raise ValueError(f"{root.name} episode count differs from info.json")
    result = []
    observed_episode_indices: set[int] = set()
    for row in episode_rows:
        episode_index = row.get("episode_index")
        length = row.get("length")
        tasks = row.get("tasks")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
            or episode_index in observed_episode_indices
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length < 2
            or not isinstance(tasks, list)
            or len(tasks) != 1
            or not isinstance(tasks[0], str)
            or tasks[0] not in task_to_index
        ):
            raise ValueError(f"{root.name} contains an invalid episode row")
        observed_episode_indices.add(episode_index)
        result.append(
            LiberoEpisode(
                dataset=root.name,
                root=root,
                episode_index=episode_index,
                length=length,
                task=tasks[0],
                task_index=task_to_index[tasks[0]],
                chunks_size=chunks_size,
                data_path_template=data_template,
                video_path_template=video_template,
            )
        )
    total_frames = info.get("total_frames")
    observed_frames = sum(episode.length for episode in result)
    if (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames != observed_frames
    ):
        raise ValueError(f"{root.name} total_frames differs from episode metadata")
    return result, observed_frames


def select_libero_episodes(
    dataset_roots: Sequence[str | Path],
    *,
    split: str,
    val_ratio: float,
    seed: int,
    excluded_episodes: Sequence[str],
    require_all_exclusions: bool,
) -> LiberoSelection:
    """Select episodes with the exact dataset exclude-before-split semantics."""

    if split not in {"train", "val"}:
        raise ValueError("LIBERO split must be 'train' or 'val'")
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("LIBERO val_ratio must be in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("LIBERO seed must be an integer")
    roots = canonical_libero_roots(dataset_roots)
    exclusions = _validate_exclusions(excluded_episodes)
    exclusion_set = set(exclusions)
    selected: list[LiberoEpisode] = []
    eligible: list[LiberoEpisode] = []
    excluded: list[LiberoEpisode] = []
    source_episode_count = 0
    source_state_row_count = 0

    for root in roots:
        source_episodes, source_rows = _root_episodes(root)
        source_episode_count += len(source_episodes)
        source_state_row_count += source_rows
        available = []
        for episode in source_episodes:
            if episode.identity in exclusion_set:
                excluded.append(episode)
            else:
                available.append(episode)
        eligible.extend(available)
        rng = random.Random(f"sana-wam-libero-split-v1:{seed}:{root.name}")
        indices = list(range(len(available)))
        rng.shuffle(indices)
        if val_ratio <= 0.0:
            n_val = 0
        elif val_ratio >= 1.0:
            n_val = len(indices)
        else:
            n_val = max(1, int(len(indices) * val_ratio))
        chosen = indices[:n_val] if split == "val" else indices[n_val:]
        selected.extend(available[index] for index in sorted(chosen))

    observed_exclusions = {episode.identity for episode in excluded}
    root_names = {root.name for root in roots}
    applicable = {
        value for value in exclusions if value.rsplit(":", 1)[0] in root_names
    }
    missing = (set(exclusions) if require_all_exclusions else applicable) - (
        observed_exclusions
    )
    if missing:
        raise ValueError(
            f"LIBERO excluded episode keys were not found: {sorted(missing)}"
        )
    if not selected:
        raise ValueError(f"no LIBERO episodes selected for split={split!r}")
    # Preserve the original loader's fail-closed behavior: every non-excluded
    # episode in every source suite must have a usable Parquet file, even when
    # a non-zero validation split selects only part of that population.
    for episode in eligible:
        path = episode.data_path()
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"LIBERO episode parquet must be a regular non-symlink file: {path}"
            )
    return LiberoSelection(
        roots=roots,
        eligible=tuple(eligible),
        selected=tuple(selected),
        excluded=tuple(excluded),
        source_episode_count=source_episode_count,
        source_state_row_count=source_state_row_count,
    )


def read_libero_episode_arrays(
    episode: LiberoEpisode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read and validate the six Parquet identity columns used by training."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO Parquet loading requires PyArrow; install the pinned "
            "benchmarks/libero/requirements-data.txt in the sana-wam environment "
            "before use"
        ) from exc
    table = pq.read_table(
        episode.data_path(),
        columns=[
            "observation.state",
            "action",
            "task_index",
            "episode_index",
            "frame_index",
            "timestamp",
        ],
        memory_map=True,
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
    episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    if states.shape != (episode.length, LIBERO_STATE_DIM):
        raise ValueError(
            f"LIBERO state array shape differs for {episode.data_path()}: {states.shape}"
        )
    if actions.shape != (episode.length, LIBERO_ACTION_DIM):
        raise ValueError(
            f"LIBERO action array shape differs for {episode.data_path()}: {actions.shape}"
        )
    if task_indices.shape != (episode.length,) or not np.all(
        task_indices == episode.task_index
    ):
        raise ValueError(f"LIBERO task_index differs within {episode.data_path()}")
    if episode_indices.shape != (episode.length,) or not np.all(
        episode_indices == episode.episode_index
    ):
        raise ValueError(f"LIBERO episode_index differs within {episode.data_path()}")
    expected_frames = np.arange(episode.length, dtype=np.int64)
    if frame_indices.shape != (episode.length,) or not np.array_equal(
        frame_indices, expected_frames
    ):
        raise ValueError(
            f"LIBERO frame_index is not contiguous in {episode.data_path()}"
        )
    expected_timestamps = expected_frames.astype(np.float64) / 20.0
    if timestamps.shape != (episode.length,) or not np.allclose(
        timestamps, expected_timestamps, rtol=0.0, atol=1e-5
    ):
        raise ValueError(
            f"LIBERO timestamp/frame alignment differs in {episode.data_path()}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(
            f"LIBERO episode contains non-finite state/action: {episode.data_path()}"
        )
    return states, actions, task_indices


__all__ = [
    "LIBERO_SPLIT_ALGORITHM",
    "LiberoEpisode",
    "LiberoSelection",
    "canonical_libero_roots",
    "load_json",
    "read_json_lines",
    "read_libero_episode_arrays",
    "select_libero_episodes",
]
