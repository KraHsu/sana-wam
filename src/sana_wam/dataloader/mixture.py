"""Deterministic weighted mixtures for training datasets."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sana_wam.dataloader.base_dataset import BaseActionDataset
from sana_wam.dataloader.robotwin_history_dagger_dataset import (
    RoboTwinHistoryDaggerDataset,
)
from sana_wam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset
from sana_wam.dataloader.transforms.multiview import DEFAULT_MULTIVIEW_CAMERA_LAYOUT


_DEPLOY_DEFAULTS = {
    "action_mode": "eef",
    "normalize_mode": "min-max",
    "num_frames": 33,
    "video_stride": 4,
    "temporal_compression": 4,
    "causal_temporal": True,
    "height": 480,
    "width": 832,
    "multiview": True,
    "camera_layout": tuple(DEFAULT_MULTIVIEW_CAMERA_LAYOUT),
    "target_camera": "head_camera",
    "delta_action": False,
}
_HISTORY_DAGGER_DEPLOY_DEFAULTS = {
    **_DEPLOY_DEFAULTS,
    "num_frames": 113,
    "height": 384,
    "width": 320,
}
_DEPLOY_FIELDS = tuple(_DEPLOY_DEFAULTS)
_POSITIVE_INT_DEPLOY_FIELDS = {
    "num_frames",
    "video_stride",
    "temporal_compression",
    "height",
    "width",
}
_BOOL_DEPLOY_FIELDS = {"causal_temporal", "multiview", "delta_action"}


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    return default if value is None else value


def _sha256_file(path: str | Path) -> tuple[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"mixture source action_stats_path is not a file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return str(resolved), digest.hexdigest()


def _stats_are_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(_stats_are_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return bool(value) and all(_stats_are_finite(item) for item in value)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return (
        array.size > 0
        and np.issubdtype(array.dtype, np.number)
        and bool(np.isfinite(array).all())
    )


def _stats_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            _stats_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        if not isinstance(right, Sequence) or isinstance(
            right, (str, bytes, bytearray)
        ):
            return False
        return len(left) == len(right) and all(
            _stats_equal(a, b) for a, b in zip(left, right)
        )
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
    except (TypeError, ValueError):
        return False
    return left_array.shape == right_array.shape and bool(
        np.array_equal(left_array, right_array)
    )


def _integer_quotas(weights: Sequence[float], epoch_size: int) -> np.ndarray:
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    expected = normalized * epoch_size
    quotas = np.floor(expected).astype(np.int64)
    remainder = epoch_size - int(quotas.sum())
    if remainder:
        fractions = expected - quotas
        # Stable sorting makes source order the deterministic tie breaker.
        order = np.argsort(-fractions, kind="stable")
        quotas[order[:remainder]] += 1
    return quotas


def _draw_child_indices(
    length: int, quota: int, rng: np.random.Generator
) -> np.ndarray:
    cycles, residual = divmod(quota, length)
    chunks = [rng.permutation(length) for _ in range(cycles)]
    if residual:
        chunks.append(rng.permutation(length)[:residual])
    return np.concatenate(chunks).astype(np.int64, copy=False)


def _leaf_datasets(dataset: Any) -> list[Any]:
    children = getattr(dataset, "_sub_datasets", None)
    return list(children) if children else [dataset]


def _canonical_deploy_value(field: str, value: Any) -> Any:
    if field in _POSITIVE_INT_DEPLOY_FIELDS:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise ValueError(
                f"mixture deploy field {field} must be a positive integer, got {value!r}"
            )
        return int(value)
    if field in _BOOL_DEPLOY_FIELDS:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"mixture deploy field {field} must be boolean, got {value!r}"
            )
        return bool(value)
    if field == "camera_layout":
        if isinstance(value, (str, bytes)):
            raise ValueError(
                "mixture deploy field camera_layout must be a camera sequence, "
                f"got {value!r}"
            )
        try:
            layout = tuple(str(camera) for camera in value)
        except TypeError as exc:
            raise ValueError(
                f"mixture deploy field camera_layout must be iterable, got {value!r}"
            ) from exc
        if len(layout) != 3:
            raise ValueError(
                "mixture deploy field camera_layout must contain three cameras, "
                f"got {layout!r}"
            )
        return layout
    if field == "normalize_mode":
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        if value is not None and not isinstance(value, str):
            raise ValueError(
                "mixture deploy field normalize_mode must be a string or null, "
                f"got {value!r}"
            )
        return value
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"mixture deploy field {field} must be a non-empty string, got {value!r}"
        )
    return value


def _config_deploy_contract(config: Any) -> dict[str, Any]:
    defaults = (
        _HISTORY_DAGGER_DEPLOY_DEFAULTS
        if str(_get(config, "type", "robotwin")) == "robotwin_history_dagger"
        else _DEPLOY_DEFAULTS
    )
    return {
        field: _canonical_deploy_value(field, _get(config, field, default))
        for field, default in defaults.items()
    }


def _dataset_deploy_contract(
    dataset: Any, fallback: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = {}
    leaves = _leaf_datasets(dataset)
    for field in _DEPLOY_FIELDS:
        values = []
        for leaf in leaves:
            value = getattr(leaf, field, None)
            if value is not None:
                values.append(_canonical_deploy_value(field, value))
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"mixture source has inconsistent leaf deploy field {field}: {values!r}"
            )
        resolved[field] = values[0] if values else fallback[field]
    return resolved


def _common_deploy_contract(
    contracts: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Any]:
    if not contracts:
        raise ValueError("mixture requires at least one enabled source")
    first = dict(contracts[0])
    for field in _DEPLOY_FIELDS:
        values = [contract[field] for contract in contracts]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"mixture {label} deploy field {field} mismatch across sources: {values!r}"
            )
    return first


def _has_explicit_value(config: Any, key: str) -> bool:
    if isinstance(config, Mapping):
        return key in config and config[key] is not None
    try:
        return getattr(config, key) is not None
    except (AttributeError, KeyError):
        return False


def _serializable_deploy_value(field: str, value: Any) -> Any:
    return list(value) if field == "camera_layout" else value


def _set_config_value(config: Any, key: str, value: Any) -> None:
    value = copy.deepcopy(value)
    if isinstance(config, dict):
        config[key] = value
        return
    try:
        from omegaconf import DictConfig, open_dict

        if isinstance(config, DictConfig):
            with open_dict(config):
                config[key] = value
            return
    except ImportError:  # pragma: no cover - OmegaConf is a core dependency
        pass
    try:
        setattr(config, key, value)
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            f"mixture config must explicitly provide deploy field {key}; "
            "the builder could not write it back"
        ) from exc


def _validate_or_fill_top_level_contract(
    config: Any, resolved: Mapping[str, Any]
) -> None:
    for field in _DEPLOY_FIELDS:
        if _has_explicit_value(config, field):
            top_value = _canonical_deploy_value(field, _get(config, field))
            if top_value != resolved[field]:
                raise ValueError(
                    f"mixture top-level deploy field {field} conflicts with sources: "
                    f"{top_value!r} != {resolved[field]!r}"
                )
            continue
        _set_config_value(
            config, field, _serializable_deploy_value(field, resolved[field])
        )


class MixtureDataset(BaseActionDataset):
    """A virtual epoch whose exact source quotas follow configured weights."""

    def __init__(
        self,
        datasets: Sequence[BaseActionDataset],
        weights: Sequence[float],
        *,
        seed: int = 42,
        epoch_size: int | None = None,
        deploy_contract: Mapping[str, Any] | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("mixture requires at least one enabled source")
        if len(datasets) != len(weights):
            raise ValueError("mixture datasets and weights must have the same length")

        self._datasets = list(datasets)
        self._weights = tuple(float(weight) for weight in weights)
        if any(not math.isfinite(weight) or weight <= 0 for weight in self._weights):
            raise ValueError(
                "every enabled mixture source must have a finite weight > 0"
            )
        if any(len(dataset) <= 0 for dataset in self._datasets):
            raise ValueError("every enabled mixture source must be non-empty")

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"mixture seed must be an integer, got {seed!r}")
        self.seed = seed

        if epoch_size is None:
            epoch_size = sum(len(dataset) for dataset in self._datasets)
        if (
            isinstance(epoch_size, bool)
            or not isinstance(epoch_size, int)
            or epoch_size <= 0
        ):
            raise ValueError(
                f"mixture epoch_size must be a positive integer, got {epoch_size!r}"
            )
        self.epoch_size = epoch_size

        self._validate_contract()
        if deploy_contract is None:
            inferred = [
                _dataset_deploy_contract(dataset, _DEPLOY_DEFAULTS)
                for dataset in self._datasets
            ]
            deploy_contract = _common_deploy_contract(inferred, label="runtime")
        self.deploy_contract = {
            field: _canonical_deploy_value(field, deploy_contract[field])
            for field in _DEPLOY_FIELDS
        }
        for field, value in self.deploy_contract.items():
            setattr(
                self,
                field,
                copy.deepcopy(_serializable_deploy_value(field, value)),
            )
        self._source_quotas = _integer_quotas(self._weights, self.epoch_size)
        self._sub_datasets = [
            leaf for dataset in self._datasets for leaf in _leaf_datasets(dataset)
        ]
        self.epoch = 0
        self._index_map = np.empty((0, 2), dtype=np.int64)
        self.set_epoch(0)

    def _validate_contract(self) -> None:
        dimensions = [
            getattr(dataset, "action_dim", None) for dataset in self._datasets
        ]
        if any(
            isinstance(dim, bool) or not isinstance(dim, (int, np.integer))
            for dim in dimensions
        ):
            raise ValueError(
                f"every mixture source must expose an integer action_dim, got {dimensions!r}"
            )
        if len(set(int(dim) for dim in dimensions)) != 1:
            raise ValueError(
                f"mixture action_dim mismatch across sources: {dimensions!r}"
            )
        self._action_dim = int(dimensions[0])

        stats_paths: list[str] = []
        stats_hashes: list[str] = []
        stats_values: list[Any] = []
        for index, dataset in enumerate(self._datasets):
            raw_path = getattr(dataset, "action_stats_path", None)
            if not raw_path:
                raise ValueError(f"mixture source {index} has no action_stats_path")
            path, digest = _sha256_file(raw_path)
            stats = getattr(dataset, "action_stats", None)
            stats = stats() if callable(stats) else stats
            if not _stats_are_finite(stats):
                raise ValueError(
                    f"mixture source {index} has missing, empty, or non-finite action_stats"
                )
            stats_paths.append(path)
            stats_hashes.append(digest)
            stats_values.append(stats)

        if len(set(stats_hashes)) != 1:
            raise ValueError(
                f"mixture action_stats_path content SHA256 mismatch: {stats_hashes!r}"
            )
        if any(not _stats_equal(stats_values[0], stats) for stats in stats_values[1:]):
            raise ValueError(
                "mixture action_stats numeric values differ across sources"
            )

        self.action_stats_path = stats_paths[0]
        self._action_stats = copy.deepcopy(stats_values[0])

    def set_epoch(self, epoch: int) -> None:
        """Rebuild the virtual index deterministically for one epoch."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(
                f"mixture epoch must be a non-negative integer, got {epoch!r}"
            )
        self.epoch = epoch
        for dataset in self._datasets:
            child_set_epoch = getattr(dataset, "set_epoch", None)
            if callable(child_set_epoch):
                child_set_epoch(epoch)

        parts = []
        for source_index, (dataset, quota) in enumerate(
            zip(self._datasets, self._source_quotas)
        ):
            quota = int(quota)
            if quota == 0:
                continue
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, epoch, source_index])
            )
            child_indices = _draw_child_indices(len(dataset), quota, rng)
            source_indices = np.full(quota, source_index, dtype=np.int64)
            parts.append(np.column_stack((source_indices, child_indices)))

        index_map = np.concatenate(parts, axis=0)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, len(self._datasets)])
        )
        self._index_map = index_map[rng.permutation(len(index_map))]

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_stats(self) -> dict:
        return copy.deepcopy(self._action_stats)

    @property
    def source_counts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._source_quotas)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index, child_index = self._index_map[index]
        return self._datasets[int(source_index)][int(child_index)]


def build_training_dataset(config: Any, split: str = "train") -> BaseActionDataset:
    """Build a native dataset or a homogeneous weighted mixture from config."""
    dataset_type = str(_get(config, "type", "robotwin"))
    if dataset_type == "robotwin":
        return MultiTaskRoboTwinDataset.from_config(config, split=split)
    if dataset_type == "robotwin_history_dagger":
        return RoboTwinHistoryDaggerDataset.from_config(config, split=split)
    if dataset_type == "libero":
        # Lazy import keeps the regular RoboTwin path independent of the
        # optional Parquet dependency used only when samples are decoded.
        from sana_wam.dataloader.libero_dataset import LiberoLeRobotDataset

        return LiberoLeRobotDataset.from_config(config, split=split)
    if dataset_type != "mixture":
        raise ValueError(f"unsupported dataloader type: {dataset_type!r}")

    sources = [
        source
        for source in _get(config, "datasets", ())
        if bool(_get(source, "enabled", True))
    ]
    configured_contracts = [_config_deploy_contract(source) for source in sources]
    _common_deploy_contract(configured_contracts, label="configured")

    datasets = [build_training_dataset(source, split=split) for source in sources]
    weights = [float(_get(source, "weight", 1.0)) for source in sources]
    runtime_contracts = [
        _dataset_deploy_contract(dataset, configured)
        for dataset, configured in zip(datasets, configured_contracts)
    ]
    resolved_contract = _common_deploy_contract(runtime_contracts, label="runtime")
    _validate_or_fill_top_level_contract(config, resolved_contract)

    epoch_size = _get(config, "epoch_size", None)
    return MixtureDataset(
        datasets,
        weights,
        seed=int(_get(config, "seed", 42)),
        epoch_size=None if epoch_size is None else int(epoch_size),
        deploy_contract=resolved_contract,
    )


__all__ = ["MixtureDataset", "build_training_dataset"]
