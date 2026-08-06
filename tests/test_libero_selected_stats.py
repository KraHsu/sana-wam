from __future__ import annotations

import copy
import json
import stat
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from sana_wam.dataloader.libero_dataset import LIBERO_TRAINING_DATASET_NAMES
from sana_wam.dataloader.libero_selected_stats import (
    build_libero_selected_row_stats,
    population_manifest_sha256,
    validate_libero_selected_stats_for_config,
    validate_libero_selected_stats_payload,
    validate_libero_window_coverage_for_config,
    write_libero_selected_row_stats,
)
from sana_wam.dataloader.libero_stats import (
    LIBERO_ACTION_DIM,
    LIBERO_ACTION_MODE,
    LIBERO_SELECTED_STATS_SCHEMA_VERSION,
    LIBERO_STATE_DIM,
    LIBERO_STATE_MODE,
    sha256_file,
)
from sana_wam.dataloader.libero_selection import select_libero_episodes
from sana_wam.train.libero_contract import validate_libero_training_config


GOAL_NAME = "libero_goal_no_noops_1.0.0_lerobot"
GOAL_EXCLUSION = f"{GOAL_NAME}:82"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _stats(dim: int) -> dict[str, list[float]]:
    return {
        "max": [1.0] * dim,
        "mean": [0.0] * dim,
        "min": [-1.0] * dim,
        "q01": [-0.9] * dim,
        "q99": [0.9] * dim,
        "std": [1.0] * dim,
    }


class _Column:
    def __init__(self, values):
        self._values = values

    def to_pylist(self):
        return self._values


class _Table:
    def __init__(self, values: dict[str, list], columns: list[str]):
        self._values = values
        self._columns = set(columns)

    def __getitem__(self, key: str) -> _Column:
        if key not in self._columns:
            raise KeyError(key)
        return _Column(self._values[key])


def _install_fake_pyarrow(
    monkeypatch: pytest.MonkeyPatch, registry: dict[str, dict[str, list]]
) -> None:
    parquet = types.ModuleType("pyarrow.parquet")

    def read_table(path, *, columns, memory_map):
        assert memory_map is True
        values = registry[str(Path(path).resolve())]
        return _Table(values, list(columns))

    parquet.read_table = read_table
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    pyarrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)


def _episode_values(
    *,
    episode_index: int,
    length: int,
    state_values: list[float],
    action_values: list[float],
) -> dict[str, list]:
    assert len(state_values) == length
    assert len(action_values) == length
    return {
        "action": [[value] * LIBERO_ACTION_DIM for value in action_values],
        "episode_index": [episode_index] * length,
        "frame_index": list(range(length)),
        "observation.state": [[value] * LIBERO_STATE_DIM for value in state_values],
        "task_index": [0] * length,
        "timestamp": [index / 20.0 for index in range(length)],
    }


def _make_suite(
    tmp_path: Path,
    *,
    name: str,
    base: float,
    registry: dict[str, dict[str, list]],
) -> Path:
    root = tmp_path / name
    episodes = [{"episode_index": 0, "length": 3, "tasks": ["synthetic task"]}]
    arrays = {
        0: _episode_values(
            episode_index=0,
            length=3,
            state_values=[base, base + 1.0, base + 50.0],
            action_values=[base, base + 1.0, 777.0],
        )
    }
    if name == GOAL_NAME:
        episodes.append({"episode_index": 82, "length": 2, "tasks": ["synthetic task"]})
        arrays[82] = _episode_values(
            episode_index=82,
            length=2,
            state_values=[999.0, 999.0],
            action_values=[999.0, 999.0],
        )
    total_frames = sum(row["length"] for row in episodes)
    _write_json(
        root / "meta/info.json",
        {
            "chunks_size": 1000,
            "codebase_version": "v2.1",
            "data_path": (
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            ),
            "features": {
                "action": {"dtype": "float32", "shape": [LIBERO_ACTION_DIM]},
                "observation.images.image": {
                    "dtype": "video",
                    "shape": [256, 256, 3],
                },
                "observation.images.wrist_image": {
                    "dtype": "video",
                    "shape": [256, 256, 3],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [LIBERO_STATE_DIM],
                },
            },
            "fps": 20,
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
        },
    )
    _write_json(
        root / "meta/stats_gr00t.json",
        {
            "statistics": {
                "action": _stats(LIBERO_ACTION_DIM),
                "observation.state": _stats(LIBERO_STATE_DIM),
            }
        },
    )
    _write_jsonl(
        root / "meta/tasks.jsonl",
        [{"task": "synthetic task", "task_index": 0}],
    )
    _write_jsonl(root / "meta/episodes.jsonl", episodes)
    for episode_index, values in arrays.items():
        parquet = root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(f"{name}:{episode_index}:synthetic-v1".encode())
        registry[str(parquet.resolve())] = values
    return root


def _benchmark_contract() -> dict:
    payload = yaml.safe_load(
        Path("benchmarks/libero/policy_config.yml").read_text(encoding="utf-8")
    )
    return payload["expected_server_contract"]


def _config(roots: list[Path], output: Path):
    return OmegaConf.create(
        {
            "dataloader": {
                "action_mode": LIBERO_ACTION_MODE,
                "action_stats_path": str(output),
                "benchmark_contract": _benchmark_contract(),
                "dataset_roots": [str(root) for root in roots],
                "delta_action": False,
                "excluded_episodes": [GOAL_EXCLUSION],
                "causal_temporal": True,
                "camera_layout": [
                    "head_camera",
                    "left_wrist_camera",
                    "right_wrist_camera",
                ],
                "filter_static_segments": False,
                "height": 384,
                "multiview": True,
                "normalize_mode": "min-max",
                "num_frames": 113,
                "repeat": 1,
                "seed": 20260806,
                "split": "train",
                "state_mode": LIBERO_STATE_MODE,
                "state_normalize_mode": "min-max",
                "target_camera": "head_camera",
                "temporal_compression": 4,
                "training_video_rotation_degrees": 0,
                "type": "libero",
                "val_ratio": 0.0,
                "verify_known_repairs": True,
                "verify_stats_source": True,
                "video_stride": 4,
                "width": 320,
                "window_stride": 4,
            },
            "model": {
                "architecture": {
                    "action_dim": LIBERO_ACTION_DIM,
                    "delta_action": False,
                    "state_dim": LIBERO_STATE_DIM,
                    "use_proprioception": True,
                },
                "video_backbone": {"continuous_timestep_conditioning": True},
            },
            "training": {
                "action_stats_population_sha256": None,
                "action_stats_sha256": None,
                "optimizer_master_weights": True,
                "preserve_frozen_input_grad_modules": ["video_backbone"],
            },
        }
    )


@pytest.fixture
def selected_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry: dict[str, dict[str, list]] = {}
    bases = [0.0, 10.0, 20.0, 30.0]
    roots = [
        _make_suite(tmp_path, name=name, base=base, registry=registry)
        for name, base in zip(LIBERO_TRAINING_DATASET_NAMES, bases, strict=True)
    ]
    _install_fake_pyarrow(monkeypatch, registry)
    return roots, registry


def _expected_statistics(values: list[float], *, dim: int) -> dict[str, np.ndarray]:
    rows = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(rows, [0.01, 0.99], method="linear")
    scalars = {
        "max": rows.max(),
        "mean": rows.mean(),
        "min": rows.min(),
        "q01": quantiles[0],
        "q99": quantiles[1],
        "std": max(rows.std(ddof=0), 1.0e-3),
    }
    return {
        field: np.full(dim, value, dtype=np.float32) for field, value in scalars.items()
    }


def test_selected_stats_exclude_episode_tail_actions_and_window_multiplicity(
    tmp_path: Path, selected_fixture
) -> None:
    roots, _registry = selected_fixture
    config = _config(list(reversed(roots)), tmp_path / "selected.npy")
    payload = build_libero_selected_row_stats(config)

    assert payload["schema_version"] == LIBERO_SELECTED_STATS_SCHEMA_VERSION
    assert payload["num_timesteps"] == 12
    assert payload["num_action_rows"] == 8
    manifest = payload["population_manifest"]
    assert manifest["counts"] == {
        "action_rows": 8,
        "excluded_episodes": 1,
        "selected_episodes": 4,
        "source_episodes": 5,
        "source_state_rows": 14,
        "state_rows": 12,
        "suites": 4,
    }
    assert manifest["population_manifest_sha256"] == population_manifest_sha256(
        manifest
    )
    expected_action = _expected_statistics(
        [0, 1, 10, 11, 20, 21, 30, 31], dim=LIBERO_ACTION_DIM
    )
    expected_state = _expected_statistics(
        [0, 1, 50, 10, 11, 60, 20, 21, 70, 30, 31, 80],
        dim=LIBERO_STATE_DIM,
    )
    for field, expected in expected_action.items():
        np.testing.assert_array_equal(payload[LIBERO_ACTION_MODE][field], expected)
    for field, expected in expected_state.items():
        np.testing.assert_array_equal(payload[LIBERO_STATE_MODE][field], expected)
    assert float(payload[LIBERO_ACTION_MODE]["max"].max()) < 777.0
    assert float(payload[LIBERO_STATE_MODE]["max"].max()) < 999.0

    repeated = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    repeated.dataloader.repeat = 7
    repeated.dataloader.num_frames = 5
    repeated.dataloader.window_stride = 2
    repeated_payload = build_libero_selected_row_stats(repeated)
    assert repeated_payload["num_timesteps"] == 12
    assert repeated_payload["num_action_rows"] == 8
    np.testing.assert_array_equal(
        repeated_payload[LIBERO_ACTION_MODE]["mean"],
        payload[LIBERO_ACTION_MODE]["mean"],
    )
    assert repeated_payload["population_manifest"] == manifest

    sparse_windows = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    sparse_windows.dataloader.num_frames = 2
    sparse_windows.dataloader.window_stride = 4
    sparse_payload = build_libero_selected_row_stats(sparse_windows)
    assert sparse_payload["population_manifest"] == manifest
    with pytest.raises(ValueError, match="does not supervise every selected"):
        validate_libero_window_coverage_for_config(sparse_windows)


def test_selected_stats_are_root_order_deterministic_exclusive_and_read_only(
    tmp_path: Path, selected_fixture
) -> None:
    roots, _registry = selected_fixture
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    first_result = write_libero_selected_row_stats(first, _config(roots, first))
    second_result = write_libero_selected_row_stats(
        second, _config(list(reversed(roots)), second)
    )

    assert first_result == second_result
    assert sha256_file(first) == sha256_file(second)
    assert stat.S_IMODE(first.stat().st_mode) == 0o444
    assert first_result["population_counts"]["selected_episodes"] == 4
    assert len(first_result["selection_contract_sha256"]) == 64
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_libero_selected_row_stats(first, _config(roots, first))

    rejected = tmp_path / "rejected.npy"
    with pytest.raises(ValueError, match="before artifact publication"):
        write_libero_selected_row_stats(
            rejected,
            _config(roots, rejected),
            expected_population_counts={"suites": 999},
        )
    assert not rejected.exists()


def test_selected_stats_live_preflight_rejects_contract_or_parquet_drift(
    tmp_path: Path, selected_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, _registry = selected_fixture
    output = tmp_path / "selected.npy"
    config = _config(roots, output)
    result = write_libero_selected_row_stats(output, config)
    config.training.action_stats_sha256 = result["artifact_sha256"]
    config.training.action_stats_population_sha256 = result[
        "population_manifest_sha256"
    ]

    payload = np.load(output, allow_pickle=True).item()
    validate_libero_selected_stats_for_config(payload, config, verify_live_sources=True)

    import sana_wam.dataloader.libero_selected_stats as selected_stats_module

    with monkeypatch.context() as manifest_only:
        manifest_only.setattr(
            selected_stats_module,
            "read_libero_episode_arrays",
            lambda _episode: pytest.fail("manifest-only validation read numeric rows"),
        )
        validate_libero_selected_stats_for_config(
            payload,
            config,
            verify_live_sources=True,
            verify_numeric_sources=False,
        )

    import sana_wam.train.libero_contract as contract_module

    monkeypatch.setattr(
        contract_module,
        "LIBERO_PRODUCTION_DATASET_ROOTS",
        tuple(sorted(str(root.resolve()) for root in roots)),
    )
    monkeypatch.setattr(
        contract_module,
        "LIBERO_PRODUCTION_STATS_PATH",
        str(output.resolve()),
    )
    monkeypatch.setattr(
        contract_module,
        "LIBERO_PRODUCTION_POPULATION_COUNTS",
        payload["population_manifest"]["counts"],
    )
    validate_libero_training_config(config, require_materialized_stats=True)

    wrong_seed = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    wrong_seed.dataloader.seed += 1
    with pytest.raises(ValueError, match="dataloader.seed"):
        validate_libero_training_config(wrong_seed, require_materialized_stats=True)

    wrong_seed_type = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    wrong_seed_type.dataloader.seed = True
    with pytest.raises(ValueError, match="dataloader.seed"):
        validate_libero_training_config(
            wrong_seed_type, require_materialized_stats=True
        )

    wrong_numeric = copy.deepcopy(payload)
    wrong_numeric[LIBERO_ACTION_MODE]["mean"] = wrong_numeric[LIBERO_ACTION_MODE][
        "mean"
    ] + np.float32(1.0)
    with pytest.raises(ValueError, match="numeric statistics differ"):
        validate_libero_selected_stats_for_config(
            wrong_numeric, config, verify_live_sources=True
        )

    parquet = roots[0] / "data/chunk-000/episode_000000.parquet"
    parquet.chmod(0o644)
    parquet.write_bytes(parquet.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="live selected sources"):
        validate_libero_training_config(config, require_materialized_stats=True)


def test_selected_stats_reject_symlinked_roots_and_malformed_pins(
    tmp_path: Path, selected_fixture
) -> None:
    roots, _registry = selected_fixture
    root_link = tmp_path / "root-link"
    root_link.symlink_to(roots[0], target_is_directory=True)
    config = _config([root_link, *roots[1:]], tmp_path / "selected.npy")
    with pytest.raises(ValueError, match="must not be a symlink"):
        build_libero_selected_row_stats(config)

    payload = build_libero_selected_row_stats(_config(roots, tmp_path / "selected.npy"))
    malformed = copy.deepcopy(payload)
    malformed["population_manifest"]["sources"][0]["metadata"]["meta/info.json"][
        "size_bytes"
    ] = 0
    malformed["population_manifest"]["population_manifest_sha256"] = (
        population_manifest_sha256(malformed["population_manifest"])
    )
    with pytest.raises(ValueError, match="digest/size differs"):
        validate_libero_selected_stats_payload(malformed)

    wrong_exclusion = copy.deepcopy(payload)
    excluded = next(
        source["excluded_episodes"]
        for source in wrong_exclusion["population_manifest"]["sources"]
        if source["excluded_episodes"]
    )
    excluded[0]["episode_index"] = 83
    wrong_exclusion["population_manifest"]["population_manifest_sha256"] = (
        population_manifest_sha256(wrong_exclusion["population_manifest"])
    )
    with pytest.raises(ValueError, match="pins differ from selection contract"):
        validate_libero_selected_stats_payload(wrong_exclusion)

    wrong_type = OmegaConf.create(
        OmegaConf.to_container(_config(roots, tmp_path / "typed.npy"), resolve=True)
    )
    wrong_type.dataloader.val_ratio = False
    with pytest.raises(ValueError, match="val_ratio"):
        build_libero_selected_row_stats(wrong_type)


def test_selector_checks_nonselected_eligible_parquet(
    selected_fixture,
) -> None:
    roots, registry = selected_fixture
    first_root = roots[0]
    info_path = first_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] += 1
    info["total_frames"] += 3
    _write_json(info_path, info)
    episodes_path = first_root / "meta/episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    episodes.append({"episode_index": 1, "length": 3, "tasks": ["synthetic task"]})
    _write_jsonl(episodes_path, episodes)
    added = first_root / "data/chunk-000/episode_000001.parquet"
    added.write_bytes(b"additional eligible episode")
    registry[str(added.resolve())] = _episode_values(
        episode_index=1,
        length=3,
        state_values=[2.0, 3.0, 4.0],
        action_values=[2.0, 3.0, 777.0],
    )
    selection = select_libero_episodes(
        roots,
        split="train",
        val_ratio=0.5,
        seed=20260806,
        excluded_episodes=[GOAL_EXCLUSION],
        require_all_exclusions=True,
    )
    selected_keys = {episode.identity for episode in selection.selected}
    nonselected = next(
        episode
        for episode in selection.eligible
        if episode.identity not in selected_keys
    )
    nonselected.data_path().unlink()
    with pytest.raises(ValueError, match="regular non-symlink file"):
        select_libero_episodes(
            roots,
            split="train",
            val_ratio=0.5,
            seed=20260806,
            excluded_episodes=[GOAL_EXCLUSION],
            require_all_exclusions=True,
        )


def test_selected_stats_reject_bad_parquet_identity_and_unknown_exclusion(
    tmp_path: Path, selected_fixture
) -> None:
    roots, registry = selected_fixture
    config = _config(roots, tmp_path / "selected.npy")
    parquet = roots[0] / "data/chunk-000/episode_000000.parquet"
    values = registry[str(parquet.resolve())]
    values["frame_index"] = [0, 2, 1]
    with pytest.raises(ValueError, match="frame_index is not contiguous"):
        build_libero_selected_row_stats(config)

    values["frame_index"] = [0, 1, 2]
    unknown = copy.deepcopy(OmegaConf.to_container(config, resolve=True))
    unknown["dataloader"]["excluded_episodes"] = [
        GOAL_EXCLUSION,
        f"{GOAL_NAME}:999",
    ]
    with pytest.raises(ValueError, match="excluded episode keys were not found"):
        build_libero_selected_row_stats(OmegaConf.create(unknown))
