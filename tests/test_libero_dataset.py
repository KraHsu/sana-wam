from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from sana_wam.dataloader import mixture
from sana_wam.dataloader.libero_dataset import LiberoEpisode, LiberoLeRobotDataset
from sana_wam.dataloader.libero_stats import (
    LIBERO_ACTION_DIM,
    LIBERO_ACTION_MODE,
    LIBERO_STATE_DIM,
    LIBERO_STATE_MODE,
    build_libero_stats_from_metadata,
    write_libero_stats,
)
from sana_wam.train.libero_contract import validate_libero_training_config


def _contract() -> dict:
    config = yaml.safe_load(
        Path("benchmarks/libero/policy_config.yml").read_text(encoding="utf-8")
    )
    return config["expected_server_contract"]


def _mode_stats(dim: int, *, mean: float = 0.0) -> dict:
    return {
        "mean": [mean] * dim,
        "std": [1.0] * dim,
        "min": ([-1.0] * (dim - 1)) + [0.0],
        "max": [1.0] * dim,
        "q01": ([-0.9] * (dim - 1)) + [0.0],
        "q99": [0.9] * (dim - 1) + [1.0],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_root(
    tmp_path: Path,
    *,
    name: str = "libero_spatial_synthetic_lerobot",
    frame_count: int = 5,
    stats_mean: float = 0.0,
) -> Path:
    root = tmp_path / name
    features = {
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": [256, 256, 3],
        },
        "observation.images.image": {
            "dtype": "video",
            "shape": [256, 256, 3],
        },
        "observation.state": {"dtype": "float32", "shape": [8]},
        "action": {"dtype": "float32", "shape": [7]},
    }
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "total_episodes": 1,
            "total_frames": frame_count,
            "fps": 20,
            "chunks_size": 1000,
            "data_path": (
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            ),
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
            "features": features,
        },
    )
    _write_json(
        root / "meta/stats_gr00t.json",
        {
            "statistics": {
                "action": _mode_stats(LIBERO_ACTION_DIM, mean=stats_mean),
                "observation.state": _mode_stats(LIBERO_STATE_DIM, mean=stats_mean),
            }
        },
    )
    _write_jsonl(
        root / "meta/tasks.jsonl", [{"task_index": 0, "task": "pick the bowl"}]
    )
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [{"episode_index": 0, "tasks": ["pick the bowl"], "length": frame_count}],
    )
    parquet = root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"synthetic parquet placeholder")
    for key in ("observation.images.image", "observation.images.wrist_image"):
        video = root / f"videos/chunk-000/{key}/episode_000000.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"synthetic video placeholder")
    return root


def _write_stats(tmp_path: Path, roots: list[Path]) -> Path:
    path = tmp_path / "action_stats.npy"
    np.save(path, build_libero_stats_from_metadata(roots), allow_pickle=True)
    return path


def _dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LiberoLeRobotDataset:
    root = _make_root(tmp_path)
    stats_path = _write_stats(tmp_path, [root])
    states = np.arange(5 * LIBERO_STATE_DIM, dtype=np.float32).reshape(5, 8)
    states = states / 100.0
    actions = np.arange(5 * LIBERO_ACTION_DIM, dtype=np.float32).reshape(5, 7)
    actions = actions / 100.0
    actions[:, -1] = [1.0, 0.0, 1.0, 0.0, 1.0]

    def read_arrays(_self, _episode):
        return states.copy(), actions.copy(), np.zeros(5, dtype=np.int64)

    def read_video(_self, _path, requested):
        frames = {}
        for index in requested:
            array = np.zeros((8, 8, 3), dtype=np.uint8)
            if "wrist" in str(_path):
                array[:, :] = [0, 255, 0]
            else:
                array[0, 0] = [255, 0, 0]
                array[-1, -1] = [0, 0, 255]
            frames[index] = Image.fromarray(array, mode="RGB")
        return frames

    monkeypatch.setattr(LiberoLeRobotDataset, "_read_episode_arrays", read_arrays)
    monkeypatch.setattr(LiberoLeRobotDataset, "_read_video_indices", read_video)
    return LiberoLeRobotDataset(
        dataset_roots=[root],
        action_stats_path=stats_path,
        benchmark_contract=_contract(),
        num_frames=5,
        video_stride=1,
        window_stride=1,
        height=96,
        width=96,
        val_ratio=0.0,
        verify_known_repairs=False,
    )


def test_stats_pool_mean_std_and_source_manifest(tmp_path: Path) -> None:
    first = _make_root(tmp_path, name="suite_a", frame_count=4, stats_mean=0.0)
    second = _make_root(tmp_path, name="suite_b", frame_count=6, stats_mean=2.0)
    payload = build_libero_stats_from_metadata([second, first])

    assert payload["num_timesteps"] == 10
    np.testing.assert_allclose(payload[LIBERO_ACTION_MODE]["mean"], 1.2)
    np.testing.assert_allclose(
        payload[LIBERO_ACTION_MODE]["std"], np.sqrt(1.96), rtol=1e-6
    )
    assert [source["dataset"] for source in payload["source_manifest"]["sources"]] == [
        "suite_a",
        "suite_b",
    ]


def test_stats_writer_is_exclusive_and_freezes_the_artifact(tmp_path: Path) -> None:
    root = _make_root(tmp_path, name="suite_stats")
    output = tmp_path / "stats.npy"
    digest = write_libero_stats(output, [root])
    assert len(digest) == 64
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_libero_stats(output, [root])


def test_dataset_preserves_obs_t_action_t_and_video_orientation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, monkeypatch)
    sample = dataset[0]
    raw_actions = np.arange(5 * LIBERO_ACTION_DIM, dtype=np.float32).reshape(5, 7)
    raw_actions = raw_actions / 100.0
    raw_actions[:, -1] = [1.0, 0.0, 1.0, 0.0, 1.0]
    expected = dataset._action_normalizer.normalize(raw_actions[:4])

    assert sample["action_alignment"] == "observation_t_to_action_t"
    torch.testing.assert_close(sample["action"], torch.from_numpy(expected))
    assert sample["action"].shape == (4, 7)
    assert sample["proprio"].shape == (1, 8)
    assert sample["proprio_seq"].shape == (5, 8)
    assert sample["action_mask"].tolist() == [True, True, True, True]
    assert sample["video_mask"].tolist() == [True, True, True, True, True]
    assert sample["prompt"].endswith("pick the bowl")

    composite = np.asarray(sample["video"][0])
    np.testing.assert_array_equal(composite[0, 0], [255, 0, 0])
    np.testing.assert_array_equal(composite[64, 0], [0, 255, 0])
    np.testing.assert_array_equal(composite[-1, -1], [0, 0, 0])


def test_dataset_tail_padding_masks_without_crossing_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, monkeypatch)
    sample = dataset[3]

    assert sample["start_frame"] == 3
    assert sample["end_frame"] == 5
    assert sample["action_mask"].tolist() == [True, False, False, False]
    assert sample["video_mask"].tolist() == [True, True, False, False, False]


def test_parquet_identity_columns_guard_video_row_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = LiberoEpisode(
        dataset="synthetic",
        root=tmp_path,
        episode_index=4,
        length=3,
        task="pick the bowl",
        task_index=2,
        chunks_size=1000,
        data_path_template="episode_{episode_index:06d}.parquet",
        video_path_template="{video_key}/episode_{episode_index:06d}.mp4",
    )
    values = {
        "observation.state": np.zeros((3, 8), dtype=np.float32).tolist(),
        "action": np.zeros((3, 7), dtype=np.float32).tolist(),
        "task_index": [2, 2, 2],
        "episode_index": [4, 4, 4],
        "frame_index": [0, 1, 2],
        "timestamp": [0.0, 0.05, 0.1],
    }

    class Column:
        def __init__(self, value):
            self.value = value

        def to_pylist(self):
            return self.value

    class Table:
        def __getitem__(self, key):
            return Column(values[key])

    parquet = types.ModuleType("pyarrow.parquet")
    parquet.read_table = lambda *_args, **_kwargs: Table()
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    pyarrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    dataset = object.__new__(LiberoLeRobotDataset)

    states, actions, task_indices = dataset._read_episode_arrays(episode)
    assert states.shape == (3, 8)
    assert actions.shape == (3, 7)
    assert task_indices.tolist() == [2, 2, 2]

    values["frame_index"] = [1, 0, 2]
    with pytest.raises(ValueError, match="frame_index is not contiguous"):
        dataset._read_episode_arrays(episode)


def test_factory_dispatches_libero_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        LiberoLeRobotDataset,
        "from_config",
        classmethod(lambda _cls, config, split="train": (sentinel, config, split)),
    )
    config = {"type": "libero"}
    assert mixture.build_training_dataset(config, split="val") == (
        sentinel,
        config,
        "val",
    )


def test_from_config_rejects_unknown_or_silent_filter_fields() -> None:
    with pytest.raises(ValueError, match="unknown LIBERO dataloader fields"):
        LiberoLeRobotDataset.from_config({"type": "libero", "typo_field": 1})
    with pytest.raises(ValueError, match="does not implement static-segment"):
        LiberoLeRobotDataset.from_config(
            {"type": "libero", "filter_static_segments": True}
        )


def test_launcher_rejects_libero_dimension_or_delta_drift() -> None:
    dataset = type("Dataset", (), {"action_dim": 7, "state_dim": 8})()
    config = {
        "model": {
            "architecture": {
                "action_dim": 7,
                "state_dim": 8,
                "use_proprioception": True,
                "delta_action": False,
            }
        },
        "dataloader": {
            "type": "libero",
            "action_mode": LIBERO_ACTION_MODE,
            "state_mode": LIBERO_STATE_MODE,
            "normalize_mode": "min-max",
            "state_normalize_mode": "min-max",
            "delta_action": False,
            "training_video_rotation_degrees": 0,
            "benchmark_contract": _contract(),
        },
    }
    validate_libero_training_config(config, dataset=dataset)

    wrong = json.loads(json.dumps(config))
    wrong["model"]["architecture"]["action_dim"] = 20
    with pytest.raises(ValueError, match="action_dim"):
        validate_libero_training_config(wrong, dataset=dataset)

    wrong = json.loads(json.dumps(config))
    wrong["dataloader"]["delta_action"] = True
    with pytest.raises(ValueError, match="dataloader.delta_action"):
        validate_libero_training_config(wrong, dataset=dataset)


def test_training_template_matches_benchmark_checkpoint_contract() -> None:
    template = yaml.safe_load(
        Path("configs/benchmarks/libero/train_libero_ar_baseline.yaml").read_text(
            encoding="utf-8"
        )
    )
    dataloader = template["dataloader"]
    architecture = template["model"]["architecture"]

    assert dataloader["benchmark_contract"] == _contract()
    assert dataloader["action_mode"] == LIBERO_ACTION_MODE
    assert dataloader["state_mode"] == LIBERO_STATE_MODE
    assert dataloader["delta_action"] is False
    assert architecture["delta_action"] is False
    assert architecture["action_dim"] == 7
    assert architecture["state_dim"] == 8
    assert architecture["ar_chunkwise_temporal_ops"] is True
