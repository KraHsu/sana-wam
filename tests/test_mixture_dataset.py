from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from sana_wam.dataloader import mixture


@dataclass
class _FakeDataset:
    name: str
    size: int
    action_stats_path: str
    action_stats: dict
    action_dim: int = 20

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"source": self.name, "index": index}


def _stats(offset=0.0):
    return {
        "mean": np.full(20, offset, dtype=np.float32),
        "std": np.ones(20, dtype=np.float32),
        "min": np.full(20, -1.0, dtype=np.float32),
        "max": np.full(20, 1.0, dtype=np.float32),
    }


def _patch_sources(monkeypatch, sources):
    def build(config, split="train"):
        assert split == "train"
        return sources[config["name"]]

    monkeypatch.setattr(
        mixture.MultiTaskRoboTwinDataset, "from_config", staticmethod(build)
    )


def _mixture_config(*, epoch_size=None, seed=17):
    config = {
        "type": "mixture",
        "seed": seed,
        "datasets": [
            {"type": "robotwin", "name": "demo", "weight": 2.0},
            {"type": "robotwin", "name": "dagger", "weight": 1.0},
        ],
    }
    if epoch_size is not None:
        config["epoch_size"] = epoch_size
    return config


def test_mixture_uses_exact_source_weight_ratio_not_child_lengths(
    tmp_path, monkeypatch
):
    stats_path = tmp_path / "stats.npy"
    stats_path.write_bytes(b"same-stats")
    sources = {
        "demo": _FakeDataset("demo", 2, str(stats_path), _stats()),
        "dagger": _FakeDataset("dagger", 19, str(stats_path), _stats()),
    }
    _patch_sources(monkeypatch, sources)

    config = _mixture_config(epoch_size=30)
    dataset = mixture.build_training_dataset(config)

    observed = [dataset[index]["source"] for index in range(len(dataset))]
    assert observed.count("demo") == 20
    assert observed.count("dagger") == 10
    assert dataset.source_counts == (20, 10)
    assert len(dataset) == 30
    assert dataset.action_dim == 20
    assert dataset.action_stats_path == str(stats_path.resolve())
    assert dataset.num_frames == 33
    assert config["num_frames"] == 33
    assert config["camera_layout"] == [
        "head_camera",
        "left_camera",
        "right_camera",
    ]


def test_mixture_default_epoch_size_and_set_epoch_are_deterministic(
    tmp_path, monkeypatch
):
    stats_path = tmp_path / "stats.npy"
    stats_path.write_bytes(b"same-stats")
    sources = {
        "demo": _FakeDataset("demo", 4, str(stats_path), _stats()),
        "dagger": _FakeDataset("dagger", 5, str(stats_path), _stats()),
    }
    _patch_sources(monkeypatch, sources)
    config = _mixture_config(seed=123)

    first = mixture.build_training_dataset(config)
    second = mixture.build_training_dataset(config)
    epoch_zero = first._index_map.copy()
    assert len(first) == 9
    np.testing.assert_array_equal(epoch_zero, second._index_map)

    first.set_epoch(1)
    assert not np.array_equal(epoch_zero, first._index_map)
    assert first.source_counts == second.source_counts

    first.set_epoch(0)
    np.testing.assert_array_equal(epoch_zero, first._index_map)


def test_short_quota_samples_from_full_child_and_changes_membership(tmp_path):
    stats_path = tmp_path / "stats.npy"
    stats_path.write_bytes(b"same-stats")
    dataset = mixture.MixtureDataset(
        [_FakeDataset("demo", 20, str(stats_path), _stats())],
        [1.0],
        seed=123,
        epoch_size=4,
    )

    epoch_zero = set(dataset._index_map[:, 1].tolist())
    dataset.set_epoch(1)
    epoch_one = set(dataset._index_map[:, 1].tolist())

    assert len(epoch_zero) == len(epoch_one) == 4
    assert max(epoch_zero | epoch_one) >= 4
    assert epoch_zero != epoch_one


def test_non_mixture_dispatches_to_existing_robotwin_builder(monkeypatch):
    sentinel = object()
    calls = []

    def build(config, split="train"):
        calls.append((config, split))
        return sentinel

    monkeypatch.setattr(
        mixture.MultiTaskRoboTwinDataset, "from_config", staticmethod(build)
    )
    config = {"type": "robotwin", "task_name": "adjust_bottle"}

    assert mixture.build_training_dataset(config, split="val") is sentinel
    assert calls == [(config, "val")]


def test_history_dagger_dispatches_to_strict_manifest_builder(monkeypatch):
    sentinel = object()
    calls = []

    def build(config, split="train"):
        calls.append((config, split))
        return sentinel

    monkeypatch.setattr(
        mixture.RoboTwinHistoryDaggerDataset, "from_config", staticmethod(build)
    )
    config = {
        "type": "robotwin_history_dagger",
        "source_manifest_path": "/strict/dataset_manifest.json",
    }

    assert mixture.build_training_dataset(config, split="train") is sentinel
    assert calls == [(config, "train")]


def test_unknown_dataset_type_fails_closed():
    with pytest.raises(ValueError, match="unsupported dataloader type"):
        mixture.build_training_dataset({"type": "robotwin_history_dager"})


def test_history_dagger_uses_its_strict_geometry_as_config_defaults():
    contract = mixture._config_deploy_contract({"type": "robotwin_history_dagger"})

    assert contract["num_frames"] == 113
    assert contract["video_stride"] == 4
    assert contract["height"] == 384
    assert contract["width"] == 320
    assert mixture._config_deploy_contract({"type": "robotwin"})["num_frames"] == 33


def test_mixture_skips_disabled_sources(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.npy"
    stats_path.write_bytes(b"same-stats")
    sources = {"demo": _FakeDataset("demo", 3, str(stats_path), _stats())}
    _patch_sources(monkeypatch, sources)
    config = _mixture_config(epoch_size=6)
    config["datasets"][1]["enabled"] = False

    dataset = mixture.build_training_dataset(config)

    assert dataset.source_counts == (6,)
    assert {dataset[index]["source"] for index in range(len(dataset))} == {"demo"}


@pytest.mark.parametrize(
    ("location", "field", "value", "match"),
    [
        (
            "source",
            "video_stride",
            2,
            "configured deploy field video_stride mismatch",
        ),
        (
            "top",
            "action_mode",
            "joint",
            "top-level deploy field action_mode conflicts",
        ),
    ],
)
def test_mixture_rejects_deploy_contract_conflicts(
    tmp_path, monkeypatch, location, field, value, match
):
    stats_path = tmp_path / "stats.npy"
    stats_path.write_bytes(b"same-stats")
    sources = {
        "demo": _FakeDataset("demo", 2, str(stats_path), _stats()),
        "dagger": _FakeDataset("dagger", 2, str(stats_path), _stats()),
    }
    _patch_sources(monkeypatch, sources)
    config = _mixture_config(epoch_size=6)
    if location == "source":
        config["datasets"][1][field] = value
    else:
        config[field] = value

    with pytest.raises(ValueError, match=match):
        mixture.build_training_dataset(config)


def test_mixture_rejects_action_stats_file_content_mismatch(tmp_path):
    first_path = tmp_path / "first.npy"
    second_path = tmp_path / "second.npy"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")

    with pytest.raises(ValueError, match="content SHA256 mismatch"):
        mixture.MixtureDataset(
            [
                _FakeDataset("demo", 2, str(first_path), _stats()),
                _FakeDataset("dagger", 2, str(second_path), _stats()),
            ],
            [1.0, 1.0],
        )


def test_mixture_rejects_action_stats_numeric_mismatch(tmp_path):
    stats_path = tmp_path / "stats.npy"
    stats_path.write_bytes(b"same-stats")

    with pytest.raises(ValueError, match="numeric values differ"):
        mixture.MixtureDataset(
            [
                _FakeDataset("demo", 2, str(stats_path), _stats()),
                _FakeDataset("dagger", 2, str(stats_path), _stats(offset=0.25)),
            ],
            [1.0, 1.0],
        )


def test_mixture_rejects_action_dimension_mismatch(tmp_path):
    stats_path = tmp_path / "stats.npy"
    stats_path.write_bytes(b"same-stats")

    with pytest.raises(ValueError, match="action_dim mismatch"):
        mixture.MixtureDataset(
            [
                _FakeDataset("demo", 2, str(stats_path), _stats(), action_dim=20),
                _FakeDataset("dagger", 2, str(stats_path), _stats(), action_dim=14),
            ],
            [1.0, 1.0],
        )
