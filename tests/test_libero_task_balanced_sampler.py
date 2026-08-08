from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from sana_wam.dataloader.libero_task_balanced_sampler import (
    LIBERO_TASK_BALANCED_SAMPLER_CONTRACT,
    LiberoTaskBalancedDistributedSampler,
)
from sana_wam.train.libero_contract import _validate_libero_sampler_config


@dataclass(frozen=True)
class _Episode:
    dataset: str
    task_index: int
    task: str


class _Dataset:
    dataset_type = "libero"
    repeat = 1
    action_dim = 7
    state_dim = 8

    def __init__(self, *, task_count: int = 40) -> None:
        self._episodes = tuple(
            _Episode(
                dataset=f"suite-{task_index // 10}",
                task_index=task_index % 10,
                task=f"task-{task_index}",
            )
            for task_index in range(task_count)
        )
        windows = []
        for episode_index in range(task_count):
            windows.extend(
                (episode_index, start) for start in range(episode_index + 1)
            )
        self._windows = tuple(windows)

    def __len__(self) -> int:
        return len(self._windows) * self.repeat


def _global_epoch(dataset: _Dataset, epoch: int) -> list[int]:
    values = []
    for rank in range(8):
        sampler = LiberoTaskBalancedDistributedSampler(
            dataset,
            num_replicas=8,
            rank=rank,
            seed=20260810,
        )
        sampler.set_epoch(epoch)
        rank_values = list(sampler)
        assert len(rank_values) == len(sampler) == 105
        values.extend(rank_values)
    return values


def test_40_tasks_receive_equal_quota_with_uniform_bounded_repetition() -> None:
    dataset = _Dataset()
    global_indices = _global_epoch(dataset, epoch=0)
    counts = Counter(global_indices)
    assert len(global_indices) == 840

    window_offset = 0
    for task_index, task_size in enumerate(range(1, 41)):
        task_counts = [counts[index] for index in range(window_offset, window_offset + task_size)]
        assert sum(task_counts) == 21
        assert max(task_counts) - min(task_counts) <= 1
        assert max(task_counts) == (21 + task_size - 1) // task_size
        window_offset += task_size

    sampler = LiberoTaskBalancedDistributedSampler(
        dataset, num_replicas=8, rank=0, seed=20260810
    )
    summary = sampler.audit_summary()
    assert summary["contract"] == LIBERO_TASK_BALANCED_SAMPLER_CONTRACT
    assert summary["task_count"] == 40
    assert summary["input_window_count"] == 820
    assert summary["draws_per_task"] == 21
    assert summary["global_draw_count"] == 840
    assert summary["draws_per_rank"] == 105
    assert summary["max_draws_per_window"] == 21
    assert sum(summary["window_draw_multiplicity_counts"].values()) == 820
    assert sum(
        int(multiplicity) * count
        for multiplicity, count in summary[
            "window_draw_multiplicity_counts"
        ].items()
    ) == 840


def test_rank_streams_are_deterministic_and_epoch_sensitive() -> None:
    dataset = _Dataset()
    first = _global_epoch(dataset, epoch=0)
    assert first == _global_epoch(dataset, epoch=0)
    assert first != _global_epoch(dataset, epoch=1)


def test_sampler_rejects_population_or_replication_drift() -> None:
    with pytest.raises(ValueError, match="exactly 40"):
        LiberoTaskBalancedDistributedSampler(
            _Dataset(task_count=39), num_replicas=8, rank=0, seed=0
        )
    repeated = _Dataset()
    repeated.repeat = 2
    with pytest.raises(ValueError, match="repeat=1"):
        LiberoTaskBalancedDistributedSampler(
            repeated, num_replicas=8, rank=0, seed=0
        )
    with pytest.raises(ValueError, match="rank must be smaller"):
        LiberoTaskBalancedDistributedSampler(
            _Dataset(), num_replicas=8, rank=8, seed=0
        )


def test_sampler_contract_is_opt_in_and_unknown_values_fail_closed() -> None:
    _validate_libero_sampler_config({"training": {}}, dataset=None)
    with pytest.raises(ValueError, match="unknown training.libero_sampler_contract"):
        _validate_libero_sampler_config(
            {"training": {"libero_sampler_contract": "almost-balanced"}},
            dataset=None,
        )


def test_trainer_preserves_legacy_distributed_sampler_fallback() -> None:
    source = Path("src/sana_wam/train/trainer.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "LiberoTaskBalancedDistributedSampler(" in source
    assert "DistributedSampler(self.dataset, shuffle=True, seed=self.base_seed)" in source
    assert 'self.t.get("libero_sampler_contract", None)' in source


def test_successor_config_and_runner_pin_the_balanced_r8_axis() -> None:
    config = Path(
        "configs/experiments/libero_formal_r8_taskbalanced_8gpu_successor1.yaml"
    )
    runner = Path(
        "scripts/train_libero_formal_r8_taskbalanced_8gpu_successor1.py"
    )
    cfg = OmegaConf.load(config)
    assert cfg.training.libero_sampler_contract == "task_balanced_40_v1"
    assert cfg.training.max_steps == cfg.training.formal_final_step == 4340
    assert cfg.training.lr_schedule_steps == 4340
    assert cfg.training.balanced_rounds_in_this_run == 1
    assert OmegaConf.select(cfg, "training.epochs_in_this_run", default=None) is None
    assert (
        OmegaConf.select(cfg, "training.cumulative_epochs_after_run", default=None)
        is None
    )
    assert cfg.training.init_checkpoint_sha256 == (
        "4a50f6b90b6d04f7c24a25ad529c2bc96d3a35ac257a6dbe42826f989539ab09"
    )
    assert cfg.training.formal_predecessor_r8_result_sha256 == (
        "11b1b68db1b138bbe0c638f0f85beb174c8ebe8c8d1925c5abd06bb1378e966c"
    )
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    assert "EXPECTED_DRAWS_PER_TASK = 868" in source
    assert "EXPECTED_WINDOW_DRAWS = 34_720" in source
    assert "EXPECTED_DRAWS_PER_RANK = 4_340" in source
    assert "EXPECTED_MAX_DRAWS_PER_WINDOW = 2" in source
    assert '"--nproc-per-node=8"' in source
    assert '"--max-restarts=0"' in source
