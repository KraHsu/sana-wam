from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from sana_wam.dataloader.libero_task_balanced_sampler import (
    LiberoTaskBalancedDistributedSampler,
)
from sana_wam.train.libero_contract import _validate_libero_sampler_config
from sana_wam.train.trainer import Trainer


CONFIG = Path("configs/experiments/libero_formal_r8_taskbalanced_8gpu_continuous2.yaml")
ONE_ROUND_CONFIG = Path(
    "configs/experiments/libero_formal_r8_taskbalanced_8gpu_successor1.yaml"
)
RUNNER = Path("scripts/train_libero_formal_r8_taskbalanced_8gpu_continuous2.py")


@dataclass(frozen=True)
class _Episode:
    dataset: str
    task_index: int
    task: str


class _Dataset:
    dataset_type = "libero"
    repeat = 1

    def __init__(self) -> None:
        self._episodes = tuple(
            _Episode(
                dataset=f"suite-{task_index // 10}",
                task_index=task_index % 10,
                task=f"task-{task_index}",
            )
            for task_index in range(40)
        )
        self._windows = tuple(
            (episode_index, start)
            for episode_index in range(40)
            for start in range(episode_index + 1)
        )

    def __len__(self) -> int:
        return len(self._windows)


def _global_epoch(dataset: _Dataset, epoch: int) -> list[int]:
    values: list[int] = []
    for rank in range(8):
        sampler = LiberoTaskBalancedDistributedSampler(
            dataset,
            num_replicas=8,
            rank=rank,
            seed=20260810,
        )
        sampler.set_epoch(epoch)
        values.extend(sampler)
    return values


def test_two_complete_sampler_epochs_preserve_equal_cumulative_task_quota() -> None:
    dataset = _Dataset()
    epoch_zero = _global_epoch(dataset, 0)
    epoch_one = _global_epoch(dataset, 1)
    assert epoch_zero != epoch_one
    assert len(epoch_zero) == len(epoch_one) == 840

    cumulative = Counter((*epoch_zero, *epoch_one))
    offset = 0
    for task_size in range(1, 41):
        assert (
            sum(cumulative[index] for index in range(offset, offset + task_size)) == 42
        )
        offset += task_size


def test_continuous_two_round_config_is_a_single_training_recipe_delta() -> None:
    candidate = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    one_round = OmegaConf.to_container(OmegaConf.load(ONE_ROUND_CONFIG), resolve=True)
    assert candidate["model"] == one_round["model"]
    assert candidate["dataloader"] == one_round["dataloader"]

    training = candidate["training"]
    baseline_training = one_round["training"]
    assert training["max_steps"] == training["formal_final_step"] == 8_680
    assert training["lr_schedule_steps"] == 8_680
    assert training["warmup_steps"] == 50
    assert training["balanced_rounds_in_this_run"] == 2
    assert training["libero_sampler_contract"] == "task_balanced_40_v1"
    assert training["optimizer_master_weights"] is True
    assert training["optimizer_foreach"] is False
    assert training["init_checkpoint_sha256"] == (
        "4a50f6b90b6d04f7c24a25ad529c2bc96d3a35ac257a6dbe42826f989539ab09"
    )

    allowed_delta = {
        "output_dir",
        "max_steps",
        "formal_final_step",
        "lr_schedule_steps",
        "balanced_rounds_in_this_run",
    }
    for key in allowed_delta:
        training.pop(key)
        baseline_training.pop(key)
    assert training == baseline_training


def test_single_cosine_has_no_round_boundary_restart() -> None:
    trainer = SimpleNamespace(t={"warmup_steps": 50})
    assert Trainer._lr_lambda(trainer, 0, 8_680) == 0.0
    assert Trainer._lr_lambda(trainer, 49, 8_680) == 0.98
    assert Trainer._lr_lambda(trainer, 50, 8_680) == 1.0
    boundary_scale = Trainer._lr_lambda(trainer, 4_340, 8_680)
    assert 0.49 < boundary_scale < 0.52
    final_scale = Trainer._lr_lambda(trainer, 8_679, 8_680)
    assert 0.0 < final_scale < 1e-6


def test_sampler_contract_scales_exact_steps_with_balanced_rounds() -> None:
    contract = {
        "dataloader": {"repeat": 1},
        "training": {
            "libero_sampler_contract": "task_balanced_40_v1",
            "balanced_rounds_in_this_run": 2,
            "batch_size": 1,
            "expected_global_batch_size": 8,
            "expected_world_size": 8,
            "formal_final_step": 8_680,
            "gradient_accumulation_steps": 1,
            "lr_schedule_steps": 8_680,
            "max_steps": 8_680,
        },
    }
    _validate_libero_sampler_config(contract, dataset=None)
    contract["training"]["max_steps"] = 4_340
    with pytest.raises(ValueError, match="training.max_steps=8680"):
        _validate_libero_sampler_config(contract, dataset=None)


def test_fail_closed_runner_pins_total_draws_and_one_trainer_call() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "EXPECTED_BALANCED_ROUNDS = 2" in source
    assert "EXPECTED_TOTAL_DRAWS_PER_TASK = 1_736" in source
    assert "EXPECTED_TOTAL_WINDOW_DRAWS = 69_440" in source
    assert "EXPECTED_TOTAL_DRAWS_PER_RANK = 8_680" in source
    assert "EXPECTED_FINAL_STEP = 8_680" in source
    assert source.count("trainer = Trainer(") == 1
    assert source.count("output = trainer.train()") == 1
    assert '"--nproc-per-node=8"' in source
    assert '"--max-restarts=0"' in source
    assert '"optimizer_state_reset_between_rounds": False' in source
    assert '"lr_restart_between_rounds": False' in source
