"""Deterministic task-balanced sampling for the fixed 40-task LIBERO corpus."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Sampler


LIBERO_TASK_BALANCED_SAMPLER_CONTRACT = "task_balanced_40_v1"
LIBERO_TASK_BALANCED_EXPECTED_TASKS = 40


def _plain_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


class LiberoTaskBalancedDistributedSampler(Sampler[int]):
    """Give every LIBERO task an equal quota and sample windows uniformly.

    A task is identified by ``(dataset, task_index, task)``.  Within a task,
    windows are shuffled without replacement.  If its fixed quota exceeds its
    number of windows, another independently shuffled full cycle is started;
    therefore per-window draw counts differ by at most one.  The complete
    global draw stream is shuffled once more and strided across ranks.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        expected_num_tasks: int = LIBERO_TASK_BALANCED_EXPECTED_TASKS,
    ) -> None:
        self.num_replicas = _plain_nonnegative_int(
            num_replicas, label="num_replicas"
        )
        if self.num_replicas == 0:
            raise ValueError("num_replicas must be positive")
        self.rank = _plain_nonnegative_int(rank, label="rank")
        if self.rank >= self.num_replicas:
            raise ValueError("rank must be smaller than num_replicas")
        self.seed = _plain_nonnegative_int(seed, label="seed")
        self.expected_num_tasks = _plain_nonnegative_int(
            expected_num_tasks, label="expected_num_tasks"
        )
        if self.expected_num_tasks == 0:
            raise ValueError("expected_num_tasks must be positive")
        if getattr(dataset, "dataset_type", None) != "libero":
            raise ValueError("task-balanced sampler requires dataset_type='libero'")
        if getattr(dataset, "repeat", None) != 1:
            raise ValueError("task-balanced sampler requires dataset.repeat=1")

        episodes = getattr(dataset, "_episodes", None)
        windows = getattr(dataset, "_windows", None)
        if not isinstance(episodes, (tuple, list)) or not episodes:
            raise ValueError("task-balanced sampler requires materialized episodes")
        if not isinstance(windows, (tuple, list)) or not windows:
            raise ValueError("task-balanced sampler requires materialized windows")
        if len(dataset) != len(windows):
            raise ValueError(
                "task-balanced sampler requires one dataset index per window"
            )

        grouped: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        for dataset_index, window in enumerate(windows):
            if (
                not isinstance(window, (tuple, list))
                or len(window) != 2
                or isinstance(window[0], bool)
                or not isinstance(window[0], int)
                or window[0] < 0
                or window[0] >= len(episodes)
            ):
                raise ValueError("LIBERO window contains an invalid episode index")
            episode = episodes[window[0]]
            dataset_name = getattr(episode, "dataset", None)
            task_index = getattr(episode, "task_index", None)
            task_name = getattr(episode, "task", None)
            if not isinstance(dataset_name, str) or not dataset_name:
                raise ValueError("LIBERO episode has an invalid dataset name")
            if (
                isinstance(task_index, bool)
                or not isinstance(task_index, int)
                or task_index < 0
            ):
                raise ValueError("LIBERO episode has an invalid task_index")
            if not isinstance(task_name, str) or not task_name:
                raise ValueError("LIBERO episode has an invalid task name")
            grouped[(dataset_name, task_index, task_name)].append(dataset_index)

        if len(grouped) != self.expected_num_tasks:
            raise ValueError(
                "task-balanced sampler requires exactly "
                f"{self.expected_num_tasks} non-empty tasks, got {len(grouped)}"
            )
        self._task_windows = tuple(
            (identity, tuple(indices))
            for identity, indices in sorted(grouped.items())
        )
        self.draws_per_task = math.ceil(len(windows) / self.expected_num_tasks)
        self.global_num_samples = self.draws_per_task * self.expected_num_tasks
        if self.global_num_samples % self.num_replicas:
            raise ValueError(
                "task-balanced global draw count must divide evenly across ranks"
            )
        self.num_samples = self.global_num_samples // self.num_replicas
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _plain_nonnegative_int(epoch, label="epoch")

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch) % (2**63 - 1))
        global_indices: list[int] = []
        for _identity, task_windows in self._task_windows:
            task_draws: list[int] = []
            while len(task_draws) < self.draws_per_task:
                permutation = torch.randperm(
                    len(task_windows), generator=generator
                ).tolist()
                remaining = self.draws_per_task - len(task_draws)
                task_draws.extend(task_windows[index] for index in permutation[:remaining])
            global_indices.extend(task_draws)
        if len(global_indices) != self.global_num_samples:
            raise RuntimeError("task-balanced sampler constructed a short draw stream")
        global_permutation = torch.randperm(
            self.global_num_samples, generator=generator
        ).tolist()
        shuffled = [global_indices[index] for index in global_permutation]
        rank_indices = shuffled[self.rank :: self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError("task-balanced sampler produced unequal rank lengths")
        return iter(rank_indices)

    def audit_summary(self) -> dict[str, Any]:
        task_window_counts = {
            f"{dataset}:{task_index}:{task}": len(indices)
            for (dataset, task_index, task), indices in self._task_windows
        }
        task_max_draws_per_window = {
            identity: math.ceil(self.draws_per_task / count)
            for identity, count in task_window_counts.items()
        }
        multiplicity_counts: dict[int, int] = defaultdict(int)
        for count in task_window_counts.values():
            full_cycles, remainder = divmod(self.draws_per_task, count)
            if count - remainder:
                multiplicity_counts[full_cycles] += count - remainder
            if remainder:
                multiplicity_counts[full_cycles + 1] += remainder
        return {
            "contract": LIBERO_TASK_BALANCED_SAMPLER_CONTRACT,
            "task_count": len(self._task_windows),
            "input_window_count": sum(task_window_counts.values()),
            "draws_per_task": self.draws_per_task,
            "global_draw_count": self.global_num_samples,
            "draws_per_rank": self.num_samples,
            "num_replicas": self.num_replicas,
            "replacement_semantics": (
                "without_replacement_until_task_exhaustion_then_new_shuffled_cycle"
            ),
            "task_window_counts": task_window_counts,
            "task_max_draws_per_window": task_max_draws_per_window,
            "max_draws_per_window": max(task_max_draws_per_window.values()),
            "window_draw_multiplicity_counts": {
                str(multiplicity): count
                for multiplicity, count in sorted(multiplicity_counts.items())
            },
        }


__all__ = [
    "LIBERO_TASK_BALANCED_EXPECTED_TASKS",
    "LIBERO_TASK_BALANCED_SAMPLER_CONTRACT",
    "LiberoTaskBalancedDistributedSampler",
]
