from __future__ import annotations

from types import SimpleNamespace

from sana_wam.dataloader.robotwin_plan_binding import (
    enumerate_phase6_candidate_pools,
    phase6_expansion_support_at,
)
from sana_wam.dataloader.task_sample_plan import SampleIdentity


class _Child:
    task_name = "task"
    _raw_window_len = 113
    video_stride = 4
    _video_sample_indices = list(range(0, 113, 4))
    causal_temporal = True
    temporal_compression = 4
    _episode_lengths = [10]

    def __init__(self) -> None:
        self._windows = [(0, 6, 113), (0, 5, 113), (0, 8, 113), (0, 0, 113)]

    def _resolve_local_index(self, local_index: int):
        return self._windows[local_index]


class _Dataset:
    def __init__(self) -> None:
        self._sub_datasets = [_Child()]
        self.identity_calls = []

    def __len__(self) -> int:
        return 4

    def resolve_global_index(self, dataset_index: int):
        return 0, dataset_index

    def phase6_identity_at(self, dataset_index: int, *, protocol_seed: int):
        del protocol_seed
        self.identity_calls.append(dataset_index)
        episode_index, start_frame, _ = self._sub_datasets[0]._resolve_local_index(
            dataset_index
        )
        return SampleIdentity(
            task_name="task",
            episode_index=episode_index,
            episode_path="/dataset/task/episode0.hdf5",
            start_frame=start_frame,
            prompt="perform task",
            dataset_index=dataset_index,
        )


def test_enumerator_filters_shared_pool_before_identity_hash_ordering(monkeypatch) -> None:
    dataset = _Dataset()
    contract = SimpleNamespace(
        train_tasks=("task",),
        protocol_seed=20260724,
        task_cycles=1,
    )
    monkeypatch.setattr(
        "sana_wam.dataloader.robotwin_plan_binding.validate_phase6_robotwin_dataset",
        lambda *_args, **_kwargs: None,
    )
    pools = enumerate_phase6_candidate_pools(
        dataset,
        contract,
        dataset_type="robotwin",
    )
    assert [identity.dataset_index for identity in pools["task"]] == [1, 3]
    assert dataset.identity_calls == [1, 3]
    assert phase6_expansion_support_at(dataset, 0)["eligible"] is False
    assert phase6_expansion_support_at(dataset, 1)["eligible"] is True
