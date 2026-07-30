"""Bind a frozen Phase-6 task plan to an ordinary MultiTask RoboTwin dataset.

The helpers in this module are deliberately torch-free. Candidate-pool
enumeration reads only the dataset's precomputed window/path metadata; runtime
materialization is delegated to the plan-aware methods on RoboTwinDataset so
the exact prompt is installed before any text transform runs.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
import re
from typing import Any

from sana_wam.dataloader.phase6_dataset_contract import (
    Phase6DatasetContract,
    runtime_window_row,
)
from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_SUPPORT_KEYS,
    ExpansionSupportError,
    expansion_support_for_child_window,
    validate_expansion_geometry,
)
from sana_wam.dataloader.task_sample_plan import (
    PlanSampler,
    PlanValidationError,
    PlannedSample,
    SampleIdentity,
    SamplerContract,
    TaskRoundRobinPlan,
)


PHASE6_PLAN_ROW_KEYS = frozenset(
    {
        "action_sigma",
        "cycle",
        "domain_seeds",
        "global_step",
        "identity",
        "identity_sha256",
        "plan_sha256",
        "position_in_cycle",
    }
)


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PlanValidationError(f"{name} must be a lowercase SHA256 digest")
    return value


def phase6_plan_row_dict(
    row: PlannedSample,
    *,
    plan_sha256: str,
    identity_sha256: str,
) -> dict[str, Any]:
    """Return the one plain-dict audit payload attached to a runtime sample."""

    if not isinstance(row, PlannedSample):
        raise PlanValidationError("phase6 plan row must be a PlannedSample")
    payload = deepcopy(row.to_dict())
    payload["plan_sha256"] = _require_sha256(plan_sha256, "plan_sha256")
    payload["identity_sha256"] = _require_sha256(
        identity_sha256, "identity_sha256"
    )
    if set(payload) != PHASE6_PLAN_ROW_KEYS:
        raise PlanValidationError("phase6_plan_row schema changed")
    return payload


def validate_phase6_robotwin_dataset(
    dataset: Any,
    contract: SamplerContract,
    *,
    dataset_type: str,
) -> None:
    """Fail unless this is the exact ordinary clean_50 42-task dataset."""

    contract.validate()
    if dataset_type != "robotwin":
        raise PlanValidationError(
            "Phase-6 requires dataloader.type='robotwin'; mixture/history/DAgger "
            f"sources are forbidden, got {dataset_type!r}"
        )
    if getattr(dataset, "dataset_type", None) != "robotwin":
        raise PlanValidationError("runtime dataset does not declare type='robotwin'")
    if getattr(dataset, "variant", None) != "clean_50":
        raise PlanValidationError("Phase-6 requires MultiTask variant='clean_50'")

    task_names = getattr(dataset, "task_names", None)
    task_names = task_names() if callable(task_names) else task_names
    if not isinstance(task_names, tuple) or task_names != contract.train_tasks:
        raise PlanValidationError(
            "runtime MultiTask cohort/order differs from the frozen 42-task tuple"
        )
    children = getattr(dataset, "_sub_datasets", None)
    if not isinstance(children, list) or len(children) != len(contract.train_tasks):
        raise PlanValidationError("runtime dataset must expose exactly 42 task children")
    cumulative = getattr(dataset, "_cumulative_lengths", None)
    if not isinstance(cumulative, list) or len(cumulative) != len(children):
        raise PlanValidationError("runtime cumulative-length index is malformed")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in cumulative
    ):
        raise PlanValidationError("runtime cumulative lengths must be positive integers")
    if any(right <= left for left, right in zip(cumulative, cumulative[1:])):
        raise PlanValidationError("runtime cumulative lengths are not strictly increasing")
    if not cumulative or cumulative[-1] != len(dataset):
        raise PlanValidationError("runtime cumulative length does not equal dataset length")
    expected_cumulative = []
    running_length = 0
    for child in children:
        running_length += len(child)
        expected_cumulative.append(running_length)
    if cumulative != expected_cumulative:
        raise PlanValidationError("runtime cumulative offsets differ from child lengths")

    for task, child in zip(contract.train_tasks, children):
        if getattr(child, "task_name", None) != task:
            raise PlanValidationError(
                f"child task key differs from canonical registry name {task!r}"
            )
        try:
            child._validate_phase6_source_contract()
        except Exception as exc:
            raise PlanValidationError(
                f"task {task!r} violates the ordinary clean_50 source contract: {exc}"
            ) from exc
        try:
            validate_expansion_geometry(
                raw_window_frames=getattr(child, "_raw_window_len", None),
                video_stride=getattr(child, "video_stride", None),
                video_sample_indices=getattr(child, "_video_sample_indices", ()),
                causal_temporal=getattr(child, "causal_temporal", None),
                temporal_compression=getattr(child, "temporal_compression", None),
            )
        except ExpansionSupportError as exc:
            raise PlanValidationError(
                f"task {task!r} violates frozen expansion-support geometry: {exc}"
            ) from exc


def phase6_expansion_support_at(
    dataset: Any,
    dataset_index: int,
) -> dict[str, int | bool]:
    """Resolve structural support without decoding frames or reading actions."""

    try:
        child_index, local_index = dataset.resolve_global_index(dataset_index)
        child = dataset._sub_datasets[child_index]
        episode_index, start_frame, logical_length = child._resolve_local_index(
            local_index
        )
        return expansion_support_for_child_window(
            child,
            episode_index=episode_index,
            start_frame=start_frame,
            window_logical_length=logical_length,
        )
    except ExpansionSupportError:
        raise
    except Exception as exc:
        raise ExpansionSupportError(
            f"cannot resolve expansion support for dataset index {dataset_index}: {exc}"
        ) from exc


def enumerate_phase6_candidate_pools(
    dataset: Any,
    contract: SamplerContract,
    *,
    dataset_type: str,
) -> dict[str, list[SampleIdentity]]:
    """Enumerate identity pools without calling any sample materializer."""

    validate_phase6_robotwin_dataset(
        dataset,
        contract,
        dataset_type=dataset_type,
    )
    pools = {task: [] for task in contract.train_tasks}
    for dataset_index in range(len(dataset)):
        try:
            support = phase6_expansion_support_at(dataset, dataset_index)
        except ExpansionSupportError as exc:
            raise PlanValidationError(
                f"cannot derive expansion support at dataset index {dataset_index}: {exc}"
            ) from exc
        if not support["eligible"]:
            continue
        identity = dataset.phase6_identity_at(
            dataset_index,
            protocol_seed=contract.protocol_seed,
        )
        if not isinstance(identity, SampleIdentity):
            raise PlanValidationError("identity enumeration returned a non-identity")
        identity.validate(expected_task=identity.task_name)
        if identity.task_name not in pools:
            raise PlanValidationError(
                f"enumerated task {identity.task_name!r} is not in the frozen cohort"
            )
        pools[identity.task_name].append(identity)
    for task, candidates in pools.items():
        if len(candidates) < contract.task_cycles:
            raise PlanValidationError(
                f"task {task!r} has only {len(candidates)} clean windows"
            )
    return pools


class PlanBoundRoboTwinDataset:
    """Runtime guard that exposes only rows present in one pinned task plan.

    Indices stay in the underlying MultiTask global-index space because
    PlanSampler yields those indices. Any unplanned access, identity drift, or
    prompt drift fails before a sample reaches the model.
    """

    def __init__(
        self,
        dataset: Any,
        plan: TaskRoundRobinPlan,
        *,
        contract: SamplerContract,
        dataset_type: str,
        world_size: int,
        expected_plan_sha256: str,
        expected_identity_sha256: str,
        dataset_contract: Phase6DatasetContract,
    ) -> None:
        if type(world_size) is not int or world_size != 1:
            raise PlanValidationError(
                "Phase-6 uses one planned identity per optimizer step and therefore "
                f"requires world_size=1, got {world_size!r}"
            )
        validate_phase6_robotwin_dataset(
            dataset,
            contract,
            dataset_type=dataset_type,
        )
        if plan.protocol_seed != contract.protocol_seed:
            raise PlanValidationError("dataset contract and plan protocol seeds differ")
        if not isinstance(dataset_contract, Phase6DatasetContract):
            raise PlanValidationError(
                "Phase-6 requires a validated dataset-contract artifact"
            )
        if dataset_contract.plan_sha256 != plan.plan_sha256:
            raise PlanValidationError("dataset contract binds a different task plan")
        if dataset_contract.identity_sha256 != plan.identity_sha256:
            raise PlanValidationError("dataset contract binds different identities")
        self._sampler_guard = PlanSampler(
            plan,
            expected_plan_sha256=expected_plan_sha256,
            expected_identity_sha256=expected_identity_sha256,
        )
        self.dataset = dataset
        self.world_size = world_size
        self.plan = plan
        self.plan_sha256 = plan.plan_sha256
        self.identity_sha256 = plan.identity_sha256
        self._rows_by_dataset_index: dict[int, PlannedSample] = {}
        self._contract_rows_by_dataset_index: dict[int, dict[str, Any]] = {}

        for row in plan.rows:
            dataset_index = row.identity.dataset_index
            if dataset_index in self._rows_by_dataset_index:
                raise PlanValidationError(
                    f"plan reuses dataset index {dataset_index}"
                )
            observed = dataset.phase6_identity_at(
                dataset_index,
                protocol_seed=plan.protocol_seed,
            )
            if observed != row.identity:
                raise PlanValidationError(
                    "plan-to-dataset binding mismatch at global step "
                    f"{row.global_step}: expected={row.identity.audit_tuple!r}, "
                    f"observed={observed.audit_tuple!r}"
                )
            contract_row = dataset_contract.row_for_dataset_index(dataset_index)
            support = phase6_expansion_support_at(dataset, dataset_index)
            contract_support = {
                key: contract_row[key] for key in EXPANSION_SUPPORT_KEYS
            }
            if contract_support != support or not support["eligible"]:
                raise PlanValidationError(
                    "dataset-contract expansion support differs from runtime at "
                    f"global step {row.global_step}"
                )
            expected_contract_fields = {
                "global_step": row.global_step,
                "dataset_index": dataset_index,
                "task_name": row.identity.task_name,
                "episode_index": row.identity.episode_index,
                "episode_path": row.identity.episode_path,
                "start_frame": row.identity.start_frame,
            }
            if any(
                contract_row[key] != value
                for key, value in expected_contract_fields.items()
            ):
                raise PlanValidationError(
                    "dataset-contract row differs from the plan at global step "
                    f"{row.global_step}"
                )
            observed_window = runtime_window_row(
                dataset,
                global_step=row.global_step,
                dataset_index=dataset_index,
            )
            if any(
                contract_row[key] != value
                for key, value in observed_window.items()
            ):
                raise PlanValidationError(
                    "dataset-contract window differs from runtime at global step "
                    f"{row.global_step}"
                )
            self._rows_by_dataset_index[dataset_index] = row
            self._contract_rows_by_dataset_index[dataset_index] = contract_row

    @property
    def action_dim(self):
        return self.dataset.action_dim

    @property
    def action_stats(self):
        return self.dataset.action_stats

    @property
    def action_stats_path(self):
        return self.dataset.action_stats_path

    @property
    def task_names(self):
        return self.dataset.task_names

    @property
    def _sub_datasets(self):
        return self.dataset._sub_datasets

    def __getattr__(self, name: str):
        # Preserve read-only dataset attributes used by checkpoint/config helpers.
        try:
            dataset = object.__getattribute__(self, "dataset")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(dataset, name)

    def __len__(self) -> int:
        # PlanSampler uses the base MultiTask global-index space.
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for dataset_index in self._sampler_guard:
            yield self[dataset_index]

    def __getitem__(self, dataset_index: int) -> dict[str, Any]:
        if isinstance(dataset_index, bool) or not isinstance(dataset_index, int):
            raise PlanValidationError("planned dataset index must be an integer")
        row = self._rows_by_dataset_index.get(dataset_index)
        if row is None:
            raise PlanValidationError(
                f"dataset index {dataset_index} is not present in the frozen plan"
            )
        observed = self.dataset.phase6_identity_at(
            dataset_index,
            protocol_seed=self.plan.protocol_seed,
        )
        if observed != row.identity:
            raise PlanValidationError(
                f"runtime identity drift at dataset index {dataset_index}"
            )
        contract_row = self._contract_rows_by_dataset_index[dataset_index]
        observed_window = runtime_window_row(
            self.dataset,
            global_step=row.global_step,
            dataset_index=dataset_index,
        )
        if any(
            contract_row[key] != value for key, value in observed_window.items()
        ):
            raise PlanValidationError(
                f"runtime logical-window drift at dataset index {dataset_index}"
            )
        sample = self.dataset.phase6_get_planned_item(
            dataset_index,
            planned_row=row,
            protocol_seed=self.plan.protocol_seed,
            plan_sha256=self.plan_sha256,
            identity_sha256=self.identity_sha256,
        )
        if not isinstance(sample, Mapping):
            raise PlanValidationError("planned materializer returned a non-mapping")
        payload = sample.get("phase6_plan_row")
        expected_payload = phase6_plan_row_dict(
            row,
            plan_sha256=self.plan_sha256,
            identity_sha256=self.identity_sha256,
        )
        if payload != expected_payload:
            raise PlanValidationError("materialized phase6_plan_row payload changed")
        return dict(sample)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != 0:
            raise PlanValidationError("the frozen Phase-6 dataset has only epoch 0")
