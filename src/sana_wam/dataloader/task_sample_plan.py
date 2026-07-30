"""Fail-closed, deterministic sample planning for the Phase-6 factorial arms.

This module deliberately has no torch dependency. It builds one immutable plan
from ordinary RoboTwin clean_50 windows and makes every arm consume the same
global dataset indices in the same order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
from pathlib import Path
from typing import Any

from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    expansion_support_contract,
)


PLAN_SCHEMA_VERSION = "sana-phase6-task-plan-v1"
IDENTITY_SCHEMA_VERSION = "sana-phase6-identity-sequence-v1"
ARTIFACT_SCHEMA_VERSION = "sana-phase6-task-plan-artifact-v2"

EXPECTED_TRAIN_TASK_COUNT = 42
EXPECTED_HOLDOUT_TASK_COUNT = 8
EXPECTED_TASK_CYCLES = 12
EXPECTED_OPTIMIZER_STEPS = 504
EXPECTED_SIGMAS = (1.0, 0.9, 0.5)
EXPECTED_SIGMA_EXPOSURES = 4
EXPECTED_TRAIN_TASK_SHA256 = (
    "68293563c113d91e4878844e7cb5ef1b70ad14656da0a7c58a54a6069cc1e67d"
)
EXPECTED_HOLDOUT_TASK_SHA256 = (
    "689f9d1509e30fc446b995edf72d4d122a78b4f0f708efece2d63791c5362f6e"
)
EXPECTED_REGISTRY_SOURCE = "RoboTwin clean_50 ordinary expert windows only"

SOURCE_DATASET = "RoboTwin"
SOURCE_VARIANT = "clean_50"
SOURCE_KIND = "ordinary_expert"

ROW_SEED_DOMAINS = (
    "video-noise",
    "action-noise",
    "expansion-noise",
    "expansion-direction",
    "reference-query",
    "prompt-choice",
)
INTERNAL_SEED_DOMAINS = (
    "task-order",
    "sample-choice",
    "action-sigma",
)
ALL_SEED_DOMAINS = INTERNAL_SEED_DOMAINS + ROW_SEED_DOMAINS

FACTORIAL_ARMS = (
    "T0_E0A0",
    "T1_E0A0",
    "T1_E1A0",
    "T1_E0A1",
    "T1_E1A1",
)


class PlanValidationError(ValueError):
    """Raised when a registry, source pool, plan, or artifact violates contract."""


def _is_plain_int(value: Any) -> bool:
    return type(value) is int


def _require_plain_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if not _is_plain_int(value) or value < minimum:
        raise PlanValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PlanValidationError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PlanValidationError(
            f"{name} keys differ; missing={missing!r}, extra={extra!r}"
        )


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PlanValidationError(f"value is not canonical-JSON serializable: {exc}") from exc
    return encoded + b"\n"


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlanValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(data: bytes | str, name: str) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PlanValidationError(f"non-finite JSON value: {value}")
            ),
        )
    except PlanValidationError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError) as exc:
        raise PlanValidationError(f"invalid {name} JSON: {exc}") from exc


def _task_list_sha256(tasks: Iterable[str]) -> str:
    serialized = "".join(f"{task}\n" for task in sorted(tasks)).encode("utf-8")
    return sha256(serialized).hexdigest()


def _framed_field(value: str | int | bytes) -> bytes:
    if isinstance(value, str):
        payload = b"s" + value.encode("utf-8")
    elif _is_plain_int(value):
        payload = b"i" + str(value).encode("ascii")
    elif isinstance(value, bytes):
        payload = b"b" + value
    else:
        raise TypeError(f"unsupported seed field type: {type(value).__name__}")
    return len(payload).to_bytes(8, "big") + payload


def _seed_digest(protocol_seed: int, domain: str, *fields: str | int | bytes) -> bytes:
    _require_plain_int(protocol_seed, "protocol_seed")
    if not isinstance(domain, str) or not domain:
        raise PlanValidationError("seed domain must be a non-empty string")
    material = bytearray()
    material.extend(_framed_field("sana-phase6-stateless-seed-v1"))
    material.extend(_framed_field(protocol_seed))
    material.extend(_framed_field(domain))
    for field in fields:
        material.extend(_framed_field(field))
    return sha256(material).digest()


def derive_seed(protocol_seed: int, domain: str, *fields: str | int | bytes) -> int:
    """Return a stable unsigned 64-bit seed from length-prefixed fields."""

    return int.from_bytes(_seed_digest(protocol_seed, domain, *fields)[:8], "big")


def derive_prompt_choice_seed(
    protocol_seed: int,
    task_name: str,
    episode_index: int,
    episode_path: str,
    start_frame: int,
    dataset_index: int,
) -> int:
    """Derive the prompt selector before a complete plan row exists."""

    return derive_seed(
        protocol_seed,
        "prompt-choice",
        task_name,
        episode_index,
        episode_path,
        start_frame,
        dataset_index,
    )


def _hash_order(
    values: Iterable[Any],
    protocol_seed: int,
    domain: str,
    common_fields: tuple[str | int | bytes, ...],
    value_bytes: Any,
) -> tuple[Any, ...]:
    decorated = []
    for value in values:
        stable_value = value_bytes(value)
        if not isinstance(stable_value, bytes):
            raise TypeError("value_bytes must return bytes")
        digest = _seed_digest(protocol_seed, domain, *common_fields, stable_value)
        decorated.append((digest, stable_value, value))
    decorated.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in decorated)


@dataclass(frozen=True)
class SampleIdentity:
    """Exact dataset identity plus the global index needed by a sampler."""

    task_name: str
    episode_index: int
    episode_path: str
    start_frame: int
    prompt: str
    dataset_index: int
    source_dataset: str = SOURCE_DATASET
    source_variant: str = SOURCE_VARIANT
    source_kind: str = SOURCE_KIND

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        dataset_index: int,
        source_dataset: str = SOURCE_DATASET,
        source_variant: str = SOURCE_VARIANT,
        source_kind: str = SOURCE_KIND,
    ) -> "SampleIdentity":
        """Copy audit fields exactly; no task or prompt normalization is applied."""

        required = (
            "task_name",
            "episode_index",
            "episode_path",
            "start_frame",
            "prompt",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise PlanValidationError(f"sample metadata is missing fields: {missing!r}")
        identity = cls(
            task_name=metadata["task_name"],
            episode_index=metadata["episode_index"],
            episode_path=metadata["episode_path"],
            start_frame=metadata["start_frame"],
            prompt=metadata["prompt"],
            dataset_index=dataset_index,
            source_dataset=source_dataset,
            source_variant=source_variant,
            source_kind=source_kind,
        )
        identity.validate()
        return identity

    @property
    def audit_tuple(self) -> tuple[str, str, int, str]:
        return (self.task_name, self.episode_path, self.start_frame, self.prompt)

    def validate(self, *, expected_task: str | None = None) -> None:
        for name, value in (
            ("task_name", self.task_name),
            ("episode_path", self.episode_path),
            ("prompt", self.prompt),
        ):
            if not isinstance(value, str) or not value:
                raise PlanValidationError(f"{name} must be a non-empty exact string")
        _require_plain_int(self.episode_index, "episode_index")
        _require_plain_int(self.start_frame, "start_frame")
        _require_plain_int(self.dataset_index, "dataset_index")
        if expected_task is not None and self.task_name != expected_task:
            raise PlanValidationError(
                f"pool task {expected_task!r} contains sample for {self.task_name!r}"
            )
        observed_source = (
            self.source_dataset,
            self.source_variant,
            self.source_kind,
        )
        expected_source = (SOURCE_DATASET, SOURCE_VARIANT, SOURCE_KIND)
        if observed_source != expected_source:
            raise PlanValidationError(
                "only RoboTwin clean_50 ordinary_expert samples are allowed; "
                f"observed={observed_source!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_index": self.dataset_index,
            "episode_index": self.episode_index,
            "episode_path": self.episode_path,
            "prompt": self.prompt,
            "source_dataset": self.source_dataset,
            "source_kind": self.source_kind,
            "source_variant": self.source_variant,
            "start_frame": self.start_frame,
            "task_name": self.task_name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SampleIdentity":
        if not isinstance(value, Mapping):
            raise PlanValidationError("identity must be a JSON object")
        expected = {
            "dataset_index",
            "episode_index",
            "episode_path",
            "prompt",
            "source_dataset",
            "source_kind",
            "source_variant",
            "start_frame",
            "task_name",
        }
        _require_exact_keys(value, expected, "identity")
        identity = cls(**{key: value[key] for key in expected})
        identity.validate()
        return identity

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())


@dataclass(frozen=True)
class PlannedSample:
    global_step: int
    cycle: int
    position_in_cycle: int
    identity: SampleIdentity
    action_sigma: float
    domain_seeds: tuple[tuple[str, int], ...]

    def seed_for(self, domain: str) -> int:
        for observed_domain, seed in self.domain_seeds:
            if observed_domain == domain:
                return seed
        raise KeyError(domain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_sigma": self.action_sigma,
            "cycle": self.cycle,
            "domain_seeds": dict(self.domain_seeds),
            "global_step": self.global_step,
            "identity": self.identity.to_dict(),
            "position_in_cycle": self.position_in_cycle,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannedSample":
        if not isinstance(value, Mapping):
            raise PlanValidationError("planned row must be a JSON object")
        expected = {
            "action_sigma",
            "cycle",
            "domain_seeds",
            "global_step",
            "identity",
            "position_in_cycle",
        }
        _require_exact_keys(value, expected, "planned row")
        seeds = value["domain_seeds"]
        if not isinstance(seeds, Mapping):
            raise PlanValidationError("domain_seeds must be a JSON object")
        if set(seeds) != set(ROW_SEED_DOMAINS):
            raise PlanValidationError("row seed domains do not match the frozen contract")
        ordered_seeds = tuple((domain, seeds[domain]) for domain in ROW_SEED_DOMAINS)
        return cls(
            global_step=value["global_step"],
            cycle=value["cycle"],
            position_in_cycle=value["position_in_cycle"],
            identity=SampleIdentity.from_dict(value["identity"]),
            action_sigma=value["action_sigma"],
            domain_seeds=ordered_seeds,
        )


@dataclass(frozen=True)
class SamplerContract:
    """Validated sampler fields extracted from the frozen Phase-6 registry."""

    protocol_seed: int
    train_tasks: tuple[str, ...]
    holdout_tasks: tuple[str, ...]
    train_task_sha256: str
    holdout_task_sha256: str
    task_cycles: int
    optimizer_steps: int
    action_sigmas: tuple[float, ...]
    sigma_exposures_per_task: int

    @classmethod
    def from_registry(cls, registry: str | Path | Mapping[str, Any]) -> "SamplerContract":
        if isinstance(registry, (str, Path)):
            path = Path(registry)
            value = _strict_json_loads(path.read_bytes(), f"registry {path}")
        else:
            value = registry
        if not isinstance(value, Mapping):
            raise PlanValidationError("registry must be a JSON object")
        try:
            data = value["data"]
            constraints = value["constraints"]
        except KeyError as exc:
            raise PlanValidationError(f"registry is missing {exc.args[0]!r}") from exc
        if not isinstance(data, Mapping) or not isinstance(constraints, Mapping):
            raise PlanValidationError("registry data and constraints must be objects")
        try:
            exposure_map = data["action_sigma_exposures_per_task"]
            raw_sigmas = data["action_sigma_plan"]
            raw_train_tasks = data["train_tasks"]
            raw_holdout_tasks = data["holdout_tasks"]
        except (KeyError, TypeError) as exc:
            raise PlanValidationError(f"registry sampler fields are malformed: {exc}") from exc
        if not isinstance(raw_sigmas, list):
            raise PlanValidationError("registry action_sigma_plan must be a JSON list")
        if not isinstance(raw_train_tasks, list) or not isinstance(raw_holdout_tasks, list):
            raise PlanValidationError("registry task cohorts must be JSON lists")
        try:
            sigma_values = tuple(raw_sigmas)
            train_tasks = tuple(sorted(raw_train_tasks))
            holdout_tasks = tuple(sorted(raw_holdout_tasks))
        except TypeError as exc:
            raise PlanValidationError(f"registry task names are malformed: {exc}") from exc
        if not isinstance(exposure_map, Mapping):
            raise PlanValidationError("sigma exposure contract must be an object")
        exposure_values = set(exposure_map.values())
        if len(exposure_values) != 1:
            raise PlanValidationError("all action sigmas must have the same exposure count")
        exposure_count = next(iter(exposure_values), None)
        contract = cls(
            protocol_seed=value.get("seed"),
            train_tasks=train_tasks,
            holdout_tasks=holdout_tasks,
            train_task_sha256=data.get("train_task_sha256"),
            holdout_task_sha256=data.get("holdout_task_sha256"),
            task_cycles=data.get("task_cycles"),
            optimizer_steps=data.get("optimizer_steps"),
            action_sigmas=sigma_values,
            sigma_exposures_per_task=exposure_count,
        )
        contract.validate()
        if data.get("source") != EXPECTED_REGISTRY_SOURCE:
            raise PlanValidationError("registry data source is not the frozen clean_50 source")
        if data.get("train_task_count") != EXPECTED_TRAIN_TASK_COUNT:
            raise PlanValidationError("registry train_task_count is inconsistent")
        if data.get("holdout_task_count") != EXPECTED_HOLDOUT_TASK_COUNT:
            raise PlanValidationError("registry holdout_task_count is inconsistent")
        if data.get("exposures_per_train_task") != EXPECTED_TASK_CYCLES:
            raise PlanValidationError("registry per-task exposure count is inconsistent")
        if (
            type(data.get("batch_size")) is not int
            or data.get("batch_size") != 1
            or type(data.get("gradient_accumulation_steps")) is not int
            or data.get("gradient_accumulation_steps") != 1
        ):
            raise PlanValidationError("the 504-row plan requires batch size and accumulation of one")
        expected_exposure_keys = {str(sigma) for sigma in EXPECTED_SIGMAS}
        if set(exposure_map) != expected_exposure_keys:
            raise PlanValidationError("registry sigma exposure keys are inconsistent")
        for key in ("recovery_annotations", "dagger"):
            if constraints.get(key) is not False:
                raise PlanValidationError(f"registry constraint {key!r} must be false")
        return contract

    def validate(self) -> None:
        _require_plain_int(self.protocol_seed, "protocol_seed")
        _require_plain_int(self.task_cycles, "task_cycles", minimum=1)
        _require_plain_int(self.optimizer_steps, "optimizer_steps", minimum=1)
        _require_plain_int(
            self.sigma_exposures_per_task,
            "sigma_exposures_per_task",
            minimum=1,
        )
        if type(self.train_tasks) is not tuple or type(self.holdout_tasks) is not tuple:
            raise PlanValidationError("task cohorts must use immutable tuples")
        if any(not isinstance(task, str) or not task for task in self.train_tasks):
            raise PlanValidationError("training task names must be non-empty strings")
        if any(not isinstance(task, str) or not task for task in self.holdout_tasks):
            raise PlanValidationError("holdout task names must be non-empty strings")
        if self.train_tasks != tuple(sorted(self.train_tasks)):
            raise PlanValidationError("training task tuple must be lexicographically canonical")
        if self.holdout_tasks != tuple(sorted(self.holdout_tasks)):
            raise PlanValidationError("holdout task tuple must be lexicographically canonical")
        if type(self.action_sigmas) is not tuple:
            raise PlanValidationError("action sigma plan must use an immutable tuple")
        if len(self.train_tasks) != EXPECTED_TRAIN_TASK_COUNT:
            raise PlanValidationError("training cohort must contain exactly 42 task names")
        if len(set(self.train_tasks)) != len(self.train_tasks):
            raise PlanValidationError("training cohort contains duplicate task names")
        if len(self.holdout_tasks) != EXPECTED_HOLDOUT_TASK_COUNT:
            raise PlanValidationError("holdout cohort must contain exactly eight task names")
        if len(set(self.holdout_tasks)) != len(self.holdout_tasks):
            raise PlanValidationError("holdout cohort contains duplicate task names")
        if set(self.train_tasks) & set(self.holdout_tasks):
            raise PlanValidationError("training and holdout cohorts overlap")
        computed_train_hash = _task_list_sha256(self.train_tasks)
        computed_holdout_hash = _task_list_sha256(self.holdout_tasks)
        _require_sha256(self.train_task_sha256, "train_task_sha256")
        _require_sha256(self.holdout_task_sha256, "holdout_task_sha256")
        if computed_train_hash != self.train_task_sha256:
            raise PlanValidationError("training task list does not match its declared hash")
        if computed_holdout_hash != self.holdout_task_sha256:
            raise PlanValidationError("holdout task list does not match its declared hash")
        if self.train_task_sha256 != EXPECTED_TRAIN_TASK_SHA256:
            raise PlanValidationError("training cohort is not the frozen Phase-6 cohort")
        if self.holdout_task_sha256 != EXPECTED_HOLDOUT_TASK_SHA256:
            raise PlanValidationError("holdout cohort is not the frozen Phase-4 cohort")
        if self.task_cycles != EXPECTED_TASK_CYCLES:
            raise PlanValidationError("task_cycles must equal 12")
        if self.optimizer_steps != EXPECTED_OPTIMIZER_STEPS:
            raise PlanValidationError("optimizer_steps must equal 504")
        if len(self.action_sigmas) != len(EXPECTED_SIGMAS) or any(
            type(observed) is not float or observed != expected
            for observed, expected in zip(self.action_sigmas, EXPECTED_SIGMAS)
        ):
            raise PlanValidationError("action sigma plan must be exactly (1.0, 0.9, 0.5)")
        if self.sigma_exposures_per_task != EXPECTED_SIGMA_EXPOSURES:
            raise PlanValidationError("each action sigma must occur four times per task")
        if len(self.action_sigmas) * self.sigma_exposures_per_task != self.task_cycles:
            raise PlanValidationError("sigma exposure counts do not fill all task cycles")
        if len(self.train_tasks) * self.task_cycles != self.optimizer_steps:
            raise PlanValidationError("task count and cycles do not produce 504 rows")


def _task_order(contract: SamplerContract, cycle: int) -> tuple[str, ...]:
    return _hash_order(
        contract.train_tasks,
        contract.protocol_seed,
        "task-order",
        (cycle,),
        lambda task: task.encode("utf-8"),
    )


def _sigma_schedule(contract: SamplerContract, task: str) -> dict[int, float]:
    cycle_order = _hash_order(
        range(contract.task_cycles),
        contract.protocol_seed,
        "action-sigma",
        (task,),
        lambda cycle: str(cycle).encode("ascii"),
    )
    return {
        cycle: contract.action_sigmas[rank % len(contract.action_sigmas)]
        for rank, cycle in enumerate(cycle_order)
    }


def _row_seed_fields(
    protocol_seed: int,
    global_step: int,
    cycle: int,
    position_in_cycle: int,
    identity: SampleIdentity,
    action_sigma: float,
) -> tuple[tuple[str, int], ...]:
    # Prompt selection happens before the complete PlannedSample exists. Keep
    # its seed on the exact metadata-only formula used by RoboTwinDataset so the
    # seed recorded in the row is the seed that selected the row's prompt.
    sigma_token = format(action_sigma, ".17g")
    fields: tuple[str | int | bytes, ...] = (
        PLAN_SCHEMA_VERSION,
        global_step,
        cycle,
        position_in_cycle,
        identity.task_name,
        identity.episode_index,
        identity.episode_path,
        identity.start_frame,
        identity.prompt,
        identity.dataset_index,
        sigma_token,
    )
    return tuple(
        (
            domain,
            (
                derive_prompt_choice_seed(
                    protocol_seed,
                    identity.task_name,
                    identity.episode_index,
                    identity.episode_path,
                    identity.start_frame,
                    identity.dataset_index,
                )
                if domain == "prompt-choice"
                else derive_seed(protocol_seed, domain, *fields)
            ),
        )
        for domain in ROW_SEED_DOMAINS
    )


@dataclass(frozen=True)
class TaskRoundRobinPlan:
    protocol_seed: int
    train_tasks: tuple[str, ...]
    holdout_tasks: tuple[str, ...]
    train_task_sha256: str
    holdout_task_sha256: str
    task_cycles: int
    action_sigmas: tuple[float, ...]
    sigma_exposures_per_task: int
    rows: tuple[PlannedSample, ...]

    @classmethod
    def build(
        cls,
        contract: SamplerContract,
        candidate_pools: Mapping[str, Sequence[SampleIdentity]],
    ) -> "TaskRoundRobinPlan":
        contract.validate()
        if not isinstance(candidate_pools, Mapping):
            raise PlanValidationError("candidate_pools must map task names to samples")
        pool_tasks = set(candidate_pools)
        expected_tasks = set(contract.train_tasks)
        if pool_tasks != expected_tasks:
            missing = sorted(expected_tasks - pool_tasks)
            extra = sorted(pool_tasks - expected_tasks)
            raise PlanValidationError(
                f"candidate task pools differ; missing={missing!r}, extra={extra!r}"
            )

        selected_by_task: dict[str, tuple[SampleIdentity, ...]] = {}
        all_dataset_indices: set[int] = set()
        for task in contract.train_tasks:
            pool = tuple(candidate_pools[task])
            if len(pool) < contract.task_cycles:
                raise PlanValidationError(
                    f"task {task!r} has {len(pool)} candidates; 12 unique windows required"
                )
            audit_identities: set[tuple[str, str, int, str]] = set()
            for identity in pool:
                if not isinstance(identity, SampleIdentity):
                    raise PlanValidationError(f"task {task!r} pool contains a non-identity")
                identity.validate(expected_task=task)
                if identity.audit_tuple in audit_identities:
                    raise PlanValidationError(
                        f"task {task!r} contains duplicate audit identity {identity.audit_tuple!r}"
                    )
                audit_identities.add(identity.audit_tuple)
                if identity.dataset_index in all_dataset_indices:
                    raise PlanValidationError(
                        f"global dataset_index {identity.dataset_index} is duplicated"
                    )
                all_dataset_indices.add(identity.dataset_index)
            ordered_pool = _hash_order(
                pool,
                contract.protocol_seed,
                "sample-choice",
                (task,),
                lambda identity: identity.canonical_bytes(),
            )
            selected_by_task[task] = ordered_pool[: contract.task_cycles]

        sigma_by_task = {
            task: _sigma_schedule(contract, task) for task in contract.train_tasks
        }
        rows: list[PlannedSample] = []
        for cycle in range(contract.task_cycles):
            for position, task in enumerate(_task_order(contract, cycle)):
                identity = selected_by_task[task][cycle]
                action_sigma = sigma_by_task[task][cycle]
                global_step = len(rows) + 1
                rows.append(
                    PlannedSample(
                        global_step=global_step,
                        cycle=cycle,
                        position_in_cycle=position,
                        identity=identity,
                        action_sigma=action_sigma,
                        domain_seeds=_row_seed_fields(
                            contract.protocol_seed,
                            global_step,
                            cycle,
                            position,
                            identity,
                            action_sigma,
                        ),
                    )
                )

        plan = cls(
            protocol_seed=contract.protocol_seed,
            train_tasks=contract.train_tasks,
            holdout_tasks=contract.holdout_tasks,
            train_task_sha256=contract.train_task_sha256,
            holdout_task_sha256=contract.holdout_task_sha256,
            task_cycles=contract.task_cycles,
            action_sigmas=contract.action_sigmas,
            sigma_exposures_per_task=contract.sigma_exposures_per_task,
            rows=tuple(rows),
        )
        plan.validate()
        return plan

    @property
    def optimizer_steps(self) -> int:
        return len(self.rows)

    @property
    def identity_sequence(self) -> tuple[tuple[str, str, int, str], ...]:
        return tuple(row.identity.audit_tuple for row in self.rows)

    def _as_contract(self) -> SamplerContract:
        return SamplerContract(
            protocol_seed=self.protocol_seed,
            train_tasks=self.train_tasks,
            holdout_tasks=self.holdout_tasks,
            train_task_sha256=self.train_task_sha256,
            holdout_task_sha256=self.holdout_task_sha256,
            task_cycles=self.task_cycles,
            optimizer_steps=EXPECTED_OPTIMIZER_STEPS,
            action_sigmas=self.action_sigmas,
            sigma_exposures_per_task=self.sigma_exposures_per_task,
        )

    def validate(self) -> None:
        contract = self._as_contract()
        contract.validate()
        if type(self.rows) is not tuple:
            raise PlanValidationError("plan rows must use an immutable tuple")
        if len(self.rows) != EXPECTED_OPTIMIZER_STEPS:
            raise PlanValidationError(
                f"plan has {len(self.rows)} rows; expected {EXPECTED_OPTIMIZER_STEPS}"
            )
        task_counts: Counter[str] = Counter()
        sigma_counts: dict[str, Counter[float]] = {
            task: Counter() for task in self.train_tasks
        }
        audit_identities: set[tuple[str, str, int, str]] = set()
        dataset_indices: set[int] = set()
        holdout = set(self.holdout_tasks)

        for cycle in range(self.task_cycles):
            start = cycle * EXPECTED_TRAIN_TASK_COUNT
            cycle_rows = self.rows[start : start + EXPECTED_TRAIN_TASK_COUNT]
            expected_order = _task_order(contract, cycle)
            observed_order = tuple(row.identity.task_name for row in cycle_rows)
            if observed_order != expected_order:
                raise PlanValidationError(f"cycle {cycle} task order is not deterministic")
            if set(observed_order) != set(self.train_tasks):
                raise PlanValidationError(f"cycle {cycle} is not one occurrence per task")
            for position, row in enumerate(cycle_rows):
                expected_global_step = start + position + 1
                if not isinstance(row, PlannedSample):
                    raise PlanValidationError("plan rows must be PlannedSample values")
                _require_plain_int(row.global_step, "row global_step", minimum=1)
                _require_plain_int(row.cycle, "row cycle")
                _require_plain_int(row.position_in_cycle, "row position_in_cycle")
                if row.global_step != expected_global_step:
                    raise PlanValidationError("global steps must be contiguous and one-based")
                if row.cycle != cycle or row.position_in_cycle != position:
                    raise PlanValidationError("row cycle or position is inconsistent")
                row.identity.validate(expected_task=expected_order[position])
                if row.identity.task_name in holdout:
                    raise PlanValidationError("held-out task appears in training plan")
                if type(row.action_sigma) is not float:
                    raise PlanValidationError("action_sigma must retain canonical float type")
                expected_sigma = _sigma_schedule(
                    contract, row.identity.task_name
                )[cycle]
                if row.action_sigma != expected_sigma:
                    raise PlanValidationError("action sigma assignment is not deterministic")
                if row.identity.audit_tuple in audit_identities:
                    raise PlanValidationError("sample audit identity is reused")
                if row.identity.dataset_index in dataset_indices:
                    raise PlanValidationError("dataset index is reused")
                audit_identities.add(row.identity.audit_tuple)
                dataset_indices.add(row.identity.dataset_index)
                if tuple(domain for domain, _ in row.domain_seeds) != ROW_SEED_DOMAINS:
                    raise PlanValidationError("row seed domains or ordering changed")
                if type(row.domain_seeds) is not tuple:
                    raise PlanValidationError("row domain seeds must use an immutable tuple")
                for domain, seed in row.domain_seeds:
                    if not _is_plain_int(seed) or not 0 <= seed < 2**64:
                        raise PlanValidationError(f"seed {domain!r} is not uint64")
                expected_seeds = _row_seed_fields(
                    self.protocol_seed,
                    row.global_step,
                    row.cycle,
                    row.position_in_cycle,
                    row.identity,
                    row.action_sigma,
                )
                if row.domain_seeds != expected_seeds:
                    raise PlanValidationError("row domain seed derivation changed")
                task_counts[row.identity.task_name] += 1
                sigma_counts[row.identity.task_name][row.action_sigma] += 1

        expected_task_counts = {task: self.task_cycles for task in self.train_tasks}
        if dict(task_counts) != expected_task_counts:
            raise PlanValidationError("per-task row counts differ from 12")
        expected_sigma_counts = {
            sigma: self.sigma_exposures_per_task for sigma in self.action_sigmas
        }
        for task in self.train_tasks:
            if dict(sigma_counts[task]) != expected_sigma_counts:
                raise PlanValidationError(
                    f"task {task!r} does not have four exposures per sigma"
                )

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_sigmas": list(self.action_sigmas),
            "holdout_task_sha256": self.holdout_task_sha256,
            "holdout_tasks": list(self.holdout_tasks),
            "optimizer_steps": self.optimizer_steps,
            "protocol_seed": self.protocol_seed,
            "rows": [row.to_dict() for row in self.rows],
            "schema_version": PLAN_SCHEMA_VERSION,
            "seed_domains": list(ALL_SEED_DOMAINS),
            "sigma_exposures_per_task": self.sigma_exposures_per_task,
            "source_contract": {
                "dataset": SOURCE_DATASET,
                "kind": SOURCE_KIND,
                "variant": SOURCE_VARIANT,
            },
            "task_cycles": self.task_cycles,
            "train_task_sha256": self.train_task_sha256,
            "train_tasks": list(self.train_tasks),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TaskRoundRobinPlan":
        if not isinstance(payload, Mapping):
            raise PlanValidationError("plan payload must be a JSON object")
        expected = {
            "action_sigmas",
            "holdout_task_sha256",
            "holdout_tasks",
            "optimizer_steps",
            "protocol_seed",
            "rows",
            "schema_version",
            "seed_domains",
            "sigma_exposures_per_task",
            "source_contract",
            "task_cycles",
            "train_task_sha256",
            "train_tasks",
        }
        _require_exact_keys(payload, expected, "plan payload")
        if payload["schema_version"] != PLAN_SCHEMA_VERSION:
            raise PlanValidationError("unsupported plan schema")
        if payload["seed_domains"] != list(ALL_SEED_DOMAINS):
            raise PlanValidationError("plan seed-domain declaration changed")
        if payload["source_contract"] != {
            "dataset": SOURCE_DATASET,
            "kind": SOURCE_KIND,
            "variant": SOURCE_VARIANT,
        }:
            raise PlanValidationError("plan source contract is not plain clean_50")
        rows_value = payload["rows"]
        if not isinstance(rows_value, list):
            raise PlanValidationError("plan rows must be a JSON list")
        try:
            train_tasks = tuple(payload["train_tasks"])
            holdout_tasks = tuple(payload["holdout_tasks"])
            action_sigmas = tuple(payload["action_sigmas"])
        except TypeError as exc:
            raise PlanValidationError("plan list fields are malformed") from exc
        plan = cls(
            protocol_seed=payload["protocol_seed"],
            train_tasks=train_tasks,
            holdout_tasks=holdout_tasks,
            train_task_sha256=payload["train_task_sha256"],
            holdout_task_sha256=payload["holdout_task_sha256"],
            task_cycles=payload["task_cycles"],
            action_sigmas=action_sigmas,
            sigma_exposures_per_task=payload["sigma_exposures_per_task"],
            rows=tuple(PlannedSample.from_dict(row) for row in rows_value),
        )
        if payload["optimizer_steps"] != len(plan.rows):
            raise PlanValidationError("declared optimizer_steps differs from row count")
        return plan

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_payload())

    @property
    def plan_sha256(self) -> str:
        return sha256(self.canonical_json_bytes()).hexdigest()

    def identity_payload(self) -> dict[str, Any]:
        return {
            "holdout_task_sha256": self.holdout_task_sha256,
            "ordered_identities": [
                {
                    "episode_path": row.identity.episode_path,
                    "prompt": row.identity.prompt,
                    "start_frame": row.identity.start_frame,
                    "task_name": row.identity.task_name,
                }
                for row in self.rows
            ],
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "train_task_sha256": self.train_task_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return sha256(_canonical_json_bytes(self.identity_payload())).hexdigest()

    def canonical_jsonl_bytes(self) -> bytes:
        self.validate()
        header = self.to_payload()
        del header["rows"]
        records: list[dict[str, Any]] = [
            {
                "header": header,
                "identity_sha256": self.identity_sha256,
                "plan_sha256": self.plan_sha256,
                "record_type": "header",
            }
        ]
        records.extend(
            {"record_type": "sample", "row": row.to_dict()} for row in self.rows
        )
        return b"".join(_canonical_json_bytes(record) for record in records)

    @property
    def jsonl_sha256(self) -> str:
        return sha256(self.canonical_jsonl_bytes()).hexdigest()

    def to_artifact_bytes(self) -> bytes:
        self.validate()
        artifact = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "expansion_eligibility_amendment_sha256": (
                EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
            ),
            "expansion_support_contract": expansion_support_contract(),
            "expansion_support_contract_sha256": (
                EXPANSION_SUPPORT_CONTRACT_SHA256
            ),
            "identity_sha256": self.identity_sha256,
            "plan": self.to_payload(),
            "plan_sha256": self.plan_sha256,
        }
        return _canonical_json_bytes(artifact)

    @classmethod
    def from_artifact_bytes(
        cls,
        data: bytes,
        *,
        expected_plan_sha256: str | None = None,
        expected_identity_sha256: str | None = None,
    ) -> "TaskRoundRobinPlan":
        if not isinstance(data, bytes):
            raise PlanValidationError("artifact input must be bytes")
        artifact = _strict_json_loads(data, "plan artifact")
        if not isinstance(artifact, Mapping):
            raise PlanValidationError("plan artifact must be a JSON object")
        expected_keys = {
            "artifact_schema_version",
            "expansion_eligibility_amendment_sha256",
            "expansion_support_contract",
            "expansion_support_contract_sha256",
            "identity_sha256",
            "plan",
            "plan_sha256",
        }
        _require_exact_keys(artifact, expected_keys, "plan artifact")
        if artifact["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
            raise PlanValidationError("unsupported artifact schema")
        support_contract = artifact["expansion_support_contract"]
        try:
            support_contract_sha256 = sha256(
                _canonical_json_bytes(support_contract)
            ).hexdigest()
        except (TypeError, ValueError) as exc:
            raise PlanValidationError(
                "artifact expansion-support contract is not canonical JSON data"
            ) from exc
        if (
            support_contract_sha256 != EXPANSION_SUPPORT_CONTRACT_SHA256
            or support_contract != expansion_support_contract()
            or artifact["expansion_support_contract_sha256"]
            != EXPANSION_SUPPORT_CONTRACT_SHA256
        ):
            raise PlanValidationError("artifact expansion-support contract differs")
        if (
            artifact["expansion_eligibility_amendment_sha256"]
            != EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        ):
            raise PlanValidationError("artifact eligibility-amendment SHA256 differs")
        if data != _canonical_json_bytes(artifact):
            raise PlanValidationError("plan artifact is not canonical JSON")
        _require_sha256(
            artifact["expansion_support_contract_sha256"],
            "artifact expansion_support_contract_sha256",
        )
        _require_sha256(
            artifact["expansion_eligibility_amendment_sha256"],
            "artifact expansion_eligibility_amendment_sha256",
        )
        _require_sha256(artifact["plan_sha256"], "artifact plan_sha256")
        _require_sha256(artifact["identity_sha256"], "artifact identity_sha256")
        payload_sha256 = sha256(_canonical_json_bytes(artifact["plan"])).hexdigest()
        if not hmac.compare_digest(payload_sha256, artifact["plan_sha256"]):
            raise PlanValidationError("embedded plan SHA256 does not match payload")
        plan = cls.from_payload(artifact["plan"])
        plan.validate()
        if not hmac.compare_digest(plan.plan_sha256, artifact["plan_sha256"]):
            raise PlanValidationError("reconstructed plan SHA256 changed")
        if not hmac.compare_digest(plan.identity_sha256, artifact["identity_sha256"]):
            raise PlanValidationError("embedded identity SHA256 does not match plan")
        if expected_plan_sha256 is not None and not hmac.compare_digest(
            plan.plan_sha256,
            _require_sha256(expected_plan_sha256, "expected_plan_sha256"),
        ):
            raise PlanValidationError("plan SHA256 differs from the pinned value")
        if expected_identity_sha256 is not None and not hmac.compare_digest(
            plan.identity_sha256,
            _require_sha256(expected_identity_sha256, "expected_identity_sha256"),
        ):
            raise PlanValidationError("identity SHA256 differs from the pinned value")
        return plan


class PlanSampler:
    """A torch-Sampler-compatible iterable with no epoch-dependent resampling."""

    def __init__(
        self,
        plan: TaskRoundRobinPlan,
        *,
        expected_plan_sha256: str,
        expected_identity_sha256: str,
    ) -> None:
        plan.validate()
        expected_plan_sha256 = _require_sha256(
            expected_plan_sha256, "expected_plan_sha256"
        )
        expected_identity_sha256 = _require_sha256(
            expected_identity_sha256, "expected_identity_sha256"
        )
        if not hmac.compare_digest(plan.plan_sha256, expected_plan_sha256):
            raise PlanValidationError("sampler plan SHA256 differs from the pinned value")
        if not hmac.compare_digest(plan.identity_sha256, expected_identity_sha256):
            raise PlanValidationError("sampler identity SHA256 differs from the pinned value")
        self._plan = plan
        self._indices = tuple(row.identity.dataset_index for row in plan.rows)

    @classmethod
    def from_artifact_bytes(
        cls,
        data: bytes,
        *,
        expected_plan_sha256: str,
        expected_identity_sha256: str,
    ) -> "PlanSampler":
        plan = TaskRoundRobinPlan.from_artifact_bytes(
            data,
            expected_plan_sha256=expected_plan_sha256,
            expected_identity_sha256=expected_identity_sha256,
        )
        return cls(
            plan,
            expected_plan_sha256=expected_plan_sha256,
            expected_identity_sha256=expected_identity_sha256,
        )

    @property
    def plan(self) -> TaskRoundRobinPlan:
        return self._plan

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def set_epoch(self, epoch: int) -> None:
        _require_plain_int(epoch, "epoch")
        if epoch != 0:
            raise PlanValidationError("the frozen 504-row sampler has no later epochs")


def verify_shared_factorial_plans(
    arm_plans: Mapping[str, TaskRoundRobinPlan],
    *,
    expected_plan_sha256: str,
    expected_identity_sha256: str,
) -> None:
    """Fail unless every supplied factorial arm binds the same pinned plan."""

    if not isinstance(arm_plans, Mapping) or len(arm_plans) < 2:
        raise PlanValidationError("at least two arm plans are required for comparison")
    unknown_arms = set(arm_plans) - set(FACTORIAL_ARMS)
    if unknown_arms:
        raise PlanValidationError(f"unknown factorial arms: {sorted(unknown_arms)!r}")
    reference_sequence: tuple[tuple[str, str, int, str], ...] | None = None
    for arm, plan in arm_plans.items():
        if not isinstance(plan, TaskRoundRobinPlan):
            raise PlanValidationError(f"arm {arm!r} did not provide a task plan")
        sampler = PlanSampler(
            plan,
            expected_plan_sha256=expected_plan_sha256,
            expected_identity_sha256=expected_identity_sha256,
        )
        sequence = sampler.plan.identity_sequence
        if reference_sequence is None:
            reference_sequence = sequence
        elif sequence != reference_sequence:
            raise PlanValidationError(f"arm {arm!r} has a different identity sequence")
