"""CPU-only contract scaffold for the CACH-A4 AV-2 tiny-data screen.

This module intentionally has no filesystem dataset adapter, CUDA/vendor
operator, optimizer, checkpoint, or run-root capability.  It validates the
future AV-2 contract with already-materialized synthetic tensors.  A separate
execution revision and user authority are required before real data is read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Final

import torch
from torch import Tensor


AV2_ARCHITECTURE_ID: Final = (
    "CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1"
)
AV2_CONFIG_SCHEMA: Final = (
    "cach.cach_a4.av2_tiny_real_data_overfit.scaffold_config.v1"
)
AV2_CONFIG_STATE: Final = "SCAFFOLD_ONLY_AV2_EXECUTION_NOT_AUTHORIZED"
AV2_OPERATOR_CLASS: Final = "VENDOR_KERNEL"
AV2_INTEGRATION_PATH: Final = "EXPERIMENTAL_PATH"
AV2_SHUFFLE_PERMUTATION: Final = (1, 0, 3, 2, 5, 4, 7, 6)

_AUTHORITY_KEYS: Final = frozenset(
    {
        "av2_execution",
        "execution_statement_sha256",
        "immutable_root_creation",
        "optimizer_updates",
        "real_data_read",
        "single_gpu_vendor",
    }
)
_DATA_KEYS: Final = frozenset(
    {
        "action_dim",
        "action_transitions",
        "batch_size",
        "heldout_window_count",
        "latent_channels",
        "latent_frames",
        "raw_rows",
        "selector_state",
        "train_window_count",
    }
)
_TRAINING_KEYS: Final = frozenset(
    {
        "checkpoint_load",
        "checkpoint_save",
        "final_screen_state_save_authorized",
        "fresh_initialization",
        "max_steps_per_arm",
        "max_wall_clock_minutes",
        "optimizer",
        "seed",
        "serial_arms",
    }
)
_THRESHOLD_KEYS: Final = frozenset(
    {
        "candidate_heldout_decrease_min",
        "candidate_vs_reference_noninferiority_ratio_max",
        "correct_vs_no_action_improvement_min",
        "correct_vs_shuffle_improvement_min",
        "min_action_variance",
        "min_motion_mse",
        "min_shuffle_mse",
    }
)
_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "architecture_id",
        "authority",
        "data_contract",
        "forbidden",
        "integration_path",
        "operator_class",
        "schema",
        "state",
        "thresholds",
        "training_contract",
    }
)
_FORBIDDEN: Final = (
    "REAL_DATA_ACCESS_IN_SCAFFOLD",
    "DATA_SELECTION_OR_RUN_ROOT_CREATION",
    "GPU_CUDA_TRITON_OR_VENDOR_EXECUTION",
    "OPTIMIZER_OR_PARAMETER_UPDATE",
    "TRAINING_OR_EVALUATION",
    "CHECKPOINT_LOAD_OR_SAVE",
    "FULL_2B",
    "FORMAL_ADMISSION",
    "AV3",
    "GLOBAL_STAGE3",
    "DEPLOY",
    "TOKEN_OR_CLAIM_OPERATION",
    "AUTOMATIC_RERUN",
    "FROZEN_PREDECESSOR_MUTATION",
)
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class AV2ContractError(ValueError):
    """The static AV-2 scaffold contract is malformed."""


class AV2AuthorityBlocked(PermissionError):
    """AV-2 execution capability was requested without complete authority."""

    code: Final = "AUTH_BLOCKED"


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise AV2ContractError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise AV2ContractError(f"{name} must be an exact bool")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AV2ContractError(f"{name} must be an int >= {minimum}")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise AV2ContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AV2ContractError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class AV2AuthorityGate:
    av2_execution: bool
    real_data_read: bool
    single_gpu_vendor: bool
    optimizer_updates: bool
    immutable_root_creation: bool
    execution_statement_sha256: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AV2AuthorityGate":
        if not isinstance(value, Mapping):
            raise AV2ContractError("authority must be a mapping")
        _exact_keys(value, _AUTHORITY_KEYS, "authority")
        statement = value["execution_statement_sha256"]
        if statement is not None and (
            type(statement) is not str or _SHA256_RE.fullmatch(statement) is None
        ):
            raise AV2ContractError(
                "authority.execution_statement_sha256 must be null or lowercase SHA256"
            )
        return cls(
            av2_execution=_exact_bool(
                value["av2_execution"], name="authority.av2_execution"
            ),
            real_data_read=_exact_bool(
                value["real_data_read"], name="authority.real_data_read"
            ),
            single_gpu_vendor=_exact_bool(
                value["single_gpu_vendor"], name="authority.single_gpu_vendor"
            ),
            optimizer_updates=_exact_bool(
                value["optimizer_updates"], name="authority.optimizer_updates"
            ),
            immutable_root_creation=_exact_bool(
                value["immutable_root_creation"],
                name="authority.immutable_root_creation",
            ),
            execution_statement_sha256=statement,
        )

    @property
    def complete(self) -> bool:
        return (
            self.av2_execution
            and self.real_data_read
            and self.single_gpu_vendor
            and self.optimizer_updates
            and self.immutable_root_creation
            and self.execution_statement_sha256 is not None
        )

    def require_execution_authority(self) -> None:
        if not self.complete:
            raise AV2AuthorityBlocked(
                "AV-2 execution requires a successor execution card and fresh "
                "real-data/GPU/update/root authority"
            )


@dataclass(frozen=True)
class AV2StaticConfig:
    schema: str
    state: str
    architecture_id: str
    operator_class: str
    integration_path: str
    data_contract: Mapping[str, object]
    training_contract: Mapping[str, object]
    thresholds: Mapping[str, float]
    authority: AV2AuthorityGate
    forbidden: tuple[str, ...]


def validate_av2_static_config(config: Mapping[str, object]) -> AV2StaticConfig:
    """Validate the scaffold JSON without touching data, CUDA, or the filesystem."""

    if not isinstance(config, Mapping):
        raise AV2ContractError("config must be a mapping")
    _exact_keys(config, _TOP_LEVEL_KEYS, "config")
    exact_literals = {
        "schema": AV2_CONFIG_SCHEMA,
        "state": AV2_CONFIG_STATE,
        "architecture_id": AV2_ARCHITECTURE_ID,
        "operator_class": AV2_OPERATOR_CLASS,
        "integration_path": AV2_INTEGRATION_PATH,
    }
    for name, expected in exact_literals.items():
        if config[name] != expected:
            raise AV2ContractError(f"{name} is frozen to {expected!r}")

    data = config["data_contract"]
    if not isinstance(data, Mapping):
        raise AV2ContractError("data_contract must be a mapping")
    _exact_keys(data, _DATA_KEYS, "data_contract")
    expected_data = {
        "action_dim": 20,
        "action_transitions": 32,
        "batch_size": 8,
        "heldout_window_count": 8,
        "latent_channels": 3,
        "latent_frames": 5,
        "raw_rows": 33,
        "selector_state": "UNBOUND_NO_DATA_ACCESS",
        "train_window_count": 8,
    }
    for name, expected in expected_data.items():
        if data[name] != expected or type(data[name]) is not type(expected):
            raise AV2ContractError(f"data_contract.{name} is frozen to {expected!r}")

    training = config["training_contract"]
    if not isinstance(training, Mapping):
        raise AV2ContractError("training_contract must be a mapping")
    _exact_keys(training, _TRAINING_KEYS, "training_contract")
    expected_training = {
        "checkpoint_load": False,
        "checkpoint_save": False,
        "final_screen_state_save_authorized": False,
        "fresh_initialization": True,
        "max_steps_per_arm": 1000,
        "max_wall_clock_minutes": 60,
        "optimizer": "AdamW",
        "seed": 2026080601,
        "serial_arms": True,
    }
    for name, expected in expected_training.items():
        if training[name] != expected or type(training[name]) is not type(expected):
            raise AV2ContractError(
                f"training_contract.{name} is frozen to {expected!r}"
            )

    threshold_values = config["thresholds"]
    if not isinstance(threshold_values, Mapping):
        raise AV2ContractError("thresholds must be a mapping")
    _exact_keys(threshold_values, _THRESHOLD_KEYS, "thresholds")
    thresholds = {
        name: _finite_number(value, name=f"thresholds.{name}")
        for name, value in threshold_values.items()
    }
    if any(thresholds[name] <= 0.0 for name in (
        "min_action_variance", "min_motion_mse", "min_shuffle_mse"
    )):
        raise AV2ContractError("data-adequacy minima must be positive")
    if any(
        not 0.0 < thresholds[name] <= 1.0
        for name in (
            "candidate_heldout_decrease_min",
            "correct_vs_no_action_improvement_min",
            "correct_vs_shuffle_improvement_min",
            "candidate_vs_reference_noninferiority_ratio_max",
        )
    ):
        raise AV2ContractError("AV-2 decision thresholds must lie in (0,1]")

    forbidden = config["forbidden"]
    if type(forbidden) is not list or tuple(forbidden) != _FORBIDDEN:
        raise AV2ContractError("forbidden capability list differs from scaffold")
    authority = AV2AuthorityGate.from_mapping(config["authority"])  # type: ignore[arg-type]
    if authority.complete:
        raise AV2ContractError("scaffold config must not carry execution authority")
    return AV2StaticConfig(
        schema=AV2_CONFIG_SCHEMA,
        state=AV2_CONFIG_STATE,
        architecture_id=AV2_ARCHITECTURE_ID,
        operator_class=AV2_OPERATOR_CLASS,
        integration_path=AV2_INTEGRATION_PATH,
        data_contract=MappingProxyType(dict(data)),
        training_contract=MappingProxyType(dict(training)),
        thresholds=MappingProxyType(thresholds),
        authority=authority,
        forbidden=_FORBIDDEN,
    )


@dataclass(frozen=True, order=True)
class WindowIdentity:
    task_name: str
    episode_id: str
    raw_start: int
    raw_end_exclusive: int

    def __post_init__(self) -> None:
        if not self.task_name or not self.episode_id:
            raise AV2ContractError("window task_name/episode_id must be non-empty")
        _exact_int(self.raw_start, name="window.raw_start", minimum=0)
        _exact_int(
            self.raw_end_exclusive, name="window.raw_end_exclusive", minimum=1
        )
        if self.raw_end_exclusive - self.raw_start != 33:
            raise AV2ContractError("every AV-2 window must contain exactly 33 raw rows")


def assert_train_holdout_disjoint(
    train: Sequence[WindowIdentity], heldout: Sequence[WindowIdentity]
) -> None:
    if len(train) != 8 or len(heldout) != 8:
        raise AV2ContractError("AV-2 requires exactly 8 train and 8 held-out windows")
    train_set = set(train)
    heldout_set = set(heldout)
    if len(train_set) != 8 or len(heldout_set) != 8:
        raise AV2ContractError("AV-2 split contains duplicate window identity")
    if train_set & heldout_set:
        raise AV2ContractError("AV-2 train and held-out windows overlap")
    for train_window in train_set:
        for heldout_window in heldout_set:
            if (
                train_window.task_name == heldout_window.task_name
                and train_window.episode_id == heldout_window.episode_id
                and train_window.raw_start < heldout_window.raw_end_exclusive
                and heldout_window.raw_start < train_window.raw_end_exclusive
            ):
                raise AV2ContractError(
                    "AV-2 train and held-out raw intervals overlap"
                )


@dataclass(frozen=True)
class DataAdequacyReport:
    action_mask_nonempty: bool
    frame_mask_nonempty: bool
    action_variance: float
    shuffle_mse: float
    motion_mse: float
    finite: bool
    adequate: bool


def _require_tensor(
    value: Tensor, *, name: str, shape: tuple[int, ...], dtype: torch.dtype | None
) -> None:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise AV2ContractError(f"{name} must have shape {shape}")
    if value.device.type != "cpu":
        raise AV2ContractError(f"{name} must stay on CPU in the scaffold")
    if dtype is not None and value.dtype is not dtype:
        raise AV2ContractError(f"{name} must have dtype {dtype}")


def evaluate_data_adequacy(
    *,
    actions: Tensor,
    action_valid_mask: Tensor,
    target: Tensor,
    frame_valid_mask: Tensor,
    shuffled_actions: Tensor,
    min_action_variance: float = 1.0e-8,
    min_shuffle_mse: float = 1.0e-8,
    min_motion_mse: float = 1.0e-8,
) -> DataAdequacyReport:
    """Evaluate only pre-registered data predicates on provided tensors."""

    _require_tensor(actions, name="actions", shape=(8, 32, 20), dtype=None)
    _require_tensor(
        shuffled_actions,
        name="shuffled_actions",
        shape=(8, 32, 20),
        dtype=None,
    )
    _require_tensor(
        action_valid_mask,
        name="action_valid_mask",
        shape=(8, 32),
        dtype=torch.bool,
    )
    _require_tensor(target, name="target", shape=(8, 5, 3, 1, 1), dtype=None)
    _require_tensor(
        frame_valid_mask,
        name="frame_valid_mask",
        shape=(8, 5),
        dtype=torch.bool,
    )
    if actions.dtype != shuffled_actions.dtype or not actions.is_floating_point():
        raise AV2ContractError("action tensors must share a floating dtype")
    if not target.is_floating_point():
        raise AV2ContractError("target must be floating point")
    minima = (
        _finite_number(min_action_variance, name="min_action_variance"),
        _finite_number(min_shuffle_mse, name="min_shuffle_mse"),
        _finite_number(min_motion_mse, name="min_motion_mse"),
    )
    if any(value <= 0.0 for value in minima):
        raise AV2ContractError("adequacy minima must be positive")

    action_mask_nonempty = bool(action_valid_mask.any().item())
    frame_mask_nonempty = bool(frame_valid_mask[:, 1:].any().item())
    active_action_mask = action_valid_mask.unsqueeze(-1).expand_as(actions)
    active_actions = actions.masked_select(active_action_mask)
    active_shuffled = shuffled_actions.masked_select(active_action_mask)
    action_variance = (
        float(active_actions.double().var(unbiased=False).item())
        if active_actions.numel()
        else 0.0
    )
    shuffle_mse = (
        float((active_actions.double() - active_shuffled.double()).square().mean().item())
        if active_actions.numel()
        else 0.0
    )
    future_mask = frame_valid_mask[:, 1:].view(8, 4, 1, 1, 1).expand(8, 4, 3, 1, 1)
    motion = target[:, 1:] - target[:, :1]
    active_motion = motion.masked_select(future_mask)
    motion_mse = (
        float(active_motion.double().square().mean().item())
        if active_motion.numel()
        else 0.0
    )
    finite = all(
        math.isfinite(value) for value in (action_variance, shuffle_mse, motion_mse)
    ) and bool(torch.isfinite(actions).all() and torch.isfinite(target).all())
    adequate = (
        finite
        and action_mask_nonempty
        and frame_mask_nonempty
        and action_variance >= minima[0]
        and shuffle_mse >= minima[1]
        and motion_mse >= minima[2]
    )
    return DataAdequacyReport(
        action_mask_nonempty=action_mask_nonempty,
        frame_mask_nonempty=frame_mask_nonempty,
        action_variance=action_variance,
        shuffle_mse=shuffle_mse,
        motion_mse=motion_mse,
        finite=finite,
        adequate=adequate,
    )


def _masked_future_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    _require_tensor(
        prediction, name="prediction", shape=(8, 5, 3, 1, 1), dtype=None
    )
    if prediction.dtype != target.dtype:
        raise AV2ContractError("prediction and target dtypes differ")
    expanded = mask[:, 1:].view(8, 4, 1, 1, 1).expand(8, 4, 3, 1, 1)
    values = (prediction[:, 1:] - target[:, 1:]).masked_select(expanded)
    if values.numel() == 0:
        raise AV2ContractError("counterfactual metric mask is empty")
    result = float(values.double().square().mean().item())
    if not math.isfinite(result):
        raise AV2ContractError("counterfactual MSE is non-finite")
    return result


def counterfactual_mse_metrics(
    *,
    correct: Tensor,
    shuffled: Tensor,
    no_action: Tensor,
    reference: Tensor,
    target: Tensor,
    frame_valid_mask: Tensor,
) -> Mapping[str, float]:
    """Compute final-step metrics without selecting a checkpoint or window."""

    _require_tensor(target, name="target", shape=(8, 5, 3, 1, 1), dtype=None)
    _require_tensor(
        frame_valid_mask,
        name="frame_valid_mask",
        shape=(8, 5),
        dtype=torch.bool,
    )
    losses = {
        "correct_mse": _masked_future_mse(correct, target, frame_valid_mask),
        "shuffled_mse": _masked_future_mse(shuffled, target, frame_valid_mask),
        "no_action_mse": _masked_future_mse(no_action, target, frame_valid_mask),
        "reference_mse": _masked_future_mse(reference, target, frame_valid_mask),
    }
    epsilon = 1.0e-30
    losses.update(
        {
            "correct_vs_shuffle_improvement": 1.0
            - losses["correct_mse"] / max(losses["shuffled_mse"], epsilon),
            "correct_vs_no_action_improvement": 1.0
            - losses["correct_mse"] / max(losses["no_action_mse"], epsilon),
            "candidate_vs_reference_improvement": 1.0
            - losses["correct_mse"] / max(losses["reference_mse"], epsilon),
        }
    )
    return MappingProxyType(losses)


def _tensor_sha256(value: Tensor) -> str:
    if value.device.type != "cpu":
        raise AV2ContractError("theta0 manifest tensors must stay on CPU")
    detached = value.detach().contiguous()
    payload = detached.view(torch.uint8).numpy().tobytes()
    header = json.dumps(
        {"dtype": str(detached.dtype), "shape": list(detached.shape)},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + payload).hexdigest()


def shared_theta0_manifest(
    *,
    reference_shared: Mapping[str, Tensor],
    candidate_shared: Mapping[str, Tensor],
    candidate_only: Mapping[str, Tensor],
) -> Mapping[str, object]:
    """Prove fresh shared bytes match and candidate-only scope is disjoint."""

    if set(reference_shared) != set(candidate_shared):
        raise AV2ContractError("reference/candidate shared parameter names differ")
    if set(candidate_only) & set(candidate_shared):
        raise AV2ContractError("candidate-only names overlap shared names")
    for scope_name, scope in (
        ("reference_shared", reference_shared),
        ("candidate_shared", candidate_shared),
        ("candidate_only", candidate_only),
    ):
        for name, value in scope.items():
            if not isinstance(value, Tensor):
                raise AV2ContractError("theta0 manifests accept tensors only")
            if value.device.type != "cpu":
                raise AV2ContractError(
                    f"{scope_name}.{name} must stay on CPU in the scaffold"
                )
    shared: dict[str, object] = {}
    for name in sorted(reference_shared, key=lambda item: item.encode("utf-8")):
        left = reference_shared[name]
        right = candidate_shared[name]
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            raise AV2ContractError("theta0 manifests accept tensors only")
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise AV2ContractError(f"shared theta0 bytes differ for {name}")
        if left.numel() and right.numel() and left.data_ptr() == right.data_ptr():
            raise AV2ContractError(f"shared theta0 storage aliases for {name}")
        shared[name] = {
            "dtype": str(left.dtype),
            "shape": list(left.shape),
            "sha256": _tensor_sha256(left),
        }
    candidate_entries = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _tensor_sha256(value),
        }
        for name, value in sorted(
            candidate_only.items(), key=lambda item: item[0].encode("utf-8")
        )
    }
    payload: dict[str, object] = {
        "candidate_only": candidate_entries,
        "fresh_initialization": True,
        "shared": shared,
        "storage_alias_free": True,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return MappingProxyType(
        {**payload, "combined_sha256": hashlib.sha256(canonical).hexdigest()}
    )


__all__ = [
    "AV2_ARCHITECTURE_ID",
    "AV2_CONFIG_SCHEMA",
    "AV2_CONFIG_STATE",
    "AV2_INTEGRATION_PATH",
    "AV2_OPERATOR_CLASS",
    "AV2_SHUFFLE_PERMUTATION",
    "AV2AuthorityBlocked",
    "AV2AuthorityGate",
    "AV2ContractError",
    "AV2StaticConfig",
    "DataAdequacyReport",
    "WindowIdentity",
    "assert_train_holdout_disjoint",
    "counterfactual_mse_metrics",
    "evaluate_data_adequacy",
    "shared_theta0_manifest",
    "validate_av2_static_config",
]
