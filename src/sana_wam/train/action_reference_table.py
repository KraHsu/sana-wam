"""Canonical Phase1 T0 paired action-error reference table for Phase-6."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
import struct
from typing import Any

from sana_wam.dataloader.task_sample_plan import TaskRoundRobinPlan

from .phase6_downstream_pins import (
    DATASET_CONTRACT_ARTIFACT_SHA256,
    FIXED_AMENDMENT_CONFIG_PINS,
    TASK_PLAN_ARTIFACT_SHA256,
)


REFERENCE_SCHEMA_VERSION = "sana-phase6-action-reference-v2"
PRECOMPUTE_CONFIG_SCHEMA_VERSION = (
    "sana-phase6-action-reference-precompute-config-v1"
)
EXPECTED_REFERENCE_CHECKPOINT_SHA256 = (
    "ba7bb59fe7e44f6efd6a4cef79f66aff7ccf5dbe82a5a166634e35055e94f089"
)
EXPECTED_ACTION_STATS_SHA256 = (
    "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"
)
EXPECTED_DATASET_CONTRACT_ARTIFACT_SHA256 = DATASET_CONTRACT_ARTIFACT_SHA256
EXPECTED_ROW_COUNT = 504
SPOT_CHECK_GLOBAL_STEPS = (1, 253, 504)
SPOT_CHECK_SCHEMA_VERSION = "sana-phase6-action-reference-live-spot-v2"

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "plan_sha256",
        "identity_sha256",
        "dataset_contract_artifact_sha256",
        "reference_checkpoint_sha256",
        "action_stats_sha256",
        "precompute_config_file_sha256",
        "precompute_config_sha256",
        "precompute_gpu_runtime",
        "precompute_gpu_runtime_sha256",
        "precompute_preflight_report_sha256",
        "precompute_preflight_request_sha256",
        "precompute_source_manifest_sha256",
        "precompute_config",
        "rows",
    }
)
ROW_KEYS = frozenset(
    {
        "action_sigma",
        "common_input_trace_sha256",
        "dataset_contract_row_sha256",
        "dataset_index",
        "error_float32_bits",
        "global_step",
        "plan_row_sha256",
        "reference_forward_context_sha256",
        "t0_reference_forward_trace_sha256",
        "task_name",
    }
)
PRECOMPUTE_CONFIG_KEYS = frozenset(
    {"schema_version", "model", "dataloader", "training", "execution"}
)
EXECUTION_KEYS = frozenset(
    {
        "backward_called",
        "autograd_context",
        "autograd_enabled",
        "closed_loop",
        "forward_only",
        "gradient_checkpointing",
        "model_mode",
        "model_result_key",
        "optimizer_created",
        "phase6_plan_rows_required",
        "reference_input_key",
        "resume_allowed",
        "row_context_contract",
        "row_noise_contract",
        "spot_check_global_steps",
        "training_started",
    }
)
SPOT_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "verification_passed",
        "gpu_forward_executed",
        "training_started",
        "closed_loop",
        "reference_artifact_sha256",
        "plan_sha256",
        "identity_sha256",
        "dataset_contract_artifact_sha256",
        "reference_checkpoint_sha256",
        "action_stats_sha256",
        "precompute_config_file_sha256",
        "precompute_config_sha256",
        "precompute_gpu_runtime_sha256",
        "precompute_preflight_report_sha256",
        "precompute_preflight_request_sha256",
        "precompute_source_manifest_sha256",
        "spot_gpu_runtime",
        "spot_gpu_runtime_sha256",
        "spot_preflight_report_sha256",
        "spot_preflight_request_sha256",
        "global_steps",
        "rows",
    }
)
SPOT_ROW_KEYS = frozenset(
    {
        "global_step",
        "expected_error_float32_bits",
        "observed_error_float32_bits",
        "exact_match",
        "plan_row_sha256",
        "dataset_contract_row_sha256",
        "common_input_trace_sha256",
        "reference_forward_context_sha256",
        "t0_reference_forward_trace_sha256",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FLOAT32_BITS_RE = re.compile(r"[0-9a-f]{8}\Z")
GPU_RUNTIME_KEYS = frozenset(
    {
        "compute_capability",
        "cuda_device_count",
        "cuda_device_index",
        "cuda_device_name",
        "cuda_device_uuid",
        "cuda_driver_version",
        "device_type",
        "multi_processor_count",
        "total_memory_bytes",
        "torch_cuda_version",
        "torch_version",
        "world_size",
    }
)


class ActionReferenceValidationError(ValueError):
    """Raised when reference bytes or their provenance contract drift."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ActionReferenceValidationError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActionReferenceValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(data: bytes, name: str) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ActionReferenceValidationError(f"non-finite JSON value: {value}")
            ),
        )
    except ActionReferenceValidationError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError, ValueError) as exc:
        raise ActionReferenceValidationError(f"invalid {name} JSON: {exc}") from exc


def _require_exact_keys(value: Any, expected: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise ActionReferenceValidationError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != set(expected):
        raise ActionReferenceValidationError(
            f"{name} keys differ; missing={sorted(set(expected) - actual)!r}, "
            f"extra={sorted(actual - set(expected))!r}"
        )


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ActionReferenceValidationError(f"{name} is not a lowercase SHA256")
    return value


def _require_plain_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ActionReferenceValidationError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _require_exact_number(value: Any, expected: float, name: str) -> None:
    if type(value) is not float:
        raise ActionReferenceValidationError(f"{name} must equal {expected}")
    observed = float(value)
    if not math.isfinite(observed) or observed != expected:
        raise ActionReferenceValidationError(f"{name} must equal {expected}")


def _nested(mapping: Mapping[str, Any], path: tuple[str, ...], name: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ActionReferenceValidationError(
                f"{name} is missing {'.'.join(path)}"
            )
        value = value[key]
    return value


def _float32_to_bits(value: float) -> str:
    try:
        cast = struct.pack(">f", float(value))
    except (OverflowError, TypeError, ValueError, struct.error) as exc:
        raise ActionReferenceValidationError(
            "reference error cannot be represented as float32"
        ) from exc
    decoded = struct.unpack(">f", cast)[0]
    if not math.isfinite(decoded) or decoded < 0.0 or cast[0] & 0x80:
        raise ActionReferenceValidationError(
            "reference error must be finite, non-negative float32"
        )
    return cast.hex()


def _float32_from_bits(value: Any) -> float:
    if not isinstance(value, str) or _FLOAT32_BITS_RE.fullmatch(value) is None:
        raise ActionReferenceValidationError("reference error bits must be 8 hex digits")
    raw = bytes.fromhex(value)
    decoded = struct.unpack(">f", raw)[0]
    if not math.isfinite(decoded) or decoded < 0.0 or raw[0] & 0x80:
        raise ActionReferenceValidationError(
            "reference error must be finite and non-negative"
        )
    return decoded


def float32_bits(value: float) -> str:
    """Return the exact canonical big-endian FP32 bit string."""

    return _float32_to_bits(value)


def validate_gpu_runtime(value: Any, name: str = "gpu_runtime") -> dict[str, Any]:
    """Validate the exact non-training CUDA runtime descriptor."""

    _require_exact_keys(value, GPU_RUNTIME_KEYS, name)
    if value["device_type"] != "cuda":
        raise ActionReferenceValidationError(f"{name}.device_type must be 'cuda'")
    for key, minimum in (
        ("cuda_device_count", 1),
        ("cuda_device_index", 0),
        ("multi_processor_count", 1),
        ("total_memory_bytes", 1),
        ("world_size", 1),
    ):
        observed = _require_plain_int(value[key], f"{name}.{key}", minimum=minimum)
        if key in ("cuda_device_index", "world_size") and observed != (
            0 if key == "cuda_device_index" else 1
        ):
            raise ActionReferenceValidationError(
                f"{name}.{key} differs from the single-process protocol"
            )
    capability = value["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(item) is not int or item < 0 for item in capability)
    ):
        raise ActionReferenceValidationError(
            f"{name}.compute_capability must be two non-negative integers"
        )
    for key in (
        "cuda_device_name",
        "cuda_device_uuid",
        "cuda_driver_version",
        "torch_cuda_version",
        "torch_version",
    ):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ActionReferenceValidationError(f"{name}.{key} must be non-empty")
    return deepcopy(dict(value))


def validate_precompute_config(
    value: Any,
    *,
    reference_checkpoint_sha256: str,
    action_stats_sha256: str,
    plan_sha256: str,
    identity_sha256: str,
    dataset_contract_artifact_sha256: str,
) -> dict[str, Any]:
    """Public torch-free validator used before any CUDA query or model load."""

    return _validate_precompute_config(
        value,
        reference_checkpoint_sha256=reference_checkpoint_sha256,
        action_stats_sha256=action_stats_sha256,
        plan_sha256=plan_sha256,
        identity_sha256=identity_sha256,
        dataset_contract_artifact_sha256=dataset_contract_artifact_sha256,
    )


def validate_reference_artifact_envelope_bytes(
    data: bytes,
    *,
    expected_artifact_sha256: str,
) -> dict[str, Any]:
    """Validate canonical v2 structure/provenance before importing torch."""

    if not isinstance(data, bytes):
        raise TypeError("reference artifact must be bytes")
    expected = _require_sha256(
        expected_artifact_sha256, "expected reference artifact SHA256"
    )
    if sha256(data).hexdigest() != expected:
        raise ActionReferenceValidationError(
            "reference artifact file SHA256 mismatch"
        )
    payload = _strict_json_loads(data, "reference artifact")
    if not isinstance(payload, Mapping) or data != canonical_json_bytes(payload):
        raise ActionReferenceValidationError(
            "reference artifact must be canonical JSON"
        )
    _require_exact_keys(payload, TOP_LEVEL_KEYS, "reference artifact")
    if payload["schema_version"] != REFERENCE_SCHEMA_VERSION:
        raise ActionReferenceValidationError(
            "unsupported reference schema; Phase-6 requires v2"
        )
    for key in TOP_LEVEL_KEYS - {"schema_version", "precompute_config", "precompute_gpu_runtime", "rows"}:
        _require_sha256(payload[key], f"reference artifact {key}")
    gpu_runtime = validate_gpu_runtime(
        payload["precompute_gpu_runtime"], "precompute_gpu_runtime"
    )
    if (
        sha256(canonical_json_bytes(gpu_runtime)).hexdigest()
        != payload["precompute_gpu_runtime_sha256"]
    ):
        raise ActionReferenceValidationError(
            "precompute GPU-runtime SHA256 mismatch"
        )
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROW_COUNT:
        raise ActionReferenceValidationError(
            f"reference artifact must contain exactly {EXPECTED_ROW_COUNT} rows"
        )
    for global_step, row in enumerate(rows, start=1):
        _require_exact_keys(row, ROW_KEYS, "reference row")
        if row["global_step"] != global_step:
            raise ActionReferenceValidationError(
                f"reference row order differs at global step {global_step}"
            )
        _float32_from_bits(row["error_float32_bits"])
        for key in (
            "common_input_trace_sha256",
            "dataset_contract_row_sha256",
            "plan_row_sha256",
            "reference_forward_context_sha256",
            "t0_reference_forward_trace_sha256",
        ):
            _require_sha256(row[key], f"reference row {global_step}.{key}")
    return deepcopy(dict(payload))


def _validate_precompute_config(
    value: Any,
    *,
    reference_checkpoint_sha256: str,
    action_stats_sha256: str,
    plan_sha256: str,
    identity_sha256: str,
    dataset_contract_artifact_sha256: str,
) -> dict[str, Any]:
    for digest, name in (
        (reference_checkpoint_sha256, "reference checkpoint SHA256"),
        (action_stats_sha256, "action stats SHA256"),
        (plan_sha256, "embedded plan SHA256"),
        (identity_sha256, "identity SHA256"),
        (dataset_contract_artifact_sha256, "dataset contract SHA256"),
    ):
        _require_sha256(digest, name)
    _require_exact_keys(value, PRECOMPUTE_CONFIG_KEYS, "precompute_config")
    if value["schema_version"] != PRECOMPUTE_CONFIG_SCHEMA_VERSION:
        raise ActionReferenceValidationError("precompute config schema differs")
    for key in ("model", "dataloader", "training", "execution"):
        if not isinstance(value[key], Mapping):
            raise ActionReferenceValidationError(
                f"precompute_config.{key} must be an object"
            )

    model = value["model"]
    if (
        _nested(
            model,
            ("video_backbone", "continuous_timestep_conditioning"),
            "precompute_config.model",
        )
        is not False
    ):
        raise ActionReferenceValidationError(
            "Phase1 reference requires T0 continuous_timestep_conditioning=false"
        )
    for path in (
        ("architecture", "video_local_expansion_weight"),
        ("architecture", "action_non_regression_weight"),
    ):
        _require_exact_number(
            _nested(model, path, "precompute_config.model"),
            0.0,
            f"precompute_config.model.{'.'.join(path)}",
        )
    if (
        _nested(
            model,
            ("architecture", "action_video_memory_adapter", "enabled"),
            "precompute_config.model",
        )
        is not False
    ):
        raise ActionReferenceValidationError(
            "Phase1 reference requires action video-memory adapter disabled"
        )

    dataloader = value["dataloader"]
    if dataloader.get("type") != "robotwin":
        raise ActionReferenceValidationError(
            "Phase1 reference requires dataloader.type='robotwin'"
        )
    if dataloader.get("filter_static_segments") is not False:
        raise ActionReferenceValidationError(
            "Phase1 reference requires filter_static_segments=false"
        )
    if dataloader.get("growing_history") is not False:
        raise ActionReferenceValidationError(
            "Phase1 reference requires growing_history=false"
        )
    if dataloader.get("text_embedding_cache_dir") is not None:
        raise ActionReferenceValidationError("reference text cache must be null")
    if dataloader.get("vae_cache_dir") is not None:
        raise ActionReferenceValidationError("reference VAE cache must be null")
    _require_exact_number(
        dataloader.get("text_embedding_dropout"),
        0.0,
        "precompute_config.dataloader.text_embedding_dropout",
    )

    training = value["training"]
    expected_training_digests = {
        "init_checkpoint_sha256": reference_checkpoint_sha256,
        "phase1_reference_checkpoint_sha256": reference_checkpoint_sha256,
        "action_stats_sha256": action_stats_sha256,
        "phase6_plan_sha256": plan_sha256,
        "phase6_identity_sha256": identity_sha256,
        "phase6_plan_artifact_sha256": TASK_PLAN_ARTIFACT_SHA256,
        "phase6_dataset_contract_artifact_sha256": (
            dataset_contract_artifact_sha256
        ),
        **FIXED_AMENDMENT_CONFIG_PINS,
    }
    for key, expected in expected_training_digests.items():
        if training.get(key) != expected:
            raise ActionReferenceValidationError(
                f"precompute_config.training.{key} differs from provenance"
            )
    if training.get("phase1_action_reference_precompute_mode") is not True:
        raise ActionReferenceValidationError(
            "precompute config must declare phase1_action_reference_precompute_mode=true"
        )
    for key, expected in (
        ("lambda_video", 1.0),
        ("lambda_action", 0.0),
    ):
        _require_exact_number(
            training.get(key), expected, f"precompute_config.training.{key}"
        )
    for key, expected in (
        ("expected_world_size", 1),
        ("batch_size", 1),
        ("gradient_accumulation_steps", 1),
        ("max_steps", EXPECTED_ROW_COUNT),
        ("seed", 20260724),
        ("num_workers", 0),
    ):
        if type(training.get(key)) is not int or training[key] != expected:
            raise ActionReferenceValidationError(
                f"precompute_config.training.{key} must equal {expected}"
            )
    forbidden_reference_keys = [
        key for key in training if key.startswith("phase1_action_reference_artifact")
    ]
    if forbidden_reference_keys:
        raise ActionReferenceValidationError(
            "precompute config may not consume an existing reference table"
        )

    execution = value["execution"]
    _require_exact_keys(execution, EXECUTION_KEYS, "precompute_config.execution")
    expected_execution = {
        "autograd_context": "torch.no_grad",
        "autograd_enabled": False,
        "backward_called": False,
        "closed_loop": False,
        "forward_only": True,
        "gradient_checkpointing": False,
        "model_mode": "eval",
        "model_result_key": "phase6_action_unweighted_mse",
        "optimizer_created": False,
        "phase6_plan_rows_required": True,
        "reference_input_key": None,
        "resume_allowed": False,
        "row_context_contract": (
            "plan-row prompt plus dataset-contract logical-length/episode/instruction"
        ),
        "row_noise_contract": "phase6_plan_row.domain_seeds",
        "spot_check_global_steps": list(SPOT_CHECK_GLOBAL_STEPS),
        "training_started": False,
    }
    if dict(execution) != expected_execution:
        raise ActionReferenceValidationError(
            "precompute execution contract differs from frozen forward-only protocol"
        )
    return deepcopy(dict(value))


def _validate_plan_and_dataset_rows(
    plan: TaskRoundRobinPlan,
    dataset_contract_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    plan.validate()
    if len(plan.rows) != EXPECTED_ROW_COUNT:
        raise ActionReferenceValidationError(
            f"reference plan must contain exactly {EXPECTED_ROW_COUNT} rows"
        )
    if isinstance(dataset_contract_rows, (str, bytes)) or not isinstance(
        dataset_contract_rows, Sequence
    ):
        raise ActionReferenceValidationError(
            "dataset_contract_rows must be an ordered sequence"
        )
    if len(dataset_contract_rows) != len(plan.rows):
        raise ActionReferenceValidationError(
            "dataset-contract row count differs from the frozen plan"
        )
    copied = []
    for expected_step, (plan_row, dataset_row) in enumerate(
        zip(plan.rows, dataset_contract_rows, strict=True), start=1
    ):
        if not isinstance(dataset_row, Mapping):
            raise ActionReferenceValidationError(
                f"dataset-contract row {expected_step} must be an object"
            )
        identity = plan_row.identity
        for key, expected in (
            ("global_step", expected_step),
            ("dataset_index", identity.dataset_index),
            ("task_name", identity.task_name),
            ("episode_index", identity.episode_index),
            ("episode_path", identity.episode_path),
            ("start_frame", identity.start_frame),
        ):
            if dataset_row.get(key) != expected:
                raise ActionReferenceValidationError(
                    f"dataset-contract row {expected_step}.{key} differs from plan"
                )
        copied.append(deepcopy(dict(dataset_row)))
    return tuple(copied)


def _row_hashes(
    plan_row: Any,
    dataset_row: Mapping[str, Any],
    *,
    provenance: Mapping[str, str],
    common_input_trace_sha256: str,
    t0_reference_forward_trace_sha256: str,
) -> tuple[str, str, str]:
    plan_row_sha256 = sha256(canonical_json_bytes(plan_row.to_dict())).hexdigest()
    dataset_row_sha256 = sha256(canonical_json_bytes(dataset_row)).hexdigest()
    context = {
        "action_stats_sha256": provenance["action_stats_sha256"],
        "dataset_contract_artifact_sha256": provenance[
            "dataset_contract_artifact_sha256"
        ],
        "dataset_contract_row_sha256": dataset_row_sha256,
        "common_input_trace_sha256": _require_sha256(
            common_input_trace_sha256, "common_input_trace_sha256"
        ),
        "plan_row_sha256": plan_row_sha256,
        "precompute_config_sha256": provenance["precompute_config_sha256"],
        "precompute_gpu_runtime_sha256": provenance[
            "precompute_gpu_runtime_sha256"
        ],
        "precompute_preflight_report_sha256": provenance[
            "precompute_preflight_report_sha256"
        ],
        "precompute_preflight_request_sha256": provenance[
            "precompute_preflight_request_sha256"
        ],
        "precompute_source_manifest_sha256": provenance[
            "precompute_source_manifest_sha256"
        ],
        "reference_checkpoint_sha256": provenance[
            "reference_checkpoint_sha256"
        ],
        "t0_reference_forward_trace_sha256": _require_sha256(
            t0_reference_forward_trace_sha256,
            "t0_reference_forward_trace_sha256",
        ),
    }
    return (
        plan_row_sha256,
        dataset_row_sha256,
        sha256(canonical_json_bytes(context)).hexdigest(),
    )


@dataclass(frozen=True)
class ActionReferenceTable:
    plan_sha256: str
    identity_sha256: str
    dataset_contract_artifact_sha256: str
    reference_checkpoint_sha256: str
    action_stats_sha256: str
    precompute_config_file_sha256: str
    precompute_config_sha256: str
    precompute_gpu_runtime_sha256: str
    precompute_preflight_request_sha256: str
    precompute_preflight_report_sha256: str
    precompute_source_manifest_sha256: str
    _precompute_config: Mapping[str, Any]
    _precompute_gpu_runtime: Mapping[str, Any]
    error_bits: tuple[str, ...]
    common_input_trace_sha256: tuple[str, ...]
    t0_reference_forward_trace_sha256: tuple[str, ...]
    plan_row_sha256: tuple[str, ...]
    dataset_contract_row_sha256: tuple[str, ...]
    reference_forward_context_sha256: tuple[str, ...]

    @property
    def precompute_config(self) -> dict[str, Any]:
        return deepcopy(dict(self._precompute_config))

    @property
    def precompute_gpu_runtime(self) -> dict[str, Any]:
        return deepcopy(dict(self._precompute_gpu_runtime))

    @classmethod
    def build(
        cls,
        plan: TaskRoundRobinPlan,
        dataset_contract_rows: Sequence[Mapping[str, Any]],
        errors: Sequence[float],
        common_input_trace_sha256: Sequence[str],
        t0_reference_forward_trace_sha256: Sequence[str],
        *,
        dataset_contract_artifact_sha256: str,
        reference_checkpoint_sha256: str,
        action_stats_sha256: str,
        precompute_config_file_sha256: str,
        precompute_config: Mapping[str, Any],
        precompute_gpu_runtime: Mapping[str, Any],
        precompute_preflight_request_sha256: str,
        precompute_preflight_report_sha256: str,
        precompute_source_manifest_sha256: str,
    ) -> "ActionReferenceTable":
        rows = _validate_plan_and_dataset_rows(plan, dataset_contract_rows)
        if isinstance(errors, (str, bytes)) or not isinstance(errors, Sequence):
            raise ActionReferenceValidationError("errors must be an ordered sequence")
        if len(errors) != len(plan.rows):
            raise ActionReferenceValidationError(
                "reference error count differs from the frozen plan"
            )
        for name, values in (
            ("common_input_trace_sha256", common_input_trace_sha256),
            (
                "t0_reference_forward_trace_sha256",
                t0_reference_forward_trace_sha256,
            ),
        ):
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ActionReferenceValidationError(f"{name} must be a sequence")
            if len(values) != len(plan.rows):
                raise ActionReferenceValidationError(f"{name} row count differs")
            for value in values:
                _require_sha256(value, name)
        dataset_digest = _require_sha256(
            dataset_contract_artifact_sha256,
            "dataset_contract_artifact_sha256",
        )
        if dataset_digest != EXPECTED_DATASET_CONTRACT_ARTIFACT_SHA256:
            raise ActionReferenceValidationError(
                "dataset-contract artifact SHA differs from the frozen real artifact"
            )
        reference_digest = _require_sha256(
            reference_checkpoint_sha256, "reference_checkpoint_sha256"
        )
        if reference_digest != EXPECTED_REFERENCE_CHECKPOINT_SHA256:
            raise ActionReferenceValidationError(
                "reference checkpoint differs from frozen Phase1"
            )
        stats_digest = _require_sha256(action_stats_sha256, "action_stats_sha256")
        if stats_digest != EXPECTED_ACTION_STATS_SHA256:
            raise ActionReferenceValidationError(
                "action stats differ from the frozen Phase-6 contract"
            )
        config_file_digest = _require_sha256(
            precompute_config_file_sha256, "precompute_config_file_sha256"
        )
        source_digest = _require_sha256(
            precompute_source_manifest_sha256,
            "precompute_source_manifest_sha256",
        )
        preflight_request_digest = _require_sha256(
            precompute_preflight_request_sha256,
            "precompute_preflight_request_sha256",
        )
        preflight_report_digest = _require_sha256(
            precompute_preflight_report_sha256,
            "precompute_preflight_report_sha256",
        )
        gpu_runtime = validate_gpu_runtime(
            precompute_gpu_runtime, "precompute_gpu_runtime"
        )
        gpu_runtime_digest = sha256(canonical_json_bytes(gpu_runtime)).hexdigest()
        config = _validate_precompute_config(
            precompute_config,
            reference_checkpoint_sha256=reference_digest,
            action_stats_sha256=stats_digest,
            plan_sha256=plan.plan_sha256,
            identity_sha256=plan.identity_sha256,
            dataset_contract_artifact_sha256=dataset_digest,
        )
        config_digest = sha256(canonical_json_bytes(config)).hexdigest()
        provenance = {
            "action_stats_sha256": stats_digest,
            "dataset_contract_artifact_sha256": dataset_digest,
            "precompute_config_sha256": config_digest,
            "precompute_gpu_runtime_sha256": gpu_runtime_digest,
            "precompute_preflight_report_sha256": preflight_report_digest,
            "precompute_preflight_request_sha256": preflight_request_digest,
            "precompute_source_manifest_sha256": source_digest,
            "reference_checkpoint_sha256": reference_digest,
        }
        hashes = [
            _row_hashes(
                plan_row,
                dataset_row,
                provenance=provenance,
                common_input_trace_sha256=common_trace,
                t0_reference_forward_trace_sha256=t0_trace,
            )
            for plan_row, dataset_row, common_trace, t0_trace in zip(
                plan.rows,
                rows,
                common_input_trace_sha256,
                t0_reference_forward_trace_sha256,
                strict=True,
            )
        ]
        table = cls(
            plan_sha256=plan.plan_sha256,
            identity_sha256=plan.identity_sha256,
            dataset_contract_artifact_sha256=dataset_digest,
            reference_checkpoint_sha256=reference_digest,
            action_stats_sha256=stats_digest,
            precompute_config_file_sha256=config_file_digest,
            precompute_config_sha256=config_digest,
            precompute_gpu_runtime_sha256=gpu_runtime_digest,
            precompute_preflight_request_sha256=preflight_request_digest,
            precompute_preflight_report_sha256=preflight_report_digest,
            precompute_source_manifest_sha256=source_digest,
            _precompute_config=config,
            _precompute_gpu_runtime=gpu_runtime,
            error_bits=tuple(_float32_to_bits(value) for value in errors),
            common_input_trace_sha256=tuple(common_input_trace_sha256),
            t0_reference_forward_trace_sha256=tuple(
                t0_reference_forward_trace_sha256
            ),
            plan_row_sha256=tuple(item[0] for item in hashes),
            dataset_contract_row_sha256=tuple(item[1] for item in hashes),
            reference_forward_context_sha256=tuple(item[2] for item in hashes),
        )
        table.validate(plan, rows)
        return table

    def validate(
        self,
        plan: TaskRoundRobinPlan,
        dataset_contract_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        rows = _validate_plan_and_dataset_rows(plan, dataset_contract_rows)
        for name, value in (
            ("plan_sha256", self.plan_sha256),
            ("identity_sha256", self.identity_sha256),
            (
                "dataset_contract_artifact_sha256",
                self.dataset_contract_artifact_sha256,
            ),
            ("reference_checkpoint_sha256", self.reference_checkpoint_sha256),
            ("action_stats_sha256", self.action_stats_sha256),
            ("precompute_config_file_sha256", self.precompute_config_file_sha256),
            ("precompute_config_sha256", self.precompute_config_sha256),
            (
                "precompute_gpu_runtime_sha256",
                self.precompute_gpu_runtime_sha256,
            ),
            (
                "precompute_preflight_request_sha256",
                self.precompute_preflight_request_sha256,
            ),
            (
                "precompute_preflight_report_sha256",
                self.precompute_preflight_report_sha256,
            ),
            (
                "precompute_source_manifest_sha256",
                self.precompute_source_manifest_sha256,
            ),
        ):
            _require_sha256(value, name)
        if self.plan_sha256 != plan.plan_sha256:
            raise ActionReferenceValidationError("reference plan SHA256 mismatch")
        if self.identity_sha256 != plan.identity_sha256:
            raise ActionReferenceValidationError("reference identity SHA256 mismatch")
        if (
            self.dataset_contract_artifact_sha256
            != EXPECTED_DATASET_CONTRACT_ARTIFACT_SHA256
        ):
            raise ActionReferenceValidationError("reference dataset-contract mismatch")
        if self.reference_checkpoint_sha256 != EXPECTED_REFERENCE_CHECKPOINT_SHA256:
            raise ActionReferenceValidationError("reference Phase1 checkpoint mismatch")
        if self.action_stats_sha256 != EXPECTED_ACTION_STATS_SHA256:
            raise ActionReferenceValidationError("reference action-stats mismatch")
        config = _validate_precompute_config(
            self._precompute_config,
            reference_checkpoint_sha256=self.reference_checkpoint_sha256,
            action_stats_sha256=self.action_stats_sha256,
            plan_sha256=self.plan_sha256,
            identity_sha256=self.identity_sha256,
            dataset_contract_artifact_sha256=(
                self.dataset_contract_artifact_sha256
            ),
        )
        if sha256(canonical_json_bytes(config)).hexdigest() != self.precompute_config_sha256:
            raise ActionReferenceValidationError("precompute config SHA256 mismatch")
        gpu_runtime = validate_gpu_runtime(
            self._precompute_gpu_runtime, "precompute_gpu_runtime"
        )
        if (
            sha256(canonical_json_bytes(gpu_runtime)).hexdigest()
            != self.precompute_gpu_runtime_sha256
        ):
            raise ActionReferenceValidationError(
                "precompute GPU-runtime SHA256 mismatch"
            )
        lengths = {
            len(self.error_bits),
            len(self.common_input_trace_sha256),
            len(self.t0_reference_forward_trace_sha256),
            len(self.plan_row_sha256),
            len(self.dataset_contract_row_sha256),
            len(self.reference_forward_context_sha256),
        }
        if lengths != {len(plan.rows)}:
            raise ActionReferenceValidationError("reference row count mismatch")
        provenance = {
            "action_stats_sha256": self.action_stats_sha256,
            "dataset_contract_artifact_sha256": (
                self.dataset_contract_artifact_sha256
            ),
            "precompute_config_sha256": self.precompute_config_sha256,
            "precompute_gpu_runtime_sha256": (
                self.precompute_gpu_runtime_sha256
            ),
            "precompute_preflight_report_sha256": (
                self.precompute_preflight_report_sha256
            ),
            "precompute_preflight_request_sha256": (
                self.precompute_preflight_request_sha256
            ),
            "precompute_source_manifest_sha256": (
                self.precompute_source_manifest_sha256
            ),
            "reference_checkpoint_sha256": self.reference_checkpoint_sha256,
        }
        for index, (plan_row, dataset_row) in enumerate(
            zip(plan.rows, rows, strict=True)
        ):
            _float32_from_bits(self.error_bits[index])
            _require_sha256(
                self.common_input_trace_sha256[index],
                "common_input_trace_sha256",
            )
            _require_sha256(
                self.t0_reference_forward_trace_sha256[index],
                "t0_reference_forward_trace_sha256",
            )
            expected = _row_hashes(
                plan_row,
                dataset_row,
                provenance=provenance,
                common_input_trace_sha256=(
                    self.common_input_trace_sha256[index]
                ),
                t0_reference_forward_trace_sha256=(
                    self.t0_reference_forward_trace_sha256[index]
                ),
            )
            observed = (
                self.plan_row_sha256[index],
                self.dataset_contract_row_sha256[index],
                self.reference_forward_context_sha256[index],
            )
            if observed != expected:
                raise ActionReferenceValidationError(
                    f"reference row provenance drift at step {index + 1}"
                )

    def to_payload(
        self,
        plan: TaskRoundRobinPlan,
        dataset_contract_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        self.validate(plan, dataset_contract_rows)
        return {
            "action_stats_sha256": self.action_stats_sha256,
            "dataset_contract_artifact_sha256": (
                self.dataset_contract_artifact_sha256
            ),
            "identity_sha256": self.identity_sha256,
            "plan_sha256": self.plan_sha256,
            "precompute_config": self.precompute_config,
            "precompute_config_file_sha256": self.precompute_config_file_sha256,
            "precompute_config_sha256": self.precompute_config_sha256,
            "precompute_gpu_runtime": self.precompute_gpu_runtime,
            "precompute_gpu_runtime_sha256": self.precompute_gpu_runtime_sha256,
            "precompute_preflight_report_sha256": (
                self.precompute_preflight_report_sha256
            ),
            "precompute_preflight_request_sha256": (
                self.precompute_preflight_request_sha256
            ),
            "precompute_source_manifest_sha256": (
                self.precompute_source_manifest_sha256
            ),
            "reference_checkpoint_sha256": self.reference_checkpoint_sha256,
            "rows": [
                {
                    "action_sigma": row.action_sigma,
                    "common_input_trace_sha256": (
                        self.common_input_trace_sha256[index]
                    ),
                    "dataset_contract_row_sha256": (
                        self.dataset_contract_row_sha256[index]
                    ),
                    "dataset_index": row.identity.dataset_index,
                    "error_float32_bits": self.error_bits[index],
                    "global_step": row.global_step,
                    "plan_row_sha256": self.plan_row_sha256[index],
                    "reference_forward_context_sha256": (
                        self.reference_forward_context_sha256[index]
                    ),
                    "t0_reference_forward_trace_sha256": (
                        self.t0_reference_forward_trace_sha256[index]
                    ),
                    "task_name": row.identity.task_name,
                }
                for index, row in enumerate(plan.rows)
            ],
            "schema_version": REFERENCE_SCHEMA_VERSION,
        }

    def to_artifact_bytes(
        self,
        plan: TaskRoundRobinPlan,
        dataset_contract_rows: Sequence[Mapping[str, Any]],
    ) -> bytes:
        return canonical_json_bytes(self.to_payload(plan, dataset_contract_rows))

    @property
    def errors(self) -> tuple[float, ...]:
        return tuple(_float32_from_bits(value) for value in self.error_bits)

    def error_for_global_step(self, global_step: int) -> float:
        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or not 1 <= global_step <= len(self.error_bits)
        ):
            raise IndexError(global_step)
        return _float32_from_bits(self.error_bits[global_step - 1])

    @classmethod
    def from_artifact_bytes(
        cls,
        data: bytes,
        plan: TaskRoundRobinPlan,
        dataset_contract_rows: Sequence[Mapping[str, Any]],
        *,
        expected_artifact_sha256: str,
        expected_dataset_contract_artifact_sha256: str,
        expected_reference_checkpoint_sha256: str,
        expected_action_stats_sha256: str,
        expected_precompute_source_manifest_sha256: str,
    ) -> "ActionReferenceTable":
        if not isinstance(data, bytes):
            raise TypeError("reference artifact must be bytes")
        expected_artifact = _require_sha256(
            expected_artifact_sha256, "expected_artifact_sha256"
        )
        actual_artifact = sha256(data).hexdigest()
        if actual_artifact != expected_artifact:
            raise ActionReferenceValidationError(
                "reference artifact file SHA256 mismatch"
            )
        payload = _strict_json_loads(data, "reference artifact")
        if not isinstance(payload, Mapping) or data != canonical_json_bytes(payload):
            raise ActionReferenceValidationError(
                "reference artifact must be canonical JSON"
            )
        _require_exact_keys(payload, TOP_LEVEL_KEYS, "reference artifact")
        if payload["schema_version"] != REFERENCE_SCHEMA_VERSION:
            raise ActionReferenceValidationError(
                "unsupported reference schema; Phase-6 requires v2"
            )
        rows = _validate_plan_and_dataset_rows(plan, dataset_contract_rows)
        observed_rows = payload["rows"]
        if not isinstance(observed_rows, list) or len(observed_rows) != len(plan.rows):
            raise ActionReferenceValidationError("reference rows differ from plan length")
        error_bits = []
        common_traces = []
        t0_traces = []
        plan_hashes = []
        dataset_hashes = []
        context_hashes = []
        for expected_row, observed in zip(plan.rows, observed_rows, strict=True):
            _require_exact_keys(observed, ROW_KEYS, "reference row")
            expected_projection = {
                "action_sigma": expected_row.action_sigma,
                "dataset_index": expected_row.identity.dataset_index,
                "global_step": expected_row.global_step,
                "task_name": expected_row.identity.task_name,
            }
            if any(
                observed[key] != value for key, value in expected_projection.items()
            ):
                raise ActionReferenceValidationError(
                    f"reference row identity drift at step {expected_row.global_step}"
                )
            error_bits.append(observed["error_float32_bits"])
            common_traces.append(observed["common_input_trace_sha256"])
            t0_traces.append(observed["t0_reference_forward_trace_sha256"])
            plan_hashes.append(observed["plan_row_sha256"])
            dataset_hashes.append(observed["dataset_contract_row_sha256"])
            context_hashes.append(observed["reference_forward_context_sha256"])
        table = cls(
            plan_sha256=payload["plan_sha256"],
            identity_sha256=payload["identity_sha256"],
            dataset_contract_artifact_sha256=payload[
                "dataset_contract_artifact_sha256"
            ],
            reference_checkpoint_sha256=payload["reference_checkpoint_sha256"],
            action_stats_sha256=payload["action_stats_sha256"],
            precompute_config_file_sha256=payload[
                "precompute_config_file_sha256"
            ],
            precompute_config_sha256=payload["precompute_config_sha256"],
            precompute_gpu_runtime_sha256=payload[
                "precompute_gpu_runtime_sha256"
            ],
            precompute_preflight_request_sha256=payload[
                "precompute_preflight_request_sha256"
            ],
            precompute_preflight_report_sha256=payload[
                "precompute_preflight_report_sha256"
            ],
            precompute_source_manifest_sha256=payload[
                "precompute_source_manifest_sha256"
            ],
            _precompute_config=deepcopy(dict(payload["precompute_config"])),
            _precompute_gpu_runtime=deepcopy(
                dict(payload["precompute_gpu_runtime"])
            ),
            error_bits=tuple(error_bits),
            common_input_trace_sha256=tuple(common_traces),
            t0_reference_forward_trace_sha256=tuple(t0_traces),
            plan_row_sha256=tuple(plan_hashes),
            dataset_contract_row_sha256=tuple(dataset_hashes),
            reference_forward_context_sha256=tuple(context_hashes),
        )
        table.validate(plan, rows)
        expected_pins = (
            (
                table.dataset_contract_artifact_sha256,
                expected_dataset_contract_artifact_sha256,
                "dataset-contract artifact",
            ),
            (
                table.reference_checkpoint_sha256,
                expected_reference_checkpoint_sha256,
                "reference checkpoint",
            ),
            (
                table.action_stats_sha256,
                expected_action_stats_sha256,
                "action stats",
            ),
            (
                table.precompute_source_manifest_sha256,
                expected_precompute_source_manifest_sha256,
                "precompute source manifest",
            ),
        )
        for observed, expected, name in expected_pins:
            if observed != _require_sha256(expected, f"expected {name} SHA256"):
                raise ActionReferenceValidationError(f"reference {name} SHA256 mismatch")
        return table

    def artifact_sha256(
        self,
        plan: TaskRoundRobinPlan,
        dataset_contract_rows: Sequence[Mapping[str, Any]],
    ) -> str:
        return sha256(
            self.to_artifact_bytes(plan, dataset_contract_rows)
        ).hexdigest()


def validate_live_spot_artifact_bytes(
    data: bytes,
    table: ActionReferenceTable,
    plan: TaskRoundRobinPlan,
    dataset_contract_rows: Sequence[Mapping[str, Any]],
    *,
    expected_artifact_sha256: str,
    expected_reference_artifact_sha256: str,
) -> dict[str, Any]:
    """Validate independent live Phase1 re-forwards at the fixed three steps."""

    if not isinstance(data, bytes):
        raise TypeError("live-spot artifact must be bytes")
    expected_artifact = _require_sha256(
        expected_artifact_sha256, "expected live-spot artifact SHA256"
    )
    if sha256(data).hexdigest() != expected_artifact:
        raise ActionReferenceValidationError("live-spot artifact file SHA256 mismatch")
    value = _strict_json_loads(data, "live-spot artifact")
    if not isinstance(value, Mapping) or data != canonical_json_bytes(value):
        raise ActionReferenceValidationError(
            "live-spot artifact must be canonical JSON"
        )
    _require_exact_keys(value, SPOT_TOP_LEVEL_KEYS, "live-spot artifact")
    if value["schema_version"] != SPOT_CHECK_SCHEMA_VERSION:
        raise ActionReferenceValidationError("live-spot schema differs")
    if value["verification_passed"] is not True:
        raise ActionReferenceValidationError("live-spot verification did not pass")
    if value["gpu_forward_executed"] is not True:
        raise ActionReferenceValidationError("live-spot GPU forward was not executed")
    if value["training_started"] is not False or value["closed_loop"] is not False:
        raise ActionReferenceValidationError(
            "live-spot artifact must be forward-only with no training/closed loop"
        )
    for key in (
        "spot_preflight_request_sha256",
        "spot_preflight_report_sha256",
        "spot_gpu_runtime_sha256",
    ):
        _require_sha256(value[key], f"live-spot {key}")
    spot_gpu_runtime = validate_gpu_runtime(
        value["spot_gpu_runtime"], "live-spot spot_gpu_runtime"
    )
    if (
        sha256(canonical_json_bytes(spot_gpu_runtime)).hexdigest()
        != value["spot_gpu_runtime_sha256"]
    ):
        raise ActionReferenceValidationError(
            "live-spot GPU-runtime SHA256 mismatch"
        )
    table.validate(plan, dataset_contract_rows)
    expected_reference = _require_sha256(
        expected_reference_artifact_sha256,
        "expected reference artifact SHA256",
    )
    if table.artifact_sha256(plan, dataset_contract_rows) != expected_reference:
        raise ActionReferenceValidationError(
            "expected reference artifact SHA256 does not identify the table"
        )
    expected_pins = {
        "reference_artifact_sha256": expected_reference,
        "plan_sha256": table.plan_sha256,
        "identity_sha256": table.identity_sha256,
        "dataset_contract_artifact_sha256": (
            table.dataset_contract_artifact_sha256
        ),
        "reference_checkpoint_sha256": table.reference_checkpoint_sha256,
        "action_stats_sha256": table.action_stats_sha256,
        "precompute_config_file_sha256": table.precompute_config_file_sha256,
        "precompute_config_sha256": table.precompute_config_sha256,
        "precompute_gpu_runtime_sha256": table.precompute_gpu_runtime_sha256,
        "precompute_preflight_report_sha256": (
            table.precompute_preflight_report_sha256
        ),
        "precompute_preflight_request_sha256": (
            table.precompute_preflight_request_sha256
        ),
        "precompute_source_manifest_sha256": (
            table.precompute_source_manifest_sha256
        ),
        "spot_preflight_request_sha256": (
            table.precompute_preflight_request_sha256
        ),
        "spot_preflight_report_sha256": (
            table.precompute_preflight_report_sha256
        ),
        "spot_gpu_runtime_sha256": table.precompute_gpu_runtime_sha256,
    }
    for key, expected in expected_pins.items():
        if value[key] != expected:
            raise ActionReferenceValidationError(f"live-spot {key} mismatch")
    if value["global_steps"] != list(SPOT_CHECK_GLOBAL_STEPS):
        raise ActionReferenceValidationError("live-spot global steps differ")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != len(SPOT_CHECK_GLOBAL_STEPS):
        raise ActionReferenceValidationError("live-spot rows differ")
    for row, global_step in zip(rows, SPOT_CHECK_GLOBAL_STEPS, strict=True):
        _require_exact_keys(row, SPOT_ROW_KEYS, "live-spot row")
        index = global_step - 1
        expected_bits = table.error_bits[index]
        expected_row = {
            "global_step": global_step,
            "expected_error_float32_bits": expected_bits,
            "observed_error_float32_bits": expected_bits,
            "exact_match": True,
            "common_input_trace_sha256": table.common_input_trace_sha256[index],
            "plan_row_sha256": table.plan_row_sha256[index],
            "dataset_contract_row_sha256": (
                table.dataset_contract_row_sha256[index]
            ),
            "reference_forward_context_sha256": (
                table.reference_forward_context_sha256[index]
            ),
            "t0_reference_forward_trace_sha256": (
                table.t0_reference_forward_trace_sha256[index]
            ),
        }
        if dict(row) != expected_row:
            raise ActionReferenceValidationError(
                f"live-spot row differs at global step {global_step}"
            )
    return deepcopy(dict(value))


__all__ = [
    "ActionReferenceTable",
    "ActionReferenceValidationError",
    "EXPECTED_ACTION_STATS_SHA256",
    "EXPECTED_DATASET_CONTRACT_ARTIFACT_SHA256",
    "EXPECTED_REFERENCE_CHECKPOINT_SHA256",
    "PRECOMPUTE_CONFIG_SCHEMA_VERSION",
    "REFERENCE_SCHEMA_VERSION",
    "SPOT_CHECK_GLOBAL_STEPS",
    "SPOT_CHECK_SCHEMA_VERSION",
    "canonical_json_bytes",
    "float32_bits",
    "validate_gpu_runtime",
    "validate_live_spot_artifact_bytes",
    "validate_precompute_config",
    "validate_reference_artifact_envelope_bytes",
]
