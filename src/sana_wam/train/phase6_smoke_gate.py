"""Torch-free, fail-closed validation for the formal Phase-6 real-2B smoke.

This module deliberately has no dependency on PyTorch or on the evolving
Phase-6 preflight implementation.  It validates the immutable request and the
result artifact that authorize later formal training.  The GPU runner lives in
``scripts/run_phase6_real_2b_smoke.py`` and imports PyTorch only after its
request and every executable/artifact pin have been checked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import hmac
import json
import math
import re
from typing import Any

from sana_wam.train.phase6_recovery import recovery_lifecycle


SMOKE_ARTIFACT_SCHEMA_VERSION = "sana-phase6-real-2b-smoke-artifact-v2"
SMOKE_ARTIFACT_ROLE = "real_2b_smoke_gate"
RUNTIME_SUPPORT_API_VERSION = "sana-phase6-real-2b-smoke-runtime-v1"
PEAK_RESERVED_LIMIT_BYTES = 130 * 2**30
MINIMUM_REAL_2B_VIDEO_PARAMETER_COUNT = 1_500_000_000
MINIMUM_FULL_CHECKPOINT_SIZE_BYTES = 10 * 2**30

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FLOAT32_BITS_RE = re.compile(r"[0-9a-f]{8}\Z")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_EMPTY_KEY_SET_SHA256 = sha256(b"[]\n").hexdigest()

_BINDING_KEYS = frozenset(
    {
        "action_stats_sha256",
        "arm_projection_manifest_sha256",
        "dataset_contract_sha256",
        "identity_sha256",
        "phase1_checkpoint_sha256",
        "plan_sha256",
        "preflight_report_sha256",
        "preflight_request_sha256",
        "reference_artifact_sha256",
        "runtime_support_manifest_sha256",
        "smoke_runtime_config_sha256",
        "source_manifest_sha256",
        "spot_artifact_sha256",
        "student_checkpoint_sha256",
    }
)

_REGISTERED_FIVE_CASES: tuple[tuple[str, tuple[bool, bool, bool]], ...] = (
    ("T0_E0A0", (False, False, False)),
    ("T1_E0A0", (True, False, False)),
    ("T1_E1A0", (True, True, False)),
    ("T1_E0A1", (True, False, True)),
    ("T1_E1A1", (True, True, True)),
)
SMOKE_AUTHORIZATION = {
    "operation": "phase6_real_2b_smoke",
    "paired_global_step": 1,
    "arms": [case_id for case_id, _factors in _REGISTERED_FIVE_CASES],
    "backward_arm": "T1_E1A1",
    "checkpoint_roundtrip_arm": "T1_E1A1",
    "identity_pair": ["T1_E0A0", "T1_E0A1"],
    "memory_limit_bytes": PEAK_RESERVED_LIMIT_BYTES,
    "reference_gpu_forward_completed": True,
    "smoke_gpu_forward_allowed": True,
    "backward_allowed": True,
    "temporary_checkpoint_roundtrip_allowed": True,
    "optimizer_creation_allowed": False,
    "optimizer_step_allowed": False,
    "formal_training_allowed": False,
    "closed_loop_allowed": False,
}

_PAIRED_ROW_KEYS = frozenset(
    {
        "action_sigma",
        "common_input_trace_sha256",
        "dataset_contract_row_sha256",
        "dataset_index",
        "global_step",
        "plan_row_sha256",
        "task_name",
    }
)
_FACTOR_KEYS = frozenset({"action_adapter", "continuous_time", "expansion"})


class Phase6SmokeValidationError(ValueError):
    """Raised when a formal smoke request or result fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact sorted UTF-8 JSON with exactly one trailing newline."""

    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise Phase6SmokeValidationError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc


def _reject_constant(value: str) -> None:
    raise Phase6SmokeValidationError(f"non-finite JSON constant is forbidden: {value}")


def _pairs_to_dict(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase6SmokeValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data,
            object_pairs_hook=_pairs_to_dict,
            parse_constant=_reject_constant,
        )
    except Phase6SmokeValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase6SmokeValidationError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise Phase6SmokeValidationError(f"{label} must be a JSON object")
    if data != canonical_json_bytes(value):
        raise Phase6SmokeValidationError(f"{label} is not canonical JSON")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise Phase6SmokeValidationError(
            f"{label} keys differ: missing={missing}, extra={extra}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise Phase6SmokeValidationError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise Phase6SmokeValidationError(f"{label} must be an array")
    return value


def _bool(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise Phase6SmokeValidationError(f"{label} must be {expected!r}")


def _validate_recovery_lifecycle_fields(value: Mapping[str, Any], name: str) -> None:
    for key, expected in recovery_lifecycle().items():
        observed = value.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise Phase6SmokeValidationError(f"{name}.{key} differs")
    if "closed_loop_started" in value:
        raise Phase6SmokeValidationError(f"{name} uses the legacy closed-loop field")


def _positive_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise Phase6SmokeValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Phase6SmokeValidationError(f"{label} must be a non-negative integer")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Phase6SmokeValidationError(f"{label} must be a non-empty trimmed string")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Phase6SmokeValidationError(f"{label} must be lowercase SHA256")
    if len(set(value)) == 1:
        raise Phase6SmokeValidationError(f"{label} is a placeholder digest")
    return value


def _float32_bits(value: Any, label: str) -> str:
    if not isinstance(value, str) or _FLOAT32_BITS_RE.fullmatch(value) is None:
        raise Phase6SmokeValidationError(f"{label} must be eight lowercase hex digits")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase6SmokeValidationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Phase6SmokeValidationError(f"{label} must be finite")
    return result


def _cases_for_mode(mode: Any) -> tuple[tuple[str, tuple[bool, bool, bool]], ...]:
    if mode == "registered_five_arms":
        return _REGISTERED_FIVE_CASES
    raise Phase6SmokeValidationError("coverage mode must be 'registered_five_arms'")


def _validate_factors(
    value: Any, expected: tuple[bool, bool, bool], label: str
) -> Mapping[str, Any]:
    factors = _mapping(value, label)
    _exact_keys(factors, _FACTOR_KEYS, label)
    for key, expected_value in zip(
        ("continuous_time", "expansion", "action_adapter"), expected
    ):
        _bool(factors[key], expected_value, f"{label}.{key}")
    return factors


def _validate_paired_row(value: Any, label: str) -> Mapping[str, Any]:
    row = _mapping(value, label)
    _exact_keys(row, _PAIRED_ROW_KEYS, label)
    step = _positive_int(row["global_step"], f"{label}.global_step")
    if step != SMOKE_AUTHORIZATION["paired_global_step"]:
        raise Phase6SmokeValidationError(
            f"{label}.global_step must be the frozen step 1"
        )
    _nonnegative_int(row["dataset_index"], f"{label}.dataset_index")
    _nonempty_string(row["task_name"], f"{label}.task_name")
    sigma = _finite_number(row["action_sigma"], f"{label}.action_sigma")
    if sigma not in {0.5, 0.9, 1.0}:
        raise Phase6SmokeValidationError(
            f"{label}.action_sigma must be one of 0.5, 0.9, 1.0"
        )
    for key in (
        "plan_row_sha256",
        "dataset_contract_row_sha256",
        "common_input_trace_sha256",
    ):
        _sha(row[key], f"{label}.{key}")
    return row


def _validate_bindings(
    value: Any, expected: Mapping[str, str] | None = None
) -> Mapping[str, Any]:
    bindings = _mapping(value, "bindings")
    _exact_keys(bindings, _BINDING_KEYS, "bindings")
    for key in sorted(_BINDING_KEYS):
        observed = _sha(bindings[key], f"bindings.{key}")
        if expected is not None and not hmac.compare_digest(observed, expected[key]):
            raise Phase6SmokeValidationError(
                f"bindings.{key} mismatch: expected {expected[key]}, got {observed}"
            )
    return bindings


def _validate_authorization(value: Any) -> Mapping[str, Any]:
    authorization = _mapping(value, "authorization")
    _exact_keys(authorization, frozenset(SMOKE_AUTHORIZATION), "authorization")
    if authorization != SMOKE_AUTHORIZATION:
        raise Phase6SmokeValidationError(
            "smoke authorization differs from the frozen preflight execution contract"
        )
    return authorization


def _registered_projection_sha_by_arm(expected_manifest_sha256: str) -> dict[str, str]:
    """Validate the recovery manifest while preserving historical GPU case pins."""

    try:
        from sana_wam.train.phase6_arm_config import (
            build_phase6_legacy_arm_projection,
            build_phase6_arm_projection_manifest,
        )
    except ImportError as exc:
        raise Phase6SmokeValidationError(
            f"cannot import the frozen arm projection builder: {exc}"
        ) from exc
    manifest = build_phase6_arm_projection_manifest()
    observed_manifest_sha256 = sha256(canonical_json_bytes(manifest)).hexdigest()
    if not hmac.compare_digest(observed_manifest_sha256, expected_manifest_sha256):
        raise Phase6SmokeValidationError(
            "rebuilt arm projection manifest differs from the externally pinned artifact"
        )
    arms = manifest.get("arms") if isinstance(manifest, dict) else None
    if not isinstance(arms, list):
        raise Phase6SmokeValidationError("rebuilt arm projection manifest has no arms")
    result: dict[str, str] = {}
    for item in arms:
        if not isinstance(item, dict):
            raise Phase6SmokeValidationError(
                "rebuilt arm projection entry is malformed"
            )
        arm = item.get("arm")
        if not isinstance(arm, str) or arm in result:
            raise Phase6SmokeValidationError("rebuilt arm projection ids are malformed")
        projection_sha256 = sha256(
            canonical_json_bytes(build_phase6_legacy_arm_projection(arm))
        ).hexdigest()
        result[arm] = _sha(projection_sha256, f"rebuilt arm projection {arm} SHA256")
    if list(result) != SMOKE_AUTHORIZATION["arms"]:
        raise Phase6SmokeValidationError(
            "rebuilt arm projection order/coverage differs"
        )
    return result


def _validate_real_2b_model(value: Any, bindings: Mapping[str, Any]) -> None:
    model = _mapping(value, "real_2b_model")
    _exact_keys(
        model,
        frozenset(
            {
                "full_student_checkpoint_loaded",
                "mini_model",
                "mock_model",
                "model_dtype",
                "model_factory",
                "model_label",
                "student_checkpoint_sha256",
                "total_parameter_count",
                "video_dit_parameter_count",
            }
        ),
        "real_2b_model",
    )
    if model["model_label"] != "SANA-Video 2B 480p":
        raise Phase6SmokeValidationError(
            "real_2b_model.model_label is not production 2B"
        )
    if model["model_factory"] != "SanaMSVideoCamCtrl_1600M_P1_D20":
        raise Phase6SmokeValidationError(
            "real_2b_model.model_factory is not the registered production factory"
        )
    if model["model_dtype"] != "torch.bfloat16":
        raise Phase6SmokeValidationError(
            "real_2b_model.model_dtype must be torch.bfloat16"
        )
    _bool(model["full_student_checkpoint_loaded"], True, "full checkpoint loaded")
    _bool(model["mini_model"], False, "real_2b_model.mini_model")
    _bool(model["mock_model"], False, "real_2b_model.mock_model")
    checkpoint_sha = _sha(
        model["student_checkpoint_sha256"],
        "real_2b_model.student_checkpoint_sha256",
    )
    if checkpoint_sha != bindings["student_checkpoint_sha256"]:
        raise Phase6SmokeValidationError("real-2B student checkpoint binding differs")
    video_count = _positive_int(
        model["video_dit_parameter_count"],
        "real_2b_model.video_dit_parameter_count",
    )
    total_count = _positive_int(
        model["total_parameter_count"], "real_2b_model.total_parameter_count"
    )
    if video_count < MINIMUM_REAL_2B_VIDEO_PARAMETER_COUNT:
        raise Phase6SmokeValidationError(
            "video parameter count is not a real-2B topology"
        )
    if total_count < video_count:
        raise Phase6SmokeValidationError("total parameter count is below video count")


def _validate_case(
    value: Any,
    *,
    expected_id: str,
    expected_factors: tuple[bool, bool, bool],
    expected_projection_sha256: str,
    paired_row: Mapping[str, Any],
    real_2b_model: Mapping[str, Any],
) -> None:
    label = f"case[{expected_id}]"
    case = _mapping(value, label)
    _exact_keys(
        case,
        frozenset(
            {
                "action_adapter_gate",
                "adapter_key_gate",
                "backward_gate",
                "batch_size",
                "case_id",
                "closed_loop_started",
                "config_projection_sha256",
                "expansion_gate",
                "factors",
                "formal_training_started",
                "frozen_action_proprio_gate",
                "gpu_forward_executed",
                "memory_gate",
                "model_parameter_count",
                "optimizer_creation_count",
                "optimizer_step_count",
                "paired_fixed_row",
                "primary_checkpoint_written",
                "timestep_gate",
                "world_size",
            }
        ),
        label,
    )
    if case["case_id"] != expected_id:
        raise Phase6SmokeValidationError(f"{label}.case_id differs")
    continuous, expansion, adapter = expected_factors
    _validate_factors(case["factors"], expected_factors, f"{label}.factors")
    projection_sha256 = _sha(
        case["config_projection_sha256"], f"{label}.config_projection_sha256"
    )
    if projection_sha256 != expected_projection_sha256:
        raise Phase6SmokeValidationError(
            f"{label} projection SHA differs from the pinned projection manifest"
        )
    _bool(case["gpu_forward_executed"], True, f"{label}.gpu_forward_executed")
    if case["batch_size"] != 1 or type(case["batch_size"]) is not int:
        raise Phase6SmokeValidationError(f"{label}.batch_size must be 1")
    if case["world_size"] != 1 or type(case["world_size"]) is not int:
        raise Phase6SmokeValidationError(f"{label}.world_size must be 1")
    if case["paired_fixed_row"] != paired_row:
        raise Phase6SmokeValidationError(f"{label} did not use the exact paired row")
    case_parameter_count = _positive_int(
        case["model_parameter_count"], f"{label}.model_parameter_count"
    )
    if case_parameter_count < real_2b_model["video_dit_parameter_count"]:
        raise Phase6SmokeValidationError(
            f"{label} is smaller than its real-2B backbone"
        )

    timestep = _mapping(case["timestep_gate"], f"{label}.timestep_gate")
    _exact_keys(
        timestep,
        frozenset(
            {
                "continuous_coordinate_consumed",
                "embedder_boundary_dtype",
                "embedder_boundary_fractional_value_count",
                "embedder_boundary_value_count",
                "embedder_boundary_value_sha256",
                "mode",
                "observed_at_t_embedder_boundary",
                "old_integer_contract_reproduced",
                "t1_long_conversion_detected",
            }
        ),
        f"{label}.timestep_gate",
    )
    expected_mode = "continuous_t1" if continuous else "integer_t0"
    if timestep["mode"] != expected_mode:
        raise Phase6SmokeValidationError(f"{label} timestep mode differs")
    _bool(
        timestep["continuous_coordinate_consumed"],
        continuous,
        f"{label}.continuous_coordinate_consumed",
    )
    _bool(
        timestep["old_integer_contract_reproduced"],
        not continuous,
        f"{label}.old_integer_contract_reproduced",
    )
    _bool(
        timestep["t1_long_conversion_detected"],
        False,
        f"{label}.t1_long_conversion_detected",
    )
    _bool(
        timestep["observed_at_t_embedder_boundary"],
        True,
        f"{label}.observed_at_t_embedder_boundary",
    )
    if timestep["embedder_boundary_dtype"] != "torch.float32":
        raise Phase6SmokeValidationError(
            f"{label} timestep embedder boundary must receive torch.float32"
        )
    boundary_count = _positive_int(
        timestep["embedder_boundary_value_count"],
        f"{label}.embedder_boundary_value_count",
    )
    fractional_count = _nonnegative_int(
        timestep["embedder_boundary_fractional_value_count"],
        f"{label}.embedder_boundary_fractional_value_count",
    )
    if fractional_count > boundary_count:
        raise Phase6SmokeValidationError(
            f"{label} fractional timestep count is invalid"
        )
    if continuous and fractional_count == 0:
        raise Phase6SmokeValidationError(
            f"{label} T1 boundary did not preserve any fractional coordinate"
        )
    if not continuous and fractional_count != 0:
        raise Phase6SmokeValidationError(
            f"{label} T0 boundary did not reproduce integer-valued coordinates"
        )
    _sha(
        timestep["embedder_boundary_value_sha256"],
        f"{label}.embedder_boundary_value_sha256",
    )

    expansion_gate = _mapping(case["expansion_gate"], f"{label}.expansion_gate")
    _exact_keys(
        expansion_gate,
        frozenset(
            {
                "additional_video_forward_count",
                "disabled_loss_exact_zero",
                "enabled",
                "loss_finite",
                "loss_float32_bits",
                "on_path_loss_present",
            }
        ),
        f"{label}.expansion_gate",
    )
    _bool(expansion_gate["enabled"], expansion, f"{label}.expansion.enabled")
    expected_queries = 1 if expansion else 0
    if (
        type(expansion_gate["additional_video_forward_count"]) is not int
        or expansion_gate["additional_video_forward_count"] != expected_queries
    ):
        raise Phase6SmokeValidationError(f"{label} expansion forward count differs")
    _bool(expansion_gate["loss_finite"], True, f"{label}.expansion.loss_finite")
    _bool(
        expansion_gate["on_path_loss_present"],
        True,
        f"{label}.expansion.on_path_loss_present",
    )
    loss_bits = _float32_bits(
        expansion_gate["loss_float32_bits"], f"{label}.expansion.loss_float32_bits"
    )
    _bool(
        expansion_gate["disabled_loss_exact_zero"],
        not expansion,
        f"{label}.expansion.disabled_loss_exact_zero",
    )
    if not expansion and loss_bits != "00000000":
        raise Phase6SmokeValidationError(f"{label} disabled expansion loss is not +0.0")

    action = _mapping(case["action_adapter_gate"], f"{label}.action_adapter_gate")
    _exact_keys(
        action,
        frozenset(
            {
                "adapter_gradient_connected",
                "adapter_gradient_finite",
                "adapter_gradient_zero_allowed",
                "enabled",
                "nr_action_backbone_gradient_absent",
                "nr_autograd_probe_executed",
                "nr_loss_finite",
                "nr_video_gradient_absent",
            }
        ),
        f"{label}.action_adapter_gate",
    )
    _bool(action["enabled"], adapter, f"{label}.action.enabled")
    for key in (
        "adapter_gradient_connected",
        "adapter_gradient_finite",
        "adapter_gradient_zero_allowed",
        "nr_autograd_probe_executed",
        "nr_loss_finite",
        "nr_video_gradient_absent",
        "nr_action_backbone_gradient_absent",
    ):
        _bool(action[key], adapter, f"{label}.action.{key}")

    frozen = _mapping(
        case["frozen_action_proprio_gate"], f"{label}.frozen_action_proprio_gate"
    )
    _exact_keys(
        frozen,
        frozenset(
            {
                "after_sha256",
                "before_sha256",
                "bitwise_unchanged",
                "state_tensor_count",
            }
        ),
        f"{label}.frozen_action_proprio_gate",
    )
    _positive_int(frozen["state_tensor_count"], f"{label}.frozen.state_tensor_count")
    before = _sha(frozen["before_sha256"], f"{label}.frozen.before_sha256")
    after = _sha(frozen["after_sha256"], f"{label}.frozen.after_sha256")
    if before != after:
        raise Phase6SmokeValidationError(f"{label} action/proprio state changed")
    _bool(frozen["bitwise_unchanged"], True, f"{label}.frozen.bitwise_unchanged")

    keys = _mapping(case["adapter_key_gate"], f"{label}.adapter_key_gate")
    _exact_keys(
        keys,
        frozenset(
            {
                "complete",
                "enabled",
                "expected_key_count",
                "expected_key_set_sha256",
                "observed_key_count",
                "observed_key_set_sha256",
                "partial",
            }
        ),
        f"{label}.adapter_key_gate",
    )
    _bool(keys["enabled"], adapter, f"{label}.adapter_keys.enabled")
    expected_count = _nonnegative_int(
        keys["expected_key_count"], f"{label}.adapter_keys.expected_key_count"
    )
    observed_count = _nonnegative_int(
        keys["observed_key_count"], f"{label}.adapter_keys.observed_key_count"
    )
    if expected_count != observed_count or (adapter and expected_count == 0):
        raise Phase6SmokeValidationError(f"{label} adapter key count is incomplete")
    if not adapter and expected_count != 0:
        raise Phase6SmokeValidationError(f"{label} A0 contains adapter keys")
    expected_key_sha = _sha(
        keys["expected_key_set_sha256"], f"{label}.adapter_keys.expected_key_set_sha256"
    )
    observed_key_sha = _sha(
        keys["observed_key_set_sha256"], f"{label}.adapter_keys.observed_key_set_sha256"
    )
    if expected_key_sha != observed_key_sha:
        raise Phase6SmokeValidationError(f"{label} adapter key-set hash differs")
    if not adapter and expected_key_sha != _EMPTY_KEY_SET_SHA256:
        raise Phase6SmokeValidationError(f"{label} A0 empty key-set hash differs")
    _bool(keys["complete"], True, f"{label}.adapter_keys.complete")
    _bool(keys["partial"], False, f"{label}.adapter_keys.partial")

    backward = _mapping(case["backward_gate"], f"{label}.backward_gate")
    _exact_keys(
        backward,
        frozenset(
            {
                "all_required_gradients_finite",
                "executed",
                "full_loss_finite",
                "optimizer_step_after_backward",
            }
        ),
        f"{label}.backward_gate",
    )
    backward_required = expected_id == "T1_E1A1"
    _bool(backward["executed"], backward_required, f"{label}.backward.executed")
    _bool(backward["full_loss_finite"], True, f"{label}.backward.full_loss_finite")
    _bool(
        backward["all_required_gradients_finite"],
        backward_required,
        f"{label}.backward.all_required_gradients_finite",
    )
    _bool(
        backward["optimizer_step_after_backward"],
        False,
        f"{label}.backward.optimizer_step_after_backward",
    )

    memory = _mapping(case["memory_gate"], f"{label}.memory_gate")
    _exact_keys(
        memory,
        frozenset(
            {
                "below_limit",
                "limit_bytes",
                "peak_reserved_bytes",
                "peak_stats_reset_before_case",
            }
        ),
        f"{label}.memory_gate",
    )
    if memory["limit_bytes"] != PEAK_RESERVED_LIMIT_BYTES:
        raise Phase6SmokeValidationError(f"{label} memory limit differs from 130 GiB")
    peak = _positive_int(memory["peak_reserved_bytes"], f"{label}.peak_reserved_bytes")
    if peak >= PEAK_RESERVED_LIMIT_BYTES:
        raise Phase6SmokeValidationError(f"{label} peak reserved reached 130 GiB")
    _bool(memory["below_limit"], True, f"{label}.memory.below_limit")
    _bool(
        memory["peak_stats_reset_before_case"],
        True,
        f"{label}.memory.peak_stats_reset_before_case",
    )

    if (
        case["optimizer_creation_count"] != 0
        or type(case["optimizer_creation_count"]) is not int
    ):
        raise Phase6SmokeValidationError(
            f"{label} optimizer creation count must be zero"
        )
    if (
        case["optimizer_step_count"] != 0
        or type(case["optimizer_step_count"]) is not int
    ):
        raise Phase6SmokeValidationError(f"{label} optimizer step count must be zero")
    _bool(case["primary_checkpoint_written"], False, f"{label}.primary checkpoint")
    _bool(case["formal_training_started"], False, f"{label}.formal training")
    _bool(case["closed_loop_started"], False, f"{label}.closed loop")


def _validate_identity_gate(value: Any) -> None:
    gate = _mapping(value, "identity_adapter_bitwise")
    _exact_keys(
        gate,
        frozenset(
            {
                "adapter_restored_to_exact_identity",
                "baseline_case_id",
                "baseline_output_sha256",
                "bitwise_equal",
                "compared_tensor_count",
                "enabled_case_id",
                "enabled_output_sha256",
                "same_prepared_input",
            }
        ),
        "identity_adapter_bitwise",
    )
    if gate["baseline_case_id"] != "T1_E0A0":
        raise Phase6SmokeValidationError("identity baseline must be T1_E0A0")
    if gate["enabled_case_id"] != "T1_E0A1":
        raise Phase6SmokeValidationError("identity enabled case must be T1_E0A1")
    _positive_int(gate["compared_tensor_count"], "identity compared tensor count")
    baseline = _sha(gate["baseline_output_sha256"], "identity baseline output")
    enabled = _sha(gate["enabled_output_sha256"], "identity enabled output")
    if baseline != enabled:
        raise Phase6SmokeValidationError("identity-enabled output is not bitwise equal")
    for key in (
        "same_prepared_input",
        "adapter_restored_to_exact_identity",
        "bitwise_equal",
    ):
        _bool(gate[key], True, f"identity_adapter_bitwise.{key}")


def _validate_checkpoint_reload(value: Any) -> None:
    gate = _mapping(value, "e1a1_checkpoint_reload")
    _exact_keys(
        gate,
        frozenset(
            {
                "adapter_keys_complete",
                "backward_completed_before_save",
                "case_id",
                "checkpoint_sha256",
                "checkpoint_size_bytes",
                "full_checkpoint",
                "path_within_ephemeral_root",
                "pre_save_state_sha256",
                "reloaded_state_sha256",
                "state_bitwise_equal",
                "state_tensor_count",
                "strict_reload",
                "temporary_checkpoint_deleted",
            }
        ),
        "e1a1_checkpoint_reload",
    )
    if gate["case_id"] != "T1_E1A1":
        raise Phase6SmokeValidationError("temporary checkpoint case must be T1_E1A1")
    _sha(gate["checkpoint_sha256"], "temporary checkpoint SHA256")
    size = _positive_int(gate["checkpoint_size_bytes"], "temporary checkpoint size")
    if size < MINIMUM_FULL_CHECKPOINT_SIZE_BYTES:
        raise Phase6SmokeValidationError(
            "temporary checkpoint is not a full 2B checkpoint"
        )
    _positive_int(gate["state_tensor_count"], "temporary checkpoint state tensor count")
    before = _sha(gate["pre_save_state_sha256"], "pre-save state SHA256")
    reloaded = _sha(gate["reloaded_state_sha256"], "reloaded state SHA256")
    if before != reloaded:
        raise Phase6SmokeValidationError(
            "strictly reloaded model state differs bitwise"
        )
    for key in (
        "adapter_keys_complete",
        "backward_completed_before_save",
        "full_checkpoint",
        "path_within_ephemeral_root",
        "state_bitwise_equal",
        "strict_reload",
        "temporary_checkpoint_deleted",
    ):
        _bool(gate[key], True, f"e1a1_checkpoint_reload.{key}")


def _validate_adapter_contract(value: Any) -> None:
    contract = _mapping(value, "adapter_key_contract")
    _exact_keys(
        contract,
        frozenset(
            {
                "a0_live_key_count",
                "a0_student_checkpoint_key_count",
                "a1_expected_key_count",
                "a1_expected_key_set_sha256",
                "a1_live_key_set_sha256",
                "a1_temporary_checkpoint_key_set_sha256",
                "adapter_prefix",
                "all_a0_absent",
                "all_a1_complete",
            }
        ),
        "adapter_key_contract",
    )
    if contract["adapter_prefix"] != "action_video_memory_adapter.":
        raise Phase6SmokeValidationError("adapter checkpoint prefix differs")
    if (
        contract["a0_live_key_count"] != 0
        or contract["a0_student_checkpoint_key_count"] != 0
    ):
        raise Phase6SmokeValidationError("A0 checkpoint/model contains adapter keys")
    count = _positive_int(contract["a1_expected_key_count"], "A1 expected key count")
    if count <= 0:
        raise Phase6SmokeValidationError("A1 expected key set is empty")
    hashes = [
        _sha(contract[key], f"adapter_key_contract.{key}")
        for key in (
            "a1_expected_key_set_sha256",
            "a1_live_key_set_sha256",
            "a1_temporary_checkpoint_key_set_sha256",
        )
    ]
    if len(set(hashes)) != 1:
        raise Phase6SmokeValidationError(
            "A1 adapter key sets are not complete and equal"
        )
    _bool(contract["all_a0_absent"], True, "adapter_key_contract.all_a0_absent")
    _bool(contract["all_a1_complete"], True, "adapter_key_contract.all_a1_complete")


_GATE_KEYS = frozenset(
    {
        "action_proprio_frozen",
        "adapter_key_contract",
        "adapter_only_nr_connected_finite",
        "case_coverage",
        "closed_loop_absent",
        "e1a1_backward",
        "expansion_contract",
        "formal_training_absent",
        "identity_enabled_bitwise",
        "memory_strictly_below_limit",
        "optimizer_step_absent",
        "optimizer_creation_absent",
        "paired_fixed_row",
        "primary_checkpoint_absent",
        "real_2b_model",
        "temporary_checkpoint_deleted",
        "temporary_full_checkpoint_reload",
        "timestep_contract",
    }
)


def validate_phase6_smoke_artifact_bytes(
    data: bytes,
    *,
    expected_artifact_sha256: str,
    expected_source_manifest_sha256: str,
    expected_runtime_support_manifest_sha256: str,
    expected_student_checkpoint_sha256: str,
    expected_phase1_checkpoint_sha256: str,
    expected_action_stats_sha256: str,
    expected_plan_sha256: str,
    expected_identity_sha256: str,
    expected_dataset_contract_sha256: str,
    expected_reference_artifact_sha256: str,
    expected_spot_artifact_sha256: str,
    expected_arm_projection_manifest_sha256: str,
    expected_preflight_request_sha256: str,
    expected_preflight_report_sha256: str,
    expected_smoke_runtime_config_sha256: str,
) -> dict[str, Any]:
    """Validate one canonical passing real-2B smoke artifact.

    Every provenance pin is supplied independently by the caller.  Merely
    copying self-declared hashes inside an artifact can therefore never satisfy
    this gate.
    """

    if not isinstance(data, bytes):
        raise TypeError("smoke artifact data must be bytes")
    expected_artifact = _sha(expected_artifact_sha256, "expected artifact SHA256")
    observed_artifact = sha256(data).hexdigest()
    if not hmac.compare_digest(observed_artifact, expected_artifact):
        raise Phase6SmokeValidationError(
            f"smoke artifact SHA mismatch: expected {expected_artifact}, got {observed_artifact}"
        )
    expected_bindings = {
        "source_manifest_sha256": expected_source_manifest_sha256,
        "runtime_support_manifest_sha256": expected_runtime_support_manifest_sha256,
        "student_checkpoint_sha256": expected_student_checkpoint_sha256,
        "phase1_checkpoint_sha256": expected_phase1_checkpoint_sha256,
        "action_stats_sha256": expected_action_stats_sha256,
        "plan_sha256": expected_plan_sha256,
        "identity_sha256": expected_identity_sha256,
        "dataset_contract_sha256": expected_dataset_contract_sha256,
        "reference_artifact_sha256": expected_reference_artifact_sha256,
        "spot_artifact_sha256": expected_spot_artifact_sha256,
        "arm_projection_manifest_sha256": expected_arm_projection_manifest_sha256,
        "preflight_request_sha256": expected_preflight_request_sha256,
        "preflight_report_sha256": expected_preflight_report_sha256,
        "smoke_runtime_config_sha256": expected_smoke_runtime_config_sha256,
    }
    for key, value in tuple(expected_bindings.items()):
        expected_bindings[key] = _sha(value, f"expected {key}")
    registered_projection_shas = _registered_projection_sha_by_arm(
        expected_bindings["arm_projection_manifest_sha256"]
    )

    artifact = _strict_json_object(data, "smoke artifact")
    _exact_keys(
        artifact,
        frozenset(
            {
                "adapter_key_contract",
                "artifact_role",
                "authorization",
                "bindings",
                "cases",
                "completed_at_utc",
                "coverage",
                "e1a1_checkpoint_reload",
                "gates",
                "gpu_smoke_executed",
                "host",
                "identity_adapter_bitwise",
                "optimizer_gate",
                "paired_fixed_row",
                "real_2b_model",
                *recovery_lifecycle(),
                "registered_before_recovery_cohort_started",
                "schema_version",
                "status",
            }
        ),
        "smoke artifact",
    )
    if artifact["schema_version"] != SMOKE_ARTIFACT_SCHEMA_VERSION:
        raise Phase6SmokeValidationError("smoke artifact schema version differs")
    if artifact["artifact_role"] != SMOKE_ARTIFACT_ROLE:
        raise Phase6SmokeValidationError(
            "smoke artifact role must be real_2b_smoke_gate"
        )
    if artifact["status"] != "pass":
        raise Phase6SmokeValidationError("only a passing smoke artifact is admissible")
    _validate_authorization(artifact["authorization"])
    if (
        not isinstance(artifact["completed_at_utc"], str)
        or _UTC_RE.fullmatch(artifact["completed_at_utc"]) is None
    ):
        raise Phase6SmokeValidationError(
            "completed_at_utc must be second-resolution UTC"
        )
    bindings = _validate_bindings(artifact["bindings"], expected_bindings)
    paired_row = _validate_paired_row(artifact["paired_fixed_row"], "paired_fixed_row")

    coverage = _mapping(artifact["coverage"], "coverage")
    _exact_keys(
        coverage,
        frozenset(
            {
                "every_case_gpu_forward",
                "every_case_same_paired_row",
                "mode",
                "observed_cases",
                "required_cases",
            }
        ),
        "coverage",
    )
    expected_cases = _cases_for_mode(coverage["mode"])
    expected_ids = [case_id for case_id, _ in expected_cases]
    if (
        coverage["required_cases"] != expected_ids
        or coverage["observed_cases"] != expected_ids
    ):
        raise Phase6SmokeValidationError(
            "coverage case list is incomplete or reordered"
        )
    _bool(coverage["every_case_gpu_forward"], True, "coverage.every_case_gpu_forward")
    _bool(
        coverage["every_case_same_paired_row"],
        True,
        "coverage.every_case_same_paired_row",
    )

    real_2b = _mapping(artifact["real_2b_model"], "real_2b_model")
    _validate_real_2b_model(real_2b, bindings)
    cases = _list(artifact["cases"], "cases")
    if len(cases) != len(expected_cases):
        raise Phase6SmokeValidationError("smoke case count differs from coverage mode")
    for case, (case_id, factors) in zip(cases, expected_cases):
        _validate_case(
            case,
            expected_id=case_id,
            expected_factors=factors,
            expected_projection_sha256=registered_projection_shas[case_id],
            paired_row=paired_row,
            real_2b_model=real_2b,
        )
    a0_counts = {
        case["model_parameter_count"]
        for case in cases
        if case["factors"]["action_adapter"] is False
    }
    a1_counts = {
        case["model_parameter_count"]
        for case in cases
        if case["factors"]["action_adapter"] is True
    }
    if a0_counts != {real_2b["total_parameter_count"]}:
        raise Phase6SmokeValidationError(
            "A0 total parameter count drifted across cases"
        )
    if len(a1_counts) != 1 or next(iter(a1_counts)) <= real_2b["total_parameter_count"]:
        raise Phase6SmokeValidationError(
            "A1 cases must share one count that adds the identity adapter to A0"
        )

    _validate_identity_gate(artifact["identity_adapter_bitwise"])
    _validate_checkpoint_reload(artifact["e1a1_checkpoint_reload"])
    _validate_adapter_contract(artifact["adapter_key_contract"])

    optimizer = _mapping(artifact["optimizer_gate"], "optimizer_gate")
    _exact_keys(
        optimizer,
        frozenset(
            {
                "blockade_installed_before_runtime_factory",
                "creation_allowed",
                "creation_count",
                "step_allowed",
                "step_count",
            }
        ),
        "optimizer_gate",
    )
    _bool(
        optimizer["blockade_installed_before_runtime_factory"],
        True,
        "optimizer_gate.blockade_installed_before_runtime_factory",
    )
    _bool(optimizer["creation_allowed"], False, "optimizer_gate.creation_allowed")
    _bool(optimizer["step_allowed"], False, "optimizer_gate.step_allowed")
    if optimizer["creation_count"] != 0 or type(optimizer["creation_count"]) is not int:
        raise Phase6SmokeValidationError("optimizer_gate.creation_count must be zero")
    if optimizer["step_count"] != 0 or type(optimizer["step_count"]) is not int:
        raise Phase6SmokeValidationError("optimizer_gate.step_count must be zero")

    host = _mapping(artifact["host"], "host")
    _exact_keys(
        host,
        frozenset(
            {
                "cuda_capability",
                "cuda_device_name",
                "cuda_device_uuid",
                "cuda_driver_version",
                "cuda_runtime_version",
                "cuda_total_memory_bytes",
                "device_index",
                "hostname",
                "multi_processor_count",
                "torch_version",
                "world_size",
            }
        ),
        "host",
    )
    _nonempty_string(host["hostname"], "host.hostname")
    device_name = _nonempty_string(host["cuda_device_name"], "host.cuda_device_name")
    if "H200" not in device_name.upper():
        raise Phase6SmokeValidationError("real-2B smoke must run on an H200")
    device_uuid = _nonempty_string(host["cuda_device_uuid"], "host.cuda_device_uuid")
    if not device_uuid.startswith("GPU-"):
        raise Phase6SmokeValidationError(
            "host.cuda_device_uuid must be an NVIDIA GPU UUID"
        )
    _nonempty_string(host["cuda_driver_version"], "host.cuda_driver_version")
    capability = _list(host["cuda_capability"], "host.cuda_capability")
    if capability != [9, 0]:
        raise Phase6SmokeValidationError("H200 CUDA capability must be [9,0]")
    total_memory = _positive_int(host["cuda_total_memory_bytes"], "host CUDA memory")
    if total_memory <= PEAK_RESERVED_LIMIT_BYTES:
        raise Phase6SmokeValidationError(
            "device has no headroom above the 130-GiB stop limit"
        )
    _positive_int(host["multi_processor_count"], "host.multi_processor_count")
    _nonnegative_int(host["device_index"], "host.device_index")
    if host["world_size"] != 1 or type(host["world_size"]) is not int:
        raise Phase6SmokeValidationError("host.world_size must be 1")
    _nonempty_string(host["torch_version"], "host.torch_version")
    _nonempty_string(host["cuda_runtime_version"], "host.cuda_runtime_version")

    gates = _mapping(artifact["gates"], "gates")
    _exact_keys(gates, _GATE_KEYS, "gates")
    for key in sorted(_GATE_KEYS):
        _bool(gates[key], True, f"gates.{key}")
    _bool(artifact["gpu_smoke_executed"], True, "gpu_smoke_executed")
    _validate_recovery_lifecycle_fields(artifact, "smoke artifact lifecycle")
    _bool(
        artifact["registered_before_recovery_cohort_started"],
        True,
        "registered_before_recovery_cohort_started",
    )
    return artifact


__all__ = [
    "MINIMUM_FULL_CHECKPOINT_SIZE_BYTES",
    "MINIMUM_REAL_2B_VIDEO_PARAMETER_COUNT",
    "PEAK_RESERVED_LIMIT_BYTES",
    "Phase6SmokeValidationError",
    "RUNTIME_SUPPORT_API_VERSION",
    "SMOKE_ARTIFACT_ROLE",
    "SMOKE_ARTIFACT_SCHEMA_VERSION",
    "SMOKE_AUTHORIZATION",
    "canonical_json_bytes",
    "validate_phase6_smoke_artifact_bytes",
]
