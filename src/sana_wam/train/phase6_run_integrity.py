"""Torch-free per-step metrics and run-integrity artifacts for Phase-6."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    expansion_support_contract,
)
from sana_wam.train.phase6_recovery import recovery_lifecycle


STEP_SCHEMA_VERSION = "sana-phase6-step-metrics-v1"
SUMMARY_SCHEMA_VERSION = "sana-phase6-metrics-summary-v1"
COMMON_INPUT_TRACE_CONTRACT = "sana-phase6-common-prepared-noise-mask-v1"
CROSS_ARM_TRACE_SCHEMA_VERSION = "sana-phase6-cross-arm-input-trace-v1"
FROZEN_TENSOR_VERIFICATION_SCHEMA_VERSION = (
    "sana-phase6-frozen-action-proprio-verification-v1"
)
RESOLVED_CONFIG_VERIFICATION_SCHEMA_VERSION = (
    "sana-phase6-resolved-config-verification-v2"
)
RUN_COMPLETION_SCHEMA_VERSION = "sana-phase6-run-completion-v1"
PLAN_ARTIFACT_SCHEMA_VERSION = "sana-phase6-task-plan-artifact-v2"
PLAN_SCHEMA_VERSION = "sana-phase6-task-plan-v1"
IDENTITY_SCHEMA_VERSION = "sana-phase6-identity-sequence-v1"

EXPECTED_STEPS = 504
EXPECTED_TASKS = 42
EXPECTED_CYCLES = 12
EXPECTED_SIGMA_EXPOSURES = 4
EXPANSION_RATE_LIMIT = 8.0
RELATIVE_EPSILON = 1.0e-4

ARM_FACTORS: dict[str, dict[str, bool]] = {
    "T0_E0A0": {"T": False, "E": False, "A": False},
    "T1_E0A0": {"T": True, "E": False, "A": False},
    "T1_E1A0": {"T": True, "E": True, "A": False},
    "T1_E0A1": {"T": True, "E": False, "A": True},
    "T1_E1A1": {"T": True, "E": True, "A": True},
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX = "action_video_memory_adapter."
_ACTION_SIGMAS = (1.0, 0.9, 0.5)
_ADAPTER_STATE_KEYS = [
    "action_video_memory_adapter.k_left",
    "action_video_memory_adapter.k_right",
    "action_video_memory_adapter.v_left",
    "action_video_memory_adapter.v_right",
    "action_video_memory_adapter.z_logits",
]
_PROPRIO_STATE_KEYS = [
    "proprio_action_embed.bias",
    "proprio_action_embed.weight",
    "proprio_encoder.bias",
    "proprio_encoder.weight",
    "proprio_video_embed.bias",
    "proprio_video_embed.weight",
]
_PROPRIO_STATE_PREFIXES = (
    "proprio_action_embed.",
    "proprio_encoder.",
    "proprio_video_embed.",
)
_TRAINING_AUTHORIZATION_KEYS = frozenset(
    {
        "arm",
        "arm_config_projection_sha256",
        "closed_loop_allowed",
        "factors",
        "operation",
        "optimizer_training_allowed",
        "real_2b_smoke_completed",
        "reference_gpu_forward_completed",
        "resume_allowed",
        "run_directory",
        "run_id",
    }
)
_SIGMA_KEYS = {1.0: "1.0", 0.9: "0.9", 0.5: "0.5"}
_ROW_SEED_DOMAINS = frozenset(
    {
        "video-noise",
        "action-noise",
        "expansion-noise",
        "expansion-direction",
        "reference-query",
        "prompt-choice",
    }
)
_ALL_SEED_DOMAINS = [
    "task-order",
    "sample-choice",
    "action-sigma",
    "video-noise",
    "action-noise",
    "expansion-noise",
    "expansion-direction",
    "reference-query",
    "prompt-choice",
]
_ARTIFACT_KEYS = frozenset(
    {
        "artifact_schema_version",
        "expansion_eligibility_amendment_sha256",
        "expansion_support_contract",
        "expansion_support_contract_sha256",
        "identity_sha256",
        "plan",
        "plan_sha256",
    }
)
_PLAN_KEYS = frozenset(
    {
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
)
_PLAN_ROW_KEYS = frozenset(
    {
        "action_sigma",
        "cycle",
        "domain_seeds",
        "global_step",
        "identity",
        "position_in_cycle",
    }
)
_IDENTITY_KEYS = frozenset(
    {
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
)
_MEASUREMENT_KEYS = frozenset(
    {
        "action",
        "common_input_trace_sha256",
        "expansion",
        "gradients",
        "learning_rates",
        "memory",
        "total_loss",
        "video_on_path_loss",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "action_reference_sha256",
        "action_stats_sha256",
        "arm_config_sha256",
        "arm_config_projection_sha256",
        "dataset_contract_sha256",
        "initial_checkpoint_sha256",
        "launch_manifest_sha256",
        "phase1_checkpoint_sha256",
        "preflight_report_sha256",
        "preflight_request_sha256",
        "runtime_support_manifest_sha256",
        "smoke_artifact_sha256",
        "source_manifest_sha256",
        "spot_verification_sha256",
    }
)
_STEP_KEYS = frozenset(
    {
        "action",
        "action_sigma",
        "arm",
        "cycle",
        "common_input_trace_sha256",
        "domain_seeds",
        "expansion",
        "factors",
        "global_step",
        "gradients",
        "identity_sha256",
        "learning_rates",
        "memory",
        "plan_row_sha256",
        "plan_sha256",
        "position_in_cycle",
        "preflight_purpose",
        "preflight_report_sha256",
        "provenance_sha256",
        "prompt_utf8_sha256",
        "record_type",
        "run_id",
        "schema_version",
        "task_name",
        "total_loss",
        "video_on_path_loss",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "action_student_error_distribution",
        "arm",
        "common_input_trace",
        "coverage",
        "factors",
        "identity_sha256",
        "integrity",
        "metrics",
        "plan_artifact_sha256",
        "plan_sha256",
        "primary_outputs",
        "preflight_purpose",
        "preflight_report_snapshot",
        "provenance",
        "record_type",
        "run_id",
        "schema_version",
    }
)
_PRIMARY_OUTPUT_ROLES = (
    "step504_checkpoint",
    "saved_arm_config",
    "resolved_config_verification_manifest",
    "action_stats",
    "frozen_tensor_verification_manifest",
)
_PRIMARY_OUTPUT_BASENAMES = {
    "action_stats": "action_stats.npy",
    "frozen_tensor_verification_manifest": ("frozen_action_proprio_verification.json"),
    "resolved_config_verification_manifest": ("resolved_config_verification.json"),
    "saved_arm_config": "resolved_config.yaml",
    "step504_checkpoint": "checkpoint_step_504.safetensors",
}


class RunIntegrityError(ValueError):
    """Raised when a metrics artifact cannot prove the frozen run contract."""


class NonFiniteGradientError(RunIntegrityError):
    """Raised after durably recording a non-finite-gradient stop row."""


class NonFiniteModelParametersError(RunIntegrityError):
    """Raised after recording non-finite trainable model parameters post-step."""


class NonFiniteOptimizerMasterParametersError(RunIntegrityError):
    """Raised after recording non-finite FP32 optimizer masters post-step."""


class NonFiniteOptimizerStateError(RunIntegrityError):
    """Raised after recording non-finite optimizer state post-step."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, sorted UTF-8 JSON with one trailing newline."""

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
        raise RunIntegrityError(f"value is not canonical JSON: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunIntegrityError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RunIntegrityError(f"non-finite JSON number: {value}")
    return result


def _strict_json_loads(data: bytes, name: str) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_parse_finite_float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RunIntegrityError(f"non-finite JSON value: {value}")
            ),
        )
    except RunIntegrityError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError, ValueError) as exc:
        raise RunIntegrityError(f"invalid {name} JSON: {exc}") from exc


def _require_exact_keys(value: Any, expected: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise RunIntegrityError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != set(expected):
        raise RunIntegrityError(
            f"{name} keys differ; missing={sorted(set(expected) - actual)!r}, "
            f"extra={sorted(actual - set(expected))!r}"
        )


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RunIntegrityError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RunIntegrityError(f"{name} must be a JSON boolean")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RunIntegrityError(f"{name} must be an integer >= {minimum}")
    return value


def _require_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise RunIntegrityError(f"{name} must be a finite JSON float")
    if minimum is not None and value < minimum:
        raise RunIntegrityError(f"{name} must be >= {minimum}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RunIntegrityError(f"{name} must be a lowercase SHA256 digest")
    return value


def _require_exact_typed_value(value: Any, expected: Any, name: str) -> None:
    if type(value) is not type(expected):
        raise RunIntegrityError(f"{name} has an inexact JSON type")
    if isinstance(expected, dict):
        _require_exact_keys(value, frozenset(expected), name)
        for key, child in expected.items():
            _require_exact_typed_value(value[key], child, f"{name}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise RunIntegrityError(f"{name} has an unexpected length")
        for index, (child, expected_child) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_exact_typed_value(child, expected_child, f"{name}[{index}]")
    elif value != expected:
        raise RunIntegrityError(f"{name} differs from its frozen value")


def _require_run_id(value: Any) -> str:
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise RunIntegrityError("run_id has an invalid spelling")
    return value


def _validate_provenance(value: Any) -> dict[str, str]:
    _require_exact_keys(value, _PROVENANCE_KEYS, "provenance")
    result = {}
    for key in sorted(_PROVENANCE_KEYS):
        result[key] = _require_sha256(value[key], f"provenance.{key}")
    return result


def _require_positive_zero(value: Any, name: str) -> None:
    if type(value) is not float or value != 0.0 or math.copysign(1.0, value) != 1.0:
        raise RunIntegrityError(f"{name} must be exact positive 0.0")


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=5.0e-5, abs_tol=1.0e-7)


def _task_list_sha256(tasks: Sequence[str]) -> str:
    payload = "".join(f"{task}\n" for task in sorted(tasks)).encode("utf-8")
    return sha256(payload).hexdigest()


def _identity_sha256(plan: Mapping[str, Any]) -> str:
    payload = {
        "holdout_task_sha256": plan["holdout_task_sha256"],
        "ordered_identities": [
            {
                "episode_path": row["identity"]["episode_path"],
                "prompt": row["identity"]["prompt"],
                "start_frame": row["identity"]["start_frame"],
                "task_name": row["identity"]["task_name"],
            }
            for row in plan["rows"]
        ],
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "train_task_sha256": plan["train_task_sha256"],
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_absolute_path(path: str | os.PathLike[str], name: str) -> str:
    value = _require_string(os.fspath(path), name)
    if not os.path.isabs(value):
        raise RunIntegrityError(f"{name} must be an absolute path")
    return os.path.abspath(value)


@dataclass(frozen=True)
class Phase6ValidatedLaunchContext:
    """Immutable launch facts returned by the torch-free launch authorizer.

    The context deliberately remains outside the scientific config.  The
    authorizer owns ticket/manifest validation; Trainer independently hashes
    the three pinned inputs and cross-checks these facts against preflight and
    the loaded config before it creates CUDA state.
    """

    arm: str
    run_id: str
    run_directory: str
    factors: tuple[tuple[str, bool], ...]
    arm_config_projection_sha256: str
    input_arm_config_path: str
    input_arm_config_sha256: str
    launch_manifest_path: str
    launch_manifest_sha256: str
    ticket_path: str
    ticket_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase6ValidatedLaunchContext":
        _require_exact_keys(
            value,
            frozenset(
                {
                    "arm",
                    "authorization",
                    "config_path",
                    "config_sha256",
                    "launch_manifest_path",
                    "launch_manifest_sha256",
                    "output_directory",
                    "run_id",
                    "ticket_path",
                    "ticket_sha256",
                }
            ),
            "validated launch context",
        )
        authorization = value["authorization"]
        _require_exact_keys(
            authorization,
            _TRAINING_AUTHORIZATION_KEYS,
            "validated launch authorization",
        )
        arm = _require_string(authorization["arm"], "authorization.arm")
        if arm not in ARM_FACTORS:
            raise RunIntegrityError(f"unsupported Phase-6 arm: {arm!r}")
        factors = authorization["factors"]
        _require_exact_keys(
            factors, frozenset({"A", "E", "T"}), "authorization.factors"
        )
        normalized_factors = {
            key: _require_bool(factors[key], f"authorization.factors.{key}")
            for key in ("A", "E", "T")
        }
        if normalized_factors != ARM_FACTORS[arm]:
            raise RunIntegrityError("launch authorization arm/factors differ")
        for key, expected in (
            ("operation", "phase6_training"),
            ("reference_gpu_forward_completed", True),
            ("real_2b_smoke_completed", True),
            ("optimizer_training_allowed", True),
            ("closed_loop_allowed", False),
            ("resume_allowed", False),
        ):
            if (
                type(authorization[key]) is not type(expected)
                or authorization[key] != expected
            ):
                raise RunIntegrityError(
                    f"launch authorization {key} must be exact {expected!r}"
                )
        run_id = _require_run_id(authorization["run_id"])
        run_directory = _require_absolute_path(
            authorization["run_directory"], "authorization.run_directory"
        )
        if os.path.basename(run_directory) != run_id:
            raise RunIntegrityError(
                "authorized run directory basename differs from run_id"
            )
        if value["arm"] != arm:
            raise RunIntegrityError("top-level launch arm differs from authorization")
        if value["run_id"] != run_id:
            raise RunIntegrityError(
                "top-level launch run_id differs from authorization"
            )
        if (
            _require_absolute_path(value["output_directory"], "output_directory")
            != run_directory
        ):
            raise RunIntegrityError(
                "top-level launch output_directory differs from authorization"
            )
        projection_sha = _require_sha256(
            authorization["arm_config_projection_sha256"],
            "authorization.arm_config_projection_sha256",
        )

        pin_fields = {
            "input_arm_config": ("config_path", "config_sha256"),
            "launch_manifest": (
                "launch_manifest_path",
                "launch_manifest_sha256",
            ),
            "launch_ticket": ("ticket_path", "ticket_sha256"),
        }
        pins = {
            role: (
                _require_absolute_path(value[path_key], f"{path_key}"),
                _require_sha256(value[sha_key], f"{sha_key}"),
            )
            for role, (path_key, sha_key) in pin_fields.items()
        }
        if len({path for path, _sha in pins.values()}) != len(pins):
            raise RunIntegrityError("launch context reuses one path for multiple roles")
        return cls(
            arm=arm,
            run_id=run_id,
            run_directory=run_directory,
            factors=tuple((key, normalized_factors[key]) for key in ("A", "E", "T")),
            arm_config_projection_sha256=projection_sha,
            input_arm_config_path=pins["input_arm_config"][0],
            input_arm_config_sha256=pins["input_arm_config"][1],
            launch_manifest_path=pins["launch_manifest"][0],
            launch_manifest_sha256=pins["launch_manifest"][1],
            ticket_path=pins["launch_ticket"][0],
            ticket_sha256=pins["launch_ticket"][1],
        )

    @property
    def factor_mapping(self) -> dict[str, bool]:
        return dict(self.factors)

    @property
    def authorization(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "arm_config_projection_sha256": self.arm_config_projection_sha256,
            "closed_loop_allowed": False,
            "factors": self.factor_mapping,
            "operation": "phase6_training",
            "optimizer_training_allowed": True,
            "real_2b_smoke_completed": True,
            "reference_gpu_forward_completed": True,
            "resume_allowed": False,
            "run_directory": self.run_directory,
            "run_id": self.run_id,
        }

    def revalidate_pinned_files(self) -> None:
        for path, expected, name in (
            (
                self.input_arm_config_path,
                self.input_arm_config_sha256,
                "input arm config",
            ),
            (
                self.launch_manifest_path,
                self.launch_manifest_sha256,
                "launch manifest",
            ),
            (self.ticket_path, self.ticket_sha256, "launch ticket"),
        ):
            observed, _data = _read_stable_file(path, name)
            if not hmac.compare_digest(observed["sha256"], expected):
                raise RunIntegrityError(f"{name} changed after launch authorization")

    def read_input_arm_config_bytes(self) -> bytes:
        observed, data = _read_stable_file(
            self.input_arm_config_path, "input arm config"
        )
        if not hmac.compare_digest(observed["sha256"], self.input_arm_config_sha256):
            raise RunIntegrityError(
                "input arm config changed after launch authorization"
            )
        return data


def _observe_stable_file(
    path: str | os.PathLike[str],
    name: str,
    *,
    capture: bool,
) -> tuple[dict[str, Any], bytes | None]:
    absolute = _require_absolute_path(path, f"{name}.path")
    try:
        resolved_before = os.path.realpath(absolute)
        lstat_before = os.lstat(absolute)
        stat_before = os.stat(absolute)
        if not stat.S_ISREG(stat_before.st_mode):
            raise RunIntegrityError(f"{name} is not a regular file")
        with open(absolute, "rb") as stream:
            descriptor_before = os.fstat(stream.fileno())
            if (descriptor_before.st_dev, descriptor_before.st_ino) != (
                stat_before.st_dev,
                stat_before.st_ino,
            ):
                raise RunIntegrityError(f"{name} changed before reading")
            digest = sha256()
            chunks: list[bytes] | None = [] if capture else None
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                if chunks is not None:
                    chunks.append(block)
            descriptor_after = os.fstat(stream.fileno())
        resolved_after = os.path.realpath(absolute)
        lstat_after = os.lstat(absolute)
        stat_after = os.stat(absolute)
    except RunIntegrityError:
        raise
    except OSError as exc:
        raise RunIntegrityError(f"cannot read {name}: {absolute}: {exc}") from exc
    before = (
        resolved_before,
        _stat_signature(lstat_before),
        _stat_signature(stat_before),
        _stat_signature(descriptor_before),
    )
    after = (
        resolved_after,
        _stat_signature(lstat_after),
        _stat_signature(stat_after),
        _stat_signature(descriptor_after),
    )
    if before != after:
        raise RunIntegrityError(f"{name} changed while reading")
    data = b"".join(chunks) if chunks is not None else None
    return (
        {
            "path": absolute,
            "realpath": resolved_before,
            "size_bytes": stat_before.st_size,
            "sha256": digest.hexdigest(),
        },
        data,
    )


def _read_stable_file(
    path: str | os.PathLike[str], name: str
) -> tuple[dict[str, Any], bytes]:
    observed, data = _observe_stable_file(path, name, capture=True)
    if data is None:
        raise AssertionError("captured stable-file bytes are absent")
    return observed, data


def read_stable_pinned_file(
    path: str | os.PathLike[str], expected_sha256: str, name: str
) -> bytes:
    """Stable-read one regular file and require its caller-pinned digest."""

    expected = _require_sha256(expected_sha256, f"{name}.sha256")
    observed, data = _read_stable_file(path, name)
    if not hmac.compare_digest(observed["sha256"], expected):
        raise RunIntegrityError(f"{name} SHA256 differs from its frozen pin")
    return data


def _validate_training_preflight_report(
    data: bytes,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    if sha256(data).hexdigest() != provenance["preflight_report_sha256"]:
        raise RunIntegrityError("preflight success-report SHA mismatch")
    value = _strict_json_loads(data, "preflight success report")
    if not isinstance(value, Mapping) or data != canonical_json_bytes(value):
        raise RunIntegrityError("preflight success report must be canonical JSON")
    required = {
        *recovery_lifecycle(),
        "purpose",
        "reference_precompute_completed",
        "registered_before_recovery_cohort_started",
        "request_sha256",
        "smoke_completed",
        "status",
    }
    if not required.issubset(value):
        raise RunIntegrityError(
            "preflight success report lacks training-purpose lifecycle fields"
        )
    if value["purpose"] != "training" or value["status"] != "pass":
        raise RunIntegrityError("only a passing training-purpose report is valid")
    if value["request_sha256"] != provenance["preflight_request_sha256"]:
        raise RunIntegrityError("preflight report/request provenance differs")
    if value["reference_precompute_completed"] is not True:
        raise RunIntegrityError("training preflight must follow reference precompute")
    if value["smoke_completed"] is not True:
        raise RunIntegrityError("training preflight must follow the real-2B smoke")
    for key, expected in recovery_lifecycle().items():
        observed = value[key]
        if type(observed) is not type(expected) or observed != expected:
            raise RunIntegrityError(f"preflight recovery lifecycle {key} differs")
    if value["registered_before_recovery_cohort_started"] is not True:
        raise RunIntegrityError("preflight report recovery registration differs")
    return dict(value)


def _validate_plan_row(
    row: Any,
    *,
    expected_step: int,
    train_tasks: set[str],
) -> None:
    name = f"plan.rows[{expected_step - 1}]"
    _require_exact_keys(row, _PLAN_ROW_KEYS, name)
    if (
        _require_int(row["global_step"], f"{name}.global_step", minimum=1)
        != expected_step
    ):
        raise RunIntegrityError(f"{name}.global_step is not contiguous")
    expected_cycle = (expected_step - 1) // EXPECTED_TASKS
    expected_position = (expected_step - 1) % EXPECTED_TASKS
    if (
        _require_int(row["cycle"], f"{name}.cycle") != expected_cycle
        or _require_int(row["position_in_cycle"], f"{name}.position_in_cycle")
        != expected_position
    ):
        raise RunIntegrityError(f"{name} cycle/position differs from global_step")
    sigma = _require_float(row["action_sigma"], f"{name}.action_sigma")
    if sigma not in _ACTION_SIGMAS:
        raise RunIntegrityError(f"{name}.action_sigma is unsupported")
    seeds = row["domain_seeds"]
    _require_exact_keys(seeds, _ROW_SEED_DOMAINS, f"{name}.domain_seeds")
    for domain, seed in seeds.items():
        value = _require_int(seed, f"{name}.domain_seeds[{domain}]")
        if value >= 2**64:
            raise RunIntegrityError(f"{name}.domain_seeds[{domain}] exceeds uint64")
    identity = row["identity"]
    _require_exact_keys(identity, _IDENTITY_KEYS, f"{name}.identity")
    task_name = _require_string(identity["task_name"], f"{name}.identity.task_name")
    if task_name not in train_tasks:
        raise RunIntegrityError(f"{name} references a non-training task")
    _require_int(identity["dataset_index"], f"{name}.identity.dataset_index")
    _require_int(identity["episode_index"], f"{name}.identity.episode_index")
    _require_int(identity["start_frame"], f"{name}.identity.start_frame")
    episode_path = _require_string(
        identity["episode_path"], f"{name}.identity.episode_path"
    )
    if not (episode_path.startswith("/") or os.path.isabs(episode_path)):
        raise RunIntegrityError(f"{name}.identity.episode_path must be absolute")
    _require_string(identity["prompt"], f"{name}.identity.prompt")
    if (
        identity["source_dataset"] != "RoboTwin"
        or identity["source_kind"] != "ordinary_expert"
        or identity["source_variant"] != "clean_50"
    ):
        raise RunIntegrityError(f"{name} source contract differs")


def _validate_plan(value: Any) -> None:
    _require_exact_keys(value, _PLAN_KEYS, "plan")
    if value["schema_version"] != PLAN_SCHEMA_VERSION:
        raise RunIntegrityError("plan schema differs")
    if _require_int(value["optimizer_steps"], "plan.optimizer_steps") != EXPECTED_STEPS:
        raise RunIntegrityError("plan optimizer_steps must be 504")
    if _require_int(value["task_cycles"], "plan.task_cycles") != EXPECTED_CYCLES:
        raise RunIntegrityError("plan task_cycles must be 12")
    if (
        _require_int(value["sigma_exposures_per_task"], "plan.sigma_exposures_per_task")
        != EXPECTED_SIGMA_EXPOSURES
    ):
        raise RunIntegrityError("plan sigma exposures must be 4 per task")
    if (
        type(value["action_sigmas"]) is not list
        or any(type(sigma) is not float for sigma in value["action_sigmas"])
        or value["action_sigmas"] != list(_ACTION_SIGMAS)
    ):
        raise RunIntegrityError("plan action_sigmas differ")
    if value["seed_domains"] != _ALL_SEED_DOMAINS:
        raise RunIntegrityError("plan seed_domains differ")
    _require_int(value["protocol_seed"], "plan.protocol_seed", minimum=1)
    source = value["source_contract"]
    _require_exact_keys(
        source, frozenset({"dataset", "kind", "variant"}), "plan.source_contract"
    )
    if source != {
        "dataset": "RoboTwin",
        "kind": "ordinary_expert",
        "variant": "clean_50",
    }:
        raise RunIntegrityError("plan source_contract differs")

    train_tasks = value["train_tasks"]
    holdout_tasks = value["holdout_tasks"]
    if (
        not isinstance(train_tasks, list)
        or len(train_tasks) != EXPECTED_TASKS
        or train_tasks != sorted(train_tasks)
        or len(set(train_tasks)) != EXPECTED_TASKS
        or any(not isinstance(task, str) or not task for task in train_tasks)
    ):
        raise RunIntegrityError("plan train_tasks must be 42 sorted unique strings")
    if (
        not isinstance(holdout_tasks, list)
        or len(holdout_tasks) != 8
        or holdout_tasks != sorted(holdout_tasks)
        or len(set(holdout_tasks)) != 8
        or any(not isinstance(task, str) or not task for task in holdout_tasks)
    ):
        raise RunIntegrityError("plan holdout_tasks must be 8 sorted unique strings")
    train_sha = _require_sha256(value["train_task_sha256"], "train_task_sha256")
    holdout_sha = _require_sha256(value["holdout_task_sha256"], "holdout_task_sha256")
    if _task_list_sha256(train_tasks) != train_sha:
        raise RunIntegrityError("plan train-task SHA differs")
    if _task_list_sha256(holdout_tasks) != holdout_sha:
        raise RunIntegrityError("plan holdout-task SHA differs")

    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_STEPS:
        raise RunIntegrityError("plan must contain exactly 504 rows")
    task_set = set(train_tasks)
    dataset_indices: set[int] = set()
    cycle_tasks: dict[int, set[str]] = defaultdict(set)
    task_sigmas: dict[str, Counter[float]] = defaultdict(Counter)
    for expected_step, row in enumerate(rows, start=1):
        _validate_plan_row(row, expected_step=expected_step, train_tasks=task_set)
        identity = row["identity"]
        dataset_index = identity["dataset_index"]
        if dataset_index in dataset_indices:
            raise RunIntegrityError("plan reuses a dataset index")
        dataset_indices.add(dataset_index)
        task = identity["task_name"]
        cycle = row["cycle"]
        if task in cycle_tasks[cycle]:
            raise RunIntegrityError("plan repeats a task within one cycle")
        cycle_tasks[cycle].add(task)
        task_sigmas[task][row["action_sigma"]] += 1
    if set(cycle_tasks) != set(range(EXPECTED_CYCLES)) or any(
        tasks != task_set for tasks in cycle_tasks.values()
    ):
        raise RunIntegrityError("plan does not contain every task once per cycle")
    expected_sigma_counts = Counter(
        {sigma: EXPECTED_SIGMA_EXPOSURES for sigma in _ACTION_SIGMAS}
    )
    if any(task_sigmas[task] != expected_sigma_counts for task in train_tasks):
        raise RunIntegrityError("plan sigma exposure counts differ by task")


@dataclass(frozen=True)
class Phase6PlanBinding:
    """Validated immutable-enough view of the exact Phase-6 504-row plan."""

    artifact_sha256: str
    plan_sha256: str
    identity_sha256: str
    train_tasks: tuple[str, ...]
    _rows: tuple[dict[str, Any], ...]
    _row_bytes: tuple[bytes, ...]

    @classmethod
    def from_artifact_bytes(
        cls,
        data: bytes,
        *,
        expected_artifact_sha256: str,
        expected_plan_sha256: str,
        expected_identity_sha256: str,
    ) -> "Phase6PlanBinding":
        if not isinstance(data, bytes):
            raise RunIntegrityError("plan artifact must be bytes")
        artifact_sha = _require_sha256(
            expected_artifact_sha256, "expected_artifact_sha256"
        )
        plan_sha = _require_sha256(expected_plan_sha256, "expected_plan_sha256")
        identity_sha = _require_sha256(
            expected_identity_sha256, "expected_identity_sha256"
        )
        if not hmac.compare_digest(sha256(data).hexdigest(), artifact_sha):
            raise RunIntegrityError("plan artifact SHA mismatch")
        value = _strict_json_loads(data, "plan artifact")
        _require_exact_keys(value, _ARTIFACT_KEYS, "plan artifact")
        if data != canonical_json_bytes(value):
            raise RunIntegrityError("plan artifact is not canonical JSON")
        if value["artifact_schema_version"] != PLAN_ARTIFACT_SCHEMA_VERSION:
            raise RunIntegrityError("plan artifact schema differs")
        if (
            value["expansion_eligibility_amendment_sha256"]
            != EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
            or value["expansion_support_contract_sha256"]
            != EXPANSION_SUPPORT_CONTRACT_SHA256
        ):
            raise RunIntegrityError("plan expansion provenance differs")
        expected_support_contract = expansion_support_contract()
        _require_exact_typed_value(
            value["expansion_support_contract"],
            expected_support_contract,
            "plan expansion_support_contract",
        )
        if (
            sha256(
                canonical_json_bytes(value["expansion_support_contract"])
            ).hexdigest()
            != EXPANSION_SUPPORT_CONTRACT_SHA256
        ):
            raise RunIntegrityError("plan expansion-support contract SHA differs")
        _validate_plan(value["plan"])
        observed_plan_sha = sha256(canonical_json_bytes(value["plan"])).hexdigest()
        if value["plan_sha256"] != plan_sha or not hmac.compare_digest(
            observed_plan_sha, plan_sha
        ):
            raise RunIntegrityError("embedded plan SHA mismatch")
        observed_identity_sha = _identity_sha256(value["plan"])
        if value["identity_sha256"] != identity_sha or not hmac.compare_digest(
            observed_identity_sha, identity_sha
        ):
            raise RunIntegrityError("embedded identity SHA mismatch")
        rows = tuple(deepcopy(value["plan"]["rows"]))
        return cls(
            artifact_sha256=artifact_sha,
            plan_sha256=plan_sha,
            identity_sha256=identity_sha,
            train_tasks=tuple(value["plan"]["train_tasks"]),
            _rows=rows,
            _row_bytes=tuple(canonical_json_bytes(row) for row in rows),
        )

    @classmethod
    def from_artifact_path(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_artifact_sha256: str,
        expected_plan_sha256: str,
        expected_identity_sha256: str,
    ) -> "Phase6PlanBinding":
        _observed, data = _read_stable_file(path, "plan artifact")
        return cls.from_artifact_bytes(
            data,
            expected_artifact_sha256=expected_artifact_sha256,
            expected_plan_sha256=expected_plan_sha256,
            expected_identity_sha256=expected_identity_sha256,
        )

    def expected_row(self, global_step: int) -> dict[str, Any]:
        step = _require_int(global_step, "global_step", minimum=1)
        if step > EXPECTED_STEPS:
            raise RunIntegrityError("global_step exceeds 504")
        return deepcopy(self._rows[step - 1])

    def expected_runtime_row(self, global_step: int) -> dict[str, Any]:
        """Return the plan row as attached to one materialized runtime sample."""

        row = self.expected_row(global_step)
        row["identity_sha256"] = self.identity_sha256
        row["plan_sha256"] = self.plan_sha256
        return row


def _validate_measurements(value: Any, factors: Mapping[str, bool]) -> bool:
    _require_exact_keys(value, _MEASUREMENT_KEYS, "measurements")
    _require_sha256(
        value["common_input_trace_sha256"],
        "measurements.common_input_trace_sha256",
    )
    total = _require_float(value["total_loss"], "total_loss", minimum=0.0)
    video = _require_float(
        value["video_on_path_loss"], "video_on_path_loss", minimum=0.0
    )

    expansion = value["expansion"]
    _require_exact_keys(
        expansion,
        frozenset({"active", "loss", "rate", "violation"}),
        "expansion",
    )
    expansion_active = _require_bool(expansion["active"], "expansion.active")
    if expansion_active is not factors["E"]:
        raise RunIntegrityError("expansion.active differs from the E factor")
    if expansion_active:
        rate = _require_float(expansion["rate"], "expansion.rate")
        expansion_loss = _require_float(
            expansion["loss"], "expansion.loss", minimum=0.0
        )
        violation = _require_bool(expansion["violation"], "expansion.violation")
        if violation is not (rate > EXPANSION_RATE_LIMIT):
            raise RunIntegrityError("expansion.violation differs from rate > 8")
        expected_loss = (
            max(
                (rate - EXPANSION_RATE_LIMIT) / EXPANSION_RATE_LIMIT,
                0.0,
            )
            ** 2
        )
        if not _close(expansion_loss, expected_loss):
            raise RunIntegrityError("expansion loss differs from the frozen formula")
    else:
        if expansion["rate"] is not None or expansion["violation"] is not None:
            raise RunIntegrityError(
                "E-off expansion measurements must be null, never numeric zero"
            )
        _require_positive_zero(expansion["loss"], "E-off expansion.loss")
        expansion_loss = 0.0

    action = value["action"]
    _require_exact_keys(
        action,
        frozenset(
            {
                "active",
                "non_regression_loss",
                "reference_error",
                "relative_excess",
                "student_error",
            }
        ),
        "action",
    )
    action_active = _require_bool(action["active"], "action.active")
    if action_active is not factors["A"]:
        raise RunIntegrityError("action.active differs from the A factor")
    student = _require_float(
        action["student_error"], "action.student_error", minimum=0.0
    )
    if action_active:
        reference = _require_float(
            action["reference_error"], "action.reference_error", minimum=0.0
        )
        relative = _require_float(action["relative_excess"], "action.relative_excess")
        nr_loss = _require_float(
            action["non_regression_loss"],
            "action.non_regression_loss",
            minimum=0.0,
        )
        expected_relative = (student - reference) / (reference + RELATIVE_EPSILON)
        if not _close(relative, expected_relative):
            raise RunIntegrityError("action relative excess differs from formula")
        if not _close(nr_loss, max(relative, 0.0) ** 2):
            raise RunIntegrityError("action NR loss differs from formula")
    else:
        for key in ("reference_error", "relative_excess"):
            if action[key] is not None:
                raise RunIntegrityError(
                    f"A-off action.{key} must be null, never numeric zero"
                )
        _require_positive_zero(
            action["non_regression_loss"], "A-off action.non_regression_loss"
        )
        nr_loss = 0.0

    learning_rates = value["learning_rates"]
    _require_exact_keys(
        learning_rates,
        frozenset({"action_adapter", "video"}),
        "learning_rates",
    )
    _require_float(learning_rates["video"], "learning_rates.video", minimum=0.0)
    if factors["A"]:
        _require_float(
            learning_rates["action_adapter"],
            "learning_rates.action_adapter",
            minimum=0.0,
        )
    elif learning_rates["action_adapter"] is not None:
        raise RunIntegrityError(
            "A-off action-adapter LR must be null, never numeric zero"
        )

    gradients = value["gradients"]
    _require_exact_keys(
        gradients,
        frozenset(
            {
                "action_trainable_param_count",
                "adapter_gradient_connected",
                "adapter_grad_norm",
                "finite",
                "global_norm",
                "model_parameters_finite",
                "nr_gradient_target_adapter_only",
                "optimizer_master_parameters_finite",
                "optimizer_state_finite",
                "proprio_trainable_param_count",
                "video_grad_norm",
            }
        ),
        "gradients",
    )
    if (
        type(gradients["action_trainable_param_count"]) is not int
        or gradients["action_trainable_param_count"] != 0
    ):
        raise RunIntegrityError("action trainable parameter count must be exact zero")
    if (
        type(gradients["proprio_trainable_param_count"]) is not int
        or gradients["proprio_trainable_param_count"] != 0
    ):
        raise RunIntegrityError("proprio trainable parameter count must be exact zero")
    if factors["A"]:
        if gradients["adapter_gradient_connected"] is not True:
            raise RunIntegrityError("A-on adapter gradient must be connected")
        if gradients["nr_gradient_target_adapter_only"] is not True:
            raise RunIntegrityError("A-on NR gradients must be asserted adapter-only")
    else:
        if gradients["adapter_gradient_connected"] is not None:
            raise RunIntegrityError("A-off adapter connectivity must be null")
        if gradients["nr_gradient_target_adapter_only"] is not None:
            raise RunIntegrityError("A-off NR gradient-target assertion must be null")
    gradients_finite = _require_bool(gradients["finite"], "gradients.finite")
    model_parameters_finite = _require_bool(
        gradients["model_parameters_finite"],
        "gradients.model_parameters_finite",
    )
    optimizer_master_parameters_finite = _require_bool(
        gradients["optimizer_master_parameters_finite"],
        "gradients.optimizer_master_parameters_finite",
    )
    optimizer_state_finite = _require_bool(
        gradients["optimizer_state_finite"],
        "gradients.optimizer_state_finite",
    )
    if gradients_finite:
        global_norm = _require_float(
            gradients["global_norm"], "gradients.global_norm", minimum=0.0
        )
        video_norm = _require_float(
            gradients["video_grad_norm"],
            "gradients.video_grad_norm",
            minimum=0.0,
        )
        if factors["A"]:
            adapter_norm = _require_float(
                gradients["adapter_grad_norm"],
                "gradients.adapter_grad_norm",
                minimum=0.0,
            )
        else:
            if gradients["adapter_grad_norm"] is not None:
                raise RunIntegrityError(
                    "A-off adapter gradient norm must be null, never numeric zero"
                )
            adapter_norm = 0.0
        expected_global_norm = math.sqrt(video_norm**2 + adapter_norm**2)
        if not _close(global_norm, expected_global_norm):
            raise RunIntegrityError(
                "global gradient norm differs from video/adapter group norms"
            )
    else:
        for key in ("global_norm", "video_grad_norm", "adapter_grad_norm"):
            if gradients[key] is not None:
                raise RunIntegrityError(
                    f"non-finite gradients.{key} must be represented as null"
                )

    memory = value["memory"]
    _require_exact_keys(memory, frozenset({"peak_reserved_bytes"}), "memory")
    _require_int(
        memory["peak_reserved_bytes"],
        "memory.peak_reserved_bytes",
        minimum=1,
    )

    expected_total = video + expansion_loss + nr_loss
    if not _close(total, expected_total):
        raise RunIntegrityError(
            "total_loss differs from video_on_path + expansion + action NR"
        )
    return (
        gradients_finite
        and model_parameters_finite
        and optimizer_master_parameters_finite
        and optimizer_state_finite
    )


def _build_step_record(
    *,
    binding: Phase6PlanBinding,
    arm: str,
    run_id: str,
    provenance: Mapping[str, str],
    global_step: int,
    observed_plan_row: Mapping[str, Any],
    measurements: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    if arm not in ARM_FACTORS:
        raise RunIntegrityError(f"unsupported Phase-6 arm: {arm!r}")
    factors = ARM_FACTORS[arm]
    step = _require_int(global_step, "global_step", minimum=1)
    if step > EXPECTED_STEPS:
        raise RunIntegrityError("global_step exceeds 504")
    expected_row = binding._rows[step - 1]
    try:
        observed_row_bytes = canonical_json_bytes(observed_plan_row)
        expected_runtime_row_bytes = canonical_json_bytes(
            binding.expected_runtime_row(step)
        )
    except RunIntegrityError as exc:
        raise RunIntegrityError("observed plan row is not canonicalizable") from exc
    if observed_row_bytes != expected_runtime_row_bytes:
        raise RunIntegrityError(f"observed plan row differs at global_step={step}")
    gradients_finite = _validate_measurements(measurements, factors)
    identity = expected_row["identity"]
    record = {
        "action": deepcopy(measurements["action"]),
        "action_sigma": expected_row["action_sigma"],
        "arm": arm,
        "common_input_trace_sha256": measurements["common_input_trace_sha256"],
        "cycle": expected_row["cycle"],
        "domain_seeds": deepcopy(expected_row["domain_seeds"]),
        "expansion": deepcopy(measurements["expansion"]),
        "factors": dict(factors),
        "global_step": step,
        "gradients": deepcopy(measurements["gradients"]),
        "identity_sha256": binding.identity_sha256,
        "learning_rates": deepcopy(measurements["learning_rates"]),
        "memory": deepcopy(measurements["memory"]),
        "plan_row_sha256": sha256(binding._row_bytes[step - 1]).hexdigest(),
        "plan_sha256": binding.plan_sha256,
        "position_in_cycle": expected_row["position_in_cycle"],
        "preflight_purpose": "training",
        "preflight_report_sha256": provenance["preflight_report_sha256"],
        "provenance_sha256": sha256(canonical_json_bytes(provenance)).hexdigest(),
        "prompt_utf8_sha256": sha256(identity["prompt"].encode("utf-8")).hexdigest(),
        "record_type": "step",
        "run_id": run_id,
        "schema_version": STEP_SCHEMA_VERSION,
        "task_name": identity["task_name"],
        "total_loss": measurements["total_loss"],
        "video_on_path_loss": measurements["video_on_path_loss"],
    }
    return record, gradients_finite


def _validate_step_record(
    value: Any,
    *,
    binding: Phase6PlanBinding,
    arm: str,
    run_id: str,
    provenance: Mapping[str, str],
    expected_step: int,
) -> bool:
    _require_exact_keys(value, _STEP_KEYS, f"metrics row {expected_step}")
    if value["schema_version"] != STEP_SCHEMA_VERSION or value["record_type"] != "step":
        raise RunIntegrityError(f"metrics row {expected_step} schema/type differs")
    if value["arm"] != arm or value["factors"] != ARM_FACTORS[arm]:
        raise RunIntegrityError(f"metrics row {expected_step} arm/factors differ")
    if value["run_id"] != run_id:
        raise RunIntegrityError(f"metrics row {expected_step} run_id differs")
    if value["preflight_purpose"] != "training":
        raise RunIntegrityError(
            f"metrics row {expected_step} preflight purpose differs"
        )
    if value["preflight_report_sha256"] != provenance["preflight_report_sha256"]:
        raise RunIntegrityError(f"metrics row {expected_step} preflight SHA differs")
    expected_provenance_sha = sha256(canonical_json_bytes(provenance)).hexdigest()
    if value["provenance_sha256"] != expected_provenance_sha:
        raise RunIntegrityError(f"metrics row {expected_step} provenance SHA differs")
    if value["plan_sha256"] != binding.plan_sha256:
        raise RunIntegrityError(f"metrics row {expected_step} plan SHA differs")
    if value["identity_sha256"] != binding.identity_sha256:
        raise RunIntegrityError(f"metrics row {expected_step} identity SHA differs")
    if value["global_step"] != expected_step:
        raise RunIntegrityError(f"metrics row {expected_step} global_step differs")
    plan_row = binding._rows[expected_step - 1]
    identity = plan_row["identity"]
    expected_projection = {
        "action_sigma": plan_row["action_sigma"],
        "cycle": plan_row["cycle"],
        "domain_seeds": plan_row["domain_seeds"],
        "plan_row_sha256": sha256(binding._row_bytes[expected_step - 1]).hexdigest(),
        "position_in_cycle": plan_row["position_in_cycle"],
        "prompt_utf8_sha256": sha256(identity["prompt"].encode("utf-8")).hexdigest(),
        "task_name": identity["task_name"],
    }
    for key, expected in expected_projection.items():
        if value[key] != expected:
            raise RunIntegrityError(
                f"metrics row {expected_step}.{key} differs from plan"
            )
    measurements = {
        "action": value["action"],
        "common_input_trace_sha256": value["common_input_trace_sha256"],
        "expansion": value["expansion"],
        "gradients": value["gradients"],
        "learning_rates": value["learning_rates"],
        "memory": value["memory"],
        "total_loss": value["total_loss"],
        "video_on_path_loss": value["video_on_path_loss"],
    }
    return _validate_measurements(measurements, ARM_FACTORS[arm])


def _student_error_distribution(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != EXPECTED_STEPS:
        raise RunIntegrityError("student-error distribution must contain 504 values")
    mean = math.fsum(values) / len(values)
    population_stddev = math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / len(values)
    )
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "maximum": max(values),
        "mean": mean,
        "minimum": min(values),
        "ordered_sequence_sha256": sha256(
            canonical_json_bytes(list(values))
        ).hexdigest(),
        "population_stddev": population_stddev,
        "sorted_multiset_sha256": sha256(
            canonical_json_bytes(sorted_values)
        ).hexdigest(),
    }


def validate_metrics_jsonl(
    path: str | os.PathLike[str],
    *,
    binding: Phase6PlanBinding,
    arm: str,
    run_id: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly reread all rows and return summary-ready structural facts."""

    if arm not in ARM_FACTORS:
        raise RunIntegrityError(f"unsupported Phase-6 arm: {arm!r}")
    run = _require_run_id(run_id)
    validated_provenance = _validate_provenance(provenance)
    observed, data = _read_stable_file(path, "metrics JSONL")
    lines = data.splitlines(keepends=True)
    if len(lines) != EXPECTED_STEPS:
        raise RunIntegrityError("metrics JSONL must contain exactly 504 rows")
    task_counts: Counter[str] = Counter()
    sigma_counts: Counter[str] = Counter()
    task_sigma_counts: dict[str, Counter[str]] = defaultdict(Counter)
    common_input_traces: list[str] = []
    student_errors: list[float] = []
    for expected_step, line in enumerate(lines, start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise RunIntegrityError(f"metrics row {expected_step} is partial/blank")
        value = _strict_json_loads(line, f"metrics row {expected_step}")
        if line != canonical_json_bytes(value):
            raise RunIntegrityError(
                f"metrics row {expected_step} is not canonical JSON"
            )
        gradients_finite = _validate_step_record(
            value,
            binding=binding,
            arm=arm,
            run_id=run,
            provenance=validated_provenance,
            expected_step=expected_step,
        )
        if not gradients_finite:
            if not value["gradients"]["finite"]:
                raise NonFiniteGradientError(
                    f"metrics row {expected_step} records non-finite gradients"
                )
            if not value["gradients"]["model_parameters_finite"]:
                raise NonFiniteModelParametersError(
                    f"metrics row {expected_step} records non-finite model parameters"
                )
            if not value["gradients"]["optimizer_master_parameters_finite"]:
                raise NonFiniteOptimizerMasterParametersError(
                    f"metrics row {expected_step} records non-finite optimizer masters"
                )
            raise NonFiniteOptimizerStateError(
                f"metrics row {expected_step} records non-finite optimizer state"
            )
        task = value["task_name"]
        sigma_key = _SIGMA_KEYS[value["action_sigma"]]
        task_counts[task] += 1
        sigma_counts[sigma_key] += 1
        task_sigma_counts[task][sigma_key] += 1
        common_input_traces.append(value["common_input_trace_sha256"])
        student_errors.append(value["action"]["student_error"])
    if set(task_counts) != set(binding.train_tasks) or any(
        task_counts[task] != EXPECTED_CYCLES for task in binding.train_tasks
    ):
        raise RunIntegrityError("metrics do not contain 12 rows for each of 42 tasks")
    expected_task_sigma = {
        _SIGMA_KEYS[sigma]: EXPECTED_SIGMA_EXPOSURES for sigma in _ACTION_SIGMAS
    }
    if any(
        dict(task_sigma_counts[task]) != expected_task_sigma
        for task in binding.train_tasks
    ):
        raise RunIntegrityError("metrics do not contain four of each sigma per task")
    return {
        "common_input_trace_sequence_sha256": sha256(
            canonical_json_bytes(common_input_traces)
        ).hexdigest(),
        "student_error_distribution": _student_error_distribution(student_errors),
        "metrics": {**observed, "row_count": len(lines)},
        "steps_by_sigma": dict(sorted(sigma_counts.items())),
        "steps_by_task": {
            task: task_counts[task] for task in sorted(binding.train_tasks)
        },
        "task_sigma_counts": {
            task: dict(sorted(task_sigma_counts[task].items()))
            for task in sorted(binding.train_tasks)
        },
    }


def validate_cross_arm_common_input_traces(
    summary_paths: Mapping[str, str | os.PathLike[str]],
) -> dict[str, Any]:
    """Require the common prepared/noise/mask trace to match across five arms."""

    if not isinstance(summary_paths, Mapping) or set(summary_paths) != set(ARM_FACTORS):
        raise RunIntegrityError("cross-arm trace validation requires all five arms")
    observed_summaries: dict[str, str] = {}
    sequence_sha: str | None = None
    plan_sha: str | None = None
    identity_sha: str | None = None
    common_provenance: dict[str, str] | None = None
    common_pin_keys = (
        "action_reference_sha256",
        "action_stats_sha256",
        "dataset_contract_sha256",
        "initial_checkpoint_sha256",
        "phase1_checkpoint_sha256",
        "source_manifest_sha256",
        "spot_verification_sha256",
    )
    for arm in ARM_FACTORS:
        observed, data = _read_stable_file(summary_paths[arm], f"{arm} metrics summary")
        value = _strict_json_loads(data, f"{arm} metrics summary")
        if not isinstance(value, Mapping) or data != canonical_json_bytes(value):
            raise RunIntegrityError(f"{arm} metrics summary is not canonical JSON")
        _require_exact_keys(value, _SUMMARY_KEYS, f"{arm} metrics summary")
        if (
            value["schema_version"] != SUMMARY_SCHEMA_VERSION
            or value["record_type"] != "summary"
            or value["arm"] != arm
            or value["factors"] != ARM_FACTORS[arm]
            or value["preflight_purpose"] != "training"
        ):
            raise RunIntegrityError(f"{arm} metrics summary identity differs")
        trace = value["common_input_trace"]
        _require_exact_keys(
            trace,
            frozenset({"contract", "sequence_sha256"}),
            f"{arm}.common_input_trace",
        )
        if trace["contract"] != COMMON_INPUT_TRACE_CONTRACT:
            raise RunIntegrityError(f"{arm} common-input trace contract differs")
        trace_sha = _require_sha256(
            trace["sequence_sha256"], f"{arm} trace sequence SHA"
        )
        if sequence_sha is None:
            sequence_sha = trace_sha
        elif trace_sha != sequence_sha:
            raise RunIntegrityError("common input trace differs across Phase-6 arms")
        current_plan_sha = _require_sha256(value["plan_sha256"], f"{arm} plan SHA")
        current_identity_sha = _require_sha256(
            value["identity_sha256"], f"{arm} identity SHA"
        )
        if plan_sha is None:
            plan_sha = current_plan_sha
            identity_sha = current_identity_sha
        elif current_plan_sha != plan_sha or current_identity_sha != identity_sha:
            raise RunIntegrityError("plan/identity differs across Phase-6 arms")
        provenance = _validate_provenance(value["provenance"])
        selected = {key: provenance[key] for key in common_pin_keys}
        if common_provenance is None:
            common_provenance = selected
        elif selected != common_provenance:
            raise RunIntegrityError("common provenance pins differ across Phase-6 arms")
        metrics = value["metrics"]
        if (
            not isinstance(metrics, Mapping)
            or metrics.get("row_count") != EXPECTED_STEPS
        ):
            raise RunIntegrityError(f"{arm} metrics summary row count differs")
        observed_summaries[arm] = observed["sha256"]
    if sequence_sha is None or plan_sha is None or identity_sha is None:
        raise AssertionError("five-arm summary loop was empty")
    return {
        "arms": list(ARM_FACTORS),
        "common_input_trace_contract": COMMON_INPUT_TRACE_CONTRACT,
        "common_input_trace_sequence_sha256": sequence_sha,
        "identity_sha256": identity_sha,
        "plan_sha256": plan_sha,
        "schema_version": CROSS_ARM_TRACE_SCHEMA_VERSION,
        "summary_sha256_by_arm": observed_summaries,
    }


def _safetensors_state_projection(
    path: str,
    *,
    selected_keys: Sequence[str],
    name: str,
) -> dict[str, Any]:
    """Hash selected tensor bytes without materializing unrelated 2B state."""

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RunIntegrityError(
            "building frozen-tensor evidence requires torch and safetensors"
        ) from exc
    descriptors = []
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            available = set(checkpoint.keys())
            missing = sorted(set(selected_keys) - available)
            if missing:
                raise RunIntegrityError(f"{name} lacks selected tensors: {missing}")
            for key in selected_keys:
                tensor = checkpoint.get_tensor(key).detach().cpu().contiguous()
                flat_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                descriptors.append(
                    {
                        "dtype": str(tensor.dtype),
                        "key": key,
                        "raw_bytes_sha256": sha256(flat_bytes).hexdigest(),
                        "shape": list(tensor.shape),
                    }
                )
    except RunIntegrityError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise RunIntegrityError(f"cannot read {name} safetensors state: {exc}") from exc
    return {
        "descriptors": descriptors,
        "state_sha256": sha256(canonical_json_bytes(descriptors)).hexdigest(),
    }


def _checkpoint_key_set(path: str, name: str) -> set[str]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RunIntegrityError(
            "safetensors is required for checkpoint evidence"
        ) from exc
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            return set(checkpoint.keys())
    except (OSError, RuntimeError, ValueError) as exc:
        raise RunIntegrityError(f"cannot enumerate {name}: {exc}") from exc


def build_frozen_tensor_verification_manifest(
    *,
    initial_checkpoint_path: str | os.PathLike[str],
    step504_checkpoint_path: str | os.PathLike[str],
    initial_checkpoint_sha256: str,
    step504_checkpoint_sha256: str,
    arm: str,
) -> dict[str, Any]:
    """Read both checkpoints and prove frozen action/proprio state bitwise."""

    if arm not in ARM_FACTORS:
        raise RunIntegrityError(f"unsupported Phase-6 arm: {arm!r}")
    initial_expected = _require_sha256(
        initial_checkpoint_sha256, "initial_checkpoint_sha256"
    )
    final_expected = _require_sha256(
        step504_checkpoint_sha256, "step504_checkpoint_sha256"
    )
    initial_observed, _ = _observe_stable_file(
        initial_checkpoint_path, "initial checkpoint", capture=False
    )
    final_observed, _ = _observe_stable_file(
        step504_checkpoint_path, "step504 checkpoint", capture=False
    )
    if initial_observed["sha256"] != initial_expected:
        raise RunIntegrityError(
            "initial checkpoint SHA changed before frozen verification"
        )
    if final_observed["sha256"] != final_expected:
        raise RunIntegrityError(
            "step504 checkpoint SHA changed before frozen verification"
        )
    initial_path = initial_observed["path"]
    final_path = final_observed["path"]
    initial_keys = _checkpoint_key_set(initial_path, "initial checkpoint")
    final_keys = _checkpoint_key_set(final_path, "step504 checkpoint")
    action_initial_keys = sorted(
        key for key in initial_keys if key.startswith("action_backbone.")
    )
    action_final_keys = sorted(
        key for key in final_keys if key.startswith("action_backbone.")
    )
    if action_initial_keys != action_final_keys or len(action_initial_keys) != 556:
        raise RunIntegrityError(
            "initial/final action_backbone key sets must match at exactly 556 tensors"
        )
    proprio_initial_keys = sorted(
        key for key in initial_keys if key.startswith(_PROPRIO_STATE_PREFIXES)
    )
    proprio_final_keys = sorted(
        key for key in final_keys if key.startswith(_PROPRIO_STATE_PREFIXES)
    )
    if proprio_initial_keys != _PROPRIO_STATE_KEYS:
        raise RunIntegrityError(
            "initial checkpoint proprio key set must contain exactly six tensors"
        )
    if proprio_final_keys != _PROPRIO_STATE_KEYS:
        raise RunIntegrityError(
            "step504 checkpoint proprio key set must contain exactly six tensors"
        )
    unexpected_initial_adapter = sorted(
        key
        for key in initial_keys
        if key.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
    )
    if unexpected_initial_adapter:
        raise RunIntegrityError(
            "student initialization must contain no adapter tensors"
        )
    final_adapter_keys = sorted(
        key for key in final_keys if key.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
    )
    expected_adapter_keys = list(_ADAPTER_STATE_KEYS) if ARM_FACTORS[arm]["A"] else []
    if final_adapter_keys != expected_adapter_keys:
        raise RunIntegrityError("step504 adapter key set differs from the A factor")

    action_initial = _safetensors_state_projection(
        initial_path,
        selected_keys=action_initial_keys,
        name="initial action_backbone",
    )
    action_final = _safetensors_state_projection(
        final_path,
        selected_keys=action_final_keys,
        name="final action_backbone",
    )
    proprio_initial = _safetensors_state_projection(
        initial_path,
        selected_keys=proprio_initial_keys,
        name="initial proprio",
    )
    proprio_final = _safetensors_state_projection(
        final_path,
        selected_keys=proprio_final_keys,
        name="final proprio",
    )

    def component(
        keys: Sequence[str], initial: Mapping[str, Any], final: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "bitwise_equal": initial["descriptors"] == final["descriptors"],
            "final_sha256": final["state_sha256"],
            "initial_sha256": initial["state_sha256"],
            "key_names_sha256": sha256(canonical_json_bytes(list(keys))).hexdigest(),
            "selected_keys": list(keys),
            "tensor_count": len(keys),
        }

    adapter_projection = (
        _safetensors_state_projection(
            final_path,
            selected_keys=expected_adapter_keys,
            name="final action video-memory adapter",
        )
        if expected_adapter_keys
        else None
    )
    manifest = {
        "action_backbone": component(action_initial_keys, action_initial, action_final),
        "action_video_memory_adapter": {
            "expected": bool(expected_adapter_keys),
            "key_count": len(expected_adapter_keys),
            "key_names_sha256": (
                sha256(canonical_json_bytes(expected_adapter_keys)).hexdigest()
                if expected_adapter_keys
                else None
            ),
            "present": bool(final_adapter_keys),
            "selected_keys": expected_adapter_keys,
            "state_sha256": (
                adapter_projection["state_sha256"]
                if adapter_projection is not None
                else None
            ),
        },
        "initial_checkpoint_sha256": initial_expected,
        "proprio": component(_PROPRIO_STATE_KEYS, proprio_initial, proprio_final),
        "schema_version": FROZEN_TENSOR_VERIFICATION_SCHEMA_VERSION,
        "step504_checkpoint_sha256": final_expected,
    }
    _validate_frozen_tensor_manifest(
        canonical_json_bytes(manifest),
        initial_checkpoint_sha256=initial_expected,
        step504_checkpoint_sha256=final_expected,
        arm=arm,
    )
    return manifest


def build_resolved_config_verification_manifest(
    *,
    arm: str,
    input_arm_config_path: str | os.PathLike[str],
    input_arm_config_sha256: str,
    resolved_config_path: str | os.PathLike[str],
    deploy_config_path: str | os.PathLike[str],
    expected_projection_sha256: str,
) -> dict[str, Any]:
    """Verify saved/deploy configs semantically and bind the frozen projection."""

    from sana_wam.train.phase6_arm_config import (
        build_phase6_arm_projection,
        canonical_json_bytes as arm_canonical_json_bytes,
        load_arm_config_bytes,
        validate_arm_config,
    )

    if arm not in ARM_FACTORS:
        raise RunIntegrityError(f"unsupported Phase-6 arm: {arm!r}")
    input_expected = _require_sha256(input_arm_config_sha256, "input_arm_config_sha256")
    projection_expected = _require_sha256(
        expected_projection_sha256, "expected_projection_sha256"
    )
    input_observed, input_bytes = _read_stable_file(
        input_arm_config_path, "input arm config"
    )
    resolved_observed, resolved_bytes = _read_stable_file(
        resolved_config_path, "resolved saved config"
    )
    deploy_observed, deploy_bytes = _read_stable_file(
        deploy_config_path, "deploy config"
    )
    if input_observed["sha256"] != input_expected:
        raise RunIntegrityError(
            "input arm config changed before saved-config verification"
        )
    if deploy_bytes != resolved_bytes:
        raise RunIntegrityError("deploy config.yaml differs from resolved_config.yaml")
    input_config = load_arm_config_bytes(input_bytes, source=input_observed["path"])
    resolved_config = load_arm_config_bytes(
        resolved_bytes, source=resolved_observed["path"]
    )
    if validate_arm_config(input_config, verify_files=False) != arm:
        raise RunIntegrityError("input arm config resolves to a different arm")
    if validate_arm_config(resolved_config, verify_files=False) != arm:
        raise RunIntegrityError("resolved saved config resolves to a different arm")
    if input_config != resolved_config:
        raise RunIntegrityError(
            "resolved saved config differs semantically from input YAML"
        )
    projection = build_phase6_arm_projection(arm)
    projection_sha = sha256(arm_canonical_json_bytes(projection)).hexdigest()
    if projection_sha != projection_expected:
        raise RunIntegrityError(
            "actual saved config projection differs from launch pin"
        )
    manifest = {
        "arm": arm,
        "deploy_config_equivalent": True,
        "deploy_config_sha256": deploy_observed["sha256"],
        "factors": dict(ARM_FACTORS[arm]),
        "frozen_projection": projection,
        "frozen_projection_sha256": projection_sha,
        "input_arm_config_sha256": input_expected,
        "resolved_saved_config_sha256": resolved_observed["sha256"],
        "schema_version": RESOLVED_CONFIG_VERIFICATION_SCHEMA_VERSION,
    }
    _validate_resolved_config_manifest(
        canonical_json_bytes(manifest),
        arm=arm,
        input_arm_config_sha256=input_expected,
        expected_projection_sha256=projection_expected,
        resolved_saved_config_sha256=resolved_observed["sha256"],
        deploy_config_sha256=deploy_observed["sha256"],
    )
    return manifest


def write_integrity_manifest(
    path: str | os.PathLike[str], value: Mapping[str, Any]
) -> str:
    """Exclusively publish one already-built canonical integrity manifest."""

    destination = _require_absolute_path(path, "integrity manifest path")
    payload = canonical_json_bytes(dict(value))
    _exclusive_atomic_write(destination, payload)
    observed, observed_bytes = _read_stable_file(destination, "integrity manifest")
    if observed_bytes != payload:
        raise RunIntegrityError("integrity manifest changed after publication")
    return observed["sha256"]


def _validate_frozen_tensor_manifest(
    data: bytes,
    *,
    initial_checkpoint_sha256: str,
    step504_checkpoint_sha256: str,
    arm: str,
) -> None:
    value = _strict_json_loads(data, "frozen-tensor verification manifest")
    if not isinstance(value, Mapping) or data != canonical_json_bytes(value):
        raise RunIntegrityError(
            "frozen-tensor verification manifest must be canonical JSON"
        )
    _require_exact_keys(
        value,
        frozenset(
            {
                "action_backbone",
                "action_video_memory_adapter",
                "initial_checkpoint_sha256",
                "proprio",
                "schema_version",
                "step504_checkpoint_sha256",
            }
        ),
        "frozen-tensor verification manifest",
    )
    if value["schema_version"] != FROZEN_TENSOR_VERIFICATION_SCHEMA_VERSION:
        raise RunIntegrityError("frozen-tensor verification schema differs")
    if value["initial_checkpoint_sha256"] != initial_checkpoint_sha256:
        raise RunIntegrityError("frozen-tensor manifest initial checkpoint differs")
    if value["step504_checkpoint_sha256"] != step504_checkpoint_sha256:
        raise RunIntegrityError("frozen-tensor manifest step504 checkpoint differs")
    component_keys = frozenset(
        {
            "bitwise_equal",
            "final_sha256",
            "initial_sha256",
            "key_names_sha256",
            "selected_keys",
            "tensor_count",
        }
    )
    for component in ("action_backbone", "proprio"):
        entry = value[component]
        _require_exact_keys(entry, component_keys, f"frozen tensors.{component}")
        initial = _require_sha256(
            entry["initial_sha256"], f"frozen tensors.{component}.initial_sha256"
        )
        final = _require_sha256(
            entry["final_sha256"], f"frozen tensors.{component}.final_sha256"
        )
        tensor_count = _require_int(
            entry["tensor_count"],
            f"frozen tensors.{component}.tensor_count",
            minimum=1,
        )
        if entry["bitwise_equal"] is not True or initial != final:
            raise RunIntegrityError(
                f"frozen {component} tensors are not bitwise identical"
            )
        selected_keys = entry["selected_keys"]
        if (
            not isinstance(selected_keys, list)
            or len(selected_keys) != tensor_count
            or selected_keys != sorted(selected_keys)
            or len(set(selected_keys)) != tensor_count
            or any(not isinstance(key, str) or not key for key in selected_keys)
        ):
            raise RunIntegrityError(f"frozen {component} selected key list is invalid")
        key_names_sha = _require_sha256(
            entry["key_names_sha256"],
            f"frozen tensors.{component}.key_names_sha256",
        )
        if key_names_sha != sha256(canonical_json_bytes(selected_keys)).hexdigest():
            raise RunIntegrityError(f"frozen {component} key-name SHA differs")
        if component == "action_backbone":
            if tensor_count != 556 or any(
                not key.startswith("action_backbone.") for key in selected_keys
            ):
                raise RunIntegrityError(
                    "frozen action_backbone selection must contain its exact 556 tensors"
                )
        elif selected_keys != _PROPRIO_STATE_KEYS or tensor_count != 6:
            raise RunIntegrityError(
                "frozen proprio selection must contain the exact six registered tensors"
            )
    adapter = value["action_video_memory_adapter"]
    _require_exact_keys(
        adapter,
        frozenset(
            {
                "expected",
                "key_count",
                "key_names_sha256",
                "present",
                "selected_keys",
                "state_sha256",
            }
        ),
        "action_video_memory_adapter verification",
    )
    adapter_expected = ARM_FACTORS[arm]["A"]
    if adapter["expected"] is not adapter_expected:
        raise RunIntegrityError("adapter expected flag differs from A factor")
    if adapter_expected:
        if adapter["present"] is not True or adapter["key_count"] != 5:
            raise RunIntegrityError(
                "A-on checkpoint must contain all five adapter keys"
            )
        expected_key_names_sha = sha256(
            canonical_json_bytes(_ADAPTER_STATE_KEYS)
        ).hexdigest()
        if adapter["key_names_sha256"] != expected_key_names_sha:
            raise RunIntegrityError("A-on adapter checkpoint key set differs")
        if adapter["selected_keys"] != _ADAPTER_STATE_KEYS:
            raise RunIntegrityError("A-on adapter selected key list differs")
        _require_sha256(adapter["state_sha256"], "adapter state_sha256")
    else:
        if (
            adapter["present"] is not False
            or type(adapter["key_count"]) is not int
            or adapter["key_count"] != 0
            or adapter["key_names_sha256"] is not None
            or adapter["selected_keys"] != []
            or adapter["state_sha256"] is not None
        ):
            raise RunIntegrityError("A-off checkpoint must contain no adapter state")


def _validate_resolved_config_manifest(
    data: bytes,
    *,
    arm: str,
    input_arm_config_sha256: str,
    expected_projection_sha256: str,
    resolved_saved_config_sha256: str,
    deploy_config_sha256: str,
) -> None:
    value = _strict_json_loads(data, "resolved-config verification manifest")
    if not isinstance(value, Mapping) or data != canonical_json_bytes(value):
        raise RunIntegrityError(
            "resolved-config verification manifest must be canonical JSON"
        )
    _require_exact_keys(
        value,
        frozenset(
            {
                "arm",
                "deploy_config_equivalent",
                "deploy_config_sha256",
                "factors",
                "frozen_projection",
                "frozen_projection_sha256",
                "input_arm_config_sha256",
                "resolved_saved_config_sha256",
                "schema_version",
            }
        ),
        "resolved-config verification manifest",
    )
    if value["schema_version"] != RESOLVED_CONFIG_VERIFICATION_SCHEMA_VERSION:
        raise RunIntegrityError("resolved-config verification schema differs")
    if value["arm"] != arm or value["factors"] != ARM_FACTORS[arm]:
        raise RunIntegrityError("resolved-config arm/factors differ")
    if value["input_arm_config_sha256"] != input_arm_config_sha256:
        raise RunIntegrityError("resolved-config input YAML SHA differs")
    if value["resolved_saved_config_sha256"] != resolved_saved_config_sha256:
        raise RunIntegrityError("resolved saved-config SHA differs")
    if (
        value["deploy_config_equivalent"] is not True
        or value["deploy_config_sha256"] != deploy_config_sha256
        or value["deploy_config_sha256"] != resolved_saved_config_sha256
    ):
        raise RunIntegrityError("deploy config is not byte-equivalent to saved config")
    projection = value["frozen_projection"]
    if not isinstance(projection, Mapping) or not projection:
        raise RunIntegrityError(
            "resolved-config frozen projection must be a non-empty object"
        )
    expected_projection_sha = sha256(canonical_json_bytes(projection)).hexdigest()
    projection_sha = _require_sha256(
        value["frozen_projection_sha256"],
        "resolved-config frozen_projection_sha256",
    )
    if (
        projection_sha != expected_projection_sha
        or projection_sha != expected_projection_sha256
    ):
        raise RunIntegrityError("resolved-config frozen projection SHA differs")


def _validate_primary_outputs(
    value: Any,
    *,
    run_directory: str,
    provenance: Mapping[str, str],
    arm: str,
) -> dict[str, dict[str, Any]]:
    _require_exact_keys(value, frozenset(_PRIMARY_OUTPUT_ROLES), "primary_outputs")
    run_realpath = os.path.realpath(run_directory)
    observed_outputs: dict[str, dict[str, Any]] = {}
    captured_manifests: dict[str, bytes] = {}
    real_targets: set[str] = set()
    for role in _PRIMARY_OUTPUT_ROLES:
        pin = value[role]
        _require_exact_keys(
            pin, frozenset({"path", "sha256"}), f"primary_outputs.{role}"
        )
        path = _require_absolute_path(pin["path"], f"primary_outputs.{role}.path")
        if os.path.basename(path) != _PRIMARY_OUTPUT_BASENAMES[role]:
            raise RunIntegrityError(
                f"primary output {role} must be named {_PRIMARY_OUTPUT_BASENAMES[role]}"
            )
        expected_sha = _require_sha256(pin["sha256"], f"primary_outputs.{role}.sha256")
        observed, captured = _observe_stable_file(
            path,
            f"primary output {role}",
            capture=role.endswith("_verification_manifest"),
        )
        if not hmac.compare_digest(observed["sha256"], expected_sha):
            raise RunIntegrityError(f"primary output {role} SHA mismatch")
        try:
            within_run = (
                os.path.commonpath([observed["realpath"], run_realpath]) == run_realpath
            )
        except ValueError:
            within_run = False
        if not within_run:
            raise RunIntegrityError(
                f"primary output {role} resolves outside exclusive run directory"
            )
        if os.path.normcase(os.path.dirname(observed["realpath"])) != os.path.normcase(
            run_realpath
        ):
            raise RunIntegrityError(
                f"primary output {role} must be directly inside the run directory"
            )
        target_key = os.path.normcase(observed["realpath"])
        if target_key in real_targets:
            raise RunIntegrityError("primary outputs reuse one real file target")
        real_targets.add(target_key)
        if observed["size_bytes"] <= 0:
            raise RunIntegrityError(f"primary output {role} is empty")
        observed_outputs[role] = observed
        if captured is not None:
            captured_manifests[role] = captured
    checkpoint_candidates = []
    for candidate in Path(run_directory).iterdir():
        if (
            candidate.name.startswith("checkpoint")
            or candidate.suffix == ".safetensors"
        ):
            checkpoint_candidates.append(candidate.name)
    if sorted(checkpoint_candidates) != ["checkpoint_step_504.safetensors"]:
        raise RunIntegrityError(
            "exclusive run directory must contain exactly one primary step504 checkpoint"
        )
    if observed_outputs["action_stats"]["sha256"] != provenance["action_stats_sha256"]:
        raise RunIntegrityError("saved action stats differ from provenance pin")
    frozen_manifest = captured_manifests.get("frozen_tensor_verification_manifest")
    config_manifest = captured_manifests.get("resolved_config_verification_manifest")
    if frozen_manifest is None or config_manifest is None:
        raise AssertionError("primary verification manifest bytes were not captured")
    deploy_config, _ = _observe_stable_file(
        os.path.join(run_directory, "config.yaml"),
        "deploy config",
        capture=False,
    )
    if os.path.normcase(os.path.dirname(deploy_config["realpath"])) != os.path.normcase(
        run_realpath
    ):
        raise RunIntegrityError(
            "deploy config must resolve directly inside run directory"
        )
    _validate_frozen_tensor_manifest(
        frozen_manifest,
        initial_checkpoint_sha256=provenance["initial_checkpoint_sha256"],
        step504_checkpoint_sha256=observed_outputs["step504_checkpoint"]["sha256"],
        arm=arm,
    )
    _validate_resolved_config_manifest(
        config_manifest,
        arm=arm,
        input_arm_config_sha256=provenance["arm_config_sha256"],
        expected_projection_sha256=provenance["arm_config_projection_sha256"],
        resolved_saved_config_sha256=observed_outputs["saved_arm_config"]["sha256"],
        deploy_config_sha256=deploy_config["sha256"],
    )
    return observed_outputs


def _fsync_parent(path: str) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    parent = os.path.dirname(path)
    try:
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RunIntegrityError(
            f"cannot fsync output directory {parent}: {exc}"
        ) from exc


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise RunIntegrityError("short write while emitting integrity artifact")
        offset += written


def _exclusive_atomic_write(path: str, payload: bytes) -> None:
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        raise RunIntegrityError(f"summary parent does not exist: {parent}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent
    )
    linked = False
    try:
        try:
            os.fchmod(descriptor, 0o640)
        except (AttributeError, OSError):
            pass
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path)
        linked = True
        _fsync_parent(path)
    except FileExistsError as exc:
        raise RunIntegrityError(f"summary path already exists: {path}") from exc
    except OSError as exc:
        raise RunIntegrityError(f"cannot publish summary {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    if not linked:
        raise RunIntegrityError("summary was not atomically linked")


def _create_exclusive_run_directory(path: str | os.PathLike[str]) -> str:
    absolute = _require_absolute_path(path, "run_directory")
    parent = os.path.dirname(absolute)
    if not os.path.isdir(parent):
        raise RunIntegrityError(f"run-directory parent does not exist: {parent}")
    try:
        os.mkdir(absolute, 0o750)
        _fsync_parent(absolute)
    except FileExistsError as exc:
        raise RunIntegrityError(
            f"Phase-6 run directory already exists; resume/append is forbidden: {absolute}"
        ) from exc
    except OSError as exc:
        raise RunIntegrityError(
            f"cannot exclusively create Phase-6 run directory: {exc}"
        ) from exc
    return absolute


@dataclass(frozen=True)
class FinalizedMetrics:
    summary: dict[str, Any]
    summary_sha256: str
    completion: dict[str, Any]
    completion_sha256: str


class Phase6MetricsWriter:
    """Exclusive O_APPEND writer that fsyncs every canonical step record."""

    def __init__(
        self,
        *,
        run_directory: str | os.PathLike[str],
        preflight_report_path: str | os.PathLike[str],
        binding: Phase6PlanBinding,
        arm: str,
        run_id: str,
        provenance: Mapping[str, Any],
    ) -> None:
        if arm not in ARM_FACTORS:
            raise RunIntegrityError(f"unsupported Phase-6 arm: {arm!r}")
        self._arm = arm
        self._run_id = _require_run_id(run_id)
        requested_run_directory = _require_absolute_path(run_directory, "run_directory")
        if os.path.basename(requested_run_directory) != self._run_id:
            raise RunIntegrityError(
                "exclusive run-directory basename must equal the launch-pinned run_id"
            )
        self._provenance = _validate_provenance(provenance)
        _source_report_observed, source_report_bytes = _read_stable_file(
            preflight_report_path, "preflight success report"
        )
        _validate_training_preflight_report(source_report_bytes, self._provenance)
        self._binding = binding
        self._run_directory = _create_exclusive_run_directory(requested_run_directory)
        self._metrics_path = os.path.join(
            self._run_directory, "phase6_step_metrics.jsonl"
        )
        self._summary_path = os.path.join(
            self._run_directory, "phase6_metrics_summary.json"
        )
        self._completion_path = os.path.join(
            self._run_directory, "phase6_run_completion_manifest.json"
        )
        self._preflight_snapshot_path = os.path.join(
            self._run_directory, "phase6_preflight_report_snapshot.json"
        )
        _exclusive_atomic_write(self._preflight_snapshot_path, source_report_bytes)
        self._preflight_snapshot, _snapshot_bytes = _read_stable_file(
            self._preflight_snapshot_path, "preflight report snapshot"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
        flags |= getattr(os, "O_BINARY", 0)
        try:
            self._descriptor = os.open(self._metrics_path, flags, 0o640)
        except OSError as exc:
            raise RunIntegrityError(
                f"cannot exclusively create metrics JSONL: {exc}"
            ) from exc
        file_stat = os.fstat(self._descriptor)
        self._device_inode = (file_stat.st_dev, file_stat.st_ino)
        self._bytes_written = 0
        self._rows_written = 0
        self._closed = False
        self._failed = False
        self._finalized = False
        try:
            _fsync_parent(self._metrics_path)
        except BaseException:
            os.close(self._descriptor)
            self._closed = True
            raise

    @property
    def rows_written(self) -> int:
        return self._rows_written

    @property
    def run_directory(self) -> str:
        return self._run_directory

    @property
    def metrics_path(self) -> str:
        return self._metrics_path

    @property
    def summary_path(self) -> str:
        return self._summary_path

    @property
    def completion_path(self) -> str:
        return self._completion_path

    @property
    def preflight_report_snapshot_path(self) -> str:
        return self._preflight_snapshot_path

    def _assert_owned_path(self) -> None:
        if self._closed:
            raise RunIntegrityError("metrics writer is closed")
        try:
            path_stat = os.stat(self._metrics_path)
            descriptor_stat = os.fstat(self._descriptor)
        except OSError as exc:
            raise RunIntegrityError(
                f"metrics path/descriptor unavailable: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or (path_stat.st_dev, path_stat.st_ino) != self._device_inode
            or (descriptor_stat.st_dev, descriptor_stat.st_ino) != self._device_inode
        ):
            raise RunIntegrityError("metrics path was replaced or retargeted")
        if (
            path_stat.st_size != self._bytes_written
            or descriptor_stat.st_size != self._bytes_written
        ):
            raise RunIntegrityError("metrics JSONL was externally appended/truncated")

    def append_step(
        self,
        *,
        global_step: int,
        observed_plan_row: Mapping[str, Any],
        measurements: Mapping[str, Any],
    ) -> None:
        """Validate one completed optimizer step, append it, and fsync it."""

        try:
            self._append_step(
                global_step=global_step,
                observed_plan_row=observed_plan_row,
                measurements=measurements,
            )
        except NonFiniteGradientError:
            raise
        except RunIntegrityError:
            self._failed = True
            self.close()
            raise

    def _append_step(
        self,
        *,
        global_step: int,
        observed_plan_row: Mapping[str, Any],
        measurements: Mapping[str, Any],
    ) -> None:
        if self._failed:
            raise RunIntegrityError("metrics writer is in a failed state")
        if global_step != self._rows_written + 1:
            raise RunIntegrityError(
                f"expected global_step={self._rows_written + 1}, got {global_step}"
            )
        self._assert_owned_path()
        record, gradients_finite = _build_step_record(
            binding=self._binding,
            arm=self._arm,
            run_id=self._run_id,
            provenance=self._provenance,
            global_step=global_step,
            observed_plan_row=observed_plan_row,
            measurements=measurements,
        )
        payload = canonical_json_bytes(record)
        try:
            _write_all(self._descriptor, payload)
            os.fsync(self._descriptor)
        except OSError as exc:
            self._failed = True
            raise RunIntegrityError(
                f"cannot durably append metrics row: {exc}"
            ) from exc
        self._bytes_written += len(payload)
        self._rows_written += 1
        self._assert_owned_path()
        if not gradients_finite:
            self._failed = True
            self.close()
            if not measurements["gradients"]["finite"]:
                raise NonFiniteGradientError(
                    f"non-finite gradient recorded at global_step={global_step}"
                )
            if not measurements["gradients"]["model_parameters_finite"]:
                raise NonFiniteModelParametersError(
                    f"non-finite model parameters at global_step={global_step}"
                )
            if not measurements["gradients"]["optimizer_master_parameters_finite"]:
                raise NonFiniteOptimizerMasterParametersError(
                    f"non-finite optimizer masters at global_step={global_step}"
                )
            raise NonFiniteOptimizerStateError(
                f"non-finite optimizer state at global_step={global_step}"
            )

    def close(self) -> None:
        if self._closed:
            return
        try:
            os.fsync(self._descriptor)
        finally:
            os.close(self._descriptor)
            self._closed = True

    def finalize(
        self,
        *,
        primary_outputs: Mapping[str, Any] | None = None,
    ) -> FinalizedMetrics:
        """Reread 504 rows, verify coverage, and exclusively publish a summary."""

        if self._finalized:
            raise RunIntegrityError("metrics writer is already finalized")
        if self._failed:
            raise RunIntegrityError("failed metrics writer cannot be finalized")
        if self._rows_written != EXPECTED_STEPS:
            raise RunIntegrityError(
                f"cannot finalize {self._rows_written} rows; expected 504"
            )
        self._assert_owned_path()
        self.close()
        validation = validate_metrics_jsonl(
            self._metrics_path,
            binding=self._binding,
            arm=self._arm,
            run_id=self._run_id,
            provenance=self._provenance,
        )
        if primary_outputs is None:
            raise RunIntegrityError(
                "primary output closure is required before run finalization"
            )
        observed_primary_outputs = _validate_primary_outputs(
            primary_outputs,
            run_directory=self._run_directory,
            provenance=self._provenance,
            arm=self._arm,
        )
        summary = {
            "action_student_error_distribution": validation[
                "student_error_distribution"
            ],
            "arm": self._arm,
            "common_input_trace": {
                "contract": COMMON_INPUT_TRACE_CONTRACT,
                "sequence_sha256": validation["common_input_trace_sequence_sha256"],
            },
            "coverage": {
                "cycles_per_task": EXPECTED_CYCLES,
                "sigma_exposures_per_task": validation["task_sigma_counts"],
                "steps_by_sigma": validation["steps_by_sigma"],
                "steps_by_task": validation["steps_by_task"],
                "task_count": EXPECTED_TASKS,
            },
            "factors": dict(ARM_FACTORS[self._arm]),
            "identity_sha256": self._binding.identity_sha256,
            "integrity": {
                "all_gradients_finite": True,
                "all_model_parameters_finite": True,
                "all_optimizer_master_parameters_finite": True,
                "all_optimizer_state_finite": True,
                "all_rows_canonical": True,
                "all_rows_plan_bound": True,
                "append_fsync_per_step": True,
                "global_steps_contiguous": True,
            },
            "metrics": validation["metrics"],
            "plan_artifact_sha256": self._binding.artifact_sha256,
            "plan_sha256": self._binding.plan_sha256,
            "preflight_purpose": "training",
            "preflight_report_snapshot": self._preflight_snapshot,
            "primary_outputs": observed_primary_outputs,
            "provenance": dict(self._provenance),
            "record_type": "summary",
            "run_id": self._run_id,
            "schema_version": SUMMARY_SCHEMA_VERSION,
        }
        payload = canonical_json_bytes(summary)
        _exclusive_atomic_write(self._summary_path, payload)
        summary_observed, summary_bytes = _read_stable_file(
            self._summary_path, "metrics summary"
        )
        if summary_bytes != payload:
            raise RunIntegrityError("metrics summary changed after publication")
        completion = {
            "action_student_error_distribution": validation[
                "student_error_distribution"
            ],
            "arm": self._arm,
            "common_input_trace": summary["common_input_trace"],
            "completed_optimizer_steps": EXPECTED_STEPS,
            "factors": dict(ARM_FACTORS[self._arm]),
            "identity_sha256": self._binding.identity_sha256,
            "metrics": validation["metrics"],
            "metrics_summary": summary_observed,
            "plan_artifact_sha256": self._binding.artifact_sha256,
            "plan_sha256": self._binding.plan_sha256,
            "preflight_purpose": "training",
            "preflight_report_snapshot": self._preflight_snapshot,
            "primary_outputs": observed_primary_outputs,
            "provenance": dict(self._provenance),
            "record_type": "run_completion",
            "run_id": self._run_id,
            "schema_version": RUN_COMPLETION_SCHEMA_VERSION,
            "status": "complete",
        }
        completion_payload = canonical_json_bytes(completion)
        _exclusive_atomic_write(self._completion_path, completion_payload)
        self._finalized = True
        return FinalizedMetrics(
            summary=summary,
            summary_sha256=sha256(payload).hexdigest(),
            completion=completion,
            completion_sha256=sha256(completion_payload).hexdigest(),
        )

    def __enter__(self) -> "Phase6MetricsWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "ARM_FACTORS",
    "COMMON_INPUT_TRACE_CONTRACT",
    "CROSS_ARM_TRACE_SCHEMA_VERSION",
    "FROZEN_TENSOR_VERIFICATION_SCHEMA_VERSION",
    "FinalizedMetrics",
    "NonFiniteGradientError",
    "NonFiniteModelParametersError",
    "NonFiniteOptimizerMasterParametersError",
    "NonFiniteOptimizerStateError",
    "Phase6MetricsWriter",
    "Phase6PlanBinding",
    "Phase6ValidatedLaunchContext",
    "RESOLVED_CONFIG_VERIFICATION_SCHEMA_VERSION",
    "RUN_COMPLETION_SCHEMA_VERSION",
    "RunIntegrityError",
    "STEP_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "build_frozen_tensor_verification_manifest",
    "build_resolved_config_verification_manifest",
    "canonical_json_bytes",
    "read_stable_pinned_file",
    "validate_cross_arm_common_input_traces",
    "validate_metrics_jsonl",
    "write_integrity_manifest",
]
