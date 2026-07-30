"""Torch-free, fail-closed authorization for the fixed Phase-6 five-arm launch."""

from __future__ import annotations

import hmac
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from sana_wam.train.phase6_arm_config import (
    ARM_DYNAMIC_PIN_FIELDS,
    ARM_FACTORS,
    ARM_NAMES,
    COMMON_DYNAMIC_PIN_FIELDS,
    CONFIG_NAMES,
    OPERATIONAL_ROOT,
    PRIMARY_RUN_DIRECTORIES,
    PRIMARY_RUN_IDS,
    Phase6ArmConfigError,
    build_phase6_arm_projection,
    canonical_json_bytes,
    load_arm_config_bytes,
    validate_arm_set,
    validate_phase6_arm_projection_manifest_bytes,
    validate_phase6_smoke_runtime_bundle_bytes,
)
from sana_wam.train.phase6_recovery import (
    PLANROW_RECOVERY_AUTHORITY_PATH,
    PLANROW_RECOVERY_AUTHORITY_SHA256,
    recovery_lifecycle,
    validate_phase6_planrow_recovery_authority,
)

LAUNCH_MANIFEST_SCHEMA_VERSION = "sana-phase6-launch-manifest-v2"
LAUNCH_TICKET_SCHEMA_VERSION = "sana-phase6-launch-ticket-v2"
LAUNCH_ID = "phase6_20260725_five_arm_planrowfix_r1"
TRAINING_REQUEST_SCHEMA_VERSION = "sana-phase6-preflight-request-v4"
TRAINING_REPORT_SCHEMA_VERSION = "sana-phase6-preflight-report-v4"
_REPOSITORY_ROOT = Path("/home/zch/workspace/sana-wam")
SOURCE_REPAIR_AMENDMENT_PATH = Path(
    "/DATA/share/sana_phase6_principled_constraints_20260724/"
    "integration_patches_v4/phase6_launch_runtime_symlink_source_repair_20260724_v2/"
    "source_repair_amendment.json"
)
SOURCE_REPAIR_AMENDMENT_SHA256 = str(
    "623390e935dcca25d62cdc532109246c589170186ff39d111803ebfc9b674d41",
)
_SOURCE_REPAIR_ROOT = SOURCE_REPAIR_AMENDMENT_PATH.parent
_SOURCE_REPAIR_REVIEW_SUBJECT_PATH = (
    _SOURCE_REPAIR_ROOT / "source_repair_review_subject.json"
)
_SOURCE_REPAIR_REVIEW_RECEIPT_PATHS = {
    "A": _SOURCE_REPAIR_ROOT / "reviewer_a_registration_receipt.json",
    "B": _SOURCE_REPAIR_ROOT / "reviewer_b_registration_receipt.json",
}
_SOURCE_REPAIR_BUILDER_PATH = (
    _SOURCE_REPAIR_ROOT / "build_phase6_launch_symlink_repair_artifact_dag.py"
)
_SOURCE_REPAIR_BOOTSTRAP_RECEIPT_PATH = (
    _SOURCE_REPAIR_ROOT / "bootstrap_publication_receipt.json"
)
_SOURCE_REPAIR_TEMPLATE_PATHS = {
    "source": _SOURCE_REPAIR_ROOT / "phase6_launch_manifest.py.template",
    "test": _SOURCE_REPAIR_ROOT / "test_phase6_launch_manifest.py.template",
}
_SOURCE_REPAIR_AMENDMENT_TOKEN = "@PHASE6_LAUNCH_" + "REPAIR_AMENDMENT_SHA256@"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMON_CONFIG_FIELDS = {
    "protocol": ("phase6_protocol_document", "phase6_protocol_document_sha256"),
    "registry": ("phase6_registry", "phase6_registry_sha256"),
    "storage_amendment": (
        "phase6_operational_storage_amendment",
        "phase6_operational_storage_amendment_sha256",
    ),
    "external_components_manifest": (
        "phase6_external_components_manifest",
        "phase6_external_components_manifest_sha256",
    ),
    "expansion_eligibility_amendment": (
        "phase6_expansion_eligibility_amendment",
        "phase6_expansion_eligibility_amendment_sha256",
    ),
    "padding_semantics_amendment": (
        "phase6_padding_semantics_amendment",
        "phase6_padding_semantics_amendment_sha256",
    ),
    "task_plan": ("phase6_plan_artifact", "phase6_plan_artifact_sha256"),
    **{role: (base, f"{base}_sha256") for role, base in COMMON_DYNAMIC_PIN_FIELDS},
    "student_checkpoint": ("init_checkpoint", "init_checkpoint_sha256"),
    "phase1_checkpoint": (
        "phase1_reference_checkpoint",
        "phase1_reference_checkpoint_sha256",
    ),
}
_COMMON_ROLE_ORDER = tuple(_COMMON_CONFIG_FIELDS) + ("action_stats",)
_PREFLIGHT_ARTIFACT_ROLES = (
    "registry",
    "protocol",
    "storage_amendment",
    "external_components_manifest",
    "expansion_eligibility_amendment",
    "padding_semantics_amendment",
    "task_plan",
    "dataset_contract",
    "runtime_support_manifest",
    "projection_equivalence_receipt",
    "recovery_authority",
    "reference_preflight_request",
    "reference_preflight_report",
    "reference_rebind_receipt",
    "action_reference",
    "action_reference_spot_check",
    "arm_projection_manifest",
    "smoke_preflight_request",
    "smoke_preflight_report",
    "smoke_rebind_receipt",
    "evidence_rebinding_receipt",
    "real_2b_smoke_gate",
)
_PREFLIGHT_RUNTIME_ROLES = (
    "student_checkpoint",
    "phase1_checkpoint",
    "action_stats",
)
_MANAGED_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "GDN_DISABLE_COMPILE",
    "LOCAL_RANK",
    "NCCL_NVLS_ENABLE",
    "PYTHONHASHSEED",
    "RANK",
    "SANA_PHASE6_FORMAL",
    "WORLD_SIZE",
)


class Phase6LaunchError(ValueError):
    """Raised before torch/CUDA when a launch binding is incomplete or changed."""


def _strict_json(data: bytes, name: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise Phase6LaunchError(f"duplicate JSON key in {name}: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Phase6LaunchError(f"non-finite JSON in {name}: {token}")
            ),
        )
    except Phase6LaunchError:
        raise
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
        raise Phase6LaunchError(f"invalid {name}: {exc}") from exc
    if type(value) is not dict:
        raise Phase6LaunchError(f"{name} must be one JSON object")
    if data != canonical_json_bytes(value):
        raise Phase6LaunchError(f"{name} must be canonical JSON")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Phase6LaunchError(f"{name} must be one lowercase SHA256")
    if value == "0" * 64:
        raise Phase6LaunchError(f"{name} cannot be all-zero")
    return value


def _absolute(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise Phase6LaunchError(f"{name} must be a non-empty exact path")
    if value.startswith("/"):
        return value
    if not Path(value).is_absolute():
        raise Phase6LaunchError(f"{name} must be absolute")
    return os.path.abspath(value)


def _exact_keys(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise Phase6LaunchError(
            f"{name} keys differ: expected={sorted(keys)}, got={observed}"
        )
    return value


def _require_exact_typed_value(value: Any, expected: Any, name: str) -> None:
    if type(value) is not type(expected):
        raise Phase6LaunchError(f"{name} type differs")
    if type(expected) is dict:
        _exact_keys(value, set(expected), name)
        for key in expected:
            _require_exact_typed_value(value[key], expected[key], f"{name}.{key}")
    elif type(expected) is list:
        if len(value) != len(expected):
            raise Phase6LaunchError(f"{name} length differs")
        for index, (observed, wanted) in enumerate(zip(value, expected, strict=True)):
            _require_exact_typed_value(observed, wanted, f"{name}[{index}]")
    elif value != expected:
        raise Phase6LaunchError(f"{name} differs")


def _validate_recovery_lifecycle_fields(value: Mapping[str, Any], name: str) -> None:
    for key, expected in recovery_lifecycle().items():
        observed = value.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise Phase6LaunchError(f"{name} {key} differs")
    if "closed_loop_started" in value:
        raise Phase6LaunchError(f"{name} uses the legacy closed-loop field")


def _utc(value: Any, name: str) -> datetime:
    if type(value) is not str:
        raise Phase6LaunchError(f"{name} UTC timestamp differs")
    try:
        observed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise Phase6LaunchError(f"{name} UTC timestamp differs") from exc
    if observed > datetime.now(timezone.utc):
        raise Phase6LaunchError(f"{name} is future-dated")
    return observed


def _closed_world(value: Any, *, name: str, expected_label: str) -> dict[str, Any]:
    snapshot = _exact_keys(
        value,
        {
            "checked_at_utc",
            "gpu_compute_applications",
            "label",
            "phase6_execution_processes",
            "torch_imported",
        },
        name,
    )
    _utc(snapshot["checked_at_utc"], f"{name} check")
    if (
        snapshot["gpu_compute_applications"] != []
        or snapshot["label"] != expected_label
        or snapshot["phase6_execution_processes"] != []
        or snapshot["torch_imported"] is not False
    ):
        raise Phase6LaunchError(f"{name} differs")
    return snapshot


def _prechange_validation(
    value: Any, *, name: str, expected_closed_world_label: str
) -> dict[str, Any]:
    validation = _exact_keys(
        value,
        {
            "closed_loop_started",
            "closed_world",
            "formal_training_started",
            "future_path_count",
            "source_before",
            "source_v3_file_count",
            "test_before",
            "test_file_count",
        },
        name,
    )
    _closed_world(
        validation["closed_world"],
        name=f"{name} closed world",
        expected_label=expected_closed_world_label,
    )
    for role in ("source", "test"):
        pin = _exact_keys(
            validation[f"{role}_before"],
            {"mode", "nlink", "path", "sha256", "size_bytes"},
            f"{name} {role} pin",
        )
        _absolute(pin["path"], f"{name} {role} path")
        _digest(pin["sha256"], f"{name} {role} SHA256")
        if (
            pin["mode"] != "0664"
            or pin["nlink"] != 1
            or type(pin["size_bytes"]) is not int
            or pin["size_bytes"] < 1
        ):
            raise Phase6LaunchError(f"{name} {role} pin differs")
    if (
        validation["closed_loop_started"] is not False
        or validation["formal_training_started"] is not False
        or validation["future_path_count"] != 31
        or validation["source_v3_file_count"] != 445
        or validation["test_file_count"] != 103
    ):
        raise Phase6LaunchError(f"{name} lifecycle differs")
    return validation


def _stable_prechange(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in value if key != "closed_world"}


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_read_regular_with_stat(
    path: str | os.PathLike[str], name: str, *, capture: bool
) -> tuple[str, bytes | None, os.stat_result]:
    absolute = os.fspath(path)
    try:
        lstat_before = os.lstat(absolute)
    except OSError as exc:
        raise Phase6LaunchError(f"cannot lstat {name}: {absolute}: {exc}") from exc
    if stat.S_ISLNK(lstat_before.st_mode) or not stat.S_ISREG(lstat_before.st_mode):
        raise Phase6LaunchError(f"{name} must be a regular non-symlink: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise Phase6LaunchError(f"cannot open {name}: {absolute}: {exc}") from exc
    digest = sha256()
    blocks: list[bytes] | None = [] if capture else None
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise Phase6LaunchError(f"{name} descriptor is not regular: {absolute}")
        if (descriptor_before.st_dev, descriptor_before.st_ino) != (
            lstat_before.st_dev,
            lstat_before.st_ino,
        ):
            raise Phase6LaunchError(f"{name} changed before hashing: {absolute}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
                if blocks is not None:
                    blocks.append(block)
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        lstat_after = os.lstat(absolute)
    except OSError as exc:
        raise Phase6LaunchError(
            f"cannot restat {name} after hashing: {absolute}: {exc}"
        ) from exc
    if stat.S_ISLNK(lstat_after.st_mode) or (
        _stat_signature(lstat_before),
        _stat_signature(descriptor_before),
    ) != (
        _stat_signature(lstat_after),
        _stat_signature(descriptor_after),
    ):
        raise Phase6LaunchError(f"{name} changed while hashing: {absolute}")
    return (
        digest.hexdigest(),
        b"".join(blocks) if blocks is not None else None,
        descriptor_after,
    )


def _stable_read_regular(
    path: str | os.PathLike[str], name: str, *, capture: bool
) -> tuple[str, bytes | None]:
    digest, data, _info = _stable_read_regular_with_stat(path, name, capture=capture)
    return digest, data


def _registered_json(
    pin: Any,
    *,
    expected_path: Path,
    expected_mode: str,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registered = _exact_keys(
        pin,
        {"mode", "nlink", "path", "sha256", "size_bytes"},
        f"{name} pin",
    )
    absolute = _absolute(registered["path"], f"{name}.path")
    if absolute != os.fspath(expected_path):
        raise Phase6LaunchError(f"{name} path differs")
    expected_sha256 = _digest(registered["sha256"], f"{name}.sha256")
    if (
        registered["mode"] != expected_mode
        or registered["nlink"] != 1
        or type(registered["size_bytes"]) is not int
        or registered["size_bytes"] < 1
    ):
        raise Phase6LaunchError(f"{name} registration metadata differs")
    observed_sha256, data, info = _stable_read_regular_with_stat(
        absolute, name, capture=True
    )
    if data is None:
        raise AssertionError(f"captured {name} bytes are absent")
    observed = {
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "nlink": int(info.st_nlink),
        "path": absolute,
        "sha256": observed_sha256,
        "size_bytes": len(data),
    }
    if observed != registered or not hmac.compare_digest(
        observed_sha256, expected_sha256
    ):
        raise Phase6LaunchError(f"{name} registered pin differs")
    return _strict_json(data, name), observed


def _registered_file(
    pin: Any,
    *,
    expected_path: Path,
    expected_mode: str,
    name: str,
) -> tuple[bytes, dict[str, Any]]:
    registered = _exact_keys(
        pin,
        {"mode", "nlink", "path", "sha256", "size_bytes"},
        f"{name} pin",
    )
    absolute = _absolute(registered["path"], f"{name}.path")
    if absolute != os.fspath(expected_path):
        raise Phase6LaunchError(f"{name} path differs")
    expected_sha256 = _digest(registered["sha256"], f"{name}.sha256")
    observed_sha256, data, info = _stable_read_regular_with_stat(
        absolute, name, capture=True
    )
    if data is None:
        raise AssertionError(f"captured {name} bytes are absent")
    observed = {
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "nlink": int(info.st_nlink),
        "path": absolute,
        "sha256": observed_sha256,
        "size_bytes": len(data),
    }
    if (
        observed != registered
        or registered["mode"] != expected_mode
        or registered["nlink"] != 1
        or not hmac.compare_digest(observed_sha256, expected_sha256)
    ):
        raise Phase6LaunchError(f"{name} registered pin differs")
    return data, observed


def _validate_source_repair_amendment() -> None:
    amendment_sha256 = _digest(
        SOURCE_REPAIR_AMENDMENT_SHA256, "source repair amendment SHA256"
    )
    amendment_sha, amendment_data, amendment_info = _stable_read_regular_with_stat(
        SOURCE_REPAIR_AMENDMENT_PATH,
        "source repair amendment",
        capture=True,
    )
    if amendment_data is None:
        raise AssertionError("captured source repair amendment bytes are absent")
    if (
        not hmac.compare_digest(amendment_sha, amendment_sha256)
        or stat.S_IMODE(amendment_info.st_mode) != 0o400
        or amendment_info.st_nlink != 1
    ):
        raise Phase6LaunchError("source repair amendment registration differs")
    amendment = _strict_json(amendment_data, "source repair amendment")
    _exact_keys(
        amendment,
        {
            "artifact_path",
            "artifact_role",
            "authorization",
            "cpu_only_rebind",
            "decision",
            "independent_review_registrations",
            "operational_builder",
            "prior_source_manifest",
            "registered_at_utc",
            "registration_state",
            "review_subject",
            "schema_version",
            "status",
            "supersedes",
            "versioned_outputs",
        },
        "source repair amendment",
    )
    expected_authorization = {
        "exact_template_render_only": True,
        "source_backlink_must_bind_this_amendment_path_and_sha256": True,
        "source_manifest_path_order_must_match_v3": True,
        "test_inventory_path_order_must_match_r1v2": True,
    }
    if (
        amendment["schema_version"]
        != "sana-phase6-launch-runtime-symlink-source-repair-amendment-v4"
        or amendment["artifact_path"] != os.fspath(SOURCE_REPAIR_AMENDMENT_PATH)
        or amendment["artifact_role"]
        != "launch_runtime_symlink_source_repair_amendment"
        or amendment["decision"] != "authorize_exact_template_render_only"
        or amendment["status"] != "registered"
        or amendment["authorization"] != expected_authorization
    ):
        raise Phase6LaunchError("source repair amendment authority differs")

    subject, subject_pin = _registered_json(
        amendment["review_subject"],
        expected_path=_SOURCE_REPAIR_REVIEW_SUBJECT_PATH,
        expected_mode="0400",
        name="source repair review subject",
    )
    _exact_keys(
        subject,
        {
            "amendment_target",
            "artifact_role",
            "authorized_delta",
            "backlink_render_contract",
            "bootstrap_publication",
            "cpu_only_rebind",
            "forbidden_bypass",
            "lifecycle_preconditions",
            "operational_builder",
            "prior_source",
            "r1v2_bundle",
            "schema_version",
            "source_independent_reuse",
            "versioned_outputs",
        },
        "source repair review subject",
    )
    if (
        subject["schema_version"]
        != "sana-phase6-launch-runtime-symlink-source-repair-review-subject-v2"
        or subject["artifact_role"]
        != "launch_runtime_symlink_source_repair_review_subject"
        or subject["amendment_target"]
        != {
            "mode": "0400",
            "path": os.fspath(SOURCE_REPAIR_AMENDMENT_PATH),
            "schema_version": (
                "sana-phase6-launch-runtime-symlink-source-repair-amendment-v4"
            ),
        }
        or amendment["cpu_only_rebind"] != subject["cpu_only_rebind"]
        or amendment["operational_builder"] != subject["operational_builder"]
        or amendment["versioned_outputs"] != subject["versioned_outputs"]
    ):
        raise Phase6LaunchError("source repair review subject authority differs")
    amendment_registered_at = _utc(
        amendment["registered_at_utc"], "source repair amendment registration"
    )
    subject_prechange = _prechange_validation(
        subject["lifecycle_preconditions"],
        name="source repair review-subject prechange",
        expected_closed_world_label="review-subject-prechange",
    )
    authorized_delta = _exact_keys(
        subject["authorized_delta"],
        {
            "runtime_roles",
            "source",
            "source_manifest_path_order_must_match_v3",
            "test",
            "test_inventory_path_order_must_match_r1v2",
            "unchanged_helpers",
        },
        "source repair authorized delta",
    )
    if (
        authorized_delta["runtime_roles"] != list(_PREFLIGHT_RUNTIME_ROLES)
        or authorized_delta["source_manifest_path_order_must_match_v3"] is not True
        or authorized_delta["test_inventory_path_order_must_match_r1v2"] is not True
    ):
        raise Phase6LaunchError("source repair authorized delta differs")
    source_delta = _exact_keys(
        authorized_delta["source"],
        {"before_sha256", "before_size_bytes", "path"},
        "source repair source delta",
    )
    test_delta = _exact_keys(
        authorized_delta["test"],
        {"before_sha256", "before_size_bytes", "path"},
        "source repair test delta",
    )
    for role, delta in (("source", source_delta), ("test", test_delta)):
        _digest(delta["before_sha256"], f"source repair {role} before SHA256")
        if (
            type(delta["path"]) is not str
            or not delta["path"]
            or Path(delta["path"]).is_absolute()
            or ".." in Path(delta["path"]).parts
            or type(delta["before_size_bytes"]) is not int
            or delta["before_size_bytes"] < 1
        ):
            raise Phase6LaunchError(f"source repair {role} delta differs")
    expected_source_before = {
        "mode": "0664",
        "nlink": 1,
        "path": os.fspath(_REPOSITORY_ROOT / source_delta["path"]),
        "sha256": source_delta["before_sha256"],
        "size_bytes": source_delta["before_size_bytes"],
    }
    expected_test_before = {
        "mode": "0664",
        "nlink": 1,
        "path": os.fspath(_REPOSITORY_ROOT / test_delta["path"]),
        "sha256": test_delta["before_sha256"],
        "size_bytes": test_delta["before_size_bytes"],
    }
    if (
        subject_prechange["source_before"] != expected_source_before
        or subject_prechange["test_before"] != expected_test_before
    ):
        raise Phase6LaunchError("source repair prechange pins differ")

    _registered_file(
        subject["operational_builder"],
        expected_path=_SOURCE_REPAIR_BUILDER_PATH,
        expected_mode="0440",
        name="source repair operational builder",
    )
    bootstrap_receipt, bootstrap_pin = _registered_json(
        subject["bootstrap_publication"],
        expected_path=_SOURCE_REPAIR_BOOTSTRAP_RECEIPT_PATH,
        expected_mode="0400",
        name="source repair bootstrap receipt",
    )
    _exact_keys(
        bootstrap_receipt,
        {
            "artifact_role",
            "builder_input",
            "closed_world_after",
            "closed_world_before",
            "formal_outputs",
            "lifecycle",
            "prechange_validation",
            "registered_at_utc",
            "r1v2_bundle",
            "schema_version",
            "status",
        },
        "source repair bootstrap receipt",
    )
    bootstrap_outputs = _exact_keys(
        bootstrap_receipt["formal_outputs"],
        {"operational_builder", "source_template", "test_template"},
        "source repair bootstrap outputs",
    )
    if (
        bootstrap_receipt["schema_version"]
        != "sana-phase6-launch-source-repair-bootstrap-publication-v3"
        or bootstrap_receipt["artifact_role"] != "source_repair_bootstrap_commit_gate"
        or bootstrap_receipt["status"] != "registered"
        or bootstrap_outputs["operational_builder"] != subject["operational_builder"]
        or bootstrap_receipt["builder_input"]
        != bootstrap_outputs["operational_builder"]
        or bootstrap_receipt["r1v2_bundle"] != subject["r1v2_bundle"]
        or bootstrap_pin != subject["bootstrap_publication"]
    ):
        raise Phase6LaunchError("source repair bootstrap authority differs")
    bootstrap_prechange = _prechange_validation(
        bootstrap_receipt["prechange_validation"],
        name="source repair bootstrap prechange",
        expected_closed_world_label="bootstrap-prechange",
    )
    bootstrap_before = _closed_world(
        bootstrap_receipt["closed_world_before"],
        name="source repair bootstrap closed world before",
        expected_label="bootstrap-before",
    )
    bootstrap_after = _closed_world(
        bootstrap_receipt["closed_world_after"],
        name="source repair bootstrap closed world after",
        expected_label="bootstrap-after",
    )
    bootstrap_registered_at = _utc(
        bootstrap_receipt["registered_at_utc"], "source repair bootstrap registration"
    )
    expected_lifecycle = {
        "closed_loop_started": False,
        "formal_training_started": False,
        "gpu_executed": False,
        "source_change_started": False,
        "torch_imported": False,
    }
    if (
        _stable_prechange(bootstrap_prechange) != _stable_prechange(subject_prechange)
        or bootstrap_receipt["lifecycle"] != expected_lifecycle
        or not (
            _utc(
                bootstrap_before["checked_at_utc"],
                "source repair bootstrap before",
            )
            <= _utc(
                bootstrap_prechange["closed_world"]["checked_at_utc"],
                "source repair bootstrap prechange",
            )
            <= _utc(
                bootstrap_after["checked_at_utc"],
                "source repair bootstrap after",
            )
            <= bootstrap_registered_at
            <= amendment_registered_at
        )
    ):
        raise Phase6LaunchError("source repair bootstrap provenance differs")
    backlink = _exact_keys(
        subject["backlink_render_contract"],
        {
            "amendment_path_constant",
            "amendment_sha256_token",
            "exact_single_replacement_per_template",
            "templates",
        },
        "source repair backlink contract",
    )
    if (
        backlink["amendment_path_constant"] != os.fspath(SOURCE_REPAIR_AMENDMENT_PATH)
        or backlink["amendment_sha256_token"] != _SOURCE_REPAIR_AMENDMENT_TOKEN
        or backlink["exact_single_replacement_per_template"] is not True
    ):
        raise Phase6LaunchError("source repair backlink contract differs")
    templates = _exact_keys(
        backlink["templates"], {"source", "test"}, "source repair templates"
    )
    if (
        bootstrap_outputs["source_template"] != templates["source"]
        or bootstrap_outputs["test_template"] != templates["test"]
    ):
        raise Phase6LaunchError("source repair bootstrap template pins differ")
    for role, path in _SOURCE_REPAIR_TEMPLATE_PATHS.items():
        template_data, _template_pin = _registered_file(
            templates[role],
            expected_path=path,
            expected_mode="0400",
            name=f"source repair {role} template",
        )
        if template_data.count(_SOURCE_REPAIR_AMENDMENT_TOKEN.encode("ascii")) != 1:
            raise Phase6LaunchError(
                f"source repair {role} template amendment token differs"
            )

    registrations = amendment["independent_review_registrations"]
    if not isinstance(registrations, list) or len(registrations) != 2:
        raise Phase6LaunchError("source repair review registrations differ")
    reviewer_identities: set[str] = set()
    reviewer_tasks: set[str] = set()
    for slot, entry in zip(("A", "B"), registrations, strict=True):
        _exact_keys(
            entry,
            {"receipt", "review_slot", "reviewer_identity_sha256"},
            f"source repair reviewer {slot} registration",
        )
        receipt, receipt_pin = _registered_json(
            entry["receipt"],
            expected_path=_SOURCE_REPAIR_REVIEW_RECEIPT_PATHS[slot],
            expected_mode="0400",
            name=f"source repair reviewer {slot} receipt",
        )
        _exact_keys(
            receipt,
            {
                "artifact_role",
                "attestations",
                "decision",
                "prechange_validation",
                "registered_at_utc",
                "registration_path",
                "review_slot",
                "review_subject",
                "reviewer",
                "schema_version",
                "status",
            },
            f"source repair reviewer {slot} receipt",
        )
        reviewer = _exact_keys(
            receipt["reviewer"],
            {"identity_claim", "identity_sha256", "session_or_task_id"},
            f"source repair reviewer {slot}",
        )
        claim = _exact_keys(
            reviewer["identity_claim"],
            {"identity", "identity_namespace"},
            f"source repair reviewer {slot} identity claim",
        )
        identity_sha256 = _digest(
            reviewer["identity_sha256"],
            f"source repair reviewer {slot} identity SHA256",
        )
        for field, observed in (
            ("identity", claim["identity"]),
            ("identity namespace", claim["identity_namespace"]),
            ("session or task id", reviewer["session_or_task_id"]),
        ):
            if (
                type(observed) is not str
                or not observed
                or observed.strip() != observed
            ):
                raise Phase6LaunchError(
                    f"source repair reviewer {slot} {field} differs"
                )
        receipt_prechange = _prechange_validation(
            receipt["prechange_validation"],
            name=f"source repair reviewer {slot} prechange",
            expected_closed_world_label=f"reviewer-{slot}-prechange",
        )
        receipt_registered_at = _utc(
            receipt["registered_at_utc"],
            f"source repair reviewer {slot} registration",
        )
        expected_attestations = {
            "allowed_delta_and_templates_verified": True,
            "amendment_not_yet_published": True,
            "closed_loop_started": False,
            "cpu_only_rebind_acknowledged": True,
            "forbidden_bypass_acknowledged": True,
            "formal_training_started": False,
            "generic_symlink_policy_preserved": True,
            "gpu_executed": False,
            "independent_review": True,
            "r1v2_evidence_and_supports_verified": True,
            "source_change_started": False,
            "source_v3_and_before_hashes_verified": True,
        }
        if (
            receipt["schema_version"]
            != "sana-phase6-launch-runtime-symlink-source-repair-review-registration-v2"
            or receipt["artifact_role"]
            != "source_repair_independent_review_registration"
            or receipt["registration_path"]
            != os.fspath(_SOURCE_REPAIR_REVIEW_RECEIPT_PATHS[slot])
            or receipt["review_slot"] != slot
            or receipt["decision"] != "approve"
            or receipt["status"] != "pass"
            or receipt["attestations"] != expected_attestations
            or receipt["review_subject"] != subject_pin
            or entry["review_slot"] != slot
            or entry["receipt"] != receipt_pin
            or entry["reviewer_identity_sha256"] != identity_sha256
            or identity_sha256 != sha256(canonical_json_bytes(claim)).hexdigest()
            or _stable_prechange(receipt_prechange)
            != _stable_prechange(subject_prechange)
            or _utc(
                receipt_prechange["closed_world"]["checked_at_utc"],
                f"source repair reviewer {slot} prechange",
            )
            > receipt_registered_at
            or receipt_registered_at > amendment_registered_at
        ):
            raise Phase6LaunchError(
                f"source repair reviewer {slot} receipt authority differs"
            )
        reviewer_identities.add(identity_sha256)
        reviewer_tasks.add(reviewer["session_or_task_id"])
    if len(reviewer_identities) != 2 or len(reviewer_tasks) != 2:
        raise Phase6LaunchError("source repair reviewers are not independent")


def _validate_recovery_governance() -> dict[str, dict[str, str]]:
    """Require both the historical repair and the new recovery authority."""

    _validate_source_repair_amendment()
    try:
        validate_phase6_planrow_recovery_authority(
            repository_root=os.fspath(_REPOSITORY_ROOT)
        )
    except (OSError, TypeError, ValueError) as exc:
        raise Phase6LaunchError(
            f"plan-row recovery authority validation failed: {exc}"
        ) from exc
    return {
        "planrow_recovery_authority": {
            "path": PLANROW_RECOVERY_AUTHORITY_PATH,
            "sha256": _digest(
                PLANROW_RECOVERY_AUTHORITY_SHA256,
                "plan-row recovery authority SHA256",
            ),
        },
        "prior_source_repair_amendment": {
            "path": os.fspath(SOURCE_REPAIR_AMENDMENT_PATH),
            "sha256": _digest(
                SOURCE_REPAIR_AMENDMENT_SHA256,
                "source repair amendment SHA256",
            ),
        },
    }


def _sha256_file(path: str | os.PathLike[str], name: str = "file") -> str:
    return _stable_read_regular(path, name, capture=False)[0]


def _regular_file(path: str, name: str) -> None:
    _stable_read_regular(path, name, capture=False)


def _followed_regular_file(path: str, name: str) -> None:
    """Allow a stable venv interpreter symlink while rejecting non-files."""

    try:
        link_before = os.lstat(path)
        target_before = os.stat(path)
        if not stat.S_ISREG(target_before.st_mode):
            raise Phase6LaunchError(f"{name} target must be regular: {path}")
        link_after = os.lstat(path)
        target_after = os.stat(path)
    except OSError as exc:
        raise Phase6LaunchError(f"cannot stat {name}: {path}: {exc}") from exc
    if (_stat_signature(link_before), _stat_signature(target_before)) != (
        _stat_signature(link_after),
        _stat_signature(target_after),
    ):
        raise Phase6LaunchError(f"{name} changed while resolving: {path}")


def _pin(
    path: Any,
    digest: Any,
    name: str,
    *,
    verify_file: bool,
    runtime_file: bool = False,
) -> dict[str, str]:
    absolute = _absolute(path, f"{name}.path")
    expected = _digest(digest, f"{name}.sha256")
    if verify_file:
        if runtime_file:
            try:
                from sana_wam.train.phase6_artifact_dag import (
                    Phase6ArtifactDagError,
                    stable_runtime_file_sha256,
                )

                observed = stable_runtime_file_sha256(
                    absolute, expected_sha256=expected
                )
            except Phase6ArtifactDagError as exc:
                raise Phase6LaunchError(
                    f"{name} runtime-file validation failed: {exc}"
                ) from exc
        else:
            observed = _sha256_file(absolute, name)
        if not hmac.compare_digest(observed, expected):
            raise Phase6LaunchError(
                f"{name} SHA256 mismatch: expected {expected}, got {observed}"
            )
    return {"path": absolute, "sha256": expected}


def _config_common_artifacts(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    training = config["training"]
    result = {
        role: _pin(training[path_key], training[digest_key], role, verify_file=False)
        for role, (path_key, digest_key) in _COMMON_CONFIG_FIELDS.items()
    }
    result["action_stats"] = _pin(
        config["dataloader"]["action_stats_path"],
        training["action_stats_sha256"],
        "action_stats",
        verify_file=False,
    )
    return {role: result[role] for role in _COMMON_ROLE_ORDER}


def _arm_artifacts(
    arm: str,
    config_path: Path,
    config: Mapping[str, Any],
    *,
    raw_yaml_sha256: str | None = None,
) -> dict[str, dict[str, str]]:
    training = config["training"]
    values = {
        "raw_yaml": {
            "path": os.path.abspath(config_path),
            "sha256": raw_yaml_sha256
            if raw_yaml_sha256 is not None
            else _sha256_file(config_path, f"{arm} raw YAML"),
        }
    }
    for role, base in ARM_DYNAMIC_PIN_FIELDS:
        values[role] = _pin(
            training[base], training[f"{base}_sha256"], role, verify_file=False
        )
    return values


def _factors(arm: str) -> dict[str, bool]:
    continuous, expansion, adapter = ARM_FACTORS[arm]
    return {"A": adapter, "E": expansion, "T": continuous}


def _authorization(arm: str, projection_sha256: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "arm_config_projection_sha256": projection_sha256,
        "closed_loop_allowed": False,
        "factors": _factors(arm),
        "operation": "phase6_training",
        "optimizer_training_allowed": True,
        "real_2b_smoke_completed": True,
        "reference_gpu_forward_completed": True,
        "resume_allowed": False,
        "run_directory": PRIMARY_RUN_DIRECTORIES[arm],
        "run_id": PRIMARY_RUN_IDS[arm],
    }


def _managed_environment(gpu_index: int) -> dict[str, str]:
    value = {
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "GDN_DISABLE_COMPILE": "1",
        "LOCAL_RANK": "0",
        "NCCL_NVLS_ENABLE": "0",
        "PYTHONHASHSEED": "20260724",
        "RANK": "0",
        "SANA_PHASE6_FORMAL": "1",
        "WORLD_SIZE": "1",
    }
    return {key: value[key] for key in _MANAGED_ENVIRONMENT}


def _dag(
    common: Mapping[str, Any], arms: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    nodes: dict[str, str] = {}
    for role in _COMMON_ROLE_ORDER:
        nodes[f"common:{role}"] = common[role]["sha256"]
    for entry in arms:
        arm = entry["arm"]
        for role in (
            "scientific_projection",
            "training_preflight_request",
            "training_preflight_report",
            "raw_yaml",
        ):
            nodes[f"arm:{arm}:{role}"] = entry["artifacts"][role]["sha256"]
    return {
        "direct_sink_inputs": list(nodes),
        "node_sha256": nodes,
        "sink": "phase6_launch_manifest",
    }


def build_phase6_launch_manifest(
    *,
    config_dir: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    python_executable: str | os.PathLike[str],
    train_script: str | os.PathLike[str],
    ticket_dir: str | os.PathLike[str],
    verify_artifact_files: bool = True,
) -> dict[str, Any]:
    """Build the immutable DAG sink; this function never starts a process."""

    governance = _validate_recovery_governance()
    root = Path(_absolute(os.fspath(repository_root), "repository_root"))
    if not root.is_dir():
        raise Phase6LaunchError("repository_root must be an existing directory")
    config_root = Path(config_dir).resolve()
    config_paths = tuple(config_root / name for name in CONFIG_NAMES)
    if not all(path.is_file() for path in config_paths):
        raise Phase6LaunchError("all five materialized raw YAML files must exist")
    configs = []
    raw_config_sha256 = {}
    for arm, path in zip(ARM_NAMES, config_paths, strict=True):
        digest, config_data = _stable_read_regular(
            path, f"{arm} raw YAML", capture=True
        )
        if config_data is None:
            raise AssertionError("captured raw YAML bytes are absent")
        configs.append(load_arm_config_bytes(config_data, source=os.fspath(path)))
        raw_config_sha256[arm] = digest
    try:
        validate_arm_set(configs, verify_files=False)
    except Phase6ArmConfigError as exc:
        raise Phase6LaunchError(f"arm config validation failed: {exc}") from exc
    by_arm = {config["training"]["phase6_arm"]: config for config in configs}
    common = _config_common_artifacts(by_arm[ARM_NAMES[0]])
    for arm in ARM_NAMES[1:]:
        if _config_common_artifacts(by_arm[arm]) != common:
            raise Phase6LaunchError("raw YAML common artifact pins differ")
    if common["recovery_authority"] != governance["planrow_recovery_authority"]:
        raise Phase6LaunchError("raw YAML recovery-authority pin differs")
    if verify_artifact_files:
        for role, pin in common.items():
            _pin(
                pin["path"],
                pin["sha256"],
                role,
                verify_file=True,
                runtime_file=role in _PREFLIGHT_RUNTIME_ROLES,
            )

    python_path = _absolute(os.fspath(python_executable), "python_executable")
    script_path = _absolute(os.fspath(train_script), "train_script")
    expected_python = os.path.abspath(root / ".venv" / "bin" / "python")
    expected_script = os.path.abspath(root / "scripts" / "train.py")
    if python_path != expected_python or script_path != expected_script:
        raise Phase6LaunchError(
            "formal CLI must use repository .venv/bin/python and scripts/train.py"
        )
    _followed_regular_file(python_path, "python_executable")
    _regular_file(script_path, "train_script")
    if os.path.commonpath((root, Path(script_path).resolve())) != os.fspath(
        root.resolve()
    ):
        raise Phase6LaunchError("train_script must resolve inside repository_root")
    ticket_root = Path(_absolute(os.fspath(ticket_dir), "ticket_dir"))
    if not ticket_root.is_dir():
        raise Phase6LaunchError("ticket_dir must already exist")

    arms = []
    for gpu_index, arm in enumerate(ARM_NAMES):
        config_path = config_root / f"train_phase6_{arm}.yaml"
        config = by_arm[arm]
        artifacts = _arm_artifacts(
            arm,
            config_path,
            config,
            raw_yaml_sha256=raw_config_sha256[arm],
        )
        if verify_artifact_files:
            for role, pin in artifacts.items():
                _pin(pin["path"], pin["sha256"], f"{arm}.{role}", verify_file=True)
        ticket_path = os.path.abspath(ticket_root / f"phase6_launch_ticket_{arm}.json")
        arms.append(
            {
                "arm": arm,
                "argv": [
                    python_path,
                    script_path,
                    "--config",
                    artifacts["raw_yaml"]["path"],
                    "--phase6-launch-ticket",
                    ticket_path,
                ],
                "artifacts": artifacts,
                "authorization": _authorization(
                    arm, artifacts["scientific_projection"]["sha256"]
                ),
                "environment": _managed_environment(gpu_index),
                "formal_cli_overrides": [],
                "gpu_index": gpu_index,
                "output_directory": PRIMARY_RUN_DIRECTORIES[arm],
                "resume_allowed": False,
                "run_id": PRIMARY_RUN_IDS[arm],
                "ticket_path": ticket_path,
            }
        )
    manifest = {
        "arms": arms,
        "common_artifacts": common,
        "dag": _dag(common, arms),
        "governance": governance,
        **recovery_lifecycle(),
        "launch_id": LAUNCH_ID,
        "launch_mode": "validate_only_ticket_issuance",
        "managed_environment_keys": list(_MANAGED_ENVIRONMENT),
        "operational_root": OPERATIONAL_ROOT,
        "registered_before_recovery_cohort_started": True,
        "repository_root": os.fspath(root.resolve()),
        "schema_version": LAUNCH_MANIFEST_SCHEMA_VERSION,
        "working_directory": os.fspath(root.resolve()),
    }
    canonical_json_bytes(manifest)
    return manifest


def _canonical_artifact(
    path: str, digest: str, name: str, *, verify_file: bool
) -> dict[str, Any]:
    pin = _pin(path, digest, name, verify_file=False)
    if not verify_file:
        return {}
    observed, data = _stable_read_regular(pin["path"], name, capture=True)
    if data is None or observed != pin["sha256"]:
        raise Phase6LaunchError(f"{name} changed before canonical parsing")
    return _strict_json(data, name)


def _validate_preflight_pair(
    arm: str,
    entry: Mapping[str, Any],
    common: Mapping[str, Mapping[str, str]],
    repository_root: str,
    *,
    verify_files: bool,
) -> None:
    artifacts = entry["artifacts"]
    request_pin = artifacts["training_preflight_request"]
    report_pin = artifacts["training_preflight_report"]
    if not verify_files:
        return
    request = _canonical_artifact(
        request_pin["path"],
        request_pin["sha256"],
        f"{arm} preflight request",
        verify_file=True,
    )
    report = _canonical_artifact(
        report_pin["path"],
        report_pin["sha256"],
        f"{arm} preflight report",
        verify_file=True,
    )
    required_request_keys = {
        "artifacts",
        "authorization",
        "effective_output_root",
        "input_config",
        *recovery_lifecycle(),
        "purpose",
        "reference_precompute_completed",
        "registered_before_recovery_cohort_started",
        "repository_root",
        "runtime_files",
        "schema_version",
        "smoke_completed",
        "source_manifest",
    }
    _exact_keys(request, required_request_keys, f"{arm} training request")
    if (
        request["schema_version"] != TRAINING_REQUEST_SCHEMA_VERSION
        or request["purpose"] != "training"
    ):
        raise Phase6LaunchError(f"{arm} training request schema/purpose differs")
    _validate_recovery_lifecycle_fields(request, f"{arm} training request lifecycle")
    for key, expected in (
        ("reference_precompute_completed", True),
        ("smoke_completed", True),
        ("registered_before_recovery_cohort_started", True),
    ):
        if request[key] is not expected:
            raise Phase6LaunchError(f"{arm} training request {key} differs")
    if os.path.abspath(request["repository_root"]) != repository_root:
        raise Phase6LaunchError(f"{arm} training request repository differs")
    _require_exact_typed_value(
        request["authorization"],
        entry["authorization"],
        f"{arm} training request authorization",
    )
    if request["input_config"] != artifacts["scientific_projection"]:
        raise Phase6LaunchError(
            f"{arm} request input_config must be the raw projection sidecar"
        )
    expected_artifacts = {role: common[role] for role in _PREFLIGHT_ARTIFACT_ROLES}
    if request["artifacts"] != expected_artifacts:
        raise Phase6LaunchError(f"{arm} request artifact DAG differs")
    if request["source_manifest"] != common["source_manifest"]:
        raise Phase6LaunchError(f"{arm} request source-manifest pin differs")
    expected_runtime = {role: common[role] for role in _PREFLIGHT_RUNTIME_ROLES}
    if request["runtime_files"] != expected_runtime:
        raise Phase6LaunchError(f"{arm} request runtime-file pins differ")

    if report.get("schema_version") != TRAINING_REPORT_SCHEMA_VERSION:
        raise Phase6LaunchError(f"{arm} preflight report schema differs")
    if report.get("purpose") != "training" or report.get("status") != "pass":
        raise Phase6LaunchError(f"{arm} preflight report is not a training pass")
    if report.get("request_sha256") != request_pin["sha256"]:
        raise Phase6LaunchError(f"{arm} report/request SHA binding differs")
    _require_exact_typed_value(
        report.get("authorization"),
        entry["authorization"],
        f"{arm} training report authorization",
    )
    _validate_recovery_lifecycle_fields(report, f"{arm} training report lifecycle")
    for key, expected in (
        ("reference_precompute_completed", True),
        ("smoke_completed", True),
        ("registered_before_recovery_cohort_started", True),
    ):
        if report.get(key) is not expected:
            raise Phase6LaunchError(f"{arm} preflight report {key} differs")
    input_observation = report.get("input_config")
    if (
        not isinstance(input_observation, Mapping)
        or input_observation.get("sha256")
        != artifacts["scientific_projection"]["sha256"]
    ):
        raise Phase6LaunchError(f"{arm} report projection observation differs")


def _validate_smoke_with_available_contract(
    common: Mapping[str, Mapping[str, str]], config: Mapping[str, Any]
) -> None:
    smoke_request = _canonical_artifact(
        common["smoke_preflight_request"]["path"],
        common["smoke_preflight_request"]["sha256"],
        "smoke preflight request",
        verify_file=True,
    )
    smoke_report = _canonical_artifact(
        common["smoke_preflight_report"]["path"],
        common["smoke_preflight_report"]["sha256"],
        "smoke preflight report",
        verify_file=True,
    )
    if smoke_request.get("purpose") != "smoke":
        raise Phase6LaunchError("smoke preflight request purpose differs")
    if smoke_request.get("input_config") != common["smoke_runtime_config"]:
        raise Phase6LaunchError("smoke request does not pin the runtime-config bundle")
    if (
        smoke_report.get("purpose") != "smoke"
        or smoke_report.get("status") != "pass"
        or smoke_report.get("request_sha256")
        != common["smoke_preflight_request"]["sha256"]
    ):
        raise Phase6LaunchError("smoke preflight report/request binding differs")
    try:
        validate_phase6_smoke_runtime_bundle_bytes(
            Path(common["smoke_runtime_config"]["path"]).read_bytes(),
            expected_artifact_sha256=common["smoke_runtime_config"]["sha256"],
            expected_projection_manifest_sha256=common["arm_projection_manifest"][
                "sha256"
            ],
        )
    except Phase6ArmConfigError as exc:
        raise Phase6LaunchError(
            f"smoke runtime-config validation failed: {exc}"
        ) from exc
    try:
        from sana_wam.train.phase6_smoke_gate import (
            validate_phase6_smoke_artifact_bytes,
        )
    except ImportError as exc:
        raise Phase6LaunchError(
            "the source-pinned real-2B smoke validator is unavailable"
        ) from exc
    training = config["training"]
    try:
        validate_phase6_smoke_artifact_bytes(
            Path(common["real_2b_smoke_gate"]["path"]).read_bytes(),
            expected_artifact_sha256=common["real_2b_smoke_gate"]["sha256"],
            expected_source_manifest_sha256=common["source_manifest"]["sha256"],
            expected_runtime_support_manifest_sha256=common["runtime_support_manifest"][
                "sha256"
            ],
            expected_student_checkpoint_sha256=common["student_checkpoint"]["sha256"],
            expected_phase1_checkpoint_sha256=common["phase1_checkpoint"]["sha256"],
            expected_action_stats_sha256=common["action_stats"]["sha256"],
            expected_plan_sha256=training["phase6_plan_sha256"],
            expected_identity_sha256=training["phase6_identity_sha256"],
            expected_dataset_contract_sha256=common["dataset_contract"]["sha256"],
            expected_reference_artifact_sha256=common["action_reference"]["sha256"],
            expected_spot_artifact_sha256=common["action_reference_spot_check"][
                "sha256"
            ],
            expected_arm_projection_manifest_sha256=common["arm_projection_manifest"][
                "sha256"
            ],
            expected_smoke_runtime_config_sha256=common["smoke_runtime_config"][
                "sha256"
            ],
            expected_preflight_request_sha256=common["smoke_preflight_request"][
                "sha256"
            ],
            expected_preflight_report_sha256=common["smoke_preflight_report"]["sha256"],
        )
    except Exception as exc:
        raise Phase6LaunchError(f"real-2B smoke gate validation failed: {exc}") from exc


def validate_phase6_launch_manifest_bytes(
    data: bytes,
    *,
    expected_manifest_sha256: str | None = None,
    verify_files: bool = True,
    require_output_directories_absent: bool = True,
) -> dict[str, Any]:
    """Validate the full five-arm DAG sink without importing torch or CUDA."""

    expected_governance = _validate_recovery_governance()
    if not isinstance(data, bytes):
        raise TypeError("launch manifest must be bytes")
    observed_manifest_sha = sha256(data).hexdigest()
    if expected_manifest_sha256 is not None and not hmac.compare_digest(
        observed_manifest_sha,
        _digest(expected_manifest_sha256, "expected launch-manifest SHA256"),
    ):
        raise Phase6LaunchError("launch-manifest file SHA256 mismatch")
    manifest = _strict_json(data, "launch manifest")
    _exact_keys(
        manifest,
        {
            "arms",
            "common_artifacts",
            "dag",
            "governance",
            *recovery_lifecycle(),
            "launch_id",
            "launch_mode",
            "managed_environment_keys",
            "operational_root",
            "registered_before_recovery_cohort_started",
            "repository_root",
            "schema_version",
            "working_directory",
        },
        "launch manifest",
    )
    _validate_recovery_lifecycle_fields(manifest, "launch manifest lifecycle")
    expected_literals = {
        "launch_id": LAUNCH_ID,
        "launch_mode": "validate_only_ticket_issuance",
        "operational_root": OPERATIONAL_ROOT,
        "registered_before_recovery_cohort_started": True,
        "schema_version": LAUNCH_MANIFEST_SCHEMA_VERSION,
    }
    for key, expected in expected_literals.items():
        if type(manifest[key]) is not type(expected) or manifest[key] != expected:
            raise Phase6LaunchError(f"launch manifest {key} differs")
    if manifest["governance"] != expected_governance:
        raise Phase6LaunchError("launch manifest recovery governance differs")
    repository_root = _absolute(manifest["repository_root"], "repository_root")
    if (
        manifest["working_directory"] != repository_root
        or not Path(repository_root).is_dir()
    ):
        raise Phase6LaunchError("launch working directory/repository differs")
    if manifest["managed_environment_keys"] != list(_MANAGED_ENVIRONMENT):
        raise Phase6LaunchError("managed environment key set/order differs")
    common = _exact_keys(
        manifest["common_artifacts"], set(_COMMON_ROLE_ORDER), "common artifacts"
    )
    normalized_common = {
        role: _pin(
            common[role].get("path") if isinstance(common[role], Mapping) else None,
            common[role].get("sha256") if isinstance(common[role], Mapping) else None,
            role,
            verify_file=verify_files,
            runtime_file=role in _PREFLIGHT_RUNTIME_ROLES,
        )
        for role in _COMMON_ROLE_ORDER
    }
    if common != normalized_common:
        raise Phase6LaunchError("common artifact pins are not normalized")

    arms = manifest["arms"]
    if not isinstance(arms, list) or [item.get("arm") for item in arms] != list(
        ARM_NAMES
    ):
        raise Phase6LaunchError("launch arm order/set differs")
    configs = []
    ticket_paths: list[str] = []
    for gpu_index, (arm, entry) in enumerate(zip(ARM_NAMES, arms, strict=True)):
        _exact_keys(
            entry,
            {
                "arm",
                "argv",
                "artifacts",
                "authorization",
                "environment",
                "formal_cli_overrides",
                "gpu_index",
                "output_directory",
                "resume_allowed",
                "run_id",
                "ticket_path",
            },
            f"launch arm {arm}",
        )
        if entry["gpu_index"] != gpu_index:
            raise Phase6LaunchError(f"{arm} must be assigned GPU {gpu_index}")
        if (
            entry["run_id"] != PRIMARY_RUN_IDS[arm]
            or entry["output_directory"] != PRIMARY_RUN_DIRECTORIES[arm]
        ):
            raise Phase6LaunchError(f"{arm} run id/directory differs")
        if entry["resume_allowed"] is not False or entry["formal_cli_overrides"] != []:
            raise Phase6LaunchError(f"{arm} resume/CLI override contract differs")
        artifacts = _exact_keys(
            entry["artifacts"],
            {"raw_yaml", *(role for role, _base in ARM_DYNAMIC_PIN_FIELDS)},
            f"{arm} artifacts",
        )
        normalized_artifacts = {
            role: _pin(
                artifacts[role].get("path")
                if isinstance(artifacts[role], Mapping)
                else None,
                artifacts[role].get("sha256")
                if isinstance(artifacts[role], Mapping)
                else None,
                f"{arm}.{role}",
                verify_file=verify_files,
            )
            for role in artifacts
        }
        if artifacts != normalized_artifacts:
            raise Phase6LaunchError(f"{arm} artifact pins are not normalized")
        raw_digest, raw_data = _stable_read_regular(
            artifacts["raw_yaml"]["path"], f"{arm} raw YAML", capture=True
        )
        if raw_data is None or raw_digest != artifacts["raw_yaml"]["sha256"]:
            raise Phase6LaunchError(f"{arm} raw YAML SHA differs")
        config = load_arm_config_bytes(raw_data, source=artifacts["raw_yaml"]["path"])
        configs.append(config)
        if config["training"]["phase6_arm"] != arm:
            raise Phase6LaunchError(f"{arm} raw YAML identifies another arm")
        if _config_common_artifacts(config) != normalized_common:
            raise Phase6LaunchError(f"{arm} raw YAML common pins differ")
        expected_arm_artifacts = _arm_artifacts(
            arm,
            Path(artifacts["raw_yaml"]["path"]),
            config,
            raw_yaml_sha256=raw_digest,
        )
        if expected_arm_artifacts != normalized_artifacts:
            raise Phase6LaunchError(f"{arm} raw YAML/artifact pins differ")
        expected_projection = canonical_json_bytes(build_phase6_arm_projection(arm))
        if (
            verify_files
            and Path(artifacts["scientific_projection"]["path"]).read_bytes()
            != expected_projection
        ):
            raise Phase6LaunchError(f"{arm} raw scientific projection differs")
        _require_exact_typed_value(
            entry["authorization"],
            _authorization(arm, artifacts["scientific_projection"]["sha256"]),
            f"{arm} authorization",
        )
        expected_env = _managed_environment(gpu_index)
        if entry["environment"] != expected_env:
            raise Phase6LaunchError(f"{arm} managed environment differs")
        ticket_path = _absolute(entry["ticket_path"], f"{arm} ticket_path")
        ticket_paths.append(ticket_path)
        expected_argv = [
            entry["argv"][0]
            if isinstance(entry["argv"], list) and entry["argv"]
            else "",
            entry["argv"][1]
            if isinstance(entry["argv"], list) and len(entry["argv"]) > 1
            else "",
            "--config",
            artifacts["raw_yaml"]["path"],
            "--phase6-launch-ticket",
            ticket_path,
        ]
        if entry["argv"] != expected_argv:
            raise Phase6LaunchError(f"{arm} argv differs or contains overrides")
        python_path = _absolute(expected_argv[0], f"{arm} python")
        expected_python = os.path.abspath(
            Path(repository_root) / ".venv" / "bin" / "python"
        )
        if python_path != expected_python:
            raise Phase6LaunchError(f"{arm} Python executable differs")
        _followed_regular_file(python_path, f"{arm} python")
        train_script = _absolute(expected_argv[1], f"{arm} train script")
        if train_script != os.path.abspath(
            Path(repository_root) / "scripts" / "train.py"
        ):
            raise Phase6LaunchError(f"{arm} train script path differs")
        _regular_file(train_script, f"{arm} train script")
        if os.path.commonpath(
            (repository_root, os.path.realpath(train_script))
        ) != os.path.realpath(repository_root):
            raise Phase6LaunchError(f"{arm} train script leaves repository")
        if (
            require_output_directories_absent
            and Path(entry["output_directory"]).exists()
        ):
            raise Phase6LaunchError(f"{arm} exact run directory already exists")
        _validate_preflight_pair(
            arm, entry, normalized_common, repository_root, verify_files=verify_files
        )
    try:
        validate_arm_set(configs, verify_files=False)
        if verify_files:
            validate_phase6_arm_projection_manifest_bytes(
                Path(normalized_common["arm_projection_manifest"]["path"]).read_bytes(),
                expected_artifact_sha256=normalized_common["arm_projection_manifest"][
                    "sha256"
                ],
            )
    except Phase6ArmConfigError as exc:
        raise Phase6LaunchError(
            f"five-arm config/projection validation failed: {exc}"
        ) from exc
    if len(set(ticket_paths)) != len(ARM_NAMES):
        raise Phase6LaunchError("launch ticket paths must be distinct")
    expected_dag = _dag(normalized_common, arms)
    if manifest["dag"] != expected_dag:
        raise Phase6LaunchError("launch DAG is not the exact direct sink closure")
    if verify_files:
        _validate_smoke_with_available_contract(normalized_common, configs[0])
    return json.loads(canonical_json_bytes(manifest))


def _exclusive_write(path: Path, data: bytes, *, mode: int = 0o444) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise Phase6LaunchError(
            "registered output needs an absolute path with existing parent"
        )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite registered artifact: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_phase6_launch_manifest(
    path: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> str:
    payload = canonical_json_bytes(dict(manifest))
    validate_phase6_launch_manifest_bytes(
        payload, verify_files=True, require_output_directories_absent=True
    )
    _exclusive_write(Path(path), payload)
    return sha256(payload).hexdigest()


def _ticket_value(
    manifest_path: str,
    manifest_sha256: str,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "arm": entry["arm"],
        "argv": entry["argv"],
        "authorization": entry["authorization"],
        "config": entry["artifacts"]["raw_yaml"],
        "environment": entry["environment"],
        **recovery_lifecycle(),
        "gpu_index": entry["gpu_index"],
        "launch_manifest": {"path": manifest_path, "sha256": manifest_sha256},
        "output_directory": entry["output_directory"],
        "registered_before_recovery_cohort_started": True,
        "resume_allowed": False,
        "run_id": entry["run_id"],
        "schema_version": LAUNCH_TICKET_SCHEMA_VERSION,
    }


def issue_phase6_launch_tickets(
    manifest_path: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
    verify_files: bool = True,
) -> tuple[Path, ...]:
    """Validate only, then publish tickets; never execute an arm command."""

    manifest_absolute = _absolute(os.fspath(manifest_path), "launch manifest path")
    expected = _digest(expected_manifest_sha256, "expected launch manifest SHA256")
    observed, data = _stable_read_regular(
        manifest_absolute, "launch manifest", capture=True
    )
    if data is None or observed != expected:
        raise Phase6LaunchError("launch manifest changed or SHA256 differs")
    manifest = validate_phase6_launch_manifest_bytes(
        data,
        expected_manifest_sha256=expected,
        verify_files=verify_files,
        require_output_directories_absent=True,
    )
    targets = tuple(Path(entry["ticket_path"]) for entry in manifest["arms"])
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite launch tickets: {existing}")
    payloads = tuple(
        canonical_json_bytes(_ticket_value(manifest_absolute, expected, entry))
        for entry in manifest["arms"]
    )
    created: list[Path] = []
    try:
        for path, payload in zip(targets, payloads, strict=True):
            _exclusive_write(path, payload)
            created.append(path)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return targets


def authorize_formal_phase6_invocation(
    *,
    ticket_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    argv: Sequence[str],
    environment: Mapping[str, str],
    working_directory: str | os.PathLike[str],
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate one exact ticket/argv/env immediately before any torch import."""

    ticket_absolute = _absolute(os.fspath(ticket_path), "launch ticket")
    _ticket_digest, ticket_bytes = _stable_read_regular(
        ticket_absolute, "launch ticket", capture=True
    )
    if ticket_bytes is None:
        raise AssertionError("captured ticket bytes are absent")
    ticket = _strict_json(ticket_bytes, "launch ticket")
    _exact_keys(
        ticket,
        {
            "arm",
            "argv",
            "authorization",
            "config",
            "environment",
            *recovery_lifecycle(),
            "gpu_index",
            "launch_manifest",
            "output_directory",
            "registered_before_recovery_cohort_started",
            "resume_allowed",
            "run_id",
            "schema_version",
        },
        "launch ticket",
    )
    _validate_recovery_lifecycle_fields(ticket, "launch ticket lifecycle")
    for key, expected in (
        ("schema_version", LAUNCH_TICKET_SCHEMA_VERSION),
        ("registered_before_recovery_cohort_started", True),
        ("resume_allowed", False),
    ):
        if type(ticket[key]) is not type(expected) or ticket[key] != expected:
            raise Phase6LaunchError(f"launch ticket {key} differs")
    manifest_pin = _pin(
        ticket["launch_manifest"].get("path")
        if isinstance(ticket["launch_manifest"], Mapping)
        else None,
        ticket["launch_manifest"].get("sha256")
        if isinstance(ticket["launch_manifest"], Mapping)
        else None,
        "ticket launch manifest",
        verify_file=False,
    )
    manifest_observed, manifest_bytes = _stable_read_regular(
        manifest_pin["path"], "ticket launch manifest", capture=True
    )
    if manifest_bytes is None or manifest_observed != manifest_pin["sha256"]:
        raise Phase6LaunchError("ticket launch manifest changed or SHA256 differs")
    manifest = validate_phase6_launch_manifest_bytes(
        manifest_bytes,
        expected_manifest_sha256=manifest_pin["sha256"],
        verify_files=verify_files,
        # Other arms may already be running; only this ticket's exact directory
        # is required to remain absent immediately before its own invocation.
        require_output_directories_absent=False,
    )
    entry = next(
        (item for item in manifest["arms"] if item["arm"] == ticket["arm"]), None
    )
    if entry is None:
        raise Phase6LaunchError("launch ticket identifies no manifest arm entry")
    _require_exact_typed_value(
        ticket,
        _ticket_value(manifest_pin["path"], manifest_pin["sha256"], entry),
        "launch ticket/manifest arm entry",
    )
    config_absolute = _absolute(os.fspath(config_path), "formal config")
    if ticket["config"]["path"] != config_absolute:
        raise Phase6LaunchError("formal config path differs from ticket")
    if _sha256_file(config_absolute, "formal raw YAML") != ticket["config"]["sha256"]:
        raise Phase6LaunchError("formal raw YAML changed after ticket issuance")
    observed_argv = [
        os.path.abspath(value) if index in (0, 1, 3, 5) else value
        for index, value in enumerate(argv)
    ]
    if observed_argv != ticket["argv"]:
        raise Phase6LaunchError("formal argv differs; overrides are forbidden")
    for key in _MANAGED_ENVIRONMENT:
        if environment.get(key) != ticket["environment"][key]:
            raise Phase6LaunchError(f"formal environment {key} differs")
    if os.path.abspath(os.fspath(working_directory)) != manifest["working_directory"]:
        raise Phase6LaunchError("formal working directory differs")
    if Path(ticket["output_directory"]).exists():
        raise Phase6LaunchError("formal exact run directory already exists")
    return {
        "arm": ticket["arm"],
        "authorization": ticket["authorization"],
        "config_path": ticket["config"]["path"],
        "config_sha256": ticket["config"]["sha256"],
        "launch_manifest_path": manifest_pin["path"],
        "launch_manifest_sha256": manifest_pin["sha256"],
        "output_directory": ticket["output_directory"],
        "run_id": ticket["run_id"],
        "ticket_path": ticket_absolute,
        "ticket_sha256": sha256(ticket_bytes).hexdigest(),
    }


__all__ = [
    "LAUNCH_MANIFEST_SCHEMA_VERSION",
    "LAUNCH_TICKET_SCHEMA_VERSION",
    "Phase6LaunchError",
    "authorize_formal_phase6_invocation",
    "build_phase6_launch_manifest",
    "issue_phase6_launch_tickets",
    "validate_phase6_launch_manifest_bytes",
    "write_phase6_launch_manifest",
]
