"""Torch-free construction of the acyclic Phase-6 artifact graph.

This module only materializes immutable metadata/configuration artifacts.  It
does not import torch, inspect CUDA, create an optimizer, start training, or
run a closed-loop evaluation.  GPU-producing steps remain separate programs
and can only consume requests produced here after the existing preflight
validator has accepted them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import sys
import tempfile
import types
from typing import Any

from .phase6_downstream_pins import (
    DATASET_CONTRACT_ARTIFACT_SHA256,
    EMBEDDED_PLAN_SHA256,
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    IDENTITY_SHA256,
    PADDING_SEMANTICS_AMENDMENT_SHA256,
    PADDING_SEMANTICS_CONTRACT_SHA256,
    TASK_PLAN_ARTIFACT_SHA256,
)
from .phase6_recovery import (
    OPERATIONAL_ROOT,
    PLANROW_RECOVERY_AUTHORITY_PATH,
    PLANROW_RECOVERY_AUTHORITY_SHA256,
    RECOVERY_RUN_DIRECTORIES,
    RECOVERY_RUN_IDS,
    RECOVERY_VERSIONED_OUTPUTS,
    recovery_lifecycle,
)


DAG_PLAN_SCHEMA_VERSION = "sana-phase6-artifact-dag-plan-v2"
REQUEST_SCHEMA_VERSION = "sana-phase6-preflight-request-v4"
MATERIALIZATION_SCHEMA_VERSION = "sana-phase6-arm-materialization-pins-v2"

REPOSITORY_ROOT = "/home/zch/workspace/sana-wam"
CONFIG_ROOT = f"{REPOSITORY_ROOT}/configs/phase6/generated_v8"
FINAL_CONFIG_ROOT = f"{REPOSITORY_ROOT}/configs/phase6/materialized_v8"
TEMPLATE_ROOT = f"{REPOSITORY_ROOT}/configs/phase6/templates"
RECOVERY_REBIND_ROOT = (
    f"{OPERATIONAL_ROOT}/integration_patches_v5/"
    "phase6_plan_row_envelope_recovery_20260725_v5"
)
RECOVERY_REBIND_BUILDER = f"{REPOSITORY_ROOT}/scripts/build_phase6_recovery_rebind.py"
REFERENCE_REBIND_RECEIPT = (
    f"{RECOVERY_REBIND_ROOT}/reference_v9_cpu_rebind_receipt.json"
)
SMOKE_REBIND_RECEIPT = f"{RECOVERY_REBIND_ROOT}/smoke_v8_cpu_rebind_receipt.json"
EVIDENCE_REBINDING_RECEIPT = (
    f"{RECOVERY_REBIND_ROOT}/evidence_rebinding_receipt_v6.json"
)

ARM_NAMES = (
    "T0_E0A0",
    "T1_E0A0",
    "T1_E1A0",
    "T1_E0A1",
    "T1_E1A1",
)
ARM_FACTORS = {
    "T0_E0A0": {"A": False, "E": False, "T": False},
    "T1_E0A0": {"A": False, "E": False, "T": True},
    "T1_E1A0": {"A": False, "E": True, "T": True},
    "T1_E0A1": {"A": True, "E": False, "T": True},
    "T1_E1A1": {"A": True, "E": True, "T": True},
}

FROZEN_SHA256 = {
    "registry": "9166e1ecde74f14d7d69ae2f5072b10732b01b9f3b0004c438d0e97ed94b00b1",
    "protocol": "765fcc884cf4cb168fac676de55801667b561fde2cc0cf98a82007ef5469d879",
    "storage_amendment": "e08dd7ef33f381a34808cc166e5eae90d7ec20fd9956b481a7bf984b5a8e5c8f",
    "external_components_manifest": "6f886e00843527ec1933e2f4ebd03dd1e7bf0069e5ce0892bd834b893a6a1b2a",
    "expansion_eligibility_amendment": EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    "padding_semantics_amendment": PADDING_SEMANTICS_AMENDMENT_SHA256,
    "task_plan": TASK_PLAN_ARTIFACT_SHA256,
    "dataset_contract": DATASET_CONTRACT_ARTIFACT_SHA256,
    "student_checkpoint": "e9549aff484da56eada21972f00fb3aa2f95f8a65397a55cf0f38c79e6254137",
    "phase1_checkpoint": "ba7bb59fe7e44f6efd6a4cef79f66aff7ccf5dbe82a5a166634e35055e94f089",
    "action_stats": "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc",
    "embedded_plan": EMBEDDED_PLAN_SHA256,
    "identity": IDENTITY_SHA256,
    "recovery_authority": PLANROW_RECOVERY_AUTHORITY_SHA256,
}

REFERENCE_AUTHORIZATION = {
    "operation": "phase1_action_reference_v2_precompute",
    "arm": "T0_E0A0",
    "continuous_time": False,
    "expansion": False,
    "action_adapter_nr": False,
    "lambda_action": 0.0,
}
SMOKE_AUTHORIZATION = {
    "operation": "phase6_real_2b_smoke",
    "paired_global_step": 1,
    "arms": list(ARM_NAMES),
    "backward_arm": "T1_E1A1",
    "checkpoint_roundtrip_arm": "T1_E1A1",
    "identity_pair": ["T1_E0A0", "T1_E0A1"],
    "memory_limit_bytes": 130 * 2**30,
    "reference_gpu_forward_completed": True,
    "smoke_gpu_forward_allowed": True,
    "backward_allowed": True,
    "temporary_checkpoint_roundtrip_allowed": True,
    "optimizer_creation_allowed": False,
    "optimizer_step_allowed": False,
    "formal_training_allowed": False,
    "closed_loop_allowed": False,
}

FIXED_ARTIFACT_ROLES = (
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
)
SMOKE_ADDITIONAL_ROLES = (
    "reference_preflight_request",
    "reference_preflight_report",
    "reference_rebind_receipt",
    "action_reference",
    "action_reference_spot_check",
    "arm_projection_manifest",
)
TRAINING_ADDITIONAL_ROLES = (
    "smoke_preflight_request",
    "smoke_preflight_report",
    "smoke_rebind_receipt",
    "evidence_rebinding_receipt",
    "real_2b_smoke_gate",
)
RUNTIME_ROLES = ("student_checkpoint", "phase1_checkpoint", "action_stats")
CPU_REBIND_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "GIT_OPTIONAL_LOCKS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
}

HISTORICAL_REBIND_INPUTS = {
    "reference_request_v5": {
        "mode": "0444",
        "path": f"{OPERATIONAL_ROOT}/preflight/reference_precompute_request_v5.json",
        "sha256": "7778bd7c6fe5f2de4b89932abac0a22b3bed91c4d412a50683792b86cb039f9c",
        "size_bytes": 3521,
    },
    "reference_report_v5": {
        "mode": "0600",
        "path": f"{OPERATIONAL_ROOT}/preflight/reference_precompute_report_v5.json",
        "sha256": "12373b36f9975f2116fc3c5f11b3fc0355fe58e792b4e2c8b19d5cdae63f5f5f",
        "size_bytes": 216244,
    },
    "action_reference_v4": {
        "mode": "0400",
        "path": f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_v4.json",
        "sha256": "85bf1fae230bb5f44a980b1c7a605bb779b069509b6b6afba4cd0e67135d0145",
        "size_bytes": 313548,
    },
    "action_reference_build_v4": {
        "mode": "0400",
        "path": (
            f"{OPERATIONAL_ROOT}/reference/"
            "phase1_action_reference_v4_build_manifest.json"
        ),
        "sha256": "f7295443a725b03cfe9f01f8f549fa45ea19b3c1fde6ece8b1234ffb72a9a057",
        "size_bytes": 1770,
    },
    "action_reference_spot_v4": {
        "mode": "0400",
        "path": (f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_spot_v4.json"),
        "sha256": "5e840a6a3f1e1f28d71e9344f2f049984f10670235dd4bceda819a8f3069b535",
        "size_bytes": 3821,
    },
    "reference_numeric_payload": {
        "mode": "0400",
        "path": (
            f"{OPERATIONAL_ROOT}/integration_patches_v4/"
            "phase6_launch_runtime_symlink_source_repair_20260724_v2/"
            "reference_numeric_payload.json"
        ),
        "sha256": "50285a7ef6cf283ae474d5a71a3e9d7065f11461ca7a0072ee571665b0435d24",
        "size_bytes": 117317,
    },
    "reference_spot_numeric_payload": {
        "mode": "0400",
        "path": (
            f"{OPERATIONAL_ROOT}/integration_patches_v4/"
            "phase6_launch_runtime_symlink_source_repair_20260724_v2/"
            "reference_spot_numeric_payload.json"
        ),
        "sha256": "a282cb418958ce650c443629742e9a4dba9cee3ae1199d379cb2dd0554d78930",
        "size_bytes": 1281,
    },
    "smoke_request_v5": {
        "mode": "0444",
        "path": f"{OPERATIONAL_ROOT}/preflight/smoke_request_v5.json",
        "sha256": "7ea2530529f0705ba4ab09d4d1246ee1376e3ac371ff2a1659b5c56897dff8c0",
        "size_bytes": 5009,
    },
    "smoke_report_v5": {
        "mode": "0600",
        "path": f"{OPERATIONAL_ROOT}/preflight/smoke_report_v5.json",
        "sha256": "73ed05b5caa7168b6cc39271ce249755994b321c644cd5734d225dc74d938c8b",
        "size_bytes": 220064,
    },
    "smoke_gate_v3": {
        "mode": "0400",
        "path": f"{OPERATIONAL_ROOT}/smoke/phase6_real_2b_smoke_gate_v3.json",
        "sha256": "f707e6f84d8273f98391a88de5fc94fae8817a3266f4e0562bbc9a61067e6346",
        "size_bytes": 18101,
    },
    "smoke_nonbinding_payload": {
        "mode": "0400",
        "path": (
            f"{OPERATIONAL_ROOT}/integration_patches_v4/"
            "phase6_launch_runtime_symlink_source_repair_20260724_v2/"
            "smoke_nonbinding_payload.json"
        ),
        "sha256": "47b84b0481fd944fd043ca299ea92c00216ee13e990c3f52b90a34c4a4a48a25",
        "size_bytes": 16789,
    },
}
REFERENCE_HISTORICAL_REBIND_ROLES = (
    "reference_request_v5",
    "reference_report_v5",
    "action_reference_v4",
    "action_reference_build_v4",
    "action_reference_spot_v4",
    "reference_numeric_payload",
    "reference_spot_numeric_payload",
)
SMOKE_HISTORICAL_REBIND_ROLES = (
    "smoke_request_v5",
    "smoke_report_v5",
    "smoke_gate_v3",
    "smoke_nonbinding_payload",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_READ_BLOCK_SIZE = 8 * 1024 * 1024
_MAX_METADATA_BYTES = 128 * 1024 * 1024


class Phase6ArtifactDagError(ValueError):
    """Raised before publication when the Phase-6 graph is not exact."""


@dataclass(frozen=True)
class Phase6Paths:
    """The single frozen Phase-6 filesystem layout."""

    repository_root: str = REPOSITORY_ROOT
    operational_root: str = OPERATIONAL_ROOT
    config_root: str = CONFIG_ROOT
    final_config_root: str = FINAL_CONFIG_ROOT
    template_root: str = TEMPLATE_ROOT

    def __post_init__(self) -> None:
        expected = (
            REPOSITORY_ROOT,
            OPERATIONAL_ROOT,
            CONFIG_ROOT,
            FINAL_CONFIG_ROOT,
            TEMPLATE_ROOT,
        )
        observed = (
            self.repository_root,
            self.operational_root,
            self.config_root,
            self.final_config_root,
            self.template_root,
        )
        if observed != expected:
            raise Phase6ArtifactDagError("Phase-6 filesystem roots are frozen")

    @property
    def artifacts(self) -> dict[str, str]:
        root = self.operational_root
        return {
            "registry": f"{root}/phase6_registry_seed20260724.json",
            "protocol": f"{root}/PHASE6_PRINCIPLED_ARCHITECTURE_PROTOCOL_20260724.md",
            "storage_amendment": (
                f"{root}/phase6_operational_storage_amendment_20260724.json"
            ),
            "external_components_manifest": (
                f"{root}/phase6_external_components_manifest_20260724.json"
            ),
            "expansion_eligibility_amendment": (
                f"{root}/phase6_expansion_eligibility_amendment_20260724.json"
            ),
            "padding_semantics_amendment": (
                f"{root}/phase6_padding_semantics_amendment_20260724.json"
            ),
            "task_plan": f"{root}/artifacts/phase6_task_plan_504.json",
            "dataset_contract": (f"{root}/artifacts/phase6_dataset_contract_504.json"),
            "source_manifest": (RECOVERY_VERSIONED_OUTPUTS["source_manifest"]),
            "runtime_support_manifest": (
                f"{root}/artifacts/phase6_runtime_support_manifest_v5.json"
            ),
            "arm_projection_manifest": (
                RECOVERY_VERSIONED_OUTPUTS["arm_projection_manifest"]
            ),
            "projection_equivalence_receipt": RECOVERY_VERSIONED_OUTPUTS[
                "projection_equivalence_receipt"
            ],
            "recovery_authority": PLANROW_RECOVERY_AUTHORITY_PATH,
            "reference_preflight_request": (
                RECOVERY_VERSIONED_OUTPUTS["reference_preflight_request"]
            ),
            "reference_preflight_report": (
                RECOVERY_VERSIONED_OUTPUTS["reference_preflight_report"]
            ),
            "reference_rebind_receipt": REFERENCE_REBIND_RECEIPT,
            "action_reference": (RECOVERY_VERSIONED_OUTPUTS["reference_artifact"]),
            "action_reference_build_manifest": (
                RECOVERY_VERSIONED_OUTPUTS["reference_build_manifest"]
            ),
            "action_reference_spot_check": (
                RECOVERY_VERSIONED_OUTPUTS["reference_spot_artifact"]
            ),
            "smoke_preflight_request": RECOVERY_VERSIONED_OUTPUTS[
                "smoke_preflight_request"
            ],
            "smoke_preflight_report": RECOVERY_VERSIONED_OUTPUTS[
                "smoke_preflight_report"
            ],
            "real_2b_smoke_gate": RECOVERY_VERSIONED_OUTPUTS["smoke_gate"],
            "smoke_rebind_receipt": SMOKE_REBIND_RECEIPT,
            "evidence_rebinding_receipt": EVIDENCE_REBINDING_RECEIPT,
            "materialization_pins": (
                RECOVERY_VERSIONED_OUTPUTS["materialization_pins"]
            ),
            "launch_manifest": RECOVERY_VERSIONED_OUTPUTS["launch_manifest"],
        }

    @property
    def runtime_files(self) -> dict[str, str]:
        return {
            "student_checkpoint": (
                f"{self.repository_root}/logs/sana_principles_audit_20260722/"
                "trajectory_mixed_step0_run/checkpoint_step_0.safetensors"
            ),
            "phase1_checkpoint": (
                f"{self.repository_root}/logs/sana_principles_audit_20260722/"
                "phase1_weights_chunkwise_arch_run/checkpoint_step_0.safetensors"
            ),
            "action_stats": (
                f"{self.repository_root}/logs/sana_principles_audit_20260722/"
                "trajectory_mixed_step0_run/action_stats.npy"
            ),
        }

    @property
    def reference_config(self) -> str:
        return f"{self.config_root}/phase1_action_reference_precompute_recovery_v5.yaml"

    @property
    def smoke_runtime_bundle(self) -> str:
        return f"{self.config_root}/phase6_real_2b_smoke_runtime_bundle.json"

    def projection(self, arm: str) -> str:
        _require_arm(arm)
        return f"{self.config_root}/phase6_scientific_projection_{arm}.json"

    def smoke_runtime_config(self, arm: str) -> str:
        _require_arm(arm)
        return f"{self.config_root}/phase6_real_2b_smoke_{arm}.json"

    def training_request(self, arm: str) -> str:
        _require_arm(arm)
        return RECOVERY_VERSIONED_OUTPUTS["training_preflight_requests"][arm]

    def training_report(self, arm: str) -> str:
        _require_arm(arm)
        return RECOVERY_VERSIONED_OUTPUTS["training_preflight_reports"][arm]

    @property
    def ticket_root(self) -> str:
        return RECOVERY_VERSIONED_OUTPUTS["ticket_root"]


FROZEN_PATHS = Phase6Paths()


def canonical_json_bytes(value: Any) -> bytes:
    """Return sorted compact finite JSON with one newline."""

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
        raise Phase6ArtifactDagError(f"value is not canonical JSON: {exc}") from exc


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase6ArtifactDagError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(data: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Phase6ArtifactDagError(f"non-finite JSON token in {name}: {token}")
            ),
        )
    except Phase6ArtifactDagError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError, ValueError) as exc:
        raise Phase6ArtifactDagError(f"invalid {name} JSON: {exc}") from exc
    if type(value) is not dict:
        raise Phase6ArtifactDagError(f"{name} must be a JSON object")
    return value


def _require_arm(arm: Any) -> str:
    if arm not in ARM_NAMES:
        raise Phase6ArtifactDagError(f"unknown Phase-6 arm: {arm!r}")
    return arm


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Phase6ArtifactDagError(f"{name} must be a lowercase SHA256")
    if value == "0" * 64:
        raise Phase6ArtifactDagError(f"{name} cannot be an all-zero placeholder")
    return value


def _require_registered_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise Phase6ArtifactDagError(f"{name} must be a second-resolution UTC string")
    try:
        observed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise Phase6ArtifactDagError(
            f"{name} must be a second-resolution UTC string"
        ) from exc
    if observed > datetime.now(timezone.utc):
        raise Phase6ArtifactDagError(f"{name} cannot be future-dated")
    return observed


def _require_posix_absolute(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise Phase6ArtifactDagError(f"{name} must be a non-empty exact path")
    if not PurePosixPath(value).is_absolute():
        raise Phase6ArtifactDagError(f"{name} must be an absolute POSIX path")
    if ".." in PurePosixPath(value).parts:
        raise Phase6ArtifactDagError(f"{name} cannot contain '..'")
    return value


def _is_within(path: str, root: str) -> bool:
    candidate = PurePosixPath(_require_posix_absolute(path, "path"))
    parent = PurePosixPath(_require_posix_absolute(root, "root"))
    return candidate == parent or parent in candidate.parents


def _require_allowed_output(path: str, *, paths: Phase6Paths = FROZEN_PATHS) -> str:
    value = _require_posix_absolute(path, "output path")
    if not (
        _is_within(value, f"{paths.repository_root}/configs")
        or _is_within(value, paths.operational_root)
    ):
        raise Phase6ArtifactDagError(
            "outputs are restricted to repository configs or the operational root"
        )
    return value


def _validate_output_parent(path: str, *, paths: Phase6Paths) -> Path:
    """Resolve an existing non-symlink parent inside one authorized root."""

    value = _require_allowed_output(path, paths=paths)
    destination = Path(value)
    parent = destination.parent
    if not parent.is_dir():
        raise Phase6ArtifactDagError(f"artifact parent must already exist: {parent}")
    lexical_roots = (
        Path(f"{paths.repository_root}/configs"),
        Path(paths.operational_root),
    )
    selected = next(
        (root for root in lexical_roots if _is_within(value, root.as_posix())),
        None,
    )
    if selected is None:
        raise Phase6ArtifactDagError("cannot select an authorized output root")
    if not selected.is_dir() or selected.is_symlink():
        raise Phase6ArtifactDagError(
            f"authorized output root must be a real directory: {selected}"
        )
    resolved_root = selected.resolve(strict=True)
    resolved_parent = parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as exc:
        raise Phase6ArtifactDagError(
            f"output parent resolves outside authorized root: {parent}"
        ) from exc
    cursor = parent
    while cursor != selected:
        if cursor.is_symlink():
            raise Phase6ArtifactDagError(
                f"output parent chain contains symlink: {cursor}"
            )
        next_cursor = cursor.parent
        if next_cursor == cursor:
            raise Phase6ArtifactDagError("output parent chain left authorized root")
        cursor = next_cursor
    return destination


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _path_and_descriptor_signatures_match(
    path_signature: tuple[int, ...], descriptor_signature: tuple[int, ...]
) -> bool:
    if os.name == "nt":
        # Windows reports different ctime views for a surviving hard link via
        # lstat(path) and fstat(fd); both views are still checked before/after.
        return path_signature[:-1] == descriptor_signature[:-1]
    return path_signature == descriptor_signature


@dataclass(frozen=True)
class _PinnedPathSnapshot:
    declared_lstat: tuple[int, ...]
    final_symlink_target: str | None
    realpath: str
    realpath_lstat: tuple[int, ...]
    target_stat: tuple[int, ...]


def _snapshot_pinned_path(source: str) -> _PinnedPathSnapshot:
    """Bind the declared entry and the regular file reached through it."""

    try:
        declared = os.lstat(source)
        is_final_symlink = stat.S_ISLNK(declared.st_mode)
        if not (is_final_symlink or stat.S_ISREG(declared.st_mode)):
            raise Phase6ArtifactDagError(
                f"declared pinned path is neither regular nor a symlink: {source}"
            )
        final_symlink_target = os.readlink(source) if is_final_symlink else None
        realpath = os.path.realpath(source)
        resolved = os.lstat(realpath)
        target = os.stat(source, follow_symlinks=True)
    except Phase6ArtifactDagError:
        raise
    except OSError as exc:
        raise Phase6ArtifactDagError(
            f"cannot inspect pinned path {source}: {exc}"
        ) from exc
    if not stat.S_ISREG(resolved.st_mode) or not stat.S_ISREG(target.st_mode):
        raise Phase6ArtifactDagError(f"pinned target is not regular: {source}")
    if declared.st_nlink < 1 or resolved.st_nlink < 1 or target.st_nlink < 1:
        raise Phase6ArtifactDagError(f"pinned path has no link: {source}")
    resolved_signature = _stat_signature(resolved)
    target_signature = _stat_signature(target)
    if resolved_signature != target_signature:
        raise Phase6ArtifactDagError(
            f"declared path and realpath target differ: {source}"
        )
    return _PinnedPathSnapshot(
        declared_lstat=_stat_signature(declared),
        final_symlink_target=final_symlink_target,
        realpath=realpath,
        realpath_lstat=resolved_signature,
        target_stat=target_signature,
    )


def stable_file_sha256(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    metadata_only: bool = False,
) -> str:
    """Hash one unchanged regular file without following a final symlink."""

    source = os.path.abspath(os.fspath(path))
    try:
        declared = os.lstat(source)
    except OSError as exc:
        raise Phase6ArtifactDagError(
            f"cannot inspect pinned file {source}: {exc}"
        ) from exc
    if not stat.S_ISREG(declared.st_mode):
        raise Phase6ArtifactDagError(
            f"generic pinned path must be a regular file, not a final symlink: {source}"
        )
    declared_signature = _stat_signature(declared)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise Phase6ArtifactDagError(
            f"cannot open pinned file {source}: {exc}"
        ) from exc
    digest = sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Phase6ArtifactDagError(f"pinned path is not regular: {source}")
        if before.st_nlink < 1:
            raise Phase6ArtifactDagError(f"pinned file has no link: {source}")
        if not _path_and_descriptor_signatures_match(
            declared_signature, _stat_signature(before)
        ):
            raise Phase6ArtifactDagError(
                f"generic pinned path changed before hashing: {source}"
            )
        if metadata_only and before.st_size > _MAX_METADATA_BYTES:
            raise Phase6ArtifactDagError(f"metadata file is too large: {source}")
        while True:
            block = os.read(descriptor, _READ_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_signature(before) != _stat_signature(after):
        raise Phase6ArtifactDagError(f"pinned file changed while hashing: {source}")
    try:
        declared_after = os.lstat(source)
    except OSError as exc:
        raise Phase6ArtifactDagError(
            f"cannot re-inspect pinned file {source}: {exc}"
        ) from exc
    if _stat_signature(declared_after) != declared_signature or not (
        _path_and_descriptor_signatures_match(
            declared_signature, _stat_signature(after)
        )
    ):
        raise Phase6ArtifactDagError(
            f"generic pinned path changed while hashing: {source}"
        )
    observed = digest.hexdigest()
    if expected_sha256 is not None and not hmac.compare_digest(
        observed, _require_sha256(expected_sha256, f"{source} expected SHA256")
    ):
        raise Phase6ArtifactDagError(
            f"pinned file SHA256 mismatch: expected {expected_sha256}, got {observed}"
        )
    return observed


def stable_runtime_file_sha256(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> str:
    """Hash a stable runtime file, allowing its final entry to be a symlink."""

    source = os.path.abspath(os.fspath(path))
    path_before = _snapshot_pinned_path(source)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        # O_NOFOLLOW applies to the resolved final entry. The descriptor inode
        # must also equal stat(declared) and lstat(realpath) before hashing.
        descriptor = os.open(path_before.realpath, flags)
    except OSError as exc:
        raise Phase6ArtifactDagError(
            f"cannot open pinned runtime file {source}: {exc}"
        ) from exc
    digest = sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Phase6ArtifactDagError(
                f"pinned runtime target is not regular: {source}"
            )
        if before.st_nlink < 1:
            raise Phase6ArtifactDagError(f"pinned runtime file has no link: {source}")
        if _stat_signature(before) != path_before.target_stat:
            raise Phase6ArtifactDagError(
                f"pinned runtime target changed before hashing: {source}"
            )
        while True:
            block = os.read(descriptor, _READ_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_signature(before) != _stat_signature(after):
        raise Phase6ArtifactDagError(
            f"pinned runtime file changed while hashing: {source}"
        )
    path_after = _snapshot_pinned_path(source)
    if path_before.declared_lstat != path_after.declared_lstat or (
        path_before.final_symlink_target != path_after.final_symlink_target
    ):
        raise Phase6ArtifactDagError(
            f"declared runtime path changed while hashing: {source}"
        )
    if path_before.realpath != path_after.realpath:
        raise Phase6ArtifactDagError(
            f"runtime path realpath changed while hashing: {source}"
        )
    if (
        path_before.realpath_lstat != path_after.realpath_lstat
        or path_before.target_stat != path_after.target_stat
        or _stat_signature(after) != path_after.target_stat
    ):
        raise Phase6ArtifactDagError(
            f"pinned runtime target changed while hashing: {source}"
        )
    observed = digest.hexdigest()
    if expected_sha256 is not None and not hmac.compare_digest(
        observed, _require_sha256(expected_sha256, f"{source} expected SHA256")
    ):
        raise Phase6ArtifactDagError(
            f"pinned runtime file SHA256 mismatch: expected {expected_sha256}, "
            f"got {observed}"
        )
    return observed


def pin_file(
    path: str,
    *,
    expected_sha256: str | None = None,
    metadata_only: bool = False,
) -> dict[str, str]:
    exact = _require_posix_absolute(path, "pin path")
    digest = stable_file_sha256(
        exact, expected_sha256=expected_sha256, metadata_only=metadata_only
    )
    return {"path": exact, "sha256": digest}


def pin_runtime_file(path: str, *, expected_sha256: str) -> dict[str, str]:
    exact = _require_posix_absolute(path, "runtime pin path")
    digest = stable_runtime_file_sha256(exact, expected_sha256=expected_sha256)
    return {"path": exact, "sha256": digest}


def _exclusive_write(path: str, payload: bytes, *, paths: Phase6Paths) -> str:
    destination = _validate_output_parent(path, paths=paths)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    creation_mode = 0o600 if os.name == "nt" else 0o444
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            creation_mode,
        )
    except FileExistsError as exc:  # pragma: no cover - cryptographic collision
        raise Phase6ArtifactDagError(
            f"temporary output collision: {temporary}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
            linked = True
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite {destination}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if linked and os.name == "nt":
        os.chmod(destination, 0o444)
    _fsync_directory(destination.parent)
    return sha256(payload).hexdigest()


def _published_signature(path: str, expected_sha256: str) -> tuple[int, int, str]:
    source = Path(path)
    digest = stable_file_sha256(
        source, expected_sha256=expected_sha256, metadata_only=True
    )
    observed = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(observed.st_mode):
        raise Phase6ArtifactDagError(f"published output is not regular: {path}")
    return int(observed.st_dev), int(observed.st_ino), digest


def _unlink_created_output(path: str, signature: tuple[int, int, str]) -> None:
    """Roll back only an inode created and still owned by this invocation."""

    source = Path(path)
    try:
        observed = source.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (int(observed.st_dev), int(observed.st_ino)) != signature[:2]:
        raise Phase6ArtifactDagError(
            f"refusing to roll back replaced output inode: {path}"
        )
    stable_file_sha256(path, expected_sha256=signature[2], metadata_only=True)
    if os.name == "nt":
        os.chmod(source, 0o600)
    source.unlink()
    _fsync_directory(source.parent)


def _rollback_created_outputs(
    created: Sequence[tuple[str, tuple[int, int, str]]],
) -> None:
    rollback_errors: list[Exception] = []
    for path, signature in reversed(created):
        try:
            _unlink_created_output(path, signature)
        except Exception as exc:  # pragma: no cover - hostile concurrent mutation
            rollback_errors.append(exc)
    if rollback_errors:
        raise Phase6ArtifactDagError(f"controlled rollback failed: {rollback_errors}")


def _publish_payload_set_tracked(
    payloads: Mapping[str, bytes], *, paths: Phase6Paths
) -> tuple[dict[str, str], list[tuple[str, tuple[int, int, str]]]]:
    """Publish a stage transaction, accepting only exact prior stage bytes.

    Exact existing outputs support recovery after process interruption without
    overwriting or resuming a training run. A caught failure rolls back only
    files whose inode and SHA prove they were created by this invocation.
    """

    if not payloads:
        raise Phase6ArtifactDagError("payload set cannot be empty")
    normalized: set[str] = set()
    expected: dict[str, str] = {}
    missing: list[str] = []
    for output, payload in payloads.items():
        path = os.fspath(_validate_output_parent(output, paths=paths))
        normalized_path = os.path.normcase(os.path.abspath(path))
        if normalized_path in normalized:
            raise Phase6ArtifactDagError(f"duplicate stage output: {path}")
        normalized.add(normalized_path)
        digest = sha256(payload).hexdigest()
        expected[path] = digest
        if Path(path).exists() or Path(path).is_symlink():
            _published_signature(path, digest)
        else:
            missing.append(path)
    created: list[tuple[str, tuple[int, int, str]]] = []
    try:
        for path in missing:
            _exclusive_write(path, payloads[path], paths=paths)
            created.append((path, _published_signature(path, expected[path])))
    except BaseException:
        _rollback_created_outputs(created)
        raise
    return expected, created


def _publish_payload_set(
    payloads: Mapping[str, bytes], *, paths: Phase6Paths
) -> dict[str, str]:
    published, _created = _publish_payload_set_tracked(payloads, paths=paths)
    return published


def write_canonical_json_exclusive(
    path: str,
    value: Mapping[str, Any],
    *,
    paths: Phase6Paths = FROZEN_PATHS,
) -> str:
    return _exclusive_write(path, canonical_json_bytes(dict(value)), paths=paths)


def _canonical_yaml_bytes(value: Mapping[str, Any]) -> bytes:
    # The source manifest pins the PyYAML distribution version, so this emitter
    # is reproducible while the plan/dry-run import path remains stdlib-only.
    try:
        import yaml
    except ImportError as exc:
        raise Phase6ArtifactDagError("PyYAML is required to emit the config") from exc
    try:
        text = yaml.safe_dump(
            dict(value),
            allow_unicode=False,
            default_flow_style=False,
            sort_keys=False,
            width=1_000_000,
        )
        loaded = yaml.safe_load(text)
    except (yaml.YAMLError, UnicodeError, TypeError, ValueError) as exc:
        raise Phase6ArtifactDagError(f"cannot encode canonical YAML: {exc}") from exc
    if loaded != dict(value):
        raise Phase6ArtifactDagError("canonical YAML round-trip changed the config")
    try:
        return text.encode("ascii")
    except UnicodeError as exc:
        raise Phase6ArtifactDagError("canonical YAML is not ASCII") from exc


def _common_fixed_artifact_pins(
    *, paths: Phase6Paths = FROZEN_PATHS
) -> dict[str, dict[str, str]]:
    return {
        role: pin_file(
            paths.artifacts[role],
            expected_sha256=FROZEN_SHA256.get(role),
            metadata_only=True,
        )
        for role in FIXED_ARTIFACT_ROLES
    }


def _runtime_pins(*, paths: Phase6Paths = FROZEN_PATHS) -> dict[str, dict[str, str]]:
    return {
        role: pin_runtime_file(
            paths.runtime_files[role], expected_sha256=FROZEN_SHA256[role]
        )
        for role in RUNTIME_ROLES
    }


def _source_pin(*, paths: Phase6Paths = FROZEN_PATHS) -> dict[str, str]:
    return pin_file(paths.artifacts["source_manifest"], metadata_only=True)


def _base_request(
    *,
    purpose: str,
    authorization: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
    input_config: Mapping[str, str],
    lifecycle: Mapping[str, bool],
    paths: Phase6Paths,
    runtime_files: Mapping[str, Mapping[str, str]] | None = None,
    source_manifest: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    runtime_pins = (
        _runtime_pins(paths=paths) if runtime_files is None else runtime_files
    )
    source_pin = (
        _source_pin(paths=paths) if source_manifest is None else source_manifest
    )
    return {
        "authorization": deepcopy(dict(authorization)),
        "artifacts": deepcopy(dict(artifacts)),
        "effective_output_root": paths.operational_root,
        "input_config": deepcopy(dict(input_config)),
        **dict(lifecycle),
        "purpose": purpose,
        "registered_before_recovery_cohort_started": True,
        "repository_root": paths.repository_root,
        "runtime_files": deepcopy(dict(runtime_pins)),
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_manifest": deepcopy(dict(source_pin)),
    }


def _build_reference_preflight_request_from_pins(
    *,
    artifacts: Mapping[str, Mapping[str, str]],
    input_config: Mapping[str, str],
    runtime_files: Mapping[str, Mapping[str, str]],
    source_manifest: Mapping[str, str],
    paths: Phase6Paths,
) -> dict[str, Any]:
    request = _base_request(
        purpose="reference_precompute",
        authorization=REFERENCE_AUTHORIZATION,
        artifacts=artifacts,
        input_config=input_config,
        lifecycle={
            **recovery_lifecycle(),
            "reference_gpu_started": False,
        },
        paths=paths,
        runtime_files=runtime_files,
        source_manifest=source_manifest,
    )
    validate_preflight_request_shape(request, expected_purpose="reference_precompute")
    return request


def build_reference_preflight_request(
    *, paths: Phase6Paths = FROZEN_PATHS
) -> dict[str, Any]:
    return _build_reference_preflight_request_from_pins(
        artifacts=_common_fixed_artifact_pins(paths=paths),
        input_config=pin_file(paths.reference_config, metadata_only=True),
        runtime_files=_runtime_pins(paths=paths),
        source_manifest=_source_pin(paths=paths),
        paths=paths,
    )


def build_smoke_preflight_request(
    *, paths: Phase6Paths = FROZEN_PATHS
) -> dict[str, Any]:
    artifacts = _common_fixed_artifact_pins(paths=paths)
    for role in SMOKE_ADDITIONAL_ROLES:
        artifacts[role] = pin_file(paths.artifacts[role], metadata_only=True)
    request = _base_request(
        purpose="smoke",
        authorization=SMOKE_AUTHORIZATION,
        artifacts=artifacts,
        input_config=pin_file(paths.smoke_runtime_bundle, metadata_only=True),
        lifecycle={
            **recovery_lifecycle(),
            "reference_precompute_completed": True,
            "smoke_gpu_started": False,
        },
        paths=paths,
    )
    validate_preflight_request_shape(request, expected_purpose="smoke")
    return request


def training_authorization(arm: str, projection_sha256: str) -> dict[str, Any]:
    name = _require_arm(arm)
    projection = _require_sha256(projection_sha256, "projection SHA256")
    return {
        "operation": "phase6_training",
        "arm": name,
        "factors": deepcopy(ARM_FACTORS[name]),
        "arm_config_projection_sha256": projection,
        "run_id": RECOVERY_RUN_IDS[name],
        "run_directory": RECOVERY_RUN_DIRECTORIES[name],
        "reference_gpu_forward_completed": True,
        "real_2b_smoke_completed": True,
        "optimizer_training_allowed": True,
        "closed_loop_allowed": False,
        "resume_allowed": False,
    }


def build_training_preflight_request(
    arm: str, *, paths: Phase6Paths = FROZEN_PATHS
) -> dict[str, Any]:
    name = _require_arm(arm)
    artifacts = _common_fixed_artifact_pins(paths=paths)
    for role in (*SMOKE_ADDITIONAL_ROLES, *TRAINING_ADDITIONAL_ROLES):
        artifacts[role] = pin_file(paths.artifacts[role], metadata_only=True)
    projection_pin = pin_file(paths.projection(name), metadata_only=True)
    request = _base_request(
        purpose="training",
        authorization=training_authorization(name, projection_pin["sha256"]),
        artifacts=artifacts,
        input_config=projection_pin,
        lifecycle={
            **recovery_lifecycle(),
            "reference_precompute_completed": True,
            "smoke_completed": True,
        },
        paths=paths,
    )
    validate_preflight_request_shape(request, expected_purpose="training")
    return request


def _exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase6ArtifactDagError(f"{name} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise Phase6ArtifactDagError(
            f"{name} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _validate_pin_shape(value: Any, name: str) -> None:
    pin = _exact_keys(value, {"path", "sha256"}, name)
    _require_posix_absolute(pin["path"], f"{name}.path")
    _require_sha256(pin["sha256"], f"{name}.sha256")


def validate_preflight_request_shape(
    request: Mapping[str, Any], *, expected_purpose: str
) -> dict[str, Any]:
    """Validate acyclic request structure without manufacturing a report."""

    common_lifecycle = recovery_lifecycle()
    lifecycle_by_purpose = {
        "reference_precompute": {
            **common_lifecycle,
            "reference_gpu_started": False,
        },
        "smoke": {
            **common_lifecycle,
            "reference_precompute_completed": True,
            "smoke_gpu_started": False,
        },
        "training": {
            **common_lifecycle,
            "reference_precompute_completed": True,
            "smoke_completed": True,
        },
    }
    if expected_purpose not in lifecycle_by_purpose:
        raise Phase6ArtifactDagError(f"invalid request purpose: {expected_purpose}")
    lifecycle = lifecycle_by_purpose[expected_purpose]
    expected_keys = {
        "authorization",
        "artifacts",
        "effective_output_root",
        "input_config",
        *lifecycle,
        "purpose",
        "registered_before_recovery_cohort_started",
        "repository_root",
        "runtime_files",
        "schema_version",
        "source_manifest",
    }
    value = _exact_keys(request, expected_keys, "preflight request")
    if value["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise Phase6ArtifactDagError("preflight request schema differs")
    if value["purpose"] != expected_purpose:
        raise Phase6ArtifactDagError("preflight request purpose differs")
    if value["registered_before_recovery_cohort_started"] is not True:
        raise Phase6ArtifactDagError("preflight request registration differs")
    if value["repository_root"] != REPOSITORY_ROOT:
        raise Phase6ArtifactDagError("preflight repository root differs")
    if value["effective_output_root"] != OPERATIONAL_ROOT:
        raise Phase6ArtifactDagError("preflight output root differs")
    for key, expected in lifecycle.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise Phase6ArtifactDagError(f"preflight lifecycle {key} differs")
    artifact_roles = list(FIXED_ARTIFACT_ROLES)
    if expected_purpose in ("smoke", "training"):
        artifact_roles.extend(SMOKE_ADDITIONAL_ROLES)
    if expected_purpose == "training":
        artifact_roles.extend(TRAINING_ADDITIONAL_ROLES)
    artifacts = _exact_keys(value["artifacts"], set(artifact_roles), "artifacts")
    for role in artifact_roles:
        _validate_pin_shape(artifacts[role], f"artifacts.{role}")
        frozen = FROZEN_SHA256.get(role)
        if frozen is not None and artifacts[role]["sha256"] != frozen:
            raise Phase6ArtifactDagError(f"frozen artifact {role} SHA differs")
    runtime = _exact_keys(value["runtime_files"], set(RUNTIME_ROLES), "runtime")
    for role in RUNTIME_ROLES:
        _validate_pin_shape(runtime[role], f"runtime_files.{role}")
        if runtime[role]["sha256"] != FROZEN_SHA256[role]:
            raise Phase6ArtifactDagError(f"frozen runtime {role} SHA differs")
    _validate_pin_shape(value["input_config"], "input_config")
    _validate_pin_shape(value["source_manifest"], "source_manifest")
    authorization = value["authorization"]
    if expected_purpose == "reference_precompute":
        if authorization != REFERENCE_AUTHORIZATION:
            raise Phase6ArtifactDagError("reference authorization differs")
    elif expected_purpose == "smoke":
        if authorization != SMOKE_AUTHORIZATION:
            raise Phase6ArtifactDagError("smoke authorization differs")
    else:
        arm = authorization.get("arm") if isinstance(authorization, Mapping) else None
        projection_sha = value["input_config"]["sha256"]
        if authorization != training_authorization(arm, projection_sha):
            raise Phase6ArtifactDagError("training authorization differs")
    canonical_json_bytes(value)
    return deepcopy(dict(value))


def validate_preflight_request_bytes(
    data: bytes,
    *,
    expected_sha256: str,
    expected_purpose: str,
    fresh_semantic_validation: bool = False,
) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, "request artifact SHA256")
    if sha256(data).hexdigest() != expected:
        raise Phase6ArtifactDagError("preflight request file SHA256 differs")
    value = _strict_json(data, "preflight request")
    if data != canonical_json_bytes(value):
        raise Phase6ArtifactDagError("preflight request is not canonical JSON")
    validated = validate_preflight_request_shape(
        value, expected_purpose=expected_purpose
    )
    if fresh_semantic_validation:
        report = _fresh_preflight_report(validated)
        if report.get("status") != "pass" or report.get("purpose") != expected_purpose:
            raise Phase6ArtifactDagError("fresh preflight validation did not pass")
    return validated


def _fresh_preflight_report(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run the production validator but deliberately do not persist its report."""

    try:
        from sana_wam.train.phase6_preflight import validate_phase6_preflight
    except ImportError as exc:
        raise Phase6ArtifactDagError(
            "the frozen phase6_preflight module is required"
        ) from exc
    try:
        report = validate_phase6_preflight(request)
    except (OSError, TypeError, ValueError) as exc:
        raise Phase6ArtifactDagError(
            f"fresh preflight validation failed: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise Phase6ArtifactDagError("preflight validator returned a non-mapping")
    return report


def write_preflight_request(
    path: str,
    request: Mapping[str, Any],
    *,
    expected_purpose: str,
    paths: Phase6Paths = FROZEN_PATHS,
    fresh_semantic_validation: bool = True,
) -> str:
    validated = validate_preflight_request_shape(
        request, expected_purpose=expected_purpose
    )
    if fresh_semantic_validation:
        report = _fresh_preflight_report(validated)
        if report.get("status") != "pass":
            raise Phase6ArtifactDagError("fresh request validation did not pass")
    digest = write_canonical_json_exclusive(path, validated, paths=paths)
    data = Path(path).read_bytes()
    validate_preflight_request_bytes(
        data,
        expected_sha256=digest,
        expected_purpose=expected_purpose,
        # The exact mapping was freshly validated immediately before its
        # exclusive publication. Repeating multi-gigabyte runtime hashing here
        # would add no binding; the independent report CLI validates it again.
        fresh_semantic_validation=False,
    )
    return digest


def _publish_validated_request(
    path: str,
    request: Mapping[str, Any],
    *,
    expected_purpose: str,
    paths: Phase6Paths,
    already_fresh_validated: bool = False,
) -> str:
    validated = validate_preflight_request_shape(
        request, expected_purpose=expected_purpose
    )
    if not already_fresh_validated:
        report = _fresh_preflight_report(validated)
        if report.get("status") != "pass" or report.get("purpose") != expected_purpose:
            raise Phase6ArtifactDagError("fresh request validation did not pass")
    payload = canonical_json_bytes(validated)
    digest = _publish_payload_set({path: payload}, paths=paths)[path]
    validate_preflight_request_bytes(
        Path(path).read_bytes(),
        expected_sha256=digest,
        expected_purpose=expected_purpose,
        fresh_semantic_validation=False,
    )
    return digest


def _precompute_execution_contract() -> dict[str, Any]:
    return {
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
        "spot_check_global_steps": [1, 253, 504],
        "training_started": False,
    }


def build_reference_precompute_config(
    *, paths: Phase6Paths = FROZEN_PATHS
) -> dict[str, Any]:
    """Derive a non-training T0/E0/A0 config from the frozen arm projection."""

    try:
        from sana_wam.train.phase6_arm_config import build_phase6_arm_projection
    except ImportError as exc:
        raise Phase6ArtifactDagError(
            "the frozen phase6_arm_config module is required"
        ) from exc
    projection = build_phase6_arm_projection("T0_E0A0")
    if projection.get("factors") != ARM_FACTORS["T0_E0A0"]:
        raise Phase6ArtifactDagError("T0 projection factors differ")
    runtime = paths.runtime_files
    artifacts = paths.artifacts
    training = {
        "phase1_action_reference_precompute_mode": True,
        "init_checkpoint": runtime["phase1_checkpoint"],
        "init_checkpoint_sha256": FROZEN_SHA256["phase1_checkpoint"],
        "phase1_reference_checkpoint_sha256": FROZEN_SHA256["phase1_checkpoint"],
        "action_stats_sha256": FROZEN_SHA256["action_stats"],
        "phase6_registry": artifacts["registry"],
        "phase6_registry_sha256": FROZEN_SHA256["registry"],
        "phase6_expansion_eligibility_amendment": artifacts[
            "expansion_eligibility_amendment"
        ],
        "phase6_expansion_eligibility_amendment_sha256": FROZEN_SHA256[
            "expansion_eligibility_amendment"
        ],
        "phase6_expansion_support_contract_sha256": (EXPANSION_SUPPORT_CONTRACT_SHA256),
        "phase6_padding_semantics_amendment": artifacts["padding_semantics_amendment"],
        "phase6_padding_semantics_amendment_sha256": FROZEN_SHA256[
            "padding_semantics_amendment"
        ],
        "phase6_padding_semantics_contract_sha256": (PADDING_SEMANTICS_CONTRACT_SHA256),
        "phase6_plan_artifact": artifacts["task_plan"],
        "phase6_plan_artifact_sha256": FROZEN_SHA256["task_plan"],
        "phase6_plan_sha256": FROZEN_SHA256["embedded_plan"],
        "phase6_identity_sha256": FROZEN_SHA256["identity"],
        "phase6_dataset_contract_artifact": artifacts["dataset_contract"],
        "phase6_dataset_contract_artifact_sha256": FROZEN_SHA256["dataset_contract"],
        "phase6_code_source_manifest": artifacts["source_manifest"],
        "phase6_code_source_manifest_sha256": stable_file_sha256(
            artifacts["source_manifest"], metadata_only=True
        ),
        "lambda_video": 1.0,
        "lambda_action": 0.0,
        "expected_world_size": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_steps": 504,
        "seed": 20260724,
        "num_workers": 0,
    }
    forbidden_fragments = (
        "phase6_arm",
        "phase6_run_id",
        "output_dir",
        "phase6_launch",
        "save_",
        "optimizer_",
        "_lr",
        "warmup",
        "schedule",
        "deepspeed",
        "trainable_parameter_patterns",
    )
    if any(fragment in key for key in training for fragment in forbidden_fragments):
        raise AssertionError(
            "precompute training allowlist gained a formal-training key"
        )
    config = {
        "model": deepcopy(projection["model"]),
        "dataloader": deepcopy(projection["dataloader"]),
        "training": training,
    }
    envelope = {
        "schema_version": "sana-phase6-action-reference-precompute-config-v1",
        **deepcopy(config),
        "execution": _precompute_execution_contract(),
    }
    try:
        from sana_wam.train.action_reference_table import validate_precompute_config

        validate_precompute_config(
            envelope,
            reference_checkpoint_sha256=FROZEN_SHA256["phase1_checkpoint"],
            action_stats_sha256=FROZEN_SHA256["action_stats"],
            plan_sha256=FROZEN_SHA256["embedded_plan"],
            identity_sha256=FROZEN_SHA256["identity"],
            dataset_contract_artifact_sha256=FROZEN_SHA256["dataset_contract"],
        )
    except ImportError as exc:
        raise Phase6ArtifactDagError(
            "the frozen action_reference_table module is required"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise Phase6ArtifactDagError(
            f"precompute contract validation failed: {exc}"
        ) from exc
    return config


def write_reference_precompute_config(*, paths: Phase6Paths = FROZEN_PATHS) -> str:
    config = build_reference_precompute_config(paths=paths)
    payload = _canonical_yaml_bytes(config)
    digest = _exclusive_write(paths.reference_config, payload, paths=paths)
    if Path(paths.reference_config).read_bytes() != payload:
        raise Phase6ArtifactDagError("published reference config bytes changed")
    return digest


def validate_frozen_source_manifest(*, paths: Phase6Paths = FROZEN_PATHS) -> str:
    """Rebuild the source inventory and require byte-for-byte frozen equality."""

    try:
        from sana_wam.train.phase6_preflight import (
            build_phase6_source_manifest,
            load_canonical_source_manifest,
        )
    except ImportError as exc:
        raise Phase6ArtifactDagError(
            "the frozen phase6_preflight module is required"
        ) from exc
    source_path = paths.artifacts["source_manifest"]
    try:
        observed = load_canonical_source_manifest(source_path)
        rebuilt = build_phase6_source_manifest(paths.repository_root)
    except (OSError, TypeError, ValueError) as exc:
        raise Phase6ArtifactDagError(f"source-freeze validation failed: {exc}") from exc
    if observed != rebuilt or Path(source_path).read_bytes() != canonical_json_bytes(
        rebuilt
    ):
        raise Phase6ArtifactDagError(
            "source manifest no longer matches the complete runtime source inventory"
        )
    return stable_file_sha256(source_path, metadata_only=True)


def materialize_after_source_freeze(
    *, paths: Phase6Paths = FROZEN_PATHS
) -> dict[str, str]:
    """Publish runtime support, projections, and smoke configs after source freeze."""

    source_sha = validate_frozen_source_manifest(paths=paths)
    try:
        from sana_wam.train.phase6_arm_config import (
            ARM_NAMES as CONFIG_ARM_NAMES,
            TEMPLATE_NAMES,
            build_phase6_arm_projection,
            build_phase6_arm_projection_manifest,
            build_phase6_projection_equivalence_receipt,
            build_phase6_smoke_runtime_bundle,
            build_phase6_smoke_runtime_config,
            load_arm_config,
            validate_phase6_arm_projection_manifest_bytes,
            validate_phase6_projection_equivalence_receipt_bytes,
            validate_phase6_smoke_runtime_bundle_bytes,
        )
        from sana_wam.train.phase6_preflight import (
            build_phase6_runtime_support_manifest,
        )
    except ImportError as exc:
        raise Phase6ArtifactDagError("frozen Phase-6 builders are required") from exc
    if tuple(CONFIG_ARM_NAMES) != ARM_NAMES:
        raise Phase6ArtifactDagError("config builder arm order differs")
    runtime_manifest = build_phase6_runtime_support_manifest()
    projection_manifest = build_phase6_arm_projection_manifest()
    projection_manifest_bytes = canonical_json_bytes(projection_manifest)
    expected_manifest_sha = sha256(projection_manifest_bytes).hexdigest()
    validate_phase6_arm_projection_manifest_bytes(
        projection_manifest_bytes,
        expected_artifact_sha256=expected_manifest_sha,
    )
    equivalence_bytes = canonical_json_bytes(
        build_phase6_projection_equivalence_receipt()
    )
    equivalence_sha = sha256(equivalence_bytes).hexdigest()
    validate_phase6_projection_equivalence_receipt_bytes(
        equivalence_bytes,
        expected_artifact_sha256=equivalence_sha,
        verify_legacy_manifest=True,
    )

    template_pins: dict[str, dict[str, str]] = {}
    smoke_configs: dict[str, dict[str, Any]] = {}
    smoke_config_pins: dict[str, dict[str, str]] = {}
    for arm, template_name in zip(ARM_NAMES, TEMPLATE_NAMES, strict=True):
        template_path = f"{paths.template_root}/{template_name}"
        template_pin = pin_file(template_path, metadata_only=True)
        template = load_arm_config(template_path)
        projection = build_phase6_arm_projection(arm)
        if template.get("model") != projection["model"]:
            raise Phase6ArtifactDagError(f"{arm} smoke template model differs")
        if template.get("dataloader") != projection["dataloader"]:
            raise Phase6ArtifactDagError(f"{arm} smoke template dataloader differs")
        template_training = template.get("training")
        if not isinstance(template_training, Mapping) or any(
            template_training.get(key) != expected
            for key, expected in projection["training"].items()
        ):
            raise Phase6ArtifactDagError(
                f"{arm} smoke template scientific training fields differ"
            )
        config = build_phase6_smoke_runtime_config(
            arm,
            template_path=template_pin["path"],
            template_sha256=template_pin["sha256"],
            projection_manifest_sha256=expected_manifest_sha,
        )
        config_bytes = canonical_json_bytes(config)
        template_pins[arm] = template_pin
        smoke_configs[arm] = config
        smoke_config_pins[arm] = {
            "path": paths.smoke_runtime_config(arm),
            "sha256": sha256(config_bytes).hexdigest(),
        }
    smoke_bundle = build_phase6_smoke_runtime_bundle(
        projection_manifest_path=paths.artifacts["arm_projection_manifest"],
        projection_manifest_sha256=expected_manifest_sha,
        template_pins=template_pins,
        runtime_config_pins=smoke_config_pins,
        verify_files=False,
    )
    smoke_bundle_bytes = canonical_json_bytes(smoke_bundle)
    smoke_bundle_sha = sha256(smoke_bundle_bytes).hexdigest()
    validate_phase6_smoke_runtime_bundle_bytes(
        smoke_bundle_bytes,
        expected_artifact_sha256=smoke_bundle_sha,
        expected_projection_manifest_sha256=expected_manifest_sha,
        verify_files=False,
    )

    payloads = {
        paths.artifacts["runtime_support_manifest"]: canonical_json_bytes(
            runtime_manifest
        ),
        paths.artifacts["arm_projection_manifest"]: projection_manifest_bytes,
        paths.artifacts["projection_equivalence_receipt"]: equivalence_bytes,
        **{
            paths.projection(arm): canonical_json_bytes(
                build_phase6_arm_projection(arm)
            )
            for arm in ARM_NAMES
        },
        **{
            paths.smoke_runtime_config(arm): canonical_json_bytes(smoke_configs[arm])
            for arm in ARM_NAMES
        },
        paths.smoke_runtime_bundle: smoke_bundle_bytes,
    }
    published = _publish_payload_set(payloads, paths=paths)
    validate_phase6_smoke_runtime_bundle_bytes(
        Path(paths.smoke_runtime_bundle).read_bytes(),
        expected_artifact_sha256=smoke_bundle_sha,
        expected_projection_manifest_sha256=expected_manifest_sha,
        verify_files=True,
    )
    return {
        "source_manifest": source_sha,
        "runtime_support_manifest": published[
            paths.artifacts["runtime_support_manifest"]
        ],
        "arm_projection_manifest": expected_manifest_sha,
        "projection_equivalence_receipt": equivalence_sha,
        **{f"projection:{arm}": published[paths.projection(arm)] for arm in ARM_NAMES},
        **{
            f"smoke_runtime:{arm}": published[paths.smoke_runtime_config(arm)]
            for arm in ARM_NAMES
        },
        "smoke_runtime_bundle": smoke_bundle_sha,
    }


def materialize_reference_inputs(
    *, paths: Phase6Paths = FROZEN_PATHS
) -> dict[str, str]:
    config = build_reference_precompute_config(paths=paths)
    config_payload = _canonical_yaml_bytes(config)
    config_sha = sha256(config_payload).hexdigest()

    # Resolve every immutable dependency, including both large checkpoints,
    # before the temporary need to expose the config to the fresh validator.
    artifacts = _common_fixed_artifact_pins(paths=paths)
    runtime_files = _runtime_pins(paths=paths)
    source_manifest = _source_pin(paths=paths)

    published_config, owned_outputs = _publish_payload_set_tracked(
        {paths.reference_config: config_payload}, paths=paths
    )
    try:
        if published_config[paths.reference_config] != config_sha:
            raise AssertionError("reference config publication SHA differs")
        request = _build_reference_preflight_request_from_pins(
            artifacts=artifacts,
            input_config={"path": paths.reference_config, "sha256": config_sha},
            runtime_files=runtime_files,
            source_manifest=source_manifest,
            paths=paths,
        )
        report = _fresh_preflight_report(request)
        if (
            report.get("status") != "pass"
            or report.get("purpose") != "reference_precompute"
        ):
            raise Phase6ArtifactDagError("fresh request validation did not pass")
        request_payload = canonical_json_bytes(request)
        request_path = paths.artifacts["reference_preflight_request"]
        published_request, request_outputs = _publish_payload_set_tracked(
            {request_path: request_payload}, paths=paths
        )
        owned_outputs.extend(request_outputs)
        request_sha = published_request[request_path]
        validate_preflight_request_bytes(
            Path(request_path).read_bytes(),
            expected_sha256=request_sha,
            expected_purpose="reference_precompute",
            fresh_semantic_validation=False,
        )
    except BaseException:
        _rollback_created_outputs(owned_outputs)
        raise
    return {
        "reference_config": config_sha,
        "reference_preflight_request": request_sha,
    }


def materialize_smoke_request(*, paths: Phase6Paths = FROZEN_PATHS) -> dict[str, str]:
    request = build_smoke_preflight_request(paths=paths)
    digest = _publish_validated_request(
        paths.artifacts["smoke_preflight_request"],
        request,
        expected_purpose="smoke",
        paths=paths,
    )
    return {"smoke_preflight_request": digest}


def _stable_registered_metadata_bytes(
    path: str,
    *,
    expected_sha256: str,
    expected_mode: int | None,
    expected_size: int | None = None,
) -> bytes:
    declared = os.lstat(path)
    if (
        not stat.S_ISREG(declared.st_mode)
        or (
            expected_mode is not None
            and stat.S_IMODE(declared.st_mode) != expected_mode
        )
        or declared.st_nlink != 1
        or (expected_size is not None and declared.st_size != expected_size)
        or declared.st_size > _MAX_METADATA_BYTES
    ):
        raise Phase6ArtifactDagError(f"registered metadata differs: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise Phase6ArtifactDagError("O_NOFOLLOW is required for receipt reads")
    descriptor = os.open(path, flags | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            declared.st_dev,
            declared.st_ino,
        ):
            raise Phase6ArtifactDagError(
                f"registered metadata descriptor differs: {path}"
            )
        digest = sha256()
        chunks = []
        while True:
            block = os.read(descriptor, _READ_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    declared_after = os.lstat(path)
    if (
        _stat_signature(declared) != _stat_signature(before)
        or _stat_signature(before) != _stat_signature(after)
        or _stat_signature(after) != _stat_signature(declared_after)
        or not hmac.compare_digest(digest.hexdigest(), expected_sha256)
    ):
        raise Phase6ArtifactDagError(
            f"registered metadata changed or SHA256 differs: {path}"
        )
    return b"".join(chunks)


def _validate_rebind_receipt_pin(
    value: Any,
    *,
    expected_path: str,
    label: str,
    expected_mode: str | None = "0400",
) -> dict[str, Any]:
    pin = _exact_keys(
        value,
        {"mode", "path", "sha256", "size_bytes"},
        f"{label} pin",
    )
    if (
        pin["mode"] not in {"0400", "0444", "0600"}
        or (expected_mode is not None and pin["mode"] != expected_mode)
        or pin["path"] != expected_path
        or type(pin["size_bytes"]) is not int
        or pin["size_bytes"] < 1
    ):
        raise Phase6ArtifactDagError(f"{label} registration differs")
    expected_sha = _require_sha256(pin["sha256"], f"{label} SHA256")
    _stable_registered_metadata_bytes(
        expected_path,
        expected_sha256=expected_sha,
        expected_mode=int(pin["mode"], 8),
        expected_size=pin["size_bytes"],
    )
    return dict(pin)


def validate_evidence_rebinding_receipt(
    expected_sha256: str, *, paths: Phase6Paths = FROZEN_PATHS
) -> dict[str, Any]:
    """Validate the receipt-last CPU evidence commit before training requests."""

    expected_sha = _require_sha256(expected_sha256, "evidence-rebinding receipt SHA256")
    receipt_path = paths.artifacts["evidence_rebinding_receipt"]
    data = _stable_registered_metadata_bytes(
        receipt_path,
        expected_sha256=expected_sha,
        expected_mode=0o400,
    )
    value = _strict_json(data, "evidence-rebinding receipt")
    if data != canonical_json_bytes(value):
        raise Phase6ArtifactDagError("evidence-rebinding receipt is not canonical")
    _exact_keys(
        value,
        {
            "artifact_role",
            "created_at_utc",
            "gpu_reexecution_performed",
            "historical_gpu_evidence_reused",
            *recovery_lifecycle(),
            "reference_rebind_receipt",
            "registered_before_recovery_cohort_started",
            "recovery_authority",
            "schema_version",
            "smoke_rebind_receipt",
            "source_manifest_v9",
            "status",
            "torch_imported",
            "validation",
            "versioned_outputs",
        },
        "evidence-rebinding receipt",
    )
    if (
        value["schema_version"] != "sana-phase6-evidence-rebinding-receipt-v6"
        or value["artifact_role"]
        != "phase6_planrow_recovery_historical_gpu_evidence_rebinding"
        or value["status"] != "pass"
        or value["gpu_reexecution_performed"] is not False
        or value["historical_gpu_evidence_reused"] is not True
        or value["torch_imported"] is not False
        or value["registered_before_recovery_cohort_started"] is not True
        or any(
            type(value.get(key)) is not type(expected) or value.get(key) != expected
            for key, expected in recovery_lifecycle().items()
        )
        or value["validation"]
        != {
            "reference_504_numeric_projection_byte_identical": True,
            "reference_spot_3_numeric_projection_byte_identical": True,
            "smoke_scientific_projection_byte_identical": True,
        }
    ):
        raise Phase6ArtifactDagError("evidence-rebinding receipt authority differs")
    evidence_created = _require_registered_utc(
        value["created_at_utc"], "evidence-rebinding receipt.created_at_utc"
    )
    reference_pin = _validate_rebind_receipt_pin(
        value["reference_rebind_receipt"],
        expected_path=paths.artifacts["reference_rebind_receipt"],
        label="reference rebind receipt",
    )
    smoke_pin = _validate_rebind_receipt_pin(
        value["smoke_rebind_receipt"],
        expected_path=paths.artifacts["smoke_rebind_receipt"],
        label="smoke rebind receipt",
    )
    authority_pin = value["recovery_authority"]
    if (
        not isinstance(authority_pin, Mapping)
        or authority_pin.get("path") != paths.artifacts["recovery_authority"]
        or authority_pin.get("sha256") != FROZEN_SHA256["recovery_authority"]
    ):
        raise Phase6ArtifactDagError("evidence recovery-authority pin differs")
    _validate_rebind_receipt_pin(
        authority_pin,
        expected_path=paths.artifacts["recovery_authority"],
        label="evidence recovery authority",
    )
    source_sha = stable_file_sha256(
        paths.artifacts["source_manifest"], metadata_only=True
    )
    source_pin = value["source_manifest_v9"]
    if (
        not isinstance(source_pin, Mapping)
        or source_pin.get("path") != paths.artifacts["source_manifest"]
        or source_pin.get("sha256") != source_sha
    ):
        raise Phase6ArtifactDagError("evidence source-v6 pin differs")
    source_pin = _validate_rebind_receipt_pin(
        source_pin,
        expected_path=paths.artifacts["source_manifest"],
        label="evidence source-v6",
    )
    source_manifest_bytes = _stable_registered_metadata_bytes(
        source_pin["path"],
        expected_sha256=source_pin["sha256"],
        expected_mode=int(source_pin["mode"], 8),
        expected_size=source_pin["size_bytes"],
    )
    source_manifest = _strict_json(source_manifest_bytes, "source manifest-v9")
    if source_manifest_bytes != canonical_json_bytes(source_manifest):
        raise Phase6ArtifactDagError("source manifest-v9 is not canonical")
    source_files = source_manifest.get("files")
    builder_relative_path = "scripts/build_phase6_recovery_rebind.py"
    if not isinstance(source_files, list):
        raise Phase6ArtifactDagError("source manifest-v9 files differ")
    builder_entries = [
        entry
        for entry in source_files
        if isinstance(entry, Mapping) and entry.get("path") == builder_relative_path
    ]
    if len(builder_entries) != 1:
        raise Phase6ArtifactDagError("source manifest-v9 builder entry differs")
    builder_entry = _exact_keys(
        builder_entries[0],
        {"path", "realpath", "sha256", "size_bytes"},
        "source manifest-v9 builder entry",
    )
    builder_sha = _require_sha256(
        builder_entry["sha256"], "source manifest-v9 builder SHA256"
    )
    if (
        builder_entry["realpath"] != RECOVERY_REBIND_BUILDER
        or os.path.realpath(RECOVERY_REBIND_BUILDER) != RECOVERY_REBIND_BUILDER
        or type(builder_entry["size_bytes"]) is not int
        or builder_entry["size_bytes"] < 1
    ):
        raise Phase6ArtifactDagError("source manifest-v9 builder registration differs")
    builder_bytes = _stable_registered_metadata_bytes(
        RECOVERY_REBIND_BUILDER,
        expected_sha256=builder_sha,
        expected_mode=None,
        expected_size=builder_entry["size_bytes"],
    )
    outputs = _exact_keys(
        value["versioned_outputs"],
        {
            "gate_v8",
            "reference_receipt",
            "report_v10",
            "request_v10",
            "smoke_receipt",
        },
        "evidence versioned outputs",
    )
    expected_output_paths = {
        "gate_v8": paths.artifacts["real_2b_smoke_gate"],
        "reference_receipt": paths.artifacts["reference_rebind_receipt"],
        "report_v10": paths.artifacts["smoke_preflight_report"],
        "request_v10": paths.artifacts["smoke_preflight_request"],
        "smoke_receipt": paths.artifacts["smoke_rebind_receipt"],
    }
    for role, path in expected_output_paths.items():
        output_pin = outputs[role]
        if role == "reference_receipt" and output_pin != reference_pin:
            raise Phase6ArtifactDagError("evidence reference receipt replay differs")
        if role == "smoke_receipt" and output_pin != smoke_pin:
            raise Phase6ArtifactDagError("evidence smoke receipt replay differs")
        _validate_rebind_receipt_pin(
            output_pin,
            expected_path=path,
            label=f"evidence output {role}",
            expected_mode=None,
        )
    module_name = "sana_phase6_rebind_receipt_validator"
    module = types.ModuleType(module_name)
    module.__file__ = RECOVERY_REBIND_BUILDER
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(
            compile(builder_bytes, RECOVERY_REBIND_BUILDER, "exec", dont_inherit=True),
            module.__dict__,
        )
        builder_source_pin = module.pin_from_value(source_pin, "evidence source-v6")
        builder_authority_pin = module.pin_from_value(
            authority_pin, "evidence recovery authority"
        )
        builder_reference_pin = module.pin_from_value(
            reference_pin, "reference rebind receipt"
        )
        builder_smoke_pin = module.pin_from_value(smoke_pin, "smoke rebind receipt")
        reference_receipt = module.validate_reference_rebind_receipt(
            builder_reference_pin,
            source_pin=builder_source_pin,
            authority_pin=builder_authority_pin,
        )
        smoke_receipt = module.validate_smoke_rebind_receipt(
            builder_smoke_pin,
            source_pin=builder_source_pin,
            authority_pin=builder_authority_pin,
            reference_receipt_pin=builder_reference_pin,
        )
        reference_created = _require_registered_utc(
            reference_receipt.get("created_at_utc"),
            "reference rebind receipt.created_at_utc",
        )
        smoke_created = _require_registered_utc(
            smoke_receipt.get("created_at_utc"),
            "smoke rebind receipt.created_at_utc",
        )
        if reference_created > smoke_created or smoke_created != evidence_created:
            raise Phase6ArtifactDagError("rebind receipt creation chronology differs")
    except (OSError, RuntimeError, SyntaxError, TypeError, ValueError) as exc:
        raise Phase6ArtifactDagError(
            f"reference/smoke rebind receipt replay failed: {exc}"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    return {"path": receipt_path, "sha256": expected_sha}


def materialize_training_requests(
    *,
    evidence_rebinding_receipt_sha256: str | None = None,
    paths: Phase6Paths = FROZEN_PATHS,
) -> dict[str, str]:
    if evidence_rebinding_receipt_sha256 is None:
        raise Phase6ArtifactDagError(
            "training requests require the externally recorded evidence-rebinding receipt SHA256"
        )
    validate_evidence_rebinding_receipt(evidence_rebinding_receipt_sha256, paths=paths)
    requests = {
        arm: build_training_preflight_request(arm, paths=paths) for arm in ARM_NAMES
    }
    # Validate all five before publishing the first immutable request.
    for arm, request in requests.items():
        report = _fresh_preflight_report(request)
        if report.get("status") != "pass" or report.get("purpose") != "training":
            raise Phase6ArtifactDagError(f"{arm} prospective preflight did not pass")
    payloads = {
        paths.training_request(arm): canonical_json_bytes(requests[arm])
        for arm in ARM_NAMES
    }
    published = _publish_payload_set(payloads, paths=paths)
    result = {}
    for arm in ARM_NAMES:
        request_path = paths.training_request(arm)
        digest = published[request_path]
        validate_preflight_request_bytes(
            Path(request_path).read_bytes(),
            expected_sha256=digest,
            expected_purpose="training",
            fresh_semantic_validation=False,
        )
        result[arm] = digest
    return result


def _load_revalidated_report_pin(
    request_path: str,
    report_path: str,
    *,
    expected_purpose: str,
) -> tuple[dict[str, str], dict[str, str]]:
    request_sha = stable_file_sha256(request_path, metadata_only=True)
    report_sha = stable_file_sha256(report_path, metadata_only=True)
    try:
        from sana_wam.train.phase6_preflight import (
            load_and_revalidate_phase6_preflight_report,
        )

        report = load_and_revalidate_phase6_preflight_report(
            request_path,
            report_path,
            expected_request_sha256=request_sha,
            expected_report_sha256=report_sha,
            expected_purpose=expected_purpose,
        )
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise Phase6ArtifactDagError(
            f"cannot revalidate {expected_purpose} report: {exc}"
        ) from exc
    if report.get("status") != "pass":
        raise Phase6ArtifactDagError(f"{expected_purpose} report is not a pass")
    return (
        {"path": request_path, "sha256": request_sha},
        {"path": report_path, "sha256": report_sha},
    )


def build_materialization_pins(*, paths: Phase6Paths = FROZEN_PATHS) -> dict[str, Any]:
    reference_request, reference_report = _load_revalidated_report_pin(
        paths.artifacts["reference_preflight_request"],
        paths.artifacts["reference_preflight_report"],
        expected_purpose="reference_precompute",
    )
    smoke_request, smoke_report = _load_revalidated_report_pin(
        paths.artifacts["smoke_preflight_request"],
        paths.artifacts["smoke_preflight_report"],
        expected_purpose="smoke",
    )
    common_paths = {
        "dataset_contract": paths.artifacts["dataset_contract"],
        "source_manifest": paths.artifacts["source_manifest"],
        "runtime_support_manifest": paths.artifacts["runtime_support_manifest"],
        "reference_preflight_request": reference_request["path"],
        "reference_preflight_report": reference_report["path"],
        "reference_rebind_receipt": paths.artifacts["reference_rebind_receipt"],
        "action_reference": paths.artifacts["action_reference"],
        "action_reference_spot_check": paths.artifacts["action_reference_spot_check"],
        "arm_projection_manifest": paths.artifacts["arm_projection_manifest"],
        "projection_equivalence_receipt": paths.artifacts[
            "projection_equivalence_receipt"
        ],
        "recovery_authority": paths.artifacts["recovery_authority"],
        "smoke_runtime_config": paths.smoke_runtime_bundle,
        "smoke_preflight_request": smoke_request["path"],
        "smoke_preflight_report": smoke_report["path"],
        "smoke_rebind_receipt": paths.artifacts["smoke_rebind_receipt"],
        "evidence_rebinding_receipt": paths.artifacts["evidence_rebinding_receipt"],
        "real_2b_smoke_gate": paths.artifacts["real_2b_smoke_gate"],
    }
    common = {
        role: pin_file(path, metadata_only=True) for role, path in common_paths.items()
    }
    arm_values: dict[str, dict[str, dict[str, str]]] = {}
    for arm in ARM_NAMES:
        request, report = _load_revalidated_report_pin(
            paths.training_request(arm),
            paths.training_report(arm),
            expected_purpose="training",
        )
        arm_values[arm] = {
            "scientific_projection": pin_file(
                paths.projection(arm), metadata_only=True
            ),
            "training_preflight_request": request,
            "training_preflight_report": report,
        }
    value = {
        "arms": arm_values,
        "common": common,
        **recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
    }
    try:
        from sana_wam.train.phase6_arm_config import (
            validate_phase6_materialization_pins,
        )

        validate_phase6_materialization_pins(value, verify_files=True)
        # Validate the exact post-serialization object before O_EXCL. This
        # catches any dependency schema that incorrectly relies on mapping
        # insertion order even though canonical JSON sorts object keys.
        serialized_value = _strict_json(
            canonical_json_bytes(value), "serialized materialization pins"
        )
        validate_phase6_materialization_pins(serialized_value, verify_files=True)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise Phase6ArtifactDagError(
            f"materialization-pins validation failed: {exc}"
        ) from exc
    return value


def materialize_training_pins(*, paths: Phase6Paths = FROZEN_PATHS) -> dict[str, str]:
    value = build_materialization_pins(paths=paths)
    destination = paths.artifacts["materialization_pins"]
    digest = _publish_payload_set(
        {destination: canonical_json_bytes(value)}, paths=paths
    )[destination]
    validate_materialization_pins_bytes(
        Path(paths.artifacts["materialization_pins"]).read_bytes(),
        expected_sha256=digest,
    )
    return {"materialization_pins": digest}


def validate_materialization_pins_bytes(
    data: bytes, *, expected_sha256: str
) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, "materialization-pins SHA256")
    if sha256(data).hexdigest() != expected:
        raise Phase6ArtifactDagError("materialization-pins SHA256 differs")
    value = _strict_json(data, "materialization pins")
    if data != canonical_json_bytes(value):
        raise Phase6ArtifactDagError("materialization pins are not canonical JSON")
    try:
        from sana_wam.train.phase6_arm_config import (
            validate_phase6_materialization_pins,
        )

        validate_phase6_materialization_pins(value, verify_files=True)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise Phase6ArtifactDagError(
            f"materialization-pins validation failed: {exc}"
        ) from exc
    return deepcopy(value)


def materialize_final_sink(
    *,
    materialization_pins_sha256: str,
    paths: Phase6Paths = FROZEN_PATHS,
) -> dict[str, str]:
    """Call the frozen config/launch modules; never issue tickets or execute arms."""

    expected_pins = _require_sha256(
        materialization_pins_sha256, "materialization-pins SHA256"
    )
    try:
        from sana_wam.train.phase6_arm_config import (
            CONFIG_NAMES,
            load_phase6_materialization_pins,
            materialize_arm_configs,
        )
        from sana_wam.train.phase6_launch_manifest import (
            build_phase6_launch_manifest,
            validate_phase6_launch_manifest_bytes,
            write_phase6_launch_manifest,
        )
    except ImportError as exc:
        raise Phase6ArtifactDagError(
            "frozen config/launch modules are required"
        ) from exc
    pins = load_phase6_materialization_pins(
        paths.artifacts["materialization_pins"], expected_sha256=expected_pins
    )
    final_root = Path(paths.final_config_root)
    if final_root.is_symlink():
        raise Phase6ArtifactDagError("final config root cannot be a symlink")
    try:
        final_root.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        if not final_root.is_dir() or final_root.is_symlink():
            raise Phase6ArtifactDagError("final config root is not a real directory")
    else:
        _fsync_directory(final_root.parent)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".phase6-render-", dir=final_root.parent)
    )
    staging_stat = staging_root.stat(follow_symlinks=False)
    staging_signature = (int(staging_stat.st_dev), int(staging_stat.st_ino))
    try:
        staged_paths = materialize_arm_configs(
            template_dir=paths.template_root,
            output_dir=staging_root,
            materialization_pins=pins,
        )
        raw_payloads = {
            f"{paths.final_config_root}/{config_name}": staged_path.read_bytes()
            for config_name, staged_path in zip(CONFIG_NAMES, staged_paths, strict=True)
        }
    finally:
        try:
            observed_staging = staging_root.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise Phase6ArtifactDagError(
                "raw-YAML staging directory disappeared before cleanup"
            ) from exc
        if (
            not stat.S_ISDIR(observed_staging.st_mode)
            or staging_root.is_symlink()
            or (int(observed_staging.st_dev), int(observed_staging.st_ino))
            != staging_signature
            or staging_root.parent.resolve(strict=True)
            != final_root.parent.resolve(strict=True)
        ):
            raise Phase6ArtifactDagError(
                "refusing to clean a replaced raw-YAML staging directory"
            )
        shutil.rmtree(staging_root)
        _fsync_directory(staging_root.parent)
    published_configs = _publish_payload_set(raw_payloads, paths=paths)
    manifest = build_phase6_launch_manifest(
        config_dir=paths.final_config_root,
        repository_root=paths.repository_root,
        python_executable=f"{paths.repository_root}/.venv/bin/python",
        train_script=f"{paths.repository_root}/scripts/train.py",
        ticket_dir=paths.ticket_root,
        verify_artifact_files=True,
    )
    manifest_payload = canonical_json_bytes(manifest)
    expected_manifest_sha = sha256(manifest_payload).hexdigest()
    # Exercise the existing writer/validator on a disposable authorized path;
    # the registered sink is then atomically and recoverably published below.
    launch_parent = Path(paths.artifacts["launch_manifest"]).parent
    validation_path = launch_parent / (
        f".phase6-launch-validation-{os.getpid()}-{secrets.token_hex(8)}.json"
    )
    validation_signature: tuple[int, int, str] | None = None
    try:
        validation_sha = write_phase6_launch_manifest(validation_path, manifest)
        validation_signature = _published_signature(
            os.fspath(validation_path), validation_sha
        )
        if (
            validation_sha != expected_manifest_sha
            or validation_path.read_bytes() != manifest_payload
        ):
            raise Phase6ArtifactDagError("existing launch writer changed sink bytes")
    finally:
        if validation_signature is not None:
            _unlink_created_output(os.fspath(validation_path), validation_signature)
    manifest_path = paths.artifacts["launch_manifest"]
    manifest_sha = _publish_payload_set({manifest_path: manifest_payload}, paths=paths)[
        manifest_path
    ]
    validate_phase6_launch_manifest_bytes(
        Path(manifest_path).read_bytes(),
        expected_manifest_sha256=manifest_sha,
        verify_files=True,
        require_output_directories_absent=True,
    )
    return {
        **{
            f"raw_yaml:{arm}": published_configs[
                f"{paths.final_config_root}/{config_name}"
            ]
            for arm, config_name in zip(ARM_NAMES, CONFIG_NAMES, strict=True)
        },
        "launch_manifest": manifest_sha,
    }


def _pin_command(request: str, report: str, *, purpose: str) -> list[str]:
    return [
        f"{REPOSITORY_ROOT}/.venv/bin/python",
        "-B",
        f"{REPOSITORY_ROOT}/scripts/validate_phase6_artifact_preflight.py",
        "--request",
        request,
        "--request-sha256",
        "<sha256-from-prior-stage>",
        "--purpose",
        purpose,
        "--report",
        report,
    ]


def _rebind_command(stage: str, *, paths: Phase6Paths) -> list[str]:
    if stage not in {"reference", "smoke"}:
        raise Phase6ArtifactDagError(f"invalid CPU rebind stage: {stage}")
    command = [
        f"{REPOSITORY_ROOT}/.venv/bin/python",
        "-B",
        RECOVERY_REBIND_BUILDER,
        stage,
        "--source-manifest-v9",
        paths.artifacts["source_manifest"],
        "--source-manifest-v9-sha256",
        "<externally-recorded-source-manifest-v8-sha256>",
        "--source-manifest-v9-size",
        "<externally-recorded-source-manifest-v8-size>",
        "--recovery-authority",
        paths.artifacts["recovery_authority"],
        "--recovery-authority-sha256",
        "<externally-recorded-recovery-authority-sha256>",
        "--recovery-authority-size",
        "<externally-recorded-recovery-authority-size>",
        "--created-at-utc",
        "<second-resolution-utc>",
        "--publish",
    ]
    if stage == "smoke":
        command.extend(
            [
                "--reference-receipt-sha256",
                "<externally-recorded-reference-rebind-receipt-sha256>",
            ]
        )
    return command


def _artifact_dag_nodes(*, paths: Phase6Paths) -> list[dict[str, Any]]:
    """Return the single exact node sequence shared by build and validation."""

    artifacts = paths.artifacts
    nodes: list[dict[str, Any]] = [
        {
            "id": "recovery_authority",
            "depends_on": [],
            "producer": "registered recovery authority (outside this CLI)",
            "outputs": [artifacts["recovery_authority"]],
            "gpu": False,
            "training": False,
        },
        {
            "id": "source_manifest",
            "depends_on": ["recovery_authority"],
            "producer": "existing source-manifest generator (outside this CLI)",
            "outputs": [artifacts["source_manifest"]],
            "gpu": False,
            "training": False,
        },
        {
            "id": "after_source_freeze",
            "depends_on": ["source_manifest"],
            "producer": "phase6_artifact_dag after-source-freeze",
            "outputs": [
                artifacts["runtime_support_manifest"],
                artifacts["arm_projection_manifest"],
                artifacts["projection_equivalence_receipt"],
                *[paths.projection(arm) for arm in ARM_NAMES],
                *[paths.smoke_runtime_config(arm) for arm in ARM_NAMES],
                paths.smoke_runtime_bundle,
            ],
            "gpu": False,
            "training": False,
        },
        {
            "id": "reference_inputs",
            "depends_on": ["after_source_freeze"],
            "producer": "phase6_artifact_dag reference-inputs",
            "outputs": [
                paths.reference_config,
                artifacts["reference_preflight_request"],
            ],
            "gpu": False,
            "training": False,
        },
        {
            "id": "reference_numeric_cpu_rebind",
            "depends_on": ["reference_inputs"],
            "producer": "CPU metadata rebind of byte-identical registered reference payloads",
            "command": _rebind_command("reference", paths=paths),
            "environment": dict(CPU_REBIND_ENVIRONMENT),
            "inputs": {
                "current": [
                    artifacts["source_manifest"],
                    artifacts["recovery_authority"],
                    paths.reference_config,
                    artifacts["reference_preflight_request"],
                ],
                "historical": {
                    role: deepcopy(HISTORICAL_REBIND_INPUTS[role])
                    for role in REFERENCE_HISTORICAL_REBIND_ROLES
                },
            },
            "rebind_contract": {
                "gpu_reexecution_allowed": False,
                "reference_rows_byte_identical": True,
                "spot_rows_byte_identical": True,
            },
            "outputs": [
                artifacts["reference_preflight_report"],
                artifacts["action_reference"],
                artifacts["action_reference_build_manifest"],
                artifacts["action_reference_spot_check"],
                artifacts["reference_rebind_receipt"],
            ],
            "gpu": False,
            "training": False,
        },
        {
            "id": "smoke_request",
            "depends_on": ["reference_numeric_cpu_rebind"],
            "producer": "phase6_artifact_dag smoke-request",
            "outputs": [artifacts["smoke_preflight_request"]],
            "gpu": False,
            "training": False,
        },
        {
            "id": "smoke_report",
            "depends_on": ["smoke_request"],
            "producer": "independent phase6_preflight CLI",
            "command": _pin_command(
                artifacts["smoke_preflight_request"],
                artifacts["smoke_preflight_report"],
                purpose="smoke",
            ),
            "outputs": [artifacts["smoke_preflight_report"]],
            "gpu": False,
            "training": False,
        },
        {
            "id": "smoke_numeric_cpu_rebind",
            "depends_on": ["smoke_report"],
            "producer": "CPU metadata rebind of byte-identical registered smoke payload",
            "command": _rebind_command("smoke", paths=paths),
            "environment": dict(CPU_REBIND_ENVIRONMENT),
            "inputs": {
                "current": [
                    artifacts["source_manifest"],
                    artifacts["recovery_authority"],
                    artifacts["reference_rebind_receipt"],
                    artifacts["projection_equivalence_receipt"],
                    artifacts["smoke_preflight_request"],
                    artifacts["smoke_preflight_report"],
                ],
                "historical": {
                    role: deepcopy(HISTORICAL_REBIND_INPUTS[role])
                    for role in SMOKE_HISTORICAL_REBIND_ROLES
                },
            },
            "rebind_contract": {
                "gpu_reexecution_allowed": False,
                "legacy_nonbinding_payload_anchor_exact": True,
                "smoke_scientific_projection_byte_identical": True,
            },
            "outputs": [
                artifacts["real_2b_smoke_gate"],
                artifacts["smoke_rebind_receipt"],
                artifacts["evidence_rebinding_receipt"],
            ],
            "gpu": False,
            "training": False,
        },
        {
            "id": "training_requests",
            "depends_on": ["smoke_numeric_cpu_rebind"],
            "producer": "phase6_artifact_dag training-requests",
            "inputs": [artifacts["evidence_rebinding_receipt"]],
            "requires_evidence_rebinding_receipt_sha256": True,
            "outputs": [paths.training_request(arm) for arm in ARM_NAMES],
            "gpu": False,
            "training": False,
        },
        {
            "id": "training_reports",
            "depends_on": ["training_requests"],
            "producer": "five independent phase6_preflight CLI invocations",
            "commands": [
                _pin_command(
                    paths.training_request(arm),
                    paths.training_report(arm),
                    purpose="training",
                )
                for arm in ARM_NAMES
            ],
            "outputs": [paths.training_report(arm) for arm in ARM_NAMES],
            "gpu": False,
            "training": False,
        },
        {
            "id": "materialization_pins",
            "depends_on": ["training_reports"],
            "producer": "phase6_artifact_dag training-pins",
            "outputs": [artifacts["materialization_pins"]],
            "gpu": False,
            "training": False,
        },
        {
            "id": "final_sink",
            "depends_on": ["materialization_pins"],
            "producer": "phase6_artifact_dag final-sink",
            "outputs": [
                *[
                    f"{paths.final_config_root}/train_phase6_{arm}.yaml"
                    for arm in ARM_NAMES
                ],
                artifacts["launch_manifest"],
            ],
            "gpu": False,
            "training": False,
        },
    ]
    return nodes


def build_artifact_dag_plan(*, paths: Phase6Paths = FROZEN_PATHS) -> dict[str, Any]:
    """Return a deterministic, non-executing plan with explicit dependencies."""

    value = {
        "closed_loop_allowed": False,
        "formal_training_allowed": False,
        "nodes": _artifact_dag_nodes(paths=paths),
        "operational_root": paths.operational_root,
        "repository_root": paths.repository_root,
        "schema_version": DAG_PLAN_SCHEMA_VERSION,
        "source_manifest_generated_by_this_cli": False,
    }
    validate_artifact_dag_plan(value)
    return value


def _require_exact_typed_value(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected):
        raise Phase6ArtifactDagError(f"{label} JSON type differs")
    if isinstance(expected, dict):
        if set(observed) != set(expected):
            raise Phase6ArtifactDagError(f"{label} keys differ")
        for key, value in expected.items():
            _require_exact_typed_value(observed[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if len(observed) != len(expected):
            raise Phase6ArtifactDagError(f"{label} length differs")
        for index, (item, value) in enumerate(zip(observed, expected, strict=True)):
            _require_exact_typed_value(item, value, f"{label}[{index}]")
    elif observed != expected:
        raise Phase6ArtifactDagError(f"{label} differs")


def validate_artifact_dag_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "closed_loop_allowed",
        "formal_training_allowed",
        "nodes",
        "operational_root",
        "repository_root",
        "schema_version",
        "source_manifest_generated_by_this_cli",
    }
    plan = _exact_keys(value, expected_keys, "artifact DAG plan")
    if plan["schema_version"] != DAG_PLAN_SCHEMA_VERSION:
        raise Phase6ArtifactDagError("artifact DAG plan schema differs")
    if plan["repository_root"] != REPOSITORY_ROOT:
        raise Phase6ArtifactDagError("artifact DAG repository differs")
    if plan["operational_root"] != OPERATIONAL_ROOT:
        raise Phase6ArtifactDagError("artifact DAG operational root differs")
    for key in (
        "closed_loop_allowed",
        "formal_training_allowed",
        "source_manifest_generated_by_this_cli",
    ):
        if plan[key] is not False:
            raise Phase6ArtifactDagError(f"artifact DAG safety flag {key} differs")
    nodes = plan["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise Phase6ArtifactDagError("artifact DAG nodes must be a non-empty list")
    expected_nodes = _artifact_dag_nodes(paths=FROZEN_PATHS)
    if len(nodes) != len(expected_nodes):
        raise Phase6ArtifactDagError("artifact DAG node count differs")
    seen: set[str] = set()
    node_order: list[str] = []
    outputs: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise Phase6ArtifactDagError(f"artifact DAG node {index} is not a mapping")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id or node_id in seen:
            raise Phase6ArtifactDagError(f"artifact DAG node id differs at {index}")
        dependencies = node.get("depends_on")
        if (
            not isinstance(dependencies, list)
            or any(dep not in seen for dep in dependencies)
            or len(dependencies) != len(set(dependencies))
        ):
            raise Phase6ArtifactDagError(
                f"artifact DAG node {node_id} is cyclic or references a future node"
            )
        if node.get("training") is not False or node.get("gpu") is not False:
            raise Phase6ArtifactDagError(f"artifact DAG node {node_id} safety differs")
        expected_rebind = {
            "reference_numeric_cpu_rebind": {
                "gpu_reexecution_allowed": False,
                "reference_rows_byte_identical": True,
                "spot_rows_byte_identical": True,
            },
            "smoke_numeric_cpu_rebind": {
                "gpu_reexecution_allowed": False,
                "legacy_nonbinding_payload_anchor_exact": True,
                "smoke_scientific_projection_byte_identical": True,
            },
        }
        if node_id in expected_rebind:
            if node.get("rebind_contract") != expected_rebind[node_id]:
                raise Phase6ArtifactDagError(
                    f"artifact DAG node {node_id} rebind contract differs"
                )
            if node_id == "reference_numeric_cpu_rebind":
                stage = "reference"
                roles = REFERENCE_HISTORICAL_REBIND_ROLES
                current_inputs = [
                    FROZEN_PATHS.artifacts["source_manifest"],
                    FROZEN_PATHS.artifacts["recovery_authority"],
                    FROZEN_PATHS.reference_config,
                    FROZEN_PATHS.artifacts["reference_preflight_request"],
                ]
            else:
                stage = "smoke"
                roles = SMOKE_HISTORICAL_REBIND_ROLES
                current_inputs = [
                    FROZEN_PATHS.artifacts["source_manifest"],
                    FROZEN_PATHS.artifacts["recovery_authority"],
                    FROZEN_PATHS.artifacts["reference_rebind_receipt"],
                    FROZEN_PATHS.artifacts["projection_equivalence_receipt"],
                    FROZEN_PATHS.artifacts["smoke_preflight_request"],
                    FROZEN_PATHS.artifacts["smoke_preflight_report"],
                ]
            expected_inputs = {
                "current": current_inputs,
                "historical": {role: HISTORICAL_REBIND_INPUTS[role] for role in roles},
            }
            if node.get("inputs") != expected_inputs:
                raise Phase6ArtifactDagError(
                    f"artifact DAG node {node_id} rebind inputs differ"
                )
            if node.get("command") != _rebind_command(stage, paths=FROZEN_PATHS):
                raise Phase6ArtifactDagError(
                    f"artifact DAG node {node_id} executable command differs"
                )
            if node.get("environment") != CPU_REBIND_ENVIRONMENT:
                raise Phase6ArtifactDagError(
                    f"artifact DAG node {node_id} CPU environment differs"
                )
        elif "rebind_contract" in node:
            raise Phase6ArtifactDagError(
                f"artifact DAG node {node_id} has an unexpected rebind contract"
            )
        if node_id == "training_requests":
            if (
                node.get("inputs")
                != [FROZEN_PATHS.artifacts["evidence_rebinding_receipt"]]
                or node.get("requires_evidence_rebinding_receipt_sha256") is not True
            ):
                raise Phase6ArtifactDagError(
                    "training-requests evidence-rebinding consumer differs"
                )
        node_outputs = node.get("outputs")
        if not isinstance(node_outputs, list) or not node_outputs:
            raise Phase6ArtifactDagError(f"artifact DAG node {node_id} has no outputs")
        for output in node_outputs:
            _require_allowed_output(output, paths=FROZEN_PATHS)
            if output in outputs:
                raise Phase6ArtifactDagError(f"duplicate artifact DAG output: {output}")
            outputs.add(output)
        _require_exact_typed_value(
            node, expected_nodes[index], f"artifact DAG node {node_id}"
        )
        seen.add(node_id)
        node_order.append(node_id)
    expected_node_order = (
        "recovery_authority",
        "source_manifest",
        "after_source_freeze",
        "reference_inputs",
        "reference_numeric_cpu_rebind",
        "smoke_request",
        "smoke_report",
        "smoke_numeric_cpu_rebind",
        "training_requests",
        "training_reports",
        "materialization_pins",
        "final_sink",
    )
    if tuple(node_order) != expected_node_order:
        raise Phase6ArtifactDagError("artifact DAG node order/set differs")
    canonical_json_bytes(plan)
    return deepcopy(dict(plan))


__all__ = [
    "ARM_FACTORS",
    "ARM_NAMES",
    "DAG_PLAN_SCHEMA_VERSION",
    "FROZEN_PATHS",
    "FROZEN_SHA256",
    "HISTORICAL_REBIND_INPUTS",
    "Phase6ArtifactDagError",
    "Phase6Paths",
    "build_artifact_dag_plan",
    "build_materialization_pins",
    "build_reference_precompute_config",
    "build_reference_preflight_request",
    "build_smoke_preflight_request",
    "build_training_preflight_request",
    "canonical_json_bytes",
    "materialize_after_source_freeze",
    "materialize_final_sink",
    "materialize_reference_inputs",
    "materialize_smoke_request",
    "materialize_training_pins",
    "materialize_training_requests",
    "pin_file",
    "pin_runtime_file",
    "stable_file_sha256",
    "stable_runtime_file_sha256",
    "training_authorization",
    "validate_artifact_dag_plan",
    "validate_evidence_rebinding_receipt",
    "validate_frozen_source_manifest",
    "validate_materialization_pins_bytes",
    "validate_preflight_request_bytes",
    "validate_preflight_request_shape",
    "write_canonical_json_exclusive",
    "write_preflight_request",
    "write_reference_precompute_config",
]
