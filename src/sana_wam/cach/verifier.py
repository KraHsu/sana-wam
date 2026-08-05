"""Framework-neutral verification helpers for the CACH Stage-1 contracts.

This module does not import a model framework, discover checkpoints, read run
roots, construct an optimizer, or turn a successful contract check into
execution authority.  Its Stage-1 artifact checker verifies internal file and
source-bundle consistency, but reports external trust-anchor verification
separately and never treats self-consistency as authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from sana_wam.cach.checkpoint_schema import (
    CACHCheckpointError,
    CheckpointSchema,
    validate_checkpoint_schema_against_inventory,
    validate_checkpoint_state,
)
from sana_wam.cach.config import (
    CACHConfigError,
    STAGE1_CONFIG_SCHEMA,
    validate_stage1_config,
)
from sana_wam.cach.initialization import (
    compare_shared_initialization,
    validate_fresh_initialization_fields,
    validate_operator_specific_delta,
)
from sana_wam.cach.model_inventory import (
    CACHInventoryError,
    InventoryEntry,
    name_set_sha256,
    validate_inventory,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_BASENAME = re.compile(
    r"checkpoint_step_(0|[1-9][0-9]*)\.safetensors\Z"
)
_FORBIDDEN_AFCC_TOKENS = (
    "actionreference",
    "actionfacingcachereference",
    "afcc",
    "losssubtraction",
    "phase6",
)
_CHECKPOINT_LOCATOR_KEYS = {
    "candidate_revision",
    "checkpoint_path",
    "checkpoint_raw_sha256",
    "checkpoint_schema_sha256",
    "endpoint_step",
}
_STAGE1_AUTHORITY_SCHEMA = "cach.stage1.authority.draft.v1"
_STAGE1_SOURCE_MANIFEST_SCHEMA = "cach.stage1.source_manifest.draft.v1"
_STAGE1_TEST_REPORT_SCHEMA = "cach.stage1.lightweight_test_report.v1"


def _compact_token(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _strict_json_bytes(payload: bytes, name: str) -> Mapping[str, Any]:
    def no_duplicate_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CACHCheckpointError(f"{name} is not strict JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CACHCheckpointError(f"{name} must contain one JSON object")
    return value


def _read_repository_file(
    repository_root: Path,
    relative_path: object,
    name: str,
) -> tuple[Path, bytes]:
    if type(relative_path) is not str or not relative_path:
        raise CACHCheckpointError(f"{name} path must be non-empty")
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise CACHCheckpointError(f"{name} path must be canonical and relative")
    candidate = repository_root.joinpath(*pure.parts)
    if candidate.is_symlink():
        raise CACHCheckpointError(f"{name} cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository_root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise CACHCheckpointError(
            f"{name} escapes or is absent from repository root"
        ) from exc
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise CACHCheckpointError(f"cannot open {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CACHCheckpointError(f"{name} must be a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return resolved, b"".join(chunks)


def _verify_sha256_pin(payload: bytes, expected: object, name: str) -> str:
    expected_digest = _lower_sha256(expected, f"{name} SHA256")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_digest:
        raise CACHCheckpointError(f"{name} SHA256 differs")
    return observed


def verify_stage1_artifact_set(
    repository_root: str | Path,
    *,
    authority_relative_path: str = (
        "docs/cach_sana_wam/stage1/CACH_STAGE1_AUTHORITY.draft.json"
    ),
    expected_authority_sha256: str | None = None,
    verify_external_bundle: bool = True,
) -> dict[str, Any]:
    """Verify the draft Stage-1 evidence set without granting execution.

    Supplying ``expected_authority_sha256`` proves only that the caller pinned
    these exact authority bytes.  Registration remains false because this
    draft has no reviewed external trust anchor or execution capability.
    """

    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise CACHCheckpointError("repository_root must be a directory")
    _, authority_bytes = _read_repository_file(
        root,
        authority_relative_path,
        "Stage-1 authority",
    )
    authority_digest = hashlib.sha256(authority_bytes).hexdigest()
    caller_pin_verified = expected_authority_sha256 is not None
    if expected_authority_sha256 is not None:
        _verify_sha256_pin(
            authority_bytes,
            expected_authority_sha256,
            "Stage-1 authority",
        )
    authority = _strict_json_bytes(authority_bytes, "Stage-1 authority")
    if authority.get("schema") != _STAGE1_AUTHORITY_SCHEMA:
        raise CACHCheckpointError("Stage-1 authority schema differs")
    if authority.get("decision") != (
        "allow_contract_implementation_and_lightweight_tests_only"
    ):
        raise CACHCheckpointError("Stage-1 authority decision differs")
    capabilities = authority.get("capabilities")
    if (
        not isinstance(capabilities, Mapping)
        or not capabilities
        or any(value is not False for value in capabilities.values())
    ):
        raise CACHCheckpointError("Stage-1 authority capability set is unsafe")
    registration = authority.get("registration")
    if (
        not isinstance(registration, Mapping)
        or registration.get("registered_execution_authority") is not False
        or registration.get("external_trust_anchor_registered") is not False
    ):
        raise CACHCheckpointError("Stage-1 draft registration state differs")

    pins = authority.get("pins")
    if not isinstance(pins, Mapping):
        raise CACHCheckpointError("Stage-1 authority pins must be a mapping")
    repository_pin_names = {
        "config",
        "development_plan",
        "lightweight_test_report",
        "readme",
        "source_manifest",
    }
    # The report appears under evidence and is normalized into the pin table
    # for one uniform verification loop.
    evidence = authority.get("evidence")
    if not isinstance(evidence, Mapping):
        raise CACHCheckpointError("Stage-1 authority evidence must be a mapping")
    report_pin = evidence.get("lightweight_test_report")
    if not isinstance(report_pin, Mapping):
        raise CACHCheckpointError("Stage-1 test-report pin is absent")
    normalized_pins = dict(pins)
    normalized_pins["lightweight_test_report"] = report_pin
    if not repository_pin_names.issubset(normalized_pins):
        raise CACHCheckpointError("Stage-1 repository pin set is incomplete")

    pinned_payloads: dict[str, bytes] = {}
    for name in sorted(repository_pin_names):
        pin = normalized_pins[name]
        if not isinstance(pin, Mapping):
            raise CACHCheckpointError(f"Stage-1 {name} pin must be a mapping")
        _, payload = _read_repository_file(
            root,
            pin.get("path"),
            f"Stage-1 {name}",
        )
        _verify_sha256_pin(payload, pin.get("sha256"), f"Stage-1 {name}")
        pinned_payloads[name] = payload

    source_manifest = _strict_json_bytes(
        pinned_payloads["source_manifest"],
        "Stage-1 source manifest",
    )
    if source_manifest.get("schema") != _STAGE1_SOURCE_MANIFEST_SCHEMA:
        raise CACHCheckpointError("Stage-1 source manifest schema differs")
    files = source_manifest.get("files")
    scope = source_manifest.get("scope")
    if (
        not isinstance(files, list)
        or not isinstance(scope, Mapping)
        or scope.get("file_count") != len(files)
    ):
        raise CACHCheckpointError("Stage-1 source manifest file count differs")
    observed_paths = []
    for index, item in enumerate(files):
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise CACHCheckpointError(
                f"Stage-1 source file entry {index} differs"
            )
        path = item["path"]
        _, payload = _read_repository_file(
            root,
            path,
            f"Stage-1 source file {index}",
        )
        _verify_sha256_pin(
            payload,
            item["sha256"],
            f"Stage-1 source file {path}",
        )
        observed_paths.append(path)
    if observed_paths != sorted(observed_paths) or len(observed_paths) != len(
        set(observed_paths)
    ):
        raise CACHCheckpointError(
            "Stage-1 source file paths must be sorted and unique"
        )

    test_report = _strict_json_bytes(
        pinned_payloads["lightweight_test_report"],
        "Stage-1 lightweight test report",
    )
    if (
        test_report.get("schema") != _STAGE1_TEST_REPORT_SCHEMA
        or test_report.get("status") != "passed"
        or not isinstance(test_report.get("gate_status"), Mapping)
        or test_report["gate_status"].get("gate_s1") != "not_claimed"
    ):
        raise CACHCheckpointError("Stage-1 lightweight test report differs")

    manifest_bundle = source_manifest.get("stage1_source_bundle")
    authority_bundle = pins.get("stage1_source_bundle")
    if (
        not isinstance(manifest_bundle, Mapping)
        or not isinstance(authority_bundle, Mapping)
        or dict(manifest_bundle) != dict(authority_bundle)
    ):
        raise CACHCheckpointError("Stage-1 source-bundle pins differ")
    bundle_verified = False
    if verify_external_bundle:
        bundle_path = authority_bundle.get("path")
        if type(bundle_path) is not str or not bundle_path.startswith("/"):
            raise CACHCheckpointError(
                "Stage-1 source bundle path must be absolute"
            )
        path = Path(bundle_path)
        if path.is_symlink():
            raise CACHCheckpointError("Stage-1 source bundle cannot be a symlink")
        try:
            metadata = path.stat()
        except OSError as exc:
            raise CACHCheckpointError(
                "Stage-1 source bundle is unavailable"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != authority_bundle.get("size_bytes")
            or f"{stat.S_IMODE(metadata.st_mode):04o}"
            != authority_bundle.get("mode")
        ):
            raise CACHCheckpointError(
                "Stage-1 source bundle metadata differs"
            )
        hasher = hashlib.sha256()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        finally:
            os.close(descriptor)
        if hasher.hexdigest() != authority_bundle.get("sha256"):
            raise CACHCheckpointError("Stage-1 source bundle SHA256 differs")
        bundle_verified = True

    return {
        "schema": "cach.stage1.artifact_verification.v1",
        "authority_sha256": authority_digest,
        "authority_internal_pins_valid": True,
        "external_source_bundle_verified": bundle_verified,
        "caller_authority_pin_verified": caller_pin_verified,
        "external_trust_anchor_verified": False,
        "registered_execution_authority": False,
        "execution_admission_valid": False,
        "model_execution_authorized": False,
        "training_authorized": False,
        "scientific_eligible": False,
        "source_file_count": len(files),
        "lightweight_tests_passed": test_report["verification"][-1]["passed"],
        "gate_s1": "not_claimed",
    }


def _raise_afcc(path: str, detail: str) -> None:
    raise CACHConfigError(
        "CACH_AFCC_ISOLATION_VIOLATION",
        f"{path} {detail}",
    )


def _validate_afcc_tree(value: Any, path: str, active: set[int]) -> None:
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise CACHConfigError(
                "CONFIG_TYPE_MISMATCH",
                f"{path} contains a recursive mapping",
            )
        active.add(marker)
        try:
            for key, child in value.items():
                if type(key) is not str:
                    raise CACHConfigError(
                        "CONFIG_TYPE_MISMATCH",
                        f"{path} keys must be strings",
                    )
                compact = _compact_token(key)
                child_path = f"{path}.{key}"
                if any(token in compact for token in _FORBIDDEN_AFCC_TOKENS):
                    _raise_afcc(child_path, "is forbidden in CACH")
                if compact == "f" and (type(child) is not int or child != 0):
                    _raise_afcc(child_path, "must be the integer 0")
                _validate_afcc_tree(child, child_path, active)
        finally:
            active.remove(marker)
        return

    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise CACHConfigError(
                "CONFIG_TYPE_MISMATCH",
                f"{path} contains a recursive sequence",
            )
        active.add(marker)
        try:
            for index, child in enumerate(value):
                _validate_afcc_tree(child, f"{path}[{index}]", active)
        finally:
            active.remove(marker)
        return

    if type(value) is str:
        compact = _compact_token(value)
        if any(token in compact for token in _FORBIDDEN_AFCC_TOKENS):
            _raise_afcc(path, "contains a forbidden AFCC/Phase-6 reference")


def validate_afcc_isolation(value: Any, *, name: str = "config") -> None:
    """Reject AFCC/Phase-6/reference contamination recursively.

    A shared legacy schema may force an ``F`` field to exist.  In that narrow
    case the only accepted value is the exact integer ``0``; booleans and
    string lookalikes are rejected.  Key spelling tricks such as ``AF-CC`` or
    ``phase_6`` are normalized before comparison.
    """

    if type(name) is not str or not name:
        raise CACHConfigError("CONFIG_TYPE_MISMATCH", "name must be non-empty")
    _validate_afcc_tree(value, name, set())


def validate_afcc_pair_isolation(reference: Any, candidate: Any) -> None:
    """Apply the same no-AFCC contract to both arms of a CACH pair."""

    validate_afcc_isolation(reference, name="reference")
    validate_afcc_isolation(candidate, name="candidate")


def _optimizer_membership(
    optimizer_groups: Mapping[str, Iterable[str]],
) -> dict[str, str]:
    if not isinstance(optimizer_groups, Mapping):
        raise CACHInventoryError("optimizer groups must be a mapping")

    membership: dict[str, str] = {}
    for group, raw_names in optimizer_groups.items():
        if type(group) is not str or not group:
            raise CACHInventoryError("optimizer group name must be non-empty")
        if isinstance(raw_names, (str, bytes)):
            raise CACHInventoryError(
                f"optimizer group {group} members must be an iterable of names"
            )
        try:
            names = tuple(raw_names)
        except TypeError as exc:
            raise CACHInventoryError(
                f"optimizer group {group} members are not iterable"
            ) from exc
        if not names:
            raise CACHInventoryError(f"optimizer group {group} is empty")
        for name in names:
            if type(name) is not str or not name:
                raise CACHInventoryError(
                    f"optimizer group {group} has a non-canonical member"
                )
            if name in membership:
                raise CACHInventoryError(
                    f"parameter entered multiple optimizer groups: {name}"
                )
            membership[name] = group
    return membership


def validate_optimizer_membership(
    entries: Iterable[InventoryEntry],
    optimizer_groups: Mapping[str, Iterable[str]],
) -> tuple[InventoryEntry, ...]:
    """Require optimizer membership to equal the trainable inventory exactly."""

    ordered = validate_inventory(entries)
    membership = _optimizer_membership(optimizer_groups)
    expected = {
        entry.fully_qualified_name
        for entry in ordered
        if entry.kind == "parameter" and entry.requires_grad
    }
    observed = set(membership)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise CACHInventoryError(
            "optimizer parameter set differs: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for entry in ordered:
        actual_group = membership.get(entry.fully_qualified_name)
        if entry.kind == "buffer" and entry.optimizer_group_or_null is not None:
            raise CACHInventoryError(
                f"buffer claims optimizer ownership: {entry.fully_qualified_name}"
            )
        if actual_group != entry.optimizer_group_or_null:
            raise CACHInventoryError(
                "optimizer group differs for "
                f"{entry.fully_qualified_name}: "
                f"inventory={entry.optimizer_group_or_null!r}, "
                f"observed={actual_group!r}"
            )
    return ordered


def validate_inventory_pair(
    reference: Iterable[InventoryEntry],
    candidate: Iterable[InventoryEntry],
    *,
    expected_candidate_operator_names: set[str],
    reference_optimizer_groups: Mapping[str, Iterable[str]],
    candidate_optimizer_groups: Mapping[str, Iterable[str]],
) -> tuple[tuple[InventoryEntry, ...], tuple[InventoryEntry, ...]]:
    """Validate optimizer closure, shared initialization, and the unique delta."""

    if (
        type(expected_candidate_operator_names) is not set
        or not all(
            type(name) is str and name
            for name in expected_candidate_operator_names
        )
    ):
        raise CACHInventoryError(
            "expected candidate operator names must be an exact string set"
        )
    reference_ordered = validate_optimizer_membership(
        reference,
        reference_optimizer_groups,
    )
    candidate_ordered = validate_optimizer_membership(
        candidate,
        candidate_optimizer_groups,
    )
    compare_shared_initialization(reference_ordered, candidate_ordered)
    validate_operator_specific_delta(
        reference_ordered,
        candidate_ordered,
        expected_candidate_names=expected_candidate_operator_names,
    )
    return reference_ordered, candidate_ordered


def _lower_sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise CACHCheckpointError(f"{name} must be one lowercase SHA256")
    return value


def validate_exact_checkpoint_locator(
    value: Any,
    schema: CheckpointSchema,
) -> Mapping[str, Any]:
    """Validate an authority-supplied exact checkpoint coordinate.

    This is a lexical contract only.  It intentionally performs no path
    discovery or file access, and it rejects ``latest`` and glob-like paths.
    """

    if not isinstance(value, Mapping):
        raise CACHCheckpointError("checkpoint locator must be a mapping")
    if not all(type(key) is str for key in value):
        raise CACHCheckpointError("checkpoint locator keys must be strings")
    if set(value) != _CHECKPOINT_LOCATOR_KEYS:
        raise CACHCheckpointError(
            "checkpoint locator key set differs: "
            f"expected={sorted(_CHECKPOINT_LOCATOR_KEYS)}, got={sorted(value)}"
        )

    revision = value["candidate_revision"]
    if revision != schema.candidate_revision:
        raise CACHCheckpointError("checkpoint candidate revision differs")
    endpoint_step = value["endpoint_step"]
    if type(endpoint_step) is not int or endpoint_step < 0:
        raise CACHCheckpointError("checkpoint endpoint step must be non-negative")
    _lower_sha256(value["checkpoint_raw_sha256"], "checkpoint raw digest")
    schema_digest = _lower_sha256(
        value["checkpoint_schema_sha256"],
        "checkpoint schema digest",
    )
    if schema_digest != schema.sha256:
        raise CACHCheckpointError("checkpoint schema digest differs")

    path = value["checkpoint_path"]
    if type(path) is not str or not path or not path.startswith("/"):
        raise CACHCheckpointError("checkpoint path must be a literal absolute path")
    if "//" in path:
        raise CACHCheckpointError("checkpoint path cannot contain double slash")
    if any(character in path for character in "*?[]{}"):
        raise CACHCheckpointError("checkpoint path cannot contain glob syntax")
    pure = PurePosixPath(path)
    if path == "/" or pure.as_posix() != path:
        raise CACHCheckpointError("checkpoint path must be canonical")
    if any(part in {".", ".."} for part in pure.parts):
        raise CACHCheckpointError("checkpoint path cannot contain dot components")
    if any("latest" in _compact_token(part) for part in pure.parts):
        raise CACHCheckpointError("checkpoint path cannot select latest")
    basename = _CHECKPOINT_BASENAME.fullmatch(pure.name)
    if basename is None:
        raise CACHCheckpointError(
            "checkpoint basename must be "
            "checkpoint_step_<endpoint_step>.safetensors"
        )
    if int(basename.group(1)) != endpoint_step:
        raise CACHCheckpointError("checkpoint path endpoint step differs")
    return value


def verify_checkpoint_contract(
    state: Mapping[str, Any],
    schema: CheckpointSchema,
    inventory: Iterable[InventoryEntry],
    *,
    locator: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify exact state/schema/inventory/locator agreement."""

    ordered = validate_inventory(inventory)
    validate_checkpoint_schema_against_inventory(schema, ordered)
    validate_exact_checkpoint_locator(locator, schema)
    validate_checkpoint_state(state, schema)
    return {
        "schema": "cach.checkpoint_verification.v1",
        "candidate_revision": schema.candidate_revision,
        "checkpoint_schema_sha256": schema.sha256,
        "checkpoint_key_set_sha256": schema.key_set_sha256,
        "checkpoint_key_count": len(schema.entries),
        "checkpoint_state_exact": True,
        "external_vae_text_state_excluded": True,
    }


def build_stage1_static_report(config: Any) -> dict[str, Any]:
    """Validate only the Stage-1 static config and return a bounded report."""

    root = validate_stage1_config(config)
    validate_afcc_isolation(root)
    architecture = root["model"]["architecture"]
    video = root["model"]["video_backbone"]
    training = root["training"]
    validate_fresh_initialization_fields(
        initialization_mode=architecture["initialization_mode"],
        model_path=video["model_path"],
        init_dit_from=video["init_dit_from"],
        init_checkpoint=training["init_checkpoint"],
        init_checkpoint_sha256=training["init_checkpoint_sha256"],
        freeze=training["freeze"],
    )
    return {
        "schema": "cach.stage1_static_report.v1",
        "config_schema": STAGE1_CONFIG_SCHEMA,
        "stage": "stage1",
        "scope": "static_config_and_pure_contract_review",
        "static_config_valid": True,
        "fresh_initialization_contract_valid": True,
        "candidate_config_afcc_isolation_valid": True,
        "pair_afcc_isolation_verified": False,
        "real_model_inventory_verified": False,
        "real_checkpoint_schema_verified": False,
        "runtime_inventory_verified": False,
        "data_admission_verified": False,
        "execution_admission_valid": False,
        "model_execution_authorized": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "scientific_eligible": False,
        "remaining_blockers": [
            "reviewed_source_and_runtime_closure",
            "real_model_exact_inventory",
            "real_model_checkpoint_schema",
            "reference_candidate_pair_afcc_isolation",
            "data_and_row_timebase_admission",
            "real_cache_codec_and_field_schema",
            "mini_model_numerical_contract",
            "executable_five_piece_review",
            "legacy_and_auxiliary_entrypoint_pre_runtime_coverage",
            "capacity_and_gpu_admission",
        ],
    }


def verify_synthetic_stage1_contract(
    config: Any,
    *,
    reference_inventory: Iterable[InventoryEntry],
    candidate_inventory: Iterable[InventoryEntry],
    expected_candidate_operator_names: set[str],
    reference_optimizer_groups: Mapping[str, Iterable[str]],
    candidate_optimizer_groups: Mapping[str, Iterable[str]],
    candidate_checkpoint_state: Mapping[str, Any],
    candidate_checkpoint_schema: CheckpointSchema,
    candidate_checkpoint_locator: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate pure synthetic Stage-1 checks without granting execution."""

    report = build_stage1_static_report(config)
    reference_ordered, candidate_ordered = validate_inventory_pair(
        reference_inventory,
        candidate_inventory,
        expected_candidate_operator_names=expected_candidate_operator_names,
        reference_optimizer_groups=reference_optimizer_groups,
        candidate_optimizer_groups=candidate_optimizer_groups,
    )
    checkpoint = verify_checkpoint_contract(
        candidate_checkpoint_state,
        candidate_checkpoint_schema,
        candidate_ordered,
        locator=candidate_checkpoint_locator,
    )
    reference_trainable = {
        entry.fully_qualified_name
        for entry in reference_ordered
        if entry.kind == "parameter" and entry.requires_grad
    }
    candidate_trainable = {
        entry.fully_qualified_name
        for entry in candidate_ordered
        if entry.kind == "parameter" and entry.requires_grad
    }
    return {
        **report,
        "schema": "cach.stage1_synthetic_contract_report.v1",
        "synthetic_inventory_pair_valid": True,
        "synthetic_checkpoint_contract_valid": True,
        "reference_trainable_name_set_sha256": name_set_sha256(
            reference_trainable
        ),
        "candidate_trainable_name_set_sha256": name_set_sha256(
            candidate_trainable
        ),
        "candidate_checkpoint": checkpoint,
        "execution_admission_valid": False,
        "scientific_eligible": False,
    }


__all__ = [
    "build_stage1_static_report",
    "validate_afcc_isolation",
    "validate_afcc_pair_isolation",
    "validate_exact_checkpoint_locator",
    "validate_inventory_pair",
    "validate_optimizer_membership",
    "verify_checkpoint_contract",
    "verify_stage1_artifact_set",
    "verify_synthetic_stage1_contract",
]
