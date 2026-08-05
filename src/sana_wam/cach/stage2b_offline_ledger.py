"""Offline single-process L2 ledger for synthetic CPU Stage-2B state.

This module provides immutable PREPARED intents, content-addressed objects,
and one terminal COMMITTED-or-ABORTED decision slot per transaction.  It is a
bounded tmp-directory implementation: it does not install manager state,
recover a manager, admit /DATA crash semantics, support multiple writers, or
grant training, evaluation, deployment, Stage-3, or scientific authority.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import Enum
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Any

from sana_wam.cach import stage2b_receipt_store as l0_receipts
from sana_wam.cach.stage2b_state_snapshot import (
    EncodedStage2BStateSnapshot,
    Stage2BStateSnapshotError,
    Stage2BTensorBlob,
    decode_stage2b_state_snapshot,
)
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_chunk_layout import ChunkActionLayout
from sana_wam.model.video_backbone.sana import hybrid_cache as base
from sana_wam.model.video_backbone.sana.hybrid_cache_stage2b import (
    Stage2BTemporalState,
)


_ROOT_SCHEMA = "cach.stage2b.offline_ledger_root.v1"
_INTENT_SCHEMA = "cach.stage2b.ledger_intent.v1"
_OPERATION_IDENTITY_SCHEMA = "cach.stage2b.ledger_operation_identity.v1"
_PAIRED_OPERATION_SCHEMA = "cach.stage2b.paired_commit_operation.v1"
_RESET_OPERATION_SCHEMA = "cach.stage2b.reset_operation.v1"
_LAYOUT_INVENTORY_SCHEMA = "cach.stage2b.layout_inventory.v1"
_DECISION_BODY_SCHEMA = "cach.stage2b.ledger_decision_body.v1"
_DECISION_RECORD_SCHEMA = "cach.stage2b.ledger_decision_record.v1"
_ABORT_RECEIPT_SCHEMA = "cach.stage2b.ledger_abort_receipt.v1"
_SNAPSHOT_SCHEMA = "cach.stage2b.state_snapshot.v1"
_SCOPE = "cpu_synthetic_single_process_only"
_ROOT_FILENAME = "LEDGER.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_INTENT_FILENAME_RE = re.compile(r"intent-([0-9a-f]{64})\.json\Z")
_DECISION_FILENAME_RE = re.compile(r"decision-([0-9a-f]{64})\.json\Z")
_OBJECT_FILENAME_RE = re.compile(r"object-([0-9a-f]{64})\.bin\Z")
_TEMP_FILENAME_RE = re.compile(r"\.tmp-([0-9a-f]{32})\Z")
_RENAME_NOREPLACE = 1
_SNAPSHOT_TOP_KEYS = frozenset(
    {
        "admission_scope",
        "blob_encoding",
        "cache_state",
        "cache_state_manifest_digest",
        "committed_action_history",
        "committed_action_history_digest",
        "history_summary_operator",
        "layout_identity",
        "logical_devices",
        "registry_identity",
        "schema",
        "staging_variant",
        "state_manifest",
        "state_manifest_digest",
    }
)


class Stage2BTransactionKind(str, Enum):
    """Operations sharing the one global transaction namespace."""

    PAIRED_COMMIT = "paired_commit"
    RESET = "reset"


class Stage2BAbortCode(str, Enum):
    """Registered state-preserving terminal abort reasons."""

    CALLER_CANCELLED = "CALLER_CANCELLED"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    PROCESS_EXIT = "PROCESS_EXIT"
    PUBLICATION_FAILED = "PUBLICATION_FAILED"
    STAGING_FAILED = "STAGING_FAILED"


class Stage2BFailurePoint(str, Enum):
    """Registered protocol boundaries at which an abort may be recorded."""

    AFTER_PREPARED = "AFTER_PREPARED"
    BEFORE_STAGING = "BEFORE_STAGING"
    RECEIPT_PUBLICATION = "RECEIPT_PUBLICATION"
    SNAPSHOT_PUBLICATION = "SNAPSHOT_PUBLICATION"
    STAGING = "STAGING"
    TERMINAL_PUBLICATION = "TERMINAL_PUBLICATION"


class Stage2BDecisionResult(str, Enum):
    """The only two terminal outcomes."""

    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


class Stage2BReconcileStatus(str, Enum):
    """Stable terminal lookup result after an unknown publication outcome."""

    EXACT = "exact"
    ABSENT = "absent"


class Stage2BOfflineLedgerError(RuntimeError):
    """Fail-closed ledger error with a stable reason code."""

    def __init__(self, code: str, message: str):
        if not isinstance(code, str) or not code:
            raise ValueError("ledger error code must be non-empty")
        self.code = code
        super().__init__(f"{code}: {message}")


class Stage2BOfflineLedgerConflictError(Stage2BOfflineLedgerError):
    """An immutable logical slot already contains different bytes."""


def _fail(code: str, message: str) -> None:
    raise Stage2BOfflineLedgerError(code, message)


def _conflict(code: str, message: str) -> None:
    raise Stage2BOfflineLedgerConflictError(code, message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise Stage2BOfflineLedgerError(
            "LEDGER_SCHEMA_MISMATCH",
            "value is not canonical-JSON encodable",
        ) from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("LEDGER_SCHEMA_MISMATCH", f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail("LEDGER_SCHEMA_MISMATCH", f"non-finite JSON constant: {value}")


def _parse_canonical_json(payload: bytes, name: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        _fail("LEDGER_SCHEMA_MISMATCH", f"{name} must be non-empty bytes")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except Stage2BOfflineLedgerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Stage2BOfflineLedgerError(
            "LEDGER_SCHEMA_MISMATCH",
            f"{name} is not strict UTF-8 JSON",
        ) from exc
    if not isinstance(value, dict):
        _fail("LEDGER_SCHEMA_MISMATCH", f"{name} must contain one object")
    if _canonical_json_bytes(value) != payload:
        _fail("LEDGER_SCHEMA_MISMATCH", f"{name} is not canonical JSON")
    return value


def _require_exact_keys(
    value: dict[str, Any], expected: frozenset[str], name: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        _fail(
            "LEDGER_SCHEMA_MISMATCH",
            f"{name} members differ "
            f"(missing={sorted(expected - observed)}, extra={sorted(observed - expected)})",
        )


def _require_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        _fail("LEDGER_SCHEMA_MISMATCH", f"{name} must be a non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise Stage2BOfflineLedgerError(
            "LEDGER_SCHEMA_MISMATCH", f"{name} is not valid UTF-8"
        ) from exc
    return value


def _require_uint(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        _fail("LEDGER_SCHEMA_MISMATCH", f"{name} must be a non-negative plain int")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail("LEDGER_SCHEMA_MISMATCH", f"{name} must be a lowercase SHA256")
    return value


def _require_optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, name)


def _require_variant(value: object) -> CACHStagingVariant:
    if type(value) is not str:
        _fail("LEDGER_SCHEMA_MISMATCH", "staging_variant must be a string")
    try:
        return CACHStagingVariant(value)
    except ValueError as exc:
        raise Stage2BOfflineLedgerError(
            "LEDGER_SCHEMA_MISMATCH", "staging_variant is unknown"
        ) from exc


def _require_kind(value: object) -> Stage2BTransactionKind:
    if type(value) is not str:
        _fail("LEDGER_SCHEMA_MISMATCH", "transaction_kind must be a string")
    try:
        return Stage2BTransactionKind(value)
    except ValueError as exc:
        raise Stage2BOfflineLedgerError(
            "LEDGER_SCHEMA_MISMATCH", "transaction_kind is unknown"
        ) from exc


def _require_result(value: object) -> Stage2BDecisionResult:
    if type(value) is not str:
        _fail("LEDGER_SCHEMA_MISMATCH", "result must be a string")
    try:
        return Stage2BDecisionResult(value)
    except ValueError as exc:
        raise Stage2BOfflineLedgerError(
            "LEDGER_SCHEMA_MISMATCH", "decision result is unknown"
        ) from exc


def _require_abort_code(value: object) -> Stage2BAbortCode:
    if type(value) is not str:
        _fail("LEDGER_SCHEMA_MISMATCH", "abort_code must be a string")
    try:
        return Stage2BAbortCode(value)
    except ValueError as exc:
        raise Stage2BOfflineLedgerError(
            "LEDGER_SCHEMA_MISMATCH", "abort_code is not registered"
        ) from exc


def _require_failure_point(value: object) -> Stage2BFailurePoint:
    if type(value) is not str:
        _fail("LEDGER_SCHEMA_MISMATCH", "failure_point must be a string")
    try:
        return Stage2BFailurePoint(value)
    except ValueError as exc:
        raise Stage2BOfflineLedgerError(
            "LEDGER_SCHEMA_MISMATCH", "failure_point is not registered"
        ) from exc


def _identifier_digest(identifier: str) -> str:
    return _sha256(_require_string(identifier, "transaction_id").encode("utf-8"))


def _layout_inventory_digest(layouts: tuple[ChunkActionLayout, ...]) -> str:
    if not isinstance(layouts, tuple) or not layouts or any(
        not isinstance(layout, ChunkActionLayout) for layout in layouts
    ):
        _fail("LEDGER_SCHEMA_MISMATCH", "layout inventory has wrong type")
    identities = sorted(
        {
            (layout.layout_spec_sha256, layout.layout_instance_digest)
            for layout in layouts
        }
    )
    return _sha256(
        _canonical_json_bytes(
            {
                "layouts": [
                    {
                        "layout_instance_digest": instance_digest,
                        "layout_spec_sha256": spec_sha256,
                    }
                    for spec_sha256, instance_digest in identities
                ],
                "schema": _LAYOUT_INVENTORY_SCHEMA,
            }
        )
    )


@dataclass(frozen=True)
class Stage2BLedgerIdentity:
    ledger_id: str
    staging_variant: CACHStagingVariant
    writer_fence_id: str
    layout_spec_sha256: str
    layout_instance_digest: str
    layout_inventory_digest: str
    layer_registry_digest: str
    genesis_state_manifest: str
    genesis_snapshot_manifest_sha256: str
    admission_scope: str = _SCOPE
    schema: str = _ROOT_SCHEMA

    def __post_init__(self) -> None:
        _require_string(self.ledger_id, "ledger_id")
        if not isinstance(self.staging_variant, CACHStagingVariant):
            _fail("LEDGER_SCHEMA_MISMATCH", "staging_variant has wrong type")
        _require_string(self.writer_fence_id, "writer_fence_id")
        _require_sha256(self.layout_spec_sha256, "layout_spec_sha256")
        _require_sha256(self.layout_instance_digest, "layout_instance_digest")
        _require_sha256(self.layout_inventory_digest, "layout_inventory_digest")
        _require_sha256(self.layer_registry_digest, "layer_registry_digest")
        _require_sha256(self.genesis_state_manifest, "genesis_state_manifest")
        _require_sha256(
            self.genesis_snapshot_manifest_sha256,
            "genesis_snapshot_manifest_sha256",
        )
        if self.admission_scope != _SCOPE or self.schema != _ROOT_SCHEMA:
            _fail("LEDGER_SCOPE_UNSUPPORTED", "ledger identity scope/schema differs")

    def to_manifest(self) -> dict[str, object]:
        return {
            "admission_scope": self.admission_scope,
            "ledger_id": self.ledger_id,
            "genesis_snapshot_manifest_sha256": (
                self.genesis_snapshot_manifest_sha256
            ),
            "genesis_state_manifest": self.genesis_state_manifest,
            "layer_registry_digest": self.layer_registry_digest,
            "layout_instance_digest": self.layout_instance_digest,
            "layout_inventory_digest": self.layout_inventory_digest,
            "layout_spec_sha256": self.layout_spec_sha256,
            "schema": self.schema,
            "staging_variant": self.staging_variant.value,
            "writer_fence_id": self.writer_fence_id,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_manifest())


@dataclass(frozen=True)
class Stage2BOperationIdentity:
    """Typed envelope binding an intent to its complete operation payload."""

    ledger_id: str
    transaction_id: str
    transaction_kind: Stage2BTransactionKind
    transaction_nonce: str
    staging_variant: CACHStagingVariant
    episode_id_before: str
    episode_epoch_before: int
    revision_before: int
    state_manifest_before: str
    layout_spec_sha256: str
    layout_instance_digest: str
    operation_payload_sha256: str
    schema: str = _OPERATION_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        _require_string(self.ledger_id, "ledger_id")
        _require_string(self.transaction_id, "transaction_id")
        if not isinstance(self.transaction_kind, Stage2BTransactionKind):
            _fail("LEDGER_SCHEMA_MISMATCH", "transaction_kind has wrong type")
        _require_string(self.transaction_nonce, "transaction_nonce")
        if not isinstance(self.staging_variant, CACHStagingVariant):
            _fail("LEDGER_SCHEMA_MISMATCH", "staging_variant has wrong type")
        _require_string(self.episode_id_before, "episode_id_before")
        _require_uint(self.episode_epoch_before, "episode_epoch_before")
        _require_uint(self.revision_before, "revision_before")
        _require_sha256(self.state_manifest_before, "state_manifest_before")
        _require_sha256(self.layout_spec_sha256, "layout_spec_sha256")
        _require_sha256(self.layout_instance_digest, "layout_instance_digest")
        _require_sha256(self.operation_payload_sha256, "operation_payload_sha256")
        if self.schema != _OPERATION_IDENTITY_SCHEMA:
            _fail("LEDGER_SCHEMA_MISMATCH", "operation identity schema differs")

    def to_manifest(self) -> dict[str, object]:
        return {
            "episode_epoch_before": self.episode_epoch_before,
            "episode_id_before": self.episode_id_before,
            "layout_instance_digest": self.layout_instance_digest,
            "layout_spec_sha256": self.layout_spec_sha256,
            "ledger_id": self.ledger_id,
            "operation_payload_sha256": self.operation_payload_sha256,
            "revision_before": self.revision_before,
            "schema": self.schema,
            "staging_variant": self.staging_variant.value,
            "state_manifest_before": self.state_manifest_before,
            "transaction_id": self.transaction_id,
            "transaction_kind": self.transaction_kind.value,
            "transaction_nonce": self.transaction_nonce,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_manifest())

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.canonical_bytes)


@dataclass(frozen=True)
class Stage2BLedgerIntent:
    ledger_id: str
    ledger_sequence: int
    transaction_id: str
    transaction_kind: Stage2BTransactionKind
    transaction_nonce: str
    writer_fence_id: str
    staging_variant: CACHStagingVariant
    episode_id_before: str
    episode_epoch_before: int
    revision_before: int
    state_manifest_before: str
    input_identity_sha256: str
    previous_decision_sha256: str | None
    status: str = "PREPARED"
    schema: str = _INTENT_SCHEMA

    def __post_init__(self) -> None:
        _require_string(self.ledger_id, "ledger_id")
        _require_uint(self.ledger_sequence, "ledger_sequence")
        _require_string(self.transaction_id, "transaction_id")
        if not isinstance(self.transaction_kind, Stage2BTransactionKind):
            _fail("LEDGER_SCHEMA_MISMATCH", "transaction_kind has wrong type")
        _require_string(self.transaction_nonce, "transaction_nonce")
        _require_string(self.writer_fence_id, "writer_fence_id")
        if not isinstance(self.staging_variant, CACHStagingVariant):
            _fail("LEDGER_SCHEMA_MISMATCH", "staging_variant has wrong type")
        _require_string(self.episode_id_before, "episode_id_before")
        _require_uint(self.episode_epoch_before, "episode_epoch_before")
        _require_uint(self.revision_before, "revision_before")
        _require_sha256(self.state_manifest_before, "state_manifest_before")
        _require_sha256(self.input_identity_sha256, "input_identity_sha256")
        _require_optional_sha256(
            self.previous_decision_sha256, "previous_decision_sha256"
        )
        if self.status != "PREPARED" or self.schema != _INTENT_SCHEMA:
            _fail("LEDGER_SCHEMA_MISMATCH", "intent status/schema differs")

    def to_manifest(self) -> dict[str, object]:
        return {
            "episode_epoch_before": self.episode_epoch_before,
            "episode_id_before": self.episode_id_before,
            "input_identity_sha256": self.input_identity_sha256,
            "ledger_id": self.ledger_id,
            "ledger_sequence": self.ledger_sequence,
            "previous_decision_sha256": self.previous_decision_sha256,
            "revision_before": self.revision_before,
            "schema": self.schema,
            "staging_variant": self.staging_variant.value,
            "state_manifest_before": self.state_manifest_before,
            "status": self.status,
            "transaction_id": self.transaction_id,
            "transaction_kind": self.transaction_kind.value,
            "transaction_nonce": self.transaction_nonce,
            "writer_fence_id": self.writer_fence_id,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_manifest())

    @property
    def intent_sha256(self) -> str:
        return _sha256(self.canonical_bytes)


@dataclass(frozen=True)
class Stage2BAbortReceipt:
    ledger_id: str
    transaction_id: str
    transaction_kind: Stage2BTransactionKind
    transaction_nonce: str
    intent_sha256: str
    staging_variant: CACHStagingVariant
    abort_code: Stage2BAbortCode
    failure_point: Stage2BFailurePoint
    state_manifest_before: str
    state_manifest_after: str
    result: str = "ABORTED"
    schema: str = _ABORT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_string(self.ledger_id, "ledger_id")
        _require_string(self.transaction_id, "transaction_id")
        if not isinstance(self.transaction_kind, Stage2BTransactionKind):
            _fail("LEDGER_SCHEMA_MISMATCH", "transaction_kind has wrong type")
        _require_string(self.transaction_nonce, "transaction_nonce")
        _require_sha256(self.intent_sha256, "intent_sha256")
        if not isinstance(self.staging_variant, CACHStagingVariant):
            _fail("LEDGER_SCHEMA_MISMATCH", "staging_variant has wrong type")
        if not isinstance(self.abort_code, Stage2BAbortCode):
            _fail("LEDGER_SCHEMA_MISMATCH", "abort_code has wrong type")
        if not isinstance(self.failure_point, Stage2BFailurePoint):
            _fail("LEDGER_SCHEMA_MISMATCH", "failure_point has wrong type")
        _require_sha256(self.state_manifest_before, "state_manifest_before")
        _require_sha256(self.state_manifest_after, "state_manifest_after")
        if self.state_manifest_before != self.state_manifest_after:
            _fail("LEDGER_TRANSITION_INVALID", "abort receipt advances state")
        if self.result != "ABORTED" or self.schema != _ABORT_RECEIPT_SCHEMA:
            _fail("LEDGER_SCHEMA_MISMATCH", "abort receipt result/schema differs")

    def to_manifest(self) -> dict[str, object]:
        return {
            "abort_code": self.abort_code.value,
            "failure_point": self.failure_point.value,
            "intent_sha256": self.intent_sha256,
            "ledger_id": self.ledger_id,
            "result": self.result,
            "schema": self.schema,
            "staging_variant": self.staging_variant.value,
            "state_manifest_after": self.state_manifest_after,
            "state_manifest_before": self.state_manifest_before,
            "transaction_id": self.transaction_id,
            "transaction_kind": self.transaction_kind.value,
            "transaction_nonce": self.transaction_nonce,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_manifest())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes)


@dataclass(frozen=True)
class Stage2BDecisionBody:
    ledger_id: str
    ledger_sequence: int
    transaction_id: str
    transaction_kind: Stage2BTransactionKind
    intent_sha256: str
    result: Stage2BDecisionResult
    writer_fence_id: str
    staging_variant: CACHStagingVariant
    episode_id_before: str
    episode_epoch_before: int
    revision_before: int
    episode_id_after: str
    episode_epoch_after: int
    revision_after: int
    state_manifest_before: str
    state_manifest_after: str
    snapshot_manifest_sha256: str | None
    success_or_abort_receipt_sha256: str
    previous_decision_sha256: str | None
    abort_code: Stage2BAbortCode | None
    schema: str = _DECISION_BODY_SCHEMA

    def __post_init__(self) -> None:
        _require_string(self.ledger_id, "ledger_id")
        _require_uint(self.ledger_sequence, "ledger_sequence")
        _require_string(self.transaction_id, "transaction_id")
        if not isinstance(self.transaction_kind, Stage2BTransactionKind):
            _fail("LEDGER_SCHEMA_MISMATCH", "transaction_kind has wrong type")
        _require_sha256(self.intent_sha256, "intent_sha256")
        if not isinstance(self.result, Stage2BDecisionResult):
            _fail("LEDGER_SCHEMA_MISMATCH", "result has wrong type")
        _require_string(self.writer_fence_id, "writer_fence_id")
        if not isinstance(self.staging_variant, CACHStagingVariant):
            _fail("LEDGER_SCHEMA_MISMATCH", "staging_variant has wrong type")
        _require_string(self.episode_id_before, "episode_id_before")
        _require_uint(self.episode_epoch_before, "episode_epoch_before")
        _require_uint(self.revision_before, "revision_before")
        _require_string(self.episode_id_after, "episode_id_after")
        _require_uint(self.episode_epoch_after, "episode_epoch_after")
        _require_uint(self.revision_after, "revision_after")
        _require_sha256(self.state_manifest_before, "state_manifest_before")
        _require_sha256(self.state_manifest_after, "state_manifest_after")
        _require_optional_sha256(
            self.snapshot_manifest_sha256, "snapshot_manifest_sha256"
        )
        _require_sha256(
            self.success_or_abort_receipt_sha256,
            "success_or_abort_receipt_sha256",
        )
        _require_optional_sha256(
            self.previous_decision_sha256, "previous_decision_sha256"
        )
        if self.schema != _DECISION_BODY_SCHEMA:
            _fail("LEDGER_SCHEMA_MISMATCH", "decision body schema differs")
        if self.result is Stage2BDecisionResult.ABORTED:
            if not isinstance(self.abort_code, Stage2BAbortCode):
                _fail("LEDGER_SCHEMA_MISMATCH", "abort_code has wrong type")
            if self.snapshot_manifest_sha256 is not None:
                _fail("LEDGER_TRANSITION_INVALID", "aborted decision has snapshot")
            if (
                self.episode_id_after != self.episode_id_before
                or self.episode_epoch_after != self.episode_epoch_before
                or self.revision_after != self.revision_before
                or self.state_manifest_after != self.state_manifest_before
            ):
                _fail("LEDGER_TRANSITION_INVALID", "aborted decision advances state")
        else:
            if self.abort_code is not None:
                _fail("LEDGER_TRANSITION_INVALID", "committed decision has abort code")
            if self.snapshot_manifest_sha256 is None:
                _fail("LEDGER_TRANSITION_INVALID", "committed decision lacks snapshot")
            if self.state_manifest_after == self.state_manifest_before:
                _fail("LEDGER_TRANSITION_INVALID", "committed decision does not advance")
            if self.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT:
                if (
                    self.episode_id_after != self.episode_id_before
                    or self.episode_epoch_after != self.episode_epoch_before
                    or self.revision_after != self.revision_before + 1
                ):
                    _fail(
                        "LEDGER_TRANSITION_INVALID",
                        "paired commit changes episode identity",
                    )
            elif (
                self.episode_epoch_after != self.episode_epoch_before + 1
                or self.revision_after != 0
            ):
                _fail("LEDGER_TRANSITION_INVALID", "reset epoch/revision differs")

    def to_manifest(self) -> dict[str, object]:
        return {
            "abort_code": (
                None if self.abort_code is None else self.abort_code.value
            ),
            "episode_epoch_after": self.episode_epoch_after,
            "episode_epoch_before": self.episode_epoch_before,
            "episode_id_after": self.episode_id_after,
            "episode_id_before": self.episode_id_before,
            "intent_sha256": self.intent_sha256,
            "ledger_id": self.ledger_id,
            "ledger_sequence": self.ledger_sequence,
            "previous_decision_sha256": self.previous_decision_sha256,
            "result": self.result.value,
            "revision_after": self.revision_after,
            "revision_before": self.revision_before,
            "schema": self.schema,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "staging_variant": self.staging_variant.value,
            "state_manifest_after": self.state_manifest_after,
            "state_manifest_before": self.state_manifest_before,
            "success_or_abort_receipt_sha256": self.success_or_abort_receipt_sha256,
            "transaction_id": self.transaction_id,
            "transaction_kind": self.transaction_kind.value,
            "writer_fence_id": self.writer_fence_id,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_manifest())


@dataclass(frozen=True)
class Stage2BDecisionRecord:
    decision_body: Stage2BDecisionBody
    decision_body_sha256: str
    schema: str = _DECISION_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.decision_body, Stage2BDecisionBody):
            _fail("LEDGER_SCHEMA_MISMATCH", "decision_body has wrong type")
        _require_sha256(self.decision_body_sha256, "decision_body_sha256")
        if self.decision_body_sha256 != _sha256(self.decision_body.canonical_bytes):
            _fail("LEDGER_REFERENCE_MISMATCH", "decision body digest differs")
        if self.schema != _DECISION_RECORD_SCHEMA:
            _fail("LEDGER_SCHEMA_MISMATCH", "decision record schema differs")

    @classmethod
    def from_body(cls, body: Stage2BDecisionBody) -> Stage2BDecisionRecord:
        if not isinstance(body, Stage2BDecisionBody):
            _fail("LEDGER_SCHEMA_MISMATCH", "decision body has wrong type")
        return cls(
            decision_body=body,
            decision_body_sha256=_sha256(body.canonical_bytes),
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "decision_body": self.decision_body.to_manifest(),
            "decision_body_sha256": self.decision_body_sha256,
            "schema": self.schema,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_manifest())

    @property
    def decision_sha256(self) -> str:
        """External full-record hash; deliberately absent from its preimage."""

        return _sha256(self.canonical_bytes)


@dataclass(frozen=True)
class StoredStage2BObject:
    sha256: str
    payload: bytes
    path: Path
    fingerprint: tuple[int, ...]


@dataclass(frozen=True)
class StoredStage2BIntent:
    intent: Stage2BLedgerIntent
    payload: bytes
    sha256: str
    path: Path
    fingerprint: tuple[int, ...]


@dataclass(frozen=True)
class StoredStage2BDecision:
    decision: Stage2BDecisionRecord
    payload: bytes
    sha256: str
    path: Path
    fingerprint: tuple[int, ...]


@dataclass(frozen=True)
class Stage2BPublicationReconciliation:
    status: Stage2BReconcileStatus
    stored: StoredStage2BDecision | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, Stage2BReconcileStatus):
            _fail("LEDGER_SCHEMA_MISMATCH", "reconciliation status has wrong type")
        if (self.status is Stage2BReconcileStatus.EXACT) != isinstance(
            self.stored, StoredStage2BDecision
        ):
            _fail("LEDGER_SCHEMA_MISMATCH", "reconciliation payload/status differs")


def _parse_identity(payload: bytes) -> Stage2BLedgerIdentity:
    value = _parse_canonical_json(payload, "ledger identity")
    _require_exact_keys(
        value,
        frozenset(
            {
                "admission_scope",
                "genesis_snapshot_manifest_sha256",
                "genesis_state_manifest",
                "layer_registry_digest",
                "layout_instance_digest",
                "layout_inventory_digest",
                "layout_spec_sha256",
                "ledger_id",
                "schema",
                "staging_variant",
                "writer_fence_id",
            }
        ),
        "ledger identity",
    )
    return Stage2BLedgerIdentity(
        ledger_id=_require_string(value["ledger_id"], "ledger_id"),
        staging_variant=_require_variant(value["staging_variant"]),
        writer_fence_id=_require_string(value["writer_fence_id"], "writer_fence_id"),
        layout_spec_sha256=_require_sha256(
            value["layout_spec_sha256"], "layout_spec_sha256"
        ),
        layout_instance_digest=_require_sha256(
            value["layout_instance_digest"], "layout_instance_digest"
        ),
        layout_inventory_digest=_require_sha256(
            value["layout_inventory_digest"], "layout_inventory_digest"
        ),
        layer_registry_digest=_require_sha256(
            value["layer_registry_digest"], "layer_registry_digest"
        ),
        genesis_state_manifest=_require_sha256(
            value["genesis_state_manifest"], "genesis_state_manifest"
        ),
        genesis_snapshot_manifest_sha256=_require_sha256(
            value["genesis_snapshot_manifest_sha256"],
            "genesis_snapshot_manifest_sha256",
        ),
        admission_scope=_require_string(value["admission_scope"], "admission_scope"),
        schema=_require_string(value["schema"], "schema"),
    )


def _parse_operation_identity(payload: bytes) -> Stage2BOperationIdentity:
    value = _parse_canonical_json(payload, "operation identity")
    _require_exact_keys(
        value,
        frozenset(
            {
                "episode_epoch_before",
                "episode_id_before",
                "layout_instance_digest",
                "layout_spec_sha256",
                "ledger_id",
                "operation_payload_sha256",
                "revision_before",
                "schema",
                "staging_variant",
                "state_manifest_before",
                "transaction_id",
                "transaction_kind",
                "transaction_nonce",
            }
        ),
        "operation identity",
    )
    return Stage2BOperationIdentity(
        ledger_id=_require_string(value["ledger_id"], "ledger_id"),
        transaction_id=_require_string(value["transaction_id"], "transaction_id"),
        transaction_kind=_require_kind(value["transaction_kind"]),
        transaction_nonce=_require_string(
            value["transaction_nonce"], "transaction_nonce"
        ),
        staging_variant=_require_variant(value["staging_variant"]),
        episode_id_before=_require_string(
            value["episode_id_before"], "episode_id_before"
        ),
        episode_epoch_before=_require_uint(
            value["episode_epoch_before"], "episode_epoch_before"
        ),
        revision_before=_require_uint(value["revision_before"], "revision_before"),
        state_manifest_before=_require_sha256(
            value["state_manifest_before"], "state_manifest_before"
        ),
        layout_spec_sha256=_require_sha256(
            value["layout_spec_sha256"], "layout_spec_sha256"
        ),
        layout_instance_digest=_require_sha256(
            value["layout_instance_digest"], "layout_instance_digest"
        ),
        operation_payload_sha256=_require_sha256(
            value["operation_payload_sha256"], "operation_payload_sha256"
        ),
        schema=_require_string(value["schema"], "schema"),
    )


def _parse_operation_payload(
    payload: bytes,
    transaction_kind: Stage2BTransactionKind,
) -> dict[str, Any]:
    value = _parse_canonical_json(payload, "operation payload")
    common = {
        "episode_epoch_before",
        "episode_id_before",
        "layout_instance_digest",
        "layout_spec_sha256",
        "revision_before",
        "schema",
        "transaction_id",
        "transaction_kind",
        "transaction_nonce",
    }
    if transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT:
        _require_exact_keys(
            value,
            frozenset(
                common
                | {
                    "committed_action_span_digest",
                    "conditioning_digest",
                    "content_time_sha256",
                    "dataset_manifest_sha256",
                    "dataset_row_identity",
                    "pair_action_digest",
                    "pair_action_mask_digest",
                    "pair_frame_valid_mask_digest",
                    "pair_video_digest",
                    "source_proof_digest",
                    "teacher_pair_payload_digest",
                }
            ),
            "paired-commit operation payload",
        )
        if value["schema"] != _PAIRED_OPERATION_SCHEMA:
            _fail("LEDGER_SCHEMA_MISMATCH", "paired operation schema differs")
        for name in (
            "committed_action_span_digest",
            "content_time_sha256",
            "dataset_manifest_sha256",
            "layout_instance_digest",
            "layout_spec_sha256",
            "pair_action_digest",
            "pair_action_mask_digest",
            "pair_frame_valid_mask_digest",
            "pair_video_digest",
            "source_proof_digest",
            "teacher_pair_payload_digest",
        ):
            _require_sha256(value[name], name)
        conditioning_digest = value["conditioning_digest"]
        if conditioning_digest is not None:
            _require_sha256(conditioning_digest, "conditioning_digest")
        _require_string(value["dataset_row_identity"], "dataset_row_identity")
    else:
        _require_exact_keys(
            value,
            frozenset(
                common
                | {
                    "action_cursor_before",
                    "action_history_digest_before",
                    "new_episode_epoch",
                    "new_episode_id",
                    "staging_variant",
                    "state_manifest_before",
                }
            ),
            "reset operation payload",
        )
        if value["schema"] != _RESET_OPERATION_SCHEMA:
            _fail("LEDGER_SCHEMA_MISMATCH", "reset operation schema differs")
        _require_string(value["new_episode_id"], "new_episode_id")
        _require_uint(value["new_episode_epoch"], "new_episode_epoch")
        _require_uint(value["action_cursor_before"], "action_cursor_before")
        _require_optional_sha256(
            value["action_history_digest_before"],
            "action_history_digest_before",
        )
        _require_sha256(value["state_manifest_before"], "state_manifest_before")
        _require_variant(value["staging_variant"])
        _require_sha256(value["layout_instance_digest"], "layout_instance_digest")
        _require_sha256(value["layout_spec_sha256"], "layout_spec_sha256")

    if value["transaction_kind"] != transaction_kind.value:
        _fail("LEDGER_REFERENCE_MISMATCH", "operation kind differs")
    _require_string(value["transaction_id"], "transaction_id")
    _require_string(value["transaction_nonce"], "transaction_nonce")
    _require_string(value["episode_id_before"], "episode_id_before")
    _require_uint(value["episode_epoch_before"], "episode_epoch_before")
    _require_uint(value["revision_before"], "revision_before")
    return value


def _parse_intent(payload: bytes) -> Stage2BLedgerIntent:
    value = _parse_canonical_json(payload, "ledger intent")
    _require_exact_keys(
        value,
        frozenset(
            {
                "episode_epoch_before",
                "episode_id_before",
                "input_identity_sha256",
                "ledger_id",
                "ledger_sequence",
                "previous_decision_sha256",
                "revision_before",
                "schema",
                "staging_variant",
                "state_manifest_before",
                "status",
                "transaction_id",
                "transaction_kind",
                "transaction_nonce",
                "writer_fence_id",
            }
        ),
        "ledger intent",
    )
    return Stage2BLedgerIntent(
        ledger_id=_require_string(value["ledger_id"], "ledger_id"),
        ledger_sequence=_require_uint(value["ledger_sequence"], "ledger_sequence"),
        transaction_id=_require_string(value["transaction_id"], "transaction_id"),
        transaction_kind=_require_kind(value["transaction_kind"]),
        transaction_nonce=_require_string(
            value["transaction_nonce"], "transaction_nonce"
        ),
        writer_fence_id=_require_string(value["writer_fence_id"], "writer_fence_id"),
        staging_variant=_require_variant(value["staging_variant"]),
        episode_id_before=_require_string(
            value["episode_id_before"], "episode_id_before"
        ),
        episode_epoch_before=_require_uint(
            value["episode_epoch_before"], "episode_epoch_before"
        ),
        state_manifest_before=_require_sha256(
            value["state_manifest_before"], "state_manifest_before"
        ),
        input_identity_sha256=_require_sha256(
            value["input_identity_sha256"], "input_identity_sha256"
        ),
        previous_decision_sha256=_require_optional_sha256(
            value["previous_decision_sha256"], "previous_decision_sha256"
        ),
        revision_before=_require_uint(value["revision_before"], "revision_before"),
        status=_require_string(value["status"], "status"),
        schema=_require_string(value["schema"], "schema"),
    )


def _parse_abort_receipt(payload: bytes) -> Stage2BAbortReceipt:
    value = _parse_canonical_json(payload, "abort receipt")
    _require_exact_keys(
        value,
        frozenset(
            {
                "abort_code",
                "failure_point",
                "intent_sha256",
                "ledger_id",
                "result",
                "schema",
                "staging_variant",
                "state_manifest_after",
                "state_manifest_before",
                "transaction_id",
                "transaction_kind",
                "transaction_nonce",
            }
        ),
        "abort receipt",
    )
    return Stage2BAbortReceipt(
        ledger_id=_require_string(value["ledger_id"], "ledger_id"),
        transaction_id=_require_string(value["transaction_id"], "transaction_id"),
        transaction_kind=_require_kind(value["transaction_kind"]),
        transaction_nonce=_require_string(
            value["transaction_nonce"], "transaction_nonce"
        ),
        intent_sha256=_require_sha256(value["intent_sha256"], "intent_sha256"),
        staging_variant=_require_variant(value["staging_variant"]),
        abort_code=_require_abort_code(value["abort_code"]),
        failure_point=_require_failure_point(value["failure_point"]),
        state_manifest_before=_require_sha256(
            value["state_manifest_before"], "state_manifest_before"
        ),
        state_manifest_after=_require_sha256(
            value["state_manifest_after"], "state_manifest_after"
        ),
        result=_require_string(value["result"], "result"),
        schema=_require_string(value["schema"], "schema"),
    )


def _parse_decision_body(value: object) -> Stage2BDecisionBody:
    if not isinstance(value, dict):
        _fail("LEDGER_SCHEMA_MISMATCH", "decision_body must be an object")
    _require_exact_keys(
        value,
        frozenset(
            {
                "abort_code",
                "episode_epoch_after",
                "episode_epoch_before",
                "episode_id_after",
                "episode_id_before",
                "intent_sha256",
                "ledger_id",
                "ledger_sequence",
                "previous_decision_sha256",
                "result",
                "revision_after",
                "revision_before",
                "schema",
                "snapshot_manifest_sha256",
                "staging_variant",
                "state_manifest_after",
                "state_manifest_before",
                "success_or_abort_receipt_sha256",
                "transaction_id",
                "transaction_kind",
                "writer_fence_id",
            }
        ),
        "decision body",
    )
    abort_code = value["abort_code"]
    if abort_code is not None:
        abort_code = _require_abort_code(abort_code)
    return Stage2BDecisionBody(
        ledger_id=_require_string(value["ledger_id"], "ledger_id"),
        ledger_sequence=_require_uint(value["ledger_sequence"], "ledger_sequence"),
        transaction_id=_require_string(value["transaction_id"], "transaction_id"),
        transaction_kind=_require_kind(value["transaction_kind"]),
        intent_sha256=_require_sha256(value["intent_sha256"], "intent_sha256"),
        result=_require_result(value["result"]),
        writer_fence_id=_require_string(value["writer_fence_id"], "writer_fence_id"),
        staging_variant=_require_variant(value["staging_variant"]),
        episode_id_before=_require_string(
            value["episode_id_before"], "episode_id_before"
        ),
        episode_epoch_before=_require_uint(
            value["episode_epoch_before"], "episode_epoch_before"
        ),
        revision_before=_require_uint(value["revision_before"], "revision_before"),
        episode_id_after=_require_string(value["episode_id_after"], "episode_id_after"),
        episode_epoch_after=_require_uint(
            value["episode_epoch_after"], "episode_epoch_after"
        ),
        revision_after=_require_uint(value["revision_after"], "revision_after"),
        state_manifest_before=_require_sha256(
            value["state_manifest_before"], "state_manifest_before"
        ),
        state_manifest_after=_require_sha256(
            value["state_manifest_after"], "state_manifest_after"
        ),
        snapshot_manifest_sha256=_require_optional_sha256(
            value["snapshot_manifest_sha256"], "snapshot_manifest_sha256"
        ),
        success_or_abort_receipt_sha256=_require_sha256(
            value["success_or_abort_receipt_sha256"],
            "success_or_abort_receipt_sha256",
        ),
        previous_decision_sha256=_require_optional_sha256(
            value["previous_decision_sha256"], "previous_decision_sha256"
        ),
        abort_code=abort_code,
        schema=_require_string(value["schema"], "schema"),
    )


def _parse_decision(payload: bytes) -> Stage2BDecisionRecord:
    value = _parse_canonical_json(payload, "decision record")
    _require_exact_keys(
        value,
        frozenset({"decision_body", "decision_body_sha256", "schema"}),
        "decision record",
    )
    return Stage2BDecisionRecord(
        decision_body=_parse_decision_body(value["decision_body"]),
        decision_body_sha256=_require_sha256(
            value["decision_body_sha256"], "decision_body_sha256"
        ),
        schema=_require_string(value["schema"], "schema"),
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            _fail("LEDGER_IO_FAILED", "immutable publication made no progress")
        view = view[written:]


def _read_all(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _file_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_rdev,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_ctime_ns,
    )


def _require_same_inode(
    left: os.stat_result, right: os.stat_result, name: str
) -> None:
    if left.st_dev != right.st_dev or left.st_ino != right.st_ino:
        _fail("LEDGER_ROOT_UNSAFE", f"{name} inode differs")


def _require_regular_single_link(value: os.stat_result, name: str) -> None:
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        _fail("LEDGER_OBJECT_CORRUPT", f"{name} is not a single-link regular file")


def _collect_snapshot_blob_references(value: object) -> set[str]:
    references: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key == "raw_blob_sha256":
                    references.add(_require_sha256(child, "raw_blob_sha256"))
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return references


def _validate_snapshot_manifest(
    manifest: dict[str, Any], expected_variant: CACHStagingVariant
) -> None:
    _require_exact_keys(manifest, _SNAPSHOT_TOP_KEYS, "snapshot manifest")
    if (
        manifest["schema"] != _SNAPSHOT_SCHEMA
        or manifest["admission_scope"] != "cpu_synthetic_only"
        or manifest["blob_encoding"] != "contiguous_raw_bytes.v1"
        or manifest["staging_variant"] != expected_variant.value
        or manifest["logical_devices"] != ["cpu"]
    ):
        _fail("LEDGER_REFERENCE_MISMATCH", "snapshot identity differs")
    state_manifest = manifest["state_manifest"]
    if not isinstance(state_manifest, dict):
        _fail("LEDGER_REFERENCE_MISMATCH", "snapshot state manifest is not an object")
    state_digest = _require_sha256(
        manifest["state_manifest_digest"], "snapshot state_manifest_digest"
    )
    if _sha256(_canonical_json_bytes(state_manifest)) != state_digest:
        _fail("LEDGER_REFERENCE_MISMATCH", "snapshot state manifest digest differs")


def _decode_snapshot_exact(
    snapshot: EncodedStage2BStateSnapshot,
    *,
    expected_layout: ChunkActionLayout,
    expected_registry: base.LayerRegistry,
    expected_variant: CACHStagingVariant,
) -> Stage2BTemporalState:
    if not isinstance(snapshot, EncodedStage2BStateSnapshot):
        _fail("LEDGER_SCHEMA_MISMATCH", "snapshot has wrong type")
    try:
        return decode_stage2b_state_snapshot(
            snapshot,
            expected_manifest_sha256=snapshot.manifest_sha256,
            expected_layout=expected_layout,
            expected_registry=expected_registry,
            expected_staging_variant=expected_variant,
            device_map={"cpu": "cpu"},
        )
    except Stage2BStateSnapshotError as exc:
        raise Stage2BOfflineLedgerError(
            "LEDGER_SNAPSHOT_INVALID",
            f"snapshot failed full L1 validation ({exc.code})",
        ) from exc


def _require_empty_state(state: Stage2BTemporalState, name: str) -> None:
    if (
        state.revision != 0
        or state.committed_through_chunk is not None
        or state.next_chunk_id != 0
        or state.action_cursor != 0
        or state.committed_action_history is not None
        or any(layer.through is not None or layer.tensors for layer in state.layer_states)
    ):
        _fail("LEDGER_TRANSITION_INVALID", f"{name} is not an empty state")


def _history_digest(state: Stage2BTemporalState) -> str | None:
    history = state.committed_action_history
    return None if history is None else history.manifest_digest


def _rename_noreplace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Linux atomic no-replace rename used by the bounded L2A publisher."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise Stage2BOfflineLedgerError(
            "LEDGER_PLATFORM_UNSUPPORTED",
            "renameat2(RENAME_NOREPLACE) is required",
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        _fail(
            "LEDGER_PLATFORM_UNSUPPORTED",
            "filesystem lacks atomic rename no-replace support",
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


class Stage2BOfflineLedger:
    """Append-only CPU/synthetic ledger in one pre-created secure directory.

    This class serializes callers within one process. It deliberately does not
    claim a multi-process fence or admitted power-loss behavior.
    """

    def __init__(
        self,
        *,
        root: Path,
        identity: Stage2BLedgerIdentity,
        expected_layout: ChunkActionLayout,
        expected_registry: base.LayerRegistry,
        genesis_snapshot: EncodedStage2BStateSnapshot,
        additional_expected_layouts: tuple[ChunkActionLayout, ...],
        initialize: bool,
    ) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("ledger root must be an absolute pathlib.Path")
        if ".." in root.parts:
            raise ValueError("ledger root must not contain parent traversal")
        if not isinstance(identity, Stage2BLedgerIdentity):
            raise ValueError("identity must be Stage2BLedgerIdentity")
        genesis_state = _decode_snapshot_exact(
            genesis_snapshot,
            expected_layout=expected_layout,
            expected_registry=expected_registry,
            expected_variant=identity.staging_variant,
        )
        _require_empty_state(genesis_state, "genesis snapshot")
        if not isinstance(additional_expected_layouts, tuple):
            _fail(
                "LEDGER_SCHEMA_MISMATCH",
                "additional_expected_layouts must be a tuple",
            )
        layouts = (expected_layout, *additional_expected_layouts)
        layout_table: dict[str, ChunkActionLayout] = {}
        for layout in layouts:
            if (
                not isinstance(layout, ChunkActionLayout)
                or layout.synthetic_test_only is not True
                or layout.layout_spec_sha256 != expected_layout.layout_spec_sha256
            ):
                _fail(
                    "LEDGER_IDENTITY_MISMATCH",
                    "expected layout inventory differs",
                )
            prior = layout_table.setdefault(layout.layout_instance_digest, layout)
            if prior.to_payload() != layout.to_payload():
                _fail("LEDGER_IDENTITY_MISMATCH", "layout digest collision differs")
        if (
            identity.layout_spec_sha256 != expected_layout.layout_spec_sha256
            or identity.layout_instance_digest
            != expected_layout.layout_instance_digest
            or identity.layout_inventory_digest != _layout_inventory_digest(layouts)
            or identity.layer_registry_digest != expected_registry.manifest_digest
            or identity.genesis_state_manifest != genesis_state.state_manifest_digest
            or identity.genesis_snapshot_manifest_sha256
            != genesis_snapshot.manifest_sha256
        ):
            _fail("LEDGER_IDENTITY_MISMATCH", "genesis/root identity differs")
        self._root = root
        self._identity = identity
        self._expected_layout = expected_layout
        self._expected_layouts = layout_table
        self._expected_registry = expected_registry
        self._genesis_state = genesis_state
        self._lock = threading.RLock()
        self._terminal_publication_capability = object()
        descriptor = self._open_root_unbound()
        try:
            descriptor_stat = os.fstat(descriptor)
            try:
                entry_stat = os.stat(root, follow_symlinks=False)
            except OSError as exc:
                raise Stage2BOfflineLedgerError(
                    "LEDGER_ROOT_UNSAFE",
                    "ledger root entry cannot be inspected",
                ) from exc
            if not stat.S_ISDIR(descriptor_stat.st_mode):
                _fail("LEDGER_ROOT_UNSAFE", "ledger root is not a directory")
            _require_same_inode(descriptor_stat, entry_stat, "ledger root")
            self._require_secure_root(descriptor_stat)
            self._root_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
            self._discard_abandoned_temporaries(descriptor)
            entries = os.listdir(descriptor)
            if initialize:
                if entries:
                    expected_objects = {
                        self.object_filename(genesis_snapshot.manifest_sha256): (
                            genesis_snapshot.manifest_bytes
                        ),
                        **{
                            self.object_filename(blob.sha256): blob.payload
                            for blob in genesis_snapshot.blobs
                        },
                    }
                    allowed = {_ROOT_FILENAME, *expected_objects}
                    if _ROOT_FILENAME not in entries or not set(entries) <= allowed:
                        _fail(
                            "LEDGER_ROOT_UNSAFE",
                            "partial initialization contains foreign entries",
                        )
                    stored_identity = self._read_from_root(
                        descriptor, _ROOT_FILENAME
                    )
                    if stored_identity.payload != identity.canonical_bytes:
                        _fail(
                            "LEDGER_IDENTITY_MISMATCH",
                            "partial initialization identity differs",
                        )
                    for filename in set(entries) - {_ROOT_FILENAME}:
                        stored_object = self._read_from_root(descriptor, filename)
                        if stored_object.payload != expected_objects[filename]:
                            _fail(
                                "LEDGER_OBJECT_CORRUPT",
                                "partial genesis object differs",
                            )
            elif _ROOT_FILENAME not in entries:
                _fail("LEDGER_IDENTITY_MISMATCH", "ledger identity file is absent")
        finally:
            os.close(descriptor)

        if initialize:
            self._publish_immutable(
                filename=_ROOT_FILENAME,
                payload=identity.canonical_bytes,
                expected_sha256=_sha256(identity.canonical_bytes),
                conflict_code="LEDGER_IDENTITY_MISMATCH",
            )
            stored_genesis = self.publish_snapshot(genesis_snapshot)
            if stored_genesis.sha256 != identity.genesis_snapshot_manifest_sha256:
                _fail("LEDGER_IDENTITY_MISMATCH", "published genesis SHA256 differs")
        else:
            stored = self._read_named(_ROOT_FILENAME)
            observed = _parse_identity(stored.payload)
            if observed.canonical_bytes != identity.canonical_bytes:
                _fail("LEDGER_IDENTITY_MISMATCH", "ledger identity differs")
        _, stored_genesis_state = self._load_snapshot(
            identity.genesis_snapshot_manifest_sha256
        )
        if (
            stored_genesis_state.state_manifest_digest
            != identity.genesis_state_manifest
            or stored_genesis_state.to_manifest() != genesis_state.to_manifest()
        ):
            _fail("LEDGER_IDENTITY_MISMATCH", "stored genesis state differs")
        self._genesis_state = stored_genesis_state
        self.scan_valid_chain()

    @classmethod
    def initialize(
        cls,
        *,
        root: Path,
        ledger_id: str,
        staging_variant: CACHStagingVariant,
        writer_fence_id: str,
        expected_layout: ChunkActionLayout,
        expected_registry: base.LayerRegistry,
        genesis_snapshot: EncodedStage2BStateSnapshot,
        additional_expected_layouts: tuple[ChunkActionLayout, ...] = (),
    ) -> Stage2BOfflineLedger:
        genesis_state = _decode_snapshot_exact(
            genesis_snapshot,
            expected_layout=expected_layout,
            expected_registry=expected_registry,
            expected_variant=staging_variant,
        )
        _require_empty_state(genesis_state, "genesis snapshot")
        return cls(
            root=root,
            identity=Stage2BLedgerIdentity(
                ledger_id=ledger_id,
                staging_variant=staging_variant,
                writer_fence_id=writer_fence_id,
                layout_spec_sha256=expected_layout.layout_spec_sha256,
                layout_instance_digest=expected_layout.layout_instance_digest,
                layout_inventory_digest=_layout_inventory_digest(
                    (expected_layout, *additional_expected_layouts)
                ),
                layer_registry_digest=expected_registry.manifest_digest,
                genesis_state_manifest=genesis_state.state_manifest_digest,
                genesis_snapshot_manifest_sha256=genesis_snapshot.manifest_sha256,
            ),
            expected_layout=expected_layout,
            expected_registry=expected_registry,
            genesis_snapshot=genesis_snapshot,
            additional_expected_layouts=additional_expected_layouts,
            initialize=True,
        )

    @classmethod
    def reopen(
        cls,
        *,
        root: Path,
        expected_ledger_id: str,
        expected_staging_variant: CACHStagingVariant,
        expected_writer_fence_id: str,
        expected_layout: ChunkActionLayout,
        expected_registry: base.LayerRegistry,
        expected_genesis_snapshot: EncodedStage2BStateSnapshot,
        additional_expected_layouts: tuple[ChunkActionLayout, ...] = (),
    ) -> Stage2BOfflineLedger:
        genesis_state = _decode_snapshot_exact(
            expected_genesis_snapshot,
            expected_layout=expected_layout,
            expected_registry=expected_registry,
            expected_variant=expected_staging_variant,
        )
        _require_empty_state(genesis_state, "expected genesis snapshot")
        return cls(
            root=root,
            identity=Stage2BLedgerIdentity(
                ledger_id=expected_ledger_id,
                staging_variant=expected_staging_variant,
                writer_fence_id=expected_writer_fence_id,
                layout_spec_sha256=expected_layout.layout_spec_sha256,
                layout_instance_digest=expected_layout.layout_instance_digest,
                layout_inventory_digest=_layout_inventory_digest(
                    (expected_layout, *additional_expected_layouts)
                ),
                layer_registry_digest=expected_registry.manifest_digest,
                genesis_state_manifest=genesis_state.state_manifest_digest,
                genesis_snapshot_manifest_sha256=(
                    expected_genesis_snapshot.manifest_sha256
                ),
            ),
            expected_layout=expected_layout,
            expected_registry=expected_registry,
            genesis_snapshot=expected_genesis_snapshot,
            additional_expected_layouts=additional_expected_layouts,
            initialize=False,
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def identity(self) -> Stage2BLedgerIdentity:
        return self._identity

    @staticmethod
    def intent_filename(transaction_id: str) -> str:
        return f"intent-{_identifier_digest(transaction_id)}.json"

    @staticmethod
    def decision_filename(transaction_id: str) -> str:
        return f"decision-{_identifier_digest(transaction_id)}.json"

    @staticmethod
    def object_filename(sha256: str) -> str:
        return f"object-{_require_sha256(sha256, 'object sha256')}.bin"

    def _open_root_unbound(self) -> int:
        if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
            _fail(
                "LEDGER_PLATFORM_UNSUPPORTED",
                "O_DIRECTORY and O_NOFOLLOW are required",
            )
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            return os.open(self._root, flags)
        except OSError as exc:
            raise Stage2BOfflineLedgerError(
                "LEDGER_ROOT_UNSAFE",
                "ledger root cannot be opened safely",
            ) from exc

    @staticmethod
    def _require_secure_root(value: os.stat_result) -> None:
        if value.st_uid != os.geteuid():
            _fail("LEDGER_ROOT_UNSAFE", "ledger root is not owned by effective UID")
        if stat.S_IMODE(value.st_mode) != 0o700:
            _fail("LEDGER_ROOT_UNSAFE", "ledger root mode must be exactly 0700")

    def _require_bound_root(self, descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        try:
            entry_stat = os.stat(self._root, follow_symlinks=False)
        except OSError as exc:
            raise Stage2BOfflineLedgerError(
                "LEDGER_ROOT_UNSAFE",
                "ledger root entry cannot be rebound",
            ) from exc
        if not stat.S_ISDIR(descriptor_stat.st_mode):
            _fail("LEDGER_ROOT_UNSAFE", "ledger root fd is not a directory")
        _require_same_inode(descriptor_stat, entry_stat, "ledger root")
        self._require_secure_root(descriptor_stat)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != self._root_identity:
            _fail("LEDGER_ROOT_UNSAFE", "ledger root was replaced")

    def _discard_abandoned_temporaries(self, directory_fd: int) -> None:
        removed = False
        for filename in os.listdir(directory_fd):
            if _TEMP_FILENAME_RE.fullmatch(filename) is None:
                continue
            before = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
            _require_regular_single_link(before, filename)
            mode = stat.S_IMODE(before.st_mode)
            if before.st_uid != os.geteuid() or mode & ~0o600:
                _fail("LEDGER_ROOT_UNSAFE", "abandoned temporary is unsafe")
            if not hasattr(os, "O_PATH"):
                _fail("LEDGER_PLATFORM_UNSUPPORTED", "O_PATH is required")
            flags = os.O_PATH | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(filename, flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(descriptor)
                _require_same_inode(before, opened, filename)
                _require_regular_single_link(opened, filename)
            finally:
                os.close(descriptor)
            entry_after = os.stat(
                filename,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if _file_fingerprint(before) != _file_fingerprint(entry_after):
                _fail("LEDGER_ROOT_UNSAFE", "abandoned temporary changed")
            os.unlink(filename, dir_fd=directory_fd)
            removed = True
        if removed:
            os.fsync(directory_fd)
        self._require_bound_root(directory_fd)

    def _open_root(self) -> int:
        descriptor = self._open_root_unbound()
        try:
            self._require_bound_root(descriptor)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _read_from_root(self, directory_fd: int, filename: str) -> StoredStage2BObject:
        before = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        _require_regular_single_link(before, filename)
        if stat.S_IMODE(before.st_mode) != 0o400:
            _fail("LEDGER_OBJECT_CORRUPT", f"{filename} mode is not 0400")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
        try:
            opened = os.fstat(descriptor)
            _require_same_inode(before, opened, filename)
            _require_regular_single_link(opened, filename)
            payload = _read_all(descriptor)
            after = os.fstat(descriptor)
            if _file_fingerprint(opened) != _file_fingerprint(after):
                _fail("LEDGER_OBJECT_CORRUPT", f"{filename} changed during read")
        finally:
            os.close(descriptor)
        entry_after = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        _require_same_inode(after, entry_after, filename)
        if _file_fingerprint(after) != _file_fingerprint(entry_after):
            _fail("LEDGER_OBJECT_CORRUPT", f"{filename} entry changed during read")
        return StoredStage2BObject(
            sha256=_sha256(payload),
            payload=payload,
            path=self._root / filename,
            fingerprint=_file_fingerprint(after),
        )

    def _read_named(self, filename: str) -> StoredStage2BObject:
        directory_fd = self._open_root()
        try:
            stored = self._read_from_root(directory_fd, filename)
            self._require_bound_root(directory_fd)
            return stored
        finally:
            os.close(directory_fd)

    def _read_optional(self, filename: str) -> StoredStage2BObject | None:
        directory_fd = self._open_root()
        try:
            try:
                stored = self._read_from_root(directory_fd, filename)
            except FileNotFoundError:
                stored = None
            self._require_bound_root(directory_fd)
            return stored
        finally:
            os.close(directory_fd)

    def _publish_immutable(
        self,
        *,
        filename: str,
        payload: bytes,
        expected_sha256: str,
        conflict_code: str,
    ) -> StoredStage2BObject:
        with self._lock:
            return self._publish_immutable_locked(
                filename=filename,
                payload=payload,
                expected_sha256=expected_sha256,
                conflict_code=conflict_code,
            )

    def _publish_immutable_locked(
        self,
        *,
        filename: str,
        payload: bytes,
        expected_sha256: str,
        conflict_code: str,
    ) -> StoredStage2BObject:
        if type(payload) is not bytes:
            _fail("LEDGER_SCHEMA_MISMATCH", "immutable payload must be bytes")
        expected = _require_sha256(expected_sha256, "expected_sha256")
        if _sha256(payload) != expected:
            _fail("LEDGER_OBJECT_HASH_MISMATCH", "immutable payload SHA256 differs")
        directory_fd = self._open_root()
        temporary_name: str | None = None
        temporary_owned = False
        renamed = False
        try:
            try:
                existing = self._read_from_root(directory_fd, filename)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if existing.payload != payload or existing.sha256 != expected:
                    _conflict(conflict_code, f"immutable slot differs: {filename}")
                self._require_bound_root(directory_fd)
                return existing

            temporary_name = f".tmp-{secrets.token_hex(16)}"
            if _TEMP_FILENAME_RE.fullmatch(temporary_name) is None:
                _fail("LEDGER_IO_FAILED", "internal temporary name is invalid")
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
            temporary_owned = True
            try:
                created = os.fstat(descriptor)
                _require_regular_single_link(created, "temporary publication")
                _write_all(descriptor, payload)
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
                writer_before = os.fstat(descriptor)
                writer_payload = _read_all(descriptor)
                writer_after = os.fstat(descriptor)
                if _file_fingerprint(writer_before) != _file_fingerprint(writer_after):
                    _fail("LEDGER_IO_FAILED", "temporary publication changed")
                if writer_payload != payload or _sha256(writer_payload) != expected:
                    _fail("LEDGER_IO_FAILED", "temporary publication read-back differs")
            finally:
                os.close(descriptor)

            try:
                _rename_noreplace(
                    directory_fd,
                    temporary_name,
                    directory_fd,
                    filename,
                )
                renamed = True
                temporary_name = None
                temporary_owned = False
            except FileExistsError:
                os.unlink(temporary_name, dir_fd=directory_fd)
                temporary_name = None
                temporary_owned = False
                os.fsync(directory_fd)
                existing = self._read_from_root(directory_fd, filename)
                if existing.payload != payload or existing.sha256 != expected:
                    _conflict(conflict_code, f"immutable slot differs: {filename}")
            os.fsync(directory_fd)
            stable = self._read_from_root(directory_fd, filename)
            if stable.payload != payload or stable.sha256 != expected:
                _fail("LEDGER_IO_FAILED", "published immutable read-back differs")
            self._require_bound_root(directory_fd)
            return stable
        except Stage2BOfflineLedgerError as exc:
            if renamed:
                raise Stage2BOfflineLedgerError(
                    "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
                    f"post-rename validation failed ({exc.code})",
                ) from exc
            raise
        except OSError as exc:
            code = (
                "LEDGER_PUBLICATION_OUTCOME_UNKNOWN"
                if renamed
                else "LEDGER_IO_FAILED"
            )
            raise Stage2BOfflineLedgerError(
                code,
                f"immutable publication failed: {type(exc).__name__}",
            ) from exc
        finally:
            if temporary_name is not None and temporary_owned:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                except OSError:
                    pass
            try:
                os.close(directory_fd)
            except OSError as exc:
                code = (
                    "LEDGER_PUBLICATION_OUTCOME_UNKNOWN"
                    if renamed
                    else "LEDGER_IO_FAILED"
                )
                raise Stage2BOfflineLedgerError(
                    code,
                    "ledger directory close failed after immutable publication",
                ) from exc

    def publish_object(
        self, *, payload: bytes, expected_sha256: str | None = None
    ) -> StoredStage2BObject:
        if type(payload) is not bytes:
            _fail("LEDGER_SCHEMA_MISMATCH", "object payload must be bytes")
        observed = _sha256(payload)
        if expected_sha256 is not None and (
            _require_sha256(expected_sha256, "expected_sha256") != observed
        ):
            _fail("LEDGER_OBJECT_HASH_MISMATCH", "object payload SHA256 differs")
        return self._publish_immutable(
            filename=self.object_filename(observed),
            payload=payload,
            expected_sha256=observed,
            conflict_code="LEDGER_OBJECT_CORRUPT",
        )

    def read_object(self, sha256: str) -> StoredStage2BObject:
        with self._lock:
            expected = _require_sha256(sha256, "object sha256")
            try:
                stored = self._read_named(self.object_filename(expected))
            except FileNotFoundError as exc:
                raise Stage2BOfflineLedgerError(
                    "LEDGER_OBJECT_MISSING", f"object is absent: {expected}"
                ) from exc
            if stored.sha256 != expected:
                _fail("LEDGER_OBJECT_CORRUPT", "object filename/content digest differs")
            return stored

    def _decode_supplied_snapshot(
        self,
        snapshot: EncodedStage2BStateSnapshot,
    ) -> tuple[dict[str, Any], Stage2BTemporalState]:
        if not isinstance(snapshot, EncodedStage2BStateSnapshot):
            _fail("LEDGER_SCHEMA_MISMATCH", "snapshot has wrong type")
        manifest = _parse_canonical_json(snapshot.manifest_bytes, "snapshot manifest")
        _validate_snapshot_manifest(manifest, self._identity.staging_variant)
        expected_layout = self._layout_for_snapshot_manifest(manifest)
        decoded = _decode_snapshot_exact(
            snapshot,
            expected_layout=expected_layout,
            expected_registry=self._expected_registry,
            expected_variant=self._identity.staging_variant,
        )
        if manifest["state_manifest_digest"] != decoded.state_manifest_digest:
            _fail("LEDGER_REFERENCE_MISMATCH", "decoded snapshot digest differs")
        return manifest, decoded

    def publish_snapshot(
        self, snapshot: EncodedStage2BStateSnapshot
    ) -> StoredStage2BObject:
        with self._lock:
            _, decoded = self._decode_supplied_snapshot(snapshot)
            for blob in snapshot.blobs:
                self.publish_object(payload=blob.payload, expected_sha256=blob.sha256)
            stored_manifest = self.publish_object(
                payload=snapshot.manifest_bytes,
                expected_sha256=snapshot.manifest_sha256,
            )
            _, stable_state = self._load_snapshot(stored_manifest.sha256)
            if stable_state.to_manifest() != decoded.to_manifest():
                _fail("LEDGER_REFERENCE_MISMATCH", "snapshot stable read differs")
            return stored_manifest

    def publish_receipt(self, payload: bytes) -> StoredStage2BObject:
        _parse_canonical_json(payload, "receipt")
        return self.publish_object(payload=payload)

    def _load_snapshot(
        self, sha256: str
    ) -> tuple[dict[str, Any], Stage2BTemporalState]:
        stored = self.read_object(sha256)
        manifest = _parse_canonical_json(stored.payload, "stored snapshot manifest")
        _validate_snapshot_manifest(manifest, self._identity.staging_variant)
        expected_layout = self._layout_for_snapshot_manifest(manifest)
        blobs = tuple(
            Stage2BTensorBlob(sha256=digest, payload=self.read_object(digest).payload)
            for digest in sorted(_collect_snapshot_blob_references(manifest))
        )
        snapshot = EncodedStage2BStateSnapshot(
            manifest_bytes=stored.payload,
            blobs=blobs,
        )
        state = _decode_snapshot_exact(
            snapshot,
            expected_layout=expected_layout,
            expected_registry=self._expected_registry,
            expected_variant=self._identity.staging_variant,
        )
        return manifest, state

    def _layout_for_snapshot_manifest(
        self,
        manifest: dict[str, Any],
    ) -> ChunkActionLayout:
        layout_identity = manifest.get("layout_identity")
        if not isinstance(layout_identity, dict):
            _fail("LEDGER_REFERENCE_MISMATCH", "snapshot layout identity is absent")
        layout_digest = _require_sha256(
            layout_identity.get("layout_instance_digest"),
            "snapshot layout_instance_digest",
        )
        layout_spec = _require_sha256(
            layout_identity.get("layout_spec_sha256"),
            "snapshot layout_spec_sha256",
        )
        try:
            layout = self._expected_layouts[layout_digest]
        except KeyError as exc:
            raise Stage2BOfflineLedgerError(
                "LEDGER_IDENTITY_MISMATCH",
                "snapshot layout is absent from the caller-pinned inventory",
            ) from exc
        if (
            layout_spec != self._identity.layout_spec_sha256
            or layout.layout_spec_sha256 != layout_spec
        ):
            _fail("LEDGER_IDENTITY_MISMATCH", "snapshot layout spec differs")
        return layout

    def _snapshot_manifest(self, sha256: str) -> dict[str, Any]:
        manifest, _ = self._load_snapshot(sha256)
        return manifest

    def _validate_operation_identity(
        self,
        identity: Stage2BOperationIdentity,
        operation_payload: bytes,
    ) -> dict[str, Any]:
        manifest = _parse_operation_payload(
            operation_payload,
            identity.transaction_kind,
        )
        if identity.layout_instance_digest not in self._expected_layouts:
            _fail(
                "LEDGER_IDENTITY_MISMATCH",
                "operation layout is absent from the caller-pinned inventory",
            )
        if (
            identity.ledger_id != self._identity.ledger_id
            or identity.staging_variant is not self._identity.staging_variant
            or identity.layout_spec_sha256 != self._identity.layout_spec_sha256
            or identity.operation_payload_sha256 != _sha256(operation_payload)
            or manifest["transaction_id"] != identity.transaction_id
            or manifest["transaction_nonce"] != identity.transaction_nonce
            or manifest["transaction_kind"] != identity.transaction_kind.value
            or manifest["episode_id_before"] != identity.episode_id_before
            or manifest["episode_epoch_before"] != identity.episode_epoch_before
            or manifest["revision_before"] != identity.revision_before
            or manifest["layout_spec_sha256"] != identity.layout_spec_sha256
            or manifest["layout_instance_digest"]
            != identity.layout_instance_digest
        ):
            _fail("LEDGER_REFERENCE_MISMATCH", "operation identity/payload differs")
        if identity.transaction_kind is Stage2BTransactionKind.RESET and (
            manifest["new_episode_epoch"] != identity.episode_epoch_before + 1
            or manifest["state_manifest_before"]
            != identity.state_manifest_before
            or manifest["staging_variant"] != identity.staging_variant.value
        ):
            _fail("LEDGER_TRANSITION_INVALID", "reset operation epoch differs")
        return manifest

    def publish_operation_identity(
        self,
        *,
        identity: Stage2BOperationIdentity,
        operation_payload: bytes,
    ) -> StoredStage2BObject:
        if not isinstance(identity, Stage2BOperationIdentity):
            _fail("LEDGER_SCHEMA_MISMATCH", "operation identity has wrong type")
        if type(operation_payload) is not bytes:
            _fail("LEDGER_SCHEMA_MISMATCH", "operation payload must be bytes")
        with self._lock:
            self._validate_operation_identity(identity, operation_payload)
            self.publish_object(
                payload=operation_payload,
                expected_sha256=identity.operation_payload_sha256,
            )
            return self.publish_object(
                payload=identity.canonical_bytes,
                expected_sha256=identity.identity_sha256,
            )

    def _validate_intent_input(
        self,
        intent: Stage2BLedgerIntent,
    ) -> tuple[Stage2BOperationIdentity, dict[str, Any]]:
        identity_object = self.read_object(intent.input_identity_sha256)
        operation_identity = _parse_operation_identity(identity_object.payload)
        if operation_identity.identity_sha256 != intent.input_identity_sha256:
            _fail("LEDGER_REFERENCE_MISMATCH", "intent identity hash differs")
        operation_object = self.read_object(
            operation_identity.operation_payload_sha256
        )
        operation_manifest = self._validate_operation_identity(
            operation_identity,
            operation_object.payload,
        )
        if (
            operation_identity.ledger_id != intent.ledger_id
            or operation_identity.transaction_id != intent.transaction_id
            or operation_identity.transaction_kind is not intent.transaction_kind
            or operation_identity.transaction_nonce != intent.transaction_nonce
            or operation_identity.staging_variant is not intent.staging_variant
            or operation_identity.episode_id_before != intent.episode_id_before
            or operation_identity.episode_epoch_before != intent.episode_epoch_before
            or operation_identity.revision_before != intent.revision_before
            or operation_identity.state_manifest_before
            != intent.state_manifest_before
        ):
            _fail("LEDGER_REFERENCE_MISMATCH", "intent/operation identity differs")
        return operation_identity, operation_manifest

    def _validate_intent_state_binding(
        self,
        intent: Stage2BLedgerIntent,
        state_before: Stage2BTemporalState,
    ) -> tuple[Stage2BOperationIdentity, dict[str, Any]]:
        operation_identity, operation_manifest = self._validate_intent_input(intent)
        if (
            intent.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT
            and (
                operation_identity.layout_spec_sha256
                != state_before.layout_spec_sha256
                or operation_identity.layout_instance_digest
                != state_before.layout_instance_digest
            )
        ):
            _fail(
                "LEDGER_TRANSITION_INVALID",
                "paired operation layout differs from before-state",
            )
        if intent.transaction_kind is Stage2BTransactionKind.RESET and (
            operation_manifest["action_cursor_before"] != state_before.action_cursor
            or operation_manifest["action_history_digest_before"]
            != _history_digest(state_before)
        ):
            _fail(
                "LEDGER_TRANSITION_INVALID",
                "reset operation before-state details differ",
            )
        return operation_identity, operation_manifest

    def _stored_intent(
        self, stored: StoredStage2BObject, filename_digest: str
    ) -> StoredStage2BIntent:
        intent = _parse_intent(stored.payload)
        if _identifier_digest(intent.transaction_id) != filename_digest:
            _fail("LEDGER_OBJECT_CORRUPT", "intent filename/ID differs")
        return StoredStage2BIntent(
            intent=intent,
            payload=stored.payload,
            sha256=stored.sha256,
            path=stored.path,
            fingerprint=stored.fingerprint,
        )

    def _stored_decision(
        self, stored: StoredStage2BObject, filename_digest: str
    ) -> StoredStage2BDecision:
        decision = _parse_decision(stored.payload)
        if _identifier_digest(decision.decision_body.transaction_id) != filename_digest:
            _fail("LEDGER_OBJECT_CORRUPT", "decision filename/ID differs")
        return StoredStage2BDecision(
            decision=decision,
            payload=stored.payload,
            sha256=stored.sha256,
            path=stored.path,
            fingerprint=stored.fingerprint,
        )

    def _scan_records(
        self,
    ) -> tuple[
        dict[str, StoredStage2BIntent],
        dict[str, StoredStage2BDecision],
        tuple[StoredStage2BDecision, ...],
        tuple[StoredStage2BIntent, ...],
    ]:
        directory_fd = self._open_root()
        try:
            directory_before = os.fstat(directory_fd)
            entries = sorted(os.listdir(directory_fd))
            intents: dict[str, StoredStage2BIntent] = {}
            decisions: dict[str, StoredStage2BDecision] = {}
            identity_seen = False
            for filename in entries:
                if filename == _ROOT_FILENAME:
                    if identity_seen:
                        _fail("LEDGER_ROOT_UNSAFE", "duplicate identity entry")
                    identity_seen = True
                    stored = self._read_from_root(directory_fd, filename)
                    if _parse_identity(stored.payload).canonical_bytes != (
                        self._identity.canonical_bytes
                    ):
                        _fail("LEDGER_IDENTITY_MISMATCH", "ledger identity differs")
                    continue
                intent_match = _INTENT_FILENAME_RE.fullmatch(filename)
                if intent_match is not None:
                    stored_intent = self._stored_intent(
                        self._read_from_root(directory_fd, filename),
                        intent_match.group(1),
                    )
                    transaction_id = stored_intent.intent.transaction_id
                    if transaction_id in intents:
                        _fail("LEDGER_OBJECT_CORRUPT", "duplicate logical intent")
                    intents[transaction_id] = stored_intent
                    continue
                decision_match = _DECISION_FILENAME_RE.fullmatch(filename)
                if decision_match is not None:
                    stored_decision = self._stored_decision(
                        self._read_from_root(directory_fd, filename),
                        decision_match.group(1),
                    )
                    transaction_id = stored_decision.decision.decision_body.transaction_id
                    if transaction_id in decisions:
                        _fail("LEDGER_OBJECT_CORRUPT", "duplicate logical decision")
                    decisions[transaction_id] = stored_decision
                    continue
                object_match = _OBJECT_FILENAME_RE.fullmatch(filename)
                if object_match is not None:
                    stored_object = self._read_from_root(directory_fd, filename)
                    if stored_object.sha256 != object_match.group(1):
                        _fail(
                            "LEDGER_OBJECT_CORRUPT",
                            "content object filename/digest differs",
                        )
                    continue
                _fail("LEDGER_ROOT_UNSAFE", f"unexpected ledger entry: {filename}")
            if not identity_seen:
                _fail("LEDGER_IDENTITY_MISMATCH", "ledger identity is absent")
            entries_after = sorted(os.listdir(directory_fd))
            directory_after = os.fstat(directory_fd)
            if (
                entries_after != entries
                or _directory_fingerprint(directory_before)
                != _directory_fingerprint(directory_after)
            ):
                _fail("LEDGER_ROOT_UNSAFE", "ledger root changed during scan")
            self._require_bound_root(directory_fd)
        finally:
            os.close(directory_fd)

        _, stored_genesis = self._load_snapshot(
            self._identity.genesis_snapshot_manifest_sha256
        )
        _require_empty_state(stored_genesis, "stored genesis snapshot")
        if (
            stored_genesis.state_manifest_digest
            != self._identity.genesis_state_manifest
            or stored_genesis.to_manifest() != self._genesis_state.to_manifest()
        ):
            _fail("LEDGER_IDENTITY_MISMATCH", "stored genesis state differs")

        for stored_intent in intents.values():
            intent = stored_intent.intent
            if (
                intent.ledger_id != self._identity.ledger_id
                or intent.staging_variant is not self._identity.staging_variant
                or intent.writer_fence_id != self._identity.writer_fence_id
            ):
                _fail("LEDGER_IDENTITY_MISMATCH", "intent identity differs")
            self._validate_intent_input(intent)

        ordered = tuple(
            sorted(
                decisions.values(),
                key=lambda item: item.decision.decision_body.ledger_sequence,
            )
        )
        previous: StoredStage2BDecision | None = None
        current_state = self._genesis_state
        seen_transaction_ids: set[str] = set()
        seen_sequences: set[int] = set()
        for expected_sequence, stored_decision in enumerate(ordered):
            decision = stored_decision.decision
            body = decision.decision_body
            if body.ledger_sequence in seen_sequences:
                _fail("LEDGER_DECISION_CONFLICT", "duplicate decision sequence")
            seen_sequences.add(body.ledger_sequence)
            if body.transaction_id in seen_transaction_ids:
                _fail("LEDGER_DECISION_CONFLICT", "duplicate decision transaction ID")
            seen_transaction_ids.add(body.transaction_id)
            if body.ledger_sequence != expected_sequence:
                _fail("LEDGER_DECISION_CONFLICT", "decision chain has a gap")
            expected_previous = None if previous is None else previous.sha256
            if body.previous_decision_sha256 != expected_previous:
                _fail("LEDGER_DECISION_CONFLICT", "decision predecessor differs")
            if (
                body.ledger_id != self._identity.ledger_id
                or body.staging_variant is not self._identity.staging_variant
                or body.writer_fence_id != self._identity.writer_fence_id
            ):
                _fail("LEDGER_IDENTITY_MISMATCH", "decision identity differs")
            intent_entry = intents.get(body.transaction_id)
            if intent_entry is None:
                _fail("LEDGER_INTENT_MISSING", "decision lacks PREPARED intent")
            if (
                body.state_manifest_before != current_state.state_manifest_digest
                or body.episode_id_before != current_state.episode_id
                or body.episode_epoch_before != current_state.episode_epoch
                or body.revision_before != current_state.revision
            ):
                _fail(
                    "LEDGER_TRANSITION_INVALID",
                    "decision before-state differs from live chain state",
                )
            current_state = self._validate_decision(
                intent_entry.intent,
                decision,
                state_before=current_state,
                allow_terminal_publication_abort=True,
            )
            previous = stored_decision

        pending = tuple(
            sorted(
                (entry for key, entry in intents.items() if key not in decisions),
                key=lambda item: item.intent.transaction_id,
            )
        )
        if len(pending) > 1:
            _fail("LEDGER_INTENT_CONFLICT", "multiple PREPARED intents are pending")
        if pending:
            intent = pending[0].intent
            expected_previous = None if previous is None else previous.sha256
            if (
                intent.ledger_sequence != len(ordered)
                or intent.previous_decision_sha256 != expected_previous
            ):
                _fail("LEDGER_INTENT_CONFLICT", "pending intent chain position differs")
            if (
                intent.state_manifest_before != current_state.state_manifest_digest
                or intent.episode_id_before != current_state.episode_id
                or intent.episode_epoch_before != current_state.episode_epoch
                or intent.revision_before != current_state.revision
            ):
                _fail(
                    "LEDGER_TRANSITION_INVALID",
                    "pending intent does not continue live chain state",
                )
            self._validate_intent_state_binding(intent, current_state)
        return intents, decisions, ordered, pending

    def _validate_decision(
        self,
        intent: Stage2BLedgerIntent,
        decision: Stage2BDecisionRecord,
        *,
        state_before: Stage2BTemporalState,
        allow_terminal_publication_abort: bool,
    ) -> Stage2BTemporalState:
        body = decision.decision_body
        if (
            body.ledger_id != intent.ledger_id
            or body.ledger_sequence != intent.ledger_sequence
            or body.transaction_id != intent.transaction_id
            or body.transaction_kind is not intent.transaction_kind
            or body.intent_sha256 != intent.intent_sha256
            or body.writer_fence_id != intent.writer_fence_id
            or body.staging_variant is not intent.staging_variant
            or body.episode_id_before != intent.episode_id_before
            or body.episode_epoch_before != intent.episode_epoch_before
            or body.revision_before != intent.revision_before
            or body.state_manifest_before != intent.state_manifest_before
            or body.previous_decision_sha256 != intent.previous_decision_sha256
        ):
            _fail("LEDGER_REFERENCE_MISMATCH", "decision/intent fields differ")
        operation_identity, operation_manifest = self._validate_intent_state_binding(
            intent,
            state_before,
        )
        receipt = self.read_object(body.success_or_abort_receipt_sha256)
        receipt_manifest = _parse_canonical_json(receipt.payload, "terminal receipt")
        if body.result is Stage2BDecisionResult.ABORTED:
            abort_receipt = _parse_abort_receipt(receipt.payload)
            if (
                abort_receipt.ledger_id != body.ledger_id
                or abort_receipt.transaction_id != body.transaction_id
                or abort_receipt.transaction_kind is not body.transaction_kind
                or abort_receipt.transaction_nonce != intent.transaction_nonce
                or abort_receipt.intent_sha256 != body.intent_sha256
                or abort_receipt.staging_variant is not body.staging_variant
                or abort_receipt.abort_code != body.abort_code
                or abort_receipt.state_manifest_before != body.state_manifest_before
                or abort_receipt.state_manifest_after != body.state_manifest_after
            ):
                _fail("LEDGER_REFERENCE_MISMATCH", "abort receipt/decision differs")
            if (
                abort_receipt.failure_point
                is Stage2BFailurePoint.TERMINAL_PUBLICATION
                and not allow_terminal_publication_abort
            ):
                _fail(
                    "LEDGER_RECONCILIATION_REQUIRED",
                    "terminal-publication decision requires reconciled absence",
                )
            return state_before

        expected_schema = (
            "cach.stage2b.paired_commit_receipt.v1"
            if body.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT
            else "cach.stage2b.cache_reset_receipt.v1"
        )
        id_field = (
            "commit_id"
            if body.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT
            else "reset_id"
        )
        try:
            strict_receipt_manifest, strict_receipt_schema = (
                l0_receipts._validate_payload(receipt.payload, body.transaction_id)
            )
        except l0_receipts.Stage2BReceiptStoreError as exc:
            raise Stage2BOfflineLedgerError(
                "LEDGER_REFERENCE_MISMATCH",
                "success receipt failed the frozen L0 parser",
            ) from exc
        if (
            strict_receipt_schema != expected_schema
            or strict_receipt_manifest != receipt_manifest
            or receipt_manifest.get("schema") != expected_schema
            or receipt_manifest.get(id_field) != body.transaction_id
            or receipt_manifest.get("staging_variant") != body.staging_variant.value
            or receipt_manifest.get("transaction_nonce") != intent.transaction_nonce
            or receipt_manifest.get("state_manifest_before")
            != body.state_manifest_before
            or receipt_manifest.get("state_manifest_after") != body.state_manifest_after
        ):
            _fail("LEDGER_REFERENCE_MISMATCH", "success receipt/decision differs")
        assert body.snapshot_manifest_sha256 is not None
        snapshot_manifest, state_after = self._load_snapshot(
            body.snapshot_manifest_sha256
        )
        if (
            snapshot_manifest.get("state_manifest_digest") != body.state_manifest_after
            or state_after.state_manifest_digest != body.state_manifest_after
            or state_after.episode_id != body.episode_id_after
            or state_after.episode_epoch != body.episode_epoch_after
            or state_after.revision != body.revision_after
        ):
            _fail("LEDGER_REFERENCE_MISMATCH", "snapshot/decision state differs")

        if body.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT:
            paired_fields = (
                "conditioning_digest",
                "dataset_manifest_sha256",
                "dataset_row_identity",
                "pair_action_digest",
                "pair_action_mask_digest",
                "pair_frame_valid_mask_digest",
                "pair_video_digest",
                "source_proof_digest",
                "teacher_pair_payload_digest",
            )
            if (
                receipt_manifest.get("episode_id") != body.episode_id_before
                or receipt_manifest.get("episode_epoch")
                != body.episode_epoch_before
                or receipt_manifest.get("revision_before") != body.revision_before
                or receipt_manifest.get("revision_after") != body.revision_after
                or receipt_manifest.get("layout_instance_digest")
                != operation_identity.layout_instance_digest
                or state_after.layout_instance_digest
                != operation_identity.layout_instance_digest
                or receipt_manifest.get("action_cursor_before")
                != state_before.action_cursor
                or receipt_manifest.get("action_cursor_after")
                != state_after.action_cursor
                or receipt_manifest.get("action_history_digest_before")
                != _history_digest(state_before)
                or receipt_manifest.get("action_history_digest_after")
                != _history_digest(state_after)
                or any(
                    receipt_manifest.get(name) != operation_manifest[name]
                    for name in paired_fields
                )
            ):
                _fail(
                    "LEDGER_REFERENCE_MISMATCH",
                    "paired receipt/input/snapshot fields differ",
                )
            before_history = state_before.committed_action_history
            after_history = state_after.committed_action_history
            before_spans = () if before_history is None else before_history.spans
            if (
                after_history is None
                or len(after_history.spans) != len(before_spans) + 1
                or any(
                    left.to_manifest() != right.to_manifest()
                    for left, right in zip(
                        before_spans,
                        after_history.spans[:-1],
                        strict=True,
                    )
                )
            ):
                _fail(
                    "LEDGER_TRANSITION_INVALID",
                    "paired history is not an exact prefix append",
                )
            last_span = after_history.spans[-1]
            cache_after = state_after.cache_state
            if (
                last_span.commit_id != body.transaction_id
                or last_span.source_proof_digest
                != operation_manifest["source_proof_digest"]
                or last_span.actions_digest
                != operation_manifest["committed_action_span_digest"]
                or last_span.action_start != state_before.action_cursor
                or last_span.action_end_exclusive != state_after.action_cursor
                or last_span.chunk_id != receipt_manifest.get("chunk_id")
                or last_span.commit_source.value
                != base.CommitSource.TEACHER_FORCING.value
                or cache_after.committed_pair_digest
                != receipt_manifest.get("paired_payload_digest")
                or cache_after.last_source_proof_digest
                != receipt_manifest.get("source_proof_digest")
                or cache_after.last_commit_source
                is not base.CommitSource.TEACHER_FORCING
                or state_after.committed_through_chunk
                != receipt_manifest.get("chunk_id")
                or state_after.next_chunk_id
                != receipt_manifest.get("chunk_id") + 1
            ):
                _fail(
                    "LEDGER_REFERENCE_MISMATCH",
                    "paired history/cache lineage differs",
                )
            layer_manifests = snapshot_manifest["cache_state"]["layer_states"]
            content_time_digests = {
                layer["content_time_sha256"]
                for layer in layer_manifests
                if layer["content_time_sha256"] is not None
            }
            if content_time_digests != {operation_manifest["content_time_sha256"]}:
                _fail(
                    "LEDGER_REFERENCE_MISMATCH",
                    "paired content-time/snapshot fields differ",
                )
        else:
            if (
                receipt_manifest.get("previous_episode_id")
                != body.episode_id_before
                or receipt_manifest.get("previous_episode_epoch")
                != body.episode_epoch_before
                or receipt_manifest.get("new_episode_id") != body.episode_id_after
                or receipt_manifest.get("new_episode_epoch")
                != body.episode_epoch_after
                or receipt_manifest.get("layout_spec_sha256")
                != operation_identity.layout_spec_sha256
                or receipt_manifest.get("layout_instance_digest")
                != operation_identity.layout_instance_digest
                or state_after.layout_spec_sha256
                != operation_identity.layout_spec_sha256
                or state_after.layout_instance_digest
                != operation_identity.layout_instance_digest
                or receipt_manifest.get("action_cursor_before")
                != state_before.action_cursor
                or operation_manifest["action_cursor_before"]
                != state_before.action_cursor
                or receipt_manifest.get("action_cursor_after")
                != state_after.action_cursor
                or receipt_manifest.get("action_history_digest_before")
                != _history_digest(state_before)
                or operation_manifest["action_history_digest_before"]
                != _history_digest(state_before)
                or receipt_manifest.get("action_history_digest_after")
                != _history_digest(state_after)
                or receipt_manifest.get("aborted_pending_commit_id") is not None
                or operation_manifest["new_episode_id"] != body.episode_id_after
                or operation_manifest["new_episode_epoch"]
                != body.episode_epoch_after
            ):
                _fail(
                    "LEDGER_REFERENCE_MISMATCH",
                    "reset receipt/input/snapshot fields differ",
                )
            _require_empty_state(state_after, "reset snapshot")
        return state_after

    def scan_valid_chain(self) -> tuple[StoredStage2BDecision, ...]:
        """Validate all immutable records and return sequence-ordered decisions."""

        with self._lock:
            _, _, ordered, _ = self._scan_records()
            return ordered

    def publish_intent(self, intent: Stage2BLedgerIntent) -> StoredStage2BIntent:
        if not isinstance(intent, Stage2BLedgerIntent):
            _fail("LEDGER_SCHEMA_MISMATCH", "intent has wrong type")
        with self._lock:
            intents, decisions, ordered, pending = self._scan_records()
            existing = intents.get(intent.transaction_id)
            if existing is not None:
                if existing.payload != intent.canonical_bytes:
                    _conflict("LEDGER_INTENT_CONFLICT", "intent ID reuse differs")
                return existing
            if intent.transaction_id in decisions:
                _conflict("LEDGER_INTENT_CONFLICT", "terminal ID lacks exact intent")
            if pending:
                _conflict("LEDGER_INTENT_CONFLICT", "another intent is pending")
            expected_previous = None if not ordered else ordered[-1].sha256
            if (
                intent.ledger_id != self._identity.ledger_id
                or intent.staging_variant is not self._identity.staging_variant
                or intent.writer_fence_id != self._identity.writer_fence_id
            ):
                _fail("LEDGER_IDENTITY_MISMATCH", "intent identity differs")
            if (
                intent.ledger_sequence != len(ordered)
                or intent.previous_decision_sha256 != expected_previous
            ):
                _fail("LEDGER_INTENT_CONFLICT", "intent chain position differs")
            current_state = self._state_after_decisions(ordered)
            expected_state = (
                current_state.episode_id,
                current_state.episode_epoch,
                current_state.revision,
                current_state.state_manifest_digest,
            )
            if (
                intent.episode_id_before,
                intent.episode_epoch_before,
                intent.revision_before,
                intent.state_manifest_before,
            ) != expected_state:
                _fail("LEDGER_TRANSITION_INVALID", "intent predecessor differs")
            self._validate_intent_state_binding(intent, current_state)
            stored = self._publish_immutable(
                filename=self.intent_filename(intent.transaction_id),
                payload=intent.canonical_bytes,
                expected_sha256=intent.intent_sha256,
                conflict_code="LEDGER_INTENT_CONFLICT",
            )
            self._scan_records()
            return self._stored_intent(
                stored, _identifier_digest(intent.transaction_id)
            )

    prepare = publish_intent

    def read_intent(self, transaction_id: str) -> StoredStage2BIntent:
        with self._lock:
            intents, _, _, _ = self._scan_records()
            try:
                return intents[_require_string(transaction_id, "transaction_id")]
            except KeyError as exc:
                raise Stage2BOfflineLedgerError(
                    "LEDGER_INTENT_MISSING", "PREPARED intent is absent"
                ) from exc

    def publish_abort_receipt(
        self, receipt: Stage2BAbortReceipt
    ) -> StoredStage2BObject:
        if not isinstance(receipt, Stage2BAbortReceipt):
            _fail("LEDGER_SCHEMA_MISMATCH", "abort receipt has wrong type")
        if receipt.failure_point is Stage2BFailurePoint.TERMINAL_PUBLICATION:
            _fail(
                "LEDGER_RECONCILIATION_REQUIRED",
                "terminal-publication abort receipt requires reconciled absence",
            )
        return self.publish_receipt(receipt.canonical_bytes)

    def _state_after_decisions(
        self,
        ordered: tuple[StoredStage2BDecision, ...],
    ) -> Stage2BTemporalState:
        for stored in reversed(ordered):
            body = stored.decision.decision_body
            if body.result is Stage2BDecisionResult.COMMITTED:
                assert body.snapshot_manifest_sha256 is not None
                _, state = self._load_snapshot(body.snapshot_manifest_sha256)
                return state
        return self._genesis_state

    def publish_decision(
        self, decision: Stage2BDecisionRecord
    ) -> StoredStage2BDecision:
        if not isinstance(decision, Stage2BDecisionRecord):
            _fail("LEDGER_SCHEMA_MISMATCH", "decision has wrong type")
        with self._lock:
            return self._publish_decision_locked(
                decision,
                terminal_publication_capability=None,
            )

    def _publish_decision_locked(
        self,
        decision: Stage2BDecisionRecord,
        *,
        terminal_publication_capability: object | None,
    ) -> StoredStage2BDecision:
        body = decision.decision_body
        intents, decisions, ordered, pending = self._scan_records()
        existing = decisions.get(body.transaction_id)
        if existing is not None:
            return self._reconcile_exact_terminal(decision, existing)
        intent_entry = intents.get(body.transaction_id)
        if intent_entry is None:
            _fail("LEDGER_INTENT_MISSING", "terminal decision lacks intent")
        if len(pending) != 1 or pending[0].intent.transaction_id != body.transaction_id:
            _fail("LEDGER_INTENT_CONFLICT", "decision is not for sole pending intent")
        expected_previous = None if not ordered else ordered[-1].sha256
        if (
            body.ledger_sequence != len(ordered)
            or body.previous_decision_sha256 != expected_previous
        ):
            _fail("LEDGER_DECISION_CONFLICT", "decision chain position differs")
        self._validate_decision(
            intent_entry.intent,
            decision,
            state_before=self._state_after_decisions(ordered),
            allow_terminal_publication_abort=(
                terminal_publication_capability
                is self._terminal_publication_capability
            ),
        )
        stored = self._publish_immutable(
            filename=self.decision_filename(body.transaction_id),
            payload=decision.canonical_bytes,
            expected_sha256=decision.decision_sha256,
            conflict_code="LEDGER_DECISION_CONFLICT",
        )
        self._scan_records()
        return self._stored_decision(
            stored, _identifier_digest(body.transaction_id)
        )

    def read_decision(self, transaction_id: str) -> StoredStage2BDecision | None:
        with self._lock:
            _, decisions, _, _ = self._scan_records()
            return decisions.get(_require_string(transaction_id, "transaction_id"))

    def _preflight_terminal_record(
        self,
        candidate: Stage2BDecisionRecord,
    ) -> StoredStage2BDecision | None:
        existing = self.read_decision(candidate.decision_body.transaction_id)
        if existing is None:
            return None
        return self._reconcile_exact_terminal(candidate, existing)

    def _reconcile_exact_terminal(
        self,
        candidate: Stage2BDecisionRecord,
        existing: StoredStage2BDecision,
    ) -> StoredStage2BDecision:
        if existing.payload != candidate.canonical_bytes:
            _conflict(
                "LEDGER_DECISION_CONFLICT",
                "terminal result differs before referenced-object publication",
            )
        reconciliation = self.reconcile_terminal(
            transaction_id=candidate.decision_body.transaction_id,
            expected_record_sha256=candidate.decision_sha256,
        )
        if (
            reconciliation.status is not Stage2BReconcileStatus.EXACT
            or reconciliation.stored is None
        ):
            _fail(
                "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
                "existing terminal record did not reconcile exactly",
            )
        return reconciliation.stored

    def commit(
        self,
        *,
        intent: Stage2BLedgerIntent,
        snapshot: EncodedStage2BStateSnapshot,
        success_receipt_bytes: bytes,
        episode_id_after: str,
        episode_epoch_after: int,
        revision_after: int,
        state_manifest_after: str,
    ) -> StoredStage2BDecision:
        with self._lock:
            return self._commit_locked(
                intent=intent,
                snapshot=snapshot,
                success_receipt_bytes=success_receipt_bytes,
                episode_id_after=episode_id_after,
                episode_epoch_after=episode_epoch_after,
                revision_after=revision_after,
                state_manifest_after=state_manifest_after,
            )

    def _commit_locked(
        self,
        *,
        intent: Stage2BLedgerIntent,
        snapshot: EncodedStage2BStateSnapshot,
        success_receipt_bytes: bytes,
        episode_id_after: str,
        episode_epoch_after: int,
        revision_after: int,
        state_manifest_after: str,
    ) -> StoredStage2BDecision:
        stored_intent = self.publish_intent(intent)
        snapshot_manifest, _ = self._decode_supplied_snapshot(snapshot)
        expected_state_after = _require_sha256(
            state_manifest_after, "state_manifest_after"
        )
        if snapshot_manifest.get("state_manifest_digest") != expected_state_after:
            _fail("LEDGER_REFERENCE_MISMATCH", "snapshot state-after differs")
        receipt_manifest = _parse_canonical_json(
            success_receipt_bytes,
            "success receipt",
        )
        try:
            _, receipt_schema = l0_receipts._validate_payload(
                success_receipt_bytes,
                intent.transaction_id,
            )
        except l0_receipts.Stage2BReceiptStoreError as exc:
            raise Stage2BOfflineLedgerError(
                "LEDGER_REFERENCE_MISMATCH",
                "success receipt failed the frozen L0 parser",
            ) from exc
        expected_receipt_schema = (
            "cach.stage2b.paired_commit_receipt.v1"
            if intent.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT
            else "cach.stage2b.cache_reset_receipt.v1"
        )
        if (
            receipt_schema != expected_receipt_schema
            or receipt_manifest.get("schema") != expected_receipt_schema
        ):
            _fail("LEDGER_REFERENCE_MISMATCH", "success receipt schema differs")
        body = Stage2BDecisionBody(
            ledger_id=intent.ledger_id,
            ledger_sequence=intent.ledger_sequence,
            transaction_id=intent.transaction_id,
            transaction_kind=intent.transaction_kind,
            intent_sha256=stored_intent.sha256,
            result=Stage2BDecisionResult.COMMITTED,
            writer_fence_id=intent.writer_fence_id,
            staging_variant=intent.staging_variant,
            episode_id_before=intent.episode_id_before,
            episode_epoch_before=intent.episode_epoch_before,
            revision_before=intent.revision_before,
            episode_id_after=_require_string(episode_id_after, "episode_id_after"),
            episode_epoch_after=_require_uint(
                episode_epoch_after, "episode_epoch_after"
            ),
            revision_after=_require_uint(revision_after, "revision_after"),
            state_manifest_before=intent.state_manifest_before,
            state_manifest_after=expected_state_after,
            snapshot_manifest_sha256=snapshot.manifest_sha256,
            success_or_abort_receipt_sha256=_sha256(success_receipt_bytes),
            previous_decision_sha256=intent.previous_decision_sha256,
            abort_code=None,
        )
        candidate = Stage2BDecisionRecord.from_body(body)
        existing = self._preflight_terminal_record(candidate)
        if existing is not None:
            return existing
        snapshot_object = self.publish_snapshot(snapshot)
        receipt = self.publish_receipt(success_receipt_bytes)
        if (
            snapshot_object.sha256 != body.snapshot_manifest_sha256
            or receipt.sha256 != body.success_or_abort_receipt_sha256
        ):
            _fail("LEDGER_REFERENCE_MISMATCH", "published references differ")
        return self.publish_decision(candidate)

    def abort(
        self,
        *,
        intent: Stage2BLedgerIntent,
        abort_code: Stage2BAbortCode,
        failure_point: Stage2BFailurePoint,
    ) -> StoredStage2BDecision:
        if failure_point is Stage2BFailurePoint.TERMINAL_PUBLICATION:
            _fail(
                "LEDGER_RECONCILIATION_REQUIRED",
                "terminal-publication abort requires stable absence reconciliation",
            )
        with self._lock:
            return self._abort_locked(
                intent=intent,
                abort_code=abort_code,
                failure_point=failure_point,
                terminal_publication_capability=None,
            )

    def abort_after_reconciled_absence(
        self,
        *,
        intent: Stage2BLedgerIntent,
        expected_terminal_record_sha256: str,
    ) -> StoredStage2BDecision:
        """Publish one terminal abort only after the candidate stayed absent."""

        with self._lock:
            stored_intent = self.read_intent(intent.transaction_id)
            if stored_intent.payload != intent.canonical_bytes:
                _conflict("LEDGER_INTENT_CONFLICT", "reconciled intent differs")
            reconciliation = self.reconcile_terminal(
                transaction_id=intent.transaction_id,
                expected_record_sha256=expected_terminal_record_sha256,
            )
            if reconciliation.status is not Stage2BReconcileStatus.ABSENT:
                _conflict(
                    "LEDGER_DECISION_CONFLICT",
                    "candidate terminal record already exists",
                )
            return self._abort_locked(
                intent=intent,
                abort_code=Stage2BAbortCode.PUBLICATION_FAILED,
                failure_point=Stage2BFailurePoint.TERMINAL_PUBLICATION,
                terminal_publication_capability=(
                    self._terminal_publication_capability
                ),
            )

    def _abort_locked(
        self,
        *,
        intent: Stage2BLedgerIntent,
        abort_code: Stage2BAbortCode,
        failure_point: Stage2BFailurePoint,
        terminal_publication_capability: object | None,
    ) -> StoredStage2BDecision:
        stored_intent = self.publish_intent(intent)
        receipt = Stage2BAbortReceipt(
            ledger_id=intent.ledger_id,
            transaction_id=intent.transaction_id,
            transaction_kind=intent.transaction_kind,
            transaction_nonce=intent.transaction_nonce,
            intent_sha256=stored_intent.sha256,
            staging_variant=intent.staging_variant,
            abort_code=abort_code,
            failure_point=failure_point,
            state_manifest_before=intent.state_manifest_before,
            state_manifest_after=intent.state_manifest_before,
        )
        body = Stage2BDecisionBody(
            ledger_id=intent.ledger_id,
            ledger_sequence=intent.ledger_sequence,
            transaction_id=intent.transaction_id,
            transaction_kind=intent.transaction_kind,
            intent_sha256=stored_intent.sha256,
            result=Stage2BDecisionResult.ABORTED,
            writer_fence_id=intent.writer_fence_id,
            staging_variant=intent.staging_variant,
            episode_id_before=intent.episode_id_before,
            episode_epoch_before=intent.episode_epoch_before,
            revision_before=intent.revision_before,
            episode_id_after=intent.episode_id_before,
            episode_epoch_after=intent.episode_epoch_before,
            revision_after=intent.revision_before,
            state_manifest_before=intent.state_manifest_before,
            state_manifest_after=intent.state_manifest_before,
            snapshot_manifest_sha256=None,
            success_or_abort_receipt_sha256=receipt.receipt_sha256,
            previous_decision_sha256=intent.previous_decision_sha256,
            abort_code=abort_code,
        )
        candidate = Stage2BDecisionRecord.from_body(body)
        existing = self._preflight_terminal_record(candidate)
        if existing is not None:
            return existing
        stored_receipt = self.publish_receipt(receipt.canonical_bytes)
        if stored_receipt.sha256 != body.success_or_abort_receipt_sha256:
            _fail("LEDGER_REFERENCE_MISMATCH", "published abort receipt differs")
        return self._publish_decision_locked(
            candidate,
            terminal_publication_capability=terminal_publication_capability,
        )

    def reconcile_terminal(
        self,
        *,
        transaction_id: str,
        expected_record_sha256: str,
    ) -> Stage2BPublicationReconciliation:
        identifier = _require_string(transaction_id, "transaction_id")
        expected = _require_sha256(
            expected_record_sha256, "expected_record_sha256"
        )
        filename = self.decision_filename(identifier)
        with self._lock:
            intents_before, decisions_before, _, pending_before = self._scan_records()
            first = self._read_optional(filename)
            directory_fd = self._open_root()
            try:
                os.fsync(directory_fd)
                self._require_bound_root(directory_fd)
            except OSError as exc:
                raise Stage2BOfflineLedgerError(
                    "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
                    "directory fsync failed during reconciliation",
                ) from exc
            finally:
                try:
                    os.close(directory_fd)
                except OSError as exc:
                    raise Stage2BOfflineLedgerError(
                        "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
                        "directory close failed during reconciliation",
                    ) from exc
            second = self._read_optional(filename)
            intents_after, decisions_after, _, pending_after = self._scan_records()
            intent_fingerprints_before = {
                key: (value.sha256, value.fingerprint)
                for key, value in intents_before.items()
            }
            intent_fingerprints_after = {
                key: (value.sha256, value.fingerprint)
                for key, value in intents_after.items()
            }
            decision_fingerprints_before = {
                key: (value.sha256, value.fingerprint)
                for key, value in decisions_before.items()
            }
            decision_fingerprints_after = {
                key: (value.sha256, value.fingerprint)
                for key, value in decisions_after.items()
            }
            if (
                intent_fingerprints_before != intent_fingerprints_after
                or decision_fingerprints_before != decision_fingerprints_after
            ):
                _fail(
                    "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
                    "ledger record fingerprints changed during reconciliation",
                )
            if first is None and second is None:
                if identifier in decisions_before or identifier in decisions_after:
                    _fail(
                        "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
                        "terminal scan/read presence differs",
                    )
                if (
                    len(pending_before) != 1
                    or len(pending_after) != 1
                    or pending_before[0].intent.transaction_id != identifier
                    or pending_after[0].intent.transaction_id != identifier
                    or pending_before[0].sha256 != pending_after[0].sha256
                    or pending_before[0].fingerprint != pending_after[0].fingerprint
                ):
                    _fail(
                        "LEDGER_INTENT_MISSING",
                        "stable absence lacks one matching PREPARED intent",
                    )
                return Stage2BPublicationReconciliation(
                    status=Stage2BReconcileStatus.ABSENT,
                    stored=None,
                )
            if (
                first is None
                or second is None
                or first.payload != second.payload
                or first.fingerprint != second.fingerprint
            ):
                _fail(
                    "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
                    "terminal presence changed during reconciliation",
                )
            before_decision = decisions_before.get(identifier)
            after_decision = decisions_after.get(identifier)
            if (
                before_decision is None
                or after_decision is None
                or before_decision.sha256 != first.sha256
                or after_decision.sha256 != second.sha256
                or before_decision.fingerprint != first.fingerprint
                or after_decision.fingerprint != second.fingerprint
            ):
                _fail(
                    "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
                    "terminal scan/read identity differs",
                )
            if first.sha256 != expected or second.sha256 != expected:
                _conflict(
                    "LEDGER_DECISION_CONFLICT",
                    "terminal record differs from expected SHA256",
                )
            self.scan_valid_chain()
            stored = self._stored_decision(first, _identifier_digest(identifier))
            return Stage2BPublicationReconciliation(
                status=Stage2BReconcileStatus.EXACT,
                stored=stored,
            )


__all__ = [
    "Stage2BAbortCode",
    "Stage2BAbortReceipt",
    "Stage2BDecisionBody",
    "Stage2BDecisionRecord",
    "Stage2BDecisionResult",
    "Stage2BFailurePoint",
    "Stage2BLedgerIdentity",
    "Stage2BLedgerIntent",
    "Stage2BOperationIdentity",
    "Stage2BOfflineLedger",
    "Stage2BOfflineLedgerConflictError",
    "Stage2BOfflineLedgerError",
    "Stage2BPublicationReconciliation",
    "Stage2BReconcileStatus",
    "Stage2BTransactionKind",
    "StoredStage2BDecision",
    "StoredStage2BIntent",
    "StoredStage2BObject",
]
