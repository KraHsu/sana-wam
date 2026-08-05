"""Deterministic CPU/synthetic restart recovery for the Stage-2B ledger.

This additive L3 module reconstructs one detached live cache manager from a
fully validated L2A ledger.  It never executes a model or a staging callback.
The surface is deliberately non-authorizing: production filesystems,
multi-process fencing, training, evaluation, and scientific use remain out of
scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Mapping

from sana_wam.cach import stage2b_offline_ledger as offline
from sana_wam.cach import stage2b_receipt_store as l0_receipts
from sana_wam.cach.stage2b_offline_ledger import (
    Stage2BAbortCode,
    Stage2BAbortReceipt,
    Stage2BDecisionBody,
    Stage2BDecisionRecord,
    Stage2BDecisionResult,
    Stage2BFailurePoint,
    Stage2BOfflineLedger,
    Stage2BOfflineLedgerError,
    Stage2BReconcileStatus,
    Stage2BTransactionKind,
    StoredStage2BDecision,
    StoredStage2BIntent,
)
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_chunk_layout import ChunkActionLayout
from sana_wam.model.video_backbone.sana import hybrid_cache as base
from sana_wam.model.video_backbone.sana import hybrid_cache_stage2b as stage2b
from sana_wam.model.video_backbone.sana import (
    hybrid_cache_stage2b_ledgered as l2b,
)


_RECOVERY_CONSTRUCTION_CAPABILITY = object()
_OBJECT_FILENAME_RE = re.compile(r"object-([0-9a-f]{64})\.bin", re.ASCII)
_INTENT_FILENAME_RE = re.compile(r"intent-([0-9a-f]{64})\.json", re.ASCII)
_DECISION_FILENAME_RE = re.compile(r"decision-([0-9a-f]{64})\.json", re.ASCII)
_LEDGER_ROOT_FILENAME = "LEDGER.json"


def _fail(code: str, message: str) -> None:
    raise base.CacheContractError(code, message)


def _sha256(payload: bytes) -> str:
    return base._sha256_bytes(payload)


@dataclass(frozen=True)
class Stage2BRecoverySummary:
    """Primitive, immutable diagnostics; never an execution authority."""

    ledger_id: str
    writer_fence_id: str
    decision_count: int
    last_decision_sha256: str | None
    recovered_state_manifest: str
    recovered_episode_id: str
    recovered_episode_epoch: int
    recovered_revision: int
    recovered_layout_instance_digest: str
    committed_commit_count: int
    committed_reset_count: int
    closed_process_exit_transaction_id: str | None
    unreferenced_content_object_sha256s: tuple[str, ...]
    model_executed: bool = field(default=False, init=False)
    recovery_admitted: bool = field(default=False, init=False)
    filesystem_admitted: bool = field(default=False, init=False)
    scientific_eligible: bool = field(default=False, init=False)


def _require_cpu_state(state: stage2b.Stage2BTemporalState) -> None:
    for layer in state.layer_states:
        for _, snapshot in layer.tensors:
            if snapshot.to_manifest().get("device") != "cpu":
                _fail("L3_CPU_SYNTHETIC_ONLY", "recovered cache tensor is not CPU")
    history = state.committed_action_history
    if history is not None and str(history.device) != "cpu":
        _fail("L3_CPU_SYNTHETIC_ONLY", "recovered action history is not CPU")


def _require_recovery_inputs(
    *,
    registry: base.LayerRegistry,
    ledger: Stage2BOfflineLedger,
) -> None:
    if type(ledger) is not Stage2BOfflineLedger:
        _fail("L3_LEDGER_MISMATCH", "ledger must be the exact L2A ledger type")
    if type(registry) is not base.LayerRegistry:
        _fail("L3_LEDGER_MISMATCH", "registry must be the exact LayerRegistry type")
    layouts = ledger._expected_layouts
    if (
        not isinstance(layouts, dict)
        or not layouts
        or any(
            type(layout) is not ChunkActionLayout
            or layout.synthetic_test_only is not True
            or digest != layout.layout_instance_digest
            or layout.layout_spec_sha256 != ledger.identity.layout_spec_sha256
            for digest, layout in layouts.items()
        )
    ):
        _fail("L3_LEDGER_MISMATCH", "caller-pinned layout inventory differs")
    if (
        ledger.identity.layer_registry_digest != registry.manifest_digest
        or type(ledger.identity.staging_variant) is not CACHStagingVariant
    ):
        _fail("L3_LEDGER_MISMATCH", "ledger registry or staging identity differs")


def _process_exit_candidate(intent_entry: StoredStage2BIntent) -> Stage2BDecisionRecord:
    intent = intent_entry.intent
    receipt = Stage2BAbortReceipt(
        ledger_id=intent.ledger_id,
        transaction_id=intent.transaction_id,
        transaction_kind=intent.transaction_kind,
        transaction_nonce=intent.transaction_nonce,
        intent_sha256=intent_entry.sha256,
        staging_variant=intent.staging_variant,
        abort_code=Stage2BAbortCode.PROCESS_EXIT,
        failure_point=Stage2BFailurePoint.AFTER_PREPARED,
        state_manifest_before=intent.state_manifest_before,
        state_manifest_after=intent.state_manifest_before,
    )
    return Stage2BDecisionRecord.from_body(
        Stage2BDecisionBody(
            ledger_id=intent.ledger_id,
            ledger_sequence=intent.ledger_sequence,
            transaction_id=intent.transaction_id,
            transaction_kind=intent.transaction_kind,
            intent_sha256=intent_entry.sha256,
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
            abort_code=Stage2BAbortCode.PROCESS_EXIT,
        )
    )


def _close_pending_process_exit(
    *,
    ledger: Stage2BOfflineLedger,
    pending: StoredStage2BIntent,
) -> StoredStage2BDecision:
    candidate = _process_exit_candidate(pending)
    intent = pending.intent
    try:
        stored = ledger.abort(
            intent=intent,
            abort_code=Stage2BAbortCode.PROCESS_EXIT,
            failure_point=Stage2BFailurePoint.AFTER_PREPARED,
        )
    except Exception as publication_error:
        try:
            reconciliation = ledger.reconcile_terminal(
                transaction_id=intent.transaction_id,
                expected_record_sha256=candidate.decision_sha256,
            )
        except Exception as reconciliation_error:
            raise base.CacheContractError(
                "L3_RECOVERY_OUTCOME_UNKNOWN",
                "PROCESS_EXIT publication did not reconcile exactly",
            ) from reconciliation_error
        stored = reconciliation.stored
        if (
            reconciliation.status is not Stage2BReconcileStatus.EXACT
            or stored is None
        ):
            raise base.CacheContractError(
                "L3_RECOVERY_OUTCOME_UNKNOWN",
                "PROCESS_EXIT publication is not an exact terminal",
            ) from publication_error
    if (
        stored.sha256 != candidate.decision_sha256
        or stored.payload != candidate.canonical_bytes
        or stored.decision != candidate
    ):
        _fail("L3_RECOVERY_OUTCOME_UNKNOWN", "PROCESS_EXIT terminal bytes differ")
    return stored


def _scan_content_objects(ledger: Stage2BOfflineLedger) -> tuple[str, ...]:
    """Return a stable exact content-object set using a strict local parser."""

    ledger._scan_records()
    try:
        before = tuple(sorted(os.listdir(ledger.root)))
    except OSError as exc:
        raise base.CacheContractError(
            "L3_LEDGER_MISMATCH", "ledger object inventory cannot be listed"
        ) from exc
    digests: list[str] = []
    for filename in before:
        object_match = _OBJECT_FILENAME_RE.fullmatch(filename)
        if object_match is not None:
            digest = object_match.group(1)
            stored = ledger.read_object(digest)
            if stored.sha256 != digest or stored.path.name != filename:
                _fail("L3_LEDGER_MISMATCH", "content object identity differs")
            digests.append(digest)
            continue
        if (
            filename == _LEDGER_ROOT_FILENAME
            or _INTENT_FILENAME_RE.fullmatch(filename) is not None
            or _DECISION_FILENAME_RE.fullmatch(filename) is not None
        ):
            continue
        _fail("L3_LEDGER_MISMATCH", f"unexpected ledger entry: {filename}")
    try:
        after = tuple(sorted(os.listdir(ledger.root)))
    except OSError as exc:
        raise base.CacheContractError(
            "L3_LEDGER_MISMATCH", "ledger object inventory cannot be relisted"
        ) from exc
    if after != before:
        _fail("L3_RECOVERY_OUTCOME_UNKNOWN", "ledger entries changed during scan")
    ledger._scan_records()
    if len(digests) != len(set(digests)):
        _fail("L3_LEDGER_MISMATCH", "duplicate logical content object")
    return tuple(sorted(digests))


def _referenced_content_objects(
    *,
    ledger: Stage2BOfflineLedger,
    intents: Mapping[str, StoredStage2BIntent],
    ordered: tuple[StoredStage2BDecision, ...],
) -> frozenset[str]:
    referenced: set[str] = set()

    def add_snapshot(manifest_sha256: str) -> None:
        stored = ledger.read_object(manifest_sha256)
        manifest, _ = ledger._load_snapshot(manifest_sha256)
        if stored.sha256 != manifest_sha256:
            _fail("L3_LEDGER_MISMATCH", "snapshot manifest object differs")
        referenced.add(manifest_sha256)
        referenced.update(offline._collect_snapshot_blob_references(manifest))

    add_snapshot(ledger.identity.genesis_snapshot_manifest_sha256)
    for intent_entry in intents.values():
        intent = intent_entry.intent
        identity_object = ledger.read_object(intent.input_identity_sha256)
        operation_identity = offline._parse_operation_identity(identity_object.payload)
        if operation_identity.identity_sha256 != intent.input_identity_sha256:
            _fail("L3_LEDGER_MISMATCH", "operation identity object differs")
        operation_object = ledger.read_object(
            operation_identity.operation_payload_sha256
        )
        offline._parse_operation_payload(
            operation_object.payload,
            operation_identity.transaction_kind,
        )
        referenced.add(intent.input_identity_sha256)
        referenced.add(operation_identity.operation_payload_sha256)
    for decision_entry in ordered:
        body = decision_entry.decision.decision_body
        ledger.read_object(body.success_or_abort_receipt_sha256)
        referenced.add(body.success_or_abort_receipt_sha256)
        if body.result is Stage2BDecisionResult.COMMITTED:
            assert body.snapshot_manifest_sha256 is not None
            add_snapshot(body.snapshot_manifest_sha256)
    return frozenset(referenced)


def _paired_receipt(
    manifest: dict[str, object], payload: bytes
) -> stage2b.Stage2BPairedCommitReceipt:
    try:
        receipt = stage2b.Stage2BPairedCommitReceipt(
            commit_id=manifest["commit_id"],
            staging_variant=CACHStagingVariant(manifest["staging_variant"]),
            transaction_nonce=manifest["transaction_nonce"],
            episode_id=manifest["episode_id"],
            episode_epoch=manifest["episode_epoch"],
            revision_before=manifest["revision_before"],
            revision_after=manifest["revision_after"],
            chunk_id=manifest["chunk_id"],
            layout_instance_digest=manifest["layout_instance_digest"],
            source_proof_digest=manifest["source_proof_digest"],
            pair_video_digest=manifest["pair_video_digest"],
            pair_frame_valid_mask_digest=manifest["pair_frame_valid_mask_digest"],
            pair_action_digest=manifest["pair_action_digest"],
            pair_action_mask_digest=manifest["pair_action_mask_digest"],
            teacher_pair_payload_digest=manifest["teacher_pair_payload_digest"],
            paired_payload_digest=manifest["paired_payload_digest"],
            conditioning_digest=manifest["conditioning_digest"],
            dataset_manifest_sha256=manifest["dataset_manifest_sha256"],
            dataset_row_identity=manifest["dataset_row_identity"],
            state_manifest_before=manifest["state_manifest_before"],
            staged_state_manifest=manifest["staged_state_manifest"],
            state_manifest_after=manifest["state_manifest_after"],
            action_history_digest_before=manifest["action_history_digest_before"],
            action_history_digest_after=manifest["action_history_digest_after"],
            action_cursor_before=manifest["action_cursor_before"],
            action_cursor_after=manifest["action_cursor_after"],
            committed_at_monotonic_ns=manifest["committed_at_monotonic_ns"],
            commit_source=base.CommitSource(manifest["commit_source"]),
            source_proof_schema=manifest["source_proof_schema"],
            result=manifest["result"],
            schema=manifest["schema"],
        )
    except (KeyError, TypeError, ValueError, base.CacheContractError) as exc:
        raise base.CacheContractError(
            "L3_LEDGER_MISMATCH", "paired receipt typed reconstruction failed"
        ) from exc
    if receipt.canonical_bytes != payload or receipt.receipt_sha256 != _sha256(payload):
        _fail("L3_LEDGER_MISMATCH", "paired receipt round trip differs")
    return receipt


def _reset_receipt(
    manifest: dict[str, object], payload: bytes
) -> stage2b.Stage2BCacheResetReceipt:
    try:
        receipt = stage2b.Stage2BCacheResetReceipt(
            reset_id=manifest["reset_id"],
            staging_variant=CACHStagingVariant(manifest["staging_variant"]),
            transaction_nonce=manifest["transaction_nonce"],
            previous_episode_id=manifest["previous_episode_id"],
            previous_episode_epoch=manifest["previous_episode_epoch"],
            new_episode_id=manifest["new_episode_id"],
            new_episode_epoch=manifest["new_episode_epoch"],
            layout_spec_sha256=manifest["layout_spec_sha256"],
            layout_instance_digest=manifest["layout_instance_digest"],
            state_manifest_before=manifest["state_manifest_before"],
            state_manifest_after=manifest["state_manifest_after"],
            action_history_digest_before=manifest["action_history_digest_before"],
            action_history_digest_after=manifest["action_history_digest_after"],
            action_cursor_before=manifest["action_cursor_before"],
            action_cursor_after=manifest["action_cursor_after"],
            aborted_pending_commit_id=manifest["aborted_pending_commit_id"],
            reset_at_monotonic_ns=manifest["reset_at_monotonic_ns"],
            schema=manifest["schema"],
        )
    except (KeyError, TypeError, ValueError, base.CacheContractError) as exc:
        raise base.CacheContractError(
            "L3_LEDGER_MISMATCH", "reset receipt typed reconstruction failed"
        ) from exc
    if receipt.canonical_bytes != payload or receipt.receipt_sha256 != _sha256(payload):
        _fail("L3_LEDGER_MISMATCH", "reset receipt round trip differs")
    return receipt


def _reset_retry_digest(
    *,
    reset_id: str,
    transaction_nonce: str,
    new_episode_id: str,
    layout: ChunkActionLayout,
    staging_variant: CACHStagingVariant,
) -> str:
    return _sha256(
        base._canonical_json_bytes(
            {
                "layout_instance_digest": layout.layout_instance_digest,
                "layout_spec_sha256": layout.layout_spec_sha256,
                "new_episode_id": new_episode_id,
                "reset_id": reset_id,
                "schema": "cach.stage2b.cache_reset_receipt.v1",
                "staging_variant": staging_variant.value,
                "transaction_nonce": transaction_nonce,
            }
        )
    )


@dataclass(frozen=True)
class _RecoveredMaps:
    facade_commits: dict[
        str, tuple[str, stage2b.Stage2BPairedCommitReceipt, str]
    ]
    facade_resets: dict[str, tuple[str, stage2b.Stage2BCacheResetReceipt, str]]
    manager_commits: dict[str, tuple[str, stage2b.Stage2BPairedCommitReceipt]]
    manager_resets: dict[str, tuple[str, stage2b.Stage2BCacheResetReceipt]]


def _reconstruct_maps(
    *,
    ledger: Stage2BOfflineLedger,
    intents: Mapping[str, StoredStage2BIntent],
    ordered: tuple[StoredStage2BDecision, ...],
) -> _RecoveredMaps:
    facade_commits: dict[
        str, tuple[str, stage2b.Stage2BPairedCommitReceipt, str]
    ] = {}
    facade_resets: dict[
        str, tuple[str, stage2b.Stage2BCacheResetReceipt, str]
    ] = {}
    manager_commits: dict[
        str, tuple[str, stage2b.Stage2BPairedCommitReceipt]
    ] = {}
    manager_resets: dict[str, tuple[str, stage2b.Stage2BCacheResetReceipt]] = {}
    all_ids: set[str] = set()
    for decision_entry in ordered:
        body = decision_entry.decision.decision_body
        if body.result is Stage2BDecisionResult.ABORTED:
            continue
        if body.transaction_id in all_ids:
            _fail("L3_LEDGER_MISMATCH", "duplicate committed transaction ID")
        all_ids.add(body.transaction_id)
        intent_entry = intents.get(body.transaction_id)
        if intent_entry is None:
            _fail("L3_LEDGER_MISMATCH", "committed decision lacks exact intent")
        operation_identity, operation_manifest = ledger._validate_intent_input(
            intent_entry.intent
        )
        receipt_object = ledger.read_object(body.success_or_abort_receipt_sha256)
        try:
            manifest, schema = l0_receipts._validate_payload(
                receipt_object.payload,
                body.transaction_id,
            )
        except l0_receipts.Stage2BReceiptStoreError as exc:
            raise base.CacheContractError(
                "L3_LEDGER_MISMATCH", "success receipt failed strict L0 parsing"
            ) from exc
        if receipt_object.sha256 != body.success_or_abort_receipt_sha256:
            _fail("L3_LEDGER_MISMATCH", "success receipt object hash differs")
        if body.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT:
            if schema != "cach.stage2b.paired_commit_receipt.v1":
                _fail("L3_LEDGER_MISMATCH", "paired receipt schema differs")
            receipt = _paired_receipt(manifest, receipt_object.payload)
            if operation_identity.transaction_kind is not body.transaction_kind:
                _fail("L3_LEDGER_MISMATCH", "paired operation kind differs")
            facade_commits[body.transaction_id] = (
                operation_identity.operation_payload_sha256,
                receipt,
                decision_entry.sha256,
            )
            manager_commits[body.transaction_id] = (
                receipt.paired_payload_digest,
                receipt,
            )
        else:
            if schema != "cach.stage2b.cache_reset_receipt.v1":
                _fail("L3_LEDGER_MISMATCH", "reset receipt schema differs")
            receipt = _reset_receipt(manifest, receipt_object.payload)
            try:
                layout = ledger._expected_layouts[
                    operation_identity.layout_instance_digest
                ]
            except KeyError as exc:
                raise base.CacheContractError(
                    "L3_LEDGER_MISMATCH", "reset target layout is not pinned"
                ) from exc
            retry_digest = _reset_retry_digest(
                reset_id=body.transaction_id,
                transaction_nonce=operation_identity.transaction_nonce,
                new_episode_id=operation_manifest["new_episode_id"],
                layout=layout,
                staging_variant=operation_identity.staging_variant,
            )
            facade_resets[body.transaction_id] = (
                retry_digest,
                receipt,
                decision_entry.sha256,
            )
            manager_resets[body.transaction_id] = (retry_digest, receipt)
    return _RecoveredMaps(
        facade_commits=facade_commits,
        facade_resets=facade_resets,
        manager_commits=manager_commits,
        manager_resets=manager_resets,
    )


class _RecoveredLedgerSeamedStage2BHybridCacheManager(
    l2b._LedgerSeamedStage2BHybridCacheManager
):
    """One-shot state/map installation seam used before any reference escapes."""

    def __init__(self, **kwargs: object) -> None:
        self._l3_state_installed = False
        super().__init__(**kwargs)

    def _install_recovered_state(
        self,
        *,
        state: stage2b.Stage2BTemporalState,
        layout: ChunkActionLayout,
        completed: Mapping[
            str, tuple[str, stage2b.Stage2BPairedCommitReceipt]
        ],
        completed_resets: Mapping[
            str, tuple[str, stage2b.Stage2BCacheResetReceipt]
        ],
    ) -> None:
        detached = state.clone()
        with self._lock:
            initial = self._state
            if (
                self._l3_state_installed
                or initial.revision != 0
                or initial.committed_through_chunk is not None
                or initial.next_chunk_id != 0
                or initial.action_cursor != 0
                or initial.committed_action_history is not None
                or any(
                    layer.through is not None or layer.tensors
                    for layer in initial.layer_states
                )
                or self._state_manifest_digest != initial.state_manifest_digest
                or self._layout.to_payload() != layout.to_payload()
                or self._completed
                or self._completed_resets
                or self._pending is not None
                or self._publication_token is not None
                or self._poisoned
            ):
                _fail("L3_INSTALL_FAILED", "manager pre-install invariants differ")
            if (
                detached.episode_id != self._state.episode_id
                or detached.episode_epoch != self._state.episode_epoch
                or detached.layout_spec_sha256 != layout.layout_spec_sha256
                or detached.layout_instance_digest != layout.layout_instance_digest
                or detached.staging_variant is not self.staging_variant
            ):
                _fail("L3_INSTALL_FAILED", "recovered state identity differs")
            self._state = detached
            self._state_manifest_digest = detached.state_manifest_digest
            self._layout = layout
            self._completed = dict(completed)
            self._completed_resets = dict(completed_resets)
            self._l3_state_installed = True
            if (
                set(self._completed) & set(self._completed_resets)
                or self._pending is not None
                or self._publication_token is not None
                or self._poisoned
                or self._state.state_manifest_digest != self._state_manifest_digest
                or self._state.to_manifest() != detached.to_manifest()
            ):
                _fail("L3_INSTALL_FAILED", "manager post-install invariants differ")


class Stage2BRecoveredLedgeredHybridCacheManager(
    l2b.Stage2BLedgeredHybridCacheManager
):
    """Recovered L2B facade for one exact CPU/synthetic ledger chain."""

    def __init__(
        self,
        *,
        _recovery_capability: object,
        registry: base.LayerRegistry,
        layout: ChunkActionLayout,
        ledger: Stage2BOfflineLedger,
        manager: _RecoveredLedgerSeamedStage2BHybridCacheManager,
        capture_bridge: l2b._CaptureBridge,
        publisher_adapter: l2b._LedgerPublisherAdapter,
        recovered_maps: _RecoveredMaps,
    ) -> None:
        if _recovery_capability is not _RECOVERY_CONSTRUCTION_CAPABILITY:
            _fail("L3_CONSTRUCTION_GUARDED", "use the L3 recovery factories")
        super().__init__(
            _capability=l2b._CONSTRUCTION_CAPABILITY,
            registry=registry,
            layout=layout,
            ledger=ledger,
            manager=manager,
            capture_bridge=capture_bridge,
            publisher_adapter=publisher_adapter,
            synthetic_test_only=True,
        )
        self._recovery_summary_value: Stage2BRecoverySummary | None = None
        with self._lock:
            if (
                self._phase != l2b._READY
                or self._active is not None
                or self._poisoned
                or self._publisher_adapter._armed_token is not None
                or self._completed_commits
                or self._completed_resets
            ):
                _fail("L3_INSTALL_FAILED", "facade pre-install invariants differ")
            self._completed_commits = dict(recovered_maps.facade_commits)
            self._completed_resets = dict(recovered_maps.facade_resets)
            if (
                self._phase != l2b._READY
                or self._active is not None
                or self._poisoned
                or self._publisher_adapter._armed_token is not None
                or set(self._completed_commits) & set(self._completed_resets)
                or set(self._completed_commits) != set(self._manager._completed)
                or set(self._completed_resets)
                != set(self._manager._completed_resets)
            ):
                _fail("L3_INSTALL_FAILED", "facade post-install invariants differ")

    def _install_recovery_summary(self, summary: Stage2BRecoverySummary) -> None:
        with self._lock:
            if (
                type(summary) is not Stage2BRecoverySummary
                or summary.model_executed
                or summary.recovery_admitted
                or summary.filesystem_admitted
                or summary.scientific_eligible
                or self._recovery_summary_value is not None
            ):
                _fail("L3_INSTALL_FAILED", "recovery summary invariants differ")
            self._assert_ready_locked()
            self._recovery_summary_value = summary

    @property
    def recovery_summary(self) -> Stage2BRecoverySummary:
        with self._lock:
            self._assert_ready_locked()
            summary = self._recovery_summary_value
            if summary is None:
                _fail("L3_INSTALL_FAILED", "recovery summary is unavailable")
            return summary

    @classmethod
    def recover_from_ledger(
        cls,
        *,
        registry: base.LayerRegistry,
        ledger: Stage2BOfflineLedger,
        synthetic_test_only: bool = False,
    ) -> Stage2BRecoveredLedgeredHybridCacheManager:
        if type(synthetic_test_only) is not bool:
            _fail("L3_CPU_SYNTHETIC_ONLY", "synthetic flag must be exact bool")
        if not synthetic_test_only:
            _fail(
                "L3_CPU_SYNTHETIC_ONLY",
                "L3 recovery is restricted to explicit CPU synthetic use",
            )
        _require_recovery_inputs(registry=registry, ledger=ledger)

        try:
            with ledger._lock:
                intents, _, ordered_before, pending = ledger._scan_records()
                closed_transaction_id: str | None = None
                if pending:
                    if len(pending) != 1:
                        _fail("L3_LEDGER_MISMATCH", "multiple pending intents exist")
                    closed = _close_pending_process_exit(
                        ledger=ledger,
                        pending=pending[0],
                    )
                    closed_transaction_id = closed.decision.decision_body.transaction_id

                intents, _, ordered, pending_after = ledger._scan_records()
                if pending_after:
                    _fail("L3_RECOVERY_OUTCOME_UNKNOWN", "pending intent remains")
                if closed_transaction_id is not None:
                    if (
                        len(ordered) != len(ordered_before) + 1
                        or ordered[-1].sha256
                        != _process_exit_candidate(
                            intents[closed_transaction_id]
                        ).decision_sha256
                    ):
                        _fail(
                            "L3_RECOVERY_OUTCOME_UNKNOWN",
                            "PROCESS_EXIT closure did not append exactly once",
                        )

                # Reachability is intentionally computed only after the pending
                # transaction has an authoritative terminal decision.
                object_set = _scan_content_objects(ledger)
                referenced = _referenced_content_objects(
                    ledger=ledger,
                    intents=intents,
                    ordered=ordered,
                )
                if not referenced <= set(object_set):
                    _fail("L3_LEDGER_MISMATCH", "authoritative object is absent")
                inert_debris = tuple(sorted(set(object_set) - referenced))

                state = ledger._state_after_decisions(ordered).clone()
                state_snapshot_sha256 = (
                    ledger.identity.genesis_snapshot_manifest_sha256
                )
                for decision_entry in reversed(ordered):
                    body = decision_entry.decision.decision_body
                    if body.result is Stage2BDecisionResult.COMMITTED:
                        assert body.snapshot_manifest_sha256 is not None
                        state_snapshot_sha256 = body.snapshot_manifest_sha256
                        break
                snapshot_manifest, decoded_state = ledger._load_snapshot(
                    state_snapshot_sha256
                )
                layout = ledger._layout_for_snapshot_manifest(snapshot_manifest)
                authoritative_state_manifest = (
                    ledger.identity.genesis_state_manifest
                    if not ordered
                    else ordered[-1].decision.decision_body.state_manifest_after
                )
                final_body = (
                    None if not ordered else ordered[-1].decision.decision_body
                )
                if (
                    state.to_manifest() != decoded_state.to_manifest()
                    or state.state_manifest_digest != authoritative_state_manifest
                    or snapshot_manifest.get("state_manifest") != state.to_manifest()
                    or snapshot_manifest.get("state_manifest_digest")
                    != state.state_manifest_digest
                    or snapshot_manifest.get("registry_identity")
                    != {
                        "layer_registry_digest": registry.manifest_digest,
                        "registry_manifest": registry.to_manifest(),
                    }
                    or (
                        final_body is not None
                        and (
                            state.episode_id != final_body.episode_id_after
                            or state.episode_epoch != final_body.episode_epoch_after
                            or state.revision != final_body.revision_after
                        )
                    )
                    or state.layout_spec_sha256 != layout.layout_spec_sha256
                    or state.layout_instance_digest != layout.layout_instance_digest
                    or state.staging_variant is not ledger.identity.staging_variant
                ):
                    _fail("L3_LEDGER_MISMATCH", "authoritative recovered state differs")
                _require_cpu_state(state)

                recovered_maps = _reconstruct_maps(
                    ledger=ledger,
                    intents=intents,
                    ordered=ordered,
                )
                bridge = l2b._CaptureBridge()
                adapter = l2b._LedgerPublisherAdapter()
                manager = _RecoveredLedgerSeamedStage2BHybridCacheManager(
                    capture_bridge=bridge,
                    registry=registry,
                    episode_id=state.episode_id,
                    layout=layout,
                    receipt_publisher=adapter,
                    staging_variant=ledger.identity.staging_variant,
                    episode_epoch=state.episode_epoch,
                    synthetic_test_only=True,
                )
                manager._install_recovered_state(
                    state=state,
                    layout=layout,
                    completed=recovered_maps.manager_commits,
                    completed_resets=recovered_maps.manager_resets,
                )
                facade = cls(
                    _recovery_capability=_RECOVERY_CONSTRUCTION_CAPABILITY,
                    registry=registry,
                    layout=layout,
                    ledger=ledger,
                    manager=manager,
                    capture_bridge=bridge,
                    publisher_adapter=adapter,
                    recovered_maps=recovered_maps,
                )
                if facade.state.to_manifest() != state.to_manifest():
                    _fail("L3_INSTALL_FAILED", "installed public state differs")
                for reset_id, (
                    digest,
                    receipt,
                    _,
                ) in recovered_maps.facade_resets.items():
                    target_layout = ledger._expected_layouts[
                        receipt.layout_instance_digest
                    ]
                    if facade._reset_retry_digest(
                        reset_id=reset_id,
                        transaction_nonce=receipt.transaction_nonce,
                        new_episode_id=receipt.new_episode_id,
                        layout=target_layout,
                    ) != digest:
                        _fail("L3_INSTALL_FAILED", "reset retry digest seam differs")

                _, _, final_ordered, final_pending = ledger._scan_records()
                final_object_set = _scan_content_objects(ledger)
                if (
                    final_pending
                    or tuple(
                        (item.sha256, item.payload, item.fingerprint)
                        for item in final_ordered
                    )
                    != tuple(
                        (item.sha256, item.payload, item.fingerprint)
                        for item in ordered
                    )
                    or final_object_set != object_set
                ):
                    _fail(
                        "L3_RECOVERY_OUTCOME_UNKNOWN",
                        "ledger chain or object inventory changed before return",
                    )

                # The summary is created after closure and the final stable
                # chain/object rescan, so it cannot describe a pre-closure root.
                summary = Stage2BRecoverySummary(
                    ledger_id=ledger.identity.ledger_id,
                    writer_fence_id=ledger.identity.writer_fence_id,
                    decision_count=len(final_ordered),
                    last_decision_sha256=(
                        None if not final_ordered else final_ordered[-1].sha256
                    ),
                    recovered_state_manifest=state.state_manifest_digest,
                    recovered_episode_id=state.episode_id,
                    recovered_episode_epoch=state.episode_epoch,
                    recovered_revision=state.revision,
                    recovered_layout_instance_digest=state.layout_instance_digest,
                    committed_commit_count=len(recovered_maps.facade_commits),
                    committed_reset_count=len(recovered_maps.facade_resets),
                    closed_process_exit_transaction_id=closed_transaction_id,
                    unreferenced_content_object_sha256s=inert_debris,
                )
                facade._install_recovery_summary(summary)
                return facade
        except base.CacheContractError:
            raise
        except Stage2BOfflineLedgerError as exc:
            raise base.CacheContractError(
                "L3_LEDGER_MISMATCH", f"ledger recovery failed: {exc.code}"
            ) from exc

    @classmethod
    def for_synthetic_recovery(
        cls,
        *,
        registry: base.LayerRegistry,
        ledger: Stage2BOfflineLedger,
    ) -> Stage2BRecoveredLedgeredHybridCacheManager:
        return cls.recover_from_ledger(
            registry=registry,
            ledger=ledger,
            synthetic_test_only=True,
        )
