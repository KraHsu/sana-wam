"""Fresh-only L2B ledger facade for the frozen Stage-2B cache manager.

This module binds one live :class:`Stage2BHybridCacheManager` to the bounded
CPU/synthetic L2A offline ledger.  It deliberately uses a pinned private seam:
an inherited manager authorizes the original production stager, a capturing
callable encodes the detached staged state, and a private publisher adapter
writes the terminal ledger decision before the inherited in-memory CAS.

It does not implement restart recovery, state installation, multi-process
writer fencing, production-filesystem admission, model execution, training,
evaluation, deployment, or scientific authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import threading
from typing import Callable

import torch

from sana_wam.cach.committed_action_history import (
    CommittedActionHistory,
    CommittedActionHistoryError,
    committed_action_tensor_digest,
)
from sana_wam.cach import stage2b_receipt_store as l0_receipts
from sana_wam.cach.stage2b_offline_ledger import (
    Stage2BAbortCode,
    Stage2BDecisionBody,
    Stage2BDecisionRecord,
    Stage2BDecisionResult,
    Stage2BFailurePoint,
    Stage2BLedgerIntent,
    Stage2BOfflineLedger,
    Stage2BOfflineLedgerError,
    Stage2BOperationIdentity,
    Stage2BReconcileStatus,
    Stage2BTransactionKind,
    StoredStage2BDecision,
)
from sana_wam.cach.stage2b_state_snapshot import (
    EncodedStage2BStateSnapshot,
    encode_stage2b_state_snapshot,
)
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_chunk_layout import ChunkActionLayout
from sana_wam.model.video_backbone.sana import hybrid_cache as base
from sana_wam.model.video_backbone.sana import hybrid_cache_stage2b as stage2b


_PAIRED_OPERATION_SCHEMA = "cach.stage2b.paired_commit_operation.v1"
_PAIRED_PAYLOAD_SCHEMA = "cach.stage2b.paired_payload.v1"
_RESET_OPERATION_SCHEMA = "cach.stage2b.reset_operation.v1"
_LEDGER_ROOT_FILENAME = "LEDGER.json"

_READY = "READY"
_PREPARING = "PREPARING"
_PREPARED = "PREPARED"
_STAGING = "STAGING"
_STAGED = "STAGED"
_TERMINAL_PUBLISHING = "TERMINAL_PUBLISHING"
_TERMINAL_EXACT = "TERMINAL_EXACT"
_CAS_CONFIRMING = "CAS_CONFIRMING"
_ABORTING = "ABORTING"
_POISONED = "POISONED"

_CONSTRUCTION_CAPABILITY = object()


def _fail(code: str, message: str) -> None:
    raise base.CacheContractError(code, message)


def _sha256(payload: bytes) -> str:
    return base._sha256_bytes(payload)


def _history_digest(state: stage2b.Stage2BTemporalState) -> str | None:
    history = state.committed_action_history
    return None if history is None else history.manifest_digest


def _strict_manifest(payload: bytes, name: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload:
        _fail("L2B_LEDGER_MISMATCH", f"{name} must be non-empty bytes")
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise base.CacheContractError(
            "L2B_LEDGER_MISMATCH",
            f"{name} is not strict UTF-8 JSON",
        ) from exc
    if not isinstance(value, dict) or base._canonical_json_bytes(value) != payload:
        _fail("L2B_LEDGER_MISMATCH", f"{name} is not one canonical JSON object")
    return value


def _snapshot_blob_digests(value: object) -> set[str]:
    """Collect only raw tensor objects referenced by an L1 manifest."""

    observed: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "raw_blob_sha256" and isinstance(nested, str):
                observed.add(nested)
            else:
                observed.update(_snapshot_blob_digests(nested))
    elif isinstance(value, list):
        for nested in value:
            observed.update(_snapshot_blob_digests(nested))
    return observed


def _authorize_paired_commit(
    *,
    commit_source: base.CommitSource | str,
    request: stage2b.Stage2BTeacherForcingCommitRequest | None,
    stage_callback: Callable[
        [stage2b.Stage2BPairedStagingContext],
        tuple[base.StagedLayerPayload, ...],
    ]
    | None,
    synthetic_test_only: bool,
) -> tuple[
    stage2b.Stage2BTeacherForcingCommitRequest,
    Callable[
        [stage2b.Stage2BPairedStagingContext],
        tuple[base.StagedLayerPayload, ...],
    ],
]:
    """Mirror the frozen public source/request/exact-stager authorization."""

    try:
        source = (
            commit_source
            if isinstance(commit_source, base.CommitSource)
            else base.CommitSource(commit_source)
        )
    except (TypeError, ValueError):
        base._fail("COMMIT_SOURCE_UNKNOWN", "unknown paired commit source")
    if source is base.CommitSource.SELF_FORCING:
        base._fail(
            "COMMIT_SELF_FORCING_OBJECTIVE_UNCLOSED",
            "self-forcing remains disabled",
        )
    if source is base.CommitSource.DEPLOY_APPLIED_ACK:
        base._fail(
            "COMMIT_DEPLOY_ACK_MISSING",
            "deploy history remains blocked on verified ACK transport",
        )
    if not isinstance(request, stage2b.Stage2BTeacherForcingCommitRequest):
        base._fail(
            "COMMIT_SOURCE_PROOF_MISSING",
            "Stage-2B teacher request and proof are required",
        )
    if not callable(stage_callback):
        base._fail("COMMIT_STAGING_CALLBACK_MISSING", "stager is required")
    if not synthetic_test_only:
        if request.conditioning_digest is None:
            base._fail(
                "COMMIT_CONDITIONING_IDENTITY_MISSING",
                "production-shaped staging requires an exact conditioning digest",
            )
        from sana_wam.model.cach_paired_stager import (
            CACHTeacherForcingPairedStager,
        )

        if type(stage_callback) is not CACHTeacherForcingPairedStager:
            base._fail(
                "COMMIT_STAGING_CALLBACK_UNAUTHORIZED",
                "non-synthetic commits require the exact production paired stager",
            )
    return request, stage_callback


@dataclass
class _ActiveTransaction:
    token: object
    transaction_kind: Stage2BTransactionKind
    transaction_id: str
    retry_digest: str
    operation_identity: Stage2BOperationIdentity
    operation_payload: bytes
    intent: Stage2BLedgerIntent
    base_state: stage2b.Stage2BTemporalState
    target_layout: ChunkActionLayout
    transaction_payload_digest: str | None = None
    prepared_pair: object | None = None
    base_history: CommittedActionHistory | None = None
    staged_state: stage2b.Stage2BTemporalState | None = None
    snapshot: EncodedStage2BStateSnapshot | None = None
    receipt_payload: bytes | None = None
    receipt_sha256: str | None = None
    terminal: StoredStage2BDecision | None = None
    publisher_entered: bool = False
    publisher_calls: int = 0


class _CaptureBridge:
    """One-shot circular binder used by the inherited private seam."""

    def __init__(self) -> None:
        self._owner: Stage2BLedgeredHybridCacheManager | None = None

    def bind(self, owner: Stage2BLedgeredHybridCacheManager) -> None:
        if self._owner is not None:
            _fail("L2B_LEDGER_MISMATCH", "capture bridge was already bound")
        self._owner = owner

    def wrap(
        self,
        delegate: Callable[
            [stage2b.Stage2BPairedStagingContext],
            tuple[base.StagedLayerPayload, ...],
        ],
    ) -> Callable[
        [stage2b.Stage2BPairedStagingContext],
        tuple[base.StagedLayerPayload, ...],
    ]:
        owner = self._owner
        if owner is None:
            _fail("L2B_LEDGER_MISMATCH", "capture bridge is unbound")
        with owner._lock:
            active = owner._active
            if (
                active is None
                or active.transaction_kind is not Stage2BTransactionKind.PAIRED_COMMIT
                or owner._phase != _PREPARED
            ):
                owner._poison_locked("capturing seam entered outside PREPARED")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "capturing seam has no matching PREPARED transaction",
                )
            token = active.token

        def capture(
            context: stage2b.Stage2BPairedStagingContext,
        ) -> tuple[base.StagedLayerPayload, ...]:
            return owner._capture_staging(
                token=token,
                delegate=delegate,
                context=context,
            )

        return capture


class _LedgerPublisherAdapter:
    """Publish exact L2A terminal decisions from the inherited callback."""

    def __init__(self) -> None:
        self._owner: Stage2BLedgeredHybridCacheManager | None = None
        self._armed_token: object | None = None

    def bind(self, owner: Stage2BLedgeredHybridCacheManager) -> None:
        if self._owner is not None:
            _fail("L2B_LEDGER_MISMATCH", "publisher adapter was already bound")
        self._owner = owner

    def arm(self, token: object) -> None:
        if self._armed_token is not None:
            _fail("L2B_TRANSACTION_BUSY", "publisher adapter is already armed")
        self._armed_token = token

    def disarm(self, token: object) -> None:
        if self._armed_token is token:
            self._armed_token = None

    def publish_exclusive(
        self,
        *,
        receipt_id: str,
        payload: bytes,
        expected_sha256: str,
    ) -> base.ReceiptPublication:
        owner = self._owner
        token = self._armed_token
        if owner is None:
            _fail(
                "L2B_PUBLICATION_POISONED",
                "publisher adapter is unbound",
            )
        if token is None:
            with owner._lock:
                owner._poison_locked("publisher adapter was invoked while unarmed")
            _fail(
                "L2B_PUBLICATION_POISONED",
                "publisher adapter is unarmed",
            )
        return owner._publish_terminal(
            token=token,
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=expected_sha256,
        )


class _LedgerSeamedStage2BHybridCacheManager(stage2b.Stage2BHybridCacheManager):
    """Pinned private subclass preserving the exact production-stager gate."""

    def __init__(self, *, capture_bridge: _CaptureBridge, **kwargs: object) -> None:
        if not isinstance(capture_bridge, _CaptureBridge):
            base._fail("CACHE_SCHEMA_MISMATCH", "capture bridge has wrong type")
        self._l2b_capture_bridge = capture_bridge
        super().__init__(**kwargs)

    def commit_paired(
        self,
        *,
        commit_source: base.CommitSource | str,
        request: stage2b.Stage2BTeacherForcingCommitRequest | None,
        stage_callback: Callable[
            [stage2b.Stage2BPairedStagingContext],
            tuple[base.StagedLayerPayload, ...],
        ]
        | None,
    ) -> stage2b.Stage2BPairedCommitReceipt:
        authorized_request, authorized_delegate = _authorize_paired_commit(
            commit_source=commit_source,
            request=request,
            stage_callback=stage_callback,
            synthetic_test_only=self._synthetic_test_mode,
        )
        capture = self._l2b_capture_bridge.wrap(authorized_delegate)
        return self._commit_teacher(
            request=authorized_request,
            stage_callback=capture,
        )


class Stage2BLedgeredHybridCacheManager:
    """Fresh-only, fail-closed facade joining the frozen manager and L2A."""

    def __init__(
        self,
        *,
        _capability: object,
        registry: base.LayerRegistry,
        layout: ChunkActionLayout,
        ledger: Stage2BOfflineLedger,
        manager: _LedgerSeamedStage2BHybridCacheManager,
        capture_bridge: _CaptureBridge,
        publisher_adapter: _LedgerPublisherAdapter,
        synthetic_test_only: bool,
    ) -> None:
        if _capability is not _CONSTRUCTION_CAPABILITY:
            _fail(
                "L2B_RECOVERY_REQUIRED",
                "use initialize_fresh or for_synthetic_tests",
            )
        self._registry = registry
        self._layout = layout
        self._ledger = ledger
        self._manager = manager
        self._capture_bridge = capture_bridge
        self._publisher_adapter = publisher_adapter
        self._synthetic_test_only = synthetic_test_only
        self._lock = threading.RLock()
        self._phase = _READY
        self._active: _ActiveTransaction | None = None
        self._poisoned = False
        self._poison_reason: str | None = None
        self._completed_commits: dict[
            str,
            tuple[str, stage2b.Stage2BPairedCommitReceipt, str],
        ] = {}
        self._completed_resets: dict[
            str,
            tuple[str, stage2b.Stage2BCacheResetReceipt, str],
        ] = {}
        capture_bridge.bind(self)
        publisher_adapter.bind(self)

    @classmethod
    def initialize_fresh(
        cls,
        *,
        registry: base.LayerRegistry,
        episode_id: str,
        layout: ChunkActionLayout,
        ledger: Stage2BOfflineLedger,
        staging_variant: CACHStagingVariant,
        episode_epoch: int = 0,
        synthetic_test_only: bool = False,
    ) -> Stage2BLedgeredHybridCacheManager:
        if not isinstance(ledger, Stage2BOfflineLedger):
            _fail("L2B_LEDGER_MISMATCH", "ledger has wrong type")
        if not isinstance(registry, base.LayerRegistry):
            base._fail("CACHE_SCHEMA_MISMATCH", "registry must be LayerRegistry")
        if not isinstance(layout, ChunkActionLayout):
            base._fail("CACHE_CONTENT_TIME_MISMATCH", "layout has wrong type")
        if not isinstance(staging_variant, CACHStagingVariant):
            base._fail("CACHE_SCHEMA_MISMATCH", "staging_variant has wrong type")
        if type(synthetic_test_only) is not bool:
            base._fail("CACHE_SCHEMA_MISMATCH", "synthetic flag must be bool")
        if not synthetic_test_only:
            _fail(
                "L2B_CPU_SYNTHETIC_ONLY",
                "the L1 snapshot codec and L2A ledger are not production-admitted",
            )

        try:
            # ``scan_valid_chain`` intentionally omits pending intents.  L2B is
            # fresh-only, so pin the reviewed L2A private scan under its own
            # lock and reject every transaction record, including PREPARED.
            with ledger._lock:
                intents, decisions, ordered, pending = ledger._scan_records()
                genesis_manifest_sha256 = (
                    ledger.identity.genesis_snapshot_manifest_sha256
                )
                genesis_manifest = _strict_manifest(
                    ledger.read_object(genesis_manifest_sha256).payload,
                    "genesis snapshot manifest",
                )
                expected_genesis_entries = {
                    _LEDGER_ROOT_FILENAME,
                    ledger.object_filename(genesis_manifest_sha256),
                    *(
                        ledger.object_filename(digest)
                        for digest in _snapshot_blob_digests(genesis_manifest)
                    ),
                }
                observed_entries = {
                    path.name for path in ledger.root.iterdir()
                }
        except Stage2BOfflineLedgerError as exc:
            raise base.CacheContractError(
                "L2B_LEDGER_MISMATCH",
                f"ledger validation failed: {exc.code}",
            ) from exc
        if intents or decisions or ordered or pending:
            _fail(
                "L2B_RECOVERY_REQUIRED",
                "fresh factory refuses a non-genesis ledger root",
            )
        if observed_entries != expected_genesis_entries:
            _fail(
                "L2B_RECOVERY_REQUIRED",
                "fresh factory refuses orphan or non-genesis ledger objects",
            )

        adapter = _LedgerPublisherAdapter()
        bridge = _CaptureBridge()
        manager = _LedgerSeamedStage2BHybridCacheManager(
            capture_bridge=bridge,
            registry=registry,
            episode_id=episode_id,
            layout=layout,
            receipt_publisher=adapter,
            staging_variant=staging_variant,
            episode_epoch=episode_epoch,
            synthetic_test_only=synthetic_test_only,
        )
        genesis = manager.state
        identity = ledger.identity
        if (
            identity.staging_variant is not staging_variant
            or identity.layout_spec_sha256 != layout.layout_spec_sha256
            or identity.layout_instance_digest != layout.layout_instance_digest
            or identity.layer_registry_digest != registry.manifest_digest
            or identity.genesis_state_manifest != genesis.state_manifest_digest
            or ledger._genesis_state.to_manifest() != genesis.to_manifest()
        ):
            _fail(
                "L2B_LEDGER_MISMATCH",
                "manager genesis and caller-pinned ledger identity differ",
            )
        return cls(
            _capability=_CONSTRUCTION_CAPABILITY,
            registry=registry,
            layout=layout,
            ledger=ledger,
            manager=manager,
            capture_bridge=bridge,
            publisher_adapter=adapter,
            synthetic_test_only=synthetic_test_only,
        )

    @classmethod
    def for_synthetic_tests(
        cls,
        *,
        registry: base.LayerRegistry,
        episode_id: str,
        layout: ChunkActionLayout,
        ledger: Stage2BOfflineLedger,
        staging_variant: CACHStagingVariant,
        episode_epoch: int = 0,
    ) -> Stage2BLedgeredHybridCacheManager:
        return cls.initialize_fresh(
            registry=registry,
            episode_id=episode_id,
            layout=layout,
            ledger=ledger,
            staging_variant=staging_variant,
            episode_epoch=episode_epoch,
            synthetic_test_only=True,
        )

    def _poison_locked(self, reason: str) -> None:
        if not self._poisoned:
            self._poison_reason = reason
        self._poisoned = True
        self._phase = _POISONED

    def _assert_ready_locked(self) -> None:
        if self._poisoned:
            _fail(
                "L2B_PUBLICATION_POISONED",
                self._poison_reason or "ledgered manager is poisoned",
            )
        if self._phase != _READY or self._active is not None:
            if self._phase in {
                _TERMINAL_PUBLISHING,
                _TERMINAL_EXACT,
                _CAS_CONFIRMING,
            }:
                self._poison_locked("public re-entry crossed terminal publication")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "public re-entry crossed terminal publication",
                )
            _fail("L2B_TRANSACTION_BUSY", "another ledger transaction is active")

    @property
    def registry(self) -> base.LayerRegistry:
        return self._registry

    @property
    def staging_variant(self) -> CACHStagingVariant:
        return self._manager.staging_variant

    @property
    def state(self) -> stage2b.Stage2BTemporalState:
        with self._lock:
            self._assert_ready_locked()
            return self._manager.state

    def snapshot_for_denoise(
        self,
        *,
        expected_episode_id: str,
        expected_episode_epoch: int,
        expected_revision: int,
        expected_next_chunk_id: int,
    ) -> stage2b.Stage2BDenoiseReadView:
        with self._lock:
            self._assert_ready_locked()
            return self._manager.snapshot_for_denoise(
                expected_episode_id=expected_episode_id,
                expected_episode_epoch=expected_episode_epoch,
                expected_revision=expected_revision,
                expected_next_chunk_id=expected_next_chunk_id,
            )

    def finish_denoise(
        self,
        *,
        read_view: stage2b.Stage2BDenoiseReadView,
        scratch: base.CacheScratch,
    ) -> None:
        with self._lock:
            self._assert_ready_locked()
            self._manager.finish_denoise(read_view=read_view, scratch=scratch)

    def _ledger_position_locked(
        self,
    ) -> tuple[int, str | None]:
        try:
            with self._ledger._lock:
                _, _, ordered, pending = self._ledger._scan_records()
        except Stage2BOfflineLedgerError as exc:
            self._poison_locked(f"ledger scan failed: {exc.code}")
            raise base.CacheContractError(
                "L2B_LEDGER_MISMATCH",
                f"ledger scan failed: {exc.code}",
            ) from exc
        if pending:
            self._poison_locked("unexpected PREPARED intent exists")
            _fail(
                "L2B_RECOVERY_REQUIRED",
                "unexpected PREPARED intent requires recovery",
            )
        return len(ordered), None if not ordered else ordered[-1].sha256

    def _existing_terminal_locked(self, transaction_id: str) -> None:
        try:
            existing = self._ledger.read_decision(transaction_id)
        except Stage2BOfflineLedgerError as exc:
            self._poison_locked(f"terminal lookup failed: {exc.code}")
            raise base.CacheContractError(
                "L2B_LEDGER_MISMATCH",
                f"terminal lookup failed: {exc.code}",
            ) from exc
        if existing is not None:
            _fail(
                "COMMIT_DUPLICATE_CONFLICT",
                "transaction ID already has a terminal ledger decision",
            )

    def _reconcile_completed_locked(
        self,
        *,
        transaction_id: str,
        transaction_kind: Stage2BTransactionKind,
        terminal_sha256: str,
        receipt_sha256: str,
    ) -> None:
        """Require a fresh stable read before returning cached success."""

        try:
            reconciliation = self._ledger.reconcile_terminal(
                transaction_id=transaction_id,
                expected_record_sha256=terminal_sha256,
            )
        except Exception as exc:
            self._poison_locked(
                f"completed terminal reconciliation failed: {type(exc).__name__}"
            )
            raise base.CacheContractError(
                "L2B_PUBLICATION_POISONED",
                "completed terminal no longer reconciles exactly",
            ) from exc
        stored = reconciliation.stored
        if (
            reconciliation.status is not Stage2BReconcileStatus.EXACT
            or stored is None
            or stored.sha256 != terminal_sha256
            or stored.decision.decision_body.transaction_id != transaction_id
            or stored.decision.decision_body.transaction_kind is not transaction_kind
            or stored.decision.decision_body.result
            is not Stage2BDecisionResult.COMMITTED
            or stored.decision.decision_body.success_or_abort_receipt_sha256
            != receipt_sha256
        ):
            self._poison_locked("completed terminal exact binding differs")
            _fail(
                "L2B_PUBLICATION_POISONED",
                "completed terminal exact binding differs",
            )

    def _publish_prepared_locked(self, active: _ActiveTransaction) -> None:
        self._active = active
        self._phase = _PREPARING
        try:
            stored_identity = self._ledger.publish_operation_identity(
                identity=active.operation_identity,
                operation_payload=active.operation_payload,
            )
            if stored_identity.sha256 != active.intent.input_identity_sha256:
                _fail(
                    "L2B_LEDGER_MISMATCH",
                    "stored operation identity differs from PREPARED intent",
                )
            stored_intent = self._ledger.prepare(active.intent)
            if stored_intent.payload != active.intent.canonical_bytes:
                _fail(
                    "L2B_LEDGER_MISMATCH",
                    "stored PREPARED bytes differ from candidate",
                )
        except Exception as exc:
            self._poison_locked(
                f"PREPARED publication failed: {type(exc).__name__}"
            )
            if isinstance(exc, base.CacheContractError):
                raise
            raise base.CacheContractError(
                "L2B_LEDGER_FAILURE",
                f"PREPARED publication failed: {type(exc).__name__}",
            ) from exc
        try:
            self._publisher_adapter.arm(active.token)
        except Exception as exc:
            self._poison_locked(
                f"publisher adapter arm failed: {type(exc).__name__}"
            )
            raise
        self._phase = _PREPARED

    def _paired_operation(
        self,
        *,
        request: stage2b.Stage2BTeacherForcingCommitRequest,
        prepared: object,
        committed_action_span_digest: str,
    ) -> bytes:
        return base._canonical_json_bytes(
            {
                "committed_action_span_digest": committed_action_span_digest,
                "conditioning_digest": request.conditioning_digest,
                "content_time_sha256": _sha256(
                    base._canonical_json_bytes(request.content_time.to_manifest())
                ),
                "dataset_manifest_sha256": request.proof.dataset_manifest_sha256,
                "dataset_row_identity": request.proof.dataset_row_identity,
                "episode_epoch_before": request.expected_episode_epoch,
                "episode_id_before": request.expected_episode_id,
                "layout_instance_digest": request.content_time.layout_instance_digest,
                "layout_spec_sha256": request.content_time.layout_spec_sha256,
                "pair_action_digest": prepared.action_digest,
                "pair_action_mask_digest": prepared.action_mask_digest,
                "pair_frame_valid_mask_digest": prepared.frame_valid_mask_digest,
                "pair_video_digest": prepared.video_digest,
                "revision_before": request.expected_revision,
                "schema": _PAIRED_OPERATION_SCHEMA,
                "source_proof_digest": prepared.source_proof_digest,
                "teacher_pair_payload_digest": prepared.paired_payload_digest,
                "transaction_id": request.commit_id,
                "transaction_kind": Stage2BTransactionKind.PAIRED_COMMIT.value,
                "transaction_nonce": request.transaction_nonce,
            }
        )

    def commit_paired(
        self,
        *,
        commit_source: base.CommitSource | str,
        request: stage2b.Stage2BTeacherForcingCommitRequest | None,
        stage_callback: Callable[
            [stage2b.Stage2BPairedStagingContext],
            tuple[base.StagedLayerPayload, ...],
        ]
        | None,
    ) -> stage2b.Stage2BPairedCommitReceipt:
        # POISONED is absorbing and a live transaction wins over all argument
        # diagnostics.  Repeat this check after pure preflight to close the
        # race between two callers that both observed READY.
        with self._lock:
            self._assert_ready_locked()
        authorized_request, authorized_delegate = _authorize_paired_commit(
            commit_source=commit_source,
            request=request,
            stage_callback=stage_callback,
            synthetic_test_only=self._synthetic_test_only,
        )

        with self._lock:
            self._assert_ready_locked()
            layout = self._layout
            content_time = authorized_request.content_time
            if (
                content_time.layout_spec_sha256 != layout.layout_spec_sha256
                or content_time.layout_instance_digest
                != layout.layout_instance_digest
                or content_time.chunk_id >= len(layout.chunks)
            ):
                base._fail(
                    "CACHE_CONTENT_TIME_MISMATCH",
                    "teacher pair does not name a manager-bound chunk",
                )
            chunk = layout.chunks[content_time.chunk_id]
            content_time.verify_layout_binding(layout=layout, chunk=chunk)
            prepared = base._prepare_teacher_pair(authorized_request, chunk=chunk)
            if any(
                tensor.device != torch.device("cpu")
                for tensor in (
                    prepared.video,
                    prepared.frame_valid_mask,
                    prepared.actions,
                    prepared.action_valid_mask,
                )
            ):
                _fail(
                    "L2B_CPU_SYNTHETIC_ONLY",
                    "L2B paired inputs must be on exact logical device cpu",
                )
            # The durable operation is derived from these detached clones.  Bind
            # the inherited second preflight to the same tensors so caller-side
            # mutation while PREPARED is being published cannot change the
            # transaction observed by the numerical stager.
            bound_request = replace(
                authorized_request,
                video=prepared.video,
                frame_valid_mask=prepared.frame_valid_mask,
                actions=prepared.actions,
                action_valid_mask=prepared.action_valid_mask,
            )
            valid_count = (
                content_time.action_token_end_exclusive
                - content_time.action_token_start
            )
            committed_span_digest = committed_action_tensor_digest(
                prepared.actions[:, :valid_count, :]
                .detach()
                .clone(memory_format=torch.contiguous_format)
            )
            operation_payload = self._paired_operation(
                request=authorized_request,
                prepared=prepared,
                committed_action_span_digest=committed_span_digest,
            )
            operation_payload_sha256 = _sha256(operation_payload)

            completed = self._completed_commits.get(authorized_request.commit_id)
            if completed is not None:
                prior_digest, prior_receipt, terminal_sha256 = completed
                if prior_digest != operation_payload_sha256:
                    base._fail(
                        "COMMIT_DUPLICATE_CONFLICT",
                        "commit_id was reused for different L2B inputs",
                    )
                self._reconcile_completed_locked(
                    transaction_id=authorized_request.commit_id,
                    transaction_kind=Stage2BTransactionKind.PAIRED_COMMIT,
                    terminal_sha256=terminal_sha256,
                    receipt_sha256=prior_receipt.receipt_sha256,
                )
                live = self._manager.state
                if (
                    live.episode_id != prior_receipt.episode_id
                    or live.episode_epoch != prior_receipt.episode_epoch
                    or live.revision < prior_receipt.revision_after
                ):
                    base._fail(
                        "COMMIT_STALE_REVISION",
                        "completed commit belongs to a stale episode",
                    )
                return prior_receipt
            if authorized_request.commit_id in self._completed_resets:
                base._fail(
                    "COMMIT_DUPLICATE_CONFLICT",
                    "transaction ID was already used by a reset",
                )
            self._existing_terminal_locked(authorized_request.commit_id)

            base_state = self._manager.state
            if (
                authorized_request.expected_episode_id != base_state.episode_id
                or authorized_request.expected_episode_epoch
                != base_state.episode_epoch
                or authorized_request.expected_revision != base_state.revision
            ):
                base._fail("COMMIT_STALE_REVISION", "teacher request is stale")
            self._manager._validate_next_content_time(base_state, content_time)
            try:
                if base_state.committed_action_history is None:
                    base_history = CommittedActionHistory.empty(
                        episode_id=base_state.episode_id,
                        episode_epoch=base_state.episode_epoch,
                        layout_spec_sha256=base_state.layout_spec_sha256,
                        layout_instance_digest=base_state.layout_instance_digest,
                        batch_size=prepared.actions.shape[0],
                        action_dim=prepared.actions.shape[2],
                        dtype=prepared.actions.dtype,
                        device=prepared.actions.device,
                    )
                else:
                    base_history = base_state.committed_action_history
                base_history.view_for_chunk(
                    chunk_id=content_time.chunk_id,
                    action_start=content_time.action_token_start,
                )
            except CommittedActionHistoryError as exc:
                stage2b._translate_history_error(exc)
                raise AssertionError("unreachable")

            transaction_payload_digest = _sha256(
                base._canonical_json_bytes(
                    {
                        "conditioning_digest": authorized_request.conditioning_digest,
                        "schema": _PAIRED_PAYLOAD_SCHEMA,
                        "staging_variant": self.staging_variant.value,
                        "teacher_pair_payload_digest": prepared.paired_payload_digest,
                    }
                )
            )
            sequence, previous = self._ledger_position_locked()
            identity = Stage2BOperationIdentity(
                ledger_id=self._ledger.identity.ledger_id,
                transaction_id=authorized_request.commit_id,
                transaction_kind=Stage2BTransactionKind.PAIRED_COMMIT,
                transaction_nonce=authorized_request.transaction_nonce,
                staging_variant=self.staging_variant,
                episode_id_before=base_state.episode_id,
                episode_epoch_before=base_state.episode_epoch,
                revision_before=base_state.revision,
                state_manifest_before=base_state.state_manifest_digest,
                layout_spec_sha256=layout.layout_spec_sha256,
                layout_instance_digest=layout.layout_instance_digest,
                operation_payload_sha256=operation_payload_sha256,
            )
            intent = Stage2BLedgerIntent(
                ledger_id=self._ledger.identity.ledger_id,
                ledger_sequence=sequence,
                transaction_id=authorized_request.commit_id,
                transaction_kind=Stage2BTransactionKind.PAIRED_COMMIT,
                transaction_nonce=authorized_request.transaction_nonce,
                writer_fence_id=self._ledger.identity.writer_fence_id,
                staging_variant=self.staging_variant,
                episode_id_before=base_state.episode_id,
                episode_epoch_before=base_state.episode_epoch,
                revision_before=base_state.revision,
                state_manifest_before=base_state.state_manifest_digest,
                input_identity_sha256=identity.identity_sha256,
                previous_decision_sha256=previous,
            )
            active = _ActiveTransaction(
                token=object(),
                transaction_kind=Stage2BTransactionKind.PAIRED_COMMIT,
                transaction_id=authorized_request.commit_id,
                retry_digest=operation_payload_sha256,
                operation_identity=identity,
                operation_payload=operation_payload,
                intent=intent,
                base_state=base_state,
                target_layout=layout,
                transaction_payload_digest=transaction_payload_digest,
                prepared_pair=prepared,
                base_history=base_history,
            )
            self._publish_prepared_locked(active)

        try:
            receipt = self._manager.commit_paired(
                commit_source=commit_source,
                request=bound_request,
                stage_callback=authorized_delegate,
            )
        except Exception as exc:
            replacement = self._finalize_preterminal_failure(active.token)
            self._publisher_adapter.disarm(active.token)
            if replacement is not None:
                raise replacement from exc
            raise
        try:
            return self._confirm_commit(active.token, receipt)
        finally:
            self._publisher_adapter.disarm(active.token)

    def _capture_staging(
        self,
        *,
        token: object,
        delegate: Callable[
            [stage2b.Stage2BPairedStagingContext],
            tuple[base.StagedLayerPayload, ...],
        ],
        context: stage2b.Stage2BPairedStagingContext,
    ) -> tuple[base.StagedLayerPayload, ...]:
        with self._lock:
            active = self._active
            if (
                active is None
                or active.token is not token
                or active.transaction_kind
                is not Stage2BTransactionKind.PAIRED_COMMIT
                or self._phase != _PREPARED
            ):
                self._poison_locked("capturing callback token/phase differs")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "capturing callback token/phase differs",
                )
            self._phase = _STAGING

        raw_payloads = delegate(context)
        if type(raw_payloads) is not tuple:
            base._fail(
                "CACHE_SCHEMA_MISMATCH",
                "Stage-2B stager must return an eagerly materialized exact tuple",
            )
        assert active.prepared_pair is not None
        assert active.base_history is not None
        assert active.transaction_payload_digest is not None
        prepared = active.prepared_pair
        layer_states = base._materialize_layer_states(
            registry=self._registry,
            content_time=context.content_time,
            payloads=raw_payloads,
        )
        try:
            staged_history, duplicate = active.base_history.append(
                chunk_id=context.content_time.chunk_id,
                action_start=context.content_time.action_token_start,
                action_end_exclusive=(
                    context.content_time.action_token_end_exclusive
                ),
                fixed_slot_actions=prepared.actions,
                action_valid_mask=prepared.action_valid_mask,
                commit_source=base.CommitSource.TEACHER_FORCING,
                source_proof_digest=prepared.source_proof_digest,
                commit_id=active.transaction_id,
            )
        except CommittedActionHistoryError as exc:
            stage2b._translate_history_error(exc)
            raise AssertionError("unreachable")
        if duplicate:
            base._fail(
                "ACTION_HISTORY_DUPLICATE_CONFLICT",
                "new transaction unexpectedly reused committed history",
            )
        base_state = active.base_state
        inner_state = base.HybridTemporalState(
            episode_id=base_state.episode_id,
            episode_epoch=base_state.episode_epoch,
            revision=base_state.revision + 1,
            committed_through_chunk=context.content_time.chunk_id,
            next_chunk_id=context.content_time.chunk_id + 1,
            action_cursor=context.content_time.action_token_end_exclusive,
            layout_spec_sha256=base_state.layout_spec_sha256,
            layout_instance_digest=base_state.layout_instance_digest,
            layer_registry_digest=self._registry.manifest_digest,
            layer_states=layer_states,
            committed_pair_digest=active.transaction_payload_digest,
            last_commit_source=base.CommitSource.TEACHER_FORCING,
            last_source_proof_digest=prepared.source_proof_digest,
            pair_evidence_mode="dataset_ground_truth",
        )
        staged_state = stage2b.Stage2BTemporalState(
            cache_state=inner_state,
            staging_variant=self.staging_variant,
            committed_action_history=staged_history,
        )
        snapshot = encode_stage2b_state_snapshot(
            state=staged_state,
            expected_layout=active.target_layout,
            expected_registry=self._registry,
        )
        if not isinstance(snapshot, EncodedStage2BStateSnapshot):
            base._fail("CACHE_SCHEMA_MISMATCH", "L1 encoder returned wrong type")
        with self._lock:
            if (
                self._active is not active
                or self._phase != _STAGING
                or self._poisoned
            ):
                self._poison_locked("capturing result raced active transaction")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "capturing result raced active transaction",
                )
            active.staged_state = staged_state
            active.snapshot = snapshot
            self._phase = _STAGED
        return raw_payloads

    def _candidate_terminal(
        self,
        *,
        active: _ActiveTransaction,
        receipt_sha256: str,
    ) -> Stage2BDecisionRecord:
        staged_state = active.staged_state
        snapshot = active.snapshot
        assert staged_state is not None
        assert snapshot is not None
        return Stage2BDecisionRecord.from_body(
            Stage2BDecisionBody(
                ledger_id=active.intent.ledger_id,
                ledger_sequence=active.intent.ledger_sequence,
                transaction_id=active.transaction_id,
                transaction_kind=active.transaction_kind,
                intent_sha256=active.intent.intent_sha256,
                result=Stage2BDecisionResult.COMMITTED,
                writer_fence_id=active.intent.writer_fence_id,
                staging_variant=active.intent.staging_variant,
                episode_id_before=active.base_state.episode_id,
                episode_epoch_before=active.base_state.episode_epoch,
                revision_before=active.base_state.revision,
                episode_id_after=staged_state.episode_id,
                episode_epoch_after=staged_state.episode_epoch,
                revision_after=staged_state.revision,
                state_manifest_before=active.base_state.state_manifest_digest,
                state_manifest_after=staged_state.state_manifest_digest,
                snapshot_manifest_sha256=snapshot.manifest_sha256,
                success_or_abort_receipt_sha256=receipt_sha256,
                previous_decision_sha256=active.intent.previous_decision_sha256,
                abort_code=None,
            )
        )

    def _publish_terminal(
        self,
        *,
        token: object,
        receipt_id: str,
        payload: bytes,
        expected_sha256: str,
    ) -> base.ReceiptPublication:
        with self._lock:
            active = self._active
            if active is None or active.token is not token:
                self._poison_locked("publisher token differs")
                _fail("L2B_PUBLICATION_POISONED", "publisher token differs")
            active.publisher_entered = True
            active.publisher_calls += 1
            if active.publisher_calls != 1 or self._phase != _STAGED:
                self._poison_locked("publisher re-entry or phase mismatch")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "publisher re-entry or phase mismatch",
                )
            self._phase = _TERMINAL_PUBLISHING

            if (
                receipt_id != active.transaction_id
                or type(payload) is not bytes
                or expected_sha256 != _sha256(payload)
            ):
                self._poison_locked("receipt ID/bytes/SHA256 differ")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "receipt ID/bytes/SHA256 differ",
                )
            try:
                receipt_manifest, receipt_schema = l0_receipts._validate_payload(
                    payload,
                    receipt_id,
                )
            except Exception as exc:
                self._poison_locked("receipt failed frozen L0 validation")
                raise base.CacheContractError(
                    "L2B_PUBLICATION_POISONED",
                    "receipt failed frozen L0 validation",
                ) from exc
            staged_state = active.staged_state
            snapshot = active.snapshot
            if staged_state is None or snapshot is None:
                self._poison_locked("publisher lacks detached staged state")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "publisher lacks detached staged state",
                )
            snapshot_manifest = _strict_manifest(
                snapshot.manifest_bytes,
                "detached snapshot manifest",
            )
            expected_receipt_schema = (
                "cach.stage2b.paired_commit_receipt.v1"
                if active.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT
                else "cach.stage2b.cache_reset_receipt.v1"
            )
            mismatch = (
                receipt_schema != expected_receipt_schema
                or receipt_manifest.get("schema") != expected_receipt_schema
                or receipt_manifest.get("state_manifest_before")
                != active.base_state.state_manifest_digest
                or receipt_manifest.get("state_manifest_after")
                != staged_state.state_manifest_digest
                or snapshot_manifest.get("state_manifest_digest")
                != staged_state.state_manifest_digest
            )
            if active.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT:
                mismatch = mismatch or (
                    receipt_manifest.get("staged_state_manifest")
                    != staged_state.state_manifest_digest
                )
            else:
                mismatch = mismatch or (
                    receipt_manifest.get("aborted_pending_commit_id") is not None
                )
            if mismatch:
                self._poison_locked("receipt/snapshot/state binding differs")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "receipt/snapshot/state binding differs",
                )
            active.receipt_payload = payload
            active.receipt_sha256 = expected_sha256
            candidate = self._candidate_terminal(
                active=active,
                receipt_sha256=expected_sha256,
            )

        stored: StoredStage2BDecision | None = None
        commit_error: Exception | None = None
        try:
            stored = self._ledger.commit(
                intent=active.intent,
                snapshot=snapshot,
                success_receipt_bytes=payload,
                episode_id_after=staged_state.episode_id,
                episode_epoch_after=staged_state.episode_epoch,
                revision_after=staged_state.revision,
                state_manifest_after=staged_state.state_manifest_digest,
            )
        except Stage2BOfflineLedgerError as exc:
            if exc.code != "LEDGER_PUBLICATION_OUTCOME_UNKNOWN":
                with self._lock:
                    self._poison_locked(
                        f"terminal validation failed: {exc.code}"
                    )
                raise base.CacheContractError(
                    "L2B_PUBLICATION_POISONED",
                    f"terminal validation failed: {exc.code}",
                ) from exc
            commit_error = exc
        except Exception as exc:
            # The pinned ledger reports its deterministic validation failures
            # with Stage2BOfflineLedgerError.  An untyped exception crossing
            # this publication call has no trustworthy outcome (and is also
            # how focused fault injection represents a post-rename failure).
            commit_error = exc

        if stored is None:
            try:
                reconciliation = self._ledger.reconcile_terminal(
                    transaction_id=active.transaction_id,
                    expected_record_sha256=candidate.decision_sha256,
                )
            except Exception as exc:
                with self._lock:
                    self._poison_locked(
                        f"terminal reconciliation failed: {type(exc).__name__}"
                    )
                raise base.CacheContractError(
                    "L2B_PUBLICATION_POISONED",
                    "terminal publication did not reconcile exactly",
                ) from (commit_error or exc)
            if reconciliation.status is Stage2BReconcileStatus.ABSENT:
                try:
                    self._ledger.abort_after_reconciled_absence(
                        intent=active.intent,
                        expected_terminal_record_sha256=candidate.decision_sha256,
                    )
                except Exception as exc:
                    with self._lock:
                        self._poison_locked(
                            f"reconciled-absence abort failed: {type(exc).__name__}"
                        )
                    raise base.CacheContractError(
                        "L2B_PUBLICATION_POISONED",
                        "stable absence could not publish its registered abort",
                    ) from (commit_error or exc)
                with self._lock:
                    self._poison_locked("terminal candidate stayed stably absent")
                raise base.CacheContractError(
                    "L2B_PUBLICATION_POISONED",
                    "terminal candidate stayed stably absent",
                ) from commit_error
            stored = reconciliation.stored

        assert stored is not None
        try:
            exact = self._ledger.reconcile_terminal(
                transaction_id=active.transaction_id,
                expected_record_sha256=candidate.decision_sha256,
            )
        except Exception as exc:
            with self._lock:
                self._poison_locked(
                    f"terminal stable read failed: {type(exc).__name__}"
                )
            raise base.CacheContractError(
                "L2B_PUBLICATION_POISONED",
                "terminal decision lacks exact stable read-back",
            ) from exc
        with self._lock:
            if (
                self._poisoned
                or self._active is not active
                or self._phase != _TERMINAL_PUBLISHING
                or exact.status is not Stage2BReconcileStatus.EXACT
                or exact.stored is None
                or exact.stored.sha256 != candidate.decision_sha256
                or stored.sha256 != candidate.decision_sha256
                or exact.stored.decision.decision_body.result
                is not Stage2BDecisionResult.COMMITTED
            ):
                self._poison_locked("terminal exact binding changed")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "terminal exact binding changed",
                )
            active.terminal = exact.stored
            self._phase = _TERMINAL_EXACT
            return base.ReceiptPublication(
                receipt_id=receipt_id,
                sha256=expected_sha256,
                durable=True,
                readback_verified=True,
            )

    def _finalize_preterminal_failure(
        self,
        token: object,
    ) -> base.CacheContractError | None:
        with self._lock:
            active = self._active
            if active is None or active.token is not token:
                self._poison_locked("failed transaction lost its active token")
                return base.CacheContractError(
                    "L2B_PUBLICATION_POISONED",
                    "failed transaction lost its active token",
                )
            if self._poisoned or active.publisher_entered or active.terminal is not None:
                self._poison_locked(
                    "manager failed after entering terminal publication"
                )
                return None
            try:
                live = self._manager.state
            except Exception as exc:
                self._poison_locked(
                    f"manager state unreadable after failure: {type(exc).__name__}"
                )
                return base.CacheContractError(
                    "L2B_PUBLICATION_POISONED",
                    "manager state unreadable after failure",
                )
            if live.state_manifest_digest != active.base_state.state_manifest_digest:
                self._poison_locked("preterminal failure advanced manager state")
                return base.CacheContractError(
                    "L2B_PUBLICATION_POISONED",
                    "preterminal failure advanced manager state",
                )
            self._phase = _ABORTING

        try:
            abort_code = (
                Stage2BAbortCode.STAGING_FAILED
                if active.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT
                else Stage2BAbortCode.PRECONDITION_FAILED
            )
            failure_point = (
                Stage2BFailurePoint.STAGING
                if active.transaction_kind is Stage2BTransactionKind.PAIRED_COMMIT
                else Stage2BFailurePoint.AFTER_PREPARED
            )
            stored = self._ledger.abort(
                intent=active.intent,
                abort_code=abort_code,
                failure_point=failure_point,
            )
            exact = self._ledger.reconcile_terminal(
                transaction_id=active.transaction_id,
                expected_record_sha256=stored.sha256,
            )
        except Exception as exc:
            with self._lock:
                self._poison_locked(
                    f"staging abort failed: {type(exc).__name__}"
                )
            return base.CacheContractError(
                "L2B_LEDGER_FAILURE",
                f"staging abort failed: {type(exc).__name__}",
            )
        body = stored.decision.decision_body
        with self._lock:
            if (
                self._active is not active
                or self._poisoned
                or exact.status is not Stage2BReconcileStatus.EXACT
                or exact.stored is None
                or exact.stored.sha256 != stored.sha256
                or body.result is not Stage2BDecisionResult.ABORTED
                or body.abort_code is not abort_code
                or body.snapshot_manifest_sha256 is not None
                or body.state_manifest_after
                != active.base_state.state_manifest_digest
            ):
                self._poison_locked("staging abort exact binding differs")
                return base.CacheContractError(
                    "L2B_PUBLICATION_POISONED",
                    "staging abort exact binding differs",
                )
            self._publisher_adapter.disarm(active.token)
            self._active = None
            self._phase = _READY
            return None

    def _confirm_commit(
        self,
        token: object,
        receipt: stage2b.Stage2BPairedCommitReceipt,
    ) -> stage2b.Stage2BPairedCommitReceipt:
        with self._lock:
            active = self._active
            if not isinstance(receipt, stage2b.Stage2BPairedCommitReceipt):
                self._poison_locked("commit returned a non-Stage2B receipt")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "commit returned a non-Stage2B receipt",
                )
            if (
                active is None
                or active.token is not token
                or self._poisoned
                or self._phase != _TERMINAL_EXACT
                or active.terminal is None
                or active.staged_state is None
                or active.receipt_payload != receipt.canonical_bytes
                or active.receipt_sha256 != receipt.receipt_sha256
            ):
                self._poison_locked("commit return differs from exact terminal")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "commit return differs from exact terminal",
                )
            self._phase = _CAS_CONFIRMING
            try:
                live = self._manager.state
            except Exception as exc:
                self._poison_locked(
                    f"manager state unreadable after CAS: {type(exc).__name__}"
                )
                raise base.CacheContractError(
                    "L2B_PUBLICATION_POISONED",
                    "manager state unreadable after CAS",
                ) from exc
            body = active.terminal.decision.decision_body
            if (
                live.state_manifest_digest
                != active.staged_state.state_manifest_digest
                or live.to_manifest() != active.staged_state.to_manifest()
                or body.state_manifest_after != live.state_manifest_digest
                or body.success_or_abort_receipt_sha256 != receipt.receipt_sha256
            ):
                self._poison_locked("post-CAS state/terminal binding differs")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "post-CAS state/terminal binding differs",
                )
            self._completed_commits[active.transaction_id] = (
                active.retry_digest,
                receipt,
                active.terminal.sha256,
            )
            self._publisher_adapter.disarm(active.token)
            self._active = None
            self._phase = _READY
            return receipt

    def _reset_retry_digest(
        self,
        *,
        reset_id: str,
        transaction_nonce: str,
        new_episode_id: str,
        layout: ChunkActionLayout,
    ) -> str:
        return _sha256(
            base._canonical_json_bytes(
                {
                    "layout_instance_digest": layout.layout_instance_digest,
                    "layout_spec_sha256": layout.layout_spec_sha256,
                    "new_episode_id": new_episode_id,
                    "reset_id": reset_id,
                    "schema": "cach.stage2b.cache_reset_receipt.v1",
                    "staging_variant": self.staging_variant.value,
                    "transaction_nonce": transaction_nonce,
                }
            )
        )

    def reset(
        self,
        *,
        reset_id: str,
        transaction_nonce: str,
        new_episode_id: str,
        layout: ChunkActionLayout,
    ) -> stage2b.Stage2BCacheResetReceipt:
        with self._lock:
            self._assert_ready_locked()
        base._require_non_empty_string(reset_id, "reset_id")
        base._require_non_empty_string(transaction_nonce, "transaction_nonce")
        base._require_non_empty_string(new_episode_id, "new_episode_id")
        if not isinstance(layout, ChunkActionLayout):
            base._fail("CACHE_CONTENT_TIME_MISMATCH", "reset layout has wrong type")
        if layout.synthetic_test_only != self._synthetic_test_only:
            base._fail(
                "CACHE_SYNTHETIC_LAYOUT_PROVENANCE",
                "reset cannot switch synthetic provenance",
            )
        with self._lock:
            self._assert_ready_locked()
            retry_digest = self._reset_retry_digest(
                reset_id=reset_id,
                transaction_nonce=transaction_nonce,
                new_episode_id=new_episode_id,
                layout=layout,
            )
            completed = self._completed_resets.get(reset_id)
            if completed is not None:
                prior_digest, prior_receipt, terminal_sha256 = completed
                if prior_digest != retry_digest:
                    base._fail(
                        "COMMIT_DUPLICATE_CONFLICT",
                        "reset_id was reused for different inputs",
                    )
                self._reconcile_completed_locked(
                    transaction_id=reset_id,
                    transaction_kind=Stage2BTransactionKind.RESET,
                    terminal_sha256=terminal_sha256,
                    receipt_sha256=prior_receipt.receipt_sha256,
                )
                if (
                    self._manager.state.state_manifest_digest
                    != prior_receipt.state_manifest_after
                ):
                    base._fail("COMMIT_STALE_REVISION", "completed reset is stale")
                return prior_receipt
            if reset_id in self._completed_commits:
                base._fail(
                    "COMMIT_DUPLICATE_CONFLICT",
                    "transaction ID was already used by a paired commit",
                )
            self._existing_terminal_locked(reset_id)

            before = self._manager.state
            target_state = stage2b.Stage2BTemporalState.empty(
                episode_id=new_episode_id,
                episode_epoch=before.episode_epoch + 1,
                layout=layout,
                registry=self._registry,
                staging_variant=self.staging_variant,
            )
            snapshot = encode_stage2b_state_snapshot(
                state=target_state,
                expected_layout=layout,
                expected_registry=self._registry,
            )
            operation_payload = base._canonical_json_bytes(
                {
                    "action_cursor_before": before.action_cursor,
                    "action_history_digest_before": _history_digest(before),
                    "episode_epoch_before": before.episode_epoch,
                    "episode_id_before": before.episode_id,
                    "layout_instance_digest": layout.layout_instance_digest,
                    "layout_spec_sha256": layout.layout_spec_sha256,
                    "new_episode_epoch": target_state.episode_epoch,
                    "new_episode_id": target_state.episode_id,
                    "revision_before": before.revision,
                    "schema": _RESET_OPERATION_SCHEMA,
                    "staging_variant": self.staging_variant.value,
                    "state_manifest_before": before.state_manifest_digest,
                    "transaction_id": reset_id,
                    "transaction_kind": Stage2BTransactionKind.RESET.value,
                    "transaction_nonce": transaction_nonce,
                }
            )
            operation_sha256 = _sha256(operation_payload)
            sequence, previous = self._ledger_position_locked()
            identity = Stage2BOperationIdentity(
                ledger_id=self._ledger.identity.ledger_id,
                transaction_id=reset_id,
                transaction_kind=Stage2BTransactionKind.RESET,
                transaction_nonce=transaction_nonce,
                staging_variant=self.staging_variant,
                episode_id_before=before.episode_id,
                episode_epoch_before=before.episode_epoch,
                revision_before=before.revision,
                state_manifest_before=before.state_manifest_digest,
                layout_spec_sha256=layout.layout_spec_sha256,
                layout_instance_digest=layout.layout_instance_digest,
                operation_payload_sha256=operation_sha256,
            )
            intent = Stage2BLedgerIntent(
                ledger_id=self._ledger.identity.ledger_id,
                ledger_sequence=sequence,
                transaction_id=reset_id,
                transaction_kind=Stage2BTransactionKind.RESET,
                transaction_nonce=transaction_nonce,
                writer_fence_id=self._ledger.identity.writer_fence_id,
                staging_variant=self.staging_variant,
                episode_id_before=before.episode_id,
                episode_epoch_before=before.episode_epoch,
                revision_before=before.revision,
                state_manifest_before=before.state_manifest_digest,
                input_identity_sha256=identity.identity_sha256,
                previous_decision_sha256=previous,
            )
            active = _ActiveTransaction(
                token=object(),
                transaction_kind=Stage2BTransactionKind.RESET,
                transaction_id=reset_id,
                retry_digest=retry_digest,
                operation_identity=identity,
                operation_payload=operation_payload,
                intent=intent,
                base_state=before,
                target_layout=layout,
                staged_state=target_state,
                snapshot=snapshot,
            )
            self._publish_prepared_locked(active)
            self._phase = _STAGED

        try:
            receipt = self._manager.reset(
                reset_id=reset_id,
                transaction_nonce=transaction_nonce,
                new_episode_id=new_episode_id,
                layout=layout,
            )
        except Exception as exc:
            replacement = self._finalize_preterminal_failure(active.token)
            self._publisher_adapter.disarm(active.token)
            if replacement is not None:
                raise replacement from exc
            raise
        try:
            return self._confirm_reset(active.token, receipt)
        finally:
            self._publisher_adapter.disarm(active.token)

    def _confirm_reset(
        self,
        token: object,
        receipt: stage2b.Stage2BCacheResetReceipt,
    ) -> stage2b.Stage2BCacheResetReceipt:
        with self._lock:
            active = self._active
            if not isinstance(receipt, stage2b.Stage2BCacheResetReceipt):
                self._poison_locked("reset returned a non-Stage2B receipt")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "reset returned a non-Stage2B receipt",
                )
            if (
                active is None
                or active.token is not token
                or self._poisoned
                or self._phase != _TERMINAL_EXACT
                or active.terminal is None
                or active.staged_state is None
                or active.receipt_payload != receipt.canonical_bytes
                or active.receipt_sha256 != receipt.receipt_sha256
            ):
                self._poison_locked("reset return differs from exact terminal")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "reset return differs from exact terminal",
                )
            self._phase = _CAS_CONFIRMING
            try:
                live = self._manager.state
            except Exception as exc:
                self._poison_locked(
                    f"manager state unreadable after reset CAS: {type(exc).__name__}"
                )
                raise base.CacheContractError(
                    "L2B_PUBLICATION_POISONED",
                    "manager state unreadable after reset CAS",
                ) from exc
            body = active.terminal.decision.decision_body
            if (
                live.state_manifest_digest
                != active.staged_state.state_manifest_digest
                or live.to_manifest() != active.staged_state.to_manifest()
                or body.state_manifest_after != live.state_manifest_digest
                or body.success_or_abort_receipt_sha256 != receipt.receipt_sha256
                or receipt.aborted_pending_commit_id is not None
            ):
                self._poison_locked("post-reset-CAS binding differs")
                _fail(
                    "L2B_PUBLICATION_POISONED",
                    "post-reset-CAS binding differs",
                )
            self._layout = active.target_layout
            self._completed_resets[active.transaction_id] = (
                active.retry_digest,
                receipt,
                active.terminal.sha256,
            )
            self._publisher_adapter.disarm(active.token)
            self._active = None
            self._phase = _READY
            return receipt


CacheContractError = base.CacheContractError
CommitSource = base.CommitSource
ContentTime = base.ContentTime
LayerKind = base.LayerKind
LayerRegistry = base.LayerRegistry
LayerSpec = base.LayerSpec
StagedLayerPayload = base.StagedLayerPayload
TeacherForcingDatasetPairProof = base.TeacherForcingDatasetPairProof


__all__ = [
    "CACHStagingVariant",
    "CacheContractError",
    "CommitSource",
    "ContentTime",
    "LayerKind",
    "LayerRegistry",
    "LayerSpec",
    "Stage2BLedgeredHybridCacheManager",
    "StagedLayerPayload",
    "TeacherForcingDatasetPairProof",
    "encode_stage2b_state_snapshot",
]
