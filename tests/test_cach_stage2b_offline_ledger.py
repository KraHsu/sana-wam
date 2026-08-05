"""CPU/synthetic-only contract tests for the Stage-2B L2A offline ledger."""

from __future__ import annotations

from dataclasses import dataclass, replace
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import threading

import pytest
import torch

from sana_wam.cach import stage2b_offline_ledger as ledger_module
from sana_wam.cach.stage2b_offline_ledger import (
    Stage2BAbortCode,
    Stage2BAbortReceipt,
    Stage2BDecisionBody,
    Stage2BDecisionRecord,
    Stage2BDecisionResult,
    Stage2BFailurePoint,
    Stage2BLedgerIntent,
    Stage2BOfflineLedger,
    Stage2BOfflineLedgerConflictError,
    Stage2BOfflineLedgerError,
    Stage2BOperationIdentity,
    Stage2BReconcileStatus,
    Stage2BTransactionKind,
)
from sana_wam.cach.stage2b_state_snapshot import (
    EncodedStage2BStateSnapshot,
    encode_stage2b_state_snapshot,
)
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_chunk_layout import (
    build_synthetic_chunk_action_layout_for_tests,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)
from sana_wam.model.video_backbone.sana import hybrid_cache as hc
from sana_wam.model.video_backbone.sana import hybrid_cache_stage2b as h2


_LEDGER_ID = "offline-ledger-test"
_FENCE_ID = "offline-fence-test"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _secure_root(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path.resolve()


def _layout(*, episode_label: str = "ledger-episode"):
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=33,
        video_stride=1,
        source_row_label=f"{episode_label}-row",
        episode_label=episode_label,
    )
    return build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=33,
        video_valid_mask=(True,) * 33,
        action_valid_mask=(True,) * 32,
    )


def _registry(*, operator_class: str = "synthetic.LedgerGDN"):
    return hc.LayerRegistry(
        layers=(
            hc.LayerSpec(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class=operator_class,
                main_shortconv_enabled=True,
            ),
        )
    )


class _RecordingPublisher:
    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}

    def publish_exclusive(self, *, receipt_id, payload, expected_sha256):
        assert _sha256(payload) == expected_sha256
        prior = self.payloads.setdefault(receipt_id, payload)
        assert prior == payload
        return hc.ReceiptPublication(
            receipt_id=receipt_id,
            sha256=expected_sha256,
            durable=True,
            readback_verified=True,
        )


def _request(layout):
    chunk = layout.chunks[0]
    frame_mask = torch.tensor(chunk.latent_valid_mask, dtype=torch.bool).view(1, -1)
    video = torch.arange(
        3 * len(chunk.latent_valid_mask) * 4,
        dtype=torch.float32,
    ).reshape(1, 3, len(chunk.latent_valid_mask), 2, 2)
    video = video.to(dtype=torch.bfloat16).masked_fill(
        (~frame_mask)[:, None, :, None, None],
        0,
    )
    actions = torch.arange(
        chunk.action_slot_capacity * 20,
        dtype=torch.float32,
    ).reshape(1, chunk.action_slot_capacity, 20)
    actions = actions.to(dtype=torch.bfloat16)
    action_mask = torch.tensor(chunk.action_valid_mask, dtype=torch.bool).view(1, -1)
    actions = actions.masked_fill((~action_mask)[:, :, None], 0)
    proof = hc.TeacherForcingDatasetPairProof(
        dataset_manifest_sha256="b" * 64,
        dataset_episode_id="ledger-episode",
        dataset_row_identity="ledger-episode:chunk-0",
        dataset_row_start=0,
        dataset_row_end_exclusive=layout.valid_raw_count,
        layout_spec_sha256=layout.layout_spec_sha256,
        layout_instance_digest=layout.layout_instance_digest,
        video_tensor_digest=hc.tensor_digest(video),
        frame_valid_mask_digest=hc.tensor_digest(frame_mask),
        action_tensor_digest=hc.tensor_digest(actions),
        action_mask_digest=hc.tensor_digest(action_mask),
        row_order_manifest_sha256="c" * 64,
    )
    return h2.Stage2BTeacherForcingCommitRequest(
        commit_id="ledger-commit-0",
        transaction_nonce="ledger-nonce-0",
        expected_episode_id="ledger-episode",
        expected_episode_epoch=0,
        expected_revision=0,
        content_time=hc.ContentTime.from_layout(
            episode_id="ledger-episode",
            episode_epoch=0,
            layout=layout,
            chunk=chunk,
        ),
        video=video,
        frame_valid_mask=frame_mask,
        actions=actions,
        action_valid_mask=action_mask,
        proof=proof,
    )


def _layer_payloads(context):
    assert context.content_time.chunk_id == 0
    return (
        hc.StagedLayerPayload(
            layer_index=0,
            kind=hc.LayerKind.GDN_FULL_HISTORY,
            tensors={
                "ffn_tconv_left_context": torch.ones(
                    (1, 2, 2, 3), dtype=torch.float32
                ),
                "main_s_kv": torch.ones((1, 2, 4, 4), dtype=torch.bfloat16),
                "main_s_z": torch.ones((1, 2, 4, 1), dtype=torch.float32),
                "main_shortconv_left_context": torch.ones(
                    (1, 2, 3), dtype=torch.bool
                ),
            },
        ),
    )


@dataclass(frozen=True)
class _LedgerCase:
    layout: object
    registry: hc.LayerRegistry
    request: h2.Stage2BTeacherForcingCommitRequest
    genesis_state: h2.Stage2BTemporalState
    committed_state: h2.Stage2BTemporalState
    genesis_snapshot: EncodedStage2BStateSnapshot
    committed_snapshot: EncodedStage2BStateSnapshot
    receipt: h2.Stage2BPairedCommitReceipt
    receipt_bytes: bytes
    operation_bytes: bytes
    reset_layout: object
    reset_state: h2.Stage2BTemporalState
    reset_snapshot: EncodedStage2BStateSnapshot
    reset_receipt: h2.Stage2BCacheResetReceipt
    reset_receipt_bytes: bytes
    reset_operation_bytes: bytes


@pytest.fixture(scope="module")
def ledger_case() -> _LedgerCase:
    layout = _layout()
    registry = _registry()
    publisher = _RecordingPublisher()
    manager = h2.Stage2BHybridCacheManager.for_synthetic_tests(
        registry=registry,
        episode_id="ledger-episode",
        layout=layout,
        receipt_publisher=publisher,
        staging_variant=CACHStagingVariant.CACH_A,
    )
    genesis_state = manager.state
    genesis_snapshot = encode_stage2b_state_snapshot(
        state=genesis_state,
        expected_layout=layout,
        expected_registry=registry,
    )
    request = _request(layout)
    receipt = manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=_layer_payloads,
    )
    committed_state = manager.state
    committed_snapshot = encode_stage2b_state_snapshot(
        state=committed_state,
        expected_layout=layout,
        expected_registry=registry,
    )
    receipt_bytes = publisher.payloads[receipt.commit_id]
    assert receipt_bytes == receipt.canonical_bytes
    operation_bytes = _canonical(
        {
            "committed_action_span_digest": (
                committed_state.committed_action_history.spans[-1].actions_digest
            ),
            "conditioning_digest": receipt.conditioning_digest,
            "content_time_sha256": _sha256(
                _canonical(request.content_time.to_manifest())
            ),
            "dataset_manifest_sha256": receipt.dataset_manifest_sha256,
            "dataset_row_identity": receipt.dataset_row_identity,
            "episode_epoch_before": request.expected_episode_epoch,
            "episode_id_before": request.expected_episode_id,
            "layout_instance_digest": layout.layout_instance_digest,
            "layout_spec_sha256": layout.layout_spec_sha256,
            "pair_action_digest": receipt.pair_action_digest,
            "pair_action_mask_digest": receipt.pair_action_mask_digest,
            "pair_frame_valid_mask_digest": receipt.pair_frame_valid_mask_digest,
            "pair_video_digest": receipt.pair_video_digest,
            "revision_before": request.expected_revision,
            "schema": "cach.stage2b.paired_commit_operation.v1",
            "source_proof_digest": receipt.source_proof_digest,
            "teacher_pair_payload_digest": receipt.teacher_pair_payload_digest,
            "transaction_id": request.commit_id,
            "transaction_kind": Stage2BTransactionKind.PAIRED_COMMIT.value,
            "transaction_nonce": request.transaction_nonce,
        }
    )
    reset_layout = _layout(episode_label="reset-episode")
    assert reset_layout.layout_spec_sha256 == layout.layout_spec_sha256
    reset_receipt = manager.reset(
        reset_id="ledger-reset-0",
        transaction_nonce="ledger-reset-nonce-0",
        new_episode_id="reset-episode",
        layout=reset_layout,
    )
    reset_state = manager.state
    reset_snapshot = encode_stage2b_state_snapshot(
        state=reset_state,
        expected_layout=reset_layout,
        expected_registry=registry,
    )
    reset_receipt_bytes = publisher.payloads[reset_receipt.reset_id]
    assert reset_receipt_bytes == reset_receipt.canonical_bytes
    assert reset_receipt.aborted_pending_commit_id is None
    assert reset_snapshot.blobs == ()
    reset_operation_bytes = _canonical(
        {
            "action_cursor_before": committed_state.action_cursor,
            "action_history_digest_before": (
                committed_state.committed_action_history.manifest_digest
            ),
            "episode_epoch_before": committed_state.episode_epoch,
            "episode_id_before": committed_state.episode_id,
            "layout_instance_digest": reset_layout.layout_instance_digest,
            "layout_spec_sha256": reset_layout.layout_spec_sha256,
            "new_episode_epoch": reset_state.episode_epoch,
            "new_episode_id": reset_state.episode_id,
            "revision_before": committed_state.revision,
            "schema": "cach.stage2b.reset_operation.v1",
            "staging_variant": CACHStagingVariant.CACH_A.value,
            "state_manifest_before": committed_state.state_manifest_digest,
            "transaction_id": reset_receipt.reset_id,
            "transaction_kind": Stage2BTransactionKind.RESET.value,
            "transaction_nonce": reset_receipt.transaction_nonce,
        }
    )
    return _LedgerCase(
        layout=layout,
        registry=registry,
        request=request,
        genesis_state=genesis_state,
        committed_state=committed_state,
        genesis_snapshot=genesis_snapshot,
        committed_snapshot=committed_snapshot,
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        operation_bytes=operation_bytes,
        reset_layout=reset_layout,
        reset_state=reset_state,
        reset_snapshot=reset_snapshot,
        reset_receipt=reset_receipt,
        reset_receipt_bytes=reset_receipt_bytes,
        reset_operation_bytes=reset_operation_bytes,
    )


def _initialize(root: Path, case: _LedgerCase) -> Stage2BOfflineLedger:
    return Stage2BOfflineLedger.initialize(
        root=_secure_root(root),
        ledger_id=_LEDGER_ID,
        staging_variant=CACHStagingVariant.CACH_A,
        writer_fence_id=_FENCE_ID,
        expected_layout=case.layout,
        expected_registry=case.registry,
        genesis_snapshot=case.genesis_snapshot,
        additional_expected_layouts=(case.reset_layout,),
    )


def _reopen(root: Path, case: _LedgerCase) -> Stage2BOfflineLedger:
    return Stage2BOfflineLedger.reopen(
        root=root.resolve(),
        expected_ledger_id=_LEDGER_ID,
        expected_staging_variant=CACHStagingVariant.CACH_A,
        expected_writer_fence_id=_FENCE_ID,
        expected_layout=case.layout,
        expected_registry=case.registry,
        expected_genesis_snapshot=case.genesis_snapshot,
        additional_expected_layouts=(case.reset_layout,),
    )


def _operation_identity(
    case: _LedgerCase,
    *,
    state_manifest_before: str | None = None,
) -> Stage2BOperationIdentity:
    return Stage2BOperationIdentity(
        ledger_id=_LEDGER_ID,
        transaction_id=case.request.commit_id,
        transaction_kind=Stage2BTransactionKind.PAIRED_COMMIT,
        transaction_nonce=case.request.transaction_nonce,
        staging_variant=CACHStagingVariant.CACH_A,
        episode_id_before=case.genesis_state.episode_id,
        episode_epoch_before=case.genesis_state.episode_epoch,
        revision_before=case.genesis_state.revision,
        state_manifest_before=(
            case.genesis_state.state_manifest_digest
            if state_manifest_before is None
            else state_manifest_before
        ),
        layout_spec_sha256=case.layout.layout_spec_sha256,
        layout_instance_digest=case.layout.layout_instance_digest,
        operation_payload_sha256=_sha256(case.operation_bytes),
    )


def _prepare(
    ledger: Stage2BOfflineLedger,
    case: _LedgerCase,
    *,
    state_manifest_before: str | None = None,
) -> Stage2BLedgerIntent:
    identity = _operation_identity(
        case,
        state_manifest_before=state_manifest_before,
    )
    stored_identity = ledger.publish_operation_identity(
        identity=identity,
        operation_payload=case.operation_bytes,
    )
    intent = Stage2BLedgerIntent(
        ledger_id=_LEDGER_ID,
        ledger_sequence=0,
        transaction_id=case.request.commit_id,
        transaction_kind=Stage2BTransactionKind.PAIRED_COMMIT,
        transaction_nonce=case.request.transaction_nonce,
        writer_fence_id=_FENCE_ID,
        staging_variant=CACHStagingVariant.CACH_A,
        episode_id_before=case.genesis_state.episode_id,
        episode_epoch_before=case.genesis_state.episode_epoch,
        revision_before=case.genesis_state.revision,
        state_manifest_before=identity.state_manifest_before,
        input_identity_sha256=stored_identity.sha256,
        previous_decision_sha256=None,
    )
    ledger.prepare(intent)
    return intent


def _prepare_with_operation_payload(
    ledger: Stage2BOfflineLedger,
    case: _LedgerCase,
    operation_payload: bytes,
) -> Stage2BLedgerIntent:
    identity = replace(
        _operation_identity(case),
        operation_payload_sha256=_sha256(operation_payload),
    )
    stored_identity = ledger.publish_operation_identity(
        identity=identity,
        operation_payload=operation_payload,
    )
    intent = Stage2BLedgerIntent(
        ledger_id=_LEDGER_ID,
        ledger_sequence=0,
        transaction_id=case.request.commit_id,
        transaction_kind=Stage2BTransactionKind.PAIRED_COMMIT,
        transaction_nonce=case.request.transaction_nonce,
        writer_fence_id=_FENCE_ID,
        staging_variant=CACHStagingVariant.CACH_A,
        episode_id_before=case.genesis_state.episode_id,
        episode_epoch_before=case.genesis_state.episode_epoch,
        revision_before=case.genesis_state.revision,
        state_manifest_before=case.genesis_state.state_manifest_digest,
        input_identity_sha256=stored_identity.sha256,
        previous_decision_sha256=None,
    )
    ledger.prepare(intent)
    return intent


def _commit(
    ledger: Stage2BOfflineLedger,
    case: _LedgerCase,
    intent: Stage2BLedgerIntent,
):
    return ledger.commit(
        intent=intent,
        snapshot=case.committed_snapshot,
        success_receipt_bytes=case.receipt_bytes,
        episode_id_after=case.committed_state.episode_id,
        episode_epoch_after=case.committed_state.episode_epoch,
        revision_after=case.committed_state.revision,
        state_manifest_after=case.committed_state.state_manifest_digest,
    )


def _prepare_reset(
    ledger: Stage2BOfflineLedger,
    case: _LedgerCase,
    *,
    previous_decision_sha256: str,
) -> Stage2BLedgerIntent:
    identity = Stage2BOperationIdentity(
        ledger_id=_LEDGER_ID,
        transaction_id=case.reset_receipt.reset_id,
        transaction_kind=Stage2BTransactionKind.RESET,
        transaction_nonce=case.reset_receipt.transaction_nonce,
        staging_variant=CACHStagingVariant.CACH_A,
        episode_id_before=case.committed_state.episode_id,
        episode_epoch_before=case.committed_state.episode_epoch,
        revision_before=case.committed_state.revision,
        state_manifest_before=case.committed_state.state_manifest_digest,
        layout_spec_sha256=case.reset_layout.layout_spec_sha256,
        layout_instance_digest=case.reset_layout.layout_instance_digest,
        operation_payload_sha256=_sha256(case.reset_operation_bytes),
    )
    stored_identity = ledger.publish_operation_identity(
        identity=identity,
        operation_payload=case.reset_operation_bytes,
    )
    intent = Stage2BLedgerIntent(
        ledger_id=_LEDGER_ID,
        ledger_sequence=1,
        transaction_id=case.reset_receipt.reset_id,
        transaction_kind=Stage2BTransactionKind.RESET,
        transaction_nonce=case.reset_receipt.transaction_nonce,
        writer_fence_id=_FENCE_ID,
        staging_variant=CACHStagingVariant.CACH_A,
        episode_id_before=case.committed_state.episode_id,
        episode_epoch_before=case.committed_state.episode_epoch,
        revision_before=case.committed_state.revision,
        state_manifest_before=case.committed_state.state_manifest_digest,
        input_identity_sha256=stored_identity.sha256,
        previous_decision_sha256=previous_decision_sha256,
    )
    ledger.prepare(intent)
    return intent


def test_initialize_and_reopen_bind_exact_genesis_layout_registry(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    root = tmp_path / "ledger"
    ledger = _initialize(root, ledger_case)

    assert ledger.scan_valid_chain() == ()
    assert ledger.identity.layout_spec_sha256 == ledger_case.layout.layout_spec_sha256
    assert (
        ledger.identity.layout_instance_digest
        == ledger_case.layout.layout_instance_digest
    )
    assert ledger.identity.layer_registry_digest == ledger_case.registry.manifest_digest
    assert (
        ledger.identity.genesis_state_manifest
        == ledger_case.genesis_state.state_manifest_digest
    )
    for path in root.iterdir():
        value = path.lstat()
        assert stat.S_ISREG(value.st_mode)
        assert value.st_nlink == 1
        assert stat.S_IMODE(value.st_mode) == 0o400

    reopened = _reopen(root, ledger_case)
    assert reopened.identity == ledger.identity
    assert reopened.scan_valid_chain() == ()

    with pytest.raises(Stage2BOfflineLedgerError, match="identity differs"):
        Stage2BOfflineLedger.reopen(
            root=root.resolve(),
            expected_ledger_id=_LEDGER_ID,
            expected_staging_variant=CACHStagingVariant.CACH_A,
            expected_writer_fence_id="wrong-fence",
            expected_layout=ledger_case.layout,
            expected_registry=ledger_case.registry,
            expected_genesis_snapshot=ledger_case.genesis_snapshot,
        )


def test_operation_identity_and_first_intent_are_pinned_to_genesis(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    before = {path.name for path in ledger.root.iterdir()}
    empty_payload = _canonical({})
    invalid_identity = replace(
        _operation_identity(ledger_case),
        operation_payload_sha256=_sha256(empty_payload),
    )
    with pytest.raises(Stage2BOfflineLedgerError, match="members differ"):
        ledger.publish_operation_identity(
            identity=invalid_identity,
            operation_payload=empty_payload,
        )
    assert {path.name for path in ledger.root.iterdir()} == before

    forged = "d" * 64
    with pytest.raises(Stage2BOfflineLedgerError, match="predecessor differs"):
        _prepare(ledger, ledger_case, state_manifest_before=forged)
    assert ledger.scan_valid_chain() == ()

    valid_intent = _prepare(ledger, ledger_case)
    with pytest.raises(Stage2BOfflineLedgerConflictError):
        ledger.prepare(replace(valid_intent, transaction_nonce="conflicting-nonce"))
    assert "/" not in Stage2BOfflineLedger.intent_filename("../../escape.json")
    with pytest.raises(Stage2BOfflineLedgerError, match="plain int"):
        replace(valid_intent, ledger_sequence=True)


def test_prepare_and_abort_are_exact_idempotent_and_state_preserving(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    root = tmp_path / "ledger"
    ledger = _initialize(root, ledger_case)
    intent = _prepare(ledger, ledger_case)
    assert ledger.prepare(intent).payload == intent.canonical_bytes

    decision = ledger.abort(
        intent=intent,
        abort_code=Stage2BAbortCode.STAGING_FAILED,
        failure_point=Stage2BFailurePoint.STAGING,
    )
    body = decision.decision.decision_body
    assert body.result is Stage2BDecisionResult.ABORTED
    assert body.snapshot_manifest_sha256 is None
    assert body.state_manifest_after == body.state_manifest_before
    assert body.revision_after == body.revision_before
    assert ledger.abort(
        intent=intent,
        abort_code=Stage2BAbortCode.STAGING_FAILED,
        failure_point=Stage2BFailurePoint.STAGING,
    ).sha256 == decision.sha256
    assert _reopen(root, ledger_case).scan_valid_chain()[0].sha256 == decision.sha256

    before_conflict = {path.name for path in root.iterdir()}
    with pytest.raises(Stage2BOfflineLedgerConflictError):
        _commit(ledger, ledger_case, intent)
    assert {path.name for path in root.iterdir()} == before_conflict


def test_real_paired_commit_round_trip_and_receipt_cross_binding(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    root = tmp_path / "ledger"
    ledger = _initialize(root, ledger_case)
    intent = _prepare(ledger, ledger_case)
    decision = _commit(ledger, ledger_case, intent)

    body = decision.decision.decision_body
    assert body.result is Stage2BDecisionResult.COMMITTED
    assert body.state_manifest_after == ledger_case.committed_state.state_manifest_digest
    assert body.snapshot_manifest_sha256 == ledger_case.committed_snapshot.manifest_sha256
    assert _commit(ledger, ledger_case, intent).sha256 == decision.sha256
    chain = _reopen(root, ledger_case).scan_valid_chain()
    assert [item.sha256 for item in chain] == [decision.sha256]

    wrong_root = tmp_path / "wrong-receipt-ledger"
    wrong_ledger = _initialize(wrong_root, ledger_case)
    wrong_intent = _prepare(wrong_ledger, ledger_case)
    wrong_receipt = replace(
        ledger_case.receipt,
        transaction_nonce="different-valid-nonce",
    )
    with pytest.raises(Stage2BOfflineLedgerError, match="receipt/decision"):
        wrong_ledger.commit(
            intent=wrong_intent,
            snapshot=ledger_case.committed_snapshot,
            success_receipt_bytes=wrong_receipt.canonical_bytes,
            episode_id_after=ledger_case.committed_state.episode_id,
            episode_epoch_after=ledger_case.committed_state.episode_epoch,
            revision_after=ledger_case.committed_state.revision,
            state_manifest_after=ledger_case.committed_state.state_manifest_digest,
        )
    assert wrong_ledger.read_decision(wrong_intent.transaction_id) is None


@pytest.mark.parametrize(
    "operation_field",
    [
        "conditioning_digest",
        "dataset_manifest_sha256",
        "dataset_row_identity",
        "committed_action_span_digest",
        "content_time_sha256",
        "source_proof_digest",
        "pair_action_digest",
        "pair_action_mask_digest",
        "pair_frame_valid_mask_digest",
        "pair_video_digest",
        "teacher_pair_payload_digest",
    ],
)
def test_paired_operation_digests_are_cross_bound_before_terminal_publication(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    operation_field: str,
) -> None:
    ledger = _initialize(tmp_path / operation_field, ledger_case)
    operation = json.loads(ledger_case.operation_bytes)
    replacement_digest = "f" * 64
    assert operation[operation_field] != replacement_digest
    operation[operation_field] = replacement_digest
    intent = _prepare_with_operation_payload(
        ledger,
        ledger_case,
        _canonical(operation),
    )

    with pytest.raises(Stage2BOfflineLedgerError, match="paired"):
        _commit(ledger, ledger_case, intent)
    assert ledger.read_decision(intent.transaction_id) is None
    assert ledger.read_intent(intent.transaction_id).payload == intent.canonical_bytes


@pytest.mark.parametrize(
    "receipt_mutation",
    [
        "episode_id",
        "episode_epoch",
        "layout",
        "cursor_after",
        "history_after",
        "state_before",
        "state_after",
        "revision_group",
    ],
)
def test_paired_receipt_fields_are_cross_bound_to_intent_and_snapshot(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    receipt_mutation: str,
) -> None:
    ledger = _initialize(tmp_path / receipt_mutation, ledger_case)
    intent = _prepare(ledger, ledger_case)
    other_digest = "f" * 64
    if receipt_mutation == "episode_id":
        changes = {"episode_id": "wrong-episode"}
    elif receipt_mutation == "episode_epoch":
        changes = {"episode_epoch": ledger_case.receipt.episode_epoch + 1}
    elif receipt_mutation == "layout":
        changes = {"layout_instance_digest": other_digest}
    elif receipt_mutation == "cursor_after":
        changes = {"action_cursor_after": ledger_case.receipt.action_cursor_after + 1}
    elif receipt_mutation == "history_after":
        changes = {"action_history_digest_after": other_digest}
    elif receipt_mutation == "state_before":
        changes = {"state_manifest_before": other_digest}
    elif receipt_mutation == "state_after":
        changes = {
            "staged_state_manifest": other_digest,
            "state_manifest_after": other_digest,
        }
    else:
        changes = {
            "revision_before": 1,
            "revision_after": 2,
            "chunk_id": 1,
            "action_history_digest_before": other_digest,
            "action_cursor_before": 1,
        }
    wrong_receipt = replace(ledger_case.receipt, **changes)

    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.commit(
            intent=intent,
            snapshot=ledger_case.committed_snapshot,
            success_receipt_bytes=wrong_receipt.canonical_bytes,
            episode_id_after=ledger_case.committed_state.episode_id,
            episode_epoch_after=ledger_case.committed_state.episode_epoch,
            revision_after=ledger_case.committed_state.revision,
            state_manifest_after=ledger_case.committed_state.state_manifest_digest,
        )
    assert caught.value.code == "LEDGER_REFERENCE_MISMATCH"
    assert ledger.read_decision(intent.transaction_id) is None
    assert ledger.scan_valid_chain() == ()


@pytest.mark.parametrize(
    "lineage_mutation",
    ["history_commit_id", "cache_paired_payload", "source_lineage"],
)
def test_l1_valid_snapshot_lineage_is_cross_bound_before_terminal_publication(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    lineage_mutation: str,
) -> None:
    ledger = _initialize(tmp_path / lineage_mutation, ledger_case)
    intent = _prepare(ledger, ledger_case)
    state = ledger_case.committed_state
    history = state.committed_action_history
    assert history is not None
    span = history.spans[-1]
    cache_state = state.cache_state
    other_digest = "f" * 64
    if lineage_mutation == "history_commit_id":
        mutated_history = replace(
            history,
            spans=(replace(span, commit_id="wrong-commit-id"),),
        )
        mutated_cache = cache_state
    elif lineage_mutation == "cache_paired_payload":
        mutated_history = history
        mutated_cache = replace(
            cache_state,
            committed_pair_digest=other_digest,
        )
    else:
        mutated_history = replace(
            history,
            spans=(replace(span, source_proof_digest=other_digest),),
        )
        mutated_cache = replace(
            cache_state,
            last_source_proof_digest=other_digest,
        )
    mutated_state = replace(
        state,
        cache_state=mutated_cache,
        committed_action_history=mutated_history,
    )
    mutated_snapshot = encode_stage2b_state_snapshot(
        state=mutated_state,
        expected_layout=ledger_case.layout,
        expected_registry=ledger_case.registry,
    )
    mutated_receipt = replace(
        ledger_case.receipt,
        staged_state_manifest=mutated_state.state_manifest_digest,
        state_manifest_after=mutated_state.state_manifest_digest,
        action_history_digest_after=mutated_history.manifest_digest,
    )

    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.commit(
            intent=intent,
            snapshot=mutated_snapshot,
            success_receipt_bytes=mutated_receipt.canonical_bytes,
            episode_id_after=mutated_state.episode_id,
            episode_epoch_after=mutated_state.episode_epoch,
            revision_after=mutated_state.revision,
            state_manifest_after=mutated_state.state_manifest_digest,
        )
    assert caught.value.code == "LEDGER_REFERENCE_MISMATCH"
    assert ledger.read_decision(intent.transaction_id) is None
    assert ledger.scan_valid_chain() == ()


def test_snapshot_requires_full_l1_decode_before_any_publication(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    before = {path.name for path in ledger.root.iterdir()}
    manifest = json.loads(ledger_case.committed_snapshot.manifest_bytes)
    tensor = manifest["cache_state"]["layer_states"][0]["tensors"][0]
    tensor["shape"] = [1, 2, 999]
    malformed = replace(
        ledger_case.committed_snapshot,
        manifest_bytes=_canonical(manifest),
    )

    with pytest.raises(Stage2BOfflineLedgerError, match="full L1 validation"):
        ledger.publish_snapshot(malformed)
    assert {path.name for path in ledger.root.iterdir()} == before


@pytest.mark.parametrize(
    "publication_boundary",
    ["first_blob", "last_blob", "snapshot_manifest", "success_receipt"],
)
def test_commit_multi_object_failure_never_publishes_terminal_early(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
    publication_boundary: str,
) -> None:
    ledger = _initialize(tmp_path / publication_boundary, ledger_case)
    intent = _prepare(ledger, ledger_case)
    blobs = ledger_case.committed_snapshot.blobs
    assert len(blobs) >= 2
    target_digest = {
        "first_blob": blobs[0].sha256,
        "last_blob": blobs[-1].sha256,
        "snapshot_manifest": ledger_case.committed_snapshot.manifest_sha256,
        "success_receipt": _sha256(ledger_case.receipt_bytes),
    }[publication_boundary]
    real_publish_object = ledger.publish_object

    def fail_at_boundary(*, payload: bytes, expected_sha256: str | None = None):
        observed = _sha256(payload)
        if observed == target_digest:
            raise Stage2BOfflineLedgerError(
                "LEDGER_IO_FAILED",
                f"injected {publication_boundary} failure",
            )
        return real_publish_object(
            payload=payload,
            expected_sha256=expected_sha256,
        )

    monkeypatch.setattr(ledger, "publish_object", fail_at_boundary)
    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        _commit(ledger, ledger_case, intent)
    assert caught.value.code == "LEDGER_IO_FAILED"
    assert ledger.read_decision(intent.transaction_id) is None
    assert ledger.read_intent(intent.transaction_id).payload == intent.canonical_bytes
    assert not (
        ledger.root / Stage2BOfflineLedger.object_filename(target_digest)
    ).exists()
    if publication_boundary in {"first_blob", "last_blob"}:
        assert not (
            ledger.root
            / Stage2BOfflineLedger.object_filename(
                ledger_case.committed_snapshot.manifest_sha256
            )
        ).exists()
    assert not any(path.name.startswith(".tmp-") for path in ledger.root.iterdir())


@pytest.mark.parametrize("durable_boundary", ["snapshot", "success_receipt"])
def test_commit_resumes_exactly_after_completed_object_group_failure(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
    durable_boundary: str,
) -> None:
    ledger = _initialize(tmp_path / durable_boundary, ledger_case)
    intent = _prepare(ledger, ledger_case)
    failed = False
    real_publish_snapshot = ledger.publish_snapshot
    real_publish_receipt = ledger.publish_receipt

    if durable_boundary == "snapshot":

        def publish_snapshot_then_fail(snapshot: EncodedStage2BStateSnapshot):
            nonlocal failed
            stored = real_publish_snapshot(snapshot)
            if not failed:
                failed = True
                raise Stage2BOfflineLedgerError(
                    "LEDGER_IO_FAILED",
                    "injected failure after durable snapshot group",
                )
            return stored

        monkeypatch.setattr(ledger, "publish_snapshot", publish_snapshot_then_fail)
    else:

        def publish_receipt_then_fail(payload: bytes):
            nonlocal failed
            stored = real_publish_receipt(payload)
            if not failed:
                failed = True
                raise Stage2BOfflineLedgerError(
                    "LEDGER_IO_FAILED",
                    "injected failure after durable success receipt",
                )
            return stored

        monkeypatch.setattr(ledger, "publish_receipt", publish_receipt_then_fail)

    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        _commit(ledger, ledger_case, intent)
    assert caught.value.code == "LEDGER_IO_FAILED"
    assert failed
    assert ledger.scan_valid_chain() == ()
    assert ledger.read_intent(intent.transaction_id).payload == intent.canonical_bytes
    assert ledger.read_decision(intent.transaction_id) is None
    assert (
        ledger.read_object(ledger_case.committed_snapshot.manifest_sha256).payload
        == ledger_case.committed_snapshot.manifest_bytes
    )
    for blob in ledger_case.committed_snapshot.blobs:
        assert ledger.read_object(blob.sha256).payload == blob.payload
    receipt_path = ledger.root / Stage2BOfflineLedger.object_filename(
        _sha256(ledger_case.receipt_bytes)
    )
    assert receipt_path.exists() is (durable_boundary == "success_receipt")
    assert not any(path.name.startswith(".tmp-") for path in ledger.root.iterdir())

    committed = _commit(ledger, ledger_case, intent)
    assert committed.decision.decision_body.result is Stage2BDecisionResult.COMMITTED
    assert [item.sha256 for item in ledger.scan_valid_chain()] == [committed.sha256]
    assert _commit(ledger, ledger_case, intent).sha256 == committed.sha256


def test_real_reset_advances_epoch_and_switches_to_pinned_layout(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    root = tmp_path / "ledger"
    ledger = _initialize(root, ledger_case)
    commit_intent = _prepare(ledger, ledger_case)
    committed = _commit(ledger, ledger_case, commit_intent)
    reset_intent = _prepare_reset(
        ledger,
        ledger_case,
        previous_decision_sha256=committed.sha256,
    )
    reset = ledger.commit(
        intent=reset_intent,
        snapshot=ledger_case.reset_snapshot,
        success_receipt_bytes=ledger_case.reset_receipt_bytes,
        episode_id_after=ledger_case.reset_state.episode_id,
        episode_epoch_after=ledger_case.reset_state.episode_epoch,
        revision_after=ledger_case.reset_state.revision,
        state_manifest_after=ledger_case.reset_state.state_manifest_digest,
    )

    body = reset.decision.decision_body
    assert body.result is Stage2BDecisionResult.COMMITTED
    assert body.transaction_kind is Stage2BTransactionKind.RESET
    assert body.previous_decision_sha256 == committed.sha256
    assert body.episode_epoch_after == body.episode_epoch_before + 1
    assert body.revision_after == 0
    assert body.snapshot_manifest_sha256 == ledger_case.reset_snapshot.manifest_sha256
    assert [item.sha256 for item in _reopen(root, ledger_case).scan_valid_chain()] == [
        committed.sha256,
        reset.sha256,
    ]

    with pytest.raises(Stage2BOfflineLedgerError, match="identity differs"):
        Stage2BOfflineLedger.reopen(
            root=root.resolve(),
            expected_ledger_id=_LEDGER_ID,
            expected_staging_variant=CACHStagingVariant.CACH_A,
            expected_writer_fence_id=_FENCE_ID,
            expected_layout=ledger_case.layout,
            expected_registry=ledger_case.registry,
            expected_genesis_snapshot=ledger_case.genesis_snapshot,
        )


def test_reconciliation_validates_exact_absent_conflict_and_whole_root(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    ledger = _initialize(tmp_path / "exact", ledger_case)
    intent = _prepare(ledger, ledger_case)
    decision = ledger.abort(
        intent=intent,
        abort_code=Stage2BAbortCode.CALLER_CANCELLED,
        failure_point=Stage2BFailurePoint.BEFORE_STAGING,
    )
    exact = ledger.reconcile_terminal(
        transaction_id=intent.transaction_id,
        expected_record_sha256=decision.sha256,
    )
    assert exact.status is Stage2BReconcileStatus.EXACT
    assert exact.stored is not None and exact.stored.sha256 == decision.sha256
    with pytest.raises(Stage2BOfflineLedgerConflictError):
        ledger.reconcile_terminal(
            transaction_id=intent.transaction_id,
            expected_record_sha256="0" * 64,
        )

    absent_ledger = _initialize(tmp_path / "absent", ledger_case)
    absent_intent = _prepare(absent_ledger, ledger_case)
    absent = absent_ledger.reconcile_terminal(
        transaction_id=absent_intent.transaction_id,
        expected_record_sha256="1" * 64,
    )
    assert absent.status is Stage2BReconcileStatus.ABSENT
    assert absent.stored is None

    foreign = absent_ledger.root / "foreign-entry"
    foreign.write_bytes(b"poison")
    foreign.chmod(0o400)
    with pytest.raises(Stage2BOfflineLedgerError, match="unexpected ledger entry"):
        absent_ledger.reconcile_terminal(
            transaction_id=absent_intent.transaction_id,
            expected_record_sha256="1" * 64,
        )


def test_reconciliation_rejects_same_bytes_pending_inode_replacement(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    intent = _prepare(ledger, ledger_case)
    pending_path = ledger.root / Stage2BOfflineLedger.intent_filename(
        intent.transaction_id
    )
    payload = pending_path.read_bytes()
    original_inode = pending_path.stat().st_ino
    real_read_optional = ledger._read_optional
    replaced = False

    def replace_pending_before_first_terminal_read(filename: str):
        nonlocal replaced
        if not replaced:
            replacement = ledger.root / ".same-bytes-replacement"
            replacement.write_bytes(payload)
            replacement.chmod(0o400)
            os.replace(replacement, pending_path)
            replaced = True
        return real_read_optional(filename)

    monkeypatch.setattr(
        ledger,
        "_read_optional",
        replace_pending_before_first_terminal_read,
    )
    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.reconcile_terminal(
            transaction_id=intent.transaction_id,
            expected_record_sha256="1" * 64,
        )
    assert caught.value.code == "LEDGER_PUBLICATION_OUTCOME_UNKNOWN"
    assert pending_path.stat().st_ino != original_inode


def test_reconciliation_never_reports_absent_when_terminal_appears_between_reads(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    intent = _prepare(ledger, ledger_case)
    real_read_optional = ledger._read_optional
    reads = 0
    appeared = None

    def publish_before_second_read(filename: str):
        nonlocal reads, appeared
        reads += 1
        if reads == 2:
            appeared = ledger.abort(
                intent=intent,
                abort_code=Stage2BAbortCode.STAGING_FAILED,
                failure_point=Stage2BFailurePoint.STAGING,
            )
        return real_read_optional(filename)

    monkeypatch.setattr(ledger, "_read_optional", publish_before_second_read)
    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.reconcile_terminal(
            transaction_id=intent.transaction_id,
            expected_record_sha256="1" * 64,
        )
    assert caught.value.code == "LEDGER_PUBLICATION_OUTCOME_UNKNOWN"
    assert appeared is not None
    assert ledger.read_decision(intent.transaction_id) is not None


def test_reconciliation_never_reports_absent_when_target_becomes_malformed(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    intent = _prepare(ledger, ledger_case)
    filename = Stage2BOfflineLedger.decision_filename(intent.transaction_id)
    real_read_optional = ledger._read_optional
    reads = 0

    def corrupt_before_second_read(observed_filename: str):
        nonlocal reads
        reads += 1
        if reads == 2:
            target = ledger.root / filename
            target.write_bytes(b"{")
            target.chmod(0o400)
        return real_read_optional(observed_filename)

    monkeypatch.setattr(ledger, "_read_optional", corrupt_before_second_read)
    with pytest.raises(Stage2BOfflineLedgerError):
        ledger.reconcile_terminal(
            transaction_id=intent.transaction_id,
            expected_record_sha256="1" * 64,
        )


def test_reconciliation_directory_fsync_failure_is_unknown(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    intent = _prepare(ledger, ledger_case)
    real_fsync = ledger_module.os.fsync
    failed = False

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal failed
        if not failed and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            failed = True
            raise OSError("injected reconciliation directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(ledger_module.os, "fsync", fail_directory_fsync)
    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.reconcile_terminal(
            transaction_id=intent.transaction_id,
            expected_record_sha256="1" * 64,
        )
    assert caught.value.code == "LEDGER_PUBLICATION_OUTCOME_UNKNOWN"
    assert failed


def test_reconciliation_directory_close_failure_is_unknown(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    intent = _prepare(ledger, ledger_case)
    real_fsync = ledger_module.os.fsync
    real_close = ledger_module.os.close
    reconciliation_fsynced = False
    failed = False

    def track_directory_fsync(descriptor: int) -> None:
        nonlocal reconciliation_fsynced
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            reconciliation_fsynced = True
        real_fsync(descriptor)

    def fail_post_fsync_directory_close(descriptor: int) -> None:
        nonlocal failed
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        real_close(descriptor)
        if reconciliation_fsynced and is_directory and not failed:
            failed = True
            raise OSError("injected reconciliation directory close failure")

    monkeypatch.setattr(ledger_module.os, "fsync", track_directory_fsync)
    monkeypatch.setattr(ledger_module.os, "close", fail_post_fsync_directory_close)
    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.reconcile_terminal(
            transaction_id=intent.transaction_id,
            expected_record_sha256="1" * 64,
        )
    assert caught.value.code == "LEDGER_PUBLICATION_OUTCOME_UNKNOWN"
    assert failed


def test_abandoned_temp_and_exact_partial_initialization_resume_safely(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    temp_root = _secure_root(tmp_path / "temp-ledger")
    abandoned = temp_root / (".tmp-" + "a" * 32)
    abandoned.write_bytes(b"partial")
    abandoned.chmod(0o000)
    ledger = Stage2BOfflineLedger.initialize(
        root=temp_root,
        ledger_id=_LEDGER_ID,
        staging_variant=CACHStagingVariant.CACH_A,
        writer_fence_id=_FENCE_ID,
        expected_layout=ledger_case.layout,
        expected_registry=ledger_case.registry,
        genesis_snapshot=ledger_case.genesis_snapshot,
        additional_expected_layouts=(ledger_case.reset_layout,),
    )
    assert not abandoned.exists()
    assert ledger.scan_valid_chain() == ()

    partial_root = _secure_root(tmp_path / "partial-ledger")
    identity = ledger.identity
    identity_path = partial_root / "LEDGER.json"
    identity_path.write_bytes(identity.canonical_bytes)
    identity_path.chmod(0o400)
    resumed = Stage2BOfflineLedger.initialize(
        root=partial_root,
        ledger_id=_LEDGER_ID,
        staging_variant=CACHStagingVariant.CACH_A,
        writer_fence_id=_FENCE_ID,
        expected_layout=ledger_case.layout,
        expected_registry=ledger_case.registry,
        genesis_snapshot=ledger_case.genesis_snapshot,
        additional_expected_layouts=(ledger_case.reset_layout,),
    )
    assert resumed.scan_valid_chain() == ()


def test_temp_name_collision_is_not_unlinked_and_root_replacement_is_rejected(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "ledger"
    ledger = _initialize(root, ledger_case)
    colliding = root / (".tmp-" + "b" * 32)
    colliding.write_bytes(b"not-owned-by-publication")
    colliding.chmod(0o600)
    monkeypatch.setattr(ledger_module.secrets, "token_hex", lambda _: "b" * 32)
    with pytest.raises(Stage2BOfflineLedgerError, match="publication failed"):
        ledger.publish_object(payload=b"new-object")
    assert colliding.read_bytes() == b"not-owned-by-publication"

    replacement_root = tmp_path / "replacement-ledger"
    replacement_ledger = _initialize(replacement_root, ledger_case)
    payload = b"stable-object"
    replacement_ledger.publish_object(payload=payload)
    moved = tmp_path / "moved-ledger"
    os.rename(replacement_root, moved)
    _secure_root(replacement_root)
    with pytest.raises(Stage2BOfflineLedgerError, match="root was replaced"):
        replacement_ledger.publish_object(payload=payload)


def test_terminal_publication_abort_requires_stable_absence_reconciliation(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    intent = _prepare(ledger, ledger_case)
    with pytest.raises(Stage2BOfflineLedgerError, match="reconciliation"):
        ledger.abort(
            intent=intent,
            abort_code=Stage2BAbortCode.PUBLICATION_FAILED,
            failure_point=Stage2BFailurePoint.TERMINAL_PUBLICATION,
        )

    terminal = ledger.abort_after_reconciled_absence(
        intent=intent,
        expected_terminal_record_sha256="f" * 64,
    )
    assert terminal.decision.decision_body.result is Stage2BDecisionResult.ABORTED
    assert (
        terminal.decision.decision_body.abort_code
        is Stage2BAbortCode.PUBLICATION_FAILED
    )


def test_public_record_apis_cannot_bypass_terminal_absence_reconciliation(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    intent = _prepare(ledger, ledger_case)
    receipt = Stage2BAbortReceipt(
        ledger_id=intent.ledger_id,
        transaction_id=intent.transaction_id,
        transaction_kind=intent.transaction_kind,
        transaction_nonce=intent.transaction_nonce,
        intent_sha256=intent.intent_sha256,
        staging_variant=intent.staging_variant,
        abort_code=Stage2BAbortCode.PUBLICATION_FAILED,
        failure_point=Stage2BFailurePoint.TERMINAL_PUBLICATION,
        state_manifest_before=intent.state_manifest_before,
        state_manifest_after=intent.state_manifest_before,
    )
    with pytest.raises(Stage2BOfflineLedgerError) as receipt_error:
        ledger.publish_abort_receipt(receipt)
    assert receipt_error.value.code == "LEDGER_RECONCILIATION_REQUIRED"

    ledger.publish_receipt(receipt.canonical_bytes)
    candidate = Stage2BDecisionRecord.from_body(
        Stage2BDecisionBody(
            ledger_id=intent.ledger_id,
            ledger_sequence=intent.ledger_sequence,
            transaction_id=intent.transaction_id,
            transaction_kind=intent.transaction_kind,
            intent_sha256=intent.intent_sha256,
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
            abort_code=Stage2BAbortCode.PUBLICATION_FAILED,
        )
    )
    with pytest.raises(Stage2BOfflineLedgerError) as decision_error:
        ledger.publish_decision(candidate)
    assert decision_error.value.code == "LEDGER_RECONCILIATION_REQUIRED"
    assert ledger.read_decision(intent.transaction_id) is None


def test_post_rename_terminal_outcome_is_unknown_then_reconciles_exact(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    intent = _prepare(ledger, ledger_case)
    real_fsync = ledger_module.os.fsync
    directory_fsync_calls = 0

    def injected_fsync(descriptor: int) -> None:
        nonlocal directory_fsync_calls
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsync_calls += 1
            if directory_fsync_calls == 2:
                raise OSError("injected post-rename directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(ledger_module.os, "fsync", injected_fsync)
    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.abort(
            intent=intent,
            abort_code=Stage2BAbortCode.STAGING_FAILED,
            failure_point=Stage2BFailurePoint.STAGING,
        )
    assert caught.value.code == "LEDGER_PUBLICATION_OUTCOME_UNKNOWN"
    monkeypatch.setattr(ledger_module.os, "fsync", real_fsync)

    stored = ledger.read_decision(intent.transaction_id)
    assert stored is not None
    real_reconcile = ledger.reconcile_terminal
    reconciliation_calls = 0

    def recording_reconcile(*, transaction_id: str, expected_record_sha256: str):
        nonlocal reconciliation_calls
        reconciliation_calls += 1
        return real_reconcile(
            transaction_id=transaction_id,
            expected_record_sha256=expected_record_sha256,
        )

    monkeypatch.setattr(ledger, "reconcile_terminal", recording_reconcile)
    retried = ledger.abort(
        intent=intent,
        abort_code=Stage2BAbortCode.STAGING_FAILED,
        failure_point=Stage2BFailurePoint.STAGING,
    )
    assert retried.sha256 == stored.sha256
    assert reconciliation_calls == 1
    monkeypatch.setattr(ledger, "reconcile_terminal", real_reconcile)
    reconciled = ledger.reconcile_terminal(
        transaction_id=intent.transaction_id,
        expected_record_sha256=stored.sha256,
    )
    assert reconciled.status is Stage2BReconcileStatus.EXACT
    assert reconciled.stored is not None
    assert reconciled.stored.sha256 == stored.sha256


def test_post_rename_directory_close_failure_is_unknown_and_reconciles_exact(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    intent = _prepare(ledger, ledger_case)
    real_rename = ledger_module._rename_noreplace
    real_close = ledger_module.os.close
    terminal_renamed = False
    close_failed = False

    def tracking_rename(*args) -> None:
        nonlocal terminal_renamed
        real_rename(*args)
        destination_name = args[3]
        if destination_name == Stage2BOfflineLedger.decision_filename(
            intent.transaction_id
        ):
            terminal_renamed = True

    def injected_close(descriptor: int) -> None:
        nonlocal close_failed
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        real_close(descriptor)
        if terminal_renamed and is_directory and not close_failed:
            close_failed = True
            raise OSError("injected post-rename directory close failure")

    monkeypatch.setattr(ledger_module, "_rename_noreplace", tracking_rename)
    monkeypatch.setattr(ledger_module.os, "close", injected_close)
    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.abort(
            intent=intent,
            abort_code=Stage2BAbortCode.STAGING_FAILED,
            failure_point=Stage2BFailurePoint.STAGING,
        )
    assert caught.value.code == "LEDGER_PUBLICATION_OUTCOME_UNKNOWN"
    assert close_failed

    monkeypatch.setattr(ledger_module.os, "close", real_close)
    monkeypatch.setattr(ledger_module, "_rename_noreplace", real_rename)
    stored = ledger.read_decision(intent.transaction_id)
    assert stored is not None
    reconciled = ledger.reconcile_terminal(
        transaction_id=intent.transaction_id,
        expected_record_sha256=stored.sha256,
    )
    assert reconciled.status is Stage2BReconcileStatus.EXACT


@pytest.mark.parametrize(
    "poison",
    ["wrong_mode", "hardlink", "symlink", "fifo", "malformed_intent"],
)
def test_scan_rejects_unsafe_or_malformed_root_entries(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    poison: str,
) -> None:
    ledger = _initialize(tmp_path / poison, ledger_case)
    root = ledger.root
    object_name = Stage2BOfflineLedger.object_filename("0" * 64)
    if poison == "wrong_mode":
        (root / "LEDGER.json").chmod(0o600)
    elif poison == "hardlink":
        os.link(root / "LEDGER.json", root / object_name)
    elif poison == "symlink":
        (root / object_name).symlink_to(root / "LEDGER.json")
    elif poison == "fifo":
        os.mkfifo(root / object_name, mode=0o400)
    else:
        malformed = root / Stage2BOfflineLedger.intent_filename("malformed")
        malformed.write_bytes(b'{"x":1,"x":2}')
        malformed.chmod(0o400)

    with pytest.raises(Stage2BOfflineLedgerError):
        ledger.scan_valid_chain()


def test_root_symlink_is_rejected_before_initialization(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    actual = _secure_root(tmp_path / "actual")
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(Stage2BOfflineLedgerError, match="cannot be opened safely"):
        Stage2BOfflineLedger.initialize(
            root=linked.absolute(),
            ledger_id=_LEDGER_ID,
            staging_variant=CACHStagingVariant.CACH_A,
            writer_fence_id=_FENCE_ID,
            expected_layout=ledger_case.layout,
            expected_registry=ledger_case.registry,
            genesis_snapshot=ledger_case.genesis_snapshot,
            additional_expected_layouts=(ledger_case.reset_layout,),
        )


@pytest.mark.parametrize("race_result", ["exact", "conflict"])
def test_destination_eexist_race_is_idempotent_or_fail_closed(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
    race_result: str,
) -> None:
    ledger = _initialize(tmp_path / race_result, ledger_case)
    payload = b"destination-eexist-race"
    digest = _sha256(payload)
    filename = Stage2BOfflineLedger.object_filename(digest)
    competing_payload = payload if race_result == "exact" else b"racing-conflict"
    real_rename = ledger_module._rename_noreplace
    injected = False

    def inject_destination(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal injected
        if destination_name != filename or injected:
            real_rename(
                source_directory_fd,
                source_name,
                destination_directory_fd,
                destination_name,
            )
            return
        injected = True
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(
            destination_name,
            flags,
            0o600,
            dir_fd=destination_directory_fd,
        )
        try:
            remaining = memoryview(competing_payload)
            while remaining:
                written = os.write(descriptor, remaining)
                assert written > 0
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(destination_directory_fd)
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination_name)

    monkeypatch.setattr(ledger_module, "_rename_noreplace", inject_destination)
    destination = ledger.root / filename
    if race_result == "exact":
        stored = ledger.publish_object(payload=payload)
        assert stored.payload == payload
        assert stored.sha256 == digest
        competing_inode = destination.stat().st_ino
        assert ledger.publish_object(payload=payload).sha256 == digest
        assert destination.stat().st_ino == competing_inode
    else:
        with pytest.raises(Stage2BOfflineLedgerConflictError) as caught:
            ledger.publish_object(payload=payload)
        assert caught.value.code == "LEDGER_OBJECT_CORRUPT"
        competing_inode = destination.stat().st_ino
        with pytest.raises(Stage2BOfflineLedgerConflictError) as retried:
            ledger.publish_object(payload=payload)
        assert retried.value.code == "LEDGER_OBJECT_CORRUPT"
        assert destination.stat().st_ino == competing_inode
        assert destination.read_bytes() == competing_payload
    assert injected
    destination_stat = destination.stat()
    assert stat.S_IMODE(destination_stat.st_mode) == 0o400
    assert destination_stat.st_nlink == 1
    assert not any(path.name.startswith(".tmp-") for path in ledger.root.iterdir())


@pytest.mark.parametrize("fault", ["write", "file_fsync", "fchmod", "rename"])
def test_pre_rename_publication_failures_leave_no_object_or_temp(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    ledger = _initialize(tmp_path / fault, ledger_case)
    before = {path.name for path in ledger.root.iterdir()}
    if fault == "write":
        monkeypatch.setattr(ledger_module.os, "write", lambda *_: 0)
    elif fault == "file_fsync":
        real_fsync = ledger_module.os.fsync
        failed = False

        def fail_first_regular_fsync(descriptor: int) -> None:
            nonlocal failed
            if not failed and stat.S_ISREG(os.fstat(descriptor).st_mode):
                failed = True
                raise OSError("injected file fsync failure")
            real_fsync(descriptor)

        monkeypatch.setattr(ledger_module.os, "fsync", fail_first_regular_fsync)
    elif fault == "fchmod":
        monkeypatch.setattr(
            ledger_module.os,
            "fchmod",
            lambda *_: (_ for _ in ()).throw(OSError("injected fchmod failure")),
        )
    else:
        monkeypatch.setattr(
            ledger_module,
            "_rename_noreplace",
            lambda *_: (_ for _ in ()).throw(OSError("injected rename failure")),
        )

    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.publish_object(payload=f"fault-{fault}".encode())
    assert caught.value.code == "LEDGER_IO_FAILED"
    assert {path.name for path in ledger.root.iterdir()} == before
    assert not any(path.name.startswith(".tmp-") for path in ledger.root.iterdir())


def test_post_rename_readback_failure_reports_unknown_and_keeps_final_object(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    payload = b"post-rename-readback"
    digest = _sha256(payload)
    filename = Stage2BOfflineLedger.object_filename(digest)
    real_read = ledger._read_from_root
    target_reads = 0

    def injected_read(directory_fd: int, observed_filename: str):
        nonlocal target_reads
        if observed_filename == filename:
            target_reads += 1
            if target_reads == 2:
                raise Stage2BOfflineLedgerError(
                    "LEDGER_OBJECT_CORRUPT",
                    "injected stable read failure",
                )
        return real_read(directory_fd, observed_filename)

    monkeypatch.setattr(ledger, "_read_from_root", injected_read)
    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.publish_object(payload=payload)
    assert caught.value.code == "LEDGER_PUBLICATION_OUTCOME_UNKNOWN"
    monkeypatch.setattr(ledger, "_read_from_root", real_read)
    assert ledger.read_object(digest).payload == payload


def test_scan_rejects_two_individually_valid_genesis_forks_at_same_sequence(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    main = _initialize(tmp_path / "main", ledger_case)
    main_intent = _prepare(main, ledger_case)
    _commit(main, ledger_case, main_intent)

    fork = _initialize(tmp_path / "fork", ledger_case)
    fork_id = "fork-transaction-1"
    fork_nonce = "fork-nonce-1"
    fork_operation = json.loads(ledger_case.operation_bytes)
    fork_operation["transaction_id"] = fork_id
    fork_operation["transaction_nonce"] = fork_nonce
    fork_operation_bytes = _canonical(fork_operation)
    fork_identity = replace(
        _operation_identity(ledger_case),
        transaction_id=fork_id,
        transaction_nonce=fork_nonce,
        operation_payload_sha256=_sha256(fork_operation_bytes),
    )
    stored_identity = fork.publish_operation_identity(
        identity=fork_identity,
        operation_payload=fork_operation_bytes,
    )
    fork_intent = Stage2BLedgerIntent(
        ledger_id=_LEDGER_ID,
        ledger_sequence=0,
        transaction_id=fork_id,
        transaction_kind=Stage2BTransactionKind.PAIRED_COMMIT,
        transaction_nonce=fork_nonce,
        writer_fence_id=_FENCE_ID,
        staging_variant=CACHStagingVariant.CACH_A,
        episode_id_before=ledger_case.genesis_state.episode_id,
        episode_epoch_before=ledger_case.genesis_state.episode_epoch,
        revision_before=ledger_case.genesis_state.revision,
        state_manifest_before=ledger_case.genesis_state.state_manifest_digest,
        input_identity_sha256=stored_identity.sha256,
        previous_decision_sha256=None,
    )
    fork.prepare(fork_intent)
    fork.abort(
        intent=fork_intent,
        abort_code=Stage2BAbortCode.STAGING_FAILED,
        failure_point=Stage2BFailurePoint.STAGING,
    )

    for source in fork.root.iterdir():
        if source.name == "LEDGER.json":
            continue
        destination = main.root / source.name
        if destination.exists():
            assert destination.read_bytes() == source.read_bytes()
            continue
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o400)

    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        main.scan_valid_chain()
    assert caught.value.code == "LEDGER_DECISION_CONFLICT"
    assert "duplicate decision sequence" in str(caught.value)


@pytest.mark.parametrize("record_kind", ["intent", "decision", "object"])
def test_scan_rejects_record_path_hash_mismatch(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    record_kind: str,
) -> None:
    ledger = _initialize(tmp_path / record_kind, ledger_case)
    intent = _prepare(ledger, ledger_case)
    decision = ledger.abort(
        intent=intent,
        abort_code=Stage2BAbortCode.STAGING_FAILED,
        failure_point=Stage2BFailurePoint.STAGING,
    )
    if record_kind == "intent":
        source = ledger.root / Stage2BOfflineLedger.intent_filename(
            intent.transaction_id
        )
        alias = ledger.root / Stage2BOfflineLedger.intent_filename("alias-intent")
    elif record_kind == "decision":
        source = decision.path
        alias = ledger.root / Stage2BOfflineLedger.decision_filename("alias-decision")
    else:
        source = ledger.root / Stage2BOfflineLedger.object_filename(
            ledger.identity.genesis_snapshot_manifest_sha256
        )
        alias = ledger.root / Stage2BOfflineLedger.object_filename("0" * 64)
    alias.write_bytes(source.read_bytes())
    alias.chmod(0o400)

    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.scan_valid_chain()
    assert caught.value.code == "LEDGER_OBJECT_CORRUPT"


def test_scan_rejects_canonical_abort_that_advances_state(
    tmp_path: Path,
    ledger_case: _LedgerCase,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    intent = _prepare(ledger, ledger_case)
    decision = ledger.abort(
        intent=intent,
        abort_code=Stage2BAbortCode.STAGING_FAILED,
        failure_point=Stage2BFailurePoint.STAGING,
    )
    record = json.loads(decision.payload)
    body = record["decision_body"]
    body["revision_after"] += 1
    record["decision_body_sha256"] = _sha256(_canonical(body))
    decision.path.chmod(0o600)
    decision.path.write_bytes(_canonical(record))
    decision.path.chmod(0o400)

    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.scan_valid_chain()
    assert caught.value.code == "LEDGER_TRANSITION_INVALID"
    assert "aborted decision advances state" in str(caught.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "gap",
        "duplicate_sequence",
        "predecessor",
        "fence",
        "stale_state",
        "body_hash",
    ],
)
def test_scan_rejects_tampered_decision_chain(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    mutation: str,
) -> None:
    ledger = _initialize(tmp_path / mutation, ledger_case)
    commit_intent = _prepare(ledger, ledger_case)
    committed = _commit(ledger, ledger_case, commit_intent)
    reset_intent = _prepare_reset(
        ledger,
        ledger_case,
        previous_decision_sha256=committed.sha256,
    )
    reset = ledger.commit(
        intent=reset_intent,
        snapshot=ledger_case.reset_snapshot,
        success_receipt_bytes=ledger_case.reset_receipt_bytes,
        episode_id_after=ledger_case.reset_state.episode_id,
        episode_epoch_after=ledger_case.reset_state.episode_epoch,
        revision_after=ledger_case.reset_state.revision,
        state_manifest_after=ledger_case.reset_state.state_manifest_digest,
    )
    record = json.loads(reset.payload)
    body = record["decision_body"]
    if mutation == "gap":
        body["ledger_sequence"] = 2
    elif mutation == "duplicate_sequence":
        body["ledger_sequence"] = 0
    elif mutation == "predecessor":
        body["previous_decision_sha256"] = "0" * 64
    elif mutation == "fence":
        body["writer_fence_id"] = "other-writer-fence"
    elif mutation == "stale_state":
        body["state_manifest_before"] = ledger_case.genesis_state.state_manifest_digest
    else:
        record["decision_body_sha256"] = "0" * 64
    if mutation != "body_hash":
        record["decision_body_sha256"] = _sha256(_canonical(body))
    reset.path.chmod(0o600)
    reset.path.write_bytes(_canonical(record))
    reset.path.chmod(0o400)

    with pytest.raises(Stage2BOfflineLedgerError):
        ledger.scan_valid_chain()


def test_chain_order_never_depends_on_directory_enumeration(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    commit_intent = _prepare(ledger, ledger_case)
    committed = _commit(ledger, ledger_case, commit_intent)
    reset_intent = _prepare_reset(
        ledger,
        ledger_case,
        previous_decision_sha256=committed.sha256,
    )
    reset = ledger.commit(
        intent=reset_intent,
        snapshot=ledger_case.reset_snapshot,
        success_receipt_bytes=ledger_case.reset_receipt_bytes,
        episode_id_after=ledger_case.reset_state.episode_id,
        episode_epoch_after=ledger_case.reset_state.episode_epoch,
        revision_after=ledger_case.reset_state.revision,
        state_manifest_after=ledger_case.reset_state.state_manifest_digest,
    )
    real_listdir = ledger_module.os.listdir

    def reversed_listdir(path):
        return list(reversed(real_listdir(path)))

    monkeypatch.setattr(ledger_module.os, "listdir", reversed_listdir)
    assert [item.sha256 for item in ledger.scan_valid_chain()] == [
        committed.sha256,
        reset.sha256,
    ]


@pytest.mark.parametrize(
    "missing_reference",
    ["receipt", "snapshot_manifest", "tensor_blob", "substituted_receipt"],
)
def test_scan_rejects_missing_or_substituted_referenced_objects(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    missing_reference: str,
) -> None:
    ledger = _initialize(tmp_path / missing_reference, ledger_case)
    intent = _prepare(ledger, ledger_case)
    decision = _commit(ledger, ledger_case, intent)
    body = decision.decision.decision_body
    if missing_reference in {"receipt", "substituted_receipt"}:
        digest = body.success_or_abort_receipt_sha256
    elif missing_reference == "snapshot_manifest":
        assert body.snapshot_manifest_sha256 is not None
        digest = body.snapshot_manifest_sha256
    else:
        digest = ledger_case.committed_snapshot.blobs[0].sha256
    path = ledger.root / Stage2BOfflineLedger.object_filename(digest)
    if missing_reference == "substituted_receipt":
        path.chmod(0o600)
        path.write_bytes(b"substituted")
        path.chmod(0o400)
    else:
        path.unlink()

    with pytest.raises(Stage2BOfflineLedgerError):
        ledger.scan_valid_chain()


@pytest.mark.parametrize(
    "receipt_mutation",
    ["nonce", "new_episode", "layout", "aborted_pending"],
)
def test_reset_receipt_is_fully_cross_bound_before_terminal_publication(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    receipt_mutation: str,
) -> None:
    ledger = _initialize(tmp_path / receipt_mutation, ledger_case)
    commit_intent = _prepare(ledger, ledger_case)
    committed = _commit(ledger, ledger_case, commit_intent)
    reset_intent = _prepare_reset(
        ledger,
        ledger_case,
        previous_decision_sha256=committed.sha256,
    )
    replacements: dict[str, object]
    if receipt_mutation == "nonce":
        replacements = {"transaction_nonce": "wrong-reset-nonce"}
    elif receipt_mutation == "new_episode":
        replacements = {"new_episode_id": "wrong-reset-episode"}
    elif receipt_mutation == "layout":
        replacements = {"layout_instance_digest": "e" * 64}
    else:
        replacements = {"aborted_pending_commit_id": "other-pending-commit"}
    wrong_receipt = replace(ledger_case.reset_receipt, **replacements)

    with pytest.raises(Stage2BOfflineLedgerError, match="receipt"):
        ledger.commit(
            intent=reset_intent,
            snapshot=ledger_case.reset_snapshot,
            success_receipt_bytes=wrong_receipt.canonical_bytes,
            episode_id_after=ledger_case.reset_state.episode_id,
            episode_epoch_after=ledger_case.reset_state.episode_epoch,
            revision_after=ledger_case.reset_state.revision,
            state_manifest_after=ledger_case.reset_state.state_manifest_digest,
        )
    assert ledger.read_decision(reset_intent.transaction_id) is None


def test_same_process_scan_cannot_observe_inflight_temporary_publication(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    real_rename = ledger_module._rename_noreplace
    rename_entered = threading.Event()
    allow_rename = threading.Event()
    scan_finished = threading.Event()
    failures: list[BaseException] = []

    def blocking_rename(*args) -> None:
        rename_entered.set()
        if not allow_rename.wait(timeout=2):
            raise AssertionError("test did not release rename boundary")
        real_rename(*args)

    def publish() -> None:
        try:
            ledger.publish_object(payload=b"concurrent-publication")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def scan() -> None:
        try:
            ledger.scan_valid_chain()
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            scan_finished.set()

    monkeypatch.setattr(ledger_module, "_rename_noreplace", blocking_rename)
    publisher = threading.Thread(target=publish)
    scanner = threading.Thread(target=scan)
    publisher.start()
    assert rename_entered.wait(timeout=2)
    scanner.start()
    assert not scan_finished.wait(timeout=0.05)
    allow_rename.set()
    publisher.join(timeout=2)
    scanner.join(timeout=2)

    assert not publisher.is_alive()
    assert not scanner.is_alive()
    assert failures == []
    assert scan_finished.is_set()


def test_fragmented_write_and_read_complete_exactly(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    real_write = ledger_module.os.write
    real_read = ledger_module.os.read

    def fragmented_write(descriptor: int, payload) -> int:
        return real_write(descriptor, payload[:3])

    def fragmented_read(descriptor: int, size: int) -> bytes:
        return real_read(descriptor, min(size, 3))

    monkeypatch.setattr(ledger_module.os, "write", fragmented_write)
    monkeypatch.setattr(ledger_module.os, "read", fragmented_read)
    payload = b"fragmented-write-and-read"
    stored = ledger.publish_object(payload=payload)
    assert stored.payload == payload
    assert ledger.read_object(_sha256(payload)).payload == payload


def test_premature_read_eof_fails_before_rename_and_cleans_temp(
    tmp_path: Path,
    ledger_case: _LedgerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _initialize(tmp_path / "ledger", ledger_case)
    before = {path.name for path in ledger.root.iterdir()}
    real_read = ledger_module.os.read
    reads = 0

    def premature_eof(descriptor: int, size: int) -> bytes:
        nonlocal reads
        reads += 1
        if reads == 1:
            return real_read(descriptor, min(size, 3))
        return b""

    monkeypatch.setattr(ledger_module.os, "read", premature_eof)
    with pytest.raises(Stage2BOfflineLedgerError) as caught:
        ledger.publish_object(payload=b"must-not-publish")
    assert caught.value.code == "LEDGER_IO_FAILED"
    assert {path.name for path in ledger.root.iterdir()} == before
