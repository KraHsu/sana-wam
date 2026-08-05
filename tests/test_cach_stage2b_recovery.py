"""CPU/synthetic contracts for bounded Stage-2B L3 restart recovery.

These tests use only private tmp_path ledgers.  They do not admit a production
filesystem, multiple writers, a model, CUDA, training, evaluation, deployment,
capture, or project-wide Stage 3.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sana_wam.cach import stage2b_offline_ledger as offline
from sana_wam.cach.stage2b_offline_ledger import (
    Stage2BAbortCode,
    Stage2BAbortReceipt,
    Stage2BDecisionBody,
    Stage2BDecisionRecord,
    Stage2BDecisionResult,
    Stage2BFailurePoint,
    Stage2BOfflineLedger,
    Stage2BOfflineLedgerError,
)
from sana_wam.cach.stage2b_state_snapshot import encode_stage2b_state_snapshot
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_chunk_layout import ChunkActionLayout
from sana_wam.model.video_backbone.sana import hybrid_cache as hc
from sana_wam.model.video_backbone.sana import (
    hybrid_cache_stage2b_ledgered as l2b,
)
from sana_wam.model.video_backbone.sana import (
    hybrid_cache_stage2b_recovery as l3,
)
from tests import test_cach_stage2b_ledgered_manager as l2b_cases


_LEDGER_ID = "bounded-l3-recovery-ledger"
_WRITER_FENCE_ID = "bounded-l3-recovery-writer"


@dataclass
class _RecoveryCase:
    layout: ChunkActionLayout
    reset_layout: ChunkActionLayout
    registry: hc.LayerRegistry
    genesis_snapshot: object
    root: Path
    _ledger: Stage2BOfflineLedger | None

    @property
    def ledger(self) -> Stage2BOfflineLedger:
        if self._ledger is None:
            raise AssertionError("recovery case has no live ledger binding")
        return self._ledger

    def retire_ledger(self) -> None:
        self._ledger = None
        gc.collect()

    def fresh_manager(self) -> l2b.Stage2BLedgeredHybridCacheManager:
        return l2b.Stage2BLedgeredHybridCacheManager.for_synthetic_tests(
            registry=self.registry,
            episode_id=l2b_cases._EPISODE_ID,
            layout=self.layout,
            ledger=self.ledger,
            staging_variant=CACHStagingVariant.CACH_A,
        )


def _case(tmp_path: Path) -> _RecoveryCase:
    layout = l2b_cases._layout(episode_id=l2b_cases._EPISODE_ID)
    reset_layout = l2b_cases._layout(episode_id=l2b_cases._RESET_EPISODE_ID)
    registry = l2b_cases._registry()
    genesis = l2b_cases.h2.Stage2BTemporalState.empty(
        episode_id=l2b_cases._EPISODE_ID,
        episode_epoch=0,
        layout=layout,
        registry=registry,
        staging_variant=CACHStagingVariant.CACH_A,
    )
    genesis_snapshot = encode_stage2b_state_snapshot(
        state=genesis,
        expected_layout=layout,
        expected_registry=registry,
    )
    root = l2b_cases._secure_root(tmp_path / "ledger")
    ledger = Stage2BOfflineLedger.initialize(
        root=root,
        ledger_id=_LEDGER_ID,
        staging_variant=CACHStagingVariant.CACH_A,
        writer_fence_id=_WRITER_FENCE_ID,
        expected_layout=layout,
        expected_registry=registry,
        genesis_snapshot=genesis_snapshot,
        additional_expected_layouts=(reset_layout,),
    )
    return _RecoveryCase(
        layout=layout,
        reset_layout=reset_layout,
        registry=registry,
        genesis_snapshot=genesis_snapshot,
        root=root,
        _ledger=ledger,
    )


def _episode_request(
    *,
    layout: ChunkActionLayout,
    episode_id: str,
    episode_epoch: int,
    chunk_id: int,
    expected_revision: int,
    commit_id: str,
    transaction_nonce: str,
):
    request = l2b_cases._request(
        layout,
        chunk_id=chunk_id,
        commit_id=commit_id,
        transaction_nonce=transaction_nonce,
    )
    chunk = layout.chunks[chunk_id]
    return replace(
        request,
        expected_episode_id=episode_id,
        expected_episode_epoch=episode_epoch,
        expected_revision=expected_revision,
        content_time=hc.ContentTime.from_layout(
            episode_id=episode_id,
            episode_epoch=episode_epoch,
            layout=layout,
            chunk=chunk,
        ),
        proof=replace(
            request.proof,
            dataset_episode_id=episode_id,
            dataset_row_identity=f"{episode_id}:chunk-{chunk_id}",
        ),
    )


def _root_fingerprint(
    ledger: Stage2BOfflineLedger,
) -> tuple[tuple[str, int, int, int, int, int, int, bytes], ...]:
    fingerprint = []
    for path in sorted(ledger.root.iterdir(), key=lambda item: item.name):
        stat = path.lstat()
        fingerprint.append(
            (
                path.name,
                stat.st_mode,
                stat.st_uid,
                stat.st_gid,
                stat.st_nlink,
                stat.st_dev,
                stat.st_ino,
                path.read_bytes(),
            )
        )
    return tuple(fingerprint)


def _reopen(case: _RecoveryCase) -> Stage2BOfflineLedger:
    """Bind a fresh ledger object to the immutable restart evidence."""

    if case._ledger is not None:
        raise AssertionError("the previous ledger object is still live")
    reopened = Stage2BOfflineLedger.reopen(
        root=case.root,
        expected_ledger_id=_LEDGER_ID,
        expected_staging_variant=CACHStagingVariant.CACH_A,
        expected_writer_fence_id=_WRITER_FENCE_ID,
        expected_layout=case.layout,
        expected_registry=case.registry,
        expected_genesis_snapshot=case.genesis_snapshot,
        additional_expected_layouts=(case.reset_layout,),
    )
    case._ledger = reopened
    return reopened


def _recover(
    case: _RecoveryCase,
    *,
    ledger: Stage2BOfflineLedger | None = None,
) -> l3.Stage2BRecoveredLedgeredHybridCacheManager:
    restarted_ledger = _reopen(case) if ledger is None else ledger
    return l3.Stage2BRecoveredLedgeredHybridCacheManager.for_synthetic_recovery(
        registry=case.registry,
        ledger=restarted_ledger,
    )


def test_genesis_recovery_is_exact_model_free_and_non_authorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    before = _root_fingerprint(case.ledger)
    expected_state = case.ledger._genesis_state.to_manifest()
    forbidden_calls = 0

    def forbid_commit_or_staging(*_args, **_kwargs):
        nonlocal forbidden_calls
        forbidden_calls += 1
        raise AssertionError("recovery entered a commit/staging surface")

    monkeypatch.setattr(
        l2b._LedgerSeamedStage2BHybridCacheManager,
        "commit_paired",
        forbid_commit_or_staging,
    )
    monkeypatch.setattr(
        l2b._CaptureBridge,
        "wrap",
        forbid_commit_or_staging,
    )

    with pytest.raises(hc.CacheContractError) as denied:
        l3.Stage2BRecoveredLedgeredHybridCacheManager.recover_from_ledger(
            registry=case.registry,
            ledger=case.ledger,
            synthetic_test_only=False,
        )
    assert denied.value.code == "L3_CPU_SYNTHETIC_ONLY"
    assert _root_fingerprint(case.ledger) == before

    case.retire_ledger()
    recovered = _recover(case)
    assert recovered.state.to_manifest() == expected_state
    assert recovered.state.state_manifest_digest == (
        case.ledger.identity.genesis_state_manifest
    )
    assert _root_fingerprint(case.ledger) == before
    assert forbidden_calls == 0
    inner = l2b_cases._inner_manager(recovered)
    assert recovered._phase == l2b._READY
    assert recovered._active is None
    assert recovered._poisoned is False
    assert recovered._publisher_adapter._armed_token is None
    assert recovered._completed_commits == {}
    assert recovered._completed_resets == {}
    assert inner._pending is None
    assert inner._publication_token is None
    assert inner._poisoned is False
    assert inner._completed == {}
    assert inner._completed_resets == {}
    summary = recovered.recovery_summary
    assert summary.decision_count == 0
    assert summary.last_decision_sha256 is None
    assert summary.closed_process_exit_transaction_id is None
    assert summary.unreferenced_content_object_sha256s == ()
    assert summary.model_executed is False
    assert summary.recovery_admitted is False
    assert summary.filesystem_admitted is False
    assert summary.scientific_eligible is False

    recovery_parameters = inspect.signature(
        l3.Stage2BRecoveredLedgeredHybridCacheManager.recover_from_ledger
    ).parameters
    factory_parameters = inspect.signature(
        l3.Stage2BRecoveredLedgeredHybridCacheManager.for_synthetic_recovery
    ).parameters
    assert tuple(recovery_parameters) == (
        "registry",
        "ledger",
        "synthetic_test_only",
    )
    assert tuple(factory_parameters) == ("registry", "ledger")
    forbidden_names = {"model", "stager", "stage_callback", "request"}
    assert forbidden_names.isdisjoint(recovery_parameters)
    assert forbidden_names.isdisjoint(factory_parameters)


def test_two_commits_recover_exact_maps_retry_and_continue_with_a_new_id(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    old = case.fresh_manager()
    request0 = l2b_cases._request(case.layout, chunk_id=0)
    receipt0 = old.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request0,
        stage_callback=l2b_cases._layer_payloads,
    )
    request1 = l2b_cases._request(case.layout, chunk_id=1)
    receipt1 = old.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request1,
        stage_callback=l2b_cases._layer_payloads,
    )
    committed_state = old.state
    intents, _, ordered, pending = case.ledger._scan_records()
    assert not pending
    assert len(ordered) == 2
    terminal0 = case.ledger.read_decision(request0.commit_id)
    terminal1 = case.ledger.read_decision(request1.commit_id)
    assert terminal0 is not None
    assert terminal1 is not None
    identity0, _ = case.ledger._validate_intent_input(
        intents[request0.commit_id].intent
    )
    identity1, _ = case.ledger._validate_intent_input(
        intents[request1.commit_id].intent
    )
    root_after_commit = _root_fingerprint(case.ledger)
    del old
    gc.collect()
    case.retire_ledger()

    recovered = _recover(case)
    assert recovered.state.to_manifest() == committed_state.to_manifest()
    assert recovered.recovery_summary.decision_count == 2
    assert recovered.recovery_summary.last_decision_sha256 == terminal1.sha256
    inner = l2b_cases._inner_manager(recovered)
    assert recovered._completed_commits[request0.commit_id] == (
        identity0.operation_payload_sha256,
        receipt0,
        terminal0.sha256,
    )
    assert recovered._completed_commits[request1.commit_id] == (
        identity1.operation_payload_sha256,
        receipt1,
        terminal1.sha256,
    )
    assert inner._completed[request0.commit_id] == (
        receipt0.paired_payload_digest,
        receipt0,
    )
    assert inner._completed[request1.commit_id] == (
        receipt1.paired_payload_digest,
        receipt1,
    )

    stage_calls = 0

    def must_not_stage(_context):
        nonlocal stage_calls
        stage_calls += 1
        raise AssertionError("exact recovered retry staged")

    retried = recovered.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request0,
        stage_callback=must_not_stage,
    )
    assert retried == receipt0
    assert stage_calls == 0
    assert _root_fingerprint(case.ledger) == root_after_commit

    continued_case = _case(tmp_path / "continued")
    continued_old = continued_case.fresh_manager()
    continued_old.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=l2b_cases._request(continued_case.layout, chunk_id=0),
        stage_callback=l2b_cases._layer_payloads,
    )
    one_commit_state = continued_old.state.to_manifest()
    del continued_old
    gc.collect()
    continued_case.retire_ledger()
    continued_recovered = _recover(continued_case)
    assert continued_recovered.state.to_manifest() == one_commit_state
    continued_calls = 0

    def continue_stage(context):
        nonlocal continued_calls
        continued_calls += 1
        return l2b_cases._layer_payloads(context)

    request1_after_restart = l2b_cases._request(
        continued_case.layout,
        chunk_id=1,
    )
    continued = continued_recovered.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request1_after_restart,
        stage_callback=continue_stage,
    )
    assert continued.commit_id == request1_after_restart.commit_id
    assert continued_calls == 1
    assert continued_recovered.state.revision == 2
    assert len(continued_case.ledger.scan_valid_chain()) == 2


def test_terminal_durable_before_cas_recovers_authoritative_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    old_holder = [case.fresh_manager()]
    request = l2b_cases._request(case.layout, chunk_id=0)
    predecessor = old_holder[0].state.state_manifest_digest
    adapter = l2b_cases._publisher_adapter(old_holder[0])
    publish_holder = [adapter.publish_exclusive]

    def publish_then_raise(*, receipt_id, payload, expected_sha256):
        publish_holder[0](
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=expected_sha256,
        )
        assert case.ledger.read_decision(receipt_id) is not None
        assert (
            l2b_cases._inner_manager(old_holder[0])._state.state_manifest_digest
            == predecessor
        )
        raise RuntimeError("simulated process exit after terminal fsync")

    monkeypatch.setattr(adapter, "publish_exclusive", publish_then_raise)
    with pytest.raises(hc.CacheContractError):
        old_holder[0].commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=l2b_cases._layer_payloads,
        )
    terminal = case.ledger.read_decision(request.commit_id)
    assert terminal is not None
    body = terminal.decision.decision_body
    assert body.result is Stage2BDecisionResult.COMMITTED
    assert body.state_manifest_after != predecessor
    assert (
        l2b_cases._inner_manager(old_holder[0])._state.state_manifest_digest
        == predecessor
    )
    monkeypatch.undo()
    old_holder.clear()
    publish_holder.clear()
    del publish_then_raise, adapter
    gc.collect()
    case.retire_ledger()

    recovered = _recover(case)
    assert recovered.state.state_manifest_digest == body.state_manifest_after
    assert recovered.state.revision == 1


class _SimulatedProcessExit(BaseException):
    pass


def _expected_process_exit_terminal(intent_entry):
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
    decision = Stage2BDecisionRecord.from_body(
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
    return receipt, decision


def test_pending_is_closed_as_process_exit_and_inert_debris_is_only_diagnostic(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    old = case.fresh_manager()
    request = l2b_cases._request(
        case.layout,
        chunk_id=0,
        commit_id="pending-process-exit",
        transaction_nonce="pending-process-exit-nonce",
    )
    inert_payloads = (
        b"inert-content-object-before-terminal-a",
        b"inert-content-object-before-terminal-z",
    )
    inert_sha256s: list[str] = []

    def crash_during_staging(_context):
        inert_sha256s.extend(
            case.ledger.publish_object(payload=payload).sha256
            for payload in inert_payloads
        )
        raise _SimulatedProcessExit

    with pytest.raises(_SimulatedProcessExit):
        old.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=crash_during_staging,
        )
    assert case.ledger.read_decision(request.commit_id) is None
    intents, _, _, pending = case.ledger._scan_records()
    assert len(pending) == 1
    expected_abort_receipt, expected_terminal = _expected_process_exit_terminal(
        intents[request.commit_id]
    )
    del old
    gc.collect()
    case.retire_ledger()

    recovered = _recover(case)
    terminal = case.ledger.read_decision(request.commit_id)
    assert terminal is not None
    assert terminal.sha256 == expected_terminal.decision_sha256
    assert terminal.payload == expected_terminal.canonical_bytes
    assert terminal.decision.decision_body.result is Stage2BDecisionResult.ABORTED
    assert terminal.decision.decision_body.abort_code is Stage2BAbortCode.PROCESS_EXIT
    assert terminal.decision.decision_body.snapshot_manifest_sha256 is None
    assert (
        terminal.decision.decision_body.state_manifest_before
        == terminal.decision.decision_body.state_manifest_after
    )
    abort_receipt = case.ledger.read_object(
        terminal.decision.decision_body.success_or_abort_receipt_sha256
    ).payload
    assert abort_receipt == expected_abort_receipt.canonical_bytes
    assert recovered.state.state_manifest_digest == (
        case.ledger.identity.genesis_state_manifest
    )
    assert recovered.recovery_summary.closed_process_exit_transaction_id == (
        request.commit_id
    )
    assert (
        recovered.recovery_summary.unreferenced_content_object_sha256s
        == tuple(sorted(inert_sha256s))
    )
    assert len(inert_sha256s) == 2
    assert tuple(inert_sha256s) != tuple(sorted(inert_sha256s))
    for digest, payload in zip(inert_sha256s, inert_payloads, strict=True):
        inert_path = case.ledger.root / case.ledger.object_filename(digest)
        assert inert_path.read_bytes() == payload

    root_after_abort = _root_fingerprint(case.ledger)
    staged = False

    def tombstone_must_not_stage(_context):
        nonlocal staged
        staged = True
        raise AssertionError("aborted transaction ID reached the stager")

    with pytest.raises(hc.CacheContractError) as reused_abort:
        recovered.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=tombstone_must_not_stage,
        )
    assert reused_abort.value.code == "COMMIT_DUPLICATE_CONFLICT"
    assert staged is False
    assert _root_fingerprint(case.ledger) == root_after_abort

    replacement = l2b_cases._request(
        case.layout,
        chunk_id=0,
        commit_id="post-process-exit-commit",
        transaction_nonce="post-process-exit-commit-nonce",
    )
    replacement_receipt = recovered.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=replacement,
        stage_callback=l2b_cases._layer_payloads,
    )
    assert replacement_receipt.commit_id == replacement.commit_id
    assert recovered.state.revision == 1
    replacement_terminal = case.ledger.read_decision(replacement.commit_id)
    assert replacement_terminal is not None

    root_after_recovery = _root_fingerprint(case.ledger)
    del recovered
    gc.collect()
    case.retire_ledger()
    recovered_again = _recover(case)
    assert recovered_again.recovery_summary.closed_process_exit_transaction_id is None
    assert recovered_again.recovery_summary.decision_count == 2
    assert (
        recovered_again.recovery_summary.last_decision_sha256
        == replacement_terminal.sha256
    )
    assert recovered_again.state.revision == 1
    assert (
        recovered_again.recovery_summary.unreferenced_content_object_sha256s
        == tuple(sorted(inert_sha256s))
    )
    assert _root_fingerprint(case.ledger) == root_after_recovery


def test_process_exit_unknown_outcome_accepts_only_exact_and_absent_returns_no_facade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_case = _case(tmp_path / "exact")
    exact_old = exact_case.fresh_manager()
    exact_request = l2b_cases._request(
        exact_case.layout,
        chunk_id=0,
        commit_id="pending-exact-abort",
        transaction_nonce="pending-exact-abort-nonce",
    )

    def simulated_exit(context):
        raise _SimulatedProcessExit

    with pytest.raises(_SimulatedProcessExit):
        exact_old.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=exact_request,
            stage_callback=simulated_exit,
        )
    exact_intents, _, _, _ = exact_case.ledger._scan_records()
    _, exact_expected = _expected_process_exit_terminal(
        exact_intents[exact_request.commit_id]
    )
    del exact_old
    gc.collect()
    exact_case.retire_ledger()
    exact_reopened = _reopen(exact_case)
    real_abort = exact_reopened.abort
    real_reconcile = exact_reopened.reconcile_terminal
    exact_reconciliations: list[tuple[str, str]] = []

    def abort_then_raise(**kwargs):
        real_abort(**kwargs)
        raise OSError("simulated lost abort acknowledgement")

    def observe_exact_reconcile(*, transaction_id, expected_record_sha256):
        exact_reconciliations.append((transaction_id, expected_record_sha256))
        return real_reconcile(
            transaction_id=transaction_id,
            expected_record_sha256=expected_record_sha256,
        )

    monkeypatch.setattr(exact_reopened, "abort", abort_then_raise)
    monkeypatch.setattr(
        exact_reopened,
        "reconcile_terminal",
        observe_exact_reconcile,
    )
    exact_recovered = _recover(exact_case, ledger=exact_reopened)
    assert exact_recovered.recovery_summary.closed_process_exit_transaction_id == (
        exact_request.commit_id
    )
    assert exact_reconciliations == [
        (exact_request.commit_id, exact_expected.decision_sha256)
    ]

    absent_case = _case(tmp_path / "absent")
    absent_old = absent_case.fresh_manager()
    absent_request = l2b_cases._request(
        absent_case.layout,
        chunk_id=0,
        commit_id="pending-absent-abort",
        transaction_nonce="pending-absent-abort-nonce",
    )
    with pytest.raises(_SimulatedProcessExit):
        absent_old.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=absent_request,
            stage_callback=simulated_exit,
        )
    del absent_old
    gc.collect()
    absent_case.retire_ledger()
    absent_reopened = _reopen(absent_case)
    absent_reconcile_holder = [absent_reopened.reconcile_terminal]
    absent_intents, _, _, _ = absent_reopened._scan_records()
    _, absent_expected = _expected_process_exit_terminal(
        absent_intents[absent_request.commit_id]
    )
    absent_reconciliations: list[tuple[str, str]] = []

    def abort_before_publication(**kwargs):
        raise OSError("simulated abort absence")

    def observe_absent_reconcile(*, transaction_id, expected_record_sha256):
        absent_reconciliations.append((transaction_id, expected_record_sha256))
        return absent_reconcile_holder[0](
            transaction_id=transaction_id,
            expected_record_sha256=expected_record_sha256,
        )

    monkeypatch.setattr(absent_reopened, "abort", abort_before_publication)
    monkeypatch.setattr(
        absent_reopened,
        "reconcile_terminal",
        observe_absent_reconcile,
    )
    before = _root_fingerprint(absent_case.ledger)
    with pytest.raises(hc.CacheContractError):
        _recover(absent_case, ledger=absent_reopened)
    assert absent_case.ledger.read_decision(absent_request.commit_id) is None
    assert _root_fingerprint(absent_case.ledger) == before
    assert absent_reconciliations == [
        (absent_request.commit_id, absent_expected.decision_sha256)
    ]

    monkeypatch.undo()
    absent_reconcile_holder.clear()
    del (
        abort_before_publication,
        observe_absent_reconcile,
        absent_reopened,
    )
    absent_case.retire_ledger()
    recovered_later = _recover(absent_case)
    assert recovered_later.recovery_summary.closed_process_exit_transaction_id == (
        absent_request.commit_id
    )


def test_process_exit_unknown_conflict_or_unreadable_never_returns_a_facade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conflict_case = _case(tmp_path / "conflict")
    conflict_old = conflict_case.fresh_manager()
    conflict_request = l2b_cases._request(
        conflict_case.layout,
        chunk_id=0,
        commit_id="pending-conflicting-abort",
        transaction_nonce="pending-conflicting-abort-nonce",
    )

    def simulated_exit(_context):
        raise _SimulatedProcessExit

    with pytest.raises(_SimulatedProcessExit):
        conflict_old.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=conflict_request,
            stage_callback=simulated_exit,
        )
    conflict_intents, _, _, _ = conflict_case.ledger._scan_records()
    _, conflict_expected = _expected_process_exit_terminal(
        conflict_intents[conflict_request.commit_id]
    )
    del conflict_old
    gc.collect()
    conflict_case.retire_ledger()
    conflict_reopened = _reopen(conflict_case)
    conflict_real_abort = conflict_reopened.abort
    conflict_real_reconcile = conflict_reopened.reconcile_terminal
    conflict_reconciliations: list[tuple[str, str]] = []

    def publish_different_abort_then_raise(**kwargs):
        conflict_real_abort(
            intent=kwargs["intent"],
            abort_code=Stage2BAbortCode.STAGING_FAILED,
            failure_point=Stage2BFailurePoint.STAGING,
        )
        raise OSError("simulated conflicting terminal acknowledgement loss")

    def observe_conflict_reconcile(*, transaction_id, expected_record_sha256):
        conflict_reconciliations.append(
            (transaction_id, expected_record_sha256)
        )
        return conflict_real_reconcile(
            transaction_id=transaction_id,
            expected_record_sha256=expected_record_sha256,
        )

    monkeypatch.setattr(
        conflict_reopened,
        "abort",
        publish_different_abort_then_raise,
    )
    monkeypatch.setattr(
        conflict_reopened,
        "reconcile_terminal",
        observe_conflict_reconcile,
    )
    with pytest.raises(hc.CacheContractError) as conflict:
        _recover(conflict_case, ledger=conflict_reopened)
    assert conflict.value.code == "L3_RECOVERY_OUTCOME_UNKNOWN"
    assert conflict_reconciliations == [
        (conflict_request.commit_id, conflict_expected.decision_sha256)
    ]
    conflicting_terminal = conflict_case.ledger.read_decision(
        conflict_request.commit_id
    )
    assert conflicting_terminal is not None
    assert conflicting_terminal.sha256 != conflict_expected.decision_sha256
    assert (
        conflicting_terminal.decision.decision_body.abort_code
        is Stage2BAbortCode.STAGING_FAILED
    )

    monkeypatch.undo()
    unreadable_case = _case(tmp_path / "unreadable")
    unreadable_old = unreadable_case.fresh_manager()
    unreadable_request = l2b_cases._request(
        unreadable_case.layout,
        chunk_id=0,
        commit_id="pending-unreadable-abort",
        transaction_nonce="pending-unreadable-abort-nonce",
    )
    with pytest.raises(_SimulatedProcessExit):
        unreadable_old.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=unreadable_request,
            stage_callback=simulated_exit,
        )
    unreadable_intents, _, _, _ = unreadable_case.ledger._scan_records()
    _, unreadable_expected = _expected_process_exit_terminal(
        unreadable_intents[unreadable_request.commit_id]
    )
    del unreadable_old
    gc.collect()
    unreadable_case.retire_ledger()
    unreadable_reopened = _reopen(unreadable_case)
    unreadable_reconciliations: list[tuple[str, str]] = []

    def fail_abort_before_publication(**_kwargs):
        raise OSError("simulated abort publication failure")

    def unreadable_reconcile(*, transaction_id, expected_record_sha256):
        unreadable_reconciliations.append(
            (transaction_id, expected_record_sha256)
        )
        raise Stage2BOfflineLedgerError(
            "LEDGER_OBJECT_CORRUPT",
            "simulated unreadable terminal",
        )

    monkeypatch.setattr(
        unreadable_reopened,
        "abort",
        fail_abort_before_publication,
    )
    monkeypatch.setattr(
        unreadable_reopened,
        "reconcile_terminal",
        unreadable_reconcile,
    )
    unreadable_before = _root_fingerprint(unreadable_case.ledger)
    with pytest.raises(hc.CacheContractError) as unreadable:
        _recover(unreadable_case, ledger=unreadable_reopened)
    assert unreadable.value.code == "L3_RECOVERY_OUTCOME_UNKNOWN"
    assert unreadable_reconciliations == [
        (unreadable_request.commit_id, unreadable_expected.decision_sha256)
    ]
    assert unreadable_case.ledger.read_decision(unreadable_request.commit_id) is None
    assert _root_fingerprint(unreadable_case.ledger) == unreadable_before


def test_commit_reset_commit_recovers_exact_maps_retries_and_global_tombstones(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    old = case.fresh_manager()
    historical_request = l2b_cases._request(case.layout, chunk_id=0)
    historical_receipt = old.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=historical_request,
        stage_callback=l2b_cases._layer_payloads,
    )
    reset_receipt = old.reset(
        reset_id="recovery-reset",
        transaction_nonce="recovery-reset-nonce",
        new_episode_id=l2b_cases._RESET_EPISODE_ID,
        layout=case.reset_layout,
    )
    initial_intents, _, initial_ordered, initial_pending = (
        case.ledger._scan_records()
    )
    assert not initial_pending
    assert len(initial_ordered) == 2
    historical_identity, _ = case.ledger._validate_intent_input(
        initial_intents[historical_request.commit_id].intent
    )
    reset_identity, _ = case.ledger._validate_intent_input(
        initial_intents["recovery-reset"].intent
    )
    assert (
        reset_identity.layout_instance_digest
        == reset_receipt.layout_instance_digest
    )
    historical_terminal = case.ledger.read_decision(
        historical_request.commit_id
    )
    reset_terminal = case.ledger.read_decision("recovery-reset")
    assert historical_terminal is not None
    assert reset_terminal is not None
    root_after_reset = _root_fingerprint(case.ledger)
    del old
    gc.collect()
    case.retire_ledger()

    recovered = _recover(case)
    state = recovered.state
    assert state.episode_id == l2b_cases._RESET_EPISODE_ID
    assert state.episode_epoch == 1
    assert state.revision == 0
    assert state.layout_instance_digest == case.reset_layout.layout_instance_digest
    inner = l2b_cases._inner_manager(recovered)
    assert recovered._completed_commits[historical_request.commit_id] == (
        historical_identity.operation_payload_sha256,
        historical_receipt,
        historical_terminal.sha256,
    )
    assert inner._completed[historical_request.commit_id] == (
        historical_receipt.paired_payload_digest,
        historical_receipt,
    )
    reset_digest = recovered._reset_retry_digest(
        reset_id="recovery-reset",
        transaction_nonce="recovery-reset-nonce",
        new_episode_id=l2b_cases._RESET_EPISODE_ID,
        layout=case.reset_layout,
    )
    assert recovered._completed_resets["recovery-reset"] == (
        reset_digest,
        reset_receipt,
        reset_terminal.sha256,
    )
    assert inner._completed_resets["recovery-reset"] == (
        reset_digest,
        reset_receipt,
    )

    retried_reset = recovered.reset(
        reset_id="recovery-reset",
        transaction_nonce="recovery-reset-nonce",
        new_episode_id=l2b_cases._RESET_EPISODE_ID,
        layout=case.reset_layout,
    )
    assert retried_reset == reset_receipt
    assert _root_fingerprint(case.ledger) == root_after_reset

    current_request = _episode_request(
        layout=case.reset_layout,
        episode_id=l2b_cases._RESET_EPISODE_ID,
        episode_epoch=1,
        chunk_id=0,
        expected_revision=0,
        commit_id="post-reset-commit",
        transaction_nonce="post-reset-commit-nonce",
    )
    current_receipt = recovered.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=current_request,
        stage_callback=l2b_cases._layer_payloads,
    )
    assert recovered.state.revision == 1
    current_intents, _, current_ordered, current_pending = (
        case.ledger._scan_records()
    )
    assert not current_pending
    assert len(current_ordered) == 3
    current_identity, _ = case.ledger._validate_intent_input(
        current_intents[current_request.commit_id].intent
    )
    current_terminal = case.ledger.read_decision(current_request.commit_id)
    assert current_terminal is not None
    final_root = _root_fingerprint(case.ledger)
    del inner, recovered
    gc.collect()
    case.retire_ledger()

    recovered_final = _recover(case)
    assert recovered_final.state.episode_id == l2b_cases._RESET_EPISODE_ID
    assert recovered_final.state.episode_epoch == 1
    assert recovered_final.state.revision == 1
    assert (
        recovered_final.state.layout_instance_digest
        == case.reset_layout.layout_instance_digest
    )
    final_inner = l2b_cases._inner_manager(recovered_final)
    assert recovered_final._completed_commits[current_request.commit_id] == (
        current_identity.operation_payload_sha256,
        current_receipt,
        current_terminal.sha256,
    )
    assert final_inner._completed[current_request.commit_id] == (
        current_receipt.paired_payload_digest,
        current_receipt,
    )

    stage_calls = 0

    def must_not_stage(_context):
        nonlocal stage_calls
        stage_calls += 1
        raise AssertionError("tombstone or exact retry reached the stager")

    retried_current = recovered_final.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=current_request,
        stage_callback=must_not_stage,
    )
    assert retried_current == current_receipt
    assert stage_calls == 0
    assert _root_fingerprint(case.ledger) == final_root

    reused_historical = _episode_request(
        layout=case.reset_layout,
        episode_id=l2b_cases._RESET_EPISODE_ID,
        episode_epoch=1,
        chunk_id=1,
        expected_revision=1,
        commit_id=historical_request.commit_id,
        transaction_nonce="historical-id-reused-in-current-episode",
    )
    with pytest.raises(hc.CacheContractError) as historical_conflict:
        recovered_final.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=reused_historical,
            stage_callback=must_not_stage,
        )
    assert historical_conflict.value.code == "COMMIT_DUPLICATE_CONFLICT"

    reset_id_as_commit = replace(
        reused_historical,
        commit_id="recovery-reset",
        transaction_nonce="reset-id-cross-kind-commit-nonce",
    )
    with pytest.raises(hc.CacheContractError) as reset_cross_kind:
        recovered_final.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=reset_id_as_commit,
            stage_callback=must_not_stage,
        )
    assert reset_cross_kind.value.code == "COMMIT_DUPLICATE_CONFLICT"

    with pytest.raises(hc.CacheContractError) as commit_cross_kind:
        recovered_final.reset(
            reset_id=current_request.commit_id,
            transaction_nonce="commit-id-cross-kind-reset-nonce",
            new_episode_id="cross-kind-reset-target",
            layout=case.layout,
        )
    assert commit_cross_kind.value.code == "COMMIT_DUPLICATE_CONFLICT"

    with pytest.raises(hc.CacheContractError) as changed_reset:
        recovered_final.reset(
            reset_id="recovery-reset",
            transaction_nonce="different-reset-nonce",
            new_episode_id=l2b_cases._RESET_EPISODE_ID,
            layout=case.reset_layout,
        )
    assert changed_reset.value.code == "COMMIT_DUPLICATE_CONFLICT"

    assert stage_calls == 0
    assert _root_fingerprint(case.ledger) == final_root


@pytest.mark.parametrize(
    "mutation",
    (
        "gap",
        "duplicate_sequence",
        "predecessor",
        "ledger_id",
        "variant",
        "writer_fence",
        "stale_state",
        "body_hash",
        "state_changing_abort",
    ),
)
def test_chain_identity_transition_mutations_fail_before_manager_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    case = _case(tmp_path / mutation)
    old = case.fresh_manager()
    if mutation == "state_changing_abort":
        request = l2b_cases._request(
            case.layout,
            chunk_id=0,
            commit_id="state-changing-abort",
            transaction_nonce="state-changing-abort-nonce",
        )

        def fail_staging(_context):
            raise RuntimeError("simulated registered staging abort")

        with pytest.raises(hc.CacheContractError):
            old.commit_paired(
                commit_source=hc.CommitSource.TEACHER_FORCING,
                request=request,
                stage_callback=fail_staging,
            )
        transaction_id = request.commit_id
    else:
        old.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=l2b_cases._request(case.layout, chunk_id=0),
            stage_callback=l2b_cases._layer_payloads,
        )
        old.reset(
            reset_id="mutated-chain-reset",
            transaction_nonce="mutated-chain-reset-nonce",
            new_episode_id=l2b_cases._RESET_EPISODE_ID,
            layout=case.reset_layout,
        )
        transaction_id = "mutated-chain-reset"
    del old
    gc.collect()
    case.retire_ledger()
    reopened = _reopen(case)
    decision = reopened.read_decision(transaction_id)
    assert decision is not None
    record = json.loads(decision.payload)
    body = record["decision_body"]
    if mutation == "gap":
        body["ledger_sequence"] = 3
    elif mutation == "duplicate_sequence":
        body["ledger_sequence"] = 0
    elif mutation == "predecessor":
        body["previous_decision_sha256"] = "0" * 64
    elif mutation == "ledger_id":
        body["ledger_id"] = "mixed-ledger-id"
    elif mutation == "variant":
        body["staging_variant"] = CACHStagingVariant.REF_GDN_CORRECTED.value
    elif mutation == "writer_fence":
        body["writer_fence_id"] = "mixed-writer-fence"
    elif mutation == "stale_state":
        body["state_manifest_before"] = reopened.identity.genesis_state_manifest
    elif mutation == "state_changing_abort":
        assert body["result"] == Stage2BDecisionResult.ABORTED.value
        body["revision_after"] += 1
    else:
        record["decision_body_sha256"] = "0" * 64
    if mutation != "body_hash":
        record["decision_body_sha256"] = hc._sha256_bytes(
            offline._canonical_json_bytes(body)
        )
    decision.path.chmod(0o600)
    decision.path.write_bytes(offline._canonical_json_bytes(record))
    decision.path.chmod(0o400)
    mutated_root = _root_fingerprint(reopened)

    manager_constructions = 0
    real_manager_init = l3._RecoveredLedgerSeamedStage2BHybridCacheManager.__init__

    def observe_manager_construction(self, **kwargs):
        nonlocal manager_constructions
        manager_constructions += 1
        return real_manager_init(self, **kwargs)

    monkeypatch.setattr(
        l3._RecoveredLedgerSeamedStage2BHybridCacheManager,
        "__init__",
        observe_manager_construction,
    )
    with pytest.raises(hc.CacheContractError) as rejected:
        _recover(case, ledger=reopened)
    assert rejected.value.code == "L3_LEDGER_MISMATCH"
    assert manager_constructions == 0
    assert _root_fingerprint(reopened) == mutated_root


def test_valid_fork_and_duplicate_logical_id_fail_before_manager_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = _case(tmp_path / "main")
    main_old = main.fresh_manager()
    main_old.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=l2b_cases._request(main.layout, chunk_id=0),
        stage_callback=l2b_cases._layer_payloads,
    )
    del main_old
    gc.collect()
    main.retire_ledger()
    main_reopened = _reopen(main)

    fork = _case(tmp_path / "fork")
    fork_old = fork.fresh_manager()
    fork_old.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=l2b_cases._request(
            fork.layout,
            chunk_id=0,
            commit_id="individually-valid-fork",
            transaction_nonce="individually-valid-fork-nonce",
        ),
        stage_callback=l2b_cases._layer_payloads,
    )
    for source in fork.ledger.root.iterdir():
        if source.name == "LEDGER.json":
            continue
        destination = main_reopened.root / source.name
        if destination.exists():
            assert destination.read_bytes() == source.read_bytes()
            continue
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o400)
    del fork_old
    gc.collect()
    fork.retire_ledger()

    manager_constructions = 0
    real_manager_init = l3._RecoveredLedgerSeamedStage2BHybridCacheManager.__init__

    def observe_manager_construction(self, **kwargs):
        nonlocal manager_constructions
        manager_constructions += 1
        return real_manager_init(self, **kwargs)

    monkeypatch.setattr(
        l3._RecoveredLedgerSeamedStage2BHybridCacheManager,
        "__init__",
        observe_manager_construction,
    )
    with pytest.raises(hc.CacheContractError) as forked:
        _recover(main, ledger=main_reopened)
    assert forked.value.code == "L3_LEDGER_MISMATCH"
    assert manager_constructions == 0

    monkeypatch.undo()
    duplicate = _case(tmp_path / "duplicate")
    duplicate_old = duplicate.fresh_manager()
    duplicate_request = l2b_cases._request(duplicate.layout, chunk_id=0)
    duplicate_old.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=duplicate_request,
        stage_callback=l2b_cases._layer_payloads,
    )
    del duplicate_old
    gc.collect()
    duplicate.retire_ledger()
    duplicate_reopened = _reopen(duplicate)
    decision = duplicate_reopened.read_decision(duplicate_request.commit_id)
    assert decision is not None
    alias = duplicate_reopened.root / duplicate_reopened.decision_filename(
        "duplicate-logical-id-alias"
    )
    alias.write_bytes(decision.payload)
    alias.chmod(0o400)
    manager_constructions = 0
    monkeypatch.setattr(
        l3._RecoveredLedgerSeamedStage2BHybridCacheManager,
        "__init__",
        observe_manager_construction,
    )
    with pytest.raises(hc.CacheContractError) as duplicated:
        _recover(duplicate, ledger=duplicate_reopened)
    assert duplicated.value.code == "L3_LEDGER_MISMATCH"
    assert manager_constructions == 0


@pytest.mark.parametrize(
    "mismatch",
    ("registry", "empty_layout_inventory", "variant", "synthetic_provenance"),
)
def test_wrong_recovery_identity_inputs_fail_before_manager_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    case = _case(tmp_path / mismatch)
    case.retire_ledger()
    reopened = _reopen(case)
    registry = case.registry
    forbidden_scan_calls = 0
    if mismatch == "registry":
        registry = hc.LayerRegistry(
            layers=(
                replace(
                    case.registry.layers[0],
                    operator_class="synthetic.WrongRecoveryRegistry",
                ),
            )
        )
    elif mismatch == "empty_layout_inventory":
        monkeypatch.setattr(reopened, "_expected_layouts", {})
    elif mismatch == "variant":
        monkeypatch.setattr(
            reopened,
            "_identity",
            replace(
                reopened.identity,
                staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
            ),
        )
    else:
        object.__setattr__(case.layout, "synthetic_test_only", False)
        monkeypatch.setattr(
            reopened,
            "_expected_layouts",
            {
                case.layout.layout_instance_digest: case.layout,
                case.reset_layout.layout_instance_digest: case.reset_layout,
            },
        )

        def provenance_must_fail_before_scan():
            nonlocal forbidden_scan_calls
            forbidden_scan_calls += 1
            raise AssertionError("non-synthetic provenance reached ledger scan")

        monkeypatch.setattr(
            reopened,
            "_scan_records",
            provenance_must_fail_before_scan,
        )

    manager_constructions = 0
    real_manager_init = l3._RecoveredLedgerSeamedStage2BHybridCacheManager.__init__

    def observe_manager_construction(self, **kwargs):
        nonlocal manager_constructions
        manager_constructions += 1
        return real_manager_init(self, **kwargs)

    monkeypatch.setattr(
        l3._RecoveredLedgerSeamedStage2BHybridCacheManager,
        "__init__",
        observe_manager_construction,
    )
    before = _root_fingerprint(reopened)
    with pytest.raises(hc.CacheContractError) as rejected:
        l3.Stage2BRecoveredLedgeredHybridCacheManager.recover_from_ledger(
            registry=registry,
            ledger=reopened,
            synthetic_test_only=True,
        )
    assert rejected.value.code == "L3_LEDGER_MISMATCH"
    assert manager_constructions == 0
    if mismatch == "synthetic_provenance":
        assert forbidden_scan_calls == 0
    assert _root_fingerprint(reopened) == before


def test_injected_non_cpu_state_fails_before_manager_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    case.retire_ledger()
    reopened = _reopen(case)
    real_cpu_guard = l3._require_cpu_state
    guard_calls = 0
    manager_constructions = 0
    real_manager_init = l3._RecoveredLedgerSeamedStage2BHybridCacheManager.__init__

    def inject_non_cpu_state(_decoded_state):
        nonlocal guard_calls
        guard_calls += 1
        snapshot = SimpleNamespace(to_manifest=lambda: {"device": "cuda:0"})
        non_cpu_state = SimpleNamespace(
            layer_states=(SimpleNamespace(tensors=(("cache", snapshot),)),),
            committed_action_history=None,
        )
        real_cpu_guard(non_cpu_state)

    def observe_manager_construction(self, **kwargs):
        nonlocal manager_constructions
        manager_constructions += 1
        return real_manager_init(self, **kwargs)

    monkeypatch.setattr(l3, "_require_cpu_state", inject_non_cpu_state)
    monkeypatch.setattr(
        l3._RecoveredLedgerSeamedStage2BHybridCacheManager,
        "__init__",
        observe_manager_construction,
    )
    before = _root_fingerprint(reopened)
    with pytest.raises(hc.CacheContractError) as rejected:
        _recover(case, ledger=reopened)
    assert rejected.value.code == "L3_CPU_SYNTHETIC_ONLY"
    assert guard_calls == 1
    assert manager_constructions == 0
    assert _root_fingerprint(reopened) == before


@pytest.mark.parametrize(
    "damage",
    (
        "missing_receipt",
        "substituted_receipt",
        "missing_snapshot",
        "missing_blob",
        "missing_operation_identity",
        "missing_operation_payload",
    ),
)
def test_missing_or_substituted_authoritative_object_fails_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    case = _case(tmp_path / damage)
    old = case.fresh_manager()
    request = l2b_cases._request(case.layout, chunk_id=0)
    old.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=l2b_cases._layer_payloads,
    )
    intents, _, _, _ = case.ledger._scan_records()
    intent_entry = intents[request.commit_id]
    operation_identity, _ = case.ledger._validate_intent_input(
        intent_entry.intent
    )
    terminal = case.ledger.read_decision(request.commit_id)
    assert terminal is not None
    body = terminal.decision.decision_body
    assert body.snapshot_manifest_sha256 is not None
    snapshot_manifest, _ = case.ledger._load_snapshot(
        body.snapshot_manifest_sha256
    )
    blob_sha256s = tuple(
        sorted(offline._collect_snapshot_blob_references(snapshot_manifest))
    )
    assert blob_sha256s
    target_sha256 = {
        "missing_receipt": body.success_or_abort_receipt_sha256,
        "substituted_receipt": body.success_or_abort_receipt_sha256,
        "missing_snapshot": body.snapshot_manifest_sha256,
        "missing_blob": blob_sha256s[0],
        "missing_operation_identity": intent_entry.intent.input_identity_sha256,
        "missing_operation_payload": (
            operation_identity.operation_payload_sha256
        ),
    }[damage]
    target = case.ledger.root / case.ledger.object_filename(target_sha256)
    if damage == "substituted_receipt":
        target.chmod(0o600)
        target.write_bytes(b"substituted-authoritative-receipt")
        target.chmod(0o400)
    else:
        target.unlink()
    del old
    gc.collect()
    case.retire_ledger()

    manager_constructions = 0
    real_manager_init = l3._RecoveredLedgerSeamedStage2BHybridCacheManager.__init__

    def observe_manager_construction(self, **kwargs):
        nonlocal manager_constructions
        manager_constructions += 1
        return real_manager_init(self, **kwargs)

    monkeypatch.setattr(
        l3._RecoveredLedgerSeamedStage2BHybridCacheManager,
        "__init__",
        observe_manager_construction,
    )
    with pytest.raises((hc.CacheContractError, Stage2BOfflineLedgerError)):
        _recover(case)
    assert manager_constructions == 0


def test_unexpected_entry_fails_before_manager_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    (case.ledger.root / "unexpected-recovery-entry").write_bytes(b"x")
    case.retire_ledger()
    manager_constructions = 0
    real_manager_init = l3._RecoveredLedgerSeamedStage2BHybridCacheManager.__init__

    def observe_manager_construction(self, **kwargs):
        nonlocal manager_constructions
        manager_constructions += 1
        return real_manager_init(self, **kwargs)

    monkeypatch.setattr(
        l3._RecoveredLedgerSeamedStage2BHybridCacheManager,
        "__init__",
        observe_manager_construction,
    )
    with pytest.raises((hc.CacheContractError, Stage2BOfflineLedgerError)):
        _recover(case)
    assert manager_constructions == 0


def test_install_failure_and_changed_final_chain_return_no_facade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_case = _case(tmp_path / "install")
    install_case.retire_ledger()
    install_reopened = _reopen(install_case)
    facade_constructions = 0
    real_facade_init = l3.Stage2BRecoveredLedgeredHybridCacheManager.__init__

    def observe_facade_construction(self, **kwargs):
        nonlocal facade_constructions
        facade_constructions += 1
        return real_facade_init(self, **kwargs)

    def fail_state_install(self, **_kwargs):
        raise hc.CacheContractError(
            "L3_INSTALL_FAILED",
            "simulated one-shot state install failure",
        )

    monkeypatch.setattr(
        l3.Stage2BRecoveredLedgeredHybridCacheManager,
        "__init__",
        observe_facade_construction,
    )
    monkeypatch.setattr(
        l3._RecoveredLedgerSeamedStage2BHybridCacheManager,
        "_install_recovered_state",
        fail_state_install,
    )
    with pytest.raises(hc.CacheContractError) as install_failure:
        _recover(install_case, ledger=install_reopened)
    assert install_failure.value.code == "L3_INSTALL_FAILED"
    assert facade_constructions == 0

    monkeypatch.undo()
    final_case = _case(tmp_path / "final")
    old = final_case.fresh_manager()
    request = l2b_cases._request(final_case.layout, chunk_id=0)
    old.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=l2b_cases._layer_payloads,
    )
    del old
    gc.collect()
    final_case.retire_ledger()
    final_reopened = _reopen(final_case)
    real_final_init = l3.Stage2BRecoveredLedgeredHybridCacheManager.__init__
    real_scan = final_reopened._scan_records
    facade_installed = False
    changed_final_scan = False

    def mark_facade_installed(self, **kwargs):
        nonlocal facade_installed
        real_final_init(self, **kwargs)
        facade_installed = True

    def change_first_post_install_scan():
        nonlocal changed_final_scan
        intents, decisions, ordered, pending = real_scan()
        if facade_installed and not changed_final_scan:
            assert len(ordered) == 1
            changed_final_scan = True
            return intents, decisions, (), pending
        return intents, decisions, ordered, pending

    monkeypatch.setattr(
        l3.Stage2BRecoveredLedgeredHybridCacheManager,
        "__init__",
        mark_facade_installed,
    )
    monkeypatch.setattr(
        final_reopened,
        "_scan_records",
        change_first_post_install_scan,
    )
    with pytest.raises(hc.CacheContractError) as final_failure:
        _recover(final_case, ledger=final_reopened)
    assert final_failure.value.code == "L3_RECOVERY_OUTCOME_UNKNOWN"
    assert facade_installed is True
    assert changed_final_scan is True


def test_recovered_facade_later_ambiguous_publication_is_absorbing_poison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    case.retire_ledger()
    recovered = _recover(case)
    request = l2b_cases._request(case.layout, chunk_id=0)
    adapter = l2b_cases._publisher_adapter(recovered)
    publish_holder = [adapter.publish_exclusive]

    def publish_then_raise(*, receipt_id, payload, expected_sha256):
        publish_holder[0](
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=expected_sha256,
        )
        raise OSError("simulated ambiguous post-recovery publication")

    monkeypatch.setattr(adapter, "publish_exclusive", publish_then_raise)
    with pytest.raises(hc.CacheContractError):
        recovered.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=l2b_cases._layer_payloads,
        )
    terminal = case.ledger.read_decision(request.commit_id)
    assert terminal is not None
    assert terminal.decision.decision_body.result is Stage2BDecisionResult.COMMITTED
    poisoned_root = _root_fingerprint(case.ledger)

    poisoned_calls = (
        lambda: recovered.state,
        lambda: recovered.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=replace(
                request,
                commit_id="post-recovery-poison-followup",
                transaction_nonce="post-recovery-poison-followup-nonce",
            ),
            stage_callback=l2b_cases._layer_payloads,
        ),
        lambda: recovered.reset(
            reset_id="post-recovery-poison-reset",
            transaction_nonce="post-recovery-poison-reset-nonce",
            new_episode_id=l2b_cases._RESET_EPISODE_ID,
            layout=case.reset_layout,
        ),
    )
    for call in poisoned_calls:
        with pytest.raises(hc.CacheContractError) as poisoned:
            call()
        l2b_cases._assert_l2b_poisoned(poisoned)
        assert _root_fingerprint(case.ledger) == poisoned_root
