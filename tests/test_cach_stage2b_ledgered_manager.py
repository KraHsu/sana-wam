"""Focused CPU/synthetic contracts for the Stage-2B L2B ledgered facade.

These tests deliberately exercise only a secure ``tmp_path`` ledger.  They do
not admit a production filesystem, recovery, a model, CUDA, training, or
evaluation.  A few assertions inspect the pinned private L2B seam so the
terminal-before-CAS ordering can be proved without re-entering the public
manager while its publisher callback is active.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from sana_wam.cach.stage2b_offline_ledger import (
    Stage2BAbortCode,
    Stage2BDecisionResult,
    Stage2BFailurePoint,
    Stage2BOfflineLedger,
    Stage2BOfflineLedgerError,
    Stage2BTransactionKind,
)
from sana_wam.cach.stage2b_state_snapshot import (
    EncodedStage2BStateSnapshot,
    encode_stage2b_state_snapshot,
)
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_chunk_layout import (
    ChunkActionLayout,
    build_synthetic_chunk_action_layout_for_tests,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)
from sana_wam.model.video_backbone.sana import hybrid_cache as hc
from sana_wam.model.video_backbone.sana import hybrid_cache_stage2b as h2
from sana_wam.model.video_backbone.sana import (
    hybrid_cache_stage2b_ledgered as l2b,
)


_EPISODE_ID = "ledgered-episode"
_RESET_EPISODE_ID = "ledgered-reset-episode"
_LEDGER_ID = "ledgered-manager-test-ledger"
_WRITER_FENCE_ID = "ledgered-manager-test-writer"


def _secure_root(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True)
    path.chmod(0o700)
    return path.resolve()


def _layout(*, episode_id: str) -> ChunkActionLayout:
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=33,
        video_stride=1,
        source_row_label=f"{episode_id}-row",
        episode_label=episode_id,
    )
    layout = build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=33,
        video_valid_mask=(True,) * 33,
        action_valid_mask=(True,) * 32,
    )
    assert len(layout.chunks) >= 2
    return layout


def _registry() -> hc.LayerRegistry:
    return hc.LayerRegistry(
        layers=(
            hc.LayerSpec(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="synthetic.LedgeredManagerGDN",
                main_shortconv_enabled=True,
                ffn_tconv_enabled=True,
            ),
        )
    )


def _request(
    layout: ChunkActionLayout,
    *,
    chunk_id: int,
    commit_id: str | None = None,
    transaction_nonce: str | None = None,
) -> h2.Stage2BTeacherForcingCommitRequest:
    chunk = layout.chunks[chunk_id]
    frame_valid_mask = torch.tensor(
        chunk.latent_valid_mask,
        dtype=torch.bool,
    ).view(1, -1)
    video = torch.arange(
        3 * len(chunk.latent_valid_mask) * 4,
        dtype=torch.float32,
    ).reshape(1, 3, len(chunk.latent_valid_mask), 2, 2)
    video = (video + 1000 * chunk_id).to(dtype=torch.bfloat16)
    video = video.masked_fill(
        (~frame_valid_mask)[:, None, :, None, None],
        0,
    )
    actions = torch.arange(
        chunk.action_slot_capacity * 20,
        dtype=torch.float32,
    ).reshape(1, chunk.action_slot_capacity, 20)
    actions = (actions + 1000 * chunk_id).to(dtype=torch.bfloat16)
    action_valid_mask = torch.tensor(
        chunk.action_valid_mask,
        dtype=torch.bool,
    ).view(1, -1)
    actions = actions.masked_fill((~action_valid_mask)[:, :, None], 0)
    proof = hc.TeacherForcingDatasetPairProof(
        dataset_manifest_sha256="b" * 64,
        dataset_episode_id=_EPISODE_ID,
        dataset_row_identity=f"{_EPISODE_ID}:chunk-{chunk_id}",
        dataset_row_start=0,
        dataset_row_end_exclusive=layout.valid_raw_count,
        layout_spec_sha256=layout.layout_spec_sha256,
        layout_instance_digest=layout.layout_instance_digest,
        video_tensor_digest=hc.tensor_digest(video),
        frame_valid_mask_digest=hc.tensor_digest(frame_valid_mask),
        action_tensor_digest=hc.tensor_digest(actions),
        action_mask_digest=hc.tensor_digest(action_valid_mask),
        row_order_manifest_sha256="c" * 64,
    )
    resolved_commit_id = commit_id or f"ledgered-commit-{chunk_id}"
    return h2.Stage2BTeacherForcingCommitRequest(
        commit_id=resolved_commit_id,
        transaction_nonce=(
            transaction_nonce or f"ledgered-commit-nonce-{chunk_id}"
        ),
        expected_episode_id=_EPISODE_ID,
        expected_episode_epoch=0,
        expected_revision=chunk_id,
        content_time=hc.ContentTime.from_layout(
            episode_id=_EPISODE_ID,
            episode_epoch=0,
            layout=layout,
            chunk=chunk,
        ),
        video=video,
        frame_valid_mask=frame_valid_mask,
        actions=actions,
        action_valid_mask=action_valid_mask,
        proof=proof,
    )


def _layer_payloads(
    context: h2.Stage2BPairedStagingContext,
) -> tuple[hc.StagedLayerPayload, ...]:
    value = float(context.content_time.chunk_id + 1)
    return (
        hc.StagedLayerPayload(
            layer_index=0,
            kind=hc.LayerKind.GDN_FULL_HISTORY,
            tensors={
                "ffn_tconv_left_context": torch.full(
                    (1, 2, 2, 3),
                    value,
                    dtype=torch.float32,
                ),
                "main_s_kv": torch.full(
                    (1, 2, 4, 4),
                    value,
                    dtype=torch.bfloat16,
                ),
                "main_s_z": torch.full(
                    (1, 2, 4, 1),
                    value,
                    dtype=torch.float32,
                ),
                "main_shortconv_left_context": torch.full(
                    (1, 2, 3),
                    context.content_time.chunk_id % 2 == 0,
                    dtype=torch.bool,
                ),
            },
        ),
    )


class _Case:
    def __init__(self, tmp_path: Path) -> None:
        self.layout = _layout(episode_id=_EPISODE_ID)
        self.reset_layout = _layout(episode_id=_RESET_EPISODE_ID)
        self.registry = _registry()
        self.genesis_state = h2.Stage2BTemporalState.empty(
            episode_id=_EPISODE_ID,
            episode_epoch=0,
            layout=self.layout,
            registry=self.registry,
            staging_variant=CACHStagingVariant.CACH_A,
        )
        self.genesis_snapshot = encode_stage2b_state_snapshot(
            state=self.genesis_state,
            expected_layout=self.layout,
            expected_registry=self.registry,
        )
        self.ledger = Stage2BOfflineLedger.initialize(
            root=_secure_root(tmp_path / "ledger"),
            ledger_id=_LEDGER_ID,
            staging_variant=CACHStagingVariant.CACH_A,
            writer_fence_id=_WRITER_FENCE_ID,
            expected_layout=self.layout,
            expected_registry=self.registry,
            genesis_snapshot=self.genesis_snapshot,
            additional_expected_layouts=(self.reset_layout,),
        )
        self.manager = l2b.Stage2BLedgeredHybridCacheManager.for_synthetic_tests(
            registry=self.registry,
            episode_id=_EPISODE_ID,
            layout=self.layout,
            ledger=self.ledger,
            staging_variant=CACHStagingVariant.CACH_A,
        )


def _inner_manager(
    manager: l2b.Stage2BLedgeredHybridCacheManager,
) -> h2.Stage2BHybridCacheManager:
    """Return the pinned inherited-manager seam without using its public API."""

    for name in ("_manager", "_inner_manager", "_inner"):
        candidate = getattr(manager, name, None)
        if isinstance(candidate, h2.Stage2BHybridCacheManager):
            return candidate
    raise AssertionError("L2B facade no longer owns the pinned Stage-2B manager seam")


def _publisher_adapter(manager: l2b.Stage2BLedgeredHybridCacheManager):
    return _inner_manager(manager)._publisher


def _root_fingerprint(ledger: Stage2BOfflineLedger) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (path.name, path.read_bytes())
        for path in sorted(ledger.root.iterdir(), key=lambda candidate: candidate.name)
        if path.is_file()
    )


def _assert_snapshot_is_exact_live_state(
    *,
    case: _Case,
    layout: ChunkActionLayout,
    transaction_id: str,
) -> None:
    decision = case.ledger.read_decision(transaction_id)
    assert decision is not None
    body = decision.decision.decision_body
    assert body.result is Stage2BDecisionResult.COMMITTED
    assert body.snapshot_manifest_sha256 is not None
    encoded_live = encode_stage2b_state_snapshot(
        state=case.manager.state,
        expected_layout=layout,
        expected_registry=case.registry,
    )
    assert encoded_live.manifest_sha256 == body.snapshot_manifest_sha256
    assert (
        case.ledger.read_object(encoded_live.manifest_sha256).payload
        == encoded_live.manifest_bytes
    )
    for blob in encoded_live.blobs:
        assert case.ledger.read_object(blob.sha256).payload == blob.payload


def _assert_l2b_poisoned(exc: pytest.ExceptionInfo[l2b.CacheContractError]) -> None:
    assert exc.value.code == "L2B_PUBLICATION_POISONED"


def test_two_chunk_commit_prepares_before_exact_once_staging_and_commits_snapshot_before_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    calls: list[str] = []
    terminal_before_cas: list[str] = []
    real_commit = case.ledger.commit

    def observe_ledger_commit(**kwargs):
        stored = real_commit(**kwargs)
        transaction_id = kwargs["intent"].transaction_id
        assert case.ledger.read_decision(transaction_id) is not None
        assert (
            _inner_manager(case.manager)._state.state_manifest_digest
            == kwargs["intent"].state_manifest_before
        )
        terminal_before_cas.append(transaction_id)
        return stored

    monkeypatch.setattr(case.ledger, "commit", observe_ledger_commit)

    for chunk_id in (0, 1):
        request = _request(case.layout, chunk_id=chunk_id)

        def stage(context, *, expected_id=request.commit_id):
            stored_intent = case.ledger.read_intent(expected_id)
            assert stored_intent.intent.status == "PREPARED"
            assert stored_intent.intent.transaction_id == expected_id
            assert case.ledger.read_decision(expected_id) is None
            calls.append(expected_id)
            return _layer_payloads(context)

        receipt = case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=stage,
        )

        assert receipt.commit_id == request.commit_id
        assert calls.count(request.commit_id) == 1
        assert terminal_before_cas.count(request.commit_id) == 1
        stored_intent = case.ledger.read_intent(request.commit_id)
        operation_identity = json.loads(
            case.ledger.read_object(stored_intent.intent.input_identity_sha256)
            .payload.decode("utf-8")
        )
        operation = json.loads(
            case.ledger.read_object(
                operation_identity["operation_payload_sha256"]
            ).payload.decode("utf-8")
        )
        history = case.manager.state.committed_action_history
        assert history is not None
        assert operation["committed_action_span_digest"] == history.spans[-1].actions_digest
        assert operation["pair_action_digest"] == receipt.pair_action_digest
        assert operation["pair_action_mask_digest"] == receipt.pair_action_mask_digest
        assert (
            operation["pair_frame_valid_mask_digest"]
            == receipt.pair_frame_valid_mask_digest
        )
        assert operation["pair_video_digest"] == receipt.pair_video_digest
        assert operation["source_proof_digest"] == receipt.source_proof_digest
        assert (
            operation["teacher_pair_payload_digest"]
            == receipt.teacher_pair_payload_digest
        )
        _assert_snapshot_is_exact_live_state(
            case=case,
            layout=case.layout,
            transaction_id=request.commit_id,
        )

    assert case.manager.state.revision == 2
    assert tuple(calls) == ("ledgered-commit-0", "ledgered-commit-1")
    assert tuple(terminal_before_cas) == tuple(calls)
    assert len(case.ledger.scan_valid_chain()) == 2


def test_terminal_then_exception_leaves_old_state_and_sticky_poisons_every_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    before_digest = case.manager.state.state_manifest_digest
    read_view = case.manager.snapshot_for_denoise(
        expected_episode_id=_EPISODE_ID,
        expected_episode_epoch=0,
        expected_revision=0,
        expected_next_chunk_id=0,
    )
    scratch = read_view.export_scratch()
    adapter = _publisher_adapter(case.manager)
    real_publish = adapter.publish_exclusive

    def publish_then_raise(*, receipt_id, payload, expected_sha256):
        real_publish(
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=expected_sha256,
        )
        terminal = case.ledger.read_decision(receipt_id)
        assert terminal is not None
        assert terminal.decision.decision_body.result is Stage2BDecisionResult.COMMITTED
        assert _inner_manager(case.manager)._state.state_manifest_digest == before_digest
        raise RuntimeError("injected after exact terminal, before inherited CAS")

    monkeypatch.setattr(adapter, "publish_exclusive", publish_then_raise)

    with pytest.raises(hc.CacheContractError):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=_layer_payloads,
        )

    terminal = case.ledger.read_decision(request.commit_id)
    assert terminal is not None
    assert terminal.decision.decision_body.result is Stage2BDecisionResult.COMMITTED
    assert _inner_manager(case.manager)._state.state_manifest_digest == before_digest
    poisoned_root = _root_fingerprint(case.ledger)

    poisoned_calls = (
        lambda: case.manager.state,
        lambda: case.manager.snapshot_for_denoise(
            expected_episode_id=_EPISODE_ID,
            expected_episode_epoch=0,
            expected_revision=0,
            expected_next_chunk_id=0,
        ),
        lambda: case.manager.finish_denoise(read_view=read_view, scratch=scratch),
        lambda: case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=replace(
                request,
                commit_id="poison-followup-commit",
                transaction_nonce="poison-followup-commit-nonce",
            ),
            stage_callback=_layer_payloads,
        ),
        lambda: case.manager.reset(
            reset_id="poison-followup-reset",
            transaction_nonce="poison-followup-reset-nonce",
            new_episode_id=_RESET_EPISODE_ID,
            layout=case.reset_layout,
        ),
        lambda: case.manager.commit_paired(
            commit_source="invalid-source-after-poison",
            request=None,
            stage_callback=None,
        ),
        lambda: case.manager.reset(
            reset_id="",
            transaction_nonce="invalid-reset-after-poison",
            new_episode_id=_RESET_EPISODE_ID,
            layout=case.reset_layout,
        ),
    )
    for call in poisoned_calls:
        with pytest.raises(l2b.CacheContractError) as exc:
            call()
        _assert_l2b_poisoned(exc)
        assert _root_fingerprint(case.ledger) == poisoned_root


def test_publisher_token_is_disarmed_inside_the_non_ready_critical_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    adapter = _publisher_adapter(case.manager)
    real_disarm = adapter.disarm
    observed_phases: list[str] = []

    def observe_disarm(token):
        if adapter._armed_token is token:
            assert case.manager._active is not None
            assert case.manager._phase != "READY"
            observed_phases.append(case.manager._phase)
        real_disarm(token)

    monkeypatch.setattr(adapter, "disarm", observe_disarm)
    case.manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=_request(case.layout, chunk_id=0),
        stage_callback=_layer_payloads,
    )

    assert observed_phases[0] == "CAS_CONFIRMING"
    assert case.manager.state.revision == 1


def test_staging_failure_records_exact_abort_without_snapshot_and_new_id_can_commit(
    tmp_path: Path,
) -> None:
    case = _Case(tmp_path)
    failed = _request(
        case.layout,
        chunk_id=0,
        commit_id="staging-failure",
        transaction_nonce="staging-failure-nonce",
    )
    before_digest = case.manager.state.state_manifest_digest

    def fail_staging(_context):
        raise RuntimeError("synthetic staging fault")

    with pytest.raises(hc.CacheContractError, match="COMMIT_STAGING_FAILED"):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=failed,
            stage_callback=fail_staging,
        )

    terminal = case.ledger.read_decision(failed.commit_id)
    assert terminal is not None
    body = terminal.decision.decision_body
    assert body.result is Stage2BDecisionResult.ABORTED
    assert body.abort_code is Stage2BAbortCode.STAGING_FAILED
    assert body.snapshot_manifest_sha256 is None
    assert body.state_manifest_before == before_digest
    assert body.state_manifest_after == before_digest
    abort_receipt = case.ledger.read_object(
        body.success_or_abort_receipt_sha256
    ).payload
    assert b'"failure_point":"STAGING"' in abort_receipt
    assert case.manager.state.state_manifest_digest == before_digest

    staged_on_reuse = False

    def must_not_restage(context):
        nonlocal staged_on_reuse
        staged_on_reuse = True
        return _layer_payloads(context)

    with pytest.raises(l2b.CacheContractError):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=failed,
            stage_callback=must_not_restage,
        )
    assert staged_on_reuse is False
    assert case.ledger.read_decision(failed.commit_id).payload == terminal.payload

    replacement = replace(
        failed,
        commit_id="staging-failure-replacement",
        transaction_nonce="staging-failure-replacement-nonce",
    )
    receipt = case.manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=replacement,
        stage_callback=_layer_payloads,
    )
    assert receipt.commit_id == replacement.commit_id
    assert case.manager.state.revision == 1


def test_prepared_publication_failure_poisons_and_orphan_root_cannot_be_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    pristine = _root_fingerprint(case.ledger)

    def fail_prepared(_intent):
        raise Stage2BOfflineLedgerError(
            "LEDGER_WRITE_FAILED",
            "injected PREPARED publication failure",
        )

    monkeypatch.setattr(case.ledger, "prepare", fail_prepared)
    with pytest.raises(hc.CacheContractError):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=_layer_payloads,
        )

    assert _root_fingerprint(case.ledger) != pristine
    with pytest.raises(Stage2BOfflineLedgerError) as missing:
        case.ledger.read_intent(request.commit_id)
    assert missing.value.code == "LEDGER_INTENT_MISSING"
    with pytest.raises(l2b.CacheContractError) as poisoned:
        _ = case.manager.state
    _assert_l2b_poisoned(poisoned)
    with pytest.raises(l2b.CacheContractError) as reuse:
        l2b.Stage2BLedgeredHybridCacheManager.for_synthetic_tests(
            registry=case.registry,
            episode_id=_EPISODE_ID,
            layout=case.layout,
            ledger=case.ledger,
            staging_variant=CACHStagingVariant.CACH_A,
        )
    assert reuse.value.code == "L2B_RECOVERY_REQUIRED"


def test_staging_abort_unknown_outcome_keeps_terminal_and_sticky_poisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    before = case.manager.state.state_manifest_digest
    real_abort = case.ledger.abort

    def abort_then_report_unknown(**kwargs):
        real_abort(**kwargs)
        raise Stage2BOfflineLedgerError(
            "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
            "injected abort stable-read uncertainty",
        )

    def fail_staging(_context):
        raise RuntimeError("force the staging abort path")

    monkeypatch.setattr(case.ledger, "abort", abort_then_report_unknown)
    with pytest.raises(hc.CacheContractError):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=fail_staging,
        )

    decision = case.ledger.read_decision(request.commit_id)
    assert decision is not None
    body = decision.decision.decision_body
    assert body.result is Stage2BDecisionResult.ABORTED
    assert body.abort_code is Stage2BAbortCode.STAGING_FAILED
    assert body.state_manifest_after == before
    assert _inner_manager(case.manager)._state.state_manifest_digest == before
    with pytest.raises(l2b.CacheContractError) as poisoned:
        _ = case.manager.state
    _assert_l2b_poisoned(poisoned)


def test_exact_retry_does_not_stage_and_any_reuse_difference_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    first = case.manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=_layer_payloads,
    )
    entries_after_first = tuple(sorted(path.name for path in case.ledger.root.iterdir()))

    def retry_must_not_stage(_context):
        raise AssertionError("exact retry called the stager")

    reconciled: list[tuple[str, str]] = []
    real_reconcile = case.ledger.reconcile_terminal

    def observe_retry_reconcile(*, transaction_id, expected_record_sha256):
        reconciled.append((transaction_id, expected_record_sha256))
        return real_reconcile(
            transaction_id=transaction_id,
            expected_record_sha256=expected_record_sha256,
        )

    monkeypatch.setattr(case.ledger, "reconcile_terminal", observe_retry_reconcile)
    retried = case.manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=retry_must_not_stage,
    )
    assert retried.canonical_bytes == first.canonical_bytes
    assert tuple(sorted(path.name for path in case.ledger.root.iterdir())) == (
        entries_after_first
    )
    assert len(case.ledger.scan_valid_chain()) == 1
    assert len(reconciled) == 1
    assert reconciled[0][0] == request.commit_id

    reused = replace(request, transaction_nonce="different-transaction-nonce")
    with pytest.raises(l2b.CacheContractError):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=reused,
            stage_callback=retry_must_not_stage,
        )
    assert len(case.ledger.scan_valid_chain()) == 1

    changed_actions = request.actions.detach().clone()
    changed_actions[0, 0, 0] += 1
    changed_proof = replace(
        request.proof,
        action_tensor_digest=hc.tensor_digest(changed_actions),
    )
    changed_payload = replace(
        request,
        actions=changed_actions,
        proof=changed_proof,
    )
    with pytest.raises(l2b.CacheContractError) as changed:
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=changed_payload,
            stage_callback=retry_must_not_stage,
        )
    assert changed.value.code == "COMMIT_DUPLICATE_CONFLICT"
    assert len(case.ledger.scan_valid_chain()) == 1

    root_before_cross_kind = _root_fingerprint(case.ledger)
    with pytest.raises(l2b.CacheContractError) as cross_kind:
        case.manager.reset(
            reset_id=request.commit_id,
            transaction_nonce="cross-kind-reset-nonce",
            new_episode_id=_RESET_EPISODE_ID,
            layout=case.reset_layout,
        )
    assert cross_kind.value.code == "COMMIT_DUPLICATE_CONFLICT"
    assert _root_fingerprint(case.ledger) == root_before_cross_kind


def test_completed_retry_requires_a_fresh_exact_terminal_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    case.manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=_layer_payloads,
    )
    staged = False

    def must_not_stage(_context):
        nonlocal staged
        staged = True
        return _layer_payloads(_context)

    def unreadable_terminal(**_kwargs):
        raise Stage2BOfflineLedgerError(
            "LEDGER_OBJECT_CORRUPT",
            "injected completed-terminal read failure",
        )

    monkeypatch.setattr(case.ledger, "reconcile_terminal", unreadable_terminal)
    with pytest.raises(l2b.CacheContractError) as failure:
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=must_not_stage,
        )
    _assert_l2b_poisoned(failure)
    assert staged is False
    with pytest.raises(l2b.CacheContractError) as poisoned:
        _ = case.manager.state
    _assert_l2b_poisoned(poisoned)


def test_reset_commits_exact_empty_snapshot_before_cas_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    case.manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=_request(case.layout, chunk_id=0),
        stage_callback=_layer_payloads,
    )
    before_digest = case.manager.state.state_manifest_digest
    observed_terminal_before_cas = False
    real_commit = case.ledger.commit

    def observe_reset_commit(**kwargs):
        nonlocal observed_terminal_before_cas
        stored = real_commit(**kwargs)
        if kwargs["intent"].transaction_kind is Stage2BTransactionKind.RESET:
            assert case.ledger.read_decision(kwargs["intent"].transaction_id) is not None
            assert _inner_manager(case.manager)._state.state_manifest_digest == before_digest
            observed_terminal_before_cas = True
        return stored

    monkeypatch.setattr(case.ledger, "commit", observe_reset_commit)
    receipt = case.manager.reset(
        reset_id="ledgered-reset-0",
        transaction_nonce="ledgered-reset-nonce-0",
        new_episode_id=_RESET_EPISODE_ID,
        layout=case.reset_layout,
    )

    assert observed_terminal_before_cas is True
    assert receipt.aborted_pending_commit_id is None
    assert case.manager.state.revision == 0
    assert case.manager.state.episode_id == _RESET_EPISODE_ID
    decision = case.ledger.read_decision(receipt.reset_id)
    assert decision is not None
    assert decision.decision.decision_body.transaction_kind is Stage2BTransactionKind.RESET
    assert decision.decision.decision_body.result is Stage2BDecisionResult.COMMITTED
    encoded = encode_stage2b_state_snapshot(
        state=case.manager.state,
        expected_layout=case.reset_layout,
        expected_registry=case.registry,
    )
    assert encoded.blobs == ()
    assert (
        decision.decision.decision_body.snapshot_manifest_sha256
        == encoded.manifest_sha256
    )

    entries = tuple(sorted(path.name for path in case.ledger.root.iterdir()))
    retried = case.manager.reset(
        reset_id=receipt.reset_id,
        transaction_nonce=receipt.transaction_nonce,
        new_episode_id=_RESET_EPISODE_ID,
        layout=case.reset_layout,
    )
    assert retried.canonical_bytes == receipt.canonical_bytes
    assert tuple(sorted(path.name for path in case.ledger.root.iterdir())) == entries

    with pytest.raises(l2b.CacheContractError):
        case.manager.reset(
            reset_id=receipt.reset_id,
            transaction_nonce="different-reset-nonce",
            new_episode_id=_RESET_EPISODE_ID,
            layout=case.reset_layout,
        )


def test_reset_during_prepared_commit_is_rejected_before_reset_object_or_intent(
    tmp_path: Path,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    reset_id = "reset-while-commit-active"

    def stage(context):
        assert case.ledger.read_intent(request.commit_id).intent.status == "PREPARED"
        entries_before_reset = tuple(
            sorted(path.name for path in case.ledger.root.iterdir())
        )
        with pytest.raises(l2b.CacheContractError) as exc:
            case.manager.reset(
                reset_id=reset_id,
                transaction_nonce="reset-while-commit-active-nonce",
                new_episode_id=_RESET_EPISODE_ID,
                layout=case.reset_layout,
            )
        assert exc.value.code == "L2B_TRANSACTION_BUSY"
        assert tuple(sorted(path.name for path in case.ledger.root.iterdir())) == (
            entries_before_reset
        )
        assert case.ledger.read_decision(reset_id) is None
        with pytest.raises(Stage2BOfflineLedgerError) as missing:
            case.ledger.read_intent(reset_id)
        assert missing.value.code == "LEDGER_INTENT_MISSING"
        return _layer_payloads(context)

    receipt = case.manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=stage,
    )
    assert receipt.commit_id == request.commit_id
    assert len(case.ledger.scan_valid_chain()) == 1


def test_publisher_reentry_is_sticky_even_when_callback_swallows_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    reentry_error_codes: list[str] = []
    real_commit = case.ledger.commit
    reset_id = "publisher-reentry-reset"

    def commit_with_swallowed_reentry(**kwargs):
        try:
            case.manager.reset(
                reset_id=reset_id,
                transaction_nonce="publisher-reentry-reset-nonce",
                new_episode_id=_RESET_EPISODE_ID,
                layout=case.reset_layout,
            )
        except l2b.CacheContractError as exc:
            reentry_error_codes.append(exc.code)
        return real_commit(**kwargs)

    monkeypatch.setattr(case.ledger, "commit", commit_with_swallowed_reentry)
    with pytest.raises(hc.CacheContractError):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=_layer_payloads,
        )

    assert reentry_error_codes
    assert case.ledger.read_decision(reset_id) is None
    with pytest.raises(Stage2BOfflineLedgerError):
        case.ledger.read_intent(reset_id)
    with pytest.raises(l2b.CacheContractError) as poisoned:
        _ = case.manager.state
    _assert_l2b_poisoned(poisoned)


def test_second_publisher_call_is_sticky_even_when_inner_failure_is_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    adapter = _publisher_adapter(case.manager)
    real_commit = case.ledger.commit
    reentry_codes: list[str] = []

    def commit_with_second_publisher_call(**kwargs):
        active = case.manager._active
        assert active is not None
        assert active.receipt_payload is not None
        assert active.receipt_sha256 is not None
        try:
            adapter.publish_exclusive(
                receipt_id=active.transaction_id,
                payload=active.receipt_payload,
                expected_sha256=active.receipt_sha256,
            )
        except l2b.CacheContractError as exc:
            reentry_codes.append(exc.code)
        return real_commit(**kwargs)

    monkeypatch.setattr(case.ledger, "commit", commit_with_second_publisher_call)
    with pytest.raises(hc.CacheContractError):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=_layer_payloads,
        )

    assert reentry_codes == ["L2B_PUBLICATION_POISONED"]
    with pytest.raises(l2b.CacheContractError) as poisoned:
        _ = case.manager.state
    _assert_l2b_poisoned(poisoned)


def test_unarmed_publisher_invocation_sticky_poisons_its_owner(tmp_path: Path) -> None:
    case = _Case(tmp_path)
    adapter = _publisher_adapter(case.manager)

    with pytest.raises(l2b.CacheContractError) as unarmed:
        adapter.publish_exclusive(
            receipt_id="unarmed-publisher",
            payload=b"{}",
            expected_sha256="0" * 64,
        )
    _assert_l2b_poisoned(unarmed)
    with pytest.raises(l2b.CacheContractError) as poisoned:
        _ = case.manager.state
    _assert_l2b_poisoned(poisoned)


def test_l1_valid_wrong_snapshot_is_rejected_before_cas_and_poisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    before_digest = case.manager.state.state_manifest_digest
    real_encode = l2b.encode_stage2b_state_snapshot

    def encode_wrong_state(*, state, expected_layout, expected_registry):
        if state.revision > 0:
            return case.genesis_snapshot
        return real_encode(
            state=state,
            expected_layout=expected_layout,
            expected_registry=expected_registry,
        )

    monkeypatch.setattr(l2b, "encode_stage2b_state_snapshot", encode_wrong_state)
    with pytest.raises(hc.CacheContractError):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=_layer_payloads,
        )

    assert case.ledger.read_decision(request.commit_id) is None
    assert _inner_manager(case.manager)._state.state_manifest_digest == before_digest
    with pytest.raises(l2b.CacheContractError) as poisoned:
        _ = case.manager.state
    _assert_l2b_poisoned(poisoned)


def test_post_cas_wrong_return_type_sticky_poisons_committed_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    inner = _inner_manager(case.manager)
    real_commit = inner.commit_paired

    def commit_then_return_wrong_type(**kwargs):
        real_commit(**kwargs)
        return object()

    monkeypatch.setattr(inner, "commit_paired", commit_then_return_wrong_type)
    with pytest.raises(l2b.CacheContractError) as failure:
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=_layer_payloads,
        )
    _assert_l2b_poisoned(failure)

    decision = case.ledger.read_decision(request.commit_id)
    assert decision is not None
    assert decision.decision.decision_body.result is Stage2BDecisionResult.COMMITTED
    assert _inner_manager(case.manager)._state.revision == 1
    with pytest.raises(l2b.CacheContractError) as poisoned:
        _ = case.manager.state
    _assert_l2b_poisoned(poisoned)


def test_unknown_terminal_exact_reconciles_and_allows_the_inherited_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    real_commit = case.ledger.commit
    real_reconcile = case.ledger.reconcile_terminal
    reconciled: list[tuple[str, str]] = []

    def commit_then_report_unknown(**kwargs):
        real_commit(**kwargs)
        raise Stage2BOfflineLedgerError(
            "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
            "injected exact terminal outcome uncertainty",
        )

    def observe_reconcile(*, transaction_id, expected_record_sha256):
        reconciled.append((transaction_id, expected_record_sha256))
        return real_reconcile(
            transaction_id=transaction_id,
            expected_record_sha256=expected_record_sha256,
        )

    monkeypatch.setattr(case.ledger, "commit", commit_then_report_unknown)
    monkeypatch.setattr(case.ledger, "reconcile_terminal", observe_reconcile)
    receipt = case.manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=_layer_payloads,
    )

    assert receipt.commit_id == request.commit_id
    assert case.manager.state.revision == 1
    decision = case.ledger.read_decision(request.commit_id)
    assert decision is not None
    assert decision.decision.decision_body.result is Stage2BDecisionResult.COMMITTED
    assert len(reconciled) == 2
    assert {transaction_id for transaction_id, _ in reconciled} == {
        request.commit_id
    }
    assert len({expected for _, expected in reconciled}) == 1


def test_unknown_terminal_stable_absence_records_registered_abort_and_poisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    before = case.manager.state.state_manifest_digest
    ordinary_aborts = 0
    absence_aborts = 0
    real_abort = case.ledger.abort
    real_absence_abort = case.ledger.abort_after_reconciled_absence

    def report_unknown_without_publication(**_kwargs):
        raise Stage2BOfflineLedgerError(
            "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
            "injected absent terminal outcome uncertainty",
        )

    def observe_ordinary_abort(**kwargs):
        nonlocal ordinary_aborts
        ordinary_aborts += 1
        return real_abort(**kwargs)

    def observe_absence_abort(**kwargs):
        nonlocal absence_aborts
        absence_aborts += 1
        return real_absence_abort(**kwargs)

    monkeypatch.setattr(case.ledger, "commit", report_unknown_without_publication)
    monkeypatch.setattr(case.ledger, "abort", observe_ordinary_abort)
    monkeypatch.setattr(
        case.ledger,
        "abort_after_reconciled_absence",
        observe_absence_abort,
    )
    with pytest.raises(hc.CacheContractError):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=_layer_payloads,
        )

    decision = case.ledger.read_decision(request.commit_id)
    assert decision is not None
    body = decision.decision.decision_body
    assert body.result is Stage2BDecisionResult.ABORTED
    assert body.abort_code is Stage2BAbortCode.PUBLICATION_FAILED
    assert body.snapshot_manifest_sha256 is None
    assert body.state_manifest_before == before
    assert body.state_manifest_after == before
    abort_receipt = case.ledger.read_object(
        body.success_or_abort_receipt_sha256
    ).payload
    assert b'"failure_point":"TERMINAL_PUBLICATION"' in abort_receipt
    assert ordinary_aborts == 0
    assert absence_aborts == 1
    assert _inner_manager(case.manager)._state.state_manifest_digest == before
    with pytest.raises(l2b.CacheContractError) as poisoned:
        _ = case.manager.state
    _assert_l2b_poisoned(poisoned)


def test_unknown_terminal_conflict_never_overwrites_or_guesses_an_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    before = case.manager.state.state_manifest_digest
    ordinary_aborts = 0
    absence_aborts = 0
    real_abort = case.ledger.abort
    real_absence_abort = case.ledger.abort_after_reconciled_absence

    def observe_ordinary_abort(**kwargs):
        nonlocal ordinary_aborts
        ordinary_aborts += 1
        return real_abort(**kwargs)

    def observe_absence_abort(**kwargs):
        nonlocal absence_aborts
        absence_aborts += 1
        return real_absence_abort(**kwargs)

    def publish_conflicting_abort_then_report_unknown(**kwargs):
        case.ledger.abort(
            intent=kwargs["intent"],
            abort_code=Stage2BAbortCode.STAGING_FAILED,
            failure_point=Stage2BFailurePoint.STAGING,
        )
        raise Stage2BOfflineLedgerError(
            "LEDGER_PUBLICATION_OUTCOME_UNKNOWN",
            "injected conflicting terminal outcome uncertainty",
        )

    monkeypatch.setattr(
        case.ledger,
        "commit",
        publish_conflicting_abort_then_report_unknown,
    )
    monkeypatch.setattr(case.ledger, "abort", observe_ordinary_abort)
    monkeypatch.setattr(
        case.ledger,
        "abort_after_reconciled_absence",
        observe_absence_abort,
    )
    with pytest.raises(hc.CacheContractError):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=_layer_payloads,
        )

    decision = case.ledger.read_decision(request.commit_id)
    assert decision is not None
    body = decision.decision.decision_body
    assert body.result is Stage2BDecisionResult.ABORTED
    assert body.abort_code is Stage2BAbortCode.STAGING_FAILED
    assert body.state_manifest_after == before
    assert ordinary_aborts == 1
    assert absence_aborts == 0
    assert _inner_manager(case.manager)._state.state_manifest_digest == before
    with pytest.raises(l2b.CacheContractError) as poisoned:
        _ = case.manager.state
    _assert_l2b_poisoned(poisoned)


def test_reset_prepublisher_failure_uses_precondition_abort_and_returns_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _Case(tmp_path)
    inner = _inner_manager(case.manager)
    before = case.manager.state.state_manifest_digest

    def fail_before_reset_publisher(**_kwargs):
        raise RuntimeError("injected reset prepublisher failure")

    monkeypatch.setattr(inner, "reset", fail_before_reset_publisher)
    with pytest.raises(RuntimeError, match="reset prepublisher"):
        case.manager.reset(
            reset_id="reset-prepublisher-failure",
            transaction_nonce="reset-prepublisher-failure-nonce",
            new_episode_id=_RESET_EPISODE_ID,
            layout=case.reset_layout,
        )

    decision = case.ledger.read_decision("reset-prepublisher-failure")
    assert decision is not None
    body = decision.decision.decision_body
    assert body.result is Stage2BDecisionResult.ABORTED
    assert body.abort_code is Stage2BAbortCode.PRECONDITION_FAILED
    assert body.snapshot_manifest_sha256 is None
    assert body.state_manifest_before == before
    assert body.state_manifest_after == before
    abort_receipt = case.ledger.read_object(
        body.success_or_abort_receipt_sha256
    ).payload
    assert b'"failure_point":"AFTER_PREPARED"' in abort_receipt
    assert case.manager.state.state_manifest_digest == before


def test_fresh_factory_rejects_committed_pending_and_identity_mismatched_ledgers(
    tmp_path: Path,
) -> None:
    committed = _Case(tmp_path / "committed")
    committed.manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=_request(committed.layout, chunk_id=0),
        stage_callback=_layer_payloads,
    )
    with pytest.raises(l2b.CacheContractError) as committed_error:
        l2b.Stage2BLedgeredHybridCacheManager.for_synthetic_tests(
            registry=committed.registry,
            episode_id=_EPISODE_ID,
            layout=committed.layout,
            ledger=committed.ledger,
            staging_variant=CACHStagingVariant.CACH_A,
        )
    assert committed_error.value.code == "L2B_RECOVERY_REQUIRED"

    pending = _Case(tmp_path / "pending")
    pending_request = _request(
        pending.layout,
        chunk_id=0,
        commit_id="fresh-factory-pending",
        transaction_nonce="fresh-factory-pending-nonce",
    )

    def inspect_pending(context):
        assert pending.ledger.read_intent(pending_request.commit_id)
        with pytest.raises(l2b.CacheContractError) as pending_error:
            l2b.Stage2BLedgeredHybridCacheManager.for_synthetic_tests(
                registry=pending.registry,
                episode_id=_EPISODE_ID,
                layout=pending.layout,
                ledger=pending.ledger,
                staging_variant=CACHStagingVariant.CACH_A,
            )
        assert pending_error.value.code == "L2B_RECOVERY_REQUIRED"
        return _layer_payloads(context)

    pending.manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=pending_request,
        stage_callback=inspect_pending,
    )

    mismatched = _Case(tmp_path / "mismatched")
    with pytest.raises(l2b.CacheContractError) as identity_error:
        l2b.Stage2BLedgeredHybridCacheManager.for_synthetic_tests(
            registry=mismatched.registry,
            episode_id="different-genesis-episode",
            layout=mismatched.layout,
            ledger=mismatched.ledger,
            staging_variant=CACHStagingVariant.CACH_A,
        )
    assert identity_error.value.code in {
        "L2B_LEDGER_MISMATCH",
        "L2B_RECOVERY_REQUIRED",
    }

    orphan = _Case(tmp_path / "orphan")
    orphan.ledger.publish_object(payload=b"orphan-before-prepared")
    with pytest.raises(l2b.CacheContractError) as orphan_error:
        l2b.Stage2BLedgeredHybridCacheManager.for_synthetic_tests(
            registry=orphan.registry,
            episode_id=_EPISODE_ID,
            layout=orphan.layout,
            ledger=orphan.ledger,
            staging_variant=CACHStagingVariant.CACH_A,
        )
    assert orphan_error.value.code == "L2B_RECOVERY_REQUIRED"


def test_invalid_request_is_rejected_without_prepared_or_root_change(
    tmp_path: Path,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    invalid = replace(request, expected_revision=1)
    entries_before = tuple(sorted(path.name for path in case.ledger.root.iterdir()))

    with pytest.raises(hc.CacheContractError, match="COMMIT_STALE_REVISION"):
        case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=invalid,
            stage_callback=_layer_payloads,
        )

    assert tuple(sorted(path.name for path in case.ledger.root.iterdir())) == (
        entries_before
    )
    assert case.ledger.scan_valid_chain() == ()
    with pytest.raises(Stage2BOfflineLedgerError) as missing:
        case.ledger.read_intent(invalid.commit_id)
    assert missing.value.code == "LEDGER_INTENT_MISSING"


def test_invalid_source_request_callback_proof_and_layout_never_write_prepared(
    tmp_path: Path,
) -> None:
    case = _Case(tmp_path)
    request = _request(case.layout, chunk_id=0)
    bad_proof = replace(
        request.proof,
        action_tensor_digest="d" * 64,
    )
    bad_proof_request = replace(request, proof=bad_proof)
    foreign_layout = _layout(episode_id="foreign-layout")
    foreign_request = _request(foreign_layout, chunk_id=0)
    calls = (
        lambda: case.manager.commit_paired(
            commit_source=hc.CommitSource.SELF_FORCING,
            request=request,
            stage_callback=_layer_payloads,
        ),
        lambda: case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=None,
            stage_callback=_layer_payloads,
        ),
        lambda: case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=None,
        ),
        lambda: case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=bad_proof_request,
            stage_callback=_layer_payloads,
        ),
        lambda: case.manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=foreign_request,
            stage_callback=_layer_payloads,
        ),
    )

    pristine = _root_fingerprint(case.ledger)
    for call in calls:
        with pytest.raises(l2b.CacheContractError):
            call()
        assert _root_fingerprint(case.ledger) == pristine
        assert case.ledger.scan_valid_chain() == ()


def test_scope_is_synthetic_cpu_only() -> None:
    assert l2b.Stage2BLedgeredHybridCacheManager.__module__.endswith(
        "hybrid_cache_stage2b_ledgered"
    )
    assert EncodedStage2BStateSnapshot.__module__.endswith(
        "stage2b_state_snapshot"
    )
    assert Stage2BFailurePoint.STAGING.value == "STAGING"


def test_non_synthetic_factory_is_rejected_before_ledger_mutation(
    tmp_path: Path,
) -> None:
    case = _Case(tmp_path)
    entries = tuple(sorted(path.name for path in case.ledger.root.iterdir()))

    with pytest.raises(l2b.CacheContractError) as exc:
        l2b.Stage2BLedgeredHybridCacheManager.initialize_fresh(
            registry=case.registry,
            episode_id=_EPISODE_ID,
            layout=case.layout,
            ledger=case.ledger,
            staging_variant=CACHStagingVariant.CACH_A,
            synthetic_test_only=False,
        )
    assert exc.value.code == "L2B_CPU_SYNTHETIC_ONLY"
    assert tuple(sorted(path.name for path in case.ledger.root.iterdir())) == entries
