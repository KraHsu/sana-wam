"""Stage-2B admission for exact history identity and atomic cache publication."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sana_wam.cach.committed_action_history import (
    CommittedActionHistory,
    CommittedActionHistoryError,
    committed_action_tensor_digest,
)
from sana_wam.cach.stage2b_receipt_store import Stage2BFilesystemReceiptStore
from sana_wam.model.action_chunk_layout import (
    build_synthetic_chunk_action_layout_for_tests,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)
from sana_wam.model.cach_numerical_core import CACHNumericalCore
from sana_wam.model.video_backbone.sana import hybrid_cache as hc
from sana_wam.model.video_backbone.sana import hybrid_cache_stage2b as h2


def _layout(*, episode_label: str = "stage2b-episode"):
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


def _fixed_actions(chunk, *, offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.arange(
        chunk.action_slot_capacity * 20,
        dtype=torch.float32,
    ).reshape(1, chunk.action_slot_capacity, 20)
    actions = actions + offset
    mask = torch.tensor(chunk.action_valid_mask, dtype=torch.bool).view(1, -1)
    return actions.masked_fill((~mask)[:, :, None], 0), mask


def _empty_history(layout) -> CommittedActionHistory:
    return CommittedActionHistory.empty(
        episode_id="stage2b-episode",
        episode_epoch=0,
        layout_spec_sha256=layout.layout_spec_sha256,
        layout_instance_digest=layout.layout_instance_digest,
        batch_size=1,
        action_dim=20,
        dtype=torch.float32,
        device="cpu",
    )


def test_history_appends_only_valid_prefix_and_exposes_detached_exact_view() -> None:
    layout = _layout()
    history = _empty_history(layout)
    first = layout.chunks[0]
    actions, mask = _fixed_actions(first)

    empty_view = history.view_for_chunk(chunk_id=0, action_start=0)
    assert empty_view.cursor == 0
    assert empty_view.chunk_ids == ()
    assert empty_view.clone_actions().shape == (1, 0, 20)

    history, duplicate = history.append(
        chunk_id=0,
        action_start=first.action_start,
        action_end_exclusive=first.action_end,
        fixed_slot_actions=actions,
        action_valid_mask=mask,
        commit_source="teacher_forcing_dataset_pair",
        source_proof_digest="a" * 64,
        commit_id="commit-0",
    )
    assert duplicate is False
    assert history.action_cursor == first.action_end
    assert history.committed_through_chunk == 0
    assert history.spans[0]._actions.shape[1] == first.valid_action_count

    view = history.view_for_chunk(
        chunk_id=1,
        action_start=first.action_end,
    )
    expected = actions[:, : first.valid_action_count]
    assert view.committed_actions_digest == committed_action_tensor_digest(expected)
    clone = view.clone_actions()
    torch.testing.assert_close(clone, expected, atol=0, rtol=0)
    clone.add_(1000)
    torch.testing.assert_close(view.clone_actions(), expected, atol=0, rtol=0)


def test_history_rejects_gap_padding_deploy_and_conflicting_duplicate() -> None:
    layout = _layout()
    history = _empty_history(layout)
    first = layout.chunks[0]
    actions, mask = _fixed_actions(first)
    kwargs = dict(
        chunk_id=0,
        action_start=first.action_start,
        action_end_exclusive=first.action_end,
        fixed_slot_actions=actions,
        action_valid_mask=mask,
        commit_source="teacher_forcing_dataset_pair",
        source_proof_digest="a" * 64,
        commit_id="commit-0",
    )

    with pytest.raises(CommittedActionHistoryError) as gap:
        history.append(**{**kwargs, "chunk_id": 1})
    assert gap.value.code == "ACTION_HISTORY_COVERAGE_GAP"

    with pytest.raises(CommittedActionHistoryError) as deploy:
        history.append(**{**kwargs, "commit_source": "deploy_applied_ack"})
    assert deploy.value.code == "ACTION_HISTORY_SOURCE_UNAUTHORIZED"

    if first.valid_action_count < first.action_slot_capacity:
        padded = actions.clone()
        padded[:, first.valid_action_count] = 1
        with pytest.raises(CommittedActionHistoryError) as padding:
            history.append(**{**kwargs, "fixed_slot_actions": padded})
        assert padding.value.code == "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH"

    committed, duplicate = history.append(**kwargs)
    assert duplicate is False
    same, duplicate = committed.append(**kwargs)
    assert same is committed and duplicate is True
    changed = actions.clone()
    changed[:, 0] += 1
    with pytest.raises(CommittedActionHistoryError) as conflict:
        committed.append(**{**kwargs, "fixed_slot_actions": changed})
    assert conflict.value.code == "ACTION_HISTORY_DUPLICATE_CONFLICT"

    corrupted = committed.clone()
    corrupted.spans[0]._actions.add_(1)
    with pytest.raises(CommittedActionHistoryError) as integrity:
        corrupted.append(**kwargs)
    assert integrity.value.code == "ACTION_HISTORY_MUTATION"


class _Publisher:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def publish_exclusive(self, *, receipt_id, payload, expected_sha256):
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        self.calls.append((receipt_id, expected_sha256))
        return hc.ReceiptPublication(
            receipt_id=receipt_id,
            sha256=expected_sha256,
            durable=True,
            readback_verified=True,
        )


def _manager_and_layout(*, publisher=None):
    layout = _layout()
    registry = hc.LayerRegistry(
        layers=(
            hc.LayerSpec(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="synthetic.Stage2BGDN",
                main_shortconv_enabled=True,
            ),
        )
    )
    manager = h2.Stage2BHybridCacheManager.for_synthetic_tests(
        registry=registry,
        episode_id="stage2b-episode",
        layout=layout,
        receipt_publisher=publisher or _Publisher(),
        staging_variant=h2.CACHStagingVariant.CACH_A,
    )
    return manager, layout


def _request(layout, chunk_id: int):
    chunk = layout.chunks[chunk_id]
    video = torch.arange(
        3 * len(chunk.latent_valid_mask) * 4,
        dtype=torch.float32,
    ).reshape(1, 3, len(chunk.latent_valid_mask), 2, 2)
    video = video + chunk_id * 1000
    frame_mask = torch.tensor(chunk.latent_valid_mask, dtype=torch.bool).view(1, -1)
    video = video.masked_fill((~frame_mask)[:, None, :, None, None], 0)
    actions, action_mask = _fixed_actions(chunk, offset=chunk_id * 1000)
    proof = hc.TeacherForcingDatasetPairProof(
        dataset_manifest_sha256="b" * 64,
        dataset_episode_id="stage2b-episode",
        dataset_row_identity=f"stage2b-episode:chunk-{chunk_id}",
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
        commit_id=f"commit-{chunk_id}",
        transaction_nonce=f"nonce-{chunk_id}",
        expected_episode_id="stage2b-episode",
        expected_episode_epoch=0,
        expected_revision=chunk_id,
        content_time=hc.ContentTime.from_layout(
            episode_id="stage2b-episode",
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


def _payloads(context):
    value = float(context.content_time.chunk_id + 1)
    return (
        hc.StagedLayerPayload(
            layer_index=0,
            kind=hc.LayerKind.GDN_FULL_HISTORY,
            tensors={
                "main_s_kv": torch.full((1, 2, 4, 4), value),
                "main_s_z": torch.full((1, 2, 4, 1), value),
                "main_shortconv_left_context": torch.full((1, 2, 3), value),
                "ffn_tconv_left_context": torch.full((1, 2, 2, 3), value),
            },
        ),
    )


def test_manager_publishes_cache_and_history_atomically_and_reset_clears_both() -> None:
    manager, layout = _manager_and_layout()
    observed: list[torch.Tensor] = []

    def stage(context):
        history = context.previous_committed_action_history
        assert history.cursor == context.content_time.action_token_start
        observed.append(history.clone_actions())
        return _payloads(context)

    first_request = _request(layout, 0)
    first_receipt = manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=first_request,
        stage_callback=stage,
    )
    assert observed[0].shape == (1, 0, 20)
    assert first_receipt.action_history_digest_before is None
    assert first_receipt.action_cursor_before == 0
    assert first_receipt.action_cursor_after == layout.chunks[0].action_end
    assert manager.state.committed_action_history is not None

    second_request = _request(layout, 1)
    second_receipt = manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=second_request,
        stage_callback=stage,
    )
    assert observed[1].shape == (1, layout.chunks[1].action_start, 20)
    assert second_receipt.action_history_digest_before == (
        first_receipt.action_history_digest_after
    )
    state = manager.state
    assert state.action_cursor == layout.chunks[1].action_end
    assert state.committed_action_history is not None
    assert state.committed_action_history.action_cursor == state.action_cursor

    retry = manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=second_request,
        stage_callback=lambda _context: (_ for _ in ()).throw(
            AssertionError("idempotent retry must not restage")
        ),
    )
    assert retry is second_receipt

    next_layout = _layout(episode_label="stage2b-next")
    reset = manager.reset(
        reset_id="reset-stage2b",
        transaction_nonce="reset-nonce",
        new_episode_id="stage2b-next",
        layout=next_layout,
    )
    assert reset.action_history_digest_before == second_receipt.action_history_digest_after
    assert reset.action_cursor_before == state.action_cursor
    assert reset.action_history_digest_after is None
    assert reset.staging_variant is h2.CACHStagingVariant.CACH_A
    assert manager.state.committed_action_history is None
    assert manager.state.staging_variant is h2.CACHStagingVariant.CACH_A
    assert manager.state.action_cursor == manager.state.revision == 0
    assert "commit-0" in manager._completed
    with pytest.raises(hc.CacheContractError) as reused_id:
        manager.reset(
            reset_id="commit-0",
            transaction_nonce="cross-type-reuse",
            new_episode_id="must-not-publish",
            layout=next_layout,
        )
    assert reused_id.value.code == "COMMIT_DUPLICATE_CONFLICT"


def test_manager_receipts_round_trip_through_strict_filesystem_store(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage2b-receipts"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    manager, layout = _manager_and_layout(publisher=store)

    commit = manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=_request(layout, 0),
        stage_callback=_payloads,
    )
    stored_commit = store.read_stable(commit.commit_id)
    assert stored_commit.payload == commit.canonical_bytes
    assert stored_commit.sha256 == commit.receipt_sha256
    with pytest.raises(hc.CacheContractError) as forged_commit:
        replace(commit, paired_payload_digest="d" * 64)
    assert forged_commit.value.code == "COMMIT_RECEIPT_INVALID"

    next_layout = _layout(episode_label="stage2b-filesystem-reset")
    reset = manager.reset(
        reset_id="reset-filesystem-store",
        transaction_nonce="reset-filesystem-store-nonce",
        new_episode_id="stage2b-filesystem-reset",
        layout=next_layout,
    )
    stored_reset = store.read_stable(reset.reset_id)
    assert stored_reset.payload == reset.canonical_bytes
    assert stored_reset.sha256 == reset.receipt_sha256
    with pytest.raises(hc.CacheContractError) as forged_reset:
        replace(reset, aborted_pending_commit_id=reset.reset_id)
    assert forged_reset.value.code == "COMMIT_RECEIPT_INVALID"
    assert len(tuple(root.iterdir())) == 2


def test_stager_history_mutation_fails_without_advancing_live_pointer() -> None:
    manager, layout = _manager_and_layout()
    manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=_request(layout, 0),
        stage_callback=_payloads,
    )
    before = manager.state.state_manifest_digest

    def mutate_history(context):
        context.previous_committed_action_history._actions.add_(1)
        return _payloads(context)

    with pytest.raises(hc.CacheContractError) as caught:
        manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=_request(layout, 1),
            stage_callback=mutate_history,
        )
    assert caught.value.code == "ACTION_HISTORY_MUTATION"
    after = manager.state
    assert after.state_manifest_digest == before
    assert after.revision == 1
    assert after.action_cursor == layout.chunks[0].action_end
    assert after.committed_action_history is not None


def test_history_mutate_then_restore_and_denoise_mutation_fail_closed() -> None:
    manager, layout = _manager_and_layout()
    manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=_request(layout, 0),
        stage_callback=_payloads,
    )
    before = manager.state.state_manifest_digest

    def mutate_then_restore(context):
        history_actions = context.previous_committed_action_history._actions
        original = history_actions.detach().clone()
        history_actions.add_(1)
        history_actions.copy_(original)
        return _payloads(context)

    with pytest.raises(hc.CacheContractError) as staged:
        manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=_request(layout, 1),
            stage_callback=mutate_then_restore,
        )
    assert staged.value.code == "ACTION_HISTORY_MUTATION"
    assert manager.state.state_manifest_digest == before

    read_view = manager.snapshot_for_denoise(
        expected_episode_id="stage2b-episode",
        expected_episode_epoch=0,
        expected_revision=1,
        expected_next_chunk_id=1,
    )
    scratch = read_view.export_scratch()
    history_view = read_view.previous_committed_action_history
    assert history_view is not None
    original = history_view._actions.detach().clone()
    history_view._actions.add_(1)
    history_view._actions.copy_(original)
    with pytest.raises(hc.CacheContractError) as denoise:
        manager.finish_denoise(read_view=read_view, scratch=scratch)
    assert denoise.value.code == "ACTION_HISTORY_MUTATION"
    assert manager.state.state_manifest_digest == before


def test_denoise_view_variant_is_bound_to_snapshot_and_live_manager() -> None:
    manager, layout = _manager_and_layout()
    manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=_request(layout, 0),
        stage_callback=_payloads,
    )
    read_view = manager.snapshot_for_denoise(
        expected_episode_id="stage2b-episode",
        expected_episode_epoch=0,
        expected_revision=1,
        expected_next_chunk_id=1,
    )
    with pytest.raises(hc.CacheContractError) as replaced:
        replace(
            read_view,
            staging_variant=h2.CACHStagingVariant.REF_GDN_CORRECTED,
        )
    assert replaced.value.code == "CACHE_VARIANT_OR_STATE_MISMATCH"

    scratch = read_view.export_scratch()
    core = CACHNumericalCore(action_dim=20)

    class ForgedView:
        next_chunk_id = read_view.next_chunk_id
        staging_variant = h2.CACHStagingVariant.REF_GDN_CORRECTED
        state_manifest_digest = read_view.state_manifest_digest
        _state_identity = read_view._state_identity

        @staticmethod
        def assert_integrity():
            return None

    with pytest.raises(TypeError, match="exact Stage2B"):
        core.run_readonly_chunk(
            staging_variant=h2.CACHStagingVariant.REF_GDN_CORRECTED,
            cache_read_view=ForgedView(),
            cache_scratch=scratch,
        )
    object.__setattr__(
        read_view,
        "staging_variant",
        h2.CACHStagingVariant.REF_GDN_CORRECTED,
    )
    with pytest.raises(hc.CacheContractError) as exported:
        read_view.export_scratch()
    assert exported.value.code == "CACHE_VARIANT_OR_STATE_MISMATCH"
    with pytest.raises(hc.CacheContractError) as numerical:
        core.run_readonly_chunk(
            staging_variant=h2.CACHStagingVariant.REF_GDN_CORRECTED,
            cache_read_view=read_view,
            cache_scratch=scratch,
        )
    assert numerical.value.code == "CACHE_VARIANT_OR_STATE_MISMATCH"
    with pytest.raises(hc.CacheContractError) as finished:
        manager.finish_denoise(read_view=read_view, scratch=scratch)
    assert finished.value.code == "CACHE_VARIANT_OR_STATE_MISMATCH"


class _FailingPublisher:
    def publish_exclusive(self, **_kwargs):
        raise OSError("injected Stage-2B durability failure")


def test_stage_and_receipt_failures_never_half_publish_history() -> None:
    manager, layout = _manager_and_layout()
    before = manager._state

    def fail_stage(_context):
        raise ValueError("injected numerical failure")

    with pytest.raises(hc.CacheContractError) as stage_failure:
        manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=_request(layout, 0),
            stage_callback=fail_stage,
        )
    assert stage_failure.value.code == "COMMIT_STAGING_FAILED"
    assert manager._state is before
    assert manager.state.committed_action_history is None
    recovered = manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=_request(layout, 0),
        stage_callback=_payloads,
    )
    assert recovered.revision_after == 1

    failed_manager, failed_layout = _manager_and_layout(
        publisher=_FailingPublisher()
    )
    failed_before = failed_manager._state
    with pytest.raises(hc.CacheContractError) as receipt_failure:
        failed_manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=_request(failed_layout, 0),
            stage_callback=_payloads,
        )
    assert receipt_failure.value.code == "COMMIT_RECEIPT_PUBLICATION_FAILED"
    assert failed_manager._state is failed_before
    assert failed_manager._state.committed_action_history is None
    with pytest.raises(hc.CacheContractError) as poisoned:
        _ = failed_manager.state
    assert poisoned.value.code == "CACHE_PUBLICATION_POISONED"


def test_lazy_staging_iterable_is_rejected_without_iteration_or_publication() -> None:
    manager, layout = _manager_and_layout()
    before = manager._state
    iterated = False

    def lazy_payloads(context):
        def generate():
            nonlocal iterated
            iterated = True
            context.actions.add_(1)
            yield from _payloads(context)

        return generate()

    with pytest.raises(hc.CacheContractError) as caught:
        manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=_request(layout, 0),
            stage_callback=lazy_payloads,
        )
    assert caught.value.code == "CACHE_SCHEMA_MISMATCH"
    assert iterated is False
    assert manager._state is before
    assert manager.state.committed_action_history is None


class _ReentrantPublisher(_Publisher):
    def __init__(self):
        super().__init__()
        self.manager = None
        self.reentry_error = None

    def publish_exclusive(self, *, receipt_id, payload, expected_sha256):
        if self.reentry_error is None:
            try:
                self.manager.reset(
                    reset_id="publisher-reentrant-reset",
                    transaction_nonce="publisher-reentrant-reset-nonce",
                    new_episode_id="publisher-must-not-resurrect",
                    layout=_layout(episode_label="publisher-must-not-resurrect"),
                )
            except hc.CacheContractError as exc:
                self.reentry_error = exc
        return super().publish_exclusive(
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=expected_sha256,
        )


def test_publisher_reentry_is_rejected_before_it_can_reset_live_state() -> None:
    publisher = _ReentrantPublisher()
    manager, layout = _manager_and_layout(publisher=publisher)
    publisher.manager = manager

    receipt = manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=_request(layout, 0),
        stage_callback=_payloads,
    )
    assert publisher.reentry_error is not None
    assert publisher.reentry_error.code == "CACHE_PUBLICATION_REENTRANT"
    assert receipt.revision_after == 1
    state = manager.state
    assert state.episode_id == "stage2b-episode"
    assert state.episode_epoch == 0
    assert state.revision == 1


def test_reset_during_staging_clears_history_and_forces_commit_cas_failure() -> None:
    manager, layout = _manager_and_layout()
    next_layout = _layout(episode_label="stage2b-reset-during-stage")

    def reset_then_stage(context):
        manager.reset(
            reset_id="stage2b-reset-during-stage",
            transaction_nonce="stage2b-reset-during-stage-nonce",
            new_episode_id="stage2b-reset-during-stage",
            layout=next_layout,
        )
        return _payloads(context)

    with pytest.raises(hc.CacheContractError) as caught:
        manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=_request(layout, 0),
            stage_callback=reset_then_stage,
        )
    assert caught.value.code == "COMMIT_STALE_REVISION"
    state = manager.state
    assert state.episode_id == "stage2b-reset-during-stage"
    assert state.episode_epoch == 1
    assert state.revision == state.action_cursor == 0
    assert state.committed_action_history is None
