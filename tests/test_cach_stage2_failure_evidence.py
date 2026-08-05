"""Synthetic failure injection and frozen temporary evidence admission."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import time

import pytest
import torch

from sana_wam.cach.stage2_failure_evidence import (
    Stage2FailureInjectionReceipt,
    freeze_stage2_failure_evidence,
)
from sana_wam.model.action_chunk_layout import (
    build_synthetic_chunk_action_layout_for_tests,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)
from sana_wam.model.video_backbone.sana import hybrid_cache as hc


class _SuccessMustNotPublish:
    def __init__(self):
        self.calls = 0

    def publish_exclusive(self, **_kwargs):
        self.calls += 1
        raise AssertionError("failed staging must not publish a success receipt")


def _request_and_manager():
    valid_raw_count = 17
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=valid_raw_count,
        video_stride=1,
        source_row_label="failure-row",
        episode_label="failure-episode",
    )
    layout = build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=valid_raw_count,
        video_valid_mask=(True,) * valid_raw_count,
        action_valid_mask=(True,) * (valid_raw_count - 1),
    )
    chunk = layout.chunks[0]
    video = torch.randn(1, 3, 3, 2, 2)
    frame_mask = torch.ones(1, 3, dtype=torch.bool)
    actions = torch.randn(1, 16, 20)
    action_mask = torch.ones(1, 16, dtype=torch.bool)
    content_time = hc.ContentTime.from_layout(
        episode_id="failure-episode",
        episode_epoch=0,
        layout=layout,
        chunk=chunk,
    )
    source_proof = hc.TeacherForcingDatasetPairProof(
        dataset_manifest_sha256="a" * 64,
        dataset_episode_id="failure-episode",
        dataset_row_identity="failure-episode:chunk-0",
        dataset_row_start=0,
        dataset_row_end_exclusive=valid_raw_count,
        layout_spec_sha256=layout.layout_spec_sha256,
        layout_instance_digest=layout.layout_instance_digest,
        video_tensor_digest=hc.tensor_digest(video),
        frame_valid_mask_digest=hc.tensor_digest(frame_mask),
        action_tensor_digest=hc.tensor_digest(actions),
        action_mask_digest=hc.tensor_digest(action_mask),
        row_order_manifest_sha256="b" * 64,
    )
    request = hc.TeacherForcingCommitRequest(
        commit_id="injected-commit",
        transaction_nonce="injected-nonce",
        expected_episode_id="failure-episode",
        expected_episode_epoch=0,
        expected_revision=0,
        content_time=content_time,
        video=video,
        frame_valid_mask=frame_mask,
        actions=actions,
        action_valid_mask=action_mask,
        proof=source_proof,
    )
    publisher = _SuccessMustNotPublish()
    registry = hc.LayerRegistry(
        layers=(
            hc.LayerSpec(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="synthetic-mini-gdn",
                main_shortconv_enabled=True,
            ),
        )
    )
    manager = hc.HybridCacheManager.for_synthetic_tests(
        registry=registry,
        episode_id="failure-episode",
        layout=layout,
        receipt_publisher=publisher,
    )
    return manager, publisher, request


def test_staging_failure_freezes_non_scientific_evidence(tmp_path) -> None:
    manager, success_publisher, request = _request_and_manager()
    before_pointer = manager._state
    before = manager.state
    callback_calls = 0

    def inject_before_vendor(_context):
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("registered Stage-2 injection")

    with pytest.raises(hc.CacheContractError) as caught:
        manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=inject_before_vendor,
        )
    assert caught.value.code == "COMMIT_STAGING_FAILED"
    after = manager.state
    assert callback_calls == 1
    assert success_publisher.calls == 0
    assert manager._state is before_pointer
    assert after.state_manifest_digest == before.state_manifest_digest
    assert after.revision == before.revision == 0
    assert after.action_cursor == before.action_cursor == 0

    receipt = Stage2FailureInjectionReceipt(
        receipt_id="stage2-injected-failure-0",
        commit_id=request.commit_id,
        transaction_nonce=request.transaction_nonce,
        episode_id=before.episode_id,
        episode_epoch=before.episode_epoch,
        chunk_id=request.content_time.chunk_id,
        layout_instance_digest=request.content_time.layout_instance_digest,
        source_proof_digest=request.proof.source_proof_digest,
        failure_code=caught.value.code,
        injection_point="stage_callback_before_vendor_forward",
        state_manifest_before=before.state_manifest_digest,
        state_manifest_after=after.state_manifest_digest,
        revision_before=before.revision,
        revision_after=after.revision,
        action_cursor_before=before.action_cursor,
        action_cursor_after=after.action_cursor,
        recorded_at_monotonic_ns=time.monotonic_ns(),
    )
    retained_root = os.environ.get("CACH_STAGE2_RETAIN_FAILURE_ROOT")
    if retained_root is None:
        evidence_root = (tmp_path / "frozen-failure").resolve()
    else:
        evidence_root = Path(retained_root)
        if not evidence_root.is_absolute():
            raise ValueError("CACH_STAGE2_RETAIN_FAILURE_ROOT must be absolute")
    frozen = freeze_stage2_failure_evidence(evidence_root, receipt)
    assert frozen.receipt_sha256 == receipt.receipt_sha256
    assert stat.S_IMODE(os.stat(frozen.root).st_mode) == 0o500
    assert stat.S_IMODE(os.stat(frozen.receipt_path).st_mode) == 0o400
    assert frozen.receipt_path.read_bytes() == receipt.canonical_bytes
    with pytest.raises(FileExistsError):
        freeze_stage2_failure_evidence(frozen.root, receipt)


def test_failure_receipt_rejects_any_live_state_advance() -> None:
    with pytest.raises(ValueError, match="unchanged live state"):
        Stage2FailureInjectionReceipt(
            receipt_id="bad",
            commit_id="commit",
            transaction_nonce="nonce",
            episode_id="episode",
            episode_epoch=0,
            chunk_id=0,
            layout_instance_digest="1" * 64,
            source_proof_digest="2" * 64,
            failure_code="COMMIT_STAGING_FAILED",
            injection_point="stage_callback_before_vendor_forward",
            state_manifest_before="3" * 64,
            state_manifest_after="4" * 64,
            revision_before=0,
            revision_after=1,
            action_cursor_before=0,
            action_cursor_after=16,
            recorded_at_monotonic_ns=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("failure_code", "OTHER_FAILURE"),
        ("injection_point", "after_vendor_forward"),
    ),
)
def test_failure_receipt_rejects_unregistered_failure_shape(field, value) -> None:
    fields = {
        "receipt_id": "bad-shape",
        "commit_id": "commit",
        "transaction_nonce": "nonce",
        "episode_id": "episode",
        "episode_epoch": 0,
        "chunk_id": 0,
        "layout_instance_digest": "1" * 64,
        "source_proof_digest": "2" * 64,
        "failure_code": "COMMIT_STAGING_FAILED",
        "injection_point": "stage_callback_before_vendor_forward",
        "state_manifest_before": "3" * 64,
        "state_manifest_after": "3" * 64,
        "revision_before": 0,
        "revision_after": 0,
        "action_cursor_before": 0,
        "action_cursor_after": 0,
        "recorded_at_monotonic_ns": 1,
    }
    fields[field] = value
    with pytest.raises(ValueError, match=f"{field} is not admitted"):
        Stage2FailureInjectionReceipt(**fields)


def test_deploy_source_never_falls_back_to_commanded_or_teacher_values() -> None:
    manager, success_publisher, teacher_request = _request_and_manager()
    before_pointer = manager._state
    callback_calls = 0

    def forbidden_fallback(_context):
        nonlocal callback_calls
        callback_calls += 1
        raise AssertionError("deploy must reject before staging")

    with pytest.raises(hc.CacheContractError) as caught:
        manager.commit_paired(
            commit_source=hc.CommitSource.DEPLOY_APPLIED_ACK,
            # A complete teacher/command tensor still cannot substitute for an
            # environment-confirmed canonical applied-action acknowledgement.
            request=teacher_request,
            stage_callback=forbidden_fallback,
        )
    assert caught.value.code == "COMMIT_DEPLOY_ACK_MISSING"
    assert callback_calls == 0
    assert success_publisher.calls == 0
    assert manager._state is before_pointer
    assert manager.state.state_manifest_digest == before_pointer.state_manifest_digest
    assert manager.state.revision == manager.state.action_cursor == 0
