from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sana_wam.model.action_chunk_layout import (
    build_synthetic_chunk_action_layout_for_tests,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)


_MODULE_NAME = "_cach_hybrid_cache_contract_under_test"
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/sana_wam/model/video_backbone/sana/hybrid_cache.py"
)
if _MODULE_NAME not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    assert _SPEC is not None and _SPEC.loader is not None
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_MODULE_NAME] = _MODULE
    _SPEC.loader.exec_module(_MODULE)
hc = sys.modules[_MODULE_NAME]


DATASET_SHA = "c" * 64
ROW_ORDER_SHA = "d" * 64


def _layout_for_latent_count(
    valid_latent_count: int,
    *,
    episode_label: str = "episode-0",
):
    valid_raw_count = (valid_latent_count - 1) * 8 + 1
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=valid_raw_count,
        video_stride=1,
        source_row_label=f"{episode_label}-row",
        episode_label=episode_label,
    )
    return build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=valid_raw_count,
        video_valid_mask=(True,) * valid_raw_count,
        action_valid_mask=(True,) * (valid_raw_count - 1),
    )


L5_LAYOUT = _layout_for_latent_count(5)
L5_LAYOUT_EPISODE_1 = _layout_for_latent_count(
    5,
    episode_label="episode-1",
)


class _FsyncReceiptPublisher:
    """Minimal test publisher exercising exclusive write/fsync/read-back."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True)
        self.calls = []
        self.state_revision = None

    def publish_exclusive(self, *, receipt_id, payload, expected_sha256):
        if self.state_revision is not None:
            revision_during_publish = self.state_revision()
        else:
            revision_during_publish = None
        name = hashlib.sha256(receipt_id.encode("utf-8")).hexdigest() + ".json"
        path = self.root / name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short receipt write")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, read_flags)
        try:
            chunks = []
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(fd)
        readback = b"".join(chunks)

        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(self.root, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

        observed_sha256 = hashlib.sha256(readback).hexdigest()
        assert readback == payload
        assert observed_sha256 == expected_sha256
        self.calls.append((receipt_id, expected_sha256, revision_during_publish))
        return hc.ReceiptPublication(
            receipt_id=receipt_id,
            sha256=observed_sha256,
            durable=True,
            readback_verified=True,
        )


class _FailingPublisher:
    def publish_exclusive(self, **_kwargs):
        raise OSError("simulated durability failure")


def _registry(*, with_softmax=False):
    layers = [
        hc.LayerSpec(
            layer_index=0,
            kind=hc.LayerKind.GDN_FULL_HISTORY,
            operator_class="GatedDeltaNet",
            main_shortconv_enabled=True,
        )
    ]
    if with_softmax:
        layers.append(
            hc.LayerSpec(
                layer_index=1,
                kind=hc.LayerKind.SOFTMAX_PREVIOUS_CHUNK,
                operator_class="MultiHeadCrossAttention",
                tokens_per_latent=2,
            )
        )
    return hc.LayerRegistry(layers=tuple(layers))


def _manager(tmp_path, *, registry=None, publisher=None, layout=L5_LAYOUT):
    publisher = publisher or _FsyncReceiptPublisher(tmp_path / "receipts")
    manager = hc.HybridCacheManager.for_synthetic_tests(
        registry=registry or _registry(),
        episode_id="episode-0",
        layout=layout,
        receipt_publisher=publisher,
    )
    if isinstance(publisher, _FsyncReceiptPublisher):
        publisher.state_revision = lambda: manager.state.revision
    return manager, publisher


def _request(
    *,
    chunk_id=0,
    revision=None,
    nonce=None,
    commit_id=None,
    layout=L5_LAYOUT,
    episode_id="episode-0",
    episode_epoch=0,
):
    if revision is None:
        revision = chunk_id
    chunk = layout.chunks[chunk_id]
    content_time = hc.ContentTime.from_layout(
        episode_id=episode_id,
        episode_epoch=episode_epoch,
        layout=layout,
        chunk=chunk,
    )
    latent_slot_count = len(chunk.latent_valid_mask)
    video = torch.arange(
        1 * 3 * latent_slot_count * 2 * 2,
        dtype=torch.float32,
    ).reshape(1, 3, latent_slot_count, 2, 2)
    video = video + chunk_id * 100
    frame_mask = torch.tensor(
        chunk.latent_valid_mask,
        dtype=torch.bool,
    ).view(1, -1)
    video = video.masked_fill(
        (~frame_mask)[:, None, :, None, None],
        0,
    )
    action_slot_count = chunk.action_slot_capacity
    actions = torch.arange(
        action_slot_count * 20,
        dtype=torch.float32,
    ).reshape(1, action_slot_count, 20)
    actions = actions + chunk_id * 100
    mask = torch.tensor(
        chunk.action_valid_mask,
        dtype=torch.bool,
    ).view(1, -1)
    actions = actions.masked_fill((~mask)[:, :, None], 0)
    proof = hc.TeacherForcingDatasetPairProof(
        dataset_manifest_sha256=DATASET_SHA,
        dataset_episode_id=episode_id,
        dataset_row_identity=f"{episode_id}:chunk-{chunk_id}",
        dataset_row_start=0,
        dataset_row_end_exclusive=layout.valid_raw_count,
        layout_spec_sha256=layout.layout_spec_sha256,
        layout_instance_digest=layout.layout_instance_digest,
        video_tensor_digest=hc.tensor_digest(video),
        frame_valid_mask_digest=hc.tensor_digest(frame_mask),
        action_tensor_digest=hc.tensor_digest(actions),
        action_mask_digest=hc.tensor_digest(mask),
        row_order_manifest_sha256=ROW_ORDER_SHA,
    )
    return hc.TeacherForcingCommitRequest(
        commit_id=commit_id or f"commit-{chunk_id}",
        transaction_nonce=nonce or f"nonce-{chunk_id}",
        expected_episode_id=episode_id,
        expected_episode_epoch=episode_epoch,
        expected_revision=revision,
        content_time=content_time,
        video=video,
        frame_valid_mask=frame_mask,
        actions=actions,
        action_valid_mask=mask,
        proof=proof,
    )


def _rebind_pair_tensors(
    request,
    *,
    video=None,
    frame_valid_mask=None,
    actions=None,
    action_valid_mask=None,
):
    video = request.video if video is None else video
    frame_valid_mask = (
        request.frame_valid_mask
        if frame_valid_mask is None
        else frame_valid_mask
    )
    actions = request.actions if actions is None else actions
    action_valid_mask = (
        request.action_valid_mask
        if action_valid_mask is None
        else action_valid_mask
    )
    proof = replace(
        request.proof,
        video_tensor_digest=hc.tensor_digest(video),
        frame_valid_mask_digest=hc.tensor_digest(frame_valid_mask),
        action_tensor_digest=hc.tensor_digest(actions),
        action_mask_digest=hc.tensor_digest(action_valid_mask),
    )
    return replace(
        request,
        video=video,
        frame_valid_mask=frame_valid_mask,
        actions=actions,
        action_valid_mask=action_valid_mask,
        proof=proof,
    )


def _payloads(context, registry, *, softmax_token_delta=0):
    value = float(context.content_time.chunk_id + 1)
    payloads = []
    for spec in registry.layers:
        if spec.kind is hc.LayerKind.GDN_FULL_HISTORY:
            tensors = {
                "main_s_kv": torch.full((1, 2, 4, 4), value),
                "main_s_z": torch.full((1, 2, 4, 1), value),
                "main_shortconv_left_context": torch.full((1, 2, 3), value),
                "ffn_tconv_left_context": torch.full((1, 2, 2, 3), value),
            }
        else:
            latent_count = (
                context.content_time.latent_end_exclusive
                - context.content_time.latent_start
            )
            token_count = latent_count * spec.tokens_per_latent + softmax_token_delta
            tensors = {
                "main_k_post_rope": torch.full((1, 2, token_count, 4), value),
                "main_v": torch.full((1, 2, token_count, 4), value),
                "ffn_tconv_left_context": torch.full((1, 2, 2, 3), value),
            }
        payloads.append(
            hc.StagedLayerPayload(
                layer_index=spec.layer_index,
                kind=spec.kind,
                tensors=tensors,
            )
        )
    return tuple(payloads)


class _Stager:
    def __init__(self, registry):
        self.registry = registry
        self.calls = []
        self.last_payloads = None

    def __call__(self, context):
        assert context.video_timestep == context.action_timestep == 0
        assert context.commit_source is hc.CommitSource.TEACHER_FORCING
        self.calls.append(context)
        self.last_payloads = _payloads(context, self.registry)
        return self.last_payloads


def _commit(manager, request, callback):
    return manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=callback,
    )


def test_teacher_pair_commits_once_after_durable_receipt_and_retry_is_idempotent(
    tmp_path,
) -> None:
    manager, publisher = _manager(tmp_path)
    stager = _Stager(manager.registry)
    request = _request()
    before = manager.state.state_manifest_digest

    receipt = _commit(manager, request, stager)

    assert len(stager.calls) == 1
    assert publisher.calls == [
        ("commit-0", receipt.receipt_sha256, 0),
    ]
    assert receipt.state_manifest_before == before
    assert receipt.state_manifest_after == manager.state.state_manifest_digest
    assert receipt.dataset_manifest_sha256 == DATASET_SHA
    assert receipt.dataset_row_identity == "episode-0:chunk-0"
    assert receipt.pair_frame_valid_mask_digest == hc.tensor_digest(
        request.frame_valid_mask
    )
    assert torch.equal(
        stager.calls[0].frame_valid_mask,
        request.frame_valid_mask,
    )
    assert (
        stager.calls[0].frame_valid_mask_digest
        == receipt.pair_frame_valid_mask_digest
    )
    assert manager.state.revision == manager.state.next_chunk_id == 1
    assert manager.state.action_cursor == 16

    source_tensor = stager.last_payloads[0].tensors["main_s_kv"]
    committed_digest = manager.state.state_manifest_digest
    source_tensor.add_(1000)
    assert manager.state.state_manifest_digest == committed_digest

    assert _commit(manager, request, stager) is receipt
    assert len(stager.calls) == 1
    assert len(publisher.calls) == 1


def test_manager_rejects_caller_reported_geometry_not_derived_from_bound_layout(
    tmp_path,
) -> None:
    manager, _ = _manager(tmp_path)
    request = _request()
    forged = replace(
        request,
        content_time=replace(
            request.content_time,
            raw_observation_end_exclusive=16,
        ),
    )
    calls = []

    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(
            manager,
            forged,
            lambda context: calls.append(context),
        )

    assert exc_info.value.code == "CACHE_CONTENT_TIME_MISMATCH"
    assert calls == []
    assert manager.state.revision == 0


def test_manager_modes_reject_opposite_layout_provenance(tmp_path) -> None:
    publisher = _FsyncReceiptPublisher(tmp_path / "normal")
    with pytest.raises(hc.CacheContractError) as normal_exc:
        hc.HybridCacheManager(
            registry=_registry(),
            episode_id="episode-0",
            layout=L5_LAYOUT,
            receipt_publisher=publisher,
        )
    assert normal_exc.value.code == "CACHE_SYNTHETIC_LAYOUT_PROVENANCE"

    non_synthetic_layout = replace(
        L5_LAYOUT,
        synthetic_test_only=False,
    )
    with pytest.raises(hc.CacheContractError) as synthetic_exc:
        hc.HybridCacheManager.for_synthetic_tests(
            registry=_registry(),
            episode_id="episode-0",
            layout=non_synthetic_layout,
            receipt_publisher=_FsyncReceiptPublisher(
                tmp_path / "synthetic"
            ),
        )
    assert synthetic_exc.value.code == "CACHE_SYNTHETIC_LAYOUT_PROVENANCE"


@pytest.mark.parametrize("valid_latent_count", [5, 8])
def test_manager_commits_exact_l5_l8_synthetic_layout_sequence(
    tmp_path, valid_latent_count
) -> None:
    layout = _layout_for_latent_count(valid_latent_count)
    manager, _ = _manager(
        tmp_path / f"l{valid_latent_count}",
        layout=layout,
    )
    stager = _Stager(manager.registry)

    for chunk in layout.chunks:
        receipt = _commit(
            manager,
            _request(chunk_id=chunk.chunk_id, layout=layout),
            stager,
        )
        expected = hc.ContentTime.from_layout(
            episode_id="episode-0",
            episode_epoch=0,
            layout=layout,
            chunk=chunk,
        )
        assert stager.calls[-1].content_time == expected
        assert stager.calls[-1].video.shape[2] == len(
            chunk.latent_valid_mask
        )
        assert torch.equal(
            stager.calls[-1].frame_valid_mask,
            torch.tensor(
                chunk.latent_valid_mask,
                dtype=torch.bool,
            ).view(1, -1),
        )
        assert stager.calls[-1].actions.shape[1] == (
            chunk.action_slot_capacity
        )
        assert torch.equal(
            stager.calls[-1].action_valid_mask,
            torch.tensor(
                chunk.action_valid_mask,
                dtype=torch.bool,
            ).view(1, -1),
        )
        assert receipt.layout_instance_digest == layout.layout_instance_digest
        assert manager.state.action_cursor == chunk.action_end

    tail = layout.chunks[-1]
    assert tail.valid_latent_count < len(tail.latent_valid_mask)
    assert tail.valid_action_count < tail.action_slot_capacity
    tail_context = stager.calls[-1]
    assert not bool(
        tail_context.video.masked_select(
            (~tail_context.frame_valid_mask)[
                :, None, :, None, None
            ].expand_as(tail_context.video)
        ).any()
    )
    assert not bool(
        tail_context.actions.masked_select(
            (~tail_context.action_valid_mask)[:, :, None].expand_as(
                tail_context.actions
            )
        ).any()
    )
    through = manager.state.layer_states[0].through
    assert through is not None
    assert through.chunk_id == len(layout.chunks) - 1
    assert through.latent_end_exclusive == layout.valid_latent_count
    assert (
        through.raw_observation_end_exclusive
        == layout.valid_raw_count
    )
    assert through.action_rope_end_exclusive == layout.valid_action_count


@pytest.mark.parametrize("valid_latent_count", [5, 8])
@pytest.mark.parametrize(
    "malformation",
    [
        "valid_only_video",
        "valid_only_actions",
        "wrong_frame_mask",
        "wrong_action_mask",
        "nonzero_video_padding",
        "nonzero_action_padding",
    ],
)
def test_l5_l8_partial_tail_rejects_noncanonical_fixed_slot_pair(
    tmp_path,
    valid_latent_count,
    malformation,
) -> None:
    layout = _layout_for_latent_count(valid_latent_count)
    chunk = layout.chunks[-1]
    request = _request(chunk_id=chunk.chunk_id, layout=layout)

    if malformation == "valid_only_video":
        request = _rebind_pair_tensors(
            request,
            video=request.video[:, :, : chunk.valid_latent_count],
            frame_valid_mask=request.frame_valid_mask[
                :, : chunk.valid_latent_count
            ],
        )
    elif malformation == "valid_only_actions":
        request = _rebind_pair_tensors(
            request,
            actions=request.actions[:, : chunk.valid_action_count],
            action_valid_mask=request.action_valid_mask[
                :, : chunk.valid_action_count
            ],
        )
    elif malformation == "wrong_frame_mask":
        request = _rebind_pair_tensors(
            request,
            frame_valid_mask=torch.ones_like(request.frame_valid_mask),
        )
    elif malformation == "wrong_action_mask":
        request = _rebind_pair_tensors(
            request,
            action_valid_mask=torch.ones_like(
                request.action_valid_mask
            ),
        )
    elif malformation == "nonzero_video_padding":
        video = request.video.clone()
        first_padding_slot = chunk.valid_latent_count
        video[:, :, first_padding_slot] = 1
        request = _rebind_pair_tensors(request, video=video)
    else:
        actions = request.actions.clone()
        first_padding_slot = chunk.valid_action_count
        actions[:, first_padding_slot] = 1
        request = _rebind_pair_tensors(request, actions=actions)

    manager, _ = _manager(
        tmp_path / f"l{valid_latent_count}-{malformation}",
        layout=layout,
    )
    prefix_stager = _Stager(manager.registry)
    for prior_chunk in layout.chunks[: chunk.chunk_id]:
        _commit(
            manager,
            _request(chunk_id=prior_chunk.chunk_id, layout=layout),
            prefix_stager,
        )
    calls = []
    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(
            manager,
            request,
            lambda context: calls.append(context),
        )
    assert exc_info.value.code == "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH"
    assert calls == []
    assert manager.state.revision == chunk.chunk_id


def test_duplicate_conflict_stale_and_out_of_order_fail_before_staging(
    tmp_path,
) -> None:
    manager, _ = _manager(tmp_path)
    stager = _Stager(manager.registry)
    request = _request()
    _commit(manager, request, stager)

    conflict = _request(commit_id="commit-0", nonce="different-nonce")
    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(manager, conflict, stager)
    assert exc_info.value.code == "COMMIT_DUPLICATE_CONFLICT"

    stale = _request(chunk_id=1, revision=0)
    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(manager, stale, stager)
    assert exc_info.value.code == "COMMIT_STALE_REVISION"
    assert len(stager.calls) == 1

    fresh, _ = _manager(tmp_path / "fresh")
    fresh_stager = _Stager(fresh.registry)
    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(fresh, _request(chunk_id=1, revision=0), fresh_stager)
    assert exc_info.value.code == "COMMIT_OUT_OF_ORDER"
    assert fresh_stager.calls == []


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            hc.CommitSource.SELF_FORCING,
            "COMMIT_SELF_FORCING_OBJECTIVE_UNCLOSED",
        ),
        (hc.CommitSource.DEPLOY_APPLIED_ACK, "COMMIT_DEPLOY_ACK_MISSING"),
        ("not-a-source", "COMMIT_SOURCE_UNKNOWN"),
    ],
)
def test_uncommissioned_commit_sources_hard_fail_without_callback(
    tmp_path, source, code
) -> None:
    manager, _ = _manager(tmp_path)
    calls = []
    with pytest.raises(hc.CacheContractError) as exc_info:
        manager.commit_paired(
            commit_source=source,
            request=None,
            stage_callback=lambda context: calls.append(context),
        )
    assert exc_info.value.code == code
    assert calls == []
    assert manager.state.revision == 0


def test_staging_and_receipt_failures_leave_old_pointer_unchanged(tmp_path) -> None:
    manager, _ = _manager(tmp_path / "staging")
    before_digest = manager.state.state_manifest_digest

    def fail_stage(_context):
        raise ValueError("simulated model failure")

    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(manager, _request(), fail_stage)
    assert exc_info.value.code == "COMMIT_STAGING_FAILED"
    assert manager.state.state_manifest_digest == before_digest

    failed_manager, _ = _manager(
        tmp_path / "publication", publisher=_FailingPublisher()
    )
    failed_before = failed_manager._state
    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(
            failed_manager,
            _request(),
            _Stager(failed_manager.registry),
        )
    assert exc_info.value.code == "COMMIT_RECEIPT_PUBLICATION_FAILED"
    assert failed_manager._state is failed_before
    assert failed_manager._state.revision == 0
    with pytest.raises(hc.CacheContractError) as poisoned:
        _ = failed_manager.state
    assert poisoned.value.code == "CACHE_PUBLICATION_POISONED"


def test_softmax_state_is_exactly_one_previous_chunk(tmp_path) -> None:
    registry = _registry(with_softmax=True)
    manager, _ = _manager(tmp_path, registry=registry)
    before_digest = manager.state.state_manifest_digest

    def oversized(context):
        return _payloads(context, registry, softmax_token_delta=1)

    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(manager, _request(), oversized)
    assert exc_info.value.code == "CACHE_SOFTMAX_HISTORY_GT_ONE"
    assert manager.state.state_manifest_digest == before_digest


@pytest.mark.parametrize("input_name", ["frame_valid_mask", "actions"])
def test_staging_callback_cannot_mutate_bound_pair_inputs(
    tmp_path,
    input_name,
) -> None:
    manager, _ = _manager(tmp_path)

    def mutate(context):
        if input_name == "frame_valid_mask":
            context.frame_valid_mask.logical_not_()
        else:
            context.actions.add_(1)
        return _payloads(context, manager.registry)

    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(manager, _request(), mutate)
    assert exc_info.value.code == "COMMIT_STAGING_INPUT_MUTATION"
    assert manager.state.revision == 0


def test_read_view_detects_tensor_restore_binding_change_and_stale_reset(
    tmp_path,
) -> None:
    manager, _ = _manager(tmp_path)
    _commit(manager, _request(), _Stager(manager.registry))

    view = manager.snapshot_for_denoise(
        expected_episode_id="episode-0",
        expected_episode_epoch=0,
        expected_revision=1,
        expected_next_chunk_id=1,
    )
    assert not hasattr(view, "_state")
    clean = view.export_scratch()
    manager.finish_denoise(read_view=view, scratch=clean)

    changed_then_restored = view.export_scratch()
    tensor = changed_then_restored.layers[0].tensors["main_s_kv"]
    tensor.add_(1)
    tensor.sub_(1)
    with pytest.raises(hc.CacheContractError) as exc_info:
        manager.finish_denoise(
            read_view=view,
            scratch=changed_then_restored,
        )
    assert exc_info.value.code == "CACHE_DENOISE_MUTATION"

    rebound = view.export_scratch()
    original = rebound.layers[0].tensors.pop("main_s_kv")
    rebound.layers[0].tensors["main_s_kv"] = original
    with pytest.raises(hc.CacheContractError) as exc_info:
        manager.finish_denoise(read_view=view, scratch=rebound)
    assert exc_info.value.code == "CACHE_DENOISE_MUTATION"

    reset_receipt = manager.reset(
        reset_id="reset-0",
        transaction_nonce="reset-nonce-0",
        new_episode_id="episode-1",
        layout=L5_LAYOUT_EPISODE_1,
    )
    assert reset_receipt.new_episode_epoch == 1
    assert manager.state.revision == manager.state.action_cursor == 0
    assert all(layer.is_empty for layer in manager.state.layer_states)
    with pytest.raises(hc.CacheContractError) as exc_info:
        manager.finish_denoise(read_view=view, scratch=clean)
    assert exc_info.value.code == "COMMIT_STALE_REVISION"


def test_reset_during_staging_aborts_commit_at_cas(tmp_path) -> None:
    manager, publisher = _manager(tmp_path)
    request = _request()

    def reset_then_stage(context):
        manager.reset(
            reset_id="reset-during-stage",
            transaction_nonce="reset-nonce",
            new_episode_id="episode-1",
            layout=L5_LAYOUT_EPISODE_1,
        )
        return _payloads(context, manager.registry)

    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(manager, request, reset_then_stage)
    assert exc_info.value.code == "COMMIT_STALE_REVISION"
    assert [call[0] for call in publisher.calls] == ["reset-during-stage"]
    assert manager.state.episode_id == "episode-1"
    assert manager.state.episode_epoch == 1
    assert manager.state.revision == 0


def test_reset_invalidates_completed_commit_idempotency_ledger(tmp_path) -> None:
    manager, _ = _manager(tmp_path)
    request = _request()
    _commit(manager, request, _Stager(manager.registry))
    manager.reset(
        reset_id="reset-after-commit",
        transaction_nonce="reset-nonce",
        new_episode_id="episode-1",
        layout=L5_LAYOUT_EPISODE_1,
    )

    with pytest.raises(hc.CacheContractError) as exc_info:
        _commit(manager, request, _Stager(manager.registry))
    assert exc_info.value.code == "COMMIT_STALE_REVISION"
