"""CPU/synthetic-only contract tests for the Stage-2B L1 snapshot codec."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest
import torch

from sana_wam.cach.stage2b_state_snapshot import (
    EncodedStage2BStateSnapshot,
    Stage2BStateSnapshotError,
    Stage2BTensorBlob,
    decode_stage2b_state_snapshot,
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


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _manifest(snapshot: EncodedStage2BStateSnapshot) -> dict:
    return json.loads(snapshot.manifest_bytes)


def _with_manifest(
    snapshot: EncodedStage2BStateSnapshot,
    manifest: dict,
) -> EncodedStage2BStateSnapshot:
    return replace(snapshot, manifest_bytes=_canonical(manifest))


def _layout(*, episode_label: str = "snapshot-episode"):
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


def _registry(*, operator_class: str = "synthetic.SnapshotGDN"):
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


class _Publisher:
    def publish_exclusive(self, *, receipt_id, payload, expected_sha256):
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        return hc.ReceiptPublication(
            receipt_id=receipt_id,
            sha256=expected_sha256,
            durable=True,
            readback_verified=True,
        )


def _fixed_actions(chunk, *, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.arange(
        chunk.action_slot_capacity * 20,
        dtype=torch.float32,
    ).reshape(1, chunk.action_slot_capacity, 20)
    actions = (actions + offset).to(dtype=torch.bfloat16)
    mask = torch.tensor(chunk.action_valid_mask, dtype=torch.bool).view(1, -1)
    return actions.masked_fill((~mask)[:, :, None], 0), mask


def _request(layout, chunk_id: int):
    chunk = layout.chunks[chunk_id]
    frame_mask = torch.tensor(chunk.latent_valid_mask, dtype=torch.bool).view(1, -1)
    video = torch.arange(
        3 * len(chunk.latent_valid_mask) * 4,
        dtype=torch.float32,
    ).reshape(1, 3, len(chunk.latent_valid_mask), 2, 2)
    video = (video + chunk_id * 1000).to(dtype=torch.bfloat16).masked_fill(
        (~frame_mask)[:, None, :, None, None],
        0,
    )
    actions, action_mask = _fixed_actions(chunk, offset=chunk_id * 1000)
    proof = hc.TeacherForcingDatasetPairProof(
        dataset_manifest_sha256="b" * 64,
        dataset_episode_id="snapshot-episode",
        dataset_row_identity=f"snapshot-episode:chunk-{chunk_id}",
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
        commit_id=f"snapshot-commit-{chunk_id}",
        transaction_nonce=f"snapshot-nonce-{chunk_id}",
        expected_episode_id="snapshot-episode",
        expected_episode_epoch=0,
        expected_revision=chunk_id,
        content_time=hc.ContentTime.from_layout(
            episode_id="snapshot-episode",
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


def _payloads(context, *, unsupported_float64: bool = False):
    value = context.content_time.chunk_id + 1
    main_dtype = torch.float64 if unsupported_float64 else torch.bfloat16
    return (
        hc.StagedLayerPayload(
            layer_index=0,
            kind=hc.LayerKind.GDN_FULL_HISTORY,
            tensors={
                "ffn_tconv_left_context": torch.full(
                    (1, 2, 2, 3),
                    float(value),
                    dtype=torch.float32,
                ),
                "main_s_kv": torch.full(
                    (1, 2, 4, 4),
                    float(value),
                    dtype=main_dtype,
                ),
                "main_s_z": torch.full(
                    (1, 2, 4, 1),
                    float(value),
                    dtype=torch.float32,
                ),
                "main_shortconv_left_context": torch.full(
                    (1, 2, 3),
                    value % 2 == 1,
                    dtype=torch.bool,
                ),
            },
        ),
    )


def _state_case(
    *,
    variant: CACHStagingVariant = CACHStagingVariant.CACH_A,
    commits: int = 2,
    unsupported_float64: bool = False,
):
    layout = _layout()
    registry = _registry()
    manager = h2.Stage2BHybridCacheManager.for_synthetic_tests(
        registry=registry,
        episode_id="snapshot-episode",
        layout=layout,
        receipt_publisher=_Publisher(),
        staging_variant=variant,
    )
    for chunk_id in range(commits):
        manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=_request(layout, chunk_id),
            stage_callback=lambda context: _payloads(
                context,
                unsupported_float64=unsupported_float64,
            ),
        )
    return manager.state, layout, registry


def _decode(
    snapshot: EncodedStage2BStateSnapshot,
    *,
    layout,
    registry,
    variant: CACHStagingVariant,
):
    return decode_stage2b_state_snapshot(
        snapshot,
        expected_manifest_sha256=snapshot.manifest_sha256,
        expected_layout=layout,
        expected_registry=registry,
        expected_staging_variant=variant,
        device_map={"cpu": "cpu"},
    )


def _assert_exact_state(left, right) -> None:
    assert right.state_manifest_digest == left.state_manifest_digest
    assert right.to_manifest() == left.to_manifest()
    assert right.cache_state.to_manifest() == left.cache_state.to_manifest()
    assert right.cache_state is not left.cache_state
    for left_layer, right_layer in zip(
        left.layer_states,
        right.layer_states,
        strict=True,
    ):
        assert right_layer.through == left_layer.through
        for (left_name, left_tensor), (right_name, right_tensor) in zip(
            left_layer.tensors,
            right_layer.tensors,
            strict=True,
        ):
            assert right_name == left_name
            assert right_tensor.logical_tensor_id == left_tensor.logical_tensor_id
            assert torch.equal(right_tensor.clone_tensor(), left_tensor.clone_tensor())
    left_history = left.committed_action_history
    right_history = right.committed_action_history
    assert (left_history is None) == (right_history is None)
    if left_history is not None and right_history is not None:
        assert right_history.manifest_digest == left_history.manifest_digest
        for left_span, right_span in zip(
            left_history.spans,
            right_history.spans,
            strict=True,
        ):
            assert right_span.to_manifest() == left_span.to_manifest()
            assert torch.equal(right_span.clone_actions(), left_span.clone_actions())


@pytest.mark.parametrize(
    "variant",
    [CACHStagingVariant.CACH_A, CACHStagingVariant.REF_GDN_CORRECTED],
)
@pytest.mark.parametrize("commits", [0, 1, 2])
def test_exact_empty_one_chunk_continuation_partial_tail_both_variant_round_trip(
    variant: CACHStagingVariant,
    commits: int,
) -> None:
    state, layout, registry = _state_case(variant=variant, commits=commits)
    assert len(layout.chunks) == 2
    assert layout.chunks[-1].is_partial_tail is True

    snapshot = encode_stage2b_state_snapshot(
        state=state,
        expected_layout=layout,
        expected_registry=registry,
    )
    decoded = _decode(
        snapshot,
        layout=layout,
        registry=registry,
        variant=variant,
    )

    _assert_exact_state(state, decoded)
    if commits == 0:
        assert snapshot.blobs == ()
    else:
        dtype_inventory = {
            tensor["dtype"]
            for layer in _manifest(snapshot)["cache_state"]["layer_states"]
            for tensor in layer["tensors"]
        }
        dtype_inventory.add(
            _manifest(snapshot)["committed_action_history"]["spans"][0][
                "actions"
            ]["dtype"]
        )
        assert dtype_inventory == {"bfloat16", "bool", "float32"}


def test_manifest_is_strict_canonical_and_bound_by_expected_sha(tmp_path) -> None:
    del tmp_path  # This codec creates no filesystem state.
    state, layout, registry = _state_case(commits=1)
    snapshot = encode_stage2b_state_snapshot(
        state=state,
        expected_layout=layout,
        expected_registry=registry,
    )

    noncanonical = replace(snapshot, manifest_bytes=snapshot.manifest_bytes + b"\n")
    with pytest.raises(Stage2BStateSnapshotError):
        decode_stage2b_state_snapshot(
            noncanonical,
            expected_manifest_sha256=noncanonical.manifest_sha256,
            expected_layout=layout,
            expected_registry=registry,
            expected_staging_variant=CACHStagingVariant.CACH_A,
            device_map={"cpu": "cpu"},
        )

    with pytest.raises(Stage2BStateSnapshotError) as caught:
        decode_stage2b_state_snapshot(
            snapshot,
            expected_manifest_sha256="0" * 64,
            expected_layout=layout,
            expected_registry=registry,
            expected_staging_variant=CACHStagingVariant.CACH_A,
            device_map={"cpu": "cpu"},
        )
    assert caught.value.code == "SNAPSHOT_MANIFEST_MISMATCH"


def test_missing_replaced_extra_duplicate_and_trailing_blobs_fail_closed() -> None:
    state, layout, registry = _state_case(commits=1)
    snapshot = encode_stage2b_state_snapshot(
        state=state,
        expected_layout=layout,
        expected_registry=registry,
    )
    assert snapshot.blobs

    missing = replace(snapshot, blobs=snapshot.blobs[1:])
    with pytest.raises(Stage2BStateSnapshotError):
        _decode(
            missing,
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )

    first = snapshot.blobs[0]
    replaced_blob = replace(
        first,
        payload=bytes([first.payload[0] ^ 1]) + first.payload[1:],
    )
    substituted = replace(snapshot, blobs=(replaced_blob,) + snapshot.blobs[1:])
    with pytest.raises(Stage2BStateSnapshotError) as replaced_error:
        _decode(
            substituted,
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )
    assert replaced_error.value.code == "SNAPSHOT_BLOB_MISMATCH"

    extra_payload = b"unreferenced-extra"
    extra_blob = Stage2BTensorBlob(
        sha256=hashlib.sha256(extra_payload).hexdigest(),
        payload=extra_payload,
    )
    extra = replace(
        snapshot,
        blobs=tuple(sorted(snapshot.blobs + (extra_blob,), key=lambda blob: blob.sha256)),
    )
    with pytest.raises(Stage2BStateSnapshotError, match="unreferenced extra"):
        _decode(
            extra,
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )

    duplicate = replace(
        snapshot,
        blobs=tuple(
            sorted(snapshot.blobs + (snapshot.blobs[0],), key=lambda blob: blob.sha256)
        ),
    )
    with pytest.raises(Stage2BStateSnapshotError, match="duplicate"):
        _decode(
            duplicate,
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )

    trailing_manifest = _manifest(snapshot)
    descriptor = trailing_manifest["committed_action_history"]["spans"][0]["actions"]
    original_sha = descriptor["raw_blob_sha256"]
    original_blob = next(blob for blob in snapshot.blobs if blob.sha256 == original_sha)
    trailing_payload = original_blob.payload + b"\x00"
    trailing_sha = hashlib.sha256(trailing_payload).hexdigest()
    descriptor["raw_blob_sha256"] = trailing_sha
    trailing_blobs = tuple(
        sorted(
            [
                Stage2BTensorBlob(trailing_sha, trailing_payload)
                if blob.sha256 == original_sha
                else blob
                for blob in snapshot.blobs
            ],
            key=lambda blob: blob.sha256,
        )
    )
    trailing = replace(
        _with_manifest(snapshot, trailing_manifest),
        blobs=trailing_blobs,
    )
    with pytest.raises(Stage2BStateSnapshotError, match="trailing raw bytes"):
        _decode(
            trailing,
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )


@pytest.mark.parametrize(
    "mutation",
    ["dtype", "shape", "digest", "logical_id", "value"],
)
def test_tensor_metadata_and_value_mutation_fail_closed(mutation: str) -> None:
    state, layout, registry = _state_case(commits=1)
    snapshot = encode_stage2b_state_snapshot(
        state=state,
        expected_layout=layout,
        expected_registry=registry,
    )
    manifest = _manifest(snapshot)
    descriptor = manifest["cache_state"]["layer_states"][0]["tensors"][0]
    blobs = snapshot.blobs
    if mutation == "dtype":
        descriptor["dtype"] = "float64"
    elif mutation == "shape":
        descriptor["shape"][0] += 1
    elif mutation == "digest":
        descriptor["raw_tensor_digest"] = "0" * 64
    elif mutation == "logical_id":
        descriptor["logical_tensor_id"] = "arbitrary/or/duplicate/id"
    else:
        original_sha = descriptor["raw_blob_sha256"]
        original_blob = next(blob for blob in blobs if blob.sha256 == original_sha)
        changed = bytes([original_blob.payload[0] ^ 1]) + original_blob.payload[1:]
        changed_sha = hashlib.sha256(changed).hexdigest()
        descriptor["raw_blob_sha256"] = changed_sha
        blobs = tuple(
            sorted(
                [
                    Stage2BTensorBlob(changed_sha, changed)
                    if blob.sha256 == original_sha
                    else blob
                    for blob in blobs
                ],
                key=lambda blob: blob.sha256,
            )
        )
    corrupted = replace(_with_manifest(snapshot, manifest), blobs=blobs)

    with pytest.raises(Stage2BStateSnapshotError):
        _decode(
            corrupted,
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )


def test_expected_layout_registry_variant_and_history_are_strictly_bound() -> None:
    state, layout, registry = _state_case(commits=1)
    snapshot = encode_stage2b_state_snapshot(
        state=state,
        expected_layout=layout,
        expected_registry=registry,
    )

    with pytest.raises(Stage2BStateSnapshotError) as layout_error:
        _decode(
            snapshot,
            layout=_layout(episode_label="other-layout"),
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )
    assert layout_error.value.code == "SNAPSHOT_IDENTITY_MISMATCH"

    layout_manifest = _manifest(snapshot)
    layout_manifest["layout_identity"]["layout_manifest"]["source_row_digest"] = (
        "0" * 64
    )
    with pytest.raises(Stage2BStateSnapshotError) as layout_tamper:
        _decode(
            _with_manifest(snapshot, layout_manifest),
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )
    assert layout_tamper.value.code == "SNAPSHOT_IDENTITY_MISMATCH"

    with pytest.raises(Stage2BStateSnapshotError) as registry_error:
        _decode(
            snapshot,
            layout=layout,
            registry=_registry(operator_class="synthetic.OtherGDN"),
            variant=CACHStagingVariant.CACH_A,
        )
    assert registry_error.value.code == "SNAPSHOT_IDENTITY_MISMATCH"

    registry_manifest = _manifest(snapshot)
    registry_manifest["registry_identity"]["registry_manifest"]["layers"][0][
        "operator_class"
    ] = "synthetic.TamperedGDN"
    with pytest.raises(Stage2BStateSnapshotError) as registry_tamper:
        _decode(
            _with_manifest(snapshot, registry_manifest),
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )
    assert registry_tamper.value.code == "SNAPSHOT_IDENTITY_MISMATCH"

    with pytest.raises(Stage2BStateSnapshotError) as variant_error:
        _decode(
            snapshot,
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.REF_GDN_CORRECTED,
        )
    assert variant_error.value.code == "SNAPSHOT_IDENTITY_MISMATCH"

    variant_manifest = _manifest(snapshot)
    variant_manifest["staging_variant"] = "ref_gdn_corrected"
    variant_manifest["history_summary_operator"] = (
        "gdn_recurrent_video_only_paired_commit_v1"
    )
    variant_mutated = _with_manifest(snapshot, variant_manifest)
    with pytest.raises(Stage2BStateSnapshotError) as variant_tamper:
        _decode(
            variant_mutated,
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )
    assert variant_tamper.value.code == "SNAPSHOT_IDENTITY_MISMATCH"

    history_manifest = _manifest(snapshot)
    history_manifest["committed_action_history"]["spans"][0]["commit_id"] = (
        "mutated-history-id"
    )
    history_mutated = _with_manifest(snapshot, history_manifest)
    with pytest.raises(Stage2BStateSnapshotError) as history_error:
        _decode(
            history_mutated,
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )
    assert history_error.value.code in {
        "SNAPSHOT_HISTORY_MISMATCH",
        "SNAPSHOT_STATE_MISMATCH",
    }

    history_id_manifest = _manifest(snapshot)
    history_id_manifest["committed_action_history"]["spans"][0]["actions"][
        "logical_tensor_id"
    ] = "arbitrary/history/id"
    history_id_mutated = _with_manifest(snapshot, history_id_manifest)
    with pytest.raises(Stage2BStateSnapshotError) as history_id_error:
        _decode(
            history_id_mutated,
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )
    assert history_id_error.value.code == "SNAPSHOT_HISTORY_MISMATCH"


def test_codec_rejects_unadmitted_dtype_and_non_cpu_device_map() -> None:
    state, layout, registry = _state_case(commits=1, unsupported_float64=True)
    with pytest.raises(Stage2BStateSnapshotError) as dtype_error:
        encode_stage2b_state_snapshot(
            state=state,
            expected_layout=layout,
            expected_registry=registry,
        )
    assert dtype_error.value.code == "SNAPSHOT_DTYPE_UNSUPPORTED"

    state, layout, registry = _state_case(commits=0)
    snapshot = encode_stage2b_state_snapshot(
        state=state,
        expected_layout=layout,
        expected_registry=registry,
    )
    with pytest.raises(Stage2BStateSnapshotError) as device_error:
        decode_stage2b_state_snapshot(
            snapshot,
            expected_manifest_sha256=snapshot.manifest_sha256,
            expected_layout=layout,
            expected_registry=registry,
            expected_staging_variant=CACHStagingVariant.CACH_A,
            device_map={"cpu": "cuda"},
        )
    assert device_error.value.code == "SNAPSHOT_CPU_SYNTHETIC_ONLY"


def test_encode_rejects_source_state_with_non_manager_logical_tensor_id() -> None:
    state, layout, registry = _state_case(commits=1)
    layer = state.cache_state.layer_states[0]
    name, tensor_snapshot = layer.tensors[0]
    forged_tensor = hc.TensorSnapshot.from_tensor(
        "arbitrary/source-state/id",
        tensor_snapshot.clone_tensor(),
    )
    forged_layer = replace(
        layer,
        tensors=((name, forged_tensor),) + layer.tensors[1:],
    )
    forged_cache = replace(state.cache_state, layer_states=(forged_layer,))
    forged_state = h2.Stage2BTemporalState(
        cache_state=forged_cache,
        staging_variant=state.staging_variant,
        committed_action_history=state.committed_action_history,
    )

    with pytest.raises(Stage2BStateSnapshotError) as caught:
        encode_stage2b_state_snapshot(
            state=forged_state,
            expected_layout=layout,
            expected_registry=registry,
        )
    assert caught.value.code == "SNAPSHOT_TENSOR_MISMATCH"


def test_encode_rejects_noncanonical_bool_storage_bytes() -> None:
    state, layout, registry = _state_case(commits=1)
    bool_snapshot = next(
        tensor_snapshot
        for layer in state.layer_states
        for _name, tensor_snapshot in layer.tensors
        if tensor_snapshot.clone_tensor().dtype is torch.bool
    )
    bool_snapshot._tensor.view(torch.uint8).fill_(2)
    assert bool(bool_snapshot._tensor.all()) is True

    with pytest.raises(Stage2BStateSnapshotError, match="not canonical") as caught:
        encode_stage2b_state_snapshot(
            state=state,
            expected_layout=layout,
            expected_registry=registry,
        )
    assert caught.value.code == "SNAPSHOT_TENSOR_MISMATCH"


def test_tensor_descriptor_rejects_non_little_endian_identity() -> None:
    state, layout, registry = _state_case(commits=1)
    snapshot = encode_stage2b_state_snapshot(
        state=state,
        expected_layout=layout,
        expected_registry=registry,
    )
    manifest = _manifest(snapshot)
    descriptor = manifest["cache_state"]["layer_states"][0]["tensors"][0]
    assert descriptor["byte_order"] == "little"
    descriptor["byte_order"] = "big"

    with pytest.raises(Stage2BStateSnapshotError) as caught:
        _decode(
            _with_manifest(snapshot, manifest),
            layout=layout,
            registry=registry,
            variant=CACHStagingVariant.CACH_A,
        )
    assert caught.value.code == "SNAPSHOT_CPU_SYNTHETIC_ONLY"


def test_bfloat16_edge_bit_patterns_round_trip_without_numeric_conversion() -> None:
    state, layout, registry = _state_case(commits=1)
    source_snapshot = next(
        tensor_snapshot
        for layer in state.layer_states
        for _name, tensor_snapshot in layer.tensors
        if tensor_snapshot._tensor.dtype is torch.bfloat16
    )
    edge_bits = torch.tensor(
        [0x8000, 0x0001, 0x3F81, 0x7F7F],
        dtype=torch.uint16,
    )
    source_bits = source_snapshot._tensor.view(torch.uint16).reshape(-1)
    source_bits[: edge_bits.numel()].copy_(edge_bits)

    snapshot = encode_stage2b_state_snapshot(
        state=state,
        expected_layout=layout,
        expected_registry=registry,
    )
    decoded = _decode(
        snapshot,
        layout=layout,
        registry=registry,
        variant=CACHStagingVariant.CACH_A,
    )
    decoded_snapshot = next(
        tensor_snapshot
        for layer in decoded.layer_states
        for _name, tensor_snapshot in layer.tensors
        if tensor_snapshot.logical_tensor_id == source_snapshot.logical_tensor_id
    )
    decoded_bits = decoded_snapshot._tensor.view(torch.uint16).reshape(-1)
    assert torch.equal(decoded_bits[: edge_bits.numel()], edge_bits)
