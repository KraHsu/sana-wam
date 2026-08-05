from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from sana_wam.model.action_chunk_layout import (
    LayoutContractError,
    LayoutReasonCode,
    build_chunk_action_layout,
    build_synthetic_chunk_action_layout_for_tests,
    canonical_sha256,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)


def _build_equal_rate(
    *,
    valid_raw_count: int,
    frame_chunk_size: int = 3,
    temporal_compression: int = 8,
    video_stride: int = 1,
    padded_raw_count: int | None = None,
):
    if padded_raw_count is None:
        padded_raw_count = valid_raw_count
    spec = synthetic_layout_spec(
        frame_chunk_size=frame_chunk_size,
        temporal_compression=temporal_compression,
        video_stride=video_stride,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=valid_raw_count,
        video_stride=video_stride,
    )
    total_video_count = ((padded_raw_count - 1) // video_stride) + 1
    valid_video_count = ((valid_raw_count - 1) // video_stride) + 1
    video_mask = tuple(
        index < valid_video_count for index in range(total_video_count)
    )
    action_mask = tuple(
        index < valid_raw_count - 1
        for index in range(padded_raw_count - 1)
    )
    layout = build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=valid_raw_count,
        video_valid_mask=video_mask,
        action_valid_mask=action_mask,
    )
    return spec, proof, layout


def test_l8_layout_has_anchor_aware_full_and_tail_chunks():
    # L=8 with tc=8,vs=1 requires N_raw=(L-1)*8+1=57.
    _, _, layout = _build_equal_rate(valid_raw_count=57)

    assert layout.valid_latent_count == 8
    assert [
        (chunk.latent_start, chunk.latent_end)
        for chunk in layout.chunks
    ] == [(0, 3), (3, 6), (6, 8)]
    assert [
        (chunk.action_start, chunk.action_end)
        for chunk in layout.chunks
    ] == [(0, 16), (16, 40), (40, 56)]
    assert [chunk.valid_action_count for chunk in layout.chunks] == [16, 24, 16]
    assert [chunk.action_slot_capacity for chunk in layout.chunks] == [16, 24, 24]
    assert layout.chunks[-1].action_valid_mask == (
        (True,) * 16 + (False,) * 8
    )
    assert layout.proprio_raw_indices == (0, 16, 40)
    assert layout.chunks[-1].is_partial_tail is True
    assert layout.chunks[-1].latent_valid_mask == (True, True, False)


def test_candidate_num_frames_33_is_l5_with_one_partial_tail():
    # The Stage-0 draft's legacy default num_frames=33 resolves to L=5.
    _, _, layout = _build_equal_rate(valid_raw_count=33)

    assert layout.valid_video_count == 33
    assert layout.valid_latent_count == 5
    assert [
        (
            chunk.latent_start,
            chunk.latent_end,
            chunk.action_start,
            chunk.action_end,
            chunk.proprio_raw_index,
        )
        for chunk in layout.chunks
    ] == [
        (0, 3, 0, 16, 0),
        (3, 5, 16, 32, 16),
    ]
    assert layout.chunks[1].is_partial_tail is True
    assert layout.chunks[1].action_rope_positions == tuple(range(16, 32))
    assert layout.chunks[1].action_slot_capacity == 24
    assert layout.chunks[1].action_valid_mask == (
        (True,) * 16 + (False,) * 8
    )


def test_single_anchor_latent_owns_no_action_and_no_rope_position():
    _, _, layout = _build_equal_rate(valid_raw_count=1)

    assert layout.valid_latent_count == 1
    assert layout.valid_action_count == 0
    assert len(layout.chunks) == 1
    chunk = layout.chunks[0]
    assert (chunk.action_start, chunk.action_end) == (0, 0)
    assert chunk.action_rope_positions == ()
    assert chunk.action_slot_capacity == 16
    assert chunk.action_valid_mask == (False,) * 16
    assert chunk.latent_action_spans[0].anchor_no_action_slot is True
    assert chunk.latent_action_spans[0].action_count == 0


def test_padded_buffer_does_not_extend_valid_ownership():
    # N_raw=17 is exactly L=3; the remainder of the fixed 33-frame buffer is pad.
    _, _, layout = _build_equal_rate(
        valid_raw_count=17,
        padded_raw_count=33,
    )

    assert layout.valid_latent_count == 3
    assert layout.valid_action_count == 16
    assert len(layout.chunks) == 1
    assert (layout.chunks[0].action_start, layout.chunks[0].action_end) == (
        0,
        16,
    )
    assert layout.chunks[0].action_valid_mask == (True,) * 16


def test_action_token_zero_belongs_to_latent_one_not_the_anchor():
    _, proof, layout = _build_equal_rate(valid_raw_count=33)

    anchor, latent_one = layout.chunks[0].latent_action_spans[:2]
    assert (anchor.action_start, anchor.action_end) == (0, 0)
    assert (latent_one.action_start, latent_one.action_end) == (0, 8)
    assert proof.action_destination_state_indices[0] == 1
    assert proof.action_destination_timestamps[0] == proof.observation_timestamps[1]


def test_non_prefix_mask_fails_closed():
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=1,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=3,
        video_stride=1,
    )
    with pytest.raises(LayoutContractError) as captured:
        build_synthetic_chunk_action_layout_for_tests(
            spec=spec,
            proof=proof,
            row_start_raw_index=0,
            valid_raw_count=3,
            video_valid_mask=(True, True, True),
            action_valid_mask=(True, False, True),
        )
    assert captured.value.code is LayoutReasonCode.NON_PREFIX_VALID_MASK


def test_nonzero_row_start_cannot_masquerade_as_episode_bootstrap():
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=33,
        video_stride=1,
    )
    with pytest.raises(LayoutContractError) as captured:
        build_synthetic_chunk_action_layout_for_tests(
            spec=spec,
            proof=proof,
            row_start_raw_index=1,
            valid_raw_count=33,
            video_valid_mask=(True,) * 33,
            action_valid_mask=(True,) * 32,
        )
    assert captured.value.code is LayoutReasonCode.NONZERO_EPISODE_ROW_START


def test_off_grid_valid_tail_fails_instead_of_becoming_padding():
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=18,
        video_stride=1,
    )
    with pytest.raises(LayoutContractError) as captured:
        build_synthetic_chunk_action_layout_for_tests(
            spec=spec,
            proof=proof,
            row_start_raw_index=0,
            valid_raw_count=18,
            video_valid_mask=(True,) * 18,
            action_valid_mask=(True,) * 17,
        )
    assert captured.value.code is LayoutReasonCode.LAYOUT_GRID_MISMATCH


def test_synthetic_proof_is_rejected_by_production_builder():
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=33,
        video_stride=1,
    )
    with pytest.raises(LayoutContractError) as captured:
        build_chunk_action_layout(
            spec=spec,
            proof=proof,
            row_start_raw_index=0,
            valid_raw_count=33,
            video_valid_mask=(True,) * 33,
            action_valid_mask=(True,) * 32,
        )
    assert captured.value.code is LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE


def test_caller_forged_verified_manifest_metadata_cannot_unlock_production():
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    synthetic = synthetic_equal_rate_proof(
        valid_raw_count=33,
        video_stride=1,
    )
    forged = replace(
        synthetic,
        proof_mode="verified_manifest",
        synthetic_test_marker=None,
    )
    with pytest.raises(LayoutContractError) as captured:
        build_chunk_action_layout(
            spec=replace(spec, synthetic_test_only=False),
            proof=forged,
            row_start_raw_index=0,
            valid_raw_count=33,
            video_valid_mask=(True,) * 33,
            action_valid_mask=(True,) * 32,
        )
    assert captured.value.code is LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE


def test_spec_and_instance_digests_are_distinct_and_deterministic():
    spec, _, first = _build_equal_rate(valid_raw_count=33)
    _, _, second = _build_equal_rate(valid_raw_count=33)
    _, _, changed = _build_equal_rate(valid_raw_count=57)

    assert first.layout_spec_sha256 == spec.layout_spec_sha256
    assert first.layout_instance_digest == second.layout_instance_digest
    assert first.layout_instance_digest != changed.layout_instance_digest
    assert first.layout_spec_sha256 != first.layout_instance_digest
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256(
        {"a": 1, "b": 2}
    )


def test_layout_dataclasses_are_immutable():
    spec, _, layout = _build_equal_rate(valid_raw_count=33)
    with pytest.raises(FrozenInstanceError):
        spec.video_stride = 2
    with pytest.raises(FrozenInstanceError):
        layout.valid_action_count = 99


def test_manual_span_and_partial_tail_tampering_is_rejected():
    _, _, layout = _build_equal_rate(valid_raw_count=33)
    first = layout.chunks[0]
    bad_span = replace(first.latent_action_spans[1], latent_index=2)
    with pytest.raises(LayoutContractError):
        replace(
            first,
            latent_action_spans=(
                first.latent_action_spans[0],
                bad_span,
                first.latent_action_spans[2],
            ),
        )

    tail = layout.chunks[-1]
    with pytest.raises(LayoutContractError):
        replace(tail, is_partial_tail=False)
    with pytest.raises(LayoutContractError):
        replace(tail, action_valid_mask=(True,) * tail.action_slot_capacity)
