from __future__ import annotations

from copy import deepcopy

import pytest

from sana_wam.cach.config import ACTION_ORDER
from sana_wam.dataloader.cach_episode_origin import (
    CachEpisodeOriginDataset,
    EpisodeOriginManifestRow,
    validate_cach_episode_origin_config,
)
from sana_wam.model.action_chunk_layout import (
    LayoutContractError,
    LayoutReasonCode,
    canonical_proprio_row_sha256,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
    validate_cach_bootstrap_config,
)


class _SyntheticLegacyDataset:
    growing_history = False
    _filter_static_segments = False
    repeat = 1
    action_mode = "eef"
    delta_action = False
    action_dim = 20
    action_stats = {"mean": [0.0] * 20, "std": [1.0] * 20}
    action_stats_path = "/synthetic/not-a-real-file.npy"
    task_names = ("lift_pot",)

    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return deepcopy(self.samples[index])


def _sample(*, valid_raw_count=33, row_start=0, clean_latent=0, clean_actions=0):
    state_sequence = [
        tuple(float(raw_index * 100 + dim) for dim in range(20))
        for raw_index in range(valid_raw_count)
    ]
    return {
        "video": [f"frame-{index}" for index in range(valid_raw_count)],
        "first_frame_image": ["frame-0"],
        "video_mask": [True] * valid_raw_count,
        "action": [
            tuple(float(raw_index * 100 + dim) for dim in range(20))
            for raw_index in range(1, valid_raw_count)
        ],
        "action_mask": [True] * (valid_raw_count - 1),
        "proprio": [state_sequence[0]],
        "proprio_mask": [True],
        "proprio_seq": state_sequence,
        "num_clean_prefix_latent": clean_latent,
        "num_clean_prefix_actions": clean_actions,
        "prompt": "synthetic prompt",
        "episode_index": 0,
        "episode_path": "/synthetic/lift_pot/episode0.hdf5",
        "start_frame": row_start,
        "end_frame": row_start + valid_raw_count,
        "episode_length": 114,
        "task_name": "lift_pot",
        "active_arm": "both",
    }


def _wrapped_dataset(*, valid_raw_count=33, sample=None):
    if sample is None:
        sample = _sample(valid_raw_count=valid_raw_count)
    source = _SyntheticLegacyDataset([sample])
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=valid_raw_count,
        video_stride=1,
        source_row_label=f"row-{valid_raw_count}",
        episode_label="lift-pot-episode-0",
    )
    row = EpisodeOriginManifestRow(
        dataset_index=0,
        episode_index=0,
        episode_path="/synthetic/lift_pot/episode0.hdf5",
        task_name="lift_pot",
        source_row_digest=proof.source_row_digest,
        episode_id_digest=proof.episode_id_digest,
        row_start_raw_index=0,
        row_end_raw_index=valid_raw_count,
        timebase_manifest_digest=proof.timebase_manifest_digest,
        timestamp_verification_receipt_digest=(
            proof.verification_receipt_digest
        ),
    )
    wrapped = CachEpisodeOriginDataset.for_synthetic_tests(
        source,
        [row],
        layout_spec=spec,
        timebase_proofs={proof.source_row_digest: proof},
    )
    return source, wrapped


def test_bootstrap_config_accepts_only_new_first_frame_contract():
    validate_cach_bootstrap_config(
        {
            "episode_bootstrap": "first_frame_pinned",
            "observed_prefix_chunks": 0,
        }
    )

    with pytest.raises(LayoutContractError) as legacy:
        validate_cach_bootstrap_config(
            {
                "episode_bootstrap": "first_frame_pinned",
                "observed_prefix_chunks": 0,
                "ar_observed_prefix_chunks": 0,
            }
        )
    assert legacy.value.code is LayoutReasonCode.LEGACY_AR_PREFIX_KEY_PRESENT

    with pytest.raises(LayoutContractError) as drift:
        validate_cach_bootstrap_config(
            {
                "episode_bootstrap": "first_frame_pinned",
                "observed_prefix_chunks": 1,
            }
        )
    assert drift.value.code is LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH


def _valid_dataloader_config():
    return {
        "action_dim": 20,
        "action_mode": "eef",
        "action_order": list(ACTION_ORDER),
        "action_representation": "absolute_eef_target_xyz_rot6d_gripper",
        "action_stats_path": "/registered/action_stats.npy",
        "allow_random_row_substitution": False,
        "causal_temporal": True,
        "dataset_manifest": "/registered/dataset_manifest.json",
        "delta_action": False,
        "episode_origin_only": True,
        "filter_static_segments": False,
        "growing_history": False,
        "normalize_mode": "min-max",
        "num_frames": 33,
        "reject_nonzero_row_start": True,
        "repeat": 1,
        "row_rate_manifest": "/registered/row_rate_manifest.json",
        "seed": 1234,
        "temporal_compression": 8,
        "train_tasks": ["lift_pot"],
        "val_ratio": 0.0,
        "variant": "clean_50",
        "video_stride": 1,
        "window_stride": 1,
    }


def test_dataloader_config_requires_all_row_changing_defaults_explicit():
    config = _valid_dataloader_config()
    validate_cach_episode_origin_config(config)

    for key in (
        "num_frames",
        "variant",
        "filter_static_segments",
        "action_stats_path",
        "dataset_manifest",
        "row_rate_manifest",
    ):
        changed = dict(config)
        changed.pop(key)
        with pytest.raises(ValueError, match="legacy defaults explicit"):
            validate_cach_episode_origin_config(changed)


def test_dataloader_config_rejects_draft_null_variant_before_construction():
    config = _valid_dataloader_config()
    config["variant"] = None

    with pytest.raises(ValueError, match="variant"):
        validate_cach_episode_origin_config(config)


def test_dataloader_config_rejects_static_resampling_and_off_grid_geometry():
    config = _valid_dataloader_config()
    config["filter_static_segments"] = True
    with pytest.raises(ValueError, match="filter_static_segments"):
        validate_cach_episode_origin_config(config)

    config = _valid_dataloader_config()
    config["num_frames"] = 34
    with pytest.raises(LayoutContractError) as captured:
        validate_cach_episode_origin_config(config)
    assert captured.value.code is LayoutReasonCode.LAYOUT_GRID_MISMATCH


def test_episode_origin_wrapper_selects_only_layout_proprio_and_removes_future():
    source, wrapped = _wrapped_dataset(valid_raw_count=33)
    original = deepcopy(source.samples[0])

    output = wrapped[0]

    assert output["proprio_per_chunk_raw_indices"] == (0, 16)
    assert output["proprio_per_chunk"] == (
        original["proprio_seq"][0],
        original["proprio_seq"][16],
    )
    assert output["proprio_per_chunk_mask"] == (True, True)
    assert tuple(
        binding.raw_index
        for binding in output["proprio_per_chunk_bindings"]
    ) == (0, 16)
    assert all(
        binding.layout_instance_digest
        == output["layout_instance_digest"]
        for binding in output["proprio_per_chunk_bindings"]
    )
    expected_value_digests = tuple(
        canonical_proprio_row_sha256(row)
        for row in output["proprio_per_chunk"]
    )
    assert output["proprio_per_chunk_value_sha256s"] == expected_value_digests
    assert tuple(
        binding.proprio_value_sha256
        for binding in output["proprio_per_chunk_bindings"]
    ) == expected_value_digests
    first_binding = output["proprio_per_chunk_bindings"][0]
    first_binding.verify(
        layout_instance_digest=output["layout_instance_digest"],
        chunk=output["action_chunk_layout"].chunks[0],
        proprio_value_sha256=expected_value_digests[0],
    )
    with pytest.raises(LayoutContractError) as captured:
        first_binding.verify(
            layout_instance_digest=output["layout_instance_digest"],
            chunk=output["action_chunk_layout"].chunks[0],
            proprio_value_sha256=canonical_proprio_row_sha256([99.0] * 20),
        )
    assert captured.value.code is LayoutReasonCode.PROPRIO_BOUNDARY_MISMATCH
    assert output["action_chunk_layout"].valid_latent_count == 5
    assert "proprio_seq" not in output
    assert "proprio" not in output
    assert "proprio_mask" not in output
    assert "num_clean_prefix_latent" not in output
    assert "num_clean_prefix_actions" not in output
    assert output["first_frame_binding_schema"] == (
        "cach.first_frame_video0_binding.v1"
    )
    assert len(output["first_frame_content_sha256"]) == 64
    assert len(output["first_frame_binding_digest"]) == 64
    assert source.samples[0] == original


def test_model_facing_sample_is_built_from_an_exact_source_key_allowlist():
    sample = _sample(valid_raw_count=33)
    sample["_is_static"] = False
    _, wrapped = _wrapped_dataset(sample=sample)

    output = wrapped[0]

    assert "_is_static" not in output

    sample = _sample(valid_raw_count=33)
    sample["unregistered_metadata"] = "must not be silently forwarded"
    _, wrapped = _wrapped_dataset(sample=sample)
    with pytest.raises(ValueError, match="exact CACH allowlist"):
        wrapped[0]


@pytest.mark.parametrize(
    "field_name",
    [
        "future_joint_positions",
        "controller_state",
        "qvel_history",
        "robot_observation",
    ],
)
def test_unknown_state_like_payload_fails_closed(field_name):
    sample = _sample(valid_raw_count=33)
    sample[field_name] = [[0.0] * 20]
    _, wrapped = _wrapped_dataset(sample=sample)

    with pytest.raises(LayoutContractError) as captured:
        wrapped[0]
    assert captured.value.code is LayoutReasonCode.FUTURE_PROPRIO_VISIBLE


def test_first_frame_pinned_requires_one_exact_video_zero_binding():
    missing = _sample(valid_raw_count=33)
    missing.pop("first_frame_image")
    _, wrapped = _wrapped_dataset(sample=missing)
    with pytest.raises(LayoutContractError) as captured:
        wrapped[0]
    assert captured.value.code is LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH

    multiple = _sample(valid_raw_count=33)
    multiple["first_frame_image"] = ["frame-0", "frame-1"]
    _, wrapped = _wrapped_dataset(sample=multiple)
    with pytest.raises(LayoutContractError) as captured:
        wrapped[0]
    assert captured.value.code is LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH

    mismatched = _sample(valid_raw_count=33)
    mismatched["first_frame_image"] = ["different-frame"]
    _, wrapped = _wrapped_dataset(sample=mismatched)
    with pytest.raises(LayoutContractError) as captured:
        wrapped[0]
    assert captured.value.code is LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH
    assert "first_frame_content_sha256" in captured.value.context
    assert "video_zero_content_sha256" in captured.value.context


def test_first_frame_binding_accepts_distinct_exact_nested_list_content():
    sample = _sample(valid_raw_count=33)
    sample["video"][0] = [[0, 1], [2, 3]]
    sample["first_frame_image"] = [deepcopy(sample["video"][0])]
    _, wrapped = _wrapped_dataset(sample=sample)

    output = wrapped[0]

    assert len(output["first_frame_content_sha256"]) == 64
    assert len(output["first_frame_binding_digest"]) == 64


def test_l8_episode_origin_wrapper_uses_boundaries_0_16_40():
    _, wrapped = _wrapped_dataset(valid_raw_count=57)

    output = wrapped[0]

    assert output["proprio_per_chunk_raw_indices"] == (0, 16, 40)
    assert [
        (chunk.action_start, chunk.action_end)
        for chunk in output["action_chunk_layout"].chunks
    ] == [(0, 16), (16, 40), (40, 56)]


@pytest.mark.parametrize(
    ("clean_latent", "clean_actions"),
    [(1, 0), (0, 1), (3, 16)],
)
def test_nonzero_legacy_clean_prefix_fails_closed(
    clean_latent,
    clean_actions,
):
    sample = _sample(
        valid_raw_count=33,
        clean_latent=clean_latent,
        clean_actions=clean_actions,
    )
    _, wrapped = _wrapped_dataset(sample=sample)

    with pytest.raises(LayoutContractError) as captured:
        wrapped[0]
    assert captured.value.code is LayoutReasonCode.LEGACY_CLEAN_PREFIX_NONZERO


def test_manifest_row_itself_cannot_have_nonzero_start():
    proof = synthetic_equal_rate_proof(
        valid_raw_count=33,
        video_stride=1,
    )
    with pytest.raises(LayoutContractError) as captured:
        EpisodeOriginManifestRow(
            dataset_index=0,
            episode_index=0,
            episode_path="/synthetic/episode0.hdf5",
            task_name="lift_pot",
            source_row_digest=proof.source_row_digest,
            episode_id_digest=proof.episode_id_digest,
            row_start_raw_index=1,
            row_end_raw_index=34,
            timebase_manifest_digest=proof.timebase_manifest_digest,
            timestamp_verification_receipt_digest=(
                proof.verification_receipt_digest
            ),
        )
    assert captured.value.code is LayoutReasonCode.NONZERO_EPISODE_ROW_START


@pytest.mark.parametrize("mutation", ["short_action", "bad_action_width"])
def test_payload_shape_cannot_drift_from_registered_masks(mutation):
    sample = _sample(valid_raw_count=33)
    if mutation == "short_action":
        sample["action"] = sample["action"][:-1]
    else:
        sample["action"][0] = sample["action"][0][:-1]
    _, wrapped = _wrapped_dataset(sample=sample)

    with pytest.raises((LayoutContractError, ValueError)):
        wrapped[0]


def test_legacy_materializer_nonzero_start_fails_at_access():
    sample = _sample(valid_raw_count=33, row_start=1)
    _, wrapped = _wrapped_dataset(sample=sample)

    with pytest.raises(LayoutContractError) as captured:
        wrapped[0]
    assert captured.value.code is LayoutReasonCode.NONZERO_EPISODE_ROW_START


def test_production_constructor_rejects_synthetic_spec_and_proof():
    source = _SyntheticLegacyDataset([_sample()])
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=33,
        video_stride=1,
    )
    row = EpisodeOriginManifestRow(
        dataset_index=0,
        episode_index=0,
        episode_path="/synthetic/lift_pot/episode0.hdf5",
        task_name="lift_pot",
        source_row_digest=proof.source_row_digest,
        episode_id_digest=proof.episode_id_digest,
        row_start_raw_index=0,
        row_end_raw_index=33,
        timebase_manifest_digest=proof.timebase_manifest_digest,
        timestamp_verification_receipt_digest=(
            proof.verification_receipt_digest
        ),
    )
    with pytest.raises(LayoutContractError) as captured:
        CachEpisodeOriginDataset(
            source,
            [row],
            layout_spec=spec,
            timebase_proofs={proof.source_row_digest: proof},
            dataset_manifest_sha256="1" * 64,
            row_order_manifest_sha256="2" * 64,
        )
    assert captured.value.code is LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE


def test_source_that_can_randomly_replace_static_rows_is_rejected():
    source = _SyntheticLegacyDataset([_sample()])
    source._filter_static_segments = True
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=33,
        video_stride=1,
    )
    row = EpisodeOriginManifestRow(
        dataset_index=0,
        episode_index=0,
        episode_path="/synthetic/lift_pot/episode0.hdf5",
        task_name="lift_pot",
        source_row_digest=proof.source_row_digest,
        episode_id_digest=proof.episode_id_digest,
        row_start_raw_index=0,
        row_end_raw_index=33,
        timebase_manifest_digest=proof.timebase_manifest_digest,
        timestamp_verification_receipt_digest=(
            proof.verification_receipt_digest
        ),
    )
    with pytest.raises(ValueError, match="filter_static_segments"):
        CachEpisodeOriginDataset.for_synthetic_tests(
            source,
            [row],
            layout_spec=spec,
            timebase_proofs={proof.source_row_digest: proof},
        )
