from __future__ import annotations

from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    ExpansionSupportError,
    expansion_support_contract,
    expansion_support_for_child_window,
    expansion_support_for_window,
    validate_expansion_geometry,
    validate_expansion_support,
)


def _canonical(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _independent_mask_counts(actual_valid_raw_frames: int) -> dict[str, int | bool]:
    sampled = [index < actual_valid_raw_frames for index in range(0, 113, 4)]
    latent = [sampled[0]]
    for offset in range(1, len(sampled), 4):
        latent.append(any(sampled[offset : offset + 4]))
    nonbootstrap = list(latent)
    nonbootstrap[0] = False
    chunks = [
        any(nonbootstrap[offset : offset + 2])
        for offset in range(0, len(nonbootstrap), 2)
    ]
    return {
        "actual_valid_raw_frames": actual_valid_raw_frames,
        "valid_sampled_video_frames": sum(sampled),
        "valid_latent_frames": sum(latent),
        "nonbootstrap_valid_latent_frames": sum(nonbootstrap),
        "eligible_chunk_count": sum(chunks),
        "eligible": any(chunks),
    }


def test_frozen_contract_has_the_amendment_pin_and_canonical_sha() -> None:
    assert EXPANSION_ELIGIBILITY_AMENDMENT_SHA256 == (
        "8632450705ee3ba35855f07ccb48ad300aac5abf711abf0307808b52fb613081"
    )
    assert sha256(_canonical(EXPANSION_SUPPORT_CONTRACT)).hexdigest() == (
        EXPANSION_SUPPORT_CONTRACT_SHA256
    )
    isolated = expansion_support_contract()
    isolated["raw_window_frames"] = 999
    assert expansion_support_contract()["raw_window_frames"] == 113


@pytest.mark.parametrize("actual_valid", [2, 3, 4])
def test_two_three_and_four_raw_frames_have_no_expansion_support(
    actual_valid: int,
) -> None:
    support = expansion_support_for_window(
        episode_length=100,
        start_frame=100 - actual_valid,
        window_logical_length=113,
    )
    assert support["actual_valid_raw_frames"] == actual_valid
    assert support["eligible"] is False
    assert support["eligible_chunk_count"] == 0
    with pytest.raises(ExpansionSupportError, match="no non-bootstrap"):
        validate_expansion_support(support, require_eligible=True)


def test_five_raw_frames_are_the_minimum_structural_support() -> None:
    support = expansion_support_for_window(
        episode_length=100,
        start_frame=95,
        window_logical_length=113,
    )
    assert support == {
        "actual_valid_raw_frames": 5,
        "valid_sampled_video_frames": 2,
        "valid_latent_frames": 2,
        "nonbootstrap_valid_latent_frames": 1,
        "eligible_chunk_count": 1,
        "eligible": True,
    }
    assert validate_expansion_support(support, require_eligible=True) == support


def test_causal_grouping_and_chunk_any_keep_partial_chunks() -> None:
    # Raw frame 20 is the first valid sampled frame in latent 2.  Latent 2 is
    # alone in its two-frame chunk, which remains supported under `any`.
    support = expansion_support_for_window(
        episode_length=21,
        start_frame=0,
        window_logical_length=113,
    )
    assert support == {
        "actual_valid_raw_frames": 21,
        "valid_sampled_video_frames": 6,
        "valid_latent_frames": 3,
        "nonbootstrap_valid_latent_frames": 2,
        "eligible_chunk_count": 2,
        "eligible": True,
    }


def test_all_raw_lengths_match_an_independent_mask_derivation() -> None:
    for actual_valid in range(114):
        observed = expansion_support_for_window(
            episode_length=113,
            start_frame=113 - actual_valid,
            window_logical_length=113,
        )
        assert observed == _independent_mask_counts(actual_valid)


def test_structural_eligibility_matches_the_cpu_torch_chunk_mask() -> None:
    torch = pytest.importorskip("torch")
    from sana_wam.model.local_expansion import current_video_chunk_mask

    for actual_valid in range(114):
        support = _independent_mask_counts(actual_valid)
        sampled = [index < actual_valid for index in range(0, 113, 4)]
        latent_valid = [sampled[0]] + [
            any(sampled[offset : offset + 4]) for offset in range(1, 29, 4)
        ]
        video_is_pad = torch.tensor(
            [[not valid for valid in latent_valid]],
            dtype=torch.bool,
        )
        if not support["eligible"]:
            with pytest.raises(ValueError, match="no non-bootstrap"):
                current_video_chunk_mask(
                    video_is_pad,
                    batch_size=1,
                    num_frames=8,
                    frame_chunk_size=2,
                    bootstrap_clean_prefix=True,
                    device=torch.device("cpu"),
                )
            continue
        selected = current_video_chunk_mask(
            video_is_pad,
            batch_size=1,
            num_frames=8,
            frame_chunk_size=2,
            bootstrap_clean_prefix=True,
            device=torch.device("cpu"),
        )
        assert bool(selected.any().item()) is True


def test_geometry_and_serialized_support_fail_closed() -> None:
    with pytest.raises(ExpansionSupportError, match="video stride"):
        validate_expansion_geometry(
            raw_window_frames=113,
            video_stride=2,
            video_sample_indices=range(0, 113, 2),
            causal_temporal=True,
            temporal_compression=4,
        )
    support = expansion_support_for_window(
        episode_length=5,
        start_frame=0,
        window_logical_length=113,
    )
    support["eligible_chunk_count"] = 2
    with pytest.raises(ExpansionSupportError, match="structural derivation"):
        validate_expansion_support(support, require_eligible=True)
    with pytest.raises(ExpansionSupportError, match="integer"):
        expansion_support_for_window(
            episode_length=True,
            start_frame=0,
            window_logical_length=113,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("raw_window_frames", 113.0),
        ("video_stride", 4.0),
        ("video_sample_indices", [float(index) for index in range(0, 113, 4)]),
        ("causal_temporal", 1),
        ("temporal_compression", 4.0),
    ),
)
def test_geometry_rejects_same_value_float_and_bool_aliases(
    field: str,
    replacement,
) -> None:
    geometry = {
        "raw_window_frames": 113,
        "video_stride": 4,
        "video_sample_indices": list(range(0, 113, 4)),
        "causal_temporal": True,
        "temporal_compression": 4,
    }
    geometry[field] = replacement
    with pytest.raises(ExpansionSupportError, match="differ"):
        validate_expansion_geometry(**geometry)


@pytest.mark.parametrize(
    "field",
    (
        "actual_valid_raw_frames",
        "valid_sampled_video_frames",
        "valid_latent_frames",
        "nonbootstrap_valid_latent_frames",
        "eligible_chunk_count",
    ),
)
def test_serialized_support_rejects_float_counts_even_when_numerically_equal(
    field: str,
) -> None:
    support = expansion_support_for_window(
        episode_length=5,
        start_frame=0,
        window_logical_length=113,
    )
    support[field] = float(support[field])
    with pytest.raises(ExpansionSupportError, match="integer"):
        validate_expansion_support(support, require_eligible=True)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("actual_valid_raw_frames", True),
        ("nonbootstrap_valid_latent_frames", True),
        ("eligible_chunk_count", True),
        ("eligible", 1),
    ),
)
def test_serialized_support_rejects_bool_int_aliases(
    field: str,
    replacement: int | bool,
) -> None:
    support = expansion_support_for_window(
        episode_length=5,
        start_frame=0,
        window_logical_length=113,
    )
    support[field] = replacement
    with pytest.raises(ExpansionSupportError, match="integer|boolean"):
        validate_expansion_support(support, require_eligible=True)


def test_child_metadata_path_uses_episode_length_without_materialization() -> None:
    child = SimpleNamespace(
        _raw_window_len=113,
        video_stride=4,
        _video_sample_indices=list(range(0, 113, 4)),
        causal_temporal=True,
        temporal_compression=4,
        _episode_lengths=[10],
    )
    assert expansion_support_for_child_window(
        child,
        episode_index=0,
        start_frame=5,
        window_logical_length=113,
    )["eligible"] is True
    assert expansion_support_for_child_window(
        child,
        episode_index=0,
        start_frame=6,
        window_logical_length=113,
    )["eligible"] is False
