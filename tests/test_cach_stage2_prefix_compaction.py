from __future__ import annotations

import pytest
import torch

from sana_wam.cach.prefix_compaction import FixedKPrefixPlan


@pytest.mark.parametrize(("mask", "valid"), [([True, True, False], 2), ([True, True, True], 3)])
def test_fixed_k_prefix_compaction_restores_video_and_bridge(mask, valid) -> None:
    frame_mask = torch.tensor([mask, mask], dtype=torch.bool)
    plan = FixedKPrefixPlan.from_mask(frame_mask)
    assert (plan.batch_size, plan.fixed_slots, plan.valid_slots) == (2, 3, valid)

    video = torch.zeros(2, 4, 3, 2, 2)
    video[:, :, :valid] = torch.arange(
        2 * 4 * valid * 2 * 2,
        dtype=torch.float32,
    ).reshape(2, 4, valid, 2, 2)
    compact_video = plan.compact_video(video)
    assert compact_video.shape == (2, 4, valid, 2, 2)
    restored_video = plan.restore_video(compact_video)
    assert restored_video.shape == video.shape
    assert torch.equal(restored_video, video)

    spatial_tokens, width = 4, 8
    bridge = torch.arange(
        2 * valid * spatial_tokens * width,
        dtype=torch.float32,
    ).reshape(2, valid * spatial_tokens, width)
    restored_bridge = plan.restore_token_sequence(bridge)
    assert restored_bridge.shape == (2, 3 * spatial_tokens, width)
    assert torch.equal(
        restored_bridge[:, : valid * spatial_tokens],
        bridge,
    )
    assert not bool(restored_bridge[:, valid * spatial_tokens :].count_nonzero())
    assert torch.equal(plan.compact_token_sequence(restored_bridge), bridge)


def test_fixed_k_prefix_compaction_rejects_heterogeneous_or_nonprefix_masks() -> None:
    with pytest.raises(ValueError, match="batch-uniform"):
        FixedKPrefixPlan.from_mask(
            torch.tensor(
                [[True, True, False], [True, False, False]],
                dtype=torch.bool,
            )
        )
    with pytest.raises(ValueError, match="contiguous prefix"):
        FixedKPrefixPlan.from_mask(
            torch.tensor([[True, False, True]], dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="valid latent prefix"):
        FixedKPrefixPlan.from_mask(torch.zeros(1, 3, dtype=torch.bool))


@pytest.mark.parametrize("kind", ["video", "condition", "tokens"])
def test_fixed_k_prefix_compaction_rejects_nonzero_padding(kind: str) -> None:
    plan = FixedKPrefixPlan.from_mask(
        torch.tensor([[True, True, False]], dtype=torch.bool)
    )
    if kind == "video":
        value = torch.zeros(1, 4, 3, 2, 2)
        value[:, :, 2] = 1
        with pytest.raises(ValueError, match="video slots must be exact zero"):
            plan.compact_video(value)
    elif kind == "condition":
        value = torch.zeros(1, 3, 20)
        value[:, 2] = 1
        with pytest.raises(ValueError, match="condition slots must be exact zero"):
            plan.compact_condition(value)
    else:
        value = torch.zeros(1, 3 * 4, 8)
        value[:, 2 * 4 :] = 1
        with pytest.raises(ValueError, match="token-sequence slots must be exact zero"):
            plan.compact_token_sequence(value)


def test_frame_index_is_compacted_with_the_same_fixed_k_plan() -> None:
    plan = FixedKPrefixPlan.from_mask(
        torch.tensor([[True, True, False]], dtype=torch.bool)
    )
    assert torch.equal(
        plan.compact_frame_index(torch.tensor([7, 8, 9])),
        torch.tensor([7, 8]),
    )
    with pytest.raises(ValueError, match="fixed-K axis"):
        plan.compact_frame_index(torch.tensor([7, 8]))
