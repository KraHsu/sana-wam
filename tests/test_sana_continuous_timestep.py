"""Unit tests for the T0/T1 SANA video timestep embedding contract."""

from __future__ import annotations

import pytest
import torch

from sana_wam.model.video_backbone.sana.blocks_split import (
    _timestep_for_embedding,
)
from sana_wam.model.video_backbone.sana.pipeline_builder import _spec_from_dict
from sana_wam.model.video_backbone.sana.scheduler import SanaFlowSchedulerAdapter


def test_video_backbone_config_defaults_t0_and_accepts_t1():
    base = {"model_kwargs": {"in_channels": 16}}
    t0 = _spec_from_dict(base)
    t1 = _spec_from_dict(
        {**base, "continuous_timestep_conditioning": True}
    )

    assert t0.continuous_timestep_conditioning is False
    assert t1.continuous_timestep_conditioning is True


def test_t0_transform_is_exactly_the_legacy_transform():
    timestep = torch.tensor([0.25, 1.75, 964.5], dtype=torch.bfloat16)
    actual = _timestep_for_embedding(
        timestep,
        1.0,
        continuous_timestep_conditioning=False,
    )
    expected = timestep.long().to(torch.float32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    normalized = _timestep_for_embedding(
        timestep,
        2.0,
        continuous_timestep_conditioning=False,
    )
    expected_normalized = (timestep.float() / 2.0).to(torch.float32)
    torch.testing.assert_close(
        normalized, expected_normalized, atol=0, rtol=0
    )


def test_t1_preserves_fractional_fp32_and_fails_closed_otherwise():
    timestep = torch.tensor([0.25, 1.75, 964.5], dtype=torch.float32)
    actual = _timestep_for_embedding(
        timestep,
        1.0,
        continuous_timestep_conditioning=True,
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, timestep, atol=0, rtol=0)

    normalized = _timestep_for_embedding(
        timestep,
        2.0,
        continuous_timestep_conditioning=True,
    )
    torch.testing.assert_close(normalized, timestep / 2.0, atol=0, rtol=0)

    with pytest.raises(TypeError, match="requires float32 video timesteps"):
        _timestep_for_embedding(
            timestep.to(torch.bfloat16),
            1.0,
            continuous_timestep_conditioning=True,
        )


@pytest.mark.parametrize("steps", [20, 50, 100, 200])
def test_t1_scheduler_grid_keeps_every_effective_timestep_unique(steps):
    scheduler = SanaFlowSchedulerAdapter()
    scheduler.set_timesteps(steps)
    effective = _timestep_for_embedding(
        scheduler.timesteps,
        1.0,
        continuous_timestep_conditioning=True,
    )

    assert effective.dtype == torch.float32
    assert torch.unique(effective).numel() == steps
    assert float(effective[-1]) > 0.0


def test_sub_integer_bin_values_split_only_under_t1():
    values = torch.tensor([100.125, 100.375], dtype=torch.float32)
    t0 = _timestep_for_embedding(
        values.to(torch.bfloat16),
        1.0,
        continuous_timestep_conditioning=False,
    )
    t1 = _timestep_for_embedding(
        values,
        1.0,
        continuous_timestep_conditioning=True,
    )

    assert t0[0].item() == t0[1].item()
    assert t1[0].item() != t1[1].item()
    assert _timestep_for_embedding(
        torch.zeros(1, dtype=torch.float32),
        1.0,
        continuous_timestep_conditioning=True,
    ).item() == 0.0
