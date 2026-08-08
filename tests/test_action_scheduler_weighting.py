"""CPU regression tests for action flow-matching loss weighting."""

import pytest
import torch

from sana_wam.model.action_backbone.scheduler import ActionScheduler


def _weights(mode):
    scheduler = ActionScheduler()
    scheduler.set_timesteps(1000, training=True, loss_weighting=mode)
    return scheduler


def test_none_mode_is_flat():
    scheduler = _weights("none")
    assert torch.allclose(
        scheduler.linear_timesteps_weights,
        torch.ones_like(scheduler.linear_timesteps_weights),
    )


def test_bsmntw_default_is_unchanged_and_mid_schedule_peaked():
    default = ActionScheduler()
    default.set_timesteps(1000, training=True)
    explicit = _weights("bsmntw")
    torch.testing.assert_close(default.linear_timesteps_weights, explicit.linear_timesteps_weights)
    weights = explicit.linear_timesteps_weights
    mid = int(torch.argmin((explicit.timesteps - explicit.num_train_timesteps / 2).abs()))
    assert weights[mid] > weights[0]
    assert weights[mid] > weights[-1]


def test_low_noise_mode_is_monotonic_and_normalized():
    scheduler = _weights("low_noise")
    weights = scheduler.linear_timesteps_weights
    assert weights[-1] > weights[0]
    assert ((weights[1:] - weights[:-1]) >= -1e-4).all()
    assert float(weights.mean()) == pytest.approx(1.0, abs=0.05)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown ActionScheduler loss_weighting"):
        _weights("bogus")


@pytest.mark.parametrize("mode", ["none", "bsmntw", "low_noise"])
def test_training_loss_weighting_does_not_change_inference_schedule(mode):
    reference = ActionScheduler()
    reference.set_timesteps(20, shift=5.0, training=False)
    candidate = ActionScheduler()
    candidate.set_timesteps(
        20,
        shift=5.0,
        training=False,
        loss_weighting=mode,
    )

    torch.testing.assert_close(candidate.timesteps, reference.timesteps)
    torch.testing.assert_close(candidate.sigmas, reference.sigmas)
    assert candidate.linear_timesteps_weights is None
