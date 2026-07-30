from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sana_wam.model.video_backbone.sana.strictly_positive_feature_map import (
    StrictlyPositiveLearnableFeatureMap,
    install_strictly_positive_feature_maps,
)


class _LearnableFeatureMap(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor([0.25]))


class _Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.kernel_func = _LearnableFeatureMap(dim)


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = _Attention(dim)


class _DiT(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(depth)])


def test_install_preserves_checkpoint_keys_and_parameter_objects():
    model = _DiT(dim=8, depth=3)
    keys_before = set(model.state_dict())
    parameters_before = {
        name: parameter for name, parameter in model.named_parameters()
    }

    count = install_strictly_positive_feature_maps(
        model, {"beta": 16.0, "delta": 1e-4}
    )

    assert count == 3
    assert set(model.state_dict()) == keys_before
    parameters_after = dict(model.named_parameters())
    assert parameters_after.keys() == parameters_before.keys()
    assert all(
        parameters_after[name] is parameter
        for name, parameter in parameters_before.items()
    )


def test_output_is_strictly_positive_and_has_finite_gradients():
    feature_map = StrictlyPositiveLearnableFeatureMap(
        4, beta=16.0, delta=1e-4
    )
    inputs = torch.tensor(
        [[[
            [-100.0, -1.0, 0.0],
            [0.1, 1.0, 100.0],
            [-10.0, 0.5, 10.0],
            [-0.1, 0.0, 0.1],
        ]]],
        requires_grad=True,
    )
    output = feature_map(inputs)
    assert bool((output >= 1e-4).all())
    output.sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in feature_map.parameters()
    )


def test_softplus_beta_approximates_relu_away_from_zero():
    feature_map = StrictlyPositiveLearnableFeatureMap(
        4, beta=16.0, delta=1e-4
    )
    feature_map.alpha.data.zero_()
    inputs = torch.tensor([[[[-2.0, -1.0], [1.0, 2.0], [4.0, -4.0], [0.5, -0.5]]]])
    actual = feature_map(inputs)
    expected = torch.relu(inputs) + 1e-4
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize(
    "config,match",
    [
        ({"delta": 0.0}, "delta must be positive"),
        ({"beta": 0.0}, "beta must be positive"),
        ({"unknown": 1}, "unknown strictly_positive_feature_map fields"),
    ],
)
def test_invalid_config_fails_closed(config, match):
    with pytest.raises((ValueError, TypeError), match=match):
        install_strictly_positive_feature_maps(_DiT(4, 1), config)
