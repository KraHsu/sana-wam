from __future__ import annotations

import pytest
import torch
from torch import nn

from sana_wam.model.action_non_regression import (
    adapter_only_loss_surrogate,
    one_sided_relative_excess_loss,
    unweighted_pad_masked_action_mse,
)


def test_unweighted_pad_masked_mse_is_per_sample_and_ignores_pad():
    prediction = torch.tensor(
        [
            [[1.0, 3.0], [100.0, 100.0], [2.0, 4.0]],
            [[2.0, 0.0], [4.0, 2.0], [9.0, 9.0]],
        ]
    )
    target = torch.zeros_like(prediction)
    padded = torch.tensor(
        [[False, True, False], [False, False, True]], dtype=torch.bool
    )
    actual = unweighted_pad_masked_action_mse(prediction, target, padded)
    expected = torch.tensor(
        [((1.0 + 9.0) / 2 + (4.0 + 16.0) / 2) / 2, 6.0]
    )
    torch.testing.assert_close(actual, expected)


def test_one_sided_loss_is_zero_for_improvement_and_per_sample_before_mean():
    student = torch.tensor([0.5, 1.5, 1.0], requires_grad=True)
    reference = torch.tensor([1.0, 1.0, 1.0])
    loss = one_sided_relative_excess_loss(
        student, reference, relative_epsilon=1.0e-4
    )
    expected = ((0.5 / 1.0001) ** 2) / 3.0
    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.backward()
    assert student.grad[0].item() == 0.0
    assert student.grad[2].item() == 0.0
    assert student.grad[1].item() > 0.0


def test_adapter_surrogate_preserves_value_and_blocks_backbone_gradient():
    adapter = nn.Linear(2, 1, bias=True)
    backbone = nn.Linear(2, 2, bias=False)
    x = torch.tensor([[0.3, -0.7], [1.2, 0.5]])
    prediction = adapter(backbone(x)).square().mean()
    expected_adapter_grads = torch.autograd.grad(
        prediction, tuple(adapter.parameters()), retain_graph=True
    )

    surrogate = adapter_only_loss_surrogate(prediction, adapter)
    torch.testing.assert_close(surrogate.detach(), prediction.detach(), atol=0, rtol=0)
    surrogate.backward()

    assert backbone.weight.grad is None
    for parameter, expected in zip(
        adapter.parameters(), expected_adapter_grads, strict=True
    ):
        torch.testing.assert_close(parameter.grad, expected, atol=0, rtol=0)


def test_action_nr_validation_fails_closed():
    with pytest.raises(ValueError, match="identical shapes"):
        one_sided_relative_excess_loss(torch.ones(1), torch.ones(2))
    with pytest.raises(ValueError, match="non-negative"):
        one_sided_relative_excess_loss(torch.ones(1), -torch.ones(1))
    with pytest.raises(ValueError, match="finite scalar"):
        adapter_only_loss_surrogate(torch.tensor(float("nan")), nn.Linear(1, 1))
    with pytest.raises(ValueError, match="non-finite"):
        unweighted_pad_masked_action_mse(
            torch.full((1, 1, 1), float("nan")), torch.zeros(1, 1, 1)
        )
