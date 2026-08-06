from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sana_wam.model.base import BaseWAMArchitecture


class _Container(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_backbone = nn.Sequential(
            nn.Linear(3, 4, bias=False),
            nn.SiLU(),
            nn.Linear(4, 2, bias=False),
        )
        with torch.no_grad():
            for parameter in self.video_backbone.parameters():
                parameter.fill_(0.25)


def _freeze(container: _Container, *, preserve_input_grad: bool = False) -> None:
    frozen = BaseWAMArchitecture.freeze_modules(
        container,
        ["video_backbone"],
        preserve_input_grad=preserve_input_grad,
    )
    assert frozen == ["video_backbone"]


def test_default_freeze_recursively_disables_input_autograd() -> None:
    container = _Container()
    _freeze(container)

    value = torch.ones(2, 3, requires_grad=True)
    assert not container.video_backbone(value).requires_grad
    assert not container.video_backbone[0](value).requires_grad
    assert all(
        not parameter.requires_grad
        for parameter in container.video_backbone.parameters()
    )
    assert all(not module.training for module in container.video_backbone.modules())
    assert all(
        getattr(module, "_sana_wam_no_grad_wrapped", False)
        for module in container.video_backbone.modules()
    )


def test_preserved_frozen_input_autograd_reaches_inputs_not_parameters() -> None:
    container = _Container()
    _freeze(container, preserve_input_grad=True)

    value = torch.ones(2, 3, requires_grad=True)
    container.video_backbone(value).sum().backward()

    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert torch.count_nonzero(value.grad).item() > 0
    assert all(
        not parameter.requires_grad
        for parameter in container.video_backbone.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in container.video_backbone.parameters()
    )
    assert all(not module.training for module in container.video_backbone.modules())
    assert all(
        not getattr(module, "_sana_wam_no_grad_wrapped", False)
        for module in container.video_backbone.modules()
    )

    direct_value = torch.ones(2, 3, requires_grad=True)
    container.video_backbone[0](direct_value).sum().backward()
    assert direct_value.grad is not None
    assert torch.count_nonzero(direct_value.grad).item() > 0


def test_preserve_input_grad_cannot_reverse_existing_no_grad_wrappers() -> None:
    container = _Container()
    _freeze(container)
    container.video_backbone.requires_grad_(True)
    container.video_backbone.train()

    with pytest.raises(RuntimeError, match="cannot preserve frozen input gradients"):
        _freeze(container, preserve_input_grad=True)
    assert all(
        parameter.requires_grad
        for parameter in container.video_backbone.parameters()
    )
    assert all(module.training for module in container.video_backbone.modules())
