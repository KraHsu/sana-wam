"""Fail-closed warm-start, deploy, and optimizer contracts for Phase-6 A-on."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors.torch import save_file

from sana_wam.model.action_video_memory_adapter import ActionVideoMemoryAdapter
from sana_wam.model.base import BaseWAMArchitecture
from sana_wam.train.trainer import (
    Trainer,
    _ACTION_VIDEO_MEMORY_ADAPTER_PREFIX,
)


class _CheckpointModel(nn.Module):
    def __init__(self, *, adapter: bool):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.action_video_memory_adapter = (
            ActionVideoMemoryAdapter(
                num_layers=1,
                num_heads=1,
                head_dim=8,
                rank=8,
                init_seed=17,
            )
            if adapter
            else None
        )


class _GroupArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_backbone = nn.Linear(2, 2)
        self.action_backbone = nn.Linear(2, 2)
        self.action_video_memory_adapter = ActionVideoMemoryAdapter(
            num_layers=1,
            num_heads=1,
            head_dim=8,
            rank=8,
        )

    def get_trainable_modules(self, freeze_list=()):
        excluded = set(freeze_list)
        return {
            name: module
            for name, module in self.named_children()
            if name not in excluded
            and any(parameter.requires_grad for parameter in module.parameters())
        }


class _WarmStartModel(_CheckpointModel):
    def __init__(self):
        super().__init__(adapter=True)
        self.load_calls = []

    def load_checkpoint(
        self, path, strict=True, allow_missing_patterns=()
    ):
        self.load_calls.append((path, strict, allow_missing_patterns))
        return BaseWAMArchitecture.load_checkpoint(
            self,
            path,
            strict=strict,
            allow_missing_patterns=allow_missing_patterns,
        )


class _WeightOnlyNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((4,), 0.01))


class _YNormWarmStartModel(_WarmStartModel):
    def __init__(self):
        super().__init__()
        self.video_backbone = nn.Module()
        self.video_backbone.dit = nn.Module()
        self.video_backbone.dit.attention_y_norm = _WeightOnlyNorm()


def _trainer_for(architecture, **training):
    trainer = object.__new__(Trainer)
    trainer.architecture = architecture
    trainer.t = OmegaConf.create(
        {
            "action_lr": 2.0e-5,
            "video_lr": 5.0e-6,
            **training,
        }
    )
    return trainer


def _save_state(path: Path, state: dict[str, torch.Tensor]) -> None:
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in state.items()},
        str(path),
    )


def _adapter_keys(model) -> set[str]:
    return {
        key
        for key in model.state_dict()
        if key.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
    }


def test_a_on_student_accepts_only_all_missing_adapter_keys(tmp_path):
    old = _CheckpointModel(adapter=False)
    checkpoint = tmp_path / "a_off.safetensors"
    _save_state(checkpoint, old.state_dict())

    student = _CheckpointModel(adapter=True)
    trainer = _trainer_for(student)
    allowed = trainer._action_adapter_warm_start_contract(str(checkpoint))

    assert allowed == (_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX,)
    adapter = student.action_video_memory_adapter
    assert torch.count_nonzero(adapter.k_right).item() == 0
    assert torch.count_nonzero(adapter.v_right).item() == 0
    assert torch.count_nonzero(adapter.z_logits).item() == 0


def test_trainer_loads_a_off_student_with_only_adapter_missing(tmp_path):
    torch.manual_seed(21)
    old = _CheckpointModel(adapter=False)
    checkpoint = tmp_path / "registered_init.safetensors"
    BaseWAMArchitecture.save_checkpoint(old, str(checkpoint))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    student = _WarmStartModel()
    trainer = _trainer_for(
        student,
        init_checkpoint=str(checkpoint),
        init_checkpoint_sha256=digest,
    )
    trainer._load_initial_checkpoint()

    assert student.load_calls == [
        (
            str(checkpoint.resolve()),
            True,
            (_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX,),
        )
    ]
    torch.testing.assert_close(student.base.weight, old.base.weight, atol=0, rtol=0)
    torch.testing.assert_close(student.base.bias, old.base.bias, atol=0, rtol=0)
    adapter = student.action_video_memory_adapter
    assert torch.count_nonzero(adapter.k_right).item() == 0
    assert torch.count_nonzero(adapter.v_right).item() == 0
    assert torch.count_nonzero(adapter.z_logits).item() == 0


def test_explicit_missing_pattern_merges_with_adapter_and_preserves_base_norm(
    tmp_path,
):
    old = _CheckpointModel(adapter=False)
    checkpoint = tmp_path / "pre_y_norm.safetensors"
    BaseWAMArchitecture.save_checkpoint(old, str(checkpoint))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    student = _YNormWarmStartModel()
    pretrained_norm = student.video_backbone.dit.attention_y_norm.weight.detach().clone()
    trainer = _trainer_for(
        student,
        init_checkpoint=str(checkpoint),
        init_checkpoint_sha256=digest,
        init_checkpoint_allow_missing_patterns=[
            "video_backbone.dit.attention_y_norm.weight"
        ],
    )
    trainer._load_initial_checkpoint()

    assert student.load_calls == [
        (
            str(checkpoint.resolve()),
            True,
            (
                "video_backbone.dit.attention_y_norm.weight",
                _ACTION_VIDEO_MEMORY_ADAPTER_PREFIX,
            ),
        )
    ]
    torch.testing.assert_close(
        student.video_backbone.dit.attention_y_norm.weight,
        pretrained_norm,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "value",
    [
        "video_backbone.dit.attention_y_norm.weight",
        [1],
        [""],
        [" video_backbone.dit.attention_y_norm.weight"],
    ],
)
def test_explicit_missing_patterns_require_strict_list_of_strings(value):
    trainer = _trainer_for(
        _CheckpointModel(adapter=False),
        init_checkpoint_allow_missing_patterns=value,
    )
    with pytest.raises(ValueError, match="init_checkpoint_allow_missing_patterns"):
        trainer._load_initial_checkpoint()


def test_explicit_missing_pattern_does_not_tolerate_other_missing_keys(tmp_path):
    checkpoint = tmp_path / "pre_y_norm.safetensors"
    BaseWAMArchitecture.save_checkpoint(
        _CheckpointModel(adapter=False), str(checkpoint)
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    student = _YNormWarmStartModel()
    trainer = _trainer_for(
        student,
        init_checkpoint=str(checkpoint),
        init_checkpoint_sha256=digest,
        init_checkpoint_allow_missing_patterns=["some_other_parameter"],
    )

    with pytest.raises(RuntimeError, match="attention_y_norm.weight"):
        trainer._load_initial_checkpoint()


def test_explicit_missing_pattern_never_tolerates_unexpected_keys(tmp_path):
    state = _CheckpointModel(adapter=False).state_dict()
    state["unexpected.weight"] = torch.ones(1)
    checkpoint = tmp_path / "unexpected.safetensors"
    _save_state(checkpoint, state)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    student = _YNormWarmStartModel()
    trainer = _trainer_for(
        student,
        init_checkpoint=str(checkpoint),
        init_checkpoint_sha256=digest,
        init_checkpoint_allow_missing_patterns=[
            "video_backbone.dit.attention_y_norm.weight"
        ],
    )

    with pytest.raises(RuntimeError, match="unexpected.weight"):
        trainer._load_initial_checkpoint()


@pytest.mark.parametrize("mode", ["partial", "complete"])
def test_a_on_student_rejects_any_adapter_state(tmp_path, mode):
    source = _CheckpointModel(adapter=True)
    state = source.state_dict()
    adapter_keys = sorted(_adapter_keys(source))
    if mode == "partial":
        keep = set(adapter_keys[:1])
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
            or key in keep
        }
    checkpoint = tmp_path / f"{mode}.safetensors"
    _save_state(checkpoint, state)

    trainer = _trainer_for(_CheckpointModel(adapter=True))
    with pytest.raises(RuntimeError, match=f"found {mode} adapter state"):
        trainer._action_adapter_warm_start_contract(str(checkpoint))


def test_a_on_student_rejects_nonidentity_default(tmp_path):
    checkpoint = tmp_path / "a_off.safetensors"
    _save_state(checkpoint, _CheckpointModel(adapter=False).state_dict())
    student = _CheckpointModel(adapter=True)
    with torch.no_grad():
        student.action_video_memory_adapter.k_right.view(-1)[0] = 1.0

    trainer = _trainer_for(student)
    with pytest.raises(RuntimeError, match="not exact identity initialization"):
        trainer._action_adapter_warm_start_contract(str(checkpoint))


def test_a_on_student_rejects_nonfinite_identity_factor(tmp_path):
    checkpoint = tmp_path / "a_off.safetensors"
    _save_state(checkpoint, _CheckpointModel(adapter=False).state_dict())
    student = _CheckpointModel(adapter=True)
    with torch.no_grad():
        student.action_video_memory_adapter.k_left.view(-1)[0] = math.nan

    trainer = _trainer_for(student)
    with pytest.raises(RuntimeError, match="contains non-finite parameters"):
        trainer._action_adapter_warm_start_contract(str(checkpoint))


def test_a_off_old_checkpoint_strict_roundtrip_is_bitwise(tmp_path):
    torch.manual_seed(3)
    source = _CheckpointModel(adapter=False)
    checkpoint = tmp_path / "a_off.safetensors"
    BaseWAMArchitecture.save_checkpoint(source, str(checkpoint))

    torch.manual_seed(9)
    restored = _CheckpointModel(adapter=False)
    BaseWAMArchitecture.load_checkpoint(restored, str(checkpoint), strict=True)

    assert _adapter_keys(restored) == set()
    for key, value in source.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[key], value, atol=0, rtol=0
        )


@pytest.mark.parametrize("mode", ["missing", "partial"])
def test_formal_a_on_deploy_rejects_incomplete_adapter_state(tmp_path, mode):
    source = _CheckpointModel(adapter=(mode == "partial"))
    state = source.state_dict()
    if mode == "partial":
        first = sorted(_adapter_keys(source))[0]
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(_ACTION_VIDEO_MEMORY_ADAPTER_PREFIX)
            or key == first
        }
    checkpoint = tmp_path / f"deploy_{mode}.safetensors"
    _save_state(checkpoint, state)

    deployed = _CheckpointModel(adapter=True)
    with pytest.raises(RuntimeError, match="Strict load failed"):
        BaseWAMArchitecture.load_checkpoint(
            deployed,
            str(checkpoint),
            strict=True,
            allow_missing_patterns=("norm_kv",),
        )


def test_full_a_on_checkpoint_save_reload_preserves_state_and_output(tmp_path):
    source = _CheckpointModel(adapter=True)
    with torch.no_grad():
        adapter = source.action_video_memory_adapter
        adapter.k_right.add_(0.02)
        adapter.v_right.sub_(0.03)
        adapter.z_logits.add_(0.04)
    checkpoint = tmp_path / "a_on_full.safetensors"
    BaseWAMArchitecture.save_checkpoint(source, str(checkpoint))

    restored = _CheckpointModel(adapter=True)
    BaseWAMArchitecture.load_checkpoint(
        restored,
        str(checkpoint),
        strict=True,
        allow_missing_patterns=("norm_kv",),
    )

    assert _adapter_keys(restored) == _adapter_keys(source)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[key], value, atol=0, rtol=0
        )
    S = torch.randn(2, 1, 8, 8)
    z = torch.randn(2, 1, 1, 8)
    expected = source.action_video_memory_adapter.forward_delta(0, S, z)
    actual = restored.action_video_memory_adapter.forward_delta(0, S, z)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


def test_a_off_deploy_rejects_a_on_checkpoint_as_unexpected(tmp_path):
    checkpoint = tmp_path / "a_on.safetensors"
    BaseWAMArchitecture.save_checkpoint(
        _CheckpointModel(adapter=True), str(checkpoint)
    )
    with pytest.raises(RuntimeError, match="unexpected"):
        BaseWAMArchitecture.load_checkpoint(
            _CheckpointModel(adapter=False), str(checkpoint), strict=True
        )


def test_action_memory_has_independent_registered_lr_group():
    architecture = _GroupArchitecture()
    trainer = _trainer_for(architecture, action_memory_lr=1.0e-4)
    groups = trainer._param_groups()

    assert [group["lr"] for group in groups] == [2.0e-5, 1.0e-4, 5.0e-6]
    expected = [
        {id(parameter) for parameter in architecture.action_backbone.parameters()},
        {
            id(parameter)
            for parameter in architecture.action_video_memory_adapter.parameters()
        },
        {id(parameter) for parameter in architecture.video_backbone.parameters()},
    ]
    actual = [{id(parameter) for parameter in group["params"]} for group in groups]
    assert actual == expected
    assert not (actual[0] & actual[1] or actual[0] & actual[2] or actual[1] & actual[2])


@pytest.mark.parametrize(
    "value", [True, 0.0, -1.0, math.inf, math.nan, "not-a-number"]
)
def test_invalid_action_memory_lr_fails_closed(value):
    trainer = _trainer_for(_GroupArchitecture(), action_memory_lr=value)
    with pytest.raises(ValueError, match="positive finite number"):
        trainer._param_groups()
