"""Fail-closed contracts for action-only warm-start training."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from sana_wam.train import trainer as trainer_module
from sana_wam.train.libero_contract import (
    LIBERO_R11_DIT_TRUNK_ADAPT_EVAL_MODULES,
    LIBERO_R11_DIT_TRUNK_ADAPT_TRAINABLE_PARAMETER_PATTERNS,
)


class _DummyDataset(torch.utils.data.Dataset):
    def __init__(self, stats_path: Path):
        self.action_stats_path = str(stats_path)
        self.action_stats = {
            "mean": np.zeros(2, dtype=np.float32),
            "std": np.ones(2, dtype=np.float32),
        }
        self.num_frames = 5
        self.epochs = []

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"index": index}

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class _DummyArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_backbone = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        self.action_backbone = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        self.proprio_encoder = nn.Linear(2, 2)
        self.root_scale = nn.Parameter(torch.ones(()))
        self.register_buffer("action_mean", torch.zeros(2))
        self.register_buffer("action_std", torch.ones(2))
        self.events = []
        self.load_calls = []
        self.freeze_calls = []

    def set_dtype_device(self, dtype, device):
        self.events.append("set_dtype_device")
        self.to(device=device)

    def load_checkpoint(self, path, strict=True):
        self.events.append("load_checkpoint")
        self.load_calls.append((path, strict))

    def init_training_schedulers(self, num_timesteps):
        self.events.append(("init_training_schedulers", num_timesteps))

    def set_training_runtime(self, **kwargs):
        self.events.append(("set_training_runtime", kwargs))

    def freeze_modules(self, names, *, preserve_input_grad=False):
        self.freeze_calls.append((tuple(names), preserve_input_grad))
        frozen = []
        for name in names:
            try:
                module = self.get_submodule(name)
            except (AttributeError, KeyError):
                continue
            module.requires_grad_(False)
            for child in module.modules():
                child.training = False
            frozen.append(name)
        return frozen

    def get_trainable_modules(self, freeze_list=()):
        excluded = set(freeze_list)
        return {
            name: module
            for name, module in self.named_children()
            if name not in excluded
            and any(parameter.requires_grad for parameter in module.parameters())
        }


class _DummyR11CaptionEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.register_buffer("y_embedding", nn.Parameter(torch.ones(2, 2)))


class _DummyR11Dit(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_y_norm = nn.LayerNorm(2)
        self.blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.t_block = nn.Sequential(nn.Linear(2, 2))
        self.t_embedder = nn.Linear(2, 2)
        self.x_embedder = nn.Linear(2, 2)
        self.y_embedder = _DummyR11CaptionEmbedder()
        self.final_layer = nn.Linear(2, 2)
        self.register_buffer("pos_embed", torch.zeros(1, 2, 2))


class _DummyR11VideoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = _DummyR11Dit()
        self.vae = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)


class _DummyR11Architecture(_DummyArchitecture):
    def __init__(self):
        super().__init__()
        self.video_backbone = _DummyR11VideoBackbone()
        self.proprio_video_embed = nn.Linear(2, 2)
        self.proprio_action_embed = nn.Linear(2, 2)


def _config(**training_overrides):
    training = {
        "max_steps": 1,
        "save_steps": 0,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_workers": 0,
        "action_lr": 1.0e-4,
        "video_lr": 1.0e-5,
        "weight_decay": 0.0,
        "lambda_video": 0.0,
        "lambda_action": 1.0,
        "seed": 7,
        "freeze": [],
    }
    training.update(training_overrides)
    return OmegaConf.create(
        {
            "model": {
                "architecture": {
                    "framework": "dual_system",
                    "variant": "autoregressive",
                },
                "action_backbone": {},
                "video_backbone": {},
            },
            "dataloader": {"type": "robotwin", "num_frames": 5},
            "training": training,
        }
    )


def _make_trainer(tmp_path, monkeypatch, cfg=None, architecture=None):
    stats_path = tmp_path / "action_stats.npy"
    stats_path.write_bytes(b"frozen base stats")
    dataset = _DummyDataset(stats_path)
    architecture = architecture or _DummyArchitecture()
    cfg = cfg or _config()

    monkeypatch.setattr(
        trainer_module, "build_training_dataset", lambda config, split: dataset
    )
    monkeypatch.setattr(
        trainer_module, "build_architecture", lambda config: architecture
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return trainer_module.Trainer(cfg), dataset, architecture


def test_init_checkpoint_sha_mismatch_fails_before_load(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"checkpoint payload")
    cfg = _config(
        init_checkpoint=str(checkpoint),
        init_checkpoint_sha256="0" * 64,
    )
    architecture = _DummyArchitecture()

    with pytest.raises(ValueError, match="init_checkpoint SHA-256 mismatch"):
        _make_trainer(tmp_path, monkeypatch, cfg, architecture)
    assert architecture.load_calls == []


def test_init_checkpoint_strict_load_precedes_fresh_scheduler(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"checkpoint payload")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg = _config(
        init_checkpoint=str(checkpoint),
        init_checkpoint_sha256=digest,
    )

    _, _, architecture = _make_trainer(tmp_path, monkeypatch, cfg)

    assert architecture.load_calls == [(str(checkpoint.resolve()), True)]
    assert architecture.events.index("load_checkpoint") < architecture.events.index(
        ("init_training_schedulers", 1000)
    )


def test_action_stats_sha_mismatch_fails_closed(tmp_path, monkeypatch):
    cfg = _config(action_stats_sha256="f" * 64)
    with pytest.raises(ValueError, match="action_stats SHA-256 mismatch"):
        _make_trainer(tmp_path, monkeypatch, cfg)


def test_exact_allowlist_freezes_every_other_parameter_and_optimizer_group(
    tmp_path, monkeypatch
):
    cfg = _config(trainable_modules=["action_backbone"])
    trainer, _, architecture = _make_trainer(tmp_path, monkeypatch, cfg)

    trainable = {
        name
        for name, parameter in architecture.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("action_backbone.") for name in trainable)
    assert not architecture.root_scale.requires_grad

    optimizer_parameters = {
        id(parameter)
        for group in trainer._param_groups()
        for parameter in group["params"]
    }
    expected_parameters = {
        id(parameter) for parameter in architecture.action_backbone.parameters()
    }
    assert optimizer_parameters == expected_parameters
    assert all(
        parameter.requires_grad
        for group in trainer._param_groups()
        for parameter in group["params"]
    )


def test_training_mode_restores_frozen_subtrees_to_eval(tmp_path, monkeypatch):
    cfg = _config(trainable_modules=["action_backbone"])
    trainer, _, architecture = _make_trainer(tmp_path, monkeypatch, cfg)

    trainer._set_training_mode()

    assert architecture.training
    assert architecture.action_backbone.training
    assert all(module.training for module in architecture.action_backbone.modules())
    assert all(not module.training for module in architecture.video_backbone.modules())
    assert all(not module.training for module in architecture.proprio_encoder.modules())


def test_allowlist_can_preserve_inputs_gradients_through_one_frozen_root(
    tmp_path, monkeypatch
):
    cfg = _config(
        trainable_modules=["action_backbone"],
        preserve_frozen_input_grad_modules=["video_backbone"],
    )
    trainer, _, architecture = _make_trainer(tmp_path, monkeypatch, cfg)

    assert trainer._frozen_input_grad_module_paths == ("video_backbone",)
    assert (("video_backbone",), True) in architecture.freeze_calls
    assert (("proprio_encoder",), False) in architecture.freeze_calls
    assert all(
        not parameter.requires_grad
        for parameter in architecture.video_backbone.parameters()
    )
    trainer._set_training_mode()
    assert all(
        not module.training for module in architecture.video_backbone.modules()
    )


def test_parameter_patterns_select_only_matching_tensors(tmp_path, monkeypatch):
    cfg = _config(
        trainable_parameter_patterns=["video_backbone.0.weight"],
        eval_modules=["video_backbone.1"],
    )
    trainer, _, architecture = _make_trainer(tmp_path, monkeypatch, cfg)

    trainable = [
        name
        for name, parameter in architecture.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable == ["video_backbone.0.weight"]
    optimizer_parameters = [
        parameter
        for group in trainer._param_groups()
        for parameter in group["params"]
    ]
    assert len(optimizer_parameters) == 1
    assert optimizer_parameters[0] is architecture.video_backbone[0].weight

    trainer._set_training_mode()
    assert architecture.video_backbone.training
    assert not architecture.video_backbone[1].training


def test_r11_patterns_train_only_action_reachable_dit_in_video_eval_mode(
    tmp_path, monkeypatch
):
    architecture = _DummyR11Architecture()
    cfg = _config(
        trainable_parameter_patterns=list(
            LIBERO_R11_DIT_TRUNK_ADAPT_TRAINABLE_PARAMETER_PATTERNS
        ),
        eval_modules=list(LIBERO_R11_DIT_TRUNK_ADAPT_EVAL_MODULES),
    )
    trainer, _, architecture = _make_trainer(
        tmp_path, monkeypatch, cfg, architecture
    )

    trainable_names = {
        name
        for name, parameter in architecture.named_parameters()
        if parameter.requires_grad
    }
    expected_names = {
        name
        for name, _ in architecture.named_parameters()
        if name.startswith(
            (
                "action_backbone.",
                "proprio_encoder.",
                "proprio_video_embed.",
                "proprio_action_embed.",
            )
        )
        or (
            name.startswith("video_backbone.dit.")
            and not name.startswith("video_backbone.dit.final_layer.")
        )
    }
    assert trainable_names == expected_names
    assert not any("final_layer" in name for name in trainable_names)
    assert not any(".vae." in name for name in trainable_names)
    assert not any(".text_encoder." in name for name in trainable_names)

    optimizer_ids = {
        id(parameter)
        for group in trainer._param_groups()
        for parameter in group["params"]
    }
    assert optimizer_ids == {
        id(parameter)
        for name, parameter in architecture.named_parameters()
        if name in expected_names
    }
    assert id(architecture.video_backbone.dit.y_embedder.y_embedding) not in (
        optimizer_ids
    )
    assert id(architecture.video_backbone.dit.pos_embed) not in optimizer_ids

    trainer._set_training_mode()
    assert all(
        not module.training for module in architecture.video_backbone.modules()
    )
    assert all(
        not parameter.requires_grad
        for parameter in architecture.video_backbone.vae.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in architecture.video_backbone.text_encoder.parameters()
    )


def test_parameter_pattern_warm_start_can_keep_fp32_optimizer_state(
    tmp_path, monkeypatch
):
    architecture = _DummyArchitecture().to(dtype=torch.bfloat16)
    original_weight = architecture.video_backbone[0].weight
    cfg = _config(
        trainable_parameter_patterns=["video_backbone.0.weight"],
        trainable_parameter_dtype="float32",
    )

    trainer, _, architecture = _make_trainer(
        tmp_path, monkeypatch, cfg, architecture
    )

    assert architecture.video_backbone[0].weight is original_weight
    assert architecture.video_backbone[0].weight.dtype == torch.float32
    assert architecture.video_backbone[0].bias.dtype == torch.bfloat16
    assert trainer._trainable_params() == [architecture.video_backbone[0].weight]


def test_fp32_optimizer_master_weights_preserve_bf16_model_storage(
    tmp_path, monkeypatch
):
    architecture = _DummyArchitecture().to(dtype=torch.bfloat16)
    cfg = _config(
        trainable_parameter_patterns=["video_backbone.0.weight"],
        optimizer_master_weights=True,
    )
    trainer, _, architecture = _make_trainer(
        tmp_path, monkeypatch, cfg, architecture
    )
    model_parameter = architecture.video_backbone[0].weight

    optimizer_groups, pairs = trainer._optimizer_param_groups(
        trainer._param_groups()
    )

    assert model_parameter.dtype == torch.bfloat16
    assert len(pairs) == 1
    assert pairs[0][0] is model_parameter
    master = pairs[0][1]
    assert optimizer_groups[0]["params"] == [master]
    assert master.dtype == torch.float32
    torch.testing.assert_close(master, model_parameter.float())

    model_parameter.grad = torch.full_like(model_parameter, 0.25)
    trainer._sync_master_gradients(pairs)
    assert master.grad is not None
    assert master.grad.dtype == torch.float32
    torch.testing.assert_close(
        master.grad, torch.full_like(master, 0.25)
    )

    with torch.no_grad():
        master.add_(0.125)
    trainer._copy_master_parameters_to_model(pairs)
    torch.testing.assert_close(model_parameter.float(), master.to(torch.bfloat16).float())


def test_fp32_optimizer_master_accumulates_sub_bf16_adamw_updates(
    tmp_path, monkeypatch
):
    architecture = _DummyArchitecture().to(dtype=torch.bfloat16)
    with torch.no_grad():
        architecture.proprio_encoder.weight.fill_(1.0)
    cfg = _config(
        trainable_parameter_patterns=["proprio_encoder.weight"],
        optimizer_master_weights=True,
        video_lr=1.0e-4,
    )
    trainer, _, architecture = _make_trainer(
        tmp_path, monkeypatch, cfg, architecture
    )
    model_parameter = architecture.proprio_encoder.weight
    optimizer_groups, pairs = trainer._optimizer_param_groups(
        trainer._param_groups()
    )
    master = pairs[0][1]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    initial_model = model_parameter.detach().clone()
    initial_master = master.detach().clone()
    first_master_delta = None

    for step in range(1, 33):
        model_parameter.grad = torch.full_like(model_parameter, 0.25)
        trainer._sync_master_gradients(pairs)
        optimizer.step()
        trainer._copy_master_parameters_to_model(pairs)
        assert torch.equal(
            model_parameter,
            master.detach().to(dtype=torch.bfloat16),
        )
        optimizer.zero_grad(set_to_none=True)

        master_delta = float((master.detach() - initial_master).abs().max())
        if step == 1:
            first_master_delta = master_delta
            assert first_master_delta > 0.0
            assert torch.equal(model_parameter, initial_model)
        elif step == 8:
            assert first_master_delta is not None
            assert master_delta > first_master_delta * 7.0
            assert torch.equal(model_parameter, initial_model)

    assert not torch.equal(master.detach(), initial_master)
    assert not torch.equal(model_parameter, initial_model)


def test_optimizer_master_weights_requires_boolean(tmp_path, monkeypatch):
    cfg = _config(optimizer_master_weights="yes")
    trainer, _, _ = _make_trainer(tmp_path, monkeypatch, cfg)
    with pytest.raises(ValueError, match="must be a boolean"):
        trainer._optimizer_param_groups(trainer._param_groups())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"trainable_parameter_dtype": "float32"},
            "requires training.trainable_parameter_patterns",
        ),
        (
            {
                "trainable_parameter_patterns": ["video_backbone.0.weight"],
                "trainable_parameter_dtype": "float64",
            },
            "supports only float32",
        ),
    ],
)
def test_invalid_trainable_parameter_dtype_fails_closed(
    tmp_path, monkeypatch, overrides, message
):
    with pytest.raises(ValueError, match=message):
        _make_trainer(tmp_path, monkeypatch, _config(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"trainable_parameter_patterns": []}, "must not be empty"),
        (
            {"trainable_parameter_patterns": ["missing.*"]},
            "matched no parameters",
        ),
        (
            {
                "trainable_parameter_patterns": ["video_backbone.*"],
                "trainable_modules": ["video_backbone"],
            },
            "cannot be combined",
        ),
        ({"eval_modules": ["missing"]}, "unknown training.eval_modules path"),
    ],
)
def test_invalid_parameter_pattern_contracts_fail_closed(
    tmp_path, monkeypatch, overrides, message
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        _make_trainer(tmp_path, monkeypatch, _config(**overrides))


@pytest.mark.parametrize(
    ("allowlist", "message"),
    [
        ([], "must not be empty"),
        (["video_backbone.dit"], "top-level module names"),
        (["missing"], "unknown training.trainable_modules"),
    ],
)
def test_invalid_trainable_allowlists_fail_closed(
    tmp_path, monkeypatch, allowlist, message
):
    cfg = _config(trainable_modules=allowlist)
    with pytest.raises((ValueError, RuntimeError), match=message):
        _make_trainer(tmp_path, monkeypatch, cfg)


def test_allowlist_rejects_competing_legacy_freeze(tmp_path, monkeypatch):
    cfg = _config(
        trainable_modules=["action_backbone"],
        freeze=["video_backbone"],
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        _make_trainer(tmp_path, monkeypatch, cfg)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"preserve_frozen_input_grad_modules": ["video_backbone"]},
            "requires training.trainable_modules",
        ),
        (
            {
                "trainable_modules": ["action_backbone"],
                "preserve_frozen_input_grad_modules": ["missing"],
            },
            "unknown training.preserve_frozen_input_grad_modules",
        ),
        (
            {
                "trainable_modules": ["action_backbone"],
                "preserve_frozen_input_grad_modules": [
                    "video_backbone",
                    "video_backbone",
                ],
            },
            "duplicate training.preserve_frozen_input_grad_modules",
        ),
        (
            {
                "trainable_modules": ["action_backbone"],
                "preserve_frozen_input_grad_modules": ["action_backbone"],
            },
            "must name a frozen module",
        ),
        (
            {
                "trainable_modules": ["action_backbone"],
                "preserve_frozen_input_grad_modules": ["video_backbone.0"],
            },
            "exact top-level module names",
        ),
    ],
)
def test_invalid_frozen_input_gradient_contracts_fail_closed(
    tmp_path, monkeypatch, overrides, message
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        _make_trainer(tmp_path, monkeypatch, _config(**overrides))


def test_save_initial_checkpoint_helper_writes_step_zero(tmp_path, monkeypatch):
    cfg = _config(
        save_initial_checkpoint=True,
        save_at_steps=[100, 500, 1000],
    )
    trainer, _, _ = _make_trainer(tmp_path, monkeypatch, cfg)
    calls = []
    monkeypatch.setattr(
        trainer, "_save", lambda output_path, step: calls.append((output_path, step))
    )

    assert trainer._save_initial_checkpoint("/tmp/output")
    assert calls == [("/tmp/output", 0)]
    assert trainer._explicit_save_steps() == {100, 500, 1000}
    assert trainer._checkpoint_due(100, 100, {100})
    assert not trainer._checkpoint_due(101, 100, {100})


def test_data_epoch_updates_dataset_and_sampler(tmp_path, monkeypatch):
    trainer, dataset, _ = _make_trainer(tmp_path, monkeypatch)

    class _Sampler:
        epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    sampler = _Sampler()
    trainer._set_data_epoch(3, sampler)

    assert dataset.epochs == [3]
    assert sampler.epochs == [3]


def test_short_run_keeps_full_lr_schedule_horizon(tmp_path, monkeypatch):
    cfg = _config(max_steps=100, lr_schedule_steps=1000)
    trainer, _, _ = _make_trainer(tmp_path, monkeypatch, cfg)

    schedule_steps = trainer._resolve_lr_schedule_steps(max_steps=100)

    assert schedule_steps == 1000
    assert trainer._lr_lambda(99, schedule_steps) > 0.9


def test_lr_schedule_horizon_cannot_end_before_training(tmp_path, monkeypatch):
    cfg = _config(max_steps=100, lr_schedule_steps=99)
    trainer, _, _ = _make_trainer(tmp_path, monkeypatch, cfg)

    with pytest.raises(ValueError, match="must be >= training.max_steps"):
        trainer._resolve_lr_schedule_steps(max_steps=100)


def test_process_seed_offsets_all_rngs_by_rank(monkeypatch):
    calls = {}
    monkeypatch.setattr(random, "seed", lambda value: calls.setdefault("random", value))
    monkeypatch.setattr(
        np.random, "seed", lambda value: calls.setdefault("numpy", value)
    )
    monkeypatch.setattr(
        torch, "manual_seed", lambda value: calls.setdefault("torch", value)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda value: calls.setdefault("torch_cuda", value),
    )

    assert trainer_module.Trainer._seed_process(41, 3) == 44
    assert calls == {"random": 44, "numpy": 44, "torch": 44, "torch_cuda": 44}


def test_expected_world_size_is_verified_and_persisted(tmp_path, monkeypatch):
    cfg = _config(expected_world_size=4)
    monkeypatch.setattr(trainer_module, "_world_size", lambda: 4)

    trainer, _, _ = _make_trainer(tmp_path, monkeypatch, cfg)

    assert trainer.t.expected_world_size == 4
    assert trainer.t.actual_world_size == 4


def test_expected_world_size_mismatch_fails_before_dataset_build(tmp_path, monkeypatch):
    cfg = _config(expected_world_size=4)
    monkeypatch.setattr(trainer_module, "_world_size", lambda: 1)
    called = False

    def build_dataset(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("dataset build must not run")

    monkeypatch.setattr(trainer_module, "build_training_dataset", build_dataset)
    with pytest.raises(RuntimeError, match="expected 4, got 1"):
        trainer_module.Trainer(cfg)
    assert called is False
