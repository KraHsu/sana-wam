from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT / "src"))


class _FakeTensor:
    ndim = 0


class _FiniteGradient:
    def all(self):
        return self

    def item(self):
        return True


def _install_trainer_import_stubs(monkeypatch: pytest.MonkeyPatch):
    torch = ModuleType("torch")
    torch.__path__ = []
    torch.Tensor = _FakeTensor
    torch.no_grad = lambda: lambda function: function
    torch.isfinite = lambda value: value
    torch.autograd = SimpleNamespace(grad=lambda *_args, **_kwargs: [])
    distributed = ModuleType("torch.distributed")
    distributed.is_available = lambda: False
    distributed.is_initialized = lambda: False
    data = ModuleType("torch.utils.data")
    data.DataLoader = object
    data.DistributedSampler = object
    utils = ModuleType("torch.utils")
    utils.__path__ = []
    utils.data = data
    torch.distributed = distributed
    torch.utils = utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data)

    modules = {
        "sana_wam.config": {"flatten_model_cfg": lambda value: value},
        "sana_wam.dataloader.mixture": {
            "build_training_dataset": lambda *_args, **_kwargs: None
        },
        "sana_wam.model": {"build_architecture": lambda *_args, **_kwargs: None},
        "sana_wam.train.checkpointing": {
            "manage_checkpoints": lambda *_args, **_kwargs: None,
            "save_action_stats": lambda *_args, **_kwargs: None,
            "save_config": lambda *_args, **_kwargs: None,
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("sana_wam.train.trainer", None)
    from sana_wam.train.trainer import Trainer

    return Trainer, torch


@pytest.mark.parametrize(
    ("key", "value", "remove"),
    (
        ("historical_checkpoint_count", None, True),
        ("historical_checkpoint_count", False, False),
        ("historical_scheduled_lr_scale_at_step0", 0, False),
        ("recovery_closed_loop_started", True, False),
    ),
)
def test_training_launch_context_rejects_incomplete_or_aliased_recovery_lifecycle(
    monkeypatch: pytest.MonkeyPatch, key: str, value: object, remove: bool
):
    from sana_wam.train.phase6_recovery import recovery_lifecycle

    Trainer, _torch = _install_trainer_import_stubs(monkeypatch)
    authorization = {"operation": "synthetic-training"}
    report = {
        "authorization": authorization,
        **recovery_lifecycle(),
        "reference_precompute_completed": True,
        "registered_before_recovery_cohort_started": True,
        "smoke_completed": True,
    }
    if remove:
        report.pop(key)
    else:
        report[key] = value
    trainer = Trainer.__new__(Trainer)
    trainer._phase6_launch_context = SimpleNamespace(
        authorization=authorization,
        revalidate_pinned_files=lambda: None,
    )
    trainer._phase6_preflight_report = report
    with pytest.raises(RuntimeError, match=key):
        trainer._validate_phase6_launch_context_early()


def test_actual_nr_surrogate_probe_requires_adapter_only_connectivity(
    monkeypatch: pytest.MonkeyPatch,
):
    Trainer, torch = _install_trainer_import_stubs(monkeypatch)
    trainer = Trainer.__new__(Trainer)
    trainer._phase6_launch_context = SimpleNamespace(factor_mapping={"A": True})
    adapter_names = [
        f"action_video_memory_adapter.{name}"
        for name in ("k_left", "k_right", "v_left", "v_right", "z_logits")
    ]
    parameters = [object() for _ in range(6)]
    named = list(zip([*adapter_names, "video_backbone.dit.weight"], parameters))
    torch.autograd.grad = lambda *_args, **_kwargs: [
        *[_FiniteGradient() for _ in adapter_names],
        None,
    ]
    result = {"phase6_action_non_regression_surrogate": _FakeTensor()}
    assert trainer._phase6_probe_nr_autograd(result, named) == (True, True)

    torch.autograd.grad = lambda *_args, **_kwargs: [
        *[_FiniteGradient() for _ in adapter_names],
        _FiniteGradient(),
    ]
    with pytest.raises(RuntimeError, match="not connected adapter-only"):
        trainer._phase6_probe_nr_autograd(result, named)


def test_a0_skips_nr_probe_and_measurements_keep_real_student_error(
    monkeypatch: pytest.MonkeyPatch,
):
    Trainer, _torch = _install_trainer_import_stubs(monkeypatch)
    trainer = Trainer.__new__(Trainer)
    trainer._phase6_launch_context = SimpleNamespace(
        factor_mapping={"A": False, "E": False, "T": True}
    )
    assert trainer._phase6_probe_nr_autograd({}, []) == (None, None)
    gradients = {
        "action_trainable_param_count": 0,
        "adapter_gradient_connected": None,
        "adapter_grad_norm": None,
        "finite": True,
        "global_norm": 1.0,
        "model_parameters_finite": True,
        "nr_gradient_target_adapter_only": None,
        "optimizer_master_parameters_finite": True,
        "optimizer_state_finite": True,
        "proprio_trainable_param_count": 0,
        "video_grad_norm": 1.0,
    }
    result = {
        "loss": 2.5,
        "loss_video_on_path": 2.5,
        "phase6_action_unweighted_mse": 0.75,
        "phase6_common_input_trace_sha256": "c" * 64,
    }
    measurements = trainer._phase6_measurements(
        result,
        gradients,
        {"action_adapter": None, "video": 5.0e-6},
        peak_reserved_bytes=123,
    )
    assert measurements["action"] == {
        "active": False,
        "non_regression_loss": 0.0,
        "reference_error": None,
        "relative_excess": None,
        "student_error": 0.75,
    }
    assert measurements["expansion"]["rate"] is None


def test_deploy_and_resolved_configs_are_byte_equivalent_and_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    Trainer, _torch = _install_trainer_import_stubs(monkeypatch)
    deploy = tmp_path / "config.yaml"
    resolved = tmp_path / "resolved_config.yaml"
    deploy.write_bytes(b"exact saved config\n")
    Trainer._phase6_link_or_copy_config(str(deploy), str(resolved))
    assert resolved.read_bytes() == deploy.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        Trainer._phase6_link_or_copy_config(str(deploy), str(resolved))
