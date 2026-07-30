from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sana_wam.dataloader.robotwin_plan_binding import phase6_plan_row_dict
from sana_wam.dataloader.task_sample_plan import PlannedSample, SampleIdentity
from sana_wam.train.trainer import Trainer


def _row() -> PlannedSample:
    domains = (
        "video-noise",
        "action-noise",
        "expansion-noise",
        "expansion-direction",
        "reference-query",
        "prompt-choice",
    )
    return PlannedSample(
        global_step=1,
        cycle=0,
        position_in_cycle=0,
        identity=SampleIdentity(
            task_name="adjust_bottle",
            episode_index=0,
            episode_path="/data/episode0.hdf5",
            start_frame=3,
            prompt="exact prompt",
            dataset_index=17,
        ),
        action_sigma=0.9,
        domain_seeds=tuple(
            (domain, index + 10) for index, domain in enumerate(domains)
        ),
    )


class _Plan:
    plan_sha256 = "a" * 64
    identity_sha256 = "b" * 64
    rows = (_row(),)


class _Architecture:
    def __init__(self):
        self.seen = None
        self.history = []

    def prepare_inputs(self, batch):
        return {"phase6_plan_rows": (batch[0]["phase6_plan_row"],)}

    def compute_loss(self, **kwargs):
        self.seen = kwargs
        self.history.append(kwargs)
        return {
            "loss": torch.tensor(0.0),
            "phase6_common_input_trace_sha256": "c" * 64,
        }


def _trainer_with_plan() -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer._phase6_plan = _Plan()
    trainer.architecture = _Architecture()
    trainer.lambda_video = 1.0
    trainer.lambda_action = 0.0
    trainer.device = torch.device("cpu")
    return trainer


def test_phase6_loss_checks_raw_and_prepared_row_before_model_call():
    trainer = _trainer_with_plan()
    payload = phase6_plan_row_dict(
        _Plan.rows[0],
        plan_sha256=_Plan.plan_sha256,
        identity_sha256=_Plan.identity_sha256,
    )
    result = trainer._compute_loss(
        [{"phase6_plan_row": payload}], expected_phase6_step=1
    )
    assert result["loss"].item() == 0.0
    assert trainer.architecture.seen["phase6_plan_rows"] == (payload,)
    assert trainer.architecture.seen["phase6_common_input_trace_required"] is True
    assert "phase6_t0_reference_forward_trace_required" not in trainer.architecture.seen
    assert "phase1_action_reference_error" not in trainer.architecture.seen

    changed = {**payload, "global_step": 2}
    with pytest.raises(RuntimeError, match="sample order drift"):
        trainer._compute_loss([{"phase6_plan_row": changed}], expected_phase6_step=1)


def test_a1_injects_one_fresh_detached_fp32_scalar_and_checks_common_trace():
    class _ReferenceTable:
        common_input_trace_sha256 = ("c" * 64,)

        @staticmethod
        def error_for_global_step(global_step):
            assert global_step == 1
            return 0.125

    trainer = _trainer_with_plan()
    trainer._phase1_action_reference_table = _ReferenceTable()
    trainer._phase1_action_reference_enabled_config = True
    payload = phase6_plan_row_dict(
        _Plan.rows[0],
        plan_sha256=_Plan.plan_sha256,
        identity_sha256=_Plan.identity_sha256,
    )

    trainer._compute_loss([{"phase6_plan_row": payload}], expected_phase6_step=1)
    trainer._compute_loss([{"phase6_plan_row": payload}], expected_phase6_step=1)

    first = trainer.architecture.history[0]["phase1_action_reference_error"]
    second = trainer.architecture.history[1]["phase1_action_reference_error"]
    assert first.dtype == second.dtype == torch.float32
    assert first.device == second.device == torch.device("cpu")
    assert first.shape == second.shape == torch.Size([])
    assert first.item() == second.item() == 0.125
    assert first.requires_grad is second.requires_grad is False
    assert first.grad_fn is second.grad_fn is None
    assert first is not second
    assert trainer.architecture.history[0]["phase6_common_input_trace_required"] is True


def test_a1_rejects_common_trace_drift_before_returning_loss():
    class _ReferenceTable:
        common_input_trace_sha256 = ("d" * 64,)

        @staticmethod
        def error_for_global_step(global_step):  # noqa: ARG004
            return 0.125

    trainer = _trainer_with_plan()
    trainer._phase1_action_reference_table = _ReferenceTable()
    trainer._phase1_action_reference_enabled_config = True
    payload = phase6_plan_row_dict(
        _Plan.rows[0],
        plan_sha256=_Plan.plan_sha256,
        identity_sha256=_Plan.identity_sha256,
    )

    with pytest.raises(RuntimeError, match="common input/noise trace differs"):
        trainer._compute_loss([{"phase6_plan_row": payload}], expected_phase6_step=1)


def test_phase6_plan_configuration_is_all_or_none_and_fixed_504_contract():
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(dataloader={"type": "robotwin"})
    trainer.dataset = object()
    trainer.t = {"phase6_registry": "/only/one/field"}
    with pytest.raises(ValueError, match="set every"):
        trainer._configure_phase6_plan()

    trainer.t = {
        "phase6_registry": "/registry",
        "phase6_registry_sha256": "a" * 64,
        "phase6_plan_artifact": "/plan",
        "phase6_plan_artifact_sha256": "b" * 64,
        "phase6_plan_sha256": "c" * 64,
        "phase6_identity_sha256": "d" * 64,
        "phase6_dataset_contract_artifact": "/dataset-contract",
        "phase6_dataset_contract_artifact_sha256": "e" * 64,
        "expected_world_size": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_steps": 504,
        "lr_schedule_steps": 504,
    }
    with pytest.raises(ValueError, match="expected_world_size=1"):
        trainer._configure_phase6_plan()


def test_legacy_configuration_does_not_touch_dataset_or_sampler():
    trainer = Trainer.__new__(Trainer)
    dataset = object()
    trainer.dataset = dataset
    trainer.t = {}
    trainer._phase6_plan = None
    trainer._phase6_plan_sampler = None
    trainer._configure_phase6_plan()
    assert trainer.dataset is dataset
    assert trainer._phase6_plan is None
    assert trainer._phase6_plan_sampler is None
