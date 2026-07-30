from __future__ import annotations

import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

from omegaconf import OmegaConf
import pytest

from sana_wam.train.phase6_arm_config import VIDEO_PATTERNS
from sana_wam.train.trainer import Trainer


SHA = "a" * 64


def _launcher_module():
    path = Path(__file__).resolve().parents[1] / "scripts/train_post_primary_afcc.py"
    spec = importlib.util.spec_from_file_location("train_post_primary_afcc", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Store:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.rows = tuple(range(504))
        self.manifest_sha256 = kwargs["manifest_sha256"]


def _trainer(monkeypatch, *, weight: float = 1.0):
    cfg = OmegaConf.create(
        {
            "model": {
                "architecture": {
                    "action_facing_cache_consistency_weight": weight,
                    "action_non_regression_weight": 0.0,
                    "action_video_memory_adapter": {"enabled": False},
                    "video_local_expansion_weight": 1.0,
                },
                "video_backbone": {"continuous_timestep_conditioning": True},
            },
            "training": {
                "batch_size": 1,
                "phase1_action_facing_cache_reference_manifest": "/tmp/manifest.json",
                "phase1_action_facing_cache_reference_manifest_sha256": SHA,
                "phase1_reference_checkpoint_sha256": SHA,
                "phase6_code_source_manifest_sha256": SHA,
                "trainable_parameter_patterns": list(VIDEO_PATTERNS),
            },
        }
    )
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.t = cfg.training
    trainer.lambda_action = 0.0
    trainer._phase6_plan = SimpleNamespace(
        plan_sha256=SHA,
        identity_sha256=SHA,
    )
    trainer._phase6_dataset_contract = SimpleNamespace(artifact_sha256=SHA)
    trainer._phase1_action_reference_table = None
    trainer._verified_phase1_reference_checkpoint_path = "/tmp/checkpoint"
    trainer._verified_action_stats_sha256 = SHA
    trainer._verify_file_sha256 = MethodType(
        lambda self, path, expected, field_name: path,
        trainer,
    )
    monkeypatch.setattr(
        "sana_wam.train.action_facing_cache_reference.ActionFacingCacheReferenceStore",
        _Store,
    )
    return trainer


def test_afcc_trainer_accepts_omegaconf_listconfig_allowlist(monkeypatch):
    trainer = _trainer(monkeypatch)
    assert type(trainer.t.trainable_parameter_patterns).__name__ == "ListConfig"
    trainer._configure_action_facing_cache_reference()
    assert trainer._action_facing_cache_consistency_enabled_config is True
    assert isinstance(trainer._action_facing_cache_reference_store, _Store)
    assert trainer._action_facing_cache_reference_store.kwargs[
        "expected_source_manifest_sha256"
    ] == SHA


def test_afcc_trainer_forbids_reference_fields_when_disabled(monkeypatch):
    trainer = _trainer(monkeypatch, weight=0.0)
    with pytest.raises(ValueError, match="forbidden when F=0"):
        trainer._configure_action_facing_cache_reference()


def test_afcc_launcher_closes_canonical_metrics_against_reference(tmp_path):
    launcher = _launcher_module()
    reference_rows = []
    metric_rows = []
    for step in range(1, 505):
        sigma = (1.0, 0.9, 0.5)[(step - 1) % 3]
        reference_rows.append(
            {
                "action_sigma": sigma,
                "common_input_trace_sha256": f"{step:064x}",
                "plan_row_sha256": f"{step + 504:064x}",
                "task_name": f"task-{(step - 1) % 42}",
            }
        )
        metric_rows.append(
            {
                "action_sigma": sigma,
                "afcc_loss": 0.25,
                "common_input_trace_sha256": f"{step:064x}",
                "global_step": step,
                "gradient_global_norm": 0.5,
                "learning_rate": 0.0 if step == 1 else 1.0e-6,
                "peak_reserved_bytes": 1,
                "plan_row_sha256": f"{step + 504:064x}",
                "schema_version": "sana-phase6-afcc-step-metrics-v1",
                "task_name": f"task-{(step - 1) % 42}",
                "total_loss": 1.0,
                "video_local_expansion_loss": 0.5,
                "video_on_path_loss": 0.5,
            }
        )
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_bytes(b"".join(launcher._canonical(row) for row in metric_rows))
    observed = launcher._strict_json_lines(metrics)
    launcher._validate_metric_rows(observed, reference_rows)

    reference_rows[503]["plan_row_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="reference binding differs"):
        launcher._validate_metric_rows(observed, reference_rows)
