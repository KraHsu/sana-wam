from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/experiments/libero_formal_single_gpu_2000.yaml"
RUNNER = ROOT / "scripts/train_libero_formal2000.py"
TRAINER = ROOT / "src/sana_wam/train/trainer.py"
RUN_ROOT = (
    "/DATA/share/sana_wam_libero_training/formal2000/"
    "libero-ar-formal2000-single-gpu-2000-dba67fee2b53ecce092898c0a70144c4"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("formal2000_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal2000_config_is_single_gpu_fresh_init_final_only() -> None:
    cfg = OmegaConf.load(CONFIG)
    expected = {
        "training.output_dir": RUN_ROOT,
        "training.max_steps": 2000,
        "training.save_steps": 0,
        "training.save_initial_checkpoint": False,
        "training.keep_last_k": 1,
        "training.batch_size": 1,
        "training.gradient_accumulation_steps": 8,
        "training.expected_world_size": 1,
        "training.exact_output_dir": True,
        "training.formal_libero_training": True,
        "training.formal_non_resumable": True,
        "training.optimizer_master_weights": True,
        "training.seed": 20260807,
        "dataloader.seed": 20260806,
    }
    for path, value in expected.items():
        assert OmegaConf.select(cfg, path) == value
    assert list(cfg.training.save_at_steps) == []
    for key in (
        "init_checkpoint",
        "init_checkpoint_sha256",
        "resume_checkpoint",
        "resume_from_checkpoint",
    ):
        assert OmegaConf.select(cfg, f"training.{key}", default=None) is None


def test_runner_binds_one_root_t16_and_no_resume() -> None:
    source = RUNNER.read_text()
    ast.parse(source)
    assert source.count("dba67fee2b53ecce092898c0a70144c4") >= 2
    assert "5e2f5d61f7715f952f1762c04f5ad99891f9eda1ae5256310498c23146bcc471" in source
    assert "T16_PAIRED_ONE_TASK_CHUNK_CLOSED_LOOP_INTERFACE_VALID" in source
    assert "Trainer(cfg, exact_output_dir=str(RUN_ROOT))" in source
    assert "checkpoint_step_2000.safetensors" in source
    assert "os.O_EXCL" in source
    assert "FAILED_CLOSED_NON_RESUMABLE" in source
    assert "torch.load(" not in source


def test_runner_canonical_json_is_sorted_compact_and_lf() -> None:
    module = _load_runner_module()
    assert module._canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n'


def test_trainer_exact_output_seam_is_not_legacy_timestamp() -> None:
    source = TRAINER.read_text()
    ast.parse(source)
    assert "elif formal_libero:" in source
    assert (
        "output_path = self._validate_exact_output_dir(self._exact_output_dir)"
        in source
    )
    assert "formal_libero = self._exact_output_dir is not None" in source
    assert "and not formal_libero" in source
    assert "self._checkpoint_due" in source
    assert "formal LIBERO training did not complete exactly 2000 steps" in source
    assert "formal LIBERO checkpoint already exists" in source


def test_exact_output_validator_accepts_only_matching_real_directory(
    tmp_path: Path,
) -> None:
    from sana_wam.train.trainer import Trainer

    target = tmp_path / "formal"
    target.mkdir()
    trainer = Trainer.__new__(Trainer)
    trainer.t = OmegaConf.create(
        {
            "output_dir": str(target),
            "exact_output_dir": True,
            "formal_libero_training": True,
            "formal_non_resumable": True,
            "expected_world_size": 1,
        }
    )
    trainer.cfg = OmegaConf.create({"dataloader": {"type": "libero"}})
    trainer._phase6_launch_context = None
    trainer._post_primary_afcc_authorized = False
    assert trainer._validate_exact_output_dir(str(target)) == str(target)

    with pytest.raises(RuntimeError, match="differs"):
        trainer._validate_exact_output_dir(str(tmp_path / "other"))

    symlink = tmp_path / "formal-link"
    os.symlink(target, symlink)
    trainer.t.output_dir = str(symlink)
    with pytest.raises(RuntimeError, match="real directory|symlinks"):
        trainer._validate_exact_output_dir(str(symlink))
