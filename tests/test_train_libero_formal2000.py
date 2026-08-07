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
    "libero-ar-formal2000-r3-single-gpu-2000-e21c6e5ad5910c00156499472f46bee6"
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
        "training.optimizer_foreach": False,
        "model.architecture.video_on_path_loss_weight": 0.0,
        "model.architecture.video_trajectory_endpoint_weight": 0.0,
        "model.architecture.video_trajectory_velocity_weight": 0.0,
        "model.architecture.video_trajectory_consistency_weight": 0.0,
        "model.architecture.video_local_expansion_weight": 0.0,
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
    assert source.count("e21c6e5ad5910c00156499472f46bee6") >= 2
    assert "5e2f5d61f7715f952f1762c04f5ad99891f9eda1ae5256310498c23146bcc471" in source
    assert "0a62b6a66ba152285d751795330d90a93558c2ca851706be0549dbd6966831c2" in source
    assert "e778230115fa595895ba16e5d8bc5aac545b426277e69b02f4681c34cfe6603a" in source
    assert "1f7ea352023a7b4af1d6aa52feb280c059bb2731b441c1c5a506b3d21038c523" in source
    assert '"training.optimizer_foreach": False' in source
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
    assert "del batch, loss, lv, la, result" in source
    assert "autograd_graph_released_before_optimizer" in source


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, None), (False, False), (True, True)],
)
def test_optimizer_foreach_setting_accepts_only_optional_boolean(
    configured: bool | None,
    expected: bool | None,
) -> None:
    from sana_wam.train.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    payload = {} if configured is None else {"optimizer_foreach": configured}
    trainer.t = OmegaConf.create(payload)
    assert trainer._optimizer_foreach_setting() is expected


@pytest.mark.parametrize("configured", ["false", 0, 1, [], {}])
def test_optimizer_foreach_setting_rejects_non_boolean(configured: object) -> None:
    from sana_wam.train.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.t = OmegaConf.create({"optimizer_foreach": configured})
    with pytest.raises(ValueError, match="optimizer_foreach must be a boolean"):
        trainer._optimizer_foreach_setting()


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
