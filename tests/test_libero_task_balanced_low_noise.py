from __future__ import annotations

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


CONFIG = Path(
    "configs/experiments/"
    "libero_formal_r8_taskbalanced_8gpu_lownoise_successor1.yaml"
)
CONTROL_CONFIG = Path(
    "configs/experiments/libero_formal_r8_taskbalanced_8gpu_successor1.yaml"
)
RUNNER = Path(
    "scripts/train_libero_formal_r8_taskbalanced_8gpu_lownoise_successor1.py"
)


def test_low_noise_successor_is_exact_single_scientific_delta() -> None:
    candidate = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    control = OmegaConf.to_container(OmegaConf.load(CONTROL_CONFIG), resolve=True)

    assert candidate["dataloader"] == control["dataloader"]
    candidate_arch = candidate["model"]["architecture"]
    control_arch = control["model"]["architecture"]
    assert candidate_arch["action_loss_weighting"] == "low_noise"
    assert control_arch["action_loss_weighting"] == "none"
    candidate_arch["action_loss_weighting"] = control_arch["action_loss_weighting"]
    assert candidate["model"] == control["model"]

    candidate_training = candidate["training"]
    control_training = control["training"]
    candidate_training.pop("output_dir")
    control_training.pop("output_dir")
    assert candidate_training == control_training


def test_low_noise_successor_pins_frozen_r8_and_fresh_balanced_round() -> None:
    cfg = OmegaConf.load(CONFIG)
    assert cfg.training.init_checkpoint == (
        "/DATA/share/sana_wam_libero_training/formal_epoch1_r8_h8_ynorm_fp32attn/"
        "libero-ar-r8-h8-ynorm-fp32attn-causal1-warmstart-uniform-8gpu-4337-"
        "01b7c080232706dddcdbe20bc0fbcdf5/checkpoint_step_4337.safetensors"
    )
    assert cfg.training.init_checkpoint_sha256 == (
        "4a50f6b90b6d04f7c24a25ad529c2bc96d3a35ac257a6dbe42826f989539ab09"
    )
    assert cfg.training.formal_predecessor_r8_checkpoint_sha256 == (
        cfg.training.init_checkpoint_sha256
    )
    assert cfg.training.formal_non_resumable is True
    assert cfg.training.max_steps == cfg.training.formal_final_step == 4_340
    assert cfg.training.lr_schedule_steps == 4_340
    assert cfg.training.balanced_rounds_in_this_run == 1
    assert cfg.training.libero_sampler_contract == "task_balanced_40_v1"
    assert cfg.training.expected_world_size == 8
    assert cfg.training.expected_global_batch_size == 8
    assert cfg.training.gradient_accumulation_steps == 1


def test_low_noise_runner_is_independent_fail_closed_and_verifies_config() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'RUN_NONCE = "709c702a6bc3bb797915942d0bb26572"' in source
    assert "EXPECTED_DRAWS_PER_TASK = 868" in source
    assert "EXPECTED_WINDOW_DRAWS = 34_720" in source
    assert "EXPECTED_FINAL_STEP = 4_340" in source
    assert "SANA_WAM_R8_TASKBALANCED_LOWNOISE_SUPERVISED_WORKER" in source
    assert source.count("trainer = Trainer(") == 1
    assert source.count("output = trainer.train()") == 1
    assert '"--nproc-per-node=8"' in source
    assert '"--max-restarts=0"' in source
    assert (
        '"configs/experiments/libero_formal_r8_taskbalanced_8gpu_successor1.yaml"'
        in source
    )
    for expected_sha256 in (
        "b72152c5bd4e1ff16338e87bd7c2d1ea8e34f71484dec1d8d1dc60de389a2eef",
        "3dcdf563df4c384828b22837b8c0fce5911a40b55c2a12472d082172355d60de",
        "a0cbae74dec990cceb92e341e909a6807e2bc9407efd4236a7d900e1700a2db7",
        "1ee3b445087033feb9c1df737d5a9603f09ebe44589bc75aede9c35d1ff928c9",
        "1608f1c3b3f8151808fb85af7852c88a406c593e661ef5034b84f09948b73008",
    ):
        assert expected_sha256 in source
    assert '_git("status", "--porcelain", "--untracked-files=no")' in source
    worker_source = source[source.index("def _worker") :]
    assert worker_source.index("source = _verify_source(args)") < worker_source.index(
        "import torch"
    )
    assert "dist.all_gather_object(gathered_source, source)" in worker_source
    assert '"control_authority": control_authority' in source
    assert '"source": source' in worker_source
    namespace = runpy.run_path(str(RUNNER))
    cfg = namespace["_load_and_verify_config"]()
    assert cfg.model.architecture.action_loss_weighting == "low_noise"


def test_low_noise_source_verifier_rejects_tracked_changes(monkeypatch) -> None:
    namespace = runpy.run_path(str(RUNNER))
    verify_source = namespace["_verify_source"]

    def fake_git(*args: str, cwd: Path | None = None) -> str:
        del cwd
        if args == ("rev-parse", "HEAD"):
            return "candidate-head"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return " M tracked.py"
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setitem(verify_source.__globals__, "_git", fake_git)
    args = SimpleNamespace(
        expected_repo_commit="candidate-head",
        expected_runner_sha256="unused",
        expected_config_sha256="unused",
    )
    with pytest.raises(RuntimeError, match="repository has tracked changes"):
        verify_source(args)
