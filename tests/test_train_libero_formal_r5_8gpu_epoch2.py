from __future__ import annotations

import ast
from pathlib import Path

from omegaconf import OmegaConf

from sana_wam.deploy.libero_policy_server import validate_libero_train_deploy_parity


CONFIG = Path("configs/experiments/libero_formal_r5_8gpu_epoch2.yaml")
RUNNER = Path("scripts/train_libero_formal_r5_8gpu_epoch2.py")
TRAINER = Path("src/sana_wam/train/trainer.py")


def test_r5_epoch2_config_is_one_full_epoch_at_preserved_global_batch() -> None:
    cfg = OmegaConf.load(CONFIG)
    assert cfg.training.max_steps == 1315
    assert cfg.training.formal_final_step == 1315
    assert cfg.training.lr_schedule_steps == 1315
    assert cfg.training.expected_world_size == 8
    assert cfg.training.batch_size == 1
    assert cfg.training.gradient_accumulation_steps == 1
    assert (
        cfg.training.expected_world_size
        * cfg.training.batch_size
        * cfg.training.gradient_accumulation_steps
        == cfg.training.expected_global_batch_size
        == 8
    )
    assert 1315 * 8 == 10520


def test_r5_epoch2_is_strict_r5_r2_warm_start_with_uniform_full_noise_support() -> None:
    cfg = OmegaConf.load(CONFIG)
    assert cfg.model.architecture.action_loss_weighting == "none"
    assert cfg.model.architecture.ar_attn_window == 1
    assert cfg.model.architecture.ar_action_horizon_rope is True
    assert cfg.model.architecture.ar_noisy_cond_prob == 0.0
    assert cfg.dataloader.video_context_mode == "causal_past"
    assert cfg.dataloader.action_horizon == 28
    assert cfg.dataloader.require_full_action_horizon is True
    assert cfg.training.formal_warm_start is True
    assert cfg.training.formal_non_resumable is True
    assert cfg.training.init_checkpoint_sha256 == (
        "0e28f8706676e49dd37d9713e8689f27a876a99498094570372294beace7e1aa"
    )
    assert cfg.training.optimizer_master_weights is True
    assert cfg.training.optimizer_foreach is False
    assert cfg.training.warmup_steps == 33
    assert cfg.training.video_lr == 5.0e-5
    assert cfg.training.action_lr == 5.0e-5
    assert cfg.training.epochs_in_this_run == 1
    assert cfg.training.cumulative_epochs_after_run == 2


def test_r5_epoch2_checkpoint_contract_matches_default_deployment() -> None:
    training_cfg = OmegaConf.load(CONFIG)
    deploy_cfg = OmegaConf.load("configs/deploy_ar_sana.yaml")
    runtime_cfg = OmegaConf.merge(training_cfg, deploy_cfg)
    validate_libero_train_deploy_parity(training_cfg, runtime_cfg)
    assert runtime_cfg.policy.history_len == training_cfg.dataloader.num_frames == 17
    assert runtime_cfg.inference.video_num_frames == 5
    assert runtime_cfg.inference.action_steps == 20


def test_r5_epoch2_runner_has_single_supervisor_and_zero_restart_torchrun() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"--nproc-per-node=8"' in source
    assert '"--max-restarts=0"' in source
    assert '"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"' in source
    assert "SANA_WAM_R5_SUPERVISED_WORKER" in source
    assert "FAILED_CLOSED_NON_RESUMABLE" in source
    assert "automatic_rerun_permitted" in source
    assert "start_new_session=True" in source
    assert "os.killpg(process_group_id, signal.SIGTERM)" in source
    assert "os.killpg(process_group_id, signal.SIGKILL)" in source
    assert "torchrun process group could not be terminated; refusing to " in source
    assert '"write or freeze a terminal artifact"' in source


def test_r5_epoch2_runner_uses_two_stage_terminal_commit_and_safe_ancestry() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    success_body = source[
        source.index("def _publish_success_result") : source.index(
            "def _discard_uncommitted_result"
        )
    ]
    freeze_offset = success_body.index("_freeze_root_contents()")
    result_offset = success_body.index('RUN_ROOT / "RESULT.json", payload, mode=0o444')
    seal_offset = success_body.index('_seal_root("RESULT.json")')
    assert freeze_offset < result_offset < seal_offset
    assert 'getattr(os, "O_NOFOLLOW", 0)' in source
    assert "_prepare_root_parent" in source
    assert 'os.lstat(path)' in source
    assert 'sana_head = _git("rev-parse", "HEAD"' in source


def test_r5_epoch2_worker_collectively_commits_summary_and_emits_monitor_logs() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "logging.basicConfig(" in source
    assert "R5 preflight complete; entering formal training" in source
    assert 'dist.broadcast_object_list(decision, src=0)' in source
    assert 'if not decision[0]["ok"]' in source


def test_trainer_parameterizes_formal_final_step_and_syncs_rank_state() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    ast.parse(source)
    assert 't.get("formal_final_step", max_steps)' in source
    assert "formal_final_step=max_steps" in source
    assert "_broadcast_initial_model_state" in source
    assert "gradient presence differs across distributed ranks" in source
    assert "_ordered_model_state_manifest" in source
    assert "structure_manifest_sha256" in source
    assert "_complete_model_state_digest_consensus" in source
    assert "final_model_state" in source
