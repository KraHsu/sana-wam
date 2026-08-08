from __future__ import annotations

import ast
import hashlib
import runpy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from sana_wam.deploy.libero_policy_server import validate_libero_train_deploy_parity


CONFIG = Path("configs/experiments/libero_formal_r8_h8_ynorm_fp32attn_8gpu_epoch1.yaml")
RUNNER = Path("scripts/train_libero_formal_r8_h8_ynorm_fp32attn_8gpu_epoch1.py")
TRAINER = Path("src/sana_wam/train/trainer.py")


def test_r8_h8_ynorm_fp32attn_is_one_epoch_at_preserved_global_batch() -> None:
    cfg = OmegaConf.load(CONFIG)
    assert cfg.training.max_steps == 4337
    assert cfg.training.formal_final_step == 4337
    assert cfg.training.lr_schedule_steps == 4337
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
    assert 4337 * 8 == 34696
    assert 34696 - 34693 == 3
    runner_source = RUNNER.read_text(encoding="utf-8")
    config_source = CONFIG.read_text(encoding="utf-8")
    assert "EXPECTED_DATASET_WINDOWS = 34_693" in runner_source
    assert "EXPECTED_WINDOW_DRAWS = 34_696" in runner_source
    assert "EXPECTED_SAMPLER_PADDING_REPEATS = 3" in runner_source
    assert "unique 0..7 RoPE slots" in config_source
    assert "unique 0..27 RoPE slots" not in config_source


def test_r8_h8_ynorm_fp32attn_is_strict_r7_warm_start() -> None:
    cfg = OmegaConf.load(CONFIG)
    assert cfg.model.architecture.action_loss_weighting == "none"
    assert cfg.model.architecture.ar_attn_window == 1
    assert cfg.model.architecture.ar_action_horizon_rope is True
    assert cfg.model.architecture.ar_noisy_cond_prob == 0.0
    assert cfg.dataloader.video_context_mode == "causal_past"
    assert cfg.dataloader.action_horizon == 8
    assert cfg.dataloader.window_stride == 8
    assert cfg.training.deployment_contract.action_horizon_rope == "unique_0_to_7"
    assert cfg.dataloader.require_full_action_horizon is True
    assert cfg.training.formal_warm_start is True
    assert cfg.training.formal_non_resumable is True
    assert cfg.model.video_backbone.model_kwargs.y_norm is True
    assert cfg.model.video_backbone.model_kwargs.y_norm_scale_factor == 0.01
    assert OmegaConf.select(
        cfg, "model.video_backbone.fp32_attention", default=False
    ) is False
    assert cfg.training.init_checkpoint_sha256 == (
        "2c461d9d55dcd8db0e5aa8055b5db780c15afa4067f1cc56a555f443a073f034"
    )
    assert OmegaConf.select(
        cfg, "training.init_checkpoint_allow_missing_patterns", default=None
    ) is None
    assert cfg.training.optimizer_master_weights is True
    assert cfg.training.optimizer_foreach is False
    assert cfg.training.warmup_steps == 50
    assert cfg.training.video_lr == 1.5e-5
    assert cfg.training.action_lr == 1.5e-5
    assert cfg.training.epochs_in_this_run == 1
    assert cfg.training.cumulative_epochs_after_run == 1
    assert cfg.training.formal_predecessor_r7_result_sha256 == (
        "8999efefeb17b5123c7cfaa7f5115da093bb54deab840d924aed050c11d22ea5"
    )
    assert cfg.training.formal_predecessor_r7_checkpoint_sha256 == (
        "2c461d9d55dcd8db0e5aa8055b5db780c15afa4067f1cc56a555f443a073f034"
    )
    assert cfg.training.scientific_predecessor_r7_formal40_result_sha256 == (
        "c2df9659346c13bc5200e546f6c37b0179ae240b77a7e1d72dfb56734ef64433"
    )


def test_r8_h8_ynorm_fp32attn_checkpoint_contract_matches_default_deployment() -> None:
    training_cfg = OmegaConf.load(CONFIG)
    deploy_cfg = OmegaConf.load("configs/deploy_ar_sana.yaml")
    runtime_cfg = OmegaConf.merge(training_cfg, deploy_cfg)
    validate_libero_train_deploy_parity(training_cfg, runtime_cfg)
    assert runtime_cfg.policy.history_len == training_cfg.dataloader.num_frames == 17
    assert runtime_cfg.inference.video_num_frames == 5
    assert runtime_cfg.inference.action_steps == 20
    assert runtime_cfg.model.video_backbone.model_kwargs.y_norm is True
    assert runtime_cfg.model.video_backbone.model_kwargs.y_norm_scale_factor == 0.01


def test_r8_runner_has_single_supervisor_and_zero_restart_torchrun() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"--nproc-per-node=8"' in source
    assert '"--max-restarts=0"' in source
    assert '"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"' in source
    assert "SANA_WAM_R8_H8_YNORM_FP32ATTN_SUPERVISED_WORKER" in source
    assert "FAILED_CLOSED_NON_RESUMABLE" in source
    assert "automatic_rerun_permitted" in source
    assert "r7_h8_ynorm_formal40" in source
    assert 'aggregate.get("episodes") != 40' in source
    assert 'aggregate.get("successes") != 4' in source
    assert "scientific_r7_h8_ynorm_formal40" in source
    assert "_verify_materialized_r7_warm_start" in source
    assert "_verify_materialized_fp32_ar_operator" in source
    assert source.count("weight.numel() != 2_240") == 2
    assert "weight.numel() != 2_304" not in source
    assert '"checkpoint_missing": []' in source
    assert '"r7_checkpoint_loaded_exactly": True' in source
    assert '"source_level_unconditional": True' in source
    assert '"legacy_config_knob_enabled": False' in source
    assert "start_new_session=True" in source
    assert "os.killpg(process_group_id, signal.SIGTERM)" in source
    assert "os.killpg(process_group_id, signal.SIGKILL)" in source
    assert "torchrun process group could not be terminated; refusing to " in source
    assert '"write or freeze a terminal artifact"' in source


def test_r8_runner_binds_r7_and_fp32_ar_sources_before_root_creation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert (
        "8999efefeb17b5123c7cfaa7f5115da093bb54deab840d924aed050c11d22ea5"
        in source
    )
    assert (
        "2c461d9d55dcd8db0e5aa8055b5db780c15afa4067f1cc56a555f443a073f034"
        in source
    )
    assert (
        "484248da5568c9366002b3404585666a7bfb1e81112094ed8036e2bdd0d9098b"
        in source
    )
    supervisor = source[source.index("def _supervise") : source.index("def _parse_args")]
    source_offset = supervisor.index("source = _verify_source(")
    predecessor_offset = supervisor.index("predecessors = _verify_predecessors()")
    root_offset = supervisor.index("_prepare_root_parent()")
    assert source_offset < predecessor_offset < root_offset
    assert "YNORM_BF16_SHA256" in source[
        source.index("def _verify_materialized_r7_warm_start") : source.index(
            "def _verify_saved_ynorm_checkpoint"
        )
    ]


def test_r8_config_is_exact_r7_single_axis_successor() -> None:
    namespace = runpy.run_path(str(RUNNER))
    cfg = namespace["_verify_config"]()
    assert cfg.training.video_lr == 1.5e-5
    assert cfg.training.action_lr == 1.5e-5
    assert cfg.training.warmup_steps == 50
    assert cfg.training.max_steps == 4337


def test_r8_runner_pins_reviewed_fp32_ar_core_bytes() -> None:
    namespace = runpy.run_path(str(RUNNER))
    expected = namespace["FP32_AR_SOURCE_SHA256"]
    predecessor = namespace["R7_FP32_AR_SOURCE_SHA256"]
    for relative_path, expected_sha256 in expected.items():
        observed = hashlib.sha256(Path(relative_path).read_bytes()).hexdigest()
        assert observed == expected_sha256
        assert observed != predecessor[relative_path]
    assert namespace["AR_LINEAR_ATTN_EPS"] == 1e-8


def test_r8_h8_ynorm_tensor_digest_rejects_single_element_drift() -> None:
    namespace = runpy.run_path(str(RUNNER))
    verify = namespace["_verify_tensor_sha256"]
    tensor = torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16)
    expected = hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()

    assert verify(tensor, expected, label="test tensor") == expected
    changed = tensor.clone()
    changed[1] = changed[1] + 1
    with pytest.raises(RuntimeError, match="SHA256 mismatch for test tensor"):
        verify(changed, expected, label="test tensor")


def test_r8_h8_ynorm_epoch1_runner_uses_two_stage_terminal_commit_and_safe_ancestry() -> None:
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


def test_r8_h8_ynorm_epoch1_worker_collectively_commits_summary_and_emits_monitor_logs() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "logging.basicConfig(" in source
    assert "R8 preflight complete; entering formal training" in source
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
    assert "init_checkpoint_allow_missing_patterns" in source
