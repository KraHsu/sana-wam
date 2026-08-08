from __future__ import annotations

import ast
import hashlib
import runpy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn


CONFIG = Path("configs/experiments/libero_formal_r10_qkv_adapt_8gpu_epoch1.yaml")
RUNNER = Path("scripts/train_libero_formal_r10_qkv_adapt_8gpu_epoch1.py")

R8_RESULT_SHA256 = "11b1b68db1b138bbe0c638f0f85beb174c8ebe8c8d1925c5abd06bb1378e966c"
R8_CHECKPOINT_SHA256 = (
    "4a50f6b90b6d04f7c24a25ad529c2bc96d3a35ac257a6dbe42826f989539ab09"
)
R8_FORMAL40_SHA256 = "29eb848d9a25207a109b95b0e576806753fc1a5a7befe8340de2bed395852e22"
R8_K1_AB40_SHA256 = "9d9251880fae1456496b504e6e3ab0df528ca048fe8ab90b4449a458d6c9dd7f"
PATTERNS = [
    "action_backbone.*",
    "proprio_encoder.*",
    "proprio_video_embed.*",
    "proprio_action_embed.*",
    "video_backbone.dit.blocks.*.attn.qkv.weight",
]


def _namespace() -> dict:
    return runpy.run_path(str(RUNNER))


def test_r10_config_is_exact_h8_qkv_adapt_successor() -> None:
    cfg = OmegaConf.load(CONFIG)
    assert cfg.training.libero_trainable_contract == "r10_qkv_adapt_v1"
    assert list(cfg.training.trainable_parameter_patterns) == PATTERNS
    assert list(cfg.training.eval_modules) == ["video_backbone"]
    for forbidden in (
        "training.trainable_modules",
        "training.preserve_frozen_input_grad_modules",
        "training.freeze",
    ):
        assert OmegaConf.select(cfg, forbidden, default="__ABSENT__") == "__ABSENT__"

    assert cfg.training.max_steps == 4337
    assert cfg.training.formal_final_step == 4337
    assert cfg.training.lr_schedule_steps == 4337
    assert cfg.training.expected_world_size == 8
    assert cfg.training.expected_global_batch_size == 8
    assert cfg.training.batch_size == 1
    assert cfg.training.gradient_accumulation_steps == 1
    assert cfg.training.video_lr == cfg.training.action_lr == 1.5e-5
    assert cfg.training.optimizer_master_weights is True
    assert cfg.training.optimizer_foreach is False
    assert cfg.training.warmup_steps == 50
    assert cfg.training.cumulative_epochs_after_run == 2
    assert cfg.training.epochs_in_this_run == 1
    assert cfg.training.qkv_adaptation_epochs == 1
    assert cfg.model.architecture.action_loss_weighting == "none"
    assert cfg.model.architecture.ar_attn_window == 1
    assert cfg.model.architecture.ar_action_horizon_rope is True
    assert cfg.dataloader.num_frames == 17
    assert cfg.dataloader.action_horizon == 8
    assert cfg.dataloader.window_stride == 8
    assert cfg.training.deployment_contract.action_steps == 20
    assert cfg.training.deployment_contract.action_horizon == 8
    assert cfg.training.deployment_contract.action_horizon_rope == "unique_0_to_7"


def test_r10_config_binds_frozen_r8_and_terminal_k1() -> None:
    cfg = OmegaConf.load(CONFIG)
    assert cfg.training.formal_predecessor_r8_result_sha256 == R8_RESULT_SHA256
    assert cfg.training.formal_predecessor_r8_checkpoint_sha256 == R8_CHECKPOINT_SHA256
    assert cfg.training.init_checkpoint_sha256 == R8_CHECKPOINT_SHA256
    assert (
        cfg.training.scientific_predecessor_r8_formal40_result_sha256
        == R8_FORMAL40_SHA256
    )
    assert (
        cfg.training.diagnostic_predecessor_r8_k1_ab40_result_sha256
        == R8_K1_AB40_SHA256
    )
    assert (
        OmegaConf.select(
            cfg, "training.init_checkpoint_allow_missing_patterns", default=None
        )
        is None
    )


def test_r10_verify_config_closes_joint_candidate_and_deploy_parity() -> None:
    cfg = _namespace()["_verify_config"]()
    assert cfg.training.libero_trainable_contract == "r10_qkv_adapt_v1"
    assert list(cfg.training.eval_modules) == ["video_backbone"]


def test_r10_runner_binds_predecessors_before_any_root_creation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    for digest in (
        R8_RESULT_SHA256,
        R8_CHECKPOINT_SHA256,
        R8_FORMAL40_SHA256,
        R8_K1_AB40_SHA256,
    ):
        assert digest in source
    supervisor = source[
        source.index("def _supervise") : source.index("def _parse_args")
    ]
    source_offset = supervisor.index("source = _verify_source(")
    config_offset = supervisor.index("_verify_config()")
    predecessor_offset = supervisor.index("predecessors = _verify_predecessors()")
    root_offset = supervisor.index("_prepare_root_parent()")
    assert source_offset < config_offset < predecessor_offset < root_offset
    assert 'k1_aggregate.get("successes") != 1' in source
    assert 'aggregate.get("successes") != 3' in source


def test_r10_runner_is_zero_restart_8gpu_qkv_treatment() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"--nproc-per-node=8"' in source
    assert '"--max-restarts=0"' in source
    assert '"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"' in source
    assert "SANA_WAM_R10_QKV_ADAPT_SUPERVISED_WORKER" in source
    assert '"name": "r10_qkv_adapt"' in source
    assert '"candidate_axis": "joint_action_visual_qkv_coadaptation"' in source
    assert '"causal_qkv_attribution": False' in source
    assert '"matched_r8_continuation_control": False' in source
    assert '"cumulative_epochs_after_run": 2' in source
    assert '"epochs_in_this_run": 1' in source
    assert '"qkv_adaptation_epochs": 1' in source
    assert '"sampler_epochs_cumulative": 2.0' in source
    assert "EXPECTED_QKV_PARAMETERS = 301_056_000" in source
    assert "EXPECTED_QKV_TENSORS = 20" in source
    assert "EXPECTED_TRAINABLE_PARAMETERS = 940_709_063" in source
    assert "EXPECTED_TRAINABLE_TENSORS = 580" in source
    assert "qkv_composite_changed" in source
    assert "video_backbone_eval_in_training" in source
    assert "video_backbone_eval_in_deployment" in source
    assert "FAILED_CLOSED_NON_RESUMABLE" in source
    assert "automatic_rerun_permitted" in source


class _TinyAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(3, 6, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _TinyAttn()
        self.ffn = nn.Linear(3, 3, bias=False)


class _TinyDit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(), _TinyBlock()])


class _TinyVideo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dit = _TinyDit()


class _TinyArchitecture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_backbone = nn.Linear(2, 3, bias=False)
        self.proprio_encoder = nn.Linear(2, 2, bias=False)
        self.proprio_video_embed = nn.Linear(2, 2, bias=False)
        self.proprio_action_embed = nn.Linear(2, 2, bias=False)
        self.video_backbone = _TinyVideo()


class _TinyTrainer:
    def __init__(self, *, keep_video_train: bool = False) -> None:
        self.architecture = _TinyArchitecture()
        self.keep_video_train = keep_video_train
        self.architecture.requires_grad_(False)
        for root in (
            self.architecture.action_backbone,
            self.architecture.proprio_encoder,
            self.architecture.proprio_video_embed,
            self.architecture.proprio_action_embed,
        ):
            root.requires_grad_(True)
        for block in self.architecture.video_backbone.dit.blocks:
            block.attn.qkv.weight.requires_grad_(True)

    def _set_training_mode(self) -> None:
        self.architecture.train()
        if not self.keep_video_train:
            self.architecture.video_backbone.eval()


def _tiny_partition_namespace() -> tuple[dict, _TinyTrainer]:
    ns = _namespace()
    qkv_names = (
        "video_backbone.dit.blocks.0.attn.qkv.weight",
        "video_backbone.dit.blocks.1.attn.qkv.weight",
    )
    trainer = _TinyTrainer()
    named = dict(trainer.architecture.named_parameters())
    trainable_names = [
        name for name, parameter in named.items() if parameter.requires_grad
    ]
    globals_dict = ns["_verify_materialized_trainable_partition"].__globals__
    globals_dict["EXPECTED_QKV_NAMES"] = qkv_names
    globals_dict["EXPECTED_QKV_TENSORS"] = len(qkv_names)
    globals_dict["EXPECTED_QKV_PARAMETERS"] = sum(
        named[name].numel() for name in qkv_names
    )
    globals_dict["EXPECTED_TRAINABLE_TENSORS"] = len(trainable_names)
    globals_dict["EXPECTED_TRAINABLE_PARAMETERS"] = sum(
        named[name].numel() for name in trainable_names
    )
    return ns, trainer


def test_materialized_partition_proves_eval_mode_and_exact_qkv_names() -> None:
    ns, trainer = _tiny_partition_namespace()
    manifest = ns["_verify_materialized_trainable_partition"](trainer)
    assert manifest["video_backbone_eval"] is True
    assert manifest["all_video_submodules_eval"] is True
    assert manifest["qkv"]["tensor_count"] == 2
    assert manifest["qkv"]["all_require_grad"] is True
    assert manifest["pattern_tensor_counts"][PATTERNS[-1]] == 2
    assert len(manifest["trainable_names"]) == 6


def test_materialized_partition_rejects_training_mode_video() -> None:
    ns, _ = _tiny_partition_namespace()
    trainer = _TinyTrainer(keep_video_train=True)
    with pytest.raises(RuntimeError, match="must remain entirely in eval mode"):
        ns["_verify_materialized_trainable_partition"](trainer)


def test_qkv_composite_digest_changes_with_one_weight() -> None:
    digest = _namespace()["_composite_parameter_digest"]
    parameter = nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    before = digest([("video_backbone.dit.blocks.0.attn.qkv.weight", parameter)])
    with torch.no_grad():
        parameter[0, 1] += 0.5
    after = digest([("video_backbone.dit.blocks.0.attn.qkv.weight", parameter)])
    assert before != after
    assert len(before) == len(after) == hashlib.sha256().digest_size * 2


def test_r10_runner_keeps_two_stage_terminal_commit() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    success_body = source[
        source.index("def _publish_success_result") : source.index(
            "def _discard_uncommitted_result"
        )
    ]
    assert success_body.index("_freeze_root_contents()") < success_body.index(
        'RUN_ROOT / "RESULT.json", payload, mode=0o444'
    )
    assert success_body.index('RUN_ROOT / "RESULT.json", payload, mode=0o444') < (
        success_body.index('_seal_root("RESULT.json")')
    )
    assert "start_new_session=True" in source
    assert "os.killpg(process_group_id, signal.SIGTERM)" in source
    assert "os.killpg(process_group_id, signal.SIGKILL)" in source
