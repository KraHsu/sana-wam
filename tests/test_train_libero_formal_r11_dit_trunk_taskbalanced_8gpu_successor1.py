from __future__ import annotations

import ast
import fnmatch
import hashlib
import runpy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn


CONFIG = Path(
    "configs/experiments/libero_formal_r11_dit_trunk_taskbalanced_8gpu_successor1.yaml"
)
RUNNER = Path(
    "scripts/train_libero_formal_r11_dit_trunk_taskbalanced_8gpu_successor1.py"
)
R8_RESULT_SHA256 = "11b1b68db1b138bbe0c638f0f85beb174c8ebe8c8d1925c5abd06bb1378e966c"
R8_CHECKPOINT_SHA256 = (
    "4a50f6b90b6d04f7c24a25ad529c2bc96d3a35ac257a6dbe42826f989539ab09"
)
CONTROL_RESULT_SHA256 = "a0cbae74dec990cceb92e341e909a6807e2bc9407efd4236a7d900e1700a2db7"
CONTROL_CHECKPOINT_SHA256 = (
    "1ee3b445087033feb9c1df737d5a9603f09ebe44589bc75aede9c35d1ff928c9"
)
CONTROL_CONFIG_SHA256 = "b72152c5bd4e1ff16338e87bd7c2d1ea8e34f71484dec1d8d1dc60de389a2eef"
CONTROL_RUNNER_SHA256 = "3dcdf563df4c384828b22837b8c0fce5911a40b55c2a12472d082172355d60de"
CONTROL_AB_SHA256 = "1608f1c3b3f8151808fb85af7852c88a406c593e661ef5034b84f09948b73008"
RUN_NONCE = "698541efd5c0b1c9f6b94cd5cbebf284"
PATTERNS = [
    "action_backbone.*",
    "proprio_encoder.*",
    "proprio_video_embed.*",
    "proprio_action_embed.*",
    "video_backbone.dit.x_embedder.*",
    "video_backbone.dit.t_embedder.*",
    "video_backbone.dit.t_block.*",
    "video_backbone.dit.y_embedder.*",
    "video_backbone.dit.attention_y_norm.*",
    "video_backbone.dit.blocks.*",
]


def _namespace() -> dict:
    return runpy.run_path(str(RUNNER))


def test_r11_config_is_exact_matched_taskbalanced_successor() -> None:
    cfg = OmegaConf.load(CONFIG)
    assert cfg.training.libero_trainable_contract == "r11_dit_trunk_adapt_v1"
    assert list(cfg.training.trainable_parameter_patterns) == PATTERNS
    assert list(cfg.training.eval_modules) == ["video_backbone"]
    for forbidden in (
        "training.trainable_modules",
        "training.preserve_frozen_input_grad_modules",
        "training.freeze",
        "training.trainable_parameter_dtype",
    ):
        assert OmegaConf.select(cfg, forbidden, default="__ABSENT__") == "__ABSENT__"
    assert cfg.training.max_steps == cfg.training.formal_final_step == 4340
    assert cfg.training.lr_schedule_steps == 4340
    assert cfg.training.libero_sampler_contract == "task_balanced_40_v1"
    assert cfg.training.expected_world_size == 8
    assert cfg.training.expected_global_batch_size == 8
    assert cfg.training.batch_size == cfg.training.gradient_accumulation_steps == 1
    assert cfg.training.video_lr == cfg.training.action_lr == 1.5e-5
    assert cfg.training.lambda_video == 0.0
    assert cfg.training.lambda_action == 1.0
    assert cfg.model.architecture.action_loss_weighting == "none"
    assert cfg.training.initialization_seed == cfg.training.seed == 20260810
    assert cfg.training.init_checkpoint_sha256 == R8_CHECKPOINT_SHA256
    assert str(cfg.training.output_dir).endswith(RUN_NONCE)


def test_r11_config_binds_r8_and_all_matched_control_evidence() -> None:
    cfg = OmegaConf.load(CONFIG)
    assert cfg.training.formal_predecessor_r8_result_sha256 == R8_RESULT_SHA256
    assert cfg.training.formal_predecessor_r8_checkpoint_sha256 == R8_CHECKPOINT_SHA256
    assert cfg.training.matched_control_result_sha256 == CONTROL_RESULT_SHA256
    assert cfg.training.matched_control_checkpoint_sha256 == CONTROL_CHECKPOINT_SHA256
    assert cfg.training.matched_control_config_sha256 == CONTROL_CONFIG_SHA256
    assert cfg.training.matched_control_runner_sha256 == CONTROL_RUNNER_SHA256
    assert cfg.training.matched_control_same40_ab_result_sha256 == CONTROL_AB_SHA256


def test_r11_verify_config_closes_relative_treatment_and_deploy_parity() -> None:
    cfg = _namespace()["_verify_config"]()
    assert cfg.training.libero_trainable_contract == "r11_dit_trunk_adapt_v1"
    assert list(cfg.training.eval_modules) == ["video_backbone"]


def test_r11_runner_binds_evidence_before_root_creation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    for digest in (
        R8_RESULT_SHA256,
        R8_CHECKPOINT_SHA256,
        CONTROL_RESULT_SHA256,
        CONTROL_CHECKPOINT_SHA256,
        CONTROL_CONFIG_SHA256,
        CONTROL_RUNNER_SHA256,
        CONTROL_AB_SHA256,
    ):
        assert digest in source
    supervisor = source[source.index("def _supervise") : source.index("def _parse_args")]
    assert supervisor.index("source = _verify_source(") < supervisor.index("_verify_config()")
    assert supervisor.index("_verify_config()") < supervisor.index(
        "predecessors = _verify_predecessors()"
    )
    assert supervisor.index("predecessors = _verify_predecessors()") < supervisor.index(
        "_prepare_root_parent()"
    )
    assert 'same40_aggregate.get("successes") != 6' in source


def test_r11_runner_declares_exact_partition_and_terminal_proofs() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert f'RUN_NONCE = "{RUN_NONCE}"' in source
    assert '"--nproc-per-node=8"' in source
    assert '"--max-restarts=0"' in source
    assert "SANA_WAM_R11_DIT_TRUNK_SUPERVISED_WORKER" in source
    assert "EXPECTED_DIT_TRUNK_PARAMETERS = 2_458_986_920" in source
    assert "EXPECTED_DIT_TRUNK_TENSORS = 693" in source
    assert "EXPECTED_TRAINABLE_PARAMETERS = 3_098_639_983" in source
    assert "EXPECTED_TRAINABLE_TENSORS = 1_253" in source
    assert "equals_all_dit_named_parameters_excluding_final_layer" in source
    assert "optimizer_source_parameter_ids_exact" in source
    assert "all_20_block_composites_changed" in source
    assert "frozen_partitions_unchanged" in source
    assert "buffers_unchanged" in source
    assert "rank_consensus" in source
    assert "FAILED_CLOSED_NON_RESUMABLE" in source


class _TinyDit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x_embedder = nn.Linear(2, 2)
        self.t_embedder = nn.Linear(2, 2)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(2, 2))
        self.y_embedder = nn.Linear(2, 2)
        self.attention_y_norm = nn.LayerNorm(2)
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(20)])
        self.final_layer = nn.Linear(2, 2)


class _TinyVideo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dit = _TinyDit()
        self.vae = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)


class _TinyArchitecture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_backbone = nn.Linear(2, 2)
        self.proprio_encoder = nn.Linear(2, 2)
        self.proprio_video_embed = nn.Linear(2, 2)
        self.proprio_action_embed = nn.Linear(2, 2)
        self.video_backbone = _TinyVideo()
        self.register_buffer("audit_buffer", torch.tensor([1.0]))


class _TinyTrainer:
    def __init__(self, *, video_train: bool = False, omit_optimizer_param: bool = False):
        self.architecture = _TinyArchitecture()
        self.video_train = video_train
        self.omit_optimizer_param = omit_optimizer_param
        self.architecture.requires_grad_(False)
        for name, parameter in self.architecture.named_parameters():
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in PATTERNS):
                parameter.requires_grad_(True)

    def _set_training_mode(self) -> None:
        self.architecture.train()
        if not self.video_train:
            self.architecture.video_backbone.eval()

    def _param_groups(self) -> list[dict]:
        named = dict(self.architecture.named_parameters())
        action = [
            parameter
            for name, parameter in named.items()
            if parameter.requires_grad and name.startswith("action_backbone.")
        ]
        video = [
            parameter
            for name, parameter in named.items()
            if parameter.requires_grad and not name.startswith("action_backbone.")
        ]
        if self.omit_optimizer_param:
            video = video[:-1]
        return [{"params": action, "lr": 1.5e-5}, {"params": video, "lr": 1.5e-5}]


def _tiny_manifest(*, video_train: bool = False, omit_optimizer_param: bool = False):
    ns = _namespace()
    trainer = _TinyTrainer(
        video_train=video_train, omit_optimizer_param=omit_optimizer_param
    )
    named = dict(trainer.architecture.named_parameters())
    trainable = [parameter for parameter in named.values() if parameter.requires_grad]
    dit = [
        parameter
        for name, parameter in named.items()
        if name.startswith("video_backbone.dit.")
        and not name.startswith("video_backbone.dit.final_layer.")
    ]
    globals_dict = ns["_verify_materialized_trainable_partition"].__globals__
    globals_dict["EXPECTED_TRAINABLE_TENSORS"] = len(trainable)
    globals_dict["EXPECTED_TRAINABLE_PARAMETERS"] = sum(p.numel() for p in trainable)
    globals_dict["EXPECTED_DIT_TRUNK_TENSORS"] = len(dit)
    globals_dict["EXPECTED_DIT_TRUNK_PARAMETERS"] = sum(p.numel() for p in dit)
    return ns, trainer


def test_materialized_partition_proves_exact_trunk_freeze_eval_and_optimizer_ids() -> None:
    ns, trainer = _tiny_manifest()
    manifest = ns["_verify_materialized_trainable_partition"](trainer)
    assert manifest["video_backbone_eval"] is True
    assert manifest["optimizer_source_parameter_ids_exact"] is True
    assert manifest["dit_trunk"][
        "equals_all_dit_named_parameters_excluding_final_layer"
    ] is True
    assert len(manifest["dit_trunk"]["block_composite_sha256"]) == 20
    assert set(manifest["frozen_partitions"]) == {"final_layer", "vae", "text_encoder"}
    assert manifest["buffers"]["tensor_count"] == 1


def test_materialized_partition_rejects_training_video_or_optimizer_omission() -> None:
    ns, trainer = _tiny_manifest(video_train=True)
    with pytest.raises(RuntimeError, match="entirely in eval mode"):
        ns["_verify_materialized_trainable_partition"](trainer)
    ns, trainer = _tiny_manifest(omit_optimizer_param=True)
    with pytest.raises(RuntimeError, match="optimizer source parameter identities"):
        ns["_verify_materialized_trainable_partition"](trainer)


def test_each_block_and_trunk_digest_changes_independently_of_frozen_roots() -> None:
    ns, trainer = _tiny_manifest()
    before = ns["_verify_materialized_trainable_partition"](trainer)
    with torch.no_grad():
        for block in trainer.architecture.video_backbone.dit.blocks:
            block.weight[0, 0] += 0.5
    after = ns["_verify_materialized_trainable_partition"](trainer)
    assert before["dit_trunk"]["composite_sha256"] != after["dit_trunk"][
        "composite_sha256"
    ]
    for index in map(str, range(20)):
        assert before["dit_trunk"]["block_composite_sha256"][index] != after[
            "dit_trunk"
        ]["block_composite_sha256"][index]
    assert before["frozen_partitions"] == after["frozen_partitions"]
    assert before["buffers"] == after["buffers"]


def test_composite_digest_is_name_and_value_sensitive() -> None:
    digest = _namespace()["_composite_parameter_digest"]
    parameter = nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    before = digest([("video_backbone.dit.blocks.0.weight", parameter)])
    with torch.no_grad():
        parameter[0, 1] += 0.5
    after = digest([("video_backbone.dit.blocks.0.weight", parameter)])
    assert before != after
    assert len(before) == len(after) == hashlib.sha256().digest_size * 2


def test_scalar_bf16_digest_helpers_hash_storage_bytes_without_view_error() -> None:
    namespace = _namespace()
    scalar = torch.tensor(1.25, dtype=torch.bfloat16)
    expected = hashlib.sha256(
        scalar.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    assert namespace["_tensor_sha256"](scalar) == expected
    composite = namespace["_composite_parameter_digest"]([("scalar", scalar)])
    assert len(composite) == hashlib.sha256().digest_size * 2
    source = RUNNER.read_text(encoding="utf-8")
    assert source.count(".reshape(-1)") == 3


def test_r11_runner_keeps_two_stage_terminal_commit() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    success = source[
        source.index("def _publish_success_result") : source.index(
            "def _discard_uncommitted_result"
        )
    ]
    assert success.index("_freeze_root_contents()") < success.index(
        'RUN_ROOT / "RESULT.json", payload, mode=0o444'
    )
    assert success.index('RUN_ROOT / "RESULT.json", payload, mode=0o444') < success.index(
        '_seal_root("RESULT.json")'
    )
    assert "start_new_session=True" in source
    assert "os.killpg(process_group_id, signal.SIGTERM)" in source
    assert "os.killpg(process_group_id, signal.SIGKILL)" in source
