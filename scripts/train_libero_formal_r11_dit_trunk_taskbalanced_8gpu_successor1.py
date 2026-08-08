#!/usr/bin/env python
"""Fail-closed 8-GPU supervisor/worker for R11 DiT-trunk adaptation."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import fnmatch
import hashlib
import json
import logging
import os
import signal
import stat
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

CONFIG_RELATIVE_PATH = (
    "configs/experiments/libero_formal_r11_dit_trunk_taskbalanced_8gpu_successor1.yaml"
)
MATCHED_CONTROL_CONFIG_RELATIVE_PATH = (
    "configs/experiments/libero_formal_r8_taskbalanced_8gpu_successor1.yaml"
)
MATCHED_CONTROL_RUNNER_RELATIVE_PATH = (
    "scripts/train_libero_formal_r8_taskbalanced_8gpu_successor1.py"
)
RUN_NONCE = "698541efd5c0b1c9f6b94cd5cbebf284"
RUN_ROOT = Path(
    "/DATA/share/sana_wam_libero_training/formal_taskbalanced_r11_dit_trunk_adapt/"
    "libero-ar-r11-dit-trunk-taskbalanced-causal1-warmstart-8gpu-4340-"
    + RUN_NONCE
)
PREDECESSOR_R8_ROOT = Path(
    "/DATA/share/sana_wam_libero_training/formal_epoch1_r8_h8_ynorm_fp32attn/"
    "libero-ar-r8-h8-ynorm-fp32attn-causal1-warmstart-uniform-8gpu-4337-"
    "01b7c080232706dddcdbe20bc0fbcdf5"
)
PREDECESSOR_R8_RESULT_SHA256 = (
    "11b1b68db1b138bbe0c638f0f85beb174c8ebe8c8d1925c5abd06bb1378e966c"
)
PREDECESSOR_R8_CHECKPOINT_SHA256 = (
    "4a50f6b90b6d04f7c24a25ad529c2bc96d3a35ac257a6dbe42826f989539ab09"
)
MATCHED_CONTROL_ROOT = Path(
    "/DATA/share/sana_wam_libero_training/formal_taskbalanced_r8_h8_ynorm_fp32attn/"
    "libero-ar-r8-taskbalanced-causal1-warmstart-8gpu-4340-"
    "aa10ecdb611a0b0b61abe0b162a66227"
)
MATCHED_CONTROL_RESULT_SHA256 = (
    "a0cbae74dec990cceb92e341e909a6807e2bc9407efd4236a7d900e1700a2db7"
)
MATCHED_CONTROL_CHECKPOINT_SHA256 = (
    "1ee3b445087033feb9c1df737d5a9603f09ebe44589bc75aede9c35d1ff928c9"
)
MATCHED_CONTROL_CONFIG_SHA256 = (
    "b72152c5bd4e1ff16338e87bd7c2d1ea8e34f71484dec1d8d1dc60de389a2eef"
)
MATCHED_CONTROL_RUNNER_SHA256 = (
    "3dcdf563df4c384828b22837b8c0fce5911a40b55c2a12472d082172355d60de"
)
MATCHED_CONTROL_SAME40_AB_RESULT = Path(
    "/DATA/share/sana_wam_libero_evaluation/r8_taskbalanced_settle10_same40/"
    "libero-r8-taskbalanced-settle10-same40-c16a4ccdfd0247b0a65c7d2e34220f91/"
    "AB_RESULT.json"
)
MATCHED_CONTROL_SAME40_AB_RESULT_SHA256 = (
    "1608f1c3b3f8151808fb85af7852c88a406c593e661ef5034b84f09948b73008"
)

FP32_AR_SOURCE_PATHS = (
    "src/sana_wam/model/ar/sana_ar_linear_attn.py",
    "src/sana_wam/model/ar/sana_ar_inference.py",
    "src/sana_wam/model/ar/sana_ar_mot_driver.py",
)
FP32_AR_SOURCE_SHA256 = {
    "src/sana_wam/model/ar/sana_ar_linear_attn.py": (
        "f79e9dfde2fc6a4fff11144daccacaf7cebd071a83f0572e8f0e9a57ef171688"
    ),
    "src/sana_wam/model/ar/sana_ar_inference.py": (
        "4cfa91d2e0de0b0ee9c00a5014b47f6a6fa82747131403a687cdc59f144d9cfc"
    ),
    "src/sana_wam/model/ar/sana_ar_mot_driver.py": (
        "1def9cf0b7f312b9f4d4c1d0c2af9a51f1a8ca6a4e75a8fe14b7b9ad823ec7ed"
    ),
}
AR_LINEAR_ATTN_EPS = 1e-8

SANA_GITLINK = "16b9cec673e3335724ba2d8db25de7f9ed229292"
STATS_SHA256 = "e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7"
GPU_UUIDS = (
    "GPU-1ec28cfb-f501-23f3-f865-275a744ca053",
    "GPU-a5a29c85-dd07-4c19-3b32-d96ab5883bdb",
    "GPU-9137aec3-6d67-daa4-3f48-74732ffced17",
    "GPU-86a21c6d-5b8a-d4b6-78f7-141f83f7ecd7",
    "GPU-97eeb56c-36d0-3aad-be8a-b2b7b049b9e3",
    "GPU-a08301c9-de3d-c3c3-feaa-ba5b41e33dad",
    "GPU-c962070c-2999-ec78-8344-cbf6689a5ced",
    "GPU-41c95a43-ce96-fff3-33e0-739a3931d603",
)
EXPECTED_DATASET_WINDOWS = 34_693
EXPECTED_TASKS = 40
EXPECTED_DRAWS_PER_TASK = 868
EXPECTED_WINDOW_DRAWS = 34_720
EXPECTED_DRAWS_PER_RANK = 4_340
EXPECTED_FINAL_STEP = 4_340
EXPECTED_MIN_TASK_WINDOWS = 536
EXPECTED_MAX_TASK_WINDOWS = 1_602
EXPECTED_MAX_DRAWS_PER_WINDOW = 2
TRAINABLE_CONTRACT = "r11_dit_trunk_adapt_v1"
TRAINABLE_PARAMETER_PATTERNS = (
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
)
DIT_TRUNK_PREFIXES = tuple(
    pattern.removesuffix("*") for pattern in TRAINABLE_PARAMETER_PATTERNS[4:]
)
FROZEN_VIDEO_PREFIXES = (
    "video_backbone.dit.final_layer.",
    "video_backbone.vae.",
    "video_backbone.text_encoder.",
)
EXPECTED_DIT_TRUNK_PARAMETERS = 2_458_986_920
EXPECTED_DIT_TRUNK_TENSORS = 693
EXPECTED_TRAINABLE_PARAMETERS = 3_098_639_983
EXPECTED_TRAINABLE_TENSORS = 1_253


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[str, int]:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"pinned artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest(), metadata.st_size


def _verify_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed, size = _sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch for {path}: {observed}")
    return {"path": str(path), "sha256": observed, "size_bytes": size}


def _tensor_sha256(tensor: Any) -> str:
    """Hash exactly the contiguous tensor storage bytes, without metadata."""
    import torch

    raw = (
        tensor.detach()
        .to(device="cpu")
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def _verify_tensor_sha256(tensor: Any, expected_sha256: str, *, label: str) -> str:
    observed = _tensor_sha256(tensor)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch for {label}: {observed}")
    return observed


def _fsync_directory(path: Path) -> None:
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"directory fsync target is unsafe: {path}")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_json(path: Path, payload: Any, *, mode: int = 0o644) -> str:
    data = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while creating {path}")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return hashlib.sha256(data).hexdigest()


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _verify_source(
    expected_commit: str, expected_runner_sha256: str, expected_config_sha256: str
) -> dict[str, Any]:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("R11 must run from the repository root")
    commit = _git("rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(f"repository HEAD differs: {commit}")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked repository state is dirty")
    gitlink = _git("ls-tree", "HEAD", "third_party/Sana").split()[2]
    if gitlink != SANA_GITLINK:
        raise RuntimeError("Sana gitlink differs")
    sana_head = _git("rev-parse", "HEAD", cwd=ROOT / "third_party" / "Sana")
    if sana_head != SANA_GITLINK:
        raise RuntimeError(f"Sana checked-out HEAD differs: {sana_head}")
    if _git(
        "status",
        "--porcelain",
        "--untracked-files=no",
        cwd=ROOT / "third_party" / "Sana",
    ):
        raise RuntimeError("Sana submodule tracked state is dirty")
    runner = _verify_file(Path(__file__).resolve(), expected_runner_sha256)
    config = _verify_file(ROOT / CONFIG_RELATIVE_PATH, expected_config_sha256)
    fp32_ar_sources = {}
    for relative_path in FP32_AR_SOURCE_PATHS:
        artifact = _verify_file(
            ROOT / relative_path, FP32_AR_SOURCE_SHA256[relative_path]
        )
        fp32_ar_sources[relative_path] = {
            **artifact,
            "inherited_unchanged_from_r8": True,
        }
    return {
        "repo_commit": commit,
        "sana_gitlink": gitlink,
        "sana_head": sana_head,
        "runner": runner,
        "config": config,
        "fp32_ar_operator": {
            "epsilon": AR_LINEAR_ATTN_EPS,
            "config_knob_added": False,
            "source_files": fp32_ar_sources,
        },
    }


def _verify_config():
    from omegaconf import OmegaConf

    from sana_wam.deploy.libero_policy_server import (
        validate_libero_train_deploy_parity,
    )
    from sana_wam.train.libero_contract import validate_libero_training_config

    cfg = OmegaConf.load(ROOT / CONFIG_RELATIVE_PATH)
    validate_libero_training_config(cfg)
    exact = {
        "training.output_dir": str(RUN_ROOT),
        "training.max_steps": EXPECTED_FINAL_STEP,
        "training.formal_final_step": EXPECTED_FINAL_STEP,
        "training.lr_schedule_steps": EXPECTED_FINAL_STEP,
        "training.expected_world_size": 8,
        "training.expected_global_batch_size": 8,
        "training.batch_size": 1,
        "training.gradient_accumulation_steps": 1,
        "training.libero_sampler_contract": "task_balanced_40_v1",
        "training.warmup_steps": 50,
        "training.formal_warm_start": True,
        "training.formal_non_resumable": True,
        "training.init_checkpoint": str(
            PREDECESSOR_R8_ROOT / "checkpoint_step_4337.safetensors"
        ),
        "training.init_checkpoint_sha256": PREDECESSOR_R8_CHECKPOINT_SHA256,
        "training.formal_predecessor_r8_result_sha256": PREDECESSOR_R8_RESULT_SHA256,
        "training.formal_predecessor_r8_checkpoint_sha256": PREDECESSOR_R8_CHECKPOINT_SHA256,
        "training.matched_control_result_sha256": MATCHED_CONTROL_RESULT_SHA256,
        "training.matched_control_checkpoint_sha256": MATCHED_CONTROL_CHECKPOINT_SHA256,
        "training.matched_control_config_sha256": MATCHED_CONTROL_CONFIG_SHA256,
        "training.matched_control_runner_sha256": MATCHED_CONTROL_RUNNER_SHA256,
        "training.matched_control_same40_ab_result_sha256": MATCHED_CONTROL_SAME40_AB_RESULT_SHA256,
        "training.libero_trainable_contract": TRAINABLE_CONTRACT,
        "training.trainable_parameter_patterns": list(TRAINABLE_PARAMETER_PATTERNS),
        "training.eval_modules": ["video_backbone"],
        "training.predecessor_optimizer_rounds": 1,
        "training.balanced_rounds_in_this_run": 1,
        "training.video_lr": 1.5e-5,
        "training.action_lr": 1.5e-5,
        "training.lambda_video": 0.0,
        "training.lambda_action": 1.0,
        "model.video_backbone.model_kwargs.y_norm": True,
        "model.video_backbone.model_kwargs.y_norm_scale_factor": 0.01,
        "model.architecture.action_loss_weighting": "none",
        "model.architecture.ar_attn_window": 1,
        "model.architecture.ar_action_horizon_rope": True,
        "model.architecture.ar_noisy_cond_prob": 0.0,
        "model.architecture.ar_cond_max_ratio": 0.0,
        "dataloader.num_frames": 17,
        "dataloader.action_horizon": 8,
        "dataloader.require_full_action_horizon": True,
        "dataloader.include_terminal_full_horizon": True,
        "dataloader.window_stride": 8,
        "dataloader.video_context_mode": "causal_past",
    }
    for path, expected in exact.items():
        observed = OmegaConf.select(cfg, path, default="__ABSENT__")
        if observed != expected:
            raise RuntimeError(
                f"R11 config requires {path}={expected!r}, got {observed!r}"
            )
    for forbidden in (
        "training.trainable_modules",
        "training.preserve_frozen_input_grad_modules",
        "training.freeze",
        "training.trainable_parameter_dtype",
    ):
        if OmegaConf.select(cfg, forbidden, default="__ABSENT__") != "__ABSENT__":
            raise RuntimeError(f"R11 config forbids {forbidden}")
    if (
        OmegaConf.select(
            cfg, "training.init_checkpoint_allow_missing_patterns", default=None
        )
        is not None
    ):
        raise RuntimeError(
            "R11 requires an exact R8 checkpoint schema with no allowlist"
        )
    if OmegaConf.select(cfg, "model.video_backbone.fp32_attention", default=False):
        raise RuntimeError(
            "R11 FP32-AR inheritance is source-level; the legacy config knob must stay off"
        )
    candidate = OmegaConf.to_container(cfg, resolve=True)
    baseline = OmegaConf.to_container(
        OmegaConf.load(
            ROOT
            / MATCHED_CONTROL_CONFIG_RELATIVE_PATH
        ),
        resolve=True,
    )
    for payload in (candidate, baseline):
        training = payload["training"]
        for key in (
            "output_dir",
            "matched_control_result_sha256",
            "matched_control_checkpoint_sha256",
            "matched_control_config_sha256",
            "matched_control_runner_sha256",
            "matched_control_same40_ab_result_sha256",
            "libero_trainable_contract",
            "trainable_modules",
            "preserve_frozen_input_grad_modules",
            "trainable_parameter_patterns",
            "eval_modules",
        ):
            training.pop(key, None)
    if candidate != baseline:
        raise RuntimeError(
            "R11 config differs from the task-balanced successor1 matched control "
            "outside output root, control binding, and the DiT-trunk trainable contract"
        )
    contract = OmegaConf.to_container(cfg.training.deployment_contract, resolve=True)
    if contract.get("action_steps") != 20 or contract.get("history_len") != 17:
        raise RuntimeError("R11 deployment parity contract differs")
    deploy_cfg = OmegaConf.load(ROOT / "configs/deploy_ar_sana.yaml")
    validate_libero_train_deploy_parity(cfg, OmegaConf.merge(cfg, deploy_cfg))
    return cfg


def _verify_predecessors() -> dict[str, Any]:
    r8_result = _verify_file(
        PREDECESSOR_R8_ROOT / "RESULT.json", PREDECESSOR_R8_RESULT_SHA256
    )
    r8_checkpoint = _verify_file(
        PREDECESSOR_R8_ROOT / "checkpoint_step_4337.safetensors",
        PREDECESSOR_R8_CHECKPOINT_SHA256,
    )
    payload = json.loads(
        (PREDECESSOR_R8_ROOT / "RESULT.json").read_text(encoding="utf-8")
    )
    if (
        payload.get("valid_run") is not True
        or payload.get("execution_result") != "PASS"
        or payload.get("verdict")
        != "LIBERO_AR_R8_H8_YNORM_FP32ATTN_8GPU_EPOCH1_TRAINING_COMPLETE"
        or payload.get("checkpoint", {}).get("sha256")
        != PREDECESSOR_R8_CHECKPOINT_SHA256
    ):
        raise RuntimeError("R11 warm-start R8 RESULT semantics differ")

    control_config = _verify_file(
        ROOT / MATCHED_CONTROL_CONFIG_RELATIVE_PATH, MATCHED_CONTROL_CONFIG_SHA256
    )
    control_runner = _verify_file(
        ROOT / MATCHED_CONTROL_RUNNER_RELATIVE_PATH, MATCHED_CONTROL_RUNNER_SHA256
    )
    control_result = _verify_file(
        MATCHED_CONTROL_ROOT / "RESULT.json", MATCHED_CONTROL_RESULT_SHA256
    )
    control_checkpoint = _verify_file(
        MATCHED_CONTROL_ROOT / "checkpoint_step_4340.safetensors",
        MATCHED_CONTROL_CHECKPOINT_SHA256,
    )
    control_payload = json.loads(
        (MATCHED_CONTROL_ROOT / "RESULT.json").read_text(encoding="utf-8")
    )
    control_training = control_payload.get("training", {})
    control_sampler = control_training.get("task_balanced_sampler", {})
    if (
        control_payload.get("valid_run") is not True
        or control_payload.get("execution_result") != "PASS"
        or control_payload.get("verdict")
        != "LIBERO_AR_R8_TASKBALANCED_H8_8GPU_SUCCESSOR1_TRAINING_COMPLETE"
        or control_payload.get("checkpoint", {}).get("sha256")
        != MATCHED_CONTROL_CHECKPOINT_SHA256
        or control_payload.get("source", {}).get("config", {}).get("sha256")
        != MATCHED_CONTROL_CONFIG_SHA256
        or control_payload.get("source", {}).get("runner", {}).get("sha256")
        != MATCHED_CONTROL_RUNNER_SHA256
        or control_training.get("optimizer_steps") != EXPECTED_FINAL_STEP
        or control_training.get("window_draws") != EXPECTED_WINDOW_DRAWS
        or control_sampler.get("contract") != "task_balanced_40_v1"
        or control_sampler.get("task_count") != EXPECTED_TASKS
        or control_sampler.get("draws_per_task") != EXPECTED_DRAWS_PER_TASK
    ):
        raise RuntimeError("R11 matched task-balanced control semantics differ")

    same40 = _verify_file(
        MATCHED_CONTROL_SAME40_AB_RESULT,
        MATCHED_CONTROL_SAME40_AB_RESULT_SHA256,
    )
    same40_payload = json.loads(
        MATCHED_CONTROL_SAME40_AB_RESULT.read_text(encoding="utf-8")
    )
    same40_aggregate = same40_payload.get("aggregate", {})
    same40_training = same40_payload.get("training", {})
    if (
        same40_payload.get("valid_run") is not True
        or same40_payload.get("execution_result") != "PASS"
        or same40_payload.get("verdict")
        != "LIBERO_R8_TASKBALANCED_SETTLE10_SAME40_COMPLETE"
        or same40_aggregate.get("episodes") != 40
        or same40_aggregate.get("successes") != 6
        or same40_training.get("checkpoint_sha256")
        != MATCHED_CONTROL_CHECKPOINT_SHA256
        or same40_training.get("result_sha256") != MATCHED_CONTROL_RESULT_SHA256
    ):
        raise RuntimeError("R11 matched-control same40 evidence differs from 6/40")
    return {
        "r8_warm_start_result": r8_result,
        "r8_warm_start_checkpoint": r8_checkpoint,
        "matched_taskbalanced_control_config": control_config,
        "matched_taskbalanced_control_runner": control_runner,
        "matched_taskbalanced_control_result": control_result,
        "matched_taskbalanced_control_checkpoint": control_checkpoint,
        "matched_taskbalanced_control_same40_ab": same40,
    }


def _validate_sampler_summary(summary: dict[str, Any]) -> None:
    expected = {
        "contract": "task_balanced_40_v1",
        "task_count": EXPECTED_TASKS,
        "input_window_count": EXPECTED_DATASET_WINDOWS,
        "draws_per_task": EXPECTED_DRAWS_PER_TASK,
        "global_draw_count": EXPECTED_WINDOW_DRAWS,
        "draws_per_rank": EXPECTED_DRAWS_PER_RANK,
        "num_replicas": 8,
        "max_draws_per_window": EXPECTED_MAX_DRAWS_PER_WINDOW,
        "window_draw_multiplicity_counts": {"0": 4_353, "1": 25_960, "2": 4_380},
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise RuntimeError(
                f"R11 task-balanced sampler {key} differs: {summary.get(key)!r}"
            )
    task_counts = summary.get("task_window_counts")
    repeat_caps = summary.get("task_max_draws_per_window")
    if not isinstance(task_counts, dict) or len(task_counts) != EXPECTED_TASKS:
        raise RuntimeError("R11 task-balanced task counts are absent")
    if min(task_counts.values()) != EXPECTED_MIN_TASK_WINDOWS:
        raise RuntimeError("R11 minimum task window count differs")
    if max(task_counts.values()) != EXPECTED_MAX_TASK_WINDOWS:
        raise RuntimeError("R11 maximum task window count differs")
    if not isinstance(repeat_caps, dict) or set(repeat_caps) != set(task_counts):
        raise RuntimeError("R11 task-balanced repeat-cap map differs")
    if max(repeat_caps.values()) != EXPECTED_MAX_DRAWS_PER_WINDOW:
        raise RuntimeError("R11 task-balanced repeat cap exceeds two")


def _verify_materialized_r8_warm_start(trainer: Any) -> dict[str, Any]:
    """Prove R11 loaded the complete R8 checkpoint with an exact schema."""
    from safetensors import safe_open

    checkpoint_path = PREDECESSOR_R8_ROOT / "checkpoint_step_4337.safetensors"
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
    architecture_keys = set(trainer.architecture.state_dict().keys())
    missing = sorted(architecture_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - architecture_keys)
    if missing or unexpected:
        raise RuntimeError(
            "R11 strict R8 warm-start schema differs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "strict_load": True,
        "allow_missing_patterns": [],
        "observed_missing_keys": missing,
        "observed_unexpected_keys": unexpected,
        "r8_checkpoint_loaded_exactly": True,
        "source_checkpoint_sha256": PREDECESSOR_R8_CHECKPOINT_SHA256,
        "architecture_state_tensor_count": len(architecture_keys),
    }


def _verify_materialized_fp32_ar_operator(trainer: Any) -> dict[str, Any]:
    """Bind the materialized trainer to the inherited R11 FP32 train/cache operator."""
    import torch

    from sana_wam.model.ar.sana_ar_linear_attn import (
        AR_LINEAR_ATTN_EPS as observed_eps,
        _ar_attention_compute_dtype,
    )
    from sana_wam.model.ar.sana_ar_mot_driver import SanaARMoTJointDriver

    driver = getattr(trainer.architecture, "_mot_driver", None)
    if not isinstance(driver, SanaARMoTJointDriver):
        raise RuntimeError("R11 trainer did not materialize SanaARMoTJointDriver")
    if observed_eps != AR_LINEAR_ATTN_EPS or driver.eps != AR_LINEAR_ATTN_EPS:
        raise RuntimeError(
            "R11 train/cache epsilon differs: "
            f"module={observed_eps!r}, driver={driver.eps!r}"
        )
    dtype_map = {
        str(dtype): str(_ar_attention_compute_dtype(dtype))
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
    }
    expected_dtype_map = {
        "torch.float16": "torch.float32",
        "torch.bfloat16": "torch.float32",
        "torch.float32": "torch.float32",
        "torch.float64": "torch.float64",
    }
    if dtype_map != expected_dtype_map:
        raise RuntimeError(f"R11 FP32-AR compute dtype map differs: {dtype_map}")
    return {
        "enabled": True,
        "source_level_unconditional": True,
        "legacy_config_knob_enabled": False,
        "driver_class": type(driver).__name__,
        "epsilon": driver.eps,
        "compute_dtype_map": dtype_map,
        "low_precision_output_dtype_restored": True,
        "cache_state_dtype": "torch.float32",
        "training_and_cache_operator_shared": True,
    }


def _composite_parameter_digest(
    named_parameters: list[tuple[str, Any]],
) -> str:
    """Hash names, shapes, dtypes, and exact tensor bytes in stable order."""
    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        detached = parameter.detach()
        if detached.is_meta:
            raise RuntimeError(f"R11 parameter is not materialized: {name}")
        metadata = {
            "name": name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
        }
        digest.update(_canonical_json_bytes(metadata))
        raw = (
            detached.to(device="cpu")
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        digest.update(raw)
    return digest.hexdigest()


def _verify_materialized_trainable_partition(trainer: Any) -> dict[str, Any]:
    """Prove R11 opens only action/proprio plus the six DiT-trunk subtrees."""
    trainer._set_training_mode()
    video_backbone = trainer.architecture.video_backbone
    non_eval_video_modules = [
        name or "<video_backbone>"
        for name, module in video_backbone.named_modules()
        if module.training
    ]
    if non_eval_video_modules:
        raise RuntimeError(
            "R11 video backbone must remain entirely in eval mode: "
            f"{non_eval_video_modules[:20]}"
        )

    named_parameters = dict(trainer.architecture.named_parameters())
    trainable_names = tuple(
        name for name, parameter in named_parameters.items() if parameter.requires_grad
    )
    if (
        len(trainable_names) != EXPECTED_TRAINABLE_TENSORS
        or sum(named_parameters[name].numel() for name in trainable_names)
        != EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("R11 trainable parameter partition differs")

    match_counts = {pattern: 0 for pattern in TRAINABLE_PARAMETER_PATTERNS}
    for name in trainable_names:
        matches = [
            pattern
            for pattern in TRAINABLE_PARAMETER_PATTERNS
            if fnmatch.fnmatchcase(name, pattern)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"R11 trainable parameter must match exactly one pattern: {name}"
            )
        match_counts[matches[0]] += 1
    if any(count == 0 for count in match_counts.values()):
        raise RuntimeError(f"R11 trainable pattern matched no tensor: {match_counts}")

    dit_trunk_parameters = [
        (name, parameter)
        for name, parameter in named_parameters.items()
        if any(name.startswith(prefix) for prefix in DIT_TRUNK_PREFIXES)
    ]
    if (
        len(dit_trunk_parameters) != EXPECTED_DIT_TRUNK_TENSORS
        or sum(parameter.numel() for _, parameter in dit_trunk_parameters)
        != EXPECTED_DIT_TRUNK_PARAMETERS
        or not all(parameter.requires_grad for _, parameter in dit_trunk_parameters)
    ):
        raise RuntimeError("R11 materialized DiT-trunk trainability/count differs")
    selected_video_names = {name for name, _ in dit_trunk_parameters}
    expected_selected_video_names = {
        name
        for name in named_parameters
        if name.startswith("video_backbone.dit.")
        and not name.startswith("video_backbone.dit.final_layer.")
    }
    if selected_video_names != expected_selected_video_names:
        raise RuntimeError(
            "R11 selected video parameters are not exactly all DiT parameters "
            "excluding final_layer"
        )

    optimizer_groups = trainer._param_groups()
    optimizer_source_parameters = [
        parameter for group in optimizer_groups for parameter in group["params"]
    ]
    trainable_parameters = [named_parameters[name] for name in trainable_names]
    optimizer_source_ids = [id(parameter) for parameter in optimizer_source_parameters]
    trainable_ids = [id(parameter) for parameter in trainable_parameters]
    if (
        len(optimizer_source_ids) != len(set(optimizer_source_ids))
        or set(optimizer_source_ids) != set(trainable_ids)
    ):
        raise RuntimeError("R11 optimizer source parameter identities differ")

    block_digests = {}
    for index in range(20):
        prefix = f"video_backbone.dit.blocks.{index}."
        block_parameters = [
            (name, parameter)
            for name, parameter in dit_trunk_parameters
            if name.startswith(prefix)
        ]
        if not block_parameters:
            raise RuntimeError(f"R11 DiT block {index} parameter partition is empty")
        block_digests[str(index)] = _composite_parameter_digest(block_parameters)

    forbidden_video_trainables = [
        name
        for name, parameter in named_parameters.items()
        if any(name.startswith(prefix) for prefix in FROZEN_VIDEO_PREFIXES)
        and parameter.requires_grad
    ]
    if forbidden_video_trainables:
        raise RuntimeError(
            "R11 opened final_layer/VAE/Gemma parameters: "
            f"{forbidden_video_trainables[:20]}"
        )

    frozen_partitions = {}
    for label, prefix in (
        ("final_layer", "video_backbone.dit.final_layer."),
        ("vae", "video_backbone.vae."),
        ("text_encoder", "video_backbone.text_encoder."),
    ):
        parameters = [
            (name, parameter)
            for name, parameter in named_parameters.items()
            if name.startswith(prefix)
        ]
        if not parameters or any(parameter.requires_grad for _, parameter in parameters):
            raise RuntimeError(f"R11 frozen {label} partition differs")
        frozen_partitions[label] = {
            "prefix": prefix,
            "names": [name for name, _ in parameters],
            "tensor_count": len(parameters),
            "parameter_count": sum(parameter.numel() for _, parameter in parameters),
            "all_require_grad_false": True,
            "composite_sha256": _composite_parameter_digest(parameters),
        }

    named_buffers = list(trainer.architecture.named_buffers())
    if not named_buffers:
        raise RuntimeError("R11 architecture buffer partition is empty")
    buffer_manifest = {
        "names": [name for name, _ in named_buffers],
        "tensor_count": len(named_buffers),
        "element_count": sum(buffer.numel() for _, buffer in named_buffers),
        "composite_sha256": _composite_parameter_digest(named_buffers),
    }

    return {
        "contract": TRAINABLE_CONTRACT,
        "patterns": list(TRAINABLE_PARAMETER_PATTERNS),
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_count": sum(
            named_parameters[name].numel() for name in trainable_names
        ),
        "trainable_names": list(trainable_names),
        "trainable_names_sha256": hashlib.sha256(
            _canonical_json_bytes(list(trainable_names))
        ).hexdigest(),
        "video_backbone_eval": True,
        "all_video_submodules_eval": True,
        "optimizer_source_parameter_ids_exact": True,
        "optimizer_group_tensor_counts": [
            len(group["params"]) for group in optimizer_groups
        ],
        "optimizer_group_lrs": [float(group["lr"]) for group in optimizer_groups],
        "dit_trunk": {
            "prefixes": list(DIT_TRUNK_PREFIXES),
            "names": [name for name, _ in dit_trunk_parameters],
            "tensor_count": len(dit_trunk_parameters),
            "parameter_count": sum(
                parameter.numel() for _, parameter in dit_trunk_parameters
            ),
            "all_require_grad": True,
            "equals_all_dit_named_parameters_excluding_final_layer": True,
            "composite_sha256": _composite_parameter_digest(dit_trunk_parameters),
            "block_composite_sha256": block_digests,
        },
        "frozen_partitions": frozen_partitions,
        "buffers": buffer_manifest,
        "pattern_tensor_counts": match_counts,
    }


def _verify_saved_checkpoint_partitions(
    path: Path,
    *,
    dit_trunk_names: list[str],
    frozen_names: dict[str, list[str]],
    buffer_names: list[str],
    expected_dit_trunk_sha256: str,
    expected_frozen_sha256: dict[str, str],
    expected_buffers_sha256: str,
) -> dict[str, Any]:
    """Bind the saved checkpoint to changed trunk and unchanged frozen roots."""
    import torch
    from safetensors import safe_open

    def _digest_keys(checkpoint: Any, keys: list[str]) -> dict[str, Any]:
        digest = hashlib.sha256()
        parameter_count = 0
        for name in sorted(keys):
            tensor = checkpoint.get_tensor(name)
            if not bool(torch.isfinite(tensor).all().item()):
                raise RuntimeError(f"R11 checkpoint contains non-finite tensor: {name}")
            digest.update(
                _canonical_json_bytes(
                    {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                )
            )
            digest.update(
                tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
            parameter_count += tensor.numel()
        return {
            "tensor_count": len(keys),
            "parameter_count": parameter_count,
            "finite": True,
            "composite_sha256": digest.hexdigest(),
        }

    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        required = set(dit_trunk_names)
        for names in frozen_names.values():
            required.update(names)
        required.update(buffer_names)
        missing = sorted(required - keys)
        if missing:
            raise RuntimeError(f"R11 checkpoint omitted partition tensors: {missing[:20]}")
        trunk = _digest_keys(checkpoint, dit_trunk_names)
        frozen = {}
        for label, names in frozen_names.items():
            frozen[label] = _digest_keys(checkpoint, names)
        buffers = _digest_keys(checkpoint, buffer_names)

    if (
        trunk["tensor_count"] != EXPECTED_DIT_TRUNK_TENSORS
        or trunk["parameter_count"] != EXPECTED_DIT_TRUNK_PARAMETERS
        or trunk["composite_sha256"] != expected_dit_trunk_sha256
    ):
        raise RuntimeError("R11 saved DiT-trunk digest/count differs from runtime")
    for label, expected in expected_frozen_sha256.items():
        if frozen.get(label, {}).get("composite_sha256") != expected:
            raise RuntimeError(f"R11 saved frozen {label} digest changed")
    if buffers["composite_sha256"] != expected_buffers_sha256:
        raise RuntimeError("R11 saved architecture buffer digest changed")
    return {"dit_trunk": trunk, "frozen_partitions": frozen, "buffers": buffers}


def _query_gpus(require_idle: bool) -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    records = []
    for raw in output.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row: {raw}")
        index, uuid, memory_used, utilization, name = fields
        record = {
            "index": int(index),
            "uuid": uuid,
            "memory_used_mib": int(memory_used),
            "utilization_percent": int(utilization),
            "name": name,
        }
        records.append(record)
    if len(records) != 8 or tuple(record["uuid"] for record in records) != GPU_UUIDS:
        raise RuntimeError("physical GPU index/UUID topology differs")
    if require_idle and any(record["memory_used_mib"] != 0 for record in records):
        raise RuntimeError(f"one or more H200 GPUs are not idle: {records}")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()
    if require_idle and processes:
        raise RuntimeError(f"unexpected GPU compute processes: {processes}")
    return records


@contextmanager
def _gpu_lock(uuid: str):
    path = f"/tmp/sana-wam-{uuid}.lock"
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"GPU lock is not a regular file: {path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield path
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validate_real_directory(path: Path) -> None:
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"run-root ancestor is unsafe: {path}")


def _prepare_root_parent() -> None:
    """Create only declared namespaces without following an unsafe ancestor."""

    fixed_ancestors = (Path("/DATA"), Path("/DATA/share"))
    for ancestor in fixed_ancestors:
        _validate_real_directory(ancestor)
        _fsync_directory(ancestor)

    namespace_paths = (
        Path("/DATA/share/sana_wam_libero_training"),
        Path(
            "/DATA/share/sana_wam_libero_training/"
            "formal_taskbalanced_r11_dit_trunk_adapt"
        ),
    )
    for namespace in namespace_paths:
        _validate_real_directory(namespace.parent)
        try:
            os.mkdir(namespace, 0o2775)
        except FileExistsError:
            _validate_real_directory(namespace)
        else:
            os.chmod(namespace, 0o2775)
            _fsync_directory(namespace)
            _fsync_directory(namespace.parent)
        _fsync_directory(namespace)
    if RUN_ROOT.parent != namespace_paths[-1]:
        raise RuntimeError("R11 root parent differs from its declared namespace")
    if os.path.lexists(RUN_ROOT):
        raise FileExistsError(f"R11 root already exists: {RUN_ROOT}")


def _freeze_regular_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"terminal root contains unsafe file: {path}")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _freeze_directory(path: Path, mode: int) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"terminal root contains unsafe directory: {path}")
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _freeze_root_contents() -> None:
    """Freeze materialized contents while deliberately leaving root at 0700."""

    _validate_real_directory(RUN_ROOT)
    if stat.S_IMODE(os.lstat(RUN_ROOT).st_mode) != 0o700:
        raise RuntimeError("R11 root must remain mode 0700 before terminal publication")
    for directory, subdirs, files in os.walk(RUN_ROOT, topdown=False):
        base = Path(directory)
        for name in files:
            child = base / name
            metadata = os.lstat(child)
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"terminal root contains unsafe file: {child}")
            _freeze_regular_file(child)
        for name in subdirs:
            child = base / name
            metadata = os.lstat(child)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"terminal root contains unsafe directory: {child}")
            _freeze_directory(child, 0o555)
        _fsync_directory(base)


def _verify_frozen_contents(*, expected_root_mode: int) -> None:
    root_metadata = os.lstat(RUN_ROOT)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != expected_root_mode
    ):
        raise RuntimeError(
            f"R11 root mode differs from {expected_root_mode:04o} during verification"
        )
    for directory, subdirs, files in os.walk(RUN_ROOT, topdown=False):
        base = Path(directory)
        for name in files:
            child = base / name
            metadata = os.lstat(child)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o444
            ):
                raise RuntimeError(f"R11 file is not frozen to 0444: {child}")
        for name in subdirs:
            child = base / name
            metadata = os.lstat(child)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o555
            ):
                raise RuntimeError(f"R11 directory is not frozen to 0555: {child}")


def _seal_root(expected_terminal: str) -> None:
    _freeze_directory(RUN_ROOT, 0o555)
    _fsync_directory(RUN_ROOT.parent)
    metadata = os.lstat(RUN_ROOT)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o555:
        raise RuntimeError("R11 terminal root did not freeze to mode 0555")
    terminal = RUN_ROOT / expected_terminal
    terminal_metadata = os.lstat(terminal)
    if (
        not stat.S_ISREG(terminal_metadata.st_mode)
        or stat.S_ISLNK(terminal_metadata.st_mode)
        or stat.S_IMODE(terminal_metadata.st_mode) != 0o444
    ):
        raise RuntimeError(f"R11 terminal artifact is not frozen: {terminal}")
    _verify_frozen_contents(expected_root_mode=0o555)


def _publish_success_result(payload: Any) -> str:
    if os.path.lexists(RUN_ROOT / "RESULT.json") or os.path.lexists(
        RUN_ROOT / "FAILED.json"
    ):
        raise FileExistsError("R11 terminal artifact already exists")
    _freeze_root_contents()
    _verify_frozen_contents(expected_root_mode=0o700)
    result_sha256 = _exclusive_json(RUN_ROOT / "RESULT.json", payload, mode=0o444)
    _seal_root("RESULT.json")
    return result_sha256


def _discard_uncommitted_result() -> None:
    result_path = RUN_ROOT / "RESULT.json"
    if not os.path.lexists(result_path):
        return
    _freeze_directory(RUN_ROOT, 0o700)
    metadata = os.lstat(result_path)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError("uncommitted R11 RESULT is unsafe")
    os.unlink(result_path)
    _fsync_directory(RUN_ROOT)


def _publish_failure(payload: Any) -> None:
    _validate_real_directory(RUN_ROOT)
    if stat.S_IMODE(os.lstat(RUN_ROOT).st_mode) != 0o700:
        _freeze_directory(RUN_ROOT, 0o700)
    _exclusive_json(RUN_ROOT / "FAILED.json", payload)
    _freeze_root_contents()
    _seal_root("FAILED.json")


def _process_group_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeError(
            f"cannot inspect torchrun process group {process_group_id}"
        ) from exc
    return True


def _wait_process_group_gone(process_group_id: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_group_alive(process_group_id):
            return True
        time.sleep(0.1)
    return not _process_group_alive(process_group_id)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    """Terminate and reap the isolated torchrun session before root finalization."""

    process_group_id = process.pid
    if process_group_id <= 1 or process_group_id == os.getpgrp():
        raise RuntimeError("refusing to terminate a non-isolated process group")
    if _process_group_alive(process_group_id):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGTERM)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            pass
        if not _wait_process_group_gone(process_group_id, 2.0):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process_group_id, signal.SIGKILL)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
            if not _wait_process_group_gone(process_group_id, 10.0):
                raise RuntimeError(
                    f"torchrun process group {process_group_id} survived SIGKILL"
                )
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("torchrun supervisor process was not reaped") from exc


@contextmanager
def _supervisor_signal_guard():
    previous_handlers: dict[int, Any] = {}

    def _raise_interrupted(signum, _frame):
        raise InterruptedError(f"R11 supervisor received signal {signum}")

    for signum in (signal.SIGHUP, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _raise_interrupted)
    try:
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _worker(args: argparse.Namespace) -> None:
    if os.environ.get("SANA_WAM_R11_DIT_TRUNK_SUPERVISED_WORKER") != "1":
        raise RuntimeError("R11 worker may only be launched by its supervisor")
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 8 or rank not in range(8) or local_rank not in range(8):
        raise RuntimeError("R11 worker rank environment differs")
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s %(levelname)s rank={rank} local_rank={local_rank} "
            "%(name)s: %(message)s"
        ),
        force=True,
    )
    worker_logger = logging.getLogger("sana_wam.r11_dit_trunk.worker")

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=dt.timedelta(minutes=30))
    try:
        worker_logger.info("NCCL process group initialized; beginning R11 preflight")

        def _fail_if_any_rank(stage: str, local_error: str | None) -> None:
            gathered_errors: list[Any] = [None] * world_size
            dist.all_gather_object(
                gathered_errors, {"rank": rank, "error": local_error}
            )
            failures = [item for item in gathered_errors if item["error"] is not None]
            if failures:
                raise RuntimeError(
                    f"R11 {stage} failed on one or more ranks: "
                    + json.dumps(failures, sort_keys=True)
                )

        cfg = None
        local_error = None
        try:
            _verify_source(
                args.expected_repo_commit,
                args.expected_runner_sha256,
                args.expected_config_sha256,
            )
            cfg = _verify_config()
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _fail_if_any_rank("source/config preflight", local_error)
        if cfg is None:
            raise RuntimeError(
                "R11 config missing after successful collective preflight"
            )

        error = [None]
        if rank == 0:
            try:
                from sana_wam.train.libero_contract import (
                    validate_libero_training_config,
                )

                validate_libero_training_config(cfg, require_materialized_stats=True)
            except BaseException as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f"rank0 materialized-data preflight failed: {error[0]}")

        from sana_wam.train.trainer import Trainer

        trainer = Trainer(cfg, exact_output_dir=str(RUN_ROOT))
        warm_start_integrity = None
        fp32_ar_operator = None
        trainable_partition = None
        local_error = None
        try:
            if len(trainer.dataset) != EXPECTED_DATASET_WINDOWS:
                raise RuntimeError("R11 dataset window count differs")
            trainable_partition = _verify_materialized_trainable_partition(trainer)
            warm_start_integrity = _verify_materialized_r8_warm_start(trainer)
            fp32_ar_operator = _verify_materialized_fp32_ar_operator(trainer)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _fail_if_any_rank("materialized trainer preflight", local_error)
        worker_logger.info("R11 preflight complete; entering formal training")
        started = time.monotonic()
        output = trainer.train()
        elapsed = time.monotonic() - started
        summary = None
        local_error = None
        try:
            if output != str(RUN_ROOT):
                raise RuntimeError("Trainer returned a different exact root")
            summary = dict(trainer.formal_libero_run_summary)
            if summary.get("optimizer_steps") != EXPECTED_FINAL_STEP:
                raise RuntimeError("Trainer final-step summary differs")
            if summary.get("dataset_windows") != EXPECTED_DATASET_WINDOWS:
                raise RuntimeError("Trainer dataset-window summary differs")
            if summary.get("window_draws") != EXPECTED_WINDOW_DRAWS:
                raise RuntimeError("Trainer task-balanced draw count differs")
            _validate_sampler_summary(summary.get("task_balanced_sampler", {}))
            if (
                warm_start_integrity is None
                or fp32_ar_operator is None
                or trainable_partition is None
            ):
                raise RuntimeError("R11 warm-start/operator/trainable proof is absent")
            _verify_materialized_r8_warm_start(trainer)
            post_training_operator = _verify_materialized_fp32_ar_operator(trainer)
            post_training_partition = _verify_materialized_trainable_partition(trainer)
            if post_training_operator != fp32_ar_operator:
                raise RuntimeError(
                    "R11 FP32-AR operator contract changed during training"
                )
            if (
                post_training_partition["trainable_names_sha256"]
                != trainable_partition["trainable_names_sha256"]
                or post_training_partition["trainable_tensor_count"]
                != EXPECTED_TRAINABLE_TENSORS
                or post_training_partition["trainable_parameter_count"]
                != EXPECTED_TRAINABLE_PARAMETERS
            ):
                raise RuntimeError("R11 trainable manifest changed during training")
            if (
                post_training_partition["dit_trunk"]["composite_sha256"]
                == trainable_partition["dit_trunk"]["composite_sha256"]
            ):
                raise RuntimeError("R11 DiT-trunk composite did not update")
            unchanged_blocks = [
                index
                for index, initial_digest in trainable_partition["dit_trunk"][
                    "block_composite_sha256"
                ].items()
                if post_training_partition["dit_trunk"]["block_composite_sha256"].get(
                    index
                )
                == initial_digest
            ]
            if unchanged_blocks:
                raise RuntimeError(
                    f"R11 one or more DiT blocks did not update: {unchanged_blocks}"
                )
            for label, initial in trainable_partition["frozen_partitions"].items():
                observed = post_training_partition["frozen_partitions"][label]
                if observed["composite_sha256"] != initial["composite_sha256"]:
                    raise RuntimeError(f"R11 frozen {label} changed during training")
            if (
                post_training_partition["buffers"]["composite_sha256"]
                != trainable_partition["buffers"]["composite_sha256"]
            ):
                raise RuntimeError("R11 architecture buffers changed during training")
            summary["warm_start_integrity"] = warm_start_integrity
            summary["fp32_ar_operator"] = fp32_ar_operator
            summary["trainable_partition"] = {
                **post_training_partition,
                "initial_dit_trunk_composite_sha256": trainable_partition["dit_trunk"][
                    "composite_sha256"
                ],
                "dit_trunk_composite_changed": True,
                "all_20_block_composites_changed": True,
                "frozen_partitions_unchanged": True,
                "buffers_unchanged": True,
            }
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _fail_if_any_rank("post-training summary preflight", local_error)
        if summary is None:
            raise RuntimeError(
                "R11 summary missing after successful collective preflight"
            )
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, {"rank": rank, "summary": summary})
        decision: list[Any] = [None]
        if rank == 0:
            try:
                canonical = [
                    _canonical_json_bytes(item["summary"]) for item in gathered
                ]
                if len(set(canonical)) != 1:
                    raise RuntimeError("R11 rank summaries differ")
                worker_result_sha256 = _exclusive_json(
                    RUN_ROOT / "WORKER_RESULT.json",
                    {
                        "schema_version": "sana-wam-libero-r11-dit-trunk-worker-result-v1",
                        "execution_result": "PASS",
                        "world_size": world_size,
                        "rank_consensus": True,
                        "elapsed_seconds_rank0": elapsed,
                        "rank_summaries": gathered,
                    },
                )
                decision[0] = {
                    "ok": True,
                    "worker_result_sha256": worker_result_sha256,
                }
            except Exception as exc:
                decision[0] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(decision, src=0)
        if not decision[0]["ok"]:
            raise RuntimeError(
                f"rank0 worker-result commit failed: {decision[0]['error']}"
            )
        dist.barrier()
        worker_logger.info("R11 worker completed with rank-consistent terminal summary")
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def _supervise(args: argparse.Namespace) -> None:
    source = _verify_source(
        args.expected_repo_commit,
        args.expected_runner_sha256,
        args.expected_config_sha256,
    )
    _verify_config()
    predecessors = _verify_predecessors()
    root_created = False
    child_process: subprocess.Popen[Any] | None = None
    success_committed = False
    attempt_sha256 = None
    started_at = _utc_now()
    started = None
    lock_stack = ExitStack()
    try:
        with _supervisor_signal_guard():
            lock_paths = [
                lock_stack.enter_context(_gpu_lock(uuid)) for uuid in GPU_UUIDS
            ]
            gpu_before = _query_gpus(require_idle=True)
            _prepare_root_parent()
            os.mkdir(RUN_ROOT, 0o700)
            root_created = True
            os.chmod(RUN_ROOT, 0o700)
            _fsync_directory(RUN_ROOT)
            _fsync_directory(RUN_ROOT.parent)
            attempt_sha256 = _exclusive_json(
                RUN_ROOT / "ATTEMPT.json",
                {
                    "schema_version": "sana-wam-libero-r11-dit-trunk-attempt-v1",
                    "state": "RUNNING_NON_RESUMABLE_WARM_START",
                    "created_at_utc": started_at,
                    "run": {"root": str(RUN_ROOT), "nonce": RUN_NONCE},
                    "source": source,
                    "predecessors": predecessors,
                    "gpu": {"devices": gpu_before, "lock_paths": lock_paths},
                    "training": {
                        "world_size": 8,
                        "global_batch_size": 8,
                        "optimizer_steps": EXPECTED_FINAL_STEP,
                        "dataset_windows": EXPECTED_DATASET_WINDOWS,
                        "window_draws": EXPECTED_WINDOW_DRAWS,
                        "sampler_contract": "task_balanced_40_v1",
                        "task_count": EXPECTED_TASKS,
                        "draws_per_task": EXPECTED_DRAWS_PER_TASK,
                        "draws_per_rank": EXPECTED_DRAWS_PER_RANK,
                        "warm_start_checkpoint_sha256": PREDECESSOR_R8_CHECKPOINT_SHA256,
                        "init_checkpoint_allow_missing_patterns": [],
                        "strict_r8_checkpoint_schema": True,
                        "optimizer_and_lr_state": "fresh",
                        "learning_rate": 1.5e-5,
                        "warmup_steps": 50,
                        "action_loss_weighting": "none",
                        "resumable": False,
                    },
                    "treatment": {
                        "name": "r11_dit_trunk_adapt",
                        "candidate_axis": "action_facing_dit_trunk_trainability",
                        "matched_taskbalanced_successor1_control": True,
                        "contract": TRAINABLE_CONTRACT,
                        "patterns": list(TRAINABLE_PARAMETER_PATTERNS),
                        "video_backbone_eval": True,
                        "dit_trunk_tensor_count": EXPECTED_DIT_TRUNK_TENSORS,
                        "dit_trunk_parameter_count": EXPECTED_DIT_TRUNK_PARAMETERS,
                        "final_layer_vae_gemma_frozen": True,
                        "matched_control": {
                            "result_sha256": MATCHED_CONTROL_RESULT_SHA256,
                            "checkpoint_sha256": MATCHED_CONTROL_CHECKPOINT_SHA256,
                            "same40_ab_result_sha256": MATCHED_CONTROL_SAME40_AB_RESULT_SHA256,
                            "same40_successes": 6,
                            "same40_episodes": 40,
                        },
                        "inherited_r8_contract": {
                            "fp32_ar_attention_operator": True,
                            "epsilon": AR_LINEAR_ATTN_EPS,
                            "y_norm": True,
                        },
                    },
                    "deploy_gate": {
                        "history_len": 17,
                        "action_steps": 20,
                        "temporal_alignment": (
                            "causal_past_video_0_to_future_action_1"
                        ),
                        "fresh_cache_each_generation": True,
                        "action_horizon": 8,
                        "anchor_stride": 8,
                        "action_horizon_rope": "unique_0_to_7",
                    },
                },
            )
            log_path = RUN_ROOT / "TRAIN.log"
            log_descriptor = os.open(
                log_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o644,
            )
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=8",
                "--max-restarts=0",
                str(Path(__file__).resolve()),
                "worker",
                "--expected-repo-commit",
                args.expected_repo_commit,
                "--expected-runner-sha256",
                args.expected_runner_sha256,
                "--expected-config-sha256",
                args.expected_config_sha256,
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                    "NCCL_NVLS_ENABLE": "0",
                    "GDN_DISABLE_COMPILE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                    "SANA_WAM_R11_DIT_TRUNK_SUPERVISED_WORKER": "1",
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                }
            )
            started = time.monotonic()
            with os.fdopen(log_descriptor, "wb", buffering=0) as log_stream:
                child_process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                returncode = child_process.wait()
            training_seconds = time.monotonic() - started
            child_process_group = child_process.pid
            if returncode != 0:
                _terminate_process_group(child_process)
                child_process = None
                raise RuntimeError(f"torchrun failed with exit code {returncode}")
            if not _wait_process_group_gone(child_process_group, 5.0):
                _terminate_process_group(child_process)
                child_process = None
                raise RuntimeError(
                    "torchrun supervisor exited while worker processes remained alive"
                )
            child_process = None

            worker_result_path = RUN_ROOT / "WORKER_RESULT.json"
            worker_result_sha256, _ = _sha256_file(worker_result_path)
            worker_result = json.loads(worker_result_path.read_text(encoding="utf-8"))
            if (
                worker_result.get("execution_result") != "PASS"
                or worker_result.get("world_size") != 8
                or worker_result.get("rank_consensus") is not True
                or len(worker_result.get("rank_summaries", [])) != 8
            ):
                raise RuntimeError("worker result is not PASS")
            checkpoint_path = (
                RUN_ROOT / f"checkpoint_step_{EXPECTED_FINAL_STEP}.safetensors"
            )
            checkpoints = sorted(RUN_ROOT.glob("checkpoint_step_*.safetensors"))
            if checkpoints != [checkpoint_path]:
                raise RuntimeError(f"unexpected R11 checkpoint set: {checkpoints}")
            checkpoint_sha256, checkpoint_size = _sha256_file(checkpoint_path)
            if checkpoint_size < 1_000_000_000:
                raise RuntimeError("R11 checkpoint is unexpectedly small")
            stats_sha256, stats_size = _sha256_file(RUN_ROOT / "action_stats.npy")
            if stats_sha256 != STATS_SHA256:
                raise RuntimeError("R11 saved action stats differ")
            saved_config_sha256, saved_config_size = _sha256_file(
                RUN_ROOT / "config.yaml"
            )
            gpu_after = _query_gpus(require_idle=False)
            summary = worker_result["rank_summaries"][0]["summary"]
            _validate_sampler_summary(summary.get("task_balanced_sampler", {}))
            warm_start_integrity = summary.get("warm_start_integrity", {})
            if (
                warm_start_integrity.get("r8_checkpoint_loaded_exactly") is not True
                or warm_start_integrity.get("allow_missing_patterns") != []
                or warm_start_integrity.get("observed_missing_keys") != []
                or warm_start_integrity.get("observed_unexpected_keys") != []
                or warm_start_integrity.get("source_checkpoint_sha256")
                != PREDECESSOR_R8_CHECKPOINT_SHA256
            ):
                raise RuntimeError("R11 worker strict R8 warm-start proof differs")
            fp32_ar_operator = summary.get("fp32_ar_operator", {})
            if (
                fp32_ar_operator.get("enabled") is not True
                or fp32_ar_operator.get("source_level_unconditional") is not True
                or fp32_ar_operator.get("legacy_config_knob_enabled") is not False
                or fp32_ar_operator.get("epsilon") != AR_LINEAR_ATTN_EPS
                or fp32_ar_operator.get("cache_state_dtype") != "torch.float32"
                or fp32_ar_operator.get("low_precision_output_dtype_restored")
                is not True
                or fp32_ar_operator.get("training_and_cache_operator_shared")
                is not True
            ):
                raise RuntimeError("R11 worker FP32-AR operator proof differs")
            trainable_partition = summary.get("trainable_partition", {})
            dit_trunk_partition = trainable_partition.get("dit_trunk", {})
            if (
                trainable_partition.get("contract") != TRAINABLE_CONTRACT
                or trainable_partition.get("patterns")
                != list(TRAINABLE_PARAMETER_PATTERNS)
                or trainable_partition.get("trainable_tensor_count")
                != EXPECTED_TRAINABLE_TENSORS
                or trainable_partition.get("trainable_parameter_count")
                != EXPECTED_TRAINABLE_PARAMETERS
                or trainable_partition.get("video_backbone_eval") is not True
                or trainable_partition.get("all_video_submodules_eval") is not True
                or trainable_partition.get("optimizer_source_parameter_ids_exact")
                is not True
                or dit_trunk_partition.get("tensor_count")
                != EXPECTED_DIT_TRUNK_TENSORS
                or dit_trunk_partition.get("parameter_count")
                != EXPECTED_DIT_TRUNK_PARAMETERS
                or dit_trunk_partition.get("all_require_grad") is not True
                or dit_trunk_partition.get(
                    "equals_all_dit_named_parameters_excluding_final_layer"
                )
                is not True
                or trainable_partition.get("dit_trunk_composite_changed") is not True
                or trainable_partition.get("all_20_block_composites_changed") is not True
                or trainable_partition.get("frozen_partitions_unchanged") is not True
                or trainable_partition.get("buffers_unchanged") is not True
                or trainable_partition.get("initial_dit_trunk_composite_sha256")
                == dit_trunk_partition.get("composite_sha256")
                or len(dit_trunk_partition.get("block_composite_sha256", {})) != 20
            ):
                raise RuntimeError("R11 worker DiT-trunk materialized proof differs")
            frozen_partitions = trainable_partition.get("frozen_partitions", {})
            if set(frozen_partitions) != {"final_layer", "vae", "text_encoder"}:
                raise RuntimeError("R11 frozen partition manifest differs")
            saved_partitions = _verify_saved_checkpoint_partitions(
                checkpoint_path,
                dit_trunk_names=dit_trunk_partition.get("names", []),
                frozen_names={
                    label: partition.get("names", [])
                    for label, partition in frozen_partitions.items()
                },
                buffer_names=trainable_partition.get("buffers", {}).get("names", []),
                expected_dit_trunk_sha256=dit_trunk_partition.get(
                    "composite_sha256", ""
                ),
                expected_frozen_sha256={
                    label: partition.get("composite_sha256", "")
                    for label, partition in frozen_partitions.items()
                },
                expected_buffers_sha256=trainable_partition.get("buffers", {}).get(
                    "composite_sha256", ""
                ),
            )
            result = {
                "schema_version": "sana-wam-libero-r11-dit-trunk-result-v1",
                "valid_run": True,
                "verdict": "LIBERO_AR_R11_DIT_TRUNK_TASKBALANCED_8GPU_SUCCESSOR1_TRAINING_COMPLETE",
                "execution_result": "PASS",
                "created_at_utc": _utc_now(),
                "run": {"root": str(RUN_ROOT), "nonce": RUN_NONCE},
                "attempt_sha256": attempt_sha256,
                "source": source,
                "predecessors": predecessors,
                "training": {
                    **summary,
                    "warm_start": True,
                    "warm_start_checkpoint_sha256": PREDECESSOR_R8_CHECKPOINT_SHA256,
                    "optimizer_and_lr_state": "fresh",
                    "warm_start_r8_optimizer_steps": 4_337,
                    "balanced_rounds_in_this_run": 1,
                    "r11_dit_trunk_optimizer_steps": EXPECTED_FINAL_STEP,
                    "action_scheduler_train_timesteps": 1000,
                    "action_scheduler_shift": 5.0,
                    "action_loss_weighting": "none",
                    "task_balanced_sampler_contract": "task_balanced_40_v1",
                    "window_draws": EXPECTED_WINDOW_DRAWS,
                    "full_dataset_epoch_claimed": False,
                    "training_seconds": training_seconds,
                    "resumable": False,
                },
                "ynorm_contract": {
                    "enabled": True,
                    "scale_factor": 0.01,
                    "inherited_from_r8": True,
                    "strict_warm_start": True,
                    "source_checkpoint_sha256": PREDECESSOR_R8_CHECKPOINT_SHA256,
                    "part_of_trainable_dit_trunk": True,
                },
                "fp32_ar_operator": {
                    **fp32_ar_operator,
                    "source_files": source["fp32_ar_operator"]["source_files"],
                },
                "dit_trunk_adapt": trainable_partition,
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_sha256,
                    "size_bytes": checkpoint_size,
                    "saved_config_sha256": saved_config_sha256,
                    "saved_config_size_bytes": saved_config_size,
                    "action_stats_sha256": stats_sha256,
                    "action_stats_size_bytes": stats_size,
                    "materialized_partition_proof": saved_partitions,
                },
                "worker_result_sha256": worker_result_sha256,
                "eight_rank_consensus": {
                    "world_size": 8,
                    "rank_summaries_equal": True,
                },
                "gpu": {"before": gpu_before, "after": gpu_after},
                "train_deploy_parity": {
                    "causal_single_chunk_contract_gated": True,
                    "history_len": 17,
                    "action_steps": 20,
                    "action_scheduler_shift": 5.0,
                    "temporal_alignment": ("causal_past_video_0_to_future_action_1"),
                    "ar_obs_chunk_mode": "rolling_buffer",
                    "ar_obs_latent_band": "auto",
                    "ar_reset_cache_each_generation": True,
                    "ar_proprio_mode": "per_step",
                    "cache_feedback_mode": "predicted",
                    "context_proprio_source": "current",
                    "chunk_proprio_source": "current",
                    "video_context_mode": "causal_past",
                    "video_num_frames": 5,
                    "video_chunks_per_window": 1,
                    "action_horizon": 8,
                    "anchor_stride": 8,
                    "full_horizon_only": True,
                    "action_tokens_per_chunk": 8,
                    "action_horizon_rope": "unique_0_to_7",
                    "y_norm": True,
                    "y_norm_scale_factor": 0.01,
                    "fp32_ar_attention_operator": True,
                    "fp32_ar_attention_epsilon": AR_LINEAR_ATTN_EPS,
                    "fp32_ar_cache_state_dtype": "torch.float32",
                    "legacy_fp32_attention_config_knob": False,
                    "video_backbone_eval_in_training": True,
                    "video_backbone_eval_in_deployment": True,
                    "same_dit_trunk_modules_in_train_and_deploy": True,
                    "checkpoint_contains_adapted_dit_trunk": True,
                },
                "scientific_scope": {
                    "establishes": (
                        "one R8-warm-start task-balanced round jointly updating "
                        "the inherited action/proprio roots and all action-facing "
                        "Video DiT parameters except final_layer"
                    ),
                    "does_not_establish": (
                        "closed-loop policy improvement or value-distribution equivalence"
                    ),
                    "candidate_axis": "action_facing_dit_trunk_trainability",
                    "matched_taskbalanced_successor1_control": True,
                    "matched_control_same40_successes": 6,
                    "matched_control_same40_episodes": 40,
                    "prior_action_teacher_forcing_present": False,
                    "future_video_conditioning_present": False,
                    "absolute_position_extrapolation_present": False,
                    "closed_loop_success_evaluated": False,
                    "training_pass_is_not_policy_success": True,
                    "video_world_model_validated": False,
                },
            }
            _publish_success_result(result)
            success_committed = True
            with contextlib.suppress(BrokenPipeError):
                print(
                    json.dumps(
                        {
                            "result": "PASS",
                            "root": str(RUN_ROOT),
                            "checkpoint_sha256": checkpoint_sha256,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    except BaseException as error:
        if child_process is not None:
            try:
                _terminate_process_group(child_process)
            except BaseException as termination_error:
                raise RuntimeError(
                    "torchrun process group could not be terminated; refusing to "
                    "write or freeze a terminal artifact"
                ) from termination_error
            finally:
                child_process = None
        if root_created and not success_committed:
            if os.path.lexists(RUN_ROOT / "RESULT.json"):
                _discard_uncommitted_result()
            if os.path.lexists(RUN_ROOT / "FAILED.json"):
                raise RuntimeError("R11 FAILED artifact already exists") from error
            _publish_failure(
                {
                    "schema_version": "sana-wam-libero-r11-dit-trunk-failure-v1",
                    "valid_run": False,
                    "verdict": "FAILED_CLOSED_NON_RESUMABLE",
                    "created_at_utc": _utc_now(),
                    "run": {"root": str(RUN_ROOT), "nonce": RUN_NONCE},
                    "attempt_sha256": attempt_sha256,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc()[-16_000:],
                    "training_started": started is not None,
                    "automatic_rerun_permitted": False,
                }
            )
        raise
    finally:
        lock_stack.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("supervise", "worker"):
        child = subparsers.add_parser(mode)
        child.add_argument("--expected-repo-commit", required=True)
        child.add_argument("--expected-runner-sha256", required=True)
        child.add_argument("--expected-config-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "worker":
        _worker(args)
    else:
        _supervise(args)


if __name__ == "__main__":
    main()
