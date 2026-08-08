#!/usr/bin/env python
"""Fail-closed 8-GPU R8 warm-start task-balanced successor scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import stat
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

CONFIG_RELATIVE_PATH = (
    "configs/experiments/libero_formal_r8_taskbalanced_8gpu_successor1.yaml"
)
BASELINE_CONFIG_RELATIVE_PATH = (
    "configs/experiments/libero_formal_r8_h8_ynorm_fp32attn_8gpu_epoch1.yaml"
)
RUN_NONCE = "aa10ecdb611a0b0b61abe0b162a66227"
RUN_ROOT = Path(
    "/DATA/share/sana_wam_libero_training/"
    "formal_taskbalanced_r8_h8_ynorm_fp32attn/"
    "libero-ar-r8-taskbalanced-causal1-warmstart-8gpu-4340-" + RUN_NONCE
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
SANA_GITLINK = "16b9cec673e3335724ba2d8db25de7f9ed229292"
EXPECTED_DATASET_WINDOWS = 34_693
EXPECTED_TASKS = 40
EXPECTED_DRAWS_PER_TASK = 868
EXPECTED_WINDOW_DRAWS = 34_720
EXPECTED_DRAWS_PER_RANK = 4_340
EXPECTED_FINAL_STEP = 4_340
EXPECTED_MIN_TASK_WINDOWS = 536
EXPECTED_MAX_TASK_WINDOWS = 1_602
EXPECTED_MAX_DRAWS_PER_WINDOW = 2
EXPECTED_TRAINABLE_PARAMETERS = 639_653_063
EXPECTED_TRAINABLE_TENSORS = 560
WORKER_MARKER = "SANA_WAM_R8_TASKBALANCED_SUPERVISED_WORKER"


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


def _exclusive_json(path: Path, payload: Any) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    data = _canonical_json_bytes(payload)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(data).hexdigest()


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def _load_and_verify_config():
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
        "training.init_checkpoint": str(
            PREDECESSOR_R8_ROOT / "checkpoint_step_4337.safetensors"
        ),
        "training.init_checkpoint_sha256": PREDECESSOR_R8_CHECKPOINT_SHA256,
        "training.formal_predecessor_r8_result_sha256": (
            PREDECESSOR_R8_RESULT_SHA256
        ),
        "training.formal_predecessor_r8_checkpoint_sha256": (
            PREDECESSOR_R8_CHECKPOINT_SHA256
        ),
        "training.formal_warm_start": True,
        "training.formal_non_resumable": True,
        "training.predecessor_optimizer_rounds": 1,
        "training.balanced_rounds_in_this_run": 1,
        "training.optimizer_master_weights": True,
        "training.optimizer_foreach": False,
        "training.video_lr": 1.5e-5,
        "training.action_lr": 1.5e-5,
        "training.lambda_video": 0.0,
        "training.lambda_action": 1.0,
    }
    for path, expected in exact.items():
        observed = OmegaConf.select(cfg, path, default="__ABSENT__")
        if observed != expected:
            raise RuntimeError(
                f"task-balanced config requires {path}={expected!r}, got {observed!r}"
            )
    if OmegaConf.select(
        cfg, "training.init_checkpoint_allow_missing_patterns", default=None
    ) is not None:
        raise RuntimeError("task-balanced successor requires strict R8 checkpoint load")

    candidate = OmegaConf.to_container(cfg, resolve=True)
    baseline = OmegaConf.to_container(
        OmegaConf.load(ROOT / BASELINE_CONFIG_RELATIVE_PATH), resolve=True
    )
    removable = (
        "output_dir",
        "max_steps",
        "formal_final_step",
        "lr_schedule_steps",
        "formal_predecessor_r7_result_sha256",
        "formal_predecessor_r7_checkpoint_sha256",
        "formal_predecessor_r8_result_sha256",
        "formal_predecessor_r8_checkpoint_sha256",
        "init_checkpoint",
        "init_checkpoint_sha256",
        "cumulative_epochs_after_run",
        "epochs_in_this_run",
        "predecessor_optimizer_rounds",
        "balanced_rounds_in_this_run",
        "libero_sampler_contract",
    )
    for payload in (candidate, baseline):
        training = payload["training"]
        for key in removable:
            training.pop(key, None)
    if candidate != baseline:
        raise RuntimeError(
            "task-balanced config differs from R8 outside lineage, step count, "
            "output root, and sampler contract"
        )
    deploy_cfg = OmegaConf.load(ROOT / "configs/deploy_ar_sana.yaml")
    validate_libero_train_deploy_parity(cfg, OmegaConf.merge(cfg, deploy_cfg))
    return cfg


def _verify_source(args: argparse.Namespace) -> dict[str, Any]:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("task-balanced run must start at the repository root")
    commit = _git("rev-parse", "HEAD")
    if commit != args.expected_repo_commit:
        raise RuntimeError(f"repository commit differs: {commit}")
    gitlink = _git("rev-parse", "HEAD:third_party/Sana")
    sana_head = _git("rev-parse", "HEAD", cwd=ROOT / "third_party" / "Sana")
    if gitlink != SANA_GITLINK or sana_head != SANA_GITLINK:
        raise RuntimeError("Sana gitlink/worktree differs")
    runner = _verify_file(Path(__file__).resolve(), args.expected_runner_sha256)
    config = _verify_file(
        ROOT / CONFIG_RELATIVE_PATH, args.expected_config_sha256
    )
    return {
        "repo_commit": commit,
        "sana_gitlink": gitlink,
        "runner": runner,
        "config": config,
    }


def _verify_predecessor() -> dict[str, Any]:
    result_path = PREDECESSOR_R8_ROOT / "RESULT.json"
    checkpoint_path = PREDECESSOR_R8_ROOT / "checkpoint_step_4337.safetensors"
    result = _verify_file(result_path, PREDECESSOR_R8_RESULT_SHA256)
    checkpoint = _verify_file(checkpoint_path, PREDECESSOR_R8_CHECKPOINT_SHA256)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        payload.get("valid_run") is not True
        or payload.get("execution_result") != "PASS"
        or payload.get("verdict")
        != "LIBERO_AR_R8_H8_YNORM_FP32ATTN_8GPU_EPOCH1_TRAINING_COMPLETE"
        or payload.get("checkpoint", {}).get("sha256")
        != PREDECESSOR_R8_CHECKPOINT_SHA256
    ):
        raise RuntimeError("R8 predecessor RESULT semantics differ")
    return {"result": result, "checkpoint": checkpoint}


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
        "window_draw_multiplicity_counts": {
            "0": 4_353,
            "1": 25_960,
            "2": 4_380,
        },
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise RuntimeError(
                f"task-balanced sampler summary {key} differs: {summary.get(key)!r}"
            )
    task_counts = summary.get("task_window_counts")
    repeat_caps = summary.get("task_max_draws_per_window")
    if not isinstance(task_counts, dict) or len(task_counts) != EXPECTED_TASKS:
        raise RuntimeError("task-balanced sampler task counts are absent")
    if min(task_counts.values()) != EXPECTED_MIN_TASK_WINDOWS:
        raise RuntimeError("minimum task window count differs")
    if max(task_counts.values()) != EXPECTED_MAX_TASK_WINDOWS:
        raise RuntimeError("maximum task window count differs")
    if not isinstance(repeat_caps, dict) or set(repeat_caps) != set(task_counts):
        raise RuntimeError("task-balanced repeat-cap map differs")
    if max(repeat_caps.values()) != EXPECTED_MAX_DRAWS_PER_WINDOW:
        raise RuntimeError("task-balanced repeat cap exceeds two")


def _worker(args: argparse.Namespace) -> None:
    if os.environ.get(WORKER_MARKER) != "1":
        raise RuntimeError("worker may only be launched by its supervisor")
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        cfg = _load_and_verify_config()
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
            raise RuntimeError(f"materialized-data preflight failed: {error[0]}")

        from sana_wam.train.libero_contract import validate_libero_training_config
        from sana_wam.train.trainer import Trainer

        trainer = Trainer(cfg, exact_output_dir=str(RUN_ROOT))
        validate_libero_training_config(cfg, dataset=trainer.dataset)
        trainables = [
            parameter
            for parameter in trainer.architecture.parameters()
            if parameter.requires_grad
        ]
        if (
            len(trainables) != EXPECTED_TRAINABLE_TENSORS
            or sum(parameter.numel() for parameter in trainables)
            != EXPECTED_TRAINABLE_PARAMETERS
        ):
            raise RuntimeError("R8 trainable partition differs")
        output = trainer.train()
        if Path(output).resolve() != RUN_ROOT.resolve():
            raise RuntimeError("Trainer returned a different output root")
        summary = dict(trainer.formal_libero_run_summary)
        if summary.get("optimizer_steps") != EXPECTED_FINAL_STEP:
            raise RuntimeError("optimizer-step summary differs")
        if summary.get("window_draws") != EXPECTED_WINDOW_DRAWS:
            raise RuntimeError("global window-draw summary differs")
        _validate_sampler_summary(summary.get("task_balanced_sampler", {}))
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, summary)
        canonical = _canonical_json_bytes(summary)
        if any(_canonical_json_bytes(item) != canonical for item in gathered):
            raise RuntimeError("rank summaries differ")
        if rank == 0:
            _exclusive_json(
                RUN_ROOT / "WORKER_RESULT.json",
                {"schema_version": "sana-wam-libero-taskbalanced-worker-v1", "training": summary},
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _freeze_root() -> None:
    for directory, subdirs, files in os.walk(RUN_ROOT, topdown=False):
        current = Path(directory)
        for name in files:
            os.chmod(current / name, 0o444)
        for name in subdirs:
            os.chmod(current / name, 0o555)
        os.chmod(current, 0o555)


def _supervise(args: argparse.Namespace) -> None:
    source = _verify_source(args)
    _load_and_verify_config()
    predecessor = _verify_predecessor()
    if os.path.lexists(RUN_ROOT):
        raise FileExistsError(f"run root already exists: {RUN_ROOT}")
    RUN_ROOT.parent.mkdir(mode=0o775, parents=True, exist_ok=True)
    os.mkdir(RUN_ROOT, 0o700)
    attempt = {
        "schema_version": "sana-wam-libero-taskbalanced-attempt-v1",
        "run": {"root": str(RUN_ROOT), "nonce": RUN_NONCE},
        "source": source,
        "predecessor": predecessor,
        "sampler_contract": {
            "contract": "task_balanced_40_v1",
            "tasks": EXPECTED_TASKS,
            "draws_per_task": EXPECTED_DRAWS_PER_TASK,
            "global_draws": EXPECTED_WINDOW_DRAWS,
            "draws_per_rank": EXPECTED_DRAWS_PER_RANK,
            "shortest_task_max_draws_per_window": EXPECTED_MAX_DRAWS_PER_WINDOW,
            "full_dataset_epoch_claimed": False,
            "window_draw_multiplicity_counts": {
                "0": 4_353,
                "1": 25_960,
                "2": 4_380,
            },
        },
    }
    attempt_sha256 = _exclusive_json(RUN_ROOT / "ATTEMPT.json", attempt)
    env = dict(os.environ)
    env.update(
        {
            WORKER_MARKER: "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "--max-restarts=0",
        str(Path(__file__).resolve()),
        "--worker",
        "--expected-repo-commit",
        args.expected_repo_commit,
        "--expected-runner-sha256",
        args.expected_runner_sha256,
        "--expected-config-sha256",
        args.expected_config_sha256,
    ]
    try:
        with (RUN_ROOT / "TRAIN.log").open("xb") as stream:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        if completed.returncode != 0:
            raise RuntimeError(f"torchrun exited {completed.returncode}")
        worker_path = RUN_ROOT / "WORKER_RESULT.json"
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        summary = worker.get("training", {})
        _validate_sampler_summary(summary.get("task_balanced_sampler", {}))
        checkpoint_path = RUN_ROOT / f"checkpoint_step_{EXPECTED_FINAL_STEP}.safetensors"
        checkpoint_sha256, checkpoint_size = _sha256_file(checkpoint_path)
        result = {
            "schema_version": "sana-wam-libero-r8-taskbalanced-result-v1",
            "valid_run": True,
            "execution_result": "PASS",
            "verdict": "LIBERO_AR_R8_TASKBALANCED_H8_8GPU_SUCCESSOR1_TRAINING_COMPLETE",
            "run": {"root": str(RUN_ROOT), "nonce": RUN_NONCE},
            "attempt_sha256": attempt_sha256,
            "source": source,
            "predecessor": predecessor,
            "training": {
                **summary,
                "balanced_rounds_in_this_run": 1,
                "full_dataset_epoch_claimed": False,
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "size_bytes": checkpoint_size,
            },
        }
        _exclusive_json(RUN_ROOT / "RESULT.json", result)
    except BaseException as exc:
        if not os.path.lexists(RUN_ROOT / "RESULT.json"):
            _exclusive_json(
                RUN_ROOT / "FAILED.json",
                {
                    "schema_version": "sana-wam-libero-taskbalanced-failure-v1",
                    "valid_run": False,
                    "execution_result": "FAIL",
                    "verdict": "FAILED_CLOSED_NON_RESUMABLE",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    finally:
        _freeze_root()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if args.worker:
        _worker(args)
    else:
        _supervise(args)


if __name__ == "__main__":
    main()
