#!/usr/bin/env python
"""Fail-closed 8-GPU supervisor/worker for the R6/H8 LIBERO architecture epoch."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
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

CONFIG_RELATIVE_PATH = "configs/experiments/libero_formal_r6_h8_8gpu_epoch1.yaml"
RUN_NONCE = "d21821529d8eba3e5027f85d624715df"
RUN_ROOT = Path(
    "/DATA/share/sana_wam_libero_training/formal_epoch1_r6_h8/"
    "libero-ar-r6-h8-causal1-warmstart-uniform-8gpu-4337-"
    + RUN_NONCE
)
PREDECESSOR_R5_ROOT = Path(
    "/DATA/share/sana_wam_libero_training/formal_epoch2_r5/"
    "libero-ar-r5-causal1-successor-uniform-8gpu-1315-"
    "b9c0140b5431ac7f9bad96efd5a03192"
)
PREDECESSOR_R5_RESULT_SHA256 = "0883d28e6a9661d21c80ca662bede3e83d88b33fc9b5fdd36261d3f653b10681"
PREDECESSOR_R5_CHECKPOINT_SHA256 = (
    "92f44591d3ac06eb81b3dad3112100d6aeede9f882b14435c573ca30cfe3e3c3"
)
FORMAL40_RESULT = Path(
    "/DATA/share/sana_wam_libero_evaluation/r5_causal1_epoch2_formal40/"
    "libero-r5-causal1-epoch2-formal40-a9bf7e50396a46799456c4316d5dfca5/FORMAL_RESULT.json"
)
FORMAL40_RESULT_SHA256 = (
    "bda92bb80f78f29bdba8b1583c99b45bed1ec682479a432845d73679968c43aa"
)
K8_AB_RESULT = Path(
    "/DATA/share/sana_wam_libero_evaluation/r5_causal1_k8_ab40/"
    "libero-r5-causal1-k8-ab40-r1-2ae5d8e27de3a4af8c62e614ff962280/AB_RESULT.json"
)
K8_AB_RESULT_SHA256 = (
    "bc216f43351c78093cb941617b1c03c1ff576c174d7a04a95da28b790637efae"
)

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
EXPECTED_WINDOW_DRAWS = 34_696
EXPECTED_SAMPLER_PADDING_REPEATS = 3
EXPECTED_FINAL_STEP = 4_337
EXPECTED_TRAINABLE_PARAMETERS = 639_653_063
EXPECTED_TRAINABLE_TENSORS = 560


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
        raise RuntimeError("R5 must run from the repository root")
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
    if _git("status", "--porcelain", "--untracked-files=no", cwd=ROOT / "third_party" / "Sana"):
        raise RuntimeError("Sana submodule tracked state is dirty")
    runner = _verify_file(Path(__file__).resolve(), expected_runner_sha256)
    config = _verify_file(ROOT / CONFIG_RELATIVE_PATH, expected_config_sha256)
    return {
        "repo_commit": commit,
        "sana_gitlink": gitlink,
        "sana_head": sana_head,
        "runner": runner,
        "config": config,
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
        "training.warmup_steps": 100,
        "training.formal_warm_start": True,
        "training.formal_non_resumable": True,
        "training.init_checkpoint": str(PREDECESSOR_R5_ROOT / "checkpoint_step_1315.safetensors"),
        "training.init_checkpoint_sha256": PREDECESSOR_R5_CHECKPOINT_SHA256,
        "training.formal_predecessor_r5_result_sha256": PREDECESSOR_R5_RESULT_SHA256,
        "training.formal_predecessor_r5_checkpoint_sha256": PREDECESSOR_R5_CHECKPOINT_SHA256,
        "training.scientific_predecessor_epoch2_formal40_result_sha256": FORMAL40_RESULT_SHA256,
        "training.scientific_predecessor_k8_ab_result_sha256": K8_AB_RESULT_SHA256,
        "training.cumulative_epochs_after_run": 1,
        "training.epochs_in_this_run": 1,
        "training.video_lr": 3.0e-5,
        "training.action_lr": 3.0e-5,
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
            raise RuntimeError(f"R5 config requires {path}={expected!r}, got {observed!r}")
    contract = OmegaConf.to_container(cfg.training.deployment_contract, resolve=True)
    if contract.get("action_steps") != 20 or contract.get("history_len") != 17:
        raise RuntimeError("R5 deployment parity contract differs")
    deploy_cfg = OmegaConf.load(ROOT / "configs/deploy_ar_sana.yaml")
    validate_libero_train_deploy_parity(cfg, OmegaConf.merge(cfg, deploy_cfg))
    return cfg


def _verify_predecessors() -> dict[str, Any]:
    result = _verify_file(
        PREDECESSOR_R5_ROOT / "RESULT.json", PREDECESSOR_R5_RESULT_SHA256
    )
    checkpoint = _verify_file(
        PREDECESSOR_R5_ROOT / "checkpoint_step_1315.safetensors",
        PREDECESSOR_R5_CHECKPOINT_SHA256,
    )
    payload = json.loads(
        (PREDECESSOR_R5_ROOT / "RESULT.json").read_text(encoding="utf-8")
    )
    if (
        payload.get("valid_run") is not True
        or payload.get("execution_result") != "PASS"
        or payload.get("verdict")
        != "LIBERO_AR_R5_8GPU_EPOCH2_SUCCESSOR_TRAINING_COMPLETE"
        or payload.get("checkpoint", {}).get("sha256")
        != PREDECESSOR_R5_CHECKPOINT_SHA256
    ):
        raise RuntimeError("R5-R2 predecessor RESULT semantics differ")
    formal40 = _verify_file(FORMAL40_RESULT, FORMAL40_RESULT_SHA256)
    formal40_payload = json.loads(FORMAL40_RESULT.read_text(encoding="utf-8"))
    aggregate = formal40_payload.get("aggregate", {})
    if (
        formal40_payload.get("valid_run") is not True
        or formal40_payload.get("execution_result") != "PASS"
        or aggregate.get("episodes") != 40
        or aggregate.get("successes") != 2
        or formal40_payload.get("training", {}).get("checkpoint_sha256")
        != PREDECESSOR_R5_CHECKPOINT_SHA256
    ):
        raise RuntimeError("R5 epoch2 formal40 predecessor semantics differ from 2/40")
    k8_ab = _verify_file(K8_AB_RESULT, K8_AB_RESULT_SHA256)
    return {
        "r5_epoch2_result": result,
        "r5_epoch2_checkpoint": checkpoint,
        "scientific_epoch2_formal40": formal40,
        "scientific_k8_ab": k8_ab,
    }


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
        Path("/DATA/share/sana_wam_libero_training/formal_epoch1_r6_h8"),
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
        raise RuntimeError("R5 root parent differs from its declared namespace")
    if os.path.lexists(RUN_ROOT):
        raise FileExistsError(f"R5 root already exists: {RUN_ROOT}")


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
        raise RuntimeError("R5 root must remain mode 0700 before terminal publication")
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
            f"R5 root mode differs from {expected_root_mode:04o} during verification"
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
                raise RuntimeError(f"R5 file is not frozen to 0444: {child}")
        for name in subdirs:
            child = base / name
            metadata = os.lstat(child)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o555
            ):
                raise RuntimeError(f"R5 directory is not frozen to 0555: {child}")


def _seal_root(expected_terminal: str) -> None:
    _freeze_directory(RUN_ROOT, 0o555)
    _fsync_directory(RUN_ROOT.parent)
    metadata = os.lstat(RUN_ROOT)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o555:
        raise RuntimeError("R5 terminal root did not freeze to mode 0555")
    terminal = RUN_ROOT / expected_terminal
    terminal_metadata = os.lstat(terminal)
    if (
        not stat.S_ISREG(terminal_metadata.st_mode)
        or stat.S_ISLNK(terminal_metadata.st_mode)
        or stat.S_IMODE(terminal_metadata.st_mode) != 0o444
    ):
        raise RuntimeError(f"R5 terminal artifact is not frozen: {terminal}")
    _verify_frozen_contents(expected_root_mode=0o555)


def _publish_success_result(payload: Any) -> str:
    if os.path.lexists(RUN_ROOT / "RESULT.json") or os.path.lexists(
        RUN_ROOT / "FAILED.json"
    ):
        raise FileExistsError("R5 terminal artifact already exists")
    _freeze_root_contents()
    _verify_frozen_contents(expected_root_mode=0o700)
    result_sha256 = _exclusive_json(
        RUN_ROOT / "RESULT.json", payload, mode=0o444
    )
    _seal_root("RESULT.json")
    return result_sha256


def _discard_uncommitted_result() -> None:
    result_path = RUN_ROOT / "RESULT.json"
    if not os.path.lexists(result_path):
        return
    _freeze_directory(RUN_ROOT, 0o700)
    metadata = os.lstat(result_path)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError("uncommitted R5 RESULT is unsafe")
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
        raise InterruptedError(f"R5 supervisor received signal {signum}")

    for signum in (signal.SIGHUP, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _raise_interrupted)
    try:
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _worker(args: argparse.Namespace) -> None:
    if os.environ.get("SANA_WAM_R6_H8_SUPERVISED_WORKER") != "1":
        raise RuntimeError("R5 worker may only be launched by its supervisor")
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 8 or rank not in range(8) or local_rank not in range(8):
        raise RuntimeError("R5 worker rank environment differs")
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s %(levelname)s rank={rank} local_rank={local_rank} "
            "%(name)s: %(message)s"
        ),
        force=True,
    )
    worker_logger = logging.getLogger("sana_wam.r5.worker")

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=dt.timedelta(minutes=30))
    try:
        worker_logger.info("NCCL process group initialized; beginning R5 preflight")

        def _fail_if_any_rank(stage: str, local_error: str | None) -> None:
            gathered_errors: list[Any] = [None] * world_size
            dist.all_gather_object(
                gathered_errors, {"rank": rank, "error": local_error}
            )
            failures = [item for item in gathered_errors if item["error"] is not None]
            if failures:
                raise RuntimeError(
                    f"R5 {stage} failed on one or more ranks: "
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
            raise RuntimeError("R5 config missing after successful collective preflight")

        error = [None]
        if rank == 0:
            try:
                from sana_wam.train.libero_contract import validate_libero_training_config

                validate_libero_training_config(cfg, require_materialized_stats=True)
            except BaseException as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f"rank0 materialized-data preflight failed: {error[0]}")

        from sana_wam.train.trainer import Trainer

        trainer = Trainer(cfg, exact_output_dir=str(RUN_ROOT))
        local_error = None
        try:
            if len(trainer.dataset) != EXPECTED_DATASET_WINDOWS:
                raise RuntimeError("R5 dataset window count differs")
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
                raise RuntimeError("R5 trainable parameter partition differs")
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _fail_if_any_rank("materialized trainer preflight", local_error)
        worker_logger.info("R5 preflight complete; entering formal training")
        started = time.monotonic()
        output = trainer.train()
        elapsed = time.monotonic() - started
        summary = None
        local_error = None
        try:
            if output != str(RUN_ROOT):
                raise RuntimeError("Trainer returned a different exact root")
            summary = trainer.formal_libero_run_summary
            if summary.get("optimizer_steps") != EXPECTED_FINAL_STEP:
                raise RuntimeError("Trainer final-step summary differs")
            if summary.get("dataset_windows") != EXPECTED_DATASET_WINDOWS:
                raise RuntimeError("Trainer dataset-window summary differs")
            if summary.get("window_draws") != EXPECTED_WINDOW_DRAWS:
                raise RuntimeError("Trainer distributed-sampler draw count differs")
            if EXPECTED_WINDOW_DRAWS - EXPECTED_DATASET_WINDOWS != EXPECTED_SAMPLER_PADDING_REPEATS:
                raise RuntimeError("Declared DistributedSampler padding repeats differ")
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _fail_if_any_rank("post-training summary preflight", local_error)
        if summary is None:
            raise RuntimeError("R5 summary missing after successful collective preflight")
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, {"rank": rank, "summary": summary})
        decision: list[Any] = [None]
        if rank == 0:
            try:
                canonical = [
                    _canonical_json_bytes(item["summary"]) for item in gathered
                ]
                if len(set(canonical)) != 1:
                    raise RuntimeError("R5 rank summaries differ")
                worker_result_sha256 = _exclusive_json(
                    RUN_ROOT / "WORKER_RESULT.json",
                    {
                        "schema_version": "sana-wam-libero-r6-h8-worker-result-v1",
                        "execution_result": "PASS",
                        "world_size": world_size,
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
            raise RuntimeError(f"rank0 worker-result commit failed: {decision[0]['error']}")
        dist.barrier()
        worker_logger.info("R5 worker completed with rank-consistent terminal summary")
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
                    "schema_version": "sana-wam-libero-r6-h8-attempt-v1",
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
                        "distributed_sampler_padding_repeats": EXPECTED_SAMPLER_PADDING_REPEATS,
                        "warm_start_checkpoint_sha256": PREDECESSOR_R5_CHECKPOINT_SHA256,
                        "optimizer_and_lr_state": "fresh",
                        "action_loss_weighting": "none",
                        "resumable": False,
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
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
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
                    "SANA_WAM_R6_H8_SUPERVISED_WORKER": "1",
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
            if worker_result.get("execution_result") != "PASS":
                raise RuntimeError("worker result is not PASS")
            checkpoint_path = RUN_ROOT / f"checkpoint_step_{EXPECTED_FINAL_STEP}.safetensors"
            checkpoints = sorted(RUN_ROOT.glob("checkpoint_step_*.safetensors"))
            if checkpoints != [checkpoint_path]:
                raise RuntimeError(f"unexpected R5 checkpoint set: {checkpoints}")
            checkpoint_sha256, checkpoint_size = _sha256_file(checkpoint_path)
            if checkpoint_size < 1_000_000_000:
                raise RuntimeError("R5 checkpoint is unexpectedly small")
            stats_sha256, stats_size = _sha256_file(RUN_ROOT / "action_stats.npy")
            if stats_sha256 != STATS_SHA256:
                raise RuntimeError("R5 saved action stats differ")
            saved_config_sha256, saved_config_size = _sha256_file(
                RUN_ROOT / "config.yaml"
            )
            gpu_after = _query_gpus(require_idle=False)
            summary = worker_result["rank_summaries"][0]["summary"]
            result = {
                "schema_version": "sana-wam-libero-r6-h8-result-v1",
                "valid_run": True,
                "verdict": "LIBERO_AR_R6_H8_8GPU_EPOCH1_TRAINING_COMPLETE",
                "execution_result": "PASS",
                "created_at_utc": _utc_now(),
                "run": {"root": str(RUN_ROOT), "nonce": RUN_NONCE},
                "attempt_sha256": attempt_sha256,
                "source": source,
                "predecessors": predecessors,
                "training": {
                    **summary,
                    "warm_start": True,
                    "warm_start_checkpoint_sha256": PREDECESSOR_R5_CHECKPOINT_SHA256,
                    "optimizer_and_lr_state": "fresh",
                    "cumulative_epochs_after_run": 1,
                    "epochs_in_this_run": 1,
                    "predecessor_r5_h28_optimizer_steps": 2630,
                    "r6_h8_uniform_optimizer_steps": EXPECTED_FINAL_STEP,
                    "action_scheduler_train_timesteps": 1000,
                    "action_scheduler_shift": 5.0,
                    "action_loss_weighting": "none",
                    "sampler_epochs_this_run": 1.0,
                    "window_draws": EXPECTED_WINDOW_DRAWS,
                    "distributed_sampler_padding_repeats": EXPECTED_SAMPLER_PADDING_REPEATS,
                    "sampler_epochs_cumulative": 1.0,
                    "training_seconds": training_seconds,
                    "resumable": False,
                },
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_sha256,
                    "size_bytes": checkpoint_size,
                    "saved_config_sha256": saved_config_sha256,
                    "saved_config_size_bytes": saved_config_size,
                    "action_stats_sha256": stats_sha256,
                    "action_stats_size_bytes": stats_size,
                },
                "worker_result_sha256": worker_result_sha256,
                "gpu": {"before": gpu_before, "after": gpu_after},
                "train_deploy_parity": {
                    "causal_single_chunk_contract_gated": True,
                    "history_len": 17,
                    "action_steps": 20,
                    "action_scheduler_shift": 5.0,
                    "temporal_alignment": (
                        "causal_past_video_0_to_future_action_1"
                    ),
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
                },
                "scientific_scope": {
                    "establishes": (
                        "causal single-chunk implementation parity and full action-noise support"
                    ),
                    "does_not_establish": (
                        "expert-observation to closed-loop-observation value-distribution equivalence"
                    ),
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
                raise RuntimeError("R5 FAILED artifact already exists") from error
            _publish_failure(
                {
                    "schema_version": "sana-wam-libero-r6-h8-failure-v1",
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
