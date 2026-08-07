#!/usr/bin/env python
"""Run the one-shot single-GPU LIBERO AR formal2000 training endpoint."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import stat
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

CONFIG_RELATIVE_PATH = "configs/experiments/libero_formal_single_gpu_2000.yaml"
RUN_NONCE = "e21c6e5ad5910c00156499472f46bee6"
RUN_ROOT = Path(
    "/DATA/share/sana_wam_libero_training/formal2000/"
    "libero-ar-formal2000-r3-single-gpu-2000-e21c6e5ad5910c00156499472f46bee6"
)
EXPECTED_SANA_COMMIT = "16b9cec673e3335724ba2d8db25de7f9ed229292"
EXPECTED_GPU_INDEX = 7
EXPECTED_GPU_UUID = "GPU-41c95a43-ce96-fff3-33e0-739a3931d603"
EXPECTED_TRAINABLE_ROOTS = {
    "action_backbone",
    "proprio_action_embed",
    "proprio_encoder",
    "proprio_video_embed",
}
EXPECTED_TRAINABLE_PARAMETER_COUNT = 639_653_063
EXPECTED_TRAINABLE_TENSOR_COUNT = 560

T16_ROOT = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t16/93e5d1cb3121/"
    "libero-t16-paired-one-task-chunk-closed-loop-fixed32-"
    "cb1ec6d46d85612dfa8523ecabe93e77"
)
T16_RESULT_SHA256 = "5e2f5d61f7715f952f1762c04f5ad99891f9eda1ae5256310498c23146bcc471"
T16_REPO_COMMIT = "93e5d1cb3121e42fc021b5fb616fbe0dd94090c1"
T16_RUNNER_SHA256 = "b53213477f04cff37798eb7cdff9e1fbed621d57e03df30e566404a66857a178"
T16_RUNNER_RELATIVE_PATH = (
    "scripts/smoke_libero_ar_t16_paired_one_task_closed_loop_gpu.py"
)
T16_EXECUTION_VERDICT = "T16_PAIRED_ONE_TASK_CHUNK_CLOSED_LOOP_INTERFACE_VALID"
T16_SCIENTIFIC_VERDICT = "T16_PAIRED_ONE_TASK_UPDATE_EFFECT_OBSERVED"
FAILED_R0_ROOT = Path(
    "/DATA/share/sana_wam_libero_training/formal2000/"
    "libero-ar-formal2000-single-gpu-2000-dba67fee2b53ecce092898c0a70144c4"
)
FAILED_R0_SHA256 = "0a62b6a66ba152285d751795330d90a93558c2ca851706be0549dbd6966831c2"
FAILED_R1_ROOT = Path(
    "/DATA/share/sana_wam_libero_training/formal2000/"
    "libero-ar-formal2000-r1-single-gpu-2000-ced0769ce3a32b441cda12a04929d6f9"
)
FAILED_R1_SHA256 = "e778230115fa595895ba16e5d8bc5aac545b426277e69b02f4681c34cfe6603a"
FAILED_R2_ROOT = Path(
    "/DATA/share/sana_wam_libero_training/formal2000/"
    "libero-ar-formal2000-r2-single-gpu-2000-41151d523f774bf3f0dc281df56ca8b7"
)
FAILED_R2_SHA256 = "1f7ea352023a7b4af1d6aa52feb280c059bb2731b441c1c5a506b3d21038c523"
STATS_SHA256 = "e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7"
STATS_POPULATION_SHA256 = (
    "7ed9772facf261299022e55169bcdaaa49fe3a7a20e0057e08cf419b5e584146"
)

ASSET_SHA256 = {
    "/DATA/share/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth": (
        "052c4022488949153fe6400631725bcb8c739362a6d34edeab26236d8a23f1e7"
    ),
    "/DATA/share/SANA-Video_2B_480p/config.json": (
        "d34650e1e7b0ab502051e8fe2e829ed2c93bbcad0aa9fa3ab46574491b3e8288"
    ),
    "/DATA/share/SANA-Video_2B_480p/vae/Wan2.1_VAE.pth": (
        "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
    ),
    "/DATA/share/gemma-2-2b-it/config.json": (
        "eacec6c5ca317a87ed2c46789d9705b9274db5027e7ba59da739bfae23addb55"
    ),
    "/DATA/share/gemma-2-2b-it/model-00001-of-00002.safetensors": (
        "532d792c9178805064170a3ec485b7dedbfccc6fd297b92c31a6091b6c7e41bf"
    ),
    "/DATA/share/gemma-2-2b-it/model-00002-of-00002.safetensors": (
        "6d6d9ce84db398fb6e0191f91542e5da0a73da2cb695e172a24edc2146dc8d20"
    ),
    "/DATA/share/gemma-2-2b-it/model.safetensors.index.json": (
        "ada0043f3e3b2e5ab2f445cad9c0fbbf9d91ad444675e6a82b822591c63abf5a"
    ),
    "/DATA/share/gemma-2-2b-it/special_tokens_map.json": (
        "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351"
    ),
    "/DATA/share/gemma-2-2b-it/tokenizer.json": (
        "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922"
    ),
    "/DATA/share/gemma-2-2b-it/tokenizer.model": (
        "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"
    ),
    "/DATA/share/gemma-2-2b-it/tokenizer_config.json": (
        "cb32b7929c62608d46572e813112b3ad8a841fb98fdd6a4da8559e368a951c89"
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_write_json(path: Path, value: Any, mode: int = 0o400) -> str:
    payload = _canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return hashlib.sha256(payload).hexdigest()


def _sha256_regular_file(path: Path) -> tuple[str, int]:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"expected a regular non-symlink file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if not stable or size != before.st_size:
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest(), size


def _verify_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    if os.path.realpath(path) != str(path):
        raise RuntimeError(f"file path or ancestry contains a symlink: {path}")
    digest, size = _sha256_regular_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch for {path}: {digest}")
    return {"path": str(path), "sha256": digest, "size_bytes": size}


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _verify_source(
    expected_repo_commit: str,
    expected_runner_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    if _git("rev-parse", "HEAD") != expected_repo_commit:
        raise RuntimeError("repository HEAD differs from the execution pin")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked repository state is not clean")
    if _git("rev-parse", "HEAD:third_party/Sana") != EXPECTED_SANA_COMMIT:
        raise RuntimeError("Sana gitlink differs from the execution pin")
    sana_root = ROOT / "third_party" / "Sana"
    if _git("rev-parse", "HEAD", cwd=sana_root) != EXPECTED_SANA_COMMIT:
        raise RuntimeError("Sana checkout differs from its gitlink")
    if _git("status", "--porcelain", "--untracked-files=no", cwd=sana_root):
        raise RuntimeError("tracked Sana checkout is not clean")

    runner = Path(__file__).absolute()
    config = ROOT / CONFIG_RELATIVE_PATH
    runner_manifest = _verify_file(runner, expected_runner_sha256)
    config_manifest = _verify_file(config, expected_config_sha256)
    t16_blob = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{T16_REPO_COMMIT}:{T16_RUNNER_RELATIVE_PATH}",
        ]
    )
    if hashlib.sha256(t16_blob).hexdigest() != T16_RUNNER_SHA256:
        raise RuntimeError("T16 runner blob no longer matches frozen evidence")
    return {
        "repo_commit": expected_repo_commit,
        "runner": runner_manifest,
        "config": config_manifest,
        "sana_commit": EXPECTED_SANA_COMMIT,
    }


def _verify_t16() -> dict[str, Any]:
    if os.path.realpath(T16_ROOT) != str(T16_ROOT):
        raise RuntimeError("T16 root ancestry contains a symlink")
    root_metadata = os.lstat(T16_ROOT)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o500
    ):
        raise RuntimeError("T16 root is not frozen mode 0500")
    if set(os.listdir(T16_ROOT)) != {"RESULT.json"}:
        raise RuntimeError("T16 frozen root contents changed")
    result_path = T16_ROOT / "RESULT.json"
    result_metadata = os.lstat(result_path)
    if stat.S_IMODE(result_metadata.st_mode) != 0o400:
        raise RuntimeError("T16 RESULT.json is not frozen mode 0400")
    result_manifest = _verify_file(result_path, T16_RESULT_SHA256)
    payload = json.loads(result_path.read_bytes())
    exact = {
        "valid_run": True,
        "execution_result": "PASS",
        "harness_result": "PASS",
        "verdict": T16_EXECUTION_VERDICT,
        "execution_verdict": T16_EXECUTION_VERDICT,
        "scientific_verdict": T16_SCIENTIFIC_VERDICT,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"T16 predecessor field {key} differs")
    identity = payload.get("identity", {})
    identity_exact = {
        "repo_commit": T16_REPO_COMMIT,
        "runner_sha256": T16_RUNNER_SHA256,
        "sana_commit": EXPECTED_SANA_COMMIT,
        "stats_sha256": STATS_SHA256,
        "stats_population_sha256": STATS_POPULATION_SHA256,
    }
    for key, expected in identity_exact.items():
        if identity.get(key) != expected:
            raise RuntimeError(f"T16 identity field {key} differs")
    scope = payload.get("scope", {})
    if (
        scope.get("formal_training_executed") is not False
        or scope.get("sana_wam_training_checkpoint_loaded") is not False
        or scope.get("sana_wam_training_checkpoint_saved") is not False
        or scope.get("optimizer_steps") != 20
    ):
        raise RuntimeError("T16 scope differs from the admitted predecessor")
    return result_manifest


def _verify_failed_r0() -> dict[str, Any]:
    if os.path.realpath(FAILED_R0_ROOT) != str(FAILED_R0_ROOT):
        raise RuntimeError("failed formal R0 ancestry contains a symlink")
    metadata = os.lstat(FAILED_R0_ROOT)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o500:
        raise RuntimeError("failed formal R0 root is not frozen mode 0500")
    expected_files = {"ATTEMPT.json", "FAILED.json", "TRAIN.log"}
    if set(os.listdir(FAILED_R0_ROOT)) != expected_files:
        raise RuntimeError("failed formal R0 root contents changed")
    for name in expected_files:
        child = os.lstat(FAILED_R0_ROOT / name)
        if not stat.S_ISREG(child.st_mode) or stat.S_IMODE(child.st_mode) != 0o400:
            raise RuntimeError(f"failed formal R0 file is not frozen: {name}")
    failed_path = FAILED_R0_ROOT / "FAILED.json"
    manifest = _verify_file(failed_path, FAILED_R0_SHA256)
    payload = json.loads(failed_path.read_bytes())
    if (
        payload.get("valid_run") is not False
        or payload.get("verdict") != "FAILED_CLOSED_NON_RESUMABLE"
        or payload.get("error_type") != "ValueError"
        or payload.get("error")
        != "every sample must contain at least one non-bootstrap, non-padded video target"
        or payload.get("training_started") is not True
        or payload.get("automatic_rerun_permitted") is not False
    ):
        raise RuntimeError("failed formal R0 receipt semantics changed")
    return {**manifest, "root": str(FAILED_R0_ROOT)}


def _verify_failed_r1() -> dict[str, Any]:
    if os.path.realpath(FAILED_R1_ROOT) != str(FAILED_R1_ROOT):
        raise RuntimeError("failed formal R1 ancestry contains a symlink")
    metadata = os.lstat(FAILED_R1_ROOT)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o500:
        raise RuntimeError("failed formal R1 root is not frozen mode 0500")
    expected_files = {"ATTEMPT.json", "FAILED.json", "TRAIN.log"}
    if set(os.listdir(FAILED_R1_ROOT)) != expected_files:
        raise RuntimeError("failed formal R1 root contents changed")
    for name in expected_files:
        child = os.lstat(FAILED_R1_ROOT / name)
        if not stat.S_ISREG(child.st_mode) or stat.S_IMODE(child.st_mode) != 0o400:
            raise RuntimeError(f"failed formal R1 file is not frozen: {name}")
    failed_path = FAILED_R1_ROOT / "FAILED.json"
    manifest = _verify_file(failed_path, FAILED_R1_SHA256)
    payload = json.loads(failed_path.read_bytes())
    traceback_text = payload.get("traceback", "")
    if (
        payload.get("valid_run") is not False
        or payload.get("verdict") != "FAILED_CLOSED_NON_RESUMABLE"
        or payload.get("error_type") != "OutOfMemoryError"
        or "CUDA out of memory" not in payload.get("error", "")
        or "_multi_tensor_adam" not in traceback_text
        or "_foreach_sqrt" not in traceback_text
        or payload.get("training_started") is not True
        or payload.get("automatic_rerun_permitted") is not False
    ):
        raise RuntimeError("failed formal R1 receipt semantics changed")
    return {**manifest, "root": str(FAILED_R1_ROOT)}


def _verify_failed_r2() -> dict[str, Any]:
    if os.path.realpath(FAILED_R2_ROOT) != str(FAILED_R2_ROOT):
        raise RuntimeError("failed formal R2 ancestry contains a symlink")
    metadata = os.lstat(FAILED_R2_ROOT)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o500:
        raise RuntimeError("failed formal R2 root is not frozen mode 0500")
    expected_files = {"ATTEMPT.json", "FAILED.json", "TRAIN.log"}
    if set(os.listdir(FAILED_R2_ROOT)) != expected_files:
        raise RuntimeError("failed formal R2 root contents changed")
    for name in expected_files:
        child = os.lstat(FAILED_R2_ROOT / name)
        if not stat.S_ISREG(child.st_mode) or stat.S_IMODE(child.st_mode) != 0o400:
            raise RuntimeError(f"failed formal R2 file is not frozen: {name}")
    failed_path = FAILED_R2_ROOT / "FAILED.json"
    manifest = _verify_file(failed_path, FAILED_R2_SHA256)
    payload = json.loads(failed_path.read_bytes())
    traceback_text = payload.get("traceback", "")
    if (
        payload.get("valid_run") is not False
        or payload.get("verdict") != "FAILED_CLOSED_NON_RESUMABLE"
        or payload.get("error_type") != "OutOfMemoryError"
        or "CUDA out of memory" not in payload.get("error", "")
        or "torch/utils/checkpoint.py" not in traceback_text
        or "basic_modules.py" not in traceback_text
        or "_foreach_sqrt" in traceback_text
        or payload.get("training_started") is not True
        or payload.get("automatic_rerun_permitted") is not False
    ):
        raise RuntimeError("failed formal R2 receipt semantics changed")
    return {**manifest, "root": str(FAILED_R2_ROOT)}


def _config_value(cfg, path: str):
    return OmegaConf.select(cfg, path, default=None)


def _verify_config(expected_config_sha256: str):
    config_path = ROOT / CONFIG_RELATIVE_PATH
    digest, _size = _sha256_regular_file(config_path)
    if digest != expected_config_sha256:
        raise RuntimeError("formal config SHA changed during preflight")
    cfg = OmegaConf.load(config_path)
    exact = {
        "model.architecture.variant": "autoregressive",
        "training.output_dir": str(RUN_ROOT),
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
        "training.formal_predecessor_t16_result_sha256": T16_RESULT_SHA256,
        "training.formal_failed_predecessor_sha256": FAILED_R2_SHA256,
        "training.formal_failed_predecessor_r1_sha256": FAILED_R1_SHA256,
        "training.formal_failed_predecessor_r0_sha256": FAILED_R0_SHA256,
        "training.formal_revision_reason": (
            "break_nonreentrant_checkpoint_output_state_cycle_after_r2_backward_oom"
        ),
        "model.architecture.video_on_path_loss_weight": 0.0,
        "model.architecture.video_trajectory_endpoint_weight": 0.0,
        "model.architecture.video_trajectory_velocity_weight": 0.0,
        "model.architecture.video_trajectory_consistency_weight": 0.0,
        "model.architecture.video_local_expansion_weight": 0.0,
        "training.optimizer_master_weights": True,
        "training.optimizer_foreach": False,
        "training.seed": 20260807,
        "dataloader.seed": 20260806,
        "training.debug": False,
    }
    for path, expected in exact.items():
        observed = _config_value(cfg, path)
        if type(observed) is not type(expected) or observed != expected:
            raise RuntimeError(
                f"formal config requires {path}={expected!r}, got {observed!r}"
            )
    if list(_config_value(cfg, "training.save_at_steps")) != []:
        raise RuntimeError("formal config forbids intermediate checkpoints")
    for key in (
        "init_checkpoint",
        "init_checkpoint_sha256",
        "resume_checkpoint",
        "resume_from_checkpoint",
    ):
        if _config_value(cfg, f"training.{key}") not in (None, ""):
            raise RuntimeError(f"formal config forbids training.{key}")
    for key in ("SANA_WAM_CHECKPOINT", "SANA_WAM_CHECKPOINT_SHA256"):
        if os.environ.get(key):
            raise RuntimeError(f"formal fresh initialization forbids environment {key}")

    from sana_wam.train.libero_contract import validate_libero_training_config

    validate_libero_training_config(cfg, require_materialized_stats=True)
    digest_after, _ = _sha256_regular_file(config_path)
    if digest_after != digest:
        raise RuntimeError("formal config changed during semantic validation")
    return cfg


def _verify_assets() -> dict[str, dict[str, Any]]:
    checkpoints = Path("/DATA/share/SANA-Video_2B_480p/checkpoints")
    if sorted(path.name for path in checkpoints.glob("*.pth")) != [
        "SANA_Video_2B_480p.pth"
    ]:
        raise RuntimeError("SANA checkpoint directory is not the pinned singleton")
    return {
        path: _verify_file(Path(path), digest) for path, digest in ASSET_SHA256.items()
    }


def _query_gpu(*, require_idle: bool) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,"
            "pci.bus_id,driver_version",
            "--format=csv,noheader,nounits",
            "-i",
            str(EXPECTED_GPU_INDEX),
        ],
        text=True,
    ).strip()
    fields = [field.strip() for field in output.split(",")]
    if len(fields) != 8:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {output!r}")
    gpu = {
        "physical_index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_used_mib": int(fields[3]),
        "memory_total_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
        "pci_bus_id": fields[6],
        "driver_version": fields[7],
    }
    if (
        gpu["physical_index"] != EXPECTED_GPU_INDEX
        or gpu["uuid"] != EXPECTED_GPU_UUID
        or "H200" not in gpu["name"]
    ):
        raise RuntimeError("GPU physical identity differs from the formal pin")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    matching = [
        line for line in processes if line.strip().startswith(EXPECTED_GPU_UUID)
    ]
    if require_idle and matching:
        raise RuntimeError(
            f"GPU {EXPECTED_GPU_INDEX} has compute processes: {matching}"
        )
    if require_idle and (
        gpu["memory_used_mib"] > 256 or gpu["utilization_percent"] > 1
    ):
        raise RuntimeError(f"GPU {EXPECTED_GPU_INDEX} is not idle: {gpu}")
    return gpu


@contextlib.contextmanager
def _gpu_lock():
    lock_path = Path(f"/tmp/sana-wam-{EXPECTED_GPU_UUID}.lock")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"GPU lock is already held: {lock_path}") from error
        yield str(lock_path)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _verify_environment_arguments(args: argparse.Namespace) -> None:
    for name in (
        "expected_repo_commit",
        "expected_runner_sha256",
        "expected_config_sha256",
    ):
        value = getattr(args, name)
        if not re.fullmatch(r"[0-9a-f]{64}|[0-9a-f]{40}", value):
            raise ValueError(f"--{name.replace('_', '-')} has an invalid digest")
    if args.config != CONFIG_RELATIVE_PATH:
        raise RuntimeError("only the frozen formal2000 config is accepted")
    if args.run_root != str(RUN_ROOT) or args.nonce != RUN_NONCE:
        raise RuntimeError("run root or nonce differs from the one-shot contract")
    if (
        args.physical_gpu != EXPECTED_GPU_INDEX
        or args.expected_gpu_uuid != EXPECTED_GPU_UUID
    ):
        raise RuntimeError("GPU argument differs from the one-shot contract")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(EXPECTED_GPU_INDEX):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose only physical GPU 7")
    expected_distributed = {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
    for key, expected in expected_distributed.items():
        observed = os.environ.get(key)
        if observed not in (None, expected):
            raise RuntimeError(f"{key} differs from the single-process contract")


def _create_run_root() -> None:
    parent = RUN_ROOT.parent
    if os.path.realpath(parent) != str(parent):
        raise RuntimeError("formal run parent ancestry contains a symlink")
    metadata = os.lstat(parent)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("formal run parent is not a directory")
    if os.path.lexists(RUN_ROOT):
        raise FileExistsError(f"formal run root already exists: {RUN_ROOT}")
    available = os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize
    if available < 50 * 1024**3:
        raise RuntimeError("formal run parent has less than 50 GiB free")
    os.mkdir(RUN_ROOT, 0o700)
    descriptor = os.open(RUN_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fchmod(descriptor, 0o700)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(parent)


def _open_training_log() -> tuple[Any, logging.Handler]:
    path = RUN_ROOT / "TRAIN.log"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    stream = os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)
    file_handler = logging.StreamHandler(stream)
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[stdout_handler, file_handler],
        force=True,
    )
    return stream, file_handler


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _verify_base_load(messages: list[str]) -> dict[str, Any]:
    matches = [message for message in messages if "SANA ckpt load partial:" in message]
    if len(matches) != 1:
        raise RuntimeError(f"expected one SANA base-load report, got {matches}")
    message = matches[0]
    expected_prefix = "SANA ckpt load partial: 280 missing, 0 unexpected keys."
    samples = (
        "blocks.0.learnable_fa_scale",
        "blocks.0.attn.kernel_func.alpha",
        "blocks.0.attn.kernel_func.w1.weight",
    )
    if not message.startswith(expected_prefix) or any(
        sample not in message for sample in samples
    ):
        raise RuntimeError(f"SANA base-load signature differs: {message}")
    return {"missing_keys": 280, "unexpected_keys": 0, "sample_missing": list(samples)}


def _verify_trainable_partition(trainer) -> dict[str, Any]:
    trainables = [
        (name, parameter)
        for name, parameter in trainer.architecture.named_parameters()
        if parameter.requires_grad
    ]
    roots = {name.split(".", 1)[0] for name, _parameter in trainables}
    parameter_count = sum(parameter.numel() for _name, parameter in trainables)
    if (
        roots != EXPECTED_TRAINABLE_ROOTS
        or len(trainables) != EXPECTED_TRAINABLE_TENSOR_COUNT
        or parameter_count != EXPECTED_TRAINABLE_PARAMETER_COUNT
    ):
        raise RuntimeError("formal trainable parameter partition differs from T16")
    return {
        "roots": sorted(roots),
        "parameter_count": parameter_count,
        "parameter_tensor_count": len(trainables),
    }


def _sync_close_log(stream, file_handler: logging.Handler | None) -> None:
    if file_handler is not None:
        logging.getLogger().removeHandler(file_handler)
        file_handler.flush()
        file_handler.close()
    if stream is not None and not stream.closed:
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()


def _freeze_run_contents() -> None:
    if not RUN_ROOT.is_dir() or RUN_ROOT.is_symlink():
        raise RuntimeError("cannot freeze a missing or symlink run root")
    for directory, subdirectories, files in os.walk(RUN_ROOT, topdown=False):
        base = Path(directory)
        for name in files:
            path = base / name
            metadata = os.lstat(path)
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"run root contains a non-regular file: {path}")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in subdirectories:
            path = base / name
            metadata = os.lstat(path)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"run root contains a non-directory child: {path}")
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o500)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _seal_run_root() -> None:
    descriptor = os.open(RUN_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fchmod(descriptor, 0o500)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(RUN_ROOT.parent)


def _freeze_run_root() -> None:
    _freeze_run_contents()
    _seal_run_root()


def _result_files() -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for path in sorted(RUN_ROOT.iterdir()):
        if path.name in {"RESULT.json", "FAILED.json"}:
            continue
        digest, size = _sha256_regular_file(path)
        manifests[path.name] = {"sha256": digest, "size_bytes": size}
    return manifests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(f"formal launcher forbids unpinned arguments: {unknown}")
    return args


def main() -> None:
    args = _parse_args()
    _verify_environment_arguments(args)
    source = _verify_source(
        args.expected_repo_commit,
        args.expected_runner_sha256,
        args.expected_config_sha256,
    )
    t16 = _verify_t16()
    failed_r0 = _verify_failed_r0()
    failed_r1 = _verify_failed_r1()
    failed_r2 = _verify_failed_r2()
    cfg = _verify_config(args.expected_config_sha256)
    assets = _verify_assets()

    root_created = False
    log_stream = None
    log_handler = None
    attempt_sha256 = None
    started_at = _utc_now()
    training_started = None
    try:
        with _gpu_lock() as lock_path:
            gpu_before = _query_gpu(require_idle=True)
            _create_run_root()
            root_created = True
            attempt = {
                "schema_version": "sana-wam-libero-formal2000-attempt-v1",
                "state": "RUNNING_NON_RESUMABLE",
                "created_at_utc": started_at,
                "run": {"nonce": RUN_NONCE, "root": str(RUN_ROOT)},
                "source": source,
                "predecessor_t16": t16,
                "failed_predecessor_r0": failed_r0,
                "failed_predecessor_r1": failed_r1,
                "failed_predecessor_r2": failed_r2,
                "assets": assets,
                "gpu": {**gpu_before, "lock_path": lock_path},
                "training": {
                    "fresh_initialization": True,
                    "optimizer": "AdamW_FP32_master_foreach_false",
                    "optimizer_steps": 2000,
                    "gradient_accumulation_steps": 8,
                    "intermediate_checkpoints": 0,
                    "final_checkpoint": "checkpoint_step_2000.safetensors",
                    "resumable": False,
                },
            }
            attempt_sha256 = _exclusive_write_json(RUN_ROOT / "ATTEMPT.json", attempt)
            log_stream, log_handler = _open_training_log()
            logger = logging.getLogger(__name__)
            logger.info("Formal2000 preflight complete; constructing Trainer")

            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError(
                    "formal launcher requires exactly one visible CUDA GPU"
                )
            if "H200" not in torch.cuda.get_device_name(0):
                raise RuntimeError("visible CUDA device is not an H200")

            from sana_wam.train.libero_contract import validate_libero_training_config
            from sana_wam.train.trainer import Trainer

            capture = _LogCapture()
            logging.getLogger().addHandler(capture)
            try:
                trainer = Trainer(cfg, exact_output_dir=str(RUN_ROOT))
            finally:
                logging.getLogger().removeHandler(capture)
            base_load = _verify_base_load(capture.messages)
            partition = _verify_trainable_partition(trainer)
            validate_libero_training_config(cfg, dataset=trainer.dataset)
            logger.info("Trainer admitted; starting 2000 optimizer steps")
            training_started = time.monotonic()
            output = trainer.train()
            training_seconds = time.monotonic() - training_started
            if output != str(RUN_ROOT):
                raise RuntimeError("Trainer returned a different output root")
            summary = getattr(trainer, "formal_libero_run_summary", None)
            if not isinstance(summary, dict) or summary.get("optimizer_steps") != 2000:
                raise RuntimeError("Trainer did not publish the formal2000 run summary")
            required_finite = (
                "final_loss_finite",
                "model_parameters_finite",
                "optimizer_master_parameters_finite",
                "optimizer_state_finite",
            )
            if any(summary.get(key) is not True for key in required_finite):
                raise RuntimeError("formal2000 final numerical state is not finite")
            if (
                summary.get("optimizer_foreach") is not False
                or summary.get("autograd_graph_released_before_optimizer") is not True
            ):
                raise RuntimeError("formal2000 memory-peak revision was not active")

            checkpoints = sorted(RUN_ROOT.glob("checkpoint_step_*.safetensors"))
            expected_checkpoint = RUN_ROOT / "checkpoint_step_2000.safetensors"
            if checkpoints != [expected_checkpoint]:
                raise RuntimeError(f"unexpected checkpoint set: {checkpoints}")
            checkpoint_sha256, checkpoint_size = _sha256_regular_file(
                expected_checkpoint
            )
            if checkpoint_size < 1_000_000_000:
                raise RuntimeError("formal checkpoint is unexpectedly small")
            from safetensors import safe_open

            with safe_open(
                expected_checkpoint, framework="pt", device="cpu"
            ) as checkpoint:
                checkpoint_tensor_count = len(checkpoint.keys())
            if checkpoint_tensor_count <= 0:
                raise RuntimeError("formal checkpoint contains no tensors")
            action_stats = RUN_ROOT / "action_stats.npy"
            action_stats_sha256, _ = _sha256_regular_file(action_stats)
            if action_stats_sha256 != STATS_SHA256:
                raise RuntimeError(
                    "saved action stats differ from the training population"
                )
            saved_config = RUN_ROOT / "config.yaml"
            saved_config_sha256, _ = _sha256_regular_file(saved_config)
            gpu_after = _query_gpu(require_idle=False)
            logger.info(
                "Formal2000 training and checkpoint verification complete; "
                "sealing the training log before the terminal manifest"
            )
            _sync_close_log(log_stream, log_handler)
            log_stream = None
            log_handler = None
            files = _result_files()
            result = {
                "schema_version": "sana-wam-libero-formal2000-result-v1",
                "valid_run": True,
                "verdict": "LIBERO_AR_FORMAL2000_TRAINING_COMPLETE",
                "execution_result": "PASS",
                "created_at_utc": _utc_now(),
                "run": {"nonce": RUN_NONCE, "root": str(RUN_ROOT)},
                "attempt_sha256": attempt_sha256,
                "source": source,
                "predecessor_t16": t16,
                "failed_predecessor_r0": failed_r0,
                "failed_predecessor_r1": failed_r1,
                "failed_predecessor_r2": failed_r2,
                "base_load": base_load,
                "trainable_partition": partition,
                "training": {
                    **summary,
                    "fresh_initialization": True,
                    "gradient_accumulation_steps": 8,
                    "training_seconds": training_seconds,
                    "resumable": False,
                },
                "checkpoint": {
                    "path": str(expected_checkpoint),
                    "sha256": checkpoint_sha256,
                    "size_bytes": checkpoint_size,
                    "tensor_count": checkpoint_tensor_count,
                    "saved_config_sha256": saved_config_sha256,
                    "action_stats_sha256": action_stats_sha256,
                },
                "gpu": {"before": gpu_before, "after": gpu_after},
                "files_before_result": files,
                "scope": {
                    "formal_training_executed": True,
                    "formal_evaluation_executed": False,
                    "real_libero_training_data_used": True,
                    "sana_base_pretrained_checkpoint_loaded": True,
                    "sana_wam_training_checkpoint_loaded": False,
                    "sana_wam_training_checkpoint_saved": True,
                    "optimizer_steps": 2000,
                },
            }
            _freeze_run_contents()
            _exclusive_write_json(RUN_ROOT / "RESULT.json", result)
            _seal_run_root()
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
        if root_created:
            with contextlib.suppress(Exception):
                logging.getLogger(__name__).exception("Formal2000 failed")
            with contextlib.suppress(Exception):
                _sync_close_log(log_stream, log_handler)
                log_stream = None
                log_handler = None
            if (
                not (RUN_ROOT / "RESULT.json").exists()
                and not (RUN_ROOT / "FAILED.json").exists()
            ):
                failure = {
                    "schema_version": "sana-wam-libero-formal2000-failure-v1",
                    "valid_run": False,
                    "verdict": "FAILED_CLOSED_NON_RESUMABLE",
                    "created_at_utc": _utc_now(),
                    "run": {"nonce": RUN_NONCE, "root": str(RUN_ROOT)},
                    "attempt_sha256": attempt_sha256,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc()[-16_000:],
                    "training_started": training_started is not None,
                    "automatic_rerun_permitted": False,
                }
                with contextlib.suppress(Exception):
                    _exclusive_write_json(RUN_ROOT / "FAILED.json", failure)
            with contextlib.suppress(Exception):
                _freeze_run_root()
        raise


if __name__ == "__main__":
    main()
