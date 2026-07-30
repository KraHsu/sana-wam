#!/usr/bin/env python3
"""Authority-bound launcher for the single post-primary AFCC treatment."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    info = os.lstat(resolved)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink")
    if _sha256_file(resolved) != _sha(expected, f"{name} SHA256"):
        raise ValueError(f"{name} SHA256 differs")
    return resolved


def _canonical(value) -> bytes:
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


def _strict_json_lines(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if not data or not data.endswith(b"\n"):
        raise ValueError("AFCC metrics JSONL must end with one newline per row")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate AFCC metrics key: {key!r}")
            result[key] = value
        return result

    rows = []
    for index, raw in enumerate(data.splitlines(keepends=True), start=1):
        try:
            value = json.loads(
                raw,
                object_pairs_hook=pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite AFCC metrics value: {token}")
                ),
            )
        except (json.JSONDecodeError, TypeError, UnicodeError, ValueError) as exc:
            raise ValueError(f"invalid AFCC metrics row {index}: {exc}") from exc
        if type(value) is not dict or raw != _canonical(value):
            raise ValueError(f"AFCC metrics row {index} is not canonical JSON")
        rows.append(value)
    return rows


def _validate_metric_rows(
    metric_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> None:
    expected_keys = {
        "action_sigma",
        "afcc_loss",
        "common_input_trace_sha256",
        "global_step",
        "gradient_global_norm",
        "learning_rate",
        "peak_reserved_bytes",
        "plan_row_sha256",
        "schema_version",
        "task_name",
        "total_loss",
        "video_local_expansion_loss",
        "video_on_path_loss",
    }
    if len(metric_rows) != 504 or len(reference_rows) != 504:
        raise RuntimeError("AFCC metrics/reference do not close 504 rows")
    for step, (row, reference) in enumerate(
        zip(metric_rows, reference_rows, strict=True), start=1
    ):
        if set(row) != expected_keys:
            raise RuntimeError(f"AFCC metrics row {step} keys differ")
        if (
            row["schema_version"] != "sana-phase6-afcc-step-metrics-v1"
            or row["global_step"] != step
            or row["action_sigma"] != reference["action_sigma"]
            or row["common_input_trace_sha256"]
            != reference["common_input_trace_sha256"]
            or row["plan_row_sha256"] != reference["plan_row_sha256"]
            or row["task_name"] != reference["task_name"]
        ):
            raise RuntimeError(f"AFCC metrics row {step} reference binding differs")
        for key in (
            "action_sigma",
            "afcc_loss",
            "gradient_global_norm",
            "learning_rate",
            "total_loss",
            "video_local_expansion_loss",
            "video_on_path_loss",
        ):
            value = row[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise RuntimeError(f"AFCC metrics row {step}.{key} is invalid")
        if (
            type(row["peak_reserved_bytes"]) is not int
            or row["peak_reserved_bytes"] < 1
        ):
            raise RuntimeError(f"AFCC metrics row {step} peak memory is invalid")


def _exclusive_json(path: Path, value) -> str:
    payload = _canonical(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        os.fchmod(descriptor, 0o400)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _validate_live_source_manifest(path: Path) -> None:
    data = path.read_bytes()
    manifest = json.loads(data)
    if (
        data != _canonical(manifest)
        or set(manifest)
        != {
            "base_runtime_source_manifest",
            "design_authority",
            "files",
            "formal_reference_gpu_started",
            "formal_training_started",
            "schema_version",
            "status",
            "test_receipt",
        }
        or manifest["schema_version"] != "sana-phase6-afcc-source-manifest-v1"
        or manifest["status"] != "pass"
        or manifest["formal_reference_gpu_started"] is not False
        or manifest["formal_training_started"] is not False
        or type(manifest["files"]) is not dict
    ):
        raise ValueError("AFCC source manifest contract differs")
    for role in (
        "base_runtime_source_manifest",
        "design_authority",
        "test_receipt",
    ):
        pin = manifest[role]
        if set(pin) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"AFCC source manifest {role} pin differs")
        artifact = _verify(Path(pin["path"]), pin["sha256"], role)
        if artifact.stat().st_size != pin["size_bytes"]:
            raise ValueError(f"AFCC source manifest {role} size differs")
    overridden_realpaths = set()
    for relative, pin in manifest["files"].items():
        if set(pin) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"AFCC source pin differs: {relative}")
        live_source = _verify(Path(pin["path"]), pin["sha256"], relative)
        if live_source.stat().st_size != pin["size_bytes"]:
            raise ValueError(f"AFCC source size differs: {relative}")
        overridden_realpaths.add(str(live_source))
    base = json.loads(
        Path(manifest["base_runtime_source_manifest"]["path"]).read_text()
    )
    if (
        base.get("schema_version") != "sana-phase6-source-runtime-manifest-v3"
        or type(base.get("files")) is not list
    ):
        raise ValueError("base runtime source manifest contract differs")
    for pin in base["files"]:
        if set(pin) != {"path", "realpath", "sha256", "size_bytes"}:
            raise ValueError("base runtime source pin keys differ")
        realpath = str(Path(pin["realpath"]).resolve())
        if realpath in overridden_realpaths:
            continue
        live_source = _verify(Path(realpath), pin["sha256"], pin["path"])
        if live_source.stat().st_size != pin["size_bytes"]:
            raise ValueError(f"base runtime source size differs: {pin['path']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--execution-authority", type=Path, required=True)
    parser.add_argument("--execution-authority-sha256", required=True)
    return parser.parse_args()


def _authorize(args) -> tuple[dict[str, Any], Path, Path]:
    authority_path = _verify(
        args.execution_authority,
        args.execution_authority_sha256,
        "AFCC training execution authority",
    )
    if (authority_path.stat().st_mode & 0o777) != 0o400:
        raise ValueError("AFCC training execution authority must have mode 0400")
    data = authority_path.read_bytes()
    authority = json.loads(data)
    if data != _canonical(authority) or set(authority) != {
        "argv",
        "config",
        "control",
        "environment",
        "hardware",
        "lifecycle",
        "output_directory",
        "prerequisites",
        "reference_manifest",
        "schema_version",
        "script",
        "source_manifest",
        "status",
        "treatment",
    }:
        raise ValueError("AFCC training execution authority is not canonical/exact")
    if (
        authority["schema_version"]
        != "sana-phase6-afcc-training-execution-authority-v1"
        or authority["status"] != "authorized"
        or authority["treatment"] != "T1_E1A0_F1"
        or authority["control"] != "T1_E1A0"
    ):
        raise ValueError("AFCC training execution authority contract differs")
    if authority["hardware"] != {
        "compute_capability": [9, 0],
        "cuda_device_name": "NVIDIA H200",
        "cuda_device_uuid": "GPU-1ec28cfb-f501-23f3-f865-275a744ca053",
        "torch_cuda_version": "12.8",
        "torch_version": "2.7.1+cu128",
    }:
        raise ValueError("AFCC training hardware authority differs")
    config_path = _verify(args.config, args.config_sha256, "AFCC training config")
    expected_pins = {
        "config": (config_path, args.config_sha256),
        "script": (
            Path(__file__).resolve(),
            _sha256_file(Path(__file__).resolve()),
        ),
    }
    for role in ("source_manifest", "reference_manifest"):
        pin = authority[role]
        expected_pins[role] = (Path(pin["path"]).resolve(), pin["sha256"])
    resolved_pins = {}
    for role, (expected_path, expected_sha) in expected_pins.items():
        pin = authority[role]
        if set(pin) != {"path", "sha256"}:
            raise ValueError(f"AFCC authority {role} pin keys differ")
        observed_path = _verify(Path(pin["path"]), pin["sha256"], role)
        if observed_path != expected_path or pin["sha256"] != expected_sha:
            raise ValueError(f"AFCC authority {role} pin differs")
        resolved_pins[role] = observed_path
    _validate_live_source_manifest(resolved_pins["source_manifest"])
    reference_value = json.loads(resolved_pins["reference_manifest"].read_text())
    reference_runtime = reference_value.get("precompute_gpu_runtime")
    if not isinstance(reference_runtime, dict) or any(
        reference_runtime.get(key) != value
        for key, value in authority["hardware"].items()
    ):
        raise ValueError("AFCC reference runtime differs from training authority")
    for role, pin in authority["prerequisites"].items():
        if set(pin) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"AFCC prerequisite {role} pin keys differ")
        prerequisite = _verify(Path(pin["path"]), pin["sha256"], role)
        if prerequisite.stat().st_size != pin["size_bytes"]:
            raise ValueError(f"AFCC prerequisite {role} size differs")
    if authority["environment"] != {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
    }:
        raise ValueError("AFCC training execution environment differs")
    if authority["lifecycle"] != {
        "closed_loop": False,
        "resume_allowed": False,
        "retry_allowed": False,
        "selection_allowed": False,
        "training_started": False,
    }:
        raise ValueError("AFCC training lifecycle differs")
    expected_argv = [
        item.replace(
            "<AFCC_TRAINING_EXECUTION_AUTHORITY_SHA256>",
            args.execution_authority_sha256,
        )
        for item in authority["argv"]
    ]
    if [sys.executable, *sys.argv] != expected_argv:
        raise ValueError("AFCC training argv differs")
    output_directory = Path(authority["output_directory"]).absolute()
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(
            f"refusing to reuse AFCC treatment output: {output_directory}"
        )
    if not output_directory.parent.is_dir():
        raise FileNotFoundError("AFCC treatment output parent must exist")
    return authority, config_path, output_directory


def main() -> None:
    args = parse_args()
    authority, config_path, output_directory = _authorize(args)
    started = time.monotonic()

    # No torch/CUDA import occurs before the immutable authority, argv,
    # environment, prerequisites, and fresh output path have all passed.
    from omegaconf import OmegaConf
    from sana_wam.train.trainer import Trainer
    import torch
    import yaml

    _validate_live_source_manifest(
        Path(authority["source_manifest"]["path"]).resolve(strict=True)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("formal AFCC training requires CUDA")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    device_uuid = str(properties.uuid)
    if not device_uuid.startswith("GPU-"):
        device_uuid = f"GPU-{device_uuid}"
    observed_hardware = {
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_device_uuid": device_uuid,
        "torch_cuda_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
    }
    if observed_hardware != authority["hardware"]:
        raise RuntimeError("AFCC training GPU/runtime differs from authority")
    cfg = OmegaConf.load(config_path)
    if OmegaConf.to_container(cfg, resolve=True, enum_to_str=True) != yaml.safe_load(
        config_path.read_text()
    ):
        raise ValueError("loaded AFCC config differs from its immutable YAML")
    if (
        cfg.training.post_primary_afcc_condition != "T1_E1A0_F1"
        or Path(cfg.training.output_dir).absolute() != output_directory
        or int(cfg.training.max_steps) != 504
        or list(cfg.training.save_at_steps) != [504]
        or float(
            cfg.model.architecture.action_facing_cache_consistency_weight
        )
        != 1.0
    ):
        raise ValueError("loaded AFCC treatment config differs from authority")
    trainer = Trainer(
        cfg,
        launch_context=None,
        post_primary_afcc_authorized=True,
    )
    observed_output = Path(trainer.train()).resolve(strict=True)
    if observed_output != output_directory.resolve(strict=True):
        raise RuntimeError("Trainer returned a different AFCC output directory")
    checkpoint = output_directory / "checkpoint_step_504.safetensors"
    config_output = output_directory / "config.yaml"
    action_stats = output_directory / "action_stats.npy"
    metrics = output_directory / "afcc_step_metrics_v1.jsonl"
    for path in (checkpoint, config_output, action_stats, metrics):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"AFCC training output is missing: {path.name}")
    metric_rows = _strict_json_lines(metrics)
    reference_value = json.loads(
        Path(authority["reference_manifest"]["path"]).read_text()
    )
    reference_rows = reference_value.get("rows")
    if type(reference_rows) is not list:
        raise RuntimeError("AFCC reference manifest rows are invalid")
    _validate_metric_rows(metric_rows, reference_rows)
    output_pins = {
        path.name: {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (checkpoint, config_output, action_stats, metrics)
    }
    from sana_wam.train.phase6_run_integrity import (
        build_frozen_tensor_verification_manifest,
    )

    frozen_verification = build_frozen_tensor_verification_manifest(
        initial_checkpoint_path=cfg.training.init_checkpoint,
        step504_checkpoint_path=checkpoint,
        initial_checkpoint_sha256=cfg.training.init_checkpoint_sha256,
        step504_checkpoint_sha256=output_pins[checkpoint.name]["sha256"],
        arm="T1_E1A0",
    )
    from safetensors import safe_open
    from sana_wam.train.phase6_arm_config import VIDEO_PATTERNS

    with safe_open(
        cfg.training.init_checkpoint, framework="pt", device="cpu"
    ) as initial_handle, safe_open(
        checkpoint, framework="pt", device="cpu"
    ) as final_handle:
        trainable_keys = sorted(
            key
            for key in initial_handle.keys()
            if any(fnmatch.fnmatchcase(key, pattern) for pattern in VIDEO_PATTERNS)
        )
        if len(trainable_keys) != 380 or trainable_keys != sorted(
            key
            for key in final_handle.keys()
            if any(fnmatch.fnmatchcase(key, pattern) for pattern in VIDEO_PATTERNS)
        ):
            raise RuntimeError("AFCC final trainable video key set differs")
        changed_video_tensors = sum(
            not bool(
                torch.equal(
                    initial_handle.get_tensor(key), final_handle.get_tensor(key)
                )
            )
            for key in trainable_keys
        )
    # Small FP32-master updates need not cross a BF16 serialization bin.
    if changed_video_tensors == 0:
        raise RuntimeError("AFCC completed without changing a trainable video tensor")
    _validate_live_source_manifest(
        Path(authority["source_manifest"]["path"]).resolve(strict=True)
    )
    completion = {
        "closed_loop": False,
        "control": authority["control"],
        "execution_authority_sha256": args.execution_authority_sha256,
        "formal_training_completed": True,
        "frozen_action_proprio_verification": frozen_verification,
        "outputs": output_pins,
        "primary_verdict_changed": False,
        "schema_version": "sana-phase6-afcc-training-completion-v1",
        "selection_performed": False,
        "status": "pass",
        "treatment": authority["treatment"],
        "trainable_video_changed_tensor_count": changed_video_tensors,
        "wall_seconds": time.monotonic() - started,
    }
    completion_path = output_directory / "afcc_training_completion_v1.json"
    completion_sha = _exclusive_json(completion_path, completion)
    for path in output_directory.iterdir():
        if path.is_file() and not path.is_symlink():
            os.chmod(path, 0o400)
            with open(path, "rb") as stream:
                os.fsync(stream.fileno())
    os.chmod(output_directory, 0o500)
    directory = os.open(output_directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(
        json.dumps(
            {
                "completion": str(completion_path),
                "completion_sha256": completion_sha,
                "status": "pass",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
