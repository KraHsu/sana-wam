#!/usr/bin/env python3
"""Precompute immutable Phase1 action-facing video numerator shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

import precompute_phase1_action_reference as scalar_precompute  # noqa: E402


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected_sha256: str, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if _sha256_file(resolved) != _sha(expected_sha256, f"{name} SHA256"):
        raise ValueError(f"{name} SHA256 differs")
    return resolved


def _strict_json(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    resolved = _verify(path, expected_sha256, name)
    data = resolved.read_bytes()

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate key in {name}: {key!r}")
            result[key] = value
        return result

    value = json.loads(
        data,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite {name} value: {token}")
        ),
    )
    if type(value) is not dict:
        raise ValueError(f"{name} must be one JSON object")
    return value


def _validate_live_source_manifest(path: Path) -> None:
    data = path.read_bytes()
    manifest = json.loads(data)
    canonical = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    expected_keys = {
        "base_runtime_source_manifest",
        "design_authority",
        "files",
        "formal_reference_gpu_started",
        "formal_training_started",
        "schema_version",
        "status",
        "test_receipt",
    }
    if (
        data != canonical
        or set(manifest) != expected_keys
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


def _exclusive_write(path: Path, payload: bytes) -> None:
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
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _parse_steps(value: str, *, formal: bool) -> tuple[int, ...]:
    if formal:
        if value != "1-504":
            raise ValueError("formal AFCC precompute must cover steps 1-504")
        return tuple(range(1, 505))
    steps = []
    for token in value.split(","):
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            steps.extend(range(start, end + 1))
        else:
            steps.append(int(token))
    if not steps or len(set(steps)) != len(steps) or any(
        step < 1 or step > 504 for step in steps
    ):
        raise ValueError("smoke steps must be unique values in [1,504]")
    return tuple(steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--scalar-reference", type=Path, required=True)
    parser.add_argument("--scalar-reference-sha256", required=True)
    parser.add_argument("--execution-authority", type=Path)
    parser.add_argument("--execution-authority-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", default="1-504")
    return parser.parse_args()


def _load_authority(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    if args.mode == "smoke":
        forbidden = (args.execution_authority, args.execution_authority_sha256)
        if any(value is not None for value in forbidden):
            raise ValueError("smoke precompute forbids formal authority fields")
        return None, hashlib.sha256(b"smoke-authority").hexdigest()
    if args.execution_authority is None or args.execution_authority_sha256 is None:
        raise ValueError("formal precompute requires its execution authority and SHA")
    authority_path = _verify(
        args.execution_authority,
        args.execution_authority_sha256,
        "AFCC reference execution authority",
    )
    if (authority_path.stat().st_mode & 0o777) != 0o400:
        raise ValueError("AFCC reference execution authority must have mode 0400")
    data = authority_path.read_bytes()
    authority = json.loads(data)
    canonical = (
        json.dumps(
            authority,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if data != canonical or set(authority) != {
        "argv",
        "config",
        "environment",
        "hardware",
        "lifecycle",
        "output_directory",
        "prerequisites",
        "scalar_reference",
        "schema_version",
        "script",
        "source_manifest",
        "status",
        "steps",
    }:
        raise ValueError("AFCC reference execution authority is not canonical/exact")
    if (
        authority["schema_version"]
        != "sana-phase6-afcc-reference-execution-authority-v1"
        or authority["status"] != "authorized"
        or authority["steps"] != [1, 504]
        or authority["output_directory"] != str(args.output_dir.expanduser().absolute())
    ):
        raise ValueError("AFCC reference execution authority contract differs")
    if authority["hardware"] != {
        "compute_capability": [9, 0],
        "cuda_device_name": "NVIDIA H200",
        "cuda_device_uuid": "GPU-1ec28cfb-f501-23f3-f865-275a744ca053",
        "torch_cuda_version": "12.8",
        "torch_version": "2.7.1+cu128",
    }:
        raise ValueError("AFCC reference hardware authority differs")
    expected_pins = {
        "config": (args.config, args.config_sha256),
        "source_manifest": (args.source_manifest, args.source_manifest_sha256),
        "scalar_reference": (
            args.scalar_reference,
            args.scalar_reference_sha256,
        ),
        "script": (
            Path(__file__),
            _sha256_file(Path(__file__).resolve()),
        ),
    }
    resolved_pins = {}
    for role, (path, digest) in expected_pins.items():
        pin = authority[role]
        if set(pin) != {"path", "sha256"}:
            raise ValueError(f"AFCC authority {role} pin keys differ")
        resolved = _verify(Path(pin["path"]), pin["sha256"], f"authority {role}")
        if resolved != Path(path).expanduser().resolve(strict=True) or pin[
            "sha256"
        ] != digest:
            raise ValueError(f"AFCC authority {role} pin differs")
        resolved_pins[role] = resolved
    _validate_live_source_manifest(resolved_pins["source_manifest"])
    for role, pin in authority["prerequisites"].items():
        if set(pin) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"AFCC prerequisite {role} pin keys differ")
        prerequisite = _verify(
            Path(pin["path"]), pin["sha256"], f"AFCC prerequisite {role}"
        )
        if prerequisite.stat().st_size != pin["size_bytes"]:
            raise ValueError(f"AFCC prerequisite {role} size differs")
    if authority["environment"] != {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
    }:
        raise ValueError("AFCC reference execution environment differs")
    lifecycle = authority["lifecycle"]
    if lifecycle != {
        "closed_loop": False,
        "optimizer_created": False,
        "resume_allowed": False,
        "training_started": False,
    }:
        raise ValueError("AFCC reference execution lifecycle differs")
    expected_argv = [
        item.replace(
            "<AFCC_REFERENCE_EXECUTION_AUTHORITY_SHA256>",
            args.execution_authority_sha256,
        )
        for item in authority["argv"]
    ]
    if [sys.executable, *sys.argv] != expected_argv:
        raise ValueError("AFCC reference execution argv differs")
    return authority, args.execution_authority_sha256


def _forward_one(architecture, dataset, plan, scalar_row, global_step: int):
    import torch

    sample = dataset[plan.rows[global_step - 1].identity.dataset_index]
    raw_row = sample.get("phase6_plan_row")
    if raw_row is None or raw_row.get("global_step") != global_step:
        raise RuntimeError("AFCC dataset row differs from the requested plan row")
    inputs = architecture.prepare_inputs([sample])
    if inputs.get("phase6_plan_rows") != (raw_row,):
        raise RuntimeError("AFCC prepare_inputs changed its plan row")
    inputs["phase6_common_input_trace_required"] = True
    inputs["phase6_t0_reference_forward_trace_required"] = True
    inputs["phase6_action_facing_cache_capture_required"] = True
    with torch.no_grad():
        result = architecture.compute_loss(
            **inputs,
            lambda_video=1.0,
            lambda_action=0.0,
        )
    capture = result.get("phase6_action_facing_cache_capture")
    if (
        not isinstance(capture, torch.Tensor)
        or tuple(capture.shape) != (1, 20, 4, 20, 28, 112)
        or capture.dtype != torch.bfloat16
        or capture.requires_grad
        or not bool(torch.isfinite(capture).all())
    ):
        raise RuntimeError("AFCC Phase1 capture tensor contract differs")
    common_trace = result.get("phase6_common_input_trace_sha256")
    t0_trace = result.get("phase6_timestep_forward_trace_sha256")
    if (
        common_trace != scalar_row["common_input_trace_sha256"]
        or t0_trace != scalar_row["t0_reference_forward_trace_sha256"]
    ):
        raise RuntimeError("AFCC Phase1 capture trace differs from scalar reference")
    error = result.get("phase6_action_unweighted_mse")
    from sana_wam.train.action_reference_table import float32_bits

    if float32_bits(float(error.detach().cpu().item())) != scalar_row[
        "error_float32_bits"
    ]:
        raise RuntimeError("AFCC capture changed the existing Phase1 action result")
    return capture.squeeze(0).detach().cpu().contiguous(), common_trace, t0_trace


def main() -> None:
    args = parse_args()
    formal = args.mode == "formal"
    steps = _parse_steps(args.steps, formal=formal)
    config_path = _verify(args.config, args.config_sha256, "AFCC config")
    source_path = _verify(
        args.source_manifest,
        args.source_manifest_sha256,
        "AFCC source manifest",
    )
    scalar_reference = _strict_json(
        args.scalar_reference,
        args.scalar_reference_sha256,
        "Phase1 scalar reference",
    )
    if len(scalar_reference.get("rows", [])) != 504:
        raise ValueError("Phase1 scalar reference must contain 504 rows")
    authority, authority_sha = _load_authority(args)
    output_dir = args.output_dir.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse AFCC output directory: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError("AFCC output parent must already exist")
    if formal and authority["output_directory"] != str(output_dir):
        raise ValueError("AFCC output directory differs from authority")

    # Torch/CUDA is intentionally imported only after all authorization and
    # freshness checks above have passed.
    import torch
    from safetensors.torch import save

    _validate_live_source_manifest(Path(args.source_manifest).resolve(strict=True))

    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("AFCC reference precompute requires one CUDA process")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg, _artifact_config, _ = scalar_precompute.load_config(
        config_path, args.config_sha256
    )
    scalar_precompute.seed_reference_process(int(cfg.training.seed))
    dataset, plan, dataset_contract = scalar_precompute.build_bound_dataset(
        cfg,
        expected_source_manifest_sha256=args.source_manifest_sha256,
    )
    architecture = scalar_precompute.build_phase1_architecture(cfg, dataset, device)
    gpu_runtime = scalar_precompute.gpu_runtime_descriptor(torch, device)
    if formal and any(
        gpu_runtime[key] != value for key, value in authority["hardware"].items()
    ):
        raise RuntimeError("AFCC reference GPU/runtime differs from authority")
    os.mkdir(output_dir, 0o700)
    parent_descriptor = os.open(output_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)

    from sana_wam.train.action_facing_cache_reference import (
        AFCC_REFERENCE_SCHEMA_VERSION,
        AFCC_REFERENCE_TENSOR_KEY,
        canonical_json_bytes,
        tensor_sha256,
    )

    rows = []
    for global_step in steps:
        scalar_row = scalar_reference["rows"][global_step - 1]
        tensor, common_trace, t0_trace = _forward_one(
            architecture,
            dataset,
            plan,
            scalar_row,
            global_step,
        )
        shard_path = output_dir / f"row_{global_step:03d}.safetensors"
        shard_bytes = save({AFCC_REFERENCE_TENSOR_KEY: tensor})
        _exclusive_write(shard_path, shard_bytes)
        rows.append(
            {
                "action_sigma": float(scalar_row["action_sigma"]),
                "common_input_trace_sha256": common_trace,
                "dataset_contract_row_sha256": scalar_row[
                    "dataset_contract_row_sha256"
                ],
                "dataset_index": int(scalar_row["dataset_index"]),
                "file_sha256": hashlib.sha256(shard_bytes).hexdigest(),
                "global_step": global_step,
                "path": str(shard_path),
                "plan_row_sha256": scalar_row["plan_row_sha256"],
                "size_bytes": len(shard_bytes),
                "task_name": scalar_row["task_name"],
                "tensor_sha256": tensor_sha256(tensor),
                "t0_reference_forward_trace_sha256": t0_trace,
            }
        )
        print(f"AFCC reference row {global_step}/{steps[-1]} complete", flush=True)
    if formal:
        _validate_live_source_manifest(
            Path(args.source_manifest).resolve(strict=True)
        )
    manifest = {
        "action_stats_sha256": str(cfg.training.action_stats_sha256),
        "dataset_contract_artifact_sha256": (
            dataset_contract.artifact_sha256
        ),
        "identity_sha256": plan.identity_sha256,
        "plan_sha256": plan.plan_sha256,
        "precompute_config_file_sha256": args.config_sha256,
        "precompute_gpu_runtime": gpu_runtime,
        "reference_execution_authority_sha256": authority_sha,
        "reference_checkpoint_sha256": str(cfg.training.init_checkpoint_sha256),
        "row_count": len(rows),
        "rows": rows,
        "scalar_reference_artifact_sha256": args.scalar_reference_sha256,
        "schema_version": (
            AFCC_REFERENCE_SCHEMA_VERSION
            if formal
            else "sana-phase6-afcc-reference-manifest-smoke-v1"
        ),
        "source_manifest_sha256": args.source_manifest_sha256,
        "tensor_contract": {
            "dtype": "BF16",
            "key": AFCC_REFERENCE_TENSOR_KEY,
            "shape": [20, 4, 20, 28, 112],
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path = output_dir / "manifest_v1.json"
    _exclusive_write(manifest_path, manifest_bytes)
    os.chmod(output_dir, 0o500)
    directory = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "mode": args.mode,
                "row_count": len(rows),
                "source_manifest": str(source_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
