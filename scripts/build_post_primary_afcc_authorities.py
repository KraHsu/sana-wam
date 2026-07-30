#!/usr/bin/env python3
"""Publish immutable design/source/reference/training authorities for AFCC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = Path(
    "/DATA/share/sana_phase6_principled_constraints_20260724/post_primary_afcc_v3"
)
PHASE6_ROOT = ARTIFACT_ROOT.parent
REPOSITORY = Path("/home/zch/workspace/sana-wam")
PYTHON = REPOSITORY / ".venv/bin/python"
SOURCE_MANIFEST_PATH = ARTIFACT_ROOT / "source_v1/afcc_source_manifest_v1.json"
TEST_RECEIPT_PATH = ARTIFACT_ROOT / "source_v1/afcc_test_receipt_v1.json"
DESIGN_PATH = ARTIFACT_ROOT / "afcc_design_authority_v1.json"
REFERENCE_CONFIG_PATH = ARTIFACT_ROOT / "reference_precompute_v1/config.yaml"
REFERENCE_AUTHORITY_PATH = (
    ARTIFACT_ROOT
    / "reference_precompute_v1/afcc_reference_execution_authority_v1.json"
)
REFERENCE_OUTPUT = ARTIFACT_ROOT / "reference_v1"
TRAINING_CONFIG_PATH = ARTIFACT_ROOT / "training_v1/config.yaml"
TRAINING_AUTHORITY_PATH = (
    ARTIFACT_ROOT / "training_v1/afcc_training_execution_authority_v1.json"
)
TRAINING_OUTPUT = ARTIFACT_ROOT / "treatment_T1_E1A0_F1_v1"
SCALAR_REFERENCE = PHASE6_ROOT / "reference/phase1_action_reference_v9.json"
SCALAR_REFERENCE_SHA = "24eb405125052a8f4f4e0bb1869e76dbed47dc43920abd1d62cbdf26fdf71b41"
BASE_REFERENCE_CONFIG = (
    REPOSITORY
    / "configs/phase6/generated_v8/phase1_action_reference_precompute_recovery_v5.yaml"
)
BASE_REFERENCE_CONFIG_SHA = (
    "63bce4e1a1ff0922d010cafad431f966bbea0c24ddc34e76bf5e6160231321c9"
)
CONTROL_CONFIG = (
    PHASE6_ROOT
    / "arms/T1_E1A0/phase6_20260725_T1_E1A0_planrowfix_r2/config.yaml"
)
CONTROL_CONFIG_SHA = "ff66e4fe28f6756df80fce1c900e0db1f05bbbaecc8c13ced1827cad78db3bfa"
OLD_SOURCE_MANIFEST = PHASE6_ROOT / "artifacts/phase6_source_runtime_manifest_v9.json"
OLD_SOURCE_MANIFEST_SHA = (
    "23224e073a731d50b17aacbb934bda703fe9cc11ac4fde6afddc8a432d3c93b4"
)
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
HARDWARE_AUTHORITY = {
    "compute_capability": [9, 0],
    "cuda_device_name": "NVIDIA H200",
    "cuda_device_uuid": "GPU-1ec28cfb-f501-23f3-f865-275a744ca053",
    "torch_cuda_version": "12.8",
    "torch_version": "2.7.1+cu128",
}
PLAN_SHA256 = "701d1436c9804df960d190586c264b80af42e8ce107751693e4c5adbe1089411"
IDENTITY_SHA256 = "08b9fcf418b0c4aabf7ea5e494cc8f603e63b797658c2c3890c2a06599a965dd"
DATASET_CONTRACT_SHA256 = (
    "29a66c9e408ae68faa10705640ee414fbe9de96e28ab0841b1df60abbe6466ab"
)
ACTION_STATS_SHA256 = (
    "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"
)
PHASE1_CHECKPOINT_SHA256 = (
    "ba7bb59fe7e44f6efd6a4cef79f66aff7ccf5dbe82a5a166634e35055e94f089"
)


SOURCE_PATHS = (
    "src/sana_wam/model/action_facing_cache_consistency.py",
    "src/sana_wam/model/ar/sana_ar_linear_attn.py",
    "src/sana_wam/model/ar/sana_mot_driver.py",
    "src/sana_wam/model/ar/sana_ar_mot_driver.py",
    "src/sana_wam/model/architecture.py",
    "src/sana_wam/train/action_facing_cache_reference.py",
    "src/sana_wam/train/trainer.py",
    "scripts/precompute_phase1_action_facing_cache_reference.py",
    "scripts/smoke_afcc_real_2b.py",
    "scripts/train_post_primary_afcc.py",
    "scripts/build_post_primary_afcc_authorities.py",
    "tests/test_action_facing_cache_consistency.py",
    "tests/test_action_facing_cache_reference.py",
    "tests/test_afcc_trainer_contract.py",
    "tests/test_sana_ar_padding_semantics.py",
)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(path: Path, expected: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    info = os.lstat(resolved)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"pin is not a regular non-symlink: {resolved}")
    digest = _hash(resolved)
    if expected is not None and digest != _sha(expected, f"{resolved} SHA256"):
        raise ValueError(f"pin SHA256 differs: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": info.st_size,
    }


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


def _strict_json(path: Path, name: str) -> dict[str, Any]:
    data = path.read_bytes()

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate {name} key: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite {name} value: {token}")
            ),
        )
    except (json.JSONDecodeError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc
    if type(value) is not dict or data != _canonical(value):
        raise ValueError(f"{name} must be one canonical JSON object")
    return value


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} keys differ")
    return value


def _finite_positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (not allow_zero and value == 0)
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _candidate_config_bytes(
    base_path: Path, source_manifest: Path, source_sha256: str
) -> bytes:
    config = yaml.safe_load(base_path.read_text())
    config["training"]["phase6_code_source_manifest"] = str(source_manifest)
    config["training"]["phase6_code_source_manifest_sha256"] = source_sha256
    return yaml.safe_dump(
        config,
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
    ).encode("utf-8")


def _write(path: Path, payload: bytes) -> str:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"publication parent is absent: {path.parent}")
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
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value) -> str:
    return _write(path, _canonical(value))


def _mkdir(path: Path, mode: int = 0o700) -> None:
    os.mkdir(path, mode)
    os.chmod(path, mode)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _run_check(argv: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        argv,
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"source gate failed: {argv}\n{result.stdout}")
    return {
        "argv": argv,
        "exit_code": result.returncode,
        "output": result.stdout,
        "output_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
    }


def _validate_base_runtime_sources(overrides: dict[str, dict[str, Any]]) -> None:
    base = json.loads(OLD_SOURCE_MANIFEST.read_text())
    if (
        base.get("schema_version") != "sana-phase6-source-runtime-manifest-v3"
        or type(base.get("files")) is not list
    ):
        raise ValueError("base runtime source manifest contract differs")
    overridden = {pin["path"] for pin in overrides.values()}
    for pin in base["files"]:
        if set(pin) != {"path", "realpath", "sha256", "size_bytes"}:
            raise ValueError("base runtime source pin keys differ")
        if str(Path(pin["realpath"]).resolve()) in overridden:
            continue
        observed = _pin(Path(pin["realpath"]), pin["sha256"])
        if observed["size_bytes"] != pin["size_bytes"]:
            raise ValueError(f"base runtime source size differs: {pin['path']}")


def _validate_afcc_source_manifest(
    path: Path,
    expected_sha256: str,
    *,
    expected_files: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    observed = _pin(path, expected_sha256)
    value = _strict_json(Path(observed["path"]), "AFCC source manifest")
    _exact_keys(
        value,
        {
            "base_runtime_source_manifest",
            "design_authority",
            "files",
            "formal_reference_gpu_started",
            "formal_training_started",
            "schema_version",
            "status",
            "test_receipt",
        },
        "AFCC source manifest",
    )
    if (
        value["schema_version"] != "sana-phase6-afcc-source-manifest-v1"
        or value["status"] != "pass"
        or value["formal_reference_gpu_started"] is not False
        or value["formal_training_started"] is not False
        or type(value["files"]) is not dict
    ):
        raise ValueError("AFCC source manifest contract differs")
    base_pin = _pin(OLD_SOURCE_MANIFEST, OLD_SOURCE_MANIFEST_SHA)
    if value["base_runtime_source_manifest"] != base_pin:
        raise ValueError("AFCC source manifest base pin differs")
    for role in ("design_authority", "test_receipt"):
        pin = _exact_keys(
            value[role], {"path", "sha256", "size_bytes"}, f"source {role}"
        )
        artifact = _pin(Path(pin["path"]), pin["sha256"])
        if artifact != pin:
            raise ValueError(f"AFCC source manifest {role} pin differs")
    for relative, pin in value["files"].items():
        _exact_keys(pin, {"path", "sha256", "size_bytes"}, relative)
        if _pin(Path(pin["path"]), pin["sha256"]) != pin:
            raise ValueError(f"AFCC source manifest pin differs: {relative}")
    if expected_files is not None and value["files"] != expected_files:
        raise ValueError("smoke source inventory differs from candidate source")
    _validate_base_runtime_sources(value["files"])
    return value


def _validate_reference_smoke(
    path: Path,
    *,
    manifest_sha256: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _strict_json(path, "AFCC reference smoke manifest")
    _exact_keys(
        value,
        {
            "action_stats_sha256",
            "dataset_contract_artifact_sha256",
            "identity_sha256",
            "plan_sha256",
            "precompute_config_file_sha256",
            "precompute_gpu_runtime",
            "reference_checkpoint_sha256",
            "reference_execution_authority_sha256",
            "row_count",
            "rows",
            "scalar_reference_artifact_sha256",
            "schema_version",
            "source_manifest_sha256",
            "tensor_contract",
        },
        "AFCC reference smoke manifest",
    )
    expected_config_sha = hashlib.sha256(
        _candidate_config_bytes(
            BASE_REFERENCE_CONFIG,
            source_manifest_path,
            source_manifest_sha256,
        )
    ).hexdigest()
    runtime = value["precompute_gpu_runtime"]
    if (
        value["schema_version"]
        != "sana-phase6-afcc-reference-manifest-smoke-v1"
        or value["action_stats_sha256"] != ACTION_STATS_SHA256
        or value["dataset_contract_artifact_sha256"] != DATASET_CONTRACT_SHA256
        or value["identity_sha256"] != IDENTITY_SHA256
        or value["plan_sha256"] != PLAN_SHA256
        or value["precompute_config_file_sha256"] != expected_config_sha
        or value["reference_checkpoint_sha256"] != PHASE1_CHECKPOINT_SHA256
        or value["reference_execution_authority_sha256"]
        != hashlib.sha256(b"smoke-authority").hexdigest()
        or value["row_count"] != 1
        or type(value["rows"]) is not list
        or len(value["rows"]) != 1
        or value["scalar_reference_artifact_sha256"] != SCALAR_REFERENCE_SHA
        or value["source_manifest_sha256"] != source_manifest_sha256
        or value["tensor_contract"]
        != {
            "dtype": "BF16",
            "key": "action_facing_video_numerator",
            "shape": [20, 4, 20, 28, 112],
        }
        or not isinstance(runtime, dict)
        or any(runtime.get(key) != expected for key, expected in HARDWARE_AUTHORITY.items())
        or runtime.get("cuda_device_count") != 1
        or runtime.get("cuda_device_index") != 0
        or runtime.get("device_type") != "cuda"
        or runtime.get("world_size") != 1
    ):
        raise ValueError("AFCC reference smoke contract differs")
    row = _exact_keys(
        value["rows"][0],
        {
            "action_sigma",
            "common_input_trace_sha256",
            "dataset_contract_row_sha256",
            "dataset_index",
            "file_sha256",
            "global_step",
            "path",
            "plan_row_sha256",
            "size_bytes",
            "task_name",
            "tensor_sha256",
            "t0_reference_forward_trace_sha256",
        },
        "AFCC reference smoke row",
    )
    for key in (
        "common_input_trace_sha256",
        "dataset_contract_row_sha256",
        "file_sha256",
        "plan_row_sha256",
        "tensor_sha256",
        "t0_reference_forward_trace_sha256",
    ):
        _sha(row[key], f"smoke row {key}")
    if (
        row["global_step"] != 1
        or row["action_sigma"] != 1.0
        or row["dataset_index"] != 83202
        or row["task_name"] != "move_can_pot"
        or type(row["size_bytes"]) is not int
        or row["size_bytes"] < 1
    ):
        raise ValueError("AFCC reference smoke row metadata differs")
    shard = _pin(Path(row["path"]), row["file_sha256"])
    if shard["size_bytes"] != row["size_bytes"]:
        raise ValueError("AFCC reference smoke shard size differs")
    if _hash(path) != manifest_sha256:
        raise ValueError("AFCC reference smoke manifest SHA changed")
    return value, shard


def _validate_real_2b_smoke(
    path: Path,
    *,
    reference_manifest_sha256: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    value = _strict_json(path, "AFCC real-2B smoke")
    _exact_keys(
        value,
        {
            "afcc_gradient_probe_norms",
            "afcc_loss",
            "closed_loop",
            "config_sha256",
            "formal_evidence",
            "frozen_gradient_count",
            "global_step",
            "gradient_checkpointing_effective",
            "manifest_sha256",
            "optimizer_created",
            "peak_reserved_bytes",
            "schema_version",
            "source_manifest_sha256",
            "total_gradient_global_norm",
            "total_gradient_missing_tensor_count",
            "total_gradient_nonzero_tensor_count",
            "total_loss",
            "training_started",
            "wall_seconds",
        },
        "AFCC real-2B smoke",
    )
    expected_config_sha = hashlib.sha256(
        _candidate_config_bytes(
            CONTROL_CONFIG,
            source_manifest_path,
            source_manifest_sha256,
        )
    ).hexdigest()
    probes = value["afcc_gradient_probe_norms"]
    expected_probes = {
        "video_backbone.dit.blocks.0.attn.qkv.weight",
        "video_backbone.dit.blocks.19.attn.qkv.weight",
    }
    if (
        value["schema_version"] != "sana-phase6-afcc-real-2b-smoke-v1"
        or value["formal_evidence"] is not False
        or value["training_started"] is not False
        or value["optimizer_created"] is not False
        or value["closed_loop"] is not False
        or value["gradient_checkpointing_effective"] is not False
        or value["global_step"] != 1
        or value["config_sha256"] != expected_config_sha
        or value["manifest_sha256"] != reference_manifest_sha256
        or value["source_manifest_sha256"] != source_manifest_sha256
        or value["frozen_gradient_count"] != 0
        or value["total_gradient_missing_tensor_count"] != 0
        or value["total_gradient_nonzero_tensor_count"] != 380
        or type(probes) is not dict
        or set(probes) != expected_probes
        or type(value["peak_reserved_bytes"]) is not int
        or value["peak_reserved_bytes"] < 1
    ):
        raise ValueError("AFCC real-2B smoke contract differs")
    for name, probe in probes.items():
        _finite_positive(probe, f"AFCC gradient probe {name}")
    for key in (
        "afcc_loss",
        "total_gradient_global_norm",
        "total_loss",
        "wall_seconds",
    ):
        _finite_positive(value[key], f"AFCC real-2B smoke {key}")
    return value


def _immutable_formal_file(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    absolute = path.absolute()
    info = os.lstat(absolute)
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o400
        or info.st_nlink != 1
    ):
        raise ValueError(f"{name} must be immutable mode-0400 regular file")
    observed_sha = _hash(absolute)
    if observed_sha != _sha(expected_sha256, f"{name} SHA256"):
        raise ValueError(f"{name} SHA256 differs")
    return {
        "path": str(absolute),
        "sha256": observed_sha,
        "size_bytes": info.st_size,
    }


def _validate_formal_reference(
    manifest_path: Path,
    manifest_sha256: str,
    *,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    expected_manifest = REFERENCE_OUTPUT / "manifest_v1.json"
    if manifest_path.absolute() != expected_manifest:
        raise ValueError("AFCC formal reference manifest path differs")
    directory_info = os.lstat(REFERENCE_OUTPUT)
    if (
        stat.S_ISLNK(directory_info.st_mode)
        or not stat.S_ISDIR(directory_info.st_mode)
        or stat.S_IMODE(directory_info.st_mode) != 0o500
    ):
        raise ValueError("AFCC formal reference directory is not frozen mode 0500")
    _immutable_formal_file(
        expected_manifest, manifest_sha256, "AFCC formal reference manifest"
    )
    value = _strict_json(expected_manifest, "AFCC formal reference manifest")
    _exact_keys(
        value,
        {
            "action_stats_sha256",
            "dataset_contract_artifact_sha256",
            "identity_sha256",
            "plan_sha256",
            "precompute_config_file_sha256",
            "precompute_gpu_runtime",
            "reference_checkpoint_sha256",
            "reference_execution_authority_sha256",
            "row_count",
            "rows",
            "scalar_reference_artifact_sha256",
            "schema_version",
            "source_manifest_sha256",
            "tensor_contract",
        },
        "AFCC formal reference manifest",
    )
    runtime = value["precompute_gpu_runtime"]
    if (
        value["schema_version"] != "sana-phase6-afcc-reference-manifest-v1"
        or value["action_stats_sha256"] != ACTION_STATS_SHA256
        or value["dataset_contract_artifact_sha256"] != DATASET_CONTRACT_SHA256
        or value["identity_sha256"] != IDENTITY_SHA256
        or value["plan_sha256"] != PLAN_SHA256
        or value["precompute_config_file_sha256"] != _hash(REFERENCE_CONFIG_PATH)
        or value["reference_checkpoint_sha256"] != PHASE1_CHECKPOINT_SHA256
        or value["reference_execution_authority_sha256"]
        != _hash(REFERENCE_AUTHORITY_PATH)
        or value["row_count"] != 504
        or type(value["rows"]) is not list
        or len(value["rows"]) != 504
        or value["scalar_reference_artifact_sha256"] != SCALAR_REFERENCE_SHA
        or value["source_manifest_sha256"] != source_manifest_sha256
        or value["tensor_contract"]
        != {
            "dtype": "BF16",
            "key": "action_facing_video_numerator",
            "shape": [20, 4, 20, 28, 112],
        }
        or not isinstance(runtime, dict)
        or any(runtime.get(key) != expected for key, expected in HARDWARE_AUTHORITY.items())
        or runtime.get("cuda_device_count") != 1
        or runtime.get("cuda_device_index") != 0
        or runtime.get("device_type") != "cuda"
        or runtime.get("world_size") != 1
    ):
        raise ValueError("AFCC formal reference provenance differs")
    scalar = json.loads(SCALAR_REFERENCE.read_text())
    scalar_rows = scalar.get("rows")
    if type(scalar_rows) is not list or len(scalar_rows) != 504:
        raise ValueError("Phase1 scalar reference rows differ")

    import torch
    from safetensors import safe_open
    from sana_wam.train.action_facing_cache_reference import tensor_sha256

    expected_row_keys = {
        "action_sigma",
        "common_input_trace_sha256",
        "dataset_contract_row_sha256",
        "dataset_index",
        "file_sha256",
        "global_step",
        "path",
        "plan_row_sha256",
        "size_bytes",
        "task_name",
        "tensor_sha256",
        "t0_reference_forward_trace_sha256",
    }
    scalar_fields = (
        "action_sigma",
        "common_input_trace_sha256",
        "dataset_contract_row_sha256",
        "dataset_index",
        "global_step",
        "plan_row_sha256",
        "task_name",
        "t0_reference_forward_trace_sha256",
    )
    for step, (row, scalar_row) in enumerate(
        zip(value["rows"], scalar_rows, strict=True), start=1
    ):
        _exact_keys(row, expected_row_keys, f"AFCC formal reference row {step}")
        if any(row[key] != scalar_row[key] for key in scalar_fields):
            raise ValueError(f"AFCC formal reference row {step} scalar binding differs")
        expected_shard = REFERENCE_OUTPUT / f"row_{step:03d}.safetensors"
        if Path(row["path"]).absolute() != expected_shard:
            raise ValueError(f"AFCC formal reference row {step} path differs")
        shard = _immutable_formal_file(
            expected_shard, row["file_sha256"], f"AFCC reference shard {step}"
        )
        if shard["size_bytes"] != row["size_bytes"]:
            raise ValueError(f"AFCC formal reference row {step} size differs")
        with safe_open(expected_shard, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["action_facing_video_numerator"]:
                raise ValueError(f"AFCC formal reference row {step} key differs")
            tensor = handle.get_tensor("action_facing_video_numerator")
        if (
            tensor.dtype != torch.bfloat16
            or tuple(tensor.shape) != (20, 4, 20, 28, 112)
            or not bool(torch.isfinite(tensor).all().item())
            or tensor_sha256(tensor) != row["tensor_sha256"]
        ):
            raise ValueError(f"AFCC formal reference row {step} tensor differs")
    return value


def _common_prerequisites() -> dict[str, dict[str, Any]]:
    return {
        "action_stats": _pin(
            REPOSITORY
            / "logs/sana_principles_audit_20260722/trajectory_mixed_step0_run/action_stats.npy",
            "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc",
        ),
        "control_checkpoint": _pin(
            PHASE6_ROOT
            / "arms/T1_E1A0/phase6_20260725_T1_E1A0_planrowfix_r2/checkpoint_step_504.safetensors",
            "1f9eda034a067aa042bb7a0860b92c588c48572c8d05a73852428a1d0c11c429",
        ),
        "dataset_contract": _pin(
            PHASE6_ROOT / "artifacts/phase6_dataset_contract_504.json",
            "29a66c9e408ae68faa10705640ee414fbe9de96e28ab0841b1df60abbe6466ab",
        ),
        "initial_student_checkpoint": _pin(
            REPOSITORY
            / "logs/sana_principles_audit_20260722/trajectory_mixed_step0_run/checkpoint_step_0.safetensors",
            "e9549aff484da56eada21972f00fb3aa2f95f8a65397a55cf0f38c79e6254137",
        ),
        "phase1_reference_checkpoint": _pin(
            REPOSITORY
            / "logs/sana_principles_audit_20260722/phase1_weights_chunkwise_arch_run/checkpoint_step_0.safetensors",
            "ba7bb59fe7e44f6efd6a4cef79f66aff7ccf5dbe82a5a166634e35055e94f089",
        ),
        "plan": _pin(
            PHASE6_ROOT / "artifacts/phase6_task_plan_504.json",
            "7a1063df1d97fa0b8859dc86bf22f1ba3ad23703b6f6e449388dc950b8dc3ce6",
        ),
        "primary_aggregate": _pin(
            PHASE6_ROOT
            / "offline_qualification_v1/formal_execution_v1/aggregate/offline_qualification_aggregate_v1.json",
            "663b536d235f35cd728a70369d11409d36a3b9a4d9776210422286a3e88db099",
        ),
        "scalar_reference": _pin(SCALAR_REFERENCE, SCALAR_REFERENCE_SHA),
    }


def _design() -> dict[str, Any]:
    return {
        "comparison": {
            "control": "existing immutable T1_E1A0 step-504",
            "secondary_reference": "existing immutable T1_E1A1 step-504",
            "treatment": "fresh T1_E1A0_F1 from the common T1 step-0 checkpoint",
        },
        "effective_gradient_checkpointing": {
            "control": False,
            "reason": (
                "the existing AR compute_loss does not forward the configured flag"
            ),
            "treatment": False,
        },
        "intervention": {
            "adapter": "disabled A0",
            "completion_change_gate": (
                "exact 380-tensor video allowlist and at least one BF16 tensor "
                "bitwise changed; actual changed count is recorded"
            ),
            "denominator_constrained": False,
            "epsilon": 1.0e-6,
            "layer_selection": False,
            "layers": 20,
            "loss": (
                "mean valid-interface ||N_student-N_Phase1||^2 / "
                "(RMS(N_Phase1)^2+1e-6)"
            ),
            "numerator": "rotated_Q_action @ video_only_S_window.T",
            "reference_dtype": "BF16",
            "reference_shape_per_row": [20, 4, 20, 28, 112],
            "treatment_weight": 1.0,
            "video_chunks": 4,
        },
        "lifecycle": {
            "closed_loop": False,
            "fresh_v3_retry_authorized": True,
            "prior_artifacts": [
                {
                    "path": str(PHASE6_ROOT / "post_primary_afcc_v1"),
                    "status": "aborted authority-only publication",
                },
                {
                    "path": str(PHASE6_ROOT / "post_primary_afcc_v2"),
                    "status": "treatment partial after an over-strict completion gate",
                },
            ],
            "prior_roots_reused": False,
            "primary_verdict_changed": False,
            "publication_attempt": 3,
            "retry_starts_from_common_step0": True,
            "registered_before_formal_reference_gpu": True,
            "registered_before_formal_training": True,
            "selection_performed": False,
        },
        "prohibitions": [
            "DAgger",
            "closed-loop collection",
            "reranking or best-of-N",
            "context/step/arm selection",
            "layer selection",
            "rank expansion",
            "checkpoint overlay",
            "resume or retry into the same artifact path",
        ],
        "schema_version": "sana-phase6-afcc-design-authority-v1",
        "status": "registered",
    }


def build_reference_stage(args) -> None:
    if ARTIFACT_ROOT.exists() or ARTIFACT_ROOT.is_symlink():
        raise FileExistsError(f"refusing to reuse AFCC artifact root: {ARTIFACT_ROOT}")
    smoke_manifest = _pin(args.reference_smoke_manifest, args.reference_smoke_sha256)
    real_smoke = _pin(args.real2b_smoke, args.real2b_smoke_sha256)
    _pin(BASE_REFERENCE_CONFIG, BASE_REFERENCE_CONFIG_SHA)
    _pin(CONTROL_CONFIG, CONTROL_CONFIG_SHA)
    _pin(OLD_SOURCE_MANIFEST, OLD_SOURCE_MANIFEST_SHA)
    source_pins_before = {
        relative: _pin(REPOSITORY / relative) for relative in SOURCE_PATHS
    }
    _validate_base_runtime_sources(source_pins_before)
    smoke_source = _pin(
        args.smoke_source_manifest, args.smoke_source_manifest_sha256
    )
    _validate_afcc_source_manifest(
        args.smoke_source_manifest,
        args.smoke_source_manifest_sha256,
        expected_files=source_pins_before,
    )
    _smoke_value, smoke_shard = _validate_reference_smoke(
        args.reference_smoke_manifest,
        manifest_sha256=args.reference_smoke_sha256,
        source_manifest_path=Path(smoke_source["path"]),
        source_manifest_sha256=args.smoke_source_manifest_sha256,
    )
    _validate_real_2b_smoke(
        args.real2b_smoke,
        reference_manifest_sha256=args.reference_smoke_sha256,
        source_manifest_path=Path(smoke_source["path"]),
        source_manifest_sha256=args.smoke_source_manifest_sha256,
    )
    checks = [
        _run_check(
            [
                str(REPOSITORY / ".venv/bin/ruff"),
                "check",
                *SOURCE_PATHS,
            ]
        ),
        _run_check(
            [
                str(PYTHON),
                "-m",
                "pytest",
                "-q",
                "tests/test_action_facing_cache_consistency.py",
                "tests/test_action_facing_cache_reference.py",
                "tests/test_afcc_trainer_contract.py",
                "tests/test_sana_ar_padding_semantics.py",
                "tests/test_phase6_trainer_plan_contract.py",
                "tests/test_trainer_finetune_contract.py",
                "tests/test_action_nr_compute_loss.py",
                "tests/test_ar_config_resolves.py",
            ]
        ),
    ]
    source_pins_after = {
        relative: _pin(REPOSITORY / relative) for relative in SOURCE_PATHS
    }
    if source_pins_before != source_pins_after:
        raise RuntimeError("AFCC source changed while its gates were running")
    _validate_base_runtime_sources(source_pins_after)
    _validate_afcc_source_manifest(
        args.smoke_source_manifest,
        args.smoke_source_manifest_sha256,
        expected_files=source_pins_after,
    )
    common_prerequisites = _common_prerequisites()

    _mkdir(ARTIFACT_ROOT, 0o700)
    _mkdir(ARTIFACT_ROOT / "source_v1", 0o700)
    _mkdir(ARTIFACT_ROOT / "reference_precompute_v1", 0o700)
    design_sha = _write_json(DESIGN_PATH, _design())
    test_receipt = {
        "checks": checks,
        "formal_reference_gpu_started": False,
        "formal_training_started": False,
        "real_2b_smoke": real_smoke,
        "reference_smoke_manifest": smoke_manifest,
        "reference_smoke_shard": smoke_shard,
        "smoke_source_manifest": smoke_source,
        "schema_version": "sana-phase6-afcc-test-receipt-v1",
        "status": "pass",
    }
    test_sha = _write_json(TEST_RECEIPT_PATH, test_receipt)
    source_manifest = {
        "base_runtime_source_manifest": _pin(
            OLD_SOURCE_MANIFEST, OLD_SOURCE_MANIFEST_SHA
        ),
        "design_authority": _pin(DESIGN_PATH, design_sha),
        "files": source_pins_after,
        "formal_reference_gpu_started": False,
        "formal_training_started": False,
        "schema_version": "sana-phase6-afcc-source-manifest-v1",
        "status": "pass",
        "test_receipt": _pin(TEST_RECEIPT_PATH, test_sha),
    }
    source_sha = _write_json(SOURCE_MANIFEST_PATH, source_manifest)

    config = yaml.safe_load(BASE_REFERENCE_CONFIG.read_text())
    config["training"]["phase6_code_source_manifest"] = str(SOURCE_MANIFEST_PATH)
    config["training"]["phase6_code_source_manifest_sha256"] = source_sha
    config["training"]["phase1_action_facing_cache_precompute_mode"] = True
    config_bytes = yaml.safe_dump(
        config,
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
    ).encode("utf-8")
    config_sha = _write(REFERENCE_CONFIG_PATH, config_bytes)
    reference_script = REPOSITORY / "scripts/precompute_phase1_action_facing_cache_reference.py"
    authority_argv = [
        str(PYTHON),
        str(reference_script),
        "--mode",
        "formal",
        "--config",
        str(REFERENCE_CONFIG_PATH),
        "--config-sha256",
        config_sha,
        "--source-manifest",
        str(SOURCE_MANIFEST_PATH),
        "--source-manifest-sha256",
        source_sha,
        "--scalar-reference",
        str(SCALAR_REFERENCE),
        "--scalar-reference-sha256",
        SCALAR_REFERENCE_SHA,
        "--execution-authority",
        str(REFERENCE_AUTHORITY_PATH),
        "--execution-authority-sha256",
        "<AFCC_REFERENCE_EXECUTION_AUTHORITY_SHA256>",
        "--output-dir",
        str(REFERENCE_OUTPUT),
        "--steps",
        "1-504",
    ]
    prerequisites = dict(common_prerequisites)
    prerequisites.update(
        {
            "design_authority": _pin(DESIGN_PATH, design_sha),
            "test_receipt": _pin(TEST_RECEIPT_PATH, test_sha),
        }
    )
    authority = {
        "argv": authority_argv,
        "config": {"path": str(REFERENCE_CONFIG_PATH), "sha256": config_sha},
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONHASHSEED": "0",
            "WORLD_SIZE": "1",
        },
        "hardware": HARDWARE_AUTHORITY,
        "lifecycle": {
            "closed_loop": False,
            "optimizer_created": False,
            "resume_allowed": False,
            "training_started": False,
        },
        "output_directory": str(REFERENCE_OUTPUT),
        "prerequisites": prerequisites,
        "scalar_reference": {
            "path": str(SCALAR_REFERENCE),
            "sha256": SCALAR_REFERENCE_SHA,
        },
        "schema_version": "sana-phase6-afcc-reference-execution-authority-v1",
        "script": {
            "path": str(reference_script),
            "sha256": _hash(reference_script),
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST_PATH),
            "sha256": source_sha,
        },
        "status": "authorized",
        "steps": [1, 504],
    }
    authority_sha = _write_json(REFERENCE_AUTHORITY_PATH, authority)
    print(
        json.dumps(
            {
                "design_authority_sha256": design_sha,
                "reference_config_sha256": config_sha,
                "reference_execution_authority_sha256": authority_sha,
                "source_manifest_sha256": source_sha,
                "test_receipt_sha256": test_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_training_stage(args) -> None:
    if not ARTIFACT_ROOT.is_dir() or (ARTIFACT_ROOT / "training_v1").exists():
        raise FileExistsError("AFCC training publication stage is absent or reused")
    _pin(CONTROL_CONFIG, CONTROL_CONFIG_SHA)
    source_sha = _hash(SOURCE_MANIFEST_PATH)
    _validate_afcc_source_manifest(SOURCE_MANIFEST_PATH, source_sha)
    _validate_formal_reference(
        args.reference_manifest,
        args.reference_manifest_sha256,
        source_manifest_sha256=source_sha,
    )
    common_prerequisites = _common_prerequisites()
    _pin(CONTROL_CONFIG, CONTROL_CONFIG_SHA)
    _validate_afcc_source_manifest(SOURCE_MANIFEST_PATH, source_sha)
    _mkdir(ARTIFACT_ROOT / "training_v1", 0o700)
    config = yaml.safe_load(CONTROL_CONFIG.read_text())
    architecture = config["model"]["architecture"]
    architecture["action_facing_cache_consistency_weight"] = 1.0
    training = config["training"]
    for key in tuple(training):
        if key.startswith("phase6_preflight_") or key in {
            "phase6_arm",
            "phase6_launch_required",
            "phase6_run_id",
            "phase6_scientific_projection",
            "phase6_scientific_projection_sha256",
        }:
            training.pop(key)
    training["phase6_code_source_manifest"] = str(SOURCE_MANIFEST_PATH)
    training["phase6_code_source_manifest_sha256"] = source_sha
    training["phase1_action_facing_cache_reference_manifest"] = str(
        args.reference_manifest
    )
    training["phase1_action_facing_cache_reference_manifest_sha256"] = (
        args.reference_manifest_sha256
    )
    training["post_primary_afcc_condition"] = "T1_E1A0_F1"
    training["post_primary_afcc_run_id"] = "post_primary_afcc_T1_E1A0_F1_v1"
    training["output_dir"] = str(TRAINING_OUTPUT)
    config_bytes = yaml.safe_dump(
        config,
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
    ).encode("utf-8")
    config_sha = _write(TRAINING_CONFIG_PATH, config_bytes)
    training_script = REPOSITORY / "scripts/train_post_primary_afcc.py"
    authority_argv = [
        str(PYTHON),
        str(training_script),
        "--config",
        str(TRAINING_CONFIG_PATH),
        "--config-sha256",
        config_sha,
        "--execution-authority",
        str(TRAINING_AUTHORITY_PATH),
        "--execution-authority-sha256",
        "<AFCC_TRAINING_EXECUTION_AUTHORITY_SHA256>",
    ]
    prerequisites = dict(common_prerequisites)
    prerequisites.update(
        {
            "control_config": _pin(CONTROL_CONFIG, CONTROL_CONFIG_SHA),
            "design_authority": _pin(DESIGN_PATH),
            "reference_execution_authority": _pin(REFERENCE_AUTHORITY_PATH),
            "test_receipt": _pin(TEST_RECEIPT_PATH),
        }
    )
    authority = {
        "argv": authority_argv,
        "config": {"path": str(TRAINING_CONFIG_PATH), "sha256": config_sha},
        "control": "T1_E1A0",
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONHASHSEED": "0",
            "WORLD_SIZE": "1",
        },
        "hardware": HARDWARE_AUTHORITY,
        "lifecycle": {
            "closed_loop": False,
            "resume_allowed": False,
            "retry_allowed": False,
            "selection_allowed": False,
            "training_started": False,
        },
        "output_directory": str(TRAINING_OUTPUT),
        "prerequisites": prerequisites,
        "reference_manifest": {
            "path": str(args.reference_manifest),
            "sha256": args.reference_manifest_sha256,
        },
        "schema_version": "sana-phase6-afcc-training-execution-authority-v1",
        "script": {
            "path": str(training_script),
            "sha256": _hash(training_script),
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST_PATH),
            "sha256": source_sha,
        },
        "status": "authorized",
        "treatment": "T1_E1A0_F1",
    }
    authority_sha = _write_json(TRAINING_AUTHORITY_PATH, authority)
    print(
        json.dumps(
            {
                "training_config_sha256": config_sha,
                "training_execution_authority_sha256": authority_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    reference = sub.add_parser("reference")
    reference.add_argument("--reference-smoke-manifest", type=Path, required=True)
    reference.add_argument("--reference-smoke-sha256", required=True)
    reference.add_argument("--real2b-smoke", type=Path, required=True)
    reference.add_argument("--real2b-smoke-sha256", required=True)
    reference.add_argument("--smoke-source-manifest", type=Path, required=True)
    reference.add_argument("--smoke-source-manifest-sha256", required=True)
    training = sub.add_parser("training")
    training.add_argument("--reference-manifest", type=Path, required=True)
    training.add_argument("--reference-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != REPOSITORY:
        raise RuntimeError(f"AFCC builder requires cwd={REPOSITORY}")
    if args.stage == "reference":
        build_reference_stage(args)
    else:
        build_training_stage(args)


if __name__ == "__main__":
    main()
