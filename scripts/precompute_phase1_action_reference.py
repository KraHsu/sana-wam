#!/usr/bin/env python3
"""Build or live-verify the Phase1 T0 action-reference table on one GPU.

The script is deliberately forward-only: no optimizer, backward, resume, or
closed-loop path exists. It must not be run until the source manifest and a
``reference_precompute`` preflight request are frozen and independently pinned.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

REFERENCE_RESULT_KEY = "phase6_action_unweighted_mse"
CUDA_DEVICE_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def verify_file(path: str | os.PathLike[str], expected: str, name: str) -> str:
    resolved = str(Path(path).expanduser().resolve(strict=True))
    observed = sha256_file(resolved)
    if observed != require_sha256(expected, name):
        raise ValueError(f"{name} SHA mismatch: expected {expected}, got {observed}")
    return resolved


def atomic_exclusive_write(path: Path, data: bytes) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"artifact parent must already exist: {path.parent}")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "spot"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--config-file-sha256", required=True)
        subparser.add_argument("--preflight-request", type=Path, required=True)
        subparser.add_argument("--preflight-request-sha256", required=True)
        subparser.add_argument("--preflight-report", type=Path, required=True)
        subparser.add_argument("--source-manifest", type=Path, required=True)
        subparser.add_argument("--source-manifest-sha256", required=True)
    build = subparsers.choices["build"]
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--build-manifest", type=Path, required=True)
    spot = subparsers.choices["spot"]
    spot.add_argument("--reference-artifact", type=Path, required=True)
    spot.add_argument("--reference-artifact-sha256", required=True)
    spot.add_argument("--preflight-report-sha256", required=True)
    spot.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def precompute_execution_contract() -> dict[str, Any]:
    return {
        "autograd_context": "torch.no_grad",
        "autograd_enabled": False,
        "backward_called": False,
        "closed_loop": False,
        "forward_only": True,
        "gradient_checkpointing": False,
        "model_mode": "eval",
        "model_result_key": REFERENCE_RESULT_KEY,
        "optimizer_created": False,
        "phase6_plan_rows_required": True,
        "reference_input_key": None,
        "resume_allowed": False,
        "row_context_contract": (
            "plan-row prompt plus dataset-contract logical-length/episode/instruction"
        ),
        "row_noise_contract": "phase6_plan_row.domain_seeds",
        "spot_check_global_steps": [1, 253, 504],
        "training_started": False,
    }


def load_config(path: Path, expected_sha256: str):
    resolved = verify_file(path, expected_sha256, "precompute config file")
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(resolved)
    plain = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    if not isinstance(plain, dict):
        raise ValueError("precompute config must resolve to one mapping")
    for key in ("model", "dataloader", "training"):
        if not isinstance(plain.get(key), dict):
            raise ValueError(f"precompute config is missing {key}")
    artifact_config = {
        "schema_version": "sana-phase6-action-reference-precompute-config-v1",
        "model": plain["model"],
        "dataloader": plain["dataloader"],
        "training": plain["training"],
        "execution": precompute_execution_contract(),
    }
    return cfg, artifact_config, resolved


def load_or_run_reference_preflight(args: argparse.Namespace) -> dict[str, Any]:
    from sana_wam.train.phase6_preflight import (
        load_and_revalidate_phase6_preflight_report,
        run_phase6_preflight,
    )

    request_sha256 = require_sha256(
        args.preflight_request_sha256, "preflight request SHA256"
    )
    if args.command == "build":
        report = run_phase6_preflight(
            args.preflight_request,
            args.preflight_report,
            expected_request_sha256=request_sha256,
            expected_purpose="reference_precompute",
        )
    else:
        # Spot verification reuses the one authorization issued before the first
        # table-build GPU forward. It never rewrites or replaces that report.
        report = load_and_revalidate_phase6_preflight_report(
            args.preflight_request,
            args.preflight_report,
            expected_request_sha256=request_sha256,
            expected_report_sha256=require_sha256(
                args.preflight_report_sha256,
                "preflight report SHA256",
            ),
            expected_purpose="reference_precompute",
        )
    if (
        report.get("purpose") != "reference_precompute"
        or report.get("status") != "pass"
        or report.get("reference_gpu_started") is not False
        or report.get("training_started") is not False
    ):
        raise RuntimeError("reference-precompute preflight report is not authorizing")
    return report


def require_fresh_output(path: Path, output_root: str, name: str) -> Path:
    root = Path(output_root).resolve(strict=True)
    parent = path.expanduser().absolute().parent.resolve(strict=True)
    target = parent / path.name
    if os.path.commonpath((str(target), str(root))) != str(root):
        raise ValueError(f"{name} must be inside the authorized output root")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite {name}: {target}")
    return target


def cuda_driver_version() -> str:
    """Return the loaded CUDA driver API version without invoking a shell."""

    try:
        driver = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        if driver.cuInit(0) != 0 or driver.cuDriverGetVersion(
            ctypes.byref(version)
        ) != 0:
            raise RuntimeError("CUDA driver query returned an error")
    except (OSError, AttributeError, RuntimeError) as exc:
        raise RuntimeError(f"cannot query CUDA driver version: {exc}") from exc
    major = version.value // 1000
    minor = (version.value % 1000) // 10
    return f"{major}.{minor}"


def _cuda_device_uuid(device_index: int) -> str:
    """Read the physical CUDA UUID through the driver API."""

    try:
        driver = ctypes.CDLL("libcuda.so.1")
        device = ctypes.c_int()
        raw_uuid = (ctypes.c_ubyte * 16)()
        if driver.cuInit(0) != 0:
            raise RuntimeError("cuInit failed")
        if driver.cuDeviceGet(ctypes.byref(device), int(device_index)) != 0:
            raise RuntimeError("cuDeviceGet failed")
        if driver.cuDeviceGetUuid(ctypes.byref(raw_uuid), device) != 0:
            raise RuntimeError("cuDeviceGetUuid failed")
    except (OSError, AttributeError, RuntimeError) as exc:
        raise RuntimeError(f"cannot query CUDA device UUID: {exc}") from exc
    hex_value = bytes(raw_uuid).hex()
    return "GPU-" + "-".join(
        (
            hex_value[0:8],
            hex_value[8:12],
            hex_value[12:16],
            hex_value[16:20],
            hex_value[20:32],
        )
    )


def _valid_property_cuda_uuid(value: Any) -> str | None:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str) and CUDA_DEVICE_UUID_PATTERN.fullmatch(value):
        return value
    return None


def gpu_runtime_descriptor(torch, device) -> dict[str, Any]:
    capability = torch.cuda.get_device_capability(device)
    properties = torch.cuda.get_device_properties(device)
    device_index = int(device.index or 0)
    device_uuid = _valid_property_cuda_uuid(getattr(properties, "uuid", None))
    if device_uuid is None:
        device_uuid = _cuda_device_uuid(device_index)
    return {
        "compute_capability": [int(capability[0]), int(capability[1])],
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_index": device_index,
        "cuda_device_name": str(torch.cuda.get_device_name(device)),
        "cuda_device_uuid": device_uuid,
        "cuda_driver_version": cuda_driver_version(),
        "device_type": "cuda",
        "multi_processor_count": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def build_bound_dataset(cfg, *, expected_source_manifest_sha256: str):
    from sana_wam.dataloader import robotwin_dataset as robotwin_dataset_module
    from sana_wam.dataloader.mixture import build_training_dataset
    from sana_wam.dataloader.phase6_dataset_contract import Phase6DatasetContract
    from sana_wam.dataloader.robotwin_plan_binding import PlanBoundRoboTwinDataset
    from sana_wam.dataloader.task_sample_plan import (
        SamplerContract,
        TaskRoundRobinPlan,
    )
    from sana_wam.dataloader.transforms import multiview as multiview_module
    from sana_wam.dataloader.transforms import normalize as normalize_module
    from sana_wam.dataloader.transforms import rotation as rotation_module

    training = cfg.training
    dataset = build_training_dataset(cfg.dataloader, split="train")
    action_stats_sha256 = require_sha256(
        str(training.action_stats_sha256), "action stats SHA256"
    )
    verify_file(dataset.action_stats_path, action_stats_sha256, "action stats")
    registry = verify_file(
        training.phase6_registry,
        training.phase6_registry_sha256,
        "Phase-6 registry",
    )
    plan_path = verify_file(
        training.phase6_plan_artifact,
        training.phase6_plan_artifact_sha256,
        "Phase-6 task plan",
    )
    dataset_contract_path = verify_file(
        training.phase6_dataset_contract_artifact,
        training.phase6_dataset_contract_artifact_sha256,
        "Phase-6 dataset contract",
    )
    contract = SamplerContract.from_registry(registry)
    plan = TaskRoundRobinPlan.from_artifact_bytes(
        Path(plan_path).read_bytes(),
        expected_plan_sha256=training.phase6_plan_sha256,
        expected_identity_sha256=training.phase6_identity_sha256,
    )
    dataset_contract_sha256 = require_sha256(
        str(training.phase6_dataset_contract_artifact_sha256),
        "dataset-contract SHA256",
    )
    dataset_contract = Phase6DatasetContract.from_artifact_bytes(
        Path(dataset_contract_path).read_bytes(),
        expected_artifact_sha256=dataset_contract_sha256,
        expected_plan_sha256=plan.plan_sha256,
        expected_identity_sha256=plan.identity_sha256,
        expected_action_stats_sha256=action_stats_sha256,
    )
    preprocessing_modules = {
        "sana_wam.dataloader.robotwin_dataset": robotwin_dataset_module,
        "sana_wam.dataloader.transforms.multiview": multiview_module,
        "sana_wam.dataloader.transforms.normalize": normalize_module,
        "sana_wam.dataloader.transforms.rotation": rotation_module,
    }
    source_paths = {
        source_id: inspect.getsourcefile(module)
        for source_id, module in preprocessing_modules.items()
    }
    if any(path is None for path in source_paths.values()):
        raise RuntimeError("cannot locate dataset preprocessing sources")
    dataset_contract.validate_runtime(
        dataset,
        plan,
        dataloader_config=cfg.dataloader,
        action_stats_sha256=action_stats_sha256,
        preprocessing_source_paths=source_paths,
        train_tasks=contract.train_tasks,
    )
    bound = PlanBoundRoboTwinDataset(
        dataset,
        plan,
        contract=contract,
        dataset_type=str(cfg.dataloader.type),
        world_size=1,
        expected_plan_sha256=plan.plan_sha256,
        expected_identity_sha256=plan.identity_sha256,
        dataset_contract=dataset_contract,
    )
    if str(training.phase6_code_source_manifest_sha256) != (
        expected_source_manifest_sha256
    ):
        raise ValueError("config/source-manifest SHA256 mismatch")
    return bound, plan, dataset_contract


def build_phase1_architecture(cfg, dataset, device):
    import numpy as np
    import torch

    from sana_wam.config import flatten_model_cfg
    from sana_wam.model import build_architecture

    training = cfg.training
    checkpoint = verify_file(
        training.init_checkpoint,
        training.init_checkpoint_sha256,
        "Phase1 reference checkpoint",
    )
    if str(training.init_checkpoint_sha256) != str(
        training.phase1_reference_checkpoint_sha256
    ):
        raise ValueError("precompute init and Phase1 checkpoint SHA256 values differ")
    architecture = build_architecture(flatten_model_cfg(cfg.model))
    architecture.set_dtype_device(torch.bfloat16, device)
    architecture.load_checkpoint(checkpoint, strict=True)
    architecture.init_training_schedulers(1000)
    architecture.set_training_runtime(use_gradient_checkpointing=False)
    architecture.eval()
    if getattr(
        architecture.video_backbone, "continuous_timestep_conditioning", None
    ) is not False:
        raise RuntimeError("Phase1 reference runtime is not exact T0")
    if architecture.action_video_memory_adapter is not None:
        raise RuntimeError("Phase1 reference runtime unexpectedly has an action adapter")
    if any(parameter.requires_grad for parameter in architecture.parameters()):
        for parameter in architecture.parameters():
            parameter.requires_grad_(False)
    stats = dataset.action_stats
    stats = stats() if callable(stats) else stats
    if stats is not None:
        mean = torch.from_numpy(stats["mean"].astype(np.float32))
        std = torch.from_numpy(np.maximum(stats["std"].astype(np.float32), 1.0e-3))
        architecture.action_mean.copy_(mean)
        architecture.action_std.copy_(std)
    return architecture


def seed_reference_process(seed: int) -> None:
    import numpy as np
    import torch

    if type(seed) is not int or seed != 20260724:
        raise ValueError("reference precompute seed must be exactly 20260724")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_one(architecture, dataset, plan, global_step: int) -> tuple[float, str, str]:
    import torch

    row = plan.rows[global_step - 1]
    sample = dataset[row.identity.dataset_index]
    batch = [sample]
    raw_row = sample.get("phase6_plan_row")
    if raw_row is None or raw_row.get("global_step") != global_step:
        raise RuntimeError("materialized reference row differs from requested step")
    inputs = architecture.prepare_inputs(batch)
    if inputs.get("phase6_plan_rows") != (raw_row,):
        raise RuntimeError("prepare_inputs changed the Phase-6 reference row")
    if "phase1_action_reference_error" in inputs:
        raise RuntimeError("reference precompute cannot consume a reference table")
    inputs["phase6_common_input_trace_required"] = True
    inputs["phase6_t0_reference_forward_trace_required"] = True
    # The architecture consumes video-noise and action-noise from this exact
    # plan row. reference-query is provenance/spot context, never a substitute.
    with torch.no_grad():
        result = architecture.compute_loss(
            **inputs,
            lambda_video=1.0,
            lambda_action=0.0,
        )
    error = result.get(REFERENCE_RESULT_KEY)
    if not isinstance(error, torch.Tensor):
        raise TypeError(f"model result {REFERENCE_RESULT_KEY!r} must be a tensor")
    if error.shape != () or error.dtype != torch.float32 or error.requires_grad:
        raise ValueError("reference result must be one detached FP32 scalar")
    value = float(error.detach().cpu().item())
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("reference result must be finite and non-negative")
    common_trace = result.get("phase6_common_input_trace_sha256")
    t0_trace = result.get("phase6_timestep_forward_trace_sha256")
    for name, trace in (
        ("common input", common_trace),
        ("T0 reference forward", t0_trace),
    ):
        require_sha256(trace, f"{name} trace SHA256")
    return value, common_trace, t0_trace


def validate_spot_reference_envelope(
    envelope: dict[str, Any],
    *,
    config_sha256: str,
    preflight_request_sha256: str,
    preflight_report_sha256: str,
    source_sha256: str,
    artifact_config: dict[str, Any],
) -> None:
    """Bind spot inputs to the table's content-addressed provenance."""

    expected = {
        "precompute_config_file_sha256": config_sha256,
        "precompute_preflight_request_sha256": preflight_request_sha256,
        "precompute_preflight_report_sha256": preflight_report_sha256,
        "precompute_source_manifest_sha256": source_sha256,
    }
    if any(envelope.get(key) != value for key, value in expected.items()):
        raise RuntimeError("reference envelope/preflight provenance differs")
    if envelope.get("precompute_config") != artifact_config:
        raise RuntimeError("reference envelope/precompute config differs")


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("reference precompute requires WORLD_SIZE=1")
    config_sha256 = require_sha256(
        args.config_file_sha256, "precompute config-file SHA256"
    )
    source_sha256 = require_sha256(
        args.source_manifest_sha256, "precompute source-manifest SHA256"
    )
    verify_file(args.source_manifest, source_sha256, "precompute source manifest")
    preflight_report = load_or_run_reference_preflight(args)
    if preflight_report["pins"]["observed_source_manifest_sha256"] != source_sha256:
        raise RuntimeError("preflight/source-manifest SHA256 mismatch")
    cfg, artifact_config, config_path = load_config(args.config, config_sha256)

    from sana_wam.train.action_reference_table import (
        canonical_json_bytes,
        validate_precompute_config,
        validate_reference_artifact_envelope_bytes,
    )

    training = cfg.training
    validate_precompute_config(
        artifact_config,
        reference_checkpoint_sha256=str(training.init_checkpoint_sha256),
        action_stats_sha256=str(training.action_stats_sha256),
        plan_sha256=str(training.phase6_plan_sha256),
        identity_sha256=str(training.phase6_identity_sha256),
        dataset_contract_artifact_sha256=str(
            training.phase6_dataset_contract_artifact_sha256
        ),
    )
    observed_input_config = preflight_report["pins"].get(
        "observed_input_config_sha256"
    )
    if observed_input_config != config_sha256:
        raise RuntimeError("preflight/input-config SHA256 mismatch")
    preflight_request_sha256 = require_sha256(
        preflight_report["request_sha256"], "validated preflight request SHA256"
    )
    preflight_report_bytes = canonical_json_bytes(preflight_report)
    if Path(args.preflight_report).read_bytes() != preflight_report_bytes:
        raise RuntimeError("preflight report bytes changed after authorization")
    preflight_report_sha256 = hashlib.sha256(preflight_report_bytes).hexdigest()
    output_root = preflight_report["effective_output_root"]["absolute_path"]
    if args.command == "build":
        output_path = require_fresh_output(
            args.output, output_root, "reference artifact"
        )
        build_manifest_path = require_fresh_output(
            args.build_manifest, output_root, "reference build manifest"
        )
        if output_path == build_manifest_path:
            raise ValueError("reference artifact and build manifest paths must differ")
    else:
        output_path = require_fresh_output(args.output, output_root, "spot artifact")
        reference_path = verify_file(
            args.reference_artifact,
            args.reference_artifact_sha256,
            "reference artifact",
        )
        envelope = validate_reference_artifact_envelope_bytes(
            Path(reference_path).read_bytes(),
            expected_artifact_sha256=args.reference_artifact_sha256,
        )
        validate_spot_reference_envelope(
            envelope,
            config_sha256=config_sha256,
            preflight_request_sha256=preflight_request_sha256,
            preflight_report_sha256=preflight_report_sha256,
            source_sha256=source_sha256,
            artifact_config=artifact_config,
        )

    # No torch or CUDA query occurs before the reference-purpose preflight passes.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("reference precompute requires one CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu_runtime = gpu_runtime_descriptor(torch, device)
    if args.command == "spot" and hashlib.sha256(
        canonical_json_bytes(gpu_runtime)
    ).hexdigest() != envelope["precompute_gpu_runtime_sha256"]:
        raise RuntimeError("spot GPU runtime differs from table-build runtime")
    seed_reference_process(int(cfg.training.seed))
    dataset, plan, dataset_contract = build_bound_dataset(
        cfg,
        expected_source_manifest_sha256=source_sha256,
    )
    architecture = build_phase1_architecture(cfg, dataset, device)

    from sana_wam.train.action_reference_table import (
        ActionReferenceTable,
        SPOT_CHECK_GLOBAL_STEPS,
        SPOT_CHECK_SCHEMA_VERSION,
        canonical_json_bytes,
        float32_bits,
        validate_live_spot_artifact_bytes,
    )

    if args.command == "build":
        forwards = [
            forward_one(architecture, dataset, plan, global_step)
            for global_step in range(1, 505)
        ]
        errors = [item[0] for item in forwards]
        common_traces = [item[1] for item in forwards]
        t0_traces = [item[2] for item in forwards]
        table = ActionReferenceTable.build(
            plan,
            dataset_contract.rows,
            errors,
            common_traces,
            t0_traces,
            dataset_contract_artifact_sha256=(
                cfg.training.phase6_dataset_contract_artifact_sha256
            ),
            reference_checkpoint_sha256=cfg.training.init_checkpoint_sha256,
            action_stats_sha256=cfg.training.action_stats_sha256,
            precompute_config_file_sha256=config_sha256,
            precompute_config=artifact_config,
            precompute_gpu_runtime=gpu_runtime,
            precompute_preflight_request_sha256=preflight_request_sha256,
            precompute_preflight_report_sha256=preflight_report_sha256,
            precompute_source_manifest_sha256=source_sha256,
        )
        artifact_bytes = table.to_artifact_bytes(plan, dataset_contract.rows)
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        manifest = {
            "schema_version": "sana-phase6-action-reference-build-v2",
            "reference_artifact_path": str(output_path),
            "reference_artifact_sha256": artifact_sha256,
            "precompute_config_path": config_path,
            "precompute_config_file_sha256": config_sha256,
            "precompute_config_sha256": table.precompute_config_sha256,
            "precompute_gpu_runtime": gpu_runtime,
            "precompute_gpu_runtime_sha256": (
                table.precompute_gpu_runtime_sha256
            ),
            "precompute_preflight_request_sha256": preflight_request_sha256,
            "precompute_preflight_report_sha256": preflight_report_sha256,
            "precompute_source_manifest_sha256": source_sha256,
            "dataset_contract_artifact_sha256": (
                table.dataset_contract_artifact_sha256
            ),
            "reference_checkpoint_sha256": table.reference_checkpoint_sha256,
            "action_stats_sha256": table.action_stats_sha256,
            "row_count": len(table.error_bits),
            "gpu_forward_executed": True,
            "training_started": False,
            "closed_loop": False,
        }
        atomic_exclusive_write(output_path, artifact_bytes)
        atomic_exclusive_write(
            build_manifest_path, canonical_json_bytes(manifest)
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    table = ActionReferenceTable.from_artifact_bytes(
        Path(reference_path).read_bytes(),
        plan,
        dataset_contract.rows,
        expected_artifact_sha256=args.reference_artifact_sha256,
        expected_dataset_contract_artifact_sha256=(
            cfg.training.phase6_dataset_contract_artifact_sha256
        ),
        expected_reference_checkpoint_sha256=cfg.training.init_checkpoint_sha256,
        expected_action_stats_sha256=cfg.training.action_stats_sha256,
        expected_precompute_source_manifest_sha256=source_sha256,
    )
    spot_rows = []
    for global_step in SPOT_CHECK_GLOBAL_STEPS:
        observed, observed_common_trace, observed_t0_trace = forward_one(
            architecture, dataset, plan, global_step
        )
        observed_bits = float32_bits(observed)
        expected_bits = table.error_bits[global_step - 1]
        spot_rows.append(
            {
                "global_step": global_step,
                "expected_error_float32_bits": expected_bits,
                "observed_error_float32_bits": observed_bits,
                "exact_match": (
                    observed_bits == expected_bits
                    and observed_common_trace
                    == table.common_input_trace_sha256[global_step - 1]
                    and observed_t0_trace
                    == table.t0_reference_forward_trace_sha256[global_step - 1]
                ),
                "common_input_trace_sha256": observed_common_trace,
                "plan_row_sha256": table.plan_row_sha256[global_step - 1],
                "dataset_contract_row_sha256": (
                    table.dataset_contract_row_sha256[global_step - 1]
                ),
                "reference_forward_context_sha256": (
                    table.reference_forward_context_sha256[global_step - 1]
                ),
                "t0_reference_forward_trace_sha256": observed_t0_trace,
            }
        )
    spot = {
        "schema_version": SPOT_CHECK_SCHEMA_VERSION,
        "verification_passed": all(row["exact_match"] for row in spot_rows),
        "gpu_forward_executed": True,
        "training_started": False,
        "closed_loop": False,
        "reference_artifact_sha256": args.reference_artifact_sha256,
        "plan_sha256": table.plan_sha256,
        "identity_sha256": table.identity_sha256,
        "dataset_contract_artifact_sha256": (
            table.dataset_contract_artifact_sha256
        ),
        "reference_checkpoint_sha256": table.reference_checkpoint_sha256,
        "action_stats_sha256": table.action_stats_sha256,
        "precompute_config_file_sha256": table.precompute_config_file_sha256,
        "precompute_config_sha256": table.precompute_config_sha256,
        "precompute_gpu_runtime_sha256": table.precompute_gpu_runtime_sha256,
        "precompute_preflight_report_sha256": (
            table.precompute_preflight_report_sha256
        ),
        "precompute_preflight_request_sha256": (
            table.precompute_preflight_request_sha256
        ),
        "precompute_source_manifest_sha256": (
            table.precompute_source_manifest_sha256
        ),
        "spot_gpu_runtime": gpu_runtime,
        "spot_gpu_runtime_sha256": hashlib.sha256(
            canonical_json_bytes(gpu_runtime)
        ).hexdigest(),
        "spot_preflight_report_sha256": preflight_report_sha256,
        "spot_preflight_request_sha256": preflight_request_sha256,
        "global_steps": list(SPOT_CHECK_GLOBAL_STEPS),
        "rows": spot_rows,
    }
    spot_bytes = canonical_json_bytes(spot)
    if not spot["verification_passed"]:
        raise RuntimeError("live Phase1 spot re-forward differs from reference table")
    validate_live_spot_artifact_bytes(
        spot_bytes,
        table,
        plan,
        dataset_contract.rows,
        expected_artifact_sha256=hashlib.sha256(spot_bytes).hexdigest(),
        expected_reference_artifact_sha256=args.reference_artifact_sha256,
    )
    atomic_exclusive_write(output_path, spot_bytes)
    print(json.dumps(spot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
