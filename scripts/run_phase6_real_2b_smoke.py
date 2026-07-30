#!/usr/bin/env python3
"""Run the formal single-GPU Phase-6 real-2B smoke gate.

PyTorch is intentionally imported only after the canonical request, all file
pins, the embedded five-arm projections, and the runtime API have been checked.
The project-specific construction seam is
``sana_wam.train.phase6_smoke_runtime.create_phase6_smoke_runtime``; this runner
owns the actual forward/autograd/backward/checkpoint/memory integrity checks.
It never creates an optimizer and never calls the training loop.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import struct
import sys
import tempfile
from typing import Any

from sana_wam.train.phase6_smoke_gate import (
    MINIMUM_FULL_CHECKPOINT_SIZE_BYTES,
    MINIMUM_REAL_2B_VIDEO_PARAMETER_COUNT,
    PEAK_RESERVED_LIMIT_BYTES,
    RUNTIME_SUPPORT_API_VERSION,
    SMOKE_ARTIFACT_ROLE,
    SMOKE_ARTIFACT_SCHEMA_VERSION,
    SMOKE_AUTHORIZATION,
    canonical_json_bytes,
    validate_phase6_smoke_artifact_bytes,
)


_HASH_BLOCK_SIZE = 8 * 1024 * 1024
_ADAPTER_PREFIX = "action_video_memory_adapter."
_RECEIPT_SCHEMA_VERSION = "sana-phase6-real-2b-smoke-receipt-v1"


class SmokeExecutionError(RuntimeError):
    """Raised before publishing an artifact when a live gate fails."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _regular_nonsymlink(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise SmokeExecutionError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SmokeExecutionError(f"{label} must be a regular non-symlink file: {path}")
    return info


def _stable_pinned_bytes(path: Path, expected_sha256: str, label: str) -> bytes:
    """Read one regular file through a stable descriptor and recheck its pin."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SmokeExecutionError(f"cannot open pinned {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SmokeExecutionError(f"pinned {label} is not a regular file")
        chunks = []
        while chunk := os.read(descriptor, _HASH_BLOCK_SIZE):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise SmokeExecutionError(f"pinned {label} changed while being read")
    data = b"".join(chunks)
    observed = sha256(data).hexdigest()
    if not hmac.compare_digest(observed, expected_sha256):
        raise SmokeExecutionError(
            f"pinned {label} SHA mismatch: expected {expected_sha256}, got {observed}"
        )
    return data


def _load_canonical_json(
    path: Path, label: str, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    data = (
        _stable_pinned_bytes(path, expected_sha256, label)
        if expected_sha256 is not None
        else path.read_bytes()
    )
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeExecutionError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict) or data != canonical_json_bytes(value):
        raise SmokeExecutionError(f"{label} is not a canonical JSON object")
    return value


def _load_preflight_pair(
    request_path: Path,
    report_path: Path,
    *,
    expected_request_sha256: str,
    expected_report_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], dict[str, str]]:
    """Re-run the torch-free smoke preflight and return its frozen pins."""

    from sana_wam.train.phase6_preflight import (
        load_canonical_preflight_report,
        load_canonical_preflight_request,
        validate_phase6_preflight,
    )

    request = load_canonical_preflight_request(
        request_path, expected_sha256=expected_request_sha256
    )
    report = load_canonical_preflight_report(
        report_path, expected_sha256=expected_report_sha256
    )
    if request.get("purpose") != "smoke" or report.get("purpose") != "smoke":
        raise SmokeExecutionError("preflight pair does not authorize the smoke purpose")
    if request.get("authorization") != SMOKE_AUTHORIZATION:
        raise SmokeExecutionError(
            "smoke preflight authorization differs from frozen contract"
        )
    if report.get("authorization") != request["authorization"]:
        raise SmokeExecutionError("smoke preflight report authorization differs")
    if report.get("request_sha256") != expected_request_sha256:
        raise SmokeExecutionError("smoke preflight report/request SHA binding differs")
    recomputed = validate_phase6_preflight(request)
    if report != recomputed:
        raise SmokeExecutionError(
            "stored smoke preflight report no longer reproduces exactly"
        )

    paths: dict[str, Path] = {}
    for role, pin in request["artifacts"].items():
        paths[role] = Path(pin["path"]).resolve(strict=True)
    for role, pin in request["runtime_files"].items():
        paths[role] = Path(pin["path"]).resolve(strict=True)
    paths["source_manifest"] = Path(request["source_manifest"]["path"]).resolve(
        strict=True
    )

    pins = report["pins"]
    bindings = {
        "action_stats_sha256": request["runtime_files"]["action_stats"]["sha256"],
        "arm_projection_manifest_sha256": request["artifacts"][
            "arm_projection_manifest"
        ]["sha256"],
        "dataset_contract_sha256": request["artifacts"]["dataset_contract"]["sha256"],
        "identity_sha256": pins["observed_embedded_identity_sha256"],
        "phase1_checkpoint_sha256": request["runtime_files"]["phase1_checkpoint"][
            "sha256"
        ],
        "plan_sha256": pins["observed_embedded_plan_sha256"],
        "preflight_report_sha256": expected_report_sha256,
        "preflight_request_sha256": expected_request_sha256,
        "reference_artifact_sha256": request["artifacts"]["action_reference"]["sha256"],
        "runtime_support_manifest_sha256": request["artifacts"][
            "runtime_support_manifest"
        ]["sha256"],
        "smoke_runtime_config_sha256": pins["observed_input_config_sha256"],
        "source_manifest_sha256": request["source_manifest"]["sha256"],
        "spot_artifact_sha256": request["artifacts"]["action_reference_spot_check"][
            "sha256"
        ],
        "student_checkpoint_sha256": request["runtime_files"]["student_checkpoint"][
            "sha256"
        ],
    }
    return request, report, paths, bindings


def _load_projection_entries(
    path: Path, authorization: dict[str, Any], *, expected_sha256: str
) -> dict[str, dict[str, Any]]:
    manifest = _load_canonical_json(
        path, "arm projection manifest", expected_sha256=expected_sha256
    )
    if (
        manifest.get("schema_version")
        != "sana-phase6-arm-config-projection-manifest-v1"
    ):
        raise SmokeExecutionError("arm projection manifest schema differs")
    if manifest.get("formal_training_started") is not False:
        raise SmokeExecutionError("projection manifest says formal training started")
    if manifest.get("closed_loop_started") is not False:
        raise SmokeExecutionError("projection manifest says closed loop started")
    arms = manifest.get("arms")
    if not isinstance(arms, list):
        raise SmokeExecutionError("projection manifest arms must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in arms:
        if not isinstance(entry, dict):
            raise SmokeExecutionError("projection entry must be an object")
        arm = entry.get("arm")
        projection = entry.get("projection")
        projection_sha = entry.get("projection_sha256")
        if not isinstance(arm, str) or arm in by_id or not isinstance(projection, dict):
            raise SmokeExecutionError(
                "projection manifest has malformed/duplicate arms"
            )
        if sha256(canonical_json_bytes(projection)).hexdigest() != projection_sha:
            raise SmokeExecutionError(f"{arm} embedded projection SHA differs")
        by_id[arm] = entry
    for case_id in authorization["arms"]:
        if case_id not in by_id:
            raise SmokeExecutionError(f"projection manifest lacks {case_id}")
        entry = by_id[case_id]
        expected_factors = {
            "T": case_id.startswith("T1_"),
            "E": "_E1" in case_id,
            "A": case_id.endswith("A1"),
        }
        if entry.get("factors") != expected_factors:
            raise SmokeExecutionError(f"{case_id} projection factors differ")
    return by_id


def _derive_fixed_row(
    action_reference_path: Path,
    global_step: int,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    reference = _load_canonical_json(
        action_reference_path,
        "action reference",
        expected_sha256=expected_sha256,
    )
    rows = reference.get("rows")
    if not isinstance(rows, list) or len(rows) != 504:
        raise SmokeExecutionError("action reference must contain the exact 504 rows")
    row = rows[global_step - 1]
    if not isinstance(row, dict) or row.get("global_step") != global_step:
        raise SmokeExecutionError("action-reference paired row is absent or reordered")
    fixed = {
        "action_sigma": row.get("action_sigma"),
        "common_input_trace_sha256": row.get("common_input_trace_sha256"),
        "dataset_contract_row_sha256": row.get("dataset_contract_row_sha256"),
        "dataset_index": row.get("dataset_index"),
        "global_step": row.get("global_step"),
        "plan_row_sha256": row.get("plan_row_sha256"),
        "task_name": row.get("task_name"),
    }
    # The final artifact validator repeats type/hash checks on this projection.
    return fixed


def _exclusive_write(path: Path, data: bytes, label: str) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise SmokeExecutionError(
            f"{label} needs an absolute path with existing parent"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _tree_inventory(roots: list[str]) -> str:
    """Hash names and metadata under primary roots without reading checkpoints."""

    records: list[dict[str, Any]] = []
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            records.append({"exists": False, "root": str(root)})
            continue
        for current, directories, files in os.walk(root):
            directories.sort()
            files.sort()
            current_path = Path(current)
            for name in directories + files:
                path = current_path / name
                info = path.lstat()
                records.append(
                    {
                        "mode": stat.S_IFMT(info.st_mode),
                        "mtime_ns": info.st_mtime_ns,
                        "path": str(path),
                        "size": info.st_size,
                    }
                )
    return sha256(canonical_json_bytes(records)).hexdigest()


def _key_set_sha256(keys: list[str] | tuple[str, ...]) -> str:
    normalized = sorted(keys)
    if len(normalized) != len(set(normalized)):
        raise SmokeExecutionError("duplicate state/checkpoint key")
    return sha256(canonical_json_bytes(normalized)).hexdigest()


def _tensor_bytes(tensor: Any, torch: Any) -> bytes:
    if not isinstance(tensor, torch.Tensor):
        raise SmokeExecutionError("runtime state item is not a torch.Tensor")
    value = tensor.detach().contiguous()
    if value.device.type != "cpu":
        value = value.view(torch.uint8).cpu()
    else:
        value = value.view(torch.uint8)
    return value.numpy().tobytes(order="C")


def _state_digest(items: Any, torch: Any, label: str) -> tuple[str, int]:
    if isinstance(items, dict):
        entries = list(items.items())
    else:
        entries = list(items)
    names = [name for name, _ in entries]
    if not entries or any(not isinstance(name, str) or not name for name in names):
        raise SmokeExecutionError(f"{label} state must contain named tensors")
    if len(names) != len(set(names)):
        raise SmokeExecutionError(f"{label} state has duplicate tensor names")
    digest = sha256()
    for name, tensor in sorted(entries):
        metadata = {
            "dtype": str(tensor.dtype),
            "name": name,
            "shape": list(tensor.shape),
        }
        header = canonical_json_bytes(metadata)
        payload = _tensor_bytes(tensor, torch)
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), len(entries)


def _tensor_tree_digest(value: Any, torch: Any) -> tuple[str, int]:
    digest = sha256()
    count = 0

    def visit(item: Any, path: str) -> None:
        nonlocal count
        if isinstance(item, torch.Tensor):
            header = canonical_json_bytes(
                {"dtype": str(item.dtype), "path": path, "shape": list(item.shape)}
            )
            payload = _tensor_bytes(item, torch)
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            count += 1
            return
        if isinstance(item, dict):
            for key in sorted(item):
                if not isinstance(key, str):
                    raise SmokeExecutionError(
                        "identity output mapping keys must be strings"
                    )
                visit(item[key], f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        raise SmokeExecutionError(
            "identity outputs may contain only tensors/containers"
        )

    visit(value, "output")
    if count == 0:
        raise SmokeExecutionError("identity comparison has no output tensors")
    return digest.hexdigest(), count


def _scalar_tensor(result: dict[str, Any], key: str, torch: Any) -> Any:
    value = result.get(key)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise SmokeExecutionError(f"forward result {key} must be one tensor scalar")
    if not bool(torch.isfinite(value).all().item()):
        raise SmokeExecutionError(f"forward result {key} is non-finite")
    return value.reshape(())


def _float32_bits(tensor: Any, torch: Any) -> str:
    if tensor.dtype is not torch.float32:
        raise SmokeExecutionError("registered smoke loss must be a float32 scalar")
    return struct.pack(">f", float(tensor.detach().cpu().item())).hex()


def _named_parameters(value: Any, torch: Any, label: str) -> list[tuple[str, Any]]:
    entries = list(value)
    names: set[str] = set()
    for name, parameter in entries:
        if not isinstance(name, str) or not name or name in names:
            raise SmokeExecutionError(
                f"{label} has malformed/duplicate parameter names"
            )
        if not isinstance(parameter, torch.nn.Parameter):
            raise SmokeExecutionError(f"{label}.{name} is not nn.Parameter")
        names.add(name)
    return entries


@contextmanager
def _block_optimizer_activity(torch: Any):
    """Hard-block optimizer construction and steps for the whole live runtime."""

    patched: list[tuple[type, str, Any]] = []
    activity = {"creation_count": 0, "step_count": 0}

    def blocked_step(self: Any, *args: Any, **kwargs: Any) -> None:
        del self, args, kwargs
        activity["step_count"] += 1
        raise SmokeExecutionError("optimizer.step() is forbidden in the smoke gate")

    def blocked_init(self: Any, *args: Any, **kwargs: Any) -> None:
        del self, args, kwargs
        activity["creation_count"] += 1
        raise SmokeExecutionError(
            "optimizer construction is forbidden in the smoke gate"
        )

    stack = [torch.optim.Optimizer]
    seen: set[type] = set()
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if "__init__" in cls.__dict__:
            original_init = cls.__dict__["__init__"]
            setattr(cls, "__init__", blocked_init)
            patched.append((cls, "__init__", original_init))
        if "step" in cls.__dict__:
            original_step = cls.__dict__["step"]
            setattr(cls, "step", blocked_step)
            patched.append((cls, "step", original_step))
    try:
        yield activity
    finally:
        for cls, name, original in reversed(patched):
            setattr(cls, name, original)


def _require_runtime_surface(runtime: Any) -> None:
    for name in (
        "checkpoint_keys",
        "close",
        "open_case",
        "reload_case_from_full_checkpoint",
    ):
        if not callable(getattr(runtime, name, None)):
            raise SmokeExecutionError(f"smoke runtime lacks callable {name}")


def _require_handle_surface(handle: Any) -> None:
    if getattr(handle, "model", None) is None:
        raise SmokeExecutionError("case handle lacks a live model")
    for name in (
        "action_backbone_named_parameters",
        "adapter_named_parameters",
        "expected_adapter_state_keys",
        "frozen_action_proprio_state_items",
        "run_forward",
        "save_full_checkpoint",
        "video_dit_parameters",
        "video_trainable_named_parameters",
    ):
        if not callable(getattr(handle, name, None)):
            raise SmokeExecutionError(f"case handle lacks callable {name}")


def _check_model_identity(
    handle: Any, bindings: dict[str, str], torch: Any
) -> dict[str, Any]:
    metadata = getattr(handle, "model_metadata", None)
    if not isinstance(metadata, dict):
        raise SmokeExecutionError("runtime handle lacks model_metadata")
    expected = {
        "model_dtype": "torch.bfloat16",
        "model_factory": "SanaMSVideoCamCtrl_1600M_P1_D20",
        "model_label": "SANA-Video 2B 480p",
        "student_checkpoint_sha256": bindings["student_checkpoint_sha256"],
    }
    for key in ("model_label", "model_factory", "model_dtype"):
        if metadata.get(key) != expected[key]:
            raise SmokeExecutionError(f"real-2B runtime metadata {key} differs")
    if (
        metadata.get("student_checkpoint_sha256")
        != expected["student_checkpoint_sha256"]
    ):
        raise SmokeExecutionError("runtime loaded a different student checkpoint")
    if metadata.get("full_student_checkpoint_loaded") is not True:
        raise SmokeExecutionError(
            "runtime did not strictly load the full student checkpoint"
        )
    if (
        metadata.get("mini_model") is not False
        or metadata.get("mock_model") is not False
    ):
        raise SmokeExecutionError("mini/mock model is forbidden in real-2B smoke")
    model = handle.model
    if not isinstance(model, torch.nn.Module):
        raise SmokeExecutionError("runtime model is not torch.nn.Module")
    total_count = sum(parameter.numel() for parameter in model.parameters())
    video_parameters = list(handle.video_dit_parameters())
    if not video_parameters or any(
        not isinstance(parameter, torch.nn.Parameter) for parameter in video_parameters
    ):
        raise SmokeExecutionError("runtime video backbone parameter set is invalid")
    video_count = sum(parameter.numel() for parameter in video_parameters)
    if video_count < MINIMUM_REAL_2B_VIDEO_PARAMETER_COUNT:
        raise SmokeExecutionError("live video backbone is not the real-2B topology")
    if total_count < video_count:
        raise SmokeExecutionError("live total parameter count is inconsistent")
    return {
        "full_student_checkpoint_loaded": True,
        "mini_model": False,
        "mock_model": False,
        "model_dtype": expected["model_dtype"],
        "model_factory": expected["model_factory"],
        "model_label": expected["model_label"],
        "student_checkpoint_sha256": expected["student_checkpoint_sha256"],
        "total_parameter_count": total_count,
        "video_dit_parameter_count": video_count,
    }


def _assert_peak(torch: Any, device: Any, case_id: str) -> int:
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_reserved(device))
    if peak >= PEAK_RESERVED_LIMIT_BYTES:
        raise SmokeExecutionError(
            f"{case_id} peak reserved {peak} reached 130-GiB stop limit"
        )
    return peak


def _cuda_driver_version() -> str:
    """Read the loaded NVIDIA driver API version without invoking a shell."""

    try:
        driver = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        if (
            driver.cuInit(0) != 0
            or driver.cuDriverGetVersion(ctypes.byref(version)) != 0
        ):
            raise RuntimeError("CUDA driver query returned an error")
    except (OSError, AttributeError, RuntimeError) as exc:
        raise SmokeExecutionError(f"cannot query CUDA driver version: {exc}") from exc
    major = version.value // 1000
    minor = (version.value % 1000) // 10
    return f"{major}.{minor}"


def _cuda_device_uuid(device_index: int) -> str:
    """Read the physical CUDA UUID from libcuda for an unambiguous descriptor."""

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
        raise SmokeExecutionError(f"cannot query CUDA device UUID: {exc}") from exc
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


def _run_nr_probe(
    *,
    nr_loss: Any,
    adapter_parameters: list[tuple[str, Any]],
    video_parameters: list[tuple[str, Any]],
    action_parameters: list[tuple[str, Any]],
    retain_graph: bool,
    torch: Any,
) -> dict[str, bool]:
    if not adapter_parameters or any(
        not parameter.requires_grad for _, parameter in adapter_parameters
    ):
        raise SmokeExecutionError("A1 adapter parameters must all require gradients")
    if not video_parameters or any(
        not parameter.requires_grad for _, parameter in video_parameters
    ):
        raise SmokeExecutionError(
            "video trainable parameter contract is empty/inconsistent"
        )
    if any(parameter.requires_grad for _, parameter in action_parameters):
        raise SmokeExecutionError("action/proprio backbone parameters are not frozen")
    requested = [parameter for _, parameter in adapter_parameters + video_parameters]
    gradients = torch.autograd.grad(
        nr_loss,
        requested,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=False,
    )
    adapter_grads = gradients[: len(adapter_parameters)]
    video_grads = gradients[len(adapter_parameters) :]
    if any(gradient is None for gradient in adapter_grads):
        raise SmokeExecutionError("NR loss is disconnected from an adapter parameter")
    if any(
        not bool(torch.isfinite(gradient).all().item()) for gradient in adapter_grads
    ):
        raise SmokeExecutionError("NR adapter gradient is non-finite")
    if any(gradient is not None for gradient in video_grads):
        raise SmokeExecutionError("NR loss leaked into video parameters")
    if any(parameter.grad is not None for _, parameter in action_parameters):
        raise SmokeExecutionError("NR loss populated an action/proprio gradient")
    return {
        "adapter_gradient_connected": True,
        "adapter_gradient_finite": True,
        "adapter_gradient_zero_allowed": True,
        "enabled": True,
        "nr_action_backbone_gradient_absent": True,
        "nr_autograd_probe_executed": True,
        "nr_loss_finite": True,
        "nr_video_gradient_absent": True,
    }


def _run_full_backward(
    loss: Any,
    *,
    model: Any,
    video_parameters: list[tuple[str, Any]],
    adapter_parameters: list[tuple[str, Any]],
    torch: Any,
) -> None:
    all_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    expected = video_parameters + adapter_parameters
    if {name for name, _ in all_trainable} != {name for name, _ in expected}:
        raise SmokeExecutionError(
            "runtime trainable set differs from video+adapter contract"
        )
    loss.backward()
    for name, parameter in expected:
        if parameter.grad is None:
            raise SmokeExecutionError(f"required E1A1 gradient is disconnected: {name}")
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise SmokeExecutionError(f"required E1A1 gradient is non-finite: {name}")


def _build_case_report(
    *,
    case_spec: dict[str, Any],
    fixed_row: dict[str, Any],
    model_count: int,
    projection_sha256: str,
    expansion_bits: str,
    additional_forward_count: int,
    timestep_observation: dict[str, Any],
    action_gate: dict[str, bool],
    frozen_before: str,
    frozen_after: str,
    frozen_count: int,
    expected_adapter_keys: list[str],
    observed_adapter_keys: list[str],
    backward_executed: bool,
    peak_reserved: int,
) -> dict[str, Any]:
    factors = case_spec["factors"]
    expansion = factors["expansion"]
    adapter = factors["action_adapter"]
    expected_key_sha = _key_set_sha256(expected_adapter_keys)
    observed_key_sha = _key_set_sha256(observed_adapter_keys)
    return {
        "action_adapter_gate": action_gate,
        "adapter_key_gate": {
            "complete": expected_adapter_keys == observed_adapter_keys,
            "enabled": adapter,
            "expected_key_count": len(expected_adapter_keys),
            "expected_key_set_sha256": expected_key_sha,
            "observed_key_count": len(observed_adapter_keys),
            "observed_key_set_sha256": observed_key_sha,
            "partial": False,
        },
        "backward_gate": {
            "all_required_gradients_finite": backward_executed,
            "executed": backward_executed,
            "full_loss_finite": True,
            "optimizer_step_after_backward": False,
        },
        "batch_size": 1,
        "case_id": case_spec["case_id"],
        "closed_loop_started": False,
        "config_projection_sha256": projection_sha256,
        "expansion_gate": {
            "additional_video_forward_count": additional_forward_count,
            "disabled_loss_exact_zero": not expansion,
            "enabled": expansion,
            "loss_finite": True,
            "loss_float32_bits": expansion_bits,
            "on_path_loss_present": True,
        },
        "factors": factors,
        "formal_training_started": False,
        "frozen_action_proprio_gate": {
            "after_sha256": frozen_after,
            "before_sha256": frozen_before,
            "bitwise_unchanged": frozen_before == frozen_after,
            "state_tensor_count": frozen_count,
        },
        "gpu_forward_executed": True,
        "memory_gate": {
            "below_limit": peak_reserved < PEAK_RESERVED_LIMIT_BYTES,
            "limit_bytes": PEAK_RESERVED_LIMIT_BYTES,
            "peak_reserved_bytes": peak_reserved,
            "peak_stats_reset_before_case": True,
        },
        "model_parameter_count": model_count,
        "optimizer_creation_count": 0,
        "optimizer_step_count": 0,
        "paired_fixed_row": fixed_row,
        "primary_checkpoint_written": False,
        "timestep_gate": timestep_observation,
        "world_size": 1,
    }


def _run_smoke(
    preflight_request: dict[str, Any],
    preflight_report: dict[str, Any],
    *,
    bindings: dict[str, str],
    fixed_row: dict[str, Any],
    output: dict[str, Any],
    artifact_paths: dict[str, Path],
    projections: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # This is the first import of torch in the executable path.
    import torch

    if not torch.cuda.is_available():
        raise SmokeExecutionError("CUDA is unavailable; no smoke artifact was produced")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SmokeExecutionError("real-2B smoke requires WORLD_SIZE=1")
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    properties = torch.cuda.get_device_properties(device)
    if "H200" not in properties.name.upper():
        raise SmokeExecutionError(f"smoke requires H200, got {properties.name}")
    if list(torch.cuda.get_device_capability(device)) != [9, 0]:
        raise SmokeExecutionError("smoke requires CUDA capability [9,0]")
    if int(properties.total_memory) <= PEAK_RESERVED_LIMIT_BYTES:
        raise SmokeExecutionError("H200 has no headroom above the 130-GiB stop limit")
    device_uuid = _cuda_device_uuid(device_index)
    driver_version = _cuda_driver_version()

    runtime_module = importlib.import_module("sana_wam.train.phase6_smoke_runtime")
    if (
        getattr(runtime_module, "RUNTIME_SUPPORT_API_VERSION", None)
        != RUNTIME_SUPPORT_API_VERSION
    ):
        raise SmokeExecutionError("phase6_smoke_runtime API version differs")
    factory = getattr(runtime_module, "create_phase6_smoke_runtime", None)
    if not callable(factory):
        raise SmokeExecutionError("phase6_smoke_runtime lacks its factory")

    ephemeral_parent = Path(output["ephemeral_root"])
    if not ephemeral_parent.is_dir() or ephemeral_parent.is_symlink():
        raise SmokeExecutionError(
            "ephemeral_root must be an existing non-symlink directory"
        )
    ephemeral_dir = Path(
        tempfile.mkdtemp(prefix="phase6-real-2b-smoke-", dir=ephemeral_parent)
    ).resolve(strict=True)
    temp_checkpoint = ephemeral_dir / "T1_E1A1_full.safetensors"
    if temp_checkpoint.exists():
        raise SmokeExecutionError("exclusive temporary checkpoint path already exists")

    primary_before = _tree_inventory(output["forbidden_primary_roots"])
    runtime = None
    case_reports: list[dict[str, Any]] = []
    model_identity: dict[str, Any] | None = None
    identity_outputs: dict[str, tuple[str, int]] = {}
    a0_live_counts: list[int] = []
    a1_live_key_shas: list[str] = []
    a1_expected_keys: list[str] | None = None
    checkpoint_gate: dict[str, Any] | None = None
    checkpoint_adapter_key_sha: str | None = None
    optimizer_activity = {"creation_count": 0, "step_count": 0}
    try:
        with _block_optimizer_activity(torch) as optimizer_activity:
            runtime = factory(
                preflight_request=preflight_request,
                preflight_report=preflight_report,
                artifact_paths={
                    key: str(value) for key, value in artifact_paths.items()
                },
                device=device,
                ephemeral_directory=str(ephemeral_dir),
            )
            _require_runtime_surface(runtime)
            student_checkpoint_keys = sorted(
                runtime.checkpoint_keys(str(artifact_paths["student_checkpoint"]))
            )
            student_adapter_keys = [
                key
                for key in student_checkpoint_keys
                if key.startswith(_ADAPTER_PREFIX)
            ]
            if student_adapter_keys:
                raise SmokeExecutionError("A0 student checkpoint contains adapter keys")

            case_specs = []
            for case_id in SMOKE_AUTHORIZATION["arms"]:
                projection_entry = projections[case_id]
                case_specs.append(
                    {
                        "case_id": case_id,
                        "config_projection_sha256": projection_entry[
                            "projection_sha256"
                        ],
                        "factors": {
                            "action_adapter": case_id.endswith("A1"),
                            "continuous_time": case_id.startswith("T1_"),
                            "expansion": "_E1" in case_id,
                        },
                    }
                )
            for case_spec in case_specs:
                case_id = case_spec["case_id"]
                projection = projections[case_id]["projection"]
                torch.cuda.reset_peak_memory_stats(device)
                handle = runtime.open_case(
                    case_id=case_id,
                    factors=dict(case_spec["factors"]),
                    config_projection=projection,
                    paired_fixed_row=dict(fixed_row),
                )
                _require_handle_surface(handle)
                current_identity = _check_model_identity(handle, bindings, torch)
                if model_identity is None:
                    model_identity = current_identity
                else:
                    invariant_keys = (
                        "full_student_checkpoint_loaded",
                        "mini_model",
                        "mock_model",
                        "model_dtype",
                        "model_factory",
                        "model_label",
                        "student_checkpoint_sha256",
                        "video_dit_parameter_count",
                    )
                    if any(
                        current_identity[key] != model_identity[key]
                        for key in invariant_keys
                    ):
                        raise SmokeExecutionError(
                            f"{case_id} real-2B model identity drifted"
                        )
                    if (
                        not case_spec["factors"]["action_adapter"]
                        and current_identity["total_parameter_count"]
                        != model_identity["total_parameter_count"]
                    ):
                        raise SmokeExecutionError(
                            f"{case_id} A0 parameter count drifted"
                        )

                expected_keys = sorted(handle.expected_adapter_state_keys())
                live_keys = sorted(
                    key
                    for key in handle.model.state_dict()
                    if key.startswith(_ADAPTER_PREFIX)
                )
                adapter_enabled = case_spec["factors"]["action_adapter"]
                if adapter_enabled:
                    if not expected_keys or live_keys != expected_keys:
                        raise SmokeExecutionError(
                            f"{case_id} A1 adapter key set is incomplete"
                        )
                    if a1_expected_keys is None:
                        a1_expected_keys = expected_keys
                    elif expected_keys != a1_expected_keys:
                        raise SmokeExecutionError("A1 expected adapter key set drifted")
                    a1_live_key_shas.append(_key_set_sha256(live_keys))
                else:
                    if expected_keys or live_keys:
                        raise SmokeExecutionError(f"{case_id} A0 contains adapter keys")
                    a0_live_counts.append(len(live_keys))

                frozen_before, frozen_count = _state_digest(
                    handle.frozen_action_proprio_state_items(),
                    torch,
                    f"{case_id} frozen action/proprio",
                )
                adapter_parameters = _named_parameters(
                    handle.adapter_named_parameters(),
                    torch,
                    f"{case_id} adapter parameters",
                )
                video_parameters = _named_parameters(
                    handle.video_trainable_named_parameters(),
                    torch,
                    f"{case_id} video parameters",
                )
                action_parameters = _named_parameters(
                    handle.action_backbone_named_parameters(),
                    torch,
                    f"{case_id} action parameters",
                )
                if bool(adapter_parameters) != adapter_enabled:
                    raise SmokeExecutionError(
                        f"{case_id} adapter parameter presence differs"
                    )

                _assert_peak(torch, device, case_id)
                result = handle.run_forward()
                if not isinstance(result, dict):
                    raise SmokeExecutionError(
                        f"{case_id} forward result must be an object"
                    )
                if result.get("paired_fixed_row") != fixed_row:
                    raise SmokeExecutionError(
                        f"{case_id} runtime changed the paired row"
                    )
                if (
                    result.get("common_input_trace_sha256")
                    != fixed_row["common_input_trace_sha256"]
                ):
                    raise SmokeExecutionError(
                        f"{case_id} common prepared/noise trace drifted"
                    )
                loss = _scalar_tensor(result, "loss", torch)
                _scalar_tensor(result, "loss_video_on_path", torch)
                expansion_loss = _scalar_tensor(
                    result, "loss_video_local_expansion", torch
                )
                expansion_bits = _float32_bits(expansion_loss, torch)
                expansion_enabled = case_spec["factors"]["expansion"]
                expected_extra = 1 if expansion_enabled else 0
                if result.get("additional_video_forward_count") != expected_extra:
                    raise SmokeExecutionError(
                        f"{case_id} expansion forward count differs"
                    )
                if not expansion_enabled and expansion_bits != "00000000":
                    raise SmokeExecutionError(
                        f"{case_id} disabled expansion loss is not +0.0"
                    )
                continuous_time = case_spec["factors"]["continuous_time"]
                if (
                    result.get("video_continuous_conditioning_enabled")
                    is not continuous_time
                ):
                    raise SmokeExecutionError(
                        f"{case_id} live video conditioning flag differs from its factor"
                    )
                boundary_values = result.get("timestep_embedder_boundary_values")
                if not isinstance(boundary_values, tuple) or not boundary_values:
                    raise SmokeExecutionError(
                        f"{case_id} did not expose the real t_embedder boundary"
                    )
                if any(
                    not isinstance(value, torch.Tensor)
                    or value.dtype is not torch.float32
                    for value in boundary_values
                ):
                    raise SmokeExecutionError(
                        f"{case_id} t_embedder boundary did not receive FP32 tensors"
                    )
                boundary_sha, _boundary_tensor_count = _tensor_tree_digest(
                    boundary_values, torch
                )
                boundary_value_count = sum(value.numel() for value in boundary_values)
                fractional_count = sum(
                    int(torch.count_nonzero(value != torch.trunc(value)).item())
                    for value in boundary_values
                )
                if continuous_time:
                    if fractional_count == 0:
                        raise SmokeExecutionError(
                            f"{case_id} T1 lost all fractional values before t_embedder"
                        )
                elif fractional_count != 0:
                    raise SmokeExecutionError(
                        f"{case_id} T0 did not reproduce integer-valued embedder inputs"
                    )
                timestep_observation = {
                    "continuous_coordinate_consumed": continuous_time
                    and fractional_count > 0,
                    "embedder_boundary_dtype": "torch.float32",
                    "embedder_boundary_fractional_value_count": fractional_count,
                    "embedder_boundary_value_count": boundary_value_count,
                    "embedder_boundary_value_sha256": boundary_sha,
                    "mode": "continuous_t1" if continuous_time else "integer_t0",
                    "observed_at_t_embedder_boundary": True,
                    "old_integer_contract_reproduced": (
                        not continuous_time and fractional_count == 0
                    ),
                    "t1_long_conversion_detected": (
                        continuous_time and fractional_count == 0
                    ),
                }

                identity_digest, identity_count = _tensor_tree_digest(
                    result.get("identity_outputs"), torch
                )
                if case_id in {"T1_E0A0", "T1_E0A1"}:
                    identity_outputs[case_id] = (identity_digest, identity_count)

                if adapter_enabled:
                    nr_loss = _scalar_tensor(
                        result, "phase6_action_non_regression_surrogate", torch
                    )
                    action_gate = _run_nr_probe(
                        nr_loss=nr_loss,
                        adapter_parameters=adapter_parameters,
                        video_parameters=video_parameters,
                        action_parameters=action_parameters,
                        retain_graph=case_id == "T1_E1A1",
                        torch=torch,
                    )
                else:
                    action_gate = {
                        "adapter_gradient_connected": False,
                        "adapter_gradient_finite": False,
                        "adapter_gradient_zero_allowed": False,
                        "enabled": False,
                        "nr_action_backbone_gradient_absent": False,
                        "nr_autograd_probe_executed": False,
                        "nr_loss_finite": False,
                        "nr_video_gradient_absent": False,
                    }

                backward_executed = case_id == "T1_E1A1"
                if backward_executed:
                    _run_full_backward(
                        loss,
                        model=handle.model,
                        video_parameters=video_parameters,
                        adapter_parameters=adapter_parameters,
                        torch=torch,
                    )
                _assert_peak(torch, device, case_id)
                frozen_after, frozen_after_count = _state_digest(
                    handle.frozen_action_proprio_state_items(),
                    torch,
                    f"{case_id} frozen action/proprio after",
                )
                if frozen_after_count != frozen_count or frozen_after != frozen_before:
                    raise SmokeExecutionError(
                        f"{case_id} action/proprio tensors changed"
                    )

                if case_id == "T1_E1A1":
                    pre_save_sha, state_count = _state_digest(
                        handle.model.state_dict(), torch, "E1A1 pre-save model"
                    )
                    handle.save_full_checkpoint(str(temp_checkpoint))
                    checkpoint_info = _regular_nonsymlink(
                        temp_checkpoint, "temporary E1A1 checkpoint"
                    )
                    if checkpoint_info.st_size < MINIMUM_FULL_CHECKPOINT_SIZE_BYTES:
                        raise SmokeExecutionError(
                            "temporary E1A1 checkpoint is not full-size"
                        )
                    checkpoint_sha = _sha256_file(temp_checkpoint)
                    checkpoint_keys = sorted(
                        runtime.checkpoint_keys(str(temp_checkpoint))
                    )
                    if checkpoint_keys != sorted(handle.model.state_dict()):
                        raise SmokeExecutionError(
                            "temporary full checkpoint key set differs"
                        )
                    checkpoint_adapter_keys = [
                        key
                        for key in checkpoint_keys
                        if key.startswith(_ADAPTER_PREFIX)
                    ]
                    if checkpoint_adapter_keys != expected_keys:
                        raise SmokeExecutionError(
                            "temporary checkpoint adapter keys incomplete"
                        )
                    checkpoint_adapter_key_sha = _key_set_sha256(
                        checkpoint_adapter_keys
                    )
                    handle.close()
                    del handle, result, loss
                    torch.cuda.empty_cache()
                    reloaded = runtime.reload_case_from_full_checkpoint(
                        case_id=case_id,
                        factors=dict(case_spec["factors"]),
                        config_projection=projection,
                        paired_fixed_row=dict(fixed_row),
                        checkpoint_path=str(temp_checkpoint),
                        strict=True,
                    )
                    _require_handle_surface(reloaded)
                    reloaded_sha, reloaded_count = _state_digest(
                        reloaded.model.state_dict(), torch, "E1A1 reloaded model"
                    )
                    if reloaded_count != state_count or reloaded_sha != pre_save_sha:
                        raise SmokeExecutionError(
                            "E1A1 strict reload is not bitwise equal"
                        )
                    reloaded.close()
                    del reloaded
                    torch.cuda.empty_cache()
                    peak = _assert_peak(torch, device, case_id)
                    temp_checkpoint.unlink()
                    if temp_checkpoint.exists():
                        raise SmokeExecutionError(
                            "temporary checkpoint deletion failed"
                        )
                    checkpoint_gate = {
                        "adapter_keys_complete": True,
                        "backward_completed_before_save": True,
                        "case_id": "T1_E1A1",
                        "checkpoint_sha256": checkpoint_sha,
                        "checkpoint_size_bytes": checkpoint_info.st_size,
                        "full_checkpoint": True,
                        "path_within_ephemeral_root": True,
                        "pre_save_state_sha256": pre_save_sha,
                        "reloaded_state_sha256": reloaded_sha,
                        "state_bitwise_equal": True,
                        "state_tensor_count": state_count,
                        "strict_reload": True,
                        "temporary_checkpoint_deleted": True,
                    }
                else:
                    peak = _assert_peak(torch, device, case_id)
                    handle.close()
                    del handle, result, loss
                    torch.cuda.empty_cache()

                case_reports.append(
                    _build_case_report(
                        case_spec=case_spec,
                        fixed_row=fixed_row,
                        model_count=current_identity["total_parameter_count"],
                        projection_sha256=case_spec["config_projection_sha256"],
                        expansion_bits=expansion_bits,
                        additional_forward_count=expected_extra,
                        timestep_observation=timestep_observation,
                        action_gate=action_gate,
                        frozen_before=frozen_before,
                        frozen_after=frozen_after,
                        frozen_count=frozen_count,
                        expected_adapter_keys=expected_keys,
                        observed_adapter_keys=live_keys,
                        backward_executed=backward_executed,
                        peak_reserved=peak,
                    )
                )
            runtime.close()
            runtime = None
            if optimizer_activity != {"creation_count": 0, "step_count": 0}:
                raise SmokeExecutionError("optimizer construction/step was attempted")
    finally:
        if runtime is not None:
            runtime.close()
        if temp_checkpoint.exists():
            temp_checkpoint.unlink()
        resolved_parent = ephemeral_parent.resolve(strict=True)
        resolved_temp = ephemeral_dir.resolve(strict=True)
        if resolved_temp.parent != resolved_parent or not resolved_temp.name.startswith(
            "phase6-real-2b-smoke-"
        ):
            raise SmokeExecutionError("refusing unsafe ephemeral cleanup")
        shutil.rmtree(resolved_temp)

    if model_identity is None or checkpoint_gate is None or a1_expected_keys is None:
        raise SmokeExecutionError("smoke did not complete all mandatory cases")
    if set(identity_outputs) != {"T1_E0A0", "T1_E0A1"}:
        raise SmokeExecutionError("identity comparison pair was not executed")
    baseline_sha, baseline_count = identity_outputs["T1_E0A0"]
    enabled_sha, enabled_count = identity_outputs["T1_E0A1"]
    if baseline_count != enabled_count or baseline_sha != enabled_sha:
        raise SmokeExecutionError("identity-enabled adapter output differs bitwise")
    expected_a1_sha = _key_set_sha256(a1_expected_keys)
    if any(value != expected_a1_sha for value in a1_live_key_shas):
        raise SmokeExecutionError("an A1 live adapter key set differs")
    if checkpoint_adapter_key_sha != expected_a1_sha:
        raise SmokeExecutionError("temporary checkpoint adapter key set differs")
    primary_after = _tree_inventory(output["forbidden_primary_roots"])
    if primary_before != primary_after:
        raise SmokeExecutionError(
            "a forbidden primary output root changed during smoke"
        )

    expected_ids = list(SMOKE_AUTHORIZATION["arms"])
    now = (
        datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    artifact = {
        "adapter_key_contract": {
            "a0_live_key_count": sum(a0_live_counts),
            "a0_student_checkpoint_key_count": 0,
            "a1_expected_key_count": len(a1_expected_keys),
            "a1_expected_key_set_sha256": expected_a1_sha,
            "a1_live_key_set_sha256": expected_a1_sha,
            "a1_temporary_checkpoint_key_set_sha256": checkpoint_adapter_key_sha,
            "adapter_prefix": _ADAPTER_PREFIX,
            "all_a0_absent": True,
            "all_a1_complete": True,
        },
        "artifact_role": SMOKE_ARTIFACT_ROLE,
        "authorization": dict(SMOKE_AUTHORIZATION),
        "bindings": dict(bindings),
        "cases": case_reports,
        "closed_loop_started": False,
        "completed_at_utc": now,
        "coverage": {
            "every_case_gpu_forward": True,
            "every_case_same_paired_row": True,
            "mode": "registered_five_arms",
            "observed_cases": expected_ids,
            "required_cases": expected_ids,
        },
        "e1a1_checkpoint_reload": checkpoint_gate,
        "formal_training_started": False,
        "gates": {
            "action_proprio_frozen": True,
            "adapter_key_contract": True,
            "adapter_only_nr_connected_finite": True,
            "case_coverage": True,
            "closed_loop_absent": True,
            "e1a1_backward": True,
            "expansion_contract": True,
            "formal_training_absent": True,
            "identity_enabled_bitwise": True,
            "memory_strictly_below_limit": True,
            "optimizer_creation_absent": True,
            "optimizer_step_absent": True,
            "paired_fixed_row": True,
            "primary_checkpoint_absent": True,
            "real_2b_model": True,
            "temporary_checkpoint_deleted": True,
            "temporary_full_checkpoint_reload": True,
            "timestep_contract": True,
        },
        "gpu_smoke_executed": True,
        "host": {
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_device_name": properties.name,
            "cuda_device_uuid": device_uuid,
            "cuda_driver_version": driver_version,
            "cuda_runtime_version": str(torch.version.cuda),
            "cuda_total_memory_bytes": int(properties.total_memory),
            "device_index": device_index,
            "hostname": socket.gethostname(),
            "multi_processor_count": int(properties.multi_processor_count),
            "torch_version": str(torch.__version__),
            "world_size": 1,
        },
        "identity_adapter_bitwise": {
            "adapter_restored_to_exact_identity": True,
            "baseline_case_id": "T1_E0A0",
            "baseline_output_sha256": baseline_sha,
            "bitwise_equal": True,
            "compared_tensor_count": baseline_count,
            "enabled_case_id": "T1_E0A1",
            "enabled_output_sha256": enabled_sha,
            "same_prepared_input": True,
        },
        "optimizer_gate": {
            "blockade_installed_before_runtime_factory": True,
            "creation_allowed": False,
            "creation_count": optimizer_activity["creation_count"],
            "step_allowed": False,
            "step_count": optimizer_activity["step_count"],
        },
        "paired_fixed_row": dict(fixed_row),
        "real_2b_model": model_identity,
        "schema_version": SMOKE_ARTIFACT_SCHEMA_VERSION,
        "status": "pass",
    }
    artifact_bytes = canonical_json_bytes(artifact)
    artifact_sha = sha256(artifact_bytes).hexdigest()
    validate_phase6_smoke_artifact_bytes(
        artifact_bytes,
        expected_artifact_sha256=artifact_sha,
        expected_source_manifest_sha256=bindings["source_manifest_sha256"],
        expected_runtime_support_manifest_sha256=bindings[
            "runtime_support_manifest_sha256"
        ],
        expected_student_checkpoint_sha256=bindings["student_checkpoint_sha256"],
        expected_phase1_checkpoint_sha256=bindings["phase1_checkpoint_sha256"],
        expected_action_stats_sha256=bindings["action_stats_sha256"],
        expected_plan_sha256=bindings["plan_sha256"],
        expected_identity_sha256=bindings["identity_sha256"],
        expected_dataset_contract_sha256=bindings["dataset_contract_sha256"],
        expected_reference_artifact_sha256=bindings["reference_artifact_sha256"],
        expected_spot_artifact_sha256=bindings["spot_artifact_sha256"],
        expected_arm_projection_manifest_sha256=bindings[
            "arm_projection_manifest_sha256"
        ],
        expected_preflight_request_sha256=bindings["preflight_request_sha256"],
        expected_preflight_report_sha256=bindings["preflight_report_sha256"],
        expected_smoke_runtime_config_sha256=bindings["smoke_runtime_config_sha256"],
    )
    receipt = {
        "artifact_role": SMOKE_ARTIFACT_ROLE,
        "bindings": dict(bindings),
        "closed_loop_started": False,
        "formal_training_started": False,
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "smoke_report": {
            "path": output["report_path"],
            "sha256": artifact_sha,
            "size_bytes": len(artifact_bytes),
        },
        "smoke_preflight_report_sha256": bindings["preflight_report_sha256"],
        "smoke_preflight_request_sha256": bindings["preflight_request_sha256"],
        "status": "pass",
    }
    return artifact, receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-request", required=True, type=Path)
    parser.add_argument("--expected-preflight-request-sha256", required=True)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--expected-preflight-report-sha256", required=True)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--receipt-path", required=True, type=Path)
    parser.add_argument("--ephemeral-root", required=True, type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all pins/projections and stop before importing torch or touching CUDA",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    request_path = args.preflight_request.resolve(strict=True)
    report_preflight_path = args.preflight_report.resolve(strict=True)
    _regular_nonsymlink(request_path, "smoke preflight request")
    _regular_nonsymlink(report_preflight_path, "smoke preflight report")
    request, preflight_report, artifacts, bindings = _load_preflight_pair(
        request_path,
        report_preflight_path,
        expected_request_sha256=args.expected_preflight_request_sha256,
        expected_report_sha256=args.expected_preflight_report_sha256,
    )
    projections = _load_projection_entries(
        artifacts["arm_projection_manifest"],
        request["authorization"],
        expected_sha256=bindings["arm_projection_manifest_sha256"],
    )
    fixed_row = _derive_fixed_row(
        artifacts["action_reference"],
        request["authorization"]["paired_global_step"],
        expected_sha256=bindings["reference_artifact_sha256"],
    )
    effective_root = Path(request["effective_output_root"]).resolve(strict=True)
    report_path = args.report_path.absolute()
    receipt_path = args.receipt_path.absolute()
    ephemeral_root = args.ephemeral_root.resolve(strict=True)
    for label, path in (
        ("report", report_path),
        ("receipt", receipt_path),
        ("ephemeral root", ephemeral_root),
    ):
        if not path.is_relative_to(effective_root):
            raise SmokeExecutionError(
                f"{label} must be under the preflight output root"
            )
    forbidden_primary_roots = [
        projections[arm]["run_directory"] for arm in request["authorization"]["arms"]
    ]
    for primary in map(Path, forbidden_primary_roots):
        primary_absolute = primary.absolute()
        if ephemeral_root == primary_absolute or ephemeral_root.is_relative_to(
            primary_absolute
        ):
            raise SmokeExecutionError("ephemeral root overlaps a primary arm directory")
    output = {
        "ephemeral_root": str(ephemeral_root),
        "forbidden_primary_roots": forbidden_primary_roots,
        "receipt_path": str(receipt_path),
        "report_path": str(report_path),
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "gpu_started": False,
                    "preflight_report_sha256": bindings["preflight_report_sha256"],
                    "preflight_request_sha256": bindings["preflight_request_sha256"],
                    "status": "validated_only",
                },
                sort_keys=True,
            )
        )
        return 0

    if report_path.exists() or receipt_path.exists():
        raise SmokeExecutionError(
            "smoke report/receipt already exists; refusing overwrite"
        )
    artifact, receipt = _run_smoke(
        request,
        preflight_report,
        bindings=bindings,
        fixed_row=fixed_row,
        output=output,
        artifact_paths=artifacts,
        projections=projections,
    )
    report_bytes = canonical_json_bytes(artifact)
    receipt_bytes = canonical_json_bytes(receipt)
    _exclusive_write(report_path, report_bytes, "smoke report")
    try:
        _exclusive_write(receipt_path, receipt_bytes, "smoke receipt")
    except BaseException:
        # A report without its detached SHA receipt must never authorize training.
        report_path.unlink()
        raise
    print(
        json.dumps(
            {
                "artifact_role": SMOKE_ARTIFACT_ROLE,
                "report_path": str(report_path),
                "report_sha256": sha256(report_bytes).hexdigest(),
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, SmokeExecutionError) as exc:
        print(f"phase6 real-2B smoke FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
