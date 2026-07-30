"""Immutable sharded Phase1 references for action-facing cache consistency."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import torch
from safetensors import safe_open


AFCC_REFERENCE_SCHEMA_VERSION = "sana-phase6-afcc-reference-manifest-v1"
AFCC_REFERENCE_TENSOR_KEY = "action_facing_video_numerator"
AFCC_REFERENCE_SHAPE = (20, 4, 20, 28, 112)
AFCC_REFERENCE_ROW_COUNT = 504
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ActionFacingCacheReferenceError(ValueError):
    """Raised when an AFCC reference manifest or shard is not exact."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ActionFacingCacheReferenceError(
            f"AFCC manifest is not canonical-JSON serializable: {exc}"
        ) from exc


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact strided tensor bytes without a dtype conversion."""

    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        tensor = tensor.detach().to(device="cpu").contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _strict_json(data: bytes) -> dict[str, Any]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ActionFacingCacheReferenceError(
                    f"duplicate AFCC manifest key: {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ActionFacingCacheReferenceError(
                    f"non-finite AFCC manifest value: {token}"
                )
            ),
        )
    except ActionFacingCacheReferenceError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError, ValueError) as exc:
        raise ActionFacingCacheReferenceError(
            f"invalid AFCC reference manifest: {exc}"
        ) from exc
    if type(value) is not dict or data != canonical_json_bytes(value):
        raise ActionFacingCacheReferenceError(
            "AFCC reference manifest must be one canonical JSON object"
        )
    return value


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ActionFacingCacheReferenceError(
            f"{name} keys differ: expected={sorted(expected)}, got={actual}"
        )
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ActionFacingCacheReferenceError(f"{name} must be a lowercase SHA256")
    return value


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _immutable_regular_file(path: Any, name: str) -> tuple[str, os.stat_result]:
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ActionFacingCacheReferenceError(f"{name} must be an absolute path")
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ActionFacingCacheReferenceError(f"cannot stat {name}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ActionFacingCacheReferenceError(
            f"{name} must be a regular non-symlink file"
        )
    if stat.S_IMODE(info.st_mode) != 0o400 or info.st_nlink != 1:
        raise ActionFacingCacheReferenceError(
            f"{name} must have mode 0400 and link count one"
        )
    return os.path.realpath(path), info


@dataclass(frozen=True)
class _ReferenceRow:
    global_step: int
    common_input_trace_sha256: str
    action_sigma: float
    plan_row_sha256: str
    task_name: str
    path: str
    file_sha256: str
    size_bytes: int
    tensor_sha256: str


class ActionFacingCacheReferenceStore:
    """Validated lazy loader for 504 immutable BF16 numerator shards."""

    def __init__(
        self,
        *,
        manifest_path: str,
        manifest_sha256: str,
        expected_plan_sha256: str,
        expected_identity_sha256: str,
        expected_dataset_contract_sha256: str,
        expected_reference_checkpoint_sha256: str,
        expected_action_stats_sha256: str,
        expected_source_manifest_sha256: str,
        expected_row_count: int = AFCC_REFERENCE_ROW_COUNT,
        expected_shape: tuple[int, ...] = AFCC_REFERENCE_SHAPE,
        verify_all_shards: bool = True,
    ) -> None:
        manifest_realpath, manifest_stat = _immutable_regular_file(
            manifest_path, "AFCC reference manifest"
        )
        observed_manifest_sha256 = _sha256_file(manifest_realpath)
        if observed_manifest_sha256 != _sha(
            manifest_sha256, "AFCC reference manifest SHA256"
        ):
            raise ActionFacingCacheReferenceError(
                "AFCC reference manifest SHA256 differs"
            )
        data = Path(manifest_realpath).read_bytes()
        if len(data) != manifest_stat.st_size:
            raise ActionFacingCacheReferenceError(
                "AFCC reference manifest size changed while reading"
            )
        value = _strict_json(data)
        _exact_keys(
            value,
            {
                "action_stats_sha256",
                "dataset_contract_artifact_sha256",
                "identity_sha256",
                "plan_sha256",
                "reference_checkpoint_sha256",
                "row_count",
                "rows",
                "scalar_reference_artifact_sha256",
                "schema_version",
                "source_manifest_sha256",
                "tensor_contract",
                "precompute_config_file_sha256",
                "precompute_gpu_runtime",
                "reference_execution_authority_sha256",
            },
            "AFCC reference manifest",
        )
        if value["schema_version"] != AFCC_REFERENCE_SCHEMA_VERSION:
            raise ActionFacingCacheReferenceError(
                "AFCC reference manifest schema differs"
            )
        expected_pins = {
            "plan_sha256": expected_plan_sha256,
            "identity_sha256": expected_identity_sha256,
            "dataset_contract_artifact_sha256": expected_dataset_contract_sha256,
            "reference_checkpoint_sha256": expected_reference_checkpoint_sha256,
            "action_stats_sha256": expected_action_stats_sha256,
            "source_manifest_sha256": expected_source_manifest_sha256,
        }
        for key, expected in expected_pins.items():
            if value.get(key) != _sha(expected, key):
                raise ActionFacingCacheReferenceError(f"AFCC manifest {key} differs")
        for key in (
            "scalar_reference_artifact_sha256",
            "precompute_config_file_sha256",
            "reference_execution_authority_sha256",
        ):
            _sha(value[key], f"AFCC manifest {key}")
        if type(value["precompute_gpu_runtime"]) is not dict:
            raise ActionFacingCacheReferenceError(
                "AFCC precompute GPU runtime must be one object"
            )
        if type(expected_row_count) is not int or expected_row_count < 1:
            raise ValueError("expected_row_count must be positive")
        if value["row_count"] != expected_row_count:
            raise ActionFacingCacheReferenceError("AFCC reference row count differs")
        contract = _exact_keys(
            value["tensor_contract"],
            {"dtype", "key", "shape"},
            "AFCC tensor contract",
        )
        if (
            contract["dtype"] != "BF16"
            or contract["key"] != AFCC_REFERENCE_TENSOR_KEY
            or contract["shape"] != list(expected_shape)
        ):
            raise ActionFacingCacheReferenceError("AFCC tensor contract differs")
        if type(value["rows"]) is not list or len(value["rows"]) != expected_row_count:
            raise ActionFacingCacheReferenceError("AFCC manifest rows differ")

        rows = []
        seen_paths = set()
        for index, raw_row in enumerate(value["rows"], start=1):
            row = _exact_keys(
                raw_row,
                {
                    "common_input_trace_sha256",
                    "action_sigma",
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
                f"AFCC reference row {index}",
            )
            if row["global_step"] != index:
                raise ActionFacingCacheReferenceError(
                    "AFCC rows are not in global-step order"
                )
            for key in (
                "common_input_trace_sha256",
                "dataset_contract_row_sha256",
                "file_sha256",
                "plan_row_sha256",
                "tensor_sha256",
                "t0_reference_forward_trace_sha256",
            ):
                _sha(row[key], f"AFCC row {index} {key}")
            if type(row["action_sigma"]) is not float or row["action_sigma"] not in {
                0.5,
                0.9,
                1.0,
            }:
                raise ActionFacingCacheReferenceError(
                    f"AFCC row {index} action sigma differs"
                )
            if (
                type(row["dataset_index"]) is not int
                or row["dataset_index"] < 0
                or not isinstance(row["task_name"], str)
                or not row["task_name"]
                or type(row["size_bytes"]) is not int
                or row["size_bytes"] < 1
            ):
                raise ActionFacingCacheReferenceError(
                    f"AFCC row {index} scalar metadata differs"
                )
            shard_realpath, shard_stat = _immutable_regular_file(
                row["path"], f"AFCC reference shard {index}"
            )
            if shard_realpath in seen_paths or shard_stat.st_size != row["size_bytes"]:
                raise ActionFacingCacheReferenceError(
                    f"AFCC reference shard {index} path/size differs"
                )
            seen_paths.add(shard_realpath)
            if verify_all_shards and _sha256_file(shard_realpath) != row["file_sha256"]:
                raise ActionFacingCacheReferenceError(
                    f"AFCC reference shard {index} SHA256 differs"
                )
            rows.append(
                _ReferenceRow(
                    global_step=index,
                    common_input_trace_sha256=row["common_input_trace_sha256"],
                    action_sigma=row["action_sigma"],
                    plan_row_sha256=row["plan_row_sha256"],
                    task_name=row["task_name"],
                    path=shard_realpath,
                    file_sha256=row["file_sha256"],
                    size_bytes=row["size_bytes"],
                    tensor_sha256=row["tensor_sha256"],
                )
            )
        self.manifest_path = manifest_realpath
        self.manifest_sha256 = observed_manifest_sha256
        self.reference_checkpoint_sha256 = value["reference_checkpoint_sha256"]
        self.source_manifest_sha256 = value["source_manifest_sha256"]
        self.tensor_shape = tuple(expected_shape)
        self.rows = tuple(rows)

    def common_trace_for_global_step(self, global_step: int) -> str:
        return self._row(global_step).common_input_trace_sha256

    def audit_metadata_for_global_step(self, global_step: int) -> dict[str, Any]:
        row = self._row(global_step)
        return {
            "action_sigma": row.action_sigma,
            "plan_row_sha256": row.plan_row_sha256,
            "task_name": row.task_name,
        }

    def tensor_for_global_step(self, global_step: int) -> torch.Tensor:
        row = self._row(global_step)
        if _sha256_file(row.path) != row.file_sha256:
            raise ActionFacingCacheReferenceError(
                f"AFCC reference shard {global_step} changed after initialization"
            )
        with safe_open(row.path, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != [AFCC_REFERENCE_TENSOR_KEY]:
                raise ActionFacingCacheReferenceError(
                    f"AFCC reference shard {global_step} tensor keys differ"
                )
            tensor = handle.get_tensor(AFCC_REFERENCE_TENSOR_KEY)
        if (
            tensor.dtype != torch.bfloat16
            or tuple(tensor.shape) != self.tensor_shape
            or tensor.requires_grad
            or not bool(torch.isfinite(tensor).all())
            or tensor_sha256(tensor) != row.tensor_sha256
        ):
            raise ActionFacingCacheReferenceError(
                f"AFCC reference shard {global_step} tensor contract differs"
            )
        return tensor.detach()

    def _row(self, global_step: int) -> _ReferenceRow:
        if type(global_step) is not int or not 1 <= global_step <= len(self.rows):
            raise IndexError("AFCC global_step is outside the reference table")
        return self.rows[global_step - 1]


__all__ = [
    "AFCC_REFERENCE_ROW_COUNT",
    "AFCC_REFERENCE_SCHEMA_VERSION",
    "AFCC_REFERENCE_SHAPE",
    "AFCC_REFERENCE_TENSOR_KEY",
    "ActionFacingCacheReferenceError",
    "ActionFacingCacheReferenceStore",
    "canonical_json_bytes",
    "tensor_sha256",
]
