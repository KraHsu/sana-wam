"""Stable tensor-bit hashing for Phase-6 paired-reference traces."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any

import torch


TRACE_SCHEMA_VERSION = "sana-phase6-reference-tensor-trace-v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _validate_canonical_metadata(value: Any, *, path: str = "metadata") -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_canonical_metadata(item, path=f"{path}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{path} keys must be non-empty strings")
            _validate_canonical_metadata(item, path=f"{path}.{key}")
        return
    raise TypeError(
        f"{path} contains non-canonical type {value_type.__name__}"
    )


def tensor_descriptor(value: torch.Tensor | None) -> dict[str, Any] | None:
    """Hash logical raw bits while retaining the original tensor metadata."""

    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError("reference trace values must be tensors or None")
    if value.layout != torch.strided:
        raise ValueError("reference trace supports only strided tensors")
    detached = value.detach()
    logical = detached.contiguous().cpu()
    raw = logical.view(torch.uint8).numpy().tobytes(order="C")
    return {
        "device_class": value.device.type,
        "dtype": str(value.dtype),
        "layout": str(value.layout),
        "numel": value.numel(),
        "raw_bits_sha256": sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "stride": list(value.stride()),
    }


def trace_sha256(
    trace_kind: str,
    tensors: dict[str, torch.Tensor | None],
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    if not isinstance(trace_kind, str) or not trace_kind:
        raise ValueError("trace_kind must be a non-empty string")
    if not isinstance(tensors, dict) or any(
        not isinstance(name, str) or not name for name in tensors
    ):
        raise TypeError("trace tensors must be one string-keyed dict")
    canonical_metadata = {} if metadata is None else metadata
    if type(canonical_metadata) is not dict:
        raise TypeError("trace metadata must be one plain dict")
    _validate_canonical_metadata(canonical_metadata)
    tensor_descriptors = {}
    for name in sorted(tensors):
        try:
            tensor_descriptors[name] = tensor_descriptor(tensors[name])
        except (TypeError, ValueError) as exc:
            value_type = type(tensors[name]).__name__
            raise type(exc)(
                f"reference trace tensor {name!r} ({value_type}): {exc}"
            ) from exc
    descriptor = {
        "metadata": canonical_metadata,
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_kind": trace_kind,
        "tensors": tensor_descriptors,
    }
    return sha256(_canonical_json_bytes(descriptor)).hexdigest()


__all__ = ["TRACE_SCHEMA_VERSION", "tensor_descriptor", "trace_sha256"]
