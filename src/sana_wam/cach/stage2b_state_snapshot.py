"""Canonical raw-tensor codec for synthetic CPU Stage-2B state snapshots.

This is the bounded L1 codec described by the Stage-2B durable-ledger design.
It serializes no Python objects and never uses pickle or ``torch.save``.
Tensor payloads are contiguous raw bytes held in content-addressed blobs; the
canonical JSON manifest carries their logical identities and exact metadata.

The codec is intentionally **CPU/synthetic-only**.  It does not provide CUDA
device or stream semantics, filesystem publication, a ledger, restart replay,
manager installation, admission, or scientific authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import sys
from typing import Any

import torch
from torch import Tensor

from sana_wam.cach.committed_action_history import (
    CommittedActionHistory,
    CommittedActionHistoryError,
    CommittedActionSource,
    CommittedActionSpan,
    committed_action_tensor_digest,
)
from sana_wam.cach.staging_variant import (
    CACHStagingVariant,
    history_summary_operator_for_variant,
)
from sana_wam.model.action_chunk_layout import (
    ChunkActionLayout,
    canonical_json_bytes as layout_canonical_json_bytes,
)
from sana_wam.model.video_backbone.sana import hybrid_cache as base
from sana_wam.model.video_backbone.sana.hybrid_cache_stage2b import (
    Stage2BTemporalState,
)


_SCHEMA = "cach.stage2b.state_snapshot.v1"
_SCOPE = "cpu_synthetic_only"
_TENSOR_ENCODING = "contiguous_raw_bytes.v1"
_CACHE_SCHEMA = "cach.hybrid_temporal_state.v1"
_HISTORY_SCHEMA = "cach.committed_action_history.v1"

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "float32": torch.float32,
}
_DTYPE_NAMES = {value: key for key, value in _DTYPES.items()}

_TOP_KEYS = frozenset(
    {
        "admission_scope",
        "blob_encoding",
        "cache_state",
        "cache_state_manifest_digest",
        "committed_action_history",
        "committed_action_history_digest",
        "history_summary_operator",
        "layout_identity",
        "logical_devices",
        "registry_identity",
        "schema",
        "staging_variant",
        "state_manifest",
        "state_manifest_digest",
    }
)
_LAYOUT_IDENTITY_KEYS = frozenset(
    {"layout_instance_digest", "layout_manifest", "layout_spec_sha256"}
)
_REGISTRY_IDENTITY_KEYS = frozenset(
    {"layer_registry_digest", "registry_manifest"}
)
_CACHE_KEYS = frozenset(
    {
        "action_cursor",
        "committed_pair_digest",
        "committed_through_chunk",
        "episode_epoch",
        "episode_id",
        "last_commit_source",
        "last_source_proof_digest",
        "layer_registry_digest",
        "layer_states",
        "layout_instance_digest",
        "layout_spec_sha256",
        "next_chunk_id",
        "pair_evidence_mode",
        "revision",
        "schema",
    }
)
_LAYER_KEYS = frozenset(
    {"content_time_sha256", "kind", "layer_index", "tensors", "through"}
)
_TENSOR_KEYS = frozenset(
    {
        "byte_length",
        "byte_order",
        "dtype",
        "field_name",
        "logical_device",
        "logical_tensor_id",
        "numel",
        "raw_blob_sha256",
        "raw_tensor_digest",
        "shape",
    }
)
_HISTORY_KEYS = frozenset(
    {
        "action_cursor",
        "action_dim",
        "batch_size",
        "committed_through_chunk",
        "device",
        "dtype",
        "episode_epoch",
        "episode_id",
        "layout_instance_digest",
        "layout_spec_sha256",
        "schema",
        "spans",
    }
)
_SPAN_KEYS = frozenset(
    {
        "action_end_exclusive",
        "action_start",
        "actions",
        "actions_digest",
        "chunk_id",
        "commit_id",
        "commit_source",
        "source_proof_digest",
    }
)
_CONTENT_TIME_KEYS = frozenset(
    {
        "action_rope_end_exclusive",
        "action_rope_start",
        "action_token_end_exclusive",
        "action_token_start",
        "bootstrap_mode",
        "chunk_id",
        "episode_epoch",
        "episode_id",
        "latent_end_exclusive",
        "latent_start",
        "layout_instance_digest",
        "layout_spec_sha256",
        "observed_prefix_chunks",
        "raw_observation_end_exclusive",
        "raw_observation_start",
    }
)


class Stage2BStateSnapshotError(RuntimeError):
    """Fail-closed snapshot validation error with a stable reason code."""

    def __init__(self, code: str, message: str):
        if not isinstance(code, str) or not code:
            raise ValueError("snapshot error code must be non-empty")
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class Stage2BTensorBlob:
    """One content-addressed raw byte object.

    Validation intentionally occurs at decode so corruption can be represented
    and deterministically rejected by tests or a future object store reader.
    """

    sha256: str
    payload: bytes


@dataclass(frozen=True)
class EncodedStage2BStateSnapshot:
    """Canonical snapshot manifest plus a canonical tuple of unique blobs."""

    manifest_bytes: bytes
    blobs: tuple[Stage2BTensorBlob, ...]

    @property
    def manifest_sha256(self) -> str:
        if type(self.manifest_bytes) is not bytes:
            _fail("SNAPSHOT_SCHEMA_MISMATCH", "manifest_bytes must be bytes")
        return _sha256(self.manifest_bytes)


def _fail(code: str, message: str) -> None:
    raise Stage2BStateSnapshotError(code, message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("SNAPSHOT_SCHEMA_MISMATCH", f"{name} must be a lowercase SHA256")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("SNAPSHOT_SCHEMA_MISMATCH", f"{name} must be a non-empty string")
    return value


def _require_uint(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        _fail("SNAPSHOT_SCHEMA_MISMATCH", f"{name} must be a plain uint")
    return value


def _require_optional_uint(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _require_uint(value, name)


def _require_optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, name)


def _require_dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail("SNAPSHOT_SCHEMA_MISMATCH", f"{name} must be a JSON object")
    return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("SNAPSHOT_SCHEMA_MISMATCH", f"{name} must be a JSON array")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        _fail(
            "SNAPSHOT_SCHEMA_MISMATCH",
            f"{name} members differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}",
        )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("SNAPSHOT_SCHEMA_MISMATCH", f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail("SNAPSHOT_SCHEMA_MISMATCH", f"non-finite JSON constant: {value}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise Stage2BStateSnapshotError(
            "SNAPSHOT_SCHEMA_MISMATCH",
            "value is not canonical-JSON encodable",
        ) from exc


def _parse_canonical_manifest(payload: object) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        _fail("SNAPSHOT_SCHEMA_MISMATCH", "manifest must be non-empty bytes")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except Stage2BStateSnapshotError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2BStateSnapshotError(
            "SNAPSHOT_SCHEMA_MISMATCH",
            "manifest is not strict UTF-8 JSON",
        ) from exc
    manifest = _require_dict(value, "snapshot manifest")
    if _canonical_json_bytes(manifest) != payload:
        _fail("SNAPSHOT_SCHEMA_MISMATCH", "snapshot manifest is not canonical JSON")
    return manifest


def _normalized_layout_manifest(layout: ChunkActionLayout) -> dict[str, Any]:
    return json.loads(layout_canonical_json_bytes(layout.to_payload()))


def _layout_identity(layout: ChunkActionLayout) -> dict[str, object]:
    return {
        "layout_instance_digest": layout.layout_instance_digest,
        "layout_manifest": _normalized_layout_manifest(layout),
        "layout_spec_sha256": layout.layout_spec_sha256,
    }


def _registry_identity(registry: base.LayerRegistry) -> dict[str, object]:
    return {
        "layer_registry_digest": registry.manifest_digest,
        "registry_manifest": registry.to_manifest(),
    }


def _require_cpu_synthetic_contract(
    *,
    layout: ChunkActionLayout,
    registry: base.LayerRegistry,
) -> None:
    if not isinstance(layout, ChunkActionLayout):
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "expected_layout has wrong type")
    if layout.synthetic_test_only is not True:
        _fail(
            "SNAPSHOT_CPU_SYNTHETIC_ONLY",
            "L1 snapshot codec accepts only synthetic test layouts",
        )
    if not isinstance(registry, base.LayerRegistry):
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "expected_registry has wrong type")
    if sys.byteorder != "little":
        _fail(
            "SNAPSHOT_CPU_SYNTHETIC_ONLY",
            "L1 raw tensor encoding requires a little-endian CPU",
        )


def _require_device_map(device_map: Mapping[str, object]) -> None:
    if not isinstance(device_map, Mapping) or set(device_map) != {"cpu"}:
        _fail(
            "SNAPSHOT_CPU_SYNTHETIC_ONLY",
            "device_map must explicitly contain only logical device 'cpu'",
        )
    try:
        target = torch.device(device_map["cpu"])
    except (TypeError, RuntimeError) as exc:
        raise Stage2BStateSnapshotError(
            "SNAPSHOT_CPU_SYNTHETIC_ONLY",
            "device_map['cpu'] is not a torch device",
        ) from exc
    if target != torch.device("cpu"):
        _fail(
            "SNAPSHOT_CPU_SYNTHETIC_ONLY",
            "L1 device mapping may target only exact CPU",
        )


def _dtype_name(dtype: torch.dtype) -> str:
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        _fail(
            "SNAPSHOT_DTYPE_UNSUPPORTED",
            f"dtype {dtype} is outside the L1 whitelist",
        )
    return name


def _raw_tensor_bytes(tensor: Tensor) -> bytes:
    return (
        tensor.detach()
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes(order="C")
    )


def _validate_tensor_for_encode(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor) or tensor.layout != torch.strided:
        _fail("SNAPSHOT_TENSOR_MISMATCH", f"{name} must be a strided tensor")
    if tensor.device != torch.device("cpu"):
        _fail(
            "SNAPSHOT_CPU_SYNTHETIC_ONLY",
            f"{name} is not on exact logical device cpu",
        )
    _dtype_name(tensor.dtype)
    if tensor.dtype is torch.bool and any(
        byte not in (0, 1) for byte in _raw_tensor_bytes(tensor)
    ):
        _fail(
            "SNAPSHOT_TENSOR_MISMATCH",
            f"{name} bool tensor bytes are not canonical",
        )
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor.detach()).all()):
        _fail("SNAPSHOT_TENSOR_MISMATCH", f"{name} contains NaN or Inf")


def _add_blob(blobs: dict[str, bytes], raw: bytes) -> str:
    digest = _sha256(raw)
    prior = blobs.get(digest)
    if prior is not None and prior != raw:  # pragma: no cover - SHA256 collision.
        _fail("SNAPSHOT_BLOB_MISMATCH", "raw blob SHA256 collision")
    blobs[digest] = raw
    return digest


def _encode_tensor(
    *,
    tensor: Tensor,
    logical_tensor_id: str,
    field_name: str,
    blobs: dict[str, bytes],
) -> dict[str, object]:
    _validate_tensor_for_encode(tensor, logical_tensor_id)
    raw = _raw_tensor_bytes(tensor)
    return {
        "byte_length": len(raw),
        "byte_order": "little",
        "dtype": _dtype_name(tensor.dtype),
        "field_name": field_name,
        "logical_device": "cpu",
        "logical_tensor_id": logical_tensor_id,
        "numel": tensor.numel(),
        "raw_blob_sha256": _add_blob(blobs, raw),
        "raw_tensor_digest": base.tensor_digest(tensor),
        "shape": list(tensor.shape),
    }


def _content_time_digest(content_time: base.ContentTime) -> str:
    return _sha256(_canonical_json_bytes(content_time.to_manifest()))


def _validate_layer_tensor_geometry(
    layer: base.LayerTemporalState,
    spec: base.LayerSpec,
) -> None:
    if layer.through is None:
        return
    tensors = {name: snapshot.clone_tensor() for name, snapshot in layer.tensors}
    if spec.kind is base.LayerKind.GDN_FULL_HISTORY:
        if tensors["main_s_kv"].ndim != 4 or tensors["main_s_z"].ndim != 4:
            _fail(
                "SNAPSHOT_TENSOR_MISMATCH",
                "GDN S_kv/S_z tensors must remain rank four",
            )
        return
    key = tensors["main_k_post_rope"]
    value = tensors["main_v"]
    if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
        _fail(
            "SNAPSHOT_TENSOR_MISMATCH",
            "softmax K/V tensors must remain equal-shape rank four",
        )
    assert spec.tokens_per_latent is not None
    expected_tokens = (
        layer.through.latent_end_exclusive - layer.through.latent_start
    ) * spec.tokens_per_latent
    if key.shape[2] != expected_tokens:
        _fail(
            "SNAPSHOT_TENSOR_MISMATCH",
            "softmax snapshot does not contain exactly the committed chunk",
        )
    if spec.camera_enabled:
        camera_key = tensors["camera_k_post_ucpe"]
        camera_value = tensors["camera_v_post_ucpe"]
        if (
            camera_key.ndim != 4
            or camera_value.ndim != 4
            or camera_key.shape != camera_value.shape
            or camera_key.shape[2] != expected_tokens
        ):
            _fail(
                "SNAPSHOT_TENSOR_MISMATCH",
                "camera softmax snapshot geometry differs",
            )


def _validate_state_bindings(
    *,
    state: Stage2BTemporalState,
    layout: ChunkActionLayout,
    registry: base.LayerRegistry,
) -> None:
    _require_cpu_synthetic_contract(layout=layout, registry=registry)
    if not isinstance(state, Stage2BTemporalState):
        _fail("SNAPSHOT_SCHEMA_MISMATCH", "state must be Stage2BTemporalState")
    if (
        state.layout_spec_sha256 != layout.layout_spec_sha256
        or state.layout_instance_digest != layout.layout_instance_digest
        or state.cache_state.layer_registry_digest != registry.manifest_digest
    ):
        _fail(
            "SNAPSHOT_IDENTITY_MISMATCH",
            "state differs from caller-bound layout/registry identity",
        )
    if len(state.layer_states) != len(registry.layers):
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "state/registry layer counts differ")
    if state.revision > len(layout.chunks):
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "state revision exceeds layout")

    for layer, spec in zip(state.layer_states, registry.layers, strict=True):
        if layer.layer_index != spec.layer_index or layer.kind is not spec.kind:
            _fail("SNAPSHOT_IDENTITY_MISMATCH", "state layer differs from registry")
        names = frozenset(name for name, _ in layer.tensors)
        if state.revision == 0:
            if layer.through is not None or names:
                _fail("SNAPSHOT_STATE_MISMATCH", "empty layer carries state")
        else:
            if names != spec.required_tensor_fields or layer.through is None:
                _fail(
                    "SNAPSHOT_STATE_MISMATCH",
                    "committed layer tensor fields differ from registry",
                )
            chunk_id = state.committed_through_chunk
            if chunk_id is None or chunk_id >= len(layout.chunks):
                _fail("SNAPSHOT_STATE_MISMATCH", "invalid committed chunk")
            try:
                layer.through.verify_layout_binding(
                    layout=layout,
                    chunk=layout.chunks[chunk_id],
                )
            except (TypeError, ValueError, base.CacheContractError) as exc:
                raise Stage2BStateSnapshotError(
                    "SNAPSHOT_IDENTITY_MISMATCH",
                    "content time differs from caller-bound layout",
                ) from exc
        for name, snapshot in layer.tensors:
            assert layer.through is not None
            expected_logical_id = (
                f"layer/{layer.layer_index}/{name}/"
                f"episode/{state.episode_epoch}/chunk/{layer.through.chunk_id}"
            )
            if snapshot.logical_tensor_id != expected_logical_id:
                _fail(
                    "SNAPSHOT_TENSOR_MISMATCH",
                    "cache logical tensor ID differs from manager identity",
                )
            _validate_tensor_for_encode(snapshot._tensor, name)
            cloned_tensor = snapshot.clone_tensor()
            if base.tensor_digest(cloned_tensor) != base.tensor_digest(
                snapshot._tensor
            ):
                _fail(
                    "SNAPSHOT_TENSOR_MISMATCH",
                    "cache tensor clone changed its exact raw identity",
                )
        _validate_layer_tensor_geometry(layer, spec)

    history = state.committed_action_history
    if history is None:
        if state.revision != 0:
            _fail("SNAPSHOT_HISTORY_MISMATCH", "committed state has no history")
    else:
        try:
            history.assert_integrity()
        except CommittedActionHistoryError as exc:
            raise Stage2BStateSnapshotError(
                "SNAPSHOT_HISTORY_MISMATCH",
                "committed history integrity failed",
            ) from exc
        if len(history.spans) != state.revision:
            _fail("SNAPSHOT_HISTORY_MISMATCH", "history revision differs")
        for span in history.spans:
            if span.chunk_id >= len(layout.chunks):
                _fail("SNAPSHOT_HISTORY_MISMATCH", "history chunk exceeds layout")
            chunk = layout.chunks[span.chunk_id]
            if (
                span.action_start != chunk.action_start
                or span.action_end_exclusive != chunk.action_end
            ):
                _fail(
                    "SNAPSHOT_IDENTITY_MISMATCH",
                    "history action interval differs from expected layout",
                )
            _validate_tensor_for_encode(span.clone_actions(), "history actions")
    # Force all existing state/history tensor digests to be recomputed now.
    _ = state.state_manifest_digest


def _encode_cache_state(
    state: base.HybridTemporalState,
    blobs: dict[str, bytes],
) -> dict[str, object]:
    layers: list[dict[str, object]] = []
    for layer in state.layer_states:
        tensor_entries: list[dict[str, object]] = []
        for field_name, snapshot in layer.tensors:
            tensor_entries.append(
                _encode_tensor(
                    tensor=snapshot.clone_tensor(),
                    logical_tensor_id=snapshot.logical_tensor_id,
                    field_name=field_name,
                    blobs=blobs,
                )
            )
        layers.append(
            {
                "content_time_sha256": (
                    None
                    if layer.through is None
                    else _content_time_digest(layer.through)
                ),
                "kind": layer.kind.value,
                "layer_index": layer.layer_index,
                "tensors": tensor_entries,
                "through": None if layer.through is None else layer.through.to_manifest(),
            }
        )
    return {
        "action_cursor": state.action_cursor,
        "committed_pair_digest": state.committed_pair_digest,
        "committed_through_chunk": state.committed_through_chunk,
        "episode_epoch": state.episode_epoch,
        "episode_id": state.episode_id,
        "last_commit_source": (
            None if state.last_commit_source is None else state.last_commit_source.value
        ),
        "last_source_proof_digest": state.last_source_proof_digest,
        "layer_registry_digest": state.layer_registry_digest,
        "layer_states": layers,
        "layout_instance_digest": state.layout_instance_digest,
        "layout_spec_sha256": state.layout_spec_sha256,
        "next_chunk_id": state.next_chunk_id,
        "pair_evidence_mode": state.pair_evidence_mode,
        "revision": state.revision,
        "schema": _CACHE_SCHEMA,
    }


def _encode_history(
    history: CommittedActionHistory | None,
    blobs: dict[str, bytes],
) -> dict[str, object] | None:
    if history is None:
        return None
    spans: list[dict[str, object]] = []
    for span in history.spans:
        logical_id = (
            f"history/actions/episode/{history.episode_epoch}/chunk/{span.chunk_id}"
        )
        actions = span.clone_actions()
        descriptor = _encode_tensor(
            tensor=actions,
            logical_tensor_id=logical_id,
            field_name="actions",
            blobs=blobs,
        )
        if descriptor["raw_tensor_digest"] != span.actions_digest:
            _fail("SNAPSHOT_HISTORY_MISMATCH", "history action digest differs")
        spans.append(
            {
                "action_end_exclusive": span.action_end_exclusive,
                "action_start": span.action_start,
                "actions": descriptor,
                "actions_digest": span.actions_digest,
                "chunk_id": span.chunk_id,
                "commit_id": span.commit_id,
                "commit_source": span.commit_source.value,
                "source_proof_digest": span.source_proof_digest,
            }
        )
    return {
        "action_cursor": history.action_cursor,
        "action_dim": history.action_dim,
        "batch_size": history.batch_size,
        "committed_through_chunk": history.committed_through_chunk,
        "device": str(history.device),
        "dtype": _dtype_name(history.dtype),
        "episode_epoch": history.episode_epoch,
        "episode_id": history.episode_id,
        "layout_instance_digest": history.layout_instance_digest,
        "layout_spec_sha256": history.layout_spec_sha256,
        "schema": _HISTORY_SCHEMA,
        "spans": spans,
    }


def encode_stage2b_state_snapshot(
    *,
    state: Stage2BTemporalState,
    expected_layout: ChunkActionLayout,
    expected_registry: base.LayerRegistry,
) -> EncodedStage2BStateSnapshot:
    """Encode one exact detached Stage-2B state into canonical CPU bytes."""

    _validate_state_bindings(
        state=state,
        layout=expected_layout,
        registry=expected_registry,
    )
    blobs: dict[str, bytes] = {}
    history = state.committed_action_history
    manifest = {
        "admission_scope": _SCOPE,
        "blob_encoding": _TENSOR_ENCODING,
        "cache_state": _encode_cache_state(state.cache_state, blobs),
        "cache_state_manifest_digest": state.cache_state.state_manifest_digest,
        "committed_action_history": _encode_history(history, blobs),
        "committed_action_history_digest": (
            None if history is None else history.manifest_digest
        ),
        "history_summary_operator": history_summary_operator_for_variant(
            state.staging_variant
        ),
        "layout_identity": _layout_identity(expected_layout),
        "logical_devices": ["cpu"],
        "registry_identity": _registry_identity(expected_registry),
        "schema": _SCHEMA,
        "staging_variant": state.staging_variant.value,
        "state_manifest": state.to_manifest(),
        "state_manifest_digest": state.state_manifest_digest,
    }
    return EncodedStage2BStateSnapshot(
        manifest_bytes=_canonical_json_bytes(manifest),
        blobs=tuple(
            Stage2BTensorBlob(sha256=digest, payload=blobs[digest])
            for digest in sorted(blobs)
        ),
    )


def _validate_blob_table(
    snapshot: EncodedStage2BStateSnapshot,
) -> dict[str, bytes]:
    if not isinstance(snapshot.blobs, tuple):
        _fail("SNAPSHOT_BLOB_MISMATCH", "blobs must be a tuple")
    result: dict[str, bytes] = {}
    observed_order: list[str] = []
    for blob in snapshot.blobs:
        if not isinstance(blob, Stage2BTensorBlob):
            _fail("SNAPSHOT_BLOB_MISMATCH", "invalid tensor blob entry")
        digest = _require_sha256(blob.sha256, "blob.sha256")
        if type(blob.payload) is not bytes:
            _fail("SNAPSHOT_BLOB_MISMATCH", "blob payload must be exact bytes")
        if digest in result:
            _fail("SNAPSHOT_BLOB_MISMATCH", "duplicate content-addressed blob")
        if _sha256(blob.payload) != digest:
            _fail("SNAPSHOT_BLOB_MISMATCH", "blob payload SHA256 differs")
        result[digest] = blob.payload
        observed_order.append(digest)
    if observed_order != sorted(observed_order):
        _fail("SNAPSHOT_BLOB_MISMATCH", "blob table is not canonically ordered")
    return result


def _decode_tensor(
    descriptor_value: object,
    *,
    expected_field_name: str,
    blob_table: dict[str, bytes],
    referenced_blobs: set[str],
) -> tuple[Tensor, str]:
    descriptor = _require_dict(descriptor_value, "tensor descriptor")
    _require_exact_keys(descriptor, _TENSOR_KEYS, "tensor descriptor")
    field_name = _require_string(descriptor["field_name"], "field_name")
    if field_name != expected_field_name:
        _fail("SNAPSHOT_TENSOR_MISMATCH", "tensor field identity differs")
    logical_id = _require_string(
        descriptor["logical_tensor_id"],
        "logical_tensor_id",
    )
    if descriptor["logical_device"] != "cpu":
        _fail("SNAPSHOT_CPU_SYNTHETIC_ONLY", "tensor logical device is not cpu")
    dtype_name = descriptor["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        _fail("SNAPSHOT_DTYPE_UNSUPPORTED", "tensor dtype is not admitted")
    dtype = _DTYPES[dtype_name]
    shape_values = _require_list(descriptor["shape"], "tensor shape")
    shape = tuple(
        _require_uint(dimension, f"shape[{index}]")
        for index, dimension in enumerate(shape_values)
    )
    numel = _require_uint(descriptor["numel"], "tensor numel")
    if math.prod(shape) != numel:
        _fail("SNAPSHOT_TENSOR_MISMATCH", "tensor shape/numel differ")
    byte_length = _require_uint(descriptor["byte_length"], "tensor byte_length")
    if descriptor["byte_order"] != "little" or sys.byteorder != "little":
        _fail(
            "SNAPSHOT_CPU_SYNTHETIC_ONLY",
            "tensor raw byte order is not admitted little-endian",
        )
    expected_length = numel * torch.empty((), dtype=dtype).element_size()
    if byte_length != expected_length:
        _fail("SNAPSHOT_TENSOR_MISMATCH", "tensor byte length metadata differs")
    blob_sha = _require_sha256(descriptor["raw_blob_sha256"], "raw_blob_sha256")
    raw = blob_table.get(blob_sha)
    if raw is None:
        _fail("SNAPSHOT_BLOB_MISMATCH", "referenced tensor blob is missing")
    referenced_blobs.add(blob_sha)
    if len(raw) != expected_length:
        _fail(
            "SNAPSHOT_TENSOR_MISMATCH",
            "tensor blob has missing or trailing raw bytes",
        )
    if dtype is torch.bool and any(byte not in (0, 1) for byte in raw):
        _fail("SNAPSHOT_TENSOR_MISMATCH", "bool tensor bytes are not canonical")

    if expected_length == 0:
        tensor = torch.empty(shape, dtype=dtype, device="cpu")
    else:
        raw_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        tensor = raw_tensor.view(dtype).reshape(shape).contiguous()
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        _fail("SNAPSHOT_TENSOR_MISMATCH", "decoded tensor contains NaN or Inf")
    tensor_digest = _require_sha256(
        descriptor["raw_tensor_digest"],
        "raw_tensor_digest",
    )
    if base.tensor_digest(tensor) != tensor_digest:
        _fail("SNAPSHOT_TENSOR_MISMATCH", "decoded tensor digest differs")
    return tensor, logical_id


def _decode_content_time(value: object, expected_digest: object) -> base.ContentTime:
    manifest = _require_dict(value, "content time")
    _require_exact_keys(manifest, _CONTENT_TIME_KEYS, "content time")
    digest = _require_sha256(expected_digest, "content_time_sha256")
    if _sha256(_canonical_json_bytes(manifest)) != digest:
        _fail("SNAPSHOT_CONTENT_TIME_MISMATCH", "content-time digest differs")
    content_time = base.ContentTime(**manifest)
    if content_time.to_manifest() != manifest:
        _fail("SNAPSHOT_CONTENT_TIME_MISMATCH", "content-time round trip differs")
    return content_time


def _decode_cache_state(
    value: object,
    *,
    registry: base.LayerRegistry,
    blob_table: dict[str, bytes],
    referenced_blobs: set[str],
) -> base.HybridTemporalState:
    manifest = _require_dict(value, "cache_state")
    _require_exact_keys(manifest, _CACHE_KEYS, "cache_state")
    if manifest["schema"] != _CACHE_SCHEMA:
        _fail("SNAPSHOT_SCHEMA_MISMATCH", "cache schema differs")
    layers_value = _require_list(manifest["layer_states"], "layer_states")
    if len(layers_value) != len(registry.layers):
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "cache/registry layer counts differ")
    cache_episode_epoch = _require_uint(
        manifest["episode_epoch"],
        "episode_epoch",
    )
    layers: list[base.LayerTemporalState] = []
    for expected_index, (layer_value, spec) in enumerate(
        zip(layers_value, registry.layers, strict=True)
    ):
        layer_manifest = _require_dict(layer_value, f"layer[{expected_index}]")
        _require_exact_keys(layer_manifest, _LAYER_KEYS, f"layer[{expected_index}]")
        layer_index = _require_uint(layer_manifest["layer_index"], "layer_index")
        if layer_index != expected_index or layer_index != spec.layer_index:
            _fail("SNAPSHOT_IDENTITY_MISMATCH", "layer index differs")
        try:
            kind = base.LayerKind(layer_manifest["kind"])
        except (TypeError, ValueError) as exc:
            raise Stage2BStateSnapshotError(
                "SNAPSHOT_SCHEMA_MISMATCH",
                "unknown layer kind",
            ) from exc
        if kind is not spec.kind:
            _fail("SNAPSHOT_IDENTITY_MISMATCH", "layer kind differs from registry")
        through_value = layer_manifest["through"]
        content_digest = layer_manifest["content_time_sha256"]
        if through_value is None:
            if content_digest is not None:
                _fail(
                    "SNAPSHOT_CONTENT_TIME_MISMATCH",
                    "empty layer claims content-time digest",
                )
            through = None
        else:
            through = _decode_content_time(through_value, content_digest)

        tensor_values = _require_list(layer_manifest["tensors"], "layer tensors")
        names: list[str] = []
        tensors: list[tuple[str, base.TensorSnapshot]] = []
        for tensor_value in tensor_values:
            tensor_manifest = _require_dict(tensor_value, "layer tensor")
            field_name = _require_string(tensor_manifest.get("field_name"), "field_name")
            tensor, logical_id = _decode_tensor(
                tensor_manifest,
                expected_field_name=field_name,
                blob_table=blob_table,
                referenced_blobs=referenced_blobs,
            )
            if through is None:
                _fail("SNAPSHOT_STATE_MISMATCH", "tensor has no content time")
            expected_logical_id = (
                f"layer/{layer_index}/{field_name}/"
                f"episode/{cache_episode_epoch}/chunk/{through.chunk_id}"
            )
            if logical_id != expected_logical_id:
                _fail(
                    "SNAPSHOT_TENSOR_MISMATCH",
                    "cache logical tensor ID differs from manager identity",
                )
            names.append(field_name)
            tensors.append(
                (
                    field_name,
                    base.TensorSnapshot.from_tensor(logical_id, tensor),
                )
            )
        if names != sorted(names) or len(names) != len(set(names)):
            _fail("SNAPSHOT_SCHEMA_MISMATCH", "layer tensors are not unique/sorted")
        observed_fields = frozenset(names)
        if through is None:
            if observed_fields:
                _fail("SNAPSHOT_STATE_MISMATCH", "empty layer has tensors")
        elif observed_fields != spec.required_tensor_fields:
            _fail(
                "SNAPSHOT_IDENTITY_MISMATCH",
                "layer tensor fields differ from expected registry",
            )
        layers.append(
            base.LayerTemporalState(
                layer_index=layer_index,
                kind=kind,
                through=through,
                tensors=tuple(tensors),
            )
        )

    last_source = manifest["last_commit_source"]
    if last_source is None:
        commit_source = None
    else:
        try:
            commit_source = base.CommitSource(last_source)
        except (TypeError, ValueError) as exc:
            raise Stage2BStateSnapshotError(
                "SNAPSHOT_SCHEMA_MISMATCH",
                "unknown last commit source",
            ) from exc
    return base.HybridTemporalState(
        episode_id=_require_string(manifest["episode_id"], "episode_id"),
        episode_epoch=cache_episode_epoch,
        revision=_require_uint(manifest["revision"], "revision"),
        committed_through_chunk=_require_optional_uint(
            manifest["committed_through_chunk"],
            "committed_through_chunk",
        ),
        next_chunk_id=_require_uint(manifest["next_chunk_id"], "next_chunk_id"),
        action_cursor=_require_uint(manifest["action_cursor"], "action_cursor"),
        layout_spec_sha256=_require_sha256(
            manifest["layout_spec_sha256"],
            "layout_spec_sha256",
        ),
        layout_instance_digest=_require_sha256(
            manifest["layout_instance_digest"],
            "layout_instance_digest",
        ),
        layer_registry_digest=_require_sha256(
            manifest["layer_registry_digest"],
            "layer_registry_digest",
        ),
        layer_states=tuple(layers),
        committed_pair_digest=_require_optional_sha256(
            manifest["committed_pair_digest"],
            "committed_pair_digest",
        ),
        last_commit_source=commit_source,
        last_source_proof_digest=_require_optional_sha256(
            manifest["last_source_proof_digest"],
            "last_source_proof_digest",
        ),
        pair_evidence_mode=_require_string(
            manifest["pair_evidence_mode"],
            "pair_evidence_mode",
        ),
    )


def _decode_history(
    value: object,
    *,
    blob_table: dict[str, bytes],
    referenced_blobs: set[str],
) -> CommittedActionHistory | None:
    if value is None:
        return None
    manifest = _require_dict(value, "committed_action_history")
    _require_exact_keys(manifest, _HISTORY_KEYS, "committed_action_history")
    if manifest["schema"] != _HISTORY_SCHEMA:
        _fail("SNAPSHOT_SCHEMA_MISMATCH", "history schema differs")
    if manifest["device"] != "cpu":
        _fail("SNAPSHOT_CPU_SYNTHETIC_ONLY", "history device is not cpu")
    dtype_name = manifest["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        _fail("SNAPSHOT_DTYPE_UNSUPPORTED", "history dtype is not admitted")
    dtype = _DTYPES[dtype_name]
    if not dtype.is_floating_point:
        _fail("SNAPSHOT_DTYPE_UNSUPPORTED", "history dtype must be floating")
    history_episode_epoch = _require_uint(
        manifest["episode_epoch"],
        "history episode_epoch",
    )
    spans_value = _require_list(manifest["spans"], "history spans")
    spans: list[CommittedActionSpan] = []
    for expected_chunk, span_value in enumerate(spans_value):
        span_manifest = _require_dict(span_value, f"history span[{expected_chunk}]")
        _require_exact_keys(span_manifest, _SPAN_KEYS, "history span")
        chunk_id = _require_uint(span_manifest["chunk_id"], "history chunk_id")
        if chunk_id != expected_chunk:
            _fail("SNAPSHOT_HISTORY_MISMATCH", "history chunks are not contiguous")
        actions, logical_id = _decode_tensor(
            span_manifest["actions"],
            expected_field_name="actions",
            blob_table=blob_table,
            referenced_blobs=referenced_blobs,
        )
        expected_logical_id = (
            f"history/actions/episode/{history_episode_epoch}/chunk/{chunk_id}"
        )
        if logical_id != expected_logical_id:
            _fail(
                "SNAPSHOT_HISTORY_MISMATCH",
                "history logical tensor ID differs",
            )
        if actions.dtype is not dtype:
            _fail("SNAPSHOT_HISTORY_MISMATCH", "history action dtype differs")
        actions_digest = _require_sha256(
            span_manifest["actions_digest"],
            "actions_digest",
        )
        if committed_action_tensor_digest(actions) != actions_digest:
            _fail("SNAPSHOT_HISTORY_MISMATCH", "history action digest differs")
        try:
            source = CommittedActionSource(span_manifest["commit_source"])
        except (TypeError, ValueError) as exc:
            raise Stage2BStateSnapshotError(
                "SNAPSHOT_HISTORY_MISMATCH",
                "unknown history source",
            ) from exc
        spans.append(
            CommittedActionSpan(
                chunk_id=chunk_id,
                action_start=_require_uint(
                    span_manifest["action_start"],
                    "action_start",
                ),
                action_end_exclusive=_require_uint(
                    span_manifest["action_end_exclusive"],
                    "action_end_exclusive",
                ),
                commit_source=source,
                source_proof_digest=_require_sha256(
                    span_manifest["source_proof_digest"],
                    "source_proof_digest",
                ),
                commit_id=_require_string(span_manifest["commit_id"], "commit_id"),
                actions_digest=actions_digest,
                _actions=actions,
            )
        )
    history = CommittedActionHistory(
        episode_id=_require_string(manifest["episode_id"], "history episode_id"),
        episode_epoch=history_episode_epoch,
        layout_spec_sha256=_require_sha256(
            manifest["layout_spec_sha256"],
            "history layout_spec_sha256",
        ),
        layout_instance_digest=_require_sha256(
            manifest["layout_instance_digest"],
            "history layout_instance_digest",
        ),
        batch_size=_require_uint(manifest["batch_size"], "history batch_size"),
        action_dim=_require_uint(manifest["action_dim"], "history action_dim"),
        dtype=dtype,
        device=torch.device("cpu"),
        spans=tuple(spans),
    )
    if history.action_cursor != _require_uint(
        manifest["action_cursor"],
        "history action_cursor",
    ):
        _fail("SNAPSHOT_HISTORY_MISMATCH", "history cursor differs")
    if history.committed_through_chunk != _require_optional_uint(
        manifest["committed_through_chunk"],
        "history committed_through_chunk",
    ):
        _fail("SNAPSHOT_HISTORY_MISMATCH", "history terminal chunk differs")
    return history


def _decode_impl(
    snapshot: EncodedStage2BStateSnapshot,
    *,
    expected_manifest_sha256: str,
    expected_layout: ChunkActionLayout,
    expected_registry: base.LayerRegistry,
    expected_staging_variant: CACHStagingVariant,
    device_map: Mapping[str, object],
) -> Stage2BTemporalState:
    if not isinstance(snapshot, EncodedStage2BStateSnapshot):
        _fail("SNAPSHOT_SCHEMA_MISMATCH", "snapshot has wrong type")
    _require_cpu_synthetic_contract(
        layout=expected_layout,
        registry=expected_registry,
    )
    _require_device_map(device_map)
    expected_snapshot_digest = _require_sha256(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    )
    if snapshot.manifest_sha256 != expected_snapshot_digest:
        _fail("SNAPSHOT_MANIFEST_MISMATCH", "snapshot manifest SHA256 differs")
    manifest = _parse_canonical_manifest(snapshot.manifest_bytes)
    _require_exact_keys(manifest, _TOP_KEYS, "snapshot manifest")
    if manifest["schema"] != _SCHEMA:
        _fail("SNAPSHOT_SCHEMA_MISMATCH", "snapshot schema differs")
    if manifest["admission_scope"] != _SCOPE:
        _fail("SNAPSHOT_CPU_SYNTHETIC_ONLY", "snapshot scope differs")
    if manifest["blob_encoding"] != _TENSOR_ENCODING:
        _fail("SNAPSHOT_SCHEMA_MISMATCH", "tensor blob encoding differs")
    if manifest["logical_devices"] != ["cpu"]:
        _fail("SNAPSHOT_CPU_SYNTHETIC_ONLY", "logical device inventory differs")
    if not isinstance(expected_staging_variant, CACHStagingVariant):
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "expected variant has wrong type")
    if manifest["staging_variant"] != expected_staging_variant.value:
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "snapshot variant differs")
    if manifest["history_summary_operator"] != history_summary_operator_for_variant(
        expected_staging_variant
    ):
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "history summary operator differs")

    layout_identity = _require_dict(manifest["layout_identity"], "layout_identity")
    _require_exact_keys(layout_identity, _LAYOUT_IDENTITY_KEYS, "layout_identity")
    if layout_identity != _layout_identity(expected_layout):
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "expected layout identity differs")
    registry_identity = _require_dict(
        manifest["registry_identity"],
        "registry_identity",
    )
    _require_exact_keys(
        registry_identity,
        _REGISTRY_IDENTITY_KEYS,
        "registry_identity",
    )
    if registry_identity != _registry_identity(expected_registry):
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "expected registry identity differs")

    blob_table = _validate_blob_table(snapshot)
    referenced_blobs: set[str] = set()
    cache_state = _decode_cache_state(
        manifest["cache_state"],
        registry=expected_registry,
        blob_table=blob_table,
        referenced_blobs=referenced_blobs,
    )
    cache_digest = _require_sha256(
        manifest["cache_state_manifest_digest"],
        "cache_state_manifest_digest",
    )
    if cache_state.state_manifest_digest != cache_digest:
        _fail("SNAPSHOT_STATE_MISMATCH", "decoded cache-state digest differs")
    history = _decode_history(
        manifest["committed_action_history"],
        blob_table=blob_table,
        referenced_blobs=referenced_blobs,
    )
    history_digest = _require_optional_sha256(
        manifest["committed_action_history_digest"],
        "committed_action_history_digest",
    )
    if (None if history is None else history.manifest_digest) != history_digest:
        _fail("SNAPSHOT_HISTORY_MISMATCH", "decoded history digest differs")
    if set(blob_table) != referenced_blobs:
        _fail("SNAPSHOT_BLOB_MISMATCH", "snapshot has unreferenced extra blobs")

    state = Stage2BTemporalState(
        cache_state=cache_state,
        staging_variant=expected_staging_variant,
        committed_action_history=history,
    )
    if cache_state.layout_spec_sha256 != expected_layout.layout_spec_sha256:
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "cache layout spec differs")
    if cache_state.layout_instance_digest != expected_layout.layout_instance_digest:
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "cache layout instance differs")
    if cache_state.layer_registry_digest != expected_registry.manifest_digest:
        _fail("SNAPSHOT_IDENTITY_MISMATCH", "cache registry digest differs")
    recorded_state_manifest = _require_dict(
        manifest["state_manifest"],
        "state_manifest",
    )
    if state.to_manifest() != recorded_state_manifest:
        _fail("SNAPSHOT_STATE_MISMATCH", "outer state manifest differs")
    state_digest = _require_sha256(
        manifest["state_manifest_digest"],
        "state_manifest_digest",
    )
    if state.state_manifest_digest != state_digest:
        _fail("SNAPSHOT_STATE_MISMATCH", "outer state digest differs")
    _validate_state_bindings(
        state=state,
        layout=expected_layout,
        registry=expected_registry,
    )
    return state


def decode_stage2b_state_snapshot(
    snapshot: EncodedStage2BStateSnapshot,
    *,
    expected_manifest_sha256: str,
    expected_layout: ChunkActionLayout,
    expected_registry: base.LayerRegistry,
    expected_staging_variant: CACHStagingVariant,
    device_map: Mapping[str, object],
) -> Stage2BTemporalState:
    """Decode and fully revalidate one CPU snapshot without replay authority."""

    try:
        return _decode_impl(
            snapshot,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_layout=expected_layout,
            expected_registry=expected_registry,
            expected_staging_variant=expected_staging_variant,
            device_map=device_map,
        )
    except Stage2BStateSnapshotError:
        raise
    except Exception as exc:
        raise Stage2BStateSnapshotError(
            "SNAPSHOT_DECODE_FAILED",
            f"snapshot reconstruction failed: {type(exc).__name__}",
        ) from exc


__all__ = [
    "EncodedStage2BStateSnapshot",
    "Stage2BStateSnapshotError",
    "Stage2BTensorBlob",
    "decode_stage2b_state_snapshot",
    "encode_stage2b_state_snapshot",
]
