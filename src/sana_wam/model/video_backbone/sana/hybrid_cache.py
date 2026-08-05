"""Typed, immutable temporal-cache contract for CACH-SANA-WAM.

This module is deliberately independent from the vendored SANA cache layout.
It defines the Stage-1 authority object, content-time identity, portable state
manifests, read-only denoise views, and a transactional teacher-forcing commit.

It does *not* implement the ``list[10]`` GDN/softmax codec and does not call a
model.  A future reviewed adapter must translate :class:`CacheScratch` to a
transaction-local legacy cache and return :class:`StagedLayerPayload` objects.
The live :class:`HybridTemporalState` is never exposed to that adapter.

Only ``teacher_forcing_dataset_pair`` is enabled here.  Self-forcing remains
blocked on a closed objective and deploy commit remains blocked on a verified
``APPLIED_ACTION_ACK`` seam.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from sana_wam.model.action_chunk_layout import ChunkActionLayout, ChunkLayout


_SCHEMA = "cach.hybrid_temporal_state.v1"
_COMMIT_RECEIPT_SCHEMA = "cach.paired_commit_receipt.v2"
_RESET_RECEIPT_SCHEMA = "cach.cache_reset_receipt.v1"
_TEACHER_PROOF_SCHEMA = "cach.teacher_forcing_dataset_pair_proof.v2"
_TENSOR_DIGEST_SCHEMA = b"cach.tensor.v1\0"
_SYNTHETIC_CACHE_TEST_MARKER = "CACH_SYNTHETIC_CACHE_TEST_ONLY_V1"


class CacheContractError(RuntimeError):
    """Fail-closed contract error carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        if not isinstance(code, str) or not code:
            raise ValueError("CacheContractError code must be non-empty")
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise CacheContractError(code, message)


def _require_plain_uint(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        _fail("CACHE_SCHEMA_MISMATCH", f"{name} must be a non-negative plain int")
    return value


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("CACHE_SCHEMA_MISMATCH", f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail("CACHE_SCHEMA_MISMATCH", f"{name} must be a lowercase SHA256")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail("CACHE_SCHEMA_MISMATCH", f"value is not canonical-JSON encodable: {exc}")
    raise AssertionError("unreachable")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dtype_name(tensor: Tensor) -> str:
    name = str(tensor.dtype)
    return name[len("torch.") :] if name.startswith("torch.") else name


def _tensor_bytes(tensor: Tensor) -> bytes:
    cpu = tensor.detach().to(device="cpu").contiguous()
    # Viewing as uint8 keeps bfloat16 and other NumPy-incompatible dtypes exact.
    return cpu.view(torch.uint8).numpy().tobytes(order="C")


def tensor_digest(tensor: Tensor) -> str:
    """Return the canonical CACH tensor digest (metadata plus exact raw bytes)."""

    if not isinstance(tensor, Tensor):
        _fail("CACHE_SCHEMA_MISMATCH", "tensor_digest requires a torch.Tensor")
    if tensor.layout != torch.strided:
        _fail("CACHE_SCHEMA_MISMATCH", "cache tensors must use strided layout")
    if tensor.device.type == "meta":
        _fail("CACHE_SCHEMA_MISMATCH", "meta tensors have no canonical bytes")
    metadata = _canonical_json_bytes(
        {
            "device_type": tensor.device.type,
            "dtype": _dtype_name(tensor),
            "shape": list(tensor.shape),
        }
    )
    material = (
        _TENSOR_DIGEST_SCHEMA
        + len(metadata).to_bytes(8, "big")
        + metadata
        + _tensor_bytes(tensor)
    )
    return _sha256_bytes(material)


def _validate_finite_tensor(tensor: Tensor, name: str) -> None:
    if tensor.layout != torch.strided:
        _fail("CACHE_SCHEMA_MISMATCH", f"{name} must use strided layout")
    if tensor.device.type == "meta":
        _fail("CACHE_SCHEMA_MISMATCH", f"{name} cannot be a meta tensor")
    if tensor.is_floating_point() or tensor.is_complex():
        if not bool(torch.isfinite(tensor.detach()).all()):
            _fail("CACHE_NONFINITE_TENSOR", f"{name} contains NaN or Inf")


@dataclass(frozen=True)
class ContentTime:
    """Exact content-time identity for one committed layout chunk."""

    episode_id: str
    episode_epoch: int
    layout_spec_sha256: str
    layout_instance_digest: str
    chunk_id: int
    latent_start: int
    latent_end_exclusive: int
    raw_observation_start: int
    raw_observation_end_exclusive: int
    action_token_start: int
    action_token_end_exclusive: int
    action_rope_start: int
    action_rope_end_exclusive: int
    bootstrap_mode: str = "first_frame_pinned"
    observed_prefix_chunks: int = 0

    def __post_init__(self) -> None:
        _require_non_empty_string(self.episode_id, "episode_id")
        for name in (
            "episode_epoch",
            "chunk_id",
            "latent_start",
            "latent_end_exclusive",
            "raw_observation_start",
            "raw_observation_end_exclusive",
            "action_token_start",
            "action_token_end_exclusive",
            "action_rope_start",
            "action_rope_end_exclusive",
            "observed_prefix_chunks",
        ):
            _require_plain_uint(getattr(self, name), name)
        _require_sha256(self.layout_spec_sha256, "layout_spec_sha256")
        _require_sha256(self.layout_instance_digest, "layout_instance_digest")
        if self.latent_end_exclusive <= self.latent_start:
            _fail("CACHE_CONTENT_TIME_MISMATCH", "latent interval must be non-empty")
        if self.raw_observation_end_exclusive <= self.raw_observation_start:
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "raw observation interval must be non-empty",
            )
        if self.action_token_end_exclusive < self.action_token_start:
            _fail("CACHE_CONTENT_TIME_MISMATCH", "action interval is reversed")
        if (
            self.action_rope_start != self.action_token_start
            or self.action_rope_end_exclusive
            != self.action_token_end_exclusive
        ):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "Action RoPE interval must equal exact action ownership",
            )
        if self.bootstrap_mode != "first_frame_pinned":
            _fail(
                "BOOTSTRAP_CONTRACT_MISMATCH",
                "bootstrap_mode must be first_frame_pinned",
            )
        if self.observed_prefix_chunks != 0:
            _fail(
                "BOOTSTRAP_CONTRACT_MISMATCH",
                "observed_prefix_chunks must be exactly zero",
            )
        if self.chunk_id == 0 and (
            self.latent_start != 0
            or self.raw_observation_start != 0
            or self.action_token_start != 0
            or self.action_rope_start != 0
        ):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "bootstrap chunk must start at latent/action index zero",
            )

    @classmethod
    def from_layout(
        cls,
        *,
        episode_id: str,
        episode_epoch: int,
        layout: ChunkActionLayout,
        chunk: ChunkLayout,
    ) -> "ContentTime":
        """Derive content time from one exact immutable layout member.

        Raw observation ownership follows the registered ``t -> t+1`` action
        convention: bootstrap owns the anchor through its final destination
        state, while each continuation begins at the state after the previous
        chunk's terminal state.  This makes adjacent raw intervals disjoint,
        exhaustive, and contiguous.
        """

        if not isinstance(layout, ChunkActionLayout):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "layout must be a ChunkActionLayout",
            )
        if not isinstance(chunk, ChunkLayout):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "chunk must be a ChunkLayout",
            )
        if chunk.chunk_id >= len(layout.chunks):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "chunk id is outside the bound layout",
            )
        if layout.chunks[chunk.chunk_id] is not chunk:
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "chunk is not the exact member owned by the bound layout",
            )
        raw_start = 0 if chunk.chunk_id == 0 else chunk.action_start + 1
        raw_end_exclusive = chunk.action_end + 1
        if raw_end_exclusive > layout.valid_raw_count:
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "chunk raw interval exceeds the bound layout",
            )
        return cls(
            episode_id=episode_id,
            episode_epoch=episode_epoch,
            layout_spec_sha256=layout.layout_spec_sha256,
            layout_instance_digest=layout.layout_instance_digest,
            chunk_id=chunk.chunk_id,
            latent_start=chunk.latent_start,
            latent_end_exclusive=chunk.latent_end,
            raw_observation_start=raw_start,
            raw_observation_end_exclusive=raw_end_exclusive,
            action_token_start=chunk.action_start,
            action_token_end_exclusive=chunk.action_end,
            action_rope_start=chunk.action_rope_start,
            action_rope_end_exclusive=chunk.action_rope_end,
        )

    def verify_layout_binding(
        self,
        *,
        layout: ChunkActionLayout,
        chunk: ChunkLayout,
    ) -> None:
        """Require exact equality with geometry derived from ``layout``."""

        expected = type(self).from_layout(
            episode_id=self.episode_id,
            episode_epoch=self.episode_epoch,
            layout=layout,
            chunk=chunk,
        )
        if self != expected:
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "caller content-time differs from the bound layout chunk",
            )

    def to_manifest(self) -> dict:
        return {
            "action_rope_end_exclusive": self.action_rope_end_exclusive,
            "action_rope_start": self.action_rope_start,
            "action_token_end_exclusive": self.action_token_end_exclusive,
            "action_token_start": self.action_token_start,
            "bootstrap_mode": self.bootstrap_mode,
            "chunk_id": self.chunk_id,
            "episode_epoch": self.episode_epoch,
            "episode_id": self.episode_id,
            "latent_end_exclusive": self.latent_end_exclusive,
            "latent_start": self.latent_start,
            "layout_instance_digest": self.layout_instance_digest,
            "layout_spec_sha256": self.layout_spec_sha256,
            "observed_prefix_chunks": self.observed_prefix_chunks,
            "raw_observation_end_exclusive": self.raw_observation_end_exclusive,
            "raw_observation_start": self.raw_observation_start,
        }


class LayerKind(str, Enum):
    GDN_FULL_HISTORY = "gdn_full_history_v1"
    SOFTMAX_PREVIOUS_CHUNK = "softmax_previous_chunk_v1"


@dataclass(frozen=True)
class LayerSpec:
    """Reviewed layer registry entry; never inferred from legacy slot 6."""

    layer_index: int
    kind: LayerKind
    operator_class: str
    camera_enabled: bool = False
    main_shortconv_enabled: bool = False
    ffn_tconv_enabled: bool = True
    tokens_per_latent: int | None = None

    def __post_init__(self) -> None:
        _require_plain_uint(self.layer_index, "layer_index")
        if not isinstance(self.kind, LayerKind):
            _fail("CACHE_LAYER_KIND_MISMATCH", "kind must be a LayerKind")
        _require_non_empty_string(self.operator_class, "operator_class")
        if "attnres" in self.operator_class.casefold():
            _fail(
                "CACHE_ATTNRES_PERSISTED",
                "AttnRes is forward-local and cannot be a temporal-cache layer",
            )
        for name in (
            "camera_enabled",
            "main_shortconv_enabled",
            "ffn_tconv_enabled",
        ):
            if type(getattr(self, name)) is not bool:
                _fail("CACHE_SCHEMA_MISMATCH", f"{name} must be bool")
        if self.kind is LayerKind.SOFTMAX_PREVIOUS_CHUNK:
            if self.main_shortconv_enabled:
                _fail(
                    "CACHE_LAYER_KIND_MISMATCH",
                    "softmax cache cannot declare GDN shortconv state",
                )
            if type(self.tokens_per_latent) is not int or self.tokens_per_latent <= 0:
                _fail(
                    "CACHE_SCHEMA_MISMATCH",
                    "softmax layers require positive tokens_per_latent",
                )
        elif self.tokens_per_latent is not None:
            _fail(
                "CACHE_SCHEMA_MISMATCH",
                "tokens_per_latent is only valid for softmax layers",
            )

    @property
    def required_tensor_fields(self) -> frozenset[str]:
        if self.kind is LayerKind.GDN_FULL_HISTORY:
            fields = {"main_s_kv", "main_s_z"}
            if self.camera_enabled:
                fields.add("camera_s_kv")
            if self.main_shortconv_enabled:
                fields.add("main_shortconv_left_context")
        else:
            fields = {"main_k_post_rope", "main_v"}
            if self.camera_enabled:
                fields.update({"camera_k_post_ucpe", "camera_v_post_ucpe"})
        if self.ffn_tconv_enabled:
            fields.add("ffn_tconv_left_context")
        return frozenset(fields)

    def to_manifest(self) -> dict:
        return {
            "camera_enabled": self.camera_enabled,
            "ffn_tconv_enabled": self.ffn_tconv_enabled,
            "kind": self.kind.value,
            "layer_index": self.layer_index,
            "main_shortconv_enabled": self.main_shortconv_enabled,
            "operator_class": self.operator_class,
            "required_tensor_fields": sorted(self.required_tensor_fields),
            "tokens_per_latent": self.tokens_per_latent,
        }


@dataclass(frozen=True)
class LayerRegistry:
    """Immutable, contiguous layer table bound into every state manifest."""

    layers: tuple[LayerSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layers, tuple) or not self.layers:
            _fail("CACHE_SCHEMA_MISMATCH", "layer registry must be a non-empty tuple")
        if any(not isinstance(layer, LayerSpec) for layer in self.layers):
            _fail("CACHE_SCHEMA_MISMATCH", "registry entries must be LayerSpec")
        expected = tuple(range(len(self.layers)))
        observed = tuple(layer.layer_index for layer in self.layers)
        if observed != expected:
            _fail(
                "CACHE_SCHEMA_MISMATCH",
                f"layer indices must be contiguous {expected}, got {observed}",
            )

    def to_manifest(self) -> dict:
        return {
            "schema": "cach.layer_registry.v1",
            "layers": [layer.to_manifest() for layer in self.layers],
        }

    @property
    def manifest_digest(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_manifest()))


def validate_cach_a_registry(registry: LayerRegistry) -> None:
    """Require CACH-A's frozen topology: 20 GDN layers and no softmax anchor."""

    if not isinstance(registry, LayerRegistry):
        _fail("CACHE_SCHEMA_MISMATCH", "registry must be a LayerRegistry")
    if len(registry.layers) != 20:
        _fail(
            "CACHE_CACH_A_TOPOLOGY_MISMATCH",
            "CACH-A requires exactly 20 temporal layers",
        )
    if any(
        layer.kind is not LayerKind.GDN_FULL_HISTORY
        for layer in registry.layers
    ):
        _fail(
            "CACHE_CACH_A_TOPOLOGY_MISMATCH",
            "CACH-A requires 20 GDN layers and zero softmax layers",
        )


@dataclass(frozen=True)
class TensorSnapshot:
    """Detached private tensor clone plus its portable logical identity."""

    logical_tensor_id: str
    _tensor: Tensor = field(repr=False, compare=False)

    @classmethod
    def from_tensor(cls, logical_tensor_id: str, tensor: Tensor) -> "TensorSnapshot":
        _require_non_empty_string(logical_tensor_id, "logical_tensor_id")
        if not isinstance(tensor, Tensor):
            _fail("CACHE_SCHEMA_MISMATCH", "layer payload values must be tensors")
        _validate_finite_tensor(tensor, logical_tensor_id)
        clone = tensor.detach().clone(memory_format=torch.contiguous_format)
        clone.requires_grad_(False)
        return cls(logical_tensor_id=logical_tensor_id, _tensor=clone)

    def clone_tensor(self) -> Tensor:
        """Return a transaction-local clone; never expose the live state tensor."""

        return self._tensor.detach().clone(memory_format=torch.contiguous_format)

    def to_manifest(self) -> dict:
        return {
            "device": str(self._tensor.device),
            "dtype": _dtype_name(self._tensor),
            "logical_tensor_id": self.logical_tensor_id,
            "numel": self._tensor.numel(),
            "raw_tensor_digest": tensor_digest(self._tensor),
            "shape": list(self._tensor.shape),
        }


@dataclass(frozen=True)
class LayerTemporalState:
    """One typed layer state; ``through=None`` is its only legal empty form."""

    layer_index: int
    kind: LayerKind
    through: ContentTime | None
    tensors: tuple[tuple[str, TensorSnapshot], ...] = ()

    def __post_init__(self) -> None:
        _require_plain_uint(self.layer_index, "layer_index")
        if not isinstance(self.kind, LayerKind):
            _fail("CACHE_LAYER_KIND_MISMATCH", "state kind must be a LayerKind")
        if not isinstance(self.tensors, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in self.tensors
        ):
            _fail("CACHE_SCHEMA_MISMATCH", "layer tensors must be name/value pairs")
        names = tuple(name for name, _ in self.tensors)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            _fail(
                "CACHE_SCHEMA_MISMATCH",
                "layer tensor fields must be unique and UTF-8-name sorted",
            )
        if self.through is None and self.tensors:
            _fail("CACHE_SCHEMA_MISMATCH", "empty layer state cannot contain tensors")
        if self.through is not None and not self.tensors:
            _fail("CACHE_SCHEMA_MISMATCH", "committed layer state requires tensors")
        for name, snapshot in self.tensors:
            _require_non_empty_string(name, "tensor field name")
            if "attnres" in name.casefold():
                _fail(
                    "CACHE_ATTNRES_PERSISTED",
                    "AttnRes tensor appeared in temporal state",
                )
            if not isinstance(snapshot, TensorSnapshot):
                _fail("CACHE_SCHEMA_MISMATCH", "invalid tensor snapshot")

    @property
    def is_empty(self) -> bool:
        return self.through is None

    def to_manifest(self) -> dict:
        return {
            "kind": self.kind.value,
            "layer_index": self.layer_index,
            "tensors": {
                name: snapshot.to_manifest() for name, snapshot in self.tensors
            },
            "through": None if self.through is None else self.through.to_manifest(),
        }


class CommitSource(str, Enum):
    TEACHER_FORCING = "teacher_forcing_dataset_pair"
    SELF_FORCING = "self_forcing_generated_pair"
    DEPLOY_APPLIED_ACK = "deploy_applied_ack"


@dataclass(frozen=True)
class HybridTemporalState:
    """Single immutable live pointer owned by :class:`HybridCacheManager`."""

    episode_id: str
    episode_epoch: int
    revision: int
    committed_through_chunk: int | None
    next_chunk_id: int
    action_cursor: int
    layout_spec_sha256: str
    layout_instance_digest: str
    layer_registry_digest: str
    layer_states: tuple[LayerTemporalState, ...]
    committed_pair_digest: str | None = None
    last_commit_source: CommitSource | None = None
    last_source_proof_digest: str | None = None
    pair_evidence_mode: str = "none"

    def __post_init__(self) -> None:
        _require_non_empty_string(self.episode_id, "episode_id")
        _require_sha256(self.layout_spec_sha256, "layout_spec_sha256")
        _require_sha256(self.layout_instance_digest, "layout_instance_digest")
        _require_sha256(self.layer_registry_digest, "layer_registry_digest")
        for name in ("episode_epoch", "revision", "next_chunk_id", "action_cursor"):
            _require_plain_uint(getattr(self, name), name)
        if self.committed_through_chunk is not None:
            _require_plain_uint(
                self.committed_through_chunk, "committed_through_chunk"
            )
        if not isinstance(self.layer_states, tuple) or not self.layer_states:
            _fail("CACHE_SCHEMA_MISMATCH", "state requires layer_states")
        if any(not isinstance(layer, LayerTemporalState) for layer in self.layer_states):
            _fail("CACHE_SCHEMA_MISMATCH", "state has an invalid layer entry")
        if tuple(layer.layer_index for layer in self.layer_states) != tuple(
            range(len(self.layer_states))
        ):
            _fail("CACHE_SCHEMA_MISMATCH", "state layer indices are not contiguous")

        if self.revision == 0:
            if (
                self.committed_through_chunk is not None
                or self.next_chunk_id != 0
                or self.action_cursor != 0
                or any(not layer.is_empty for layer in self.layer_states)
                or self.committed_pair_digest is not None
                or self.last_commit_source is not None
                or self.last_source_proof_digest is not None
                or self.pair_evidence_mode != "none"
            ):
                _fail("CACHE_SCHEMA_MISMATCH", "invalid empty cache state")
            return

        if self.committed_through_chunk != self.next_chunk_id - 1:
            _fail("CACHE_GDN_NONMONOTONIC", "committed/next chunk cursor mismatch")
        if self.revision != self.next_chunk_id:
            _fail(
                "CACHE_GDN_NONMONOTONIC",
                "revision must equal the count of contiguous committed chunks",
            )
        _require_sha256(self.committed_pair_digest, "committed_pair_digest")
        _require_sha256(self.last_source_proof_digest, "last_source_proof_digest")
        if self.last_commit_source is not CommitSource.TEACHER_FORCING:
            _fail(
                "COMMIT_SOURCE_PROOF_MIXED",
                "Stage-1 state only admits teacher-forcing commits",
            )
        if self.pair_evidence_mode != "dataset_ground_truth":
            _fail(
                "COMMIT_SOURCE_PROOF_MIXED",
                "teacher source must map to dataset_ground_truth evidence",
            )
        through_values = [layer.through for layer in self.layer_states]
        if any(value is None for value in through_values):
            _fail("CACHE_SCHEMA_MISMATCH", "committed state has an empty layer")
        first = through_values[0]
        if any(value != first for value in through_values[1:]):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "all layer states must share exact content-time",
            )
        assert first is not None
        if (
            first.chunk_id != self.committed_through_chunk
            or first.episode_id != self.episode_id
            or first.episode_epoch != self.episode_epoch
            or first.layout_spec_sha256 != self.layout_spec_sha256
            or first.layout_instance_digest != self.layout_instance_digest
            or first.action_token_end_exclusive != self.action_cursor
            or first.action_rope_end_exclusive != self.action_cursor
        ):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "layer content-time does not match committed state identity",
            )

    @classmethod
    def empty(
        cls,
        *,
        episode_id: str,
        episode_epoch: int,
        layout_spec_sha256: str,
        layout_instance_digest: str,
        registry: LayerRegistry,
    ) -> "HybridTemporalState":
        return cls(
            episode_id=episode_id,
            episode_epoch=episode_epoch,
            revision=0,
            committed_through_chunk=None,
            next_chunk_id=0,
            action_cursor=0,
            layout_spec_sha256=layout_spec_sha256,
            layout_instance_digest=layout_instance_digest,
            layer_registry_digest=registry.manifest_digest,
            layer_states=tuple(
                LayerTemporalState(
                    layer_index=spec.layer_index,
                    kind=spec.kind,
                    through=None,
                )
                for spec in registry.layers
            ),
        )

    def to_manifest(self) -> dict:
        return {
            "action_cursor": self.action_cursor,
            "committed_pair_digest": self.committed_pair_digest,
            "committed_through_chunk": self.committed_through_chunk,
            "episode_epoch": self.episode_epoch,
            "episode_id": self.episode_id,
            "last_commit_source": (
                None
                if self.last_commit_source is None
                else self.last_commit_source.value
            ),
            "last_source_proof_digest": self.last_source_proof_digest,
            "layer_registry_digest": self.layer_registry_digest,
            "layer_states": [layer.to_manifest() for layer in self.layer_states],
            "layout_instance_digest": self.layout_instance_digest,
            "layout_spec_sha256": self.layout_spec_sha256,
            "next_chunk_id": self.next_chunk_id,
            "pair_evidence_mode": self.pair_evidence_mode,
            "revision": self.revision,
            "schema": _SCHEMA,
        }

    @property
    def state_manifest_digest(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_manifest()))


def _runtime_tensor_fingerprint(tensor: Tensor) -> tuple:
    storage = tensor.untyped_storage()
    return (
        id(tensor),
        storage.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        _dtype_name(tensor),
        str(tensor.device),
        bool(tensor.requires_grad),
        tensor._version,
        tensor_digest(tensor),
    )


class _TrackedTensorMap(dict[str, Tensor]):
    """Dictionary whose mutation counter cannot be restored by undoing a write."""

    def __init__(self, values: Mapping[str, Tensor]):
        super().__init__(values)
        self.mutation_count = 0

    def _changed(self) -> None:
        self.mutation_count += 1

    def __setitem__(self, key: str, value: Tensor) -> None:
        self._changed()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._changed()
        super().__delitem__(key)

    def clear(self) -> None:
        self._changed()
        super().clear()

    def pop(self, key: str, default=...):
        self._changed()
        if default is ...:
            return super().pop(key)
        return super().pop(key, default)

    def popitem(self):
        self._changed()
        return super().popitem()

    def setdefault(self, key: str, default: Tensor | None = None):
        if key not in self:
            self._changed()
        return super().setdefault(key, default)

    def update(self, *args, **kwargs) -> None:
        self._changed()
        super().update(*args, **kwargs)

    def __ior__(self, other):
        self._changed()
        return super().__ior__(other)


@dataclass(frozen=True)
class LayerScratch:
    layer_index: int
    kind: LayerKind
    tensors: _TrackedTensorMap


@dataclass(frozen=True)
class CacheScratch:
    """Transaction-local tensor clones, never the live authority object."""

    layers: tuple[LayerScratch, ...]
    source_state_manifest_digest: str
    _source_state_identity: int = field(repr=False)
    _baseline_fingerprint: tuple = field(repr=False)

    @classmethod
    def from_state(
        cls,
        state: HybridTemporalState,
        *,
        _source_state_identity: int | None = None,
    ) -> "CacheScratch":
        layers = []
        for layer in state.layer_states:
            layers.append(
                LayerScratch(
                    layer_index=layer.layer_index,
                    kind=layer.kind,
                    tensors=_TrackedTensorMap(
                        {
                            name: snapshot.clone_tensor()
                            for name, snapshot in layer.tensors
                        }
                    ),
                )
            )
        provisional = cls(
            layers=tuple(layers),
            source_state_manifest_digest=state.state_manifest_digest,
            _source_state_identity=(
                id(state)
                if _source_state_identity is None
                else _source_state_identity
            ),
            _baseline_fingerprint=(),
        )
        object.__setattr__(
            provisional,
            "_baseline_fingerprint",
            provisional.runtime_fingerprint(),
        )
        return provisional

    def runtime_fingerprint(self) -> tuple:
        return tuple(
            (
                id(layer.tensors),
                layer.tensors.mutation_count,
                layer.layer_index,
                layer.kind.value,
                tuple(
                    (name, _runtime_tensor_fingerprint(tensor))
                    for name, tensor in sorted(layer.tensors.items())
                ),
            )
            for layer in self.layers
        )

    def assert_unchanged(self) -> None:
        if self.runtime_fingerprint() != self._baseline_fingerprint:
            _fail(
                "CACHE_DENOISE_MUTATION",
                "read-only denoise changed scratch bindings or tensor bytes",
            )


@dataclass(frozen=True)
class DenoiseReadView:
    episode_id: str
    episode_epoch: int
    revision: int
    next_chunk_id: int
    layout_instance_digest: str
    state_manifest_digest: str
    _state_identity: int = field(repr=False)
    _state_snapshot: HybridTemporalState = field(repr=False, compare=False)

    def export_scratch(self) -> CacheScratch:
        # Export from a detached deep snapshot, never from the live manager
        # authority object.  The original live identity remains bound solely
        # as an opaque CAS token checked by ``finish_denoise``.
        return CacheScratch.from_state(
            self._state_snapshot,
            _source_state_identity=self._state_identity,
        )


@dataclass(frozen=True)
class TeacherForcingDatasetPairProof:
    dataset_manifest_sha256: str
    dataset_episode_id: str
    dataset_row_identity: str
    dataset_row_start: int
    dataset_row_end_exclusive: int
    layout_spec_sha256: str
    layout_instance_digest: str
    video_tensor_digest: str
    frame_valid_mask_digest: str
    action_tensor_digest: str
    action_mask_digest: str
    row_order_manifest_sha256: str
    schema: str = _TEACHER_PROOF_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _TEACHER_PROOF_SCHEMA:
            _fail("COMMIT_SOURCE_PROOF_MIXED", "invalid teacher proof schema")
        for name in (
            "dataset_manifest_sha256",
            "layout_spec_sha256",
            "layout_instance_digest",
            "video_tensor_digest",
            "frame_valid_mask_digest",
            "action_tensor_digest",
            "action_mask_digest",
            "row_order_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_non_empty_string(self.dataset_episode_id, "dataset_episode_id")
        _require_non_empty_string(self.dataset_row_identity, "dataset_row_identity")
        _require_plain_uint(self.dataset_row_start, "dataset_row_start")
        _require_plain_uint(
            self.dataset_row_end_exclusive, "dataset_row_end_exclusive"
        )
        if self.dataset_row_start != 0:
            _fail(
                "NONZERO_EPISODE_ROW_START",
                "CACH teacher-forcing rows must start at episode origin",
            )
        if self.dataset_row_end_exclusive <= self.dataset_row_start:
            _fail(
                "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
                "dataset row interval must be non-empty",
            )

    def to_manifest(self) -> dict:
        return {
            "action_mask_digest": self.action_mask_digest,
            "action_tensor_digest": self.action_tensor_digest,
            "dataset_episode_id": self.dataset_episode_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_row_end_exclusive": self.dataset_row_end_exclusive,
            "dataset_row_identity": self.dataset_row_identity,
            "dataset_row_start": self.dataset_row_start,
            "frame_valid_mask_digest": self.frame_valid_mask_digest,
            "layout_instance_digest": self.layout_instance_digest,
            "layout_spec_sha256": self.layout_spec_sha256,
            "row_order_manifest_sha256": self.row_order_manifest_sha256,
            "schema": self.schema,
            "video_tensor_digest": self.video_tensor_digest,
        }

    @property
    def source_proof_digest(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_manifest()))


@dataclass(frozen=True)
class TeacherForcingCommitRequest:
    commit_id: str
    transaction_nonce: str
    expected_episode_id: str
    expected_episode_epoch: int
    expected_revision: int
    content_time: ContentTime
    video: Tensor = field(repr=False, compare=False)
    frame_valid_mask: Tensor = field(repr=False, compare=False)
    actions: Tensor = field(repr=False, compare=False)
    action_valid_mask: Tensor = field(repr=False, compare=False)
    proof: TeacherForcingDatasetPairProof

    def __post_init__(self) -> None:
        _require_non_empty_string(self.commit_id, "commit_id")
        _require_non_empty_string(self.transaction_nonce, "transaction_nonce")
        _require_non_empty_string(self.expected_episode_id, "expected_episode_id")
        _require_plain_uint(self.expected_episode_epoch, "expected_episode_epoch")
        _require_plain_uint(self.expected_revision, "expected_revision")
        if not isinstance(self.content_time, ContentTime):
            _fail("CACHE_CONTENT_TIME_MISMATCH", "content_time has wrong type")
        if not isinstance(self.proof, TeacherForcingDatasetPairProof):
            _fail("COMMIT_SOURCE_PROOF_MISSING", "teacher proof is required")


@dataclass(frozen=True)
class StagedLayerPayload:
    """Typed output of the not-yet-implemented low-level staging adapter."""

    layer_index: int
    kind: LayerKind
    tensors: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        _require_plain_uint(self.layer_index, "layer_index")
        if not isinstance(self.kind, LayerKind):
            _fail("CACHE_LAYER_KIND_MISMATCH", "payload kind must be LayerKind")
        if not isinstance(self.tensors, Mapping):
            _fail("CACHE_SCHEMA_MISMATCH", "payload tensors must be a mapping")
        copied = dict(self.tensors)
        object.__setattr__(self, "tensors", MappingProxyType(copied))


@dataclass(frozen=True)
class PairedStagingContext:
    """Inputs supplied exactly once to a caller-owned paired ``t=0`` callback."""

    commit_source: CommitSource
    content_time: ContentTime
    previous_state_manifest_digest: str
    previous_scratch: CacheScratch
    video: Tensor = field(repr=False, compare=False)
    frame_valid_mask: Tensor = field(repr=False, compare=False)
    frame_valid_mask_digest: str
    actions: Tensor = field(repr=False, compare=False)
    action_valid_mask: Tensor = field(repr=False, compare=False)
    video_timestep: int = 0
    action_timestep: int = 0

    def __post_init__(self) -> None:
        _require_sha256(
            self.frame_valid_mask_digest,
            "frame_valid_mask_digest",
        )
        if tensor_digest(self.frame_valid_mask) != self.frame_valid_mask_digest:
            _fail(
                "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
                "staging frame mask differs from its bound digest",
            )


@dataclass(frozen=True)
class ReceiptPublication:
    """Publisher attestation consumed before the live pointer may advance."""

    receipt_id: str
    sha256: str
    durable: bool
    readback_verified: bool

    def __post_init__(self) -> None:
        _require_non_empty_string(self.receipt_id, "receipt_id")
        _require_sha256(self.sha256, "receipt publication sha256")
        if type(self.durable) is not bool or type(self.readback_verified) is not bool:
            _fail("COMMIT_RECEIPT_INVALID", "publication flags must be bool")


class DurableReceiptPublisher(Protocol):
    """External durability boundary.

    Production implementations must exclusive-create, fsync, stable-read, and
    verify the receipt.  The cache manager refuses to publish state unless the
    returned attestation says both durability and read-back verification passed.
    """

    def publish_exclusive(
        self, *, receipt_id: str, payload: bytes, expected_sha256: str
    ) -> ReceiptPublication:
        ...


@dataclass(frozen=True)
class PairedCommitReceipt:
    commit_id: str
    transaction_nonce: str
    episode_id: str
    episode_epoch: int
    revision_before: int
    revision_after: int
    chunk_id: int
    layout_instance_digest: str
    source_proof_digest: str
    pair_video_digest: str
    pair_frame_valid_mask_digest: str
    pair_action_digest: str
    pair_action_mask_digest: str
    paired_payload_digest: str
    dataset_manifest_sha256: str
    dataset_row_identity: str
    state_manifest_before: str
    staged_state_manifest: str
    state_manifest_after: str
    committed_at_monotonic_ns: int
    commit_source: CommitSource = CommitSource.TEACHER_FORCING
    source_proof_schema: str = _TEACHER_PROOF_SCHEMA
    result: str = "committed"
    schema: str = _COMMIT_RECEIPT_SCHEMA

    def to_manifest(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "commit_id": self.commit_id,
            "commit_source": self.commit_source.value,
            "committed_at_monotonic_ns": self.committed_at_monotonic_ns,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_row_identity": self.dataset_row_identity,
            "episode_epoch": self.episode_epoch,
            "episode_id": self.episode_id,
            "layout_instance_digest": self.layout_instance_digest,
            "pair_frame_valid_mask_digest": self.pair_frame_valid_mask_digest,
            "pair_action_digest": self.pair_action_digest,
            "pair_action_mask_digest": self.pair_action_mask_digest,
            "pair_video_digest": self.pair_video_digest,
            "paired_payload_digest": self.paired_payload_digest,
            "result": self.result,
            "revision_after": self.revision_after,
            "revision_before": self.revision_before,
            "schema": self.schema,
            "source_proof_digest": self.source_proof_digest,
            "source_proof_schema": self.source_proof_schema,
            "staged_state_manifest": self.staged_state_manifest,
            "state_manifest_after": self.state_manifest_after,
            "state_manifest_before": self.state_manifest_before,
            "transaction_nonce": self.transaction_nonce,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_manifest())

    @property
    def receipt_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes)


@dataclass(frozen=True)
class CacheResetReceipt:
    reset_id: str
    transaction_nonce: str
    previous_episode_id: str
    previous_episode_epoch: int
    new_episode_id: str
    new_episode_epoch: int
    layout_spec_sha256: str
    layout_instance_digest: str
    state_manifest_before: str
    state_manifest_after: str
    aborted_pending_commit_id: str | None
    reset_at_monotonic_ns: int
    schema: str = _RESET_RECEIPT_SCHEMA

    def to_manifest(self) -> dict:
        return {
            "aborted_pending_commit_id": self.aborted_pending_commit_id,
            "new_episode_epoch": self.new_episode_epoch,
            "new_episode_id": self.new_episode_id,
            "layout_instance_digest": self.layout_instance_digest,
            "layout_spec_sha256": self.layout_spec_sha256,
            "previous_episode_epoch": self.previous_episode_epoch,
            "previous_episode_id": self.previous_episode_id,
            "reset_at_monotonic_ns": self.reset_at_monotonic_ns,
            "reset_id": self.reset_id,
            "schema": self.schema,
            "state_manifest_after": self.state_manifest_after,
            "state_manifest_before": self.state_manifest_before,
            "transaction_nonce": self.transaction_nonce,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_manifest())

    @property
    def receipt_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes)


@dataclass(frozen=True)
class _PreparedTeacherPair:
    video: Tensor
    frame_valid_mask: Tensor
    actions: Tensor
    action_valid_mask: Tensor
    video_digest: str
    frame_valid_mask_digest: str
    action_digest: str
    action_mask_digest: str
    source_proof_digest: str
    paired_payload_digest: str


def _prepare_teacher_pair(
    request: TeacherForcingCommitRequest,
    *,
    chunk: ChunkLayout,
) -> _PreparedTeacherPair:
    if not isinstance(chunk, ChunkLayout):
        _fail("CACHE_CONTENT_TIME_MISMATCH", "bound chunk has wrong type")
    for name, tensor in (
        ("video", request.video),
        ("frame_valid_mask", request.frame_valid_mask),
        ("actions", request.actions),
        ("action_valid_mask", request.action_valid_mask),
    ):
        if not isinstance(tensor, Tensor):
            _fail("COMMIT_SOURCE_PROOF_MISSING", f"{name} tensor is required")
    video = request.video.detach().clone(memory_format=torch.contiguous_format)
    frame_mask = request.frame_valid_mask.detach().clone(
        memory_format=torch.contiguous_format
    )
    actions = request.actions.detach().clone(memory_format=torch.contiguous_format)
    action_mask = request.action_valid_mask.detach().clone(
        memory_format=torch.contiguous_format
    )
    for name, tensor in (
        ("video", video),
        ("frame_valid_mask", frame_mask),
        ("actions", actions),
        ("action_valid_mask", action_mask),
    ):
        _validate_finite_tensor(tensor, name)
    if video.ndim != 5:
        _fail(
            "COMMIT_MISSING_VIDEO_PAIR",
            "video must have exact shape [batch, channels, latents, height, width]",
        )
    if video.shape[0] <= 0:
        _fail("COMMIT_MISSING_VIDEO_PAIR", "video batch must be non-empty")
    if video.shape[2] != len(chunk.latent_valid_mask):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "video must contain the bound layout's fixed K latent slots",
        )
    if frame_mask.dtype is not torch.bool or frame_mask.ndim != 2:
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "frame_valid_mask must be boolean [batch, fixed_K]",
        )
    expected_frame_shape = (video.shape[0], len(chunk.latent_valid_mask))
    if tuple(frame_mask.shape) != expected_frame_shape:
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "frame_valid_mask shape does not match fixed-K video",
        )
    if actions.ndim != 3 or actions.shape[-1] != 20:
        _fail(
            "COMMIT_MISSING_ACTION_PAIR",
            "actions must have exact shape [batch, tokens, 20]",
        )
    if actions.shape[1] != chunk.action_slot_capacity:
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "actions must contain the bound layout's fixed action-slot capacity",
        )
    if action_mask.dtype is not torch.bool or action_mask.ndim != 2:
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "action_valid_mask must be boolean [batch, tokens]",
        )
    if tuple(action_mask.shape) != tuple(actions.shape[:2]):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "action mask shape does not match action tokens",
        )
    if (
        video.shape[0] != actions.shape[0]
        or frame_mask.shape[0] != actions.shape[0]
    ):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "video/frame/action batch dimensions differ",
        )
    if (
        not video.is_floating_point()
        or not actions.is_floating_point()
        or video.dtype != actions.dtype
        or video.device != actions.device
    ):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "video/actions must be same-device, same-dtype floating tensors",
        )
    if (
        frame_mask.device != video.device
        or action_mask.device != actions.device
    ):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "frame/action masks must share their paired tensor device",
        )
    expected_frame_mask = torch.tensor(
        chunk.latent_valid_mask,
        dtype=torch.bool,
        device=frame_mask.device,
    ).view(1, -1).expand(frame_mask.shape[0], -1)
    if not torch.equal(frame_mask, expected_frame_mask):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "frame_valid_mask differs from the exact bound layout prefix mask",
        )
    expected_action_mask = torch.tensor(
        chunk.action_valid_mask,
        dtype=torch.bool,
        device=action_mask.device,
    ).view(1, -1).expand(action_mask.shape[0], -1)
    if not torch.equal(action_mask, expected_action_mask):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "action_valid_mask differs from the exact bound layout prefix mask",
        )
    invalid_video_values = video.masked_select(
        (~frame_mask)[:, None, :, None, None].expand_as(video)
    )
    if invalid_video_values.numel() and not bool(
        (invalid_video_values == 0).all()
    ):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "video padding slots must be exact zero",
        )
    invalid_action_values = actions.masked_select(
        (~action_mask)[:, :, None].expand_as(actions)
    )
    if invalid_action_values.numel() and not bool(
        (invalid_action_values == 0).all()
    ):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "action padding slots must be exact zero",
        )
    if (
        request.content_time.latent_end_exclusive
        - request.content_time.latent_start
        != chunk.valid_latent_count
        or request.content_time.action_token_end_exclusive
        - request.content_time.action_token_start
        != chunk.valid_action_count
    ):
        _fail(
            "CACHE_CONTENT_TIME_MISMATCH",
            "content-time valid intervals differ from the bound chunk",
        )
    valid_counts = action_mask.long().sum(dim=1)
    expected_valid = chunk.valid_action_count
    if not bool((valid_counts == expected_valid).all()):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "valid action count does not match the bound layout",
        )
    frame_valid_counts = frame_mask.long().sum(dim=1)
    if not bool((frame_valid_counts == chunk.valid_latent_count).all()):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "valid frame count does not match the bound layout",
        )

    video_digest = tensor_digest(video)
    frame_mask_digest = tensor_digest(frame_mask)
    action_digest = tensor_digest(actions)
    action_mask_digest = tensor_digest(action_mask)
    proof = request.proof
    if (
        proof.video_tensor_digest != video_digest
        or proof.frame_valid_mask_digest != frame_mask_digest
        or proof.action_tensor_digest != action_digest
        or proof.action_mask_digest != action_mask_digest
    ):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "teacher tensor digest does not match proof",
        )
    ct = request.content_time
    if (
        proof.dataset_episode_id != ct.episode_id
        or proof.layout_spec_sha256 != ct.layout_spec_sha256
        or proof.layout_instance_digest != ct.layout_instance_digest
        or proof.dataset_row_end_exclusive < ct.raw_observation_end_exclusive
    ):
        _fail(
            "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
            "teacher row/layout identity does not match content-time",
        )
    source_proof_digest = proof.source_proof_digest
    paired_payload_digest = _sha256_bytes(
        _canonical_json_bytes(
            {
                "action_digest": action_digest,
                "action_mask_digest": action_mask_digest,
                "commit_id": request.commit_id,
                "commit_source": CommitSource.TEACHER_FORCING.value,
                "content_time": ct.to_manifest(),
                "frame_valid_mask_digest": frame_mask_digest,
                "source_proof_digest": source_proof_digest,
                "transaction_nonce": request.transaction_nonce,
                "video_digest": video_digest,
            }
        )
    )
    return _PreparedTeacherPair(
        video=video,
        frame_valid_mask=frame_mask,
        actions=actions,
        action_valid_mask=action_mask,
        video_digest=video_digest,
        frame_valid_mask_digest=frame_mask_digest,
        action_digest=action_digest,
        action_mask_digest=action_mask_digest,
        source_proof_digest=source_proof_digest,
        paired_payload_digest=paired_payload_digest,
    )


def _materialize_layer_states(
    *,
    registry: LayerRegistry,
    content_time: ContentTime,
    payloads: Sequence[StagedLayerPayload],
) -> tuple[LayerTemporalState, ...]:
    if len(payloads) != len(registry.layers):
        _fail(
            "CACHE_SCHEMA_MISMATCH",
            "staging callback must return exactly one payload per layer",
        )
    by_index: dict[int, StagedLayerPayload] = {}
    for payload in payloads:
        if not isinstance(payload, StagedLayerPayload):
            _fail("CACHE_SCHEMA_MISMATCH", "invalid staged layer payload")
        if payload.layer_index in by_index:
            _fail("CACHE_SCHEMA_MISMATCH", "duplicate staged layer payload")
        by_index[payload.layer_index] = payload

    states = []
    for spec in registry.layers:
        payload = by_index.get(spec.layer_index)
        if payload is None:
            _fail("CACHE_SCHEMA_MISMATCH", "missing staged layer payload")
        if payload.kind is not spec.kind:
            _fail(
                "CACHE_LAYER_KIND_MISMATCH",
                f"layer {spec.layer_index} payload kind differs from registry",
            )
        observed = frozenset(payload.tensors)
        if observed != spec.required_tensor_fields:
            if any("attnres" in name.casefold() for name in observed):
                _fail(
                    "CACHE_ATTNRES_PERSISTED",
                    "AttnRes tensor appeared in staged temporal state",
                )
            _fail(
                "CACHE_SCHEMA_MISMATCH",
                f"layer {spec.layer_index} tensor fields do not match registry",
            )
        snapshots = []
        for name in sorted(payload.tensors):
            snapshots.append(
                (
                    name,
                    TensorSnapshot.from_tensor(
                        (
                            f"layer/{spec.layer_index}/{name}/"
                            f"episode/{content_time.episode_epoch}/"
                            f"chunk/{content_time.chunk_id}"
                        ),
                        payload.tensors[name],
                    ),
                )
            )

        tensor_map = dict(payload.tensors)
        if spec.kind is LayerKind.GDN_FULL_HISTORY:
            if tensor_map["main_s_kv"].ndim != 4 or tensor_map["main_s_z"].ndim != 4:
                _fail(
                    "CACHE_SCHEMA_MISMATCH",
                    "GDN S_kv/S_z tensors must both be rank four",
                )
        else:
            k = tensor_map["main_k_post_rope"]
            v = tensor_map["main_v"]
            if k.ndim != 4 or v.ndim != 4 or tuple(k.shape) != tuple(v.shape):
                _fail(
                    "CACHE_SCHEMA_MISMATCH",
                    "softmax K/V must be equal-shape rank-four tensors",
                )
            assert spec.tokens_per_latent is not None
            expected_tokens = (
                content_time.latent_end_exclusive - content_time.latent_start
            ) * spec.tokens_per_latent
            if k.shape[2] != expected_tokens:
                _fail(
                    "CACHE_SOFTMAX_HISTORY_GT_ONE",
                    "softmax state token count is not exactly the committed chunk",
                )
            if spec.camera_enabled:
                ck = tensor_map["camera_k_post_ucpe"]
                cv = tensor_map["camera_v_post_ucpe"]
                if ck.ndim != 4 or cv.ndim != 4 or tuple(ck.shape) != tuple(cv.shape):
                    _fail(
                        "CACHE_SCHEMA_MISMATCH",
                        "camera softmax K/V shape mismatch",
                    )
                if ck.shape[2] != expected_tokens:
                    _fail(
                        "CACHE_SOFTMAX_HISTORY_GT_ONE",
                        "camera softmax state exceeds one committed chunk",
                    )
        states.append(
            LayerTemporalState(
                layer_index=spec.layer_index,
                kind=spec.kind,
                through=content_time,
                tensors=tuple(snapshots),
            )
        )
    return tuple(states)


def _clone_state_for_inspection(
    state: HybridTemporalState,
) -> HybridTemporalState:
    """Return a deep tensor clone so callers never receive the live authority."""

    layers = []
    for layer in state.layer_states:
        layers.append(
            LayerTemporalState(
                layer_index=layer.layer_index,
                kind=layer.kind,
                through=layer.through,
                tensors=tuple(
                    (
                        name,
                        TensorSnapshot.from_tensor(
                            snapshot.logical_tensor_id,
                            snapshot.clone_tensor(),
                        ),
                    )
                    for name, snapshot in layer.tensors
                ),
            )
        )
    return HybridTemporalState(
        episode_id=state.episode_id,
        episode_epoch=state.episode_epoch,
        revision=state.revision,
        committed_through_chunk=state.committed_through_chunk,
        next_chunk_id=state.next_chunk_id,
        action_cursor=state.action_cursor,
        layout_spec_sha256=state.layout_spec_sha256,
        layout_instance_digest=state.layout_instance_digest,
        layer_registry_digest=state.layer_registry_digest,
        layer_states=tuple(layers),
        committed_pair_digest=state.committed_pair_digest,
        last_commit_source=state.last_commit_source,
        last_source_proof_digest=state.last_source_proof_digest,
        pair_evidence_mode=state.pair_evidence_mode,
    )


class HybridCacheManager:
    """Own one atomic :class:`HybridTemporalState` pointer.

    Staging happens outside the publication lock against transaction-local
    clones.  The lock is reacquired for CAS, durable receipt publication, and
    the single pointer swap.  A receipt publisher is mandatory; no in-memory
    fallback is silently treated as durable.
    """

    def __init__(
        self,
        *,
        registry: LayerRegistry,
        episode_id: str,
        layout: ChunkActionLayout,
        receipt_publisher: DurableReceiptPublisher,
        episode_epoch: int = 0,
        _synthetic_test_marker: str | None = None,
    ):
        if not isinstance(registry, LayerRegistry):
            _fail("CACHE_SCHEMA_MISMATCH", "registry must be a LayerRegistry")
        if not isinstance(layout, ChunkActionLayout):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "manager requires an immutable ChunkActionLayout",
            )
        if _synthetic_test_marker is None:
            if layout.synthetic_test_only:
                _fail(
                    "CACHE_SYNTHETIC_LAYOUT_PROVENANCE",
                    "normal cache manager rejects synthetic layout provenance",
                )
            validate_cach_a_registry(registry)
            synthetic_test_mode = False
        elif _synthetic_test_marker != _SYNTHETIC_CACHE_TEST_MARKER:
            _fail(
                "CACHE_SCHEMA_MISMATCH",
                "invalid synthetic cache test marker",
            )
        elif not layout.synthetic_test_only:
            _fail(
                "CACHE_SYNTHETIC_LAYOUT_PROVENANCE",
                "synthetic cache manager requires synthetic layout provenance",
            )
        else:
            synthetic_test_mode = True
        if receipt_publisher is None or not hasattr(
            receipt_publisher, "publish_exclusive"
        ):
            _fail(
                "COMMIT_RECEIPT_INVALID",
                "an explicit durable receipt publisher is required",
            )
        self._registry = registry
        self._layout = layout
        self._synthetic_test_mode = synthetic_test_mode
        self._publisher = receipt_publisher
        self._lock = threading.RLock()
        self._state = HybridTemporalState.empty(
            episode_id=episode_id,
            episode_epoch=episode_epoch,
            layout_spec_sha256=layout.layout_spec_sha256,
            layout_instance_digest=layout.layout_instance_digest,
            registry=registry,
        )
        self._state_manifest_digest = self._state.state_manifest_digest
        self._pending: tuple[str, str, object] | None = None
        self._completed: dict[str, tuple[str, PairedCommitReceipt]] = {}
        self._completed_resets: dict[str, tuple[str, CacheResetReceipt]] = {}
        self._poisoned = False

    @classmethod
    def for_synthetic_tests(
        cls,
        *,
        registry: LayerRegistry,
        episode_id: str,
        layout: ChunkActionLayout,
        receipt_publisher: DurableReceiptPublisher,
        episode_epoch: int = 0,
    ) -> "HybridCacheManager":
        """Construct a visibly test-only manager with a reduced fake registry."""

        return cls(
            registry=registry,
            episode_id=episode_id,
            layout=layout,
            receipt_publisher=receipt_publisher,
            episode_epoch=episode_epoch,
            _synthetic_test_marker=_SYNTHETIC_CACHE_TEST_MARKER,
        )

    @property
    def registry(self) -> LayerRegistry:
        return self._registry

    @property
    def state(self) -> HybridTemporalState:
        with self._lock:
            self._assert_not_poisoned()
            self._assert_live_integrity()
            return _clone_state_for_inspection(self._state)

    def _assert_not_poisoned(self) -> None:
        if self._poisoned:
            _fail(
                "CACHE_PUBLICATION_POISONED",
                "cache publication failed after a durable decision",
            )

    def _assert_live_integrity(self) -> None:
        if self._state.state_manifest_digest != self._state_manifest_digest:
            self._poisoned = True
            _fail(
                "CACHE_DENOISE_MUTATION",
                "live temporal state changed outside an atomic publication",
            )

    def snapshot_for_denoise(
        self,
        *,
        expected_episode_id: str,
        expected_episode_epoch: int,
        expected_revision: int,
        expected_next_chunk_id: int,
    ) -> DenoiseReadView:
        with self._lock:
            self._assert_not_poisoned()
            self._assert_live_integrity()
            state = self._state
            if (
                state.episode_id != expected_episode_id
                or state.episode_epoch != expected_episode_epoch
                or state.revision != expected_revision
                or state.next_chunk_id != expected_next_chunk_id
            ):
                _fail(
                    "CACHE_CONTENT_TIME_MISMATCH",
                    "denoise request does not match live episode/revision/chunk",
                )
            if state.next_chunk_id >= len(self._layout.chunks):
                _fail(
                    "COMMIT_OUT_OF_ORDER",
                    "bound layout has no next chunk to denoise",
                )
            if (
                state.layout_spec_sha256 != self._layout.layout_spec_sha256
                or state.layout_instance_digest
                != self._layout.layout_instance_digest
            ):
                _fail(
                    "CACHE_CONTENT_TIME_MISMATCH",
                    "live state differs from the manager-bound layout",
                )
            if state.revision > 0:
                through = state.layer_states[0].through
                assert through is not None
                if (
                    self._layout.layout_instance_digest
                    != through.layout_instance_digest
                ):
                    _fail(
                        "CACHE_CONTENT_TIME_MISMATCH",
                        "denoise layout instance differs from committed cache",
                    )
            return DenoiseReadView(
                episode_id=state.episode_id,
                episode_epoch=state.episode_epoch,
                revision=state.revision,
                next_chunk_id=state.next_chunk_id,
                layout_instance_digest=self._layout.layout_instance_digest,
                state_manifest_digest=state.state_manifest_digest,
                _state_identity=id(state),
                _state_snapshot=_clone_state_for_inspection(state),
            )

    def finish_denoise(
        self, *, read_view: DenoiseReadView, scratch: CacheScratch
    ) -> None:
        """Validate both transaction-local scratch and the live pointer."""

        if not isinstance(read_view, DenoiseReadView) or not isinstance(
            scratch, CacheScratch
        ):
            _fail("CACHE_SCHEMA_MISMATCH", "invalid denoise view/scratch")
        scratch.assert_unchanged()
        if (
            scratch.source_state_manifest_digest != read_view.state_manifest_digest
            or scratch._source_state_identity != read_view._state_identity
        ):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "denoise scratch was not exported from this read view",
            )
        with self._lock:
            self._assert_not_poisoned()
            self._assert_live_integrity()
            if (
                id(self._state) != read_view._state_identity
                or self._state.revision != read_view.revision
                or self._state.episode_epoch != read_view.episode_epoch
            ):
                _fail(
                    "COMMIT_STALE_REVISION",
                    "denoise read view became stale before completion",
                )
            if self._state.state_manifest_digest != read_view.state_manifest_digest:
                _fail(
                    "CACHE_DENOISE_MUTATION",
                    "live state manifest changed during denoise",
                )

    def commit_paired(
        self,
        *,
        commit_source: CommitSource | str,
        request: TeacherForcingCommitRequest | None,
        stage_callback: Callable[
            [PairedStagingContext], Sequence[StagedLayerPayload]
        ]
        | None,
    ) -> PairedCommitReceipt:
        """Commit one pair, or reject non-commissioned sources before staging."""

        try:
            source = (
                commit_source
                if isinstance(commit_source, CommitSource)
                else CommitSource(commit_source)
            )
        except (TypeError, ValueError):
            _fail("COMMIT_SOURCE_UNKNOWN", "unknown paired commit source")
        if source is CommitSource.SELF_FORCING:
            _fail(
                "COMMIT_SELF_FORCING_OBJECTIVE_UNCLOSED",
                "self-forcing commits are disabled until the objective is closed",
            )
        if source is CommitSource.DEPLOY_APPLIED_ACK:
            _fail(
                "COMMIT_DEPLOY_ACK_MISSING",
                "deploy commits are disabled until APPLIED_ACTION_ACK is implemented",
            )
        if not isinstance(request, TeacherForcingCommitRequest):
            _fail(
                "COMMIT_SOURCE_PROOF_MISSING",
                "teacher-forcing request and proof are required",
            )
        if not callable(stage_callback):
            _fail(
                "COMMIT_STAGING_CALLBACK_MISSING",
                "paired t=0 staging callback is required",
            )
        return self._commit_teacher(request=request, stage_callback=stage_callback)

    def _validate_next_content_time(
        self, state: HybridTemporalState, content_time: ContentTime
    ) -> None:
        if state.next_chunk_id >= len(self._layout.chunks):
            _fail(
                "COMMIT_OUT_OF_ORDER",
                "commit exceeds the bound layout chunk count",
            )
        if (
            content_time.episode_id != state.episode_id
            or content_time.episode_epoch != state.episode_epoch
            or content_time.layout_spec_sha256 != state.layout_spec_sha256
            or content_time.layout_instance_digest
            != state.layout_instance_digest
            or state.layout_spec_sha256 != self._layout.layout_spec_sha256
            or state.layout_instance_digest
            != self._layout.layout_instance_digest
        ):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "commit content-time does not match live episode/spec",
            )
        if content_time.chunk_id != state.next_chunk_id:
            _fail("COMMIT_OUT_OF_ORDER", "commit chunk is not live next_chunk_id")
        if content_time.action_token_start != state.action_cursor:
            _fail("COMMIT_OUT_OF_ORDER", "commit action cursor is not contiguous")
        bound_chunk = self._layout.chunks[state.next_chunk_id]
        content_time.verify_layout_binding(
            layout=self._layout,
            chunk=bound_chunk,
        )
        if state.revision == 0:
            if (
                content_time.latent_start != 0
                or content_time.raw_observation_start != 0
            ):
                _fail(
                    "COMMIT_OUT_OF_ORDER",
                    "first commit must begin at latent/raw index zero",
                )
        else:
            previous = state.layer_states[0].through
            assert previous is not None
            if content_time.latent_start != previous.latent_end_exclusive:
                _fail("COMMIT_OUT_OF_ORDER", "latent intervals are not contiguous")
            if (
                content_time.raw_observation_start
                != previous.raw_observation_end_exclusive
            ):
                _fail(
                    "COMMIT_OUT_OF_ORDER",
                    "raw observation intervals are not contiguous",
                )
            if (
                content_time.action_rope_start
                != previous.action_rope_end_exclusive
            ):
                _fail(
                    "COMMIT_OUT_OF_ORDER",
                    "Action RoPE intervals are not contiguous",
                )
            if (
                content_time.layout_instance_digest
                != previous.layout_instance_digest
            ):
                _fail(
                    "CACHE_CONTENT_TIME_MISMATCH",
                    "layout instance changed within one episode",
                )

    def _commit_teacher(
        self,
        *,
        request: TeacherForcingCommitRequest,
        stage_callback: Callable[
            [PairedStagingContext], Sequence[StagedLayerPayload]
        ],
    ) -> PairedCommitReceipt:
        with self._lock:
            self._assert_not_poisoned()
            self._assert_live_integrity()
            live_state = self._state
            if (
                request.commit_id not in self._completed
                and (
                    request.expected_episode_id != live_state.episode_id
                    or request.expected_episode_epoch != live_state.episode_epoch
                    or request.expected_revision != live_state.revision
                )
            ):
                _fail(
                    "COMMIT_STALE_REVISION",
                    "teacher request expected state is stale",
                )
            layout_snapshot = self._layout
            content_time = request.content_time
            if (
                content_time.layout_spec_sha256
                != layout_snapshot.layout_spec_sha256
                or content_time.layout_instance_digest
                != layout_snapshot.layout_instance_digest
                or content_time.chunk_id >= len(layout_snapshot.chunks)
            ):
                _fail(
                    "CACHE_CONTENT_TIME_MISMATCH",
                    "teacher pair does not name a chunk in the manager-bound layout",
                )
            bound_chunk = layout_snapshot.chunks[content_time.chunk_id]
            content_time.verify_layout_binding(
                layout=layout_snapshot,
                chunk=bound_chunk,
            )
        prepared = _prepare_teacher_pair(request, chunk=bound_chunk)
        pending_token = object()
        base_state: HybridTemporalState

        with self._lock:
            self._assert_not_poisoned()
            self._assert_live_integrity()
            completed = self._completed.get(request.commit_id)
            if completed is not None:
                prior_digest, prior_receipt = completed
                if prior_digest != prepared.paired_payload_digest:
                    _fail(
                        "COMMIT_DUPLICATE_CONFLICT",
                        "commit_id was already used for a different payload",
                    )
                live_state = self._state
                if (
                    live_state.episode_id != prior_receipt.episode_id
                    or live_state.episode_epoch != prior_receipt.episode_epoch
                    or live_state.revision < prior_receipt.revision_after
                ):
                    _fail(
                        "COMMIT_STALE_REVISION",
                        "completed commit belongs to a stale episode epoch",
                    )
                return prior_receipt
            if self._pending is not None:
                pending_id, pending_digest, _ = self._pending
                if (
                    pending_id == request.commit_id
                    and pending_digest != prepared.paired_payload_digest
                ):
                    _fail(
                        "COMMIT_DUPLICATE_CONFLICT",
                        "pending commit_id has a different payload",
                    )
                _fail(
                    "COMMIT_DUPLICATE_PENDING",
                    "another paired transaction is already pending",
                )
            base_state = self._state
            if (
                request.expected_episode_id != base_state.episode_id
                or request.expected_episode_epoch != base_state.episode_epoch
                or request.expected_revision != base_state.revision
            ):
                _fail(
                    "COMMIT_STALE_REVISION",
                    "teacher request expected state is stale",
                )
            self._validate_next_content_time(base_state, request.content_time)
            self._pending = (
                request.commit_id,
                prepared.paired_payload_digest,
                pending_token,
            )

        try:
            context = PairedStagingContext(
                commit_source=CommitSource.TEACHER_FORCING,
                content_time=request.content_time,
                previous_state_manifest_digest=base_state.state_manifest_digest,
                previous_scratch=CacheScratch.from_state(base_state),
                video=prepared.video.detach().clone(),
                frame_valid_mask=prepared.frame_valid_mask.detach().clone(),
                frame_valid_mask_digest=prepared.frame_valid_mask_digest,
                actions=prepared.actions.detach().clone(),
                action_valid_mask=prepared.action_valid_mask.detach().clone(),
            )
            input_fingerprint = (
                _runtime_tensor_fingerprint(context.video),
                _runtime_tensor_fingerprint(context.frame_valid_mask),
                _runtime_tensor_fingerprint(context.actions),
                _runtime_tensor_fingerprint(context.action_valid_mask),
            )
            raw_payloads = stage_callback(context)
            context.previous_scratch.assert_unchanged()
            if input_fingerprint != (
                _runtime_tensor_fingerprint(context.video),
                _runtime_tensor_fingerprint(context.frame_valid_mask),
                _runtime_tensor_fingerprint(context.actions),
                _runtime_tensor_fingerprint(context.action_valid_mask),
            ):
                _fail(
                    "COMMIT_STAGING_INPUT_MUTATION",
                    "paired staging callback mutated video/frame/action inputs",
                )
            payloads = tuple(raw_payloads)
            layer_states = _materialize_layer_states(
                registry=self._registry,
                content_time=request.content_time,
                payloads=payloads,
            )
            staged_state = HybridTemporalState(
                episode_id=base_state.episode_id,
                episode_epoch=base_state.episode_epoch,
                revision=base_state.revision + 1,
                committed_through_chunk=request.content_time.chunk_id,
                next_chunk_id=request.content_time.chunk_id + 1,
                action_cursor=request.content_time.action_token_end_exclusive,
                layout_spec_sha256=base_state.layout_spec_sha256,
                layout_instance_digest=base_state.layout_instance_digest,
                layer_registry_digest=self._registry.manifest_digest,
                layer_states=layer_states,
                committed_pair_digest=prepared.paired_payload_digest,
                last_commit_source=CommitSource.TEACHER_FORCING,
                last_source_proof_digest=prepared.source_proof_digest,
                pair_evidence_mode="dataset_ground_truth",
            )
        except CacheContractError:
            with self._lock:
                if self._pending is not None and self._pending[2] is pending_token:
                    self._pending = None
            raise
        except Exception as exc:
            with self._lock:
                if self._pending is not None and self._pending[2] is pending_token:
                    self._pending = None
            raise CacheContractError(
                "COMMIT_STAGING_FAILED",
                f"paired staging callback failed: {type(exc).__name__}",
            ) from exc

        with self._lock:
            self._assert_not_poisoned()
            self._assert_live_integrity()
            if (
                self._pending is None
                or self._pending[2] is not pending_token
                or self._state is not base_state
                or self._state.episode_epoch != request.expected_episode_epoch
                or self._state.revision != request.expected_revision
            ):
                if self._pending is not None and self._pending[2] is pending_token:
                    self._pending = None
                _fail(
                    "COMMIT_STALE_REVISION",
                    "live state changed before commit CAS",
                )
            before_digest = base_state.state_manifest_digest
            after_digest = staged_state.state_manifest_digest
            receipt = PairedCommitReceipt(
                commit_id=request.commit_id,
                transaction_nonce=request.transaction_nonce,
                episode_id=base_state.episode_id,
                episode_epoch=base_state.episode_epoch,
                revision_before=base_state.revision,
                revision_after=staged_state.revision,
                chunk_id=request.content_time.chunk_id,
                layout_instance_digest=request.content_time.layout_instance_digest,
                source_proof_digest=prepared.source_proof_digest,
                pair_video_digest=prepared.video_digest,
                pair_frame_valid_mask_digest=(
                    prepared.frame_valid_mask_digest
                ),
                pair_action_digest=prepared.action_digest,
                pair_action_mask_digest=prepared.action_mask_digest,
                paired_payload_digest=prepared.paired_payload_digest,
                dataset_manifest_sha256=request.proof.dataset_manifest_sha256,
                dataset_row_identity=request.proof.dataset_row_identity,
                state_manifest_before=before_digest,
                staged_state_manifest=after_digest,
                state_manifest_after=after_digest,
                committed_at_monotonic_ns=time.monotonic_ns(),
            )
            try:
                publication = self._publisher.publish_exclusive(
                    receipt_id=request.commit_id,
                    payload=receipt.canonical_bytes,
                    expected_sha256=receipt.receipt_sha256,
                )
            except Exception as exc:
                self._poisoned = True
                self._pending = None
                raise CacheContractError(
                    "COMMIT_RECEIPT_PUBLICATION_FAILED",
                    f"receipt publisher failed: {type(exc).__name__}",
                ) from exc
            if (
                not isinstance(publication, ReceiptPublication)
                or publication.receipt_id != request.commit_id
                or publication.sha256 != receipt.receipt_sha256
                or not publication.durable
                or not publication.readback_verified
            ):
                self._poisoned = True
                self._pending = None
                _fail(
                    "COMMIT_RECEIPT_INVALID",
                    "publisher did not attest exact durable read-back bytes",
                )

            # The only live-state mutation in a successful transaction.
            self._state = staged_state
            self._state_manifest_digest = after_digest
            if self._state.state_manifest_digest != receipt.state_manifest_after:
                self._poisoned = True
                self._pending = None
                _fail(
                    "CACHE_PUBLICATION_POISONED",
                    "published state does not match durable commit decision",
                )
            self._completed[request.commit_id] = (
                prepared.paired_payload_digest,
                receipt,
            )
            self._pending = None
            return receipt

    def reset(
        self,
        *,
        reset_id: str,
        transaction_nonce: str,
        new_episode_id: str,
        layout: ChunkActionLayout,
    ) -> CacheResetReceipt:
        """Durably publish an epoch-advancing empty-state reset."""

        _require_non_empty_string(reset_id, "reset_id")
        _require_non_empty_string(transaction_nonce, "transaction_nonce")
        _require_non_empty_string(new_episode_id, "new_episode_id")
        if not isinstance(layout, ChunkActionLayout):
            _fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "reset requires an immutable ChunkActionLayout",
            )
        if layout.synthetic_test_only != self._synthetic_test_mode:
            _fail(
                "CACHE_SYNTHETIC_LAYOUT_PROVENANCE",
                "cache reset cannot switch synthetic layout provenance mode",
            )
        reset_payload_digest = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "layout_instance_digest": layout.layout_instance_digest,
                    "layout_spec_sha256": layout.layout_spec_sha256,
                    "new_episode_id": new_episode_id,
                    "reset_id": reset_id,
                    "transaction_nonce": transaction_nonce,
                }
            )
        )
        with self._lock:
            self._assert_not_poisoned()
            self._assert_live_integrity()
            prior = self._completed_resets.get(reset_id)
            if prior is not None:
                prior_digest, prior_receipt = prior
                if prior_digest != reset_payload_digest:
                    _fail(
                        "COMMIT_DUPLICATE_CONFLICT",
                        "reset_id was already used for a different payload",
                    )
                if (
                    self._state.episode_id != prior_receipt.new_episode_id
                    or self._state.episode_epoch
                    != prior_receipt.new_episode_epoch
                    or self._state.state_manifest_digest
                    != prior_receipt.state_manifest_after
                    or self._layout.layout_instance_digest
                    != prior_receipt.layout_instance_digest
                ):
                    _fail(
                        "COMMIT_STALE_REVISION",
                        "completed reset no longer names the live empty state",
                    )
                return prior_receipt
            before = self._state
            new_state = HybridTemporalState.empty(
                episode_id=new_episode_id,
                episode_epoch=before.episode_epoch + 1,
                layout_spec_sha256=layout.layout_spec_sha256,
                layout_instance_digest=layout.layout_instance_digest,
                registry=self._registry,
            )
            aborted = self._pending[0] if self._pending is not None else None
            receipt = CacheResetReceipt(
                reset_id=reset_id,
                transaction_nonce=transaction_nonce,
                previous_episode_id=before.episode_id,
                previous_episode_epoch=before.episode_epoch,
                new_episode_id=new_state.episode_id,
                new_episode_epoch=new_state.episode_epoch,
                layout_spec_sha256=layout.layout_spec_sha256,
                layout_instance_digest=layout.layout_instance_digest,
                state_manifest_before=before.state_manifest_digest,
                state_manifest_after=new_state.state_manifest_digest,
                aborted_pending_commit_id=aborted,
                reset_at_monotonic_ns=time.monotonic_ns(),
            )
            try:
                publication = self._publisher.publish_exclusive(
                    receipt_id=reset_id,
                    payload=receipt.canonical_bytes,
                    expected_sha256=receipt.receipt_sha256,
                )
            except Exception as exc:
                self._poisoned = True
                raise CacheContractError(
                    "COMMIT_RECEIPT_PUBLICATION_FAILED",
                    f"reset receipt publisher failed: {type(exc).__name__}",
                ) from exc
            if (
                not isinstance(publication, ReceiptPublication)
                or publication.receipt_id != reset_id
                or publication.sha256 != receipt.receipt_sha256
                or not publication.durable
                or not publication.readback_verified
            ):
                self._poisoned = True
                _fail(
                    "COMMIT_RECEIPT_INVALID",
                    "reset publisher did not attest exact durable bytes",
                )
            self._state = new_state
            self._layout = layout
            self._state_manifest_digest = new_state.state_manifest_digest
            self._pending = None
            self._completed.clear()
            self._completed_resets[reset_id] = (reset_payload_digest, receipt)
            return receipt


__all__ = [
    "CacheContractError",
    "CacheResetReceipt",
    "CacheScratch",
    "CommitSource",
    "ContentTime",
    "DenoiseReadView",
    "DurableReceiptPublisher",
    "HybridCacheManager",
    "HybridTemporalState",
    "LayerKind",
    "LayerRegistry",
    "LayerScratch",
    "LayerSpec",
    "LayerTemporalState",
    "PairedCommitReceipt",
    "PairedStagingContext",
    "ReceiptPublication",
    "StagedLayerPayload",
    "TeacherForcingCommitRequest",
    "TeacherForcingDatasetPairProof",
    "TensorSnapshot",
    "tensor_digest",
    "validate_cach_a_registry",
]
