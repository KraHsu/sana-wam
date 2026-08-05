"""Immutable CACH raw-frame/latent/action layout contracts.

This module is deliberately independent of torch, model weights, CUDA, HDF5,
and the filesystem.  Production callers must provide an externally verified
row-timebase proof.  Synthetic proofs are accepted only by the explicitly
test-named entry point and can never be promoted by flipping a boolean in the
proof payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
import struct
from typing import Any, Mapping, Sequence


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SYNTHETIC_TEST_MARKER = "CACH_SYNTHETIC_LAYOUT_TEST_ONLY_V1"
_LAYOUT_SPEC_SCHEMA = "cach.chunk_action_layout_spec.v1"
_TIMEBASE_PROOF_SCHEMA = "cach.row_timebase_proof.v1"
_LAYOUT_INSTANCE_SCHEMA = "cach.chunk_action_layout_instance.v1"


class LayoutReasonCode(str, Enum):
    """Stable failure codes frozen by the Stage-0 layout design."""

    MISSING_VERIFIED_ROW_TIMEBASE = "MISSING_VERIFIED_ROW_TIMEBASE"
    NON_MONOTONIC_TIMESTAMP = "NON_MONOTONIC_TIMESTAMP"
    AMBIGUOUS_ACTION_EFFECTIVE_TIME = "AMBIGUOUS_ACTION_EFFECTIVE_TIME"
    DROPPED_OR_DUPLICATED_STEP = "DROPPED_OR_DUPLICATED_STEP"
    VAE_CONTRACT_MISMATCH = "VAE_CONTRACT_MISMATCH"
    LAYOUT_GRID_MISMATCH = "LAYOUT_GRID_MISMATCH"
    NONZERO_EPISODE_ROW_START = "NONZERO_EPISODE_ROW_START"
    ACTION_COVERAGE_GAP = "ACTION_COVERAGE_GAP"
    ACTION_COVERAGE_OVERLAP = "ACTION_COVERAGE_OVERLAP"
    SILENT_VALID_TAIL_DROP = "SILENT_VALID_TAIL_DROP"
    PAD_MARKED_VALID = "PAD_MARKED_VALID"
    NON_PREFIX_VALID_MASK = "NON_PREFIX_VALID_MASK"
    LEGACY_AR_PREFIX_KEY_PRESENT = "LEGACY_AR_PREFIX_KEY_PRESENT"
    LEGACY_CLEAN_PREFIX_NONZERO = "LEGACY_CLEAN_PREFIX_NONZERO"
    UNREGISTERED_PREFIX_DISCARD = "UNREGISTERED_PREFIX_DISCARD"
    BOOTSTRAP_CONTRACT_MISMATCH = "BOOTSTRAP_CONTRACT_MISMATCH"
    PROPRIO_BOUNDARY_MISMATCH = "PROPRIO_BOUNDARY_MISMATCH"
    FUTURE_PROPRIO_VISIBLE = "FUTURE_PROPRIO_VISIBLE"
    LAYOUT_SPEC_SHA_MISMATCH = "LAYOUT_SPEC_SHA_MISMATCH"
    LAYOUT_INSTANCE_DIGEST_MISMATCH = "LAYOUT_INSTANCE_DIGEST_MISMATCH"
    ACTION_ROPE_CURSOR_MISMATCH = "ACTION_ROPE_CURSOR_MISMATCH"


class LayoutContractError(ValueError):
    """A fail-closed layout error with one stable machine reason code."""

    def __init__(
        self,
        code: LayoutReasonCode,
        detail: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(code, LayoutReasonCode):
            raise TypeError("code must be a LayoutReasonCode")
        self.code = code
        self.detail = str(detail)
        self.context = dict(context or {})
        super().__init__(f"{code.value}: {self.detail}")


def _require_plain_int(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a plain integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _canonical_value(value: Any, path: str = "$") -> Any:
    """Convert a contract payload to strict canonical-JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return _canonical_value(value.value, path)
    if isinstance(value, tuple):
        return [
            _canonical_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, list):
        return [
            _canonical_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string JSON key")
            normalized[key] = _canonical_value(item, f"{path}.{key}")
        return normalized
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _canonical_value(to_payload(), path)
    raise TypeError(
        f"{path} contains a non-canonical value of type {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one strict payload using stable UTF-8 canonical JSON."""

    return (
        json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_proprio_row_sha256(value: Any) -> str:
    """Digest one semantic float32 proprio row independently of container type."""

    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        dtype_name = str(dtype)
        if dtype_name.startswith("torch."):
            dtype_name = dtype_name[len("torch.") :]
        if dtype_name not in {"float32", "float"}:
            raise TypeError(
                f"proprio row dtype must be semantic float32, got {dtype}"
            )
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError("proprio row must be a finite sequence") from exc
    if len(items) != 20:
        raise ValueError("proprio row must contain exactly 20 float32 values")
    normalized = []
    for index, item in enumerate(items):
        scalar_item = getattr(item, "item", None)
        if callable(scalar_item):
            item = scalar_item()
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"proprio row item {index} is not numeric")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError(f"proprio row item {index} is non-finite")
        normalized.append(numeric)
    material = (
        b"cach.proprio.semantic_float32_be.v1\0"
        + struct.pack(">20f", *normalized)
    )
    return hashlib.sha256(material).hexdigest()


def _strict_bool_tuple(value: Sequence[Any], name: str) -> tuple[bool, ...]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite boolean sequence") from exc
    if any(type(item) is not bool for item in items):
        raise TypeError(f"{name} must contain plain booleans only")
    return items


def _strict_int_tuple(
    value: Sequence[Any],
    name: str,
    *,
    minimum: int = 0,
) -> tuple[int, ...]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite integer sequence") from exc
    for index, item in enumerate(items):
        _require_plain_int(item, f"{name}[{index}]", minimum=minimum)
    return items


def _validate_contiguous_prefix(mask: tuple[bool, ...], name: str) -> int:
    seen_pad = False
    valid = 0
    for item in mask:
        if item:
            if seen_pad:
                raise LayoutContractError(
                    LayoutReasonCode.NON_PREFIX_VALID_MASK,
                    f"{name} contains a valid token after padding began",
                )
            valid += 1
        else:
            seen_pad = True
    return valid


def _validate_strictly_increasing(
    values: tuple[int, ...],
    name: str,
) -> None:
    if any(right <= left for left, right in zip(values, values[1:])):
        raise LayoutContractError(
            LayoutReasonCode.NON_MONOTONIC_TIMESTAMP,
            f"{name} must be strictly increasing",
        )


@dataclass(frozen=True)
class ChunkActionLayoutSpec:
    """Static layout policy whose canonical digest is ``layout_spec_sha256``."""

    implementation_source_sha256: str
    candidate_revision: str
    causal_vae_source_sha256: str
    action_schema_sha256: str
    normalization_schema_sha256: str
    frame_chunk_size: int
    temporal_compression: int
    video_stride: int
    action_dim: int = 20
    episode_bootstrap: str = "first_frame_pinned"
    observed_prefix_chunks: int = 0
    action_time_semantics: str = "token_t_enters_raw_frame_t_plus_1"
    action_reducer: str = "end_of_bin_command"
    proprio_boundary: str = "previous_committed_action_terminal_state"
    action_rope_policy: str = "episode_valid_action_token_ordinal"
    synthetic_test_only: bool = False
    schema: str = _LAYOUT_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _LAYOUT_SPEC_SCHEMA:
            raise ValueError(f"unsupported layout spec schema: {self.schema!r}")
        _require_sha256(
            self.implementation_source_sha256,
            "implementation_source_sha256",
        )
        _require_nonempty_string(self.candidate_revision, "candidate_revision")
        _require_sha256(
            self.causal_vae_source_sha256,
            "causal_vae_source_sha256",
        )
        _require_sha256(self.action_schema_sha256, "action_schema_sha256")
        _require_sha256(
            self.normalization_schema_sha256,
            "normalization_schema_sha256",
        )
        _require_plain_int(
            self.frame_chunk_size,
            "frame_chunk_size",
            minimum=1,
        )
        _require_plain_int(
            self.temporal_compression,
            "temporal_compression",
            minimum=1,
        )
        _require_plain_int(self.video_stride, "video_stride", minimum=1)
        _require_plain_int(self.action_dim, "action_dim", minimum=1)
        if self.episode_bootstrap != "first_frame_pinned":
            raise LayoutContractError(
                LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
                "CACH v0 requires episode_bootstrap='first_frame_pinned'",
            )
        if self.observed_prefix_chunks != 0:
            raise LayoutContractError(
                LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
                "CACH v0 requires observed_prefix_chunks=0",
            )
        required_literals = {
            "action_time_semantics": "token_t_enters_raw_frame_t_plus_1",
            "action_reducer": "end_of_bin_command",
            "proprio_boundary": "previous_committed_action_terminal_state",
            "action_rope_policy": "episode_valid_action_token_ordinal",
        }
        for name, expected in required_literals.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} must be the CACH v0 literal {expected!r}"
                )
        if type(self.synthetic_test_only) is not bool:
            raise TypeError("synthetic_test_only must be a plain boolean")

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "action_reducer": self.action_reducer,
            "action_rope_policy": self.action_rope_policy,
            "action_schema_sha256": self.action_schema_sha256,
            "action_time_semantics": self.action_time_semantics,
            "candidate_revision": self.candidate_revision,
            "causal_vae_source_sha256": self.causal_vae_source_sha256,
            "episode_bootstrap": self.episode_bootstrap,
            "frame_chunk_size": self.frame_chunk_size,
            "implementation_source_sha256": self.implementation_source_sha256,
            "normalization_schema_sha256": self.normalization_schema_sha256,
            "observed_prefix_chunks": self.observed_prefix_chunks,
            "proprio_boundary": self.proprio_boundary,
            "schema": self.schema,
            "synthetic_test_only": self.synthetic_test_only,
            "temporal_compression": self.temporal_compression,
            "video_stride": self.video_stride,
        }

    @property
    def layout_spec_sha256(self) -> str:
        return canonical_sha256(self.to_payload())


def synthetic_layout_spec(
    *,
    frame_chunk_size: int,
    temporal_compression: int,
    video_stride: int,
    action_dim: int = 20,
) -> ChunkActionLayoutSpec:
    """Create a visibly synthetic spec usable only by test-named builders."""

    return ChunkActionLayoutSpec(
        implementation_source_sha256=hashlib.sha256(
            b"cach.synthetic.layout.implementation.v1"
        ).hexdigest(),
        candidate_revision="synthetic-test-only",
        causal_vae_source_sha256=hashlib.sha256(
            b"cach.synthetic.causal-vae.v1"
        ).hexdigest(),
        action_schema_sha256=hashlib.sha256(
            f"cach.synthetic.action-schema.{action_dim}".encode("ascii")
        ).hexdigest(),
        normalization_schema_sha256=hashlib.sha256(
            b"cach.synthetic.normalization.v1"
        ).hexdigest(),
        frame_chunk_size=frame_chunk_size,
        temporal_compression=temporal_compression,
        video_stride=video_stride,
        action_dim=action_dim,
        synthetic_test_only=True,
    )


@dataclass(frozen=True)
class RowTimebaseProof:
    """Externally produced proof describing one concrete row's timebase."""

    source_row_digest: str
    episode_id_digest: str
    timebase_manifest_digest: str
    verification_receipt_digest: str
    proprio_source_receipt_digest: str
    observation_timestamps: tuple[int, ...]
    sampled_video_raw_indices: tuple[int, ...]
    action_destination_timestamps: tuple[int, ...]
    action_destination_state_indices: tuple[int, ...]
    equal_rate_proven: bool
    proof_mode: str
    synthetic_test_marker: str | None = None
    schema: str = _TIMEBASE_PROOF_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _TIMEBASE_PROOF_SCHEMA:
            raise ValueError(f"unsupported timebase proof schema: {self.schema!r}")
        _require_sha256(self.source_row_digest, "source_row_digest")
        _require_sha256(self.episode_id_digest, "episode_id_digest")
        _require_sha256(
            self.timebase_manifest_digest,
            "timebase_manifest_digest",
        )
        _require_sha256(
            self.verification_receipt_digest,
            "verification_receipt_digest",
        )
        _require_sha256(
            self.proprio_source_receipt_digest,
            "proprio_source_receipt_digest",
        )
        object.__setattr__(
            self,
            "observation_timestamps",
            _strict_int_tuple(
                self.observation_timestamps,
                "observation_timestamps",
            ),
        )
        object.__setattr__(
            self,
            "sampled_video_raw_indices",
            _strict_int_tuple(
                self.sampled_video_raw_indices,
                "sampled_video_raw_indices",
            ),
        )
        object.__setattr__(
            self,
            "action_destination_timestamps",
            _strict_int_tuple(
                self.action_destination_timestamps,
                "action_destination_timestamps",
            ),
        )
        object.__setattr__(
            self,
            "action_destination_state_indices",
            _strict_int_tuple(
                self.action_destination_state_indices,
                "action_destination_state_indices",
            ),
        )
        if type(self.equal_rate_proven) is not bool:
            raise TypeError("equal_rate_proven must be a plain boolean")
        if self.proof_mode not in {"verified_manifest", "synthetic_test"}:
            raise ValueError(f"unsupported proof_mode: {self.proof_mode!r}")
        if self.proof_mode == "synthetic_test":
            if self.synthetic_test_marker != _SYNTHETIC_TEST_MARKER:
                raise ValueError("synthetic timebase proof has the wrong marker")
        elif self.synthetic_test_marker is not None:
            raise ValueError("verified_manifest proof cannot carry a synthetic marker")

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_destination_state_indices": (
                self.action_destination_state_indices
            ),
            "action_destination_timestamps": (
                self.action_destination_timestamps
            ),
            "episode_id_digest": self.episode_id_digest,
            "equal_rate_proven": self.equal_rate_proven,
            "observation_timestamps": self.observation_timestamps,
            "proof_mode": self.proof_mode,
            "proprio_source_receipt_digest": (
                self.proprio_source_receipt_digest
            ),
            "sampled_video_raw_indices": self.sampled_video_raw_indices,
            "schema": self.schema,
            "source_row_digest": self.source_row_digest,
            "synthetic_test_marker": self.synthetic_test_marker,
            "timebase_manifest_digest": self.timebase_manifest_digest,
            "verification_receipt_digest": self.verification_receipt_digest,
        }

    @property
    def proof_digest(self) -> str:
        return canonical_sha256(self.to_payload())

    def validate_for_production(self) -> None:
        del self
        raise LayoutContractError(
            LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
            "Stage 1 has no external manifest/receipt verifier capability; "
            "caller-supplied proof metadata cannot authorize production layout",
        )

    def validate_for_synthetic_test(self) -> None:
        if (
            self.proof_mode != "synthetic_test"
            or self.synthetic_test_marker != _SYNTHETIC_TEST_MARKER
        ):
            raise LayoutContractError(
                LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
                "synthetic test builder requires an explicit synthetic proof",
            )


def synthetic_equal_rate_proof(
    *,
    valid_raw_count: int,
    video_stride: int,
    source_row_label: str = "row-0",
    episode_label: str = "episode-0",
) -> RowTimebaseProof:
    """Build a deterministic synthetic same-clock proof for unit tests."""

    valid_raw_count = _require_plain_int(
        valid_raw_count,
        "valid_raw_count",
        minimum=1,
    )
    video_stride = _require_plain_int(
        video_stride,
        "video_stride",
        minimum=1,
    )
    _require_nonempty_string(source_row_label, "source_row_label")
    _require_nonempty_string(episode_label, "episode_label")
    observation_timestamps = tuple(range(valid_raw_count))
    sampled_video_raw_indices = tuple(
        range(0, valid_raw_count, video_stride)
    )
    action_destination_state_indices = tuple(range(1, valid_raw_count))
    action_destination_timestamps = action_destination_state_indices
    source_row_digest = hashlib.sha256(
        f"synthetic-row:{source_row_label}".encode("utf-8")
    ).hexdigest()
    episode_id_digest = hashlib.sha256(
        f"synthetic-episode:{episode_label}".encode("utf-8")
    ).hexdigest()
    timebase_manifest_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "observation_timestamps": observation_timestamps,
                "sampled_video_raw_indices": sampled_video_raw_indices,
                "action_destination_timestamps": (
                    action_destination_timestamps
                ),
            }
        )
    ).hexdigest()
    receipt_digest = hashlib.sha256(
        f"synthetic-receipt:{timebase_manifest_digest}".encode("ascii")
    ).hexdigest()
    proprio_receipt_digest = hashlib.sha256(
        f"synthetic-proprio:{timebase_manifest_digest}".encode("ascii")
    ).hexdigest()
    return RowTimebaseProof(
        source_row_digest=source_row_digest,
        episode_id_digest=episode_id_digest,
        timebase_manifest_digest=timebase_manifest_digest,
        verification_receipt_digest=receipt_digest,
        proprio_source_receipt_digest=proprio_receipt_digest,
        observation_timestamps=observation_timestamps,
        sampled_video_raw_indices=sampled_video_raw_indices,
        action_destination_timestamps=action_destination_timestamps,
        action_destination_state_indices=action_destination_state_indices,
        equal_rate_proven=True,
        proof_mode="synthetic_test",
        synthetic_test_marker=_SYNTHETIC_TEST_MARKER,
    )


@dataclass(frozen=True)
class LatentActionSpan:
    latent_index: int
    action_start: int
    action_end: int
    anchor_no_action_slot: bool

    def __post_init__(self) -> None:
        _require_plain_int(self.latent_index, "latent_index", minimum=0)
        _require_plain_int(self.action_start, "action_start", minimum=0)
        _require_plain_int(self.action_end, "action_end", minimum=0)
        if self.action_end < self.action_start:
            raise ValueError("latent action span end precedes its start")
        if type(self.anchor_no_action_slot) is not bool:
            raise TypeError("anchor_no_action_slot must be a plain boolean")
        if self.anchor_no_action_slot and (
            self.latent_index != 0 or self.action_start != self.action_end
        ):
            raise ValueError("only latent 0 may carry the empty no-action slot")

    @property
    def action_count(self) -> int:
        return self.action_end - self.action_start

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_end": self.action_end,
            "action_start": self.action_start,
            "anchor_no_action_slot": self.anchor_no_action_slot,
            "latent_index": self.latent_index,
        }


@dataclass(frozen=True)
class ChunkLayout:
    chunk_id: int
    latent_start: int
    latent_end: int
    latent_valid_mask: tuple[bool, ...]
    action_start: int
    action_end: int
    action_slot_capacity: int
    action_valid_mask: tuple[bool, ...]
    latent_action_spans: tuple[LatentActionSpan, ...]
    proprio_raw_index: int
    proprio_timestamp: int
    proprio_source_receipt_digest: str
    action_rope_start: int
    action_rope_end: int
    is_bootstrap: bool
    is_partial_tail: bool
    anchor_no_action_slot: bool

    def __post_init__(self) -> None:
        _require_plain_int(self.chunk_id, "chunk_id", minimum=0)
        _require_plain_int(self.latent_start, "latent_start", minimum=0)
        _require_plain_int(self.latent_end, "latent_end", minimum=1)
        if self.latent_end <= self.latent_start:
            raise ValueError("chunk latent interval must be non-empty")
        object.__setattr__(
            self,
            "latent_valid_mask",
            _strict_bool_tuple(self.latent_valid_mask, "latent_valid_mask"),
        )
        valid_mask_count = _validate_contiguous_prefix(
            self.latent_valid_mask,
            "latent_valid_mask",
        )
        if valid_mask_count != self.latent_end - self.latent_start:
            raise ValueError(
                "latent_valid_mask differs from the valid latent interval"
            )
        _require_plain_int(self.action_start, "action_start", minimum=0)
        _require_plain_int(self.action_end, "action_end", minimum=0)
        if self.action_end < self.action_start:
            raise ValueError("chunk action end precedes its start")
        _require_plain_int(
            self.action_slot_capacity,
            "action_slot_capacity",
            minimum=0,
        )
        if self.action_slot_capacity < self.action_end - self.action_start:
            raise ValueError(
                "action_slot_capacity is smaller than the valid action interval"
            )
        object.__setattr__(
            self,
            "action_valid_mask",
            _strict_bool_tuple(self.action_valid_mask, "action_valid_mask"),
        )
        if len(self.action_valid_mask) != self.action_slot_capacity:
            raise ValueError(
                "action_valid_mask length differs from action_slot_capacity"
            )
        valid_action_mask_count = _validate_contiguous_prefix(
            self.action_valid_mask,
            "action_valid_mask",
        )
        if valid_action_mask_count != self.action_end - self.action_start:
            raise LayoutContractError(
                LayoutReasonCode.PAD_MARKED_VALID,
                "action_valid_mask differs from the valid action interval",
            )
        object.__setattr__(
            self,
            "latent_action_spans",
            tuple(self.latent_action_spans),
        )
        if any(
            not isinstance(span, LatentActionSpan)
            for span in self.latent_action_spans
        ):
            raise TypeError(
                "latent_action_spans must contain LatentActionSpan values"
            )
        if len(self.latent_action_spans) != self.latent_end - self.latent_start:
            raise ValueError(
                "latent_action_spans length differs from the valid latent interval"
            )
        expected_latents = tuple(range(self.latent_start, self.latent_end))
        observed_latents = tuple(
            span.latent_index for span in self.latent_action_spans
        )
        if observed_latents != expected_latents:
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_GRID_MISMATCH,
                "latent action spans do not match the chunk latent interval",
            )
        if self.latent_action_spans[0].action_start != self.action_start:
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_GAP,
                "first latent span does not begin at chunk.action_start",
            )
        if self.latent_action_spans[-1].action_end != self.action_end:
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_GAP,
                "last latent span does not end at chunk.action_end",
            )
        for left, right in zip(
            self.latent_action_spans,
            self.latent_action_spans[1:],
        ):
            if left.action_end != right.action_start:
                code = (
                    LayoutReasonCode.ACTION_COVERAGE_GAP
                    if left.action_end < right.action_start
                    else LayoutReasonCode.ACTION_COVERAGE_OVERLAP
                )
                raise LayoutContractError(
                    code,
                    "adjacent latent action spans have a gap or overlap",
                )
        for span in self.latent_action_spans:
            if span.anchor_no_action_slot:
                if (
                    not self.is_bootstrap
                    or span.latent_index != 0
                    or span.action_count != 0
                ):
                    raise LayoutContractError(
                        LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
                        "only bootstrap latent 0 may own the no-action slot",
                    )
            elif span.action_count <= 0:
                raise LayoutContractError(
                    LayoutReasonCode.ACTION_COVERAGE_GAP,
                    "every non-anchor latent must own at least one valid action",
                )
        _require_plain_int(
            self.proprio_raw_index,
            "proprio_raw_index",
            minimum=0,
        )
        _require_plain_int(
            self.proprio_timestamp,
            "proprio_timestamp",
            minimum=0,
        )
        _require_sha256(
            self.proprio_source_receipt_digest,
            "proprio_source_receipt_digest",
        )
        _require_plain_int(
            self.action_rope_start,
            "action_rope_start",
            minimum=0,
        )
        _require_plain_int(
            self.action_rope_end,
            "action_rope_end",
            minimum=0,
        )
        if (
            self.action_rope_start != self.action_start
            or self.action_rope_end != self.action_end
        ):
            raise LayoutContractError(
                LayoutReasonCode.ACTION_ROPE_CURSOR_MISMATCH,
                "chunk Action RoPE must equal its valid action ownership interval",
            )
        for name in ("is_bootstrap", "is_partial_tail", "anchor_no_action_slot"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a plain boolean")
        if self.is_bootstrap != (self.chunk_id == 0):
            raise LayoutContractError(
                LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
                "only chunk 0 may be the bootstrap chunk",
            )
        if self.anchor_no_action_slot != self.is_bootstrap:
            raise LayoutContractError(
                LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
                "the no-action anchor flag must occur exactly on chunk 0",
            )
        expected_partial = self.valid_latent_count < len(self.latent_valid_mask)
        if self.is_partial_tail != expected_partial:
            raise LayoutContractError(
                LayoutReasonCode.SILENT_VALID_TAIL_DROP,
                "is_partial_tail differs from the explicit latent valid mask",
            )

    @property
    def valid_latent_count(self) -> int:
        return self.latent_end - self.latent_start

    @property
    def valid_action_count(self) -> int:
        return self.action_end - self.action_start

    @property
    def action_rope_positions(self) -> tuple[int, ...]:
        return tuple(range(self.action_rope_start, self.action_rope_end))

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_end": self.action_end,
            "action_rope_end": self.action_rope_end,
            "action_rope_start": self.action_rope_start,
            "action_slot_capacity": self.action_slot_capacity,
            "action_start": self.action_start,
            "action_valid_mask": self.action_valid_mask,
            "anchor_no_action_slot": self.anchor_no_action_slot,
            "chunk_id": self.chunk_id,
            "is_bootstrap": self.is_bootstrap,
            "is_partial_tail": self.is_partial_tail,
            "latent_action_spans": self.latent_action_spans,
            "latent_end": self.latent_end,
            "latent_start": self.latent_start,
            "latent_valid_mask": self.latent_valid_mask,
            "proprio_raw_index": self.proprio_raw_index,
            "proprio_source_receipt_digest": (
                self.proprio_source_receipt_digest
            ),
            "proprio_timestamp": self.proprio_timestamp,
        }


@dataclass(frozen=True)
class SelectedProprioBinding:
    """Layout-derived identity for one model-facing proprio tensor."""

    layout_instance_digest: str
    chunk_id: int
    raw_index: int
    timestamp: int
    source_receipt_digest: str
    proprio_value_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.layout_instance_digest,
            "layout_instance_digest",
        )
        _require_plain_int(self.chunk_id, "chunk_id", minimum=0)
        _require_plain_int(self.raw_index, "raw_index", minimum=0)
        _require_plain_int(self.timestamp, "timestamp", minimum=0)
        _require_sha256(
            self.source_receipt_digest,
            "source_receipt_digest",
        )
        _require_sha256(
            self.proprio_value_sha256,
            "proprio_value_sha256",
        )

    def verify(
        self,
        *,
        layout_instance_digest: str,
        chunk: ChunkLayout,
        proprio_value_sha256: str,
    ) -> None:
        if not isinstance(chunk, ChunkLayout):
            raise TypeError("chunk must be a ChunkLayout")
        expected = (
            _require_sha256(
                layout_instance_digest,
                "layout_instance_digest",
            ),
            chunk.chunk_id,
            chunk.proprio_raw_index,
            chunk.proprio_timestamp,
            chunk.proprio_source_receipt_digest,
            _require_sha256(
                proprio_value_sha256,
                "proprio_value_sha256",
            ),
        )
        observed = (
            self.layout_instance_digest,
            self.chunk_id,
            self.raw_index,
            self.timestamp,
            self.source_receipt_digest,
            self.proprio_value_sha256,
        )
        if observed != expected:
            raise LayoutContractError(
                LayoutReasonCode.PROPRIO_BOUNDARY_MISMATCH,
                "selected proprio binding differs from the layout chunk",
            )


@dataclass(frozen=True)
class ChunkActionLayout:
    """One immutable concrete row layout."""

    synthetic_test_only: bool
    layout_spec_sha256: str
    source_row_digest: str
    episode_id_digest: str
    row_start_raw_index: int
    timebase_manifest_digest: str
    timestamp_verification_receipt_digest: str
    frame_chunk_size: int
    temporal_compression: int
    video_stride: int
    equal_rate_proven: bool
    action_tokens_per_non_anchor_latent: int | None
    valid_raw_count: int
    valid_video_count: int
    valid_latent_count: int
    valid_action_count: int
    chunks: tuple[ChunkLayout, ...]
    coverage_receipt_digest: str
    padding_receipt_digest: str
    schema: str = _LAYOUT_INSTANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _LAYOUT_INSTANCE_SCHEMA:
            raise ValueError(
                f"unsupported layout instance schema: {self.schema!r}"
            )
        if type(self.synthetic_test_only) is not bool:
            raise TypeError("synthetic_test_only must be a plain boolean")
        _require_sha256(self.layout_spec_sha256, "layout_spec_sha256")
        _require_sha256(self.source_row_digest, "source_row_digest")
        _require_sha256(self.episode_id_digest, "episode_id_digest")
        _require_plain_int(
            self.row_start_raw_index,
            "row_start_raw_index",
            minimum=0,
        )
        if self.row_start_raw_index != 0:
            raise LayoutContractError(
                LayoutReasonCode.NONZERO_EPISODE_ROW_START,
                "CACH v0 layout instances must begin at raw index 0",
            )
        _require_sha256(
            self.timebase_manifest_digest,
            "timebase_manifest_digest",
        )
        _require_sha256(
            self.timestamp_verification_receipt_digest,
            "timestamp_verification_receipt_digest",
        )
        _require_plain_int(
            self.frame_chunk_size,
            "frame_chunk_size",
            minimum=1,
        )
        _require_plain_int(
            self.temporal_compression,
            "temporal_compression",
            minimum=1,
        )
        _require_plain_int(self.video_stride, "video_stride", minimum=1)
        if type(self.equal_rate_proven) is not bool:
            raise TypeError("equal_rate_proven must be a plain boolean")
        if self.action_tokens_per_non_anchor_latent is not None:
            _require_plain_int(
                self.action_tokens_per_non_anchor_latent,
                "action_tokens_per_non_anchor_latent",
                minimum=1,
            )
        if self.equal_rate_proven != (
            self.action_tokens_per_non_anchor_latent is not None
        ):
            raise ValueError(
                "equal-rate layouts must carry r and irregular layouts must not"
            )
        _require_plain_int(self.valid_raw_count, "valid_raw_count", minimum=1)
        _require_plain_int(
            self.valid_video_count,
            "valid_video_count",
            minimum=1,
        )
        _require_plain_int(
            self.valid_latent_count,
            "valid_latent_count",
            minimum=1,
        )
        _require_plain_int(
            self.valid_action_count,
            "valid_action_count",
            minimum=0,
        )
        if self.valid_action_count != self.valid_raw_count - 1:
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_GAP,
                "valid_action_count must equal valid_raw_count-1",
            )
        if (self.valid_raw_count - 1) % self.video_stride != 0:
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_GRID_MISMATCH,
                "valid raw endpoint is not on the video-stride grid",
            )
        expected_video_count = (
            (self.valid_raw_count - 1) // self.video_stride
        ) + 1
        if self.valid_video_count != expected_video_count:
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_GRID_MISMATCH,
                "valid_video_count differs from raw/video stride geometry",
            )
        if (self.valid_video_count - 1) % self.temporal_compression != 0:
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_GRID_MISMATCH,
                "valid video endpoint is not on the causal-VAE grid",
            )
        expected_latent_count = 1 + (
            (self.valid_video_count - 1) // self.temporal_compression
        )
        if self.valid_latent_count != expected_latent_count:
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_GRID_MISMATCH,
                "valid_latent_count differs from causal-VAE geometry",
            )
        if self.equal_rate_proven:
            expected_rate = self.temporal_compression * self.video_stride
            if self.action_tokens_per_non_anchor_latent != expected_rate:
                raise LayoutContractError(
                    LayoutReasonCode.DROPPED_OR_DUPLICATED_STEP,
                    "equal-rate layout carries the wrong action rate",
                )
        object.__setattr__(self, "chunks", tuple(self.chunks))
        if not self.chunks or any(
            not isinstance(chunk, ChunkLayout) for chunk in self.chunks
        ):
            raise TypeError("chunks must be a non-empty tuple of ChunkLayout")
        if tuple(chunk.chunk_id for chunk in self.chunks) != tuple(
            range(len(self.chunks))
        ):
            raise ValueError("chunk ids must be contiguous from zero")
        expected_chunk_count = (
            self.valid_latent_count + self.frame_chunk_size - 1
        ) // self.frame_chunk_size
        if len(self.chunks) != expected_chunk_count:
            raise LayoutContractError(
                LayoutReasonCode.SILENT_VALID_TAIL_DROP,
                "chunk count differs from the ceiling latent grid",
            )
        for chunk in self.chunks:
            if len(chunk.latent_valid_mask) != self.frame_chunk_size:
                raise ValueError(
                    "every chunk latent mask must use frame_chunk_size entries"
                )
            valid_latent_mask_count = _validate_contiguous_prefix(
                chunk.latent_valid_mask,
                f"chunks[{chunk.chunk_id}].latent_valid_mask",
            )
            if valid_latent_mask_count != chunk.valid_latent_count:
                raise ValueError(
                    "chunk latent mask differs from its valid latent interval"
                )
            expected_start = chunk.chunk_id * self.frame_chunk_size
            expected_end = min(
                expected_start + self.frame_chunk_size,
                self.valid_latent_count,
            )
            if (
                chunk.latent_start != expected_start
                or chunk.latent_end != expected_end
            ):
                raise LayoutContractError(
                    LayoutReasonCode.LAYOUT_GRID_MISMATCH,
                    "chunk latent interval differs from the ceiling grid",
                )
            expected_partial = (
                chunk.chunk_id == expected_chunk_count - 1
                and chunk.valid_latent_count < self.frame_chunk_size
            )
            if chunk.is_partial_tail != expected_partial:
                raise LayoutContractError(
                    LayoutReasonCode.SILENT_VALID_TAIL_DROP,
                    "partial-tail flag differs from the ceiling grid",
                )
            if self.action_tokens_per_non_anchor_latent is None:
                raise LayoutContractError(
                    LayoutReasonCode.DROPPED_OR_DUPLICATED_STEP,
                    "CACH v0 requires a proven equal action/video rate",
                )
            expected_action_capacity = (
                self.frame_chunk_size - 1
                if chunk.is_bootstrap
                else self.frame_chunk_size
            ) * self.action_tokens_per_non_anchor_latent
            if chunk.action_slot_capacity != expected_action_capacity:
                raise LayoutContractError(
                    LayoutReasonCode.LAYOUT_GRID_MISMATCH,
                    "chunk action capacity differs from the fixed K/r grid",
                )
        if self.chunks[0].latent_start != 0:
            raise LayoutContractError(
                LayoutReasonCode.SILENT_VALID_TAIL_DROP,
                "first chunk does not begin at latent 0",
            )
        if self.chunks[-1].latent_end != self.valid_latent_count:
            raise LayoutContractError(
                LayoutReasonCode.SILENT_VALID_TAIL_DROP,
                "last chunk does not end at valid_latent_count",
            )
        if self.chunks[0].action_start != 0:
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_GAP,
                "first chunk does not begin at action token 0",
            )
        if self.chunks[-1].action_end != self.valid_action_count:
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_GAP,
                "last chunk does not end at valid_action_count",
            )
        for left, right in zip(self.chunks, self.chunks[1:]):
            if left.latent_end != right.latent_start:
                raise LayoutContractError(
                    LayoutReasonCode.SILENT_VALID_TAIL_DROP,
                    "adjacent chunks have a latent gap or overlap",
                )
            if left.action_end != right.action_start:
                code = (
                    LayoutReasonCode.ACTION_COVERAGE_GAP
                    if left.action_end < right.action_start
                    else LayoutReasonCode.ACTION_COVERAGE_OVERLAP
                )
                raise LayoutContractError(
                    code,
                    "adjacent chunks have an action gap or overlap",
                )
        _require_sha256(
            self.coverage_receipt_digest,
            "coverage_receipt_digest",
        )
        _require_sha256(
            self.padding_receipt_digest,
            "padding_receipt_digest",
        )
        proprio_receipts = {
            chunk.proprio_source_receipt_digest for chunk in self.chunks
        }
        if len(proprio_receipts) != 1:
            raise LayoutContractError(
                LayoutReasonCode.PROPRIO_BOUNDARY_MISMATCH,
                "chunks disagree on the proprio source receipt",
            )
        if self.equal_rate_proven:
            for chunk in self.chunks:
                expected_proprio = (
                    0 if chunk.chunk_id == 0 else chunk.action_start
                )
                if chunk.proprio_raw_index != expected_proprio:
                    raise LayoutContractError(
                        LayoutReasonCode.PROPRIO_BOUNDARY_MISMATCH,
                        "equal-rate proprio index differs from action_start",
                    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_tokens_per_non_anchor_latent": (
                self.action_tokens_per_non_anchor_latent
            ),
            "chunks": self.chunks,
            "coverage_receipt_digest": self.coverage_receipt_digest,
            "episode_id_digest": self.episode_id_digest,
            "equal_rate_proven": self.equal_rate_proven,
            "frame_chunk_size": self.frame_chunk_size,
            "layout_spec_sha256": self.layout_spec_sha256,
            "padding_receipt_digest": self.padding_receipt_digest,
            "row_start_raw_index": self.row_start_raw_index,
            "schema": self.schema,
            "source_row_digest": self.source_row_digest,
            "synthetic_test_only": self.synthetic_test_only,
            "temporal_compression": self.temporal_compression,
            "timebase_manifest_digest": self.timebase_manifest_digest,
            "timestamp_verification_receipt_digest": (
                self.timestamp_verification_receipt_digest
            ),
            "valid_action_count": self.valid_action_count,
            "valid_latent_count": self.valid_latent_count,
            "valid_raw_count": self.valid_raw_count,
            "valid_video_count": self.valid_video_count,
            "video_stride": self.video_stride,
        }

    @property
    def layout_instance_digest(self) -> str:
        return canonical_sha256(self.to_payload())

    @property
    def proprio_raw_indices(self) -> tuple[int, ...]:
        return tuple(chunk.proprio_raw_index for chunk in self.chunks)

    def bind_selected_proprio_digests(
        self,
        proprio_value_sha256s: Sequence[str],
    ) -> tuple[SelectedProprioBinding, ...]:
        """Bind each layout-selected state to its actual semantic float32 row."""

        try:
            value_digests = tuple(proprio_value_sha256s)
        except TypeError as exc:
            raise TypeError(
                "proprio_value_sha256s must be a finite sequence"
            ) from exc
        if len(value_digests) != len(self.chunks):
            raise LayoutContractError(
                LayoutReasonCode.PROPRIO_BOUNDARY_MISMATCH,
                "proprio value digest count differs from layout chunks",
            )
        for index, value_digest in enumerate(value_digests):
            _require_sha256(
                value_digest,
                f"proprio_value_sha256s[{index}]",
            )
        digest = self.layout_instance_digest
        return tuple(
            SelectedProprioBinding(
                layout_instance_digest=digest,
                chunk_id=chunk.chunk_id,
                raw_index=chunk.proprio_raw_index,
                timestamp=chunk.proprio_timestamp,
                source_receipt_digest=(
                    chunk.proprio_source_receipt_digest
                ),
                proprio_value_sha256=value_digests[chunk.chunk_id],
            )
            for chunk in self.chunks
        )

    def verify_instance_digest(self, expected_sha256: str) -> None:
        expected_sha256 = _require_sha256(
            expected_sha256,
            "expected_layout_instance_digest",
        )
        if self.layout_instance_digest != expected_sha256:
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_INSTANCE_DIGEST_MISMATCH,
                "layout instance digest differs from the expected digest",
            )


def validate_cach_bootstrap_config(config: Mapping[str, Any]) -> None:
    """Validate the CACH-only bootstrap schema without importing OmegaConf."""

    if not isinstance(config, Mapping):
        raise TypeError("architecture config must be a mapping")
    if "ar_observed_prefix_chunks" in config:
        raise LayoutContractError(
            LayoutReasonCode.LEGACY_AR_PREFIX_KEY_PRESENT,
            "legacy ar_observed_prefix_chunks is forbidden even when its value is 0",
        )
    if config.get("episode_bootstrap") != "first_frame_pinned":
        raise LayoutContractError(
            LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
            "episode_bootstrap must be 'first_frame_pinned'",
        )
    observed = config.get("observed_prefix_chunks")
    if type(observed) is not int or observed != 0:
        raise LayoutContractError(
            LayoutReasonCode.BOOTSTRAP_CONTRACT_MISMATCH,
            "observed_prefix_chunks must be the plain integer 0",
        )


def _validate_timebase_shape(
    proof: RowTimebaseProof,
    *,
    spec: ChunkActionLayoutSpec,
    valid_raw_count: int,
    valid_video_count: int,
    valid_action_count: int,
) -> None:
    if len(proof.observation_timestamps) != valid_raw_count:
        raise LayoutContractError(
            LayoutReasonCode.DROPPED_OR_DUPLICATED_STEP,
            "observation timestamp count differs from valid_raw_count",
        )
    if len(proof.action_destination_timestamps) != valid_action_count:
        raise LayoutContractError(
            LayoutReasonCode.DROPPED_OR_DUPLICATED_STEP,
            "action destination timestamp count differs from valid_action_count",
        )
    if len(proof.action_destination_state_indices) != valid_action_count:
        raise LayoutContractError(
            LayoutReasonCode.AMBIGUOUS_ACTION_EFFECTIVE_TIME,
            "every valid action must name one destination state",
        )
    if len(proof.sampled_video_raw_indices) != valid_video_count:
        raise LayoutContractError(
            LayoutReasonCode.LAYOUT_GRID_MISMATCH,
            "sampled-video index count differs from the valid video mask",
        )
    expected_sampled = tuple(
        range(0, valid_raw_count, spec.video_stride)
    )
    if proof.sampled_video_raw_indices != expected_sampled:
        raise LayoutContractError(
            LayoutReasonCode.DROPPED_OR_DUPLICATED_STEP,
            "sampled video raw indices differ from range(0,N,video_stride)",
        )
    _validate_strictly_increasing(
        proof.observation_timestamps,
        "observation_timestamps",
    )
    _validate_strictly_increasing(
        proof.action_destination_timestamps,
        "action_destination_timestamps",
    )


def _latent_action_bounds_from_timestamps(
    proof: RowTimebaseProof,
    *,
    spec: ChunkActionLayoutSpec,
    valid_latent_count: int,
) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = [(0, 0)]
    cursor = 0
    action_times = proof.action_destination_timestamps
    video_indices = proof.sampled_video_raw_indices
    observation_times = proof.observation_timestamps
    for latent_index in range(1, valid_latent_count):
        left_video_index = (latent_index - 1) * spec.temporal_compression
        right_video_index = latent_index * spec.temporal_compression
        left_time = observation_times[video_indices[left_video_index]]
        right_time = observation_times[video_indices[right_video_index]]
        if right_time <= left_time:
            raise LayoutContractError(
                LayoutReasonCode.NON_MONOTONIC_TIMESTAMP,
                "latent ownership boundaries are not strictly increasing",
            )
        start = cursor
        while cursor < len(action_times) and action_times[cursor] <= right_time:
            if action_times[cursor] <= left_time:
                raise LayoutContractError(
                    LayoutReasonCode.ACTION_COVERAGE_OVERLAP,
                    "action destination falls on or before the latent left boundary",
                )
            cursor += 1
        spans.append((start, cursor))
    if cursor != len(action_times):
        raise LayoutContractError(
            LayoutReasonCode.ACTION_COVERAGE_GAP,
            "one or more valid actions fall outside all latent ownership intervals",
        )
    return tuple(spans)


def _build_layout(
    *,
    spec: ChunkActionLayoutSpec,
    proof: RowTimebaseProof,
    row_start_raw_index: int,
    valid_raw_count: int,
    video_valid_mask: Sequence[Any],
    action_valid_mask: Sequence[Any],
    synthetic_test: bool,
) -> ChunkActionLayout:
    if not isinstance(spec, ChunkActionLayoutSpec):
        raise TypeError("spec must be a ChunkActionLayoutSpec")
    if not isinstance(proof, RowTimebaseProof):
        raise TypeError("proof must be a RowTimebaseProof")
    if synthetic_test:
        if not spec.synthetic_test_only:
            raise LayoutContractError(
                LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
                "synthetic builder requires a synthetic-only layout spec",
            )
        proof.validate_for_synthetic_test()
    else:
        if spec.synthetic_test_only:
            raise LayoutContractError(
                LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
                "a synthetic-only layout spec cannot enter production",
            )
        proof.validate_for_production()

    row_start_raw_index = _require_plain_int(
        row_start_raw_index,
        "row_start_raw_index",
        minimum=0,
    )
    if row_start_raw_index != 0:
        raise LayoutContractError(
            LayoutReasonCode.NONZERO_EPISODE_ROW_START,
            "CACH v0 rows must begin at episode raw index 0",
            context={"row_start_raw_index": row_start_raw_index},
        )
    valid_raw_count = _require_plain_int(
        valid_raw_count,
        "valid_raw_count",
        minimum=1,
    )
    video_mask = _strict_bool_tuple(video_valid_mask, "video_valid_mask")
    action_mask = _strict_bool_tuple(action_valid_mask, "action_valid_mask")
    valid_video_count = _validate_contiguous_prefix(
        video_mask,
        "video_valid_mask",
    )
    valid_action_count = _validate_contiguous_prefix(
        action_mask,
        "action_valid_mask",
    )
    if valid_video_count < 1:
        raise LayoutContractError(
            LayoutReasonCode.PAD_MARKED_VALID,
            "the episode anchor video frame must be valid",
        )
    if valid_action_count != valid_raw_count - 1:
        raise LayoutContractError(
            LayoutReasonCode.ACTION_COVERAGE_GAP,
            "valid action count must equal valid_raw_count-1 under t+1 indexing",
        )
    if not proof.equal_rate_proven:
        raise LayoutContractError(
            LayoutReasonCode.AMBIGUOUS_ACTION_EFFECTIVE_TIME,
            "CACH v0 requires an externally proven equal observation/action rate",
        )
    if (valid_raw_count - 1) % spec.video_stride != 0:
        raise LayoutContractError(
            LayoutReasonCode.LAYOUT_GRID_MISMATCH,
            "the valid row endpoint is not on the video-stride grid",
        )
    expected_valid_video = (
        (valid_raw_count - 1) // spec.video_stride
    ) + 1
    if valid_video_count != expected_valid_video:
        raise LayoutContractError(
            LayoutReasonCode.LAYOUT_GRID_MISMATCH,
            "video mask does not match the registered raw-frame sampling grid",
        )
    if (
        valid_video_count - 1
    ) % spec.temporal_compression != 0:
        raise LayoutContractError(
            LayoutReasonCode.LAYOUT_GRID_MISMATCH,
            "valid video endpoint is not on the causal-VAE temporal grid",
        )
    valid_latent_count = 1 + (
        (valid_video_count - 1) // spec.temporal_compression
    )

    _validate_timebase_shape(
        proof,
        spec=spec,
        valid_raw_count=valid_raw_count,
        valid_video_count=valid_video_count,
        valid_action_count=valid_action_count,
    )
    destination_indices = proof.action_destination_state_indices
    if destination_indices != tuple(range(1, valid_raw_count)):
        raise LayoutContractError(
            LayoutReasonCode.AMBIGUOUS_ACTION_EFFECTIVE_TIME,
            "action token t must name destination state t+1",
        )
    if any(
        proof.action_destination_timestamps[index]
        != proof.observation_timestamps[state_index]
        for index, state_index in enumerate(destination_indices)
    ):
        raise LayoutContractError(
            LayoutReasonCode.AMBIGUOUS_ACTION_EFFECTIVE_TIME,
            "action destination timestamps do not match destination states",
        )

    latent_bounds = _latent_action_bounds_from_timestamps(
        proof,
        spec=spec,
        valid_latent_count=valid_latent_count,
    )
    rate = spec.temporal_compression * spec.video_stride
    expected_bounds = tuple(
        (0, 0) if latent == 0 else ((latent - 1) * rate, latent * rate)
        for latent in range(valid_latent_count)
    )
    if latent_bounds != expected_bounds:
        raise LayoutContractError(
            LayoutReasonCode.DROPPED_OR_DUPLICATED_STEP,
            "timestamp-derived spans differ from the equal-rate simplification",
        )
    if latent_bounds[0] != (0, 0):
        raise LayoutContractError(
            LayoutReasonCode.ACTION_COVERAGE_OVERLAP,
            "latent 0 must own no action tokens",
        )

    latent_spans = tuple(
        LatentActionSpan(
            latent_index=latent,
            action_start=bounds[0],
            action_end=bounds[1],
            anchor_no_action_slot=(latent == 0),
        )
        for latent, bounds in enumerate(latent_bounds)
    )
    chunks: list[ChunkLayout] = []
    action_cursor = 0
    chunk_count = (
        valid_latent_count + spec.frame_chunk_size - 1
    ) // spec.frame_chunk_size
    for chunk_id in range(chunk_count):
        latent_start = chunk_id * spec.frame_chunk_size
        latent_end = min(
            (chunk_id + 1) * spec.frame_chunk_size,
            valid_latent_count,
        )
        chunk_latent_spans = latent_spans[latent_start:latent_end]
        action_start = chunk_latent_spans[0].action_start
        action_end = chunk_latent_spans[-1].action_end
        if action_start != action_cursor:
            code = (
                LayoutReasonCode.ACTION_COVERAGE_GAP
                if action_start > action_cursor
                else LayoutReasonCode.ACTION_COVERAGE_OVERLAP
            )
            raise LayoutContractError(
                code,
                "chunk action ownership does not continue from the prior cursor",
            )

        if chunk_id == 0:
            proprio_raw_index = 0
        else:
            if action_start <= 0:
                raise LayoutContractError(
                    LayoutReasonCode.PROPRIO_BOUNDARY_MISMATCH,
                    "continuation chunk has no preceding committed action",
                )
            proprio_raw_index = destination_indices[action_start - 1]
        if proprio_raw_index >= valid_raw_count:
            raise LayoutContractError(
                LayoutReasonCode.FUTURE_PROPRIO_VISIBLE,
                "selected proprio state is outside the valid observed prefix",
            )
        proprio_timestamp = proof.observation_timestamps[proprio_raw_index]
        prediction_boundary_timestamp = (
            proof.observation_timestamps[0]
            if chunk_id == 0
            else proof.action_destination_timestamps[action_start - 1]
        )
        if proprio_timestamp > prediction_boundary_timestamp:
            raise LayoutContractError(
                LayoutReasonCode.FUTURE_PROPRIO_VISIBLE,
                "selected proprio timestamp exceeds the prediction boundary",
            )

        latent_valid_count = latent_end - latent_start
        latent_valid_mask = tuple(
            index < latent_valid_count
            for index in range(spec.frame_chunk_size)
        )
        action_count = action_end - action_start
        action_slot_capacity = (
            spec.frame_chunk_size - 1
            if chunk_id == 0
            else spec.frame_chunk_size
        ) * rate
        if action_count > action_slot_capacity:
            raise LayoutContractError(
                LayoutReasonCode.ACTION_COVERAGE_OVERLAP,
                "valid action interval exceeds the registered chunk capacity",
            )
        action_valid_for_chunk = (
            (True,) * action_count
            + (False,) * (action_slot_capacity - action_count)
        )
        chunks.append(
            ChunkLayout(
                chunk_id=chunk_id,
                latent_start=latent_start,
                latent_end=latent_end,
                latent_valid_mask=latent_valid_mask,
                action_start=action_start,
                action_end=action_end,
                action_slot_capacity=action_slot_capacity,
                action_valid_mask=action_valid_for_chunk,
                latent_action_spans=chunk_latent_spans,
                proprio_raw_index=proprio_raw_index,
                proprio_timestamp=proprio_timestamp,
                proprio_source_receipt_digest=(
                    proof.proprio_source_receipt_digest
                ),
                action_rope_start=action_start,
                action_rope_end=action_end,
                is_bootstrap=(chunk_id == 0),
                is_partial_tail=(
                    latent_valid_count < spec.frame_chunk_size
                ),
                anchor_no_action_slot=(chunk_id == 0),
            )
        )
        action_cursor = action_end

    if action_cursor != valid_action_count:
        raise LayoutContractError(
            LayoutReasonCode.ACTION_COVERAGE_GAP,
            "final action cursor does not cover every valid action",
        )
    latent_union = tuple(
        latent
        for chunk in chunks
        for latent in range(chunk.latent_start, chunk.latent_end)
    )
    if latent_union != tuple(range(valid_latent_count)):
        raise LayoutContractError(
            LayoutReasonCode.SILENT_VALID_TAIL_DROP,
            "chunk grid does not cover every valid latent exactly once",
        )
    for chunk in chunks:
        if (
            chunk.action_rope_start != chunk.action_start
            or chunk.action_rope_end != chunk.action_end
        ):
            raise LayoutContractError(
                LayoutReasonCode.ACTION_ROPE_CURSOR_MISMATCH,
                "Action RoPE differs from cumulative valid action ownership",
            )

    coverage_receipt_digest = canonical_sha256(
        {
            "invariants": tuple(f"I{index}" for index in range(1, 17)),
            "latent_union": latent_union,
            "action_union": tuple(range(valid_action_count)),
            "latent_bounds": latent_bounds,
            "proprio_indices": tuple(
                chunk.proprio_raw_index for chunk in chunks
            ),
            "timebase_proof_digest": proof.proof_digest,
        }
    )
    padding_receipt_digest = canonical_sha256(
        {
            "action_valid_mask": action_mask,
            "video_valid_mask": video_mask,
            "chunk_latent_valid_masks": tuple(
                chunk.latent_valid_mask for chunk in chunks
            ),
        }
    )
    return ChunkActionLayout(
        synthetic_test_only=synthetic_test,
        layout_spec_sha256=spec.layout_spec_sha256,
        source_row_digest=proof.source_row_digest,
        episode_id_digest=proof.episode_id_digest,
        row_start_raw_index=row_start_raw_index,
        timebase_manifest_digest=proof.timebase_manifest_digest,
        timestamp_verification_receipt_digest=(
            proof.verification_receipt_digest
        ),
        frame_chunk_size=spec.frame_chunk_size,
        temporal_compression=spec.temporal_compression,
        video_stride=spec.video_stride,
        equal_rate_proven=proof.equal_rate_proven,
        action_tokens_per_non_anchor_latent=rate,
        valid_raw_count=valid_raw_count,
        valid_video_count=valid_video_count,
        valid_latent_count=valid_latent_count,
        valid_action_count=valid_action_count,
        chunks=tuple(chunks),
        coverage_receipt_digest=coverage_receipt_digest,
        padding_receipt_digest=padding_receipt_digest,
    )


def build_chunk_action_layout(
    *,
    spec: ChunkActionLayoutSpec,
    proof: RowTimebaseProof,
    row_start_raw_index: int,
    valid_raw_count: int,
    video_valid_mask: Sequence[Any],
    action_valid_mask: Sequence[Any],
) -> ChunkActionLayout:
    """Build a production layout from an externally verified manifest proof."""

    del (
        spec,
        proof,
        row_start_raw_index,
        valid_raw_count,
        video_valid_mask,
        action_valid_mask,
    )
    raise LayoutContractError(
        LayoutReasonCode.MISSING_VERIFIED_ROW_TIMEBASE,
        "Stage 1 has no production timebase-verifier capability",
    )


def build_synthetic_chunk_action_layout_for_tests(
    *,
    spec: ChunkActionLayoutSpec,
    proof: RowTimebaseProof,
    row_start_raw_index: int,
    valid_raw_count: int,
    video_valid_mask: Sequence[Any],
    action_valid_mask: Sequence[Any],
) -> ChunkActionLayout:
    """Build a synthetic layout through a test-only, visibly named API."""

    return _build_layout(
        spec=spec,
        proof=proof,
        row_start_raw_index=row_start_raw_index,
        valid_raw_count=valid_raw_count,
        video_valid_mask=video_valid_mask,
        action_valid_mask=action_valid_mask,
        synthetic_test=True,
    )


__all__ = [
    "ChunkActionLayout",
    "ChunkActionLayoutSpec",
    "ChunkLayout",
    "LatentActionSpan",
    "LayoutContractError",
    "LayoutReasonCode",
    "RowTimebaseProof",
    "SelectedProprioBinding",
    "build_chunk_action_layout",
    "build_synthetic_chunk_action_layout_for_tests",
    "canonical_json_bytes",
    "canonical_proprio_row_sha256",
    "canonical_sha256",
    "synthetic_equal_rate_proof",
    "synthetic_layout_spec",
    "validate_cach_bootstrap_config",
]
