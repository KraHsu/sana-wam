"""Independent, fail-closed verifier for ``APPLIED_ACTION_ACK`` v1.

This module verifies transport evidence only.  It deliberately has no import or
call edge to ``HybridCacheManager`` and cannot enable a deploy commit.  An
``immutable_ref`` is trusted only when its URI is present in an explicit
authority registry; the local filesystem is treated as mutable input and is
rechecked through one stable ``O_NOFOLLOW`` file descriptor.

``ack_payload_digest`` is defined as SHA256 of UTF-8 JSON produced with
``sort_keys=True``, ``separators=(",", ":")``, and ``allow_nan=False`` after
removing the ``ack_payload_digest`` field itself.  No trailing newline is part
of those canonical bytes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import stat
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

import torch


_SCHEMA = "cach.applied_action_ack.v1"
_TOKEN_ORDER = "layout_action_token_ascending_v1"
_UINT64_MAX = (1 << 64) - 1
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_ROOT_KEYS = frozenset(
    {
        "schema",
        "ack_id",
        "episode_id",
        "episode_epoch",
        "command_id",
        "chunk_id",
        "layout_instance_digest",
        "canonical_applied_tensor_or_immutable_ref",
        "dtype",
        "shape",
        "token_order",
        "action_representation_id",
        "action_units_id",
        "action_values_digest",
        "applied_count",
        "controller_transform_digest",
        "observation_interval_digest",
        "controller_sequence_start",
        "controller_sequence_end_exclusive",
        "controller_monotonic_start_ns",
        "controller_monotonic_end_ns",
        "completion_status",
        "ack_payload_digest",
    }
)
_INLINE_KEYS = frozenset({"kind", "data_base64"})
_IMMUTABLE_REF_KEYS = frozenset(
    {"kind", "uri", "byte_offset", "byte_length", "object_sha256"}
)


class AppliedActionAckError(ValueError):
    """Stable-code rejection from the v1 ACK verifier."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> None:
    raise AppliedActionAckError(code, detail)


def _is_nonempty_string(value: object) -> bool:
    return type(value) is str and bool(value)


def _is_uint64(value: object) -> bool:
    return type(value) is int and 0 <= value <= _UINT64_MAX


def _is_sha256(value: object) -> bool:
    return type(value) is str and _HEX_SHA256.fullmatch(value) is not None


def _canonical_json_without_ack_digest(payload: Mapping[str, Any]) -> bytes:
    value = dict(payload)
    value.pop("ack_payload_digest", None)
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise AppliedActionAckError(
            "ACK_PAYLOAD_NOT_CANONICAL_JSON",
            "payload is not canonical JSON data",
        ) from exc
    return encoded.encode("utf-8")


def compute_ack_payload_digest(payload: Mapping[str, Any]) -> str:
    """Compute the v1 digest after removing ``ack_payload_digest`` itself."""

    if type(payload) is not dict:
        _fail("ACK_SCHEMA_INVALID", "ACK payload must be an exact JSON object")
    return hashlib.sha256(_canonical_json_without_ack_digest(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class AppliedActionAckContext:
    """Live identities and allowlists against which one ACK is checked."""

    episode_id: str
    episode_epoch: int
    pending_command_id: str
    next_chunk_id: int
    layout_instance_digest: str
    applied_count: int
    action_dim: int
    action_representation_id: str
    action_units_id: str
    controller_sequence_start: int
    controller_transform_allowlist: frozenset[str]
    observation_interval_digest: str
    commanded_tensor: torch.Tensor

    def __post_init__(self) -> None:
        if not _is_nonempty_string(self.episode_id):
            _fail("ACK_CONTEXT_INVALID", "episode_id must be non-empty")
        if not _is_uint64(self.episode_epoch):
            _fail("ACK_CONTEXT_INVALID", "episode_epoch must be uint64")
        if not _is_nonempty_string(self.pending_command_id):
            _fail("ACK_CONTEXT_INVALID", "pending_command_id must be non-empty")
        if not _is_uint64(self.next_chunk_id):
            _fail("ACK_CONTEXT_INVALID", "next_chunk_id must be uint64")
        if not _is_sha256(self.layout_instance_digest):
            _fail("ACK_CONTEXT_INVALID", "layout digest must be lowercase SHA256")
        if not _is_uint64(self.applied_count):
            _fail("ACK_CONTEXT_INVALID", "applied_count must be uint64")
        if type(self.action_dim) is not int or self.action_dim <= 0:
            _fail("ACK_CONTEXT_INVALID", "action_dim must be positive")
        if not _is_nonempty_string(self.action_representation_id):
            _fail("ACK_CONTEXT_INVALID", "representation id must be non-empty")
        if not _is_nonempty_string(self.action_units_id):
            _fail("ACK_CONTEXT_INVALID", "units id must be non-empty")
        if not _is_uint64(self.controller_sequence_start):
            _fail("ACK_CONTEXT_INVALID", "controller sequence start must be uint64")
        expected_end = self.controller_sequence_start + self.applied_count
        if expected_end > _UINT64_MAX:
            _fail("ACK_CONTEXT_INVALID", "controller sequence interval overflows")
        if (
            type(self.controller_transform_allowlist) is not frozenset
            or not self.controller_transform_allowlist
            or not all(
                _is_sha256(item) for item in self.controller_transform_allowlist
            )
        ):
            _fail("ACK_CONTEXT_INVALID", "transform allowlist must contain SHA256s")
        if not _is_sha256(self.observation_interval_digest):
            _fail("ACK_CONTEXT_INVALID", "observation interval digest is invalid")
        command = self.commanded_tensor
        if (
            type(command) is not torch.Tensor
            or command.device.type != "cpu"
            or command.dtype is not torch.float32
            or command.ndim != 2
            or tuple(command.shape) != (self.applied_count, self.action_dim)
            or not command.is_contiguous()
            or not bool(torch.isfinite(command).all())
        ):
            _fail(
                "ACK_CONTEXT_INVALID",
                "commanded tensor must be finite contiguous CPU float32 with expected shape",
            )


@dataclass(frozen=True, slots=True)
class ImmutableObjectAuthority:
    """Registry claim for one local object; not a filesystem immutability claim."""

    path: Path
    object_size: int
    object_sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        object.__setattr__(self, "path", path)
        if not path.is_absolute():
            _fail("ACK_AUTHORITY_REGISTRY_INVALID", "authority path must be absolute")
        if not _is_uint64(self.object_size):
            _fail("ACK_AUTHORITY_REGISTRY_INVALID", "object size must be uint64")
        if not _is_sha256(self.object_sha256):
            _fail("ACK_AUTHORITY_REGISTRY_INVALID", "object digest is invalid")


@dataclass(frozen=True, slots=True)
class ImmutableAuthorityRegistry:
    """Exact registered URI to local authority-object mapping."""

    entries: Mapping[str, ImmutableObjectAuthority]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, Mapping):
            _fail("ACK_AUTHORITY_REGISTRY_INVALID", "registry must be a mapping")
        copied: dict[str, ImmutableObjectAuthority] = {}
        for uri, authority in self.entries.items():
            parsed = urlsplit(uri) if type(uri) is str else None
            if (
                parsed is None
                or not parsed.scheme
                or parsed.scheme == "file"
                or not isinstance(authority, ImmutableObjectAuthority)
            ):
                _fail(
                    "ACK_AUTHORITY_REGISTRY_INVALID",
                    "registry requires non-file URIs and typed authority entries",
                )
            copied[uri] = authority
        object.__setattr__(self, "entries", MappingProxyType(copied))

    @classmethod
    def empty(cls) -> "ImmutableAuthorityRegistry":
        return cls(entries={})


class AppliedActionAckReplayRegistry:
    """Process-local idempotency registry used only after all proof checks pass."""

    def __init__(self) -> None:
        self._accepted: dict[str, str] = {}
        self._lock = threading.Lock()

    def record(self, ack_id: str, payload_digest: str) -> bool:
        """Return true for an identical duplicate; reject an id conflict."""

        with self._lock:
            previous = self._accepted.get(ack_id)
            if previous is None:
                self._accepted[ack_id] = payload_digest
                return False
            if previous != payload_digest:
                _fail("ACK_ID_CONFLICT", "ack_id was already bound to another payload")
            return True


@dataclass(frozen=True, slots=True)
class VerifiedAppliedActionAck:
    canonical_applied_tensor: torch.Tensor
    canonical_applied_bytes: bytes
    commanded_differs_from_applied: bool
    ack_id: str
    ack_payload_digest: str
    action_values_digest: str
    is_duplicate: bool


def _require_exact_keys(value: object, expected: frozenset[str], code: str) -> dict:
    if type(value) is not dict or set(value) != expected:
        _fail(code, "JSON object keys differ from the v1 schema")
    return value


def _decode_inline(value: dict, expected_byte_length: int) -> bytes:
    if set(value) != _INLINE_KEYS:
        _fail("ACK_INLINE_SCHEMA_MISMATCH", "inline tensor keys differ")
    encoded = value["data_base64"]
    if isinstance(encoded, (list, tuple)):
        _fail(
            "ACK_JSON_NUMERIC_TENSOR_REJECTED",
            "JSON numeric tensors are not canonical v1 bytes",
        )
    if type(encoded) is not str:
        _fail("ACK_CANONICAL_BYTES_REQUIRED", "inline data_base64 is required")
    try:
        encoded_ascii = encoded.encode("ascii")
        raw = base64.b64decode(encoded_ascii, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise AppliedActionAckError(
            "ACK_INLINE_BASE64_INVALID",
            "inline tensor is not strict base64",
        ) from exc
    if base64.b64encode(raw) != encoded_ascii:
        _fail("ACK_INLINE_BASE64_INVALID", "base64 spelling is not canonical")
    if len(raw) != expected_byte_length:
        _fail("ACK_CANONICAL_BYTE_LENGTH_MISMATCH", "inline byte length differs")
    return raw


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _read_registered_range(
    value: dict,
    expected_byte_length: int,
    registry: ImmutableAuthorityRegistry,
) -> bytes:
    if set(value) != _IMMUTABLE_REF_KEYS:
        _fail("ACK_IMMUTABLE_SCHEMA_MISMATCH", "immutable ref keys differ")
    uri = value["uri"]
    if not _is_nonempty_string(uri):
        _fail("ACK_IMMUTABLE_URI_UNREGISTERED", "immutable URI is missing")
    authority = registry.entries.get(uri)
    if authority is None:
        _fail("ACK_IMMUTABLE_URI_UNREGISTERED", "immutable URI is not registered")
    offset = value["byte_offset"]
    length = value["byte_length"]
    if not _is_uint64(offset) or not _is_uint64(length):
        _fail("ACK_IMMUTABLE_RANGE_INVALID", "byte range must use uint64 values")
    if length != expected_byte_length:
        _fail("ACK_CANONICAL_BYTE_LENGTH_MISMATCH", "ref byte length differs")
    claimed_digest = value["object_sha256"]
    if not _is_sha256(claimed_digest) or claimed_digest != authority.object_sha256:
        _fail("ACK_IMMUTABLE_OBJECT_MISMATCH", "object digest is not authoritative")
    if offset + length > authority.object_size:
        _fail("ACK_IMMUTABLE_SHORT_READ", "registered object cannot cover byte range")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        _fail("ACK_IMMUTABLE_NOFOLLOW_UNAVAILABLE", "O_NOFOLLOW is required")
    path = authority.path
    try:
        path_before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as exc:
        code = "ACK_IMMUTABLE_REF_SYMLINK" if path.is_symlink() else "ACK_IMMUTABLE_OPEN_FAILED"
        raise AppliedActionAckError(code, "cannot open registered authority object") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(path_before.st_mode):
            _fail("ACK_IMMUTABLE_NOT_REGULAR", "authority object is not regular")
        if not _same_file_identity(before, path_before):
            _fail("ACK_IMMUTABLE_OBJECT_CHANGED", "path changed before stable read")
        if before.st_size != authority.object_size:
            _fail("ACK_IMMUTABLE_OBJECT_MISMATCH", "object size differs from registry")

        digest = hashlib.sha256()
        selected: list[bytes] = []
        position = 0
        while position < authority.object_size:
            chunk = os.read(descriptor, min(1024 * 1024, authority.object_size - position))
            if not chunk:
                _fail("ACK_IMMUTABLE_SHORT_READ", "authority object read was short")
            digest.update(chunk)
            chunk_end = position + len(chunk)
            overlap_start = max(position, offset)
            overlap_end = min(chunk_end, offset + length)
            if overlap_start < overlap_end:
                selected.append(
                    chunk[overlap_start - position : overlap_end - position]
                )
            position = chunk_end
        if os.read(descriptor, 1):
            _fail("ACK_IMMUTABLE_OBJECT_CHANGED", "authority object grew during read")
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise AppliedActionAckError(
                "ACK_IMMUTABLE_OBJECT_CHANGED",
                "authority path disappeared during read",
            ) from exc
        if not _same_file_identity(before, after) or not _same_file_identity(
            after, path_after
        ):
            _fail("ACK_IMMUTABLE_OBJECT_CHANGED", "authority object changed during read")
        if digest.hexdigest() != authority.object_sha256:
            _fail("ACK_IMMUTABLE_OBJECT_MISMATCH", "object digest differs from registry")
        raw = b"".join(selected)
        if len(raw) != length:
            _fail("ACK_IMMUTABLE_SHORT_READ", "selected tensor byte range was short")
        return raw
    finally:
        os.close(descriptor)


def _canonical_bytes(
    source: object,
    expected_byte_length: int,
    registry: ImmutableAuthorityRegistry,
) -> bytes:
    if type(source) is not dict:
        _fail("ACK_CANONICAL_BYTES_REQUIRED", "canonical tensor source is required")
    kind = source.get("kind")
    if kind == "inline_tensor":
        return _decode_inline(source, expected_byte_length)
    if kind == "immutable_ref":
        return _read_registered_range(source, expected_byte_length, registry)
    _fail(
        "ACK_CANONICAL_BYTES_REQUIRED",
        "digest-only or unknown canonical tensor source is not accepted",
    )


def _decode_float32_le(raw: bytes, shape: tuple[int, int]) -> torch.Tensor:
    words = struct.iter_unpack("<I", raw)
    if any(word[0] == 0x80000000 for word in words):
        _fail("ACK_ACTION_NEGATIVE_ZERO", "canonical tensor contains negative zero")
    count = shape[0] * shape[1]
    values = struct.unpack(f"<{count}f", raw) if count else ()
    if not all(math.isfinite(value) for value in values):
        _fail("ACK_ACTION_NONFINITE", "canonical tensor contains NaN or Inf")
    return torch.tensor(values, dtype=torch.float32).reshape(shape).contiguous()


def verify_applied_action_ack(
    payload: Mapping[str, Any],
    *,
    context: AppliedActionAckContext,
    replay_registry: AppliedActionAckReplayRegistry,
    immutable_registry: ImmutableAuthorityRegistry | None = None,
) -> VerifiedAppliedActionAck:
    """Verify one ACK without mutating or calling any temporal-cache manager."""

    if not isinstance(context, AppliedActionAckContext):
        _fail("ACK_CONTEXT_INVALID", "typed ACK context is required")
    if not isinstance(replay_registry, AppliedActionAckReplayRegistry):
        _fail("ACK_REPLAY_REGISTRY_REQUIRED", "explicit replay registry is required")
    if immutable_registry is None:
        immutable_registry = ImmutableAuthorityRegistry.empty()
    if not isinstance(immutable_registry, ImmutableAuthorityRegistry):
        _fail("ACK_AUTHORITY_REGISTRY_INVALID", "typed authority registry is required")
    if type(payload) is not dict:
        _fail("ACK_SCHEMA_INVALID", "ACK payload must be an exact JSON object")
    if "canonical_applied_tensor_or_immutable_ref" not in payload:
        _fail(
            "ACTION_EVIDENCE_COMMANDED_ONLY",
            "command evidence cannot substitute for applied action bytes",
        )
    ack = _require_exact_keys(payload, _ROOT_KEYS, "ACK_SCHEMA_KEYS_MISMATCH")
    if ack["schema"] != _SCHEMA:
        _fail("ACK_SCHEMA_VERSION_MISMATCH", "schema/version differs")
    if not _is_nonempty_string(ack["ack_id"]):
        _fail("ACK_SCHEMA_INVALID", "ack_id must be non-empty")

    if (
        ack["episode_id"] != context.episode_id
        or ack["episode_epoch"] != context.episode_epoch
        or not _is_uint64(ack["episode_epoch"])
    ):
        _fail("ACK_EPISODE_MISMATCH", "ACK belongs to another episode or epoch")
    if ack["command_id"] != context.pending_command_id:
        _fail("ACK_COMMAND_UNKNOWN_OR_STALE", "command is not the live pending command")
    if ack["chunk_id"] != context.next_chunk_id or not _is_uint64(ack["chunk_id"]):
        _fail("ACK_CHUNK_MISMATCH", "chunk is not the live next chunk")
    if (
        ack["layout_instance_digest"] != context.layout_instance_digest
        or not _is_sha256(ack["layout_instance_digest"])
    ):
        _fail("ACK_LAYOUT_MISMATCH", "layout instance digest differs")
    if ack["completion_status"] != "fully_applied":
        _fail("ACK_COMPLETION_INCOMPLETE", "controller did not fully apply the interval")

    expected_byte_length = context.applied_count * context.action_dim * 4
    raw = _canonical_bytes(
        ack["canonical_applied_tensor_or_immutable_ref"],
        expected_byte_length,
        immutable_registry,
    )
    if ack["dtype"] != "float32_le":
        _fail("ACK_DTYPE_MISMATCH", "dtype must be float32_le")
    expected_shape = [context.applied_count, context.action_dim]
    shape = ack["shape"]
    if (
        type(shape) is not list
        or len(shape) != 2
        or not all(_is_uint64(item) for item in shape)
        or shape != expected_shape
    ):
        _fail("ACK_SHAPE_MISMATCH", "rank-2 shape differs from layout coverage")
    if ack["token_order"] != _TOKEN_ORDER:
        _fail("ACK_TOKEN_ORDER_MISMATCH", "action token order differs")
    if ack["action_representation_id"] != context.action_representation_id:
        _fail("ACK_REPRESENTATION_MISMATCH", "action representation differs")
    if ack["action_units_id"] != context.action_units_id:
        _fail("ACK_UNITS_MISMATCH", "action units differ")

    tensor = _decode_float32_le(raw, (context.applied_count, context.action_dim))
    observed_values_digest = hashlib.sha256(raw).hexdigest()
    if (
        not _is_sha256(ack["action_values_digest"])
        or ack["action_values_digest"] != observed_values_digest
    ):
        _fail("ACK_ACTION_VALUES_DIGEST_MISMATCH", "action tensor digest differs")
    if ack["applied_count"] != context.applied_count or not _is_uint64(
        ack["applied_count"]
    ):
        _fail("ACK_APPLIED_COUNT_MISMATCH", "applied count differs from layout")

    sequence_start = ack["controller_sequence_start"]
    sequence_end = ack["controller_sequence_end_exclusive"]
    expected_end = context.controller_sequence_start + context.applied_count
    if (
        not _is_uint64(sequence_start)
        or not _is_uint64(sequence_end)
        or sequence_start != context.controller_sequence_start
        or sequence_end != expected_end
    ):
        _fail(
            "ACK_CONTROLLER_SEQUENCE_NONCONTIGUOUS",
            "controller sequence has a gap or overlap",
        )
    monotonic_start = ack["controller_monotonic_start_ns"]
    monotonic_end = ack["controller_monotonic_end_ns"]
    if (
        not _is_uint64(monotonic_start)
        or not _is_uint64(monotonic_end)
        or monotonic_end < monotonic_start
        or (context.applied_count > 0 and monotonic_end == monotonic_start)
    ):
        _fail(
            "ACK_CONTROLLER_MONOTONIC_INTERVAL_INVALID",
            "controller monotonic interval is invalid",
        )
    transform_digest = ack["controller_transform_digest"]
    if (
        not _is_sha256(transform_digest)
        or transform_digest not in context.controller_transform_allowlist
    ):
        _fail(
            "ACK_CONTROLLER_TRANSFORM_UNREGISTERED",
            "controller transform digest is not allowlisted",
        )
    if (
        ack["observation_interval_digest"]
        != context.observation_interval_digest
        or not _is_sha256(ack["observation_interval_digest"])
    ):
        _fail("ACK_OBSERVATION_INTERVAL_MISMATCH", "observation interval differs")

    supplied_payload_digest = ack["ack_payload_digest"]
    observed_payload_digest = compute_ack_payload_digest(ack)
    if (
        not _is_sha256(supplied_payload_digest)
        or supplied_payload_digest != observed_payload_digest
    ):
        _fail("ACK_PAYLOAD_DIGEST_MISMATCH", "ACK payload digest differs")
    duplicate = replay_registry.record(ack["ack_id"], supplied_payload_digest)

    differs = not torch.equal(context.commanded_tensor, tensor)
    return VerifiedAppliedActionAck(
        canonical_applied_tensor=tensor,
        canonical_applied_bytes=raw,
        commanded_differs_from_applied=differs,
        ack_id=ack["ack_id"],
        ack_payload_digest=supplied_payload_digest,
        action_values_digest=observed_values_digest,
        is_duplicate=duplicate,
    )


__all__ = [
    "AppliedActionAckContext",
    "AppliedActionAckError",
    "AppliedActionAckReplayRegistry",
    "ImmutableAuthorityRegistry",
    "ImmutableObjectAuthority",
    "VerifiedAppliedActionAck",
    "compute_ack_payload_digest",
    "verify_applied_action_ack",
]
