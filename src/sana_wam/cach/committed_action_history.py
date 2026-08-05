"""Immutable, source-authorized committed-action history for CACH.

The temporal cache is the numerical summary of earlier paired commits, while
this module preserves their exact action-byte identity.  A history contains
only valid action prefixes.  Fixed-capacity padding is validated and dropped
at the append boundary.

This module deliberately does not import the cache manager.  The manager owns
and publishes a history together with its live temporal state; callers receive
detached read views only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor


_HISTORY_SCHEMA = "cach.committed_action_history.v1"
_TENSOR_DIGEST_SCHEMA = b"cach.tensor.v1\0"


class CommittedActionHistoryError(RuntimeError):
    """Fail-closed committed-history error with a stable reason code."""

    def __init__(self, code: str, message: str):
        if not isinstance(code, str) or not code:
            raise ValueError("history error code must be non-empty")
        self.code = code
        super().__init__(f"{code}: {message}")


class CommittedActionSource(str, Enum):
    TEACHER_FORCING = "teacher_forcing_dataset_pair"
    DEPLOY_APPLIED_ACK = "deploy_applied_ack"


def _fail(code: str, message: str) -> None:
    raise CommittedActionHistoryError(code, message)


def _plain_uint(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        _fail("ACTION_HISTORY_SCHEMA_MISMATCH", f"{name} must be a plain uint")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(
            "ACTION_HISTORY_SCHEMA_MISMATCH",
            f"{name} must be a positive plain integer",
        )
    return value


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("ACTION_HISTORY_SCHEMA_MISMATCH", f"{name} must be non-empty")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(
            "ACTION_HISTORY_SCHEMA_MISMATCH",
            f"{name} must be a lowercase SHA256",
        )
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
        _fail(
            "ACTION_HISTORY_SCHEMA_MISMATCH",
            f"value is not canonical-JSON encodable: {exc}",
        )
    raise AssertionError("unreachable")


def _dtype_name(dtype: torch.dtype) -> str:
    value = str(dtype)
    return value[len("torch.") :] if value.startswith("torch.") else value


def _tensor_bytes(tensor: Tensor) -> bytes:
    return (
        tensor.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes(order="C")
    )


def committed_action_tensor_digest(tensor: Tensor) -> str:
    """Return the canonical CACH tensor digest for exact action bytes."""

    if not isinstance(tensor, Tensor) or tensor.layout != torch.strided:
        _fail(
            "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
            "action history requires a strided torch tensor",
        )
    if tensor.device.type == "meta":
        _fail(
            "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
            "meta actions have no canonical bytes",
        )
    metadata = _canonical_json_bytes(
        {
            "device_type": tensor.device.type,
            "dtype": _dtype_name(tensor.dtype),
            "shape": list(tensor.shape),
        }
    )
    material = (
        _TENSOR_DIGEST_SCHEMA
        + len(metadata).to_bytes(8, "big")
        + metadata
        + _tensor_bytes(tensor)
    )
    return hashlib.sha256(material).hexdigest()


def _runtime_fingerprint(tensor: Tensor) -> tuple[object, ...]:
    storage = tensor.untyped_storage()
    return (
        id(tensor),
        storage.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor._version,
        committed_action_tensor_digest(tensor),
    )


def _normalize_source(value: object) -> CommittedActionSource:
    raw = value.value if isinstance(value, Enum) else value
    try:
        source = CommittedActionSource(raw)
    except (TypeError, ValueError) as exc:
        raise CommittedActionHistoryError(
            "ACTION_HISTORY_SOURCE_UNAUTHORIZED",
            "unknown committed-action source",
        ) from exc
    if source is not CommittedActionSource.TEACHER_FORCING:
        _fail(
            "ACTION_HISTORY_SOURCE_UNAUTHORIZED",
            "deploy history remains blocked on the production ACK seam",
        )
    return source


@dataclass(frozen=True)
class CommittedActionSpan:
    """One contiguous, valid-only committed action span."""

    chunk_id: int
    action_start: int
    action_end_exclusive: int
    commit_source: CommittedActionSource
    source_proof_digest: str
    commit_id: str
    actions_digest: str
    _actions: Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _plain_uint(self.chunk_id, "chunk_id")
        _plain_uint(self.action_start, "action_start")
        _plain_uint(self.action_end_exclusive, "action_end_exclusive")
        if self.action_end_exclusive <= self.action_start:
            _fail(
                "ACTION_HISTORY_COVERAGE_GAP",
                "committed action span must be non-empty",
            )
        if self.commit_source is not CommittedActionSource.TEACHER_FORCING:
            _fail(
                "ACTION_HISTORY_SOURCE_UNAUTHORIZED",
                "only teacher-forcing history is commissioned",
            )
        _sha256(self.source_proof_digest, "source_proof_digest")
        _non_empty(self.commit_id, "commit_id")
        _sha256(self.actions_digest, "actions_digest")
        if not isinstance(self._actions, Tensor) or self._actions.ndim != 3:
            _fail(
                "ACTION_HISTORY_SCHEMA_MISMATCH",
                "span actions must have shape [B, valid_actions, A]",
            )
        if self._actions.shape[1] != self.action_end_exclusive - self.action_start:
            _fail(
                "ACTION_HISTORY_COVERAGE_GAP",
                "span tensor length differs from its half-open interval",
            )
        if committed_action_tensor_digest(self._actions) != self.actions_digest:
            _fail(
                "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                "span tensor differs from its exact digest",
            )

    def clone_actions(self) -> Tensor:
        return self._actions.detach().clone(memory_format=torch.contiguous_format)

    def assert_integrity(self) -> None:
        if committed_action_tensor_digest(self._actions) != self.actions_digest:
            _fail(
                "ACTION_HISTORY_MUTATION",
                "stored committed action bytes changed after publication",
            )

    def to_manifest(self) -> dict[str, object]:
        return {
            "action_end_exclusive": self.action_end_exclusive,
            "action_start": self.action_start,
            "actions_digest": self.actions_digest,
            "chunk_id": self.chunk_id,
            "commit_id": self.commit_id,
            "commit_source": self.commit_source.value,
            "source_proof_digest": self.source_proof_digest,
        }


@dataclass(frozen=True)
class CommittedActionHistoryView:
    """Detached exact prefix supplied to one staging or denoise operation."""

    episode_id: str
    episode_epoch: int
    layout_spec_sha256: str
    layout_instance_digest: str
    chunk_id: int
    cursor: int
    chunk_ids: tuple[int, ...]
    batch_size: int
    action_dim: int
    dtype: torch.dtype
    device: torch.device
    history_manifest_digest: str
    committed_actions_digest: str | None
    _actions: Tensor = field(repr=False, compare=False)
    _initial_fingerprint: tuple[object, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _non_empty(self.episode_id, "episode_id")
        _plain_uint(self.episode_epoch, "episode_epoch")
        _sha256(self.layout_spec_sha256, "layout_spec_sha256")
        _sha256(self.layout_instance_digest, "layout_instance_digest")
        _plain_uint(self.chunk_id, "chunk_id")
        _plain_uint(self.cursor, "cursor")
        if self.chunk_ids != tuple(range(self.chunk_id)):
            _fail(
                "ACTION_HISTORY_COVERAGE_GAP",
                "history view chunk ids must be the exact prior prefix",
            )
        _positive_int(self.batch_size, "batch_size")
        _positive_int(self.action_dim, "action_dim")
        _sha256(self.history_manifest_digest, "history_manifest_digest")
        if tuple(self._actions.shape) != (
            self.batch_size,
            self.cursor,
            self.action_dim,
        ):
            _fail(
                "ACTION_HISTORY_COVERAGE_GAP",
                "history view does not cover exact [0, action_start)",
            )
        if self._actions.dtype != self.dtype or self._actions.device != self.device:
            _fail(
                "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                "history view dtype/device metadata differs",
            )
        observed = committed_action_tensor_digest(self._actions)
        if self.cursor == 0:
            if self.committed_actions_digest is not None:
                _fail(
                    "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                    "empty prefix must not claim a committed tensor digest",
                )
        elif observed != self.committed_actions_digest:
            _fail(
                "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                "history view differs from its committed tensor digest",
            )
        if _runtime_fingerprint(self._actions) != self._initial_fingerprint:
            _fail(
                "ACTION_HISTORY_MUTATION",
                "history view was not initialized from its current tensor",
            )

    @property
    def action_cursor(self) -> int:
        return self.cursor

    @property
    def action_start(self) -> int:
        return 0

    @property
    def action_end_exclusive(self) -> int:
        return self.cursor

    def clone_actions(self) -> Tensor:
        self.assert_unchanged()
        return self._actions.detach().clone(memory_format=torch.contiguous_format)

    def assert_unchanged(self) -> None:
        if _runtime_fingerprint(self._actions) != self._initial_fingerprint:
            _fail(
                "ACTION_HISTORY_MUTATION",
                "staging changed committed-history bindings or bytes",
            )


@dataclass(frozen=True)
class CommittedActionHistory:
    """Persistent, immutable per-episode committed-action history."""

    episode_id: str
    episode_epoch: int
    layout_spec_sha256: str
    layout_instance_digest: str
    batch_size: int
    action_dim: int
    dtype: torch.dtype
    device: torch.device
    spans: tuple[CommittedActionSpan, ...] = ()

    def __post_init__(self) -> None:
        _non_empty(self.episode_id, "episode_id")
        _plain_uint(self.episode_epoch, "episode_epoch")
        _sha256(self.layout_spec_sha256, "layout_spec_sha256")
        _sha256(self.layout_instance_digest, "layout_instance_digest")
        _positive_int(self.batch_size, "batch_size")
        _positive_int(self.action_dim, "action_dim")
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            _fail(
                "ACTION_HISTORY_SCHEMA_MISMATCH",
                "history dtype must be a floating torch dtype",
            )
        object.__setattr__(self, "device", torch.device(self.device))
        if self.device.type == "meta":
            _fail(
                "ACTION_HISTORY_SCHEMA_MISMATCH",
                "history cannot use the meta device",
            )
        if not isinstance(self.spans, tuple) or any(
            not isinstance(span, CommittedActionSpan) for span in self.spans
        ):
            _fail(
                "ACTION_HISTORY_SCHEMA_MISMATCH",
                "history spans must be a tuple of CommittedActionSpan",
            )
        cursor = 0
        for expected_chunk, span in enumerate(self.spans):
            span.assert_integrity()
            if span.chunk_id != expected_chunk or span.action_start != cursor:
                _fail(
                    "ACTION_HISTORY_COVERAGE_GAP",
                    "history chunk ids and action intervals must be contiguous",
                )
            if tuple(span._actions.shape[:1] + span._actions.shape[2:]) != (
                self.batch_size,
                self.action_dim,
            ):
                _fail(
                    "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                    "history span batch/action width differs",
                )
            if span._actions.dtype != self.dtype or span._actions.device != self.device:
                _fail(
                    "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                    "history span dtype/device differs",
                )
            cursor = span.action_end_exclusive

    @classmethod
    def empty(
        cls,
        *,
        episode_id: str,
        episode_epoch: int,
        layout_spec_sha256: str,
        layout_instance_digest: str,
        batch_size: int,
        action_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "CommittedActionHistory":
        return cls(
            episode_id=episode_id,
            episode_epoch=episode_epoch,
            layout_spec_sha256=layout_spec_sha256,
            layout_instance_digest=layout_instance_digest,
            batch_size=batch_size,
            action_dim=action_dim,
            dtype=dtype,
            device=torch.device(device),
        )

    @property
    def action_cursor(self) -> int:
        return 0 if not self.spans else self.spans[-1].action_end_exclusive

    @property
    def committed_through_chunk(self) -> int | None:
        return None if not self.spans else self.spans[-1].chunk_id

    @property
    def manifest_digest(self) -> str:
        self.assert_integrity()
        return hashlib.sha256(_canonical_json_bytes(self.to_manifest())).hexdigest()

    def assert_integrity(self) -> None:
        cursor = 0
        for expected_chunk, span in enumerate(self.spans):
            span.assert_integrity()
            if span.chunk_id != expected_chunk or span.action_start != cursor:
                _fail(
                    "ACTION_HISTORY_MUTATION",
                    "published history structure is no longer contiguous",
                )
            cursor = span.action_end_exclusive

    def to_manifest(self) -> dict[str, object]:
        return {
            "action_cursor": self.action_cursor,
            "action_dim": self.action_dim,
            "batch_size": self.batch_size,
            "committed_through_chunk": self.committed_through_chunk,
            "device": str(self.device),
            "dtype": _dtype_name(self.dtype),
            "episode_epoch": self.episode_epoch,
            "episode_id": self.episode_id,
            "layout_instance_digest": self.layout_instance_digest,
            "layout_spec_sha256": self.layout_spec_sha256,
            "schema": _HISTORY_SCHEMA,
            "spans": [span.to_manifest() for span in self.spans],
        }

    def clone(self) -> "CommittedActionHistory":
        return CommittedActionHistory(
            episode_id=self.episode_id,
            episode_epoch=self.episode_epoch,
            layout_spec_sha256=self.layout_spec_sha256,
            layout_instance_digest=self.layout_instance_digest,
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            dtype=self.dtype,
            device=self.device,
            spans=tuple(
                CommittedActionSpan(
                    chunk_id=span.chunk_id,
                    action_start=span.action_start,
                    action_end_exclusive=span.action_end_exclusive,
                    commit_source=span.commit_source,
                    source_proof_digest=span.source_proof_digest,
                    commit_id=span.commit_id,
                    actions_digest=span.actions_digest,
                    _actions=span.clone_actions(),
                )
                for span in self.spans
            ),
        )

    def append(
        self,
        *,
        chunk_id: int,
        action_start: int,
        action_end_exclusive: int,
        fixed_slot_actions: Tensor,
        action_valid_mask: Tensor,
        commit_source: CommittedActionSource | str | object,
        source_proof_digest: str,
        commit_id: str,
    ) -> tuple["CommittedActionHistory", bool]:
        """Append one valid prefix, returning ``(history, was_duplicate)``."""

        self.assert_integrity()
        source = _normalize_source(commit_source)
        _plain_uint(chunk_id, "chunk_id")
        _plain_uint(action_start, "action_start")
        _plain_uint(action_end_exclusive, "action_end_exclusive")
        _sha256(source_proof_digest, "source_proof_digest")
        _non_empty(commit_id, "commit_id")
        if not isinstance(fixed_slot_actions, Tensor) or fixed_slot_actions.ndim != 3:
            _fail(
                "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                "fixed-slot actions must have shape [B, slots, A]",
            )
        if tuple(fixed_slot_actions.shape[::2]) != (
            self.batch_size,
            self.action_dim,
        ):
            _fail(
                "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                "fixed-slot action batch/width differs from history",
            )
        if (
            fixed_slot_actions.dtype != self.dtype
            or fixed_slot_actions.device != self.device
        ):
            _fail(
                "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                "fixed-slot action dtype/device differs from history",
            )
        if not isinstance(action_valid_mask, Tensor):
            _fail(
                "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                "action_valid_mask must be a torch tensor",
            )
        if (
            action_valid_mask.dtype is not torch.bool
            or tuple(action_valid_mask.shape) != tuple(fixed_slot_actions.shape[:2])
            or action_valid_mask.device != fixed_slot_actions.device
        ):
            _fail(
                "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                "action mask must be boolean [B, slots] on the action device",
            )
        valid_count = action_end_exclusive - action_start
        if valid_count <= 0 or valid_count > fixed_slot_actions.shape[1]:
            _fail(
                "ACTION_HISTORY_COVERAGE_GAP",
                "append interval differs from fixed-slot capacity",
            )
        expected_mask = torch.zeros_like(action_valid_mask)
        expected_mask[:, :valid_count] = True
        if not torch.equal(action_valid_mask, expected_mask):
            _fail(
                "ACTION_HISTORY_COVERAGE_GAP",
                "action mask must be the exact valid-prefix interval",
            )
        invalid_values = fixed_slot_actions[:, valid_count:, :]
        if invalid_values.numel() and bool(invalid_values.detach().count_nonzero()):
            _fail(
                "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                "fixed-slot padding must be exact zero",
            )
        if not bool(torch.isfinite(fixed_slot_actions.detach()).all()):
            _fail(
                "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                "committed actions contain NaN or Inf",
            )
        valid_actions = fixed_slot_actions[:, :valid_count, :].detach().clone(
            memory_format=torch.contiguous_format
        )
        actions_digest = committed_action_tensor_digest(valid_actions)

        for prior in self.spans:
            if prior.commit_id != commit_id:
                continue
            identical = (
                prior.chunk_id == chunk_id
                and prior.action_start == action_start
                and prior.action_end_exclusive == action_end_exclusive
                and prior.commit_source is source
                and prior.source_proof_digest == source_proof_digest
                and prior.actions_digest == actions_digest
            )
            if identical:
                return self, True
            _fail(
                "ACTION_HISTORY_DUPLICATE_CONFLICT",
                "commit_id was already used for different action bytes or identity",
            )

        if chunk_id != len(self.spans) or action_start != self.action_cursor:
            _fail(
                "ACTION_HISTORY_COVERAGE_GAP",
                "append must be the next contiguous chunk/action interval",
            )
        span = CommittedActionSpan(
            chunk_id=chunk_id,
            action_start=action_start,
            action_end_exclusive=action_end_exclusive,
            commit_source=source,
            source_proof_digest=source_proof_digest,
            commit_id=commit_id,
            actions_digest=actions_digest,
            _actions=valid_actions,
        )
        return (
            CommittedActionHistory(
                episode_id=self.episode_id,
                episode_epoch=self.episode_epoch,
                layout_spec_sha256=self.layout_spec_sha256,
                layout_instance_digest=self.layout_instance_digest,
                batch_size=self.batch_size,
                action_dim=self.action_dim,
                dtype=self.dtype,
                device=self.device,
                spans=self.spans + (span,),
            ),
            False,
        )

    def view_for_chunk(
        self, *, chunk_id: int, action_start: int
    ) -> CommittedActionHistoryView:
        _plain_uint(chunk_id, "chunk_id")
        _plain_uint(action_start, "action_start")
        self.assert_integrity()
        if chunk_id != len(self.spans) or action_start != self.action_cursor:
            _fail(
                "ACTION_HISTORY_COVERAGE_GAP",
                "history view must cover exact prefix for the next chunk",
            )
        if self.spans:
            actions = torch.cat(
                [span.clone_actions() for span in self.spans], dim=1
            )
            actions_digest: str | None = committed_action_tensor_digest(actions)
        else:
            actions = torch.empty(
                (self.batch_size, 0, self.action_dim),
                dtype=self.dtype,
                device=self.device,
            )
            actions_digest = None
        fingerprint = _runtime_fingerprint(actions)
        return CommittedActionHistoryView(
            episode_id=self.episode_id,
            episode_epoch=self.episode_epoch,
            layout_spec_sha256=self.layout_spec_sha256,
            layout_instance_digest=self.layout_instance_digest,
            chunk_id=chunk_id,
            cursor=action_start,
            chunk_ids=tuple(span.chunk_id for span in self.spans),
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            dtype=self.dtype,
            device=self.device,
            history_manifest_digest=self.manifest_digest,
            committed_actions_digest=actions_digest,
            _actions=actions,
            _initial_fingerprint=fingerprint,
        )

    def reset(
        self,
        *,
        episode_id: str,
        episode_epoch: int,
        layout_spec_sha256: str,
        layout_instance_digest: str,
    ) -> "CommittedActionHistory":
        return self.empty(
            episode_id=episode_id,
            episode_epoch=episode_epoch,
            layout_spec_sha256=layout_spec_sha256,
            layout_instance_digest=layout_instance_digest,
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            dtype=self.dtype,
            device=self.device,
        )


__all__ = [
    "CommittedActionHistory",
    "CommittedActionHistoryError",
    "CommittedActionHistoryView",
    "CommittedActionSource",
    "CommittedActionSpan",
    "committed_action_tensor_digest",
]
