"""Additive Stage-2B cache/history transaction manager.

The frozen Stage-1 ``hybrid_cache.py`` remains byte-identical to its immutable
source bundle.  This module composes its typed cache primitives with exact
committed-action history in a new outer state pointer.  Cache state and action
history therefore advance only after one durable Stage-2B receipt, while all
Stage-1, training, loader, deploy, and server authority remains unchanged.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from torch import Tensor

from sana_wam.cach.committed_action_history import (
    CommittedActionHistory,
    CommittedActionHistoryError,
    CommittedActionHistoryView,
)
from sana_wam.cach.staging_variant import (
    CACHStagingVariant,
    history_summary_operator_for_variant,
)
from sana_wam.model.action_chunk_layout import ChunkActionLayout
from sana_wam.model.video_backbone.sana import hybrid_cache as base


_STATE_SCHEMA = "cach.stage2b.hybrid_temporal_state.v1"
_COMMIT_RECEIPT_SCHEMA = "cach.stage2b.paired_commit_receipt.v1"
_RESET_RECEIPT_SCHEMA = "cach.stage2b.cache_reset_receipt.v1"


def _translate_history_error(exc: CommittedActionHistoryError) -> None:
    message = str(exc)
    prefix = f"{exc.code}: "
    if message.startswith(prefix):
        message = message[len(prefix) :]
    raise base.CacheContractError(exc.code, message) from exc


def _history_digest(history: CommittedActionHistory | None) -> str | None:
    if history is None:
        return None
    try:
        return history.manifest_digest
    except CommittedActionHistoryError as exc:
        _translate_history_error(exc)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Stage2BTeacherForcingCommitRequest(base.TeacherForcingCommitRequest):
    """Stage-1 teacher pair plus an optional exact conditioning identity."""

    conditioning_digest: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.conditioning_digest is not None:
            base._require_sha256(self.conditioning_digest, "conditioning_digest")


@dataclass(frozen=True)
class Stage2BTemporalState:
    """One immutable authority pointer containing cache and action history."""

    cache_state: base.HybridTemporalState = field(repr=False, compare=False)
    staging_variant: CACHStagingVariant
    committed_action_history: CommittedActionHistory | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.cache_state, base.HybridTemporalState):
            base._fail("CACHE_SCHEMA_MISMATCH", "Stage-2B cache state has wrong type")
        if not isinstance(self.staging_variant, CACHStagingVariant):
            base._fail("CACHE_SCHEMA_MISMATCH", "Stage-2B variant has wrong type")
        state = self.cache_state
        if state.revision == 0:
            if self.committed_action_history is not None:
                base._fail(
                    "ACTION_HISTORY_SCHEMA_MISMATCH",
                    "empty Stage-2B cache requires untyped empty history",
                )
            return
        history = self.committed_action_history
        if not isinstance(history, CommittedActionHistory):
            base._fail(
                "ACTION_HISTORY_SCHEMA_MISMATCH",
                "committed Stage-2B cache requires exact action history",
            )
        try:
            history.assert_integrity()
        except CommittedActionHistoryError as exc:
            _translate_history_error(exc)
        if (
            history.episode_id != state.episode_id
            or history.episode_epoch != state.episode_epoch
            or history.layout_spec_sha256 != state.layout_spec_sha256
            or history.layout_instance_digest != state.layout_instance_digest
            or history.action_cursor != state.action_cursor
            or history.committed_through_chunk != state.committed_through_chunk
            or len(history.spans) != state.revision
        ):
            base._fail(
                "ACTION_HISTORY_COVERAGE_GAP",
                "cache and committed-action history identities differ",
            )
        last_span = history.spans[-1]
        if (
            state.last_commit_source is not base.CommitSource.TEACHER_FORCING
            or last_span.commit_source.value != state.last_commit_source.value
            or last_span.source_proof_digest != state.last_source_proof_digest
        ):
            base._fail(
                "ACTION_HISTORY_SOURCE_UNAUTHORIZED",
                "cache and history source proofs differ",
            )

    @classmethod
    def empty(
        cls,
        *,
        episode_id: str,
        episode_epoch: int,
        layout: ChunkActionLayout,
        registry: base.LayerRegistry,
        staging_variant: CACHStagingVariant,
    ) -> "Stage2BTemporalState":
        return cls(
            cache_state=base.HybridTemporalState.empty(
                episode_id=episode_id,
                episode_epoch=episode_epoch,
                layout_spec_sha256=layout.layout_spec_sha256,
                layout_instance_digest=layout.layout_instance_digest,
                registry=registry,
            ),
            staging_variant=staging_variant,
        )

    @property
    def episode_id(self) -> str:
        return self.cache_state.episode_id

    @property
    def episode_epoch(self) -> int:
        return self.cache_state.episode_epoch

    @property
    def revision(self) -> int:
        return self.cache_state.revision

    @property
    def committed_through_chunk(self) -> int | None:
        return self.cache_state.committed_through_chunk

    @property
    def next_chunk_id(self) -> int:
        return self.cache_state.next_chunk_id

    @property
    def action_cursor(self) -> int:
        return self.cache_state.action_cursor

    @property
    def layout_spec_sha256(self) -> str:
        return self.cache_state.layout_spec_sha256

    @property
    def layout_instance_digest(self) -> str:
        return self.cache_state.layout_instance_digest

    @property
    def layer_states(self) -> tuple[base.LayerTemporalState, ...]:
        return self.cache_state.layer_states

    def to_manifest(self) -> dict[str, object]:
        return {
            "action_cursor": self.action_cursor,
            "cache_state_manifest_digest": self.cache_state.state_manifest_digest,
            "committed_action_history_digest": _history_digest(
                self.committed_action_history
            ),
            "committed_through_chunk": self.committed_through_chunk,
            "episode_epoch": self.episode_epoch,
            "episode_id": self.episode_id,
            "history_summary_operator": history_summary_operator_for_variant(
                self.staging_variant
            ),
            "layout_instance_digest": self.layout_instance_digest,
            "layout_spec_sha256": self.layout_spec_sha256,
            "next_chunk_id": self.next_chunk_id,
            "revision": self.revision,
            "schema": _STATE_SCHEMA,
            "staging_variant": self.staging_variant.value,
        }

    @property
    def state_manifest_digest(self) -> str:
        return base._sha256_bytes(base._canonical_json_bytes(self.to_manifest()))

    def clone(self) -> "Stage2BTemporalState":
        return Stage2BTemporalState(
            cache_state=base._clone_state_for_inspection(self.cache_state),
            staging_variant=self.staging_variant,
            committed_action_history=(
                None
                if self.committed_action_history is None
                else self.committed_action_history.clone()
            ),
        )


def _scratch_from_state(
    state: Stage2BTemporalState,
    *,
    source_state_identity: int,
) -> base.CacheScratch:
    inner = base.CacheScratch.from_state(
        state.cache_state,
        _source_state_identity=source_state_identity,
    )
    scratch = base.CacheScratch(
        layers=inner.layers,
        source_state_manifest_digest=state.state_manifest_digest,
        _source_state_identity=source_state_identity,
        _baseline_fingerprint=(),
    )
    object.__setattr__(scratch, "_baseline_fingerprint", scratch.runtime_fingerprint())
    return scratch


@dataclass(frozen=True)
class Stage2BDenoiseReadView:
    episode_id: str
    episode_epoch: int
    revision: int
    next_chunk_id: int
    staging_variant: CACHStagingVariant
    layout_instance_digest: str
    state_manifest_digest: str
    previous_committed_action_history: CommittedActionHistoryView | None = field(
        repr=False,
        compare=False,
    )
    _state_identity: int = field(repr=False)
    _state_snapshot: Stage2BTemporalState = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        self.assert_integrity()

    def assert_integrity(self) -> None:
        state = self._state_snapshot
        if not isinstance(state, Stage2BTemporalState):
            base._fail("CACHE_SCHEMA_MISMATCH", "read view snapshot has wrong type")
        if type(self._state_identity) is not int or self._state_identity <= 0:
            base._fail("CACHE_SCHEMA_MISMATCH", "read view identity is invalid")
        if (
            self.episode_id != state.episode_id
            or self.episode_epoch != state.episode_epoch
            or self.revision != state.revision
            or self.next_chunk_id != state.next_chunk_id
            or self.staging_variant is not state.staging_variant
            or self.layout_instance_digest != state.layout_instance_digest
            or self.state_manifest_digest != state.state_manifest_digest
        ):
            base._fail(
                "CACHE_VARIANT_OR_STATE_MISMATCH",
                "read view differs from its immutable Stage-2B snapshot",
            )
        history_view = self.previous_committed_action_history
        history = state.committed_action_history
        if history is None:
            if history_view is not None:
                base._fail(
                    "ACTION_HISTORY_SCHEMA_MISMATCH",
                    "empty read view must not carry committed history",
                )
        else:
            if not isinstance(history_view, CommittedActionHistoryView):
                base._fail(
                    "ACTION_HISTORY_SCHEMA_MISMATCH",
                    "committed read view lacks exact history",
                )
            try:
                history_view.assert_unchanged()
            except CommittedActionHistoryError as exc:
                _translate_history_error(exc)
            if history_view.history_manifest_digest != history.manifest_digest:
                base._fail(
                    "ACTION_HISTORY_TENSOR_IDENTITY_MISMATCH",
                    "read view history differs from its state snapshot",
                )

    def export_scratch(self) -> base.CacheScratch:
        self.assert_integrity()
        return _scratch_from_state(
            self._state_snapshot,
            source_state_identity=self._state_identity,
        )


@dataclass(frozen=True)
class Stage2BPairedStagingContext:
    commit_source: base.CommitSource
    staging_variant: CACHStagingVariant
    content_time: base.ContentTime
    previous_state_manifest_digest: str
    previous_scratch: base.CacheScratch
    video: Tensor = field(repr=False, compare=False)
    frame_valid_mask: Tensor = field(repr=False, compare=False)
    frame_valid_mask_digest: str
    actions: Tensor = field(repr=False, compare=False)
    action_valid_mask: Tensor = field(repr=False, compare=False)
    previous_committed_action_history: CommittedActionHistoryView = field(
        repr=False,
        compare=False,
    )
    conditioning_digest: str | None = None
    video_timestep: int = 0
    action_timestep: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.staging_variant, CACHStagingVariant):
            base._fail("CACHE_SCHEMA_MISMATCH", "staging variant has wrong type")
        base._require_sha256(
            self.previous_state_manifest_digest,
            "previous_state_manifest_digest",
        )
        base._require_sha256(self.frame_valid_mask_digest, "frame_valid_mask_digest")
        if self.conditioning_digest is not None:
            base._require_sha256(self.conditioning_digest, "conditioning_digest")
        if base.tensor_digest(self.frame_valid_mask) != self.frame_valid_mask_digest:
            base._fail(
                "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH",
                "staging frame mask differs from its bound digest",
            )
        history = self.previous_committed_action_history
        if not isinstance(history, CommittedActionHistoryView):
            base._fail(
                "ACTION_HISTORY_SCHEMA_MISMATCH",
                "paired staging requires typed prior history",
            )
        if (
            history.episode_id != self.content_time.episode_id
            or history.episode_epoch != self.content_time.episode_epoch
            or history.layout_spec_sha256 != self.content_time.layout_spec_sha256
            or history.layout_instance_digest
            != self.content_time.layout_instance_digest
            or history.chunk_id != self.content_time.chunk_id
            or history.cursor != self.content_time.action_token_start
        ):
            base._fail(
                "ACTION_HISTORY_COVERAGE_GAP",
                "staging history is not the exact prior prefix",
            )
        try:
            history.assert_unchanged()
        except CommittedActionHistoryError as exc:
            _translate_history_error(exc)


@dataclass(frozen=True)
class Stage2BPairedCommitReceipt:
    commit_id: str
    staging_variant: CACHStagingVariant
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
    teacher_pair_payload_digest: str
    paired_payload_digest: str
    conditioning_digest: str | None
    dataset_manifest_sha256: str
    dataset_row_identity: str
    state_manifest_before: str
    staged_state_manifest: str
    state_manifest_after: str
    action_history_digest_before: str | None
    action_history_digest_after: str
    action_cursor_before: int
    action_cursor_after: int
    committed_at_monotonic_ns: int
    commit_source: base.CommitSource = base.CommitSource.TEACHER_FORCING
    source_proof_schema: str = base._TEACHER_PROOF_SCHEMA
    result: str = "committed"
    schema: str = _COMMIT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _COMMIT_RECEIPT_SCHEMA:
            base._fail("COMMIT_RECEIPT_INVALID", "Stage-2B receipt schema differs")
        if not isinstance(self.staging_variant, CACHStagingVariant):
            base._fail("COMMIT_RECEIPT_INVALID", "receipt variant has wrong type")
        for name in (
            "commit_id",
            "transaction_nonce",
            "episode_id",
            "dataset_row_identity",
        ):
            base._require_non_empty_string(getattr(self, name), name)
        for name in (
            "episode_epoch",
            "revision_before",
            "revision_after",
            "chunk_id",
            "action_cursor_before",
            "action_cursor_after",
            "committed_at_monotonic_ns",
        ):
            base._require_plain_uint(getattr(self, name), name)
        for name in (
            "layout_instance_digest",
            "source_proof_digest",
            "pair_video_digest",
            "pair_frame_valid_mask_digest",
            "pair_action_digest",
            "pair_action_mask_digest",
            "teacher_pair_payload_digest",
            "paired_payload_digest",
            "dataset_manifest_sha256",
            "state_manifest_before",
            "staged_state_manifest",
            "state_manifest_after",
            "action_history_digest_after",
        ):
            base._require_sha256(getattr(self, name), name)
        if self.action_history_digest_before is not None:
            base._require_sha256(
                self.action_history_digest_before,
                "action_history_digest_before",
            )
        if (self.revision_before == 0) != (self.action_history_digest_before is None):
            base._fail(
                "COMMIT_RECEIPT_INVALID",
                "history-before must be absent exactly for the initial revision",
            )
        if (self.revision_before == 0) != (self.action_cursor_before == 0):
            base._fail(
                "COMMIT_RECEIPT_INVALID",
                "cursor-before must be zero exactly for the initial revision",
            )
        if self.conditioning_digest is not None:
            base._require_sha256(self.conditioning_digest, "conditioning_digest")
        if (
            self.revision_after != self.revision_before + 1
            or self.chunk_id != self.revision_before
            or self.staged_state_manifest != self.state_manifest_after
            or self.state_manifest_after == self.state_manifest_before
        ):
            base._fail(
                "COMMIT_RECEIPT_INVALID",
                "receipt revision/chunk/staged-state transition differs",
            )
        if self.action_cursor_after <= self.action_cursor_before:
            base._fail("COMMIT_RECEIPT_INVALID", "receipt must advance action cursor")
        if (
            self.action_history_digest_before is not None
            and self.action_history_digest_after == self.action_history_digest_before
        ):
            base._fail(
                "COMMIT_RECEIPT_INVALID",
                "receipt must advance the action-history digest",
            )
        expected_paired_payload_digest = base._sha256_bytes(
            base._canonical_json_bytes(
                {
                    "conditioning_digest": self.conditioning_digest,
                    "schema": "cach.stage2b.paired_payload.v1",
                    "staging_variant": self.staging_variant.value,
                    "teacher_pair_payload_digest": self.teacher_pair_payload_digest,
                }
            )
        )
        if self.paired_payload_digest != expected_paired_payload_digest:
            base._fail(
                "COMMIT_RECEIPT_INVALID",
                "receipt paired-payload digest relation differs",
            )
        if (
            self.commit_source is not base.CommitSource.TEACHER_FORCING
            or self.source_proof_schema != base._TEACHER_PROOF_SCHEMA
            or self.result != "committed"
        ):
            base._fail("COMMIT_RECEIPT_INVALID", "receipt source/result differs")

    def to_manifest(self) -> dict[str, object]:
        return {
            "action_cursor_after": self.action_cursor_after,
            "action_cursor_before": self.action_cursor_before,
            "action_history_digest_after": self.action_history_digest_after,
            "action_history_digest_before": self.action_history_digest_before,
            "chunk_id": self.chunk_id,
            "commit_id": self.commit_id,
            "commit_source": self.commit_source.value,
            "committed_at_monotonic_ns": self.committed_at_monotonic_ns,
            "conditioning_digest": self.conditioning_digest,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_row_identity": self.dataset_row_identity,
            "episode_epoch": self.episode_epoch,
            "episode_id": self.episode_id,
            "layout_instance_digest": self.layout_instance_digest,
            "pair_action_digest": self.pair_action_digest,
            "pair_action_mask_digest": self.pair_action_mask_digest,
            "pair_frame_valid_mask_digest": self.pair_frame_valid_mask_digest,
            "pair_video_digest": self.pair_video_digest,
            "paired_payload_digest": self.paired_payload_digest,
            "result": self.result,
            "revision_after": self.revision_after,
            "revision_before": self.revision_before,
            "schema": self.schema,
            "staging_variant": self.staging_variant.value,
            "source_proof_digest": self.source_proof_digest,
            "source_proof_schema": self.source_proof_schema,
            "staged_state_manifest": self.staged_state_manifest,
            "state_manifest_after": self.state_manifest_after,
            "state_manifest_before": self.state_manifest_before,
            "teacher_pair_payload_digest": self.teacher_pair_payload_digest,
            "transaction_nonce": self.transaction_nonce,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return base._canonical_json_bytes(self.to_manifest())

    @property
    def receipt_sha256(self) -> str:
        return base._sha256_bytes(self.canonical_bytes)


@dataclass(frozen=True)
class Stage2BCacheResetReceipt:
    reset_id: str
    staging_variant: CACHStagingVariant
    transaction_nonce: str
    previous_episode_id: str
    previous_episode_epoch: int
    new_episode_id: str
    new_episode_epoch: int
    layout_spec_sha256: str
    layout_instance_digest: str
    state_manifest_before: str
    state_manifest_after: str
    action_history_digest_before: str | None
    action_history_digest_after: str | None
    action_cursor_before: int
    action_cursor_after: int
    aborted_pending_commit_id: str | None
    reset_at_monotonic_ns: int
    schema: str = _RESET_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _RESET_RECEIPT_SCHEMA:
            base._fail("COMMIT_RECEIPT_INVALID", "Stage-2B reset schema differs")
        if not isinstance(self.staging_variant, CACHStagingVariant):
            base._fail("COMMIT_RECEIPT_INVALID", "reset variant has wrong type")
        for name in (
            "reset_id",
            "transaction_nonce",
            "previous_episode_id",
            "new_episode_id",
        ):
            base._require_non_empty_string(getattr(self, name), name)
        for name in (
            "previous_episode_epoch",
            "new_episode_epoch",
            "action_cursor_before",
            "action_cursor_after",
            "reset_at_monotonic_ns",
        ):
            base._require_plain_uint(getattr(self, name), name)
        for name in (
            "layout_spec_sha256",
            "layout_instance_digest",
            "state_manifest_before",
            "state_manifest_after",
        ):
            base._require_sha256(getattr(self, name), name)
        if self.action_history_digest_before is not None:
            base._require_sha256(
                self.action_history_digest_before,
                "action_history_digest_before",
            )
        if self.action_history_digest_after is not None:
            base._fail(
                "COMMIT_RECEIPT_INVALID",
                "reset must clear the action-history digest",
            )
        if (self.action_cursor_before == 0) != (
            self.action_history_digest_before is None
        ):
            base._fail(
                "COMMIT_RECEIPT_INVALID",
                "reset cursor/history-before emptiness differs",
            )
        if (
            self.new_episode_epoch != self.previous_episode_epoch + 1
            or self.action_cursor_after != 0
            or self.state_manifest_after == self.state_manifest_before
        ):
            base._fail("COMMIT_RECEIPT_INVALID", "reset transition differs")
        if self.aborted_pending_commit_id is not None:
            base._require_non_empty_string(
                self.aborted_pending_commit_id,
                "aborted_pending_commit_id",
            )
            if self.aborted_pending_commit_id == self.reset_id:
                base._fail(
                    "COMMIT_RECEIPT_INVALID",
                    "reset cannot abort its own transaction ID",
                )

    def to_manifest(self) -> dict[str, object]:
        return {
            "aborted_pending_commit_id": self.aborted_pending_commit_id,
            "action_cursor_after": self.action_cursor_after,
            "action_cursor_before": self.action_cursor_before,
            "action_history_digest_after": self.action_history_digest_after,
            "action_history_digest_before": self.action_history_digest_before,
            "layout_instance_digest": self.layout_instance_digest,
            "layout_spec_sha256": self.layout_spec_sha256,
            "new_episode_epoch": self.new_episode_epoch,
            "new_episode_id": self.new_episode_id,
            "previous_episode_epoch": self.previous_episode_epoch,
            "previous_episode_id": self.previous_episode_id,
            "reset_at_monotonic_ns": self.reset_at_monotonic_ns,
            "reset_id": self.reset_id,
            "schema": self.schema,
            "staging_variant": self.staging_variant.value,
            "state_manifest_after": self.state_manifest_after,
            "state_manifest_before": self.state_manifest_before,
            "transaction_nonce": self.transaction_nonce,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return base._canonical_json_bytes(self.to_manifest())

    @property
    def receipt_sha256(self) -> str:
        return base._sha256_bytes(self.canonical_bytes)


class Stage2BHybridCacheManager:
    """Own one atomic outer pointer containing typed cache and history."""

    def __init__(
        self,
        *,
        registry: base.LayerRegistry,
        episode_id: str,
        layout: ChunkActionLayout,
        receipt_publisher: base.DurableReceiptPublisher,
        staging_variant: CACHStagingVariant,
        episode_epoch: int = 0,
        synthetic_test_only: bool = False,
    ) -> None:
        if not isinstance(registry, base.LayerRegistry):
            base._fail("CACHE_SCHEMA_MISMATCH", "registry must be LayerRegistry")
        if not isinstance(layout, ChunkActionLayout):
            base._fail("CACHE_CONTENT_TIME_MISMATCH", "layout has wrong type")
        if not isinstance(staging_variant, CACHStagingVariant):
            base._fail("CACHE_SCHEMA_MISMATCH", "staging_variant has wrong type")
        if type(synthetic_test_only) is not bool:
            base._fail("CACHE_SCHEMA_MISMATCH", "synthetic flag must be bool")
        if layout.synthetic_test_only != synthetic_test_only:
            base._fail(
                "CACHE_SYNTHETIC_LAYOUT_PROVENANCE",
                "manager/layout synthetic provenance differs",
            )
        if synthetic_test_only:
            if not registry.layers:
                base._fail("CACHE_SCHEMA_MISMATCH", "synthetic registry is empty")
        else:
            base.validate_cach_a_registry(registry)
        if receipt_publisher is None or not hasattr(
            receipt_publisher, "publish_exclusive"
        ):
            base._fail("COMMIT_RECEIPT_INVALID", "durable publisher is required")
        self._registry = registry
        self._layout = layout
        self._staging_variant = staging_variant
        self._synthetic_test_mode = synthetic_test_only
        self._publisher = receipt_publisher
        self._lock = threading.RLock()
        self._state = Stage2BTemporalState.empty(
            episode_id=episode_id,
            episode_epoch=episode_epoch,
            layout=layout,
            registry=registry,
            staging_variant=staging_variant,
        )
        self._state_manifest_digest = self._state.state_manifest_digest
        self._pending: tuple[str, str, object] | None = None
        self._completed: dict[str, tuple[str, Stage2BPairedCommitReceipt]] = {}
        self._completed_resets: dict[
            str, tuple[str, Stage2BCacheResetReceipt]
        ] = {}
        self._publication_token: object | None = None
        self._poisoned = False

    @classmethod
    def for_synthetic_tests(
        cls,
        *,
        registry: base.LayerRegistry,
        episode_id: str,
        layout: ChunkActionLayout,
        receipt_publisher: base.DurableReceiptPublisher,
        staging_variant: CACHStagingVariant,
        episode_epoch: int = 0,
    ) -> "Stage2BHybridCacheManager":
        return cls(
            registry=registry,
            episode_id=episode_id,
            layout=layout,
            receipt_publisher=receipt_publisher,
            staging_variant=staging_variant,
            episode_epoch=episode_epoch,
            synthetic_test_only=True,
        )

    @property
    def registry(self) -> base.LayerRegistry:
        return self._registry

    @property
    def staging_variant(self) -> CACHStagingVariant:
        return self._staging_variant

    @property
    def state(self) -> Stage2BTemporalState:
        with self._lock:
            self._assert_not_publishing()
            self._assert_not_poisoned()
            self._assert_live_integrity()
            return self._state.clone()

    def _assert_not_poisoned(self) -> None:
        if self._poisoned:
            base._fail(
                "CACHE_PUBLICATION_POISONED",
                "Stage-2B publication failed after a durable decision",
            )

    def _assert_not_publishing(self) -> None:
        if self._publication_token is not None:
            base._fail(
                "CACHE_PUBLICATION_REENTRANT",
                "publisher callbacks may not re-enter the Stage-2B manager",
            )

    def _assert_live_integrity(self) -> None:
        if self._state.state_manifest_digest != self._state_manifest_digest:
            self._poisoned = True
            base._fail(
                "CACHE_DENOISE_MUTATION",
                "live Stage-2B state changed outside atomic publication",
            )

    def snapshot_for_denoise(
        self,
        *,
        expected_episode_id: str,
        expected_episode_epoch: int,
        expected_revision: int,
        expected_next_chunk_id: int,
    ) -> Stage2BDenoiseReadView:
        with self._lock:
            self._assert_not_publishing()
            self._assert_not_poisoned()
            self._assert_live_integrity()
            state = self._state
            if (
                state.episode_id != expected_episode_id
                or state.episode_epoch != expected_episode_epoch
                or state.revision != expected_revision
                or state.next_chunk_id != expected_next_chunk_id
            ):
                base._fail(
                    "CACHE_CONTENT_TIME_MISMATCH",
                    "denoise request differs from live Stage-2B state",
                )
            if state.next_chunk_id >= len(self._layout.chunks):
                base._fail("COMMIT_OUT_OF_ORDER", "layout has no next chunk")
            try:
                history_view = (
                    None
                    if state.committed_action_history is None
                    else state.committed_action_history.view_for_chunk(
                        chunk_id=state.next_chunk_id,
                        action_start=state.action_cursor,
                    )
                )
            except CommittedActionHistoryError as exc:
                _translate_history_error(exc)
            return Stage2BDenoiseReadView(
                episode_id=state.episode_id,
                episode_epoch=state.episode_epoch,
                revision=state.revision,
                next_chunk_id=state.next_chunk_id,
                staging_variant=state.staging_variant,
                layout_instance_digest=state.layout_instance_digest,
                state_manifest_digest=state.state_manifest_digest,
                previous_committed_action_history=history_view,
                _state_identity=id(state),
                _state_snapshot=state.clone(),
            )

    def finish_denoise(
        self,
        *,
        read_view: Stage2BDenoiseReadView,
        scratch: base.CacheScratch,
    ) -> None:
        if not isinstance(read_view, Stage2BDenoiseReadView) or not isinstance(
            scratch, base.CacheScratch
        ):
            base._fail("CACHE_SCHEMA_MISMATCH", "invalid Stage-2B denoise view")
        read_view.assert_integrity()
        with self._lock:
            self._assert_not_publishing()
        scratch.assert_unchanged()
        history_view = read_view.previous_committed_action_history
        if history_view is not None:
            try:
                history_view.assert_unchanged()
            except CommittedActionHistoryError as exc:
                _translate_history_error(exc)
        if (
            scratch.source_state_manifest_digest != read_view.state_manifest_digest
            or scratch._source_state_identity != read_view._state_identity
        ):
            base._fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "scratch was not exported from this Stage-2B view",
            )
        with self._lock:
            self._assert_not_publishing()
            self._assert_not_poisoned()
            self._assert_live_integrity()
            if (
                id(self._state) != read_view._state_identity
                or self._state.revision != read_view.revision
                or self._state.episode_epoch != read_view.episode_epoch
                or self._state.staging_variant is not read_view.staging_variant
            ):
                base._fail("COMMIT_STALE_REVISION", "denoise view became stale")
            if self._state.state_manifest_digest != read_view.state_manifest_digest:
                base._fail("CACHE_DENOISE_MUTATION", "live state changed during denoise")

    def _validate_next_content_time(
        self,
        state: Stage2BTemporalState,
        content_time: base.ContentTime,
    ) -> None:
        inner = state.cache_state
        if inner.next_chunk_id >= len(self._layout.chunks):
            base._fail("COMMIT_OUT_OF_ORDER", "commit exceeds layout chunk count")
        if (
            content_time.episode_id != inner.episode_id
            or content_time.episode_epoch != inner.episode_epoch
            or content_time.layout_spec_sha256 != inner.layout_spec_sha256
            or content_time.layout_instance_digest != inner.layout_instance_digest
            or inner.layout_spec_sha256 != self._layout.layout_spec_sha256
            or inner.layout_instance_digest != self._layout.layout_instance_digest
        ):
            base._fail(
                "CACHE_CONTENT_TIME_MISMATCH",
                "commit content-time differs from live episode/layout",
            )
        if content_time.chunk_id != inner.next_chunk_id:
            base._fail("COMMIT_OUT_OF_ORDER", "commit is not live next chunk")
        if content_time.action_token_start != inner.action_cursor:
            base._fail("COMMIT_OUT_OF_ORDER", "action cursor is not contiguous")
        bound_chunk = self._layout.chunks[inner.next_chunk_id]
        content_time.verify_layout_binding(layout=self._layout, chunk=bound_chunk)
        if inner.revision == 0:
            if content_time.latent_start != 0 or content_time.raw_observation_start != 0:
                base._fail("COMMIT_OUT_OF_ORDER", "first commit must start at zero")
            return
        previous = inner.layer_states[0].through
        assert previous is not None
        if (
            content_time.latent_start != previous.latent_end_exclusive
            or content_time.raw_observation_start
            != previous.raw_observation_end_exclusive
            or content_time.action_rope_start
            != previous.action_rope_end_exclusive
        ):
            base._fail("COMMIT_OUT_OF_ORDER", "content intervals are not contiguous")

    def commit_paired(
        self,
        *,
        commit_source: base.CommitSource | str,
        request: Stage2BTeacherForcingCommitRequest | None,
        stage_callback: Callable[
            [Stage2BPairedStagingContext], tuple[base.StagedLayerPayload, ...]
        ]
        | None,
    ) -> Stage2BPairedCommitReceipt:
        try:
            source = (
                commit_source
                if isinstance(commit_source, base.CommitSource)
                else base.CommitSource(commit_source)
            )
        except (TypeError, ValueError):
            base._fail("COMMIT_SOURCE_UNKNOWN", "unknown paired commit source")
        if source is base.CommitSource.SELF_FORCING:
            base._fail(
                "COMMIT_SELF_FORCING_OBJECTIVE_UNCLOSED",
                "self-forcing remains disabled",
            )
        if source is base.CommitSource.DEPLOY_APPLIED_ACK:
            base._fail(
                "COMMIT_DEPLOY_ACK_MISSING",
                "deploy history remains blocked on verified ACK transport",
            )
        if not isinstance(request, Stage2BTeacherForcingCommitRequest):
            base._fail(
                "COMMIT_SOURCE_PROOF_MISSING",
                "Stage-2B teacher request and proof are required",
            )
        if not callable(stage_callback):
            base._fail("COMMIT_STAGING_CALLBACK_MISSING", "stager is required")
        if not self._synthetic_test_mode:
            if request.conditioning_digest is None:
                base._fail(
                    "COMMIT_CONDITIONING_IDENTITY_MISSING",
                    "production-shaped staging requires an exact conditioning digest",
                )
            from sana_wam.model.cach_paired_stager import (
                CACHTeacherForcingPairedStager,
            )

            if type(stage_callback) is not CACHTeacherForcingPairedStager:
                base._fail(
                    "COMMIT_STAGING_CALLBACK_UNAUTHORIZED",
                    "non-synthetic commits require the exact production paired stager",
                )
        return self._commit_teacher(request=request, stage_callback=stage_callback)

    def _commit_teacher(
        self,
        *,
        request: Stage2BTeacherForcingCommitRequest,
        stage_callback: Callable[
            [Stage2BPairedStagingContext], tuple[base.StagedLayerPayload, ...]
        ],
    ) -> Stage2BPairedCommitReceipt:
        with self._lock:
            self._assert_not_publishing()
            self._assert_not_poisoned()
            self._assert_live_integrity()
            if request.commit_id in self._completed_resets:
                base._fail(
                    "COMMIT_DUPLICATE_CONFLICT",
                    "receipt id was already used by a reset transaction",
                )
            live = self._state
            if (
                request.commit_id not in self._completed
                and (
                    request.expected_episode_id != live.episode_id
                    or request.expected_episode_epoch != live.episode_epoch
                    or request.expected_revision != live.revision
                )
            ):
                base._fail("COMMIT_STALE_REVISION", "teacher request is stale")
            layout_snapshot = self._layout
            content_time = request.content_time
            if (
                content_time.layout_spec_sha256
                != layout_snapshot.layout_spec_sha256
                or content_time.layout_instance_digest
                != layout_snapshot.layout_instance_digest
                or content_time.chunk_id >= len(layout_snapshot.chunks)
            ):
                base._fail(
                    "CACHE_CONTENT_TIME_MISMATCH",
                    "teacher pair does not name a manager-bound chunk",
                )
            bound_chunk = layout_snapshot.chunks[content_time.chunk_id]
            content_time.verify_layout_binding(
                layout=layout_snapshot,
                chunk=bound_chunk,
            )
        prepared = base._prepare_teacher_pair(request, chunk=bound_chunk)
        transaction_payload_digest = base._sha256_bytes(
            base._canonical_json_bytes(
                {
                    "conditioning_digest": request.conditioning_digest,
                    "schema": "cach.stage2b.paired_payload.v1",
                    "staging_variant": self._staging_variant.value,
                    "teacher_pair_payload_digest": prepared.paired_payload_digest,
                }
            )
        )
        pending_token = object()

        with self._lock:
            self._assert_not_publishing()
            self._assert_not_poisoned()
            self._assert_live_integrity()
            completed = self._completed.get(request.commit_id)
            if completed is not None:
                prior_digest, prior_receipt = completed
                if prior_digest != transaction_payload_digest:
                    base._fail(
                        "COMMIT_DUPLICATE_CONFLICT",
                        "commit_id was reused for different Stage-2B inputs",
                    )
                live = self._state
                if (
                    live.episode_id != prior_receipt.episode_id
                    or live.episode_epoch != prior_receipt.episode_epoch
                    or live.revision < prior_receipt.revision_after
                ):
                    base._fail(
                        "COMMIT_STALE_REVISION",
                        "completed commit belongs to a stale episode",
                    )
                return prior_receipt
            if self._pending is not None:
                pending_id, pending_digest, _ = self._pending
                if pending_id == request.commit_id and pending_digest != transaction_payload_digest:
                    base._fail(
                        "COMMIT_DUPLICATE_CONFLICT",
                        "pending commit_id has different inputs",
                    )
                base._fail("COMMIT_DUPLICATE_PENDING", "another commit is pending")
            base_state = self._state
            if (
                request.expected_episode_id != base_state.episode_id
                or request.expected_episode_epoch != base_state.episode_epoch
                or request.expected_revision != base_state.revision
            ):
                base._fail("COMMIT_STALE_REVISION", "teacher request is stale")
            self._validate_next_content_time(base_state, request.content_time)
            try:
                if base_state.committed_action_history is None:
                    base_history = CommittedActionHistory.empty(
                        episode_id=base_state.episode_id,
                        episode_epoch=base_state.episode_epoch,
                        layout_spec_sha256=base_state.layout_spec_sha256,
                        layout_instance_digest=base_state.layout_instance_digest,
                        batch_size=prepared.actions.shape[0],
                        action_dim=prepared.actions.shape[2],
                        dtype=prepared.actions.dtype,
                        device=prepared.actions.device,
                    )
                else:
                    base_history = base_state.committed_action_history
                history_view = base_history.view_for_chunk(
                    chunk_id=request.content_time.chunk_id,
                    action_start=request.content_time.action_token_start,
                )
            except CommittedActionHistoryError as exc:
                _translate_history_error(exc)
            self._pending = (
                request.commit_id,
                transaction_payload_digest,
                pending_token,
            )

        try:
            context = Stage2BPairedStagingContext(
                commit_source=base.CommitSource.TEACHER_FORCING,
                staging_variant=self._staging_variant,
                content_time=request.content_time,
                previous_state_manifest_digest=base_state.state_manifest_digest,
                previous_scratch=_scratch_from_state(
                    base_state,
                    source_state_identity=id(base_state),
                ),
                video=prepared.video.detach().clone(),
                frame_valid_mask=prepared.frame_valid_mask.detach().clone(),
                frame_valid_mask_digest=prepared.frame_valid_mask_digest,
                actions=prepared.actions.detach().clone(),
                action_valid_mask=prepared.action_valid_mask.detach().clone(),
                previous_committed_action_history=history_view,
                conditioning_digest=request.conditioning_digest,
            )
            input_fingerprint = (
                base._runtime_tensor_fingerprint(context.video),
                base._runtime_tensor_fingerprint(context.frame_valid_mask),
                base._runtime_tensor_fingerprint(context.actions),
                base._runtime_tensor_fingerprint(context.action_valid_mask),
            )
            raw_payloads = stage_callback(context)
            if type(raw_payloads) is not tuple:
                base._fail(
                    "CACHE_SCHEMA_MISMATCH",
                    "Stage-2B stager must return an eagerly materialized exact tuple",
                )
            layer_states = base._materialize_layer_states(
                registry=self._registry,
                content_time=request.content_time,
                payloads=raw_payloads,
            )
            context.previous_scratch.assert_unchanged()
            try:
                context.previous_committed_action_history.assert_unchanged()
            except CommittedActionHistoryError as exc:
                _translate_history_error(exc)
            if input_fingerprint != (
                base._runtime_tensor_fingerprint(context.video),
                base._runtime_tensor_fingerprint(context.frame_valid_mask),
                base._runtime_tensor_fingerprint(context.actions),
                base._runtime_tensor_fingerprint(context.action_valid_mask),
            ):
                base._fail(
                    "COMMIT_STAGING_INPUT_MUTATION",
                    "stager mutated paired inputs",
                )
            try:
                staged_history, duplicate = base_history.append(
                    chunk_id=request.content_time.chunk_id,
                    action_start=request.content_time.action_token_start,
                    action_end_exclusive=request.content_time.action_token_end_exclusive,
                    fixed_slot_actions=prepared.actions,
                    action_valid_mask=prepared.action_valid_mask,
                    commit_source=base.CommitSource.TEACHER_FORCING,
                    source_proof_digest=prepared.source_proof_digest,
                    commit_id=request.commit_id,
                )
            except CommittedActionHistoryError as exc:
                _translate_history_error(exc)
            if duplicate:
                base._fail(
                    "ACTION_HISTORY_DUPLICATE_CONFLICT",
                    "new transaction unexpectedly reused committed history",
                )
            inner_state = base.HybridTemporalState(
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
                committed_pair_digest=transaction_payload_digest,
                last_commit_source=base.CommitSource.TEACHER_FORCING,
                last_source_proof_digest=prepared.source_proof_digest,
                pair_evidence_mode="dataset_ground_truth",
            )
            staged_state = Stage2BTemporalState(
                cache_state=inner_state,
                staging_variant=self._staging_variant,
                committed_action_history=staged_history,
            )
        except base.CacheContractError:
            with self._lock:
                if self._pending is not None and self._pending[2] is pending_token:
                    self._pending = None
            raise
        except Exception as exc:
            with self._lock:
                if self._pending is not None and self._pending[2] is pending_token:
                    self._pending = None
            raise base.CacheContractError(
                "COMMIT_STAGING_FAILED",
                f"paired staging failed: {type(exc).__name__}",
            ) from exc

        with self._lock:
            self._assert_not_publishing()
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
                base._fail("COMMIT_STALE_REVISION", "live state changed before CAS")
            before_digest = base_state.state_manifest_digest
            after_digest = staged_state.state_manifest_digest
            receipt = Stage2BPairedCommitReceipt(
                commit_id=request.commit_id,
                staging_variant=self._staging_variant,
                transaction_nonce=request.transaction_nonce,
                episode_id=base_state.episode_id,
                episode_epoch=base_state.episode_epoch,
                revision_before=base_state.revision,
                revision_after=staged_state.revision,
                chunk_id=request.content_time.chunk_id,
                layout_instance_digest=request.content_time.layout_instance_digest,
                source_proof_digest=prepared.source_proof_digest,
                pair_video_digest=prepared.video_digest,
                pair_frame_valid_mask_digest=prepared.frame_valid_mask_digest,
                pair_action_digest=prepared.action_digest,
                pair_action_mask_digest=prepared.action_mask_digest,
                teacher_pair_payload_digest=prepared.paired_payload_digest,
                paired_payload_digest=transaction_payload_digest,
                conditioning_digest=request.conditioning_digest,
                dataset_manifest_sha256=request.proof.dataset_manifest_sha256,
                dataset_row_identity=request.proof.dataset_row_identity,
                state_manifest_before=before_digest,
                staged_state_manifest=after_digest,
                state_manifest_after=after_digest,
                action_history_digest_before=_history_digest(
                    base_state.committed_action_history
                ),
                action_history_digest_after=staged_history.manifest_digest,
                action_cursor_before=base_history.action_cursor,
                action_cursor_after=staged_history.action_cursor,
                committed_at_monotonic_ns=time.monotonic_ns(),
            )
            publication_token = object()
            self._publication_token = publication_token
            try:
                publication = self._publisher.publish_exclusive(
                    receipt_id=request.commit_id,
                    payload=receipt.canonical_bytes,
                    expected_sha256=receipt.receipt_sha256,
                )
            except Exception as exc:
                self._publication_token = None
                self._poisoned = True
                self._pending = None
                raise base.CacheContractError(
                    "COMMIT_RECEIPT_PUBLICATION_FAILED",
                    f"receipt publisher failed: {type(exc).__name__}",
                ) from exc
            if (
                self._publication_token is not publication_token
                or self._pending is None
                or self._pending[2] is not pending_token
                or self._state is not base_state
            ):
                self._publication_token = None
                self._poisoned = True
                self._pending = None
                base._fail(
                    "CACHE_PUBLICATION_POISONED",
                    "live state changed during durable commit publication",
                )
            self._publication_token = None
            if (
                not isinstance(publication, base.ReceiptPublication)
                or publication.receipt_id != request.commit_id
                or publication.sha256 != receipt.receipt_sha256
                or not publication.durable
                or not publication.readback_verified
            ):
                self._poisoned = True
                self._pending = None
                base._fail(
                    "COMMIT_RECEIPT_INVALID",
                    "publisher did not attest exact durable read-back bytes",
                )
            self._state = staged_state
            self._state_manifest_digest = after_digest
            if self._state.state_manifest_digest != receipt.state_manifest_after:
                self._poisoned = True
                self._pending = None
                base._fail(
                    "CACHE_PUBLICATION_POISONED",
                    "published state differs from durable decision",
                )
            self._completed[request.commit_id] = (
                transaction_payload_digest,
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
    ) -> Stage2BCacheResetReceipt:
        base._require_non_empty_string(reset_id, "reset_id")
        base._require_non_empty_string(transaction_nonce, "transaction_nonce")
        base._require_non_empty_string(new_episode_id, "new_episode_id")
        if not isinstance(layout, ChunkActionLayout):
            base._fail("CACHE_CONTENT_TIME_MISMATCH", "reset layout has wrong type")
        if layout.synthetic_test_only != self._synthetic_test_mode:
            base._fail(
                "CACHE_SYNTHETIC_LAYOUT_PROVENANCE",
                "reset cannot switch synthetic provenance",
            )
        payload_digest = base._sha256_bytes(
            base._canonical_json_bytes(
                {
                    "layout_instance_digest": layout.layout_instance_digest,
                    "layout_spec_sha256": layout.layout_spec_sha256,
                    "new_episode_id": new_episode_id,
                    "reset_id": reset_id,
                    "schema": _RESET_RECEIPT_SCHEMA,
                    "staging_variant": self._staging_variant.value,
                    "transaction_nonce": transaction_nonce,
                }
            )
        )
        with self._lock:
            self._assert_not_publishing()
            self._assert_not_poisoned()
            self._assert_live_integrity()
            if reset_id in self._completed:
                base._fail(
                    "COMMIT_DUPLICATE_CONFLICT",
                    "receipt id was already used by a commit transaction",
                )
            if self._pending is not None and self._pending[0] == reset_id:
                base._fail(
                    "COMMIT_DUPLICATE_CONFLICT",
                    "reset id conflicts with the pending commit transaction",
                )
            prior = self._completed_resets.get(reset_id)
            if prior is not None:
                prior_digest, prior_receipt = prior
                if prior_digest != payload_digest:
                    base._fail(
                        "COMMIT_DUPLICATE_CONFLICT",
                        "reset_id was reused for different inputs",
                    )
                if self._state.state_manifest_digest != prior_receipt.state_manifest_after:
                    base._fail("COMMIT_STALE_REVISION", "completed reset is stale")
                return prior_receipt
            before = self._state
            new_state = Stage2BTemporalState.empty(
                episode_id=new_episode_id,
                episode_epoch=before.episode_epoch + 1,
                layout=layout,
                registry=self._registry,
                staging_variant=self._staging_variant,
            )
            pending_before = self._pending
            aborted = self._pending[0] if self._pending is not None else None
            receipt = Stage2BCacheResetReceipt(
                reset_id=reset_id,
                staging_variant=self._staging_variant,
                transaction_nonce=transaction_nonce,
                previous_episode_id=before.episode_id,
                previous_episode_epoch=before.episode_epoch,
                new_episode_id=new_state.episode_id,
                new_episode_epoch=new_state.episode_epoch,
                layout_spec_sha256=layout.layout_spec_sha256,
                layout_instance_digest=layout.layout_instance_digest,
                state_manifest_before=before.state_manifest_digest,
                state_manifest_after=new_state.state_manifest_digest,
                action_history_digest_before=_history_digest(
                    before.committed_action_history
                ),
                action_history_digest_after=None,
                action_cursor_before=before.action_cursor,
                action_cursor_after=0,
                aborted_pending_commit_id=aborted,
                reset_at_monotonic_ns=time.monotonic_ns(),
            )
            publication_token = object()
            self._publication_token = publication_token
            try:
                publication = self._publisher.publish_exclusive(
                    receipt_id=reset_id,
                    payload=receipt.canonical_bytes,
                    expected_sha256=receipt.receipt_sha256,
                )
            except Exception as exc:
                self._publication_token = None
                self._poisoned = True
                raise base.CacheContractError(
                    "COMMIT_RECEIPT_PUBLICATION_FAILED",
                    f"reset publisher failed: {type(exc).__name__}",
                ) from exc
            if (
                self._publication_token is not publication_token
                or self._state is not before
                or self._pending is not pending_before
            ):
                self._publication_token = None
                self._poisoned = True
                base._fail(
                    "CACHE_PUBLICATION_POISONED",
                    "live state changed during durable reset publication",
                )
            self._publication_token = None
            if (
                not isinstance(publication, base.ReceiptPublication)
                or publication.receipt_id != reset_id
                or publication.sha256 != receipt.receipt_sha256
                or not publication.durable
                or not publication.readback_verified
            ):
                self._poisoned = True
                base._fail("COMMIT_RECEIPT_INVALID", "invalid reset publication")
            self._state = new_state
            self._layout = layout
            self._state_manifest_digest = new_state.state_manifest_digest
            self._pending = None
            self._completed_resets[reset_id] = (payload_digest, receipt)
            return receipt


CacheContractError = base.CacheContractError
CommitSource = base.CommitSource
ContentTime = base.ContentTime
LayerKind = base.LayerKind
LayerRegistry = base.LayerRegistry
LayerSpec = base.LayerSpec
ReceiptPublication = base.ReceiptPublication
StagedLayerPayload = base.StagedLayerPayload
TeacherForcingDatasetPairProof = base.TeacherForcingDatasetPairProof
tensor_digest = base.tensor_digest


__all__ = [
    "CACHStagingVariant",
    "CacheContractError",
    "CommitSource",
    "ContentTime",
    "LayerKind",
    "LayerRegistry",
    "LayerSpec",
    "ReceiptPublication",
    "Stage2BCacheResetReceipt",
    "Stage2BDenoiseReadView",
    "Stage2BHybridCacheManager",
    "Stage2BPairedCommitReceipt",
    "Stage2BPairedStagingContext",
    "Stage2BTeacherForcingCommitRequest",
    "Stage2BTemporalState",
    "StagedLayerPayload",
    "TeacherForcingDatasetPairProof",
    "tensor_digest",
]
