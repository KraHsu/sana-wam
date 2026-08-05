"""Frozen evidence for a synthetic Stage-2 failure-injection admission.

This is deliberately not a production run-root publisher and not the core
paired-transaction ``ABORTED`` ledger.  It gives mini admission a fail-closed,
exclusive, fsync/read-back-verified receipt without broadening its authority.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import stat


_SCHEMA = "cach.stage2_failure_injection_receipt.v1"
_SCOPE = "synthetic_mini_admission_only"
_ALLOWED_FAILURE_CODES = frozenset({"COMMIT_STAGING_FAILED"})
_ALLOWED_INJECTION_POINTS = frozenset({"stage_callback_before_vendor_forward"})


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short failure-receipt write")
        view = view[written:]


def _read_all(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 64 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _require_same_inode(
    left: os.stat_result,
    right: os.stat_result,
    name: str,
) -> None:
    if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
        raise OSError(f"{name} inode changed during failure-evidence freeze")


def _require_regular_single_link(value: os.stat_result, name: str) -> None:
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise OSError(f"{name} must remain a single-link regular file")


@dataclass(frozen=True)
class Stage2FailureInjectionReceipt:
    receipt_id: str
    commit_id: str
    transaction_nonce: str
    episode_id: str
    episode_epoch: int
    chunk_id: int
    layout_instance_digest: str
    source_proof_digest: str
    failure_code: str
    injection_point: str
    state_manifest_before: str
    state_manifest_after: str
    revision_before: int
    revision_after: int
    action_cursor_before: int
    action_cursor_after: int
    recorded_at_monotonic_ns: int
    result: str = "aborted"
    evidence_scope: str = _SCOPE
    scientific_eligible: bool = False
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "receipt_id",
            "commit_id",
            "transaction_nonce",
            "episode_id",
            "failure_code",
            "injection_point",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "episode_epoch",
            "chunk_id",
            "revision_before",
            "revision_after",
            "action_cursor_before",
            "action_cursor_after",
            "recorded_at_monotonic_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative plain int")
        for name in (
            "layout_instance_digest",
            "source_proof_digest",
            "state_manifest_before",
            "state_manifest_after",
        ):
            _require_sha(getattr(self, name), name)
        if self.schema != _SCHEMA or self.evidence_scope != _SCOPE:
            raise ValueError("failure receipt scope/schema differs")
        if self.failure_code not in _ALLOWED_FAILURE_CODES:
            raise ValueError("failure_code is not admitted for Stage-2 failure evidence")
        if self.injection_point not in _ALLOWED_INJECTION_POINTS:
            raise ValueError("injection_point is not admitted for Stage-2 failure evidence")
        if self.scientific_eligible is not False or self.result != "aborted":
            raise ValueError("Stage-2 failure evidence cannot claim success/science")
        if (
            self.state_manifest_after != self.state_manifest_before
            or self.revision_after != self.revision_before
            or self.action_cursor_after != self.action_cursor_before
        ):
            raise ValueError("aborted failure receipt requires an unchanged live state")

    def to_manifest(self) -> dict[str, object]:
        return {
            "action_cursor_after": self.action_cursor_after,
            "action_cursor_before": self.action_cursor_before,
            "chunk_id": self.chunk_id,
            "commit_id": self.commit_id,
            "episode_epoch": self.episode_epoch,
            "episode_id": self.episode_id,
            "evidence_scope": self.evidence_scope,
            "failure_code": self.failure_code,
            "injection_point": self.injection_point,
            "layout_instance_digest": self.layout_instance_digest,
            "receipt_id": self.receipt_id,
            "recorded_at_monotonic_ns": self.recorded_at_monotonic_ns,
            "result": self.result,
            "revision_after": self.revision_after,
            "revision_before": self.revision_before,
            "schema": self.schema,
            "scientific_eligible": self.scientific_eligible,
            "source_proof_digest": self.source_proof_digest,
            "state_manifest_after": self.state_manifest_after,
            "state_manifest_before": self.state_manifest_before,
            "transaction_nonce": self.transaction_nonce,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_manifest(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes)


@dataclass(frozen=True)
class FrozenFailureEvidence:
    root: Path
    receipt_path: Path
    receipt_sha256: str


def freeze_stage2_failure_evidence(
    root: Path,
    receipt: Stage2FailureInjectionReceipt,
) -> FrozenFailureEvidence:
    """Exclusive-create, verify, and freeze one temporary failure root."""

    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("failure evidence root must be an absolute pathlib.Path")
    if not isinstance(receipt, Stage2FailureInjectionReceipt):
        raise TypeError("receipt must be Stage2FailureInjectionReceipt")
    if root.name in {"", ".", ".."}:
        raise ValueError("failure evidence root must name a new child directory")

    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    except AttributeError as error:  # pragma: no cover - H200 is Linux.
        raise OSError("failure evidence requires O_DIRECTORY and O_NOFOLLOW") from error
    parent_fd = os.open(root.parent, directory_flags)
    try:
        os.mkdir(root.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        directory_fd = os.open(root.name, directory_flags, dir_fd=parent_fd)
        try:
            root_inode = os.fstat(directory_fd)
            root_entry = os.stat(
                root.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(root_inode.st_mode):
                raise OSError("failure evidence root fd is not a directory")
            _require_same_inode(root_inode, root_entry, "failure evidence root")

            file_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            descriptor = os.open(
                "FAILURE.json",
                file_flags,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                receipt_inode = os.fstat(descriptor)
                _require_regular_single_link(receipt_inode, "failure receipt")
                _write_all(descriptor, receipt.canonical_bytes)
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)

                writer_readback = _read_all(descriptor)
                writer_after = os.fstat(descriptor)
                _require_same_inode(receipt_inode, writer_after, "failure receipt")
                _require_regular_single_link(writer_after, "failure receipt")
                if stat.S_IMODE(writer_after.st_mode) != 0o400:
                    raise OSError("failure receipt did not freeze to mode 0400")
                if (
                    writer_readback != receipt.canonical_bytes
                    or _sha256(writer_readback) != receipt.receipt_sha256
                ):
                    raise OSError("failure receipt fd read-back verification failed")
            finally:
                os.close(descriptor)

            receipt_entry = os.stat(
                "FAILURE.json",
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            _require_same_inode(receipt_inode, receipt_entry, "failure receipt")
            _require_regular_single_link(receipt_entry, "failure receipt")

            read_flags = os.O_RDONLY | os.O_NOFOLLOW
            descriptor = os.open(
                "FAILURE.json",
                read_flags,
                dir_fd=directory_fd,
            )
            try:
                reader_inode = os.fstat(descriptor)
                _require_same_inode(receipt_inode, reader_inode, "failure receipt")
                _require_regular_single_link(reader_inode, "failure receipt")
                reader_readback = _read_all(descriptor)
                reader_after = os.fstat(descriptor)
                _require_same_inode(reader_inode, reader_after, "failure receipt")
                _require_regular_single_link(reader_after, "failure receipt")
            finally:
                os.close(descriptor)
            if reader_readback != receipt.canonical_bytes:
                raise OSError("failure receipt differs on stable fd read-back")

            os.fsync(directory_fd)
            os.fchmod(directory_fd, 0o500)
            os.fsync(directory_fd)

            root_after = os.fstat(directory_fd)
            root_entry_after = os.stat(
                root.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            _require_same_inode(root_inode, root_after, "failure evidence root")
            _require_same_inode(root_inode, root_entry_after, "failure evidence root")
            if stat.S_IMODE(root_after.st_mode) != 0o500:
                raise OSError("failure evidence root did not freeze to mode 0500")

            receipt_entry_after = os.stat(
                "FAILURE.json",
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            _require_same_inode(receipt_inode, receipt_entry_after, "failure receipt")
            _require_regular_single_link(receipt_entry_after, "failure receipt")
            if stat.S_IMODE(receipt_entry_after.st_mode) != 0o400:
                raise OSError("failure receipt did not retain mode 0400")

            descriptor = os.open(
                "FAILURE.json",
                read_flags,
                dir_fd=directory_fd,
            )
            try:
                final_reader_inode = os.fstat(descriptor)
                _require_same_inode(
                    receipt_inode,
                    final_reader_inode,
                    "failure receipt",
                )
                _require_regular_single_link(final_reader_inode, "failure receipt")
                final_readback = _read_all(descriptor)
                final_reader_after = os.fstat(descriptor)
                _require_same_inode(
                    final_reader_inode,
                    final_reader_after,
                    "failure receipt",
                )
                _require_regular_single_link(final_reader_after, "failure receipt")
            finally:
                os.close(descriptor)
            if final_readback != receipt.canonical_bytes:
                raise OSError("frozen failure receipt differs on final fd read-back")

        finally:
            os.close(directory_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)

    receipt_path = root / "FAILURE.json"
    return FrozenFailureEvidence(
        root=root,
        receipt_path=receipt_path,
        receipt_sha256=_sha256(final_readback),
    )


__all__ = [
    "FrozenFailureEvidence",
    "Stage2FailureInjectionReceipt",
    "freeze_stage2_failure_evidence",
]
