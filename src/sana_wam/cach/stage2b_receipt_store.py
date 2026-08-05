"""Strict offline filesystem publisher for Stage-2B receipts.

This module is only a local durability boundary for the two successful
Stage-2B transaction receipts.  It deliberately does **not** provide an
``ABORTED`` ledger, a state snapshot, restart/replay authority, or scientific
admission/eligibility.  A verified receipt says only that the exact canonical
JSON bytes supplied by the caller were durably published under an exclusive
identifier in this dedicated directory.

The directory must already exist, be absolute, be empty when the store is
constructed, be owned by the effective user, be neither group- nor
world-writable, and must not be a symlink. Receipt filenames are SHA256 hashes
of the logical receipt identifiers; identifiers are never interpreted as
paths. Parent/ancestor fencing remains outside this bounded L0 implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any

from sana_wam.model.video_backbone.sana import hybrid_cache as base


_COMMIT_SCHEMA = "cach.stage2b.paired_commit_receipt.v1"
_RESET_SCHEMA = "cach.stage2b.cache_reset_receipt.v1"
_TEACHER_PROOF_SCHEMA = "cach.teacher_forcing_dataset_pair_proof.v2"
_STAGING_VARIANTS = frozenset({"cach_a", "ref_gdn_corrected"})

_COMMIT_KEYS = frozenset(
    {
        "action_cursor_after",
        "action_cursor_before",
        "action_history_digest_after",
        "action_history_digest_before",
        "chunk_id",
        "commit_id",
        "commit_source",
        "committed_at_monotonic_ns",
        "conditioning_digest",
        "dataset_manifest_sha256",
        "dataset_row_identity",
        "episode_epoch",
        "episode_id",
        "layout_instance_digest",
        "pair_action_digest",
        "pair_action_mask_digest",
        "pair_frame_valid_mask_digest",
        "pair_video_digest",
        "paired_payload_digest",
        "result",
        "revision_after",
        "revision_before",
        "schema",
        "source_proof_digest",
        "source_proof_schema",
        "staged_state_manifest",
        "staging_variant",
        "state_manifest_after",
        "state_manifest_before",
        "teacher_pair_payload_digest",
        "transaction_nonce",
    }
)

_RESET_KEYS = frozenset(
    {
        "aborted_pending_commit_id",
        "action_cursor_after",
        "action_cursor_before",
        "action_history_digest_after",
        "action_history_digest_before",
        "layout_instance_digest",
        "layout_spec_sha256",
        "new_episode_epoch",
        "new_episode_id",
        "previous_episode_epoch",
        "previous_episode_id",
        "reset_at_monotonic_ns",
        "reset_id",
        "schema",
        "staging_variant",
        "state_manifest_after",
        "state_manifest_before",
        "transaction_nonce",
    }
)


class Stage2BReceiptStoreError(RuntimeError):
    """A receipt failed validation or stable filesystem verification."""


class Stage2BReceiptConflictError(FileExistsError):
    """The logical receipt identifier already has a directory entry."""


@dataclass(frozen=True)
class StoredStage2BReceipt:
    """Stable, read-only receipt bytes and their verified identity."""

    receipt_id: str
    schema: str
    payload: bytes
    sha256: str
    path: Path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage2BReceiptStoreError(f"{name} must be a lowercase SHA256")
    return value


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Stage2BReceiptStoreError(f"{name} must be a non-empty string")
    return value


def _require_plain_uint(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise Stage2BReceiptStoreError(f"{name} must be a non-negative plain int")
    return value


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2BReceiptStoreError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise Stage2BReceiptStoreError(f"non-finite JSON constant is forbidden: {value}")


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
        raise Stage2BReceiptStoreError(
            "receipt is not canonical-JSON encodable"
        ) from exc


def _parse_canonical_manifest(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise Stage2BReceiptStoreError("receipt payload must be non-empty bytes")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except Stage2BReceiptStoreError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2BReceiptStoreError("receipt payload is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise Stage2BReceiptStoreError("receipt payload must be one JSON object")
    if _canonical_json_bytes(value) != payload:
        raise Stage2BReceiptStoreError("receipt payload is not canonical JSON")
    return value


def _require_exact_keys(manifest: dict[str, Any], expected: frozenset[str]) -> None:
    observed = frozenset(manifest)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise Stage2BReceiptStoreError(
            f"receipt schema members differ (missing={missing}, extra={extra})"
        )


def _require_optional_sha256(value: object, name: str) -> None:
    if value is not None:
        _require_sha256(value, name)


def _validate_commit(manifest: dict[str, Any], receipt_id: str) -> None:
    _require_exact_keys(manifest, _COMMIT_KEYS)
    if manifest["commit_id"] != receipt_id:
        raise Stage2BReceiptStoreError("embedded commit_id differs from receipt_id")
    if manifest["schema"] != _COMMIT_SCHEMA:
        raise Stage2BReceiptStoreError("commit receipt schema differs")
    if manifest["commit_source"] != "teacher_forcing_dataset_pair":
        raise Stage2BReceiptStoreError(
            "Stage-2B commit must be teacher_forcing_dataset_pair"
        )
    if manifest["source_proof_schema"] != _TEACHER_PROOF_SCHEMA:
        raise Stage2BReceiptStoreError("teacher source proof schema differs")
    if manifest["result"] != "committed":
        raise Stage2BReceiptStoreError("Stage-2B receipt is not committed")
    if (
        not isinstance(manifest["staging_variant"], str)
        or manifest["staging_variant"] not in _STAGING_VARIANTS
    ):
        raise Stage2BReceiptStoreError("unknown Stage-2B staging variant")

    for name in (
        "commit_id",
        "transaction_nonce",
        "episode_id",
        "dataset_row_identity",
    ):
        _require_non_empty_string(manifest[name], name)
    for name in (
        "episode_epoch",
        "revision_before",
        "revision_after",
        "chunk_id",
        "action_cursor_before",
        "action_cursor_after",
        "committed_at_monotonic_ns",
    ):
        _require_plain_uint(manifest[name], name)
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
        _require_sha256(manifest[name], name)
    _require_optional_sha256(
        manifest["action_history_digest_before"],
        "action_history_digest_before",
    )
    _require_optional_sha256(manifest["conditioning_digest"], "conditioning_digest")

    revision_before = manifest["revision_before"]
    if manifest["revision_after"] != revision_before + 1:
        raise Stage2BReceiptStoreError("commit revision must advance exactly once")
    if manifest["chunk_id"] != revision_before:
        raise Stage2BReceiptStoreError("commit chunk must equal the prior revision")
    if manifest["state_manifest_after"] == manifest["state_manifest_before"]:
        raise Stage2BReceiptStoreError("commit state manifest must advance")
    if manifest["staged_state_manifest"] != manifest["state_manifest_after"]:
        raise Stage2BReceiptStoreError(
            "staged state must equal the committed state-after manifest"
        )
    if (revision_before == 0) != (manifest["action_history_digest_before"] is None):
        raise Stage2BReceiptStoreError(
            "history-before must be absent exactly at initial revision"
        )
    if (revision_before == 0) != (manifest["action_cursor_before"] == 0):
        raise Stage2BReceiptStoreError(
            "cursor-before must be zero exactly at initial revision"
        )
    if manifest["action_cursor_after"] <= manifest["action_cursor_before"]:
        raise Stage2BReceiptStoreError("commit action cursor must advance")
    if (
        manifest["action_history_digest_before"] is not None
        and manifest["action_history_digest_after"]
        == manifest["action_history_digest_before"]
    ):
        raise Stage2BReceiptStoreError("commit action-history digest must advance")
    expected_paired_payload_digest = _sha256(
        _canonical_json_bytes(
            {
                "conditioning_digest": manifest["conditioning_digest"],
                "schema": "cach.stage2b.paired_payload.v1",
                "staging_variant": manifest["staging_variant"],
                "teacher_pair_payload_digest": manifest[
                    "teacher_pair_payload_digest"
                ],
            }
        )
    )
    if manifest["paired_payload_digest"] != expected_paired_payload_digest:
        raise Stage2BReceiptStoreError("paired payload digest relation differs")


def _validate_reset(manifest: dict[str, Any], receipt_id: str) -> None:
    _require_exact_keys(manifest, _RESET_KEYS)
    if manifest["reset_id"] != receipt_id:
        raise Stage2BReceiptStoreError("embedded reset_id differs from receipt_id")
    if manifest["schema"] != _RESET_SCHEMA:
        raise Stage2BReceiptStoreError("reset receipt schema differs")
    if (
        not isinstance(manifest["staging_variant"], str)
        or manifest["staging_variant"] not in _STAGING_VARIANTS
    ):
        raise Stage2BReceiptStoreError("unknown Stage-2B staging variant")

    for name in (
        "reset_id",
        "transaction_nonce",
        "previous_episode_id",
        "new_episode_id",
    ):
        _require_non_empty_string(manifest[name], name)
    for name in (
        "previous_episode_epoch",
        "new_episode_epoch",
        "action_cursor_before",
        "action_cursor_after",
        "reset_at_monotonic_ns",
    ):
        _require_plain_uint(manifest[name], name)
    for name in (
        "layout_instance_digest",
        "layout_spec_sha256",
        "state_manifest_before",
        "state_manifest_after",
    ):
        _require_sha256(manifest[name], name)
    _require_optional_sha256(
        manifest["action_history_digest_before"],
        "action_history_digest_before",
    )
    _require_optional_sha256(
        manifest["action_history_digest_after"],
        "action_history_digest_after",
    )
    aborted = manifest["aborted_pending_commit_id"]
    if aborted is not None:
        _require_non_empty_string(aborted, "aborted_pending_commit_id")
    if manifest["new_episode_epoch"] != manifest["previous_episode_epoch"] + 1:
        raise Stage2BReceiptStoreError("reset episode epoch must advance exactly once")
    if manifest["action_cursor_after"] != 0:
        raise Stage2BReceiptStoreError("reset action cursor must be zero")
    if manifest["action_history_digest_after"] is not None:
        raise Stage2BReceiptStoreError("reset must clear committed-action history")
    if (manifest["action_cursor_before"] == 0) != (
        manifest["action_history_digest_before"] is None
    ):
        raise Stage2BReceiptStoreError(
            "reset cursor/history-before emptiness differs"
        )
    if manifest["state_manifest_after"] == manifest["state_manifest_before"]:
        raise Stage2BReceiptStoreError("reset state manifest must change")
    if manifest["aborted_pending_commit_id"] == receipt_id:
        raise Stage2BReceiptStoreError("reset cannot abort its own transaction ID")


def _validate_payload(payload: bytes, receipt_id: str) -> tuple[dict[str, Any], str]:
    manifest = _parse_canonical_manifest(payload)
    schema = manifest.get("schema")
    if schema == _COMMIT_SCHEMA:
        _validate_commit(manifest, receipt_id)
    elif schema == _RESET_SCHEMA:
        _validate_reset(manifest, receipt_id)
    else:
        raise Stage2BReceiptStoreError("only Stage-2B commit/reset receipts are accepted")
    return manifest, schema


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short Stage-2B receipt write")
        remaining = remaining[written:]


def _read_all(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 64 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _require_regular_single_link(value: os.stat_result, name: str) -> None:
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise Stage2BReceiptStoreError(f"{name} must be a single-link regular file")


def _require_same_inode(
    left: os.stat_result,
    right: os.stat_result,
    name: str,
) -> None:
    if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
        raise Stage2BReceiptStoreError(f"{name} inode changed")


def _require_secure_store_directory(value: os.stat_result) -> None:
    if value.st_uid != os.geteuid():
        raise Stage2BReceiptStoreError(
            "receipt root must be owned by the effective user"
        )
    if stat.S_IMODE(value.st_mode) & 0o022:
        raise Stage2BReceiptStoreError(
            "receipt root must not be group/world writable"
        )


def _stable_file_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class Stage2BFilesystemReceiptStore:
    """Exclusive filesystem implementation of ``DurableReceiptPublisher``.

    Construction proves that ``root`` is an existing, absolute, empty,
    owner-only-writable, non-symlink directory. The stored inode is revalidated
    at the beginning and end of every call. The class creates no directories
    and never replaces or removes entries.
    """

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("receipt root must be an absolute pathlib.Path")
        self._root = root
        directory_fd = self._open_directory_unbound()
        try:
            directory_stat = os.fstat(directory_fd)
            entry_stat = os.stat(root, follow_symlinks=False)
            if not stat.S_ISDIR(directory_stat.st_mode):
                raise Stage2BReceiptStoreError("receipt root is not a directory")
            _require_same_inode(directory_stat, entry_stat, "receipt root")
            _require_secure_store_directory(directory_stat)
            if os.listdir(directory_fd):
                raise Stage2BReceiptStoreError(
                    "receipt root must be a dedicated empty directory"
                )
            self._root_identity = (directory_stat.st_dev, directory_stat.st_ino)
        finally:
            os.close(directory_fd)

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def receipt_filename(receipt_id: str) -> str:
        identifier = _require_non_empty_string(receipt_id, "receipt_id")
        try:
            encoded = identifier.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise Stage2BReceiptStoreError("receipt_id is not valid UTF-8") from exc
        return hashlib.sha256(encoded).hexdigest() + ".json"

    def receipt_path(self, receipt_id: str) -> Path:
        return self._root / self.receipt_filename(receipt_id)

    def _open_directory_unbound(self) -> int:
        required = ("O_DIRECTORY", "O_NOFOLLOW")
        if any(not hasattr(os, name) for name in required):  # pragma: no cover
            raise OSError("Stage-2B receipt store requires O_DIRECTORY/O_NOFOLLOW")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        return os.open(self._root, flags)

    def _open_directory(self) -> int:
        descriptor = self._open_directory_unbound()
        try:
            self._require_bound_directory(descriptor)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _require_bound_directory(self, descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        entry_stat = os.stat(self._root, follow_symlinks=False)
        if not stat.S_ISDIR(descriptor_stat.st_mode):
            raise Stage2BReceiptStoreError("receipt root fd is not a directory")
        _require_same_inode(descriptor_stat, entry_stat, "receipt root")
        _require_secure_store_directory(descriptor_stat)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != self._root_identity:
            raise Stage2BReceiptStoreError("receipt root was replaced")

    def publish_exclusive(
        self,
        *,
        receipt_id: str,
        payload: bytes,
        expected_sha256: str,
    ) -> base.ReceiptPublication:
        """Validate and durably exclusive-create one Stage-2B receipt.

        Reusing an identifier is always a conflict, even when the existing
        bytes are identical.  This method never claims idempotent publication.
        """

        filename = self.receipt_filename(receipt_id)
        expected = _require_sha256(expected_sha256, "expected_sha256")
        _validate_payload(payload, receipt_id)
        observed = _sha256(payload)
        if observed != expected:
            raise Stage2BReceiptStoreError("receipt payload SHA256 differs")

        directory_fd = self._open_directory()
        try:
            descriptor: int | None = None
            try:
                flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                flags |= getattr(os, "O_CLOEXEC", 0)
                try:
                    descriptor = os.open(
                        filename,
                        flags,
                        0o600,
                        dir_fd=directory_fd,
                    )
                except FileExistsError as exc:
                    raise Stage2BReceiptConflictError(
                        f"receipt_id already exists: {receipt_id!r}"
                    ) from exc

                created_stat = os.fstat(descriptor)
                _require_regular_single_link(created_stat, "created receipt")
                _write_all(descriptor, payload)
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)

                writer_before = os.fstat(descriptor)
                _require_same_inode(created_stat, writer_before, "created receipt")
                _require_regular_single_link(writer_before, "created receipt")
                writer_payload = _read_all(descriptor)
                writer_after = os.fstat(descriptor)
                if _stable_file_fingerprint(
                    writer_before
                ) != _stable_file_fingerprint(writer_after):
                    raise Stage2BReceiptStoreError(
                        "receipt changed during writer-fd read-back"
                    )
                if stat.S_IMODE(writer_after.st_mode) != 0o400:
                    raise Stage2BReceiptStoreError("receipt mode is not 0400")
                if writer_payload != payload or _sha256(writer_payload) != expected:
                    raise Stage2BReceiptStoreError(
                        "writer-fd receipt read-back differs"
                    )
            finally:
                if descriptor is not None:
                    os.close(descriptor)

            entry_stat = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
            _require_same_inode(created_stat, entry_stat, "published receipt")
            _require_regular_single_link(entry_stat, "published receipt")
            if stat.S_IMODE(entry_stat.st_mode) != 0o400:
                raise Stage2BReceiptStoreError("published receipt mode is not 0400")

            os.fsync(directory_fd)
            stable = self._read_from_directory(
                directory_fd=directory_fd,
                receipt_id=receipt_id,
                filename=filename,
            )
            if stable.payload != payload or stable.sha256 != expected:
                raise Stage2BReceiptStoreError("reopened receipt read-back differs")
            self._require_bound_directory(directory_fd)
            return base.ReceiptPublication(
                receipt_id=receipt_id,
                sha256=expected,
                durable=True,
                readback_verified=True,
            )
        finally:
            os.close(directory_fd)

    def _read_from_directory(
        self,
        *,
        directory_fd: int,
        receipt_id: str,
        filename: str,
    ) -> StoredStage2BReceipt:
        try:
            entry_before = os.stat(
                filename,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            raise
        _require_regular_single_link(entry_before, "receipt entry")
        if stat.S_IMODE(entry_before.st_mode) != 0o400:
            raise Stage2BReceiptStoreError("receipt entry mode is not 0400")

        flags = os.O_RDONLY | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
        try:
            reader_before = os.fstat(descriptor)
            _require_same_inode(entry_before, reader_before, "receipt entry")
            _require_regular_single_link(reader_before, "receipt fd")
            payload = _read_all(descriptor)
            reader_after = os.fstat(descriptor)
            if _stable_file_fingerprint(reader_before) != _stable_file_fingerprint(
                reader_after
            ):
                raise Stage2BReceiptStoreError("receipt changed during stable read")
        finally:
            os.close(descriptor)

        entry_after = os.stat(
            filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        _require_same_inode(reader_after, entry_after, "receipt entry")
        _require_regular_single_link(entry_after, "receipt entry")
        if _stable_file_fingerprint(reader_after) != _stable_file_fingerprint(
            entry_after
        ):
            raise Stage2BReceiptStoreError("receipt entry changed during stable read")
        _, schema = _validate_payload(payload, receipt_id)
        return StoredStage2BReceipt(
            receipt_id=receipt_id,
            schema=schema,
            payload=payload,
            sha256=_sha256(payload),
            path=self._root / filename,
        )

    def read_stable(self, receipt_id: str) -> StoredStage2BReceipt:
        """Read and revalidate one immutable receipt without replay authority."""

        filename = self.receipt_filename(receipt_id)
        directory_fd = self._open_directory()
        try:
            stored = self._read_from_directory(
                directory_fd=directory_fd,
                receipt_id=receipt_id,
                filename=filename,
            )
            self._require_bound_directory(directory_fd)
            return stored
        finally:
            os.close(directory_fd)


__all__ = [
    "Stage2BFilesystemReceiptStore",
    "Stage2BReceiptConflictError",
    "Stage2BReceiptStoreError",
    "StoredStage2BReceipt",
]
