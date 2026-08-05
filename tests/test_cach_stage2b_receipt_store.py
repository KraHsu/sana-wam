from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from sana_wam.cach import stage2b_receipt_store as receipt_store_module
from sana_wam.cach.stage2b_receipt_store import (
    Stage2BFilesystemReceiptStore,
    Stage2BReceiptConflictError,
    Stage2BReceiptStoreError,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _mkdir_secure(path: Path) -> None:
    path.mkdir(mode=0o700)
    path.chmod(0o700)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _commit_manifest(receipt_id: str) -> dict[str, object]:
    manifest: dict[str, object] = {
        "action_cursor_after": 4,
        "action_cursor_before": 0,
        "action_history_digest_after": SHA_A,
        "action_history_digest_before": None,
        "chunk_id": 0,
        "commit_id": receipt_id,
        "commit_source": "teacher_forcing_dataset_pair",
        "committed_at_monotonic_ns": 123,
        "conditioning_digest": SHA_B,
        "dataset_manifest_sha256": SHA_C,
        "dataset_row_identity": "row-0",
        "episode_epoch": 0,
        "episode_id": "episode-0",
        "layout_instance_digest": SHA_A,
        "pair_action_digest": SHA_B,
        "pair_action_mask_digest": SHA_C,
        "pair_frame_valid_mask_digest": SHA_A,
        "pair_video_digest": SHA_B,
        "paired_payload_digest": "",
        "result": "committed",
        "revision_after": 1,
        "revision_before": 0,
        "schema": "cach.stage2b.paired_commit_receipt.v1",
        "source_proof_digest": SHA_A,
        "source_proof_schema": "cach.teacher_forcing_dataset_pair_proof.v2",
        "staged_state_manifest": SHA_C,
        "staging_variant": "cach_a",
        "state_manifest_after": SHA_C,
        "state_manifest_before": SHA_A,
        "teacher_pair_payload_digest": SHA_B,
        "transaction_nonce": "nonce-0",
    }
    manifest["paired_payload_digest"] = hashlib.sha256(
        _canonical(
            {
                "conditioning_digest": manifest["conditioning_digest"],
                "schema": "cach.stage2b.paired_payload.v1",
                "staging_variant": manifest["staging_variant"],
                "teacher_pair_payload_digest": manifest[
                    "teacher_pair_payload_digest"
                ],
            }
        )
    ).hexdigest()
    return manifest


def _reset_manifest(receipt_id: str) -> dict[str, object]:
    return {
        "aborted_pending_commit_id": None,
        "action_cursor_after": 0,
        "action_cursor_before": 4,
        "action_history_digest_after": None,
        "action_history_digest_before": SHA_A,
        "layout_instance_digest": SHA_B,
        "layout_spec_sha256": SHA_C,
        "new_episode_epoch": 1,
        "new_episode_id": "episode-1",
        "previous_episode_epoch": 0,
        "previous_episode_id": "episode-0",
        "reset_at_monotonic_ns": 456,
        "reset_id": receipt_id,
        "schema": "cach.stage2b.cache_reset_receipt.v1",
        "staging_variant": "ref_gdn_corrected",
        "state_manifest_after": SHA_A,
        "state_manifest_before": SHA_B,
        "transaction_nonce": "reset-nonce-0",
    }


def _publish(
    store: Stage2BFilesystemReceiptStore,
    receipt_id: str,
    manifest: dict[str, object],
):
    payload = _canonical(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    publication = store.publish_exclusive(
        receipt_id=receipt_id,
        payload=payload,
        expected_sha256=digest,
    )
    return payload, digest, publication


def test_publish_commit_is_exact_frozen_and_stably_readable(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = "commit-0"

    payload, digest, publication = _publish(
        store,
        receipt_id,
        _commit_manifest(receipt_id),
    )

    expected_name = hashlib.sha256(receipt_id.encode("utf-8")).hexdigest() + ".json"
    path = root / expected_name
    assert [entry.name for entry in root.iterdir()] == [expected_name]
    assert path.read_bytes() == payload
    file_stat = path.lstat()
    assert stat.S_ISREG(file_stat.st_mode)
    assert file_stat.st_nlink == 1
    assert stat.S_IMODE(file_stat.st_mode) == 0o400
    assert publication.receipt_id == receipt_id
    assert publication.sha256 == digest
    assert publication.durable is True
    assert publication.readback_verified is True

    stable = store.read_stable(receipt_id)
    assert stable.receipt_id == receipt_id
    assert stable.schema == "cach.stage2b.paired_commit_receipt.v1"
    assert stable.payload == payload
    assert stable.sha256 == digest
    assert stable.path == path


def test_publish_reset_schema_is_accepted(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())

    payload, digest, _ = _publish(store, "reset-0", _reset_manifest("reset-0"))
    stable = store.read_stable("reset-0")
    assert stable.schema == "cach.stage2b.cache_reset_receipt.v1"
    assert stable.payload == payload
    assert stable.sha256 == digest


def test_same_and_different_payload_reuse_are_both_exclusive_conflicts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = "commit-collision"
    first_manifest = _commit_manifest(receipt_id)
    first_payload, first_digest, _ = _publish(store, receipt_id, first_manifest)

    with pytest.raises(Stage2BReceiptConflictError):
        store.publish_exclusive(
            receipt_id=receipt_id,
            payload=first_payload,
            expected_sha256=first_digest,
        )

    second_manifest = _commit_manifest(receipt_id)
    second_manifest["transaction_nonce"] = "different-nonce"
    second_payload = _canonical(second_manifest)
    with pytest.raises(Stage2BReceiptConflictError):
        store.publish_exclusive(
            receipt_id=receipt_id,
            payload=second_payload,
            expected_sha256=hashlib.sha256(second_payload).hexdigest(),
        )

    assert store.read_stable(receipt_id).payload == first_payload


def test_receipt_id_is_hashed_and_never_interpreted_as_path(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = "../../outside/receipt.json"

    payload, _, _ = _publish(store, receipt_id, _commit_manifest(receipt_id))

    entries = list(root.iterdir())
    assert len(entries) == 1
    assert entries[0].name == store.receipt_filename(receipt_id)
    assert entries[0].parent == root
    assert "/" not in entries[0].name
    assert store.read_stable(receipt_id).payload == payload
    assert not (tmp_path / "outside").exists()


def test_bad_hash_and_noncanonical_json_fail_before_creation(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = "commit-invalid"
    manifest = _commit_manifest(receipt_id)
    payload = _canonical(manifest)

    with pytest.raises(Stage2BReceiptStoreError, match="SHA256 differs"):
        store.publish_exclusive(
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256="0" * 64,
        )
    assert list(root.iterdir()) == []

    noncanonical = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(Stage2BReceiptStoreError, match="not canonical JSON"):
        store.publish_exclusive(
            receipt_id=receipt_id,
            payload=noncanonical,
            expected_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )
    assert list(root.iterdir()) == []


def test_embedded_id_and_schema_are_verified_before_creation(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())

    wrong_id = _canonical(_commit_manifest("embedded-other"))
    with pytest.raises(Stage2BReceiptStoreError, match="embedded commit_id"):
        store.publish_exclusive(
            receipt_id="logical-id",
            payload=wrong_id,
            expected_sha256=hashlib.sha256(wrong_id).hexdigest(),
        )

    unknown = _commit_manifest("logical-id")
    unknown["schema"] = "cach.stage2b.aborted_receipt.v1"
    unknown_payload = _canonical(unknown)
    with pytest.raises(Stage2BReceiptStoreError, match="only Stage-2B commit/reset"):
        store.publish_exclusive(
            receipt_id="logical-id",
            payload=unknown_payload,
            expected_sha256=hashlib.sha256(unknown_payload).hexdigest(),
        )
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("chunk_id", 1, "chunk must equal"),
        ("staged_state_manifest", SHA_B, "staged state must equal"),
        ("state_manifest_after", SHA_A, "state manifest must advance"),
    ),
)
def test_commit_transition_relations_are_independently_verified(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = "commit-transition"
    manifest = _commit_manifest(receipt_id)
    manifest[field] = value
    payload = _canonical(manifest)

    with pytest.raises(Stage2BReceiptStoreError, match=message):
        store.publish_exclusive(
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
    assert list(root.iterdir()) == []


def test_reset_must_change_state_manifest(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = "reset-transition"
    manifest = _reset_manifest(receipt_id)
    manifest["state_manifest_after"] = manifest["state_manifest_before"]
    payload = _canonical(manifest)

    with pytest.raises(Stage2BReceiptStoreError, match="state manifest must change"):
        store.publish_exclusive(
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("action_cursor_before", 1, "zero exactly"),
        ("paired_payload_digest", SHA_A, "digest relation"),
    ),
)
def test_commit_cross_field_identity_relations_are_verified(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = "commit-cross-field"
    manifest = _commit_manifest(receipt_id)
    manifest[field] = value
    payload = _canonical(manifest)

    with pytest.raises(Stage2BReceiptStoreError, match=message):
        store.publish_exclusive(
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
    assert list(root.iterdir()) == []


def test_reset_rejects_self_abort_and_cursor_history_mismatch(tmp_path: Path) -> None:
    for receipt_id, mutation, message in (
        (
            "reset-self-abort",
            {"aborted_pending_commit_id": "reset-self-abort"},
            "cannot abort its own",
        ),
        (
            "reset-history-mismatch",
            {"action_cursor_before": 0},
            "emptiness differs",
        ),
    ):
        root = tmp_path / receipt_id
        _mkdir_secure(root)
        store = Stage2BFilesystemReceiptStore(root.resolve())
        manifest = _reset_manifest(receipt_id)
        manifest.update(mutation)
        payload = _canonical(manifest)
        with pytest.raises(Stage2BReceiptStoreError, match=message):
            store.publish_exclusive(
                receipt_id=receipt_id,
                payload=payload,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
            )
        assert list(root.iterdir()) == []


def test_root_must_be_absolute_existing_empty_secure_and_not_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="absolute"):
        Stage2BFilesystemReceiptStore(Path("relative/receipts"))

    with pytest.raises(FileNotFoundError):
        Stage2BFilesystemReceiptStore((tmp_path / "missing").resolve())

    nonempty = tmp_path / "nonempty"
    _mkdir_secure(nonempty)
    (nonempty / "foreign").write_bytes(b"x")
    with pytest.raises(Stage2BReceiptStoreError, match="dedicated empty"):
        Stage2BFilesystemReceiptStore(nonempty.resolve())

    writable = tmp_path / "group-world-writable"
    writable.mkdir(mode=0o777)
    writable.chmod(0o777)
    with pytest.raises(Stage2BReceiptStoreError, match="group/world writable"):
        Stage2BFilesystemReceiptStore(writable.resolve())

    actual = tmp_path / "actual"
    _mkdir_secure(actual)
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(OSError):
        Stage2BFilesystemReceiptStore(linked.absolute())


def test_symlink_receipt_is_conflict_and_never_followed(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = "symlink-receipt"
    target = tmp_path / "outside-target"
    target.write_bytes(b"outside")
    link = store.receipt_path(receipt_id)
    link.symlink_to(target)
    payload = _canonical(_commit_manifest(receipt_id))

    with pytest.raises(Stage2BReceiptConflictError):
        store.publish_exclusive(
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
    with pytest.raises(Stage2BReceiptStoreError, match="single-link regular"):
        store.read_stable(receipt_id)
    assert target.read_bytes() == b"outside"


def test_hardlinked_receipt_is_rejected_by_stable_read(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = "hardlinked-receipt"
    _publish(store, receipt_id, _commit_manifest(receipt_id))
    os.link(store.receipt_path(receipt_id), tmp_path / "second-link")

    with pytest.raises(Stage2BReceiptStoreError, match="single-link regular"):
        store.read_stable(receipt_id)


def test_root_replacement_during_publish_is_detected_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    original_read = store._read_from_directory
    moved = tmp_path / "moved-receipts"

    def read_then_replace_root(**kwargs):
        stored = original_read(**kwargs)
        root.rename(moved)
        _mkdir_secure(root)
        return stored

    monkeypatch.setattr(store, "_read_from_directory", read_then_replace_root)
    receipt_id = "root-replaced"
    payload = _canonical(_commit_manifest(receipt_id))
    with pytest.raises(Stage2BReceiptStoreError, match="receipt root"):
        store.publish_exclusive(
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
    assert list(root.iterdir()) == []
    assert len(tuple(moved.iterdir())) == 1


def test_partial_writes_complete_and_fsync_order_is_file_mode_then_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "receipts"
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = "partial-write"
    manifest = _commit_manifest(receipt_id)
    real_write = os.write
    real_fsync = os.fsync
    events: list[tuple[str, int]] = []

    def partial_write(descriptor: int, payload) -> int:
        return real_write(descriptor, payload[: min(7, len(payload))])

    def recording_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        kind = "directory" if stat.S_ISDIR(metadata.st_mode) else "file"
        events.append((kind, stat.S_IMODE(metadata.st_mode)))
        real_fsync(descriptor)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "write", partial_write)
        scoped.setattr(os, "fsync", recording_fsync)
        payload, _, _ = _publish(store, receipt_id, manifest)

    assert store.read_stable(receipt_id).payload == payload
    assert events == [("file", 0o600), ("file", 0o400), ("directory", 0o700)]


def test_fsync_and_writer_readback_failures_never_return_durable_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for receipt_id, injection, expected in (
        ("fsync-failure", "fsync", OSError),
        ("readback-failure", "readback", Stage2BReceiptStoreError),
    ):
        root = tmp_path / receipt_id
        _mkdir_secure(root)
        store = Stage2BFilesystemReceiptStore(root.resolve())
        payload = _canonical(_commit_manifest(receipt_id))

        with monkeypatch.context() as scoped:
            if injection == "fsync":
                scoped.setattr(
                    os,
                    "fsync",
                    lambda _descriptor: (_ for _ in ()).throw(
                        OSError("injected fsync failure")
                    ),
                )
            else:
                real_read_all = receipt_store_module._read_all

                def corrupt_readback(descriptor: int) -> bytes:
                    return real_read_all(descriptor) + b"corrupt"

                scoped.setattr(
                    receipt_store_module,
                    "_read_all",
                    corrupt_readback,
                )
            with pytest.raises(expected):
                store.publish_exclusive(
                    receipt_id=receipt_id,
                    payload=payload,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                )

        assert len(tuple(root.iterdir())) == 1


@pytest.mark.parametrize("entry_kind", ("directory", "fifo"))
def test_non_regular_existing_receipt_entries_are_never_followed(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    root = tmp_path / entry_kind
    _mkdir_secure(root)
    store = Stage2BFilesystemReceiptStore(root.resolve())
    receipt_id = f"non-regular-{entry_kind}"
    path = store.receipt_path(receipt_id)
    if entry_kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    payload = _canonical(_commit_manifest(receipt_id))

    with pytest.raises(Stage2BReceiptConflictError):
        store.publish_exclusive(
            receipt_id=receipt_id,
            payload=payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
    with pytest.raises(Stage2BReceiptStoreError, match="single-link regular"):
        store.read_stable(receipt_id)
