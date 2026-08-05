#!/usr/bin/env python3
"""Fail-closed verifier/builder for the CACH Stage-2B L2A source delta.

Verification authenticates source and deterministic archive bytes. It does not
run or replay tests and grants no runtime, filesystem, recovery, training,
evaluation, deployment, capture, Stage-3, or scientific authority.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
import sys
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_HOST = "H200"
CANONICAL_WORKTREE = "/home/zch/workspace/sana-wam"
DEFAULT_MANIFEST = ROOT / "docs/cach_sana_wam/stage2b_l2/SOURCE_MANIFEST.json"
CANONICAL_BUNDLE_DIRECTORY = Path("/DATA/share/sana_cach_source_bundles")
SOURCE_DATE_EPOCH = 1_785_456_000
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

STAGE2B_MANIFEST_PIN = {
    "path": "docs/cach_sana_wam/stage2b/SOURCE_MANIFEST.json",
    "sha256": "87a5f36d13e06cdd5d951cf1dd96f4e353edc50fc555c96b3269d19859126c0a",
}
STAGE2B_VERIFIER_PIN = {
    "path": "scripts/verify_cach_stage2b.py",
    "sha256": "a9a2a58c0dbc831f886e43a0c299a52a59683e443f05dcd646dc99a41ff5a1fe",
}
STAGE2B_BUNDLE_PIN = {
    "mode": "0444",
    "path": (
        "/DATA/share/sana_cach_source_bundles/"
        "cach_stage2b_0feb797618cb9e02150abd52b357c03d2a4986177fc1e8ebeca626ee808389d2_"
        "20260731.tar"
    ),
    "sha256": "0feb797618cb9e02150abd52b357c03d2a4986177fc1e8ebeca626ee808389d2",
    "size_bytes": 389_120,
}
STAGE0_BUNDLE_PIN = {
    "mode": "0444",
    "path": "/DATA/share/sana_cach_source_bundles/cach_stage0_85d16ee5_20260731.tar",
    "sha256": "bd427ef6a604abef49c75b20d14a21d9412f5c21b539a1240fb052187f8889fd",
    "size_bytes": 645_120,
}

EXPECTED_BASE_LINEAGE = {
    "sana_afcc_handoff_head": "9586486f2a9f5172d57b325e32093a3e018d34c0",
    "sana_head": "16b9cec673e3335724ba2d8db25de7f9ed229292",
    "sana_wam_head": "605f1c134b4c983ff80f8489c4bc8847036329e2",
    "stage2_source_manifest": {
        "path": "docs/cach_sana_wam/stage2/SOURCE_MANIFEST.json",
        "sha256": "8333c74d9bc8115937f98d9fc12a60da62cab49dd8895cd403047027315052fb",
    },
    "stage2b_source_bundle": STAGE2B_BUNDLE_PIN,
    "stage2b_source_manifest": STAGE2B_MANIFEST_PIN,
    "stage2b_verifier": STAGE2B_VERIFIER_PIN,
}

SOURCE_FILES = (
    "docs/cach_sana_wam/stage2b_l2/README.md",
    "docs/cach_sana_wam/stage2b_l2/STAGE2B_L2A_LIGHTWEIGHT_TEST_REPORT.json",
    "docs/cach_sana_wam/stage2b_l2/STAGE2B_L2A_SOURCE_FILES.txt",
    "docs/cach_sana_wam/stage2b_l2/STAGE2B_L2_OFFLINE_LEDGER_IMPLEMENTATION_PLAN_20260731.md",
    "scripts/verify_cach_stage2b_l2a.py",
    "src/sana_wam/cach/stage2b_offline_ledger.py",
    "tests/test_cach_stage2b_offline_ledger.py",
)

HARNESS_PINS = {
    ".gitmodules": "22e41c488f68ca762c6433e8a53bb504cdcddd899b2e7f4aa6935ae67297d066",
    ".python-version": "7b55f8e67b5623c4bef3fa691288da9437d79d3aba156de48d481db32ac7d16d",
    "pyproject.toml": "c84bce95525fcbe03800602bd6dc3e6a151f53888b0dbc91c405ac9bf6f6bafa",
    "src/sana_wam/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/sana_wam/cach/__init__.py": "0daf40668bd7042ff0de6d2c33a045e13d32f17722df3d915aa255268091c75b",
    "src/sana_wam/model/__init__.py": "78e9e9403c02e3af004d18b2a8026647fc0e5e873a8684b4f96251b9cc530f0d",
    "src/sana_wam/model/video_backbone/__init__.py": (
        "72958a5200f4494a8740f4ff71db0d3d4489a84d2ae48e8e2555c0b35658eef9"
    ),
    "src/sana_wam/model/video_backbone/sana/__init__.py": (
        "dc4acb3ca1022a0ed62904a61d09394ed03e7e8fedb022ebf6d25b2871f0d75f"
    ),
    "tests/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "tests/conftest.py": "893ce1a02854323e0ebdd3574a74364c6f15aa489e8f8fd7187126efd81d39fc",
    "uv.lock": "31803c58048ff07600116ac389bd025f4def00cea0282a29637017b4acfec08c",
}

INHERITED_PINS = {
    "src/sana_wam/cach/stage2b_receipt_store.py": (
        "8d8f313c5d08622c74b313d32249fbb0f79284e6f1f0c6a9c36590e4f9d3b83f"
    ),
    "src/sana_wam/cach/stage2b_state_snapshot.py": (
        "269951d7c7e3145ffaad146eb580dc5e41d0572dccea11eadc23da2a3072805c"
    ),
    "src/sana_wam/cach/staging_variant.py": (
        "99da23af8c5cd62fa2a7e468b6b372a298900608be22a18c4634b82ef0dea220"
    ),
    "src/sana_wam/model/action_chunk_layout.py": (
        "8e6521a93614b4ce7e43ba51d2af4f67d2650b03cc60e7d7a152e36dfa3c8e39"
    ),
    "src/sana_wam/model/video_backbone/sana/hybrid_cache.py": (
        "5b61ab6492670e8021b3603f965e9f3e8921e9d7f0ea77b49e40d92a77f78a74"
    ),
    "src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b.py": (
        "8ab968b77660df1092b43a52eecb4dba16316060e7f83c0ad6fd214230a8a5dd"
    ),
}

EXPECTED_SCOPE = {
    "execution_scope": "synthetic_cpu_single_process_offline_stage2b_l2a_only",
    "file_count": 7,
    "kind": "stage2b_l2a_zero_collision_delta_over_verified_stage2b",
    "source_manifest_excluded_to_avoid_self_reference": True,
    "transitive_runtime_closure": False,
}

EXPECTED_GIT_RECOVERY_BUNDLES = {
    "local_afcc_handoff": None,
    "sana_submodule": None,
    "sana_wam": None,
}

MANIFEST_KEYS = {
    "base_lineage",
    "canonical_host",
    "canonical_worktree",
    "created_at",
    "execution_harness_pins",
    "files",
    "git_recovery_bundles",
    "inherited_source_pins",
    "overlay",
    "retained_stage2b_l2a_evidence",
    "schema",
    "scope",
    "stage2b_l2a_source_bundle",
    "status",
}


class Stage2BL2AVerificationError(RuntimeError):
    """Raised when an L2A source checkpoint invariant differs."""


def _fail(message: str) -> None:
    raise Stage2BL2AVerificationError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_json(payload: bytes, label: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} has duplicate JSON member {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        _fail(f"{label} has non-finite JSON value {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2BL2AVerificationError(f"{label} is invalid JSON") from exc


def _exact_equal(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _exact_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return len(observed) == len(expected) and all(
            _exact_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_rdev,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _stable_read_regular(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        _fail(f"{label} is not a single-link regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if _fingerprint(before) != _fingerprint(opened):
            _fail(f"{label} changed before open")
        payload = _read_all(descriptor)
        after = os.fstat(descriptor)
        if _fingerprint(opened) != _fingerprint(after):
            _fail(f"{label} changed during read")
    finally:
        os.close(descriptor)
    entry_after = os.stat(path, follow_symlinks=False)
    if _fingerprint(after) != _fingerprint(entry_after):
        _fail(f"{label} entry changed during read")
    return payload, entry_after


def _repo_path(raw: object, label: str) -> tuple[str, Path]:
    if type(raw) is not str or not raw:
        _fail(f"{label} is not a non-empty string")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or str(pure) != raw or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        _fail(f"{label} is not a canonical relative repository path")
    lexical = ROOT.joinpath(*pure.parts)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Stage2BL2AVerificationError(f"{label} cannot be resolved") from exc
    if resolved != lexical or ROOT not in resolved.parents:
        _fail(f"{label} crosses a symlink or repository boundary")
    return raw, lexical


def _read_repo_file(raw: object, label: str) -> bytes:
    relative, path = _repo_path(raw, label)
    payload, _ = _stable_read_regular(path, relative)
    return payload


def _pin_entries(pins: dict[str, str]) -> list[dict[str, str]]:
    return [{"path": path, "sha256": pins[path]} for path in sorted(pins)]


def _verify_pin_set(observed: object, pins: dict[str, str], label: str) -> None:
    expected = _pin_entries(pins)
    if not _exact_equal(observed, expected):
        _fail(f"{label} declaration differs")
    for entry in expected:
        payload = _read_repo_file(entry["path"], f"{label} member")
        if _sha256(payload) != entry["sha256"]:
            _fail(f"{label} member differs: {entry['path']}")


def _verify_predecessor_pins() -> None:
    for pin, label in (
        (STAGE2B_MANIFEST_PIN, "Stage-2B manifest"),
        (STAGE2B_VERIFIER_PIN, "Stage-2B verifier"),
    ):
        payload = _read_repo_file(pin["path"], label)
        if _sha256(payload) != pin["sha256"]:
            _fail(f"{label} differs from its fixed pin")
    bundle_path = Path(STAGE2B_BUNDLE_PIN["path"])
    payload, metadata = _stable_read_regular(bundle_path, "Stage-2B source bundle")
    if (
        _sha256(payload) != STAGE2B_BUNDLE_PIN["sha256"]
        or len(payload) != STAGE2B_BUNDLE_PIN["size_bytes"]
        or stat.S_IMODE(metadata.st_mode) != 0o444
    ):
        _fail("Stage-2B source bundle differs from its fixed pin")


def _run_predecessor_verifier() -> dict[str, Any]:
    command = [
        sys.executable,
        os.fspath(ROOT / STAGE2B_VERIFIER_PIN["path"]),
        "--manifest",
        os.fspath(ROOT / STAGE2B_MANIFEST_PIN["path"]),
        "--expected-manifest-sha256",
        STAGE2B_MANIFEST_PIN["sha256"],
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=False,
    )
    if completed.returncode != 0:
        _fail("fixed Stage-2B predecessor verifier failed")
    result = _strict_json(completed.stdout.strip(), "Stage-2B verifier output")
    if not isinstance(result, dict) or (
        result.get("additive_artifact_valid") is not True
        or result.get("caller_manifest_pin_verified") is not True
        or result.get("manifest_sha256") != STAGE2B_MANIFEST_PIN["sha256"]
        or result.get("overlay_verified") is not True
        or result.get("stage2b_source_bundle_verified") is not True
        or result.get("execution_replayed") is not False
        or result.get("training_authorized") is not False
        or result.get("evaluation_authorized") is not False
        or result.get("deploy_authorized") is not False
        or result.get("capture_authorized") is not False
        or result.get("scientific_eligible") is not False
        or result.get("stage3_authorized") is not False
        or result.get("transitive_runtime_closure") is not False
    ):
        _fail("fixed Stage-2B verifier output grants or omits required claims")
    return result


def _manifest_file_map(manifest: object, label: str) -> dict[str, str]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        _fail(f"{label} file inventory is absent")
    result: dict[str, str] = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or type(entry.get("path")) is not str:
            _fail(f"{label} has malformed file entry")
        path = entry["path"]
        digest = entry.get("sha256")
        if (
            type(digest) is not str
            or SHA256_RE.fullmatch(digest) is None
            or path in result
        ):
            _fail(f"{label} has invalid or duplicate file entry")
        PurePosixPath(path)
        result[path] = digest
    return result


def _tar_regular_file_map(payload: bytes, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isreg() or member.name in result:
                _fail(f"{label} contains a non-regular or duplicate member")
            extracted = archive.extractfile(member)
            if extracted is None:
                _fail(f"{label} regular member cannot be read")
            member_payload = extracted.read()
            if len(member_payload) != member.size:
                _fail(f"{label} regular member has a short payload")
            result[member.name] = _sha256(member_payload)
    return result


def _resolved_predecessor_inventory() -> tuple[dict[str, str], str]:
    stage0_raw, stage0_metadata = _stable_read_regular(
        Path(STAGE0_BUNDLE_PIN["path"]),
        "Stage-0 source bundle",
    )
    if (
        _sha256(stage0_raw) != STAGE0_BUNDLE_PIN["sha256"]
        or len(stage0_raw) != STAGE0_BUNDLE_PIN["size_bytes"]
        or stat.S_IMODE(stage0_metadata.st_mode) != 0o444
    ):
        _fail("Stage-0 source bundle differs from its transitive fixed pin")
    resolved = _tar_regular_file_map(stage0_raw, "Stage-0 source bundle")
    for path, label in (
        (
            "docs/cach_sana_wam/stage1/SOURCE_MANIFEST.draft.json",
            "Stage-1 source manifest",
        ),
        ("docs/cach_sana_wam/stage2/SOURCE_MANIFEST.json", "Stage-2 source manifest"),
        (STAGE2B_MANIFEST_PIN["path"], "Stage-2B source manifest"),
    ):
        manifest = _strict_json(_read_repo_file(path, label), label)
        resolved.update(_manifest_file_map(manifest, label))
    if len(resolved) != 105:
        _fail("resolved predecessor inventory does not contain exactly 105 paths")
    inventory = {
        "files": [
            {"path": path, "sha256": resolved[path]} for path in sorted(resolved)
        ],
        "schema": "cach.resolved_source_inventory.v1",
    }
    return resolved, _sha256(_canonical_json_bytes(inventory))


def _verify_report(payload: bytes) -> None:
    report = _strict_json(payload, "L2A lightweight report")
    expected_keys = {
        "artifact_state_at_lightweight_test_time",
        "authority",
        "canonical_host",
        "canonical_worktree",
        "limitations",
        "predecessor_pins",
        "prohibited_actions_attestation",
        "recorded_on",
        "results",
        "retained_stage2b_l2a_evidence",
        "schema",
        "scope",
        "status",
        "tested_source_pins",
    }
    if not isinstance(report, dict) or set(report) != expected_keys:
        _fail("L2A lightweight report top-level schema differs")
    if (
        report["schema"] != "cach.stage2b_l2a.live_lightweight_report.v1"
        or report["status"]
        != "bounded_l2a_lightweight_green_no_p0_p1_not_filesystem_admitted_non_scientific"
        or report["canonical_host"] != CANONICAL_HOST
        or report["canonical_worktree"] != CANONICAL_WORKTREE
        or report["scope"]
        != "synthetic_cpu_single_process_offline_stage2b_l2a_only"
        or report["retained_stage2b_l2a_evidence"] is not None
    ):
        _fail("L2A lightweight report identity differs")
    expected_source_pins = {
        "src/sana_wam/cach/stage2b_offline_ledger.py": (
            "030a67332310d695395729161de7eb26999e391d469ff24b9c2d532666e77da6"
        ),
        "tests/test_cach_stage2b_offline_ledger.py": (
            "0c342390bef912c2f00e9840aa9dbb79393e7ca12ae9137df6a6939746483407"
        ),
    }
    if not _exact_equal(report["tested_source_pins"], expected_source_pins):
        _fail("L2A lightweight report tested source pins differ")
    focused = report["results"].get("focused_l2a_cpu", {})
    regression = report["results"].get("full_cach_cpu_regression", {})
    if (
        focused.get("passed") != 82
        or focused.get("failed") != 0
        or focused.get("skipped") != 0
        or focused.get("deselected") != 0
        or focused.get("test_function_count") != 38
        or regression.get("passed") != 397
        or regression.get("failed") != 0
        or regression.get("skipped") != 0
        or regression.get("deselected") != 1
        or regression.get("warnings") != 13
        or regression.get("diagnostic_only") is not True
    ):
        _fail("L2A lightweight report results differ")
    for section in (
        report["artifact_state_at_lightweight_test_time"],
        report["authority"],
        report["prohibited_actions_attestation"],
    ):
        if not isinstance(section, dict) or any(value is not False for value in section.values()):
            _fail("L2A lightweight report grants prohibited authority")
    if report["limitations"].get("transitive_runtime_closure") is not False:
        _fail("L2A lightweight report claims transitive runtime closure")


def _source_bytes() -> tuple[dict[str, bytes], list[dict[str, object]]]:
    if tuple(sorted(SOURCE_FILES)) != SOURCE_FILES or len(set(SOURCE_FILES)) != 7:
        _fail("internal L2A source inventory is not sorted and unique")
    payloads: dict[str, bytes] = {}
    entries: list[dict[str, object]] = []
    for path in SOURCE_FILES:
        payload = _read_repo_file(path, "L2A source member")
        payloads[path] = payload
        entries.append(
            {"path": path, "sha256": _sha256(payload), "size_bytes": len(payload)}
        )
    expected_list = ("\n".join(SOURCE_FILES) + "\n").encode("utf-8")
    source_list_path = (
        "docs/cach_sana_wam/stage2b_l2/STAGE2B_L2A_SOURCE_FILES.txt"
    )
    if payloads[source_list_path] != expected_list:
        _fail("L2A source-list bytes differ from the exact sorted inventory")
    _verify_report(
        payloads[
            "docs/cach_sana_wam/stage2b_l2/"
            "STAGE2B_L2A_LIGHTWEIGHT_TEST_REPORT.json"
        ]
    )
    return payloads, entries


def _canonical_tar(payloads: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for path in SOURCE_FILES:
            payload = payloads[path]
            member = tarfile.TarInfo(path)
            member.size = len(payload)
            member.mode = 0o644
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = SOURCE_DATE_EPOCH
            member.type = tarfile.REGTYPE
            member.linkname = ""
            archive.addfile(member, io.BytesIO(payload))
    return buffer.getvalue()


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            _fail("bundle publication made no write progress")
        remaining = remaining[written:]


def _rename_noreplace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _fail("renameat2 is required for bundle no-replace publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _publish_bundle(payload: bytes, directory: Path) -> dict[str, object]:
    if Path(os.path.abspath(os.fspath(directory))) != CANONICAL_BUNDLE_DIRECTORY:
        _fail("bundle output directory must be the canonical /DATA source directory")
    digest = _sha256(payload)
    filename = f"cach_stage2b_l2a_{digest}_20260731.tar"
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(directory, flags | getattr(os, "O_CLOEXEC", 0))
    temporary = f".tmp-cach-stage2b-l2a-{secrets.token_hex(16)}"
    temporary_owned = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_owned = True
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            _rename_noreplace(directory_fd, temporary, directory_fd, filename)
            temporary_owned = False
        except FileExistsError:
            os.unlink(temporary, dir_fd=directory_fd)
            temporary_owned = False
        os.fsync(directory_fd)
    finally:
        if temporary_owned:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except OSError:
                pass
        os.close(directory_fd)
    target = directory / filename
    observed, metadata = _stable_read_regular(target, "L2A source bundle")
    if (
        observed != payload
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
    ):
        _fail("published L2A bundle differs or is unsafe")
    return {
        "mode": "0444",
        "path": os.fspath(target),
        "sha256": digest,
        "size_bytes": len(payload),
    }


def build_bundle(directory: Path) -> dict[str, object]:
    if str(ROOT) != CANONICAL_WORKTREE:
        _fail("bundle construction must run in the canonical H200 worktree")
    _verify_predecessor_pins()
    _run_predecessor_verifier()
    _verify_pin_set(_pin_entries(HARNESS_PINS), HARNESS_PINS, "execution harness")
    _verify_pin_set(_pin_entries(INHERITED_PINS), INHERITED_PINS, "inherited source")
    predecessor, inventory_digest = _resolved_predecessor_inventory()
    if predecessor.keys() & set(SOURCE_FILES):
        _fail("L2A source inventory collides with the resolved predecessor")
    payloads, file_entries = _source_bytes()
    bundle = _publish_bundle(_canonical_tar(payloads), directory)
    return {
        "bundle": bundle,
        "files": file_entries,
        "predecessor_resolved_file_count": len(predecessor),
        "predecessor_resolved_inventory_sha256": inventory_digest,
        "resolved_file_count": len(predecessor) + len(SOURCE_FILES),
        "schema": "cach.stage2b_l2a.bundle_build.v1",
        "zero_collision_verified": True,
    }


def _verify_manifest_files(
    observed: object,
    payloads: dict[str, bytes],
    entries: list[dict[str, object]],
) -> None:
    if not _exact_equal(observed, entries):
        _fail("L2A source manifest file inventory differs")
    for entry in entries:
        path = entry["path"]
        if _sha256(payloads[path]) != entry["sha256"]:
            _fail(f"L2A source member digest differs: {path}")


def _verify_bundle(pin: object, expected: bytes) -> None:
    if not isinstance(pin, dict) or set(pin) != {
        "mode",
        "path",
        "sha256",
        "size_bytes",
    }:
        _fail("L2A source bundle pin schema differs")
    digest = _sha256(expected)
    expected_path = (
        CANONICAL_BUNDLE_DIRECTORY
        / f"cach_stage2b_l2a_{digest}_20260731.tar"
    )
    if pin != {
        "mode": "0444",
        "path": os.fspath(expected_path),
        "sha256": digest,
        "size_bytes": len(expected),
    }:
        _fail("L2A source bundle pin differs")
    observed, metadata = _stable_read_regular(expected_path, "L2A source bundle")
    if (
        observed != expected
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
    ):
        _fail("L2A source bundle bytes or metadata differ")


def verify(manifest_path: Path, expected_manifest_sha256: str) -> dict[str, object]:
    if str(ROOT) != CANONICAL_WORKTREE:
        _fail("verification must run in the canonical H200 worktree")
    if SHA256_RE.fullmatch(expected_manifest_sha256) is None:
        _fail("expected manifest SHA256 must be 64 lowercase hexadecimal digits")
    if Path(os.path.abspath(os.fspath(manifest_path))) != DEFAULT_MANIFEST:
        _fail("L2A manifest must use the canonical default repository path")
    manifest_raw, _ = _stable_read_regular(manifest_path, "L2A source manifest")
    manifest_sha256 = _sha256(manifest_raw)
    if manifest_sha256 != expected_manifest_sha256:
        _fail("L2A source manifest differs from the caller trust anchor")
    manifest = _strict_json(manifest_raw, "L2A source manifest")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        _fail("L2A source manifest top-level schema differs")
    if (
        manifest["schema"] != "cach.stage2b_l2a.source_manifest.v1"
        or manifest["status"]
        != "l2a_source_checkpoint_verified_scope_not_admitted_non_scientific"
        or manifest["canonical_host"] != CANONICAL_HOST
        or manifest["canonical_worktree"] != CANONICAL_WORKTREE
        or manifest["created_at"] != "2026-07-31T00:00:00Z"
    ):
        _fail("L2A source manifest identity differs")
    if not _exact_equal(manifest["base_lineage"], EXPECTED_BASE_LINEAGE):
        _fail("L2A predecessor lineage differs")
    if not _exact_equal(manifest["scope"], EXPECTED_SCOPE):
        _fail("L2A source scope differs")
    if not _exact_equal(
        manifest["git_recovery_bundles"], EXPECTED_GIT_RECOVERY_BUNDLES
    ):
        _fail("L2A git recovery-bundle declaration differs")
    if manifest["retained_stage2b_l2a_evidence"] is not None:
        _fail("L2A source checkpoint must not claim retained execution evidence")
    _verify_predecessor_pins()
    _run_predecessor_verifier()
    _verify_pin_set(
        manifest["execution_harness_pins"], HARNESS_PINS, "execution harness"
    )
    _verify_pin_set(
        manifest["inherited_source_pins"], INHERITED_PINS, "inherited source"
    )
    predecessor, inventory_digest = _resolved_predecessor_inventory()
    collisions = sorted(predecessor.keys() & set(SOURCE_FILES))
    expected_overlay = {
        "order": ["stage0", "stage1", "stage2", "stage2b", "stage2b_l2a"],
        "predecessor_resolved_file_count": 105,
        "predecessor_resolved_inventory_sha256": inventory_digest,
        "resolved_file_count": 112,
        "stage2b_l2a_predecessor_collisions": [],
        "stage2b_l2a_requires_zero_predecessor_collisions": True,
    }
    if collisions or not _exact_equal(manifest["overlay"], expected_overlay):
        _fail("L2A zero-collision overlay declaration differs")
    payloads, entries = _source_bytes()
    _verify_manifest_files(manifest["files"], payloads, entries)
    _verify_bundle(manifest["stage2b_l2a_source_bundle"], _canonical_tar(payloads))
    return {
        "additive_artifact_valid": True,
        "caller_manifest_pin_verified": True,
        "capture_authorized": False,
        "deploy_authorized": False,
        "evaluation_authorized": False,
        "execution_replayed": False,
        "l2a_source_checkpoint_verified": True,
        "l2b_authorized": False,
        "l3_authorized": False,
        "manifest_sha256": manifest_sha256,
        "retained_immutable_execution_evidence": False,
        "runtime_closure_verified": False,
        "schema": "cach.stage2b_l2a.additive_artifact_verification.v1",
        "scientific_eligible": False,
        "source_file_count": 7,
        "stage3_authorized": False,
        "training_authorized": False,
        "transitive_runtime_closure": False,
        "zero_collision_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify or build the byte-scoped Stage-2B L2A source checkpoint."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--emit-canonical-bundle", action="store_true")
    parser.add_argument(
        "--bundle-directory",
        type=Path,
        default=CANONICAL_BUNDLE_DIRECTORY,
    )
    arguments = parser.parse_args()
    try:
        if arguments.emit_canonical_bundle:
            if arguments.expected_manifest_sha256 is not None:
                _fail("bundle build mode does not accept a manifest trust anchor")
            result = build_bundle(arguments.bundle_directory)
        else:
            if arguments.expected_manifest_sha256 is None:
                _fail("verification requires --expected-manifest-sha256")
            result = verify(arguments.manifest, arguments.expected_manifest_sha256)
    except (
        OSError,
        Stage2BL2AVerificationError,
        subprocess.SubprocessError,
        tarfile.TarError,
    ) as exc:
        failure = {
            "additive_artifact_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "schema": "cach.stage2b_l2a.additive_artifact_verification_failure.v1",
            "scientific_eligible": False,
            "training_authorized": False,
            "transitive_runtime_closure": False,
        }
        print(_canonical_json_bytes(failure).decode("utf-8"), file=sys.stderr)
        return 1
    print(_canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
