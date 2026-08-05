#!/usr/bin/env python3
"""Build or verify the byte-scoped CACH Stage-2B L2B source checkpoint.

The verifier authenticates one additive, zero-collision source archive over the
caller-pinned L2A checkpoint.  It never imports the L2B implementation, replays
tests, executes a model, installs state, or grants runtime, recovery, training,
evaluation, deployment, capture, filesystem-admission, or scientific authority.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
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
DEFAULT_MANIFEST = (
    ROOT / "docs/cach_sana_wam/stage2b_l2b/SOURCE_MANIFEST.json"
)
CANONICAL_BUNDLE_DIRECTORY = Path("/DATA/share/sana_cach_source_bundles")
SOURCE_DATE_EPOCH = 1_785_456_000
CHECKPOINT_DATE = "20260731"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

L2A_MANIFEST_PIN = {
    "path": "docs/cach_sana_wam/stage2b_l2/SOURCE_MANIFEST.json",
    "sha256": "5d27a778bff120ff7a27c18ebaffbfb1761b2ced3d5f2e4c46a4de8c5763b8bf",
}
L2A_VERIFIER_PIN = {
    "path": "scripts/verify_cach_stage2b_l2a.py",
    "sha256": "cd433d338ffb9f493f291905f0d5313c64591eafbf729bd7ae16c96fac1c9255",
}
L2A_BUNDLE_PIN = {
    "mode": "0444",
    "path": (
        "/DATA/share/sana_cach_source_bundles/"
        "cach_stage2b_l2a_163737a9b8c9b88bfd8d7e05064052fc2948661f270c4a9a9f4ea55e96494b7d_"
        "20260731.tar"
    ),
    "sha256": "163737a9b8c9b88bfd8d7e05064052fc2948661f270c4a9a9f4ea55e96494b7d",
    "size_bytes": 276_480,
}
STAGE0_BUNDLE_PIN = {
    "mode": "0444",
    "path": "/DATA/share/sana_cach_source_bundles/cach_stage0_85d16ee5_20260731.tar",
    "sha256": "bd427ef6a604abef49c75b20d14a21d9412f5c21b539a1240fb052187f8889fd",
    "size_bytes": 645_120,
}
STAGE2_MANIFEST_PIN = {
    "path": "docs/cach_sana_wam/stage2/SOURCE_MANIFEST.json",
    "sha256": "8333c74d9bc8115937f98d9fc12a60da62cab49dd8895cd403047027315052fb",
}
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

EXPECTED_BASE_LINEAGE = {
    "sana_afcc_handoff_head": "9586486f2a9f5172d57b325e32093a3e018d34c0",
    "sana_head": "16b9cec673e3335724ba2d8db25de7f9ed229292",
    "sana_wam_head": "605f1c134b4c983ff80f8489c4bc8847036329e2",
    "stage2_source_manifest": STAGE2_MANIFEST_PIN,
    "stage2b_source_bundle": STAGE2B_BUNDLE_PIN,
    "stage2b_source_manifest": STAGE2B_MANIFEST_PIN,
    "stage2b_verifier": STAGE2B_VERIFIER_PIN,
    "stage2b_l2a_source_bundle": L2A_BUNDLE_PIN,
    "stage2b_l2a_source_manifest": L2A_MANIFEST_PIN,
    "stage2b_l2a_verifier": L2A_VERIFIER_PIN,
}

# This order is part of the archive contract.  In particular it is not inferred
# from directory traversal, git state, or locale-dependent sorting.
SOURCE_FILES = (
    "docs/cach_sana_wam/stage2b_l2b/README.md",
    "docs/cach_sana_wam/stage2b_l2b/STAGE2B_L2B_LIGHTWEIGHT_TEST_REPORT.json",
    "docs/cach_sana_wam/stage2b_l2b/STAGE2B_L2B_SOURCE_FILES.txt",
    (
        "docs/cach_sana_wam/stage2b_l2/"
        "STAGE2B_L2B_LEDGERED_MANAGER_IMPLEMENTATION_PLAN_20260731.md"
    ),
    "scripts/verify_cach_stage2b_l2b.py",
    (
        "src/sana_wam/model/video_backbone/sana/"
        "hybrid_cache_stage2b_ledgered.py"
    ),
    "tests/test_cach_stage2b_ledgered_manager.py",
)

FROZEN_L2B_CORE_PINS = {
    (
        "docs/cach_sana_wam/stage2b_l2/"
        "STAGE2B_L2B_LEDGERED_MANAGER_IMPLEMENTATION_PLAN_20260731.md"
    ): "9aa65a9aea456ff87a001024ee0c9d4ea0ed267d474091f809a17e8121692d5c",
    (
        "src/sana_wam/model/video_backbone/sana/"
        "hybrid_cache_stage2b_ledgered.py"
    ): "716130f7fbe352cc2e8c179a33edb4cf904b869823ce6261367c499d7e35816d",
    "tests/test_cach_stage2b_ledgered_manager.py": (
        "fe5094ef930b63afa79b57fba95970a059fe9b9710c4ab14746ac6cbad5a4f7f"
    ),
}

HARNESS_PINS = {
    ".gitmodules": "22e41c488f68ca762c6433e8a53bb504cdcddd899b2e7f4aa6935ae67297d066",
    ".python-version": "7b55f8e67b5623c4bef3fa691288da9437d79d3aba156de48d481db32ac7d16d",
    "pyproject.toml": "c84bce95525fcbe03800602bd6dc3e6a151f53888b0dbc91c405ac9bf6f6bafa",
    "src/sana_wam/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "src/sana_wam/cach/__init__.py": (
        "0daf40668bd7042ff0de6d2c33a045e13d32f17722df3d915aa255268091c75b"
    ),
    "src/sana_wam/model/__init__.py": (
        "78e9e9403c02e3af004d18b2a8026647fc0e5e873a8684b4f96251b9cc530f0d"
    ),
    "src/sana_wam/model/video_backbone/__init__.py": (
        "72958a5200f4494a8740f4ff71db0d3d4489a84d2ae48e8e2555c0b35658eef9"
    ),
    "src/sana_wam/model/video_backbone/sana/__init__.py": (
        "dc4acb3ca1022a0ed62904a61d09394ed03e7e8fedb022ebf6d25b2871f0d75f"
    ),
    "tests/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "tests/conftest.py": (
        "893ce1a02854323e0ebdd3574a74364c6f15aa489e8f8fd7187126efd81d39fc"
    ),
    "uv.lock": "31803c58048ff07600116ac389bd025f4def00cea0282a29637017b4acfec08c",
}

INHERITED_SOURCE_PINS = {
    "src/sana_wam/cach/committed_action_history.py": (
        "162d7e7ee866b18cdba0f0e319b2c9b277d73d933ff5a4dd38c92ebcc762b5e9"
    ),
    "src/sana_wam/cach/stage2b_offline_ledger.py": (
        "030a67332310d695395729161de7eb26999e391d469ff24b9c2d532666e77da6"
    ),
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
    "src/sana_wam/model/cach_paired_stager.py": (
        "c1455af51df4e0a0eff9971fa98e9272aac580084029a7990161817d246a6ef4"
    ),
    "src/sana_wam/model/video_backbone/sana/hybrid_cache.py": (
        "5b61ab6492670e8021b3603f965e9f3e8921e9d7f0ea77b49e40d92a77f78a74"
    ),
    "src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b.py": (
        "8ab968b77660df1092b43a52eecb4dba16316060e7f83c0ad6fd214230a8a5dd"
    ),
}

EXPECTED_SCOPE = {
    "execution_scope": "fresh_only_synthetic_cpu_single_process_stage2b_l2b_manager_facade",
    "file_count": 7,
    "kind": "stage2b_l2b_zero_collision_delta_over_verified_stage2b_l2a",
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
    "frozen_l2b_core_pins",
    "git_recovery_bundles",
    "inherited_source_pins",
    "overlay",
    "retained_stage2b_l2b_evidence",
    "schema",
    "scope",
    "stage2b_l2b_source_bundle",
    "status",
}

REPORT_KEYS = {
    "artifact_state_at_lightweight_test_time",
    "authority",
    "canonical_host",
    "canonical_worktree",
    "checkpoint_candidate_pins",
    "independent_final_read_only_audits",
    "limitations",
    "predecessor_pins",
    "prohibited_actions_attestation",
    "recorded_on",
    "results",
    "retained_stage2b_l2b_evidence",
    "schema",
    "scope",
    "status",
    "tested_source_pins",
}

EXPECTED_REPORT_ARTIFACT_STATE = {
    "source_manifest_created": False,
    "stage2b_l2b_source_bundle_created": False,
}
EXPECTED_REPORT_AUTHORITY = {
    "capture_authorized": False,
    "deploy_authorized": False,
    "evaluation_authorized": False,
    "formal_filesystem_admission": False,
    "formal_ledger_root_authorized": False,
    "full_gate_s2_claimed": False,
    "l2b_production_runtime_authorized": False,
    "l3_authorized": False,
    "recovery_authorized": False,
    "scientific_eligible": False,
    "training_authorized": False,
}
EXPECTED_REPORT_PROHIBITED_ACTIONS = {
    "checkpoint_loaded": False,
    "dataset_accessed": False,
    "evaluation_started": False,
    "formal_ledger_root_created": False,
    "gpu_used": False,
    "full_model_executed": False,
    "optimizer_created": False,
    "training_started": False,
}
EXPECTED_REPORT_LIMITATIONS = {
    "callable_scope": "cpu_synthetic_only",
    "fresh_only": True,
    "l3_restart_state_install": "not_implemented",
    "multi_process_writer_fencing": "not_admitted",
    "parent_mount_and_power_loss_semantics": "not_admitted",
    "production_exact_stager_execution": "not_callable_in_this_slice",
    "retained_immutable_execution_evidence": "not_created",
    "runtime_source_and_dynamic_import_closure": "not_established",
    "single_process_only": True,
    "transitive_runtime_closure": False,
}
EXPECTED_REPORT_PREDECESSOR_PINS = {
    "exact_production_paired_stager_sha256": (
        "c1455af51df4e0a0eff9971fa98e9272aac580084029a7990161817d246a6ef4"
    ),
    "frozen_stage2b_manager_sha256": (
        "8ab968b77660df1092b43a52eecb4dba16316060e7f83c0ad6fd214230a8a5dd"
    ),
    "l2a_offline_ledger_sha256": (
        "030a67332310d695395729161de7eb26999e391d469ff24b9c2d532666e77da6"
    ),
    "l2a_source_bundle_sha256": L2A_BUNDLE_PIN["sha256"],
    "l2a_source_manifest_sha256": L2A_MANIFEST_PIN["sha256"],
    "l2a_verifier_sha256": L2A_VERIFIER_PIN["sha256"],
    "stage2b_source_bundle_sha256": (
        "0feb797618cb9e02150abd52b357c03d2a4986177fc1e8ebeca626ee808389d2"
    ),
    "stage2b_source_manifest_sha256": (
        "87a5f36d13e06cdd5d951cf1dd96f4e353edc50fc555c96b3269d19859126c0a"
    ),
    "stage2b_verifier_sha256": (
        "a9a2a58c0dbc831f886e43a0c299a52a59683e443f05dcd646dc99a41ff5a1fe"
    ),
    "verification_replayed_by_this_report": False,
}
EXPECTED_REPORT_AUDITS = {
    "aggregate_open_p0": 0,
    "aggregate_open_p1": 0,
    "review_count": 3,
    "status": "source_checkpoint_not_blocked",
}
EXPECTED_TESTED_SOURCE_PINS = {
    path: digest
    for path, digest in FROZEN_L2B_CORE_PINS.items()
    if path
    != (
        "docs/cach_sana_wam/stage2b_l2/"
        "STAGE2B_L2B_LEDGERED_MANAGER_IMPLEMENTATION_PLAN_20260731.md"
    )
}

EXPECTED_L2A_VERIFIER_OUTPUT = {
    "additive_artifact_valid": True,
    "caller_manifest_pin_verified": True,
    "capture_authorized": False,
    "deploy_authorized": False,
    "evaluation_authorized": False,
    "execution_replayed": False,
    "l2a_source_checkpoint_verified": True,
    "l2b_authorized": False,
    "l3_authorized": False,
    "manifest_sha256": L2A_MANIFEST_PIN["sha256"],
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


class Stage2BL2BVerificationError(RuntimeError):
    """Raised when any L2B source-checkpoint invariant differs."""


def _fail(message: str) -> None:
    raise Stage2BL2BVerificationError(message)


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
        raise Stage2BL2BVerificationError(f"{label} is invalid JSON") from exc


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


def _canonical_relative_path(raw: object, label: str) -> str:
    if type(raw) is not str or not raw or "\\" in raw:
        _fail(f"{label} is not a non-empty canonical POSIX path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or str(pure) != raw or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        _fail(f"{label} is not a canonical relative repository path")
    return raw


def _repo_path(raw: object, label: str) -> tuple[str, Path]:
    relative = _canonical_relative_path(raw, label)
    pure = PurePosixPath(relative)
    lexical = ROOT.joinpath(*pure.parts)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Stage2BL2BVerificationError(f"{label} cannot be resolved") from exc
    if resolved != lexical or ROOT not in resolved.parents:
        _fail(f"{label} crosses a symlink or repository boundary")
    return relative, lexical


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


def _verify_core_pins(observed: object) -> None:
    expected = _pin_entries(FROZEN_L2B_CORE_PINS)
    if not _exact_equal(observed, expected):
        _fail("frozen L2B core-pin declaration differs")
    for entry in expected:
        payload = _read_repo_file(entry["path"], "frozen L2B core member")
        if _sha256(payload) != entry["sha256"]:
            _fail(f"frozen L2B core member differs: {entry['path']}")


def _verify_fixed_file_pin(pin: dict[str, str], label: str) -> bytes:
    payload = _read_repo_file(pin["path"], label)
    if _sha256(payload) != pin["sha256"]:
        _fail(f"{label} differs from its fixed pin")
    return payload


def _verify_fixed_bundle(pin: dict[str, object], label: str) -> bytes:
    payload, metadata = _stable_read_regular(Path(str(pin["path"])), label)
    if (
        _sha256(payload) != pin["sha256"]
        or len(payload) != pin["size_bytes"]
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
    ):
        _fail(f"{label} differs from its fixed bytes or metadata")
    return payload


def _run_l2a_verifier() -> dict[str, Any]:
    _verify_fixed_file_pin(L2A_VERIFIER_PIN, "L2A verifier")
    command = [
        sys.executable,
        os.fspath(ROOT / L2A_VERIFIER_PIN["path"]),
        "--manifest",
        os.fspath(ROOT / L2A_MANIFEST_PIN["path"]),
        "--expected-manifest-sha256",
        L2A_MANIFEST_PIN["sha256"],
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=False,
    )
    if completed.returncode != 0 or completed.stderr.strip():
        _fail("fixed L2A predecessor verifier failed or emitted stderr")
    result = _strict_json(completed.stdout.strip(), "L2A verifier output")
    if not _exact_equal(result, EXPECTED_L2A_VERIFIER_OUTPUT):
        _fail("fixed L2A verifier output differs or grants authority")
    return result


def _load_l2a_manifest() -> dict[str, Any]:
    payload = _verify_fixed_file_pin(L2A_MANIFEST_PIN, "L2A outer manifest")
    value = _strict_json(payload, "L2A outer manifest")
    if (
        not isinstance(value, dict)
        or value.get("schema") != "cach.stage2b_l2a.source_manifest.v1"
        or not _exact_equal(value.get("stage2b_l2a_source_bundle"), L2A_BUNDLE_PIN)
    ):
        _fail("caller-pinned L2A outer manifest identity differs")
    return value


def _verify_l2a_predecessor() -> dict[str, Any]:
    manifest = _load_l2a_manifest()
    _verify_fixed_bundle(L2A_BUNDLE_PIN, "L2A source bundle")
    _run_l2a_verifier()
    return manifest


def _manifest_file_map(manifest: object, label: str) -> dict[str, str]:
    if not isinstance(manifest, dict) or type(manifest.get("files")) is not list:
        _fail(f"{label} file inventory is absent")
    result: dict[str, str] = {}
    for index, entry in enumerate(manifest["files"]):
        if not isinstance(entry, dict):
            _fail(f"{label} file entry {index} is not an object")
        path = _canonical_relative_path(
            entry.get("path"), f"{label} file entry {index} path"
        )
        digest = entry.get("sha256")
        if (
            type(digest) is not str
            or SHA256_RE.fullmatch(digest) is None
            or path in result
        ):
            _fail(f"{label} has invalid or duplicate file entry")
        result[path] = digest
    return result


def _tar_regular_file_map(payload: bytes, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            path = _canonical_relative_path(member.name, f"{label} member path")
            if not member.isreg() or path in result:
                _fail(f"{label} contains a non-regular or duplicate member")
            extracted = archive.extractfile(member)
            if extracted is None:
                _fail(f"{label} regular member cannot be read")
            member_payload = extracted.read()
            if len(member_payload) != member.size:
                _fail(f"{label} regular member has a short payload")
            result[path] = _sha256(member_payload)
    return result


def _resolved_predecessor_inventory(
    l2a_manifest: dict[str, Any],
) -> tuple[dict[str, str], str]:
    stage0_raw = _verify_fixed_bundle(STAGE0_BUNDLE_PIN, "Stage-0 source bundle")
    resolved = _tar_regular_file_map(stage0_raw, "Stage-0 source bundle")
    for path, label in (
        (
            "docs/cach_sana_wam/stage1/SOURCE_MANIFEST.draft.json",
            "Stage-1 source manifest",
        ),
        ("docs/cach_sana_wam/stage2/SOURCE_MANIFEST.json", "Stage-2 source manifest"),
        (
            "docs/cach_sana_wam/stage2b/SOURCE_MANIFEST.json",
            "Stage-2B source manifest",
        ),
    ):
        manifest = _strict_json(_read_repo_file(path, label), label)
        resolved.update(_manifest_file_map(manifest, label))
    if len(resolved) != 105:
        _fail("resolved pre-L2A inventory does not contain exactly 105 paths")

    l2a_files = _manifest_file_map(l2a_manifest, "L2A outer manifest")
    if len(l2a_files) != 7 or resolved.keys() & l2a_files.keys():
        _fail("L2A delta is not an exact seven-file zero-collision overlay")
    resolved.update(l2a_files)
    if len(resolved) != 112:
        _fail("resolved L2A predecessor inventory does not contain 112 paths")
    inventory = {
        "files": [
            {"path": path, "sha256": resolved[path]} for path in sorted(resolved)
        ],
        "schema": "cach.resolved_source_inventory.v1",
    }
    return resolved, _sha256(_canonical_json_bytes(inventory))


def _require_duration(value: object, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        _fail(f"{label} must be one finite non-negative number")


def _verify_report(payload: bytes) -> None:
    report = _strict_json(payload, "L2B lightweight report")
    if not isinstance(report, dict) or set(report) != REPORT_KEYS:
        _fail("L2B lightweight report top-level schema differs")
    if (
        report["schema"] != "cach.stage2b_l2b.live_lightweight_report.v1"
        or report["status"]
        != (
            "bounded_l2b_lightweight_green_three_audits_no_p0_p1_"
            "not_filesystem_admitted_non_scientific"
        )
        or report["canonical_host"] != CANONICAL_HOST
        or report["canonical_worktree"] != CANONICAL_WORKTREE
        or report["recorded_on"] != "2026-07-31"
        or report["scope"] != EXPECTED_SCOPE["execution_scope"]
        or report["retained_stage2b_l2b_evidence"] is not None
    ):
        _fail("L2B lightweight report identity differs")
    for observed, expected, label in (
        (
            report["artifact_state_at_lightweight_test_time"],
            EXPECTED_REPORT_ARTIFACT_STATE,
            "artifact-state",
        ),
        (report["authority"], EXPECTED_REPORT_AUTHORITY, "authority"),
        (
            report["prohibited_actions_attestation"],
            EXPECTED_REPORT_PROHIBITED_ACTIONS,
            "prohibited-actions",
        ),
        (report["limitations"], EXPECTED_REPORT_LIMITATIONS, "limitations"),
        (
            report["predecessor_pins"],
            EXPECTED_REPORT_PREDECESSOR_PINS,
            "predecessor-pins",
        ),
        (
            report["checkpoint_candidate_pins"],
            FROZEN_L2B_CORE_PINS,
            "checkpoint-candidate-pins",
        ),
        (
            report["independent_final_read_only_audits"],
            EXPECTED_REPORT_AUDITS,
            "independent-audits",
        ),
        (
            report["tested_source_pins"],
            EXPECTED_TESTED_SOURCE_PINS,
            "tested-source-pins",
        ),
    ):
        if not _exact_equal(observed, expected):
            _fail(f"L2B lightweight report {label} differs")

    results = report["results"]
    if not isinstance(results, dict) or set(results) != {
        "focused_l2b_cpu_synthetic",
        "full_cach_cpu_regression",
        "py_compile",
        "ruff",
    }:
        _fail("L2B lightweight report result inventory differs")
    focused = results["focused_l2b_cpu_synthetic"]
    focused_keys = {
        "command",
        "deselected",
        "duration_seconds",
        "evidence_kind",
        "failed",
        "passed",
        "skipped",
        "test_function_count",
        "warnings",
    }
    if not isinstance(focused, dict) or set(focused) != focused_keys:
        _fail("focused L2B result schema differs")
    _require_duration(focused["duration_seconds"], "focused duration")
    focused_without_duration = dict(focused)
    focused_without_duration.pop("duration_seconds")
    if not _exact_equal(
        focused_without_duration,
        {
            "command": (
                ".venv/bin/python -m pytest -q "
                "tests/test_cach_stage2b_ledgered_manager.py"
            ),
            "deselected": 0,
            "evidence_kind": "live_lightweight_rerun_not_retained_execution_evidence",
            "failed": 0,
            "passed": 24,
            "skipped": 0,
            "test_function_count": 24,
            "warnings": 0,
        },
    ):
        _fail("focused L2B result differs")

    regression = results["full_cach_cpu_regression"]
    regression_keys = {
        "command",
        "deselected",
        "diagnostic_only",
        "duration_seconds",
        "evidence_kind",
        "failed",
        "known_deselected_node_id",
        "passed",
        "skipped",
        "warnings",
    }
    if not isinstance(regression, dict) or set(regression) != regression_keys:
        _fail("full CACH regression result schema differs")
    _require_duration(regression["duration_seconds"], "full CACH duration")
    regression_without_duration = dict(regression)
    regression_without_duration.pop("duration_seconds")
    if not _exact_equal(
        regression_without_duration,
        {
            "command": (
                ".venv/bin/python -m pytest -q tests/test_cach_*.py --deselect "
                "tests/test_cach_config_schema.py::"
                "test_stage1_artifact_set_is_internally_pinned_but_not_registered"
            ),
            "deselected": 1,
            "diagnostic_only": True,
            "evidence_kind": "live_lightweight_rerun_not_retained_execution_evidence",
            "failed": 0,
            "known_deselected_node_id": (
                "tests/test_cach_config_schema.py::"
                "test_stage1_artifact_set_is_internally_pinned_but_not_registered"
            ),
            "passed": 421,
            "skipped": 0,
            "warnings": 13,
        },
    ):
        _fail("full CACH regression result differs")

    lint_selection = [
        (
            "src/sana_wam/model/video_backbone/sana/"
            "hybrid_cache_stage2b_ledgered.py"
        ),
        "tests/test_cach_stage2b_ledgered_manager.py",
    ]
    expected_compile = {
        "command": (
            ".venv/bin/python -m py_compile "
            "src/sana_wam/model/video_backbone/sana/"
            "hybrid_cache_stage2b_ledgered.py "
            "tests/test_cach_stage2b_ledgered_manager.py"
        ),
        "selection": lint_selection,
        "status": "passed",
    }
    if not _exact_equal(results["py_compile"], expected_compile):
        _fail("L2B py_compile result differs")
    expected_ruff = {
        "command": (
            ".venv/bin/ruff check "
            "src/sana_wam/model/video_backbone/sana/"
            "hybrid_cache_stage2b_ledgered.py "
            "tests/test_cach_stage2b_ledgered_manager.py"
        ),
        "output": "All checks passed!",
        "selection": lint_selection,
        "status": "passed",
    }
    if not _exact_equal(results["ruff"], expected_ruff):
        _fail("L2B Ruff result differs")


def _source_bytes() -> tuple[dict[str, bytes], list[dict[str, object]]]:
    if len(SOURCE_FILES) != 7 or len(set(SOURCE_FILES)) != 7:
        _fail("internal L2B source inventory is not seven unique paths")
    payloads: dict[str, bytes] = {}
    entries: list[dict[str, object]] = []
    for path in SOURCE_FILES:
        payload = _read_repo_file(path, "L2B source member")
        payloads[path] = payload
        entries.append(
            {"path": path, "sha256": _sha256(payload), "size_bytes": len(payload)}
        )
    source_list_path = (
        "docs/cach_sana_wam/stage2b_l2b/STAGE2B_L2B_SOURCE_FILES.txt"
    )
    expected_list = ("\n".join(SOURCE_FILES) + "\n").encode("utf-8")
    if payloads[source_list_path] != expected_list:
        _fail("L2B source-list bytes differ from the exact ordered inventory")
    _verify_report(
        payloads[
            "docs/cach_sana_wam/stage2b_l2b/"
            "STAGE2B_L2B_LIGHTWEIGHT_TEST_REPORT.json"
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


def _verify_tar_structure(payload: bytes, payloads: dict[str, bytes]) -> None:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != list(SOURCE_FILES):
            _fail("L2B tar member order or inventory differs")
        for member in members:
            if (
                not member.isreg()
                or member.type != tarfile.REGTYPE
                or member.mode != 0o644
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != SOURCE_DATE_EPOCH
                or member.linkname != ""
                or member.pax_headers
            ):
                _fail(f"L2B tar metadata differs: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None or extracted.read() != payloads[member.name]:
                _fail(f"L2B tar member bytes differ: {member.name}")
            if member.size != len(payloads[member.name]):
                _fail(f"L2B tar member size differs: {member.name}")


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
    filename = f"cach_stage2b_l2b_{digest}_{CHECKPOINT_DATE}.tar"
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(directory, flags | getattr(os, "O_CLOEXEC", 0))
    temporary = f".tmp-cach-stage2b-l2b-{secrets.token_hex(16)}"
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
    observed, metadata = _stable_read_regular(target, "L2B source bundle")
    if (
        observed != payload
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
    ):
        _fail("published L2B source bundle differs or is unsafe")
    return {
        "mode": "0444",
        "path": os.fspath(target),
        "sha256": digest,
        "size_bytes": len(payload),
    }


def build_bundle(directory: Path) -> dict[str, object]:
    if str(ROOT) != CANONICAL_WORKTREE:
        _fail("bundle construction must run in the canonical H200 worktree")
    l2a_manifest = _verify_l2a_predecessor()
    _verify_pin_set(_pin_entries(HARNESS_PINS), HARNESS_PINS, "execution harness")
    _verify_pin_set(
        _pin_entries(INHERITED_SOURCE_PINS),
        INHERITED_SOURCE_PINS,
        "inherited source",
    )
    _verify_core_pins(_pin_entries(FROZEN_L2B_CORE_PINS))
    predecessor, inventory_digest = _resolved_predecessor_inventory(l2a_manifest)
    collisions = sorted(predecessor.keys() & set(SOURCE_FILES))
    if collisions:
        _fail(f"L2B source inventory collides with predecessor: {collisions}")
    payloads, file_entries = _source_bytes()
    tar_payload = _canonical_tar(payloads)
    _verify_tar_structure(tar_payload, payloads)
    bundle = _publish_bundle(tar_payload, directory)
    return {
        "bundle": bundle,
        "files": file_entries,
        "predecessor_resolved_file_count": len(predecessor),
        "predecessor_resolved_inventory_sha256": inventory_digest,
        "resolved_file_count": len(predecessor) + len(SOURCE_FILES),
        "schema": "cach.stage2b_l2b.bundle_build.v1",
        "zero_collision_verified": True,
    }


def _verify_manifest_files(
    observed: object,
    payloads: dict[str, bytes],
    entries: list[dict[str, object]],
) -> None:
    if not _exact_equal(observed, entries):
        _fail("L2B source manifest file inventory differs")
    for entry in entries:
        path = entry["path"]
        if (
            _sha256(payloads[path]) != entry["sha256"]
            or len(payloads[path]) != entry["size_bytes"]
        ):
            _fail(f"L2B source member hash/size differs: {path}")


def _verify_bundle(pin: object, expected: bytes, payloads: dict[str, bytes]) -> None:
    if not isinstance(pin, dict) or set(pin) != {
        "mode",
        "path",
        "sha256",
        "size_bytes",
    }:
        _fail("L2B source bundle pin schema differs")
    digest = _sha256(expected)
    expected_path = (
        CANONICAL_BUNDLE_DIRECTORY
        / f"cach_stage2b_l2b_{digest}_{CHECKPOINT_DATE}.tar"
    )
    expected_pin = {
        "mode": "0444",
        "path": os.fspath(expected_path),
        "sha256": digest,
        "size_bytes": len(expected),
    }
    if not _exact_equal(pin, expected_pin):
        _fail("L2B source bundle pin differs")
    observed, metadata = _stable_read_regular(expected_path, "L2B source bundle")
    if (
        observed != expected
        or _sha256(observed) != digest
        or len(observed) != len(expected)
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
    ):
        _fail("L2B source bundle bytes, hash, size, or mode differ")
    _verify_tar_structure(observed, payloads)


def verify(manifest_path: Path, expected_manifest_sha256: str) -> dict[str, object]:
    if str(ROOT) != CANONICAL_WORKTREE:
        _fail("verification must run in the canonical H200 worktree")
    if SHA256_RE.fullmatch(expected_manifest_sha256) is None:
        _fail("expected manifest SHA256 must be 64 lowercase hexadecimal digits")
    if Path(os.path.abspath(os.fspath(manifest_path))) != DEFAULT_MANIFEST:
        _fail("L2B manifest must use the canonical default repository path")
    manifest_raw, _ = _stable_read_regular(manifest_path, "L2B outer manifest")
    manifest_sha256 = _sha256(manifest_raw)
    if manifest_sha256 != expected_manifest_sha256:
        _fail("L2B outer manifest differs from the caller trust anchor")
    manifest = _strict_json(manifest_raw, "L2B outer manifest")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        _fail("L2B outer manifest top-level schema differs")
    if (
        manifest["schema"] != "cach.stage2b_l2b.source_manifest.v1"
        or manifest["status"]
        != "l2b_source_checkpoint_verified_scope_not_admitted_non_scientific"
        or manifest["canonical_host"] != CANONICAL_HOST
        or manifest["canonical_worktree"] != CANONICAL_WORKTREE
        or manifest["created_at"] != "2026-07-31T00:00:00Z"
    ):
        _fail("L2B outer manifest identity differs")
    if not _exact_equal(manifest["base_lineage"], EXPECTED_BASE_LINEAGE):
        _fail("L2B predecessor lineage differs")
    if not _exact_equal(manifest["scope"], EXPECTED_SCOPE):
        _fail("L2B source scope differs")
    if not _exact_equal(
        manifest["git_recovery_bundles"], EXPECTED_GIT_RECOVERY_BUNDLES
    ):
        _fail("L2B git recovery-bundle declaration differs")
    if manifest["retained_stage2b_l2b_evidence"] is not None:
        _fail("L2B source checkpoint must not claim retained execution evidence")
    _verify_pin_set(
        manifest["execution_harness_pins"], HARNESS_PINS, "execution harness"
    )
    _verify_pin_set(
        manifest["inherited_source_pins"],
        INHERITED_SOURCE_PINS,
        "inherited source",
    )
    _verify_core_pins(manifest["frozen_l2b_core_pins"])

    l2a_manifest = _verify_l2a_predecessor()
    predecessor, inventory_digest = _resolved_predecessor_inventory(l2a_manifest)
    collisions = sorted(predecessor.keys() & set(SOURCE_FILES))
    expected_overlay = {
        "order": [
            "stage0",
            "stage1",
            "stage2",
            "stage2b",
            "stage2b_l2a",
            "stage2b_l2b",
        ],
        "predecessor_resolved_file_count": 112,
        "predecessor_resolved_inventory_sha256": inventory_digest,
        "resolved_file_count": 119,
        "stage2b_l2b_predecessor_collisions": [],
        "stage2b_l2b_requires_zero_predecessor_collisions": True,
    }
    if collisions or not _exact_equal(manifest["overlay"], expected_overlay):
        _fail("L2B zero-collision overlay declaration differs")

    payloads, entries = _source_bytes()
    _verify_manifest_files(manifest["files"], payloads, entries)
    canonical_tar = _canonical_tar(payloads)
    _verify_bundle(
        manifest["stage2b_l2b_source_bundle"], canonical_tar, payloads
    )
    return {
        "additive_artifact_valid": True,
        "caller_manifest_pin_verified": True,
        "capture_authorized": False,
        "deploy_authorized": False,
        "evaluation_authorized": False,
        "execution_replayed": False,
        "formal_filesystem_admission": False,
        "l2a_predecessor_verified": True,
        "l2b_runtime_authorized": False,
        "l2b_source_checkpoint_verified": True,
        "l3_authorized": False,
        "manifest_sha256": manifest_sha256,
        "model_executed": False,
        "recovery_authorized": False,
        "retained_immutable_execution_evidence": False,
        "runtime_closure_verified": False,
        "schema": "cach.stage2b_l2b.additive_artifact_verification.v1",
        "scientific_eligible": False,
        "source_file_count": 7,
        "stage3_authorized": False,
        "training_authorized": False,
        "transitive_runtime_closure": False,
        "zero_collision_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify or build the byte-scoped Stage-2B L2B source checkpoint."
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
        Stage2BL2BVerificationError,
        subprocess.SubprocessError,
        tarfile.TarError,
    ) as exc:
        failure = {
            "additive_artifact_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "l2b_runtime_authorized": False,
            "recovery_authorized": False,
            "schema": "cach.stage2b_l2b.additive_artifact_verification_failure.v1",
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
