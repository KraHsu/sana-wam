#!/usr/bin/env python3
"""Read-only verifier for the scoped CACH Stage-2 artifact set.

This verifier authenticates source/report/archive bytes.  It does not replay
the historical lightweight execution and therefore cannot grant Gate-S2,
scientific, training, evaluation, capture, deployment, or Stage-3 admission.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/cach_sana_wam/stage2/SOURCE_MANIFEST.json"
CANONICAL_HOST = "H200"
CANONICAL_WORKTREE = "/home/zch/workspace/sana-wam"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")

EXPECTED_HEADS = {
    "sana_afcc_handoff_head": "9586486f2a9f5172d57b325e32093a3e018d34c0",
    "sana_head": "16b9cec673e3335724ba2d8db25de7f9ed229292",
    "sana_wam_head": "605f1c134b4c983ff80f8489c4bc8847036329e2",
}
EXPECTED_STAGE1_BUNDLE = {
    "mode": "0444",
    "path": (
        "/DATA/share/sana_cach_source_bundles/"
        "cach_stage1_a6e7849b91464b6e939db6ac676ca8e0d139a8e889e2b6fedb730457d3ee31dc_"
        "20260731.tar"
    ),
    "sha256": "a6e7849b91464b6e939db6ac676ca8e0d139a8e889e2b6fedb730457d3ee31dc",
    "size_bytes": 870400,
}

MANIFEST_KEYS = {
    "base_lineage",
    "canonical_host",
    "canonical_worktree",
    "created_at",
    "files",
    "protected_worktree_state",
    "schema",
    "scope",
    "stage2_source_bundle",
    "status",
}
EXECUTION_EVIDENCE_KEYS = {"files", "root_mode", "root_path"}
EXECUTION_EVIDENCE_FILE_KEYS = {"mode", "name", "sha256", "size_bytes"}
FAILURE_EVIDENCE_KEYS = {"receipt", "root_mode", "root_path"}
SCOPE_KEYS = {
    "file_count",
    "kind",
    "source_manifest_excluded_to_avoid_self_reference",
    "transitive_runtime_closure",
}
PROTECTED_STATE_KEYS = {
    "phase6_candidate_eligibility_binary_diff_sha256",
    "phase6_candidate_eligibility_worktree_sha256",
    "stage0_config_guard_cli_sha256",
    "stage0_config_guard_sha256",
}
REPORT_KEYS = {
    "canonical_host",
    "canonical_worktree",
    "environment",
    "generated_at",
    "gate_status",
    "initial_numeric_hardening",
    "limitations",
    "prohibited_actions_attestation",
    "repository_lineage",
    "results",
    "schema",
    "scoped_contract",
    "status",
    "verification",
}
REPORT_GATE_STATUS = {
    "full_gate_s2": "blocked_not_claimed",
    "offline_test_owned_teacher_forcing_subset": "passed",
    "scope": "random_mini_models_and_synthetic_tensors_only",
    "stage3_authorized": False,
    "training_authorized": False,
}
REPORT_RESULTS = {
    "gate_s2": "blocked",
    "scoped_applied_action_ack_verifier": "passed",
    "scoped_failure_receipt_evidence": "passed",
    "scoped_paired_forward_cache_admission": "passed",
}
REPORT_PROHIBITED_ACTIONS = {
    "capture_started": False,
    "checkpoint_loaded": False,
    "dataset_scanned": False,
    "evaluation_started": False,
    "formal_root_created": False,
    "optimizer_created": False,
    "training_started": False,
}
REPORT_INITIAL_NUMERIC_HARDENING = {
    "failure_code": "COMMIT_STAGING_FAILED",
    "reason": "FP32 action timestep supplied to BF16 mini-ActionDiT",
    "receipt_published": False,
    "remediation": (
        "construct action t=0 explicitly in model BF16 dtype; keep video t=0 FP32"
    ),
    "state_advanced": False,
    "tolerance_relaxed": False,
}
REPORT_LIMITATIONS = {
    "committed_action_identity_and_history_summary": "blocked",
    "formal_failure_or_run_root": "not_created",
    "full_model_gradient_jvp_evidence": "blocked",
    "model_owned_no_action_in_production_stager": "blocked",
    "production_aborted_ledger": "not_implemented",
    "production_architecture_and_stager": "not_implemented",
    "production_deploy_ack_transport_commit_and_durable_replay": "blocked",
    "production_receipt_publisher": "not_implemented",
    "retained_immutable_raw_execution_evidence": (
        "retained_read_only_scoped_nonformal_artifact"
    ),
    "scientific_eligible": False,
    "transitive_runtime_closure": "not_established",
}
STAGE2_SELECTED_FILES = [
    "tests/test_cach_stage2_prefix_compaction.py",
    "tests/test_cach_stage2_hybrid_cache_codec.py",
    "tests/test_cach_stage2_failure_evidence.py",
    "tests/test_cach_stage2_mini_gdn.py",
    "tests/test_cach_stage2_paired_mini_gdn.py",
    "tests/test_cach_action_to_video_conditioning.py",
    "tests/test_cach_stage2_applied_action_ack.py",
]


def _stable_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_rdev,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_stable_descriptor(
    descriptor: int,
    name: str,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} must be a regular file")
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if _stable_signature(before) != _stable_signature(after):
        raise ValueError(f"{name} changed during stable read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise ValueError(f"{name} size changed during stable read")
    return raw, before


def _stable_read_regular(path: Path, name: str) -> tuple[bytes, os.stat_result]:
    """Read a regular file once through one nofollow fd and detect mutation."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("O_NOFOLLOW is required for Stage-2 verification")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        return _read_stable_descriptor(descriptor, name)
    finally:
        os.close(descriptor)


def _strict_json_bytes(raw: bytes, name: str) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"{name} contains non-finite number {token}")
        return value

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{name} contains non-finite JSON token {token}")
        ),
        parse_float=finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _repository_file(relative_path: object) -> Path:
    if type(relative_path) is not str:
        raise ValueError("manifest path must be a string")
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"non-canonical repository path {relative_path!r}")
    candidate = ROOT.joinpath(*pure.parts)
    candidate.relative_to(ROOT)
    return candidate


def _manifest_file(path: Path) -> tuple[Path, bool]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        absolute.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("Stage-2 manifest must be inside the repository") from exc
    return absolute, absolute == DEFAULT_MANIFEST


def _git_output(*arguments: str, cwd: Path = ROOT) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _validate_external_pin(pin: object, name: str) -> tuple[Path, str, int]:
    if not isinstance(pin, dict) or set(pin) != {
        "mode",
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError(f"{name} pin schema differs")
    if pin["mode"] != "0444":
        raise ValueError(f"{name} pin mode must be 0444")
    if type(pin["path"]) is not str or not pin["path"].startswith("/"):
        raise ValueError(f"{name} pin path is invalid")
    pure_path = PurePosixPath(pin["path"])
    if (
        pure_path.as_posix() != pin["path"]
        or any(part in {".", ".."} for part in pure_path.parts)
    ):
        raise ValueError(f"{name} pin path is non-canonical")
    if type(pin["sha256"]) is not str or SHA256.fullmatch(pin["sha256"]) is None:
        raise ValueError(f"{name} pin SHA256 is invalid")
    if type(pin["size_bytes"]) is not int or pin["size_bytes"] <= 0:
        raise ValueError(f"{name} pin size is invalid")
    return Path(pin["path"]), pin["sha256"], pin["size_bytes"]


def _read_external_regular_file(pin: object, name: str) -> tuple[Path, bytes]:
    path, expected_sha256, expected_size = _validate_external_pin(pin, name)
    raw, metadata = _stable_read_regular(path, name)
    if (
        stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_size != expected_size
        or _sha_bytes(raw) != expected_sha256
    ):
        raise ValueError(f"{name} bytes/metadata differ")
    return path, raw


def _verify_retained_execution_evidence(evidence: object) -> bool:
    """Authenticate retained logs without treating them as execution replay."""

    if not isinstance(evidence, dict) or set(evidence) != EXECUTION_EVIDENCE_KEYS:
        raise ValueError("Stage-2 execution-evidence schema differs")
    if evidence["root_mode"] != "0500":
        raise ValueError("Stage-2 execution-evidence root mode must be 0500")
    if type(evidence["root_path"]) is not str:
        raise ValueError("Stage-2 execution-evidence root path is invalid")
    pure_root = PurePosixPath(evidence["root_path"])
    evidence_parent = PurePosixPath("/DATA/share/sana_cach_stage2_evidence")
    if (
        not pure_root.is_absolute()
        or pure_root.as_posix() != evidence["root_path"]
        or pure_root.parent != evidence_parent
        or pure_root == evidence_parent
        or any(part in {".", ".."} for part in pure_root.parts)
    ):
        raise ValueError("Stage-2 execution-evidence root path is non-canonical")

    files = evidence["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Stage-2 execution-evidence file set is empty")
    expected_names = []
    pins = {}
    for pin in files:
        if not isinstance(pin, dict) or set(pin) != EXECUTION_EVIDENCE_FILE_KEYS:
            raise ValueError("Stage-2 execution-evidence file schema differs")
        if pin["mode"] != "0400":
            raise ValueError("Stage-2 execution-evidence file mode must be 0400")
        if type(pin["name"]) is not str:
            raise ValueError("Stage-2 execution-evidence file name is invalid")
        pure_name = PurePosixPath(pin["name"])
        if (
            pure_name.is_absolute()
            or len(pure_name.parts) != 1
            or pure_name.as_posix() != pin["name"]
            or pure_name.name in {"", ".", ".."}
        ):
            raise ValueError("Stage-2 execution-evidence file name is non-canonical")
        if type(pin["sha256"]) is not str or SHA256.fullmatch(pin["sha256"]) is None:
            raise ValueError("Stage-2 execution-evidence file SHA256 is invalid")
        if type(pin["size_bytes"]) is not int or pin["size_bytes"] <= 0:
            raise ValueError("Stage-2 execution-evidence file size is invalid")
        expected_names.append(pin["name"])
        pins[pin["name"]] = pin
    if expected_names != sorted(expected_names) or len(pins) != len(files):
        raise ValueError("Stage-2 execution-evidence files must be sorted and unique")

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError("O_NOFOLLOW and O_DIRECTORY are required for evidence verification")
    root_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_descriptor = os.open(Path(evidence["root_path"]), root_flags)
    try:
        root_before = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or stat.S_IMODE(root_before.st_mode) != 0o500
        ):
            raise ValueError("Stage-2 execution-evidence root metadata differs")
        if sorted(os.listdir(root_descriptor)) != expected_names:
            raise ValueError("Stage-2 execution-evidence root inventory differs")

        file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        for name in expected_names:
            descriptor = os.open(name, file_flags, dir_fd=root_descriptor)
            try:
                raw, metadata = _read_stable_descriptor(
                    descriptor,
                    f"Stage-2 execution evidence {name}",
                )
            finally:
                os.close(descriptor)
            pin = pins[name]
            if (
                stat.S_IMODE(metadata.st_mode) != 0o400
                or metadata.st_size != pin["size_bytes"]
                or _sha_bytes(raw) != pin["sha256"]
            ):
                raise ValueError(f"Stage-2 execution evidence differs: {name}")

        if sorted(os.listdir(root_descriptor)) != expected_names:
            raise ValueError("Stage-2 execution-evidence root inventory changed")
        root_after = os.fstat(root_descriptor)
        if _stable_signature(root_before) != _stable_signature(root_after):
            raise ValueError("Stage-2 execution-evidence root changed during verification")
    finally:
        os.close(root_descriptor)
    return True


def _verify_retained_failure_evidence(evidence: object) -> bool:
    """Authenticate the separately frozen synthetic failure receipt root."""

    if not isinstance(evidence, dict) or set(evidence) != FAILURE_EVIDENCE_KEYS:
        raise ValueError("Stage-2 retained-failure-evidence schema differs")
    if evidence["root_mode"] != "0500":
        raise ValueError("Stage-2 retained-failure-evidence root mode must be 0500")
    if type(evidence["root_path"]) is not str:
        raise ValueError("Stage-2 retained-failure-evidence root path is invalid")
    pure_root = PurePosixPath(evidence["root_path"])
    evidence_parent = PurePosixPath(
        "/DATA/share/sana_cach_stage2_failure_evidence"
    )
    if (
        not pure_root.is_absolute()
        or pure_root.as_posix() != evidence["root_path"]
        or pure_root.parent != evidence_parent
        or pure_root == evidence_parent
        or any(part in {".", ".."} for part in pure_root.parts)
    ):
        raise ValueError("Stage-2 retained-failure-evidence root is non-canonical")

    receipt = evidence["receipt"]
    if (
        not isinstance(receipt, dict)
        or set(receipt) != EXECUTION_EVIDENCE_FILE_KEYS
        or receipt["name"] != "FAILURE.json"
        or receipt["mode"] != "0400"
        or type(receipt["sha256"]) is not str
        or SHA256.fullmatch(receipt["sha256"]) is None
        or type(receipt["size_bytes"]) is not int
        or receipt["size_bytes"] <= 0
    ):
        raise ValueError("Stage-2 retained failure receipt pin differs")

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError("O_NOFOLLOW and O_DIRECTORY are required for evidence verification")
    root_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_descriptor = os.open(Path(evidence["root_path"]), root_flags)
    try:
        root_before = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or stat.S_IMODE(root_before.st_mode) != 0o500
            or os.listdir(root_descriptor) != ["FAILURE.json"]
        ):
            raise ValueError("Stage-2 retained-failure-evidence root differs")
        descriptor = os.open(
            "FAILURE.json",
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=root_descriptor,
        )
        try:
            raw, metadata = _read_stable_descriptor(
                descriptor,
                "Stage-2 retained failure receipt",
            )
        finally:
            os.close(descriptor)
        if (
            stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_size != receipt["size_bytes"]
            or _sha_bytes(raw) != receipt["sha256"]
        ):
            raise ValueError("Stage-2 retained failure receipt differs")
        if os.listdir(root_descriptor) != ["FAILURE.json"]:
            raise ValueError("Stage-2 retained-failure-evidence inventory changed")
        root_after = os.fstat(root_descriptor)
        if _stable_signature(root_before) != _stable_signature(root_after):
            raise ValueError("Stage-2 retained-failure-evidence root changed")
    finally:
        os.close(root_descriptor)
    return True


def _verify_report(report: dict) -> dict[str, dict]:
    if set(report) != REPORT_KEYS:
        raise ValueError("Stage-2 report top-level schema differs")
    if (
        report["schema"] != "cach.stage2.offline_subset_report.v2"
        or report["status"]
        != "offline_teacher_forcing_subset_passed_full_gate_s2_blocked"
        or report["canonical_host"] != CANONICAL_HOST
        or report["canonical_worktree"] != CANONICAL_WORKTREE
    ):
        raise ValueError("Stage-2 report identity/scope differs")
    if report["gate_status"] != REPORT_GATE_STATUS:
        raise ValueError("Stage-2 report gate boundary differs")
    if report["results"] != REPORT_RESULTS:
        raise ValueError("Stage-2 report result boundary differs")
    if report["prohibited_actions_attestation"] != REPORT_PROHIBITED_ACTIONS:
        raise ValueError("Stage-2 report prohibited-action boundary differs")
    if report["initial_numeric_hardening"] != REPORT_INITIAL_NUMERIC_HARDENING:
        raise ValueError("Stage-2 numeric-hardening boundary differs")
    if report["limitations"] != REPORT_LIMITATIONS:
        raise ValueError("Stage-2 report limitation boundary differs")

    expected_lineage = {
        **EXPECTED_HEADS,
        "stage1_source_bundle_sha256": EXPECTED_STAGE1_BUNDLE["sha256"],
    }
    if report["repository_lineage"] != expected_lineage:
        raise ValueError("Stage-2 report repository lineage differs")

    scoped_contract = report["scoped_contract"]
    if not isinstance(scoped_contract, dict) or set(scoped_contract) != {
        "action_timestep",
        "applied_action_ack",
        "digest_binding",
        "geometry",
        "receipt",
        "video_timestep",
    }:
        raise ValueError("Stage-2 report scoped-contract schema differs")
    if (
        scoped_contract["action_timestep"] != "BF16 zero"
        or scoped_contract["applied_action_ack"]
        != (
            "standalone verifier only; manager/server deploy commit remains "
            "hard-disabled"
        )
        or scoped_contract["video_timestep"] != "FP32 zero"
        or scoped_contract["geometry"]
        != {
            "K": 3,
            "L": 8,
            "action_capacities": [16, 24, 24],
            "r": 8,
            "valid_action_counts": [16, 24, 16],
            "video_valid_lengths": [3, 3, 2],
        }
        or scoped_contract["receipt"]
        != (
            "test-local success receipts plus separately retained read-only "
            "synthetic failure receipt; not a production ledger/publisher or "
            "formal root"
        )
        or type(scoped_contract["digest_binding"]) is not str
        or not scoped_contract["digest_binding"]
    ):
        raise ValueError("Stage-2 report scoped-contract values differ")

    if type(report["generated_at"]) is not str or not report["generated_at"]:
        raise ValueError("Stage-2 report generation time is invalid")
    if not isinstance(report["environment"], dict) or set(report["environment"]) != {
        "gpu",
        "kernel",
        "pytest",
        "python",
        "torch",
    }:
        raise ValueError("Stage-2 report environment schema differs")
    verification = report["verification"]
    if not isinstance(verification, list) or len(verification) != 5:
        raise ValueError("Stage-2 historical verification attestation is absent")
    if any(not isinstance(record, dict) for record in verification):
        raise ValueError("Stage-2 verification record schema differs")

    def one_record(field: str, value: str) -> dict:
        matches = [record for record in verification if record.get(field) == value]
        if len(matches) != 1:
            raise ValueError(f"Stage-2 verification lacks unique {field}={value}")
        return matches[0]

    stage2 = one_record(
        "command_class",
        "CUDA_VISIBLE_DEVICES=0 GDN_DISABLE_COMPILE=1 pytest Stage-2 selected suite",
    )
    if (
        set(stage2)
        != {
            "capture_root",
            "capture_status",
            "command_class",
            "failed",
            "final_rerun_confirmation_pending",
            "junit",
            "junit_xml_sha256",
            "log_sha256",
            "passed",
            "result",
            "retained_raw_log",
            "selected_files",
            "warnings",
        }
        or stage2["capture_status"] != "final_retained_scoped_nonformal"
        or stage2["failed"] != 0
        or stage2["final_rerun_confirmation_pending"] is not False
        or stage2["junit"] is not True
        or stage2["passed"] != 75
        or stage2["result"] != "retained_final_pass"
        or stage2["retained_raw_log"] is not True
        or stage2["selected_files"] != STAGE2_SELECTED_FILES
        or stage2["warnings"] != 13
    ):
        raise ValueError("Stage-2 selected-suite verification differs")

    cpu = one_record(
        "command_class",
        "CUDA_VISIBLE_DEVICES='' affected CPU regression suite",
    )
    if (
        set(cpu)
        != {
            "capture_root",
            "capture_status",
            "command_class",
            "deselected",
            "failed",
            "junit",
            "junit_xml_sha256",
            "log_sha256",
            "passed",
            "result",
            "retained_raw_log",
            "skipped",
        }
        or cpu["capture_status"] != "final_retained_scoped_nonformal"
        or cpu["deselected"] != 1
        or cpu["failed"] != 0
        or cpu["junit"] is not True
        or cpu["passed"] != 264
        or cpu["result"] != "retained_final_pass"
        or cpu["retained_raw_log"] is not True
        or cpu["skipped"] != 10
    ):
        raise ValueError("Stage-2 CPU verification differs")

    static = one_record("type", "static_checks")
    if (
        set(static)
        != {
            "capture_root",
            "capture_status",
            "files",
            "log_sha256",
            "result",
            "retained_raw_log",
            "type",
        }
        or static["capture_status"] != "final_retained_scoped_nonformal"
        or static["files"] != 13
        or static["result"] != "AST_COMPILE_WHITESPACE_OK 13"
        or static["retained_raw_log"] is not True
    ):
        raise ValueError("Stage-2 static verification differs")

    aborted = one_record("capture_status", "CAPTURE_ABORTED")
    if aborted != {
        "capture_root": (
            "/DATA/share/sana_cach_stage2_evidence/"
            "stage2_offline_subset_20260731_pPSovwEo"
        ),
        "capture_status": "CAPTURE_ABORTED",
        "reason": "post-test metadata quoting failure",
        "retained_read_only": True,
        "root_mode": "0500",
    }:
        raise ValueError("Stage-2 aborted capture record differs")

    failure = one_record("scope", "synthetic_scoped_nonformal")
    if (
        set(failure)
        != {
            "failure_receipt_sha256",
            "failure_root",
            "receipt_mode",
            "retained_read_only",
            "root_mode",
            "scope",
        }
        or failure["receipt_mode"] != "0400"
        or failure["retained_read_only"] is not True
        or failure["root_mode"] != "0500"
    ):
        raise ValueError("Stage-2 retained failure record differs")

    for record, digest_fields in (
        (stage2, ("junit_xml_sha256", "log_sha256")),
        (cpu, ("junit_xml_sha256", "log_sha256")),
        (static, ("log_sha256",)),
        (failure, ("failure_receipt_sha256",)),
    ):
        if any(
            type(record[field]) is not str
            or SHA256.fullmatch(record[field]) is None
            for field in digest_fields
        ):
            raise ValueError("Stage-2 verification digest differs")
    if not (
        type(stage2["capture_root"]) is str
        and stage2["capture_root"] == cpu["capture_root"] == static["capture_root"]
        and type(failure["failure_root"]) is str
    ):
        raise ValueError("Stage-2 verification roots differ")
    return {
        "aborted": aborted,
        "cpu": cpu,
        "failure": failure,
        "stage2": stage2,
        "static": static,
    }


def _verify_stage2_bundle(
    bundle: object,
    *,
    paths: list[str],
    expected: dict[str, str],
) -> tuple[bool, bool]:
    bundle_path, bundle_sha256, _ = _validate_external_pin(
        bundle,
        "Stage-2 source bundle",
    )
    expected_parent = Path("/DATA/share/sana_cach_source_bundles")
    name_match = re.fullmatch(
        r"cach_stage2_([0-9a-f]{64})_20260731\.tar",
        bundle_path.name,
    )
    if (
        bundle_path.parent != expected_parent
        or name_match is None
        or name_match.group(1) != bundle_sha256
    ):
        raise ValueError("Stage-2 source-bundle path is not digest-bound")

    _, bundle_raw = _read_external_regular_file(bundle, "Stage-2 source bundle")
    with tarfile.open(fileobj=io.BytesIO(bundle_raw), mode="r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != paths or not all(
            member.isfile() for member in members
        ):
            raise ValueError("Stage-2 bundle member inventory differs")
        observed = {}
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("Stage-2 bundle member is unreadable")
            member_raw = extracted.read()
            if len(member_raw) != member.size:
                raise ValueError("Stage-2 bundle member size differs")
            observed[member.name] = _sha_bytes(member_raw)
    if observed != expected:
        raise ValueError("Stage-2 bundle member bytes differ")
    return True, True


def _verify_report_evidence_binding(
    records: dict[str, dict],
    execution_evidence: object,
    failure_evidence: object,
) -> None:
    if not isinstance(execution_evidence, dict) or not isinstance(
        failure_evidence, dict
    ):
        raise ValueError("Stage-2 report claims unpinned retained evidence")
    if records["stage2"]["capture_root"] != execution_evidence.get("root_path"):
        raise ValueError("Stage-2 report execution root differs from manifest")
    files = execution_evidence.get("files")
    if not isinstance(files, list):
        raise ValueError("Stage-2 execution-evidence pins are absent")
    pins = {
        pin.get("name"): pin.get("sha256")
        for pin in files
        if isinstance(pin, dict)
    }
    expected = {
        "STATIC_CHECKS.log": records["static"]["log_sha256"],
        "cpu_regression.log": records["cpu"]["log_sha256"],
        "cpu_regression.xml": records["cpu"]["junit_xml_sha256"],
        "stage2_pytest.log": records["stage2"]["log_sha256"],
        "stage2_pytest.xml": records["stage2"]["junit_xml_sha256"],
    }
    if any(pins.get(name) != digest for name, digest in expected.items()):
        raise ValueError("Stage-2 report log/JUnit pins differ from manifest")
    if records["failure"]["failure_root"] != failure_evidence.get("root_path"):
        raise ValueError("Stage-2 report failure root differs from manifest")
    receipt = failure_evidence.get("receipt")
    if (
        not isinstance(receipt, dict)
        or receipt.get("sha256")
        != records["failure"]["failure_receipt_sha256"]
    ):
        raise ValueError("Stage-2 report failure receipt differs from manifest")


def verify(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str | None,
    verify_bundle: bool,
) -> dict[str, object]:
    manifest_path, default_manifest_path = _manifest_file(manifest_path)
    manifest_raw, _ = _stable_read_regular(manifest_path, "Stage-2 manifest")
    manifest_sha256 = _sha_bytes(manifest_raw)
    caller_manifest_pin_verified = expected_manifest_sha256 is not None
    if expected_manifest_sha256 is not None:
        if SHA256.fullmatch(expected_manifest_sha256) is None:
            raise ValueError("expected manifest SHA256 is invalid")
        if manifest_sha256 != expected_manifest_sha256:
            raise ValueError("Stage-2 manifest differs from caller pin")
    manifest = _strict_json_bytes(manifest_raw, "Stage-2 manifest")
    manifest_keys = set(manifest)
    optional_manifest_keys = {
        "execution_evidence",
        "retained_failure_evidence",
    }
    if not MANIFEST_KEYS.issubset(manifest_keys) or not manifest_keys.issubset(
        MANIFEST_KEYS | optional_manifest_keys
    ):
        raise ValueError("Stage-2 manifest top-level schema differs")
    if (
        manifest["schema"] != "cach.stage2.source_manifest.v1"
        or manifest["status"]
        != "offline_teacher_forcing_subset_passed_full_gate_s2_blocked"
        or manifest["canonical_host"] != CANONICAL_HOST
        or manifest["canonical_worktree"] != CANONICAL_WORKTREE
        or type(manifest["created_at"]) is not str
        or not manifest["created_at"]
    ):
        raise ValueError("Stage-2 manifest identity/scope differs")

    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Stage-2 manifest file set is empty")
    paths: list[str] = []
    source_bytes: dict[str, bytes] = {}
    expected_files: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("Stage-2 manifest file-entry schema differs")
        if type(entry["sha256"]) is not str or SHA256.fullmatch(entry["sha256"]) is None:
            raise ValueError("Stage-2 file SHA256 is invalid")
        relative_path = entry["path"]
        path = _repository_file(relative_path)
        raw, _ = _stable_read_regular(path, f"Stage-2 source {relative_path}")
        if _sha_bytes(raw) != entry["sha256"]:
            raise ValueError(f"Stage-2 source SHA differs: {relative_path}")
        paths.append(relative_path)
        source_bytes[relative_path] = raw
        expected_files[relative_path] = entry["sha256"]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Stage-2 paths must be sorted and unique")

    scope = manifest["scope"]
    if (
        not isinstance(scope, dict)
        or set(scope) != SCOPE_KEYS
        or type(scope["file_count"]) is not int
        or scope["file_count"] != len(files)
        or scope["kind"] != "stage2_delta_subset_over_stage1_bundle"
        or scope["transitive_runtime_closure"] is not False
        or scope["source_manifest_excluded_to_avoid_self_reference"] is not True
        or "docs/cach_sana_wam/stage2/SOURCE_MANIFEST.json" in paths
    ):
        raise ValueError("Stage-2 source scope differs")

    source_list_path = "docs/cach_sana_wam/stage2/STAGE2_SOURCE_FILES.txt"
    if source_list_path not in source_bytes:
        raise ValueError("Stage-2 source list is absent")
    try:
        source_list_text = source_bytes[source_list_path].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Stage-2 source list is not UTF-8") from exc
    if not source_list_text.endswith("\n") or "\r" in source_list_text:
        raise ValueError("Stage-2 source list encoding/newline differs")
    source_list = source_list_text[:-1].split("\n")
    if source_list != paths:
        raise ValueError("Stage-2 source list differs from manifest ordering")

    report_path = "docs/cach_sana_wam/stage2/STAGE2_MINI_ADMISSION_REPORT.json"
    if report_path not in source_bytes:
        raise ValueError("Stage-2 report is absent from source closure")
    report = _strict_json_bytes(source_bytes[report_path], "Stage-2 report")
    report_records = _verify_report(report)
    if "execution_evidence" in manifest or "retained_failure_evidence" in manifest:
        _verify_report_evidence_binding(
            report_records,
            manifest.get("execution_evidence"),
            manifest.get("retained_failure_evidence"),
        )

    lineage = manifest["base_lineage"]
    if (
        not isinstance(lineage, dict)
        or set(lineage) != {*EXPECTED_HEADS, "stage1_source_bundle"}
        or any(lineage[key] != value for key, value in EXPECTED_HEADS.items())
        or lineage["stage1_source_bundle"] != EXPECTED_STAGE1_BUNDLE
    ):
        raise ValueError("Stage-2 base-lineage schema/pins differ")
    observed_heads = {
        "sana_wam_head": _git_output("rev-parse", "HEAD").decode().strip(),
        "sana_head": _git_output(
            "rev-parse",
            "HEAD",
            cwd=ROOT / "third_party/Sana",
        )
        .decode()
        .strip(),
        "sana_afcc_handoff_head": _git_output(
            "rev-parse",
            "HEAD",
            cwd=ROOT.parent / "sana-afcc-handoff",
        )
        .decode()
        .strip(),
    }
    if observed_heads != EXPECTED_HEADS:
        raise ValueError("Stage-2 repository lineage differs")
    _read_external_regular_file(
        lineage["stage1_source_bundle"],
        "Stage-1 base source bundle",
    )
    stage1_lineage_verified = True

    protected = manifest["protected_worktree_state"]
    if not isinstance(protected, dict) or set(protected) != PROTECTED_STATE_KEYS:
        raise ValueError("Stage-2 protected-state schema differs")
    if any(
        type(protected[key]) is not str or SHA256.fullmatch(protected[key]) is None
        for key in PROTECTED_STATE_KEYS
    ):
        raise ValueError("Stage-2 protected-state SHA256 is invalid")
    protected_files = {
        "phase6_candidate_eligibility_worktree_sha256": (
            "tests/test_phase6_candidate_eligibility.py"
        ),
        "stage0_config_guard_sha256": "src/sana_wam/train/cach_stage0_guard.py",
        "stage0_config_guard_cli_sha256": (
            "scripts/check_cach_stage0_reserved_config.py"
        ),
    }
    for pin_name, relative_path in protected_files.items():
        raw, _ = _stable_read_regular(
            _repository_file(relative_path),
            f"protected state {relative_path}",
        )
        if _sha_bytes(raw) != protected[pin_name]:
            raise ValueError(f"protected state differs: {pin_name}")
    phase6_diff = _git_output(
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--binary",
        "--",
        "tests/test_phase6_candidate_eligibility.py",
    )
    if _sha_bytes(phase6_diff) != protected[
        "phase6_candidate_eligibility_binary_diff_sha256"
    ]:
        raise ValueError("protected Phase-6 binary diff differs")

    stage2_source_bundle_verified = False
    stage2_bundle_members_verified = False
    if verify_bundle:
        (
            stage2_source_bundle_verified,
            stage2_bundle_members_verified,
        ) = _verify_stage2_bundle(
            manifest["stage2_source_bundle"],
            paths=paths,
            expected=expected_files,
        )
    else:
        _validate_external_pin(
            manifest["stage2_source_bundle"],
            "Stage-2 source bundle",
        )

    retained_execution_evidence_verified = False
    if "execution_evidence" in manifest:
        retained_execution_evidence_verified = _verify_retained_execution_evidence(
            manifest["execution_evidence"]
        )
    retained_failure_evidence_verified = False
    if "retained_failure_evidence" in manifest:
        retained_failure_evidence_verified = _verify_retained_failure_evidence(
            manifest["retained_failure_evidence"]
        )

    canonical_worktree_runtime_verified = str(ROOT) == CANONICAL_WORKTREE
    scoped_stage2_artifact_set_valid = all(
        (
            default_manifest_path,
            caller_manifest_pin_verified,
            verify_bundle,
            stage1_lineage_verified,
            stage2_source_bundle_verified,
            stage2_bundle_members_verified,
            retained_execution_evidence_verified,
            retained_failure_evidence_verified,
            canonical_worktree_runtime_verified,
        )
    )
    diagnostic_reasons = []
    if not default_manifest_path:
        diagnostic_reasons.append("non_default_manifest_path")
    if not caller_manifest_pin_verified:
        diagnostic_reasons.append("caller_manifest_sha256_absent")
    if not verify_bundle:
        diagnostic_reasons.append("stage2_external_bundle_skipped")
    if not retained_execution_evidence_verified:
        diagnostic_reasons.append("retained_execution_evidence_absent")
    if not retained_failure_evidence_verified:
        diagnostic_reasons.append("retained_failure_evidence_absent")
    if not canonical_worktree_runtime_verified:
        diagnostic_reasons.append("non_canonical_runtime_worktree")

    return {
        "caller_manifest_pin_verified": caller_manifest_pin_verified,
        "canonical_worktree_runtime_verified": canonical_worktree_runtime_verified,
        "capture_authorized": False,
        "default_manifest_path_verified": default_manifest_path,
        "deploy_applied_ack_enabled": False,
        "diagnostic_only": not scoped_stage2_artifact_set_valid,
        "diagnostic_reasons": diagnostic_reasons,
        "evaluation_authorized": False,
        "execution_replayed": False,
        "external_trust_anchor_verified": False,
        "formal_root_created": False,
        "historical_execution_attestation_only": True,
        "manifest_sha256": manifest_sha256,
        "protected_state_verified": True,
        "retained_execution_evidence_verified": (
            retained_execution_evidence_verified
        ),
        "retained_failure_evidence_verified": retained_failure_evidence_verified,
        "registered_execution_authority": False,
        "repository_lineage_verified": True,
        "schema": "cach.stage2.artifact_verification.v2",
        "scientific_eligible": False,
        "scoped_stage2_artifact_set_valid": scoped_stage2_artifact_set_valid,
        "source_file_count": len(files),
        "source_list_verified": True,
        "stage1_source_bundle_verified": stage1_lineage_verified,
        "stage2_bundle_members_verified": stage2_bundle_members_verified,
        "stage2_source_bundle_verified": stage2_source_bundle_verified,
        "stage3_authorized": False,
        "training_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--skip-external-bundle", action="store_true")
    args = parser.parse_args()
    try:
        report = verify(
            args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            verify_bundle=not args.skip_external_bundle,
        )
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        tarfile.TarError,
    ) as exc:
        print(str(exc), file=os.sys.stderr)
        return 2
    print(json.dumps(report, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
