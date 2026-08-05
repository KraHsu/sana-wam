#!/usr/bin/env python3
"""Review CACH Stage 0 denial drafts without importing torch or mutating state.

This externally anchored self-check is a review aid, not an independent
executable trust root: Python startup precedes self-hashing and ``git`` is not
binary-pinned in Stage 0.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import types
from hashlib import sha256
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_AUTHORITY_RELATIVE = "docs/cach_sana_wam/stage0/CACH_A_AUTHORITY.draft.json"
_CONTRACT_RELATIVE = "src/sana_wam/train/cach_stage0_contract.py"
_SELF_RELATIVE = "scripts/verify_cach_stage0.py"
_BOOTSTRAP_LIMIT = 8 * 1024 * 1024


class _BootstrapError(ValueError):
    """Raised before the externally pinned Stage-0 contract is loaded."""


def _bootstrap_expected_sha256(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _BootstrapError("expected authority SHA256 must be lowercase hex")
    return value


def _bootstrap_read_regular(path: Path, name: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _BootstrapError(f"cannot open bootstrap {name}: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _BootstrapError(f"bootstrap {name} is not a regular file: {path}")
        if before.st_size > _BOOTSTRAP_LIMIT:
            raise _BootstrapError(f"bootstrap {name} exceeds size limit: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _BOOTSTRAP_LIMIT:
                raise _BootstrapError(f"bootstrap {name} exceeds size limit: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise _BootstrapError(f"bootstrap {name} changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _bootstrap_json(data: bytes) -> dict:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise _BootstrapError(f"duplicate bootstrap authority key: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                _BootstrapError(f"non-finite bootstrap authority value: {token}")
            ),
        )
    except _BootstrapError:
        raise
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
        raise _BootstrapError(f"invalid bootstrap authority: {exc}") from exc
    if type(value) is not dict:
        raise _BootstrapError("bootstrap authority must be one JSON object")
    return value


def _bootstrap_pin(authority: dict, role: str, expected_path: str) -> str:
    designs = authority.get("designs")
    pin = designs.get(role) if type(designs) is dict else None
    if (
        type(pin) is not dict
        or set(pin) != {"path", "sha256"}
        or pin.get("path") != expected_path
    ):
        raise _BootstrapError(f"bootstrap authority {role} pin differs")
    digest = pin.get("sha256")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise _BootstrapError(f"bootstrap authority {role} SHA256 differs")
    return digest


def _load_pinned_contract(
    repository_root: Path,
    authority_path: Path,
    expected_authority_sha256: str,
):
    """Authenticate authority, this verifier, and contract before executing repo code."""

    expected = _bootstrap_expected_sha256(expected_authority_sha256)
    authority_bytes = _bootstrap_read_regular(authority_path, "authority")
    if sha256(authority_bytes).hexdigest() != expected:
        raise _BootstrapError("authority trust anchor differs")
    bootstrap_authority = _bootstrap_json(authority_bytes)

    root = repository_root.resolve(strict=True)
    self_path = root / _SELF_RELATIVE
    if Path(__file__).resolve(strict=True) != self_path.resolve(strict=True):
        raise _BootstrapError("verifier path differs from authority repository root")
    self_expected = _bootstrap_pin(
        bootstrap_authority, "static_verifier", _SELF_RELATIVE
    )
    self_bytes = _bootstrap_read_regular(self_path, "static verifier")
    if sha256(self_bytes).hexdigest() != self_expected:
        raise _BootstrapError("static verifier bytes differ from authority pin")

    contract_expected = _bootstrap_pin(
        bootstrap_authority, "stage0_contract", _CONTRACT_RELATIVE
    )
    contract_path = root / _CONTRACT_RELATIVE
    contract_bytes = _bootstrap_read_regular(contract_path, "Stage-0 contract")
    if sha256(contract_bytes).hexdigest() != contract_expected:
        raise _BootstrapError("Stage-0 contract bytes differ from authority pin")

    contract = types.ModuleType("_cach_stage0_contract_authority_pinned")
    contract.__file__ = str(contract_path)
    exec(compile(contract_bytes, str(contract_path), "exec"), contract.__dict__)
    return contract, authority_bytes


def build_report(
    repository_root: Path,
    *,
    expected_authority_sha256: str,
    verify_external: bool,
    require_governance_mirrors: bool,
) -> dict:
    repository_root = repository_root.resolve(strict=True)
    authority_path = repository_root / _AUTHORITY_RELATIVE
    contract, authority_bytes = _load_pinned_contract(
        repository_root,
        authority_path,
        expected_authority_sha256,
    )

    source_path = contract.resolve_repository_file(
        repository_root,
        "docs/cach_sana_wam/stage0/SOURCE_MANIFEST.draft.json",
        "source manifest",
    )
    candidate_path = contract.resolve_repository_file(
        repository_root,
        "docs/cach_sana_wam/stage0/CACH_A_CANDIDATE_SPEC.draft.json",
        "candidate spec",
    )
    authority_path = contract.resolve_repository_file(
        repository_root,
        _AUTHORITY_RELATIVE,
        "authority",
    )
    source_bytes = contract.read_regular_bytes(source_path, "source manifest")
    candidate_bytes = contract.read_regular_bytes(candidate_path, "candidate spec")

    source = contract.validate_source_manifest(
        contract.strict_json_bytes(source_bytes, "source manifest")
    )
    candidate = contract.validate_candidate_spec(
        contract.strict_json_bytes(candidate_bytes, "candidate spec")
    )
    authority = contract.validate_authority(
        contract.strict_json_bytes(authority_bytes, "authority")
    )

    config_path = contract.resolve_repository_file(
        repository_root, authority["config"]["path"], "authority config"
    )
    plan_path = contract.resolve_repository_file(
        repository_root, authority["plan"]["path"], "authority plan"
    )
    config_bytes = contract.read_regular_bytes(config_path, "config")
    observed_pin_hashes = {
        "source_manifest": sha256(source_bytes).hexdigest(),
        "candidate_spec": sha256(candidate_bytes).hexdigest(),
        "config": sha256(config_bytes).hexdigest(),
        "plan": contract.file_sha256(plan_path, "plan"),
    }
    for role, observed in observed_pin_hashes.items():
        if observed != authority[role]["sha256"]:
            raise contract.CACHStage0Error(f"authority pin changed: {role}")

    for role, pin in authority["designs"].items():
        path = contract.resolve_repository_file(
            repository_root, pin["path"], f"authority design {role}"
        )
        if contract.file_sha256(path, f"design {role}") != pin["sha256"]:
            raise contract.CACHStage0Error(f"authority design pin changed: {role}")

    if candidate["source_manifest"]["sha256"] != authority["source_manifest"]["sha256"]:
        raise contract.CACHStage0Error("candidate/source authority binding differs")
    if candidate["plan"]["sha256"] != authority["plan"]["sha256"]:
        raise contract.CACHStage0Error("candidate/plan authority binding differs")
    if (
        candidate["architecture"]["layout_spec"]["sha256"]
        != authority["designs"]["layout_bootstrap"]["sha256"]
    ):
        raise contract.CACHStage0Error("candidate/layout authority binding differs")

    # Only consume paths/runtime/dataset/external inputs after the authenticated
    # authority has bound every local source/spec/config/design byte.
    contract.verify_repository_inputs(
        repository_root, source, verify_external=verify_external
    )
    if require_governance_mirrors:
        contract.verify_governance_mirrors(repository_root)

    draft_pin_graph_valid = verify_external and require_governance_mirrors
    return {
        "schema_version": "cach-stage0-static-report-v1",
        "draft_denial_pin_graph_valid": draft_pin_graph_valid,
        "static_graph_valid": False,
        "inspection_complete": False,
        "execution_admission_valid": False,
        "execution_authorized": False,
        "scientific_eligible": False,
        "decision": authority["decision"],
        "blockers": authority["blockers"],
        "verified_external_model_bytes": verify_external,
        "verified_dataset_snapshot_counts": True,
        "verified_config_raw_bytes": True,
        "config_semantic_parser_executed": False,
        "verified_runtime_distribution_subset": True,
        "runtime_inventory_complete": False,
        "verified_root_governance_mirrors": require_governance_mirrors,
        "expected_authority_sha256": expected_authority_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=_ROOT,
    )
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="quick inspection only; report remains static_graph_valid=false",
    )
    parser.add_argument(
        "--skip-governance",
        action="store_true",
        help="quick inspection only; report remains static_graph_valid=false",
    )
    args = parser.parse_args()
    report = build_report(
        args.repository_root.resolve(),
        expected_authority_sha256=args.expected_authority_sha256,
        verify_external=not args.skip_external,
        require_governance_mirrors=not args.skip_governance,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
