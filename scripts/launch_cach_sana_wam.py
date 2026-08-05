#!/usr/bin/env python3
"""Non-executable CACH launcher skeleton.

Stage 0 intentionally contains no torch import, worker spawn, GPU reservation,
root creation, or training command.  Even a syntactically changed authority
cannot pass this hard stop; Stage 1 must replace it under a new reviewed source
and registered authority.  Its post-start self-check is a review aid, not an
independent executable trust root.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import types
from hashlib import sha256
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_RELATIVE = "src/sana_wam/train/cach_stage0_contract.py"
_SELF_RELATIVE = "scripts/launch_cach_sana_wam.py"
_BOOTSTRAP_LIMIT = 8 * 1024 * 1024


class _BootstrapError(ValueError):
    """Raised before the externally pinned denial contract is loaded."""


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


def _load_pinned_contract(authority_path: Path, expected_authority_sha256: str):
    expected = _bootstrap_expected_sha256(expected_authority_sha256)
    authority_bytes = _bootstrap_read_regular(authority_path, "authority")
    if sha256(authority_bytes).hexdigest() != expected:
        raise _BootstrapError("authority trust anchor differs")
    bootstrap_authority = _bootstrap_json(authority_bytes)

    self_path = _ROOT / _SELF_RELATIVE
    if Path(__file__).resolve(strict=True) != self_path.resolve(strict=True):
        raise _BootstrapError("launcher path differs from repository root")
    self_expected = _bootstrap_pin(
        bootstrap_authority, "candidate_launcher", _SELF_RELATIVE
    )
    self_bytes = _bootstrap_read_regular(self_path, "candidate launcher")
    if sha256(self_bytes).hexdigest() != self_expected:
        raise _BootstrapError("candidate launcher bytes differ from authority pin")

    contract_expected = _bootstrap_pin(
        bootstrap_authority, "stage0_contract", _CONTRACT_RELATIVE
    )
    contract_path = _ROOT / _CONTRACT_RELATIVE
    contract_bytes = _bootstrap_read_regular(contract_path, "Stage-0 contract")
    if sha256(contract_bytes).hexdigest() != contract_expected:
        raise _BootstrapError("Stage-0 contract bytes differ from authority pin")

    contract = types.ModuleType("_cach_stage0_contract_authority_pinned")
    contract.__file__ = str(contract_path)
    exec(compile(contract_bytes, str(contract_path), "exec"), contract.__dict__)
    return contract, authority_bytes


def refuse_launch(authority_path: Path, expected_authority_sha256: str) -> None:
    contract, authority_bytes = _load_pinned_contract(
        authority_path,
        expected_authority_sha256,
    )
    authority = contract.validate_authority(
        contract.strict_json_bytes(
            authority_bytes,
            "CACH Stage 0 authority",
        )
    )
    raise contract.CACHStage0Error(
        "CACH execution is blocked: "
        f"decision={authority['decision']}, blockers={len(authority['blockers'])}; "
        "this Stage 0 skeleton has no execution path"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authority",
        type=Path,
        default=_ROOT / "docs/cach_sana_wam/stage0/CACH_A_AUTHORITY.draft.json",
    )
    parser.add_argument("--expected-authority-sha256", required=True)
    args = parser.parse_args()
    try:
        refuse_launch(
            args.authority,
            args.expected_authority_sha256,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    raise AssertionError("unreachable: Stage 0 launcher must always refuse")


if __name__ == "__main__":
    raise SystemExit(main())
