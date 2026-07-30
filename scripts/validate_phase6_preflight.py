#!/usr/bin/env python3
"""Validate a frozen Phase-6 preflight request without importing torch."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys


_CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _CANDIDATE_ROOT / "src"
if _SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(_SOURCE_ROOT))

from sana_wam.train.phase6_preflight import (  # noqa: E402
    PreflightValidationError,
    canonical_json_bytes,
    load_canonical_preflight_request,
    validate_phase6_preflight,
    write_canonical_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed Phase-6 artifact/source/runtime preflight"
    )
    parser.add_argument("--request", required=True, help="canonical request JSON")
    parser.add_argument(
        "--request-sha256",
        required=True,
        help="lowercase SHA256 pin from the arm launch configuration",
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--report", help="atomic canonical success-report path")
    output.add_argument(
        "--check-only",
        action="store_true",
        help="validate and print the would-be report SHA without writing it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        request = load_canonical_preflight_request(
            args.request, expected_sha256=args.request_sha256
        )
        report = validate_phase6_preflight(request)
        if args.check_only:
            report_sha = sha256(canonical_json_bytes(report)).hexdigest()
            print(f"PASS report_sha256={report_sha}")
        else:
            report_sha = write_canonical_report(args.report, report)
            print(f"PASS report={Path(args.report).resolve()} sha256={report_sha}")
    except (PreflightValidationError, OSError) as exc:
        print(f"Phase-6 preflight failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
