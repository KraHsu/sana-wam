#!/usr/bin/env python3
"""Independently issue one recovery-aware Phase-6 preflight-v6 report."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wam.train.phase6_preflight import (  # noqa: E402
    PreflightValidationError,
    canonical_json_bytes,
    load_canonical_preflight_request,
    run_phase6_preflight,
    validate_phase6_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue or check one real torch-free Phase-6 recovery preflight"
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument(
        "--purpose",
        required=True,
        choices=("reference_precompute", "smoke", "training"),
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--report")
    output.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check_only:
            request = load_canonical_preflight_request(
                args.request, expected_sha256=args.request_sha256
            )
            if request.get("purpose") != args.purpose:
                raise PreflightValidationError("request purpose differs from CLI latch")
            report = validate_phase6_preflight(request)
            digest = sha256(canonical_json_bytes(report)).hexdigest()
            print(f"PASS check_only=true report_sha256={digest}")
        else:
            report = run_phase6_preflight(
                args.request,
                args.report,
                expected_request_sha256=args.request_sha256,
                expected_purpose=args.purpose,
            )
            digest = sha256(canonical_json_bytes(report)).hexdigest()
            print(f"PASS report={Path(args.report).resolve()} sha256={digest}")
    except (OSError, PreflightValidationError, TypeError, ValueError) as exc:
        print(f"Phase-6 preflight failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
