#!/usr/bin/env python3
"""Validate the complete launch DAG and issue tickets; never start a process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wam.train.phase6_launch_manifest import (  # noqa: E402
    Phase6LaunchError,
    issue_phase6_launch_tickets,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        required=True,
        help="Required safety latch: issue immutable tickets but execute no command.",
    )
    args = parser.parse_args()
    try:
        paths = issue_phase6_launch_tickets(
            args.manifest, expected_manifest_sha256=args.manifest_sha256
        )
    except (OSError, Phase6LaunchError) as exc:
        print(f"Phase-6 launch preparation failed: {exc}", file=sys.stderr)
        return 2
    print(f"PASS validate_only=true tickets={len(paths)}")
    for path in paths:
        print(Path(path).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
