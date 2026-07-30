#!/usr/bin/env python3
"""Exclusively publish the five-arm launch DAG sink without starting training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wam.train.phase6_launch_manifest import (  # noqa: E402
    Phase6LaunchError,
    build_phase6_launch_manifest,
    write_phase6_launch_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--train-script", required=True)
    parser.add_argument("--ticket-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        value = build_phase6_launch_manifest(
            config_dir=args.config_dir,
            repository_root=args.repository_root,
            python_executable=args.python_executable,
            train_script=args.train_script,
            ticket_dir=args.ticket_dir,
            verify_artifact_files=True,
        )
        digest = write_phase6_launch_manifest(args.output, value)
    except (OSError, Phase6LaunchError) as exc:
        print(f"Phase-6 launch-manifest generation failed: {exc}", file=sys.stderr)
        return 2
    print(f"PASS output={Path(args.output).resolve()} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
