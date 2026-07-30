#!/usr/bin/env python3
"""Strict static preflight for one materialized Phase-6 arm config set."""

from __future__ import annotations

import argparse
from pathlib import Path

from sana_wam.train.phase6_arm_config import validate_arm_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="Hash every directly pinned artifact; this includes the 8 GB init checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = validate_arm_directory(
        args.config_dir,
        verify_files=args.verify_files,
    )
    print(f"validated {len(paths)} frozen Phase-6 arm configs")


if __name__ == "__main__":
    main()
