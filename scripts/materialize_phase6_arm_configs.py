#!/usr/bin/env python3
"""Render the frozen five-arm config set after real artifacts exist."""

from __future__ import annotations

import argparse
from pathlib import Path

from sana_wam.train.phase6_arm_config import (
    load_phase6_materialization_pins,
    materialize_arm_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pins-manifest", type=Path, required=True)
    parser.add_argument("--pins-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pins = load_phase6_materialization_pins(
        args.pins_manifest, expected_sha256=args.pins_manifest_sha256
    )
    paths = materialize_arm_configs(
        template_dir=args.template_dir,
        output_dir=args.output_dir,
        materialization_pins=pins,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
