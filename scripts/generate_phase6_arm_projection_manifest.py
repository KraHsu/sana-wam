#!/usr/bin/env python3
"""Exclusively publish the frozen five-arm scientific config projection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wam.train.phase6_arm_config import (  # noqa: E402
    ARM_NAMES,
    Phase6ArmConfigError,
    write_phase6_arm_projection_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Canonical bundle manifest")
    parser.add_argument(
        "--projection-dir",
        required=True,
        help="Existing repository directory for five raw projection sidecars",
    )
    args = parser.parse_args()
    try:
        projection_dir = Path(args.projection_dir).resolve()
        digests = write_phase6_arm_projection_bundle(
            args.output,
            {
                arm: projection_dir / f"phase6_scientific_projection_{arm}.json"
                for arm in ARM_NAMES
            },
        )
    except (OSError, Phase6ArmConfigError) as exc:
        print(f"Phase-6 arm projection generation failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"PASS output={Path(args.output).resolve()} "
        f"sha256={digests['manifest']} sidecars={len(ARM_NAMES)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
