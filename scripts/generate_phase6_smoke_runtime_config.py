#!/usr/bin/env python3
"""Publish the acyclic five-arm full-config sidecar consumed by the 2B smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wam.train.phase6_arm_config import (  # noqa: E402
    ARM_NAMES,
    Phase6ArmConfigError,
    write_phase6_smoke_runtime_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-manifest", required=True)
    parser.add_argument("--projection-manifest-sha256", required=True)
    parser.add_argument("--template-dir", required=True)
    parser.add_argument("--runtime-config-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_dir = Path(args.runtime_config_dir).resolve()
    try:
        digests = write_phase6_smoke_runtime_bundle(
            bundle_path=args.output,
            projection_manifest_path=str(Path(args.projection_manifest).resolve()),
            projection_manifest_sha256=args.projection_manifest_sha256,
            template_dir=args.template_dir,
            runtime_config_paths={
                arm: output_dir / f"phase6_smoke_runtime_{arm}.json"
                for arm in ARM_NAMES
            },
        )
    except (OSError, Phase6ArmConfigError) as exc:
        print(f"Phase-6 smoke-runtime generation failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"PASS output={Path(args.output).resolve()} "
        f"sha256={digests['bundle']} arm_configs={len(ARM_NAMES)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
