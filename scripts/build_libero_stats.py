#!/usr/bin/env python
"""Build sana-wam normalization stats from LIBERO LeRobot metadata only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wam.dataloader.libero_stats import write_libero_stats  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pool meta/stats_gr00t.json from one or more LIBERO LeRobot roots "
            "into sana-wam action_stats.npy; videos and Parquet are not read."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("dataset_roots", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    digest = write_libero_stats(output, args.dataset_roots)
    print(
        json.dumps(
            {
                "output": str(output),
                "schema_version": "sana-wam-libero-stats-build-result-v1",
                "sha256": digest,
                "source_count": len(args.dataset_roots),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
