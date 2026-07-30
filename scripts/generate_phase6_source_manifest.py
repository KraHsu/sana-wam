#!/usr/bin/env python3
"""Generate the Phase-6 source/template/test inventory after integration freezes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


_CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _CANDIDATE_ROOT / "src"
if _SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(_SOURCE_ROOT))

from sana_wam.train.phase6_preflight import (  # noqa: E402
    PreflightValidationError,
    build_phase6_source_manifest,
    write_phase6_source_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the complete Phase-6 src/scripts Python inventory"
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_phase6_source_manifest(
            str(Path(args.repository_root).resolve())
        )
        digest = write_phase6_source_manifest(args.output, manifest)
        print(
            f"PASS output={Path(args.output).resolve()} "
            f"files={len(manifest['files'])} sha256={digest}"
        )
    except (PreflightValidationError, OSError) as exc:
        print(f"Phase-6 source-manifest generation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
