#!/usr/bin/env python3
"""Reject a reserved CACH Stage-0 YAML before a shell creates state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sana_wam.train.cach_stage0_guard import (  # noqa: E402
    reject_cach_stage0_base_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--entrypoint", required=True)
    args = parser.parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    base_cfg = OmegaConf.load(args.config)
    reject_cach_stage0_base_config(base_cfg, entrypoint=args.entrypoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
