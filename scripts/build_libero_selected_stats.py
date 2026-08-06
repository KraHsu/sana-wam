#!/usr/bin/env python
"""Build exact exclusion-aware LIBERO stats from one frozen training config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wam.dataloader.libero_selected_stats import (  # noqa: E402
    validate_libero_window_coverage_for_config,
    write_libero_selected_row_stats,
)
from sana_wam.train.libero_contract import (  # noqa: E402
    LIBERO_PRODUCTION_POPULATION_COUNTS,
    validate_libero_production_stats_source_config,
    validate_libero_training_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser()
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError(
            f"LIBERO selected-row config must be a regular file: {config_path}"
        )
    cfg = OmegaConf.load(config_path.resolve())
    validate_libero_training_config(cfg, require_materialized_stats=False)
    validate_libero_production_stats_source_config(cfg)
    validate_libero_window_coverage_for_config(cfg)
    if cfg.training.action_stats_sha256 is not None:
        raise ValueError("selected-row source config must leave stats SHA null")
    if cfg.training.action_stats_population_sha256 is not None:
        raise ValueError("selected-row source config must leave population SHA null")
    output = Path(str(cfg.dataloader.action_stats_path)).expanduser()
    if not output.is_absolute():
        raise ValueError("selected-row stats output must be an absolute config path")
    result = write_libero_selected_row_stats(
        output,
        cfg,
        expected_population_counts=LIBERO_PRODUCTION_POPULATION_COUNTS,
    )
    print(
        json.dumps(
            {
                **result,
                "output": str(output.resolve()),
                "schema_version": "sana-wam-libero-selected-row-build-result-v2",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
