#!/usr/bin/env python3
"""Emit a review-only CACH Stage-1 contract plan.

This script parses a config and invokes only torch-free validators.  It does
not construct a model, discover checkpoints, create roots, or launch workers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sana_wam.cach.launcher import build_contract_review_plan  # noqa: E402
from sana_wam.cach.verifier import verify_stage1_artifact_set  # noqa: E402

_DEFAULT_CONFIG = _ROOT / "configs/experiments/cach_sana_wam_stage1.yaml"


def _load_mapping(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix not in {".yaml", ".yml"}:
        raise ValueError("Stage-1 config must be JSON or YAML")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ValueError(
            "PyYAML is required to inspect a YAML config; no dependency was installed"
        ) from exc
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid Stage-1 YAML: {exc}") from exc


def build_report(config: Any) -> dict[str, Any]:
    """Build a deterministic non-execution report for programmatic callers."""

    return build_contract_review_plan(config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--expected-authority-sha256")
    parser.add_argument(
        "--skip-external-bundle",
        action="store_true",
        help="verify repository pins but do not read the external source tar",
    )
    args = parser.parse_args()
    try:
        report = build_report(_load_mapping(args.config))
        report["artifact_verification"] = verify_stage1_artifact_set(
            _ROOT,
            expected_authority_sha256=args.expected_authority_sha256,
            verify_external_bundle=not args.skip_external_bundle,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
