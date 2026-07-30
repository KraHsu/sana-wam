#!/usr/bin/env python
"""Fail-closed comparison for action-only fine-tune checkpoints.

The comparison streams one tensor at a time from each safetensors file so the
11+ GB AR checkpoints do not need to coexist in host memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def compare_checkpoints(
    base_path: str,
    candidate_path: str,
    *,
    trainable_prefixes: tuple[str, ...] = ("action_backbone.",),
    expect: str = "action-only-change",
    max_relative_l2: float | None = None,
) -> dict:
    """Compare two checkpoints and raise if the requested boundary is violated."""
    if expect not in {"identical", "action-only-change"}:
        raise ValueError(f"unsupported expectation: {expect!r}")
    if not trainable_prefixes or any(not prefix for prefix in trainable_prefixes):
        raise ValueError("trainable_prefixes must contain non-empty prefixes")

    changed_trainable: list[str] = []
    changed_frozen: list[str] = []
    shape_or_dtype_mismatch: list[str] = []
    max_abs_delta = 0.0
    trainable_delta_l2_sq = 0.0
    trainable_base_l2_sq = 0.0

    with (
        safe_open(base_path, framework="pt", device="cpu") as base,
        safe_open(candidate_path, framework="pt", device="cpu") as candidate,
    ):
        base_keys = set(base.keys())
        candidate_keys = set(candidate.keys())
        if base_keys != candidate_keys:
            missing = sorted(base_keys - candidate_keys)
            unexpected = sorted(candidate_keys - base_keys)
            raise RuntimeError(
                f"checkpoint key mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )

        for key in sorted(base_keys):
            before = base.get_tensor(key)
            after = candidate.get_tensor(key)
            if before.shape != after.shape or before.dtype != after.dtype:
                shape_or_dtype_mismatch.append(key)
                continue
            is_trainable = any(key.startswith(prefix) for prefix in trainable_prefixes)
            before_float = before.float() if before.is_floating_point() else None
            if is_trainable and before_float is not None:
                trainable_base_l2_sq += float(
                    before_float.double().square().sum().item()
                )
            if torch.equal(before, after):
                continue

            (changed_trainable if is_trainable else changed_frozen).append(key)
            if before_float is not None:
                delta_float = before_float - after.float()
                max_abs_delta = max(
                    max_abs_delta, float(delta_float.abs().max().item())
                )
                if is_trainable:
                    trainable_delta_l2_sq += float(
                        delta_float.double().square().sum().item()
                    )

    if shape_or_dtype_mismatch:
        raise RuntimeError(
            f"checkpoint tensor shape/dtype changed: {shape_or_dtype_mismatch[:8]}"
        )
    if changed_frozen:
        raise RuntimeError(
            "frozen tensors changed outside the trainable allowlist: "
            f"{changed_frozen[:8]}"
        )
    if expect == "identical" and changed_trainable:
        raise RuntimeError(
            f"expected identical checkpoints, but {len(changed_trainable)} trainable tensors changed: "
            f"{changed_trainable[:8]}"
        )
    if expect == "action-only-change" and not changed_trainable:
        raise RuntimeError("no trainable tensor changed after the fine-tune step")

    relative_l2 = (trainable_delta_l2_sq / max(trainable_base_l2_sq, 1e-30)) ** 0.5
    if max_relative_l2 is not None and relative_l2 > max_relative_l2:
        raise RuntimeError(
            f"trainable relative L2 drift {relative_l2:.6g} exceeds limit {max_relative_l2:.6g}"
        )

    return {
        "expectation": expect,
        "trainable_prefixes": list(trainable_prefixes),
        "changed_trainable_tensors": len(changed_trainable),
        "changed_frozen_tensors": 0,
        "max_abs_delta": max_abs_delta,
        "trainable_relative_l2": relative_l2,
        "changed_trainable_sample": changed_trainable[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--trainable-prefix",
        action="append",
        default=None,
        help="Allowed state-dict prefix; repeat for multiple prefixes.",
    )
    parser.add_argument(
        "--expect",
        choices=("identical", "action-only-change"),
        default="action-only-change",
    )
    parser.add_argument(
        "--max-relative-l2",
        type=float,
        default=None,
        help="Optional fail-closed upper bound on ||candidate-base||_2 / ||base||_2.",
    )
    args = parser.parse_args()
    prefixes = tuple(args.trainable_prefix or ["action_backbone."])
    report = compare_checkpoints(
        str(args.base),
        str(args.candidate),
        trainable_prefixes=prefixes,
        expect=args.expect,
        max_relative_l2=args.max_relative_l2,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
