#!/usr/bin/env python
"""Fit the final group-balanced summary-action outcome ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from scripts.evaluate_outcome_action_separability import _features
except ModuleNotFoundError:  # Direct script execution.
    from evaluate_outcome_action_separability import _features


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_ranker(dataset_manifest: Path, gate_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    manifest = json.loads(dataset_manifest.read_text())
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "GO_ACTION_ONLY_CRITIC" or gate.get("gate_pass") is not True:
        raise ValueError("action separability gate did not pass")
    dataset_root = dataset_manifest.parent
    groups = []
    for group in manifest["groups"]:
        if not group["mixed"]:
            continue
        with np.load(
            dataset_root / group["group_array"]["path"], allow_pickle=False
        ) as arrays:
            actions = arrays["predicted_actions_normalized"].copy()
            labels = arrays["success"].astype(np.bool_, copy=True)
        groups.append(
            {
                "environment_seed": group["environment_seed"],
                "prompt_group": group["prompt_group"],
                "features": _features(actions, "summary"),
                "labels": labels,
            }
        )
    if len(groups) != 22:
        raise ValueError(f"expected 22 verified mixed groups, got {len(groups)}")
    all_features = np.concatenate([group["features"] for group in groups])
    feature_mean = all_features.mean(axis=0)
    feature_scale = all_features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    directions = []
    for group in groups:
        features = (group["features"] - feature_mean) / feature_scale
        labels = group["labels"]
        direction = features[labels].mean(axis=0) - features[~labels].mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm <= 0:
            raise ValueError(
                f"zero preference direction for seed {group['environment_seed']}"
            )
        directions.append(direction / norm)
    weight = np.mean(directions, axis=0)
    weight_norm = np.linalg.norm(weight)
    if weight_norm <= 0:
        raise ValueError("aggregate preference direction is zero")
    weight /= weight_norm

    comparisons = 0
    correct = 0.0
    group_metrics = []
    for group in groups:
        scores = ((group["features"] - feature_mean) / feature_scale) @ weight
        labels = group["labels"]
        group_correct = 0.0
        group_comparisons = 0
        for positive in scores[labels]:
            for negative in scores[~labels]:
                group_correct += float(positive > negative) + 0.5 * float(
                    positive == negative
                )
                group_comparisons += 1
        comparisons += group_comparisons
        correct += group_correct
        group_metrics.append(
            {
                "environment_seed": group["environment_seed"],
                "prompt_group": group["prompt_group"],
                "pairwise_accuracy": group_correct / group_comparisons,
            }
        )

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "ranker.npz"
    np.savez_compressed(
        artifact_path,
        schema_version=np.asarray(1, dtype=np.int16),
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
        weight=weight.astype(np.float32),
        action_tokens=np.asarray(28, dtype=np.int16),
        action_dim=np.asarray(20, dtype=np.int16),
        feature_dim=np.asarray(weight.shape[0], dtype=np.int16),
    )
    report = {
        "schema_version": 1,
        "kind": "group_balanced_summary_action_outcome_ranker",
        "complete": True,
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "separability_gate": str(gate_path),
        "separability_gate_sha256": _sha256(gate_path),
        "representation": "nine per-action-dimension temporal summaries",
        "training_objective": (
            "mean of L2-normalized within-scene positive-minus-negative "
            "centroid directions"
        ),
        "training_scene_count": len(groups),
        "training_candidate_count": sum(len(group["labels"]) for group in groups),
        "feature_dimension": int(weight.shape[0]),
        "in_sample_pairwise_accuracy": correct / comparisons,
        "cross_validated_pairwise_accuracy": gate["representations"]["summary"][
            "pairwise_accuracy"
        ],
        "cross_validated_permutation_p_value": gate["representations"]["summary"][
            "permutation_p_value"
        ],
        "training_environment_seeds": [
            group["environment_seed"] for group in groups
        ],
        "artifact": {
            "path": str(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
        },
        "group_metrics": group_metrics,
    }
    report_path = output_dir / "ranker_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "ranker_manifest.sha256").write_text(
        f"{_sha256(report_path)}  {report_path}\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("separability_gate", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = train_ranker(
        args.dataset_manifest, args.separability_gate, args.output_dir
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "complete",
                    "training_scene_count",
                    "training_candidate_count",
                    "feature_dimension",
                    "in_sample_pairwise_accuracy",
                    "cross_validated_pairwise_accuracy",
                    "cross_validated_permutation_p_value",
                    "artifact",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())