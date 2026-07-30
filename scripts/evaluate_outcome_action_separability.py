#!/usr/bin/env python
"""Evaluate whether first action chunks rank terminal outcomes across scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _features(actions: np.ndarray, representation: str) -> np.ndarray:
    if actions.ndim != 3 or actions.shape[1:] != (28, 20):
        raise ValueError(f"expected [candidate,28,20] actions, got {actions.shape}")
    if representation == "raw":
        return actions.reshape(actions.shape[0], -1).astype(np.float64)
    if representation != "summary":
        raise ValueError(f"unsupported representation: {representation}")
    delta = np.diff(actions, axis=1)
    values = (
        actions[:, 0],
        actions[:, -1],
        actions.mean(axis=1),
        actions.std(axis=1),
        actions.min(axis=1),
        actions.max(axis=1),
        actions[:, -1] - actions[:, 0],
        np.abs(delta).sum(axis=1),
        np.abs(delta).max(axis=1),
    )
    return np.concatenate(values, axis=1).astype(np.float64)


def _fold_assignments(groups: list[dict], fold_count: int = 5) -> dict[int, int]:
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"outcome-separability-v1|{group['environment_seed']}".encode()
        ).hexdigest(),
    )
    return {
        group["group_index"]: index % fold_count
        for index, group in enumerate(ordered)
    }


def _score_fold(
    train_groups: list[dict],
    test_groups: list[dict],
    *,
    label_key: str,
) -> dict[int, np.ndarray]:
    train_features = np.concatenate([group["features"] for group in train_groups])
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    directions = []
    for group in train_groups:
        features = (group["features"] - mean) / scale
        labels = group[label_key]
        direction = features[labels].mean(axis=0) - features[~labels].mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > 0:
            directions.append(direction / norm)
    if not directions:
        raise ValueError("training fold contains no preference directions")
    weight = np.mean(directions, axis=0)
    norm = np.linalg.norm(weight)
    if norm > 0:
        weight /= norm
    return {
        group["group_index"]: ((group["features"] - mean) / scale) @ weight
        for group in test_groups
    }


def _cross_validated_scores(
    groups: list[dict], assignments: dict[int, int], *, label_key: str
) -> dict[int, np.ndarray]:
    predictions = {}
    fold_count = max(assignments.values()) + 1
    for fold in range(fold_count):
        train = [
            group
            for group in groups
            if assignments[group["group_index"]] != fold
        ]
        test = [
            group
            for group in groups
            if assignments[group["group_index"]] == fold
        ]
        predictions.update(_score_fold(train, test, label_key=label_key))
    return predictions


def _ranking_metrics(
    groups: list[dict], predictions: dict[int, np.ndarray], *, label_key: str
) -> dict:
    correct = 0.0
    comparisons = 0
    per_group = []
    subgroup_totals = {
        key: {"correct": 0.0, "comparisons": 0}
        for key in ("left", "right", "unspecified")
    }
    for group in groups:
        labels = group[label_key]
        scores = predictions[group["group_index"]]
        positive_scores = scores[labels]
        negative_scores = scores[~labels]
        group_correct = 0.0
        group_comparisons = 0
        for positive in positive_scores:
            for negative in negative_scores:
                group_correct += float(positive > negative) + 0.5 * float(
                    positive == negative
                )
                group_comparisons += 1
        correct += group_correct
        comparisons += group_comparisons
        accuracy = group_correct / group_comparisons
        per_group.append(
            {
                "environment_seed": group["environment_seed"],
                "prompt_group": group["prompt_group"],
                "positive_count": int(labels.sum()),
                "pairwise_accuracy": accuracy,
                "scores": scores.tolist(),
                "labels": labels.tolist(),
            }
        )
        subgroup = subgroup_totals[group["prompt_group"]]
        subgroup["correct"] += group_correct
        subgroup["comparisons"] += group_comparisons
    subgroup_metrics = {}
    for name, values in subgroup_totals.items():
        subgroup_metrics[name] = {
            "comparisons": values["comparisons"],
            "pairwise_accuracy": (
                values["correct"] / values["comparisons"]
                if values["comparisons"]
                else None
            ),
        }
    return {
        "pairwise_accuracy": correct / comparisons,
        "pairwise_comparisons": comparisons,
        "mean_group_accuracy": float(
            np.mean([group["pairwise_accuracy"] for group in per_group])
        ),
        "prompt_subgroups": subgroup_metrics,
        "groups": per_group,
    }


def evaluate(
    dataset_manifest: Path,
    output_path: Path,
    *,
    permutations: int,
    seed: int,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite output: {output_path}")
    manifest = json.loads(dataset_manifest.read_text())
    if manifest.get("complete") is not True or manifest.get("group_count") != 30:
        raise ValueError("invalid or incomplete outcome preference dataset")
    dataset_root = dataset_manifest.parent
    mixed_manifest_groups = [group for group in manifest["groups"] if group["mixed"]]
    groups_by_representation = {"summary": [], "raw": []}
    for group in mixed_manifest_groups:
        array_path = dataset_root / group["group_array"]["path"]
        with np.load(array_path, allow_pickle=False) as arrays:
            actions = arrays["predicted_actions_normalized"].copy()
            labels = arrays["success"].astype(np.bool_, copy=True)
        if labels.shape != (5,) or not labels.any() or labels.all():
            raise ValueError(f"invalid mixed labels for group {group['group_index']}")
        for representation in groups_by_representation:
            groups_by_representation[representation].append(
                {
                    "group_index": group["group_index"],
                    "environment_seed": group["environment_seed"],
                    "prompt_group": group["prompt_group"],
                    "features": _features(actions, representation),
                    "labels": labels,
                }
            )
    assignments = _fold_assignments(groups_by_representation["summary"])
    rng = np.random.default_rng(seed)
    reports = {}
    for representation, groups in groups_by_representation.items():
        observed_predictions = _cross_validated_scores(
            groups, assignments, label_key="labels"
        )
        metrics = _ranking_metrics(
            groups, observed_predictions, label_key="labels"
        )
        null_accuracies = []
        for _ in range(permutations):
            for group in groups:
                group["permuted_labels"] = rng.permutation(group["labels"])
            null_predictions = _cross_validated_scores(
                groups, assignments, label_key="permuted_labels"
            )
            null_metrics = _ranking_metrics(
                groups, null_predictions, label_key="permuted_labels"
            )
            null_accuracies.append(null_metrics["pairwise_accuracy"])
        observed = metrics["pairwise_accuracy"]
        p_value = (1 + sum(value >= observed for value in null_accuracies)) / (
            permutations + 1
        )
        reports[representation] = {
            **metrics,
            "permutation_count": permutations,
            "permutation_p_value": p_value,
            "null_mean": float(np.mean(null_accuracies)),
            "null_std": float(np.std(null_accuracies)),
            "feature_dimension": groups[0]["features"].shape[1],
        }
    primary = reports["summary"]
    gate_pass = (
        primary["pairwise_accuracy"] >= 0.60
        and primary["permutation_p_value"] <= 0.05
    )
    report = {
        "schema_version": 1,
        "kind": "generation_zero_action_outcome_separability_gate",
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": hashlib.sha256(
            dataset_manifest.read_bytes()
        ).hexdigest(),
        "mixed_groups": len(mixed_manifest_groups),
        "cross_validation": "deterministic five-fold scene-disjoint",
        "ranker": "group-balanced standardized centroid preference direction",
        "primary_representation": "summary",
        "gate_definition": {
            "minimum_pairwise_accuracy": 0.60,
            "maximum_permutation_p_value": 0.05,
        },
        "gate_pass": gate_pass,
        "status": "GO_ACTION_ONLY_CRITIC" if gate_pass else "NO_GO_ACTION_ONLY_CRITIC",
        "representations": reports,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    report = evaluate(
        args.dataset_manifest,
        args.output,
        permutations=args.permutations,
        seed=args.seed,
    )
    compact = {
        "status": report["status"],
        "gate_pass": report["gate_pass"],
        "mixed_groups": report["mixed_groups"],
        "summary": {
            key: report["representations"]["summary"][key]
            for key in (
                "pairwise_accuracy",
                "mean_group_accuracy",
                "permutation_p_value",
                "null_mean",
                "null_std",
                "prompt_subgroups",
            )
        },
        "raw": {
            key: report["representations"]["raw"][key]
            for key in (
                "pairwise_accuracy",
                "mean_group_accuracy",
                "permutation_p_value",
                "null_mean",
                "null_std",
                "prompt_subgroups",
            )
        },
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())