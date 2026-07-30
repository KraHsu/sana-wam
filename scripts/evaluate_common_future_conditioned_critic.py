#!/usr/bin/env python
"""Train-lock an observation-conditioned critic and evaluate the held-out split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from sana_wam.deploy.outcome_ranker import summary_action_features


REPRESENTATIONS = ("action", "prompt", "full")


def _fold(environment_seed: int, count: int = 5) -> int:
    digest = hashlib.sha256(f"common-future-critic-v1|{environment_seed}".encode())
    return int.from_bytes(digest.digest()[:4], "big") % count


def _text_features(prompt: str, size: int = 16) -> np.ndarray:
    values = np.zeros(size, dtype=np.float64)
    for token in re.findall(r"[a-z0-9]+", prompt.lower()):
        digest = hashlib.sha256(f"prompt-token-v1|{token}".encode()).digest()
        index = int.from_bytes(digest[:2], "big") % size
        sign = 1.0 if digest[2] & 1 else -1.0
        values[index] += sign
    norm = np.linalg.norm(values)
    return values if norm == 0 else values / norm


def _image_features(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((16, 12)), dtype=np.float64) / 255.0
    gray = rgb.mean(axis=2)
    dx = np.diff(gray, axis=1, append=gray[:, -1:])
    dy = np.diff(gray, axis=0, append=gray[-1:, :])
    return np.concatenate((rgb.reshape(-1), dx.reshape(-1), dy.reshape(-1)))


def _pca_fit(values: np.ndarray, dimension: int) -> dict:
    mean = values.mean(axis=0)
    centered = values - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    dimension = min(dimension, vt.shape[0], vt.shape[1])
    return {"mean": mean, "components": vt[:dimension]}


def _pca_apply(model: dict, values: np.ndarray) -> np.ndarray:
    return (values - model["mean"]) @ model["components"].T


def _load_groups(plan_path: Path) -> list[dict]:
    plan = json.loads(plan_path.read_text())
    source_manifest_path = Path(plan["source_dataset_manifest"])
    source_manifest = json.loads(source_manifest_path.read_text())
    source_root = source_manifest_path.parent
    source_by_seed = {
        int(group["environment_seed"]): group for group in source_manifest["groups"]
    }
    labels = []
    for replica in plan["replicas"]:
        rows = [
            json.loads(line)
            for line in (
                Path(replica["run_dir"]) / "telemetry/client/episodes.jsonl"
            ).read_text().splitlines()
            if line.strip()
        ]
        labels.append(
            [bool(row["success"]) for row in rows if row.get("event") == "episode_end"]
        )
    labels = np.asarray(labels, dtype=np.bool_).T
    if labels.shape != (30, 5):
        raise ValueError(f"expected [30,5] labels, got {labels.shape}")

    groups = []
    for index, planned in enumerate(plan["groups"]):
        source = source_by_seed[int(planned["environment_seed"])]
        array_path = source_root / source["group_array"]["path"]
        with np.load(array_path, allow_pickle=False) as arrays:
            actions = arrays["predicted_actions_normalized"].copy()
            proprio = arrays["observed_proprio"][0].astype(np.float64, copy=True)
        image_path = source_root / source["candidates"][0]["initial_head_image"]["path"]
        prompt_group = np.zeros(3, dtype=np.float64)
        prompt_group[("left", "right", "unspecified").index(planned["prompt_group"])] = 1
        groups.append(
            {
                "environment_seed": int(planned["environment_seed"]),
                "split": planned["split"],
                "prompt_group": planned["prompt_group"],
                "labels": labels[index],
                "action": summary_action_features(actions),
                "image": _image_features(image_path),
                "proprio": proprio,
                "prompt": np.concatenate(
                    (prompt_group, _text_features(planned["prompt"]))
                ),
            }
        )
    return groups


def _fit_feature_builder(groups: list[dict], representation: str) -> dict:
    action = np.concatenate([group["action"] for group in groups])
    action_mean = action.mean(axis=0)
    action_scale = action.std(axis=0)
    action_scale[action_scale < 1e-8] = 1.0
    action_z = (action - action_mean) / action_scale
    action_pca = _pca_fit(action_z, 12)
    model = {
        "representation": representation,
        "action_mean": action_mean,
        "action_scale": action_scale,
        "action_pca": action_pca,
    }
    if representation == "full":
        images = np.stack([group["image"] for group in groups])
        model["image_pca"] = _pca_fit(images, 6)
        proprio = np.stack([group["proprio"] for group in groups])
        model["proprio_mean"] = proprio.mean(axis=0)
        scale = proprio.std(axis=0)
        scale[scale < 1e-8] = 1.0
        model["proprio_scale"] = scale
    return model


def _candidate_features(group: dict, model: dict) -> np.ndarray:
    action_z = (group["action"] - model["action_mean"]) / model["action_scale"]
    representation = model["representation"]
    if representation == "action":
        return action_z
    action_low = _pca_apply(model["action_pca"], action_z)
    if representation == "prompt":
        state = group["prompt"][:3]
    else:
        image = _pca_apply(model["image_pca"], group["image"][None])[0]
        proprio = (group["proprio"] - model["proprio_mean"]) / model[
            "proprio_scale"
        ]
        state = np.concatenate((image, proprio, group["prompt"]))
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
    interaction = np.einsum("ca,s->cas", action_low, state).reshape(
        action_low.shape[0], -1
    )
    return np.concatenate((action_z, action_low, interaction), axis=1)


def _fit_ranker(groups: list[dict], representation: str) -> dict:
    builder = _fit_feature_builder(groups, representation)
    features = [_candidate_features(group, builder) for group in groups]
    all_features = np.concatenate(features)
    mean = all_features.mean(axis=0)
    scale = all_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    directions = []
    for group, values in zip(groups, features):
        labels = group["labels"]
        if not labels.any() or labels.all():
            continue
        values = (values - mean) / scale
        direction = values[labels].mean(axis=0) - values[~labels].mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > 0:
            directions.append(direction / norm)
    if not directions:
        raise ValueError("training split has no mixed groups")
    weight = np.mean(directions, axis=0)
    weight /= np.linalg.norm(weight)
    return {"builder": builder, "mean": mean, "scale": scale, "weight": weight}


def _score(group: dict, model: dict) -> np.ndarray:
    features = _candidate_features(group, model["builder"])
    return ((features - model["mean"]) / model["scale"]) @ model["weight"]


def _metrics(groups: list[dict], scores: list[np.ndarray]) -> dict:
    comparisons = 0
    correct = 0.0
    top_one = 0
    mixed = 0
    selected_success = 0
    candidate_zero_success = 0
    records = []
    for group, values in zip(groups, scores):
        labels = group["labels"]
        selected = int(np.argmax(values))
        selected_success += int(labels[selected])
        candidate_zero_success += int(labels[0])
        if labels.any() and not labels.all():
            mixed += 1
            top_one += int(labels[selected])
            for positive in values[labels]:
                for negative in values[~labels]:
                    correct += float(positive > negative) + 0.5 * float(
                        positive == negative
                    )
                    comparisons += 1
        records.append(
            {
                "environment_seed": group["environment_seed"],
                "prompt_group": group["prompt_group"],
                "labels": labels.astype(int).tolist(),
                "scores": values.tolist(),
                "selected_candidate": selected,
                "selected_success": bool(labels[selected]),
            }
        )
    return {
        "scenes": len(groups),
        "mixed_scenes": mixed,
        "pairwise_comparisons": comparisons,
        "pairwise_accuracy": None if comparisons == 0 else correct / comparisons,
        "top_one_success_on_mixed": top_one,
        "top_one_rate_on_mixed": None if mixed == 0 else top_one / mixed,
        "selected_successes_all_scenes": selected_success,
        "candidate_zero_successes_all_scenes": candidate_zero_success,
        "records": records,
    }


def _cross_validate(groups: list[dict], representation: str) -> tuple[dict, dict[int, np.ndarray]]:
    predictions = {}
    for fold in range(5):
        train = [group for group in groups if _fold(group["environment_seed"]) != fold]
        test = [group for group in groups if _fold(group["environment_seed"]) == fold]
        if not test:
            continue
        model = _fit_ranker(train, representation)
        for group in test:
            predictions[group["environment_seed"]] = _score(group, model)
    scores = [predictions[group["environment_seed"]] for group in groups]
    return _metrics(groups, scores), predictions


def _calibrate_anchor(groups: list[dict], predictions: dict[int, np.ndarray]) -> dict:
    margins = sorted(
        {
            float(np.max(scores[1:]) - scores[0])
            for scores in predictions.values()
        }
    )
    thresholds = [max(margins, default=0.0) + 1.0] + margins
    choices = []
    for threshold in thresholds:
        successes = regressions = recoveries = switches = 0
        for group in groups:
            scores = predictions[group["environment_seed"]]
            best_nonzero = 1 + int(np.argmax(scores[1:]))
            selected = (
                best_nonzero
                if scores[best_nonzero] - scores[0] > threshold
                else 0
            )
            labels = group["labels"]
            successes += int(labels[selected])
            switches += int(selected != 0)
            regressions += int(labels[0] and not labels[selected])
            recoveries += int(not labels[0] and labels[selected])
        choices.append(
            {
                "threshold": threshold,
                "successes": successes,
                "regressions": regressions,
                "recoveries": recoveries,
                "switches": switches,
            }
        )
    safe = [choice for choice in choices if choice["regressions"] == 0]
    selected = max(
        safe,
        key=lambda choice: (
            choice["successes"],
            choice["recoveries"],
            -choice["switches"],
            choice["threshold"],
        ),
    )
    return {"selected": selected, "candidates": choices}


def _anchor_metrics(groups: list[dict], scores: list[np.ndarray], threshold: float) -> dict:
    successes = regressions = recoveries = switches = 0
    records = []
    for group, values in zip(groups, scores):
        best_nonzero = 1 + int(np.argmax(values[1:]))
        selected = best_nonzero if values[best_nonzero] - values[0] > threshold else 0
        labels = group["labels"]
        successes += int(labels[selected])
        switches += int(selected != 0)
        regressions += int(labels[0] and not labels[selected])
        recoveries += int(not labels[0] and labels[selected])
        records.append(
            {
                "environment_seed": group["environment_seed"],
                "selected_candidate": selected,
                "candidate_zero_success": bool(labels[0]),
                "selected_success": bool(labels[selected]),
                "margin": float(values[best_nonzero] - values[0]),
            }
        )
    return {
        "threshold": threshold,
        "successes": successes,
        "candidate_zero_successes": sum(int(group["labels"][0]) for group in groups),
        "regressions": regressions,
        "recoveries": recoveries,
        "switches": switches,
        "records": records,
    }


def evaluate(plan_path: Path) -> dict:
    groups = _load_groups(plan_path)
    train = [group for group in groups if group["split"] == "train"]
    heldout = [group for group in groups if group["split"] == "heldout"]
    cv = {}
    predictions = {}
    for representation in REPRESENTATIONS:
        metrics, pred = _cross_validate(train, representation)
        cv[representation] = metrics
        predictions[representation] = pred
    locked = max(
        REPRESENTATIONS,
        key=lambda name: (
            cv[name]["top_one_rate_on_mixed"],
            cv[name]["pairwise_accuracy"],
            -REPRESENTATIONS.index(name),
        ),
    )
    calibration = _calibrate_anchor(train, predictions[locked])
    threshold = calibration["selected"]["threshold"]
    final_model = _fit_ranker(train, locked)
    heldout_scores = [_score(group, final_model) for group in heldout]
    heldout_metrics = _metrics(heldout, heldout_scores)
    anchor = _anchor_metrics(heldout, heldout_scores, threshold)
    gate = (
        heldout_metrics["mixed_scenes"] >= 3
        and heldout_metrics["top_one_success_on_mixed"] >= 2
        and anchor["regressions"] == 0
        and anchor["successes"] >= anchor["candidate_zero_successes"] + 1
    )
    return {
        "schema_version": 1,
        "kind": "common_future_observation_conditioned_critic_gate",
        "plan": str(plan_path.resolve()),
        "representations": list(REPRESENTATIONS),
        "cross_validation": cv,
        "locked_representation": locked,
        "anchor_calibration": calibration,
        "heldout": heldout_metrics,
        "heldout_candidate_zero_anchor": anchor,
        "gate_definition": {
            "minimum_mixed_heldout_scenes": 3,
            "minimum_top_one_successes_on_mixed": 2,
            "maximum_anchor_regressions": 0,
            "minimum_anchor_success_gain": 1,
        },
        "gate_pass": gate,
        "status": "GO_ONLINE_PILOT" if gate else "NO_GO_CONDITIONED_CRITIC",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    report = evaluate(args.plan)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "locked_representation": report["locked_representation"],
                "cross_validation": {
                    key: {
                        metric: value[metric]
                        for metric in (
                            "pairwise_accuracy",
                            "top_one_success_on_mixed",
                            "mixed_scenes",
                        )
                    }
                    for key, value in report["cross_validation"].items()
                },
                "heldout": {
                    key: report["heldout"][key]
                    for key in (
                        "pairwise_accuracy",
                        "top_one_success_on_mixed",
                        "mixed_scenes",
                    )
                },
                "heldout_anchor": report["heldout_candidate_zero_anchor"],
                "gate_pass": report["gate_pass"],
                "status": report["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
