#!/usr/bin/env python
"""Compare matched-context SANA distribution scores against a fixed baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="NAME=video_distribution_score.json",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def bootstrap_mean_ci(
    values: np.ndarray, *, rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def representation(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in report["representations"] if item["name"] == name)


def metric(representation_report: dict[str, Any], steps: int) -> dict[str, Any]:
    return next(item for item in representation_report["metrics"] if item["steps"] == steps)


def contexts(
    representation_report: dict[str, Any], steps: int
) -> dict[int, dict[str, Any]]:
    return {
        int(item["episode_index"]): item
        for item in representation_report["context_metrics"][str(steps)]
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Expanded SANA distributional ranking",
        "",
        f"- Baseline: `{report['baseline']}`",
        f"- Selected checkpoint: `{report['selected']}`",
        f"- Contexts: `{report['context_count']}`",
        f"- Seeds per context: `{report['seed_count']}`",
        f"- Selection rule: {report['selection_rule']}",
        "",
    ]
    for representation_report in report["representations"]:
        lines.extend(
            [
                f"## {representation_report['name']}",
                "",
                "| candidate | paired RMSE | corrected mean RMSE | seed RMS | energy | energy delta vs baseline | paired 95% CI |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in representation_report["candidates"]:
            lines.append(
                f"| {row['name']} | {row['paired_rmse']:.6f} | "
                f"{row['monte_carlo_corrected_mean_rmse']:.6f} | "
                f"{row['within_seed_rms']:.6f} | "
                f"{row['energy_score']:.6f} | "
                f"{row['energy_delta_vs_baseline']:+.6f} | "
                f"[{row['energy_delta_ci95_low']:+.6f}, "
                f"{row['energy_delta_ci95_high']:+.6f}] |"
            )
        lines.append("")
    lines.extend(["## Gate", "", "| candidate | result |", "| --- | --- |"])
    for candidate in report["candidates"]:
        lines.append(f"| {candidate['name']} | {candidate['distribution_gate']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    reports = {}
    for value in args.input:
        name, separator, path = value.partition("=")
        if not separator or not name or name in reports:
            raise ValueError(f"invalid or duplicate input: {value!r}")
        reports[name] = json.loads(Path(path).read_text())
    if args.baseline not in reports:
        raise ValueError("baseline is not one of the named inputs")

    baseline_report = reports[args.baseline]
    manifest_hashes = {item["prepared_manifest_sha256"] for item in reports.values()}
    context_counts = {int(item["context_count"]) for item in reports.values()}
    seed_counts = {int(item["seed_count"]) for item in reports.values()}
    if len(manifest_hashes) != 1 or len(context_counts) != 1 or len(seed_counts) != 1:
        raise RuntimeError("candidate reports do not share one evaluation protocol")
    representation_names = [
        item["name"] for item in baseline_report["representations"]
    ]
    rng = np.random.default_rng(args.seed)
    representation_results = []
    candidate_gate_components = {name: [] for name in reports}

    for representation_name in representation_names:
        base_representation = representation(baseline_report, representation_name)
        base_contexts = contexts(base_representation, args.steps)
        candidate_rows = []
        for name, candidate_report in reports.items():
            candidate_representation = representation(
                candidate_report, representation_name
            )
            candidate_metric = metric(candidate_representation, args.steps)
            candidate_contexts = contexts(candidate_representation, args.steps)
            if candidate_contexts.keys() != base_contexts.keys():
                raise RuntimeError(f"context identity mismatch for {name}")
            episodes = sorted(base_contexts)
            differences = np.asarray(
                [
                    candidate_contexts[episode]["energy_score"]
                    - base_contexts[episode]["energy_score"]
                    for episode in episodes
                ],
                dtype=np.float64,
            )
            low, high = bootstrap_mean_ci(
                differences, rng=rng, samples=args.bootstrap_samples
            )
            improved = name == args.baseline or high < 0.0
            candidate_gate_components[name].append(improved)
            candidate_rows.append(
                {
                    "name": name,
                    "paired_rmse": candidate_metric["paired_rmse"],
                    "monte_carlo_corrected_mean_rmse": candidate_metric[
                        "monte_carlo_corrected_mean_rmse"
                    ],
                    "within_seed_rms": candidate_metric["within_seed_rms"],
                    "energy_score": candidate_metric["energy_score"],
                    "energy_delta_vs_baseline": float(differences.mean()),
                    "energy_delta_ci95_low": low,
                    "energy_delta_ci95_high": high,
                }
            )
        representation_results.append(
            {"name": representation_name, "candidates": candidate_rows}
        )

    candidates = []
    passing = []
    for name, components in candidate_gate_components.items():
        passed = all(components)
        result = "BASELINE" if name == args.baseline else ("PASS" if passed else "FAIL")
        candidates.append({"name": name, "distribution_gate": result})
        if passed:
            passing.append(name)
    nonbaseline_passing = [name for name in passing if name != args.baseline]
    if nonbaseline_passing:
        latent = next(
            item for item in representation_results if item["name"] == "latent"
        )
        latent_energy = {
            item["name"]: item["energy_score"] for item in latent["candidates"]
        }
        selected = min(nonbaseline_passing, key=latent_energy.__getitem__)
    else:
        selected = args.baseline

    report = {
        "baseline": args.baseline,
        "selected": selected,
        "selection_rule": (
            "Replace the fixed baseline only when the paired-bootstrap upper "
            "95% bound of the 10-step energy-score difference is below zero "
            "in every registered representation; otherwise retain baseline."
        ),
        "steps": args.steps,
        "context_count": next(iter(context_counts)),
        "seed_count": next(iter(seed_counts)),
        "prepared_manifest_sha256": next(iter(manifest_hashes)),
        "bootstrap_samples": args.bootstrap_samples,
        "representations": representation_results,
        "candidates": candidates,
    }
    args.output_dir.mkdir(parents=True)
    json_path = args.output_dir / "distribution_ranking.json"
    markdown_path = args.output_dir / "distribution_ranking.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
