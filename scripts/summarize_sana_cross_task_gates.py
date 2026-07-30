#!/usr/bin/env python
"""Summarize paired SANA gates across a fixed cohort of RoboTwin tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPRESENTATIONS = ("latent", "fixed_coarse_motion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated task names in the registered cohort order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument(
        "--action-margin",
        type=float,
        default=None,
        help=(
            "Optional pre-registered selected-minus-Phase1 action-MSE harm "
            "margin. Without it, non-inferiority is reported as not tested."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rng(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    derived = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(derived)


def finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap(
    values: np.ndarray, *, seed: int, label: str, samples: int
) -> tuple[float, float]:
    rng = stable_rng(seed, label)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return interval(values[indices].mean(axis=1))


def task_bootstrap(
    values_by_task: list[np.ndarray], *, seed: int, label: str, samples: int
) -> dict[str, Any]:
    task_means = np.asarray([values.mean() for values in values_by_task])
    rng = stable_rng(seed, f"{label}:task")
    indices = rng.integers(
        0, len(task_means), size=(samples, len(task_means))
    )
    task_low, task_high = interval(task_means[indices].mean(axis=1))

    flat = np.concatenate(values_by_task)
    rng = stable_rng(seed, f"{label}:pooled-context")
    indices = rng.integers(0, len(flat), size=(samples, len(flat)))
    context_low, context_high = interval(flat[indices].mean(axis=1))

    task_count = len(values_by_task)
    rng = stable_rng(seed, f"{label}:hierarchical")
    selected_tasks = rng.integers(0, task_count, size=(samples, task_count))
    within_means = np.empty((samples, task_count, task_count), dtype=np.float64)
    for slot in range(task_count):
        for task_index, values in enumerate(values_by_task):
            episode_indices = rng.integers(
                0, len(values), size=(samples, len(values))
            )
            within_means[:, slot, task_index] = values[episode_indices].mean(axis=1)
    sample_index = np.arange(samples)[:, None]
    slot_index = np.arange(task_count)[None, :]
    hierarchical_means = within_means[
        sample_index, slot_index, selected_tasks
    ].mean(axis=1)
    hierarchical_low, hierarchical_high = interval(hierarchical_means)

    return {
        "equal_task_mean": float(task_means.mean()),
        "task_bootstrap_ci95_low": task_low,
        "task_bootstrap_ci95_high": task_high,
        "hierarchical_task_episode_ci95_low": hierarchical_low,
        "hierarchical_task_episode_ci95_high": hierarchical_high,
        "pooled_context_ci95_low": context_low,
        "pooled_context_ci95_high": context_high,
        "task_means": task_means.tolist(),
    }


def representation(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in report["representations"] if item["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"representation {name!r} is missing or duplicated")
    return matches[0]


def metric(report: dict[str, Any], steps: int) -> dict[str, Any]:
    matches = [item for item in report["metrics"] if item["steps"] == steps]
    if len(matches) != 1:
        raise RuntimeError(f"step {steps} metric is missing or duplicated")
    return matches[0]


def distribution_contexts(
    report: dict[str, Any], steps: int
) -> dict[int, dict[str, Any]]:
    rows = report["context_metrics"][str(steps)]
    contexts = {int(item["episode_index"]): item for item in rows}
    if len(contexts) != len(rows):
        raise RuntimeError("duplicate episode identity in distribution report")
    return contexts


def validate_protocol_pair(
    control: dict[str, Any], treatment: dict[str, Any], manifest_hash: str
) -> None:
    fields = (
        "prepared_dir",
        "prepared_manifest_sha256",
        "pair_count",
        "context_count",
        "seed_count",
        "chunk",
        "step_counts",
        "seed",
    )
    for field in fields:
        if control[field] != treatment[field]:
            raise RuntimeError(f"control/treatment protocol mismatch: {field}")
    if control["prepared_manifest_sha256"] != manifest_hash:
        raise RuntimeError("score report does not match the current manifest")


def summarize_distribution(
    root: Path,
    tasks: list[str],
    manifests: dict[str, dict[str, Any]],
    manifest_hashes: dict[str, str],
    *,
    steps: int,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    reports = {}
    for task in tasks:
        control = load_json(
            root / f"score_phase1_{task}" / "video_distribution_score.json"
        )
        treatment = load_json(
            root / f"score_mixed_{task}" / "video_distribution_score.json"
        )
        validate_protocol_pair(control, treatment, manifest_hashes[task])
        if int(control["context_count"]) != int(manifests[task]["context_count"]):
            raise RuntimeError(f"manifest/score context mismatch for {task}")
        reports[task] = (control, treatment)

    representation_reports = []
    for representation_name in REPRESENTATIONS:
        task_rows = []
        values_by_task = []
        for task in tasks:
            control, treatment = reports[task]
            control_rep = representation(control, representation_name)
            treatment_rep = representation(treatment, representation_name)
            control_metric = metric(control_rep, steps)
            treatment_metric = metric(treatment_rep, steps)
            control_contexts = distribution_contexts(control_rep, steps)
            treatment_contexts = distribution_contexts(treatment_rep, steps)
            if control_contexts.keys() != treatment_contexts.keys():
                raise RuntimeError(f"distribution identities differ for {task}")
            expected = set(int(value) for value in manifests[task]["episode_indices"])
            if set(control_contexts) != expected:
                raise RuntimeError(f"distribution episodes differ from {task} manifest")
            differences = np.asarray(
                [
                    treatment_contexts[episode]["energy_score"]
                    - control_contexts[episode]["energy_score"]
                    for episode in sorted(control_contexts)
                ],
                dtype=np.float64,
            )
            low, high = paired_bootstrap(
                differences,
                seed=seed,
                label=f"distribution:{representation_name}:{task}",
                samples=samples,
            )
            values_by_task.append(differences)
            task_rows.append(
                {
                    "task": task,
                    "phase1_energy_score": control_metric["energy_score"],
                    "mixed_energy_score": treatment_metric["energy_score"],
                    "energy_delta": float(differences.mean()),
                    "energy_delta_ci95_low": low,
                    "energy_delta_ci95_high": high,
                    "result": (
                        "IMPROVED"
                        if high < 0.0
                        else ("DEGRADED" if low > 0.0 else "INCONCLUSIVE")
                    ),
                    "phase1_corrected_mean_rmse": control_metric[
                        "monte_carlo_corrected_mean_rmse"
                    ],
                    "mixed_corrected_mean_rmse": treatment_metric[
                        "monte_carlo_corrected_mean_rmse"
                    ],
                    "corrected_mean_rmse_delta": treatment_metric[
                        "monte_carlo_corrected_mean_rmse"
                    ]
                    - control_metric["monte_carlo_corrected_mean_rmse"],
                    "phase1_seed_rms": control_metric["within_seed_rms"],
                    "mixed_seed_rms": treatment_metric["within_seed_rms"],
                }
            )
        aggregate = task_bootstrap(
            values_by_task,
            seed=seed,
            label=f"distribution:{representation_name}",
            samples=samples,
        )
        aggregate["average_improvement"] = (
            aggregate["task_bootstrap_ci95_high"] < 0.0
        )
        aggregate["all_tasks_improved"] = all(
            row["energy_delta_ci95_high"] < 0.0 for row in task_rows
        )
        representation_reports.append(
            {"name": representation_name, "tasks": task_rows, "aggregate": aggregate}
        )
    return {
        "steps": steps,
        "representations": representation_reports,
        "average_gate": all(
            item["aggregate"]["average_improvement"]
            for item in representation_reports
        ),
        "universal_task_gate": all(
            item["aggregate"]["all_tasks_improved"]
            for item in representation_reports
        ),
    }


def action_context_differences(
    report: dict[str, Any], sigma: float
) -> np.ndarray:
    base = report["context_metrics"]["phase1_video"][str(sigma)]
    selected = report["context_metrics"]["selected_video"][str(sigma)]
    if base.keys() != selected.keys():
        raise RuntimeError("action control/treatment context identities differ")
    episodes = sorted(base, key=int)
    return np.asarray(
        [float(selected[episode]) - float(base[episode]) for episode in episodes]
    )


def summarize_action(
    root: Path,
    tasks: list[str],
    manifests: dict[str, dict[str, Any]],
    *,
    seed: int,
    samples: int,
    margin: float | None,
) -> dict[str, Any]:
    reports = {}
    sigmas = None
    for task in tasks:
        report = load_json(
            root / f"action_gate_mixed_{task}" / "action_cache_gate.json"
        )
        if int(report["context_count"]) != int(manifests[task]["context_count"]):
            raise RuntimeError(f"action/manifest context mismatch for {task}")
        current_sigmas = [float(value) for value in report["sigmas"]]
        if sigmas is None:
            sigmas = current_sigmas
        elif sigmas != current_sigmas:
            raise RuntimeError("action sigma protocols differ across tasks")
        reports[task] = report
    assert sigmas is not None

    task_rows = []
    arrays_by_sigma: dict[float, list[np.ndarray]] = {sigma: [] for sigma in sigmas}
    integrated_arrays = []
    detected_harm = []
    for task in tasks:
        report = reports[task]
        metrics_by_sigma = {float(row["sigma"]): row for row in report["metrics"]}
        sigma_rows = []
        task_arrays = []
        for sigma in sigmas:
            row = metrics_by_sigma[sigma]
            differences = action_context_differences(report, sigma)
            if not math.isclose(
                float(differences.mean()), float(row["mse_delta"]), abs_tol=1e-12
            ):
                raise RuntimeError(f"action context aggregate mismatch for {task}")
            arrays_by_sigma[sigma].append(differences)
            task_arrays.append(differences)
            result = (
                "IMPROVED"
                if row["mse_delta_ci95_high"] < 0.0
                else (
                    "DEGRADED"
                    if row["mse_delta_ci95_low"] > 0.0
                    else "INCONCLUSIVE"
                )
            )
            if result == "DEGRADED":
                detected_harm.append({"task": task, "sigma": sigma})
            sigma_rows.append(
                {
                    "sigma": sigma,
                    "phase1_rmse": row["phase1_rmse"],
                    "mixed_rmse": row["selected_rmse"],
                    "mse_delta": row["mse_delta"],
                    "mse_delta_ci95_low": row["mse_delta_ci95_low"],
                    "mse_delta_ci95_high": row["mse_delta_ci95_high"],
                    "result": result,
                    "margin_result": (
                        "NOT_TESTED"
                        if margin is None
                        else (
                            "WITHIN_MARGIN"
                            if row["mse_delta_ci95_high"] < margin
                            else "EXCEEDS_MARGIN"
                        )
                    ),
                }
            )
        integrated = np.stack(task_arrays).mean(axis=0)
        integrated_arrays.append(integrated)
        task_rows.append(
            {
                "task": task,
                "sigmas": sigma_rows,
                "integrated_mse_delta": report["integrated"]["mse_delta"],
                "integrated_ci95_low": report["integrated"][
                    "mse_delta_ci95_low"
                ],
                "integrated_ci95_high": report["integrated"][
                    "mse_delta_ci95_high"
                ],
            }
        )

    aggregate = {}
    for sigma in sigmas:
        aggregate[str(sigma)] = task_bootstrap(
            arrays_by_sigma[sigma],
            seed=seed,
            label=f"action:{sigma}",
            samples=samples,
        )
    aggregate["integrated"] = task_bootstrap(
        integrated_arrays,
        seed=seed,
        label="action:integrated",
        samples=samples,
    )
    return {
        "sigmas": sigmas,
        "tasks": task_rows,
        "aggregate": aggregate,
        "detected_harm": detected_harm,
        "detected_harm_gate": "FAIL" if detected_harm else "NO_DETECTED_HARM",
        "noninferiority": (
            "NOT_TESTED_NO_PREREGISTERED_MARGIN"
            if margin is None
            else (
                "PASS"
                if all(
                    row["margin_result"] == "WITHIN_MARGIN"
                    for task in task_rows
                    for row in task["sigmas"]
                )
                else "FAIL"
            )
        ),
        "action_margin": margin,
    }


def solver_step_row(report: dict[str, Any], steps: int) -> dict[str, Any]:
    matches = [row for row in report["solutions"] if int(row["steps"]) == steps]
    if len(matches) != 1:
        raise RuntimeError(f"solver step {steps} is missing or duplicated")
    return matches[0]


def solver_adjacent_row(
    report: dict[str, Any], coarse: int, fine_steps: int
) -> dict[str, Any]:
    matches = [
        row
        for row in report["adjacent_deltas"]
        if int(row["coarse_steps"]) == coarse
        and int(row["fine_steps"]) == fine_steps
    ]
    if len(matches) != 1:
        raise RuntimeError(f"solver adjacent delta {coarse}->{fine_steps} missing")
    return matches[0]


def summarize_solver(
    root: Path,
    tasks: list[str],
    manifests: dict[str, dict[str, Any]],
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    task_rows = []
    for task in tasks:
        report = load_json(
            root
            / f"solver100_context_mixed_{task}"
            / "video_solver_convergence.json"
        )
        if report.get("task_name") != task:
            raise RuntimeError(f"solver task identity mismatch for {task}")
        if report.get("prepared_manifest_sha256") != manifest_hashes[task]:
            raise RuntimeError(f"solver manifest mismatch for {task}")
        if int(report["context_count"]) != int(manifests[task]["context_count"]):
            raise RuntimeError(f"solver context mismatch for {task}")
        if [int(value) for value in report["step_counts"]] != [20, 50, 100]:
            raise RuntimeError(f"unexpected solver schedule for {task}")
        d20_100 = float(solver_step_row(report, 20)["delta_from_finest_rms"])
        d50_100 = float(solver_step_row(report, 50)["delta_from_finest_rms"])
        d20_50 = float(solver_adjacent_row(report, 20, 50)["output_delta_rms"])
        d50_100_adjacent = float(
            solver_adjacent_row(report, 50, 100)["output_delta_rms"]
        )
        context_rows = report["context_metrics"]
        expected = set(int(value) for value in manifests[task]["episode_indices"])
        observed = {int(row["episode_index"]) for row in context_rows}
        if observed != expected or len(observed) != len(context_rows):
            raise RuntimeError(f"solver context identities differ for {task}")
        context_d50 = np.asarray(
            [
                float(solver_step_row(row, 50)["delta_from_finest_rms"])
                for row in context_rows
            ]
        )
        context_d20_50 = np.asarray(
            [
                float(solver_adjacent_row(row, 20, 50)["output_delta_rms"])
                for row in context_rows
            ]
        )
        context_d50_100 = np.asarray(
            [
                float(solver_adjacent_row(row, 50, 100)["output_delta_rms"])
                for row in context_rows
            ]
        )
        all_values = [d20_100, d50_100, d20_50, d50_100_adjacent]
        all_values.extend(context_d20_50.tolist())
        all_values.extend(context_d50_100.tolist())
        worst_index = int(np.argmax(context_d50))
        aggregate_refines = d20_100 > d50_100 > 0.0
        aggregate_adjacent_shrinks = d20_50 > d50_100_adjacent > 0.0
        task_rows.append(
            {
                "task": task,
                "delta_20_to_100": d20_100,
                "delta_50_to_100": d50_100,
                "delta_20_to_50": d20_50,
                "aggregate_refines_toward_100": aggregate_refines,
                "aggregate_adjacent_delta_shrinks": aggregate_adjacent_shrinks,
                "context_refinement_fraction": float(
                    np.mean(
                        [
                            bool(row["distance_to_finest_strictly_decreases"])
                            for row in context_rows
                        ]
                    )
                ),
                "context_adjacent_shrink_fraction": float(
                    np.mean(
                        [
                            bool(row["adjacent_delta_strictly_decreases"])
                            for row in context_rows
                        ]
                    )
                ),
                "delta_50_to_100_p50": float(np.quantile(context_d50, 0.50)),
                "delta_50_to_100_p90": float(np.quantile(context_d50, 0.90)),
                "delta_50_to_100_p95": float(np.quantile(context_d50, 0.95)),
                "delta_50_to_100_max": float(context_d50[worst_index]),
                "worst_episode_index": int(
                    context_rows[worst_index]["episode_index"]
                ),
                "all_finite": finite(all_values)
                and all(bool(row["all_finite"]) for row in context_rows),
            }
        )
    return {
        "tasks": task_rows,
        "aggregate_gate": all(
            row["all_finite"]
            and row["aggregate_refines_toward_100"]
            and row["aggregate_adjacent_delta_shrinks"]
            for row in task_rows
        ),
        "all_contexts_refine": all(
            row["context_refinement_fraction"] == 1.0
            and row["context_adjacent_shrink_fraction"] == 1.0
            for row in task_rows
        ),
    }


def summarize_attention(root: Path, tasks: list[str]) -> dict[str, Any]:
    task_rows = []
    for task in tasks:
        report = load_json(
            root / f"attention_spot_mixed_{task}" / "ar_attention_probe.json"
        )
        query_rows = report["query_group_records"]
        if not query_rows:
            raise RuntimeError(f"attention capture is empty for {task}")
        tracked_fields = (
            "denominator_zero_fraction",
            "denominator_below_1e_6_fraction",
            "denominator_min",
            "output_abs_max",
            "output_rms",
        )
        values = [float(row[field]) for row in query_rows for field in tracked_fields]
        equivalence = report["output_equivalence"]
        row = {
            "task": task,
            "pair_count": int(report["pair_count"]),
            "max_zero_denominator_fraction": max(
                float(item["denominator_zero_fraction"]) for item in query_rows
            ),
            "max_below_1e_6_denominator_fraction": max(
                float(item["denominator_below_1e_6_fraction"])
                for item in query_rows
            ),
            "minimum_denominator": min(
                float(item["denominator_min"]) for item in query_rows
            ),
            "maximum_attention_abs": max(
                float(item["output_abs_max"]) for item in query_rows
            ),
            "maximum_attention_rms": max(
                float(item["output_rms"]) for item in query_rows
            ),
            "output_equivalence": equivalence["status"],
            "action_max_abs_delta": float(equivalence["action_max_abs_delta"]),
            "all_finite": finite(values),
        }
        row["result"] = (
            "PASS"
            if row["all_finite"]
            and row["max_zero_denominator_fraction"] == 0.0
            and row["max_below_1e_6_denominator_fraction"] == 0.0
            and row["minimum_denominator"] > 0.0
            and row["output_equivalence"] == "PASS"
            and row["action_max_abs_delta"] == 0.0
            else "FAIL"
        )
        task_rows.append(row)
    return {
        "scope": "one prepared pair (two contexts) per task",
        "tasks": task_rows,
        "spot_gate": all(row["result"] == "PASS" for row in task_rows),
    }


def summarize_normalization(root: Path, tasks: list[str]) -> dict[str, Any]:
    report = load_json(root / "normalization_support_audit.json")
    rows_by_task = {row["task"]: row for row in report["tasks"]}
    expected = ["adjust_bottle", *tasks]
    if any(task not in rows_by_task for task in expected):
        raise RuntimeError("normalization audit does not cover the registered tasks")
    rows = []
    for task in expected:
        source = rows_by_task[task]
        rows.append(
            {
                "task": task,
                "action_outside_element_fraction": source["actions"][
                    "outside_element_fraction"
                ],
                "action_outside_row_fraction": source["actions"][
                    "outside_row_fraction"
                ],
                "action_max_abs": source["actions"]["max_abs"],
                "proprio_outside_element_fraction": source["proprio"][
                    "outside_element_fraction"
                ],
                "proprio_outside_row_fraction": source["proprio"][
                    "outside_row_fraction"
                ],
                "proprio_max_abs": source["proprio"]["max_abs"],
            }
        )
    return {"support": "[-1, 1] from adjust_bottle action stats", "tasks": rows}


def summarize_protocol(root: Path, tasks: list[str]) -> dict[str, Any]:
    cohort = load_json(root / "task_cohort_manifest.json")
    checkpoint = load_json(root / "checkpoint_protocol_diff.json")
    if cohort["registered_sorted_order"] != tasks:
        raise RuntimeError("registered task cohort differs from the requested tasks")
    if checkpoint["unexpected_changed_keys"]:
        raise RuntimeError("checkpoint contains changes outside the allowed groups")
    if checkpoint["action_contract"]["changed_element_count"] != 0:
        raise RuntimeError("action weights are not frozen")
    if checkpoint["proprio_action_contract"]["changed_element_count"] != 0:
        raise RuntimeError("proprio-action weights are not frozen")
    if checkpoint["config_sha256"] != checkpoint["mixed_config_sha256"]:
        raise RuntimeError("control and treatment configs differ")
    if checkpoint["action_stats_sha256"] != checkpoint["mixed_action_stats_sha256"]:
        raise RuntimeError("control and treatment action stats differ")
    return {"task_cohort": cohort, "checkpoint_diff": checkpoint}


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cross-task SANA distributional gates",
        "",
        "This report separates average cross-task evidence from universal "
        "per-task claims. Lower energy and MSE are better.",
        "",
        "## Verdict",
        "",
        "| Claim | Result |",
        "| --- | --- |",
    ]
    for name, result in report["verdict"].items():
        lines.append(f"| {name.replace('_', ' ')} | {result} |")

    protocol = report["protocol"]
    checkpoint = protocol["checkpoint_diff"]
    cohort = protocol["task_cohort"]
    lines.extend(
        [
            "",
            "## Registered protocol",
            "",
            f"- Task candidates: `{cohort['candidate_count']}`; list SHA256 "
            f"`{cohort['candidate_list_sha256']}`; NumPy `{cohort['numpy_version']}`; "
            f"seed `{cohort['seed']}`.",
            f"- Phase1/control SHA256: `{checkpoint['base_checkpoint_sha256']}`.",
            f"- Mixed/treatment SHA256: `{checkpoint['mixed_checkpoint_sha256']}`.",
            f"- Changed tensors: `{checkpoint['changed_tensor_count']}`; all changes "
            "fall inside the three registered video groups.",
            f"- Frozen action tensors: `{checkpoint['action_contract']['tensor_count']}`; "
            f"frozen proprio-action tensors: "
            f"`{checkpoint['proprio_action_contract']['tensor_count']}`.",
            f"- Shared config SHA256: `{checkpoint['config_sha256']}`; shared action "
            f"stats SHA256: `{checkpoint['action_stats_sha256']}`.",
        ]
    )

    lines.extend(["", "## Distribution scores", ""])
    for representation_report in report["distribution"]["representations"]:
        lines.extend(
            [
                f"### {representation_report['name']}",
                "",
                "| Task | Phase1 energy | Mixed energy | Delta | 95% CI | Result |",
                "| --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in representation_report["tasks"]:
            lines.append(
                f"| {row['task']} | {row['phase1_energy_score']:.6f} | "
                f"{row['mixed_energy_score']:.6f} | {row['energy_delta']:+.6f} | "
                f"[{row['energy_delta_ci95_low']:+.6f}, "
                f"{row['energy_delta_ci95_high']:+.6f}] | {row['result']} |"
            )
        aggregate = representation_report["aggregate"]
        lines.extend(
            [
                "",
                f"Equal-task mean delta: `{aggregate['equal_task_mean']:+.6f}`; "
                f"task-bootstrap 95% CI "
                f"`[{aggregate['task_bootstrap_ci95_low']:+.6f}, "
                f"{aggregate['task_bootstrap_ci95_high']:+.6f}]`; hierarchical "
                f"task-episode CI "
                f"`[{aggregate['hierarchical_task_episode_ci95_low']:+.6f}, "
                f"{aggregate['hierarchical_task_episode_ci95_high']:+.6f}]`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Cache-teacher action path",
            "",
            "A positive CI wholly above zero is detected functional harm. A "
            "non-inferiority claim is not made without a pre-registered margin.",
            "",
            "| Task | Sigma | Phase1 RMSE | Mixed RMSE | MSE delta | 95% CI | Result |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for task in report["action"]["tasks"]:
        for row in task["sigmas"]:
            lines.append(
                f"| {task['task']} | {row['sigma']:.1f} | "
                f"{row['phase1_rmse']:.6f} | {row['mixed_rmse']:.6f} | "
                f"{row['mse_delta']:+.6f} | "
                f"[{row['mse_delta_ci95_low']:+.6f}, "
                f"{row['mse_delta_ci95_high']:+.6f}] | {row['result']} |"
            )

    lines.extend(
        [
            "",
            "## Same-seed solver refinement",
            "",
            "The 100-step solution is a finer numerical reference, not ground truth.",
            "",
            "| Task | 20->100 | 50->100 | 20->50 | Context refine | Context adjacent shrink | Result |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["solver"]["tasks"]:
        result = (
            "PASS"
            if row["all_finite"]
            and row["aggregate_refines_toward_100"]
            and row["aggregate_adjacent_delta_shrinks"]
            else "FAIL"
        )
        lines.append(
            f"| {row['task']} | {row['delta_20_to_100']:.6f} | "
            f"{row['delta_50_to_100']:.6f} | {row['delta_20_to_50']:.6f} | "
            f"{row['context_refinement_fraction']:.3f} | "
            f"{row['context_adjacent_shrink_fraction']:.3f} | {result} |"
        )

    lines.extend(
        [
            "",
            "## Attention stability spot-check",
            "",
            "One prepared pair (two contexts) is captured per task; this is not "
            "a full-context stability proof.",
            "",
            "| Task | Min denominator | Max attention abs | Max attention RMS | Equivalence | Result |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report["attention"]["tasks"]:
        lines.append(
            f"| {row['task']} | {row['minimum_denominator']:.6g} | "
            f"{row['maximum_attention_abs']:.6g} | "
            f"{row['maximum_attention_rms']:.6g} | "
            f"{row['output_equivalence']} | {row['result']} |"
        )

    lines.extend(
        [
            "",
            "## Fixed-normalization support audit",
            "",
            "| Task | Action outside elements | Action outside rows | Action max abs | Proprio outside elements |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["normalization"]["tasks"]:
        lines.append(
            f"| {row['task']} | "
            f"{row['action_outside_element_fraction']:.3%} | "
            f"{row['action_outside_row_fraction']:.3%} | "
            f"{row['action_max_abs']:.3f} | "
            f"{row['proprio_outside_element_fraction']:.3%} |"
        )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- The primary statistic treats task as the cross-task inference unit; "
            "the pooled-context interval is retained only as a conditional diagnostic.",
            "- Every score uses expert history and chunk 1. This is a teacher-forced "
            "first-continuation stress test, not an autoregressive rollout result.",
            "- The fixed action normalization remains the checkpoint's adjust_bottle "
            "deployment contract; results are not native per-task action calibration.",
            "- Attention is only a one-pair-per-task spot-check; no closed-loop "
            "simulation is included.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    if len(tasks) < 2 or len(tasks) != len(set(tasks)):
        raise ValueError("tasks must contain at least two unique names")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    manifests = {}
    manifest_hashes = {}
    for task in tasks:
        path = (
            args.root
            / f"{task}_contexts_seed20260724"
            / "prepared_context_manifest.json"
        )
        manifest = load_json(path)
        if manifest["task_name"] != task:
            raise RuntimeError(f"manifest task identity mismatch for {task}")
        if len(set(manifest["episode_indices"])) != int(manifest["context_count"]):
            raise RuntimeError(f"manifest episode coverage is invalid for {task}")
        manifests[task] = manifest
        manifest_hashes[task] = sha256_file(path)

    distribution = summarize_distribution(
        args.root,
        tasks,
        manifests,
        manifest_hashes,
        steps=args.steps,
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    action = summarize_action(
        args.root,
        tasks,
        manifests,
        seed=args.seed,
        samples=args.bootstrap_samples,
        margin=args.action_margin,
    )
    solver = summarize_solver(args.root, tasks, manifests, manifest_hashes)
    attention = summarize_attention(args.root, tasks)
    normalization = summarize_normalization(args.root, tasks)
    protocol = summarize_protocol(args.root, tasks)
    verdict = {
        "average_cross_task_distribution": (
            "PASS" if distribution["average_gate"] else "FAIL"
        ),
        "universal_per_task_distribution": (
            "PASS" if distribution["universal_task_gate"] else "NOT_ESTABLISHED"
        ),
        "cache_teacher_detected_harm": action["detected_harm_gate"],
        "action_noninferiority": action["noninferiority"],
        "aggregate_solver_refinement": (
            "PASS" if solver["aggregate_gate"] else "FAIL"
        ),
        "all_context_solver_refinement": (
            "PASS" if solver["all_contexts_refine"] else "NOT_ESTABLISHED"
        ),
        "attention_stability_spot_check": (
            "PASS" if attention["spot_gate"] else "FAIL"
        ),
        "closed_loop_eligibility": (
            "PASS"
            if distribution["average_gate"]
            and action["detected_harm_gate"] != "FAIL"
            and solver["aggregate_gate"]
            and attention["spot_gate"]
            else "FAIL"
        ),
        "closed_loop_permission": "NOT_GRANTED",
    }
    report = {
        "tasks": tasks,
        "task_count": len(tasks),
        "contexts_per_task": [int(manifests[task]["context_count"]) for task in tasks],
        "total_contexts": sum(
            int(manifests[task]["context_count"]) for task in tasks
        ),
        "manifest_sha256": manifest_hashes,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "distribution": distribution,
        "action": action,
        "solver": solver,
        "attention": attention,
        "normalization": normalization,
        "protocol": protocol,
        "verdict": verdict,
    }
    args.output_dir.mkdir(parents=True)
    json_path = args.output_dir / "cross_task_gates.json"
    markdown_path = args.output_dir / "cross_task_gates.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
