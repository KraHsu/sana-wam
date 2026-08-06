#!/usr/bin/env python3
"""Assemble the frozen LIBERO T9 four-arm Latin-square result on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SHA256 = "5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45"
T9_ARM_SOURCE_COMMIT = "f73a7950eded2787ad57b26523da686dc5722512"
T9_ARM_RUNNER_SHA256 = (
    "d0d4cacf35d4116db3a8c166ffcdc92578a28f5037614a960e22b4ccbcc4a84c"
)
T7_SOURCE_COMMIT = "d19109a2314f8e7186571afed6b4acd816d7cfab"
T7_RUNNER_SHA256 = "b6e764bf3d3566bf3d1fe5e2e3802dafef691f6a0162568337b49ca1a183310b"
T7_RESULT_SHA256 = "9f5181a30cd0d6b676f09315244f7560cd3902004f5f17ca833330e33cb44d3a"
T7_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t7/d19109a2314f/"
    "libero-t7-foursuite-cyclic-fixed20-61e0a0ff817970e994b0875be4840ed6/"
    "RESULT.json"
)
T8_SOURCE_COMMIT = "9cd1c490d14b3c2437e82225ee8cfdf58646837e"
T8_RUNNER_SHA256 = "9ec74d83c8443d78d2ba765dd766f82a96cccef894ae63d08cf9e2a8dbf97669"
T8_RESULT_SHA256 = "bfdb852e14a5bd9b1c8e776be9f4ff108899eae65d557cebe42b06f1991b0a18"
T8_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t8/9cd1c490d14b/"
    "libero-t8-phase-rotated-fixed20-2e1efcc69bc552affb5c85b7feeb5175/"
    "RESULT.json"
)
T9_AGGREGATE_NAMESPACE = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t9_aggregate"
)
T9_ARM_NAMESPACE = Path("/DATA/share/sana_wam_libero_nonformal_screens/t9")
T9_ARM_SCHEMA = "sana-wam-libero-t9-latin-square-arm-v1"
T9_RESULT_SCHEMA = "sana-wam-libero-t9-latin-square-combined-v1"
T9_FAILURE_SCHEMA = "sana-wam-libero-t9-latin-square-combined-failure-v1"

SUITE_ORDER = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)
LABELS = ("A0", "A1", "A2", "A3", "H0", "H1", "H2", "H3")
EXPECTED_PRE_ACTION_LOSSES = {
    "A0": 13.679718971252441,
    "A1": 14.696584701538086,
    "A2": 12.06648063659668,
    "A3": 20.43115234375,
    "H0": 13.379679679870605,
    "H1": 14.698094367980957,
    "H2": 12.026171684265137,
    "H3": 20.885684967041016,
}
ARM_SPECS = {
    "A": {
        "schedule": ("A0", "A1", "A2", "A3"),
        "trailing": {"A0": 3, "A1": 2, "A2": 1, "A3": 0},
        "schema": "sana-wam-libero-t7-four-suite-cyclic-v1",
        "verdict": "T7_FOUR_SUITE_CYCLIC_GO",
        "source_commit": T7_SOURCE_COMMIT,
        "runner_sha256": T7_RUNNER_SHA256,
    },
    "B": {
        "schedule": ("A1", "A2", "A3", "A0"),
        "trailing": {"A0": 0, "A1": 3, "A2": 2, "A3": 1},
        "schema": "sana-wam-libero-t8-phase-rotated-recency-v1",
        "verdict": "T8_PHASE_ROTATED_MIXED_INCONCLUSIVE",
        "source_commit": T8_SOURCE_COMMIT,
        "runner_sha256": T8_RUNNER_SHA256,
    },
    "C": {
        "schedule": ("A2", "A3", "A0", "A1"),
        "trailing": {"A0": 1, "A1": 0, "A2": 3, "A3": 2},
        "schema": T9_ARM_SCHEMA,
        "verdict": "T9_LATIN_ARM_C_VALID",
        "root_slug": "libero-t9-arm-c-latin-fixed20",
    },
    "D": {
        "schedule": ("A3", "A0", "A1", "A2"),
        "trailing": {"A0": 2, "A1": 1, "A2": 0, "A3": 3},
        "schema": T9_ARM_SCHEMA,
        "verdict": "T9_LATIN_ARM_D_VALID",
        "root_slug": "libero-t9-arm-d-latin-fixed20",
    },
}

_ACTIVE_ROOT: Path | None = None
ROOT = Path(__file__).resolve().parent.parent


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _repo_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _aggregate_source_identity(
    expected_commit: str,
    expected_runner_sha256: str,
) -> dict[str, str]:
    actual_commit = _validate_hex(_repo_commit(), 40, "aggregate repository HEAD")
    runner_candidate = Path(__file__).absolute()
    if runner_candidate.is_symlink() or not runner_candidate.is_file():
        raise RuntimeError("aggregate runner must be a regular non-symlink file")
    runner_path = runner_candidate.resolve()
    actual_runner_sha256 = _sha256_bytes(runner_path.read_bytes())
    if (
        actual_commit != expected_commit
        or actual_runner_sha256 != expected_runner_sha256
    ):
        raise RuntimeError(
            "aggregate harness identity differs: "
            f"expected_commit={expected_commit} actual_commit={actual_commit} "
            f"expected_runner={expected_runner_sha256} actual_runner={actual_runner_sha256}"
        )
    return {
        "repo_commit": actual_commit,
        "runner_path": str(runner_path),
        "runner_sha256": actual_runner_sha256,
    }


def _validate_hex(value: str, length: int, label: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be exactly {length} lowercase hex characters")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_canonical_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} top level must be an object")
    if _canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical sorted compact JSON plus LF")
    return value


def _lstat_no_symlink_ancestry(path: Path, *, final_directory: bool) -> os.stat_result:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be normalized and absolute: {path}")
    current = Path(path.anchor)
    parts = path.parts[1:]
    final_stat: os.stat_result | None = None
    for index, part in enumerate(parts):
        current /= part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"symlink path component is forbidden: {current}")
        is_final = index == len(parts) - 1
        if not is_final or final_directory:
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"directory path component required: {current}")
        elif not stat.S_ISREG(info.st_mode):
            raise ValueError(f"regular result file required: {current}")
        final_stat = info
    if final_stat is None:
        raise ValueError(f"root path is not a valid target: {path}")
    return final_stat


def _load_frozen_result(
    path: Path,
    expected_sha256: str,
    *,
    arm: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one single-file immutable root without following symlinks."""
    expected_sha256 = _validate_hex(expected_sha256, 64, f"arm {arm} result SHA256")
    path = path.expanduser()
    file_stat = _lstat_no_symlink_ancestry(path, final_directory=False)
    root = path.parent
    root_stat = _lstat_no_symlink_ancestry(root, final_directory=True)
    if path.name != "RESULT.json":
        raise ValueError(f"arm {arm} input must be named RESULT.json")
    if stat.S_IMODE(root_stat.st_mode) != 0o500:
        raise PermissionError(f"arm {arm} root mode must be exactly 0500")
    if stat.S_IMODE(file_stat.st_mode) != 0o400:
        raise PermissionError(f"arm {arm} RESULT mode must be exactly 0400")
    entries = list(os.scandir(root))
    if len(entries) != 1 or entries[0].name != "RESULT.json":
        raise ValueError(f"arm {arm} frozen root must contain only RESULT.json")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            file_stat.st_dev,
            file_stat.st_ino,
        ):
            raise RuntimeError(f"arm {arm} RESULT changed during secure open")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"arm {arm} RESULT SHA256 differs from its frozen pin")
    result = _parse_canonical_json(payload, f"arm {arm} RESULT")
    return result, {
        "file_mode": "0400",
        "result_path": str(path),
        "result_sha256": expected_sha256,
        "root": str(root),
        "root_mode": "0500",
        "single_result_file": True,
        "size_bytes": len(payload),
        "uid": file_stat.st_uid,
        "gid": file_stat.st_gid,
    }


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{label} must be finite and strictly positive")
    return converted


def _sample_identity(result: dict[str, Any], label: str) -> dict[str, Any]:
    role = result.get("sample_roles", {}).get(label)
    if not isinstance(role, dict):
        raise ValueError(f"missing sample role for {label}")
    required = (
        "action_alignment",
        "assets",
        "dataset",
        "episode_index",
        "episode_length",
        "role",
        "start_frame",
        "task",
        "task_index",
        "update_exposure_count",
    )
    if any(key not in role for key in required):
        raise ValueError(f"sample role {label} is incomplete")
    return {key: role[key] for key in required}


def _extract_loss_rows(result: dict[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for section, expected_labels, exposure in (
        ("cyclic_training", {"A0", "A1", "A2", "A3"}, 5),
        ("fresh_heldout_transfer", {"H0", "H1", "H2", "H3"}, 0),
    ):
        raw_rows = result.get(section, {}).get("per_sample")
        if not isinstance(raw_rows, list) or len(raw_rows) != 4:
            raise ValueError(f"arm {arm} {section} must contain four rows")
        section_rows = {}
        for row in raw_rows:
            if not isinstance(row, dict) or not isinstance(row.get("label"), str):
                raise ValueError(f"arm {arm} {section} row is malformed")
            label = row["label"]
            if label in section_rows:
                raise ValueError(f"arm {arm} duplicates loss row {label}")
            section_rows[label] = row
        if set(section_rows) != expected_labels:
            raise ValueError(f"arm {arm} {section} labels differ")
        for label, row in section_rows.items():
            before = _finite_positive(
                row.get("before_action_loss"), f"{arm}/{label} pre"
            )
            terminal = _finite_positive(
                row.get("after_action_loss"), f"{arm}/{label} terminal"
            )
            if before != EXPECTED_PRE_ACTION_LOSSES[label]:
                raise ValueError(f"arm {arm} {label} pre-action loss differs")
            ratio = terminal / before
            if row.get("post_to_pre_ratio") != ratio:
                raise ValueError(f"arm {arm} {label} reported ratio differs")
            if (
                row.get("before_total_loss") != before
                or row.get("after_total_loss") != terminal
            ):
                raise ValueError(
                    f"arm {arm} {label} total/action loss identity differs"
                )
            sample = _sample_identity(result, label)
            if (
                sample["update_exposure_count"] != exposure
                or row.get("update_exposure_count") != exposure
            ):
                raise ValueError(f"arm {arm} {label} exposure count differs")
            for key in (
                "assets",
                "dataset",
                "episode_index",
                "episode_length",
                "start_frame",
                "task",
                "task_index",
            ):
                if row.get(key) != sample[key]:
                    raise ValueError(
                        f"arm {arm} {label} row/sample identity differs: {key}"
                    )
            rows[label] = {
                "pre_action_loss": before,
                "terminal_action_loss": terminal,
                "terminal_to_pre_ratio": ratio,
                "sample_identity": sample,
            }
    return rows


def _validate_arm_result(
    arm: str,
    result: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    spec = ARM_SPECS[arm]
    if result.get("schema_version") != spec["schema"]:
        raise ValueError(f"arm {arm} schema differs")
    if result.get("valid_run") is not True:
        raise ValueError(f"arm {arm} is not a valid frozen run")
    if any(
        result.get(key) != spec["verdict"]
        for key in ("result", "scientific_verdict", "verdict")
    ):
        raise ValueError(f"arm {arm} verdict differs")
    if (
        result.get("execution_result") != "PASS"
        or result.get("harness_result") != "PASS"
    ):
        raise ValueError(f"arm {arm} execution or harness result differs")
    identity = result.get("identity")
    run = result.get("run")
    scope = result.get("scope")
    if (
        not isinstance(identity, dict)
        or not isinstance(run, dict)
        or not isinstance(scope, dict)
    ):
        raise ValueError(f"arm {arm} identity/run/scope is malformed")
    source_commit = spec.get("source_commit", T9_ARM_SOURCE_COMMIT)
    runner_sha256 = spec.get("runner_sha256", T9_ARM_RUNNER_SHA256)
    if (
        identity.get("repo_commit") != source_commit
        or identity.get("runner_sha256") != runner_sha256
    ):
        raise ValueError(f"arm {arm} source or runner pin differs")
    if identity.get("config_sha256") != CONFIG_SHA256:
        raise ValueError(f"arm {arm} config pin differs")
    nonce = run.get("nonce")
    _validate_hex(nonce, 32, f"arm {arm} nonce")
    if run.get("root") != evidence["root"]:
        raise ValueError(f"arm {arm} embedded root differs")
    expected_schedule = list(spec["schedule"]) * 5
    if result.get("cyclic_schedule", {}).get("labels") != expected_schedule:
        raise ValueError(f"arm {arm} schedule differs")
    expected_forwards = 36 if arm == "A" else 44
    expected_scope = {
        "architecture_forwards_total": expected_forwards,
        "architecture_measurement_forwards": 16 if arm == "A" else 24,
        "architecture_training_forwards": 20,
        "backward_calls": 20,
        "benchmark_evaluation_executed": False,
        "optimizer_steps": 20,
        "prepare_inputs_calls": 8,
        "fresh_heldout_samples_in_backward_or_update": 0,
        "formal_training_executed": False,
        "sana_wam_training_checkpoint_loaded": False,
        "sana_wam_training_checkpoint_saved": False,
        "simulator_executed": False,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            raise ValueError(f"arm {arm} scope differs: {key}")
    if arm in {"C", "D"}:
        if identity.get("arm") != arm or run.get("arm") != arm:
            raise ValueError(f"arm {arm} enumerated identity differs")
        expected_root = (
            T9_ARM_NAMESPACE
            / T9_ARM_SOURCE_COMMIT[:12]
            / f"{spec['root_slug']}-{nonce}"
        )
        if Path(evidence["root"]) != expected_root:
            raise ValueError(f"arm {arm} root namespace differs")
        if (
            identity.get("t8_predecessor_result_sha256") != T8_RESULT_SHA256
            or identity.get("t8_predecessor_runner_sha256") != T8_RUNNER_SHA256
            or identity.get("t8_predecessor_source_commit") != T8_SOURCE_COMMIT
        ):
            raise ValueError(f"arm {arm} direct T8 predecessor pins differ")
        latin = result.get("latin_square_arm")
        if not isinstance(latin, dict):
            raise ValueError(f"arm {arm} Latin report is missing")
        if latin.get("arm_id") != arm or latin.get("typed_verdict") != spec["verdict"]:
            raise ValueError(f"arm {arm} Latin report identity differs")
        if latin.get("cycle_labels") != list(spec["schedule"]):
            raise ValueError(f"arm {arm} Latin cycle differs")
        if latin.get("trailing_updates_by_label") != spec["trailing"]:
            raise ValueError(f"arm {arm} trailing map differs")
        expected_terminal_labels = {"C": {"A1", "H1"}, "D": {"A2", "H2"}}[arm]
        terminal_replay = latin.get("terminal_phase_reproduction")
        if (
            not isinstance(terminal_replay, dict)
            or set(terminal_replay) != expected_terminal_labels
            or any(value is not True for value in terminal_replay.values())
        ):
            raise ValueError(f"arm {arm} terminal phase replay differs")
    rows = _extract_loss_rows(result, arm)
    phase_overwrite_diagnostic = None
    if arm in {"C", "D"}:
        diagnostics = result["latin_square_arm"].get("phase_overwrite_diagnostic")
        if not isinstance(diagnostics, list) or len(diagnostics) != 4:
            raise ValueError(
                f"arm {arm} phase overwrite diagnostics must have four rows"
            )
        by_update_label = {}
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                raise ValueError(f"arm {arm} phase overwrite diagnostic is malformed")
            update_label = diagnostic.get("update_label")
            if update_label in by_update_label:
                raise ValueError(
                    f"arm {arm} duplicates phase diagnostic {update_label}"
                )
            by_update_label[update_label] = diagnostic
        if set(by_update_label) != {"A0", "A1", "A2", "A3"}:
            raise ValueError(f"arm {arm} phase overwrite labels differ")
        validated_diagnostics = []
        for index, suite in enumerate(SUITE_ORDER):
            update_label = f"A{index}"
            heldout_label = f"H{index}"
            diagnostic = by_update_label[update_label]
            if (
                diagnostic.get("heldout_label") != heldout_label
                or diagnostic.get("dataset") != suite
            ):
                raise ValueError(
                    f"arm {arm} phase diagnostic identity differs for {update_label}"
                )
            r_a_phase = _finite_positive(
                diagnostic.get("rA_phase"), f"{arm}/{update_label}/rA_phase"
            )
            r_h_phase = _finite_positive(
                diagnostic.get("rH_phase"), f"{arm}/{heldout_label}/rH_phase"
            )
            q_phase = statistics.mean((r_a_phase, r_h_phase))
            if diagnostic.get("q_phase") != q_phase:
                raise ValueError(
                    f"arm {arm} phase diagnostic q differs for {update_label}"
                )
            q_terminal = statistics.mean(
                (
                    rows[update_label]["terminal_to_pre_ratio"],
                    rows[heldout_label]["terminal_to_pre_ratio"],
                )
            )
            overwrite_penalty = q_terminal / q_phase
            if diagnostic.get("overwrite_penalty") != overwrite_penalty:
                raise ValueError(
                    f"arm {arm} overwrite penalty differs for {update_label}"
                )
            validated_diagnostics.append(
                {
                    "dataset": suite,
                    "heldout_label": heldout_label,
                    "overwrite_penalty": overwrite_penalty,
                    "q_phase": q_phase,
                    "q_terminal": q_terminal,
                    "rA_phase": r_a_phase,
                    "rH_phase": r_h_phase,
                    "update_label": update_label,
                }
            )
        phase_overwrite_diagnostic = validated_diagnostics
    return {
        "arm": arm,
        "config_sha256": CONFIG_SHA256,
        "nonce": nonce,
        "result_path": evidence["result_path"],
        "result_sha256": evidence["result_sha256"],
        "root": evidence["root"],
        "runner_sha256": runner_sha256,
        "schema_version": spec["schema"],
        "source_commit": source_commit,
        "verdict": spec["verdict"],
        "rows": rows,
        "input_verification": evidence,
        "phase_overwrite_diagnostic": phase_overwrite_diagnostic,
    }


def _rankdata(values: list[float]) -> list[float]:
    ranked = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in ordered[cursor:end]:
            ranked[index] = rank
        cursor = end
    return ranked


def _spearman_rho(left: list[float], right: list[float]) -> float | None:
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_mean = statistics.mean(left_ranks)
    right_mean = statistics.mean(right_ranks)
    covariance = sum(
        (left - left_mean) * (right - right_mean)
        for left, right in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return covariance / (left_scale * right_scale)


def _classify_latin_square_results(
    q_by_suite_by_trailing: dict[str, dict[int, float]],
) -> dict[str, Any]:
    if set(q_by_suite_by_trailing) != set(SUITE_ORDER):
        raise ValueError("combined suite keys differ")
    per_suite = []
    endpoint_count = 0
    strong_count = 0
    for suite in SUITE_ORDER:
        mapping = q_by_suite_by_trailing[suite]
        if set(mapping) != {0, 1, 2, 3}:
            raise ValueError(f"combined trailing keys differ for {suite}")
        q_values = [
            _finite_positive(mapping[index], f"{suite}/q({index})")
            for index in range(4)
        ]
        average_ranks = _rankdata(q_values)
        rho = _spearman_rho(q_values, [0.0, 1.0, 2.0, 3.0])
        endpoint = q_values[3] > q_values[0]
        strong = rho is not None and (
            rho > 0.8 or math.isclose(rho, 0.8, rel_tol=0.0, abs_tol=1.0e-12)
        )
        endpoint_count += int(endpoint)
        strong_count += int(strong)
        per_suite.append(
            {
                "average_ranks": average_ranks,
                "endpoint_q3_gt_q0": endpoint,
                "q_by_trailing": {str(index): q_values[index] for index in range(4)},
                "rho": rho,
                "strong_rho_at_least_0_8": strong,
                "suite": suite,
                "trailing_average_ranks": [1.0, 2.0, 3.0, 4.0],
            }
        )
    if endpoint_count == 4 and strong_count >= 3:
        verdict = "T9_COMMON_POSITION_EFFECT_SUPPORTED"
    elif endpoint_count == 4:
        verdict = "T9_POSITION_WITH_IDENTITY_INTERACTION"
    else:
        verdict = "T9_NO_COMMON_POSITION_EFFECT"
    return {
        "endpoint_count": endpoint_count,
        "per_suite": per_suite,
        "strong_count": strong_count,
        "strong_rho_threshold_inclusive": 0.8,
        "typed_verdict": verdict,
    }


def _assemble_latin_square(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(arms) != {"A", "B", "C", "D"}:
        raise ValueError("Latin-square arm keys differ")
    reference_samples = {
        label: arms["A"]["rows"][label]["sample_identity"] for label in LABELS
    }
    for arm, artifact in arms.items():
        for label in LABELS:
            if artifact["rows"][label]["sample_identity"] != reference_samples[label]:
                raise ValueError(f"arm {arm} sample identity differs for {label}")
    cells = []
    by_suite: dict[str, dict[int, dict[str, Any]]] = {
        suite: {} for suite in SUITE_ORDER
    }
    for arm in ("A", "B", "C", "D"):
        artifact = arms[arm]
        for index, suite in enumerate(SUITE_ORDER):
            update_label = f"A{index}"
            heldout_label = f"H{index}"
            trailing = ARM_SPECS[arm]["trailing"][update_label]
            update = artifact["rows"][update_label]
            heldout = artifact["rows"][heldout_label]
            r_a = update["terminal_to_pre_ratio"]
            r_h = heldout["terminal_to_pre_ratio"]
            q = statistics.mean((r_a, r_h))
            _finite_positive(q, f"{arm}/{suite}/q")
            cell = {
                "arm": arm,
                "evidence": {
                    key: artifact[key]
                    for key in (
                        "config_sha256",
                        "nonce",
                        "result_path",
                        "result_sha256",
                        "root",
                        "runner_sha256",
                        "schema_version",
                        "source_commit",
                        "verdict",
                    )
                },
                "heldout_label": heldout_label,
                "heldout_sample_identity": heldout["sample_identity"],
                "pre_heldout_action_loss": heldout["pre_action_loss"],
                "pre_update_action_loss": update["pre_action_loss"],
                "q": q,
                "rA": r_a,
                "rH": r_h,
                "suite": suite,
                "terminal_heldout_action_loss": heldout["terminal_action_loss"],
                "terminal_update_action_loss": update["terminal_action_loss"],
                "trailing_updates": trailing,
                "update_label": update_label,
                "update_sample_identity": update["sample_identity"],
            }
            if trailing in by_suite[suite]:
                raise ValueError(f"duplicate Latin cell for {suite}/k={trailing}")
            by_suite[suite][trailing] = cell
            cells.append(cell)
    if any(set(mapping) != {0, 1, 2, 3} for mapping in by_suite.values()):
        raise ValueError("Latin-square cell coverage is incomplete")
    q_matrix = {
        suite: {position: by_suite[suite][position]["q"] for position in range(4)}
        for suite in SUITE_ORDER
    }
    classifier = _classify_latin_square_results(q_matrix)
    return {
        "arm_to_cell": {
            arm: {
                f"A{index}": ARM_SPECS[arm]["trailing"][f"A{index}"]
                for index in range(4)
            }
            for arm in ("A", "B", "C", "D")
        },
        "cells": sorted(
            cells,
            key=lambda row: (SUITE_ORDER.index(row["suite"]), row["trailing_updates"]),
        ),
        "classifier": classifier,
        "matrices": {
            metric: {
                suite: {
                    str(position): by_suite[suite][position][metric]
                    for position in range(4)
                }
                for suite in SUITE_ORDER
            }
            for metric in ("rA", "rH", "q")
        },
        "q_by_suite_by_trailing": {
            suite: {str(position): q_matrix[suite][position] for position in range(4)}
            for suite in SUITE_ORDER
        },
        "sample_identities": reference_samples,
        "suite_order": list(SUITE_ORDER),
        "typed_verdict": classifier["typed_verdict"],
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_run_root(candidate: Path) -> Path:
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("aggregate run root must be normalized and absolute")
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"refusing to reuse aggregate root: {candidate}")
    _lstat_no_symlink_ancestry(candidate.parent, final_directory=True)
    os.mkdir(candidate, 0o700)
    _fsync_directory(candidate.parent)
    return candidate


def _write_terminal_report(path: Path, report: dict[str, Any]) -> None:
    payload = _canonical_json_bytes(report)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _freeze_run_root(root: Path) -> None:
    for child in root.iterdir():
        info = os.lstat(child)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"unexpected aggregate-root entry: {child}")
        os.chmod(child, 0o444, follow_symlinks=False)
    descriptor = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(root.parent)


def _terminalize_failure(root: Path, error: BaseException) -> None:
    marker = root / "FAILED.json"
    if not marker.exists() and not marker.is_symlink():
        _write_terminal_report(
            marker,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "error_type": type(error).__name__,
                "result": "FAIL",
                "schema_version": T9_FAILURE_SCHEMA,
            },
        )
    _freeze_run_root(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-c-result", required=True, type=Path)
    parser.add_argument("--arm-c-result-sha256", required=True)
    parser.add_argument("--arm-d-result", required=True, type=Path)
    parser.add_argument("--arm-d-result-sha256", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    args = parser.parse_args()

    nonce = _validate_hex(args.nonce, 32, "aggregate nonce")
    expected_commit = _validate_hex(args.expected_repo_commit, 40, "T9 source commit")
    expected_runner = _validate_hex(args.expected_runner_sha256, 64, "T9 runner SHA256")
    expected_root = (
        T9_AGGREGATE_NAMESPACE
        / expected_commit[:12]
        / f"libero-t9-latin-square-combined-{nonce}"
    )
    if args.run_root.expanduser() != expected_root:
        raise ValueError(
            f"aggregate root differs: expected={expected_root} got={args.run_root}"
        )

    global _ACTIVE_ROOT
    run_root = _create_run_root(expected_root)
    _ACTIVE_ROOT = run_root
    aggregate_identity = _aggregate_source_identity(expected_commit, expected_runner)

    inputs = {
        "A": (T7_RESULT_PATH, T7_RESULT_SHA256),
        "B": (T8_RESULT_PATH, T8_RESULT_SHA256),
        "C": (args.arm_c_result.expanduser(), args.arm_c_result_sha256),
        "D": (args.arm_d_result.expanduser(), args.arm_d_result_sha256),
    }
    if len({str(path) for path, _sha in inputs.values()}) != 4:
        raise ValueError("all four RESULT paths must be distinct")
    arms = {}
    ownership = set()
    for arm, (path, expected_sha) in inputs.items():
        result, evidence = _load_frozen_result(path, expected_sha, arm=arm)
        ownership.add((evidence["uid"], evidence["gid"]))
        arms[arm] = _validate_arm_result(arm, result, evidence)
    if len(ownership) != 1:
        raise PermissionError("four frozen roots do not share one owner/group identity")
    if (
        arms["C"]["nonce"] == arms["D"]["nonce"]
        or arms["C"]["root"] == arms["D"]["root"]
    ):
        raise ValueError("T9 arms C and D must use different nonces and roots")
    if arms["C"]["result_sha256"] == arms["D"]["result_sha256"]:
        raise ValueError("T9 arms C and D must have distinct frozen results")

    assembly = _assemble_latin_square(arms)
    arm_identities = {
        arm: {
            key: artifact[key]
            for key in (
                "arm",
                "config_sha256",
                "nonce",
                "result_path",
                "result_sha256",
                "root",
                "runner_sha256",
                "schema_version",
                "source_commit",
                "verdict",
            )
        }
        for arm, artifact in arms.items()
    }
    report = {
        "arm_identities": arm_identities,
        "assembly": assembly,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_result": "PASS",
        "harness_result": "PASS",
        "input_verification": {
            arm: artifact["input_verification"] for arm, artifact in arms.items()
        },
        "identity": {
            **aggregate_identity,
            "config_sha256": CONFIG_SHA256,
            "t9_arm_runner_sha256": T9_ARM_RUNNER_SHA256,
            "t9_arm_source_commit": T9_ARM_SOURCE_COMMIT,
        },
        "phase_overwrite_diagnostics": {
            "C": arms["C"]["phase_overwrite_diagnostic"],
            "D": arms["D"]["phase_overwrite_diagnostic"],
            "excluded_from_combined_classifier": True,
        },
        "result": assembly["typed_verdict"],
        "run": {"nonce": nonce, "root": str(run_root)},
        "schema_version": T9_RESULT_SCHEMA,
        "scope": {
            "checkpoint_loaded_or_saved": False,
            "formal_training_or_evaluation_executed": False,
            "gpu_or_model_executed": False,
            "input_result_count": 4,
            "read_only_evidence_assembly": True,
        },
        "valid_run": True,
        "scientific_verdict": assembly["typed_verdict"],
        "verdict": assembly["typed_verdict"],
    }
    if any(run_root.iterdir()):
        raise RuntimeError("aggregate root was not empty before terminal write")
    _write_terminal_report(run_root / "RESULT.json", report)
    _freeze_run_root(run_root)
    _ACTIVE_ROOT = None
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as error:
        if _ACTIVE_ROOT is not None:
            try:
                _terminalize_failure(_ACTIVE_ROOT, error)
            except BaseException as freeze_error:
                print(
                    "failed to terminalize T9 aggregate root: "
                    f"{type(freeze_error).__name__}: {freeze_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
