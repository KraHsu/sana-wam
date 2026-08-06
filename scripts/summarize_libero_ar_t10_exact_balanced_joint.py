#!/usr/bin/env python3
"""Assemble the frozen paired LIBERO T10 SEQ/JOINT screen on CPU."""

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


T10_CONFIG_SHA256 = "5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45"
T10_ARM_SCHEMA = "sana-wam-libero-t10-matched-core-arm-v1"
T10_AGGREGATE_SCHEMA = "sana-wam-libero-t10-exact-balanced-joint-combined-v1"
T10_AGGREGATE_FAILURE_SCHEMA = (
    "sana-wam-libero-t10-exact-balanced-joint-combined-failure-v1"
)
T10_ARM_VERDICTS = {
    "SEQ": "T10_SEQ_BRIDGE_VALID",
    "JOINT": "T10_JOINT_ARM_VALID",
}
T10_ARM_ROOT_SLUGS = {
    "SEQ": "libero-t10-seq-matched-core-fixed20",
    "JOINT": "libero-t10-joint-matched-core-fixed20",
}
T10_ARM_NAMESPACE = Path("/DATA/share/sana_wam_libero_nonformal_screens/t10")
T10_AGGREGATE_NAMESPACE = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t10_aggregate"
)

# These two pins must be mechanically replaced after the paired-arm source
# commit exists.  A draft containing either literal is not executable.
FUTURE_T10_ARM_SOURCE_COMMIT = "6861e5a13fa8110986f1b46f3062de9f0b0e3954"
FUTURE_T10_ARM_RUNNER_SHA256 = (
    "d7e3e7da3cd0a13e5217f0a044b2f36a4a703be7227149a897268f82b74857dd"
)

T9_AGGREGATE_SOURCE_COMMIT = "b7cded5fd9cf83ffabeba18a7e352b1cb4438b66"
T9_AGGREGATE_RUNNER_SHA256 = (
    "b884ef028a48ee37afda70e730c431db48ed30f62be7f59cb2c665b6072ffc0d"
)
T9_AGGREGATE_RESULT_SHA256 = (
    "0cfc53b820939123de4bc2a626380495878d4d5c9a37a0be9be667173254149d"
)
T9_AGGREGATE_ROOT = (
    "/DATA/share/sana_wam_libero_nonformal_screens/t9_aggregate/"
    "b7cded5fd9cf/libero-t9-latin-square-combined-"
    "ab99dd8758ef03bb191d5fb48f3b9fbc"
)

SUITE_ORDER = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)
UPDATE_LABELS = ("A0", "A1", "A2", "A3")
HELDOUT_LABELS = ("H0", "H1", "H2", "H3")
ALL_LABELS = (*UPDATE_LABELS, *HELDOUT_LABELS)
LABEL_TO_SUITE = {
    label: SUITE_ORDER[index]
    for labels in (UPDATE_LABELS, HELDOUT_LABELS)
    for index, label in enumerate(labels)
}
T9_FOUR_POSITION_Q_MEDIANS = {
    SUITE_ORDER[0]: 0.7488868555671038,
    SUITE_ORDER[1]: 0.6095596365043602,
    SUITE_ORDER[2]: 0.5490714609077374,
    SUITE_ORDER[3]: 0.3536560999394767,
}
T7_PRE_ACTION_LOSSES = {
    "A0": 13.679718971252441,
    "A1": 14.696584701538086,
    "A2": 12.06648063659668,
    "A3": 20.43115234375,
    "H0": 13.379679679870605,
    "H1": 14.698094367980957,
    "H2": 12.026171684265137,
    "H3": 20.885684967041016,
}
T7_TERMINAL_ACTION_LOSSES = {
    "A0": 14.97652816772461,
    "A1": 11.250626564025879,
    "A2": 5.85097074508667,
    "A3": 2.8098158836364746,
    "H0": 14.401052474975586,
    "H1": 11.137908935546875,
    "H2": 6.1973876953125,
    "H3": 2.8864619731903076,
}
T7_REPRO_ABSOLUTE_TOLERANCE = 1.0e-6
T7_REPRO_RELATIVE_TOLERANCE = 1.0e-6
T10_INITIALIZATION_SEED = 20260806
T10_TRAINING_LOSS_RECIPE_SEED = 20260826
T10_TRAINING_RECIPE_SIGNATURE_COUNT = 80
T10_FIXED_RECIPE_SIGNATURE_SHA256 = (
    "ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad"
)
T10_SANA_COMMIT = "16b9cec673e3335724ba2d8db25de7f9ed229292"
T10_STATS_SHA256 = "e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7"
T10_STATS_POPULATION_SHA256 = (
    "7ed9772facf261299022e55169bcdaaa49fe3a7a20e0057e08cf419b5e584146"
)
T7_SOURCE_COMMIT = "d19109a2314f8e7186571afed6b4acd816d7cfab"
T7_RUNNER_SHA256 = "b6e764bf3d3566bf3d1fe5e2e3802dafef691f6a0162568337b49ca1a183310b"
T7_RESULT_SHA256 = "9f5181a30cd0d6b676f09315244f7560cd3902004f5f17ca833330e33cb44d3a"
T10_MANIFEST_SHA256 = {
    "eligible": "0b6bfe4c51e8b9e2ac3f538bac7853b5e1e871ea96dda5df6976ee27b5b25370",
    "selection": "e969e76e4bcb8a3c2b0b3fffff35e0ec10c478e45cb6535b7bd91061d3a6a7ee",
    "update": "5199c87af876c437ecec35b57da3e118a2924c44a9bb454c9846e7c0ca968aa6",
}
T10_NO_MUTATION_COMPONENTS = (
    "model_parameters",
    "fp32_master_parameters",
    "optimizer_state",
)
T10_NO_MUTATION_EVIDENCE_SCOPE = (
    "all_architecture_model_parameter_identity_and_version",
    "all_fp32_master_parameter_identity_and_version",
    "all_optimizer_state_identity_and_version",
)
T10_NO_MUTATION_VERIFICATION_POINT = (
    "after_four_microbacks_before_gradient_clip_master_sync_and_optimizer_step"
)

ROOT = Path(__file__).resolve().parent.parent
_ACTIVE_ROOT: Path | None = None


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


def _compact_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _validate_hex(value: Any, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be exactly {length} lowercase hex characters")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{label} must be finite and strictly positive")
    return converted


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
    expected_commit: str, expected_runner_sha256: str
) -> dict[str, str]:
    actual_commit = _validate_hex(_repo_commit(), 40, "aggregate repository HEAD")
    candidate = Path(__file__).absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("aggregate runner must be a regular non-symlink file")
    runner_path = candidate.resolve()
    actual_runner_sha256 = _sha256_bytes(runner_path.read_bytes())
    if (
        actual_commit != expected_commit
        or actual_runner_sha256 != expected_runner_sha256
    ):
        raise RuntimeError(
            "aggregate source identity differs: "
            f"expected_commit={expected_commit} actual_commit={actual_commit} "
            f"expected_runner={expected_runner_sha256} actual_runner={actual_runner_sha256}"
        )
    return {
        "repo_commit": actual_commit,
        "runner_path": str(runner_path),
        "runner_sha256": actual_runner_sha256,
    }


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
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} top level must be an object")
    if _canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical sorted compact JSON plus LF")
    return value


def _lstat_no_symlink_ancestry(path: Path, *, final_directory: bool) -> os.stat_result:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be normalized and absolute: {path}")
    current = Path(path.anchor)
    final_stat: os.stat_result | None = None
    for index, part in enumerate(path.parts[1:]):
        current /= part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"symlink path component is forbidden: {current}")
        final = index == len(path.parts[1:]) - 1
        if not final or final_directory:
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"directory path component required: {current}")
        elif not stat.S_ISREG(info.st_mode):
            raise ValueError(f"regular result file required: {current}")
        final_stat = info
    if final_stat is None:
        raise ValueError(f"path has no inspectable component: {path}")
    return final_stat


def _load_frozen_result(
    path: Path, expected_sha256: str, *, arm: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_sha256 = _validate_hex(expected_sha256, 64, f"{arm} RESULT SHA256")
    path = path.expanduser()
    file_stat = _lstat_no_symlink_ancestry(path, final_directory=False)
    root = path.parent
    root_stat = _lstat_no_symlink_ancestry(root, final_directory=True)
    if path.name != "RESULT.json":
        raise ValueError(f"{arm} input must be named RESULT.json")
    if stat.S_IMODE(root_stat.st_mode) != 0o500:
        raise PermissionError(f"{arm} root mode must be exactly 0500")
    if stat.S_IMODE(file_stat.st_mode) != 0o400:
        raise PermissionError(f"{arm} RESULT mode must be exactly 0400")
    entries = list(os.scandir(root))
    if len(entries) != 1 or entries[0].name != "RESULT.json":
        raise ValueError(f"{arm} frozen root must contain only RESULT.json")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened_stat = os.fstat(descriptor)
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            file_stat.st_dev,
            file_stat.st_ino,
        ):
            raise RuntimeError(f"{arm} RESULT changed during secure open")
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
        raise ValueError(f"{arm} RESULT SHA256 differs from its supplied pin")
    return _parse_canonical_json(payload, f"{arm} RESULT"), {
        "file_mode": "0400",
        "gid": file_stat.st_gid,
        "result_path": str(path),
        "result_sha256": expected_sha256,
        "root": str(root),
        "root_mode": "0500",
        "single_result_file": True,
        "size_bytes": len(payload),
        "uid": file_stat.st_uid,
    }


def _expected_weights(arm: str, macro_index: int) -> dict[str, float]:
    if arm == "JOINT":
        return {label: 0.25 for label in UPDATE_LABELS}
    active = UPDATE_LABELS[macro_index % len(UPDATE_LABELS)]
    return {label: 1.0 if label == active else 0.0 for label in UPDATE_LABELS}


def _validate_schedule(arm: str, result: dict[str, Any]) -> dict[str, Any]:
    schedule = result.get("matched_core_schedule")
    if not isinstance(schedule, dict):
        raise ValueError(f"{arm} matched_core_schedule is missing")
    expected_scalars = {
        "backward_calls_per_macro_step": 4,
        "forward_exposures_per_sample": 20,
        "macro_steps": 20,
        "optimizer_steps_per_macro_step": 1,
        "parameters_constant_across_four_component_backward_calls": True,
    }
    for key, expected in expected_scalars.items():
        if schedule.get(key) != expected:
            raise ValueError(f"{arm} schedule differs: {key}")
    if schedule.get("forward_order_within_macro_step") != list(UPDATE_LABELS):
        raise ValueError(f"{arm} micro forward order differs")
    expected_cumulative = {label: 5.0 for label in UPDATE_LABELS}
    if schedule.get("effective_cumulative_loss_weights") != expected_cumulative:
        raise ValueError(f"{arm} cumulative scalar loss weights differ")
    per_macro = schedule.get("per_macro_loss_weights")
    if not isinstance(per_macro, list) or len(per_macro) != 20:
        raise ValueError(f"{arm} must report exactly 20 macro weight rows")
    validated_weights = []
    for index, row in enumerate(per_macro):
        expected = _expected_weights(arm, index)
        if not isinstance(row, dict) or row.get("macro_step") != index + 1:
            raise ValueError(f"{arm} macro row {index + 1} identity differs")
        if row.get("weights") != expected:
            raise ValueError(f"{arm} macro row {index + 1} weights differ")
        validated_weights.append({"macro_step": index + 1, "weights": expected})
    per_step = result.get("per_step")
    if not isinstance(per_step, list) or len(per_step) != 20:
        raise ValueError(f"{arm} per_step must contain exactly 20 rows")
    for index, row in enumerate(per_step):
        if not isinstance(row, dict):
            raise ValueError(f"{arm} per_step row {index + 1} is malformed")
        if row.get("macro_step") != index + 1:
            raise ValueError(f"{arm} per_step macro index differs")
        if row.get("labels") != list(UPDATE_LABELS):
            raise ValueError(f"{arm} per_step micro order differs")
        if row.get("loss_weights") != _expected_weights(arm, index):
            raise ValueError(f"{arm} per_step loss weights differ")
        for mapping_key in (
            "component_datasets",
            "component_episode_indices",
            "component_losses",
            "component_recipe_signatures",
            "component_task_indices",
        ):
            mapping = row.get(mapping_key)
            if not isinstance(mapping, dict) or set(mapping) != set(UPDATE_LABELS):
                raise ValueError(f"{arm} per_step {mapping_key} coverage differs")
    no_mutation = schedule.get("no_intra_macro_mutation")
    if not isinstance(no_mutation, dict):
        raise ValueError(f"{arm} no-intra-macro-mutation report is missing")
    expected_no_mutation_summary = {
        "all_macro_steps_verified": True,
        "evidence_scope": list(T10_NO_MUTATION_EVIDENCE_SCOPE),
        "macro_steps_verified": 20,
        "verification_point": T10_NO_MUTATION_VERIFICATION_POINT,
    }
    for key, expected in expected_no_mutation_summary.items():
        if no_mutation.get(key) != expected:
            raise ValueError(f"{arm} no-intra-macro-mutation summary differs: {key}")
    evidence = no_mutation.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 20:
        raise ValueError(f"{arm} no-intra-macro-mutation evidence coverage differs")
    validated_evidence = []
    expected_components = set(T10_NO_MUTATION_COMPONENTS)
    for index, row in enumerate(evidence):
        if not isinstance(row, dict) or row.get("macro_step") != index + 1:
            raise ValueError(
                f"{arm} no-intra-macro-mutation evidence step {index + 1} differs"
            )
        component_flags = row.get("components_unchanged")
        if (
            not isinstance(component_flags, dict)
            or set(component_flags) != expected_components
            or any(
                component_flags[component] is not True
                for component in expected_components
            )
        ):
            raise ValueError(
                f"{arm} no-intra-macro-mutation component flags differ at "
                f"step {index + 1}"
            )
        component_sha256 = row.get("component_state_sha256")
        if (
            not isinstance(component_sha256, dict)
            or set(component_sha256) != expected_components
        ):
            raise ValueError(
                f"{arm} no-intra-macro-mutation component SHA coverage differs at "
                f"step {index + 1}"
            )
        validated_component_sha256 = {
            component: _validate_hex(
                component_sha256[component],
                64,
                f"{arm} macro {index + 1} {component} state SHA256",
            )
            for component in T10_NO_MUTATION_COMPONENTS
        }
        pre_sha256 = _validate_hex(
            row.get("pre_microback_state_sha256"),
            64,
            f"{arm} macro {index + 1} pre-microback state SHA256",
        )
        post_sha256 = _validate_hex(
            row.get("post_microback_pre_optimizer_state_sha256"),
            64,
            f"{arm} macro {index + 1} post-microback state SHA256",
        )
        if pre_sha256 != post_sha256 or row.get("state_unchanged") is not True:
            raise ValueError(
                f"{arm} no-intra-macro-mutation composite state differs at "
                f"step {index + 1}"
            )
        validated_evidence.append(
            {
                "component_state_sha256": validated_component_sha256,
                "components_unchanged": {
                    component: True for component in T10_NO_MUTATION_COMPONENTS
                },
                "macro_step": index + 1,
                "post_microback_pre_optimizer_state_sha256": post_sha256,
                "pre_microback_state_sha256": pre_sha256,
                "state_unchanged": True,
            }
        )
    return {
        "effective_cumulative_loss_weights": expected_cumulative,
        "forward_order_within_macro_step": list(UPDATE_LABELS),
        "macro_steps": 20,
        "no_intra_macro_mutation": {
            **expected_no_mutation_summary,
            "evidence": validated_evidence,
        },
        "per_macro_loss_weights": validated_weights,
    }


def _validate_scope(arm: str, result: dict[str, Any]) -> dict[str, Any]:
    scope = result.get("scope")
    if not isinstance(scope, dict):
        raise ValueError(f"{arm} scope is missing")
    expected = {
        "architecture_forwards_total": 96,
        "architecture_measurement_forwards": 16,
        "architecture_training_forwards": 80,
        "backward_calls": 80,
        "benchmark_evaluation_executed": False,
        "effective_cumulative_loss_weight_per_update_sample": 5.0,
        "formal_training_executed": False,
        "fresh_heldout_measurement_forwards": 8,
        "fresh_heldout_samples_in_backward_or_update": 0,
        "macro_optimizer_steps": 20,
        "optimizer_steps": 20,
        "prepare_inputs_calls": 8,
        "raw_training_forward_exposures_per_sample": 20,
        "sana_wam_training_checkpoint_loaded": False,
        "sana_wam_training_checkpoint_saved": False,
        "simulator_executed": False,
    }
    for key, value in expected.items():
        if scope.get(key) != value:
            raise ValueError(f"{arm} scope differs: {key}")
    return {key: scope[key] for key in expected}


def _extract_rows(result: dict[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for section, labels, exposure, coefficient in (
        ("update_samples", set(UPDATE_LABELS), 20, 5.0),
        ("fresh_heldout_transfer", set(HELDOUT_LABELS), 0, 0.0),
    ):
        raw = result.get(section, {}).get("per_sample")
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError(f"{arm} {section} must contain four rows")
        section_rows: dict[str, dict[str, Any]] = {}
        for row in raw:
            if not isinstance(row, dict) or not isinstance(row.get("label"), str):
                raise ValueError(f"{arm} {section} row is malformed")
            label = row["label"]
            if label in section_rows:
                raise ValueError(f"{arm} duplicates row {label}")
            section_rows[label] = row
        if set(section_rows) != labels:
            raise ValueError(f"{arm} {section} labels differ")
        for label, row in section_rows.items():
            before = _finite_positive(
                row.get("before_action_loss"), f"{arm}/{label}/pre"
            )
            terminal = _finite_positive(
                row.get("after_action_loss"), f"{arm}/{label}/terminal"
            )
            ratio = terminal / before
            if row.get("post_to_pre_ratio") != ratio:
                raise ValueError(f"{arm}/{label} reported ratio differs")
            if row.get("dataset") != LABEL_TO_SUITE[label]:
                raise ValueError(f"{arm}/{label} suite identity differs")
            if row.get("update_exposure_count") != exposure:
                raise ValueError(f"{arm}/{label} raw exposure count differs")
            if row.get("effective_cumulative_loss_weight") != coefficient:
                raise ValueError(f"{arm}/{label} cumulative coefficient differs")
            allowed = T7_REPRO_ABSOLUTE_TOLERANCE + (
                T7_REPRO_RELATIVE_TOLERANCE * abs(T7_PRE_ACTION_LOSSES[label])
            )
            if abs(before - T7_PRE_ACTION_LOSSES[label]) > allowed:
                raise ValueError(
                    f"{arm}/{label} pre-action loss differs from frozen T7"
                )
            rows[label] = {
                "pre_action_loss": before,
                "terminal_action_loss": terminal,
                "terminal_to_pre_ratio": ratio,
            }
    return rows


def _validate_t9_pins(arm: str, result: dict[str, Any]) -> None:
    identity = result["identity"]
    expected_identity = {
        "t9_aggregate_predecessor_result_sha256": T9_AGGREGATE_RESULT_SHA256,
        "t9_aggregate_predecessor_runner_sha256": T9_AGGREGATE_RUNNER_SHA256,
        "t9_aggregate_predecessor_source_commit": T9_AGGREGATE_SOURCE_COMMIT,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"{arm} T9 identity pin differs: {key}")
    direct = result.get("direct_predecessor")
    expected_direct = {
        "result_sha256": T9_AGGREGATE_RESULT_SHA256,
        "root": T9_AGGREGATE_ROOT,
        "runner_sha256": T9_AGGREGATE_RUNNER_SHA256,
        "source_commit": T9_AGGREGATE_SOURCE_COMMIT,
        "stage": "T9_AGGREGATE",
    }
    if not isinstance(direct, dict):
        raise ValueError(f"{arm} direct T9 predecessor is missing")
    for key, expected in expected_direct.items():
        if direct.get(key) != expected:
            raise ValueError(f"{arm} direct T9 pin differs: {key}")


def _validate_randomness_and_manifests(
    arm: str, result: dict[str, Any]
) -> dict[str, Any]:
    identity = result.get("identity")
    randomness = result.get("randomness")
    assets = result.get("assets")
    if not isinstance(identity, dict) or not isinstance(randomness, dict):
        raise ValueError(f"{arm} identity/randomness report is missing")
    if not isinstance(assets, dict):
        raise ValueError(f"{arm} assets report is missing")

    expected_randomness = {
        "initialization_seed": T10_INITIALIZATION_SEED,
        "rng_state_restored_after_every_forward": True,
        "total_recipe_signatures": 96,
        "training_loss_recipe_seed": T10_TRAINING_LOSS_RECIPE_SEED,
        "training_recipe_signature_sha256": T10_FIXED_RECIPE_SIGNATURE_SHA256,
        "training_recipe_signatures": T10_TRAINING_RECIPE_SIGNATURE_COUNT,
        "unique_loss_recipe_signature_count": 1,
        "unique_training_recipe_signatures": 1,
    }
    for key, expected in expected_randomness.items():
        if randomness.get(key) != expected:
            raise ValueError(f"{arm} randomness contract differs: {key}")

    expected_identity_pins = {
        "sana_commit": T10_SANA_COMMIT,
        "stats_population_sha256": T10_STATS_POPULATION_SHA256,
        "stats_sha256": T10_STATS_SHA256,
        "t7_predecessor_result_sha256": T7_RESULT_SHA256,
        "t7_predecessor_runner_sha256": T7_RUNNER_SHA256,
        "t7_predecessor_source_commit": T7_SOURCE_COMMIT,
    }
    for key, expected in expected_identity_pins.items():
        if identity.get(key) != expected:
            raise ValueError(f"{arm} runtime identity pin differs: {key}")
    expected_asset_pins = {
        "config_sha256": T10_CONFIG_SHA256,
        "stats_population_sha256": T10_STATS_POPULATION_SHA256,
        "stats_sha256": T10_STATS_SHA256,
    }
    for key, expected in expected_asset_pins.items():
        if assets.get(key) != expected:
            raise ValueError(f"{arm} assets pin differs: {key}")
    for assets_key, identity_key in (
        ("external_files_manifest_sha256", "external_assets_sha256"),
        ("role_files_post_sample_manifest_sha256", "role_assets_post_sample_sha256"),
        (
            "suite_metadata_post_sample_manifest_sha256",
            "cross_suite_metadata_post_sample_sha256",
        ),
    ):
        assets_sha256 = _validate_hex(
            assets.get(assets_key), 64, f"{arm} assets {assets_key}"
        )
        if identity.get(identity_key) != assets_sha256:
            raise ValueError(f"{arm} identity/assets binding differs: {assets_key}")

    per_macro_signatures = randomness.get("per_macro_recipe_signatures")
    if not isinstance(per_macro_signatures, list) or len(per_macro_signatures) != 20:
        raise ValueError(f"{arm} per-macro recipe signature coverage differs")
    flattened_signatures = []
    per_step = result.get("per_step")
    if not isinstance(per_step, list) or len(per_step) != 20:
        raise ValueError(f"{arm} per-step recipe signature coverage differs")
    for index, signatures in enumerate(per_macro_signatures):
        if not isinstance(signatures, dict) or set(signatures) != set(UPDATE_LABELS):
            raise ValueError(
                f"{arm} per-macro recipe signature labels differ at step {index + 1}"
            )
        if any(
            signatures[label] != T10_FIXED_RECIPE_SIGNATURE_SHA256
            for label in UPDATE_LABELS
        ):
            raise ValueError(
                f"{arm} per-macro recipe signature differs at step {index + 1}"
            )
        if per_step[index].get("component_recipe_signatures") != signatures:
            raise ValueError(
                f"{arm} per-step/per-macro recipe signatures differ at step {index + 1}"
            )
        flattened_signatures.extend(signatures[label] for label in UPDATE_LABELS)
    if len(flattened_signatures) != T10_TRAINING_RECIPE_SIGNATURE_COUNT or set(
        flattened_signatures
    ) != {T10_FIXED_RECIPE_SIGNATURE_SHA256}:
        raise ValueError(f"{arm} flattened training recipe signatures differ")

    manifest_fields = {
        "eligible": (
            "eligible_manifest_sha256",
            "eligible_same_task_probe_manifest",
            "eligible_same_task_probe_manifest_sha256",
        ),
        "selection": (
            "selection_manifest_sha256",
            "selected_same_task_probe_manifest",
            "selected_same_task_probe_manifest_sha256",
        ),
        "update": (
            "update_sample_manifest_sha256",
            "update_sample_manifest",
            "update_sample_manifest_sha256",
        ),
    }
    manifest_payloads = {}
    for manifest_name, (
        identity_key,
        payload_key,
        randomness_sha_key,
    ) in manifest_fields.items():
        expected_sha256 = T10_MANIFEST_SHA256[manifest_name]
        if identity.get(identity_key) != expected_sha256:
            raise ValueError(f"{arm} identity {manifest_name} manifest SHA256 differs")
        if randomness.get(randomness_sha_key) != expected_sha256:
            raise ValueError(
                f"{arm} randomness {manifest_name} manifest SHA256 differs"
            )
        payload = randomness.get(payload_key)
        if not isinstance(payload, dict):
            raise ValueError(f"{arm} {manifest_name} manifest payload is missing")
        if _compact_json_sha256(payload) != expected_sha256:
            raise ValueError(f"{arm} {manifest_name} manifest payload SHA256 differs")
        manifest_payloads[manifest_name] = payload

    common_identity = {
        key: value
        for key, value in identity.items()
        if key not in {"arm", "identity_sha256"}
    }
    return {
        "assets": assets,
        "common_identity": common_identity,
        "manifest_payloads": manifest_payloads,
        "manifest_sha256": dict(T10_MANIFEST_SHA256),
        "randomness": expected_randomness,
        "training_recipe_signatures_validated": len(flattened_signatures),
    }


def _validate_metrics(
    arm: str, result: dict[str, Any], rows: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    metrics = result.get("matched_core_metrics")
    raw_rows = metrics.get("per_suite") if isinstance(metrics, dict) else None
    if not isinstance(raw_rows, list) or len(raw_rows) != 4:
        raise ValueError(f"{arm} matched_core_metrics must contain four suite rows")
    by_suite = {}
    for row in raw_rows:
        if not isinstance(row, dict) or row.get("dataset") in by_suite:
            raise ValueError(
                f"{arm} matched-core metric row is malformed or duplicated"
            )
        by_suite[row.get("dataset")] = row
    if set(by_suite) != set(SUITE_ORDER):
        raise ValueError(f"{arm} matched-core suite coverage differs")
    validated = []
    for index, suite in enumerate(SUITE_ORDER):
        update_label = UPDATE_LABELS[index]
        heldout_label = HELDOUT_LABELS[index]
        r_a = rows[update_label]["terminal_to_pre_ratio"]
        r_h = rows[heldout_label]["terminal_to_pre_ratio"]
        q = statistics.mean((r_a, r_h))
        raw = by_suite[suite]
        expected = {
            "rA": r_a,
            "rH": r_h,
            "q": q,
            "t9_four_position_q_median": T9_FOUR_POSITION_Q_MEDIANS[suite],
        }
        for key, value in expected.items():
            if raw.get(key) != value:
                raise ValueError(f"{arm}/{suite} matched-core metric differs: {key}")
        validated.append(
            {
                "dataset": suite,
                "heldout_label": heldout_label,
                "pre_heldout_action_loss": rows[heldout_label]["pre_action_loss"],
                "pre_update_action_loss": rows[update_label]["pre_action_loss"],
                "q": q,
                "rA": r_a,
                "rH": r_h,
                "t9_four_position_q_median": T9_FOUR_POSITION_Q_MEDIANS[suite],
                "terminal_heldout_action_loss": rows[heldout_label][
                    "terminal_action_loss"
                ],
                "terminal_update_action_loss": rows[update_label][
                    "terminal_action_loss"
                ],
                "update_label": update_label,
            }
        )
    return validated


def _validate_arm_result(
    arm: str,
    result: dict[str, Any],
    evidence: dict[str, Any],
    *,
    expected_source_commit: str,
    expected_runner_sha256: str,
) -> dict[str, Any]:
    verdict = T10_ARM_VERDICTS[arm]
    if result.get("schema_version") != T10_ARM_SCHEMA:
        raise ValueError(f"{arm} schema differs")
    if result.get("valid_run") is not True:
        raise ValueError(f"{arm} is not a valid frozen run")
    if any(
        result.get(key) != verdict
        for key in ("result", "scientific_verdict", "verdict")
    ):
        raise ValueError(f"{arm} typed arm verdict differs")
    if (
        result.get("execution_result") != "PASS"
        or result.get("harness_result") != "PASS"
    ):
        raise ValueError(f"{arm} execution or harness result differs")
    identity = result.get("identity")
    run = result.get("run")
    if not isinstance(identity, dict) or not isinstance(run, dict):
        raise ValueError(f"{arm} identity/run is malformed")
    if identity.get("arm") != arm or run.get("arm") != arm:
        raise ValueError(f"{arm} enumerated identity differs")
    if (
        identity.get("repo_commit") != expected_source_commit
        or identity.get("runner_sha256") != expected_runner_sha256
        or identity.get("config_sha256") != T10_CONFIG_SHA256
    ):
        raise ValueError(f"{arm} source/runner/config identity differs")
    nonce = _validate_hex(run.get("nonce"), 32, f"{arm} nonce")
    expected_root = (
        T10_ARM_NAMESPACE
        / expected_source_commit[:12]
        / f"{T10_ARM_ROOT_SLUGS[arm]}-{nonce}"
    )
    if run.get("root") != evidence["root"] or Path(evidence["root"]) != expected_root:
        raise ValueError(f"{arm} root namespace or embedded root differs")
    _validate_t9_pins(arm, result)
    scope = _validate_scope(arm, result)
    schedule = _validate_schedule(arm, result)
    runtime_identity = _validate_randomness_and_manifests(arm, result)
    rows = _extract_rows(result, arm)
    metrics = _validate_metrics(arm, result, rows)
    return {
        "arm": arm,
        "config_sha256": T10_CONFIG_SHA256,
        "input_verification": evidence,
        "metrics": metrics,
        "nonce": nonce,
        "result_path": evidence["result_path"],
        "result_sha256": evidence["result_sha256"],
        "root": evidence["root"],
        "rows": rows,
        "runner_sha256": expected_runner_sha256,
        "runtime_identity": runtime_identity,
        "schedule": schedule,
        "schema_version": T10_ARM_SCHEMA,
        "scope": scope,
        "source_commit": expected_source_commit,
        "verdict": verdict,
    }


def _tolerance_record(observed: float, expected: float) -> dict[str, Any]:
    absolute_error = abs(observed - expected)
    allowed_error = T7_REPRO_ABSOLUTE_TOLERANCE + (
        T7_REPRO_RELATIVE_TOLERANCE * abs(expected)
    )
    return {
        "absolute_error": absolute_error,
        "allowed_error": allowed_error,
        "expected": expected,
        "observed": observed,
        "passed": absolute_error <= allowed_error,
    }


def _validate_seq_t7_reproduction(seq: dict[str, Any]) -> dict[str, Any]:
    components = {}
    for label in ALL_LABELS:
        components[label] = {
            "pre": _tolerance_record(
                seq["rows"][label]["pre_action_loss"], T7_PRE_ACTION_LOSSES[label]
            ),
            "terminal": _tolerance_record(
                seq["rows"][label]["terminal_action_loss"],
                T7_TERMINAL_ACTION_LOSSES[label],
            ),
        }
    reproduced = all(
        record[stage]["passed"]
        for record in components.values()
        for stage in ("pre", "terminal")
    )
    report = {"components": components, "reproduced": reproduced}
    if not reproduced:
        failures = [
            f"{label}/{stage}"
            for label, record in components.items()
            for stage in ("pre", "terminal")
            if not record[stage]["passed"]
        ]
        raise ValueError(
            "SEQ bridge does not reproduce frozen T7 pre+terminal losses: "
            + ",".join(failures)
        )
    return report


def _classify_joint_ratios(per_suite: list[dict[str, Any]]) -> str:
    if len(per_suite) != 4 or [row.get("dataset") for row in per_suite] != list(
        SUITE_ORDER
    ):
        raise ValueError("JOINT suite rows differ")
    r_a_values = [
        _finite_positive(row.get("rA"), f"JOINT/{row.get('dataset')}/rA")
        for row in per_suite
    ]
    r_h_values = [
        _finite_positive(row.get("rH"), f"JOINT/{row.get('dataset')}/rH")
        for row in per_suite
    ]
    if all(value < 1.0 for value in (*r_a_values, *r_h_values)):
        return "T10_BALANCED_JOINT_SIMULTANEOUS_RETENTION_SUPPORTS_OVERWRITE"
    if any(value >= 1.0 for value in r_a_values):
        return "T10_MATCHED_DOSE_JOINT_TRAINING_FIT_FAILURE"
    return "T10_JOINT_FIT_WITH_HELDOUT_GAP"


def _validate_paired_arm_binding(seq: dict[str, Any], joint: dict[str, Any]) -> None:
    if seq["nonce"] == joint["nonce"]:
        raise ValueError("SEQ and JOINT nonces must be distinct")
    if seq["root"] == joint["root"]:
        raise ValueError("SEQ and JOINT roots must be distinct")
    for key, label in (
        ("common_identity", "common identity"),
        ("assets", "assets"),
        ("manifest_payloads", "manifest payloads"),
        ("manifest_sha256", "manifest SHA256 pins"),
        ("randomness", "randomness contract"),
    ):
        if seq["runtime_identity"][key] != joint["runtime_identity"][key]:
            raise ValueError(f"SEQ and JOINT {label} differ")


def _assemble_paired_result(
    seq: dict[str, Any], joint: dict[str, Any]
) -> dict[str, Any]:
    seq_reproduction = _validate_seq_t7_reproduction(seq)
    raw_metrics = {"SEQ": seq["metrics"], "JOINT": joint["metrics"]}
    verdict = _classify_joint_ratios(joint["metrics"])
    pareto_rows = []
    for seq_row, joint_row in zip(seq["metrics"], joint["metrics"], strict=True):
        if seq_row["dataset"] != joint_row["dataset"]:
            raise ValueError("SEQ/JOINT suite order differs")
        weak_both = (
            joint_row["rA"] <= seq_row["rA"] and joint_row["rH"] <= seq_row["rH"]
        )
        strict_one = joint_row["rA"] < seq_row["rA"] or joint_row["rH"] < seq_row["rH"]
        pareto_rows.append(
            {
                "dataset": seq_row["dataset"],
                "joint_below_t9_median": (
                    joint_row["q"] < joint_row["t9_four_position_q_median"]
                ),
                "joint_minus_seq_q": joint_row["q"] - seq_row["q"],
                "joint_pareto_dominates_seq": weak_both and strict_one,
                "joint_rA_le_seq": joint_row["rA"] <= seq_row["rA"],
                "joint_rH_le_seq": joint_row["rH"] <= seq_row["rH"],
                "joint_q": joint_row["q"],
                "seq_q": seq_row["q"],
                "t9_four_position_q_median": joint_row["t9_four_position_q_median"],
            }
        )
    diagnostics = {
        "excluded_from_primary_classifier": True,
        "joint_below_t9_median_count": sum(
            row["joint_below_t9_median"] for row in pareto_rows
        ),
        "joint_pareto_dominates_seq_count": sum(
            row["joint_pareto_dominates_seq"] for row in pareto_rows
        ),
        "per_suite": pareto_rows,
        "t9_four_position_q_medians": T9_FOUR_POSITION_Q_MEDIANS,
    }
    return {
        "diagnostics": diagnostics,
        "primary_classifier": {
            "joint_all_eight_ratios_strictly_below_one": all(
                row[key] < 1.0 for row in joint["metrics"] for key in ("rA", "rH")
            ),
            "joint_any_update_ratio_at_least_one": any(
                row["rA"] >= 1.0 for row in joint["metrics"]
            ),
            "ordered_rule": [
                "all eight JOINT rA/rH ratios < 1",
                "otherwise any JOINT rA ratio >= 1",
                "otherwise heldout gap",
            ],
            "typed_verdict": verdict,
        },
        "raw_metrics": raw_metrics,
        "seq_t7_reproduction": seq_reproduction,
        "typed_verdict": verdict,
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
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
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
    entries = list(root.iterdir())
    if not entries:
        _write_terminal_report(
            marker,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "error_type": type(error).__name__,
                "result": "FAIL",
                "schema_version": T10_AGGREGATE_FAILURE_SCHEMA,
            },
        )
    elif len(entries) != 1 or entries[0].name not in {"FAILED.json", "RESULT.json"}:
        raise RuntimeError("aggregate failure root contains unexpected entries")
    _freeze_run_root(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-result", required=True, type=Path)
    parser.add_argument("--seq-result-sha256", required=True)
    parser.add_argument("--joint-result", required=True, type=Path)
    parser.add_argument("--joint-result-sha256", required=True)
    parser.add_argument("--expected-arm-source-commit", required=True)
    parser.add_argument("--expected-arm-runner-sha256", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    args = parser.parse_args()

    nonce = _validate_hex(args.nonce, 32, "aggregate nonce")
    arm_source_commit = _validate_hex(
        args.expected_arm_source_commit, 40, "T10 arm source commit"
    )
    arm_runner_sha256 = _validate_hex(
        args.expected_arm_runner_sha256, 64, "T10 arm runner SHA256"
    )
    if FUTURE_T10_ARM_SOURCE_COMMIT.startswith("PLACEHOLDER_") or (
        FUTURE_T10_ARM_RUNNER_SHA256.startswith("PLACEHOLDER_")
    ):
        raise RuntimeError("T10 future arm pins have not been materialized")
    if arm_source_commit != FUTURE_T10_ARM_SOURCE_COMMIT:
        raise ValueError("T10 arm source commit differs from materialized pin")
    if arm_runner_sha256 != FUTURE_T10_ARM_RUNNER_SHA256:
        raise ValueError("T10 arm runner SHA256 differs from materialized pin")
    aggregate_commit = _validate_hex(
        args.expected_repo_commit, 40, "aggregate source commit"
    )
    aggregate_runner_sha256 = _validate_hex(
        args.expected_runner_sha256, 64, "aggregate runner SHA256"
    )
    expected_root = (
        T10_AGGREGATE_NAMESPACE
        / aggregate_commit[:12]
        / f"libero-t10-exact-balanced-joint-combined-{nonce}"
    )
    if args.run_root.expanduser() != expected_root:
        raise ValueError(
            f"aggregate root differs: expected={expected_root} got={args.run_root}"
        )

    global _ACTIVE_ROOT
    run_root = _create_run_root(expected_root)
    _ACTIVE_ROOT = run_root
    aggregate_identity = _aggregate_source_identity(
        aggregate_commit, aggregate_runner_sha256
    )

    inputs = {
        "SEQ": (args.seq_result.expanduser(), args.seq_result_sha256),
        "JOINT": (args.joint_result.expanduser(), args.joint_result_sha256),
    }
    if len({str(path) for path, _sha256 in inputs.values()}) != 2:
        raise ValueError("SEQ and JOINT RESULT paths must be distinct")
    if args.seq_result_sha256 == args.joint_result_sha256:
        raise ValueError("SEQ and JOINT RESULT hashes must be distinct")
    arms = {}
    ownership = set()
    for arm in ("SEQ", "JOINT"):
        path, expected_sha256 = inputs[arm]
        result, evidence = _load_frozen_result(path, expected_sha256, arm=arm)
        ownership.add((evidence["uid"], evidence["gid"]))
        arms[arm] = _validate_arm_result(
            arm,
            result,
            evidence,
            expected_source_commit=arm_source_commit,
            expected_runner_sha256=arm_runner_sha256,
        )
    if len(ownership) != 1:
        raise PermissionError("SEQ and JOINT frozen roots have different owners/groups")
    _validate_paired_arm_binding(arms["SEQ"], arms["JOINT"])

    assembly = _assemble_paired_result(arms["SEQ"], arms["JOINT"])
    verdict = assembly["typed_verdict"]
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
        "identity": {
            **aggregate_identity,
            "config_sha256": T10_CONFIG_SHA256,
            "t10_arm_runner_sha256": arm_runner_sha256,
            "t10_arm_source_commit": arm_source_commit,
            "t9_aggregate_result_sha256": T9_AGGREGATE_RESULT_SHA256,
            "t9_aggregate_runner_sha256": T9_AGGREGATE_RUNNER_SHA256,
            "t9_aggregate_source_commit": T9_AGGREGATE_SOURCE_COMMIT,
        },
        "input_verification": {
            arm: artifact["input_verification"] for arm, artifact in arms.items()
        },
        "result": verdict,
        "run": {"nonce": nonce, "root": str(run_root)},
        "schema_version": T10_AGGREGATE_SCHEMA,
        "scope": {
            "formal_training_or_evaluation_executed": False,
            "gpu_or_model_executed": False,
            "input_result_count": 2,
            "read_only_evidence_assembly": True,
            "sana_base_construction_checkpoint_loaded_by_input_arms": True,
            "sana_wam_training_checkpoint_loaded_or_saved": False,
        },
        "scientific_verdict": verdict,
        "valid_run": True,
        "verdict": verdict,
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
                    "failed to terminalize T10 aggregate root: "
                    f"{type(freeze_error).__name__}: {freeze_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
