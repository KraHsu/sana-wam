from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_libero_ar_t10_exact_balanced_joint.py"
)
SPEC = importlib.util.spec_from_file_location("t10_aggregate_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
t10 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t10
SPEC.loader.exec_module(t10)


ARM_SOURCE_COMMIT = "a" * 40
ARM_RUNNER_SHA256 = "b" * 64
SEQ_NONCE = "1" * 32
JOINT_NONCE = "2" * 32
PRODUCTION_MANIFEST_SHA256 = dict(t10.T10_MANIFEST_SHA256)
FIXTURE_MANIFEST_PAYLOADS = {
    "eligible": {"schema_version": "eligible-fixture-v1"},
    "selection": {"schema_version": "selection-fixture-v1"},
    "update": {"schema_version": "update-fixture-v1"},
}
FIXTURE_MANIFEST_SHA256 = {
    name: hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for name, payload in FIXTURE_MANIFEST_PAYLOADS.items()
}


def _install_fixture_manifest_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t10, "T10_MANIFEST_SHA256", FIXTURE_MANIFEST_SHA256)


def _joint_metrics(
    r_a: tuple[float, float, float, float],
    r_h: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    return [
        {"dataset": suite, "rA": r_a[index], "rH": r_h[index]}
        for index, suite in enumerate(t10.SUITE_ORDER)
    ]


def _arm_result(
    arm: str,
    *,
    source_commit: str = ARM_SOURCE_COMMIT,
    runner_sha256: str = ARM_RUNNER_SHA256,
    nonce: str | None = None,
    ratios: tuple[float, float, float, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if nonce is None:
        nonce = SEQ_NONCE if arm == "SEQ" else JOINT_NONCE
    if ratios is None:
        ratios = (0.72, 0.73, 0.74, 0.75)
    root = str(
        t10.T10_ARM_NAMESPACE
        / source_commit[:12]
        / f"{t10.T10_ARM_ROOT_SLUGS[arm]}-{nonce}"
    )

    rows: dict[str, dict[str, Any]] = {}
    for labels, exposure, coefficient in (
        (t10.UPDATE_LABELS, 20, 5.0),
        (t10.HELDOUT_LABELS, 0, 0.0),
    ):
        for index, label in enumerate(labels):
            before = t10.T7_PRE_ACTION_LOSSES[label]
            if arm == "SEQ":
                terminal = t10.T7_TERMINAL_ACTION_LOSSES[label]
            else:
                terminal = before * ratios[index]
            rows[label] = {
                "after_action_loss": terminal,
                "before_action_loss": before,
                "dataset": t10.LABEL_TO_SUITE[label],
                "effective_cumulative_loss_weight": coefficient,
                "label": label,
                "post_to_pre_ratio": terminal / before,
                "update_exposure_count": exposure,
            }

    per_macro = []
    per_step = []
    for index in range(20):
        weights = t10._expected_weights(arm, index)
        per_macro.append({"macro_step": index + 1, "weights": weights})
        per_step.append(
            {
                "component_datasets": {
                    label: t10.LABEL_TO_SUITE[label] for label in t10.UPDATE_LABELS
                },
                "component_episode_indices": {label: 0 for label in t10.UPDATE_LABELS},
                "component_losses": {label: 1.0 for label in t10.UPDATE_LABELS},
                "component_recipe_signatures": {
                    label: "recipe" for label in t10.UPDATE_LABELS
                },
                "component_task_indices": {
                    label: sample_index
                    for sample_index, label in enumerate(t10.UPDATE_LABELS)
                },
                "labels": list(t10.UPDATE_LABELS),
                "loss_weights": weights,
                "macro_step": index + 1,
            }
        )

    per_suite = []
    for index, suite in enumerate(t10.SUITE_ORDER):
        update_label = t10.UPDATE_LABELS[index]
        heldout_label = t10.HELDOUT_LABELS[index]
        r_a = rows[update_label]["post_to_pre_ratio"]
        r_h = rows[heldout_label]["post_to_pre_ratio"]
        q = (r_a + r_h) / 2.0
        per_suite.append(
            {
                "dataset": suite,
                "q": q,
                "rA": r_a,
                "rH": r_h,
                "t9_four_position_q_median": t10.T9_FOUR_POSITION_Q_MEDIANS[suite],
            }
        )

    no_mutation_evidence = []
    for index in range(20):
        component_sha256 = {
            component: hashlib.sha256(
                f"{component}:{index + 1}".encode("utf-8")
            ).hexdigest()
            for component in t10.T10_NO_MUTATION_COMPONENTS
        }
        composite_sha256 = hashlib.sha256(
            f"composite:{index + 1}".encode("utf-8")
        ).hexdigest()
        no_mutation_evidence.append(
            {
                "component_state_sha256": component_sha256,
                "components_unchanged": {
                    component: True for component in t10.T10_NO_MUTATION_COMPONENTS
                },
                "macro_step": index + 1,
                "post_microback_pre_optimizer_state_sha256": composite_sha256,
                "pre_microback_state_sha256": composite_sha256,
                "state_unchanged": True,
            }
        )
    per_macro_recipe_signatures = [
        {label: t10.T10_FIXED_RECIPE_SIGNATURE_SHA256 for label in t10.UPDATE_LABELS}
        for _index in range(20)
    ]
    for index, signatures in enumerate(per_macro_recipe_signatures):
        per_step[index]["component_recipe_signatures"] = signatures
    manifest_payloads = copy.deepcopy(FIXTURE_MANIFEST_PAYLOADS)
    assets = {
        "config_sha256": t10.T10_CONFIG_SHA256,
        "external_files_manifest_sha256": "e" * 64,
        "role_files_post_sample_manifest_sha256": "f" * 64,
        "stats_population_sha256": t10.T10_STATS_POPULATION_SHA256,
        "stats_sha256": t10.T10_STATS_SHA256,
        "suite_metadata_post_sample_manifest_sha256": "9" * 64,
    }
    verdict = t10.T10_ARM_VERDICTS[arm]
    result = {
        "assets": assets,
        "direct_predecessor": {
            "result_sha256": t10.T9_AGGREGATE_RESULT_SHA256,
            "root": t10.T9_AGGREGATE_ROOT,
            "runner_sha256": t10.T9_AGGREGATE_RUNNER_SHA256,
            "source_commit": t10.T9_AGGREGATE_SOURCE_COMMIT,
            "stage": "T9_AGGREGATE",
        },
        "execution_result": "PASS",
        "fresh_heldout_transfer": {
            "per_sample": [rows[label] for label in t10.HELDOUT_LABELS]
        },
        "harness_result": "PASS",
        "identity": {
            "arm": arm,
            "config_sha256": t10.T10_CONFIG_SHA256,
            "cross_suite_metadata_post_sample_sha256": assets[
                "suite_metadata_post_sample_manifest_sha256"
            ],
            "eligible_manifest_sha256": t10.T10_MANIFEST_SHA256["eligible"],
            "external_assets_sha256": assets["external_files_manifest_sha256"],
            "repo_commit": source_commit,
            "role_assets_post_sample_sha256": assets[
                "role_files_post_sample_manifest_sha256"
            ],
            "runner_sha256": runner_sha256,
            "sana_commit": t10.T10_SANA_COMMIT,
            "selection_manifest_sha256": t10.T10_MANIFEST_SHA256["selection"],
            "stats_population_sha256": t10.T10_STATS_POPULATION_SHA256,
            "stats_sha256": t10.T10_STATS_SHA256,
            "t9_aggregate_predecessor_result_sha256": (t10.T9_AGGREGATE_RESULT_SHA256),
            "t9_aggregate_predecessor_runner_sha256": (t10.T9_AGGREGATE_RUNNER_SHA256),
            "t9_aggregate_predecessor_source_commit": (t10.T9_AGGREGATE_SOURCE_COMMIT),
            "t7_predecessor_result_sha256": t10.T7_RESULT_SHA256,
            "t7_predecessor_runner_sha256": t10.T7_RUNNER_SHA256,
            "t7_predecessor_source_commit": t10.T7_SOURCE_COMMIT,
            "update_sample_manifest_sha256": t10.T10_MANIFEST_SHA256["update"],
        },
        "matched_core_metrics": {"per_suite": per_suite},
        "matched_core_schedule": {
            "backward_calls_per_macro_step": 4,
            "effective_cumulative_loss_weights": {
                label: 5.0 for label in t10.UPDATE_LABELS
            },
            "forward_exposures_per_sample": 20,
            "forward_order_within_macro_step": list(t10.UPDATE_LABELS),
            "macro_steps": 20,
            "no_intra_macro_mutation": {
                "all_macro_steps_verified": True,
                "evidence": no_mutation_evidence,
                "evidence_scope": list(t10.T10_NO_MUTATION_EVIDENCE_SCOPE),
                "macro_steps_verified": 20,
                "verification_point": t10.T10_NO_MUTATION_VERIFICATION_POINT,
            },
            "optimizer_steps_per_macro_step": 1,
            "parameters_constant_across_four_component_backward_calls": True,
            "per_macro_loss_weights": per_macro,
        },
        "per_step": per_step,
        "randomness": {
            "eligible_same_task_probe_manifest": manifest_payloads["eligible"],
            "eligible_same_task_probe_manifest_sha256": (
                t10.T10_MANIFEST_SHA256["eligible"]
            ),
            "initialization_seed": t10.T10_INITIALIZATION_SEED,
            "per_macro_recipe_signatures": per_macro_recipe_signatures,
            "rng_state_restored_after_every_forward": True,
            "selected_same_task_probe_manifest": manifest_payloads["selection"],
            "selected_same_task_probe_manifest_sha256": (
                t10.T10_MANIFEST_SHA256["selection"]
            ),
            "total_recipe_signatures": 96,
            "training_loss_recipe_seed": t10.T10_TRAINING_LOSS_RECIPE_SEED,
            "training_recipe_signature_sha256": (t10.T10_FIXED_RECIPE_SIGNATURE_SHA256),
            "training_recipe_signatures": 80,
            "unique_loss_recipe_signature_count": 1,
            "unique_training_recipe_signatures": 1,
            "update_sample_manifest": manifest_payloads["update"],
            "update_sample_manifest_sha256": t10.T10_MANIFEST_SHA256["update"],
        },
        "result": verdict,
        "run": {"arm": arm, "nonce": nonce, "root": root},
        "schema_version": t10.T10_ARM_SCHEMA,
        "scientific_verdict": verdict,
        "scope": {
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
        },
        "update_samples": {"per_sample": [rows[label] for label in t10.UPDATE_LABELS]},
        "valid_run": True,
        "verdict": verdict,
    }
    evidence = {
        "file_mode": "0400",
        "gid": 1000,
        "result_path": f"{root}/RESULT.json",
        "result_sha256": ("c" if arm == "SEQ" else "d") * 64,
        "root": root,
        "root_mode": "0500",
        "single_result_file": True,
        "size_bytes": 123,
        "uid": 1000,
    }
    return result, evidence


@pytest.mark.parametrize(
    ("r_a", "r_h", "expected"),
    [
        (
            (0.4, 0.5, 0.6, 0.999999),
            (0.7, 0.8, 0.9, 0.3),
            "T10_BALANCED_JOINT_SIMULTANEOUS_RETENTION_SUPPORTS_OVERWRITE",
        ),
        (
            (0.4, 1.0, 0.6, 0.7),
            (0.7, 1.3, 0.9, 0.3),
            "T10_MATCHED_DOSE_JOINT_TRAINING_FIT_FAILURE",
        ),
        (
            (0.4, 0.5, 0.6, 0.7),
            (0.7, 1.0, 0.9, 0.3),
            "T10_JOINT_FIT_WITH_HELDOUT_GAP",
        ),
    ],
)
def test_primary_classifier_covers_all_three_ordered_branches(
    r_a: tuple[float, float, float, float],
    r_h: tuple[float, float, float, float],
    expected: str,
) -> None:
    assert t10._classify_joint_ratios(_joint_metrics(r_a, r_h)) == expected


def test_primary_classifier_rejects_wrong_order_and_nonpositive_ratio() -> None:
    rows = _joint_metrics((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="suite rows differ"):
        t10._classify_joint_ratios(rows)

    rows = _joint_metrics((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.0))
    with pytest.raises(ValueError, match="strictly positive"):
        t10._classify_joint_ratios(rows)


def test_seq_exact_t7_reproduction_accepts_pin_and_rejects_drift() -> None:
    seq = {
        "rows": {
            label: {
                "pre_action_loss": t10.T7_PRE_ACTION_LOSSES[label],
                "terminal_action_loss": t10.T7_TERMINAL_ACTION_LOSSES[label],
            }
            for label in t10.ALL_LABELS
        }
    }
    report = t10._validate_seq_t7_reproduction(seq)
    assert report["reproduced"] is True
    assert all(
        stage["passed"]
        for component in report["components"].values()
        for stage in component.values()
    )

    drifted = copy.deepcopy(seq)
    drifted["rows"]["A2"]["terminal_action_loss"] += 1.0e-3
    with pytest.raises(ValueError, match="A2/terminal"):
        t10._validate_seq_t7_reproduction(drifted)


def test_arm_validation_binds_source_runner_config_root_and_t9_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fixture_manifest_pins(monkeypatch)
    result, evidence = _arm_result("SEQ")
    validated = t10._validate_arm_result(
        "SEQ",
        result,
        evidence,
        expected_source_commit=ARM_SOURCE_COMMIT,
        expected_runner_sha256=ARM_RUNNER_SHA256,
    )
    assert validated["root"] == evidence["root"]
    assert validated["nonce"] == SEQ_NONCE
    assert validated["source_commit"] == ARM_SOURCE_COMMIT
    assert validated["runner_sha256"] == ARM_RUNNER_SHA256

    wrong_source = copy.deepcopy(result)
    wrong_source["identity"]["repo_commit"] = "e" * 40
    with pytest.raises(ValueError, match="source/runner/config identity differs"):
        t10._validate_arm_result(
            "SEQ",
            wrong_source,
            evidence,
            expected_source_commit=ARM_SOURCE_COMMIT,
            expected_runner_sha256=ARM_RUNNER_SHA256,
        )

    wrong_root = copy.deepcopy(result)
    wrong_root["run"]["root"] += "-different"
    with pytest.raises(ValueError, match="root namespace or embedded root differs"):
        t10._validate_arm_result(
            "SEQ",
            wrong_root,
            evidence,
            expected_source_commit=ARM_SOURCE_COMMIT,
            expected_runner_sha256=ARM_RUNNER_SHA256,
        )

    wrong_predecessor = copy.deepcopy(result)
    wrong_predecessor["direct_predecessor"]["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="direct T9 pin differs"):
        t10._validate_arm_result(
            "SEQ",
            wrong_predecessor,
            evidence,
            expected_source_commit=ARM_SOURCE_COMMIT,
            expected_runner_sha256=ARM_RUNNER_SHA256,
        )


def test_frozen_runtime_identity_pins_are_exact() -> None:
    assert PRODUCTION_MANIFEST_SHA256 == {
        "eligible": (
            "0b6bfe4c51e8b9e2ac3f538bac7853b5e1e871ea96dda5df6976ee27b5b25370"
        ),
        "selection": (
            "e969e76e4bcb8a3c2b0b3fffff35e0ec10c478e45cb6535b7bd91061d3a6a7ee"
        ),
        "update": ("5199c87af876c437ecec35b57da3e118a2924c44a9bb454c9846e7c0ca968aa6"),
    }
    assert t10.T10_SANA_COMMIT == "16b9cec673e3335724ba2d8db25de7f9ed229292"
    assert t10.T7_SOURCE_COMMIT == "d19109a2314f8e7186571afed6b4acd816d7cfab"
    assert (
        t10.T7_RUNNER_SHA256
        == "b6e764bf3d3566bf3d1fe5e2e3802dafef691f6a0162568337b49ca1a183310b"
    )
    assert (
        t10.T7_RESULT_SHA256
        == "9f5181a30cd0d6b676f09315244f7560cd3902004f5f17ca833330e33cb44d3a"
    )


def test_no_intra_macro_mutation_requires_full_runtime_evidence() -> None:
    result, _evidence = _arm_result("SEQ")
    validated = t10._validate_schedule("SEQ", result)
    no_mutation = validated["no_intra_macro_mutation"]
    assert no_mutation["macro_steps_verified"] == 20
    assert [row["macro_step"] for row in no_mutation["evidence"]] == list(range(1, 21))
    assert all(
        set(row["components_unchanged"]) == set(t10.T10_NO_MUTATION_COMPONENTS)
        and all(row["components_unchanged"].values())
        and row["pre_microback_state_sha256"]
        == row["post_microback_pre_optimizer_state_sha256"]
        for row in no_mutation["evidence"]
    )

    missing_step = copy.deepcopy(result)
    missing_step["matched_core_schedule"]["no_intra_macro_mutation"]["evidence"].pop()
    with pytest.raises(ValueError, match="evidence coverage differs"):
        t10._validate_schedule("SEQ", missing_step)

    changed_component = copy.deepcopy(result)
    changed_component["matched_core_schedule"]["no_intra_macro_mutation"]["evidence"][
        7
    ]["components_unchanged"]["optimizer_state"] = False
    with pytest.raises(ValueError, match="component flags differ at step 8"):
        t10._validate_schedule("SEQ", changed_component)

    changed_composite = copy.deepcopy(result)
    changed_composite["matched_core_schedule"]["no_intra_macro_mutation"]["evidence"][
        3
    ]["post_microback_pre_optimizer_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="composite state differs at step 4"):
        t10._validate_schedule("SEQ", changed_composite)

    wrong_point = copy.deepcopy(result)
    wrong_point["matched_core_schedule"]["no_intra_macro_mutation"][
        "verification_point"
    ] = "after_optimizer_step"
    with pytest.raises(ValueError, match="summary differs: verification_point"):
        t10._validate_schedule("SEQ", wrong_point)


def test_randomness_and_manifest_payloads_are_recomputed_and_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fixture_manifest_pins(monkeypatch)
    result, _evidence = _arm_result("SEQ")
    validated = t10._validate_randomness_and_manifests("SEQ", result)
    assert validated["training_recipe_signatures_validated"] == 80
    assert validated["randomness"]["initialization_seed"] == 20260806
    assert validated["randomness"]["training_loss_recipe_seed"] == 20260826
    assert validated["randomness"]["unique_training_recipe_signatures"] == 1

    wrong_seed = copy.deepcopy(result)
    wrong_seed["randomness"]["initialization_seed"] += 1
    with pytest.raises(ValueError, match="randomness contract differs"):
        t10._validate_randomness_and_manifests("SEQ", wrong_seed)

    wrong_count = copy.deepcopy(result)
    wrong_count["randomness"]["training_recipe_signatures"] = 79
    with pytest.raises(ValueError, match="training_recipe_signatures"):
        t10._validate_randomness_and_manifests("SEQ", wrong_count)

    wrong_signature = copy.deepcopy(result)
    wrong_signature["randomness"]["per_macro_recipe_signatures"][4]["A2"] = "0" * 64
    with pytest.raises(ValueError, match="recipe signature differs at step 5"):
        t10._validate_randomness_and_manifests("SEQ", wrong_signature)

    forged_payload = copy.deepcopy(result)
    forged_payload["randomness"]["update_sample_manifest"]["forged"] = True
    with pytest.raises(ValueError, match="payload SHA256 differs"):
        t10._validate_randomness_and_manifests("SEQ", forged_payload)

    wrong_self_report = copy.deepcopy(result)
    wrong_self_report["identity"]["eligible_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity eligible manifest SHA256 differs"):
        t10._validate_randomness_and_manifests("SEQ", wrong_self_report)

    wrong_t7 = copy.deepcopy(result)
    wrong_t7["identity"]["t7_predecessor_result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime identity pin differs"):
        t10._validate_randomness_and_manifests("SEQ", wrong_t7)


def test_paired_binding_requires_common_identity_assets_and_manifest_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fixture_manifest_pins(monkeypatch)
    seq_result, seq_evidence = _arm_result("SEQ")
    joint_result, joint_evidence = _arm_result("JOINT")
    seq = t10._validate_arm_result(
        "SEQ",
        seq_result,
        seq_evidence,
        expected_source_commit=ARM_SOURCE_COMMIT,
        expected_runner_sha256=ARM_RUNNER_SHA256,
    )
    joint = t10._validate_arm_result(
        "JOINT",
        joint_result,
        joint_evidence,
        expected_source_commit=ARM_SOURCE_COMMIT,
        expected_runner_sha256=ARM_RUNNER_SHA256,
    )
    t10._validate_paired_arm_binding(seq, joint)

    for key, match in (
        ("common_identity", "common identity differ"),
        ("assets", "assets differ"),
        ("manifest_payloads", "manifest payloads differ"),
    ):
        changed = copy.deepcopy(joint)
        changed["runtime_identity"][key]["forged"] = True
        with pytest.raises(ValueError, match=match):
            t10._validate_paired_arm_binding(seq, changed)


def test_paired_assembly_requires_bound_suite_order_and_preserves_raw_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fixture_manifest_pins(monkeypatch)
    seq_result, seq_evidence = _arm_result("SEQ")
    joint_result, joint_evidence = _arm_result("JOINT", ratios=(0.4, 0.5, 0.6, 0.7))
    seq = t10._validate_arm_result(
        "SEQ",
        seq_result,
        seq_evidence,
        expected_source_commit=ARM_SOURCE_COMMIT,
        expected_runner_sha256=ARM_RUNNER_SHA256,
    )
    joint = t10._validate_arm_result(
        "JOINT",
        joint_result,
        joint_evidence,
        expected_source_commit=ARM_SOURCE_COMMIT,
        expected_runner_sha256=ARM_RUNNER_SHA256,
    )
    assembled = t10._assemble_paired_result(seq, joint)
    assert assembled["seq_t7_reproduction"]["reproduced"] is True
    assert assembled["raw_metrics"]["SEQ"] == seq["metrics"]
    assert assembled["raw_metrics"]["JOINT"] == joint["metrics"]
    assert (
        assembled["typed_verdict"]
        == "T10_BALANCED_JOINT_SIMULTANEOUS_RETENTION_SUPPORTS_OVERWRITE"
    )

    reordered = copy.deepcopy(joint)
    reordered["metrics"][0], reordered["metrics"][1] = (
        reordered["metrics"][1],
        reordered["metrics"][0],
    )
    with pytest.raises(ValueError, match="suite (?:order differs|rows differ)"):
        t10._assemble_paired_result(seq, reordered)


def _main_argv(
    *,
    seq_result: Path,
    joint_result: Path,
    seq_sha256: str,
    joint_sha256: str,
    run_root: Path,
) -> list[str]:
    return [
        str(SCRIPT),
        "--seq-result",
        str(seq_result),
        "--seq-result-sha256",
        seq_sha256,
        "--joint-result",
        str(joint_result),
        "--joint-result-sha256",
        joint_sha256,
        "--expected-arm-source-commit",
        ARM_SOURCE_COMMIT,
        "--expected-arm-runner-sha256",
        ARM_RUNNER_SHA256,
        "--run-root",
        str(run_root),
        "--nonce",
        "3" * 32,
        "--expected-repo-commit",
        "4" * 40,
        "--expected-runner-sha256",
        "5" * 64,
    ]


def test_placeholder_pins_fail_closed_before_root_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(t10, "FUTURE_T10_ARM_SOURCE_COMMIT", "PLACEHOLDER_SOURCE")
    monkeypatch.setattr(t10, "FUTURE_T10_ARM_RUNNER_SHA256", "PLACEHOLDER_RUNNER")
    monkeypatch.setattr(
        t10,
        "_create_run_root",
        lambda _path: pytest.fail("placeholder draft must not create a root"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _main_argv(
            seq_result=tmp_path / "seq" / "RESULT.json",
            joint_result=tmp_path / "joint" / "RESULT.json",
            seq_sha256="6" * 64,
            joint_sha256="7" * 64,
            run_root=tmp_path / "aggregate",
        ),
    )
    with pytest.raises(
        RuntimeError, match="future arm pins have not been materialized"
    ):
        t10.main()


@pytest.mark.parametrize("duplicate", ["path", "hash", "nonce", "root", "owner"])
def test_main_rejects_unbound_or_aliased_arm_pair(
    duplicate: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(t10, "FUTURE_T10_ARM_SOURCE_COMMIT", ARM_SOURCE_COMMIT)
    monkeypatch.setattr(t10, "FUTURE_T10_ARM_RUNNER_SHA256", ARM_RUNNER_SHA256)
    aggregate_namespace = tmp_path / "aggregate-namespace"
    monkeypatch.setattr(t10, "T10_AGGREGATE_NAMESPACE", aggregate_namespace)
    expected_root = (
        aggregate_namespace
        / ("4" * 12)
        / f"libero-t10-exact-balanced-joint-combined-{'3' * 32}"
    )
    created_root = tmp_path / "created-root"
    created_root.mkdir()
    monkeypatch.setattr(t10, "_create_run_root", lambda path: created_root)
    monkeypatch.setattr(
        t10,
        "_aggregate_source_identity",
        lambda commit, runner: {
            "repo_commit": commit,
            "runner_path": str(SCRIPT),
            "runner_sha256": runner,
        },
    )

    seq_path = tmp_path / "seq" / "RESULT.json"
    joint_path = seq_path if duplicate == "path" else tmp_path / "joint" / "RESULT.json"
    seq_sha = "6" * 64
    joint_sha = seq_sha if duplicate == "hash" else "7" * 64

    def fake_load(path: Path, expected_sha256: str, *, arm: str):
        owner = 1001 if duplicate == "owner" and arm == "JOINT" else 1000
        return {"arm": arm}, {"uid": owner, "gid": 1000}

    def fake_validate(
        arm: str,
        _result: dict[str, Any],
        _evidence: dict[str, Any],
        *,
        expected_source_commit: str,
        expected_runner_sha256: str,
    ) -> dict[str, Any]:
        assert expected_source_commit == ARM_SOURCE_COMMIT
        assert expected_runner_sha256 == ARM_RUNNER_SHA256
        nonce = SEQ_NONCE if arm == "SEQ" else JOINT_NONCE
        root = f"/{arm.lower()}"
        if duplicate == "nonce" and arm == "JOINT":
            nonce = SEQ_NONCE
        if duplicate == "root" and arm == "JOINT":
            root = "/seq"
        return {"nonce": nonce, "root": root}

    monkeypatch.setattr(t10, "_load_frozen_result", fake_load)
    monkeypatch.setattr(t10, "_validate_arm_result", fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        _main_argv(
            seq_result=seq_path,
            joint_result=joint_path,
            seq_sha256=seq_sha,
            joint_sha256=joint_sha,
            run_root=expected_root,
        ),
    )
    try:
        match = {
            "path": "paths must be distinct",
            "hash": "hashes must be distinct",
            "nonce": "nonces must be distinct",
            "root": "roots must be distinct",
            "owner": "different owners/groups",
        }[duplicate]
        with pytest.raises((ValueError, PermissionError), match=match):
            t10.main()
    finally:
        t10._ACTIVE_ROOT = None


def test_aggregate_source_identity_is_bound_to_commit_and_runner_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = tmp_path / "runner.py"
    runner.write_bytes(b"frozen aggregate runner\n")
    commit = "8" * 40
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    monkeypatch.setattr(t10, "__file__", str(runner))
    monkeypatch.setattr(t10, "_repo_commit", lambda: commit)

    identity = t10._aggregate_source_identity(commit, runner_sha)
    assert identity == {
        "repo_commit": commit,
        "runner_path": str(runner.resolve()),
        "runner_sha256": runner_sha,
    }
    with pytest.raises(RuntimeError, match="aggregate source identity differs"):
        t10._aggregate_source_identity("9" * 40, runner_sha)
    with pytest.raises(RuntimeError, match="aggregate source identity differs"):
        t10._aggregate_source_identity(commit, "0" * 64)

    symlink = tmp_path / "runner-link.py"
    symlink.symlink_to(runner)
    monkeypatch.setattr(t10, "__file__", str(symlink))
    with pytest.raises(RuntimeError, match="regular non-symlink"):
        t10._aggregate_source_identity(commit, runner_sha)


def test_canonical_json_round_trip_and_strict_rejections() -> None:
    value = {"z": [2, 1], "a": {"truth": True}}
    payload = t10._canonical_json_bytes(value)
    assert payload == b'{"a":{"truth":true},"z":[2,1]}\n'
    assert t10._parse_canonical_json(payload, "fixture") == value

    with pytest.raises(ValueError, match="duplicate JSON key"):
        t10._parse_canonical_json(b'{"a":1,"a":2}\n', "fixture")
    with pytest.raises(ValueError, match="not canonical"):
        t10._parse_canonical_json(b'{"z":1,"a":2}\n', "fixture")
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        t10._parse_canonical_json(b'{"a":NaN}\n', "fixture")


def test_frozen_result_loader_enforces_modes_single_file_hash_and_canonical_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "arm-root"
    root.mkdir(mode=0o700)
    result_path = root / "RESULT.json"
    payload = t10._canonical_json_bytes({"result": "PASS", "valid_run": True})
    result_path.write_bytes(payload)
    result_path.chmod(0o400)
    root.chmod(0o500)
    expected_sha = hashlib.sha256(payload).hexdigest()
    try:
        result, evidence = t10._load_frozen_result(result_path, expected_sha, arm="SEQ")
        assert result == {"result": "PASS", "valid_run": True}
        assert evidence["root_mode"] == "0500"
        assert evidence["file_mode"] == "0400"
        assert evidence["single_result_file"] is True

        with pytest.raises(ValueError, match="SHA256 differs"):
            t10._load_frozen_result(result_path, "0" * 64, arm="SEQ")

        root.chmod(0o700)
        extra = root / "extra"
        extra.write_text("forbidden", encoding="utf-8")
        root.chmod(0o500)
        with pytest.raises(ValueError, match="must contain only RESULT.json"):
            t10._load_frozen_result(result_path, expected_sha, arm="SEQ")
    finally:
        root.chmod(0o700)
        for child in root.iterdir():
            child.chmod(0o600)


def test_terminal_write_is_exclusive_canonical_and_frozen(tmp_path: Path) -> None:
    root = tmp_path / "terminal-root"
    root.mkdir(mode=0o700)
    result_path = root / "RESULT.json"
    report = {"z": 2, "a": 1}
    try:
        t10._write_terminal_report(result_path, report)
        assert result_path.read_bytes() == b'{"a":1,"z":2}\n'
        assert stat.S_IMODE(os.lstat(result_path).st_mode) == 0o444
        with pytest.raises(FileExistsError):
            t10._write_terminal_report(result_path, report)

        t10._freeze_run_root(root)
        assert stat.S_IMODE(os.lstat(root).st_mode) == 0o555
        assert stat.S_IMODE(os.lstat(result_path).st_mode) == 0o444
    finally:
        root.chmod(0o700)
        if result_path.exists():
            result_path.chmod(0o600)
