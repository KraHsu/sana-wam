from __future__ import annotations

import ast
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/smoke_libero_ar_t12_new_task_balanced_joint_gpu.py"


def _source_and_tree() -> tuple[str, ast.Module]:
    source = RUNNER.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(RUNNER))


def _literal_value(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(value, names) for value in node.elts)
    if isinstance(node, ast.List):
        return [_literal_value(value, names) for value in node.elts]
    if isinstance(node, ast.Set):
        return {_literal_value(value, names) for value in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _literal_value(key, names): _literal_value(value, names)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _literal_value(node.operand, names)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _literal_value(node.left, names)
        right = _literal_value(node.right, names)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Mult):
            return left * right
    raise ValueError(f"not a static literal expression: {ast.dump(node)}")


def _top_level_literals(tree: ast.Module) -> dict[str, Any]:
    names: dict[str, Any] = {}
    for node in tree.body:
        target = None
        value = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
            value = node.value
        if target is None or value is None:
            continue
        try:
            names[target] = _literal_value(value, names)
        except (KeyError, TypeError, ValueError):
            continue
    return names


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _load_function(tree: ast.Module, name: str, namespace: dict[str, Any]):
    module = ast.Module(body=[_function_node(tree, name)], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(RUNNER), "exec"), namespace)  # noqa: S102
    return namespace[name]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _main_macro_loop(tree: ast.Module) -> ast.For:
    matches = [
        node
        for node in ast.walk(_function_node(tree, "main"))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "step_index"
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "T12_UPDATE_STEPS"
    ]
    assert len(matches) == 1
    return matches[0]


def test_t12_budget_single_joint_contract_and_root_are_frozen() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T12_MACRO_STEPS"] == 20
    assert literals["T12_UPDATE_STEPS"] == 20
    assert literals["T12_FORWARD_EXPOSURES_PER_SAMPLE"] == 20
    assert literals["T12_TRAINING_FORWARDS"] == 80
    assert literals["T12_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO"] == 0.25
    assert literals["T12_CUMULATIVE_LOSS_WEIGHT_PER_SAMPLE"] == 5.0
    assert literals["T12_INITIALIZATION_SEED"] == 20260806
    assert literals["T12_LOSS_RECIPE_SEED"] == 20260826
    assert literals["T12_TRAIN_LABELS"] == ("A8", "A9", "A10", "A11")
    assert literals["T12_HELDOUT_LABELS"] == ("H8", "H9", "H10", "H11")
    assert literals["T12_ROOT_SLUG"] == ("libero-t12-new-task-balanced-joint-fixed20")
    assert literals["T12_EXECUTION_VERDICT"] == "T12_NEW_TASK_JOINT_ARM_VALID"
    assert 'parser.add_argument("--arm"' not in source
    assert (
        'T12_RUN_NAMESPACE = Path("/DATA/share/sana_wam_libero_nonformal_screens/t12")'
        in source
    )
    assert 'expected_root_name = f"{T12_ROOT_SLUG}-{args.nonce}"' in source


def test_t12_exact_new_task_samples_assets_and_payloads_are_frozen() -> None:
    _source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    training = literals["T12_TRAIN_SAMPLES"]
    heldout = literals["T12_HELDOUT_SAMPLES"]
    assert [row["task_index"] for row in training] == [5, 9, 4, 6]
    assert [row["task_index"] for row in heldout] == [5, 9, 4, 6]
    assert [row["episode_index"] for row in training] == [202, 407, 89, 316]
    assert [row["episode_index"] for row in heldout] == [226, 27, 387, 314]
    assert [row["episode_length"] for row in training] == [97, 148, 92, 396]
    assert [row["episode_length"] for row in heldout] == [103, 159, 121, 383]
    assert [row["label"] for row in training] == ["A8", "A9", "A10", "A11"]
    assert [row["label"] for row in heldout] == ["H8", "H9", "H10", "H11"]
    predecessor_sha = literals["T11_PREDECESSOR_RESULT_SHA256"]
    eligible_sha = literals["T12_ELIGIBLE_MANIFEST_SHA256"]
    observed_assets = []
    for row in (*training, *heldout):
        assert len(row["assets"]) == 3
        assert all(len(value) == 64 for value in row["assets"].values())
        observed_assets.extend(row["assets"].values())
        task_payload = (
            "SANA-WAM/LIBERO/T12_NEW_TASK_SELECTION_V1\n"
            f"T11_RESULT_SHA256={predecessor_sha}\n"
            f"ELIGIBLE_MANIFEST_SHA256={eligible_sha}\n"
            f"DATASET={row['dataset']}\n"
            f"TASK_INDEX={row['task_index']}\n"
            "START_FRAME=0\n"
        ).encode("ascii")
        episode_payload = (
            "SANA-WAM/LIBERO/T12_NEW_TASK_EPISODE_SELECTION_V1\n"
            f"T11_RESULT_SHA256={predecessor_sha}\n"
            f"ELIGIBLE_MANIFEST_SHA256={eligible_sha}\n"
            f"DATASET={row['dataset']}\n"
            f"TASK_INDEX={row['task_index']}\n"
            f"EPISODE_INDEX={row['episode_index']}\n"
            "START_FRAME=0\n"
        ).encode("ascii")
        assert (
            hashlib.sha256(task_payload).hexdigest()
            == row["task_selection_payload_sha256"]
        )
        assert (
            hashlib.sha256(episode_payload).hexdigest()
            == row["episode_selection_payload_sha256"]
        )
    assert len(observed_assets) == len(set(observed_assets)) == 24
    for update, probe in zip(training, heldout, strict=True):
        assert update["dataset"] == probe["dataset"]
        assert update["task_index"] == probe["task_index"]
        assert update["task"] == probe["task"]
        assert update["episode_index"] != probe["episode_index"]
    prior_tasks = {
        (row["dataset"], row["task_index"])
        for row in literals["T12_PRIOR_MODEL_FACING_TASKS"]
    }
    assert not prior_tasks & {
        (row["dataset"], row["task_index"]) for row in (*training, *heldout)
    }


def test_t12_manifest_pins_and_canonical_selection_are_exact() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T12_ELIGIBLE_MANIFEST_BYTES"] == 53728
    assert literals["T12_ELIGIBLE_MANIFEST_SHA256"] == (
        "50d97577a4e0120ef9e6b42e1a227321b05ed19450626dc3dfa0c0810c997c47"
    )
    assert literals["T12_SELECTION_MANIFEST_BYTES"] == 7044
    assert literals["T12_SELECTION_MANIFEST_SHA256"] == (
        "7f9ba75f17c7bdeaf3e68a70eaa9d752b26cf057250ff7f9c3fa37e2a895fe89"
    )
    samples = [
        row
        for pair in zip(
            literals["T12_TRAIN_SAMPLES"],
            literals["T12_HELDOUT_SAMPLES"],
            strict=True,
        )
        for row in pair
    ]
    manifest = {
        "eligible_manifest_sha256": literals["T12_ELIGIBLE_MANIFEST_SHA256"],
        "samples": samples,
        "schema_version": "sana-wam-libero-t12-new-task-balanced-joint-selection-v1",
        "selection_rule": literals["T12_SELECTION_RULE"],
        "suite_order": list(literals["T12_SUITE_ORDER"]),
        "t11_predecessor_result_sha256": literals["T11_PREDECESSOR_RESULT_SHA256"],
    }
    encoded = _canonical_json_bytes(manifest)
    assert len(encoded) == literals["T12_SELECTION_MANIFEST_BYTES"]
    assert (
        hashlib.sha256(encoded).hexdigest() == literals["T12_SELECTION_MANIFEST_SHA256"]
    )
    assert len(literals["T12_PRIOR_MODEL_FACING_TASKS"]) == 7
    assert len(literals["T12_PRIOR_MODEL_FACING_SAMPLES"]) == 22
    assert literals["T12_CONFIG_EXCLUDED_SAMPLES"] == (
        {
            "dataset": "libero_goal_no_noops_1.0.0_lerobot",
            "episode_index": 82,
        },
    )
    assert 'manifest["candidate_task_count"] != 33' in source
    assert 'manifest["candidate_episode_count"] != 1391' in source
    assert "_task_selection_payload_sha256" in source
    assert "T12_NEW_TASK_EPISODE_SELECTION_V1" in source
    assert "PLACEHOLDER" not in source


def test_t12_loss_weights_are_equal_quarter_for_all_twenty_macros() -> None:
    _source, tree = _source_and_tree()
    namespace = {
        "T12_UPDATE_STEPS": 20,
        "T12_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO": 0.25,
        "T12_TRAIN_LABELS": ("A8", "A9", "A10", "A11"),
    }
    weights = _load_function(tree, "_loss_weights_for_macro", namespace)
    expected = {"A8": 0.25, "A9": 0.25, "A10": 0.25, "A11": 0.25}
    assert [weights(step) for step in range(20)] == [expected] * 20
    with pytest.raises(ValueError):
        weights(-1)
    with pytest.raises(ValueError):
        weights(20)


def test_t12_scientific_classifier_has_exact_ordered_strict_branches() -> None:
    _source, tree = _source_and_tree()
    samples_a = tuple(
        {
            "dataset": dataset,
            "episode_index": index,
            "label": f"A{index + 8}",
            "start_frame": 0,
            "task_index": index,
        }
        for index, dataset in enumerate(("s", "o", "g", "l"))
    )
    samples_h = tuple(
        {
            "dataset": dataset,
            "episode_index": index + 10,
            "label": f"H{index + 8}",
            "start_frame": 0,
            "task_index": index,
        }
        for index, dataset in enumerate(("s", "o", "g", "l"))
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "math": math,
        "statistics": statistics,
        "T12_TRAIN_SAMPLES": samples_a,
        "T12_HELDOUT_SAMPLES": samples_h,
        "T12_SUITE_ORDER": ("s", "o", "g", "l"),
        "T11_Q_BY_SUITE": {"s": 0.6, "o": 0.4, "g": 0.6, "l": 0.4},
        "T12_REPLICATED_VERDICT": "T12_NEW_TASK_BALANCED_JOINT_REPLICATED",
        "T12_TRAINING_FIT_FAILURE_VERDICT": ("T12_NEW_TASK_JOINT_TRAINING_FIT_FAILURE"),
        "T12_HELDOUT_GAP_VERDICT": "T12_NEW_TASK_JOINT_HELDOUT_GAP",
    }
    namespace["_require_strictly_positive_finite"] = _load_function(
        tree, "_require_strictly_positive_finite", namespace
    )
    namespace["_paired_four_sample_summary"] = _load_function(
        tree, "_paired_four_sample_summary", namespace
    )
    classify = _load_function(tree, "_new_task_summary", namespace)
    before_a = {f"A{index}": 2.0 for index in range(8, 12)}
    before_h = {f"H{index}": 2.0 for index in range(8, 12)}
    replicated = classify(
        before_a,
        {label: 1.0 for label in before_a},
        before_h,
        {label: 1.0 for label in before_h},
    )
    assert replicated["scientific_verdict"] == (
        "T12_NEW_TASK_BALANCED_JOINT_REPLICATED"
    )
    assert replicated["t11_q_comparison"] == {
        "excluded_from_scientific_classifier": True,
        "per_suite": [
            {
                "dataset": dataset,
                "q": 0.5,
                "q_minus_t11_q": 0.5 - threshold,
                "q_strictly_below_t11_q": 0.5 < threshold,
                "t11_q": threshold,
            }
            for dataset, threshold in (("s", 0.6), ("o", 0.4), ("g", 0.6), ("l", 0.4))
        ],
        "q_strictly_below_t11_count": 2,
    }
    fit_after = {label: 1.0 for label in before_a}
    fit_after["A8"] = 2.0
    fit_failure = classify(
        before_a,
        fit_after,
        before_h,
        {label: 1.0 for label in before_h},
    )
    assert fit_failure["scientific_verdict"] == (
        "T12_NEW_TASK_JOINT_TRAINING_FIT_FAILURE"
    )
    gap_after = {label: 1.0 for label in before_h}
    gap_after["H8"] = 2.0
    heldout_gap = classify(
        before_a,
        {label: 1.0 for label in before_a},
        before_h,
        gap_after,
    )
    assert heldout_gap["scientific_verdict"] == ("T12_NEW_TASK_JOINT_HELDOUT_GAP")
    assert heldout_gap["any_update_ratio_at_least_one"] is False
    with pytest.raises(ValueError):
        classify(
            before_a,
            {**{label: 1.0 for label in before_a}, "A8": math.nan},
            before_h,
            {label: 1.0 for label in before_h},
        )


def test_t12_macro_loop_has_four_microbacks_then_one_optimizer_step() -> None:
    source, tree = _source_and_tree()
    macro = _main_macro_loop(tree)
    macro_source = ast.get_source_segment(source, macro)
    assert macro_source is not None
    assert "for update_label in T12_TRAIN_LABELS" in macro_source
    assert "backward_scale=step_loss_weights[update_label]" in macro_source
    assert macro_source.count("optimizer.step()") == 1
    assert macro_source.index("for update_label in T12_TRAIN_LABELS") < (
        macro_source.index("optimizer.step()")
    )
    assert "macro_non_gradient_state_snapshot()" in macro_source
    assert "_no_intra_macro_mutation_evidence(" in macro_source
    assert "trainer._sync_master_gradients(master_pairs)" in macro_source
    assert "trainer._copy_master_parameters_to_model(master_pairs)" in macro_source


def test_t12_context_pairing_uses_the_frozen_a8_to_h11_labels() -> None:
    source, _tree = _source_and_tree()
    assert "for train_label, heldout_label in zip(" in source
    assert "T12_TRAIN_LABELS, T12_HELDOUT_LABELS, strict=True" in source
    assert 'contexts_by_label[f"A{index}"]' not in source
    assert 'contexts_by_label[f"H{index}"]' not in source


def test_t12_predecessor_and_nonformal_scope_are_pinned_fail_closed() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T10_ARM_SOURCE_COMMIT"] == (
        "6861e5a13fa8110986f1b46f3062de9f0b0e3954"
    )
    assert literals["T10_ARM_RUNNER_SHA256"] == (
        "d7e3e7da3cd0a13e5217f0a044b2f36a4a703be7227149a897268f82b74857dd"
    )
    assert literals["T10_SEQ_RESULT_SHA256"] == (
        "d92fe05fe523c346e90ab6a392ddad9c3ec41764d5581211e6223895a42e8937"
    )
    assert literals["T10_JOINT_RESULT_SHA256"] == (
        "423ce3e01bea7368786b1c470a790af666baf7504a2235894a59b82efef3ea9b"
    )
    assert literals["T10_AGGREGATE_SOURCE_COMMIT"] == (
        "128e1be8cd48f8cef1b7c5f24d1bdecfbe45054a"
    )
    assert literals["T10_AGGREGATE_RUNNER_SHA256"] == (
        "b45e9536b35790c06e599ba919ed1370c64e4e1dff477019fab9f0ccc40b61c5"
    )
    assert literals["T10_AGGREGATE_RESULT_SHA256"] == (
        "8fcd26ea3a59fe6e01cf8f279301c899ed5dab541ada9c57e2ce85888abd147b"
    )
    assert literals["T11_PREDECESSOR_SOURCE_COMMIT"] == (
        "7d53d618234e3e37367c5b1e39e169742c2cf514"
    )
    assert literals["T11_PREDECESSOR_RUNNER_SHA256"] == (
        "ab66fcf6052582631424c97f149c9187d942b1da253e201fc698142e5ad41b54"
    )
    assert literals["T11_PREDECESSOR_RESULT_SHA256"] == (
        "6da12f2e3622d4e6427070bfd10b3dab9550c338a834b7662309cefea98d7417"
    )
    required_fragments = (
        'external_assets["t10_arm_predecessor/runner.py"]',
        'external_assets["t10_seq_predecessor/RESULT.json"]',
        'external_assets["t10_joint_predecessor/RESULT.json"]',
        'external_assets["t10_aggregate_predecessor/RESULT.json"]',
        'external_assets["t11_predecessor/runner.py"]',
        'external_assets["t11_predecessor/RESULT.json"]',
        '"stage": "T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT"',
        '"architecture_forwards_total": 96',
        '"architecture_training_forwards": 80',
        '"backward_calls": backward_calls',
        '"optimizer_steps": optimizer_step_calls',
        '"formal_training_executed": False',
        '"benchmark_evaluation_executed": False',
        '"sana_wam_training_checkpoint_loaded": False',
        '"sana_wam_training_checkpoint_saved": False',
        '"simulator_executed": False',
    )
    for fragment in required_fragments:
        assert fragment in source
    for forbidden in (
        "training.init_checkpoint",
        "training.resume_checkpoint",
        "training.resume_from_checkpoint",
        "training.resume_manifest",
        "model.video_backbone.init_dit_from",
    ):
        assert f'"{forbidden}"' in source


def test_t12_terminal_result_carries_both_verdicts_without_retry_path() -> None:
    source, _tree = _source_and_tree()
    assert '"execution_verdict": execution_verdict' in source
    assert '"scientific_verdict": scientific_verdict' in source
    assert '"verdict": execution_verdict' in source
    assert '"valid_run": True' in source
    assert "T12_NEW_TASK_BALANCED_JOINT_REPLICATED" in source
    assert "T12_NEW_TASK_JOINT_TRAINING_FIT_FAILURE" in source
    assert "T12_NEW_TASK_JOINT_HELDOUT_GAP" in source
    assert source.count('_write_terminal_report(run_root / "RESULT.json", report)') == 1
    assert "retry" not in source.lower()
    assert '"formal_training_admission_granted": False' in source
