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
RUNNER = (
    ROOT / "scripts/smoke_libero_ar_t11_same_task_new_episode_balanced_joint_gpu.py"
)


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
        and node.iter.args[0].id == "T11_UPDATE_STEPS"
    ]
    assert len(matches) == 1
    return matches[0]


def test_t11_budget_single_joint_contract_and_root_are_frozen() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T11_MACRO_STEPS"] == 20
    assert literals["T11_UPDATE_STEPS"] == 20
    assert literals["T11_FORWARD_EXPOSURES_PER_SAMPLE"] == 20
    assert literals["T11_TRAINING_FORWARDS"] == 80
    assert literals["T11_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO"] == 0.25
    assert literals["T11_CUMULATIVE_LOSS_WEIGHT_PER_SAMPLE"] == 5.0
    assert literals["T11_INITIALIZATION_SEED"] == 20260806
    assert literals["T11_LOSS_RECIPE_SEED"] == 20260826
    assert literals["T11_TRAIN_LABELS"] == ("A4", "A5", "A6", "A7")
    assert literals["T11_HELDOUT_LABELS"] == ("H4", "H5", "H6", "H7")
    assert literals["T11_ROOT_SLUG"] == (
        "libero-t11-same-task-new-episode-balanced-joint-fixed20"
    )
    assert literals["T11_EXECUTION_VERDICT"] == "T11_NEW_EPISODE_JOINT_ARM_VALID"
    assert 'parser.add_argument("--arm"' not in source
    assert (
        'T11_RUN_NAMESPACE = Path("/DATA/share/sana_wam_libero_nonformal_screens/t11")'
        in source
    )
    assert 'expected_root_name = f"{T11_ROOT_SLUG}-{args.nonce}"' in source


def test_t11_exact_new_episode_samples_assets_and_payloads_are_frozen() -> None:
    _source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    training = literals["T11_TRAIN_SAMPLES"]
    heldout = literals["T11_HELDOUT_SAMPLES"]
    assert [row["episode_index"] for row in training] == [76, 64, 30, 303]
    assert [row["episode_index"] for row in heldout] == [265, 337, 180, 272]
    assert [row["episode_length"] for row in training] == [133, 129, 185, 244]
    assert [row["episode_length"] for row in heldout] == [130, 134, 182, 272]
    assert [row["label"] for row in training] == ["A4", "A5", "A6", "A7"]
    assert [row["label"] for row in heldout] == ["H4", "H5", "H6", "H7"]
    aggregate_sha = literals["T10_AGGREGATE_RESULT_SHA256"]
    observed_assets = []
    for row in (*training, *heldout):
        assert len(row["assets"]) == 3
        assert all(len(value) == 64 for value in row["assets"].values())
        observed_assets.extend(row["assets"].values())
        payload = (
            "SANA-WAM/LIBERO/T11_NEW_SAMPLE_SELECTION_V1\n"
            f"T10_AGGREGATE_RESULT_SHA256={aggregate_sha}\n"
            f"DATASET={row['dataset']}\n"
            f"TASK_INDEX={row['task_index']}\n"
            f"EPISODE_INDEX={row['episode_index']}\n"
            "START_FRAME=0\n"
        ).encode("ascii")
        assert (
            hashlib.sha256(payload).hexdigest()
            == row["episode_selection_payload_sha256"]
        )
    assert len(observed_assets) == len(set(observed_assets)) == 24
    for update, probe in zip(training, heldout, strict=True):
        assert update["dataset"] == probe["dataset"]
        assert update["task_index"] == probe["task_index"]
        assert update["task"] == probe["task"]
        assert update["episode_index"] != probe["episode_index"]


def test_t11_manifest_pins_and_canonical_selection_are_exact() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T11_ELIGIBLE_MANIFEST_BYTES"] == 8676
    assert literals["T11_ELIGIBLE_MANIFEST_SHA256"] == (
        "25486093e047d38d95151b36277874b89851f934720e3418e1f817d0b736745b"
    )
    assert literals["T11_SELECTION_MANIFEST_BYTES"] == 6198
    assert literals["T11_SELECTION_MANIFEST_SHA256"] == (
        "29e9ce96b8d4d6b0d5e41c6cc2d0fb887c4061990d4c2669a498d806dbd82dcd"
    )
    samples = [
        row
        for pair in zip(
            literals["T11_TRAIN_SAMPLES"],
            literals["T11_HELDOUT_SAMPLES"],
            strict=True,
        )
        for row in pair
    ]
    manifest = {
        "eligible_manifest_sha256": literals["T11_ELIGIBLE_MANIFEST_SHA256"],
        "samples": samples,
        "schema_version": ("sana-wam-libero-t11-same-task-new-episode-selection-v1"),
        "selection_rule": literals["T11_SELECTION_RULE"],
        "suite_order": list(literals["T11_SUITE_ORDER"]),
        "t10_aggregate_result_sha256": literals["T10_AGGREGATE_RESULT_SHA256"],
    }
    encoded = _canonical_json_bytes(manifest)
    assert len(encoded) == literals["T11_SELECTION_MANIFEST_BYTES"]
    assert (
        hashlib.sha256(encoded).hexdigest() == literals["T11_SELECTION_MANIFEST_SHA256"]
    )
    assert len(literals["T11_EXCLUDED_SAMPLES"]) == 15
    assert 'manifest["candidate_episode_count"] != 158' in source
    assert "PLACEHOLDER" not in source


def test_t11_loss_weights_are_equal_quarter_for_all_twenty_macros() -> None:
    _source, tree = _source_and_tree()
    namespace = {
        "T11_UPDATE_STEPS": 20,
        "T11_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO": 0.25,
        "T11_TRAIN_LABELS": ("A4", "A5", "A6", "A7"),
    }
    weights = _load_function(tree, "_loss_weights_for_macro", namespace)
    expected = {"A4": 0.25, "A5": 0.25, "A6": 0.25, "A7": 0.25}
    assert [weights(step) for step in range(20)] == [expected] * 20
    with pytest.raises(ValueError):
        weights(-1)
    with pytest.raises(ValueError):
        weights(20)


def test_t11_scientific_classifier_has_exact_ordered_strict_branches() -> None:
    _source, tree = _source_and_tree()
    samples_a = tuple(
        {
            "dataset": dataset,
            "episode_index": index,
            "label": f"A{index + 4}",
            "start_frame": 0,
            "task_index": index,
        }
        for index, dataset in enumerate(("s", "o", "g", "l"))
    )
    samples_h = tuple(
        {
            "dataset": dataset,
            "episode_index": index + 10,
            "label": f"H{index + 4}",
            "start_frame": 0,
            "task_index": index,
        }
        for index, dataset in enumerate(("s", "o", "g", "l"))
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "math": math,
        "statistics": statistics,
        "T11_TRAIN_SAMPLES": samples_a,
        "T11_HELDOUT_SAMPLES": samples_h,
        "T11_SUITE_ORDER": ("s", "o", "g", "l"),
        "T10_JOINT_Q_BY_SUITE": {"s": 0.6, "o": 0.4, "g": 0.6, "l": 0.4},
        "T11_REPLICATED_VERDICT": (
            "T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT_REPLICATED"
        ),
        "T11_TRAINING_FIT_FAILURE_VERDICT": (
            "T11_SAME_TASK_NEW_EPISODE_JOINT_TRAINING_FIT_FAILURE"
        ),
        "T11_HELDOUT_GAP_VERDICT": ("T11_SAME_TASK_NEW_EPISODE_JOINT_HELDOUT_GAP"),
    }
    namespace["_require_strictly_positive_finite"] = _load_function(
        tree, "_require_strictly_positive_finite", namespace
    )
    namespace["_paired_four_sample_summary"] = _load_function(
        tree, "_paired_four_sample_summary", namespace
    )
    classify = _load_function(tree, "_same_task_new_episode_summary", namespace)
    before_a = {f"A{index}": 2.0 for index in range(4, 8)}
    before_h = {f"H{index}": 2.0 for index in range(4, 8)}
    replicated = classify(
        before_a,
        {label: 1.0 for label in before_a},
        before_h,
        {label: 1.0 for label in before_h},
    )
    assert replicated["scientific_verdict"] == (
        "T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT_REPLICATED"
    )
    assert replicated["t10_joint_q_comparison"] == {
        "excluded_from_scientific_classifier": True,
        "per_suite": [
            {
                "dataset": dataset,
                "q": 0.5,
                "q_minus_t10_joint_q": 0.5 - threshold,
                "q_strictly_below_t10_joint_q": 0.5 < threshold,
                "t10_joint_q": threshold,
            }
            for dataset, threshold in (("s", 0.6), ("o", 0.4), ("g", 0.6), ("l", 0.4))
        ],
        "q_strictly_below_t10_joint_count": 2,
    }
    fit_after = {label: 1.0 for label in before_a}
    fit_after["A4"] = 2.0
    fit_failure = classify(
        before_a,
        fit_after,
        before_h,
        {label: 1.0 for label in before_h},
    )
    assert fit_failure["scientific_verdict"] == (
        "T11_SAME_TASK_NEW_EPISODE_JOINT_TRAINING_FIT_FAILURE"
    )
    gap_after = {label: 1.0 for label in before_h}
    gap_after["H4"] = 2.0
    heldout_gap = classify(
        before_a,
        {label: 1.0 for label in before_a},
        before_h,
        gap_after,
    )
    assert heldout_gap["scientific_verdict"] == (
        "T11_SAME_TASK_NEW_EPISODE_JOINT_HELDOUT_GAP"
    )
    assert heldout_gap["any_update_ratio_at_least_one"] is False
    with pytest.raises(ValueError):
        classify(
            before_a,
            {**{label: 1.0 for label in before_a}, "A4": math.nan},
            before_h,
            {label: 1.0 for label in before_h},
        )


def test_t11_macro_loop_has_four_microbacks_then_one_optimizer_step() -> None:
    source, tree = _source_and_tree()
    macro = _main_macro_loop(tree)
    macro_source = ast.get_source_segment(source, macro)
    assert macro_source is not None
    assert "for update_label in T11_TRAIN_LABELS" in macro_source
    assert "backward_scale=step_loss_weights[update_label]" in macro_source
    assert macro_source.count("optimizer.step()") == 1
    assert macro_source.index("for update_label in T11_TRAIN_LABELS") < (
        macro_source.index("optimizer.step()")
    )
    assert "macro_non_gradient_state_snapshot()" in macro_source
    assert "_no_intra_macro_mutation_evidence(" in macro_source
    assert "trainer._sync_master_gradients(master_pairs)" in macro_source
    assert "trainer._copy_master_parameters_to_model(master_pairs)" in macro_source


def test_t11_context_pairing_uses_the_frozen_a4_to_h7_labels() -> None:
    source, _tree = _source_and_tree()
    assert "for train_label, heldout_label in zip(" in source
    assert "T11_TRAIN_LABELS, T11_HELDOUT_LABELS, strict=True" in source
    assert 'contexts_by_label[f"A{index}"]' not in source
    assert 'contexts_by_label[f"H{index}"]' not in source


def test_t11_predecessor_and_nonformal_scope_are_pinned_fail_closed() -> None:
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
    required_fragments = (
        'external_assets["t10_arm_predecessor/runner.py"]',
        'external_assets["t10_seq_predecessor/RESULT.json"]',
        'external_assets["t10_joint_predecessor/RESULT.json"]',
        'external_assets["t10_aggregate_predecessor/RESULT.json"]',
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


def test_t11_terminal_result_carries_both_verdicts_without_retry_path() -> None:
    source, _tree = _source_and_tree()
    assert '"execution_verdict": execution_verdict' in source
    assert '"scientific_verdict": scientific_verdict' in source
    assert '"verdict": execution_verdict' in source
    assert '"valid_run": True' in source
    assert "T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT_REPLICATED" in source
    assert "T11_SAME_TASK_NEW_EPISODE_JOINT_TRAINING_FIT_FAILURE" in source
    assert "T11_SAME_TASK_NEW_EPISODE_JOINT_HELDOUT_GAP" in source
    assert source.count('_write_terminal_report(run_root / "RESULT.json", report)') == 1
    assert "retry" not in source.lower()
    assert '"formal_training_admission_granted": False' in source
