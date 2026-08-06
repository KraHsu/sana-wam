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
RUNNER = ROOT / "scripts/smoke_libero_ar_t14_initseed_replication_accum8_gpu.py"
CONFIG = ROOT / "configs/experiments/libero_t14_initseed_replication_accum8.yaml"
BASELINE_CONFIG = ROOT / "configs/benchmarks/libero/train_libero_ar_baseline.yaml"


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
        and node.iter.args[0].id == "T14_UPDATE_STEPS"
    ]
    assert len(matches) == 1
    return matches[0]


def test_t14_production_accum8_budget_seed_and_root_are_frozen() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T14_MACRO_STEPS"] == 20
    assert literals["T14_UPDATE_STEPS"] == 20
    assert literals["T14_FORWARD_EXPOSURES_PER_SAMPLE"] == 20
    assert literals["T14_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO"] == 0.125
    assert literals["T14_CUMULATIVE_LOSS_WEIGHT_PER_SAMPLE"] == 2.5
    assert literals["T14_CUMULATIVE_LOSS_WEIGHT_PER_SUITE"] == 5.0
    assert literals["T14_TRAINING_FORWARDS"] == 160
    assert literals["T14_MEASUREMENT_FORWARDS"] == 32
    assert literals["T14_ARCHITECTURE_FORWARDS"] == 192
    assert literals["T14_INITIALIZATION_SEED"] == 20260807
    assert literals["T14_DATALOADER_SEED"] == 20260806
    assert literals["T14_LOSS_RECIPE_SEED"] == 20260826
    assert literals["T14_ROOT_SLUG"] == (
        "libero-t14-initseed-replication-accum8-balanced-joint-fixed20"
    )
    assert literals["T14_EXECUTION_VERDICT"] == (
        "T14_INITSEED_REPLICATION_ACCUM8_JOINT_ARM_VALID"
    )
    assert "int(cfg.training.batch_size) != 1" in source
    assert "int(cfg.training.gradient_accumulation_steps) != 8" in source
    assert 'parser.add_argument("--arm"' not in source
    assert (
        'T14_RUN_NAMESPACE = Path("/DATA/share/sana_wam_libero_nonformal_screens/t14")'
        in source
    )
    assert 'expected_root_name = f"{T14_ROOT_SLUG}-{args.nonce}"' in source
    assert "int(cfg.training.seed) != args.initialization_seed" in source
    assert "int(cfg.dataloader.seed) != T14_DATALOADER_SEED" in source


def test_t14_config_differs_from_production_baseline_only_by_init_seed() -> None:
    baseline = BASELINE_CONFIG.read_text(encoding="utf-8")
    candidate = CONFIG.read_text(encoding="utf-8")
    old = "  seed: 20260806\n  debug: false\n"
    new = "  seed: 20260807\n  debug: false\n"
    assert baseline.count(old) == 1
    assert candidate == baseline.replace(old, new, 1)
    assert hashlib.sha256(candidate.encode("utf-8")).hexdigest() == (
        "36daaada09409c22ef2e57e3382ca2ef28c6847f9d2e043cad354464c3dde929"
    )


def test_t14_inherits_exact_t13_samples_assets_payloads_and_groups() -> None:
    _source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    training = literals["T13_TRAIN_SAMPLES"]
    heldout = literals["T13_HELDOUT_SAMPLES"]
    assert [row["label"] for row in training] == [
        "A12",
        "A13",
        "A14",
        "A15",
        "A16",
        "A17",
        "A18",
        "A19",
    ]
    assert [row["label"] for row in heldout] == [
        "H12",
        "H13",
        "H14",
        "H15",
        "H16",
        "H17",
        "H18",
        "H19",
    ]
    assert [row["episode_index"] for row in training] == [
        191,
        144,
        19,
        170,
        75,
        38,
        70,
        10,
    ]
    assert [row["episode_index"] for row in heldout] == [
        130,
        397,
        394,
        34,
        154,
        195,
        23,
        306,
    ]
    assert [row["episode_length"] for row in training] == [
        96,
        92,
        161,
        160,
        116,
        93,
        416,
        383,
    ]
    assert [row["episode_length"] for row in heldout] == [
        96,
        101,
        148,
        224,
        113,
        104,
        455,
        505,
    ]
    assert literals["T13_CONTEXT_GROUPS"] == (
        ("A12", "A13", "H12", "H13"),
        ("A14", "A15", "H14", "H15"),
        ("A16", "A17", "H16", "H17"),
        ("A18", "A19", "H18", "H19"),
    )
    observed_assets = []
    for row in (*training, *heldout):
        payload = (
            "SANA-WAM/LIBERO/T13_MULTI_EPISODE_ACCUM8_SELECTION_V1\n"
            f"T12_RESULT_SHA256={literals['T12_PREDECESSOR_RESULT_SHA256']}\n"
            f"T12_SELECTION_MANIFEST_SHA256={literals['T12_SELECTION_MANIFEST_SHA256']}\n"
            f"DATASET={row['dataset']}\n"
            f"TASK_INDEX={row['task_index']}\n"
            f"EPISODE_INDEX={row['episode_index']}\n"
            "START_FRAME=0\n"
        ).encode("ascii")
        assert (
            hashlib.sha256(payload).hexdigest()
            == (row["episode_selection_payload_sha256"])
        )
        assert len(row["assets"]) == 3
        assert all(len(value) == 64 for value in row["assets"].values())
        observed_assets.extend(row["assets"].values())
    assert len(observed_assets) == len(set(observed_assets)) == 48
    assert "TODO" not in json.dumps([training, heldout])


def test_t14_inherits_exact_t13_canonical_selection_manifest() -> None:
    _source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    samples_by_label = {
        row["label"]: row
        for row in (*literals["T13_TRAIN_SAMPLES"], *literals["T13_HELDOUT_SAMPLES"])
    }
    samples = []
    for suite_index, group in enumerate(literals["T13_CONTEXT_GROUPS"]):
        for zero_rank, label in enumerate(group):
            samples.append(
                {
                    **samples_by_label[label],
                    "role": "update" if zero_rank < 2 else "heldout",
                    "selection_rank": zero_rank + 1,
                    "suite_index": suite_index,
                    "within_suite_role_index": zero_rank % 2,
                }
            )
    manifest = {
        "candidate_episode_count": 161,
        "candidate_episode_counts_by_suite": {
            literals["SPATIAL_NAME"]: 41,
            literals["OBJECT_NAME"]: 48,
            literals["GOAL_NAME"]: 45,
            literals["LIBERO10_NAME"]: 27,
        },
        "candidate_filter": {
            "episode_length_filter": "none",
            "model_facing_exclusion_through": "T12",
            "start_frame": 0,
            "task_identity_source": "T12 selected task per suite",
        },
        "excluded_model_facing_samples_within_selected_tasks": list(
            literals["T13_T12_MODEL_FACING_SAMPLES"]
        ),
        "heldout_probe_order": list(literals["T13_HELDOUT_LABELS"]),
        "samples": samples,
        "schema_version": "sana-wam-libero-t13-multi-episode-accum8-selection-v1",
        "selection_payload_template": (
            "SANA-WAM/LIBERO/T13_MULTI_EPISODE_ACCUM8_SELECTION_V1\n"
            f"T12_RESULT_SHA256={literals['T12_PREDECESSOR_RESULT_SHA256']}\n"
            f"T12_SELECTION_MANIFEST_SHA256={literals['T12_SELECTION_MANIFEST_SHA256']}\n"
            "DATASET={dataset}\n"
            "TASK_INDEX={task_index}\n"
            "EPISODE_INDEX={episode_index}\n"
            "START_FRAME=0\n"
        ),
        "selection_rule": literals["T13_SELECTION_RULE"],
        "suite_order": list(literals["T13_SUITE_ORDER"]),
        "t12_eligible_manifest_sha256": literals["T12_ELIGIBLE_MANIFEST_SHA256"],
        "t12_predecessor_result_sha256": literals["T12_PREDECESSOR_RESULT_SHA256"],
        "t12_selection_manifest_sha256": literals["T12_SELECTION_MANIFEST_SHA256"],
        "update_accumulation_order": list(literals["T13_TRAIN_LABELS"]),
    }
    encoded = _canonical_json_bytes(manifest)
    assert len(encoded) == literals["T13_SELECTION_MANIFEST_BYTES"] == 16763
    assert literals["T13_SELECTION_MANIFEST_SHA256"] == (
        "fa00b477558eb26ec5657b87e11e4d6023181f88ea023889e2e86d94b3fd5305"
    )
    assert (
        hashlib.sha256(encoded).hexdigest() == literals["T13_SELECTION_MANIFEST_SHA256"]
    )
    assert literals["T13_ELIGIBLE_MANIFEST_BYTES"] == 53728
    assert literals["T13_ELIGIBLE_MANIFEST_SHA256"] == (
        "50d97577a4e0120ef9e6b42e1a227321b05ed19450626dc3dfa0c0810c997c47"
    )


def test_t14_loss_weights_are_equal_eighth_for_all_twenty_macros() -> None:
    _source, tree = _source_and_tree()
    labels = ("A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19")
    namespace = {
        "T14_UPDATE_STEPS": 20,
        "T14_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO": 0.125,
        "T13_TRAIN_LABELS": labels,
    }
    weights = _load_function(tree, "_loss_weights_for_macro", namespace)
    expected = {label: 0.125 for label in labels}
    assert [weights(step) for step in range(20)] == [expected] * 20
    assert math.fsum(expected.values()) == 1.0
    with pytest.raises(ValueError):
        weights(-1)
    with pytest.raises(ValueError):
        weights(20)


def test_t14_classifier_has_exact_ordered_strict_sixteen_ratio_branches() -> None:
    _source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    namespace: dict[str, Any] = {
        "Any": Any,
        "math": math,
        "statistics": statistics,
        "T13_SUITE_ORDER": literals["T13_SUITE_ORDER"],
        "T13_TRAIN_SAMPLES": literals["T13_TRAIN_SAMPLES"],
        "T13_HELDOUT_SAMPLES": literals["T13_HELDOUT_SAMPLES"],
        "T13_Q_BY_SUITE": literals["T13_Q_BY_SUITE"],
        "T14_REPLICATED_VERDICT": literals["T14_REPLICATED_VERDICT"],
        "T14_TRAINING_FIT_FAILURE_VERDICT": literals[
            "T14_TRAINING_FIT_FAILURE_VERDICT"
        ],
        "T14_HELDOUT_GAP_VERDICT": literals["T14_HELDOUT_GAP_VERDICT"],
    }
    namespace["_require_strictly_positive_finite"] = _load_function(
        tree, "_require_strictly_positive_finite", namespace
    )
    namespace["_multi_episode_sample_summary"] = _load_function(
        tree, "_multi_episode_sample_summary", namespace
    )
    classify = _load_function(tree, "_multi_episode_summary", namespace)
    before_a = {label: 1.0 for label in literals["T13_TRAIN_LABELS"]}
    before_h = {label: 1.0 for label in literals["T13_HELDOUT_LABELS"]}

    replicated = classify(
        before_a,
        {label: 0.5 for label in before_a},
        before_h,
        {label: 0.5 for label in before_h},
    )
    assert replicated["all_sixteen_ratios_strictly_below_one"] is True
    assert replicated["scientific_verdict"] == literals["T14_REPLICATED_VERDICT"]

    fit_after = {label: 0.5 for label in before_a}
    fit_after["A12"] = 1.0
    fit_failure = classify(
        before_a,
        fit_after,
        before_h,
        {label: 0.5 for label in before_h},
    )
    assert fit_failure["any_update_ratio_at_least_one"] is True
    assert (
        fit_failure["scientific_verdict"]
        == literals["T14_TRAINING_FIT_FAILURE_VERDICT"]
    )

    heldout_after = {label: 0.5 for label in before_h}
    heldout_after["H19"] = 1.0
    heldout_gap = classify(
        before_a,
        {label: 0.5 for label in before_a},
        before_h,
        heldout_after,
    )
    assert heldout_gap["any_update_ratio_at_least_one"] is False
    assert heldout_gap["scientific_verdict"] == literals["T14_HELDOUT_GAP_VERDICT"]

    with pytest.raises(ValueError):
        classify(
            before_a,
            {label: 0.5 for label in before_a},
            before_h,
            {**{label: 0.5 for label in before_h}, "H19": float("nan")},
        )


def test_t14_macro_loop_has_eight_microbacks_then_one_optimizer_step() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    macro_source = ast.get_source_segment(source, _main_macro_loop(tree))
    assert macro_source is not None
    assert len(literals["T13_TRAIN_LABELS"]) == 8
    assert "for update_label in T13_TRAIN_LABELS" in macro_source
    assert "backward_scale=step_loss_weights[update_label]" in macro_source
    assert macro_source.count("optimizer.step()") == 1
    assert macro_source.index("for update_label in T13_TRAIN_LABELS") < (
        macro_source.index("optimizer.step()")
    )
    assert "_no_intra_macro_mutation_evidence(" in macro_source
    assert "trainer._sync_master_gradients(master_pairs)" in macro_source
    assert "trainer._copy_master_parameters_to_model(master_pairs)" in macro_source


def test_t14_grouped_lists_prevent_dataset_key_overwrite_and_lock_contexts() -> None:
    source, _tree = _source_and_tree()
    assert 'row["dataset"]: row for row in joint_update_summary' not in source
    assert 'row["dataset"]: row for row in fresh_heldout_summary' not in source
    assert "dataset: [" in source
    assert "for group in T13_CONTEXT_GROUPS" in source
    assert "len(contexts_by_label) != 16" in source
    assert "len(context_digests) != 4" in source
    assert "T13_TRAIN_LABELS, T13_HELDOUT_LABELS, strict=True" not in source
    assert "architecture.prepare_inputs([raw_sample])" in source


def test_t14_t13_direct_predecessor_seed_axis_and_nonformal_scope_are_pinned() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T13_PREDECESSOR_SOURCE_COMMIT"] == (
        "68fa88157973f59383a87be1cb3107f5824e64cf"
    )
    assert literals["T13_PREDECESSOR_RUNNER_SHA256"] == (
        "ddbd141f3c32f9f89c4e936b1432d379cf56f0b886f84f38406fa4481654efb8"
    )
    assert literals["T13_PREDECESSOR_RESULT_SHA256"] == (
        "a75739991561965d212ad98cc2504cabf8696b2dcaf2e0ad5b46aa774ca00763"
    )
    required_fragments = (
        'external_assets["t13_predecessor/runner.py"]',
        'external_assets["t13_predecessor/RESULT.json"]',
        '"stage": "T13_MULTI_EPISODE_ACCUM8_BALANCED_JOINT"',
        '"architecture_forwards_total": T14_ARCHITECTURE_FORWARDS',
        '"architecture_training_forwards": T14_TRAINING_FORWARDS',
        '"architecture_measurement_forwards": T14_MEASUREMENT_FORWARDS',
        '"prepare_inputs_calls": prepare_inputs_calls',
        '"production_accumulation_boundary_matched": True',
        '"fresh_initialization_seed_axis_only": True',
        '"t13_initialization_seed": 20260806',
        '"t14_initialization_seed": T14_INITIALIZATION_SEED',
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


def test_t14_terminal_result_carries_both_verdicts_without_retry_path() -> None:
    source, _tree = _source_and_tree()
    assert '"execution_verdict": execution_verdict' in source
    assert '"scientific_verdict": scientific_verdict' in source
    assert '"verdict": execution_verdict' in source
    assert '"valid_run": True' in source
    assert "T14_INITSEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED" in source
    assert "T14_INITSEED_REPLICATION_ACCUM8_JOINT_TRAINING_FIT_FAILURE" in source
    assert "T14_INITSEED_REPLICATION_ACCUM8_JOINT_HELDOUT_GAP" in source
    assert source.count('_write_terminal_report(run_root / "RESULT.json", report)') == 1
    assert "retry" not in source.lower()
    assert '"formal_training_admission_granted": False' in source
