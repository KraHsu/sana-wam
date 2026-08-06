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
RUNNER = ROOT / "scripts/smoke_libero_ar_t10_exact_balanced_joint_gpu.py"

T9_AGGREGATE_SOURCE_COMMIT = "b7cded5fd9cf83ffabeba18a7e352b1cb4438b66"
T9_AGGREGATE_RUNNER_SHA256 = (
    "b884ef028a48ee37afda70e730c431db48ed30f62be7f59cb2c665b6072ffc0d"
)
T9_AGGREGATE_RESULT_SHA256 = (
    "0cfc53b820939123de4bc2a626380495878d4d5c9a37a0be9be667173254149d"
)
T9_AGGREGATE_RESULT_PATH = (
    "/DATA/share/sana_wam_libero_nonformal_screens/t9_aggregate/b7cded5fd9cf/"
    "libero-t9-latin-square-combined-ab99dd8758ef03bb191d5fb48f3b9fbc/"
    "RESULT.json"
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


def _top_level_path_argument(tree: ast.Module, name: str) -> str:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "Path"
            and len(node.value.args) == 1
        ):
            return ast.literal_eval(node.value.args[0])
    raise AssertionError(f"runner lacks Path constant {name}")


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"runner must define exactly one {name}"
    return matches[0]


def _load_function(tree: ast.Module, name: str, namespace: dict[str, Any]):
    module = ast.Module(body=[_function_node(tree, name)], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(RUNNER), "exec"), namespace)  # noqa: S102
    return namespace[name]


def _call_name(call: ast.Call) -> str | None:
    function = call.func
    parts: list[str] = []
    while isinstance(function, ast.Attribute):
        parts.append(function.attr)
        function = function.value
    if isinstance(function, ast.Name):
        parts.append(function.id)
    return ".".join(reversed(parts)) if parts else None


def _call_names(node: ast.AST) -> list[str]:
    return [
        name
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        if (name := _call_name(candidate)) is not None
    ]


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
        and node.iter.args[0].id == "T10_UPDATE_STEPS"
    ]
    assert len(matches) == 1
    return matches[0]


def _micro_loop(macro_loop: ast.For) -> ast.For:
    matches = [
        node
        for node in ast.walk(macro_loop)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "update_label"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "T10_TRAIN_LABELS"
    ]
    assert len(matches) == 1
    return matches[0]


def _arm_argument(tree: ast.Module) -> ast.Call:
    matches = [
        node
        for node in ast.walk(_function_node(tree, "main"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--arm"
    ]
    assert len(matches) == 1
    return matches[0]


def test_t10_arm_choice_and_loss_weights_are_closed_and_exactly_matched() -> None:
    _source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T10_ARMS"] == ("SEQ", "JOINT")
    assert literals["T10_TRAIN_LABELS"] == ("A0", "A1", "A2", "A3")
    assert literals["T10_HELDOUT_LABELS"] == ("H0", "H1", "H2", "H3")

    argument = _arm_argument(tree)
    keywords = {keyword.arg: keyword.value for keyword in argument.keywords}
    assert _literal_value(keywords["required"], {}) is True
    assert isinstance(keywords["choices"], ast.Name)
    assert keywords["choices"].id == "T10_ARMS"

    weights_for_macro = _load_function(
        tree,
        "_loss_weights_for_macro",
        {
            "T10_ARMS": ("SEQ", "JOINT"),
            "T10_LOSS_WEIGHT_PER_SAMPLE_PER_MACRO": 0.25,
            "T10_TRAIN_LABELS": ("A0", "A1", "A2", "A3"),
            "T10_UPDATE_STEPS": 20,
        },
    )
    schedules = {
        arm: [weights_for_macro(arm, step) for step in range(20)]
        for arm in ("SEQ", "JOINT")
    }
    expected_labels = {"A0", "A1", "A2", "A3"}
    assert all(
        set(weights) == expected_labels
        for rows in schedules.values()
        for weights in rows
    )

    assert schedules["SEQ"][:4] == [
        {"A0": 1.0, "A1": 0.0, "A2": 0.0, "A3": 0.0},
        {"A0": 0.0, "A1": 1.0, "A2": 0.0, "A3": 0.0},
        {"A0": 0.0, "A1": 0.0, "A2": 1.0, "A3": 0.0},
        {"A0": 0.0, "A1": 0.0, "A2": 0.0, "A3": 1.0},
    ]
    assert schedules["SEQ"] == schedules["SEQ"][:4] * 5
    assert all(
        weights == {label: 0.25 for label in expected_labels}
        for weights in schedules["JOINT"]
    )
    for arm in ("SEQ", "JOINT"):
        assert {
            label: sum(row[label] for row in schedules[arm])
            for label in expected_labels
        } == {label: 5.0 for label in expected_labels}

    with pytest.raises(ValueError, match="unknown T10 arm"):
        weights_for_macro("BOTH", 0)
    with pytest.raises(ValueError, match="outside the frozen 20-step schedule"):
        weights_for_macro("SEQ", -1)
    with pytest.raises(ValueError, match="outside the frozen 20-step schedule"):
        weights_for_macro("JOINT", 20)


def test_t10_budget_is_twenty_macro_steps_with_four_micros_each_per_arm() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T10_MACRO_STEPS"] == 20
    assert literals["T10_UPDATE_STEPS"] == 20
    assert literals["T10_FORWARD_EXPOSURES_PER_SAMPLE"] == 20
    assert literals["T10_TRAINING_FORWARDS"] == 80
    assert literals["T10_CUMULATIVE_LOSS_WEIGHT_PER_SAMPLE"] == 5.0

    for fragment in (
        "architecture_forward_calls != 96",
        "backward_calls != 80",
        "optimizer_step_calls != 20",
        "prepare_inputs_calls != 8",
        '"architecture_forwards_total": 96',
        '"architecture_measurement_forwards": 16',
        '"architecture_training_forwards": 80',
        '"backward_calls": backward_calls',
        '"macro_optimizer_steps": optimizer_step_calls',
        '"optimizer_steps": optimizer_step_calls',
        '"raw_training_forward_exposures_per_sample": 20',
        '"effective_cumulative_loss_weight_per_update_sample": 5.0',
        '"fresh_heldout_samples_in_backward_or_update": 0',
    ):
        assert fragment in source


def test_t10_macro_loop_has_fixed_micro_order_and_no_intra_macro_update() -> None:
    source, tree = _source_and_tree()
    macro_loop = _main_macro_loop(tree)
    micro_loop = _micro_loop(macro_loop)
    micro_calls = _call_names(micro_loop)
    macro_calls = _call_names(macro_loop)

    assert micro_calls.count("fixed_forward") == 1
    for forbidden in (
        "optimizer.zero_grad",
        "optimizer.step",
        "trainer._sync_master_gradients",
        "trainer._copy_master_parameters_to_model",
        "torch.nn.utils.clip_grad_norm_",
    ):
        assert forbidden not in micro_calls
    assert macro_calls.count("optimizer.step") == 1
    assert macro_calls.count("trainer._sync_master_gradients") == 1
    assert macro_calls.count("trainer._copy_master_parameters_to_model") == 1
    assert macro_calls.count("torch.nn.utils.clip_grad_norm_") == 1
    assert micro_loop.end_lineno is not None
    optimizer_steps = [
        node
        for node in ast.walk(macro_loop)
        if isinstance(node, ast.Call) and _call_name(node) == "optimizer.step"
    ]
    assert len(optimizer_steps) == 1
    assert optimizer_steps[0].lineno > micro_loop.end_lineno

    loop_source = ast.get_source_segment(source, macro_loop)
    assert loop_source is not None
    weight_assignments = [
        node
        for node in macro_loop.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == "_loss_weights_for_macro"
    ]
    assert len(weight_assignments) == 1
    weight_name = weight_assignments[0].targets[0].id
    weight_call = weight_assignments[0].value
    assert isinstance(weight_call, ast.Call)
    assert len(weight_call.args) == 2
    assert ast.unparse(weight_call.args[0]) == "args.arm"
    assert ast.unparse(weight_call.args[1]) == "step_index"

    fixed_forward_calls = [
        node
        for node in ast.walk(micro_loop)
        if isinstance(node, ast.Call) and _call_name(node) == "fixed_forward"
    ]
    assert len(fixed_forward_calls) == 1
    forward_keywords = {
        keyword.arg: keyword.value for keyword in fixed_forward_calls[0].keywords
    }
    backward_scale = forward_keywords["backward_scale"]
    assert isinstance(backward_scale, ast.Subscript)
    assert isinstance(backward_scale.value, ast.Name)
    assert backward_scale.value.id == weight_name
    assert isinstance(backward_scale.slice, ast.Name)
    assert backward_scale.slice.id == "update_label"

    weight_accumulators = [
        node
        for node in ast.walk(micro_loop)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == weight_name
        and isinstance(node.value.slice, ast.Name)
        and node.value.slice.id == "update_label"
    ]
    assert len(weight_accumulators) == 1
    assert "for update_label in T10_TRAIN_LABELS" in loop_source
    assert loop_source.index("for update_label in T10_TRAIN_LABELS") < (
        loop_source.index("optimizer.step()")
    )

    snapshot_calls = [
        node
        for node in ast.walk(macro_loop)
        if isinstance(node, ast.Call)
        and _call_name(node) == "macro_non_gradient_state_snapshot"
    ]
    evidence_calls = [
        node
        for node in ast.walk(macro_loop)
        if isinstance(node, ast.Call)
        and _call_name(node) == "_no_intra_macro_mutation_evidence"
    ]
    assert len(snapshot_calls) == 2
    assert len(evidence_calls) == 1
    assert min(call.lineno for call in snapshot_calls) < micro_loop.lineno
    assert micro_loop.end_lineno is not None
    assert micro_loop.end_lineno < evidence_calls[0].lineno < optimizer_steps[0].lineno

    for fragment in (
        '"all_macro_steps_verified": all_macro_states_unchanged',
        '"evidence": per_macro_no_mutation_evidence',
        '"macro_steps_verified": len(per_macro_no_mutation_evidence)',
        '"parameters_constant_across_four_component_backward_calls": (',
        "all_macro_states_unchanged",
        "after_four_microbacks_before_gradient_clip_master_sync_and_",
    ):
        assert fragment in source
    assert '"parameters_constant_across_four_component_backward_calls": True' not in (
        source
    )


def test_t10_loss_and_ratio_boundaries_are_strictly_positive() -> None:
    source, tree = _source_and_tree()
    require_positive = _load_function(
        tree,
        "_require_strictly_positive_finite",
        {"math": math},
    )
    paired_summary = _load_function(
        tree,
        "_paired_four_sample_summary",
        {
            "Any": Any,
            "T10_MEDIAN_RATIO": 0.95,
            "T10_MIN_IMPROVED": 3,
            "_require_strictly_positive_finite": require_positive,
            "statistics": statistics,
        },
    )
    samples = tuple(
        {
            "dataset": f"suite-{index}",
            "episode_index": index,
            "label": f"X{index}",
            "start_frame": 0,
            "task_index": index,
        }
        for index in range(4)
    )
    before = {f"X{index}": 2.0 for index in range(4)}
    after = {f"X{index}": 1.0 for index in range(4)}
    summary = paired_summary(before, after, samples=samples)
    assert all(
        row["before_action_loss"] > 0.0
        and row["after_action_loss"] > 0.0
        and row["post_to_pre_ratio"] > 0.0
        for row in summary["per_sample"]
    )

    for bad in (0.0, -1.0, math.nan, math.inf, -math.inf):
        bad_before = {**before, "X0": bad}
        with pytest.raises(ValueError, match="finite and strictly positive"):
            paired_summary(bad_before, after, samples=samples)
        bad_after = {**after, "X0": bad}
        with pytest.raises(ValueError, match="finite and strictly positive"):
            paired_summary(before, bad_after, samples=samples)

    underflow_after = {**after, "X0": float.fromhex("0x0.0000000000001p-1022")}
    overflow_before = {**before, "X0": float.fromhex("0x1.fffffffffffffp+1023")}
    with pytest.raises(ValueError, match="post_to_pre_ratio"):
        paired_summary(overflow_before, underflow_after, samples=samples)

    tiny_before = {**before, "X0": float.fromhex("0x0.0000000000001p-1022")}
    huge_after = {**after, "X0": float.fromhex("0x1.fffffffffffffp+1023")}
    with pytest.raises(ValueError, match="post_to_pre_ratio"):
        paired_summary(tiny_before, huge_after, samples=samples)

    assert "or bool((loss <= 0).item())" in source
    assert "or bool((loss_action <= 0).item())" in source
    assert "or bool((loss < 0).item())" not in source
    assert "or bool((loss_action < 0).item())" not in source


def test_t10_no_intra_macro_evidence_fails_closed_on_each_state_component() -> None:
    _source, tree = _source_and_tree()

    def sha256_json(value: Any) -> str:
        payload = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    evidence_for = _load_function(
        tree,
        "_no_intra_macro_mutation_evidence",
        {
            "Any": Any,
            "T10_MACRO_STEPS": 20,
            "_sha256_json": sha256_json,
        },
    )
    snapshot = {
        "model_parameters": [("p", 1, 2, 3, "torch.bfloat16", (4,))],
        "fp32_master_parameters": [("p", 5, 6, 7, "torch.float32", (4,))],
        "optimizer_state": [("p", [("step", ("tensor", 8, 9, 10))])],
    }
    evidence = evidence_for(snapshot, dict(snapshot), macro_step=1)
    assert evidence["state_unchanged"] is True
    assert all(evidence["components_unchanged"].values())
    assert (
        evidence["pre_microback_state_sha256"]
        == evidence["post_microback_pre_optimizer_state_sha256"]
    )

    for component in snapshot:
        changed = {key: list(value) for key, value in snapshot.items()}
        changed[component] = [*changed[component], ("mutation",)]
        with pytest.raises(RuntimeError, match="mutated within macro"):
            evidence_for(snapshot, changed, macro_step=1)

    with pytest.raises(ValueError, match="component schema differs"):
        evidence_for(snapshot, {"model_parameters": []}, macro_step=1)
    with pytest.raises(ValueError, match="outside 1..20"):
        evidence_for(snapshot, snapshot, macro_step=0)
    with pytest.raises(ValueError, match="outside 1..20"):
        evidence_for(snapshot, snapshot, macro_step=21)


def test_t10_t9_aggregate_identity_is_fully_pinned() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T9_AGGREGATE_SOURCE_COMMIT"] == T9_AGGREGATE_SOURCE_COMMIT
    assert literals["T9_AGGREGATE_RUNNER_SHA256"] == T9_AGGREGATE_RUNNER_SHA256
    assert literals["T9_AGGREGATE_RESULT_SHA256"] == T9_AGGREGATE_RESULT_SHA256
    assert _top_level_path_argument(tree, "T9_AGGREGATE_RESULT_PATH") == (
        T9_AGGREGATE_RESULT_PATH
    )

    for fragment in (
        "T9_COMMON_POSITION_EFFECT_SUPPORTED",
        'external_assets["t9_aggregate_predecessor/RESULT.json"]',
        'external_assets["t9_aggregate_predecessor/runner.py"]',
        't9_aggregate.get("identity", {}).get("repo_commit")',
        't9_aggregate.get("identity", {}).get("runner_sha256")',
        't9_aggregate.get("run", {}).get("root")',
        't9_aggregate.get("scope", {}).get("input_result_count") != 4',
        't9_aggregate.get("scope", {}).get("gpu_or_model_executed") is not False',
        '"t9_aggregate_predecessor_result_sha256"',
        '"t9_aggregate_predecessor_runner_sha256"',
        '"t9_aggregate_predecessor_source_commit"',
    ):
        assert fragment in source


def test_t10_root_is_bound_to_arm_source_commit_and_lowercase_nonce() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T10_ROOT_SLUGS"] == {
        "SEQ": "libero-t10-seq-matched-core-fixed20",
        "JOINT": "libero-t10-joint-matched-core-fixed20",
    }
    assert _top_level_path_argument(tree, "T10_RUN_NAMESPACE") == (
        "/DATA/share/sana_wam_libero_nonformal_screens/t10"
    )
    for fragment in (
        "len(args.nonce) != 32",
        'character not in "0123456789abcdef"',
        'expected_root_name = f"{T10_ROOT_SLUGS[args.arm]}-{args.nonce}"',
        "args.run_root.parent.name != args.expected_repo_commit[:12]",
        "T10_RUN_NAMESPACE / args.expected_repo_commit[:12] / expected_root_name",
        'raise ValueError("T10 run-root basename does not bind the nonce")',
        'raise ValueError("T10 run-root parent does not bind the source commit")',
        'counterpart_arm = "JOINT" if args.arm == "SEQ" else "SEQ"',
        'raise ValueError("T10 SEQ/JOINT arms must use distinct nonces")',
        'raise RuntimeError("T10 run root was not empty before terminal write")',
    ):
        assert fragment in source
    calls = _call_names(tree)
    assert calls.count("_create_run_root") == 1
    assert calls.count("_write_terminal_report") == 2
    assert calls.count("_freeze_run_root") == 2


def test_t10_gpu_results_are_per_arm_typed_only() -> None:
    source, tree = _source_and_tree()
    literals = _top_level_literals(tree)
    assert literals["T10_ARM_VERDICTS"] == {
        "SEQ": "T10_SEQ_BRIDGE_VALID",
        "JOINT": "T10_JOINT_ARM_VALID",
    }
    assert "verdict = T10_ARM_VERDICTS[args.arm]" in source
    assert '"arm": args.arm' in source
    for verdict in (
        "T10_SEQ_BRIDGE_VALID",
        "T10_JOINT_ARM_VALID",
    ):
        assert verdict in source
    for aggregate_only_verdict in (
        "T10_BALANCED_JOINT_SIMULTANEOUS_RETENTION_SUPPORTS_OVERWRITE",
        "T10_MATCHED_DOSE_JOINT_TRAINING_FIT_FAILURE",
        "T10_JOINT_FIT_WITH_HELDOUT_GAP",
    ):
        assert aggregate_only_verdict not in source


def test_t10_scope_is_nonformal_checkpoint_free_and_no_hidden_extra_updates() -> None:
    source, tree = _source_and_tree()
    calls = _call_names(tree)
    assert calls.count("optimizer.step") == 1
    assert calls.count("_loss_weights_for_macro") >= 1
    backward_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "backward"
    ]
    assert len(backward_calls) == 1
    backward_receiver = backward_calls[0].func
    assert isinstance(backward_receiver, ast.Attribute)
    assert isinstance(backward_receiver.value, ast.BinOp)
    assert isinstance(backward_receiver.value.op, ast.Mult)
    assert ast.unparse(backward_receiver.value) == "loss * backward_scale"
    assert "trainer._sync_master_gradients" in calls
    assert "trainer._copy_master_parameters_to_model" in calls
    for forbidden_call in (
        "trainer.train",
        "torch.save",
        "torch.load",
        "architecture.save_checkpoint",
        "architecture.load_checkpoint",
    ):
        assert forbidden_call not in calls

    for fragment in (
        '"formal_training_executed": False',
        '"benchmark_evaluation_executed": False',
        '"simulator_executed": False',
        '"sana_base_pretrained_checkpoint_loaded": True',
        '"sana_wam_training_checkpoint_loaded": False',
        '"sana_wam_training_checkpoint_saved": False',
        '"rollout_transfer_claimed": False',
        '"benchmark_success_claimed": False',
        '"training_or_benchmark_readiness_claimed": False',
        '"training.init_checkpoint"',
        '"training.resume_checkpoint"',
        '"training.resume_from_checkpoint"',
        '"training.resume_manifest"',
        '"model.video_backbone.init_dit_from"',
    ):
        assert fragment in source
    for forbidden in (
        '"formal_training_executed": True',
        '"benchmark_evaluation_executed": True',
        '"simulator_executed": True',
        '"sana_wam_training_checkpoint_loaded": True',
        '"sana_wam_training_checkpoint_saved": True',
        "T10_MACRO_STEPS = 40",
        "T10_UPDATE_STEPS = 40",
    ):
        assert forbidden not in source
