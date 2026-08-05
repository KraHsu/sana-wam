"""Contracts for the CACH-A2 R1 launcher-fix vendor-GDN screen.

The default tests in this module are source/card/config checks only.  They do
not import the bridge or runner and therefore cannot initialize Torch, CUDA,
Triton, or the vendor kernel.  The CPU helper checks, CUDA operator smoke, and
terminal-classifier checks are separately gated for a future execution phase.

Nothing in this test module creates or mutates a run root, token namespace,
claim, checkpoint, dataset, or result artifact.  CACH-A2 inherits an already
consumed predecessor review token and creates no replacement token.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import fields, is_dataclass
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = (
    REPO_ROOT
    / "docs/cach_sana_wam/architecture_validation/cach_a2_r1/"
    "CACH_A2_R1_RUN_CARD.json"
)
DECISION_PATH = (
    REPO_ROOT
    / "docs/cach_sana_wam/architecture_validation/cach_a2_r1/"
    "CACH_A2_R1_LAUNCHER_FIX_DECISION.md"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/cach_av1b_a2_zero_anchor_vendor_gdn_bridge_r1.yaml"
)
BRIDGE_PATH = (
    REPO_ROOT
    / "src/sana_wam/model/cach_av1b_a2_zero_anchor_vendor_gdn_bridge_r1.py"
)
R0_BRIDGE_PATH = (
    REPO_ROOT
    / "src/sana_wam/model/cach_av1b_a2_zero_anchor_vendor_gdn_bridge.py"
)
RUNNER_PATH = (
    REPO_ROOT
    / "scripts/run_cach_av1b_a2_zero_anchor_vendor_gdn_bridge_r1.py"
)
TEST_PATH = (
    REPO_ROOT
    / "tests/test_cach_av1b_a2_zero_anchor_vendor_gdn_bridge_r1.py"
)

CARD_SHA256 = "c7b710177defda6d44066c7b7b25c4b32d11ec106a92662b0701ca5f87f3401e"
R0_CARD_SHA256 = "9d3c3973234df8a9593b254161ca07bf72f343c4475d9985e2901153b82a9713"
DECISION_SHA256 = "db51c2d2fca6b0327a1466595d7297630a39a6091dfc45bc2f9633b10d283c23"
CONFIG_SHA256 = "9446e895ae2b6f8274feef2fa37212073b5fbf67ce964cc764bc69808de64407"
TASK_RECIPE_SHA256 = "9d2a749d722775324ee233862772def530fbffe0ea9e55aeea87c160b87a51e4"
PREDECESSOR_RESULT_SHA256 = (
    "3ca4ac673893923d29f60b999879132a93d515382f94bdaa462d91d3cdd1786d"
)
PREDECESSOR_CLAIM_SHA256 = (
    "e7c355d3c4e8bfb942f8baa8a5511fb5964c2c7f2d6c975ef5aec0daa458a91f"
)
PREDECESSOR_TOKEN_ID = (
    "cadfd92f1c72ebd9cc9aa8bfad029ddc6f0e72d9f7942f34e74629ca1ae37252"
)
ROOT = (
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a2_r1/06f5d09127f8/"
    "cach-a2-r1-launcher-fix-dc17cb7bdbd70955ea4b91212f0d4ac3"
)
NONCE = "dc17cb7bdbd70955ea4b91212f0d4ac3"
SOURCE_AUTHORITY_SHA256 = (
    "37a6939250543f0aad3e9c0c4067e40e48e4453cca242ad2ad6750c34e1575cb"
)
SOURCE_AUTHORITY_STATEMENT = "修复"

EXPECTED_QUARTET = [
    "src/sana_wam/model/cach_av1b_a2_zero_anchor_vendor_gdn_bridge_r1.py",
    "configs/experiments/cach_av1b_a2_zero_anchor_vendor_gdn_bridge_r1.yaml",
    "scripts/run_cach_av1b_a2_zero_anchor_vendor_gdn_bridge_r1.py",
    "tests/test_cach_av1b_a2_zero_anchor_vendor_gdn_bridge_r1.py",
]
EXPECTED_TOP_LEVEL_CONFIG_KEYS = {
    "schema",
    "identity",
    "authority",
    "classification",
    "runtime",
    "architecture",
    "topology",
    "recipe",
    "initialization",
    "optimizer",
    "budget",
    "diagnostics",
    "metrics",
    "thresholds",
    "verdict",
    "predecessor_token_policy",
    "artifacts",
}
EXPECTED_CANDIDATE_ONLY_NAMES = {
    "action_conditioner.0.weight",
    "action_conditioner.2.weight",
    *{
        f"blocks.{index}.action_output_projection.weight"
        for index in range(20)
    },
}
EXPECTED_SCREEN_KEYS = {
    "schema",
    "validity",
    "metrics",
    "loss_trace",
    "theta0_manifests",
    "final_parameter_digests",
    "diagnostics",
    "jit_inventory",
    "resource_usage",
}
EXPECTED_METRIC_KEYS = {
    "reference_correct_mse_by_horizon",
    "candidate_correct_mse_by_horizon",
    "candidate_shuffle_mse_by_horizon",
    "candidate_no_action_mse_by_horizon",
    "primary_reference_step300",
    "primary_candidate_correct_step300",
    "primary_candidate_shuffle_step300",
    "primary_candidate_no_action_step300",
    "seam_grad_rms_step0",
    "seam_update_rms_final",
    "last_block_local_seam_parameter_jvp_rms_theta0",
    "last_block_local_action_condition_jvp_rms_step300",
    "last_block_local_raw_action_jvp_rms_step300",
    "last_block_local_raw_action_jvp_rms_theta0",
    "no_action_raw_action_jvp_rms_theta0",
    "no_action_raw_action_jvp_rms_step300",
    "loss_drop_rel",
    "counterfactual_delta_nmse_correct",
    "counterfactual_delta_nmse_correct_by_horizon",
    "shuffle_gap_rel",
    "shuffle_gap_rel_by_horizon",
    "no_action_gap_rel",
    "no_action_gap_rel_by_horizon",
    "candidate_gain_vs_reference",
    "paired_common_delta",
    "loss_stability",
}
EXPECTED_COMMON_DELTA_KEYS = {
    "paired_prediction_common_mse",
    "half_delta_mse",
    "half_delta_nmse",
    "half_delta_energy_ratio",
    "half_delta_alignment_cosine",
    "target_total_energy",
    "paired_common_mse_div_target_total_energy",
    "correct_mse_div_target_total_energy",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object_pairs,
    )


def _load_card() -> dict[str, Any]:
    return _load_strict_json(CARD_PATH)


def _load_config() -> dict[str, Any]:
    # The .yaml file is deliberately strict JSON, a YAML 1.2 subset.
    return _load_strict_json(CONFIG_PATH)


def _function_source(source: str, tree: ast.AST, name: str) -> str:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1, name
    segment = ast.get_source_segment(source, matches[0])
    assert segment is not None
    return segment


def _load_bridge():
    return importlib.import_module(
        "sana_wam.model.cach_av1b_a2_zero_anchor_vendor_gdn_bridge_r1"
    )


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "_cach_a2_r1_runner_contract_test",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _tensor_leaves(value: Any, torch) -> Iterator[Any]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if is_dataclass(value):
        for one_field in fields(value):
            yield from _tensor_leaves(getattr(value, one_field.name), torch)
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _tensor_leaves(value[key], torch)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item, torch)


def _vendor_modules(arm) -> tuple[Any, ...]:
    return tuple(
        module
        for module in arm.modules()
        if type(module).__module__
        == "diffusion.model.nets.sana_gdn_blocks_triton"
        and type(module).__name__ == "ChunkCausalGDNTriton"
    )


def _rms(tensors: tuple[Any, ...]) -> float:
    numerator = sum(
        float(value.detach().double().square().sum().item())
        for value in tensors
    )
    denominator = sum(value.numel() for value in tensors)
    return math.sqrt(numerator / denominator) if denominator else 0.0


def _forbidden(*_args, **_kwargs):
    raise AssertionError("forbidden proxy/data/checkpoint/CUDA fallback was used")


def test_frozen_card_config_identity_and_consumed_token_policy() -> None:
    assert _sha256(CARD_PATH) == CARD_SHA256
    assert _sha256(DECISION_PATH) == DECISION_SHA256
    assert _sha256(CONFIG_PATH) == CONFIG_SHA256
    card = _load_card()
    config = _load_config()

    assert card["schema"] == (
        "cach.architecture_validation.cach_a2_r1_launcher_fix_run_card.v1"
    )
    assert card["card_id"] == "cach-a2-r1-launcher-preflight-fix-v1"
    assert card["card_state"] == (
        "FROZEN_R1_LAUNCHER_FIX_SOURCE_AND_EXECUTION_AUTHORIZED"
    )
    assert card["identity"]["canonical_ssh_alias"] == "H200"
    assert card["identity"]["expected_platform_node"] == "huaxiyun"
    assert card["classification"]["architecture_id"] == (
        "CACH-A2-ZERO-ANCHORED-BIAS-FREE-v1"
    )
    assert card["identity"]["architecture_decision"] == {
        "path": (
            "docs/cach_sana_wam/architecture_validation/cach_a2_r1/"
            "CACH_A2_R1_LAUNCHER_FIX_DECISION.md"
        ),
        "sha256": DECISION_SHA256,
    }
    predecessor = card["identity"]["predecessor_review300"]
    assert predecessor["typed_verdict"] == (
        "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED"
    )
    assert predecessor["result"]["sha256"] == PREDECESSOR_RESULT_SHA256
    assert predecessor["consumed_claim"]["sha256"] == PREDECESSOR_CLAIM_SHA256
    assert predecessor["consumed_claim"]["token_id"] == PREDECESSOR_TOKEN_ID
    assert predecessor["consumed_claim"]["token_state"] == "CONSUMED"

    assert CONFIG_PATH.suffix == ".yaml"
    assert CONFIG_PATH.read_bytes().startswith(b"{")
    assert set(config) == EXPECTED_TOP_LEVEL_CONFIG_KEYS
    assert config["schema"] == (
        "cach.cach_a2_r1.zero_anchor_vendor_gdn_bridge.config.v1"
    )
    identity = config["identity"]
    assert identity["run_card"] == {
        "path": card["identity"]["self_path"],
        "sha256": CARD_SHA256,
    }
    assert identity["architecture_decision"] == (
        card["identity"]["architecture_decision"]
    )
    assert identity["planned_runtime"]["resolved_root"] == ROOT
    assert identity["planned_runtime"]["run_nonce"] == NONCE
    assert identity["planned_runtime"][
        "root_or_namespace_create_authorized_at_source_materialization"
    ] is False
    assert config["runtime"]["canonical_ssh_alias"] == "H200"
    assert config["runtime"]["expected_platform_node"] == "huaxiyun"

    materialized = identity["generated_paths"]
    assert [materialized[name]["path"] for name in materialized] == (
        EXPECTED_QUARTET
    )
    assert all(
        value["sha256"] in {
            "SOURCE_MATERIALIZATION_BIND_FULL_SHA256_OUT_OF_BAND",
            "SELF_OUT_OF_BAND_FULL_LITERAL_SHA256",
        }
        for value in materialized.values()
    )

    token_policy = config["predecessor_token_policy"]
    assert token_policy["predecessor_review_token_id"] == PREDECESSOR_TOKEN_ID
    assert token_policy["predecessor_top_level_state"] == "CONSUMED"
    assert token_policy["a2_review_or_budget_extension_token"] is None
    assert token_policy["create_token_or_ledger_namespace"] is False
    assert token_policy["derive_reset_replace_or_consume_token"] is False
    assert token_policy["atomic_claim_or_claim_recovery"] is False
    assert config["artifacts"][
        "valid_result_terminal_without_claim_ledger"
    ] is True
    assert config["artifacts"]["provisional_result"] is False
    assert "review_token" not in config

    authority = config["authority"]
    statement_bytes = SOURCE_AUTHORITY_STATEMENT.encode("utf-8")
    assert authority["source_statement"] == SOURCE_AUTHORITY_STATEMENT
    assert authority["source_statement_utf8_bytes"] == len(statement_bytes)
    assert authority["source_statement_sha256"] == SOURCE_AUTHORITY_SHA256
    assert hashlib.sha256(statement_bytes).hexdigest() == SOURCE_AUTHORITY_SHA256
    assert authority["runtime_authority_schema"] == (
        "cach.cach_a2_r1.runtime_execution_authority.v1"
    )
    assert authority["runtime_authority_operation"] == "EXECUTE_CACH_A2_R1_ONCE"


def test_card_architecture_metrics_thresholds_and_terminal_verdicts() -> None:
    card = _load_card()
    config = _load_config()
    architecture = card["architecture_contract"]

    assert architecture["candidate_formula"] == (
        "PHI_A=W2_SILU_W1_A; DELTA_L=U_L_TIMES_(PHI_A_MINUS_PHI_A0); "
        "H_PLUS=WHERE(ACTION_PRESENT_MASK,H_POST_VENDOR_PLUS_DELTA_L,"
        "H_POST_VENDOR)"
    )
    assert architecture["zero_anchor"]["value"] == (
        "EXACT_ALL_ZERO_ACTION_DIM_20"
    )
    assert architecture["zero_anchor"]["phi_zero_is_exact_zero"] is True
    assert architecture["bias_policy"] == {
        "action_conditioner_linear_bias": False,
        "per_block_action_output_projection_bias": False,
        "candidate_only_linear_bias_attributes_must_be_none": True,
        "candidate_only_non_none_bias_parameter_buffer_or_state_dict_key": (
            "FORBIDDEN"
        ),
    }
    assert set(architecture["candidate_only_trainable_name_patterns"]) == {
        "action_conditioner.0.weight",
        "action_conditioner.2.weight",
        "blocks.{0..19}.action_output_projection.weight",
    }
    mask = architecture["action_presence_mask"]
    assert mask["source"] == (
        "TYPED_ACTION_LAYOUT_PROVENANCE_ONLY_NOT_ACTION_NUMERIC_VALUE"
    )
    assert mask["correct_expected_row"] == [False, True, True, True, True]
    assert mask["shuffle_expected_row"] == [False, True, True, True, True]
    assert mask["no_action_expected_row"] == [False] * 5
    assert architecture["inactive_semantics"]["full_no_action"] == (
        "DO_NOT_READ_OR_EXECUTE_CONDITIONER_OR_OUTPUT_PROJECTION"
    )
    assert architecture["inactive_semantics"][
        "masked_zero_addition_as_identity"
    ] == "FORBIDDEN"

    optimizer = card["experiment_contract"]["optimizer"]
    assert optimizer["optimizer_steps_per_arm"] == 300
    assert optimizer["max_optimizer_steps_total"] == 600
    assert optimizer["loss_trace_points_per_arm"] == 301
    assert optimizer["metric_steps"] == [0, 300]
    assert optimizer["best_step_selection"] is False
    assert optimizer["automatic_budget_extension"] is False

    metrics = card["metric_contract"]
    assert metrics["accumulator_dtype"] == "float64"
    assert metrics["epsilon"] == 1.0e-12
    assert metrics["same_theta_same_batch_modes"] == [
        "correct",
        "shuffle",
        "no_action",
        "seam_disabled",
    ]
    assert "(PRED_PLUS+PRED_MINUS)/2" in metrics["paired_common_formula"]
    assert metrics["paired_half_delta_formula"].startswith(
        "DELTA_HAT=(PRED_PLUS-PRED_MINUS)/2"
    )
    assert "CEIL(0.90*N)-1" in metrics["p90_formula"]

    thresholds = card["threshold_contract"]
    go = thresholds["operator_go_all_required"]
    assert go["candidate_gain_vs_reference_min"] == 0.5
    assert go["loss_drop_rel_min"] == 0.8
    assert go["counterfactual_delta_nmse_correct_max"] == 0.25
    assert go["shuffle_gap_rel_min"] == 0.5
    assert go["no_action_gap_rel_min"] == 0.5
    common_blocked = thresholds["delta_live_common_mode_blocked_all_required"]
    assert common_blocked["half_delta_energy_ratio_min"] == 0.5
    assert common_blocked["half_delta_energy_ratio_max"] == 1.5
    assert common_blocked["half_delta_alignment_cosine_min"] == 0.8
    assert common_blocked[
        "paired_common_mse_div_target_total_energy_strictly_greater_than"
    ] == 1.0

    verdict = card["verdict_contract"]
    assert verdict["valid_all_operator_go"] == "OPERATOR_GO"
    assert verdict["valid_delta_live_common_mode_blocked"] == {
        "typed_verdict": "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED",
        "diagnostic_subclassification": "A2_DELTA_LIVE_COMMON_MODE_BLOCKED",
    }
    assert verdict["valid_all_strong_stop"] == "OPERATOR_STOP"
    assert verdict["valid_matching_none"]["diagnostic_subclassification"] == (
        "A2_VALID_NEITHER"
    )
    assert config["verdict"]["valid_result_publication"] == (
        "DIRECT_FINAL_RESULT_INSIDE_ROOT_BEFORE_SINGLE_TERMINAL_FREEZE"
    )
    assert config["verdict"]["provisional_result_or_external_claim"] is False
    assert "NO_AUTOMATIC_AV2_UNLOCK" in config["verdict"][
        "operator_go_effect"
    ]


def test_bridge_runner_and_test_are_static_a2_not_review_token_harness() -> None:
    config = _load_config()
    bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
    r0_bridge_source = R0_BRIDGE_PATH.read_text(encoding="utf-8")
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    test_source = TEST_PATH.read_text(encoding="utf-8")
    bridge_tree = ast.parse(bridge_source, filename=str(BRIDGE_PATH))
    runner_tree = ast.parse(runner_source, filename=str(RUNNER_PATH))
    ast.parse(test_source, filename=str(TEST_PATH))

    normalized_r1_bridge = bridge_source.replace(
        "CACH-A2 R1 single-call bridge",
        "CACH-A2 single-call bridge",
        1,
    ).replace(
        CARD_SHA256,
        R0_CARD_SHA256,
        1,
    ).replace(
        "cach.cach_a2_r1.zero_anchor_vendor_gdn_bridge.config.v1",
        "cach.cach_a2.zero_anchor_vendor_gdn_bridge.config.v1",
        1,
    )
    assert normalized_r1_bridge == r0_bridge_source

    function_names = {
        node.name
        for node in ast.walk(bridge_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    class_names = {
        node.name for node in ast.walk(bridge_tree) if isinstance(node, ast.ClassDef)
    }
    assert {
        "condition_zero_anchor_actions",
        "apply_zero_anchor_action_residual",
        "build_av1b_vendor_task",
        "build_av1b_vendor_gdn_pair",
        "pure_torch_last_block_tail",
        "last_block_local_seam_parameter_jvp",
        "last_block_local_action_condition_jvp",
        "_screen_common_delta_metrics",
        "_screen_loss_window_statistics",
        "_screen_no_action_candidate_gradients",
        "_screen_padding_seam_helper_fixture",
        "run_cach_a2_screen",
    } <= function_names
    assert {
        "AV1BVendorGDNBridgeSpec",
        "AV1BVendorTask",
        "AV1BVendorGDNBridgeArm",
        "AV1BVendorGDNBridgePair",
        "AV1BVendorSequenceOutput",
        "AV1BLocalJVPDiagnostic",
    } <= class_names

    repo_imports = sorted(
        {
            node.module
            for node in ast.walk(bridge_tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (
                node.module.startswith("sana_wam.")
                or node.module.startswith("diffusion.")
            )
        }
    )
    import_manifest = config["identity"]["direct_import_dependency_manifest"]
    assert import_manifest["schema"] == (
        "cach.cach_a2_r1.direct_import_dependency_manifest.v1"
    )
    assert import_manifest["repo_local_module_names_sorted"] == repo_imports
    assert [entry["module"] for entry in import_manifest["entries"]] == (
        repo_imports
    )
    source_pins = config["identity"]["source_pins"]
    for entry in import_manifest["entries"]:
        assert _sha256(REPO_ROOT / entry["path"]) == entry["sha256"]
        pin = source_pins[entry["source_pin_label"]]
        assert pin["path"] == entry["path"]
        assert pin["sha256"] == entry["sha256"]
    assert import_manifest["unmanifested_repo_local_import"] == (
        "IMPLEMENTATION_INVALID"
    )

    names_source = _function_source(
        bridge_source,
        bridge_tree,
        "_candidate_only_names",
    )
    assert "action_conditioner.0.weight" in names_source
    assert "action_conditioner.2.weight" in names_source
    assert "action_output_projection.weight" in names_source
    assert ".bias" not in names_source

    enable_sources = [
        ast.get_source_segment(bridge_source, node) or ""
        for node in ast.walk(bridge_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "enable_candidate_action_seam"
    ]
    assert len(enable_sources) == 2
    assert sum(source.count("bias=False") for source in enable_sources) >= 3
    assert all("register_parameter" not in source for source in enable_sources)
    assert 'register_buffer("no_action_slot", None, persistent=True)' in bridge_source
    assert "AV1B_CANDIDATE_ONLY_SEED: Final = 2026080421" in bridge_source
    assert (
        'AV1B_INITIALIZER_REVISION: Final = "cach-a2-zero-anchored-bias-free-v1"'
        in bridge_source
    )

    condition_source = _function_source(
        bridge_source,
        bridge_tree,
        "condition_zero_anchor_actions",
    )
    residual_source = _function_source(
        bridge_source,
        bridge_tree,
        "apply_zero_anchor_action_residual",
    )
    candidate_condition_source = _function_source(
        bridge_source,
        bridge_tree,
        "_candidate_action_condition",
    )
    assert "index_select" in condition_source
    assert "if not active_index.numel()" in condition_source
    assert "index_select" in residual_source
    assert "index_copy" in residual_source
    assert "torch.where" in residual_source
    assert "return hidden, residual, 0" in residual_source
    assert "latent_valid_mask" in candidate_condition_source
    assert "latent_action_spans" in candidate_condition_source
    assert "anchor_no_action_slot" in candidate_condition_source
    assert "count_nonzero" not in candidate_condition_source.split(
        "local_presence =", 1
    )[1]

    no_action_gradient_source = _function_source(
        bridge_source,
        bridge_tree,
        "_screen_no_action_candidate_gradients",
    )
    assert "allow_unused=True" in no_action_gradient_source
    assert "materialize_grads=False" in no_action_gradient_source
    assert "gradient is not None" in no_action_gradient_source

    common_source = _function_source(
        bridge_source,
        bridge_tree,
        "_screen_common_delta_metrics",
    )
    window_source = _function_source(
        bridge_source,
        bridge_tree,
        "_screen_loss_window_statistics",
    )
    assert common_source.count("/ 2.0") >= 4
    assert ".double()" in common_source
    assert "half_delta_energy_ratio" in common_source
    assert "half_delta_alignment_cosine" in common_source
    assert "math.ceil(0.90 * window) - 1" in window_source
    assert "loss_trace[-window:]" in window_source
    assert "len(loss_trace) != 301" in window_source

    runner_functions = {
        node.name
        for node in ast.walk(runner_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_classify_screen" in runner_functions
    assert CARD_SHA256 in runner_source
    assert ROOT in runner_source
    assert NONCE in runner_source
    assert "run_cach_a2_screen" in runner_source
    assert "A2_DELTA_LIVE_COMMON_MODE_BLOCKED" in runner_source
    assert "A2_VALID_NEITHER" in runner_source

    # R1 fixes launcher identity only: the SSH alias is canonical metadata,
    # while platform.node() reports the server's observed hostname.  Never
    # reject huaxiyun merely because it is not the alias string H200.
    runner_literal_assignments: dict[str, object] = {}
    for node in runner_tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            runner_literal_assignments[node.targets[0].id] = node.value.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            runner_literal_assignments[node.target.id] = node.value.value
    assert runner_literal_assignments["EXPECTED_CANONICAL_SSH_ALIAS"] == "H200"
    assert runner_literal_assignments["EXPECTED_OBSERVED_HOSTNAME"] == "huaxiyun"
    assert "canonical_ssh_alias" in runner_source
    assert "expected_platform_node" in runner_source
    assert "observed_hostname" in runner_source

    def is_platform_node_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "node"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "platform"
        )

    for comparison in (
        node for node in ast.walk(runner_tree) if isinstance(node, ast.Compare)
    ):
        operands = (comparison.left, *comparison.comparators)
        compares_h200 = any(
            isinstance(operand, ast.Constant) and operand.value == "H200"
            for operand in operands
        )
        assert not (
            compares_h200 and any(is_platform_node_call(item) for item in operands)
        ), "platform.node() must not be compared with canonical alias H200"

    # Exact environment identity includes the CUDA local-version suffix.
    # A base-version comparison would admit a different Torch/CUDA build.
    toolchain = config["runtime"][
        "expected_toolchain_when_execution_is_authorized"
    ]
    assert toolchain["torch"] == "2.7.1+cu128"
    assert 'importlib.metadata.version("torch")' in runner_source

    def is_torch_distribution(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id == "torch_distribution"

    def is_toolchain_torch(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "toolchain"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "torch"
        )

    exact_torch_distribution_comparisons = [
        node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
        and len(node.comparators) == 1
        and (
            (
                is_torch_distribution(node.left)
                and is_toolchain_torch(node.comparators[0])
            )
            or (
                is_toolchain_torch(node.left)
                and is_torch_distribution(node.comparators[0])
            )
        )
    ]
    assert exact_torch_distribution_comparisons
    plus_split_calls = [
        node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "split"
        and any(
            isinstance(argument, ast.Constant) and argument.value == "+"
            for argument in node.args
        )
    ]
    assert not plus_split_calls, "Torch version must not discard its +cu128 suffix"
    assert '.split("+")[0]' not in runner_source
    for forbidden in (
        "VALID_REVIEW_PROVISIONAL_NO_VERDICT",
        "FROZEN_PROVISIONAL_CLAIM_PENDING_RECOVERY_NO_VERDICT",
        "claim_transaction_contract",
        "consume_review_token",
        "create_review_token",
        "reset_review_token",
        "derive_review_token",
    ):
        assert forbidden not in runner_source
    # The predecessor's consumed claim path is a required read-only pin.  Its
    # literal token-ledger path is therefore legal; claim publication logic is
    # not.  The runner must classify and freeze one in-root terminal RESULT.
    assert "predecessor_token_policy" in runner_source
    assert "atomic_claim_or_claim_recovery" in runner_source
    assert "valid_result_terminal_without_claim_ledger" in runner_source
    assert "DIRECT_FINAL_RESULT_INSIDE_ROOT_BEFORE_SINGLE_TERMINAL_FREEZE" in (
        runner_source
    )


@pytest.mark.skipif(
    os.environ.get("CACH_A2_R1_AUTHORIZED_CPU_TEST") != "1",
    reason="requires future explicit CACH-A2 R1 CPU helper/model authority",
)
def test_future_authorized_cpu_bias_free_zero_origin_and_helper_bypass(
    monkeypatch,
) -> None:
    assert os.environ.get("GDN_DISABLE_COMPILE") == "1"
    assert os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    assert os.environ.get("FUSED_GDN_PRECISION") == "0"
    torch = importlib.import_module("torch")
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "_lazy_init", _forbidden)
    monkeypatch.setattr(torch, "load", _forbidden)
    bridge = _load_bridge()

    spec = bridge.AV1BVendorGDNBridgeSpec()
    reference = bridge.AV1BVendorGDNBridgeArm(
        spec=spec,
        staging_variant=bridge.CACHStagingVariant.REF_GDN_CORRECTED,
    )
    candidate = copy.deepcopy(reference)
    candidate.enable_candidate_action_seam()
    assert candidate.arm_name == "CACH-A2"

    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    candidate_only = set(candidate_parameters) - set(reference_parameters)
    assert candidate_only == EXPECTED_CANDIDATE_ONLY_NAMES
    assert candidate.trainable_name_set == (
        reference.trainable_name_set | EXPECTED_CANDIDATE_ONLY_NAMES
    )
    assert all(".bias" not in name for name in candidate_only)
    assert candidate.action_conditioner[0].bias is None
    assert candidate.action_conditioner[2].bias is None
    assert all(
        block.action_output_projection is not None
        and block.action_output_projection.bias is None
        for block in candidate.blocks
    )
    assert all(
        not bool(
            block.action_output_projection.weight.detach().count_nonzero()
        )
        for block in candidate.blocks
    )

    candidate_buffers = dict(candidate.named_buffers())
    assert "no_action_slot" in candidate_buffers
    anchor = candidate_buffers["no_action_slot"]
    assert not anchor.requires_grad
    assert not bool(anchor.detach().count_nonzero())
    assert "no_action_slot" not in candidate_parameters
    zero = torch.zeros((2, 3, 20), dtype=torch.float32)
    assert torch.equal(candidate.action_conditioner(zero), zero)

    hidden = (
        torch.arange(192, dtype=torch.float32).reshape(1, 3, 64) / 193.0
    )
    action = (
        torch.arange(60, dtype=torch.float32).reshape(1, 3, 20) / 61.0
    )
    mask = torch.tensor([True, False, False]).view(1, 3, 1)
    conditioned, conditioner_calls = bridge.condition_zero_anchor_actions(
        action,
        mask,
        candidate.action_conditioner,
    )
    assert conditioned is not None
    assert conditioner_calls == 1
    assert not bool(conditioned[:, 1:].detach().count_nonzero())

    projection = candidate.blocks[0].action_output_projection
    assert projection is not None
    with torch.no_grad():
        projection.weight.fill_(1.0 / 64.0)
    output, residual, projection_calls = (
        bridge.apply_zero_anchor_action_residual(
            hidden,
            conditioned,
            mask,
            projection,
        )
    )
    assert projection_calls == 1
    assert torch.equal(output[:, 1:], hidden[:, 1:])
    assert not bool(residual[:, 1:].detach().count_nonzero())
    assert not torch.equal(output[:, :1], hidden[:, :1])

    inactive = torch.zeros_like(mask)
    no_condition, conditioner_calls = bridge.condition_zero_anchor_actions(
        action,
        inactive,
        candidate.action_conditioner,
    )
    assert no_condition is None
    assert conditioner_calls == 0
    bypass, bypass_residual, projection_calls = (
        bridge.apply_zero_anchor_action_residual(
            hidden,
            no_condition,
            inactive,
            projection,
        )
    )
    assert projection_calls == 0
    assert torch.equal(bypass, hidden)
    assert not bool(bypass_residual.detach().count_nonzero())

    common_names = set(reference_parameters) & set(candidate_parameters)
    assert common_names
    for name in sorted(common_names):
        left = reference_parameters[name].detach()
        right = candidate_parameters[name].detach()
        assert torch.equal(left, right), name
        assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
    assert not torch.cuda.is_initialized()


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("CACH_A2_R1_AUTHORIZED_GPU_TEST") != "1",
    reason="requires future exact CACH-A2 R1 single-GPU execution authority",
)
def test_future_authorized_vendor_theta0_bypass_gradient_and_jvp(
    monkeypatch,
) -> None:
    expected_uuid = os.environ.get("CACH_A2_R1_EXECUTION_GPU_UUID")
    assert expected_uuid and expected_uuid.startswith("GPU-")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == expected_uuid
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    assert os.environ.get("FUSED_GDN_PRECISION") == "0"
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    torch = importlib.import_module("torch")
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    bridge = _load_bridge()
    av1 = importlib.import_module("sana_wam.model.cach_av1_transition_proxy")
    monkeypatch.setattr(av1, "build_av1_pair", _forbidden)
    monkeypatch.setattr(av1, "run_av1_screen", _forbidden)
    monkeypatch.setattr(torch, "load", _forbidden)

    device = torch.device("cuda:0")
    spec = bridge.AV1BVendorGDNBridgeSpec()
    task = bridge.build_av1b_vendor_task(spec, device=device)
    pair = bridge.build_av1b_vendor_gdn_pair(spec, device=device)
    assert tuple(_tensor_leaves(task, torch))
    assert len(_vendor_modules(pair.reference)) == 20
    assert len(_vendor_modules(pair.candidate)) == 20

    reference = pair.reference(
        task,
        mode="correct",
        capture_diagnostics=True,
        run_action_diagnostic=False,
    )
    correct = pair.candidate(
        task,
        mode="correct",
        capture_diagnostics=True,
        run_action_diagnostic=False,
    )
    no_action = pair.candidate(
        task,
        mode="no_action",
        capture_diagnostics=True,
        run_action_diagnostic=False,
    )
    seam_disabled = pair.candidate(
        task,
        mode="seam_disabled",
        capture_diagnostics=True,
        run_action_diagnostic=False,
    )
    assert torch.equal(correct.video_prediction, reference.video_prediction)
    expected_mask = torch.tensor(
        [False, True, True, True, True],
        dtype=torch.bool,
        device=device,
    ).view(1, 5, 1).expand(8, 5, 1)
    assert torch.equal(correct.action_condition_valid_mask, expected_mask)
    assert correct.action_conditioner_calls_this_forward == 1
    assert correct.action_projection_calls_this_forward == 20
    assert no_action.action_conditioner_calls_this_forward == 0
    assert no_action.action_projection_calls_this_forward == 0
    assert seam_disabled.action_conditioner_calls_this_forward == 0
    assert seam_disabled.action_projection_calls_this_forward == 0
    assert torch.equal(no_action.video_prediction, seam_disabled.video_prediction)

    assert len(correct.block_post_vendor_hidden) == 20
    assert len(correct.block_post_seam_hidden) == 20
    assert len(correct.block_action_residuals) == 20
    for post_vendor, post_seam, residual in zip(
        correct.block_post_vendor_hidden,
        correct.block_post_seam_hidden,
        correct.block_action_residuals,
        strict=True,
    ):
        assert torch.equal(post_seam[:, 0], post_vendor[:, 0])
        assert not bool(residual[:, 0].detach().count_nonzero())
    for post_vendor, post_seam, residual in zip(
        no_action.block_post_vendor_hidden,
        no_action.block_post_seam_hidden,
        no_action.block_action_residuals,
        strict=True,
    ):
        assert torch.equal(post_seam, post_vendor)
        assert not bool(residual.detach().count_nonzero())

    named = dict(pair.candidate.named_parameters())
    candidate_only_parameters = tuple(
        named[name] for name in sorted(EXPECTED_CANDIDATE_ONLY_NAMES)
    )
    gradients = torch.autograd.grad(
        no_action.video_prediction.float().sum(),
        candidate_only_parameters,
        allow_unused=True,
        materialize_grads=False,
    )
    assert all(gradient is None for gradient in gradients)

    no_action_jvp = bridge.last_block_local_action_condition_jvp(
        pair.candidate,
        no_action,
        mode="no_action",
    )
    correct_jvp = bridge.last_block_local_action_condition_jvp(
        pair.candidate,
        correct,
        mode="correct",
    )
    seam_parameter_jvp = bridge.last_block_local_seam_parameter_jvp(
        pair.candidate,
        correct,
    )
    assert float(no_action_jvp.rms.detach().item()) == 0.0
    assert float(correct_jvp.rms.detach().item()) == 0.0
    assert float(seam_parameter_jvp.rms.detach().item()) > 1.0e-8

    loss = (
        correct.video_prediction[:, 1:5] - task.video_target[:, 1:5]
    ).square().mean()
    seam_names = tuple(
        name
        for name in sorted(EXPECTED_CANDIDATE_ONLY_NAMES)
        if name.endswith("action_output_projection.weight")
    )
    seam_gradients = torch.autograd.grad(
        loss,
        tuple(named[name] for name in seam_names),
        allow_unused=True,
    )
    assert all(gradient is not None for gradient in seam_gradients)
    present_gradients = tuple(
        gradient for gradient in seam_gradients if gradient is not None
    )
    assert all(bool(torch.isfinite(value).all()) for value in present_gradients)
    assert _rms(present_gradients) > 1.0e-8


def _valid_screen_metrics() -> dict[str, Any]:
    by_horizon_good = {str(index): 0.10 for index in (1, 2, 3, 4)}
    gap_horizon_good = {str(index): 0.30 for index in (1, 2, 3, 4)}
    common = {
        "paired_prediction_common_mse": 1.2,
        "half_delta_mse": 0.01,
        "half_delta_nmse": 0.10,
        "half_delta_energy_ratio": 1.0,
        "half_delta_alignment_cosine": 0.95,
        "target_total_energy": 1.0,
        "paired_common_mse_div_target_total_energy": 1.2,
        "correct_mse_div_target_total_energy": 0.2,
    }
    return {
        "candidate_gain_vs_reference": 0.60,
        "loss_drop_rel": 0.90,
        "counterfactual_delta_nmse_correct": 0.10,
        "counterfactual_delta_nmse_correct_by_horizon": by_horizon_good,
        "shuffle_gap_rel": 0.60,
        "shuffle_gap_rel_by_horizon": gap_horizon_good,
        "no_action_gap_rel": 0.60,
        "no_action_gap_rel_by_horizon": gap_horizon_good,
        "seam_grad_rms_step0": 1.0e-4,
        "seam_update_rms_final": 1.0e-3,
        "last_block_local_raw_action_jvp_rms_step300": 1.0e-4,
        "paired_common_delta": {
            "aggregate": dict(common),
            "by_horizon": {
                str(index): dict(common) for index in (1, 2, 3, 4)
            },
        },
    }


@pytest.mark.skipif(
    os.environ.get("CACH_A2_R1_AUTHORIZED_CLASSIFIER_TEST") != "1",
    reason="requires future explicit pure terminal-classifier authority",
)
def test_future_authorized_terminal_classifier_four_valid_branches() -> None:
    runner = _load_runner_module()
    config = _load_config()

    def screen(metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "validity": {
                "implementation_valid": True,
                "data_valid": True,
                "numerics_valid": True,
                "finite_all": True,
                "all_card_validity_checks_pass": True,
                "reasons": [],
                "checks": {
                    "same_arm_no_action_vs_seam_disabled_bitwise_equal_theta0_and_final": True,
                    "same_arm_no_action_vs_seam_disabled_max_abs_exact_zero": True,
                    "no_action_candidate_only_parameter_gradients_all_none_theta0_and_final": True,
                    "no_action_raw_action_jvp_exact_zero_theta0_and_final": True,
                },
            },
            "metrics": dict(metrics),
        }

    go_metrics = _valid_screen_metrics()
    assert runner._classify_screen(screen(go_metrics), config) == (
        "OPERATOR_GO",
        None,
    )

    common_blocked_metrics = _valid_screen_metrics()
    common_blocked_metrics["shuffle_gap_rel"] = 0.10
    common_blocked_metrics["no_action_gap_rel"] = 0.10
    common_blocked_metrics["shuffle_gap_rel_by_horizon"] = {
        str(index): 0.10 for index in (1, 2, 3, 4)
    }
    common_blocked_metrics["no_action_gap_rel_by_horizon"] = {
        str(index): 0.10 for index in (1, 2, 3, 4)
    }
    assert runner._classify_screen(screen(common_blocked_metrics), config) == (
        "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED",
        "A2_DELTA_LIVE_COMMON_MODE_BLOCKED",
    )

    stop_metrics = _valid_screen_metrics()
    stop_metrics.update(
        {
            "candidate_gain_vs_reference": 0.01,
            "loss_drop_rel": 0.05,
            "counterfactual_delta_nmse_correct": 1.0,
            "shuffle_gap_rel": 0.01,
            "no_action_gap_rel": 0.01,
        }
    )
    assert runner._classify_screen(screen(stop_metrics), config) == (
        "OPERATOR_STOP",
        None,
    )

    neither_metrics = _valid_screen_metrics()
    neither_metrics["counterfactual_delta_nmse_correct"] = 0.50
    neither_metrics["shuffle_gap_rel"] = 0.20
    assert runner._classify_screen(screen(neither_metrics), config) == (
        "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED",
        "A2_VALID_NEITHER",
    )
