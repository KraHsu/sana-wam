"""Contracts for the AV-1B review300 vendor-GDN successor.

The default tests are text/config/card checks only.  They must not import the
bridge or runner, initialize torch/CUDA, create a run or ledger namespace,
write a token claim, or execute a model.  Construction and reserved-GPU
smokes are future-execution-only and require explicit environment gates in
addition to the exact frozen execution authority.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import fields, is_dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = (
    REPO_ROOT
    / "docs/cach_sana_wam/architecture_validation/av1b_review300/AV1B_REVIEW300_RUN_CARD.json"
)
CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_av1b_vendor_gdn_bridge_review300.yaml"
BRIDGE_PATH = REPO_ROOT / "src/sana_wam/model/cach_av1b_vendor_gdn_bridge_review300.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_cach_av1b_vendor_gdn_bridge_review300.py"
TEST_PATH = REPO_ROOT / "tests/test_cach_av1b_vendor_gdn_bridge_review300.py"

CARD_SHA256 = "a552522a951384996aa8f18ce9cb8a5e6e130ac4d2db68515b5ea6303c6722bc"
ROOT = (
    "/DATA/share/sana_cach_wam_nonformal_screens/av1b_review300/06f5d09127f8/"
    "av1b-review300-bd32b08576c9d598c629b90286331b87"
)
NONCE = "bd32b08576c9d598c629b90286331b87"
GPU_UUID = "GPU-41c95a43-ce96-fff3-33e0-739a3931d603"
TOKEN_ID = "cadfd92f1c72ebd9cc9aa8bfad029ddc6f0e72d9f7942f34e74629ca1ae37252"
CLAIM_PATH = (
    "/DATA/share/sana_cach_wam_nonformal_screens/token_ledgers/"
    "c73a943047295338cd9de14e6e53199b463ea66ae92b99d4b0b3393f5ec34030/"
    "cadfd92f1c72ebd9cc9aa8bfad029ddc6f0e72d9f7942f34e74629ca1ae37252.claim.json"
)
PREDECESSOR_ROOT = (
    "/DATA/share/sana_cach_wam_nonformal_screens/av1b/06f5d09127f8/"
    "av1b-1a9a3a6ca4b18fac904fdb806c2a8fb3"
)
PREDECESSOR_RESULT_SHA256 = (
    "d33e2e6bdf59cc1d2ffe51b32d48ce22daa4da1a902e3e9bded28cce29adb5de"
)
TASK_RECIPE_SHA256 = "9d2a749d722775324ee233862772def530fbffe0ea9e55aeea87c160b87a51e4"
SHUFFLE_PERMUTATION = (1, 0, 3, 2, 5, 4, 7, 6)
SOURCE_STATEMENT = (
    "授权在 H200 严格按 AV1B_REVIEW300_RUN_CARD.json SHA256 "
    "a552522a951384996aa8f18ce9cb8a5e6e130ac4d2db68515b5ea6303c6722bc "
    "执行 source implementation phase：仅排他新增卡内固定的四个 review300 "
    "bridge/config/runner/test 文件，保持原 AV‑1B card及原四文件字节不变；仅允许"
    "200→300 steps、相关 final-step 标签和 token claim harness 的机械变化。允许静态 "
    "JSON/AST/schema/SHA 校验，完成后将四文件冻结为 0444并报告完整 SHA；禁止创建运行"
    "或 token namespace、运行 root、claim、GPU/CUDA/JIT、模型或测试执行、参数更新、"
    "训练、真实数据及 checkpoint。"
)
SOURCE_STATEMENT_SHA256 = (
    "2bd18d09c77ae03451c0fa46e0270793c4e19802bab3906057103f405b40b573"
)

EXPECTED_TOP_LEVEL_CONFIG_KEYS = {
    "schema",
    "identity",
    "authority",
    "classification",
    "runtime",
    "topology",
    "recipe",
    "initialization",
    "optimizer",
    "budget",
    "diagnostics",
    "metrics",
    "thresholds",
    "review_token",
    "artifacts",
}
EXPECTED_REVIEW300_PATHS = [
    "src/sana_wam/model/cach_av1b_vendor_gdn_bridge_review300.py",
    "configs/experiments/cach_av1b_vendor_gdn_bridge_review300.yaml",
    "scripts/run_cach_av1b_vendor_gdn_bridge_review300.py",
    "tests/test_cach_av1b_vendor_gdn_bridge_review300.py",
]
EXPECTED_DURABILITY_SEQUENCE = [
    "WRITE_CANONICAL_JSON_THROUGH_O_CREAT_O_EXCL_O_NOFOLLOW_REGULAR_FILE_DESCRIPTOR",
    "FSYNC_CLAIM_FILE_CONTENT",
    "FCHMOD_CLAIM_FILE_0444",
    "FSYNC_CLAIM_FILE_AFTER_FCHMOD",
    "FSYNC_LEDGER_LEAF_DIRECTORY_FOR_DIRECTORY_ENTRY",
    "FCHMOD_LEDGER_LEAF_DIRECTORY_0555",
    "FSYNC_LEDGER_LEAF_DIRECTORY_AFTER_FCHMOD",
    "FSYNC_LEDGER_PARENT_AND_ANY_NEWLY_CREATED_ANCESTOR_IN_REVERSE_ORDER",
]
EXPECTED_CLAIM_BINDINGS = [
    "token_id_and_selected_choice",
    "this_full_card_path_and_sha256",
    "predecessor_full_card_result_raw_evidence_and_receipt_pins",
    "materialized_review300_source_pins",
    "new_exact_root_and_nonce",
    "fresh_initialization_digest",
    "current_frozen_provisional_result_path_and_sha256",
    "current_raw_evidence_path_and_sha256",
    "current_freeze_receipt_path_and_sha256",
    "new_choice_authority_artifact_path_and_full_sha256",
    "new_source_and_execution_authority_run_context_artifact_path_and_full_sha256",
    "choice_source_and_execution_authority_statement_bytes_and_sha256",
    "final_exact_typed_verdict_and_all_final_metrics",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_card() -> dict[str, Any]:
    return json.loads(
        CARD_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object_pairs,
    )


def _load_config() -> dict[str, Any]:
    # The .yaml artifact is intentionally strict JSON (a YAML 1.2 subset), so
    # launcher preflight can remain standard-library-only.
    return json.loads(
        CONFIG_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object_pairs,
    )


def _load_bridge():
    return importlib.import_module("sana_wam.model.cach_av1b_vendor_gdn_bridge_review300")


def _tensor_leaves(value: Any, torch) -> Iterator[Any]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if is_dataclass(value):
        for field in fields(value):
            yield from _tensor_leaves(getattr(value, field.name), torch)
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
        if type(module).__module__ == "diffusion.model.nets.sana_gdn_blocks_triton"
        and type(module).__name__ == "ChunkCausalGDNTriton"
    )


def _prediction(output):
    value = getattr(output, "video_prediction", None)
    assert value is not None, "AV1BVendorSequenceOutput.video_prediction is required"
    return value


def _diagnostic_rms(value: Any) -> float:
    for name in ("rms", "jvp_rms"):
        if hasattr(value, name):
            result = float(getattr(value, name))
            assert math.isfinite(result)
            return result
    raise AssertionError("AV1BLocalJVPDiagnostic must expose rms or jvp_rms")


def _forbidden(*_args, **_kwargs):
    raise AssertionError("forbidden proxy/data/checkpoint/CUDA fallback was used")


def test_frozen_card_config_source_authority_and_scope_contract() -> None:
    assert _sha256(CARD_PATH) == CARD_SHA256
    card = _load_card()
    config = _load_config()

    assert CONFIG_PATH.suffix == ".yaml"
    assert CONFIG_PATH.read_bytes().startswith(b"{")
    assert card["schema"] == "cach.architecture_validation.av1b_review300_run_card.v1"
    assert card["card_id"] == "cach-av1b-review300-no-bug-budget-extension-v1"
    assert card["card_state"] == "FROZEN_CARD_ONLY_EXECUTION_NOT_AUTHORIZED"
    assert card["classification"]["operator_level"] == "VENDOR_KERNEL"
    assert card["classification"]["integration_path"] == "EXPERIMENTAL_PATH"
    assert card["classification"]["only_semantic_delta"] == "CACH-A_ONLY_ACTION_TO_VIDEO_SEAM"
    assert card["classification"]["proxy_or_reference_operator_fallback"] == "FORBIDDEN"

    assert set(config) == EXPECTED_TOP_LEVEL_CONFIG_KEYS
    assert config["schema"] == "cach.av1b.vendor_gdn_bridge_review300.config.v1"
    identity = config["identity"]
    assert identity["run_card_path"] == card["identity"]["self_path"]
    assert identity["run_card_sha256"] == CARD_SHA256
    assert identity["resolved_root"] == card["root_contract"]["resolved_root"] == ROOT
    assert identity["run_nonce"] == card["root_contract"]["run_nonce"] == NONCE
    assert identity["gpu_physical_index"] == 7
    assert identity["gpu_uuid"] == GPU_UUID
    assert identity["predecessor_av1b_root"] == PREDECESSOR_ROOT
    assert identity["predecessor_av1b_result_sha256"] == PREDECESSOR_RESULT_SHA256

    generated = [entry["path"] for entry in identity["generated_paths"].values()]
    assert generated == EXPECTED_REVIEW300_PATHS
    assert card["additive_policy"]["exact_additive_paths"] == [
        card["identity"]["self_path"],
        *EXPECTED_REVIEW300_PATHS,
    ]
    assert identity["generated_paths"]["config"]["sha256"] == (
        "SELF_OUT_OF_BAND_FULL_LITERAL_SHA256"
    )
    for name in ("bridge", "runner", "test"):
        assert identity["generated_paths"][name]["sha256"] == (
            "SOURCE_MATERIALIZATION_BIND_FULL_SHA256"
        )

    authority = config["authority"]
    assert SOURCE_STATEMENT.encode("utf-8") == authority["source_statement"].encode("utf-8")
    assert len(SOURCE_STATEMENT.encode("utf-8")) == authority["source_statement_utf8_bytes"] == 643
    assert hashlib.sha256(SOURCE_STATEMENT.encode("utf-8")).hexdigest() == (
        authority["source_statement_sha256"]
        == SOURCE_STATEMENT_SHA256
    )
    assert authority["choice_statement"] == card["authority"]["selection_statement"]
    assert authority["choice_statement_sha256"] == card["authority"]["selection_statement_sha256"]
    assert authority["card_freeze_statement"] == card["authority"]["card_freeze_statement"]
    assert authority["card_freeze_statement_sha256"] == card["authority"]["card_freeze_statement_sha256"]
    assert authority["execution_statement"] == (
        "FUTURE_RUNTIME_BIND_VERBATIM_EXECUTION_AUTHORITY_FROM_STDIN"
    )
    assert authority["execution_statement_sha256"] == (
        "FUTURE_RUNTIME_BIND_SHA256_UTF8_WITHOUT_TRAILING_NEWLINE"
    )
    assert authority["execution_authority_state"] == "NOT_AUTHORIZED_AT_SOURCE_MATERIALIZATION"
    assert authority["execution_authority_must_bind_frozen_quartet_full_sha256"] is True
    assert authority["source_phase_capabilities"] == {
        "exclusive_four_path_create": True,
        "static_json_ast_schema_sha_validation": True,
        "chmod_four_paths_0444": True,
        "run_or_token_namespace_create": False,
        "root_create": False,
        "claim_create_or_consume": False,
        "gpu_or_cuda_or_jit": False,
        "torch_vendor_or_model_import": False,
        "model_or_test_execute": False,
        "optimizer_or_parameter_update": False,
        "synthetic_training": False,
        "real_data": False,
        "checkpoint": False,
    }

    for pin in identity["source_pins"].values():
        path = pin.get("path")
        digest = pin.get("sha256")
        if path is not None:
            assert _sha256(REPO_ROOT / path) == digest, path


def test_review300_scientific_semantics_budget_metrics_and_thresholds() -> None:
    card = _load_card()
    config = _load_config()

    expected_env = config["runtime"]["environment_before_any_torch_or_vendor_import"]
    assert expected_env == card["launcher_contract"][
        "environment_must_be_set_before_any_torch_or_vendor_import"
    ]
    assert expected_env["CUDA_VISIBLE_DEVICES"] == GPU_UUID
    assert expected_env["TRITON_CACHE_DIR"] == f"{ROOT}/triton_cache"
    assert config["runtime"]["allow_real_data"] is False
    assert config["runtime"]["allow_checkpoint"] is False
    assert config["runtime"]["allow_proxy_or_reference_fallback"] is False

    topology = config["topology"]
    card_topology = card["experiment_contract"]["topology"]
    for key in (
        "batch_size",
        "counterfactual_pair_count",
        "latent_channels",
        "valid_raw_count",
        "valid_latent_count",
        "valid_action_count",
        "action_dim",
        "context_dim",
        "vendor_hidden_dim",
        "vendor_heads",
        "vendor_head_dim",
        "vendor_gdn_depth",
        "vendor_conv_kernel_size",
        "vendor_k_conv_only",
        "vendor_qk_norm",
        "vendor_use_output_gate",
        "vendor_use_autograd_kernel",
        "frame_count",
        "chunk_count",
        "chunk_boundaries",
        "chunk_valid_latent_counts",
        "temporal_compression",
        "video_stride",
        "scored_future_horizons",
        "primary_horizons",
        "action_prediction_loss_weight",
    ):
        assert topology[key] == card_topology[key], key
    assert topology["HW"] == [5, 1, 1]
    assert topology["persistent_state_input_or_output"] is False

    recipe = config["recipe"]
    card_recipe = card["experiment_contract"]["synthetic_recipe"]
    assert recipe["task_recipe_sha256"] == card_recipe["task_recipe_sha256"] == TASK_RECIPE_SHA256
    assert tuple(recipe["shuffle_permutation"]) == SHUFFLE_PERMUTATION
    assert recipe["shuffle_permutation"] == card_recipe["shuffle_permutation"]
    assert recipe["end_of_bin_action_indices"] == (
        card_recipe["end_of_bin_action_indices_for_future_horizons_1_2_3_4"]
    )
    assert recipe["target_visibility"] == "TARGET_BYTES_ENTER_LOSS_ONLY"

    initialization = config["initialization"]
    card_initialization = card["experiment_contract"]["initialization"]
    assert initialization["seed0"] == card_initialization["seed0"]
    assert initialization["shared_named_seed"] == card_initialization["shared_named_seed"]
    assert initialization["candidate_only_seed"] == card_initialization["candidate_only_seed"]
    assert initialization["initializer_revision"] == card_initialization["initializer_revision"]
    assert initialization["fresh_initialization"] is True
    assert initialization["continuation_from_predecessor_parameters_optimizer_or_rng"] is False
    assert initialization["arm_execution_order"] == ["REF-GDN-CORRECTED", "CACH-A"]
    assert initialization["data_order"] == "SINGLE_FIXED_FULL_BATCH_NO_SHUFFLE"
    assert initialization["predecessor_theta0_manifest_sha256"] == (
        card_initialization["fresh_theta0_manifest_equivalence_to_predecessor"][
            "predecessor_sha256"
        ]
    )

    optimizer = config["optimizer"]
    assert optimizer["name"] == "AdamW"
    assert optimizer["learning_rate"] == 0.003
    assert optimizer["betas"] == [0.9, 0.99]
    assert optimizer["epsilon"] == 1.0e-8
    assert optimizer["weight_decay"] == 0.0
    assert optimizer["gradient_clip_global_l2"] == 1.0
    assert optimizer["optimizer_steps_per_arm"] == 300
    assert optimizer["metric_steps"] == [0, 300]
    assert optimizer["loss_trace_points_per_arm"] == 301
    assert optimizer["best_step_selection"] is False
    assert optimizer["intermediate_checkpoint_selection"] is False

    budget = config["budget"]
    assert budget["max_wall_seconds_total"] == 1350
    assert budget["max_optimizer_steps_total"] == 600
    assert budget["exact_optimizer_steps_per_arm"] == 300
    assert budget["completion_rule"] == (
        "BOTH_ARMS_COMPLETE_EXACTLY_300_STEPS_AND_ALL_REQUIRED_DIAGNOSTICS"
    )
    assert budget["automatic_budget_extension"] is False

    required_execution = config["diagnostics"]["required_execution"]
    assert "300_ADAMW_STEPS_PER_ARM" in required_execution
    assert "LAST_BLOCK_POST_VENDOR_LOCAL_ACTION_CONDITION_DIRECTIONAL_JVP_AT_STEP300" in (
        required_execution
    )
    assert all("STEP200" not in value and "200_ADAMW" not in value for value in required_execution)
    assert "step300_action_condition_tangent" in config["diagnostics"]["local_jvp"]
    assert "step200_action_condition_tangent" not in config["diagnostics"]["local_jvp"]

    metrics = config["metrics"]
    assert "STEP300" in metrics["loss_drop_rel_formula"]
    assert "STEP300" in metrics["shuffle_gap_rel_formula"]
    assert "STEP300" in metrics["no_action_gap_rel_formula"]
    assert "STEP300" in metrics["candidate_gain_vs_reference_formula"]
    assert "STEP200" not in json.dumps(metrics, sort_keys=True)
    assert metrics["final_step_only"] is True
    assert metrics["best_step_forbidden"] is True

    validity = config["thresholds"]["validity"]
    go = config["thresholds"]["operator_go"]
    stop = config["thresholds"]["operator_strong_stop"]
    assert validity["last_block_local_action_condition_jvp_rms_step300_gt"] == 1.0e-8
    assert validity["exactly_300_optimizer_steps_completed_per_arm"] is True
    assert go["loss_drop_rel_min"] == 0.8
    assert go["counterfactual_delta_nmse_correct_max"] == 0.25
    assert go["shuffle_gap_rel_min"] == 0.5
    assert go["no_action_gap_rel_min"] == 0.5
    assert go["candidate_gain_vs_reference_min"] == 0.5
    assert stop["loss_drop_rel_max"] == 0.1
    assert stop["counterfactual_delta_nmse_correct_min"] == 0.9
    assert stop["shuffle_gap_rel_max"] == 0.05
    assert stop["no_action_gap_rel_max"] == 0.05
    assert stop["candidate_gain_vs_reference_max"] == 0.05


def test_review_token_provisional_freeze_and_atomic_claim_contract() -> None:
    card = _load_card()
    config = _load_config()
    token = config["review_token"]
    card_token = card["review_token"]
    claim = card_token["claim_transaction_contract"]

    assert token["token_id"] == card_token["token_id"] == TOKEN_ID
    assert token["state_before_execution"] == "UNCONSUMED"
    assert token["selected_choice"] == card_token["selected_choice"]
    assert token["rejected_choice"] == card_token["rejected_choice"]
    assert token["both_choices_forbidden"] is True
    assert token["canonical_claim_path"] == card_token["canonical_claim_path"] == CLAIM_PATH
    assert token["latest_state_head_path"] == card_token["latest_token_state_head"]["path"]
    assert token["latest_state_head_sha256"] == PREDECESSOR_RESULT_SHA256
    assert token["provisional_root_typed_state"] == (
        claim["provisional_root_typed_state"]
        == "VALID_REVIEW_PROVISIONAL_NO_VERDICT"
    )
    assert token["final_result_location"] == "CANONICAL_ATOMIC_TOKEN_CLAIM_OUTSIDE_FROZEN_ROOT"
    assert token["invalid_or_blocked_attempts_claim"] is False
    assert token["serial_review_execution"] is True
    assert token["claim_atomic_create_flags"] == ["O_CREAT", "O_EXCL", "O_NOFOLLOW"]
    assert token["claim_file_final_mode"] == "0444"
    assert token["ledger_leaf_final_mode"] == "0555"
    assert token["durability_sequence"] == EXPECTED_DURABILITY_SEQUENCE
    assert token["durability_sequence"] == (
        claim["ledger_namespace_parent_contract"]["successful_claim_durability_sequence"]
    )
    assert token["claim_payload_required_bindings"] == EXPECTED_CLAIM_BINDINGS
    assert token["claim_payload_required_bindings"] == claim["claim_payload_required_bindings"]
    assert token["no_postclaim_root_or_verdict_write"] is True
    assert token["claim_race_loser_state"] == (
        "HARNESS_REJECTED_TOKEN_CLAIM_CONFLICT_NO_ARCHITECTURE_VERDICT"
    )

    recovery = token["post_provisional_preclaim_failure"]
    assert recovery["external_typed_state"] == (
        "FROZEN_PROVISIONAL_CLAIM_PENDING_RECOVERY_NO_VERDICT"
    )
    assert recovery["model_optimizer_or_data_rerun"] is False
    assert recovery["automatic_claim_retry"] is False
    assert recovery["future_exact_claim_only_recovery_authority_required"] is True
    assert recovery["partial_claim_requires_separate_ledger_recovery_authority"] is True
    assert recovery["frozen_root_mutation"] is False

    artifacts = config["artifacts"]
    assert artifacts["provisional_result_typed_state"] == (
        "VALID_REVIEW_PROVISIONAL_NO_VERDICT"
    )
    assert artifacts["provisional_result_must_not_publish_final_verdict"] is True
    assert artifacts["freeze_root_before_claim"] is True
    assert artifacts["modify_after_freeze"] is False
    assert artifacts["formal_promotion"] is False

    verdict = card["verdict_contract"]
    assert verdict["provisional_result_forbidden"] == [
        "FINAL_OPERATOR_GO_STOP_OR_INCONCLUSIVE_VERDICT",
        "TOKEN_CONSUMED_CLAIM",
        "STAGE_UNLOCK",
        "POST_FREEZE_MUTATION",
    ]
    assert verdict["successful_claim_final_verdicts"] == {
        "all_go_conditions": "OPERATOR_GO",
        "all_strong_stop_conditions": "OPERATOR_STOP",
        "valid_neither_all_go_nor_all_strong_stop": (
            "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED"
        ),
    }


def test_review300_bridge_and_runner_static_contract() -> None:
    bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    bridge_tree = ast.parse(bridge_source, filename=str(BRIDGE_PATH))
    runner_tree = ast.parse(runner_source, filename=str(RUNNER_PATH))

    imported_from: dict[str, set[str]] = {}
    imported_modules: set[str] = set()
    function_names: set[str] = set()
    class_names: set[str] = set()
    for node in ast.walk(bridge_tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_modules.add(module)
            imported_from.setdefault(module, set()).update(alias.name for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            class_names.add(node.name)

    direct_vendor = "diffusion.model.nets.sana_gdn_blocks_triton" in imported_modules
    pinned_predecessor_bridge = (
        "sana_wam.model.cach_av1b_vendor_gdn_bridge" in imported_modules
    )
    assert direct_vendor or pinned_predecessor_bridge
    if direct_vendor:
        assert "ChunkCausalGDNTriton" in imported_from[
            "diffusion.model.nets.sana_gdn_blocks_triton"
        ]
    assert {
        "build_av1b_vendor_task",
        "build_av1b_vendor_gdn_pair",
        "pure_torch_last_block_tail",
        "last_block_local_seam_parameter_jvp",
        "last_block_local_action_condition_jvp",
        "run_av1b_screen",
    } <= function_names or pinned_predecessor_bridge
    assert {
        "AV1BVendorGDNBridgeSpec",
        "AV1BVendorTask",
        "AV1BVendorTaskChunk",
        "AV1BVendorGDNBridgeArm",
        "AV1BVendorGDNBridgePair",
        "AV1BVendorSequenceOutput",
        "AV1BLocalJVPDiagnostic",
    } <= class_names or pinned_predecessor_bridge
    assert "fused_gdn_chunkwise" not in imported_modules
    assert "sana_v2v_attn_blocks" not in imported_modules
    assert CARD_SHA256 in bridge_source
    assert TASK_RECIPE_SHA256 in bridge_source
    assert "(0, 3, 5)" in bridge_source or "[0, 3, 5]" in bridge_source

    runner_names = {
        node.id for node in ast.walk(runner_tree) if isinstance(node, ast.Name)
    }
    assert "torch" not in runner_names or "main" in {
        node.name for node in ast.walk(runner_tree) if isinstance(node, ast.FunctionDef)
    }
    for literal in (
        CARD_SHA256,
        ROOT,
        NONCE,
        CLAIM_PATH,
        SOURCE_STATEMENT_SHA256,
        "VALID_REVIEW_PROVISIONAL_NO_VERDICT",
        "FUTURE_RUNTIME_BIND_VERBATIM_EXECUTION_AUTHORITY_FROM_STDIN",
        "O_EXCL",
        "O_NOFOLLOW",
        "fsync",
        "fchmod",
        "300",
        "301",
        "1350",
    ):
        assert literal in runner_source, literal
    assert "cach_av1b_vendor_gdn_bridge_review300" in runner_source
    assert "import yaml" not in runner_source
    assert "_duplicate_rejecting_yaml" not in runner_source
    assert RUNNER_PATH.name == "run_cach_av1b_vendor_gdn_bridge_review300.py"
    assert TEST_PATH.name == "test_cach_av1b_vendor_gdn_bridge_review300.py"


@pytest.mark.skipif(
    os.environ.get("CACH_AV1B_REVIEW300_AUTHORIZED_MODEL_TEST") != "1",
    reason="requires future exact review300 model-test authority",
)
def test_future_authorized_cpu_construction_preserves_old_scientific_semantics(
    monkeypatch,
) -> None:
    assert os.environ.get("GDN_DISABLE_COMPILE") == "1"
    assert os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    assert os.environ.get("FUSED_GDN_PRECISION") == "0"
    torch = importlib.import_module("torch")
    assert not torch.cuda.is_initialized()
    bridge = _load_bridge()
    av1 = importlib.import_module("sana_wam.model.cach_av1_transition_proxy")
    monkeypatch.setattr(av1, "build_av1_pair", _forbidden)
    monkeypatch.setattr(av1, "run_av1_screen", _forbidden)
    monkeypatch.setattr(torch, "load", _forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _forbidden)

    spec = bridge.AV1BVendorGDNBridgeSpec()
    assert spec.batch_size == 8
    assert spec.hidden_dim == 64
    assert spec.heads == 2
    assert spec.head_dim == 32
    assert spec.depth == 20
    assert tuple(spec.chunk_boundaries) == (0, 3, 5)
    assert spec.use_autograd_kernel is True
    assert spec.conv_kernel_size == 0

    source_spec = av1.AV1TransitionProxySpec()
    task = av1.build_av1_synthetic_task(spec=source_spec)
    assert task.recipe_digest == TASK_RECIPE_SHA256
    tensors = tuple(_tensor_leaves(task, torch))
    assert tensors
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert all(
        bool(torch.isfinite(tensor).all())
        for tensor in tensors
        if tensor.is_floating_point()
    )

    with pytest.raises(ValueError, match="cuda:0"):
        bridge.build_av1b_vendor_task(spec, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="cuda:0"):
        bridge.build_av1b_vendor_gdn_pair(spec, device=torch.device("cpu"))

    reference = bridge.AV1BVendorGDNBridgeArm(
        spec=spec,
        staging_variant=bridge.CACHStagingVariant.REF_GDN_CORRECTED,
    )
    candidate = copy.deepcopy(reference)
    candidate.enable_candidate_action_seam()
    reference.eval()
    candidate.eval()
    assert reference is not candidate
    reference_vendor = _vendor_modules(reference)
    candidate_vendor = _vendor_modules(candidate)
    assert len(reference_vendor) == len(candidate_vendor) == 20
    assert all(
        module.use_autograd_kernel is True
        for module in (*reference_vendor, *candidate_vendor)
    )

    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert not {
        name
        for name in reference_parameters
        if "action_conditioner" in name or "action_output_projection" in name
    }
    seam_names = sorted(
        name
        for name in candidate_parameters
        if name.endswith("action_output_projection.weight")
        or name.endswith("action_output_projection.bias")
    )
    assert len(seam_names) == 40
    assert all(
        not bool(candidate_parameters[name].detach().count_nonzero())
        for name in seam_names
    )

    common_names = set(reference_parameters) & set(candidate_parameters)
    assert common_names
    for name in sorted(common_names):
        left = reference_parameters[name].detach()
        right = candidate_parameters[name].detach()
        assert left.dtype == right.dtype
        assert tuple(left.shape) == tuple(right.shape)
        assert torch.equal(left, right), name
        assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
    assert not torch.cuda.is_initialized()


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("CACH_AV1B_REVIEW300_AUTHORIZED_GPU_TEST") != "1",
    reason="requires future exact review300 reserved-GPU authority",
)
def test_future_authorized_vendor_forward_backward_and_local_jvp_smoke(
    monkeypatch,
) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    assert os.environ.get("GDN_DISABLE_COMPILE") == "1"
    assert os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    assert os.environ.get("FUSED_GDN_PRECISION") == "0"
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    assert os.environ.get("TRITON_CACHE_DIR") == f"{ROOT}/triton_cache"

    torch = importlib.import_module("torch")
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert "H200" in torch.cuda.get_device_name(0)
    bridge = _load_bridge()
    av1 = importlib.import_module("sana_wam.model.cach_av1_transition_proxy")
    monkeypatch.setattr(av1, "build_av1_pair", _forbidden)
    monkeypatch.setattr(av1, "run_av1_screen", _forbidden)
    monkeypatch.setattr(torch, "load", _forbidden)

    device = torch.device("cuda:0")
    spec = bridge.AV1BVendorGDNBridgeSpec()
    task = bridge.build_av1b_vendor_task(spec, device=device)
    pair = bridge.build_av1b_vendor_gdn_pair(spec, device=device)
    reference_vendor = _vendor_modules(pair.reference)
    candidate_vendor = _vendor_modules(pair.candidate)
    assert len(reference_vendor) == len(candidate_vendor) == 20

    vendor_calls: list[int] = []
    hooks = [
        module.register_forward_hook(
            lambda module, _inputs, _output: vendor_calls.append(id(module))
        )
        for module in candidate_vendor
    ]
    try:
        candidate_output = pair.candidate(
            task,
            mode="correct",
            target_override=None,
            capture_diagnostics=True,
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert len(vendor_calls) == 20
    assert len(set(vendor_calls)) == 20

    reference_output = pair.reference(
        task,
        mode="correct",
        target_override=None,
        capture_diagnostics=True,
    )
    candidate_prediction = _prediction(candidate_output)
    reference_prediction = _prediction(reference_output)
    assert tuple(candidate_prediction.shape) == (8, 5, 3, 1, 1)
    assert candidate_prediction.dtype is torch.float32
    assert bool(torch.isfinite(candidate_prediction).all())
    assert torch.equal(candidate_prediction, reference_prediction)

    parameters = dict(pair.candidate.named_parameters())
    seam_names = tuple(
        sorted(
            name
            for name, value in parameters.items()
            if value.requires_grad
            and name.endswith("action_output_projection.weight")
        )
    )
    assert len(seam_names) == 20
    loss = (candidate_prediction[:, 1:] - task.video_target[:, 1:]).square().mean()
    seam_gradients = torch.autograd.grad(
        loss,
        tuple(parameters[name] for name in seam_names),
        retain_graph=True,
        allow_unused=True,
    )
    assert all(gradient is not None for gradient in seam_gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in seam_gradients)
    seam_rms = math.sqrt(
        sum(gradient.double().square().sum().item() for gradient in seam_gradients)
        / sum(gradient.numel() for gradient in seam_gradients)
    )
    assert seam_rms > 1.0e-8

    jvp = bridge.last_block_local_seam_parameter_jvp(
        pair.candidate,
        candidate_output,
    )
    assert _diagnostic_rms(jvp) > 1.0e-8
    assert bool(getattr(jvp, "finite", True))
    assert getattr(jvp, "primal_digest", None)
    assert not hasattr(candidate_output, "final_state")
    assert not hasattr(candidate_output, "committed_state")
    assert not hasattr(candidate_output, "cache_commit")
