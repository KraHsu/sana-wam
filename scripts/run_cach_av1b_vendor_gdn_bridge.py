#!/usr/bin/env python3
"""Run the frozen AV-1B single-call vendor-GDN architecture screen.

The launcher intentionally imports only the Python standard library until the
card, the two-message authority chain, every source pin, the exact root, the
environment, and the reserved physical GPU have been checked.  Once the root
is exclusively created, every terminal path is fail-closed and freezes the
root.  Model, optimizer, checkpoint, and real-data bytes are never serialized.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, NoReturn, Sequence


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WORKTREE = Path("/home/zch/workspace/sana-wam")
EXPECTED_CARD = (
    "docs/cach_sana_wam/architecture_validation/av1b/AV1B_RUN_CARD.json"
)
EXPECTED_CARD_SHA256 = (
    "c0e15bcb8c20441d7d989ec6d375284e45f24962c98e15b984dbbbbd198248fb"
)
EXPECTED_CONFIG = "configs/experiments/cach_av1b_vendor_gdn_bridge.yaml"
EXPECTED_BRIDGE = "src/sana_wam/model/cach_av1b_vendor_gdn_bridge.py"
EXPECTED_RUNNER = "scripts/run_cach_av1b_vendor_gdn_bridge.py"
EXPECTED_TEST = "tests/test_cach_av1b_vendor_gdn_bridge.py"
EXPECTED_NONCE = "1a9a3a6ca4b18fac904fdb806c2a8fb3"
EXPECTED_ROOT = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/av1b/06f5d09127f8/"
    "av1b-1a9a3a6ca4b18fac904fdb806c2a8fb3"
)
EXPECTED_ROOT_NAMESPACE_BASE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens"
)
EXPECTED_ROOT_NAMESPACE_PARTS = ("av1b", "06f5d09127f8")
EXPECTED_GPU_PHYSICAL_INDEX = 7
EXPECTED_GPU_UUID = "GPU-41c95a43-ce96-fff3-33e0-739a3931d603"
EXPECTED_GPU_NAME = "NVIDIA H200"
EXPECTED_EXECUTION_AUTHORITY_SHA256 = (
    "d96e7681321ef58d1450632d6816be95733605fe392dc9781205ac7e4d565dd4"
)
EXPECTED_CLARIFICATION_AUTHORITY_SHA256 = (
    "6ba9eaf13617f87b87c09b0f3ca64b72403c84188011e0e9185f304c3365a654"
)
EXPECTED_CARD_FREEZE_AUTHORITY_SHA256 = (
    "52e996ad4395712e8a644589058c04d389318bd7b7678fcb163805d2fbc7b676"
)
EXPECTED_REVIEW_TOKEN = (
    "cadfd92f1c72ebd9cc9aa8bfad029ddc6f0e72d9f7942f34e74629ca1ae37252"
)
EXPECTED_REVIEW_CLAIM = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/token_ledgers/"
    "c73a943047295338cd9de14e6e53199b463ea66ae92b99d4b0b3393f5ec34030/"
    f"{EXPECTED_REVIEW_TOKEN}.claim.json"
)
EXPECTED_PLAN_SHA256 = (
    "06f5d09127f8bc2a945cdbdc095054f40b1fe8d9757a90eccbdf0604ea89e079"
)
EXPECTED_REPO_HEADS = {
    "sana_wam": "605f1c134b4c983ff80f8489c4bc8847036329e2",
    "third_party_sana": "16b9cec673e3335724ba2d8db25de7f9ed229292",
    "sana_afcc_handoff": "9586486f2a9f5172d57b325e32093a3e018d34c0",
}
AV1B_CODE_PATHS = (EXPECTED_BRIDGE, EXPECTED_CONFIG, EXPECTED_RUNNER, EXPECTED_TEST)
CARD_ADDITIVE_PATHS = (EXPECTED_CARD, *AV1B_CODE_PATHS)
EXPECTED_ENV = {
    "CUDA_VISIBLE_DEVICES": EXPECTED_GPU_UUID,
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "GDN_DISABLE_COMPILE": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "FUSED_GDN_PRECISION": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": (
        "/home/zch/workspace/sana-wam/src:"
        "/home/zch/workspace/sana-wam/third_party/Sana"
    ),
    "TRITON_CACHE_DIR": str(EXPECTED_ROOT / "triton_cache"),
}
REQUIRED_ROOT_ARTIFACTS = (
    "RUN_CONTEXT.json",
    "PROGRESS_EVENTS.json",
    "THETA0_MANIFESTS.json",
    "LOSS_TRACE.json",
    "RAW_METRICS.json",
    "RAW_EVIDENCE.json",
    "FINAL_PARAMETER_DIGESTS.json",
    "JIT_INVENTORY.json",
    "RESULT.json",
    "FREEZE_RECEIPT.json",
)
ALLOWED_TERMINAL_STATES = frozenset(
    {
        "AUTH_BLOCKED",
        "ENV_BLOCKED",
        "HARNESS_REJECTED",
        "INVALID_RUN",
        "IMPLEMENTATION_INVALID",
        "NUMERICAL_INVALID",
        "OPERATOR_UNVERIFIED",
        "REVIEW_ONCE",
        "OPERATOR_GO",
    }
)
EXPECTED_ALLOWED_CAPABILITIES = frozenset(
    {
        "code_write",
        "root_create",
        "single_gpu",
        "cuda",
        "triton_jit",
        "vendor_autograd_kernel",
        "model_execute",
        "forward_backward",
        "jvp",
        "optimizer",
        "parameter_update",
        "synthetic_nonformal_training",
        "lightweight_tests",
        "diagnostic_metrics",
    }
)
EXPECTED_DENIED_CAPABILITIES = frozenset(
    {
        "real_data",
        "checkpoint_load",
        "checkpoint_save",
        "review_token_consume",
        "full_2b",
        "formal_evaluation",
        "admission",
        "deploy_capture",
        "av2",
        "global_stage3",
    }
)


class PreflightError(RuntimeError):
    """A typed failure raised before the immutable run root exists."""

    def __init__(self, state: str, token: str, detail: str):
        super().__init__(f"{token}: {detail}")
        self.state = state
        self.token = token
        self.detail = detail


class RunInterrupted(RuntimeError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _duplicate_rejecting_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs_hook)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _duplicate_rejecting_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise PreflightError("ENV_BLOCKED", "PYYAML_MISSING", str(exc)) from exc

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate YAML key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError("top-level YAML value must be a mapping")
    return value


def _require(condition: bool, state: str, token: str, detail: str) -> None:
    if not condition:
        raise PreflightError(state, token, detail)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    _require(
        actual == expected,
        "AUTH_BLOCKED",
        "SCHEMA_MISMATCH",
        f"{label}: expected={sorted(expected)!r} actual={sorted(actual)!r}",
    )


def _git_head(path: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        raise PreflightError(
            "ENV_BLOCKED", "GIT_HEAD_UNAVAILABLE", f"{path}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def _regular_repo_file(relative: str) -> Path:
    relative_path = Path(relative)
    _require(
        not relative_path.is_absolute() and ".." not in relative_path.parts,
        "AUTH_BLOCKED",
        "UNSAFE_REPO_PATH",
        relative,
    )
    path = REPO_ROOT / relative_path
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise PreflightError("ENV_BLOCKED", "PINNED_SOURCE_MISSING", relative) from exc
    _require(
        stat.S_ISREG(info.st_mode) and not path.is_symlink(),
        "ENV_BLOCKED",
        "PINNED_SOURCE_NOT_REGULAR",
        relative,
    )
    return path


def _verify_pin(relative: str, expected_sha256: str) -> None:
    observed = _sha256_file(_regular_repo_file(relative))
    _require(
        observed == expected_sha256,
        "AUTH_BLOCKED",
        "SOURCE_PIN_MISMATCH",
        f"{relative}: expected={expected_sha256} observed={observed}",
    )


def _verify_absolute_artifact(path_text: str, expected_sha256: str) -> None:
    path = Path(path_text)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise PreflightError("ENV_BLOCKED", "PREDECESSOR_ARTIFACT_MISSING", path_text) from exc
    _require(
        path.is_absolute() and stat.S_ISREG(info.st_mode) and not path.is_symlink(),
        "ENV_BLOCKED",
        "PREDECESSOR_ARTIFACT_UNSAFE",
        path_text,
    )
    observed = _sha256_file(path)
    _require(
        observed == expected_sha256,
        "AUTH_BLOCKED",
        "PREDECESSOR_ARTIFACT_PIN_MISMATCH",
        f"{path_text}: expected={expected_sha256} observed={observed}",
    )


def _validate_card(card_argument: Path) -> dict[str, Any]:
    expected_path = _regular_repo_file(EXPECTED_CARD)
    _require(
        card_argument.resolve(strict=True) == expected_path.resolve(strict=True),
        "AUTH_BLOCKED",
        "CARD_PATH_MISMATCH",
        str(card_argument),
    )
    _require(
        _sha256_file(expected_path) == EXPECTED_CARD_SHA256,
        "AUTH_BLOCKED",
        "CARD_FULL_SHA256_MISMATCH",
        EXPECTED_CARD,
    )
    try:
        card = _duplicate_rejecting_json(expected_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PreflightError("AUTH_BLOCKED", "CARD_PARSE_REJECTED", str(exc)) from exc

    _require(
        card.get("schema") == "cach.architecture_validation.av1b_run_card.v1"
        and card.get("card_id") == "cach-av1b-vendor-gdn-single-call-v1"
        and card.get("card_state") == "FROZEN_CARD_ONLY_EXECUTION_NOT_AUTHORIZED",
        "AUTH_BLOCKED",
        "CARD_IDENTITY_MISMATCH",
        "schema/card_id/card_state",
    )
    _require(
        card.get("additive_policy", {}).get("exact_additive_paths")
        == list(CARD_ADDITIVE_PATHS),
        "AUTH_BLOCKED",
        "ADDITIVE_PATH_SET_MISMATCH",
        repr(card.get("additive_policy", {}).get("exact_additive_paths")),
    )
    _require(
        card.get("identity", {}).get("repository_heads") == EXPECTED_REPO_HEADS,
        "AUTH_BLOCKED",
        "REPOSITORY_HEAD_CARD_MISMATCH",
        repr(card.get("identity", {}).get("repository_heads")),
    )
    _require(
        card.get("identity", {}).get("governing_plan", {}).get("sha256")
        == EXPECTED_PLAN_SHA256,
        "AUTH_BLOCKED",
        "PLAN_CARD_PIN_MISMATCH",
        repr(card.get("identity", {}).get("governing_plan")),
    )
    card_authority = card.get("authority", {})
    statement = card_authority.get("statement")
    _require(
        isinstance(statement, str)
        and _sha256_bytes(statement.encode("utf-8"))
        == EXPECTED_CARD_FREEZE_AUTHORITY_SHA256
        and card_authority.get("statement_sha256")
        == EXPECTED_CARD_FREEZE_AUTHORITY_SHA256,
        "AUTH_BLOCKED",
        "CARD_FREEZE_AUTHORITY_MISMATCH",
        "statement bytes or pin differ",
    )
    _require(
        card.get("classification", {}).get("operator_level") == "VENDOR_KERNEL"
        and card.get("classification", {}).get("integration_path")
        == "EXPERIMENTAL_PATH"
        and card.get("classification", {})
        .get("single_call_chunk_causal_semantics", {})
        .get("chunk_index")
        == [0, 3, 5],
        "AUTH_BLOCKED",
        "CARD_CLASSIFICATION_MISMATCH",
        "operator/integration/single-call chunks",
    )
    root_contract = card.get("root_contract", {})
    _require(
        root_contract.get("resolved_root") == str(EXPECTED_ROOT)
        and root_contract.get("run_nonce") == EXPECTED_NONCE
        and root_contract.get("required_artifacts_if_root_created")
        == list(REQUIRED_ROOT_ARTIFACTS),
        "AUTH_BLOCKED",
        "CARD_ROOT_CONTRACT_MISMATCH",
        repr(root_contract),
    )
    gpu = card.get("gpu_contract", {})
    _require(
        gpu.get("physical_index_at_card_freeze") == EXPECTED_GPU_PHYSICAL_INDEX
        and gpu.get("uuid") == EXPECTED_GPU_UUID
        and gpu.get("no_fallback_gpu") is True,
        "AUTH_BLOCKED",
        "CARD_GPU_CONTRACT_MISMATCH",
        repr(gpu),
    )
    review = card.get("review_token", {})
    _require(
        review.get("token_id") == EXPECTED_REVIEW_TOKEN
        and review.get("canonical_claim_path") == str(EXPECTED_REVIEW_CLAIM)
        and review.get("state") == "UNCONSUMED"
        and review.get("first_av1b_execution_must_not_consume") is True,
        "AUTH_BLOCKED",
        "REVIEW_TOKEN_CARD_MISMATCH",
        repr(review),
    )
    allowed = card.get("verdict_contract", {}).get("allowed_first_execution_states")
    _require(
        isinstance(allowed, list) and set(allowed) == set(ALLOWED_TERMINAL_STATES),
        "AUTH_BLOCKED",
        "TERMINAL_STATE_SET_MISMATCH",
        repr(allowed),
    )

    source_pins = card.get("source_pins")
    _require(
        isinstance(source_pins, Mapping),
        "AUTH_BLOCKED",
        "CARD_SOURCE_PINS_MISSING",
        repr(source_pins),
    )
    for label, item in source_pins.items():
        if label == "future_generated_files_pin_policy" or not isinstance(item, Mapping):
            continue
        relative = item.get("path")
        sha256 = item.get("sha256")
        _require(
            isinstance(relative, str) and isinstance(sha256, str),
            "AUTH_BLOCKED",
            "CARD_SOURCE_PIN_INVALID",
            f"{label}: {item!r}",
        )
        _verify_pin(relative, sha256)

    predecessor = card.get("identity", {}).get("predecessor_av1_artifacts")
    _require(
        isinstance(predecessor, Mapping) and predecessor,
        "AUTH_BLOCKED",
        "PREDECESSOR_PINS_MISSING",
        repr(predecessor),
    )
    for label, item in predecessor.items():
        _require(
            isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha256"), str),
            "AUTH_BLOCKED",
            "PREDECESSOR_PIN_INVALID",
            f"{label}: {item!r}",
        )
        _verify_absolute_artifact(item["path"], item["sha256"])
    predecessor_root = Path(card["identity"]["predecessor_av1_root"])
    _require(
        predecessor_root.is_dir()
        and not predecessor_root.is_symlink()
        and stat.S_IMODE(predecessor_root.stat().st_mode) == 0o555,
        "ENV_BLOCKED",
        "PREDECESSOR_ROOT_NOT_FROZEN",
        str(predecessor_root),
    )
    return card


def _validate_authority(authority: Mapping[str, Any]) -> None:
    execution = authority.get("execution_statement")
    clarification = authority.get("confirmation_statement")
    _require(
        isinstance(execution, str)
        and _sha256_bytes(execution.encode("utf-8"))
        == EXPECTED_EXECUTION_AUTHORITY_SHA256
        and authority.get("execution_statement_sha256")
        == EXPECTED_EXECUTION_AUTHORITY_SHA256,
        "AUTH_BLOCKED",
        "EXECUTION_AUTHORITY_PIN_MISMATCH",
        "original execution statement bytes or pin differ",
    )
    _require(
        isinstance(clarification, str)
        and _sha256_bytes(clarification.encode("utf-8"))
        == EXPECTED_CLARIFICATION_AUTHORITY_SHA256
        and authority.get("confirmation_statement_sha256")
        == EXPECTED_CLARIFICATION_AUTHORITY_SHA256,
        "AUTH_BLOCKED",
        "CLARIFICATION_AUTHORITY_PIN_MISMATCH",
        "clarification statement bytes or pin differ",
    )
    _require(
        authority.get("statement_sha256_rule")
        == "SHA256_UTF8_WITHOUT_TRAILING_NEWLINE"
        and authority.get("review_token_state") == "UNCONSUMED",
        "AUTH_BLOCKED",
        "AUTHORITY_INTERPRETATION_MISMATCH",
        repr(authority),
    )
    capabilities = authority.get("capabilities")
    _require(
        isinstance(capabilities, Mapping),
        "AUTH_BLOCKED",
        "AUTHORITY_CAPABILITIES_MISSING",
        repr(capabilities),
    )
    for name in EXPECTED_ALLOWED_CAPABILITIES:
        _require(
            capabilities.get(name) is True,
            "AUTH_BLOCKED",
            "CAPABILITY_NOT_AUTHORIZED",
            name,
        )
    for name in EXPECTED_DENIED_CAPABILITIES:
        _require(
            capabilities.get(name) is False,
            "AUTH_BLOCKED",
            "FORBIDDEN_CAPABILITY_ENABLED",
            name,
        )


def _validate_config(
    config: Mapping[str, Any], config_argument: Path, card: Mapping[str, Any]
) -> None:
    expected_path = _regular_repo_file(EXPECTED_CONFIG)
    _require(
        config_argument.resolve(strict=True) == expected_path.resolve(strict=True),
        "AUTH_BLOCKED",
        "CONFIG_PATH_MISMATCH",
        str(config_argument),
    )
    _require_exact_keys(
        config,
        {
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
            "artifacts",
        },
        "config",
    )
    _require(
        config.get("schema") == "cach.av1b.vendor_gdn_bridge.config.v1",
        "AUTH_BLOCKED",
        "CONFIG_SCHEMA_MISMATCH",
        repr(config.get("schema")),
    )
    identity = config["identity"]
    _require(
        isinstance(identity, Mapping)
        and identity.get("canonical_host") == "H200"
        and identity.get("canonical_worktree") == str(EXPECTED_WORKTREE)
        and identity.get("run_card_path") == EXPECTED_CARD
        and identity.get("run_card_sha256") == EXPECTED_CARD_SHA256
        and identity.get("governing_plan_sha256") == EXPECTED_PLAN_SHA256
        and identity.get("resolved_root") == str(EXPECTED_ROOT)
        and identity.get("run_nonce") == EXPECTED_NONCE
        and identity.get("gpu_physical_index") == EXPECTED_GPU_PHYSICAL_INDEX
        and identity.get("gpu_uuid") == EXPECTED_GPU_UUID,
        "AUTH_BLOCKED",
        "CONFIG_IDENTITY_MISMATCH",
        repr(identity),
    )
    _validate_authority(config["authority"])
    classification = config["classification"]
    _require(
        isinstance(classification, Mapping)
        and classification.get("operator_level") == "VENDOR_KERNEL"
        and classification.get("integration_path") == "EXPERIMENTAL_PATH"
        and classification.get("result_scope")
        == card["classification"]["result_scope"]
        and classification.get("proxy_or_reference_operator_fallback") == "FORBIDDEN",
        "AUTH_BLOCKED",
        "CONFIG_CLASSIFICATION_MISMATCH",
        repr(classification),
    )
    runtime = config["runtime"]
    _require(
        isinstance(runtime, Mapping)
        and runtime.get("device") == "cuda:0"
        and runtime.get("dtype") == "float32"
        and runtime.get("single_process") is True
        and runtime.get("allow_network") is False
        and runtime.get("allow_checkpoint") is False
        and runtime.get("allow_real_data") is False
        and runtime.get("allow_proxy_or_reference_fallback") is False
        and runtime.get("environment_before_any_torch_or_vendor_import")
        == EXPECTED_ENV,
        "AUTH_BLOCKED",
        "CONFIG_RUNTIME_MISMATCH",
        repr(runtime),
    )
    topology = config["topology"]
    card_topology = card["experiment_contract"]["topology"]
    for name in (
        "batch_size", "counterfactual_pair_count", "latent_channels",
        "valid_raw_count", "valid_latent_count", "valid_action_count",
        "action_dim", "context_dim", "vendor_hidden_dim", "vendor_heads",
        "vendor_head_dim", "vendor_gdn_depth", "vendor_conv_kernel_size",
        "vendor_k_conv_only", "vendor_qk_norm", "vendor_use_output_gate",
        "vendor_use_autograd_kernel", "frame_count", "chunk_count",
        "chunk_boundaries", "chunk_valid_latent_counts", "temporal_compression",
        "video_stride", "scored_future_horizons", "primary_horizons",
        "frozen_action_backbone", "action_prediction_loss_weight",
    ):
        _require(
            topology.get(name) == card_topology.get(name),
            "AUTH_BLOCKED", "CONFIG_CARD_SECTION_MISMATCH", f"topology.{name}",
        )
    optimizer = config["optimizer"]
    card_optimizer = card["experiment_contract"]["optimizer"]
    for name in (
        "name", "learning_rate", "betas", "epsilon", "weight_decay",
        "gradient_clip_global_l2", "optimizer_steps_per_arm", "metric_steps",
        "best_step_selection", "intermediate_checkpoint_selection",
        "write_optimizer_state", "unlisted_trainable_parameter_allowed",
    ):
        _require(
            optimizer.get(name) == card_optimizer.get(name),
            "AUTH_BLOCKED", "CONFIG_CARD_SECTION_MISMATCH", f"optimizer.{name}",
        )
    budget = config["budget"]
    card_budget = card["experiment_contract"]["budget"]
    for name in (
        "max_wall_seconds_total", "max_gpu_memory_bytes", "max_process_rss_bytes",
        "max_root_bytes", "max_optimizer_steps_total", "automatic_budget_extension",
        "completion_rule", "budget_failure_state",
    ):
        _require(
            budget.get(name) == card_budget.get(name),
            "AUTH_BLOCKED", "CONFIG_CARD_SECTION_MISMATCH", f"budget.{name}",
        )
    _require(
        config["recipe"].get("task_recipe_sha256")
        == card["experiment_contract"]["synthetic_recipe"]["task_recipe_sha256"]
        and config["initialization"].get("initializer_revision")
        == card["experiment_contract"]["initialization"]["initializer_revision"]
        and config["thresholds"].get("operator_go", {}).get("loss_drop_rel_min")
        == card["threshold_contract"]["operator_go_all_required"]["loss_drop_rel_min"],
        "AUTH_BLOCKED", "CONFIG_CARD_SECTION_MISMATCH", "recipe/init/threshold anchors",
    )
    artifacts = config["artifacts"]
    _require(
        isinstance(artifacts, Mapping)
        and artifacts.get("required_if_root_created")
        == list(REQUIRED_ROOT_ARTIFACTS)
        and artifacts.get("write_model_checkpoint") is False
        and artifacts.get("write_optimizer_state") is False
        and artifacts.get("freeze_root_on_every_postcreate_terminal") is True,
        "AUTH_BLOCKED",
        "CONFIG_ARTIFACT_SCOPE_MISMATCH",
        repr(artifacts),
    )


def _validate_environment(root_argument: Path, nonce: str, card: Mapping[str, Any]) -> None:
    _require(
        REPO_ROOT == EXPECTED_WORKTREE and Path.cwd().resolve() == EXPECTED_WORKTREE,
        "ENV_BLOCKED",
        "CANONICAL_WORKTREE_REQUIRED",
        f"repo={REPO_ROOT} cwd={Path.cwd().resolve()}",
    )
    _require(
        root_argument == EXPECTED_ROOT and str(root_argument) == str(EXPECTED_ROOT),
        "AUTH_BLOCKED",
        "ROOT_BINDING_MISMATCH",
        str(root_argument),
    )
    _require(
        nonce == EXPECTED_NONCE,
        "AUTH_BLOCKED",
        "NONCE_BINDING_MISMATCH",
        nonce,
    )
    _require(
        not root_argument.exists() and not root_argument.is_symlink(),
        "ENV_BLOCKED",
        "ROOT_ALREADY_EXISTS",
        str(root_argument),
    )
    namespace_base = EXPECTED_ROOT_NAMESPACE_BASE
    try:
        base_info = namespace_base.lstat()
        resolved_base = namespace_base.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreflightError(
            "ENV_BLOCKED", "ROOT_NAMESPACE_BASE_MISSING", str(namespace_base)
        ) from exc
    _require(
        stat.S_ISDIR(base_info.st_mode)
        and not namespace_base.is_symlink()
        and resolved_base == namespace_base,
        "ENV_BLOCKED",
        "ROOT_NAMESPACE_BASE_UNSAFE",
        str(namespace_base),
    )
    _require(
        base_info.st_uid == os.geteuid()
        and os.access(namespace_base, os.W_OK | os.X_OK),
        "ENV_BLOCKED",
        "ROOT_NAMESPACE_BASE_OWNER_OR_ACCESS_MISMATCH",
        (
            f"owner={base_info.st_uid} euid={os.geteuid()} "
            f"base={namespace_base}"
        ),
    )
    current = namespace_base
    for component in EXPECTED_ROOT_NAMESPACE_PARTS:
        current = current / component
        if not current.exists() and not current.is_symlink():
            continue
        try:
            info = current.lstat()
            resolved = current.resolve(strict=True)
        except FileNotFoundError as exc:
            raise PreflightError(
                "ENV_BLOCKED", "ROOT_NAMESPACE_RACE", str(current)
            ) from exc
        _require(
            stat.S_ISDIR(info.st_mode)
            and not current.is_symlink()
            and resolved == current
            and info.st_uid == os.geteuid()
            and os.access(current, os.W_OK | os.X_OK),
            "ENV_BLOCKED",
            "ROOT_NAMESPACE_COMPONENT_UNSAFE",
            str(current),
        )
    minimum_free = int(card["root_contract"]["minimum_free_bytes_before_create"])
    free_bytes = shutil.disk_usage(namespace_base).free
    _require(
        free_bytes >= minimum_free,
        "ENV_BLOCKED",
        "INSUFFICIENT_ROOT_CAPACITY",
        f"free={free_bytes} required={minimum_free}",
    )
    for name, expected in EXPECTED_ENV.items():
        _require(
            os.environ.get(name) == expected,
            "ENV_BLOCKED",
            "LAUNCH_ENV_MISMATCH",
            f"{name}: expected={expected!r} observed={os.environ.get(name)!r}",
        )
    _require(
        os.environ.get("CUDA_CACHE_DISABLE") in (None, ""),
        "ENV_BLOCKED",
        "CUDA_CACHE_DISABLE_FORBIDDEN",
        repr(os.environ.get("CUDA_CACHE_DISABLE")),
    )
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        _require(
            os.environ.get(name) in (None, "", "1"),
            "ENV_BLOCKED",
            "MULTIPROCESS_ENV_FORBIDDEN",
            f"{name}={os.environ.get(name)!r}",
        )
    _require(
        "torch" not in sys.modules
        and "triton" not in sys.modules
        and not any(name.startswith(("diffusion.", "sana_wam.")) for name in sys.modules),
        "ENV_BLOCKED",
        "TORCH_OR_VENDOR_IMPORTED_BEFORE_PREFLIGHT",
        "runtime module already present",
    )
    observed_heads = {
        "sana_wam": _git_head(REPO_ROOT),
        "third_party_sana": _git_head(REPO_ROOT / "third_party/Sana"),
        "sana_afcc_handoff": _git_head(REPO_ROOT.parent / "sana-afcc-handoff"),
    }
    _require(
        observed_heads == EXPECTED_REPO_HEADS,
        "ENV_BLOCKED",
        "REPOSITORY_HEAD_MISMATCH",
        repr(observed_heads),
    )
    _require(
        not EXPECTED_REVIEW_CLAIM.exists() and not EXPECTED_REVIEW_CLAIM.is_symlink(),
        "AUTH_BLOCKED",
        "REVIEW_TOKEN_ALREADY_CLAIMED",
        str(EXPECTED_REVIEW_CLAIM),
    )


def _ensure_root_namespace() -> None:
    """Create only the two card-derived namespace parents, without a root."""

    current = EXPECTED_ROOT_NAMESPACE_BASE
    for component in EXPECTED_ROOT_NAMESPACE_PARTS:
        current = current / component
        try:
            os.mkdir(current, 0o775)
        except FileExistsError:
            pass
        info = current.lstat()
        _require(
            stat.S_ISDIR(info.st_mode)
            and not current.is_symlink()
            and current.resolve(strict=True) == current
            and info.st_uid == os.geteuid()
            and os.access(current, os.W_OK | os.X_OK),
            "ENV_BLOCKED",
            "ROOT_NAMESPACE_COMPONENT_UNSAFE_AFTER_CREATE",
            str(current),
        )
    _require(
        current == EXPECTED_ROOT.parent,
        "AUTH_BLOCKED",
        "ROOT_NAMESPACE_DERIVATION_MISMATCH",
        str(current),
    )


def _source_pins(card: Mapping[str, Any]) -> dict[str, str]:
    pins = {relative: _sha256_file(_regular_repo_file(relative)) for relative in AV1B_CODE_PATHS}
    pins[EXPECTED_CARD] = EXPECTED_CARD_SHA256
    for item in card["source_pins"].values():
        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
            pins[item["path"]] = _sha256_file(_regular_repo_file(item["path"]))
    return dict(sorted(pins.items()))


def _run_command(command: Sequence[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError("ENV_BLOCKED", "GPU_QUERY_FAILED", str(exc)) from exc


def _gpu_snapshot(label: str) -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    _require(
        nvidia_smi is not None,
        "ENV_BLOCKED",
        "NVIDIA_SMI_MISSING",
        "nvidia-smi not found",
    )
    query = _run_command(
        [
            nvidia_smi,
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    _require(
        query.returncode == 0,
        "ENV_BLOCKED",
        "GPU_QUERY_FAILED",
        query.stderr.strip(),
    )
    rows: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        _require(
            len(fields) == 5,
            "ENV_BLOCKED",
            "GPU_QUERY_PARSE_FAILED",
            line,
        )
        try:
            rows.append(
                {
                    "physical_index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_used_mib": int(fields[3]),
                    "utilization_percent": int(fields[4]),
                }
            )
        except ValueError as exc:
            raise PreflightError("ENV_BLOCKED", "GPU_QUERY_PARSE_FAILED", line) from exc
    matches = [row for row in rows if row["uuid"] == EXPECTED_GPU_UUID]
    _require(
        len(matches) == 1,
        "ENV_BLOCKED",
        "GPU_UUID_MAPPING_MISSING_OR_DUPLICATE",
        repr(matches),
    )
    selected = matches[0]
    _require(
        selected["physical_index"] == EXPECTED_GPU_PHYSICAL_INDEX
        and selected["name"] == EXPECTED_GPU_NAME,
        "ENV_BLOCKED",
        "GPU_IDENTITY_MISMATCH",
        repr(selected),
    )
    _require(
        selected["memory_used_mib"] == 0 and selected["utilization_percent"] == 0,
        "ENV_BLOCKED",
        "GPU_NOT_IDLE",
        repr(selected),
    )

    apps = _run_command(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    _require(
        apps.returncode == 0,
        "ENV_BLOCKED",
        "GPU_COMPUTE_APP_QUERY_FAILED",
        apps.stderr.strip(),
    )
    selected_apps = [
        line.strip()
        for line in apps.stdout.splitlines()
        if line.strip() and line.split(",", 1)[0].strip() == EXPECTED_GPU_UUID
    ]
    _require(
        not selected_apps,
        "ENV_BLOCKED",
        "GPU_EXTERNAL_COMPUTE_APP_PRESENT",
        repr(selected_apps),
    )
    pmon = _run_command(
        [nvidia_smi, "pmon", "-i", str(EXPECTED_GPU_PHYSICAL_INDEX), "-c", "1"]
    )
    _require(
        pmon.returncode == 0,
        "ENV_BLOCKED",
        "GPU_PMON_FAILED",
        pmon.stderr.strip(),
    )
    pmon_rows: list[str] = []
    for line in pmon.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        _require(
            len(fields) >= 2 and fields[0] == str(EXPECTED_GPU_PHYSICAL_INDEX),
            "ENV_BLOCKED",
            "GPU_PMON_PARSE_FAILED",
            stripped,
        )
        # NVIDIA pmon emits one all-dash placeholder row for an idle GPU.
        # Only a concrete PID denotes an external process.
        if fields[1] != "-":
            pmon_rows.append(stripped)
    _require(
        not pmon_rows,
        "ENV_BLOCKED",
        "GPU_PMON_PROCESS_PRESENT",
        repr(pmon_rows),
    )
    fuser = shutil.which("fuser")
    _require(fuser is not None, "ENV_BLOCKED", "FUSER_MISSING", "fuser not found")
    fuser_result = _run_command([fuser, f"/dev/nvidia{EXPECTED_GPU_PHYSICAL_INDEX}"])
    fuser_tokens = [token for token in fuser_result.stdout.split() if token.isdigit()]
    fuser_tokens.extend(token for token in fuser_result.stderr.split() if token.isdigit())
    _require(
        fuser_result.returncode in (0, 1),
        "ENV_BLOCKED",
        "GPU_FUSER_FAILED",
        fuser_result.stderr.strip(),
    )
    _require(
        not fuser_tokens,
        "ENV_BLOCKED",
        "GPU_EXTERNAL_FUSER_PRESENT",
        repr(fuser_tokens),
    )
    return {
        "label": label,
        "observed_unix_ns": time.time_ns(),
        "selected_gpu": selected,
        "compute_apps": selected_apps,
        "pmon_rows": pmon_rows,
        "fuser_pids": fuser_tokens,
        "nvidia_smi_sha256": _sha256_file(Path(nvidia_smi)),
    }


def _rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _root_size(root: Path) -> int:
    total = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        for name in directory_names:
            child = Path(directory) / name
            if child.is_symlink():
                raise RuntimeError(f"symlink forbidden in run root: {child}")
        for name in file_names:
            child = Path(directory) / name
            info = child.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"non-regular artifact forbidden: {child}")
            total += info.st_size
    return total


def _recursive_inventory(root: Path, exclude: frozenset[str]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort(key=lambda value: value.encode("utf-8"))
        file_names.sort(key=lambda value: value.encode("utf-8"))
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            relative = child.relative_to(root).as_posix()
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise RuntimeError(f"symlink forbidden in run root: {relative}")
            if not stat.S_ISDIR(info.st_mode):
                raise RuntimeError(f"non-directory entry forbidden: {relative}")
            inventory.append({"path": relative, "kind": "directory"})
        for name in file_names:
            child = directory_path / name
            relative = child.relative_to(root).as_posix()
            if relative in exclude:
                continue
            info = child.lstat()
            if not stat.S_ISREG(info.st_mode) or child.is_symlink():
                raise RuntimeError(f"non-regular artifact forbidden: {relative}")
            forbidden_suffixes = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")
            if not relative.startswith("triton_cache/") and child.suffix.lower() in forbidden_suffixes:
                raise RuntimeError(f"checkpoint-like artifact forbidden: {relative}")
            inventory.append(
                {
                    "path": relative,
                    "kind": "regular",
                    "sha256": _sha256_file(child),
                    "size_bytes": info.st_size,
                }
            )
    return inventory


class RootLifecycle:
    def __init__(self, root: Path, context: Mapping[str, Any]):
        self.root = root
        self.context = dict(context)
        self.created = False
        self.finalized = False
        self._finalizing = False

    def create(self) -> None:
        trapped = {
            getattr(signal, name)
            for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT", "SIGALRM")
            if hasattr(signal, name)
        }
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, trapped)
        try:
            os.mkdir(self.root, 0o700)
            self.created = True
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        os.mkdir(self.root / "triton_cache", 0o700)

    def publish_context(self) -> str:
        return self.publish_or_match_json("RUN_CONTEXT.json", self.context)

    def publish_bytes(self, name: str, payload: bytes, mode: int = 0o600) -> str:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"unsafe artifact name: {name!r}")
        destination = self.root / name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(destination, flags, mode)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        return _sha256_bytes(payload)

    def publish_json(self, name: str, value: Any) -> str:
        return self.publish_bytes(name, _canonical_json_bytes(value))

    def publish_or_match_json(self, name: str, value: Any) -> str:
        payload = _canonical_json_bytes(value)
        destination = self.root / name
        if destination.exists():
            observed = destination.read_bytes()
            if observed != payload:
                raise RuntimeError(f"artifact changed after publication: {name}")
            return _sha256_bytes(observed)
        return self.publish_bytes(name, payload)

    def _ensure_required_failure_artifacts(self, state: str, reasons: Sequence[str]) -> None:
        placeholder = {
            "schema": "cach.av1b.vendor_gdn_bridge.unavailable_artifact.v1",
            "exact_typed_state": state,
            "validity_reason": list(reasons),
            "available": False,
        }
        for name in REQUIRED_ROOT_ARTIFACTS:
            if name in {"RAW_EVIDENCE.json", "RESULT.json", "FREEZE_RECEIPT.json"}:
                continue
            destination = self.root / name
            if not destination.exists():
                if name == "RUN_CONTEXT.json":
                    self.publish_json(name, {**self.context, "postcreate_failure": True})
                else:
                    self.publish_json(name, {**placeholder, "artifact": name})

    def _freeze(self) -> None:
        for directory, directory_names, file_names in os.walk(
            self.root, topdown=False, followlinks=False
        ):
            directory_path = Path(directory)
            for name in file_names:
                child = directory_path / name
                if not child.is_symlink() and child.is_file():
                    os.chmod(child, 0o444, follow_symlinks=False)
            for name in directory_names:
                child = directory_path / name
                if not child.is_symlink() and child.is_dir():
                    os.chmod(child, 0o555)
        os.chmod(self.root, 0o555)

    def finalize(
        self,
        *,
        state: str,
        validity_reasons: list[str],
        raw_evidence: Mapping[str, Any],
        source_pins: Mapping[str, str],
        started_monotonic: float,
    ) -> dict[str, Any]:
        if self.finalized:
            return {"exact_typed_state": state, "already_finalized": True}
        if self._finalizing:
            raise RuntimeError("recursive root finalization")
        self._finalizing = True
        if state not in ALLOWED_TERMINAL_STATES:
            validity_reasons = ["UNLISTED_TERMINAL_STATE", *validity_reasons]
            state = "IMPLEMENTATION_INVALID"
        try:
            self._ensure_required_failure_artifacts(state, validity_reasons)
            raw_sha = self.publish_or_match_json("RAW_EVIDENCE.json", raw_evidence)
            result = {
                "schema": "cach.av1b.vendor_gdn_bridge.result.v1",
                "exact_typed_state": state,
                "operator_level": "VENDOR_KERNEL",
                "integration_path": "EXPERIMENTAL_PATH",
                "single_call_semantic_scope": (
                    "FIVE_FRAMES_CHUNK_INDEX_0_3_5_EPHEMERAL_VENDOR_STATE_ONLY"
                ),
                "result_scope": (
                    "SYNTHETIC_SINGLE_BATCH_SINGLE_CALL_VENDOR_OPERATOR_ONLY"
                ),
                "explicitly_not_assessed": [
                    "CROSS_CALL_INIT_STATE_OR_FINAL_STATE_IO",
                    "TWO_CALL_DIFFERENTIABLE_SCRATCH_STATE_CONTINUATION",
                    "LIVE_CACHE_COMMIT_OR_CONTENT_TIME",
                    "PUBLIC_WRAPPER_DISPATCHER_OR_OWNER_PARITY",
                    "REAL_DATA_OR_CLOSED_LOOP_CAPABILITY",
                    "FULL_2B",
                ],
                "validity_reason": validity_reasons,
                "raw_evidence_path": "RAW_EVIDENCE.json",
                "raw_evidence_sha256": raw_sha,
                "run_card_self_path_and_full_sha256": {
                    "path": EXPECTED_CARD,
                    "sha256": EXPECTED_CARD_SHA256,
                },
                "review_token_id_and_consumption_state": {
                    "token_id": EXPECTED_REVIEW_TOKEN,
                    "state": "UNCONSUMED",
                    "claim_path_written": False,
                },
                "code_config_runner_test_sha256": dict(source_pins),
                "vendor_source_sha256": {
                    key: value
                    for key, value in source_pins.items()
                    if key.startswith("third_party/Sana/")
                },
                "authority_chain": {
                    "card_freeze_statement_sha256": EXPECTED_CARD_FREEZE_AUTHORITY_SHA256,
                    "execution_statement_sha256": EXPECTED_EXECUTION_AUTHORITY_SHA256,
                    "confirmation_statement_sha256": EXPECTED_CLARIFICATION_AUTHORITY_SHA256,
                    "statement_sha256_rule": "SHA256_UTF8_WITHOUT_TRAILING_NEWLINE",
                    "clarification_controls_conflict": True,
                },
                "gpu_reservation": self.context.get("gpu_reservation"),
                "resolved_root": str(self.root),
                "run_nonce": EXPECTED_NONCE,
                "root_lifecycle_and_freeze_receipt": {
                    "exclusive_create": True,
                    "terminal_root_frozen": True,
                    "freeze_receipt_path": "FREEZE_RECEIPT.json",
                    "freeze_receipt_pin_location": "launcher_stdout_after_publish",
                },
                "formal_admission_evidence": False,
                "global_stage3_effect": "NONE",
            }
            result_sha = self.publish_or_match_json("RESULT.json", result)
            elapsed = time.monotonic() - started_monotonic
            inventory = _recursive_inventory(
                self.root, frozenset({"FREEZE_RECEIPT.json"})
            )
            receipt = {
                "schema": "cach.av1b.vendor_gdn_bridge.freeze_receipt.v1",
                "resolved_root": str(self.root),
                "run_nonce": EXPECTED_NONCE,
                "terminal_state": state,
                "result_path": "RESULT.json",
                "result_sha256": result_sha,
                "inventory_excludes": ["FREEZE_RECEIPT.json"],
                "inventory": inventory,
                "required_artifacts_present": all(
                    (self.root / name).is_file() for name in REQUIRED_ROOT_ARTIFACTS[:-1]
                ),
                "root_size_bytes_before_receipt": _root_size(self.root),
                "elapsed_seconds_at_freeze": elapsed,
                "process_peak_rss_bytes": _rss_bytes(),
                "file_mode_after_freeze": "0444",
                "directory_mode_after_freeze": "0555",
                "immutable": True,
            }
            receipt_sha = self.publish_json("FREEZE_RECEIPT.json", receipt)
            self._freeze()
            self.finalized = True
            return {
                "exact_typed_state": state,
                "resolved_root": str(self.root),
                "result_sha256": result_sha,
                "freeze_receipt_sha256": receipt_sha,
                "elapsed_seconds": elapsed,
                "process_peak_rss_bytes": _rss_bytes(),
                "root_size_bytes": _root_size(self.root),
                "review_token_state": "UNCONSUMED",
            }
        finally:
            self._finalizing = False

    def emergency_finalize(self) -> None:
        if not self.created or self.finalized or self._finalizing:
            return
        try:
            reasons = ["PROCESS_EXITED_BEFORE_NORMAL_FINALIZATION"]
            raw_evidence = {
                "schema": "cach.av1b.vendor_gdn_bridge.emergency_failure.v1",
                "proposed_typed_state": "IMPLEMENTATION_INVALID",
                "validity_reason": reasons,
            }
            self.finalize(
                state="IMPLEMENTATION_INVALID",
                validity_reasons=reasons,
                raw_evidence=raw_evidence,
                source_pins=self.context.get("source_pins", {}),
                started_monotonic=float(
                    self.context.get("started_monotonic", time.monotonic())
                ),
            )
        except BaseException:
            try:
                self._ensure_required_failure_artifacts(
                    "IMPLEMENTATION_INVALID", ["EMERGENCY_FINALIZATION_FAILED"]
                )
                if not (self.root / "FREEZE_RECEIPT.json").exists():
                    self.publish_json(
                        "FREEZE_RECEIPT.json",
                        {
                            "schema": (
                                "cach.av1b.vendor_gdn_bridge.emergency_freeze_receipt.v1"
                            ),
                            "resolved_root": str(self.root),
                            "terminal_state": "IMPLEMENTATION_INVALID",
                            "inventory": _recursive_inventory(
                                self.root, frozenset({"FREEZE_RECEIPT.json"})
                            ),
                            "immutable": True,
                        },
                    )
                self._freeze()
                self.finalized = True
            except BaseException:
                try:
                    # Even a malformed compiler-cache entry must not leave the
                    # ordinary files/directories mutable after a terminal exit.
                    self._freeze()
                    self.finalized = True
                except BaseException:
                    try:
                        os.chmod(self.root, 0o555)
                    except BaseException:
                        pass


def _install_failure_traps(lifecycle: RootLifecycle) -> None:
    def handler(signum: int, _frame: Any) -> NoReturn:
        raise RunInterrupted(f"received signal {signum}")

    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT", "SIGALRM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)
    atexit.register(lifecycle.emergency_finalize)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _assert_no_nonfinite(value: Any, path: str = "screen") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite value at {path}: {value!r}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_nonfinite(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_nonfinite(child, f"{path}[{index}]")


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not _finite_number(value):
        raise ValueError(f"missing or non-finite metric: {name}")
    return float(value)


def _horizon_metric(metrics: Mapping[str, Any], name: str, horizon: int) -> float:
    value = metrics.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing horizon metric mapping: {name}")
    selected = value.get(str(horizon), value.get(horizon))
    if not _finite_number(selected):
        raise ValueError(f"missing or non-finite metric: {name}[{horizon}]")
    return float(selected)


def _screen_evidence_errors(screen: Mapping[str, Any], card: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required_keys = {
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
    if set(screen) != required_keys or screen.get("schema") != (
        "cach.av1b.vendor_gdn_bridge.screen.v1"
    ):
        return ["SCREEN_SCHEMA_MISMATCH"]
    steps = int(card["experiment_contract"]["optimizer"]["optimizer_steps_per_arm"])
    for name in ("theta0_manifests", "final_parameter_digests"):
        value = screen.get(name)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"reference", "candidate"}
            or any(not isinstance(item, Mapping) or not item for item in value.values())
        ):
            errors.append(f"{name.upper()}_INVALID")
    trace = screen.get("loss_trace")
    if not isinstance(trace, Mapping) or set(trace) != {"reference", "candidate"}:
        errors.append("LOSS_TRACE_INVALID")
    else:
        for arm in ("reference", "candidate"):
            values = trace[arm]
            if (
                not isinstance(values, list)
                or len(values) != steps + 1
                or any(not _finite_number(item) for item in values)
            ):
                errors.append(f"LOSS_TRACE_{arm.upper()}_INVALID")

    diagnostics = screen.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return [*errors, "DIAGNOSTICS_INVALID"]
    exact = {
        "operator_level": "VENDOR_KERNEL",
        "integration_path": "EXPERIMENTAL_PATH",
        "result_scope": "SYNTHETIC_SINGLE_BATCH_SINGLE_CALL_VENDOR_OPERATOR_ONLY",
        "device": "cuda:0",
        "dtype": "float32",
        "vendor_entry_class": "ChunkCausalGDNTriton",
        "vendor_use_autograd_kernel": True,
        "hw": [5, 1, 1],
        "chunk_boundaries": [0, 3, 5],
        "fused_gdn_precision": "0",
        "cross_call_state_or_commit_assessed": False,
        "no_action_frozen_exact_zero": True,
    }
    for name, expected in exact.items():
        if diagnostics.get(name) != expected:
            errors.append(f"DIAGNOSTIC_{name.upper()}_INVALID")
    completed = diagnostics.get("optimizer_steps_completed_by_arm")
    if completed != {"REF-GDN-CORRECTED": steps, "CACH-A": steps}:
        errors.append("EXACT_OPTIMIZER_STEPS_NOT_COMPLETED")
    vendor_calls = diagnostics.get("actual_vendor_forward_call_count")
    if not isinstance(vendor_calls, int) or isinstance(vendor_calls, bool) or vendor_calls <= 0:
        errors.append("VENDOR_FORWARD_NOT_OBSERVED")
    finite_positive = (
        "vendor_common_parameter_gradient_rms_step0",
        "last_block_local_seam_parameter_jvp_rms_theta0",
        "last_block_local_action_condition_jvp_rms_step200",
    )
    for name in finite_positive:
        value = diagnostics.get(name)
        if not _finite_number(value) or float(value) <= 1.0e-8:
            errors.append(f"DIAGNOSTIC_{name.upper()}_NOT_LIVE")
    prefix_abs = diagnostics.get("single_call_first_chunk_prefix_invariance_max_abs")
    prefix_rel = diagnostics.get("single_call_first_chunk_prefix_invariance_max_rel")
    if not _finite_number(prefix_abs) or float(prefix_abs) > 1.0e-4:
        errors.append("PREFIX_INVARIANCE_ABS_FAILED")
    if not _finite_number(prefix_rel) or float(prefix_rel) > 1.0e-4:
        errors.append("PREFIX_INVARIANCE_REL_FAILED")
    target = diagnostics.get("target_leakage")
    if (
        not isinstance(target, Mapping)
        or target.get("target_perturbation_changes_prediction_or_post_vendor_hidden")
        is not False
        or target.get("prediction_target_reverse_gradient_nonzero") is not False
    ):
        errors.append("TARGET_LEAKAGE_INVALID")
    identity = diagnostics.get("theta0_identity")
    required_identity = {
        "reference_candidate_video_output_bitwise_equal_all_modes": True,
        "reference_correct_shuffle_no_action_video_outputs_bitwise_equal": True,
        "no_action_candidate_pair_predictions_bitwise_equal": True,
        "common_theta0_parameter_bytes_equal": True,
        "valid_action_exact_once_and_padding_zero": True,
        "non_seam_bytes_bitwise_equal_across_modes": True,
    }
    if not isinstance(identity, Mapping) or any(
        identity.get(name) is not expected for name, expected in required_identity.items()
    ):
        errors.append("THETA0_OR_SEAM_IDENTITY_INVALID")
    observability = diagnostics.get("post_vendor_observability")
    if (
        not isinstance(observability, Mapping)
        or observability.get("finite_output") is not True
        or observability.get("finite_post_vendor_hidden_gradient") is not True
        or observability.get("finite_post_vendor_hidden_update") is not True
    ):
        errors.append("POST_VENDOR_OBSERVABILITY_INVALID")
    runtime = screen.get("resource_usage")
    forbidden = {
        "checkpoint_loaded": False,
        "checkpoint_saved": False,
        "real_data_used": False,
        "cpu_proxy_or_reference_fallback_used": False,
        "review_token_consumed": False,
    }
    if not isinstance(runtime, Mapping) or any(
        runtime.get(name) is not expected for name, expected in forbidden.items()
    ):
        errors.append("RUNTIME_SCOPE_EVIDENCE_INVALID")
    return errors


def _classify_screen(
    screen: Mapping[str, Any], card: Mapping[str, Any]
) -> tuple[str, list[str]]:
    errors = _screen_evidence_errors(screen, card)
    if errors:
        return "IMPLEMENTATION_INVALID", errors
    validity = screen.get("validity")
    metrics = screen.get("metrics")
    if not isinstance(validity, Mapping) or not isinstance(metrics, Mapping):
        return "IMPLEMENTATION_INVALID", ["MISSING_VALIDITY_OR_METRICS"]
    supplied = validity.get("reasons", [])
    reasons = [str(item) for item in supplied] if isinstance(supplied, list) else []
    if validity.get("implementation_valid") is not True:
        return "IMPLEMENTATION_INVALID", reasons or ["IMPLEMENTATION_VALIDITY_FAILED"]
    if validity.get("data_valid") is not True:
        return "INVALID_RUN", reasons or ["DATA_VALIDITY_FAILED"]
    if validity.get("numerics_valid") is not True or validity.get("finite_all") is not True:
        return "NUMERICAL_INVALID", reasons or ["NUMERICAL_VALIDITY_FAILED"]
    if validity.get("all_card_validity_checks_pass") is not True:
        return "IMPLEMENTATION_INVALID", reasons or ["CARD_VALIDITY_CHECK_FAILED"]

    try:
        values = {
            "seam_grad": _metric(metrics, "seam_grad_rms_step0"),
            "seam_update": _metric(metrics, "seam_update_rms_final"),
            "parameter_jvp": _metric(
                metrics, "last_block_local_seam_parameter_jvp_rms_theta0"
            ),
            "action_jvp": _metric(
                metrics, "last_block_local_action_condition_jvp_rms_step200"
            ),
            "loss_drop": _metric(metrics, "loss_drop_rel"),
            "dnmse": _metric(metrics, "counterfactual_delta_nmse_correct"),
            "shuffle": _metric(metrics, "shuffle_gap_rel"),
            "no_action": _metric(metrics, "no_action_gap_rel"),
            "gain": _metric(metrics, "candidate_gain_vs_reference"),
        }
        dnmse_h = {
            h: _horizon_metric(metrics, "counterfactual_delta_nmse_correct_by_horizon", h)
            for h in (1, 2, 4)
        }
        shuffle_h = {
            h: _horizon_metric(metrics, "shuffle_gap_rel_by_horizon", h)
            for h in (1, 2, 4)
        }
        no_action_h = {
            h: _horizon_metric(metrics, "no_action_gap_rel_by_horizon", h)
            for h in (1, 2, 4)
        }
    except ValueError as exc:
        return "IMPLEMENTATION_INVALID", ["METRIC_CONTRACT_FAILED", str(exc)]

    go = card["threshold_contract"]["operator_go_all_required"]
    go_checks = {
        "seam_grad": values["seam_grad"] > go["seam_grad_rms_step0_strictly_greater_than"],
        "seam_update": values["seam_update"] > go["seam_update_rms_final_strictly_greater_than"],
        "parameter_jvp": values["parameter_jvp"] > go["last_block_local_seam_parameter_jvp_rms_theta0_strictly_greater_than"],
        "action_jvp": values["action_jvp"] > go["last_block_local_action_condition_jvp_rms_step200_strictly_greater_than"],
        "loss_drop": values["loss_drop"] >= go["loss_drop_rel_min"],
        "dnmse": values["dnmse"] <= go["counterfactual_delta_nmse_correct_max"],
        "shuffle": values["shuffle"] >= go["shuffle_gap_rel_min"],
        "no_action": values["no_action"] >= go["no_action_gap_rel_min"],
        "gain": values["gain"] >= go["candidate_gain_vs_reference_min"],
    }
    for horizon in (1, 2, 4):
        go_checks[f"dnmse_h{horizon}"] = dnmse_h[horizon] <= go[
            f"counterfactual_delta_nmse_correct_h{horizon}_max"
        ]
        go_checks[f"shuffle_h{horizon}"] = shuffle_h[horizon] >= go[
            f"shuffle_gap_rel_h{horizon}_min"
        ]
        go_checks[f"no_action_h{horizon}"] = no_action_h[horizon] >= go[
            f"no_action_gap_rel_h{horizon}_min"
        ]
    if all(go_checks.values()):
        return "OPERATOR_GO", ["ALL_OPERATOR_GO_THRESHOLDS_MET"]

    stop = card["threshold_contract"]["operator_strong_stop_all_required"]
    stop_checks = {
        "seam_grad": values["seam_grad"] > stop["seam_grad_rms_step0_strictly_greater_than"],
        "seam_update": values["seam_update"] > stop["seam_update_rms_final_strictly_greater_than"],
        "loss_drop": values["loss_drop"] <= stop["loss_drop_rel_max"],
        "dnmse": values["dnmse"] >= stop["counterfactual_delta_nmse_correct_min"],
        "shuffle": values["shuffle"] <= stop["shuffle_gap_rel_max"],
        "no_action": values["no_action"] <= stop["no_action_gap_rel_max"],
        "gain": values["gain"] <= stop["candidate_gain_vs_reference_max"],
    }
    failed_go = sorted(name for name, passed in go_checks.items() if not passed)
    failed_stop = sorted(name for name, passed in stop_checks.items() if not passed)
    return "REVIEW_ONCE", [
        "VALID_FIRST_RUN_NON_GO_REVIEW_TOKEN_REMAINS_UNCONSUMED",
        "GO_FAILED:" + ",".join(failed_go),
        "STRONG_STOP_MATCH:" + str(all(stop_checks.values())).lower(),
        "STRONG_STOP_FAILED:" + ",".join(failed_stop),
    ]


def _screen_artifacts(screen: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "RAW_METRICS.json": screen.get("metrics", {}),
        "LOSS_TRACE.json": screen.get("loss_trace", {}),
        "THETA0_MANIFESTS.json": screen.get("theta0_manifests", {}),
        "FINAL_PARAMETER_DIGESTS.json": screen.get("final_parameter_digests", {}),
        "JIT_INVENTORY.json": screen.get("jit_inventory", {}),
    }


def _postcreate_failure_state(exc: BaseException) -> tuple[str, list[str]]:
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, (TimeoutError, RunInterrupted, MemoryError)):
        return "HARNESS_REJECTED", [type(exc).__name__, message]
    if (
        "out of memory" in lowered
        or "triton" in lowered
        or "compile" in lowered
        or "immediate gpu" in lowered
        or "reservation" in lowered
    ):
        return "HARNESS_REJECTED", [type(exc).__name__, message]
    if isinstance(exc, OSError) and exc.errno in {
        getattr(os, "ENOSPC", 28),
        getattr(os, "EFBIG", 27),
        getattr(os, "EDQUOT", 122),
    }:
        return "HARNESS_REJECTED", [type(exc).__name__, message]
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return "ENV_BLOCKED", ["RUNTIME_DEPENDENCY_OR_KERNEL_MISSING", message]
    if isinstance(exc, (FloatingPointError, OverflowError)):
        return "NUMERICAL_INVALID", [type(exc).__name__, message]
    return "IMPLEMENTATION_INVALID", [type(exc).__name__, message]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--card", required=True)
    parser.add_argument("--card-sha256", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--nonce", required=True)
    return parser.parse_args(argv)


def _precreate_block(exc: PreflightError) -> int:
    event = {
        "schema": "cach.av1b.vendor_gdn_bridge.precreate_block.v1",
        "exact_typed_state": exc.state,
        "validity_reason": [exc.token, exc.detail],
        "resolved_root": None,
        "root_created": False,
        "gpu_used": False,
        "review_token_state": "UNCONSUMED",
    }
    sys.stderr.buffer.write(_canonical_json_bytes(event))
    return 2


def main(argv: list[str] | None = None) -> int:
    started_monotonic = time.monotonic()
    arguments = _parse_args(argv)
    config_path = Path(arguments.config)
    card_path = Path(arguments.card)
    root_argument = Path(arguments.root)
    lifecycle: RootLifecycle | None = None
    config: Mapping[str, Any] | None = None
    card: Mapping[str, Any] | None = None
    pins: dict[str, str] = {}
    snapshots: list[dict[str, Any]] = []
    try:
        _require(
            arguments.card_sha256 == EXPECTED_CARD_SHA256,
            "AUTH_BLOCKED",
            "CALLER_CARD_PIN_MISMATCH",
            arguments.card_sha256,
        )
        card = _validate_card(card_path)
        try:
            config = _duplicate_rejecting_yaml(config_path)
        except PreflightError:
            raise
        except (OSError, ValueError) as exc:
            raise PreflightError("AUTH_BLOCKED", "CONFIG_PARSE_REJECTED", str(exc)) from exc
        _validate_config(config, config_path, card)
        _validate_environment(root_argument, arguments.nonce, card)
        pins = _source_pins(card)
        context = {
            "schema": "cach.av1b.vendor_gdn_bridge.run_context.v1",
            "canonical_host_alias": "H200",
            "observed_hostname": platform.node(),
            "canonical_worktree": str(REPO_ROOT),
            "resolved_root": str(root_argument),
            "run_nonce": EXPECTED_NONCE,
            "pid": os.getpid(),
            "device": "cuda:0_AFTER_EXACT_UUID_FILTER",
            "dtype": "float32",
            "source_pins": pins,
            "card_path": EXPECTED_CARD,
            "card_sha256": EXPECTED_CARD_SHA256,
            "authority_chain_sha256": [
                EXPECTED_CARD_FREEZE_AUTHORITY_SHA256,
                EXPECTED_EXECUTION_AUTHORITY_SHA256,
                EXPECTED_CLARIFICATION_AUTHORITY_SHA256,
            ],
            "gpu_reservation": {
                "physical_index": EXPECTED_GPU_PHYSICAL_INDEX,
                "uuid": EXPECTED_GPU_UUID,
                "logical_index": 0,
                "snapshots": snapshots,
                "reservation_state": "PENDING_TWO_PREFLIGHT_AND_ONE_IMMEDIATE_RECHECK",
                "no_fallback_gpu": True,
            },
            "launch_environment": dict(EXPECTED_ENV),
            "review_token_state": "UNCONSUMED",
            "formal_admission_evidence": False,
            "started_monotonic": started_monotonic,
        }
        lifecycle = RootLifecycle(root_argument, context)
        _install_failure_traps(lifecycle)
        snapshots.append(_gpu_snapshot("PRECREATE_RESERVATION_CHECK"))
        time.sleep(120.0)
        snapshots.append(_gpu_snapshot("PRECREATE_RESERVATION_RECHECK_AFTER_120S"))
        precreate_observation_seconds = (
            snapshots[-1]["observed_unix_ns"] - snapshots[0]["observed_unix_ns"]
        ) / 1_000_000_000.0
        lifecycle.context["gpu_reservation"] = {
            **lifecycle.context["gpu_reservation"],
            "snapshots": list(snapshots),
            "precreate_observation_seconds": precreate_observation_seconds,
            "reservation_state": "TWO_PREFLIGHT_SNAPSHOTS_CLEAR",
        }
        try:
            snapshots.append(_gpu_snapshot("IMMEDIATE_PRECREATE_PREIMPORT_RECHECK"))
        except PreflightError as exc:
            raise RuntimeError(f"immediate GPU reservation recheck failed: {exc}") from exc
        lifecycle.context["gpu_reservation"] = {
            **lifecycle.context["gpu_reservation"],
            "snapshots": list(snapshots),
            "reservation_state": "EXACT_UUID_TRIPLE_SNAPSHOT_CLEAR_SINGLE_PROCESS_LAUNCH",
            "immediate_recheck_before_root_create_and_import": True,
            "mechanical_interprocess_lock": False,
        }
        _ensure_root_namespace()
        lifecycle.create()
        lifecycle.publish_context()
    except PreflightError as exc:
        return _precreate_block(exc)
    except Exception as exc:
        if lifecycle is None or not lifecycle.created:
            return _precreate_block(
                PreflightError("ENV_BLOCKED", type(exc).__name__, str(exc))
            )
        state, reasons = _postcreate_failure_state(exc)
        raw_evidence = {
            "schema": "cach.av1b.vendor_gdn_bridge.raw_evidence.v1",
            "proposed_typed_state": state,
            "validity_reason": reasons,
            "screen": {
                "schema": "cach.av1b.vendor_gdn_bridge.failed_create.v1",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "resource_usage": {
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "process_peak_rss_bytes": _rss_bytes(),
                "root_size_bytes_before_terminal_artifacts": _root_size(lifecycle.root),
            },
            "forbidden_artifacts_written": False,
            "model_or_optimizer_state_serialized": False,
            "review_token_state": "UNCONSUMED",
        }
        try:
            terminal = lifecycle.finalize(
                state=state,
                validity_reasons=reasons,
                raw_evidence=raw_evidence,
                source_pins=pins,
                started_monotonic=started_monotonic,
            )
            sys.stdout.buffer.write(_canonical_json_bytes(terminal))
        except BaseException:
            lifecycle.emergency_finalize()
        return 2

    assert lifecycle is not None and lifecycle.created and config is not None and card is not None
    screen: Mapping[str, Any] = {}
    state = "IMPLEMENTATION_INVALID"
    reasons: list[str] = ["SCREEN_DID_NOT_START"]
    try:
        remaining = float(card["experiment_contract"]["budget"]["max_wall_seconds_total"]) - (
            time.monotonic() - started_monotonic
        )
        if remaining <= 0:
            raise TimeoutError("wall budget exhausted before vendor import")
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, remaining)

        import torch
        import triton

        if torch.__version__ != "2.7.1+cu128":
            raise RuntimeError(f"torch version mismatch: {torch.__version__}")
        if triton.__version__ != "3.5.1":
            raise RuntimeError(f"triton version mismatch: {triton.__version__}")
        if torch.version.cuda != "12.8":
            raise RuntimeError(f"torch CUDA mismatch: {torch.version.cuda}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one visible CUDA device is required")
        properties = torch.cuda.get_device_properties(0)
        if properties.name != EXPECTED_GPU_NAME or (properties.major, properties.minor) != (9, 0):
            raise RuntimeError(
                f"logical cuda:0 identity mismatch: {properties.name} "
                f"{properties.major}.{properties.minor}"
            )
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)

        sys.path.insert(0, str(REPO_ROOT / "third_party/Sana"))
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from sana_wam.model.cach_av1b_vendor_gdn_bridge import run_av1b_screen

        progress_events: list[dict[str, Any]] = []
        theta0_published = False

        def progress_callback(event: Mapping[str, Any]) -> None:
            nonlocal theta0_published
            if not isinstance(event, Mapping):
                raise TypeError("progress event must be a mapping")
            materialized = dict(event)
            _canonical_json_bytes(materialized)
            if len(progress_events) >= 1024:
                raise RuntimeError("progress event limit exceeded")
            progress_events.append(materialized)
            if materialized.get("event") == "theta0_manifests_ready":
                manifests = materialized.get(
                    "payload", materialized.get("theta0_manifests")
                )
                if not isinstance(manifests, Mapping):
                    raise TypeError("theta0 progress event lacks manifests")
                lifecycle.publish_or_match_json(
                    "THETA0_MANIFESTS.json", dict(manifests)
                )
                theta0_published = True

        result = run_av1b_screen(config, progress_callback=progress_callback)
        if not isinstance(result, Mapping):
            raise TypeError("run_av1b_screen must return a mapping")
        screen = dict(result)
        _assert_no_nonfinite(screen)
        _canonical_json_bytes(screen)
        if not theta0_published:
            raise RuntimeError("theta0 manifests were not published before first forward")
        lifecycle.publish_json("PROGRESS_EVENTS.json", progress_events)
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)

        for artifact_name, artifact_value in _screen_artifacts(screen).items():
            lifecycle.publish_or_match_json(artifact_name, artifact_value)

        post_screen_pins = _source_pins(card)
        if post_screen_pins != pins:
            raise RuntimeError("code/config/runner/test bytes changed during AV-1B execution")
        _validate_card(card_path)
        _validate_authority(config["authority"])
        if EXPECTED_REVIEW_CLAIM.exists() or EXPECTED_REVIEW_CLAIM.is_symlink():
            raise RuntimeError("review token claim appeared during first AV-1B execution")

        elapsed = time.monotonic() - started_monotonic
        rss = _rss_bytes()
        root_bytes = _root_size(lifecycle.root)
        gpu_peak_allocated = int(torch.cuda.max_memory_allocated(0))
        gpu_peak_reserved = int(torch.cuda.max_memory_reserved(0))
        budget = card["experiment_contract"]["budget"]
        budget_reasons: list[str] = []
        if elapsed + 2.0 > float(budget["max_wall_seconds_total"]):
            budget_reasons.append("WALL_BUDGET_EXCEEDED")
        if rss + 67_108_864 > int(budget["max_process_rss_bytes"]):
            budget_reasons.append("RSS_BUDGET_EXCEEDED")
        if gpu_peak_reserved > int(budget["max_gpu_memory_bytes"]):
            budget_reasons.append("GPU_MEMORY_BUDGET_EXCEEDED")
        if root_bytes + 4_194_304 > int(budget["max_root_bytes"]):
            budget_reasons.append("ROOT_BUDGET_EXCEEDED")
        completed = screen.get("diagnostics", {}).get("optimizer_steps_completed_by_arm")
        expected_steps = int(budget["max_optimizer_steps_total"]) // 2
        if completed != {
            "REF-GDN-CORRECTED": expected_steps,
            "CACH-A": expected_steps,
        }:
            budget_reasons.append("EXACT_OPTIMIZER_STEPS_NOT_COMPLETED")
        if budget_reasons:
            state, reasons = "HARNESS_REJECTED", budget_reasons
        else:
            state, reasons = _classify_screen(screen, card)
        screen = {
            **screen,
            "resource_usage": {
                **dict(screen.get("resource_usage", {})),
                "elapsed_seconds_launcher": elapsed,
                "process_peak_rss_bytes_launcher": rss,
                "root_size_bytes_before_terminal_artifacts": root_bytes,
                "cuda_peak_memory_allocated_bytes": gpu_peak_allocated,
                "cuda_peak_memory_reserved_bytes": gpu_peak_reserved,
            },
        }
    except BaseException as exc:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        state, reasons = _postcreate_failure_state(exc)
        screen = {
            "schema": "cach.av1b.vendor_gdn_bridge.failed_screen.v1",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    raw_evidence = {
        "schema": "cach.av1b.vendor_gdn_bridge.raw_evidence.v1",
        "proposed_typed_state": state,
        "validity_reason": reasons,
        "screen": screen,
        "resource_usage": {
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "process_peak_rss_bytes": _rss_bytes(),
            "root_size_bytes_before_terminal_artifacts": _root_size(lifecycle.root),
        },
        "forbidden_artifacts_written": False,
        "model_or_optimizer_state_serialized": False,
        "real_data_or_checkpoint_used": False,
        "review_token_state": "UNCONSUMED",
    }
    try:
        terminal = lifecycle.finalize(
            state=state,
            validity_reasons=reasons,
            raw_evidence=raw_evidence,
            source_pins=pins,
            started_monotonic=started_monotonic,
        )
    except BaseException as exc:
        lifecycle.emergency_finalize()
        sys.stderr.write(f"terminal finalization failed: {type(exc).__name__}: {exc}\n")
        return 3
    sys.stdout.buffer.write(_canonical_json_bytes(terminal))
    return 0 if state in {"OPERATOR_GO", "REVIEW_ONCE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
