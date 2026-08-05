#!/usr/bin/env python3
"""One-shot immutable runner for the CACH-A4 AV-3 reduced-scale screen.

Static preflight is standard-library-only and never imports torch, decodes
RoboTwin data, probes a GPU, sleeps, or creates a directory.  Full execution
is possible only when a caller supplies the exact frozen execution-card SHA;
it then verifies the selected files, observes the fixed GPU for 120 seconds,
loads the CPU cohort, creates one exclusive root, and lazily invokes the AV-3
core.  No checkpoint, token, formal admission, AV-4, or rerun path exists.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import grp
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, NoReturn


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
DECISION_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_PAIRED_REDUCED_SCALE_DECISION.md"
)
PLANNING_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_PAIRED_REDUCED_SCALE_RUN_CARD.json"
)
DATA_MANIFEST_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_DATA_SELECTION.json"
)
SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale.py"
CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale.yaml"
RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale.py"
TEST_PATH = REPO_ROOT / "tests/test_cach_a4_av3_paired_reduced_scale.py"
EXECUTION_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_EXECUTION_RUN_CARD.json"
)
AV2_BASE_EXECUTION_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av2_execution/"
    "CACH_A4_AV2_EXECUTION_RUN_CARD.json"
)

CARD_SCHEMA = "cach.architecture_validation.cach_a4_av3_execution_run_card.v1"
CARD_STATE = "FROZEN_AV3_EXECUTION_PENDING_OUT_OF_BAND_AUTHORITY"
CONFIG_SCHEMA = "cach.cach_a4.av3_paired_reduced_scale.config.v1"
DATA_SCHEMA = "cach.cach_a4.av3_paired_reduced_scale.data_selection.v1"
RESULT_SCHEMA = "cach.cach_a4.av3_paired_reduced_scale.terminal_result.v1"
RESOURCE_SCHEMA = "cach.cach_a4.av3_paired_reduced_scale.resource_observation.v1"
EXPECTED_ARCHITECTURE_ID = (
    "CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1"
)
EXPECTED_GPU_INDEX = 0
EXPECTED_GPU_UUID = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
EXPECTED_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_av3_paired_reduced_scale"
)
EXPECTED_NAMESPACE_LEAF = EXPECTED_NAMESPACE / "06f5d09127f8"
EXPECTED_NONCE = "97a25a9e1d614340697fbee523e19559"
EXPECTED_ROOT = EXPECTED_NAMESPACE_LEAF / f"cach-a4-av3-{EXPECTED_NONCE}"
MATERIALIZATION_AUTHORITY = "授权"
MATERIALIZATION_AUTHORITY_SHA256 = (
    "6cf2531ab1119b67ad2010040ae1ac73f817785684fccc26111b3b70ef5fcbe5"
)
PLANNING_CARD_SHA256 = (
    "33566c3291e49e4a87b4f83807d80255be67a5083615358c92bde924046bc82f"
)
DECISION_SHA256 = (
    "a0a211915704b3230747021a566e06ff47478b20f3acb2681973e6ac9e3ebd30"
)
KNOWN_MATERIALIZED_SHA256 = {
    "config": "8aa04f561d2ba47dcddc91d7504151978caa6f1072a0992d88a1b37100f1564d",
    "data_manifest": "fecd1a9fb2f4371011d9c3c86469fed8f24f7f6aa5b3f1be9811fd447fb2ab95",
    "source": "bafdb5475e1d7dddfba6d93e68ab2b67bf3960dba613da09ac26c48c8627bb6c",
    "test": "4aa1b4f6a5f2ef249e4954df154f1488b536a39aafaee2d196e8897ba258c577",
}
PRE_ROOT_OBSERVATION_SECONDS = 120
GPU_SAMPLE_INTERVAL_SECONDS = 5
MAX_DRIVER_MEMORY_MIB = 8
MAX_TOTAL_WALL_SECONDS = 60 * 60
MAX_ROOT_BYTES = 64 * 1024**2
MAX_RSS_BYTES = 8 * 1024**3
MAX_GPU_ALLOCATED_BYTES = 8 * 1024**3

EXPECTED_ENV = {
    "CUDA_VISIBLE_DEVICES": EXPECTED_GPU_UUID,
    "FUSED_GDN_PRECISION": "0",
    "GDN_DISABLE_COMPILE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "TRITON_CACHE_DIR": str(EXPECTED_ROOT / "triton_cache"),
}
MATERIALIZED_PATHS = {
    "config": CONFIG_PATH,
    "data_manifest": DATA_MANIFEST_PATH,
    "runner": RUNNER_PATH,
    "source": SOURCE_PATH,
    "test": TEST_PATH,
}
EXPECTED_PREDECESSOR_ROLES = frozenset(
    {
        "a4_r2_bridge",
        "a4_r2_card",
        "architecture_validation_first_plan",
        "av0_card",
        "av2_base_runner",
        "av2_config",
        "av2_data_selection",
        "av2_execution_card",
        "av2_model_source",
        "av2_r1_card",
        "av2_r1_runner",
        "av2_r2_card",
        "av2_r2_data_evidence",
        "av2_r2_freeze_receipt",
        "av2_r2_jit_inventory",
        "av2_r2_result",
        "av2_r2_run_context",
        "av2_r2_runner",
        "av2_r2_screen",
        "av2_scaffold_card",
        "av2_scaffold_decision",
        "av2_test",
        "review_token_claim",
    }
)
REQUIRED_ARTIFACTS = (
    "RESOURCE_OBSERVATION.json",
    "RUN_CONTEXT.json",
    "THETA0_MANIFEST.json",
    "DATA_EVIDENCE.json",
    "PROGRESS_EVENTS.json",
    "SCREEN_RESULT.json",
    "JIT_INVENTORY.json",
    "RESULT.json",
    "FREEZE_RECEIPT.json",
)
EXPECTED_CARD_KEYS = frozenset(
    {
        "architecture_contract",
        "artifact_contract",
        "card_id",
        "current_permissions",
        "data_contract",
        "execution",
        "execution_authority",
        "forbidden",
        "identity",
        "materialization_authority",
        "materialized_pins",
        "metric_contract",
        "observation_contract",
        "predecessor_pins",
        "resource_contract",
        "root_lifecycle_contract",
        "schema",
        "state",
        "training_contract",
        "transitive_av2_predecessor_pins",
        "validity_contract",
        "verdict_contract",
    }
)
EXPECTED_CONFIG_KEYS = frozenset(
    {
        "architecture_id",
        "authority",
        "data_contract",
        "forbidden",
        "hardware_contract",
        "integration_path",
        "metric_contract",
        "operator_class",
        "operator_level",
        "resource_contract",
        "rollout_contract",
        "run_contract",
        "schema",
        "state",
        "thresholds",
        "training_contract",
        "verdict_contract",
    }
)
EXPECTED_MANIFEST_KEYS = frozenset(
    {
        "action_normalization",
        "action_rows",
        "adapter_recipe",
        "adequacy",
        "camera",
        "created_on",
        "dataset_root",
        "derived_aggregates",
        "derived_microbatches",
        "episode_fallback",
        "frame_indices",
        "heldout_episode_ids",
        "heldout_microbatches",
        "manifest_serialization",
        "raw_rows",
        "recipe_source",
        "schema",
        "selected_files",
        "selection_digest",
        "state",
        "task",
        "train_episode_ids",
        "train_microbatches",
        "variant",
        "within_microbatch_shuffle_permutation",
    }
)
REQUIRED_FORBIDDEN = frozenset(
    {
        "AUTOMATIC_EXECUTION_OR_RERUN",
        "AV4_EXECUTION",
        "CHECKPOINT_LOAD_OR_SAVE",
        "DEPLOY",
        "FORMAL_EVALUATION_OR_ADMISSION",
        "FULL_2B",
        "GLOBAL_STAGE3",
        "INTERFERE_WITH_OTHER_GPU_OR_PROCESSES",
        "LOSS_SUBTRACTION_OR_TARGET_FORWARD_ACCESS",
        "OTHER_GPU_OR_MULTI_GPU",
        "POST_FREEZE_MUTATION",
        "PREDECESSOR_OR_MATERIALIZED_FILE_MUTATION",
        "REAL_DATA_OUTSIDE_FROZEN_SELECTION_OR_RESELECTION",
        "ROOT_REUSE",
        "SEED1_SEED2_OR_EXTRA_STEPS",
        "TOKEN_OR_CLAIM_OPERATION",
    }
)
VALID_ARCHITECTURE_VERDICTS = frozenset(
    {
        "GO",
        "REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED",
        "REDUCED_ARCH_STOP",
    }
)


class PreflightError(RuntimeError):
    pass


class RunnerInterrupted(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise PreflightError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return str(value)
    if value is None or type(value) in (bool, int, float):
        return value
    raise TypeError(f"non-JSON result value: {type(value).__name__}")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_ancestor_chain(path: Path, *, label: str) -> None:
    current = path.parent
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            _fail(f"missing ancestor for {label}: {current}")
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _fail(f"{label} has a symlink/non-directory ancestor: {current}")
        if current == Path("/"):
            break
        current = current.parent


def _regular_file(path: Path, *, label: str, frozen: bool = True) -> os.stat_result:
    _verify_ancestor_chain(path, label=label)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        _fail(f"missing {label}: {path}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} is not a regular non-symlink file: {path}")
    if frozen and stat.S_IMODE(metadata.st_mode) != 0o444:
        _fail(f"{label} is not frozen 0444: {path}")
    return metadata


def _strict_json(path: Path, *, require_canonical: bool = True) -> dict[str, Any]:
    _regular_file(path, label="JSON input")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        _fail(f"non-finite JSON constant {value!r} in {path}")

    payload = path.read_bytes()
    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"JSON root must be an object: {path}")
    if require_canonical and payload != _canonical(value):
        _fail(f"JSON is not canonical sorted compact UTF-8 plus LF: {path}")
    return value


def _resolve_pin_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        _fail(f"{label}.path must be a non-empty string")
    path = Path(raw)
    if ".." in path.parts:
        _fail(f"{label}.path contains traversal")
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if not path.is_absolute() and not resolved.is_relative_to(REPO_ROOT):
        _fail(f"{label}.path escapes the repository")
    return resolved


def _verify_pin(
    pin: object,
    *,
    expected_path: Path | None,
    label: str,
    frozen: bool = True,
) -> str:
    if not isinstance(pin, Mapping) or frozenset(pin) != {"path", "sha256"}:
        _fail(f"{label} must contain only path and sha256")
    path = _resolve_pin_path(pin["path"], label=label)
    if expected_path is not None and path != expected_path:
        _fail(f"{label} path differs: {path}")
    expected_sha = pin["sha256"]
    if not _is_sha256(expected_sha):
        _fail(f"{label}.sha256 is malformed")
    _regular_file(path, label=label, frozen=frozen)
    actual = _sha(path)
    if actual != expected_sha:
        _fail(f"{label} SHA mismatch: expected {expected_sha}, got {actual}")
    return actual


def _verify_planning_predecessors(planning: Mapping[str, object]) -> None:
    governing = planning.get("governing_pins")
    predecessor = planning.get("predecessor_pins")
    if not isinstance(governing, Mapping) or not isinstance(predecessor, Mapping):
        _fail("planning-card predecessor mappings are absent")
    if frozenset(predecessor) != EXPECTED_PREDECESSOR_ROLES - {
        "architecture_validation_first_plan",
        "av0_card",
    }:
        _fail("planning-card predecessor roles differ")
    if frozenset(governing) != {"architecture_validation_first_plan", "av0_card"}:
        _fail("planning-card governing roles differ")
    for role, pin in {**dict(governing), **dict(predecessor)}.items():
        _verify_pin(
            pin,
            expected_path=None,
            label=f"planning_predecessor.{role}",
            frozen=False,
        )


def _verify_execution_predecessor_equality(
    card: Mapping[str, object], planning: Mapping[str, object]
) -> None:
    governing = planning.get("governing_pins")
    planning_predecessors = planning.get("predecessor_pins")
    execution_predecessors = card.get("predecessor_pins")
    if not all(
        isinstance(value, Mapping)
        for value in (governing, planning_predecessors, execution_predecessors)
    ):
        _fail("AV-3 predecessor equality inputs are absent")
    assert isinstance(governing, Mapping)
    assert isinstance(planning_predecessors, Mapping)
    assert isinstance(execution_predecessors, Mapping)
    expected = {**dict(governing), **dict(planning_predecessors)}
    observed = {
        role: pin
        for role, pin in execution_predecessors.items()
        if role not in {"av3_decision", "av3_planning_card"}
    }
    if observed != expected:
        _fail("execution predecessor pins do not exactly equal the planning card")


def _verify_card(
    card: Mapping[str, object], *, authorized_card_sha256: str
) -> dict[str, str]:
    if not _is_sha256(authorized_card_sha256):
        _fail("--execution-card-sha256 must be one lowercase SHA256")
    actual_card_sha = _sha(EXECUTION_CARD_PATH)
    if actual_card_sha != authorized_card_sha256:
        _fail("execution-card full SHA differs from supplied authority binding")
    if frozenset(card) != EXPECTED_CARD_KEYS:
        _fail("execution-card top-level keys differ")
    if card.get("schema") != CARD_SCHEMA or card.get("state") != CARD_STATE:
        _fail("execution-card schema/state differs")
    identity = card.get("identity")
    authority = card.get("materialization_authority")
    execution_authority = card.get("execution_authority")
    execution = card.get("execution")
    if not all(
        isinstance(value, Mapping)
        for value in (identity, authority, execution_authority, execution)
    ):
        _fail("execution-card identity/authority/execution mapping is absent")
    assert isinstance(identity, Mapping)
    assert isinstance(authority, Mapping)
    assert isinstance(execution_authority, Mapping)
    assert isinstance(execution, Mapping)
    if (
        identity.get("architecture_id") != EXPECTED_ARCHITECTURE_ID
        or identity.get("canonical_worktree") != str(REPO_ROOT)
        or identity.get("self_path")
        != str(EXECUTION_CARD_PATH.relative_to(REPO_ROOT))
        or identity.get("self_pin_policy")
        != "OUT_OF_BAND_FULL_SHA256_AFTER_H200_SERIALIZATION"
    ):
        _fail("execution-card identity differs")
    if (
        authority.get("statement") != MATERIALIZATION_AUTHORITY
        or authority.get("statement_utf8_bytes") != 6
        or authority.get("statement_sha256") != MATERIALIZATION_AUTHORITY_SHA256
        or authority.get("interpretation")
        != "MATERIALIZE_AV3_SIX_FILES_CPU_STATIC_ONLY_NO_EXECUTION"
    ):
        _fail("materialization authority differs")
    if (
        execution_authority.get("state")
        != "PENDING_NEW_USER_AUTHORITY_BINDING_FULL_EXECUTION_CARD_SHA256"
        or execution_authority.get("card_mutation_required") is not False
        or execution_authority.get("automatic_execution") is not False
    ):
        _fail("out-of-band execution authority contract differs")
    expected_execution = {
        "gpu_index": EXPECTED_GPU_INDEX,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "namespace": str(EXPECTED_NAMESPACE),
        "namespace_leaf": str(EXPECTED_NAMESPACE_LEAF),
        "nonce": EXPECTED_NONCE,
        "root": str(EXPECTED_ROOT),
    }
    if any(
        execution.get(name) != expected
        or type(execution.get(name)) is not type(expected)
        for name, expected in expected_execution.items()
    ):
        _fail("execution identity differs")
    if (
        execution.get("root_reuse") is not False
        or execution.get("automatic_rerun") is not False
        or execution.get("checkpoint_load") is not False
        or execution.get("checkpoint_save") is not False
    ):
        _fail("execution must deny reuse/rerun/checkpoints")
    env = execution.get("exact_environment")
    if not isinstance(env, Mapping) or dict(env) != EXPECTED_ENV:
        _fail("execution exact environment differs")
    materialized = card.get("materialized_pins")
    if not isinstance(materialized, Mapping) or frozenset(materialized) != frozenset(
        MATERIALIZED_PATHS
    ):
        _fail("execution-card materialized roles differ")
    verified: dict[str, str] = {}
    for role, path in MATERIALIZED_PATHS.items():
        verified[f"materialized:{role}"] = _verify_pin(
            materialized[role],
            expected_path=path,
            label=f"materialized_pins.{role}",
        )
        if role in KNOWN_MATERIALIZED_SHA256 and (
            verified[f"materialized:{role}"] != KNOWN_MATERIALIZED_SHA256[role]
        ):
            _fail(f"known materialized {role} SHA differs")
    predecessor = card.get("predecessor_pins")
    if not isinstance(predecessor, Mapping) or frozenset(
        predecessor
    ) != EXPECTED_PREDECESSOR_ROLES | {"av3_decision", "av3_planning_card"}:
        _fail("execution-card predecessor roles differ")
    expected_fixed = {
        "av3_decision": DECISION_PATH,
        "av3_planning_card": PLANNING_CARD_PATH,
        "av2_execution_card": AV2_BASE_EXECUTION_CARD_PATH,
    }
    for role, pin in predecessor.items():
        verified[f"predecessor:{role}"] = _verify_pin(
            pin,
            expected_path=expected_fixed.get(role),
            label=f"predecessor_pins.{role}",
            frozen=False,
        )
    if verified["predecessor:av3_decision"] != DECISION_SHA256:
        _fail("AV-3 decision SHA differs")
    if verified["predecessor:av3_planning_card"] != PLANNING_CARD_SHA256:
        _fail("AV-3 planning-card SHA differs")
    transitive = card.get("transitive_av2_predecessor_pins")
    av2_base_card = _strict_json(
        AV2_BASE_EXECUTION_CARD_PATH,
        require_canonical=False,
    )
    av2_base_predecessors = av2_base_card.get("predecessor_pins")
    if (
        not isinstance(transitive, Mapping)
        or not isinstance(av2_base_predecessors, Mapping)
        or dict(transitive) != dict(av2_base_predecessors)
    ):
        _fail("transitive AV-2 predecessor map differs from the frozen base card")
    for role, pin in transitive.items():
        verified[f"transitive_av2:{role}"] = _verify_pin(
            pin,
            expected_path=None,
            label=f"transitive_av2_predecessor_pins.{role}",
            frozen=False,
        )
    artifacts = card.get("artifact_contract")
    resources = card.get("resource_contract")
    observation = card.get("observation_contract")
    training = card.get("training_contract")
    metric = card.get("metric_contract")
    verdict = card.get("verdict_contract")
    lifecycle = card.get("root_lifecycle_contract")
    permissions = card.get("current_permissions")
    if not all(
        isinstance(value, Mapping)
        for value in (
            artifacts,
            resources,
            observation,
            training,
            metric,
            verdict,
            lifecycle,
            permissions,
        )
    ):
        _fail("execution-card artifact/resource contract is absent")
    assert isinstance(artifacts, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(observation, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(metric, Mapping)
    assert isinstance(verdict, Mapping)
    assert isinstance(lifecycle, Mapping)
    assert isinstance(permissions, Mapping)
    if artifacts.get("exact_paths_in_publish_order") != list(REQUIRED_ARTIFACTS):
        _fail("artifact publish order differs")
    if (
        artifacts.get("success_requires_exact_complete_set") is not True
        or artifacts.get("failure_fills_missing_with_typed_placeholders") is not True
        or artifacts.get("canonical_json_sorted_compact_utf8_lf") is not True
    ):
        _fail("artifact closure/serialization contract differs")
    expected_resources = {
        "max_gpu_allocated_memory_bytes": MAX_GPU_ALLOCATED_BYTES,
        "max_process_rss_bytes": MAX_RSS_BYTES,
        "max_root_bytes": MAX_ROOT_BYTES,
        "max_total_wall_minutes": 60,
        "pre_root_observation_seconds": PRE_ROOT_OBSERVATION_SECONDS,
        "total_deadline_includes_pre_root_observation": True,
    }
    if dict(resources) != expected_resources:
        _fail("execution-card resource contract differs")
    expected_observation = {
        "duration_seconds": PRE_ROOT_OBSERVATION_SECONDS,
        "gpu_memory_used_mib_max": MAX_DRIVER_MEMORY_MIB,
        "gpu_utilization_percent": 0,
        "interval_seconds": GPU_SAMPLE_INTERVAL_SECONDS,
        "max_sample_gap_seconds": GPU_SAMPLE_INTERVAL_SECONDS + 2,
        "min_filesystem_free_bytes": 4 * MAX_ROOT_BYTES,
        "min_filesystem_free_inodes": 1024,
        "min_samples": PRE_ROOT_OBSERVATION_SECONDS
        // GPU_SAMPLE_INTERVAL_SECONDS
        + 1,
        "no_compute_apps": True,
        "post_cpu_data_gpu_recheck": True,
        "post_root_immediate_gpu_recheck": True,
        "target_base_group": "sharegrp",
        "target_base_owner_is_effective_user": True,
    }
    if dict(observation) != expected_observation:
        _fail("execution-card observation contract differs")
    expected_training = {
        "arm_order": ["reference", "candidate"],
        "checkpoint_load": False,
        "checkpoint_save": False,
        "final_model_state_save": False,
        "fresh_initialization": True,
        "optimizer_steps_per_arm": 1000,
        "serial_arms": True,
        "train_action_mode": "CORRECT_ONLY",
        "train_batch_schedule": "STEP_INDEX_MODULO_2",
    }
    if dict(training) != expected_training:
        _fail("execution-card training contract differs")
    if (
        metric.get("scored_horizons") != [1, 2, 4]
        or metric.get("horizon_weights") != {"1": 0.2, "2": 0.3, "4": 0.5}
        or metric.get("accumulator_dtype") != "float64"
        or metric.get("verdict_uses_final_step_only") is not True
    ):
        _fail("execution-card metric contract differs")
    if (
        verdict.get("go_state") != "GO"
        or verdict.get("strong_stop_state") != "REDUCED_ARCH_STOP"
        or verdict.get("review_band_state")
        != "REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED"
        or verdict.get("review_token_state") != "CONSUMED_NO_NEW_TOKEN_OR_RERUN"
        or verdict.get("automatic_rerun") is not False
    ):
        _fail("execution-card verdict contract differs")
    if (
        lifecycle.get("exclusive_create") is not True
        or lifecycle.get("root_reuse") is not False
        or lifecycle.get("terminal_directory_mode") != "0555"
        or lifecycle.get("terminal_file_mode") != "0444"
        or lifecycle.get("post_freeze_mutation") is not False
    ):
        _fail("execution-card root lifecycle differs")
    if (
        permissions.get("static_preflight") is not True
        or permissions.get("single_gpu_execution") is not False
        or permissions.get("root_creation") is not False
        or permissions.get("cpu_synthetic_tests") is not True
    ):
        _fail("execution-card current permissions differ")
    forbidden = card.get("forbidden")
    if not isinstance(forbidden, list) or frozenset(forbidden) != REQUIRED_FORBIDDEN:
        _fail("execution-card forbidden capability set differs")
    return verified


def _verify_config(config: Mapping[str, object], manifest_sha256: str) -> None:
    if frozenset(config) != EXPECTED_CONFIG_KEYS:
        _fail("AV-3 config top-level keys differ")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("architecture_id") != EXPECTED_ARCHITECTURE_ID
        or config.get("state") != "MATERIALIZED_NOT_EXECUTION_AUTHORIZED"
        or config.get("operator_level") != "VENDOR_KERNEL"
        or config.get("integration_path") != "EXPERIMENTAL_PATH"
    ):
        _fail("AV-3 config identity differs")
    authority = config.get("authority")
    data = config.get("data_contract")
    hardware = config.get("hardware_contract")
    run = config.get("run_contract")
    resources = config.get("resource_contract")
    training = config.get("training_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (authority, data, hardware, run, resources, training)
    ):
        _fail("AV-3 config contract mapping is absent")
    assert isinstance(authority, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(hardware, Mapping)
    assert isinstance(run, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(training, Mapping)
    if (
        authority.get("execution_authorized") is not False
        or authority.get("root_creation_authorized") is not False
        or authority.get("materialization_authorized") is not True
    ):
        _fail("AV-3 config authority state differs")
    if (
        data.get("manifest_path")
        != str(DATA_MANIFEST_PATH.relative_to(REPO_ROOT))
        or data.get("manifest_sha256") != manifest_sha256
        or data.get("train_microbatches")
        != [list(range(16, 24)), list(range(24, 32))]
        or data.get("heldout_microbatches")
        != [list(range(32, 40)), list(range(40, 48))]
        or data.get("target_forward_access") is not False
    ):
        _fail("AV-3 config data binding differs")
    if (
        hardware.get("physical_gpu_index") != EXPECTED_GPU_INDEX
        or hardware.get("gpu_uuid") != EXPECTED_GPU_UUID
        or hardware.get("cuda_visible_devices") != EXPECTED_GPU_UUID
        or hardware.get("logical_device") != "cuda:0"
    ):
        _fail("AV-3 config GPU binding differs")
    if (
        run.get("namespace") != str(EXPECTED_NAMESPACE)
        or run.get("namespace_leaf") != str(EXPECTED_NAMESPACE_LEAF)
        or run.get("root") != str(EXPECTED_ROOT)
        or run.get("nonce") != EXPECTED_NONCE
        or run.get("created") is not False
        or run.get("reuse") is not False
        or run.get("automatic_rerun") is not False
    ):
        _fail("AV-3 config run identity differs")
    expected_resources = {
        "max_gpu_allocated_memory_bytes": MAX_GPU_ALLOCATED_BYTES,
        "max_process_rss_bytes": MAX_RSS_BYTES,
        "max_root_bytes": MAX_ROOT_BYTES,
        "max_total_wall_minutes": 60,
        "pre_root_observation_seconds": PRE_ROOT_OBSERVATION_SECONDS,
        "total_deadline_includes_pre_root_observation": True,
    }
    if dict(resources) != expected_resources:
        _fail("AV-3 config resource contract differs")
    if (
        training.get("optimizer_steps_per_arm") != 1000
        or training.get("train_action_mode") != "CORRECT_ONLY"
        or training.get("checkpoint_load") is not False
        or training.get("checkpoint_save") is not False
        or training.get("final_model_state_save") is not False
    ):
        _fail("AV-3 config training contract differs")


def _verify_manifest_structure(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    if frozenset(manifest) != EXPECTED_MANIFEST_KEYS:
        _fail("AV-3 manifest top-level keys differ")
    if manifest.get("schema") != DATA_SCHEMA or manifest.get("state") != "FROZEN_PRE_MODEL_PRE_GPU":
        _fail("AV-3 data manifest identity differs")
    if (
        manifest.get("train_episode_ids") != list(range(16, 32))
        or manifest.get("heldout_episode_ids") != list(range(32, 48))
        or manifest.get("train_microbatches")
        != [list(range(16, 24)), list(range(24, 32))]
        or manifest.get("heldout_microbatches")
        != [list(range(32, 40)), list(range(40, 48))]
    ):
        _fail("AV-3 data selection differs")
    selection = manifest.get("selection_digest")
    if (
        not isinstance(selection, Mapping)
        or frozenset(selection)
        != {"canonical_json", "payload_has_trailing_lf", "serialization", "sha256"}
        or selection.get("payload_has_trailing_lf") is not False
        or selection.get("serialization")
        != "UTF8_JSON_SORTED_KEYS_COMPACT_SEPARATORS_NO_TRAILING_LF"
        or selection.get("sha256")
        != "49a56dc087d693a4f25005095f390996dab8639eb4e387bf8c70b872dfeb50d9"
        or _sha_bytes(str(selection.get("canonical_json")).encode("utf-8"))
        != selection.get("sha256")
    ):
        _fail("AV-3 manifest selection digest differs")
    derived = manifest.get("derived_microbatches")
    if not isinstance(derived, list) or len(derived) != 4:
        _fail("AV-3 manifest derived microbatch count differs")
    expected_derived = [
        ("train", 0, list(range(16, 24))),
        ("train", 1, list(range(24, 32))),
        ("heldout", 0, list(range(32, 40))),
        ("heldout", 1, list(range(40, 48))),
    ]
    tensor_roles = {
        "action_valid_mask",
        "actions",
        "frame_valid_mask",
        "noisy_video",
        "proprio",
        "target",
    }
    for index, (row, expected) in enumerate(zip(derived, expected_derived, strict=True)):
        if not isinstance(row, Mapping) or frozenset(row) != {
            "episode_ids",
            "microbatch_index",
            "split",
            "tensor_sha256",
        }:
            _fail(f"derived_microbatches[{index}] keys differ")
        split, microbatch_index, episode_ids = expected
        hashes = row.get("tensor_sha256")
        if (
            row.get("split") != split
            or row.get("microbatch_index") != microbatch_index
            or row.get("episode_ids") != episode_ids
            or not isinstance(hashes, Mapping)
            or frozenset(hashes) != tensor_roles
            or not all(_is_sha256(value) for value in hashes.values())
        ):
            _fail(f"derived_microbatches[{index}] identity/tensor pins differ")
    adequacy = manifest.get("adequacy")
    if not isinstance(adequacy, Mapping) or adequacy.get("all_adequate") is not True:
        _fail("AV-3 manifest adequacy evidence differs")
    rows = manifest.get("selected_files")
    if not isinstance(rows, list) or len(rows) != 32:
        _fail("AV-3 manifest must contain exactly 32 selected files")
    expected: list[tuple[str, int, int]] = []
    for split, groups in (
        ("train", (range(16, 24), range(24, 32))),
        ("heldout", (range(32, 40), range(40, 48))),
    ):
        for microbatch_index, episode_ids in enumerate(groups):
            expected.extend((split, microbatch_index, episode_id) for episode_id in episode_ids)
    for index, (row, identity) in enumerate(zip(rows, expected, strict=True)):
        if not isinstance(row, Mapping) or frozenset(row) != {
            "episode_id",
            "episode_length",
            "microbatch_index",
            "path",
            "sha256",
            "size_bytes",
            "split",
        }:
            _fail(f"selected_files[{index}] keys differ")
        split, microbatch_index, episode_id = identity
        if (
            row.get("split") != split
            or row.get("microbatch_index") != microbatch_index
            or row.get("episode_id") != episode_id
            or type(row.get("episode_length")) is not int
            or int(row["episode_length"]) < 33
            or type(row.get("size_bytes")) is not int
            or int(row["size_bytes"]) <= 0
            or not _is_sha256(row.get("sha256"))
        ):
            _fail(f"selected_files[{index}] identity/evidence is malformed")
        expected_path = Path(
            f"/DATA/share/RoboTwin2.0/dataset/adjust_bottle/"
            f"aloha-agilex_clean_50/data/episode{episode_id}.hdf5"
        )
        if row.get("path") != str(expected_path):
            _fail(f"selected_files[{index}] path differs")
    return rows


def _static_preflight(authorized_card_sha256: str) -> dict[str, object]:
    card = _strict_json(EXECUTION_CARD_PATH)
    planning = _strict_json(PLANNING_CARD_PATH, require_canonical=False)
    config = _strict_json(CONFIG_PATH)
    manifest = _strict_json(DATA_MANIFEST_PATH)
    verified = _verify_card(card, authorized_card_sha256=authorized_card_sha256)
    _verify_planning_predecessors(planning)
    _verify_execution_predecessor_equality(card, planning)
    _verify_config(config, verified["materialized:data_manifest"])
    selected_files = _verify_manifest_structure(manifest)
    _require_paths_absent(
        (EXPECTED_NAMESPACE, EXPECTED_NAMESPACE_LEAF, EXPECTED_ROOT),
        label="future AV-3 namespace/root",
    )
    return {
        "authorized_card_sha256": authorized_card_sha256,
        "card": card,
        "config": config,
        "manifest": manifest,
        "selected_files": selected_files,
        "verified_pins": verified,
    }


def _verify_selected_data_files(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for index, row in enumerate(rows):
        path = Path(str(row["path"]))
        metadata = _regular_file(path, label=f"selected_files[{index}]", frozen=False)
        if metadata.st_size != row["size_bytes"]:
            _fail(f"selected_files[{index}] size differs")
        digest = _sha(path)
        if digest != row["sha256"]:
            _fail(f"selected_files[{index}] SHA differs")
        verified[str(path)] = digest
    return verified


def _require_paths_absent(paths: Sequence[Path], *, label: str) -> None:
    for path in paths:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        kind = "symlink" if stat.S_ISLNK(metadata.st_mode) else "existing entry"
        _fail(f"{label} is not absent ({kind}): {path}")


def _nvidia_rows() -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=15,
    )
    rows: list[dict[str, object]] = []
    for raw in completed.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 5:
            raise RuntimeError("unexpected nvidia-smi GPU row")
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_used_mib": int(parts[3]),
                "utilization_gpu_percent": int(parts[4]),
            }
        )
    return rows


def _compute_apps() -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=15,
    )
    rows: list[dict[str, str]] = []
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) == 4:
            rows.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": parts[1],
                    "process_name": parts[2],
                    "used_memory_mib": parts[3],
                }
            )
    return rows


def _gpu_snapshot() -> Mapping[str, object]:
    matches = [row for row in _nvidia_rows() if row["index"] == EXPECTED_GPU_INDEX]
    if len(matches) != 1 or matches[0]["uuid"] != EXPECTED_GPU_UUID:
        raise RuntimeError("physical GPU identity differs from AV-3 binding")
    row = matches[0]
    apps = [item for item in _compute_apps() if item["gpu_uuid"] == EXPECTED_GPU_UUID]
    if (
        int(row["memory_used_mib"]) > MAX_DRIVER_MEMORY_MIB
        or int(row["utilization_gpu_percent"]) != 0
        or apps
    ):
        raise RuntimeError(f"fixed GPU is not stably idle: gpu={row}, apps={apps}")
    return {
        **row,
        "compute_apps": apps,
        "captured_unix_ns": time.time_ns(),
    }


def _filesystem_snapshot() -> Mapping[str, object]:
    base = EXPECTED_NAMESPACE.parent
    metadata = base.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("screen namespace base is not a non-symlink directory")
    filesystem = os.statvfs(base)
    free_bytes = filesystem.f_bavail * filesystem.f_frsize
    free_inodes = filesystem.f_favail
    try:
        group_name = grp.getgrgid(metadata.st_gid).gr_name
    except KeyError as exc:
        raise RuntimeError("screen namespace base group is unresolved") from exc
    if metadata.st_uid != os.geteuid() or group_name != "sharegrp":
        raise RuntimeError(
            "screen namespace base owner/group differs from effective-user:sharegrp"
        )
    if free_bytes < 4 * MAX_ROOT_BYTES or free_inodes < 1024:
        raise RuntimeError("screen filesystem free bytes/inodes are inadequate")
    ancestors: list[dict[str, object]] = []
    current = base
    while True:
        one = current.lstat()
        if stat.S_ISLNK(one.st_mode) or not stat.S_ISDIR(one.st_mode):
            raise RuntimeError(f"invalid target ancestor: {current}")
        ancestors.append(
            {
                "gid": one.st_gid,
                "mode": f"{stat.S_IMODE(one.st_mode):04o}",
                "path": str(current),
                "uid": one.st_uid,
            }
        )
        if current == Path("/"):
            break
        current = current.parent
    return {
        "ancestors": ancestors,
        "base_gid": metadata.st_gid,
        "base_uid": metadata.st_uid,
        "free_bytes": free_bytes,
        "free_inodes": free_inodes,
        "group_name": group_name,
        "path": str(base),
    }


def _related_process_snapshot() -> Mapping[str, object]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,user=,stat=,etimes=,args="],
        capture_output=True,
        check=True,
        text=True,
        timeout=15,
    )
    rows = [
        line.strip()
        for line in completed.stdout.splitlines()
        if "cach_a4_av3" in line or "run_cach_a4" in line
    ]
    return {"matching_rows": rows, "self_pid": os.getpid()}


def _observe_resources(
    *,
    seconds: int = PRE_ROOT_OBSERVATION_SECONDS,
    interval: int = GPU_SAMPLE_INTERVAL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    snapshot: Callable[[], Mapping[str, object]] = _gpu_snapshot,
) -> Mapping[str, object]:
    if seconds < 0 or interval <= 0:
        raise ValueError("resource observation duration/interval is malformed")
    filesystem = dict(_filesystem_snapshot())
    related_processes = dict(_related_process_snapshot())
    started = monotonic()
    samples: list[dict[str, object]] = []
    while True:
        captured = monotonic()
        one = dict(snapshot())
        one["monotonic_offset_seconds"] = captured - started
        samples.append(one)
        now = monotonic()
        if now - started >= float(seconds):
            break
        next_due = min(started + len(samples) * float(interval), started + float(seconds))
        sleeper(max(next_due - now, 0.0))
    elapsed = monotonic() - started
    if elapsed < float(seconds):
        raise RuntimeError("resource observation ended early")
    minimum_samples = seconds // interval + 1
    offsets = [float(item["monotonic_offset_seconds"]) for item in samples]
    maximum_gap = max(
        (right - left for left, right in zip(offsets, offsets[1:], strict=True)),
        default=0.0,
    )
    if len(samples) < minimum_samples or maximum_gap > float(interval + 2):
        raise RuntimeError("resource observation sample count/gap differs")
    return {
        "schema": RESOURCE_SCHEMA,
        "duration_required_seconds": seconds,
        "duration_observed_seconds": elapsed,
        "gpu_samples": samples,
        "maximum_sample_gap_seconds": maximum_gap,
        "sample_count": len(samples),
        "filesystem": filesystem,
        "related_processes": related_processes,
    }


def _load_source() -> ModuleType:
    repository_src = str(REPO_ROOT / "src")
    if repository_src in sys.path:
        sys.path.remove(repository_src)
    sys.path.insert(0, repository_src)
    name = "_cach_a4_av3_paired_reduced_scale_execution"
    spec = importlib.util.spec_from_file_location(name, SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct AV-3 source loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_cpu_bundle(
    source: ModuleType,
    config: Mapping[str, object],
    manifest: Mapping[str, object],
) -> tuple[object, Mapping[str, object]]:
    data = config.get("data_contract")
    if not isinstance(data, Mapping):
        raise RuntimeError("AV-3 data contract is absent")
    bundle = source.load_av3_real_data(
        data["dataset_root"],
        manifest,
        manifest_path=DATA_MANIFEST_PATH,
        expected_manifest_sha256=data["manifest_sha256"],
    )
    adequacy = source.assess_av3_data_adequacy(bundle)
    if not bool(adequacy["all_adequate"]):
        raise RuntimeError("DATA_INADEQUATE: fixed AV-3 cohort failed adequacy")
    return bundle, adequacy


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, value: object) -> str:
    payload = _canonical(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_dir(path.parent)
    return _sha_bytes(payload)


def _inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink in run root: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            rows.append(
                {
                    "kind": "directory",
                    "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                    "path": relative,
                }
            )
        elif stat.S_ISREG(metadata.st_mode):
            rows.append(
                {
                    "kind": "file",
                    "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                    "path": relative,
                    "sha256": _sha(path),
                    "size_bytes": metadata.st_size,
                }
            )
        else:
            raise RuntimeError(f"non-regular run-root entry: {relative}")
    return rows


def _root_size_bytes(root: Path) -> int:
    return sum(
        path.lstat().st_size
        for path in root.rglob("*")
        if stat.S_ISREG(path.lstat().st_mode)
    )


class RootLifecycle:
    def __init__(self) -> None:
        self.root = EXPECTED_ROOT
        self.created = False
        self.frozen = False
        self.published: dict[str, str] = {}

    def create(self) -> None:
        base = EXPECTED_NAMESPACE.parent
        metadata = base.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("screen namespace base is invalid")
        _require_paths_absent(
            (EXPECTED_NAMESPACE, EXPECTED_NAMESPACE_LEAF, EXPECTED_ROOT),
            label="AV-3 future root component",
        )
        old_umask = os.umask(0o002)
        blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        created_paths: list[Path] = []
        try:
            for path in (EXPECTED_NAMESPACE, EXPECTED_NAMESPACE_LEAF):
                os.mkdir(path, 0o2775)
                created_paths.append(path)
                os.chmod(path, 0o2775)
                _fsync_dir(path)
                _fsync_dir(path.parent)
            os.mkdir(self.root, 0o770)
            self.created = True
            created_paths.append(self.root)
            os.chmod(self.root, 0o770)
            _fsync_dir(self.root)
            _fsync_dir(self.root.parent)
            cache = self.root / "triton_cache"
            os.mkdir(cache, 0o770)
            created_paths.append(cache)
            _fsync_dir(cache)
            _fsync_dir(self.root)
        except BaseException:
            if not self.created:
                for path in reversed(created_paths):
                    try:
                        metadata = path.lstat()
                        if not stat.S_ISLNK(metadata.st_mode):
                            os.chmod(path, 0o555)
                            _fsync_dir(path)
                            _fsync_dir(path.parent)
                    except OSError:
                        pass
            raise
        finally:
            os.umask(old_umask)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def publish(self, name: str, value: object) -> str:
        expected_next = (
            REQUIRED_ARTIFACTS[len(self.published)]
            if len(self.published) < len(REQUIRED_ARTIFACTS)
            else None
        )
        if (
            not self.created
            or self.frozen
            or name in self.published
            or "/" in name
            or name not in REQUIRED_ARTIFACTS
            or name != expected_next
        ):
            raise RuntimeError(f"invalid AV-3 artifact publication: {name}")
        digest = _write_exclusive(self.root / name, value)
        self.published[name] = digest
        return digest

    def prospective_final_size_bytes(
        self,
        *,
        terminal: Mapping[str, object],
        terminal_state: str,
    ) -> int:
        if "RESULT.json" in self.published or "FREEZE_RECEIPT.json" in self.published:
            raise RuntimeError("AV-3 final-size projection must precede RESULT")
        result_payload = _canonical(terminal)
        inventory = _inventory(self.root)
        inventory.append(
            {
                "kind": "file",
                "mode": "0444",
                "path": "RESULT.json",
                "sha256": _sha_bytes(result_payload),
                "size_bytes": len(result_payload),
            }
        )
        inventory.sort(key=lambda item: str(item["path"]))
        receipt = {
            "schema": "cach.cach_a4.av3_paired_reduced_scale.freeze_receipt.v1",
            "automatic_rerun_allowed": False,
            "nonce": EXPECTED_NONCE,
            "post_freeze_mutation_allowed": False,
            "pre_receipt_inventory": inventory,
            "result_sha256": _sha_bytes(result_payload),
            "root": str(self.root),
            "terminal_directory_mode": "0555",
            "terminal_file_mode": "0444",
            "terminal_state": terminal_state,
        }
        return _root_size_bytes(self.root) + len(result_payload) + len(_canonical(receipt))

    def freeze(self, *, terminal_state: str, result_sha256: str) -> None:
        if self.frozen:
            return
        required_before_receipt = set(REQUIRED_ARTIFACTS) - {"FREEZE_RECEIPT.json"}
        if set(self.published) != required_before_receipt:
            missing = sorted(required_before_receipt - set(self.published))
            extra = sorted(set(self.published) - required_before_receipt)
            raise RuntimeError(
                f"AV-3 artifact closure differs before freeze: missing={missing}, extra={extra}"
            )
        receipt = {
            "schema": "cach.cach_a4.av3_paired_reduced_scale.freeze_receipt.v1",
            "automatic_rerun_allowed": False,
            "nonce": EXPECTED_NONCE,
            "post_freeze_mutation_allowed": False,
            "pre_receipt_inventory": _inventory(self.root),
            "result_sha256": result_sha256,
            "root": str(self.root),
            "terminal_directory_mode": "0555",
            "terminal_file_mode": "0444",
            "terminal_state": terminal_state,
        }
        self.publish("FREEZE_RECEIPT.json", receipt)
        for path in sorted(self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"symlink before AV-3 freeze: {path}")
            os.chmod(path, 0o555 if stat.S_ISDIR(metadata.st_mode) else 0o444)
        os.chmod(self.root, 0o555)
        _fsync_dir(self.root)
        _fsync_dir(self.root.parent)
        self.frozen = True

    def freeze_failure(
        self,
        exc: BaseException,
        *,
        terminal_state: str,
        phase: str,
    ) -> None:
        if not self.created or self.frozen:
            return
        failure = {
            "schema": RESULT_SCHEMA,
            "automatic_rerun_allowed": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "nonce": EXPECTED_NONCE,
            "reason_code": type(exc).__name__,
            "root": str(self.root),
            "phase": phase,
            "terminal_state": terminal_state,
            "typed_verdict": terminal_state,
            "valid_result": False,
        }
        try:
            placeholder_schema = (
                "cach.cach_a4.av3_paired_reduced_scale."
                "failure_artifact_placeholder.v1"
            )
            for name in REQUIRED_ARTIFACTS[:-2]:
                if name not in self.published:
                    self.publish(
                        name,
                        {
                            "schema": placeholder_schema,
                            "artifact": name,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "phase": phase,
                            "terminal_state": terminal_state,
                            "valid": False,
                        },
                    )
            result_sha = self.published.get("RESULT.json")
            if result_sha is None:
                result_sha = self.publish("RESULT.json", failure)
            self.freeze(terminal_state=terminal_state, result_sha256=result_sha)
        except BaseException:
            for path in sorted(self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                try:
                    metadata = path.lstat()
                    if not stat.S_ISLNK(metadata.st_mode):
                        os.chmod(path, 0o555 if stat.S_ISDIR(metadata.st_mode) else 0o444)
                except OSError:
                    pass
            try:
                os.chmod(self.root, 0o555)
                _fsync_dir(self.root.parent)
            except OSError:
                pass
            self.frozen = True


def _jit_inventory() -> Mapping[str, object]:
    cache = EXPECTED_ROOT / "triton_cache"
    rows = _inventory(cache) if cache.is_dir() else []
    return {
        "cache_path": str(cache),
        "entries": rows,
        "entry_count": len(rows),
        "vendor_jit_or_cache_observed": bool(rows),
    }


def _check_exact_env() -> None:
    for name, expected in EXPECTED_ENV.items():
        if os.environ.get(name) != expected:
            _fail(f"environment {name} differs: expected {expected!r}")


def _typed_failure_state(exc: BaseException, *, phase: str) -> str:
    message = str(exc).lower()
    if "data_inadequate" in message or "cohort failed adequacy" in message:
        return "DATA_INADEQUATE"
    if "non-finite" in message or "numerical" in message:
        return "NUMERICAL_INVALID"
    if "deadline" in message or "resource" in message or "ceiling" in message:
        if phase in {
            "exact_environment",
            "selected_data_file_verification",
            "resource_observation",
            "cpu_data_load_and_adequacy",
            "post_cpu_data_gpu_recheck",
        }:
            return "ENV_BLOCKED"
        return "INVALID_RUN"
    if phase in {
        "exact_environment",
        "resource_observation",
        "post_cpu_data_gpu_recheck",
        "post_root_gpu_recheck",
    } or "gpu" in message:
        return "ENV_BLOCKED"
    if phase == "static_preflight":
        if "authority" in message or "execution-card" in message:
            return "AUTH_BLOCKED"
        return "HARNESS_REJECTED"
    if isinstance(exc, RunnerInterrupted):
        return "INVALID_RUN"
    return "IMPLEMENTATION_INVALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--execution-card-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lifecycle: RootLifecycle | None = None
    previous_handlers: dict[int, object] = {}
    started = time.monotonic()
    deadline = started + MAX_TOTAL_WALL_SECONDS
    phase = "static_preflight"
    alarm_installed = False
    completed_terminal: Mapping[str, object] | None = None

    def interrupted(signum: int, _frame: object) -> None:
        raise RunnerInterrupted(f"received signal {signum}")

    try:
        preflight = _static_preflight(args.execution_card_sha256)
        if args.static_preflight:
            print(
                _canonical(
                    {
                        "card_sha256": args.execution_card_sha256,
                        "data_manifest_sha256": preflight["verified_pins"][
                            "materialized:data_manifest"
                        ],
                        "root": str(EXPECTED_ROOT),
                        "schema": CARD_SCHEMA,
                        "state": "STATIC_PREFLIGHT_OK",
                        "verified_pin_count": len(preflight["verified_pins"]),
                    }
                ).decode("utf-8"),
                end="",
            )
            return 0

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise RuntimeError("AV-3 total deadline expired before full execution")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        alarm_installed = True
        phase = "exact_environment"
        _check_exact_env()
        phase = "selected_data_file_verification"
        data_file_pins = _verify_selected_data_files(preflight["selected_files"])
        phase = "resource_observation"
        observation = _observe_resources()
        if time.monotonic() >= deadline:
            raise RuntimeError("AV-3 total deadline expired during resource observation")
        phase = "cpu_data_load_and_adequacy"
        source = _load_source()
        bundle, adequacy = _load_cpu_bundle(
            source,
            preflight["config"],
            preflight["manifest"],
        )
        phase = "post_cpu_data_gpu_recheck"
        observation = {
            **dict(observation),
            "post_cpu_data_gpu_snapshot": dict(_gpu_snapshot()),
        }
        phase = "root_creation"
        lifecycle = RootLifecycle()
        lifecycle.create()
        phase = "post_root_gpu_recheck"
        post_root_gpu = _gpu_snapshot()
        observation = {
            **dict(observation),
            "post_root_immediate_gpu_snapshot": dict(post_root_gpu),
        }
        phase = "initial_artifact_publication"
        lifecycle.publish("RESOURCE_OBSERVATION.json", observation)
        lifecycle.publish(
            "RUN_CONTEXT.json",
            {
                "schema": "cach.cach_a4.av3_paired_reduced_scale.run_context.v1",
                "authorized_execution_card_sha256": args.execution_card_sha256,
                "checkpoint_load": False,
                "checkpoint_save": False,
                "data_manifest_sha256": preflight["verified_pins"][
                    "materialized:data_manifest"
                ],
                "exact_environment": dict(EXPECTED_ENV),
                "formal_admission": False,
                "fresh_initialization": True,
                "gpu": post_root_gpu,
                "nonce": EXPECTED_NONCE,
                "root": str(EXPECTED_ROOT),
                "runner_sha256": preflight["verified_pins"]["materialized:runner"],
                "source_sha256": preflight["verified_pins"]["materialized:source"],
                "test_sha256": preflight["verified_pins"]["materialized:test"],
                "total_deadline_monotonic": deadline,
            },
        )
        progress: list[object] = [
            {"event": "post_root_pre_cuda_gpu_idle_snapshot", "payload": post_root_gpu}
        ]

        def theta0_callback(value: Mapping[str, object]) -> None:
            lifecycle.publish("THETA0_MANIFEST.json", value)

        def progress_callback(value: Mapping[str, object]) -> None:
            progress.append({"event": "optimizer_progress", **dict(value)})

        phase = "model_execution"
        screen = source.run_av3_paired_reduced_scale(
            preflight["config"],
            expected_gpu_uuid=EXPECTED_GPU_UUID,
            total_deadline_monotonic=deadline,
            source_data_and_predecessor_pins_match=True,
            manifest=preflight["manifest"],
            manifest_path=DATA_MANIFEST_PATH,
            theta0_callback=theta0_callback,
            progress_callback=progress_callback,
            preloaded_bundle=bundle,
        )
        _canonical(screen)
        phase = "result_artifact_publication"
        lifecycle.publish(
            "DATA_EVIDENCE.json",
            {
                "adequacy": adequacy,
                "selected_file_sha256": data_file_pins,
            },
        )
        lifecycle.publish("PROGRESS_EVENTS.json", progress)
        lifecycle.publish("SCREEN_RESULT.json", screen)
        lifecycle.publish("JIT_INVENTORY.json", _jit_inventory())
        validity = screen.get("validity") if isinstance(screen, Mapping) else None
        verdict = screen.get("verdict") if isinstance(screen, Mapping) else None
        reason = screen.get("reason_code") if isinstance(screen, Mapping) else None
        resources = screen.get("resources") if isinstance(screen, Mapping) else None
        if (
            not isinstance(validity, Mapping)
            or not isinstance(verdict, str)
            or not isinstance(resources, Mapping)
        ):
            raise RuntimeError("AV-3 core returned a malformed result")
        run_valid = validity.get("run_valid")
        if type(run_valid) is not bool:
            raise RuntimeError("AV-3 core run_valid must be an exact bool")
        if run_valid and verdict not in VALID_ARCHITECTURE_VERDICTS:
            raise RuntimeError("valid AV-3 core verdict is outside the frozen state set")
        if not run_valid and verdict != "INVALID_RUN":
            raise RuntimeError("invalid AV-3 core result must use INVALID_RUN")
        if (
            time.monotonic() >= deadline
            or int(resources["process_rss_high_water_bytes"]) > MAX_RSS_BYTES
            or int(resources["peak_cuda_allocated_bytes"]) > MAX_GPU_ALLOCATED_BYTES
        ):
            raise RuntimeError("AV-3 deadline/resource/root ceiling exceeded")
        terminal = {
            "schema": RESULT_SCHEMA,
            "automatic_rerun_allowed": False,
            "av4_unlocked": verdict == "GO" and run_valid,
            "checkpoint_load_or_save": False,
            "formal_claim_allowed": False,
            "gpu_uuid": EXPECTED_GPU_UUID,
            "nonce": EXPECTED_NONCE,
            "reason_code": reason,
            "root": str(EXPECTED_ROOT),
            "screen_result_sha256": lifecycle.published["SCREEN_RESULT.json"],
            "terminal_state": verdict,
            "typed_verdict": verdict,
            "valid_result": run_valid,
        }
        projected_final_size = lifecycle.prospective_final_size_bytes(
            terminal=terminal,
            terminal_state=verdict,
        )
        if projected_final_size > MAX_ROOT_BYTES:
            raise RuntimeError(
                f"AV-3 projected final root exceeds ceiling: {projected_final_size}"
            )
        critical_signals = {
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGHUP,
            signal.SIGALRM,
        }
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, critical_signals)
        try:
            if alarm_installed:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                alarm_installed = False
            result_sha = lifecycle.publish("RESULT.json", terminal)
            phase = "terminal_freeze"
            lifecycle.freeze(terminal_state=verdict, result_sha256=result_sha)
            completed_terminal = terminal
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        print(_canonical(terminal).decode("utf-8"), end="")
        return 0 if bool(terminal["valid_result"]) else 2
    except BaseException as exc:
        if lifecycle is not None and lifecycle.frozen and completed_terminal is not None:
            print(_canonical(completed_terminal).decode("utf-8"), end="")
            return 0 if bool(completed_terminal["valid_result"]) else 2
        if lifecycle is not None:
            terminal_state = _typed_failure_state(exc, phase=phase)
            try:
                lifecycle.freeze_failure(
                    exc,
                    terminal_state=terminal_state,
                    phase=phase,
                )
            except BaseException:
                pass
        else:
            terminal_state = _typed_failure_state(exc, phase=phase)
        print(
            _canonical(
                {
                    "automatic_rerun_allowed": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "root_created": bool(lifecycle and lifecycle.created),
                    "schema": RESULT_SCHEMA,
                    "phase": phase,
                    "state": terminal_state,
                }
            ).decode("utf-8"),
            end="",
        )
        return 2
    finally:
        if alarm_installed:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(main())
