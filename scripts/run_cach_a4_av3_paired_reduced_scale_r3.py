#!/usr/bin/env python3
"""One-shot AV3-R3 runner for init-before-peak-reset on a fresh root."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.util
import os
from pathlib import Path
import signal
import stat
import sys
from types import ModuleType
from typing import NoReturn


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
R2_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale_r2.py"
R2_CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale_r2.yaml"
R2_SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale_r2.py"
R2_TEST_PATH = REPO_ROOT / "tests/test_cach_a4_av3_paired_reduced_scale_r2.py"
R2_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_EXECUTION_R2_RUN_CARD.json"
)
R2_ATTEMPT_DIR = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    ".cach-a4-av3-r2-attempt-58cf90abd40b552226636f3c9455c1fb"
)
R2_ATTEMPT_PATH = R2_ATTEMPT_DIR / "ATTEMPT.json"
R2_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_av3_paired_reduced_scale_r2"
)
R2_NAMESPACE_LEAF = R2_NAMESPACE / "06f5d09127f8"
R2_ROOT = R2_NAMESPACE_LEAF / (
    "cach-a4-av3-r2-58cf90abd40b552226636f3c9455c1fb"
)
R2_RESULT_PATH = R2_ROOT / "RESULT.json"
R2_FREEZE_RECEIPT_PATH = R2_ROOT / "FREEZE_RECEIPT.json"
R2_RUN_CONTEXT_PATH = R2_ROOT / "RUN_CONTEXT.json"

BASE_CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale.yaml"
DATA_MANIFEST_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_DATA_SELECTION.json"
)

R3_SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale_r3.py"
R3_CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale_r3.yaml"
R3_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale_r3.py"
R3_TEST_PATH = REPO_ROOT / "tests/test_cach_a4_av3_paired_reduced_scale_r3.py"
R3_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_EXECUTION_R3_RUN_CARD.json"
)

R3_CARD_SCHEMA = "cach.architecture_validation.cach_a4_av3_execution_r3_run_card.v1"
R3_CARD_STATE = "FROZEN_AV3_R3_EXECUTION_AUTHORIZED"
R3_CARD_ID = "CACH-A4-AV3-PAIRED-REDUCED-SCALE-SEED0-EXECUTION-R3-v1"
R3_CONFIG_SCHEMA = "cach.cach_a4.av3_paired_reduced_scale_r3.runtime_overlay.v1"
R3_AUTHORITY = {
    "interpretation": "MATERIALIZE_VALIDATE_AND_EXECUTE_AV3_R3_CUDA_ALLOCATOR_INIT_FIX_FRESH_ROOT_SINGLE_GPU",
    "statement": "授权 AV3-R3",
    "statement_sha256": "f3f56879ce505ef88311726f6da026f1e603f0fc3ac678a11b539a92ad9bc3c5",
    "statement_utf8_bytes": 13,
    "unlisted_capability_default": "DENIED",
}
R3_FIX_CONTRACT = {
    "new_expression": "torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)",
    "old_expression": "torch.cuda.reset_peak_memory_stats(device)",
    "only_semantic_delta": "CUDA_ALLOCATOR_INITIALIZATION_BEFORE_PEAK_MEMORY_RESET",
    "scope": "MODULE_LOCAL_TORCH_PROXY_WITH_TRY_FINALLY_RESTORE",
}

EXPECTED_GPU_INDEX = 0
EXPECTED_GPU_UUID = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
EXPECTED_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_av3_paired_reduced_scale_r3"
)
EXPECTED_NAMESPACE_LEAF = EXPECTED_NAMESPACE / "06f5d09127f8"
EXPECTED_NONCE = "3bf88ade4d71e73035e9196fe00043fb"
EXPECTED_ROOT = EXPECTED_NAMESPACE_LEAF / f"cach-a4-av3-r3-{EXPECTED_NONCE}"
ATTEMPT_DIR = EXPECTED_NAMESPACE.parent / (
    f".cach-a4-av3-r3-attempt-{EXPECTED_NONCE}"
)
EXPECTED_ENV = {
    "CUDA_VISIBLE_DEVICES": EXPECTED_GPU_UUID,
    "FUSED_GDN_PRECISION": "0",
    "GDN_DISABLE_COMPILE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "TRITON_CACHE_DIR": str(EXPECTED_ROOT / "triton_cache"),
}

R2_RUNNER_SHA256 = "bed61fc7a9b5f7b14ca20f947b8ccb01d44ddae2f48055374abbe7b2dc7a67d7"
R2_CONFIG_SHA256 = "88ab7cfbd526a436df16a12f696fb5fb984b73c12187f1896f60998042cdb0ae"
R2_SOURCE_SHA256 = "d069c3af8a324842cdb8b9f3bdc1cc6d052c3e16e7398de5cd5290fcb8852d75"
R2_TEST_SHA256 = "30419e8a98ffd1ef5edfc8789ccb2eed1592f3c5711408aa20c41aa3e457c65f"
R2_CARD_SHA256 = "9a3c7f4763b2cbef22618301df1321e3cbc93f0b5fccf11c0620fc461eed2c02"
R2_ATTEMPT_SHA256 = "879539d4fab5a711c8d20821ee40d23d642ef51cb54759f27532287f43b583ee"
R2_RESULT_SHA256 = "b2573479850abafa05724e15d0d5d937ff5af3fe8f3672cbee179c70e2f9f46a"
R2_FREEZE_RECEIPT_SHA256 = "a6937788349de86f1427e767721fbd76e687b0aa68d37c2c78e6867f780daedc"
R2_RUN_CONTEXT_SHA256 = "9ca5442c4c2f76cf4ca318c4e7eb308d4381900da9f65dff150f51305914a899"
R2_CURRENT_INVENTORY_SHA256 = "e0d231b11a60789bcc8ac05fdf284e01fe804755d30dcb90ae512aa0e15591e6"
R2_PRE_RECEIPT_INVENTORY_SHA256 = "640183bf5568c514641d559d9a5134b04a536f9dbde8da5ed3f1bf8e98a033c3"
BASE_CONFIG_SHA256 = "8aa04f561d2ba47dcddc91d7504151978caa6f1072a0992d88a1b37100f1564d"
DATA_MANIFEST_SHA256 = "fecd1a9fb2f4371011d9c3c86469fed8f24f7f6aa5b3f1be9811fd447fb2ab95"
R3_SOURCE_SHA256 = "750c02f03cf571c5a532eb6316d912c64d886b036b11eb4fe344a68125292030"
R3_CONFIG_SHA256 = "0f25f57b757a21161016824b329104d2b1b8f11de6f23450b8f92b5c830f9a81"

R2_ATTEMPT_CONTENT = {
    "authority_statement_sha256": "14c15e396245db28c2aab632bfd00c0b6c9ba3ab1090fb316c4077b740b4bd6a",
    "authorized_execution_card_sha256": R2_CARD_SHA256,
    "automatic_rerun_allowed": False,
    "nonce": "58cf90abd40b552226636f3c9455c1fb",
    "root": str(R2_ROOT),
    "schema": "cach.cach_a4.av3_r2.attempt_receipt.v1",
    "state": "R2_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY",
}
R2_RESULT_CONTENT = {
    "automatic_rerun_allowed": False,
    "error": "Invalid device argument ",
    "error_type": "RuntimeError",
    "nonce": "58cf90abd40b552226636f3c9455c1fb",
    "phase": "model_execution",
    "reason_code": "RuntimeError",
    "root": str(R2_ROOT),
    "schema": "cach.cach_a4.av3_paired_reduced_scale.terminal_result.v1",
    "terminal_state": "IMPLEMENTATION_INVALID",
    "typed_verdict": "IMPLEMENTATION_INVALID",
    "valid_result": False,
}

INHERITED_PIN_SPECS = {
    "base_config": (BASE_CONFIG_PATH, BASE_CONFIG_SHA256),
    "data_manifest": (DATA_MANIFEST_PATH, DATA_MANIFEST_SHA256),
    "r2_attempt_receipt": (R2_ATTEMPT_PATH, R2_ATTEMPT_SHA256),
    "r2_config": (R2_CONFIG_PATH, R2_CONFIG_SHA256),
    "r2_execution_card": (R2_CARD_PATH, R2_CARD_SHA256),
    "r2_freeze_receipt": (R2_FREEZE_RECEIPT_PATH, R2_FREEZE_RECEIPT_SHA256),
    "r2_result": (R2_RESULT_PATH, R2_RESULT_SHA256),
    "r2_run_context": (R2_RUN_CONTEXT_PATH, R2_RUN_CONTEXT_SHA256),
    "r2_runner": (R2_RUNNER_PATH, R2_RUNNER_SHA256),
    "r2_source": (R2_SOURCE_PATH, R2_SOURCE_SHA256),
    "r2_test": (R2_TEST_PATH, R2_TEST_SHA256),
}
R3_MATERIALIZED_PATHS = {
    "config": R3_CONFIG_PATH,
    "runner": R3_RUNNER_PATH,
    "source": R3_SOURCE_PATH,
    "test": R3_TEST_PATH,
}
EXPECTED_CARD_KEYS = frozenset(
    {
        "authority",
        "card_id",
        "execution",
        "fix_contract",
        "forbidden",
        "identity",
        "inherited_contract",
        "inherited_pins",
        "materialized_pins",
        "predecessor_revision",
        "root_lifecycle_contract",
        "schema",
        "state",
    }
)
EXPECTED_FORBIDDEN = frozenset(
    {
        "AUTOMATIC_RERUN",
        "AV4",
        "BASE_R1_R2_AND_R3_DUAL_EXECUTION",
        "CHECKPOINT_LOAD_OR_SAVE",
        "DEPLOY",
        "FORMAL_EVALUATION_OR_ADMISSION",
        "FULL_2B",
        "GLOBAL_STAGE3",
        "OTHER_GPU_OR_MULTI_GPU",
        "POST_FREEZE_MUTATION",
        "PREDECESSOR_MUTATION",
        "R2_ROOT_REUSE_OR_MUTATION",
        "REVIEW_TOKEN_OR_CLAIM_OPERATION",
        "ROOT_REUSE",
    }
)


class R3ContractError(RuntimeError):
    """Raised before an unauthorized R3 capability can be reached."""


def _fail(message: str) -> NoReturn:
    raise R3ContractError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_r2_runner() -> ModuleType:
    metadata = R2_RUNNER_PATH.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail("frozen R2 runner is not a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        _fail("frozen R2 runner mode differs from 0444")
    if _sha(R2_RUNNER_PATH) != R2_RUNNER_SHA256:
        _fail("frozen R2 runner SHA differs")
    name = "_cach_a4_av3_paired_reduced_scale_frozen_r2_runner_for_r3"
    spec = importlib.util.spec_from_file_location(name, R2_RUNNER_PATH)
    if spec is None or spec.loader is None:
        _fail("cannot construct frozen R2 runner loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_r2 = _load_frozen_r2_runner()
_base = _r2._base


def _expected_execution() -> dict[str, object]:
    return {
        "automatic_rerun": False,
        "checkpoint_load": False,
        "checkpoint_save": False,
        "exact_environment": dict(EXPECTED_ENV),
        "gpu_count": 1,
        "gpu_index": EXPECTED_GPU_INDEX,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "logical_device": "cuda:0",
        "namespace": str(EXPECTED_NAMESPACE),
        "namespace_leaf": str(EXPECTED_NAMESPACE_LEAF),
        "nonce": EXPECTED_NONCE,
        "python": "/home/zch/workspace/sana-wam/.venv/bin/python",
        "root": str(EXPECTED_ROOT),
        "root_reuse": False,
    }


def _expected_predecessor_revision() -> dict[str, object]:
    return {
        "r2_attempt_receipt": {
            "expected_content": dict(R2_ATTEMPT_CONTENT),
            "path": str(R2_ATTEMPT_PATH),
            "sha256": R2_ATTEMPT_SHA256,
        },
        "r2_execution_card": {
            "path": str(R2_CARD_PATH.relative_to(REPO_ROOT)),
            "sha256": R2_CARD_SHA256,
        },
        "r2_terminal_evidence": {
            "current_inventory": {
                "row_count": 10,
                "serialization": "CANONICAL_UTF8_SORTED_COMPACT_JSON_PLUS_LF",
                "sha256": R2_CURRENT_INVENTORY_SHA256,
            },
            "freeze_receipt": {
                "path": str(R2_FREEZE_RECEIPT_PATH),
                "sha256": R2_FREEZE_RECEIPT_SHA256,
            },
            "nonce": "58cf90abd40b552226636f3c9455c1fb",
            "receipt_pre_inventory": {
                "row_count": 9,
                "serialization": "CANONICAL_UTF8_SORTED_COMPACT_JSON_PLUS_LF",
                "sha256": R2_PRE_RECEIPT_INVENTORY_SHA256,
            },
            "result": {
                "expected_content": dict(R2_RESULT_CONTENT),
                "path": str(R2_RESULT_PATH),
                "sha256": R2_RESULT_SHA256,
            },
            "root": str(R2_ROOT),
            "run_context": {
                "path": str(R2_RUN_CONTEXT_PATH),
                "sha256": R2_RUN_CONTEXT_SHA256,
            },
        },
        "runtime_preconditions": {
            "base_and_r1_failed_namespaces_absent": True,
            "r2_attempt_receipt_present_and_frozen": True,
            "r2_terminal_root_present_and_frozen": True,
        },
    }


def _verify_overlay(overlay: Mapping[str, object]) -> None:
    if frozenset(overlay) != {
        "authority",
        "base_config",
        "execution_override",
        "fix_contract",
        "predecessor_revision",
        "schema",
        "state",
    }:
        _fail("R3 overlay top-level keys differ")
    if (
        overlay.get("schema") != R3_CONFIG_SCHEMA
        or overlay.get("state") != "MATERIALIZED_VALIDATED_EXECUTION_AUTHORIZED"
        or overlay.get("authority") != R3_AUTHORITY
        or overlay.get("fix_contract") != R3_FIX_CONTRACT
    ):
        _fail("R3 overlay identity/authority/fix differs")
    if overlay.get("base_config") != {
        "path": str(BASE_CONFIG_PATH.relative_to(REPO_ROOT)),
        "sha256": BASE_CONFIG_SHA256,
    }:
        _fail("R3 overlay base config pin differs")
    expected_override = dict(_expected_execution())
    for key in ("gpu_count", "logical_device", "python"):
        expected_override.pop(key)
    if overlay.get("execution_override") != expected_override:
        _fail("R3 overlay execution identity differs")
    if overlay.get("predecessor_revision") != {
        "r2_attempt_receipt": {
            "path": str(R2_ATTEMPT_PATH),
            "sha256": R2_ATTEMPT_SHA256,
        },
        "r2_execution_card": {
            "path": str(R2_CARD_PATH.relative_to(REPO_ROOT)),
            "sha256": R2_CARD_SHA256,
        },
        "r2_freeze_receipt": {
            "path": str(R2_FREEZE_RECEIPT_PATH),
            "sha256": R2_FREEZE_RECEIPT_SHA256,
        },
        "r2_terminal_result": {
            "path": str(R2_RESULT_PATH),
            "sha256": R2_RESULT_SHA256,
        },
    }:
        _fail("R3 overlay predecessor revision differs")


def _verify_r2_frozen_failure() -> None:
    for path, expected_mode in (
        (R2_NAMESPACE, 0o2775),
        (R2_NAMESPACE_LEAF, 0o2775),
        (R2_ROOT, 0o555),
    ):
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != expected_mode
        ):
            _fail(f"R2 frozen directory type/mode differs: {path}")
    attempt_directory = R2_ATTEMPT_DIR.lstat()
    attempt_file = R2_ATTEMPT_PATH.lstat()
    if (
        stat.S_ISLNK(attempt_directory.st_mode)
        or not stat.S_ISDIR(attempt_directory.st_mode)
        or stat.S_IMODE(attempt_directory.st_mode) != 0o555
        or stat.S_ISLNK(attempt_file.st_mode)
        or not stat.S_ISREG(attempt_file.st_mode)
        or stat.S_IMODE(attempt_file.st_mode) != 0o444
    ):
        _fail("R2 attempt receipt type/mode differs")
    if _sha(R2_ATTEMPT_PATH) != R2_ATTEMPT_SHA256:
        _fail("R2 attempt receipt SHA differs")
    if _base._strict_json(R2_ATTEMPT_PATH) != R2_ATTEMPT_CONTENT:
        _fail("R2 attempt receipt content differs")

    inventory = _base._inventory(R2_ROOT)
    if len(inventory) != 10 or _base._sha_bytes(
        _base._canonical(inventory)
    ) != R2_CURRENT_INVENTORY_SHA256:
        _fail("R2 frozen root current inventory differs")
    expected_names = {
        "DATA_EVIDENCE.json",
        "FREEZE_RECEIPT.json",
        "JIT_INVENTORY.json",
        "PROGRESS_EVENTS.json",
        "RESOURCE_OBSERVATION.json",
        "RESULT.json",
        "RUN_CONTEXT.json",
        "SCREEN_RESULT.json",
        "THETA0_MANIFEST.json",
        "triton_cache",
    }
    if {path.name for path in R2_ROOT.iterdir()} != expected_names:
        _fail("R2 frozen root child set differs")
    cache = R2_ROOT / "triton_cache"
    if stat.S_IMODE(cache.lstat().st_mode) != 0o555 or any(cache.iterdir()):
        _fail("R2 frozen Triton cache is not empty/0555")

    result = _base._strict_json(R2_RESULT_PATH)
    receipt = _base._strict_json(R2_FREEZE_RECEIPT_PATH)
    context = _base._strict_json(R2_RUN_CONTEXT_PATH)
    if _sha(R2_RESULT_PATH) != R2_RESULT_SHA256 or result != R2_RESULT_CONTENT:
        _fail("R2 frozen RESULT differs")
    if _sha(R2_FREEZE_RECEIPT_PATH) != R2_FREEZE_RECEIPT_SHA256:
        _fail("R2 frozen freeze receipt SHA differs")
    pre_inventory = receipt.get("pre_receipt_inventory")
    if (
        not isinstance(pre_inventory, list)
        or len(pre_inventory) != 9
        or _base._sha_bytes(_base._canonical(pre_inventory))
        != R2_PRE_RECEIPT_INVENTORY_SHA256
        or receipt.get("result_sha256") != R2_RESULT_SHA256
        or receipt.get("root") != str(R2_ROOT)
        or receipt.get("nonce") != "58cf90abd40b552226636f3c9455c1fb"
        or receipt.get("terminal_state") != "IMPLEMENTATION_INVALID"
        or receipt.get("post_freeze_mutation_allowed") is not False
    ):
        _fail("R2 frozen freeze receipt content differs")
    if _sha(R2_RUN_CONTEXT_PATH) != R2_RUN_CONTEXT_SHA256 or (
        context.get("authorized_execution_card_sha256") != R2_CARD_SHA256
        or context.get("runner_sha256") != R2_RUNNER_SHA256
        or context.get("source_sha256") != R2_SOURCE_SHA256
        or context.get("test_sha256") != R2_TEST_SHA256
        or context.get("formal_admission") is not False
        or context.get("checkpoint_load") is not False
        or context.get("checkpoint_save") is not False
    ):
        _fail("R2 frozen run context differs")
    theta0 = _base._strict_json(R2_ROOT / "THETA0_MANIFEST.json")
    if (
        theta0.get("valid") is not False
        or theta0.get("phase") != "model_execution"
        or theta0.get("error") != "Invalid device argument "
    ):
        _fail("R2 theta0 failure boundary differs")


def _verify_card(
    card: Mapping[str, object], *, authorized_card_sha256: str
) -> dict[str, str]:
    if not _base._is_sha256(authorized_card_sha256):
        _fail("--execution-card-sha256 must be one lowercase SHA256")
    if _sha(R3_CARD_PATH) != authorized_card_sha256:
        _fail("R3 execution-card full SHA differs from supplied binding")
    if frozenset(card) != EXPECTED_CARD_KEYS:
        _fail("R3 execution-card top-level keys differ")
    if (
        card.get("schema") != R3_CARD_SCHEMA
        or card.get("state") != R3_CARD_STATE
        or card.get("card_id") != R3_CARD_ID
        or card.get("authority") != R3_AUTHORITY
        or card.get("fix_contract") != R3_FIX_CONTRACT
        or card.get("execution") != _expected_execution()
        or card.get("predecessor_revision") != _expected_predecessor_revision()
    ):
        _fail("R3 execution-card identity/authority/predecessor differs")
    if card.get("identity") != {
        "architecture_id": _base.EXPECTED_ARCHITECTURE_ID,
        "canonical_ssh_alias": "H200",
        "canonical_worktree": str(REPO_ROOT),
        "created_on": "2026-08-06",
        "self_path": str(R3_CARD_PATH.relative_to(REPO_ROOT)),
        "self_pin_policy": "OUT_OF_BAND_FULL_SHA256_AT_INVOCATION",
    }:
        _fail("R3 execution-card identity differs")
    if card.get("inherited_contract") != {
        "architecture_data_training_metric_verdict_artifact_resource_unchanged": True,
        "base_r1_r2_failures_are_not_architecture_evidence": True,
        "r2_card_transitive_closure_revalidated_at_preflight": True,
        "source_only_override": "CUDA_ALLOCATOR_INIT_BEFORE_EXISTING_PEAK_RESET",
    }:
        _fail("R3 inherited contract differs")
    if card.get("root_lifecycle_contract") != {
        "attempt_receipt": {
            "directory": str(ATTEMPT_DIR),
            "exclusive_create_before_data_or_gpu": True,
            "file": str(ATTEMPT_DIR / "ATTEMPT.json"),
            "static_preflight_consumes": False,
            "terminal_directory_mode": "0555",
            "terminal_file_mode": "0444",
        },
        "exclusive_create": True,
        "post_freeze_mutation": False,
        "root_reuse": False,
        "terminal_directory_mode": "0555",
        "terminal_file_mode": "0444",
    }:
        _fail("R3 root lifecycle differs")
    forbidden = card.get("forbidden")
    if (
        not isinstance(forbidden, list)
        or len(forbidden) != len(EXPECTED_FORBIDDEN)
        or frozenset(forbidden) != EXPECTED_FORBIDDEN
    ):
        _fail("R3 forbidden capability set differs")

    inherited = card.get("inherited_pins")
    if not isinstance(inherited, Mapping) or frozenset(inherited) != frozenset(
        INHERITED_PIN_SPECS
    ):
        _fail("R3 inherited pin roles differ")
    verified: dict[str, str] = {}
    for role, (path, expected_sha) in INHERITED_PIN_SPECS.items():
        expected_pin = {
            "path": (
                str(path)
                if path.is_absolute() and not path.is_relative_to(REPO_ROOT)
                else str(path.relative_to(REPO_ROOT))
            ),
            "sha256": expected_sha,
        }
        if inherited[role] != expected_pin:
            _fail(f"R3 inherited pin differs: {role}")
        verified[f"inherited:{role}"] = _base._verify_pin(
            inherited[role],
            expected_path=path,
            label=f"R3 inherited.{role}",
        )

    materialized = card.get("materialized_pins")
    if not isinstance(materialized, Mapping) or frozenset(materialized) != frozenset(
        R3_MATERIALIZED_PATHS
    ):
        _fail("R3 materialized pin roles differ")
    for role, path in R3_MATERIALIZED_PATHS.items():
        verified[f"materialized:{role}"] = _base._verify_pin(
            materialized[role],
            expected_path=path,
            label=f"R3 materialized.{role}",
        )
    if verified["materialized:config"] != R3_CONFIG_SHA256:
        _fail("R3 config SHA differs")
    if verified["materialized:source"] != R3_SOURCE_SHA256:
        _fail("R3 source SHA differs")
    verified["materialized:data_manifest"] = verified["inherited:data_manifest"]
    return verified


def _revalidate_r2_transitive_closure() -> dict[str, str]:
    r2_card = _base._strict_json(R2_CARD_PATH)
    r2_overlay = _base._strict_json(R2_CONFIG_PATH)
    r2_verified = _r2._verify_card(
        r2_card,
        authorized_card_sha256=R2_CARD_SHA256,
    )
    _r2._verify_overlay(r2_overlay)
    r1_closure = _r2._revalidate_r1_transitive_closure()
    _r2._verify_predecessor_runtime_state()
    return {
        **{
            f"predecessor_r2:{role}": digest
            for role, digest in r2_verified.items()
        },
        **{
            f"predecessor_r2:{role}": digest
            for role, digest in r1_closure.items()
        },
    }


def _static_preflight(authorized_card_sha256: str) -> dict[str, object]:
    card = _base._strict_json(R3_CARD_PATH)
    overlay = _base._strict_json(R3_CONFIG_PATH)
    base_config = _base._strict_json(BASE_CONFIG_PATH)
    manifest = _base._strict_json(DATA_MANIFEST_PATH)
    verified = _verify_card(card, authorized_card_sha256=authorized_card_sha256)
    _verify_overlay(overlay)
    verified.update(_revalidate_r2_transitive_closure())
    _verify_r2_frozen_failure()
    selected_files = _base._verify_manifest_structure(manifest)
    _base._require_paths_absent(
        (ATTEMPT_DIR,),
        label="AV3-R3 one-shot attempt receipt",
    )
    _base._require_paths_absent(
        (EXPECTED_NAMESPACE, EXPECTED_NAMESPACE_LEAF, EXPECTED_ROOT),
        label="future AV3-R3 namespace/root",
    )
    return {
        "authorized_card_sha256": authorized_card_sha256,
        "card": card,
        "config": base_config,
        "manifest": manifest,
        "overlay": overlay,
        "selected_files": selected_files,
        "verified_pins": verified,
    }


_CONSUME_ATTEMPT = False
_AUTHORIZED_CARD_SHA256: str | None = None


def _consume_attempt(authorized_card_sha256: str) -> None:
    global _AUTHORIZED_CARD_SHA256
    _base._require_paths_absent(
        (ATTEMPT_DIR,),
        label="AV3-R3 one-shot attempt receipt",
    )
    _base._verify_ancestor_chain(ATTEMPT_DIR, label="R3 attempt path")
    receipt_path = ATTEMPT_DIR / "ATTEMPT.json"
    payload = _base._canonical(
        {
            "authorized_execution_card_sha256": authorized_card_sha256,
            "authority_statement_sha256": R3_AUTHORITY["statement_sha256"],
            "automatic_rerun_allowed": False,
            "nonce": EXPECTED_NONCE,
            "root": str(EXPECTED_ROOT),
            "schema": "cach.cach_a4.av3_r3.attempt_receipt.v1",
            "state": "R3_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY",
        }
    )
    old_umask = os.umask(0o077)
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    directory_created = False
    descriptor: int | None = None
    try:
        os.mkdir(ATTEMPT_DIR, 0o700)
        directory_created = True
        _base._fsync_dir(ATTEMPT_DIR)
        _base._fsync_dir(ATTEMPT_DIR.parent)
        descriptor = os.open(
            receipt_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing R3 attempt receipt")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _base._fsync_dir(ATTEMPT_DIR)
        os.chmod(ATTEMPT_DIR, 0o555)
        _base._fsync_dir(ATTEMPT_DIR)
        _base._fsync_dir(ATTEMPT_DIR.parent)
        _AUTHORIZED_CARD_SHA256 = authorized_card_sha256
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if directory_created:
            try:
                if receipt_path.exists():
                    os.chmod(receipt_path, 0o444)
                os.chmod(ATTEMPT_DIR, 0o555)
                _base._fsync_dir(ATTEMPT_DIR.parent)
            except OSError:
                pass
        raise
    finally:
        os.umask(old_umask)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _verify_consumed_attempt() -> None:
    if _AUTHORIZED_CARD_SHA256 is None:
        raise RuntimeError("R3 attempt authority was not consumed")
    receipt_path = ATTEMPT_DIR / "ATTEMPT.json"
    directory = ATTEMPT_DIR.lstat()
    receipt_file = receipt_path.lstat()
    if (
        stat.S_ISLNK(directory.st_mode)
        or not stat.S_ISDIR(directory.st_mode)
        or stat.S_IMODE(directory.st_mode) != 0o555
        or stat.S_ISLNK(receipt_file.st_mode)
        or not stat.S_ISREG(receipt_file.st_mode)
        or stat.S_IMODE(receipt_file.st_mode) != 0o444
    ):
        raise RuntimeError("R3 attempt receipt is not frozen")
    receipt = _base._strict_json(receipt_path)
    if (
        receipt.get("authorized_execution_card_sha256") != _AUTHORIZED_CARD_SHA256
        or receipt.get("nonce") != EXPECTED_NONCE
        or receipt.get("root") != str(EXPECTED_ROOT)
        or receipt.get("state") != "R3_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY"
    ):
        raise RuntimeError("R3 attempt receipt content differs")


def _reject_parallel_predecessor_execution() -> None:
    snapshot = _base._related_process_snapshot()
    rows = snapshot.get("matching_rows")
    if not isinstance(rows, list):
        raise RuntimeError("R3 process observation is malformed")
    names = {
        "run_cach_a4_av3_paired_reduced_scale.py",
        "run_cach_a4_av3_paired_reduced_scale_r1.py",
        "run_cach_a4_av3_paired_reduced_scale_r2.py",
    }
    conflicting = [row for row in rows if any(name in str(row) for name in names)]
    if conflicting:
        raise RuntimeError(f"parallel predecessor AV3 execution observed: {conflicting}")


_original_static_preflight = _static_preflight


def _static_preflight_with_attempt(authorized_card_sha256: str) -> dict[str, object]:
    result = _original_static_preflight(authorized_card_sha256)
    if _CONSUME_ATTEMPT:
        _reject_parallel_predecessor_execution()
        _verify_r2_frozen_failure()
        _consume_attempt(authorized_card_sha256)
    return result


_BaseRootLifecycle = _r2._BaseRootLifecycle


class R3RootLifecycle(_BaseRootLifecycle):
    def create(self) -> None:
        _verify_r2_frozen_failure()
        _verify_consumed_attempt()
        _reject_parallel_predecessor_execution()
        super().create()


_base.CARD_SCHEMA = R3_CARD_SCHEMA
_base.CARD_STATE = R3_CARD_STATE
_base.CONFIG_PATH = R3_CONFIG_PATH
_base.DATA_MANIFEST_PATH = DATA_MANIFEST_PATH
_base.EXECUTION_CARD_PATH = R3_CARD_PATH
_base.EXPECTED_ENV = dict(EXPECTED_ENV)
_base.EXPECTED_GPU_INDEX = EXPECTED_GPU_INDEX
_base.EXPECTED_GPU_UUID = EXPECTED_GPU_UUID
_base.EXPECTED_NAMESPACE = EXPECTED_NAMESPACE
_base.EXPECTED_NAMESPACE_LEAF = EXPECTED_NAMESPACE_LEAF
_base.EXPECTED_NONCE = EXPECTED_NONCE
_base.EXPECTED_ROOT = EXPECTED_ROOT
_base.RUNNER_PATH = R3_RUNNER_PATH
_base.SOURCE_PATH = R3_SOURCE_PATH
_base.TEST_PATH = R3_TEST_PATH
_base.RootLifecycle = R3RootLifecycle
_base._observe_resources = _r2._r1._observe_resources
_base._static_preflight = _static_preflight_with_attempt


def main(argv: list[str] | None = None) -> int:
    global _CONSUME_ATTEMPT
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--execution-card-sha256", required=True)
    parsed = parser.parse_args(arguments)
    _CONSUME_ATTEMPT = not parsed.static_preflight
    return int(_base.main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
