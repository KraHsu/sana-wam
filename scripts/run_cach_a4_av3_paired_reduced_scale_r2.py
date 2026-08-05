#!/usr/bin/env python3
"""One-shot AV3-R2 runner for the native 16+16 disjointness fix."""

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
R1_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale_r1.py"
R1_CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale_r1.yaml"
R1_TEST_PATH = REPO_ROOT / "tests/test_cach_a4_av3_paired_reduced_scale_r1.py"
R1_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_EXECUTION_R1_RUN_CARD.json"
)
R1_ATTEMPT_DIR = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    ".cach-a4-av3-r1-attempt-5d0a6205440f43921351b32a5d5588a8"
)
R1_ATTEMPT_PATH = R1_ATTEMPT_DIR / "ATTEMPT.json"
R1_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_av3_paired_reduced_scale_r1"
)
R1_NAMESPACE_LEAF = R1_NAMESPACE / "06f5d09127f8"
R1_ROOT = R1_NAMESPACE_LEAF / (
    "cach-a4-av3-r1-5d0a6205440f43921351b32a5d5588a8"
)

BASE_CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale.yaml"
BASE_SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale.py"
DATA_MANIFEST_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_DATA_SELECTION.json"
)
PLANNING_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_PAIRED_REDUCED_SCALE_RUN_CARD.json"
)
BASE_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_av3_paired_reduced_scale"
)
BASE_NAMESPACE_LEAF = BASE_NAMESPACE / "06f5d09127f8"
BASE_ROOT = BASE_NAMESPACE_LEAF / (
    "cach-a4-av3-97a25a9e1d614340697fbee523e19559"
)

R2_SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale_r2.py"
R2_CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale_r2.yaml"
R2_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale_r2.py"
R2_TEST_PATH = REPO_ROOT / "tests/test_cach_a4_av3_paired_reduced_scale_r2.py"
R2_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_EXECUTION_R2_RUN_CARD.json"
)

R2_CARD_SCHEMA = "cach.architecture_validation.cach_a4_av3_execution_r2_run_card.v1"
R2_CARD_STATE = "FROZEN_AV3_R2_EXECUTION_AUTHORIZED"
R2_CARD_ID = "CACH-A4-AV3-PAIRED-REDUCED-SCALE-SEED0-EXECUTION-R2-v1"
R2_CONFIG_SCHEMA = "cach.cach_a4.av3_paired_reduced_scale_r2.runtime_overlay.v1"
R2_AUTHORITY = {
    "interpretation": "MATERIALIZE_VALIDATE_AND_EXECUTE_AV3_R2_NATIVE_16X16_DISJOINT_FIX_FRESH_ROOT_SINGLE_GPU",
    "statement": "授权 AV3-R2",
    "statement_sha256": "14c15e396245db28c2aab632bfd00c0b6c9ba3ab1090fb316c4077b740b4bd6a",
    "statement_utf8_bytes": 13,
    "unlisted_capability_default": "DENIED",
}
R2_FIX_CONTRACT = {
    "new": "AV3_NATIVE_16_TRAIN_16_HELDOUT_DISJOINT_VALIDATOR",
    "old": "av2.av2_scaffold.assert_train_holdout_disjoint(train_windows, heldout_windows)",
    "only_semantic_delta": "AV3_FLATTENED_COHORT_CARDINALITY_AND_DISJOINT_VALIDATION",
}

EXPECTED_GPU_INDEX = 0
EXPECTED_GPU_UUID = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
EXPECTED_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_av3_paired_reduced_scale_r2"
)
EXPECTED_NAMESPACE_LEAF = EXPECTED_NAMESPACE / "06f5d09127f8"
EXPECTED_NONCE = "58cf90abd40b552226636f3c9455c1fb"
EXPECTED_ROOT = EXPECTED_NAMESPACE_LEAF / f"cach-a4-av3-r2-{EXPECTED_NONCE}"
ATTEMPT_DIR = EXPECTED_NAMESPACE.parent / (
    f".cach-a4-av3-r2-attempt-{EXPECTED_NONCE}"
)
EXPECTED_ENV = {
    "CUDA_VISIBLE_DEVICES": EXPECTED_GPU_UUID,
    "FUSED_GDN_PRECISION": "0",
    "GDN_DISABLE_COMPILE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "TRITON_CACHE_DIR": str(EXPECTED_ROOT / "triton_cache"),
}

R1_RUNNER_SHA256 = "1ab67bf220f2e4b882e6f2cdf8bc430b33370d65df688abb0fd502b18cc96d5a"
R1_CONFIG_SHA256 = "335a559dd79cfaccea36df32212de10c2b604427701e2609321c17ebfc259a90"
R1_TEST_SHA256 = "cf03e4ffd556d4cd52b784134928b5f85f78897cdc61e0c25cf8be42b4cdf228"
R1_CARD_SHA256 = "9e84e816fc9f6349a188a1adaebd8ede0ce92430a9db48d29ee52cd678a04c3e"
R1_ATTEMPT_SHA256 = "63a7afdc9b2f0ba7c704d6b2b6366c196dd9b091c0a57ff997e4f989a545add3"
R1_FAILURE_SHA256 = "6a748764781f90cba3491a44e74b4f52b292fb41dbe6efd83f344d70f398941e"
BASE_CONFIG_SHA256 = "8aa04f561d2ba47dcddc91d7504151978caa6f1072a0992d88a1b37100f1564d"
BASE_SOURCE_SHA256 = "bafdb5475e1d7dddfba6d93e68ab2b67bf3960dba613da09ac26c48c8627bb6c"
DATA_MANIFEST_SHA256 = "fecd1a9fb2f4371011d9c3c86469fed8f24f7f6aa5b3f1be9811fd447fb2ab95"
R2_SOURCE_SHA256 = "d069c3af8a324842cdb8b9f3bdc1cc6d052c3e16e7398de5cd5290fcb8852d75"
R2_CONFIG_SHA256 = "88ab7cfbd526a436df16a12f696fb5fb984b73c12187f1896f60998042cdb0ae"

R1_FAILURE_RECORD = {
    "automatic_rerun_allowed": False,
    "error": "AV-2 requires exactly 8 train and 8 held-out windows",
    "error_type": "AV2ContractError",
    "phase": "cpu_data_load_and_adequacy",
    "root_created": False,
    "schema": "cach.cach_a4.av3_paired_reduced_scale.terminal_result.v1",
    "state": "IMPLEMENTATION_INVALID",
}
R1_ATTEMPT_CONTENT = {
    "authority_statement_sha256": "fef23c2741999ead4f5f6bce68d87e68f6d75d0f503bca11fb2e348d2ecda574",
    "authorized_execution_card_sha256": R1_CARD_SHA256,
    "automatic_rerun_allowed": False,
    "nonce": "5d0a6205440f43921351b32a5d5588a8",
    "root": str(R1_ROOT),
    "schema": "cach.cach_a4.av3_r1.attempt_receipt.v1",
    "state": "R1_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY",
}

INHERITED_PIN_SPECS = {
    "base_config": (BASE_CONFIG_PATH, BASE_CONFIG_SHA256),
    "base_source": (BASE_SOURCE_PATH, BASE_SOURCE_SHA256),
    "data_manifest": (DATA_MANIFEST_PATH, DATA_MANIFEST_SHA256),
    "r1_attempt_receipt": (R1_ATTEMPT_PATH, R1_ATTEMPT_SHA256),
    "r1_config": (R1_CONFIG_PATH, R1_CONFIG_SHA256),
    "r1_execution_card": (R1_CARD_PATH, R1_CARD_SHA256),
    "r1_runner": (R1_RUNNER_PATH, R1_RUNNER_SHA256),
    "r1_test": (R1_TEST_PATH, R1_TEST_SHA256),
}
R2_MATERIALIZED_PATHS = {
    "config": R2_CONFIG_PATH,
    "runner": R2_RUNNER_PATH,
    "source": R2_SOURCE_PATH,
    "test": R2_TEST_PATH,
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
        "BASE_R1_AND_R2_DUAL_EXECUTION",
        "CHECKPOINT_LOAD_OR_SAVE",
        "DEPLOY",
        "FORMAL_EVALUATION_OR_ADMISSION",
        "FULL_2B",
        "GLOBAL_STAGE3",
        "OTHER_GPU_OR_MULTI_GPU",
        "POST_FREEZE_MUTATION",
        "PREDECESSOR_MUTATION",
        "REVIEW_TOKEN_OR_CLAIM_OPERATION",
        "ROOT_REUSE",
    }
)


class R2ContractError(RuntimeError):
    """Raised before an unauthorized R2 capability can be reached."""


def _fail(message: str) -> NoReturn:
    raise R2ContractError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_r1_runner() -> ModuleType:
    metadata = R1_RUNNER_PATH.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail("frozen R1 runner is not a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        _fail("frozen R1 runner mode differs from 0444")
    if _sha(R1_RUNNER_PATH) != R1_RUNNER_SHA256:
        _fail("frozen R1 runner SHA differs")
    name = "_cach_a4_av3_paired_reduced_scale_frozen_r1_for_r2"
    spec = importlib.util.spec_from_file_location(name, R1_RUNNER_PATH)
    if spec is None or spec.loader is None:
        _fail("cannot construct frozen R1 runner loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_r1 = _load_frozen_r1_runner()
_base = _r1._base


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
        "r1_attempt_receipt": {
            "expected_content": dict(R1_ATTEMPT_CONTENT),
            "path": str(R1_ATTEMPT_PATH),
            "sha256": R1_ATTEMPT_SHA256,
        },
        "r1_execution_card": {
            "path": str(R1_CARD_PATH.relative_to(REPO_ROOT)),
            "sha256": R1_CARD_SHA256,
        },
        "r1_terminal_record": {
            "canonical_json_plus_lf_sha256": R1_FAILURE_SHA256,
            "record": dict(R1_FAILURE_RECORD),
            "serialization": "CANONICAL_UTF8_SORTED_COMPACT_JSON_PLUS_LF",
        },
        "runtime_preconditions": {
            "base_namespace_root_absent": True,
            "r1_attempt_receipt_present_and_frozen": True,
            "r1_namespace_root_absent": True,
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
        _fail("R2 overlay top-level keys differ")
    if (
        overlay.get("schema") != R2_CONFIG_SCHEMA
        or overlay.get("state") != "MATERIALIZED_VALIDATED_EXECUTION_AUTHORIZED"
        or overlay.get("authority") != R2_AUTHORITY
        or overlay.get("fix_contract") != R2_FIX_CONTRACT
    ):
        _fail("R2 overlay identity/authority/fix differs")
    if overlay.get("base_config") != {
        "path": str(BASE_CONFIG_PATH.relative_to(REPO_ROOT)),
        "sha256": BASE_CONFIG_SHA256,
    }:
        _fail("R2 overlay base config pin differs")
    expected_override = dict(_expected_execution())
    for key in ("gpu_count", "logical_device", "python"):
        expected_override.pop(key)
    if overlay.get("execution_override") != expected_override:
        _fail("R2 overlay execution identity differs")
    predecessor = overlay.get("predecessor_revision")
    if predecessor != {
        "r1_attempt_receipt": {
            "path": str(R1_ATTEMPT_PATH),
            "sha256": R1_ATTEMPT_SHA256,
        },
        "r1_execution_card": {
            "path": str(R1_CARD_PATH.relative_to(REPO_ROOT)),
            "sha256": R1_CARD_SHA256,
        },
        "r1_terminal_record_sha256": R1_FAILURE_SHA256,
    }:
        _fail("R2 overlay predecessor revision differs")


def _verify_r1_attempt_receipt() -> None:
    directory_metadata = R1_ATTEMPT_DIR.lstat()
    receipt_metadata = R1_ATTEMPT_PATH.lstat()
    if (
        stat.S_ISLNK(directory_metadata.st_mode)
        or not stat.S_ISDIR(directory_metadata.st_mode)
        or stat.S_IMODE(directory_metadata.st_mode) != 0o555
        or stat.S_ISLNK(receipt_metadata.st_mode)
        or not stat.S_ISREG(receipt_metadata.st_mode)
        or stat.S_IMODE(receipt_metadata.st_mode) != 0o444
    ):
        _fail("R1 attempt receipt type/mode differs")
    if _sha(R1_ATTEMPT_PATH) != R1_ATTEMPT_SHA256:
        _fail("R1 attempt receipt SHA differs")
    if _base._strict_json(R1_ATTEMPT_PATH) != R1_ATTEMPT_CONTENT:
        _fail("R1 attempt receipt content differs")


def _verify_predecessor_runtime_state() -> None:
    _verify_r1_attempt_receipt()
    _base._require_paths_absent(
        (BASE_NAMESPACE, BASE_NAMESPACE_LEAF, BASE_ROOT),
        label="failed base AV3 namespace/root",
    )
    _base._require_paths_absent(
        (R1_NAMESPACE, R1_NAMESPACE_LEAF, R1_ROOT),
        label="failed AV3-R1 namespace/root",
    )


def _verify_card(
    card: Mapping[str, object], *, authorized_card_sha256: str
) -> dict[str, str]:
    if not _base._is_sha256(authorized_card_sha256):
        _fail("--execution-card-sha256 must be one lowercase SHA256")
    if _sha(R2_CARD_PATH) != authorized_card_sha256:
        _fail("R2 execution-card full SHA differs from supplied binding")
    if frozenset(card) != EXPECTED_CARD_KEYS:
        _fail("R2 execution-card top-level keys differ")
    if (
        card.get("schema") != R2_CARD_SCHEMA
        or card.get("state") != R2_CARD_STATE
        or card.get("card_id") != R2_CARD_ID
        or card.get("authority") != R2_AUTHORITY
        or card.get("fix_contract") != R2_FIX_CONTRACT
        or card.get("execution") != _expected_execution()
        or card.get("predecessor_revision") != _expected_predecessor_revision()
    ):
        _fail("R2 execution-card identity/authority/predecessor differs")
    if card.get("identity") != {
        "architecture_id": _base.EXPECTED_ARCHITECTURE_ID,
        "canonical_ssh_alias": "H200",
        "canonical_worktree": str(REPO_ROOT),
        "created_on": "2026-08-05",
        "self_path": str(R2_CARD_PATH.relative_to(REPO_ROOT)),
        "self_pin_policy": "OUT_OF_BAND_FULL_SHA256_AT_INVOCATION",
    }:
        _fail("R2 execution-card identity differs")
    if card.get("inherited_contract") != {
        "architecture_data_training_metric_verdict_artifact_resource_unchanged": True,
        "base_and_r1_failures_are_not_architecture_evidence": True,
        "r1_card_transitive_closure_revalidated_at_preflight": True,
        "source_only_override": "AV3_NATIVE_16X16_DISJOINT_VALIDATOR",
    }:
        _fail("R2 inherited contract differs")
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
        _fail("R2 root lifecycle differs")
    forbidden = card.get("forbidden")
    if not isinstance(forbidden, list) or frozenset(forbidden) != EXPECTED_FORBIDDEN:
        _fail("R2 forbidden capability set differs")

    inherited_pins = card.get("inherited_pins")
    if not isinstance(inherited_pins, Mapping) or frozenset(
        inherited_pins
    ) != frozenset(INHERITED_PIN_SPECS):
        _fail("R2 inherited pin roles differ")
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
        if inherited_pins[role] != expected_pin:
            _fail(f"R2 inherited pin differs: {role}")
        verified[f"inherited:{role}"] = _base._verify_pin(
            inherited_pins[role],
            expected_path=path,
            label=f"R2 inherited.{role}",
        )

    materialized = card.get("materialized_pins")
    if not isinstance(materialized, Mapping) or frozenset(materialized) != frozenset(
        R2_MATERIALIZED_PATHS
    ):
        _fail("R2 materialized pin roles differ")
    for role, path in R2_MATERIALIZED_PATHS.items():
        verified[f"materialized:{role}"] = _base._verify_pin(
            materialized[role],
            expected_path=path,
            label=f"R2 materialized.{role}",
        )
    if verified["materialized:config"] != R2_CONFIG_SHA256:
        _fail("R2 config SHA differs")
    if verified["materialized:source"] != R2_SOURCE_SHA256:
        _fail("R2 source SHA differs")
    verified["materialized:data_manifest"] = verified["inherited:data_manifest"]
    return verified


def _revalidate_r1_transitive_closure() -> dict[str, str]:
    r1_card = _base._strict_json(R1_CARD_PATH)
    r1_overlay = _base._strict_json(R1_CONFIG_PATH)
    r1_verified = _r1._verify_card(
        r1_card,
        authorized_card_sha256=R1_CARD_SHA256,
    )
    verified = {
        f"r1_closure:{role}": digest for role, digest in r1_verified.items()
    }
    _r1._verify_overlay(r1_overlay)
    base_card = _base._strict_json(_r1.BASE_CARD_PATH)
    planning = _base._strict_json(PLANNING_CARD_PATH, require_canonical=False)
    _base._verify_planning_predecessors(planning)
    _base._verify_execution_predecessor_equality(base_card, planning)
    direct = base_card.get("predecessor_pins")
    transitive = base_card.get("transitive_av2_predecessor_pins")
    if not isinstance(direct, Mapping) or not isinstance(transitive, Mapping):
        _fail("base AV3 predecessor maps are absent")
    for role, pin in direct.items():
        verified[f"base_predecessor:{role}"] = _base._verify_pin(
            pin,
            expected_path=None,
            label=f"R2 base predecessor.{role}",
            frozen=False,
        )
    av2_card = _base._strict_json(
        _base.AV2_BASE_EXECUTION_CARD_PATH,
        require_canonical=False,
    )
    if transitive != av2_card.get("predecessor_pins"):
        _fail("base AV3 transitive AV2 predecessor map differs")
    for role, pin in transitive.items():
        verified[f"base_transitive_av2:{role}"] = _base._verify_pin(
            pin,
            expected_path=None,
            label=f"R2 base transitive AV2.{role}",
            frozen=False,
        )
    return verified


def _static_preflight(authorized_card_sha256: str) -> dict[str, object]:
    card = _base._strict_json(R2_CARD_PATH)
    overlay = _base._strict_json(R2_CONFIG_PATH)
    base_config = _base._strict_json(BASE_CONFIG_PATH)
    manifest = _base._strict_json(DATA_MANIFEST_PATH)
    verified = _verify_card(card, authorized_card_sha256=authorized_card_sha256)
    _verify_overlay(overlay)
    verified.update(_revalidate_r1_transitive_closure())
    _verify_predecessor_runtime_state()
    if _base._sha_bytes(_base._canonical(R1_FAILURE_RECORD)) != R1_FAILURE_SHA256:
        _fail("R1 terminal record canonical SHA differs")
    selected_files = _base._verify_manifest_structure(manifest)
    _base._require_paths_absent(
        (ATTEMPT_DIR,),
        label="AV3-R2 one-shot attempt receipt",
    )
    _base._require_paths_absent(
        (EXPECTED_NAMESPACE, EXPECTED_NAMESPACE_LEAF, EXPECTED_ROOT),
        label="future AV3-R2 namespace/root",
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
        label="AV3-R2 one-shot attempt receipt",
    )
    _base._verify_ancestor_chain(ATTEMPT_DIR, label="R2 attempt path")
    receipt_path = ATTEMPT_DIR / "ATTEMPT.json"
    payload = _base._canonical(
        {
            "authorized_execution_card_sha256": authorized_card_sha256,
            "authority_statement_sha256": R2_AUTHORITY["statement_sha256"],
            "automatic_rerun_allowed": False,
            "nonce": EXPECTED_NONCE,
            "root": str(EXPECTED_ROOT),
            "schema": "cach.cach_a4.av3_r2.attempt_receipt.v1",
            "state": "R2_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY",
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
                raise OSError("short write while publishing R2 attempt receipt")
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
        raise RuntimeError("R2 attempt authority was not consumed")
    receipt_path = ATTEMPT_DIR / "ATTEMPT.json"
    if (
        stat.S_IMODE(ATTEMPT_DIR.lstat().st_mode) != 0o555
        or stat.S_IMODE(receipt_path.lstat().st_mode) != 0o444
    ):
        raise RuntimeError("R2 attempt receipt is not frozen")
    receipt = _base._strict_json(receipt_path)
    if (
        receipt.get("authorized_execution_card_sha256")
        != _AUTHORIZED_CARD_SHA256
        or receipt.get("nonce") != EXPECTED_NONCE
        or receipt.get("root") != str(EXPECTED_ROOT)
        or receipt.get("state") != "R2_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY"
    ):
        raise RuntimeError("R2 attempt receipt content differs")


def _reject_parallel_predecessor_execution() -> None:
    snapshot = _base._related_process_snapshot()
    rows = snapshot.get("matching_rows")
    if not isinstance(rows, list):
        raise RuntimeError("R2 process observation is malformed")
    names = {
        "run_cach_a4_av3_paired_reduced_scale.py",
        "run_cach_a4_av3_paired_reduced_scale_r1.py",
    }
    conflicting = [row for row in rows if any(name in str(row) for name in names)]
    if conflicting:
        raise RuntimeError(f"parallel predecessor AV3 execution observed: {conflicting}")


_original_static_preflight = _static_preflight


def _static_preflight_with_attempt(authorized_card_sha256: str) -> dict[str, object]:
    result = _original_static_preflight(authorized_card_sha256)
    if _CONSUME_ATTEMPT:
        _reject_parallel_predecessor_execution()
        _verify_predecessor_runtime_state()
        _consume_attempt(authorized_card_sha256)
    return result


_BaseRootLifecycle = _r1._BaseRootLifecycle


class R2RootLifecycle(_BaseRootLifecycle):
    def create(self) -> None:
        _verify_predecessor_runtime_state()
        _verify_consumed_attempt()
        _reject_parallel_predecessor_execution()
        super().create()


_base.CARD_SCHEMA = R2_CARD_SCHEMA
_base.CARD_STATE = R2_CARD_STATE
_base.CONFIG_PATH = R2_CONFIG_PATH
_base.DATA_MANIFEST_PATH = DATA_MANIFEST_PATH
_base.EXECUTION_CARD_PATH = R2_CARD_PATH
_base.EXPECTED_ENV = dict(EXPECTED_ENV)
_base.EXPECTED_NAMESPACE = EXPECTED_NAMESPACE
_base.EXPECTED_NAMESPACE_LEAF = EXPECTED_NAMESPACE_LEAF
_base.EXPECTED_NONCE = EXPECTED_NONCE
_base.EXPECTED_ROOT = EXPECTED_ROOT
_base.RUNNER_PATH = R2_RUNNER_PATH
_base.SOURCE_PATH = R2_SOURCE_PATH
_base.TEST_PATH = R2_TEST_PATH
_base.RootLifecycle = R2RootLifecycle
_base._observe_resources = _r1._observe_resources
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
