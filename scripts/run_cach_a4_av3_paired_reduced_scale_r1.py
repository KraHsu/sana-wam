#!/usr/bin/env python3
"""Fresh-root AV3-R1 runner with one mechanical observation-loop fix.

This additive launcher keeps the frozen AV3 architecture, data, training,
metric, and verdict contracts.  It loads the frozen base launcher using only
the standard library, replaces the invalid adjacent-pair expression, and
binds execution to the AV3-R1 overlay/card and fresh immutable root.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time
from types import ModuleType
from typing import Any, NoReturn


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
BASE_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale.py"
BASE_CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale.yaml"
BASE_TEST_PATH = REPO_ROOT / "tests/test_cach_a4_av3_paired_reduced_scale.py"
BASE_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_EXECUTION_RUN_CARD.json"
)
DATA_MANIFEST_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_DATA_SELECTION.json"
)
SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale.py"
DECISION_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_PAIRED_REDUCED_SCALE_DECISION.md"
)
PLANNING_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_PAIRED_REDUCED_SCALE_RUN_CARD.json"
)

R1_CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale_r1.yaml"
R1_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale_r1.py"
R1_TEST_PATH = REPO_ROOT / "tests/test_cach_a4_av3_paired_reduced_scale_r1.py"
R1_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_EXECUTION_R1_RUN_CARD.json"
)

R1_CARD_SCHEMA = "cach.architecture_validation.cach_a4_av3_execution_r1_run_card.v1"
R1_CARD_STATE = "FROZEN_AV3_R1_EXECUTION_AUTHORIZED"
R1_CARD_ID = "CACH-A4-AV3-PAIRED-REDUCED-SCALE-SEED0-EXECUTION-R1-v1"
R1_CONFIG_SCHEMA = "cach.cach_a4.av3_paired_reduced_scale_r1.runtime_overlay.v1"
R1_AUTHORITY = {
    "interpretation": "MATERIALIZE_AND_EXECUTE_AV3_R1_MINIMAL_OBSERVATION_FIX_FRESH_ROOT_SINGLE_GPU",
    "statement": "授权 AV3-R1",
    "statement_sha256": "fef23c2741999ead4f5f6bce68d87e68f6d75d0f503bca11fb2e348d2ecda574",
    "statement_utf8_bytes": 13,
    "unlisted_capability_default": "DENIED",
}
R1_FIX_CONTRACT = {
    "new_expression": "zip(offsets[:-1], offsets[1:], strict=True)",
    "old_expression": "zip(offsets, offsets[1:], strict=True)",
    "only_semantic_delta": "RESOURCE_OBSERVATION_ADJACENT_PAIR_LENGTH_ALIGNMENT",
}

EXPECTED_GPU_INDEX = 0
EXPECTED_GPU_UUID = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
EXPECTED_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_av3_paired_reduced_scale_r1"
)
EXPECTED_NAMESPACE_LEAF = EXPECTED_NAMESPACE / "06f5d09127f8"
EXPECTED_NONCE = "5d0a6205440f43921351b32a5d5588a8"
EXPECTED_ROOT = EXPECTED_NAMESPACE_LEAF / f"cach-a4-av3-r1-{EXPECTED_NONCE}"
BASE_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_av3_paired_reduced_scale"
)
BASE_NAMESPACE_LEAF = BASE_NAMESPACE / "06f5d09127f8"
BASE_ROOT = BASE_NAMESPACE_LEAF / (
    "cach-a4-av3-97a25a9e1d614340697fbee523e19559"
)
ATTEMPT_DIR = EXPECTED_NAMESPACE.parent / (
    f".cach-a4-av3-r1-attempt-{EXPECTED_NONCE}"
)
EXPECTED_ENV = {
    "CUDA_VISIBLE_DEVICES": EXPECTED_GPU_UUID,
    "FUSED_GDN_PRECISION": "0",
    "GDN_DISABLE_COMPILE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "TRITON_CACHE_DIR": str(EXPECTED_ROOT / "triton_cache"),
}

BASE_PIN_SPECS = {
    "base_config": (
        BASE_CONFIG_PATH,
        "8aa04f561d2ba47dcddc91d7504151978caa6f1072a0992d88a1b37100f1564d",
    ),
    "base_execution_card": (
        BASE_CARD_PATH,
        "72124b63e278490ff39262aaa5bcf5a71121a533df6f55aff7147d98b11e06f9",
    ),
    "base_runner": (
        BASE_RUNNER_PATH,
        "7f20aee0662cbabf00e2138506ddafb2fbfe75965cdc32ec973e74813b8b6c27",
    ),
    "base_test": (
        BASE_TEST_PATH,
        "4aa1b4f6a5f2ef249e4954df154f1488b536a39aafaee2d196e8897ba258c577",
    ),
    "data_manifest": (
        DATA_MANIFEST_PATH,
        "fecd1a9fb2f4371011d9c3c86469fed8f24f7f6aa5b3f1be9811fd447fb2ab95",
    ),
    "decision": (
        DECISION_PATH,
        "a0a211915704b3230747021a566e06ff47478b20f3acb2681973e6ac9e3ebd30",
    ),
    "planning_card": (
        PLANNING_CARD_PATH,
        "33566c3291e49e4a87b4f83807d80255be67a5083615358c92bde924046bc82f",
    ),
    "source": (
        SOURCE_PATH,
        "bafdb5475e1d7dddfba6d93e68ab2b67bf3960dba613da09ac26c48c8627bb6c",
    ),
}
R1_CONFIG_SHA256 = "335a559dd79cfaccea36df32212de10c2b604427701e2609321c17ebfc259a90"
R1_MATERIALIZED_PATHS = {
    "config": R1_CONFIG_PATH,
    "runner": R1_RUNNER_PATH,
    "test": R1_TEST_PATH,
}
EXPECTED_CARD_KEYS = frozenset(
    {
        "authority",
        "base_contract",
        "card_id",
        "execution",
        "failure_predecessor",
        "fix_contract",
        "forbidden",
        "identity",
        "inherited_pins",
        "materialized_pins",
        "root_lifecycle_contract",
        "schema",
        "state",
    }
)
EXPECTED_FORBIDDEN = frozenset(
    {
        "AUTOMATIC_RERUN",
        "AV4",
        "CHECKPOINT_LOAD_OR_SAVE",
        "DEPLOY",
        "FORMAL_EVALUATION_OR_ADMISSION",
        "FULL_2B",
        "GLOBAL_STAGE3",
        "OTHER_GPU_OR_MULTI_GPU",
        "POST_FREEZE_MUTATION",
        "PREDECESSOR_MUTATION",
        "ROOT_REUSE",
        "BASE_AV3_AND_R1_DUAL_EXECUTION",
        "REVIEW_TOKEN_OR_CLAIM_OPERATION",
    }
)
EXPECTED_FAILURE = {
    "authorized_base_card_sha256": BASE_PIN_SPECS["base_execution_card"][1],
    "base_execution_authority": {
        "statement": "授权执行 AV3 72124b63…e06f9",
        "statement_sha256": "76d4a7c4145ef7efeaad1588fca30aef4879ad4217c578ae8d8d83baa9c613e4",
        "statement_utf8_bytes": 33,
    },
    "base_namespace_absent_after_failure": True,
    "terminal_record": {
        "automatic_rerun_allowed": False,
        "error": "zip() argument 2 is shorter than argument 1",
        "error_type": "ValueError",
        "phase": "resource_observation",
        "root_created": False,
        "schema": "cach.cach_a4.av3_paired_reduced_scale.terminal_result.v1",
        "state": "ENV_BLOCKED",
    },
    "terminal_record_serialization": "CANONICAL_UTF8_SORTED_COMPACT_JSON_PLUS_LF",
    "terminal_record_sha256": "bea9929347b2bf61acceeb663eed5f787d348f7e9656e22cb420f43aaa355767",
}


class R1ContractError(RuntimeError):
    """Raised before any R1 data, GPU, model, optimizer, or root capability."""


def _fail(message: str) -> NoReturn:
    raise R1ContractError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_base_runner() -> ModuleType:
    metadata = BASE_RUNNER_PATH.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail("frozen base runner is not a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        _fail("frozen base runner mode differs from 0444")
    expected = BASE_PIN_SPECS["base_runner"][1]
    if _sha(BASE_RUNNER_PATH) != expected:
        _fail("frozen base runner SHA differs")
    name = "_cach_a4_av3_paired_reduced_scale_frozen_base_for_r1"
    spec = importlib.util.spec_from_file_location(name, BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        _fail("cannot construct frozen base runner loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_frozen_base_runner()


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


def _verify_overlay(overlay: Mapping[str, object]) -> None:
    if frozenset(overlay) != {
        "authority",
        "base_config",
        "execution_override",
        "fix_contract",
        "schema",
        "state",
    }:
        _fail("R1 overlay top-level keys differ")
    if (
        overlay.get("schema") != R1_CONFIG_SCHEMA
        or overlay.get("state") != "MATERIALIZED_EXECUTION_AUTHORIZED"
        or overlay.get("authority") != R1_AUTHORITY
        or overlay.get("fix_contract") != R1_FIX_CONTRACT
    ):
        _fail("R1 overlay identity/authority/fix differs")
    base_path, base_sha = BASE_PIN_SPECS["base_config"]
    if overlay.get("base_config") != {
        "path": str(base_path.relative_to(REPO_ROOT)),
        "sha256": base_sha,
    }:
        _fail("R1 overlay base config pin differs")
    expected_override = dict(_expected_execution())
    for key in ("gpu_count", "logical_device", "python"):
        expected_override.pop(key)
    if overlay.get("execution_override") != expected_override:
        _fail("R1 overlay fresh execution identity differs")


def _verify_card(
    card: Mapping[str, object], *, authorized_card_sha256: str
) -> dict[str, str]:
    if not _base._is_sha256(authorized_card_sha256):
        _fail("--execution-card-sha256 must be one lowercase SHA256")
    if _sha(R1_CARD_PATH) != authorized_card_sha256:
        _fail("R1 execution-card full SHA differs from supplied binding")
    if frozenset(card) != EXPECTED_CARD_KEYS:
        _fail("R1 execution-card top-level keys differ")
    if (
        card.get("schema") != R1_CARD_SCHEMA
        or card.get("state") != R1_CARD_STATE
        or card.get("card_id") != R1_CARD_ID
        or card.get("authority") != R1_AUTHORITY
        or card.get("fix_contract") != R1_FIX_CONTRACT
        or card.get("failure_predecessor") != EXPECTED_FAILURE
        or card.get("execution") != _expected_execution()
    ):
        _fail("R1 execution-card identity/authority/execution differs")
    identity = card.get("identity")
    if identity != {
        "architecture_id": _base.EXPECTED_ARCHITECTURE_ID,
        "canonical_ssh_alias": "H200",
        "canonical_worktree": str(REPO_ROOT),
        "created_on": "2026-08-05",
        "self_path": str(R1_CARD_PATH.relative_to(REPO_ROOT)),
        "self_pin_policy": "OUT_OF_BAND_FULL_SHA256_AT_INVOCATION",
    }:
        _fail("R1 execution-card identity differs")
    if card.get("base_contract") != {
        "architecture_data_training_metric_verdict_unchanged": True,
        "automatic_rerun_of_base_attempt": False,
        "base_execution_card_sha256": BASE_PIN_SPECS["base_execution_card"][1],
        "base_root_created": False,
        "fresh_revision": True,
        "only_overrides": [
            "LAUNCHER_ADJACENT_PAIR_LENGTH_ALIGNMENT",
            "FRESH_R1_RUNTIME_IDENTITY",
        ],
    }:
        _fail("R1 base-contract inheritance differs")
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
        _fail("R1 root lifecycle differs")
    forbidden = card.get("forbidden")
    if not isinstance(forbidden, list) or frozenset(forbidden) != EXPECTED_FORBIDDEN:
        _fail("R1 forbidden capability set differs")

    inherited = card.get("inherited_pins")
    if not isinstance(inherited, Mapping) or frozenset(inherited) != frozenset(
        BASE_PIN_SPECS
    ):
        _fail("R1 inherited pin roles differ")
    verified: dict[str, str] = {}
    for role, (path, expected_sha) in BASE_PIN_SPECS.items():
        expected_pin = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": expected_sha,
        }
        if inherited[role] != expected_pin:
            _fail(f"R1 inherited pin differs: {role}")
        verified[f"inherited:{role}"] = _base._verify_pin(
            inherited[role],
            expected_path=path,
            label=f"inherited_pins.{role}",
        )

    materialized = card.get("materialized_pins")
    if not isinstance(materialized, Mapping) or frozenset(materialized) != frozenset(
        R1_MATERIALIZED_PATHS
    ):
        _fail("R1 materialized pin roles differ")
    for role, path in R1_MATERIALIZED_PATHS.items():
        verified[f"materialized:{role}"] = _base._verify_pin(
            materialized[role],
            expected_path=path,
            label=f"materialized_pins.{role}",
        )
    if verified["materialized:config"] != R1_CONFIG_SHA256:
        _fail("R1 config SHA differs from frozen launcher constant")
    verified["materialized:data_manifest"] = verified["inherited:data_manifest"]
    verified["materialized:source"] = verified["inherited:source"]
    return verified


def _static_preflight(authorized_card_sha256: str) -> dict[str, object]:
    card = _base._strict_json(R1_CARD_PATH)
    overlay = _base._strict_json(R1_CONFIG_PATH)
    base_card = _base._strict_json(BASE_CARD_PATH)
    base_config = _base._strict_json(BASE_CONFIG_PATH)
    manifest = _base._strict_json(DATA_MANIFEST_PATH)
    verified = _verify_card(card, authorized_card_sha256=authorized_card_sha256)
    _verify_overlay(overlay)
    if base_card.get("materialized_pins", {}).get("config") != {
        "path": str(BASE_CONFIG_PATH.relative_to(REPO_ROOT)),
        "sha256": BASE_PIN_SPECS["base_config"][1],
    }:
        _fail("frozen base card/config relationship differs")
    if base_card.get("training_contract", {}).get("optimizer_steps_per_arm") != 1000:
        _fail("frozen base card training steps differ")
    planning = _base._strict_json(PLANNING_CARD_PATH, require_canonical=False)
    _base._verify_planning_predecessors(planning)
    _base._verify_execution_predecessor_equality(base_card, planning)
    direct = base_card.get("predecessor_pins")
    transitive = base_card.get("transitive_av2_predecessor_pins")
    if not isinstance(direct, Mapping) or not isinstance(transitive, Mapping):
        _fail("frozen base predecessor maps are absent")
    for role, pin in direct.items():
        verified[f"base_predecessor:{role}"] = _base._verify_pin(
            pin,
            expected_path=None,
            label=f"base_predecessor.{role}",
            frozen=False,
        )
    av2_base_card = _base._strict_json(
        _base.AV2_BASE_EXECUTION_CARD_PATH,
        require_canonical=False,
    )
    if transitive != av2_base_card.get("predecessor_pins"):
        _fail("frozen base transitive AV2 predecessor map differs")
    for role, pin in transitive.items():
        verified[f"base_transitive_av2:{role}"] = _base._verify_pin(
            pin,
            expected_path=None,
            label=f"base_transitive_av2.{role}",
            frozen=False,
        )
    selected_files = _base._verify_manifest_structure(manifest)
    _base._require_paths_absent(
        (BASE_NAMESPACE, BASE_NAMESPACE_LEAF, BASE_ROOT),
        label="failed base AV3 namespace/root",
    )
    _base._require_paths_absent(
        (ATTEMPT_DIR,),
        label="AV3-R1 one-shot attempt receipt",
    )
    _base._require_paths_absent(
        (EXPECTED_NAMESPACE, EXPECTED_NAMESPACE_LEAF, EXPECTED_ROOT),
        label="future AV3-R1 namespace/root",
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


def _observe_resources(
    *,
    seconds: int = 120,
    interval: int = 5,
    monotonic: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
    snapshot: Callable[[], Mapping[str, object]] | None = None,
) -> Mapping[str, object]:
    """Observe the fixed GPU with the sole R1 adjacent-pair length fix."""

    if seconds < 0 or interval <= 0:
        raise ValueError("resource observation duration/interval is malformed")
    monotonic = time.monotonic if monotonic is None else monotonic
    sleeper = time.sleep if sleeper is None else sleeper
    snapshot = _base._gpu_snapshot if snapshot is None else snapshot
    filesystem = dict(_base._filesystem_snapshot())
    related_processes = dict(_base._related_process_snapshot())
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
        (
            right - left
            for left, right in zip(offsets[:-1], offsets[1:], strict=True)
        ),
        default=0.0,
    )
    if len(samples) < minimum_samples or maximum_gap > float(interval + 2):
        raise RuntimeError("resource observation sample count/gap differs")
    return {
        "schema": _base.RESOURCE_SCHEMA,
        "duration_required_seconds": seconds,
        "duration_observed_seconds": elapsed,
        "gpu_samples": samples,
        "maximum_sample_gap_seconds": maximum_gap,
        "sample_count": len(samples),
        "filesystem": filesystem,
        "related_processes": related_processes,
    }


_CONSUME_ATTEMPT = False
_AUTHORIZED_CARD_SHA256: str | None = None


def _consume_attempt(authorized_card_sha256: str) -> None:
    """Atomically consume the one allowed full R1 invocation."""

    global _AUTHORIZED_CARD_SHA256
    _base._require_paths_absent(
        (ATTEMPT_DIR,),
        label="AV3-R1 one-shot attempt receipt",
    )
    _base._verify_ancestor_chain(ATTEMPT_DIR, label="R1 attempt path")
    receipt_path = ATTEMPT_DIR / "ATTEMPT.json"
    payload = _base._canonical(
        {
            "authorized_execution_card_sha256": authorized_card_sha256,
            "authority_statement_sha256": R1_AUTHORITY["statement_sha256"],
            "automatic_rerun_allowed": False,
            "nonce": EXPECTED_NONCE,
            "root": str(EXPECTED_ROOT),
            "schema": "cach.cach_a4.av3_r1.attempt_receipt.v1",
            "state": "R1_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY",
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
                raise OSError("short write while publishing R1 attempt receipt")
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
        raise RuntimeError("R1 attempt authority was not consumed")
    metadata = ATTEMPT_DIR.lstat()
    receipt_path = ATTEMPT_DIR / "ATTEMPT.json"
    receipt_metadata = receipt_path.lstat()
    if (
        stat.S_IMODE(metadata.st_mode) != 0o555
        or stat.S_IMODE(receipt_metadata.st_mode) != 0o444
    ):
        raise RuntimeError("R1 attempt receipt is not frozen")
    receipt = _base._strict_json(receipt_path)
    if (
        receipt.get("authorized_execution_card_sha256")
        != _AUTHORIZED_CARD_SHA256
        or receipt.get("nonce") != EXPECTED_NONCE
        or receipt.get("root") != str(EXPECTED_ROOT)
        or receipt.get("state") != "R1_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY"
    ):
        raise RuntimeError("R1 attempt receipt content differs")


_original_static_preflight = _static_preflight


def _reject_parallel_base_execution() -> None:
    snapshot = _base._related_process_snapshot()
    rows = snapshot.get("matching_rows")
    if not isinstance(rows, list):
        raise RuntimeError("R1 process observation is malformed")
    base_name = "run_cach_a4_av3_paired_reduced_scale.py"
    conflicting = [row for row in rows if base_name in str(row)]
    if conflicting:
        raise RuntimeError(f"parallel base AV3 execution observed: {conflicting}")


def _static_preflight_with_attempt(authorized_card_sha256: str) -> dict[str, object]:
    result = _original_static_preflight(authorized_card_sha256)
    if _CONSUME_ATTEMPT:
        _reject_parallel_base_execution()
        _consume_attempt(authorized_card_sha256)
    return result


_BaseRootLifecycle = _base.RootLifecycle


class R1RootLifecycle(_BaseRootLifecycle):
    def create(self) -> None:
        # Revalidate the failed base choice immediately before the fresh R1
        # exclusive-create boundary, after data preload and 120 s observation.
        _base._require_paths_absent(
            (BASE_NAMESPACE, BASE_NAMESPACE_LEAF, BASE_ROOT),
            label="failed base AV3 namespace/root at R1 create boundary",
        )
        _verify_consumed_attempt()
        super().create()


_base.CARD_SCHEMA = R1_CARD_SCHEMA
_base.CARD_STATE = R1_CARD_STATE
_base.CONFIG_PATH = R1_CONFIG_PATH
_base.EXECUTION_CARD_PATH = R1_CARD_PATH
_base.EXPECTED_ENV = dict(EXPECTED_ENV)
_base.EXPECTED_NAMESPACE = EXPECTED_NAMESPACE
_base.EXPECTED_NAMESPACE_LEAF = EXPECTED_NAMESPACE_LEAF
_base.EXPECTED_NONCE = EXPECTED_NONCE
_base.EXPECTED_ROOT = EXPECTED_ROOT
_base.RUNNER_PATH = R1_RUNNER_PATH
_base.TEST_PATH = R1_TEST_PATH
_base.RootLifecycle = R1RootLifecycle
_base._observe_resources = _observe_resources
_base._static_preflight = _static_preflight_with_attempt


def main(argv: list[str] | None = None) -> int:
    global _CONSUME_ATTEMPT
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--execution-card-sha256", required=True)
    parsed = parser.parse_args(arguments)
    _CONSUME_ATTEMPT = not parsed.static_preflight
    return int(_base.main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
