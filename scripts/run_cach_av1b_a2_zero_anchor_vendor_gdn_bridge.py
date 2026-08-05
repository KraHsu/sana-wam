#!/usr/bin/env python3
"""Fail-closed future-execution harness for the frozen CACH-A2 screen.

This launcher deliberately has a standard-library-only pre-root phase.  It
does not import Torch, Triton, the vendor tree, or the CACH-A2 bridge until the
card, source, predecessor, execution-authority, environment, and dynamically
selected idle-GPU bindings have all been verified and the unique run root has
been created.  A created root always becomes one immutable terminal root.

CACH-A2 inherits the predecessor review300 token as read-only consumed
evidence.  It creates, derives, resets, replaces, or consumes no token and
publishes no intermediate verdict: RESULT.json is the direct final verdict.
"""

from __future__ import annotations

import argparse
import ast
import grp
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import pwd
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WORKTREE = Path("/home/zch/workspace/sana-wam")
EXPECTED_HANDOFF = Path("/home/zch/workspace/sana-afcc-handoff")

EXPECTED_CARD = (
    "docs/cach_sana_wam/architecture_validation/cach_a2/"
    "CACH_A2_ZERO_ANCHOR_RUN_CARD.json"
)
EXPECTED_CARD_SHA256 = (
    "9d3c3973234df8a9593b254161ca07bf72f343c4475d9985e2901153b82a9713"
)
EXPECTED_DECISION = (
    "docs/cach_sana_wam/architecture_validation/cach_a2/"
    "CACH_A2_ZERO_ANCHOR_ARCHITECTURE_DECISION.md"
)
EXPECTED_DECISION_SHA256 = (
    "280e17b6e9a7433e64b20ceed587e998ab0ef1322fe7cc7467ee6a56f9589363"
)
EXPECTED_PLAN = (
    "docs/cach_sana_wam/global_stage3/post_phase_c_planning/"
    "ARCHITECTURE_VALIDATION_FIRST_PLAN_20260803_v1.md"
)
EXPECTED_PLAN_SHA256 = (
    "06f5d09127f8bc2a945cdbdc095054f40b1fe8d9757a90eccbdf0604ea89e079"
)

EXPECTED_BRIDGE = "src/sana_wam/model/cach_av1b_a2_zero_anchor_vendor_gdn_bridge.py"
EXPECTED_CONFIG = "configs/experiments/cach_av1b_a2_zero_anchor_vendor_gdn_bridge.yaml"
EXPECTED_RUNNER = "scripts/run_cach_av1b_a2_zero_anchor_vendor_gdn_bridge.py"
EXPECTED_TEST = "tests/test_cach_av1b_a2_zero_anchor_vendor_gdn_bridge.py"
SOURCE_QUARTET = (EXPECTED_BRIDGE, EXPECTED_CONFIG, EXPECTED_RUNNER, EXPECTED_TEST)

EXPECTED_NONCE = "ed3e6dabfbdb836136cb9d24bf9e7a78"
EXPECTED_ROOT = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a2/06f5d09127f8/"
    "cach-a2-zero-anchor-ed3e6dabfbdb836136cb9d24bf9e7a78"
)
EXPECTED_NAMESPACE_BASE = Path("/DATA/share/sana_cach_wam_nonformal_screens")
EXPECTED_NAMESPACE_PATHS = (
    EXPECTED_NAMESPACE_BASE / "cach_a2",
    EXPECTED_NAMESPACE_BASE / "cach_a2" / "06f5d09127f8",
)

EXPECTED_REPO_HEADS = {
    "sana_wam": "605f1c134b4c983ff80f8489c4bc8847036329e2",
    "third_party_sana": "16b9cec673e3335724ba2d8db25de7f9ed229292",
    "sana_afcc_handoff": "9586486f2a9f5172d57b325e32093a3e018d34c0",
}
EXPECTED_PREDECESSOR_MANIFEST_SHA256 = (
    "866f9e02a1fe1f9be61614e072872ced241f7d49e73a1fd4aae32b52a81db6bb"
)

AUTHORITY_ENVELOPE_SCHEMA = "cach.cach_a2.runtime_execution_authority.v1"
CONFIG_SCHEMA = "cach.cach_a2.zero_anchor_vendor_gdn_bridge.config.v1"
SCREEN_SCHEMA = "cach.cach_a2.zero_anchor_vendor_gdn_bridge.screen.v1"
NAMESPACE_MODE = 0o2775
TERMINAL_DIR_MODE = 0o555
TERMINAL_FILE_MODE = 0o444
EXPECTED_NAMESPACE_OWNER = "zch"
EXPECTED_NAMESPACE_GROUP = "sharegrp"

INVALID_TERMINAL_STATES = frozenset(
    {
        "AUTH_BLOCKED",
        "ENV_BLOCKED",
        "HARNESS_REJECTED",
        "INVALID_RUN",
        "IMPLEMENTATION_INVALID",
        "NUMERICAL_INVALID",
    }
)
VALID_TERMINAL_VERDICTS = frozenset(
    {
        "OPERATOR_GO",
        "OPERATOR_STOP",
        "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED",
    }
)
REQUIRED_ROOT_ARTIFACTS = (
    "RUN_CONTEXT.json",
    "PROGRESS_EVENTS.json",
    "THETA0_MANIFESTS.json",
    "LOSS_TRACE.json",
    "RAW_METRICS.json",
    "FINAL_PARAMETER_DIGESTS.json",
    "JIT_INVENTORY.json",
    "RAW_EVIDENCE.json",
    "RESULT.json",
    "FREEZE_RECEIPT.json",
)


class PreflightError(RuntimeError):
    """Typed failure which must occur before root creation."""

    def __init__(self, state: str, token: str, detail: str):
        if state not in {"AUTH_BLOCKED", "ENV_BLOCKED", "HARNESS_REJECTED"}:
            raise ValueError(f"invalid preflight state: {state}")
        self.state = state
        self.token = token
        self.detail = detail
        super().__init__(f"{state}:{token}:{detail}")


class RunInterrupted(RuntimeError):
    """Raised by installed signal handlers after root creation."""


def _fail(state: str, token: str, detail: str) -> NoReturn:
    raise PreflightError(state, token, detail)


def _require(condition: bool, state: str, token: str, detail: str) -> None:
    if not condition:
        _fail(state, token, detail)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not duplicate-free UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} top level must be an object")
    return value


def _duplicate_rejecting_json(path: Path) -> dict[str, Any]:
    return _parse_json_bytes(path.read_bytes(), label=str(path))


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}")


def _regular_repo_file(relative: str) -> Path:
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        _fail("AUTH_BLOCKED", "UNSAFE_REPO_PATH", relative)
    path = REPO_ROOT / relative
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        _fail("AUTH_BLOCKED", "PIN_PATH_MISSING", relative)
    _require(stat.S_ISREG(metadata.st_mode), "AUTH_BLOCKED", "PIN_NOT_REGULAR", relative)
    _require(not path.is_symlink(), "AUTH_BLOCKED", "PIN_SYMLINK", relative)
    return path


def _verify_repo_pin(relative: str, expected_sha256: str) -> None:
    path = _regular_repo_file(relative)
    observed = _sha256_file(path)
    _require(
        observed == expected_sha256,
        "AUTH_BLOCKED",
        "SOURCE_PIN_MISMATCH",
        f"{relative}: expected {expected_sha256}, observed {observed}",
    )


def _verify_absolute_pin(path_text: str, expected_sha256: str) -> dict[str, Any]:
    path = Path(path_text)
    _require(path.is_absolute(), "AUTH_BLOCKED", "ABSOLUTE_PIN_REQUIRED", path_text)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        _fail("AUTH_BLOCKED", "PREDECESSOR_ARTIFACT_MISSING", path_text)
    _require(
        stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
        "AUTH_BLOCKED",
        "PREDECESSOR_ARTIFACT_NOT_REGULAR",
        path_text,
    )
    observed = _sha256_file(path)
    _require(
        observed == expected_sha256,
        "AUTH_BLOCKED",
        "PREDECESSOR_ARTIFACT_PIN_MISMATCH",
        f"{path_text}: expected {expected_sha256}, observed {observed}",
    )
    _require(
        stat.S_IMODE(metadata.st_mode) == TERMINAL_FILE_MODE,
        "AUTH_BLOCKED",
        "PREDECESSOR_ARTIFACT_NOT_FROZEN",
        path_text,
    )
    return {"path": path_text, "sha256": observed, "mode": "0444"}


def _git_head(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail("ENV_BLOCKED", "GIT_HEAD_UNAVAILABLE", f"{path}: {exc}")
    return completed.stdout.strip()


def _predecessor_pin_manifest(review_card: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the review300 card's frozen transitive pin manifest."""

    identity = review_card["identity"]
    return {
        "source_pins": {
            label: value
            for label, value in review_card["source_pins"].items()
            if label != "future_review300_generated_files"
        },
        "predecessor_cards": {
            "av0": identity["predecessor_av0_card"],
            "av1b": identity["predecessor_av1b_card"],
        },
        "predecessor_roots": {
            "av1": identity["predecessor_av1_root"],
            "av1b": identity["predecessor_av1b_root"],
            "av1b_nonce": identity["predecessor_av1b_nonce"],
        },
        "predecessor_artifacts": {
            "av1": identity["predecessor_av1_artifacts"],
            "av1b": identity["predecessor_av1b_artifacts"],
        },
        "latest_review_token_state_head": review_card["review_token"][
            "latest_token_state_head"
        ],
    }


def _read_authority_envelope() -> tuple[dict[str, Any], str]:
    limit = 1024 * 1024
    payload = sys.stdin.buffer.read(limit + 1)
    _require(
        0 < len(payload) <= limit,
        "AUTH_BLOCKED",
        "AUTHORITY_ENVELOPE_SIZE",
        f"stdin bytes={len(payload)}",
    )
    try:
        value = _parse_json_bytes(payload, label="authority stdin")
    except ValueError as exc:
        _fail("AUTH_BLOCKED", "AUTHORITY_ENVELOPE_JSON", str(exc))
    _require(
        payload == _canonical_json_bytes(value),
        "AUTH_BLOCKED",
        "AUTHORITY_ENVELOPE_NOT_CANONICAL",
        "stdin must be sorted compact UTF-8 JSON with one trailing LF",
    )
    return value, _sha256_bytes(payload)


def _validate_card(card_argument: Path) -> dict[str, Any]:
    expected_path = REPO_ROOT / EXPECTED_CARD
    _require(
        card_argument == expected_path,
        "AUTH_BLOCKED",
        "CARD_PATH_MISMATCH",
        f"expected {expected_path}, observed {card_argument}",
    )
    _verify_repo_pin(EXPECTED_CARD, EXPECTED_CARD_SHA256)
    card = _duplicate_rejecting_json(expected_path)
    _require(
        card.get("schema") == "cach.architecture_validation.cach_a2_zero_anchor_run_card.v1",
        "AUTH_BLOCKED",
        "CARD_SCHEMA_MISMATCH",
        repr(card.get("schema")),
    )
    _require(
        card.get("card_id") == "cach-a2-zero-anchored-bias-free-v1",
        "AUTH_BLOCKED",
        "CARD_ID_MISMATCH",
        repr(card.get("card_id")),
    )
    _require(
        card.get("card_state") == "FROZEN_CARD_ONLY_SOURCE_AND_EXECUTION_NOT_AUTHORIZED",
        "AUTH_BLOCKED",
        "CARD_STATE_MISMATCH",
        repr(card.get("card_state")),
    )
    identity = card.get("identity")
    _require(isinstance(identity, Mapping), "AUTH_BLOCKED", "CARD_IDENTITY_INVALID", "not mapping")
    assert isinstance(identity, Mapping)
    _require(
        identity.get("self_path") == EXPECTED_CARD,
        "AUTH_BLOCKED",
        "CARD_SELF_PATH_MISMATCH",
        repr(identity.get("self_path")),
    )
    _require(
        identity.get("architecture_decision")
        == {"path": EXPECTED_DECISION, "sha256": EXPECTED_DECISION_SHA256},
        "AUTH_BLOCKED",
        "DECISION_BINDING_MISMATCH",
        repr(identity.get("architecture_decision")),
    )
    _require(
        identity.get("governing_plan")
        == {"path": EXPECTED_PLAN, "sha256": EXPECTED_PLAN_SHA256},
        "AUTH_BLOCKED",
        "PLAN_BINDING_MISMATCH",
        repr(identity.get("governing_plan")),
    )
    _verify_repo_pin(EXPECTED_DECISION, EXPECTED_DECISION_SHA256)
    _verify_repo_pin(EXPECTED_PLAN, EXPECTED_PLAN_SHA256)
    _require(
        identity.get("repository_heads") == EXPECTED_REPO_HEADS,
        "AUTH_BLOCKED",
        "REPOSITORY_HEAD_CARD_BINDING_MISMATCH",
        repr(identity.get("repository_heads")),
    )
    runtime = card.get("planned_runtime")
    _require(isinstance(runtime, Mapping), "AUTH_BLOCKED", "CARD_RUNTIME_INVALID", "not mapping")
    assert isinstance(runtime, Mapping)
    _require(runtime.get("run_nonce") == EXPECTED_NONCE, "AUTH_BLOCKED", "NONCE_MISMATCH", repr(runtime.get("run_nonce")))
    _require(runtime.get("resolved_root") == str(EXPECTED_ROOT), "AUTH_BLOCKED", "ROOT_MISMATCH", repr(runtime.get("resolved_root")))
    future_gpu = runtime.get("future_gpu")
    _require(isinstance(future_gpu, Mapping), "AUTH_BLOCKED", "GPU_CARD_CONTRACT_INVALID", "not mapping")
    assert isinstance(future_gpu, Mapping)
    _require(
        future_gpu.get("physical_index") is None and future_gpu.get("uuid") is None,
        "AUTH_BLOCKED",
        "CARD_HARDCODES_GPU",
        repr(future_gpu),
    )
    closure = card.get("transitive_predecessor_closure")
    _require(isinstance(closure, Mapping), "AUTH_BLOCKED", "CLOSURE_INVALID", "not mapping")
    assert isinstance(closure, Mapping)
    _require(
        closure.get("canonical_manifest_sha256") == EXPECTED_PREDECESSOR_MANIFEST_SHA256,
        "AUTH_BLOCKED",
        "CLOSURE_SHA_MISMATCH",
        repr(closure.get("canonical_manifest_sha256")),
    )
    return card


def _verify_card_source_pins(card: Mapping[str, Any]) -> dict[str, str]:
    pins = card.get("source_pins")
    _require(isinstance(pins, Mapping), "AUTH_BLOCKED", "SOURCE_PINS_INVALID", "not mapping")
    assert isinstance(pins, Mapping)
    observed: dict[str, str] = {}
    for label, pin in pins.items():
        _require(isinstance(pin, Mapping), "AUTH_BLOCKED", "SOURCE_PIN_INVALID", str(label))
        assert isinstance(pin, Mapping)
        path = pin.get("path")
        digest = pin.get("sha256")
        _require(isinstance(path, str) and isinstance(digest, str), "AUTH_BLOCKED", "SOURCE_PIN_INVALID", str(label))
        _verify_repo_pin(path, digest)
        observed[path] = digest
    return observed


def _verify_predecessors(card: Mapping[str, Any]) -> dict[str, Any]:
    identity = card["identity"]
    for key in ("predecessor_av0_card",):
        pin = identity[key]
        _verify_repo_pin(pin["path"], pin["sha256"])
    av1b = identity["predecessor_av1b"]
    review = identity["predecessor_review300"]
    for predecessor in (av1b, review):
        _verify_repo_pin(predecessor["card"]["path"], predecessor["card"]["sha256"])
    absolute_pins: list[Mapping[str, str]] = [
        av1b["result"],
        av1b["freeze_receipt"],
        review["result"],
        review["raw_evidence"],
        review["raw_metrics"],
        review["loss_trace"],
        review["theta0_manifests"],
        review["run_context"],
        review["freeze_receipt"],
        review["consumed_claim"],
    ]
    verified = [_verify_absolute_pin(pin["path"], pin["sha256"]) for pin in absolute_pins]
    for root_text in (av1b["root"], review["root"]):
        root = Path(root_text)
        try:
            metadata = root.lstat()
        except FileNotFoundError:
            _fail("AUTH_BLOCKED", "PREDECESSOR_ROOT_MISSING", root_text)
        _require(
            stat.S_ISDIR(metadata.st_mode)
            and not root.is_symlink()
            and stat.S_IMODE(metadata.st_mode) == TERMINAL_DIR_MODE,
            "AUTH_BLOCKED",
            "PREDECESSOR_ROOT_NOT_FROZEN",
            root_text,
        )
    consumed = _duplicate_rejecting_json(Path(review["consumed_claim"]["path"]))
    _require(
        consumed.get("token_id") == review["consumed_claim"]["token_id"]
        and consumed.get("token_state") == "CONSUMED",
        "AUTH_BLOCKED",
        "PREDECESSOR_CONSUMED_EVIDENCE_INVALID",
        "pinned review300 consumed evidence differs",
    )
    review_card = _duplicate_rejecting_json(REPO_ROOT / review["card"]["path"])
    manifest = _predecessor_pin_manifest(review_card)
    manifest_sha = _sha256_bytes(_canonical_json_bytes(manifest))
    _require(
        manifest_sha == EXPECTED_PREDECESSOR_MANIFEST_SHA256
        and review.get("predecessor_pin_manifest_sha256") == manifest_sha,
        "AUTH_BLOCKED",
        "PREDECESSOR_CLOSURE_MISMATCH",
        manifest_sha,
    )
    return {
        "verified_absolute_pins": verified,
        "consumed_predecessor_token_id": consumed["token_id"],
        "consumed_predecessor_state": "CONSUMED",
        "canonical_manifest": manifest,
        "canonical_manifest_sha256": manifest_sha,
    }


def _verify_frozen_quartet(authority: Mapping[str, Any]) -> dict[str, str]:
    quartet = authority.get("source_quartet_sha256")
    _require(isinstance(quartet, Mapping), "AUTH_BLOCKED", "QUARTET_BINDING_INVALID", "not mapping")
    assert isinstance(quartet, Mapping)
    _require(
        frozenset(quartet) == frozenset(SOURCE_QUARTET),
        "AUTH_BLOCKED",
        "QUARTET_PATH_SET_MISMATCH",
        repr(sorted(quartet)),
    )
    result: dict[str, str] = {}
    for relative in SOURCE_QUARTET:
        expected = quartet[relative]
        _require(
            isinstance(expected, str) and len(expected) == 64,
            "AUTH_BLOCKED",
            "QUARTET_SHA_INVALID",
            relative,
        )
        path = _regular_repo_file(relative)
        metadata = path.lstat()
        _require(
            stat.S_IMODE(metadata.st_mode) == TERMINAL_FILE_MODE,
            "AUTH_BLOCKED",
            "QUARTET_NOT_FROZEN_0444",
            relative,
        )
        observed = _sha256_file(path)
        _require(
            observed == expected,
            "AUTH_BLOCKED",
            "QUARTET_SHA_MISMATCH",
            f"{relative}: expected {expected}, observed {observed}",
        )
        result[relative] = observed
    return result


def _source_authority_binding(config: Mapping[str, Any]) -> Mapping[str, Any]:
    authority = config.get("authority")
    if not isinstance(authority, Mapping):
        raise ValueError("config.authority must be a mapping")
    return {
        "statement": authority.get("source_statement"),
        "statement_sha256": authority.get("source_statement_sha256"),
    }


def _validate_authority(
    envelope: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[int, str, str]:
    required = frozenset(
        {
            "schema",
            "operation",
            "card",
            "source_authority",
            "execution_authority",
            "successor_plan_deviation_accepted",
            "no_new_review_token_accepted",
            "source_quartet_sha256",
            "root",
            "nonce",
            "gpu",
            "capabilities",
        }
    )
    try:
        _require_exact_keys(envelope, required, label="authority envelope")
    except ValueError as exc:
        _fail("AUTH_BLOCKED", "AUTHORITY_KEYS", str(exc))
    _require(envelope.get("schema") == AUTHORITY_ENVELOPE_SCHEMA, "AUTH_BLOCKED", "AUTHORITY_SCHEMA", repr(envelope.get("schema")))
    _require(envelope.get("operation") == "EXECUTE_CACH_A2_ONCE", "AUTH_BLOCKED", "AUTHORITY_OPERATION", repr(envelope.get("operation")))
    _require(
        envelope.get("card") == {"path": EXPECTED_CARD, "sha256": EXPECTED_CARD_SHA256},
        "AUTH_BLOCKED",
        "AUTHORITY_CARD_BINDING",
        repr(envelope.get("card")),
    )
    _require(envelope.get("root") == str(EXPECTED_ROOT), "AUTH_BLOCKED", "AUTHORITY_ROOT_BINDING", repr(envelope.get("root")))
    _require(envelope.get("nonce") == EXPECTED_NONCE, "AUTH_BLOCKED", "AUTHORITY_NONCE_BINDING", repr(envelope.get("nonce")))
    _require(envelope.get("successor_plan_deviation_accepted") is True, "AUTH_BLOCKED", "PLAN_DEVIATION_NOT_ACCEPTED", "must be literal true")
    _require(envelope.get("no_new_review_token_accepted") is True, "AUTH_BLOCKED", "NO_NEW_TOKEN_NOT_ACCEPTED", "must be literal true")

    source = envelope.get("source_authority")
    _require(isinstance(source, Mapping), "AUTH_BLOCKED", "SOURCE_AUTHORITY_INVALID", "not mapping")
    assert isinstance(source, Mapping)
    try:
        _require_exact_keys(source, frozenset({"statement", "statement_sha256"}), label="source authority")
    except ValueError as exc:
        _fail("AUTH_BLOCKED", "SOURCE_AUTHORITY_KEYS", str(exc))
    statement = source.get("statement")
    digest = source.get("statement_sha256")
    _require(isinstance(statement, str) and isinstance(digest, str), "AUTH_BLOCKED", "SOURCE_AUTHORITY_TYPES", "statement/hash")
    _require(_sha256_bytes(statement.encode("utf-8")) == digest, "AUTH_BLOCKED", "SOURCE_AUTHORITY_HASH", str(digest))
    try:
        frozen_source = _source_authority_binding(config)
    except ValueError as exc:
        _fail("AUTH_BLOCKED", "CONFIG_SOURCE_AUTHORITY", str(exc))
    frozen_statement = frozen_source.get("statement")
    frozen_digest = frozen_source.get("statement_sha256", frozen_source.get("sha256"))
    _require(
        statement == frozen_statement and digest == frozen_digest,
        "AUTH_BLOCKED",
        "SOURCE_AUTHORITY_CONFIG_BINDING",
        "stdin source authority differs from frozen config",
    )

    execution = envelope.get("execution_authority")
    _require(isinstance(execution, Mapping), "AUTH_BLOCKED", "EXECUTION_AUTHORITY_INVALID", "not mapping")
    assert isinstance(execution, Mapping)
    try:
        _require_exact_keys(execution, frozenset({"statement", "statement_sha256"}), label="execution authority")
    except ValueError as exc:
        _fail("AUTH_BLOCKED", "EXECUTION_AUTHORITY_KEYS", str(exc))
    execution_statement = execution.get("statement")
    execution_sha = execution.get("statement_sha256")
    _require(
        isinstance(execution_statement, str)
        and len(execution_statement.encode("utf-8")) >= 64
        and isinstance(execution_sha, str),
        "AUTH_BLOCKED",
        "EXECUTION_AUTHORITY_TYPES",
        "statement must be explicit and non-empty",
    )
    _require(
        _sha256_bytes(execution_statement.encode("utf-8")) == execution_sha,
        "AUTH_BLOCKED",
        "EXECUTION_AUTHORITY_HASH",
        str(execution_sha),
    )

    gpu = envelope.get("gpu")
    _require(isinstance(gpu, Mapping), "AUTH_BLOCKED", "GPU_AUTHORITY_INVALID", "not mapping")
    assert isinstance(gpu, Mapping)
    try:
        _require_exact_keys(
            gpu,
            frozenset({"physical_index", "uuid", "selection_policy"}),
            label="GPU authority",
        )
    except ValueError as exc:
        _fail("AUTH_BLOCKED", "GPU_AUTHORITY_KEYS", str(exc))
    physical_index = gpu.get("physical_index")
    gpu_uuid = gpu.get("uuid")
    _require(type(physical_index) is int and physical_index >= 0, "AUTH_BLOCKED", "GPU_INDEX_INVALID", repr(physical_index))
    _require(isinstance(gpu_uuid, str) and gpu_uuid.startswith("GPU-") and len(gpu_uuid) >= 20, "AUTH_BLOCKED", "GPU_UUID_INVALID", repr(gpu_uuid))
    _require(
        gpu.get("selection_policy")
        == "DYNAMIC_EXACT_IDLE_PHYSICAL_INDEX_AND_UUID_AFTER_READ_ONLY_AVAILABILITY_CHECK",
        "AUTH_BLOCKED",
        "GPU_SELECTION_POLICY_INVALID",
        repr(gpu.get("selection_policy")),
    )
    quartet_binding = envelope.get("source_quartet_sha256")
    _require(
        isinstance(quartet_binding, Mapping)
        and frozenset(quartet_binding) == frozenset(SOURCE_QUARTET),
        "AUTH_BLOCKED",
        "EXECUTION_QUARTET_BINDING_INVALID",
        repr(quartet_binding),
    )
    assert isinstance(quartet_binding, Mapping)
    required_execution_statement_fragments = (
        EXPECTED_CARD_SHA256,
        EXPECTED_DECISION_SHA256,
        EXPECTED_PREDECESSOR_MANIFEST_SHA256,
        str(digest),
        str(EXPECTED_ROOT),
        EXPECTED_NONCE,
        str(physical_index),
        gpu_uuid,
        *(
            str(quartet_binding[relative])
            for relative in SOURCE_QUARTET
        ),
    )
    for required_fragment in required_execution_statement_fragments:
        _require(
            required_fragment in execution_statement,
            "AUTH_BLOCKED",
            "EXECUTION_STATEMENT_INCOMPLETE",
            f"missing literal binding {required_fragment}",
        )

    expected_capabilities = {
        "single_gpu_cuda": True,
        "triton_jit": True,
        "vendor_autograd_kernel": True,
        "model_construct_execute": True,
        "forward_backward": True,
        "local_jvp": True,
        "synthetic_adamw_300_steps_per_arm": True,
        "real_data": False,
        "checkpoint_load": False,
        "checkpoint_save": False,
        "review_token_consumption": False,
        "full_2b": False,
        "formal_evaluation": False,
        "av2": False,
        "global_stage3": False,
    }
    _require(
        envelope.get("capabilities") == expected_capabilities,
        "AUTH_BLOCKED",
        "CAPABILITY_SET_MISMATCH",
        repr(envelope.get("capabilities")),
    )
    return physical_index, gpu_uuid, str(execution_sha)


def _manifest_from_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = config.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("config.identity must map")
    manifest = identity.get("direct_import_dependency_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("config identity lacks direct_import_dependency_manifest")
    return manifest


def _validate_config(
    config: Mapping[str, Any], card: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> None:
    _require(config.get("schema") == CONFIG_SCHEMA, "AUTH_BLOCKED", "CONFIG_SCHEMA", repr(config.get("schema")))
    identity = config.get("identity")
    _require(isinstance(identity, Mapping), "AUTH_BLOCKED", "CONFIG_IDENTITY", "not mapping")
    assert isinstance(identity, Mapping)
    _require(
        identity.get("run_card")
        == {"path": EXPECTED_CARD, "sha256": EXPECTED_CARD_SHA256},
        "AUTH_BLOCKED",
        "CONFIG_CARD_BINDING",
        repr(identity.get("run_card")),
    )
    _require(
        identity.get("architecture_decision")
        == {"path": EXPECTED_DECISION, "sha256": EXPECTED_DECISION_SHA256},
        "AUTH_BLOCKED",
        "CONFIG_DECISION_BINDING",
        repr(identity.get("architecture_decision")),
    )
    _require(
        identity.get("governing_plan")
        == {"path": EXPECTED_PLAN, "sha256": EXPECTED_PLAN_SHA256},
        "AUTH_BLOCKED",
        "CONFIG_PLAN_BINDING",
        repr(identity.get("governing_plan")),
    )
    _require(
        identity.get("repository_heads") == EXPECTED_REPO_HEADS,
        "AUTH_BLOCKED",
        "CONFIG_REPOSITORY_HEADS",
        repr(identity.get("repository_heads")),
    )
    planned_runtime = identity.get("planned_runtime")
    _require(
        isinstance(planned_runtime, Mapping)
        and planned_runtime.get("resolved_root") == str(EXPECTED_ROOT)
        and planned_runtime.get("run_nonce") == EXPECTED_NONCE
        and planned_runtime.get("run_namespace_parents")
        == [str(path) for path in EXPECTED_NAMESPACE_PATHS],
        "AUTH_BLOCKED",
        "CONFIG_PLANNED_RUNTIME_BINDING",
        repr(planned_runtime),
    )
    _require(
        identity.get("source_pins") == card.get("source_pins"),
        "AUTH_BLOCKED",
        "CONFIG_SOURCE_PINS_DIFFER",
        "config identity source pins differ from frozen card",
    )
    expected_generated_paths = {
        "bridge": {
            "path": EXPECTED_BRIDGE,
            "sha256": "SOURCE_MATERIALIZATION_BIND_FULL_SHA256_OUT_OF_BAND",
        },
        "config": {
            "path": EXPECTED_CONFIG,
            "sha256": "SELF_OUT_OF_BAND_FULL_LITERAL_SHA256",
        },
        "runner": {
            "path": EXPECTED_RUNNER,
            "sha256": "SOURCE_MATERIALIZATION_BIND_FULL_SHA256_OUT_OF_BAND",
        },
        "test": {
            "path": EXPECTED_TEST,
            "sha256": "SOURCE_MATERIALIZATION_BIND_FULL_SHA256_OUT_OF_BAND",
        },
    }
    _require(
        identity.get("generated_paths") == expected_generated_paths,
        "AUTH_BLOCKED",
        "CONFIG_GENERATED_PATHS_BINDING",
        repr(identity.get("generated_paths")),
    )
    config_predecessor = identity.get("predecessor")
    card_review = card["identity"]["predecessor_review300"]
    config_review = (
        config_predecessor.get("review300")
        if isinstance(config_predecessor, Mapping)
        else None
    )
    review_common_keys = (
        "card",
        "root",
        "nonce",
        "typed_verdict",
        "result",
        "raw_evidence",
        "raw_metrics",
        "loss_trace",
        "theta0_manifests",
        "run_context",
        "freeze_receipt",
        "predecessor_pin_manifest_sha256",
    )
    _require(
        isinstance(config_predecessor, Mapping)
        and config_predecessor.get("av0_card") == card["identity"]["predecessor_av0_card"]
        and config_predecessor.get("av1b") == card["identity"]["predecessor_av1b"]
        and isinstance(config_review, Mapping)
        and all(config_review.get(name) == card_review.get(name) for name in review_common_keys)
        and config_review.get("consumed_claim", {}).get("path")
        == card_review["consumed_claim"]["path"]
        and config_review.get("consumed_claim", {}).get("sha256")
        == card_review["consumed_claim"]["sha256"]
        and config_review.get("consumed_claim", {}).get("token_id")
        == card_review["consumed_claim"]["token_id"]
        and config_review.get("consumed_claim", {}).get("top_level_token_state")
        == card_review["consumed_claim"]["token_state"],
        "AUTH_BLOCKED",
        "CONFIG_PREDECESSOR_BINDING",
        "config predecessor closure differs from card",
    )

    try:
        direct_manifest = _manifest_from_config(config)
    except ValueError as exc:
        _fail("AUTH_BLOCKED", "DIRECT_DEPENDENCY_MANIFEST", str(exc))
    _require(
        direct_manifest.get("schema")
        == "cach.cach_a2.direct_import_dependency_manifest.v1",
        "AUTH_BLOCKED",
        "DIRECT_DEPENDENCY_MANIFEST_SCHEMA",
        repr(direct_manifest.get("schema")),
    )
    entries = direct_manifest.get("entries")
    _require(
        isinstance(entries, list) and len(entries) == 8,
        "AUTH_BLOCKED",
        "DIRECT_DEPENDENCY_MANIFEST_ENTRIES",
        repr(entries),
    )
    assert isinstance(entries, list)
    direct_paths: list[str] = []
    direct_modules: list[str] = []
    for entry in entries:
        _require(isinstance(entry, Mapping), "AUTH_BLOCKED", "DIRECT_DEPENDENCY_ENTRY", repr(entry))
        assert isinstance(entry, Mapping)
        path = entry.get("path")
        digest = entry.get("sha256")
        label = entry.get("source_pin_label")
        _require(
            isinstance(path, str)
            and isinstance(digest, str)
            and isinstance(label, str)
            and card["source_pins"].get(label)
            == {"path": path, "sha256": digest},
            "AUTH_BLOCKED",
            "DIRECT_DEPENDENCY_ENTRY_PIN",
            repr(entry),
        )
        _verify_repo_pin(path, digest)
        direct_paths.append(path)
        module = entry.get("module")
        _require(
            isinstance(module, str),
            "AUTH_BLOCKED",
            "DIRECT_DEPENDENCY_MODULE",
            repr(entry),
        )
        direct_modules.append(module)
    _require(
        direct_modules
        == sorted(direct_modules, key=lambda value: value.encode("utf-8"))
        == direct_manifest.get("repo_local_module_names_sorted")
        and len(set(direct_paths)) == len(direct_paths)
        and len(set(direct_modules)) == len(direct_modules),
        "AUTH_BLOCKED",
        "DIRECT_DEPENDENCY_ORDER_OR_DUPLICATE",
        repr({"modules": direct_modules, "paths": direct_paths}),
    )
    try:
        bridge_tree = ast.parse(
            _regular_repo_file(EXPECTED_BRIDGE).read_text(encoding="utf-8"),
            filename=EXPECTED_BRIDGE,
        )
    except (SyntaxError, UnicodeError) as exc:
        _fail("AUTH_BLOCKED", "BRIDGE_AST_INVALID", str(exc))
    observed_repo_imports = sorted(
        {
            node.module
            for node in ast.walk(bridge_tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (
                node.module.startswith("sana_wam.")
                or node.module.startswith("diffusion.")
            )
        },
        key=lambda value: value.encode("utf-8"),
    )
    _require(
        observed_repo_imports == direct_modules,
        "AUTH_BLOCKED",
        "UNMANIFESTED_REPO_LOCAL_IMPORT",
        repr({"manifest": direct_modules, "bridge_ast": observed_repo_imports}),
    )
    _require(
        direct_manifest.get("vendor_transitive_support_pins")
        == ["vendor_base_gdn", "vendor_autograd_kernel", "vendor_chunk_utility"]
        and direct_manifest.get("external_runtime_imports_without_repo_file_pin")
        == ["torch", "triton"]
        and direct_manifest.get("unmanifested_repo_local_import")
        == "IMPLEMENTATION_INVALID",
        "AUTH_BLOCKED",
        "DIRECT_DEPENDENCY_MANIFEST_POLICY",
        repr(direct_manifest),
    )
    _require(
        predecessor["canonical_manifest_sha256"]
        == EXPECTED_PREDECESSOR_MANIFEST_SHA256,
        "AUTH_BLOCKED",
        "TRANSITIVE_PREDECESSOR_CLOSURE",
        repr(predecessor["canonical_manifest_sha256"]),
    )

    runtime = config.get("runtime")
    optimizer = config.get("optimizer")
    budget = config.get("budget")
    thresholds = config.get("thresholds")
    _require(all(isinstance(value, Mapping) for value in (runtime, optimizer, budget, thresholds)), "AUTH_BLOCKED", "CONFIG_RUNTIME_SECTIONS", "runtime/optimizer/budget/thresholds")
    assert isinstance(runtime, Mapping) and isinstance(optimizer, Mapping)
    assert isinstance(budget, Mapping) and isinstance(thresholds, Mapping)
    gpu_binding = runtime.get("gpu_binding")
    _require(isinstance(gpu_binding, Mapping), "AUTH_BLOCKED", "CONFIG_GPU_BINDING", "not mapping")
    assert isinstance(gpu_binding, Mapping)
    _require(
        gpu_binding.get("physical_index") is None
        and gpu_binding.get("uuid") is None
        and gpu_binding.get("count") == 1,
        "AUTH_BLOCKED",
        "CONFIG_HARDCODES_GPU",
        repr(gpu_binding),
    )
    _require(
        optimizer.get("optimizer_steps_per_arm") == 300
        and optimizer.get("max_optimizer_steps_total") == 600,
        "AUTH_BLOCKED",
        "OPTIMIZER_BUDGET_MISMATCH",
        repr(optimizer),
    )
    card_experiment = card["experiment_contract"]
    card_budget = card_experiment["budget"]
    _require(
        all(budget.get(name) == value for name, value in card_budget.items())
        and budget.get("max_optimizer_steps_total") == 600
        and budget.get("exact_optimizer_steps_per_arm") == 300
        and budget.get("automatic_budget_extension") is False,
        "AUTH_BLOCKED",
        "BUDGET_CONTRACT_MISMATCH",
        "config budget differs from card",
    )
    card_thresholds = card["threshold_contract"]
    _require(
        thresholds.get("operator_go") == card_thresholds["operator_go_all_required"]
        and thresholds.get("delta_live_common_mode_blocked")
        == card_thresholds["delta_live_common_mode_blocked_all_required"]
        and thresholds.get("operator_strong_stop")
        == card_thresholds["operator_strong_stop_all_required"]
        and thresholds.get("post_result_threshold_or_formula_change") is False,
        "AUTH_BLOCKED",
        "THRESHOLD_CONTRACT_MISMATCH",
        "config thresholds differ from card",
    )
    policy = config.get("predecessor_token_policy")
    _require(isinstance(policy, Mapping), "AUTH_BLOCKED", "PREDECESSOR_TOKEN_POLICY", "not mapping")
    assert isinstance(policy, Mapping)
    _require(
        policy.get("predecessor_top_level_state") == "CONSUMED"
        and policy.get("a2_review_or_budget_extension_token") is None
        and policy.get("create_token_or_ledger_namespace") is False
        and policy.get("derive_reset_replace_or_consume_token") is False
        and policy.get("atomic_claim_or_claim_recovery") is False,
        "AUTH_BLOCKED",
        "TOKEN_POLICY_MISMATCH",
        repr(policy),
    )


def _run_command(command: Sequence[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail("ENV_BLOCKED", "COMMAND_FAILED", f"{list(command)!r}: {exc}")


def _gpu_snapshot(physical_index: int, gpu_uuid: str, label: str) -> dict[str, Any]:
    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu,compute_mode,pstate,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for raw in query.stdout.splitlines():
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.split(",")]
        _require(len(fields) == 8, "ENV_BLOCKED", "NVIDIA_SMI_GPU_ROW", raw)
        try:
            row = {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_used_mib": int(fields[3]),
                "utilization_gpu_percent": int(fields[4]),
                "compute_mode": fields[5],
                "pstate": fields[6],
                "driver_version": fields[7],
            }
        except ValueError as exc:
            _fail("ENV_BLOCKED", "NVIDIA_SMI_GPU_PARSE", f"{raw}: {exc}")
        rows.append(row)
        if row["physical_index"] == physical_index:
            selected = row
    _require(selected is not None, "AUTH_BLOCKED", "GPU_INDEX_NOT_PRESENT", str(physical_index))
    assert selected is not None
    _require(selected["uuid"] == gpu_uuid, "AUTH_BLOCKED", "GPU_INDEX_UUID_MISMATCH", repr(selected))

    processes_result = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes: list[dict[str, Any]] = []
    for raw in processes_result.stdout.splitlines():
        if not raw.strip() or "No running processes" in raw:
            continue
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 3 or fields[0] != gpu_uuid:
            continue
        try:
            processes.append(
                {"gpu_uuid": fields[0], "pid": int(fields[1]), "used_memory_mib": int(fields[2])}
            )
        except ValueError as exc:
            _fail("ENV_BLOCKED", "NVIDIA_SMI_PROCESS_PARSE", f"{raw}: {exc}")
    pmon_result = _run_command(
        ["nvidia-smi", "pmon", "-c", "1", "-i", str(physical_index)]
    )
    pmon_active_pids: list[int] = []
    for raw in pmon_result.stdout.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        _require(
            len(fields) >= 2 and fields[0] == str(physical_index),
            "ENV_BLOCKED",
            "NVIDIA_SMI_PMON_ROW",
            raw,
        )
        if fields[1] != "-":
            try:
                pmon_active_pids.append(int(fields[1]))
            except ValueError as exc:
                _fail("ENV_BLOCKED", "NVIDIA_SMI_PMON_PID", f"{raw}: {exc}")
    idle = (
        not processes
        and not pmon_active_pids
        and selected["utilization_gpu_percent"] == 0
        and selected["memory_used_mib"] == 0
        and selected["compute_mode"] in {"Default", "Exclusive_Process"}
    )
    _require(
        selected["name"] == "NVIDIA H200",
        "ENV_BLOCKED",
        "AUTHORIZED_GPU_NOT_H200",
        repr(selected["name"]),
    )
    _require(idle, "ENV_BLOCKED", "AUTHORIZED_GPU_NOT_IDLE", repr({"gpu": selected, "processes": processes}))
    return {
        "label": label,
        "monotonic_seconds": time.monotonic(),
        "selected_gpu": selected,
        "selected_compute_processes": processes,
        "pmon_active_pids": pmon_active_pids,
        "pmon_stdout": pmon_result.stdout,
        "all_gpu_rows": rows,
        "idle": True,
    }


def _directory_owner_group(path: Path) -> tuple[str, str]:
    metadata = path.lstat()
    return pwd.getpwuid(metadata.st_uid).pw_name, grp.getgrgid(metadata.st_gid).gr_name


def _validate_directory_chain(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        _require(stat.S_ISDIR(metadata.st_mode) and not current.is_symlink(), "ENV_BLOCKED", "ANCESTOR_NOT_REAL_DIRECTORY", str(current))


def _validate_precreate_environment(
    root_argument: Path, physical_index: int, gpu_uuid: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    _require(REPO_ROOT == EXPECTED_WORKTREE, "ENV_BLOCKED", "WORKTREE_PATH_MISMATCH", f"{REPO_ROOT}")
    _require(root_argument == EXPECTED_ROOT, "AUTH_BLOCKED", "CLI_ROOT_MISMATCH", f"{root_argument}")
    _require(platform.node().split(".")[0] == "H200", "ENV_BLOCKED", "HOST_MISMATCH", platform.node())
    heads = {
        "sana_wam": _git_head(REPO_ROOT),
        "third_party_sana": _git_head(REPO_ROOT / "third_party" / "Sana"),
        "sana_afcc_handoff": _git_head(EXPECTED_HANDOFF),
    }
    _require(heads == EXPECTED_REPO_HEADS, "AUTH_BLOCKED", "REPOSITORY_HEAD_MISMATCH", repr(heads))
    _validate_directory_chain(EXPECTED_ROOT)
    try:
        base_meta = EXPECTED_NAMESPACE_BASE.lstat()
    except FileNotFoundError:
        _fail("ENV_BLOCKED", "NAMESPACE_BASE_MISSING", str(EXPECTED_NAMESPACE_BASE))
    _require(stat.S_ISDIR(base_meta.st_mode) and not EXPECTED_NAMESPACE_BASE.is_symlink(), "ENV_BLOCKED", "NAMESPACE_BASE_INVALID", str(EXPECTED_NAMESPACE_BASE))
    _require(_directory_owner_group(EXPECTED_NAMESPACE_BASE) == (EXPECTED_NAMESPACE_OWNER, EXPECTED_NAMESPACE_GROUP), "ENV_BLOCKED", "NAMESPACE_BASE_OWNER", repr(_directory_owner_group(EXPECTED_NAMESPACE_BASE)))
    _require(
        stat.S_IMODE(base_meta.st_mode) == NAMESPACE_MODE,
        "ENV_BLOCKED",
        "NAMESPACE_BASE_MODE",
        f"{stat.S_IMODE(base_meta.st_mode):04o}",
    )
    for path in (*EXPECTED_NAMESPACE_PATHS, EXPECTED_ROOT):
        _require(not path.exists() and not path.is_symlink(), "ENV_BLOCKED", "EXCLUSIVE_PATH_EXISTS", str(path))
    usage = shutil.disk_usage(EXPECTED_NAMESPACE_BASE)
    budget = config["budget"]
    _require(usage.free >= int(budget["minimum_free_bytes_before_root_create"]), "ENV_BLOCKED", "INSUFFICIENT_FREE_SPACE", str(usage.free))
    expected_env = {
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "GDN_DISABLE_COMPILE": "1",
        "TORCHDYNAMO_DISABLE": "1",
        "FUSED_GDN_PRECISION": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TRITON_CACHE_DIR": str(EXPECTED_ROOT / "triton_cache"),
    }
    for name, expected in expected_env.items():
        _require(os.environ.get(name) == expected, "ENV_BLOCKED", "ENVIRONMENT_BINDING_MISMATCH", f"{name}: {os.environ.get(name)!r} != {expected!r}")
    _require(
        os.environ.get("PYTHONPATH")
        == config["runtime"]["environment_template_before_any_torch_or_vendor_import"]["PYTHONPATH"],
        "ENV_BLOCKED",
        "PYTHONPATH_BINDING_MISMATCH",
        repr(os.environ.get("PYTHONPATH")),
    )
    _require(
        os.environ.get("CUDA_CACHE_DISABLE") is None,
        "ENV_BLOCKED",
        "CUDA_CACHE_DISABLE_MUST_BE_UNSET",
        repr(os.environ.get("CUDA_CACHE_DISABLE")),
    )
    pythonpath = os.environ.get("PYTHONPATH", "")
    _require(str(REPO_ROOT / "src") in pythonpath.split(os.pathsep), "ENV_BLOCKED", "PYTHONPATH_SRC_MISSING", pythonpath)
    _require(str(REPO_ROOT / "third_party" / "Sana") in pythonpath.split(os.pathsep), "ENV_BLOCKED", "PYTHONPATH_VENDOR_MISSING", pythonpath)
    toolchain = config["runtime"]["expected_toolchain_when_execution_is_authorized"]
    torch_distribution = importlib.metadata.version("torch")
    triton_distribution = importlib.metadata.version("triton")
    _require(
        platform.python_version() == toolchain["python"],
        "ENV_BLOCKED",
        "PYTHON_VERSION_MISMATCH",
        platform.python_version(),
    )
    _require(
        torch_distribution == str(toolchain["torch"]).split("+")[0],
        "ENV_BLOCKED",
        "TORCH_DISTRIBUTION_VERSION_MISMATCH",
        torch_distribution,
    )
    _require(
        triton_distribution == toolchain["triton"],
        "ENV_BLOCKED",
        "TRITON_DISTRIBUTION_VERSION_MISMATCH",
        triton_distribution,
    )
    nvcc = _run_command(["nvcc", "--version"])
    _require(
        f"V{toolchain['nvcc']}" in nvcc.stdout,
        "ENV_BLOCKED",
        "NVCC_VERSION_MISMATCH",
        nvcc.stdout,
    )
    return {
        "repository_heads": heads,
        "disk_free_bytes": usage.free,
        "environment": expected_env,
        "python_version": platform.python_version(),
        "torch_distribution_version_preimport": torch_distribution,
        "triton_distribution_version_preimport": triton_distribution,
        "nvcc_version_output": nvcc.stdout,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_create_namespace() -> None:
    old_umask = os.umask(0o002)
    created: list[Path] = []
    try:
        for path in EXPECTED_NAMESPACE_PATHS:
            os.mkdir(path, NAMESPACE_MODE)
            created.append(path)
            os.chmod(path, NAMESPACE_MODE)
            _require(
                _directory_owner_group(path) == (EXPECTED_NAMESPACE_OWNER, EXPECTED_NAMESPACE_GROUP),
                "ENV_BLOCKED",
                "NAMESPACE_OWNER_MISMATCH",
                f"{path}: {_directory_owner_group(path)!r}",
            )
            _fsync_directory(path)
            _fsync_directory(path.parent)
        os.mkdir(EXPECTED_ROOT, 0o770)
        created.append(EXPECTED_ROOT)
        _fsync_directory(EXPECTED_ROOT)
        _fsync_directory(EXPECTED_ROOT.parent)
    except FileExistsError as exc:
        _fail("ENV_BLOCKED", "EXCLUSIVE_CREATE_RACE", str(exc))
    finally:
        os.umask(old_umask)
    _require(created == [*EXPECTED_NAMESPACE_PATHS, EXPECTED_ROOT], "ENV_BLOCKED", "NAMESPACE_CREATE_INCOMPLETE", repr(created))


def _root_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
    return total


def _inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix().encode("utf-8")):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink forbidden in run root: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            rows.append({"path": relative, "kind": "directory", "mode": f"{stat.S_IMODE(metadata.st_mode):04o}"})
        elif stat.S_ISREG(metadata.st_mode):
            rows.append({"path": relative, "kind": "file", "mode": f"{stat.S_IMODE(metadata.st_mode):04o}", "size_bytes": metadata.st_size, "sha256": _sha256_file(path)})
        else:
            raise RuntimeError(f"non-regular root entry forbidden: {relative}")
    return rows


class RootLifecycle:
    """Single-owner artifact publisher and irreversible root freezer."""

    def __init__(self, root: Path, budget: Mapping[str, Any]):
        self.root = root
        self.budget = budget
        self.created = False
        self.frozen = False
        self.published: set[str] = set()
        self.started_monotonic = time.monotonic()

    def create(self) -> None:
        try:
            _exclusive_create_namespace()
        finally:
            try:
                metadata = self.root.lstat()
                self.created = stat.S_ISDIR(metadata.st_mode) and not self.root.is_symlink()
            except FileNotFoundError:
                self.created = False
        if not self.created:
            raise RuntimeError("exclusive root creation returned without a real root")
        if _directory_owner_group(self.root) != (
            EXPECTED_NAMESPACE_OWNER,
            EXPECTED_NAMESPACE_GROUP,
        ):
            raise RuntimeError(
                f"root owner/group differs: {_directory_owner_group(self.root)!r}"
            )
        triton_cache = self.root / "triton_cache"
        os.mkdir(triton_cache, 0o770)
        _fsync_directory(triton_cache)
        _fsync_directory(self.root)

    def publish(self, name: str, payload: Any) -> str:
        if not self.created or self.frozen:
            raise RuntimeError("artifact publish outside mutable root lifecycle")
        if name in self.published or "/" in name or name.startswith("."):
            raise RuntimeError(f"artifact name is duplicate or unsafe: {name}")
        path = self.root / name
        material = _canonical_json_bytes(payload)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            offset = 0
            while offset < len(material):
                offset += os.write(descriptor, material[offset:])
            os.fsync(descriptor)
            os.fchmod(descriptor, TERMINAL_FILE_MODE)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.root)
        self.published.add(name)
        return _sha256_bytes(material)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_monotonic

    def freeze(self, terminal_state: str, result_sha256: str) -> None:
        if self.frozen:
            return
        missing = sorted(set(REQUIRED_ROOT_ARTIFACTS) - self.published - {"FREEZE_RECEIPT.json"})
        if missing:
            raise RuntimeError(f"cannot freeze with missing artifacts: {missing}")
        root_bytes = _root_size(self.root)
        if root_bytes > int(self.budget["max_root_bytes"]):
            raise RuntimeError(f"root budget exceeded: {root_bytes}")
        pre_receipt = _inventory(self.root)
        receipt = {
            "schema": "cach.cach_a2.freeze_receipt.v1",
            "root": str(self.root),
            "nonce": EXPECTED_NONCE,
            "terminal_state": terminal_state,
            "result_sha256": result_sha256,
            "root_bytes_before_receipt": root_bytes,
            "pre_receipt_inventory": pre_receipt,
            "terminal_directory_mode": "0555",
            "terminal_file_mode": "0444",
            "post_freeze_mutation_allowed": False,
            "automatic_rerun_allowed": False,
        }
        self.publish("FREEZE_RECEIPT.json", receipt)
        for path in sorted(self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"symlink before freeze: {path}")
            os.chmod(path, TERMINAL_DIR_MODE if stat.S_ISDIR(metadata.st_mode) else TERMINAL_FILE_MODE)
        os.chmod(self.root, TERMINAL_DIR_MODE)
        _fsync_directory(self.root)
        _fsync_directory(self.root.parent)
        self.frozen = True

    def emergency_freeze(self) -> None:
        """Best-effort deny-write fallback for any partially published root."""

        if not self.created or self.frozen:
            return
        try:
            paths = sorted(
                self.root.rglob("*"),
                key=lambda item: len(item.parts),
                reverse=True,
            )
            for path in paths:
                try:
                    metadata = path.lstat()
                    if stat.S_ISLNK(metadata.st_mode):
                        continue
                    if stat.S_ISDIR(metadata.st_mode):
                        os.chmod(path, TERMINAL_DIR_MODE)
                    elif stat.S_ISREG(metadata.st_mode):
                        os.chmod(path, TERMINAL_FILE_MODE)
                except OSError:
                    continue
            try:
                os.chmod(self.root, TERMINAL_DIR_MODE)
            except OSError:
                pass
            try:
                _fsync_directory(self.root)
                _fsync_directory(self.root.parent)
            except OSError:
                pass
        finally:
            try:
                self.frozen = (
                    stat.S_IMODE(self.root.lstat().st_mode)
                    == TERMINAL_DIR_MODE
                )
            except OSError:
                self.frozen = False


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _assert_no_nonfinite(value: Any, path: str = "screen") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_nonfinite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_nonfinite(child, f"{path}[{index}]")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not _finite_number(value):
        raise ValueError(f"missing/non-finite metric: {name}")
    return float(value)


def _horizon_metric(metrics: Mapping[str, Any], name: str, horizon: int) -> float:
    values = metrics.get(name)
    if not isinstance(values, Mapping):
        raise ValueError(f"horizon metric is not mapping: {name}")
    value = values.get(str(horizon))
    if not _finite_number(value):
        raise ValueError(f"missing/non-finite metric: {name}.{horizon}")
    return float(value)


def _screen_check(screen: Mapping[str, Any], name: str) -> bool:
    validity = screen.get("validity")
    if not isinstance(validity, Mapping):
        raise ValueError("screen.validity must map")
    checks = validity.get("checks")
    if not isinstance(checks, Mapping) or type(checks.get(name)) is not bool:
        raise ValueError(f"missing boolean validity check: {name}")
    return bool(checks[name])


def _classification_inputs(screen: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    validity = screen.get("validity")
    metrics = screen.get("metrics")
    thresholds = config.get("thresholds")
    if not isinstance(validity, Mapping) or not isinstance(metrics, Mapping) or not isinstance(thresholds, Mapping):
        raise ValueError("screen validity/metrics and config thresholds must map")
    all_validity = all(
        validity.get(name) is True
        for name in (
            "implementation_valid",
            "data_valid",
            "numerics_valid",
            "finite_all",
            "all_card_validity_checks_pass",
        )
    )
    go_t = thresholds["operator_go"]
    common_t = thresholds["delta_live_common_mode_blocked"]
    stop_t = thresholds["operator_strong_stop"]
    primary_horizons = (1, 2, 4)
    bypass_equal = _screen_check(
        screen,
        "same_arm_no_action_vs_seam_disabled_bitwise_equal_theta0_and_final",
    )
    bypass_max_zero = _screen_check(
        screen, "same_arm_no_action_vs_seam_disabled_max_abs_exact_zero"
    )
    no_action_grad_none = _screen_check(
        screen,
        "no_action_candidate_only_parameter_gradients_all_none_theta0_and_final",
    )
    no_action_jvp_zero = _screen_check(
        screen, "no_action_raw_action_jvp_exact_zero_theta0_and_final"
    )
    counterfactual_h = all(
        _horizon_metric(metrics, "counterfactual_delta_nmse_correct_by_horizon", h)
        <= float(go_t["counterfactual_delta_nmse_correct_h1_h2_h4_max_each"])
        for h in primary_horizons
    )
    shuffle_h = all(
        _horizon_metric(metrics, "shuffle_gap_rel_by_horizon", h)
        >= float(go_t["shuffle_gap_rel_h1_h2_h4_min_each"])
        for h in primary_horizons
    )
    no_action_h = all(
        _horizon_metric(metrics, "no_action_gap_rel_by_horizon", h)
        >= float(go_t["no_action_gap_rel_h1_h2_h4_min_each"])
        for h in primary_horizons
    )
    shuffle_separation = (
        _metric(metrics, "shuffle_gap_rel") >= float(go_t["shuffle_gap_rel_min"])
        and shuffle_h
    )
    no_action_separation = (
        _metric(metrics, "no_action_gap_rel") >= float(go_t["no_action_gap_rel_min"])
        and no_action_h
    )
    seam_live = (
        _metric(metrics, "seam_grad_rms_step0")
        > float(go_t["seam_gradient_rms_step0_strictly_greater_than"])
        and _metric(metrics, "seam_update_rms_final")
        > float(go_t["seam_update_rms_final_strictly_greater_than"])
    )
    operator_go = all_validity and all(
        (
            _metric(metrics, "candidate_gain_vs_reference")
            >= float(go_t["candidate_gain_vs_reference_min"]),
            _metric(metrics, "loss_drop_rel") >= float(go_t["loss_drop_rel_min"]),
            _metric(metrics, "counterfactual_delta_nmse_correct")
            <= float(go_t["counterfactual_delta_nmse_correct_max"]),
            counterfactual_h,
            shuffle_separation,
            no_action_separation,
            bypass_equal,
            bypass_max_zero,
            no_action_grad_none,
            no_action_jvp_zero,
            _metric(metrics, "last_block_local_raw_action_jvp_rms_step300")
            > float(go_t["correct_action_local_jvp_rms_step300_strictly_greater_than"]),
            seam_live,
        )
    )
    common = metrics.get("paired_common_delta")
    if not isinstance(common, Mapping) or not isinstance(common.get("aggregate"), Mapping):
        raise ValueError("paired_common_delta.aggregate must map")
    aggregate = common["aggregate"]
    common_mode_blocked = all_validity and not operator_go and all(
        (
            bypass_equal,
            _metric(metrics, "counterfactual_delta_nmse_correct")
            <= float(common_t["counterfactual_delta_nmse_correct_max"]),
            all(
                _horizon_metric(metrics, "counterfactual_delta_nmse_correct_by_horizon", h)
                <= float(common_t["counterfactual_delta_nmse_correct_h1_h2_h4_max_each"])
                for h in primary_horizons
            ),
            _metric(aggregate, "half_delta_energy_ratio")
            >= float(common_t["half_delta_energy_ratio_min"]),
            _metric(aggregate, "half_delta_energy_ratio")
            <= float(common_t["half_delta_energy_ratio_max"]),
            _metric(aggregate, "half_delta_alignment_cosine")
            >= float(common_t["half_delta_alignment_cosine_min"]),
            _metric(aggregate, "paired_common_mse_div_target_total_energy")
            > float(common_t["paired_common_mse_div_target_total_energy_strictly_greater_than"]),
            not shuffle_separation or not no_action_separation,
        )
    )
    strong_stop = all_validity and all(
        (
            _metric(metrics, "candidate_gain_vs_reference")
            <= float(stop_t["candidate_gain_vs_reference_max"]),
            _metric(metrics, "loss_drop_rel") <= float(stop_t["loss_drop_rel_max"]),
            _metric(metrics, "counterfactual_delta_nmse_correct")
            >= float(stop_t["counterfactual_delta_nmse_correct_min"]),
            _metric(metrics, "shuffle_gap_rel") <= float(stop_t["shuffle_gap_rel_max"]),
            _metric(metrics, "no_action_gap_rel") <= float(stop_t["no_action_gap_rel_max"]),
            _metric(metrics, "seam_grad_rms_step0")
            > float(stop_t["seam_gradient_rms_step0_strictly_greater_than"]),
            _metric(metrics, "seam_update_rms_final")
            > float(stop_t["seam_update_rms_final_strictly_greater_than"]),
        )
    )
    return {
        "all_validity": all_validity,
        "operator_go": operator_go,
        "delta_live_common_mode_blocked": common_mode_blocked,
        "operator_strong_stop": strong_stop,
        "shuffle_separation": shuffle_separation,
        "no_action_separation": no_action_separation,
        "seam_live": seam_live,
    }


def _classify_screen(
    screen: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, str | None]:
    """Pure card-ordered direct classifier used by the launcher and tests."""

    validity = screen.get("validity")
    if not isinstance(validity, Mapping):
        return "INVALID_RUN", None
    if validity.get("implementation_valid") is not True:
        return "IMPLEMENTATION_INVALID", None
    if validity.get("numerics_valid") is not True or validity.get("finite_all") is not True:
        return "NUMERICAL_INVALID", None
    if validity.get("data_valid") is not True or validity.get("all_card_validity_checks_pass") is not True:
        return "INVALID_RUN", None
    inputs = _classification_inputs(screen, config)
    if inputs["operator_go"]:
        return "OPERATOR_GO", None
    if inputs["delta_live_common_mode_blocked"]:
        return (
            "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED",
            "A2_DELTA_LIVE_COMMON_MODE_BLOCKED",
        )
    if inputs["operator_strong_stop"]:
        return "OPERATOR_STOP", None
    return "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED", "A2_VALID_NEITHER"


def _validate_progress_events(events: Sequence[Mapping[str, Any]]) -> None:
    names = [event.get("event") for event in events]
    if names.count("task_tensor_manifest_ready") != 1 or names.count("theta0_manifests_ready") != 1:
        raise ValueError("task/theta0 progress event multiplicity differs")
    expected_steps = [1, 50, 100, 150, 200, 250, 300]
    for arm in ("REF-GDN-CORRECTED", "CACH-A2"):
        observed = [
            int(event["payload"]["optimizer_steps_completed"])
            for event in events
            if event.get("event") == "optimizer_progress"
            and isinstance(event.get("payload"), Mapping)
            and event["payload"].get("arm") == arm
        ]
        if observed != expected_steps:
            raise ValueError(f"optimizer progress differs for {arm}: {observed}")


def _screen_evidence_errors(
    screen: Mapping[str, Any], events: Sequence[Mapping[str, Any]], config: Mapping[str, Any], gpu_uuid: str
) -> list[str]:
    errors: list[str] = []
    try:
        if screen.get("schema") != SCREEN_SCHEMA:
            raise ValueError(f"screen schema differs: {screen.get('schema')!r}")
        expected_keys = frozenset(
            {
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
        )
        _require_exact_keys(screen, expected_keys, label="screen")
        _assert_no_nonfinite(screen)
        validity = screen["validity"]
        if not isinstance(validity, Mapping):
            raise ValueError("validity must map")
        for name in (
            "implementation_valid",
            "data_valid",
            "numerics_valid",
            "finite_all",
            "all_card_validity_checks_pass",
        ):
            if type(validity.get(name)) is not bool:
                raise ValueError(f"validity.{name} must be bool")
        traces = screen["loss_trace"]
        if not isinstance(traces, Mapping):
            raise ValueError("loss_trace must map")
        for arm in ("reference", "candidate"):
            trace = traces.get(arm)
            if not isinstance(trace, list) or len(trace) != 301 or not all(_finite_number(item) for item in trace):
                raise ValueError(f"{arm} loss trace must contain 301 finite points")
        diagnostics = screen["diagnostics"]
        if not isinstance(diagnostics, Mapping):
            raise ValueError("diagnostics must map")
        if diagnostics.get("optimizer_steps_completed_by_arm") != {
            "REF-GDN-CORRECTED": 300,
            "CACH-A2": 300,
        }:
            raise ValueError("optimizer step counts differ")
        if diagnostics.get("operator_level") != "VENDOR_KERNEL" or diagnostics.get("integration_path") != "EXPERIMENTAL_PATH":
            raise ValueError("operator level/integration path differs")
        resources = screen["resource_usage"]
        if not isinstance(resources, Mapping):
            raise ValueError("resource_usage must map")
        forbidden_facts = {
            "checkpoint_loaded": False,
            "checkpoint_saved": False,
            "real_data_used": False,
            "cpu_proxy_or_reference_fallback_used": False,
            "new_review_token_created_derived_reset_or_consumed": False,
            "predecessor_consumed_claim_mutated": False,
        }
        for name, expected in forbidden_facts.items():
            if resources.get(name) is not expected:
                raise ValueError(f"resource fact differs: {name}")
        budget = config["budget"]
        if int(resources.get("cuda_peak_memory_reserved_bytes_screen", -1)) > int(budget["max_gpu_memory_bytes"]):
            raise ValueError("GPU memory budget exceeded")
        if int(resources.get("process_peak_rss_bytes_screen", -1)) > int(budget["max_process_rss_bytes"]):
            raise ValueError("RSS budget exceeded")
        jit = screen["jit_inventory"]
        if not isinstance(jit, Mapping):
            raise ValueError("jit inventory must map")
        if jit.get("cuda_visible_devices") != gpu_uuid or jit.get("cuda_device_count") != 1:
            raise ValueError("runtime GPU UUID visibility differs")
        if jit.get("vendor_kernel_executed") is not True:
            raise ValueError("vendor kernel was not executed")
        _validate_progress_events(events)
        _classification_inputs(screen, config)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def _screen_artifacts(screen: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "THETA0_MANIFESTS.json": screen["theta0_manifests"],
        "LOSS_TRACE.json": screen["loss_trace"],
        "RAW_METRICS.json": screen["metrics"],
        "FINAL_PARAMETER_DIGESTS.json": screen["final_parameter_digests"],
        "JIT_INVENTORY.json": screen["jit_inventory"],
        "RAW_EVIDENCE.json": evidence,
    }


def _postcreate_failure_state(exc: BaseException) -> str:
    if isinstance(exc, MemoryError):
        return "HARNESS_REJECTED"
    if isinstance(exc, FloatingPointError):
        return "NUMERICAL_INVALID"
    if isinstance(exc, (ImportError, ModuleNotFoundError, OSError)):
        return "ENV_BLOCKED"
    if isinstance(exc, (AssertionError, TypeError, ValueError)):
        return "IMPLEMENTATION_INVALID"
    return "INVALID_RUN"


def _failure_placeholder(name: str, state: str, reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": "cach.cach_a2.failure_placeholder.v1",
        "artifact": name,
        "available": False,
        "terminal_state": state,
        "reasons": list(reasons),
    }


def _publish_failure(
    lifecycle: RootLifecycle,
    *,
    state: str,
    reasons: Sequence[str],
    context: Mapping[str, Any],
    progress_events: Sequence[Mapping[str, Any]],
) -> None:
    if "RUN_CONTEXT.json" not in lifecycle.published:
        lifecycle.publish("RUN_CONTEXT.json", context)
    if "PROGRESS_EVENTS.json" not in lifecycle.published:
        lifecycle.publish("PROGRESS_EVENTS.json", {"events": list(progress_events)})
    for name in REQUIRED_ROOT_ARTIFACTS:
        if name in {"RUN_CONTEXT.json", "PROGRESS_EVENTS.json", "RESULT.json", "FREEZE_RECEIPT.json"}:
            continue
        if name not in lifecycle.published:
            lifecycle.publish(name, _failure_placeholder(name, state, reasons))
    result = {
        "schema": "cach.cach_a2.direct_final_result.v1",
        "card_path": EXPECTED_CARD,
        "card_sha256": EXPECTED_CARD_SHA256,
        "root": str(EXPECTED_ROOT),
        "nonce": EXPECTED_NONCE,
        "terminal_state": state,
        "typed_verdict": state,
        "diagnostic_subclassification": None,
        "valid_result": False,
        "reasons": list(reasons),
        "new_review_token_created_or_consumed": False,
        "formal_admission_effect": "NONE",
        "global_stage3_effect": "NONE",
        "automatic_rerun": False,
        "valid_result_terminal_without_claim_ledger": True,
        "valid_result_publication": (
            "DIRECT_FINAL_RESULT_INSIDE_ROOT_BEFORE_SINGLE_TERMINAL_FREEZE"
        ),
    }
    result_sha = lifecycle.publish("RESULT.json", result)
    lifecycle.freeze(state, result_sha)


def _install_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> NoReturn:
        raise RunInterrupted(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    return parser.parse_args(argv)


def _precreate_block(exc: PreflightError) -> int:
    payload = {
        "schema": "cach.cach_a2.precreate_block.v1",
        "terminal_state": exc.state,
        "token": exc.token,
        "detail": exc.detail,
        "root_created": False,
    }
    sys.stderr.buffer.write(_canonical_json_bytes(payload))
    return 2


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    lifecycle: RootLifecycle | None = None
    progress_events: list[Mapping[str, Any]] = []
    context: dict[str, Any] = {
        "schema": "cach.cach_a2.run_context.v1",
        "card_path": EXPECTED_CARD,
        "card_sha256": EXPECTED_CARD_SHA256,
        "root": str(EXPECTED_ROOT),
        "nonce": EXPECTED_NONCE,
        "phase": "PRE_ROOT",
    }
    previous_handlers: dict[int, Any] = {}
    try:
        _require(args.nonce == EXPECTED_NONCE, "AUTH_BLOCKED", "CLI_NONCE_MISMATCH", repr(args.nonce))
        card_path = args.card.resolve(strict=False)
        root_path = args.root.resolve(strict=False)
        card = _validate_card(card_path)
        authority, authority_envelope_sha = _read_authority_envelope()
        quartet_pre = _verify_frozen_quartet(authority)
        config = _duplicate_rejecting_json(REPO_ROOT / EXPECTED_CONFIG)
        predecessor = _verify_predecessors(card)
        _verify_card_source_pins(card)
        _validate_config(config, card, predecessor)
        physical_index, gpu_uuid, execution_authority_sha = _validate_authority(authority, config)
        environment = _validate_precreate_environment(root_path, physical_index, gpu_uuid, config)
        gpu_snapshot_1 = _gpu_snapshot(physical_index, gpu_uuid, "authority_bound_idle_snapshot_1")
        _require(
            gpu_snapshot_1["selected_gpu"]["driver_version"]
            == config["runtime"]["expected_toolchain_when_execution_is_authorized"]["driver"],
            "ENV_BLOCKED",
            "DRIVER_VERSION_MISMATCH",
            repr(gpu_snapshot_1["selected_gpu"]["driver_version"]),
        )
        time.sleep(120.0)
        gpu_snapshot_2 = _gpu_snapshot(physical_index, gpu_uuid, "authority_bound_idle_snapshot_2")
        _require(not EXPECTED_ROOT.exists(), "ENV_BLOCKED", "ROOT_APPEARED_DURING_IDLE_WINDOW", str(EXPECTED_ROOT))

        previous_handlers = _install_signal_handlers()
        lifecycle = RootLifecycle(EXPECTED_ROOT, config["budget"])
        lifecycle.create()
        gpu_snapshot_3 = _gpu_snapshot(physical_index, gpu_uuid, "post_root_pre_import_idle_snapshot")
        context.update(
            {
                "phase": "ROOT_CREATED_PRE_IMPORT",
                "authority_envelope_sha256": authority_envelope_sha,
                "source_authority_sha256": authority["source_authority"]["statement_sha256"],
                "execution_authority_sha256": execution_authority_sha,
                "successor_plan_deviation_accepted": True,
                "no_new_review_token_accepted": True,
                "source_quartet_sha256_pre": quartet_pre,
                "predecessor_closure_sha256": predecessor["canonical_manifest_sha256"],
                "predecessor_consumed_state": predecessor["consumed_predecessor_state"],
                "gpu_binding": {"physical_index": physical_index, "uuid": gpu_uuid, "cuda_logical_index": 0},
                "gpu_idle_snapshots": [gpu_snapshot_1, gpu_snapshot_2, gpu_snapshot_3],
                "environment_preflight": environment,
                "capabilities": authority["capabilities"],
                "real_data": False,
                "checkpoint_load": False,
                "checkpoint_save": False,
                "new_review_token": None,
            }
        )
        lifecycle.publish("RUN_CONTEXT.json", context)

        # This is the first point at which Torch, Triton, vendor, or model code
        # may enter the process.  PYTHONDONTWRITEBYTECODE is already pinned.
        from sana_wam.model.cach_av1b_a2_zero_anchor_vendor_gdn_bridge import (
            run_cach_a2_screen,
        )

        import torch
        import triton

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("execution must expose exactly one CUDA GPU")
        runtime_toolchain = config["runtime"][
            "expected_toolchain_when_execution_is_authorized"
        ]
        if (
            torch.__version__ != runtime_toolchain["torch"]
            or torch.version.cuda != runtime_toolchain["cuda_toolkit"]
            or triton.__version__ != runtime_toolchain["triton"]
        ):
            raise RuntimeError(
                "runtime Torch/CUDA/Triton versions differ from frozen config"
            )
        properties = torch.cuda.get_device_properties(0)
        if properties.name != gpu_snapshot_3["selected_gpu"]["name"]:
            raise RuntimeError("Torch CUDA device name differs from dynamic binding")
        torch.cuda.reset_peak_memory_stats(0)

        def progress_callback(event: Mapping[str, Any]) -> None:
            if not isinstance(event, Mapping):
                raise TypeError("progress event must map")
            canonical_copy = json.loads(_canonical_json_bytes(event))
            canonical_copy["sequence"] = len(progress_events)
            canonical_copy["elapsed_seconds"] = lifecycle.elapsed()
            progress_events.append(canonical_copy)

        screen = run_cach_a2_screen(
            config,
            expected_gpu_uuid=gpu_uuid,
            progress_callback=progress_callback,
        )
        if not isinstance(screen, Mapping):
            raise TypeError("run_cach_a2_screen must return a mapping")
        elapsed = lifecycle.elapsed()
        errors = _screen_evidence_errors(screen, progress_events, config, gpu_uuid)
        quartet_post = _verify_frozen_quartet(authority)
        if quartet_post != quartet_pre:
            errors.append("source quartet changed during execution")
        if elapsed > float(config["budget"]["max_wall_seconds_total"]):
            errors.append(f"wall budget exceeded: {elapsed}")
        if _root_size(EXPECTED_ROOT) > int(config["budget"]["max_root_bytes"]):
            errors.append("root budget exceeded before artifact publication")
        if errors:
            raise ValueError("; ".join(errors))

        typed_verdict, diagnostic_subclassification = _classify_screen(screen, config)
        classification_inputs = _classification_inputs(screen, config)
        evidence = {
            "schema": "cach.cach_a2.raw_evidence.v1",
            "validity": screen["validity"],
            "diagnostics": screen["diagnostics"],
            "resource_usage": screen["resource_usage"],
            "classification_inputs": classification_inputs,
            "typed_verdict": typed_verdict,
            "diagnostic_subclassification": diagnostic_subclassification,
            "source_quartet_sha256_pre": quartet_pre,
            "source_quartet_sha256_post": quartet_post,
            "predecessor_closure_sha256": predecessor["canonical_manifest_sha256"],
            "predecessor_consumed_evidence_read_only": True,
            "new_review_token_created_or_consumed": False,
            "gpu_binding": {"physical_index": physical_index, "uuid": gpu_uuid},
            "elapsed_seconds_total": elapsed,
        }
        lifecycle.publish("PROGRESS_EVENTS.json", {"events": progress_events})
        artifact_hashes: dict[str, str] = {}
        for name, payload in _screen_artifacts(screen, evidence).items():
            artifact_hashes[name] = lifecycle.publish(name, payload)
        valid_result = typed_verdict in VALID_TERMINAL_VERDICTS
        result = {
            "schema": "cach.cach_a2.direct_final_result.v1",
            "card_path": EXPECTED_CARD,
            "card_sha256": EXPECTED_CARD_SHA256,
            "architecture_decision_path": EXPECTED_DECISION,
            "architecture_decision_sha256": EXPECTED_DECISION_SHA256,
            "root": str(EXPECTED_ROOT),
            "nonce": EXPECTED_NONCE,
            "terminal_state": typed_verdict,
            "typed_verdict": typed_verdict,
            "diagnostic_subclassification": diagnostic_subclassification,
            "valid_result": valid_result,
            "all_card_validity_checks_pass": screen["validity"]["all_card_validity_checks_pass"],
            "classification_inputs": classification_inputs,
            "metrics": screen["metrics"],
            "artifact_sha256": artifact_hashes,
            "source_quartet_sha256": quartet_post,
            "predecessor_closure_sha256": predecessor["canonical_manifest_sha256"],
            "predecessor_consumed_evidence_read_only": True,
            "new_review_token_created_or_consumed": False,
            "formal_admission_effect": "NONE",
            "global_stage3_effect": "NONE",
            "automatic_av2_or_rerun": False,
            "valid_result_terminal_without_claim_ledger": True,
            "valid_result_publication": (
                "DIRECT_FINAL_RESULT_INSIDE_ROOT_BEFORE_SINGLE_TERMINAL_FREEZE"
            ),
        }
        result_sha = lifecycle.publish("RESULT.json", result)
        lifecycle.freeze(typed_verdict, result_sha)
        sys.stdout.buffer.write(_canonical_json_bytes(result))
        return 0 if valid_result else 3
    except PreflightError as exc:
        if lifecycle is None or not lifecycle.created:
            return _precreate_block(exc)
        reasons = [str(exc)]
        try:
            _publish_failure(
                lifecycle,
                state=exc.state,
                reasons=reasons,
                context=context,
                progress_events=progress_events,
            )
        except BaseException as freeze_exc:
            lifecycle.emergency_freeze()
            sys.stderr.write(f"fatal fail-closed freeze error: {freeze_exc}\n")
        return 4
    except BaseException as exc:
        if lifecycle is None or not lifecycle.created:
            blocked = PreflightError("HARNESS_REJECTED", "UNEXPECTED_PRE_ROOT_EXCEPTION", f"{type(exc).__name__}: {exc}")
            return _precreate_block(blocked)
        state = _postcreate_failure_state(exc)
        reasons = [f"{type(exc).__name__}: {exc}", traceback.format_exc()]
        try:
            _publish_failure(
                lifecycle,
                state=state,
                reasons=reasons,
                context=context,
                progress_events=progress_events,
            )
        except BaseException as freeze_exc:
            lifecycle.emergency_freeze()
            sys.stderr.write(f"fatal fail-closed freeze error: {freeze_exc}\n")
        return 4
    finally:
        if previous_handlers:
            _restore_signal_handlers(previous_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
