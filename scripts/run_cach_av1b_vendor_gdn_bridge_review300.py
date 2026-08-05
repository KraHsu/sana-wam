#!/usr/bin/env python3
"""Run or claim-recover the frozen AV-1B review300 vendor-GDN screen.

Only the Python standard library is imported at module load.  The launcher
fail-closes before CUDA/vendor import unless the frozen card, materialized
quartet, predecessor evidence, authority envelope, namespace, environment,
and exact reserved GPU all match.  A valid run first freezes an immutable
``VALID_REVIEW_PROVISIONAL_NO_VERDICT`` root.  Its final review verdict is then
published exactly once through the canonical atomic token claim.  The
claim-only recovery path never imports or executes torch, CUDA, vendor code,
the model, optimizer, data recipe, or tests.
"""

from __future__ import annotations

import argparse
import atexit
import ast
import errno
import grp
import hashlib
from importlib import metadata as importlib_metadata
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
from typing import Any, Mapping, NoReturn, Sequence


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WORKTREE = Path("/home/zch/workspace/sana-wam")
EXPECTED_CARD = (
    "docs/cach_sana_wam/architecture_validation/av1b_review300/"
    "AV1B_REVIEW300_RUN_CARD.json"
)
EXPECTED_CARD_SHA256 = (
    "a552522a951384996aa8f18ce9cb8a5e6e130ac4d2db68515b5ea6303c6722bc"
)
EXPECTED_CONFIG = "configs/experiments/cach_av1b_vendor_gdn_bridge_review300.yaml"
EXPECTED_BRIDGE = "src/sana_wam/model/cach_av1b_vendor_gdn_bridge_review300.py"
EXPECTED_RUNNER = "scripts/run_cach_av1b_vendor_gdn_bridge_review300.py"
EXPECTED_TEST = "tests/test_cach_av1b_vendor_gdn_bridge_review300.py"
EXPECTED_NONCE = "bd32b08576c9d598c629b90286331b87"
EXPECTED_ROOT = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/av1b_review300/06f5d09127f8/"
    "av1b-review300-bd32b08576c9d598c629b90286331b87"
)
EXPECTED_ROOT_NAMESPACE_BASE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens"
)
EXPECTED_ROOT_NAMESPACE_PATHS = (
    EXPECTED_ROOT_NAMESPACE_BASE / "av1b_review300",
    EXPECTED_ROOT_NAMESPACE_BASE / "av1b_review300" / "06f5d09127f8",
)
EXPECTED_TOKEN_NAMESPACE_PATHS = (
    EXPECTED_ROOT_NAMESPACE_BASE / "token_ledgers",
    EXPECTED_ROOT_NAMESPACE_BASE
    / "token_ledgers"
    / "c73a943047295338cd9de14e6e53199b463ea66ae92b99d4b0b3393f5ec34030",
)
EXPECTED_GPU_PHYSICAL_INDEX = 7
EXPECTED_GPU_UUID = "GPU-41c95a43-ce96-fff3-33e0-739a3931d603"
EXPECTED_GPU_NAME = "NVIDIA H200"
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_TORCH_VERSION = "2.7.1+cu128"
EXPECTED_TORCH_DISTRIBUTION_VERSION = "2.7.1"
EXPECTED_TRITON_VERSION = "3.5.1"
EXPECTED_CUDA_VERSION = "12.8"
EXPECTED_NVCC_VERSION = "12.8.61"
EXPECTED_DRIVER_VERSION = "570.211.01"
EXPECTED_SOURCE_AUTHORITY_SHA256 = (
    "2bd18d09c77ae03451c0fa46e0270793c4e19802bab3906057103f405b40b573"
)
EXPECTED_SOURCE_AUTHORITY_UTF8_BYTES = 643
EXPECTED_SELECTION_AUTHORITY_SHA256 = (
    "4ffbe5d81c788957680c29a47457f3e332152672b8143748c75e85297dcb0cc1"
)
EXPECTED_CARD_FREEZE_AUTHORITY_SHA256 = (
    "7bbf4a93f11deea41a9f2418c4e29628002e2150816559aa8c88af4031cafb3c"
)
EXPECTED_SELECTION = "NO_BUG_BUDGET_EXTENSION_FRESH_ROOT_FRESH_INITIALIZATION_300_STEPS"
EXPECTED_REJECTED_SELECTION = "BUG_FIX_FRESH_REVISION_FRESH_ROOT_FRESH_INITIALIZATION_200_STEPS"
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
INVALID_TERMINAL_STATES = frozenset(
    {
        "AUTH_BLOCKED",
        "ENV_BLOCKED",
        "HARNESS_REJECTED",
        "INVALID_RUN",
        "IMPLEMENTATION_INVALID",
        "NUMERICAL_INVALID",
        "TOKEN_LEDGER_CORRUPT",
    }
)
VALID_PROVISIONAL_STATE = "VALID_REVIEW_PROVISIONAL_NO_VERDICT"
FINAL_REVIEW_STATES = frozenset(
    {"OPERATOR_GO", "OPERATOR_STOP", "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED"}
)
AUTHORITY_ENVELOPE_SCHEMA = "cach.av1b_review300.runtime_authority_envelope.v1"
CLAIM_SCHEMA = "cach.av1b_review300.review_token_claim.v1"
NAMESPACE_MODE = 0o2775
EXPECTED_NAMESPACE_OWNER = "zch"
EXPECTED_NAMESPACE_GROUP = "sharegrp"


class PreflightError(RuntimeError):
    """A typed failure raised before the immutable run root exists."""

    def __init__(self, state: str, token: str, detail: str):
        super().__init__(f"{token}: {detail}")
        self.state = state
        self.token = token
        self.detail = detail


class RunInterrupted(RuntimeError):
    pass


class TokenLedgerCorrupt(RuntimeError):
    """The canonical claim path exists after an incomplete local publication."""

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


def _predecessor_pin_manifest(card: Mapping[str, Any]) -> dict[str, Any]:
    identity = card["identity"]
    return {
        "source_pins": {
            label: value
            for label, value in card["source_pins"].items()
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
        "latest_review_token_state_head": card["review_token"][
            "latest_token_state_head"
        ],
    }


def _expected_execution_authority_statement(
    quartet: Mapping[str, str], card: Mapping[str, Any]
) -> str:
    """Return the one exact future execution-authority statement.

    The statement is intentionally derived from the already-materialized
    quartet, including this runner's out-of-band full hash.  It therefore does
    not introduce a self-hash field into the frozen config.
    """

    predecessor_manifest_sha256 = _sha256_bytes(
        _canonical_json_bytes(_predecessor_pin_manifest(card))
    )
    return (
        "授权在 H200 严格按 AV1B_REVIEW300_RUN_CARD.json SHA256 "
        f"{EXPECTED_CARD_SHA256} 执行 execution phase：固定 materialized review300 "
        f"quartet SHA256 bridge={quartet[EXPECTED_BRIDGE]}, "
        f"config={quartet[EXPECTED_CONFIG]}, runner={quartet[EXPECTED_RUNNER]}, "
        f"test={quartet[EXPECTED_TEST]}；选择固定为 {EXPECTED_SELECTION}，拒绝 "
        f"{EXPECTED_REJECTED_SELECTION}，禁止同时或顺序使用两个 choice；逐项绑定该 frozen "
        "card 的全部 predecessor card/source/result pins，即 source_pins（排除另由 quartet "
        "绑定的 future_review300_generated_files）、identity.predecessor_av0_card、"
        "identity.predecessor_av1b_card、identity.predecessor_av1_artifacts、"
        "identity.predecessor_av1b_artifacts、predecessor roots/nonce 与 review_token."
        "latest_token_state_head，按 canonical UTF-8 sorted compact JSON+LF 形成的完整 "
        f"predecessor pin manifest SHA256={predecessor_manifest_sha256}；仅排他创建卡内固定的 "
        "run namespace "
        f"{EXPECTED_ROOT_NAMESPACE_PATHS[0]}、{EXPECTED_ROOT_NAMESPACE_PATHS[1]}"
        f"（02775 {EXPECTED_NAMESPACE_OWNER}:{EXPECTED_NAMESPACE_GROUP}）、唯一 root "
        f"{EXPECTED_ROOT}（nonce {EXPECTED_NONCE}），并仅在 valid provisional root "
        f"冻结且 final verdict 已计算后排他创建 token namespace "
        f"{EXPECTED_TOKEN_NAMESPACE_PATHS[0]}、{EXPECTED_TOKEN_NAMESPACE_PATHS[1]}"
        f"及 review token {EXPECTED_REVIEW_TOKEN} 的 canonical claim "
        f"{EXPECTED_REVIEW_CLAIM}；所有 namespace 逐祖先 lstat 并拒绝 symlink/non-directory，"
        "严格按声明顺序排他 mkdir，EEXIST/race fail-closed，每层新目录及父目录 fsync；"
        "latest token head 固定为 "
        "predecessor AV1B RESULT SHA256 "
        "d33e2e6bdf59cc1d2ffe51b32d48ce22daa4da1a902e3e9bded28cce29adb5de；"
        f"独占物理 GPU {EXPECTED_GPU_PHYSICAL_INDEX} / UUID {EXPECTED_GPU_UUID}，"
        "允许单 GPU CUDA/Triton JIT、vendor autograd、forward/backward、卡内局部 "
        "JVP、每臂 fresh-init AdamW 300-step synthetic non-formal training、轻量测试"
        "和诊断；禁止真实数据、checkpoint load/save、完整 2B、正式评测、admission、"
        "deploy、AV2 和 Global Stage 3；valid run 必须先冻结 "
        "VALID_REVIEW_PROVISIONAL_NO_VERDICT RESULT 并将 root 冻结为 dirs 0555/files 0444，"
        "再按 O_CREAT|O_EXCL|O_NOFOLLOW 原子创建 canonical JSON claim，依次执行 claim "
        "content fsync、fchmod 0444、再次 fsync、ledger leaf fsync、leaf fchmod 0555、再次 "
        "fsync 及祖先反向 fsync；所有失败 fail-closed，禁止自动模型重跑、root 复用、"
        "post-freeze root mutation，所有未列能力默认禁止。"
    )


def _read_authority_envelope() -> tuple[dict[str, Any], str]:
    limit = 1024 * 1024
    payload = sys.stdin.buffer.read(limit + 1)
    _require(
        0 < len(payload) <= limit,
        "AUTH_BLOCKED",
        "AUTHORITY_STDIN_SIZE_INVALID",
        f"bytes={len(payload)}",
    )
    _require(
        not payload.startswith(b"\xef\xbb\xbf"),
        "AUTH_BLOCKED",
        "AUTHORITY_STDIN_BOM_FORBIDDEN",
        "UTF-8 BOM is forbidden",
    )
    try:
        text = payload.decode("utf-8")

        def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, child in pairs:
                if key in value:
                    raise ValueError(f"duplicate JSON key: {key!r}")
                value[key] = child
            return value

        envelope = json.loads(text, object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise PreflightError("AUTH_BLOCKED", "AUTHORITY_STDIN_PARSE_REJECTED", str(exc)) from exc
    _require(
        isinstance(envelope, dict) and payload == _canonical_json_bytes(envelope),
        "AUTH_BLOCKED",
        "AUTHORITY_STDIN_NOT_CANONICAL",
        "require sorted compact UTF-8 JSON plus one LF",
    )
    return envelope, _sha256_bytes(payload)


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
        card.get("schema")
        == "cach.architecture_validation.av1b_review300_run_card.v1"
        and card.get("card_id") == "cach-av1b-review300-no-bug-budget-extension-v1"
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
    statement = card_authority.get("card_freeze_statement")
    _require(
        isinstance(statement, str)
        and _sha256_bytes(statement.encode("utf-8"))
        == EXPECTED_CARD_FREEZE_AUTHORITY_SHA256
        and card_authority.get("card_freeze_statement_sha256")
        == EXPECTED_CARD_FREEZE_AUTHORITY_SHA256,
        "AUTH_BLOCKED",
        "CARD_FREEZE_AUTHORITY_MISMATCH",
        "statement bytes or pin differ",
    )
    selection = card_authority.get("selection_statement")
    _require(
        selection == EXPECTED_SELECTION
        and card_authority.get("selected_choice") == EXPECTED_SELECTION
        and card_authority.get("rejected_choice") == EXPECTED_REJECTED_SELECTION
        and _sha256_bytes(selection.encode("utf-8"))
        == EXPECTED_SELECTION_AUTHORITY_SHA256
        and card_authority.get("selection_statement_sha256")
        == EXPECTED_SELECTION_AUTHORITY_SHA256
        and card_authority.get("simultaneous_or_sequential_use_of_both_choices")
        is False,
        "AUTH_BLOCKED",
        "REVIEW_CHOICE_AUTHORITY_MISMATCH",
        repr(card_authority),
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
    launcher = card.get("launcher_contract", {})
    _require(
        launcher.get("expected_environment")
        == {
            "python": EXPECTED_PYTHON_VERSION,
            "torch": EXPECTED_TORCH_VERSION,
            "triton": EXPECTED_TRITON_VERSION,
            "cuda_toolkit": EXPECTED_CUDA_VERSION,
            "nvcc": EXPECTED_NVCC_VERSION,
            "driver": EXPECTED_DRIVER_VERSION,
        }
        and launcher.get("environment_must_be_set_before_any_torch_or_vendor_import")
        == EXPECTED_ENV
        and launcher.get("successor_launcher_requirements", {}).get(
            "standard_library_only_before_all_preflight_checks"
        )
        is True,
        "AUTH_BLOCKED",
        "CARD_LAUNCHER_ENVIRONMENT_MISMATCH",
        repr(launcher),
    )
    review = card.get("review_token", {})
    _require(
        review.get("token_id") == EXPECTED_REVIEW_TOKEN
        and review.get("canonical_claim_path") == str(EXPECTED_REVIEW_CLAIM)
        and review.get("state_at_successor_card_freeze") == "UNCONSUMED"
        and review.get("selected_choice") == EXPECTED_SELECTION
        and review.get("rejected_choice") == EXPECTED_REJECTED_SELECTION
        and review.get("simultaneous_or_sequential_use_of_both_choices_forbidden")
        is True
        and review.get("selection_itself_consumes_token") is False,
        "AUTH_BLOCKED",
        "REVIEW_TOKEN_CARD_MISMATCH",
        repr(review),
    )
    allowed = card.get("verdict_contract", {}).get("invalid_or_blocked_states")
    _require(
        isinstance(allowed, list) and set(allowed) == set(INVALID_TERMINAL_STATES),
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
        if label == "future_review300_generated_files" or not isinstance(item, Mapping):
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

    identity = card["identity"]
    for artifact_group in ("predecessor_av1_artifacts", "predecessor_av1b_artifacts"):
        predecessor = identity.get(artifact_group)
        _require(
            isinstance(predecessor, Mapping) and predecessor,
            "AUTH_BLOCKED",
            "PREDECESSOR_PINS_MISSING",
            artifact_group,
        )
        for label, item in predecessor.items():
            _require(
                isinstance(item, Mapping)
                and isinstance(item.get("path"), str)
                and isinstance(item.get("sha256"), str),
                "AUTH_BLOCKED",
                "PREDECESSOR_PIN_INVALID",
                f"{artifact_group}.{label}: {item!r}",
            )
            _verify_absolute_artifact(item["path"], item["sha256"])
    for card_label in ("predecessor_av0_card", "predecessor_av1b_card"):
        item = identity.get(card_label, {})
        _require(
            isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha256"), str),
            "AUTH_BLOCKED",
            "PREDECESSOR_CARD_PIN_INVALID",
            card_label,
        )
        _verify_pin(item["path"], item["sha256"])
    for root_label in ("predecessor_av1_root", "predecessor_av1b_root"):
        predecessor_root = Path(identity[root_label])
        _require(
            predecessor_root.is_dir()
            and not predecessor_root.is_symlink()
            and stat.S_IMODE(predecessor_root.stat().st_mode) == 0o555,
            "ENV_BLOCKED",
            "PREDECESSOR_ROOT_NOT_FROZEN",
            str(predecessor_root),
        )
    head = review.get("latest_token_state_head", {})
    _require(
        isinstance(head, Mapping)
        and head.get("typed_state") == "REVIEW_ONCE"
        and head.get("token_state") == "UNCONSUMED",
        "AUTH_BLOCKED",
        "LATEST_TOKEN_HEAD_STATE_MISMATCH",
        repr(head),
    )
    _verify_absolute_artifact(str(head.get("path")), str(head.get("sha256")))
    for support_name in ("supporting_raw_evidence", "supporting_freeze_receipt"):
        support = head.get(support_name, {})
        _verify_absolute_artifact(str(support.get("path")), str(support.get("sha256")))
    return card


def _validate_authority(
    authority: Mapping[str, Any],
    quartet: Mapping[str, str],
    card: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    operation: str,
) -> dict[str, Any]:
    _require(
        authority.get("choice_statement") == EXPECTED_SELECTION
        and authority.get("choice_statement_sha256")
        == EXPECTED_SELECTION_AUTHORITY_SHA256
        and authority.get("card_freeze_statement_sha256")
        == EXPECTED_CARD_FREEZE_AUTHORITY_SHA256
        and isinstance(authority.get("card_freeze_statement"), str)
        and _sha256_bytes(authority["card_freeze_statement"].encode("utf-8"))
        == EXPECTED_CARD_FREEZE_AUTHORITY_SHA256,
        "AUTH_BLOCKED",
        "CONFIG_CHOICE_OR_CARD_FREEZE_AUTHORITY_MISMATCH",
        repr(authority),
    )
    _require_exact_keys(
        envelope,
        {
            "schema",
            "operation",
            "source_authority_statement",
            "source_authority_statement_sha256",
            "execution_authority_statement",
            "execution_authority_statement_sha256",
            "claim_only_recovery_authority_statement",
            "claim_only_recovery_authority_statement_sha256",
            "claim_only_recovery_caller_transcript_event",
            "claim_only_recovery_caller_transcript_event_sha256",
            "claim_only_recovery_platform_pin",
            "claim_only_recovery_platform_pin_sha256",
        },
        "runtime authority envelope",
    )
    _require(
        envelope.get("schema") == AUTHORITY_ENVELOPE_SCHEMA
        and envelope.get("operation") == operation,
        "AUTH_BLOCKED",
        "AUTHORITY_ENVELOPE_IDENTITY_MISMATCH",
        repr({"schema": envelope.get("schema"), "operation": envelope.get("operation")}),
    )
    source = envelope.get("source_authority_statement")
    _require(
        isinstance(source, str)
        and source == authority.get("source_statement")
        and len(source.encode("utf-8")) == EXPECTED_SOURCE_AUTHORITY_UTF8_BYTES
        and _sha256_bytes(source.encode("utf-8")) == EXPECTED_SOURCE_AUTHORITY_SHA256
        and authority.get("source_statement_sha256") == EXPECTED_SOURCE_AUTHORITY_SHA256
        and authority.get("source_statement_utf8_bytes")
        == EXPECTED_SOURCE_AUTHORITY_UTF8_BYTES
        and envelope.get("source_authority_statement_sha256")
        == EXPECTED_SOURCE_AUTHORITY_SHA256,
        "AUTH_BLOCKED",
        "SOURCE_AUTHORITY_PIN_MISMATCH",
        "source statement bytes or pin differ",
    )
    execution = envelope.get("execution_authority_statement")
    expected_execution = _expected_execution_authority_statement(quartet, card)
    execution_sha256 = _sha256_bytes(expected_execution.encode("utf-8"))
    _require(
        execution == expected_execution
        and envelope.get("execution_authority_statement_sha256") == execution_sha256
        and authority.get("execution_statement")
        == "FUTURE_RUNTIME_BIND_VERBATIM_EXECUTION_AUTHORITY_FROM_STDIN"
        and authority.get("execution_statement_sha256")
        == "FUTURE_RUNTIME_BIND_SHA256_UTF8_WITHOUT_TRAILING_NEWLINE",
        "AUTH_BLOCKED",
        "EXECUTION_AUTHORITY_EXACT_MATCH_FAILED",
        f"expected_sha256={execution_sha256}",
    )
    _require(
        authority.get("statement_sha256_rule")
        == "SHA256_UTF8_WITHOUT_TRAILING_NEWLINE",
        "AUTH_BLOCKED",
        "AUTHORITY_INTERPRETATION_MISMATCH",
        repr(authority),
    )
    source_capabilities = authority.get("source_phase_capabilities", {})
    future_capabilities = authority.get("future_execution_required_capabilities", {})
    _require(
        isinstance(source_capabilities, Mapping)
        and source_capabilities.get("exclusive_four_path_create") is True
        and source_capabilities.get("static_json_ast_schema_sha_validation") is True
        and source_capabilities.get("chmod_four_paths_0444") is True
        and all(
            source_capabilities.get(name) is False
            for name in (
                "run_or_token_namespace_create",
                "root_create",
                "claim_create_or_consume",
                "gpu_or_cuda_or_jit",
                "torch_vendor_or_model_import",
                "model_or_test_execute",
                "optimizer_or_parameter_update",
                "synthetic_training",
                "real_data",
                "checkpoint",
            )
        )
        and isinstance(future_capabilities, Mapping)
        and all(
            future_capabilities.get(name) is True
            for name in (
                "exact_four_frozen_source_pins",
                "exact_run_and_token_namespace_parents",
                "exact_root_and_nonce",
                "exact_gpu_uuid",
                "single_gpu_cuda_triton_jit",
                "vendor_autograd_forward_backward_local_jvp",
                "two_arm_300_step_adamw_updates",
                "synthetic_nonformal_training",
                "lightweight_tests_and_diagnostics",
                "provisional_freeze_and_atomic_claim",
            )
        )
        and future_capabilities.get("real_data") is False
        and future_capabilities.get("checkpoint_load_or_save") is False
        and future_capabilities.get(
            "full_2b_formal_admission_deploy_av2_or_global_stage3"
        )
        is False,
        "AUTH_BLOCKED",
        "CONFIG_CAPABILITY_ENVELOPE_MISMATCH",
        "source/future capability booleans",
    )
    return {
        "source_authority_statement": source,
        "source_authority_statement_sha256": EXPECTED_SOURCE_AUTHORITY_SHA256,
        "execution_authority_statement": execution,
        "execution_authority_statement_sha256": execution_sha256,
        "claim_only_recovery_authority_statement": envelope.get(
            "claim_only_recovery_authority_statement"
        ),
        "claim_only_recovery_authority_statement_sha256": envelope.get(
            "claim_only_recovery_authority_statement_sha256"
        ),
        "claim_only_recovery_caller_transcript_event": envelope.get(
            "claim_only_recovery_caller_transcript_event"
        ),
        "claim_only_recovery_caller_transcript_event_sha256": envelope.get(
            "claim_only_recovery_caller_transcript_event_sha256"
        ),
        "claim_only_recovery_platform_pin": envelope.get(
            "claim_only_recovery_platform_pin"
        ),
        "claim_only_recovery_platform_pin_sha256": envelope.get(
            "claim_only_recovery_platform_pin_sha256"
        ),
    }


def _validate_config(
    config: Mapping[str, Any],
    config_argument: Path,
    card: Mapping[str, Any],
    quartet: Mapping[str, str],
    envelope: Mapping[str, Any],
    *,
    operation: str,
) -> dict[str, Any]:
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
            "review_token",
            "artifacts",
        },
        "config",
    )
    _require(
        config.get("schema") == "cach.av1b.vendor_gdn_bridge_review300.config.v1",
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
    generated = identity.get("generated_paths", {})
    _require(
        isinstance(generated, Mapping)
        and generated.get("bridge", {}).get("path") == EXPECTED_BRIDGE
        and generated.get("config", {}).get("path") == EXPECTED_CONFIG
        and generated.get("runner", {}).get("path") == EXPECTED_RUNNER
        and generated.get("test", {}).get("path") == EXPECTED_TEST
        and generated.get("bridge", {}).get("sha256")
        == "SOURCE_MATERIALIZATION_BIND_FULL_SHA256"
        and generated.get("config", {}).get("sha256")
        == "SELF_OUT_OF_BAND_FULL_LITERAL_SHA256"
        and generated.get("runner", {}).get("sha256")
        == "SOURCE_MATERIALIZATION_BIND_FULL_SHA256"
        and generated.get("test", {}).get("sha256")
        == "SOURCE_MATERIALIZATION_BIND_FULL_SHA256",
        "AUTH_BLOCKED",
        "CONFIG_GENERATED_PATH_CONTRACT_MISMATCH",
        repr(generated),
    )
    expected_closure = {
        label: item
        for label, item in card["source_pins"].items()
        if label != "future_review300_generated_files"
        and isinstance(item, Mapping)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("sha256"), str)
    }
    _require(
        identity.get("source_pins") == expected_closure,
        "AUTH_BLOCKED",
        "CONFIG_SOURCE_CLOSURE_MISMATCH",
        "identity.source_pins must exactly match the card's pinned predecessor closure",
    )
    authority_bindings = _validate_authority(
        config["authority"], quartet, card, envelope, operation=operation
    )
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
        and runtime.get("expected_environment")
        == card["launcher_contract"]["expected_environment"]
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
        "write_optimizer_state", "reference_parameter_scope",
        "candidate_parameter_scope", "unlisted_trainable_parameter_allowed",
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
        "completion_rule", "budget_failure_state", "wall_budget_source",
    ):
        _require(
            budget.get(name) == card_budget.get(name),
            "AUTH_BLOCKED", "CONFIG_CARD_SECTION_MISMATCH", f"budget.{name}",
        )
    metrics = config["metrics"]
    card_metrics = card["metric_contract"]
    for name in (
        "accumulator_dtype", "epsilon", "train_loss_formula",
        "primary_horizon_weights", "counterfactual_delta_nmse_formula",
        "loss_drop_rel_formula", "shuffle_gap_rel_formula",
        "no_action_gap_rel_formula", "candidate_gain_vs_reference_formula",
        "seam_grad_rms_step0_formula", "seam_update_rms_final_formula",
        "last_block_local_seam_parameter_jvp_rms_theta0_formula",
        "last_block_local_action_condition_jvp_rms_step300_formula",
        "prefix_invariance_max_abs_formula", "final_step_only",
        "best_step_forbidden",
    ):
        _require(
            metrics.get(name) == card_metrics.get(name),
            "AUTH_BLOCKED", "CONFIG_CARD_SECTION_MISMATCH", f"metrics.{name}",
        )
    _require(
        optimizer.get("optimizer_steps_per_arm") == 300
        and optimizer.get("metric_steps") == [0, 300]
        and optimizer.get("loss_trace_points_per_arm") == 301
        and budget.get("max_optimizer_steps_total") == 600
        and budget.get("exact_optimizer_steps_per_arm") == 300
        and budget.get("max_wall_seconds_total") == 1350
        and config["recipe"].get("task_recipe_sha256")
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
    review = config["review_token"]
    _require(
        isinstance(review, Mapping)
        and review.get("token_id") == EXPECTED_REVIEW_TOKEN
        and review.get("state_before_execution") == "UNCONSUMED"
        and review.get("selected_choice") == EXPECTED_SELECTION
        and review.get("rejected_choice") == EXPECTED_REJECTED_SELECTION
        and review.get("both_choices_forbidden") is True
        and review.get("canonical_claim_path") == str(EXPECTED_REVIEW_CLAIM)
        and review.get("latest_state_head_path")
        == card["review_token"]["latest_token_state_head"]["path"]
        and review.get("latest_state_head_sha256")
        == card["review_token"]["latest_token_state_head"]["sha256"]
        and review.get("provisional_root_typed_state") == VALID_PROVISIONAL_STATE
        and review.get("claim_atomic_create_flags")
        == ["O_CREAT", "O_EXCL", "O_NOFOLLOW"]
        and review.get("claim_file_final_mode") == "0444"
        and review.get("ledger_leaf_final_mode") == "0555"
        and review.get("ledger_namespace_paths_in_creation_order")
        == [str(path) for path in EXPECTED_TOKEN_NAMESPACE_PATHS],
        "AUTH_BLOCKED",
        "CONFIG_REVIEW_TOKEN_CONTRACT_MISMATCH",
        repr(review),
    )
    _require(
        artifacts.get("run_namespace_paths_in_creation_order")
        == [str(path) for path in EXPECTED_ROOT_NAMESPACE_PATHS]
        and artifacts.get("namespace_creation_mode") == "02775"
        and artifacts.get("namespace_expected_owner") == "zch:sharegrp"
        and artifacts.get("provisional_result_typed_state")
        == VALID_PROVISIONAL_STATE
        and artifacts.get("provisional_result_must_not_publish_final_verdict") is True
        and artifacts.get("freeze_root_before_claim") is True,
        "AUTH_BLOCKED",
        "CONFIG_NAMESPACE_OR_PROVISIONAL_CONTRACT_MISMATCH",
        repr(artifacts),
    )
    return authority_bindings


def _path_absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    return False


def _owner_group(path: Path) -> tuple[str, str]:
    info = path.lstat()
    try:
        owner = pwd.getpwuid(info.st_uid).pw_name
        group = grp.getgrgid(info.st_gid).gr_name
    except KeyError as exc:
        raise PreflightError(
            "ENV_BLOCKED", "NAMESPACE_OWNER_LOOKUP_FAILED", f"{path}: {exc}"
        ) from exc
    return owner, group


def _validate_directory_chain(path: Path) -> None:
    _require(path.is_absolute(), "AUTH_BLOCKED", "ABSOLUTE_PATH_REQUIRED", str(path))
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise PreflightError(
                "ENV_BLOCKED", "DIRECTORY_CHAIN_MISSING", str(current)
            ) from exc
        _require(
            stat.S_ISDIR(info.st_mode)
            and not stat.S_ISLNK(info.st_mode)
            and not current.is_symlink(),
            "ENV_BLOCKED",
            "DIRECTORY_CHAIN_SYMLINK_OR_NON_DIRECTORY",
            str(current),
        )
    _require(
        path.resolve(strict=True) == path,
        "ENV_BLOCKED",
        "DIRECTORY_CHAIN_RESOLUTION_MISMATCH",
        str(path),
    )


def _validate_namespace_directory(path: Path, mode: int = NAMESPACE_MODE) -> None:
    _validate_directory_chain(path)
    info = path.lstat()
    owner, group = _owner_group(path)
    _require(
        stat.S_IMODE(info.st_mode) == mode
        and owner == EXPECTED_NAMESPACE_OWNER
        and group == EXPECTED_NAMESPACE_GROUP,
        "ENV_BLOCKED",
        "NAMESPACE_MODE_OR_OWNER_MISMATCH",
        (
            f"{path}: mode={stat.S_IMODE(info.st_mode):04o} "
            f"owner={owner}:{group} expected={mode:04o} "
            f"{EXPECTED_NAMESPACE_OWNER}:{EXPECTED_NAMESPACE_GROUP}"
        ),
    )


def _assert_absent(path: Path, token: str) -> None:
    _require(_path_absent(path), "AUTH_BLOCKED", token, str(path))


def _validate_environment_common(root_argument: Path, nonce: str) -> None:
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
    _validate_namespace_directory(EXPECTED_ROOT_NAMESPACE_BASE)
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
    _assert_absent(EXPECTED_REVIEW_CLAIM, "REVIEW_TOKEN_ALREADY_CLAIMED")


def _static_torch_version_metadata() -> dict[str, str]:
    try:
        torch_distribution = importlib_metadata.distribution("torch")
        torch_distribution_version = torch_distribution.version
        triton_distribution_version = importlib_metadata.version("triton")
    except importlib_metadata.PackageNotFoundError as exc:
        raise PreflightError(
            "ENV_BLOCKED", "RUNTIME_DISTRIBUTION_METADATA_MISSING", str(exc)
        ) from exc
    _require(
        torch_distribution_version
        in {EXPECTED_TORCH_DISTRIBUTION_VERSION, EXPECTED_TORCH_VERSION}
        and triton_distribution_version == EXPECTED_TRITON_VERSION,
        "ENV_BLOCKED",
        "RUNTIME_DISTRIBUTION_VERSION_MISMATCH",
        repr(
            {
                "torch": torch_distribution_version,
                "triton": triton_distribution_version,
            }
        ),
    )
    version_path = Path(torch_distribution.locate_file("torch/version.py"))
    try:
        version_info = version_path.lstat()
        source = version_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(version_path), mode="exec")
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise PreflightError(
            "ENV_BLOCKED", "TORCH_STATIC_VERSION_METADATA_UNREADABLE", str(exc)
        ) from exc
    _require(
        stat.S_ISREG(version_info.st_mode)
        and not stat.S_ISLNK(version_info.st_mode)
        and not version_path.is_symlink(),
        "ENV_BLOCKED",
        "TORCH_STATIC_VERSION_METADATA_UNSAFE",
        str(version_path),
    )
    assignments: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {"__version__", "cuda"}:
                try:
                    assignments[target.id] = ast.literal_eval(value_node)
                except (ValueError, TypeError):
                    assignments[target.id] = None
    _require(
        assignments.get("__version__") == EXPECTED_TORCH_VERSION
        and assignments.get("cuda") == EXPECTED_CUDA_VERSION,
        "ENV_BLOCKED",
        "TORCH_STATIC_RUNTIME_VERSION_MISMATCH",
        repr(assignments),
    )
    return {
        "torch_distribution_version": torch_distribution_version,
        "torch_static_runtime_version": str(assignments["__version__"]),
        "torch_static_cuda_version": str(assignments["cuda"]),
        "torch_version_metadata_path": str(version_path),
        "torch_version_metadata_sha256": _sha256_file(version_path),
        "triton_distribution_version": triton_distribution_version,
    }


def _validate_runtime_toolchain_preflight() -> dict[str, Any]:
    python_version = platform.python_version()
    _require(
        python_version == EXPECTED_PYTHON_VERSION,
        "ENV_BLOCKED",
        "PYTHON_VERSION_MISMATCH",
        f"expected={EXPECTED_PYTHON_VERSION} observed={python_version}",
    )
    package_metadata = _static_torch_version_metadata()
    nvcc = shutil.which("nvcc")
    _require(nvcc is not None, "ENV_BLOCKED", "NVCC_MISSING", "nvcc not found")
    nvcc_path = Path(nvcc).resolve(strict=True)
    _require(
        nvcc_path.is_file(),
        "ENV_BLOCKED",
        "NVCC_NOT_REGULAR",
        str(nvcc_path),
    )
    nvcc_result = _run_command([str(nvcc_path), "--version"])
    nvcc_text = f"{nvcc_result.stdout}\n{nvcc_result.stderr}"
    _require(
        nvcc_result.returncode == 0
        and f"release {EXPECTED_CUDA_VERSION}" in nvcc_text
        and f"V{EXPECTED_NVCC_VERSION}" in nvcc_text,
        "ENV_BLOCKED",
        "NVCC_VERSION_MISMATCH",
        nvcc_text.strip(),
    )
    return {
        "python_version": python_version,
        **package_metadata,
        "cuda_toolkit_version_from_nvcc": EXPECTED_CUDA_VERSION,
        "nvcc_version": EXPECTED_NVCC_VERSION,
        "nvcc_path": str(nvcc_path),
        "nvcc_sha256": _sha256_file(nvcc_path),
    }


def _validate_environment_execute(
    root_argument: Path, nonce: str, card: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_environment_common(root_argument, nonce)
    _assert_absent(root_argument, "ROOT_ALREADY_EXISTS")
    for path in (*EXPECTED_ROOT_NAMESPACE_PATHS, *EXPECTED_TOKEN_NAMESPACE_PATHS):
        _assert_absent(path, "NAMESPACE_PREEXISTED_OR_RACED")
    minimum_free = int(card["root_contract"]["minimum_free_bytes_before_create"])
    free_bytes = shutil.disk_usage(EXPECTED_ROOT_NAMESPACE_BASE).free
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
    return _validate_runtime_toolchain_preflight()


def _validate_environment_recovery(root_argument: Path, nonce: str) -> None:
    _validate_environment_common(root_argument, nonce)
    for path in EXPECTED_ROOT_NAMESPACE_PATHS:
        _validate_namespace_directory(path)
    _validate_directory_chain(root_argument)
    _require(
        stat.S_IMODE(root_argument.lstat().st_mode) == 0o555,
        "AUTH_BLOCKED",
        "RECOVERY_ROOT_NOT_FROZEN",
        str(root_argument),
    )
    for path in EXPECTED_TOKEN_NAMESPACE_PATHS:
        if not _path_absent(path):
            _validate_namespace_directory(path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"directory descriptor required: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_create_namespace(paths: Sequence[Path]) -> None:
    for path in paths:
        _assert_absent(path, "NAMESPACE_PREEXISTED_OR_RACED")
    for path in paths:
        try:
            os.mkdir(path, NAMESPACE_MODE)
        except FileExistsError as exc:
            raise PreflightError(
                "AUTH_BLOCKED", "NAMESPACE_UNEXPECTED_EEXIST_OR_RACE", str(path)
            ) from exc
        os.chmod(path, NAMESPACE_MODE, follow_symlinks=False)
        info = path.lstat()
        owner, group = _owner_group(path)
        _require(
            stat.S_ISDIR(info.st_mode)
            and not stat.S_ISLNK(info.st_mode)
            and path.resolve(strict=True) == path
            and stat.S_IMODE(info.st_mode) == NAMESPACE_MODE
            and owner == EXPECTED_NAMESPACE_OWNER
            and group == EXPECTED_NAMESPACE_GROUP,
            "ENV_BLOCKED",
            "NAMESPACE_COMPONENT_UNSAFE_AFTER_EXCLUSIVE_CREATE",
            str(path),
        )
        _fsync_directory(path)
        _fsync_directory(path.parent)
    _require(
        paths
        and (
            Path(paths[-1]) == EXPECTED_ROOT.parent
            or Path(paths[-1]) == EXPECTED_REVIEW_CLAIM.parent
        ),
        "AUTH_BLOCKED",
        "NAMESPACE_DERIVATION_MISMATCH",
        repr([str(path) for path in paths]),
    )


def _recovery_create_or_validate_namespace(paths: Sequence[Path]) -> None:
    """Claim-only recovery may resume after durable parent creation.

    Existing entries are accepted only when they are the exact card-declared
    real directories, still 02775 and owned by zch:sharegrp.  An EEXIST after
    an observed absence remains a forbidden race.
    """

    for path in paths:
        if not _path_absent(path):
            _validate_namespace_directory(path)
            continue
        try:
            os.mkdir(path, NAMESPACE_MODE)
        except FileExistsError as exc:
            raise PreflightError(
                "AUTH_BLOCKED", "RECOVERY_NAMESPACE_UNEXPECTED_EEXIST_RACE", str(path)
            ) from exc
        os.chmod(path, NAMESPACE_MODE, follow_symlinks=False)
        _validate_namespace_directory(path)
        _fsync_directory(path)
        _fsync_directory(path.parent)
    _require(
        paths and Path(paths[-1]) == EXPECTED_REVIEW_CLAIM.parent,
        "AUTH_BLOCKED",
        "RECOVERY_NAMESPACE_DERIVATION_MISMATCH",
        repr([str(path) for path in paths]),
    )


def _token_namespace_snapshot() -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for path in EXPECTED_TOKEN_NAMESPACE_PATHS:
        try:
            info = path.lstat()
        except FileNotFoundError:
            snapshot.append({"path": str(path), "state": "ABSENT"})
            continue
        except OSError as exc:
            snapshot.append(
                {
                    "path": str(path),
                    "state": "ERROR",
                    "errno": exc.errno,
                    "error": type(exc).__name__,
                }
            )
            continue
        try:
            owner = pwd.getpwuid(info.st_uid).pw_name
        except KeyError:
            owner = f"uid:{info.st_uid}"
        try:
            group = grp.getgrgid(info.st_gid).gr_name
        except KeyError:
            group = f"gid:{info.st_gid}"
        snapshot.append(
            {
                "path": str(path),
                "state": "PRESENT",
                "kind": (
                    "directory"
                    if stat.S_ISDIR(info.st_mode)
                    else "symlink"
                    if stat.S_ISLNK(info.st_mode)
                    else "other"
                ),
                "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                "owner": owner,
                "group": group,
            }
        )
    return snapshot


def _ordinary_recovery_namespace_prefix_is_safe(
    snapshot: Sequence[Mapping[str, Any]],
) -> bool:
    if len(snapshot) != len(EXPECTED_TOKEN_NAMESPACE_PATHS):
        return False
    missing_suffix_started = False
    for expected_path, entry in zip(EXPECTED_TOKEN_NAMESPACE_PATHS, snapshot):
        if entry.get("path") != str(expected_path):
            return False
        state = entry.get("state")
        if state == "ABSENT":
            missing_suffix_started = True
            if set(entry) != {"path", "state"}:
                return False
            continue
        if missing_suffix_started or state != "PRESENT":
            return False
        if entry != {
            "path": str(expected_path),
            "state": "PRESENT",
            "kind": "directory",
            "mode": "2775",
            "owner": EXPECTED_NAMESPACE_OWNER,
            "group": EXPECTED_NAMESPACE_GROUP,
        }:
            return False
    return True


def _source_pins(card: Mapping[str, Any]) -> dict[str, str]:
    pins = {
        relative: _sha256_file(_regular_repo_file(relative))
        for relative in AV1B_CODE_PATHS
    }
    pins[EXPECTED_CARD] = EXPECTED_CARD_SHA256
    for item in card["source_pins"].values():
        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
            pins[item["path"]] = _sha256_file(_regular_repo_file(item["path"]))
    return dict(sorted(pins.items()))


def _verify_frozen_quartet() -> dict[str, str]:
    pins: dict[str, str] = {}
    for relative in AV1B_CODE_PATHS:
        path = _regular_repo_file(relative)
        info = path.lstat()
        _require(
            stat.S_IMODE(info.st_mode) == 0o444,
            "AUTH_BLOCKED",
            "MATERIALIZED_SOURCE_NOT_FROZEN_0444",
            f"{relative}: mode={stat.S_IMODE(info.st_mode):04o}",
        )
        pins[relative] = _sha256_file(path)
    return pins


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
            "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu",
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
            len(fields) == 6,
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
                    "driver_version": fields[3],
                    "memory_used_mib": int(fields[4]),
                    "utilization_percent": int(fields[5]),
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
        and selected["name"] == EXPECTED_GPU_NAME
        and selected["driver_version"] == EXPECTED_DRIVER_VERSION,
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


def _normalized_theta0_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize manifests to the card's per-arm/per-entry comparison tuple."""

    normalized: dict[str, list[dict[str, Any]]] = {}
    for arm in ("reference", "candidate"):
        arm_value = value.get(arm)
        if not isinstance(arm_value, Mapping):
            raise ValueError(f"theta0 manifest arm missing: {arm}")
        entries: list[dict[str, Any]] = []

        def visit(node: Any, kind: str | None, inferred_name: str | None) -> None:
            if isinstance(node, Mapping):
                lowered = {str(key).lower(): key for key in node}
                digest_key = next(
                    (
                        lowered[name]
                        for name in ("sha256", "byte_sha256", "byte_digest", "digest")
                        if name in lowered
                    ),
                    None,
                )
                if "shape" in lowered and "dtype" in lowered and digest_key is not None:
                    name_value = node.get(lowered.get("name", ""), inferred_name)
                    kind_value = node.get(lowered.get("kind", ""), kind)
                    digest = node[digest_key]
                    shape = node[lowered["shape"]]
                    dtype = node[lowered["dtype"]]
                    if (
                        not isinstance(name_value, str)
                        or kind_value not in {"parameter", "buffer"}
                        or not isinstance(shape, (list, tuple))
                        or not isinstance(dtype, str)
                        or not isinstance(digest, str)
                        or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)
                    ):
                        raise ValueError(
                            f"invalid theta0 entry arm={arm} name={name_value!r}"
                        )
                    entries.append(
                        {
                            "kind": kind_value,
                            "name": name_value,
                            "shape": list(shape),
                            "dtype": dtype,
                            "byte_sha256": digest,
                        }
                    )
                    return
                for key, child in node.items():
                    key_text = str(key)
                    key_lower = key_text.lower()
                    child_kind = kind
                    if "parameter" in key_lower:
                        child_kind = "parameter"
                    elif "buffer" in key_lower:
                        child_kind = "buffer"
                    elif key_lower == "candidate_only":
                        # The bridge's candidate-only manifest uses the
                        # compact parameter-only {names, entries, digest}
                        # form, rather than a parameters/buffers split.
                        child_kind = "parameter"
                    child_name = inferred_name
                    if isinstance(child, Mapping) and child_kind is not None:
                        child_name = key_text
                    visit(child, child_kind, child_name)
            elif isinstance(node, list):
                for child in node:
                    visit(child, kind, inferred_name)

        visit(arm_value, None, None)
        if not entries:
            raise ValueError(f"theta0 manifest arm has no normalized entries: {arm}")
        entries.sort(key=lambda item: (item["kind"], item["name"].encode("utf-8")))
        identities = [(item["kind"], item["name"]) for item in entries]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate normalized theta0 entry: {arm}")
        normalized[arm] = entries
    return {
        "schema": "cach.av1b_review300.normalized_theta0_manifest.v1",
        "arms": normalized,
    }


def _verify_theta0_predecessor_equivalence(
    current: Mapping[str, Any], card: Mapping[str, Any]
) -> tuple[str, bool]:
    predecessor_item = card["identity"]["predecessor_av1b_artifacts"][
        "theta0_manifests"
    ]
    _verify_absolute_artifact(predecessor_item["path"], predecessor_item["sha256"])
    try:
        predecessor = _duplicate_rejecting_json(Path(predecessor_item["path"]))
        normalized_current = _normalized_theta0_manifest(current)
        normalized_predecessor = _normalized_theta0_manifest(predecessor)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"theta0 normalized comparison rejected: {exc}") from exc
    equivalent = normalized_current == normalized_predecessor
    digest = _sha256_bytes(_canonical_json_bytes(normalized_current))
    return digest, equivalent


class RootLifecycle:
    def __init__(self, root: Path, context: Mapping[str, Any]):
        self.root = root
        self.context = dict(context)
        self.created = False
        self.finalized = False
        self._finalizing = False
        self.context_sha256: str | None = None

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
        os.chmod(self.root, 0o700, follow_symlinks=False)
        root_info = self.root.lstat()
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            raise RuntimeError("exclusive run root is not a real directory")
        os.mkdir(self.root / "triton_cache", 0o700)
        os.chmod(self.root / "triton_cache", 0o700, follow_symlinks=False)
        _fsync_directory(self.root / "triton_cache")
        _fsync_directory(self.root)
        _fsync_directory(self.root.parent)

    def publish_context(self) -> str:
        self.context_sha256 = self.publish_or_match_json("RUN_CONTEXT.json", self.context)
        return self.context_sha256

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
        _fsync_directory(self.root)
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
            "schema": "cach.av1b_review300.unavailable_artifact.v1",
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

    def _artifact_bindings(self) -> dict[str, dict[str, Any]]:
        bindings: dict[str, dict[str, Any]] = {}
        for name in REQUIRED_ROOT_ARTIFACTS:
            if name in {"RESULT.json", "FREEZE_RECEIPT.json"}:
                continue
            path = self.root / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise RuntimeError(f"required root artifact is unsafe: {name}")
            bindings[name] = {
                "path": name,
                "sha256": _sha256_file(path),
                "size_bytes": info.st_size,
            }
        return bindings

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
                    _fsync_directory(child)
        os.chmod(self.root, 0o555)
        _fsync_directory(self.root)
        _fsync_directory(self.root.parent)

    def finalize(
        self,
        *,
        state: str,
        validity_reasons: list[str],
        raw_evidence: Mapping[str, Any],
        source_pins_before: Mapping[str, str],
        source_pins_after: Mapping[str, str] | None,
        started_monotonic: float,
        verdict_inputs: Mapping[str, Any] | None = None,
        normalized_theta0_digest: str | None = None,
        normalized_theta0_predecessor_equivalent: bool | None = None,
    ) -> dict[str, Any]:
        if self.finalized:
            return {"exact_typed_state": state, "already_finalized": True}
        if self._finalizing:
            raise RuntimeError("recursive root finalization")
        self._finalizing = True
        if state not in INVALID_TERMINAL_STATES | {VALID_PROVISIONAL_STATE}:
            validity_reasons = ["UNLISTED_TERMINAL_STATE", *validity_reasons]
            state = "IMPLEMENTATION_INVALID"
        try:
            if state == VALID_PROVISIONAL_STATE:
                for name in REQUIRED_ROOT_ARTIFACTS:
                    if name in {"RAW_EVIDENCE.json", "RESULT.json", "FREEZE_RECEIPT.json"}:
                        continue
                    if not (self.root / name).is_file():
                        raise RuntimeError(f"valid provisional artifact missing: {name}")
                if verdict_inputs is None:
                    raise RuntimeError("valid provisional result requires verdict inputs")
                if source_pins_after != source_pins_before:
                    raise RuntimeError("source pins changed before provisional freeze")
            else:
                self._ensure_required_failure_artifacts(state, validity_reasons)
            raw_sha = self.publish_or_match_json("RAW_EVIDENCE.json", raw_evidence)
            artifact_bindings = self._artifact_bindings()
            run_context_binding = artifact_bindings["RUN_CONTEXT.json"]
            if self.context_sha256 is not None and (
                run_context_binding["sha256"] != self.context_sha256
            ):
                raise RuntimeError("RUN_CONTEXT changed before terminal freeze")
            _assert_absent(
                EXPECTED_REVIEW_CLAIM, "REVIEW_TOKEN_CLAIM_APPEARED_BEFORE_ROOT_FREEZE"
            )
            result = {
                "schema": "cach.av1b_review300.provisional_result.v1",
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
                    "STATEFUL_CHUNKWISE_FORWARD_BACKWARD_QUALIFICATION",
                    "REAL_DATA_OR_CLOSED_LOOP_CAPABILITY",
                    "FULL_2B",
                ],
                "validity_reason": validity_reasons,
                "selected_review_choice": EXPECTED_SELECTION,
                "rejected_review_choice": EXPECTED_REJECTED_SELECTION,
                "run_card_self_path_and_full_sha256": {
                    "path": EXPECTED_CARD,
                    "sha256": EXPECTED_CARD_SHA256,
                },
                "review_token_id_and_consumption_state": {
                    "token_id": EXPECTED_REVIEW_TOKEN,
                    "state": "UNCONSUMED",
                    "canonical_claim_path": str(EXPECTED_REVIEW_CLAIM),
                    "claim_path_absent_at_root_freeze": True,
                    "claim_path_written": False,
                },
                "materialized_source_pins_before_execution": dict(source_pins_before),
                "materialized_source_pins_after_execution": (
                    dict(source_pins_after) if source_pins_after is not None else None
                ),
                "source_pin_rehash_equal": source_pins_after == source_pins_before,
                "predecessor_av1b_bindings": {
                    "card": self.context.get("predecessor_av1b_card"),
                    "root": self.context.get("predecessor_av1b_root"),
                    "nonce": self.context.get("predecessor_av1b_nonce"),
                    "artifacts": self.context.get("predecessor_av1b_artifacts"),
                },
                "artifact_bindings": artifact_bindings,
                "raw_evidence_path": "RAW_EVIDENCE.json",
                "raw_evidence_sha256": raw_sha,
                "vendor_source_sha256": {
                    key: value
                    for key, value in source_pins_before.items()
                    if key.startswith("third_party/Sana/")
                },
                "authority_artifacts_and_chain": {
                    "choice_authority_artifact": {
                        "path": EXPECTED_CARD,
                        "sha256": EXPECTED_CARD_SHA256,
                    },
                    "source_and_execution_authority_artifact": {
                        "path": str(self.root / "RUN_CONTEXT.json"),
                        "sha256": run_context_binding["sha256"],
                    },
                    "selection_statement": EXPECTED_SELECTION,
                    "selection_statement_sha256": EXPECTED_SELECTION_AUTHORITY_SHA256,
                    "card_freeze_statement": self.context.get(
                        "card_freeze_statement"
                    ),
                    "card_freeze_statement_sha256": EXPECTED_CARD_FREEZE_AUTHORITY_SHA256,
                    "source_authority_statement": self.context.get(
                        "source_authority_statement"
                    ),
                    "source_authority_statement_sha256": (
                        EXPECTED_SOURCE_AUTHORITY_SHA256
                    ),
                    "execution_authority_statement": self.context.get(
                        "execution_authority_statement"
                    ),
                    "execution_authority_statement_sha256": self.context.get(
                        "execution_authority_statement_sha256"
                    ),
                    "statement_sha256_rule": "SHA256_UTF8_WITHOUT_TRAILING_NEWLINE",
                },
                "gpu_reservation": self.context.get("gpu_reservation"),
                "resolved_root": str(self.root),
                "run_nonce": EXPECTED_NONCE,
                "root_lifecycle_and_freeze_receipt": {
                    "exclusive_create": True,
                    "terminal_root_frozen": True,
                    "freeze_receipt_path": "FREEZE_RECEIPT.json",
                    "freeze_receipt_must_bind_this_result_full_sha256": True,
                },
                "fresh_initialization": {
                    "required": True,
                    "normalized_theta0_digest": normalized_theta0_digest,
                    "normalized_predecessor_equivalent": (
                        normalized_theta0_predecessor_equivalent
                    ),
                    "raw_manifest_file_sha_equality_required": False,
                },
                "final_operator_verdict_published": False,
                "claim_is_only_final_result_publication": True,
                "all_validity_flags_all_final_metrics_and_threshold_inputs": (
                    dict(verdict_inputs) if verdict_inputs is not None else None
                ),
                "forbidden_scope_flags": {
                    "real_data_used": False,
                    "checkpoint_loaded": False,
                    "checkpoint_saved": False,
                    "proxy_or_reference_fallback_used": False,
                    "full_2b_used": False,
                    "formal_evaluation_used": False,
                    "admission_used": False,
                    "deploy_used": False,
                    "av2_used": False,
                    "global_stage3_used": False,
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
                "schema": "cach.av1b_review300.freeze_receipt.v1",
                "resolved_root": str(self.root),
                "run_nonce": EXPECTED_NONCE,
                "terminal_state": state,
                "result_path": "RESULT.json",
                "result_sha256": result_sha,
                "run_context_path": "RUN_CONTEXT.json",
                "run_context_sha256": run_context_binding["sha256"],
                "raw_evidence_path": "RAW_EVIDENCE.json",
                "raw_evidence_sha256": raw_sha,
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
                "artifact_bindings": artifact_bindings,
            }
        finally:
            self._finalizing = False

    def emergency_finalize(self) -> None:
        if not self.created or self.finalized or self._finalizing:
            return
        try:
            reasons = ["PROCESS_EXITED_BEFORE_NORMAL_FINALIZATION"]
            raw_evidence = {
                "schema": "cach.av1b_review300.emergency_failure.v1",
                "proposed_typed_state": "IMPLEMENTATION_INVALID",
                "validity_reason": reasons,
            }
            self.finalize(
                state="IMPLEMENTATION_INVALID",
                validity_reasons=reasons,
                raw_evidence=raw_evidence,
                source_pins_before=self.context.get("source_pins", {}),
                source_pins_after=None,
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
                                "cach.av1b_review300.emergency_freeze_receipt.v1"
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
        "cach.av1b.vendor_gdn_bridge_review300.screen.v1"
    ):
        return ["SCREEN_SCHEMA_MISMATCH"]
    steps = int(card["experiment_contract"]["optimizer"]["optimizer_steps_per_arm"])
    if steps != 300:
        return ["CARD_REVIEW300_STEP_CONTRACT_MISMATCH"]
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
        "last_block_local_action_condition_jvp_rms_step300",
    )
    for name in finite_positive:
        value = diagnostics.get(name)
        if not _finite_number(value) or float(value) <= 1.0e-8:
            errors.append(f"DIAGNOSTIC_{name.upper()}_NOT_LIVE")
    metrics = screen.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("METRICS_INVALID")
    else:
        for name in (
            "primary_reference_step300",
            "primary_candidate_correct_step300",
            "primary_candidate_shuffle_step300",
            "primary_candidate_no_action_step300",
            "last_block_local_action_condition_jvp_rms_step300",
        ):
            if not _finite_number(metrics.get(name)):
                errors.append(f"FINAL_METRIC_{name.upper()}_INVALID")
        if (
            _finite_number(metrics.get("last_block_local_action_condition_jvp_rms_step300"))
            and _finite_number(
                diagnostics.get("last_block_local_action_condition_jvp_rms_step300")
            )
            and float(metrics["last_block_local_action_condition_jvp_rms_step300"])
            != float(diagnostics["last_block_local_action_condition_jvp_rms_step300"])
        ):
            errors.append("STEP300_ACTION_JVP_METRIC_DIAGNOSTIC_MISMATCH")
        if isinstance(trace, Mapping) and isinstance(trace.get("candidate"), list):
            candidate_trace = trace["candidate"]
            if len(candidate_trace) == 301 and all(
                _finite_number(item) for item in candidate_trace
            ):
                expected_drop = (
                    float(candidate_trace[0]) - float(candidate_trace[-1])
                ) / max(float(candidate_trace[0]), 1.0e-12)
                observed_drop = metrics.get("loss_drop_rel")
                if not _finite_number(observed_drop) or not math.isclose(
                    float(observed_drop), expected_drop, rel_tol=1.0e-12, abs_tol=1.0e-12
                ):
                    errors.append("LOSS_DROP_REL_TRACE_FORMULA_MISMATCH")
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
    jit = screen.get("jit_inventory")
    if (
        not isinstance(jit, Mapping)
        or jit.get("cuda_visible_devices") != EXPECTED_GPU_UUID
        or jit.get("cuda_compute_capability") != [9, 0]
        or jit.get("vendor_kernel_executed") is not True
        or not isinstance(jit.get("triton_cache_files"), list)
        or not isinstance(jit.get("triton_cache_file_count"), int)
        or int(jit.get("triton_cache_file_count", 0)) <= 0
    ):
        errors.append("CUDA_OR_JIT_INVENTORY_INVALID")
    for section_name in ("metrics", "diagnostics"):
        section = screen.get(section_name, {})
        if isinstance(section, Mapping) and any(
            ("step" + str(200)) in str(key).lower() for key in section
        ):
            errors.append(f"STALE_STEP200_LABEL_IN_{section_name.upper()}")
    return errors


def _evaluate_screen(
    screen: Mapping[str, Any], card: Mapping[str, Any]
) -> tuple[str, list[str], dict[str, Any] | None]:
    errors = _screen_evidence_errors(screen, card)
    if errors:
        return "IMPLEMENTATION_INVALID", errors, None
    validity = screen.get("validity")
    metrics = screen.get("metrics")
    if not isinstance(validity, Mapping) or not isinstance(metrics, Mapping):
        return "IMPLEMENTATION_INVALID", ["MISSING_VALIDITY_OR_METRICS"], None
    supplied = validity.get("reasons", [])
    reasons = [str(item) for item in supplied] if isinstance(supplied, list) else []
    if validity.get("implementation_valid") is not True:
        return "IMPLEMENTATION_INVALID", reasons or ["IMPLEMENTATION_VALIDITY_FAILED"], None
    if validity.get("data_valid") is not True:
        return "INVALID_RUN", reasons or ["DATA_VALIDITY_FAILED"], None
    if validity.get("numerics_valid") is not True or validity.get("finite_all") is not True:
        return "NUMERICAL_INVALID", reasons or ["NUMERICAL_VALIDITY_FAILED"], None
    if validity.get("all_card_validity_checks_pass") is not True:
        return "IMPLEMENTATION_INVALID", reasons or ["CARD_VALIDITY_CHECK_FAILED"], None

    try:
        values = {
            "seam_grad": _metric(metrics, "seam_grad_rms_step0"),
            "seam_update": _metric(metrics, "seam_update_rms_final"),
            "parameter_jvp": _metric(
                metrics, "last_block_local_seam_parameter_jvp_rms_theta0"
            ),
            "action_jvp": _metric(
                metrics, "last_block_local_action_condition_jvp_rms_step300"
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
        return "IMPLEMENTATION_INVALID", ["METRIC_CONTRACT_FAILED", str(exc)], None

    go = card["threshold_contract"]["operator_go_all_required"]
    go_checks = {
        "seam_grad": values["seam_grad"] > go["seam_grad_rms_step0_strictly_greater_than"],
        "seam_update": values["seam_update"] > go["seam_update_rms_final_strictly_greater_than"],
        "parameter_jvp": values["parameter_jvp"] > go["last_block_local_seam_parameter_jvp_rms_theta0_strictly_greater_than"],
        "action_jvp": values["action_jvp"] > go["last_block_local_action_condition_jvp_rms_step300_strictly_greater_than"],
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
    all_go = all(go_checks.values())
    all_strong_stop = all(stop_checks.values())
    if all_go and all_strong_stop:
        return (
            "IMPLEMENTATION_INVALID",
            ["GO_AND_STRONG_STOP_SIMULTANEOUSLY_TRUE"],
            None,
        )
    verdict_inputs = {
        "validity_flags": dict(validity),
        "all_final_metrics": dict(metrics),
        "go_boolean_inputs": go_checks,
        "strong_stop_boolean_inputs": stop_checks,
        "all_go_conditions": all_go,
        "all_strong_stop_conditions": all_strong_stop,
        "valid_neither": not all_go and not all_strong_stop,
        "final_step": 300,
        "best_step_selection": False,
    }
    return (
        VALID_PROVISIONAL_STATE,
        ["VALID_REVIEW_EVIDENCE_READY_FOR_PROVISIONAL_FREEZE"],
        verdict_inputs,
    )


def _final_verdict_from_inputs(verdict_inputs: Mapping[str, Any]) -> str:
    all_go = verdict_inputs.get("all_go_conditions")
    all_stop = verdict_inputs.get("all_strong_stop_conditions")
    neither = verdict_inputs.get("valid_neither")
    if (all_go, all_stop, neither) == (True, False, False):
        return "OPERATOR_GO"
    if (all_go, all_stop, neither) == (False, True, False):
        return "OPERATOR_STOP"
    if (all_go, all_stop, neither) == (False, False, True):
        return "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED"
    raise RuntimeError("frozen verdict booleans are not one-hot")


def _screen_artifacts(screen: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "RAW_METRICS.json": screen.get("metrics", {}),
        "LOSS_TRACE.json": screen.get("loss_trace", {}),
        "THETA0_MANIFESTS.json": screen.get("theta0_manifests", {}),
        "FINAL_PARAMETER_DIGESTS.json": screen.get("final_parameter_digests", {}),
        "JIT_INVENTORY.json": screen.get("jit_inventory", {}),
    }


def _validate_progress_events_300(events: Sequence[Mapping[str, Any]]) -> None:
    theta_indices = [
        index
        for index, event in enumerate(events)
        if event.get("event") == "theta0_manifests_ready"
    ]
    if len(theta_indices) != 1:
        raise RuntimeError("exactly one theta0_manifests_ready event is required")
    by_arm: dict[str, list[int]] = {
        "REF-GDN-CORRECTED": [],
        "CACH-A": [],
    }
    for index, event in enumerate(events):
        if event.get("event") != "optimizer_progress":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError("optimizer progress payload must be a mapping")
        arm = payload.get("arm")
        completed = payload.get("optimizer_steps_completed")
        if arm not in by_arm or not isinstance(completed, int) or isinstance(completed, bool):
            raise RuntimeError(f"optimizer progress event invalid: {event!r}")
        if index <= theta_indices[0]:
            raise RuntimeError("optimizer progress preceded theta0 manifest publication")
        by_arm[str(arm)].append(completed)
    expected = [1, 50, 100, 150, 200, 250, 300]
    if any(observed != expected for observed in by_arm.values()):
        raise RuntimeError(
            f"review300 progress checkpoints mismatch: {by_arm!r} expected={expected!r}"
        )


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
    if "version mismatch" in lowered or "torch cuda mismatch" in lowered:
        return "ENV_BLOCKED", ["RUNTIME_ENVIRONMENT_VERSION_MISMATCH", message]
    if isinstance(exc, (FloatingPointError, OverflowError)):
        return "NUMERICAL_INVALID", [type(exc).__name__, message]
    return "IMPLEMENTATION_INVALID", [type(exc).__name__, message]


def _canonical_json_file(path: Path) -> tuple[dict[str, Any], bytes, str]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or path.is_symlink():
        raise RuntimeError(f"canonical evidence file is unsafe: {path}")
    payload = path.read_bytes()
    value = _duplicate_rejecting_json(path)
    if payload != _canonical_json_bytes(value):
        raise RuntimeError(f"evidence is not canonical JSON: {path}")
    return value, payload, _sha256_bytes(payload)


def _verify_frozen_provisional_root(
    root: Path, current_source_pins: Mapping[str, str]
) -> dict[str, Any]:
    _validate_directory_chain(root)
    if stat.S_IMODE(root.lstat().st_mode) != 0o555:
        raise RuntimeError("provisional root is not frozen 0555")
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_info = directory_path.lstat()
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or stat.S_ISLNK(directory_info.st_mode)
            or stat.S_IMODE(directory_info.st_mode) != 0o555
        ):
            raise RuntimeError(f"frozen root directory unsafe: {directory_path}")
        for name in directory_names:
            child = directory_path / name
            if child.is_symlink():
                raise RuntimeError(f"symlink in frozen root: {child}")
        for name in file_names:
            child = directory_path / name
            info = child.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o444
            ):
                raise RuntimeError(f"frozen root file unsafe: {child}")
    for name in REQUIRED_ROOT_ARTIFACTS:
        if not (root / name).is_file():
            raise RuntimeError(f"required frozen artifact missing: {name}")

    result, _, result_sha = _canonical_json_file(root / "RESULT.json")
    raw, _, raw_sha = _canonical_json_file(root / "RAW_EVIDENCE.json")
    receipt, _, receipt_sha = _canonical_json_file(root / "FREEZE_RECEIPT.json")
    context, _, context_sha = _canonical_json_file(root / "RUN_CONTEXT.json")
    if (
        result.get("schema") != "cach.av1b_review300.provisional_result.v1"
        or result.get("exact_typed_state") != VALID_PROVISIONAL_STATE
        or result.get("final_operator_verdict_published") is not False
        or result.get("review_token_id_and_consumption_state", {}).get("state")
        != "UNCONSUMED"
    ):
        raise RuntimeError("frozen root is not a valid review provisional result")
    for forbidden_key in (
        "final_exact_typed_verdict",
        "final_operator_verdict",
        "token_claim_sha256",
        "stage_unlock",
    ):
        if forbidden_key in result:
            raise RuntimeError(f"provisional RESULT contains forbidden key: {forbidden_key}")
    if (
        result.get("run_card_self_path_and_full_sha256")
        != {"path": EXPECTED_CARD, "sha256": EXPECTED_CARD_SHA256}
        or result.get("resolved_root") != str(root)
        or result.get("run_nonce") != EXPECTED_NONCE
        or result.get("materialized_source_pins_before_execution")
        != dict(current_source_pins)
        or result.get("materialized_source_pins_after_execution")
        != dict(current_source_pins)
        or result.get("source_pin_rehash_equal") is not True
    ):
        raise RuntimeError("frozen result card/root/source binding mismatch")
    artifact_bindings = result.get("artifact_bindings")
    if not isinstance(artifact_bindings, Mapping):
        raise RuntimeError("frozen result artifact bindings missing")
    for name, binding in artifact_bindings.items():
        if (
            name not in REQUIRED_ROOT_ARTIFACTS
            or name in {"RESULT.json", "FREEZE_RECEIPT.json"}
            or not isinstance(binding, Mapping)
            or binding.get("path") != name
            or binding.get("sha256") != _sha256_file(root / name)
        ):
            raise RuntimeError(f"frozen artifact binding mismatch: {name}")
    if artifact_bindings.get("RAW_EVIDENCE.json", {}).get("sha256") != raw_sha:
        raise RuntimeError("RAW_EVIDENCE result binding mismatch")
    if (
        receipt.get("schema") != "cach.av1b_review300.freeze_receipt.v1"
        or receipt.get("terminal_state") != VALID_PROVISIONAL_STATE
        or receipt.get("result_sha256") != result_sha
        or receipt.get("run_context_sha256") != context_sha
        or receipt.get("raw_evidence_sha256") != raw_sha
        or receipt.get("immutable") is not True
        or receipt.get("required_artifacts_present") is not True
    ):
        raise RuntimeError("freeze receipt binding mismatch")
    expected_inventory = _recursive_inventory(
        root, frozenset({"FREEZE_RECEIPT.json"})
    )
    if receipt.get("inventory") != expected_inventory:
        raise RuntimeError("freeze receipt recursive inventory mismatch")
    authority = result.get("authority_artifacts_and_chain", {})
    if (
        authority.get("source_and_execution_authority_artifact", {}).get("sha256")
        != context_sha
        or context.get("source_pins") != dict(current_source_pins)
        or context.get("card_sha256") != EXPECTED_CARD_SHA256
        or context.get("resolved_root") != str(root)
        or context.get("run_nonce") != EXPECTED_NONCE
    ):
        raise RuntimeError("RUN_CONTEXT authority/source binding mismatch")
    verdict_inputs = result.get(
        "all_validity_flags_all_final_metrics_and_threshold_inputs"
    )
    if not isinstance(verdict_inputs, Mapping):
        raise RuntimeError("frozen verdict inputs missing")
    _final_verdict_from_inputs(verdict_inputs)
    if raw.get("proposed_typed_state") != VALID_PROVISIONAL_STATE:
        raise RuntimeError("RAW_EVIDENCE provisional state mismatch")
    return {
        "result": result,
        "result_sha256": result_sha,
        "raw_evidence": raw,
        "raw_evidence_sha256": raw_sha,
        "freeze_receipt": receipt,
        "freeze_receipt_sha256": receipt_sha,
        "run_context": context,
        "run_context_sha256": context_sha,
        "verdict_inputs": dict(verdict_inputs),
    }


def _recovery_platform_pin(frozen: Mapping[str, Any]) -> dict[str, Any]:
    source_pins = frozen["result"]["materialized_source_pins_after_execution"]
    return {
        "schema": "cach.av1b_review300.claim_recovery_platform_pin.v1",
        "canonical_host_alias": "H200",
        "observed_hostname": platform.node(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "card_sha256": EXPECTED_CARD_SHA256,
        "runner_sha256": source_pins[EXPECTED_RUNNER],
    }


def _expected_claim_only_recovery_statement(
    frozen: Mapping[str, Any], transcript_sha256: str, platform_pin_sha256: str
) -> str:
    return (
        "授权在 H200 严格按 AV1B_REVIEW300_RUN_CARD.json SHA256 "
        f"{EXPECTED_CARD_SHA256} 对冻结 root {EXPECTED_ROOT} 执行一次 claim-only "
        f"recovery：固定 RESULT SHA256 {frozen['result_sha256']}、RAW_EVIDENCE "
        f"SHA256 {frozen['raw_evidence_sha256']}、FREEZE_RECEIPT SHA256 "
        f"{frozen['freeze_receipt_sha256']}、RUN_CONTEXT SHA256 "
        f"{frozen['run_context_sha256']} 和 canonical claim {EXPECTED_REVIEW_CLAIM}；"
        f"绑定 caller claim-failure transcript SHA256 {transcript_sha256} 和 "
        f"recovery platform pin SHA256 {platform_pin_sha256}；"
        "仅允许只读复核冻结证据；canonical claim 必须不存在，token namespace 仅可为"
        "全 absent，或与 caller transcript 完全一致且由此前 claim failure 留下的安全"
        "前缀：每个既存项必须是卡片 exact path 的 real directory、02775 zch:sharegrp，"
        "缺失后缀才可按声明顺序 O_EXCL 创建；禁止修改 root、模型/测试执行、GPU/CUDA/JIT、"
        "参数更新、训练、真实数据、checkpoint 及自动重跑；任何非前缀、unsafe entry、"
        "unexpected EEXIST/race、claim 已存在、证据或 source pin 不一致均 fail-closed，"
        "不得删除或覆盖 partial claim。"
    )


def _validate_recovery_authority(
    authority: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    transcript = authority.get("claim_only_recovery_caller_transcript_event")
    if (
        not isinstance(transcript, Mapping)
        or transcript.get("schema")
        != "cach.av1b_review300.claim_failure_external_event.v1"
        or transcript.get("external_typed_state")
        != "FROZEN_PROVISIONAL_CLAIM_PENDING_RECOVERY_NO_VERDICT"
        or transcript.get("frozen_root") != str(EXPECTED_ROOT)
        or transcript.get("root_mutated_after_freeze") is not False
        or transcript.get("automatic_retry") is not False
        or transcript.get("ordinary_claim_only_recovery_allowed") is not True
        or transcript.get("canonical_claim_path") != str(EXPECTED_REVIEW_CLAIM)
        or transcript.get("canonical_claim_path_absent_after_failure") is not True
    ):
        raise PreflightError(
            "AUTH_BLOCKED",
            "CLAIM_ONLY_RECOVERY_CALLER_TRANSCRIPT_INVALID",
            repr(transcript),
        )
    transcript_namespace_snapshot = transcript.get(
        "token_namespace_state_after_failure"
    )
    current_namespace_snapshot = _token_namespace_snapshot()
    if (
        not isinstance(transcript_namespace_snapshot, list)
        or transcript_namespace_snapshot != current_namespace_snapshot
        or not _ordinary_recovery_namespace_prefix_is_safe(
            transcript_namespace_snapshot
        )
    ):
        raise PreflightError(
            "AUTH_BLOCKED",
            "CLAIM_ONLY_RECOVERY_NAMESPACE_TRANSCRIPT_MISMATCH",
            repr(
                {
                    "transcript": transcript_namespace_snapshot,
                    "current": current_namespace_snapshot,
                }
            ),
        )
    transcript_sha = _sha256_bytes(_canonical_json_bytes(dict(transcript)))
    if authority.get("claim_only_recovery_caller_transcript_event_sha256") != transcript_sha:
        raise PreflightError(
            "AUTH_BLOCKED",
            "CLAIM_ONLY_RECOVERY_CALLER_TRANSCRIPT_SHA_MISMATCH",
            transcript_sha,
        )
    platform_pin = authority.get("claim_only_recovery_platform_pin")
    expected_platform_pin = _recovery_platform_pin(frozen)
    platform_pin_sha = _sha256_bytes(_canonical_json_bytes(expected_platform_pin))
    if (
        platform_pin != expected_platform_pin
        or authority.get("claim_only_recovery_platform_pin_sha256")
        != platform_pin_sha
    ):
        raise PreflightError(
            "AUTH_BLOCKED",
            "CLAIM_ONLY_RECOVERY_PLATFORM_PIN_MISMATCH",
            f"expected_sha256={platform_pin_sha}",
        )
    statement = authority.get("claim_only_recovery_authority_statement")
    expected = _expected_claim_only_recovery_statement(
        frozen, transcript_sha, platform_pin_sha
    )
    expected_sha = _sha256_bytes(expected.encode("utf-8"))
    if (
        statement != expected
        or authority.get("claim_only_recovery_authority_statement_sha256")
        != expected_sha
    ):
        raise PreflightError(
            "AUTH_BLOCKED",
            "CLAIM_ONLY_RECOVERY_AUTHORITY_EXACT_MATCH_FAILED",
            f"expected_sha256={expected_sha}",
        )
    return {
        **dict(authority),
        "claim_only_recovery_authority_statement": statement,
        "claim_only_recovery_authority_statement_sha256": expected_sha,
        "claim_only_recovery_caller_transcript_event": dict(transcript),
        "claim_only_recovery_caller_transcript_event_sha256": transcript_sha,
        "claim_only_recovery_platform_pin": expected_platform_pin,
        "claim_only_recovery_platform_pin_sha256": platform_pin_sha,
    }


def _claim_payload(
    frozen: Mapping[str, Any],
    card: Mapping[str, Any],
    authority: Mapping[str, Any],
    final_verdict: str,
) -> dict[str, Any]:
    if final_verdict not in FINAL_REVIEW_STATES:
        raise RuntimeError(f"invalid final review state: {final_verdict}")
    result = frozen["result"]
    context = frozen["run_context"]
    return {
        "schema": CLAIM_SCHEMA,
        "token_id": EXPECTED_REVIEW_TOKEN,
        "token_state": "CONSUMED",
        "selected_choice": EXPECTED_SELECTION,
        "rejected_choice": EXPECTED_REJECTED_SELECTION,
        "card": {"path": EXPECTED_CARD, "sha256": EXPECTED_CARD_SHA256},
        "predecessor_av1b": {
            "card": card["identity"]["predecessor_av1b_card"],
            "root": card["identity"]["predecessor_av1b_root"],
            "nonce": card["identity"]["predecessor_av1b_nonce"],
            "artifacts": card["identity"]["predecessor_av1b_artifacts"],
            "latest_token_state_head": card["review_token"][
                "latest_token_state_head"
            ],
        },
        "materialized_review300_source_pins": result[
            "materialized_source_pins_after_execution"
        ],
        "review_root": {"path": str(EXPECTED_ROOT), "nonce": EXPECTED_NONCE},
        "fresh_initialization": result["fresh_initialization"],
        "frozen_provisional_artifacts": {
            "result": {
                "path": str(EXPECTED_ROOT / "RESULT.json"),
                "sha256": frozen["result_sha256"],
            },
            "raw_evidence": {
                "path": str(EXPECTED_ROOT / "RAW_EVIDENCE.json"),
                "sha256": frozen["raw_evidence_sha256"],
            },
            "freeze_receipt": {
                "path": str(EXPECTED_ROOT / "FREEZE_RECEIPT.json"),
                "sha256": frozen["freeze_receipt_sha256"],
            },
        },
        "authority_artifacts": {
            "choice_card": {"path": EXPECTED_CARD, "sha256": EXPECTED_CARD_SHA256},
            "run_context": {
                "path": str(EXPECTED_ROOT / "RUN_CONTEXT.json"),
                "sha256": frozen["run_context_sha256"],
            },
        },
        "authority_statements": {
            "choice_statement": card["authority"]["selection_statement"],
            "choice_statement_sha256": card["authority"][
                "selection_statement_sha256"
            ],
            "card_freeze_statement": card["authority"]["card_freeze_statement"],
            "card_freeze_statement_sha256": card["authority"][
                "card_freeze_statement_sha256"
            ],
            "source_authority_statement": context["source_authority_statement"],
            "source_authority_statement_sha256": context[
                "source_authority_statement_sha256"
            ],
            "execution_authority_statement": context["execution_authority_statement"],
            "execution_authority_statement_sha256": context[
                "execution_authority_statement_sha256"
            ],
            "claim_only_recovery_authority_statement": authority.get(
                "claim_only_recovery_authority_statement"
            ),
            "claim_only_recovery_authority_statement_sha256": authority.get(
                "claim_only_recovery_authority_statement_sha256"
            ),
            "claim_only_recovery_caller_transcript_event": authority.get(
                "claim_only_recovery_caller_transcript_event"
            ),
            "claim_only_recovery_caller_transcript_event_sha256": authority.get(
                "claim_only_recovery_caller_transcript_event_sha256"
            ),
            "claim_only_recovery_platform_pin": authority.get(
                "claim_only_recovery_platform_pin"
            ),
            "claim_only_recovery_platform_pin_sha256": authority.get(
                "claim_only_recovery_platform_pin_sha256"
            ),
            "statement_sha256_rule": "SHA256_UTF8_WITHOUT_TRAILING_NEWLINE",
        },
        "final_exact_typed_verdict": final_verdict,
        "all_final_metrics": frozen["verdict_inputs"]["all_final_metrics"],
        "final_verdict_boolean_inputs": frozen["verdict_inputs"],
        "claim_publication": {
            "canonical_path": str(EXPECTED_REVIEW_CLAIM),
            "serialization": (
                "CANONICAL_UTF8_JSON_SORTED_KEYS_COMPACT_SEPARATORS_"
                "SINGLE_TRAILING_NEWLINE_NO_BOM"
            ),
            "atomic_flags": ["O_CREAT", "O_EXCL", "O_NOFOLLOW"],
            "file_mode": "0444",
            "ledger_leaf_mode": "0555",
            "root_result_remains_provisional_and_immutable": True,
        },
        "formal_admission_evidence": False,
        "global_stage3_effect": "NONE",
    }


def _atomic_publish_claim(payload: Mapping[str, Any], *, recovery: bool) -> str:
    _assert_absent(EXPECTED_REVIEW_CLAIM, "REVIEW_TOKEN_ALREADY_CLAIMED")
    if recovery:
        _recovery_create_or_validate_namespace(EXPECTED_TOKEN_NAMESPACE_PATHS)
    else:
        _exclusive_create_namespace(EXPECTED_TOKEN_NAMESPACE_PATHS)
    _assert_absent(EXPECTED_REVIEW_CLAIM, "CLAIM_PATH_RACED_BEFORE_OPEN")
    canonical = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    opened_by_this_process = False
    try:
        descriptor = os.open(EXPECTED_REVIEW_CLAIM, flags, 0o600)
        opened_by_this_process = True
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError("claim descriptor is not regular")
            offset = 0
            while offset < len(canonical):
                written = os.write(descriptor, canonical[offset:])
                if written <= 0:
                    raise OSError(errno.EIO, "short claim write")
                offset += written
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        info = EXPECTED_REVIEW_CLAIM.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o444
            or EXPECTED_REVIEW_CLAIM.read_bytes() != canonical
        ):
            raise RuntimeError("canonical claim post-write verification failed")
        parsed, parsed_bytes, claim_sha = _canonical_json_file(EXPECTED_REVIEW_CLAIM)
        if (
            parsed != dict(payload)
            or parsed_bytes != canonical
            or parsed.get("schema") != CLAIM_SCHEMA
        ):
            raise RuntimeError("canonical claim schema/content verification failed")

        leaf = EXPECTED_REVIEW_CLAIM.parent
        _fsync_directory(leaf)
        leaf_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        leaf_descriptor = os.open(leaf, leaf_flags)
        try:
            os.fchmod(leaf_descriptor, 0o555)
            os.fsync(leaf_descriptor)
        finally:
            os.close(leaf_descriptor)
        _fsync_directory(leaf.parent)
        _fsync_directory(EXPECTED_ROOT_NAMESPACE_BASE)
        if (
            stat.S_IMODE(leaf.lstat().st_mode) != 0o555
            or _owner_group(leaf)
            != (EXPECTED_NAMESPACE_OWNER, EXPECTED_NAMESPACE_GROUP)
        ):
            raise RuntimeError("ledger leaf final mode/owner mismatch")
        return claim_sha
    except FileExistsError:
        # O_EXCL loser: the path was created by another caller, not by this
        # process.  The external state is the card's claim-conflict state.
        raise
    except BaseException as exc:
        if opened_by_this_process and not _path_absent(EXPECTED_REVIEW_CLAIM):
            raise TokenLedgerCorrupt(
                "TOKEN_LEDGER_CORRUPT_CANONICAL_PATH_EXISTS_AFTER_PARTIAL_OR_"
                f"UNDURABLE_LOCAL_PUBLICATION: {type(exc).__name__}: {exc}"
            ) from exc
        raise


def _claim_failure_event(exc: BaseException, *, recovery: bool) -> int:
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    state = "FROZEN_PROVISIONAL_CLAIM_PENDING_RECOVERY_NO_VERDICT"
    conflict_tokens = ("EEXIST", "RACED", "ALREADY_CLAIMED", "CLAIM_CONFLICT")
    if isinstance(exc, FileExistsError) or any(
        token in str(exc).upper() for token in conflict_tokens
    ):
        state = "HARNESS_REJECTED_TOKEN_CLAIM_CONFLICT_NO_ARCHITECTURE_VERDICT"
    if isinstance(exc, TokenLedgerCorrupt):
        state = (
            "TOKEN_LEDGER_CORRUPT_FAIL_CLOSED_NO_VERDICT_"
            "SEPARATE_LEDGER_RECOVERY_AUTHORITY_REQUIRED"
        )
    namespace_snapshot = _token_namespace_snapshot()
    claim_absent = _path_absent(EXPECTED_REVIEW_CLAIM)
    ordinary_recovery_allowed = (
        state == "FROZEN_PROVISIONAL_CLAIM_PENDING_RECOVERY_NO_VERDICT"
        and claim_absent
        and _ordinary_recovery_namespace_prefix_is_safe(namespace_snapshot)
    )
    if (
        state == "FROZEN_PROVISIONAL_CLAIM_PENDING_RECOVERY_NO_VERDICT"
        and not ordinary_recovery_allowed
    ):
        state = "HARNESS_REJECTED_TOKEN_CLAIM_CONFLICT_NO_ARCHITECTURE_VERDICT"
    event = {
        "schema": "cach.av1b_review300.claim_failure_external_event.v1",
        "external_typed_state": state,
        "frozen_root": str(EXPECTED_ROOT),
        "root_mutated_after_freeze": False,
        "automatic_retry": False,
        "ordinary_claim_only_recovery_allowed": ordinary_recovery_allowed,
        "claim_only_recovery_attempt": recovery,
        "canonical_claim_path": str(EXPECTED_REVIEW_CLAIM),
        "canonical_claim_path_absent_after_failure": claim_absent,
        "token_namespace_state_after_failure": namespace_snapshot,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }
    sys.stderr.buffer.write(_canonical_json_bytes(event))
    return 4


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--card", required=True)
    parser.add_argument("--card-sha256", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument(
        "--claim-only-recovery",
        action="store_true",
        help="reverify an already-frozen provisional root and publish only its absent claim",
    )
    return parser.parse_args(argv)


def _precreate_block(exc: PreflightError) -> int:
    event = {
        "schema": "cach.av1b_review300.precreate_block.v1",
        "exact_typed_state": exc.state,
        "validity_reason": [exc.token, exc.detail],
        "resolved_root": None,
        "root_created": False,
        "gpu_used": False,
        "review_token_state": "UNCONSUMED",
        "claim_created": False,
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
    post_pins: dict[str, str] | None = None
    authority: dict[str, Any] = {}
    envelope: dict[str, Any] = {}
    envelope_sha256 = ""
    snapshots: list[dict[str, Any]] = []
    runtime_toolchain_preflight: dict[str, Any] = {}
    operation = "CLAIM_ONLY_RECOVERY" if arguments.claim_only_recovery else "EXECUTE"
    try:
        _require(
            arguments.card_sha256 == EXPECTED_CARD_SHA256,
            "AUTH_BLOCKED",
            "CALLER_CARD_PIN_MISMATCH",
            arguments.card_sha256,
        )
        envelope, envelope_sha256 = _read_authority_envelope()
        card = _validate_card(card_path)
        quartet = _verify_frozen_quartet()
        pins = _source_pins(card)
        _require(
            all(pins.get(path) == digest for path, digest in quartet.items()),
            "AUTH_BLOCKED",
            "MATERIALIZED_QUARTET_PIN_COLLECTION_MISMATCH",
            repr(quartet),
        )
        try:
            # The .yaml contract file is serialized as strict JSON, which is a
            # YAML 1.2 subset.  This keeps the complete preflight stdlib-only.
            config = _duplicate_rejecting_json(config_path)
        except PreflightError:
            raise
        except (OSError, ValueError) as exc:
            raise PreflightError("AUTH_BLOCKED", "CONFIG_PARSE_REJECTED", str(exc)) from exc
        authority = _validate_config(
            config,
            config_path,
            card,
            quartet,
            envelope,
            operation=operation,
        )
        if arguments.claim_only_recovery:
            _validate_environment_recovery(root_argument, arguments.nonce)
            frozen = _verify_frozen_provisional_root(root_argument, pins)
            context = frozen["run_context"]
            _require(
                context.get("source_authority_statement")
                == authority.get("source_authority_statement")
                and context.get("source_authority_statement_sha256")
                == authority.get("source_authority_statement_sha256")
                and context.get("execution_authority_statement")
                == authority.get("execution_authority_statement")
                and context.get("execution_authority_statement_sha256")
                == authority.get("execution_authority_statement_sha256"),
                "AUTH_BLOCKED",
                "RECOVERY_RUN_CONTEXT_AUTHORITY_MISMATCH",
                str(root_argument / "RUN_CONTEXT.json"),
            )
            authority = _validate_recovery_authority(authority, frozen)
            final_verdict = _final_verdict_from_inputs(frozen["verdict_inputs"])
            payload = _claim_payload(frozen, card, authority, final_verdict)
            try:
                claim_sha256 = _atomic_publish_claim(payload, recovery=True)
            except BaseException as exc:
                return _claim_failure_event(exc, recovery=True)
            publication = {
                "schema": "cach.av1b_review300.claim_publication_receipt.v1",
                "final_exact_typed_verdict": final_verdict,
                "canonical_claim_path": str(EXPECTED_REVIEW_CLAIM),
                "canonical_claim_sha256": claim_sha256,
                "claim_only_recovery": True,
                "frozen_root_unchanged": True,
            }
            sys.stdout.buffer.write(_canonical_json_bytes(publication))
            return 0

        _require(
            all(
                authority.get(name) is None
                for name in (
                    "claim_only_recovery_authority_statement",
                    "claim_only_recovery_authority_statement_sha256",
                    "claim_only_recovery_caller_transcript_event",
                    "claim_only_recovery_caller_transcript_event_sha256",
                    "claim_only_recovery_platform_pin",
                    "claim_only_recovery_platform_pin_sha256",
                )
            ),
            "AUTH_BLOCKED",
            "RECOVERY_AUTHORITY_FORBIDDEN_DURING_MODEL_EXECUTION",
            "normal execution envelope must use null recovery fields",
        )
        runtime_toolchain_preflight = _validate_environment_execute(
            root_argument, arguments.nonce, card
        )
        context = {
            "schema": "cach.av1b_review300.run_context.v1",
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
            "selection_statement": EXPECTED_SELECTION,
            "selection_statement_sha256": EXPECTED_SELECTION_AUTHORITY_SHA256,
            "card_freeze_statement": card["authority"]["card_freeze_statement"],
            "card_freeze_statement_sha256": EXPECTED_CARD_FREEZE_AUTHORITY_SHA256,
            "source_authority_statement": authority["source_authority_statement"],
            "source_authority_statement_sha256": authority[
                "source_authority_statement_sha256"
            ],
            "execution_authority_statement": authority[
                "execution_authority_statement"
            ],
            "execution_authority_statement_sha256": authority[
                "execution_authority_statement_sha256"
            ],
            "runtime_authority_envelope_sha256": envelope_sha256,
            "authority_chain_sha256": [
                EXPECTED_SELECTION_AUTHORITY_SHA256,
                EXPECTED_CARD_FREEZE_AUTHORITY_SHA256,
                EXPECTED_SOURCE_AUTHORITY_SHA256,
                authority["execution_authority_statement_sha256"],
            ],
            "predecessor_av1b_card": card["identity"]["predecessor_av1b_card"],
            "predecessor_av1b_root": card["identity"]["predecessor_av1b_root"],
            "predecessor_av1b_nonce": card["identity"]["predecessor_av1b_nonce"],
            "predecessor_av1b_artifacts": card["identity"][
                "predecessor_av1b_artifacts"
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
            "runtime_toolchain_preflight": runtime_toolchain_preflight,
            "review_token_state": "UNCONSUMED",
            "canonical_claim_path": str(EXPECTED_REVIEW_CLAIM),
            "claim_path_absent_before_execution": True,
            "run_namespace_paths_created_exclusively": [
                str(path) for path in EXPECTED_ROOT_NAMESPACE_PATHS
            ],
            "token_namespace_creation_deferred_until_after_provisional_freeze": [
                str(path) for path in EXPECTED_TOKEN_NAMESPACE_PATHS
            ],
            "claim_only_recovery_contract": {
                "no_model_optimizer_or_data_rerun": True,
                "root_must_remain_frozen": True,
                "partial_claim_requires_separate_ledger_recovery_authority": True,
                "automatic_retry": False,
            },
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
        immediate_pins = _source_pins(card)
        immediate_quartet = _verify_frozen_quartet()
        _require(
            immediate_pins == pins
            and all(immediate_pins.get(path) == digest for path, digest in immediate_quartet.items()),
            "AUTH_BLOCKED",
            "SOURCE_PIN_CHANGED_BEFORE_NAMESPACE_OR_ROOT_CREATE",
            "materialized source rehash mismatch",
        )
        _assert_absent(EXPECTED_REVIEW_CLAIM, "CLAIM_APPEARED_BEFORE_ROOT_CREATE")
        for token_path in EXPECTED_TOKEN_NAMESPACE_PATHS:
            _assert_absent(token_path, "TOKEN_NAMESPACE_APPEARED_BEFORE_ROOT_CREATE")
        _exclusive_create_namespace(EXPECTED_ROOT_NAMESPACE_PATHS)
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
            "schema": "cach.av1b_review300.raw_evidence.v1",
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
                source_pins_before=pins,
                source_pins_after=post_pins,
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
    verdict_inputs: dict[str, Any] | None = None
    normalized_theta0_digest: str | None = None
    normalized_theta0_equivalent: bool | None = None
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

        if torch.__version__ != EXPECTED_TORCH_VERSION:
            raise RuntimeError(f"torch version mismatch: {torch.__version__}")
        if triton.__version__ != EXPECTED_TRITON_VERSION:
            raise RuntimeError(f"triton version mismatch: {triton.__version__}")
        if torch.version.cuda != EXPECTED_CUDA_VERSION:
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
        from sana_wam.model.cach_av1b_vendor_gdn_bridge_review300 import run_av1b_screen

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
        _validate_progress_events_300(progress_events)
        lifecycle.publish_json("PROGRESS_EVENTS.json", progress_events)
        for artifact_name, artifact_value in _screen_artifacts(screen).items():
            lifecycle.publish_or_match_json(artifact_name, artifact_value)

        post_pins = _source_pins(card)
        post_quartet = _verify_frozen_quartet()
        if post_pins != pins or any(
            post_pins.get(path) != digest for path, digest in post_quartet.items()
        ):
            raise RuntimeError(
                "card/bridge/config/runner/test or pinned closure bytes changed during review300"
            )
        _validate_card(card_path)
        _validate_config(
            config,
            config_path,
            card,
            post_quartet,
            envelope,
            operation="EXECUTE",
        )
        _assert_absent(EXPECTED_REVIEW_CLAIM, "REVIEW_TOKEN_CLAIM_APPEARED_DURING_RUN")
        for path in EXPECTED_TOKEN_NAMESPACE_PATHS:
            _assert_absent(path, "TOKEN_NAMESPACE_APPEARED_DURING_RUN")
        normalized_theta0_digest, normalized_theta0_equivalent = (
            _verify_theta0_predecessor_equivalence(
                screen["theta0_manifests"], card
            )
        )
        if not normalized_theta0_equivalent:
            raise RuntimeError(
                "normalized theta0 manifests differ from predecessor per-arm entries"
            )

        elapsed = time.monotonic() - started_monotonic
        rss = _rss_bytes()
        root_bytes = _root_size(lifecycle.root)
        gpu_peak_allocated = int(torch.cuda.max_memory_allocated(0))
        gpu_peak_reserved = int(torch.cuda.max_memory_reserved(0))
        budget = card["experiment_contract"]["budget"]
        budget_reasons: list[str] = []
        if elapsed + 30.0 > float(budget["max_wall_seconds_total"]):
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
            state, reasons, verdict_inputs = _evaluate_screen(screen, card)
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
        "schema": "cach.av1b_review300.raw_evidence.v1",
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
        "canonical_claim_path_absent_before_root_freeze": _path_absent(
            EXPECTED_REVIEW_CLAIM
        ),
        "normalized_theta0_digest": normalized_theta0_digest,
        "normalized_theta0_predecessor_equivalent": normalized_theta0_equivalent,
    }
    try:
        terminal = lifecycle.finalize(
            state=state,
            validity_reasons=reasons,
            raw_evidence=raw_evidence,
            source_pins_before=pins,
            source_pins_after=post_pins,
            started_monotonic=started_monotonic,
            verdict_inputs=verdict_inputs,
            normalized_theta0_digest=normalized_theta0_digest,
            normalized_theta0_predecessor_equivalent=normalized_theta0_equivalent,
        )
    except BaseException as exc:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        lifecycle.emergency_finalize()
        sys.stderr.write(f"terminal finalization failed: {type(exc).__name__}: {exc}\n")
        return 3
    if state != VALID_PROVISIONAL_STATE:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        sys.stdout.buffer.write(_canonical_json_bytes(terminal))
        return 2

    try:
        frozen = _verify_frozen_provisional_root(root_argument, pins)
        final_verdict = _final_verdict_from_inputs(frozen["verdict_inputs"])
        payload = _claim_payload(frozen, card, authority, final_verdict)
        claim_sha256 = _atomic_publish_claim(payload, recovery=False)
    except BaseException as exc:
        return _claim_failure_event(exc, recovery=False)
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    publication = {
        "schema": "cach.av1b_review300.claim_publication_receipt.v1",
        "final_exact_typed_verdict": final_verdict,
        "canonical_claim_path": str(EXPECTED_REVIEW_CLAIM),
        "canonical_claim_sha256": claim_sha256,
        "claim_only_recovery": False,
        "frozen_provisional_root": str(EXPECTED_ROOT),
        "frozen_result_sha256": frozen["result_sha256"],
        "frozen_root_unchanged_after_claim": True,
    }
    sys.stdout.buffer.write(_canonical_json_bytes(publication))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
