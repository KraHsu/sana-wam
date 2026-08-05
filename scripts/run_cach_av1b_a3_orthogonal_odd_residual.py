#!/usr/bin/env python3
"""One-shot immutable runner for the non-formal CACH-A3 synthetic screen."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
from typing import Any, NoReturn


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
EXPECTED_CARD = Path(
    "docs/cach_sana_wam/architecture_validation/cach_a3_orthogonal/"
    "CACH_A3_ORTHOGONAL_ODD_RESIDUAL_RUN_CARD.json"
)
EXPECTED_CONFIG = Path(
    "configs/experiments/cach_av1b_a3_orthogonal_odd_residual.yaml"
)
EXPECTED_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a3_orthogonal"
)
EXPECTED_NAMESPACE_LEAF = EXPECTED_NAMESPACE / "06f5d09127f8"
AUTHORITY_SCHEMA = "cach.cach_a3.runtime_execution_authority.v1"
AUTHORITY_OPERATION = "EXECUTE_CACH_A3_ONCE"
USER_AUTHORITY_SHA256 = (
    "7c9691192f1b73408bbe4c0cb6d00db94375ca9d8fce0a0d5985e7a5178f083f"
)
PREDECESSOR_ROOT = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a2_r1/06f5d09127f8/"
    "cach-a2-r1-launcher-fix-dc17cb7bdbd70955ea4b91212f0d4ac3"
)
PREDECESSOR_PINS = {
    "RESULT.json": "a7767e47d718deda41762db4e250dbca0602f78fcff1651ef06a0d6da508c2ef",
    "RAW_METRICS.json": "557cb38344b81c0d6d3ab9a366d78f9d21807cff1d7214f6d36aa981cb50146c",
    "RAW_EVIDENCE.json": "c12f89bb808aeb4762ce9a377f67dc733616fe543018d3413e53c23d277bceae",
    "FREEZE_RECEIPT.json": "5e96575d2057d395e283924c7519e7cfcc51c560bf846d070547c1a26484cb97",
}
SOURCE_PATHS = (
    "docs/cach_sana_wam/architecture_validation/cach_a3_orthogonal/"
    "CACH_A3_ORTHOGONAL_ODD_RESIDUAL_ARCHITECTURE_DECISION.md",
    str(EXPECTED_CARD),
    "src/sana_wam/model/cach_av1b_a3_orthogonal_odd_residual.py",
    str(EXPECTED_CONFIG),
    "scripts/run_cach_av1b_a3_orthogonal_odd_residual.py",
    "tests/test_cach_av1b_a3_orthogonal_odd_residual.py",
)
REQUIRED_ARTIFACTS = (
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
ALLOWED_CAPABILITIES = (
    "SINGLE_GPU_CUDA",
    "TRITON_JIT",
    "VENDOR_AUTOGRAD",
    "MODEL_CONSTRUCT_EXECUTE",
    "FORWARD_BACKWARD",
    "LOCAL_JVP",
    "FRESH_INIT_ADAMW_200_STEP_ORTHOGONAL_SCREEN",
    "SYNTHETIC_ONLY",
)


class PreflightError(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise PreflightError(message)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"invalid JSON in {label}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _strict_json_file(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        _fail(f"not a regular source file: {path}")
    return _strict_json_bytes(path.read_bytes(), str(path))


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
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_dir(path.parent)
    return _sha_bytes(payload)


def _inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink in run root: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            rows.append({"kind": "directory", "mode": f"{stat.S_IMODE(metadata.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(metadata.st_mode):
            rows.append({"kind": "file", "mode": f"{stat.S_IMODE(metadata.st_mode):04o}", "path": relative, "sha256": _sha(path), "size_bytes": metadata.st_size})
        else:
            raise RuntimeError(f"non-regular run entry: {relative}")
    return rows


class RootLifecycle:
    def __init__(self, root: Path, nonce: str) -> None:
        self.root = root
        self.nonce = nonce
        self.created = False
        self.frozen = False
        self.published: dict[str, str] = {}

    def create(self) -> None:
        base = EXPECTED_NAMESPACE.parent
        metadata = base.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or base.is_symlink():
            raise RuntimeError("screen namespace base is invalid")
        old_umask = os.umask(0o002)
        try:
            for path in (EXPECTED_NAMESPACE, EXPECTED_NAMESPACE_LEAF):
                os.mkdir(path, 0o2775)
                os.chmod(path, 0o2775)
                _fsync_dir(path)
                _fsync_dir(path.parent)
            os.mkdir(self.root, 0o770)
            os.chmod(self.root, 0o770)
            self.created = True
            _fsync_dir(self.root)
            _fsync_dir(self.root.parent)
            cache = self.root / "triton_cache"
            os.mkdir(cache, 0o770)
            _fsync_dir(cache)
            _fsync_dir(self.root)
        finally:
            os.umask(old_umask)

    def publish(self, name: str, value: object) -> str:
        if not self.created or self.frozen or name in self.published or "/" in name:
            raise RuntimeError(f"invalid artifact publication: {name}")
        digest = _write_exclusive(self.root / name, value)
        self.published[name] = digest
        return digest

    def freeze(self, terminal_state: str, result_sha256: str) -> None:
        if self.frozen:
            return
        receipt = {
            "schema": "cach.cach_a3.freeze_receipt.v1",
            "root": str(self.root),
            "nonce": self.nonce,
            "terminal_state": terminal_state,
            "result_sha256": result_sha256,
            "pre_receipt_inventory": _inventory(self.root),
            "terminal_directory_mode": "0555",
            "terminal_file_mode": "0444",
            "post_freeze_mutation_allowed": False,
            "automatic_rerun_allowed": False,
        }
        self.publish("FREEZE_RECEIPT.json", receipt)
        for path in sorted(self.root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"symlink before freeze: {path}")
            os.chmod(path, 0o555 if stat.S_ISDIR(mode) else 0o444)
        os.chmod(self.root, 0o555)
        _fsync_dir(self.root)
        _fsync_dir(self.root.parent)
        self.frozen = True

    def emergency_freeze(self) -> None:
        if not self.created or self.frozen:
            return
        for path in sorted(self.root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                mode = path.lstat().st_mode
                if not stat.S_ISLNK(mode):
                    os.chmod(path, 0o555 if stat.S_ISDIR(mode) else 0o444)
            except OSError:
                pass
        try:
            os.chmod(self.root, 0o555)
            _fsync_dir(self.root)
            _fsync_dir(self.root.parent)
        finally:
            self.frozen = True


def _finite(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(child) for child in value)
    if type(value) is float:
        return math.isfinite(value)
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    return parser.parse_args()


def _validate_precreate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    if Path(args.card) != EXPECTED_CARD or Path(args.config) != EXPECTED_CONFIG:
        _fail("card/config path differs from the A3 contract")
    if not re.fullmatch(r"[0-9a-f]{32}", args.nonce):
        _fail("nonce must be lowercase 128-bit hex")
    expected_root = EXPECTED_NAMESPACE_LEAF / f"cach-a3-orthogonal-{args.nonce}"
    if Path(args.root) != expected_root:
        _fail("root differs from nonce-derived fresh root")
    if not isinstance(args.gpu_uuid, str) or not args.gpu_uuid.startswith("GPU-"):
        _fail("invalid GPU UUID")
    if expected_root.exists() or expected_root.is_symlink():
        _fail("fresh root already exists")
    if EXPECTED_NAMESPACE.exists() or EXPECTED_NAMESPACE_LEAF.exists():
        _fail("A3 namespace already exists; this card is single execution only")
    card = _strict_json_file(REPO_ROOT / EXPECTED_CARD)
    config = _strict_json_file(REPO_ROOT / EXPECTED_CONFIG)
    if card.get("schema") != "cach.architecture_validation.cach_a3_orthogonal_odd_residual_run_card.v1":
        _fail("card schema mismatch")
    if config.get("schema") != "cach.cach_a3.normalized_common_exact_odd_output_residual.config.v1":
        _fail("config schema mismatch")
    if card.get("architecture_id") != config.get("architecture_id"):
        _fail("card/config architecture mismatch")

    authority_payload = sys.stdin.buffer.read()
    authority = _strict_json_bytes(authority_payload, "stdin authority")
    required_authority = {
        "schema",
        "operation",
        "user_authority_sha256",
        "card_sha256",
        "source_sha256",
        "root",
        "nonce",
        "gpu_uuid",
        "capabilities",
        "denied",
    }
    if set(authority) != required_authority:
        _fail("authority keys differ from the minimal envelope")
    if authority["schema"] != AUTHORITY_SCHEMA or authority["operation"] != AUTHORITY_OPERATION:
        _fail("authority schema/operation mismatch")
    if authority["user_authority_sha256"] != USER_AUTHORITY_SHA256:
        _fail("user authority SHA mismatch")
    if authority["root"] != args.root or authority["nonce"] != args.nonce or authority["gpu_uuid"] != args.gpu_uuid:
        _fail("authority runtime identity mismatch")
    if authority["capabilities"] != list(ALLOWED_CAPABILITIES):
        _fail("authority capabilities mismatch")
    if authority["denied"] != card.get("forbidden"):
        _fail("authority denied scope mismatch")

    source_hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            _fail(f"source is not regular: {relative}")
        if stat.S_IMODE(metadata.st_mode) != 0o444:
            _fail(f"source is not frozen 0444: {relative}")
        source_hashes[relative] = _sha(path)
    if authority["source_sha256"] != source_hashes:
        _fail("materialized six-file SHA binding mismatch")
    if authority["card_sha256"] != source_hashes[str(EXPECTED_CARD)]:
        _fail("card SHA binding mismatch")

    for name, expected in PREDECESSOR_PINS.items():
        path = PREDECESSOR_ROOT / name
        if _sha(path) != expected:
            _fail(f"predecessor pin drift: {name}")
    predecessor_result = _strict_json_file(PREDECESSOR_ROOT / "RESULT.json")
    if predecessor_result.get("typed_verdict") != "OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED":
        _fail("predecessor terminal verdict drifted")
    if predecessor_result.get("diagnostic_subclassification") != "A2_DELTA_LIVE_COMMON_MODE_BLOCKED":
        _fail("predecessor subclassification drifted")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        _fail("CUDA_VISIBLE_DEVICES differs from authority")
    if os.environ.get("TRITON_CACHE_DIR") != str(expected_root / "triton_cache"):
        _fail("TRITON_CACHE_DIR differs from fresh root")
    if os.environ.get("FUSED_GDN_PRECISION") != "0":
        _fail("FUSED_GDN_PRECISION must be 0")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        _fail("PYTHONDONTWRITEBYTECODE must be 1")
    return card, config, authority, source_hashes


def _timeout(_signum: int, _frame: object) -> NoReturn:
    raise TimeoutError("A3 wall-time budget exceeded")


def main() -> int:
    args = _parse_args()
    lifecycle: RootLifecycle | None = None
    try:
        card, config, authority, source_hashes = _validate_precreate(args)
    except BaseException as exc:
        print(json.dumps({"state": "AUTH_OR_ENV_BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2

    lifecycle = RootLifecycle(Path(args.root), args.nonce)
    previous_alarm = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(int(config["runtime"]["max_wall_seconds"]))
    try:
        lifecycle.create()
        run_context = {
            "schema": "cach.cach_a3.run_context.v1",
            "root": args.root,
            "nonce": args.nonce,
            "gpu_uuid": args.gpu_uuid,
            "authority": authority,
            "authority_sha256": _sha_bytes(_canonical(authority)),
            "card_sha256": source_hashes[str(EXPECTED_CARD)],
            "source_sha256": source_hashes,
            "predecessor_pins": PREDECESSOR_PINS,
            "started_unix_ns": time.time_ns(),
            "real_data": False,
            "checkpoint_load_or_save": False,
            "review_token_create_or_consume": False,
            "formal_or_av2": False,
        }
        lifecycle.publish("RUN_CONTEXT.json", run_context)
        progress: list[Mapping[str, object]] = []

        def on_progress(event: Mapping[str, object]) -> None:
            progress.append(dict(event))

        from sana_wam.model.cach_av1b_a3_orthogonal_odd_residual import (
            run_cach_a3_screen,
        )

        screen = run_cach_a3_screen(
            config,
            expected_gpu_uuid=args.gpu_uuid,
            progress_callback=on_progress,
        )
        required = {
            "schema", "metrics", "validity", "diagnostics",
            "classification_inputs", "typed_verdict",
            "diagnostic_subclassification", "theta0_manifests",
            "loss_trace", "final_parameter_digests", "jit_inventory",
            "resource_usage", "elapsed_seconds_total",
        }
        if not required <= set(screen) or not _finite(screen):
            raise RuntimeError("screen result is incomplete or non-finite")
        if not isinstance(screen["validity"], Mapping):
            raise RuntimeError("screen validity is not a mapping")

        artifact_values: dict[str, object] = {
            "PROGRESS_EVENTS.json": progress,
            "THETA0_MANIFESTS.json": screen["theta0_manifests"],
            "LOSS_TRACE.json": screen["loss_trace"],
            "RAW_METRICS.json": screen["metrics"],
            "RAW_EVIDENCE.json": {
                "schema": "cach.cach_a3.raw_evidence.v1",
                "validity": screen["validity"],
                "diagnostics": screen["diagnostics"],
                "classification_inputs": screen["classification_inputs"],
                "typed_verdict": screen["typed_verdict"],
                "diagnostic_subclassification": screen["diagnostic_subclassification"],
                "resource_usage": screen["resource_usage"],
                "elapsed_seconds_total": screen["elapsed_seconds_total"],
            },
            "FINAL_PARAMETER_DIGESTS.json": screen["final_parameter_digests"],
            "JIT_INVENTORY.json": screen["jit_inventory"],
        }
        artifact_sha = {
            name: lifecycle.publish(name, value)
            for name, value in artifact_values.items()
        }
        result = {
            "schema": "cach.cach_a3.direct_final_result.v1",
            "root": args.root,
            "nonce": args.nonce,
            "architecture_id": card["architecture_id"],
            "card_sha256": source_hashes[str(EXPECTED_CARD)],
            "source_sha256": source_hashes,
            "artifact_sha256": artifact_sha,
            "valid_result": True,
            "all_card_validity_checks_pass": screen["validity"].get("all_card_validity_checks_pass"),
            "metrics": screen["metrics"],
            "classification_inputs": screen["classification_inputs"],
            "typed_verdict": screen["typed_verdict"],
            "terminal_state": screen["typed_verdict"],
            "diagnostic_subclassification": screen["diagnostic_subclassification"],
            "automatic_av2_or_rerun": False,
            "formal_admission_effect": "NONE",
            "global_stage3_effect": "NONE",
            "new_review_token_created_or_consumed": False,
            "real_data_or_checkpoint_used": False,
        }
        result_sha = lifecycle.publish("RESULT.json", result)
        missing = sorted(set(REQUIRED_ARTIFACTS) - set(lifecycle.published) - {"FREEZE_RECEIPT.json"})
        if missing:
            raise RuntimeError(f"required artifacts missing before freeze: {missing}")
        lifecycle.freeze(str(screen["typed_verdict"]), result_sha)
        print(_canonical(result).decode("utf-8"), end="")
        return 0
    except BaseException as exc:
        if lifecycle.created and not lifecycle.frozen:
            try:
                if "RESULT.json" not in lifecycle.published:
                    failure = {
                        "schema": "cach.cach_a3.failure_result.v1",
                        "root": args.root,
                        "nonce": args.nonce,
                        "valid_result": False,
                        "terminal_state": "HARNESS_REJECTED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "automatic_rerun": False,
                    }
                    result_sha = lifecycle.publish("RESULT.json", failure)
                    lifecycle.freeze("HARNESS_REJECTED", result_sha)
                else:
                    lifecycle.emergency_freeze()
            except BaseException:
                lifecycle.emergency_freeze()
        print(json.dumps({"state": "HARNESS_REJECTED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)


if __name__ == "__main__":
    raise SystemExit(main())
