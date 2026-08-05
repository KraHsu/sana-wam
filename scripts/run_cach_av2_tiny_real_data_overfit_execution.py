#!/usr/bin/env python3
"""One-shot immutable runner for the non-formal CACH-A4 AV-2 screen.

The pre-root path is standard-library only.  It verifies the execution card,
all additive and predecessor source pins, the frozen data-selection manifest,
and every selected HDF5 file before importing the execution core.  A full run
then loads the exact CPU tensors, proves their frozen digests/adequacy, checks
one idle GPU, creates one exclusive root, and performs one fresh-init screen.
No checkpoint, token, formal admission, AV-3, or rerun capability exists.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.util
import json
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
EXPECTED_CARD = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av2_execution/"
    "CACH_A4_AV2_EXECUTION_RUN_CARD.json"
)
EXPECTED_DATA_MANIFEST = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av2_execution/"
    "CACH_A4_AV2_DATA_SELECTION.json"
)
EXPECTED_SOURCE = REPO_ROOT / (
    "src/sana_wam/model/cach_av2_tiny_real_data_overfit_execution.py"
)
EXPECTED_CONFIG = REPO_ROOT / (
    "configs/experiments/cach_av2_tiny_real_data_overfit_execution.yaml"
)
EXPECTED_RUNNER = REPO_ROOT / (
    "scripts/run_cach_av2_tiny_real_data_overfit_execution.py"
)
EXPECTED_TEST = REPO_ROOT / (
    "tests/test_cach_av2_tiny_real_data_overfit_execution.py"
)
EXPECTED_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_av2_tiny_real_data"
)
EXPECTED_NAMESPACE_LEAF = EXPECTED_NAMESPACE / "06f5d09127f8"
EXPECTED_NONCE = "96c7c972d34733dd37475b183c136ee0"
EXPECTED_ROOT = EXPECTED_NAMESPACE_LEAF / f"cach-a4-av2-{EXPECTED_NONCE}"
EXPECTED_GPU_INDEX = 0
EXPECTED_GPU_UUID = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
AUTHORITY_STATEMENT = "授权 AV2 tiny real-data 单GPU运行"
AUTHORITY_SHA256 = (
    "7a2c2957f794de03fb46715d2446fc8780d09e237d70e6ac4051f69bf284aef2"
)
CARD_SCHEMA = "cach.architecture_validation.cach_a4_av2_execution_run_card.v1"
CONFIG_SCHEMA = "cach.cach_a4.av2_tiny_real_data_overfit.execution.v1"
RESULT_SCHEMA = "cach.cach_a4.av2_tiny_real_data_overfit.terminal_result.v1"
EXPECTED_ENV = {
    "CUDA_VISIBLE_DEVICES": EXPECTED_GPU_UUID,
    "FUSED_GDN_PRECISION": "0",
    "GDN_DISABLE_COMPILE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "TRITON_CACHE_DIR": str(EXPECTED_ROOT / "triton_cache"),
}
MATERIALIZED_ROLES = {
    "data_manifest": EXPECTED_DATA_MANIFEST,
    "source": EXPECTED_SOURCE,
    "config": EXPECTED_CONFIG,
    "runner": EXPECTED_RUNNER,
    "test": EXPECTED_TEST,
}


class PreflightError(RuntimeError):
    pass


class RunnerInterrupted(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise PreflightError(message)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (bool, int, float, str):
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


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        _fail(f"missing {label}: {path}")
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} is not a regular non-symlink file: {path}")
    return metadata


def _strict_json(path: Path) -> dict[str, Any]:
    _regular_file(path, label="JSON input")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"JSON root must be an object: {path}")
    return value


def _resolve_pin_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        _fail(f"{label}.path must be a non-empty string")
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _verify_pin(pin: object, *, expected_path: Path | None, label: str) -> str:
    if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256"}:
        _fail(f"{label} must contain only path and sha256")
    path = _resolve_pin_path(pin["path"], label=label)
    if expected_path is not None and path != expected_path:
        _fail(f"{label} path differs: {path}")
    expected_sha = pin["sha256"]
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha)
    ):
        _fail(f"{label}.sha256 is malformed")
    _regular_file(path, label=label)
    actual = _sha(path)
    if actual != expected_sha:
        _fail(f"{label} SHA mismatch: expected {expected_sha}, got {actual}")
    return actual


def _verify_materialized_and_predecessor_pins(card: Mapping[str, object]) -> dict[str, str]:
    materialized = card.get("materialized_pins")
    if not isinstance(materialized, Mapping) or set(materialized) != set(MATERIALIZED_ROLES):
        _fail("card.materialized_pins roles differ")
    verified: dict[str, str] = {}
    for role, path in MATERIALIZED_ROLES.items():
        verified[f"materialized:{role}"] = _verify_pin(
            materialized[role], expected_path=path, label=f"materialized_pins.{role}"
        )
    predecessor = card.get("predecessor_pins")
    if not isinstance(predecessor, Mapping) or not predecessor:
        _fail("card.predecessor_pins must be a non-empty mapping")
    for role, pin in predecessor.items():
        if not isinstance(role, str) or not role:
            _fail("predecessor pin role must be a non-empty string")
        verified[f"predecessor:{role}"] = _verify_pin(
            pin, expected_path=None, label=f"predecessor_pins.{role}"
        )
    return verified


def _verify_card_identity(card: Mapping[str, object]) -> None:
    if card.get("schema") != CARD_SCHEMA:
        _fail("execution card schema differs")
    if card.get("state") != "AUTHORIZED_ONE_SHOT_AV2_EXECUTION":
        _fail("execution card state differs")
    authority = card.get("authority")
    if not isinstance(authority, Mapping):
        _fail("card.authority is required")
    if (
        authority.get("statement") != AUTHORITY_STATEMENT
        or authority.get("statement_utf8_bytes") != 38
        or authority.get("statement_sha256") != AUTHORITY_SHA256
        or authority.get("interpretation")
        != "ONE_AV2_TINY_REAL_DATA_SINGLE_GPU_FRESH_ROOT_RUN"
    ):
        _fail("execution authority differs")
    execution = card.get("execution")
    expected = {
        "gpu_index": EXPECTED_GPU_INDEX,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "nonce": EXPECTED_NONCE,
        "root": str(EXPECTED_ROOT),
    }
    if not isinstance(execution, Mapping) or any(
        execution.get(name) != value or type(execution.get(name)) is not type(value)
        for name, value in expected.items()
    ):
        _fail("card execution identity differs")
    if execution.get("automatic_rerun") is not False or execution.get("root_reuse") is not False:
        _fail("card must deny rerun and root reuse")


def _verify_config(config: Mapping[str, object]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        _fail("execution config schema differs")
    authority = config.get("authority")
    if not isinstance(authority, Mapping):
        _fail("config authority is required")
    if (
        authority.get("execution_statement") != AUTHORITY_STATEMENT
        or authority.get("execution_statement_sha256") != AUTHORITY_SHA256
        or authority.get("execution_statement_utf8_bytes") != 38
    ):
        _fail("config authority differs")
    hardware = config.get("hardware_contract")
    run = config.get("run_contract")
    if not isinstance(hardware, Mapping) or not isinstance(run, Mapping):
        _fail("config hardware/run contract is required")
    if (
        hardware.get("gpu_uuid") != EXPECTED_GPU_UUID
        or hardware.get("physical_gpu_index") != EXPECTED_GPU_INDEX
        or hardware.get("cuda_visible_devices") != EXPECTED_GPU_UUID
        or run.get("nonce") != EXPECTED_NONCE
        or run.get("root") != str(EXPECTED_ROOT)
        or run.get("reuse") is not False
        or run.get("automatic_rerun") is not False
    ):
        _fail("config hardware/root identity differs")


def _verify_data_manifest(manifest: Mapping[str, object]) -> dict[str, str]:
    if manifest.get("schema") != "cach.cach_a4.av2_tiny_real_data.data_selection.v1":
        _fail("data selection schema differs")
    if manifest.get("state") != "FROZEN_PRE_MODEL_PRE_GPU":
        _fail("data selection is not frozen")
    rows = manifest.get("selected_files")
    if not isinstance(rows, list) or len(rows) != 16:
        _fail("data selection must contain exactly sixteen files")
    seen: set[tuple[str, int]] = set()
    verified: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"selected_files[{index}] is not a mapping")
        if not {"split", "episode_id", "episode_length", "path", "size_bytes", "sha256"} <= set(row):
            _fail(f"selected_files[{index}] fields differ")
        split = row["split"]
        episode_id = row["episode_id"]
        if split not in ("train", "heldout") or type(episode_id) is not int:
            _fail(f"selected_files[{index}] identity is malformed")
        expected_split = "train" if episode_id < 8 else "heldout"
        if not 0 <= episode_id < 16 or split != expected_split:
            _fail(f"selected_files[{index}] split/episode differs")
        identity = (split, episode_id)
        if identity in seen:
            _fail("duplicate selected episode")
        seen.add(identity)
        path = _resolve_pin_path(row["path"], label=f"selected_files[{index}]")
        metadata = _regular_file(path, label=f"selected episode {episode_id}")
        if type(row["size_bytes"]) is not int or metadata.st_size != row["size_bytes"]:
            _fail(f"selected episode {episode_id} size differs")
        expected_sha = row["sha256"]
        if not isinstance(expected_sha, str) or _sha(path) != expected_sha:
            _fail(f"selected episode {episode_id} SHA differs")
        verified[str(path)] = expected_sha
    if seen != ({("train", i) for i in range(8)} | {("heldout", i) for i in range(8, 16)}):
        _fail("selected episode identity set differs")
    return verified


def _static_preflight() -> dict[str, object]:
    card = _strict_json(EXPECTED_CARD)
    config = _strict_json(EXPECTED_CONFIG)
    manifest = _strict_json(EXPECTED_DATA_MANIFEST)
    _verify_card_identity(card)
    _verify_config(config)
    source_pins = _verify_materialized_and_predecessor_pins(card)
    data_pins = _verify_data_manifest(manifest)
    return {
        "card": card,
        "config": config,
        "manifest": manifest,
        "source_pins": source_pins,
        "data_pins": data_pins,
        "state": "STATIC_PREFLIGHT_OK",
        "schema": CONFIG_SCHEMA,
    }


def _load_source() -> ModuleType:
    name = "_cach_av2_tiny_real_data_overfit_execution"
    spec = importlib.util.spec_from_file_location(name, EXPECTED_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct AV-2 execution source loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_real_data_after_preflight(
    preflight: Mapping[str, object],
) -> tuple[ModuleType, object, Mapping[str, object]]:
    source = _load_source()
    config = preflight["config"]
    manifest = preflight["manifest"]
    assert isinstance(config, Mapping) and isinstance(manifest, Mapping)
    data_contract = config.get("data_contract")
    if not isinstance(data_contract, Mapping) or not isinstance(data_contract.get("dataset_root"), str):
        raise RuntimeError("config dataset root is absent")
    bundle = source.load_av2_real_data(data_contract["dataset_root"])
    adequacy = source.assess_data_adequacy(bundle, config.get("thresholds"))
    if not bool(adequacy["all_adequate"]):
        raise RuntimeError("DATA_INADEQUATE: frozen slices failed adequacy")
    evidence = manifest.get("derived_cpu_tensor_evidence")
    if not isinstance(evidence, Mapping):
        raise RuntimeError("data tensor evidence is absent")
    key_map = {
        "actions": "actions_sha256",
        "target": "target_sha256",
        "noisy_video": "noisy_video_sha256",
        "proprio": "proprio_sha256",
    }
    for split_name in ("train", "heldout"):
        expected = evidence.get(split_name)
        split = getattr(bundle, split_name)
        if not isinstance(expected, Mapping):
            raise RuntimeError(f"missing {split_name} tensor evidence")
        for tensor_name, manifest_name in key_map.items():
            if split.tensor_sha256[tensor_name] != expected.get(manifest_name):
                raise RuntimeError(
                    f"{split_name}.{tensor_name} derived SHA differs from frozen selection"
                )
    return source, bundle, adequacy


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
    rows = []
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
    rows = []
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) == 4:
            rows.append(
                {"gpu_uuid": parts[0], "pid": parts[1], "process_name": parts[2], "used_memory_mib": parts[3]}
            )
    return rows


def _probe_gpu_after_preflight() -> Mapping[str, object]:
    matches = [row for row in _nvidia_rows() if row["index"] == EXPECTED_GPU_INDEX]
    if len(matches) != 1 or matches[0]["uuid"] != EXPECTED_GPU_UUID:
        raise RuntimeError("physical GPU identity differs from AV-2 binding")
    apps = [row for row in _compute_apps() if row["gpu_uuid"] == EXPECTED_GPU_UUID]
    row = matches[0]
    if row["memory_used_mib"] != 0 or row["utilization_gpu_percent"] != 0 or apps:
        raise RuntimeError(f"reserved GPU is not idle: gpu={row}, apps={apps}")
    return {
        **row,
        "compute_apps": apps,
        "captured_unix_ns": time.time_ns(),
    }


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
            rows.append({"kind": "directory", "mode": f"{stat.S_IMODE(metadata.st_mode):04o}", "path": relative})
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


class RootLifecycle:
    def __init__(self) -> None:
        self.root = EXPECTED_ROOT
        self.created = False
        self.frozen = False
        self.published: dict[str, str] = {}

    def create(self) -> None:
        base = EXPECTED_NAMESPACE.parent
        metadata = base.lstat()
        if base.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("screen namespace base is invalid")
        old_umask = os.umask(0o002)
        try:
            for path in (EXPECTED_NAMESPACE, EXPECTED_NAMESPACE_LEAF):
                os.mkdir(path, 0o2775)
                os.chmod(path, 0o2775)
                _fsync_dir(path)
                _fsync_dir(path.parent)
            os.mkdir(self.root, 0o770)
            self.created = True
            os.chmod(self.root, 0o770)
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

    def freeze(self, *, terminal_state: str, result_sha256: str) -> None:
        if self.frozen:
            return
        receipt = {
            "schema": "cach.cach_a4.av2_tiny_real_data.freeze_receipt.v1",
            "root": str(self.root),
            "nonce": EXPECTED_NONCE,
            "terminal_state": terminal_state,
            "result_sha256": result_sha256,
            "pre_receipt_inventory": _inventory(self.root),
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
            os.chmod(path, 0o555 if stat.S_ISDIR(metadata.st_mode) else 0o444)
        os.chmod(self.root, 0o555)
        _fsync_dir(self.root)
        _fsync_dir(self.root.parent)
        self.frozen = True

    def freeze_failure(self, exc: BaseException) -> None:
        if not self.created or self.frozen:
            return
        failure = {
            "schema": RESULT_SCHEMA,
            "terminal_state": "AV2_INVALID_FAIL_CLOSED",
            "typed_verdict": "AV2_INVALID_FAIL_CLOSED",
            "valid_result": False,
            "reason_code": type(exc).__name__,
            "error": str(exc),
            "root": str(self.root),
            "nonce": EXPECTED_NONCE,
            "automatic_rerun_allowed": False,
        }
        try:
            if "RESULT.json" not in self.published:
                result_sha = self.publish("RESULT.json", failure)
            else:
                result_sha = self.published["RESULT.json"]
            self.freeze(terminal_state="AV2_INVALID_FAIL_CLOSED", result_sha256=result_sha)
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


def _create_root_after_preflight() -> RootLifecycle:
    lifecycle = RootLifecycle()
    lifecycle.create()
    return lifecycle


def _jit_inventory(root: Path) -> Mapping[str, object]:
    cache = root / "triton_cache"
    rows = _inventory(cache) if cache.is_dir() else []
    return {
        "cache_path": str(cache),
        "entries": rows,
        "entry_count": len(rows),
        "vendor_jit_or_cache_observed": bool(rows),
    }


def _execute_after_preflight(
    source: ModuleType,
    config: Mapping[str, object],
    bundle: object,
    progress: list[object],
) -> Mapping[str, object]:
    def callback(event: Mapping[str, object]) -> None:
        progress.append({"event": "optimizer_progress", **dict(event)})

    return source.run_av2_tiny_real_data_overfit(
        config,
        expected_gpu_uuid=EXPECTED_GPU_UUID,
        source_and_predecessor_pins_match=True,
        progress_callback=callback,
        preloaded_bundle=bundle,
    )


def _check_exact_env() -> None:
    for name, expected in EXPECTED_ENV.items():
        if os.environ.get(name) != expected:
            _fail(f"environment {name} differs: expected {expected!r}")


def _check_full_args(args: argparse.Namespace) -> None:
    expected = {
        "card": str(EXPECTED_CARD),
        "config": str(EXPECTED_CONFIG),
        "root": str(EXPECTED_ROOT),
        "nonce": EXPECTED_NONCE,
        "gpu_index": EXPECTED_GPU_INDEX,
        "gpu_uuid": EXPECTED_GPU_UUID,
    }
    for name, value in expected.items():
        if getattr(args, name) != value or type(getattr(args, name)) is not type(value):
            _fail(f"argument --{name.replace('_', '-')} differs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--card", default=str(EXPECTED_CARD))
    parser.add_argument("--config", default=str(EXPECTED_CONFIG))
    parser.add_argument("--root", default=str(EXPECTED_ROOT))
    parser.add_argument("--nonce", default=EXPECTED_NONCE)
    parser.add_argument("--gpu-index", type=int, default=EXPECTED_GPU_INDEX)
    parser.add_argument("--gpu-uuid", default=EXPECTED_GPU_UUID)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lifecycle: RootLifecycle | None = None
    previous_handlers: dict[int, object] = {}

    def interrupted(signum: int, _frame: object) -> None:
        raise RunnerInterrupted(f"received signal {signum}")

    try:
        _check_full_args(args)
        preflight = _static_preflight()
        if args.static_preflight:
            print(
                _canonical(
                    {
                        "schema": CONFIG_SCHEMA,
                        "state": "STATIC_PREFLIGHT_OK",
                        "card_sha256": _sha(EXPECTED_CARD),
                        "data_manifest_sha256": _sha(EXPECTED_DATA_MANIFEST),
                        "verified_source_pin_count": len(preflight["source_pins"]),
                        "verified_data_file_count": len(preflight["data_pins"]),
                    }
                ).decode("utf-8"),
                end="",
            )
            return 0
        _check_exact_env()
        source, bundle, adequacy = _load_real_data_after_preflight(preflight)
        pre_root_gpu = _probe_gpu_after_preflight()
        lifecycle = _create_root_after_preflight()
        post_root_gpu = _probe_gpu_after_preflight()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        run_context = {
            "schema": "cach.cach_a4.av2_tiny_real_data.run_context.v1",
            "root": str(EXPECTED_ROOT),
            "nonce": EXPECTED_NONCE,
            "gpu": post_root_gpu,
            "pre_root_gpu": pre_root_gpu,
            "authority_sha256": AUTHORITY_SHA256,
            "card_sha256": _sha(EXPECTED_CARD),
            "data_manifest_sha256": _sha(EXPECTED_DATA_MANIFEST),
            "materialized_source_sha256": _sha(EXPECTED_SOURCE),
            "config_sha256": _sha(EXPECTED_CONFIG),
            "runner_sha256": _sha(EXPECTED_RUNNER),
            "test_sha256": _sha(EXPECTED_TEST),
            "exact_env": dict(EXPECTED_ENV),
            "checkpoint_load": False,
            "checkpoint_save": False,
            "formal_admission": False,
            "automatic_rerun": False,
        }
        lifecycle.publish("RUN_CONTEXT.json", run_context)
        progress: list[object] = [
            {"event": "post_root_pre_import_gpu_idle_snapshot", "payload": post_root_gpu}
        ]
        screen = _execute_after_preflight(
            source,
            preflight["config"],  # type: ignore[arg-type]
            bundle,
            progress,
        )
        lifecycle.publish("DATA_EVIDENCE.json", {"adequacy": adequacy})
        lifecycle.publish("PROGRESS_EVENTS.json", progress)
        lifecycle.publish("SCREEN_RESULT.json", screen)
        jit = _jit_inventory(EXPECTED_ROOT)
        lifecycle.publish("JIT_INVENTORY.json", jit)
        validity = screen.get("validity") if isinstance(screen, Mapping) else None
        verdict = screen.get("verdict") if isinstance(screen, Mapping) else None
        reason = screen.get("reason_code") if isinstance(screen, Mapping) else None
        if not isinstance(validity, Mapping) or not isinstance(verdict, str):
            raise RuntimeError("execution core returned malformed result")
        terminal = {
            "schema": RESULT_SCHEMA,
            "terminal_state": verdict,
            "typed_verdict": verdict,
            "reason_code": reason,
            "valid_result": bool(validity.get("run_valid")),
            "av3_unlocked": verdict == "AV2_GO" and bool(validity.get("run_valid")),
            "root": str(EXPECTED_ROOT),
            "nonce": EXPECTED_NONCE,
            "gpu_uuid": EXPECTED_GPU_UUID,
            "screen_result_sha256": lifecycle.published["SCREEN_RESULT.json"],
            "jit_inventory_sha256": lifecycle.published["JIT_INVENTORY.json"],
            "automatic_rerun_allowed": False,
            "checkpoint_load_or_save": False,
            "formal_claim_allowed": False,
        }
        result_sha = lifecycle.publish("RESULT.json", terminal)
        lifecycle.freeze(terminal_state=verdict, result_sha256=result_sha)
        print(_canonical(terminal).decode("utf-8"), end="")
        return 0 if bool(terminal["valid_result"]) else 2
    except BaseException as exc:
        if lifecycle is not None:
            lifecycle.freeze_failure(exc)
        print(
            _canonical(
                {
                    "schema": RESULT_SCHEMA,
                    "state": "AV2_INVALID_FAIL_CLOSED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "root_created": bool(lifecycle and lifecycle.created),
                    "automatic_rerun_allowed": False,
                }
            ).decode("utf-8"),
            end="",
        )
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(main())
