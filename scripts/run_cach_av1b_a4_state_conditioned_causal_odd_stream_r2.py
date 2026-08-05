#!/usr/bin/env python3
"""One-shot immutable runner scaffold for the non-formal CACH-A4-R2 screen.

R2 is an additive diagnostic-mask-shape repair with no architecture, task,
seed, optimizer, budget, threshold, or launcher delta.  Its standard-library
pre-root path fail-closes unless fresh user authority and the exact
root/nonce/GPU binding match.  It then verifies the
frozen A3/A4/A4-R1 predecessors and imports the additive R2 bridge.  Every
created root must terminate with immutable evidence; no token/claim,
checkpoint, real-data, AV2, formal, or automatic rerun path is implemented.
"""

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
import subprocess
import sys
import time
from typing import Any, NoReturn


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
EXPECTED_DECISION = Path(
    "docs/cach_sana_wam/architecture_validation/cach_a4_r2/"
    "CACH_A4_R2_DIAGNOSTIC_MASK_SHAPE_FIX_DECISION.md"
)
EXPECTED_CARD = Path(
    "docs/cach_sana_wam/architecture_validation/cach_a4_r2/CACH_A4_R2_RUN_CARD.json"
)
EXPECTED_BRIDGE = Path(
    "src/sana_wam/model/cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py"
)
EXPECTED_CONFIG = Path(
    "configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r2.yaml"
)
EXPECTED_RUNNER = Path(
    "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py"
)
EXPECTED_TEST = Path(
    "tests/test_cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py"
)
EXPECTED_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r2"
)
EXPECTED_NAMESPACE_LEAF = EXPECTED_NAMESPACE / "06f5d09127f8"
AUTHORITY_SCHEMA = "cach.cach_a4_r2.runtime_execution_authority.v1"
AUTHORITY_OPERATION = "EXECUTE_CACH_A4_R2_ONCE"
USER_AUTHORITY_STATEMENT = "授权 A4-R2 单GPU合成运行"
USER_AUTHORITY_SHA256 = (
    "8dc3ad2e8bd1edbf8c276ff8c9979c3a923b24d026e0a9eca8dcf518311028b0"
)
EXPECTED_NONCE = "f294a23c285082ae61b1685accec416e"
EXPECTED_ROOT = EXPECTED_NAMESPACE_LEAF / (
    "cach-a4-state-stream-r2-f294a23c285082ae61b1685accec416e"
)
EXPECTED_GPU_INDEX = 0
EXPECTED_GPU_UUID = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
EXPECTED_EXACT_ENV = {
    "CUDA_VISIBLE_DEVICES": EXPECTED_GPU_UUID,
    "TRITON_CACHE_DIR": str(EXPECTED_ROOT / "triton_cache"),
    "FUSED_GDN_PRECISION": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "GDN_DISABLE_COMPILE": "1",
    "TORCHDYNAMO_DISABLE": "1",
}
EXPECTED_EXACT_ARGV = [
    "/home/zch/workspace/sana-wam/.venv/bin/python",
    str(EXPECTED_RUNNER),
    "--card",
    str(EXPECTED_CARD),
    "--config",
    str(EXPECTED_CONFIG),
    "--root",
    str(EXPECTED_ROOT),
    "--nonce",
    EXPECTED_NONCE,
    "--gpu-index",
    str(EXPECTED_GPU_INDEX),
    "--gpu-uuid",
    EXPECTED_GPU_UUID,
]
EXPECTED_EXACT_COMMAND = (
    "env CUDA_VISIBLE_DEVICES=GPU-1ec28cfb-f501-23f3-f865-275a744ca053 "
    "FUSED_GDN_PRECISION=0 GDN_DISABLE_COMPILE=1 PYTHONDONTWRITEBYTECODE=1 "
    "TORCHDYNAMO_DISABLE=1 "
    "TRITON_CACHE_DIR=/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_state_stream_r2/06f5d09127f8/"
    "cach-a4-state-stream-r2-f294a23c285082ae61b1685accec416e/triton_cache "
    "/home/zch/workspace/sana-wam/.venv/bin/python "
    "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py "
    "--card docs/cach_sana_wam/architecture_validation/cach_a4_r2/"
    "CACH_A4_R2_RUN_CARD.json "
    "--config configs/experiments/"
    "cach_av1b_a4_state_conditioned_causal_odd_stream_r2.yaml "
    "--root /DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_state_stream_r2/06f5d09127f8/"
    "cach-a4-state-stream-r2-f294a23c285082ae61b1685accec416e "
    "--nonce f294a23c285082ae61b1685accec416e --gpu-index 0 "
    "--gpu-uuid GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
)
PREDECESSOR_ROOT = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a3_orthogonal/06f5d09127f8/"
    "cach-a3-orthogonal-b2e55836da6fd5a51b568af613f74eef"
)
PREDECESSOR_PINS = {
    "RESULT.json": "81345a1c9b6e03d60dd587cb439ebdd9c2d60fc92a7120d80042c6a9e4dc78c3",
    "RAW_METRICS.json": "19ab6c95d3aed8915a887506425a2de3fd4c6d1e3f500478144586415c0cca9f",
    "RAW_EVIDENCE.json": "ee43f57d503a1db78bc26457efe048fbfe5c0a3be14ccaf79e6489b62363f9ef",
    "FREEZE_RECEIPT.json": "69b72fd730b316509548e4dab198e4dd7d84565ee9f1d05a6cccd1e8a673c63c",
}
A3_SOURCE_PINS = {
    "docs/cach_sana_wam/architecture_validation/cach_a3_orthogonal/CACH_A3_ORTHOGONAL_ODD_RESIDUAL_ARCHITECTURE_DECISION.md": "ec99746eb5abbe6de29974d4fcf4e56a043fcbbf6df490264fe9fe4581b30979",
    "docs/cach_sana_wam/architecture_validation/cach_a3_orthogonal/CACH_A3_ORTHOGONAL_ODD_RESIDUAL_RUN_CARD.json": "674073b39c1964bbc9c955022437390445dbb2bddfe42403b2c495f5a900be19",
    "src/sana_wam/model/cach_av1b_a3_orthogonal_odd_residual.py": "e1b5dba81880c0524d79222995111ab820b29901158309c58a5b4688adf68597",
    "configs/experiments/cach_av1b_a3_orthogonal_odd_residual.yaml": "8b3aa219826fedbe53e6aa33ffe7b357c7d6bf0ee4d334b116cb13a55037012f",
    "scripts/run_cach_av1b_a3_orthogonal_odd_residual.py": "3d6177472454681e74c25ddc347651512ba43a7ea298b32a126e89bb2f2d1adc",
    "tests/test_cach_av1b_a3_orthogonal_odd_residual.py": "2d484bdd6f13352c6388c6b36622d12d9ea187c503f4d57c7edfd2b1e8b4e98f",
}
A4_FAILED_ROOT = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream/"
    "06f5d09127f8/cach-a4-state-stream-ac7fb52eb22be3cce184a60a41f7e98e"
)
A4_FAILED_PINS = {
    "RESULT.json": "07f5246d1ad0d8b076f60fdb9f41c5ab53a3db8f9c4c9d31524f705b98a26323",
    "FREEZE_RECEIPT.json": "0f93621d81498ce80187c67d9ce42ae233fa560a36784e401303d102c7403620",
    "RUN_CONTEXT.json": "d46213a56f33f02a7a9b61e0c8fcc83f687d4739c50d9d10cbde56f046b331d1",
}
A4_SOURCE_PINS = {
    "docs/cach_sana_wam/architecture_validation/cach_a4_state_stream/CACH_A4_STATE_CONDITIONED_CAUSAL_STREAM_ARCHITECTURE_DECISION.md": "2a2efb3bda76dd36d2d1b98c0d26703de40704bc0744098a93b9820eac0b3a92",
    "docs/cach_sana_wam/architecture_validation/cach_a4_state_stream/CACH_A4_STATE_CONDITIONED_CAUSAL_STREAM_RUN_CARD.json": "b069c5fcbfde49e1f2e7a7ae0feb47067910ed680d09b28eb8459deb22ef618d",
    "src/sana_wam/model/cach_av1b_a4_state_conditioned_causal_odd_stream.py": "54924361009668a5c1cd113799812cc0f3d758b83528bf0aa917efba55a5fa6a",
    "configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream.yaml": "c49e03ad22a3ed8f79160817309714a674e0c8aa9766a1beb0eb5c8a7dabf835",
    "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream.py": "8c106c23b579c8ab0742a9d66a99b886223c063bf876fe3f368067251dcd9cce",
    "tests/test_cach_av1b_a4_state_conditioned_causal_odd_stream.py": "892b6e8d673a180a377b74d44a6fdd2818ced8a8b27dfbd64d628569baab0977",
}
R1_FAILED_ROOT = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r1/"
    "06f5d09127f8/cach-a4-state-stream-r1-9e604d41d671ac4e705816857457e5a4"
)
R1_FAILED_PINS = {
    "RESULT.json": "ba24e73a316dcc92e2d759f90d9e309db9747db88f60842440bcd9e52b3875e0",
    "FREEZE_RECEIPT.json": "383ca8321f454d62e9ec966f5db9e737532a2d2d6d1ee266b11e57972914c6f7",
    "RUN_CONTEXT.json": "b34c8317fdce4e452be1ef209f3e3ea485d6990165b0b302053da42eba1a8a15",
    "PROGRESS_EVENTS.json": "6be08df7a57e0871e3c278334242fb8ef4f944e26182fe9a22252abd3ebfca3e",
}
R1_SOURCE_PINS = {
    "docs/cach_sana_wam/architecture_validation/cach_a4_r1/CACH_A4_R1_LAUNCHER_ENV_FIX_DECISION.md": "aa06060c40ea2fe1abad1041dd8ebfb236619ce51cc36ac1718cca9e4d5e67f6",
    "docs/cach_sana_wam/architecture_validation/cach_a4_r1/CACH_A4_R1_RUN_CARD.json": "faddcddcb7a1a6a69f013f615f39e99e78b128b50fe7d4a6f67fe3b8a335c0a7",
    "src/sana_wam/model/cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py": "abd8a4f1d671cc7d0af735724a4034f86b85f05b5c0448dcfb2fc4ba2af31d1c",
    "configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r1.yaml": "9acf0425fa8370be546686e7ee23e5c764a23a804b70dc125ee25c772e3317ae",
    "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py": "4c273186ebc0ff61902f25b3a38b5e59fad51a701d59e7fc0b4351bac3bdf600",
    "tests/test_cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py": "b4b386d439344b0c6c041544a5c79b0d016113d4f1e030c88e77cc5f6c27f606",
}
SOURCE_PATHS = (
    str(EXPECTED_DECISION),
    str(EXPECTED_CARD),
    str(EXPECTED_BRIDGE),
    str(EXPECTED_CONFIG),
    str(EXPECTED_RUNNER),
    str(EXPECTED_TEST),
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
    "FRESH_INIT_ADAMW_200_STEP_STATE_CONDITIONED_CAUSAL_ODD_STREAM",
    "CAUSAL_DIAGNOSTICS",
    "SYNTHETIC_ONLY",
)
CARD_AUTHORIZED_CAPABILITIES = [
    "SOURCE_IMPLEMENTATION",
    "REDUCED_SYNTHETIC_CPU_TESTS",
    "ONE_IDLE_SINGLE_GPU_FRESH_ROOT_SYNTHETIC_RUN",
]
ARCHITECTURE_ID = "CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1"
CARD_SCHEMA = (
    "cach.architecture_validation.cach_a4_r2_diagnostic_mask_shape_fix_run_card.v1"
)
CONFIG_SCHEMA = "cach.cach_a4.state_conditioned_causal_odd_stream.config.v1"
SCREEN_SCHEMA = "cach.cach_a4.state_conditioned_causal_odd_stream.screen.v1"


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


def _inventory(
    root: Path, *, exclude: frozenset[str] = frozenset()
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
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
        if (
            not self.created
            or self.frozen
            or name in self.published
            or name not in REQUIRED_ARTIFACTS
        ):
            raise RuntimeError(f"invalid artifact publication: {name}")
        digest = _write_exclusive(self.root / name, value)
        self.published[name] = digest
        return digest

    def freeze(self, terminal_state: str, result_sha256: str) -> None:
        if self.frozen:
            return
        missing = (
            set(REQUIRED_ARTIFACTS) - set(self.published) - {"FREEZE_RECEIPT.json"}
        )
        if missing:
            raise RuntimeError(
                f"required artifacts missing before freeze: {sorted(missing)}"
            )
        receipt = {
            "schema": "cach.cach_a4_r2.freeze_receipt.v1",
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
        for path in sorted(
            self.root.rglob("*"), key=lambda p: len(p.parts), reverse=True
        ):
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
        for path in sorted(
            self.root.rglob("*"), key=lambda p: len(p.parts), reverse=True
        ):
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
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--gpu-uuid", required=True)
    return parser.parse_args()


def _gpu_idle_snapshot(physical_index: int, gpu_uuid: str) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            _fail(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "utilization_gpu_percent": int(fields[3]),
                "memory_used_mib": int(fields[4]),
            }
        )
    matches = [
        row
        for row in rows
        if row["physical_index"] == physical_index and row["uuid"] == gpu_uuid
    ]
    if len(matches) != 1:
        _fail("physical GPU index/UUID binding is not unique")
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    selected_apps = [
        line.strip()
        for line in apps.stdout.splitlines()
        if line.strip().startswith(gpu_uuid + ",")
    ]
    selected = matches[0]
    if selected["utilization_gpu_percent"] != 0 or selected_apps:
        _fail("authority-bound GPU is not idle")
    return {
        **selected,
        "compute_apps": selected_apps,
        "captured_unix_ns": time.time_ns(),
    }


def _validate_precreate(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
]:
    if (
        not USER_AUTHORITY_STATEMENT
        or _sha_bytes(USER_AUTHORITY_STATEMENT.encode("utf-8")) != USER_AUTHORITY_SHA256
    ):
        _fail("AUTH_BLOCKED: exact fresh CACH-A4-R2 authority drifted")
    if Path.cwd() != REPO_ROOT:
        _fail("canonical working directory differs from the authorized binding")
    if Path(args.card) != EXPECTED_CARD or Path(args.config) != EXPECTED_CONFIG:
        _fail("card/config path differs from the A4-R2 contract")
    if args.nonce != EXPECTED_NONCE or not re.fullmatch(r"[0-9a-f]{32}", args.nonce):
        _fail("nonce differs from the fresh-authority binding")
    expected_root = EXPECTED_NAMESPACE_LEAF / f"cach-a4-state-stream-r2-{args.nonce}"
    if expected_root != EXPECTED_ROOT or Path(args.root) != EXPECTED_ROOT:
        _fail("root differs from the fresh-authority binding")
    if args.gpu_uuid != EXPECTED_GPU_UUID or args.gpu_index != EXPECTED_GPU_INDEX:
        _fail("physical GPU index/UUID differs from the fresh-authority binding")
    if expected_root.exists() or expected_root.is_symlink():
        _fail("fresh root already exists")
    if EXPECTED_NAMESPACE.exists() or EXPECTED_NAMESPACE_LEAF.exists():
        _fail("A4-R2 namespace already exists; this card is single execution only")
    card = _strict_json_file(REPO_ROOT / EXPECTED_CARD)
    config = _strict_json_file(REPO_ROOT / EXPECTED_CONFIG)
    if card.get("schema") != CARD_SCHEMA:
        _fail("card schema mismatch")
    if config.get("schema") != CONFIG_SCHEMA:
        _fail("config schema mismatch")
    card_identity = card.get("identity")
    if (
        not isinstance(card_identity, Mapping)
        or card_identity.get("architecture_id") != ARCHITECTURE_ID
        or config.get("architecture_id") != ARCHITECTURE_ID
    ):
        _fail("card/config architecture mismatch")
    runtime = config.get("runtime")
    required_launcher_environment = {
        "GDN_DISABLE_COMPILE": "1",
        "TORCHDYNAMO_DISABLE": "1",
    }
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("launcher_environment_before_torch_or_vendor_import")
        != required_launcher_environment
    ):
        _fail("R2 config launcher environment drifted")
    r1_config = _strict_json_file(
        REPO_ROOT
        / "configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r1.yaml"
    )
    if config != r1_config:
        _fail("R2 config differs byte-semantically from frozen A4-R1 config")
    if (
        card.get("state")
        != "ONE_IDLE_SINGLE_GPU_FRESH_ROOT_SYNTHETIC_EXECUTION_AUTHORIZED"
    ):
        _fail("card is not in the one-shot authorized A4-R2 state")
    training = config.get("training")
    if not isinstance(training, Mapping):
        _fail("config.training must be a mapping")
    if (
        training.get("macrosteps_per_arm") != 200
        or training.get("max_macrosteps_total") != 400
    ):
        _fail("A4 must use exactly 200 macrosteps per arm and 400 total")
    if (
        training.get("fresh_initialization") is not True
        or training.get("predecessor_continuation") is not False
    ):
        _fail("A4 must use fresh initialization without predecessor continuation")
    task = config.get("task")
    if (
        not isinstance(task, Mapping)
        or task.get("source") != "PINNED_A3_SYNTHETIC_TASK_AND_TARGET_BYTES_UNCHANGED"
        or task.get("recipe_sha256")
        != "9d2a749d722775324ee233862772def530fbffe0ea9e55aeea87c160b87a51e4"
        or task.get("target_bytes_modified") is not False
        or task.get("real_data") is not False
    ):
        _fail("pinned A3 synthetic task/target contract drifted")
    card_source = card.get("source_six")
    if (
        not isinstance(card_source, Mapping)
        or card_source.get("planned_paths") != list(SOURCE_PATHS)
        or card_source.get("bind_and_verify_all_six_sha256_before_gpu_execution")
        is not True
    ):
        _fail("card source-six contract drifted")
    execution = card.get("execution")
    if not isinstance(execution, Mapping) or card.get(
        "future_root_namespace_family"
    ) != str(EXPECTED_NAMESPACE_LEAF):
        _fail("card execution namespace contract drifted")
    if (
        execution.get("source_and_cpu_authorized") is not True
        or execution.get("gpu_execution_authorized") is not True
        or execution.get("execution_requires_fresh_user_authority") is not False
        or execution.get("fresh_user_authority_satisfied") is not True
        or execution.get("one_shot_gpu_execution_max") != 1
        or execution.get("automatic_rerun") is not False
        or execution.get("post_authority_capabilities") != CARD_AUTHORIZED_CAPABILITIES
    ):
        _fail("authorized card authority/capabilities drifted")

    execution_binding = card.get("execution_binding")
    if (
        not isinstance(execution_binding, Mapping)
        or execution_binding.get("binding_state") != "BOUND_BY_FRESH_USER_AUTHORITY"
        or execution_binding.get("canonical_working_directory") != str(REPO_ROOT)
        or execution_binding.get("root") != str(EXPECTED_ROOT)
        or execution_binding.get("nonce") != EXPECTED_NONCE
        or execution_binding.get("physical_gpu_index") != EXPECTED_GPU_INDEX
        or execution_binding.get("gpu_uuid") != EXPECTED_GPU_UUID
        or execution_binding.get("exact_env") != EXPECTED_EXACT_ENV
        or execution_binding.get("exact_argv") != EXPECTED_EXACT_ARGV
        or execution_binding.get("exact_command") != EXPECTED_EXACT_COMMAND
        or execution_binding.get("stdin_contract")
        != "CANONICAL_UTF8_SORTED_COMPACT_JSON_PLUS_LF_RUNTIME_AUTHORITY"
    ):
        _fail("card exact execution binding drifted")
    fresh_authority = card.get("fresh_user_authority")
    if (
        not isinstance(fresh_authority, Mapping)
        or fresh_authority.get("statement") != USER_AUTHORITY_STATEMENT
        or fresh_authority.get("statement_sha256") != USER_AUTHORITY_SHA256
        or fresh_authority.get("exact_utf8_length_bytes") != 31
        or fresh_authority.get("satisfied") is not True
        or fresh_authority.get("scope") != "THIS_CACH_A4_R2_ONE_SHOT_BINDING_ONLY"
        or fresh_authority.get("inherited_from_failed_a4_authority") is not False
    ):
        _fail("card fresh user authority binding drifted")

    authority_payload = sys.stdin.buffer.read()
    authority = _strict_json_bytes(authority_payload, "stdin authority")
    if authority_payload != _canonical(authority):
        _fail("stdin authority must be canonical JSON with one trailing LF")
    required_authority = {
        "schema",
        "operation",
        "user_authority_statement",
        "user_authority_sha256",
        "execution_statement",
        "execution_statement_sha256",
        "exact_argv",
        "exact_env",
        "card_sha256",
        "source_sha256",
        "root",
        "nonce",
        "gpu_physical_index",
        "gpu_uuid",
        "capabilities",
        "denied",
    }
    if set(authority) != required_authority:
        _fail("authority keys differ from the minimal envelope")
    if (
        authority["schema"] != AUTHORITY_SCHEMA
        or authority["operation"] != AUTHORITY_OPERATION
    ):
        _fail("authority schema/operation mismatch")
    if (
        authority["user_authority_statement"] != USER_AUTHORITY_STATEMENT
        or authority["user_authority_sha256"] != USER_AUTHORITY_SHA256
    ):
        _fail("user authority literal/SHA mismatch")
    if _sha_bytes(USER_AUTHORITY_STATEMENT.encode("utf-8")) != USER_AUTHORITY_SHA256:
        _fail("internal user authority SHA mismatch")
    if (
        authority["root"] != args.root
        or authority["nonce"] != args.nonce
        or authority["gpu_physical_index"] != args.gpu_index
        or authority["gpu_uuid"] != args.gpu_uuid
    ):
        _fail("authority runtime identity mismatch")
    if authority["capabilities"] != list(ALLOWED_CAPABILITIES):
        _fail("authority capabilities mismatch")
    if authority["denied"] != card.get("forbidden"):
        _fail("authority denied scope mismatch")
    expected_argv = [sys.executable, *sys.argv]
    expected_env = {
        "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
        "TRITON_CACHE_DIR": str(expected_root / "triton_cache"),
        "FUSED_GDN_PRECISION": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GDN_DISABLE_COMPILE": "1",
        "TORCHDYNAMO_DISABLE": "1",
    }
    if expected_argv != EXPECTED_EXACT_ARGV or expected_env != EXPECTED_EXACT_ENV:
        _fail("live env/argv differs from the exact card binding")
    if authority["exact_argv"] != expected_argv:
        _fail("authority exact_argv differs from the current process argv")
    if authority["exact_env"] != expected_env:
        _fail("authority exact_env differs from the required runtime environment")

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
        _fail("materialized R2 six-file SHA binding mismatch")
    if authority["card_sha256"] != source_hashes[str(EXPECTED_CARD)]:
        _fail("card SHA binding mismatch")

    for relative, expected in A4_SOURCE_PINS.items():
        path = REPO_ROOT / relative
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or _sha(path) != expected
        ):
            _fail(f"frozen original A4 source pin drift: {relative}")

    failed_root_metadata = A4_FAILED_ROOT.lstat()
    if (
        not stat.S_ISDIR(failed_root_metadata.st_mode)
        or A4_FAILED_ROOT.is_symlink()
        or stat.S_IMODE(failed_root_metadata.st_mode) != 0o555
    ):
        _fail("failed A4 root is not immutable 0555")
    for name, expected in A4_FAILED_PINS.items():
        path = A4_FAILED_ROOT / name
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or _sha(path) != expected
        ):
            _fail(f"failed A4 artifact pin drift: {name}")
    failed_result = _strict_json_file(A4_FAILED_ROOT / "RESULT.json")
    if (
        failed_result.get("root") != str(A4_FAILED_ROOT)
        or failed_result.get("nonce") != "ac7fb52eb22be3cce184a60a41f7e98e"
        or failed_result.get("valid_result") is not False
        or failed_result.get("terminal_state") != "HARNESS_REJECTED"
        or failed_result.get("typed_verdict") != "INVALID_RUN"
        or failed_result.get("error_type") != "BackendCompilerFailed"
        or "triton_key" not in str(failed_result.get("error", ""))
        or failed_result.get("automatic_av2_or_rerun") is not False
    ):
        _fail("failed A4 result is not the pinned launcher incompatibility")
    failed_receipt = _strict_json_file(A4_FAILED_ROOT / "FREEZE_RECEIPT.json")
    if (
        failed_receipt.get("root") != str(A4_FAILED_ROOT)
        or failed_receipt.get("result_sha256") != A4_FAILED_PINS["RESULT.json"]
        or failed_receipt.get("terminal_state") != "HARNESS_REJECTED"
        or failed_receipt.get("automatic_rerun_allowed") is not False
        or failed_receipt.get("post_freeze_mutation_allowed") is not False
    ):
        _fail("failed A4 freeze receipt drifted")
    failed_context = _strict_json_file(A4_FAILED_ROOT / "RUN_CONTEXT.json")
    if (
        failed_context.get("root") != str(A4_FAILED_ROOT)
        or failed_context.get("nonce") != "ac7fb52eb22be3cce184a60a41f7e98e"
        or failed_context.get("source_sha256") != A4_SOURCE_PINS
        or failed_context.get("real_data") is not False
        or failed_context.get("checkpoint_load_or_save") is not False
        or failed_context.get("token_or_claim_operation") is not False
        or failed_context.get("formal_or_av2") is not False
    ):
        _fail("failed A4 run context drifted")

    for relative, expected in R1_SOURCE_PINS.items():
        path = REPO_ROOT / relative
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or _sha(path) != expected
        ):
            _fail(f"frozen A4-R1 source pin drift: {relative}")
    r1_root_metadata = R1_FAILED_ROOT.lstat()
    if (
        not stat.S_ISDIR(r1_root_metadata.st_mode)
        or R1_FAILED_ROOT.is_symlink()
        or stat.S_IMODE(r1_root_metadata.st_mode) != 0o555
    ):
        _fail("failed A4-R1 root is not immutable 0555")
    for name, expected in R1_FAILED_PINS.items():
        path = R1_FAILED_ROOT / name
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or _sha(path) != expected
        ):
            _fail(f"failed A4-R1 artifact pin drift: {name}")
    r1_result = _strict_json_file(R1_FAILED_ROOT / "RESULT.json")
    if (
        r1_result.get("root") != str(R1_FAILED_ROOT)
        or r1_result.get("nonce") != "9e604d41d671ac4e705816857457e5a4"
        or r1_result.get("valid_result") is not False
        or r1_result.get("terminal_state") != "HARNESS_REJECTED"
        or r1_result.get("typed_verdict") != "INVALID_RUN"
        or r1_result.get("error_type") != "RuntimeError"
        or "Target sizes: [8, 5, 3, 1, 1]" not in str(r1_result.get("error", ""))
        or "Tensor sizes: [8, 5, 1]" not in str(r1_result.get("error", ""))
        or r1_result.get("automatic_av2_or_rerun") is not False
    ):
        _fail("failed A4-R1 result is not the pinned diagnostic mask shape error")

    execution_statement = authority["execution_statement"]
    execution_sha = authority["execution_statement_sha256"]
    if not isinstance(execution_statement, str) or not isinstance(execution_sha, str):
        _fail("execution statement and SHA must be strings")
    if _sha_bytes(execution_statement.encode("utf-8")) != execution_sha:
        _fail("execution statement SHA mismatch")
    required_literals = (
        args.root,
        args.nonce,
        str(args.gpu_index),
        args.gpu_uuid,
        EXPECTED_EXACT_COMMAND,
        source_hashes[str(EXPECTED_CARD)],
        *(source_hashes[path] for path in SOURCE_PATHS),
        str(A4_FAILED_ROOT),
        *(A4_FAILED_PINS[name] for name in sorted(A4_FAILED_PINS)),
        *(A4_SOURCE_PINS[name] for name in sorted(A4_SOURCE_PINS)),
        str(R1_FAILED_ROOT),
        *(R1_FAILED_PINS[name] for name in sorted(R1_FAILED_PINS)),
        *(R1_SOURCE_PINS[name] for name in sorted(R1_SOURCE_PINS)),
    )
    if any(literal not in execution_statement for literal in required_literals):
        _fail("execution statement omits a required exact runtime/source binding")

    for relative, expected in A3_SOURCE_PINS.items():
        path = REPO_ROOT / relative
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or _sha(path) != expected
        ):
            _fail(f"frozen A3 source pin drift: {relative}")
    predecessor_metadata = PREDECESSOR_ROOT.lstat()
    if (
        not stat.S_ISDIR(predecessor_metadata.st_mode)
        or PREDECESSOR_ROOT.is_symlink()
        or stat.S_IMODE(predecessor_metadata.st_mode) != 0o555
    ):
        _fail("A3 predecessor root is not immutable 0555")
    for name, expected in PREDECESSOR_PINS.items():
        path = PREDECESSOR_ROOT / name
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or _sha(path) != expected
        ):
            _fail(f"predecessor pin drift: {name}")
    predecessor_result = _strict_json_file(PREDECESSOR_ROOT / "RESULT.json")
    if (
        predecessor_result.get("valid_result") is not True
        or predecessor_result.get("typed_verdict") != "OPERATOR_GO_COMMON_STABLE"
    ):
        _fail("predecessor terminal verdict drifted")
    a4_role_paths = {
        "decision": "docs/cach_sana_wam/architecture_validation/cach_a4_state_stream/CACH_A4_STATE_CONDITIONED_CAUSAL_STREAM_ARCHITECTURE_DECISION.md",
        "card": "docs/cach_sana_wam/architecture_validation/cach_a4_state_stream/CACH_A4_STATE_CONDITIONED_CAUSAL_STREAM_RUN_CARD.json",
        "bridge": "src/sana_wam/model/cach_av1b_a4_state_conditioned_causal_odd_stream.py",
        "config": "configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream.yaml",
        "runner": "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream.py",
        "test": "tests/test_cach_av1b_a4_state_conditioned_causal_odd_stream.py",
    }
    expected_a4_source_six = {
        role: {"path": path, "sha256": A4_SOURCE_PINS[path]}
        for role, path in a4_role_paths.items()
    }
    predecessor_a4 = card.get("predecessor_a4")
    if (
        not isinstance(predecessor_a4, Mapping)
        or predecessor_a4.get("read_only") is not True
        or predecessor_a4.get("mutation_allowed") is not False
        or predecessor_a4.get("source_six") != expected_a4_source_six
    ):
        _fail("card original A4 predecessor binding drifted")
    failed_card_evidence = card.get("failed_run_evidence")
    if (
        not isinstance(failed_card_evidence, Mapping)
        or failed_card_evidence.get("root") != str(A4_FAILED_ROOT)
        or failed_card_evidence.get("nonce") != "ac7fb52eb22be3cce184a60a41f7e98e"
        or failed_card_evidence.get("terminal_state") != "HARNESS_REJECTED"
        or failed_card_evidence.get("typed_verdict") != "INVALID_RUN"
        or failed_card_evidence.get("optimizer_updates_completed") != 0
        or failed_card_evidence.get("failure_type") != "BackendCompilerFailed"
        or failed_card_evidence.get("result_sha256") != A4_FAILED_PINS["RESULT.json"]
        or failed_card_evidence.get("freeze_receipt_sha256")
        != A4_FAILED_PINS["FREEZE_RECEIPT.json"]
        or failed_card_evidence.get("run_context_sha256")
        != A4_FAILED_PINS["RUN_CONTEXT.json"]
        or failed_card_evidence.get("root_reuse_or_mutation_allowed") is not False
    ):
        _fail("card failed A4 evidence binding drifted")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        _fail("CUDA_VISIBLE_DEVICES differs from authority")
    if os.environ.get("TRITON_CACHE_DIR") != str(expected_root / "triton_cache"):
        _fail("TRITON_CACHE_DIR differs from fresh root")
    if os.environ.get("FUSED_GDN_PRECISION") != "0":
        _fail("FUSED_GDN_PRECISION must be 0")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        _fail("PYTHONDONTWRITEBYTECODE must be 1")
    if os.environ.get("GDN_DISABLE_COMPILE") != "1":
        _fail("GDN_DISABLE_COMPILE must be 1")
    if os.environ.get("TORCHDYNAMO_DISABLE") != "1":
        _fail("TORCHDYNAMO_DISABLE must be 1")
    gpu_snapshot = _gpu_idle_snapshot(args.gpu_index, args.gpu_uuid)
    return card, config, authority, source_hashes, gpu_snapshot


def _timeout(_signum: int, _frame: object) -> NoReturn:
    raise TimeoutError("A4-R2 wall-time budget exceeded")


def _interrupted(signum: int, _frame: object) -> NoReturn:
    raise RuntimeError(f"A4-R2 interrupted by signal {signum}")


def _find_numeric(value: object, names: frozenset[str]) -> list[float]:
    found: list[float] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                key in names
                and type(child) in (int, float)
                and math.isfinite(float(child))
            ):
                found.append(float(child))
            else:
                found.extend(_find_numeric(child, names))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_find_numeric(child, names))
    return found


def _require_diagnostic(value: object, *names: str) -> float:
    found = _find_numeric(value, frozenset(names))
    if not found:
        raise RuntimeError(f"missing finite diagnostic: {'/'.join(names)}")
    return found[0]


def _validate_screen(
    screen: Mapping[str, Any], config: Mapping[str, Any], gpu_uuid: str
) -> dict[str, float]:
    required = {
        "schema",
        "architecture_id",
        "metrics",
        "validity",
        "diagnostics",
        "classification_inputs",
        "typed_verdict",
        "diagnostic_subclassification",
        "theta0_manifests",
        "loss_trace",
        "final_parameter_digests",
        "jit_inventory",
        "resource_usage",
        "elapsed_seconds_total",
    }
    if not required <= set(screen) or not _finite(screen):
        raise RuntimeError("screen result is incomplete or non-finite")
    if (
        screen["schema"] != SCREEN_SCHEMA
        or screen["architecture_id"] != ARCHITECTURE_ID
    ):
        raise RuntimeError("screen schema/architecture differs")
    for name in (
        "metrics",
        "validity",
        "diagnostics",
        "classification_inputs",
        "theta0_manifests",
        "loss_trace",
        "final_parameter_digests",
        "jit_inventory",
        "resource_usage",
    ):
        if not isinstance(screen[name], Mapping):
            raise RuntimeError(f"screen.{name} must be a mapping")
    verdict_values = {
        config["verdict"][name]
        for name in (
            "valid_go_common_stable",
            "valid_go_common_blocked",
            "valid_all_weak_stop",
            "valid_other",
            "invalid",
        )
    }
    if screen["typed_verdict"] not in verdict_values:
        raise RuntimeError("typed verdict is outside the frozen config")
    diagnostics = screen["diagnostics"]
    completed = diagnostics.get(
        "optimizer_macrosteps_completed_by_arm",
        diagnostics.get("optimizer_steps_completed_by_arm"),
    )
    if completed != {"reference": 200, "candidate": 200}:
        raise RuntimeError("screen did not complete exactly 200 macrosteps per arm")
    resources = screen["resource_usage"]
    forbidden_runtime_facts = {
        "real_data_used": False,
        "checkpoint_loaded": False,
        "checkpoint_saved": False,
        "token_or_claim_operation": False,
    }
    for name, expected in forbidden_runtime_facts.items():
        if resources.get(name, False) is not expected:
            raise RuntimeError(f"forbidden runtime fact: {name}")
    jit = screen["jit_inventory"]
    if jit.get("cuda_visible_devices", gpu_uuid) != gpu_uuid:
        raise RuntimeError("JIT inventory GPU UUID differs from authority")
    if (
        jit.get("gdn_disable_compile") != "1"
        or jit.get("torchdynamo_disable") != "1"
        or jit.get("vendor_kernel_executed") is not True
        or jit.get("torch_version") != "2.7.1+cu128"
        or jit.get("triton_version") != "3.5.1"
    ):
        raise RuntimeError("A4-R2 launcher/vendor JIT evidence drifted")

    combined = {"metrics": screen["metrics"], "diagnostics": diagnostics}
    observed = {
        "future_to_prefix_leakage_max_abs": _require_diagnostic(
            combined,
            "future_to_prefix_max_abs",
            "future_to_prefix_leakage_max_abs",
            "causal_prefix_future_leakage_max_abs",
            "prefix_future_leakage_max_abs",
        ),
        "future_to_prefix_leakage_max_rel": _require_diagnostic(
            combined,
            "future_to_prefix_max_rel",
            "future_to_prefix_leakage_max_rel",
            "causal_prefix_future_leakage_rel",
            "prefix_future_leakage_max_rel",
        ),
        "common_state_to_output_jvp_rms": _require_diagnostic(
            combined, "common_state_to_output_jvp_rms"
        ),
        "past_action_to_future_output_jvp_rms": _require_diagnostic(
            combined, "past_action_to_future_output_jvp_rms"
        ),
        "delta_state_to_future_output_jvp_rms": _require_diagnostic(
            combined, "delta_state_to_future_output_jvp_rms"
        ),
        "local_action_jvp_rms": _require_diagnostic(
            combined, "action_jvp_rms", "local_action_jvp_rms"
        ),
        "state_decay_mean": _require_diagnostic(
            combined, "state_decay_mean", "decay_mean"
        ),
        "state_rms": _require_diagnostic(combined, "state_rms"),
        "delta_nmse": _require_diagnostic(
            screen["metrics"], "delta_nmse", "counterfactual_delta_nmse"
        ),
        "delta_energy_ratio": _require_diagnostic(
            screen["metrics"], "delta_energy_ratio"
        ),
        "delta_alignment_cosine": _require_diagnostic(
            screen["metrics"], "delta_alignment_cosine"
        ),
        "action_explained_fraction": _require_diagnostic(
            screen["metrics"], "action_explained_fraction"
        ),
        "no_action_recovery": _require_diagnostic(
            screen["metrics"], "no_action_recovery"
        ),
        "shuffle_penalty": _require_diagnostic(screen["metrics"], "shuffle_penalty"),
    }
    validity_thresholds = config.get("thresholds", {}).get("validity", {})
    if not isinstance(validity_thresholds, Mapping):
        raise RuntimeError("config validity thresholds must be a mapping")
    jvp_thresholds = {
        "common_state_to_output_jvp_rms": "common_state_to_output_jvp_rms_gt",
        "past_action_to_future_output_jvp_rms": "past_action_to_future_output_jvp_rms_gt",
        "delta_state_to_future_output_jvp_rms": "delta_state_to_future_output_jvp_rms_gt",
    }
    for metric_name, threshold_name in jvp_thresholds.items():
        threshold = validity_thresholds.get(threshold_name)
        if type(threshold) not in (int, float) or not math.isfinite(float(threshold)):
            raise RuntimeError(f"missing finite config threshold: {threshold_name}")
        if observed[metric_name] <= float(threshold):
            raise RuntimeError(f"causal JVP did not exceed threshold: {metric_name}")
    return observed


def _failure_placeholder(
    name: str, state: str, error_type: str, error: str
) -> dict[str, Any]:
    return {
        "schema": "cach.cach_a4_r2.failure_placeholder.v1",
        "artifact": name,
        "available": False,
        "terminal_state": state,
        "error_type": error_type,
        "error": error,
    }


def _publish_failure(
    lifecycle: RootLifecycle,
    run_context: Mapping[str, Any],
    progress: Sequence[Mapping[str, object]],
    exc: BaseException,
) -> None:
    state = "HARNESS_REJECTED"
    error_type = type(exc).__name__
    error = str(exc)
    if "RUN_CONTEXT.json" not in lifecycle.published:
        lifecycle.publish("RUN_CONTEXT.json", run_context)
    if "PROGRESS_EVENTS.json" not in lifecycle.published:
        lifecycle.publish("PROGRESS_EVENTS.json", list(progress))
    for name in REQUIRED_ARTIFACTS:
        if name in {
            "RUN_CONTEXT.json",
            "PROGRESS_EVENTS.json",
            "RESULT.json",
            "FREEZE_RECEIPT.json",
        }:
            continue
        if name not in lifecycle.published:
            lifecycle.publish(
                name, _failure_placeholder(name, state, error_type, error)
            )
    failure = {
        "schema": "cach.cach_a4_r2.failure_result.v1",
        "root": str(lifecycle.root),
        "nonce": lifecycle.nonce,
        "valid_result": False,
        "terminal_state": state,
        "typed_verdict": "INVALID_RUN",
        "error_type": error_type,
        "error": error,
        "automatic_av2_or_rerun": False,
        "formal_admission_effect": "NONE",
        "global_stage3_effect": "NONE",
        "token_or_claim_operation": False,
        "real_data_or_checkpoint_used": False,
    }
    result_sha = lifecycle.publish("RESULT.json", failure)
    lifecycle.freeze(state, result_sha)


def main() -> int:
    args = _parse_args()
    lifecycle: RootLifecycle | None = None
    try:
        card, config, authority, source_hashes, gpu_snapshot = _validate_precreate(args)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "state": "AUTH_OR_ENV_BLOCKED",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    lifecycle = RootLifecycle(Path(args.root), args.nonce)
    previous_alarm = signal.signal(signal.SIGALRM, _timeout)
    previous_handlers = {
        signum: signal.signal(signum, _interrupted)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    signal.alarm(int(config["runtime"]["max_wall_seconds"]))
    progress: list[Mapping[str, object]] = []
    run_context: dict[str, Any] = {
        "schema": "cach.cach_a4_r2.run_context.v1",
        "root": args.root,
        "nonce": args.nonce,
        "gpu_physical_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "gpu_idle_snapshot_pre_root": gpu_snapshot,
        "authority": authority,
        "authority_sha256": _sha_bytes(_canonical(authority)),
        "card_sha256": source_hashes[str(EXPECTED_CARD)],
        "source_sha256": source_hashes,
        "predecessor_pins": PREDECESSOR_PINS,
        "original_a4_source_pins": A4_SOURCE_PINS,
        "failed_a4_root": str(A4_FAILED_ROOT),
        "failed_a4_pins": A4_FAILED_PINS,
        "frozen_r1_source_pins": R1_SOURCE_PINS,
        "failed_r1_root": str(R1_FAILED_ROOT),
        "failed_r1_pins": R1_FAILED_PINS,
        "exact_execution_binding": {
            "root": str(EXPECTED_ROOT),
            "nonce": EXPECTED_NONCE,
            "gpu_physical_index": EXPECTED_GPU_INDEX,
            "gpu_uuid": EXPECTED_GPU_UUID,
            "exact_env": EXPECTED_EXACT_ENV,
            "exact_argv": EXPECTED_EXACT_ARGV,
            "exact_command": EXPECTED_EXACT_COMMAND,
        },
        "launcher_contract": {
            "gdn_disable_compile": "1",
            "torchdynamo_disable": "1",
            "unchanged_from_frozen_r1": True,
        },
        "diagnostic_mask_shape_fix": {
            "only_scientific_code_delta": (
                "correct.action_present_mask.unsqueeze(-1).unsqueeze(-1)"
                ".expand_as(correct.action_delta)"
            ),
            "architecture_task_seed_budget_threshold_delta": False,
        },
        "started_unix_ns": time.time_ns(),
        "synthetic_only": True,
        "real_data": False,
        "checkpoint_load_or_save": False,
        "token_or_claim_operation": False,
        "formal_or_av2": False,
    }
    try:
        lifecycle.create()
        lifecycle.publish("RUN_CONTEXT.json", run_context)
        post_root_gpu_snapshot = _gpu_idle_snapshot(args.gpu_index, args.gpu_uuid)
        progress.append(
            {
                "event": "post_root_pre_import_gpu_idle_snapshot",
                "payload": post_root_gpu_snapshot,
            }
        )

        def on_progress(event: Mapping[str, object]) -> None:
            if not isinstance(event, Mapping) or not _finite(event):
                raise RuntimeError("invalid progress event")
            progress.append(dict(event))

        from sana_wam.model.cach_av1b_a4_state_conditioned_causal_odd_stream_r2 import (
            run_cach_a4_screen,
        )

        screen = run_cach_a4_screen(
            config,
            expected_gpu_uuid=args.gpu_uuid,
            progress_callback=on_progress,
            source_six_and_predecessor_pins_match=True,
        )
        if not isinstance(screen, Mapping):
            raise RuntimeError("run_cach_a4_screen did not return a mapping")
        required_diagnostics = _validate_screen(screen, config, args.gpu_uuid)

        artifact_values: dict[str, object] = {
            "PROGRESS_EVENTS.json": progress,
            "THETA0_MANIFESTS.json": screen["theta0_manifests"],
            "LOSS_TRACE.json": screen["loss_trace"],
            "RAW_METRICS.json": screen["metrics"],
            "RAW_EVIDENCE.json": {
                "schema": "cach.cach_a4_r2.raw_evidence.v1",
                "validity": screen["validity"],
                "diagnostics": screen["diagnostics"],
                "classification_inputs": screen["classification_inputs"],
                "typed_verdict": screen["typed_verdict"],
                "diagnostic_subclassification": screen["diagnostic_subclassification"],
                "resource_usage": screen["resource_usage"],
                "elapsed_seconds_total": screen["elapsed_seconds_total"],
                "gpu_idle_snapshots": {
                    "pre_root": gpu_snapshot,
                    "post_root_pre_import": post_root_gpu_snapshot,
                },
                "required_causal_and_normalized_diagnostics": required_diagnostics,
                "launcher_only_repair": {
                    "gdn_disable_compile": "1",
                    "torchdynamo_disable": "1",
                    "vendor_kernel_required": True,
                },
                "synthetic_only": True,
                "token_or_claim_operation": False,
                "real_data_or_checkpoint_used": False,
            },
            "FINAL_PARAMETER_DIGESTS.json": screen["final_parameter_digests"],
            "JIT_INVENTORY.json": screen["jit_inventory"],
        }
        artifact_sha = {
            name: lifecycle.publish(name, value)
            for name, value in artifact_values.items()
        }
        if sum(row.get("size_bytes", 0) for row in _inventory(lifecycle.root)) > int(
            config["runtime"]["max_root_bytes"]
        ):
            raise RuntimeError("root byte budget exceeded")
        result = {
            "schema": "cach.cach_a4_r2.direct_final_result.v1",
            "root": args.root,
            "nonce": args.nonce,
            "architecture_id": ARCHITECTURE_ID,
            "card_sha256": source_hashes[str(EXPECTED_CARD)],
            "source_sha256": source_hashes,
            "original_a4_source_pins": A4_SOURCE_PINS,
            "failed_a4_root": str(A4_FAILED_ROOT),
            "failed_a4_pins": A4_FAILED_PINS,
            "artifact_sha256": artifact_sha,
            "valid_result": screen["typed_verdict"] != "INVALID_RUN"
            and screen["validity"].get("all_card_validity_checks_pass") is True,
            "all_card_validity_checks_pass": screen["validity"].get(
                "all_card_validity_checks_pass"
            ),
            "metrics": screen["metrics"],
            "classification_inputs": screen["classification_inputs"],
            "typed_verdict": screen["typed_verdict"],
            "terminal_state": screen["typed_verdict"],
            "diagnostic_subclassification": screen["diagnostic_subclassification"],
            "automatic_av2_or_rerun": False,
            "formal_admission_effect": "NONE",
            "global_stage3_effect": "NONE",
            "new_review_token_created_or_consumed": False,
            "claim_ledger_created_or_mutated": False,
            "real_data_or_checkpoint_used": False,
        }
        result_sha = lifecycle.publish("RESULT.json", result)
        lifecycle.freeze(str(screen["typed_verdict"]), result_sha)
        print(_canonical(result).decode("utf-8"), end="")
        return 0 if result["valid_result"] else 3
    except BaseException as exc:
        if lifecycle.created and not lifecycle.frozen:
            try:
                if "RESULT.json" not in lifecycle.published:
                    _publish_failure(lifecycle, run_context, progress, exc)
                else:
                    lifecycle.emergency_freeze()
            except BaseException:
                lifecycle.emergency_freeze()
        print(
            json.dumps(
                {"state": "HARNESS_REJECTED", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
