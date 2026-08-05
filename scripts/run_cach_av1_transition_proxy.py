#!/usr/bin/env python3
"""Run the frozen AV-1 CPU transition-proxy screen.

This launcher is deliberately self-contained.  It performs every authority,
source, environment, and root check before importing torch or creating the
run root.  Once the exact root has been created, every terminal path emits a
typed result and freezes the root; model and optimizer state are never
serialized.
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
from typing import Any, Mapping, NoReturn


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WORKTREE = Path("/home/zch/workspace/sana-wam")
EXPECTED_CONFIG = "configs/experiments/cach_av1_transition_proxy.yaml"
EXPECTED_CONFIG_SHA256 = (
    "e12f052ee63d3e93c69ddc58146b1ee18d1d1eef39c8e85b2fa38e11f91c6361"
)
EXPECTED_CARD = "docs/cach_sana_wam/architecture_validation/av1/AV0_RUN_CARD.json"
EXPECTED_CARD_SHA256 = (
    "c73a943047295338cd9de14e6e53199b463ea66ae92b99d4b0b3393f5ec34030"
)
EXPECTED_PLAN_SHA256 = (
    "06f5d09127f8bc2a945cdbdc095054f40b1fe8d9757a90eccbdf0604ea89e079"
)
EXPECTED_FIXED_LIST_SHA256 = (
    "59ff254f82f546a8f0102d7fa03a340392da4a879a46ae8e267357db762cb216"
)
EXPECTED_AUTHORITY_SHA256 = (
    "d6007c24ae618896aef4b679999afeb51c0a2f5f279f1cc2f6c0e2d46e77ef5c"
)
EXPECTED_NONCE = "d755b67c667d9a22686c8d90d187832d"
EXPECTED_ROOT = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/av1/06f5d09127f8/"
    "c73a94304729-d755b67c667d9a22686c8d90d187832d"
)
EXPECTED_REPO_HEADS = {
    "sana_wam": "605f1c134b4c983ff80f8489c4bc8847036329e2",
    "third_party_sana": "16b9cec673e3335724ba2d8db25de7f9ed229292",
    "sana_afcc_handoff": "9586486f2a9f5172d57b325e32093a3e018d34c0",
}
AV1_CODE_PATHS = (
    "src/sana_wam/model/cach_av1_transition_proxy.py",
    EXPECTED_CONFIG,
    "scripts/run_cach_av1_transition_proxy.py",
    "tests/test_cach_av1_transition_proxy.py",
)
CARD_ADDITIVE_PATHS = (EXPECTED_CARD, *AV1_CODE_PATHS)
ALLOWED_TERMINAL_STATES = frozenset(
    {
        "AUTH_BLOCKED",
        "ENV_BLOCKED",
        "HARNESS_REJECTED",
        "INVALID_RUN",
        "IMPLEMENTATION_INVALID",
        "NUMERICAL_INVALID",
        "VALID_REVIEW_PROVISIONAL_NO_VERDICT",
        "PROXY_GO",
        "PROXY_STOP",
        "PROXY_INCONCLUSIVE",
    }
)


class PreflightError(RuntimeError):
    """A fail-closed error raised before the run root exists."""

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
        "CONFIG_SCHEMA_MISMATCH",
        f"{label} keys: expected={sorted(expected)!r} actual={sorted(actual)!r}",
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
            "ENV_BLOCKED",
            "GIT_HEAD_UNAVAILABLE",
            f"{path}: {process.stderr.strip()}",
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
        raise PreflightError(
            "ENV_BLOCKED", "PINNED_SOURCE_MISSING", relative
        ) from exc
    _require(
        stat.S_ISREG(info.st_mode) and not path.is_symlink(),
        "ENV_BLOCKED",
        "PINNED_SOURCE_NOT_REGULAR",
        relative,
    )
    return path


def _verify_pin(relative: str, expected_sha256: str) -> None:
    path = _regular_repo_file(relative)
    observed = _sha256_file(path)
    _require(
        observed == expected_sha256,
        "AUTH_BLOCKED",
        "SOURCE_PIN_MISMATCH",
        f"{relative}: expected={expected_sha256} observed={observed}",
    )


def _validate_config(config: Mapping[str, Any], config_argument: Path) -> None:
    _require(
        _sha256_file(config_argument.resolve(strict=True))
        == EXPECTED_CONFIG_SHA256,
        "AUTH_BLOCKED",
        "CONFIG_PIN_MISMATCH",
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
            "metrics",
            "thresholds",
            "artifacts",
        },
        "config",
    )
    _require(
        config.get("schema") == "cach.av1.transition_proxy.config.v1",
        "AUTH_BLOCKED",
        "CONFIG_SCHEMA_MISMATCH",
        repr(config.get("schema")),
    )
    expected_config_path = REPO_ROOT / EXPECTED_CONFIG
    _require(
        config_argument.resolve(strict=True) == expected_config_path.resolve(strict=True),
        "AUTH_BLOCKED",
        "CONFIG_PATH_MISMATCH",
        str(config_argument),
    )

    identity = config["identity"]
    _require(isinstance(identity, dict), "AUTH_BLOCKED", "BAD_IDENTITY", "not mapping")
    expected_identity = {
        "canonical_host": "H200",
        "canonical_worktree": str(EXPECTED_WORKTREE),
        "run_card_path": EXPECTED_CARD,
        "run_card_sha256": EXPECTED_CARD_SHA256,
        "governing_plan_path": (
            "docs/cach_sana_wam/global_stage3/post_phase_c_planning/"
            "ARCHITECTURE_VALIDATION_FIRST_PLAN_20260803_v1.md"
        ),
        "governing_plan_sha256": EXPECTED_PLAN_SHA256,
        "phase_c_fixed_member_list_path": (
            "docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/"
            "PHASE_C_SOURCE_FILES.txt"
        ),
        "phase_c_fixed_member_list_sha256": EXPECTED_FIXED_LIST_SHA256,
        "resolved_root": str(EXPECTED_ROOT),
        "run_nonce": EXPECTED_NONCE,
    }
    _require(
        identity == expected_identity,
        "AUTH_BLOCKED",
        "IDENTITY_BINDING_MISMATCH",
        repr(identity),
    )

    authority = config["authority"]
    _require(isinstance(authority, dict), "AUTH_BLOCKED", "BAD_AUTHORITY", "not mapping")
    statement = authority.get("statement")
    _require(
        isinstance(statement, str)
        and _sha256_bytes(statement.encode("utf-8")) == EXPECTED_AUTHORITY_SHA256
        and authority.get("statement_sha256") == EXPECTED_AUTHORITY_SHA256,
        "AUTH_BLOCKED",
        "AUTHORITY_PIN_MISMATCH",
        "authority statement or literal pin differs",
    )
    required_true = {
        "code_write",
        "screen_root_create",
        "cpu_execute",
        "model_construct",
        "forward_backward",
        "jvp_diagnostic",
        "optimizer_construct",
        "parameter_update",
        "lightweight_tests",
        "training",
        "diagnostic_metrics",
    }
    required_false = {
        "gpu_or_cuda_or_jit",
        "real_data_access",
        "checkpoint_load",
        "checkpoint_save",
        "full_2b",
        "formal_evaluation",
        "global_stage3",
    }
    for name in required_true:
        _require(
            authority.get(name) is True,
            "AUTH_BLOCKED",
            "CAPABILITY_NOT_AUTHORIZED",
            name,
        )
    for name in required_false:
        _require(
            authority.get(name) is False,
            "AUTH_BLOCKED",
            "FORBIDDEN_CAPABILITY_ENABLED",
            name,
        )

    runtime = config["runtime"]
    _require(
        runtime
        == {
            "device": "cpu",
            "dtype": "float32",
            "single_process": True,
            "cuda_visible_devices": "",
            "allow_cuda_initialization": False,
            "allow_compile_or_jit": False,
            "allow_checkpoint": False,
            "allow_real_data": False,
            "allow_network": False,
        },
        "AUTH_BLOCKED",
        "RUNTIME_SCOPE_MISMATCH",
        repr(runtime),
    )
    budget = config["budget"]
    _require(
        budget.get("optimizer_steps_per_arm") == 200
        and budget.get("final_step") == 200
        and budget.get("metric_steps") == [0, 200]
        and budget.get("max_wall_seconds_total") == 900
        and budget.get("max_process_rss_bytes") == 4_294_967_296
        and budget.get("max_root_bytes") == 67_108_864
        and budget.get("best_step_selection") is False
        and budget.get("intermediate_checkpoint_selection") is False,
        "AUTH_BLOCKED",
        "BUDGET_MISMATCH",
        repr(budget),
    )
    artifacts = config["artifacts"]
    _require(
        artifacts.get("write_model_checkpoint") is False
        and artifacts.get("write_optimizer_state") is False
        and artifacts.get("freeze_root_on_every_postcreate_terminal") is True,
        "AUTH_BLOCKED",
        "ARTIFACT_SCOPE_MISMATCH",
        repr(artifacts),
    )


def _validate_card_and_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    card_path = _regular_repo_file(EXPECTED_CARD)
    _require(
        _sha256_file(card_path) == EXPECTED_CARD_SHA256,
        "AUTH_BLOCKED",
        "RUN_CARD_PIN_MISMATCH",
        EXPECTED_CARD,
    )
    try:
        card = _duplicate_rejecting_json(card_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PreflightError(
            "AUTH_BLOCKED", "RUN_CARD_PARSE_REJECTED", str(exc)
        ) from exc
    _require(
        card.get("schema") == "cach.architecture_validation.av0_run_card.v1"
        and card.get("card_id") == "cach-av1-transition-proxy-v1",
        "AUTH_BLOCKED",
        "RUN_CARD_IDENTITY_MISMATCH",
        "schema/card_id",
    )
    card_paths = card.get("additive_policy", {}).get("exact_additive_paths")
    _require(
        card_paths == list(CARD_ADDITIVE_PATHS),
        "AUTH_BLOCKED",
        "ADDITIVE_PATH_SET_MISMATCH",
        repr(card_paths),
    )
    allowed_states = card.get("verdict_contract", {}).get("av1_allowed_states")
    _require(
        isinstance(allowed_states, list)
        and set(allowed_states) == set(ALLOWED_TERMINAL_STATES),
        "AUTH_BLOCKED",
        "TERMINAL_STATE_SET_MISMATCH",
        repr(allowed_states),
    )
    expected_heads = card.get("identity", {}).get("repository_heads")
    _require(
        expected_heads == EXPECTED_REPO_HEADS,
        "AUTH_BLOCKED",
        "REPOSITORY_HEAD_CARD_MISMATCH",
        repr(expected_heads),
    )

    for item in card.get("identity", {}).get("ancestor_pins", {}).values():
        _verify_pin(item["path"], item["sha256"])
    for item in card.get("experiment_contract", {}).get(
        "implementation_sources_observed_not_executed", {}
    ).values():
        _verify_pin(item["path"], item["sha256"])

    identity = config["identity"]
    _verify_pin(identity["governing_plan_path"], identity["governing_plan_sha256"])
    _verify_pin(
        identity["phase_c_fixed_member_list_path"],
        identity["phase_c_fixed_member_list_sha256"],
    )
    fixed_members = {
        line.strip()
        for line in _regular_repo_file(identity["phase_c_fixed_member_list_path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    _require(
        fixed_members.isdisjoint(CARD_ADDITIVE_PATHS),
        "AUTH_BLOCKED",
        "PHASE_C_FIXED_MEMBER_OVERLAP",
        repr(sorted(fixed_members.intersection(CARD_ADDITIVE_PATHS))),
    )
    manifest_path = _regular_repo_file(
        "docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/"
        "SOURCE_MANIFEST.json"
    )
    try:
        manifest = _duplicate_rejecting_json(manifest_path)
    except (OSError, ValueError) as exc:
        raise PreflightError(
            "AUTH_BLOCKED", "PHASE_C_MANIFEST_PARSE_REJECTED", str(exc)
        ) from exc
    entries = manifest.get("files")
    _require(
        manifest.get("schema") == "cach.global_stage3.phase_c.source_manifest.v1"
        and isinstance(entries, list)
        and len(entries) == 18,
        "AUTH_BLOCKED",
        "PHASE_C_MANIFEST_SCHEMA_MISMATCH",
        "schema/files",
    )
    manifest_members: set[str] = set()
    for entry in entries:
        _require(
            isinstance(entry, dict)
            and set(entry) == {"path", "sha256", "size_bytes"}
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("sha256"), str)
            and isinstance(entry.get("size_bytes"), int),
            "AUTH_BLOCKED",
            "PHASE_C_MANIFEST_ENTRY_INVALID",
            repr(entry),
        )
        relative = entry["path"]
        pinned_path = _regular_repo_file(relative)
        _require(
            pinned_path.stat().st_size == entry["size_bytes"],
            "AUTH_BLOCKED",
            "PHASE_C_SOURCE_SIZE_MISMATCH",
            relative,
        )
        _verify_pin(relative, entry["sha256"])
        manifest_members.add(relative)
    _require(
        manifest_members == fixed_members,
        "AUTH_BLOCKED",
        "PHASE_C_MANIFEST_MEMBER_MISMATCH",
        repr(sorted(manifest_members.symmetric_difference(fixed_members))),
    )
    dependency_entries = manifest.get("runtime_dependency_pins")
    _require(
        isinstance(dependency_entries, list) and dependency_entries,
        "AUTH_BLOCKED",
        "RUNTIME_DEPENDENCY_PINS_MISSING",
        repr(dependency_entries),
    )
    dependency_paths: set[str] = set()
    for entry in dependency_entries:
        _require(
            isinstance(entry, dict)
            and set(entry) == {"path", "sha256"}
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("sha256"), str),
            "AUTH_BLOCKED",
            "RUNTIME_DEPENDENCY_PIN_INVALID",
            repr(entry),
        )
        _verify_pin(entry["path"], entry["sha256"])
        dependency_paths.add(entry["path"])
    _require(
        len(dependency_paths) == len(dependency_entries),
        "AUTH_BLOCKED",
        "RUNTIME_DEPENDENCY_PIN_DUPLICATE",
        repr(sorted(dependency_paths)),
    )
    for relative in AV1_CODE_PATHS:
        _regular_repo_file(relative)
    return card


def _validate_environment(root_argument: Path, config: Mapping[str, Any]) -> None:
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
        root_argument.name
        == f"{EXPECTED_CARD_SHA256[:12]}-{config['identity']['run_nonce']}",
        "AUTH_BLOCKED",
        "ROOT_TEMPLATE_MISMATCH",
        root_argument.name,
    )
    _require(
        not root_argument.exists() and not root_argument.is_symlink(),
        "ENV_BLOCKED",
        "ROOT_ALREADY_EXISTS",
        str(root_argument),
    )
    parent = root_argument.parent
    try:
        parent_info = parent.lstat()
        resolved_parent = parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreflightError(
            "ENV_BLOCKED", "ROOT_PARENT_MISSING", str(parent)
        ) from exc
    _require(
        stat.S_ISDIR(parent_info.st_mode)
        and not parent.is_symlink()
        and resolved_parent == parent,
        "ENV_BLOCKED",
        "ROOT_PARENT_UNSAFE",
        str(parent),
    )
    _require(
        parent_info.st_uid == os.geteuid(),
        "ENV_BLOCKED",
        "ROOT_PARENT_OWNER_MISMATCH",
        f"owner={parent_info.st_uid} euid={os.geteuid()}",
    )
    _require(
        os.access(parent, os.W_OK | os.X_OK),
        "ENV_BLOCKED",
        "ROOT_PARENT_NOT_WRITABLE",
        str(parent),
    )
    free_bytes = shutil.disk_usage(parent).free
    required_free = int(config["budget"]["max_root_bytes"]) * 2
    _require(
        free_bytes >= required_free,
        "ENV_BLOCKED",
        "INSUFFICIENT_ROOT_CAPACITY",
        f"free={free_bytes} required={required_free}",
    )
    _require(
        "torch" not in sys.modules,
        "ENV_BLOCKED",
        "TORCH_IMPORTED_BEFORE_PREFLIGHT",
        "torch already present in sys.modules",
    )
    _require(
        "sana_wam" not in sys.modules
        and not any(name.startswith("sana_wam.") for name in sys.modules),
        "ENV_BLOCKED",
        "LOCAL_PACKAGE_IMPORTED_BEFORE_PREFLIGHT",
        "sana_wam already present in sys.modules",
    )
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        value = os.environ.get(name)
        _require(
            value in (None, "", "1"),
            "ENV_BLOCKED",
            "MULTIPROCESS_ENV_FORBIDDEN",
            f"{name}={value!r}",
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


def _source_pins(config_path: Path) -> dict[str, str]:
    pins = {relative: _sha256_file(_regular_repo_file(relative)) for relative in AV1_CODE_PATHS}
    pins[EXPECTED_CARD] = EXPECTED_CARD_SHA256
    pins[str(config_path.relative_to(REPO_ROOT))] = _sha256_file(config_path)
    return dict(sorted(pins.items()))


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
        self.publish_json("RUN_CONTEXT.json", self.context)

    def publish_bytes(self, name: str, payload: bytes, mode: int = 0o600) -> str:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"unsafe artifact name: {name!r}")
        destination = self.root / name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
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

    def _inventory(self) -> list[dict[str, Any]]:
        inventory: list[dict[str, Any]] = []
        for child in sorted(self.root.iterdir(), key=lambda item: item.name.encode("utf-8")):
            if child.name == "FREEZE_RECEIPT.json":
                continue
            info = child.lstat()
            if not stat.S_ISREG(info.st_mode) or child.is_symlink():
                raise RuntimeError(f"unsafe artifact before freeze: {child}")
            inventory.append(
                {
                    "path": child.name,
                    "sha256": _sha256_file(child),
                    "size_bytes": info.st_size,
                }
            )
        return inventory

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

    def _recursive_inventory(self) -> list[dict[str, Any]]:
        inventory: list[dict[str, Any]] = []
        for directory, directory_names, file_names in os.walk(
            self.root, followlinks=False
        ):
            directory_path = Path(directory)
            for name in sorted((*directory_names, *file_names)):
                child = directory_path / name
                relative = child.relative_to(self.root).as_posix()
                if relative == "FREEZE_RECEIPT.json":
                    continue
                info = child.lstat()
                if stat.S_ISREG(info.st_mode):
                    inventory.append(
                        {
                            "path": relative,
                            "kind": "regular",
                            "sha256": _sha256_file(child),
                            "size_bytes": info.st_size,
                        }
                    )
                elif stat.S_ISDIR(info.st_mode):
                    inventory.append({"path": relative, "kind": "directory"})
                elif stat.S_ISLNK(info.st_mode):
                    inventory.append(
                        {
                            "path": relative,
                            "kind": "forbidden_symlink",
                            "target": os.readlink(child),
                        }
                    )
                else:
                    inventory.append({"path": relative, "kind": "forbidden_other"})
        return inventory

    def finalize(
        self,
        *,
        state: str,
        validity_reasons: list[str],
        raw_evidence: Mapping[str, Any],
        source_pins: Mapping[str, str],
        authority: Mapping[str, Any],
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
            raw_sha = self.publish_json("RAW_EVIDENCE.json", raw_evidence)
            result = {
                "schema": "cach.av1.transition_proxy.result.v1",
                "exact_typed_state": state,
                "operator_level": "TRANSITION_PROXY",
                "integration_path": "EXPERIMENTAL_PATH",
                "result_scope": "PROXY_ONLY_ACTUAL_ARCHITECTURE_NOT_ASSESSED",
                "validity_reason": validity_reasons,
                "raw_evidence_path": "RAW_EVIDENCE.json",
                "raw_evidence_sha256": raw_sha,
                "run_card_self_path_and_full_sha256": {
                    "path": EXPECTED_CARD,
                    "sha256": EXPECTED_CARD_SHA256,
                },
                "review_token_id_and_consumption_state": {
                    "token_id": (
                        "cadfd92f1c72ebd9cc9aa8bfad029ddc6f0e72d9f7942f34e74629ca1ae37252"
                    ),
                    "state": "UNCONSUMED",
                },
                "code_and_config_pins": dict(source_pins),
                "resolved_root": str(self.root),
                "root_lifecycle_and_freeze_receipt": {
                    "exclusive_create": True,
                    "terminal_root_frozen": True,
                    "freeze_receipt_path": "FREEZE_RECEIPT.json",
                    "freeze_receipt_pin_location": "launcher_stdout_after_publish",
                },
                "authority_pin": {
                    "statement_sha256": authority["statement_sha256"],
                    "statement_sha256_rule": authority["statement_sha256_rule"],
                },
                "formal_admission_evidence": False,
                "global_stage3_effect": "NONE",
            }
            result_sha = self.publish_json("RESULT.json", result)
            elapsed = time.monotonic() - started_monotonic
            receipt = {
                "schema": "cach.av1.transition_proxy.freeze_receipt.v1",
                "resolved_root": str(self.root),
                "terminal_state": state,
                "result_path": "RESULT.json",
                "result_sha256": result_sha,
                "inventory_excludes": ["FREEZE_RECEIPT.json"],
                "inventory": self._inventory(),
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
            }
        finally:
            self._finalizing = False

    def emergency_finalize(self) -> None:
        if not self.created or self.finalized or self._finalizing:
            return
        try:
            payload = {
                "schema": "cach.av1.transition_proxy.emergency_failure.v1",
                "exact_typed_state": "IMPLEMENTATION_INVALID",
                "validity_reason": ["PROCESS_EXITED_BEFORE_NORMAL_FINALIZATION"],
            }
            failure_sha: str | None = None
            try:
                failure_sha = self.publish_json("EMERGENCY_FAILURE.json", payload)
            except FileExistsError:
                failure_sha = _sha256_file(self.root / "EMERGENCY_FAILURE.json")
            if not (self.root / "RESULT.json").exists():
                emergency_result = {
                    "schema": "cach.av1.transition_proxy.result.v1",
                    "exact_typed_state": "IMPLEMENTATION_INVALID",
                    "operator_level": "TRANSITION_PROXY",
                    "integration_path": "EXPERIMENTAL_PATH",
                    "result_scope": "PROXY_ONLY_ACTUAL_ARCHITECTURE_NOT_ASSESSED",
                    "validity_reason": [
                        "PROCESS_EXITED_BEFORE_NORMAL_FINALIZATION"
                    ],
                    "raw_evidence_path": "EMERGENCY_FAILURE.json",
                    "raw_evidence_sha256": failure_sha,
                    "run_card_self_path_and_full_sha256": {
                        "path": EXPECTED_CARD,
                        "sha256": EXPECTED_CARD_SHA256,
                    },
                    "review_token_id_and_consumption_state": {
                        "token_id": (
                            "cadfd92f1c72ebd9cc9aa8bfad029ddc6f0e72d9f7942f34e74629ca1ae37252"
                        ),
                        "state": "UNCONSUMED",
                    },
                    "code_and_config_pins": self.context.get("source_pins", {}),
                    "resolved_root": str(self.root),
                    "root_lifecycle_and_freeze_receipt": {
                        "exclusive_create": True,
                        "terminal_root_frozen": True,
                        "freeze_receipt_path": "FREEZE_RECEIPT.json",
                    },
                    "authority_pin": {
                        "statement_sha256": EXPECTED_AUTHORITY_SHA256,
                        "statement_sha256_rule": (
                            "SHA256_UTF8_WITHOUT_TRAILING_NEWLINE"
                        ),
                    },
                    "formal_admission_evidence": False,
                    "global_stage3_effect": "NONE",
                }
                self.publish_json("RESULT.json", emergency_result)
            try:
                if not (self.root / "FREEZE_RECEIPT.json").exists():
                    receipt = {
                        "schema": "cach.av1.transition_proxy.emergency_freeze_receipt.v1",
                        "resolved_root": str(self.root),
                        "terminal_state": "IMPLEMENTATION_INVALID",
                        "inventory_excludes": ["FREEZE_RECEIPT.json"],
                        "inventory": self._recursive_inventory(),
                        "immutable": True,
                    }
                    self.publish_json("FREEZE_RECEIPT.json", receipt)
            finally:
                self._freeze()
                self.finalized = True
        except Exception:
            try:
                os.chmod(self.root, 0o555)
            except Exception:
                pass


def _install_failure_traps(lifecycle: RootLifecycle) -> None:
    def handler(signum: int, _frame: Any) -> NoReturn:
        raise RunInterrupted(f"received signal {signum}")

    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT", "SIGALRM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)
    atexit.register(lifecycle.emergency_finalize)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


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
    if name not in metrics:
        raise KeyError(f"missing metric: {name}")
    value = metrics[name]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"metric is not numeric: {name}")
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite metric: {name}")
    return float(value)


def _horizon_metric(metrics: Mapping[str, Any], name: str, horizon: int) -> float:
    value = metrics.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"missing horizon metric mapping: {name}")
    selected = value.get(str(horizon), value.get(horizon))
    if selected is None:
        raise KeyError(f"missing metric: {name}[{horizon}]")
    if not isinstance(selected, (int, float)) or isinstance(selected, bool):
        raise TypeError(f"metric is not numeric: {name}[{horizon}]")
    if not math.isfinite(selected):
        raise FloatingPointError(f"non-finite metric: {name}[{horizon}]")
    return float(selected)


def _screen_evidence_errors(
    screen: Mapping[str, Any], config: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    steps = int(config["budget"]["optimizer_steps_per_arm"])
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
    exact_diagnostics = {
        "operator_level": "TRANSITION_PROXY",
        "integration_path": "EXPERIMENTAL_PATH",
        "result_scope": "PROXY_ONLY_ACTUAL_ARCHITECTURE_NOT_ASSESSED",
        "device": "cpu",
        "dtype": "float32",
        "registered_200_step_budget_executed": True,
        "state_order_independent": True,
        "empty_state_zero_tensor_equality": True,
        "no_action_frozen_exact_zero": True,
    }
    for name, expected in exact_diagnostics.items():
        if diagnostics.get(name) != expected:
            errors.append(f"DIAGNOSTIC_{name.upper()}_INVALID")
    jvp = diagnostics.get("seam_directional_jvp_rms_theta0")
    if not _finite_number(jvp) or float(jvp) <= 0.0:
        errors.append("DIAGNOSTIC_SEAM_JVP_NOT_LIVE")
    action_diagnostic = diagnostics.get("action_diagnostic")
    if (
        not isinstance(action_diagnostic, Mapping)
        or action_diagnostic.get("class") != "ActionDiT"
        or action_diagnostic.get("depth") != 20
        or action_diagnostic.get("frozen") is not True
        or float(action_diagnostic.get("loss_weight", -1.0)) != 0.0
    ):
        errors.append("DIAGNOSTIC_ACTIONDIT_INVALID")
    target = diagnostics.get("target_leakage")
    if (
        not isinstance(target, Mapping)
        or target.get("target_perturbation_changes_prediction") is not False
        or target.get("target_perturbation_changes_state") is not False
        or target.get("target_reverse_gradient_nonzero") is not False
        or target.get("target_jvp_exact_zero") is not True
        or float(target.get("target_jvp_rms", math.inf)) != 0.0
    ):
        errors.append("DIAGNOSTIC_TARGET_LEAKAGE_INVALID")
    commit = diagnostics.get("commit_diagnostic")
    if (
        not isinstance(commit, Mapping)
        or commit.get("update_free") is not True
        or commit.get("target_bytes_read") is not False
        or commit.get("reference_final_chunk_id") != 2
        or commit.get("candidate_final_chunk_id") != 2
        or commit.get("reference_action_cursor") != 32
        or commit.get("candidate_action_cursor") != 32
    ):
        errors.append("DIAGNOSTIC_COMMIT_INVALID")
    task = diagnostics.get("task")
    if not isinstance(task, Mapping):
        errors.append("DIAGNOSTIC_TASK_INVALID")
    else:
        pair_delta = task.get("target_pair_delta_rms_by_horizon")
        if not isinstance(pair_delta, Mapping):
            errors.append("DIAGNOSTIC_TARGET_PAIR_DELTA_INVALID")
        else:
            for horizon in (1, 2, 3, 4):
                value = pair_delta.get(str(horizon), pair_delta.get(horizon))
                if not _finite_number(value) or not (0.999998 <= float(value) <= 1.000002):
                    errors.append(f"DIAGNOSTIC_TARGET_PAIR_DELTA_H{horizon}_INVALID")

    metrics = screen.get("metrics")
    if isinstance(metrics, Mapping):
        reference_delta = metrics.get("reference_delta_nmse_by_horizon")
        if not isinstance(reference_delta, Mapping):
            errors.append("REFERENCE_DELTA_NMSE_INVALID")
        else:
            for horizon in (1, 2, 3, 4):
                value = reference_delta.get(str(horizon), reference_delta.get(horizon))
                if not _finite_number(value) or abs(float(value) - 1.0) > 1.0e-6:
                    errors.append(f"REFERENCE_DELTA_NMSE_H{horizon}_INVALID")
    runtime = screen.get("resource_usage")
    forbidden_runtime = {
        "filesystem_writes": 0,
        "run_root_created": False,
        "gpu_used": False,
        "checkpoint_loaded": False,
        "checkpoint_saved": False,
        "real_data_used": False,
    }
    if not isinstance(runtime, Mapping) or any(
        runtime.get(name) != expected for name, expected in forbidden_runtime.items()
    ):
        errors.append("RUNTIME_SCOPE_EVIDENCE_INVALID")
    return errors


def _classify_screen(
    screen: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, list[str]]:
    expected_screen_keys = {
        "schema",
        "validity",
        "metrics",
        "loss_trace",
        "theta0_manifests",
        "final_parameter_digests",
        "diagnostics",
        "resource_usage",
    }
    if set(screen) != expected_screen_keys or screen.get("schema") != (
        "cach.av1.transition_proxy.screen.v1"
    ):
        return "IMPLEMENTATION_INVALID", ["SCREEN_SCHEMA_MISMATCH"]
    evidence_errors = _screen_evidence_errors(screen, config)
    if evidence_errors:
        return "IMPLEMENTATION_INVALID", evidence_errors
    validity = screen.get("validity")
    metrics = screen.get("metrics")
    if not isinstance(validity, Mapping) or not isinstance(metrics, Mapping):
        return "IMPLEMENTATION_INVALID", ["MISSING_VALIDITY_OR_METRICS"]
    supplied_reasons = validity.get("reasons", [])
    reasons = [str(item) for item in supplied_reasons] if isinstance(supplied_reasons, list) else []
    if validity.get("implementation_valid") is not True:
        return "IMPLEMENTATION_INVALID", reasons or ["IMPLEMENTATION_VALIDITY_FAILED"]
    if validity.get("data_valid") is not True:
        return "INVALID_RUN", reasons or ["DATA_VALIDITY_FAILED"]
    if validity.get("numerics_valid") is not True or validity.get("finite_all") is not True:
        return "NUMERICAL_INVALID", reasons or ["NUMERICAL_VALIDITY_FAILED"]
    if validity.get("theta0_exact_identity") is not True:
        return "IMPLEMENTATION_INVALID", reasons or ["THETA0_IDENTITY_FAILED"]

    try:
        values = {
            "seam_grad": _metric(metrics, "seam_grad_rms_step0"),
            "seam_update": _metric(metrics, "seam_update_rms_final"),
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
    except FloatingPointError as exc:
        return "NUMERICAL_INVALID", ["METRIC_NONFINITE", str(exc)]
    except (KeyError, TypeError, ValueError) as exc:
        return "IMPLEMENTATION_INVALID", ["METRIC_CONTRACT_FAILED", str(exc)]

    go = config["thresholds"]["proxy_go"]
    go_checks = {
        "seam_grad_rms_step0": values["seam_grad"] > go["seam_grad_rms_step0_gt"],
        "seam_update_rms_final": values["seam_update"] > go["seam_update_rms_final_gt"],
        "loss_drop_rel": values["loss_drop"] >= go["loss_drop_rel_min"],
        "counterfactual_delta_nmse_correct": values["dnmse"] <= go["counterfactual_delta_nmse_correct_max"],
        "shuffle_gap_rel": values["shuffle"] >= go["shuffle_gap_rel_min"],
        "no_action_gap_rel": values["no_action"] >= go["no_action_gap_rel_min"],
        "candidate_gain_vs_reference": values["gain"] >= go["candidate_gain_vs_reference_min"],
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
        return "PROXY_GO", ["ALL_PROXY_GO_THRESHOLDS_MET"]

    stop = config["thresholds"]["proxy_strong_stop"]
    stop_checks = {
        "seam_grad_rms_step0": values["seam_grad"] > stop["seam_grad_rms_step0_gt"],
        "seam_update_rms_final": values["seam_update"] > stop["seam_update_rms_final_gt"],
        "loss_drop_rel": values["loss_drop"] <= stop["loss_drop_rel_max"],
        "counterfactual_delta_nmse_correct": values["dnmse"] >= stop["counterfactual_delta_nmse_correct_min"],
        "shuffle_gap_rel": values["shuffle"] <= stop["shuffle_gap_rel_max"],
        "no_action_gap_rel": values["no_action"] <= stop["no_action_gap_rel_max"],
        "candidate_gain_vs_reference": values["gain"] <= stop["candidate_gain_vs_reference_max"],
    }
    if all(stop_checks.values()):
        return "PROXY_STOP", ["ALL_PROXY_STRONG_STOP_THRESHOLDS_MET"]
    failed_go = sorted(name for name, passed in go_checks.items() if not passed)
    failed_stop = sorted(name for name, passed in stop_checks.items() if not passed)
    return "PROXY_INCONCLUSIVE", [
        "VALID_REVIEW_BAND",
        "GO_FAILED:" + ",".join(failed_go),
        "STRONG_STOP_FAILED:" + ",".join(failed_stop),
    ]


def _screen_artifacts(screen: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "RAW_METRICS.json": screen.get("metrics", {}),
        "LOSS_TRACE.json": screen.get("loss_trace", {}),
        "THETA0_MANIFESTS.json": screen.get("theta0_manifests", {}),
        "FINAL_PARAMETER_DIGESTS.json": screen.get("final_parameter_digests", {}),
        "DIAGNOSTICS.json": screen.get("diagnostics", {}),
    }


def _publish_or_match_json(
    lifecycle: RootLifecycle, name: str, value: Any
) -> str:
    payload = _canonical_json_bytes(value)
    destination = lifecycle.root / name
    if destination.exists():
        observed = destination.read_bytes()
        if observed != payload:
            raise RuntimeError(f"pre-forward artifact changed after publication: {name}")
        return _sha256_bytes(observed)
    return lifecycle.publish_bytes(name, payload)


def _postcreate_failure_state(exc: BaseException) -> tuple[str, list[str]]:
    if isinstance(exc, (TimeoutError, RunInterrupted, MemoryError)):
        return "HARNESS_REJECTED", [type(exc).__name__, str(exc)]
    if isinstance(exc, OSError) and exc.errno in {
        getattr(os, "ENOSPC", 28),
        getattr(os, "EFBIG", 27),
        getattr(os, "EDQUOT", 122),
    }:
        return "HARNESS_REJECTED", [type(exc).__name__, str(exc)]
    if isinstance(exc, ModuleNotFoundError):
        return "ENV_BLOCKED", ["RUNTIME_DEPENDENCY_MISSING", str(exc)]
    if isinstance(exc, (FloatingPointError, OverflowError)):
        return "NUMERICAL_INVALID", [type(exc).__name__, str(exc)]
    return "IMPLEMENTATION_INVALID", [type(exc).__name__, str(exc)]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--card-sha256", default=EXPECTED_CARD_SHA256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started_monotonic = time.monotonic()
    arguments = _parse_args(argv)
    config_path = Path(arguments.config)
    root_argument = Path(arguments.root)
    lifecycle: RootLifecycle | None = None
    config: Mapping[str, Any] | None = None
    pins: dict[str, str] = {}
    try:
        _require(
            arguments.card_sha256 == EXPECTED_CARD_SHA256,
            "AUTH_BLOCKED",
            "CALLER_CARD_PIN_MISMATCH",
            arguments.card_sha256,
        )
        try:
            config = _duplicate_rejecting_yaml(config_path)
        except PreflightError:
            raise
        except (OSError, ValueError) as exc:
            raise PreflightError(
                "AUTH_BLOCKED", "CONFIG_PARSE_REJECTED", str(exc)
            ) from exc
        _validate_config(config, config_path)
        _validate_card_and_sources(config)
        _validate_environment(root_argument, config)
        pins = _source_pins(config_path.resolve(strict=True))

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["GDN_DISABLE_COMPILE"] = "1"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        context = {
            "schema": "cach.av1.transition_proxy.run_context.v1",
            "canonical_host_alias": "H200",
            "observed_hostname": platform.node(),
            "canonical_worktree": str(REPO_ROOT),
            "resolved_root": str(root_argument),
            "run_nonce": EXPECTED_NONCE,
            "pid": os.getpid(),
            "device": "cpu",
            "dtype": "float32",
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "source_pins": pins,
            "authority_statement_sha256": EXPECTED_AUTHORITY_SHA256,
            "formal_admission_evidence": False,
        }
        lifecycle = RootLifecycle(root_argument, context)
        _install_failure_traps(lifecycle)
        lifecycle.create()
    except PreflightError as exc:
        event = {
            "schema": "cach.av1.transition_proxy.precreate_block.v1",
            "exact_typed_state": exc.state,
            "validity_reason": [exc.token, exc.detail],
            "resolved_root": None,
            "root_created": False,
        }
        sys.stderr.buffer.write(_canonical_json_bytes(event))
        return 2
    except Exception as exc:
        if lifecycle is not None and lifecycle.created:
            state, reasons = _postcreate_failure_state(exc)
            raw_evidence = {
                "schema": "cach.av1.transition_proxy.raw_evidence.v1",
                "proposed_typed_state": state,
                "validity_reason": reasons,
                "screen": {
                    "schema": "cach.av1.transition_proxy.failed_create.v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "resource_usage": {
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                    "process_peak_rss_bytes": _rss_bytes(),
                    "root_size_bytes_before_terminal_artifacts": _root_size(
                        lifecycle.root
                    ),
                },
                "forbidden_artifacts_written": False,
                "model_or_optimizer_state_serialized": False,
            }
            try:
                terminal = lifecycle.finalize(
                    state=state,
                    validity_reasons=reasons,
                    raw_evidence=raw_evidence,
                    source_pins=pins,
                    authority=config["authority"] if config is not None else {
                        "statement_sha256": EXPECTED_AUTHORITY_SHA256,
                        "statement_sha256_rule": (
                            "SHA256_UTF8_WITHOUT_TRAILING_NEWLINE"
                        ),
                    },
                    started_monotonic=started_monotonic,
                )
                sys.stdout.buffer.write(_canonical_json_bytes(terminal))
            except BaseException:
                lifecycle.emergency_finalize()
            return 2
        event = {
            "schema": "cach.av1.transition_proxy.precreate_block.v1",
            "exact_typed_state": "ENV_BLOCKED",
            "validity_reason": [type(exc).__name__, str(exc)],
            "resolved_root": None,
            "root_created": False,
        }
        sys.stderr.buffer.write(_canonical_json_bytes(event))
        return 2

    assert lifecycle is not None and config is not None and lifecycle.created
    screen: Mapping[str, Any] = {}
    state = "IMPLEMENTATION_INVALID"
    reasons: list[str] = ["SCREEN_DID_NOT_START"]
    try:
        remaining = float(config["budget"]["max_wall_seconds_total"]) - (
            time.monotonic() - started_monotonic
        )
        if remaining <= 0:
            raise TimeoutError("wall budget exhausted before screen")
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, remaining)

        sys.path.insert(0, str(REPO_ROOT / "src"))
        from sana_wam.model.cach_av1_transition_proxy import run_av1_screen

        progress_events: list[dict[str, Any]] = []
        theta0_published = False

        def progress_callback(event: Mapping[str, Any]) -> None:
            nonlocal theta0_published
            if not isinstance(event, Mapping):
                raise TypeError("progress event must be a mapping")
            materialized = dict(event)
            _canonical_json_bytes(materialized)
            if len(progress_events) >= 512:
                raise RuntimeError("progress event limit exceeded")
            progress_events.append(materialized)
            if materialized.get("event") == "theta0_manifests_ready":
                manifests = materialized.get(
                    "payload", materialized.get("theta0_manifests")
                )
                if not isinstance(manifests, Mapping):
                    raise TypeError("theta0 progress event lacks manifests")
                _publish_or_match_json(
                    lifecycle, "THETA0_MANIFESTS.json", dict(manifests)
                )
                theta0_published = True

        result = run_av1_screen(config, progress_callback=progress_callback)
        if not isinstance(result, Mapping):
            raise TypeError("run_av1_screen must return a mapping")
        screen = dict(result)
        _assert_no_nonfinite(screen)
        _canonical_json_bytes(screen)
        if not theta0_published:
            raise RuntimeError("theta0 manifests were not published before first forward")
        lifecycle.publish_json("PROGRESS_EVENTS.json", progress_events)
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)

        for artifact_name, artifact_value in _screen_artifacts(screen).items():
            _publish_or_match_json(lifecycle, artifact_name, artifact_value)

        post_screen_pins = _source_pins(config_path.resolve(strict=True))
        if post_screen_pins != pins:
            raise RuntimeError("code or config bytes changed during AV-1 execution")
        _validate_card_and_sources(config)

        elapsed = time.monotonic() - started_monotonic
        rss = _rss_bytes()
        root_bytes = _root_size(lifecycle.root)
        terminal_reserve_bytes = len(_canonical_json_bytes(screen)) + 1_048_576
        budget_reasons: list[str] = []
        if elapsed + 2.0 > float(config["budget"]["max_wall_seconds_total"]):
            budget_reasons.append("WALL_BUDGET_EXCEEDED")
        if rss + 67_108_864 > int(config["budget"]["max_process_rss_bytes"]):
            budget_reasons.append("RSS_BUDGET_EXCEEDED")
        if root_bytes + terminal_reserve_bytes > int(
            config["budget"]["max_root_bytes"]
        ):
            budget_reasons.append("ROOT_BUDGET_EXCEEDED")
        completed = screen.get("diagnostics", {}).get("optimizer_steps_completed_by_arm")
        expected_steps = int(config["budget"]["optimizer_steps_per_arm"])
        exact_completed = {
            "REF-GDN-CORRECTED": expected_steps,
            "CACH-A": expected_steps,
        }
        if (
            not isinstance(completed, Mapping)
            or any(isinstance(value, bool) for value in completed.values())
            or dict(completed) != exact_completed
        ):
            budget_reasons.append("EXACT_OPTIMIZER_STEPS_NOT_COMPLETED")
        if budget_reasons:
            state, reasons = "HARNESS_REJECTED", budget_reasons
        else:
            state, reasons = _classify_screen(screen, config)
    except BaseException as exc:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        state, reasons = _postcreate_failure_state(exc)
        screen = {
            "schema": "cach.av1.transition_proxy.failed_screen.v1",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    raw_evidence = {
        "schema": "cach.av1.transition_proxy.raw_evidence.v1",
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
    }
    try:
        terminal = lifecycle.finalize(
            state=state,
            validity_reasons=reasons,
            raw_evidence=raw_evidence,
            source_pins=pins,
            authority=config["authority"],
            started_monotonic=started_monotonic,
        )
    except BaseException as exc:
        lifecycle.emergency_finalize()
        sys.stderr.write(f"terminal finalization failed: {type(exc).__name__}: {exc}\n")
        return 3
    sys.stdout.buffer.write(_canonical_json_bytes(terminal))
    return 0 if state in {"PROXY_GO", "PROXY_STOP", "PROXY_INCONCLUSIVE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
