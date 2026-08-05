#!/usr/bin/env python3
"""Fail-closed verifier for the scoped CACH Stage-2B additive artifact.

This script authenticates source and archive bytes.  It does not run tests,
replay the live lightweight execution, or grant runtime, training, evaluation,
deployment, capture, scientific, Gate-S2, or Stage-3 authority.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/cach_sana_wam/stage2b/SOURCE_MANIFEST.json"
CANONICAL_HOST = "H200"
CANONICAL_WORKTREE = "/home/zch/workspace/sana-wam"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
RFC3339_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?Z\Z"
)

STAGE2B_SOURCE_DATE_EPOCH = 1_785_456_000
STAGE2B_REPORT_SHA256 = (
    "1aae6daaed746c79a692f410e118aba85d3a9dbbf56d1e5c6d79e4a704bf661f"
)

EXPECTED_HEADS = {
    "sana_afcc_handoff_head": "9586486f2a9f5172d57b325e32093a3e018d34c0",
    "sana_head": "16b9cec673e3335724ba2d8db25de7f9ed229292",
    "sana_wam_head": "605f1c134b4c983ff80f8489c4bc8847036329e2",
    "sana_wam_gitlink_for_sana": "16b9cec673e3335724ba2d8db25de7f9ed229292",
}

STAGE0_BUNDLE_PIN = {
    "mode": "0444",
    "path": "/DATA/share/sana_cach_source_bundles/cach_stage0_85d16ee5_20260731.tar",
    "sha256": "bd427ef6a604abef49c75b20d14a21d9412f5c21b539a1240fb052187f8889fd",
    "size_bytes": 645_120,
}
STAGE1_BUNDLE_PIN = {
    "mode": "0444",
    "path": (
        "/DATA/share/sana_cach_source_bundles/"
        "cach_stage1_a6e7849b91464b6e939db6ac676ca8e0d139a8e889e2b6fedb730457d3ee31dc_"
        "20260731.tar"
    ),
    "sha256": "a6e7849b91464b6e939db6ac676ca8e0d139a8e889e2b6fedb730457d3ee31dc",
    "size_bytes": 870_400,
}
STAGE2_BUNDLE_PIN = {
    "mode": "0444",
    "path": (
        "/DATA/share/sana_cach_source_bundles/"
        "cach_stage2_71e73b3f2b1efb125f0b57f8c47913bab47d492920310f0a53848fe7903c229c_"
        "20260731.tar"
    ),
    "sha256": "71e73b3f2b1efb125f0b57f8c47913bab47d492920310f0a53848fe7903c229c",
    "size_bytes": 327_680,
}

EXPECTED_BASE_LINEAGE = {
    **EXPECTED_HEADS,
    "stage0_authority": {
        "path": "docs/cach_sana_wam/stage0/CACH_A_AUTHORITY.draft.json",
        "sha256": (
            "cdc38feede2e151c1de4070942aede1ba5af16f98f13326dd2f083708ffd00b6"
        ),
    },
    "stage0_source_bundle": STAGE0_BUNDLE_PIN,
    "stage0_source_manifest": {
        "path": "docs/cach_sana_wam/stage0/SOURCE_MANIFEST.draft.json",
        "sha256": (
            "71f3ed9b0ab431f73742ba9bb6ceb21a74b6379f84e288a1e2178432e47e9277"
        ),
    },
    "stage1_authority": {
        "path": "docs/cach_sana_wam/stage1/CACH_STAGE1_AUTHORITY.draft.json",
        "sha256": (
            "c9870136cdf6297cd0d74d90037525941b35c199a1dcf598cf4312a010e7241a"
        ),
    },
    "stage1_source_bundle": STAGE1_BUNDLE_PIN,
    "stage1_source_manifest": {
        "path": "docs/cach_sana_wam/stage1/SOURCE_MANIFEST.draft.json",
        "sha256": (
            "6d27becd6ac14d8553080c3c2c01bb31b334f13e3c7f31da5b91fbf589d1e287"
        ),
    },
    "stage2_source_bundle": STAGE2_BUNDLE_PIN,
    "stage2_source_manifest": {
        "path": "docs/cach_sana_wam/stage2/SOURCE_MANIFEST.json",
        "sha256": (
            "8333c74d9bc8115937f98d9fc12a60da62cab49dd8895cd403047027315052fb"
        ),
    },
}

EXPECTED_STAGE2B_FILES = [
    "docs/cach_sana_wam/stage2b/README.md",
    "docs/cach_sana_wam/stage2b/STAGE2B_DURABLE_LEDGER_AND_REPLAY_DESIGN_20260731.md",
    "docs/cach_sana_wam/stage2b/STAGE2B_LIGHTWEIGHT_TEST_REPORT.json",
    "docs/cach_sana_wam/stage2b/STAGE2B_OFFLINE_PRODUCTION_INTEGRATION_PLAN_20260731.md",
    "docs/cach_sana_wam/stage2b/STAGE2B_SOURCE_FILES.txt",
    "scripts/verify_cach_stage2b.py",
    "src/sana_wam/cach/committed_action_history.py",
    "src/sana_wam/cach/stage2b_receipt_store.py",
    "src/sana_wam/cach/stage2b_state_snapshot.py",
    "src/sana_wam/cach/staging_variant.py",
    "src/sana_wam/model/cach_numerical_core.py",
    "src/sana_wam/model/cach_paired_stager.py",
    "src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b.py",
    "tests/test_cach_stage2b_committed_action_history.py",
    "tests/test_cach_stage2b_production_core.py",
    "tests/test_cach_stage2b_receipt_store.py",
    "tests/test_cach_stage2b_state_snapshot.py",
]

EXPECTED_EXECUTION_HARNESS_PINS = [
    {
        "path": ".gitmodules",
        "sha256": "22e41c488f68ca762c6433e8a53bb504cdcddd899b2e7f4aa6935ae67297d066",
    },
    {
        "path": ".python-version",
        "sha256": "7b55f8e67b5623c4bef3fa691288da9437d79d3aba156de48d481db32ac7d16d",
    },
    {
        "path": "pyproject.toml",
        "sha256": "c84bce95525fcbe03800602bd6dc3e6a151f53888b0dbc91c405ac9bf6f6bafa",
    },
    {
        "path": "src/sana_wam/__init__.py",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    {
        "path": "src/sana_wam/cach/__init__.py",
        "sha256": "0daf40668bd7042ff0de6d2c33a045e13d32f17722df3d915aa255268091c75b",
    },
    {
        "path": "src/sana_wam/model/__init__.py",
        "sha256": "78e9e9403c02e3af004d18b2a8026647fc0e5e873a8684b4f96251b9cc530f0d",
    },
    {
        "path": "src/sana_wam/model/video_backbone/__init__.py",
        "sha256": "72958a5200f4494a8740f4ff71db0d3d4489a84d2ae48e8e2555c0b35658eef9",
    },
    {
        "path": "src/sana_wam/model/video_backbone/sana/__init__.py",
        "sha256": "dc4acb3ca1022a0ed62904a61d09394ed03e7e8fedb022ebf6d25b2871f0d75f",
    },
    {
        "path": "tests/__init__.py",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    {
        "path": "tests/conftest.py",
        "sha256": "893ce1a02854323e0ebdd3574a74364c6f15aa489e8f8fd7187126efd81d39fc",
    },
    {
        "path": "uv.lock",
        "sha256": "31803c58048ff07600116ac389bd025f4def00cea0282a29637017b4acfec08c",
    },
]

EXPECTED_INHERITED_SOURCE_PINS = [
    {
        "path": "scripts/verify_cach_stage2.py",
        "sha256": "1a2961770b3004e9b481a68fd92b724e2505c31253a170d8c40a8d5582bcf6ea",
    },
    {
        "path": "src/sana_wam/cach/action_conditioning.py",
        "sha256": "8323efa1544d5c0520e13192b55ed5f2e85490d03ac21c558d6009db38ee11c6",
    },
    {
        "path": "src/sana_wam/cach/authority.py",
        "sha256": "3f729d61fe47fddc864ba4cd831e1b358e50444a338e7221fc60b42db5934f22",
    },
    {
        "path": "src/sana_wam/cach/prefix_compaction.py",
        "sha256": "1a2f12f38bb9699c65af7e4ea72183f9d122b6f81aae4d9628129040a3cfcdb7",
    },
    {
        "path": "src/sana_wam/model/action_chunk_layout.py",
        "sha256": "8e6521a93614b4ce7e43ba51d2af4f67d2650b03cc60e7d7a152e36dfa3c8e39",
    },
    {
        "path": "src/sana_wam/model/causal_action_hybrid.py",
        "sha256": "e714ea6ade41223ed1b93e643cbd179fa63ff7ab48d568f36dc95e5eaab17560",
    },
    {
        "path": "src/sana_wam/model/video_backbone/adapter.py",
        "sha256": "6dd6cb5d8825d1275b397780e496dca1e6d4572124add4df814741c571d2a425",
    },
    {
        "path": "src/sana_wam/model/video_backbone/sana/adapter.py",
        "sha256": "fdda391181e6f30455cf54fe1559b4bd2ee33bc9274b42712f0f5e826aef58d5",
    },
    {
        "path": "src/sana_wam/model/video_backbone/sana/blocks_split.py",
        "sha256": "723437935867c7cf013ac686b8fdec546a0edc6035fb63b0bbecc32fe3652b7f",
    },
    {
        "path": "src/sana_wam/model/video_backbone/sana/hybrid_cache.py",
        "sha256": "5b61ab6492670e8021b3603f965e9f3e8921e9d7f0ea77b49e40d92a77f78a74",
    },
    {
        "path": "src/sana_wam/model/video_backbone/sana/hybrid_cache_codec.py",
        "sha256": "563454779c5e45ceee4dbae9a99f2636114cc7645a5492fab2b487e05f59e638",
    },
    {
        "path": "src/sana_wam/model/video_backbone/sana/pipeline_builder.py",
        "sha256": "2a7e4f8688b1bec04a4290b6a5deb1a51064d9477ac34c9de3135d618acaba20",
    },
    {
        "path": "third_party/Sana/diffusion/model/nets/basic_modules.py",
        "sha256": "f5b8345d7ba35a32f612ebc2e98976d1707198bb8e24ccd95dfbfa3b4a77e2ca",
    },
    {
        "path": "third_party/Sana/diffusion/model/nets/sana_gdn_blocks.py",
        "sha256": "ef15578c63f9815d670dc3ef4cf2876c4f236754bfdfb0d2070da387048a495e",
    },
    {
        "path": "third_party/Sana/diffusion/model/nets/sana_gdn_camctrl_blocks.py",
        "sha256": "6e6859c0e35a810538f2c046a285e88d06978c56df17451b62a508ec56154de7",
    },
    {
        "path": "third_party/Sana/diffusion/model/nets/sana_multi_scale_video_camctrl.py",
        "sha256": "bfbd72dfb22a44e2843f1985049ecfbb1f887f5e5f5ed852905c4670e2029fbe",
    },
    {
        "path": "third_party/Sana/diffusion/model/ops/fused_gdn.py",
        "sha256": "863cbbb601ddebc5666179e2b209b777d5dc213c94a1450cde695a33b0a8aded",
    },
    {
        "path": "third_party/Sana/diffusion/model/ops/fused_streaming.py",
        "sha256": "4775d1ea1101d7075d087428cf2f8bb2643a7de922fd241e15b65e3c4f4ea206",
    },
    {
        "path": "third_party/Sana/diffusion/scheduler/self_forcing_flow_euler_sampler.py",
        "sha256": "8c246685cb2a7e9cf5c8954afdc7b39973876fe403a662929764788406bec023",
    },
]

EXPECTED_SCOPE = {
    "execution_scope": "synthetic_cpu_offline_stage2b_only",
    "file_count": 17,
    "kind": "stage2b_delta_subset_over_stage2_bundle",
    "source_manifest_excluded_to_avoid_self_reference": True,
    "transitive_runtime_closure": False,
}

EXPECTED_OVERLAY = {
    "order": ["stage0", "stage1", "stage2", "stage2b"],
    "pairwise_collision_allowlist": {
        "stage0_stage1": [
            "benchmarks/robotwin/eval_policy_wrapper.py",
            "docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md",
            "src/sana_wam/deploy/__init__.py",
            "src/sana_wam/deploy/model_loader.py",
            "src/sana_wam/deploy/policy_server.py",
        ],
        "stage0_stage2": [
            "docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md"
        ],
        "stage0_stage2b": [],
        "stage1_stage2": [
            "docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md",
            "src/sana_wam/model/video_backbone/sana/adapter.py",
            "tests/test_cach_action_to_video_conditioning.py",
        ],
        "stage1_stage2b": [],
        "stage2_stage2b": [],
    },
    "stage2b_requires_zero_predecessor_collisions": True,
}

EXPECTED_GIT_RECOVERY_BUNDLES = {
    "local_afcc_handoff": None,
    "sana_submodule": None,
    "sana_wam": None,
}

MANIFEST_KEYS = {
    "base_lineage",
    "canonical_host",
    "canonical_worktree",
    "created_at",
    "execution_harness_pins",
    "files",
    "git_recovery_bundles",
    "inherited_source_pins",
    "overlay",
    "retained_stage2b_evidence",
    "schema",
    "scope",
    "stage2b_source_bundle",
    "status",
}


class Stage2BVerificationError(RuntimeError):
    """Raised when any required additive-artifact invariant differs."""


def _fail(message: str) -> None:
    raise Stage2BVerificationError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _exact_json_equal(observed: object, expected: object) -> bool:
    """Compare decoded JSON without Python's bool/int/float coercions."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _exact_json_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _exact_json_equal(observed_value, expected_value)
            for observed_value, expected_value in zip(observed, expected, strict=True)
        )
    return observed == expected


def _stable_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_rdev,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_ctime_ns,
    )


def _ancestor_signatures(path: Path, name: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    result: list[tuple[str, tuple[int, ...]]] = []
    for part in absolute.parts[1:-1]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _fail(f"{name} ancestor is not a nofollow directory: {current}")
        result.append((os.fspath(current), _directory_signature(metadata)))
    return tuple(result)


def _stable_read_regular(
    path: Path,
    name: str,
    *,
    expected_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    if not hasattr(os, "O_NOFOLLOW"):
        _fail("O_NOFOLLOW is required for Stage-2B verification")
    ancestors_before = _ancestor_signatures(path, name)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{name} is not a regular file")
        if before.st_nlink != 1:
            _fail(f"{name} must have exactly one hard link")
        if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
            _fail(f"{name} mode differs")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stable_signature(before) != _stable_signature(after):
            _fail(f"{name} changed during stable read")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            _fail(f"{name} size changed during stable read")
        path_after = os.stat(path, follow_symlinks=False)
        if (
            path_after.st_dev != before.st_dev
            or path_after.st_ino != before.st_ino
            or stat.S_ISLNK(path_after.st_mode)
        ):
            _fail(f"{name} path no longer resolves to the opened inode")
    finally:
        os.close(descriptor)
    if ancestors_before != _ancestor_signatures(path, name):
        _fail(f"{name} ancestor changed during stable read")
    return raw, before


def _strict_json(raw: bytes, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            _fail(f"{name} contains non-finite float {token!r}")
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Stage2BVerificationError(
                    f"{name} contains non-finite JSON token {token!r}"
                )
            ),
            parse_float=finite_float,
        )
    except UnicodeDecodeError as exc:
        raise Stage2BVerificationError(f"{name} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise Stage2BVerificationError(f"{name} is not valid JSON") from exc
    except ValueError as exc:
        raise Stage2BVerificationError(
            f"{name} contains a JSON number outside the accepted range"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"{name} must contain one JSON object")
    return value


def _canonical_repo_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        _fail("repository path must be a non-empty canonical POSIX string")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"non-canonical repository path: {value!r}")
    return value


def _repository_file(relative_path: object) -> Path:
    canonical = _canonical_repo_path(relative_path)
    candidate = ROOT.joinpath(*PurePosixPath(canonical).parts)
    candidate.relative_to(ROOT)
    return candidate


def _canonical_tar_name(name: str, *, directory: bool) -> str:
    if type(name) is not str or not name or "\\" in name or "\x00" in name:
        _fail("tar member has an invalid name")
    logical = name[:-1] if directory and name.endswith("/") else name
    canonical = _canonical_repo_path(logical)
    if directory and name not in {canonical, f"{canonical}/"}:
        _fail(f"tar directory name is non-canonical: {name!r}")
    if not directory and name != canonical:
        _fail(f"tar file name is non-canonical: {name!r}")
    return canonical


def _repo_pin(path: str, sha256: str) -> dict[str, str]:
    return {"path": path, "sha256": sha256}


def _verify_repo_pin(pin: object, name: str) -> bytes:
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
        _fail(f"{name} repository pin schema differs")
    path = _repository_file(pin["path"])
    expected_sha256 = pin["sha256"]
    if type(expected_sha256) is not str or SHA256_RE.fullmatch(expected_sha256) is None:
        _fail(f"{name} repository pin SHA256 is invalid")
    raw, _ = _stable_read_regular(path, name)
    if _sha256(raw) != expected_sha256:
        _fail(f"{name} repository bytes differ")
    return raw


def _validate_external_pin(pin: object, name: str) -> tuple[Path, str, int]:
    if not isinstance(pin, dict) or set(pin) != {
        "mode",
        "path",
        "sha256",
        "size_bytes",
    }:
        _fail(f"{name} external pin schema differs")
    if pin["mode"] != "0444":
        _fail(f"{name} mode pin must be 0444")
    path_value = pin["path"]
    if type(path_value) is not str or not path_value.startswith("/"):
        _fail(f"{name} path must be absolute")
    pure = PurePosixPath(path_value)
    if pure.as_posix() != path_value or any(part in {".", ".."} for part in pure.parts):
        _fail(f"{name} path is non-canonical")
    expected_sha256 = pin["sha256"]
    if type(expected_sha256) is not str or SHA256_RE.fullmatch(expected_sha256) is None:
        _fail(f"{name} SHA256 pin is invalid")
    expected_size = pin["size_bytes"]
    if type(expected_size) is not int or expected_size <= 0:
        _fail(f"{name} size pin is invalid")
    return Path(path_value), expected_sha256, expected_size


def _read_external_pin(pin: object, name: str) -> tuple[Path, bytes]:
    path, expected_sha256, expected_size = _validate_external_pin(pin, name)
    raw, metadata = _stable_read_regular(path, name, expected_mode=0o444)
    if metadata.st_size != expected_size or _sha256(raw) != expected_sha256:
        _fail(f"{name} size or bytes differ")
    return path, raw


def _git_output(cwd: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _verify_repository_lineage() -> None:
    observed = {
        "sana_afcc_handoff_head": _git_output(
            ROOT.parent / "sana-afcc-handoff", "rev-parse", "HEAD"
        ),
        "sana_head": _git_output(ROOT / "third_party/Sana", "rev-parse", "HEAD"),
        "sana_wam_head": _git_output(ROOT, "rev-parse", "HEAD"),
    }
    if observed != {key: EXPECTED_HEADS[key] for key in observed}:
        _fail("repository HEAD lineage differs")
    gitlink = _git_output(ROOT, "ls-tree", "HEAD", "--", "third_party/Sana")
    match = re.fullmatch(
        r"160000 commit ([0-9a-f]{40})\tthird_party/Sana",
        gitlink,
    )
    if match is None or match.group(1) != EXPECTED_HEADS["sana_wam_gitlink_for_sana"]:
        _fail("main repository Sana gitlink differs")


def _read_tar_members(
    raw: bytes,
    name: str,
    *,
    allow_directories: bool,
) -> tuple[list[str], dict[str, str], dict[str, bytes], int]:
    names: list[str] = []
    digests: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    directory_count = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                if not allow_directories:
                    _fail(f"{name} contains an unexpected directory member")
                _canonical_tar_name(member.name, directory=True)
                directory_count += 1
                continue
            if not member.isfile():
                _fail(f"{name} contains a non-regular member: {member.name!r}")
            canonical = _canonical_tar_name(member.name, directory=False)
            if canonical in payloads:
                _fail(f"{name} contains duplicate member {canonical!r}")
            extracted = archive.extractfile(member)
            if extracted is None:
                _fail(f"{name} member is unreadable: {canonical!r}")
            payload = extracted.read()
            if len(payload) != member.size:
                _fail(f"{name} member size changed: {canonical!r}")
            names.append(canonical)
            payloads[canonical] = payload
            digests[canonical] = _sha256(payload)
    return names, digests, payloads, directory_count


def _manifest_file_map(
    manifest: dict[str, Any],
    name: str,
) -> tuple[list[str], dict[str, str]]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        _fail(f"{name} files are absent")
    paths: list[str] = []
    digests: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            _fail(f"{name} file pin schema differs")
        path = _canonical_repo_path(entry["path"])
        digest = entry["sha256"]
        if type(digest) is not str or SHA256_RE.fullmatch(digest) is None:
            _fail(f"{name} file SHA256 is invalid")
        paths.append(path)
        digests[path] = digest
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        _fail(f"{name} file paths are not sorted and unique")
    return paths, digests


def _verify_embedded_source_list(
    payloads: dict[str, bytes],
    source_list_path: str,
    expected_paths: list[str],
    name: str,
) -> None:
    raw = payloads.get(source_list_path)
    if raw is None:
        _fail(f"{name} source list is absent from its archive")
    expected = ("\n".join(expected_paths) + "\n").encode()
    if raw != expected:
        _fail(f"{name} source list differs from its archive inventory")


def _verify_ancestor_chain() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    for key in (
        "stage0_authority",
        "stage0_source_manifest",
        "stage1_authority",
        "stage1_source_manifest",
        "stage2_source_manifest",
    ):
        _verify_repo_pin(EXPECTED_BASE_LINEAGE[key], key)

    _, stage0_raw = _read_external_pin(STAGE0_BUNDLE_PIN, "Stage-0 source bundle")
    _, stage1_raw = _read_external_pin(STAGE1_BUNDLE_PIN, "Stage-1 source bundle")
    _, stage2_raw = _read_external_pin(STAGE2_BUNDLE_PIN, "Stage-2 source bundle")

    stage0_names, stage0_map, _, stage0_directories = _read_tar_members(
        stage0_raw,
        "Stage-0 source bundle",
        allow_directories=True,
    )
    if len(stage0_names) != 40 or stage0_directories != 2:
        _fail("Stage-0 archive inventory count differs")

    stage1_manifest_raw = _verify_repo_pin(
        EXPECTED_BASE_LINEAGE["stage1_source_manifest"],
        "Stage-1 source manifest",
    )
    stage1_manifest = _strict_json(stage1_manifest_raw, "Stage-1 source manifest")
    stage1_expected_paths, stage1_expected_map = _manifest_file_map(
        stage1_manifest,
        "Stage-1 source manifest",
    )
    stage1_names, stage1_map, stage1_payloads, stage1_directories = _read_tar_members(
        stage1_raw,
        "Stage-1 source bundle",
        allow_directories=False,
    )
    if (
        stage1_directories != 0
        or stage1_names != stage1_expected_paths
        or stage1_map != stage1_expected_map
    ):
        _fail("Stage-1 archive member map differs from its manifest")
    _verify_embedded_source_list(
        stage1_payloads,
        "docs/cach_sana_wam/stage1/STAGE1_SOURCE_FILES.txt",
        stage1_expected_paths,
        "Stage-1",
    )

    stage2_manifest_raw = _verify_repo_pin(
        EXPECTED_BASE_LINEAGE["stage2_source_manifest"],
        "Stage-2 source manifest",
    )
    stage2_manifest = _strict_json(stage2_manifest_raw, "Stage-2 source manifest")
    stage2_expected_paths, stage2_expected_map = _manifest_file_map(
        stage2_manifest,
        "Stage-2 source manifest",
    )
    stage2_names, stage2_map, stage2_payloads, stage2_directories = _read_tar_members(
        stage2_raw,
        "Stage-2 source bundle",
        allow_directories=False,
    )
    if (
        stage2_directories != 0
        or stage2_names != stage2_expected_paths
        or stage2_map != stage2_expected_map
    ):
        _fail("Stage-2 archive member map differs from its manifest")
    _verify_embedded_source_list(
        stage2_payloads,
        "docs/cach_sana_wam/stage2/STAGE2_SOURCE_FILES.txt",
        stage2_expected_paths,
        "Stage-2",
    )
    return stage0_map, stage1_map, stage2_map


def _verify_overlay(
    stage0: dict[str, str],
    stage1: dict[str, str],
    stage2: dict[str, str],
) -> None:
    stage2b = set(EXPECTED_STAGE2B_FILES)
    maps = {
        "stage0_stage1": sorted(set(stage0) & set(stage1)),
        "stage0_stage2": sorted(set(stage0) & set(stage2)),
        "stage0_stage2b": sorted(set(stage0) & stage2b),
        "stage1_stage2": sorted(set(stage1) & set(stage2)),
        "stage1_stage2b": sorted(set(stage1) & stage2b),
        "stage2_stage2b": sorted(set(stage2) & stage2b),
    }
    if maps != EXPECTED_OVERLAY["pairwise_collision_allowlist"]:
        _fail("ancestor or Stage-2B overlay collision map differs")
    predecessor_union = set(stage0) | set(stage1) | set(stage2)
    if len(predecessor_union) != 88:
        _fail("predecessor resolved inventory count differs")
    if predecessor_union & stage2b or len(predecessor_union | stage2b) != 105:
        _fail("Stage-2B is not a zero-collision 17-member delta")


def _verify_pinned_repo_set(
    observed: object,
    expected: list[dict[str, str]],
    name: str,
) -> None:
    if not _exact_json_equal(observed, expected):
        _fail(f"{name} declarations differ")
    paths = [entry["path"] for entry in expected]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        _fail(f"internal {name} constants are not sorted and unique")
    for entry in expected:
        _verify_repo_pin(entry, f"{name} {entry['path']}")


def _verify_stage2b_files(
    files: object,
) -> tuple[dict[str, bytes], dict[str, str]]:
    if not isinstance(files, list) or len(files) != len(EXPECTED_STAGE2B_FILES):
        _fail("Stage-2B file inventory count differs")
    paths: list[str] = []
    source_bytes: dict[str, bytes] = {}
    source_digests: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            _fail("Stage-2B file pin schema differs")
        path = _canonical_repo_path(entry["path"])
        digest = entry["sha256"]
        size = entry["size_bytes"]
        if type(digest) is not str or SHA256_RE.fullmatch(digest) is None:
            _fail(f"Stage-2B file SHA256 is invalid: {path}")
        if type(size) is not int or size < 0:
            _fail(f"Stage-2B file size is invalid: {path}")
        raw, metadata = _stable_read_regular(
            _repository_file(path),
            f"Stage-2B source {path}",
        )
        if metadata.st_size != size or _sha256(raw) != digest:
            _fail(f"Stage-2B source bytes differ: {path}")
        paths.append(path)
        source_bytes[path] = raw
        source_digests[path] = digest
    if paths != EXPECTED_STAGE2B_FILES:
        _fail("Stage-2B paths differ from the exact sorted 17-member inventory")

    expected_list = ("\n".join(EXPECTED_STAGE2B_FILES) + "\n").encode()
    source_list_path = "docs/cach_sana_wam/stage2b/STAGE2B_SOURCE_FILES.txt"
    if source_bytes[source_list_path] != expected_list:
        _fail("Stage-2B source-list bytes differ")

    report_path = "docs/cach_sana_wam/stage2b/STAGE2B_LIGHTWEIGHT_TEST_REPORT.json"
    report_raw = source_bytes[report_path]
    if _sha256(report_raw) != STAGE2B_REPORT_SHA256:
        _fail("Stage-2B lightweight report differs from the fixed live record")
    report = _strict_json(report_raw, "Stage-2B lightweight report")
    if (
        report.get("schema") != "cach.stage2b.live_lightweight_report.v1"
        or report.get("status")
        != "live_lightweight_l0_l1_passed_l2_l3_blocked_non_scientific"
        or report.get("canonical_host") != CANONICAL_HOST
        or report.get("canonical_worktree") != CANONICAL_WORKTREE
    ):
        _fail("Stage-2B lightweight report identity differs")
    return source_bytes, source_digests


def _canonical_stage2b_tar(source_bytes: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output,
        mode="w:",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for path in EXPECTED_STAGE2B_FILES:
            payload = source_bytes[path]
            member = tarfile.TarInfo(path)
            member.type = tarfile.REGTYPE
            member.mode = 0o644
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.linkname = ""
            member.mtime = STAGE2B_SOURCE_DATE_EPOCH
            member.size = len(payload)
            member.pax_headers = {}
            archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _verify_stage2b_bundle(
    pin: object,
    source_bytes: dict[str, bytes],
    source_digests: dict[str, str],
) -> None:
    path, expected_sha256, _ = _validate_external_pin(pin, "Stage-2B source bundle")
    expected_parent = Path("/DATA/share/sana_cach_source_bundles")
    match = re.fullmatch(
        r"cach_stage2b_([0-9a-f]{64})_20260731\.tar",
        path.name,
    )
    if (
        path.parent != expected_parent
        or match is None
        or match.group(1) != expected_sha256
    ):
        _fail("Stage-2B bundle path is not digest-bound")
    _, raw = _read_external_pin(pin, "Stage-2B source bundle")
    if raw != _canonical_stage2b_tar(source_bytes):
        _fail("Stage-2B tar bytes are not the canonical deterministic USTAR")
    if len(raw) % 512 != 0:
        _fail("Stage-2B tar size is not block-aligned")

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        if archive.pax_headers:
            _fail("Stage-2B tar has global PAX headers")
        members = archive.getmembers()
        if [member.name for member in members] != EXPECTED_STAGE2B_FILES:
            _fail("Stage-2B tar member ordering differs")
        for member in members:
            if (
                member.type != tarfile.REGTYPE
                or member.mode != 0o644
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.linkname != ""
                or member.mtime != STAGE2B_SOURCE_DATE_EPOCH
                or member.pax_headers
            ):
                _fail(f"Stage-2B tar metadata differs: {member.name}")
            _canonical_tar_name(member.name, directory=False)
            extracted = archive.extractfile(member)
            if extracted is None:
                _fail(f"Stage-2B tar member is unreadable: {member.name}")
            payload = extracted.read()
            if (
                len(payload) != member.size
                or payload != source_bytes[member.name]
                or _sha256(payload) != source_digests[member.name]
            ):
                _fail(f"Stage-2B tar member bytes differ: {member.name}")


def _manifest_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute != DEFAULT_MANIFEST:
        _fail("Stage-2B manifest must use the canonical default repository path")
    return absolute


def verify(manifest_path: Path, expected_manifest_sha256: str) -> dict[str, object]:
    if SHA256_RE.fullmatch(expected_manifest_sha256) is None:
        _fail("expected manifest SHA256 must be 64 lowercase hexadecimal digits")
    if str(ROOT) != CANONICAL_WORKTREE:
        _fail("verification is not running from the canonical H200 worktree")

    manifest_path = _manifest_path(manifest_path)
    manifest_raw, _ = _stable_read_regular(manifest_path, "Stage-2B source manifest")
    manifest_sha256 = _sha256(manifest_raw)
    if manifest_sha256 != expected_manifest_sha256:
        _fail("Stage-2B source manifest differs from the caller trust anchor")
    manifest = _strict_json(manifest_raw, "Stage-2B source manifest")
    if set(manifest) != MANIFEST_KEYS:
        _fail("Stage-2B source manifest top-level schema differs")
    if (
        manifest["schema"] != "cach.stage2b.source_manifest.v1"
        or manifest["status"]
        != "live_lightweight_l0_l1_passed_l2_l3_blocked_non_scientific"
        or manifest["canonical_host"] != CANONICAL_HOST
        or manifest["canonical_worktree"] != CANONICAL_WORKTREE
        or type(manifest["created_at"]) is not str
        or RFC3339_UTC_RE.fullmatch(manifest["created_at"]) is None
    ):
        _fail("Stage-2B source manifest identity or timestamp differs")
    if not _exact_json_equal(manifest["scope"], EXPECTED_SCOPE):
        _fail("Stage-2B source scope differs")
    if not _exact_json_equal(manifest["base_lineage"], EXPECTED_BASE_LINEAGE):
        _fail("Stage-2B predecessor lineage pins differ")
    if not _exact_json_equal(manifest["overlay"], EXPECTED_OVERLAY):
        _fail("Stage-2B overlay declaration differs")
    if not _exact_json_equal(
        manifest["git_recovery_bundles"], EXPECTED_GIT_RECOVERY_BUNDLES
    ):
        _fail("Stage-2B git-recovery-bundle absence declaration differs")
    if manifest["retained_stage2b_evidence"] is not None:
        _fail("Stage-2B v1 must not claim retained immutable execution evidence")

    _verify_repository_lineage()
    stage0, stage1, stage2 = _verify_ancestor_chain()
    _verify_overlay(stage0, stage1, stage2)
    _verify_pinned_repo_set(
        manifest["execution_harness_pins"],
        EXPECTED_EXECUTION_HARNESS_PINS,
        "execution harness pins",
    )
    _verify_pinned_repo_set(
        manifest["inherited_source_pins"],
        EXPECTED_INHERITED_SOURCE_PINS,
        "inherited source pins",
    )
    source_bytes, source_digests = _verify_stage2b_files(manifest["files"])
    _verify_stage2b_bundle(
        manifest["stage2b_source_bundle"],
        source_bytes,
        source_digests,
    )

    return {
        "additive_artifact_valid": True,
        "caller_manifest_pin_verified": True,
        "capture_authorized": False,
        "deploy_authorized": False,
        "evaluation_authorized": False,
        "execution_replayed": False,
        "git_recovery_bundles_verified": False,
        "manifest_sha256": manifest_sha256,
        "overlay_verified": True,
        "retained_immutable_stage2b_execution_evidence": False,
        "schema": "cach.stage2b.additive_artifact_verification.v1",
        "scientific_eligible": False,
        "source_file_count": len(EXPECTED_STAGE2B_FILES),
        "stage2b_source_bundle_verified": True,
        "stage3_authorized": False,
        "training_authorized": False,
        "transitive_runtime_closure": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the byte-scoped Stage-2B additive source artifact without "
            "running or replaying tests."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="canonical Stage-2B SOURCE_MANIFEST.json path",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="mandatory out-of-band SHA256 trust anchor for the manifest",
    )
    arguments = parser.parse_args()
    try:
        result = verify(arguments.manifest, arguments.expected_manifest_sha256)
    except (
        OSError,
        Stage2BVerificationError,
        subprocess.SubprocessError,
        tarfile.TarError,
    ) as exc:
        failure = {
            "additive_artifact_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "schema": "cach.stage2b.additive_artifact_verification_failure.v1",
            "scientific_eligible": False,
            "stage3_authorized": False,
            "training_authorized": False,
            "transitive_runtime_closure": False,
        }
        print(
            json.dumps(failure, allow_nan=False, separators=(",", ":"), sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
