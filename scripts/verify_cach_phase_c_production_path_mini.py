#!/usr/bin/env python3
"""Read-only verifier for the Phase-C production-path mini source checkpoint.

This verifier is intentionally torch-free.  It authenticates a caller-pinned
additive source inventory and a historical CPU/synthetic test attestation.  It
does not replay tests, import the mini model, create a bundle/root, execute a
model, or authorize Stage 3, GPU, training, evaluation, or scientific use.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_HOST = "H200"
CANONICAL_WORKTREE = "/home/zch/workspace/sana-wam"
AUTHORIZED_ON = "2026-08-02"
RECORDED_AT_UTC = "2026-08-02T17:18:46Z"
RECORDED_ON = "2026-08-03"
RECORDED_TIMEZONE = "Asia/Shanghai"
MANIFEST_PATH = ROOT / (
    "docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/"
    "SOURCE_MANIFEST.json"
)
SOURCE_LIST_PATH = ROOT / (
    "docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/"
    "PHASE_C_SOURCE_FILES.txt"
)
REPORT_PATH = ROOT / (
    "docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/"
    "PHASE_C_LIGHTWEIGHT_TEST_REPORT.json"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

SOURCE_FILES = (
    "docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/README.md",
    (
        "docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/"
        "PRODUCTION_PATH_MINI_C0_C8_CLOSURE_20260802.md"
    ),
    (
        "docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/"
        "PHASE_C_LIGHTWEIGHT_TEST_REPORT.json"
    ),
    (
        "docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/"
        "PHASE_C_SOURCE_FILES.txt"
    ),
    "scripts/verify_cach_phase_c_production_path_mini.py",
    "src/sana_wam/model/action_backbone/__init__.py",
    "src/sana_wam/model/action_backbone/backbone.py",
    "src/sana_wam/model/action_backbone/components.py",
    "src/sana_wam/model/action_backbone/joint_action_dit.py",
    "src/sana_wam/model/action_backbone/scheduler.py",
    "src/sana_wam/model/cach_minimal_cpu.py",
    "src/sana_wam/model/cach_production_path_mini.py",
    "src/sana_wam/model/video_backbone/wan/__init__.py",
    "src/sana_wam/model/video_backbone/wan/shared/__init__.py",
    "src/sana_wam/model/video_backbone/wan/shared/core/__init__.py",
    "src/sana_wam/model/video_backbone/wan/shared/core/gradient/__init__.py",
    (
        "src/sana_wam/model/video_backbone/wan/shared/core/gradient/"
        "gradient_checkpoint.py"
    ),
    "tests/test_cach_production_path_mini.py",
)

HEAD_PINS = {
    "sana_afcc_handoff_head": "9586486f2a9f5172d57b325e32093a3e018d34c0",
    "sana_head": "16b9cec673e3335724ba2d8db25de7f9ed229292",
    "sana_wam_head": "605f1c134b4c983ff80f8489c4bc8847036329e2",
}
PREDECESSOR_PINS = {
    "development_plan": {
        "path": (
            "docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_"
            "DEVELOPMENT_PLAN_20260731.md"
        ),
        "sha256": "969d9668ffed07176844fcbd84062825f9d5f0af0325959b0068c38fe481f0dd",
    },
    "global_stage3_plan": {
        "path": (
            "docs/cach_sana_wam/global_stage3/"
            "GLOBAL_STAGE2_CLOSURE_AND_STAGE3_UPDATE_FREE_ADMISSION_PLAN_20260802.md"
        ),
        "sha256": "93c015982b421161c137d40dee825de1f3ac5996b051a46591d9487b0eacecd8",
    },
    "stage2b_l3_manifest": {
        "path": "docs/cach_sana_wam/stage2b_l3/SOURCE_MANIFEST.json",
        "sha256": "3c494faab4428ec8b7ba705b61c8bdd7bfe6f9f623366610ca1e9e94179f7b93",
    },
    "stage2b_l3_verifier": {
        "path": "scripts/verify_cach_stage2b_l3.py",
        "sha256": "b058c46b01c1041677dd84a14bfc974593909daad5023b5b8bb4e432b643812f",
    },
    "stage2b_l3_bundle": {
        "mode": "0444",
        "path": (
            "/DATA/share/sana_cach_source_bundles/"
            "cach_stage2b_l3_"
            "1a2f984b179d6f0a719873e4eda7344b99b8093c75f4ad8528f63a373ca784c3_"
            "20260731.tar"
        ),
        "sha256": "1a2f984b179d6f0a719873e4eda7344b99b8093c75f4ad8528f63a373ca784c3",
        "size_bytes": 184320,
    },
    "minimal_cpu_source": {
        "path": "src/sana_wam/model/cach_minimal_cpu.py",
        "sha256": "2b91b3ce76a560034ddeb03cf634de1acb72807d1a5d37b2d14c14acbaabefa8",
    },
    "minimal_cpu_test": {
        "path": "tests/test_cach_minimal_cpu.py",
        "sha256": "724fc8447811279098dc5beb65501095170f8e4d873a9a8a3df0da6133c2d3c3",
    },
    "prior_cuda_smoke_source_not_executed": {
        "path": "src/sana_wam/model/cach_minimal_cuda_smoke.py",
        "sha256": "5c392cee053b7f0b05b881ad3ef998e24217e761f6ccd5ddc957bf2e72023a4e",
    },
}
PREDECESSOR_RESOLVED_COUNT = 126
PREDECESSOR_RESOLVED_INVENTORY_SHA256 = (
    "6f8714ff6ff5cf35aa9ef11267c742e77a9b9317a8a8a8f1edba9331db890a2a"
)

RUNTIME_DEPENDENCY_PINS = {
    ".python-version": "7b55f8e67b5623c4bef3fa691288da9437d79d3aba156de48d481db32ac7d16d",
    "pyproject.toml": "c84bce95525fcbe03800602bd6dc3e6a151f53888b0dbc91c405ac9bf6f6bafa",
    "src/sana_wam/cach/action_conditioning.py": (
        "8323efa1544d5c0520e13192b55ed5f2e85490d03ac21c558d6009db38ee11c6"
    ),
    "src/sana_wam/cach/committed_action_history.py": (
        "162d7e7ee866b18cdba0f0e319b2c9b277d73d933ff5a4dd38c92ebcc762b5e9"
    ),
    "src/sana_wam/cach/prefix_compaction.py": (
        "1a2f12f38bb9699c65af7e4ea72183f9d122b6f81aae4d9628129040a3cfcdb7"
    ),
    "src/sana_wam/cach/stage2_failure_evidence.py": (
        "eef771a2ca0d1b423cdcced62ff7acf627bba8a852233cc3ffd9c4ce7658a03b"
    ),
    "src/sana_wam/cach/staging_variant.py": (
        "99da23af8c5cd62fa2a7e468b6b372a298900608be22a18c4634b82ef0dea220"
    ),
    "src/sana_wam/model/__init__.py": (
        "78e9e9403c02e3af004d18b2a8026647fc0e5e873a8684b4f96251b9cc530f0d"
    ),
    "src/sana_wam/model/action_backbone/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "src/sana_wam/model/action_backbone/backbone.py": (
        "eedf5a6415b9083977c1c4cd5c87be56b65045cb997e386901ad4910fd72c934"
    ),
    "src/sana_wam/model/action_backbone/components.py": (
        "abd48045622daad2072acba3e1a298206a056043acd4bfec3aee210399ff5ba2"
    ),
    "src/sana_wam/model/action_backbone/joint_action_dit.py": (
        "e0f2fcde9914688e467a209346f9796af87335220754bca763065fa1d8960290"
    ),
    "src/sana_wam/model/action_backbone/scheduler.py": (
        "b7243eb00006105650af960a121b343670633d131f90c835aff70b7c279c20a4"
    ),
    "src/sana_wam/model/action_chunk_layout.py": (
        "8e6521a93614b4ce7e43ba51d2af4f67d2650b03cc60e7d7a152e36dfa3c8e39"
    ),
    "src/sana_wam/model/cach_minimal_cpu.py": (
        "2b91b3ce76a560034ddeb03cf634de1acb72807d1a5d37b2d14c14acbaabefa8"
    ),
    "src/sana_wam/model/cach_numerical_core.py": (
        "5259958d547a5e3fee7a64443551726d8c97974fbc9894afa37646eb73397dc9"
    ),
    "src/sana_wam/model/cach_paired_stager.py": (
        "c1455af51df4e0a0eff9971fa98e9272aac580084029a7990161817d246a6ef4"
    ),
    "src/sana_wam/model/video_backbone/sana/adapter.py": (
        "fdda391181e6f30455cf54fe1559b4bd2ee33bc9274b42712f0f5e826aef58d5"
    ),
    "src/sana_wam/model/video_backbone/sana/hybrid_cache.py": (
        "5b61ab6492670e8021b3603f965e9f3e8921e9d7f0ea77b49e40d92a77f78a74"
    ),
    "src/sana_wam/model/video_backbone/sana/hybrid_cache_codec.py": (
        "563454779c5e45ceee4dbae9a99f2636114cc7645a5492fab2b487e05f59e638"
    ),
    "src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b.py": (
        "8ab968b77660df1092b43a52eecb4dba16316060e7f83c0ad6fd214230a8a5dd"
    ),
    "src/sana_wam/model/video_backbone/sana/pipeline_builder.py": (
        "2a7e4f8688b1bec04a4290b6a5deb1a51064d9477ac34c9de3135d618acaba20"
    ),
    "src/sana_wam/model/video_backbone/wan/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "src/sana_wam/model/video_backbone/wan/shared/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "src/sana_wam/model/video_backbone/wan/shared/core/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "src/sana_wam/model/video_backbone/wan/shared/core/gradient/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    (
        "src/sana_wam/model/video_backbone/wan/shared/core/gradient/"
        "gradient_checkpoint.py"
    ): "0bc3d9d50c3fd9479f1c4556dcdbfa169115b97be32eba2ece855dc3a9f57930",
    "tests/conftest.py": (
        "893ce1a02854323e0ebdd3574a74364c6f15aa489e8f8fd7187126efd81d39fc"
    ),
    "uv.lock": "31803c58048ff07600116ac389bd025f4def00cea0282a29637017b4acfec08c",
}

EXPECTED_NODE_IDS = (
    "test_phase_c_c4_builder_rejects_drift_before_model_allocation",
    "test_phase_c_c1_c4_c8_graph_inventory_and_named_init",
    "test_phase_c_c1_exact_reference_bypass_and_zero_candidate_identity",
    "test_phase_c_c0_future_and_target_separation_reverse_grad_and_jvp",
    "test_phase_c_c1_zero_adapter_reverse_grad_and_parameter_jvp",
    "test_phase_c_c2_c7_c8_full_vs_chunk_and_atomic_growth",
    "test_phase_c_c3_one_layout_binding_and_pre_operator_rejections",
    "test_phase_c_c4_c6_bootstrap_continuation_tail_exact_action_coverage",
    "test_phase_c_c7_failure_duplicate_reset_and_stale_view",
    "test_phase_c_failure_receipt_is_exclusive_immutable_tmp_only",
    "test_phase_c_cpu_only_and_no_prohibited_runtime_side_effects",
)


class PhaseCVerificationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise PhaseCVerificationError(message)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
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


def _strict_json(payload: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not strict UTF-8 JSON: {exc}")
    return value


def _canonical_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{label} is not a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        _fail(f"{label} is not a canonical relative path")
    return value


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_read_regular(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        _fail(f"{label} must be a single-link regular file")
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _fingerprint(before) != _fingerprint(opened):
            _fail(f"{label} changed before open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    entry_after = os.stat(path, follow_symlinks=False)
    if (
        _fingerprint(opened) != _fingerprint(after)
        or _fingerprint(after) != _fingerprint(entry_after)
    ):
        _fail(f"{label} changed during stable read")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        _fail(f"{label} had a short stable read")
    return payload, after


def _read_repo_file(path: str, label: str) -> bytes:
    relative = _canonical_relative_path(path, label)
    current = ROOT
    for part in PurePosixPath(relative).parts:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            _fail(f"{label} contains a symlink path component")
    return _stable_read_regular(ROOT / relative, label)[0]


def _verify_pin(pin: dict[str, object], label: str) -> None:
    if set(pin) != {"path", "sha256"}:
        _fail(f"{label} pin schema differs")
    path = _canonical_relative_path(pin["path"], f"{label} path")
    digest = pin["sha256"]
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        _fail(f"{label} digest is invalid")
    if _sha256(_read_repo_file(path, label)) != digest:
        _fail(f"{label} bytes differ")


def _pin_entries(values: dict[str, str]) -> list[dict[str, object]]:
    return [
        {"path": path, "sha256": values[path]} for path in sorted(values)
    ]


def _verify_heads() -> None:
    commands = {
        "sana_wam_head": ["git", "rev-parse", "HEAD"],
        "sana_head": ["git", "-C", "third_party/Sana", "rev-parse", "HEAD"],
        "sana_afcc_handoff_head": [
            "git",
            "-C",
            "../sana-afcc-handoff",
            "rev-parse",
            "HEAD",
        ],
    }
    for name, command in commands.items():
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() != HEAD_PINS[name]:
            _fail(f"{name} differs from the fixed lineage")


def _verify_l3_predecessor() -> dict[str, str]:
    verifier_pin = PREDECESSOR_PINS["stage2b_l3_verifier"]
    manifest_pin = PREDECESSOR_PINS["stage2b_l3_manifest"]
    _verify_pin(verifier_pin, "Stage2B L3 verifier")
    _verify_pin(manifest_pin, "Stage2B L3 manifest")
    verifier_source = _read_repo_file(
        str(verifier_pin["path"]), "Stage2B L3 verifier execution bytes"
    )
    if _sha256(verifier_source) != verifier_pin["sha256"]:
        _fail("Stage2B L3 verifier execution bytes differ")
    bundle_pin = PREDECESSOR_PINS["stage2b_l3_bundle"]
    bundle_path = Path(str(bundle_pin["path"]))
    bundle, metadata = _stable_read_regular(bundle_path, "Stage2B L3 bundle")
    if (
        len(bundle) != bundle_pin["size_bytes"]
        or _sha256(bundle) != bundle_pin["sha256"]
        or stat.S_IMODE(metadata.st_mode) != 0o444
    ):
        _fail("Stage2B L3 bundle bytes or mode differ")

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    execution_wrapper = r"""
import sys

path = sys.argv[1]
sys.argv = [path, *sys.argv[2:]]
source = sys.stdin.buffer.read()
namespace = {
    "__builtins__": __builtins__,
    "__file__": path,
    "__name__": "__main__",
    "__package__": None,
}
exec(compile(source, path, "exec"), namespace)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            execution_wrapper,
            str(ROOT / str(verifier_pin["path"])),
            "--manifest",
            str(ROOT / str(manifest_pin["path"])),
            "--expected-manifest-sha256",
            str(manifest_pin["sha256"]),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        input=verifier_source,
    )
    predecessor_result = _strict_json(result.stdout, "Stage2B L3 verifier output")
    if not isinstance(predecessor_result, dict) or any(
        predecessor_result.get(key) is not expected
        for key, expected in {
            "additive_artifact_valid": True,
            "caller_manifest_pin_verified": True,
            "execution_replayed": False,
            "l3_source_checkpoint_verified": True,
            "model_executed": False,
            "scientific_eligible": False,
            "stage3_authorized": False,
            "training_authorized": False,
            "transitive_runtime_closure": False,
        }.items()
    ):
        _fail("Stage2B L3 predecessor verifier result differs")

    # Execute the already byte-pinned predecessor inventory logic in an
    # isolated child.  The Phase-C verifier process itself never imports or
    # executes predecessor modules through runpy.
    inventory_wrapper = r"""
import json
import sys

path = sys.argv[1]
source = sys.stdin.buffer.read()
namespace = {
    "__builtins__": __builtins__,
    "__file__": path,
    "__name__": "__phase_c_pinned_l3__",
    "__package__": None,
}
exec(compile(source, path, "exec"), namespace)
l2b_manifest = namespace["_verify_l2b_predecessor"]()
inventory, _ = namespace["_resolved_predecessor_inventory"](l2b_manifest)
payloads, _ = namespace["_source_bytes"]()
inventory.update(
    {path: namespace["_sha256"](payload) for path, payload in payloads.items()}
)
print(json.dumps(
    {
        "digest": namespace["_resolved_inventory_sha256"](inventory),
        "inventory": inventory,
    },
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
))
"""
    inventory_result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            inventory_wrapper,
            str(ROOT / str(verifier_pin["path"])),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        input=verifier_source,
    )
    resolved = _strict_json(
        inventory_result.stdout, "isolated Stage2B L3 inventory output"
    )
    if not isinstance(resolved, dict) or set(resolved) != {"digest", "inventory"}:
        _fail("isolated Stage2B L3 inventory schema differs")
    inventory = resolved["inventory"]
    digest = resolved["digest"]
    if (
        not isinstance(inventory, dict)
        or len(inventory) != PREDECESSOR_RESOLVED_COUNT
        or digest != PREDECESSOR_RESOLVED_INVENTORY_SHA256
    ):
        _fail("resolved 126-path Stage2B L3 predecessor inventory differs")
    for path, source_digest in inventory.items():
        _canonical_relative_path(path, "resolved predecessor path")
        if not isinstance(source_digest, str) or SHA256_RE.fullmatch(
            source_digest
        ) is None:
            _fail("resolved predecessor inventory contains an invalid digest")
    return inventory


def _parse_python(payload: bytes, label: str) -> ast.Module:
    try:
        source = payload.decode("utf-8")
        parsed = ast.parse(source, filename=label)
    except (UnicodeDecodeError, SyntaxError) as exc:
        _fail(f"{label} is not valid UTF-8 Python: {exc}")
    return parsed


def _verify_test_ast(payload: bytes) -> None:
    tree = _parse_python(payload, "Phase-C focused test")
    observed = tuple(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )
    if observed != EXPECTED_NODE_IDS:
        _fail("Phase-C focused test function inventory/order differs")
    forbidden_marks = {"parametrize", "skip", "skipif", "xfail", "importorskip"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            terminal = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else None
            )
            if terminal in forbidden_marks:
                _fail(f"Phase-C focused tests contain forbidden {terminal} call")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                rendered = ast.dump(decorator, include_attributes=False).casefold()
                if any(mark in rendered for mark in forbidden_marks):
                    _fail("Phase-C focused tests contain a conditional test marker")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in targets
            ):
                _fail("Phase-C focused tests contain module-level pytestmark")


def _verify_harness_ast(payload: bytes) -> None:
    tree = _parse_python(payload, "Phase-C mini harness")
    forbidden_segments = {
        "afcc",
        "checkpoint",
        "dataloader",
        "dataset",
        "optimizer",
        "trainer",
        "training",
    }
    imports: list[str] = []
    classes: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node
    for module in imports:
        segments = {part.casefold() for part in module.split(".")}
        if segments & forbidden_segments:
            _fail(f"Phase-C harness imports prohibited module {module}")
    required_methods = {
        "ProductionPathMiniVideoBackbone": {
            "_apply",
            "from_mini_config",
            "from_pretrained",
            "load_state_dict",
            "run_chunk",
        },
        "ProductionPathMiniArm": {
            "_apply",
            "load_state_dict",
            "requires_grad_",
            "train",
        },
        "ProductionPathMiniEphemeralStateOwner": {
            "_validate_next_content_time",
            "commit_paired",
            "finish_denoise",
            "reset",
            "snapshot_for_denoise",
        },
        "ProductionPathMiniDispatcher": {
            "commit_paired",
            "reset_episode_for_tests",
            "run_full_sequence",
            "run_readonly_chunk",
        },
    }
    for class_name, expected in required_methods.items():
        definition = classes.get(class_name)
        if definition is None:
            _fail(f"Phase-C harness lacks {class_name}")
        methods = {
            node.name
            for node in definition.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if not expected <= methods:
            _fail(f"Phase-C harness {class_name} sealed interface differs")


def _verify_static_sources(payloads: dict[str, bytes]) -> None:
    _verify_test_ast(payloads["tests/test_cach_production_path_mini.py"])
    _verify_harness_ast(
        payloads["src/sana_wam/model/cach_production_path_mini.py"]
    )


def _verify_report(payload: bytes) -> None:
    report = _strict_json(payload, "Phase-C lightweight report")
    required_keys = {
        "authority",
        "authorized_on",
        "canonical_host",
        "canonical_worktree",
        "contracts",
        "environment",
        "gate_state",
        "hard_suite",
        "implementation",
        "limitations",
        "recorded_at_utc",
        "recorded_on",
        "recorded_timezone",
        "regressions",
        "schema",
        "status",
    }
    if not isinstance(report, dict) or set(report) != required_keys:
        _fail("Phase-C lightweight report top-level schema differs")
    if (
        report["schema"]
        != "cach.global_stage3.phase_c.production_path_mini_report.v1"
        or report["status"]
        != (
            "phase_c_scoped_interface_proxy_suite_passed_"
            "section11_blocked_not_global_admission"
        )
        or report["canonical_host"] != CANONICAL_HOST
        or report["canonical_worktree"] != CANONICAL_WORKTREE
        or report["authorized_on"] != AUTHORIZED_ON
        or report["recorded_at_utc"] != RECORDED_AT_UTC
        or report["recorded_on"] != RECORDED_ON
        or report["recorded_timezone"] != RECORDED_TIMEZONE
    ):
        _fail("Phase-C lightweight report identity differs")
    authority = report["authority"]
    if not isinstance(authority, dict) or authority.get("allowed") != [
        "update_development_documentation_and_source_manifest",
        "implement_production_shaped_mini_harness",
        "run_cpu_synthetic_lightweight_tests",
    ]:
        _fail("Phase-C report allowed authority differs")
    authorization = authority.get("authorization")
    if (
        set(authority) != {"allowed", "authorization"}
        or not isinstance(authorization, dict)
        or set(authorization)
        != {
            "capture",
            "checkpoint",
            "complete_2b",
            "deployment",
            "evaluation",
            "formal_admission_root",
            "gpu",
            "optimizer",
            "production_builder_enablement",
            "real_data",
            "scientific_claim",
            "stage3",
            "training",
        }
        or set(authorization.values()) != {False}
    ):
        _fail("Phase-C report non-authorized capabilities must remain all false")
    environment = report["environment"]
    if (
        not isinstance(environment, dict)
        or environment.get("CUDA_VISIBLE_DEVICES") != ""
        or environment.get("cuda_initialized") is not False
        or environment.get("gpu_used") is not False
        or environment.get("device") != "cpu"
        or environment.get("dtype") != "float32"
        or environment.get("python") != "3.12.13"
        or environment.get("torch") != "2.7.1+cu128"
    ):
        _fail("Phase-C CPU-only environment attestation differs")
    suite = report["hard_suite"]
    expected_counts = {
        "collected": 11,
        "deselected": 0,
        "failed": 0,
        "passed": 11,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    if not isinstance(suite, dict) or any(
        suite.get(name) != value for name, value in expected_counts.items()
    ):
        _fail("Phase-C hard-suite counts differ")
    if suite.get("node_ids") != list(EXPECTED_NODE_IDS):
        _fail("Phase-C hard-suite node inventory differs")
    if suite.get("pytest_cache_disabled") is not True:
        _fail("Phase-C hard suite did not attest disabled pytest cache")
    if suite.get("readonly_repetition_verified") is not True:
        _fail("Phase-C hard suite did not attest repeated readonly execution")
    duration = suite.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration < 0
    ):
        _fail("Phase-C hard-suite duration is invalid")
    contracts = report["contracts"]
    if not isinstance(contracts, dict) or set(contracts) != {
        f"C{index}" for index in range(9)
    }:
        _fail("Phase-C C0--C8 report matrix differs")
    for name, result in contracts.items():
        expected_status = (
            "phase_c_source_delta_verified_global_contract_open"
            if name == "C5"
            else "phase_c_interface_proxy_evidence_passed_global_contract_open"
        )
        if (
            not isinstance(result, dict)
            or result.get("status") != expected_status
            or result.get("global_contract_closed") is not False
        ):
            _fail(f"Phase-C {name} status differs")
    if (
        contracts["C5"].get("status")
        != "phase_c_source_delta_verified_global_contract_open"
        or contracts["C5"].get("global_c5_closed") is not False
        or contracts["C5"].get("transitive_runtime_closure") is not False
    ):
        _fail("Phase-C report overclaims C5 runtime closure")
    gates = report["gate_state"]
    if gates != {
        "global_gate_s0": "NOT_CLAIMED",
        "global_gate_s1": "NOT_CLAIMED",
        "global_gate_s2": "BLOCKED_NOT_CLAIMED",
        "global_stage3": "NOT_AUTHORIZED",
    }:
        _fail("Phase-C report changes a global gate")
    implementation = report["implementation"]
    if (
        not isinstance(implementation, dict)
        or implementation.get("complete_2b") is not False
        or implementation.get("depth") != 20
        or implementation.get("action_dim") != 20
        or implementation.get("vendor_gdn_numerical_parity") is not False
        or implementation.get("durable_publication") is not False
        or implementation.get("exact_future_g3_dispatcher_reused") is not False
        or implementation.get("graph_evidence_scope")
        != "registered_inventory_not_transitive_reachability"
        or implementation.get("pair_digest")
        != "30edd874743599bb7e810d942d58c5b162428f5c4611a38ad03a647801e704fe"
        or implementation.get("layer_registry_digest")
        != "3995f5120ebc499f44aae7d651ce30e387f29a5e91024e7912b644f6739bd3ed"
        or implementation.get("candidate_only_tensor_count") != 44
        or implementation.get("zero_adapter_tensor_count") != 40
        or implementation.get("clean_noisy_inputs_distinct") is not True
        or implementation.get("readonly_timesteps_nonzero") is not True
        or implementation.get("synthetic_dataset_ground_truth_schema_shim")
        is not True
    ):
        _fail("Phase-C implementation scope differs")
    limitations = report["limitations"]
    if limitations != {
        "complete_2b": False,
        "dynamic_import_closure": False,
        "exact_future_g3_dispatcher_reused": False,
        "formal_admission_root": False,
        "global_c0_c8_closed": False,
        "native_library_closure": False,
        "production_timebase_verified": False,
        "real_data_source_proof": False,
        "retained_execution_evidence": False,
        "synthetic_dataset_ground_truth_schema_shim": True,
        "transitive_runtime_closure": False,
        "vendor_gdn_numerical_parity": False,
    }:
        _fail("Phase-C report limitations differ")
    regressions = report["regressions"]
    if (
        not isinstance(regressions, dict)
        or regressions.get("collected") != 48
        or regressions.get("passed") != 48
        or regressions.get("failed") != 0
        or regressions.get("skipped") != 0
        or regressions.get("xfailed") != 0
        or regressions.get("gpu_used") is not False
        or regressions.get("files")
        != [
            "tests/test_cach_minimal_cpu.py",
            "tests/test_cach_stage2b_production_core.py",
            "tests/test_cach_action_chunk_layout.py",
            "tests/test_cach_stage2_failure_evidence.py",
        ]
    ):
        _fail("Phase-C regression report differs")


def _verify_manifest_files(
    entries: object,
) -> tuple[dict[str, bytes], list[dict[str, object]]]:
    if not isinstance(entries, list) or len(entries) != len(SOURCE_FILES):
        _fail("Phase-C source file inventory length differs")
    payloads: dict[str, bytes] = {}
    expected_entries: list[dict[str, object]] = []
    for path in SOURCE_FILES:
        payload = _read_repo_file(path, f"Phase-C source {path}")
        payloads[path] = payload
        expected_entries.append(
            {"path": path, "sha256": _sha256(payload), "size_bytes": len(payload)}
        )
    if entries != expected_entries:
        _fail("Phase-C source file hashes/sizes differ")
    return payloads, expected_entries


def verify(manifest_path: Path, expected_manifest_sha256: str) -> dict[str, object]:
    if str(ROOT) != CANONICAL_WORKTREE:
        _fail("Phase-C verification must run in the canonical H200 worktree")
    if Path(os.path.abspath(os.fspath(manifest_path))) != MANIFEST_PATH:
        _fail("Phase-C manifest must use the canonical default path")
    if SHA256_RE.fullmatch(expected_manifest_sha256) is None:
        _fail("expected manifest SHA256 must be lowercase hexadecimal")
    raw_manifest, _ = _stable_read_regular(manifest_path, "Phase-C manifest")
    manifest_sha256 = _sha256(raw_manifest)
    if manifest_sha256 != expected_manifest_sha256:
        _fail("Phase-C manifest differs from the caller trust anchor")
    manifest = _strict_json(raw_manifest, "Phase-C manifest")
    required_keys = {
        "authorized_on",
        "base_lineage",
        "canonical_host",
        "canonical_worktree",
        "created_at",
        "files",
        "gate_state",
        "overlay",
        "retained_execution_evidence",
        "runtime_dependency_pins",
        "schema",
        "scope",
        "source_bundle",
        "status",
    }
    if not isinstance(manifest, dict) or set(manifest) != required_keys:
        _fail("Phase-C manifest top-level schema differs")
    if (
        manifest["schema"] != "cach.global_stage3.phase_c.source_manifest.v1"
        or manifest["status"]
        != (
            "phase_c_worktree_source_delta_verified_not_runtime_closed_"
            "not_admitted_non_scientific"
        )
        or manifest["canonical_host"] != CANONICAL_HOST
        or manifest["canonical_worktree"] != CANONICAL_WORKTREE
        or manifest["authorized_on"] != AUTHORIZED_ON
        or manifest["created_at"] != RECORDED_AT_UTC
    ):
        _fail("Phase-C manifest identity differs")
    expected_lineage = {**HEAD_PINS, **PREDECESSOR_PINS}
    if manifest["base_lineage"] != expected_lineage:
        _fail("Phase-C predecessor lineage differs")
    _verify_heads()
    for name, pin in PREDECESSOR_PINS.items():
        if name == "stage2b_l3_bundle":
            continue
        _verify_pin(pin, name)
    predecessor_inventory = _verify_l3_predecessor()
    payloads, _ = _verify_manifest_files(manifest["files"])
    if any(path in predecessor_inventory for path in SOURCE_FILES):
        _fail("Phase-C additive source paths collide with the 126-path predecessor")
    expected_overlay = {
        "order": ["stage2b_l3", "global_stage3_phase_c_production_path_mini"],
        "phase_c_predecessor_collisions": [],
        "predecessor_resolved_file_count": PREDECESSOR_RESOLVED_COUNT,
        "predecessor_resolved_inventory_sha256": (
            PREDECESSOR_RESOLVED_INVENTORY_SHA256
        ),
        "resolved_file_count": PREDECESSOR_RESOLVED_COUNT + len(SOURCE_FILES),
        "zero_collision_required": True,
    }
    if manifest["overlay"] != expected_overlay:
        _fail("Phase-C zero-collision overlay declaration differs")
    expected_scope = {
        "complete_2b": False,
        "dynamic_import_closure": False,
        "exact_future_g3_dispatcher_reused": False,
        "execution_scope": "cpu_synthetic_production_shaped_interface_proxy_c0_c8",
        "file_count": len(SOURCE_FILES),
        "formal_admission_root_created": False,
        "kind": "phase_c_additive_delta_over_verified_stage2b_l3",
        "native_library_closure": False,
        "source_manifest_excluded_to_avoid_self_reference": True,
        "transitive_runtime_closure": False,
        "vendor_gdn_numerical_parity": False,
    }
    if manifest["scope"] != expected_scope:
        _fail("Phase-C scope differs")
    if manifest["retained_execution_evidence"] is not None:
        _fail("Phase-C must not claim retained raw execution evidence")
    if manifest["source_bundle"] is not None:
        _fail("Phase-C authorization did not permit a new source bundle")
    expected_gate_state = {
        "global_gate_s0": "NOT_CLAIMED",
        "global_gate_s1": "NOT_CLAIMED",
        "global_gate_s2": "BLOCKED_NOT_CLAIMED",
        "global_stage3": "NOT_AUTHORIZED",
    }
    if manifest["gate_state"] != expected_gate_state:
        _fail("Phase-C manifest changes a global gate")
    expected_runtime_pins = _pin_entries(RUNTIME_DEPENDENCY_PINS)
    if manifest["runtime_dependency_pins"] != expected_runtime_pins:
        _fail("Phase-C runtime dependency pins differ")
    for entry in expected_runtime_pins:
        _verify_pin(entry, f"runtime dependency {entry['path']}")
    expected_source_list = "".join(f"{path}\n" for path in SOURCE_FILES).encode()
    if payloads[str(SOURCE_LIST_PATH.relative_to(ROOT))] != expected_source_list:
        _fail("Phase-C source-list bytes differ")
    _verify_static_sources(payloads)
    _verify_report(payloads[str(REPORT_PATH.relative_to(ROOT))])
    if "torch" in sys.modules:
        _fail("Phase-C source verifier imported torch")
    return {
        "caller_manifest_pin_verified": True,
        "capture_authorized": False,
        "checkpoint_or_data_access_authorized": False,
        "deploy_authorized": False,
        "evaluation_authorized": False,
        "execution_replayed": False,
        "external_trust_anchor_verified": False,
        "formal_admission_root_authorized": False,
        "global_c0_c8_closed": False,
        "global_gate_transition_authorized": False,
        "historical_execution_attestation_only": True,
        "manifest_sha256": manifest_sha256,
        "model_executed": False,
        "phase_c_scoped_artifact_set_valid": True,
        "predecessor_resolved_file_count": PREDECESSOR_RESOLVED_COUNT,
        "predecessor_verified": True,
        "runtime_closure_verified": False,
        "schema": "cach.global_stage3.phase_c.artifact_verification.v1",
        "scientific_eligible": False,
        "source_bundle_verified": False,
        "source_file_count": len(SOURCE_FILES),
        "stage3_authorized": False,
        "torch_imported": False,
        "training_authorized": False,
        "transitive_runtime_closure": False,
        "zero_collision_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the caller-pinned Phase-C production-path mini sources."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    arguments = parser.parse_args()
    try:
        result = verify(arguments.manifest, arguments.expected_manifest_sha256)
    except (
        OSError,
        PhaseCVerificationError,
        subprocess.SubprocessError,
    ) as exc:
        failure = {
            "caller_manifest_pin_verified": False,
            "capture_authorized": False,
            "deploy_authorized": False,
            "error": f"{type(exc).__name__}: {exc}",
            "evaluation_authorized": False,
            "execution_replayed": False,
            "formal_admission_root_authorized": False,
            "global_gate_transition_authorized": False,
            "historical_execution_attestation_only": False,
            "model_executed": False,
            "phase_c_scoped_artifact_set_valid": False,
            "runtime_closure_verified": False,
            "schema": "cach.global_stage3.phase_c.artifact_verification_failure.v1",
            "scientific_eligible": False,
            "source_bundle_verified": False,
            "stage3_authorized": False,
            "training_authorized": False,
            "transitive_runtime_closure": False,
        }
        print(_canonical_json_bytes(failure).decode("utf-8"), file=sys.stderr)
        return 1
    print(_canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
