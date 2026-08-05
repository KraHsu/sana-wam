"""CPU/static contracts for the CACH-A4-R1 environment-only repair.

These tests never import the frozen base A4 module, Torch, Triton, or the
vendor tree.  They do not construct or execute a model and do not create a
run root.  The successful delegation case substitutes a standard-library
fake before calling the R1 wrapper.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = (
    REPO_ROOT / "src/sana_wam/model/"
    "cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py"
)
RUNNER_PATH = (
    REPO_ROOT / "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py"
)
CARD_PATH = (
    REPO_ROOT / "docs/cach_sana_wam/architecture_validation/cach_a4_r1/"
    "CACH_A4_R1_RUN_CARD.json"
)
CONFIG_PATH = (
    REPO_ROOT / "configs/experiments/"
    "cach_av1b_a4_state_conditioned_causal_odd_stream_r1.yaml"
)
BASE_CONFIG_PATH = (
    REPO_ROOT / "configs/experiments/"
    "cach_av1b_a4_state_conditioned_causal_odd_stream.yaml"
)

BASE_MODULE = "sana_wam.model.cach_av1b_a4_state_conditioned_causal_odd_stream"
ARCHITECTURE_ID = "CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1"
CONFIG_SCHEMA = "cach.cach_a4.state_conditioned_causal_odd_stream.config.v1"
SCREEN_SCHEMA = "cach.cach_a4.state_conditioned_causal_odd_stream.screen.v1"
NAMESPACE_FRAGMENT = "cach_a4_state_stream_r1/06f5d09127f8"
AUTHORITY_STATEMENT = "继续 A4"
AUTHORITY_SHA256 = "c3631ea159561fb7d13154c592e16e019384f111b798b3380c7bcf4a4e94350f"
AUTHORIZED_STATE = "ONE_IDLE_SINGLE_GPU_FRESH_ROOT_SYNTHETIC_EXECUTION_AUTHORIZED"
NONCE = "9e604d41d671ac4e705816857457e5a4"
ROOT = (
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r1/"
    "06f5d09127f8/cach-a4-state-stream-r1-9e604d41d671ac4e705816857457e5a4"
)
GPU_INDEX = 0
GPU_UUID = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
EXACT_COMMAND = (
    "env CUDA_VISIBLE_DEVICES=GPU-1ec28cfb-f501-23f3-f865-275a744ca053 "
    "FUSED_GDN_PRECISION=0 GDN_DISABLE_COMPILE=1 PYTHONDONTWRITEBYTECODE=1 "
    "TORCHDYNAMO_DISABLE=1 "
    "TRITON_CACHE_DIR=/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_state_stream_r1/06f5d09127f8/"
    "cach-a4-state-stream-r1-9e604d41d671ac4e705816857457e5a4/triton_cache "
    "/home/zch/workspace/sana-wam/.venv/bin/python "
    "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py "
    "--card docs/cach_sana_wam/architecture_validation/cach_a4_r1/"
    "CACH_A4_R1_RUN_CARD.json "
    "--config configs/experiments/"
    "cach_av1b_a4_state_conditioned_causal_odd_stream_r1.yaml "
    f"--root {ROOT} --nonce {NONCE} --gpu-index 0 --gpu-uuid {GPU_UUID}"
)
RUNTIME_GUARDS = {
    "GDN_DISABLE_COMPILE": "1",
    "TORCHDYNAMO_DISABLE": "1",
}


def _load_wrapper():
    name = "_cach_a4_r1_lazy_wrapper_contract_test"
    spec = importlib.util.spec_from_file_location(name, BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _load_runner():
    name = "_cach_a4_r1_runner_contract_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    assert isinstance(value, dict)
    return value


def _canonical_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def test_wrapper_is_stdlib_only_and_imports_base_after_guard() -> None:
    source = BRIDGE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_import_roots = {"torch", "triton", "diffusion", "sana_wam"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not forbidden_import_roots.intersection(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".", 1)[0] not in forbidden_import_roots

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    delegate = functions["run_cach_a4_screen"]
    calls = [node for node in ast.walk(delegate) if isinstance(node, ast.Call)]
    guard_lines = [
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "require_runtime_environment"
    ]
    import_lines = [
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
        and node.func.attr == "import_module"
    ]
    assert len(guard_lines) == len(import_lines) == 1
    assert guard_lines[0] < import_lines[0]


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"GDN_DISABLE_COMPILE": "1"},
        {"TORCHDYNAMO_DISABLE": "1"},
        {"GDN_DISABLE_COMPILE": "0", "TORCHDYNAMO_DISABLE": "1"},
        {"GDN_DISABLE_COMPILE": "1", "TORCHDYNAMO_DISABLE": "true"},
    ],
)
def test_missing_or_wrong_runtime_guard_fails_before_import(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, str],
) -> None:
    wrapper = _load_wrapper()
    for name in RUNTIME_GUARDS:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    imported: list[str] = []

    def forbidden_import(name: str):
        imported.append(name)
        raise AssertionError("base import occurred before environment rejection")

    monkeypatch.setattr(wrapper.importlib, "import_module", forbidden_import)
    with pytest.raises(wrapper.A4R1EnvironmentError):
        wrapper.run_cach_a4_screen()
    assert imported == []


def test_exact_runtime_guards_delegate_without_real_model_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _load_wrapper()
    for name, value in RUNTIME_GUARDS.items():
        monkeypatch.setenv(name, value)

    imports: list[str] = []
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_delegate(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return {"delegated": True}

    def fake_import(name: str) -> object:
        imports.append(name)
        return SimpleNamespace(run_cach_a4_screen=fake_delegate)

    monkeypatch.setattr(wrapper.importlib, "import_module", fake_import)
    result = wrapper.run_cach_a4_screen("payload", count=7)
    assert result == {"delegated": True}
    assert imports == [BASE_MODULE]
    assert calls == [(("payload",), {"count": 7})]


def test_r1_preserves_frozen_architecture_and_screen_identities() -> None:
    wrapper = _load_wrapper()
    assert wrapper.CACH_A4_ARCHITECTURE_ID == ARCHITECTURE_ID
    assert wrapper.A4_CONFIG_SCHEMA == CONFIG_SCHEMA
    assert wrapper.A4_SCREEN_SCHEMA == SCREEN_SCHEMA
    assert wrapper.BASE_A4_MODULE == BASE_MODULE
    assert wrapper.REQUIRED_RUNTIME_ENV == RUNTIME_GUARDS


def test_r1_card_and_config_bind_exact_fresh_authority_and_execution() -> None:
    card = _strict_json(CARD_PATH)
    config = _strict_json(CONFIG_PATH)
    base_config = _strict_json(BASE_CONFIG_PATH)
    card_text = _canonical_text(card)

    assert ARCHITECTURE_ID in card_text
    assert config.get("architecture_id") == ARCHITECTURE_ID
    assert CONFIG_SCHEMA in card_text
    assert config.get("schema") == CONFIG_SCHEMA
    assert SCREEN_SCHEMA in card_text
    assert NAMESPACE_FRAGMENT in card_text
    assert card.get("state") == AUTHORIZED_STATE

    execution = card.get("execution")
    assert isinstance(execution, dict)
    assert execution.get("source_and_cpu_authorized") is True
    assert execution.get("gpu_execution_authorized") is True
    assert execution.get("execution_requires_fresh_user_authority") is False
    assert execution.get("fresh_user_authority_satisfied") is True
    assert execution.get("one_shot_gpu_execution_max") == 1
    assert execution.get("automatic_rerun") is False
    assert execution.get("post_authority_capabilities") == [
        "SOURCE_IMPLEMENTATION",
        "REDUCED_SYNTHETIC_CPU_TESTS",
        "ONE_IDLE_SINGLE_GPU_FRESH_ROOT_SYNTHETIC_RUN",
    ]
    execution_binding = card.get("execution_binding")
    assert isinstance(execution_binding, dict)
    assert execution_binding.get("binding_state") == "BOUND_BY_FRESH_USER_AUTHORITY"
    assert execution_binding.get("canonical_working_directory") == str(REPO_ROOT)
    assert execution_binding.get("root") == ROOT
    assert execution_binding.get("nonce") == NONCE
    assert execution_binding.get("physical_gpu_index") == GPU_INDEX
    assert execution_binding.get("gpu_uuid") == GPU_UUID
    assert execution_binding.get("exact_command") == EXACT_COMMAND
    assert execution_binding.get("exact_argv") == [
        str(REPO_ROOT / ".venv/bin/python"),
        "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py",
        "--card",
        "docs/cach_sana_wam/architecture_validation/cach_a4_r1/CACH_A4_R1_RUN_CARD.json",
        "--config",
        "configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r1.yaml",
        "--root",
        ROOT,
        "--nonce",
        NONCE,
        "--gpu-index",
        "0",
        "--gpu-uuid",
        GPU_UUID,
    ]
    assert execution_binding.get("exact_env") == {
        "CUDA_VISIBLE_DEVICES": GPU_UUID,
        "TRITON_CACHE_DIR": f"{ROOT}/triton_cache",
        "FUSED_GDN_PRECISION": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        **RUNTIME_GUARDS,
    }
    fresh_authority = card.get("fresh_user_authority")
    assert isinstance(fresh_authority, dict)
    assert fresh_authority.get("statement") == AUTHORITY_STATEMENT
    assert fresh_authority.get("statement_sha256") == AUTHORITY_SHA256
    assert fresh_authority.get("exact_utf8_length_bytes") == 9
    assert fresh_authority.get("satisfied") is True
    assert fresh_authority.get("inherited_from_failed_a4_authority") is False
    phase = card.get("current_phase_facts")
    assert isinstance(phase, dict)
    assert phase.get("nonce_bound") is True
    assert phase.get("namespace_or_root_created") is False
    assert phase.get("gpu_binding_fixed") is True
    assert phase.get("gpu_queried_or_used") is True
    assert phase.get("cuda_or_vendor_model_imported_or_executed") is False
    assert phase.get("source_frozen") is True
    assert phase.get("source_freeze_committed_before_execution") is True

    runtime = config.get("runtime")
    assert isinstance(runtime, dict)
    assert runtime.get("launcher_environment_before_torch_or_vendor_import") == (
        RUNTIME_GUARDS
    )
    mechanically_reduced = json.loads(json.dumps(config))
    del mechanically_reduced["runtime"][
        "launcher_environment_before_torch_or_vendor_import"
    ]
    assert mechanically_reduced == base_config

    for name, value in RUNTIME_GUARDS.items():
        assert f'"{name}":"{value}"' in card_text


def test_r1_runner_statically_binds_authorized_identity_before_delegate() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for required in (
        "cach_a4_state_stream_r1",
        "06f5d09127f8",
        AUTHORITY_STATEMENT,
        AUTHORITY_SHA256,
        AUTHORIZED_STATE,
        NONCE,
        GPU_UUID,
        "GDN_DISABLE_COMPILE",
        "TORCHDYNAMO_DISABLE",
        "cach_av1b_a4_state_conditioned_causal_odd_stream_r1",
    ):
        assert required in source

    validation_lines: list[int] = []
    delegate_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "_validate_precreate" in segment:
            validation_lines.append(node.lineno)
        if isinstance(node.func, ast.Name) and node.func.id == "run_cach_a4_screen":
            delegate_lines.append(node.lineno)

    assert validation_lines, "runner lacks pre-root authorized binding validation"
    assert len(delegate_lines) == 1
    assert min(validation_lines) < delegate_lines[0]
    assert "PENDING_FRESH_USER_AUTHORITY" not in source
    runner = _load_runner()
    assert str(runner.EXPECTED_ROOT) == ROOT
    assert runner.EXPECTED_NONCE == NONCE
    assert runner.EXPECTED_GPU_INDEX == GPU_INDEX
    assert runner.EXPECTED_GPU_UUID == GPU_UUID
    assert runner.EXPECTED_EXACT_COMMAND == EXACT_COMMAND
