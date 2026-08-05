"""CPU/static contracts for the additive CACH-A4-R2 mask-shape repair.

No test imports the R2 model module or executes vendor/CUDA code.  The sole
tensor regression is a tiny CPU broadcast check for the final diagnostic.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_BRIDGE = (
    REPO_ROOT / "src/sana_wam/model/cach_av1b_a4_state_conditioned_causal_odd_stream.py"
)
R2_BRIDGE = (
    REPO_ROOT
    / "src/sana_wam/model/cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py"
)
R1_CONFIG = (
    REPO_ROOT
    / "configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r1.yaml"
)
R2_CONFIG = (
    REPO_ROOT
    / "configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r2.yaml"
)
R2_CARD = (
    REPO_ROOT
    / "docs/cach_sana_wam/architecture_validation/cach_a4_r2/CACH_A4_R2_RUN_CARD.json"
)
R2_RUNNER = (
    REPO_ROOT / "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py"
)
FIXED_BLOCK = (
    "    inactive_mask = "
    "~correct.action_present_mask.unsqueeze(-1).unsqueeze(-1).expand_as(\n"
    "        correct.action_delta\n"
    "    )"
)
ORIGINAL_LINE = (
    "    inactive_mask = ~correct.action_present_mask.expand_as(correct.action_delta)"
)
AUTHORIZED_STATE = "ONE_IDLE_SINGLE_GPU_FRESH_ROOT_SYNTHETIC_EXECUTION_AUTHORIZED"
EXPECTED_NAMESPACE = (
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r2/06f5d09127f8"
)
AUTHORITY_STATEMENT = "授权 A4-R2 单GPU合成运行"
AUTHORITY_SHA256 = "8dc3ad2e8bd1edbf8c276ff8c9979c3a923b24d026e0a9eca8dcf518311028b0"
NONCE = "f294a23c285082ae61b1685accec416e"
ROOT = f"{EXPECTED_NAMESPACE}/cach-a4-state-stream-r2-f294a23c285082ae61b1685accec416e"
GPU_INDEX = 0
GPU_UUID = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
EXACT_COMMAND = (
    "env CUDA_VISIBLE_DEVICES=GPU-1ec28cfb-f501-23f3-f865-275a744ca053 "
    "FUSED_GDN_PRECISION=0 GDN_DISABLE_COMPILE=1 PYTHONDONTWRITEBYTECODE=1 "
    "TORCHDYNAMO_DISABLE=1 "
    f"TRITON_CACHE_DIR={ROOT}/triton_cache "
    "/home/zch/workspace/sana-wam/.venv/bin/python "
    "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py "
    "--card docs/cach_sana_wam/architecture_validation/cach_a4_r2/"
    "CACH_A4_R2_RUN_CARD.json "
    "--config configs/experiments/"
    "cach_av1b_a4_state_conditioned_causal_odd_stream_r2.yaml "
    f"--root {ROOT} --nonce {NONCE} --gpu-index 0 --gpu-uuid {GPU_UUID}"
)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    assert isinstance(value, dict)
    return value


def _load_runner():
    name = "_cach_a4_r2_pending_runner_contract_test"
    spec = importlib.util.spec_from_file_location(name, R2_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_r2_bridge_is_exact_frozen_a4_copy_with_one_diagnostic_delta() -> None:
    base = BASE_BRIDGE.read_text(encoding="utf-8")
    r2 = R2_BRIDGE.read_text(encoding="utf-8")
    assert base.count(ORIGINAL_LINE) == 1
    assert FIXED_BLOCK in r2
    assert ORIGINAL_LINE not in r2
    assert r2.count(FIXED_BLOCK) == 1
    assert r2.replace(FIXED_BLOCK, ORIGINAL_LINE) == base
    ast.parse(r2)


def test_mask_shape_regression_catches_b_t_1_to_b_t_c_1_1_bug() -> None:
    action_present_mask = torch.tensor(
        [[[True], [False], [True]], [[False], [True], [False]]],
        dtype=torch.bool,
        device="cpu",
    )
    action_delta = torch.zeros((2, 3, 4, 1, 1), dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError):
        action_present_mask.expand_as(action_delta)

    expanded = action_present_mask.unsqueeze(-1).unsqueeze(-1).expand_as(action_delta)
    inactive = ~expanded
    assert expanded.shape == action_delta.shape
    assert torch.equal(expanded[:, :, 0, 0, 0], action_present_mask[:, :, 0])
    assert inactive.sum().item() == 12


def test_r2_config_is_exact_r1_config_and_keeps_launcher_guards() -> None:
    r1_bytes = R1_CONFIG.read_bytes()
    r2_bytes = R2_CONFIG.read_bytes()
    assert r2_bytes == r1_bytes
    config = _strict_json(R2_CONFIG)
    assert config["training"]["macrosteps_per_arm"] == 200
    assert config["runtime"]["launcher_environment_before_torch_or_vendor_import"] == {
        "GDN_DISABLE_COMPILE": "1",
        "TORCHDYNAMO_DISABLE": "1",
    }


def test_card_binds_exact_fresh_authority_and_execution_identity() -> None:
    card = _strict_json(R2_CARD)
    assert card["state"] == AUTHORIZED_STATE
    assert card["future_root_namespace_family"] == EXPECTED_NAMESPACE
    assert card["execution"] == {
        "source_and_cpu_authorized": True,
        "gpu_execution_authorized": True,
        "execution_requires_fresh_user_authority": False,
        "fresh_user_authority_satisfied": True,
        "one_shot_gpu_execution_max": 1,
        "automatic_rerun": False,
        "post_authority_capabilities": [
            "SOURCE_IMPLEMENTATION",
            "REDUCED_SYNTHETIC_CPU_TESTS",
            "ONE_IDLE_SINGLE_GPU_FRESH_ROOT_SYNTHETIC_RUN",
        ],
    }
    assert card["execution_binding"] == {
        "binding_state": "BOUND_BY_FRESH_USER_AUTHORITY",
        "canonical_working_directory": "/home/zch/workspace/sana-wam",
        "root": ROOT,
        "nonce": NONCE,
        "physical_gpu_index": GPU_INDEX,
        "gpu_uuid": GPU_UUID,
        "exact_env": {
            "CUDA_VISIBLE_DEVICES": GPU_UUID,
            "TRITON_CACHE_DIR": f"{ROOT}/triton_cache",
            "FUSED_GDN_PRECISION": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "GDN_DISABLE_COMPILE": "1",
            "TORCHDYNAMO_DISABLE": "1",
        },
        "exact_argv": [
            "/home/zch/workspace/sana-wam/.venv/bin/python",
            "scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py",
            "--card",
            "docs/cach_sana_wam/architecture_validation/cach_a4_r2/CACH_A4_R2_RUN_CARD.json",
            "--config",
            "configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r2.yaml",
            "--root",
            ROOT,
            "--nonce",
            NONCE,
            "--gpu-index",
            "0",
            "--gpu-uuid",
            GPU_UUID,
        ],
        "exact_command": EXACT_COMMAND,
        "stdin_contract": "CANONICAL_UTF8_SORTED_COMPACT_JSON_PLUS_LF_RUNTIME_AUTHORITY",
    }
    assert card["fresh_user_authority"]["statement"] == AUTHORITY_STATEMENT
    assert card["fresh_user_authority"]["statement_sha256"] == AUTHORITY_SHA256
    assert card["fresh_user_authority"]["exact_utf8_length_bytes"] == 31
    assert card["fresh_user_authority"]["satisfied"] is True
    assert card["current_phase_facts"]["namespace_or_root_created"] is False
    assert card["current_phase_facts"]["gpu_queried_or_used"] is True
    assert (
        card["current_phase_facts"]["cuda_or_vendor_model_imported_or_executed"]
        is False
    )


def test_runner_statically_binds_authorized_identity() -> None:
    runner = _load_runner()
    assert runner.USER_AUTHORITY_STATEMENT == AUTHORITY_STATEMENT
    assert runner.USER_AUTHORITY_SHA256 == AUTHORITY_SHA256
    assert (
        hashlib.sha256(AUTHORITY_STATEMENT.encode("utf-8")).hexdigest()
        == AUTHORITY_SHA256
    )
    assert len(AUTHORITY_STATEMENT.encode("utf-8")) == 31
    assert str(runner.EXPECTED_ROOT) == ROOT
    assert runner.EXPECTED_NONCE == NONCE
    assert runner.EXPECTED_GPU_INDEX == GPU_INDEX
    assert runner.EXPECTED_GPU_UUID == GPU_UUID
    assert runner.EXPECTED_EXACT_COMMAND == EXACT_COMMAND


def test_runner_imports_r2_only_after_precreate_and_uses_unique_namespace() -> None:
    source = R2_RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "cach_a4_state_stream_r2" in source
    assert AUTHORITY_STATEMENT in source
    assert AUTHORITY_SHA256 in source
    assert "PENDING_FRESH_USER_AUTHORITY" not in source
    assert "automatic_rerun" in source
    preflight_calls: list[int] = []
    model_imports: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            segment = ast.get_source_segment(source, node) or ""
            if "_validate_precreate" in segment:
                preflight_calls.append(node.lineno)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.endswith(
                "cach_av1b_a4_state_conditioned_causal_odd_stream_r2"
            )
        ):
            model_imports.append(node.lineno)
    assert preflight_calls and len(model_imports) == 1
    assert min(preflight_calls) < model_imports[0]
