from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_COUNT_KEYS = (
    "actual_valid_raw_frames",
    "valid_sampled_video_frames",
    "valid_latent_frames",
    "nonbootstrap_valid_latent_frames",
    "eligible_chunk_count",
)


def _load_standalone_guard(script_name: str, function_name: str):
    path = ROOT / "scripts" / script_name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": object,
        "SUPPORT_COUNT_KEYS": SUPPORT_COUNT_KEYS,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def _valid_support() -> dict[str, int | bool]:
    return {
        "actual_valid_raw_frames": 5,
        "valid_sampled_video_frames": 2,
        "valid_latent_frames": 2,
        "nonbootstrap_valid_latent_frames": 1,
        "eligible_chunk_count": 1,
        "eligible": True,
    }


@pytest.mark.parametrize(
    ("script_name", "function_name"),
    (
        ("verify_phase6_dataset_contract.py", "require_support_types"),
        ("verify_phase6_task_plan.py", "_require_support_types"),
    ),
)
def test_independent_verifier_guards_reject_json_numeric_aliases(
    script_name: str,
    function_name: str,
) -> None:
    guard = _load_standalone_guard(script_name, function_name)
    guard(_valid_support(), "support")

    for key in SUPPORT_COUNT_KEYS:
        value = _valid_support()
        value[key] = float(value[key])
        with pytest.raises(ValueError, match="integer"):
            guard(value, "support")

    value = _valid_support()
    value["eligible_chunk_count"] = True
    with pytest.raises(ValueError, match="integer"):
        guard(value, "support")

    value = _valid_support()
    value["eligible"] = 1
    with pytest.raises(ValueError, match="boolean"):
        guard(value, "support")


@pytest.mark.parametrize(
    ("script_name", "function_name"),
    (
        (
            "verify_phase6_dataset_contract.py",
            "require_frozen_expansion_geometry",
        ),
        ("verify_phase6_task_plan.py", "_require_frozen_expansion_geometry"),
    ),
)
def test_independent_geometry_guards_reject_same_value_numeric_aliases(
    script_name: str,
    function_name: str,
) -> None:
    guard = _load_standalone_guard(script_name, function_name)
    base = {
        "_raw_window_len": 113,
        "video_stride": 4,
        "_video_sample_indices": list(range(0, 113, 4)),
        "causal_temporal": True,
        "temporal_compression": 4,
    }
    guard(SimpleNamespace(**base))
    mutations = {
        "_raw_window_len": 113.0,
        "video_stride": 4.0,
        "_video_sample_indices": [float(index) for index in range(0, 113, 4)],
        "causal_temporal": 1,
        "temporal_compression": 4.0,
    }
    for key, replacement in mutations.items():
        value = dict(base)
        value[key] = replacement
        with pytest.raises(ValueError, match="geometry"):
            guard(SimpleNamespace(**value))
