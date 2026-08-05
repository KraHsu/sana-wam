"""Torch-free Stage-1 CACH config and execution-boundary tests."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

from sana_wam.cach.authority import CACHAuthorityError
from sana_wam.cach.config import CACHConfigError, validate_stage1_config
from sana_wam.model import build_architecture
from sana_wam.train.cach_stage0_guard import reject_cach_stage0_base_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/cach_sana_wam_stage1.yaml"


def _config() -> dict:
    return OmegaConf.to_container(
        OmegaConf.load(CONFIG),
        resolve=True,
        enum_to_str=True,
    )


def test_stage1_config_closes_legacy_defaults_but_remains_nonexecutable() -> None:
    value = validate_stage1_config(_config())
    assert value["dataloader"]["type"] == "cach_episode_origin"
    assert value["dataloader"]["num_frames"] == 33
    assert value["dataloader"]["normalize_mode"] is None
    assert value["dataloader"]["filter_static_segments"] is False
    assert value["dataloader"]["allow_random_row_substitution"] is False
    assert value["dataloader"]["real_data_admission"] is False
    assert value["model"]["architecture"]["proprio_value_binding"] == (
        "semantic_float32_sha256_v1"
    )
    action_condition = value["model"]["architecture"]["action_condition"]
    assert action_condition["condition_shape"] == "fixed_b_k_a_v1"
    assert action_condition["partial_tail_mask"] == (
        "explicit_boolean_b_k_v1"
    )
    assert action_condition["padded_condition_value"] == "exact_zero"
    assert value["model"]["architecture"]["hybrid_cache"][
        "layout_binding"
    ] == "exact_chunk_action_layout_instance_v1"
    assert value["admission"]["allow_contract_tests"] is True
    assert all(
        value["admission"][field] is False
        for field in (
            "allow_model_execution",
            "allow_training",
            "allow_evaluation",
            "allow_capture",
            "allow_gpu",
            "allow_run_root_creation",
        )
    )


def test_stage1_artifact_set_is_internally_pinned_but_not_registered() -> None:
    from sana_wam.cach.verifier import verify_stage1_artifact_set

    authority = (
        ROOT
        / "docs/cach_sana_wam/stage1/CACH_STAGE1_AUTHORITY.draft.json"
    )
    authority_sha256 = hashlib.sha256(authority.read_bytes()).hexdigest()
    report = verify_stage1_artifact_set(
        ROOT,
        expected_authority_sha256=authority_sha256,
        verify_external_bundle=True,
    )
    assert report["authority_internal_pins_valid"] is True
    assert report["external_source_bundle_verified"] is True
    assert report["caller_authority_pin_verified"] is True
    assert report["external_trust_anchor_verified"] is False
    assert report["registered_execution_authority"] is False
    assert report["execution_admission_valid"] is False
    assert report["gate_s1"] == "not_claimed"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("model", "video_backbone", "model_path"), "/tmp/pretrained.pth"),
        (("model", "video_backbone", "init_dit_from"), "/tmp/model.pt"),
        (("training", "init_checkpoint"), "/tmp/checkpoint.safetensors"),
        (("dataloader", "delta_action"), True),
        (("dataloader", "episode_origin_only"), 1),
        (("dataloader", "allow_random_row_substitution"), True),
    ],
)
def test_stage1_config_rejects_semantic_drift(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    value = deepcopy(_config())
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(CACHConfigError):
        validate_stage1_config(value)


@pytest.mark.parametrize(
    "forbidden",
    [
        {"afcc_weight": 0},
        {"phase6_authority": None},
        {"action_reference": None},
        {"F": 1},
        {"F": False},
    ],
)
def test_stage1_config_rejects_afcc_and_phase6_fields(
    forbidden: dict,
) -> None:
    value = deepcopy(_config())
    value["model"]["architecture"]["forbidden_fixture"] = forbidden
    with pytest.raises(CACHConfigError, match="CACH_AFCC_ISOLATION_VIOLATION"):
        validate_stage1_config(value)


def test_reserved_stage1_marker_is_rejected_before_entrypoint_merge() -> None:
    value = _config()
    assert value["cach_stage0"] == {
        "compatibility_denial_latch": "cach_stage1_nonexecutable_v1"
    }
    with pytest.raises(RuntimeError, match="non-executable"):
        reject_cach_stage0_base_config(
            value,
            entrypoint="tests/stage1",
        )


def test_direct_cach_build_is_rejected_without_importing_model_backend() -> None:
    with pytest.raises(CACHAuthorityError, match="Stage 2 authority"):
        build_architecture({"variant": "cach_sana_wam_v0"})


def test_conflicting_top_level_variant_cannot_hide_nested_cach() -> None:
    from sana_wam.cach.authority import reject_cach_training_before_runtime

    with pytest.raises(CACHAuthorityError):
        reject_cach_training_before_runtime(
            {
                "variant": "autoregressive",
                "model": {
                    "architecture": {"variant": "cach_sana_wam_v0"}
                },
            }
        )


def test_stage1_marker_cannot_be_hidden_by_rewriting_variant() -> None:
    from sana_wam.cach.authority import (
        reject_cach_deploy_before_runtime,
        reject_cach_training_before_runtime,
    )

    disguised = {
        "cach_stage1": None,
        "model": {"architecture": {"variant": "gdn_autoregressive"}},
    }
    with pytest.raises(CACHAuthorityError):
        reject_cach_training_before_runtime(disguised)
    with pytest.raises(CACHAuthorityError):
        reject_cach_deploy_before_runtime(disguised)


def _function(tree: ast.Module, class_name: str | None, function_name: str):
    body = tree.body
    if class_name is not None:
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = owner.body
    return next(
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )


def _call_lines(node: ast.AST, call_name: str) -> list[int]:
    result = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name) and child.func.id == call_name:
            result.append(child.lineno)
        elif (
            isinstance(child.func, ast.Attribute)
            and child.func.attr == call_name
        ):
            result.append(child.lineno)
    return sorted(result)


def test_direct_trainer_guard_precedes_cuda_dataset_and_model() -> None:
    path = ROOT / "src/sana_wam/train/trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    init = _function(tree, "Trainer", "__init__")
    guard = _call_lines(init, "reject_cach_training_before_runtime")
    dataset = _call_lines(init, "build_training_dataset")
    model = _call_lines(init, "build_architecture")
    cuda = [
        node.lineno
        for node in ast.walk(init)
        if isinstance(node, ast.Attribute) and node.attr in {"is_available", "set_device"}
    ]
    assert len(guard) == 1
    assert dataset and model and cuda
    assert guard[0] < min(dataset + model + cuda)


def test_legacy_deploy_loader_rejects_cach_before_torch_and_latest_glob() -> None:
    path = ROOT / "src/sana_wam/deploy/model_loader.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    load = _function(tree, None, "load_from_checkpoint_dir")
    guard = _call_lines(load, "reject_cach_deploy_before_runtime")
    latest = _call_lines(load, "_find_latest_checkpoint")
    torch_imports = [
        node.lineno
        for node in ast.walk(load)
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
    ]
    assert len(guard) == 1
    assert latest and torch_imports
    assert guard[0] < min(latest + torch_imports)


def test_policy_server_rejects_marker_free_cach_before_checkpoint_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sana_wam.deploy import model_loader, policy_server

    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("checkpoint loading must follow the CACH guard")

    monkeypatch.setattr(model_loader, "load_from_checkpoint_dir", forbidden_loader)
    with pytest.raises(CACHAuthorityError):
        policy_server.build_server_from_config(
            OmegaConf.create(
                {
                    "model": {
                        "architecture": {"variant": "cach_sana_wam_v0"}
                    }
                }
            ),
            "/not/used",
        )


def test_policy_server_mock_rejects_dotlist_injected_cach(
    tmp_path: Path,
) -> None:
    from sana_wam.deploy import policy_server

    config = tmp_path / "legacy-deploy.yaml"
    config.write_text(
        "model:\n  architecture:\n    variant: gdn_autoregressive\n",
        encoding="utf-8",
    )
    with pytest.raises(CACHAuthorityError):
        policy_server.main(
            [
                "--mock",
                "--config",
                str(config),
                "model.architecture.variant=cach_sana_wam_v0",
            ]
        )


def test_eval_wrapper_rejects_marker_free_nested_cach(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = ROOT / "benchmarks/robotwin/eval_policy_wrapper.py"
    spec = importlib.util.spec_from_file_location(
        "cach_stage1_eval_guard",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "cach-eval.yaml"
    config.write_text(
        "model:\n  architecture:\n    variant: cach_sana_wam_v0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [str(path), "--config", str(config)])
    with pytest.raises(RuntimeError, match="not authorized"):
        module._reject_reserved_stage0_cli_config()


def test_action_output_projection_reset_occurs_after_factory() -> None:
    path = ROOT / "src/sana_wam/model/video_backbone/sana/pipeline_builder.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    build = _function(tree, None, "_build_pipe_from_spec")
    factory_lines = _call_lines(build, "factory")
    reset_lines = _call_lines(
        build,
        "_reset_and_verify_delta_pose_output_projections",
    )
    assert len(factory_lines) == 1
    assert len(reset_lines) == 1
    assert factory_lines[0] < reset_lines[0]
    helper = _function(
        tree,
        None,
        "_reset_and_verify_delta_pose_output_projections",
    )
    helper_calls = {
        child.func.attr
        for child in ast.walk(helper)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }
    assert "zero_" in helper_calls
    assert "count_nonzero" in helper_calls
