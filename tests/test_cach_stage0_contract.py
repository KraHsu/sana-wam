"""Stage 0 fail-closed contract tests.

These tests are added for the Stage 0 review diff and are intentionally not run
until the user separately authorizes the Stage 1 validation phase.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from sana_wam.train.cach_stage0_contract import (
    CACHStage0Error,
    file_sha256,
    read_regular_bytes,
    strict_json_bytes,
    validate_authority,
    validate_candidate_spec,
    validate_source_manifest,
    validate_stage0_config,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE0 = ROOT / "docs/cach_sana_wam/stage0"


def _artifact(name: str) -> dict:
    return strict_json_bytes((STAGE0 / name).read_bytes(), name)


def test_source_manifest_is_explicitly_incomplete() -> None:
    value = validate_source_manifest(_artifact("SOURCE_MANIFEST.draft.json"))
    assert value["status"] == "draft_blocked"
    assert value["dataset"]["full_content_sha256"] is None
    assert value["dataset"]["rate_provenance_receipt"] is None
    assert value["runtime"]["package_inventory_sha256"] is None


def test_candidate_has_one_delta_and_no_execution_coordinates() -> None:
    value = validate_candidate_spec(_artifact("CACH_A_CANDIDATE_SPEC.draft.json"))
    assert value["campaign"]["reference"] == "REF-GDN-CORRECTED"
    assert value["campaign"]["candidate"] == "CACH-A"
    assert value["campaign"]["unique_delta"]
    assert all(item is None for item in value["root"].values())
    assert value["checkpoint"]["final_endpoint"] is None
    assert value["data"]["delta_action"] is False
    assert value["data"]["action_representation"].startswith("absolute_eef_target")
    assert len(value["data"]["action_order"]) == value["data"]["action_dim"] == 20
    assert value["architecture"]["proprio_boundary"]["chunk0_raw_index"] == 0


def test_draft_config_is_complete_random_and_nonexecutable() -> None:
    cfg = OmegaConf.load(ROOT / "configs/experiments/cach_sana_wam_v0.yaml")
    validate_stage0_config(
        OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    )
    assert cfg.cach_stage0.status == "draft_blocked"
    assert cfg.model.architecture.initialization_mode == "complete_random_v1"
    assert cfg.model.video_backbone.model_path is None
    assert cfg.model.video_backbone.init_dit_from is None
    assert cfg.model.architecture.proprio_per_chunk is True
    assert (
        cfg.model.architecture.proprio_boundary
        == "layout_committed_boundary_v1"
    )
    assert cfg.dataloader.delta_action is False
    assert cfg.training.init_checkpoint is None
    assert cfg.training.init_checkpoint_sha256 is None
    assert list(cfg.training.freeze) == [
        "video_backbone.vae",
        "video_backbone.text_encoder",
    ]
    assert all(
        cfg.admission[field] is False
        for field in (
            "allow_launch",
            "allow_test",
            "allow_training",
            "allow_evaluation",
            "allow_capture",
        )
    )


def test_candidate_cannot_enable_execution_by_field_flip() -> None:
    value = _artifact("CACH_A_CANDIDATE_SPEC.draft.json")
    value["execution_authorized"] = True
    with pytest.raises(CACHStage0Error, match="cannot authorize execution"):
        validate_candidate_spec(value)


def test_candidate_cannot_drop_all_blockers() -> None:
    value = _artifact("CACH_A_CANDIDATE_SPEC.draft.json")
    value["blockers"] = []
    with pytest.raises(CACHStage0Error, match="non-empty blockers"):
        validate_candidate_spec(value)


def test_authority_is_a_denial_not_a_registration() -> None:
    value = validate_authority(_artifact("CACH_A_AUTHORITY.draft.json"))
    assert value["decision"] == "deny_execution"
    assert value["registered"] is False
    assert value["execution_authorized"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("decision", "authorize"),
        ("registered", True),
        ("execution_authorized", True),
        ("scientific_eligible", True),
    ],
)
def test_authority_field_flips_fail_closed(field: str, replacement: object) -> None:
    value = deepcopy(_artifact("CACH_A_AUTHORITY.draft.json"))
    value[field] = replacement
    with pytest.raises(CACHStage0Error):
        validate_authority(value)


def test_source_manifest_cannot_drop_import_closure() -> None:
    value = deepcopy(_artifact("SOURCE_MANIFEST.draft.json"))
    value["repository_files"] = {}
    with pytest.raises(CACHStage0Error, match="source path closure"):
        validate_source_manifest(value)


def test_candidate_cannot_change_bootstrap_or_afcc_isolation() -> None:
    value = deepcopy(_artifact("CACH_A_CANDIDATE_SPEC.draft.json"))
    value["architecture"]["bootstrap"]["observed_prefix_chunks"] = 1
    with pytest.raises(CACHStage0Error, match="bootstrap"):
        validate_candidate_spec(value)

    value = deepcopy(_artifact("CACH_A_CANDIDATE_SPEC.draft.json"))
    value["afcc_isolation"]["candidate_F"] = 1
    with pytest.raises(CACHStage0Error, match="afcc"):
        validate_candidate_spec(value)


def test_nested_literal_types_are_exact() -> None:
    candidate = deepcopy(_artifact("CACH_A_CANDIDATE_SPEC.draft.json"))
    candidate["architecture"]["cache"]["commit_source_policy"][
        "commanded_action_only_commit_allowed"
    ] = 0
    with pytest.raises(CACHStage0Error, match="type differs"):
        validate_candidate_spec(candidate)

    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs/experiments/cach_sana_wam_v0.yaml"),
        resolve=True,
        enum_to_str=True,
    )
    config["model"]["architecture"]["use_proprioception"] = 1
    with pytest.raises(CACHStage0Error, match="type differs"):
        validate_stage0_config(config)


@pytest.mark.parametrize(
    "replacement",
    [
        "../outside.md",
        "/tmp/outside.md",
        "docs/./cach_sana_wam/stage0/README.md",
    ],
)
def test_authority_rejects_noncanonical_or_escaping_design_paths(
    replacement: str,
) -> None:
    value = deepcopy(_artifact("CACH_A_AUTHORITY.draft.json"))
    value["designs"]["stage0_readme"]["path"] = replacement
    with pytest.raises(CACHStage0Error, match="path"):
        validate_authority(value)


def test_json_parser_rejects_duplicates_and_nonfinite_values() -> None:
    with pytest.raises(CACHStage0Error, match="duplicate"):
        strict_json_bytes(b'{"x":1,"x":2}', "duplicate")
    with pytest.raises(CACHStage0Error, match="non-finite"):
        strict_json_bytes(b'{"x":NaN}', "nonfinite")


def test_small_file_reader_rejects_large_file_and_symlink(tmp_path: Path) -> None:
    large = tmp_path / "large.json"
    large.write_bytes(b"x" * 17)
    with pytest.raises(CACHStage0Error, match="exceeds"):
        read_regular_bytes(large, "large", max_bytes=16)

    link = tmp_path / "link.json"
    link.symlink_to(large)
    with pytest.raises(CACHStage0Error, match="cannot open"):
        read_regular_bytes(link, "symlink")


def _load_script(relative: str, module_name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_is_a_hard_denial_with_external_trust_anchor() -> None:
    module = _load_script(
        "scripts/launch_cach_sana_wam.py", "cach_stage0_denial_launcher"
    )
    authority = STAGE0 / "CACH_A_AUTHORITY.draft.json"
    with pytest.raises(ValueError, match="no execution path"):
        module.refuse_launch(
            authority,
            file_sha256(authority, "authority"),
        )


def test_launcher_rejects_wrong_external_trust_anchor() -> None:
    module = _load_script(
        "scripts/launch_cach_sana_wam.py", "cach_stage0_wrong_anchor_launcher"
    )
    authority = STAGE0 / "CACH_A_AUTHORITY.draft.json"
    with pytest.raises(ValueError, match="trust anchor differs"):
        module.refuse_launch(authority, "1" * 64)


def test_denial_launcher_has_no_execution_or_root_mutation_imports() -> None:
    tree = ast.parse(
        (ROOT / "scripts/launch_cach_sana_wam.py").read_text(encoding="utf-8")
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert "torch" not in imported
    assert "subprocess" not in imported

    forbidden_calls = {"mkdir", "makedirs", "Popen", "run", "system"}
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    assert called.isdisjoint(forbidden_calls)


def test_verifier_rejects_wrong_external_trust_anchor() -> None:
    module = _load_script(
        "scripts/verify_cach_stage0.py", "cach_stage0_wrong_anchor_verifier"
    )
    with pytest.raises(ValueError, match="trust anchor differs"):
        module.build_report(
            ROOT,
            expected_authority_sha256="1" * 64,
            verify_external=False,
            require_governance_mirrors=False,
        )


def test_quick_verifier_never_claims_static_or_execution_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(
        "scripts/verify_cach_stage0.py", "cach_stage0_denial_verifier"
    )
    original_loader = module._load_pinned_contract

    def load_without_h200_inputs(*args, **kwargs):
        contract, authority_bytes = original_loader(*args, **kwargs)
        contract.verify_repository_inputs = lambda *_a, **_k: None
        return contract, authority_bytes

    monkeypatch.setattr(module, "_load_pinned_contract", load_without_h200_inputs)
    authority = STAGE0 / "CACH_A_AUTHORITY.draft.json"
    report = module.build_report(
        ROOT,
        expected_authority_sha256=file_sha256(authority, "authority"),
        verify_external=False,
        require_governance_mirrors=False,
    )
    assert report["draft_denial_pin_graph_valid"] is False
    assert report["static_graph_valid"] is False
    assert report["inspection_complete"] is False
    assert report["execution_admission_valid"] is False
    assert report["verified_config_raw_bytes"] is True
    assert report["config_semantic_parser_executed"] is False
    assert report["verified_runtime_distribution_subset"] is True
    assert report["runtime_inventory_complete"] is False
