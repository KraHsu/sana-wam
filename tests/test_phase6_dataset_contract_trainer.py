from __future__ import annotations

import ast
from pathlib import Path


TRAINER = (
    Path(__file__).resolve().parents[1] / "src" / "sana_wam" / "train" / "trainer.py"
)


def _configure_source() -> str:
    source = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Trainer"
    )
    method = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_configure_phase6_plan"
    )
    return ast.get_source_segment(source, method)


def test_every_phase6_plan_requires_the_second_artifact_and_file_sha_pin():
    method = _configure_source()
    assert '"phase6_dataset_contract_artifact"' in method
    assert '"phase6_dataset_contract_artifact_sha256"' in method
    assert "if not all(configured.values())" in method
    assert "missing=" in method


def test_trainer_verifies_external_and_embedded_pins_before_wrapping_dataset():
    method = _configure_source()
    verify_offset = method.index('field_name="phase6_dataset_contract_artifact"')
    parse_offset = method.index("Phase6DatasetContract.from_artifact_bytes")
    runtime_offset = method.index("dataset_contract.validate_runtime")
    wrapper_offset = method.index("self.dataset = PlanBoundRoboTwinDataset")
    assert verify_offset < parse_offset < runtime_offset < wrapper_offset
    assert "expected_artifact_sha256=expected_dataset_contract_sha256" in method
    assert "expected_plan_sha256=expected_plan_sha256" in method
    assert "expected_identity_sha256=expected_identity_sha256" in method
    assert "expected_action_stats_sha256=expected_action_stats_sha256" in method
    assert "dataset_contract=dataset_contract" in method


def test_trainer_locates_and_hashes_all_loaded_preprocessing_sources():
    method = _configure_source()
    for source_id in (
        "sana_wam.dataloader.robotwin_dataset",
        "sana_wam.dataloader.transforms.multiview",
        "sana_wam.dataloader.transforms.normalize",
        "sana_wam.dataloader.transforms.rotation",
    ):
        assert source_id in method
    assert "inspect.getsourcefile(module)" in method
    assert "preprocessing_source_paths=preprocessing_source_paths" in method
