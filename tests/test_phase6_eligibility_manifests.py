from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_task_plan_builder_reports_clean_pool_rebuild_and_support_provenance() -> None:
    source = _source("build_phase6_task_plan.py")
    assert 'dataset.phase6_candidate_pools(contract, dataset_type="robotwin")' in source
    for field in (
        "raw_candidate_pool_counts",
        "raw_candidate_total",
        "eligible_candidate_pool_counts",
        "eligible_candidate_total",
        "excluded_candidate_pool_counts",
        "excluded_candidate_total",
        "selected_minimum_support",
        "expansion_support_contract_sha256",
        "expansion_eligibility_amendment_sha256",
    ):
        assert f'"{field}"' in source
    assert '"sana-phase6-task-plan-build-manifest-v2"' in source


def test_task_plan_verifier_independently_filters_before_rebuilding() -> None:
    source = _source("verify_phase6_task_plan.py")
    assert "def _independent_support(" in source
    assert "def _independent_candidate_pools(" in source
    assert "rebuilt = TaskRoundRobinPlan.build(contract, pools)" in source
    assert "clean eligible-pool rebuild differs" in source
    assert '"sana-phase6-task-plan-verification-v2"' in source


def test_dataset_builder_and_verifier_publish_v2_support_summaries() -> None:
    builder = _source("build_phase6_dataset_contract.py")
    verifier = _source("verify_phase6_dataset_contract.py")
    assert '"sana-phase6-dataset-contract-build-manifest-v2"' in builder
    assert '"selected_minimum_support"' in builder
    assert '"expansion_support_contract"' in builder
    for field in (
        "padding_semantics_amendment_sha256",
        "padding_semantics_contract",
        "padding_semantics_contract_sha256",
        "padding_eligibility_dependency",
    ):
        assert f'"{field}"' in builder
    assert '"sana-phase6-dataset-contract-verification-v2"' in verifier
    assert "def independent_support(" in verifier
    assert 'row["eligible"] is not True' in verifier
    assert '"selected_minimum_support"' in verifier
    for field in (
        "padding_semantics_amendment_sha256",
        "padding_semantics_contract",
        "padding_semantics_contract_sha256",
        "padding_eligibility_dependency",
    ):
        assert f'"{field}"' in verifier


def test_dataset_scripts_use_independently_published_plan_pins() -> None:
    builder = _source("build_phase6_dataset_contract.py")
    verifier = _source("verify_phase6_dataset_contract.py")
    actual_pins = (
        "7a1063df1d97fa0b8859dc86bf22f1ba3ad23703b6f6e449388dc950b8dc3ce6",
        "701d1436c9804df960d190586c264b80af42e8ce107751693e4c5adbe1089411",
        "08b9fcf418b0c4aabf7ea5e494cc8f603e63b797658c2c3890c2a06599a965dd",
    )
    superseded_pins = (
        "3c76127f45bbc1219a2a0deb6dac73cca870cbc82d16fcc105a07f34b224b3b6",
        "8c7789761c96ceb3fb4b37b7deec2d64aac85fb281ace6db5680f604eb5b3009",
        "a30bb2958166d60ffc07be624be7c65025421c8a159b6728eaa3020ce8581ccd",
    )
    for source in (builder, verifier):
        assert all(pin in source for pin in actual_pins)
        assert all(pin not in source for pin in superseded_pins)
