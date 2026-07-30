from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pytest


VERIFIER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_phase6_dataset_contract.py"
)


def _load_padding_verifier():
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"), filename=str(VERIFIER))
    assignment_names = {
        "EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256",
        "EXPECTED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256",
        "EXPECTED_PADDING_SEMANTICS_AMENDMENT_SHA256",
        "EXPECTED_PADDING_SEMANTICS_CONTRACT_SHA256",
        "EXPECTED_PADDING_ELIGIBILITY_DEPENDENCY",
        "EXPECTED_PADDING_SEMANTICS_CONTRACT",
    }
    function_names = {
        "canonical_json_bytes",
        "require_sha",
        "require_keys",
        "verify_padding_provenance",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in assignment_names
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected.append(node)
    namespace = {
        "Any": Any,
        "hashlib": hashlib,
        "json": json,
        "re": re,
    }
    exec(compile(ast.Module(selected, type_ignores=[]), str(VERIFIER), "exec"), namespace)
    return namespace


def _valid_padding_provenance(namespace):
    return {
        "expansion_eligibility_amendment_sha256": namespace[
            "EXPECTED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256"
        ],
        "expansion_support_contract_sha256": namespace[
            "EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256"
        ],
        "padding_semantics_amendment_sha256": namespace[
            "EXPECTED_PADDING_SEMANTICS_AMENDMENT_SHA256"
        ],
        "padding_semantics_contract": deepcopy(
            namespace["EXPECTED_PADDING_SEMANTICS_CONTRACT"]
        ),
        "padding_semantics_contract_sha256": namespace[
            "EXPECTED_PADDING_SEMANTICS_CONTRACT_SHA256"
        ],
    }


def test_verifier_does_not_import_or_call_the_primary_contract_implementation():
    source = VERIFIER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "sana_wam.dataloader.phase6_dataset_contract" not in imported_modules
    assert "sana_wam.dataloader.phase6_padding_semantics" not in imported_modules
    assert "Phase6DatasetContract" not in source


def test_verifier_rebinds_rows_and_rehashes_all_frozen_source_domains():
    source = VERIFIER.read_text(encoding="utf-8")
    assert "len(rows) != 504" in source
    assert "resolve_window(dataset, dataset_index)" in source
    assert "independent_support(" in source
    assert "require_support_types(row" in source
    assert 'row["eligible"] is not True' in source
    assert "EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256" in source
    assert "verify_padding_provenance(artifact)" in source
    assert "EXPECTED_PADDING_SEMANTICS_CONTRACT" in source
    assert "EXPECTED_PADDING_ELIGIBILITY_DEPENDENCY" in source
    assert "stable_file_descriptor(path)" in source
    assert "expected_instruction(child" in source
    for source_id in (
        "sana_wam.dataloader.robotwin_dataset",
        "sana_wam.dataloader.transforms.multiview",
        "sana_wam.dataloader.transforms.normalize",
        "sana_wam.dataloader.transforms.rotation",
    ):
        assert source_id in source


def test_verifier_publishes_a_canonical_separately_hashed_manifest():
    source = VERIFIER.read_text(encoding="utf-8")
    assert '"sana-phase6-dataset-contract-verification-v2"' in source
    assert '"row_binding_sequence_sha256"' in source
    assert '"episode_descriptor_sequence_sha256"' in source
    assert '"selected_minimum_support"' in source
    assert "atomic_exclusive_write(manifest_path, manifest_bytes)" in source
    assert 'print(f"verification_manifest_sha256={manifest_sha}")' in source


def test_independent_padding_literal_and_dependency_authenticate() -> None:
    namespace = _load_padding_verifier()
    value = _valid_padding_provenance(namespace)
    namespace["verify_padding_provenance"](value)
    observed = hashlib.sha256(
        namespace["canonical_json_bytes"](
            namespace["EXPECTED_PADDING_SEMANTICS_CONTRACT"]
        )
    ).hexdigest()
    assert observed == namespace["EXPECTED_PADDING_SEMANTICS_CONTRACT_SHA256"]


@pytest.mark.parametrize(
    ("mutation", "replacement", "match"),
    (
        ("nested_type", 1, "value or type"),
        ("nested_type", True, "value or type"),
        ("nested_type", 1.0, "value or type"),
        ("nested_extra", "extra", "unexpected keys"),
        ("adjacent_hash", 1, "lowercase SHA256"),
        ("amendment_hash", True, "lowercase SHA256"),
        ("dependency", "0" * 64, "dependency pin mismatch"),
    ),
)
def test_independent_padding_verifier_rejects_schema_type_and_dependency_drift(
    mutation,
    replacement,
    match,
) -> None:
    namespace = _load_padding_verifier()
    value = _valid_padding_provenance(namespace)
    if mutation == "nested_type":
        value["padding_semantics_contract"]["schema_version"] = replacement
        value["padding_semantics_contract_sha256"] = hashlib.sha256(
            namespace["canonical_json_bytes"](value["padding_semantics_contract"])
        ).hexdigest()
    elif mutation == "nested_extra":
        value["padding_semantics_contract"]["extra"] = replacement
    elif mutation == "adjacent_hash":
        value["padding_semantics_contract_sha256"] = replacement
    elif mutation == "amendment_hash":
        value["padding_semantics_amendment_sha256"] = replacement
    else:
        value["expansion_support_contract_sha256"] = replacement
    with pytest.raises(ValueError, match=match):
        namespace["verify_padding_provenance"](value)
