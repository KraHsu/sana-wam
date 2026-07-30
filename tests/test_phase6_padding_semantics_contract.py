from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from sana_wam.dataloader.phase6_padding_semantics import (
    PADDING_SEMANTICS_AMENDMENT_SHA256,
    PADDING_SEMANTICS_CONTRACT_SHA256,
    PaddingSemanticsError,
    padding_eligibility_dependency,
    padding_semantics_contract,
    validate_padding_eligibility_dependency,
    validate_padding_semantics_contract,
)


PADDING_AMENDMENT_PATH_ENV = "SANA_PHASE6_PADDING_SEMANTICS_AMENDMENT"
DEFAULT_PADDING_AMENDMENT_PATH = Path(
    "/DATA/share/sana_phase6_principled_constraints_20260724/"
    "phase6_padding_semantics_amendment_20260724.json"
)


def registered_padding_amendment_path() -> Path:
    configured = os.environ.get(PADDING_AMENDMENT_PATH_ENV)
    if configured is not None:
        return Path(configured).expanduser()
    return DEFAULT_PADDING_AMENDMENT_PATH


def test_registered_padding_amendment_path_is_configurable(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(PADDING_AMENDMENT_PATH_ENV, raising=False)
    assert registered_padding_amendment_path() == DEFAULT_PADDING_AMENDMENT_PATH

    configured = tmp_path / "registered-padding-amendment.json"
    monkeypatch.setenv(PADDING_AMENDMENT_PATH_ENV, str(configured))
    assert registered_padding_amendment_path() == configured


def canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def test_helper_exactly_matches_registered_padding_amendment() -> None:
    amendment_path = registered_padding_amendment_path()
    if not amendment_path.exists():
        pytest.skip(f"registered padding amendment is unavailable: {amendment_path}")
    amendment_bytes = amendment_path.read_bytes()
    amendment = json.loads(amendment_bytes)
    assert amendment_bytes == canonical_bytes(amendment)
    assert hashlib.sha256(amendment_bytes).hexdigest() == (
        PADDING_SEMANTICS_AMENDMENT_SHA256
    )
    assert padding_semantics_contract() == amendment["padding_semantics_contract"]
    assert hashlib.sha256(canonical_bytes(padding_semantics_contract())).hexdigest() == (
        PADDING_SEMANTICS_CONTRACT_SHA256
    )
    assert padding_eligibility_dependency() == amendment["eligibility_dependency"]


def test_helper_returns_defensive_provenance_copies() -> None:
    contract = padding_semantics_contract()
    contract["schema_version"] = "mutated"
    assert padding_semantics_contract()["schema_version"] == (
        "sana-phase6-padding-semantics-contract-v1"
    )

    dependency = padding_eligibility_dependency()
    dependency["amendment_sha256"] = "0" * 64
    assert padding_eligibility_dependency()["amendment_sha256"] != "0" * 64


@pytest.mark.parametrize("replacement", (1, True, 1.0))
def test_padding_contract_validator_rejects_numeric_type_substitutions(
    replacement,
) -> None:
    contract = padding_semantics_contract()
    contract["schema_version"] = replacement
    with pytest.raises(PaddingSemanticsError, match="value or type"):
        validate_padding_semantics_contract(contract)


def test_padding_contract_validator_requires_exact_nested_keys() -> None:
    contract = padding_semantics_contract()
    contract["extra"] = "forbidden"
    with pytest.raises(PaddingSemanticsError, match="unexpected schema"):
        validate_padding_semantics_contract(contract)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("expansion_eligibility_amendment_sha256", "0" * 64),
        ("expansion_support_contract_sha256", "0" * 64),
        ("expansion_eligibility_amendment_sha256", 1),
        ("expansion_support_contract_sha256", True),
        ("expansion_support_contract_sha256", 1.0),
    ),
)
def test_padding_eligibility_dependency_rejects_drift_and_numeric_aliases(
    field,
    replacement,
) -> None:
    dependency = padding_eligibility_dependency()
    arguments = {
        "expansion_eligibility_amendment_sha256": dependency["amendment_sha256"],
        "expansion_support_contract_sha256": dependency[
            "support_contract_sha256"
        ],
    }
    arguments[field] = replacement
    with pytest.raises(PaddingSemanticsError, match="eligibility dependency"):
        validate_padding_eligibility_dependency(**arguments)
