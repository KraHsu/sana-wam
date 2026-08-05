"""Pure mapping tests for CACH/AFCC lineage isolation."""

from __future__ import annotations

import pytest

from sana_wam.cach.config import CACHConfigError
from sana_wam.cach.verifier import (
    validate_afcc_isolation,
    validate_afcc_pair_isolation,
)


def test_reference_and_candidate_accept_only_absent_or_integer_zero_f() -> None:
    validate_afcc_pair_isolation(
        {"model": {"architecture": {"F": 0}}},
        {"model": {"architecture": {"F": 0}}},
    )
    validate_afcc_pair_isolation(
        {"model": {"architecture": {"variant": "cach_sana_wam_v0"}}},
        {"model": {"architecture": {"variant": "cach_sana_wam_v0"}}},
    )

    for value in (1, False, "0", 0.0, None):
        with pytest.raises(
            CACHConfigError,
            match="CACH_AFCC_ISOLATION_VIOLATION",
        ):
            validate_afcc_isolation({"model": {"F": value}})


@pytest.mark.parametrize(
    "forbidden",
    [
        {"afcc_weight": "/DATA/legacy/weight.safetensors"},
        {"AF-CC": {"enabled": False}},
        {"phase_6_authority": "legacy.json"},
        {"phase-6-root": "/DATA/frozen"},
        {"action_reference": None},
        {"action-reference-path": None},
        {"action_facing_cache_reference": None},
        {"loss_subtraction": 0.0},
    ],
)
def test_forbidden_keys_fail_closed_even_when_disabled_or_null(
    forbidden: dict,
) -> None:
    with pytest.raises(
        CACHConfigError,
        match="CACH_AFCC_ISOLATION_VIOLATION",
    ):
        validate_afcc_isolation({"nested": [{"deeper": forbidden}]})


@pytest.mark.parametrize(
    "forbidden_value",
    [
        "/home/zch/workspace/sana-afcc-handoff",
        "/DATA/formal/phase_6/root",
        "AFCC formal reference",
        "loss-subtraction",
    ],
)
def test_forbidden_lineage_cannot_be_hidden_in_string_values(
    forbidden_value: str,
) -> None:
    with pytest.raises(
        CACHConfigError,
        match="CACH_AFCC_ISOLATION_VIOLATION",
    ):
        validate_afcc_isolation({"authority": forbidden_value})


def test_one_arm_cannot_reintroduce_afcc_state() -> None:
    reference = {"model": {"F": 0}}
    candidate = {
        "model": {"F": 0},
        "adapter": {"action_reference_checkpoint": None},
    }
    with pytest.raises(
        CACHConfigError,
        match="CACH_AFCC_ISOLATION_VIOLATION",
    ):
        validate_afcc_pair_isolation(reference, candidate)
