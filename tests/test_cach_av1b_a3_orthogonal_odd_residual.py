from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from sana_wam.model.cach_av1b_a3_orthogonal_odd_residual import (
    A3_CONFIG_SCHEMA,
    A3BridgeSpec,
    CACH_A3_ARCHITECTURE_ID,
    ExactOddCausalActionResidual,
    classify_cach_a3,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "src/sana_wam/model/cach_av1b_a3_orthogonal_odd_residual.py"
CONFIG = REPO_ROOT / "configs/experiments/cach_av1b_a3_orthogonal_odd_residual.yaml"
RUNNER = REPO_ROOT / "scripts/run_cach_av1b_a3_orthogonal_odd_residual.py"
CARD = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a3_orthogonal/"
    "CACH_A3_ORTHOGONAL_ODD_RESIDUAL_RUN_CARD.json"
)


def _strict_json(path: Path) -> dict[str, object]:
    def reject(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=reject)


def test_a3_sources_parse_and_contract_identity_is_consistent() -> None:
    ast.parse(BRIDGE.read_text())
    ast.parse(RUNNER.read_text())
    config = _strict_json(CONFIG)
    card = _strict_json(CARD)
    assert config["schema"] == A3_CONFIG_SCHEMA
    assert config["architecture_id"] == CACH_A3_ARCHITECTURE_ID
    assert card["architecture_id"] == CACH_A3_ARCHITECTURE_ID
    assert card["execution"]["automatic_av2_or_rerun"] is False
    assert config["training"]["macrosteps"] == 200
    assert config["training"]["losses"] == {
        "action": "POSITIVE_MSE_ACTION_DELTA_ONLY",
        "common": "POSITIVE_MSE_PAIRED_COMMON_ONLY",
        "loss_subtraction": False,
    }


def test_exact_odd_head_is_zero_origin_and_typed_masked_on_cpu() -> None:
    spec = A3BridgeSpec()
    head = ExactOddCausalActionResidual(spec)
    raw = torch.linspace(-1.0, 1.0, 2 * 5 * 20).reshape(2, 5, 20)
    mask = torch.tensor([False, True, True, True, True]).view(1, 5, 1)
    mask = mask.expand(2, 5, 1).clone()

    theta0, _ = head(raw, mask, count_calls=False)
    assert torch.count_nonzero(theta0).item() == 0
    with torch.no_grad():
        weight = torch.arange(60, dtype=torch.float32).reshape(3, 20)
        head.output_projection.weight.copy_((weight - weight.mean()) / 100.0)

    positive, _ = head(raw, mask, count_calls=False)
    negative, _ = head(-raw, mask, count_calls=False)
    zero, _ = head(torch.zeros_like(raw), mask, count_calls=False)
    inactive, _ = head(raw, torch.zeros_like(mask), count_calls=False)

    assert float((positive + negative).abs().max()) <= 1.0e-6
    assert torch.count_nonzero(zero).item() == 0
    assert torch.count_nonzero(inactive).item() == 0
    assert torch.count_nonzero(positive[:, 0]).item() == 0


def test_a3_source_keeps_action_out_of_vendor_blocks() -> None:
    source = BRIDGE.read_text()
    assert "video_prediction = common_prediction + delta_video" not in source
    assert "prediction = common + delta" in source
    assert "0.5 * (positive - negative)" in source
    assert "block.enable_candidate_action_seam" not in source
    assert "mode=\"no_action\"" in source


def test_registered_classifier_uses_action_energy_metrics_and_common_subtype() -> None:
    config = _strict_json(CONFIG)
    good = {
        "target_half_delta_energy": 0.25,
        "delta_mse": 0.0025,
        "delta_nmse": 0.01,
        "delta_energy_ratio": 1.0,
        "delta_alignment_cosine": 0.99,
        "action_explained_fraction": 0.99,
        "correct_mse": 0.01,
        "no_action_mse": 0.26,
        "shuffle_mse": 1.01,
        "no_action_recovery": 1.0,
        "shuffle_penalty": 4.0,
    }
    action_metrics = {
        "aggregate": good,
        "by_horizon": {str(horizon): dict(good) for horizon in (1, 2, 3, 4)},
    }
    stable = classify_cach_a3(
        config,
        all_validity=True,
        action_metrics=action_metrics,
        common_stable=True,
    )
    blocked = classify_cach_a3(
        config,
        all_validity=True,
        action_metrics=action_metrics,
        common_stable=False,
    )
    invalid = classify_cach_a3(
        config,
        all_validity=False,
        action_metrics=action_metrics,
        common_stable=True,
    )
    assert stable[0] == "OPERATOR_GO_COMMON_STABLE"
    assert blocked[:2] == (
        "OPERATOR_GO_COMMON_BLOCKED",
        "A3_ACTION_GO_COMMON_BLOCKED",
    )
    assert invalid[0] == "INVALID_RUN"
    assert stable[2]["aggregate_go_checks"]["shuffle_penalty"] is True


def test_runner_is_one_shot_and_has_no_token_or_av2_path() -> None:
    source = RUNNER.read_text()
    assert "EXECUTE_CACH_A3_ONCE" in source
    assert "O_EXCL" in source
    assert "automatic_av2_or_rerun" in source
    assert "token_ledgers" not in source
    assert "checkpoint.load" not in source


@pytest.mark.skip(reason="requires separately authorized single-GPU vendor execution")
def test_a3_vendor_gpu_screen_is_explicitly_gated() -> None:
    raise AssertionError("GPU screen must be launched only through the frozen runner")
