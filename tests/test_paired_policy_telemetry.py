from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.verify_paired_policy_telemetry import _require_generation_cadence


def _cadence_run(step_count: int, horizon: int) -> dict:
    steps = []
    generations = []
    generation_index = 0
    for offset in range(step_count):
        generated = offset % horizon == 0
        steps.append(
            {
                "record": {
                    "policy": {
                        "generated": generated,
                        "chunk_offset": offset % horizon,
                    }
                }
            }
        )
        if generated:
            generations.append(
                {
                    "request_id": offset + 1,
                    "generation_index": generation_index,
                }
            )
            generation_index += 1
    return {"steps": [steps], "generations": [generations]}


@pytest.mark.parametrize(("step_count", "horizon"), [(400, 28), (400, 14), (85, 14)])
def test_generation_cadence_accepts_complete_and_partial_windows(step_count, horizon):
    _require_generation_cadence(
        _cadence_run(step_count, horizon), label="test", horizon=horizon
    )


def test_generation_cadence_rejects_unexpected_replan():
    run = _cadence_run(85, 14)
    run["generations"][0][1]["request_id"] += 1
    with pytest.raises(SystemExit, match="generation cadence mismatch"):
        _require_generation_cadence(run, label="test", horizon=14)


def test_h14_config_changes_only_execution_horizon():
    control = OmegaConf.to_container(
        OmegaConf.load("configs/deploy_ar_lownoise_paired_predicted.yaml"), resolve=True
    )
    treatment = OmegaConf.to_container(
        OmegaConf.load("configs/deploy_ar_lownoise_paired_predicted_h14.yaml"),
        resolve=True,
    )
    assert control["policy"]["execute_horizon"] is None
    assert treatment["policy"]["execute_horizon"] == 14
    normalized = deepcopy(treatment)
    normalized["policy"]["execute_horizon"] = None
    assert normalized == control


def test_paired_runner_allows_immutable_deploy_config_override():
    runner = Path("scripts/run_ar_lownoise_paired_arm.sh").read_text()
    assert 'DEPLOY_CONFIG="${DEPLOY_CONFIG:-' in runner