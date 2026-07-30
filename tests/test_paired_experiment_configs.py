from __future__ import annotations

from copy import deepcopy

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("mode", ["reencode_predicted", "measured"])
def test_paired_configs_are_single_variable_ab(mode):
    control = OmegaConf.to_container(
        OmegaConf.load("configs/deploy_ar_lownoise_paired_predicted.yaml"), resolve=True
    )
    treatment = OmegaConf.to_container(
        OmegaConf.load(f"configs/deploy_ar_lownoise_paired_{mode}.yaml"), resolve=True
    )
    assert control["inference"]["episode_noise_mode"] == "paired"
    assert (
        control["inference"]["episode_noise_base_seed"]
        == treatment["inference"]["episode_noise_base_seed"]
    )
    assert control["inference"]["cache_feedback_mode"] == "predicted"
    assert treatment["inference"]["cache_feedback_mode"] == mode

    normalized = deepcopy(treatment)
    normalized["inference"]["cache_feedback_mode"] = "predicted"
    assert normalized == control


def test_canonical_baseline_does_not_opt_into_experimental_protocol():
    canonical = OmegaConf.load("configs/baselines/deploy_ar_lownoise_seedfixed.yaml")
    assert (
        OmegaConf.select(canonical, "inference.episode_noise_mode", default="ambient")
        == "ambient"
    )
    assert (
        OmegaConf.select(
            canonical, "inference.cache_feedback_mode", default="predicted"
        )
        == "predicted"
    )
    assert OmegaConf.select(canonical, "telemetry.enabled", default=False) is False
