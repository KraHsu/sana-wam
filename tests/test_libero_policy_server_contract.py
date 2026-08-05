from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf

from sana_wam.deploy.policy_server import PolicyServer


class _IdentityEngine:
    architecture = None

    def generate(self, _conditions):
        return {"actions": np.zeros((1, 7), dtype=np.float32)}

    def reset(self, episode_context=None):
        self._context = dict(episode_context or {})

    @property
    def runtime_info(self):
        context = getattr(self, "_context", {})
        return {
            "episode_noise_mode": "paired",
            "current_model_noise_seed": context.get("model_noise_seed"),
        }


def _expected_contract() -> dict:
    config = yaml.safe_load(
        Path("benchmarks/libero/policy_config.yml").read_text(encoding="utf-8")
    )
    return config["expected_server_contract"]


def test_policy_server_exposes_libero_checkpoint_compatibility_identity() -> None:
    expected = _expected_contract()
    cfg = OmegaConf.create(
        {
            "model": {"architecture": {"action_dim": 7, "state_dim": 8}},
            "dataloader": {
                "type": "libero",
                "action_mode": "libero_relative_eef",
                "normalize_mode": "min-max",
                "multiview": True,
                "camera_layout": [
                    "head_camera",
                    "left_wrist_camera",
                    "right_wrist_camera",
                ],
                "target_camera": "head_camera",
                "benchmark_contract": expected,
            },
            "inference": {"episode_noise_mode": "paired"},
            "policy": {
                "history_len": 10,
                "execute_horizon": None,
                "temporal_ensemble": False,
            },
            "telemetry": {"enabled": False},
        }
    )
    server = PolicyServer(_IdentityEngine(), cfg)
    identity = server.get_info()["inference_runtime"]["deployment_identity"]

    assert identity["dataloader_type"] == "libero"
    assert identity["action_dim"] == 7
    assert identity["state_dim"] == 8
    assert identity["action_mode"] == "libero_relative_eef"
    assert identity["normalize_mode"] == "min-max"
    assert identity["multiview"] is True
    assert identity["camera_layout"] == [
        "head_camera",
        "left_wrist_camera",
        "right_wrist_camera",
    ]
    assert identity["target_camera"] == "head_camera"
    assert identity["benchmark_contract"] == expected
