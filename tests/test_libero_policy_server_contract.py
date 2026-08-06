from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from sana_wam.dataloader.transforms.normalize import ActionNormalizer, Normalizer
from sana_wam.deploy.libero_model_loader import LiberoActionStateNormalizer
from sana_wam.deploy.libero_policy_server import LiberoPolicyServer
from sana_wam.deploy.policy_server import PolicyServer


class _IdentityEngine:
    def __init__(self):
        stats7 = {
            "mean": np.zeros(7, dtype=np.float32),
            "std": np.ones(7, dtype=np.float32),
            "min": -np.ones(7, dtype=np.float32),
            "max": np.ones(7, dtype=np.float32),
            "q01": -np.ones(7, dtype=np.float32),
            "q99": np.ones(7, dtype=np.float32),
        }
        stats8 = {
            key: np.resize(value, 8).astype(np.float32) for key, value in stats7.items()
        }
        self.architecture = SimpleNamespace(
            action_normalizer=LiberoActionStateNormalizer(
                action_normalizer=ActionNormalizer(mode="min_max", stats=stats7),
                state_normalizer=Normalizer(mode="min_max", stats=stats8),
            ),
        )

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
            "model": {
                "architecture": {
                    "action_dim": 7,
                    "state_dim": 8,
                    "use_proprioception": True,
                    "delta_action": False,
                }
            },
            "dataloader": {
                "type": "libero",
                "action_mode": "libero_relative_eef",
                "state_mode": "libero_eef_axis_angle_gripper",
                "normalize_mode": "min-max",
                "state_normalize_mode": "min-max",
                "delta_action": False,
                "training_video_rotation_degrees": 0,
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
    server = LiberoPolicyServer(_IdentityEngine(), cfg, identity_cfg=cfg)
    identity = server.get_info()["inference_runtime"]["deployment_identity"]

    assert identity["dataloader_type"] == "libero"
    assert identity["action_dim"] == 7
    assert identity["state_dim"] == 8
    assert identity["action_mode"] == "libero_relative_eef"
    assert identity["state_mode"] == "libero_eef_axis_angle_gripper"
    assert identity["normalize_mode"] == "min-max"
    assert identity["multiview"] is True
    assert identity["camera_layout"] == [
        "head_camera",
        "left_wrist_camera",
        "right_wrist_camera",
    ]
    assert identity["target_camera"] == "head_camera"
    assert identity["benchmark_contract"] == expected
    assert identity["normalizers"] == {
        "action": {
            "active": True,
            "configured_mode": "min-max",
            "dim": 7,
            "explicit": True,
        },
        "state": {
            "active": True,
            "configured_mode": "min-max",
            "dim": 8,
            "explicit": True,
        },
    }


def test_libero_predict_echoes_active_episode_and_model_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PolicyServer,
        "predict",
        lambda _self, _obs: {"action": [0.0] * 7, "step": 1},
    )
    server = object.__new__(LiberoPolicyServer)
    server._active_episode_key = "libero/run/test"
    server._deployment_identity = {}
    server.engine = SimpleNamespace(runtime_info={"current_model_noise_seed": 1234})

    response = server.predict({})
    assert response["episode_key"] == "libero/run/test"
    assert response["model_noise_seed"] == 1234
