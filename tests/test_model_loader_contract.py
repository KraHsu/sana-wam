"""Checkpoint-loader construction and pinning contracts."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf


def test_base_threads_private_build_context_to_video_backbone(monkeypatch):
    from sana_wam.model.base import BaseWAMArchitecture
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone

    captured = {}

    def fake_from_pretrained(source, **kwargs):
        captured["source"] = source
        captured["kwargs"] = kwargs
        return nn.Identity()

    monkeypatch.setattr(SanaVideoBackbone, "from_pretrained", fake_from_pretrained)

    class MinimalArchitecture(BaseWAMArchitecture):
        def forward(self, *args, **kwargs):
            raise NotImplementedError

    cfg = OmegaConf.create(
        {
            "video_backbone": {
                "model_path": "/models/sana",
                "_device": "cpu",
                "_ckpt_dir": "/checkpoints/run",
            }
        }
    )
    architecture = MinimalArchitecture(cfg)

    assert architecture.video_backbone is not None
    assert captured["source"] is cfg
    assert captured["kwargs"] == {"device": "cpu", "ckpt_dir": "/checkpoints/run"}


def test_loader_injects_device_and_uses_exact_checkpoint_name(tmp_path, monkeypatch):
    from sana_wam.deploy import model_loader

    OmegaConf.save(
        OmegaConf.create(
            {
                "accelerate": {"mixed_precision": "no"},
                "model": {
                    "architecture": {"framework": "dual_system", "variant": "autoregressive"},
                    "video_backbone": {"name": "sana_video_2b"},
                    "action_backbone": {},
                },
                "dataloader": {"normalize_mode": None},
            }
        ),
        tmp_path / "config.yaml",
    )
    checkpoint = tmp_path / "checkpoint_step_12000.safetensors"
    checkpoint.touch()
    captured = {}

    class DummyArchitecture:
        def set_dtype_device(self, dtype, device):
            captured["dtype_device"] = (dtype, device)

        def load_checkpoint(self, path, **kwargs):
            captured["load"] = (path, kwargs)

        def eval(self):
            return self

        def attach_action_normalizer(self, normalizer):
            captured["normalizer"] = normalizer

    def fake_build(flat_cfg):
        captured["flat_cfg"] = flat_cfg
        return DummyArchitecture()

    monkeypatch.setattr(model_loader, "build_architecture", fake_build)
    _, architecture = model_loader.load_from_checkpoint_dir(
        str(tmp_path),
        device="cpu",
        ckpt_name=checkpoint.name,
    )

    assert isinstance(architecture, DummyArchitecture)
    assert captured["flat_cfg"].video_backbone._device == "cpu"
    assert Path(captured["flat_cfg"].video_backbone._ckpt_dir) == tmp_path
    assert captured["dtype_device"] == (torch.float32, torch.device("cpu"))
    assert captured["load"] == (
        str(checkpoint),
        {"strict": True, "allow_missing_patterns": ("norm_kv",)},
    )
    assert captured["normalizer"] is None
