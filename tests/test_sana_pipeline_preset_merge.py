"""SANA bundle presets remain intact when YAML adds model kwargs."""

from __future__ import annotations

import json

import pytest


def _make_bundle(tmp_path, *, with_checkpoint: bool):
    bundle = tmp_path / "sana_bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text(
        json.dumps({"model_name": "SANA-Video-2B-480p"}),
        encoding="utf-8",
    )
    if with_checkpoint:
        checkpoint_dir = bundle / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.pth").touch()
    return bundle


@pytest.mark.parametrize("with_checkpoint", [True, False])
def test_incremental_model_kwargs_extend_discovered_2b_preset(tmp_path, with_checkpoint):
    from sana_wam.model.video_backbone.sana.pipeline_builder import _resolve_model_path_and_kwargs

    bundle = _make_bundle(tmp_path, with_checkpoint=with_checkpoint)
    extras = {
        "additional_flash_attn": "window_flash",
        "flash_attn_window_count": [1, 1, 4],
        "linear_feature_map": "learnable",
        "class_dropout_prob": 0.0,
    }
    model_path, kwargs, _ = _resolve_model_path_and_kwargs(
        str(bundle),
        extras,
        None,
        ckpt_dir=None if with_checkpoint else "/tmp/native-checkpoint",
    )

    assert model_path == (str(bundle / "checkpoints" / "model.pth") if with_checkpoint else None)
    assert kwargs["in_channels"] == 16
    assert kwargs["attn_type"] == "LiteLAReLURope"
    assert kwargs["ffn_type"] == "GLUMBConvTemp"
    assert kwargs["linear_head_dim"] == 112
    assert kwargs["pred_sigma"] is False
    for key, value in extras.items():
        assert kwargs[key] == value
