"""CPU gate (no SANA weights): the AR train config flattens correctly.

Pins that
``configs/train_ar_sana.yaml`` carries the canonical AR knobs + linear_relu
kernel and that ``flatten_model_cfg`` merges architecture/action_backbone/
video_backbone into the flat dict the architecture ``__init__`` consumes.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from sana_wam.config import flatten_model_cfg

CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "train_ar_sana.yaml"


def test_ar_config_flattens_to_autoregressive():
    cfg = OmegaConf.load(CFG_PATH)
    p = flatten_model_cfg(cfg.model)

    assert p["framework"] == "dual_system"
    assert p["variant"] == "autoregressive"

    # AR knobs survive flattening (read by DualSystemARArchitecture.__init__).
    assert int(p["ar_frame_chunk_size"]) == 2
    assert int(p["ar_attn_window"]) == 72
    assert float(p["ar_noisy_cond_prob"]) == 0.5
    assert bool(p["proprio_per_chunk"]) is True
    assert bool(p["ar_bootstrap_clean_prefix"]) is True
    assert float(p["proprio_action_dropout_prob"]) == 0.0
    assert p["action_loss_weighting"] == "none"

    # SANA linear-attn dispatch on both sides: video_backbone subdict kept
    # nested; action-backbone fields (incl. attn_kernel) merged to top level.
    assert p["video_backbone"]["attn_kernel"] == "linear_relu"
    assert p["attn_kernel"] == "linear_relu"
