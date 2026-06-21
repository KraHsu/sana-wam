"""Architecture dispatch by ``variant`` (build_architecture).

CPU-safe: only checks that ``build_architecture`` selects the right class for
each ``variant`` and raises on unknown. Does NOT build the (GPU-only) GDN video
backbone — uses ``cfg=None``-style flat cfgs with no ``video_backbone`` so the
class is constructed without instantiating a backbone.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from sana_wam.config import flatten_model_cfg
from sana_wam.model import build_architecture


def _flat(variant: str):
    # Minimal model cfg: architecture.variant set, no video_backbone (so no
    # backbone is built — we only assert the dispatched class). bridge_interval +
    # num_dit_layers let the action-side bridge resolution succeed without a
    # video backbone present.
    cfg = OmegaConf.create(
        {
            "model": {
                "architecture": {
                    "framework": "dual_system",
                    "variant": variant,
                    "bridge_interval": 1,
                    "num_dit_layers": 2,
                    "video_dim": 128,
                    "num_heads": 2,
                    "attn_head_dim": 64,
                    "dim": 128,
                }
            }
        }
    )
    return flatten_model_cfg(cfg.model)


def test_dispatch_autoregressive():
    from sana_wam.model.architecture import DualSystemARArchitecture

    arch = build_architecture(_flat("autoregressive"))
    assert isinstance(arch, DualSystemARArchitecture)


def test_dispatch_cross_attn():
    from sana_wam.model.cross_attn import DualSystemCrossAttnArchitecture

    arch = build_architecture(_flat("joint_cross_attn"))
    assert isinstance(arch, DualSystemCrossAttnArchitecture)


def test_dispatch_self_attn():
    from sana_wam.model.joint_self_attn import DualSystemSelfAttnArchitecture

    arch = build_architecture(_flat("joint_self_attn"))
    assert isinstance(arch, DualSystemSelfAttnArchitecture)


def test_dispatch_gdn_autoregressive():
    from sana_wam.model.gdn_ar import DualSystemGDNARArchitecture

    arch = build_architecture(_flat("gdn_autoregressive"))
    assert isinstance(arch, DualSystemGDNARArchitecture)


def test_dispatch_unknown_raises():
    with pytest.raises(ValueError, match="Unknown architecture.variant"):
        build_architecture(_flat("nonsense_variant"))


def test_gdn_cross_config_loads_and_dispatches():
    cfg = OmegaConf.load("configs/train_gdn_cross.yaml")
    flat = flatten_model_cfg(cfg.model)
    assert flat.variant == "joint_cross_attn"
    assert flat.video_backbone.attn_kernel == "gdn"
    assert bool(flat.video_backbone.use_first_frame_cond) is True
    # model_path null → GDN trains from scratch (no incompatible linear_relu load).
    assert flat.video_backbone.model_path is None


def test_gdn_ar_config_loads_and_dispatches():
    cfg = OmegaConf.load("configs/train_gdn_ar.yaml")
    flat = flatten_model_cfg(cfg.model)
    assert flat.variant == "gdn_autoregressive"
    assert flat.video_backbone.attn_kernel == "gdn"
    # frame_chunk_size must equal the GDN backbone chunk_size and the dataloader's
    # gdn_chunk_size so the growing-history clean prefix is chunk-aligned.
    assert int(flat.frame_chunk_size) == int(flat.video_backbone.chunk_size) == 3
    # v2: growing-history rolling AR — num_frames is auto (no hard divisibility);
    # the rolling loss covers cache depths via variable valid-length clips.
    assert bool(cfg.dataloader.growing_history) is True
    assert int(cfg.dataloader.gdn_chunk_size) == int(flat.frame_chunk_size) == 3
    assert int(cfg.dataloader.video_stride) == 2
    assert int(flat.ar_observed_prefix_chunks) == 1
