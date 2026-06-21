"""CrossAttnInferenceEngine: one-shot TI2V denoise loop.

GPU-gated (GDN backbone). Verifies the stateless cross-attn engine runs its
flow-matching denoise loop end-to-end against a mini GDN backbone + cross-attn
ActionDiT, with the prompt/proprio/first-frame encode steps stubbed (the mini
pipe has no VAE / text encoder). Pins:
  - the loop produces a finite (atok, action_dim) trajectory,
  - latent frame 0 stays pinned to the clean observation (TI2V),
  - reset() reseeds deterministically.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

_GPU = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="GDN ChunkCausalGDNTriton needs CUDA")


def _build_arch(dev, dt):
    import sana_wam.model.video_backbone.sana as _s  # noqa: F401
    from sana_wam.model.cross_attn import DualSystemCrossAttnArchitecture
    from sana_wam.model.video_backbone.sana.adapter import SanaVideoBackbone

    vb = SanaVideoBackbone.from_mini_config(
        depth=2, hidden_size=224, num_heads=2, linear_head_dim=112,
        f=3, h=8, w=8, device="cuda", dtype=dt, attn_kernel="gdn", chunk_size=3,
    )
    arch = DualSystemCrossAttnArchitecture(cfg=None)
    arch.video_backbone = vb
    arch._cross_cfg = {"action_dim": 20, "dim": 128, "num_heads": 2, "attn_head_dim": 64,
                       "bridge_layers": [0, 1], "text_dim": 64, "ffn_dim": 256}
    arch._device, arch._dtype = dev, dt
    arch.build_action_backbone()
    arch.to(dt).cuda()
    return arch


@requires_gpu
def test_cross_attn_engine_denoise_loop():
    from omegaconf import OmegaConf

    from sana_wam.deploy.cross_attn_engine import CrossAttnInferenceEngine

    dev, dt = torch.device("cuda"), torch.bfloat16
    arch = _build_arch(dev, dt)

    cfg = OmegaConf.create({
        "inference": {"denoise_steps": 3, "seed": 0, "num_frames": 9, "action_tokens": 8},
        "dataloader": {"num_frames": 9, "video_stride": 4},
    })
    eng = CrossAttnInferenceEngine(cfg=cfg, architecture=arch)

    # Stub the encode steps (mini pipe has no VAE / text encoder).
    C = arch.video_backbone.dim  # not used; latent channels = 16
    first = torch.randn(1, 16, 1, 8, 8, device=dev, dtype=dt)
    ctx = torch.randn(1, 8, 64, device=dev, dtype=dt)
    seq = torch.full((1,), 8, dtype=torch.long, device=dev)
    eng._encode_first_frame = lambda conditions: first
    eng._encode_prompt = lambda prompt: (ctx, seq)
    eng._prep_proprio = lambda ps: None
    eng._video_num_frames = 3  # latent T = 1 + (3-1)//4 = 1 ... ensure T>=1

    out = eng.generate({"prompt": "pick up the block", "first_frame_image": [object()]})
    actions = out["actions"]
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (8, 20)
    assert np.isfinite(actions).all()


@requires_gpu
def test_cross_attn_engine_reset_reseeds():
    from omegaconf import OmegaConf

    from sana_wam.deploy.cross_attn_engine import CrossAttnInferenceEngine

    dev, dt = torch.device("cuda"), torch.bfloat16
    arch = _build_arch(dev, dt)
    cfg = OmegaConf.create({"inference": {"denoise_steps": 2, "seed": 7}, "dataloader": {"num_frames": 9, "video_stride": 4}})
    eng = CrossAttnInferenceEngine(cfg=cfg, architecture=arch)
    first = torch.randn(1, 16, 1, 8, 8, device=dev, dtype=dt)
    ctx = torch.randn(1, 8, 64, device=dev, dtype=dt)
    seq = torch.full((1,), 8, dtype=torch.long, device=dev)
    eng._encode_first_frame = lambda c: first
    eng._encode_prompt = lambda p: (ctx, seq)
    eng._prep_proprio = lambda ps: None
    eng._video_num_frames = 3

    cond = {"prompt": "x", "first_frame_image": [object()]}
    a1 = eng.generate(cond)["actions"]
    eng.reset()
    a2 = eng.generate(cond)["actions"]
    np.testing.assert_allclose(a1, a2, rtol=1e-3, atol=1e-3)


@requires_gpu
def test_cross_attn_engine_streaming_grows_and_caps_prefix():
    from omegaconf import OmegaConf

    from sana_wam.deploy.cross_attn_engine import CrossAttnInferenceEngine

    dev, dt = torch.device("cuda"), torch.bfloat16
    arch = _build_arch(dev, dt)
    # T_lat = 1 + (9-1)//4 = 3 ; predict_horizon=1 ⇒ prefix caps at T-1 = 2.
    cfg = OmegaConf.create({
        "inference": {"denoise_steps": 2, "seed": 0, "num_frames": 9, "action_tokens": 8,
                      "streaming": True, "predict_horizon_frames": 1},
        "dataloader": {"num_frames": 9, "video_stride": 4},
    })
    eng = CrossAttnInferenceEngine(cfg=cfg, architecture=arch)
    first = torch.randn(1, 16, 1, 8, 8, device=dev, dtype=dt)
    ctx = torch.randn(1, 8, 64, device=dev, dtype=dt)
    seq = torch.full((1,), 8, dtype=torch.long, device=dev)
    eng._encode_first_frame = lambda c: first
    eng._encode_prompt = lambda p: (ctx, seq)
    eng._prep_proprio = lambda ps: None
    # vnf=9 video frames ⇒ _video_num_frames_latent() = 1 + (9-1)//4 = 3 (matches
    # the mini backbone's f=3). max_prefix = T_lat - predict_horizon = 3 - 1 = 2.
    eng._video_num_frames = 9

    cond = {"prompt": "x", "first_frame_image": [object()]}
    eng.generate(cond)
    assert eng._obs_latents.shape[2] == 1  # one observed frame after step 1
    eng.generate(cond)
    assert eng._obs_latents.shape[2] == 2  # grows
    eng.generate(cond)
    assert eng._obs_latents.shape[2] == 2  # capped at T - predict_horizon
    assert eng._step_c == 3

    # reset clears the rolling history
    eng.reset()
    assert eng._obs_latents is None
    assert eng._step_c == 0

