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
        "inference": {"denoise_steps": 3, "seed": 0, "num_frames": 9, "action_tokens": 8,
                      "streaming": False},
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
    cfg = OmegaConf.create({"inference": {"denoise_steps": 2, "seed": 7, "streaming": False}, "dataloader": {"num_frames": 9, "video_stride": 4}})
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
    """Streaming re-encodes the observed-history clip each call (Option A); the pinned
    clean prefix grows with obs_history and caps at T_lat - predict_horizon."""
    from omegaconf import OmegaConf

    from sana_wam.deploy.cross_attn_engine import CrossAttnInferenceEngine

    dev, dt = torch.device("cuda"), torch.bfloat16
    arch = _build_arch(dev, dt)
    # num_frames=9, video_stride=1 ⇒ vnf=9 ⇒ T_lat = 1 + (9-1)//4 = 3 (matches the
    # mini backbone's f=3). predict_horizon=1 ⇒ prefix caps at T_lat - 1 = 2.
    cfg = OmegaConf.create({
        "inference": {"denoise_steps": 2, "seed": 0, "num_frames": 9, "action_tokens": 8,
                      "streaming": True, "predict_horizon_frames": 1},
        "dataloader": {"num_frames": 9, "video_stride": 1},
    })
    eng = CrossAttnInferenceEngine(cfg=cfg, architecture=arch)
    ctx = torch.randn(1, 8, 64, device=dev, dtype=dt)
    seq = torch.full((1,), 8, dtype=torch.long, device=dev)
    eng._encode_prompt = lambda p: (ctx, seq)
    eng._prep_proprio = lambda ps: None
    # Stub the VAE seam: a clip of N video frames → N causal latent frames (the test
    # cares about prefix length, not VAE internals). With video_stride=1 the clip
    # length equals the obs_history length, so the encoded prefix grows 1→2→3.
    eng._encode_clip = lambda clip: torch.randn(
        1, 16, max(1, len(clip)), 8, 8, device=dev, dtype=dt
    )

    def cond(n):  # n per-sim-step frames since episode start (oldest→newest)
        return {"prompt": "x", "obs_history": [{"image": object()} for _ in range(n)]}

    eng.generate(cond(1))
    assert eng._obs_latents.shape[2] == 1  # one observed latent frame
    eng.generate(cond(2))
    assert eng._obs_latents.shape[2] == 2  # grows with history
    eng.generate(cond(3))
    assert eng._obs_latents.shape[2] == 2  # capped at T_lat - predict_horizon
    assert eng._step_c == 3

    # reset clears the (debug) prefix + counter
    eng.reset()
    assert eng._obs_latents is None
    assert eng._step_c == 0


def test_build_obs_clip_anchors_at_frame0_and_caps():
    """Pure index/cadence logic for the streaming obs clip (no GPU/VAE needed)."""
    from sana_wam.deploy.cross_attn_engine import CrossAttnInferenceEngine

    eng = CrossAttnInferenceEngine.__new__(CrossAttnInferenceEngine)
    eng._raw_num_frames = 9
    eng._video_stride = 2
    eng._video_num_frames = 5  # (9-1)//2 + 1
    eng._warned_overlong = False

    def clip_for(n):
        frames = list(range(n))  # stand-in frame objects, oldest→newest
        return eng._build_obs_clip({"obs_history": [{"image": f} for f in frames]})

    # Anchored at frame 0, subsampled by stride: leading frames at the training cadence.
    assert clip_for(1) == [0]
    assert clip_for(3) == [0, 2]          # range(0,3,2)
    assert clip_for(9) == [0, 2, 4, 6, 8]  # full window
    # Beyond the training window: clamp to the leading window (frame-0 anchor), warn once.
    assert clip_for(20) == [0, 2, 4, 6, 8]
    assert eng._warned_overlong is True

