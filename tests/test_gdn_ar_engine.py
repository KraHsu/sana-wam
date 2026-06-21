"""GDNARInferenceEngine: cached-GDN streaming deploy loop (one chunk per generate).

GPU/Triton-gated. Drives the engine's per-step core (`_step_with_obs_latent`)
directly with a synthetic obs latent chunk (bypassing VAE/text encode, mirroring
the ar_engine equivalence test) to pin: one AR step returns the right action
shape, the rolling GDN cache advances across steps and runs PAST the built clip
length (arbitrary-length deploy), and reset() reseeds to a deterministic rollout.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

_GPU = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="cached GDN kernels need CUDA")

FCS = 3
ATC = 4  # action tokens per chunk (override so divisibility is explicit)


def _build_engine(dev, dt):
    import sana_wam.model.video_backbone.sana as _s  # noqa: F401
    from omegaconf import OmegaConf

    from sana_wam.deploy.gdn_ar_engine import GDNARInferenceEngine
    from sana_wam.model.gdn_ar import DualSystemGDNARArchitecture
    from sana_wam.model.video_backbone.sana.adapter import SanaVideoBackbone

    vb = SanaVideoBackbone.from_mini_config(
        depth=2, hidden_size=224, num_heads=2, linear_head_dim=112,
        f=FCS, h=8, w=8, device="cuda", dtype=dt, attn_kernel="gdn", chunk_size=FCS,
    )
    arch = DualSystemGDNARArchitecture(cfg=None)
    arch.video_backbone = vb
    arch._cross_cfg = {"action_dim": 20, "dim": 128, "num_heads": 2, "attn_head_dim": 64,
                       "bridge_layers": [0, 1], "text_dim": 64, "ffn_dim": 256}
    arch._device, arch._dtype = dev, dt
    arch.build_action_backbone()
    arch._frame_chunk_size = FCS
    arch.to(dt).cuda()

    cfg = OmegaConf.create({
        "inference": {"video_steps": 2, "action_steps": 2, "seed": 0,
                      "video_num_frames": 9, "action_tokens_per_chunk": ATC},
        "dataloader": {"num_frames": 9, "video_stride": 4},
    })
    eng = GDNARInferenceEngine(cfg=cfg, architecture=arch)
    return eng, arch


def _ctx(dev, dt):
    return (torch.randn(1, 8, 64, device=dev, dtype=dt),
            torch.full((1,), 8, dtype=torch.long, device=dev))


@requires_gpu
def test_step_runs_and_advances_cache():
    dev, dt = torch.device("cuda"), torch.bfloat16
    eng, arch = _build_engine(dev, dt)
    ctx, seq = _ctx(dev, dt)
    obs = torch.randn(1, 16, FCS, 8, 8, device=dev, dtype=dt)

    # step 0 = bootstrap (first-frame pin, empty cache, predicts chunk 0): NO ingest.
    a0 = eng._step_with_obs_latent(obs, ctx, seq, None)
    assert a0.shape == (1, ATC, 20) and torch.isfinite(a0).all()
    assert eng._cache[0][0] is None  # nothing ingested yet at episode start
    assert eng._step_c == 1

    # step 1 ingests the just-observed chunk 0 → cache advances.
    a1 = eng._step_with_obs_latent(obs, ctx, seq, None)
    assert a1.shape == (1, ATC, 20) and torch.isfinite(a1).all()
    assert eng._cache[0][0] is not None  # chunk 0 ingested into the GDN state
    assert eng._step_c == 2


@requires_gpu
def test_streaming_runs_past_training_horizon():
    """Five AR steps via the rolling cache — predict window advances to chunk 4
    [4*FCS,5*FCS) > built f=FCS, proving arbitrary-length deploy. Cache (None at the
    bootstrap step 0) advances from step 1 on."""
    dev, dt = torch.device("cuda"), torch.bfloat16
    eng, arch = _build_engine(dev, dt)
    ctx, seq = _ctx(dev, dt)

    prev_state = None
    for step in range(5):
        obs = torch.randn(1, 16, FCS, 8, 8, device=dev, dtype=dt)
        a = eng._step_with_obs_latent(obs, ctx, seq, None)
        assert a.shape == (1, ATC, 20) and torch.isfinite(a).all()
        state = eng._cache[0][0]
        if step == 0:
            assert state is None  # bootstrap: nothing ingested yet
        else:
            assert state is not None  # chunk step-1 ingested
            if prev_state is not None:
                assert not torch.equal(prev_state, state), f"cache stalled at step {step}"
            prev_state = state.detach().clone()
    assert eng._step_c == 5


@requires_gpu
def test_reset_reseeds_deterministic():
    dev, dt = torch.device("cuda"), torch.bfloat16
    eng, arch = _build_engine(dev, dt)
    ctx, seq = _ctx(dev, dt)
    obs = torch.randn(1, 16, FCS, 8, 8, device=dev, dtype=dt)

    a1 = eng._step_with_obs_latent(obs, ctx, seq, None).float().cpu().numpy()
    eng.reset()
    assert eng._step_c == 0 and eng._cache[0][0] is None  # cache cleared
    a2 = eng._step_with_obs_latent(obs, ctx, seq, None).float().cpu().numpy()
    np.testing.assert_allclose(a1, a2, rtol=1e-3, atol=1e-3)


def test_deploy_gdn_ar_config_loads():
    """deploy_gdn_ar.yaml loads with the knobs GDNARInferenceEngine reads (no GPU)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/deploy_gdn_ar.yaml")
    assert int(cfg.inference.video_steps) >= 1
    assert int(cfg.inference.action_steps) >= 1
    assert cfg.inference.ar_obs_latent_band == "auto"
    # action_tokens_per_chunk=0 → engine infers it; the policy must stay greedy.
    assert int(cfg.inference.action_tokens_per_chunk) == 0
    assert cfg.policy.temporal_ensemble is False
    assert cfg.optimization.async_inference.mode == "none"


@requires_gpu
def test_build_engine_dispatch_picks_gdn_ar():
    """build_engine must route the GDN-AR arch to GDNARInferenceEngine, NOT its
    cross-attn parent's engine (isinstance order bug guard)."""
    from omegaconf import OmegaConf

    from sana_wam.deploy import build_engine
    from sana_wam.deploy.gdn_ar_engine import GDNARInferenceEngine

    dev, dt = torch.device("cuda"), torch.bfloat16
    _, arch = _build_engine(dev, dt)  # reuse the built arch
    cfg = OmegaConf.create({
        "inference": {"video_steps": 2, "action_steps": 2, "video_num_frames": 9,
                      "action_tokens_per_chunk": ATC},
        "dataloader": {"num_frames": 9, "video_stride": 4},
    })
    eng = build_engine(cfg, arch)
    assert isinstance(eng, GDNARInferenceEngine)
