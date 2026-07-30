"""Regression tests for entropy-by-default deployment noise."""

import inspect

import pytest
import torch
from omegaconf import OmegaConf

from sana_wam.model.architecture import DualSystemARArchitecture
from tests.test_ar_rollout_engine import _build_arch, requires_sana


@pytest.mark.parametrize(
    "path",
    [
        "configs/deploy_ar_sana.yaml",
        "configs/deploy_gdn_ar.yaml",
        "configs/deploy_gdn_cross.yaml",
        "configs/deploy_sana_frozen.yaml",
        "configs/deploy_sana_frozen_nostream.yaml",
        "configs/deploy_sana_frozen_openloop.yaml",
    ],
)
def test_deploy_configs_default_seed_to_null(path):
    assert (
        OmegaConf.select(OmegaConf.load(path), "inference.seed", default="missing")
        is None
    )


def test_ar_rollout_default_seed_is_none():
    assert (
        inspect.signature(DualSystemARArchitecture.ar_rollout)
        .parameters["seed"]
        .default
        is None
    )


def test_episode_noise_seed_derivation_is_stable_and_episode_specific():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    first = ARInferenceEngine.derive_episode_noise_seed(
        20260715, "adjust_bottle/100001"
    )
    assert first == ARInferenceEngine.derive_episode_noise_seed(
        20260715, "adjust_bottle/100001"
    )
    assert first != ARInferenceEngine.derive_episode_noise_seed(
        20260715, "adjust_bottle/100002"
    )
    assert 0 <= first < 2**63


@requires_sana
def test_ar_engine_default_varies_across_episode_reset():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    cfg = OmegaConf.create(
        {
            "inference": {
                "video_steps": 2,
                "action_steps": 2,
                "ar_action_tokens_per_chunk": 2,
                "video_num_frames": 13,
            }
        }
    )
    engine = ARInferenceEngine(cfg=cfg, architecture=arch)
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    first = engine._step_with_obs_latent(obs, context, seq_lens, proprio=None).clone()
    engine.reset()
    second = engine._step_with_obs_latent(obs, context, seq_lens, proprio=None).clone()
    assert not torch.allclose(first, second)


@requires_sana
def test_ar_engine_explicit_seed_remains_reproducible():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    cfg = OmegaConf.create(
        {
            "inference": {
                "video_steps": 2,
                "action_steps": 2,
                "seed": 0,
                "ar_action_tokens_per_chunk": 2,
                "video_num_frames": 13,
            }
        }
    )
    engine = ARInferenceEngine(cfg=cfg, architecture=arch)
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)
    first = engine._step_with_obs_latent(obs, context, seq_lens, proprio=None).clone()
    engine.reset()
    second = engine._step_with_obs_latent(obs, context, seq_lens, proprio=None).clone()
    torch.testing.assert_close(first, second, atol=0, rtol=0)


@requires_sana
def test_ar_engine_paired_seed_matches_by_episode_key():
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    cfg = OmegaConf.create(
        {
            "inference": {
                "video_steps": 2,
                "action_steps": 2,
                "episode_noise_mode": "paired",
                "episode_noise_base_seed": 17,
                "ar_action_tokens_per_chunk": 2,
                "video_num_frames": 13,
            }
        }
    )
    engine = ARInferenceEngine(cfg=cfg, architecture=arch)
    context = torch.randn(1, 4, vb.context_dim)
    seq_lens = torch.full((1,), 4, dtype=torch.long)
    obs = torch.randn(1, vb._dit.in_channels, 2, 8, 8)

    engine.reset({"episode_key": "attempt-a", "noise_pair_key": "scene-100001"})
    first = engine._step_with_obs_latent(obs, context, seq_lens, proprio=None).clone()
    seed = engine.runtime_info["current_model_noise_seed"]
    engine.reset({"episode_key": "attempt-b", "noise_pair_key": "scene-100001"})
    repeated = engine._step_with_obs_latent(
        obs, context, seq_lens, proprio=None
    ).clone()
    assert engine.runtime_info["current_model_noise_seed"] == seed
    torch.testing.assert_close(first, repeated, atol=0, rtol=0)

    engine.reset({"episode_key": "attempt-c", "noise_pair_key": "scene-100002"})
    different = engine._step_with_obs_latent(obs, context, seq_lens, proprio=None)
    assert engine.runtime_info["current_model_noise_seed"] != seed
    assert not torch.allclose(first, different)


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("sana_wam.deploy.ar_engine", "ARInferenceEngine"),
        ("sana_wam.deploy.cross_attn_engine", "CrossAttnInferenceEngine"),
        ("sana_wam.deploy.gdn_ar_engine", "GDNARInferenceEngine"),
    ],
)
def test_engine_generator_helper_honors_optional_seed(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    engine = cls.__new__(cls)
    engine._device = torch.device("cpu")
    engine._seed = None
    assert engine._make_generator() is None
    engine._seed = 0
    assert isinstance(engine._make_generator(), torch.Generator)
