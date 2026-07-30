"""Video-only T1 dtype plumbing for training, trajectory, and deployment."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from sana_wam.model.architecture import DualSystemARArchitecture


class _Scheduler:
    def __init__(self):
        self.timesteps = torch.tensor(
            [999.25, 500.5, 125.125, 0.25], dtype=torch.float32
        )
        self.sigmas = torch.tensor([1.0, 0.5, 0.125, 0.0])
        self.linear_timesteps_weights = torch.ones(4, dtype=torch.float32)
        self.num_train_timesteps = 1000
        self.flow_shift = 3.0


class _VideoBackbone(nn.Module):
    def __init__(self, *, continuous: bool):
        super().__init__()
        self.scheduler = _Scheduler()
        self.continuous_timestep_conditioning = continuous
        self.prepare_calls: list[dict[str, torch.Tensor]] = []

    def prepare(self, **kwargs):
        self.prepare_calls.append(
            {
                key: value.detach().clone()
                for key, value in kwargs.items()
                if key in {"latents", "timestep", "frame_timesteps"}
            }
        )
        return SimpleNamespace(latents=kwargs["latents"])

    def finalize(self, state):
        return torch.zeros_like(state.latents)


class _ActionBackbone(nn.Module):
    action_dim = 1

    def __init__(self):
        super().__init__()
        self.scheduler = _Scheduler()
        self.prepare_calls: list[dict[str, torch.Tensor]] = []

    def prepare_state(
        self,
        x,
        timestep,
        *,
        token_timesteps,
        **kwargs,
    ):
        del kwargs
        self.prepare_calls.append(
            {
                "timestep": timestep.detach().clone(),
                "token_timesteps": token_timesteps.detach().clone(),
            }
        )
        return SimpleNamespace(x=x)

    def extract_prediction(self, state):
        return torch.zeros_like(state.x)


class _NoOpDriver:
    def run_ar_chunk_through_backbone(
        self, backbone, state, cache, frame_id, *, store_clean, is_pred=False
    ):
        del backbone, cache, frame_id, store_clean, is_pred
        return state


def _architecture(*, continuous: bool):
    arch = DualSystemARArchitecture(cfg=None)
    video = _VideoBackbone(continuous=continuous)
    action = _ActionBackbone()
    arch.video_backbone = video
    arch.action_backbone = action
    arch._device = torch.device("cpu")
    arch._dtype = torch.bfloat16
    arch._ar_frame_chunk_size = 1
    arch._ar_attn_window = 8
    arch._ar_noisy_cond_prob = 1.0
    arch._ar_cond_max_ratio = 0.3
    return arch, video, action


def _install_forward_capture(arch):
    calls = []

    def capture(self, noisy_actions, action_timestep, **kwargs):
        del self, action_timestep
        calls.append(
            {
                "noisy_actions": noisy_actions.detach().clone(),
                **{
                    key: value.detach().clone()
                    for key, value in kwargs.items()
                    if isinstance(value, torch.Tensor)
                },
            }
        )
        return torch.zeros_like(kwargs["latents"]), torch.zeros_like(
            noisy_actions
        )

    arch.forward = MethodType(capture, arch)
    return calls


def _compute_loss_capture(*, continuous: bool):
    arch, _, _ = _architecture(continuous=continuous)
    calls = _install_forward_capture(arch)
    torch.manual_seed(1234)
    result = arch.compute_loss(
        input_latents=torch.randn(1, 1, 2, 1, 1),
        actions=torch.randn(1, 2, 1),
        lambda_video=1.0,
        lambda_action=0.0,
    )
    assert torch.isfinite(result["loss"])
    assert len(calls) == 1
    return calls[0]


def test_compute_loss_keeps_only_t1_video_timesteps_fp32():
    t0 = _compute_loss_capture(continuous=False)
    t1 = _compute_loss_capture(continuous=True)

    video_keys = (
        "ar_video_frame_timesteps",
        "ar_clean_video_frame_timesteps",
        "timestep",
    )
    action_keys = (
        "ar_action_token_timesteps",
        "ar_clean_action_token_timesteps",
    )
    assert all(t0[key].dtype == torch.bfloat16 for key in video_keys)
    assert all(t1[key].dtype == torch.float32 for key in video_keys)
    assert all(t0[key].dtype == torch.bfloat16 for key in action_keys)
    assert all(t1[key].dtype == torch.bfloat16 for key in action_keys)

    for key in ("latents", "ar_clean_latents", "noisy_actions"):
        torch.testing.assert_close(t0[key], t1[key], atol=0, rtol=0)
    for key in action_keys:
        torch.testing.assert_close(t0[key], t1[key], atol=0, rtol=0)


def test_t0_clean_copy_is_bitwise_equal_to_legacy_rng_path():
    arch, video, _ = _architecture(continuous=False)
    arch._ar_noisy_cond_prob = 0.5
    arch._ar_cond_max_ratio = 0.75
    clean = torch.arange(6, dtype=torch.bfloat16).reshape(2, 1, 3, 1, 1)

    torch.manual_seed(77)
    actual_copy, actual_timestep = arch._make_clean_copy(
        clean,
        video.scheduler,
        2,
        3,
        torch.device("cpu"),
        torch.bfloat16,
        frames=True,
        timestep_dtype=torch.bfloat16,
    )
    actual_rng = torch.get_rng_state().clone()

    torch.manual_seed(77)
    do_cond = torch.rand(2) < arch._ar_noisy_cond_prob
    max_cond = max(
        1, int(arch._ar_cond_max_ratio * len(video.scheduler.timesteps))
    )
    cond_ids = torch.randint(0, max_cond, (2,))
    raw_sigma = video.scheduler.sigmas[cond_ids].to(torch.bfloat16)
    raw_timestep = video.scheduler.timesteps[cond_ids].to(torch.bfloat16)
    zeros = torch.zeros(2, dtype=torch.bfloat16)
    sigma = torch.where(do_cond, raw_sigma, zeros)
    timestep = torch.where(do_cond, raw_timestep, zeros)
    noise = torch.randn_like(clean)
    expected_copy = (
        (1 - sigma.view(2, 1, 1, 1, 1)) * clean
        + sigma.view(2, 1, 1, 1, 1) * noise
    )
    expected_timestep = timestep.view(2, 1).expand(2, 3).contiguous()

    torch.testing.assert_close(actual_copy, expected_copy, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_timestep, expected_timestep, atol=0, rtol=0
    )
    assert torch.equal(actual_rng, torch.get_rng_state())


def test_trajectory_helper_uses_scheduler_coordinate_and_video_dtype():
    arch, _, _ = _architecture(continuous=True)
    calls = _install_forward_capture(arch)
    arch._video_trajectory_steps = 1
    arch._video_trajectory_schedule_mode = "linear_random"
    arch._video_trajectory_supervision_mode = "final"
    arch._video_trajectory_min_sigma = 0.5
    arch._video_trajectory_max_sigma = 0.5

    losses = arch._trajectory_video_losses(
        clean_video=torch.zeros(1, 1, 2, 1, 1, dtype=torch.bfloat16),
        video_noise=torch.ones(1, 1, 2, 1, 1, dtype=torch.bfloat16),
        actions=torch.zeros(1, 2, 1, dtype=torch.bfloat16),
        fwd_inputs={},
        proprio_state=None,
        proprio_per_chunk=None,
        frame_chunk_size=1,
    )

    assert len(calls) == 2
    assert all(torch.isfinite(loss) for loss in losses[:3])
    expected = (1000.0, 500.0)
    for call, value in zip(calls, expected):
        assert call["ar_video_frame_timesteps"].dtype == torch.float32
        assert call["ar_clean_video_frame_timesteps"].dtype == torch.float32
        assert call["ar_action_token_timesteps"].dtype == torch.bfloat16
        torch.testing.assert_close(
            call["ar_video_frame_timesteps"],
            torch.full((1, 2), value),
            atol=0,
            rtol=0,
        )


def test_deploy_video_is_t1_fp32_but_action_time_stays_model_dtype():
    arch, video, action = _architecture(continuous=True)
    driver = _NoOpDriver()
    like = torch.empty(1, 1, 2, 1, 1, dtype=torch.bfloat16)

    predicted = arch._denoise_video_chunk(
        driver,
        cache=object(),
        frame_id=2,
        like=like,
        v_sigmas=[1.0, 0.0],
        v_ts=torch.tensor([999.25], dtype=torch.float32),
        context=None,
        context_mask=None,
        gen=torch.Generator().manual_seed(7),
    )
    arch._ingest_clean_video(
        driver,
        object(),
        predicted,
        frame_id=2,
        context=None,
        context_mask=None,
    )

    assert predicted.dtype == torch.bfloat16
    assert all(call["timestep"].dtype == torch.float32 for call in video.prepare_calls)
    assert all(
        call["frame_timesteps"].dtype == torch.float32
        for call in video.prepare_calls
    )
    assert video.prepare_calls[0]["timestep"].item() == 999.25
    assert video.prepare_calls[-1]["timestep"].item() == 0.0

    arch._denoise_action_chunk(
        driver,
        cache=object(),
        frame_id=3,
        batch=1,
        action_tokens=1,
        a_sigmas=[1.0, 0.0],
        a_ts=torch.tensor([999.25], dtype=torch.float32),
        context=None,
        context_mask=None,
        gen=torch.Generator().manual_seed(8),
    )
    assert all(
        call["timestep"].dtype == torch.bfloat16
        and call["token_timesteps"].dtype == torch.bfloat16
        for call in action.prepare_calls
    )
