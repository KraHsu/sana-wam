"""Regression tests for the deployment video Euler master-state contract."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from sana_wam.model.architecture import DualSystemARArchitecture


class _ConstantVelocityVideoBackbone(nn.Module):
    def __init__(self, velocity: float):
        super().__init__()
        self.velocity = float(velocity)
        self.model_input_dtypes: list[torch.dtype] = []
        self.timestep_dtypes: list[torch.dtype] = []
        self.frame_timestep_dtypes: list[torch.dtype] = []
        self.model_inputs: list[torch.Tensor] = []

    def prepare(self, **kwargs):
        latents = kwargs["latents"]
        self.model_input_dtypes.append(latents.dtype)
        self.timestep_dtypes.append(kwargs["timestep"].dtype)
        self.frame_timestep_dtypes.append(kwargs["frame_timesteps"].dtype)
        self.model_inputs.append(latents.detach().clone())
        return SimpleNamespace(latents=latents)

    def finalize(self, state):
        return torch.full_like(state.latents, self.velocity)


class _NoOpDriver:
    def __init__(self):
        self.calls: list[tuple[bool, bool]] = []

    def run_ar_chunk_through_backbone(
        self, backbone, state, cache, frame_id, *, store_clean, is_pred=False
    ):
        self.calls.append((bool(store_clean), bool(is_pred)))
        return state


def test_video_euler_uses_fp32_master_with_model_dtype_forwards():
    arch = DualSystemARArchitecture(cfg=None)
    arch._device = torch.device("cpu")
    arch._dtype = torch.bfloat16
    backbone = _ConstantVelocityVideoBackbone(velocity=-0.1)
    arch.video_backbone = backbone
    driver = _NoOpDriver()

    steps = 200
    sigmas = [1.0 - i / steps for i in range(steps)] + [0.0]
    timesteps = [sigma * 1000.0 for sigma in sigmas[:-1]]
    shape = (1, 1, 1, 1, 1)

    expected_gen = torch.Generator().manual_seed(0)
    expected = torch.randn(shape, generator=expected_gen, dtype=torch.bfloat16).float()
    expected_generator_state = expected_gen.get_state().clone()
    velocity = torch.full(shape, -0.1, dtype=torch.bfloat16)
    legacy = expected.to(torch.bfloat16)
    for index in range(steps):
        delta = sigmas[index + 1] - sigmas[index]
        expected = expected + velocity.float() * delta
        legacy = legacy + velocity * delta

    actual_gen = torch.Generator().manual_seed(0)
    actual = arch._denoise_video_chunk(
        driver,
        cache=object(),
        frame_id=2,
        like=torch.empty(shape, dtype=torch.bfloat16),
        v_sigmas=sigmas,
        v_ts=timesteps,
        context=None,
        context_mask=None,
        gen=actual_gen,
    )

    expected_model_state = expected.to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected_model_state, atol=0, rtol=0)
    assert not torch.equal(actual, legacy)
    assert backbone.model_input_dtypes == [torch.bfloat16] * (steps + 1)
    assert backbone.timestep_dtypes == [torch.bfloat16] * (steps + 1)
    assert backbone.frame_timestep_dtypes == [torch.bfloat16] * (steps + 1)
    torch.testing.assert_close(
        backbone.model_inputs[-1], expected_model_state, atol=0, rtol=0
    )
    assert driver.calls == [(False, False)] * steps + [(True, True)]
    assert torch.equal(actual_gen.get_state(), expected_generator_state)


def test_fp32_model_path_remains_bitwise_equal_to_native_recurrence():
    arch = DualSystemARArchitecture(cfg=None)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    backbone = _ConstantVelocityVideoBackbone(velocity=0.25)
    arch.video_backbone = backbone
    driver = _NoOpDriver()

    sigmas = [1.0, 0.7, 0.2, 0.0]
    timesteps = [1000.0, 700.0, 200.0]
    shape = (1, 1, 1, 1, 1)
    expected_gen = torch.Generator().manual_seed(11)
    expected = torch.randn(shape, generator=expected_gen, dtype=torch.float32)
    velocity = torch.full(shape, 0.25, dtype=torch.float32)
    for index in range(len(timesteps)):
        expected = expected + velocity * (sigmas[index + 1] - sigmas[index])

    actual = arch._denoise_video_chunk(
        driver,
        cache=object(),
        frame_id=2,
        like=torch.empty(shape, dtype=torch.float32),
        v_sigmas=sigmas,
        v_ts=timesteps,
        context=None,
        context_mask=None,
        gen=torch.Generator().manual_seed(11),
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fp32_master_accumulates_updates_lost_by_bf16_state():
    initial = torch.tensor([1.0], dtype=torch.bfloat16)
    velocity = torch.tensor([0.5], dtype=torch.bfloat16)
    delta_sigma = -0.001

    native = initial.clone()
    master = initial.float()
    for _ in range(200):
        native = native + velocity * delta_sigma
        master = master + velocity.float() * delta_sigma

    assert native.item() == 1.0
    torch.testing.assert_close(master, torch.tensor([0.9]), atol=5e-6, rtol=0)
