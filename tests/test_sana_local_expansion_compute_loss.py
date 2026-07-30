from __future__ import annotations

from types import MethodType

import pytest
import torch
from torch import nn

from sana_wam.model.architecture import DualSystemARArchitecture
from sana_wam.model.local_expansion import masked_video_rms


class _Scheduler:
    def __init__(self, sigma=0.5):
        self.sigmas = torch.tensor([sigma], dtype=torch.float32)
        self.timesteps = self.sigmas * 1000.0
        self.linear_timesteps_weights = torch.ones(1, dtype=torch.float32)
        self.num_train_timesteps = 1000
        self.flow_shift = 3.0

    def training_weight(self, indices):
        return torch.ones_like(indices, dtype=torch.float32)


class _VideoBackbone(nn.Module):
    continuous_timestep_conditioning = False

    def __init__(self):
        super().__init__()
        self.scheduler = _Scheduler()


class _ActionBackbone(nn.Module):
    loss_weighting = "none"

    def __init__(self):
        super().__init__()
        self.scheduler = _Scheduler()


def _plan_row(global_step=1, *, noise_seed=101, direction_seed=202):
    zero_based = global_step - 1
    return {
        "action_sigma": (1.0, 0.9, 0.5)[zero_based % 3],
        "cycle": zero_based // 42,
        "domain_seeds": {
            "video-noise": 11,
            "action-noise": 12,
            "expansion-noise": noise_seed,
            "expansion-direction": direction_seed,
            "reference-query": 15,
            "prompt-choice": 16,
        },
        "global_step": global_step,
        "identity": {
            "dataset_index": zero_based,
            "episode_index": 0,
            "episode_path": "/clean/episode0.hdf5",
            "prompt": "exact prompt",
            "source_dataset": "RoboTwin",
            "source_kind": "ordinary_expert",
            "source_variant": "clean_50",
            "start_frame": 0,
            "task_name": "adjust_bottle",
        },
        "identity_sha256": "b" * 64,
        "plan_sha256": "a" * 64,
        "position_in_cycle": zero_based % 42,
    }


def _architecture(*, expansion_enabled=True, on_path_weight=0.0):
    architecture = DualSystemARArchitecture(cfg=None)
    architecture.video_backbone = _VideoBackbone()
    architecture.action_backbone = _ActionBackbone()
    architecture._device = torch.device("cpu")
    architecture._dtype = torch.float32
    architecture._ar_frame_chunk_size = 2
    architecture._ar_attn_window = 8
    architecture._ar_noisy_cond_prob = 0.0
    architecture._ar_cond_max_ratio = 0.3
    architecture._ar_bootstrap_clean_prefix = False
    architecture._ar_chunkwise_temporal_ops = True
    architecture._proprio_per_chunk = False
    architecture._video_on_path_loss_weight = on_path_weight
    architecture._video_trajectory_endpoint_weight = 0.0
    architecture._video_trajectory_velocity_weight = 0.0
    architecture._video_trajectory_consistency_weight = 0.0
    architecture._video_local_expansion_weight = 1.0 if expansion_enabled else 0.0
    architecture.expansion_test_slope = nn.Parameter(torch.tensor(-100.0))
    architecture.expansion_forward_states = []

    def fake_forward(self, noisy_actions, action_timestep, **inputs):  # noqa: ARG001
        state = inputs["latents"]
        self.expansion_forward_states.append(state.detach().clone())
        velocity = self.expansion_test_slope * state
        return velocity, torch.zeros_like(noisy_actions)

    architecture.forward = MethodType(fake_forward, architecture)
    return architecture


def _batch(*, batch_size=1, plan_row=None):
    clean = torch.tensor(
        [0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4, -1.6],
        dtype=torch.float32,
    ).reshape(1, 1, 4, 1, 2).expand(batch_size, -1, -1, -1, -1).clone()
    batch = {
        "input_latents": clean,
        "actions": torch.zeros(batch_size, 4, 2),
        "video_is_pad": torch.tensor(
            [[False, False, False, True]]
        ).expand(batch_size, -1).clone(),
        "action_is_pad": torch.tensor(
            [[False, False, False, True]]
        ).expand(batch_size, -1).clone(),
    }
    if plan_row is not None:
        batch["phase6_plan_rows"] = (plan_row,)
    return batch


def test_compute_loss_uses_analytic_center_and_exactly_one_extra_forward():
    architecture = _architecture()
    row = _plan_row(1, noise_seed=101)
    batch = _batch(plan_row=row)
    output = architecture.compute_loss(
        lambda_video=1.0, lambda_action=0.0, **batch
    )

    assert len(architecture.expansion_forward_states) == 2
    center, plus = architecture.expansion_forward_states
    changed = plus != center
    assert not bool(changed[:, :, :2].any())
    assert bool(changed[:, :, 2].any())
    assert not bool(changed[:, :, 3].any())

    generator = torch.Generator().manual_seed(101)
    expansion_noise = torch.randn(batch["input_latents"].shape, generator=generator)
    expected_center = (
        0.5 * batch["input_latents"] + 0.5 * expansion_noise
    )
    torch.testing.assert_close(center[:, :, 2], expected_center[:, :, 2])

    actual_chord = plus - center
    mask = torch.tensor([[False, False, True, False]])
    torch.testing.assert_close(
        masked_video_rms(actual_chord, mask), torch.tensor([0.005])
    )
    assert torch.isfinite(output["loss"])
    assert output["loss"].item() > 0
    assert output["loss_video_on_path"].item() == 0
    assert output["loss_action"].item() == 0
    assert output["video_local_expansion_extra_forwards"].item() == 1
    assert output["video_local_expansion_direction_index"].item() == 0
    assert output["video_local_expansion_epsilon_rms"].item() == pytest.approx(
        0.005
    )
    assert output["video_local_expansion_action_sigma"].item() == 1.0
    assert output["video_local_expansion_delta_sigma"].item() < 0

    output["loss"].backward()
    gradient = architecture.expansion_test_slope.grad
    assert gradient is not None and torch.isfinite(gradient)
    assert gradient.abs().item() > 0


def test_local_expansion_ignores_nan_storage_padding():
    row = _plan_row(1, noise_seed=101)
    baseline_batch = _batch(plan_row=row)
    changed_batch = _batch(plan_row=row)
    changed_batch["input_latents"][:, :, 3] = float("nan")
    changed_batch["actions"][:, 3:] = float("nan")

    torch.manual_seed(707)
    baseline = _architecture().compute_loss(
        lambda_video=1.0, lambda_action=0.0, **baseline_batch
    )
    torch.manual_seed(707)
    changed = _architecture().compute_loss(
        lambda_video=1.0, lambda_action=0.0, **changed_batch
    )
    for name in (
        "loss",
        "loss_video_local_expansion",
        "video_local_expansion_rate",
        "video_local_expansion_factor",
        "video_local_expansion_chord_rms",
        "video_local_expansion_next_chord_rms",
        "phase6_action_unweighted_mse",
    ):
        assert torch.isfinite(changed[name]).all()
        torch.testing.assert_close(changed[name], baseline[name], atol=0, rtol=0)

    value = torch.tensor([[[[[3.0]], [[4.0]], [[float("nan")]]]]])
    mask = torch.tensor([[True, True, False]])
    torch.testing.assert_close(masked_video_rms(value, mask), torch.tensor([3.535534]))


def test_compute_loss_orthogonal_query_is_seeded_and_row_scheduled():
    row = _plan_row(2, noise_seed=303, direction_seed=404)

    def run(global_seed):
        torch.manual_seed(global_seed)
        architecture = _architecture()
        output = architecture.compute_loss(
            lambda_video=1.0,
            lambda_action=0.0,
            **_batch(plan_row=row),
        )
        center, plus = architecture.expansion_forward_states
        return center[:, :, 2], plus[:, :, 2], output

    center_a, plus_a, output_a = run(1)
    center_b, plus_b, output_b = run(999)
    assert torch.equal(center_a, center_b)
    assert torch.equal(plus_a, plus_b)
    assert output_a["video_local_expansion_direction_index"].item() == 1
    assert output_a["video_local_expansion_epsilon_rms"].item() == pytest.approx(
        0.01
    )
    assert output_b["video_local_expansion_action_sigma"].item() == pytest.approx(
        0.9
    )


def test_compute_loss_disabled_is_zero_and_requires_no_phase6_row():
    architecture = _architecture(expansion_enabled=False, on_path_weight=1.0)
    output = architecture.compute_loss(
        lambda_video=1.0, lambda_action=0.0, **_batch()
    )
    assert len(architecture.expansion_forward_states) == 1
    assert output["loss_video_local_expansion"].item() == 0
    assert output["video_local_expansion_extra_forwards"].item() == 0
    assert torch.equal(output["loss"].detach(), output["loss_video_on_path"])


def test_phase6_e0_e1_use_bitwise_identical_current_chunk_center():
    row = _plan_row(2, noise_seed=707, direction_seed=808)
    batch = _batch(plan_row=row)

    torch.manual_seed(909)
    control = _architecture(expansion_enabled=False, on_path_weight=1.0)
    control_output = control.compute_loss(
        lambda_video=1.0, lambda_action=0.0, **batch
    )
    control_rng_state = torch.random.get_rng_state()

    torch.manual_seed(909)
    treatment = _architecture(expansion_enabled=True, on_path_weight=1.0)
    treatment_output = treatment.compute_loss(
        lambda_video=1.0, lambda_action=0.0, **batch
    )
    treatment_rng_state = torch.random.get_rng_state()

    assert torch.equal(
        control.expansion_forward_states[0],
        treatment.expansion_forward_states[0],
    )
    assert len(control.expansion_forward_states) == 1
    assert len(treatment.expansion_forward_states) == 2
    assert torch.equal(control_rng_state, treatment_rng_state)
    assert control_output["loss_video_local_expansion"].item() == 0.0
    assert control_output["video_local_expansion_extra_forwards"].item() == 0
    assert treatment_output["video_local_expansion_extra_forwards"].item() == 1


def test_compute_loss_expansion_contract_fails_closed():
    architecture = _architecture()
    with pytest.raises(ValueError, match="phase6_plan_rows"):
        architecture.compute_loss(
            lambda_video=1.0, lambda_action=0.0, **_batch()
        )
    with pytest.raises(ValueError, match="batch size exactly 1"):
        architecture.compute_loss(
            lambda_video=1.0,
            lambda_action=0.0,
            **_batch(batch_size=2, plan_row=_plan_row()),
        )
    architecture._video_trajectory_endpoint_weight = 0.1
    with pytest.raises(ValueError, match="legacy video trajectory"):
        architecture.compute_loss(
            lambda_video=1.0,
            lambda_action=0.0,
            **_batch(plan_row=_plan_row()),
        )


def test_config_rejects_nonfrozen_weight_and_legacy_mix():
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        DualSystemARArchitecture._validate_local_expansion_contract(
            weight=0.5,
            legacy_weights=(0.0, 0.0, 0.0),
            chunkwise_temporal_ops=True,
        )
    with pytest.raises(ValueError, match="legacy video trajectory"):
        DualSystemARArchitecture._validate_local_expansion_contract(
            weight=1.0,
            legacy_weights=(0.1, 0.0, 0.0),
            chunkwise_temporal_ops=True,
        )
