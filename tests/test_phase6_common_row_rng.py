from __future__ import annotations

from types import MethodType

import pytest
import torch
from torch import nn

from sana_wam.model.architecture import (
    DualSystemARArchitecture,
    _phase6_child_seed,
)


class _Scheduler:
    def __init__(self):
        self.sigmas = torch.tensor([0.15, 0.35, 0.55, 0.75], dtype=torch.float32)
        self.timesteps = torch.tensor(
            [137.25, 351.5, 598.75, 877.125], dtype=torch.float32
        )
        self.linear_timesteps_weights = torch.ones(4, dtype=torch.float32)
        self.num_train_timesteps = 1000
        self.flow_shift = 3.0

    def training_weight(self, indices):
        return torch.ones_like(indices, dtype=torch.float32)


class _VideoBackbone(nn.Module):
    def __init__(self, *, continuous_time: bool):
        super().__init__()
        self.continuous_timestep_conditioning = continuous_time
        self.scheduler = _Scheduler()


class _ActionBackbone(nn.Module):
    loss_weighting = "none"

    def __init__(self):
        super().__init__()
        self.scheduler = _Scheduler()


class _ScalarAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(()))


def _plan_row(global_step=2):
    zero_based = global_step - 1
    return {
        "action_sigma": (1.0, 0.9, 0.5)[zero_based % 3],
        "cycle": zero_based // 42,
        "domain_seeds": {
            "video-noise": 11,
            "action-noise": 12,
            "expansion-noise": 13,
            "expansion-direction": 14,
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


def _batch(*, row=None):
    batch = {
        "input_latents": torch.tensor(
            [0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4, -1.6]
        ).reshape(1, 1, 4, 1, 2),
        "actions": torch.tensor(
            [[[-0.7, 0.3], [0.2, -0.5], [0.8, 0.4], [-0.1, 0.6]]]
        ),
        # SANA preprocess emits this prepared control as a plain Python int.
        "num_clean_prefix_frames": 0,
        "video_is_pad": torch.zeros(1, 4, dtype=torch.bool),
        "action_is_pad": torch.zeros(1, 4, dtype=torch.bool),
    }
    if row is not None:
        batch["phase6_plan_rows"] = (row,)
    return batch


def _architecture(
    *,
    continuous_time=False,
    expansion=False,
    action_non_regression=False,
    dtype=torch.bfloat16,
):
    architecture = DualSystemARArchitecture(cfg=None)
    architecture.video_backbone = _VideoBackbone(continuous_time=continuous_time)
    architecture.action_backbone = _ActionBackbone()
    architecture._device = torch.device("cpu")
    architecture._dtype = dtype
    architecture._ar_frame_chunk_size = 2
    architecture._ar_attn_window = 8
    architecture._ar_noisy_cond_prob = 1.0
    architecture._ar_cond_max_ratio = 1.0
    architecture._ar_bootstrap_clean_prefix = False
    architecture._ar_chunkwise_temporal_ops = True
    architecture._proprio_per_chunk = False
    architecture._video_on_path_loss_weight = 1.0
    architecture._video_trajectory_endpoint_weight = 0.0
    architecture._video_trajectory_velocity_weight = 0.0
    architecture._video_trajectory_consistency_weight = 0.0
    architecture._video_local_expansion_weight = 1.0 if expansion else 0.0
    architecture._action_non_regression_weight = (
        1.0 if action_non_regression else 0.0
    )
    architecture.action_video_memory_adapter = (
        _ScalarAdapter() if action_non_regression else None
    )
    architecture.forward_calls = []

    def fake_forward(self, noisy_actions, action_timestep, **inputs):  # noqa: ARG001
        keys = (
            "latents",
            "ar_clean_latents",
            "ar_clean_actions",
            "ar_video_frame_timesteps",
            "ar_clean_video_frame_timesteps",
            "ar_action_token_timesteps",
            "ar_clean_action_token_timesteps",
        )
        capture = {key: inputs[key].detach().clone() for key in keys}
        capture["noisy_actions"] = noisy_actions.detach().clone()
        self.forward_calls.append(capture)
        action_prediction = torch.zeros_like(noisy_actions)
        if self.action_video_memory_adapter is not None:
            action_prediction = action_prediction + (
                self.action_video_memory_adapter.gain * (noisy_actions + 1.0)
            )
        return torch.zeros_like(inputs["latents"]), action_prediction

    architecture.forward = MethodType(fake_forward, architecture)
    return architecture


def _run_phase6(
    *, continuous_time, expansion, action_non_regression=False, global_seed
):
    architecture = _architecture(
        continuous_time=continuous_time,
        expansion=expansion,
        action_non_regression=action_non_regression,
    )
    if not continuous_time:
        architecture.eval()
    torch.manual_seed(global_seed)
    before = torch.random.get_rng_state().clone()
    batch = _batch(row=_plan_row())
    if action_non_regression:
        batch["phase1_action_reference_error"] = torch.tensor(
            1.0, dtype=torch.float32
        )
    output = architecture.compute_loss(
        lambda_video=1.0,
        lambda_action=0.0,
        phase6_common_input_trace_required=True,
        phase6_t0_reference_forward_trace_required=not continuous_time,
        **batch,
    )
    after = torch.random.get_rng_state().clone()
    return architecture, output, before, after


def test_phase6_e0_e1_t0_t1_share_row_noise_and_common_trace():
    t0_e0, t0_result, t0_before, t0_after = _run_phase6(
        continuous_time=False, expansion=False, global_seed=101
    )
    t1_e0, t1_result, t1_before, t1_after = _run_phase6(
        continuous_time=True, expansion=False, global_seed=202
    )
    t1_e1, t1e1_result, t1e1_before, t1e1_after = _run_phase6(
        continuous_time=True, expansion=True, global_seed=303
    )
    t1_a1, t1a1_result, t1a1_before, t1a1_after = _run_phase6(
        continuous_time=True,
        expansion=False,
        action_non_regression=True,
        global_seed=404,
    )
    t1e1_a1, t1e1a1_result, t1e1a1_before, t1e1a1_after = _run_phase6(
        continuous_time=True,
        expansion=True,
        action_non_regression=True,
        global_seed=505,
    )

    assert torch.equal(t0_before, t0_after)
    assert torch.equal(t1_before, t1_after)
    assert torch.equal(t1e1_before, t1e1_after)
    assert torch.equal(t1a1_before, t1a1_after)
    assert torch.equal(t1e1a1_before, t1e1a1_after)
    assert len(t0_e0.forward_calls) == len(t1_e0.forward_calls) == 1
    assert len(t1_e1.forward_calls) == 2
    assert len(t1_a1.forward_calls) == 1
    assert len(t1e1_a1.forward_calls) == 2
    assert (
        t0_result["phase6_common_input_trace_sha256"]
        == t1_result["phase6_common_input_trace_sha256"]
        == t1e1_result["phase6_common_input_trace_sha256"]
        == t1a1_result["phase6_common_input_trace_sha256"]
        == t1e1a1_result["phase6_common_input_trace_sha256"]
    )
    # T0 and T1 intentionally expose different effective video timesteps, so
    # their model-output trace is not a cross-arm equality contract.
    assert t0_result["phase6_timestep_forward_trace_sha256"] is not None
    assert t1_result["phase6_timestep_forward_trace_sha256"] is None
    assert t1e1_result["phase6_timestep_forward_trace_sha256"] is None

    t0_center = t0_e0.forward_calls[0]
    t1_center = t1_e0.forward_calls[0]
    e1_center = t1_e1.forward_calls[0]
    paired_keys = set(t0_center) - {
        "ar_video_frame_timesteps",
        "ar_clean_video_frame_timesteps",
    }
    for key in paired_keys:
        assert torch.equal(t0_center[key], t1_center[key]), key
    for key in t1_center:
        assert torch.equal(t1_center[key], e1_center[key]), key

    assert t0_center["ar_video_frame_timesteps"].dtype == torch.bfloat16
    assert t1_center["ar_video_frame_timesteps"].dtype == torch.float32
    for capture in (t0_center, t1_center, e1_center):
        action_ts = capture["ar_action_token_timesteps"]
        assert action_ts.dtype == torch.bfloat16
        assert torch.equal(action_ts, torch.full_like(action_ts, 900.0))

    row = _plan_row()
    actions = _batch()["actions"].to(torch.bfloat16)
    action_generator = torch.Generator().manual_seed(
        _phase6_child_seed(row["domain_seeds"]["action-noise"], "noisy-epsilon")
    )
    action_noise = torch.randn(
        actions.shape, generator=action_generator, dtype=torch.bfloat16
    )
    sigma = torch.full((1, 4), 0.9, dtype=torch.bfloat16)
    expected_noisy_actions = (
        (1 - sigma.unsqueeze(-1)) * actions + sigma.unsqueeze(-1) * action_noise
    )
    assert torch.equal(t1_center["noisy_actions"], expected_noisy_actions)

    clean = _batch()["input_latents"].to(torch.bfloat16)
    expansion_generator = torch.Generator().manual_seed(
        row["domain_seeds"]["expansion-noise"]
    )
    expansion_noise = torch.randn(
        clean.shape, generator=expansion_generator, dtype=torch.bfloat16
    )
    timestep_generator = torch.Generator().manual_seed(
        _phase6_child_seed(
            row["domain_seeds"]["video-noise"], "noisy-chunk-timestep"
        )
    )
    video_ids = torch.randint(0, 4, (1, 2), generator=timestep_generator)
    video_ids = video_ids.repeat_interleave(2, dim=1)
    video_sigma = _Scheduler().sigmas[video_ids].reshape(1, 1, 4, 1, 1)
    expected_current = (
        (1.0 - video_sigma[:, :, 2:]) * clean[:, :, 2:].float()
        + video_sigma[:, :, 2:] * expansion_noise[:, :, 2:].float()
    ).to(torch.bfloat16)
    assert torch.equal(t1_center["latents"][:, :, 2:], expected_current)


def _common_trace_with_frame_prefix(value):
    architecture = _architecture(continuous_time=True, expansion=False)
    batch = _batch(row=_plan_row())
    batch["num_clean_prefix_frames"] = value
    result = architecture.compute_loss(
        lambda_video=1.0,
        lambda_action=0.0,
        phase6_common_input_trace_required=True,
        **batch,
    )
    return result["phase6_common_input_trace_sha256"]


def _common_trace_with_action_prefix(value):
    architecture = _architecture(continuous_time=True, expansion=False)
    batch = _batch(row=_plan_row())
    batch["num_clean_prefix_actions"] = value
    result = architecture.compute_loss(
        lambda_video=1.0,
        lambda_action=0.0,
        phase6_common_input_trace_required=True,
        **batch,
    )
    return result["phase6_common_input_trace_sha256"]


def test_phase6_prefix_count_partitions_scalar_and_tensor_trace_values():
    scalar_zero = _common_trace_with_frame_prefix(0)
    tensor_zero = _common_trace_with_frame_prefix(
        torch.tensor([0], dtype=torch.long)
    )

    assert scalar_zero == _common_trace_with_frame_prefix(0)
    assert tensor_zero == _common_trace_with_frame_prefix(
        torch.tensor([0], dtype=torch.long)
    )
    assert scalar_zero != tensor_zero
    assert scalar_zero != _common_trace_with_frame_prefix(None)
    assert scalar_zero != _common_trace_with_frame_prefix(1)

    action_absent = scalar_zero
    action_scalar = _common_trace_with_action_prefix(0)
    action_tensor = _common_trace_with_action_prefix(
        torch.tensor([0], dtype=torch.long)
    )
    assert len({action_absent, action_scalar, action_tensor}) == 3


@pytest.mark.parametrize("mask_name", ["video_is_pad", "action_is_pad"])
def test_phase6_common_trace_binds_every_padding_mask_bit(mask_name):
    architecture = _architecture(continuous_time=True, expansion=False)
    baseline_batch = _batch(row=_plan_row())
    baseline = architecture.compute_loss(
        lambda_video=1.0,
        lambda_action=0.0,
        phase6_common_input_trace_required=True,
        **baseline_batch,
    )["phase6_common_input_trace_sha256"]

    changed_batch = _batch(row=_plan_row())
    changed_batch[mask_name][:, -1] = True
    changed = _architecture(continuous_time=True, expansion=False).compute_loss(
        lambda_video=1.0,
        lambda_action=0.0,
        phase6_common_input_trace_required=True,
        **changed_batch,
    )["phase6_common_input_trace_sha256"]
    assert changed != baseline


@pytest.mark.parametrize("missing_name", ["video_is_pad", "action_is_pad"])
def test_phase6_requires_both_explicit_padding_masks(missing_name):
    architecture = _architecture(continuous_time=True, expansion=False)
    batch = _batch(row=_plan_row())
    del batch[missing_name]
    with pytest.raises(ValueError, match=f"explicit {missing_name}"):
        architecture.compute_loss(
            lambda_video=1.0,
            lambda_action=0.0,
            phase6_common_input_trace_required=True,
            **batch,
        )
    assert architecture.forward_calls == []


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (True, TypeError),
        (0.0, TypeError),
        ([0], TypeError),
        ("0", TypeError),
        (torch.tensor(0, dtype=torch.long), TypeError),
        (torch.tensor([0], dtype=torch.int32), TypeError),
        (torch.tensor([0, 0], dtype=torch.long), TypeError),
        (-1, ValueError),
        (5, ValueError),
    ),
)
def test_phase6_prefix_count_rejects_noncanonical_values_before_forward(
    value, error
):
    architecture = _architecture(continuous_time=True, expansion=False)
    batch = _batch(row=_plan_row())
    batch["num_clean_prefix_frames"] = value

    with pytest.raises(error, match="num_clean_prefix_frames"):
        architecture.compute_loss(
            lambda_video=1.0,
            lambda_action=0.0,
            phase6_common_input_trace_required=True,
            **batch,
        )

    assert architecture.forward_calls == []


def test_phase6_common_trace_rejects_non_tensor_data_before_forward():
    architecture = _architecture(continuous_time=True, expansion=False)
    batch = _batch(row=_plan_row())
    batch["proprio_seq"] = ["not", "a", "tensor"]

    with pytest.raises(TypeError, match="tensors or None"):
        architecture.compute_loss(
            lambda_video=1.0,
            lambda_action=0.0,
            phase6_common_input_trace_required=True,
            **batch,
        )

    assert architecture.forward_calls == []


def test_phase6_child_seeds_are_sha_pinned_and_order_independent():
    expected = {
        name: _phase6_child_seed(123456789, name)
        for name in ("noisy-epsilon", "clean-copy-timestep", "clean-copy-epsilon")
    }
    observed = {
        name: _phase6_child_seed(123456789, name)
        for name in reversed(tuple(expected))
    }
    assert observed == expected
    assert len(set(expected.values())) == len(expected)
    assert expected["noisy-epsilon"] == 3565948219461637023


def test_phase6_rejects_action_loss_before_touching_rng():
    architecture = _architecture(continuous_time=True, expansion=False)
    torch.manual_seed(404)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="lambda_action exactly 0"):
        architecture.compute_loss(
            lambda_video=1.0,
            lambda_action=1.0,
            **_batch(row=_plan_row()),
        )
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize(
    ("global_step", "action_sigma"), ((1, 1.0), (2, 0.9), (3, 0.5))
)
def test_phase6_uses_registered_action_sigma_for_every_chunk(
    global_step, action_sigma
):
    architecture = _architecture(continuous_time=True, expansion=False)
    architecture.compute_loss(
        lambda_video=1.0,
        lambda_action=0.0,
        **_batch(row=_plan_row(global_step)),
    )
    action_ts = architecture.forward_calls[0]["ar_action_token_timesteps"]
    assert action_ts.dtype == torch.bfloat16
    assert torch.equal(
        action_ts,
        torch.full_like(action_ts, action_sigma * 1000.0),
    )


def test_legacy_no_row_path_retains_exact_draw_order_and_rng_state():
    architecture = _architecture(
        continuous_time=False, expansion=False, dtype=torch.float32
    )
    architecture._ar_noisy_cond_prob = 0.5
    architecture._ar_cond_max_ratio = 0.75
    batch = _batch()

    torch.manual_seed(505)
    architecture.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)
    observed_state = torch.random.get_rng_state().clone()
    capture = architecture.forward_calls[0]

    clean = batch["input_latents"].float()
    actions = batch["actions"].float()
    scheduler = architecture.video_backbone.scheduler
    torch.manual_seed(505)
    video_chunk_ids = torch.randint(0, 4, (1, 2))
    video_ids = video_chunk_ids.repeat_interleave(2, dim=1)
    video_sigma = scheduler.sigmas[video_ids]
    video_noise = torch.randn_like(clean)
    expected_video = (
        (1 - video_sigma.reshape(1, 1, 4, 1, 1)) * clean
        + video_sigma.reshape(1, 1, 4, 1, 1) * video_noise
    )
    video_do_cond = torch.rand(1) < 0.5
    video_cond_ids = torch.randint(0, 3, (1,))
    video_cond_sigma = torch.where(
        video_do_cond, scheduler.sigmas[video_cond_ids], torch.zeros(1)
    )
    video_cond_ts = torch.where(
        video_do_cond, scheduler.timesteps[video_cond_ids], torch.zeros(1)
    )
    video_clean_noise = torch.randn_like(clean)
    expected_clean_video = (
        (1 - video_cond_sigma.reshape(1, 1, 1, 1, 1)) * clean
        + video_cond_sigma.reshape(1, 1, 1, 1, 1) * video_clean_noise
    )

    action_chunk_ids = torch.randint(0, 4, (1, 2))
    action_ids = action_chunk_ids.repeat_interleave(2, dim=1)
    action_sigma = scheduler.sigmas[action_ids]
    action_noise = torch.randn_like(actions)
    expected_actions = (
        (1 - action_sigma.unsqueeze(-1)) * actions
        + action_sigma.unsqueeze(-1) * action_noise
    )
    action_do_cond = torch.rand(1) < 0.5
    action_cond_ids = torch.randint(0, 3, (1,))
    action_cond_sigma = torch.where(
        action_do_cond, scheduler.sigmas[action_cond_ids], torch.zeros(1)
    )
    action_cond_ts = torch.where(
        action_do_cond, scheduler.timesteps[action_cond_ids], torch.zeros(1)
    )
    action_clean_noise = torch.randn_like(actions)
    expected_clean_actions = (
        (1 - action_cond_sigma.reshape(1, 1, 1)) * actions
        + action_cond_sigma.reshape(1, 1, 1) * action_clean_noise
    )
    expected_state = torch.random.get_rng_state().clone()

    assert torch.equal(capture["latents"], expected_video)
    assert torch.equal(capture["ar_clean_latents"], expected_clean_video)
    assert torch.equal(capture["noisy_actions"], expected_actions)
    assert torch.equal(capture["ar_clean_actions"], expected_clean_actions)
    assert torch.equal(
        capture["ar_video_frame_timesteps"], scheduler.timesteps[video_ids]
    )
    assert torch.equal(
        capture["ar_clean_video_frame_timesteps"],
        video_cond_ts.reshape(1, 1).expand(1, 4),
    )
    assert torch.equal(
        capture["ar_action_token_timesteps"], scheduler.timesteps[action_ids]
    )
    assert torch.equal(
        capture["ar_clean_action_token_timesteps"],
        action_cond_ts.reshape(1, 1).expand(1, 4),
    )
    assert torch.equal(observed_state, expected_state)
