from __future__ import annotations

import math

import pytest
import torch

from sana_wam.model.local_expansion import (
    current_video_chunk_mask,
    fixed_50_step_shifted_sigmas,
    local_reverse_euler_expansion_loss,
    masked_video_rms,
    nearest_reverse_delta_sigma,
    phase6_expansion_spec,
    quantized_one_sided_chord,
    rademacher_orthogonal_direction,
)


def _video(values, *, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype).reshape(1, 1, 2, 1, 2)


def _plan_row(global_step=1):
    zero_based = global_step - 1
    return {
        "action_sigma": (1.0, 0.9, 0.5)[zero_based % 3],
        "cycle": zero_based // 42,
        "domain_seeds": {
            "video-noise": 1,
            "action-noise": 2,
            "expansion-noise": 3,
            "expansion-direction": 4,
            "reference-query": 5,
            "prompt-choice": 6,
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


def test_quantized_chord_uses_actual_model_visible_delta():
    center = _video([1.0, -1.0, 0.5, -0.5])
    direction = _video([1.0, 1.0, -1.0, -1.0])
    mask = torch.tensor([[True, False]])

    center_visible, plus_visible, chord = quantized_one_sided_chord(
        center,
        direction,
        mask,
        epsilon_rms=0.01,
        model_dtype=torch.bfloat16,
    )

    assert center_visible.dtype == plus_visible.dtype == torch.bfloat16
    assert chord.dtype == torch.float32
    torch.testing.assert_close(
        chord, plus_visible.float() - center_visible.float(), atol=0, rtol=0
    )
    assert torch.count_nonzero(chord[:, :, 1]) == 0
    assert masked_video_rms(chord, mask).item() > 0


def test_quantized_chord_detaches_center_direction_and_mask():
    center = _video([0.1, -0.2, 0.3, -0.4]).requires_grad_(True)
    direction = _video([1.0, -1.0, 1.0, -1.0]).requires_grad_(True)
    mask = torch.ones(1, 2, requires_grad=True)
    center_visible, plus_visible, chord = quantized_one_sided_chord(
        center,
        direction,
        mask,
        epsilon_rms=0.01,
        model_dtype=torch.float32,
    )
    assert not center_visible.requires_grad
    assert not plus_visible.requires_grad
    assert not chord.requires_grad


def test_quantized_chord_fails_closed_when_bf16_collapses_it():
    center = torch.ones(1, 1, 1, 1, 1)
    direction = torch.ones_like(center)
    mask = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="collapsed"):
        quantized_one_sided_chord(
            center,
            direction,
            mask,
            epsilon_rms=1e-6,
            model_dtype=torch.bfloat16,
        )


def test_linear_negative_slope_matches_reverse_euler_expansion_formula():
    chord = _video([1.0, -2.0, 3.0, -4.0])
    mask = torch.ones(1, 2, dtype=torch.bool)
    slope = -2.0
    center = torch.zeros_like(chord)
    plus = slope * chord
    result = local_reverse_euler_expansion_loss(
        chord=chord,
        velocity_center=center,
        velocity_plus=plus,
        mask=mask,
        delta_sigma=-0.1,
        maximum_unpenalized_rate=1.0,
    )

    expected_expansion = 1.2
    expected_rate = math.log(expected_expansion) / 0.1
    torch.testing.assert_close(
        result.expansion, torch.tensor([expected_expansion]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        result.rate, torch.tensor([expected_rate]), atol=1e-6, rtol=0
    )
    assert result.loss.item() > 0


def test_contracting_field_has_zero_one_sided_penalty():
    chord = _video([1.0, -2.0, 3.0, -4.0])
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = local_reverse_euler_expansion_loss(
        chord=chord,
        velocity_center=torch.zeros_like(chord),
        velocity_plus=2.0 * chord,
        mask=mask,
        delta_sigma=-0.1,
        maximum_unpenalized_rate=1.0,
    )
    torch.testing.assert_close(result.expansion, torch.tensor([0.8]))
    assert result.loss.item() == 0.0


def test_expansion_loss_backpropagates_only_through_velocities():
    chord = _video([1.0, -2.0, 3.0, -4.0]).requires_grad_(True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    parameter = torch.tensor(-10.0, requires_grad=True)
    center = torch.zeros_like(chord)
    plus = parameter * chord.detach()
    result = local_reverse_euler_expansion_loss(
        chord=chord,
        velocity_center=center,
        velocity_plus=plus,
        mask=mask,
        delta_sigma=-0.1,
        maximum_unpenalized_rate=1.0,
    )
    result.loss.backward()

    assert parameter.grad is not None and torch.isfinite(parameter.grad)
    assert parameter.grad.abs().item() > 0
    assert chord.grad is None


def test_expansion_loss_detaches_mask_and_delta_sigma():
    chord = _video([1.0, -2.0, 3.0, -4.0])
    mask = torch.ones(1, 2, requires_grad=True)
    delta_sigma = torch.tensor(-0.1, requires_grad=True)
    parameter = torch.tensor(-10.0, requires_grad=True)
    result = local_reverse_euler_expansion_loss(
        chord=chord,
        velocity_center=torch.zeros_like(chord),
        velocity_plus=parameter * chord,
        mask=mask,
        delta_sigma=delta_sigma,
        maximum_unpenalized_rate=1.0,
    )
    result.loss.backward()
    assert parameter.grad is not None and parameter.grad.abs() > 0
    assert mask.grad is None
    assert delta_sigma.grad is None


def test_rademacher_direction_is_masked_normalized_and_orthogonal():
    tangent = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 2, 2, 4)
    mask = torch.tensor([[True, False]])
    orth = rademacher_orthogonal_direction(
        tangent,
        mask,
        generator=torch.Generator().manual_seed(13),
    )
    tangent_visible = tangent.clone()
    tangent_visible[:, :, 1] = 0
    inner = (orth * tangent_visible).sum()
    assert abs(inner.item()) < 1e-5
    torch.testing.assert_close(masked_video_rms(orth, mask), torch.ones(1))
    assert torch.count_nonzero(orth[:, :, 1]) == 0


def test_nearest_reverse_delta_sigma_uses_fixed_descending_grid():
    sigma = torch.tensor([0.92, 0.51, 0.05])
    actual = nearest_reverse_delta_sigma(sigma, [1.0, 0.9, 0.5, 0.0])
    torch.testing.assert_close(actual, torch.tensor([-0.4, -0.5, -0.5]))


def test_fixed_50_step_grid_is_shifted_descending_and_includes_zero():
    schedule = fixed_50_step_shifted_sigmas(3.0, device=torch.device("cpu"))
    assert schedule.shape == (51,)
    assert schedule[0].item() == 1.0
    assert schedule[-1].item() == 0.0
    assert bool((schedule[1:] < schedule[:-1]).all())


def test_current_chunk_mask_excludes_bootstrap_and_padding():
    mask = current_video_chunk_mask(
        torch.tensor([[False, False, False, True]]),
        batch_size=1,
        num_frames=4,
        frame_chunk_size=2,
        bootstrap_clean_prefix=True,
        device=torch.device("cpu"),
    )
    assert torch.equal(mask, torch.tensor([[False, False, True, False]]))


def test_phase6_row_drives_fixed_epsilon_direction_and_seeds():
    first = phase6_expansion_spec((_plan_row(1),), batch_size=1)
    second = phase6_expansion_spec((_plan_row(2),), batch_size=1)
    third = phase6_expansion_spec((_plan_row(3),), batch_size=1)
    assert first.epsilon_rms == 0.005
    assert second.epsilon_rms == 0.01
    assert third.epsilon_rms == 0.02
    assert first.direction == "flow_target_tangent"
    assert second.direction == "deterministic_rademacher_orthogonal"
    assert first.expansion_noise_seed == 3
    assert first.expansion_direction_seed == 4


def test_phase6_row_validation_fails_closed():
    with pytest.raises(ValueError, match="batch size exactly 1"):
        phase6_expansion_spec((_plan_row(1),), batch_size=2)
    partial = _plan_row(1)
    del partial["domain_seeds"]["expansion-noise"]
    with pytest.raises(ValueError, match="seed domains"):
        phase6_expansion_spec((partial,), batch_size=1)
    inconsistent = _plan_row(1)
    inconsistent["global_step"] = 2
    with pytest.raises(ValueError, match="inconsistent"):
        phase6_expansion_spec((inconsistent,), batch_size=1)
    bad_hash = _plan_row(1)
    bad_hash["plan_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA256"):
        phase6_expansion_spec((bad_hash,), batch_size=1)


def test_nearest_reverse_delta_sigma_rejects_non_descending_schedule():
    with pytest.raises(ValueError, match="strictly descending"):
        nearest_reverse_delta_sigma(torch.tensor([0.5]), [1.0, 0.7, 0.7, 0.0])
