"""Focused CPU/FP32 checks for the AV-1 transition-proxy screen.

These tests exercise only the frozen synthetic transition proxy.  They do not
create an AV-1 result root, run the registered 200-step screen, initialize CUDA,
read real data, or load/save a checkpoint.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import math

import pytest
import torch
from torch import Tensor

import sana_wam.model.cach_av1_transition_proxy as av1
from sana_wam.model.cach_av1_transition_proxy import (
    AV1TransitionProxySpec,
    build_av1_pair,
    build_av1_synthetic_task,
    run_av1_screen,
)


SHUFFLE_PERMUTATION = (1, 0, 3, 2, 5, 4, 7, 6)
COMMON_EXACT_NAMES = {
    "video_input_projection.weight",
    "video_input_projection.bias",
    "video_timestep_projection.weight",
    "video_output_projection.weight",
    "video_output_projection.bias",
    "proprio_context_encoder.weight",
    "proprio_context_encoder.bias",
}
COMMON_BLOCK_SUFFIXES = {
    "q_projection.weight",
    "k_projection.weight",
    "v_projection.weight",
    "beta_projection.weight",
    "beta_projection.bias",
    "decay_projection.weight",
    "decay_projection.bias",
    "attention_output_projection.weight",
    "attention_output_projection.bias",
    "ffn_input_projection.weight",
    "ffn_input_projection.bias",
    "ffn_output_projection.weight",
    "ffn_output_projection.bias",
}
CANDIDATE_EXACT_NAMES = {
    "action_conditioner.0.weight",
    "action_conditioner.0.bias",
    "action_conditioner.2.weight",
    "action_conditioner.2.bias",
}


@pytest.fixture(scope="module")
def spec() -> AV1TransitionProxySpec:
    assert not torch.cuda.is_initialized()
    value = AV1TransitionProxySpec()
    assert value.batch_size == 8
    assert value.gdn_depth == 20
    assert value.device == "cpu"
    assert value.dtype is torch.float32
    return value


@pytest.fixture(scope="module")
def task(spec: AV1TransitionProxySpec):
    return build_av1_synthetic_task(spec=spec)


@pytest.fixture()
def pair(spec: AV1TransitionProxySpec, task):
    # Pair state is deliberately function-scoped: every test starts at theta0.
    assert task.spec == spec
    return build_av1_pair(spec=spec, task=task)


def _assert_exact(left: Tensor, right: Tensor) -> None:
    assert left.device.type == right.device.type == "cpu"
    assert left.dtype == right.dtype
    assert tuple(left.shape) == tuple(right.shape)
    torch.testing.assert_close(left, right, atol=0, rtol=0)


def _tensor_fingerprint(module: torch.nn.Module) -> tuple[tuple[str, int, bytes], ...]:
    return tuple(
        (
            name,
            tensor._version,
            tensor.detach().contiguous().cpu().numpy().tobytes(),
        )
        for name, tensor in (
            list(module.named_parameters()) + list(module.named_buffers())
        )
    )


def _expected_common_names(depth: int = 20) -> set[str]:
    return COMMON_EXACT_NAMES | {
        f"blocks.{index}.{suffix}"
        for index in range(depth)
        for suffix in COMMON_BLOCK_SUFFIXES
    }


def _expected_candidate_only_names(depth: int = 20) -> set[str]:
    return CANDIDATE_EXACT_NAMES | {
        f"blocks.{index}.action_output_projection.{leaf}"
        for index in range(depth)
        for leaf in ("weight", "bias")
    }


def _assert_finite_nonzero(value: Tensor) -> None:
    assert bool(torch.isfinite(value).all())
    assert bool(value.count_nonzero())


def _effect_rms(task, horizon: int) -> float:
    effect = task.video_target[:, horizon] - task.base_target[:, horizon]
    return math.sqrt(effect.double().square().mean().item())


def _tensor_leaves(value) -> tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        return (value,)
    if is_dataclass(value):
        return tuple(
            tensor
            for field in fields(value)
            for tensor in _tensor_leaves(getattr(value, field.name))
        )
    if isinstance(value, dict):
        return tuple(
            tensor
            for key in sorted(value)
            for tensor in _tensor_leaves(value[key])
        )
    if isinstance(value, (tuple, list)):
        return tuple(tensor for item in value for tensor in _tensor_leaves(item))
    return ()


def _assert_state_exact(left, right) -> None:
    left_tensors = _tensor_leaves(left)
    right_tensors = _tensor_leaves(right)
    assert len(left_tensors) == len(right_tensors)
    assert left_tensors
    for left_tensor, right_tensor in zip(left_tensors, right_tensors):
        _assert_exact(left_tensor, right_tensor)


def test_recipe_pair_bytes_masks_and_layout(
    spec: AV1TransitionProxySpec,
    task,
) -> None:
    assert task.recipe_digest == (
        "9d2a749d722775324ee233862772def530fbffe0ea9e55aeea87c160b87a51e4"
    )
    rebuilt = build_av1_synthetic_task(spec=spec)
    assert rebuilt.recipe_digest == task.recipe_digest
    assert rebuilt.tensor_digests == task.tensor_digests

    float_tensors = {
        "global_actions": (task.global_actions, (8, 32, 20)),
        "noisy_video": (task.noisy_video, (8, 5, 3, 1, 1)),
        "base_target": (task.base_target, (8, 5, 3, 1, 1)),
        "video_target": (task.video_target, (8, 5, 3, 1, 1)),
        "context": (task.context, (8, 2, 2, 4)),
        "proprio": (task.proprio, (8, 2, 20)),
        "video_timestep": (task.video_timestep, (8, 2)),
        "action_timestep": (task.action_timestep, (8, 2)),
    }
    for name, (value, shape) in float_tensors.items():
        assert value.device.type == "cpu", name
        assert value.dtype is torch.float32, name
        assert tuple(value.shape) == shape, name
        assert value.is_contiguous(), name
        assert bool(torch.isfinite(value).all()), name
    assert task.context_mask.dtype is torch.bool
    assert tuple(task.context_mask.shape) == (8, 2, 2)
    assert bool(task.context_mask.all())
    assert task.score_mask.dtype is torch.bool
    assert tuple(task.score_mask.shape) == (8, 4)
    assert bool(task.score_mask.all())

    for first in range(0, 8, 2):
        second = first + 1
        _assert_exact(task.global_actions[first], -task.global_actions[second])
        for value in (
            task.noisy_video,
            task.base_target,
            task.context,
            task.proprio,
            task.video_timestep,
            task.action_timestep,
        ):
            _assert_exact(value[first], value[second])
        for horizon in range(1, 5):
            assert not torch.equal(
                task.video_target[first, horizon],
                task.video_target[second, horizon],
            )

    _assert_exact(task.video_target[:, 0], task.base_target[:, 0])
    for horizon in range(1, 5):
        assert 0.499999 <= _effect_rms(task, horizon) <= 0.500001
        delta = (
            task.video_target[0::2, horizon]
            - task.video_target[1::2, horizon]
        )
        delta_rms = math.sqrt(delta.double().square().mean().item())
        assert 0.999998 <= delta_rms <= 1.000002

    assert len(task.chunks) == 2
    chunk0, chunk1 = task.chunks
    assert tuple(chunk0.actions.shape) == (8, 16, 20)
    assert tuple(chunk1.actions.shape) == (8, 24, 20)
    _assert_exact(chunk0.actions, task.global_actions[:, :16])
    _assert_exact(chunk1.actions[:, :16], task.global_actions[:, 16:32])
    assert not bool(chunk1.actions[:, 16:].count_nonzero())
    assert tuple(chunk0.frame_valid_mask.shape) == (8, 3)
    assert tuple(chunk1.frame_valid_mask.shape) == (8, 3)
    assert bool(chunk0.frame_valid_mask.all())
    assert bool(chunk1.frame_valid_mask[:, :2].all())
    assert not bool(chunk1.frame_valid_mask[:, 2].any())
    assert tuple(chunk0.action_valid_mask.shape) == (8, 16)
    assert tuple(chunk1.action_valid_mask.shape) == (8, 24)
    assert bool(chunk0.action_valid_mask.all())
    assert bool(chunk1.action_valid_mask[:, :16].all())
    assert not bool(chunk1.action_valid_mask[:, 16:].any())
    assert tuple(chunk0.video_latents.shape) == (8, 3, 3, 1, 1)
    assert tuple(chunk1.video_latents.shape) == (8, 3, 3, 1, 1)
    _assert_exact(chunk0.video_latents, task.noisy_video[:, :3])
    _assert_exact(chunk1.video_latents[:, :2], task.noisy_video[:, 3:5])
    assert not bool(chunk1.video_latents[:, 2].count_nonzero())
    assert task.layout.valid_raw_count == 33
    assert task.layout.valid_latent_count == 5
    assert task.layout.valid_action_count == 32
    assert len(task.layout.chunks) == 2


def test_named_theta0_parameter_allowlist_and_frozen_components(
    spec: AV1TransitionProxySpec,
    task,
    pair,
) -> None:
    expected_common = _expected_common_names(spec.gdn_depth)
    expected_candidate_only = _expected_candidate_only_names(spec.gdn_depth)
    assert set(pair.common_trainable_names) == expected_common
    assert set(pair.candidate_only_trainable_names) == expected_candidate_only

    reference_parameters = dict(pair.reference.named_parameters())
    candidate_parameters = dict(pair.candidate.named_parameters())
    reference_trainable = {
        name for name, value in reference_parameters.items() if value.requires_grad
    }
    candidate_trainable = {
        name for name, value in candidate_parameters.items() if value.requires_grad
    }
    assert reference_trainable == expected_common
    assert candidate_trainable == expected_common | expected_candidate_only

    for name in sorted(expected_common):
        _assert_exact(reference_parameters[name], candidate_parameters[name])
        assert reference_parameters[name].untyped_storage().data_ptr() != (
            candidate_parameters[name].untyped_storage().data_ptr()
        )
    for name in sorted(expected_candidate_only):
        assert name not in reference_parameters
        assert name in candidate_parameters

    seam_names = sorted(
        name
        for name in expected_candidate_only
        if ".action_output_projection." in name
    )
    assert len(seam_names) == 40
    assert all(not bool(candidate_parameters[name].detach().count_nonzero()) for name in seam_names)
    assert any(
        bool(candidate_parameters[name].detach().count_nonzero())
        for name in expected_candidate_only
        if name.startswith("action_conditioner.") and name.endswith(".weight")
    )

    candidate_tensors = {
        **candidate_parameters,
        **dict(pair.candidate.named_buffers()),
    }
    no_action = candidate_tensors["cach_core.no_action_slot"]
    assert no_action.dtype is torch.float32
    assert no_action.device.type == "cpu"
    assert not no_action.requires_grad
    assert not bool(no_action.detach().count_nonzero())
    action_backbone_parameters = {
        name: value
        for name, value in candidate_parameters.items()
        if name.startswith("action_backbone.")
    }
    assert action_backbone_parameters
    assert all(not value.requires_grad for value in action_backbone_parameters.values())
    assert pair.reference.action_backbone.training is False
    assert pair.candidate.action_backbone.training is False

    diagnostic = pair.reference.run_sequence(
        task,
        mode="correct",
        run_action_diagnostic=True,
    )
    assert len(diagnostic.action_predictions) == 2
    assert tuple(diagnostic.action_predictions[0].shape) == (8, 16, 20)
    assert tuple(diagnostic.action_predictions[1].shape) == (8, 16, 20)
    assert all(not value.requires_grad for value in diagnostic.action_predictions)

    for model in (pair.reference, pair.candidate):
        for value in tuple(model.parameters()) + tuple(model.buffers()):
            assert value.device.type == "cpu"
            assert value.dtype is torch.float32

    rebuilt = build_av1_pair(spec=spec, task=task)
    assert rebuilt.theta0_manifests == pair.theta0_manifests


def test_theta0_exact_identity_ref_bypass_and_fresh_mode_order(task, pair) -> None:
    reference_before = _tensor_fingerprint(pair.reference)
    candidate_before = _tensor_fingerprint(pair.candidate)
    modes = ("correct", "shuffle", "no_action")
    reference_outputs = {
        mode: pair.reference.run_sequence(
            task,
            mode=mode,
            run_action_diagnostic=False,
        )
        for mode in modes
    }
    candidate_outputs = {
        mode: pair.candidate.run_sequence(
            task,
            mode=mode,
            run_action_diagnostic=False,
        )
        for mode in modes
    }

    baseline = reference_outputs["correct"].video_prediction
    assert tuple(baseline.shape) == (8, 5, 3, 1, 1)
    for mode in modes:
        _assert_exact(reference_outputs[mode].video_prediction, baseline)
        _assert_exact(candidate_outputs[mode].video_prediction, baseline)
    for first in range(0, 8, 2):
        _assert_exact(baseline[first], baseline[first + 1])

    # Every call without an explicit state owns a fresh empty state.  Reverse
    # execution order must reproduce both prediction and recurrent tensors.
    for mode in reversed(modes):
        replay = pair.candidate.run_sequence(
            task,
            mode=mode,
            run_action_diagnostic=False,
        )
        _assert_exact(
            replay.video_prediction,
            candidate_outputs[mode].video_prediction,
        )
        _assert_state_exact(replay.final_state, candidate_outputs[mode].final_state)
        replay_leaves = _tensor_leaves(replay.final_state)
        baseline_leaves = _tensor_leaves(candidate_outputs[mode].final_state)
        assert all(
            left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
            for left, right in zip(replay_leaves, baseline_leaves)
        )

    reference_empty = pair.reference.empty_state(batch_size=8)
    candidate_empty = pair.candidate.empty_state(batch_size=8)
    _assert_state_exact(reference_empty, candidate_empty)
    assert all(not bool(value.count_nonzero()) for value in _tensor_leaves(reference_empty))
    assert _tensor_fingerprint(pair.reference) == reference_before
    assert _tensor_fingerprint(pair.candidate) == candidate_before


def test_candidate_seam_reverse_grad_parameter_jvp_and_target_isolation(
    task,
    pair,
) -> None:
    candidate = pair.candidate
    parameters = dict(candidate.named_parameters())
    seam_weight_names = tuple(
        sorted(
            name
            for name in pair.candidate_only_trainable_names
            if name.endswith("action_output_projection.weight")
        )
    )
    assert len(seam_weight_names) == 20

    result = candidate.run_sequence(
        task,
        mode="correct",
        run_action_diagnostic=False,
    )
    loss = (
        result.video_prediction[:, 1:] - task.video_target[:, 1:]
    ).square().mean()
    seam_gradients = torch.autograd.grad(
        loss,
        tuple(parameters[name] for name in seam_weight_names),
        retain_graph=False,
    )
    assert len(seam_gradients) == 20
    for gradient in seam_gradients:
        assert bool(torch.isfinite(gradient).all())
    seam_grad_rms = math.sqrt(
        sum(gradient.double().square().sum().item() for gradient in seam_gradients)
        / sum(gradient.numel() for gradient in seam_gradients)
    )
    assert seam_grad_rms > 1.0e-8

    base_parameters = dict(candidate.named_parameters())
    primals = tuple(
        base_parameters[name].detach().clone().requires_grad_(True)
        for name in seam_weight_names
    )
    tangents = tuple(
        torch.full_like(value, (index + 1) / 1000.0)
        for index, value in enumerate(primals)
    )

    def seam_direction(*values: Tensor) -> Tensor:
        mapping = dict(base_parameters)
        mapping.update(dict(zip(seam_weight_names, values)))
        output = torch.func.functional_call(
            candidate,
            mapping,
            (task,),
            {"mode": "correct", "run_action_diagnostic": False},
            strict=False,
        )
        return output.video_prediction

    _, parameter_jvp = torch.func.jvp(seam_direction, primals, tangents)
    _assert_finite_nonzero(parameter_jvp)

    target = task.video_target.detach().clone().requires_grad_(True)
    target_result = candidate.run_sequence(
        task,
        mode="correct",
        target_override=target,
        run_action_diagnostic=False,
    )
    _assert_exact(target_result.video_prediction, result.video_prediction)
    target_gradient = torch.autograd.grad(
        target_result.video_prediction.square().sum(),
        target,
        allow_unused=True,
    )[0]
    assert target_gradient is None or not bool(target_gradient.count_nonzero())

    changed_target = target.detach() + 17.0
    changed_result = candidate.run_sequence(
        task,
        mode="correct",
        target_override=changed_target,
        run_action_diagnostic=False,
    )
    _assert_exact(changed_result.video_prediction, target_result.video_prediction)
    _assert_state_exact(changed_result.final_state, target_result.final_state)

    def prediction_from_target(value: Tensor) -> Tensor:
        return candidate.run_sequence(
            task,
            mode="correct",
            target_override=value,
            run_action_diagnostic=False,
        ).video_prediction

    _, target_jvp = torch.autograd.functional.jvp(
        prediction_from_target,
        target.detach(),
        torch.ones_like(target),
        create_graph=False,
        strict=False,
    )
    assert not bool(target_jvp.count_nonzero())


def test_tiny_candidate_update_stays_inside_allowlist_and_keeps_no_action_frozen(
    task,
    pair,
) -> None:
    reference_before = _tensor_fingerprint(pair.reference)
    candidate = pair.candidate
    parameters = dict(candidate.named_parameters())
    before = {name: value.detach().clone() for name, value in parameters.items()}
    trainable_names = set(pair.common_trainable_names) | set(
        pair.candidate_only_trainable_names
    )
    optimizer = torch.optim.AdamW(
        [parameters[name] for name in sorted(trainable_names)],
        lr=0.003,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = candidate.run_sequence(
        task,
        mode="correct",
        run_action_diagnostic=False,
    ).video_prediction
    loss = (prediction[:, 1:] - task.video_target[:, 1:]).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameters[name] for name in sorted(trainable_names)],
        max_norm=1.0,
    )
    optimizer.step()

    changed_names = {
        name
        for name, value in parameters.items()
        if not torch.equal(before[name], value.detach())
    }
    assert changed_names
    assert changed_names <= trainable_names
    changed_seam_weights = {
        name
        for name in changed_names
        if name.endswith("action_output_projection.weight")
    }
    assert changed_seam_weights
    seam_values = [
        (parameters[name].detach() - before[name]).double().flatten()
        for name in sorted(changed_seam_weights)
    ]
    seam_update_rms = torch.cat(seam_values).square().mean().sqrt().item()
    assert seam_update_rms > 1.0e-6

    for name, value in parameters.items():
        if not value.requires_grad:
            _assert_exact(value.detach(), before[name])
    candidate_tensors = {
        **parameters,
        **dict(candidate.named_buffers()),
    }
    no_action = candidate_tensors["cach_core.no_action_slot"]
    assert not no_action.requires_grad
    assert not bool(no_action.detach().count_nonzero())
    assert _tensor_fingerprint(pair.reference) == reference_before

    correct = candidate.run_sequence(
        task,
        mode="correct",
        run_action_diagnostic=False,
    ).video_prediction
    no_action_prediction = candidate.run_sequence(
        task,
        mode="no_action",
        run_action_diagnostic=False,
    ).video_prediction
    assert not torch.equal(correct, no_action_prediction)
    shuffled = candidate.run_sequence(
        task,
        mode="shuffle",
        run_action_diagnostic=False,
    ).video_prediction
    _assert_exact(
        shuffled,
        correct[list(SHUFFLE_PERMUTATION)],
    )
    for first in range(0, 8, 2):
        _assert_exact(
            no_action_prediction[first],
            no_action_prediction[first + 1],
        )


def test_run_av1_screen_one_step_is_in_memory_and_reports_both_arms(
    spec: AV1TransitionProxySpec,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = tuple(tmp_path.iterdir())
    progress_events: list[dict[str, object]] = []
    first_forward_observations: list[tuple[str, str]] = []
    original_builder = av1.build_av1_pair

    def observed_builder(*args, **kwargs):
        observed_pair = original_builder(*args, **kwargs)
        for arm_name in ("reference", "candidate"):
            arm = getattr(observed_pair, arm_name)
            original_run = arm.run_sequence

            def observed_run(*run_args, _arm=arm_name, _run=original_run, **run_kwargs):
                assert progress_events
                assert progress_events[0].get("event") == "theta0_manifests_ready"
                first_forward_observations.append(
                    (_arm, str(progress_events[0]["event"]))
                )
                return _run(*run_args, **run_kwargs)

            monkeypatch.setattr(arm, "run_sequence", observed_run)
        return observed_pair

    monkeypatch.setattr(av1, "build_av1_pair", observed_builder)
    screen = run_av1_screen(
        {"spec": spec, "steps": 1},
        progress_callback=progress_events.append,
    )
    assert tuple(tmp_path.iterdir()) == before
    assert set(screen) == {
        "schema",
        "validity",
        "metrics",
        "loss_trace",
        "theta0_manifests",
        "final_parameter_digests",
        "diagnostics",
        "resource_usage",
    }
    assert screen["schema"] == "cach.av1.transition_proxy.screen.v1"
    completed = screen["diagnostics"]["optimizer_steps_completed_by_arm"]
    assert completed == {"REF-GDN-CORRECTED": 1, "CACH-A": 1}
    assert screen["diagnostics"]["registered_budget"] is False
    assert screen["diagnostics"]["test_only"] is True
    assert screen["diagnostics"]["registered_200_step_budget_executed"] is False
    assert screen["diagnostics"]["device"] == "cpu"
    assert screen["diagnostics"]["dtype"] == "float32"
    assert screen["resource_usage"]["device"] == "cpu"
    assert screen["resource_usage"]["dtype"] == "float32"
    assert screen["resource_usage"]["run_root_created"] is False
    assert screen["resource_usage"]["filesystem_writes"] == 0
    assert screen["validity"]["implementation_valid"] is True
    assert screen["validity"]["data_valid"] is True
    assert screen["validity"]["numerics_valid"] is True
    assert screen["validity"]["finite_all"] is True
    assert screen["validity"]["theta0_exact_identity"] is True
    assert screen["theta0_manifests"]["reference"]
    assert screen["theta0_manifests"]["candidate"]
    theta0_events = [
        event
        for event in progress_events
        if event.get("event") == "theta0_manifests_ready"
    ]
    assert len(theta0_events) == 1
    assert progress_events[0] is theta0_events[0]
    assert theta0_events[0]["payload"] == screen["theta0_manifests"]
    assert {name for name, _event in first_forward_observations} == {
        "reference",
        "candidate",
    }
    assert screen["final_parameter_digests"]["reference"]
    assert screen["final_parameter_digests"]["candidate"]
    assert screen["loss_trace"]["reference"]
    assert screen["loss_trace"]["candidate"]

    metrics = screen["metrics"]
    scalar_metrics = {
        "seam_grad_rms_step0",
        "seam_update_rms_final",
        "loss_drop_rel",
        "counterfactual_delta_nmse_correct",
        "shuffle_gap_rel",
        "no_action_gap_rel",
        "candidate_gain_vs_reference",
    }
    for name in scalar_metrics:
        assert math.isfinite(float(metrics[name])), name
    for name in (
        "counterfactual_delta_nmse_correct_by_horizon",
        "shuffle_gap_rel_by_horizon",
        "no_action_gap_rel_by_horizon",
    ):
        assert {int(key) for key in metrics[name]} == {1, 2, 3, 4}
        assert all(math.isfinite(float(value)) for value in metrics[name].values())


def test_spec_and_in_memory_screen_reject_forbidden_runtime_scope(
    monkeypatch,
) -> None:
    allocations = 0

    def forbidden_builder(*_args, **_kwargs):
        nonlocal allocations
        allocations += 1
        raise AssertionError("model allocated before runtime rejection")

    monkeypatch.setattr(av1, "build_av1_pair", forbidden_builder)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        AV1TransitionProxySpec(device="cuda")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        AV1TransitionProxySpec(dtype=torch.bfloat16)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        run_av1_screen(
            {"steps": 1, "runtime": {"allow_checkpoint": True}}
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        run_av1_screen(
            {"steps": 1, "runtime": {"allow_real_data": True}}
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        run_av1_screen({"steps": 1, "runtime": {"device": "cuda"}})
    assert allocations == 0
