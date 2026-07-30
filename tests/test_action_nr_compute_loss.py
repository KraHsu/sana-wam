from __future__ import annotations

from types import MethodType

import pytest
import torch
import torch.utils.checkpoint
from torch import nn

from sana_wam.model.architecture import DualSystemARArchitecture


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
        self.scale = nn.Parameter(torch.tensor(0.4))


class _ActionBackbone(nn.Module):
    loss_weighting = "none"

    def __init__(self):
        super().__init__()
        self.scheduler = _Scheduler()
        self.scale = nn.Parameter(torch.tensor(0.7))


class _ScalarAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(()))


def _plan_row(global_step=1):
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


def _architecture(
    *,
    nr_enabled: bool,
    adapter: bool = True,
    expansion_enabled: bool = False,
    checkpointed: bool = False,
):
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
    architecture._video_on_path_loss_weight = 0.0
    architecture._video_trajectory_endpoint_weight = 0.0
    architecture._video_trajectory_velocity_weight = 0.0
    architecture._video_trajectory_consistency_weight = 0.0
    architecture._video_local_expansion_weight = 1.0 if expansion_enabled else 0.0
    architecture._action_non_regression_weight = 1.0 if nr_enabled else 0.0
    architecture.action_video_memory_adapter = _ScalarAdapter() if adapter else None
    architecture.proprio_nr_scale = nn.Parameter(torch.tensor(0.2))
    architecture.nr_forward_count = 0

    def fake_forward(self, noisy_actions, action_timestep, **inputs):  # noqa: ARG001
        self.nr_forward_count += 1

        def field(latents, action_input):
            latent_feature = latents.float().mean(dim=(1, 2, 3, 4)).view(-1, 1, 1)
            action_prediction = (
                self.action_backbone.scale * action_input
                + self.video_backbone.scale * latent_feature
                + self.proprio_nr_scale
            )
            if self.action_video_memory_adapter is not None:
                action_prediction = action_prediction + (
                    self.action_video_memory_adapter.gain * (action_input + 1.0)
                )
            video_prediction = self.video_backbone.scale * latents
            return video_prediction, action_prediction

        if checkpointed:
            return torch.utils.checkpoint.checkpoint(
                field,
                inputs["latents"],
                noisy_actions,
                use_reentrant=False,
            )
        return field(inputs["latents"], noisy_actions)

    architecture.forward = MethodType(fake_forward, architecture)
    return architecture


def _batch(*, reference=None, batch_size=1):
    clean = torch.tensor(
        [0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4, -1.6],
        dtype=torch.float32,
    ).reshape(1, 1, 4, 1, 2).expand(batch_size, -1, -1, -1, -1).clone()
    result = {
        "input_latents": clean,
        "actions": torch.zeros(batch_size, 4, 2),
        "action_is_pad": torch.tensor(
            [[False, False, True, True]], dtype=torch.bool
        ).expand(batch_size, -1).clone(),
        "video_is_pad": torch.tensor(
            [[False, False, False, True]], dtype=torch.bool
        ).expand(batch_size, -1).clone(),
        "phase6_plan_rows": (_plan_row(),),
    }
    if reference is not None:
        result["phase1_action_reference_error"] = reference
    return result


def _phase1_error(seed=123):
    torch.manual_seed(seed)
    architecture = _architecture(nr_enabled=False, adapter=False)
    output = architecture.compute_loss(
        lambda_video=0.0, lambda_action=0.0, **_batch()
    )
    return output["phase6_action_unweighted_mse"].clone()


def test_a0_reports_phase1_mse_without_reference_and_nr_is_exact_zero():
    torch.manual_seed(123)
    architecture = _architecture(nr_enabled=False, adapter=False)
    output = architecture.compute_loss(
        lambda_video=0.0, lambda_action=0.0, **_batch()
    )
    assert output["phase6_action_unweighted_mse"].item() > 0
    assert output["loss_action_non_regression"].item() == 0.0
    surrogate = output["phase6_action_non_regression_surrogate"]
    assert surrogate.item() == 0.0
    assert surrogate.requires_grad is False
    assert output["loss_action"].item() == 0.0
    assert output["loss"].item() == 0.0
    assert output["action_non_regression_active"].item() == 0
    assert architecture.nr_forward_count == 1


def test_identity_reference_has_zero_nr_and_preserves_common_action_mse():
    reference = _phase1_error()
    torch.manual_seed(123)
    architecture = _architecture(nr_enabled=True)
    output = architecture.compute_loss(
        lambda_video=0.0,
        lambda_action=0.0,
        **_batch(reference=reference),
    )
    assert torch.equal(output["phase6_action_unweighted_mse"], reference)
    assert output["loss_action_non_regression"].item() == 0.0
    assert output["loss"].item() == 0.0
    assert output["loss_action"].item() == 0.0


def test_harm_gradient_is_injected_only_into_adapter_with_checkpointing():
    reference = _phase1_error() * 0.5
    torch.manual_seed(123)
    architecture = _architecture(nr_enabled=True, checkpointed=True)
    output = architecture.compute_loss(
        lambda_video=0.0,
        lambda_action=0.0,
        **_batch(reference=reference),
    )
    assert output["loss_action_non_regression"].item() > 0
    surrogate = output["phase6_action_non_regression_surrogate"]
    assert torch.equal(
        surrogate.detach(), output["loss_action_non_regression"]
    )
    gradients = torch.autograd.grad(
        surrogate,
        (
            architecture.action_video_memory_adapter.gain,
            architecture.video_backbone.scale,
            architecture.action_backbone.scale,
            architecture.proprio_nr_scale,
        ),
        allow_unused=True,
        retain_graph=True,
    )
    assert gradients[0] is not None and gradients[0].abs().item() > 0
    assert gradients[1:] == (None, None, None)
    output["loss"].backward()

    adapter_gradient = architecture.action_video_memory_adapter.gain.grad
    assert adapter_gradient is not None and torch.isfinite(adapter_gradient)
    assert adapter_gradient.abs().item() > 0
    assert architecture.video_backbone.scale.grad is None
    assert architecture.action_backbone.scale.grad is None
    assert architecture.proprio_nr_scale.grad is None


def test_improvement_receives_exact_zero_loss_and_zero_adapter_gradient():
    reference = _phase1_error() + 1.0
    torch.manual_seed(123)
    architecture = _architecture(nr_enabled=True)
    output = architecture.compute_loss(
        lambda_video=0.0,
        lambda_action=0.0,
        **_batch(reference=reference),
    )
    assert output["loss_action_non_regression"].item() == 0.0
    assert output["loss"].item() == 0.0
    output["loss"].backward()
    gradient = architecture.action_video_memory_adapter.gain.grad
    assert gradient is not None
    assert gradient.item() == 0.0


def test_action_nr_is_compatible_with_expansion_one_extra_forward():
    reference = _phase1_error() * 0.5
    torch.manual_seed(123)
    architecture = _architecture(nr_enabled=True, expansion_enabled=True)
    output = architecture.compute_loss(
        lambda_video=0.0,
        lambda_action=0.0,
        **_batch(reference=reference),
    )
    assert architecture.nr_forward_count == 2
    assert output["video_local_expansion_extra_forwards"].item() == 1
    assert output["loss_action_non_regression"].item() > 0
    output["loss"].backward()
    assert architecture.action_video_memory_adapter.gain.grad is not None


def test_action_nr_contract_fails_closed():
    reference = torch.tensor(1.0, dtype=torch.float32)
    missing_adapter = _architecture(nr_enabled=True, adapter=False)
    with pytest.raises(RuntimeError, match="requires an enabled"):
        missing_adapter.compute_loss(
            lambda_video=0.0,
            lambda_action=0.0,
            **_batch(reference=reference),
        )

    missing_row = _batch(reference=reference)
    del missing_row["phase6_plan_rows"]
    with pytest.raises(ValueError, match="phase6_plan_rows"):
        _architecture(nr_enabled=True).compute_loss(
            lambda_video=0.0, lambda_action=0.0, **missing_row
        )

    with pytest.raises(ValueError, match="lambda_action=0"):
        _architecture(nr_enabled=True).compute_loss(
            lambda_video=0.0,
            lambda_action=1.0,
            **_batch(reference=reference),
        )

    with pytest.raises(ValueError, match="one FP32 scalar"):
        _architecture(nr_enabled=True).compute_loss(
            lambda_video=0.0,
            lambda_action=0.0,
            **_batch(reference=torch.tensor(1.0, dtype=torch.float64)),
        )


def test_action_nr_weight_is_frozen_to_zero_or_one():
    assert DualSystemARArchitecture._validate_action_non_regression_weight(0) == 0
    assert DualSystemARArchitecture._validate_action_non_regression_weight(1.0) == 1
    for value in (True, 0.5, float("nan"), "1"):
        with pytest.raises(ValueError, match="exactly 0 or 1"):
            DualSystemARArchitecture._validate_action_non_regression_weight(value)
