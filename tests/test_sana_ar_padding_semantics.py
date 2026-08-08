from __future__ import annotations

from fnmatch import fnmatchcase
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from sana_wam.model.ar.sana_ar_linear_attn import (
    _ar_chunked_linear_attn,
    _ar_expanded_reference,
    ar_build_dense_mask,
    build_ar_seq_meta,
)
from sana_wam.model.architecture import DualSystemARArchitecture
from sana_wam.model.action_non_regression import unweighted_pad_masked_action_mse
from sana_wam.model.action_facing_cache_consistency import (
    action_facing_cache_consistency_loss,
)
from sana_wam.model.video_backbone.sana.blocks_split import (
    _validate_masked_window_partition,
)
from sana_wam.train.libero_contract import (
    LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS,
)


def _attention_inputs(*, batch: int, heads: int, tokens: int, dim: int):
    generator = torch.Generator().manual_seed(1701)
    tilde_q = torch.randn(
        batch, heads, tokens, dim, generator=generator, dtype=torch.float64
    )
    tilde_k = torch.randn(
        batch, heads, tokens, dim, generator=generator, dtype=torch.float64
    )
    value = torch.randn(
        batch, heads, tokens, dim, generator=generator, dtype=torch.float64
    )
    phi_q = torch.rand(
        batch, heads, tokens, dim, generator=generator, dtype=torch.float64
    ) + 0.25
    phi_k = torch.rand(
        batch, heads, tokens, dim, generator=generator, dtype=torch.float64
    ) + 0.25
    return tilde_q, tilde_k, value, phi_q, phi_k


def test_masked_fast_path_matches_batch_dense_oracle():
    num_chunks = 3
    video_tokens_per_chunk = 2
    action_tokens_per_chunk = 3
    tokens = 2 * num_chunks * (
        video_tokens_per_chunk + action_tokens_per_chunk
    )
    key_is_pad = torch.zeros(2, tokens, dtype=torch.bool)
    key_is_pad[0, [5, 11, 20, 29]] = True
    key_is_pad[1, [3, 8, 14, 17, 24]] = True
    meta = build_ar_seq_meta(
        num_chunks=num_chunks,
        video_tokens_per_chunk=video_tokens_per_chunk,
        action_tokens_per_chunk=action_tokens_per_chunk,
        window=5,
        key_is_pad=key_is_pad,
    )
    assert meta.key_is_pad_validated is True
    inputs = list(_attention_inputs(batch=2, heads=2, tokens=tokens, dim=4))
    pad = key_is_pad[:, None, :, None]
    for tensor_index in (1, 2, 4):  # tilde_k, value, phi_k
        inputs[tensor_index] = inputs[tensor_index].masked_fill(pad, float("nan"))

    dense_mask = ar_build_dense_mask(meta)
    assert dense_mask.shape == (2, 1, tokens, tokens)
    expected = _ar_expanded_reference(*inputs, meta, eps=1.0e-12)
    actual = _ar_chunked_linear_attn(*inputs, meta, eps=1.0e-12)
    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1.0e-10, rtol=1.0e-10)


def test_prevalidated_fast_path_does_not_rescan_mask_values(monkeypatch):
    key_is_pad = torch.tensor([[False, True, False, True]])
    meta = build_ar_seq_meta(
        num_chunks=1,
        video_tokens_per_chunk=1,
        action_tokens_per_chunk=1,
        window=2,
        key_is_pad=key_is_pad,
    )
    inputs = _attention_inputs(batch=1, heads=1, tokens=4, dim=2)
    tensor_all = torch.Tensor.all

    def reject_hot_path_rescan(value, *args, **kwargs):
        if value is meta.key_is_pad:
            raise AssertionError("prevalidated key mask was rescanned")
        return tensor_all(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "all", reject_hot_path_rescan)
    output = _ar_chunked_linear_attn(*inputs, meta)
    assert torch.isfinite(output).all()


class _RecordingAdapter:
    def __init__(self):
        self.inputs = []

    def forward_delta(self, layer_id, state, normalizer):
        self.inputs.append((layer_id, state.detach().clone(), normalizer.detach().clone()))
        return 0.25 * state, 0.125 * normalizer


@pytest.mark.parametrize("with_adapter", [False, True])
def test_padded_key_values_cannot_change_valid_queries(with_adapter):
    # Layout for one chunk: [v_noisy(2), v_clean(2), a_noisy(4), a_clean(4)].
    tokens = 12
    padded = torch.zeros(1, tokens, dtype=torch.bool)
    padded[:, [1, 3, 7, 11]] = True
    meta = build_ar_seq_meta(
        num_chunks=1,
        video_tokens_per_chunk=2,
        action_tokens_per_chunk=4,
        window=4,
        key_is_pad=padded,
    )
    inputs = list(_attention_inputs(batch=1, heads=2, tokens=tokens, dim=3))
    changed = [value.clone() for value in inputs]
    for tensor_index in (1, 2, 4):  # tilde_k, value, phi_k
        changed[tensor_index][:, :, padded[0], :] = float("nan")

    adapter_a = _RecordingAdapter() if with_adapter else None
    adapter_b = _RecordingAdapter() if with_adapter else None
    baseline = _ar_chunked_linear_attn(
        *inputs,
        meta,
        eps=1.0e-12,
        action_video_memory_adapter=adapter_a,
        layer_id=0 if with_adapter else None,
    )
    perturbed = _ar_chunked_linear_attn(
        *changed,
        meta,
        eps=1.0e-12,
        action_video_memory_adapter=adapter_b,
        layer_id=0 if with_adapter else None,
    )
    valid_queries = ~padded[0]
    assert torch.isfinite(perturbed[:, :, valid_queries]).all()
    torch.testing.assert_close(
        baseline[:, :, valid_queries],
        perturbed[:, :, valid_queries],
        atol=0,
        rtol=0,
    )
    if with_adapter:
        assert len(adapter_a.inputs) == len(adapter_b.inputs) == 1
        assert torch.isfinite(adapter_b.inputs[0][1]).all()
        assert torch.isfinite(adapter_b.inputs[0][2]).all()
        torch.testing.assert_close(adapter_a.inputs[0][1], adapter_b.inputs[0][1])
        torch.testing.assert_close(adapter_a.inputs[0][2], adapter_b.inputs[0][2])

    # Prove the fixture exercises real same-block visibility rather than a
    # causally unreachable suffix.
    unmasked = build_ar_seq_meta(
        num_chunks=1,
        video_tokens_per_chunk=2,
        action_tokens_per_chunk=4,
        window=4,
    )
    before = _ar_chunked_linear_attn(*inputs, unmasked, eps=1.0e-12)
    after = _ar_chunked_linear_attn(*changed, unmasked, eps=1.0e-12)
    assert not torch.equal(
        before[:, :, valid_queries], after[:, :, valid_queries]
    )


@pytest.mark.parametrize(
    ("mask", "error", "message"),
    [
        (torch.ones(1, 8, dtype=torch.bool), ValueError, "non-padded"),
        (torch.zeros(1, 7, dtype=torch.bool), ValueError, "tokens=8"),
        (torch.zeros(1, 8), TypeError, "boolean"),
    ],
)
def test_ar_meta_padding_mask_fails_closed(mask, error, message):
    with pytest.raises(error, match=message):
        build_ar_seq_meta(
            num_chunks=1,
            video_tokens_per_chunk=2,
            action_tokens_per_chunk=2,
            window=2,
            key_is_pad=mask,
        )


class _Scheduler:
    def __init__(self):
        self.sigmas = torch.tensor([0.5], dtype=torch.float32)
        self.timesteps = torch.tensor([500.0], dtype=torch.float32)
        self.linear_timesteps_weights = torch.ones(1, dtype=torch.float32)
        self.num_train_timesteps = 1000
        self.flow_shift = 3.0

    def training_weight(self, indices):
        return torch.ones_like(indices, dtype=torch.float32)


class _Backbone(nn.Module):
    loss_weighting = "none"
    continuous_timestep_conditioning = False

    def __init__(self):
        super().__init__()
        self.scheduler = _Scheduler()
        self.anchor = nn.Parameter(torch.zeros(()))


def _loss_architecture():
    architecture = DualSystemARArchitecture(cfg=None)
    architecture.video_backbone = _Backbone()
    architecture.action_backbone = _Backbone()
    architecture._device = torch.device("cpu")
    architecture._dtype = torch.float32
    architecture._ar_frame_chunk_size = 2
    architecture._ar_attn_window = 8
    architecture._ar_noisy_cond_prob = 0.0
    architecture._ar_cond_max_ratio = 0.3
    architecture._ar_bootstrap_clean_prefix = True
    architecture._ar_chunkwise_temporal_ops = False
    architecture._proprio_per_chunk = False
    architecture._video_on_path_loss_weight = 1.0
    architecture._video_trajectory_endpoint_weight = 0.0
    architecture._video_trajectory_velocity_weight = 0.0
    architecture._video_trajectory_consistency_weight = 0.0
    architecture._video_local_expansion_weight = 0.0
    architecture._action_non_regression_weight = 0.0
    architecture.action_video_memory_adapter = None
    architecture.proprio_video_embed = None
    architecture.proprio_action_embed = None
    architecture.forward_masks = []

    def fake_forward(self, noisy_actions, action_timestep, **inputs):
        del action_timestep
        self.forward_masks.append(
            (inputs["ar_video_is_pad"].clone(), inputs["ar_action_is_pad"].clone())
        )
        # sigma=0.5 on all supervised frames, so target = 2*(noisy-clean).
        target = 2.0 * (inputs["latents"] - inputs["ar_clean_latents"])
        frame_error = torch.tensor([0.0, 1.0, 2.0, 100.0]).view(1, 1, 4, 1, 1)
        video_prediction = target + frame_error + self.video_backbone.anchor
        action_prediction = (
            torch.zeros_like(noisy_actions) + self.action_backbone.anchor
        )
        return video_prediction, action_prediction

    architecture.forward = MethodType(fake_forward, architecture)
    return architecture


def _loss_batch():
    return {
        "input_latents": torch.zeros(1, 1, 4, 1, 1),
        "actions": torch.zeros(1, 4, 2),
        "video_is_pad": torch.tensor([[False, False, False, True]]),
        "action_is_pad": torch.tensor([[False, False, True, True]]),
    }


def _padding_plan_row():
    return {
        "action_sigma": 1.0,
        "cycle": 0,
        "domain_seeds": {
            "video-noise": 11,
            "action-noise": 12,
            "expansion-noise": 13,
            "expansion-direction": 14,
            "reference-query": 15,
            "prompt-choice": 16,
        },
        "global_step": 1,
        "identity": {
            "dataset_index": 0,
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
        "position_in_cycle": 0,
    }


def test_on_path_video_loss_uses_only_nonbootstrap_nonpadded_frames():
    architecture = _loss_architecture()
    output = architecture.compute_loss(
        lambda_video=1.0, lambda_action=0.0, **_loss_batch()
    )
    # Errors 1 and 2 are the only supervised frames: (1^2 + 2^2) / 2.
    torch.testing.assert_close(output["loss_video_on_path"], torch.tensor(2.5))
    assert len(architecture.forward_masks) == 1
    assert torch.equal(
        architecture.forward_masks[0][0],
        torch.tensor([[False, False, False, True]]),
    )
    assert torch.equal(
        architecture.forward_masks[0][1],
        torch.tensor([[False, False, True, True]]),
    )


@pytest.mark.parametrize("lambda_video", [0.0, 1.0])
def test_nonphase_action_only_allows_no_supervised_video_target(lambda_video):
    architecture = _loss_architecture()
    architecture._video_on_path_loss_weight = 0.0
    batch = _loss_batch()
    batch["video_is_pad"] = torch.tensor([[False, True, True, True]])

    torch.manual_seed(303)
    output = architecture.compute_loss(
        lambda_video=lambda_video, lambda_action=1.0, **batch
    )
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["loss_action"])
    assert output["loss_video"].item() == 0.0
    torch.testing.assert_close(output["loss"], output["loss_action"])
    assert len(architecture.forward_masks) == 1


@pytest.mark.parametrize(
    "weight_name",
    [
        "_video_on_path_loss_weight",
        "_video_trajectory_endpoint_weight",
        "_video_trajectory_velocity_weight",
        "_video_trajectory_consistency_weight",
        "_video_local_expansion_weight",
    ],
)
def test_nonphase_enabled_video_component_requires_video_target(weight_name):
    architecture = _loss_architecture()
    architecture._video_on_path_loss_weight = 0.0
    setattr(architecture, weight_name, 1.0)
    if weight_name == "_video_local_expansion_weight":
        architecture._ar_chunkwise_temporal_ops = True
    batch = _loss_batch()
    batch["video_is_pad"] = torch.tensor([[False, True, True, True]])

    with pytest.raises(ValueError, match="video target"):
        architecture.compute_loss(
            lambda_video=0.0, lambda_action=1.0, **batch
        )
    assert architecture.forward_masks == []


def test_phase6_requires_video_target_even_when_all_video_weights_are_zero():
    architecture = _loss_architecture()
    architecture._video_on_path_loss_weight = 0.0
    batch = _loss_batch()
    batch["video_is_pad"] = torch.tensor([[False, True, True, True]])
    batch["phase6_plan_rows"] = (_padding_plan_row(),)

    with pytest.raises(ValueError, match="video target"):
        architecture.compute_loss(
            lambda_video=0.0, lambda_action=0.0, **batch
        )
    assert architecture.forward_masks == []


def test_padded_nan_video_and_action_do_not_poison_losses():
    baseline_batch = _loss_batch()
    changed_batch = _loss_batch()
    changed_batch["input_latents"][:, :, 3] = float("nan")
    changed_batch["actions"][:, 2:] = float("nan")

    torch.manual_seed(404)
    baseline_architecture = _loss_architecture()
    baseline = baseline_architecture.compute_loss(
        lambda_video=1.0, lambda_action=1.0, **baseline_batch
    )
    torch.manual_seed(404)
    changed_architecture = _loss_architecture()
    changed = changed_architecture.compute_loss(
        lambda_video=1.0, lambda_action=1.0, **changed_batch
    )

    for name in ("loss", "loss_video_on_path", "loss_action"):
        assert torch.isfinite(changed[name])
        torch.testing.assert_close(changed[name], baseline[name], atol=0, rtol=0)
    changed["loss"].backward()
    parameter_grads = [
        parameter.grad
        for parameter in changed_architecture.parameters()
        if parameter.grad is not None
    ]
    assert parameter_grads
    assert all(torch.isfinite(gradient).all() for gradient in parameter_grads)


def test_action_non_regression_reduction_sanitizes_padded_nan():
    prediction = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    target = torch.zeros_like(prediction)
    action_is_pad = torch.tensor([[False, False, True, True]])
    baseline = unweighted_pad_masked_action_mse(
        prediction, target, action_is_pad
    )
    prediction[:, 2:] = float("nan")
    target[:, 2:] = float("nan")
    prediction.requires_grad_()
    changed = unweighted_pad_masked_action_mse(
        prediction, target, action_is_pad
    )
    assert torch.isfinite(changed).all()
    torch.testing.assert_close(changed, baseline, atol=0, rtol=0)
    changed.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[:, 2:]) == 0


def test_trajectory_losses_thread_masks_and_ignore_padded_nan():
    def configured_architecture():
        architecture = _loss_architecture()
        architecture._video_on_path_loss_weight = 0.0
        architecture._video_trajectory_endpoint_weight = 1.0
        architecture._video_trajectory_velocity_weight = 1.0
        architecture._video_trajectory_consistency_weight = 1.0
        architecture._video_trajectory_steps = 2
        architecture._video_trajectory_schedule_mode = "inference"
        architecture._video_trajectory_supervision_mode = "final"
        return architecture

    baseline_batch = _loss_batch()
    changed_batch = _loss_batch()
    changed_batch["input_latents"][:, :, 3] = float("nan")
    changed_batch["actions"][:, 2:] = float("nan")

    torch.manual_seed(505)
    baseline_architecture = configured_architecture()
    baseline = baseline_architecture.compute_loss(
        lambda_video=1.0, lambda_action=0.0, **baseline_batch
    )
    torch.manual_seed(505)
    changed_architecture = configured_architecture()
    changed = changed_architecture.compute_loss(
        lambda_video=1.0, lambda_action=0.0, **changed_batch
    )

    for name in (
        "loss",
        "loss_video_trajectory_endpoint",
        "loss_video_trajectory_velocity",
        "loss_video_trajectory_consistency",
    ):
        assert torch.isfinite(changed[name])
        torch.testing.assert_close(changed[name], baseline[name], atol=0, rtol=0)
    changed["loss"].backward()
    assert torch.isfinite(changed_architecture.video_backbone.anchor.grad)
    assert len(changed_architecture.forward_masks) == 3
    for observed_video_mask, observed_action_mask in changed_architecture.forward_masks:
        assert torch.equal(observed_video_mask, changed_batch["video_is_pad"])
        assert torch.equal(observed_action_mask, changed_batch["action_is_pad"])


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        (
            "video_is_pad",
            torch.tensor([[False, True, True, True]]),
            ValueError,
            "video target",
        ),
        (
            "video_is_pad",
            torch.ones(1, 4, dtype=torch.bool),
            ValueError,
            "video target",
        ),
        (
            "action_is_pad",
            torch.ones(1, 4, dtype=torch.bool),
            ValueError,
            "action token",
        ),
        (
            "video_is_pad",
            torch.tensor([[False, True, False, True]]),
            ValueError,
            "padded suffix",
        ),
        (
            "action_is_pad",
            torch.zeros(1, 4),
            TypeError,
            "boolean",
        ),
        (
            "video_is_pad",
            torch.zeros(1, 4),
            TypeError,
            "boolean",
        ),
        (
            "action_is_pad",
            torch.zeros(1, 3, dtype=torch.bool),
            ValueError,
            "exactly match",
        ),
        (
            "video_is_pad",
            torch.zeros(1, 3, dtype=torch.bool),
            ValueError,
            "exactly match",
        ),
    ],
)
def test_compute_loss_padding_contract_fails_closed(
    field, value, error, message
):
    batch = _loss_batch()
    batch[field] = value
    with pytest.raises(error, match=message):
        _loss_architecture().compute_loss(
            lambda_video=1.0, lambda_action=0.0, **batch
        )


class _ForwardVideoBackbone:
    def prepare(self, *, latents, **kwargs):
        del kwargs
        batch, _, frames, _, _ = latents.shape
        return SimpleNamespace(
            h=2,
            w=3,
            x=torch.zeros(batch, frames * 6, 4),
            latents=latents,
        )

    def finalize(self, state):
        return state.latents


class _ForwardActionBackbone:
    def prepare_state(self, actions, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(payload=SimpleNamespace(x_action=actions))

    def extract_prediction(self, state):
        return state.payload.x_action


class _CaptureDriver:
    def __init__(self):
        self.meta = None

    def run_ar_joint_loop(self, video, action, *, ar_meta, **kwargs):
        del video, action, kwargs
        self.meta = ar_meta


def test_forward_expands_frame_masks_in_exact_duplicated_layout():
    architecture = DualSystemARArchitecture(cfg=None)
    architecture.video_backbone = _ForwardVideoBackbone()
    architecture.action_backbone = _ForwardActionBackbone()
    architecture._mot_driver = _CaptureDriver()
    architecture.proprio_video_embed = None
    architecture.proprio_action_embed = None

    batch, frames, actions = 2, 4, 8
    video_is_pad = torch.tensor(
        [[False, False, True, True], [False, False, False, True]]
    )
    action_is_pad = torch.tensor(
        [
            [False, False, False, True, True, True, True, True],
            [False, False, False, False, False, True, True, True],
        ]
    )
    architecture.forward(
        torch.zeros(batch, actions, 2),
        None,
        latents=torch.zeros(batch, 1, frames, 1, 1),
        ar_clean_latents=torch.zeros(batch, 1, frames, 1, 1),
        ar_clean_actions=torch.zeros(batch, actions, 2),
        ar_video_frame_timesteps=torch.zeros(batch, frames),
        ar_clean_video_frame_timesteps=torch.zeros(batch, frames),
        ar_action_token_timesteps=torch.zeros(batch, actions),
        ar_clean_action_token_timesteps=torch.zeros(batch, actions),
        ar_video_is_pad=video_is_pad,
        ar_action_is_pad=action_is_pad,
        ar_frame_chunk_size=2,
        ar_attn_window=8,
    )

    video_tokens = video_is_pad.repeat_interleave(6, dim=1)
    expected = torch.cat(
        [video_tokens, video_tokens, action_is_pad, action_is_pad], dim=1
    )
    assert torch.equal(architecture._mot_driver.meta.key_is_pad, expected)


def test_frozen_window_partition_keeps_each_chunk_frame_isolated():
    safe = SimpleNamespace(temporal_window_count=4)
    _validate_masked_window_partition(safe, temporal_chunk_frames=2)

    with pytest.raises(ValueError, match="one latent frame"):
        _validate_masked_window_partition(
            SimpleNamespace(temporal_window_count=1),
            temporal_chunk_frames=2,
        )
    with pytest.raises(TypeError, match="positive integer"):
        _validate_masked_window_partition(
            SimpleNamespace(temporal_window_count=None),
            temporal_chunk_frames=2,
        )


def test_glumb_temporal_conv_masks_after_spatial_bias_and_zeros_pad_output():
    from diffusion.model.nets.basic_modules import GLUMBConvTemp

    torch.manual_seed(911)
    module = GLUMBConvTemp(
        in_features=4,
        hidden_features=8,
        use_bias=(True, True, False),
        norm=(None, None, None),
        act=("silu", "silu", None),
        t_kernel_size=3,
    )
    torch.nn.init.normal_(module.t_conv.weight, std=0.2)
    value = torch.randn(1, 2, 4)
    changed = value.clone()
    changed[:, 1] = float("nan")
    frame_is_pad = torch.tensor([[False, True]])

    baseline = module(value, HW=(2, 1, 1), frame_is_pad=frame_is_pad)
    perturbed = module(changed, HW=(2, 1, 1), frame_is_pad=frame_is_pad)
    assert torch.isfinite(perturbed).all()
    torch.testing.assert_close(baseline[:, :1], perturbed[:, :1], atol=0, rtol=0)
    assert torch.count_nonzero(baseline[:, 1:]) == 0
    assert torch.count_nonzero(perturbed[:, 1:]) == 0

    unmasked_baseline = module(value, HW=(2, 1, 1))
    unmasked_perturbed = module(changed, HW=(2, 1, 1))
    assert not torch.equal(unmasked_baseline[:, :1], unmasked_perturbed[:, :1])


def _mini_full_architecture():
    # Import the real mini SANA graph without initializing Triton's CUDA-only
    # benchmark driver; this test is intentionally CPU-only.
    from diffusion.utils import import_utils

    def _triton_unavailable():
        return False

    import_utils.is_triton_module_available = _triton_unavailable
    from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone

    video = SanaVideoBackbone.from_mini_config(
        depth=2,
        hidden_size=128,
        num_heads=4,
        linear_head_dim=32,
        h=8,
        w=8,
        additional_flash_attn="window_flash",
        flash_attn_window_count=[1, 1, 4],
    )
    for block in video._dit.blocks:
        torch.nn.init.normal_(block.flash_attn_additional.proj.weight, std=0.02)
        torch.nn.init.normal_(block.flash_attn_additional.proj.bias, std=0.02)
        torch.nn.init.normal_(block.mlp.t_conv.weight, std=0.02)
    action = ActionDiT(
        action_dim=8,
        dim=64,
        ffn_dim=128,
        num_heads=video.num_heads,
        num_layers=video.num_layers,
        video_dim=video.dim,
        bridge_layers=tuple(range(video.num_layers)),
        variant="joint_self_attn",
        attn_head_dim=video.head_dim,
        text_dim=video.context_dim,
        attn_kernel="linear_relu",
    )
    architecture = DualSystemARArchitecture(cfg=None)
    architecture.video_backbone = video
    architecture.action_backbone = action
    architecture._ar_frame_chunk_size = 2
    architecture._ar_chunkwise_temporal_ops = True
    architecture._mot_driver_kwargs = {
        "attention_mask_mode": "joint",
        "video_attention_mask_mode": "first_frame_causal",
        "mot_checkpoint_mixed_attn": False,
    }
    architecture.build_mot_driver()
    architecture.eval()
    return architecture, video, action


def test_full_ar_model_valid_outputs_are_invariant_to_pad_values():
    torch.manual_seed(1201)
    architecture, video, action = _mini_full_architecture()
    batch, frames, action_tokens = 1, 4, 8
    noisy_video = torch.randn(batch, video._dit.in_channels, frames, 8, 8)
    clean_video = torch.randn_like(noisy_video)
    noisy_action = torch.randn(batch, action_tokens, action.action_dim)
    clean_action = torch.randn_like(noisy_action)
    changed_noisy_video = noisy_video.clone()
    changed_clean_video = clean_video.clone()
    changed_noisy_action = noisy_action.clone()
    changed_clean_action = clean_action.clone()
    changed_noisy_video[:, :, 3] = float("nan")
    changed_clean_video[:, :, 3] = float("nan")
    changed_noisy_action[:, 6:] = float("nan")
    changed_clean_action[:, 6:] = float("nan")

    video_is_pad = torch.tensor([[False, False, False, True]])
    action_is_pad = torch.tensor(
        [[False, False, False, False, False, False, True, True]]
    )
    context = torch.randn(batch, 4, video.context_dim)
    context_mask = torch.ones(batch, 4, dtype=torch.bool)
    video_timesteps = torch.full((batch, frames), 500.0)
    action_timesteps = torch.full((batch, action_tokens), 500.0)

    def run(nv, cv, na, ca, *, masked):
        mask_args = (
            {
                "ar_video_is_pad": video_is_pad,
                "ar_action_is_pad": action_is_pad,
            }
            if masked
            else {}
        )
        with torch.inference_mode():
            return architecture.forward(
                na,
                None,
                latents=nv,
                ar_clean_latents=cv,
                ar_clean_actions=ca,
                ar_video_frame_timesteps=video_timesteps,
                ar_clean_video_frame_timesteps=torch.zeros_like(video_timesteps),
                ar_action_token_timesteps=action_timesteps,
                ar_clean_action_token_timesteps=torch.zeros_like(action_timesteps),
                ar_frame_chunk_size=2,
                ar_attn_window=72,
                context=context,
                context_mask=context_mask,
                **mask_args,
            )

    baseline_video, baseline_action = run(
        noisy_video, clean_video, noisy_action, clean_action, masked=True
    )
    changed_video, changed_action = run(
        changed_noisy_video,
        changed_clean_video,
        changed_noisy_action,
        changed_clean_action,
        masked=True,
    )
    assert torch.isfinite(changed_video[:, :, :3]).all()
    assert torch.isfinite(changed_action[:, :6]).all()
    torch.testing.assert_close(
        baseline_video[:, :, :3], changed_video[:, :, :3], atol=0, rtol=0
    )
    torch.testing.assert_close(
        baseline_action[:, :6], changed_action[:, :6], atol=0, rtol=0
    )

    unmasked_video, unmasked_action = run(
        noisy_video, clean_video, noisy_action, clean_action, masked=False
    )
    changed_unmasked_video, changed_unmasked_action = run(
        changed_noisy_video,
        changed_clean_video,
        changed_noisy_action,
        changed_clean_action,
        masked=False,
    )
    assert not torch.equal(
        unmasked_video[:, :, :3], changed_unmasked_video[:, :, :3]
    )
    assert not torch.equal(
        unmasked_action[:, :6], changed_unmasked_action[:, :6]
    )


def test_checkpoint_recompute_backward_is_finite_with_nan_padding():
    torch.manual_seed(2203)
    architecture, video, action = _mini_full_architecture()
    architecture.eval()
    batch, frames, action_tokens = 1, 4, 8

    noisy_video = torch.randn(batch, video._dit.in_channels, frames, 8, 8)
    clean_video = torch.randn_like(noisy_video)
    noisy_action = torch.randn(batch, action_tokens, action.action_dim)
    clean_action = torch.randn_like(noisy_action)
    noisy_video[:, :, 3] = float("nan")
    clean_video[:, :, 3] = float("nan")
    noisy_action[:, 6:] = float("nan")
    clean_action[:, 6:] = float("nan")
    leaves = [
        value.requires_grad_()
        for value in (noisy_video, clean_video, noisy_action, clean_action)
    ]
    noisy_video, clean_video, noisy_action, clean_action = leaves

    video_is_pad = torch.tensor([[False, False, False, True]])
    action_is_pad = torch.tensor(
        [[False, False, False, False, False, False, True, True]]
    )
    context = torch.randn(batch, 4, video.context_dim)
    context_mask = torch.ones(batch, 4, dtype=torch.bool)
    video_timesteps = torch.full((batch, frames), 500.0)
    action_timesteps = torch.full((batch, action_tokens), 500.0)

    video_prediction, action_prediction = architecture.forward(
        noisy_action,
        None,
        latents=noisy_video,
        ar_clean_latents=clean_video,
        ar_clean_actions=clean_action,
        ar_video_frame_timesteps=video_timesteps,
        ar_clean_video_frame_timesteps=torch.zeros_like(video_timesteps),
        ar_action_token_timesteps=action_timesteps,
        ar_clean_action_token_timesteps=torch.zeros_like(action_timesteps),
        ar_video_is_pad=video_is_pad,
        ar_action_is_pad=action_is_pad,
        ar_frame_chunk_size=2,
        ar_attn_window=72,
        context=context,
        context_mask=context_mask,
        use_gradient_checkpointing=True,
    )
    loss = (
        video_prediction[:, :, :3].float().square().mean()
        + action_prediction[:, :6].float().square().mean()
    )
    assert torch.isfinite(loss)
    loss.backward()

    for value in leaves:
        if value.grad is not None:
            assert torch.isfinite(value.grad).all()
    assert torch.count_nonzero(noisy_video.grad[:, :, 3]) == 0
    assert torch.count_nonzero(clean_video.grad[:, :, 3]) == 0
    assert torch.count_nonzero(noisy_action.grad[:, 6:]) == 0
    assert torch.count_nonzero(clean_action.grad[:, 6:]) == 0


def test_checkpointed_action_video_numerator_capture_is_differentiable_and_exact():
    torch.manual_seed(2607)
    architecture, video, action = _mini_full_architecture()
    architecture.train()
    batch, frames, action_tokens = 1, 4, 8
    noisy_video = torch.randn(batch, video._dit.in_channels, frames, 8, 8)
    clean_video = torch.randn_like(noisy_video)
    noisy_action = torch.randn(batch, action_tokens, action.action_dim)
    clean_action = torch.randn_like(noisy_action)
    video_is_pad = torch.tensor([[False, False, False, True]])
    action_is_pad = torch.tensor(
        [[False, False, False, False, False, False, True, True]]
    )
    context = torch.randn(batch, 4, video.context_dim)
    context_mask = torch.ones(batch, 4, dtype=torch.bool)
    video_timesteps = torch.full((batch, frames), 500.0)
    action_timesteps = torch.full((batch, action_tokens), 500.0)

    kwargs = {
        "latents": noisy_video,
        "ar_clean_latents": clean_video,
        "ar_clean_actions": clean_action,
        "ar_video_frame_timesteps": video_timesteps,
        "ar_clean_video_frame_timesteps": torch.zeros_like(video_timesteps),
        "ar_action_token_timesteps": action_timesteps,
        "ar_clean_action_token_timesteps": torch.zeros_like(action_timesteps),
        "ar_video_is_pad": video_is_pad,
        "ar_action_is_pad": action_is_pad,
        "ar_frame_chunk_size": 2,
        "ar_attn_window": 72,
        "context": context,
        "context_mask": context_mask,
        "use_gradient_checkpointing": True,
    }
    torch.manual_seed(99)
    baseline_video, baseline_action = architecture.forward(
        noisy_action, None, **kwargs
    )
    torch.manual_seed(99)
    captured_video, captured_action, numerators = architecture.forward(
        noisy_action,
        None,
        return_action_video_numerators=True,
        **kwargs,
    )
    assert torch.equal(captured_video, baseline_video)
    assert torch.equal(captured_action, baseline_action)
    assert numerators.shape[:3] == (batch, video.num_layers, 2)
    assert numerators.shape[3:] == (
        video.num_heads,
        action_tokens // 2,
        video.head_dim,
    )
    assert numerators.requires_grad

    reference = numerators.detach().clone().add_(0.125)
    valid = (~action_is_pad).reshape(batch, 2, action_tokens // 2)
    loss = action_facing_cache_consistency_loss(numerators, reference, valid)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    video_gradients = [
        parameter.grad
        for parameter in video.parameters()
        if parameter.grad is not None
    ]
    assert video_gradients
    assert all(torch.isfinite(gradient).all() for gradient in video_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in video_gradients)
    assert reference.grad is None

    parameter_grads = [
        parameter.grad
        for parameter in architecture.parameters()
        if parameter.grad is not None
    ]
    assert parameter_grads
    assert all(torch.isfinite(gradient).all() for gradient in parameter_grads)


def test_r10_qkv_patterns_receive_action_loss_gradients_without_preserve_mode():
    torch.manual_seed(2708)
    architecture, video, action = _mini_full_architecture()
    architecture.train()
    # Mirrors Trainer._set_training_mode with eval_modules=['video_backbone'].
    architecture.video_backbone.eval()
    assert not architecture.video_backbone.training
    assert all(
        not module.training for module in architecture.video_backbone.modules()
    )
    for name, parameter in architecture.named_parameters():
        parameter.requires_grad_(
            any(
                fnmatchcase(name, pattern)
                for pattern in LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS
            )
        )

    named_parameters = dict(architecture.named_parameters())
    qkv_parameters = {
        name: parameter
        for name, parameter in named_parameters.items()
        if fnmatchcase(
            name,
            LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS[-1],
        )
    }
    assert len(qkv_parameters) == video.num_layers == 2
    assert all(parameter.requires_grad for parameter in qkv_parameters.values())
    assert all(
        not parameter.requires_grad
        for name, parameter in named_parameters.items()
        if name.startswith("video_backbone.") and name not in qkv_parameters
    )

    batch, frames, action_tokens = 1, 4, 8
    noisy_video = torch.randn(batch, video._dit.in_channels, frames, 8, 8)
    clean_video = torch.randn_like(noisy_video)
    noisy_action = torch.randn(batch, action_tokens, action.action_dim)
    clean_action = torch.randn_like(noisy_action)
    context = torch.randn(batch, 4, video.context_dim)
    context_mask = torch.ones(batch, 4, dtype=torch.bool)
    video_timesteps = torch.full((batch, frames), 500.0)
    action_timesteps = torch.full((batch, action_tokens), 500.0)

    _, action_prediction = architecture.forward(
        noisy_action,
        None,
        latents=noisy_video,
        ar_clean_latents=clean_video,
        ar_clean_actions=clean_action,
        ar_video_frame_timesteps=video_timesteps,
        ar_clean_video_frame_timesteps=torch.zeros_like(video_timesteps),
        ar_action_token_timesteps=action_timesteps,
        ar_clean_action_token_timesteps=torch.zeros_like(action_timesteps),
        ar_frame_chunk_size=2,
        ar_attn_window=72,
        context=context,
        context_mask=context_mask,
        use_gradient_checkpointing=True,
    )
    target = torch.randn_like(action_prediction)
    action_loss = unweighted_pad_masked_action_mse(
        action_prediction,
        target,
    ).mean()
    assert torch.isfinite(action_loss)
    action_loss.backward()

    for parameter in qkv_parameters.values():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_afcc_checkpoint_on_off_loss_and_video_gradient_match_with_recompute():
    torch.manual_seed(2711)
    off_architecture, off_video, off_action = _mini_full_architecture()
    torch.manual_seed(2711)
    on_architecture, on_video, on_action = _mini_full_architecture()
    off_architecture.train()
    on_architecture.train()
    for off_value, on_value in zip(
        off_architecture.state_dict().values(),
        on_architecture.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(off_value, on_value)

    generator = torch.Generator().manual_seed(2712)
    batch, frames, action_tokens = 1, 4, 8
    noisy_video = torch.randn(
        batch,
        off_video._dit.in_channels,
        frames,
        8,
        8,
        generator=generator,
    )
    clean_video = torch.randn(noisy_video.shape, generator=generator)
    noisy_action = torch.randn(
        batch, action_tokens, off_action.action_dim, generator=generator
    )
    clean_action = torch.randn(noisy_action.shape, generator=generator)
    context = torch.randn(batch, 4, off_video.context_dim, generator=generator)
    context_mask = torch.ones(batch, 4, dtype=torch.bool)
    video_timesteps = torch.full((batch, frames), 500.0)
    action_timesteps = torch.full((batch, action_tokens), 500.0)
    common = {
        "latents": noisy_video,
        "ar_clean_latents": clean_video,
        "ar_clean_actions": clean_action,
        "ar_video_frame_timesteps": video_timesteps,
        "ar_clean_video_frame_timesteps": torch.zeros_like(video_timesteps),
        "ar_action_token_timesteps": action_timesteps,
        "ar_clean_action_token_timesteps": torch.zeros_like(action_timesteps),
        "ar_frame_chunk_size": 2,
        "ar_attn_window": 72,
        "context": context,
        "context_mask": context_mask,
        "return_action_video_numerators": True,
    }

    on_driver = on_architecture._mot_driver
    calls = 0
    original_mixed_attention = on_driver._mixed_attention

    def counted_mixed_attention(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_mixed_attention(*args, **kwargs)

    on_driver._mixed_attention = MethodType(counted_mixed_attention, on_driver)
    torch.manual_seed(2713)
    off_prediction = off_architecture.forward(
        noisy_action,
        None,
        use_gradient_checkpointing=False,
        **common,
    )
    torch.manual_seed(2713)
    on_prediction = on_architecture.forward(
        noisy_action,
        None,
        use_gradient_checkpointing=True,
        **common,
    )
    assert calls == on_video.num_layers
    for off_value, on_value in zip(off_prediction, on_prediction, strict=True):
        assert torch.equal(off_value, on_value)

    reference = off_prediction[2].detach().clone().add_(0.125)
    valid = torch.ones(batch, 2, action_tokens // 2, dtype=torch.bool)
    off_loss = action_facing_cache_consistency_loss(
        off_prediction[2], reference, valid
    )
    on_loss = action_facing_cache_consistency_loss(
        on_prediction[2], reference, valid
    )
    torch.testing.assert_close(on_loss, off_loss, atol=0, rtol=0)
    off_parameter = dict(off_architecture.named_parameters())[
        "video_backbone.dit.blocks.0.attn.qkv.weight"
    ]
    on_parameter = dict(on_architecture.named_parameters())[
        "video_backbone.dit.blocks.0.attn.qkv.weight"
    ]
    off_gradient = torch.autograd.grad(off_loss, off_parameter)[0]
    on_gradient = torch.autograd.grad(on_loss, on_parameter)[0]
    assert calls > on_video.num_layers
    torch.testing.assert_close(on_gradient, off_gradient, atol=1.0e-7, rtol=1.0e-6)
