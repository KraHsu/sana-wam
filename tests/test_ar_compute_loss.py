"""DualSystemARArchitecture.compute_loss — finite loss + gradient flow.

Builds a mini SANA video backbone + ActionDiT + AR driver, runs one AR
flow-matching loss step over a tiny batch, and asserts: the loss dict is finite,
and backward produces finite gradients on BOTH the video and action backbones
(i.e. the duplicated-sequence forward is differentiable end-to-end). CPU-only;
gated on third_party/Sana.
"""

from __future__ import annotations

import pytest
import torch


def _sana_importable() -> bool:
    try:
        import diffusion.model.nets.sana_multi_scale_video  # noqa: F401
    except Exception:
        return False
    return True


requires_sana = pytest.mark.skipif(
    not _sana_importable(), reason="third_party/Sana not importable"
)


def test_prepare_inputs_rejects_mixed_action_mask_presence():
    from sana_wam.model.base import BaseWAMArchitecture

    class FakeArchitecture:
        dtype = torch.float32
        device = torch.device("cpu")
        uses_proprioception = False
        _use_gradient_checkpointing = False
        _use_gradient_checkpointing_offload = False
        _max_timestep_boundary = 1.0
        _min_timestep_boundary = 0.0

        def preprocess(self, **kwargs):
            del kwargs
            return {}

    batch = [
        {
            "video": [object()],
            "prompt": "first",
            "action": torch.zeros(4, 2),
            "action_mask": torch.ones(4, dtype=torch.bool),
        },
        {
            "video": [object()],
            "prompt": "second",
            "action": torch.zeros(4, 2),
        },
    ]

    with pytest.raises(ValueError, match="Mixed action masks"):
        BaseWAMArchitecture.prepare_inputs(FakeArchitecture(), batch)


def _build_arch():
    from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
    from sana_wam.model.architecture import DualSystemARArchitecture
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone

    vb = SanaVideoBackbone.from_mini_config(
        depth=2, hidden_size=128, num_heads=4, linear_head_dim=32, h=8, w=8
    )
    ab = ActionDiT(
        action_dim=8,
        dim=64,
        ffn_dim=128,
        num_heads=vb.num_heads,
        num_layers=vb.num_layers,
        video_dim=vb.dim,
        bridge_layers=tuple(range(vb.num_layers)),
        variant="joint_self_attn",
        attn_head_dim=vb.head_dim,
        text_dim=vb.context_dim,
        attn_kernel="linear_relu",
        loss_weighting="none",
    )
    arch = DualSystemARArchitecture(cfg=None)
    arch.video_backbone = vb
    arch.action_backbone = ab
    arch._ar_frame_chunk_size = 1
    arch._ar_attn_window = 72
    arch._ar_noisy_cond_prob = 0.5
    arch._ar_cond_max_ratio = 0.3
    arch._mot_driver_kwargs = {
        "attention_mask_mode": "joint",
        "video_attention_mask_mode": "first_frame_causal",
        "mot_checkpoint_mixed_attn": False,
    }
    arch.build_mot_driver()
    # The trainer calls this to set the top-level device/dtype authority
    # (compute_loss reads self.device/self.dtype). Mini backbones are CPU/fp32.
    arch.set_dtype_device(torch.float32, torch.device("cpu"))
    vb.scheduler.set_timesteps(num_inference_steps=1000, training=True)
    ab.scheduler.set_timesteps(
        num_inference_steps=1000,
        training=True,
        loss_weighting=ab.loss_weighting,
    )
    return arch, vb, ab


@requires_sana
def test_ar_compute_loss_finite_and_grads():
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()

    B, T = 2, 2  # 2 latent frames, frame_chunk_size=1 -> 2 chunks
    Ta = 4  # 2 action tokens per chunk
    in_ch = vb._dit.in_channels
    Hl = Wl = 8
    L = 4

    batch = dict(
        input_latents=torch.randn(B, in_ch, T, Hl, Wl),
        actions=torch.randn(B, Ta, ab.action_dim),
        context=torch.randn(B, L, vb.context_dim),
        seq_lens=torch.full((B,), L, dtype=torch.long),
    )

    out = arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)
    assert torch.isfinite(out["loss"]).all()
    assert torch.isfinite(out["loss_video"]).all()
    assert torch.isfinite(out["loss_action"]).all()
    assert out["loss"].item() > 0

    out["loss"].backward()

    def _has_finite_grad(module) -> bool:
        found = False
        for p in module.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    return False
                found = True
        return found

    assert _has_finite_grad(vb._dit), "no finite grad reached the video backbone"
    assert _has_finite_grad(ab), "no finite grad reached the action backbone"


@requires_sana
def test_trajectory_endpoint_loss_updates_video_but_not_frozen_action():
    torch.manual_seed(11)
    arch, vb, ab = _build_arch()
    arch._ar_chunkwise_temporal_ops = True
    arch._configure_ar_chunkwise_temporal_ops()
    arch._video_on_path_loss_weight = 0.0
    arch._video_trajectory_endpoint_weight = 1.0
    arch._video_trajectory_consistency_weight = 0.25
    arch._video_trajectory_steps = 1
    arch._video_trajectory_min_sigma = 0.5
    arch._video_trajectory_max_sigma = 0.5
    for parameter in ab.parameters():
        parameter.requires_grad_(False)

    batch = dict(
        input_latents=torch.randn(1, vb._dit.in_channels, 2, 8, 8),
        actions=torch.randn(1, 4, ab.action_dim),
        context=torch.randn(1, 4, vb.context_dim),
        seq_lens=torch.full((1,), 4, dtype=torch.long),
    )
    out = arch.compute_loss(lambda_video=1.0, lambda_action=0.0, **batch)

    assert torch.isfinite(out["loss"])
    assert float(out["loss_video_on_path"]) == 0.0
    assert float(out["loss_action"]) == 0.0
    assert float(out["loss_video_trajectory_endpoint"]) > 0.0
    assert float(out["loss_video_trajectory_consistency"]) >= 0.0
    assert int(out["video_trajectory_supervised_states"]) == 1
    assert int(out["video_trajectory_supervised_state_index"]) == 0
    out["loss"].backward()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in vb._dit.parameters()
    )
    assert all(parameter.grad is None for parameter in ab.parameters())


@requires_sana
def test_trajectory_samples_generated_state_with_on_path_anchor():
    torch.manual_seed(17)
    arch, vb, ab = _build_arch()
    arch._ar_chunkwise_temporal_ops = True
    arch._configure_ar_chunkwise_temporal_ops()
    arch._video_on_path_loss_weight = 1.0
    arch._video_trajectory_endpoint_weight = 0.0
    arch._video_trajectory_velocity_weight = 1.0
    arch._video_trajectory_consistency_weight = 0.25
    arch._video_trajectory_steps = 3
    arch._video_trajectory_schedule_mode = "inference"
    arch._video_trajectory_supervision_mode = "random"
    arch._video_trajectory_min_sigma = 0.5
    arch._video_trajectory_max_sigma = 0.5
    for parameter in ab.parameters():
        parameter.requires_grad_(False)

    batch = dict(
        input_latents=torch.randn(1, vb._dit.in_channels, 2, 8, 8),
        actions=torch.randn(1, 4, ab.action_dim),
        context=torch.randn(1, 4, vb.context_dim),
        seq_lens=torch.full((1,), 4, dtype=torch.long),
    )
    out = arch.compute_loss(lambda_video=1.0, lambda_action=0.0, **batch)

    assert torch.isfinite(out["loss"])
    assert float(out["loss_video_on_path"]) > 0.0
    assert float(out["loss_video_trajectory_endpoint"]) > 0.0
    assert torch.isfinite(out["loss_video_trajectory_velocity"])
    assert float(out["loss_video_trajectory_velocity"]) > 0.0
    assert float(out["loss_video_trajectory_consistency"]) > 0.0
    assert int(out["video_trajectory_supervised_states"]) == 1
    assert int(out["video_trajectory_supervised_state_index"]) in {0, 1}
    assert float(out["loss_action"]) == 0.0
    out["loss"].backward()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in vb._dit.parameters()
    )
    assert all(parameter.grad is None for parameter in ab.parameters())


@requires_sana
def test_ar_compute_loss_rejects_action_mask_shape_mismatch():
    arch, vb, ab = _build_arch()
    batch = dict(
        input_latents=torch.randn(1, vb._dit.in_channels, 2, 8, 8),
        actions=torch.randn(1, 4, ab.action_dim),
        action_is_pad=torch.zeros(1, 3, dtype=torch.bool),
        context=torch.randn(1, 4, vb.context_dim),
        seq_lens=torch.full((1,), 4, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="must exactly match"):
        arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)


@requires_sana
def test_ar_compute_loss_rejects_sample_without_action_labels():
    arch, vb, ab = _build_arch()
    batch = dict(
        input_latents=torch.randn(1, vb._dit.in_channels, 2, 8, 8),
        actions=torch.randn(1, 4, ab.action_dim),
        action_is_pad=torch.ones(1, 4, dtype=torch.bool),
        context=torch.randn(1, 4, vb.context_dim),
        seq_lens=torch.full((1,), 4, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="at least one supervised action token"):
        arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)


@requires_sana
def test_ar_compute_loss_action_loss_weighting_wiring():
    """The low-noise mode must change AR action loss from the flat default."""
    B, T, Ta, L = 2, 2, 4, 4

    def _run(weighting):
        torch.manual_seed(7)
        arch, vb, ab = _build_arch()
        ab.loss_weighting = weighting
        ab.scheduler.set_timesteps(1000, training=True, loss_weighting=weighting)
        torch.manual_seed(42)
        batch = dict(
            input_latents=torch.randn(B, vb._dit.in_channels, T, 8, 8),
            actions=torch.randn(B, Ta, ab.action_dim),
            context=torch.randn(B, L, vb.context_dim),
            seq_lens=torch.full((B,), L, dtype=torch.long),
        )
        with torch.no_grad():
            return arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)[
                "loss_action"
            ].item()

    assert _run("none") != pytest.approx(_run("low_noise"), rel=1e-6)


@requires_sana
def test_ar_compute_loss_clean_copy_is_deterministic_at_zero_prob():
    """With noisy_cond_prob=0 the clean copy is exactly t=0 (no cond noise)."""
    torch.manual_seed(1)
    arch, vb, ab = _build_arch()
    arch._ar_noisy_cond_prob = 0.0
    clean = torch.randn(2, vb._dit.in_channels, 2, 8, 8)
    copy, ts = arch._make_clean_copy(
        clean, vb.scheduler, 2, 2, clean.device, clean.dtype, frames=True
    )
    torch.testing.assert_close(copy, clean)
    assert torch.count_nonzero(ts) == 0


@requires_sana
@pytest.mark.parametrize(
    "trajectory_only",
    [False, True],
)
def test_ar_compute_loss_propagates_checkpoint_controls(
    monkeypatch, trajectory_only
):
    arch, vb, ab = _build_arch()
    observed = []

    def fake_forward(noisy_actions, _action_timestep, **kwargs):
        observed.append(
            (
                kwargs["use_gradient_checkpointing"],
                kwargs["use_gradient_checkpointing_offload"],
            )
        )
        return torch.zeros_like(kwargs["latents"]), torch.zeros_like(noisy_actions)

    monkeypatch.setattr(arch, "forward", fake_forward)
    if trajectory_only:
        arch._ar_chunkwise_temporal_ops = True
        arch._configure_ar_chunkwise_temporal_ops()
        arch._video_on_path_loss_weight = 0.0
        arch._video_trajectory_endpoint_weight = 1.0
        arch._video_trajectory_steps = 1
        arch._video_trajectory_min_sigma = 0.5
        arch._video_trajectory_max_sigma = 0.5

    batch = dict(
        input_latents=torch.randn(1, vb._dit.in_channels, 2, 8, 8),
        actions=torch.randn(1, 4, ab.action_dim),
        context=torch.randn(1, 4, vb.context_dim),
        seq_lens=torch.full((1,), 4, dtype=torch.long),
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=True,
    )
    arch.compute_loss(
        lambda_video=1.0,
        lambda_action=0.0 if trajectory_only else 1.0,
        **batch,
    )

    expected = [(False, False), (True, True)] if trajectory_only else [(True, True)]
    assert observed == expected


@requires_sana
@pytest.mark.parametrize(
    "field",
    ["use_gradient_checkpointing", "use_gradient_checkpointing_offload"],
)
def test_ar_compute_loss_rejects_non_boolean_checkpoint_controls(field):
    arch, vb, ab = _build_arch()
    batch = dict(
        input_latents=torch.randn(1, vb._dit.in_channels, 2, 8, 8),
        actions=torch.randn(1, 4, ab.action_dim),
        context=torch.randn(1, 4, vb.context_dim),
        seq_lens=torch.full((1,), 4, dtype=torch.long),
        **{field: "true"},
    )

    with pytest.raises(TypeError, match=field):
        arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)


@requires_sana
def test_ar_compute_loss_rejects_offload_without_checkpointing():
    arch, vb, ab = _build_arch()
    batch = dict(
        input_latents=torch.randn(1, vb._dit.in_channels, 2, 8, 8),
        actions=torch.randn(1, 4, ab.action_dim),
        context=torch.randn(1, 4, vb.context_dim),
        seq_lens=torch.full((1,), 4, dtype=torch.long),
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=True,
    )

    with pytest.raises(ValueError, match="requires use_gradient_checkpointing"):
        arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)
