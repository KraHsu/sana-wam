"""First-principles contract for SANA's aligned positive RoPE kernel."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
from sana_wam.model.ar.sana_ar_linear_attn import (
    _ar_chunked_linear_attn,
    ar_build_dense_mask,
    build_ar_seq_meta,
)
from sana_wam.model.ar.sana_ar_mot_driver import SanaARMoTJointDriver


def _action_dit(*, aligned: bool, mode: str = "post_rope") -> ActionDiT:
    return ActionDiT(
        action_dim=8,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=128,
        bridge_layers=(0, 1),
        variant="joint_self_attn",
        attn_head_dim=16,
        text_dim=32,
        attn_kernel="linear_relu",
        aligned_positive_rope_kernel=aligned,
        aligned_feature_beta=16.0,
        aligned_feature_delta=1e-4,
        aligned_positive_rope_mode=mode,
    )


@pytest.mark.parametrize("mode", ["post_rope", "unrotated", "absolute_rope"])
def test_action_aligned_kernel_publishes_one_strictly_positive_track(mode):
    torch.manual_seed(0)
    dit = _action_dit(aligned=True, mode=mode).eval()
    actions = torch.randn(2, 6, dit.action_dim)
    context = torch.randn(2, 3, dit.text_dim)
    state = dit.prepare_state(
        actions,
        torch.full((2,), 0.5),
        context=context,
        rope_positions=torch.tensor([1, 1, 3, 3, 5, 5]),
    )

    q, k, _, post = dit.pre_attn_at_layer(0, state)

    assert dit.aligned_positive_rope_kernel
    assert dit.aligned_positive_rope_mode == mode
    expected_q = q.abs() if mode == "absolute_rope" else q
    expected_k = k.abs() if mode == "absolute_rope" else k
    assert torch.equal(expected_q, post["q_unrot"])
    assert torch.equal(expected_k, post["k_unrot"])
    floor = 1e-4 if mode == "post_rope" else 0.0
    assert bool((post["q_unrot"] >= floor).all())
    assert bool((post["k_unrot"] >= floor).all())


def test_absolute_rope_normalizer_bounds_signed_attention():
    torch.manual_seed(3)
    meta = build_ar_seq_meta(
        num_chunks=3,
        video_tokens_per_chunk=2,
        action_tokens_per_chunk=1,
        window=3,
    )
    n_tokens = meta.frame_ids.numel()
    q = torch.randn(2, 2, n_tokens, 8)
    k = torch.randn(2, 2, n_tokens, 8)
    values = torch.randn(2, 2, n_tokens, 8)

    output = _ar_chunked_linear_attn(
        q, k, values, q.abs(), k.abs(), meta, eps=0.0
    )
    mask = ar_build_dense_mask(meta)

    assert torch.isfinite(output).all()
    for query in range(n_tokens):
        max_abs_value = values[:, :, mask[query], :].abs().amax(dim=2)
        assert bool((output[:, :, query, :].abs() <= max_abs_value + 1e-5).all())


def test_aligned_ar_kernel_is_a_convex_value_average():
    torch.manual_seed(1)
    meta = build_ar_seq_meta(
        num_chunks=3,
        video_tokens_per_chunk=2,
        action_tokens_per_chunk=1,
        window=3,
    )
    n_tokens = meta.frame_ids.numel()
    q = F.softplus(torch.randn(2, 2, n_tokens, 8)) + 1e-4
    k = F.softplus(torch.randn(2, 2, n_tokens, 8)) + 1e-4
    values = torch.randn(2, 2, n_tokens, 8)

    output = _ar_chunked_linear_attn(q, k, values, q, k, meta, eps=0.0)
    mask = ar_build_dense_mask(meta)

    assert torch.isfinite(output).all()
    for query in range(n_tokens):
        visible = values[:, :, mask[query], :]
        lower = visible.amin(dim=2)
        upper = visible.amax(dim=2)
        assert bool((output[:, :, query, :] >= lower - 1e-5).all())
        assert bool((output[:, :, query, :] <= upper + 1e-5).all())


class _FakeBackbone(nn.Module):
    def __init__(self, *, aligned: bool):
        super().__init__()
        self.dim = 16
        self.num_heads = 2
        self.head_dim = 8
        self.num_layers = 2
        self.aligned_positive_rope_kernel = aligned

    @property
    def attn_kernel(self) -> str:
        return "linear_relu"

    video_attention_mask_mode = "bidirectional"


def test_driver_rejects_cross_stream_kernel_misalignment():
    with pytest.raises(ValueError, match="enabled on both video and action"):
        SanaARMoTJointDriver(
            _FakeBackbone(aligned=True),
            _FakeBackbone(aligned=False),
            attention_mask_mode="joint",
        )


@pytest.mark.parametrize("mode", ["post_rope", "unrotated", "absolute_rope"])
def test_video_split_publishes_the_expected_denominator_track_without_new_keys(
    mode,
):
    pytest.importorskip("diffusion.model.nets.sana_multi_scale_video")
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone
    from sana_wam.model.video_backbone.sana.strictly_positive_feature_map import (
        StrictlyPositiveLearnableFeatureMap,
    )

    torch.manual_seed(2)
    backbone = SanaVideoBackbone.from_mini_config(
        depth=2,
        hidden_size=128,
        num_heads=4,
        linear_head_dim=32,
        h=8,
        w=8,
    )
    for block in backbone._dit.blocks:
        block.attn.kernel_func = StrictlyPositiveLearnableFeatureMap(
            backbone.head_dim, beta=16.0, delta=1e-4
        )
    keys_before = set(backbone.state_dict())

    assert backbone.configure_aligned_positive_rope_kernel(
        beta=16.0, delta=1e-4, mode=mode
    ) == backbone.num_layers
    assert set(backbone.state_dict()) == keys_before

    batch, frames, height, width = 1, 2, 8, 8
    state = backbone.prepare(
        latents=torch.randn(batch, backbone._dit.in_channels, frames, height, width),
        timestep=torch.full((batch,), 500.0),
        context=torch.randn(batch, 3, backbone.context_dim),
        seq_lens=torch.full((batch,), 3, dtype=torch.long),
    )
    q, k, _, post = backbone.pre_attn_at_layer(0, state)

    assert backbone.aligned_positive_rope_kernel
    assert backbone.aligned_positive_rope_mode == mode
    expected_q = q.abs() if mode == "absolute_rope" else q
    expected_k = k.abs() if mode == "absolute_rope" else k
    assert torch.equal(expected_q, post["q_unrot"])
    assert torch.equal(expected_k, post["k_unrot"])
    floor = 0.0 if mode == "absolute_rope" else 1e-4
    assert bool((post["q_unrot"] >= floor).all())
    assert bool((post["k_unrot"] >= floor).all())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"attn_kernel": "softmax", "aligned_positive_rope_kernel": True},
            "requires attn_kernel='linear_relu'",
        ),
        (
            {
                "attn_kernel": "linear_relu",
                "aligned_positive_rope_kernel": True,
                "aligned_feature_delta": 0.0,
            },
            "aligned_feature_delta must be positive",
        ),
    ],
)
def test_invalid_action_alignment_config_fails_closed(kwargs, message):
    base = dict(
        action_dim=8,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=1,
        video_dim=64,
        bridge_layers=(0,),
        variant="joint_self_attn",
        attn_head_dim=16,
    )
    with pytest.raises(ValueError, match=message):
        ActionDiT(**base, **kwargs)
