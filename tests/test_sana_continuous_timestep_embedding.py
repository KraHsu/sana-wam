"""CPU mini-SANA tests for the continuous video timestep embedding boundary."""

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


def _mini_backbone(*, continuous: bool):
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone

    return SanaVideoBackbone.from_mini_config(
        depth=1,
        hidden_size=128,
        num_heads=4,
        linear_head_dim=32,
        h=8,
        w=8,
        continuous_timestep_conditioning=continuous,
    )


@requires_sana
def test_t1_reaches_sample_and_frame_embedders_as_fractional_fp32():
    torch.manual_seed(0)
    backbone = _mini_backbone(continuous=True)
    assert backbone.continuous_timestep_conditioning is True

    seen: list[torch.Tensor] = []
    embedder = backbone._dit.t_embedder
    original_forward = embedder.forward

    def capture(timestep):
        seen.append(timestep.detach().clone())
        return original_forward(timestep)

    embedder.forward = capture
    try:
        backbone.prepare(
            latents=torch.randn(1, backbone._dit.in_channels, 2, 8, 8),
            timestep=torch.tensor([321.25], dtype=torch.float32),
            frame_timesteps=torch.tensor(
                [[100.125, 100.875]], dtype=torch.float32
            ),
            context=torch.randn(1, 4, backbone.context_dim),
            seq_lens=torch.full((1,), 4, dtype=torch.long),
        )
    finally:
        embedder.forward = original_forward

    assert len(seen) == 2
    assert all(value.dtype == torch.float32 for value in seen)
    torch.testing.assert_close(
        seen[0], torch.tensor([321.25]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        seen[1], torch.tensor([100.125, 100.875]), atol=0, rtol=0
    )


@requires_sana
def test_t1_rejects_a_prequantized_video_timestep():
    backbone = _mini_backbone(continuous=True)
    with pytest.raises(TypeError, match="requires float32 video timesteps"):
        backbone.prepare(
            latents=torch.randn(1, backbone._dit.in_channels, 2, 8, 8),
            timestep=torch.tensor([321.25], dtype=torch.bfloat16),
            frame_timesteps=torch.tensor(
                [[100.125, 100.875]], dtype=torch.float32
            ),
            context=torch.randn(1, 4, backbone.context_dim),
            seq_lens=torch.full((1,), 4, dtype=torch.long),
        )


@requires_sana
def test_t1_flag_adds_no_checkpoint_parameters():
    torch.manual_seed(9)
    t0 = _mini_backbone(continuous=False)
    torch.manual_seed(9)
    t1 = _mini_backbone(continuous=True)

    assert t0.continuous_timestep_conditioning is False
    assert t1.continuous_timestep_conditioning is True
    t0_state = t0.state_dict()
    t1_state = t1.state_dict()
    assert tuple(t0_state) == tuple(t1_state)
    for key in t0_state:
        torch.testing.assert_close(t0_state[key], t1_state[key], atol=0, rtol=0)
