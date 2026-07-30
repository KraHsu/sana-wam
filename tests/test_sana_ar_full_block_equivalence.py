"""Full SANA video block must match full-layout and cache execution."""

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


@requires_sana
def test_chunkwise_full_video_block_matches_empty_cache_bootstrap():
    from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
    from sana_wam.model.architecture import DualSystemARArchitecture
    from sana_wam.model.ar.sana_ar_inference import ARLinearStateCache
    from sana_wam.model.video_backbone.sana import SanaVideoBackbone

    torch.manual_seed(0)
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
    # Upstream zero-initializes these residuals. Make both inconsistent
    # temporal paths observable so this test cannot pass by degeneracy.
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
    architecture._mot_driver_kwargs = {
        "attention_mask_mode": "joint",
        "video_attention_mask_mode": "first_frame_causal",
        "mot_checkpoint_mixed_attn": False,
    }
    driver = architecture.build_mot_driver()
    architecture.eval()

    batch, frames, action_tokens = 1, 4, 2
    channels = video._dit.in_channels
    noisy_video = torch.randn(batch, channels, frames, 8, 8)
    clean_video = torch.randn_like(noisy_video)
    noisy_action = torch.randn(batch, action_tokens, action.action_dim)
    clean_action = torch.randn_like(noisy_action)
    context = torch.randn(batch, 4, video.context_dim)
    context_mask = torch.ones(batch, 4, dtype=torch.bool)
    video_timesteps = torch.full((batch, frames), 500.0)
    action_timesteps = torch.full((batch, action_tokens), 500.0)

    def run_pair():
        with torch.inference_mode():
            full, _ = architecture.forward(
                noisy_action,
                None,
                latents=noisy_video,
                ar_clean_latents=clean_video,
                ar_clean_actions=clean_action,
                ar_video_frame_timesteps=video_timesteps,
                ar_action_token_timesteps=action_timesteps,
                ar_frame_chunk_size=2,
                ar_attn_window=72,
                context=context,
                context_mask=context_mask,
            )
            state = video.prepare(
                latents=noisy_video[:, :, :2],
                timestep=torch.full((batch,), 500.0),
                frame_timesteps=video_timesteps[:, :2],
                rope_frame_index=torch.arange(2),
                context=context,
                context_mask=context_mask,
            )
            cache = ARLinearStateCache(num_layers=video.num_layers, window=72)
            driver.run_ar_chunk_through_backbone(
                video, state, cache, frame_id=0, store_clean=False
            )
            isolated = video.finalize(state)
        return full[:, :, :2], isolated

    unchunked_full, unchunked_isolated = run_pair()
    unchunked_delta = (unchunked_full - unchunked_isolated).abs().max()
    assert float(unchunked_delta) > 1e-3

    architecture._ar_chunkwise_temporal_ops = True
    architecture._configure_ar_chunkwise_temporal_ops()
    assert video.ar_temporal_chunk_frames == 2
    chunked_full, chunked_isolated = run_pair()
    torch.testing.assert_close(
        chunked_full, chunked_isolated, atol=3e-6, rtol=1e-5
    )
