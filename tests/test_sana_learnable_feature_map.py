"""CPU coverage for SANA's optional learnable linear-attention feature map."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _sana_importable() -> bool:
    try:
        import diffusion.model.nets.sana_multi_scale_video  # noqa: F401
    except Exception:
        return False
    return True


requires_sana = pytest.mark.skipif(not _sana_importable(), reason="third_party/Sana not importable")


@requires_sana
def test_learnable_feature_map_is_relu_at_zero_init_and_stays_positive():
    from diffusion.model.nets.sana_blocks import LearnableFeatureMap

    feature_map = LearnableFeatureMap(dim=8)
    x = torch.randn(2, 3, 8, 5)

    torch.testing.assert_close(feature_map(x), torch.relu(x), rtol=0.0, atol=0.0)

    with torch.no_grad():
        feature_map.alpha.fill_(0.75)
    learned = feature_map(x)
    assert torch.all(learned >= 0)
    assert not torch.equal(learned, torch.relu(x))


@requires_sana
def test_fixed_relu_default_keeps_parameter_schema_unchanged():
    from diffusion.model.nets.sana_blocks import LiteLAReLURope

    attention = LiteLAReLURope(32, 32, heads=2, dim=16)

    assert isinstance(attention.kernel_func, nn.ReLU)
    assert not any(key.startswith("kernel_func.") for key in attention.state_dict())


@requires_sana
def test_sana_constructor_threads_learnable_feature_map_to_every_block():
    from diffusion.model.nets.sana_blocks import LearnableFeatureMap
    from diffusion.model.nets.sana_multi_scale_video import SanaMSVideo

    model = SanaMSVideo(
        input_size=4,
        patch_size=(1, 2, 2),
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        pred_sigma=False,
        attn_type="LiteLAReLURope",
        ffn_type="mlp",
        use_pe=True,
        pos_embed_type="wan_rope",
        qk_norm=True,
        cross_norm=True,
        y_norm=True,
        linear_head_dim=16,
        model_max_length=8,
        caption_channels=16,
        linear_feature_map="learnable",
    )

    assert all(isinstance(block.attn.kernel_func, LearnableFeatureMap) for block in model.blocks)
    expected_suffixes = {
        "alpha": (1,),
        "w1.weight": (16, 16),
        "w1.bias": (16,),
        "w2.weight": (16, 16),
        "w2.bias": (16,),
    }
    state = model.state_dict()
    for block_id in range(2):
        for suffix, shape in expected_suffixes.items():
            assert tuple(state[f"blocks.{block_id}.attn.kernel_func.{suffix}"].shape) == shape
