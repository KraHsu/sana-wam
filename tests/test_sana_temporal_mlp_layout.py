"""Layout regression for SANA temporal GLUMBConv variants."""

from __future__ import annotations

import pytest
import torch


def _sana_importable() -> bool:
    try:
        import diffusion.model.nets.basic_modules  # noqa: F401
    except Exception:
        return False
    return True


requires_sana = pytest.mark.skipif(not _sana_importable(), reason="third_party/Sana not importable")


class _CapturedSpatialInput(RuntimeError):
    pass


@requires_sana
@pytest.mark.parametrize(
    ("class_name", "forward_kwargs"),
    [
        ("GLUMBConvTemp", {}),
        ("ChunkGLUMBConvTemp", {"chunk_index": [0]}),
        ("CachedGLUMBConvTemp", {}),
    ],
)
def test_temporal_glumbconv_materializes_contiguous_nchw(monkeypatch, class_name, forward_kwargs):
    from diffusion.model.nets import basic_modules

    cls = getattr(basic_modules, class_name)
    module = cls(
        in_features=4,
        hidden_features=8,
        use_bias=(True, True, False),
        norm=(None, None, None),
        act=("silu", "silu", None),
        t_kernel_size=3,
    )

    def capture_spatial(x):
        assert x.shape == (2, 4, 2, 2)
        assert x.is_contiguous()
        raise _CapturedSpatialInput

    monkeypatch.setattr(module, "_apply_spatial_autochunked", capture_spatial)
    with pytest.raises(_CapturedSpatialInput):
        module(torch.randn(1, 8, 4), HW=(2, 2, 2), **forward_kwargs)
