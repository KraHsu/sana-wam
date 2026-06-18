"""F3: chunk-0 clean-prefix bootstrap in training.

With ar_bootstrap_clean_prefix=True, compute_loss supplies latent frame 0 clean
(real init obs, t=0) and excludes it from the video loss. Verifies the loss still
runs finite + differentiable, and that the frame-0 clamp actually changes the
video loss vs the no-bootstrap path (frame 0 no longer contributes its noised
target). CPU-only mini SANA + ActionDiT; gated on third_party/Sana.
"""

from __future__ import annotations

import torch

from tests.test_ar_compute_loss import _build_arch, requires_sana


def _batch(vb, ab, B=2, T=3, Ta=6):  # T>=3 so >1 video-loss frame remains after frame-0 exclusion
    return dict(
        input_latents=torch.randn(B, vb._dit.in_channels, T, 8, 8),
        actions=torch.randn(B, Ta, ab.action_dim),
        context=torch.randn(B, 4, vb.context_dim),
        seq_lens=torch.full((B,), 4, dtype=torch.long),
    )


@requires_sana
def test_bootstrap_clean_prefix_finite_and_differs():
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    arch._ar_noisy_cond_prob = 0.0  # determinism
    batch = _batch(vb, ab)

    arch._ar_bootstrap_clean_prefix = False
    torch.manual_seed(42)
    base = arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)

    arch._ar_bootstrap_clean_prefix = True
    torch.manual_seed(42)
    boot = arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)

    assert torch.isfinite(boot["loss"]).all()
    # Frame 0 clamped clean + excluded from the video loss ⇒ video loss differs.
    assert not torch.allclose(base["loss_video"], boot["loss_video"], atol=1e-6)

    boot["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in ab.parameters())
