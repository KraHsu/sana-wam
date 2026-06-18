"""F4: per-chunk proprioception conditioning.

Verifies that the additive per-chunk proprio AdaLN delta (a) is a no-op at init
(zero-initialized encoders ⇒ identical loss to the single-proprio path, so the
retrain warm-starts from the current checkpoint), and (b) once the encoders carry
weight, different per-chunk robot states change the predictions. CPU-only mini
SANA + ActionDiT; gated on third_party/Sana.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tests.test_ar_compute_loss import _build_arch, requires_sana

S = 6  # proprio/state dim


def _enable_proprio(arch, vb, ab, *, per_chunk: bool):
    arch._use_proprioception_context = True
    arch.proprio_dim = S
    arch.context_dim = vb.context_dim
    arch.proprio_encoder = nn.Linear(S, vb.context_dim)
    arch._proprio_per_chunk = per_chunk
    arch.proprio_video_embed = None
    arch.proprio_action_embed = None
    if per_chunk:
        arch._maybe_init_per_chunk_proprio()  # zero-init
    arch.set_dtype_device(torch.float32, torch.device("cpu"))


def _batch(vb, ab, B=2, T=2, Ta=4):
    return dict(
        input_latents=torch.randn(B, vb._dit.in_channels, T, 8, 8),
        actions=torch.randn(B, Ta, ab.action_dim),
        context=torch.randn(B, 4, vb.context_dim),
        seq_lens=torch.full((B,), 4, dtype=torch.long),
        proprio_state=torch.randn(B, S),
        proprio_seq=torch.randn(B, Ta, S),  # full state seq; idx [0,atok] picked per chunk
    )


@requires_sana
def test_per_chunk_proprio_zero_init_is_noop():
    """Zero-init per-chunk encoders ⇒ loss identical to the single-proprio path."""
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    _enable_proprio(arch, vb, ab, per_chunk=True)
    batch = _batch(vb, ab)

    torch.manual_seed(123)
    loss_pc = arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)["loss"]

    # Disable per-chunk (encoders unused) and recompute with the SAME seed.
    arch._proprio_per_chunk = False
    torch.manual_seed(123)
    loss_base = arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)["loss"]

    torch.testing.assert_close(loss_pc, loss_base, atol=1e-6, rtol=1e-5)


@requires_sana
def test_per_chunk_proprio_changes_prediction_when_trained():
    """With non-zero encoders, different per-chunk states change the loss + grads flow."""
    torch.manual_seed(0)
    arch, vb, ab = _build_arch()
    _enable_proprio(arch, vb, ab, per_chunk=True)
    # Give the per-chunk encoders real weight (simulate a trained model).
    for m in (arch.proprio_video_embed, arch.proprio_action_embed):
        nn.init.normal_(m.weight, std=0.5)
        nn.init.normal_(m.bias, std=0.5)

    batch = _batch(vb, ab)
    batch_b = dict(batch)
    batch_b["proprio_seq"] = batch["proprio_seq"] + 3.0  # different per-chunk states

    torch.manual_seed(7)
    out_a = arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch)
    torch.manual_seed(7)
    loss_b = arch.compute_loss(lambda_video=1.0, lambda_action=1.0, **batch_b)["loss"]

    assert not torch.allclose(out_a["loss"], loss_b, atol=1e-5)

    out_a["loss"].backward()
    assert arch.proprio_video_embed.weight.grad is not None
    assert arch.proprio_action_embed.weight.grad is not None
    assert torch.isfinite(arch.proprio_video_embed.weight.grad).all()
    assert torch.isfinite(arch.proprio_action_embed.weight.grad).all()
