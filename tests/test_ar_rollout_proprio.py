"""F4: ar_rollout must thread the FastWAM proprio context token.

Training appends a proprio token to the text context seen by both streams'
cross-attention (``_append_proprio_context_token``); the closed-loop rollout
previously ignored proprio entirely, so a model trained with
``use_proprioception=True`` ran off-distribution at inference. ``ar_rollout`` now
builds the augmented context per step via ``_rollout_step_context``.

These pin the new seam on CPU (no SANA): the proprio token is appended with an
authoritative mask, the path is a no-op when proprio is disabled, and a missing
proprio state raises when proprio is enabled.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sana_wam.model.architecture import DualSystemARArchitecture


def _arch_with_proprio(state_dim: int, context_dim: int) -> DualSystemARArchitecture:
    arch = DualSystemARArchitecture(cfg=None)
    # Mimic _init_proprio_context(use_proprioception=True) without building backbones.
    arch._use_proprioception_context = True
    arch.proprio_dim = state_dim
    arch.context_dim = context_dim
    arch.proprio_encoder = nn.Linear(state_dim, context_dim)
    return arch


def test_step_context_appends_proprio_token():
    torch.manual_seed(0)
    B, L, D, S = 2, 5, 6, 4
    arch = _arch_with_proprio(state_dim=S, context_dim=D)
    context = torch.randn(B, L, D)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    proprio = torch.randn(B, S)

    ctx, mask = arch._rollout_step_context(context, seq_lens, proprio)

    assert ctx.shape == (B, L + 1, D)
    assert mask.shape == (B, L + 1)
    assert mask.dtype == torch.bool and bool(mask.all())  # all text valid + proprio
    # The appended token is exactly the encoded proprio state.
    expected = arch.proprio_encoder(proprio)
    torch.testing.assert_close(ctx[:, -1, :], expected)
    # Existing text tokens are untouched.
    torch.testing.assert_close(ctx[:, :L, :], context)


def test_step_context_noop_without_proprio():
    """cfg=None ⇒ proprio disabled ⇒ context unchanged, mask from seq_lens."""
    B, L, D = 2, 5, 6
    arch = DualSystemARArchitecture(cfg=None)
    assert not arch.uses_proprioception
    context = torch.randn(B, L, D)
    seq_lens = torch.tensor([5, 3])

    ctx, mask = arch._rollout_step_context(context, seq_lens, proprio_state=None)

    assert ctx is context  # untouched (no proprio token)
    expected_mask = torch.arange(L).unsqueeze(0) < seq_lens.unsqueeze(1)
    torch.testing.assert_close(mask, expected_mask)


def test_step_context_requires_proprio_when_enabled():
    arch = _arch_with_proprio(state_dim=4, context_dim=6)
    context = torch.randn(1, 3, 6)
    with pytest.raises(ValueError, match="requires `proprio_state`"):
        arch._rollout_step_context(context, seq_lens=None, proprio_state=None)
