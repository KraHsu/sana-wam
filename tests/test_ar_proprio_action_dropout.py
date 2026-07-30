"""Action-only proprio dropout routing for the block-AR architecture."""

from __future__ import annotations

import pytest
import torch

from tests.test_ar_compute_loss import _build_arch, requires_sana
from tests.test_ar_per_chunk_proprio import S, _enable_proprio


class _CapturedActionState(RuntimeError):
    pass


def _capture_routing(monkeypatch, *, training: bool, drop_prob: float):
    arch, vb, ab = _build_arch()
    _enable_proprio(arch, vb, ab, per_chunk=True)
    arch._proprio_action_dropout_prob = drop_prob
    arch.train(training)

    with torch.no_grad():
        arch.proprio_action_embed.weight.fill_(0.25)
        arch.proprio_action_embed.bias.fill_(0.5)

    captured = {}
    original_video_prepare = vb.prepare

    def capture_video(**kwargs):
        captured["video_context"] = kwargs["context"].detach().clone()
        captured["video_mask"] = kwargs["context_mask"].detach().clone()
        return original_video_prepare(**kwargs)

    def capture_action(*args, **kwargs):
        captured["action_context"] = kwargs["context"].detach().clone()
        captured["action_mask"] = kwargs["context_mask"].detach().clone()
        captured["action_proprio"] = kwargs["token_proprio_emb"].detach().clone()
        raise _CapturedActionState

    monkeypatch.setattr(vb, "prepare", capture_video)
    monkeypatch.setattr(ab, "prepare_state", capture_action)

    batch = 2
    frames = 2
    action_tokens = 4
    text_tokens = 3
    with pytest.raises(_CapturedActionState):
        arch.forward(
            torch.randn(batch, action_tokens, ab.action_dim),
            None,
            latents=torch.randn(batch, vb._dit.in_channels, frames, 8, 8),
            ar_clean_latents=torch.randn(batch, vb._dit.in_channels, frames, 8, 8),
            ar_clean_actions=torch.randn(batch, action_tokens, ab.action_dim),
            ar_video_frame_timesteps=torch.full((batch, frames), 500.0),
            ar_action_token_timesteps=torch.full((batch, action_tokens), 500.0),
            ar_frame_chunk_size=1,
            ar_attn_window=72,
            context=torch.randn(batch, text_tokens, vb.context_dim),
            seq_lens=torch.full((batch,), text_tokens, dtype=torch.long),
            proprio_state=torch.randn(batch, S),
            proprio_per_chunk=torch.ones(batch, frames, S),
        )
    return captured


@requires_sana
@pytest.mark.parametrize(
    ("training", "drop_prob", "expect_drop"),
    [(True, 1.0, True), (True, 0.0, False), (False, 1.0, False)],
)
def test_proprio_dropout_only_changes_action_branch(monkeypatch, training, drop_prob, expect_drop):
    captured = _capture_routing(monkeypatch, training=training, drop_prob=drop_prob)

    video_context = captured["video_context"]
    action_context = captured["action_context"]
    assert torch.count_nonzero(video_context[:, -1]) > 0
    torch.testing.assert_close(captured["video_mask"], captured["action_mask"])
    assert bool(captured["action_mask"][:, -1].all())

    if expect_drop:
        assert torch.count_nonzero(action_context[:, -1]) == 0
        assert torch.count_nonzero(captured["action_proprio"]) == 0
        torch.testing.assert_close(action_context[:, :-1], video_context[:, :-1])
    else:
        torch.testing.assert_close(action_context, video_context)
        assert torch.count_nonzero(captured["action_proprio"]) > 0
