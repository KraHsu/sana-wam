"""CPU coverage for the optional SANA window-flash residual in split blocks."""

from __future__ import annotations

from types import SimpleNamespace

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


class _ZeroCrossAttention(nn.Module):
    def forward(self, x, _y, mask=None):
        del mask
        return torch.zeros_like(x)


class _ZeroMLP(nn.Module):
    def forward(self, x, HW=None):
        del HW
        return torch.zeros_like(x)


class _TemporalMean(nn.Module):
    """Deliberately mixes every frame passed in one temporal call."""

    def forward(self, x, HW=None, **kwargs):
        del kwargs
        t, h, w = HW
        batch, _, channels = x.shape
        framed = x.reshape(batch, t, h * w, channels)
        mixed = framed.mean(dim=1, keepdim=True).expand_as(framed)
        return mixed.reshape_as(x)


class _RecordingGraft(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output
        self.seen_input_contiguous = False
        self.seen_rotary = None
        self.seen_hw = None

    def forward(self, x, *, rotary_emb=None, HW=None):
        self.seen_input_contiguous = x.is_contiguous()
        self.seen_rotary = rotary_emb
        self.seen_hw = HW
        return self.output


class _GateRecorder:
    """Tensor-like gate that records the layout reaching the residual gate."""

    def __init__(self, scale: float):
        self.scale = scale
        self.seen_input_contiguous = False

    def dim(self):
        return 3

    def __mul__(self, value):
        self.seen_input_contiguous = value.is_contiguous()
        return self.scale * value


def _block(*, graft=None, scale=1.0, mlp=None):
    return SimpleNamespace(
        attn=SimpleNamespace(proj=nn.Identity()),
        flash_attn_additional=graft,
        learnable_fa_scale=torch.tensor(scale),
        drop_path=nn.Identity(),
        cross_attn_image_embeds=False,
        cross_attn=_ZeroCrossAttention(),
        norm2=nn.Identity(),
        mlp=_ZeroMLP() if mlp is None else mlp,
    )


def _post_state(block, gate_msa, residual):
    return {
        "block": block,
        "residual_x": residual,
        "gate_msa": gate_msa,
        "shift_mlp": torch.zeros(1, 1, residual.shape[-1]),
        "scale_mlp": torch.zeros(1, 1, residual.shape[-1]),
        "gate_mlp": torch.zeros(1, 1, residual.shape[-1]),
    }


@requires_sana
def test_sana_constructor_threads_window_flash_options_to_every_block():
    from diffusion.model.nets.sana_blocks import WindowAttention
    from diffusion.model.nets.sana_multi_scale_video import SanaMSVideo
    from sana_wam.model.video_backbone.sana.blocks_split import SanaMSVideoSplit

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
        additional_flash_attn="window_flash",
        flash_attn_window_count=[1, 1, 4],
    )

    assert all(isinstance(block.flash_attn_additional, WindowAttention) for block in model.blocks)
    assert all(block.flash_attn_additional.window_count == [1, 1, 4] for block in model.blocks)

    x = torch.randn(1, 4, 32)
    t0 = torch.zeros(1, 6 * 32)
    rotary = torch.ones(1, 1, 4, 8, dtype=torch.complex128)
    *_, state = SanaMSVideoSplit(model).block_pre_attn(0, x, t0, rotary)
    shift_msa, scale_msa, *_ = (
        model.blocks[0].scale_shift_table[None] + t0.reshape(1, 6, 32)
    ).chunk(6, dim=1)
    expected_x_sa_in = model.blocks[0].norm1(x) * (1 + scale_msa) + shift_msa
    torch.testing.assert_close(state["_x_sa_in"], expected_x_sa_in)
    assert state["_rotary_emb"] is rotary


def test_split_post_without_graft_keeps_original_path_and_state_contract():
    from sana_wam.model.video_backbone.sana.blocks_split import SanaMSVideoSplit

    attn_out = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    residual = torch.full_like(attn_out, 10.0)
    gate = torch.full((1, 1, 3), 0.25)
    block = _block(graft=None)
    split = SanaMSVideoSplit(SimpleNamespace(blocks=[block]))
    state = _post_state(block, gate, residual)

    # Deliberately omit the new graft-only state keys: the default path must
    # neither read them nor perturb the pre-existing residual formula.
    actual = split.block_post_attn(
        0,
        attn_out,
        state,
        y=torch.empty(1, 0, 3),
        y_lens=torch.empty(1, 0),
        f=1,
        h=2,
        w=2,
    )

    torch.testing.assert_close(actual, residual + gate * attn_out)


def test_split_window_flash_runs_before_gate_with_contiguous_boundaries():
    from sana_wam.model.video_backbone.sana.blocks_split import SanaMSVideoSplit

    # Both split inputs use a dense but non-contiguous B,N,C layout, matching
    # the layout that exposed the CUDA alignment failure in AR rollout.
    attn_out = torch.full((1, 3, 4), 2.0).transpose(1, 2)
    x_sa_in = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).transpose(1, 2)
    graft_output = torch.full((1, 3, 4), 3.0).transpose(1, 2)
    assert not attn_out.is_contiguous()
    assert not x_sa_in.is_contiguous()
    assert not graft_output.is_contiguous()

    graft = _RecordingGraft(graft_output)
    block = _block(graft=graft, scale=4.0)
    gate = _GateRecorder(scale=0.5)
    residual = torch.full_like(attn_out, 10.0)
    state = _post_state(block, gate, residual)
    rotary = object()
    state.update({"_x_sa_in": x_sa_in, "_rotary_emb": rotary})
    split = SanaMSVideoSplit(SimpleNamespace(blocks=[block]))

    actual = split.block_post_attn(
        0,
        attn_out,
        state,
        y=torch.empty(1, 0, 3),
        y_lens=torch.empty(1, 0),
        f=1,
        h=2,
        w=2,
    )

    # residual + gate * (linear projection + learnable scale * graft)
    torch.testing.assert_close(actual, torch.full_like(actual, 17.0))
    assert graft.seen_input_contiguous
    assert graft.seen_rotary is rotary
    assert graft.seen_hw == (1, 2, 2)
    assert gate.seen_input_contiguous


def test_chunkwise_temporal_suffix_matches_independent_chunk_execution():
    from sana_wam.model.video_backbone.sana.blocks_split import SanaMSVideoSplit

    torch.manual_seed(7)
    batch, frames, channels = 1, 4, 3
    chunk_frames = 2
    residual = torch.randn(batch, frames, channels)
    attn_out = torch.randn_like(residual)
    x_sa_in = torch.randn_like(residual)
    rotary = torch.randn(1, 1, frames, 2, dtype=torch.complex64)
    block = _block(graft=_TemporalMean(), mlp=_TemporalMean())
    split = SanaMSVideoSplit(SimpleNamespace(blocks=[block]))

    gate_msa = torch.randn(batch, frames, 1, channels)
    shift_mlp = torch.randn(batch, frames, 1, channels)
    scale_mlp = torch.randn(batch, frames, 1, channels)
    gate_mlp = torch.randn(batch, frames, 1, channels)
    state = {
        "block": block,
        "residual_x": residual,
        "gate_msa": gate_msa,
        "shift_mlp": shift_mlp,
        "scale_mlp": scale_mlp,
        "gate_mlp": gate_mlp,
        "_x_sa_in": x_sa_in,
        "_rotary_emb": rotary,
    }
    full = split.block_post_attn(
        0,
        attn_out,
        state,
        y=torch.empty(batch, 0, channels),
        y_lens=torch.empty(batch, 0),
        f=frames,
        h=1,
        w=1,
        temporal_chunk_frames=chunk_frames,
    )

    independent = []
    for start in range(0, frames, chunk_frames):
        end = start + chunk_frames
        chunk_state = {
            "block": block,
            "residual_x": residual[:, start:end],
            "gate_msa": gate_msa[:, start:end],
            "shift_mlp": shift_mlp[:, start:end],
            "scale_mlp": scale_mlp[:, start:end],
            "gate_mlp": gate_mlp[:, start:end],
            "_x_sa_in": x_sa_in[:, start:end],
            "_rotary_emb": rotary[..., start:end, :],
        }
        independent.append(
            split.block_post_attn(
                0,
                attn_out[:, start:end],
                chunk_state,
                y=torch.empty(batch, 0, channels),
                y_lens=torch.empty(batch, 0),
                f=chunk_frames,
                h=1,
                w=1,
            )
        )

    torch.testing.assert_close(full, torch.cat(independent, dim=1))

    changed_state = dict(state)
    changed_x_sa = x_sa_in.clone()
    changed_x_sa[:, chunk_frames:] += 100.0
    changed_state["_x_sa_in"] = changed_x_sa
    changed = split.block_post_attn(
        0,
        attn_out,
        changed_state,
        y=torch.empty(batch, 0, channels),
        y_lens=torch.empty(batch, 0),
        f=frames,
        h=1,
        w=1,
        temporal_chunk_frames=chunk_frames,
    )
    torch.testing.assert_close(changed[:, :chunk_frames], full[:, :chunk_frames])


def test_chunkwise_temporal_suffix_rejects_partial_chunk():
    from sana_wam.model.video_backbone.sana.blocks_split import SanaMSVideoSplit

    residual = torch.zeros(1, 3, 2)
    block = _block()
    split = SanaMSVideoSplit(SimpleNamespace(blocks=[block]))
    state = _post_state(block, torch.ones(1, 1, 2), residual)

    with pytest.raises(ValueError, match="must be divisible"):
        split.block_post_attn(
            0,
            residual,
            state,
            y=torch.empty(1, 0, 2),
            y_lens=torch.empty(1, 0),
            f=3,
            h=1,
            w=1,
            temporal_chunk_frames=2,
        )
