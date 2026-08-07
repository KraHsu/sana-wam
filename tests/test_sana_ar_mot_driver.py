"""SanaARMoTJointDriver._mixed_attention ≡ AR dense oracle.

Validates that the driver-level reshape + descriptor plumbing routes the
concatenated ``[v_noisy, v_clean, a_noisy, a_clean]`` sequence through the
structured AR kernel correctly — and that it refuses the failure modes (missing
descriptor, dense mask, missing phi) that would silently corrupt an AR forward.
"""

from __future__ import annotations

import gc
import types
import weakref

import pytest
import torch
import torch.nn as nn

from sana_wam.model.ar.sana_ar_linear_attn import (
    _ar_expanded_reference,
    build_ar_seq_meta,
)
from sana_wam.model.ar.sana_ar_mot_driver import SanaARMoTJointDriver


class _FakeBackbone(nn.Module):
    """Minimal linear_relu backbone role for driver construction."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, num_layers: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._num_layers = num_layers
        self.dim = dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def attn_kernel(self) -> str:
        return "linear_relu"

    video_attention_mask_mode = "bidirectional"


def _build_ar_driver(num_heads=2, head_dim=8, num_layers=2) -> SanaARMoTJointDriver:
    kw = dict(num_heads=num_heads, head_dim=head_dim, num_layers=num_layers)
    vb = _FakeBackbone(dim=16, **kw)
    ab = _FakeBackbone(dim=24, **kw)
    return SanaARMoTJointDriver(vb, ab, attention_mask_mode="joint")


def _rand_tracks(N, num_heads=2, head_dim=8, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    HD = num_heads * head_dim
    tq = torch.randn(B, N, HD, generator=g, dtype=torch.float64)
    tk = torch.randn(B, N, HD, generator=g, dtype=torch.float64)
    v = torch.randn(B, N, HD, generator=g, dtype=torch.float64)
    pq = torch.randn(B, N, HD, generator=g, dtype=torch.float64).relu()
    pk = torch.randn(B, N, HD, generator=g, dtype=torch.float64).relu()
    return tq, tk, v, pq, pk


@pytest.mark.parametrize("num_chunks,vtok,atok,window", [(1, 2, 1, 99), (3, 2, 1, 99), (3, 3, 2, 2), (4, 2, 2, 1)])
def test_driver_mixed_attention_matches_ar_oracle(num_chunks, vtok, atok, window):
    driver = _build_ar_driver()
    driver.eps = 1e-12  # tighten for the float64 comparison
    meta = build_ar_seq_meta(
        num_chunks=num_chunks, video_tokens_per_chunk=vtok, action_tokens_per_chunk=atok, window=window
    )
    N = meta.frame_ids.numel()
    tq, tk, v, pq, pk = _rand_tracks(N, seed=num_chunks * 7 + window)

    driver._ar_meta = meta
    got = driver._mixed_attention(tq, tk, v, None, phi_q=pq, phi_k=pk)

    # Oracle in (B, H, N, d) layout.
    from einops import rearrange

    n = driver.num_heads
    ref = _ar_expanded_reference(
        rearrange(tq, "b s (n d) -> b n s d", n=n),
        rearrange(tk, "b s (n d) -> b n s d", n=n),
        rearrange(v, "b s (n d) -> b n s d", n=n),
        rearrange(pq, "b s (n d) -> b n s d", n=n),
        rearrange(pk, "b s (n d) -> b n s d", n=n),
        meta,
        eps=1e-12,
    )
    ref = rearrange(ref, "b n s d -> b s (n d)", n=n)
    torch.testing.assert_close(got, ref, atol=1e-9, rtol=1e-7)


def test_driver_refuses_missing_meta():
    driver = _build_ar_driver()
    tq, tk, v, pq, pk = _rand_tracks(4)
    with pytest.raises(RuntimeError, match="without an AR descriptor"):
        driver._mixed_attention(tq, tk, v, None, phi_q=pq, phi_k=pk)


def test_driver_refuses_dense_mask():
    driver = _build_ar_driver()
    meta = build_ar_seq_meta(num_chunks=2, video_tokens_per_chunk=1, action_tokens_per_chunk=1, window=9)
    driver._ar_meta = meta
    N = meta.frame_ids.numel()
    tq, tk, v, pq, pk = _rand_tracks(N)
    dense = torch.ones(N, N, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="dense attn_mask"):
        driver._mixed_attention(tq, tk, v, dense, phi_q=pq, phi_k=pk)


def test_driver_refuses_missing_phi():
    driver = _build_ar_driver()
    meta = build_ar_seq_meta(num_chunks=2, video_tokens_per_chunk=1, action_tokens_per_chunk=1, window=9)
    driver._ar_meta = meta
    N = meta.frame_ids.numel()
    tq, tk, v, _, _ = _rand_tracks(N)
    with pytest.raises(RuntimeError, match="requires phi_q and phi_k"):
        driver._mixed_attention(tq, tk, v, None, phi_q=None, phi_k=None)


def test_checkpoint_closure_does_not_retain_mutated_output_states():
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        driver = _build_ar_driver(num_layers=1)
        driver._ar_meta = build_ar_seq_meta(
            num_chunks=1,
            video_tokens_per_chunk=1,
            action_tokens_per_chunk=1,
            window=1,
        )

        class State:
            pass

        video_state = State()
        action_state = State()
        payload = State()
        video_input = torch.randn(2, 3, requires_grad=True)
        action_input = torch.randn(2, 3, requires_grad=True)
        video_state.x = video_input
        action_state.payload = payload
        payload.x_action = action_input

        def fake_step_impl(
            self,
            layer_id,
            local_video,
            local_action,
            *,
            attn_mask,
            suppress_inner_attn_ckpt,
        ):
            del self, layer_id, attn_mask, suppress_inner_attn_ckpt
            video = local_video.x
            action = local_action.payload.x_action
            local_video.x = video.tanh() + action.mean()
            local_action.payload.x_action = action.sin() + video.mean()
            return local_video, local_action

        driver._step_impl = types.MethodType(fake_step_impl, driver)
        output_video, output_action = driver._step_checkpointed(
            0,
            video_state,
            action_state,
            attn_mask=None,
            offload=False,
        )
        driver._ar_meta = None
        loss = output_video.x.sum() + output_action.payload.x_action.sum()
        loss.backward()
        assert video_input.grad is not None
        assert action_input.grad is not None

        video_state_ref = weakref.ref(video_state)
        action_state_ref = weakref.ref(action_state)
        payload_ref = weakref.ref(payload)
        del loss, output_video, output_action
        del video_input, action_input, video_state, action_state, payload
        assert video_state_ref() is None
        assert action_state_ref() is None
        assert payload_ref() is None
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()
