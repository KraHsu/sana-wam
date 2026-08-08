"""SANA autoregressive structured linear attention ≡ dense oracle.

The AR fast path :func:`_ar_chunked_linear_attn` realizes the LingBot-VA
block-causal attention topology (clean/noise duplicated sequence, modality-parity
``frame_id``, sliding window) under SANA's dual-track ReLU linear attention via a
windowed-clean-prefix + within-frame-noise decomposition.

:func:`_ar_expanded_reference` builds the *exact* LingBot dense ``(N, N)`` mask
(``ar_build_dense_mask``) and routes it through the audited
``_expanded_linear_attn`` — the unambiguous ground truth.

This test pins their equivalence across a grid of chunk sizes, window widths,
and chunk counts. It is the gate for every downstream AR phase: if the kernel
and oracle disagree, the architecture/loss/rollout built on top would be wrong
only at masked positions and slip past coarse integration tests.

Pure CPU math — no SANA dependency — so a SANA pin-bump cannot mask a regression.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from sana_wam.model.ar.sana_ar_linear_attn import (
    ARSeqMeta,
    _ar_chunked_linear_attn,
    _ar_expanded_reference,
    ar_build_dense_mask,
    build_ar_seq_meta,
)


def _make_qkv(N: int, B: int = 2, H: int = 3, d: int = 8, seed: int = 0):
    """Dual-track inputs in float64. ``phi`` is ReLU'd (nonneg, matching SANA's
    un-rotated kernel track); ``tilde`` is unconstrained (post-RoPE)."""
    g = torch.Generator().manual_seed(seed)
    tilde_q = torch.randn(B, H, N, d, generator=g, dtype=torch.float64)
    tilde_k = torch.randn(B, H, N, d, generator=g, dtype=torch.float64)
    v = torch.randn(B, H, N, d, generator=g, dtype=torch.float64)
    phi_q = torch.randn(B, H, N, d, generator=g, dtype=torch.float64).relu()
    phi_k = torch.randn(B, H, N, d, generator=g, dtype=torch.float64).relu()
    return tilde_q, tilde_k, v, phi_q, phi_k


# Small, irregular per-chunk token counts stress the within-frame block and the
# parity arithmetic without exploding the O(N^2) oracle.
_GRID = list(
    itertools.product(
        [1, 2, 3],  # num_chunks
        [2, 3],  # video_tokens_per_chunk
        [1, 2],  # action_tokens_per_chunk
        [1, 2, 4, 65],  # window (frame_id units; 65 ⇒ effectively unbounded here)
    )
)


@pytest.mark.parametrize("num_chunks,vtok,atok,window", _GRID)
def test_ar_chunked_matches_expanded(num_chunks, vtok, atok, window):
    meta = build_ar_seq_meta(
        num_chunks=num_chunks,
        video_tokens_per_chunk=vtok,
        action_tokens_per_chunk=atok,
        window=window,
    )
    N = meta.frame_ids.numel()
    tq, tk, v, pq, pk = _make_qkv(N, seed=num_chunks * 100 + vtok * 10 + atok + window)

    eps = 1e-12
    ref = _ar_expanded_reference(tq, tk, v, pq, pk, meta, eps=eps)
    fast = _ar_chunked_linear_attn(tq, tk, v, pq, pk, meta, eps=eps)

    assert fast.shape == ref.shape == (2, 3, N, 8)
    torch.testing.assert_close(fast, ref, atol=1e-9, rtol=1e-7)


def test_dense_mask_topology():
    """Spot-check the dense mask encodes the asymmetric video↔action visibility
    and clean/noise rules directly (independent of the kernel)."""
    meta = build_ar_seq_meta(
        num_chunks=2, video_tokens_per_chunk=1, action_tokens_per_chunk=1, window=99
    )
    # Layout: [v_noisy(c0,c1), v_clean(c0,c1), a_noisy(c0,c1), a_clean(c0,c1)]
    # frame_ids = [0,2, 0,2, 1,3, 1,3]; noise = [0,0, 1,1, 0,0, 1,1]
    fids = meta.frame_ids.tolist()
    nids = meta.noise_ids.tolist()
    assert fids == [0, 2, 0, 2, 1, 3, 1, 3]
    assert nids == [0, 0, 1, 1, 0, 0, 1, 1]

    m = ar_build_dense_mask(meta)

    # indices: v_noisy c0=0, v_noisy c1=1, v_clean c0=2, v_clean c1=3,
    #          a_noisy c0=4, a_noisy c1=5, a_clean c0=6, a_clean c1=7
    # clean action chunk0 (idx6, frame1) attends clean video chunk0 (idx2, frame0): 0<=1 ✓
    assert bool(m[6, 2])
    # clean video chunk0 (idx2, frame0) must NOT attend clean action chunk0 (idx6, frame1): 1<=0 ✗
    assert not bool(m[2, 6])
    # noisy video chunk1 (idx1, frame2) attends its own noisy block (idx1): self ✓
    assert bool(m[1, 1])
    # noisy never attends a different-frame noisy key (block_self only)
    assert not bool(m[1, 0])  # frame2 noisy vs frame0 noisy
    # noisy query attends strictly-past clean (idx1 frame2 → clean video c0 idx2 frame0): 0<2 ✓
    assert bool(m[1, 2])
    # noisy query does NOT attend its own-frame clean (strict causal): idx1 frame2 → clean video c1 idx3 frame2: 2<2 ✗
    assert not bool(m[1, 3])
    # clean never attends noisy
    assert not bool(m[2, 0])


def test_precomputed_plan_matches_fallback():
    """``build_ar_seq_meta`` precomputes a per-frame index plan so the kernel hot
    path needs no ``.item()`` / ``bool(.any())`` / ``.nonzero()`` device syncs.
    The plan must (a) be populated and (b) yield bit-identical output to an ad-hoc
    ``ARSeqMeta`` that carries no plan (forcing the derive-on-the-fly fallback)."""
    meta = build_ar_seq_meta(
        num_chunks=3, video_tokens_per_chunk=2, action_tokens_per_chunk=2, window=2
    )
    assert meta.num_frames == int(meta.frame_ids.max().item()) + 1
    assert len(meta.noisy_idx_per_frame) == meta.num_frames
    assert len(meta.clean_idx_per_frame) == meta.num_frames

    bare = ARSeqMeta(meta.frame_ids, meta.noise_ids, meta.window)  # no plan -> fallback
    assert bare.num_frames == 0 and bare.noisy_idx_per_frame == ()

    N = meta.frame_ids.numel()
    tq, tk, v, pq, pk = _make_qkv(N, seed=123)
    fast = _ar_chunked_linear_attn(tq, tk, v, pq, pk, meta, eps=1e-12)
    fallback = _ar_chunked_linear_attn(tq, tk, v, pq, pk, bare, eps=1e-12)
    torch.testing.assert_close(fast, fallback, atol=0.0, rtol=0.0)


def test_window_truncates_history():
    """A tight window must drop far-past clean frames; verify via the oracle and
    that the kernel tracks it."""
    meta_wide = build_ar_seq_meta(
        num_chunks=4, video_tokens_per_chunk=2, action_tokens_per_chunk=1, window=99
    )
    meta_tight = ARSeqMeta(meta_wide.frame_ids, meta_wide.noise_ids, window=1)
    N = meta_wide.frame_ids.numel()
    tq, tk, v, pq, pk = _make_qkv(N, seed=7)

    wide = _ar_chunked_linear_attn(tq, tk, v, pq, pk, meta_wide, eps=1e-12)
    tight = _ar_chunked_linear_attn(tq, tk, v, pq, pk, meta_tight, eps=1e-12)
    # Windowing must change the result for a multi-chunk sequence.
    assert not torch.allclose(wide, tight, atol=1e-6)
    # And the tight kernel must still match its own dense oracle.
    tight_ref = _ar_expanded_reference(tq, tk, v, pq, pk, meta_tight, eps=1e-12)
    torch.testing.assert_close(tight, tight_ref, atol=1e-9, rtol=1e-7)


def test_bfloat16_kernel_uses_fp32_prefix_math_and_restores_output_dtype():
    """Low-precision feature tracks enter one fp32 operator/state definition."""

    class RecordingIdentityAdapter:
        def __init__(self):
            self.state_dtypes = []

        def forward_delta(self, layer_id, S, z):  # noqa: N803
            assert layer_id == 0
            self.state_dtypes.append((S.dtype, z.dtype))
            return torch.zeros_like(S), torch.zeros_like(z)

    meta = build_ar_seq_meta(
        num_chunks=2,
        video_tokens_per_chunk=2,
        action_tokens_per_chunk=2,
        window=4,
    )
    values = list(_make_qkv(meta.frame_ids.numel(), B=1, H=2, d=4, seed=91))
    values[3] = values[3] + 0.25
    values[4] = values[4] + 0.25
    low = tuple(
        value.to(torch.bfloat16).detach().requires_grad_(True) for value in values
    )

    adapter = RecordingIdentityAdapter()
    output, captured = _ar_chunked_linear_attn(
        *low,
        meta,
        action_video_memory_adapter=adapter,
        layer_id=0,
        return_action_video_numerators=True,
    )
    fp32_output, fp32_captured = _ar_chunked_linear_attn(
        *(value.detach().float() for value in low),
        meta,
        action_video_memory_adapter=RecordingIdentityAdapter(),
        layer_id=0,
        return_action_video_numerators=True,
    )

    assert output.dtype == captured.dtype == torch.bfloat16
    assert adapter.state_dtypes
    assert all(
        state_dtype == normalizer_dtype == torch.float32
        for state_dtype, normalizer_dtype in adapter.state_dtypes
    )
    assert torch.equal(output, fp32_output.to(torch.bfloat16))
    assert torch.equal(captured, fp32_captured.to(torch.bfloat16))

    (output.float().square().mean() + captured.float().square().mean()).backward()
    assert all(value.grad is not None for value in low)
    assert all(value.grad.dtype == torch.bfloat16 for value in low)
    assert all(torch.isfinite(value.grad).all() for value in low)
