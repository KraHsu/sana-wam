"""Parity: fork's linear-attn recurrences ≡ upstream Sana-wm's actual code.

The fork's ``_chunked_linear_attn`` (training) and ``ARLinearStateCache`` +
``ar_inference_attn`` (streaming) are already tested against the fork's *own*
``_expanded_linear_attn`` oracle. This module adds the missing guard: that the
recurrence matches **upstream's actual kernels** —
``ChunkCausalAttention.forward`` and ``CachedCausalAttention.forward`` in
``../Sana/diffusion/model/nets/sana_blocks.py`` (upstream Sana-wm's chunk-causal
linear attention) — not just an internal reference.

Upstream stores ``(B, h, h_d, N)`` and runs, per frame-chunk:

    _vk         = v @ k_rotated.transpose(-1, -2)      # (B, h, h_d, h_d)
    cumsum_vk  += _vk
    cumsum_ksum += k.sum(-1, keepdim).transpose(-2, -1)  # un-rotated k
    z           = 1 / (cumsum_ksum @ q + eps)            # un-rotated q
    out         = (cumsum_vk @ q_rotated) * z

The fork's dual-track names map directly: ``tilde`` = RoPE-rotated (numerator),
``phi`` = un-rotated ReLU (denominator). The ports below reproduce upstream's
loop verbatim in the fork's ``(B, H, N, d)`` layout so any future drift in the
fork's cumsum/cache trips here.
"""

from __future__ import annotations

import pytest
import torch

from sana_wam.model.ar.sana_ar_inference import (
    ARLinearStateCache,
    ar_inference_attn,
    clean_state_from_tokens,
)
from sana_wam.model.ar.sana_linear_attn import _chunked_linear_attn


def _upstream_chunkcausal_reference(tilde_q, tilde_k, v, phi_q, phi_k, chunk_index, eps):
    """Verbatim port of ``ChunkCausalAttention.forward`` cumsum loop.

    Inputs are ``(B, H, N, d)`` (fork layout); we operate on the last two dims
    transposed to mirror upstream's ``(B, h, h_d, N)`` storage. ``chunk_index``
    is the ascending boundary list ``[0, s0, s0+s1, ..., N]`` (a frame chunk per
    segment) — the same monotonic block-causal layout ``_chunked_linear_attn``
    consumes.
    """
    B, H, N, d = tilde_q.shape
    # to (B, H, d, N) — upstream's (B, h, h_d, N)
    tq = tilde_q.transpose(-1, -2)
    tk = tilde_k.transpose(-1, -2)
    vv = v.transpose(-1, -2)
    pq = phi_q.transpose(-1, -2)
    pk = phi_k.transpose(-1, -2)

    sizes = [chunk_index[i + 1] - chunk_index[i] for i in range(len(chunk_index) - 1)]
    tq_l = tq.split(sizes, dim=-1)
    tk_l = tk.split(sizes, dim=-1)
    vv_l = vv.split(sizes, dim=-1)
    pq_l = pq.split(sizes, dim=-1)
    pk_l = pk.split(sizes, dim=-1)

    cumsum_vk = torch.zeros(B, H, d, d, dtype=v.dtype, device=v.device)
    cumsum_ksum = torch.zeros(B, H, 1, d, dtype=v.dtype, device=v.device)
    out_list = []
    for _tq, _tk, _v, _pq, _pk in zip(tq_l, tk_l, vv_l, pq_l, pk_l):
        cumsum_vk = cumsum_vk + torch.matmul(_v, _tk.transpose(-1, -2))  # (B,H,d,d)
        cumsum_ksum = cumsum_ksum + _pk.sum(dim=-1, keepdim=True).transpose(-2, -1)  # (B,H,1,d)
        z = 1.0 / (cumsum_ksum @ _pq + eps)  # (B,H,1,n)
        out = torch.matmul(cumsum_vk, _tq) * z  # (B,H,d,n)
        out_list.append(out)
    out = torch.cat(out_list, dim=-1)  # (B,H,d,N)
    return out.transpose(-1, -2)  # back to (B,H,N,d)


def _rand(N, B=2, H=2, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    tq = torch.randn(B, H, N, d, generator=g, dtype=torch.float64)
    tk = torch.randn(B, H, N, d, generator=g, dtype=torch.float64)
    v = torch.randn(B, H, N, d, generator=g, dtype=torch.float64)
    pq = torch.randn(B, H, N, d, generator=g, dtype=torch.float64).relu()
    pk = torch.randn(B, H, N, d, generator=g, dtype=torch.float64).relu()
    return tq, tk, v, pq, pk


@pytest.mark.parametrize(
    "chunk_index",
    [[0, 2, 4, 6], [0, 3, 3, 7], [0, 4, 8, 12, 16], [0, 1, 2, 3, 4, 5]],
)
def test_fork_chunked_matches_upstream_chunkcausal(chunk_index):
    """Fork's ``_chunked_linear_attn`` ≡ upstream ``ChunkCausalAttention`` loop."""
    N = chunk_index[-1]
    tq, tk, v, pq, pk = _rand(N, seed=N * 7 + len(chunk_index))
    eps = 1e-12

    got = _chunked_linear_attn(tq, tk, v, pq, pk, chunk_index, eps=eps)
    ref = _upstream_chunkcausal_reference(tq, tk, v, pq, pk, chunk_index, eps=eps)

    torch.testing.assert_close(got, ref, atol=1e-9, rtol=1e-7)


def _upstream_cached_reference(tilde_q, tilde_k, v, phi_q, phi_k, chunk_index, eps):
    """Verbatim port of ``CachedCausalAttention.forward`` streamed over chunks.

    Mirrors upstream's ``kv_cache=[cumsum_vk, cumsum_k_sum]`` accumulation: each
    chunk reads the running cache, adds its own ``vk`` / ``k_sum``, computes its
    rows, and writes the cache back. Equivalent to the chunk-causal kernel — this
    is the inference-time form upstream uses for streaming.
    """
    B, H, N, d = tilde_q.shape
    tq = tilde_q.transpose(-1, -2)
    tk = tilde_k.transpose(-1, -2)
    vv = v.transpose(-1, -2)
    pq = phi_q.transpose(-1, -2)
    pk = phi_k.transpose(-1, -2)

    sizes = [chunk_index[i + 1] - chunk_index[i] for i in range(len(chunk_index) - 1)]
    splits = [t.split(sizes, dim=-1) for t in (tq, tk, vv, pq, pk)]

    cache_vk = None
    cache_ksum = None
    out_list = []
    for _tq, _tk, _v, _pq, _pk in zip(*splits):
        vk = torch.matmul(_v, _tk.transpose(-1, -2))  # (B,H,d,d)
        ksum = _pk.sum(dim=-1, keepdim=True).transpose(-2, -1)  # (B,H,1,d)
        if cache_vk is not None:
            vk = vk + cache_vk
            ksum = ksum + cache_ksum
        cache_vk = vk
        cache_ksum = ksum
        z = 1.0 / (ksum @ _pq + eps)
        out = torch.matmul(vk, _tq) * z
        out_list.append(out)
    return torch.cat(out_list, dim=-1).transpose(-1, -2)


@pytest.mark.parametrize("chunk_index", [[0, 2, 4, 6], [0, 4, 8, 12, 16]])
def test_fork_cache_matches_upstream_cached(chunk_index):
    """Fork ``ARLinearStateCache`` + ``ar_inference_attn`` ≡ upstream ``CachedCausalAttention``.

    No within-frame noise block here (the shared, non-duplicated topology each
    chunk is purely clean history + its own clean keys) so the fork cache path
    reduces to exactly upstream's streamed cumsum.
    """
    N = chunk_index[-1]
    tq, tk, v, pq, pk = _rand(N, seed=N + 100)
    eps = 1e-12

    ref = _upstream_cached_reference(tq, tk, v, pq, pk, chunk_index, eps=eps)

    # Stream the fork cache chunk-by-chunk: each frame is one chunk; a chunk's
    # query attends to all clean history up to and including itself.
    cache = ARLinearStateCache(num_layers=1, window=10_000)
    out = torch.empty_like(v)
    for c in range(len(chunk_index) - 1):
        lo, hi = chunk_index[c], chunk_index[c + 1]
        idx = torch.arange(lo, hi)
        S, z = clean_state_from_tokens(tk[:, :, idx, :], v[:, :, idx, :], pk[:, :, idx, :])
        cache.update(0, c, S, z, is_pred=False)
        win = cache.windowed_state(0, c, hi_inclusive=c)  # inclusive: sees itself
        got = ar_inference_attn(
            tq[:, :, idx, :],
            pq[:, :, idx, :],
            win,
            tk[:, :, idx, :][:, :, :0, :],  # empty within-frame noise block
            pk[:, :, idx, :][:, :, :0, :],
            v[:, :, idx, :][:, :, :0, :],
            eps=eps,
        )
        out[:, :, idx, :] = got

    torch.testing.assert_close(out, ref, atol=1e-9, rtol=1e-7)
