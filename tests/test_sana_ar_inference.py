"""Cache-based AR inference attention ≡ training kernel.

The rollout must produce, for the current noisy chunk, exactly what the training
kernel computes for that chunk's noisy rows — otherwise train/inference diverge.
This pins: building :class:`ARLinearStateCache` from the clean frames and running
:func:`ar_inference_attn` per noisy chunk reproduces the noisy-row slice of
``_ar_chunked_linear_attn`` over the full duplicated sequence.

Also covers the cache's predicted/confirmed replacement and window eviction.
"""

from __future__ import annotations

import pytest
import torch

from sana_wam.model.ar.sana_ar_inference import (
    ARLinearStateCache,
    ar_inference_attn,
    clean_state_from_tokens,
)
from sana_wam.model.ar.sana_ar_linear_attn import (
    _ar_chunked_linear_attn,
    build_ar_seq_meta,
)


def _rand(N, B=2, H=2, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    tq = torch.randn(B, H, N, d, generator=g, dtype=torch.float64)
    tk = torch.randn(B, H, N, d, generator=g, dtype=torch.float64)
    v = torch.randn(B, H, N, d, generator=g, dtype=torch.float64)
    pq = torch.randn(B, H, N, d, generator=g, dtype=torch.float64).relu()
    pk = torch.randn(B, H, N, d, generator=g, dtype=torch.float64).relu()
    return tq, tk, v, pq, pk


@pytest.mark.parametrize(
    "num_chunks,vtok,atok,window", [(3, 2, 1, 99), (4, 2, 2, 2), (3, 3, 1, 1)]
)
def test_cache_inference_matches_training_kernel(num_chunks, vtok, atok, window):
    meta = build_ar_seq_meta(
        num_chunks=num_chunks,
        video_tokens_per_chunk=vtok,
        action_tokens_per_chunk=atok,
        window=window,
    )
    N = meta.frame_ids.numel()
    tq, tk, v, pq, pk = _rand(N, seed=num_chunks * 13 + window)
    eps = 1e-12

    full = _ar_chunked_linear_attn(tq, tk, v, pq, pk, meta, eps=eps)

    frame_ids = meta.frame_ids
    noise_ids = meta.noise_ids
    G = int(frame_ids.max().item()) + 1

    # Build the cache from the clean frames (one layer). Use a large eviction
    # window (populate-then-query test order) and pass the real lookup window to
    # windowed_state — rollout instead adds+queries incrementally per step.
    cache = ARLinearStateCache(num_layers=1, window=10_000)
    for g in range(G):
        sel = (noise_ids == 1) & (frame_ids == g)
        if not bool(sel.any()):
            continue
        idx = sel.nonzero(as_tuple=False).squeeze(-1)
        S, z = clean_state_from_tokens(
            tk[:, :, idx, :], v[:, :, idx, :], pk[:, :, idx, :]
        )
        cache.update(0, g, S, z, is_pred=False)

    # For each noisy frame, the cache-based inference must reproduce the kernel rows.
    for f in range(G):
        qsel = (noise_ids == 0) & (frame_ids == f)
        if not bool(qsel.any()):
            continue
        qidx = qsel.nonzero(as_tuple=False).squeeze(-1)
        win = cache.windowed_state(0, f, hi_inclusive=f - 1, window=window)
        got = ar_inference_attn(
            tq[:, :, qidx, :],
            pq[:, :, qidx, :],
            win,
            tk[:, :, qidx, :],  # within-frame noise block = this chunk's own noisy keys
            pk[:, :, qidx, :],
            v[:, :, qidx, :],
            eps=eps,
        )
        torch.testing.assert_close(got, full[:, :, qidx, :], atol=1e-9, rtol=1e-7)


def test_clear_pred_drops_predicted_only():
    cache = ARLinearStateCache(num_layers=1, window=99)
    S = torch.ones(1, 1, 4, 4)
    z = torch.ones(1, 1, 1, 4)
    cache.update(0, 0, S, z, is_pred=False)  # confirmed
    cache.update(0, 2, S * 2, z * 2, is_pred=True)  # predicted
    assert cache.windowed_state(0, 3, hi_inclusive=2) is not None
    cache.clear_pred()
    ws = cache.windowed_state(0, 3, hi_inclusive=2)
    # only the confirmed frame-0 state remains
    assert ws is not None
    torch.testing.assert_close(ws[0], S)


def test_window_evicts_old_frames():
    cache = ARLinearStateCache(num_layers=1, window=2)
    S = torch.ones(1, 1, 2, 2)
    z = torch.ones(1, 1, 1, 2)
    cache.update(0, 0, S, z, is_pred=False)
    cache.update(0, 1, S, z, is_pred=False)
    cache.update(0, 5, S, z, is_pred=False)  # frame 5 evicts frames < 3
    ws = cache.windowed_state(0, 5, hi_inclusive=5)
    assert ws is not None
    # only frame 5 survives (0 and 1 are < 5 - window=3)
    torch.testing.assert_close(ws[0], S)


def test_cache_frame_pop_restore_is_transactional():
    cache = ARLinearStateCache(num_layers=2, window=99)
    S = torch.ones(1, 1, 2, 2)
    z = torch.ones(1, 1, 1, 2)
    for layer in range(2):
        cache.update(layer, 1, S * (layer + 1), z, is_pred=False)
        cache.update(layer, 3, S * 10, z * 10, is_pred=True)

    snapshot = cache.pop_frame(1)
    assert [len(entries) for entries in snapshot] == [1, 1]
    assert cache.info()["frame_ids"] == [3]
    cache.restore_frame(snapshot)
    assert cache.info()["frame_ids"] == [1, 3]
    assert cache.info()["confirmed_entries"] == 2
    assert cache.info()["predicted_entries"] == 2


def test_cache_rejects_duplicate_layer_frame():
    cache = ARLinearStateCache(num_layers=1, window=99)
    S = torch.ones(1, 1, 2, 2)
    z = torch.ones(1, 1, 1, 2)
    cache.update(0, 1, S, z, is_pred=False)
    with pytest.raises(ValueError, match="already contains"):
        cache.update(0, 1, S * 2, z * 2, is_pred=False)

    state = cache.windowed_state(0, 2, hi_inclusive=1)
    assert state is not None
    torch.testing.assert_close(state[0], S)
