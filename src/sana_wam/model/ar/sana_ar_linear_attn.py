"""Structured linear-attention primitives for the SANA **autoregressive** path.

This is the AR (block-causal / diffusion-forcing) analogue of
:mod:`sana_wam.model.ar.sana_linear_attn`. It realizes the
LingBot-VA causal attention topology *exactly* under SANA's dual-track ReLU
linear attention, without ever materializing a dense ``(N, N)`` mask in the fast
path.

Why a new module
----------------
The LingBot-VA mask (reference: ``/mnt/cpfs/zch/lingbot-va/wan_va/modules/model.py``
``FlexAttnFunc._get_mask_mod``) operates on a *duplicated* sequence

    [ video_noisy | video_clean | action_noisy | action_clean ]

where each token carries

- ``frame_id``  — its chunk index with **modality parity**: a video token in
  chunk ``c`` gets ``frame_id = 2*c``; an action token in chunk ``c`` gets
  ``frame_id = 2*c + 1``. This single monotonic axis encodes the asymmetric
  video↔action visibility for free: action chunk ``c`` (``2c+1``) can causally
  see video chunk ``c`` (``2c <= 2c+1``), but video chunk ``c`` cannot see
  action chunk ``c`` (``2c+1 > 2c``).
- ``noise_id``  — ``0`` for the noisy copy, ``1`` for the clean (conditioning)
  copy.

and the visible set is

    mask = OR( clean2clean ∧ causal(f_kv <= f_q),
               noise2clean ∧ strict_causal(f_kv < f_q),
               noise2noise ∧ block_self(f_kv == f_q) )
         ∧ window(|f_q - f_kv| <= W)

Cross-sample (batch) isolation is handled by the real ``B`` dimension here — we
never flatten batch into the sequence the way LingBot's flex path does — so
``seq_id`` is unnecessary.

This mask is **not** monotonic block-causal in the sense
:func:`sana_linear_attn._mask_to_chunk_index` requires (the clean/noise
duplication + the block-diagonal noise self-term break rectangularity), so the
existing cumsum path always falls back to ``O(N²)`` *and* would be semantically
wrong if handed this mask. Hence a dedicated kernel.

The structured fast path
------------------------
Decompose every query's visible set into two additive pieces that each have a
cheap form under linear attention:

1. **Windowed clean history.** Clean keys are indexed by ``frame_id`` on a single
   axis. A query at frame ``f`` sees clean keys with ``frame_id`` in
   ``[f - W, hi]`` where ``hi = f`` for a clean query (inclusive causal) and
   ``hi = f - 1`` for a noisy query (strict causal). Accumulate per-frame clean
   states ``Sf[g] = Σ v ⊗ tilde_k`` (``d×d``) and ``zf[g] = Σ phi_k`` (``d``),
   prefix-sum them over ``g``, and read off any window as a difference of two
   prefix snapshots. This is ``O(G)`` states (``G`` = number of frames, i.e.
   ``2·num_chunks`` — *not* token count), the whole reason linear attention wins.

2. **Within-frame noise block.** A noisy query at frame ``f`` additionally
   attends, fully, to the noisy keys sharing ``frame_id == f`` (``noise2noise``).
   Because ``frame_id`` carries modality parity, "same frame" already restricts
   this to the query's own modality. It is a small dense linear attention over
   one chunk's worth of noisy tokens.

The numerator uses the RoPE-rotated ``tilde`` track and the denominator the
un-rotated ``phi`` track, exactly as
:func:`sana_linear_attn._expanded_linear_attn` — and crucially the *same*
visible-key set drives both, so ``num`` and ``denom`` are never windowed
independently.

:func:`_ar_expanded_reference` builds the exact LingBot dense mask and routes it
through the audited :func:`sana_linear_attn._expanded_linear_attn`; it is the
ground-truth oracle the fast path is unit-tested against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .sana_linear_attn import _expanded_linear_attn

__all__ = [
    "AR_LINEAR_ATTN_EPS",
    "ARSeqMeta",
    "build_ar_seq_meta",
    "ar_build_dense_mask",
    "_ar_expanded_reference",
    "_ar_chunked_linear_attn",
]


AR_LINEAR_ATTN_EPS: float = 1e-8


def _ar_attention_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the AR attention operator/state dtype for an input dtype.

    Production fp16/bf16 feature tracks are always accumulated in fp32.  Keeping
    fp32/fp64 callers unchanged preserves the high-precision oracle surface while
    making the mixed-precision training and cache paths share one definition.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


@dataclass(frozen=True)
class ARSeqMeta:
    """Per-token AR sequence metadata shared by the kernel and its oracle.

    Attributes
    ----------
    frame_ids
        ``(N,)`` long tensor. Chunk index with modality parity (video ``2c``,
        action ``2c+1``).
    noise_ids
        ``(N,)`` long tensor. ``0`` = noisy copy, ``1`` = clean copy.
    window
        Sliding-window half-width ``W`` in ``frame_id`` units (``|f_q - f_kv| <=
        W``). Use a value ``>= max(frame_ids)`` to disable windowing.
    num_frames
        ``G`` (= ``max(frame_ids) + 1``). Precomputed so the kernel never calls
        ``frame_ids.max().item()`` (a device sync) in the per-layer hot path.
        ``0`` means "not precomputed" → the kernel derives the plan once (the
        ad-hoc/test construction path).
    noisy_idx_per_frame / clean_idx_per_frame
        Length-``G`` tuples; entry ``g`` is the ``long`` index tensor of the
        noisy / clean tokens at frame ``g`` (or ``None`` if empty). Precomputed in
        :func:`build_ar_seq_meta` (on CPU, sync-free) so the kernel avoids
        ``bool(.any())`` / ``.nonzero()`` device syncs every frame, every layer.
    key_is_pad
        Optional ``(B, N)`` boolean tensor. ``True`` keys are excluded from both
        clean-history sufficient states and within-frame noisy attention. Query
        rows are retained; applying the same key mask at every layer prevents a
        padded query state from becoming visible to a later valid query.
    key_is_pad_validated
        ``True`` only when :func:`build_ar_seq_meta` has already performed the
        data-dependent all-padded-row check. This keeps the per-layer fast path
        free of a CUDA synchronization while retaining fail-closed validation
        for directly constructed test/ad-hoc metadata.
    """

    frame_ids: Tensor
    noise_ids: Tensor
    window: int
    num_frames: int = 0
    noisy_idx_per_frame: Tuple[Optional[Tensor], ...] = ()
    clean_idx_per_frame: Tuple[Optional[Tensor], ...] = ()
    key_is_pad: Optional[Tensor] = None
    key_is_pad_validated: bool = False


def build_ar_seq_meta(
    *,
    num_chunks: int,
    video_tokens_per_chunk: int,
    action_tokens_per_chunk: int,
    window: int,
    device: torch.device | str = "cpu",
    key_is_pad: Optional[Tensor] = None,
) -> ARSeqMeta:
    """Build :class:`ARSeqMeta` for the canonical duplicated layout.

    The token order is ``[v_noisy, v_clean, a_noisy, a_clean]`` with each segment
    laid out chunk-major (chunk 0's tokens, then chunk 1's, ...). This matches
    LingBot's ``forward_train`` concat order
    (``model.py`` ``torch.cat([latent_noisy, latent_clean, action_noisy, action_clean])``).

    Parameters
    ----------
    num_chunks
        Number of AR chunks ``C``.
    video_tokens_per_chunk
        Token count of one video chunk (``frame_chunk_size * H_lat * W_lat`` after
        patchify), assumed equal across chunks.
    action_tokens_per_chunk
        Token count of one action chunk (``frame_chunk_size * action_per_frame``),
        assumed equal across chunks.
    window
        Sliding-window half-width in ``frame_id`` units.
    """
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")

    # Build the static layout on CPU so the per-frame index plan can be derived
    # sync-free (the `.nonzero()` / `bool(.any())` below would force CUDA syncs on
    # device); the small tensors are moved to `device` once at the end.
    cpu = torch.device("cpu")
    chunk_idx = torch.arange(num_chunks, device=cpu, dtype=torch.long)
    video_frames = (chunk_idx * 2).repeat_interleave(video_tokens_per_chunk)  # 2c
    action_frames = (chunk_idx * 2 + 1).repeat_interleave(action_tokens_per_chunk)  # 2c+1

    frame_ids = torch.cat([video_frames, video_frames, action_frames, action_frames])

    nv = video_frames.numel()
    na = action_frames.numel()
    noise_ids = torch.cat(
        [
            torch.zeros(nv, device=cpu, dtype=torch.long),  # v_noisy
            torch.ones(nv, device=cpu, dtype=torch.long),  # v_clean
            torch.zeros(na, device=cpu, dtype=torch.long),  # a_noisy
            torch.ones(na, device=cpu, dtype=torch.long),  # a_clean
        ]
    )

    num_frames = int(frame_ids.max().item()) + 1  # CPU tensor → no GPU sync
    noisy_plan, clean_plan = _index_plan(frame_ids, noise_ids, num_frames)

    dev = torch.device(device)
    if key_is_pad is not None:
        if not isinstance(key_is_pad, Tensor):
            raise TypeError("key_is_pad must be a tensor or None")
        if key_is_pad.dtype != torch.bool or key_is_pad.layout != torch.strided:
            raise TypeError("key_is_pad must be a strided boolean tensor")
        if key_is_pad.ndim != 2 or key_is_pad.shape[1] != frame_ids.numel():
            raise ValueError(
                "key_is_pad must have shape [batch, tokens] with "
                f"tokens={frame_ids.numel()}, got {tuple(key_is_pad.shape)}"
            )
        if key_is_pad.device != dev:
            raise ValueError(
                f"key_is_pad must be on {dev}, got {key_is_pad.device}"
            )
        if bool(key_is_pad.all(dim=1).any()):
            raise ValueError("every sample needs at least one non-padded AR key")
        key_is_pad = key_is_pad.detach()
    return ARSeqMeta(
        frame_ids=frame_ids.to(dev),
        noise_ids=noise_ids.to(dev),
        window=int(window),
        num_frames=num_frames,
        noisy_idx_per_frame=tuple(None if t is None else t.to(dev) for t in noisy_plan),
        clean_idx_per_frame=tuple(None if t is None else t.to(dev) for t in clean_plan),
        key_is_pad=key_is_pad,
        key_is_pad_validated=key_is_pad is not None,
    )


def _index_plan(
    frame_ids: Tensor, noise_ids: Tensor, num_frames: int
) -> Tuple[Tuple[Optional[Tensor], ...], Tuple[Optional[Tensor], ...]]:
    """Per-frame noisy / clean token index lists (``None`` where empty).

    Intended to run on CPU tensors (sync-free); the kernel falls back to calling
    it on the input device only for ad-hoc :class:`ARSeqMeta` built without a plan.
    """
    noisy: list[Optional[Tensor]] = []
    clean: list[Optional[Tensor]] = []
    for g in range(num_frames):
        at = frame_ids == g
        ni = at & (noise_ids == 0)
        ci = at & (noise_ids == 1)
        noisy.append(ni.nonzero(as_tuple=False).squeeze(-1) if bool(ni.any()) else None)
        clean.append(ci.nonzero(as_tuple=False).squeeze(-1) if bool(ci.any()) else None)
    return tuple(noisy), tuple(clean)


def _resolve_index_plan(
    meta: ARSeqMeta, device: torch.device, n_tokens: int
) -> Tuple[Tuple[Optional[Tensor], ...], Tuple[Optional[Tensor], ...], int]:
    """Return ``(noisy_per_frame, clean_per_frame, G)`` for ``meta`` on ``device``.

    Fast path: reuse the plan precomputed by :func:`build_ar_seq_meta` (``.to`` is
    a no-op when already on ``device``). Fallback: derive it once from
    ``frame_ids`` / ``noise_ids`` (ad-hoc meta, e.g. a direct ``ARSeqMeta(...)``
    in a test) — this is the only place the old per-call sync remains.
    """
    if meta.num_frames > 0 and (meta.noisy_idx_per_frame or meta.clean_idx_per_frame):
        noisy = tuple(None if t is None else t.to(device) for t in meta.noisy_idx_per_frame)
        clean = tuple(None if t is None else t.to(device) for t in meta.clean_idx_per_frame)
        return noisy, clean, int(meta.num_frames)

    fids = meta.frame_ids.to(device)
    nids = meta.noise_ids.to(device)
    G = int(fids.max().item()) + 1
    noisy, clean = _index_plan(fids, nids, G)
    return noisy, clean, G


def _resolve_key_is_pad(
    meta: ARSeqMeta,
    *,
    device: torch.device,
    n_tokens: int,
    batch_size: Optional[int] = None,
) -> Optional[Tensor]:
    """Validate and return the dynamic batch key-padding mask."""
    key_is_pad = meta.key_is_pad
    if key_is_pad is None:
        return None
    if not isinstance(key_is_pad, Tensor):
        raise TypeError("meta.key_is_pad must be a tensor or None")
    if key_is_pad.dtype != torch.bool or key_is_pad.layout != torch.strided:
        raise TypeError("meta.key_is_pad must be a strided boolean tensor")
    if key_is_pad.ndim != 2:
        raise ValueError(
            "meta.key_is_pad must have shape [batch, tokens], "
            f"got {tuple(key_is_pad.shape)}"
        )
    expected = (
        (batch_size, n_tokens)
        if batch_size is not None
        else (key_is_pad.shape[0], n_tokens)
    )
    if tuple(key_is_pad.shape) != expected:
        raise ValueError(
            f"meta.key_is_pad must have shape {expected}, got {tuple(key_is_pad.shape)}"
        )
    if key_is_pad.device != device:
        raise ValueError(
            f"meta.key_is_pad must be on {device}, got {key_is_pad.device}"
        )
    if type(meta.key_is_pad_validated) is not bool:
        raise TypeError("meta.key_is_pad_validated must be boolean")
    if not meta.key_is_pad_validated and bool(key_is_pad.all(dim=1).any()):
        raise ValueError("every sample needs at least one non-padded AR key")
    return key_is_pad.detach()


def ar_build_dense_mask(meta: ARSeqMeta) -> Tensor:
    """Materialize the exact LingBot-VA ``(N, N)`` bool attention mask.

    ``mask[i, j] = True`` iff query ``i`` may attend to key ``j`` under

        ( clean2clean ∧ f_j <= f_i )
      ∨ ( noise2clean ∧ f_j <  f_i )
      ∨ ( noise2noise ∧ f_j == f_i )
      ∧ |f_i - f_j| <= W

    Ported verbatim from ``FlexAttnFunc._get_mask_mod`` (LingBot ``model.py``).
    """
    f = meta.frame_ids
    n = meta.noise_ids
    fi = f.unsqueeze(-1)  # (N, 1) query
    fj = f.unsqueeze(0)  # (1, N) key
    ni = n.unsqueeze(-1)
    nj = n.unsqueeze(0)

    clean = ni.bool() & nj.bool()
    noisy_q = ~ni.bool()
    clean_k = nj.bool()
    noisy_k = ~nj.bool()

    clean2clean = clean & (fj <= fi)
    noise2clean = noisy_q & clean_k & (fj < fi)
    noise2noise = noisy_q & noisy_k & (fj == fi)

    base = clean2clean | noise2clean | noise2noise
    window_ok = (fi - fj).abs() <= meta.window
    base = base & window_ok
    key_is_pad = _resolve_key_is_pad(
        meta,
        device=f.device,
        n_tokens=f.numel(),
    )
    if key_is_pad is None:
        return base
    # Keep every query row, but exclude padded keys. The singleton head axis
    # makes this directly broadcastable to (B, H, N, N).
    return base.unsqueeze(0).unsqueeze(1) & (~key_is_pad[:, None, None, :])


def _ar_expanded_reference(
    tilde_q: Tensor,
    tilde_k: Tensor,
    v: Tensor,
    phi_q: Tensor,
    phi_k: Tensor,
    meta: ARSeqMeta,
    eps: float = AR_LINEAR_ATTN_EPS,
) -> Tensor:
    """Ground-truth oracle: dense LingBot mask through the audited expanded path.

    ``O(N²)`` — for tests and as the fallback the fast path is validated against.
    """
    key_is_pad = _resolve_key_is_pad(
        meta,
        device=tilde_q.device,
        n_tokens=tilde_q.shape[-2],
        batch_size=tilde_q.shape[0],
    )
    if key_is_pad is not None:
        pad = key_is_pad[:, None, :, None]
        # A zero attention score is not sufficient when a padded K/V contains
        # NaN: 0 * NaN remains NaN. Sanitize padded keys before every reduction.
        tilde_k = tilde_k.masked_fill(pad, 0)
        phi_k = phi_k.masked_fill(pad, 0)
        v = v.masked_fill(pad, 0)
    mask = ar_build_dense_mask(meta).to(tilde_q.device)
    return _expanded_linear_attn(tilde_q, tilde_k, v, phi_q, phi_k, mask=mask, eps=eps)


def _per_frame_clean_states(
    tilde_k: Tensor,
    v: Tensor,
    phi_k: Tensor,
    clean_idx_per_frame: Tuple[Optional[Tensor], ...],
    num_frames: int,
    key_is_pad: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Accumulate clean-key dual-track states bucketed by ``frame_id``.

    ``clean_idx_per_frame[g]`` is the precomputed index tensor of clean tokens at
    frame ``g`` (``None`` if empty) — no per-frame mask/``nonzero`` (= no device
    sync) here.

    Returns
    -------
    Sf : ``(G, B, H, d, d)``  where ``Sf[g] = Σ_{clean key j @ frame g} v_j ⊗ tilde_k_j``
    zf : ``(G, B, H, 1, d)``  where ``zf[g] = Σ_{clean key j @ frame g} phi_k_j``
    """
    B, H, _, d = tilde_k.shape
    Sf = torch.zeros(num_frames, B, H, d, d, dtype=v.dtype, device=v.device)
    zf = torch.zeros(num_frames, B, H, 1, d, dtype=v.dtype, device=v.device)

    for g in range(num_frames):
        idx = clean_idx_per_frame[g]
        if idx is None:
            continue
        tk = tilde_k[:, :, idx, :]
        vv = v[:, :, idx, :]
        pk = phi_k[:, :, idx, :]
        if key_is_pad is not None:
            pad = key_is_pad[:, idx][:, None, :, None]
            tk = tk.masked_fill(pad, 0)
            pk = pk.masked_fill(pad, 0)
            vv = vv.masked_fill(pad, 0)
        Sf[g] = vv.transpose(-1, -2) @ tk  # (B, H, d, d)
        zf[g] = pk.sum(dim=-2, keepdim=True)  # (B, H, 1, d)
    return Sf, zf


def _ar_chunked_linear_attn(
    tilde_q: Tensor,
    tilde_k: Tensor,
    v: Tensor,
    phi_q: Tensor,
    phi_k: Tensor,
    meta: ARSeqMeta,
    eps: float = AR_LINEAR_ATTN_EPS,
    *,
    action_video_memory_adapter=None,
    layer_id: Optional[int] = None,
    return_action_video_numerators: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    """``O(N · d²)`` structured kernel — equivalent to :func:`_ar_expanded_reference`.

    See module docstring for the windowed-clean + within-frame-noise
    decomposition. All inputs are ``(B, H, N, d)``.
    """
    if tilde_q.dim() != 4 or tilde_k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "_ar_chunked_linear_attn expects (B, H, N, d) inputs; got "
            f"tilde_q={tuple(tilde_q.shape)}, tilde_k={tuple(tilde_k.shape)}, v={tuple(v.shape)}."
        )
    if phi_q.shape != tilde_q.shape or phi_k.shape != tilde_k.shape:
        raise ValueError("phi_q/phi_k must match tilde_q/tilde_k shapes.")
    if action_video_memory_adapter is not None and layer_id is None:
        raise ValueError("an action video-memory adapter requires layer_id")
    if type(return_action_video_numerators) is not bool:
        raise TypeError("return_action_video_numerators must be boolean")

    B, H, N, d = tilde_q.shape
    orig_dtype = v.dtype
    compute_dtype = _ar_attention_compute_dtype(orig_dtype)
    if compute_dtype != orig_dtype:
        # Feature mapping (ReLU/RoPE) happens in the caller.  From this boundary
        # onward every operator and sufficient state is fp32 for fp16/bf16 runs:
        # per-frame S/z, their prefixes, query-side num/denom, and the noisy
        # within-frame block.  Cast only the public result back at the end.
        tilde_q = tilde_q.to(compute_dtype)
        tilde_k = tilde_k.to(compute_dtype)
        v = v.to(compute_dtype)
        phi_q = phi_q.to(compute_dtype)
        phi_k = phi_k.to(compute_dtype)
    frame_ids = meta.frame_ids
    noise_ids = meta.noise_ids
    W = int(meta.window)
    if frame_ids.shape != (N,) or noise_ids.shape != (N,):
        raise ValueError(
            f"meta.frame_ids/noise_ids must be ({N},); got {tuple(frame_ids.shape)} / "
            f"{tuple(noise_ids.shape)}."
        )

    # Precomputed per-frame index plan (sync-free in the hot path; see
    # build_ar_seq_meta). noisy_idx[g] doubles as the within-frame noise-block keys.
    noisy_idx, clean_idx, G = _resolve_index_plan(meta, tilde_q.device, N)
    key_is_pad = _resolve_key_is_pad(
        meta,
        device=tilde_q.device,
        n_tokens=N,
        batch_size=B,
    )

    # Per-frame clean states and their inclusive prefix sums.
    # Pcum_S[g] = Σ_{g' < g} Sf[g']  (so Pcum_S[hi+1] - Pcum_S[lo] = sum over [lo, hi]).
    Sf, zf = _per_frame_clean_states(
        tilde_k, v, phi_k, clean_idx, G, key_is_pad
    )
    Pcum_S = torch.zeros(G + 1, B, H, d, d, dtype=v.dtype, device=v.device)
    Pcum_z = torch.zeros(G + 1, B, H, 1, d, dtype=v.dtype, device=v.device)
    Pcum_S[1:] = torch.cumsum(Sf, dim=0)
    Pcum_z[1:] = torch.cumsum(zf, dim=0)

    # The canonical prefix above remains untouched.  The extra prefix is only a
    # read-side view over even frame ids, used to compute an action-query delta.
    # Keeping the original reduction is what makes identity-init endpoints exact.
    video_Pcum_S: Optional[Tensor] = None
    video_Pcum_z: Optional[Tensor] = None
    if action_video_memory_adapter is not None or return_action_video_numerators:
        video_frame = (
            torch.arange(G, device=v.device, dtype=torch.long) % 2 == 0
        )
        video_Sf = Sf * video_frame.view(G, 1, 1, 1, 1)
        video_zf = zf * video_frame.view(G, 1, 1, 1, 1)
        video_Pcum_S = torch.zeros_like(Pcum_S)
        video_Pcum_z = torch.zeros_like(Pcum_z)
        video_Pcum_S[1:] = torch.cumsum(video_Sf, dim=0)
        video_Pcum_z[1:] = torch.cumsum(video_zf, dim=0)

    out = torch.empty_like(v)
    action_video_numerators: list[Tensor] = []

    def _window_state(lo: int, hi: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Σ over clean frames in [lo, hi] (inclusive); None if empty."""
        if hi < lo:
            return None
        lo = max(lo, 0)
        if hi < lo:
            return None
        s = Pcum_S[hi + 1] - Pcum_S[lo]  # (B, H, d, d)
        z = Pcum_z[hi + 1] - Pcum_z[lo]  # (B, H, 1, d)
        return s, z

    def _video_window_state(
        lo: int, hi: int
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Even-frame part of the same clean-history window."""
        if video_Pcum_S is None or video_Pcum_z is None or hi < lo:
            return None
        lo = max(lo, 0)
        if hi < lo:
            return None
        s = video_Pcum_S[hi + 1] - video_Pcum_S[lo]
        z = video_Pcum_z[hi + 1] - video_Pcum_z[lo]
        return s, z

    for g in range(G):
        if noisy_idx[g] is None and clean_idx[g] is None:
            continue
        lo = g - W
        for noise_flag in (0, 1):
            qidx = noisy_idx[g] if noise_flag == 0 else clean_idx[g]
            if qidx is None:
                continue
            tq = tilde_q[:, :, qidx, :]
            pq = phi_q[:, :, qidx, :]

            # 1. windowed clean history (inclusive hi=g for clean queries, strict
            #    hi=g-1 for noisy queries).
            hi = g if noise_flag == 1 else g - 1
            ws = _window_state(lo, hi)
            video_ws = None
            if (
                (action_video_memory_adapter is not None or return_action_video_numerators)
                and g % 2 == 1
                and noise_flag == 0
            ):
                video_ws = _video_window_state(lo, hi)
                if video_ws is not None:
                    if ws is None:
                        raise RuntimeError(
                            "video window exists while canonical window is empty"
                        )
                    if return_action_video_numerators:
                        action_video_numerators.append(
                            tq @ video_ws[0].transpose(-1, -2)
                        )
                    if action_video_memory_adapter is not None:
                        delta_s, delta_z = (
                            action_video_memory_adapter.forward_delta(
                                int(layer_id), *video_ws
                            )
                        )
                        ws = (ws[0] + delta_s, ws[1] + delta_z)
            if ws is not None:
                s_win, z_win = ws
                num = tq @ s_win.transpose(-1, -2)  # (B, H, |q|, d)
                denom = pq @ z_win.transpose(-1, -2)  # (B, H, |q|, 1)
            else:
                num = torch.zeros(B, H, qidx.numel(), d, dtype=v.dtype, device=v.device)
                denom = torch.zeros(B, H, qidx.numel(), 1, dtype=v.dtype, device=v.device)

            # 2. within-frame noise block (noisy queries only). frame_id parity
            #    already restricts these keys to the query's own modality; the
            #    within-frame noisy keys ARE the noisy queries (qidx).
            if noise_flag == 0:
                kidx = qidx
                tk = tilde_k[:, :, kidx, :]
                pk = phi_k[:, :, kidx, :]
                vv = v[:, :, kidx, :]
                if key_is_pad is not None:
                    pad = key_is_pad[:, kidx][:, None, :, None]
                    tk = tk.masked_fill(pad, 0)
                    pk = pk.masked_fill(pad, 0)
                    vv = vv.masked_fill(pad, 0)
                a = tq @ tk.transpose(-1, -2)  # (B, H, |q|, |k|)
                bb = pq @ pk.transpose(-1, -2)  # (B, H, |q|, |k|)
                num = num + a @ vv
                denom = denom + bb.sum(dim=-1, keepdim=True)

            out[:, :, qidx, :] = num / (denom + eps)

    result = out.to(orig_dtype) if compute_dtype != orig_dtype else out
    if not return_action_video_numerators:
        return result
    expected_chunks = (G + 1) // 2
    if len(action_video_numerators) != expected_chunks:
        raise RuntimeError(
            "action-facing numerator capture did not cover every action chunk: "
            f"expected {expected_chunks}, got {len(action_video_numerators)}"
        )
    query_widths = {int(tensor.shape[-2]) for tensor in action_video_numerators}
    if len(query_widths) != 1:
        raise RuntimeError(
            "action-facing numerator chunks have inconsistent query widths"
        )
    numerators = torch.stack(action_video_numerators, dim=1)
    if compute_dtype != orig_dtype:
        numerators = numerators.to(orig_dtype)
    return result, numerators
