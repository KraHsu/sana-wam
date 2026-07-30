"""SanaMoTJointDriver: MoT driver for SANA-style linear-attention backbones.

Replaces the SDPA-based :meth:`MoTJointDriver._mixed_attention` with a cumsum
expansion of SANA's ReLU-kernel linear attention, preserving the pretrained
dual-track normalization (rotated phi(q)/phi(k) in the numerator, *un*-rotated
phi(q)/phi(k) in the denominator) while supporting the joint mask topology.

Two extensions of the base contract:

1. Override :meth:`_step_impl` so the unrotated ReLU'd Q/K (sitting in each
   backbone's ``post_state`` dict under ``"q_unrot"`` / ``"k_unrot"``) can be
   threaded into the mixed-attention call. The base class only passes
   ``(q_cat, k_cat, v_cat, attn_mask)``, which is fine for SDPA but loses
   the second-track inputs the SANA math needs.
2. Override :meth:`_mixed_attention` with kw-only ``phi_q`` / ``phi_k`` /
   ``use_ckpt`` extensions. This stays Liskov-compatible with the base
   signature (callers passing only the four positional args still get a
   well-defined error path, see :meth:`_mixed_attention`).

Everything else — ``_build_joint_mask``, ``_build_attention_mask``,
``_video_tokens_per_frame``, ``run_joint_loop``, ``step``, ``_step_checkpointed``
— is inherited unchanged.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor

from sana_wam.model.ar.mot_driver import MoTJointDriver
from sana_wam.model.ar.sana_linear_attn import (
    _chunked_linear_attn,
    _chunked_linear_attn_checkpointed,
    _expanded_linear_attn,
    _mask_to_chunk_index,
)

if TYPE_CHECKING:
    from sana_wam.model.action_backbone.backbone import ActionBackbone
    from sana_wam.model.base import ActionState
    from sana_wam.model.video_backbone.adapter import BlockLoopState, VideoBackbone

logger = logging.getLogger(__name__)


class SanaMoTJointDriver(MoTJointDriver):
    """MoT driver for SANA-style ReLU-kernel linear attention.

    Strict kernel alignment is required at construction time: both
    ``vb.attn_kernel`` and ``ab.attn_kernel`` must be ``"linear_relu"``.
    A mixed-kernel pair (e.g. SANA video + softmax action) is rejected
    because the cross-modality inner products
    ``tilde_q_v · tilde_k_a`` / ``tilde_q_a · tilde_k_v`` have no meaning
    if one side was kernel-applied and the other wasn't.
    """

    # Denominator floor for the per-modality action-row rebalance (see
    # _rebalance_action_rows). Large enough to stop the few-key action
    # denominator from exploding the ratio, small enough to be a no-op once
    # features are trained.
    _DEN_FLOOR: float = 1e-4

    def __init__(
        self,
        vb: "VideoBackbone",
        ab: "ActionBackbone",
        *,
        eps: float = 1e-15,
        action_self_attn_weight: Optional[float] = None,
        action_self_attn_mode: str = "linear",
        action_video_attn_mode: str = "linear",
        **kw,
    ) -> None:
        super().__init__(vb, ab, **kw)
        self.eps = float(eps)
        # How the action-self track (action queries over action keys) is
        # computed in :meth:`_rebalance_action_rows`:
        #   "linear"  — SANA's ReLU-kernel linear attention, normalized over
        #               action keys only (the per-modality renorm).
        #   "softmax" — exact softmax over the action-key score matrix. Linear
        #               attention is low-rank and cannot concentrate on a few
        #               keys; softmax can (sharp retrieval), which action
        #               self-attention may need. The video-context track stays
        #               linear either way. Only active when action_self_attn_weight
        #               is set.
        if action_self_attn_mode not in ("linear", "softmax"):
            raise ValueError(
                f"action_self_attn_mode must be 'linear' or 'softmax', got {action_self_attn_mode!r}."
            )
        self.action_self_attn_mode: str = action_self_attn_mode
        # Same choice for the video-context track (action queries over video
        # keys). Default "linear" keeps the SANA dual-track grounding; "softmax"
        # lets the action token sharply attend to the relevant spatial region of
        # the current frame (linear attention cannot concentrate). Kept separate
        # from action_self_attn_mode so the self-only-softmax variants stay
        # reproducible. Affordable: the (Sa × Sv) score matrix has only ~tens of
        # query rows even though Sv ~ thousands.
        if action_video_attn_mode not in ("linear", "softmax"):
            raise ValueError(
                f"action_video_attn_mode must be 'linear' or 'softmax', got {action_video_attn_mode!r}."
            )
        self.action_video_attn_mode: str = action_video_attn_mode
        # Token-imbalance fix. SANA's single-track linear-attention denominator
        # for an action-query row sums phi(q)·phi(k) over *all* keys it sees —
        # i.e. ~thousands of video tokens plus a few dozen action tokens. Since
        # the ReLU features are non-negative, the video keys dominate the
        # normalization and action↔action self-attention is washed out (the
        # core "why SANA is bad" failure mode). When this weight is set, action
        # rows are recomputed with a per-modality normalized two-track mix:
        #
        #     out_a = (1-w)·(num_video / den_video) + w·(num_action / den_action)
        #
        # so the action-self track is normalized over action keys *only* and no
        # longer drowned by token count. Video-query rows are left byte-for-byte
        # identical to the pretrained SANA path. ``None`` => original behavior.
        self.action_self_attn_weight: Optional[float] = (
            None if action_self_attn_weight is None else float(action_self_attn_weight)
        )
        # `run_joint_loop` builds attn_mask once and reuses across all N layers,
        # so the chunk-index translation (O(N) row-scan over an N×N bool mask;
        # N can be ~32k for 81f × 480p joint masks) is identical every layer.
        # Cache by id() to skip the rescan within one forward.
        #
        # Cache lifetime is bounded to a single ``run_joint_loop`` call:
        # ``run_joint_loop`` clears the cache on entry below, so Python's
        # id-recycling (a freed mask's id reassigned to a new mask of the
        # same N but different topology, e.g. switching
        # ``attention_mask_mode``) cannot deliver a stale ``chunk_index`` to
        # a subsequent forward.
        self._chunk_cache_key: Optional[int] = None
        self._chunk_cache_value: Optional[List[int]] = None

        v_kernel = getattr(vb, "attn_kernel", "softmax")
        a_kernel = getattr(ab, "attn_kernel", "softmax")
        if v_kernel != "linear_relu":
            raise ValueError(
                f"SanaMoTJointDriver requires video_backbone.attn_kernel='linear_relu', "
                f"got '{v_kernel}'. The SANA linear-attn path cannot be combined with "
                "a softmax video backbone — use MoTJointDriver instead."
            )
        if a_kernel != "linear_relu":
            raise ValueError(
                f"SanaMoTJointDriver requires action_backbone.attn_kernel='linear_relu', "
                f"got '{a_kernel}'. Cross-modality inner products require kernel-aligned "
                "Q/K on both sides; set ActionDiT(attn_kernel='linear_relu') in the config."
            )
        video_aligned = bool(
            getattr(vb, "aligned_positive_rope_kernel", False)
        )
        action_aligned = bool(
            getattr(ab, "aligned_positive_rope_kernel", False)
        )
        if video_aligned != action_aligned:
            raise ValueError(
                "aligned_positive_rope_kernel must be enabled on both video "
                "and action streams or neither"
            )
        self.aligned_positive_rope_kernel = video_aligned
        video_mode = getattr(vb, "aligned_positive_rope_mode", None)
        action_mode = getattr(ab, "aligned_positive_rope_mode", None)
        if video_aligned and video_mode != action_mode:
            raise ValueError(
                "aligned_positive_rope_mode must match on video and action streams"
            )
        self.aligned_positive_rope_mode = video_mode if video_aligned else None

    def run_joint_loop(self, *args, **kwargs):
        """Reset the chunk-index cache, then delegate to the base loop.

        The cache key is ``id(attn_mask)``; Python is free to reuse the id
        of a previously-freed mask for a new mask of the same shape but
        different topology (e.g. an ``attention_mask_mode`` change between
        forwards). Clearing the cache here bounds its lifetime to one
        ``run_joint_loop`` invocation, which is exactly the window over
        which the mask object is guaranteed live and reused by all layers.
        Within a single loop the cache is still a per-layer O(1) hit; the
        first-layer rescan cost is unavoidable (we have to materialize the
        chunk boundary somewhere) and unchanged.
        """
        self._chunk_cache_key = None
        self._chunk_cache_value = None
        return super().run_joint_loop(*args, **kwargs)

    def _step_impl(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        attn_mask: Optional[Tensor] = None,
        *,
        suppress_inner_attn_ckpt: bool = False,
    ) -> Tuple["BlockLoopState", "ActionState"] | tuple[
        "BlockLoopState", "ActionState", Tensor
    ]:
        """Override of :meth:`MoTJointDriver._step_impl`.

        Mirrors the parent's structure (pre_attn → concat → mixed → split →
        post_attn), but additionally threads ``post_state["q_unrot"]`` /
        ``["k_unrot"]`` from each backbone into the SANA linear-attn call.
        The parent's INVARIANT (only ``vstate.x`` and ``astate.payload.x_action``
        may be mutated; layer-invariant fields stay shared by reference) is
        preserved.
        """
        vb = self.vb
        ab = self.ab

        q_v, k_v, v_v, vpost = vb.pre_attn_at_layer(layer_id, vstate)
        q_a, k_a, v_a, apost = ab.pre_attn_at_layer(layer_id, astate)

        if q_v.dtype != q_a.dtype:
            raise RuntimeError(
                f"SanaMoTJointDriver: dtype mismatch at layer {layer_id} "
                f"(video={q_v.dtype}, action={q_a.dtype}). Both backbones "
                "must produce attention inputs in matching dtype."
            )
        if q_v.device != q_a.device:
            raise RuntimeError(
                f"SanaMoTJointDriver: device mismatch at layer {layer_id} "
                f"(video={q_v.device}, action={q_a.device})."
            )
        if not (vpost.get("uses_linear_attn") and apost.get("uses_linear_attn")):
            raise RuntimeError(
                "SanaMoTJointDriver expected both backbones to publish "
                "uses_linear_attn=True in post_state. Got "
                f"video={vpost.get('uses_linear_attn')}, action={apost.get('uses_linear_attn')}. "
                "Check that the backbones implement the linear-attn pre_attn_at_layer "
                "contract (see sana_wam/model/video_backbone/sana/blocks_split.py)."
            )

        s_video = q_v.shape[1]
        s_action = q_a.shape[1]
        q_cat = torch.cat([q_v, q_a], dim=1)
        k_cat = torch.cat([k_v, k_a], dim=1)
        v_cat = torch.cat([v_v, v_a], dim=1)
        phi_q = torch.cat([vpost["q_unrot"], apost["q_unrot"]], dim=1)
        phi_k = torch.cat([vpost["k_unrot"], apost["k_unrot"]], dim=1)
        # `run_joint_loop` pre-builds ``attn_mask`` and reuses it across layers;
        # SanaMoTJointDriver follows the same contract — _step_impl does NOT
        # rebuild the mask itself.

        use_ckpt = (
            self.mot_checkpoint_mixed_attn
            and ab.training
            and not suppress_inner_attn_ckpt
        )

        mixed_result = self._mixed_attention(
            q_cat,
            k_cat,
            v_cat,
            attn_mask,
            phi_q=phi_q,
            phi_k=phi_k,
            use_ckpt=use_ckpt,
            s_video=s_video,
            s_action=s_action,
        )
        attention_aux = None
        if isinstance(mixed_result, tuple):
            if len(mixed_result) != 2:
                raise RuntimeError(
                    "mixed attention auxiliary return must be (output, auxiliary)"
                )
            mixed, attention_aux = mixed_result
            if not isinstance(mixed, Tensor) or not isinstance(attention_aux, Tensor):
                raise TypeError("mixed attention output and auxiliary must be tensors")
        else:
            mixed = mixed_result

        attn_v, attn_a = mixed.split([s_video, s_action], dim=1)
        vstate = vb.post_attn_at_layer(layer_id, vstate, attn_v.contiguous(), vpost)
        astate = ab.post_attn_at_layer(layer_id, astate, attn_a.contiguous(), apost)
        if attention_aux is not None:
            return vstate, astate, attention_aux
        return vstate, astate

    def _mixed_attention(  # type: ignore[override]
        self,
        q_cat: Tensor,
        k_cat: Tensor,
        v_cat: Tensor,
        attn_mask: Optional[Tensor],
        *,
        phi_q: Optional[Tensor] = None,
        phi_k: Optional[Tensor] = None,
        use_ckpt: bool = False,
        s_video: Optional[int] = None,
        s_action: Optional[int] = None,
    ) -> Tensor:
        """SANA cumsum-linear-attention replacement for SDPA.

        Layout pivot ``(B, S, H*D) ↔ (B, H, S, d)`` happens here so the
        underlying primitives (:func:`_chunked_linear_attn`,
        :func:`_expanded_linear_attn`) can stay in the upstream-SANA layout
        without forcing every caller to rearrange.

        Dispatch rules:

        - If ``_mask_to_chunk_index(attn_mask)`` returns a non-``None`` chunk
          boundary list, use the cumsum fast path
          (:func:`_chunked_linear_attn` or, when ``use_ckpt`` is true,
          :func:`_chunked_linear_attn_checkpointed` for per-chunk gradient
          checkpointing).
        - Otherwise (mask doesn't factorize into monotonic block-causal
          chunks, or ``attn_mask is None``), fall back to
          :func:`_expanded_linear_attn`. When ``use_ckpt`` is true, wrap that
          single call in :func:`torch.utils.checkpoint.checkpoint` to match
          the memory profile of the base SDPA path's ``mot_checkpoint_mixed_attn``.

        ``phi_q`` / ``phi_k`` carry the unrotated ReLU'd Q/K (the second
        track required for SANA's dual-track denominator). If either is
        ``None``, we raise immediately rather than silently falling back to
        SDPA — that downgrade would produce mathematically meaningless output
        and mask a real bug in the caller.
        """
        if phi_q is None or phi_k is None:
            raise RuntimeError(
                "SanaMoTJointDriver._mixed_attention requires phi_q and phi_k "
                "(unrotated ReLU'd Q/K from post_state). Got "
                f"phi_q={'set' if phi_q is not None else 'None'}, "
                f"phi_k={'set' if phi_k is not None else 'None'}. "
                "This usually means _step_impl was bypassed by a direct SDPA-style "
                "call site that doesn't know about the dual-track contract."
            )

        n = self.num_heads
        tilde_q = rearrange(q_cat, "b s (n d) -> b n s d", n=n)
        tilde_k = rearrange(k_cat, "b s (n d) -> b n s d", n=n)
        v = rearrange(v_cat, "b s (n d) -> b n s d", n=n)
        pq = rearrange(phi_q, "b s (n d) -> b n s d", n=n)
        pk = rearrange(phi_k, "b s (n d) -> b n s d", n=n)

        if attn_mask is None:
            # Bidirectional: every row attends to every column. Equivalent to
            # a single chunk spanning [0, N] in the block-causal cumsum form
            # (state accumulates over all keys before any query reads it).
            # Routing through the chunked path keeps memory at O(N · d²)
            # instead of the O(N²) expanded fallback — critical for the long
            # 32k-token joint sequences where bidirectional was the original
            # motivation for SANA's linear attention.
            n_tokens = tilde_q.shape[-2]
            chunk_index: Optional[List[int]] = [0, int(n_tokens)]
        elif self._chunk_cache_key == id(attn_mask):
            chunk_index = self._chunk_cache_value
        else:
            chunk_index = _mask_to_chunk_index(attn_mask)
            self._chunk_cache_key = id(attn_mask)
            self._chunk_cache_value = chunk_index

        if chunk_index is not None:
            attn_fn = _chunked_linear_attn_checkpointed if use_ckpt else _chunked_linear_attn
            out = attn_fn(tilde_q, tilde_k, v, pq, pk, chunk_index, eps=self.eps)
        else:
            if use_ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    _expanded_linear_attn,
                    tilde_q,
                    tilde_k,
                    v,
                    pq,
                    pk,
                    attn_mask,
                    self.eps,
                    use_reentrant=False,
                )
            else:
                out = _expanded_linear_attn(
                    tilde_q, tilde_k, v, pq, pk, mask=attn_mask, eps=self.eps
                )

        out = self._rebalance_action_rows(
            out, tilde_q, tilde_k, v, pq, pk, s_video=s_video, s_action=s_action
        )

        return rearrange(out, "b n s d -> b s (n d)", n=n)

    def _rebalance_action_rows(
        self,
        out: Tensor,
        tilde_q: Tensor,
        tilde_k: Tensor,
        v: Tensor,
        pq: Tensor,
        pk: Tensor,
        *,
        s_video: Optional[int],
        s_action: Optional[int],
    ) -> Tensor:
        """Replace action-query rows of ``out`` with a per-modality two-track mix.

        No-op (returns ``out`` unchanged) unless ``action_self_attn_weight`` is
        set and there is at least one action token. All tensors are in the
        ``(B, H, N, d)`` upstream-SANA layout. Action queries see *all* keys
        unmasked (mask topology: ``[Sv:, :Sv]`` and ``[Sv:, Sv:]`` are fully
        True — see ``MoTJointDriver._build_joint_mask``), so the per-modality
        sums below are exact dense reductions with no sub-mask to apply.

        Video-query rows (``out[:, :, :Sv]``) are returned untouched, keeping
        the pretrained SANA video path byte-for-byte identical.
        """
        w = self.action_self_attn_weight
        if w is None or not s_action or s_action <= 0:
            return out

        n_tokens = tilde_q.shape[-2]
        sv = int(s_video) if s_video is not None else int(n_tokens - s_action)
        if sv <= 0 or sv >= n_tokens:
            # Degenerate split (no video keys, or no action rows) — nothing to
            # rebalance. Leave the original single-track output in place.
            return out
        out_dtype = out.dtype

        # Numerical stability: the action-self denominator sums ReLU products
        # over only ~tens of action keys (vs ~thousands of video keys for the
        # original single-track SANA denominator), so it can underflow toward
        # ``eps`` and make ``num_a / den_a`` explode — empirically this caused a
        # hard divergence (loss → inf → NaN) in late training for 2/3 of an
        # alpha sweep. Two guards: (1) compute the tracks in fp32 so bf16
        # mantissa loss in the matmul/reduction doesn't manufacture a tiny
        # denominator, and (2) floor each denominator at ``_DEN_FLOOR`` instead
        # of the original ``1e-15`` so the ratio can't blow up. The floor is a
        # no-op for healthy denominators (which are ≫ 1e-4 once features train).
        f32 = torch.float32
        tq_a = tilde_q[:, :, sv:, :].to(f32)
        pq_a = pq[:, :, sv:, :].to(f32)
        tk_v, v_v, pk_v = tilde_k[:, :, :sv, :].to(f32), v[:, :, :sv, :].to(f32), pk[:, :, :sv, :].to(f32)
        tk_a, v_a, pk_a = tilde_k[:, :, sv:, :].to(f32), v[:, :, sv:, :].to(f32), pk[:, :, sv:, :].to(f32)

        floor = self._DEN_FLOOR

        # Video-context track: action queries over video keys.
        if self.action_video_attn_mode == "softmax":
            # Sharp spatial grounding: softmax lets the action token attend to
            # the few video tokens that matter (gripper/object region) instead
            # of a linear count-weighted blur over all ~thousands of them.
            d = tk_v.shape[-1]
            scores_v = (tq_a @ tk_v.transpose(-1, -2)) / math.sqrt(d)  # (B, H, Sa, Sv)
            out_vtrack = torch.softmax(scores_v, dim=-1) @ v_v
        else:
            num_v = (tq_a @ tk_v.transpose(-1, -2)) @ v_v
            den_v = (pq_a @ pk_v.transpose(-1, -2)).sum(dim=-1, keepdim=True).clamp_min(floor)
            out_vtrack = num_v / den_v

        # Action-self track: action queries over action keys only.
        if self.action_self_attn_mode == "softmax":
            # Exact softmax over the action-key score matrix. Unlike linear
            # attention (a low-rank count-weighted average), softmax can place
            # nearly all mass on a few keys — the sharp retrieval action
            # dynamics may need. Scores reuse the rotated ReLU features already
            # computed; scaled by 1/sqrt(d) per standard scaled-dot-product.
            d = tk_a.shape[-1]
            scores = (tq_a @ tk_a.transpose(-1, -2)) / math.sqrt(d)  # (B, H, Sa, Sa)
            out_atrack = torch.softmax(scores, dim=-1) @ v_a
        else:
            num_a = (tq_a @ tk_a.transpose(-1, -2)) @ v_a
            den_a = (pq_a @ pk_a.transpose(-1, -2)).sum(dim=-1, keepdim=True).clamp_min(floor)
            out_atrack = num_a / den_a

        out_a = ((1.0 - w) * out_vtrack + w * out_atrack).to(out_dtype)
        return torch.cat([out[:, :, :sv, :], out_a], dim=2)


__all__ = ["SanaMoTJointDriver"]
