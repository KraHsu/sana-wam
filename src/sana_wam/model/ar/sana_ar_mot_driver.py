"""SanaARMoTJointDriver: autoregressive (block-causal) MoT driver for SANA.

Extends :class:`SanaMoTJointDriver` to realize the LingBot-VA block-causal
attention topology on the *duplicated* ``[v_noisy, v_clean, a_noisy, a_clean]``
sequence, using the structured linear-attention kernel in
:mod:`sana_ar_linear_attn` instead of the monotonic-block cumsum path.

Sequence alignment
------------------
The base :meth:`MoTJointDriver._step_impl` concatenates the video stream and the
action stream: ``q_cat = [q_video, q_action]``. The AR architecture builds each
stream as its own duplicated copy — video stream ``= [v_noisy ++ v_clean]``,
action stream ``= [a_noisy ++ a_clean]`` — so the concatenation is exactly
``[v_noisy, v_clean, a_noisy, a_clean]``, matching
:func:`sana_ar_linear_attn.build_ar_seq_meta`'s token order. The per-token
``frame_id`` / ``noise_id`` metadata for that full sequence lives in
``self._ar_meta``, set once per forward by :meth:`run_ar_joint_loop`.

Differences from the parent
---------------------------
- :meth:`_mixed_attention` ignores ``attn_mask`` (the AR run loop passes
  ``None``) and instead drives :func:`_ar_chunked_linear_attn` with
  ``self._ar_meta``. A non-``None`` ``attn_mask`` is rejected — silently routing
  an AR forward through the parent's dense-mask path would be both ``O(N²)`` and
  semantically wrong (the parent has no notion of the clean/noise duplication).
- ``_rebalance_action_rows`` is **not** applied: it assumes action queries see
  all video/action keys densely, which contradicts the windowed causal AR
  visibility. ``action_self_attn_weight`` is therefore inert for the AR variant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from einops import rearrange
from torch import Tensor

from sana_wam.model.ar.sana_ar_inference import (
    ARLinearStateCache,
    ar_inference_attn,
    clean_state_from_tokens,
)
from sana_wam.model.ar.sana_ar_linear_attn import (
    ARSeqMeta,
    _ar_chunked_linear_attn,
)
from sana_wam.model.ar.sana_mot_driver import SanaMoTJointDriver

if TYPE_CHECKING:
    from sana_wam.model.base import ActionState
    from sana_wam.model.video_backbone.adapter import BlockLoopState


class SanaARMoTJointDriver(SanaMoTJointDriver):
    """Block-causal (diffusion-forcing) MoT driver for SANA linear attention."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ar_meta: Optional[ARSeqMeta] = None

    # ------------------------------------------------------------------
    # AR loop entry point
    # ------------------------------------------------------------------

    def run_ar_joint_loop(
        self,
        vstate: "BlockLoopState",
        astate: "ActionState",
        *,
        ar_meta: ARSeqMeta,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> None:
        """Run the per-layer joint loop under the AR descriptor.

        Mirrors :meth:`MoTJointDriver.run_joint_loop` but pins ``attn_mask=None``
        and stashes ``ar_meta`` for :meth:`_mixed_attention`. The duplicated
        sequence must already be materialized in ``vstate`` (``[v_noisy ++
        v_clean]``) and ``astate`` (``[a_noisy ++ a_clean]``) by the caller.
        """
        self._ar_meta = ar_meta
        try:
            for layer_id in range(self.num_layers):
                self.step(
                    layer_id,
                    vstate,
                    astate,
                    attn_mask=None,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                )
        finally:
            self._ar_meta = None

    # ------------------------------------------------------------------
    # Mixed attention — structured AR kernel
    # ------------------------------------------------------------------

    def _mixed_attention(  # type: ignore[override]
        self,
        q_cat: Tensor,
        k_cat: Tensor,
        v_cat: Tensor,
        attn_mask: Optional[Tensor],
        *,
        phi_q: Optional[Tensor] = None,
        phi_k: Optional[Tensor] = None,
        use_ckpt: bool = False,  # noqa: ARG002 — AR kernel is already O(N·d²); per-chunk ckpt is future work
        s_video: Optional[int] = None,  # noqa: ARG002 — visibility encoded in ar_meta, not the modality split
        s_action: Optional[int] = None,  # noqa: ARG002
    ) -> Tensor:
        if self._ar_meta is None:
            raise RuntimeError(
                "SanaARMoTJointDriver._mixed_attention called without an AR descriptor. "
                "Use run_ar_joint_loop (which sets self._ar_meta) rather than the parent's run_joint_loop."
            )
        if attn_mask is not None:
            raise RuntimeError(
                "SanaARMoTJointDriver received a dense attn_mask. The AR path must run with "
                "attn_mask=None and drive visibility through self._ar_meta — a dense mask would "
                "route to the parent's O(N²) path and ignore the clean/noise duplication."
            )
        if phi_q is None or phi_k is None:
            raise RuntimeError(
                "SanaARMoTJointDriver._mixed_attention requires phi_q and phi_k "
                "(unrotated ReLU'd Q/K from post_state)."
            )

        meta = self._ar_meta
        n = self.num_heads
        tilde_q = rearrange(q_cat, "b s (n d) -> b n s d", n=n)
        tilde_k = rearrange(k_cat, "b s (n d) -> b n s d", n=n)
        v = rearrange(v_cat, "b s (n d) -> b n s d", n=n)
        pq = rearrange(phi_q, "b s (n d) -> b n s d", n=n)
        pk = rearrange(phi_k, "b s (n d) -> b n s d", n=n)

        if meta.frame_ids.shape[0] != tilde_q.shape[-2]:
            raise ValueError(
                f"ar_meta covers {meta.frame_ids.shape[0]} tokens but the concatenated "
                f"sequence has {tilde_q.shape[-2]}. The duplicated [v_noisy,v_clean,a_noisy,"
                "a_clean] layout must match build_ar_seq_meta's token order."
            )

        out = _ar_chunked_linear_attn(tilde_q, tilde_k, v, pq, pk, meta, eps=self.eps)
        # NB: no _rebalance_action_rows here — see class docstring.
        return rearrange(out, "b n s d -> b s (n d)", n=n)

    # ------------------------------------------------------------------
    # Inference: run ONE chunk (single modality) through its backbone with
    # cache-aware attention. The reusable heart of the AR rollout.
    # ------------------------------------------------------------------

    def run_ar_chunk_through_backbone(
        self,
        backbone,
        state,
        cache: ARLinearStateCache,
        frame_id: int,
        *,
        store_clean: bool,
        is_pred: bool = False,
    ):
        """Run a chunk through all DiT layers using the linear-state cache.

        Unlike :meth:`run_ar_joint_loop` (training, full duplicated sequence in
        one mixed attention), inference processes one modality's chunk at a time:
        the chunk's queries attend to the windowed **cached** clean history
        (frames ``[frame_id - W, frame_id - 1]``, both modalities via the shared
        frame-id axis) plus the chunk's own within-frame block. This is exactly
        the noisy-row / clean-row computation of the training kernel with the
        clean history supplied by the cache (proven equivalent in
        ``tests/test_sana_ar_inference.py``).

        Parameters
        ----------
        backbone
            The video or action backbone (must expose ``pre_attn_at_layer`` /
            ``post_attn_at_layer`` with the ``q_unrot`` / ``k_unrot`` linear-relu
            contract and ``num_layers``).
        state
            The backbone's per-chunk loop state (``BlockLoopState`` / ``ActionState``).
        cache
            Shared per-layer linear-state cache (keyed by ``frame_id``).
        frame_id
            This chunk's modality-parity frame id (video ``2c`` / action ``2c+1``).
        store_clean
            If True, write this chunk's per-layer clean ``(S, z)`` into the cache
            (real-obs ingestion ⇒ ``is_pred=False``; predicted chunk ⇒ ``True``).
            Always uses strict-causal history (``hi = frame_id - 1``) + own block,
            so clean2clean inclusivity comes from the own-block term.
        """
        n = self.num_heads
        for layer_id in range(backbone.num_layers):
            q, k, v, post = backbone.pre_attn_at_layer(layer_id, state)
            if not post.get("uses_linear_attn", False):
                raise RuntimeError("run_ar_chunk_through_backbone requires a linear_relu backbone.")
            tilde_q = rearrange(q, "b s (n d) -> b n s d", n=n)
            tilde_k = rearrange(k, "b s (n d) -> b n s d", n=n)
            v_h = rearrange(v, "b s (n d) -> b n s d", n=n)
            phi_q = rearrange(post["q_unrot"], "b s (n d) -> b n s d", n=n)
            phi_k = rearrange(post["k_unrot"], "b s (n d) -> b n s d", n=n)

            s_self, z_self = clean_state_from_tokens(tilde_k, v_h, phi_k)
            win = cache.windowed_state(layer_id, frame_id, hi_inclusive=frame_id - 1)
            out = ar_inference_attn(tilde_q, phi_q, win, tilde_k, phi_k, v_h, eps=self.eps)
            out = rearrange(out, "b n s d -> b s (n d)", n=n)
            state = backbone.post_attn_at_layer(layer_id, state, out, post)

            if store_clean:
                cache.update(layer_id, frame_id, s_self, z_self, is_pred=is_pred)
        return state
