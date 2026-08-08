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

import copy
from typing import TYPE_CHECKING, Optional

import torch
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor

from sana_wam.model.ar.sana_ar_inference import (
    ARLinearStateCache,
    ar_inference_attn,
    clean_state_from_tokens,
)
from sana_wam.model.ar.sana_ar_linear_attn import (
    AR_LINEAR_ATTN_EPS,
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
        self.action_video_memory_adapter = kwargs.pop(
            "action_video_memory_adapter", None
        )
        # AR training and cache inference both pass ``self.eps`` to their shared
        # fp32 attention definition.  Keep the non-AR driver default untouched.
        kwargs.setdefault("eps", AR_LINEAR_ATTN_EPS)
        super().__init__(*args, **kwargs)
        self._ar_meta: Optional[ARSeqMeta] = None
        self._active_ar_layer_id: Optional[int] = None
        self._return_action_video_numerators = False

    def _step_impl(self, layer_id, *args, **kwargs):
        """Publish the layer id to the structured kernel for adapter lookup."""
        if self._active_ar_layer_id is not None:
            raise RuntimeError("nested SanaARMoTJointDriver layer execution")
        self._active_ar_layer_id = int(layer_id)
        try:
            return super()._step_impl(layer_id, *args, **kwargs)
        finally:
            self._active_ar_layer_id = None

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
        return_action_video_numerators: bool = False,
    ) -> Optional[Tensor]:
        """Run the per-layer joint loop under the AR descriptor.

        Mirrors :meth:`MoTJointDriver.run_joint_loop` but pins ``attn_mask=None``
        and stashes ``ar_meta`` for :meth:`_mixed_attention`. The duplicated
        sequence must already be materialized in ``vstate`` (``[v_noisy ++
        v_clean]``) and ``astate`` (``[a_noisy ++ a_clean]``) by the caller.
        """
        if type(return_action_video_numerators) is not bool:
            raise TypeError("return_action_video_numerators must be boolean")
        if self._ar_meta is not None or self._return_action_video_numerators:
            raise RuntimeError("overlapping Sana AR joint loops")
        layer_numerators: list[Tensor] = []
        self._ar_meta = ar_meta
        self._return_action_video_numerators = return_action_video_numerators
        try:
            for layer_id in range(self.num_layers):
                step_result = self.step(
                    layer_id,
                    vstate,
                    astate,
                    attn_mask=None,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                )
                if return_action_video_numerators:
                    if not isinstance(step_result, tuple) or len(step_result) != 3:
                        raise RuntimeError(
                            "action-facing capture requires one auxiliary tensor per layer"
                        )
                    vstate, astate, layer_numerator = step_result
                    if not isinstance(layer_numerator, Tensor):
                        raise TypeError("action-facing layer capture must be a tensor")
                    layer_numerators.append(layer_numerator)
                else:
                    if not isinstance(step_result, tuple) or len(step_result) != 2:
                        raise RuntimeError(
                            "ordinary AR layer execution returned an unexpected auxiliary"
                        )
                    vstate, astate = step_result
        finally:
            self._ar_meta = None
            self._return_action_video_numerators = False
        if not return_action_video_numerators:
            return None
        if len(layer_numerators) != self.num_layers:
            raise RuntimeError(
                "action-facing capture did not cover every AR layer: "
                f"expected {self.num_layers}, got {len(layer_numerators)}"
            )
        return torch.stack(layer_numerators, dim=1)

    def _step_checkpointed(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        *,
        attn_mask: Optional[Tensor],
        offload: bool,
    ):
        """Checkpoint one layer while retaining its immutable AR descriptor.

        The base checkpoint closure is recomputed after ``run_ar_joint_loop``
        has cleared ``self._ar_meta``. Capture the descriptor in this closure
        and publish it only for the duration of each forward/recompute call.
        """
        ar_meta = self._ar_meta
        if ar_meta is None:
            raise RuntimeError("checkpointed AR step requires an AR descriptor")
        return_action_video_numerators = self._return_action_video_numerators
        outer_payload = astate.payload
        vx0 = vstate.x
        ax0 = outer_payload.x_action

        # A non-reentrant checkpoint frame retains its recompute closure until
        # backward finishes.  Capturing the live mutable states here would form
        # a cycle after the checkpoint outputs are written back below:
        # output graph -> checkpoint frame -> closure -> state -> output graph.
        # Keep tensor-free templates for the two fields mutated by every layer;
        # the closure receives their current values only through checkpoint's
        # explicit tensor arguments.
        vstate_template = copy.copy(vstate)
        astate_template = copy.copy(astate)
        payload_template = copy.copy(outer_payload)
        vstate_template.x = None
        astate_template.payload = None
        payload_template.x_action = None

        def _run(vx: Tensor, ax: Tensor):
            local_vstate = copy.copy(vstate_template)
            local_astate = copy.copy(astate_template)
            local_payload = copy.copy(payload_template)
            local_astate.payload = local_payload
            local_vstate.x = vx
            local_payload.x_action = ax

            previous_meta = self._ar_meta
            previous_capture = self._return_action_video_numerators
            if previous_meta is not None and previous_meta is not ar_meta:
                raise RuntimeError("overlapping Sana AR checkpoint recomputation")
            self._ar_meta = ar_meta
            self._return_action_video_numerators = (
                return_action_video_numerators
            )
            try:
                step_result = self._step_impl(
                    layer_id,
                    local_vstate,
                    local_astate,
                    attn_mask=attn_mask,
                    suppress_inner_attn_ckpt=True,
                )
            finally:
                self._ar_meta = previous_meta
                self._return_action_video_numerators = previous_capture
            if return_action_video_numerators:
                if not isinstance(step_result, tuple) or len(step_result) != 3:
                    raise RuntimeError(
                        "checkpointed action-facing capture lost its auxiliary"
                    )
                return local_vstate.x, local_payload.x_action, step_result[2]
            if not isinstance(step_result, tuple) or len(step_result) != 2:
                raise RuntimeError("checkpointed AR step returned an unexpected auxiliary")
            return local_vstate.x, local_payload.x_action

        if offload:
            with torch.autograd.graph.save_on_cpu():
                checkpoint_result = torch.utils.checkpoint.checkpoint(
                    _run, vx0, ax0, use_reentrant=False
                )
        else:
            checkpoint_result = torch.utils.checkpoint.checkpoint(
                _run, vx0, ax0, use_reentrant=False
            )
        if return_action_video_numerators:
            new_vx, new_ax, layer_numerator = checkpoint_result
        else:
            new_vx, new_ax = checkpoint_result
        vstate.x = new_vx
        outer_payload.x_action = new_ax
        if return_action_video_numerators:
            return vstate, astate, layer_numerator
        return vstate, astate

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
    ) -> Tensor | tuple[Tensor, Tensor]:
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

        result = _ar_chunked_linear_attn(
            tilde_q,
            tilde_k,
            v,
            pq,
            pk,
            meta,
            eps=self.eps,
            action_video_memory_adapter=self.action_video_memory_adapter,
            layer_id=self._active_ar_layer_id,
            return_action_video_numerators=(
                self._return_action_video_numerators
            ),
        )
        # NB: no _rebalance_action_rows here — see class docstring.
        if isinstance(result, tuple):
            out, layer_numerators = result
            return (
                rearrange(out, "b n s d -> b s (n d)", n=n),
                layer_numerators,
            )
        return rearrange(result, "b n s d -> b s (n d)", n=n)

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
            if (
                self.action_video_memory_adapter is not None
                and backbone is self.ab
                and not store_clean
            ):
                video_win = cache.windowed_video_state(
                    layer_id, frame_id, hi_inclusive=frame_id - 1
                )
                if video_win is not None:
                    if win is None:
                        raise RuntimeError(
                            "video window exists while canonical window is empty"
                        )
                    delta_s, delta_z = (
                        self.action_video_memory_adapter.forward_delta(
                            layer_id, *video_win
                        )
                    )
                    win = (win[0] + delta_s, win[1] + delta_z)
            out = ar_inference_attn(tilde_q, phi_q, win, tilde_k, phi_k, v_h, eps=self.eps)
            out = rearrange(out, "b n s d -> b s (n d)", n=n)
            state = backbone.post_attn_at_layer(layer_id, state, out, post)

            if store_clean:
                cache.update(layer_id, frame_id, s_self, z_self, is_pred=is_pred)
        return state
