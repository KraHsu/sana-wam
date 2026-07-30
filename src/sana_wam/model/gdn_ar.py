"""True-autoregressive cached-GDN world-action architecture (align to sana-wm).

This is the missing fourth track. The fork already has:

- ``DualSystemSelfAttnArchitecture`` — linear_relu MoT joint self-attention.
- ``DualSystemARArchitecture`` — block-AR over the linear_relu operator
  (``ARLinearStateCache``): true streaming, but the "效果不好" operator.
- ``DualSystemCrossAttnArchitecture`` — the GDN operator, but in a one-shot
  bounded wrapper (``ChunkCausalGDNTriton``, no cache) → not autoregressive.

``DualSystemGDNARArchitecture`` composes the **right operator in the right
paradigm**: the video stream runs upstream Sana-wm's cached-GDN chunk streaming
(``forward_long`` + a rolling GDN **state cache**) so it generates past its
training horizon, and the action stream reuses the proven ``joint_cross_attn``
ActionDiT (MAE~0.08) by cross-attending to *this chunk's* per-layer video
features. It subclasses :class:`DualSystemCrossAttnArchitecture` to inherit the
action-backbone build, proprio-as-context handling, and ``bridge_tuple_from_dict``
verbatim — only the whole-clip video forward is swapped for the per-chunk cached
path (:meth:`forward_chunk`).

Per chunk c (absolute latent-frame window ``[start_f, end_f)``):

    video:  vb.run_chunk(chunk_latents, kv_cache) → (video_pred_c, bridges_c, kv_cache)
    action: ActionDiT.forward_with_bridge_tuple(actions_c, bridges_c, t, context)

The GDN state cache is full-history by construction (no finite window needed for
short robot episodes); ``reset_cache`` clears it at an episode boundary.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from sana_wam.model.cross_attn import DualSystemCrossAttnArchitecture


class DualSystemGDNARArchitecture(DualSystemCrossAttnArchitecture):
    """Cached-GDN autoregressive video + per-chunk cross-attn action bridge."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        c = self._cross_cfg or {}
        # Latent frames per autoregressive chunk. Must match the GDN chunk_size
        # the video backbone was built with (the cached kernel operates on whole
        # GDN chunks); surfaced as a config knob so train/deploy stay aligned.
        self._frame_chunk_size = int(c.get("frame_chunk_size", 0) or 0)
        if self._frame_chunk_size <= 0 and self.video_backbone is not None:
            self._frame_chunk_size = int(getattr(self.video_backbone, "_chunk_size", 3))
        # Action tokens emitted per chunk (0 ⇒ resolved by the caller/engine).
        self._action_tokens_per_chunk = int(c.get("action_tokens_per_chunk", 0) or 0)
        # Teacher-forced training: number of leading chunks supplied as the clean
        # observed prefix (ingested into the cache, NOT supervised) — mirrors the
        # streaming deploy loop, which ingests the real first observation before
        # predicting forward. Default 1 (one real obs); also keeps every denoised
        # chunk's cache non-empty, sidestepping the chunk-0-from-scratch edge.
        self._observed_prefix_chunks = int(c.get("ar_observed_prefix_chunks", 1) or 0)
        # Proprio modality-dropout / noise (training-time): force a vision-grounded
        # action pathway (see base._append_proprio_context_token). Default off.
        self._proprio_dropout = float(c.get("proprio_dropout", 0.0) or 0.0)
        self._proprio_noise_std = float(c.get("proprio_noise_std", 0.0) or 0.0)

    # ------------------------------------------------------------------
    # Streaming lifecycle helpers
    # ------------------------------------------------------------------

    @property
    def frame_chunk_size(self) -> int:
        return self._frame_chunk_size

    @property
    def action_tokens_per_chunk(self) -> int:
        return self._action_tokens_per_chunk

    def enable_streaming(self) -> None:
        """Switch the video backbone to its cached chunk-streaming variant."""
        if self.video_backbone is None:
            raise RuntimeError("enable_streaming requires a video_backbone.")
        self.video_backbone.enable_cached_streaming()

    def empty_kv_cache(self) -> list:
        """Fresh all-None rolling GDN state cache for a new episode/rollout."""
        if self.video_backbone is None:
            raise RuntimeError("empty_kv_cache requires a video_backbone.")
        return self.video_backbone.empty_kv_cache(self.video_backbone.num_layers)

    # alias for deploy-engine ergonomics (episode boundary)
    reset_cache = empty_kv_cache

    # ------------------------------------------------------------------
    # forward_chunk: one cached-GDN video chunk → action bridge
    # ------------------------------------------------------------------

    def forward_chunk(
        self,
        chunk_latents: Tensor,
        *,
        start_f: int,
        end_f: int,
        kv_cache: list,
        video_timestep: Tensor,
        noisy_actions: Optional[Tensor] = None,
        action_timestep: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        seq_lens: Optional[Tensor] = None,
        proprio_state: Optional[Tensor] = None,
        action_freqs: Optional[Tensor] = None,
        save_kv_cache: bool = True,
        frame_index: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], list]:
        """Run one autoregressive chunk and return ``(video_pred, action_pred, kv_cache)``.

        ``kv_cache`` is advanced in place and returned so the caller carries the
        rolling GDN state chunk→chunk. When ``noisy_actions`` is None (video-only
        / CFG nega pass) the action term is None.
        """
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — build the architecture with a video_backbone config."
            )

        # Append the proprio token to the raw caption context (augments context
        # for the action stream); mirrors DualSystemCrossAttnArchitecture.forward.
        pin = self._append_proprio_context_token(
            {"context": context, "context_mask": context_mask, "seq_lens": seq_lens},
            proprio_state,
        )
        a_context = pin.get("context")
        a_context_mask = pin.get("context_mask")
        if a_context is not None and a_context_mask is None and seq_lens is not None:
            seq_lens_d = seq_lens.to(device=a_context.device)
            positions = torch.arange(a_context.shape[1], device=a_context.device)
            a_context_mask = positions.unsqueeze(0) < seq_lens_d.unsqueeze(1)

        video_pred, bridges, kv_cache = vb.run_chunk(
            chunk_latents,
            video_timestep,
            context=a_context,
            context_mask=a_context_mask,
            seq_lens=seq_lens,
            kv_cache=kv_cache,
            start_f=start_f,
            end_f=end_f,
            save_kv_cache=save_kv_cache,
            frame_index=frame_index,
            bridge_layers=self._bridge_layers,
        )

        if noisy_actions is None or ab is None:
            return video_pred, None, kv_cache

        bridge_tuple = ab.bridge_tuple_from_dict(bridges)
        action_pred = ab.forward_with_bridge_tuple(
            noisy_actions,
            bridge_tuple,
            action_timestep,
            context=a_context,
            context_mask=a_context_mask,
            action_freqs=action_freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return video_pred, action_pred, kv_cache

    # ------------------------------------------------------------------
    # compute_loss: teacher-forced per-chunk flow-matching over the chunk loop
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        *,
        actions: Optional[Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        current_step: int = 0,  # noqa: ARG002
        **inputs,
    ) -> dict:
        """Teacher-forced rolling dual flow-matching loss (true-AR paradigm).

        Each sample is rolled over its OWN valid chunks (growing-history clips are
        frame-0-anchored and padded to the max-episode length; ``video_is_pad``
        gives the valid latent length per sample). A single rolling GDN state cache
        is carried across the sample's chunks. For each chunk c (in order):

        1. **Denoise pass** (``save_kv_cache=False``): the cache holds the CLEAN
           chunks ``0..c-1`` (teacher forcing) — i.e. cache DEPTH c; predict the
           noisy chunk c at a sampled timestep and accumulate flow-matching MSE.
        2. **Clean ingest** (``save_kv_cache=True``, ``no_grad``): advance the
           cache with the ground-truth chunk c so chunk ``c+1`` conditions on a
           clean, detached past — exactly the streaming-deploy cache contract.

        The first ``ar_observed_prefix_chunks`` chunks (default 1) are the observed
        prefix: ingested but NOT supervised (deploy is handed the real first obs).
        Supervising every chunk c≥prefix at cache depth c — across samples of
        varying valid length — covers the full range of cache depths a deploy
        rollout hits (the fix for the depth-1-only collapse). The rolling loop
        itself GENERATES the growing-history curriculum, so it reads only the valid
        length, not the per-sample clean-prefix boundary.

        Memory: each denoise ``forward_chunk`` is gradient-checkpointed (recompute
        in backward, bounded to one chunk) when ``use_gradient_checkpointing`` is
        set; the cache slot-lists are snapshotted per chunk so the recompute reads
        the correct depth (the live cache is mutated in place by later ingests).

        Positions are absolute but the RoPE is relative (translation-invariant), so
        no remapping is needed while indices stay < max_seq_len / max_action_len.

        This is genuinely sequential autoregression (rolling state cache), NOT the
        bounded duplicated-``[noisy++clean]`` sequence the linear_relu AR track
        uses. Only the flow-matching scaffold is shared with the cross-attn track.
        """
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint as _grad_ckpt

        vb = self.video_backbone
        ab = self.action_backbone
        device, dtype = self.device, self.dtype
        if vb is None or ab is None:
            raise RuntimeError("compute_loss requires both video and action backbones.")
        if not vb.cached_streaming_enabled:
            self.enable_streaming()

        clean_video = inputs["input_latents"].to(device=device, dtype=dtype)
        B, _, T, _, _ = clean_video.shape
        K = self._frame_chunk_size
        if K <= 0:
            raise ValueError(f"frame_chunk_size must be positive, got {K}.")
        # Full-clip chunk grid. The rolling loop rolls only each sample's VALID
        # chunks (growing-history clips are padded to the max-episode length), so
        # latent_T need not be a multiple of K — the partial tail is dropped.
        total_chunks = max(1, T // K)

        if actions is None or lambda_action <= 0:
            raise RuntimeError("DualSystemGDNARArchitecture.compute_loss requires actions.")
        actions = actions.to(device=device, dtype=dtype)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        Ta = actions.shape[1]
        # Uniform action-tokens-per-chunk over the full grid (same approximation as
        # the existing AR track); remainder tokens fall in the dropped partial tail.
        atc = max(1, Ta // total_chunks)

        context = inputs.get("context")
        context_mask = inputs.get("context_mask")
        seq_lens = inputs.get("seq_lens")
        proprio_state = inputs.get("proprio_state")
        ckpt = bool(inputs.get("use_gradient_checkpointing", False))
        ckpt_offload = bool(inputs.get("use_gradient_checkpointing_offload", False))

        # Per-sample valid latent length (growing-history pads to the max episode).
        # video_is_pad is (B, T) True=pad; absent ⇒ the whole clip is valid.
        video_is_pad = inputs.get("video_is_pad")
        if isinstance(video_is_pad, Tensor):
            valid_lat = (T - video_is_pad.to(device=device).long().sum(dim=1)).clamp(min=0, max=T)
        else:
            valid_lat = torch.full((B,), T, dtype=torch.long, device=device)
        action_is_pad = inputs.get("action_is_pad")
        if isinstance(action_is_pad, Tensor):
            action_is_pad = action_is_pad.to(device=device).bool()

        # Per-chunk proprio (gap2): condition each predicted chunk on the robot state
        # at its prediction boundary (the latest observed frame), matching deploy which
        # advances proprio every step (training previously reused ONE static state for
        # all chunks). proprio_seq is the full per-raw-frame state (B, T_state, D);
        # index it with the dataloader's own formula raw_idx = tc*(P_lat-1)*video_stride.
        # video_stride is derived self-contained from shapes (no extra plumbing):
        # num_video_frames = 1+(T-1)*tc and T_state = num_frames.
        proprio_seq = inputs.get("proprio_seq")
        tc = int(getattr(vb, "temporal_compression", 4) or 4)
        prop_stride = 1
        if isinstance(proprio_seq, Tensor):
            proprio_seq = proprio_seq.to(device=device, dtype=dtype)
            t_state = proprio_seq.shape[1]
            nvf = 1 + (T - 1) * tc
            prop_stride = max(1, (t_state - 1) // max(1, nvf - 1))

        def _chunk_proprio(b_, c_, fallback):
            # P_lat = observed latent frames before predicting chunk c_: 1 for the
            # bootstrap chunk 0 (only the pinned first frame), else c_*K.
            if not isinstance(proprio_seq, Tensor):
                return fallback
            p_lat = 1 if c_ == 0 else c_ * K
            raw_idx = min(tc * (max(1, p_lat) - 1) * prop_stride, proprio_seq.shape[1] - 1)
            return proprio_seq[b_ : b_ + 1, raw_idx]  # (1, D)

        num_ts_v = len(vb.scheduler.timesteps)
        num_ts_a = len(ab.scheduler.timesteps)
        w_v = vb.scheduler.linear_timesteps_weights

        loss_v_sum = clean_video.new_zeros((), dtype=torch.float32)
        loss_a_sum = clean_video.new_zeros((), dtype=torch.float32)
        n_supervised = 0

        # Per-sample rolling loop: each sample carries its own valid length (ragged
        # under growing-history), so we roll its own chunk count + its own cache.
        for b in range(B):
            valid_chunks = int(valid_lat[b].item()) // K
            if valid_chunks < 1:  # need at least the bootstrap chunk 0
                continue
            vid_b = clean_video[b : b + 1]
            act_b = actions[b : b + 1]
            # Per-sample slices of the conditioning (the video forward runs one
            # sample at a time, so context/proprio batch must match).
            ctx_b = context[b : b + 1] if isinstance(context, Tensor) else context
            cmask_b = context_mask[b : b + 1] if isinstance(context_mask, Tensor) else context_mask
            seq_b = seq_lens[b : b + 1] if isinstance(seq_lens, Tensor) else seq_lens
            prop_b = proprio_state[b : b + 1] if isinstance(proprio_state, Tensor) else proprio_state
            kv = self.empty_kv_cache()

            # Every chunk is supervised. Chunk 0 is the BOOTSTRAP (gap1): its first
            # latent frame is pinned clean (the real first observation) and the rest
            # denoised from an EMPTY cache — exactly the deploy episode start, which
            # predicts chunk 0 from the first obs (no one-chunk look-ahead). Chunk c≥1
            # is full-noisy at cache depth c (clean chunks 0..c-1 ingested).
            for c in range(valid_chunks):
                clean_c = vid_b[:, :, c * K : (c + 1) * K]  # (1, C, K, H, W)
                proprio_c = _chunk_proprio(b, c, prop_b)
                clean_a_c = act_b[:, c * atc : (c + 1) * atc]  # (1, atc, Ad)
                # Delta actions (per-chunk): predict the displacement from THIS chunk's
                # proprio anchor (the state the chunk is conditioned on) instead of the
                # absolute target. Anchoring to proprio_c — the SAME state fed as the
                # per-chunk context and the one deploy advances every step — keeps train
                # and AR rollout consistent (deploy adds the current proprio back). A
                # single window anchor (cross-attn style) would NOT match GDN-AR's
                # advancing per-chunk proprio. proprio_c is (1, D); broadcast over atok.
                if self._delta_action:
                    if not isinstance(proprio_c, Tensor):
                        raise RuntimeError(
                            "model.architecture.delta_action requires per-chunk proprio "
                            "(set proprio_per_chunk + provide proprio_seq)."
                        )
                    clean_a_c = clean_a_c - proprio_c.unsqueeze(1)

                # ---- coupled video+action timestep (Fix A: train/deploy bridge match) ----
                # Deploy co-denoises video and action on the SAME schedule index (step i
                # drives v_ts[i] AND a_ts[i]). Training previously sampled the two
                # timesteps INDEPENDENTLY, so the action was frequently supervised while
                # cross-attending a near-CLEAN GT-future video bridge it never sees at
                # deploy (where the bridge is the still-noisy / degenerate dream). Sample
                # ONE shared normalized position u and map it into each scheduler so the
                # action always reads the video at the matching noise level (identical
                # index when the two training schedules have equal length).
                u = float(torch.rand(1).item())
                v_ids = torch.tensor([min(num_ts_v - 1, int(u * num_ts_v))])
                a_ids = torch.tensor([min(num_ts_a - 1, int(u * num_ts_a))])

                # ---- video noising at the coupled per-chunk timestep ----
                v_sigma = vb.scheduler.sigmas[v_ids].to(device=device, dtype=dtype).view(1, 1, 1, 1, 1)
                v_ts_val = vb.scheduler.timesteps[v_ids].to(device=device, dtype=dtype)
                v_noise = torch.randn_like(clean_c)
                if c == 0:
                    # First-frame pin: latent 0 clean (sigma=0), [1,K) noised.
                    v_sig_f = v_sigma.expand(1, 1, K, 1, 1).clone()
                    v_sig_f[:, :, 0] = 0.0
                    noisy_c = (1 - v_sig_f) * clean_c + v_sig_f * v_noise
                else:
                    noisy_c = (1 - v_sigma) * clean_c + v_sigma * v_noise
                v_target = vb.scheduler.training_target(clean_c, v_noise, v_ts_val)

                # ---- action noising (a_ids coupled to the video timestep above) ----
                a_sigma = ab.scheduler.sigmas[a_ids].to(device=device, dtype=dtype).view(1, 1, 1)
                a_ts_val = ab.scheduler.timesteps[a_ids].to(device=device, dtype=dtype)
                a_noise = torch.randn_like(clean_a_c)
                noisy_a_c = (1 - a_sigma) * clean_a_c + a_sigma * a_noise
                a_target = ab.scheduler.training_target(clean_a_c, a_noise)

                # Action RoPE positions for this chunk (relative RoPE ⇒ translation-
                # invariant; absolute index only needs < max_action_len).
                positions = torch.arange(c * atc, (c + 1) * atc, device=device)
                action_freqs = ab._get_rope_freqs_at(positions)

                sf, ef = c * K, (c + 1) * K

                # Snapshot the cache slot-lists NOW (forward time) and freeze them, so a
                # checkpoint recompute in backward reads THIS chunk's depth — NOT the
                # live cache, which later no_grad ingests mutate in place (empty→depth-1
                # for the bootstrap chunk would otherwise change the saved-tensor count
                # and trip the recompute check). Shallow copy shares tensors (cheap);
                # save=False keeps it read-only, and the ingest rebinds the LIVE list's
                # slots (new tensors), never this frozen copy.
                kv_snap = [list(slot) for slot in kv]

                def _denoise(nc, na, _sf=sf, _ef=ef, _af=action_freqs, _kv=kv_snap, _pc=proprio_c):
                    vp, ap, _ = self.forward_chunk(
                        nc, start_f=_sf, end_f=_ef, kv_cache=_kv,
                        video_timestep=v_ts_val, noisy_actions=na, action_timestep=a_ts_val,
                        context=ctx_b, context_mask=cmask_b, seq_lens=seq_b,
                        proprio_state=_pc, action_freqs=_af,
                        save_kv_cache=False, use_gradient_checkpointing=False,
                        use_gradient_checkpointing_offload=ckpt_offload,
                    )
                    return vp, ap

                if ckpt:
                    v_pred, a_pred = _grad_ckpt(_denoise, noisy_c, noisy_a_c, use_reentrant=False)
                else:
                    v_pred, a_pred = _denoise(noisy_c, noisy_a_c)

                # Per-frame video MSE; for the bootstrap chunk exclude the pinned frame 0
                # (it is the given observation, not a denoising target).
                per_vf = F.mse_loss(v_pred.float(), v_target.float(), reduction="none").mean(dim=(1, 3, 4))  # (1,K)
                if c == 0 and K > 1:
                    per_v = per_vf[:, 1:].mean(dim=1)  # exclude pinned frame 0
                elif c == 0:
                    per_v = None  # K==1 bootstrap: only the pinned frame, no video target
                else:
                    per_v = per_vf.mean(dim=1)  # (1,)
                if per_v is not None:
                    wv = (
                        w_v[v_ids].to(dtype=torch.float32, device=device)
                        if w_v is not None
                        else torch.ones(1, dtype=torch.float32, device=device)
                    )
                    loss_v_sum = loss_v_sum + (per_v * wv).mean()

                # Action loss, masked by this chunk's action-pad slice (the causal-VAE
                # mapping can leave a few padded tokens in the tail chunk).
                per_a = F.mse_loss(a_pred.float(), a_target.float(), reduction="none").mean(dim=2)  # (1, atc)
                if action_is_pad is not None:
                    sl = action_is_pad[b : b + 1, c * atc : (c + 1) * atc]
                    if sl.shape[1] == per_a.shape[1]:
                        keep = (~sl).float()
                        loss_a_sum = loss_a_sum + (per_a * keep).sum() / keep.sum().clamp_min(1.0)
                    else:
                        loss_a_sum = loss_a_sum + per_a.mean()
                else:
                    loss_a_sum = loss_a_sum + per_a.mean()
                n_supervised += 1

                # ---- advance cache with the CLEAN chunk c (teacher forcing) ----
                # Detached (no_grad): the past is given ground truth, not a target.
                if c < valid_chunks - 1:
                    with torch.no_grad():
                        _, _, kv = self.forward_chunk(
                            clean_c,
                            start_f=c * K,
                            end_f=(c + 1) * K,
                            kv_cache=kv,
                            video_timestep=torch.zeros(1, device=device),
                            noisy_actions=None,
                            context=ctx_b,
                            context_mask=cmask_b,
                            seq_lens=seq_b,
                            proprio_state=proprio_c,
                            save_kv_cache=True,
                        )

        if n_supervised == 0:
            # No supervised chunk (every sample in this micro-batch was shorter than
            # one observed + one predicted chunk). Return a grad-connected zero so the
            # trainer's backward is a clean no-op instead of crashing on a graph-less
            # loss. Rare once history_min_frames guarantees ≥2 chunks, but defensive.
            p = next(ab.parameters())
            z = p.sum() * 0.0
            return {"loss": z, "loss_video": z.detach(), "loss_action": z.detach()}

        loss_video = loss_v_sum / n_supervised
        loss_action = loss_a_sum / n_supervised
        loss = lambda_video * loss_video + lambda_action * loss_action
        return {
            "loss": loss,
            "loss_video": (lambda_video * loss_video).detach(),
            "loss_action": (lambda_action * loss_action).detach(),
        }

    # ------------------------------------------------------------------
    # forward: ABC contract. The cached-GDN track is driven per chunk via
    # forward_chunk; the inherited whole-clip forward/compute_loss are invalid
    # once streaming is enabled (rope/blocks swapped). Guard against accidental
    # whole-clip use so the failure is loud, not a silent wrong-paradigm run.
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):  # type: ignore[override]
        if self.video_backbone is not None and getattr(
            self.video_backbone, "cached_streaming_enabled", False
        ):
            raise RuntimeError(
                "DualSystemGDNARArchitecture is a cached-streaming (autoregressive) "
                "architecture — drive it with forward_chunk() per chunk, not the "
                "whole-clip forward(). The per-chunk training loop lands in Phase 2."
            )
        # Before streaming is enabled the inherited whole-clip forward still works
        # (e.g. for parity checks against the cross-attn track).
        return super().forward(*args, **kwargs)


__all__ = ["DualSystemGDNARArchitecture"]
