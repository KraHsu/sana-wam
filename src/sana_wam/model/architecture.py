"""DualSystem autoregressive (block-causal / diffusion-forcing) architecture.

Aligns the SANA dual-system MoT path to LingBot-VA: at every layer the video and
action streams are run as a *duplicated* ``[noisy ++ clean]`` sequence and mixed
through the structured AR linear-attention kernel
(:class:`SanaARMoTJointDriver`), giving block-causal visibility with a clean
conditioning history.

Forward contract (driven by this architecture's own ``compute_loss`` / the AR
inference engine):

- ``latents``        — noisy video latent ``(B, C, T, H, W)`` (T = C_chunks · frame_chunk_size).
- ``ar_clean_latents`` — clean (conditioning) video latent, same shape.
- ``ar_video_frame_timesteps`` — ``(B, T)`` per-noisy-frame diffusion timestep.
- ``ar_clean_video_frame_timesteps`` — ``(B, T)`` for the clean copy (zeros, or a
  small ``cond`` timestep for robustness). Optional; defaults to zeros.
- ``noisy_actions`` ``(B, Ta, Ad)`` / ``ar_clean_actions`` ``(B, Ta, Ad)``.
- ``ar_action_token_timesteps`` ``(B, Ta)`` / ``ar_clean_action_token_timesteps`` ``(B, Ta)``.
- ``ar_frame_chunk_size`` (int), ``ar_attn_window`` (int, frame-id units).

Returns ``(video_noise_pred, action_noise_pred)`` for the **noisy copy only**
(shapes match the noisy ``latents`` / ``noisy_actions`` inputs).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

from sana_wam.model.joint_self_attn import DualSystemSelfAttnArchitecture
from sana_wam.model.ar.sana_ar_linear_attn import build_ar_seq_meta

logger = logging.getLogger(__name__)


class DualSystemARArchitecture(DualSystemSelfAttnArchitecture):
    """Block-autoregressive DualSystem variant (SANA linear-attention only)."""

    def __init__(self, cfg=None):
        super().__init__(cfg)

        def _get(key, default):
            if cfg is None:
                return default
            if hasattr(cfg, "get"):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        self._ar_frame_chunk_size = int(_get("ar_frame_chunk_size", 1))
        self._ar_attn_window = int(_get("ar_attn_window", 10_000))
        self._ar_noisy_cond_prob = float(_get("ar_noisy_cond_prob", 0.5))
        # Clean-copy "cond" timestep is sampled in [0, ar_cond_max_ratio · num_train_ts)
        # when the noisy_cond_prob bernoulli fires (LingBot-style robustness aug).
        self._ar_cond_max_ratio = float(_get("ar_cond_max_ratio", 0.3))

        # F4: per-chunk proprio. When True, in addition to the FastWAM clip-level
        # proprio context token, each chunk is conditioned on the robot state at
        # that chunk's time via an additive per-frame/per-token AdaLN delta
        # (zero-init encoders => identical to the current checkpoint at init).
        self._proprio_per_chunk = bool(_get("proprio_per_chunk", False))
        # F3: supply chunk-0's first latent frame clean (real init obs) in training
        # and emit a chunk-0 action in the rollout (LingBot bootstrap).
        self._ar_bootstrap_clean_prefix = bool(_get("ar_bootstrap_clean_prefix", False))
        self.proprio_video_embed = None
        self.proprio_action_embed = None
        self._maybe_init_per_chunk_proprio()

    def _maybe_init_per_chunk_proprio(self) -> None:
        """Create the zero-init per-chunk proprio encoders once both backbones exist.

        Called at __init__ (real cfg path: backbones already built → encoders land
        in the optimizer) and from build_mot_driver (test path: backbones set later).
        Idempotent. Zero-init so the model output is identical to the pre-F4
        checkpoint at step 0 (warm-startable retrain)."""
        if not (getattr(self, "_proprio_per_chunk", False) and bool(getattr(self, "_use_proprioception_context", False))):
            return
        if getattr(self, "proprio_video_embed", None) is not None:
            return
        if self.video_backbone is None or self.action_backbone is None:
            return
        import torch.nn as nn

        sd = int(self.proprio_dim)
        self.proprio_video_embed = nn.Linear(sd, int(self.video_backbone.dim))
        self.proprio_action_embed = nn.Linear(sd, int(self.action_backbone.dim))
        for m in (self.proprio_video_embed, self.proprio_action_embed):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def set_dtype_device(self, dtype, device) -> None:
        super().set_dtype_device(dtype, device)
        for name in ("proprio_video_embed", "proprio_action_embed"):
            m = getattr(self, name, None)
            if m is not None:
                m.to(dtype=dtype, device=device)

    def build_mot_driver(self):
        """Construct the AR MoT driver (SANA linear-attn only)."""
        if self.video_backbone is None or self.action_backbone is None:
            raise RuntimeError("DualSystemARArchitecture.build_mot_driver requires both backbones.")
        kernel = getattr(self.video_backbone, "attn_kernel", "softmax")
        if kernel != "linear_relu":
            raise ValueError(
                f"DualSystemARArchitecture requires SANA linear_relu attention, got attn_kernel='{kernel}'."
            )
        from sana_wam.model.ar.sana_ar_mot_driver import SanaARMoTJointDriver

        # The AR kernel does not use the dense-mask rebalance knobs; drop them.
        kwargs = {
            k: v
            for k, v in self._mot_driver_kwargs.items()
            if k not in ("action_self_attn_weight", "action_self_attn_mode", "action_video_attn_mode")
        }
        self._mot_driver = SanaARMoTJointDriver(self.video_backbone, self.action_backbone, **kwargs)
        self._maybe_init_per_chunk_proprio()  # test path: backbones set after __init__
        return self._mot_driver

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],  # noqa: ARG002 — AR uses per-token ar_action_token_timesteps
        *,
        proprio_state: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None or ab is None:
            raise RuntimeError("DualSystemARArchitecture.forward requires both video and action backbones.")
        if noisy_actions is None:
            raise RuntimeError("DualSystemARArchitecture.forward requires noisy_actions (joint AR path).")

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio_state)
        # F4: per-chunk proprio (B, num_chunks, state_dim); popped before vb.prepare(**kwargs).
        proprio_per_chunk = pipeline_inputs.pop("proprio_per_chunk", None)

        # --- AR config + extras (pop all AR keys up front so none leak into
        #     the video backbone's prepare(**kwargs)) ---
        frame_chunk_size = int(pipeline_inputs.pop("ar_frame_chunk_size"))
        attn_window = int(pipeline_inputs.pop("ar_attn_window"))
        clean_actions = pipeline_inputs.pop("ar_clean_actions", None)
        a_noisy_ts = pipeline_inputs.pop("ar_action_token_timesteps")  # (B, Ta)
        a_clean_ts = pipeline_inputs.pop("ar_clean_action_token_timesteps", None)

        # --- video stream: [noisy ++ clean] along the latent frame axis ---
        noisy_latents = pipeline_inputs.pop("latents")
        clean_latents = pipeline_inputs.pop("ar_clean_latents")
        B, _, T, _, _ = noisy_latents.shape
        if T % frame_chunk_size != 0:
            raise ValueError(f"latent frames T={T} must be divisible by frame_chunk_size={frame_chunk_size}.")
        num_chunks = T // frame_chunk_size

        v_noisy_ts = pipeline_inputs.pop("ar_video_frame_timesteps")  # (B, T)
        v_clean_ts = pipeline_inputs.pop(
            "ar_clean_video_frame_timesteps", torch.zeros_like(v_noisy_ts)
        )
        dup_latents = torch.cat([noisy_latents, clean_latents], dim=2)  # (B, C, 2T, H, W)
        frame_timesteps = torch.cat([v_noisy_ts, v_clean_ts], dim=1)  # (B, 2T)

        # Absolute-offset RoPE for the duplicated sequence: a chunk's noisy and
        # clean copies are the SAME frame, so both halves use frame positions
        # [0..T-1] (not 0..T-1 then T..2T-1) — phases match across copies.
        frame_idx = torch.arange(T, device=dup_latents.device)
        rope_frame_index = torch.cat([frame_idx, frame_idx])  # (2T,)

        # per-sample timestep (final_layer modulation): use the noisy mean.
        timestep = pipeline_inputs.pop("timestep", v_noisy_ts.mean(dim=1))

        # F4 video: additive per-frame proprio delta into the time embedding.
        frame_proprio_emb = self._per_chunk_proprio_emb(
            proprio_per_chunk, self.proprio_video_embed, num_chunks, frame_chunk_size, dup_latents.device
        )  # (B, 2T, D_video) or None

        vstate = vb.prepare(
            latents=dup_latents,
            timestep=timestep,
            frame_timesteps=frame_timesteps,
            rope_frame_index=rope_frame_index,
            frame_proprio_emb=frame_proprio_emb,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        h, w = int(vstate.h), int(vstate.w)
        video_tokens_per_chunk = frame_chunk_size * h * w

        # --- action stream: [noisy ++ clean] along the token axis ---
        if clean_actions is None:
            raise RuntimeError("DualSystemARArchitecture.forward requires ar_clean_actions.")
        Ta = noisy_actions.shape[1]
        if Ta % num_chunks != 0:
            raise ValueError(
                f"action tokens Ta={Ta} must be divisible by num_chunks={num_chunks} "
                "(action_tokens_per_chunk must be integer)."
            )
        action_tokens_per_chunk = Ta // num_chunks

        if a_clean_ts is None:
            a_clean_ts = torch.zeros_like(a_noisy_ts)
        dup_actions = torch.cat([noisy_actions, clean_actions], dim=1)  # (B, 2Ta, Ad)
        token_timesteps = torch.cat([a_noisy_ts, a_clean_ts], dim=1)  # (B, 2Ta)

        # frame_ids / rope positions for the duplicated action stream: chunk
        # index with modality parity (2c+1), identical for a token's noisy/clean
        # copies so they share a rotary phase.
        chunk_of_token = torch.arange(Ta, device=dup_actions.device) // action_tokens_per_chunk
        action_frame_ids_single = chunk_of_token * 2 + 1  # (Ta,)
        action_frame_ids = torch.cat([action_frame_ids_single, action_frame_ids_single])  # (2Ta,)

        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

        # F4 action: additive per-token proprio delta into the time embedding.
        token_proprio_emb = self._per_chunk_proprio_emb(
            proprio_per_chunk, self.proprio_action_embed, num_chunks, action_tokens_per_chunk, dup_actions.device
        )  # (B, 2Ta, D_action) or None

        astate = ab.prepare_state(
            dup_actions,
            action_timestep,
            context=action_context,
            context_mask=action_context_mask,
            token_timesteps=token_timesteps,
            frame_ids=action_frame_ids,
            rope_positions=action_frame_ids,
            token_proprio_emb=token_proprio_emb,
        )

        # --- AR descriptor over the full concatenated [v|a] sequence ---
        ar_meta = build_ar_seq_meta(
            num_chunks=num_chunks,
            video_tokens_per_chunk=video_tokens_per_chunk,
            action_tokens_per_chunk=action_tokens_per_chunk,
            window=attn_window,
            device=dup_actions.device,
        )

        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()
        driver.run_ar_joint_loop(
            vstate,
            astate,
            ar_meta=ar_meta,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

        # --- finalize + extract the NOISY copy only ---
        video_full = vb.finalize(vstate)  # (B, C, 2T, H, W)
        video_noise_pred = video_full[:, :, :T]
        action_full = ab.extract_prediction(astate)  # (B, 2Ta, Ad)
        action_noise_pred = action_full[:, :Ta]
        return video_noise_pred, action_noise_pred

    @staticmethod
    def _per_chunk_proprio_emb(proprio_per_chunk, encoder, num_chunks, tokens_per_chunk, device):
        """Expand a per-chunk proprio ``(B, num_chunks, D_state)`` to per-token, duplicate
        for the noisy+clean copies, and embed → ``(B, 2*num_chunks*tokens_per_chunk, D_out)``.
        Returns None when proprio_per_chunk or the encoder is absent (backward-compat)."""
        if proprio_per_chunk is None or encoder is None:
            return None
        pp = proprio_per_chunk.to(device=device)
        if pp.ndim != 3 or pp.shape[1] != num_chunks:
            raise ValueError(
                f"proprio_per_chunk must be (B, num_chunks={num_chunks}, D); got {tuple(pp.shape)}."
            )
        per_token = pp.repeat_interleave(tokens_per_chunk, dim=1)  # (B, num_chunks*tpc, D)
        dup = torch.cat([per_token, per_token], dim=1)  # noisy ++ clean
        return encoder(dup.to(encoder.weight.dtype))

    # ------------------------------------------------------------------
    # AR compute_loss (override — base assumes a single non-duplicated forward)
    # ------------------------------------------------------------------

    def compute_loss(  # type: ignore[override]
        self,
        *,
        actions: Optional[Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        current_step: int = 0,  # noqa: ARG002 — AR samples its own per-chunk schedule
        decoupled_sampler=None,  # noqa: ARG002 — AR has its own per-chunk timestep sampling
        action_timestep_per_token: bool = False,  # noqa: ARG002 — AR is inherently per-chunk
        **inputs,
    ) -> dict:
        """Block-AR flow-matching loss over the duplicated [noisy|clean] sequence.

        Samples an independent diffusion timestep **per chunk** for the noisy copy
        (video and action streams have separate schedulers / snr_shift); the clean
        copy is left at t=0, or — with probability ``ar_noisy_cond_prob`` — noised
        at a small random "cond" timestep for closed-loop robustness (LingBot's
        ``noisy_cond_prob``). Loss is per-frame flow-matching MSE on the noisy-copy
        predictions only.
        """
        import torch.nn.functional as F

        vb = self.video_backbone
        action_scheduler = self.action_backbone.scheduler

        device, dtype = self.device, self.dtype
        clean_video = inputs["input_latents"].to(device=device, dtype=dtype)
        B, _, T, _, _ = clean_video.shape
        fcs = self._ar_frame_chunk_size
        if T % fcs != 0:
            raise ValueError(f"latent frames T={T} not divisible by ar_frame_chunk_size={fcs}.")
        num_chunks = T // fcs

        # ---- noisy video copy: per-chunk timestep -> per-frame sigma ----
        # NB: scheduler.sigmas/timesteps/weights live on CPU — index with CPU ids
        # then move the results to ``device`` (CUDA ids on a CPU tensor raise).
        num_ts_v = len(vb.scheduler.timesteps)
        v_chunk_ids = torch.randint(0, num_ts_v, (B, num_chunks))
        v_frame_ids = v_chunk_ids.repeat_interleave(fcs, dim=1)  # (B, T) on CPU
        v_sigma = vb.scheduler.sigmas[v_frame_ids].to(device=device, dtype=dtype)  # (B, T)
        v_ts_val = vb.scheduler.timesteps[v_frame_ids].to(device=device, dtype=dtype)  # (B, T)
        if self._ar_bootstrap_clean_prefix:
            # F3 bootstrap: chunk-0 frame-0 is the real initial obs, supplied CLEAN at
            # t=0 (sigma=0 ⇒ noisy_video[:,:,0]==clean_video[:,:,0]). Excluded from the
            # video loss below (LingBot inpaints frame-0 of chunk 0).
            v_sigma[:, 0] = 0.0
            v_ts_val[:, 0] = 0.0
        v_noise = torch.randn_like(clean_video)
        s = v_sigma.view(B, 1, T, 1, 1)
        noisy_video = (1 - s) * clean_video + s * v_noise
        v_target = v_noise - clean_video

        # ---- clean video copy: t=0 or (prob p) a small cond timestep ----
        clean_video_in, clean_video_ts = self._make_clean_copy(
            clean_video, vb.scheduler, B, T, device, dtype, frames=True
        )
        if self._ar_bootstrap_clean_prefix:
            # F3: frame 0 is the real init obs in BOTH copies (don't let the
            # clean-copy cond-noise aug touch the bootstrap frame). _make_clean_copy
            # returns contiguous tensors, so in-place writes are safe.
            clean_video_in[:, :, 0] = clean_video[:, :, 0]
            clean_video_ts[:, 0] = 0.0

        # ---- actions ----
        if actions is None or lambda_action <= 0:
            raise RuntimeError("DualSystemARArchitecture.compute_loss requires actions (joint AR path).")
        actions = actions.to(device=device, dtype=dtype)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        Ta = actions.shape[1]
        if Ta % num_chunks != 0:
            raise ValueError(f"action tokens Ta={Ta} not divisible by num_chunks={num_chunks}.")
        atok = Ta // num_chunks

        num_ts_a = len(action_scheduler.timesteps)
        a_chunk_ids = torch.randint(0, num_ts_a, (B, num_chunks))  # CPU (index CPU scheduler tensors)
        a_tok_ids = a_chunk_ids.repeat_interleave(atok, dim=1)  # (B, Ta) on CPU
        a_sigma = action_scheduler.sigmas[a_tok_ids].to(device=device, dtype=dtype)  # (B, Ta)
        a_ts_val = action_scheduler.timesteps[a_tok_ids].to(device=device, dtype=dtype)  # (B, Ta)
        a_noise = torch.randn_like(actions)
        noisy_actions = (1 - a_sigma.unsqueeze(-1)) * actions + a_sigma.unsqueeze(-1) * a_noise
        a_target = a_noise - actions

        clean_action_in, clean_action_ts = self._make_clean_copy(
            actions, action_scheduler, B, Ta, device, dtype, frames=False
        )

        # ---- forward over the duplicated sequence ----
        fwd_inputs = {
            k: v
            for k, v in inputs.items()
            if k in ("context", "context_mask", "seq_lens")
        }
        # Proprioception: forward()'s _append_proprio_context_token consumes this
        # (appends a proprio token to the text context seen by both the video and
        # action cross-attention). Required whenever use_proprioception=true.
        proprio_state = inputs.get("proprio_state")
        # F4: per-chunk proprio = robot state at each chunk's start. Either passed
        # directly as (B, num_chunks, D), or extracted from the full state sequence
        # proprio_seq (B, >=Ta, D) at indices [c*atok]. None ⇒ disabled (no-op).
        proprio_per_chunk = inputs.get("proprio_per_chunk")
        if proprio_per_chunk is None and self._proprio_per_chunk:
            proprio_seq = inputs.get("proprio_seq")
            if proprio_seq is not None:
                proprio_seq = proprio_seq.to(device=device, dtype=dtype)
                if proprio_seq.dim() == 2:
                    proprio_seq = proprio_seq.unsqueeze(0)
                idx = [c * atok for c in range(num_chunks)]
                proprio_per_chunk = proprio_seq[:, idx]  # (B, num_chunks, D)
            elif not getattr(self, "_warned_missing_proprio_seq", False):
                self._warned_missing_proprio_seq = True
                import logging

                logging.getLogger(__name__).warning(
                    "proprio_per_chunk enabled but no 'proprio_seq' in batch — per-chunk "
                    "proprio is silently INACTIVE this step (training degraded to clip-level only)."
                )
        v_pred, a_pred = self.forward(
            noisy_actions,
            None,
            proprio_state=proprio_state,
            proprio_per_chunk=proprio_per_chunk,
            latents=noisy_video,
            ar_clean_latents=clean_video_in,
            ar_clean_actions=clean_action_in,
            ar_video_frame_timesteps=v_ts_val,
            ar_clean_video_frame_timesteps=clean_video_ts,
            ar_action_token_timesteps=a_ts_val,
            ar_clean_action_token_timesteps=clean_action_ts,
            ar_frame_chunk_size=fcs,
            ar_attn_window=self._ar_attn_window,
            timestep=v_ts_val.mean(dim=1),
            **fwd_inputs,
        )

        # ---- per-frame weighted video MSE ----
        per_v = F.mse_loss(v_pred.float(), v_target.float(), reduction="none").mean(dim=(1, 3, 4))  # (B, T)
        wv = vb.scheduler.linear_timesteps_weights[v_frame_ids].to(dtype=torch.float32, device=device)  # (B, T)
        if self._ar_bootstrap_clean_prefix:
            # Exclude the clean-prefix frame 0 from the video loss (it's the real obs).
            fmask = torch.ones(B, T, dtype=torch.float32, device=device)
            fmask[:, 0] = 0.0
            loss_video = (per_v * wv * fmask).sum() / fmask.sum().clamp_min(1.0)
        else:
            loss_video = (per_v * wv).mean()

        # ---- per-token action MSE (pad-masked) ----
        per_a = F.mse_loss(a_pred.float(), a_target.float(), reduction="none").mean(dim=2)  # (B, Ta)
        action_is_pad = inputs.get("action_is_pad")
        if action_is_pad is not None and action_is_pad.shape == per_a.shape:
            keep = (~action_is_pad.bool()).to(dtype=torch.float32, device=device)
            loss_action = (per_a * keep).sum() / keep.sum().clamp(min=1.0)
        else:
            loss_action = per_a.mean()

        loss = lambda_video * loss_video + lambda_action * loss_action
        return {
            "loss": loss,
            "loss_video": (lambda_video * loss_video).detach(),
            "loss_action": (lambda_action * loss_action).detach(),
        }

    # ------------------------------------------------------------------
    # Closed-loop AR rollout (replaces base.generate's parallel loop)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ar_rollout(
        self,
        obs_latents,
        *,
        context: Tensor,
        seq_lens: Optional[Tensor] = None,
        proprio_states=None,
        frame_chunk_size: Optional[int] = None,
        attn_window: Optional[int] = None,
        video_steps: int = 4,
        action_steps: int = 4,
        action_tokens_per_chunk: int = 1,
        seed: int = 0,
    ):
        """Chunked sliding-window AR rollout over a sequence of observations.

        Operates on latents (VAE encode/decode + obs I/O belong to the deploy
        engine). Per step ``c`` with realized observation ``obs_latents[c]``:

        1. ``cache.clear_pred()`` then ingest the realized obs as a **confirmed**
           clean video chunk at frame ``2c`` (LingBot stage 3: real obs replaces
           the previous step's prediction).
        2. Denoise the next video chunk (frame ``2c+2``) for ``video_steps``
           flow-matching steps, each attending the windowed cached clean history;
           store the prediction back into the cache as ``is_pred=True``.
        3. Denoise the action chunk (frame ``2c+3``) — it sees the just-predicted
           video chunk through the cache (``2c+2 <= 2c+2``) — then ingests itself
           as **confirmed** clean history so future chunks condition on the
           executed action (see :meth:`_denoise_action_chunk`). Returns its actions.

        Returns a list of ``(B, action_tokens_per_chunk, action_dim)`` action
        chunks, one per step. The numerical *core* (cache + cache-aware attention)
        is proven equivalent to the training kernel (``test_sana_ar_inference``).
        The frame *alignment* was checked against the LingBot-VA reference server
        (``/mnt/cpfs/zch/lingbot-va/wan_va/wan_va_server.py`` ``_infer`` /
        ``_compute_kv_cache``):

        - Steady state MATCHES. Step ``c`` emits action chunk ``c+1`` conditioned
          on confirmed real-obs history through chunk ``c`` plus the just-predicted
          video chunk ``c+1`` — exactly LingBot's action chunk ``i`` (``i = c+1``)
          at ``frame_st_id = i·fcs``, which sees confirmed obs through chunk
          ``i-1`` plus its own ``update_cache=1`` predicted video. The
          ``clear_pred`` + confirmed re-ingest of the next obs at the same even
          frame id mirrors ``_compute_kv_cache`` (``clear_pred_cache`` +
          ``update_cache=2``). Window units line up too: both axes advance by
          ``fcs`` per chunk (video on even ids), so ``attn_window`` is comparable.

        - GAP [F3] — deploy-engine TODO, NOT yet handled here. LingBot additionally
          BOOTSTRAPS chunk 0 (``frame_st_id == 0``): it clamps the chunk's frame 0
          to the real init obs (inpainting) and emits an action for that first
          chunk. ``ar_rollout`` starts at action chunk 1, so it never emits the
          bootstrap action for the initial observation — a one-chunk phase offset
          at episode start. LingBot's coarse chunk (real obs + predicted frames
          inpainted together) maps non-trivially onto our finer modality-parity
          scheme (obs and prediction are SEPARATE chunks), so the deploy engine
          must decide AND verify against RoboTwin (LingBot-VA 92%) how step 0 is
          seeded — do not assume the steady-state loop is faithful at the boundary.

        RoPE uses absolute frame offsets (video ``rope_frame_index =
        arange(c*fcs, ...)``, action ``frame_ids = 2c+1``) so cached clean-K phases
        align across chunks (commit "absolute-offset RoPE"). Horizon is bounded by
        the backbones' precomputed RoPE tables (``rope.freqs`` / ``max_action_len``).

        Proprioception [F4]: when ``use_proprioception=True`` the FastWAM proprio
        context token MUST be threaded here too — training appends it to the text
        context seen by both streams' cross-attention, so omitting it at inference
        is off-distribution. ``proprio_states`` is the measured robot state; pass a
        single ``(B, D)`` / ``(B, 1, D)`` tensor (same state every step) or a
        per-step sequence aligned with ``obs_latents`` (the deploy form, since
        proprio updates each step). It is re-appended per step via
        :meth:`_append_proprio_context_token` and required (raises otherwise) when
        proprio is enabled. NOTE — semantic caveat to eval-verify: training
        conditions an entire clip on ONE proprio token, whereas the per-step rollout
        gives each chunk its own; with a single tensor here the rollout matches
        training's one-proprio semantics exactly.
        """
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None or ab is None:
            raise RuntimeError("ar_rollout requires both backbones.")
        driver = self._mot_driver or self.build_mot_driver()
        from sana_wam.model.ar.sana_ar_inference import ARLinearStateCache

        # frame_chunk_size is implied by each obs latent's frame count; the param
        # is kept for API symmetry / validation by the deploy engine.
        _ = frame_chunk_size
        window = int(attn_window or self._ar_attn_window)
        device, dtype = self.device, self.dtype
        gen = torch.Generator(device=device).manual_seed(seed)
        cache = ARLinearStateCache(num_layers=vb.num_layers, window=window)

        # Flow-matching schedules (descending sigmas, 0 appended as final target).
        vb.scheduler.set_timesteps(video_steps)
        v_sigmas = [float(s) for s in vb.scheduler.sigmas.tolist()] + [0.0]
        v_ts = vb.scheduler.timesteps
        ab.scheduler.set_timesteps(action_steps)
        a_sigmas = [float(s) for s in ab.scheduler.sigmas.tolist()] + [0.0]
        a_ts = ab.scheduler.timesteps

        actions_out = []
        for c, obs in enumerate(obs_latents):
            obs = obs.to(device=device, dtype=dtype)
            B = obs.shape[0]
            cache.clear_pred()

            # Proprio context token for this step (re-appended per step; no-op when
            # use_proprioception=False). `proprio_states` may be a single tensor
            # (broadcast to every step) or a per-step sequence aligned with obs.
            proprio_c = proprio_states[c] if isinstance(proprio_states, (list, tuple)) else proprio_states
            step_ctx, step_mask = self._rollout_step_context(context, seq_lens, proprio_c)
            # F4: per-step proprio AdaLN deltas (one robot state per step).
            v_pe, a_pe = self._rollout_proprio_deltas(proprio_c)

            # 1. ingest realized obs as confirmed clean video chunk at frame 2c.
            self._ingest_clean_video(driver, cache, obs, frame_id=2 * c, context=step_ctx, context_mask=step_mask, v_proprio=v_pe)

            if self._ar_bootstrap_clean_prefix:
                # F3 (LingBot bootstrap): emit THIS chunk's action (frame 2c+1)
                # conditioned on the just-ingested REAL obs — training-consistent
                # (action chunk c ← clean video chunk c) and offset-free at the
                # episode boundary. No look-ahead video prediction is needed.
                pred_action = self._denoise_action_chunk(
                    driver, cache, frame_id=2 * c + 1, batch=B,
                    action_tokens=action_tokens_per_chunk, a_sigmas=a_sigmas, a_ts=a_ts,
                    context=step_ctx, context_mask=step_mask, gen=gen, a_proprio=a_pe,
                )
            else:
                # Steady-state (default): predict next video chunk (frame 2c+2)...
                pred_video = self._denoise_video_chunk(
                    driver, cache, frame_id=2 * (c + 1), like=obs, v_sigmas=v_sigmas, v_ts=v_ts,
                    context=step_ctx, context_mask=step_mask, gen=gen, v_proprio=v_pe,
                )
                # ...then the action chunk (frame 2c+3) conditioned on it.
                pred_action = self._denoise_action_chunk(
                    driver, cache, frame_id=2 * (c + 1) + 1, batch=B,
                    action_tokens=action_tokens_per_chunk, a_sigmas=a_sigmas, a_ts=a_ts,
                    context=step_ctx, context_mask=step_mask, gen=gen, a_proprio=a_pe,
                )
                del pred_video  # kept in the cache; free local
            actions_out.append(pred_action)
        return actions_out

    def _rollout_step_context(self, context, seq_lens, proprio_state):
        """Augment the text context with the FastWAM proprio token (if enabled) and
        return ``(context, context_mask)`` with the mask made authoritative.

        ``_append_proprio_context_token`` appends the proprio token after the text
        (so valid tokens are not a contiguous prefix) and leaves ``seq_lens``
        unchanged — hence the rollout passes the explicit ``context_mask`` to both
        backbones rather than ``seq_lens``. No-op (returns the raw context + a
        seq_lens/all-attend mask) when ``use_proprioception=False``.
        """
        ctx_inputs = {"context": context}
        if seq_lens is not None:
            ctx_inputs["seq_lens"] = seq_lens
        ctx_inputs = self._append_proprio_context_token(ctx_inputs, proprio_state)
        ctx = ctx_inputs["context"]
        mask = ctx_inputs.get("context_mask")
        if mask is None:
            if seq_lens is not None:
                positions = torch.arange(ctx.shape[1], device=ctx.device)
                mask = positions.unsqueeze(0) < seq_lens.to(ctx.device).unsqueeze(1)
            else:
                mask = torch.ones(ctx.shape[:2], dtype=torch.bool, device=ctx.device)
        return ctx, mask

    def _rollout_proprio_deltas(self, proprio_c):
        """Per-step proprio AdaLN base vectors ``(B, D_video)`` and ``(B, D_action)``
        for the rollout/deploy path (one robot state per step), or ``(None, None)``
        when per-chunk proprio is off. Mirrors training's additive AdaLN proprio."""
        if not (
            getattr(self, "_proprio_per_chunk", False)
            and proprio_c is not None
            and getattr(self, "proprio_video_embed", None) is not None
        ):
            return None, None
        p = proprio_c
        if p.ndim == 1:
            p = p.unsqueeze(0)
        elif p.ndim == 3 and p.shape[1] == 1:
            p = p[:, 0, :]
        w = self.proprio_video_embed.weight
        p = p.to(device=w.device, dtype=w.dtype)
        return self.proprio_video_embed(p), self.proprio_action_embed(p)

    def _ingest_clean_video(self, driver, cache, clean, *, frame_id, context, context_mask, v_proprio=None):
        vb = self.video_backbone
        B, n_frames = clean.shape[0], clean.shape[2]
        zeros = torch.zeros(B, device=clean.device, dtype=clean.dtype)
        # Absolute frame offset: video chunk c is at frame_id 2c -> abs frames c*fcs..
        start_f = (frame_id // 2) * n_frames
        rope_idx = torch.arange(start_f, start_f + n_frames, device=clean.device)
        fpe = None if v_proprio is None else v_proprio[:, None, :].expand(B, n_frames, -1)
        extra = {} if fpe is None else {"frame_proprio_emb": fpe}
        vstate = vb.prepare(
            latents=clean, timestep=zeros,
            frame_timesteps=torch.zeros(B, n_frames, device=clean.device, dtype=clean.dtype),
            rope_frame_index=rope_idx,
            context=context, context_mask=context_mask, **extra,
        )
        driver.run_ar_chunk_through_backbone(vb, vstate, cache, frame_id, store_clean=True, is_pred=False)

    def _denoise_video_chunk(self, driver, cache, *, frame_id, like, v_sigmas, v_ts, context, context_mask, gen, v_proprio=None):
        vb = self.video_backbone
        device, dtype = like.device, like.dtype
        B, n_frames = like.shape[0], like.shape[2]
        start_f = (frame_id // 2) * n_frames
        rope_idx = torch.arange(start_f, start_f + n_frames, device=device)
        fpe = None if v_proprio is None else v_proprio[:, None, :].expand(B, n_frames, -1)
        extra = {} if fpe is None else {"frame_proprio_emb": fpe}
        x = torch.randn(like.shape, generator=gen, device=device, dtype=dtype)
        for i in range(len(v_sigmas) - 1):
            t_val = float(v_ts[i])
            frame_ts = torch.full((B, n_frames), t_val, device=device, dtype=dtype)
            vstate = vb.prepare(
                latents=x, timestep=torch.full((B,), t_val, device=device, dtype=dtype),
                frame_timesteps=frame_ts, rope_frame_index=rope_idx,
                context=context, context_mask=context_mask, **extra,
            )
            driver.run_ar_chunk_through_backbone(vb, vstate, cache, frame_id, store_clean=False)
            v = vb.finalize(vstate)
            x = x + v * (v_sigmas[i + 1] - v_sigmas[i])
        # store the predicted (now clean) chunk for downstream chunks.
        vstate = vb.prepare(
            latents=x, timestep=torch.zeros(B, device=device, dtype=dtype),
            frame_timesteps=torch.zeros(B, n_frames, device=device, dtype=dtype),
            rope_frame_index=rope_idx, context=context, context_mask=context_mask, **extra,
        )
        driver.run_ar_chunk_through_backbone(vb, vstate, cache, frame_id, store_clean=True, is_pred=True)
        return x

    def _denoise_action_chunk(self, driver, cache, *, frame_id, batch, action_tokens, a_sigmas, a_ts, context, context_mask, gen, a_proprio=None):
        ab = self.action_backbone
        device, dtype = self.device, self.dtype
        x = torch.randn(batch, action_tokens, ab.action_dim, generator=gen, device=device, dtype=dtype)
        frame_ids = torch.full((action_tokens,), frame_id, dtype=torch.long, device=device)
        ctx_mask = context_mask
        tpe = None if a_proprio is None else a_proprio[:, None, :]  # (B,1,Da) broadcasts over tokens
        a_extra = {} if tpe is None else {"token_proprio_emb": tpe}
        for i in range(len(a_sigmas) - 1):
            t_val = float(a_ts[i])
            token_ts = torch.full((batch, action_tokens), t_val, device=device, dtype=dtype)
            astate = ab.prepare_state(
                x, torch.full((batch,), t_val, device=device, dtype=dtype),
                context=context, context_mask=ctx_mask,
                token_timesteps=token_ts, frame_ids=frame_ids, rope_positions=frame_ids,
                **a_extra,
            )
            driver.run_ar_chunk_through_backbone(ab, astate, cache, frame_id, store_clean=False)
            v = ab.extract_prediction(astate)
            x = x + v * (a_sigmas[i + 1] - a_sigmas[i])

        # Ingest the denoised action chunk as CONFIRMED clean history (frame
        # ``2c+1``) so downstream video/action chunks condition on executed
        # actions. The training kernel puts ``a_clean`` in every noisy query's
        # visible set (``noise2clean``, ``fj < fi``); omitting it here would make
        # the closed-loop world model blind to actions it was trained to use —
        # a silent train/inference distribution shift.
        #
        # ``is_pred=False`` (unlike the predicted video chunk): there is no later
        # "real action observation" that ``clear_pred`` swaps in, so the executed
        # action must persist across steps. This mirrors LingBot-VA's two-write
        # cache scheme (``wan_va_server._compute_kv_cache`` re-ingests the real
        # measured ``state`` with ``update_cache=2`` ⇒ confirmed). Here, with no
        # external proprioceptive feedback, the model's own denoised action is the
        # best available executed history; a deploy engine that has measured robot
        # state should ingest that instead at this frame.
        astate = ab.prepare_state(
            x, torch.zeros(batch, device=device, dtype=dtype),
            context=context, context_mask=ctx_mask,
            token_timesteps=torch.zeros(batch, action_tokens, device=device, dtype=dtype),
            frame_ids=frame_ids, rope_positions=frame_ids,
            **a_extra,
        )
        driver.run_ar_chunk_through_backbone(ab, astate, cache, frame_id, store_clean=True, is_pred=False)
        return x

    def _make_clean_copy(self, clean, scheduler, B, L, device, dtype, *, frames: bool):
        """Build the clean conditioning copy: t=0, or (prob ar_noisy_cond_prob) a
        small random cond timestep applied per-sample. Returns (copy, ts_values)
        where ``ts_values`` is ``(B, L)`` for blocks_split / action per-token t-mod."""
        num_ts = len(scheduler.timesteps)
        max_cond = max(1, int(self._ar_cond_max_ratio * num_ts))
        do_cond = torch.rand(B, device=device) < self._ar_noisy_cond_prob  # (B,)
        cond_ids = torch.randint(0, max_cond, (B,))  # CPU — index CPU scheduler tensors
        raw_sigma = scheduler.sigmas[cond_ids].to(device=device, dtype=dtype)
        raw_ts = scheduler.timesteps[cond_ids].to(device=device, dtype=dtype)
        zeros = torch.zeros(B, device=device, dtype=dtype)
        cond_sigma = torch.where(do_cond, raw_sigma, zeros)
        cond_ts = torch.where(do_cond, raw_ts, zeros)
        noise = torch.randn_like(clean)
        if frames:  # (B, C, L, H, W)
            s = cond_sigma.view(B, 1, 1, 1, 1)
        else:  # actions (B, L, Ad)
            s = cond_sigma.view(B, 1, 1)
        copy = (1 - s) * clean + s * noise
        ts_values = cond_ts.view(B, 1).expand(B, L).contiguous()
        return copy, ts_values


__all__ = ["DualSystemARArchitecture"]
