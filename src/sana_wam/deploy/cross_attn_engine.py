"""Inference engine for the GDN cross-attention TI2V world-action model.

Unlike the block-AR :class:`ARInferenceEngine` (stateful KV cache, one chunk per
``generate``), the cross-attn architecture is **not** autoregressive: each
``generate`` is a one-shot flow-matching denoise of the whole clip, conditioned
on the observed real frames (TI2V clean-prefix). The GDN backbone's own
chunk-causal masking provides temporal structure inside one forward.

Two modes:
  - **non-streaming** (``inference.streaming=false``): conditions on the single
    newest observation frame (legacy frame-0-only TI2V).
  - **streaming** (default): growing-history clean-prefix. The engine rebuilds the
    real observed history clip from ``conditions['obs_history']`` (WAMPolicy appends
    one frame per sim step), anchored at episode frame 0 and subsampled by
    ``video_stride`` — *exactly* the dataset's ``range(0, num_frames, video_stride)``
    — then VAE-encodes it so the clean-prefix latents are genuine causal latents
    carrying real motion (latent frame i≥1 aggregates ``video_stride*4`` real frames),
    matching the growing-history training distribution. The leading observed latents
    are pinned as the clean prefix (capped at ``T - predict_horizon`` so there is
    always room to predict); the rest of the clip is denoised. This is *stateless*
    w.r.t. latents — the history lives in ``obs_history`` and is re-encoded each call,
    so there is no rolling buffer to bleed across episodes (cf. the earlier
    single-image-stacking prefix, which fed off-distribution frame-0-type latents at
    every position). ``reset`` only reseeds the RNG + step counter.

NOTE: streaming requires that ``obs_history`` reach back to episode frame 0, i.e.
``policy.history_len >= dataloader.num_frames`` (= the training ``raw_window_len``,
which growing-history sets to span the longest episode). If a deploy episode runs
longer than that window, the frame-0 anchor can no longer represent "now"; the
engine logs a warning and clamps to the leading window.

Per ``generate``:
  1. encode prompt → context; (streaming) rebuild + VAE-encode the observed-history
     clip → real causal clean-prefix latents; normalize proprio.
  2. initialise noisy video + action at sigma=1 (pure noise), pin the clean prefix
     [0, P) = observed history.
  3. run the action + video flow-matching schedules jointly: at each step call
     ``architecture.forward`` to get (v_pred, a_pred), step both schedulers, re-pin
     the clean prefix.
  4. denormalize and return the predicted action trajectory (+ optionally decoded
     video).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from sana_wam.deploy.base import BaseInferenceEngine
from sana_wam.model.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)

_PROMPT_CTX_CACHE_MAXSIZE = 32


class CrossAttnInferenceEngine(BaseInferenceEngine):
    """Stateless one-shot TI2V denoise engine for the cross-attn GDN architecture."""

    require_architecture = True

    def __init__(self, cfg, architecture: Optional[BaseWAMArchitecture] = None, action_backbone=None):
        super().__init__(cfg, architecture=architecture, action_backbone=action_backbone)
        arch = self.architecture
        if arch.video_backbone is None or arch.action_backbone is None:
            raise ValueError("CrossAttnInferenceEngine requires both backbones.")
        self._device = arch.device
        self._dtype = arch.dtype

        def _inf(key, default=None):
            return OmegaConf.select(cfg, f"inference.{key}", default=default)

        denoise_steps = int(_inf("denoise_steps", 4) or 4)
        self._video_steps = int(_inf("video_steps", denoise_steps) or denoise_steps)
        self._action_steps = int(_inf("action_steps", denoise_steps) or denoise_steps)
        self._seed = int(_inf("seed", 0) or 0)

        self._action_dim = int(arch.action_dim)
        self._action_tokens = int(_inf("action_tokens", 0) or 0)  # 0 ⇒ infer from clip
        self._raw_num_frames = int(
            OmegaConf.select(cfg, "inference.num_frames", default=None)
            or OmegaConf.select(cfg, "dataloader.num_frames", default=49)
        )
        self._video_stride = max(1, int(OmegaConf.select(cfg, "dataloader.video_stride", default=4) or 4))
        # VAE temporal compression (Wan=4 default, LTX2/SANA-WM=8) — see GDNAR engine.
        self._temporal_compression = max(1, int(OmegaConf.select(cfg, "dataloader.temporal_compression", default=4) or 4))
        self._video_num_frames = (self._raw_num_frames - 1) // self._video_stride + 1

        # Streaming (growing-history) deploy. When enabled, each generate rebuilds
        # the observed-history clip from conditions['obs_history'] (anchored at
        # episode frame 0, subsampled by video_stride) and VAE-encodes it, so the
        # clean-prefix latents are REAL causal latents at the training cadence —
        # matching the growing-history training distribution. The leading observed
        # latents are pinned as the clean prefix (capped at T - predict_horizon).
        self._streaming = bool(_inf("streaming", True))
        self._predict_horizon = max(1, int(_inf("predict_horizon_frames", 1) or 1))
        self._obs_latents: Optional[torch.Tensor] = None  # (1, C, P, Hl, Wl) last pinned prefix (debug)
        self._warned_overlong = False
        self._step_c = 0
        # Design B: pin the executed-past actions as a clean prefix so the model
        # conditions on them. Gated on the architecture flag (set only when the
        # checkpoint was trained with action_clean_prefix); otherwise legacy behavior.
        self._action_clean_prefix = bool(getattr(arch, "_action_clean_prefix", False))

        self._prompt_ctx_cache: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._gen = torch.Generator(device=self._device).manual_seed(self._seed)

        logger.info(
            "CrossAttnInferenceEngine ready: video_steps=%d action_steps=%d action_dim=%d vnf=%d "
            "streaming=%s predict_horizon=%d",
            self._video_steps, self._action_steps, self._action_dim, self._video_num_frames,
            self._streaming, self._predict_horizon,
        )

    # ----------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        arch = self.architecture
        vb, ab = arch.video_backbone, arch.action_backbone

        prompt = conditions.get("prompt", "") or ""
        context, seq_lens = self._encode_prompt(prompt)
        proprio = self._prep_proprio(conditions.get("proprio_state"))

        B = 1
        T = self._video_num_frames_latent()
        atok = self._action_tokens or (self._raw_num_frames - 1)
        if T < 1:
            raise ValueError(f"generated latent length T={T} must be >= 1 (video_num_frames too small).")

        # Clean prefix. Non-streaming: legacy frame-0-only TI2V (single newest obs).
        # Streaming: rebuild the observed-history clip from obs_history (anchored at
        # episode frame 0, subsampled at the training video_stride) and VAE-encode it,
        # so prefix frames are REAL causal latents carrying motion — matching the
        # growing-history training distribution. Pin the leading observed latents,
        # capped at ``T - predict_horizon`` so there is always room to predict.
        if self._streaming:
            max_prefix = max(1, T - self._predict_horizon)
            obs_latents = self._encode_clip(self._build_obs_clip(conditions))  # (1, C, P_real, Hl, Wl)
            P = min(obs_latents.shape[2], max_prefix)
            prefix = obs_latents[:, :, :P].contiguous()
        else:
            prefix = self._encode_first_frame(conditions)  # (1, C, 1, Hl, Wl)
            # Must be exactly one latent frame; otherwise the pin below would silently
            # broadcast/clip and corrupt the TI2V conditioning.
            if prefix.shape[2] != 1:
                raise ValueError(
                    f"first_frame_latent must have 1 latent frame, got {prefix.shape[2]} "
                    f"(shape {tuple(prefix.shape)})."
                )
            P = 1
        self._obs_latents = prefix  # last pinned prefix (observability/debug)
        C, Hl, Wl = prefix.shape[1], prefix.shape[3], prefix.shape[4]

        # Flow-matching schedules.
        vb.scheduler.set_timesteps(self._video_steps)
        ab.scheduler.set_timesteps(self._action_steps)
        v_sigmas = [float(s) for s in vb.scheduler.sigmas.tolist()] + [0.0]
        a_sigmas = [float(s) for s in ab.scheduler.sigmas.tolist()] + [0.0]
        v_ts, a_ts = vb.scheduler.timesteps, ab.scheduler.timesteps
        n_steps = min(len(v_ts), len(a_ts))

        # Clean action prefix (Design B): the executed past actions, re-normalized.
        # Pin them so the action stream conditions on the real past (mirrors the video
        # clean prefix). k_a = number of executed actions (capped to leave a future);
        # 0 at episode start (bootstrap) ⇒ no action prefix.
        act_prefix = (
            self._prep_action_prefix(conditions, atok)
            if (self._streaming and self._action_clean_prefix)
            else None
        )
        k_a = act_prefix.shape[1] if act_prefix is not None else 0

        # Init: pure noise (sigma≈1); pin the clean prefix [0, P) = observed history.
        video = torch.randn(B, C, T, Hl, Wl, generator=self._gen, device=self._device, dtype=self._dtype)
        actions = torch.randn(B, atok, self._action_dim, generator=self._gen, device=self._device, dtype=self._dtype)
        video[:, :, :P] = prefix  # TI2V clean prefix (observed history)
        if k_a > 0:
            actions[:, :k_a] = act_prefix  # clean action prefix (executed past)

        for i in range(n_steps):
            # Cast timesteps to the model dtype: TimestepEmbedding's sinusoid
            # inherits the timestep dtype, and a float32 timestep would feed a
            # bf16 MLP (dtype mismatch). Keep both streams in self._dtype.
            v_t = v_ts[i].to(device=self._device, dtype=self._dtype)
            a_t = a_ts[i].to(device=self._device, dtype=self._dtype)
            v_pred, a_pred = arch.forward(
                actions,
                a_t.reshape(1).expand(B),
                proprio_state=proprio,
                latents=video,
                timestep=v_t.reshape(1).expand(B),
                context=context,
                seq_lens=seq_lens,
            )
            # Euler flow step: x <- x + pred * (sigma_next - sigma).
            v_dsig = v_sigmas[i + 1] - v_sigmas[i]
            a_dsig = a_sigmas[i + 1] - a_sigmas[i]
            video = video + v_pred.to(self._dtype) * v_dsig
            actions = actions + a_pred.to(self._dtype) * a_dsig
            video[:, :, :P] = prefix  # re-pin the clean observed history
            if k_a > 0:
                actions[:, :k_a] = act_prefix  # re-pin the clean action prefix

        # Streaming: the action trajectory is anchored at episode frame 0, but the
        # leading steps lead up to the current state (the observed/executed past).
        # Return only the FUTURE actions so the policy executes from "now"
        # (actions[0] = next action). The boundary is where the past ends: the pinned
        # action prefix k_a if present, else the video boundary tc*(P-1)*video_stride.
        if self._streaming:
            boundary = max(k_a, self._temporal_compression * (P - 1) * self._video_stride)
            boundary = min(boundary, actions.shape[1] - 1)
            if boundary > 0:
                actions = actions[:, boundary:]

        self._step_c += 1
        actions_np = actions.squeeze(0).float().cpu().numpy()
        normalizer = getattr(arch, "action_normalizer", None)
        if normalizer is not None:
            actions_np = normalizer.unnormalize(actions_np)
        return {"actions": actions_np, "video": None}

    def reset(self) -> None:
        """Reseed the RNG + step counter at episode boundaries.

        Streaming sources its clean-prefix history from ``conditions['obs_history']``
        (owned by WAMPolicy) and re-encodes it each call, so the engine holds no
        rolling latent buffer that could bleed across episodes — ``_obs_latents`` is
        only the last pinned prefix kept for observability. We still reset it and the
        step counter, and reseed the RNG for reproducible rollouts. WAMPolicy calls
        this on episode reset (which also clears its own obs_history).
        """
        self._obs_latents = None
        self._warned_overlong = False
        self._step_c = 0
        self._gen = torch.Generator(device=self._device).manual_seed(self._seed)

    # --------------------------------------------------------------- conditions
    def _encode_prompt(self, prompt: str):
        if prompt in self._prompt_ctx_cache:
            self._prompt_ctx_cache.move_to_end(prompt)
            return self._prompt_ctx_cache[prompt]
        vb = self.architecture.video_backbone
        context = vb._encode_text(prompt).to(device=self._device, dtype=self._dtype)
        seq_lens = torch.full((context.shape[0],), context.shape[1], dtype=torch.long, device=self._device)
        self._prompt_ctx_cache[prompt] = (context, seq_lens)
        while len(self._prompt_ctx_cache) > _PROMPT_CTX_CACHE_MAXSIZE:
            self._prompt_ctx_cache.popitem(last=False)
        return context, seq_lens

    def _prep_proprio(self, proprio_state):
        if not self.architecture.uses_proprioception or proprio_state is None:
            return None
        norm = self.architecture.normalize_deploy_proprio(
            np.asarray(proprio_state, dtype=np.float32).reshape(-1)
        )
        return torch.from_numpy(np.asarray(norm, dtype=np.float32)).to(
            device=self._device, dtype=self._dtype
        ).unsqueeze(0)

    def _prep_action_prefix(self, conditions: dict, atok: int) -> Optional[torch.Tensor]:
        """Executed-past actions → clean action prefix ``(1, k_a, action_dim)``.

        ``conditions['action_history']`` is the actions the policy sent to the env
        (oldest→newest, raw/unnormalized scale). Re-normalize into the model's
        training space (the inverse of the ``action_normalizer.unnormalize`` applied
        to the engine's output) and align token i ↔ the i-th executed action (token i
        is the action into frame i+1, anchored at episode frame 0 — which requires
        ``history_len`` to span the episode). Cap ``k_a`` to leave >=1 future token;
        returns None when no history (episode-start bootstrap)."""
        hist = conditions.get("action_history") or []
        if not hist:
            return None
        arr = np.asarray(hist, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        k_a = min(arr.shape[0], max(0, int(atok) - 1))  # keep at least one future token
        if k_a <= 0:
            return None
        arr = arr[:k_a]
        normalizer = getattr(self.architecture, "action_normalizer", None)
        if normalizer is not None:
            arr = np.asarray(normalizer.normalize(arr), dtype=np.float32)
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device=self._device, dtype=self._dtype)
        return t.unsqueeze(0)  # (1, k_a, action_dim)

    def _recent_frames(self, conditions: dict) -> list:
        frames = [
            o.get("image")
            for o in (conditions.get("obs_history") or [])
            if isinstance(o, dict) and o.get("image") is not None
        ]
        if not frames:
            ff = conditions.get("first_frame_image")
            if ff:
                frames = [ff[0]]
        if not frames:
            raise ValueError(
                "CrossAttnInferenceEngine.generate requires conditions['obs_history'] or ['first_frame_image']."
            )
        return frames

    def _encode_first_frame(self, conditions: dict) -> torch.Tensor:
        """Encode the current observation frame → a single clean latent frame.

        The observation is the real first frame the TI2V model conditions on. We
        tile it to the VAE's minimum temporal window, encode, and take latent
        frame 0 (the causal VAE's first latent depends only on input frame 0)."""
        vb = self.architecture.video_backbone
        if vb._pipe.vae is None:
            raise RuntimeError("CrossAttnInferenceEngine requires a loaded Wan VAE.")
        from sana_wam.model.video_backbone.sana.adapter import _pil_video_to_tensor

        frame = self._recent_frames(conditions)[-1]
        clip = [frame] * self._video_num_frames  # zero-motion tile; only frame-0 latent used
        video = _pil_video_to_tensor([clip]).to(device=self._device, dtype=self._dtype)
        latents = vb._pipe.vae.encode([video[0]], device=self._device, tiled=True)
        latents = latents.to(device=self._device, dtype=self._dtype)
        return latents[:, :, :1].contiguous()  # (1, C, 1, Hl, Wl)

    def _build_obs_clip(self, conditions: dict) -> list:
        """Rebuild the observed-history video clip for streaming, anchored at episode
        frame 0 and subsampled at the training ``video_stride``.

        Mirrors the dataset's ``range(0, num_frames, video_stride)`` so the encoded
        latents carry the same per-frame motion the model trained on. ``obs_history``
        holds per-sim-step frames oldest→newest; we take the leading window (from
        episode frame 0) up to the full training window and subsample. The clip grows
        with the episode, so the encoded clean prefix grows exactly like training's
        growing logical length. We do NOT pad the clip out to the full clip length —
        only the genuinely observed frames are encoded; the unobserved future is what
        the denoise loop predicts.
        """
        frames = self._recent_frames(conditions)  # oldest→newest, frame 0 = episode start
        # Anchor at frame 0; one training window spans raw_num_frames raw frames.
        if len(frames) > self._raw_num_frames and not self._warned_overlong:
            logger.warning(
                "CrossAttnInferenceEngine: obs_history (%d frames) exceeds the training "
                "window raw_num_frames=%d; the frame-0 anchor can no longer represent "
                "'now' — clamping to the leading window (out-of-distribution).",
                len(frames), self._raw_num_frames,
            )
            self._warned_overlong = True
        recent = frames[: self._raw_num_frames]
        clip = recent[:: self._video_stride]  # training cadence; len grows with the episode
        if len(clip) > self._video_num_frames:  # never encode beyond the clip length
            clip = clip[: self._video_num_frames]
        # Snap to a VAE-valid temporal length. The causal VAE requires (N-1) % tc == 0
        # — the LTX2 VAE (tc=8) ENFORCES this (N=64 raises an unflatten error mid-VAE),
        # whereas the Wan VAE (tc=4) tolerated arbitrary lengths. The obs clip grows one
        # frame per step, so without this it eventually hits an invalid length and
        # crashes the episode. Truncate to the largest valid 1+tc*k <= len (drops at
        # most tc-1 trailing frames; the window includes them at the next valid length;
        # proprio still carries 'now'). No-op when already valid (e.g. capped windows).
        tc = self._temporal_compression
        n = len(clip)
        if tc > 1 and n > 1:
            valid = 1 + ((n - 1) // tc) * tc
            if valid != n:
                clip = clip[:valid]
        return clip or [frames[0]]

    def _encode_clip(self, clip: list) -> torch.Tensor:
        """VAE-encode an observation clip → ``(1, C, P, Hl, Wl)`` causal latents.

        Stubbable seam (the mini test backbone has no VAE). For a clip of N video
        frames the causal Wan VAE returns ``1 + (N-1)//4`` latent frames; latent
        frame i≥1 aggregates 4 successive real frames, so these are genuine
        motion-bearing latents (unlike a stack of single-image frame-0 encodes).
        """
        vb = self.architecture.video_backbone
        if vb._pipe.vae is None:
            raise RuntimeError("CrossAttnInferenceEngine streaming requires a loaded Wan VAE.")
        from sana_wam.model.video_backbone.sana.adapter import _pil_video_to_tensor

        video = _pil_video_to_tensor([clip]).to(device=self._device, dtype=self._dtype)
        latents = vb._pipe.vae.encode([video[0]], device=self._device, tiled=True)
        return latents.to(device=self._device, dtype=self._dtype)  # (1, C, P, Hl, Wl)

    def _video_num_frames_latent(self) -> int:
        """Latent temporal length T for the generated clip (causal VAE)."""
        return 1 + (self._video_num_frames - 1) // self._temporal_compression
