"""Closed-loop streaming engine for the true-AR cached-GDN world-action model.

:class:`DualSystemGDNARArchitecture` is genuinely autoregressive: a rolling GDN
**state cache** carries the clean observed past, and each step predicts the next
chunk. This engine bridges that to the one-shot ``engine.generate`` seam the
deploy server / :class:`WAMPolicy` use, executing **exactly one AR step per
``generate`` call** — and it is built to MIRROR the teacher-forced training loop
(``gdn_ar.compute_loss``) step-for-step so train/deploy cannot drift:

    training, chunk 0:  first-frame pinned (real first obs), empty cache → predict chunk 0.
    training, chunk c:  full-noisy, cache(clean 0..c-1) → predict chunk c; ingest clean c.
    deploy,   step  0:  pin real first obs as chunk-0 frame 0, empty cache → predict chunk 0.
    deploy,   step  c:  ingest just-observed chunk c-1 (save=True) → cache depth c;
                        full-noisy → predict chunk c; execute chunk c's actions.

Step c predicts chunk c on BOTH sides — NO one-chunk look-ahead. The episode-start
chunk-0 bootstrap (first-frame pin) means the robot's first executed actions are
chunk 0's (the immediate motions), not a future chunk. Per ``generate``: at step 0,
N first-frame-pinned denoise forwards from an empty cache; at step c≥1, one clean-
ingest forward (``save_kv_cache=True``) + N denoise forwards (``save_kv_cache=False``).

Required usage (see ``configs/deploy_gdn_ar.yaml``), same contract as the block-AR
engine: :class:`WAMPolicy` greedy (``temporal_ensemble=false``,
``async_inference.mode=none``); the policy/server calls :meth:`reset` at each
episode boundary so the GDN cache + step counter clear between episodes.
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


class GDNARInferenceEngine(BaseInferenceEngine):
    """One-AR-step-per-``generate`` engine for ``DualSystemGDNARArchitecture``."""

    require_architecture = True

    def __init__(self, cfg, architecture: Optional[BaseWAMArchitecture] = None, action_backbone=None):
        super().__init__(cfg, architecture=architecture, action_backbone=action_backbone)
        arch = self.architecture
        if arch.video_backbone is None or arch.action_backbone is None:
            raise ValueError("GDNARInferenceEngine requires an architecture with both backbones.")
        self._device = arch.device
        self._dtype = arch.dtype

        # Latent frames per AR chunk (authoritative: the architecture, not the YAML).
        self._fcs = int(getattr(arch, "_frame_chunk_size", 0) or getattr(arch.video_backbone, "_chunk_size", 3))

        def _inf(key, default=None):
            return OmegaConf.select(cfg, f"inference.{key}", default=default)

        denoise_steps = int(_inf("denoise_steps", 4) or 4)
        self._video_steps = int(_inf("video_steps", denoise_steps) or denoise_steps)
        self._action_steps = int(_inf("action_steps", denoise_steps) or denoise_steps)
        seed_cfg = _inf("seed", None)
        self._seed: Optional[int] = None if seed_cfg is None else int(seed_cfg)
        # Delta actions (GDN-AR per-chunk): each generate predicts ONE chunk whose
        # tokens are displacements from the current proprio (matching training, where
        # chunk c's target is anchored to its per-chunk proprio_c). Reconstruct absolute
        # = unnormalize(delta + normalize(current_state)). Read the MODEL-side flag (the
        # GDN-AR loss applies the delta per-chunk; the dataloader stays absolute).
        self._delta_action = bool(OmegaConf.select(cfg, "model.architecture.delta_action", default=False))
        if self._delta_action and not bool(getattr(self.architecture, "uses_proprioception", False)):
            raise ValueError("model.architecture.delta_action requires proprioception (the delta anchor).")
        # Which latent band of the freshly-encoded clip is the current obs chunk:
        # leading at episode start (training chunk 0, incl. the causal first-frame
        # token), trailing (latest chunk) thereafter. Mirrors ar_engine + training.
        self._obs_latent_band = str(_inf("ar_obs_latent_band", "auto"))

        self._video_num_frames = int(self._resolve_video_num_frames(cfg))
        self._raw_num_frames = int(
            OmegaConf.select(cfg, "inference.num_frames", default=None)
            or OmegaConf.select(cfg, "dataloader.num_frames", default=81)
        )
        self._video_stride = max(1, int(OmegaConf.select(cfg, "dataloader.video_stride", default=4) or 4))
        # VAE temporal compression: Wan=4 (default, preserves existing configs),
        # LTX2/SANA-WM=8. Drives the latent-frame count used to derive num_chunks
        # and action_tokens_per_chunk — a stale 4 here under tc=8 halves the latent
        # length estimate, doubling num_chunks and misaligning actions-to-chunks
        # (the prior-0% horizon-failure class).
        self._temporal_compression = max(1, int(OmegaConf.select(cfg, "dataloader.temporal_compression", default=4) or 4))

        self._action_dim = int(arch.action_dim)
        self._action_tokens_per_chunk = self._resolve_action_tokens_per_chunk(cfg)

        self._prompt_ctx_cache: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._init_schedules_and_cache()

        logger.info(
            "GDNARInferenceEngine ready: fcs=%d video_steps=%d action_steps=%d action_dim=%d "
            "tokens/chunk=%d band=%s vnf=%d",
            self._fcs, self._video_steps, self._action_steps, self._action_dim,
            self._action_tokens_per_chunk, self._obs_latent_band, self._video_num_frames,
        )

    # ------------------------------------------------------------------ setup
    def _resolve_video_num_frames(self, cfg) -> int:
        vnf = OmegaConf.select(cfg, "inference.video_num_frames", default=None)
        if vnf is not None:
            return int(vnf)
        raw = int(OmegaConf.select(cfg, "inference.num_frames", default=None)
                  or OmegaConf.select(cfg, "dataloader.num_frames", default=81))
        stride = int(OmegaConf.select(cfg, "dataloader.video_stride", default=4) or 4)
        stride = stride if stride > 0 else 1
        return (raw - 1) // stride + 1

    def _resolve_action_tokens_per_chunk(self, cfg) -> int:
        override = (
            OmegaConf.select(cfg, "inference.action_tokens_per_chunk", default=None)
            or getattr(self.architecture, "_action_tokens_per_chunk", 0)
        )
        t_lat = 1 + (self._video_num_frames - 1) // self._temporal_compression
        num_chunks = max(1, t_lat // self._fcs)
        if override:
            tokens = int(override)
        else:
            raw = int(OmegaConf.select(cfg, "inference.num_frames", default=None)
                      or OmegaConf.select(cfg, "dataloader.num_frames", default=81))
            total_actions = raw - 1  # dataset keeps frame 0 as proprio, 1.. as actions
            # Floor (matches training's atc = Ta // total_chunks): the growing-history
            # raw window need not divide cleanly; remainder tokens fall in the dropped
            # partial tail chunk on both sides, so train/deploy atc stay consistent.
            tokens = total_actions // num_chunks
        if tokens <= 0:
            raise ValueError(f"action_tokens_per_chunk must be positive, got {tokens}")
        return tokens

    def _init_schedules_and_cache(self) -> None:
        arch = self.architecture
        vb, ab = arch.video_backbone, arch.action_backbone
        # Switch the video backbone to its cached chunk-streaming variant once.
        arch.enable_streaming()

        vb.scheduler.set_timesteps(self._video_steps)
        self._v_sigmas = [float(s) for s in vb.scheduler.sigmas.tolist()] + [0.0]
        self._v_ts = vb.scheduler.timesteps
        ab.scheduler.set_timesteps(self._action_steps)
        self._a_sigmas = [float(s) for s in ab.scheduler.sigmas.tolist()] + [0.0]
        self._a_ts = ab.scheduler.timesteps

        self._cache = arch.empty_kv_cache()
        self._gen = self._make_generator()
        self._step_c = 0

    def _make_generator(self) -> Optional[torch.Generator]:
        """Use ambient entropy by default; explicit seeds retain debug determinism."""
        if self._seed is None:
            return None
        return torch.Generator(device=self._device).manual_seed(self._seed)

    # ------------------------------------------------------------- per-step core
    @torch.no_grad()
    def _step_with_obs_latent(self, obs_latent, context, seq_lens, proprio) -> torch.Tensor:
        """One AR step from a ready obs *latent* chunk (the deploy loop body).

        Mirrors training EXACTLY (no one-chunk look-ahead): step c predicts chunk c.
        - step 0 (episode start): the robot has only the first observation. Pin its
          latent as chunk-0 frame 0 (first-frame bootstrap), denoise the rest from an
          EMPTY cache, and return chunk 0's actions — the immediate next motions.
        - step c≥1: ingest the just-observed PREVIOUS chunk (chunk c-1, executed since
          the last step) as confirmed clean past (``save_kv_cache=True``) → cache depth
          c, then denoise chunk c (full noisy, read-only cache) and return its actions.

        Bypasses VAE / text encoding so unit tests can drive it with mini backbones.
        Returns ``(B, action_tokens_per_chunk, action_dim)``.
        """
        arch = self.architecture
        ab = arch.action_backbone
        obs = obs_latent.to(device=self._device, dtype=self._dtype)
        c = self._step_c  # the chunk index to PREDICT
        K = self._fcs
        B, C, _, Hl, Wl = obs.shape
        atc = self._action_tokens_per_chunk

        # 1. Ingest the just-observed previous chunk (chunk c-1) as clean past. Skipped
        #    at episode start (c==0): nothing executed yet, cache empty, frame-0 pinned.
        if c >= 1:
            _, _, self._cache = arch.forward_chunk(
                obs,
                start_f=(c - 1) * K,
                end_f=c * K,
                kv_cache=self._cache,
                video_timestep=torch.zeros(B, device=self._device),
                noisy_actions=None,
                context=context,
                context_mask=None,
                seq_lens=seq_lens,
                proprio_state=proprio,
                save_kv_cache=True,
            )

        # 2. Denoise chunk c given the clean cache (read-only). Full-noisy for c≥1;
        #    first-frame-pinned (the real first obs) for the bootstrap chunk 0.
        cur_start, cur_end = c * K, (c + 1) * K
        positions = torch.arange(c * atc, (c + 1) * atc, device=self._device)
        action_freqs = ab._get_rope_freqs_at(positions)
        n_steps = min(len(self._v_ts), len(self._a_ts))

        pin0 = c == 0
        first_latent = obs[:, :, 0] if pin0 else None  # real first-obs latent to pin

        video = torch.randn(B, C, K, Hl, Wl, generator=self._gen, device=self._device, dtype=self._dtype)
        if pin0:
            video[:, :, 0] = first_latent  # pin frame 0 = real first observation
        actions = torch.randn(B, atc, self._action_dim, generator=self._gen, device=self._device, dtype=self._dtype)
        for i in range(n_steps):
            v_t = self._v_ts[i].to(device=self._device, dtype=self._dtype)
            a_t = self._a_ts[i].to(device=self._device, dtype=self._dtype)
            v_pred, a_pred, _ = arch.forward_chunk(
                video,
                start_f=cur_start,
                end_f=cur_end,
                kv_cache=self._cache,
                video_timestep=v_t.reshape(1).expand(B),
                noisy_actions=actions,
                action_timestep=a_t.reshape(1).expand(B),
                context=context,
                context_mask=None,
                seq_lens=seq_lens,
                proprio_state=proprio,
                action_freqs=action_freqs,
                save_kv_cache=False,  # cache stays the clean past
            )
            v_dsig = self._v_sigmas[i + 1] - self._v_sigmas[i]
            a_dsig = self._a_sigmas[i + 1] - self._a_sigmas[i]
            video = video + v_pred.to(self._dtype) * v_dsig
            actions = actions + a_pred.to(self._dtype) * a_dsig
            if pin0:
                video[:, :, 0] = first_latent  # re-pin the clean first observation

        self._step_c += 1
        return actions

    # ----------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        prompt = conditions.get("prompt", "") or ""
        obs_latent = self._build_obs_chunk(conditions)
        context, seq_lens = self._encode_prompt(prompt)
        proprio = self._prep_proprio(conditions.get("proprio_state"))

        pred_action = self._step_with_obs_latent(obs_latent, context, seq_lens, proprio)

        actions = pred_action.squeeze(0).float().cpu().numpy()
        # Delta actions: the chunk's tokens are displacements from the current state in
        # the normalized space — add the normalized current proprio (the per-chunk
        # anchor training used) BEFORE unnormalizing, so rot6d/grip land on the training
        # manifold (eef20d→ee16d then orthonormalizes the recovered 6d rotation).
        if self._delta_action:
            ps = conditions.get("proprio_state")
            if ps is None:
                raise ValueError(
                    "delta_action deploy requires conditions['proprio_state'] (the delta anchor)."
                )
            anchor_norm = np.asarray(
                self.architecture.normalize_deploy_proprio(np.asarray(ps, dtype=np.float32).reshape(-1)),
                dtype=np.float32,
            )
            actions = actions + anchor_norm[None, :]
        normalizer = getattr(self.architecture, "action_normalizer", None)
        if normalizer is not None:
            actions = normalizer.unnormalize(actions)
        return {"actions": actions, "video": None}

    def reset(self) -> None:
        """Clear GDN state and rebuild the configured optional RNG."""
        self._cache = self.architecture.empty_kv_cache()
        self._step_c = 0
        self._gen = self._make_generator()

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
                "GDNARInferenceEngine.generate requires conditions['obs_history'] or ['first_frame_image']."
            )
        return frames

    def _build_obs_clip(self, conditions: dict) -> list:
        """Reconstruct the training-cadence clip: last ``num_frames`` per-step frames
        subsampled by ``video_stride`` (= the dataset's ``range(0, num_frames,
        video_stride)``) so encoded latents match training's motion. Left-pads at
        episode start."""
        frames = self._recent_frames(conditions)
        recent = frames[-self._raw_num_frames:]
        if len(recent) < self._raw_num_frames:
            recent = [recent[0]] * (self._raw_num_frames - len(recent)) + recent
        clip = recent[:: self._video_stride]
        if len(clip) > self._video_num_frames:
            clip = clip[-self._video_num_frames:]
        elif len(clip) < self._video_num_frames:
            clip = [clip[0]] * (self._video_num_frames - len(clip)) + clip
        return clip

    def _build_obs_chunk(self, conditions: dict) -> torch.Tensor:
        """Encode the recent observation into a ``(B, C, fcs, H_lat, W_lat)`` latent chunk."""
        vb = self.architecture.video_backbone
        if vb._pipe.vae is None:
            raise RuntimeError("GDNARInferenceEngine obs encoding requires a loaded Wan VAE.")
        from sana_wam.model.video_backbone.sana.adapter import _pil_video_to_tensor

        clip = self._build_obs_clip(conditions)
        video = _pil_video_to_tensor([clip]).to(device=self._device, dtype=self._dtype)
        latents = vb._pipe.vae.encode([video[0]], device=self._device, tiled=True)
        latents = latents.to(device=self._device, dtype=self._dtype)
        if latents.shape[2] < self._fcs:
            raise RuntimeError(
                f"VAE produced {latents.shape[2]} latent frames < frame_chunk_size={self._fcs}; "
                f"increase inference.video_num_frames (got {self._video_num_frames})."
            )
        return self._select_obs_band(latents)

    def _select_obs_band(self, latents: torch.Tensor) -> torch.Tensor:
        """Pick the obs latent band — leading (training chunk-0, incl. causal first-frame)
        at episode start, trailing (latest chunk) thereafter."""
        band = self._obs_latent_band
        if band == "auto":
            band = "leading" if self._step_c == 0 else "trailing"
        if band == "leading":
            return latents[:, :, : self._fcs].contiguous()
        return latents[:, :, -self._fcs:].contiguous()


__all__ = ["GDNARInferenceEngine"]
