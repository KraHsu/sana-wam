"""Closed-loop inference engine for the block-autoregressive SANA WAM.

The block-AR architecture (:class:`DualSystemARArchitecture`) is *stateful*:
``ar_rollout`` walks a sequence of realized observations, ingesting each one as
confirmed clean history into a persistent linear-state KV cache and emitting one
action chunk per step. That does not fit the one-shot ``engine.generate`` seam
the deploy server / :class:`WAMPolicy` are built around — which is stateless and
re-runs from scratch each call.

:class:`ARInferenceEngine` bridges the two: it owns the persistent cache, driver,
flow-matching schedules, monotonic RNG, step counter, prompt-context cache and a
rolling pixel-frame buffer, and executes **exactly one AR step per ``generate``
call** by reusing the architecture's own per-step helpers verbatim
(``_rollout_step_context`` / ``_ingest_clean_video`` / ``_denoise_video_chunk`` /
``_denoise_action_chunk``). Driving ``generate`` N times over the realized obs
stream is therefore numerically identical to a single ``ar_rollout`` over those
same observations (see ``tests/test_ar_rollout_engine.py`` equivalence gate).

Required usage (see ``configs/deploy_ar_sana.yaml``):

- :class:`WAMPolicy` MUST run in **greedy** mode (``execute_horizon=null``,
  ``temporal_ensemble=false``, ``async_inference.mode=none``). Each ``generate``
  ingests one realized obs and advances the cache by one chunk; receding-horizon,
  temporal ensembling and async all double-advance / reorder the cache and break
  the AR frame alignment.
- The policy/server must call :meth:`reset` at each episode boundary (wired via
  ``WAMPolicy.reset`` → ``engine.reset``) so the cache + step counter + pixel
  buffer are cleared between episodes.

Deploy-semantics knobs (eval-verification targets, handoff §2c F3/F4):

- ``inference.ar_obs_chunk_mode``: ``rolling_buffer`` (default) rebuilds the obs
  clip from the per-sim-step ``obs_history`` (WAMPolicy appends one frame every
  step) by taking the last ``num_frames`` frames and subsampling by
  ``video_stride`` — exactly the dataset's ``range(0, num_frames, video_stride)``
  — so the encoded latents match training's pixel cadence/motion. (A naive
  one-frame-per-``generate`` buffer would be ~``action_tokens_per_chunk``× too
  sparse and motion-off-distribution.) ``repeat_frame`` tiles the current frame
  (zero motion; debug only).
- ``inference.ar_obs_latent_band``: ``auto`` (default) takes the leading latents
  (training chunk 0, incl. the causal first-frame token) at episode start (step 0,
  whose obs is assigned rope[0,1]) and the trailing latents (latest chunk) after —
  matching how training assigns the first-chunk vs continuation-chunk band.
  ``leading`` / ``trailing`` force one band (F3 eval A/B). NOTE: the remaining F3
  piece — emitting a bootstrap action for the initial obs (training's chunk-0
  action at frame 1) to remove the one-chunk phase offset — is not yet done here
  (touches the validated rollout core); the obs at deploy start is in-distribution,
  but the first emitted action is still the look-ahead chunk.
- ``inference.ar_proprio_mode``: ``per_step`` (default) conditions each chunk on
  the current robot state; ``single`` latches the episode's first state (exact
  training-clip parity). This is the F4 train/inference choice to A/B at eval.
- F3 bootstrap: this engine starts at action chunk 1 (``ar_rollout`` semantics);
  there is a one-chunk phase offset at episode start vs LingBot's inpainted
  chunk-0 bootstrap. Verify against RoboTwin before trusting boundary behaviour.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from sana_wam.deploy.base import BaseInferenceEngine
from sana_wam.model.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)

_PROMPT_CTX_CACHE_MAXSIZE = 32


class ARInferenceEngine(BaseInferenceEngine):
    """One-AR-step-per-``generate`` engine for ``DualSystemARArchitecture``."""

    require_architecture = True

    def __init__(self, cfg, architecture: Optional[BaseWAMArchitecture] = None, action_backbone=None):
        super().__init__(cfg, architecture=architecture, action_backbone=action_backbone)
        arch = self.architecture
        if arch.video_backbone is None or arch.action_backbone is None:
            raise ValueError("ARInferenceEngine requires an architecture with both backbones.")
        self._device = arch.device
        self._dtype = arch.dtype

        # --- AR knobs off the architecture (authoritative) ---
        self._fcs = int(getattr(arch, "_ar_frame_chunk_size", 1))
        self._window = int(getattr(arch, "_ar_attn_window", 10_000))

        # --- deploy knobs off cfg.inference (with sane fallbacks) ---
        def _inf(key, default=None):
            return OmegaConf.select(cfg, f"inference.{key}", default=default)

        denoise_steps = int(_inf("denoise_steps", 4) or 4)
        self._video_steps = int(_inf("video_steps", denoise_steps) or denoise_steps)
        self._action_steps = int(_inf("action_steps", denoise_steps) or denoise_steps)
        self._seed = int(_inf("seed", 0) or 0)
        self._obs_chunk_mode = str(_inf("ar_obs_chunk_mode", "rolling_buffer"))
        self._proprio_mode = str(_inf("ar_proprio_mode", "per_step"))
        # Which latent band of the freshly-encoded clip is the current obs chunk.
        #   auto (default) = leading at step 0, trailing after — matches training,
        #     whose chunk 0 is lat[0:fcs] (incl. the causal first-frame token) @ rope[0,1]
        #     and whose later/current chunk is the trailing band.
        #   leading  = always first latents (training chunk-0 band).
        #   trailing = always most-recent latents (a training clip's last chunk).
        self._obs_latent_band = str(_inf("ar_obs_latent_band", "auto"))

        self._video_num_frames = int(self._resolve_video_num_frames(cfg))
        self._height = int(_inf("height", 480) or 480)
        self._width = int(_inf("width", 832) or 832)
        # Raw (pre-stride) frame window + stride — the engine must rebuild the obs
        # clip at the SAME pixel cadence training used (dataset takes
        # range(0, num_frames, video_stride)); greedy WAMPolicy otherwise only
        # hands the engine one frame per chunk (~24 sim steps apart) vs training's
        # video_stride (~4), making every obs latent motion-off-distribution.
        self._raw_num_frames = int(
            OmegaConf.select(cfg, "inference.num_frames", default=None)
            or OmegaConf.select(cfg, "dataloader.num_frames", default=33)
        )
        self._video_stride = max(1, int(OmegaConf.select(cfg, "dataloader.video_stride", default=1) or 1))
        # VAE temporal compression (Wan=4 default, LTX2/SANA-WM=8) — see GDNAR engine.
        self._temporal_compression = max(1, int(OmegaConf.select(cfg, "dataloader.temporal_compression", default=4) or 4))

        # --- action geometry (from the trained backbone, never the YAML literal) ---
        self._action_dim = int(arch.action_dim)
        self._action_tokens_per_chunk = self._resolve_action_tokens_per_chunk(cfg)

        # --- persistent rollout state (per-step frames come from WAMPolicy.obs_history) ---
        self._prompt_ctx_cache: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._latched_proprio: Optional[np.ndarray] = None
        self._init_schedules_and_cache()

        logger.info(
            "ARInferenceEngine ready: fcs=%d window=%d video_steps=%d action_steps=%d "
            "action_dim=%d tokens/chunk=%d obs_chunk_mode=%s proprio_mode=%s vnf=%d res=%dx%d",
            self._fcs, self._window, self._video_steps, self._action_steps, self._action_dim,
            self._action_tokens_per_chunk, self._obs_chunk_mode, self._proprio_mode,
            self._video_num_frames, self._height, self._width,
        )

    # ------------------------------------------------------------------ setup
    def _resolve_video_num_frames(self, cfg) -> int:
        vnf = OmegaConf.select(cfg, "inference.video_num_frames", default=None)
        if vnf is not None:
            return int(vnf)
        raw = int(OmegaConf.select(cfg, "inference.num_frames", default=None)
                  or OmegaConf.select(cfg, "dataloader.num_frames", default=33))
        stride = int(OmegaConf.select(cfg, "dataloader.video_stride", default=1) or 1)
        stride = stride if stride > 0 else 1
        return (raw - 1) // stride + 1

    def _resolve_action_tokens_per_chunk(self, cfg) -> int:
        override = OmegaConf.select(cfg, "inference.ar_action_tokens_per_chunk", default=None)
        # Latent frame count after the causal Wan VAE, then chunked.
        t_lat = 1 + (self._video_num_frames - 1) // self._temporal_compression
        num_chunks = max(1, t_lat // self._fcs)
        if override is not None:
            tokens = int(override)
        else:
            raw = int(OmegaConf.select(cfg, "inference.num_frames", default=None)
                      or OmegaConf.select(cfg, "dataloader.num_frames", default=33))
            total_actions = raw - 1  # dataset keeps frame 0 as proprio, 1.. as actions
            if total_actions % num_chunks != 0:
                raise ValueError(
                    f"action count {total_actions} not divisible by num_chunks={num_chunks} "
                    f"(t_lat={t_lat}, fcs={self._fcs}); set inference.ar_action_tokens_per_chunk."
                )
            tokens = total_actions // num_chunks
        if tokens <= 0:
            raise ValueError(f"action_tokens_per_chunk must be positive, got {tokens}")
        return tokens

    def _init_schedules_and_cache(self) -> None:
        """Build driver, flow-matching schedules, cache and RNG (mirrors ar_rollout)."""
        from sana_wam.model.ar.sana_ar_inference import ARLinearStateCache

        arch = self.architecture
        vb, ab = arch.video_backbone, arch.action_backbone
        self._driver = arch._mot_driver or arch.build_mot_driver()

        vb.scheduler.set_timesteps(self._video_steps)
        self._v_sigmas = [float(s) for s in vb.scheduler.sigmas.tolist()] + [0.0]
        self._v_ts = vb.scheduler.timesteps
        ab.scheduler.set_timesteps(self._action_steps)
        self._a_sigmas = [float(s) for s in ab.scheduler.sigmas.tolist()] + [0.0]
        self._a_ts = ab.scheduler.timesteps

        self._cache = ARLinearStateCache(num_layers=vb.num_layers, window=self._window)
        self._gen = torch.Generator(device=self._device).manual_seed(self._seed)
        self._step_c = 0

    # ------------------------------------------------------------- per-step core
    @torch.no_grad()
    def _step_with_obs_latent(self, obs_latent, context, seq_lens, proprio) -> torch.Tensor:
        """One AR step from a ready obs *latent* — the loop body of ``ar_rollout``.

        Bypasses VAE / text encoding so the equivalence gate can drive it with the
        same mini backbones used to test ``ar_rollout``. Returns the predicted
        action chunk ``(B, action_tokens_per_chunk, action_dim)``.
        """
        arch = self.architecture
        obs = obs_latent.to(device=self._device, dtype=self._dtype)
        c = self._step_c

        self._cache.clear_pred()
        step_ctx, step_mask = arch._rollout_step_context(context, seq_lens, proprio)
        # F4: per-step proprio AdaLN deltas (in-distribution with per-chunk training).
        v_pe, a_pe = arch._rollout_proprio_deltas(proprio)

        # 1. ingest realized obs as confirmed clean video chunk at frame 2c.
        arch._ingest_clean_video(self._driver, self._cache, obs, frame_id=2 * c,
                                 context=step_ctx, context_mask=step_mask, v_proprio=v_pe)
        if getattr(arch, "_ar_bootstrap_clean_prefix", False):
            # F3 bootstrap: emit THIS chunk's action (frame 2c+1) from the real obs —
            # training-consistent + offset-free (no look-ahead video prediction).
            pred_action = arch._denoise_action_chunk(
                self._driver, self._cache, frame_id=2 * c + 1, batch=obs.shape[0],
                action_tokens=self._action_tokens_per_chunk,
                a_sigmas=self._a_sigmas, a_ts=self._a_ts,
                context=step_ctx, context_mask=step_mask, gen=self._gen, a_proprio=a_pe,
            )
        else:
            # Steady-state (default): predict next video chunk (frame 2c+2)...
            pred_video = arch._denoise_video_chunk(
                self._driver, self._cache, frame_id=2 * (c + 1), like=obs,
                v_sigmas=self._v_sigmas, v_ts=self._v_ts,
                context=step_ctx, context_mask=step_mask, gen=self._gen, v_proprio=v_pe,
            )
            # ...then the action chunk (frame 2c+3) conditioned on it.
            pred_action = arch._denoise_action_chunk(
                self._driver, self._cache, frame_id=2 * (c + 1) + 1, batch=obs.shape[0],
                action_tokens=self._action_tokens_per_chunk,
                a_sigmas=self._a_sigmas, a_ts=self._a_ts,
                context=step_ctx, context_mask=step_mask, gen=self._gen, a_proprio=a_pe,
            )
            del pred_video  # kept alive in the cache; free local
        self._step_c += 1
        return pred_action

    # ----------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        prompt = conditions.get("prompt", "") or ""
        obs_latent = self._build_obs_chunk(conditions)
        context, seq_lens = self._encode_prompt(prompt)
        proprio = self._prep_proprio(conditions.get("proprio_state"))

        pred_action = self._step_with_obs_latent(obs_latent, context, seq_lens, proprio)

        actions = pred_action.squeeze(0).float().cpu().numpy()
        normalizer = getattr(self.architecture, "action_normalizer", None)
        if normalizer is not None:
            actions = normalizer.unnormalize(actions)
        return {"actions": actions, "video": None}

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
        """Normalize raw deploy proprio and return a ``(B, D)`` tensor (or None)."""
        if not self.architecture.uses_proprioception:
            return None
        if proprio_state is None:
            return None
        if self._proprio_mode == "single":
            if self._latched_proprio is None:
                self._latched_proprio = np.asarray(proprio_state, dtype=np.float32).reshape(-1)
            proprio_state = self._latched_proprio
        norm = self.architecture.normalize_deploy_proprio(
            np.asarray(proprio_state, dtype=np.float32).reshape(-1)
        )
        return torch.from_numpy(np.asarray(norm, dtype=np.float32)).to(
            device=self._device, dtype=self._dtype
        ).unsqueeze(0)

    def _recent_frames(self, conditions: dict) -> list:
        """Per-sim-step head frames, oldest→newest. Prefers ``obs_history`` (WAMPolicy
        appends one every step) over the single ``first_frame_image`` so the obs clip
        can be rebuilt at the training pixel cadence."""
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
                "ARInferenceEngine.generate requires conditions['obs_history'] or ['first_frame_image']."
            )
        return frames

    def _build_obs_clip(self, conditions: dict) -> list:
        """Reconstruct the training-cadence video clip: take the last ``num_frames``
        per-step frames and subsample by ``video_stride`` — mirroring the dataset's
        ``range(0, num_frames, video_stride)`` so the encoded latents match training's
        motion distribution. Left-pads at episode start (bootstrap; F3/R4 caveat)."""
        frames = self._recent_frames(conditions)
        if self._obs_chunk_mode == "repeat_frame":  # debug only: zero-motion clip
            return [frames[-1]] * self._video_num_frames
        recent = frames[-self._raw_num_frames:]
        if len(recent) < self._raw_num_frames:  # episode-start pad to a full window
            recent = [recent[0]] * (self._raw_num_frames - len(recent)) + recent
        clip = recent[:: self._video_stride]  # newest frame included (num_frames-1 % stride == 0)
        if len(clip) > self._video_num_frames:
            clip = clip[-self._video_num_frames:]
        elif len(clip) < self._video_num_frames:
            clip = [clip[0]] * (self._video_num_frames - len(clip)) + clip
        return clip

    def _build_obs_chunk(self, conditions: dict) -> torch.Tensor:
        """Encode the recent observation into a ``(B, C, fcs, H_lat, W_lat)`` latent chunk."""
        vb = self.architecture.video_backbone
        if vb._pipe.vae is None:
            raise RuntimeError("ARInferenceEngine obs encoding requires a loaded Wan VAE.")

        from sana_wam.model.video_backbone.sana.adapter import _pil_video_to_tensor

        clip = self._build_obs_clip(conditions)
        video = _pil_video_to_tensor([clip]).to(device=self._device, dtype=self._dtype)  # (1,3,T,H,W)
        latents = vb._pipe.vae.encode([video[0]], device=self._device, tiled=True)
        latents = latents.to(device=self._device, dtype=self._dtype)  # (1,C,T_lat,H_lat,W_lat)
        if latents.shape[2] < self._fcs:
            raise RuntimeError(
                f"VAE produced {latents.shape[2]} latent frames < ar_frame_chunk_size={self._fcs}; "
                f"increase inference.video_num_frames (got {self._video_num_frames})."
            )
        return self._select_obs_band(latents)

    def _select_obs_band(self, latents: torch.Tensor) -> torch.Tensor:
        """Pick the obs latent band. ``auto`` uses the leading band (training chunk 0,
        incl. the causal first-frame latent) at episode start (step 0) and the trailing
        band (latest chunk) thereafter — matching how training assigns rope[0,1] to the
        first chunk vs the continuation chunk. F3 bootstrap eval knob."""
        band = self._obs_latent_band
        if band == "auto":
            band = "leading" if self._step_c == 0 else "trailing"
        if band == "leading":  # training chunk 0 band (incl. causal first-frame)
            return latents[:, :, : self._fcs].contiguous()
        return latents[:, :, -self._fcs:].contiguous()  # trailing = latest chunk of a training clip

    # -------------------------------------------------------------------- reset
    def reset(self) -> None:
        """Clear per-episode state so the next ``generate`` starts a fresh rollout."""
        self._cache.reset()
        self._step_c = 0
        self._gen = torch.Generator(device=self._device).manual_seed(self._seed)
        self._latched_proprio = None
