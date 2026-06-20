"""Inference engine for the GDN cross-attention TI2V world-action model.

Unlike the block-AR :class:`ARInferenceEngine` (stateful KV cache, one chunk per
``generate``), the cross-attn architecture is **not** autoregressive: each
``generate`` is a one-shot flow-matching denoise of the whole clip, conditioned
on the real first frame (TI2V clean-prefix at t=0). The GDN backbone's own
chunk-causal masking provides temporal structure inside one forward; there is no
persistent cache to carry across calls, so this engine is stateless (``reset`` is
a no-op).

Per ``generate``:
  1. encode prompt → context; encode the current observation frame → first-frame
     latent (clean prefix); normalize proprio.
  2. initialise noisy video + action at sigma=1 (pure noise), keep latent frame 0
     clean (the observation).
  3. run the action + video flow-matching schedules jointly: at each step call
     ``architecture.forward`` to get (v_pred, a_pred), step both schedulers.
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
        self._video_num_frames = (self._raw_num_frames - 1) // self._video_stride + 1

        self._prompt_ctx_cache: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._gen = torch.Generator(device=self._device).manual_seed(self._seed)

        logger.info(
            "CrossAttnInferenceEngine ready: video_steps=%d action_steps=%d action_dim=%d vnf=%d",
            self._video_steps, self._action_steps, self._action_dim, self._video_num_frames,
        )

    # ----------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        arch = self.architecture
        vb, ab = arch.video_backbone, arch.action_backbone

        prompt = conditions.get("prompt", "") or ""
        context, seq_lens = self._encode_prompt(prompt)
        proprio = self._prep_proprio(conditions.get("proprio_state"))
        first_frame_latent = self._encode_first_frame(conditions)  # (1, C, 1, Hl, Wl)

        B = 1
        C, Hl, Wl = first_frame_latent.shape[1], first_frame_latent.shape[3], first_frame_latent.shape[4]
        T = self._video_num_frames_latent()
        atok = self._action_tokens or (self._raw_num_frames - 1)

        # first_frame_latent must be exactly one latent frame (the observation),
        # and the generated clip must have room for it. Without these guards a
        # mismatch would silently broadcast/clip ``video[:, :, :1] = ...`` below
        # and corrupt the TI2V conditioning.
        if first_frame_latent.shape[2] != 1:
            raise ValueError(
                f"first_frame_latent must have 1 latent frame, got {first_frame_latent.shape[2]} "
                f"(shape {tuple(first_frame_latent.shape)})."
            )
        if T < 1:
            raise ValueError(f"generated latent length T={T} must be >= 1 (video_num_frames too small).")

        # Flow-matching schedules.
        vb.scheduler.set_timesteps(self._video_steps)
        ab.scheduler.set_timesteps(self._action_steps)
        v_sigmas = [float(s) for s in vb.scheduler.sigmas.tolist()] + [0.0]
        a_sigmas = [float(s) for s in ab.scheduler.sigmas.tolist()] + [0.0]
        v_ts, a_ts = vb.scheduler.timesteps, ab.scheduler.timesteps
        n_steps = min(len(v_ts), len(a_ts))

        # Init: pure noise (sigma≈1); keep latent frame 0 = the clean observation.
        video = torch.randn(B, C, T, Hl, Wl, generator=self._gen, device=self._device, dtype=self._dtype)
        actions = torch.randn(B, atok, self._action_dim, generator=self._gen, device=self._device, dtype=self._dtype)
        video[:, :, :1] = first_frame_latent  # TI2V clean prefix

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
            video[:, :, :1] = first_frame_latent  # re-pin the clean observation

        actions_np = actions.squeeze(0).float().cpu().numpy()
        normalizer = getattr(arch, "action_normalizer", None)
        if normalizer is not None:
            actions_np = normalizer.unnormalize(actions_np)
        return {"actions": actions_np, "video": None}

    def reset(self) -> None:
        """Stateless engine — only reseed the RNG for reproducibility."""
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

    def _video_num_frames_latent(self) -> int:
        """Latent temporal length T for the generated clip (causal Wan VAE)."""
        return 1 + (self._video_num_frames - 1) // 4
