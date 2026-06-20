"""Cross-attention DualSystem architecture for the GDN video backbone.

This is the non-MoT coupling path: the video backbone (here GDN /
``ChunkCausalGDNTriton``) runs its blocks natively, and the action stream reads
the per-layer video features through cross-attention bridges
(``ActionDiT(variant="joint_cross_attn")``). It is the architecture that pairs
with ``video_backbone.attn_kernel == "gdn"`` — GDN is a frame-wise recurrence
that cannot do the concatenated joint self-attention ``SanaMoTJointDriver``
needs, so the two modalities are coupled by cross-attention instead.

Layout per forward:

    video:  prepare → [run_block × N] (capture state.x at bridge_layers) → finalize
    action: ActionDiT.forward_with_bridge_tuple(actions, bridges, t, context)

Loss is a non-AR dual flow-matching MSE (one sampled timestep per sample for the
whole clip, no duplicated [noisy|clean] sequence — that block-causal trick is
the AR architecture's job; GDN's chunk-causal masking handles temporal causality
inside the backbone). TI2V (first-frame conditioning) is orthogonal and flows
through ``use_first_frame_cond`` on the video backbone exactly as for linear_relu.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
from sana_wam.model.base import BaseWAMArchitecture
from sana_wam.utils import resolve_bridge_layers


class DualSystemCrossAttnArchitecture(BaseWAMArchitecture):
    """DualSystem coupled by cross-attention bridges (GDN video backbone)."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._cross_cfg: dict = {}
        if cfg is None:
            return
        cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
        self._cross_cfg = cfg
        if self.video_backbone is not None:
            self.build_action_backbone()

    def build_action_backbone(self) -> ActionDiT:
        """Construct the cross-attn ActionDiT from the current backbones.

        Re-callable; tests that attach a mini video backbone after ``cfg=None``
        construction call this to wire the action stream afterwards (mirrors
        ``DualSystemSelfAttnArchitecture.build_mot_driver``).
        """
        if self.video_backbone is None:
            raise RuntimeError("build_action_backbone requires a video_backbone.")
        cfg = dict(self._cross_cfg)
        cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
        cfg.setdefault("video_dim", self.video_backbone.dim)

        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        text_dim = self._resolve_text_dim(cfg)
        self._init_proprio_context(cfg, text_dim=text_dim)

        action_dim_hidden = int(cfg.get("dim", 1024))
        num_heads = int(cfg.get("num_heads", 16))
        attn_head_dim = int(cfg.get("attn_head_dim", action_dim_hidden // num_heads))

        # Cross-attn ActionDiT: one block per bridge layer, each cross-attending
        # to its mapped video DiT layer's features. No MoT driver — coupling is
        # the bridge cross-attention itself.
        self.action_backbone = ActionDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=int(cfg.get("ffn_dim", 4 * action_dim_hidden)),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_cross_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
        )
        self._bridge_layers = tuple(bl)
        return self.action_backbone

    # ------------------------------------------------------------------
    # forward: video run_block (capture bridges) → action cross-attn
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio_state: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError("video_backbone is None — build the architecture with a video_backbone config.")

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio_state)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        # Run the video DiT natively, capturing per-layer hidden states (B, T_v, C)
        # at the bridge layers. Each captured feature is a cross-attn KV source.
        bridge_set = set(self._bridge_layers)
        bridges: dict[int, Tensor] = {}
        for block_id in range(vb.num_layers):
            vstate = vb.run_block(block_id, vstate)
            if block_id in bridge_set:
                bridges[block_id] = vstate.x
        video_pred = vb.finalize(vstate)

        if noisy_actions is None or ab is None:
            return video_pred, None

        bridge_tuple = ab.bridge_tuple_from_dict(bridges)
        action_pred = ab.forward_with_bridge_tuple(
            noisy_actions,
            bridge_tuple,
            action_timestep,
            context=action_context,
            context_mask=action_context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return video_pred, action_pred

    # ------------------------------------------------------------------
    # compute_loss: non-AR dual flow-matching MSE
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
        """One sampled timestep per sample; flow-matching MSE on video + action.

        TI2V: when the video backbone emits ``first_frame_latents`` (clean-prefix
        conditioning), frame 0 is the real observation and is excluded from the
        video loss (``video_is_pad``-style skip handled via the per-frame mask).
        """
        import torch.nn.functional as F

        vb = self.video_backbone
        ab = self.action_backbone
        device, dtype = self.device, self.dtype

        clean_video = inputs["input_latents"].to(device=device, dtype=dtype)
        B, _, T, _, _ = clean_video.shape

        # ---- video: one timestep per sample → sigma broadcast over the clip ----
        num_ts_v = len(vb.scheduler.timesteps)
        v_ids = torch.randint(0, num_ts_v, (B,))  # CPU ids index CPU scheduler tensors
        v_sigma = vb.scheduler.sigmas[v_ids].to(device=device, dtype=dtype)  # (B,)
        v_ts_val = vb.scheduler.timesteps[v_ids].to(device=device, dtype=dtype)  # (B,)
        v_noise = torch.randn_like(clean_video)
        s = v_sigma.view(B, 1, 1, 1, 1)
        noisy_video = (1 - s) * clean_video + s * v_noise
        v_target = v_noise - clean_video

        # ---- action: one timestep per sample ----
        if actions is None or lambda_action <= 0:
            raise RuntimeError("DualSystemCrossAttnArchitecture.compute_loss requires actions.")
        actions = actions.to(device=device, dtype=dtype)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        Ta = actions.shape[1]
        a_scheduler = ab.scheduler
        num_ts_a = len(a_scheduler.timesteps)
        a_ids = torch.randint(0, num_ts_a, (B,))
        a_sigma = a_scheduler.sigmas[a_ids].to(device=device, dtype=dtype)  # (B,)
        a_ts_val = a_scheduler.timesteps[a_ids].to(device=device, dtype=dtype)  # (B,)
        a_noise = torch.randn_like(actions)
        noisy_actions = (1 - a_sigma.view(B, 1, 1)) * actions + a_sigma.view(B, 1, 1) * a_noise
        a_target = a_noise - actions

        fwd_inputs = {k: v for k, v in inputs.items() if k in ("context", "context_mask", "seq_lens")}
        proprio_state = inputs.get("proprio_state")

        v_pred, a_pred = self.forward(
            noisy_actions,
            a_ts_val,
            proprio_state=proprio_state,
            latents=noisy_video,
            timestep=v_ts_val,
            **fwd_inputs,
        )
        return self._dual_mse(
            v_pred, v_target, a_pred, a_target,
            vb=vb, v_ids=v_ids, T=T, B=B,
            lambda_video=lambda_video, lambda_action=lambda_action,
            action_is_pad=inputs.get("action_is_pad"),
            video_is_pad=inputs.get("video_is_pad"),
            device=device,
        )

    @staticmethod
    def _dual_mse(
        v_pred, v_target, a_pred, a_target, *, vb, v_ids, T, B,
        lambda_video, lambda_action, action_is_pad, video_is_pad, device,
    ) -> dict:
        import torch.nn.functional as F

        # ---- per-frame weighted video MSE ----
        per_v = F.mse_loss(v_pred.float(), v_target.float(), reduction="none").mean(dim=(1, 3, 4))  # (B, T)
        w = vb.scheduler.linear_timesteps_weights
        if w is not None:
            wv = w[v_ids].to(dtype=torch.float32, device=device).view(B, 1)  # (B,1) broadcast over T
        else:
            wv = torch.ones(B, 1, dtype=torch.float32, device=device)
        if video_is_pad is not None and video_is_pad.shape == per_v.shape:
            keep = (~video_is_pad.bool()).to(dtype=torch.float32, device=device)
            loss_video = (per_v * wv * keep).sum() / keep.sum().clamp_min(1.0)
        else:
            loss_video = (per_v * wv).mean()

        # ---- per-token action MSE (pad-masked) ----
        per_a = F.mse_loss(a_pred.float(), a_target.float(), reduction="none").mean(dim=2)  # (B, Ta)
        if action_is_pad is not None and action_is_pad.shape == per_a.shape:
            keep_a = (~action_is_pad.bool()).to(dtype=torch.float32, device=device)
            loss_action = (per_a * keep_a).sum() / keep_a.sum().clamp_min(1.0)
        else:
            loss_action = per_a.mean()

        loss = lambda_video * loss_video + lambda_action * loss_action
        return {
            "loss": loss,
            "loss_video": (lambda_video * loss_video).detach(),
            "loss_action": (lambda_action * loss_action).detach(),
        }


__all__ = ["DualSystemCrossAttnArchitecture"]
