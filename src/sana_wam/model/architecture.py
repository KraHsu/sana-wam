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
import math
from hashlib import sha256
from typing import Optional, Tuple

import torch
from torch import Tensor

from sana_wam.model.action_non_regression import (
    adapter_only_loss_surrogate,
    one_sided_relative_excess_loss,
    unweighted_pad_masked_action_mse,
)
from sana_wam.model.action_video_memory_adapter import ActionVideoMemoryAdapter
from sana_wam.model.action_facing_cache_consistency import (
    action_facing_cache_consistency_loss,
    validate_afcc_reference,
    validate_afcc_weight,
)
from sana_wam.model.local_expansion import (
    current_video_chunk_mask,
    fixed_50_step_shifted_sigmas,
    local_reverse_euler_expansion_loss,
    nearest_reverse_delta_sigma,
    phase6_expansion_spec,
    quantized_one_sided_chord,
    rademacher_orthogonal_direction,
)
from sana_wam.model.phase6_reference_trace import trace_sha256
from sana_wam.model.joint_self_attn import DualSystemSelfAttnArchitecture
from sana_wam.model.ar.sana_ar_linear_attn import build_ar_seq_meta

logger = logging.getLogger(__name__)


def _phase6_child_seed(parent_seed: int, child_domain: str) -> int:
    """Derive a stable substream seed without depending on RNG draw order."""
    if type(parent_seed) is not int or not 0 <= parent_seed < 2**64:
        raise ValueError("Phase-6 parent seed must be an unsigned 64-bit integer")
    if not isinstance(child_domain, str) or not child_domain:
        raise ValueError("Phase-6 child RNG domain must be a non-empty string")
    material = bytearray()
    for field in (
        b"sana-phase6-compute-loss-child-seed-v1",
        str(parent_seed).encode("ascii"),
        child_domain.encode("utf-8"),
    ):
        material.extend(len(field).to_bytes(8, "big"))
        material.extend(field)
    return int.from_bytes(sha256(material).digest()[:8], "big")


def _phase6_generator(
    parent_seed: int, child_domain: str, *, device: torch.device | str
) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(
        _phase6_child_seed(parent_seed, child_domain)
    )


def _phase6_partition_prefix_count(
    value,
    *,
    name: str,
    batch_size: int,
    upper_bound: int,
) -> tuple[Tensor | None, int | None]:
    """Keep tensor controls bit-exact while moving plain counts to metadata."""
    if value is None:
        return None, None
    if type(value) is int:
        count = value
        tensor_value = None
    elif isinstance(value, Tensor):
        if (
            value.layout != torch.strided
            or value.dtype != torch.long
            or tuple(value.shape) != (batch_size,)
            or value.requires_grad
        ):
            raise TypeError(
                f"{name} tensor must be detached, strided, torch.long, and have "
                f"shape ({batch_size},)"
            )
        count = int(value.detach().cpu().item())
        tensor_value = value
    else:
        raise TypeError(
            f"{name} must be None, a plain int, or a torch.long batch tensor"
        )
    if not 0 <= count <= upper_bound:
        raise ValueError(f"{name} must be in [0, {upper_bound}], got {count}")
    scalar_value = count if tensor_value is None else None
    return tensor_value, scalar_value


def _init_fp32_euler_master(
    shape,
    *,
    generator: Optional[torch.Generator],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Tensor:
    """Preserve the model-dtype noise draw while accumulating in FP32."""
    noise = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=model_dtype,
    )
    return noise.float()


def _fp32_euler_update(master: Tensor, velocity: Tensor, delta_sigma: float) -> Tensor:
    """Apply one Euler update without rounding the persistent state to model dtype."""
    if master.dtype != torch.float32:
        raise TypeError(f"Euler master state must be float32, got {master.dtype}")
    return master + velocity.float() * float(delta_sigma)


class DualSystemARArchitecture(DualSystemSelfAttnArchitecture):
    """Block-autoregressive DualSystem variant (SANA linear-attention only)."""

    @staticmethod
    def _coerce_ar_padding_mask(
        value,
        *,
        name: str,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        default_all_valid: bool,
    ) -> Tensor | None:
        """Validate the suffix-padding contract and move the mask to the model."""
        if value is None:
            if not default_all_valid:
                return None
            return torch.zeros(
                batch_size, num_tokens, dtype=torch.bool, device=device
            )
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor or None")
        if value.dtype != torch.bool or value.layout != torch.strided:
            raise TypeError(f"{name} must be a strided boolean tensor")
        if tuple(value.shape) != (batch_size, num_tokens):
            raise ValueError(
                f"{name} must exactly match ({batch_size}, {num_tokens}), "
                f"got {tuple(value.shape)}"
            )
        mask = value.detach().to(device=device)
        if num_tokens > 1 and bool((mask[:, :-1] & ~mask[:, 1:]).any()):
            raise ValueError(f"{name} must be a monotonic padded suffix")
        return mask

    @staticmethod
    def _validate_local_expansion_contract(
        *,
        weight: float,
        legacy_weights: tuple[float, float, float],
        chunkwise_temporal_ops: bool,
    ) -> None:
        if not math.isfinite(weight) or weight not in {0.0, 1.0}:
            raise ValueError(
                "Phase-6 video_local_expansion_weight must be exactly 0 or 1"
            )
        if weight == 0.0:
            return
        if any(legacy_weight != 0.0 for legacy_weight in legacy_weights):
            raise ValueError(
                "Phase-6 local expansion requires all legacy video trajectory "
                "weights to be exactly zero"
            )
        if not chunkwise_temporal_ops:
            raise ValueError(
                "Phase-6 local expansion requires ar_chunkwise_temporal_ops=true"
            )

    @staticmethod
    def _validate_action_non_regression_weight(value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "Phase-6 action_non_regression_weight must be exactly 0 or 1"
            )
        weight = float(value)
        if not math.isfinite(weight) or weight not in {0.0, 1.0}:
            raise ValueError(
                "Phase-6 action_non_regression_weight must be exactly 0 or 1"
            )
        return weight

    @staticmethod
    def _validate_phase1_action_reference_error(
        value, *, device: torch.device
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError("phase1_action_reference_error must be a tensor")
        if value.shape != () or value.dtype != torch.float32:
            raise ValueError(
                "phase1_action_reference_error must be one FP32 scalar"
            )
        if value.device != device:
            raise ValueError(
                "phase1_action_reference_error must be on the model device"
            )
        if value.requires_grad:
            raise ValueError("phase1_action_reference_error must be detached")
        if not bool(torch.isfinite(value)) or bool(value < 0):
            raise ValueError(
                "phase1_action_reference_error must be finite and non-negative"
            )
        return value.detach()

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
        self._ar_chunkwise_temporal_ops = bool(
            _get("ar_chunkwise_temporal_ops", False)
        )
        if self._ar_chunkwise_temporal_ops:
            self._configure_ar_chunkwise_temporal_ops()
        self._video_on_path_loss_weight = float(
            _get("video_on_path_loss_weight", 1.0)
        )
        self._video_trajectory_endpoint_weight = float(
            _get("video_trajectory_endpoint_weight", 0.0)
        )
        self._video_trajectory_velocity_weight = float(
            _get("video_trajectory_velocity_weight", 0.0)
        )
        self._video_trajectory_consistency_weight = float(
            _get("video_trajectory_consistency_weight", 0.0)
        )
        self._video_trajectory_steps = int(_get("video_trajectory_steps", 2))
        self._video_trajectory_schedule_mode = str(
            _get("video_trajectory_schedule_mode", "linear_random")
        )
        if self._video_trajectory_schedule_mode not in {
            "linear_random",
            "inference",
        }:
            raise ValueError(
                "video_trajectory_schedule_mode must be 'linear_random' or "
                "'inference'"
            )
        self._video_trajectory_supervision_mode = str(
            _get("video_trajectory_supervision_mode", "final")
        )
        if self._video_trajectory_supervision_mode not in {"final", "random"}:
            raise ValueError(
                "video_trajectory_supervision_mode must be 'final' or 'random'"
            )
        self._video_trajectory_min_sigma = float(
            _get("video_trajectory_min_sigma", 0.2)
        )
        self._video_trajectory_max_sigma = float(
            _get("video_trajectory_max_sigma", 0.9)
        )
        trajectory_enabled = (
            self._video_trajectory_endpoint_weight > 0
            or self._video_trajectory_velocity_weight > 0
            or self._video_trajectory_consistency_weight > 0
        )
        if min(
            self._video_on_path_loss_weight,
            self._video_trajectory_endpoint_weight,
            self._video_trajectory_velocity_weight,
            self._video_trajectory_consistency_weight,
        ) < 0:
            raise ValueError("video loss weights must be non-negative")
        if self._video_trajectory_steps <= 0:
            raise ValueError("video_trajectory_steps must be positive")
        if (
            self._video_trajectory_schedule_mode == "inference"
            and self._video_trajectory_steps < 2
        ):
            raise ValueError(
                "inference video trajectory supervision requires at least 2 steps"
            )
        if not (
            0 < self._video_trajectory_min_sigma
            <= self._video_trajectory_max_sigma
            < 1
        ):
            raise ValueError(
                "video trajectory sigma range must satisfy 0 < min <= max < 1"
            )
        if trajectory_enabled and not self._ar_chunkwise_temporal_ops:
            raise ValueError(
                "video trajectory supervision requires ar_chunkwise_temporal_ops=true"
            )
        self._video_local_expansion_weight = float(
            _get("video_local_expansion_weight", 0.0)
        )
        self._validate_local_expansion_contract(
            weight=self._video_local_expansion_weight,
            legacy_weights=(
                self._video_trajectory_endpoint_weight,
                self._video_trajectory_velocity_weight,
                self._video_trajectory_consistency_weight,
            ),
            chunkwise_temporal_ops=self._ar_chunkwise_temporal_ops,
        )
        self._action_non_regression_weight = (
            self._validate_action_non_regression_weight(
                _get("action_non_regression_weight", 0.0)
            )
        )
        self._action_facing_cache_consistency_weight = validate_afcc_weight(
            _get("action_facing_cache_consistency_weight", 0.0)
        )
        if (
            self._action_non_regression_weight > 0
            and self._action_facing_cache_consistency_weight > 0
        ):
            raise ValueError(
                "action non-regression and AFCC treatments are mutually exclusive"
            )

        # F4: per-chunk proprio. When True, in addition to the FastWAM clip-level
        # proprio context token, each chunk is conditioned on the robot state at
        # that chunk's time via an additive per-frame/per-token AdaLN delta
        # (zero-init encoders => identical to the current checkpoint at init).
        self._proprio_per_chunk = bool(_get("proprio_per_chunk", False))
        # F3: supply chunk-0's first latent frame clean (real init obs) in training
        # and emit a chunk-0 action in the rollout (LingBot bootstrap).
        self._ar_bootstrap_clean_prefix = bool(_get("ar_bootstrap_clean_prefix", False))
        # Training-only, per-sample dropout for the action branch. The video
        # branch keeps proprio so its imagined future remains state-conditioned.
        self._proprio_action_dropout_prob = float(
            _get("proprio_action_dropout_prob", 0.0)
        )
        # AR historically used an unweighted action loss. Keep that default while
        # allowing the validated low-noise recipe to opt in explicitly.
        if self.action_backbone is not None:
            self.action_backbone.loss_weighting = str(
                _get("action_loss_weighting", "none")
            )
        self.proprio_video_embed = None
        self.proprio_action_embed = None
        self._maybe_init_per_chunk_proprio()
        self.action_video_memory_adapter = None
        adapter_cfg = _get("action_video_memory_adapter", None)
        if adapter_cfg is None:
            self._action_video_memory_adapter_cfg = None
        else:
            adapter_cfg = (
                dict(adapter_cfg.items())
                if hasattr(adapter_cfg, "items")
                else dict(adapter_cfg)
            )
            allowed = {
                "enabled",
                "rank",
                "z_scale_min",
                "z_scale_max",
                "init_seed",
            }
            unknown = sorted(set(adapter_cfg) - allowed)
            if unknown:
                raise ValueError(
                    f"unknown action_video_memory_adapter keys: {unknown}"
                )
            enabled = adapter_cfg.get("enabled", False)
            if not isinstance(enabled, bool):
                raise ValueError(
                    "action_video_memory_adapter.enabled must be a boolean"
                )
            adapter_cfg["enabled"] = enabled
            self._action_video_memory_adapter_cfg = adapter_cfg
        self._maybe_init_action_video_memory_adapter()
        if (
            self._action_non_regression_weight > 0
            and self.video_backbone is not None
            and self.action_backbone is not None
            and self.action_video_memory_adapter is None
        ):
            raise ValueError(
                "action_non_regression_weight=1 requires an enabled action "
                "video-memory adapter"
            )
        if (
            self._action_facing_cache_consistency_weight > 0
            and self.action_video_memory_adapter is not None
        ):
            raise ValueError("AFCC requires the action video-memory adapter disabled")
        if self._mot_driver is not None:
            self._mot_driver.action_video_memory_adapter = (
                self.action_video_memory_adapter
            )

    def _maybe_init_per_chunk_proprio(self) -> None:
        """Create the zero-init per-chunk proprio encoders once both backbones exist.

        Called at __init__ (real cfg path: backbones already built → encoders land
        in the optimizer) and from build_mot_driver (test path: backbones set later).
        Idempotent. Zero-init so the model output is identical to the pre-F4
        checkpoint at step 0 (warm-startable retrain)."""
        if not (
            getattr(self, "_proprio_per_chunk", False)
            and bool(getattr(self, "_use_proprioception_context", False))
        ):
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

    def _maybe_init_action_video_memory_adapter(self) -> None:
        """Create the optional identity-init action video-memory read adapter."""
        cfg = getattr(self, "_action_video_memory_adapter_cfg", None)
        if not cfg or not cfg.get("enabled", False):
            return
        if self.action_video_memory_adapter is not None:
            return
        if self.video_backbone is None or self.action_backbone is None:
            return
        rank = int(cfg.get("rank", 8))
        z_scale_min = float(cfg.get("z_scale_min", 0.5))
        z_scale_max = float(cfg.get("z_scale_max", 2.0))
        if rank != 8:
            raise ValueError(
                "Phase-6 action_video_memory_adapter.rank is frozen at 8"
            )
        if z_scale_min != 0.5 or z_scale_max != 2.0:
            raise ValueError(
                "Phase-6 action-video normalizer bounds are frozen at [0.5, 2.0]"
            )
        self.action_video_memory_adapter = ActionVideoMemoryAdapter(
            num_layers=int(self.video_backbone.num_layers),
            num_heads=int(self.video_backbone.num_heads),
            head_dim=int(self.video_backbone.head_dim),
            rank=rank,
            z_scale_min=z_scale_min,
            z_scale_max=z_scale_max,
            init_seed=int(cfg.get("init_seed", 0)),
        )

    def set_dtype_device(self, dtype, device) -> None:
        super().set_dtype_device(dtype, device)
        for name in ("proprio_video_embed", "proprio_action_embed"):
            m = getattr(self, name, None)
            if m is not None:
                m.to(dtype=dtype, device=device)
        adapter = getattr(self, "action_video_memory_adapter", None)
        if adapter is not None:
            # Adapter parameters remain FP32; its deltas are cast to cache dtype.
            adapter.to(device=device)

    def build_mot_driver(self):
        """Construct the AR MoT driver (SANA linear-attn only)."""
        if self.video_backbone is None or self.action_backbone is None:
            raise RuntimeError(
                "DualSystemARArchitecture.build_mot_driver requires both backbones."
            )
        kernel = getattr(self.video_backbone, "attn_kernel", "softmax")
        if kernel != "linear_relu":
            raise ValueError(
                f"DualSystemARArchitecture requires SANA linear_relu attention, got attn_kernel='{kernel}'."
            )
        if getattr(self, "_ar_chunkwise_temporal_ops", False):
            self._configure_ar_chunkwise_temporal_ops()
        from sana_wam.model.ar.sana_ar_mot_driver import SanaARMoTJointDriver

        # The AR kernel does not use the dense-mask rebalance knobs; drop them.
        kwargs = {
            k: v
            for k, v in self._mot_driver_kwargs.items()
            if k
            not in (
                "action_self_attn_weight",
                "action_self_attn_mode",
                "action_video_attn_mode",
            )
        }
        self._maybe_init_action_video_memory_adapter()
        self._mot_driver = SanaARMoTJointDriver(
            self.video_backbone,
            self.action_backbone,
            action_video_memory_adapter=getattr(
                self, "action_video_memory_adapter", None
            ),
            **kwargs,
        )
        self._maybe_init_per_chunk_proprio()  # test path: backbones set after __init__
        return self._mot_driver

    def _configure_ar_chunkwise_temporal_ops(self) -> None:
        """Make all non-cached SANA temporal operators chunk-local."""
        vb = self.video_backbone
        if vb is None:
            return
        configure = getattr(vb, "configure_ar_chunkwise_temporal_ops", None)
        if not callable(configure):
            raise TypeError(
                "ar_chunkwise_temporal_ops requires a video backbone that "
                "implements configure_ar_chunkwise_temporal_ops"
            )
        configure(self._ar_frame_chunk_size)

    def _video_timestep_dtype(self, model_dtype: torch.dtype) -> torch.dtype:
        """Resolve the video-only T0/T1 conditioning dtype from backbone config."""
        if getattr(
            self.video_backbone, "continuous_timestep_conditioning", False
        ):
            return torch.float32
        return model_dtype

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],  # noqa: ARG002 — AR uses per-token ar_action_token_timesteps
        *,
        proprio_state: Optional[Tensor] = None,
        phase6_proprio_context_prepared: bool = False,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        return_action_video_numerators: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]] | tuple[Tensor, Optional[Tensor], Tensor]:
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None or ab is None:
            raise RuntimeError(
                "DualSystemARArchitecture.forward requires both video and action backbones."
            )
        if noisy_actions is None:
            raise RuntimeError(
                "DualSystemARArchitecture.forward requires noisy_actions (joint AR path)."
            )
        if type(return_action_video_numerators) is not bool:
            raise TypeError("return_action_video_numerators must be boolean")

        if type(phase6_proprio_context_prepared) is not bool:
            raise TypeError("phase6_proprio_context_prepared must be boolean")
        pipeline_inputs = dict(pipeline_inputs)
        if phase6_proprio_context_prepared:
            if not bool(getattr(self, "_use_proprioception_context", False)):
                raise ValueError(
                    "prepared proprio context is invalid when proprio is disabled"
                )
            if proprio_state is None:
                raise ValueError(
                    "prepared proprio context still requires the bound proprio state"
                )
        else:
            pipeline_inputs = self._append_proprio_context_token(
                pipeline_inputs, proprio_state
            )
        # F4: per-chunk proprio (B, num_chunks, state_dim); popped before vb.prepare(**kwargs).
        proprio_per_chunk = pipeline_inputs.pop("proprio_per_chunk", None)

        # --- AR config + extras (pop all AR keys up front so none leak into
        #     the video backbone's prepare(**kwargs)) ---
        frame_chunk_size = int(pipeline_inputs.pop("ar_frame_chunk_size"))
        attn_window = int(pipeline_inputs.pop("ar_attn_window"))
        clean_actions = pipeline_inputs.pop("ar_clean_actions", None)
        a_noisy_ts = pipeline_inputs.pop("ar_action_token_timesteps")  # (B, Ta)
        a_clean_ts = pipeline_inputs.pop("ar_clean_action_token_timesteps", None)
        raw_video_is_pad = pipeline_inputs.pop("ar_video_is_pad", None)
        raw_action_is_pad = pipeline_inputs.pop("ar_action_is_pad", None)

        # --- video stream: [noisy ++ clean] along the latent frame axis ---
        noisy_latents = pipeline_inputs.pop("latents")
        clean_latents = pipeline_inputs.pop("ar_clean_latents")
        B, _, T, _, _ = noisy_latents.shape
        if T % frame_chunk_size != 0:
            raise ValueError(
                f"latent frames T={T} must be divisible by frame_chunk_size={frame_chunk_size}."
            )
        num_chunks = T // frame_chunk_size
        video_is_pad = self._coerce_ar_padding_mask(
            raw_video_is_pad,
            name="ar_video_is_pad",
            batch_size=B,
            num_tokens=T,
            device=noisy_latents.device,
            default_all_valid=False,
        )
        if video_is_pad is not None:
            frame_pad = video_is_pad[:, None, :, None, None]
            # Storage padding is not model input. Sanitize it before any
            # parameterized embedding so NaN padding cannot poison gradients.
            noisy_latents = noisy_latents.masked_fill(frame_pad, 0)
            clean_latents = clean_latents.masked_fill(frame_pad, 0)

        v_noisy_ts = pipeline_inputs.pop("ar_video_frame_timesteps")  # (B, T)
        v_clean_ts = pipeline_inputs.pop(
            "ar_clean_video_frame_timesteps", torch.zeros_like(v_noisy_ts)
        )
        dup_latents = torch.cat(
            [noisy_latents, clean_latents], dim=2
        )  # (B, C, 2T, H, W)
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
            proprio_per_chunk,
            self.proprio_video_embed,
            num_chunks,
            frame_chunk_size,
            dup_latents.device,
        )  # (B, 2T, D_video) or None
        duplicated_video_is_pad = (
            None
            if video_is_pad is None
            else torch.cat([video_is_pad, video_is_pad], dim=1)
        )

        vstate = vb.prepare(
            latents=dup_latents,
            timestep=timestep,
            frame_timesteps=frame_timesteps,
            rope_frame_index=rope_frame_index,
            frame_proprio_emb=frame_proprio_emb,
            frame_is_pad=duplicated_video_is_pad,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        h, w = int(vstate.h), int(vstate.w)
        video_tokens_per_chunk = frame_chunk_size * h * w

        # --- action stream: [noisy ++ clean] along the token axis ---
        if clean_actions is None:
            raise RuntimeError(
                "DualSystemARArchitecture.forward requires ar_clean_actions."
            )
        Ta = noisy_actions.shape[1]
        if Ta % num_chunks != 0:
            raise ValueError(
                f"action tokens Ta={Ta} must be divisible by num_chunks={num_chunks} "
                "(action_tokens_per_chunk must be integer)."
            )
        action_tokens_per_chunk = Ta // num_chunks
        action_is_pad = self._coerce_ar_padding_mask(
            raw_action_is_pad,
            name="ar_action_is_pad",
            batch_size=B,
            num_tokens=Ta,
            device=noisy_actions.device,
            default_all_valid=False,
        )
        if action_is_pad is not None:
            token_pad = action_is_pad[:, :, None]
            noisy_actions = noisy_actions.masked_fill(token_pad, 0)
            clean_actions = clean_actions.masked_fill(token_pad, 0)

        if a_clean_ts is None:
            a_clean_ts = torch.zeros_like(a_noisy_ts)
        dup_actions = torch.cat([noisy_actions, clean_actions], dim=1)  # (B, 2Ta, Ad)
        token_timesteps = torch.cat([a_noisy_ts, a_clean_ts], dim=1)  # (B, 2Ta)

        # frame_ids / rope positions for the duplicated action stream: chunk
        # index with modality parity (2c+1), identical for a token's noisy/clean
        # copies so they share a rotary phase.
        chunk_of_token = (
            torch.arange(Ta, device=dup_actions.device) // action_tokens_per_chunk
        )
        action_frame_ids_single = chunk_of_token * 2 + 1  # (Ta,)
        action_frame_ids = torch.cat(
            [action_frame_ids_single, action_frame_ids_single]
        )  # (2Ta,)

        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if (
            action_context is not None
            and action_context_mask is None
            and pipeline_inputs.get("seq_lens") is not None
        ):
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(
                action_context.shape[1], device=action_context.device
            )
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

        # Drop only the action branch's appended proprio value. Keep the token
        # attendable to avoid all-masked cross-attention rows for empty text.
        act_drop = None
        if (
            self.training
            and self._proprio_action_dropout_prob > 0.0
            and bool(getattr(self, "_use_proprioception_context", False))
            and action_context is not None
        ):
            act_drop = (
                torch.rand(action_context.shape[0], device=action_context.device)
                < self._proprio_action_dropout_prob
            )
            if bool(act_drop.any()):
                action_context = action_context.clone()
                action_context[act_drop, -1, :] = 0.0

        # F4 action: additive per-token proprio delta into the time embedding.
        token_proprio_emb = self._per_chunk_proprio_emb(
            proprio_per_chunk,
            self.proprio_action_embed,
            num_chunks,
            action_tokens_per_chunk,
            dup_actions.device,
        )  # (B, 2Ta, D_action) or None
        if act_drop is not None and token_proprio_emb is not None:
            token_proprio_emb = token_proprio_emb * (~act_drop).view(-1, 1, 1).to(
                token_proprio_emb.dtype
            )

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
        key_is_pad = None
        if video_is_pad is not None or action_is_pad is not None:
            if video_is_pad is None:
                video_is_pad = torch.zeros(
                    B, T, dtype=torch.bool, device=noisy_latents.device
                )
            if action_is_pad is None:
                action_is_pad = torch.zeros(
                    B, Ta, dtype=torch.bool, device=noisy_actions.device
                )
            video_token_is_pad = video_is_pad.repeat_interleave(h * w, dim=1)
            key_is_pad = torch.cat(
                [
                    video_token_is_pad,
                    video_token_is_pad,
                    action_is_pad,
                    action_is_pad,
                ],
                dim=1,
            )
        ar_meta = build_ar_seq_meta(
            num_chunks=num_chunks,
            video_tokens_per_chunk=video_tokens_per_chunk,
            action_tokens_per_chunk=action_tokens_per_chunk,
            window=attn_window,
            device=dup_actions.device,
            key_is_pad=key_is_pad,
        )

        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()
        action_video_numerators = driver.run_ar_joint_loop(
            vstate,
            astate,
            ar_meta=ar_meta,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            return_action_video_numerators=return_action_video_numerators,
        )

        # --- finalize + extract the NOISY copy only ---
        video_full = vb.finalize(vstate)  # (B, C, 2T, H, W)
        video_noise_pred = video_full[:, :, :T]
        action_full = ab.extract_prediction(astate)  # (B, 2Ta, Ad)
        action_noise_pred = action_full[:, :Ta]
        if return_action_video_numerators:
            if action_video_numerators is None:
                raise RuntimeError("AFCC numerator capture was requested but is absent")
            return video_noise_pred, action_noise_pred, action_video_numerators
        return video_noise_pred, action_noise_pred

    @staticmethod
    def _per_chunk_proprio_emb(
        proprio_per_chunk, encoder, num_chunks, tokens_per_chunk, device
    ):
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
        per_token = pp.repeat_interleave(
            tokens_per_chunk, dim=1
        )  # (B, num_chunks*tpc, D)
        dup = torch.cat([per_token, per_token], dim=1)  # noisy ++ clean
        return encoder(dup.to(encoder.weight.dtype))

    def _trajectory_video_losses(
        self,
        *,
        clean_video: Tensor,
        video_noise: Tensor,
        actions: Tensor,
        fwd_inputs: dict,
        proprio_state: Optional[Tensor],
        proprio_per_chunk: Optional[Tensor],
        frame_chunk_size: int,
        video_is_pad: Optional[Tensor] = None,
        action_is_pad: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        """Supervise model-generated denoising states with the expert endpoint."""
        import torch.nn.functional as F

        batch, _, frames, _, _ = clean_video.shape
        action_tokens = actions.shape[1]
        device, dtype = clean_video.device, clean_video.dtype
        if video_is_pad is None:
            video_pad_for_loss = torch.zeros(
                batch, frames, dtype=torch.bool, device=device
            )
        else:
            if video_is_pad.dtype != torch.bool or tuple(video_is_pad.shape) != (
                batch,
                frames,
            ):
                raise ValueError(
                    "video_is_pad must be boolean with shape [batch, frames]"
                )
            if video_is_pad.device != device:
                raise ValueError("video_is_pad must be on the model device")
            video_pad_for_loss = video_is_pad.detach()
        if action_is_pad is not None:
            if action_is_pad.dtype != torch.bool or tuple(action_is_pad.shape) != (
                batch,
                action_tokens,
            ):
                raise ValueError(
                    "action_is_pad must be boolean with shape [batch, action_tokens]"
                )
            if action_is_pad.device != device:
                raise ValueError("action_is_pad must be on the model device")
        video_timestep_dtype = self._video_timestep_dtype(dtype)
        video_timestep_scale = (
            float(self.video_backbone.scheduler.num_train_timesteps)
            if getattr(
                self.video_backbone,
                "continuous_timestep_conditioning",
                False,
            )
            else 1000.0
        )
        if self._video_trajectory_schedule_mode == "inference":
            unshifted = torch.linspace(
                1.0,
                0.0,
                self._video_trajectory_steps + 1,
                device=device,
                dtype=torch.float32,
            )
            shift = float(self.video_backbone.scheduler.flow_shift)
            schedule = shift * unshifted / (1.0 + (shift - 1.0) * unshifted)
        else:
            sigma_min = self._video_trajectory_min_sigma
            sigma_max = self._video_trajectory_max_sigma
            target_sigma = sigma_min + (sigma_max - sigma_min) * torch.rand(
                (), device=device, dtype=torch.float32
            )
            schedule = torch.linspace(
                1.0,
                float(target_sigma.item()),
                self._video_trajectory_steps + 1,
                device=device,
                dtype=torch.float32,
            )
        clean_video_timesteps = torch.zeros(
            batch, frames, device=device, dtype=video_timestep_dtype
        )
        action_timesteps = torch.zeros(
            batch, action_tokens, device=device, dtype=dtype
        )
        state = video_noise.detach().clone()
        if self._ar_bootstrap_clean_prefix:
            state[:, :, 0] = clean_video[:, :, 0]

        def forward_video(current: Tensor, sigma: Tensor) -> Tensor:
            video_timesteps = torch.full(
                (batch, frames),
                float(sigma.item()) * video_timestep_scale,
                device=device,
                dtype=video_timestep_dtype,
            )
            if self._ar_bootstrap_clean_prefix:
                video_timesteps[:, 0] = 0.0
            prediction, _ = self.forward(
                actions,
                None,
                proprio_state=proprio_state,
                proprio_per_chunk=proprio_per_chunk,
                latents=current,
                ar_clean_latents=clean_video,
                ar_clean_actions=actions,
                ar_video_frame_timesteps=video_timesteps,
                ar_clean_video_frame_timesteps=clean_video_timesteps,
                ar_action_token_timesteps=action_timesteps,
                ar_clean_action_token_timesteps=action_timesteps,
                ar_video_is_pad=video_is_pad,
                ar_action_is_pad=action_is_pad,
                ar_frame_chunk_size=frame_chunk_size,
                ar_attn_window=self._ar_attn_window,
                timestep=video_timesteps.mean(dim=1),
                **fwd_inputs,
            )
            return prediction

        generated_states = []
        with torch.no_grad():
            for index in range(self._video_trajectory_steps):
                sigma = schedule[index]
                next_sigma = schedule[index + 1]
                prediction = forward_video(state, sigma)
                adjacent_endpoint = (
                    state.float() - sigma * prediction.float()
                ).detach()
                delta = (next_sigma - sigma).to(dtype=dtype)
                state = state + prediction * delta
                if self._ar_bootstrap_clean_prefix:
                    state[:, :, 0] = clean_video[:, :, 0]
                if float(next_sigma.item()) > 0.0:
                    generated_states.append(
                        (state.detach(), next_sigma, adjacent_endpoint)
                    )

        if self._video_trajectory_supervision_mode == "random":
            supervised_index = int(
                torch.randint(
                    len(generated_states), (), device=device
                ).item()
            )
        else:
            supervised_index = len(generated_states) - 1
        supervised_states = generated_states[supervised_index : supervised_index + 1]
        frame_keep = ~video_pad_for_loss
        if self._ar_bootstrap_clean_prefix:
            frame_keep = frame_keep.clone()
            frame_keep[:, 0] = False
        denominator = frame_keep.sum()
        if bool(denominator <= 0):
            raise ValueError(
                "trajectory loss needs a non-bootstrap, non-padded video target"
            )
        endpoint_terms = []
        velocity_terms = []
        consistency_terms = []
        frame_pad_5d = (~frame_keep)[:, None, :, None, None]
        clean_video_for_loss = clean_video.float().masked_fill(frame_pad_5d, 0)
        for generated_state, sigma, adjacent_endpoint in supervised_states:
            prediction = forward_video(generated_state, sigma)
            endpoint = generated_state.float() - sigma * prediction.float()
            velocity_target = (
                generated_state.float() - clean_video.float()
            ) / sigma
            endpoint_for_loss = endpoint.masked_fill(frame_pad_5d, 0)
            prediction_for_loss = prediction.float().masked_fill(frame_pad_5d, 0)
            velocity_target_for_loss = velocity_target.masked_fill(
                frame_pad_5d, 0
            )
            adjacent_endpoint_for_loss = adjacent_endpoint.masked_fill(
                frame_pad_5d, 0
            )
            endpoint_per_frame = F.mse_loss(
                endpoint_for_loss, clean_video_for_loss, reduction="none"
            ).mean(dim=(1, 3, 4))
            velocity_per_frame = F.mse_loss(
                prediction_for_loss, velocity_target_for_loss, reduction="none"
            ).mean(dim=(1, 3, 4))
            consistency_per_frame = F.mse_loss(
                endpoint_for_loss, adjacent_endpoint_for_loss, reduction="none"
            ).mean(dim=(1, 3, 4))
            endpoint_per_frame = endpoint_per_frame.masked_fill(~frame_keep, 0)
            velocity_per_frame = velocity_per_frame.masked_fill(~frame_keep, 0)
            consistency_per_frame = consistency_per_frame.masked_fill(
                ~frame_keep, 0
            )
            endpoint_terms.append(
                endpoint_per_frame.sum() / denominator
            )
            velocity_terms.append(
                velocity_per_frame.sum() / denominator
            )
            consistency_terms.append(
                consistency_per_frame.sum() / denominator
            )
        endpoint_loss = torch.stack(endpoint_terms).mean()
        velocity_loss = torch.stack(velocity_terms).mean()
        consistency_loss = torch.stack(consistency_terms).mean()
        return (
            endpoint_loss,
            velocity_loss,
            consistency_loss,
            len(supervised_states),
            supervised_index,
        )

    # ------------------------------------------------------------------
    # AR compute_loss (override — base assumes a single non-duplicated forward)
    # ------------------------------------------------------------------

    def compute_loss(  # type: ignore[override]
        self,
        *,
        actions: Optional[Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        current_step: int = 0,  # noqa: ARG002 — AR samples its own schedules
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
        video_timestep_dtype = self._video_timestep_dtype(dtype)
        clean_video = inputs["input_latents"].to(device=device, dtype=dtype)
        B, _, T, _, _ = clean_video.shape
        fcs = self._ar_frame_chunk_size
        if T % fcs != 0:
            raise ValueError(
                f"latent frames T={T} not divisible by ar_frame_chunk_size={fcs}."
            )
        num_chunks = T // fcs
        raw_video_is_pad = inputs.get("video_is_pad")
        video_is_pad = self._coerce_ar_padding_mask(
            raw_video_is_pad,
            name="video_is_pad",
            batch_size=B,
            num_tokens=T,
            device=device,
            default_all_valid=True,
        )
        video_loss_keep = ~video_is_pad
        if self._ar_bootstrap_clean_prefix:
            video_loss_keep = video_loss_keep.clone()
            video_loss_keep[:, 0] = False
        local_expansion_enabled = self._video_local_expansion_weight > 0
        action_non_regression_enabled = self._action_non_regression_weight > 0
        afcc_enabled = self._action_facing_cache_consistency_weight > 0
        validate_afcc_weight(self._action_facing_cache_consistency_weight)
        afcc_capture_required = inputs.get(
            "phase6_action_facing_cache_capture_required", False
        )
        if type(afcc_capture_required) is not bool:
            raise TypeError(
                "phase6_action_facing_cache_capture_required must be boolean"
            )
        self._validate_action_non_regression_weight(
            self._action_non_regression_weight
        )
        if action_non_regression_enabled:
            self._maybe_init_action_video_memory_adapter()
        if (
            action_non_regression_enabled
            and self.action_video_memory_adapter is None
        ):
            raise RuntimeError(
                "action non-regression requires an enabled action video-memory adapter"
            )
        if afcc_enabled and self.action_video_memory_adapter is not None:
            raise RuntimeError("AFCC requires the action video-memory adapter disabled")
        if afcc_capture_required and afcc_enabled:
            raise ValueError("AFCC teacher capture and student treatment are distinct modes")
        if afcc_capture_required:
            if self.training:
                raise ValueError("AFCC teacher capture requires model eval mode")
            if self.action_video_memory_adapter is not None:
                raise ValueError("AFCC teacher capture forbids an action adapter")
        expansion_spec = None
        expansion_frame_mask = None
        if local_expansion_enabled:
            self._validate_local_expansion_contract(
                weight=self._video_local_expansion_weight,
                legacy_weights=(
                    self._video_trajectory_endpoint_weight,
                    self._video_trajectory_velocity_weight,
                    self._video_trajectory_consistency_weight,
                ),
                chunkwise_temporal_ops=self._ar_chunkwise_temporal_ops,
            )
        phase6_plan_rows = inputs.get("phase6_plan_rows")
        if phase6_plan_rows is not None and raw_video_is_pad is None:
            raise ValueError("Phase-6 requires an explicit video_is_pad mask")
        video_target_required = (
            phase6_plan_rows is not None
            or self._video_on_path_loss_weight > 0
            or self._video_trajectory_endpoint_weight > 0
            or self._video_trajectory_velocity_weight > 0
            or self._video_trajectory_consistency_weight > 0
            or local_expansion_enabled
        )
        if video_target_required and bool(
            (video_loss_keep.sum(dim=1) == 0).any()
        ):
            raise ValueError(
                "every sample must contain at least one non-bootstrap, "
                "non-padded video target"
            )
        if "phase6_reference_trace_required" in inputs:
            raise ValueError(
                "legacy phase6_reference_trace_required is forbidden; use the "
                "split common/T0 trace flags"
            )
        common_trace_required = inputs.get(
            "phase6_common_input_trace_required", False
        )
        t0_forward_trace_required = inputs.get(
            "phase6_t0_reference_forward_trace_required", False
        )
        if type(common_trace_required) is not bool:
            raise TypeError(
                "phase6_common_input_trace_required must be boolean"
            )
        if type(t0_forward_trace_required) is not bool:
            raise TypeError(
                "phase6_t0_reference_forward_trace_required must be boolean"
            )
        if t0_forward_trace_required and not common_trace_required:
            raise ValueError("T0 forward tracing requires common input tracing")
        if t0_forward_trace_required:
            if bool(
                getattr(
                    self.video_backbone,
                    "continuous_timestep_conditioning",
                    False,
                )
            ):
                raise ValueError("T0 forward tracing requires exact T0 conditioning")
            if self.training:
                raise ValueError("T0 forward tracing requires model eval mode")
            if local_expansion_enabled or action_non_regression_enabled:
                raise ValueError("T0 forward tracing requires exact E0/A0")
            if self.action_video_memory_adapter is not None:
                raise ValueError("T0 forward tracing forbids an action adapter")
        if (
            local_expansion_enabled or action_non_regression_enabled or afcc_enabled
        ) and phase6_plan_rows is None:
            raise ValueError(
                "Phase-6 expansion/action treatment requires phase6_plan_rows"
            )
        if afcc_capture_required and phase6_plan_rows is None:
            raise ValueError("AFCC teacher capture requires one Phase-6 plan row")
        phase6_seeds = None
        if phase6_plan_rows is not None:
            expansion_spec = phase6_expansion_spec(
                phase6_plan_rows, batch_size=B
            )
            if lambda_action != 0.0:
                raise ValueError(
                    "Phase-6 plan rows require lambda_action exactly 0 "
                    "(lambda_action=0)"
                )
            phase6_seeds = phase6_plan_rows[0]["domain_seeds"]
            expansion_frame_mask = current_video_chunk_mask(
                video_is_pad,
                batch_size=B,
                num_frames=T,
                frame_chunk_size=fcs,
                bootstrap_clean_prefix=self._ar_bootstrap_clean_prefix,
                device=device,
            )
            if common_trace_required and B != 1:
                raise ValueError("Phase-6 reference tracing requires batch size 1")
        elif common_trace_required or t0_forward_trace_required:
            raise ValueError("Phase-6 reference tracing requires one plan row")
        if action_non_regression_enabled:
            phase1_action_reference_error = (
                self._validate_phase1_action_reference_error(
                    inputs.get("phase1_action_reference_error"), device=device
                )
            )
        else:
            phase1_action_reference_error = None
        if not afcc_enabled and "phase1_action_facing_cache_reference" in inputs:
            raise ValueError("AFCC reference tensor is forbidden when AFCC is disabled")

        # ---- noisy video copy: per-chunk timestep -> per-frame sigma ----
        # NB: scheduler.sigmas/timesteps/weights live on CPU — index with CPU ids
        # then move the results to ``device`` (CUDA ids on a CPU tensor raise).
        num_ts_v = len(vb.scheduler.timesteps)
        if phase6_seeds is None:
            v_chunk_ids = torch.randint(0, num_ts_v, (B, num_chunks))
        else:
            v_chunk_ids = torch.randint(
                0,
                num_ts_v,
                (B, num_chunks),
                generator=_phase6_generator(
                    phase6_seeds["video-noise"],
                    "noisy-chunk-timestep",
                    device="cpu",
                ),
            )
        v_frame_ids = v_chunk_ids.repeat_interleave(fcs, dim=1)  # (B, T) on CPU
        v_sigma_fp32 = vb.scheduler.sigmas[v_frame_ids].to(
            device=device, dtype=torch.float32
        )
        v_sigma = v_sigma_fp32.to(dtype=dtype)  # (B, T)
        v_ts_val = vb.scheduler.timesteps[v_frame_ids].to(
            device=device, dtype=video_timestep_dtype
        )  # (B, T)
        if self._ar_bootstrap_clean_prefix:
            # F3 bootstrap: chunk-0 frame-0 is the real initial obs, supplied CLEAN at
            # t=0 (sigma=0 ⇒ noisy_video[:,:,0]==clean_video[:,:,0]). Excluded from the
            # video loss below (LingBot inpaints frame-0 of chunk 0).
            v_sigma[:, 0] = 0.0
            v_sigma_fp32[:, 0] = 0.0
            v_ts_val[:, 0] = 0.0
        if phase6_seeds is None:
            v_noise = torch.randn_like(clean_video)
        else:
            v_noise = torch.randn(
                clean_video.shape,
                generator=_phase6_generator(
                    phase6_seeds["video-noise"],
                    "noisy-epsilon",
                    device=device,
                ),
                device=device,
                dtype=dtype,
            )
        expansion_tangent = None
        if expansion_spec is not None:
            expansion_noise_generator = torch.Generator(device=device).manual_seed(
                expansion_spec.expansion_noise_seed
            )
            expansion_noise = torch.randn(
                clean_video.shape,
                generator=expansion_noise_generator,
                device=device,
                dtype=dtype,
            )
            expansion_mask_5d = expansion_frame_mask[:, None, :, None, None]
            v_noise = torch.where(expansion_mask_5d, expansion_noise, v_noise)
            expansion_tangent = v_noise.float() - clean_video.float()
        s = v_sigma.view(B, 1, T, 1, 1)
        noisy_video = (1 - s) * clean_video + s * v_noise
        if expansion_spec is not None:
            s_fp32 = v_sigma_fp32.view(B, 1, T, 1, 1)
            analytic_center = (
                (1.0 - s_fp32) * clean_video.float()
                + s_fp32 * v_noise.float()
            ).to(dtype=dtype)
            noisy_video = torch.where(
                expansion_mask_5d, analytic_center, noisy_video
            )
        v_target = v_noise - clean_video

        # ---- clean video copy: t=0 or (prob p) a small cond timestep ----
        clean_video_in, clean_video_ts = self._make_clean_copy(
            clean_video,
            vb.scheduler,
            B,
            T,
            device,
            dtype,
            frames=True,
            timestep_dtype=video_timestep_dtype,
            phase6_parent_seed=(
                None if phase6_seeds is None else phase6_seeds["video-noise"]
            ),
        )
        if self._ar_bootstrap_clean_prefix:
            # F3: frame 0 is the real init obs in BOTH copies (don't let the
            # clean-copy cond-noise aug touch the bootstrap frame). _make_clean_copy
            # returns contiguous tensors, so in-place writes are safe.
            clean_video_in[:, :, 0] = clean_video[:, :, 0]
            clean_video_ts[:, 0] = 0.0

        # ---- actions ----
        if actions is None:
            raise RuntimeError(
                "DualSystemARArchitecture.compute_loss requires actions (joint AR path)."
            )
        if lambda_video < 0 or lambda_action < 0:
            raise ValueError("lambda_video and lambda_action must be non-negative")
        if action_non_regression_enabled and lambda_action != 0.0:
            raise ValueError(
                "Phase-6 action non-regression requires lambda_action=0"
            )
        actions = actions.to(device=device, dtype=dtype)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        Ta = actions.shape[1]
        if Ta % num_chunks != 0:
            raise ValueError(
                f"action tokens Ta={Ta} not divisible by num_chunks={num_chunks}."
            )
        atok = Ta // num_chunks
        raw_action_is_pad = inputs.get("action_is_pad")
        action_is_pad = self._coerce_ar_padding_mask(
            raw_action_is_pad,
            name="action_is_pad",
            batch_size=B,
            num_tokens=Ta,
            device=device,
            default_all_valid=True,
        )
        valid_per_sample = (~action_is_pad).sum(dim=1)
        if bool((valid_per_sample == 0).any()):
            raise ValueError(
                "every sample must contain at least one supervised action token"
            )
        if phase6_plan_rows is not None and raw_action_is_pad is None:
            raise ValueError("Phase-6 requires an explicit action_is_pad mask")
        forward_video_is_pad = (
            video_is_pad if raw_video_is_pad is not None else None
        )
        forward_action_is_pad = (
            action_is_pad if raw_action_is_pad is not None else None
        )
        if afcc_enabled:
            phase1_action_facing_cache_reference = validate_afcc_reference(
                inputs.get("phase1_action_facing_cache_reference"),
                batch_size=B,
                num_layers=int(vb.num_layers),
                num_chunks=num_chunks,
                num_heads=int(vb.num_heads),
                action_tokens_per_chunk=atok,
                head_dim=int(vb.head_dim),
                device=device,
                dtype=dtype,
            )
        else:
            phase1_action_facing_cache_reference = None

        if expansion_spec is None:
            num_ts_a = len(action_scheduler.timesteps)
            a_chunk_ids = torch.randint(
                0, num_ts_a, (B, num_chunks)
            )  # CPU (index CPU scheduler tensors)
            a_tok_ids = a_chunk_ids.repeat_interleave(atok, dim=1)
            a_sigma = action_scheduler.sigmas[a_tok_ids].to(
                device=device, dtype=dtype
            )
            a_ts_val = action_scheduler.timesteps[a_tok_ids].to(
                device=device, dtype=dtype
            )
            a_noise = torch.randn_like(actions)
        else:
            a_tok_ids = None
            action_sigma = expansion_spec.action_sigma
            a_sigma = torch.full(
                (B, Ta), action_sigma, device=device, dtype=dtype
            )
            # Action time remains in the legacy model dtype for both T0 and T1.
            a_ts_val = torch.full(
                (B, Ta), action_sigma * 1000.0, device=device, dtype=dtype
            )
            a_noise = torch.randn(
                actions.shape,
                generator=_phase6_generator(
                    phase6_seeds["action-noise"],
                    "noisy-epsilon",
                    device=device,
                ),
                device=device,
                dtype=dtype,
            )
        noisy_actions = (1 - a_sigma.unsqueeze(-1)) * actions + a_sigma.unsqueeze(
            -1
        ) * a_noise
        a_target = a_noise - actions

        clean_action_in, clean_action_ts = self._make_clean_copy(
            actions,
            action_scheduler,
            B,
            Ta,
            device,
            dtype,
            frames=False,
            phase6_parent_seed=(
                None if phase6_seeds is None else phase6_seeds["action-noise"]
            ),
        )

        # ---- forward over the duplicated sequence ----
        fwd_inputs = {
            k: v
            for k, v in inputs.items()
            if k in ("context", "context_mask", "seq_lens")
        }
        pre_proprio_fwd_inputs = fwd_inputs
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
        phase6_common_input_trace_sha256 = None
        if common_trace_required:
            (
                prefix_action_tensor,
                prefix_action_scalar,
            ) = _phase6_partition_prefix_count(
                inputs.get("num_clean_prefix_actions"),
                name="num_clean_prefix_actions",
                batch_size=B,
                upper_bound=Ta,
            )
            (
                prefix_frame_tensor,
                prefix_frame_scalar,
            ) = _phase6_partition_prefix_count(
                inputs.get("num_clean_prefix_frames"),
                name="num_clean_prefix_frames",
                batch_size=B,
                upper_bound=T,
            )
            fwd_inputs = self._append_proprio_context_token(
                dict(fwd_inputs), proprio_state
            )
            post_context = fwd_inputs.get("context")
            post_context_mask = fwd_inputs.get("context_mask")
            post_seq_lens = fwd_inputs.get("seq_lens")
            if (
                post_context is not None
                and post_context_mask is None
                and post_seq_lens is not None
            ):
                positions = torch.arange(
                    post_context.shape[1], device=post_context.device
                )
                post_context_mask = positions.unsqueeze(0) < post_seq_lens.to(
                    device=post_context.device
                ).unsqueeze(1)
            common_trace_tensors = {
                "action_clean_copy": clean_action_in,
                "action_clean_timestep": clean_action_ts,
                "action_is_pad": action_is_pad,
                "action_noise": a_noise,
                "action_noisy": noisy_actions,
                "action_requested_sigma": a_sigma,
                "action_requested_timestep": a_ts_val,
                "action_target": a_target,
                "actions": actions,
                "context_post_proprio": post_context,
                "context_mask_post_proprio": post_context_mask,
                "context_pre_proprio": pre_proprio_fwd_inputs.get("context"),
                "context_mask_pre_proprio": pre_proprio_fwd_inputs.get(
                    "context_mask"
                ),
                "expansion_frame_mask": expansion_frame_mask,
                "expansion_tangent": expansion_tangent,
                "input_latents": clean_video,
                "num_clean_prefix_actions": prefix_action_tensor,
                "num_clean_prefix_frames": prefix_frame_tensor,
                "proprio_per_chunk": proprio_per_chunk,
                "proprio_seq": inputs.get("proprio_seq"),
                "proprio_state": proprio_state,
                "seq_lens_post_proprio": post_seq_lens,
                "video_clean_copy": clean_video_in,
                "video_is_pad": video_is_pad,
                "video_noise_effective": v_noise,
                "video_noisy": noisy_video,
                "video_requested_chunk_ids": v_chunk_ids,
                "video_requested_frame_ids": v_frame_ids,
                "video_requested_sigma_fp32": v_sigma_fp32,
                "video_target": v_target,
            }
            common_trace_metadata = {
                "action_sigma": expansion_spec.action_sigma,
                "bootstrap_clean_prefix": self._ar_bootstrap_clean_prefix,
                "domain_seeds": dict(phase6_seeds),
                "frame_chunk_size": fcs,
                "prepared_scalar_inputs": {
                    "num_clean_prefix_actions": prefix_action_scalar,
                    "num_clean_prefix_frames": prefix_frame_scalar,
                },
                "prepared_input_keys": sorted(
                    key
                    for key in inputs
                    if key
                    not in {
                        "phase1_action_reference_error",
                        "phase1_action_facing_cache_reference",
                        "phase6_action_facing_cache_capture_required",
                        "phase6_common_input_trace_required",
                        "phase6_t0_reference_forward_trace_required",
                    }
                ),
            }
            # Validate and freeze the common inputs before any model forward.
            phase6_common_input_trace_sha256 = trace_sha256(
                "phase6-common-input-v1",
                common_trace_tensors,
                metadata=common_trace_metadata,
            )
        phase6_proprio_context_prepared = (
            common_trace_required
            and bool(getattr(self, "_use_proprioception_context", False))
        )
        needs_on_path_forward = (
            self._video_on_path_loss_weight > 0 or lambda_action > 0
            or local_expansion_enabled
            or action_non_regression_enabled
            or afcc_enabled
            or afcc_capture_required
            or phase6_plan_rows is not None
        )
        if needs_on_path_forward:
            forward_result = self.forward(
                noisy_actions,
                None,
                proprio_state=proprio_state,
                phase6_proprio_context_prepared=(
                    phase6_proprio_context_prepared
                ),
                proprio_per_chunk=proprio_per_chunk,
                latents=noisy_video,
                ar_clean_latents=clean_video_in,
                ar_clean_actions=clean_action_in,
                ar_video_frame_timesteps=v_ts_val,
                ar_clean_video_frame_timesteps=clean_video_ts,
                ar_action_token_timesteps=a_ts_val,
                ar_clean_action_token_timesteps=clean_action_ts,
                ar_video_is_pad=forward_video_is_pad,
                ar_action_is_pad=forward_action_is_pad,
                ar_frame_chunk_size=fcs,
                ar_attn_window=self._ar_attn_window,
                timestep=v_ts_val.mean(dim=1),
                return_action_video_numerators=(
                    afcc_enabled or afcc_capture_required
                ),
                **fwd_inputs,
            )
            if afcc_enabled or afcc_capture_required:
                v_pred, a_pred, action_video_numerators = forward_result
            else:
                v_pred, a_pred = forward_result
                action_video_numerators = None
        else:
            v_pred = a_pred = None
            action_video_numerators = None

        if afcc_enabled:
            if action_video_numerators is None:
                raise RuntimeError("AFCC student numerator capture is absent")
            expected_student_shape = tuple(
                phase1_action_facing_cache_reference.shape
            )
            if tuple(action_video_numerators.shape) != expected_student_shape:
                raise RuntimeError(
                    "AFCC student capture shape differs from its reference: "
                    f"{tuple(action_video_numerators.shape)} != "
                    f"{expected_student_shape}"
                )
            if (
                action_video_numerators.dtype != dtype
                or action_video_numerators.device != device
            ):
                raise RuntimeError("AFCC student capture dtype/device drifted")
            afcc_loss = action_facing_cache_consistency_loss(
                action_video_numerators,
                phase1_action_facing_cache_reference,
                (~action_is_pad).reshape(B, num_chunks, atok),
            )
        else:
            afcc_loss = clean_video.new_zeros((), dtype=torch.float32)
        if afcc_capture_required:
            if action_video_numerators is None:
                raise RuntimeError("AFCC teacher numerator capture is absent")
            expected_capture_shape = (
                B,
                int(vb.num_layers),
                num_chunks,
                int(vb.num_heads),
                atok,
                int(vb.head_dim),
            )
            if tuple(action_video_numerators.shape) != expected_capture_shape:
                raise RuntimeError(
                    "AFCC teacher capture has the wrong coverage/shape: "
                    f"expected {expected_capture_shape}, got "
                    f"{tuple(action_video_numerators.shape)}"
                )
            if (
                action_video_numerators.dtype != dtype
                or action_video_numerators.device != device
                or action_video_numerators.requires_grad
            ):
                raise RuntimeError("AFCC teacher capture dtype/device/autograd drifted")
            if not bool(torch.isfinite(action_video_numerators).all()):
                raise RuntimeError("AFCC teacher capture contains non-finite values")
            afcc_teacher_capture = action_video_numerators.detach()
        else:
            afcc_teacher_capture = None

        # ---- per-frame weighted video MSE ----
        if self._video_on_path_loss_weight > 0:
            video_loss_pad = (~video_loss_keep)[:, None, :, None, None]
            per_v = F.mse_loss(
                v_pred.float().masked_fill(video_loss_pad, 0),
                v_target.float().masked_fill(video_loss_pad, 0),
                reduction="none",
            ).mean(dim=(1, 3, 4))
            wv = vb.scheduler.linear_timesteps_weights[v_frame_ids].to(
                dtype=torch.float32, device=device
            )
            frame_mask = video_loss_keep.to(dtype=torch.float32, device=device)
            per_v = per_v.masked_fill(~video_loss_keep, 0)
            loss_video_on_path = (per_v * wv).sum() / frame_mask.sum()
        else:
            loss_video_on_path = clean_video.new_zeros((), dtype=torch.float32)

        # ---- per-token action MSE (pad-masked, optionally timestep-weighted) ----
        if lambda_action > 0:
            action_loss_pad = action_is_pad[:, :, None]
            per_a = F.mse_loss(
                a_pred.float().masked_fill(action_loss_pad, 0),
                a_target.float().masked_fill(action_loss_pad, 0),
                reduction="none",
            ).mean(dim=2)
            wa = action_scheduler.training_weight(a_tok_ids).to(
                dtype=torch.float32, device=device
            )
            if action_is_pad is not None:
                keep_mask = ~action_is_pad
                keep = keep_mask.to(dtype=torch.float32, device=device)
                per_a = per_a.masked_fill(~keep_mask, 0)
                loss_action = (per_a * wa).sum() / keep.sum().clamp(min=1.0)
            else:
                loss_action = (per_a * wa).mean()
        else:
            loss_action = clean_video.new_zeros((), dtype=torch.float32)

        if phase6_plan_rows is not None:
            if a_pred is None:
                raise RuntimeError(
                    "Phase-6 action MSE requires the common action prediction"
                )
            phase6_action_unweighted_mse = (
                unweighted_pad_masked_action_mse(
                    a_pred, a_target, action_is_pad
                )
            )
        else:
            phase6_action_unweighted_mse = clean_video.new_zeros(
                1, dtype=torch.float32
            )

        phase6_timestep_forward_trace_sha256 = None
        if t0_forward_trace_required:
            phase6_timestep_forward_trace_sha256 = trace_sha256(
                "phase6-timestep-forward-v1",
                {
                    "action_prediction": a_pred,
                    "action_target": a_target,
                    "clean_video_timestep_effective": clean_video_ts,
                    "phase6_action_unweighted_mse": (
                        phase6_action_unweighted_mse
                    ),
                    "video_prediction": v_pred,
                    "video_target": v_target,
                    "video_timestep_effective": v_ts_val,
                },
                metadata={
                    "common_input_trace_sha256": (
                        phase6_common_input_trace_sha256
                    ),
                    "continuous_timestep_conditioning": bool(
                        getattr(
                            self.video_backbone,
                            "continuous_timestep_conditioning",
                            False,
                        )
                    ),
                },
            )

        if action_non_regression_enabled:
            action_non_regression_student_error = phase6_action_unweighted_mse
            action_non_regression_reference_error = (
                phase1_action_reference_error.reshape(1)
            )
            action_non_regression_raw_loss = one_sided_relative_excess_loss(
                action_non_regression_student_error,
                action_non_regression_reference_error,
                relative_epsilon=1.0e-4,
            )
            action_non_regression_loss = adapter_only_loss_surrogate(
                action_non_regression_raw_loss,
                self.action_video_memory_adapter,
            )
            action_non_regression_relative_excess = (
                action_non_regression_student_error.detach()
                - action_non_regression_reference_error
            ) / (action_non_regression_reference_error + 1.0e-4)
        else:
            action_non_regression_loss = clean_video.new_zeros(
                (), dtype=torch.float32
            )
            action_non_regression_raw_loss = clean_video.new_zeros(
                (), dtype=torch.float32
            )
            action_non_regression_student_error = clean_video.new_zeros(
                1, dtype=torch.float32
            )
            action_non_regression_reference_error = clean_video.new_zeros(
                1, dtype=torch.float32
            )
            action_non_regression_relative_excess = clean_video.new_zeros(
                1, dtype=torch.float32
            )

        if local_expansion_enabled:
            if v_pred is None:
                raise RuntimeError("local expansion requires the center velocity")
            if expansion_spec.direction == "flow_target_tangent":
                expansion_direction = expansion_tangent
                expansion_direction_index = 0
            else:
                direction_generator = torch.Generator(device=device).manual_seed(
                    expansion_spec.expansion_direction_seed
                )
                expansion_direction = rademacher_orthogonal_direction(
                    expansion_tangent,
                    expansion_frame_mask,
                    generator=direction_generator,
                )
                expansion_direction_index = 1
            center_visible, plus_visible, expansion_chord = (
                quantized_one_sided_chord(
                    noisy_video,
                    expansion_direction,
                    expansion_frame_mask,
                    epsilon_rms=expansion_spec.epsilon_rms,
                    model_dtype=dtype,
                )
            )
            video_pad_5d = video_is_pad[:, None, :, None, None]
            if not torch.equal(
                center_visible.masked_fill(video_pad_5d, 0),
                noisy_video.detach().masked_fill(video_pad_5d, 0),
            ):
                raise RuntimeError(
                    "analytic expansion center changed at the model-dtype boundary"
                )
            plus_pred, _ = self.forward(
                noisy_actions,
                None,
                proprio_state=proprio_state,
                phase6_proprio_context_prepared=(
                    phase6_proprio_context_prepared
                ),
                proprio_per_chunk=proprio_per_chunk,
                latents=plus_visible,
                ar_clean_latents=clean_video_in,
                ar_clean_actions=clean_action_in,
                ar_video_frame_timesteps=v_ts_val,
                ar_clean_video_frame_timesteps=clean_video_ts,
                ar_action_token_timesteps=a_ts_val,
                ar_clean_action_token_timesteps=clean_action_ts,
                ar_video_is_pad=forward_video_is_pad,
                ar_action_is_pad=forward_action_is_pad,
                ar_frame_chunk_size=fcs,
                ar_attn_window=self._ar_attn_window,
                timestep=v_ts_val.mean(dim=1),
                **fwd_inputs,
            )
            selected_sigmas = v_sigma_fp32[expansion_frame_mask]
            if selected_sigmas.numel() == 0 or not torch.equal(
                selected_sigmas, selected_sigmas[:1].expand_as(selected_sigmas)
            ):
                raise RuntimeError(
                    "the selected expansion chunk must have one shared sigma"
                )
            deployment_sigmas = fixed_50_step_shifted_sigmas(
                float(vb.scheduler.flow_shift), device=device
            )
            expansion_delta_sigma = nearest_reverse_delta_sigma(
                selected_sigmas[:1], deployment_sigmas
            )
            expansion_result = local_reverse_euler_expansion_loss(
                chord=expansion_chord,
                velocity_center=v_pred,
                velocity_plus=plus_pred,
                mask=expansion_frame_mask,
                delta_sigma=expansion_delta_sigma,
                maximum_unpenalized_rate=8.0,
            )
            local_expansion_loss = expansion_result.loss
            expansion_rate = expansion_result.rate.mean()
            expansion_factor = expansion_result.expansion.mean()
            expansion_chord_rms = expansion_result.chord_rms.mean()
            expansion_next_chord_rms = expansion_result.next_chord_rms.mean()
            expansion_epsilon_rms = clean_video.new_tensor(
                expansion_spec.epsilon_rms, dtype=torch.float32
            )
            expansion_action_sigma = clean_video.new_tensor(
                expansion_spec.action_sigma, dtype=torch.float32
            )
            expansion_global_step = clean_video.new_tensor(
                expansion_spec.global_step, dtype=torch.int64
            )
            expansion_extra_forwards = 1
        else:
            local_expansion_loss = clean_video.new_zeros((), dtype=torch.float32)
            expansion_rate = clean_video.new_zeros((), dtype=torch.float32)
            expansion_factor = clean_video.new_zeros((), dtype=torch.float32)
            expansion_chord_rms = clean_video.new_zeros((), dtype=torch.float32)
            expansion_next_chord_rms = clean_video.new_zeros(
                (), dtype=torch.float32
            )
            expansion_delta_sigma = clean_video.new_zeros((), dtype=torch.float32)
            expansion_epsilon_rms = clean_video.new_zeros((), dtype=torch.float32)
            expansion_action_sigma = clean_video.new_zeros((), dtype=torch.float32)
            expansion_global_step = clean_video.new_zeros((), dtype=torch.int64)
            expansion_direction_index = -1
            expansion_extra_forwards = 0

        trajectory_enabled = (
            self._video_trajectory_endpoint_weight > 0
            or self._video_trajectory_velocity_weight > 0
            or self._video_trajectory_consistency_weight > 0
        )
        if trajectory_enabled:
            trajectory_losses = self._trajectory_video_losses(
                clean_video=clean_video,
                video_noise=v_noise,
                actions=actions,
                fwd_inputs=fwd_inputs,
                proprio_state=proprio_state,
                proprio_per_chunk=proprio_per_chunk,
                frame_chunk_size=fcs,
                video_is_pad=forward_video_is_pad,
                action_is_pad=forward_action_is_pad,
            )
            (
                endpoint_loss,
                velocity_loss,
                consistency_loss,
                supervised_states,
                supervised_state_index,
            ) = trajectory_losses
        else:
            endpoint_loss = velocity_loss = consistency_loss = clean_video.new_zeros(
                (), dtype=torch.float32
            )
            supervised_states = 0
            supervised_state_index = -1

        loss_video = (
            self._video_on_path_loss_weight * loss_video_on_path
            + self._video_trajectory_endpoint_weight * endpoint_loss
            + self._video_trajectory_velocity_weight * velocity_loss
            + self._video_trajectory_consistency_weight * consistency_loss
            + self._video_local_expansion_weight * local_expansion_loss
        )

        loss = lambda_video * loss_video + lambda_action * loss_action
        if action_non_regression_enabled:
            loss = loss + action_non_regression_loss
        if afcc_enabled:
            loss = loss + afcc_loss
        return {
            "loss": loss,
            "loss_video": (lambda_video * loss_video).detach(),
            "loss_action": (lambda_action * loss_action).detach(),
            "loss_action_non_regression": action_non_regression_raw_loss.detach(),
            "phase6_action_non_regression_surrogate": (
                action_non_regression_loss
            ),
            "loss_action_facing_cache_consistency": afcc_loss.detach(),
            "phase6_action_facing_cache_consistency_surrogate": afcc_loss,
            "phase6_action_facing_cache_capture": afcc_teacher_capture,
            "action_facing_cache_consistency_active": clean_video.new_tensor(
                int(afcc_enabled), dtype=torch.int64
            ),
            "phase6_action_unweighted_mse": (
                phase6_action_unweighted_mse.detach().mean()
            ),
            "phase6_common_input_trace_sha256": (
                phase6_common_input_trace_sha256
            ),
            "phase6_timestep_forward_trace_sha256": (
                phase6_timestep_forward_trace_sha256
            ),
            "action_non_regression_student_error": (
                action_non_regression_student_error.detach().mean()
            ),
            "action_non_regression_reference_error": (
                action_non_regression_reference_error.detach().mean()
            ),
            "action_non_regression_relative_excess": (
                action_non_regression_relative_excess.detach().mean()
            ),
            "action_non_regression_active": clean_video.new_tensor(
                int(action_non_regression_enabled), dtype=torch.int64
            ),
            "loss_video_on_path": loss_video_on_path.detach(),
            "loss_video_trajectory_endpoint": endpoint_loss.detach(),
            "loss_video_trajectory_velocity": velocity_loss.detach(),
            "loss_video_trajectory_consistency": consistency_loss.detach(),
            "loss_video_local_expansion": local_expansion_loss.detach(),
            "video_local_expansion_rate": expansion_rate.detach(),
            "video_local_expansion_factor": expansion_factor.detach(),
            "video_local_expansion_chord_rms": expansion_chord_rms.detach(),
            "video_local_expansion_next_chord_rms": (
                expansion_next_chord_rms.detach()
            ),
            "video_local_expansion_delta_sigma": (
                expansion_delta_sigma.detach().mean()
            ),
            "video_local_expansion_epsilon_rms": expansion_epsilon_rms.detach(),
            "video_local_expansion_action_sigma": expansion_action_sigma.detach(),
            "video_local_expansion_global_step": expansion_global_step.detach(),
            "video_local_expansion_direction_index": clean_video.new_tensor(
                expansion_direction_index, dtype=torch.int64
            ),
            "video_local_expansion_extra_forwards": clean_video.new_tensor(
                expansion_extra_forwards, dtype=torch.int64
            ),
            "video_trajectory_supervised_states": clean_video.new_tensor(
                supervised_states, dtype=torch.int64
            ),
            "video_trajectory_supervised_state_index": clean_video.new_tensor(
                supervised_state_index, dtype=torch.int64
            ),
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
        seed: Optional[int] = None,
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

        - F3 bootstrap is a chunk-0-only boundary special case: clamp the real
          initial observation and emit chunk 0's action without a look-ahead video.
          Chunks c>=1 always use the steady-state predicted-video path.

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
        gen = None if seed is None else torch.Generator(device=device).manual_seed(seed)
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
            proprio_c = (
                proprio_states[c]
                if isinstance(proprio_states, (list, tuple))
                else proprio_states
            )
            step_ctx, step_mask = self._rollout_step_context(
                context, seq_lens, proprio_c
            )
            # F4: per-step proprio AdaLN deltas (one robot state per step).
            v_pe, a_pe = self._rollout_proprio_deltas(proprio_c)

            # 1. ingest realized obs as confirmed clean video chunk at frame 2c.
            self._ingest_clean_video(
                driver,
                cache,
                obs,
                frame_id=2 * c,
                context=step_ctx,
                context_mask=step_mask,
                v_proprio=v_pe,
            )

            if self._ar_bootstrap_clean_prefix and c == 0:
                # F3 bootstrap is an episode-boundary special case. Later chunks
                # must exercise the steady-state imagination-conditioned path.
                pred_action = self._denoise_action_chunk(
                    driver,
                    cache,
                    frame_id=2 * c + 1,
                    batch=B,
                    action_tokens=action_tokens_per_chunk,
                    a_sigmas=a_sigmas,
                    a_ts=a_ts,
                    context=step_ctx,
                    context_mask=step_mask,
                    gen=gen,
                    a_proprio=a_pe,
                )
            else:
                # Steady-state (default): predict next video chunk (frame 2c+2)...
                pred_video = self._denoise_video_chunk(
                    driver,
                    cache,
                    frame_id=2 * (c + 1),
                    like=obs,
                    v_sigmas=v_sigmas,
                    v_ts=v_ts,
                    context=step_ctx,
                    context_mask=step_mask,
                    gen=gen,
                    v_proprio=v_pe,
                )
                # ...then the action chunk (frame 2c+3) conditioned on it.
                pred_action = self._denoise_action_chunk(
                    driver,
                    cache,
                    frame_id=2 * (c + 1) + 1,
                    batch=B,
                    action_tokens=action_tokens_per_chunk,
                    a_sigmas=a_sigmas,
                    a_ts=a_ts,
                    context=step_ctx,
                    context_mask=step_mask,
                    gen=gen,
                    a_proprio=a_pe,
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

    def _ingest_clean_video(
        self, driver, cache, clean, *, frame_id, context, context_mask, v_proprio=None
    ):
        vb = self.video_backbone
        B, n_frames = clean.shape[0], clean.shape[2]
        timestep_dtype = self._video_timestep_dtype(clean.dtype)
        zeros = torch.zeros(B, device=clean.device, dtype=timestep_dtype)
        # Absolute frame offset: video chunk c is at frame_id 2c -> abs frames c*fcs..
        start_f = (frame_id // 2) * n_frames
        rope_idx = torch.arange(start_f, start_f + n_frames, device=clean.device)
        fpe = (
            None if v_proprio is None else v_proprio[:, None, :].expand(B, n_frames, -1)
        )
        extra = {} if fpe is None else {"frame_proprio_emb": fpe}
        vstate = vb.prepare(
            latents=clean,
            timestep=zeros,
            frame_timesteps=torch.zeros(
                B, n_frames, device=clean.device, dtype=timestep_dtype
            ),
            rope_frame_index=rope_idx,
            context=context,
            context_mask=context_mask,
            **extra,
        )
        driver.run_ar_chunk_through_backbone(
            vb, vstate, cache, frame_id, store_clean=True, is_pred=False
        )

    def _denoise_video_chunk(
        self,
        driver,
        cache,
        *,
        frame_id,
        like,
        v_sigmas,
        v_ts,
        context,
        context_mask,
        gen,
        v_proprio=None,
    ):
        vb = self.video_backbone
        device, dtype = like.device, like.dtype
        timestep_dtype = self._video_timestep_dtype(dtype)
        B, n_frames = like.shape[0], like.shape[2]
        start_f = (frame_id // 2) * n_frames
        rope_idx = torch.arange(start_f, start_f + n_frames, device=device)
        fpe = (
            None if v_proprio is None else v_proprio[:, None, :].expand(B, n_frames, -1)
        )
        extra = {} if fpe is None else {"frame_proprio_emb": fpe}
        # Draw the same model-dtype noise as the legacy path, then retain an FP32
        # master across steps. Only the model-visible input is cast to model dtype.
        x = _init_fp32_euler_master(
            like.shape,
            generator=gen,
            device=device,
            model_dtype=dtype,
        )
        for i in range(len(v_sigmas) - 1):
            t_val = float(v_ts[i])
            frame_ts = torch.full(
                (B, n_frames), t_val, device=device, dtype=timestep_dtype
            )
            vstate = vb.prepare(
                latents=x.to(dtype=dtype),
                timestep=torch.full(
                    (B,), t_val, device=device, dtype=timestep_dtype
                ),
                frame_timesteps=frame_ts,
                rope_frame_index=rope_idx,
                context=context,
                context_mask=context_mask,
                **extra,
            )
            driver.run_ar_chunk_through_backbone(
                vb, vstate, cache, frame_id, store_clean=False
            )
            v = vb.finalize(vstate)
            x = _fp32_euler_update(x, v, v_sigmas[i + 1] - v_sigmas[i])
        # store the predicted (now clean) chunk for downstream chunks.
        final_x = x.to(dtype=dtype)
        vstate = vb.prepare(
            latents=final_x,
            timestep=torch.zeros(B, device=device, dtype=timestep_dtype),
            frame_timesteps=torch.zeros(
                B, n_frames, device=device, dtype=timestep_dtype
            ),
            rope_frame_index=rope_idx,
            context=context,
            context_mask=context_mask,
            **extra,
        )
        driver.run_ar_chunk_through_backbone(
            vb, vstate, cache, frame_id, store_clean=True, is_pred=True
        )
        return final_x

    def _denoise_action_chunk(
        self,
        driver,
        cache,
        *,
        frame_id,
        batch,
        action_tokens,
        a_sigmas,
        a_ts,
        context,
        context_mask,
        gen,
        a_proprio=None,
    ):
        ab = self.action_backbone
        device, dtype = self.device, self.dtype
        x = torch.randn(
            batch,
            action_tokens,
            ab.action_dim,
            generator=gen,
            device=device,
            dtype=dtype,
        )
        frame_ids = torch.full(
            (action_tokens,), frame_id, dtype=torch.long, device=device
        )
        ctx_mask = context_mask
        tpe = (
            None if a_proprio is None else a_proprio[:, None, :]
        )  # (B,1,Da) broadcasts over tokens
        a_extra = {} if tpe is None else {"token_proprio_emb": tpe}
        for i in range(len(a_sigmas) - 1):
            t_val = float(a_ts[i])
            token_ts = torch.full(
                (batch, action_tokens), t_val, device=device, dtype=dtype
            )
            astate = ab.prepare_state(
                x,
                torch.full((batch,), t_val, device=device, dtype=dtype),
                context=context,
                context_mask=ctx_mask,
                token_timesteps=token_ts,
                frame_ids=frame_ids,
                rope_positions=frame_ids,
                **a_extra,
            )
            driver.run_ar_chunk_through_backbone(
                ab, astate, cache, frame_id, store_clean=False
            )
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
            x,
            torch.zeros(batch, device=device, dtype=dtype),
            context=context,
            context_mask=ctx_mask,
            token_timesteps=torch.zeros(
                batch, action_tokens, device=device, dtype=dtype
            ),
            frame_ids=frame_ids,
            rope_positions=frame_ids,
            **a_extra,
        )
        driver.run_ar_chunk_through_backbone(
            ab, astate, cache, frame_id, store_clean=True, is_pred=False
        )
        return x

    def _make_clean_copy(
        self,
        clean,
        scheduler,
        B,
        L,
        device,
        dtype,
        *,
        frames: bool,
        timestep_dtype: Optional[torch.dtype] = None,
        phase6_parent_seed: Optional[int] = None,
    ):
        """Build the clean conditioning copy: t=0, or (prob ar_noisy_cond_prob) a
        small random cond timestep applied per-sample. Returns (copy, ts_values)
        where ``ts_values`` is ``(B, L)`` for blocks_split / action per-token t-mod."""
        num_ts = len(scheduler.timesteps)
        max_cond = max(1, int(self._ar_cond_max_ratio * num_ts))
        if phase6_parent_seed is None:
            do_cond = torch.rand(B, device=device) < self._ar_noisy_cond_prob
            cond_ids = torch.randint(0, max_cond, (B,))
        else:
            do_cond = (
                torch.rand(
                    B,
                    device=device,
                    generator=_phase6_generator(
                        phase6_parent_seed,
                        "clean-copy-decision",
                        device=device,
                    ),
                )
                < self._ar_noisy_cond_prob
            )
            cond_ids = torch.randint(
                0,
                max_cond,
                (B,),
                generator=_phase6_generator(
                    phase6_parent_seed,
                    "clean-copy-timestep",
                    device="cpu",
                ),
            )
        raw_sigma = scheduler.sigmas[cond_ids].to(device=device, dtype=dtype)
        if timestep_dtype is None or timestep_dtype == dtype:
            raw_ts = scheduler.timesteps[cond_ids].to(device=device, dtype=dtype)
            zeros = torch.zeros(B, device=device, dtype=dtype)
            cond_sigma = torch.where(do_cond, raw_sigma, zeros)
            cond_ts = torch.where(do_cond, raw_ts, zeros)
        else:
            raw_ts = scheduler.timesteps[cond_ids].to(
                device=device, dtype=timestep_dtype
            )
            sigma_zeros = torch.zeros(B, device=device, dtype=dtype)
            timestep_zeros = torch.zeros(
                B, device=device, dtype=timestep_dtype
            )
            cond_sigma = torch.where(do_cond, raw_sigma, sigma_zeros)
            cond_ts = torch.where(do_cond, raw_ts, timestep_zeros)
        if phase6_parent_seed is None:
            noise = torch.randn_like(clean)
        else:
            noise = torch.randn(
                clean.shape,
                generator=_phase6_generator(
                    phase6_parent_seed,
                    "clean-copy-epsilon",
                    device=device,
                ),
                device=device,
                dtype=dtype,
            )
        if frames:  # (B, C, L, H, W)
            s = cond_sigma.view(B, 1, 1, 1, 1)
        else:  # actions (B, L, Ad)
            s = cond_sigma.view(B, 1, 1)
        copy = (1 - s) * clean + s * noise
        ts_values = cond_ts.view(B, 1).expand(B, L).contiguous()
        return copy, ts_values


__all__ = ["DualSystemARArchitecture"]
