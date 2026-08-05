"""Shared numerical core for offline CACH staging and synthetic admission.

This module deliberately contains no architecture or execution authority.  It
does not construct or retain either backbone, load external state, publish
cache state, or authorize training/deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from sana_wam.cach.action_conditioning import (
    reduce_end_of_bin_action_condition,
)
from sana_wam.cach.prefix_compaction import FixedKPrefixPlan
from sana_wam.cach.staging_variant import (
    CACHStagingVariant,
    require_cach_staging_variant,
)
from sana_wam.model.action_chunk_layout import (
    ChunkActionLayout,
    SelectedProprioBinding,
)
from sana_wam.model.video_backbone.sana.hybrid_cache import (
    CacheScratch,
    LayerRegistry,
    StagedLayerPayload,
    tensor_digest,
)
from sana_wam.model.video_backbone.sana.hybrid_cache_codec import (
    scratch_to_vendor_cache,
    vendor_cache_to_staged_payloads,
)
from sana_wam.model.video_backbone.sana.hybrid_cache_stage2b import (
    Stage2BDenoiseReadView,
)


class _ConditioningProtocol(Protocol):
    """Structural seam used to avoid a core/stager import cycle."""

    layout_instance_digest: str
    chunk_id: int
    source_state_manifest_digest: str
    context: Tensor
    context_mask: Tensor | None
    seq_lens: Tensor | None
    proprio_state: Tensor
    proprio_bindings: tuple[SelectedProprioBinding, ...]
    conditioning_digest: str

    def verify_binding(
        self,
        *,
        layout: ChunkActionLayout,
        chunk_id: int,
        source_state_manifest_digest: str,
    ) -> None:
        ...


@dataclass(frozen=True)
class CACHNumericalResult:
    """Typed output of one CACH chunk execution.

    ``staged_layer_payloads`` is present only for the paired ``t=0`` path.
    Read-only denoise never exposes a transaction-local vendor cache as a
    candidate for publication.
    """

    video_prediction: Tensor = field(repr=False, compare=False)
    action_prediction: Tensor = field(repr=False, compare=False)
    staged_layer_payloads: tuple[StagedLayerPayload, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("video_prediction", "action_prediction"):
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch Tensor")
            if not bool(torch.isfinite(value.detach()).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        if self.staged_layer_payloads is not None:
            if not isinstance(self.staged_layer_payloads, tuple) or any(
                not isinstance(payload, StagedLayerPayload)
                for payload in self.staged_layer_payloads
            ):
                raise TypeError(
                    "staged_layer_payloads must be a tuple of StagedLayerPayload"
                )


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _exact_layout_mask(
    values: tuple[bool, ...], *, batch: int, device: torch.device
) -> Tensor:
    return (
        torch.tensor(values, dtype=torch.bool, device=device)
        .view(1, len(values))
        .expand(batch, len(values))
    )


class CACHNumericalCore(nn.Module):
    """Shared CACH read-only/paired numerical implementation.

    The stable state-dict key is ``no_action_slot`` on this module (and
    ``cach_core.no_action_slot`` when owned by the architecture).  No backbone
    is accepted by the constructor or retained as an attribute.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if type(action_dim) is not int or action_dim <= 0:
            raise ValueError("action_dim must be a positive plain integer")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("NO_ACTION dtype must be a floating torch dtype")
        self.register_parameter(
            "no_action_slot",
            nn.Parameter(torch.zeros(action_dim, dtype=dtype, device=device)),
        )
        if bool(self.no_action_slot.detach().count_nonzero()):
            raise RuntimeError("NO_ACTION must initialize to exact zero")

    @property
    def action_dim(self) -> int:
        return int(self.no_action_slot.shape[0])

    def _append_proprio_context(
        self,
        conditioning: _ConditioningProtocol,
        *,
        proprio_context_encoder: nn.Module,
    ) -> tuple[Tensor, Tensor]:
        """Deterministically append the bound proprio token.

        Training-time modality dropout/noise is deliberately absent from this
        offline staging primitive.  The caller-owned encoder is used but never
        retained or registered by the core.
        """

        if not isinstance(proprio_context_encoder, nn.Module):
            raise TypeError("proprio_context_encoder must be an nn.Module")
        context = conditioning.context
        proprio_state = conditioning.proprio_state
        if context.ndim != 3 or proprio_state.ndim != 2:
            raise ValueError("context/proprio_state must be exact [B,L,D]/[B,P]")
        if context.shape[0] != proprio_state.shape[0]:
            raise ValueError("context and proprio_state batch axes differ")
        encoder_parameter = next(proprio_context_encoder.parameters(), None)
        encoder_dtype = (
            encoder_parameter.dtype
            if encoder_parameter is not None
            else proprio_state.dtype
        )
        encoder_device = (
            encoder_parameter.device
            if encoder_parameter is not None
            else context.device
        )
        if encoder_device != context.device:
            raise ValueError("proprio encoder and bound context devices differ")
        try:
            token = proprio_context_encoder(
                proprio_state.to(device=encoder_device, dtype=encoder_dtype)
            )
        except TypeError as exc:
            raise TypeError("proprio_context_encoder rejected the bound state") from exc
        if not isinstance(token, Tensor) or tuple(token.shape) != (
            context.shape[0],
            context.shape[2],
        ):
            raise ValueError(
                "proprio_context_encoder must return exact [B,context_dim]"
            )
        token = token.to(device=context.device, dtype=context.dtype)
        if not bool(torch.isfinite(token.detach()).all()):
            raise ValueError("encoded proprio token contains NaN or Inf")

        if conditioning.context_mask is not None:
            mask = conditioning.context_mask
        elif conditioning.seq_lens is not None:
            positions = torch.arange(context.shape[1], device=context.device)
            mask = positions.view(1, -1) < conditioning.seq_lens.to(
                device=context.device
            ).view(-1, 1)
        else:
            mask = torch.ones(
                context.shape[:2], dtype=torch.bool, device=context.device
            )
        if mask.dtype is not torch.bool or tuple(mask.shape) != tuple(context.shape[:2]):
            raise TypeError("resolved context mask must be exact boolean [B,L]")
        full_context = torch.cat([context, token.unsqueeze(1)], dim=1)
        full_mask = torch.cat(
            [
                mask,
                torch.ones(
                    (mask.shape[0], 1), dtype=torch.bool, device=mask.device
                ),
            ],
            dim=1,
        )
        return full_context, full_mask

    def _run_common(
        self,
        *,
        staging_variant: CACHStagingVariant,
        video_backbone: nn.Module,
        action_backbone: nn.Module,
        proprio_context_encoder: nn.Module,
        registry: LayerRegistry,
        action_layout: ChunkActionLayout,
        chunk_id: int,
        chunk_latents: Tensor,
        frame_valid_mask: Tensor,
        noisy_actions: Tensor,
        action_valid_mask: Tensor,
        committed_actions: Tensor | None,
        committed_history_digest: str,
        expected_committed_actions_digest: str | None,
        cache_scratch: CacheScratch,
        conditioning: _ConditioningProtocol,
        video_timestep: Tensor,
        action_timestep: Tensor,
        save_kv_cache: bool,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> CACHNumericalResult:
        staging_variant = require_cach_staging_variant(staging_variant)
        if not isinstance(video_backbone, nn.Module):
            raise TypeError("video_backbone must be an nn.Module")
        if not isinstance(action_backbone, nn.Module):
            raise TypeError("action_backbone must be an nn.Module")
        if not isinstance(registry, LayerRegistry):
            raise TypeError("registry must be a LayerRegistry")
        if not isinstance(cache_scratch, CacheScratch):
            raise TypeError("cache_scratch must be a CacheScratch")
        if not isinstance(action_layout, ChunkActionLayout):
            raise TypeError("action_layout must be a ChunkActionLayout")
        if type(chunk_id) is not int or not 0 <= chunk_id < len(action_layout.chunks):
            raise IndexError("chunk_id is outside action_layout")
        chunk = action_layout.chunks[chunk_id]
        action_layout.verify_instance_digest(conditioning.layout_instance_digest)
        conditioning.verify_binding(
            layout=action_layout,
            chunk_id=chunk_id,
            source_state_manifest_digest=cache_scratch.source_state_manifest_digest,
        )
        _require_sha256(committed_history_digest, "committed_history_digest")
        if len(cache_scratch.layers) != len(registry.layers):
            raise ValueError("cache scratch and layer registry depths differ")
        cache_scratch.assert_unchanged()

        if not isinstance(chunk_latents, Tensor) or chunk_latents.ndim != 5:
            raise ValueError("chunk_latents must have exact rank 5")
        batch, _, fixed_k = chunk_latents.shape[:3]
        if fixed_k != len(chunk.latent_valid_mask):
            raise ValueError("chunk_latents must retain the layout fixed-K axis")
        if (
            not isinstance(frame_valid_mask, Tensor)
            or frame_valid_mask.dtype is not torch.bool
            or tuple(frame_valid_mask.shape) != (batch, fixed_k)
        ):
            raise TypeError("frame_valid_mask must be exact boolean [B,K]")
        expected_frame_mask = _exact_layout_mask(
            chunk.latent_valid_mask,
            batch=batch,
            device=frame_valid_mask.device,
        )
        if not torch.equal(frame_valid_mask, expected_frame_mask):
            raise ValueError("frame_valid_mask differs from the bound layout chunk")
        if chunk_latents.device != frame_valid_mask.device:
            raise ValueError("chunk latents and frame mask devices differ")

        if not isinstance(noisy_actions, Tensor) or noisy_actions.ndim != 3:
            raise ValueError("noisy_actions must have exact rank 3")
        if tuple(noisy_actions.shape) != (
            batch,
            chunk.action_slot_capacity,
            self.action_dim,
        ):
            raise ValueError("noisy_actions differ from the fixed layout capacity")
        if (
            not isinstance(action_valid_mask, Tensor)
            or action_valid_mask.dtype is not torch.bool
            or tuple(action_valid_mask.shape)
            != (batch, chunk.action_slot_capacity)
        ):
            raise TypeError("action_valid_mask must be exact boolean [B,A_slots]")
        expected_action_mask = _exact_layout_mask(
            chunk.action_valid_mask,
            batch=batch,
            device=action_valid_mask.device,
        )
        if not torch.equal(action_valid_mask, expected_action_mask):
            raise ValueError("action_valid_mask differs from the bound layout chunk")
        if noisy_actions.device != action_valid_mask.device:
            raise ValueError("action tensor and action mask devices differ")
        if noisy_actions.dtype != chunk_latents.dtype:
            raise TypeError("video/action dtypes differ")
        if noisy_actions.device != chunk_latents.device:
            raise ValueError("video/action devices differ")
        if committed_actions is None:
            if chunk.action_start != 0:
                raise ValueError("continuation chunk requires committed history")
            if expected_committed_actions_digest is not None:
                raise ValueError("empty committed history cannot carry a tensor digest")
        else:
            if tuple(committed_actions.shape) != (
                batch,
                chunk.action_start,
                self.action_dim,
            ):
                raise ValueError("committed history must cover [0,action_start)")
            if committed_actions.dtype != noisy_actions.dtype:
                raise TypeError("committed/current action dtypes differ")
            if committed_actions.device != noisy_actions.device:
                raise ValueError("committed/current action devices differ")
            _require_sha256(
                expected_committed_actions_digest,
                "expected_committed_actions_digest",
            )
            if tensor_digest(committed_actions) != expected_committed_actions_digest:
                raise ValueError("committed action prefix digest differs")

        if staging_variant is CACHStagingVariant.CACH_A:
            if self.no_action_slot.dtype != noisy_actions.dtype:
                raise TypeError("model-owned NO_ACTION and actions dtypes differ")
            if self.no_action_slot.device != noisy_actions.device:
                raise ValueError("model-owned NO_ACTION and actions devices differ")
            if not bool(torch.isfinite(self.no_action_slot.detach()).all()):
                raise ValueError("model-owned NO_ACTION contains NaN or Inf")
            condition = reduce_end_of_bin_action_condition(
                noisy_actions,
                committed_actions=committed_actions,
                chunk=chunk,
                no_action_slot=self.no_action_slot,
            )
            if not torch.equal(condition.latent_valid_mask, frame_valid_mask):
                raise RuntimeError(
                    "reduced condition mask differs from paired video mask"
                )
            video_action_condition: Tensor | None = condition.condition
            video_frame_valid_mask = condition.latent_valid_mask
        elif staging_variant is CACHStagingVariant.REF_GDN_CORRECTED:
            # The corrected reference retains the same layout, action stream,
            # absolute Action RoPE, history identity, and recurrent cache path,
            # but the action-to-video branch is an exact identity bypass.  In
            # particular, this branch must not inspect the numerical value of
            # the model-owned NO_ACTION parameter.
            video_action_condition = None
            video_frame_valid_mask = frame_valid_mask
        else:  # pragma: no cover - guarded by the strict enum validator above
            raise AssertionError("unreachable CACH staging variant")
        full_context, full_context_mask = self._append_proprio_context(
            conditioning,
            proprio_context_encoder=proprio_context_encoder,
        )
        if full_context.device != chunk_latents.device:
            raise ValueError("bound context and chunk devices differ")
        if full_context.dtype != chunk_latents.dtype:
            raise TypeError("bound context and chunk dtypes differ")

        vendor_cache = scratch_to_vendor_cache(cache_scratch, registry)
        bridge_layers = tuple(getattr(action_backbone, "bridge_layers", ()))
        if not bridge_layers:
            raise ValueError("action backbone must declare non-empty bridge_layers")
        if any(type(index) is not int or index < 0 for index in bridge_layers):
            raise TypeError("action bridge layers must be non-negative plain integers")
        if any(index >= len(registry.layers) for index in bridge_layers):
            raise ValueError("action bridge layer lies outside the registry")

        video_prediction, bridges, vendor_cache = video_backbone.run_chunk(
            chunk_latents,
            video_timestep,
            context=full_context,
            context_mask=full_context_mask,
            seq_lens=None,
            kv_cache=vendor_cache,
            start_f=chunk.latent_start,
            end_f=chunk.latent_end,
            save_kv_cache=save_kv_cache,
            bridge_layers=bridge_layers,
            action_condition=video_action_condition,
            frame_valid_mask=video_frame_valid_mask,
        )
        if not isinstance(bridges, dict) or set(bridges) != set(bridge_layers):
            raise RuntimeError("video backbone returned an incomplete bridge set")

        action_plan = FixedKPrefixPlan.from_mask(action_valid_mask)
        compact_actions = action_plan.compact_condition(noisy_actions)
        if action_plan.valid_slots != chunk.valid_action_count:
            raise RuntimeError("action prefix compaction differs from layout ownership")
        rope_positions = torch.arange(
            chunk.action_rope_start,
            chunk.action_rope_end,
            device=noisy_actions.device,
            dtype=torch.long,
        )
        if rope_positions.numel() != action_plan.valid_slots:
            raise RuntimeError("Action RoPE range differs from valid action ownership")
        get_action_rope = getattr(action_backbone, "_get_rope_freqs_at", None)
        if not callable(get_action_rope):
            raise TypeError("action backbone lacks absolute Action RoPE lookup")
        action_freqs = get_action_rope(rope_positions).to(device=noisy_actions.device)
        video_plan = FixedKPrefixPlan.from_mask(frame_valid_mask)
        compact_bridges = {
            index: video_plan.compact_token_sequence(bridges[index])
            for index in bridge_layers
        }
        bridge_tuple_builder = getattr(action_backbone, "bridge_tuple_from_dict", None)
        if not callable(bridge_tuple_builder):
            raise TypeError("action backbone lacks bridge_tuple_from_dict")
        action_prediction_compact = action_backbone.forward_with_bridge_tuple(
            compact_actions,
            bridge_tuple_builder(compact_bridges),
            action_timestep,
            context=full_context,
            context_mask=full_context_mask,
            action_freqs=action_freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        action_prediction = action_plan.restore_token_sequence(
            action_prediction_compact
        )
        cache_scratch.assert_unchanged()
        staged = (
            vendor_cache_to_staged_payloads(vendor_cache, registry)
            if save_kv_cache
            else None
        )
        return CACHNumericalResult(
            video_prediction=video_prediction,
            action_prediction=action_prediction,
            staged_layer_payloads=staged,
        )

    def run_readonly_chunk(
        self,
        *,
        staging_variant: CACHStagingVariant,
        cache_read_view: Stage2BDenoiseReadView,
        cache_scratch: CacheScratch,
        **kwargs: Any,
    ) -> CACHNumericalResult:
        """Run a denoise pass without producing publishable cache payloads."""

        if type(cache_read_view) is not Stage2BDenoiseReadView:
            raise TypeError("cache_read_view must be the exact Stage2B read-view type")
        cache_read_view.assert_integrity()
        chunk_id = kwargs.get("chunk_id")
        conditioning = kwargs.get("conditioning")
        if (
            cache_read_view.next_chunk_id != chunk_id
            or cache_read_view.staging_variant is not staging_variant
            or cache_read_view.state_manifest_digest
            != cache_scratch.source_state_manifest_digest
            or cache_read_view._state_identity != cache_scratch._source_state_identity
        ):
            raise RuntimeError("read view and detached cache scratch differ")
        if (
            conditioning is None
            or conditioning.source_state_manifest_digest
            != cache_read_view.state_manifest_digest
        ):
            raise RuntimeError("conditioning is not bound to the read view state")
        return self._run_common(
            staging_variant=staging_variant,
            cache_scratch=cache_scratch,
            save_kv_cache=False,
            **kwargs,
        )

    def run_paired_t0(
        self,
        *,
        staging_variant: CACHStagingVariant,
        cache_scratch: CacheScratch,
        video_timestep: Tensor,
        action_timestep: Tensor,
        **kwargs: Any,
    ) -> CACHNumericalResult:
        """Run the sole clean paired stage into a transaction-local cache."""

        for name, value in (
            ("video_timestep", video_timestep),
            ("action_timestep", action_timestep),
        ):
            if not isinstance(value, Tensor) or bool(value.detach().count_nonzero()):
                raise ValueError(f"paired {name} must be an exact-zero tensor")
        with torch.no_grad():
            return self._run_common(
                staging_variant=staging_variant,
                cache_scratch=cache_scratch,
                video_timestep=video_timestep,
                action_timestep=action_timestep,
                save_kv_cache=True,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
                **kwargs,
            )


__all__ = ["CACHNumericalCore", "CACHNumericalResult"]
