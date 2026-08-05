"""Production-shaped offline paired staging for CACH Stage 2B.

This module is deliberately below every loader, trainer, policy server, and
deploy boundary.  A stager owns no cache manager and no sequence of requests.
It binds exactly one chunk's immutable conditioning, consumes the
manager-created :class:`PairedStagingContext`, and returns only typed staged
layer payloads.  Receipt publication and live-pointer advancement remain the
manager's responsibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
import torch
from torch import Tensor, nn

from sana_wam.cach.committed_action_history import CommittedActionHistoryView
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_chunk_layout import (
    ChunkActionLayout,
    SelectedProprioBinding,
    canonical_proprio_row_sha256,
)
from sana_wam.model.cach_numerical_core import CACHNumericalCore
from sana_wam.model.video_backbone.sana.hybrid_cache import (
    LayerKind,
    LayerRegistry,
    LayerSpec,
    StagedLayerPayload,
    tensor_digest,
    validate_cach_a_registry,
)
from sana_wam.model.video_backbone.sana.hybrid_cache_stage2b import (
    Stage2BPairedStagingContext,
)


_CONDITIONING_SCHEMA = "cach.chunk_conditioning.v1"


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("conditioning manifest is not canonical JSON") from exc


def _tensor_manifest(value: Tensor) -> dict[str, object]:
    return {
        "device": str(value.device),
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": tensor_digest(value),
        "shape": list(value.shape),
    }


def _binding_manifest(binding: SelectedProprioBinding) -> dict[str, object]:
    return {
        "chunk_id": binding.chunk_id,
        "layout_instance_digest": binding.layout_instance_digest,
        "proprio_value_sha256": binding.proprio_value_sha256,
        "raw_index": binding.raw_index,
        "source_receipt_digest": binding.source_receipt_digest,
        "timestamp": binding.timestamp,
    }


def _conditioning_manifest(
    *,
    layout_instance_digest: str,
    chunk_id: int,
    source_state_manifest_digest: str,
    context: Tensor,
    context_mask: Tensor | None,
    seq_lens: Tensor | None,
    proprio_state: Tensor,
    proprio_bindings: tuple[SelectedProprioBinding, ...],
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "context": _tensor_manifest(context),
        "context_mask": (
            None if context_mask is None else _tensor_manifest(context_mask)
        ),
        "layout_instance_digest": layout_instance_digest,
        "proprio_bindings": [
            _binding_manifest(binding) for binding in proprio_bindings
        ],
        "proprio_state": _tensor_manifest(proprio_state),
        "schema": _CONDITIONING_SCHEMA,
        "seq_lens": None if seq_lens is None else _tensor_manifest(seq_lens),
        "source_state_manifest_digest": source_state_manifest_digest,
    }


def _conditioning_digest(**kwargs: object) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(_conditioning_manifest(**kwargs))
    ).hexdigest()


def _owned_clone(value: Tensor) -> Tensor:
    clone = value.detach().clone(memory_format=torch.contiguous_format)
    clone.requires_grad_(False)
    return clone


@dataclass(frozen=True)
class CACHChunkConditioning:
    """Exact immutable identity of every non-pair chunk condition.

    Tensor fields are detached private clones.  ``verify_binding`` recomputes
    the complete digest before every numerical call, so mutation through a
    retained tensor reference fails closed.
    """

    layout_instance_digest: str
    chunk_id: int
    source_state_manifest_digest: str
    context: Tensor = field(repr=False, compare=False)
    proprio_state: Tensor = field(repr=False, compare=False)
    proprio_bindings: tuple[SelectedProprioBinding, ...]
    conditioning_digest: str
    context_mask: Tensor | None = field(default=None, repr=False, compare=False)
    seq_lens: Tensor | None = field(default=None, repr=False, compare=False)
    schema: str = _CONDITIONING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _CONDITIONING_SCHEMA:
            raise ValueError("unsupported CACH chunk-conditioning schema")
        _require_sha256(self.layout_instance_digest, "layout_instance_digest")
        _require_sha256(
            self.source_state_manifest_digest,
            "source_state_manifest_digest",
        )
        if type(self.chunk_id) is not int or self.chunk_id < 0:
            raise TypeError("chunk_id must be a non-negative plain integer")
        if not isinstance(self.context, Tensor) or self.context.ndim != 3:
            raise TypeError("context must be an exact rank-three Tensor")
        if not self.context.is_floating_point():
            raise TypeError("context must have floating dtype")
        if not bool(torch.isfinite(self.context.detach()).all()):
            raise ValueError("context contains NaN or Inf")
        batch, length = self.context.shape[:2]
        if self.context_mask is not None:
            if (
                not isinstance(self.context_mask, Tensor)
                or self.context_mask.dtype is not torch.bool
                or tuple(self.context_mask.shape) != (batch, length)
            ):
                raise TypeError("context_mask must be exact boolean [B,L]")
            if self.context_mask.device != self.context.device:
                raise ValueError("context/context_mask devices differ")
        if self.seq_lens is not None:
            if (
                not isinstance(self.seq_lens, Tensor)
                or self.seq_lens.dtype is not torch.long
                or tuple(self.seq_lens.shape) != (batch,)
            ):
                raise TypeError("seq_lens must be exact torch.long [B]")
            if self.seq_lens.device != self.context.device:
                raise ValueError("context/seq_lens devices differ")
            if bool((self.seq_lens < 0).any()) or bool((self.seq_lens > length).any()):
                raise ValueError("seq_lens lies outside the context length")
        if self.context_mask is not None and self.seq_lens is not None:
            positions = torch.arange(length, device=self.context.device).view(1, -1)
            expected_mask = positions < self.seq_lens.view(-1, 1)
            if not torch.equal(self.context_mask, expected_mask):
                raise ValueError("context_mask and seq_lens describe different tokens")
        if (
            not isinstance(self.proprio_state, Tensor)
            or self.proprio_state.dtype is not torch.float32
            or self.proprio_state.ndim != 2
            or self.proprio_state.shape[0] != batch
        ):
            raise TypeError("proprio_state must be semantic float32 [B,P]")
        if self.proprio_state.device != self.context.device:
            raise ValueError("context/proprio_state devices differ")
        if not bool(torch.isfinite(self.proprio_state.detach()).all()):
            raise ValueError("proprio_state contains NaN or Inf")
        if (
            not isinstance(self.proprio_bindings, tuple)
            or len(self.proprio_bindings) != batch
            or any(
                not isinstance(binding, SelectedProprioBinding)
                for binding in self.proprio_bindings
            )
        ):
            raise TypeError(
                "proprio_bindings must contain one SelectedProprioBinding per row"
            )
        for row, binding in enumerate(self.proprio_bindings):
            if binding.layout_instance_digest != self.layout_instance_digest:
                raise ValueError("proprio binding layout digest differs")
            if binding.chunk_id != self.chunk_id:
                raise ValueError("proprio binding chunk id differs")
            if binding.proprio_value_sha256 != canonical_proprio_row_sha256(
                self.proprio_state[row]
            ):
                raise ValueError("proprio binding value digest differs")

        object.__setattr__(self, "context", _owned_clone(self.context))
        object.__setattr__(
            self,
            "context_mask",
            None if self.context_mask is None else _owned_clone(self.context_mask),
        )
        object.__setattr__(
            self,
            "seq_lens",
            None if self.seq_lens is None else _owned_clone(self.seq_lens),
        )
        object.__setattr__(self, "proprio_state", _owned_clone(self.proprio_state))
        expected = self._recompute_digest()
        _require_sha256(self.conditioning_digest, "conditioning_digest")
        if self.conditioning_digest != expected:
            raise ValueError("conditioning_digest differs from exact tensor identity")

    @classmethod
    def create(
        cls,
        *,
        layout_instance_digest: str,
        chunk_id: int,
        source_state_manifest_digest: str,
        context: Tensor,
        context_mask: Tensor | None,
        seq_lens: Tensor | None,
        proprio_state: Tensor,
        proprio_bindings: tuple[SelectedProprioBinding, ...],
    ) -> "CACHChunkConditioning":
        values = {
            "layout_instance_digest": layout_instance_digest,
            "chunk_id": chunk_id,
            "source_state_manifest_digest": source_state_manifest_digest,
            "context": context,
            "context_mask": context_mask,
            "seq_lens": seq_lens,
            "proprio_state": proprio_state,
            "proprio_bindings": proprio_bindings,
        }
        return cls(
            **values,
            conditioning_digest=_conditioning_digest(**values),
        )

    def _recompute_digest(self) -> str:
        return _conditioning_digest(
            layout_instance_digest=self.layout_instance_digest,
            chunk_id=self.chunk_id,
            source_state_manifest_digest=self.source_state_manifest_digest,
            context=self.context,
            context_mask=self.context_mask,
            seq_lens=self.seq_lens,
            proprio_state=self.proprio_state,
            proprio_bindings=self.proprio_bindings,
        )

    def verify_binding(
        self,
        *,
        layout: ChunkActionLayout,
        chunk_id: int,
        source_state_manifest_digest: str,
    ) -> None:
        if not isinstance(layout, ChunkActionLayout):
            raise TypeError("layout must be a ChunkActionLayout")
        if (
            self.layout_instance_digest != layout.layout_instance_digest
            or self.chunk_id != chunk_id
            or self.source_state_manifest_digest != source_state_manifest_digest
            or not 0 <= chunk_id < len(layout.chunks)
        ):
            raise ValueError("chunk conditioning binding differs")
        chunk = layout.chunks[chunk_id]
        for row, binding in enumerate(self.proprio_bindings):
            binding.verify(
                layout_instance_digest=layout.layout_instance_digest,
                chunk=chunk,
                proprio_value_sha256=canonical_proprio_row_sha256(
                    self.proprio_state[row]
                ),
            )
        if self._recompute_digest() != self.conditioning_digest:
            raise RuntimeError("chunk conditioning mutated after construction")


def _operator_class(module: nn.Module) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


def _validate_production_variant_seam(
    *,
    video_backbone: nn.Module,
    core: CACHNumericalCore,
    staging_variant: CACHStagingVariant,
) -> None:
    """Bind the typed variant to the executable adapter configuration."""

    run_chunk = getattr(video_backbone, "run_chunk", None)
    pipe = getattr(video_backbone, "_pipe", None)
    config = getattr(pipe, "config", None)
    get_config = getattr(config, "get", None)
    if not callable(run_chunk) or not callable(get_config):
        raise RuntimeError("production stager lacks an executable adapter config")
    action_seam_enabled = get_config("use_delta_pose_additive")
    if type(action_seam_enabled) is not bool:
        raise RuntimeError("action seam enablement must be an explicit boolean")
    if staging_variant is CACHStagingVariant.CACH_A:
        if action_seam_enabled is not True:
            raise RuntimeError("CACH-A requires the enabled additive action seam")
        if get_config("delta_pose_additive_dim") != core.action_dim:
            raise RuntimeError("CACH-A action seam width differs from the core")
        if getattr(
            getattr(video_backbone, "_dit", None),
            "_cach_delta_pose_output_zero_verified",
            False,
        ) is not True:
            raise RuntimeError("CACH-A lacks exact-zero action adapter attestation")
    elif staging_variant is CACHStagingVariant.REF_GDN_CORRECTED:
        if action_seam_enabled is not False:
            raise RuntimeError(
                "REF-GDN-CORRECTED requires an explicit action-seam identity bypass"
            )
    else:  # pragma: no cover - constructor requires the closed enum.
        raise AssertionError("unreachable CACH staging variant")


def build_layer_registry_from_backbone(
    video_backbone: nn.Module,
    *,
    synthetic_test_only: bool = False,
) -> LayerRegistry:
    """Derive the registry only from the executable cached topology."""

    if not isinstance(video_backbone, nn.Module):
        raise TypeError("video_backbone must be an nn.Module")
    if type(synthetic_test_only) is not bool:
        raise TypeError("synthetic_test_only must be a plain boolean")
    if getattr(video_backbone, "cached_streaming_enabled", False) is not True:
        raise RuntimeError("registry requires an already-enabled streaming backbone")
    dit = getattr(video_backbone, "_dit", None)
    blocks = getattr(dit, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("video backbone lacks executable streaming blocks")

    from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
    from diffusion.model.nets.sana_gdn_blocks import CachedChunkCausalGDN

    layers: list[LayerSpec] = []
    for layer_index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        feed_forward = getattr(block, "mlp", None)
        if type(attention) is not CachedChunkCausalGDN:
            raise RuntimeError(
                "CACH registry rejects softmax, camera, and non-CachedChunkCausalGDN attention"
            )
        if type(feed_forward) is not CachedGLUMBConvTemp:
            raise RuntimeError(
                "CACH registry requires exact CachedGLUMBConvTemp on every layer"
            )
        if any(
            getattr(attention, name, None) is not None
            for name in (
                "conv_k_cam",
                "camera_qkv",
                "camera_proj",
                "qkv_cam",
                "proj_cam",
            )
        ):
            raise RuntimeError("CACH-A registry forbids camera attention state")
        layers.append(
            LayerSpec(
                layer_index=layer_index,
                kind=LayerKind.GDN_FULL_HISTORY,
                operator_class=_operator_class(attention),
                camera_enabled=False,
                main_shortconv_enabled=getattr(attention, "conv_k", None)
                is not None,
                ffn_tconv_enabled=True,
            )
        )
    registry = LayerRegistry(layers=tuple(layers))
    if synthetic_test_only:
        if len(registry.layers) != 1:
            raise RuntimeError("synthetic CACH mini registry requires exact depth 1")
    else:
        validate_cach_a_registry(registry)
    return registry


class CACHTeacherForcingPairedStager:
    """One-chunk offline teacher-forcing staging callback.

    The object intentionally has no manager/request-list argument or field.
    Prior action bytes come only from the typed manager-created history view in
    the call context and are used for identity/shape validation; the numerical
    cross-chunk summary is the paired GDN cache in ``previous_scratch``.
    """

    def __init__(
        self,
        *,
        core: CACHNumericalCore,
        video_backbone: nn.Module,
        action_backbone: nn.Module,
        proprio_context_encoder: nn.Module,
        registry: LayerRegistry,
        layout: ChunkActionLayout,
        conditioning: CACHChunkConditioning,
        staging_variant: CACHStagingVariant,
        synthetic_test_only: bool = False,
    ) -> None:
        if not isinstance(core, CACHNumericalCore):
            raise TypeError("core must be CACHNumericalCore")
        if not isinstance(video_backbone, nn.Module):
            raise TypeError("video_backbone must be an nn.Module")
        if not isinstance(action_backbone, nn.Module):
            raise TypeError("action_backbone must be an nn.Module")
        if not isinstance(proprio_context_encoder, nn.Module):
            raise TypeError("proprio_context_encoder must be an nn.Module")
        if not isinstance(registry, LayerRegistry):
            raise TypeError("registry must be a LayerRegistry")
        if not isinstance(layout, ChunkActionLayout):
            raise TypeError("layout must be a ChunkActionLayout")
        if not isinstance(conditioning, CACHChunkConditioning):
            raise TypeError("conditioning must be CACHChunkConditioning")
        if not isinstance(staging_variant, CACHStagingVariant):
            raise TypeError("staging_variant must be CACHStagingVariant")
        if type(synthetic_test_only) is not bool:
            raise TypeError("synthetic_test_only must be a plain boolean")
        if layout.synthetic_test_only is not synthetic_test_only:
            raise RuntimeError("synthetic layout requires an explicit synthetic stager")
        if synthetic_test_only:
            if len(registry.layers) != 1:
                raise RuntimeError("synthetic stager requires an exact depth-1 registry")
        else:
            validate_cach_a_registry(registry)
            _validate_production_variant_seam(
                video_backbone=video_backbone,
                core=core,
                staging_variant=staging_variant,
            )
        conditioning.verify_binding(
            layout=layout,
            chunk_id=conditioning.chunk_id,
            source_state_manifest_digest=conditioning.source_state_manifest_digest,
        )
        self._core = core
        self._video_backbone = video_backbone
        self._action_backbone = action_backbone
        self._proprio_context_encoder = proprio_context_encoder
        self._registry = registry
        self._layout = layout
        self._conditioning = conditioning
        self._staging_variant = staging_variant

    def __call__(
        self, context: Stage2BPairedStagingContext
    ) -> tuple[StagedLayerPayload, ...]:
        if not isinstance(context, Stage2BPairedStagingContext):
            raise TypeError("stager requires a manager-created Stage2B context")
        if context.video_timestep != 0 or context.action_timestep != 0:
            raise ValueError("paired staging context must be exact t=0")
        if context.staging_variant is not self._staging_variant:
            raise ValueError("manager and stager variants differ")
        chunk_id = context.content_time.chunk_id
        if chunk_id != self._conditioning.chunk_id:
            raise ValueError("stager conditioning belongs to another chunk")
        if context.conditioning_digest != self._conditioning.conditioning_digest:
            raise ValueError("manager context did not bind the exact conditioning digest")
        chunk = self._layout.chunks[chunk_id]
        context.content_time.verify_layout_binding(layout=self._layout, chunk=chunk)
        if (
            context.previous_state_manifest_digest
            != context.previous_scratch.source_state_manifest_digest
        ):
            raise RuntimeError("paired context scratch source differs")
        self._conditioning.verify_binding(
            layout=self._layout,
            chunk_id=chunk_id,
            source_state_manifest_digest=context.previous_state_manifest_digest,
        )

        history = context.previous_committed_action_history
        if not isinstance(history, CommittedActionHistoryView):
            raise TypeError("paired context lacks typed committed-action history")
        if (
            history.batch_size != context.actions.shape[0]
            or history.action_dim != context.actions.shape[2]
            or history.dtype != context.actions.dtype
            or history.device != context.actions.device
            or history.cursor != chunk.action_start
            or history.action_start != 0
            or history.action_end_exclusive != chunk.action_start
            or history.chunk_ids != tuple(range(chunk_id))
        ):
            raise RuntimeError("committed-action history differs from chunk ownership")
        _require_sha256(history.history_manifest_digest, "history_manifest_digest")
        committed_actions = history.clone_actions()
        if (
            not isinstance(committed_actions, Tensor)
            or tuple(committed_actions.shape)
            != (context.actions.shape[0], chunk.action_start, context.actions.shape[2])
            or not committed_actions.is_contiguous()
            or committed_actions.requires_grad
            or not bool(torch.isfinite(committed_actions).all())
        ):
            raise RuntimeError("committed-action history clone is not exact")

        video_timestep = context.video.new_zeros((context.video.shape[0],))
        action_timestep = context.actions.new_zeros((context.actions.shape[0],))
        result = self._core.run_paired_t0(
            video_backbone=self._video_backbone,
            action_backbone=self._action_backbone,
            proprio_context_encoder=self._proprio_context_encoder,
            registry=self._registry,
            action_layout=self._layout,
            chunk_id=chunk_id,
            chunk_latents=context.video,
            frame_valid_mask=context.frame_valid_mask,
            noisy_actions=context.actions,
            action_valid_mask=context.action_valid_mask,
            committed_actions=committed_actions,
            committed_history_digest=history.history_manifest_digest,
            expected_committed_actions_digest=tensor_digest(committed_actions),
            cache_scratch=context.previous_scratch,
            conditioning=self._conditioning,
            staging_variant=self._staging_variant,
            video_timestep=video_timestep,
            action_timestep=action_timestep,
        )
        if result.staged_layer_payloads is None:
            raise RuntimeError("paired numerical core returned no typed payloads")
        return result.staged_layer_payloads


__all__ = [
    "CACHChunkConditioning",
    "CACHTeacherForcingPairedStager",
    "build_layer_registry_from_backbone",
]
