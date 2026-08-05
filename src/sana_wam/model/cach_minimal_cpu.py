"""CPU-only synthetic mini pair for REF-GDN-CORRECTED and CACH-A.

This module is deliberately isolated from the production architecture
dispatcher.  It provides a tiny-dimension, 20-layer, pure-PyTorch reference
implementation for checking the CACH action-conditioning seam and temporal
state rules without importing the vendor CUDA/Triton operator.

It is *not* the complete 2B model, a checkpoint loader, a training surface, an
evaluator, or Global Stage-3 admission evidence.  Inputs must be synthetic,
CPU, FP32 tensors bound to a synthetic :class:`ChunkActionLayout`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from itertools import count
import math
from typing import Final, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sana_wam.cach.action_conditioning import (
    reduce_end_of_bin_action_condition,
)
from sana_wam.cach.staging_variant import (
    CACHStagingVariant,
    require_cach_staging_variant,
)
from sana_wam.model.action_chunk_layout import (
    ChunkActionLayout,
    ChunkLayout,
    SelectedProprioBinding,
    canonical_proprio_row_sha256,
    canonical_sha256,
)
from sana_wam.model.video_backbone.sana.hybrid_cache import (
    LayerKind,
    LayerRegistry,
    LayerSpec,
    tensor_digest,
    validate_cach_a_registry,
)


_OPERATOR_CONTRACT: Final = "pure_torch_chunk_causal_gdn_proxy_v1"
_ARCHITECTURE_REVISION: Final = "cach_cpu_synthetic_pair_v1"
_INSTANCE_COUNTER = count()
_DISABLED_COMPONENTS: Final = (
    "afcc",
    "anchors",
    "attnres",
    "best_of_n",
    "dagger",
    "phase6_reference",
    "recovery_policy",
    "rerank",
    "self_forcing",
    "temporal_ensemble",
)


class MinimalCACHContractError(RuntimeError):
    """Fail-closed error for the synthetic mini execution boundary."""


def _require_plain_int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a plain integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{name} must be >= {minimum}{suffix}")
    return value


def _require_canonical_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _named_seed(base_seed: int, initializer_role: str, parameter_name: str) -> int:
    material = (
        f"{_ARCHITECTURE_REVISION}:{initializer_role}:{base_seed}:{parameter_name}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _tensor_is_exact_zero(value: Tensor) -> bool:
    return not bool(value.detach().count_nonzero())


def _clone_tensor(value: Tensor) -> Tensor:
    clone = value.detach().clone(memory_format=torch.contiguous_format)
    clone.requires_grad_(False)
    return clone


def _require_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have exact shape {shape}")
    if value.device.type != "cpu":
        raise MinimalCACHContractError(f"{name} must remain on CPU")
    if value.dtype is not dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.layout is not torch.strided:
        raise TypeError(f"{name} must use strided layout")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return value


@dataclass(frozen=True)
class MinimalCACHSpec:
    """Fixed CPU synthetic-mini architecture contract.

    Depth and action width are intentionally fixed to the proposed CACH-A
    topology.  Latent/hidden dimensions stay small so this type cannot be
    silently repurposed as the full model.
    """

    latent_dim: int = 6
    hidden_dim: int = 8
    action_dim: int = 20
    depth: int = 20
    chunk_size: int = 2
    shared_init_seed: int = 20260802
    operator_init_seed: int = 20260803
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    synthetic_test_only: bool = True
    architecture_revision: str = _ARCHITECTURE_REVISION

    def __post_init__(self) -> None:
        _require_plain_int(self.latent_dim, "latent_dim", minimum=2, maximum=32)
        _require_plain_int(self.hidden_dim, "hidden_dim", minimum=4, maximum=32)
        _require_plain_int(self.action_dim, "action_dim", minimum=20, maximum=20)
        _require_plain_int(self.depth, "depth", minimum=20, maximum=20)
        _require_plain_int(self.chunk_size, "chunk_size", minimum=1, maximum=8)
        _require_plain_int(
            self.shared_init_seed,
            "shared_init_seed",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        _require_plain_int(
            self.operator_init_seed,
            "operator_init_seed",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        if self.operator_init_seed == self.shared_init_seed:
            raise ValueError("operator_init_seed must differ from shared_init_seed")
        if self.device != "cpu":
            raise MinimalCACHContractError("synthetic mini supports CPU only")
        if self.dtype is not torch.float32:
            raise MinimalCACHContractError("synthetic mini supports FP32 only")
        if self.synthetic_test_only is not True:
            raise MinimalCACHContractError(
                "synthetic_test_only must be the literal True"
            )
        if self.architecture_revision != _ARCHITECTURE_REVISION:
            raise ValueError("unsupported synthetic mini architecture revision")

    def to_manifest(self) -> dict[str, object]:
        return {
            "action_dim": self.action_dim,
            "architecture_revision": self.architecture_revision,
            "chunk_size": self.chunk_size,
            "depth": self.depth,
            "device": self.device,
            "dtype": "float32",
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "operator_contract": _OPERATOR_CONTRACT,
            "operator_init_seed": self.operator_init_seed,
            "shared_init_seed": self.shared_init_seed,
            "synthetic_test_only": self.synthetic_test_only,
        }


@dataclass(frozen=True)
class MinimalChunkBatch:
    """One synthetic fixed-capacity video/action chunk.

    Validation is performed by the layout-bound architecture, because exact
    tensor shapes depend on the selected :class:`ChunkLayout`.
    """

    layout_instance_digest: str
    chunk_id: int
    video_latents: Tensor = field(repr=False, compare=False)
    frame_valid_mask: Tensor = field(repr=False, compare=False)
    noisy_actions: Tensor = field(repr=False, compare=False)
    action_valid_mask: Tensor = field(repr=False, compare=False)
    proprio_state: Tensor = field(repr=False, compare=False)
    proprio_bindings: tuple[SelectedProprioBinding, ...]
    video_timestep: Tensor = field(repr=False, compare=False)
    action_timestep: Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_canonical_string(
            self.layout_instance_digest,
            "layout_instance_digest",
        )
        _require_plain_int(self.chunk_id, "chunk_id", minimum=0)
        if not isinstance(self.proprio_bindings, tuple) or any(
            not isinstance(binding, SelectedProprioBinding)
            for binding in self.proprio_bindings
        ):
            raise TypeError(
                "proprio_bindings must be a tuple of SelectedProprioBinding"
            )


@dataclass(frozen=True)
class MinimalLayerState:
    """Pure-Torch full-history GDN proxy state for one layer."""

    main_s_kv: Tensor = field(repr=False, compare=False)
    main_s_z: Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.main_s_kv, Tensor) or self.main_s_kv.ndim != 3:
            raise TypeError("main_s_kv must be a rank-three Tensor")
        if not isinstance(self.main_s_z, Tensor) or self.main_s_z.ndim != 2:
            raise TypeError("main_s_z must be a rank-two Tensor")
        if self.main_s_kv.shape[:2] != self.main_s_z.shape:
            raise ValueError("main_s_kv/main_s_z leading axes differ")

    def detached_clone(self) -> "MinimalLayerState":
        return type(self)(
            main_s_kv=_clone_tensor(self.main_s_kv),
            main_s_z=_clone_tensor(self.main_s_z),
        )

    def to_digest_payload(self) -> dict[str, str]:
        return {
            "main_s_kv": tensor_digest(self.main_s_kv),
            "main_s_z": tensor_digest(self.main_s_z),
        }


@dataclass(frozen=True)
class MinimalGDNState:
    """Immutable-value in-memory state; it is never a durable receipt."""

    model_instance_id: str
    architecture_instance_token: str
    parameter_manifest_digest: str
    staging_variant: CACHStagingVariant
    layout_instance_digest: str
    batch_size: int
    revision: int
    next_chunk_id: int
    action_cursor: int
    committed_actions: Tensor = field(repr=False, compare=False)
    layers: tuple[MinimalLayerState, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_canonical_string(self.model_instance_id, "model_instance_id")
        _require_canonical_string(
            self.architecture_instance_token,
            "architecture_instance_token",
        )
        _require_canonical_string(
            self.parameter_manifest_digest,
            "parameter_manifest_digest",
        )
        require_cach_staging_variant(self.staging_variant)
        _require_canonical_string(
            self.layout_instance_digest,
            "layout_instance_digest",
        )
        _require_plain_int(self.batch_size, "batch_size", minimum=1, maximum=8)
        _require_plain_int(self.revision, "revision", minimum=0)
        _require_plain_int(self.next_chunk_id, "next_chunk_id", minimum=0)
        _require_plain_int(self.action_cursor, "action_cursor", minimum=0)
        if not isinstance(self.committed_actions, Tensor):
            raise TypeError("committed_actions must be a Tensor")
        if not isinstance(self.layers, tuple) or any(
            not isinstance(layer, MinimalLayerState) for layer in self.layers
        ):
            raise TypeError("layers must be a tuple of MinimalLayerState")

    @property
    def state_digest(self) -> str:
        return canonical_sha256(
            {
                "action_cursor": self.action_cursor,
                "batch_size": self.batch_size,
                "committed_actions": tensor_digest(self.committed_actions),
                "layers": tuple(layer.to_digest_payload() for layer in self.layers),
                "layout_instance_digest": self.layout_instance_digest,
                "model_instance_id": self.model_instance_id,
                "architecture_instance_token": self.architecture_instance_token,
                "next_chunk_id": self.next_chunk_id,
                "parameter_manifest_digest": self.parameter_manifest_digest,
                "revision": self.revision,
                "schema": "cach.minimal_gdn_state.v1",
                "staging_variant": self.staging_variant.value,
            }
        )

    def detached_clone(self) -> "MinimalGDNState":
        return type(self)(
            model_instance_id=self.model_instance_id,
            architecture_instance_token=self.architecture_instance_token,
            parameter_manifest_digest=self.parameter_manifest_digest,
            staging_variant=self.staging_variant,
            layout_instance_digest=self.layout_instance_digest,
            batch_size=self.batch_size,
            revision=self.revision,
            next_chunk_id=self.next_chunk_id,
            action_cursor=self.action_cursor,
            committed_actions=_clone_tensor(self.committed_actions),
            layers=tuple(layer.detached_clone() for layer in self.layers),
        )


@dataclass(frozen=True)
class MinimalChunkOutput:
    """Read-only numerical output for one complete video/action chunk."""

    chunk_id: int
    video_prediction: Tensor = field(repr=False, compare=False)
    action_prediction: Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_plain_int(self.chunk_id, "chunk_id", minimum=0)
        for name in ("video_prediction", "action_prediction"):
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if not bool(torch.isfinite(value.detach()).all()):
                raise ValueError(f"{name} contains NaN or Inf")

    @property
    def output_digest(self) -> str:
        return canonical_sha256(
            {
                "action_prediction": tensor_digest(self.action_prediction),
                "chunk_id": self.chunk_id,
                "schema": "cach.minimal_chunk_output.v1",
                "video_prediction": tensor_digest(self.video_prediction),
            }
        )


@dataclass(frozen=True)
class MinimalChunkProposal:
    """Commit-eligible proposal produced only by the paired ``t=0`` path."""

    owner_id: str
    owner_epoch: int
    model_instance_id: str
    architecture_instance_token: str
    parameter_manifest_digest: str
    staging_variant: CACHStagingVariant
    layout_instance_digest: str
    chunk_id: int
    source_revision: int
    source_state_digest: str
    output: MinimalChunkOutput = field(repr=False, compare=False)
    proposed_state: MinimalGDNState = field(repr=False, compare=False)
    proposal_id: str

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        owner_epoch: int,
        source_state: MinimalGDNState,
        output: MinimalChunkOutput,
        proposed_state: MinimalGDNState,
    ) -> "MinimalChunkProposal":
        payload = {
            "chunk_id": output.chunk_id,
            "layout_instance_digest": source_state.layout_instance_digest,
            "model_instance_id": source_state.model_instance_id,
            "architecture_instance_token": (source_state.architecture_instance_token),
            "output_digest": output.output_digest,
            "owner_epoch": owner_epoch,
            "owner_id": owner_id,
            "proposed_state_digest": proposed_state.state_digest,
            "schema": "cach.minimal_chunk_proposal.v1",
            "parameter_manifest_digest": source_state.parameter_manifest_digest,
            "source_revision": source_state.revision,
            "source_state_digest": source_state.state_digest,
            "staging_variant": source_state.staging_variant.value,
        }
        return cls(
            owner_id=owner_id,
            owner_epoch=owner_epoch,
            model_instance_id=source_state.model_instance_id,
            architecture_instance_token=source_state.architecture_instance_token,
            parameter_manifest_digest=source_state.parameter_manifest_digest,
            staging_variant=source_state.staging_variant,
            layout_instance_digest=source_state.layout_instance_digest,
            chunk_id=output.chunk_id,
            source_revision=source_state.revision,
            source_state_digest=source_state.state_digest,
            output=output,
            proposed_state=proposed_state,
            proposal_id=canonical_sha256(payload),
        )

    def __post_init__(self) -> None:
        _require_canonical_string(self.owner_id, "owner_id")
        _require_plain_int(self.owner_epoch, "owner_epoch", minimum=0)
        _require_canonical_string(self.model_instance_id, "model_instance_id")
        _require_canonical_string(
            self.architecture_instance_token,
            "architecture_instance_token",
        )
        _require_canonical_string(
            self.parameter_manifest_digest,
            "parameter_manifest_digest",
        )
        require_cach_staging_variant(self.staging_variant)
        _require_canonical_string(
            self.layout_instance_digest,
            "layout_instance_digest",
        )
        _require_plain_int(self.chunk_id, "chunk_id", minimum=0)
        _require_plain_int(self.source_revision, "source_revision", minimum=0)
        _require_canonical_string(self.source_state_digest, "source_state_digest")
        _require_canonical_string(self.proposal_id, "proposal_id")
        if not isinstance(self.output, MinimalChunkOutput):
            raise TypeError("output must be a MinimalChunkOutput")
        if not isinstance(self.proposed_state, MinimalGDNState):
            raise TypeError("proposed_state must be a MinimalGDNState")

    def assert_integrity(self) -> None:
        payload = {
            "chunk_id": self.chunk_id,
            "layout_instance_digest": self.layout_instance_digest,
            "model_instance_id": self.model_instance_id,
            "architecture_instance_token": self.architecture_instance_token,
            "output_digest": self.output.output_digest,
            "owner_epoch": self.owner_epoch,
            "owner_id": self.owner_id,
            "proposed_state_digest": self.proposed_state.state_digest,
            "schema": "cach.minimal_chunk_proposal.v1",
            "parameter_manifest_digest": self.parameter_manifest_digest,
            "source_revision": self.source_revision,
            "source_state_digest": self.source_state_digest,
            "staging_variant": self.staging_variant.value,
        }
        if canonical_sha256(payload) != self.proposal_id:
            raise MinimalCACHContractError("proposal was mutated after staging")


@dataclass(frozen=True)
class MinimalSequenceResult:
    """Read-only result of the independent layer-major sequence path."""

    outputs: tuple[MinimalChunkOutput, ...]
    final_state: MinimalGDNState = field(repr=False, compare=False)

    @property
    def video_prediction(self) -> Tensor:
        return torch.cat(
            tuple(output.video_prediction for output in self.outputs),
            dim=1,
        )

    @property
    def action_prediction(self) -> Tensor:
        return torch.cat(
            tuple(output.action_prediction for output in self.outputs),
            dim=1,
        )


class PureTorchChunkCausalGDNProxy(nn.Module):
    """Tiny pure-Torch gated-delta recurrence with ``S_kv``/``S_z`` state.

    The scan mirrors the production operator's state topology and
    chunk-causal directionality (stateful forward scan plus chunk-local reverse
    scan).  It is intentionally labelled a proxy: it does not claim numerical
    parity with the fused vendor kernel.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_dim: int,
        staging_variant: CACHStagingVariant,
    ) -> None:
        super().__init__()
        variant = require_cach_staging_variant(staging_variant)
        self.hidden_dim = hidden_dim
        self.staging_variant = variant
        linear_kwargs = {"device": "cpu", "dtype": torch.float32}
        self.q_projection = nn.Linear(
            hidden_dim, hidden_dim, bias=False, **linear_kwargs
        )
        self.k_projection = nn.Linear(
            hidden_dim, hidden_dim, bias=False, **linear_kwargs
        )
        self.v_projection = nn.Linear(
            hidden_dim, hidden_dim, bias=False, **linear_kwargs
        )
        self.beta_projection = nn.Linear(hidden_dim, 1, bias=True, **linear_kwargs)
        self.decay_projection = nn.Linear(hidden_dim, 1, bias=True, **linear_kwargs)
        self.attention_output_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=True,
            **linear_kwargs,
        )
        self.ffn_input_projection = nn.Linear(
            hidden_dim,
            hidden_dim * 2,
            bias=True,
            **linear_kwargs,
        )
        self.ffn_output_projection = nn.Linear(
            hidden_dim * 2,
            hidden_dim,
            bias=True,
            **linear_kwargs,
        )
        if variant is CACHStagingVariant.CACH_A:
            self.action_output_projection: nn.Linear | None = nn.Linear(
                action_dim,
                hidden_dim,
                bias=True,
                **linear_kwargs,
            )
        else:
            self.action_output_projection = None

    @staticmethod
    def _rms_norm(value: Tensor) -> Tensor:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)

    def _delta_step(
        self,
        hidden: Tensor,
        state: MinimalLayerState,
        valid: Tensor,
    ) -> tuple[Tensor, MinimalLayerState]:
        normalized = self._rms_norm(hidden)
        query = F.silu(self.q_projection(normalized))
        key = F.silu(self.k_projection(normalized))
        key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
        value = self.v_projection(normalized)
        beta = torch.sigmoid(self.beta_projection(normalized))
        decay = 0.5 + 0.49 * torch.sigmoid(self.decay_projection(normalized))

        prior = torch.einsum("bi,bij->bj", key, state.main_s_kv)
        delta = beta * (value - prior)
        next_s_kv = decay.unsqueeze(-1) * state.main_s_kv + torch.einsum(
            "bi,bj->bij",
            key,
            delta,
        )
        next_s_z = decay * state.main_s_z + beta * key
        mask_matrix = valid.view(-1, 1, 1)
        mask_vector = valid.view(-1, 1)
        next_s_kv = torch.where(mask_matrix, next_s_kv, state.main_s_kv)
        next_s_z = torch.where(mask_vector, next_s_z, state.main_s_z)

        numerator = torch.einsum("bi,bij->bj", query, next_s_kv)
        denominator = 1.0 + torch.einsum("bi,bi->b", query, next_s_z).abs()
        output = numerator / denominator.unsqueeze(-1)
        output = torch.where(mask_vector, output, torch.zeros_like(output))
        return output, MinimalLayerState(next_s_kv, next_s_z)

    def _scan_direction(
        self,
        hidden: Tensor,
        initial_state: MinimalLayerState,
        frame_valid_mask: Tensor,
        *,
        reverse: bool,
    ) -> tuple[Tensor, MinimalLayerState]:
        indices: Iterable[int]
        if reverse:
            indices = range(hidden.shape[1] - 1, -1, -1)
        else:
            indices = range(hidden.shape[1])
        outputs: list[Tensor | None] = [None] * hidden.shape[1]
        state = initial_state
        for index in indices:
            output, state = self._delta_step(
                hidden[:, index],
                state,
                frame_valid_mask[:, index],
            )
            outputs[index] = output
        if any(value is None for value in outputs):
            raise AssertionError("GDN proxy scan left an uncomputed token")
        return torch.stack(tuple(outputs), dim=1), state  # type: ignore[arg-type]

    def scan_chunk(
        self,
        hidden: Tensor,
        state: MinimalLayerState,
        *,
        frame_valid_mask: Tensor,
        action_embedding: Tensor | None,
    ) -> tuple[Tensor, MinimalLayerState]:
        if self.staging_variant is CACHStagingVariant.CACH_A:
            if action_embedding is None:
                raise MinimalCACHContractError(
                    "CACH-A requires explicit action conditioning"
                )
        elif action_embedding is not None:
            raise MinimalCACHContractError(
                "REF-GDN-CORRECTED requires the identity action bypass"
            )

        forward_output, next_state = self._scan_direction(
            hidden,
            state,
            frame_valid_mask,
            reverse=False,
        )
        zero_state = MinimalLayerState(
            main_s_kv=torch.zeros_like(state.main_s_kv),
            main_s_z=torch.zeros_like(state.main_s_z),
        )
        reverse_output, _ = self._scan_direction(
            hidden,
            zero_state,
            frame_valid_mask,
            reverse=True,
        )
        mask = frame_valid_mask.unsqueeze(-1)
        attention = 0.5 * (forward_output + reverse_output)
        post_attention = hidden + self.attention_output_projection(attention)
        if self.action_output_projection is not None:
            assert action_embedding is not None
            # This is the only candidate delta and matches the reviewed
            # post-recurrence, pre-FFN vendor insertion point.
            post_attention = post_attention + self.action_output_projection(
                action_embedding
            )
        post_attention = torch.where(mask, post_attention, torch.zeros_like(hidden))
        ffn = self.ffn_output_projection(
            F.silu(self.ffn_input_projection(self._rms_norm(post_attention)))
        )
        output = torch.where(
            mask,
            post_attention + ffn,
            torch.zeros_like(post_attention),
        )
        return output, next_state


class MinimalCorrectedGDNArchitecture(nn.Module):
    """One arm of the CPU synthetic REF/CACH mini pair."""

    def __init__(
        self,
        *,
        spec: MinimalCACHSpec,
        layout: ChunkActionLayout,
        staging_variant: CACHStagingVariant,
    ) -> None:
        super().__init__()
        if not isinstance(spec, MinimalCACHSpec):
            raise TypeError("spec must be a MinimalCACHSpec")
        if not isinstance(layout, ChunkActionLayout):
            raise TypeError("layout must be a ChunkActionLayout")
        if layout.synthetic_test_only is not True:
            raise MinimalCACHContractError(
                "minimal architecture accepts synthetic layouts only"
            )
        if layout.frame_chunk_size != spec.chunk_size:
            raise ValueError("layout frame_chunk_size differs from spec.chunk_size")
        if layout.action_tokens_per_non_anchor_latent is None:
            raise MinimalCACHContractError("synthetic mini requires proven equal rate")
        if max(chunk.action_slot_capacity for chunk in layout.chunks) > 32:
            raise MinimalCACHContractError("synthetic action capacity is too large")
        self.spec = spec
        self.layout = layout
        self.staging_variant = require_cach_staging_variant(staging_variant)
        self.layer_registry = LayerRegistry(
            tuple(
                LayerSpec(
                    layer_index=index,
                    kind=LayerKind.GDN_FULL_HISTORY,
                    operator_class=(
                        "sana_wam.model.cach_minimal_cpu.PureTorchChunkCausalGDNProxy"
                    ),
                    camera_enabled=False,
                    main_shortconv_enabled=False,
                    ffn_tconv_enabled=False,
                )
                for index in range(spec.depth)
            )
        )
        validate_cach_a_registry(self.layer_registry)

        linear_kwargs = {"device": "cpu", "dtype": spec.dtype}
        self.video_input_projection = nn.Linear(
            spec.latent_dim,
            spec.hidden_dim,
            bias=True,
            **linear_kwargs,
        )
        self.video_timestep_projection = nn.Linear(
            1,
            spec.hidden_dim,
            bias=False,
            **linear_kwargs,
        )
        self.blocks = nn.ModuleList(
            [
                PureTorchChunkCausalGDNProxy(
                    hidden_dim=spec.hidden_dim,
                    action_dim=spec.hidden_dim,
                    staging_variant=self.staging_variant,
                )
                for _ in range(spec.depth)
            ]
        )
        self.video_output_projection = nn.Linear(
            spec.hidden_dim,
            spec.latent_dim,
            bias=True,
            **linear_kwargs,
        )

        # Shared action/proprio path.  It remains present in the corrected
        # reference; only action-to-video conditioning is bypassed there.
        self.action_input_projection = nn.Linear(
            spec.action_dim,
            spec.hidden_dim,
            bias=True,
            **linear_kwargs,
        )
        self.action_video_projection = nn.Linear(
            spec.hidden_dim,
            spec.hidden_dim,
            bias=False,
            **linear_kwargs,
        )
        self.action_proprio_projection = nn.Linear(
            spec.action_dim,
            spec.hidden_dim,
            bias=False,
            **linear_kwargs,
        )
        self.action_position_projection = nn.Linear(
            2,
            spec.hidden_dim,
            bias=False,
            **linear_kwargs,
        )
        self.action_timestep_projection = nn.Linear(
            1,
            spec.hidden_dim,
            bias=False,
            **linear_kwargs,
        )
        self.action_output_projection = nn.Linear(
            spec.hidden_dim,
            spec.action_dim,
            bias=True,
            **linear_kwargs,
        )

        if self.staging_variant is CACHStagingVariant.CACH_A:
            self.no_action_slot = nn.Parameter(
                torch.zeros(spec.action_dim, dtype=spec.dtype, device="cpu")
            )
            self.action_conditioner: nn.Module | None = nn.Sequential(
                nn.Linear(
                    spec.action_dim,
                    spec.hidden_dim,
                    bias=True,
                    **linear_kwargs,
                ),
                nn.SiLU(),
                nn.Linear(
                    spec.hidden_dim,
                    spec.hidden_dim,
                    bias=True,
                    **linear_kwargs,
                ),
            )
        else:
            self.register_parameter("no_action_slot", None)
            self.action_conditioner = None

        self._initialize_named_parameters()
        self._diagnostic_nonzero_seam = False
        instance_index = next(_INSTANCE_COUNTER)
        self.architecture_instance_token = hashlib.sha256(
            (
                f"{_ARCHITECTURE_REVISION}:process-local-instance:{instance_index}"
            ).encode()
        ).hexdigest()
        self._registered_parameter_manifest_digest = (
            self._current_parameter_manifest_digest()
        )
        self._registered_parameter_storage_signature = (
            self._current_parameter_storage_signature()
        )
        self._refresh_model_instance_id()
        self.eval()
        self._assert_cpu_model()

    @property
    def disabled_components(self) -> tuple[str, ...]:
        return _DISABLED_COMPONENTS

    def graph_manifest(self) -> dict[str, object]:
        return {
            "architecture_revision": _ARCHITECTURE_REVISION,
            "complete_2b": False,
            "depth": self.spec.depth,
            "device": "cpu",
            "disabled_components": self.disabled_components,
            "dtype": "float32",
            "full_model_admission": False,
            "layer_registry_digest": self.layer_registry.manifest_digest,
            "operator_contract": _OPERATOR_CONTRACT,
            "parameter_manifest_digest": self.parameter_manifest_digest,
            "scientific_result": False,
            "staging_variant": self.staging_variant.value,
            "synthetic_nonzero_diagnostic": self._diagnostic_nonzero_seam,
            "synthetic_test_only": True,
            "vendor_gdn_numerical_parity": False,
        }

    def _initialize_named_parameters(self) -> None:
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name == "no_action_slot" or ".action_output_projection." in name:
                    parameter.zero_()
                    continue
                if name.endswith(".bias"):
                    parameter.zero_()
                    continue
                operator_specific = name.startswith("action_conditioner.")
                initializer_role = (
                    "candidate_operator_named_uniform_v1"
                    if operator_specific
                    else "shared_named_uniform_v1"
                )
                base_seed = (
                    self.spec.operator_init_seed
                    if operator_specific
                    else self.spec.shared_init_seed
                )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(_named_seed(base_seed, initializer_role, name))
                fan_in = parameter.shape[-1] if parameter.ndim >= 2 else 1
                bound = 0.35 / math.sqrt(float(fan_in))
                parameter.uniform_(-bound, bound, generator=generator)

    def _current_parameter_manifest_digest(self) -> str:
        return canonical_sha256(
            {
                "buffers": tuple(
                    {
                        "digest": tensor_digest(buffer),
                        "dtype": str(buffer.dtype),
                        "name": name,
                        "shape": tuple(buffer.shape),
                    }
                    for name, buffer in self.named_buffers()
                ),
                "parameters": tuple(
                    {
                        "digest": tensor_digest(parameter),
                        "dtype": str(parameter.dtype),
                        "name": name,
                        "requires_grad": parameter.requires_grad,
                        "shape": tuple(parameter.shape),
                    }
                    for name, parameter in self.named_parameters()
                ),
                "schema": "cach.minimal_parameter_manifest.v1",
            }
        )

    @property
    def parameter_manifest_digest(self) -> str:
        return self._registered_parameter_manifest_digest

    def _refresh_model_instance_id(self) -> None:
        self.model_instance_id = canonical_sha256(
            {
                "architecture": self.spec.to_manifest(),
                "diagnostic_nonzero_seam": self._diagnostic_nonzero_seam,
                "layout_instance_digest": self.layout.layout_instance_digest,
                "parameter_manifest_digest": self.parameter_manifest_digest,
                "staging_variant": self.staging_variant.value,
            }
        )

    def _current_parameter_storage_signature(
        self,
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                name,
                parameter.untyped_storage().data_ptr(),
                parameter.untyped_storage().nbytes(),
                parameter.storage_offset(),
                tuple(parameter.stride()),
            )
            for name, parameter in self.named_parameters()
        )

    def install_synthetic_nonzero_action_seam_for_tests(self) -> None:
        """Install the one registered nonzero wiring diagnostic.

        This test-named operation is not an optimizer step.  It is allowed
        only before creating state for a CACH-A synthetic mini instance and
        refreshes the parameter/model identity so old state cannot cross the
        diagnostic boundary.  If this model came from a :class:`MinimalCACHPair`,
        that pair's original inventory and pair digest are intentionally
        invalidated; the mutated arm is thereafter a standalone diagnostic.
        """

        self._assert_cpu_model()
        if self.staging_variant is not CACHStagingVariant.CACH_A:
            raise MinimalCACHContractError(
                "nonzero action-seam diagnostic is candidate-only"
            )
        if self._diagnostic_nonzero_seam:
            raise MinimalCACHContractError(
                "nonzero action-seam diagnostic was already installed"
            )
        with torch.no_grad():
            for index, block in enumerate(self.blocks):
                projection = block.action_output_projection
                if projection is None:
                    raise MinimalCACHContractError(
                        "candidate action projection is missing"
                    )
                projection.weight.zero_()
                projection.weight.diagonal().fill_((index + 1) / 200.0)
                projection.bias.fill_((index + 1) / 2000.0)
        self._diagnostic_nonzero_seam = True
        self._registered_parameter_manifest_digest = (
            self._current_parameter_manifest_digest()
        )
        self._refresh_model_instance_id()
        self._assert_cpu_model()

    def _assert_cpu_model(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.device.type != "cpu" or parameter.dtype is not torch.float32:
                raise MinimalCACHContractError(
                    f"parameter {name} escaped the CPU/FP32 boundary"
                )
            if not bool(torch.isfinite(parameter.detach()).all()):
                raise MinimalCACHContractError(f"parameter {name} is non-finite")
            if parameter.requires_grad is not True:
                raise MinimalCACHContractError(
                    f"parameter {name} changed its trainable classification"
                )
        for name, buffer in self.named_buffers():
            if buffer.device.type != "cpu":
                raise MinimalCACHContractError(f"buffer {name} escaped CPU")
        if self._current_parameter_manifest_digest() != (
            self._registered_parameter_manifest_digest
        ):
            raise MinimalCACHContractError(
                "parameter/buffer snapshot changed outside the registered diagnostic"
            )
        if self._current_parameter_storage_signature() != (
            self._registered_parameter_storage_signature
        ):
            raise MinimalCACHContractError(
                "parameter storage topology changed after construction"
            )

    def empty_state(self, *, batch_size: int) -> MinimalGDNState:
        self._assert_cpu_model()
        batch_size = _require_plain_int(
            batch_size,
            "batch_size",
            minimum=1,
            maximum=8,
        )
        layers = tuple(
            MinimalLayerState(
                main_s_kv=torch.zeros(
                    batch_size,
                    self.spec.hidden_dim,
                    self.spec.hidden_dim,
                    dtype=self.spec.dtype,
                    device="cpu",
                ),
                main_s_z=torch.zeros(
                    batch_size,
                    self.spec.hidden_dim,
                    dtype=self.spec.dtype,
                    device="cpu",
                ),
            )
            for _ in range(self.spec.depth)
        )
        return MinimalGDNState(
            model_instance_id=self.model_instance_id,
            architecture_instance_token=self.architecture_instance_token,
            parameter_manifest_digest=self.parameter_manifest_digest,
            staging_variant=self.staging_variant,
            layout_instance_digest=self.layout.layout_instance_digest,
            batch_size=batch_size,
            revision=0,
            next_chunk_id=0,
            action_cursor=0,
            committed_actions=torch.empty(
                batch_size,
                0,
                self.spec.action_dim,
                dtype=self.spec.dtype,
                device="cpu",
            ),
            layers=layers,
        )

    def _validate_state(self, state: MinimalGDNState) -> None:
        self._assert_cpu_model()
        if not isinstance(state, MinimalGDNState):
            raise TypeError("state must be a MinimalGDNState")
        expected_identity = (
            self.model_instance_id,
            self.architecture_instance_token,
            self.parameter_manifest_digest,
            self.staging_variant,
            self.layout.layout_instance_digest,
        )
        observed_identity = (
            state.model_instance_id,
            state.architecture_instance_token,
            state.parameter_manifest_digest,
            state.staging_variant,
            state.layout_instance_digest,
        )
        if observed_identity != expected_identity:
            raise MinimalCACHContractError("state/model/variant identity mismatch")
        if state.revision != state.next_chunk_id:
            raise MinimalCACHContractError("state revision/chunk cursor mismatch")
        if state.next_chunk_id > len(self.layout.chunks):
            raise MinimalCACHContractError("state is beyond the bound layout")
        expected_action_cursor = (
            self.layout.valid_action_count
            if state.next_chunk_id == len(self.layout.chunks)
            else self.layout.chunks[state.next_chunk_id].action_start
        )
        if state.action_cursor != expected_action_cursor:
            raise MinimalCACHContractError("state action cursor mismatches layout")
        _require_tensor(
            state.committed_actions,
            name="state.committed_actions",
            shape=(state.batch_size, state.action_cursor, self.spec.action_dim),
        )
        if len(state.layers) != self.spec.depth:
            raise MinimalCACHContractError("state layer depth mismatch")
        for index, layer in enumerate(state.layers):
            _require_tensor(
                layer.main_s_kv,
                name=f"state.layers[{index}].main_s_kv",
                shape=(
                    state.batch_size,
                    self.spec.hidden_dim,
                    self.spec.hidden_dim,
                ),
            )
            _require_tensor(
                layer.main_s_z,
                name=f"state.layers[{index}].main_s_z",
                shape=(state.batch_size, self.spec.hidden_dim),
            )

    def _validate_batch(
        self,
        batch: MinimalChunkBatch,
        *,
        state: MinimalGDNState | None,
    ) -> tuple[ChunkLayout, int]:
        self._assert_cpu_model()
        if not isinstance(batch, MinimalChunkBatch):
            raise TypeError("batch must be a MinimalChunkBatch")
        if batch.layout_instance_digest != self.layout.layout_instance_digest:
            raise MinimalCACHContractError("batch layout digest mismatch")
        self.layout.verify_instance_digest(batch.layout_instance_digest)
        if not 0 <= batch.chunk_id < len(self.layout.chunks):
            raise IndexError("chunk_id is outside the bound layout")
        chunk = self.layout.chunks[batch.chunk_id]
        if state is not None:
            self._validate_state(state)
            if batch.chunk_id != state.next_chunk_id:
                raise MinimalCACHContractError("chunk is stale or out of order")
            batch_size = state.batch_size
        else:
            if (
                not isinstance(batch.video_latents, Tensor)
                or batch.video_latents.ndim != 3
            ):
                raise ValueError("video_latents must have exact rank 3")
            batch_size = int(batch.video_latents.shape[0])
            _require_plain_int(batch_size, "batch_size", minimum=1, maximum=8)

        _require_tensor(
            batch.video_latents,
            name="video_latents",
            shape=(batch_size, self.spec.chunk_size, self.spec.latent_dim),
        )
        if (
            not isinstance(batch.frame_valid_mask, Tensor)
            or batch.frame_valid_mask.dtype is not torch.bool
            or tuple(batch.frame_valid_mask.shape) != (batch_size, self.spec.chunk_size)
            or batch.frame_valid_mask.device.type != "cpu"
        ):
            raise TypeError("frame_valid_mask must be exact CPU bool [B,K]")
        expected_frame_mask = (
            torch.tensor(
                chunk.latent_valid_mask,
                dtype=torch.bool,
                device="cpu",
            )
            .view(1, -1)
            .expand(batch_size, -1)
        )
        if not torch.equal(batch.frame_valid_mask, expected_frame_mask):
            raise MinimalCACHContractError("frame_valid_mask differs from layout")
        if bool(
            batch.video_latents.masked_select(
                ~batch.frame_valid_mask.unsqueeze(-1)
            ).count_nonzero()
        ):
            raise MinimalCACHContractError("padded video latents must be exact zero")

        _require_tensor(
            batch.noisy_actions,
            name="noisy_actions",
            shape=(batch_size, chunk.action_slot_capacity, self.spec.action_dim),
        )
        if (
            not isinstance(batch.action_valid_mask, Tensor)
            or batch.action_valid_mask.dtype is not torch.bool
            or tuple(batch.action_valid_mask.shape)
            != (batch_size, chunk.action_slot_capacity)
            or batch.action_valid_mask.device.type != "cpu"
        ):
            raise TypeError(
                "action_valid_mask must be exact CPU bool [B,action_capacity]"
            )
        expected_action_mask = (
            torch.tensor(
                chunk.action_valid_mask,
                dtype=torch.bool,
                device="cpu",
            )
            .view(1, -1)
            .expand(batch_size, -1)
        )
        if not torch.equal(batch.action_valid_mask, expected_action_mask):
            raise MinimalCACHContractError("action_valid_mask differs from layout")
        if bool(
            batch.noisy_actions.masked_select(
                ~batch.action_valid_mask.unsqueeze(-1)
            ).count_nonzero()
        ):
            raise MinimalCACHContractError("padded actions must be exact zero")

        _require_tensor(
            batch.proprio_state,
            name="proprio_state",
            shape=(batch_size, self.spec.action_dim),
        )
        if len(batch.proprio_bindings) != batch_size:
            raise MinimalCACHContractError("one proprio binding is required per row")
        for index, binding in enumerate(batch.proprio_bindings):
            binding.verify(
                layout_instance_digest=self.layout.layout_instance_digest,
                chunk=chunk,
                proprio_value_sha256=canonical_proprio_row_sha256(
                    batch.proprio_state[index]
                ),
            )

        for name in ("video_timestep", "action_timestep"):
            timestep = _require_tensor(
                getattr(batch, name),
                name=name,
                shape=(batch_size,),
            )
            if bool((timestep.detach() < 0).any()):
                raise ValueError(f"{name} must be non-negative")
        return chunk, batch_size

    def _action_embedding(
        self,
        batch: MinimalChunkBatch,
        chunk: ChunkLayout,
        *,
        committed_actions: Tensor,
    ) -> Tensor | None:
        if self.staging_variant is CACHStagingVariant.REF_GDN_CORRECTED:
            if self.action_conditioner is not None or self.no_action_slot is not None:
                raise MinimalCACHContractError(
                    "reference action bypass is not structural"
                )
            return None
        if self.action_conditioner is None or self.no_action_slot is None:
            raise MinimalCACHContractError("candidate action conditioner is missing")
        condition = reduce_end_of_bin_action_condition(
            batch.noisy_actions,
            committed_actions=(
                None if committed_actions.shape[1] == 0 else committed_actions
            ),
            chunk=chunk,
            no_action_slot=self.no_action_slot,
        )
        if not torch.equal(condition.latent_valid_mask, batch.frame_valid_mask):
            raise MinimalCACHContractError("reducer/layout frame masks differ")
        embedded = self.action_conditioner(condition.condition)
        return torch.where(
            batch.frame_valid_mask.unsqueeze(-1),
            embedded,
            torch.zeros_like(embedded),
        )

    def _video_embedding(self, batch: MinimalChunkBatch) -> Tensor:
        timestep = batch.video_timestep.view(-1, 1, 1) / 1000.0
        hidden = self.video_input_projection(batch.video_latents)
        hidden = hidden + self.video_timestep_projection(timestep)
        return torch.where(
            batch.frame_valid_mask.unsqueeze(-1),
            hidden,
            torch.zeros_like(hidden),
        )

    @staticmethod
    def _action_to_latent_indices(chunk: ChunkLayout) -> tuple[int, ...]:
        owners = [-1] * chunk.action_slot_capacity
        for local_latent, span in enumerate(chunk.latent_action_spans):
            for action_index in range(span.action_start, span.action_end):
                owners[action_index - chunk.action_start] = local_latent
        if any(index < 0 for index in owners[: chunk.valid_action_count]):
            raise MinimalCACHContractError("action-to-latent ownership has a gap")
        return tuple(owners)

    def _predict_actions(
        self,
        batch: MinimalChunkBatch,
        chunk: ChunkLayout,
        video_hidden: Tensor,
    ) -> Tensor:
        batch_size = video_hidden.shape[0]
        capacity = chunk.action_slot_capacity
        bridge = torch.zeros(
            batch_size,
            capacity,
            self.spec.hidden_dim,
            dtype=self.spec.dtype,
            device="cpu",
        )
        for action_index, latent_index in enumerate(
            self._action_to_latent_indices(chunk)
        ):
            if latent_index >= 0:
                bridge[:, action_index] = video_hidden[:, latent_index]

        positions = torch.zeros(
            capacity,
            2,
            dtype=self.spec.dtype,
            device="cpu",
        )
        if chunk.valid_action_count:
            ordinal = torch.tensor(
                chunk.action_rope_positions,
                dtype=self.spec.dtype,
                device="cpu",
            )
            positions[: chunk.valid_action_count, 0] = torch.sin(ordinal)
            positions[: chunk.valid_action_count, 1] = torch.cos(ordinal)
        hidden = self.action_input_projection(batch.noisy_actions)
        hidden = hidden + self.action_video_projection(bridge)
        hidden = hidden + self.action_proprio_projection(batch.proprio_state).unsqueeze(
            1
        )
        hidden = hidden + self.action_position_projection(positions).unsqueeze(0)
        hidden = hidden + self.action_timestep_projection(
            batch.action_timestep.view(-1, 1, 1) / 1000.0
        )
        prediction = self.action_output_projection(torch.tanh(hidden))
        return torch.where(
            batch.action_valid_mask.unsqueeze(-1),
            prediction,
            torch.zeros_like(prediction),
        )

    @staticmethod
    def _require_exact_t0(batch: MinimalChunkBatch) -> None:
        if not _tensor_is_exact_zero(batch.video_timestep) or not _tensor_is_exact_zero(
            batch.action_timestep
        ):
            raise MinimalCACHContractError(
                "commit-eligible staging requires exact video/action t=0"
            )

    def _execute_streaming_chunk(
        self,
        batch: MinimalChunkBatch,
        source_state: MinimalGDNState,
    ) -> tuple[MinimalChunkOutput, MinimalGDNState]:
        chunk, _ = self._validate_batch(batch, state=source_state)
        action_embedding = self._action_embedding(
            batch,
            chunk,
            committed_actions=source_state.committed_actions,
        )
        hidden = self._video_embedding(batch)
        next_layers: list[MinimalLayerState] = []
        for block, layer_state in zip(self.blocks, source_state.layers):
            hidden, next_layer = block.scan_chunk(
                hidden,
                layer_state,
                frame_valid_mask=batch.frame_valid_mask,
                action_embedding=action_embedding,
            )
            next_layers.append(next_layer)
        video_prediction = self.video_output_projection(hidden)
        video_prediction = torch.where(
            batch.frame_valid_mask.unsqueeze(-1),
            video_prediction,
            torch.zeros_like(video_prediction),
        )
        action_prediction = self._predict_actions(batch, chunk, hidden)
        output = MinimalChunkOutput(
            chunk_id=batch.chunk_id,
            video_prediction=video_prediction,
            action_prediction=action_prediction,
        )
        valid_actions = batch.noisy_actions[:, : chunk.valid_action_count]
        proposed_state = MinimalGDNState(
            model_instance_id=self.model_instance_id,
            architecture_instance_token=self.architecture_instance_token,
            parameter_manifest_digest=self.parameter_manifest_digest,
            staging_variant=self.staging_variant,
            layout_instance_digest=self.layout.layout_instance_digest,
            batch_size=source_state.batch_size,
            revision=source_state.revision + 1,
            next_chunk_id=source_state.next_chunk_id + 1,
            action_cursor=chunk.action_end,
            committed_actions=torch.cat(
                [source_state.committed_actions, valid_actions],
                dim=1,
            ),
            layers=tuple(next_layers),
        )
        self._validate_state(proposed_state)
        return output, proposed_state

    def forward(
        self,
        batch: MinimalChunkBatch,
        source_state: MinimalGDNState,
    ) -> MinimalChunkOutput:
        """Run one read-only denoise chunk and discard its scratch state."""

        output, _ = self._execute_streaming_chunk(batch, source_state)
        return output

    def run_denoise_chunk(
        self,
        batch: MinimalChunkBatch,
        source_state: MinimalGDNState,
    ) -> MinimalChunkOutput:
        """Explicit alias for the read-only :meth:`forward` path."""

        return self.forward(batch, source_state)

    def stage_paired_t0(
        self,
        batch: MinimalChunkBatch,
        source_state: MinimalGDNState,
        *,
        owner_id: str,
        owner_epoch: int,
    ) -> MinimalChunkProposal:
        """Create a commit-eligible proposal for one exact paired ``t=0`` call."""

        self._validate_batch(batch, state=source_state)
        self._require_exact_t0(batch)
        _require_canonical_string(owner_id, "owner_id")
        _require_plain_int(owner_epoch, "owner_epoch", minimum=0)
        output, proposed_state = self._execute_streaming_chunk(batch, source_state)
        return MinimalChunkProposal.create(
            owner_id=owner_id,
            owner_epoch=owner_epoch,
            source_state=source_state,
            output=output,
            proposed_state=proposed_state,
        )

    def forward_sequence(
        self,
        batches: tuple[MinimalChunkBatch, ...],
    ) -> MinimalSequenceResult:
        """Run an independent layer-major full-layout ``t=0`` scan.

        This path does not call :meth:`forward`, :meth:`stage_paired_t0`, or
        the chunk-major execution helper.  It is read-only and returns no
        commit authority.
        """

        self._assert_cpu_model()
        if not isinstance(batches, tuple) or len(batches) != len(self.layout.chunks):
            raise MinimalCACHContractError(
                "forward_sequence requires exactly the complete bound layout"
            )
        batch_size: int | None = None
        chunks: list[ChunkLayout] = []
        history: Tensor | None = None
        hidden_chunks: list[Tensor] = []
        action_embeddings: list[Tensor | None] = []
        for expected_chunk_id, batch in enumerate(batches):
            chunk, current_batch_size = self._validate_batch(batch, state=None)
            if batch.chunk_id != expected_chunk_id:
                raise MinimalCACHContractError("sequence chunks are out of order")
            self._require_exact_t0(batch)
            if batch_size is None:
                batch_size = current_batch_size
                history = torch.empty(
                    batch_size,
                    0,
                    self.spec.action_dim,
                    dtype=self.spec.dtype,
                    device="cpu",
                )
            elif current_batch_size != batch_size:
                raise MinimalCACHContractError("sequence batch size changed")
            assert history is not None
            chunks.append(chunk)
            hidden_chunks.append(self._video_embedding(batch))
            action_embeddings.append(
                self._action_embedding(
                    batch,
                    chunk,
                    committed_actions=history,
                )
            )
            history = torch.cat(
                [history, batch.noisy_actions[:, : chunk.valid_action_count]],
                dim=1,
            )
        assert batch_size is not None and history is not None

        final_layers: list[MinimalLayerState] = []
        for block in self.blocks:
            layer_state = MinimalLayerState(
                main_s_kv=torch.zeros(
                    batch_size,
                    self.spec.hidden_dim,
                    self.spec.hidden_dim,
                    dtype=self.spec.dtype,
                    device="cpu",
                ),
                main_s_z=torch.zeros(
                    batch_size,
                    self.spec.hidden_dim,
                    dtype=self.spec.dtype,
                    device="cpu",
                ),
            )
            next_hidden_chunks: list[Tensor] = []
            for batch, hidden, action_embedding in zip(
                batches,
                hidden_chunks,
                action_embeddings,
            ):
                hidden, layer_state = block.scan_chunk(
                    hidden,
                    layer_state,
                    frame_valid_mask=batch.frame_valid_mask,
                    action_embedding=action_embedding,
                )
                next_hidden_chunks.append(hidden)
            hidden_chunks = next_hidden_chunks
            final_layers.append(layer_state)

        outputs: list[MinimalChunkOutput] = []
        for batch, chunk, hidden in zip(batches, chunks, hidden_chunks):
            video_prediction = self.video_output_projection(hidden)
            video_prediction = torch.where(
                batch.frame_valid_mask.unsqueeze(-1),
                video_prediction,
                torch.zeros_like(video_prediction),
            )
            outputs.append(
                MinimalChunkOutput(
                    chunk_id=batch.chunk_id,
                    video_prediction=video_prediction,
                    action_prediction=self._predict_actions(batch, chunk, hidden),
                )
            )
        final_state = MinimalGDNState(
            model_instance_id=self.model_instance_id,
            architecture_instance_token=self.architecture_instance_token,
            parameter_manifest_digest=self.parameter_manifest_digest,
            staging_variant=self.staging_variant,
            layout_instance_digest=self.layout.layout_instance_digest,
            batch_size=batch_size,
            revision=len(batches),
            next_chunk_id=len(batches),
            action_cursor=self.layout.valid_action_count,
            committed_actions=history,
            layers=tuple(final_layers),
        )
        self._validate_state(final_state)
        return MinimalSequenceResult(outputs=tuple(outputs), final_state=final_state)


class MinimalEphemeralStateOwner:
    """Single-writer, in-memory owner for paired synthetic state commits."""

    def __init__(
        self,
        architecture: MinimalCorrectedGDNArchitecture,
        *,
        batch_size: int,
        owner_id: str,
    ) -> None:
        if not isinstance(architecture, MinimalCorrectedGDNArchitecture):
            raise TypeError("architecture must be MinimalCorrectedGDNArchitecture")
        self._architecture = architecture
        self._owner_id = _require_canonical_string(owner_id, "owner_id")
        self._batch_size = _require_plain_int(
            batch_size,
            "batch_size",
            minimum=1,
            maximum=8,
        )
        self._epoch = 0
        self._state = architecture.empty_state(batch_size=batch_size)
        self._issued_proposals: dict[str, MinimalChunkProposal] = {}
        self._committed_proposals: set[str] = set()

    @property
    def revision(self) -> int:
        return self._state.revision

    @property
    def next_chunk_id(self) -> int:
        return self._state.next_chunk_id

    def snapshot(self) -> MinimalGDNState:
        return self._state.detached_clone()

    def run_denoise_chunk(self, batch: MinimalChunkBatch) -> MinimalChunkOutput:
        source_digest = self._state.state_digest
        output = self._architecture.run_denoise_chunk(batch, self._state)
        if self._state.state_digest != source_digest:
            raise MinimalCACHContractError("denoise mutated committed state")
        return output

    def stage_paired_t0(self, batch: MinimalChunkBatch) -> MinimalChunkProposal:
        source_digest = self._state.state_digest
        proposal = self._architecture.stage_paired_t0(
            batch,
            self._state,
            owner_id=self._owner_id,
            owner_epoch=self._epoch,
        )
        if self._state.state_digest != source_digest:
            raise MinimalCACHContractError("staging mutated committed state")
        issued = self._issued_proposals.get(proposal.proposal_id)
        if issued is not None:
            return issued
        self._issued_proposals[proposal.proposal_id] = proposal
        return proposal

    def commit_paired(self, proposal: MinimalChunkProposal) -> MinimalGDNState:
        if not isinstance(proposal, MinimalChunkProposal):
            raise TypeError("proposal must be a MinimalChunkProposal")
        proposal.assert_integrity()
        if proposal.owner_id != self._owner_id or proposal.owner_epoch != self._epoch:
            raise MinimalCACHContractError("proposal belongs to another owner/epoch")
        if proposal.proposal_id in self._committed_proposals:
            raise MinimalCACHContractError("proposal was already committed")
        if self._issued_proposals.get(proposal.proposal_id) is not proposal:
            raise MinimalCACHContractError(
                "proposal was not issued by this owner instance"
            )
        expected_identity = (
            self._architecture.model_instance_id,
            self._architecture.architecture_instance_token,
            self._architecture.parameter_manifest_digest,
            self._architecture.staging_variant,
            self._architecture.layout.layout_instance_digest,
        )
        observed_identity = (
            proposal.model_instance_id,
            proposal.architecture_instance_token,
            proposal.parameter_manifest_digest,
            proposal.staging_variant,
            proposal.layout_instance_digest,
        )
        if observed_identity != expected_identity:
            raise MinimalCACHContractError("proposal arm/model/layout mismatch")
        if (
            proposal.source_revision != self._state.revision
            or proposal.source_state_digest != self._state.state_digest
            or proposal.chunk_id != self._state.next_chunk_id
        ):
            raise MinimalCACHContractError("proposal is stale or out of order")
        self._architecture._validate_state(proposal.proposed_state)
        if proposal.proposed_state.revision != self._state.revision + 1:
            raise MinimalCACHContractError("proposal must advance exactly once")
        next_state = proposal.proposed_state.detached_clone()
        # All validation precedes the only live-state mutation.
        self._state = next_state
        self._issued_proposals.pop(proposal.proposal_id)
        self._committed_proposals.add(proposal.proposal_id)
        return self.snapshot()

    def reset(self) -> MinimalGDNState:
        self._epoch += 1
        self._state = self._architecture.empty_state(batch_size=self._batch_size)
        self._issued_proposals.clear()
        self._committed_proposals.clear()
        return self.snapshot()


@dataclass(frozen=True)
class MinimalParameterInventoryEntry:
    """Immutable named tensor identity plus its registered initializer role."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    digest: str
    initializer_role: str
    requires_grad: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "dtype": self.dtype,
            "initializer_role": self.initializer_role,
            "name": self.name,
            "requires_grad": self.requires_grad,
            "shape": self.shape,
        }


def _initializer_role(name: str, *, candidate_only: bool) -> str:
    if candidate_only:
        if name == "no_action_slot" or ".action_output_projection." in name:
            return "candidate_exact_zero_output_seam_v1"
        if name.startswith("action_conditioner.") and name.endswith(".bias"):
            return "candidate_operator_exact_zero_bias_v1"
        if name.startswith("action_conditioner."):
            return "candidate_operator_named_uniform_v1"
        raise MinimalCACHContractError(
            f"unregistered candidate-only initializer role: {name}"
        )
    if name.endswith(".bias"):
        return "shared_exact_zero_bias_v1"
    return "shared_named_uniform_v1"


def _inventory_entry(
    name: str,
    parameter: Tensor,
    *,
    candidate_only: bool,
) -> MinimalParameterInventoryEntry:
    return MinimalParameterInventoryEntry(
        name=name,
        shape=tuple(parameter.shape),
        dtype=str(parameter.dtype),
        digest=tensor_digest(parameter),
        initializer_role=_initializer_role(name, candidate_only=candidate_only),
        requires_grad=parameter.requires_grad,
    )


@dataclass(frozen=True)
class MinimalCACHPair:
    """Independent random-init reference/candidate arms plus inventories."""

    reference: MinimalCorrectedGDNArchitecture
    candidate: MinimalCorrectedGDNArchitecture
    shared_parameter_inventory: tuple[MinimalParameterInventoryEntry, ...]
    candidate_only_parameter_inventory: tuple[MinimalParameterInventoryEntry, ...]
    pair_digest: str

    @property
    def shared_parameter_digests(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (entry.name, entry.digest) for entry in self.shared_parameter_inventory
        )

    @property
    def candidate_only_parameters(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.candidate_only_parameter_inventory)

    def _digest_payload(self) -> dict[str, object]:
        return {
            "candidate_model_instance_id": self.candidate.model_instance_id,
            "candidate_only_parameter_inventory": tuple(
                entry.to_payload() for entry in self.candidate_only_parameter_inventory
            ),
            "layout_instance_digest": self.reference.layout.layout_instance_digest,
            "reference_model_instance_id": self.reference.model_instance_id,
            "schema": "cach.minimal_ref_candidate_pair.v1",
            "shared_parameter_inventory": tuple(
                entry.to_payload() for entry in self.shared_parameter_inventory
            ),
            "spec": self.reference.spec.to_manifest(),
            "storage_alias_free": True,
        }

    def assert_integrity(self) -> None:
        self.reference._assert_cpu_model()
        self.candidate._assert_cpu_model()
        ref_inventory = _parameter_inventory(self.reference)
        candidate_inventory = _parameter_inventory(self.candidate)
        ref_storages = tuple(
            parameter.untyped_storage().data_ptr()
            for parameter in ref_inventory.values()
        )
        candidate_storages = tuple(
            parameter.untyped_storage().data_ptr()
            for parameter in candidate_inventory.values()
        )
        if (
            len(ref_storages) != len(set(ref_storages))
            or len(candidate_storages) != len(set(candidate_storages))
            or not set(ref_storages).isdisjoint(candidate_storages)
        ):
            raise MinimalCACHContractError(
                "parameter storage alias topology is not pair-independent"
            )
        expected_shared = tuple(
            _inventory_entry(name, ref_inventory[name], candidate_only=False)
            for name in sorted(ref_inventory.keys() & candidate_inventory.keys())
        )
        expected_candidate_only = tuple(
            _inventory_entry(name, candidate_inventory[name], candidate_only=True)
            for name in sorted(candidate_inventory.keys() - ref_inventory.keys())
        )
        if (
            expected_shared != self.shared_parameter_inventory
            or expected_candidate_only != self.candidate_only_parameter_inventory
        ):
            raise MinimalCACHContractError("pair inventory changed after construction")
        for entry in expected_shared:
            if tensor_digest(candidate_inventory[entry.name]) != entry.digest:
                raise MinimalCACHContractError(
                    f"candidate shared parameter changed at {entry.name}"
                )
        if canonical_sha256(self._digest_payload()) != self.pair_digest:
            raise MinimalCACHContractError("pair digest/inventory mismatch")


def _parameter_inventory(module: nn.Module) -> dict[str, Tensor]:
    return {name: parameter for name, parameter in module.named_parameters()}


def build_minimal_ref_cach_pair(
    *,
    spec: MinimalCACHSpec,
    layout: ChunkActionLayout,
) -> MinimalCACHPair:
    """Build independent, named-initialized CPU synthetic reference/candidate."""

    reference = MinimalCorrectedGDNArchitecture(
        spec=spec,
        layout=layout,
        staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
    )
    candidate = MinimalCorrectedGDNArchitecture(
        spec=spec,
        layout=layout,
        staging_variant=CACHStagingVariant.CACH_A,
    )
    ref_inventory = _parameter_inventory(reference)
    candidate_inventory = _parameter_inventory(candidate)
    shared_names = tuple(sorted(ref_inventory.keys() & candidate_inventory.keys()))
    reference_only = tuple(sorted(ref_inventory.keys() - candidate_inventory.keys()))
    candidate_only = tuple(sorted(candidate_inventory.keys() - ref_inventory.keys()))
    if reference_only:
        raise MinimalCACHContractError(
            f"unexpected reference-only parameters: {reference_only}"
        )
    if not candidate_only or any(
        not (
            name == "no_action_slot"
            or name.startswith("action_conditioner.")
            or ".action_output_projection." in name
        )
        for name in candidate_only
    ):
        raise MinimalCACHContractError("candidate-only inventory escaped action seam")

    shared_entries: list[MinimalParameterInventoryEntry] = []
    for name in shared_names:
        ref_parameter = ref_inventory[name]
        candidate_parameter = candidate_inventory[name]
        if (
            ref_parameter.shape != candidate_parameter.shape
            or ref_parameter.dtype != candidate_parameter.dtype
            or tensor_digest(ref_parameter) != tensor_digest(candidate_parameter)
        ):
            raise MinimalCACHContractError(f"shared named init differs at {name}")
        if ref_parameter.untyped_storage().data_ptr() == (
            candidate_parameter.untyped_storage().data_ptr()
        ):
            raise MinimalCACHContractError(f"pair aliases shared parameter {name}")
        shared_entries.append(
            _inventory_entry(name, ref_parameter, candidate_only=False)
        )

    candidate_zero_keys = tuple(
        name
        for name in candidate_only
        if name == "no_action_slot" or ".action_output_projection." in name
    )
    if not candidate_zero_keys or any(
        not _tensor_is_exact_zero(candidate_inventory[name])
        for name in candidate_zero_keys
    ):
        raise MinimalCACHContractError("candidate output seam is not exact zero")

    candidate_only_entries = tuple(
        _inventory_entry(name, candidate_inventory[name], candidate_only=True)
        for name in candidate_only
    )
    pair = MinimalCACHPair(
        reference=reference,
        candidate=candidate,
        shared_parameter_inventory=tuple(shared_entries),
        candidate_only_parameter_inventory=candidate_only_entries,
        pair_digest="pending",
    )
    object.__setattr__(pair, "pair_digest", canonical_sha256(pair._digest_payload()))
    pair.assert_integrity()
    return pair


__all__ = [
    "MinimalCACHContractError",
    "MinimalCACHPair",
    "MinimalCACHSpec",
    "MinimalChunkBatch",
    "MinimalChunkOutput",
    "MinimalChunkProposal",
    "MinimalCorrectedGDNArchitecture",
    "MinimalEphemeralStateOwner",
    "MinimalGDNState",
    "MinimalLayerState",
    "MinimalParameterInventoryEntry",
    "MinimalSequenceResult",
    "PureTorchChunkCausalGDNProxy",
    "build_minimal_ref_cach_pair",
]
