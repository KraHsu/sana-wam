"""CPU-only production-shaped CACH mini harness for Phase-C contract closure.

This module is deliberately test-only and non-scientific.  It exercises the
real public chunk adapter, ActionDiT, layout, typed cache codec, conditioning,
and Stage-2B read-view/context types with a tiny pure-Torch recurrent operator.
It cannot load a checkpoint or dataset, construct an optimizer/trainer, use
CUDA, publish a durable receipt, create an admission root, or build the 2B
model.  The ordinary CACH production builder remains fail-closed.

The pure-Torch recurrent operator is an interface/transition proxy.  It does
not claim numerical parity with the vendored CUDA/Triton GDN kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType, SimpleNamespace
import threading
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn

from sana_wam.cach.action_conditioning import (
    reduce_end_of_bin_action_condition,
)
from sana_wam.cach.committed_action_history import (
    CommittedActionHistory,
    CommittedActionHistoryView,
)
from sana_wam.cach.prefix_compaction import FixedKPrefixPlan
from sana_wam.cach.staging_variant import (
    CACHStagingVariant,
    require_cach_staging_variant,
)
from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
from sana_wam.model.action_chunk_layout import (
    ChunkActionLayout,
    SelectedProprioBinding,
    build_synthetic_chunk_action_layout_for_tests,
    canonical_proprio_row_sha256,
    canonical_sha256,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)
from sana_wam.model.cach_minimal_cpu import (
    MinimalLayerState,
    PureTorchChunkCausalGDNProxy,
)
from sana_wam.model.cach_numerical_core import (
    CACHNumericalCore,
    CACHNumericalResult,
)
from sana_wam.model.cach_paired_stager import CACHChunkConditioning
from sana_wam.model.video_backbone.sana.adapter import SanaVideoBackbone
from sana_wam.model.video_backbone.sana import hybrid_cache as hc
from sana_wam.model.video_backbone.sana import hybrid_cache_stage2b as h2


_REVISION = "cach-production-path-mini-phase-c-v1"
_EMPTY_HISTORY_DIGEST = hashlib.sha256(
    b"cach.production_path_mini.empty_history.v1"
).hexdigest()
_FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "F",
        "afcc",
        "afcc_reference",
        "anchor",
        "anchors",
        "ar_observed_prefix_chunks",
        "attnres",
        "checkpoint",
        "checkpoint_path",
        "cuda",
        "data",
        "dataset",
        "dataset_path",
        "device_map",
        "eval",
        "evaluation",
        "fixed_atc",
        "init_dit_from",
        "model_path",
        "num_clean_prefix_actions",
        "num_clean_prefix_latents",
        "optimizer",
        "resume",
        "self_forcing",
        "trainer",
        "training",
    }
)
_ALLOWED_CONFIG = MappingProxyType(
    {
        "device": "cpu",
        "dtype": "float32",
        "episode_bootstrap": "first_frame_pinned",
        "observed_prefix_chunks": 0,
        "synthetic_test_only": True,
    }
)


class ProductionPathMiniContractError(RuntimeError):
    """A Phase-C mini request escaped its closed CPU/synthetic contract."""


def _require_cpu_fp32(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch Tensor")
    if value.device.type != "cpu" or value.dtype is not torch.float32:
        raise ProductionPathMiniContractError(f"{name} must stay CPU/FP32")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ProductionPathMiniContractError(f"{name} contains NaN or Inf")


def _named_seed(base_seed: int, role: str, name: str) -> int:
    material = f"{_REVISION}:{base_seed}:{role}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _is_exact_zero(value: Tensor) -> bool:
    return not bool(value.detach().count_nonzero())


def _raw_tensor_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and left.device == right.device
        and tuple(left.shape) == tuple(right.shape)
        and hc.tensor_digest(left) == hc.tensor_digest(right)
    )


@dataclass(frozen=True)
class ProductionPathMiniSpec:
    """Fixed Phase-C sizes; topology and action semantics are not miniaturized."""

    latent_channels: int = 3
    hidden_dim: int = 4
    action_dim: int = 20
    context_dim: int = 4
    depth: int = 20
    chunk_size: int = 3
    temporal_compression: int = 8
    video_stride: int = 1
    valid_raw_count: int = 57
    batch_size: int = 1
    shared_init_seed: int = 2026080201
    candidate_init_seed: int = 2026080202
    input_seed: int = 2026080203

    def __post_init__(self) -> None:
        integer_fields = (
            "latent_channels",
            "hidden_dim",
            "action_dim",
            "context_dim",
            "depth",
            "chunk_size",
            "temporal_compression",
            "video_stride",
            "valid_raw_count",
            "batch_size",
            "shared_init_seed",
            "candidate_init_seed",
            "input_seed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive plain integer")
        fixed_values = {
            "action_dim": 20,
            "batch_size": 1,
            "candidate_init_seed": 2026080202,
            "chunk_size": 3,
            "context_dim": 4,
            "depth": 20,
            "hidden_dim": 4,
            "input_seed": 2026080203,
            "latent_channels": 3,
            "shared_init_seed": 2026080201,
            "temporal_compression": 8,
            "valid_raw_count": 57,
            "video_stride": 1,
        }
        drift = {
            name: (getattr(self, name), expected)
            for name, expected in fixed_values.items()
            if getattr(self, name) != expected
        }
        if drift:
            raise ProductionPathMiniContractError(
                f"Phase-C mini sizes/seeds are immutable: {drift}"
            )

    def to_manifest(self) -> dict[str, object]:
        return {
            "action_dim": self.action_dim,
            "batch_size": self.batch_size,
            "candidate_init_seed": self.candidate_init_seed,
            "chunk_size": self.chunk_size,
            "complete_2b": False,
            "context_dim": self.context_dim,
            "depth": self.depth,
            "device": "cpu",
            "dtype": "float32",
            "hidden_dim": self.hidden_dim,
            "input_seed": self.input_seed,
            "latent_channels": self.latent_channels,
            "shared_init_seed": self.shared_init_seed,
            "synthetic_test_only": True,
            "temporal_compression": self.temporal_compression,
            "valid_raw_count": self.valid_raw_count,
            "vendor_gdn_numerical_parity": False,
            "video_stride": self.video_stride,
        }


def build_production_path_mini_layout_for_tests(
    spec: ProductionPathMiniSpec,
    *,
    valid_raw_count: int | None = None,
) -> ChunkActionLayout:
    """Build the exact typed layout through the visibly synthetic API."""

    if not isinstance(spec, ProductionPathMiniSpec):
        raise TypeError("spec must be ProductionPathMiniSpec")
    raw_count = spec.valid_raw_count if valid_raw_count is None else valid_raw_count
    if type(raw_count) is not int or raw_count <= 1:
        raise ValueError("valid_raw_count must be a plain integer greater than one")
    if raw_count not in {9, 17, 57}:
        raise ProductionPathMiniContractError(
            "Phase-C layout helper admits only the three fixed synthetic cases"
        )
    layout_spec = synthetic_layout_spec(
        frame_chunk_size=spec.chunk_size,
        temporal_compression=spec.temporal_compression,
        video_stride=spec.video_stride,
        action_dim=spec.action_dim,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=raw_count,
        video_stride=spec.video_stride,
        source_row_label="phase-c-production-path-mini-row",
        episode_label="phase-c-production-path-mini-episode",
    )
    return build_synthetic_chunk_action_layout_for_tests(
        spec=layout_spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=raw_count,
        video_valid_mask=(True,) * raw_count,
        action_valid_mask=(True,) * (raw_count - 1),
    )


@dataclass(frozen=True)
class ProductionPathMiniChunk:
    """Synthetic model inputs plus clean paired targets kept outside denoise."""

    chunk_id: int
    noisy_video: Tensor = field(repr=False, compare=False)
    noisy_actions: Tensor = field(repr=False, compare=False)
    clean_video_target: Tensor = field(repr=False, compare=False)
    clean_action_target: Tensor = field(repr=False, compare=False)
    frame_valid_mask: Tensor = field(repr=False, compare=False)
    action_valid_mask: Tensor = field(repr=False, compare=False)
    context: Tensor = field(repr=False, compare=False)
    context_mask: Tensor = field(repr=False, compare=False)
    seq_lens: Tensor = field(repr=False, compare=False)
    proprio_state: Tensor = field(repr=False, compare=False)
    video_timestep: Tensor = field(repr=False, compare=False)
    action_timestep: Tensor = field(repr=False, compare=False)

    def owned_clone(self, **updates: Tensor) -> "ProductionPathMiniChunk":
        values: dict[str, object] = {"chunk_id": self.chunk_id}
        for name in (
            "noisy_video",
            "noisy_actions",
            "clean_video_target",
            "clean_action_target",
            "frame_valid_mask",
            "action_valid_mask",
            "context",
            "context_mask",
            "seq_lens",
            "proprio_state",
            "video_timestep",
            "action_timestep",
        ):
            source = updates.get(name, getattr(self, name))
            values[name] = source.detach().clone()
        return type(self)(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductionPathMiniEpisode:
    layout: ChunkActionLayout
    chunks: tuple[ProductionPathMiniChunk, ...]
    recipe_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.layout, ChunkActionLayout):
            raise TypeError("layout must be ChunkActionLayout")
        if self.layout.synthetic_test_only is not True:
            raise ProductionPathMiniContractError(
                "Phase-C accepts only synthetic-provenance layouts"
            )
        if len(self.chunks) != len(self.layout.chunks):
            raise ValueError("episode chunk count differs from layout")
        if [chunk.chunk_id for chunk in self.chunks] != list(range(len(self.chunks))):
            raise ValueError("episode chunks must be contiguous and ordered")
        if (
            not isinstance(self.recipe_digest, str)
            or len(self.recipe_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.recipe_digest)
        ):
            raise ValueError("recipe_digest must be a SHA256")
        for batch, chunk_layout in zip(self.chunks, self.layout.chunks):
            expected_video = (
                1,
                3,
                len(chunk_layout.latent_valid_mask),
                1,
                1,
            )
            if tuple(batch.noisy_video.shape) != expected_video:
                raise ValueError("noisy video must retain exact fixed-K shape")
            if tuple(batch.clean_video_target.shape) != expected_video:
                raise ValueError("clean video target differs from noisy video shape")
            if tuple(batch.noisy_actions.shape) != (
                1,
                chunk_layout.action_slot_capacity,
                20,
            ):
                raise ValueError("noisy actions differ from layout capacity")
            if tuple(batch.clean_action_target.shape) != tuple(
                batch.noisy_actions.shape
            ):
                raise ValueError("clean action target differs from noisy action shape")
            for name in (
                "noisy_video",
                "noisy_actions",
                "clean_video_target",
                "clean_action_target",
                "context",
                "proprio_state",
                "video_timestep",
                "action_timestep",
            ):
                _require_cpu_fp32(getattr(batch, name), name)
            if (
                tuple(batch.frame_valid_mask.shape)
                != (1, len(chunk_layout.latent_valid_mask))
                or batch.frame_valid_mask.dtype is not torch.bool
                or batch.frame_valid_mask.device.type != "cpu"
            ):
                raise ProductionPathMiniContractError(
                    "frame_valid_mask must stay exact CPU bool [1,K]"
                )
            if (
                tuple(batch.action_valid_mask.shape)
                != (1, chunk_layout.action_slot_capacity)
                or batch.action_valid_mask.dtype is not torch.bool
                or batch.action_valid_mask.device.type != "cpu"
            ):
                raise ProductionPathMiniContractError(
                    "action_valid_mask must stay exact CPU bool [1,A]"
                )
            expected_frame_mask = torch.tensor(
                chunk_layout.latent_valid_mask,
                dtype=torch.bool,
                device="cpu",
            ).view(1, -1)
            expected_action_mask = torch.tensor(
                chunk_layout.action_valid_mask,
                dtype=torch.bool,
                device="cpu",
            ).view(1, -1)
            if not torch.equal(batch.frame_valid_mask, expected_frame_mask):
                raise ValueError("frame_valid_mask differs from layout")
            if not torch.equal(batch.action_valid_mask, expected_action_mask):
                raise ValueError("action_valid_mask differs from layout")
            if tuple(batch.context.shape) != (1, 2, 4):
                raise ValueError("context must have exact Phase-C shape [1,2,4]")
            if (
                tuple(batch.context_mask.shape) != (1, 2)
                or batch.context_mask.dtype is not torch.bool
                or batch.context_mask.device.type != "cpu"
                or not bool(batch.context_mask.all())
            ):
                raise ValueError("context_mask must be exact CPU all-true [1,2]")
            if (
                tuple(batch.seq_lens.shape) != (1,)
                or batch.seq_lens.dtype is not torch.long
                or batch.seq_lens.device.type != "cpu"
                or not torch.equal(
                    batch.seq_lens,
                    torch.tensor([2], dtype=torch.long, device="cpu"),
                )
            ):
                raise ValueError("seq_lens must be exact CPU int64 [2]")
            if tuple(batch.proprio_state.shape) != (1, 20):
                raise ValueError("proprio_state must have exact Phase-C shape [1,20]")
            if tuple(batch.video_timestep.shape) != (1,):
                raise ValueError("video_timestep must have exact Phase-C shape [1]")
            if tuple(batch.action_timestep.shape) != (1,):
                raise ValueError("action_timestep must have exact Phase-C shape [1]")
            expected_video_timestep = 701.0 + 17.0 * batch.chunk_id
            expected_action_timestep = 503.0 + 19.0 * batch.chunk_id
            if not torch.equal(
                batch.video_timestep,
                torch.tensor(
                    [expected_video_timestep], dtype=torch.float32, device="cpu"
                ),
            ):
                raise ValueError("video_timestep differs from fixed Phase-C recipe")
            if not torch.equal(
                batch.action_timestep,
                torch.tensor(
                    [expected_action_timestep], dtype=torch.float32, device="cpu"
                ),
            ):
                raise ValueError("action_timestep differs from fixed Phase-C recipe")
            if bool(
                batch.noisy_video.masked_select(
                    (~batch.frame_valid_mask)[:, None, :, None, None]
                ).count_nonzero()
            ) or bool(
                batch.clean_video_target.masked_select(
                    (~batch.frame_valid_mask)[:, None, :, None, None]
                ).count_nonzero()
            ):
                raise ValueError("fixed-K video padding must be exact zero")
            if bool(
                batch.noisy_actions.masked_select(
                    ~batch.action_valid_mask.unsqueeze(-1)
                ).count_nonzero()
            ) or bool(
                batch.clean_action_target.masked_select(
                    ~batch.action_valid_mask.unsqueeze(-1)
                ).count_nonzero()
            ):
                raise ValueError("fixed-capacity action padding must be exact zero")

    def owned_clone(
        self,
        *,
        chunk_updates: Mapping[int, Mapping[str, Tensor]] | None = None,
    ) -> "ProductionPathMiniEpisode":
        updates = dict(chunk_updates or {})
        chunks = tuple(
            chunk.owned_clone(**dict(updates.get(chunk.chunk_id, {})))
            for chunk in self.chunks
        )
        return type(self)(
            layout=self.layout,
            chunks=chunks,
            recipe_digest=self.recipe_digest,
        )


def build_production_path_mini_episode_for_tests(
    spec: ProductionPathMiniSpec,
    layout: ChunkActionLayout,
) -> ProductionPathMiniEpisode:
    if not isinstance(spec, ProductionPathMiniSpec):
        raise TypeError("spec must be ProductionPathMiniSpec")
    if not isinstance(layout, ChunkActionLayout) or layout.synthetic_test_only is not True:
        raise TypeError("layout must be an exact synthetic ChunkActionLayout")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.input_seed)
    chunks: list[ProductionPathMiniChunk] = []
    for chunk in layout.chunks:
        noisy_video = torch.randn(
            spec.batch_size,
            spec.latent_channels,
            spec.chunk_size,
            1,
            1,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        frame_mask = torch.tensor(
            chunk.latent_valid_mask, dtype=torch.bool, device="cpu"
        ).view(1, -1)
        noisy_video = noisy_video.masked_fill(
            (~frame_mask)[:, None, :, None, None], 0.0
        )
        clean_video = torch.randn(
            spec.batch_size,
            spec.latent_channels,
            spec.chunk_size,
            1,
            1,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).masked_fill((~frame_mask)[:, None, :, None, None], 0.0)
        noisy_actions = torch.randn(
            spec.batch_size,
            chunk.action_slot_capacity,
            spec.action_dim,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        action_mask = torch.tensor(
            chunk.action_valid_mask, dtype=torch.bool, device="cpu"
        ).view(1, -1)
        noisy_actions = noisy_actions.masked_fill(
            ~action_mask.unsqueeze(-1), 0.0
        )
        clean_actions = torch.randn(
            spec.batch_size,
            chunk.action_slot_capacity,
            spec.action_dim,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).masked_fill(~action_mask.unsqueeze(-1), 0.0)
        context = torch.randn(
            spec.batch_size,
            2,
            spec.context_dim,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        proprio = torch.randn(
            spec.batch_size,
            spec.action_dim,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        chunks.append(
            ProductionPathMiniChunk(
                chunk_id=chunk.chunk_id,
                noisy_video=noisy_video,
                noisy_actions=noisy_actions,
                clean_video_target=clean_video,
                clean_action_target=clean_actions,
                frame_valid_mask=frame_mask,
                action_valid_mask=action_mask,
                context=context,
                context_mask=torch.ones(
                    spec.batch_size, 2, dtype=torch.bool, device="cpu"
                ),
                seq_lens=torch.full(
                    (spec.batch_size,), 2, dtype=torch.long, device="cpu"
                ),
                proprio_state=proprio,
                video_timestep=torch.full(
                    (spec.batch_size,),
                    701.0 + 17.0 * chunk.chunk_id,
                    dtype=torch.float32,
                    device="cpu",
                ),
                action_timestep=torch.full(
                    (spec.batch_size,),
                    503.0 + 19.0 * chunk.chunk_id,
                    dtype=torch.float32,
                    device="cpu",
                ),
            )
        )
    recipe_digest = canonical_sha256(
        {
            "input_seed": spec.input_seed,
            "layout_instance_digest": layout.layout_instance_digest,
            "schema": "cach.production_path_mini.input_recipe.v1",
            "tensor_digests": tuple(
                {
                    "clean_action": hc.tensor_digest(chunk.clean_action_target),
                    "clean_video": hc.tensor_digest(chunk.clean_video_target),
                    "context": hc.tensor_digest(chunk.context),
                    "noisy_action": hc.tensor_digest(chunk.noisy_actions),
                    "noisy_video": hc.tensor_digest(chunk.noisy_video),
                    "proprio": hc.tensor_digest(chunk.proprio_state),
                }
                for chunk in chunks
            ),
        }
    )
    return ProductionPathMiniEpisode(
        layout=layout,
        chunks=tuple(chunks),
        recipe_digest=recipe_digest,
    )


class _ProductionPathMiniGDNBlock(nn.Module):
    """Pure-Torch GDN proxy plus both registered temporal-conv cache fields."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_dim: int,
        staging_variant: CACHStagingVariant,
    ) -> None:
        super().__init__()
        self.operator = PureTorchChunkCausalGDNProxy(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            staging_variant=staging_variant,
        )
        self.hidden_dim = hidden_dim

    @staticmethod
    def _causal_context_mix(
        hidden: Tensor,
        context: Tensor,
        frame_valid_mask: Tensor,
        *,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        if (
            context.ndim != 3
            or context.shape[0] != hidden.shape[0]
            or context.shape[2] != hidden.shape[2]
            or context.shape[1] <= 0
        ):
            raise ValueError("temporal-conv context has the wrong shape")
        rolling = context
        outputs: list[Tensor] = []
        for index in range(hidden.shape[1]):
            valid = frame_valid_mask[:, index].view(-1, 1)
            mixed = hidden[:, index] + scale * rolling.mean(dim=1)
            mixed = torch.where(valid, mixed, torch.zeros_like(mixed))
            outputs.append(mixed)
            proposed = torch.cat([rolling[:, 1:], hidden[:, index : index + 1]], dim=1)
            rolling = torch.where(valid.unsqueeze(-1), proposed, rolling)
        return torch.stack(outputs, dim=1), rolling

    @classmethod
    def _ffn_context_mix(
        cls,
        hidden: Tensor,
        context: Tensor,
        frame_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        expected = (hidden.shape[0], hidden.shape[2], 1, 1)
        if tuple(context.shape) != expected:
            raise ValueError("FFN temporal-conv context has the wrong shape")
        token_context = context[:, :, :, 0].transpose(1, 2)
        output, next_context = cls._causal_context_mix(
            hidden,
            token_context,
            frame_valid_mask,
            scale=0.015625,
        )
        return output, next_context.transpose(1, 2).unsqueeze(-1)

    def scan_chunk(
        self,
        hidden: Tensor,
        *,
        main_s_kv: Tensor,
        main_s_z: Tensor,
        main_shortconv_left_context: Tensor,
        ffn_tconv_left_context: Tensor,
        frame_valid_mask: Tensor,
        action_embedding: Tensor | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        if main_s_kv.ndim != 4 or main_s_z.ndim != 4:
            raise ValueError("GDN mini cache must retain production rank-four state")
        mixed, next_short = self._causal_context_mix(
            hidden,
            main_shortconv_left_context,
            frame_valid_mask,
            scale=0.03125,
        )
        proxy_state = MinimalLayerState(
            main_s_kv=main_s_kv[:, 0],
            main_s_z=main_s_z[:, 0, :, 0],
        )
        output, next_proxy = self.operator.scan_chunk(
            mixed,
            proxy_state,
            frame_valid_mask=frame_valid_mask,
            action_embedding=action_embedding,
        )
        output, next_ffn = self._ffn_context_mix(
            output,
            ffn_tconv_left_context,
            frame_valid_mask,
        )
        return output, (
            next_proxy.main_s_kv.unsqueeze(1),
            next_proxy.main_s_z.unsqueeze(1).unsqueeze(-1),
            next_short,
            next_ffn,
        )


class _ProductionPathMiniDiT(nn.Module):
    """Tiny forward_long implementation consumed by the real SANA adapter."""

    def __init__(
        self,
        *,
        spec: ProductionPathMiniSpec,
        staging_variant: CACHStagingVariant,
    ) -> None:
        super().__init__()
        self.hidden_size = spec.hidden_dim
        self.staging_variant = require_cach_staging_variant(staging_variant)
        self.video_input_projection = nn.Linear(
            spec.latent_channels, spec.hidden_dim
        )
        self.video_timestep_projection = nn.Linear(1, spec.hidden_dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                _ProductionPathMiniGDNBlock(
                    hidden_dim=spec.hidden_dim,
                    action_dim=spec.hidden_dim,
                    staging_variant=self.staging_variant,
                )
                for _ in range(spec.depth)
            ]
        )
        self.video_output_projection = nn.Linear(
            spec.hidden_dim, spec.latent_channels
        )
        if self.staging_variant is CACHStagingVariant.CACH_A:
            self.delta_pose_embedder: nn.Module | None = nn.Sequential(
                nn.Linear(spec.action_dim, spec.hidden_dim),
                nn.SiLU(),
                nn.Linear(spec.hidden_dim, spec.hidden_dim),
            )
            self._cach_delta_pose_output_zero_verified = True
        else:
            self.delta_pose_embedder = None
            self._cach_delta_pose_output_zero_verified = False
        self.forward_long_call_count = 0

    @staticmethod
    def _empty_layer_state(
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, _, width = hidden.shape
        return (
            hidden.new_zeros((batch, 1, width, width)),
            hidden.new_zeros((batch, 1, width, 1)),
            hidden.new_zeros((batch, 3, width)),
            hidden.new_zeros((batch, width, 1, 1)),
        )

    @staticmethod
    def _state_from_slots(
        slots: list[object], hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if all(value is None for value in slots):
            return _ProductionPathMiniDiT._empty_layer_state(hidden)
        required = (slots[0], slots[1], slots[4], slots[9])
        if not all(isinstance(value, Tensor) for value in required):
            raise ValueError("populated mini vendor cache lacks an owned tensor")
        if type(slots[6]) is not float or slots[6] != 1.0:
            raise ValueError("mini GDN vendor cache lacks the exact type flag")
        if any(slots[index] is not None for index in (2, 3, 5, 7, 8)):
            raise ValueError("mini vendor cache populated an unowned slot")
        expected_shapes = (
            (hidden.shape[0], 1, hidden.shape[2], hidden.shape[2]),
            (hidden.shape[0], 1, hidden.shape[2], 1),
            (hidden.shape[0], 3, hidden.shape[2]),
            (hidden.shape[0], hidden.shape[2], 1, 1),
        )
        if tuple(tuple(value.shape) for value in required) != expected_shapes:
            raise ValueError("mini vendor cache shape differs from the pinned codec")
        return required  # type: ignore[return-value]

    @staticmethod
    def _write_slots(
        slots: list[object], state: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> None:
        slots[:] = [None] * 10
        slots[0], slots[1], slots[4], slots[9] = state
        slots[6] = 1.0

    def prepare_hidden(
        self,
        video: Tensor,
        timestep: Tensor,
        context: Tensor,
        context_mask: Tensor,
    ) -> Tensor:
        if video.ndim != 5 or video.shape[-2:] != (1, 1):
            raise ValueError("mini video must have exact [B,C,T,1,1] shape")
        tokens = video[:, :, :, 0, 0].transpose(1, 2)
        if context.ndim == 4:
            context = context[:, 0]
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError("mini context must be exact [B,L,D]")
        weights = context_mask.to(dtype=context.dtype).unsqueeze(-1)
        context_mean = (context * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        hidden = self.video_input_projection(tokens)
        hidden = hidden + self.video_timestep_projection(
            timestep.view(-1, 1) / 1000.0
        ).unsqueeze(1)
        return hidden + context_mean.unsqueeze(1)

    def action_embedding(self, delta_actions: Tensor | None) -> Tensor | None:
        if self.staging_variant is CACHStagingVariant.CACH_A:
            if delta_actions is None or self.delta_pose_embedder is None:
                raise ProductionPathMiniContractError(
                    "CACH-A mini requires explicit action conditioning"
                )
            return self.delta_pose_embedder(delta_actions)
        if delta_actions is not None:
            raise ProductionPathMiniContractError(
                "reference mini requires structural action bypass"
            )
        return None

    def project_video(self, hidden: Tensor) -> Tensor:
        tokens = self.video_output_projection(hidden)
        return tokens.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

    def forward_long(
        self,
        video: Tensor,
        timestep: Tensor,
        context: Tensor,
        *,
        mask: Tensor,
        start_f: int,
        end_f: int,
        frame_index: Tensor | None,
        kv_cache: list[list[object]],
        save_kv_cache: bool,
        bridge_layers: tuple[int, ...],
        bridge_out: dict[int, Tensor],
        use_gradient_checkpointing: bool,
        delta_actions: Tensor | None = None,
    ) -> tuple[Tensor, list[list[object]]]:
        del frame_index, save_kv_cache
        if use_gradient_checkpointing:
            raise ProductionPathMiniContractError(
                "Phase-C forbids gradient checkpointing/compile paths"
            )
        if video.device.type != "cpu" or video.dtype is not torch.float32:
            raise ProductionPathMiniContractError("mini forward_long escaped CPU/FP32")
        if end_f - start_f != video.shape[2]:
            raise ValueError("absolute latent interval differs from compact video")
        if type(kv_cache) is not list or len(kv_cache) != len(self.blocks):
            raise ValueError("mini vendor cache depth differs from executable graph")
        if set(bridge_layers) - set(range(len(self.blocks))):
            raise ValueError("mini bridge layer lies outside executable graph")
        self.forward_long_call_count += 1
        context_mask = mask.to(dtype=torch.bool)
        hidden = self.prepare_hidden(video, timestep, context, context_mask)
        valid = torch.ones(
            hidden.shape[:2], dtype=torch.bool, device=hidden.device
        )
        action_embedding = self.action_embedding(delta_actions)
        for index, block in enumerate(self.blocks):
            slots = kv_cache[index]
            if type(slots) is not list or len(slots) != 10:
                raise ValueError("mini vendor layer must have exact list[10] shape")
            state = self._state_from_slots(slots, hidden)
            hidden, state = block.scan_chunk(
                hidden,
                main_s_kv=state[0],
                main_s_z=state[1],
                main_shortconv_left_context=state[2],
                ffn_tconv_left_context=state[3],
                frame_valid_mask=valid,
                action_embedding=action_embedding,
            )
            self._write_slots(slots, state)
            if index in bridge_layers:
                bridge_out[index] = hidden
        return self.project_video(hidden), kv_cache


class ProductionPathMiniVideoBackbone(SanaVideoBackbone):
    """SANA public run_chunk adapter backed by a tiny pure-Torch forward_long."""

    def __init__(
        self,
        *,
        spec: ProductionPathMiniSpec,
        staging_variant: CACHStagingVariant,
    ) -> None:
        if torch.get_default_dtype() is not torch.float32:
            raise ProductionPathMiniContractError(
                "Phase-C mini requires global default dtype float32 before allocation"
            )
        nn.Module.__init__(self)
        variant = require_cach_staging_variant(staging_variant)
        with torch.device("cpu"):
            dit = _ProductionPathMiniDiT(spec=spec, staging_variant=variant)
        self._pipe = SimpleNamespace(
            dit=dit,
            config={
                "delta_pose_additive_dim": spec.action_dim,
                "use_delta_pose_additive": variant is CACHStagingVariant.CACH_A,
            },
        )
        self.add_module("dit", dit)
        self._cached_streaming_enabled = True
        self._dtype = torch.float32
        self._device = torch.device("cpu")
        self._dit_patch_size = (1, 1, 1)
        self._temporal_compression = spec.temporal_compression
        self._causal_temporal = True
        self._spatial_compression = 1
        super().train(False)
        self._assert_closed_runtime()

    @classmethod
    def from_pretrained(
        cls, *_args: object, **_kwargs: object
    ) -> "ProductionPathMiniVideoBackbone":
        raise ProductionPathMiniContractError(
            "Phase-C mini forbids pipeline/checkpoint construction"
        )

    @classmethod
    def from_mini_config(
        cls, **_mini_kwargs: object
    ) -> "ProductionPathMiniVideoBackbone":
        raise ProductionPathMiniContractError(
            "Phase-C mini has one sealed synthetic constructor"
        )

    def _assert_closed_runtime(self) -> None:
        if self.training or any(module.training for module in self.modules()):
            raise ProductionPathMiniContractError(
                "Phase-C video adapter escaped eval-only mode"
            )
        if self._device != torch.device("cpu") or self._dtype is not torch.float32:
            raise ProductionPathMiniContractError(
                "Phase-C video adapter escaped CPU/FP32"
            )
        for name, parameter in self.named_parameters():
            _require_cpu_fp32(parameter, f"video_backbone.{name}")
        for name, buffer in self.named_buffers():
            if buffer.device.type != "cpu":
                raise ProductionPathMiniContractError(
                    f"video_backbone buffer {name} escaped CPU"
                )

    def _apply(self, _fn: Callable[[Tensor], Tensor], recurse: bool = True) -> nn.Module:
        del recurse
        raise ProductionPathMiniContractError(
            "Phase-C mini forbids device/dtype/module transforms"
        )

    def load_state_dict(self, *_args: object, **_kwargs: object) -> Any:
        raise ProductionPathMiniContractError(
            "Phase-C mini forbids checkpoint/state loading"
        )

    def train(self, mode: bool = True) -> "ProductionPathMiniVideoBackbone":
        if mode is not False:
            raise ProductionPathMiniContractError(
                "Phase-C video adapter is update-free and eval-only"
            )
        return super().train(False)

    def run_chunk(self, *args: Any, **kwargs: Any) -> tuple[Tensor, dict, list]:
        self._assert_closed_runtime()
        return super().run_chunk(*args, **kwargs)


@dataclass(frozen=True)
class ProductionPathMiniFullResult:
    video_predictions: tuple[Tensor, ...] = field(repr=False, compare=False)
    action_predictions: tuple[Tensor, ...] = field(repr=False, compare=False)
    final_layer_payloads: tuple[hc.StagedLayerPayload, ...] = field(
        repr=False, compare=False
    )
    committed_actions: Tensor = field(repr=False, compare=False)


@dataclass(frozen=True)
class ProductionPathMiniInventoryEntry:
    name: str
    tensor_kind: str
    shape: tuple[int, ...]
    dtype: str
    digest: str
    initializer_role: str
    requires_grad: bool

    def to_manifest(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "dtype": self.dtype,
            "initializer_role": self.initializer_role,
            "name": self.name,
            "requires_grad": self.requires_grad,
            "shape": self.shape,
            "tensor_kind": self.tensor_kind,
        }


class ProductionPathMiniArm(nn.Module):
    """One independent REF-GDN-CORRECTED or CACH-A Phase-C graph."""

    def __init__(
        self,
        *,
        spec: ProductionPathMiniSpec,
        layout: ChunkActionLayout,
        staging_variant: CACHStagingVariant,
    ) -> None:
        super().__init__()
        if not isinstance(spec, ProductionPathMiniSpec):
            raise TypeError("spec must be ProductionPathMiniSpec")
        if not isinstance(layout, ChunkActionLayout) or layout.synthetic_test_only is not True:
            raise TypeError("mini arm requires a synthetic typed layout")
        canonical_layout = build_production_path_mini_layout_for_tests(spec)
        if (
            layout.layout_spec_sha256 != canonical_layout.layout_spec_sha256
            or layout.layout_instance_digest
            != canonical_layout.layout_instance_digest
            or layout.valid_raw_count != 57
            or len(layout.chunks) != 3
        ):
            raise ProductionPathMiniContractError(
                "mini arm requires the canonical fixed 57-raw/3-chunk layout"
            )
        if torch.get_default_dtype() is not torch.float32:
            raise ProductionPathMiniContractError(
                "Phase-C mini requires global default dtype float32 before allocation"
            )
        self.spec = spec
        self.layout = layout
        self.staging_variant = require_cach_staging_variant(staging_variant)
        with torch.device("cpu"):
            self.cach_core = CACHNumericalCore(action_dim=spec.action_dim)
            self.video_backbone = ProductionPathMiniVideoBackbone(
                spec=spec,
                staging_variant=self.staging_variant,
            )
            self.action_backbone = ActionDiT(
                action_dim=spec.action_dim,
                dim=spec.hidden_dim,
                ffn_dim=spec.hidden_dim * 2,
                num_heads=1,
                num_layers=spec.depth,
                video_dim=spec.hidden_dim,
                bridge_layers=tuple(range(spec.depth)),
                variant="joint_cross_attn",
                attn_head_dim=spec.hidden_dim,
                text_dim=spec.context_dim,
                freq_dim=max(4, spec.hidden_dim),
                max_action_len=128,
                attn_kernel="softmax",
            )
            self.proprio_context_encoder = nn.Linear(
                spec.action_dim, spec.context_dim
            )
        self.registry = hc.LayerRegistry(
            layers=tuple(
                hc.LayerSpec(
                    layer_index=index,
                    kind=hc.LayerKind.GDN_FULL_HISTORY,
                    operator_class=(
                        "sana_wam.model.cach_production_path_mini."
                        "_ProductionPathMiniGDNBlock"
                    ),
                    camera_enabled=False,
                    main_shortconv_enabled=True,
                    ffn_tconv_enabled=True,
                )
                for index in range(spec.depth)
            )
        )
        hc.validate_cach_a_registry(self.registry)
        self._initialize_named_parameters()
        self.eval()
        self._assert_closed_runtime()
        self._initial_tensor_manifest = MappingProxyType(
            {
                name: (tensor._version, hc.tensor_digest(tensor))
                for name, tensor in (
                    list(self.named_parameters()) + list(self.named_buffers())
                )
            }
        )

    def _apply(self, _fn: Callable[[Tensor], Tensor], recurse: bool = True) -> nn.Module:
        del recurse
        raise ProductionPathMiniContractError(
            "Phase-C mini forbids device/dtype/module transforms"
        )

    def load_state_dict(self, *_args: object, **_kwargs: object) -> Any:
        raise ProductionPathMiniContractError(
            "Phase-C mini forbids checkpoint/state loading"
        )

    def requires_grad_(self, _requires_grad: bool = True) -> "ProductionPathMiniArm":
        raise ProductionPathMiniContractError(
            "Phase-C mini parameter classification is immutable"
        )

    def train(self, mode: bool = True) -> "ProductionPathMiniArm":
        if mode is not False:
            raise ProductionPathMiniContractError(
                "Phase-C mini is update-free and eval-only"
            )
        return super().train(False)

    @staticmethod
    def _candidate_only_name(name: str) -> bool:
        return (
            name.startswith("video_backbone.dit.delta_pose_embedder.")
            or ".operator.action_output_projection." in name
        )

    @staticmethod
    def _zero_adapter_name(name: str) -> bool:
        return ".operator.action_output_projection." in name

    def _initialize_named_parameters(self) -> None:
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name == "cach_core.no_action_slot" or self._zero_adapter_name(name):
                    parameter.zero_()
                    continue
                if name.endswith(".bias"):
                    parameter.zero_()
                    continue
                candidate_only = self._candidate_only_name(name)
                role = (
                    "candidate_operator_named_uniform_v1"
                    if candidate_only
                    else "shared_named_uniform_v1"
                )
                seed = (
                    self.spec.candidate_init_seed
                    if candidate_only
                    else self.spec.shared_init_seed
                )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(_named_seed(seed, role, name))
                fan_in = parameter.shape[-1] if parameter.ndim >= 2 else 1
                bound = 0.25 / math.sqrt(float(fan_in))
                parameter.uniform_(-bound, bound, generator=generator)

    def _assert_closed_runtime(self) -> None:
        if torch.cuda.is_initialized():
            raise ProductionPathMiniContractError(
                "CUDA was initialized before Phase-C mini construction"
            )
        functorch_wrapped = False
        if self.training or any(module.training for module in self.modules()):
            raise ProductionPathMiniContractError(
                "Phase-C mini graph escaped eval-only mode"
            )
        for name, parameter in self.named_parameters():
            _require_cpu_fp32(parameter, name)
            wrapped = torch._C._functorch.is_functorch_wrapped_tensor(parameter)
            functorch_wrapped = functorch_wrapped or wrapped
            if not wrapped and parameter.requires_grad is not True:
                raise ProductionPathMiniContractError(
                    f"parameter {name} changed its trainable classification"
                )
        for name, buffer in self.named_buffers():
            if buffer.device.type != "cpu":
                raise ProductionPathMiniContractError(f"buffer {name} escaped CPU")
        self.video_backbone._assert_closed_runtime()
        if not _is_exact_zero(self.cach_core.no_action_slot):
            raise ProductionPathMiniContractError("model-owned NO_ACTION is not zero")
        if self.staging_variant is CACHStagingVariant.CACH_A:
            zero_names = [
                name
                for name, _ in self.named_parameters()
                if self._zero_adapter_name(name)
            ]
            if len(zero_names) != 2 * self.spec.depth:
                raise ProductionPathMiniContractError(
                    "candidate zero-adapter inventory must contain 40 tensors"
                )
            inventory = dict(self.named_parameters())
            if any(not _is_exact_zero(inventory[name]) for name in zero_names):
                raise ProductionPathMiniContractError(
                    "candidate adapter output seam is not exact zero"
                )
        initial = getattr(self, "_initial_tensor_manifest", None)
        if initial is not None and not functorch_wrapped:
            current = {
                name: (tensor._version, hc.tensor_digest(tensor))
                for name, tensor in (
                    list(self.named_parameters()) + list(self.named_buffers())
                )
            }
            if current != dict(initial):
                raise ProductionPathMiniContractError(
                    "Phase-C mini parameter/buffer state changed"
                )

    def _assert_episode_runtime(self, episode: ProductionPathMiniEpisode) -> None:
        self._assert_closed_runtime()
        if not isinstance(episode, ProductionPathMiniEpisode):
            raise TypeError("episode must be ProductionPathMiniEpisode")
        if episode.layout is not self.layout:
            raise ProductionPathMiniContractError(
                "execution requires the exact arm-owned layout instance"
            )
        episode.__post_init__()
        for batch in episode.chunks:
            for name in (
                "noisy_video",
                "noisy_actions",
                "clean_video_target",
                "clean_action_target",
                "context",
                "proprio_state",
                "video_timestep",
                "action_timestep",
            ):
                _require_cpu_fp32(getattr(batch, name), name)
            if batch.frame_valid_mask.device.type != "cpu":
                raise ProductionPathMiniContractError(
                    "frame_valid_mask escaped CPU"
                )
            if batch.action_valid_mask.device.type != "cpu":
                raise ProductionPathMiniContractError(
                    "action_valid_mask escaped CPU"
                )

    def graph_manifest(self) -> dict[str, object]:
        return {
            "afcc_reachable": False,
            "anchors": 0,
            "attnres_modules": 0,
            "complete_2b": False,
            "depth": self.spec.depth,
            "device": "cpu",
            "dtype": "float32",
            "formal_admission": False,
            "layer_registry_digest": self.registry.manifest_digest,
            "public_action_interface": (
                "ActionDiT.forward_with_bridge_tuple/absolute_rope"
            ),
            "public_video_interface": "SanaVideoBackbone.run_chunk",
            "revision": _REVISION,
            "scientific_result": False,
            "self_forcing_modules": 0,
            "staging_variant": self.staging_variant.value,
            "synthetic_test_only": True,
            "vendor_gdn_numerical_parity": False,
        }

    def _full_context(self, batch: ProductionPathMiniChunk) -> tuple[Tensor, Tensor]:
        token = self.proprio_context_encoder(batch.proprio_state)
        return (
            torch.cat([batch.context, token.unsqueeze(1)], dim=1),
            torch.cat(
                [
                    batch.context_mask,
                    torch.ones(
                        batch.context.shape[0],
                        1,
                        dtype=torch.bool,
                        device=batch.context.device,
                    ),
                ],
                dim=1,
            ),
        )

    def _run_full_video(
        self,
        episode: ProductionPathMiniEpisode,
    ) -> tuple[
        tuple[Tensor, ...],
        tuple[dict[int, Tensor], ...],
        tuple[hc.StagedLayerPayload, ...],
        Tensor,
    ]:
        self._assert_episode_runtime(episode)
        noisy_hidden_chunks: list[Tensor] = []
        clean_hidden_chunks: list[Tensor] = []
        valid_masks: list[Tensor] = []
        noisy_action_embeddings: list[Tensor | None] = []
        clean_action_embeddings: list[Tensor | None] = []
        history = episode.chunks[0].clean_action_target.new_empty(
            (self.spec.batch_size, 0, self.spec.action_dim)
        )
        dit = self.video_backbone._dit
        for batch, chunk in zip(episode.chunks, self.layout.chunks):
            video_plan = FixedKPrefixPlan.from_mask(batch.frame_valid_mask)
            compact_noisy_video = video_plan.compact_video(batch.noisy_video)
            compact_clean_video = video_plan.compact_video(
                batch.clean_video_target
            )
            full_context, full_mask = self._full_context(batch)
            noisy_hidden_chunks.append(
                dit.prepare_hidden(
                    compact_noisy_video,
                    batch.video_timestep,
                    full_context.unsqueeze(1),
                    full_mask,
                )
            )
            clean_hidden_chunks.append(
                dit.prepare_hidden(
                    compact_clean_video,
                    batch.video_timestep.new_zeros(
                        (batch.video_timestep.shape[0],)
                    ),
                    full_context.unsqueeze(1),
                    full_mask,
                )
            )
            valid_masks.append(
                torch.ones(
                    self.spec.batch_size,
                    chunk.valid_latent_count,
                    dtype=torch.bool,
                    device=batch.frame_valid_mask.device,
                )
            )
            if self.staging_variant is CACHStagingVariant.CACH_A:
                noisy_condition = reduce_end_of_bin_action_condition(
                    batch.noisy_actions,
                    committed_actions=(None if chunk.chunk_id == 0 else history),
                    chunk=chunk,
                    no_action_slot=self.cach_core.no_action_slot,
                )
                clean_condition = reduce_end_of_bin_action_condition(
                    batch.clean_action_target,
                    committed_actions=(None if chunk.chunk_id == 0 else history),
                    chunk=chunk,
                    no_action_slot=self.cach_core.no_action_slot,
                )
                noisy_action_embeddings.append(
                    dit.action_embedding(
                        video_plan.compact_condition(noisy_condition.condition)
                    )
                )
                clean_action_embeddings.append(
                    dit.action_embedding(
                        video_plan.compact_condition(clean_condition.condition)
                    )
                )
            else:
                noisy_action_embeddings.append(dit.action_embedding(None))
                clean_action_embeddings.append(dit.action_embedding(None))
            history = torch.cat(
                [
                    history,
                    batch.clean_action_target[:, : chunk.valid_action_count],
                ],
                dim=1,
            )

        bridges: list[dict[int, Tensor]] = [dict() for _ in episode.chunks]
        final_states: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for layer_index, block in enumerate(dit.blocks):
            clean_layer_state = dit._empty_layer_state(clean_hidden_chunks[0])
            next_noisy_hidden_chunks: list[Tensor] = []
            next_clean_hidden_chunks: list[Tensor] = []
            for chunk_index, (
                noisy_hidden,
                clean_hidden,
                valid,
                noisy_action_embedding,
                clean_action_embedding,
            ) in enumerate(
                zip(
                    noisy_hidden_chunks,
                    clean_hidden_chunks,
                    valid_masks,
                    noisy_action_embeddings,
                    clean_action_embeddings,
                )
            ):
                noisy_state_snapshot = tuple(
                    value.clone() for value in clean_layer_state
                )
                noisy_hidden, _ = block.scan_chunk(
                    noisy_hidden,
                    main_s_kv=noisy_state_snapshot[0],
                    main_s_z=noisy_state_snapshot[1],
                    main_shortconv_left_context=noisy_state_snapshot[2],
                    ffn_tconv_left_context=noisy_state_snapshot[3],
                    frame_valid_mask=valid,
                    action_embedding=noisy_action_embedding,
                )
                clean_hidden, clean_layer_state = block.scan_chunk(
                    clean_hidden,
                    main_s_kv=clean_layer_state[0],
                    main_s_z=clean_layer_state[1],
                    main_shortconv_left_context=clean_layer_state[2],
                    ffn_tconv_left_context=clean_layer_state[3],
                    frame_valid_mask=valid,
                    action_embedding=clean_action_embedding,
                )
                next_noisy_hidden_chunks.append(noisy_hidden)
                next_clean_hidden_chunks.append(clean_hidden)
                bridges[chunk_index][layer_index] = noisy_hidden
            noisy_hidden_chunks = next_noisy_hidden_chunks
            clean_hidden_chunks = next_clean_hidden_chunks
            final_states.append(clean_layer_state)

        video_predictions: list[Tensor] = []
        for batch, hidden in zip(episode.chunks, noisy_hidden_chunks):
            plan = FixedKPrefixPlan.from_mask(batch.frame_valid_mask)
            video_predictions.append(plan.restore_video(dit.project_video(hidden)))
        payloads = tuple(
            hc.StagedLayerPayload(
                layer_index=index,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                tensors={
                    "ffn_tconv_left_context": state[3],
                    "main_s_kv": state[0],
                    "main_s_z": state[1],
                    "main_shortconv_left_context": state[2],
                },
            )
            for index, state in enumerate(final_states)
        )
        return tuple(video_predictions), tuple(bridges), payloads, history

    def run_full_sequence(
        self, episode: ProductionPathMiniEpisode
    ) -> ProductionPathMiniFullResult:
        """Independent layer-major full-layout path; returns no commit authority."""

        self._assert_episode_runtime(episode)
        video_predictions, bridges, payloads, history = self._run_full_video(episode)
        action_predictions: list[Tensor] = []
        for batch, chunk, chunk_bridges in zip(
            episode.chunks, self.layout.chunks, bridges
        ):
            action_plan = FixedKPrefixPlan.from_mask(batch.action_valid_mask)
            compact_actions = action_plan.compact_condition(batch.noisy_actions)
            rope_positions = torch.arange(
                chunk.action_rope_start,
                chunk.action_rope_end,
                dtype=torch.long,
                device=batch.noisy_actions.device,
            )
            action_freqs = self.action_backbone._get_rope_freqs_at(rope_positions)
            full_context, full_mask = self._full_context(batch)
            compact_prediction = self.action_backbone.forward_with_bridge_tuple(
                compact_actions,
                self.action_backbone.bridge_tuple_from_dict(chunk_bridges),
                batch.action_timestep,
                context=full_context,
                context_mask=full_mask,
                action_freqs=action_freqs,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
            action_predictions.append(
                action_plan.restore_token_sequence(compact_prediction)
            )
        return ProductionPathMiniFullResult(
            video_predictions=video_predictions,
            action_predictions=tuple(action_predictions),
            final_layer_payloads=payloads,
            committed_actions=history,
        )

    def forward(self, episode: ProductionPathMiniEpisode) -> Tensor:
        """Video-only full path used for registered parameter/input JVP probes."""

        self._assert_episode_runtime(episode)
        outputs, _, _, _ = self._run_full_video(episode)
        return torch.cat([output.flatten() for output in outputs])


def _tensor_inventory(module: nn.Module) -> dict[str, tuple[str, Tensor]]:
    values = {
        name: ("parameter", tensor) for name, tensor in module.named_parameters()
    }
    for name, tensor in module.named_buffers():
        if name in values:
            raise ProductionPathMiniContractError(f"duplicate tensor name: {name}")
        values[name] = ("buffer", tensor)
    return values


def _inventory_role(name: str, tensor_kind: str, *, candidate_only: bool) -> str:
    if tensor_kind == "buffer":
        return "shared_constructor_buffer_v1"
    if candidate_only:
        if ProductionPathMiniArm._zero_adapter_name(name):
            return "candidate_exact_zero_output_adapter_v1"
        if name.endswith(".bias"):
            return "candidate_operator_exact_zero_bias_v1"
        return "candidate_operator_named_uniform_v1"
    if name == "cach_core.no_action_slot":
        return "shared_model_owned_exact_zero_no_action_v1"
    if name.endswith(".bias"):
        return "shared_exact_zero_bias_v1"
    return "shared_named_uniform_v1"


def _inventory_entry(
    name: str,
    tensor_kind: str,
    tensor: Tensor,
    *,
    candidate_only: bool,
) -> ProductionPathMiniInventoryEntry:
    return ProductionPathMiniInventoryEntry(
        name=name,
        tensor_kind=tensor_kind,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        digest=hc.tensor_digest(tensor),
        initializer_role=_inventory_role(
            name, tensor_kind, candidate_only=candidate_only
        ),
        requires_grad=bool(tensor.requires_grad),
    )


@dataclass(frozen=True)
class ProductionPathMiniPair:
    spec: ProductionPathMiniSpec
    layout: ChunkActionLayout
    reference: ProductionPathMiniArm
    candidate: ProductionPathMiniArm
    shared_inventory: tuple[ProductionPathMiniInventoryEntry, ...]
    candidate_only_inventory: tuple[ProductionPathMiniInventoryEntry, ...]
    pair_digest: str

    @property
    def zero_adapter_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            entry.name
            for entry in self.candidate_only_inventory
            if entry.initializer_role == "candidate_exact_zero_output_adapter_v1"
        )

    def _digest_payload(self) -> dict[str, object]:
        return {
            "candidate_graph": self.candidate.graph_manifest(),
            "candidate_only_inventory": tuple(
                entry.to_manifest() for entry in self.candidate_only_inventory
            ),
            "layout_instance_digest": self.layout.layout_instance_digest,
            "reference_graph": self.reference.graph_manifest(),
            "schema": "cach.production_path_mini.pair.v1",
            "shared_inventory": tuple(
                entry.to_manifest() for entry in self.shared_inventory
            ),
            "spec": self.spec.to_manifest(),
            "storage_alias_free": True,
        }

    def assert_integrity(self) -> None:
        self.reference._assert_closed_runtime()
        self.candidate._assert_closed_runtime()
        reference = _tensor_inventory(self.reference)
        candidate = _tensor_inventory(self.candidate)
        reference_only = sorted(reference.keys() - candidate.keys())
        candidate_only = sorted(candidate.keys() - reference.keys())
        if reference_only:
            raise ProductionPathMiniContractError(
                f"unexpected reference-only tensors: {reference_only}"
            )
        expected_candidate_only = sorted(
            name
            for name in candidate
            if ProductionPathMiniArm._candidate_only_name(name)
        )
        if candidate_only != expected_candidate_only:
            raise ProductionPathMiniContractError(
                "candidate-only inventory escaped the registered action seam"
            )
        shared_names = sorted(reference.keys() & candidate.keys())
        for name in shared_names:
            ref_kind, ref_tensor = reference[name]
            candidate_kind, candidate_tensor = candidate[name]
            if (
                ref_kind != candidate_kind
                or not _raw_tensor_equal(ref_tensor, candidate_tensor)
            ):
                raise ProductionPathMiniContractError(
                    f"shared named initialization differs at {name}"
                )
            if ref_tensor.untyped_storage().data_ptr() == (
                candidate_tensor.untyped_storage().data_ptr()
            ):
                raise ProductionPathMiniContractError(
                    f"reference/candidate storage aliases at {name}"
                )
        expected_shared = tuple(
            _inventory_entry(
                name,
                reference[name][0],
                reference[name][1],
                candidate_only=False,
            )
            for name in shared_names
        )
        expected_candidate = tuple(
            _inventory_entry(
                name,
                candidate[name][0],
                candidate[name][1],
                candidate_only=True,
            )
            for name in candidate_only
        )
        if (
            expected_shared != self.shared_inventory
            or expected_candidate != self.candidate_only_inventory
        ):
            raise ProductionPathMiniContractError("pair tensor inventory changed")
        if len(self.zero_adapter_parameter_names) != 2 * self.spec.depth:
            raise ProductionPathMiniContractError(
                "pair must register exactly 40 zero adapter tensors"
            )
        if canonical_sha256(self._digest_payload()) != self.pair_digest:
            raise ProductionPathMiniContractError("pair digest changed")


def _validate_build_config(requested_config: Mapping[str, object] | None) -> None:
    if requested_config is None:
        return
    if not isinstance(requested_config, Mapping):
        raise TypeError("requested_config must be a mapping")
    keys = set(requested_config)
    forbidden = sorted(keys & _FORBIDDEN_CONFIG_KEYS)
    if forbidden:
        raise ProductionPathMiniContractError(
            f"forbidden Phase-C build keys: {forbidden}"
        )
    unknown = sorted(keys - set(_ALLOWED_CONFIG))
    if unknown:
        raise ProductionPathMiniContractError(
            f"unknown Phase-C build keys: {unknown}"
        )
    for key, expected in _ALLOWED_CONFIG.items():
        if key in requested_config and requested_config[key] != expected:
            raise ProductionPathMiniContractError(
                f"Phase-C build value differs for {key}"
            )


def build_production_path_mini_pair_for_tests(
    *,
    spec: ProductionPathMiniSpec | None = None,
    requested_config: Mapping[str, object] | None = None,
) -> ProductionPathMiniPair:
    """Only public constructor for the bounded Phase-C pair.

    Config validation runs before either arm is allocated.  The function has no
    externally forgeable capability token and is intentionally named for tests.
    """

    _validate_build_config(requested_config)
    resolved_spec = ProductionPathMiniSpec() if spec is None else spec
    if not isinstance(resolved_spec, ProductionPathMiniSpec):
        raise TypeError("spec must be ProductionPathMiniSpec")
    if torch.cuda.is_initialized():
        raise ProductionPathMiniContractError(
            "Phase-C refuses construction after CUDA initialization"
        )
    layout = build_production_path_mini_layout_for_tests(resolved_spec)
    reference = ProductionPathMiniArm(
        spec=resolved_spec,
        layout=layout,
        staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
    )
    candidate = ProductionPathMiniArm(
        spec=resolved_spec,
        layout=layout,
        staging_variant=CACHStagingVariant.CACH_A,
    )
    ref_inventory = _tensor_inventory(reference)
    candidate_inventory = _tensor_inventory(candidate)
    shared_names = sorted(ref_inventory.keys() & candidate_inventory.keys())
    candidate_only = sorted(candidate_inventory.keys() - ref_inventory.keys())
    shared_entries = tuple(
        _inventory_entry(
            name,
            ref_inventory[name][0],
            ref_inventory[name][1],
            candidate_only=False,
        )
        for name in shared_names
    )
    candidate_entries = tuple(
        _inventory_entry(
            name,
            candidate_inventory[name][0],
            candidate_inventory[name][1],
            candidate_only=True,
        )
        for name in candidate_only
    )
    pair = ProductionPathMiniPair(
        spec=resolved_spec,
        layout=layout,
        reference=reference,
        candidate=candidate,
        shared_inventory=shared_entries,
        candidate_only_inventory=candidate_entries,
        pair_digest="pending",
    )
    object.__setattr__(pair, "pair_digest", canonical_sha256(pair._digest_payload()))
    pair.assert_integrity()
    return pair


@dataclass(frozen=True)
class ProductionPathMiniCommitReceipt:
    commit_id: str
    staging_variant: CACHStagingVariant
    revision_before: int
    revision_after: int
    action_cursor_before: int
    action_cursor_after: int
    state_manifest_before: str
    state_manifest_after: str
    action_history_digest_before: str | None
    action_history_digest_after: str
    chunk_id: int
    ephemeral: bool = True
    durable_publication: bool = False
    controller_commit: bool = False


@dataclass(frozen=True)
class ProductionPathMiniResetReceipt:
    reset_id: str
    revision_before: int
    state_manifest_before: str
    state_manifest_after: str
    action_cursor_before: int
    action_cursor_after: int = 0
    ephemeral: bool = True
    durable_publication: bool = False


class ProductionPathMiniEphemeralStateOwner:
    """Single-pointer in-memory owner aligned to Stage2B typed state semantics."""

    def __init__(
        self,
        *,
        registry: hc.LayerRegistry,
        episode_id: str,
        layout: ChunkActionLayout,
        staging_variant: CACHStagingVariant,
    ) -> None:
        if not isinstance(registry, hc.LayerRegistry):
            raise TypeError("registry must be LayerRegistry")
        hc.validate_cach_a_registry(registry)
        if len(registry.layers) != 20 or any(
            layer.layer_index != index
            or layer.kind is not hc.LayerKind.GDN_FULL_HISTORY
            or layer.operator_class
            != (
                "sana_wam.model.cach_production_path_mini."
                "_ProductionPathMiniGDNBlock"
            )
            or layer.camera_enabled
            or not layer.main_shortconv_enabled
            or not layer.ffn_tconv_enabled
            for index, layer in enumerate(registry.layers)
        ):
            raise ProductionPathMiniContractError(
                "ephemeral owner requires the exact 20-layer mini registry"
            )
        if not isinstance(layout, ChunkActionLayout) or layout.synthetic_test_only is not True:
            raise TypeError("ephemeral owner requires a synthetic typed layout")
        canonical_layout = build_production_path_mini_layout_for_tests(
            ProductionPathMiniSpec()
        )
        if (
            layout.layout_spec_sha256 != canonical_layout.layout_spec_sha256
            or layout.layout_instance_digest
            != canonical_layout.layout_instance_digest
            or layout.valid_raw_count != 57
            or len(layout.chunks) != 3
        ):
            raise ProductionPathMiniContractError(
                "ephemeral owner requires the canonical fixed layout"
            )
        self._registry = registry
        self._layout = layout
        self._staging_variant = require_cach_staging_variant(staging_variant)
        self._state = h2.Stage2BTemporalState.empty(
            episode_id=episode_id,
            episode_epoch=0,
            layout=layout,
            registry=registry,
            staging_variant=self._staging_variant,
        )
        self._state_manifest_digest = self._state.state_manifest_digest
        self._completed_ids: set[str] = set()
        self._pending: tuple[str, object] | None = None
        self._lock = threading.RLock()
        self.pointer_swap_count = 0

    @property
    def state(self) -> h2.Stage2BTemporalState:
        with self._lock:
            self._assert_live_integrity()
            return self._state.clone()

    @property
    def live_pointer_identity_for_tests(self) -> int:
        return id(self._state)

    def _assert_live_integrity(self) -> None:
        if self._state.state_manifest_digest != self._state_manifest_digest:
            raise ProductionPathMiniContractError(
                "ephemeral live state mutated outside a pointer swap"
            )

    def _validate_staged_payloads(
        self, payloads: tuple[hc.StagedLayerPayload, ...]
    ) -> None:
        if len(payloads) != 20:
            raise ProductionPathMiniContractError(
                "mini staging must return exactly 20 layer payloads"
            )
        expected_shapes = {
            "ffn_tconv_left_context": (1, 4, 1, 1),
            "main_s_kv": (1, 1, 4, 4),
            "main_s_z": (1, 1, 4, 1),
            "main_shortconv_left_context": (1, 3, 4),
        }
        for index, payload in enumerate(payloads):
            if (
                type(payload) is not hc.StagedLayerPayload
                or payload.layer_index != index
                or payload.kind is not hc.LayerKind.GDN_FULL_HISTORY
                or set(payload.tensors) != set(expected_shapes)
            ):
                raise ProductionPathMiniContractError(
                    "mini staged payload registry/field identity differs"
                )
            for name, expected_shape in expected_shapes.items():
                value = payload.tensors[name]
                _require_cpu_fp32(value, f"staged layer {index} {name}")
                if tuple(value.shape) != expected_shape:
                    raise ProductionPathMiniContractError(
                        f"staged layer {index} {name} shape differs"
                    )

    def snapshot_for_denoise(
        self,
        *,
        expected_episode_id: str,
        expected_episode_epoch: int,
        expected_revision: int,
        expected_next_chunk_id: int,
    ) -> h2.Stage2BDenoiseReadView:
        with self._lock:
            self._assert_live_integrity()
            state = self._state
            if (
                state.episode_id != expected_episode_id
                or state.episode_epoch != expected_episode_epoch
                or state.revision != expected_revision
                or state.next_chunk_id != expected_next_chunk_id
            ):
                raise hc.CacheContractError(
                    "CACHE_CONTENT_TIME_MISMATCH",
                    "mini denoise request differs from live state",
                )
            history_view = (
                None
                if state.committed_action_history is None
                else state.committed_action_history.view_for_chunk(
                    chunk_id=state.next_chunk_id,
                    action_start=state.action_cursor,
                )
            )
            return h2.Stage2BDenoiseReadView(
                episode_id=state.episode_id,
                episode_epoch=state.episode_epoch,
                revision=state.revision,
                next_chunk_id=state.next_chunk_id,
                staging_variant=state.staging_variant,
                layout_instance_digest=state.layout_instance_digest,
                state_manifest_digest=state.state_manifest_digest,
                previous_committed_action_history=history_view,
                _state_identity=id(state),
                _state_snapshot=state.clone(),
            )

    def finish_denoise(
        self,
        *,
        read_view: h2.Stage2BDenoiseReadView,
        scratch: hc.CacheScratch,
    ) -> None:
        if type(read_view) is not h2.Stage2BDenoiseReadView:
            raise TypeError("read_view must be the exact Stage2B type")
        read_view.assert_integrity()
        scratch.assert_unchanged()
        with self._lock:
            self._assert_live_integrity()
            if (
                id(self._state) != read_view._state_identity
                or self._state.state_manifest_digest
                != read_view.state_manifest_digest
                or scratch._source_state_identity != read_view._state_identity
                or scratch.source_state_manifest_digest
                != read_view.state_manifest_digest
            ):
                raise hc.CacheContractError(
                    "COMMIT_STALE_REVISION", "mini denoise view became stale"
                )

    def _validate_next_content_time(
        self,
        state: h2.Stage2BTemporalState,
        content_time: hc.ContentTime,
    ) -> None:
        if state.next_chunk_id >= len(self._layout.chunks):
            raise hc.CacheContractError(
                "COMMIT_OUT_OF_ORDER", "mini commit exceeds layout chunk count"
            )
        if (
            content_time.episode_id != state.episode_id
            or content_time.episode_epoch != state.episode_epoch
            or content_time.layout_spec_sha256 != state.layout_spec_sha256
            or content_time.layout_instance_digest != state.layout_instance_digest
            or state.layout_spec_sha256 != self._layout.layout_spec_sha256
            or state.layout_instance_digest != self._layout.layout_instance_digest
        ):
            raise hc.CacheContractError(
                "CACHE_CONTENT_TIME_MISMATCH",
                "mini commit content-time differs from live episode/layout",
            )
        if content_time.chunk_id != state.next_chunk_id:
            raise hc.CacheContractError(
                "COMMIT_OUT_OF_ORDER", "mini commit is not the live next chunk"
            )
        if content_time.action_token_start != state.action_cursor:
            raise hc.CacheContractError(
                "COMMIT_OUT_OF_ORDER", "mini action cursor is not contiguous"
            )
        bound_chunk = self._layout.chunks[state.next_chunk_id]
        content_time.verify_layout_binding(
            layout=self._layout, chunk=bound_chunk
        )
        if state.revision == 0:
            if (
                content_time.latent_start != 0
                or content_time.raw_observation_start != 0
            ):
                raise hc.CacheContractError(
                    "COMMIT_OUT_OF_ORDER", "mini first commit must start at zero"
                )
            return
        previous = state.layer_states[0].through
        if previous is None:
            raise hc.CacheContractError(
                "CACHE_SCHEMA_MISMATCH", "mini prior committed layer is empty"
            )
        if (
            content_time.latent_start != previous.latent_end_exclusive
            or content_time.raw_observation_start
            != previous.raw_observation_end_exclusive
            or content_time.action_rope_start
            != previous.action_rope_end_exclusive
        ):
            raise hc.CacheContractError(
                "COMMIT_OUT_OF_ORDER", "mini content intervals are not contiguous"
            )

    def commit_paired(
        self,
        *,
        request: h2.Stage2BTeacherForcingCommitRequest,
        stage_callback: Callable[
            [h2.Stage2BPairedStagingContext], tuple[hc.StagedLayerPayload, ...]
        ],
        inject_failure_after_materialize_for_tests: bool = False,
    ) -> ProductionPathMiniCommitReceipt:
        if type(request) is not h2.Stage2BTeacherForcingCommitRequest:
            raise TypeError("mini commit requires the exact Stage2B request type")
        if not callable(stage_callback):
            raise TypeError("mini commit requires a staging callback")
        if request.conditioning_digest is None:
            raise hc.CacheContractError(
                "COMMIT_CONDITIONING_IDENTITY_MISSING",
                "mini commit requires exact conditioning identity",
            )
        with self._lock:
            self._assert_live_integrity()
            base_state = self._state
            if request.commit_id in self._completed_ids:
                raise hc.CacheContractError(
                    "COMMIT_DUPLICATE_CONFLICT", "mini commit id was already used"
                )
            if (
                request.expected_episode_id != base_state.episode_id
                or request.expected_episode_epoch != base_state.episode_epoch
                or request.expected_revision != base_state.revision
            ):
                raise hc.CacheContractError(
                    "COMMIT_STALE_REVISION", "mini paired request is stale"
                )
            self._validate_next_content_time(base_state, request.content_time)
            chunk = self._layout.chunks[base_state.next_chunk_id]
        prepared = hc._prepare_teacher_pair(request, chunk=chunk)
        _require_cpu_fp32(prepared.video, "prepared paired video")
        _require_cpu_fp32(prepared.actions, "prepared paired actions")
        if tuple(prepared.video.shape) != (
            1,
            3,
            len(chunk.latent_valid_mask),
            1,
            1,
        ):
            raise ProductionPathMiniContractError(
                "prepared paired video shape escaped the fixed mini"
            )
        if tuple(prepared.actions.shape) != (
            1,
            chunk.action_slot_capacity,
            20,
        ):
            raise ProductionPathMiniContractError(
                "prepared paired action shape escaped the fixed mini"
            )
        if (
            prepared.frame_valid_mask.device.type != "cpu"
            or prepared.frame_valid_mask.dtype is not torch.bool
            or tuple(prepared.frame_valid_mask.shape)
            != (1, len(chunk.latent_valid_mask))
        ):
            raise ProductionPathMiniContractError(
                "prepared frame mask escaped exact CPU bool shape"
            )
        if (
            prepared.action_valid_mask.device.type != "cpu"
            or prepared.action_valid_mask.dtype is not torch.bool
            or tuple(prepared.action_valid_mask.shape)
            != (1, chunk.action_slot_capacity)
        ):
            raise ProductionPathMiniContractError(
                "prepared action mask escaped exact CPU bool shape"
            )
        if base_state.committed_action_history is None:
            base_history = CommittedActionHistory.empty(
                episode_id=base_state.episode_id,
                episode_epoch=base_state.episode_epoch,
                layout_spec_sha256=base_state.layout_spec_sha256,
                layout_instance_digest=base_state.layout_instance_digest,
                batch_size=prepared.actions.shape[0],
                action_dim=prepared.actions.shape[2],
                dtype=prepared.actions.dtype,
                device=prepared.actions.device,
            )
        else:
            base_history = base_state.committed_action_history
        history_view = base_history.view_for_chunk(
            chunk_id=chunk.chunk_id, action_start=chunk.action_start
        )
        context = h2.Stage2BPairedStagingContext(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            staging_variant=self._staging_variant,
            content_time=request.content_time,
            previous_state_manifest_digest=base_state.state_manifest_digest,
            previous_scratch=h2._scratch_from_state(
                base_state, source_state_identity=id(base_state)
            ),
            video=prepared.video.detach().clone(),
            frame_valid_mask=prepared.frame_valid_mask.detach().clone(),
            frame_valid_mask_digest=prepared.frame_valid_mask_digest,
            actions=prepared.actions.detach().clone(),
            action_valid_mask=prepared.action_valid_mask.detach().clone(),
            previous_committed_action_history=history_view,
            conditioning_digest=request.conditioning_digest,
        )
        input_digests = (
            hc.tensor_digest(context.video),
            hc.tensor_digest(context.frame_valid_mask),
            hc.tensor_digest(context.actions),
            hc.tensor_digest(context.action_valid_mask),
        )
        pending_token = object()
        with self._lock:
            self._assert_live_integrity()
            if self._pending is not None:
                raise hc.CacheContractError(
                    "COMMIT_DUPLICATE_PENDING",
                    "another mini commit is pending",
                )
            if self._state is not base_state:
                raise hc.CacheContractError(
                    "COMMIT_STALE_REVISION",
                    "mini live state changed before operator reservation",
                )
            self._pending = (request.commit_id, pending_token)
        try:
            payloads = stage_callback(context)
            if type(payloads) is not tuple:
                raise TypeError("mini stager must return an exact tuple")
            self._validate_staged_payloads(payloads)
            layer_states = hc._materialize_layer_states(
                registry=self._registry,
                content_time=request.content_time,
                payloads=payloads,
            )
            if inject_failure_after_materialize_for_tests:
                raise RuntimeError(
                    "registered Phase-C post-materialization failure"
                )
            context.previous_scratch.assert_unchanged()
            history_view.assert_unchanged()
            if input_digests != (
                hc.tensor_digest(context.video),
                hc.tensor_digest(context.frame_valid_mask),
                hc.tensor_digest(context.actions),
                hc.tensor_digest(context.action_valid_mask),
            ):
                raise RuntimeError("mini stager mutated its paired inputs")
            staged_history, duplicate = base_history.append(
                chunk_id=chunk.chunk_id,
                action_start=chunk.action_start,
                action_end_exclusive=chunk.action_end,
                fixed_slot_actions=prepared.actions,
                action_valid_mask=prepared.action_valid_mask,
                commit_source=hc.CommitSource.TEACHER_FORCING,
                source_proof_digest=prepared.source_proof_digest,
                commit_id=request.commit_id,
            )
            if duplicate:
                raise RuntimeError("new mini transaction duplicated history")
            transaction_digest = hc._sha256_bytes(
                hc._canonical_json_bytes(
                    {
                        "conditioning_digest": request.conditioning_digest,
                        "schema": "cach.production_path_mini.paired_payload.v1",
                        "staging_variant": self._staging_variant.value,
                        "teacher_pair_payload_digest": prepared.paired_payload_digest,
                    }
                )
            )
            staged_state = h2.Stage2BTemporalState(
                cache_state=hc.HybridTemporalState(
                    episode_id=base_state.episode_id,
                    episode_epoch=base_state.episode_epoch,
                    revision=base_state.revision + 1,
                    committed_through_chunk=chunk.chunk_id,
                    next_chunk_id=chunk.chunk_id + 1,
                    action_cursor=chunk.action_end,
                    layout_spec_sha256=base_state.layout_spec_sha256,
                    layout_instance_digest=base_state.layout_instance_digest,
                    layer_registry_digest=self._registry.manifest_digest,
                    layer_states=layer_states,
                    committed_pair_digest=transaction_digest,
                    last_commit_source=hc.CommitSource.TEACHER_FORCING,
                    last_source_proof_digest=prepared.source_proof_digest,
                    # The inherited typed state admits this exact literal for
                    # teacher-forcing source proofs.  Every proof field above
                    # is visibly synthetic; no real dataset is read.
                    pair_evidence_mode="dataset_ground_truth",
                ),
                staging_variant=self._staging_variant,
                committed_action_history=staged_history,
            )
        except hc.CacheContractError:
            with self._lock:
                if self._pending is not None and self._pending[1] is pending_token:
                    self._pending = None
            raise
        except Exception as exc:
            with self._lock:
                if self._pending is not None and self._pending[1] is pending_token:
                    self._pending = None
            raise hc.CacheContractError(
                "COMMIT_STAGING_FAILED",
                f"mini paired staging failed: {type(exc).__name__}",
            ) from exc
        with self._lock:
            self._assert_live_integrity()
            if (
                self._pending is None
                or self._pending[1] is not pending_token
                or self._state is not base_state
            ):
                if self._pending is not None and self._pending[1] is pending_token:
                    self._pending = None
                raise hc.CacheContractError(
                    "COMMIT_STALE_REVISION", "mini live state changed before CAS"
                )
            history_before = (
                None
                if base_state.committed_action_history is None
                else base_state.committed_action_history.manifest_digest
            )
            receipt = ProductionPathMiniCommitReceipt(
                commit_id=request.commit_id,
                staging_variant=self._staging_variant,
                revision_before=base_state.revision,
                revision_after=staged_state.revision,
                action_cursor_before=base_state.action_cursor,
                action_cursor_after=staged_state.action_cursor,
                state_manifest_before=base_state.state_manifest_digest,
                state_manifest_after=staged_state.state_manifest_digest,
                action_history_digest_before=history_before,
                action_history_digest_after=staged_history.manifest_digest,
                chunk_id=chunk.chunk_id,
            )
            self._state = staged_state
            self._state_manifest_digest = staged_state.state_manifest_digest
            self._completed_ids.add(request.commit_id)
            self._pending = None
            self.pointer_swap_count += 1
            return receipt

    def reset(
        self,
        *,
        reset_id: str,
        new_episode_id: str,
        layout: ChunkActionLayout,
    ) -> ProductionPathMiniResetReceipt:
        if (
            type(reset_id) is not str
            or type(new_episode_id) is not str
            or not reset_id
            or not new_episode_id
        ):
            raise ValueError("reset_id/new_episode_id must be non-empty")
        if layout is not self._layout:
            raise ProductionPathMiniContractError(
                "Phase-C reset requires the exact bound layout instance"
            )
        with self._lock:
            self._assert_live_integrity()
            if self._pending is not None:
                raise hc.CacheContractError(
                    "COMMIT_DUPLICATE_PENDING",
                    "mini reset cannot overlap a pending commit",
                )
            if reset_id in self._completed_ids:
                raise hc.CacheContractError(
                    "COMMIT_DUPLICATE_CONFLICT", "mini reset id was already used"
                )
            before = self._state
            after = h2.Stage2BTemporalState.empty(
                episode_id=new_episode_id,
                episode_epoch=before.episode_epoch + 1,
                layout=layout,
                registry=self._registry,
                staging_variant=self._staging_variant,
            )
            receipt = ProductionPathMiniResetReceipt(
                reset_id=reset_id,
                revision_before=before.revision,
                state_manifest_before=before.state_manifest_digest,
                state_manifest_after=after.state_manifest_digest,
                action_cursor_before=before.action_cursor,
            )
            self._state = after
            self._state_manifest_digest = after.state_manifest_digest
            self._completed_ids.add(reset_id)
            self.pointer_swap_count += 1
            return receipt


class _ProductionPathMiniPairedStager:
    def __init__(
        self,
        *,
        arm: ProductionPathMiniArm,
        conditioning: CACHChunkConditioning,
    ) -> None:
        self._arm = arm
        self._conditioning = conditioning

    def __call__(
        self, context: h2.Stage2BPairedStagingContext
    ) -> tuple[hc.StagedLayerPayload, ...]:
        if type(context) is not h2.Stage2BPairedStagingContext:
            raise TypeError("mini stager requires the exact Stage2B context")
        arm = self._arm
        if context.staging_variant is not arm.staging_variant:
            raise ValueError("mini owner/stager variants differ")
        chunk_id = context.content_time.chunk_id
        chunk = arm.layout.chunks[chunk_id]
        if context.conditioning_digest != self._conditioning.conditioning_digest:
            raise ValueError("mini conditioning digest differs")
        self._conditioning.verify_binding(
            layout=arm.layout,
            chunk_id=chunk_id,
            source_state_manifest_digest=context.previous_state_manifest_digest,
        )
        history = context.previous_committed_action_history
        if not isinstance(history, CommittedActionHistoryView):
            raise TypeError("mini paired context lacks typed history")
        committed_actions = history.clone_actions()
        if tuple(committed_actions.shape) != (
            context.actions.shape[0],
            chunk.action_start,
            context.actions.shape[2],
        ):
            raise ValueError("mini committed history coverage differs")
        result = arm.cach_core.run_paired_t0(
            video_backbone=arm.video_backbone,
            action_backbone=arm.action_backbone,
            proprio_context_encoder=arm.proprio_context_encoder,
            registry=arm.registry,
            action_layout=arm.layout,
            chunk_id=chunk_id,
            chunk_latents=context.video,
            frame_valid_mask=context.frame_valid_mask,
            noisy_actions=context.actions,
            action_valid_mask=context.action_valid_mask,
            committed_actions=committed_actions,
            committed_history_digest=history.history_manifest_digest,
            expected_committed_actions_digest=hc.tensor_digest(committed_actions),
            cache_scratch=context.previous_scratch,
            conditioning=self._conditioning,
            staging_variant=arm.staging_variant,
            video_timestep=context.video.new_zeros((context.video.shape[0],)),
            action_timestep=context.actions.new_zeros((context.actions.shape[0],)),
        )
        if result.staged_layer_payloads is None:
            raise RuntimeError("mini paired core returned no typed payloads")
        return result.staged_layer_payloads


class ProductionPathMiniDispatcher:
    """Public Phase-C dispatcher over full/read-only/paired production seams."""

    def __init__(
        self,
        *,
        arm: ProductionPathMiniArm,
        episode: ProductionPathMiniEpisode,
        episode_id: str,
    ) -> None:
        if not isinstance(arm, ProductionPathMiniArm):
            raise TypeError("arm must be ProductionPathMiniArm")
        if not isinstance(episode, ProductionPathMiniEpisode):
            raise TypeError("episode must be ProductionPathMiniEpisode")
        if episode.layout is not arm.layout:
            raise ProductionPathMiniContractError(
                "dispatcher requires one exact shared layout instance"
            )
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        arm._assert_episode_runtime(episode)
        self.arm = arm
        self.episode = episode
        self.episode_id = episode_id
        self.state_owner = ProductionPathMiniEphemeralStateOwner(
            registry=arm.registry,
            episode_id=episode_id,
            layout=arm.layout,
            staging_variant=arm.staging_variant,
        )

    @property
    def layout(self) -> ChunkActionLayout:
        return self.arm.layout

    def _conditioning(
        self, *, chunk_id: int, source_state_manifest_digest: str
    ) -> CACHChunkConditioning:
        batch = self.episode.chunks[chunk_id]
        chunk = self.layout.chunks[chunk_id]
        binding = SelectedProprioBinding(
            layout_instance_digest=self.layout.layout_instance_digest,
            chunk_id=chunk_id,
            raw_index=chunk.proprio_raw_index,
            timestamp=chunk.proprio_timestamp,
            source_receipt_digest=chunk.proprio_source_receipt_digest,
            proprio_value_sha256=canonical_proprio_row_sha256(
                batch.proprio_state[0]
            ),
        )
        return CACHChunkConditioning.create(
            layout_instance_digest=self.layout.layout_instance_digest,
            chunk_id=chunk_id,
            source_state_manifest_digest=source_state_manifest_digest,
            context=batch.context,
            context_mask=batch.context_mask,
            seq_lens=batch.seq_lens,
            proprio_state=batch.proprio_state,
            proprio_bindings=(binding,),
        )

    def run_full_sequence(self) -> ProductionPathMiniFullResult:
        self.arm._assert_episode_runtime(self.episode)
        return self.arm.run_full_sequence(self.episode)

    def run_readonly_chunk(self, *, chunk_id: int) -> CACHNumericalResult:
        self.arm._assert_episode_runtime(self.episode)
        state = self.state_owner.state
        view = self.state_owner.snapshot_for_denoise(
            expected_episode_id=state.episode_id,
            expected_episode_epoch=state.episode_epoch,
            expected_revision=state.revision,
            expected_next_chunk_id=chunk_id,
        )
        scratch = view.export_scratch()
        conditioning = self._conditioning(
            chunk_id=chunk_id,
            source_state_manifest_digest=view.state_manifest_digest,
        )
        history = view.previous_committed_action_history
        committed_actions = None if history is None else history.clone_actions()
        batch = self.episode.chunks[chunk_id]
        result = self.arm.cach_core.run_readonly_chunk(
            staging_variant=self.arm.staging_variant,
            cache_read_view=view,
            cache_scratch=scratch,
            video_backbone=self.arm.video_backbone,
            action_backbone=self.arm.action_backbone,
            proprio_context_encoder=self.arm.proprio_context_encoder,
            registry=self.arm.registry,
            action_layout=self.layout,
            chunk_id=chunk_id,
            chunk_latents=batch.noisy_video,
            frame_valid_mask=batch.frame_valid_mask,
            noisy_actions=batch.noisy_actions,
            action_valid_mask=batch.action_valid_mask,
            committed_actions=committed_actions,
            committed_history_digest=(
                _EMPTY_HISTORY_DIGEST
                if history is None
                else history.history_manifest_digest
            ),
            expected_committed_actions_digest=(
                None
                if committed_actions is None
                else hc.tensor_digest(committed_actions)
            ),
            conditioning=conditioning,
            video_timestep=batch.video_timestep,
            action_timestep=batch.action_timestep,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        )
        self.state_owner.finish_denoise(read_view=view, scratch=scratch)
        return result

    def _teacher_request(
        self,
        *,
        chunk_id: int,
        conditioning_digest: str,
    ) -> h2.Stage2BTeacherForcingCommitRequest:
        state = self.state_owner.state
        chunk = self.layout.chunks[chunk_id]
        batch = self.episode.chunks[chunk_id]
        proof = hc.TeacherForcingDatasetPairProof(
            dataset_manifest_sha256=hashlib.sha256(
                b"phase-c-synthetic-input-recipe"
            ).hexdigest(),
            dataset_episode_id=state.episode_id,
            dataset_row_identity=f"synthetic-phase-c:chunk-{chunk_id}",
            dataset_row_start=0,
            dataset_row_end_exclusive=self.layout.valid_raw_count,
            layout_spec_sha256=self.layout.layout_spec_sha256,
            layout_instance_digest=self.layout.layout_instance_digest,
            video_tensor_digest=hc.tensor_digest(batch.clean_video_target),
            frame_valid_mask_digest=hc.tensor_digest(batch.frame_valid_mask),
            action_tensor_digest=hc.tensor_digest(batch.clean_action_target),
            action_mask_digest=hc.tensor_digest(batch.action_valid_mask),
            row_order_manifest_sha256=self.episode.recipe_digest,
        )
        transaction_identity = hashlib.sha256(
            (
                f"{state.episode_id}:{state.episode_epoch}:"
                f"{self.arm.staging_variant.value}:{chunk_id}"
            ).encode("utf-8")
        ).hexdigest()
        return h2.Stage2BTeacherForcingCommitRequest(
            commit_id=f"phase-c-{transaction_identity}",
            transaction_nonce=f"phase-c-nonce-{transaction_identity}",
            expected_episode_id=state.episode_id,
            expected_episode_epoch=state.episode_epoch,
            expected_revision=state.revision,
            content_time=hc.ContentTime.from_layout(
                episode_id=state.episode_id,
                episode_epoch=state.episode_epoch,
                layout=self.layout,
                chunk=chunk,
            ),
            video=batch.clean_video_target,
            frame_valid_mask=batch.frame_valid_mask,
            actions=batch.clean_action_target,
            action_valid_mask=batch.action_valid_mask,
            proof=proof,
            conditioning_digest=conditioning_digest,
        )

    def commit_paired(
        self,
        *,
        chunk_id: int,
        inject_failure_before_vendor_for_tests: bool = False,
        inject_failure_after_vendor_for_tests: bool = False,
        inject_failure_after_materialize_for_tests: bool = False,
    ) -> ProductionPathMiniCommitReceipt:
        self.arm._assert_episode_runtime(self.episode)
        injection_count = sum(
            (
                inject_failure_before_vendor_for_tests,
                inject_failure_after_vendor_for_tests,
                inject_failure_after_materialize_for_tests,
            )
        )
        if injection_count > 1:
            raise ValueError("Phase-C accepts exactly one failure injection point")
        state = self.state_owner.state
        conditioning = self._conditioning(
            chunk_id=chunk_id,
            source_state_manifest_digest=state.state_manifest_digest,
        )
        request = self._teacher_request(
            chunk_id=chunk_id,
            conditioning_digest=conditioning.conditioning_digest,
        )
        paired_stager = _ProductionPathMiniPairedStager(
            arm=self.arm, conditioning=conditioning
        )
        if inject_failure_before_vendor_for_tests:
            def stage_callback(
                _context: h2.Stage2BPairedStagingContext,
            ) -> tuple[hc.StagedLayerPayload, ...]:
                raise RuntimeError("registered Phase-C pre-vendor failure")
        elif inject_failure_after_vendor_for_tests:
            def stage_callback(
                context: h2.Stage2BPairedStagingContext,
            ) -> tuple[hc.StagedLayerPayload, ...]:
                paired_stager(context)
                raise RuntimeError("registered Phase-C post-vendor failure")
        else:
            stage_callback = paired_stager
        return self.state_owner.commit_paired(
            request=request,
            stage_callback=stage_callback,
            inject_failure_after_materialize_for_tests=(
                inject_failure_after_materialize_for_tests
            ),
        )

    def reset_episode_for_tests(
        self, *, reset_id: str, new_episode_id: str
    ) -> ProductionPathMiniResetReceipt:
        receipt = self.state_owner.reset(
            reset_id=reset_id,
            new_episode_id=new_episode_id,
            layout=self.layout,
        )
        self.episode_id = new_episode_id
        return receipt


__all__ = [
    "ProductionPathMiniArm",
    "ProductionPathMiniCommitReceipt",
    "ProductionPathMiniContractError",
    "ProductionPathMiniDispatcher",
    "ProductionPathMiniEpisode",
    "ProductionPathMiniEphemeralStateOwner",
    "ProductionPathMiniFullResult",
    "ProductionPathMiniInventoryEntry",
    "ProductionPathMiniPair",
    "ProductionPathMiniResetReceipt",
    "ProductionPathMiniSpec",
    "ProductionPathMiniVideoBackbone",
    "build_production_path_mini_episode_for_tests",
    "build_production_path_mini_layout_for_tests",
    "build_production_path_mini_pair_for_tests",
]
