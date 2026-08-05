"""CACH-A2 R1 single-call bridge to Sana's autograd Triton ChunkCausalGDN.

This module is deliberately limited to the frozen CACH-A2 non-formal screen.
Every video block invokes the real vendor ``ChunkCausalGDNTriton`` exactly
once over five frames with chunk boundaries ``[0, 3, 5]``.  The vendor's
internal scan state is ephemeral to that call; this module exposes no
cross-call state input, state output, cache commit, or production dispatcher.

The reference arm is a structural action bypass.  Its video path never reads
an action tensor or a no-action tensor.  The candidate arm is byte-identical
on common initialization and adds only the card-pinned action seam:

``typed end-of-bin reducer -> bias-free 20/64/20 conditioner ->
per-block bias-free zero-init 20/64 -> active-only residual scatter``.

The bootstrap slot, padding helper slots, and full ``no_action`` mode are
structural identity bypasses.  Presence is derived only from the typed layout;
it is never inferred from an action value.

The local JVP helpers intentionally run the vendor primal before entering the
forward-mode transform.  ``torch.func.jvp`` sees only a pure-Torch tail made
from the detached last-block post-vendor hidden, action seam, FFN, and output
projection.  They therefore do not claim custom-autograd forward-mode support
from the vendor kernel.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
from typing import Final, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from sana_wam.cach.action_conditioning import (
    reduce_end_of_bin_action_condition,
)
from sana_wam.cach.prefix_compaction import FixedKPrefixPlan
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
from sana_wam.model.action_chunk_layout import ChunkActionLayout, ChunkLayout
from sana_wam.model.cach_av1_transition_proxy import (
    AV1_TASK_RECIPE_SHA256,
    AV1SyntheticTask,
    build_av1_synthetic_task,
)
from sana_wam.model.video_backbone.sana.hybrid_cache import tensor_digest


CACH_A2_CARD_SHA256: Final = (
    "c7b710177defda6d44066c7b7b25c4b32d11ec106a92662b0701ca5f87f3401e"
)
AV1B_TASK_RECIPE_SHA256: Final = (
    "9d2a749d722775324ee233862772def530fbffe0ea9e55aeea87c160b87a51e4"
)
AV1B_CHUNK_BOUNDARIES: Final = (0, 3, 5)
AV1B_HW: Final = (5, 1, 1)
AV1B_INITIALIZER_REVISION: Final = "cach-a2-zero-anchored-bias-free-v1"
AV1B_SEED0: Final = 202608033
AV1B_SHARED_NAMED_SEED: Final = 2026080331
AV1B_CANDIDATE_ONLY_SEED: Final = 2026080421
AV1B_SHUFFLE_PERMUTATION: Final = (1, 0, 3, 2, 5, 4, 7, 6)

ActionMode = Literal["correct", "shuffle", "no_action", "seam_disabled"]


@dataclass(frozen=True)
class AV1BVendorGDNBridgeSpec:
    """Exact reduced topology frozen by the CACH-A2 run card."""

    batch_size: int = 8
    counterfactual_pair_count: int = 4
    latent_channels: int = 3
    spatial_height: int = 1
    spatial_width: int = 1
    valid_raw_count: int = 33
    valid_latent_count: int = 5
    valid_action_count: int = 32
    action_dim: int = 20
    context_dim: int = 4
    context_token_count: int = 2
    hidden_dim: int = 64
    heads: int = 2
    head_dim: int = 32
    depth: int = 20
    conv_kernel_size: int = 0
    k_conv_only: bool = True
    qk_norm: bool = True
    use_output_gate: bool = True
    use_autograd_kernel: bool = True
    frame_count: int = 5
    chunk_boundaries: tuple[int, ...] = AV1B_CHUNK_BOUNDARIES
    ffn_hidden_dim: int = 128
    temporal_compression: int = 8
    video_stride: int = 1
    scored_future_horizons: tuple[int, ...] = (1, 2, 3, 4)
    primary_horizons: tuple[int, ...] = (1, 2, 4)
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        exact: Mapping[str, object] = {
            "batch_size": 8,
            "counterfactual_pair_count": 4,
            "latent_channels": 3,
            "spatial_height": 1,
            "spatial_width": 1,
            "valid_raw_count": 33,
            "valid_latent_count": 5,
            "valid_action_count": 32,
            "action_dim": 20,
            "context_dim": 4,
            "context_token_count": 2,
            "hidden_dim": 64,
            "heads": 2,
            "head_dim": 32,
            "depth": 20,
            "conv_kernel_size": 0,
            "k_conv_only": True,
            "qk_norm": True,
            "use_output_gate": True,
            "use_autograd_kernel": True,
            "frame_count": 5,
            "chunk_boundaries": AV1B_CHUNK_BOUNDARIES,
            "ffn_hidden_dim": 128,
            "temporal_compression": 8,
            "video_stride": 1,
            "scored_future_horizons": (1, 2, 3, 4),
            "primary_horizons": (1, 2, 4),
            "device": "cuda:0",
            "dtype": torch.float32,
        }
        for name, required in exact.items():
            value = getattr(self, name)
            if isinstance(required, bool):
                valid = type(value) is bool and value is required
            elif isinstance(required, int):
                valid = type(value) is int and value == required
            else:
                valid = value == required
            if not valid:
                raise ValueError(
                    f"{name} is frozen to {required!r}, got {value!r}"
                )
        if self.hidden_dim != self.heads * self.head_dim:
            raise ValueError("hidden_dim must equal heads * head_dim")


@dataclass(frozen=True)
class AV1BVendorTaskChunk:
    """One transferred AV-1 layout chunk used only by the action reducer."""

    chunk_id: int
    layout: ChunkLayout
    actions: Tensor = field(repr=False, compare=False)
    frame_valid_mask: Tensor = field(repr=False, compare=False)
    action_valid_mask: Tensor = field(repr=False, compare=False)


@dataclass(frozen=True)
class AV1BVendorTask:
    """Pinned AV-1 synthetic bytes materialized on the reserved CUDA device."""

    spec: AV1BVendorGDNBridgeSpec
    layout: ChunkActionLayout
    global_actions: Tensor = field(repr=False, compare=False)
    noisy_video: Tensor = field(repr=False, compare=False)
    base_target: Tensor = field(repr=False, compare=False)
    video_target: Tensor = field(repr=False, compare=False)
    context: Tensor = field(repr=False, compare=False)
    context_mask: Tensor = field(repr=False, compare=False)
    proprio: Tensor = field(repr=False, compare=False)
    video_timestep: Tensor = field(repr=False, compare=False)
    action_timestep: Tensor = field(repr=False, compare=False)
    chunks: tuple[AV1BVendorTaskChunk, ...]
    score_mask: Tensor = field(repr=False, compare=False)
    row_to_pair: tuple[int, ...]
    row_sign: tuple[int, ...]
    shuffle_permutation: tuple[int, ...]
    recipe_digest: str
    source_tensor_digests: Mapping[str, str]


@dataclass(frozen=True)
class AV1BVendorSequenceOutput:
    """One complete five-frame call and its explicitly scoped observables."""

    video_prediction: Tensor = field(repr=False, compare=False)
    last_block_post_vendor_hidden: Tensor = field(
        repr=False,
        compare=False,
    )
    last_block_raw_action_condition: Tensor | None = field(
        repr=False,
        compare=False,
    )
    last_block_action_condition: Tensor | None = field(
        repr=False,
        compare=False,
    )
    action_condition_valid_mask: Tensor | None = field(
        repr=False,
        compare=False,
    )
    block_action_residuals: tuple[Tensor, ...] = field(
        repr=False,
        compare=False,
    )
    block_post_vendor_hidden: tuple[Tensor, ...] = field(
        repr=False,
        compare=False,
    )
    block_post_seam_hidden: tuple[Tensor, ...] = field(
        repr=False,
        compare=False,
    )
    block_bridges: tuple[Tensor, ...] = field(repr=False, compare=False)
    action_predictions: tuple[Tensor, ...] = field(
        repr=False,
        compare=False,
    )
    vendor_calls_this_forward: int
    action_conditioner_calls_this_forward: int
    action_projection_calls_this_forward: int

    @property
    def action_prediction(self) -> Tensor | None:
        if not self.action_predictions:
            return None
        return torch.cat(self.action_predictions, dim=1)

    @property
    def vendor_forward_call_count(self) -> int:
        """Compatibility spelling for the per-forward call count."""

        return self.vendor_calls_this_forward


@dataclass(frozen=True)
class AV1BLocalJVPDiagnostic:
    """Pure-Torch last-block-tail JVP with its exact input direction."""

    scope: str
    primal: Tensor = field(repr=False, compare=False)
    tangent_output: Tensor = field(repr=False, compare=False)
    input_direction: Tensor = field(repr=False, compare=False)
    rms: Tensor = field(repr=False, compare=False)

    @property
    def finite(self) -> bool:
        return bool(
            torch.isfinite(self.primal.detach()).all()
            and torch.isfinite(self.tangent_output.detach()).all()
            and torch.isfinite(self.rms.detach()).all()
        )

    @property
    def primal_digest(self) -> str:
        return tensor_digest(self.primal.detach())


@dataclass(frozen=True)
class AV1BVendorGDNBridgePair:
    """Fresh common-template/deepcopy reference and candidate arms."""

    spec: AV1BVendorGDNBridgeSpec
    reference: "AV1BVendorGDNBridgeArm"
    candidate: "AV1BVendorGDNBridgeArm"
    common_trainable_names: frozenset[str]
    candidate_only_trainable_names: frozenset[str]
    reference_trainable_names: frozenset[str]
    candidate_trainable_names: frozenset[str]
    theta0_manifests: Mapping[str, Mapping[str, object]]


def _require_cuda_zero(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index != 0:
        raise ValueError("CACH-A2 is frozen to logical device cuda:0")
    return resolved


def _to_device(value: Tensor, device: torch.device) -> Tensor:
    return value.detach().to(
        device=device,
        dtype=value.dtype,
        non_blocking=False,
        copy=True,
    ).contiguous()


def _validate_source_task(source: AV1SyntheticTask) -> None:
    if source.recipe_digest != AV1B_TASK_RECIPE_SHA256:
        raise RuntimeError("the source task recipe digest differs from CACH-A2")
    if AV1_TASK_RECIPE_SHA256 != AV1B_TASK_RECIPE_SHA256:
        raise RuntimeError("the imported AV-1 task recipe constant drifted")
    if source.shuffle_permutation != AV1B_SHUFFLE_PERMUTATION:
        raise RuntimeError("the AV-1 shuffle permutation drifted")
    if tuple(chunk.layout.valid_latent_count for chunk in source.chunks) != (
        3,
        2,
    ):
        raise RuntimeError("the AV-1 valid latent chunk counts drifted")
    if tuple(chunk.layout.valid_action_count for chunk in source.chunks) != (
        16,
        16,
    ):
        raise RuntimeError("the AV-1 valid action chunk counts drifted")


def build_av1b_vendor_task(
    spec: AV1BVendorGDNBridgeSpec | None = None,
    *,
    device: torch.device | str = torch.device("cuda:0"),
) -> AV1BVendorTask:
    """Reuse the pinned AV-1 bytes and transfer them without new RNG.

    The runner must call this before model construction and persist the
    returned ``source_tensor_digests`` as the pre-model task manifest.
    """

    spec = AV1BVendorGDNBridgeSpec() if spec is None else spec
    if not isinstance(spec, AV1BVendorGDNBridgeSpec):
        raise TypeError("spec must be an AV1BVendorGDNBridgeSpec")
    resolved = _require_cuda_zero(device)
    if str(resolved) != spec.device:
        raise ValueError("device differs from the card-frozen spec device")

    source = build_av1_synthetic_task()
    _validate_source_task(source)

    chunks = tuple(
        AV1BVendorTaskChunk(
            chunk_id=chunk.chunk_id,
            layout=chunk.layout,
            actions=_to_device(chunk.actions, resolved),
            frame_valid_mask=_to_device(
                chunk.frame_valid_mask,
                resolved,
            ),
            action_valid_mask=_to_device(
                chunk.action_valid_mask,
                resolved,
            ),
        )
        for chunk in source.chunks
    )
    task = AV1BVendorTask(
        spec=spec,
        layout=source.layout,
        global_actions=_to_device(source.global_actions, resolved),
        noisy_video=_to_device(source.noisy_video, resolved),
        base_target=_to_device(source.base_target, resolved),
        video_target=_to_device(source.video_target, resolved),
        context=_to_device(source.context, resolved),
        context_mask=_to_device(source.context_mask, resolved),
        proprio=_to_device(source.proprio, resolved),
        video_timestep=_to_device(source.video_timestep, resolved),
        action_timestep=_to_device(source.action_timestep, resolved),
        chunks=chunks,
        score_mask=_to_device(source.score_mask, resolved),
        row_to_pair=source.row_to_pair,
        row_sign=source.row_sign,
        shuffle_permutation=source.shuffle_permutation,
        recipe_digest=source.recipe_digest,
        source_tensor_digests=dict(source.tensor_digests),
    )
    _validate_vendor_task(task)
    return task


def build_av1b_synthetic_task(
    spec: AV1BVendorGDNBridgeSpec | None = None,
    *,
    device: torch.device | str = torch.device("cuda:0"),
) -> AV1BVendorTask:
    """Compatibility spelling for :func:`build_av1b_vendor_task`."""

    return build_av1b_vendor_task(spec, device=device)


def _validate_vendor_task(task: AV1BVendorTask) -> None:
    if not isinstance(task, AV1BVendorTask):
        raise TypeError("task must be an AV1BVendorTask")
    spec = task.spec
    device = torch.device(spec.device)
    float_tensors = (
        task.global_actions,
        task.noisy_video,
        task.base_target,
        task.video_target,
        task.context,
        task.proprio,
        task.video_timestep,
        task.action_timestep,
    )
    if any(
        tensor.device != device or tensor.dtype != torch.float32
        for tensor in float_tensors
    ):
        raise TypeError("all CACH-A2 floating task tensors must be cuda:0 FP32")
    bool_tensors = (task.context_mask, task.score_mask) + tuple(
        tensor
        for chunk in task.chunks
        for tensor in (chunk.frame_valid_mask, chunk.action_valid_mask)
    )
    if any(
        tensor.device != device or tensor.dtype is not torch.bool
        for tensor in bool_tensors
    ):
        raise TypeError("all CACH-A2 mask tensors must be cuda:0 bool")
    exact_shapes = {
        "global_actions": (8, 32, 20),
        "noisy_video": (8, 5, 3, 1, 1),
        "base_target": (8, 5, 3, 1, 1),
        "video_target": (8, 5, 3, 1, 1),
        "context": (8, 2, 2, 4),
        "context_mask": (8, 2, 2),
        "proprio": (8, 2, 20),
        "video_timestep": (8, 2),
        "action_timestep": (8, 2),
        "score_mask": (8, 4),
    }
    for name, shape in exact_shapes.items():
        if tuple(getattr(task, name).shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if task.recipe_digest != AV1B_TASK_RECIPE_SHA256:
        raise ValueError("task recipe digest differs from the frozen card")
    if task.shuffle_permutation != AV1B_SHUFFLE_PERMUTATION:
        raise ValueError("task shuffle permutation differs from the card")
    if len(task.chunks) != 2:
        raise ValueError("CACH-A2 requires exactly two layout chunks")


def _parameter_free_rms_norm(value: Tensor, eps: float = 1.0e-6) -> Tensor:
    return value * torch.rsqrt(
        value.float().square().mean(dim=-1, keepdim=True) + eps
    ).to(dtype=value.dtype)


def _vendor_gdn_class() -> type[nn.Module]:
    """Import the pinned vendor class only when a model is constructed."""

    from diffusion.model.nets.sana_gdn_blocks_triton import (
        ChunkCausalGDNTriton,
    )

    return ChunkCausalGDNTriton


def _active_action_indices(action_present_mask: Tensor) -> Tensor:
    if action_present_mask.dtype is not torch.bool:
        raise TypeError("action_present_mask must be boolean")
    if action_present_mask.ndim != 3 or action_present_mask.shape[-1] != 1:
        raise ValueError("action_present_mask must be [batch,time,1]")
    return action_present_mask.squeeze(-1).reshape(-1).nonzero(
        as_tuple=False
    ).flatten()


def condition_zero_anchor_actions(
    raw_action: Tensor,
    action_present_mask: Tensor,
    conditioner: nn.Module,
) -> tuple[Tensor | None, int]:
    """Run the bias-free conditioner on active typed-layout slots only."""

    if raw_action.ndim != 3:
        raise ValueError("raw_action must be [batch,time,action_dim]")
    if tuple(action_present_mask.shape) != (*raw_action.shape[:-1], 1):
        raise ValueError("raw_action and action_present_mask axes differ")
    active_index = _active_action_indices(action_present_mask)
    if not active_index.numel():
        return None, 0
    active_raw = raw_action.reshape(-1, raw_action.shape[-1]).index_select(
        0,
        active_index,
    )
    active_condition = conditioner(active_raw)
    conditioned = torch.zeros_like(raw_action.reshape(-1, raw_action.shape[-1]))
    conditioned = conditioned.index_copy(0, active_index, active_condition)
    return conditioned.reshape_as(raw_action), 1


def apply_zero_anchor_action_residual(
    hidden: Tensor,
    action_condition: Tensor | None,
    action_present_mask: Tensor,
    projection: nn.Linear,
) -> tuple[Tensor, Tensor, int]:
    """Gather, project, scatter, then direct-select inactive hidden bytes."""

    if tuple(action_present_mask.shape) != (*hidden.shape[:-1], 1):
        raise ValueError("hidden and action_present_mask axes differ")
    active_index = _active_action_indices(action_present_mask)
    residual = torch.zeros_like(hidden)
    if not active_index.numel():
        if action_condition is not None and bool(
            action_condition.detach().count_nonzero()
        ):
            raise RuntimeError("inactive action condition must be exact zero")
        return hidden, residual, 0
    if action_condition is None:
        raise RuntimeError("active mask requires action condition")
    active_condition = action_condition.reshape(
        -1,
        action_condition.shape[-1],
    ).index_select(0, active_index)
    active_residual = projection(active_condition)
    residual = torch.zeros_like(
        hidden.reshape(-1, hidden.shape[-1])
    ).index_copy(0, active_index, active_residual).reshape_as(hidden)
    post_seam = torch.where(
        action_present_mask,
        hidden + residual,
        hidden,
    )
    return post_seam, residual, 1


class AV1BVendorGDNBlock(nn.Module):
    """One real vendor GDN block plus the card-pinned residual FFN."""

    def __init__(self, spec: AV1BVendorGDNBridgeSpec) -> None:
        super().__init__()
        self.spec = spec
        vendor_class = _vendor_gdn_class()
        self.vendor = vendor_class(
            in_dim=spec.hidden_dim,
            out_dim=spec.hidden_dim,
            heads=spec.heads,
            dim=spec.head_dim,
            use_bias=False,
            qk_norm=spec.qk_norm,
            norm_eps=1.0e-5,
            use_output_gate=spec.use_output_gate,
            conv_kernel_size=spec.conv_kernel_size,
            k_conv_only=spec.k_conv_only,
            use_autograd_kernel=spec.use_autograd_kernel,
        )
        self.ffn_input_projection = nn.Linear(
            spec.hidden_dim,
            spec.ffn_hidden_dim,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.ffn_output_projection = nn.Linear(
            spec.ffn_hidden_dim,
            spec.hidden_dim,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.action_output_projection: nn.Linear | None = None
        self.vendor_forward_call_count = 0
        self.action_projection_call_count = 0
        self._assert_vendor_identity()

    def _assert_vendor_identity(self) -> None:
        if type(self.vendor) is not _vendor_gdn_class():
            raise RuntimeError("vendor class is not exactly ChunkCausalGDNTriton")
        if self.vendor.use_autograd_kernel is not True:
            raise RuntimeError("vendor autograd kernel must be enabled")
        if self.vendor.heads != 2 or self.vendor.dim != 32:
            raise RuntimeError("vendor head topology differs from 2 x 32")
        if self.vendor.conv_kernel_size != 0:
            raise RuntimeError("vendor conv_kernel_size must be zero")
        if self.vendor.k_conv_only is not True:
            raise RuntimeError("vendor k_conv_only must be true")
        if self.vendor.conv_q is not None or self.vendor.conv_v is not None:
            raise RuntimeError("vendor q/v convolution must be absent")
        if self.vendor.conv_k is not None:
            raise RuntimeError("vendor k convolution must be absent at conv=0")
        if isinstance(self.vendor.q_norm, nn.Identity):
            raise RuntimeError("vendor qk_norm must be enabled")
        if self.vendor.use_output_gate is not True:
            raise RuntimeError("vendor output gate must be enabled")

    def enable_candidate_action_seam(self) -> None:
        if self.action_output_projection is not None:
            raise RuntimeError("candidate action seam was already enabled")
        self.action_output_projection = nn.Linear(
            self.spec.action_dim,
            self.spec.hidden_dim,
            bias=False,
            device="cpu",
            dtype=torch.float32,
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        action_condition: Tensor | None,
        action_present_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        normalized = _parameter_free_rms_norm(hidden)
        vendor_output = self.vendor(
            normalized,
            HW=AV1B_HW,
            chunk_size=None,
            chunk_split_strategy="uniform",
            chunk_index=list(AV1B_CHUNK_BOUNDARIES),
        )
        self.vendor_forward_call_count += 1
        post_vendor = hidden + vendor_output
        post_seam = post_vendor
        action_residual = torch.zeros_like(post_vendor)
        if self.action_output_projection is not None:
            if action_present_mask is None:
                raise RuntimeError("candidate block requires an action mask")
            post_seam, action_residual, projection_calls = (
                apply_zero_anchor_action_residual(
                    post_vendor,
                    action_condition,
                    action_present_mask,
                    self.action_output_projection,
                )
            )
            self.action_projection_call_count += projection_calls
        elif action_condition is not None or action_present_mask is not None:
            raise RuntimeError("reference block cannot consume action condition")
        output = post_seam + self.ffn_output_projection(
            F.silu(
                self.ffn_input_projection(
                    _parameter_free_rms_norm(post_seam)
                )
            )
        )
        return output, post_vendor, post_seam, action_residual


def _candidate_only_names(depth: int) -> frozenset[str]:
    names = {
        "action_conditioner.0.weight",
        "action_conditioner.2.weight",
    }
    for block_id in range(depth):
        names.add(f"blocks.{block_id}.action_output_projection.weight")
    return frozenset(names)


def _candidate_seed_for_name(name: str) -> int:
    material = (
        f"{AV1B_INITIALIZER_REVISION}:{AV1B_CANDIDATE_ONLY_SEED}:"
        f"candidate_operator_named_uniform_v1:{name}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


class AV1BVendorGDNBridgeArm(nn.Module):
    """One CACH-A2 arm; only CACH-A2 owns or evaluates an action seam."""

    def __init__(
        self,
        *,
        spec: AV1BVendorGDNBridgeSpec,
        staging_variant: CACHStagingVariant,
    ) -> None:
        super().__init__()
        if not isinstance(spec, AV1BVendorGDNBridgeSpec):
            raise TypeError("spec must be an AV1BVendorGDNBridgeSpec")
        if staging_variant is not CACHStagingVariant.REF_GDN_CORRECTED:
            raise ValueError("the common template must be constructed as REF")
        self.spec = spec
        self.staging_variant = staging_variant
        self.video_input_projection = nn.Linear(
            spec.latent_channels,
            spec.hidden_dim,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.video_timestep_projection = nn.Linear(
            1,
            spec.hidden_dim,
            bias=False,
            device="cpu",
            dtype=torch.float32,
        )
        self.proprio_context_encoder = nn.Linear(
            spec.action_dim,
            spec.context_dim,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.context_projection = nn.Linear(
            spec.context_dim,
            spec.hidden_dim,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.blocks = nn.ModuleList(
            [AV1BVendorGDNBlock(spec) for _ in range(spec.depth)]
        )
        self.video_output_projection = nn.Linear(
            spec.hidden_dim,
            spec.latent_channels,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.action_conditioner: nn.Sequential | None = None
        self.register_buffer("no_action_slot", None, persistent=True)
        self.action_conditioner_call_count = 0
        self.action_backbone = ActionDiT(
            action_dim=spec.action_dim,
            dim=spec.hidden_dim,
            ffn_dim=spec.ffn_hidden_dim,
            num_heads=spec.heads,
            num_layers=spec.depth,
            video_dim=spec.hidden_dim,
            bridge_layers=tuple(range(spec.depth)),
            variant="joint_cross_attn",
            attn_head_dim=spec.head_dim,
            text_dim=spec.context_dim,
            freq_dim=spec.context_dim,
            max_action_len=128,
            attn_kernel="softmax",
        )
        for parameter in self.action_backbone.parameters():
            parameter.requires_grad_(False)
        self.action_backbone.eval()

    @property
    def arm_name(self) -> str:
        if self.staging_variant is CACHStagingVariant.CACH_A:
            return "CACH-A2"
        return "REF-GDN-CORRECTED"

    @property
    def trainable_name_set(self) -> frozenset[str]:
        return frozenset(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    @property
    def vendor_forward_call_count(self) -> int:
        return sum(block.vendor_forward_call_count for block in self.blocks)

    def train(self, mode: bool = True) -> "AV1BVendorGDNBridgeArm":
        super().train(mode)
        self.action_backbone.eval()
        return self

    def enable_candidate_action_seam(self) -> None:
        if self.staging_variant is not CACHStagingVariant.REF_GDN_CORRECTED:
            raise RuntimeError("arm is not a fresh reference deepcopy")
        self.staging_variant = CACHStagingVariant.CACH_A
        self.action_conditioner = nn.Sequential(
            nn.Linear(
                self.spec.action_dim,
                self.spec.hidden_dim,
                bias=False,
                device="cpu",
                dtype=torch.float32,
            ),
            nn.SiLU(),
            nn.Linear(
                self.spec.hidden_dim,
                self.spec.action_dim,
                bias=False,
                device="cpu",
                dtype=torch.float32,
            ),
        )
        self.no_action_slot = torch.zeros(
            self.spec.action_dim,
            dtype=torch.float32,
            device="cpu",
        )
        for block in self.blocks:
            block.enable_candidate_action_seam()
        self._initialize_candidate_only_parameters()

    def _initialize_candidate_only_parameters(self) -> None:
        expected = _candidate_only_names(self.spec.depth)
        actual = frozenset(
            name
            for name, _ in self.named_parameters()
            if name.startswith("action_conditioner.")
            or ".action_output_projection." in name
        )
        if actual != expected:
            raise RuntimeError("candidate-only parameter names drifted")
        with torch.no_grad():
            for name, parameter in sorted(
                self.named_parameters(),
                key=lambda item: item[0].encode("utf-8"),
            ):
                if name not in expected:
                    continue
                if ".action_output_projection." in name:
                    parameter.zero_()
                    continue
                fan_in = int(parameter.shape[-1])
                bound = 0.25 / math.sqrt(float(fan_in))
                generator = torch.Generator(device="cpu")
                generator.manual_seed(_candidate_seed_for_name(name))
                parameter.uniform_(-bound, bound, generator=generator)
        if self.no_action_slot is None:
            raise RuntimeError("candidate no-action slot is absent")
        self.no_action_slot.requires_grad_(False)
        self.no_action_slot.zero_()

    def _validate_non_action_task(self, task: AV1BVendorTask) -> None:
        if not isinstance(task, AV1BVendorTask):
            raise TypeError("task must be an AV1BVendorTask")
        if task.spec != self.spec:
            raise ValueError("task spec differs from arm spec")
        device = next(self.parameters()).device
        non_action = (
            task.noisy_video,
            task.context,
            task.context_mask,
            task.proprio,
            task.video_timestep,
            task.action_timestep,
        )
        if any(value.device != device for value in non_action):
            raise ValueError("task and arm devices differ")

    def _input_hidden(self, task: AV1BVendorTask) -> tuple[Tensor, Tensor]:
        video_tokens = task.noisy_video[:, :, :, 0, 0]
        hidden = self.video_input_projection(video_tokens)
        mask_float = task.context_mask.to(dtype=torch.float32).unsqueeze(-1)
        context_summary = (task.context * mask_float).sum(dim=2) / (
            mask_float.sum(dim=2).clamp_min(1.0)
        )
        proprio_token = self.proprio_context_encoder(task.proprio)
        context_hidden = self.context_projection(
            context_summary + proprio_token
        )
        timestep_hidden = self.video_timestep_projection(
            task.video_timestep.unsqueeze(-1)
        )
        frame_to_chunk = torch.tensor(
            (0, 0, 0, 1, 1),
            dtype=torch.long,
            device=hidden.device,
        )
        hidden = (
            hidden
            + context_hidden.index_select(1, frame_to_chunk)
            + timestep_hidden.index_select(1, frame_to_chunk)
        )
        return hidden, proprio_token

    def _candidate_actions_for_mode(
        self,
        task: AV1BVendorTask,
        mode: ActionMode,
    ) -> Tensor:
        if self.staging_variant is not CACHStagingVariant.CACH_A:
            raise RuntimeError("reference cannot enter the action seam")
        if mode == "correct":
            return task.global_actions
        if mode == "shuffle":
            permutation = torch.tensor(
                task.shuffle_permutation,
                dtype=torch.long,
                device=task.global_actions.device,
            )
            return task.global_actions.index_select(0, permutation)
        if mode in ("no_action", "seam_disabled"):
            raise RuntimeError("inactive modes must short-circuit before action read")
        raise ValueError("mode must be correct, shuffle, no_action, or seam_disabled")

    @staticmethod
    def _fixed_capacity_chunk_actions(
        global_actions: Tensor,
        task_chunk: AV1BVendorTaskChunk,
    ) -> Tensor:
        chunk = task_chunk.layout
        valid = global_actions[:, chunk.action_start : chunk.action_end, :]
        padding_count = chunk.action_slot_capacity - chunk.valid_action_count
        if padding_count:
            padding = torch.zeros(
                (
                    global_actions.shape[0],
                    padding_count,
                    global_actions.shape[2],
                ),
                dtype=global_actions.dtype,
                device=global_actions.device,
            )
            return torch.cat((valid, padding), dim=1)
        return valid

    def _candidate_action_condition(
        self,
        task: AV1BVendorTask,
        mode: ActionMode,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        if mode in ("no_action", "seam_disabled"):
            raw = torch.zeros(
                (self.spec.batch_size, 5, self.spec.action_dim),
                dtype=task.noisy_video.dtype,
                device=task.noisy_video.device,
            )
            inactive = torch.zeros(
                (self.spec.batch_size, 5, 1),
                dtype=torch.bool,
                device=task.noisy_video.device,
            )
            return raw, None, inactive
        if self.action_conditioner is None or self.no_action_slot is None:
            raise RuntimeError("candidate action seam is incomplete")
        global_actions = self._candidate_actions_for_mode(task, mode)
        reduced_parts: list[Tensor] = []
        presence_parts: list[Tensor] = []
        for task_chunk in task.chunks:
            chunk_actions = self._fixed_capacity_chunk_actions(
                global_actions,
                task_chunk,
            )
            padding_start = task_chunk.layout.valid_action_count
            if bool(chunk_actions[:, padding_start:, :].detach().count_nonzero()):
                raise RuntimeError("padded action slots must be exact zero")
            committed = (
                None
                if task_chunk.layout.action_start == 0
                else global_actions[:, : task_chunk.layout.action_start, :]
            )
            reduced = reduce_end_of_bin_action_condition(
                chunk_actions,
                committed_actions=committed,
                chunk=task_chunk.layout,
                no_action_slot=self.no_action_slot,
            )
            valid_count = task_chunk.layout.valid_latent_count
            reduced_parts.append(reduced.condition[:, :valid_count, :])
            local_presence = torch.tensor(
                tuple(
                    bool(task_chunk.layout.latent_valid_mask[local_latent])
                    and not span.anchor_no_action_slot
                    for local_latent, span in enumerate(
                        task_chunk.layout.latent_action_spans
                    )
                ),
                dtype=torch.bool,
                device=global_actions.device,
            ).view(1, -1).expand(self.spec.batch_size, -1)
            local_presence = local_presence[:, :valid_count]
            if not torch.equal(
                reduced.latent_valid_mask[:, :valid_count],
                torch.ones_like(local_presence),
            ):
                raise RuntimeError("current vendor input must contain only valid frames")
            presence_parts.append(local_presence)
        raw = torch.cat(reduced_parts, dim=1)
        present_mask = torch.cat(presence_parts, dim=1).unsqueeze(-1)
        if tuple(raw.shape) != (8, 5, 20):
            raise RuntimeError("single-call action condition must be [8,5,20]")
        expected_row = torch.tensor(
            (False, True, True, True, True),
            dtype=torch.bool,
            device=raw.device,
        ).view(1, 5, 1).expand(self.spec.batch_size, 5, 1)
        if not torch.equal(present_mask, expected_row):
            raise RuntimeError("typed action-presence mask differs from the card")
        conditioned, conditioner_calls = condition_zero_anchor_actions(
            raw,
            present_mask,
            self.action_conditioner,
        )
        if conditioned is None or conditioner_calls != 1:
            raise RuntimeError("active action condition was not materialized")
        self.action_conditioner_call_count += conditioner_calls
        return raw, conditioned, present_mask

    def _diagnostic_actions_for_mode(
        self,
        task: AV1BVendorTask,
        mode: ActionMode,
    ) -> Tensor:
        """ActionDiT-only input, kept outside the reference video path."""

        if mode == "correct":
            return task.global_actions
        if mode == "shuffle":
            permutation = torch.tensor(
                task.shuffle_permutation,
                dtype=torch.long,
                device=task.global_actions.device,
            )
            return task.global_actions.index_select(0, permutation)
        if mode in ("no_action", "seam_disabled"):
            return torch.zeros_like(task.global_actions)
        raise ValueError("unsupported action diagnostic mode")

    def _run_frozen_action_diagnostic(
        self,
        *,
        task: AV1BVendorTask,
        mode: ActionMode,
        bridges: tuple[Tensor, ...],
        proprio_tokens: Tensor,
    ) -> tuple[Tensor, ...]:
        diagnostic_actions = self._diagnostic_actions_for_mode(task, mode)
        predictions: list[Tensor] = []
        frame_start = 0
        for task_chunk in task.chunks:
            chunk = task_chunk.layout
            local_actions = self._fixed_capacity_chunk_actions(
                diagnostic_actions,
                task_chunk,
            )
            action_plan = FixedKPrefixPlan.from_mask(
                task_chunk.action_valid_mask
            )
            compact_actions = action_plan.compact_condition(local_actions)
            frame_plan = FixedKPrefixPlan.from_mask(
                task_chunk.frame_valid_mask
            )
            padded_bridges: list[Tensor] = []
            for bridge in bridges:
                local = bridge[
                    :, frame_start : frame_start + chunk.valid_latent_count
                ]
                padding = len(chunk.latent_valid_mask) - chunk.valid_latent_count
                if padding:
                    local = torch.cat(
                        (
                            local,
                            torch.zeros(
                                (
                                    local.shape[0],
                                    padding,
                                    local.shape[2],
                                ),
                                dtype=local.dtype,
                                device=local.device,
                            ),
                        ),
                        dim=1,
                    )
                padded_bridges.append(frame_plan.compact_condition(local))
            context = torch.cat(
                (
                    task.context[:, chunk.chunk_id],
                    proprio_tokens[:, chunk.chunk_id].unsqueeze(1),
                ),
                dim=1,
            )
            context_mask = torch.cat(
                (
                    task.context_mask[:, chunk.chunk_id],
                    torch.ones(
                        (self.spec.batch_size, 1),
                        dtype=torch.bool,
                        device=context.device,
                    ),
                ),
                dim=1,
            )
            positions = torch.arange(
                chunk.action_rope_start,
                chunk.action_rope_end,
                dtype=torch.long,
                device=context.device,
            )
            with torch.no_grad():
                frequencies = self.action_backbone._get_rope_freqs_at(
                    positions
                ).to(device=context.device)
                prediction = self.action_backbone.forward_with_bridge_tuple(
                    compact_actions.detach(),
                    tuple(value.detach() for value in padded_bridges),
                    task.action_timestep[:, chunk.chunk_id].detach(),
                    context=context.detach(),
                    context_mask=context_mask,
                    action_freqs=frequencies,
                    use_gradient_checkpointing=False,
                    use_gradient_checkpointing_offload=False,
                )
            predictions.append(prediction)
            frame_start += chunk.valid_latent_count
        return tuple(predictions)

    def forward(
        self,
        task: AV1BVendorTask,
        mode: ActionMode = "correct",
        *,
        target_override: Tensor | None = None,
        capture_diagnostics: bool = True,
        run_action_diagnostic: bool = False,
    ) -> AV1BVendorSequenceOutput:
        """Run one complete five-frame vendor call in every block.

        ``target_override`` is metadata-validated but its values are never
        read.  It exists solely for the target-leakage diagnostic.
        """

        if mode not in ("correct", "shuffle", "no_action", "seam_disabled"):
            raise ValueError(
                "mode must be correct, shuffle, no_action, or seam_disabled"
            )
        self._validate_non_action_task(task)
        if target_override is not None:
            if not isinstance(target_override, Tensor):
                raise TypeError("target_override must be a Tensor")
            if (
                tuple(target_override.shape) != (8, 5, 3, 1, 1)
                or target_override.dtype != torch.float32
                or target_override.device != task.video_target.device
            ):
                raise ValueError("target_override metadata differs from target")
            # Deliberately do not read target_override values.

        starting_conditioner_calls = self.action_conditioner_call_count
        starting_projection_calls = sum(
            block.action_projection_call_count for block in self.blocks
        )
        raw_action: Tensor | None = None
        action_condition: Tensor | None = None
        action_valid_mask: Tensor | None = None
        if self.staging_variant is CACHStagingVariant.CACH_A:
            raw_action, action_condition, action_valid_mask = (
                self._candidate_action_condition(task, mode)
            )

        hidden, proprio_tokens = self._input_hidden(task)
        starting_calls = self.vendor_forward_call_count
        bridges: list[Tensor] = []
        action_residuals: list[Tensor] = []
        post_vendor_values: list[Tensor] = []
        post_seam_values: list[Tensor] = []
        last_post_vendor: Tensor | None = None
        for block_id, block in enumerate(self.blocks):
            hidden, post_vendor, post_seam, action_residual = block(
                hidden,
                action_condition=action_condition,
                action_present_mask=action_valid_mask,
            )
            bridges.append(hidden)
            action_residuals.append(action_residual)
            post_vendor_values.append(post_vendor)
            post_seam_values.append(post_seam)
            if block_id == self.spec.depth - 1:
                last_post_vendor = post_vendor
                if capture_diagnostics and post_vendor.requires_grad:
                    post_vendor.retain_grad()
        if last_post_vendor is None:
            raise AssertionError("CACH-A2 depth must be nonzero")
        prediction = self.video_output_projection(hidden)
        prediction = prediction.unsqueeze(-1).unsqueeze(-1)
        action_predictions: tuple[Tensor, ...] = ()
        if run_action_diagnostic:
            action_predictions = self._run_frozen_action_diagnostic(
                task=task,
                mode=mode,
                bridges=tuple(bridges),
                proprio_tokens=proprio_tokens,
            )
        calls = self.vendor_forward_call_count - starting_calls
        conditioner_calls = (
            self.action_conditioner_call_count - starting_conditioner_calls
        )
        projection_calls = sum(
            block.action_projection_call_count for block in self.blocks
        ) - starting_projection_calls
        if calls != self.spec.depth:
            raise RuntimeError("each forward must call every vendor block once")
        if mode in ("no_action", "seam_disabled") and (
            conditioner_calls != 0 or projection_calls != 0
        ):
            raise RuntimeError("inactive mode executed the action path")
        return AV1BVendorSequenceOutput(
            video_prediction=prediction,
            last_block_post_vendor_hidden=last_post_vendor,
            last_block_raw_action_condition=raw_action,
            last_block_action_condition=action_condition,
            action_condition_valid_mask=action_valid_mask,
            block_action_residuals=tuple(action_residuals),
            block_post_vendor_hidden=tuple(post_vendor_values),
            block_post_seam_hidden=tuple(post_seam_values),
            block_bridges=tuple(bridges) if capture_diagnostics else (),
            action_predictions=action_predictions,
            vendor_calls_this_forward=calls,
            action_conditioner_calls_this_forward=conditioner_calls,
            action_projection_calls_this_forward=projection_calls,
        )


def _named_parameter_manifest(
    module: nn.Module,
    names: frozenset[str],
) -> Mapping[str, object]:
    parameters = dict(module.named_parameters())
    ordered = sorted(names, key=lambda value: value.encode("utf-8"))
    entries: dict[str, object] = {}
    for name in ordered:
        parameter = parameters[name].detach()
        entries[name] = {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "sha256": tensor_digest(parameter),
        }
    canonical = json.dumps(
        entries,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "names": ordered,
        "entries": entries,
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _named_state_manifest(
    module: nn.Module,
    *,
    parameter_names: frozenset[str],
    buffer_names: frozenset[str],
) -> Mapping[str, object]:
    """Manifest exact parameter and buffer bytes, including frozen state."""

    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    ordered_parameters = sorted(
        parameter_names,
        key=lambda value: value.encode("utf-8"),
    )
    ordered_buffers = sorted(
        buffer_names,
        key=lambda value: value.encode("utf-8"),
    )
    parameter_entries: dict[str, object] = {}
    buffer_entries: dict[str, object] = {}
    for name in ordered_parameters:
        value = parameters[name].detach()
        parameter_entries[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": parameters[name].requires_grad,
            "sha256": tensor_digest(value),
        }
    for name in ordered_buffers:
        value = buffers[name].detach()
        buffer_entries[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": tensor_digest(value),
        }
    entries = {
        "parameters": parameter_entries,
        "buffers": buffer_entries,
    }
    canonical = json.dumps(
        entries,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "parameter_names": ordered_parameters,
        "buffer_names": ordered_buffers,
        "entries": entries,
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def build_av1b_vendor_gdn_pair(
    spec: AV1BVendorGDNBridgeSpec | None = None,
    *,
    device: torch.device | str = torch.device("cuda:0"),
) -> AV1BVendorGDNBridgePair:
    """Construct one CPU common template, deepcopy, add seam, then move."""

    spec = AV1BVendorGDNBridgeSpec() if spec is None else spec
    if not isinstance(spec, AV1BVendorGDNBridgeSpec):
        raise TypeError("spec must be an AV1BVendorGDNBridgeSpec")
    resolved = _require_cuda_zero(device)
    if str(resolved) != spec.device:
        raise ValueError("device differs from the card-frozen spec device")

    torch.manual_seed(AV1B_SHARED_NAMED_SEED)
    torch.cuda.manual_seed_all(AV1B_SHARED_NAMED_SEED)
    with torch.device("cpu"):
        reference = AV1BVendorGDNBridgeArm(
            spec=spec,
            staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
        )
    candidate = copy.deepcopy(reference)
    candidate.enable_candidate_action_seam()

    reference = reference.to(device=resolved, dtype=torch.float32)
    candidate = candidate.to(device=resolved, dtype=torch.float32)
    reference.train(True)
    candidate.train(True)

    reference_names = reference.trainable_name_set
    candidate_names = candidate.trainable_name_set
    candidate_only = _candidate_only_names(spec.depth)
    common = reference_names
    if candidate_names != common | candidate_only:
        raise RuntimeError("candidate trainable allowlist differs from the card")
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    reference_buffers = dict(reference.named_buffers())
    candidate_buffers = dict(candidate.named_buffers())
    common_parameter_state = frozenset(reference_parameters)
    common_buffer_state = frozenset(reference_buffers)
    if not common_parameter_state <= frozenset(candidate_parameters):
        raise RuntimeError("candidate is missing common parameters")
    if not common_buffer_state <= frozenset(candidate_buffers):
        raise RuntimeError("candidate is missing common buffers")
    expected_candidate_extra_parameters = candidate_only
    if (
        frozenset(candidate_parameters) - common_parameter_state
        != expected_candidate_extra_parameters
    ):
        raise RuntimeError("candidate-only parameter state differs from card")
    expected_candidate_extra_buffers = frozenset({"no_action_slot"})
    if (
        frozenset(candidate_buffers) - common_buffer_state
        != expected_candidate_extra_buffers
    ):
        raise RuntimeError("candidate-only buffer state differs from card")
    for name in sorted(
        common_parameter_state,
        key=lambda value: value.encode("utf-8"),
    ):
        left = reference_parameters[name].detach()
        right = candidate_parameters[name].detach()
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.device != right.device
            or not torch.equal(left, right)
        ):
            raise RuntimeError(f"common theta0 parameter bytes differ for {name}")
    for name in sorted(
        common_buffer_state,
        key=lambda value: value.encode("utf-8"),
    ):
        left = reference_buffers[name].detach()
        right = candidate_buffers[name].detach()
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.device != right.device
            or not torch.equal(left, right)
        ):
            raise RuntimeError(f"common theta0 buffer bytes differ for {name}")
    if candidate.no_action_slot is None:
        raise RuntimeError("candidate no-action slot is absent")
    if candidate.no_action_slot.requires_grad:
        raise RuntimeError("candidate no-action slot must be update-free")
    if bool(candidate.no_action_slot.detach().count_nonzero()):
        raise RuntimeError("candidate no-action slot must be exact zero")
    if candidate.action_conditioner is None:
        raise RuntimeError("candidate action conditioner is absent")
    for layer_id in (0, 2):
        layer = candidate.action_conditioner[layer_id]
        if not isinstance(layer, nn.Linear) or layer.bias is not None:
            raise RuntimeError("candidate conditioner must be bias-free")
    for block in candidate.blocks:
        projection = block.action_output_projection
        if projection is None:
            raise RuntimeError("candidate block seam is absent")
        if bool(projection.weight.detach().count_nonzero()):
            raise RuntimeError("candidate action projection weight must be zero")
        if projection.bias is not None:
            raise RuntimeError("candidate action projection bias must be absent")

    manifests = {
        "reference_common": _named_state_manifest(
            reference,
            parameter_names=common_parameter_state,
            buffer_names=common_buffer_state,
        ),
        "candidate_common": _named_state_manifest(
            candidate,
            parameter_names=common_parameter_state,
            buffer_names=common_buffer_state,
        ),
        "candidate_only": _named_state_manifest(
            candidate,
            parameter_names=expected_candidate_extra_parameters,
            buffer_names=expected_candidate_extra_buffers,
        ),
    }
    return AV1BVendorGDNBridgePair(
        spec=spec,
        reference=reference,
        candidate=candidate,
        common_trainable_names=common,
        candidate_only_trainable_names=candidate_only,
        reference_trainable_names=reference_names,
        candidate_trainable_names=candidate_names,
        theta0_manifests=manifests,
    )


def build_av1b_pair(
    spec: AV1BVendorGDNBridgeSpec | None = None,
    *,
    device: torch.device | str = torch.device("cuda:0"),
) -> AV1BVendorGDNBridgePair:
    """Compatibility spelling for :func:`build_av1b_vendor_gdn_pair`."""

    return build_av1b_vendor_gdn_pair(spec, device=device)


def pure_torch_last_block_tail(
    post_vendor_hidden: Tensor,
    raw_action: Tensor,
    action_present_mask: Tensor,
    conditioner_input_weight: Tensor,
    conditioner_output_weight: Tensor,
    action_output_weight: Tensor,
    ffn_input_weight: Tensor,
    ffn_input_bias: Tensor,
    ffn_output_weight: Tensor,
    ffn_output_bias: Tensor,
    video_output_weight: Tensor,
    video_output_bias: Tensor,
    *,
    bypass_action: bool = False,
) -> Tensor:
    """Evaluate the bias-free active-only seam and pure-Torch output tail."""

    post_seam = post_vendor_hidden
    if not bypass_action:
        active_index = _active_action_indices(action_present_mask)
        active_raw = raw_action.reshape(-1, raw_action.shape[-1]).index_select(
            0,
            active_index,
        )
        active_condition = F.linear(
            F.silu(F.linear(active_raw, conditioner_input_weight, None)),
            conditioner_output_weight,
            None,
        )
        active_residual = F.linear(
            active_condition,
            action_output_weight,
            None,
        )
        residual = torch.zeros_like(
            post_vendor_hidden.reshape(-1, post_vendor_hidden.shape[-1])
        ).index_copy(0, active_index, active_residual).reshape_as(
            post_vendor_hidden
        )
        post_seam = torch.where(
            action_present_mask,
            post_vendor_hidden + residual,
            post_vendor_hidden,
        )
    hidden = post_seam + F.linear(
        F.silu(
            F.linear(
                _parameter_free_rms_norm(post_seam),
                ffn_input_weight,
                ffn_input_bias,
            )
        ),
        ffn_output_weight,
        ffn_output_bias,
    )
    return F.linear(hidden, video_output_weight, video_output_bias)


def _last_tail_fixed_inputs(
    arm: AV1BVendorGDNBridgeArm,
    output: AV1BVendorSequenceOutput,
) -> tuple[AV1BVendorGDNBlock, Tensor, Tensor, Tensor]:
    if not isinstance(arm, AV1BVendorGDNBridgeArm):
        raise TypeError("arm must be an AV1BVendorGDNBridgeArm")
    if arm.staging_variant is not CACHStagingVariant.CACH_A:
        raise ValueError("local seam JVP requires the candidate arm")
    if not isinstance(output, AV1BVendorSequenceOutput):
        raise TypeError("output must be an AV1BVendorSequenceOutput")
    if output.last_block_raw_action_condition is None:
        raise ValueError("candidate output lacks its raw action condition")
    if output.action_condition_valid_mask is None:
        raise ValueError("candidate output lacks its typed action mask")
    block = arm.blocks[-1]
    if block.action_output_projection is None:
        raise RuntimeError("last candidate action projection is absent")
    return (
        block,
        output.last_block_post_vendor_hidden.detach(),
        output.last_block_raw_action_condition.detach(),
        output.action_condition_valid_mask.detach(),
    )


def _alternating_unit_direction(value: Tensor, mask: Tensor | None = None) -> Tensor:
    flat_index = torch.arange(
        value.numel(),
        dtype=torch.int64,
        device=value.device,
    )
    direction = torch.where(
        flat_index.remainder(2) == 0,
        torch.ones_like(flat_index, dtype=value.dtype),
        -torch.ones_like(flat_index, dtype=value.dtype),
    ).reshape_as(value)
    if mask is not None:
        if mask.dtype is not torch.bool or tuple(mask.shape) != tuple(
            value.shape[:-1]
        ):
            raise ValueError("JVP valid mask shape or dtype differs")
        direction = torch.where(
            mask.unsqueeze(-1),
            direction,
            torch.zeros_like(direction),
        )
    norm = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise RuntimeError("JVP direction cannot be normalized")
    return direction / norm


def last_block_local_seam_parameter_jvp(
    arm: AV1BVendorGDNBridgeArm,
    output: AV1BVendorSequenceOutput,
) -> AV1BLocalJVPDiagnostic:
    """JVP over only the last block action-projection weight."""

    block, post_vendor, raw_action, action_present_mask = (
        _last_tail_fixed_inputs(arm, output)
    )
    projection = block.action_output_projection
    assert projection is not None
    conditioner = arm.action_conditioner
    if conditioner is None:
        raise RuntimeError("candidate action conditioner is absent")
    conditioner_input = conditioner[0]
    conditioner_output = conditioner[2]
    if not isinstance(conditioner_input, nn.Linear) or not isinstance(
        conditioner_output,
        nn.Linear,
    ):
        raise RuntimeError("candidate action conditioner topology differs")
    weight = projection.weight.detach()
    direction = _alternating_unit_direction(weight)

    def tail(dynamic_weight: Tensor) -> Tensor:
        return pure_torch_last_block_tail(
            post_vendor,
            raw_action,
            action_present_mask,
            conditioner_input.weight.detach(),
            conditioner_output.weight.detach(),
            dynamic_weight,
            block.ffn_input_projection.weight.detach(),
            block.ffn_input_projection.bias.detach(),
            block.ffn_output_projection.weight.detach(),
            block.ffn_output_projection.bias.detach(),
            arm.video_output_projection.weight.detach(),
            arm.video_output_projection.bias.detach(),
        )

    primal, tangent_output = torch.func.jvp(
        tail,
        (weight,),
        (direction,),
        strict=True,
    )
    rms = tangent_output.float().square().mean().sqrt()
    return AV1BLocalJVPDiagnostic(
        scope="LAST_BLOCK_POST_VENDOR_ACTION_PROJECTION_WEIGHT_PURE_TORCH_TAIL",
        primal=primal,
        tangent_output=tangent_output,
        input_direction=direction,
        rms=rms,
    )


def last_block_local_action_condition_jvp(
    arm: AV1BVendorGDNBridgeArm,
    output: AV1BVendorSequenceOutput,
    *,
    mode: ActionMode = "correct",
) -> AV1BLocalJVPDiagnostic:
    """JVP over raw action; inactive modes structurally ignore the primal."""

    if mode not in ("correct", "no_action", "seam_disabled"):
        raise ValueError("local raw-action JVP mode differs from the card")
    block, post_vendor, raw_action, action_present_mask = (
        _last_tail_fixed_inputs(arm, output)
    )
    projection = block.action_output_projection
    assert projection is not None
    conditioner = arm.action_conditioner
    if conditioner is None:
        raise RuntimeError("candidate action conditioner is absent")
    conditioner_input = conditioner[0]
    conditioner_output = conditioner[2]
    if not isinstance(conditioner_input, nn.Linear) or not isinstance(
        conditioner_output,
        nn.Linear,
    ):
        raise RuntimeError("candidate action conditioner topology differs")
    tangent_mask = torch.tensor(
        (False, True, True, True, True),
        dtype=torch.bool,
        device=raw_action.device,
    ).view(1, 5).expand(raw_action.shape[0], 5)
    direction = _alternating_unit_direction(raw_action, tangent_mask)
    bypass_action = mode in ("no_action", "seam_disabled")

    def tail(dynamic_raw_action: Tensor) -> Tensor:
        return pure_torch_last_block_tail(
            post_vendor,
            dynamic_raw_action,
            action_present_mask,
            conditioner_input.weight.detach(),
            conditioner_output.weight.detach(),
            projection.weight.detach(),
            block.ffn_input_projection.weight.detach(),
            block.ffn_input_projection.bias.detach(),
            block.ffn_output_projection.weight.detach(),
            block.ffn_output_projection.bias.detach(),
            arm.video_output_projection.weight.detach(),
            arm.video_output_projection.bias.detach(),
            bypass_action=bypass_action,
        )

    primal, tangent_output = torch.func.jvp(
        tail,
        (raw_action,),
        (direction,),
        strict=not bypass_action,
    )
    rms = tangent_output.float().square().mean().sqrt()
    return AV1BLocalJVPDiagnostic(
        scope=(
            "LAST_BLOCK_POST_VENDOR_RAW_ACTION_PURE_TORCH_TAIL_"
            + mode.upper()
        ),
        primal=primal,
        tangent_output=tangent_output,
        input_direction=direction,
        rms=rms,
    )


def _screen_emit(
    callback: Callable[[Mapping[str, object]], None] | None,
    event: str,
    payload: Mapping[str, object],
) -> None:
    if callback is not None:
        callback({"event": event, "payload": dict(payload)})


def _screen_require_config(config: Mapping[str, object]) -> None:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if config.get("schema") != "cach.cach_a2_r1.zero_anchor_vendor_gdn_bridge.config.v1":
        raise ValueError("CACH-A2 config schema differs")
    topology = config.get("topology")
    optimizer = config.get("optimizer")
    recipe = config.get("recipe")
    runtime = config.get("runtime")
    if not all(
        isinstance(value, Mapping)
        for value in (topology, optimizer, recipe, runtime)
    ):
        raise TypeError("config topology/optimizer/recipe/runtime must map")
    assert isinstance(topology, Mapping)
    assert isinstance(optimizer, Mapping)
    assert isinstance(recipe, Mapping)
    assert isinstance(runtime, Mapping)
    expected_topology: Mapping[str, object] = {
        "batch_size": 8,
        "counterfactual_pair_count": 4,
        "latent_channels": 3,
        "valid_raw_count": 33,
        "valid_latent_count": 5,
        "valid_action_count": 32,
        "action_dim": 20,
        "context_dim": 4,
        "vendor_hidden_dim": 64,
        "vendor_heads": 2,
        "vendor_head_dim": 32,
        "vendor_gdn_depth": 20,
        "vendor_conv_kernel_size": 0,
        "vendor_k_conv_only": True,
        "vendor_qk_norm": True,
        "vendor_use_output_gate": True,
        "vendor_use_autograd_kernel": True,
        "frame_count": 5,
        "HW": [5, 1, 1],
        "chunk_count": 2,
        "chunk_boundaries": [0, 3, 5],
        "chunk_size": None,
        "chunk_split_strategy": "uniform",
        "chunk_valid_latent_counts": [3, 2],
        "temporal_compression": 8,
        "video_stride": 1,
        "scored_future_horizons": [1, 2, 3, 4],
        "primary_horizons": [1, 2, 4],
        "ffn_hidden_dim": 128,
        "action_conditioner": "BIAS_FREE_LINEAR_20_TO_64_SILU_BIAS_FREE_LINEAR_64_TO_20",
        "action_output_projection": "BIAS_FREE_LINEAR_20_TO_64_ZERO_INIT_AFTER_VENDOR_GDN_BEFORE_FFN",
        "action_presence_mask_correct_or_shuffle": [False, True, True, True, True],
        "action_presence_mask_no_action_or_seam_disabled": [False, False, False, False, False],
        "inactive_algorithm": "GATHER_ACTIVE_CONDITION_PROJECT_SCATTER_THEN_DIRECT_WHERE_IDENTITY",
        "action_prediction_loss_weight": 0.0,
        "frame_valid_mask": None,
        "persistent_state_input_or_output": False,
        "every_forward_is_one_complete_five_frame_vendor_call": True,
    }
    for name, expected in expected_topology.items():
        if topology.get(name) != expected:
            raise ValueError(f"config topology.{name} differs from CACH-A2")
    expected_optimizer: Mapping[str, object] = {
        "name": "AdamW",
        "learning_rate": 0.003,
        "betas": [0.9, 0.99],
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "gradient_clip_global_l2": 1.0,
        "optimizer_steps_per_arm": 300,
        "metric_steps": [0, 300],
        "best_step_selection": False,
        "intermediate_checkpoint_selection": False,
        "write_optimizer_state": False,
        "unlisted_trainable_parameter_allowed": False,
    }
    for name, expected in expected_optimizer.items():
        if optimizer.get(name) != expected:
            raise ValueError(f"config optimizer.{name} differs from CACH-A2")
    if recipe.get("task_recipe_sha256") != AV1B_TASK_RECIPE_SHA256:
        raise ValueError("config task recipe digest differs")
    if runtime.get("device") != "cuda:0" or runtime.get("dtype") != "float32":
        raise ValueError("config runtime device/dtype differs")
    for forbidden in (
        "allow_real_data",
        "allow_checkpoint",
        "allow_proxy_or_reference_fallback",
    ):
        if runtime.get(forbidden) is not False:
            raise ValueError(f"config runtime.{forbidden} must be false")


def _screen_task_manifest(task: AV1BVendorTask) -> Mapping[str, object]:
    tensors: Mapping[str, Tensor] = {
        "global_actions": task.global_actions,
        "noisy_video": task.noisy_video,
        "base_target": task.base_target,
        "video_target": task.video_target,
        "context": task.context,
        "context_mask": task.context_mask,
        "proprio": task.proprio,
        "video_timestep": task.video_timestep,
        "action_timestep": task.action_timestep,
        "score_mask": task.score_mask,
    }
    for chunk in task.chunks:
        tensors = {
            **tensors,
            f"chunk{chunk.chunk_id}_actions": chunk.actions,
            f"chunk{chunk.chunk_id}_frame_valid_mask": (
                chunk.frame_valid_mask
            ),
            f"chunk{chunk.chunk_id}_action_valid_mask": (
                chunk.action_valid_mask
            ),
        }
    device_entries = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": tensor_digest(value),
        }
        for name, value in sorted(tensors.items())
    }
    material = json.dumps(
        device_entries,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "recipe_sha256": task.recipe_digest,
        "source_cpu_tensor_digests": dict(task.source_tensor_digests),
        "cuda_tensor_entries": device_entries,
        "cuda_manifest_sha256": hashlib.sha256(material).hexdigest(),
    }


def _screen_data_contract(task: AV1BVendorTask) -> Mapping[str, bool]:
    transferred: dict[str, Tensor] = {
        "global_actions": task.global_actions,
        "noisy_video": task.noisy_video,
        "base_target": task.base_target,
        "video_target": task.video_target,
        "context": task.context,
        "context_mask": task.context_mask,
        "proprio": task.proprio,
        "video_timestep": task.video_timestep,
        "action_timestep": task.action_timestep,
        "score_mask": task.score_mask,
    }
    for chunk in task.chunks:
        transferred[f"chunk{chunk.chunk_id}_actions"] = chunk.actions
        transferred[f"chunk{chunk.chunk_id}_frame_valid_mask"] = (
            chunk.frame_valid_mask
        )
        transferred[f"chunk{chunk.chunk_id}_action_valid_mask"] = (
            chunk.action_valid_mask
        )
    source_transfer_exact = all(
        task.source_tensor_digests.get(name)
        == tensor_digest(value.detach().to(device="cpu"))
        for name, value in transferred.items()
    )
    non_action = (
        task.noisy_video,
        task.base_target,
        task.context,
        task.context_mask,
        task.proprio,
        task.video_timestep,
        task.action_timestep,
    )
    non_action_pairs_equal = all(
        torch.equal(value[0::2], value[1::2]) for value in non_action
    )
    action_pairs_are_negations = torch.equal(
        task.global_actions[0::2], -task.global_actions[1::2]
    )
    target_pairs_differ = all(
        not torch.equal(
            task.video_target[0::2, horizon],
            task.video_target[1::2, horizon],
        )
        for horizon in (1, 2, 3, 4)
    )
    finite_floating_tensors = all(
        bool(torch.isfinite(value).all())
        for value in transferred.values()
        if value.is_floating_point()
    )
    return {
        "source_cpu_to_cuda_bytes_exact": source_transfer_exact,
        "counterfactual_non_action_pairs_equal": non_action_pairs_equal,
        "counterfactual_action_pairs_are_negations": (
            action_pairs_are_negations
        ),
        "counterfactual_target_pairs_differ_at_all_scored_horizons": (
            target_pairs_differ
        ),
        "counterfactual_row_order_exact": (
            task.row_to_pair == (0, 0, 1, 1, 2, 2, 3, 3)
            and task.row_sign == (1, -1, 1, -1, 1, -1, 1, -1)
        ),
        "all_floating_task_tensors_finite": finite_floating_tensors,
    }


def _screen_train_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction[:, 1:5] - target[:, 1:5]).square().mean()


def _screen_mse_by_horizon(
    prediction: Tensor,
    target: Tensor,
) -> dict[str, float]:
    return {
        str(horizon): float(
            (prediction[:, horizon] - target[:, horizon])
            .double()
            .square()
            .mean()
            .item()
        )
        for horizon in (1, 2, 3, 4)
    }


def _screen_primary(mse: Mapping[str, float]) -> float:
    return 0.20 * mse["1"] + 0.30 * mse["2"] + 0.50 * mse["4"]


def _screen_counterfactual_delta_nmse_by_horizon(
    prediction: Tensor,
    target: Tensor,
) -> dict[str, float]:
    predicted_delta = prediction[0::2] - prediction[1::2]
    target_delta = target[0::2] - target[1::2]
    result: dict[str, float] = {}
    for horizon in (1, 2, 3, 4):
        residual = predicted_delta[:, horizon] - target_delta[:, horizon]
        numerator = residual.double().square().sum()
        denominator = target_delta[:, horizon].double().square().sum()
        result[str(horizon)] = float(
            (numerator / (denominator + 1.0e-12)).item()
        )
    return result


def _screen_counterfactual_delta_nmse(
    prediction: Tensor,
    target: Tensor,
) -> float:
    predicted_delta = prediction[0::2, 1:5] - prediction[1::2, 1:5]
    target_delta = target[0::2, 1:5] - target[1::2, 1:5]
    numerator = (predicted_delta - target_delta).double().square().sum()
    denominator = target_delta.double().square().sum()
    return float((numerator / (denominator + 1.0e-12)).item())


def _screen_gap_by_horizon(
    altered: Mapping[str, float],
    correct: Mapping[str, float],
) -> dict[str, float]:
    return {
        horizon: (
            (altered[horizon] - correct[horizon])
            / max(altered[horizon], 1.0e-12)
        )
        for horizon in ("1", "2", "3", "4")
    }


def _screen_common_delta_metrics(
    prediction: Tensor,
    target: Tensor,
) -> Mapping[str, object]:
    """Card-exact float64 paired common/half-delta decomposition."""

    epsilon = 1.0e-12

    def one(predicted: Tensor, expected: Tensor) -> dict[str, float]:
        predicted = predicted.detach().double()
        expected = expected.detach().double()
        pred_plus, pred_minus = predicted[0::2], predicted[1::2]
        target_plus, target_minus = expected[0::2], expected[1::2]
        common_hat = (pred_plus + pred_minus) / 2.0
        common_target = (target_plus + target_minus) / 2.0
        delta_hat = (pred_plus - pred_minus) / 2.0
        delta_target = (target_plus - target_minus) / 2.0
        common_mse = float(
            (common_hat - common_target).square().mean().item()
        )
        delta_error = delta_hat - delta_target
        delta_error_sum = float(delta_error.square().sum().item())
        delta_target_sum = float(delta_target.square().sum().item())
        delta_hat_sum = float(delta_hat.square().sum().item())
        delta_dot = float((delta_hat * delta_target).sum().item())
        target_energy = float(expected.square().mean().item())
        correct_mse = float((predicted - expected).square().mean().item())
        return {
            "paired_prediction_common_mse": common_mse,
            "half_delta_mse": float(delta_error.square().mean().item()),
            "half_delta_nmse": delta_error_sum
            / max(delta_target_sum, epsilon),
            "half_delta_energy_ratio": delta_hat_sum
            / max(delta_target_sum, epsilon),
            "half_delta_alignment_cosine": delta_dot
            / max(
                math.sqrt(delta_hat_sum * delta_target_sum),
                epsilon,
            ),
            "target_total_energy": target_energy,
            "paired_common_mse_div_target_total_energy": common_mse
            / max(target_energy, epsilon),
            "correct_mse_div_target_total_energy": correct_mse
            / max(target_energy, epsilon),
        }

    future_prediction = prediction[:, 1:5]
    future_target = target[:, 1:5]
    aggregate = one(future_prediction, future_target)
    by_horizon = {
        str(horizon): one(
            prediction[:, horizon : horizon + 1],
            target[:, horizon : horizon + 1],
        )
        for horizon in (1, 2, 3, 4)
    }
    return {"aggregate": aggregate, "by_horizon": by_horizon}


def _screen_loss_window_statistics(
    loss_trace: tuple[float, ...],
) -> Mapping[str, object]:
    if len(loss_trace) != 301:
        raise ValueError("loss trace must contain 301 fixed points")

    def one(window: int) -> dict[str, float]:
        values = tuple(float(value) for value in loss_trace[-window:])
        ordered = sorted(values)
        middle = window // 2
        median = (
            ordered[middle]
            if window % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        p90_index = math.ceil(0.90 * window) - 1
        return {
            "mean": sum(values) / float(window),
            "median": median,
            "p90_nearest_rank": ordered[p90_index],
            "min": ordered[0],
            "max": ordered[-1],
        }

    trace_min = min(loss_trace)
    return {
        "last25": one(25),
        "last50": one(50),
        "final_over_trace_min_ratio": loss_trace[-1]
        / max(trace_min, 1.0e-12),
    }


def _screen_numeric_leaves(value: object) -> tuple[float, ...]:
    leaves: list[float] = []

    def visit(item: object) -> None:
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float)):
            leaves.append(float(item))
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(leaves)


def _screen_no_action_candidate_gradients(
    arm: AV1BVendorGDNBridgeArm,
    task: AV1BVendorTask,
) -> Mapping[str, object]:
    named = dict(arm.named_parameters())
    candidate_only = sorted(
        _candidate_only_names(arm.spec.depth),
        key=lambda value: value.encode("utf-8"),
    )
    parameters = tuple(named[name] for name in candidate_only)
    output = arm(
        task,
        mode="no_action",
        capture_diagnostics=True,
        run_action_diagnostic=False,
    )
    gradients = torch.autograd.grad(
        output.video_prediction.float().sum(),
        parameters,
        allow_unused=True,
        materialize_grads=False,
    )
    non_none = [
        name
        for name, gradient in zip(candidate_only, gradients, strict=True)
        if gradient is not None
    ]
    gradient_tensors = [
        gradient for gradient in gradients if gradient is not None
    ]
    return {
        "all_none": not non_none,
        "non_none_names": non_none,
        "canonical_rms": _screen_rms(gradient_tensors),
        "output": output,
    }


def _screen_padding_seam_helper_fixture(
    arm: AV1BVendorGDNBridgeArm,
) -> Mapping[str, object]:
    if arm.action_conditioner is None:
        raise RuntimeError("candidate action conditioner is absent")
    projection = arm.blocks[0].action_output_projection
    if projection is None:
        raise RuntimeError("candidate first action projection is absent")
    device = next(arm.parameters()).device
    hidden = torch.arange(192, dtype=torch.float32, device=device).reshape(
        1,
        3,
        64,
    ) / 193.0
    raw_action = torch.arange(
        60,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 3, 20) / 61.0
    mask = torch.tensor(
        (True, False, False),
        dtype=torch.bool,
        device=device,
    ).view(1, 3, 1)
    conditioned, conditioner_calls = condition_zero_anchor_actions(
        raw_action,
        mask,
        arm.action_conditioner,
    )
    post_seam, residual, projection_calls = apply_zero_anchor_action_residual(
        hidden,
        conditioned,
        mask,
        projection,
    )
    inactive = (~mask).expand_as(hidden)
    difference = post_seam.masked_select(inactive) - hidden.masked_select(
        inactive
    )
    return {
        "bitwise_equal": torch.equal(
            post_seam.masked_select(inactive),
            hidden.masked_select(inactive),
        ),
        "max_abs": float(difference.detach().abs().max().item()),
        "conditioner_calls": conditioner_calls,
        "projection_calls": projection_calls,
        "inactive_residual_exact_zero": not bool(
            residual.masked_select(inactive).detach().count_nonzero()
        ),
        "vendor_used": False,
    }


def _screen_zero_origin_diagnostics(
    arm: AV1BVendorGDNBridgeArm,
) -> Mapping[str, object]:
    """Evaluate the bias-free composite seam at the exact zero origin."""

    if arm.action_conditioner is None:
        raise RuntimeError("candidate action conditioner is absent")
    device = next(arm.parameters()).device
    zero_action = torch.zeros(
        (1, arm.spec.action_dim),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        phi_zero = arm.action_conditioner(zero_action)
        residuals = tuple(
            block.action_output_projection(phi_zero)
            for block in arm.blocks
            if block.action_output_projection is not None
        )
    if len(residuals) != arm.spec.depth:
        raise RuntimeError("candidate action projection depth differs")
    action_state_keys = sorted(
        (
            name
            for name in arm.state_dict()
            if name.startswith("action_conditioner.")
            or ".action_output_projection." in name
        ),
        key=lambda value: value.encode("utf-8"),
    )
    bias_keys = [name for name in action_state_keys if name.endswith(".bias")]
    return {
        "phi_zero_bitwise_exact_zero": not bool(
            phi_zero.detach().count_nonzero()
        ),
        "all_block_residual_zero_bitwise_exact_zero": all(
            not bool(value.detach().count_nonzero()) for value in residuals
        ),
        "candidate_action_state_keys": action_state_keys,
        "candidate_action_bias_state_keys": bias_keys,
        "candidate_action_bias_state_absent": not bias_keys,
    }


def _screen_block_bypass_diagnostics(
    output: AV1BVendorSequenceOutput,
) -> Mapping[str, object]:
    if not (
        len(output.block_post_vendor_hidden)
        == len(output.block_post_seam_hidden)
        == len(output.block_action_residuals)
        == 20
    ):
        raise RuntimeError("block bypass diagnostic depth differs")
    bootstrap_equal = []
    bootstrap_max_abs = []
    residual_rms = []
    for post_vendor, post_seam, residual in zip(
        output.block_post_vendor_hidden,
        output.block_post_seam_hidden,
        output.block_action_residuals,
        strict=True,
    ):
        bootstrap_equal.append(
            torch.equal(post_vendor[:, 0], post_seam[:, 0])
        )
        bootstrap_max_abs.append(
            float(
                (post_vendor[:, 0] - post_seam[:, 0])
                .detach()
                .abs()
                .max()
                .item()
            )
        )
        residual_rms.append(
            float(residual.detach().double().square().mean().sqrt().item())
        )
    return {
        "bootstrap_bitwise_equal_by_block": bootstrap_equal,
        "bootstrap_max_abs_by_block": bootstrap_max_abs,
        "action_residual_rms_by_block": residual_rms,
        "all_action_residuals_exact_zero": all(
            not bool(value.detach().count_nonzero())
            for value in output.block_action_residuals
        ),
    }


def _screen_rms(tensors: list[Tensor], *, include_none_as_zero: int = 0) -> float:
    numerator = 0.0
    denominator = int(include_none_as_zero)
    for value in tensors:
        detached = value.detach().double()
        numerator += float(detached.square().sum().item())
        denominator += detached.numel()
    if denominator <= 0:
        return 0.0
    return math.sqrt(numerator / denominator)


def _screen_seam_weights(
    arm: AV1BVendorGDNBridgeArm,
) -> dict[str, Tensor]:
    return {
        name: parameter
        for name, parameter in arm.named_parameters()
        if parameter.requires_grad
        and name.endswith("action_output_projection.weight")
    }


def _screen_seam_update_rms(
    arm: AV1BVendorGDNBridgeArm,
    initial: Mapping[str, Tensor],
) -> float:
    current = _screen_seam_weights(arm)
    if set(current) != set(initial):
        raise RuntimeError("candidate seam weight names changed during training")
    return _screen_rms(
        [current[name].detach() - initial[name] for name in sorted(current)]
    )


def _screen_gradient_rms(
    arm: AV1BVendorGDNBridgeArm,
    names: list[str],
) -> float:
    parameters = dict(arm.named_parameters())
    gradients: list[Tensor] = []
    missing_numel = 0
    for name in names:
        parameter = parameters[name]
        if parameter.grad is None:
            missing_numel += parameter.numel()
        else:
            gradients.append(parameter.grad)
    return _screen_rms(gradients, include_none_as_zero=missing_numel)


def _screen_all_parameters_finite(module: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(value.detach()).all())
        for value in module.parameters()
    )


def _screen_optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor) and not bool(
                torch.isfinite(value.detach()).all()
            ):
                return False
    return True


@dataclass(frozen=True)
class _AV1BTrainResult:
    loss_trace: tuple[float, ...]
    final_output: AV1BVendorSequenceOutput = field(repr=False, compare=False)
    optimizer_steps: int
    seam_grad_rms_step0: float
    vendor_grad_rms_step0: float
    post_vendor_hidden_gradient_finite_step0: bool
    optimizer_scalars_finite: bool
    parameters_finite: bool


def _screen_train_arm(
    arm: AV1BVendorGDNBridgeArm,
    task: AV1BVendorTask,
    trainable_names: frozenset[str],
    *,
    progress_callback: Callable[[Mapping[str, object]], None] | None,
) -> _AV1BTrainResult:
    named = dict(arm.named_parameters())
    if frozenset(
        name for name, value in named.items() if value.requires_grad
    ) != trainable_names:
        raise RuntimeError(f"{arm.arm_name} trainable allowlist changed")
    ordered_names = sorted(trainable_names, key=lambda value: value.encode("utf-8"))
    parameters = [named[name] for name in ordered_names]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=0.003,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    loss_trace: list[float] = []
    seam_grad = 0.0
    vendor_grad = 0.0
    hidden_gradient_finite = False
    optimizer_scalars_finite = True
    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        output = arm(
            task,
            mode="correct",
            target_override=None,
            capture_diagnostics=step == 0,
            run_action_diagnostic=False,
        )
        loss = _screen_train_loss(output.video_prediction, task.video_target)
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"{arm.arm_name} loss is non-finite")
        loss_trace.append(loss_value)
        loss.backward()
        if step == 0:
            seam_names = [
                name
                for name in ordered_names
                if name.endswith("action_output_projection.weight")
            ]
            if seam_names:
                seam_grad = _screen_gradient_rms(arm, seam_names)
            vendor_names = [
                name
                for name in ordered_names
                if name.startswith("blocks.") and ".vendor." in name
            ]
            vendor_grad = _screen_gradient_rms(arm, vendor_names)
            observed_gradient = output.last_block_post_vendor_hidden.grad
            hidden_gradient_finite = (
                observed_gradient is not None
                and bool(torch.isfinite(observed_gradient.detach()).all())
            )
        unclipped = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        if not bool(torch.isfinite(unclipped.detach())):
            raise FloatingPointError(f"{arm.arm_name} gradient norm is non-finite")
        optimizer.step()
        optimizer_scalars_finite = (
            optimizer_scalars_finite and _screen_optimizer_finite(optimizer)
        )
        if step in (0, 49, 99, 149, 199, 249, 299):
            _screen_emit(
                progress_callback,
                "optimizer_progress",
                {
                    "arm": arm.arm_name,
                    "optimizer_steps_completed": step + 1,
                    "loss_before_update": loss_value,
                },
            )

    final_output = arm(
        task,
        mode="correct",
        target_override=None,
        capture_diagnostics=True,
        run_action_diagnostic=False,
    )
    final_loss = float(
        _screen_train_loss(
            final_output.video_prediction,
            task.video_target,
        ).detach().item()
    )
    if not math.isfinite(final_loss):
        raise FloatingPointError(f"{arm.arm_name} final loss is non-finite")
    loss_trace.append(final_loss)
    if len(loss_trace) != 301:
        raise RuntimeError("loss trace must contain steps 0 through 300")
    return _AV1BTrainResult(
        loss_trace=tuple(loss_trace),
        final_output=final_output,
        optimizer_steps=300,
        seam_grad_rms_step0=seam_grad,
        vendor_grad_rms_step0=vendor_grad,
        post_vendor_hidden_gradient_finite_step0=hidden_gradient_finite,
        optimizer_scalars_finite=optimizer_scalars_finite,
        parameters_finite=_screen_all_parameters_finite(arm),
    )


def _screen_prefix_invariance(
    arm: AV1BVendorGDNBridgeArm,
    task: AV1BVendorTask,
) -> tuple[float, float]:
    changed_video = task.noisy_video.detach().clone()
    changed_video[:, 3:5] = changed_video[:, 3:5] + 0.375
    video_task = replace(task, noisy_video=changed_video)
    changed_actions = task.global_actions.detach().clone()
    changed_actions[:, 16:32] = -changed_actions[:, 16:32] + 0.125
    action_task = replace(task, global_actions=changed_actions)
    with torch.no_grad():
        baseline = arm(
            task,
            mode="correct",
            capture_diagnostics=False,
            run_action_diagnostic=False,
        ).video_prediction
        video_output = arm(
            video_task,
            mode="correct",
            capture_diagnostics=False,
            run_action_diagnostic=False,
        ).video_prediction
        action_output = arm(
            action_task,
            mode="correct",
            capture_diagnostics=False,
            run_action_diagnostic=False,
        ).video_prediction
    prefix = baseline[:, :3]
    video_delta = (video_output[:, :3] - prefix).abs()
    action_delta = (action_output[:, :3] - prefix).abs()
    max_abs_tensor = torch.maximum(video_delta.max(), action_delta.max())
    denominator = prefix.detach().abs().max().clamp_min(1.0e-12)
    max_abs = float(max_abs_tensor.double().item())
    max_rel = float((max_abs_tensor.double() / denominator.double()).item())
    return max_abs, max_rel


def _screen_jit_inventory(
    reference: AV1BVendorGDNBridgeArm,
    candidate: AV1BVendorGDNBridgeArm,
) -> Mapping[str, object]:
    triton = __import__("triton")
    properties = torch.cuda.get_device_properties(0)
    cache_text = os.environ.get("TRITON_CACHE_DIR")
    if not cache_text:
        raise RuntimeError("TRITON_CACHE_DIR is required for CACH-A2")
    cache_root = Path(cache_text)
    if not cache_root.is_dir() or cache_root.is_symlink():
        raise RuntimeError("TRITON_CACHE_DIR must be a real directory")
    cache_files: list[Mapping[str, object]] = []
    total_cache_bytes = 0
    for path in sorted(
        cache_root.rglob("*"),
        key=lambda value: value.relative_to(cache_root).as_posix().encode(
            "utf-8"
        ),
    ):
        relative = path.relative_to(cache_root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in Triton cache: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(
                f"non-regular Triton cache entry is forbidden: {relative}"
            )
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as stream:
            for payload in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(payload)
                size_bytes += len(payload)
        total_cache_bytes += size_bytes
        cache_files.append(
            {
                "path": relative,
                "size_bytes": size_bytes,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "triton_version": getattr(triton, "__version__", "UNKNOWN"),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_current_device": torch.cuda.current_device(),
        "cuda_device_name": properties.name,
        "cuda_compute_capability": [properties.major, properties.minor],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "triton_cache_dir": cache_text,
        "triton_cache_file_count": len(cache_files),
        "triton_cache_total_bytes": total_cache_bytes,
        "triton_cache_files": cache_files,
        "gdn_disable_compile": os.environ.get("GDN_DISABLE_COMPILE"),
        "torchdynamo_disable": os.environ.get("TORCHDYNAMO_DISABLE"),
        "fused_gdn_precision": os.environ.get("FUSED_GDN_PRECISION"),
        "reference_vendor_forward_call_count": (
            reference.vendor_forward_call_count
        ),
        "candidate_vendor_forward_call_count": (
            candidate.vendor_forward_call_count
        ),
        "vendor_kernel_executed": (
            reference.vendor_forward_call_count
            + candidate.vendor_forward_call_count
            > 0
        ),
        "inventory_scope": "RUNTIME_FACTS_AND_READ_ONLY_TRITON_CACHE_FILE_HASHES",
    }


def _screen_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def run_cach_a2_screen(
    config: Mapping[str, object],
    *,
    expected_gpu_uuid: str,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> Mapping[str, object]:
    """Run the complete in-memory CACH-A2 two-arm non-formal screen.

    The caller owns preflight, root lifecycle, artifact publication, terminal
    classification, and freezing.  This function only reads the in-root
    Triton cache for its final inventory; it performs no filesystem write,
    network, checkpoint, real-data, token-ledger, or root-lifecycle operation.
    """

    started = time.monotonic()
    _screen_require_config(config)
    if not isinstance(expected_gpu_uuid, str) or not expected_gpu_uuid.startswith(
        "GPU-"
    ):
        raise ValueError("execution-bound GPU UUID is invalid")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from execution binding")
    if os.environ.get("FUSED_GDN_PRECISION") != "0":
        raise RuntimeError("CACH-A2 requires FUSED_GDN_PRECISION=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CACH-A2 requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    spec = AV1BVendorGDNBridgeSpec()

    # The task and its byte manifest exist before model construction.
    task = build_av1b_vendor_task(spec, device=device)
    task_manifest_before = _screen_task_manifest(task)
    data_checks = _screen_data_contract(task)
    data_contract_valid = all(data_checks.values())
    _screen_emit(
        progress_callback,
        "task_tensor_manifest_ready",
        {"task_tensor_manifest": task_manifest_before},
    )
    pair = build_av1b_vendor_gdn_pair(spec, device=device)
    theta0_manifests: Mapping[str, object] = {
        "reference": {
            "common": pair.theta0_manifests["reference_common"],
        },
        "candidate": {
            "common": pair.theta0_manifests["candidate_common"],
            "candidate_only": pair.theta0_manifests["candidate_only"],
        },
    }
    _screen_emit(
        progress_callback,
        "theta0_manifests_ready",
        theta0_manifests,
    )

    # All theta0 identity forwards precede either arm's first update.
    with torch.no_grad():
        reference_theta0 = {
            mode: pair.reference(
                task,
                mode=mode,
                capture_diagnostics=mode == "correct",
                run_action_diagnostic=False,
            )
            for mode in ("correct", "shuffle", "no_action")
        }
        candidate_theta0 = {
            mode: pair.candidate(
                task,
                mode=mode,
                capture_diagnostics=True,
                run_action_diagnostic=False,
            )
            for mode in (
                "correct",
                "shuffle",
                "no_action",
                "seam_disabled",
            )
        }
    theta0_parameter_jvp = last_block_local_seam_parameter_jvp(
        pair.candidate,
        candidate_theta0["correct"],
    )
    if not theta0_parameter_jvp.finite:
        raise FloatingPointError("theta0 local seam-parameter JVP is non-finite")
    theta0_correct_action_jvp = last_block_local_action_condition_jvp(
        pair.candidate,
        candidate_theta0["correct"],
        mode="correct",
    )
    theta0_no_action_jvp = last_block_local_action_condition_jvp(
        pair.candidate,
        candidate_theta0["no_action"],
        mode="no_action",
    )
    theta0_no_action_gradient = dict(
        _screen_no_action_candidate_gradients(pair.candidate, task)
    )
    theta0_no_action_gradient_output = theta0_no_action_gradient.pop("output")
    assert isinstance(theta0_no_action_gradient_output, AV1BVendorSequenceOutput)
    with torch.no_grad():
        theta0_padding_fixture = _screen_padding_seam_helper_fixture(
            pair.candidate
        )
        theta0_zero_origin = _screen_zero_origin_diagnostics(pair.candidate)
    theta0_projection_weights_exact_zero = all(
        block.action_output_projection is not None
        and not bool(
            block.action_output_projection.weight.detach().count_nonzero()
        )
        for block in pair.candidate.blocks
    )
    theta0_correct_bypass = _screen_block_bypass_diagnostics(
        candidate_theta0["correct"]
    )
    theta0_no_action_bypass = _screen_block_bypass_diagnostics(
        candidate_theta0["no_action"]
    )
    candidate_theta0_post_vendor = (
        candidate_theta0["correct"]
        .last_block_post_vendor_hidden.detach()
        .clone()
    )

    reference_candidate_equal = all(
        torch.equal(
            reference_theta0[mode].video_prediction,
            candidate_theta0[mode].video_prediction,
        )
        for mode in ("correct", "shuffle", "no_action")
    )
    reference_modes_equal = all(
        torch.equal(
            reference_theta0["correct"].video_prediction,
            reference_theta0[mode].video_prediction,
        )
        for mode in ("shuffle", "no_action")
    )
    candidate_modes_equal = all(
        torch.equal(
            candidate_theta0["correct"].video_prediction,
            candidate_theta0[mode].video_prediction,
        )
        for mode in ("shuffle", "no_action", "seam_disabled")
    )
    theta0_no_action_vs_seam_disabled = torch.equal(
        candidate_theta0["no_action"].video_prediction,
        candidate_theta0["seam_disabled"].video_prediction,
    )
    theta0_no_action_vs_seam_disabled_max_abs = float(
        (
            candidate_theta0["no_action"].video_prediction
            - candidate_theta0["seam_disabled"].video_prediction
        )
        .detach()
        .abs()
        .max()
        .item()
    )
    theta0_no_action = candidate_theta0["no_action"].video_prediction
    no_action_pair_equal_theta0 = torch.equal(
        theta0_no_action[0::2],
        theta0_no_action[1::2],
    )
    common_theta0_equal = (
        pair.theta0_manifests["reference_common"]["combined_sha256"]
        == pair.theta0_manifests["candidate_common"]["combined_sha256"]
    )
    raw_condition = candidate_theta0["correct"].last_block_raw_action_condition
    if raw_condition is None:
        raise RuntimeError("candidate theta0 raw action condition is absent")
    expected_condition = torch.cat(
        (
            torch.zeros(
                (spec.batch_size, 1, spec.action_dim),
                dtype=torch.float32,
                device=device,
            ),
            task.global_actions[:, [7, 15, 23, 31], :],
        ),
        dim=1,
    )
    reducer_exact = torch.equal(raw_condition, expected_condition)
    theta0_correct_mask = candidate_theta0[
        "correct"
    ].action_condition_valid_mask
    theta0_no_action_mask = candidate_theta0[
        "no_action"
    ].action_condition_valid_mask
    expected_present_mask = torch.tensor(
        (False, True, True, True, True),
        dtype=torch.bool,
        device=device,
    ).view(1, 5, 1).expand(spec.batch_size, 5, 1)
    expected_inactive_mask = torch.zeros_like(expected_present_mask)
    typed_mask_exact = (
        theta0_correct_mask is not None
        and theta0_no_action_mask is not None
        and torch.equal(theta0_correct_mask, expected_present_mask)
        and torch.equal(theta0_no_action_mask, expected_inactive_mask)
    )
    padding_zero = all(
        not bool(
            chunk.actions[:, chunk.layout.valid_action_count :, :]
            .detach()
            .count_nonzero()
        )
        for chunk in task.chunks
    )
    candidate_only_bias_absent = (
        pair.candidate.action_conditioner is not None
        and all(
            isinstance(pair.candidate.action_conditioner[layer_id], nn.Linear)
            and pair.candidate.action_conditioner[layer_id].bias is None
            for layer_id in (0, 2)
        )
        and all(
            block.action_output_projection is not None
            and block.action_output_projection.bias is None
            for block in pair.candidate.blocks
        )
        and not any(
            name.endswith(".bias")
            for name in _candidate_only_names(spec.depth)
        )
        and bool(theta0_zero_origin["candidate_action_bias_state_absent"])
    )

    # Target values are accepted as loss-only metadata and never enter forward.
    perturbed_target = (
        task.video_target.detach().clone().add(0.731).requires_grad_(True)
    )
    target_baseline = pair.candidate(
        task,
        mode="correct",
        target_override=task.video_target,
        capture_diagnostics=True,
        run_action_diagnostic=False,
    )
    target_output = pair.candidate(
        task,
        mode="correct",
        target_override=perturbed_target,
        capture_diagnostics=True,
        run_action_diagnostic=False,
    )
    target_prediction_changed = not torch.equal(
        target_output.video_prediction.detach(),
        target_baseline.video_prediction.detach(),
    )
    target_hidden_changed = not torch.equal(
        target_output.last_block_post_vendor_hidden.detach(),
        target_baseline.last_block_post_vendor_hidden.detach(),
    )
    target_gradient = torch.autograd.grad(
        target_output.video_prediction.float().sum(),
        perturbed_target,
        allow_unused=True,
        materialize_grads=False,
    )[0]
    target_reverse_nonzero = target_gradient is not None and bool(
        target_gradient.detach().count_nonzero()
    )
    del target_baseline, target_output, target_gradient, perturbed_target

    seam_initial = {
        name: value.detach().clone()
        for name, value in _screen_seam_weights(pair.candidate).items()
    }
    reference_train = _screen_train_arm(
        pair.reference,
        task,
        pair.reference_trainable_names,
        progress_callback=progress_callback,
    )
    candidate_train = _screen_train_arm(
        pair.candidate,
        task,
        pair.candidate_trainable_names,
        progress_callback=progress_callback,
    )
    final_action_jvp = last_block_local_action_condition_jvp(
        pair.candidate,
        candidate_train.final_output,
        mode="correct",
    )
    if not final_action_jvp.finite:
        raise FloatingPointError("step300 local raw-action JVP is non-finite")
    final_no_action_gradient = dict(
        _screen_no_action_candidate_gradients(pair.candidate, task)
    )
    final_no_action_gradient_output = final_no_action_gradient.pop("output")
    assert isinstance(final_no_action_gradient_output, AV1BVendorSequenceOutput)
    final_no_action_jvp = last_block_local_action_condition_jvp(
        pair.candidate,
        final_no_action_gradient_output,
        mode="no_action",
    )

    with torch.no_grad():
        candidate_shuffle = pair.candidate(
            task,
            mode="shuffle",
            capture_diagnostics=False,
            run_action_diagnostic=False,
        )
        candidate_no_action = pair.candidate(
            task,
            mode="no_action",
            capture_diagnostics=True,
            run_action_diagnostic=False,
        )
        candidate_seam_disabled = pair.candidate(
            task,
            mode="seam_disabled",
            capture_diagnostics=True,
            run_action_diagnostic=False,
        )
        final_padding_fixture = _screen_padding_seam_helper_fixture(
            pair.candidate
        )
        final_zero_origin = _screen_zero_origin_diagnostics(pair.candidate)
    final_correct_bypass = _screen_block_bypass_diagnostics(
        candidate_train.final_output
    )
    final_no_action_bypass = _screen_block_bypass_diagnostics(
        candidate_no_action
    )
    reference_prediction = reference_train.final_output.video_prediction.detach()
    candidate_prediction = candidate_train.final_output.video_prediction.detach()
    shuffle_prediction = candidate_shuffle.video_prediction.detach()
    no_action_prediction = candidate_no_action.video_prediction.detach()
    seam_disabled_prediction = (
        candidate_seam_disabled.video_prediction.detach()
    )
    target = task.video_target.detach()

    reference_mse = _screen_mse_by_horizon(reference_prediction, target)
    candidate_mse = _screen_mse_by_horizon(candidate_prediction, target)
    shuffle_mse = _screen_mse_by_horizon(shuffle_prediction, target)
    no_action_mse = _screen_mse_by_horizon(no_action_prediction, target)
    reference_primary = _screen_primary(reference_mse)
    candidate_primary = _screen_primary(candidate_mse)
    shuffle_primary = _screen_primary(shuffle_mse)
    no_action_primary = _screen_primary(no_action_mse)
    shuffle_gap_horizon = _screen_gap_by_horizon(shuffle_mse, candidate_mse)
    no_action_gap_horizon = _screen_gap_by_horizon(no_action_mse, candidate_mse)
    counterfactual_horizon = _screen_counterfactual_delta_nmse_by_horizon(
        candidate_prediction,
        target,
    )
    common_delta = _screen_common_delta_metrics(
        candidate_prediction,
        target,
    )
    reference_loss_stability = _screen_loss_window_statistics(
        reference_train.loss_trace
    )
    candidate_loss_stability = _screen_loss_window_statistics(
        candidate_train.loss_trace
    )
    seam_update = _screen_seam_update_rms(pair.candidate, seam_initial)
    prefix_abs, prefix_rel = _screen_prefix_invariance(pair.candidate, task)
    final_post_vendor_update = (
        candidate_train.final_output.last_block_post_vendor_hidden.detach()
        - candidate_theta0_post_vendor
    )
    task_manifest_after = _screen_task_manifest(task)

    final_no_action_vs_seam_disabled = torch.equal(
        no_action_prediction,
        seam_disabled_prediction,
    )
    final_no_action_vs_seam_disabled_max_abs = float(
        (no_action_prediction - seam_disabled_prediction)
        .abs()
        .max()
        .item()
    )

    candidate_no_action_pair_equal = torch.equal(
        no_action_prediction[0::2],
        no_action_prediction[1::2],
    )
    no_action_slot_valid = (
        pair.candidate.no_action_slot is not None
        and not pair.candidate.no_action_slot.requires_grad
        and not bool(pair.candidate.no_action_slot.detach().count_nonzero())
        and "no_action_slot" not in dict(pair.candidate.named_parameters())
        and "no_action_slot" in dict(pair.candidate.named_buffers())
    )
    outputs_finite = all(
        bool(torch.isfinite(value).all())
        for value in (
            reference_prediction,
            candidate_prediction,
            shuffle_prediction,
            no_action_prediction,
            seam_disabled_prediction,
        )
    )
    post_vendor_update_finite = bool(
        torch.isfinite(final_post_vendor_update).all()
    )
    exact_vendor_identity = all(
        block.vendor.use_autograd_kernel is True
        and type(block.vendor).__name__ == "ChunkCausalGDNTriton"
        and type(block.vendor).__module__
        == "diffusion.model.nets.sana_gdn_blocks_triton"
        for arm in (pair.reference, pair.candidate)
        for block in arm.blocks
    )
    actual_vendor_calls = (
        pair.reference.vendor_forward_call_count
        + pair.candidate.vendor_forward_call_count
    )
    jit_inventory = _screen_jit_inventory(pair.reference, pair.candidate)

    metrics: dict[str, object] = {
        "reference_correct_mse_by_horizon": reference_mse,
        "candidate_correct_mse_by_horizon": candidate_mse,
        "candidate_shuffle_mse_by_horizon": shuffle_mse,
        "candidate_no_action_mse_by_horizon": no_action_mse,
        "primary_reference_step300": reference_primary,
        "primary_candidate_correct_step300": candidate_primary,
        "primary_candidate_shuffle_step300": shuffle_primary,
        "primary_candidate_no_action_step300": no_action_primary,
        "seam_grad_rms_step0": candidate_train.seam_grad_rms_step0,
        "seam_update_rms_final": seam_update,
        "last_block_local_seam_parameter_jvp_rms_theta0": float(
            theta0_parameter_jvp.rms.detach().item()
        ),
        "last_block_local_action_condition_jvp_rms_step300": float(
            final_action_jvp.rms.detach().item()
        ),
        "last_block_local_raw_action_jvp_rms_step300": float(
            final_action_jvp.rms.detach().item()
        ),
        "last_block_local_raw_action_jvp_rms_theta0": float(
            theta0_correct_action_jvp.rms.detach().item()
        ),
        "no_action_raw_action_jvp_rms_theta0": float(
            theta0_no_action_jvp.rms.detach().item()
        ),
        "no_action_raw_action_jvp_rms_step300": float(
            final_no_action_jvp.rms.detach().item()
        ),
        "loss_drop_rel": (
            candidate_train.loss_trace[0]
            - candidate_train.loss_trace[-1]
        )
        / max(candidate_train.loss_trace[0], 1.0e-12),
        "counterfactual_delta_nmse_correct": (
            _screen_counterfactual_delta_nmse(candidate_prediction, target)
        ),
        "counterfactual_delta_nmse_correct_by_horizon": (
            counterfactual_horizon
        ),
        "shuffle_gap_rel": (
            (shuffle_primary - candidate_primary)
            / max(shuffle_primary, 1.0e-12)
        ),
        "shuffle_gap_rel_by_horizon": shuffle_gap_horizon,
        "no_action_gap_rel": (
            (no_action_primary - candidate_primary)
            / max(no_action_primary, 1.0e-12)
        ),
        "no_action_gap_rel_by_horizon": no_action_gap_horizon,
        "candidate_gain_vs_reference": (
            (reference_primary - candidate_primary)
            / max(reference_primary, 1.0e-12)
        ),
        "paired_common_delta": common_delta,
        "loss_stability": {
            "reference": reference_loss_stability,
            "candidate": candidate_loss_stability,
        },
    }
    finite_metric_values = all(
        math.isfinite(float(value))
        for value in (
            candidate_train.vendor_grad_rms_step0,
            prefix_abs,
            prefix_rel,
            *reference_train.loss_trace,
            *candidate_train.loss_trace,
            *_screen_numeric_leaves(metrics),
        )
    )
    identity_checks: Mapping[str, bool] = {
        "reference_candidate_video_output_bitwise_equal_all_modes": (
            reference_candidate_equal and candidate_modes_equal
        ),
        "reference_correct_shuffle_no_action_video_outputs_bitwise_equal": (
            reference_modes_equal
        ),
        "no_action_candidate_pair_predictions_bitwise_equal": (
            no_action_pair_equal_theta0 and candidate_no_action_pair_equal
        ),
        "common_theta0_parameter_bytes_equal": common_theta0_equal,
        "valid_action_exact_once_and_padding_zero": (
            reducer_exact and padding_zero
        ),
        "non_seam_bytes_bitwise_equal_across_modes": (
            task_manifest_before == task_manifest_after
        ),
        "typed_action_presence_mask_exact": typed_mask_exact,
        "same_arm_no_action_vs_seam_disabled_bitwise_equal_theta0_and_final": (
            theta0_no_action_vs_seam_disabled
            and final_no_action_vs_seam_disabled
        ),
        "same_arm_no_action_vs_seam_disabled_max_abs_exact_zero": (
            theta0_no_action_vs_seam_disabled_max_abs == 0.0
            and final_no_action_vs_seam_disabled_max_abs == 0.0
        ),
    }
    card_validity_checks: Mapping[str, bool] = {
        **identity_checks,
        "candidate_only_bias_absent": candidate_only_bias_absent,
        "all_action_output_projection_weights_exact_zero_at_theta0": (
            theta0_projection_weights_exact_zero
        ),
        "zero_origin_phi_and_residual_exact_zero_theta0_and_final": (
            bool(theta0_zero_origin["phi_zero_bitwise_exact_zero"])
            and bool(
                theta0_zero_origin[
                    "all_block_residual_zero_bitwise_exact_zero"
                ]
            )
            and bool(final_zero_origin["phi_zero_bitwise_exact_zero"])
            and bool(
                final_zero_origin[
                    "all_block_residual_zero_bitwise_exact_zero"
                ]
            )
        ),
        "no_action_candidate_slot_exact_zero_and_update_free": (
            no_action_slot_valid
        ),
        "no_action_candidate_only_parameter_gradients_all_none_theta0_and_final": (
            bool(theta0_no_action_gradient["all_none"])
            and bool(final_no_action_gradient["all_none"])
        ),
        "no_action_raw_action_jvp_exact_zero_theta0_and_final": (
            float(theta0_no_action_jvp.rms.detach().item()) == 0.0
            and float(final_no_action_jvp.rms.detach().item()) == 0.0
        ),
        "actual_call_bootstrap_direct_identity_theta0_and_final": (
            all(theta0_correct_bypass["bootstrap_bitwise_equal_by_block"])
            and all(final_correct_bypass["bootstrap_bitwise_equal_by_block"])
            and max(theta0_correct_bypass["bootstrap_max_abs_by_block"])
            == 0.0
            and max(final_correct_bypass["bootstrap_max_abs_by_block"])
            == 0.0
        ),
        "padding_seam_helper_inactive_identity_theta0_and_final": (
            bool(theta0_padding_fixture["bitwise_equal"])
            and bool(final_padding_fixture["bitwise_equal"])
            and float(theta0_padding_fixture["max_abs"]) == 0.0
            and float(final_padding_fixture["max_abs"]) == 0.0
            and bool(theta0_padding_fixture["inactive_residual_exact_zero"])
            and bool(final_padding_fixture["inactive_residual_exact_zero"])
            and not bool(theta0_padding_fixture["vendor_used"])
            and not bool(final_padding_fixture["vendor_used"])
        ),
        "no_action_module_call_counts_exact_zero": all(
            output.action_conditioner_calls_this_forward == 0
            and output.action_projection_calls_this_forward == 0
            for output in (
                candidate_theta0["no_action"],
                candidate_theta0["seam_disabled"],
                candidate_no_action,
                candidate_seam_disabled,
            )
        ),
        "no_action_per_block_action_residuals_exact_zero": (
            bool(theta0_no_action_bypass["all_action_residuals_exact_zero"])
            and bool(final_no_action_bypass["all_action_residuals_exact_zero"])
        ),
        "correct_module_call_counts_exact": (
            candidate_theta0["correct"].action_conditioner_calls_this_forward
            == 1
            and candidate_theta0["correct"].action_projection_calls_this_forward
            == 20
            and candidate_train.final_output.action_conditioner_calls_this_forward
            == 1
            and candidate_train.final_output.action_projection_calls_this_forward
            == 20
        ),
        "synthetic_data_and_transfer_contract": data_contract_valid,
        "target_perturbation_does_not_change_prediction_or_hidden": (
            not target_prediction_changed and not target_hidden_changed
        ),
        "prediction_target_reverse_gradient_zero_or_unused": (
            not target_reverse_nonzero
        ),
        "prefix_invariance_abs": prefix_abs <= 1.0e-4,
        "prefix_invariance_rel": prefix_rel <= 1.0e-4,
        "exact_vendor_identity": exact_vendor_identity,
        "actual_vendor_forward_call_count_positive": actual_vendor_calls > 0,
        "vendor_common_parameter_gradient_live": (
            candidate_train.vendor_grad_rms_step0 > 1.0e-8
        ),
        "local_seam_parameter_jvp_live": (
            float(theta0_parameter_jvp.rms.detach().item()) > 1.0e-8
        ),
        "local_action_condition_jvp_live": (
            float(final_action_jvp.rms.detach().item()) > 1.0e-8
        ),
        "exact_optimizer_steps": (
            reference_train.optimizer_steps == 300
            and candidate_train.optimizer_steps == 300
        ),
        "cuda_and_jit_inventory_recorded": (
            jit_inventory.get("cuda_visible_devices")
            == expected_gpu_uuid
            and jit_inventory.get("cuda_compute_capability") == [9, 0]
            and jit_inventory.get("vendor_kernel_executed") is True
            and isinstance(jit_inventory.get("triton_cache_files"), list)
            and int(jit_inventory.get("triton_cache_file_count", 0)) > 0
        ),
        "finite_outputs_gradients_updates_optimizer_and_metrics": (
            outputs_finite
            and reference_train.post_vendor_hidden_gradient_finite_step0
            and candidate_train.post_vendor_hidden_gradient_finite_step0
            and post_vendor_update_finite
            and reference_train.optimizer_scalars_finite
            and candidate_train.optimizer_scalars_finite
            and reference_train.parameters_finite
            and candidate_train.parameters_finite
            and finite_metric_values
        ),
    }
    failed_validity = sorted(
        name for name, passed in card_validity_checks.items() if not passed
    )
    final_parameter_digests: Mapping[str, object] = {
        "reference": _named_state_manifest(
            pair.reference,
            parameter_names=frozenset(
                name for name, _ in pair.reference.named_parameters()
            ),
            buffer_names=frozenset(
                name for name, _ in pair.reference.named_buffers()
            ),
        ),
        "candidate": _named_state_manifest(
            pair.candidate,
            parameter_names=frozenset(
                name for name, _ in pair.candidate.named_parameters()
            ),
            buffer_names=frozenset(
                name for name, _ in pair.candidate.named_buffers()
            ),
        ),
    }
    diagnostics: Mapping[str, object] = {
        "operator_level": "VENDOR_KERNEL",
        "integration_path": "EXPERIMENTAL_PATH",
        "result_scope": "SYNTHETIC_SINGLE_BATCH_SINGLE_CALL_VENDOR_OPERATOR_ONLY",
        "device": "cuda:0",
        "dtype": "float32",
        "vendor_entry_class": "ChunkCausalGDNTriton",
        "vendor_use_autograd_kernel": True,
        "hw": [5, 1, 1],
        "chunk_boundaries": [0, 3, 5],
        "fused_gdn_precision": os.environ.get("FUSED_GDN_PRECISION"),
        "cross_call_state_or_commit_assessed": False,
        "zero_action_anchor_buffer_exact_zero_update_free": no_action_slot_valid,
        "candidate_only_bias_absent": candidate_only_bias_absent,
        "all_action_output_projection_weights_exact_zero_at_theta0": (
            theta0_projection_weights_exact_zero
        ),
        "zero_origin": {
            "theta0": theta0_zero_origin,
            "final": final_zero_origin,
        },
        "typed_action_presence_mask": {
            "correct_or_shuffle_row": [False, True, True, True, True],
            "no_action_or_seam_disabled_row": [False, False, False, False, False],
            "observed_exact": typed_mask_exact,
            "source": "LATENT_VALID_MASK_AND_NOT_LATENT_ACTION_SPAN_ANCHOR_NO_ACTION_SLOT",
        },
        "optimizer_steps_completed_by_arm": {
            "REF-GDN-CORRECTED": reference_train.optimizer_steps,
            "CACH-A2": candidate_train.optimizer_steps,
        },
        "actual_vendor_forward_call_count": actual_vendor_calls,
        "vendor_common_parameter_gradient_rms_step0": (
            candidate_train.vendor_grad_rms_step0
        ),
        "last_block_local_seam_parameter_jvp_rms_theta0": float(
            theta0_parameter_jvp.rms.detach().item()
        ),
        "last_block_local_action_condition_jvp_rms_step300": float(
            final_action_jvp.rms.detach().item()
        ),
        "single_call_first_chunk_prefix_invariance_max_abs": prefix_abs,
        "single_call_first_chunk_prefix_invariance_max_rel": prefix_rel,
        "zero_anchor_bypass": {
            "theta0_no_action_vs_seam_disabled_bitwise_equal": (
                theta0_no_action_vs_seam_disabled
            ),
            "theta0_no_action_vs_seam_disabled_max_abs": (
                theta0_no_action_vs_seam_disabled_max_abs
            ),
            "final_no_action_vs_seam_disabled_bitwise_equal": (
                final_no_action_vs_seam_disabled
            ),
            "final_no_action_vs_seam_disabled_max_abs": (
                final_no_action_vs_seam_disabled_max_abs
            ),
            "theta0_candidate_only_gradients": dict(
                theta0_no_action_gradient
            ),
            "final_candidate_only_gradients": dict(
                final_no_action_gradient
            ),
            "theta0_correct_blocks": theta0_correct_bypass,
            "theta0_no_action_blocks": theta0_no_action_bypass,
            "final_correct_blocks": final_correct_bypass,
            "final_no_action_blocks": final_no_action_bypass,
            "theta0_padding_helper": theta0_padding_fixture,
            "final_padding_helper": final_padding_fixture,
            "padding_helper_vendor_used": False,
            "no_action_conditioner_projection_calls": {
                "theta0_no_action": [
                    candidate_theta0[
                        "no_action"
                    ].action_conditioner_calls_this_forward,
                    candidate_theta0[
                        "no_action"
                    ].action_projection_calls_this_forward,
                ],
                "theta0_seam_disabled": [
                    candidate_theta0[
                        "seam_disabled"
                    ].action_conditioner_calls_this_forward,
                    candidate_theta0[
                        "seam_disabled"
                    ].action_projection_calls_this_forward,
                ],
                "final_no_action": [
                    candidate_no_action.action_conditioner_calls_this_forward,
                    candidate_no_action.action_projection_calls_this_forward,
                ],
                "final_seam_disabled": [
                    candidate_seam_disabled.action_conditioner_calls_this_forward,
                    candidate_seam_disabled.action_projection_calls_this_forward,
                ],
            },
        },
        "target_leakage": {
            "target_perturbation_changes_prediction_or_post_vendor_hidden": (
                target_prediction_changed or target_hidden_changed
            ),
            "prediction_target_reverse_gradient_nonzero": (
                target_reverse_nonzero
            ),
        },
        "theta0_identity": dict(identity_checks),
        "post_vendor_observability": {
            "finite_output": outputs_finite,
            "finite_post_vendor_hidden_gradient": (
                reference_train.post_vendor_hidden_gradient_finite_step0
                and candidate_train.post_vendor_hidden_gradient_finite_step0
            ),
            "finite_post_vendor_hidden_update": post_vendor_update_finite,
            "last_block_theta0_post_vendor_hidden_sha256": tensor_digest(
                candidate_theta0_post_vendor
            ),
            "last_block_step300_post_vendor_hidden_sha256": tensor_digest(
                candidate_train.final_output.last_block_post_vendor_hidden.detach()
            ),
        },
        "task_tensor_manifest_before_model": task_manifest_before,
        "task_tensor_manifest_after_screen": task_manifest_after,
        "synthetic_data_contract": dict(data_checks),
        "vendor_identity_by_arm": {
            "reference": vendor_identity_manifest(pair.reference),
            "candidate": vendor_identity_manifest(pair.candidate),
        },
        "local_jvp": {
            "theta0_parameter_scope": theta0_parameter_jvp.scope,
            "theta0_parameter_primal_digest": (
                theta0_parameter_jvp.primal_digest
            ),
            "step300_action_condition_scope": final_action_jvp.scope,
            "step300_action_condition_primal_digest": (
                final_action_jvp.primal_digest
            ),
            "vendor_primal_inside_forward_mode_transform": False,
            "theta0_correct_raw_action_rms": float(
                theta0_correct_action_jvp.rms.detach().item()
            ),
            "theta0_no_action_raw_action_rms": float(
                theta0_no_action_jvp.rms.detach().item()
            ),
            "step300_correct_raw_action_rms": float(
                final_action_jvp.rms.detach().item()
            ),
            "step300_no_action_raw_action_rms": float(
                final_no_action_jvp.rms.detach().item()
            ),
        },
    }
    resource_usage: Mapping[str, object] = {
        "elapsed_seconds_screen": time.monotonic() - started,
        "process_peak_rss_bytes_screen": _screen_peak_rss_bytes(),
        "cuda_peak_memory_allocated_bytes_screen": int(
            torch.cuda.max_memory_allocated(0)
        ),
        "cuda_peak_memory_reserved_bytes_screen": int(
            torch.cuda.max_memory_reserved(0)
        ),
        "checkpoint_loaded": False,
        "checkpoint_saved": False,
        "real_data_used": False,
        "cpu_proxy_or_reference_fallback_used": False,
        "new_review_token_created_derived_reset_or_consumed": False,
        "predecessor_consumed_claim_mutated": False,
    }
    return {
        "schema": "cach.cach_a2.zero_anchor_vendor_gdn_bridge.screen.v1",
        "validity": {
            "implementation_valid": (
                exact_vendor_identity
                and candidate_only_bias_absent
                and no_action_slot_valid
                and theta0_projection_weights_exact_zero
                and bool(theta0_zero_origin["phi_zero_bitwise_exact_zero"])
                and bool(final_zero_origin["phi_zero_bitwise_exact_zero"])
            ),
            "data_valid": (
                data_contract_valid and reducer_exact and padding_zero
            ),
            "numerics_valid": finite_metric_values and outputs_finite,
            "finite_all": finite_metric_values and outputs_finite,
            "all_card_validity_checks_pass": not failed_validity,
            "reasons": failed_validity,
            "checks": dict(card_validity_checks),
        },
        "metrics": metrics,
        "loss_trace": {
            "reference": list(reference_train.loss_trace),
            "candidate": list(candidate_train.loss_trace),
        },
        "theta0_manifests": theta0_manifests,
        "final_parameter_digests": final_parameter_digests,
        "diagnostics": diagnostics,
        "jit_inventory": jit_inventory,
        "resource_usage": resource_usage,
    }


def vendor_identity_manifest(
    arm: AV1BVendorGDNBridgeArm,
) -> Mapping[str, object]:
    """Return model-local identity facts; the runner adds CUDA UUID facts."""

    if not isinstance(arm, AV1BVendorGDNBridgeArm):
        raise TypeError("arm must be an AV1BVendorGDNBridgeArm")
    for block in arm.blocks:
        block._assert_vendor_identity()
    return {
        "exact_class": (
            "diffusion.model.nets.sana_gdn_blocks_triton."
            "ChunkCausalGDNTriton"
        ),
        "use_autograd_kernel": True,
        "HW": list(AV1B_HW),
        "chunk_boundaries": list(AV1B_CHUNK_BOUNDARIES),
        "chunk_size": None,
        "chunk_split_strategy": "uniform",
        "depth": arm.spec.depth,
        "heads": arm.spec.heads,
        "head_dim": arm.spec.head_dim,
        "conv_kernel_size": arm.spec.conv_kernel_size,
        "k_conv_only": arm.spec.k_conv_only,
        "qk_norm": arm.spec.qk_norm,
        "use_output_gate": arm.spec.use_output_gate,
        "vendor_forward_call_count": arm.vendor_forward_call_count,
        "operator_lifecycle": "EPHEMERAL_SINGLE_FORWARD_CALL",
        "cross_call_state_io": False,
    }


__all__ = [
    "CACH_A2_CARD_SHA256",
    "AV1B_CHUNK_BOUNDARIES",
    "AV1B_HW",
    "AV1B_TASK_RECIPE_SHA256",
    "AV1BLocalJVPDiagnostic",
    "AV1BVendorGDNBridgeArm",
    "AV1BVendorGDNBridgePair",
    "AV1BVendorGDNBridgeSpec",
    "AV1BVendorSequenceOutput",
    "AV1BVendorTask",
    "AV1BVendorTaskChunk",
    "ActionMode",
    "apply_zero_anchor_action_residual",
    "build_av1b_pair",
    "build_av1b_synthetic_task",
    "build_av1b_vendor_gdn_pair",
    "build_av1b_vendor_task",
    "condition_zero_anchor_actions",
    "last_block_local_action_condition_jvp",
    "last_block_local_seam_parameter_jvp",
    "pure_torch_last_block_tail",
    "run_cach_a2_screen",
    "vendor_identity_manifest",
]
