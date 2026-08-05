"""CPU/FP32 AV-1 transition-proxy screen for REF-GDN-CORRECTED and CACH-A.

The implementation in this module is intentionally an experimental,
synthetic transition proxy.  It is not a production dispatcher, checkpoint
loader, full-size model, evaluator, or admission-root writer.  The public
screen function performs no filesystem access and cannot select CUDA.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import math
import resource
import time
from typing import Any, Final, Literal

import torch
from torch import Tensor, nn

from sana_wam.cach.action_conditioning import (
    reduce_end_of_bin_action_condition,
)
from sana_wam.cach.prefix_compaction import FixedKPrefixPlan
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_backbone.joint_action_dit import ActionDiT
from sana_wam.model.action_chunk_layout import (
    ChunkActionLayout,
    ChunkLayout,
    build_synthetic_chunk_action_layout_for_tests,
    canonical_sha256,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)
from sana_wam.model.cach_minimal_cpu import (
    MinimalLayerState,
    PureTorchChunkCausalGDNProxy,
)
from sana_wam.model.cach_numerical_core import CACHNumericalCore
from sana_wam.model.video_backbone.sana.hybrid_cache import tensor_digest


AV0_CARD_SHA256: Final = (
    "c73a943047295338cd9de14e6e53199b463ea66ae92b99d4b0b3393f5ec34030"
)
AV1_TASK_RECIPE_SHA256: Final = (
    "9d2a749d722775324ee233862772def530fbffe0ea9e55aeea87c160b87a51e4"
)
AV1_TASK_RECIPE_MATERIAL: Final = (
    "cach.av1.synthetic_counterfactual.dct_teacher.raw33.k3.tc8.vs1.b8.h4.v1"
)
_INITIALIZER_REVISION: Final = "cach-av1-transition-proxy-v1"
_SHARED_INITIALIZER_ROLE: Final = "shared_named_uniform_v1"
_CANDIDATE_INITIALIZER_ROLE: Final = (
    "candidate_operator_named_uniform_v1"
)
_SHARED_SEED: Final = 2026080301
_CANDIDATE_SEED: Final = 2026080302
_SHUFFLE_PERMUTATION: Final = (1, 0, 3, 2, 5, 4, 7, 6)
_ROW_TO_PAIR: Final = (0, 0, 1, 1, 2, 2, 3, 3)
_ROW_SIGN: Final = (1, -1, 1, -1, 1, -1, 1, -1)
_VIDEO_TIMESTEPS: Final = (701.0, 718.0)
_ACTION_TIMESTEPS: Final = (503.0, 522.0)
_END_OF_BIN_INDICES: Final = (7, 15, 23, 31)
_EPSILON: Final = 1.0e-12

ActionMode = Literal["correct", "shuffle", "no_action"]


@dataclass(frozen=True)
class AV1TransitionProxySpec:
    """The frozen reduced topology from the AV0 run card."""

    batch_size: int = 8
    counterfactual_pair_count: int = 4
    latent_channels: int = 3
    spatial_height: int = 1
    spatial_width: int = 1
    hidden_dim: int = 4
    context_dim: int = 4
    context_token_count: int = 2
    action_dim: int = 20
    gdn_depth: int = 20
    gdn_heads: int = 1
    frame_chunk_size: int = 3
    temporal_compression: int = 8
    video_stride: int = 1
    valid_raw_count: int = 33
    valid_latent_count: int = 5
    valid_action_count: int = 32
    chunk_count: int = 2
    scored_future_horizons: tuple[int, ...] = (1, 2, 3, 4)
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        expected = {
            "batch_size": 8,
            "counterfactual_pair_count": 4,
            "latent_channels": 3,
            "spatial_height": 1,
            "spatial_width": 1,
            "hidden_dim": 4,
            "context_dim": 4,
            "context_token_count": 2,
            "action_dim": 20,
            "gdn_depth": 20,
            "gdn_heads": 1,
            "frame_chunk_size": 3,
            "temporal_compression": 8,
            "video_stride": 1,
            "valid_raw_count": 33,
            "valid_latent_count": 5,
            "valid_action_count": 32,
            "chunk_count": 2,
        }
        for name, required in expected.items():
            value = getattr(self, name)
            if type(value) is not int or value != required:
                raise ValueError(
                    f"{name} is frozen to {required} by the AV0 card"
                )
        if self.scored_future_horizons != (1, 2, 3, 4):
            raise ValueError("scored_future_horizons is frozen to (1, 2, 3, 4)")
        if self.device != "cpu":
            raise ValueError("AV1 transition proxy is CPU-only")
        if self.dtype is not torch.float32:
            raise ValueError("AV1 transition proxy is float32-only")


@dataclass(frozen=True)
class AV1TaskChunk:
    """One fixed-capacity synthetic chunk bound to the typed layout."""

    chunk_id: int
    layout: ChunkLayout
    video_latents: Tensor = field(repr=False, compare=False)
    actions: Tensor = field(repr=False, compare=False)
    frame_valid_mask: Tensor = field(repr=False, compare=False)
    action_valid_mask: Tensor = field(repr=False, compare=False)


@dataclass(frozen=True)
class AV1SyntheticTask:
    """Deterministic counterfactual task; target bytes are loss-only."""

    spec: AV1TransitionProxySpec
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
    chunks: tuple[AV1TaskChunk, ...]
    score_mask: Tensor = field(repr=False, compare=False)
    row_to_pair: tuple[int, ...]
    row_sign: tuple[int, ...]
    shuffle_permutation: tuple[int, ...]
    recipe_digest: str
    tensor_digests: Mapping[str, str]


@dataclass(frozen=True)
class AV1TransitionState:
    """Read-only differentiable recurrent state for one arm."""

    arm: str
    layout_instance_digest: str
    batch_size: int
    next_chunk_id: int
    action_cursor: int
    layers: tuple[MinimalLayerState, ...] = field(repr=False, compare=False)

    def detached_clone(self) -> "AV1TransitionState":
        return type(self)(
            arm=self.arm,
            layout_instance_digest=self.layout_instance_digest,
            batch_size=self.batch_size,
            next_chunk_id=self.next_chunk_id,
            action_cursor=self.action_cursor,
            layers=tuple(layer.detached_clone() for layer in self.layers),
        )


@dataclass(frozen=True)
class AV1SequenceOutput:
    """Video prediction, frozen ActionDiT diagnostics, and final state."""

    video_prediction: Tensor = field(repr=False, compare=False)
    action_predictions: tuple[Tensor, ...] = field(
        repr=False,
        compare=False,
    )
    final_state: AV1TransitionState = field(repr=False, compare=False)

    @property
    def action_prediction(self) -> Tensor | None:
        if not self.action_predictions:
            return None
        return torch.cat(self.action_predictions, dim=1)


@dataclass(frozen=True)
class AV1TransitionProxyPair:
    """Fresh, independently owned REF and CACH-A arms."""

    spec: AV1TransitionProxySpec
    layout: ChunkActionLayout
    reference: "AV1TransitionProxyArm"
    candidate: "AV1TransitionProxyArm"
    common_trainable_names: frozenset[str]
    candidate_only_trainable_names: frozenset[str]
    reference_trainable_names: frozenset[str]
    candidate_trainable_names: frozenset[str]
    theta0_manifests: Mapping[str, Mapping[str, object]]

    @property
    def reference_trainable_name_set(self) -> frozenset[str]:
        return self.reference_trainable_names

    @property
    def candidate_trainable_name_set(self) -> frozenset[str]:
        return self.candidate_trainable_names

    @property
    def reference_theta0_manifest(self) -> Mapping[str, object]:
        return self.theta0_manifests["reference"]

    @property
    def candidate_theta0_manifest(self) -> Mapping[str, object]:
        return self.theta0_manifests["candidate"]


def _build_av1_layout(spec: AV1TransitionProxySpec) -> ChunkActionLayout:
    layout_spec = synthetic_layout_spec(
        frame_chunk_size=spec.frame_chunk_size,
        temporal_compression=spec.temporal_compression,
        video_stride=spec.video_stride,
        action_dim=spec.action_dim,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=spec.valid_raw_count,
        video_stride=spec.video_stride,
        source_row_label="synthetic_counterfactual_pairs_0000_0003",
        episode_label="cach_av1_transition_proxy_seed0",
    )
    layout = build_synthetic_chunk_action_layout_for_tests(
        spec=layout_spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=spec.valid_raw_count,
        video_valid_mask=(True,) * spec.valid_raw_count,
        action_valid_mask=(True,) * spec.valid_action_count,
    )
    capacities = tuple(chunk.action_slot_capacity for chunk in layout.chunks)
    valid_actions = tuple(chunk.valid_action_count for chunk in layout.chunks)
    valid_latents = tuple(chunk.valid_latent_count for chunk in layout.chunks)
    if capacities != (16, 24):
        raise RuntimeError(f"unexpected action capacities: {capacities}")
    if valid_actions != (16, 16):
        raise RuntimeError(f"unexpected valid action counts: {valid_actions}")
    if valid_latents != (3, 2):
        raise RuntimeError(f"unexpected valid latent counts: {valid_latents}")
    return layout


def _tensor_from_binary64_values(
    values: list[float],
    shape: tuple[int, ...],
) -> Tensor:
    return torch.tensor(values, dtype=torch.float32, device="cpu").reshape(
        shape
    ).contiguous()


def build_av1_synthetic_task(
    spec: AV1TransitionProxySpec | None = None,
) -> AV1SyntheticTask:
    """Build the card-pinned binary64 analytic task with no RNG."""

    spec = AV1TransitionProxySpec() if spec is None else spec
    if not isinstance(spec, AV1TransitionProxySpec):
        raise TypeError("spec must be an AV1TransitionProxySpec")
    layout = _build_av1_layout(spec)

    action_values: list[float] = []
    for row in range(spec.batch_size):
        pair = _ROW_TO_PAIR[row]
        sign = _ROW_SIGN[row]
        for action_index in range(spec.valid_action_count):
            for action_channel in range(spec.action_dim):
                value = sign * (
                    0.75
                    * math.sin(
                        (pair + 1)
                        * (action_index + 1)
                        * (action_channel + 1)
                        / 17.0
                    )
                    + 0.25
                    * math.cos(
                        (pair + 2)
                        * (action_index + 1)
                        / (action_channel + 2)
                    )
                )
                action_values.append(value)
    global_actions = _tensor_from_binary64_values(
        action_values,
        (spec.batch_size, spec.valid_action_count, spec.action_dim),
    )

    noisy_values: list[float] = []
    base_values: list[float] = []
    for row in range(spec.batch_size):
        pair = _ROW_TO_PAIR[row]
        for latent_index in range(spec.valid_latent_count):
            for channel in range(spec.latent_channels):
                noisy = 0.10 * math.sin(
                    (pair + 1)
                    * (latent_index + 1)
                    * (channel + 1)
                    / 7.0
                ) + 0.03 * math.cos(
                    (pair + 2) * (latent_index + 1) / (channel + 1)
                )
                base = 0.60 * noisy + 0.05 * math.cos(
                    (pair + 1)
                    * (latent_index + 1)
                    * (channel + 1)
                    / 11.0
                )
                noisy_values.append(noisy)
                base_values.append(base)
    video_shape = (
        spec.batch_size,
        spec.valid_latent_count,
        spec.latent_channels,
        1,
        1,
    )
    noisy_video = _tensor_from_binary64_values(noisy_values, video_shape)
    base_target = _tensor_from_binary64_values(base_values, video_shape)

    context_values: list[float] = []
    for row in range(spec.batch_size):
        pair = _ROW_TO_PAIR[row]
        for chunk_id in range(spec.chunk_count):
            for token in range(spec.context_token_count):
                for dimension in range(spec.context_dim):
                    context_values.append(
                        0.10
                        * math.cos(
                            (pair + 1)
                            * (chunk_id + 1)
                            * (token + 1)
                            * (dimension + 1)
                            / 13.0
                        )
                    )
    context = _tensor_from_binary64_values(
        context_values,
        (
            spec.batch_size,
            spec.chunk_count,
            spec.context_token_count,
            spec.context_dim,
        ),
    )
    context_mask = torch.ones(
        (
            spec.batch_size,
            spec.chunk_count,
            spec.context_token_count,
        ),
        dtype=torch.bool,
        device="cpu",
    )

    proprio_values: list[float] = []
    for row in range(spec.batch_size):
        pair = _ROW_TO_PAIR[row]
        for chunk_id in range(spec.chunk_count):
            for action_channel in range(spec.action_dim):
                proprio_values.append(
                    0.10
                    * math.cos(
                        (pair + 1)
                        * (chunk_id + 1)
                        * (action_channel + 1)
                        / 19.0
                    )
                )
    proprio = _tensor_from_binary64_values(
        proprio_values,
        (spec.batch_size, spec.chunk_count, spec.action_dim),
    )

    teacher_matrix: list[list[float]] = []
    for action_channel in range(spec.action_dim):
        row_values: list[float] = []
        for latent_channel in range(spec.latent_channels):
            row_values.append(
                math.sqrt(2.0 / 20.0)
                * math.cos(
                    math.pi
                    * (action_channel + 0.5)
                    * (latent_channel + 1)
                    / 20.0
                )
            )
        teacher_matrix.append(row_values)

    raw_effect = [
        [
            [0.0 for _ in range(spec.latent_channels)]
            for _ in range(spec.valid_latent_count)
        ]
        for _ in range(spec.batch_size)
    ]
    for horizon, end_index in enumerate(_END_OF_BIN_INDICES, start=1):
        for row in range(spec.batch_size):
            for channel in range(spec.latent_channels):
                action_sum = 0.0
                for action_channel in range(spec.action_dim):
                    action_value = action_values[
                        (
                            row * spec.valid_action_count * spec.action_dim
                            + end_index * spec.action_dim
                            + action_channel
                        )
                    ]
                    action_sum += (
                        action_value
                        * teacher_matrix[action_channel][channel]
                    )
                raw_effect[row][horizon][channel] = (
                    0.5 * raw_effect[row][horizon - 1][channel]
                    + action_sum
                )

    normalized_effect = [
        [
            [0.0 for _ in range(spec.latent_channels)]
            for _ in range(spec.valid_latent_count)
        ]
        for _ in range(spec.batch_size)
    ]
    for horizon in spec.scored_future_horizons:
        square_sum = 0.0
        for row in range(spec.batch_size):
            for channel in range(spec.latent_channels):
                value = raw_effect[row][horizon][channel]
                square_sum += value * value
        rms = math.sqrt(
            square_sum / (spec.batch_size * spec.latent_channels)
        )
        if not math.isfinite(rms) or rms <= 0.0:
            raise RuntimeError("teacher effect normalization is invalid")
        for row in range(spec.batch_size):
            for channel in range(spec.latent_channels):
                normalized_effect[row][horizon][channel] = (
                    0.5 * raw_effect[row][horizon][channel] / rms
                )

    target_values: list[float] = []
    for row in range(spec.batch_size):
        for latent_index in range(spec.valid_latent_count):
            for channel in range(spec.latent_channels):
                flat_index = (
                    row * spec.valid_latent_count * spec.latent_channels
                    + latent_index * spec.latent_channels
                    + channel
                )
                target_values.append(
                    base_values[flat_index]
                    + normalized_effect[row][latent_index][channel]
                )
    video_target = _tensor_from_binary64_values(target_values, video_shape)

    video_timestep = (
        torch.tensor(_VIDEO_TIMESTEPS, dtype=torch.float32)
        .view(1, spec.chunk_count)
        .expand(spec.batch_size, spec.chunk_count)
        .clone()
        .contiguous()
    )
    action_timestep = (
        torch.tensor(_ACTION_TIMESTEPS, dtype=torch.float32)
        .view(1, spec.chunk_count)
        .expand(spec.batch_size, spec.chunk_count)
        .clone()
        .contiguous()
    )

    task_chunks: list[AV1TaskChunk] = []
    for chunk in layout.chunks:
        valid_video = noisy_video[
            :,
            chunk.latent_start : chunk.latent_end,
            :,
            :,
            :,
        ].clone()
        video_padding_count = (
            spec.frame_chunk_size - chunk.valid_latent_count
        )
        if video_padding_count:
            video_latents = torch.cat(
                (
                    valid_video,
                    torch.zeros(
                        (
                            spec.batch_size,
                            video_padding_count,
                            spec.latent_channels,
                            1,
                            1,
                        ),
                        dtype=torch.float32,
                        device="cpu",
                    ),
                ),
                dim=1,
            )
        else:
            video_latents = valid_video
        valid_actions = global_actions[
            :,
            chunk.action_start : chunk.action_end,
            :,
        ].clone()
        padding_count = chunk.action_slot_capacity - chunk.valid_action_count
        if padding_count:
            padded = torch.zeros(
                (spec.batch_size, padding_count, spec.action_dim),
                dtype=torch.float32,
                device="cpu",
            )
            chunk_actions = torch.cat((valid_actions, padded), dim=1)
        else:
            chunk_actions = valid_actions
        frame_valid_mask = (
            torch.tensor(chunk.latent_valid_mask, dtype=torch.bool)
            .view(1, spec.frame_chunk_size)
            .expand(spec.batch_size, spec.frame_chunk_size)
            .clone()
        )
        action_valid_mask = (
            torch.tensor(chunk.action_valid_mask, dtype=torch.bool)
            .view(1, chunk.action_slot_capacity)
            .expand(spec.batch_size, chunk.action_slot_capacity)
            .clone()
        )
        task_chunks.append(
            AV1TaskChunk(
                chunk_id=chunk.chunk_id,
                layout=chunk,
                video_latents=video_latents.contiguous(),
                actions=chunk_actions.contiguous(),
                frame_valid_mask=frame_valid_mask.contiguous(),
                action_valid_mask=action_valid_mask.contiguous(),
            )
        )

    score_mask = torch.ones(
        (spec.batch_size, len(spec.scored_future_horizons)),
        dtype=torch.bool,
        device="cpu",
    )
    digests = {
        "global_actions": tensor_digest(global_actions),
        "noisy_video": tensor_digest(noisy_video),
        "base_target": tensor_digest(base_target),
        "video_target": tensor_digest(video_target),
        "context": tensor_digest(context),
        "context_mask": tensor_digest(context_mask),
        "proprio": tensor_digest(proprio),
        "video_timestep": tensor_digest(video_timestep),
        "action_timestep": tensor_digest(action_timestep),
        "score_mask": tensor_digest(score_mask),
    }
    for task_chunk in task_chunks:
        digests[f"chunk{task_chunk.chunk_id}_video_latents"] = tensor_digest(
            task_chunk.video_latents
        )
        digests[f"chunk{task_chunk.chunk_id}_actions"] = tensor_digest(
            task_chunk.actions
        )
        digests[f"chunk{task_chunk.chunk_id}_frame_valid_mask"] = (
            tensor_digest(task_chunk.frame_valid_mask)
        )
        digests[f"chunk{task_chunk.chunk_id}_action_valid_mask"] = (
            tensor_digest(task_chunk.action_valid_mask)
        )

    return AV1SyntheticTask(
        spec=spec,
        layout=layout,
        global_actions=global_actions,
        noisy_video=noisy_video,
        base_target=base_target,
        video_target=video_target,
        context=context,
        context_mask=context_mask,
        proprio=proprio,
        video_timestep=video_timestep,
        action_timestep=action_timestep,
        chunks=tuple(task_chunks),
        score_mask=score_mask,
        row_to_pair=_ROW_TO_PAIR,
        row_sign=_ROW_SIGN,
        shuffle_permutation=_SHUFFLE_PERMUTATION,
        recipe_digest=AV1_TASK_RECIPE_SHA256,
        tensor_digests=digests,
    )


def _common_trainable_names(depth: int) -> frozenset[str]:
    names = {
        "video_input_projection.weight",
        "video_input_projection.bias",
        "video_timestep_projection.weight",
        "video_output_projection.weight",
        "video_output_projection.bias",
        "proprio_context_encoder.weight",
        "proprio_context_encoder.bias",
    }
    suffixes = (
        "q_projection.weight",
        "k_projection.weight",
        "v_projection.weight",
        "beta_projection.weight",
        "beta_projection.bias",
        "decay_projection.weight",
        "decay_projection.bias",
        "attention_output_projection.weight",
        "attention_output_projection.bias",
        "ffn_input_projection.weight",
        "ffn_input_projection.bias",
        "ffn_output_projection.weight",
        "ffn_output_projection.bias",
    )
    for block_id in range(depth):
        for suffix in suffixes:
            names.add(f"blocks.{block_id}.{suffix}")
    return frozenset(names)


def _candidate_only_trainable_names(depth: int) -> frozenset[str]:
    names = {
        "action_conditioner.0.weight",
        "action_conditioner.0.bias",
        "action_conditioner.2.weight",
        "action_conditioner.2.bias",
    }
    for block_id in range(depth):
        names.add(f"blocks.{block_id}.action_output_projection.weight")
        names.add(f"blocks.{block_id}.action_output_projection.bias")
    return frozenset(names)


def _is_candidate_operator_parameter(name: str) -> bool:
    return name.startswith("action_conditioner.") or (
        name.startswith("blocks.")
        and ".action_output_projection." in name
    )


def _derived_seed(*, base_seed: int, role: str, name: str) -> int:
    material = (
        f"{_INITIALIZER_REVISION}:{base_seed}:{role}:{name}".encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _initialize_arm(arm: "AV1TransitionProxyArm") -> None:
    with torch.no_grad():
        for name, parameter in sorted(
            arm.named_parameters(),
            key=lambda item: item[0].encode("utf-8"),
        ):
            if name == "cach_core.no_action_slot":
                parameter.zero_()
                continue
            if (
                arm.staging_variant is CACHStagingVariant.CACH_A
                and name.startswith("blocks.")
                and ".action_output_projection." in name
            ):
                parameter.zero_()
                continue
            if name.endswith(".bias"):
                parameter.zero_()
                continue
            candidate_only = _is_candidate_operator_parameter(name)
            role = (
                _CANDIDATE_INITIALIZER_ROLE
                if candidate_only
                else _SHARED_INITIALIZER_ROLE
            )
            base_seed = _CANDIDATE_SEED if candidate_only else _SHARED_SEED
            fan_in = int(parameter.shape[-1]) if parameter.ndim >= 2 else 1
            bound = 0.25 / math.sqrt(float(fan_in))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _derived_seed(base_seed=base_seed, role=role, name=name)
            )
            parameter.uniform_(-bound, bound, generator=generator)

    arm.cach_core.no_action_slot.requires_grad_(False)
    for parameter in arm.action_backbone.parameters():
        parameter.requires_grad_(False)
    arm.action_backbone.eval()


class AV1TransitionProxyArm(nn.Module):
    """One trainable transition-proxy arm with a frozen real ActionDiT."""

    def __init__(
        self,
        *,
        spec: AV1TransitionProxySpec,
        layout: ChunkActionLayout,
        staging_variant: CACHStagingVariant,
    ) -> None:
        super().__init__()
        if not isinstance(spec, AV1TransitionProxySpec):
            raise TypeError("spec must be an AV1TransitionProxySpec")
        if not isinstance(layout, ChunkActionLayout):
            raise TypeError("layout must be a ChunkActionLayout")
        if staging_variant not in (
            CACHStagingVariant.REF_GDN_CORRECTED,
            CACHStagingVariant.CACH_A,
        ):
            raise TypeError("staging_variant must be a supported typed variant")
        if not layout.synthetic_test_only:
            raise ValueError("AV1 transition proxy accepts synthetic layouts only")

        self.spec = spec
        self.layout = layout
        self.staging_variant = staging_variant
        self.cach_core = CACHNumericalCore(
            action_dim=spec.action_dim,
            dtype=torch.float32,
            device="cpu",
        )
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
        self.blocks = nn.ModuleList(
            [
                PureTorchChunkCausalGDNProxy(
                    hidden_dim=spec.hidden_dim,
                    action_dim=spec.action_dim,
                    staging_variant=staging_variant,
                )
                for _ in range(spec.gdn_depth)
            ]
        )
        self.video_output_projection = nn.Linear(
            spec.hidden_dim,
            spec.latent_channels,
            bias=True,
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
        if staging_variant is CACHStagingVariant.CACH_A:
            self.action_conditioner: nn.Sequential | None = nn.Sequential(
                nn.Linear(
                    spec.action_dim,
                    spec.hidden_dim,
                    bias=True,
                    device="cpu",
                    dtype=torch.float32,
                ),
                nn.SiLU(),
                nn.Linear(
                    spec.hidden_dim,
                    spec.action_dim,
                    bias=True,
                    device="cpu",
                    dtype=torch.float32,
                ),
            )
        else:
            self.action_conditioner = None
        self.action_backbone = ActionDiT(
            action_dim=spec.action_dim,
            dim=spec.hidden_dim,
            ffn_dim=spec.hidden_dim * 2,
            num_heads=1,
            num_layers=spec.gdn_depth,
            video_dim=spec.hidden_dim,
            bridge_layers=tuple(range(spec.gdn_depth)),
            variant="joint_cross_attn",
            attn_head_dim=spec.hidden_dim,
            text_dim=spec.context_dim,
            freq_dim=spec.context_dim,
            max_action_len=128,
            attn_kernel="softmax",
        )
        _initialize_arm(self)
        self._assert_parameter_contract()

    @property
    def arm_name(self) -> str:
        if self.staging_variant is CACHStagingVariant.CACH_A:
            return "CACH-A"
        return "REF-GDN-CORRECTED"

    @property
    def trainable_name_set(self) -> frozenset[str]:
        return frozenset(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    def _assert_parameter_contract(self) -> None:
        common = _common_trainable_names(self.spec.gdn_depth)
        candidate = _candidate_only_trainable_names(self.spec.gdn_depth)
        expected = (
            common | candidate
            if self.staging_variant is CACHStagingVariant.CACH_A
            else common
        )
        actual = self.trainable_name_set
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                f"trainable allowlist mismatch; missing={missing}, extra={extra}"
            )
        if self.cach_core.no_action_slot.requires_grad:
            raise RuntimeError("NO_ACTION must be frozen")
        if bool(self.cach_core.no_action_slot.detach().count_nonzero()):
            raise RuntimeError("NO_ACTION must be exact zero")
        if any(
            parameter.requires_grad
            for parameter in self.action_backbone.parameters()
        ):
            raise RuntimeError("ActionDiT diagnostic backbone must be frozen")

    def train(self, mode: bool = True) -> "AV1TransitionProxyArm":
        super().train(mode)
        self.action_backbone.eval()
        return self

    def empty_state(self, batch_size: int | None = None) -> AV1TransitionState:
        batch = self.spec.batch_size if batch_size is None else batch_size
        if type(batch) is not int or batch != self.spec.batch_size:
            raise ValueError(
                f"batch_size is frozen to {self.spec.batch_size}"
            )
        layers = tuple(
            MinimalLayerState(
                main_s_kv=torch.zeros(
                    (batch, self.spec.hidden_dim, self.spec.hidden_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
                main_s_z=torch.zeros(
                    (batch, self.spec.hidden_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
            )
            for _ in range(self.spec.gdn_depth)
        )
        return AV1TransitionState(
            arm=self.arm_name,
            layout_instance_digest=self.layout.layout_instance_digest,
            batch_size=batch,
            next_chunk_id=0,
            action_cursor=0,
            layers=layers,
        )

    def _validate_task(self, task: AV1SyntheticTask) -> None:
        if not isinstance(task, AV1SyntheticTask):
            raise TypeError("task must be an AV1SyntheticTask")
        if task.spec != self.spec:
            raise ValueError("task spec differs from arm spec")
        if (
            task.layout.layout_instance_digest
            != self.layout.layout_instance_digest
        ):
            raise ValueError("task layout differs from arm layout")
        required_float32 = (
            task.global_actions,
            task.noisy_video,
            task.context,
            task.proprio,
            task.video_timestep,
            task.action_timestep,
        )
        if any(
            value.device.type != "cpu" or value.dtype != torch.float32
            for value in required_float32
        ):
            raise TypeError("AV1 forward requires CPU float32 task tensors")

    def _validate_initial_state(
        self,
        state: AV1TransitionState,
    ) -> None:
        if not isinstance(state, AV1TransitionState):
            raise TypeError("initial_state must be an AV1TransitionState")
        if state.arm != self.arm_name:
            raise ValueError("initial_state belongs to a different arm")
        if (
            state.layout_instance_digest
            != self.layout.layout_instance_digest
        ):
            raise ValueError("initial_state layout identity differs")
        if (
            state.batch_size != self.spec.batch_size
            or state.next_chunk_id != 0
            or state.action_cursor != 0
            or len(state.layers) != self.spec.gdn_depth
        ):
            raise ValueError("full-sequence forward requires a fresh empty state")

    def _global_actions_for_mode(
        self,
        task: AV1SyntheticTask,
        mode: ActionMode,
        *,
        diagnostic_reference: bool = False,
    ) -> Tensor:
        if mode == "correct":
            return task.global_actions
        if mode == "shuffle":
            permutation = torch.tensor(
                task.shuffle_permutation,
                dtype=torch.long,
                device="cpu",
            )
            return task.global_actions.index_select(0, permutation)
        if mode == "no_action":
            if (
                self.staging_variant is CACHStagingVariant.CACH_A
                and not diagnostic_reference
            ):
                return (
                    self.cach_core.no_action_slot.view(1, 1, -1)
                    .expand_as(task.global_actions)
                )
            return torch.zeros_like(task.global_actions)
        raise ValueError(
            "mode must be one of: correct, shuffle, no_action"
        )

    def _fixed_capacity_chunk_actions(
        self,
        global_actions: Tensor,
        task_chunk: AV1TaskChunk,
    ) -> Tensor:
        chunk = task_chunk.layout
        valid = global_actions[:, chunk.action_start : chunk.action_end, :]
        padding_count = chunk.action_slot_capacity - chunk.valid_action_count
        if padding_count:
            padding = torch.zeros(
                (
                    self.spec.batch_size,
                    padding_count,
                    self.spec.action_dim,
                ),
                dtype=torch.float32,
                device="cpu",
            )
            return torch.cat((valid, padding), dim=1)
        return valid

    def _video_input_for_chunk(
        self,
        task: AV1SyntheticTask,
        task_chunk: AV1TaskChunk,
    ) -> tuple[Tensor, Tensor]:
        chunk = task_chunk.layout
        tokens = task_chunk.video_latents[:, :, :, 0, 0]
        hidden = self.video_input_projection(tokens)
        timestep = self.video_timestep_projection(
            task.video_timestep[:, task_chunk.chunk_id].view(-1, 1)
        )
        context_mask = task.context_mask[:, task_chunk.chunk_id]
        context = task.context[:, task_chunk.chunk_id]
        mask_float = context_mask.to(dtype=torch.float32).unsqueeze(-1)
        context_summary = (context * mask_float).sum(dim=1) / (
            mask_float.sum(dim=1).clamp_min(1.0)
        )
        proprio_token = self.proprio_context_encoder(
            task.proprio[:, task_chunk.chunk_id]
        )
        hidden = (
            hidden
            + timestep.unsqueeze(1)
            + (context_summary + proprio_token).unsqueeze(1)
        )
        hidden = torch.where(
            task_chunk.frame_valid_mask.unsqueeze(-1),
            hidden,
            torch.zeros_like(hidden),
        )
        return hidden, proprio_token

    def _run_action_diagnostic(
        self,
        *,
        task: AV1SyntheticTask,
        task_chunk: AV1TaskChunk,
        chunk_actions: Tensor,
        bridges: tuple[Tensor, ...],
        proprio_token: Tensor,
    ) -> Tensor:
        action_plan = FixedKPrefixPlan.from_mask(
            task_chunk.action_valid_mask
        )
        compact_actions = action_plan.compact_condition(chunk_actions)
        frame_plan = FixedKPrefixPlan.from_mask(
            task_chunk.frame_valid_mask
        )
        compact_bridges = tuple(
            frame_plan.compact_condition(bridge) for bridge in bridges
        )
        context = torch.cat(
            (
                task.context[:, task_chunk.chunk_id],
                proprio_token.unsqueeze(1),
            ),
            dim=1,
        )
        context_mask = torch.cat(
            (
                task.context_mask[:, task_chunk.chunk_id],
                torch.ones(
                    (self.spec.batch_size, 1),
                    dtype=torch.bool,
                    device="cpu",
                ),
            ),
            dim=1,
        )
        positions = torch.arange(
            task_chunk.layout.action_rope_start,
            task_chunk.layout.action_rope_end,
            dtype=torch.long,
            device="cpu",
        )
        with torch.no_grad():
            frequencies = self.action_backbone._get_rope_freqs_at(
                positions
            ).to(device="cpu")
            prediction = self.action_backbone.forward_with_bridge_tuple(
                compact_actions.detach(),
                tuple(bridge.detach() for bridge in compact_bridges),
                task.action_timestep[:, task_chunk.chunk_id].detach(),
                context=context.detach(),
                context_mask=context_mask,
                action_freqs=frequencies,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
        return prediction

    def run_sequence(
        self,
        task: AV1SyntheticTask,
        *,
        mode: ActionMode = "correct",
        initial_state: AV1TransitionState | None = None,
        target_override: Tensor | None = None,
        run_action_diagnostic: bool = True,
    ) -> AV1SequenceOutput:
        """Run two chunks from a fresh state without reading target bytes."""

        self._validate_task(task)
        if mode not in ("correct", "shuffle", "no_action"):
            raise ValueError("unsupported AV1 action mode")
        if target_override is not None:
            if not isinstance(target_override, Tensor):
                raise TypeError("target_override must be a Tensor")
            if (
                tuple(target_override.shape) != tuple(task.video_target.shape)
                or target_override.dtype != torch.float32
                or target_override.device.type != "cpu"
            ):
                raise ValueError(
                    "target_override must match the loss-only target contract"
                )
            # Deliberately do not read target_override values.
        state = self.empty_state() if initial_state is None else initial_state
        self._validate_initial_state(state)

        seam_actions: Tensor | None = None
        if self.staging_variant is CACHStagingVariant.CACH_A:
            seam_actions = self._global_actions_for_mode(task, mode)
        diagnostic_actions: Tensor | None = None
        if run_action_diagnostic:
            diagnostic_actions = (
                seam_actions
                if seam_actions is not None
                else self._global_actions_for_mode(
                    task,
                    mode,
                    diagnostic_reference=True,
                )
            )

        layer_states = state.layers
        video_chunks: list[Tensor] = []
        action_predictions: list[Tensor] = []
        for task_chunk in task.chunks:
            hidden, proprio_token = self._video_input_for_chunk(
                task,
                task_chunk,
            )
            action_embedding: Tensor | None = None
            local_seam_actions: Tensor | None = None
            if seam_actions is not None:
                local_seam_actions = self._fixed_capacity_chunk_actions(
                    seam_actions,
                    task_chunk,
                )
                if (
                    task_chunk.layout.action_slot_capacity
                    > task_chunk.layout.valid_action_count
                    and bool(
                        local_seam_actions[
                            :, task_chunk.layout.valid_action_count :, :
                        ]
                        .detach()
                        .count_nonzero()
                    )
                ):
                    raise RuntimeError("padded action slots must be exact zero")
                committed = (
                    None
                    if task_chunk.layout.action_start == 0
                    else seam_actions[:, : task_chunk.layout.action_start, :]
                )
                reduced = reduce_end_of_bin_action_condition(
                    local_seam_actions,
                    committed_actions=committed,
                    chunk=task_chunk.layout,
                    no_action_slot=self.cach_core.no_action_slot,
                )
                if self.action_conditioner is None:
                    raise AssertionError("candidate action conditioner is absent")
                action_embedding = self.action_conditioner(reduced.condition)
                action_embedding = torch.where(
                    reduced.latent_valid_mask.unsqueeze(-1),
                    action_embedding,
                    torch.zeros_like(action_embedding),
                )

            next_layer_states: list[MinimalLayerState] = []
            bridges: list[Tensor] = []
            for block_id, block in enumerate(self.blocks):
                hidden, next_layer_state = block.scan_chunk(
                    hidden,
                    layer_states[block_id],
                    frame_valid_mask=task_chunk.frame_valid_mask,
                    action_embedding=action_embedding,
                )
                next_layer_states.append(next_layer_state)
                bridges.append(hidden)
            layer_states = tuple(next_layer_states)
            valid_hidden = hidden[:, : task_chunk.layout.valid_latent_count]
            prediction = self.video_output_projection(valid_hidden)
            video_chunks.append(prediction)

            if run_action_diagnostic:
                assert diagnostic_actions is not None
                local_diagnostic_actions = (
                    local_seam_actions
                    if local_seam_actions is not None
                    and diagnostic_actions is seam_actions
                    else self._fixed_capacity_chunk_actions(
                        diagnostic_actions,
                        task_chunk,
                    )
                )
                action_predictions.append(
                    self._run_action_diagnostic(
                        task=task,
                        task_chunk=task_chunk,
                        chunk_actions=local_diagnostic_actions,
                        bridges=tuple(bridges),
                        proprio_token=proprio_token,
                    )
                )

        video_prediction = torch.cat(video_chunks, dim=1)
        video_prediction = video_prediction.unsqueeze(-1).unsqueeze(-1)
        final_state = AV1TransitionState(
            arm=self.arm_name,
            layout_instance_digest=self.layout.layout_instance_digest,
            batch_size=self.spec.batch_size,
            next_chunk_id=len(task.chunks),
            action_cursor=self.spec.valid_action_count,
            layers=layer_states,
        )
        return AV1SequenceOutput(
            video_prediction=video_prediction,
            action_predictions=tuple(action_predictions),
            final_state=final_state,
        )

    def forward(
        self,
        task: AV1SyntheticTask,
        mode: ActionMode = "correct",
        initial_state: AV1TransitionState | None = None,
        target_override: Tensor | None = None,
        run_action_diagnostic: bool = True,
    ) -> AV1SequenceOutput:
        return self.run_sequence(
            task,
            mode=mode,
            initial_state=initial_state,
            target_override=target_override,
            run_action_diagnostic=run_action_diagnostic,
        )


def _tensor_manifest(
    arm: AV1TransitionProxyArm,
) -> Mapping[str, object]:
    entries: list[dict[str, object]] = []
    for name, parameter in sorted(
        arm.named_parameters(),
        key=lambda item: item[0].encode("utf-8"),
    ):
        if name == "cach_core.no_action_slot":
            role = "frozen_no_action"
        elif name.startswith("action_backbone."):
            role = "frozen_shared_action_diagnostic"
        elif _is_candidate_operator_parameter(name):
            role = "candidate_operator"
        else:
            role = "shared_trainable"
        entries.append(
            {
                "kind": "parameter",
                "name": name,
                "role": role,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
                "tensor_sha256": tensor_digest(parameter),
            }
        )
    for name, buffer in sorted(
        arm.named_buffers(),
        key=lambda item: item[0].encode("utf-8"),
    ):
        entries.append(
            {
                "kind": "buffer",
                "name": name,
                "role": "frozen_buffer",
                "shape": list(buffer.shape),
                "dtype": str(buffer.dtype),
                "requires_grad": bool(buffer.requires_grad),
                "tensor_sha256": tensor_digest(buffer),
            }
        )
    payload: dict[str, object] = {
        "schema": "cach.av1.transition_proxy.tensor_manifest.v1",
        "arm": arm.arm_name,
        "entries": entries,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def build_av1_pair(
    spec: AV1TransitionProxySpec | None = None,
    *,
    task: AV1SyntheticTask | None = None,
) -> AV1TransitionProxyPair:
    """Construct fresh independently owned arms with named initialization."""

    if task is not None and not isinstance(task, AV1SyntheticTask):
        raise TypeError("task must be an AV1SyntheticTask")
    if spec is None:
        spec = task.spec if task is not None else AV1TransitionProxySpec()
    if not isinstance(spec, AV1TransitionProxySpec):
        raise TypeError("spec must be an AV1TransitionProxySpec")
    if task is not None and task.spec != spec:
        raise ValueError("task spec differs from requested pair spec")
    layout = task.layout if task is not None else _build_av1_layout(spec)
    reference = AV1TransitionProxyArm(
        spec=spec,
        layout=layout,
        staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
    )
    candidate = AV1TransitionProxyArm(
        spec=spec,
        layout=layout,
        staging_variant=CACHStagingVariant.CACH_A,
    )
    common = _common_trainable_names(spec.gdn_depth)
    candidate_only = _candidate_only_trainable_names(spec.gdn_depth)
    reference_names = reference.trainable_name_set
    candidate_names = candidate.trainable_name_set
    if reference_names != common or candidate_names != common | candidate_only:
        raise RuntimeError("constructed pair violates the trainable allowlist")

    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    shared_names = set(reference_parameters) & set(candidate_parameters)
    unequal_shared = [
        name
        for name in sorted(shared_names)
        if not torch.equal(
            reference_parameters[name].detach(),
            candidate_parameters[name].detach(),
        )
    ]
    if unequal_shared:
        raise RuntimeError(
            f"theta0 shared parameter mismatch: {unequal_shared[:5]}"
        )
    theta0_manifests = {
        "reference": _tensor_manifest(reference),
        "candidate": _tensor_manifest(candidate),
    }
    return AV1TransitionProxyPair(
        spec=spec,
        layout=layout,
        reference=reference,
        candidate=candidate,
        common_trainable_names=common,
        candidate_only_trainable_names=candidate_only,
        reference_trainable_names=reference_names,
        candidate_trainable_names=candidate_names,
        theta0_manifests=theta0_manifests,
    )


def _state_digest(state: AV1TransitionState) -> str:
    return canonical_sha256(
        {
            "schema": "cach.av1.transition_proxy.state.v1",
            "arm": state.arm,
            "layout_instance_digest": state.layout_instance_digest,
            "batch_size": state.batch_size,
            "next_chunk_id": state.next_chunk_id,
            "action_cursor": state.action_cursor,
            "layers": [
                {
                    "main_s_kv": tensor_digest(layer.main_s_kv),
                    "main_s_z": tensor_digest(layer.main_s_z),
                }
                for layer in state.layers
            ],
        }
    )


def _states_numerically_equal(
    left: AV1TransitionState,
    right: AV1TransitionState,
) -> bool:
    if (
        left.layout_instance_digest != right.layout_instance_digest
        or left.batch_size != right.batch_size
        or left.next_chunk_id != right.next_chunk_id
        or left.action_cursor != right.action_cursor
        or len(left.layers) != len(right.layers)
    ):
        return False
    return all(
        torch.equal(left_layer.main_s_kv, right_layer.main_s_kv)
        and torch.equal(left_layer.main_s_z, right_layer.main_s_z)
        for left_layer, right_layer in zip(left.layers, right.layers)
    )


def _state_is_finite(state: AV1TransitionState) -> bool:
    return all(
        bool(torch.isfinite(layer.main_s_kv.detach()).all())
        and bool(torch.isfinite(layer.main_s_z.detach()).all())
        for layer in state.layers
    )


def _output_is_finite(output: AV1SequenceOutput) -> bool:
    return (
        bool(torch.isfinite(output.video_prediction.detach()).all())
        and all(
            bool(torch.isfinite(value.detach()).all())
            for value in output.action_predictions
        )
        and _state_is_finite(output.final_state)
    )


def _validate_synthetic_data(
    task: AV1SyntheticTask,
) -> tuple[bool, list[str], Mapping[str, object]]:
    reasons: list[str] = []
    spec = task.spec
    expected_shapes = {
        "global_actions": (
            spec.batch_size,
            spec.valid_action_count,
            spec.action_dim,
        ),
        "noisy_video": (
            spec.batch_size,
            spec.valid_latent_count,
            spec.latent_channels,
            1,
            1,
        ),
        "base_target": (
            spec.batch_size,
            spec.valid_latent_count,
            spec.latent_channels,
            1,
            1,
        ),
        "video_target": (
            spec.batch_size,
            spec.valid_latent_count,
            spec.latent_channels,
            1,
            1,
        ),
        "context": (
            spec.batch_size,
            spec.chunk_count,
            spec.context_token_count,
            spec.context_dim,
        ),
        "proprio": (
            spec.batch_size,
            spec.chunk_count,
            spec.action_dim,
        ),
    }
    for name, shape in expected_shapes.items():
        value = getattr(task, name)
        if (
            tuple(value.shape) != shape
            or value.dtype is not torch.float32
            or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())
        ):
            reasons.append(f"DATA_{name.upper()}_CONTRACT_MISMATCH")
    if task.recipe_digest != AV1_TASK_RECIPE_SHA256:
        reasons.append("DATA_RECIPE_DIGEST_MISMATCH")
    if task.row_to_pair != _ROW_TO_PAIR or task.row_sign != _ROW_SIGN:
        reasons.append("DATA_COUNTERFACTUAL_ROW_ORDER_MISMATCH")
    if task.shuffle_permutation != _SHUFFLE_PERMUTATION:
        reasons.append("DATA_SHUFFLE_PERMUTATION_MISMATCH")

    non_action_tensors = (
        task.noisy_video,
        task.base_target,
        task.context,
        task.context_mask,
        task.proprio,
        task.video_timestep,
        task.action_timestep,
    )
    for pair_id in range(spec.counterfactual_pair_count):
        positive = 2 * pair_id
        negative = positive + 1
        if any(
            not torch.equal(value[positive], value[negative])
            for value in non_action_tensors
        ):
            reasons.append(f"DATA_NON_ACTION_PAIR_{pair_id}_DIFFERS")
        if not torch.equal(
            task.global_actions[positive],
            -task.global_actions[negative],
        ):
            reasons.append(f"DATA_ACTION_PAIR_{pair_id}_NOT_NEGATION")
        for horizon in spec.scored_future_horizons:
            if torch.equal(
                task.video_target[positive, horizon],
                task.video_target[negative, horizon],
            ):
                reasons.append(
                    f"DATA_TARGET_PAIR_{pair_id}_H{horizon}_IDENTICAL"
                )

    reconstructed_actions: list[Tensor] = []
    for task_chunk in task.chunks:
        chunk = task_chunk.layout
        expected_video_shape = (
            spec.batch_size,
            spec.frame_chunk_size,
            spec.latent_channels,
            1,
            1,
        )
        expected_action_shape = (
            spec.batch_size,
            chunk.action_slot_capacity,
            spec.action_dim,
        )
        if tuple(task_chunk.video_latents.shape) != expected_video_shape:
            reasons.append(
                f"DATA_CHUNK_{chunk.chunk_id}_VIDEO_SHAPE_MISMATCH"
            )
        if tuple(task_chunk.actions.shape) != expected_action_shape:
            reasons.append(
                f"DATA_CHUNK_{chunk.chunk_id}_ACTION_SHAPE_MISMATCH"
            )
        if (
            chunk.valid_latent_count < spec.frame_chunk_size
            and bool(
                task_chunk.video_latents[
                    :, chunk.valid_latent_count :, :, :, :
                ].count_nonzero()
            )
        ):
            reasons.append(
                f"DATA_CHUNK_{chunk.chunk_id}_VIDEO_PADDING_NONZERO"
            )
        if (
            chunk.valid_action_count < chunk.action_slot_capacity
            and bool(
                task_chunk.actions[
                    :, chunk.valid_action_count :, :
                ].count_nonzero()
            )
        ):
            reasons.append(
                f"DATA_CHUNK_{chunk.chunk_id}_ACTION_PADDING_NONZERO"
            )
        reconstructed_actions.append(
            task_chunk.actions[:, : chunk.valid_action_count, :]
        )
        expected_frame_mask = (
            torch.tensor(chunk.latent_valid_mask, dtype=torch.bool)
            .view(1, -1)
            .expand(spec.batch_size, -1)
        )
        expected_action_mask = (
            torch.tensor(chunk.action_valid_mask, dtype=torch.bool)
            .view(1, -1)
            .expand(spec.batch_size, -1)
        )
        if not torch.equal(
            task_chunk.frame_valid_mask,
            expected_frame_mask,
        ):
            reasons.append(
                f"DATA_CHUNK_{chunk.chunk_id}_FRAME_MASK_MISMATCH"
            )
        if not torch.equal(
            task_chunk.action_valid_mask,
            expected_action_mask,
        ):
            reasons.append(
                f"DATA_CHUNK_{chunk.chunk_id}_ACTION_MASK_MISMATCH"
            )
    if not torch.equal(
        torch.cat(reconstructed_actions, dim=1),
        task.global_actions,
    ):
        reasons.append("DATA_VALID_ACTIONS_NOT_EXACTLY_ONCE")

    effect_rms: dict[str, float] = {}
    pair_delta_rms: dict[str, float] = {}
    effect = task.video_target - task.base_target
    if bool(effect[:, 0].count_nonzero()):
        reasons.append("DATA_ANCHOR_ACTION_EFFECT_NONZERO")
    for horizon in spec.scored_future_horizons:
        horizon_effect = effect[:, horizon, :, 0, 0].double()
        rms = float(horizon_effect.square().mean().sqrt().item())
        effect_rms[str(horizon)] = rms
        if not (0.499999 <= rms <= 0.500001):
            reasons.append(f"DATA_EFFECT_RMS_H{horizon}_OUT_OF_RANGE")
        deltas = (
            task.video_target[0::2, horizon, :, 0, 0]
            - task.video_target[1::2, horizon, :, 0, 0]
        ).double()
        delta_rms = float(deltas.square().mean().sqrt().item())
        pair_delta_rms[str(horizon)] = delta_rms
        if not (0.999998 <= delta_rms <= 1.000002):
            reasons.append(
                f"DATA_PAIR_DELTA_RMS_H{horizon}_OUT_OF_RANGE"
            )

    digest_sources: dict[str, Tensor] = {
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
    for task_chunk in task.chunks:
        prefix = f"chunk{task_chunk.chunk_id}"
        digest_sources[f"{prefix}_video_latents"] = task_chunk.video_latents
        digest_sources[f"{prefix}_actions"] = task_chunk.actions
        digest_sources[
            f"{prefix}_frame_valid_mask"
        ] = task_chunk.frame_valid_mask
        digest_sources[
            f"{prefix}_action_valid_mask"
        ] = task_chunk.action_valid_mask
    for name, value in digest_sources.items():
        if task.tensor_digests.get(name) != tensor_digest(value):
            reasons.append(f"DATA_TENSOR_DIGEST_MISMATCH_{name.upper()}")

    diagnostics: Mapping[str, object] = {
        "effect_rms_by_horizon": effect_rms,
        "target_pair_delta_rms_by_horizon": pair_delta_rms,
        "layout_instance_digest": task.layout.layout_instance_digest,
        "task_recipe_sha256": task.recipe_digest,
        "tensor_digests": dict(task.tensor_digests),
    }
    return not reasons, reasons, diagnostics


def _train_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction[:, 1:] - target[:, 1:]).square().mean()


def _mse_by_horizon(
    prediction: Tensor,
    target: Tensor,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for horizon in range(1, 5):
        difference = (
            prediction[:, horizon, :, 0, 0].double()
            - target[:, horizon, :, 0, 0].double()
        )
        result[str(horizon)] = float(difference.square().mean().item())
    return result


def _primary_metric(mse_by_horizon: Mapping[str, float]) -> float:
    return (
        0.20 * float(mse_by_horizon["1"])
        + 0.30 * float(mse_by_horizon["2"])
        + 0.50 * float(mse_by_horizon["4"])
    )


def _counterfactual_delta_nmse_by_horizon(
    prediction: Tensor,
    target: Tensor,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for horizon in range(1, 5):
        predicted_delta = (
            prediction[0::2, horizon, :, 0, 0].double()
            - prediction[1::2, horizon, :, 0, 0].double()
        )
        target_delta = (
            target[0::2, horizon, :, 0, 0].double()
            - target[1::2, horizon, :, 0, 0].double()
        )
        numerator = (predicted_delta - target_delta).square().sum()
        denominator = target_delta.square().sum() + _EPSILON
        result[str(horizon)] = float((numerator / denominator).item())
    return result


def _counterfactual_delta_nmse(
    prediction: Tensor,
    target: Tensor,
) -> float:
    predicted_delta = (
        prediction[0::2, 1:, :, 0, 0].double()
        - prediction[1::2, 1:, :, 0, 0].double()
    )
    target_delta = (
        target[0::2, 1:, :, 0, 0].double()
        - target[1::2, 1:, :, 0, 0].double()
    )
    numerator = (predicted_delta - target_delta).square().sum()
    denominator = target_delta.square().sum() + _EPSILON
    return float((numerator / denominator).item())


def _relative_gap_by_horizon(
    worse: Mapping[str, float],
    better: Mapping[str, float],
) -> dict[str, float]:
    return {
        str(horizon): (
            float(worse[str(horizon)]) - float(better[str(horizon)])
        )
        / max(float(worse[str(horizon)]), _EPSILON)
        for horizon in range(1, 5)
    }


def _all_finite_scalars(value: object) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite_scalars(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite_scalars(item) for item in value)
    return True


def _seam_weight_names(arm: AV1TransitionProxyArm) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in arm.named_parameters()
        if parameter.requires_grad
        and name.startswith("blocks.")
        and name.endswith("action_output_projection.weight")
    )


def _seam_grad_rms(arm: AV1TransitionProxyArm) -> float:
    parameters = dict(arm.named_parameters())
    square_sum = torch.zeros((), dtype=torch.float64)
    count = 0
    for name in _seam_weight_names(arm):
        gradient = parameters[name].grad
        if gradient is None:
            raise RuntimeError(f"missing seam gradient for {name}")
        square_sum = square_sum + gradient.detach().double().square().sum()
        count += gradient.numel()
    if count <= 0:
        raise RuntimeError("candidate exposes no seam weights")
    return float((square_sum / count).sqrt().item())


def _seam_update_rms(
    arm: AV1TransitionProxyArm,
    theta0_weights: Mapping[str, Tensor],
) -> float:
    parameters = dict(arm.named_parameters())
    square_sum = torch.zeros((), dtype=torch.float64)
    count = 0
    for name, initial in theta0_weights.items():
        difference = parameters[name].detach().double() - initial.double()
        square_sum = square_sum + difference.square().sum()
        count += difference.numel()
    if count <= 0:
        raise RuntimeError("candidate exposes no seam update weights")
    return float((square_sum / count).sqrt().item())


def _parameter_digest_manifest(
    arm: AV1TransitionProxyArm,
) -> Mapping[str, object]:
    parameter_digests = {
        name: tensor_digest(parameter)
        for name, parameter in sorted(
            arm.named_parameters(),
            key=lambda item: item[0].encode("utf-8"),
        )
    }
    buffer_digests = {
        name: tensor_digest(buffer)
        for name, buffer in sorted(
            arm.named_buffers(),
            key=lambda item: item[0].encode("utf-8"),
        )
    }
    payload: dict[str, object] = {
        "schema": "cach.av1.transition_proxy.final_parameters.v1",
        "arm": arm.arm_name,
        "parameters": parameter_digests,
        "buffers": buffer_digests,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def _train_arm(
    arm: AV1TransitionProxyArm,
    task: AV1SyntheticTask,
    *,
    steps: int,
    progress_callback: Callable[[Mapping[str, object]], object] | None,
) -> tuple[list[float], float | None, bool]:
    named_parameters = dict(arm.named_parameters())
    trainable_names = sorted(
        arm.trainable_name_set,
        key=lambda value: value.encode("utf-8"),
    )
    optimizer_parameters = [named_parameters[name] for name in trainable_names]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=0.003,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {id(named_parameters[name]) for name in trainable_names}
    if optimizer_ids != expected_ids:
        raise RuntimeError("optimizer parameter group differs from allowlist")

    arm.train(True)
    loss_trace: list[float] = []
    seam_grad_step0: float | None = None
    finite = True
    for optimizer_step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = arm.run_sequence(
            task,
            mode="correct",
            run_action_diagnostic=False,
        )
        loss = _train_loss(output.video_prediction, task.video_target)
        loss_value = float(loss.detach().double().item())
        loss_trace.append(loss_value)
        if not math.isfinite(loss_value) or not _output_is_finite(output):
            finite = False
            raise FloatingPointError("non-finite AV1 training forward")
        loss.backward()
        for name in trainable_names:
            gradient = named_parameters[name].grad
            if gradient is not None and not bool(
                torch.isfinite(gradient.detach()).all()
            ):
                finite = False
                raise FloatingPointError(f"non-finite gradient for {name}")
        if (
            optimizer_step == 1
            and arm.staging_variant is CACHStagingVariant.CACH_A
        ):
            seam_grad_step0 = _seam_grad_rms(arm)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            optimizer_parameters,
            max_norm=1.0,
        )
        if not bool(torch.isfinite(gradient_norm.detach())):
            finite = False
            raise FloatingPointError("non-finite global gradient norm")
        optimizer.step()
        if bool(arm.cach_core.no_action_slot.detach().count_nonzero()):
            finite = False
            raise RuntimeError("NO_ACTION changed during optimizer update")
        if any(
            not bool(torch.isfinite(parameter.detach()).all())
            for parameter in optimizer_parameters
        ):
            finite = False
            raise FloatingPointError("non-finite parameter update")
    with torch.no_grad():
        final_output = arm.run_sequence(
            task,
            mode="correct",
            run_action_diagnostic=False,
        )
        final_loss = _train_loss(
            final_output.video_prediction,
            task.video_target,
        )
    final_loss_value = float(final_loss.double().item())
    loss_trace.append(final_loss_value)
    finite = (
        finite
        and math.isfinite(final_loss_value)
        and _output_is_finite(final_output)
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "arm_complete",
                "arm": arm.arm_name,
                "optimizer_steps_completed": steps,
                "final_loss": final_loss_value,
            }
        )
    return loss_trace, seam_grad_step0, finite


def _candidate_seam_directional_jvp(
    arm: AV1TransitionProxyArm,
    task: AV1SyntheticTask,
) -> float:
    parameters = {
        name: parameter
        for name, parameter in arm.named_parameters()
        if parameter.requires_grad
    }
    seam_names = set(_seam_weight_names(arm))
    total_seam_elements = sum(
        parameters[name].numel() for name in seam_names
    )
    if total_seam_elements <= 0:
        raise RuntimeError("candidate has no seam parameters for JVP")
    scale = 1.0 / math.sqrt(float(total_seam_elements))
    tangents: dict[str, Tensor] = {}
    offset = 0
    for name, parameter in parameters.items():
        if name not in seam_names:
            tangents[name] = torch.zeros_like(parameter)
            continue
        flat_index = torch.arange(
            offset,
            offset + parameter.numel(),
            dtype=torch.int64,
            device="cpu",
        )
        direction = torch.where(
            flat_index.remainder(2) == 0,
            torch.ones_like(flat_index, dtype=torch.float32),
            -torch.ones_like(flat_index, dtype=torch.float32),
        )
        tangents[name] = direction.reshape_as(parameter) * scale
        offset += parameter.numel()

    def prediction_from_parameters(
        values: Mapping[str, Tensor],
    ) -> Tensor:
        output = torch.func.functional_call(
            arm,
            values,
            (task,),
            {
                "mode": "correct",
                "initial_state": None,
                "target_override": None,
                "run_action_diagnostic": False,
            },
            strict=False,
        )
        return output.video_prediction

    _, tangent = torch.func.jvp(
        prediction_from_parameters,
        (parameters,),
        (tangents,),
    )
    return float(tangent.detach().double().square().mean().sqrt().item())


def _target_leakage_diagnostics(
    arm: AV1TransitionProxyArm,
    task: AV1SyntheticTask,
) -> Mapping[str, object]:
    baseline = arm.run_sequence(
        task,
        mode="correct",
        target_override=task.video_target,
        run_action_diagnostic=False,
    )
    perturbed_target = (
        task.video_target.detach().clone() * -7.0 + 13.0
    )
    perturbed = arm.run_sequence(
        task,
        mode="correct",
        target_override=perturbed_target,
        run_action_diagnostic=False,
    )
    perturbation_changes_prediction = not torch.equal(
        baseline.video_prediction,
        perturbed.video_prediction,
    )
    perturbation_changes_state = not _states_numerically_equal(
        baseline.final_state,
        perturbed.final_state,
    )

    reverse_target = task.video_target.detach().clone().requires_grad_(True)
    reverse_output = arm.run_sequence(
        task,
        mode="correct",
        target_override=reverse_target,
        run_action_diagnostic=False,
    )
    reverse_gradient = torch.autograd.grad(
        reverse_output.video_prediction.sum(),
        reverse_target,
        allow_unused=True,
    )[0]
    reverse_gradient_nonzero = (
        reverse_gradient is not None
        and bool(reverse_gradient.detach().count_nonzero())
    )

    def prediction_from_target(target: Tensor) -> Tensor:
        return arm.run_sequence(
            task,
            mode="correct",
            target_override=target,
            run_action_diagnostic=False,
        ).video_prediction

    _, target_tangent = torch.func.jvp(
        prediction_from_target,
        (task.video_target,),
        (torch.ones_like(task.video_target),),
    )
    target_jvp_rms = float(
        target_tangent.detach().double().square().mean().sqrt().item()
    )
    return {
        "target_perturbation_changes_prediction": (
            perturbation_changes_prediction
        ),
        "target_perturbation_changes_state": perturbation_changes_state,
        "target_reverse_gradient_nonzero": reverse_gradient_nonzero,
        "target_jvp_rms": target_jvp_rms,
        "target_jvp_exact_zero": bool(
            not target_tangent.detach().count_nonzero()
        ),
    }


def _read_steps_and_spec(
    config: Mapping[str, object],
) -> tuple[AV1TransitionProxySpec, int, bool]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a Mapping")
    if config.get("device", "cpu") != "cpu":
        raise ValueError("AV1 transition proxy refuses non-CPU device")
    if config.get("dtype", "float32") not in ("float32", torch.float32):
        raise ValueError("AV1 transition proxy refuses non-float32 dtype")
    if config.get("checkpoint") is not None:
        raise ValueError("AV1 transition proxy refuses checkpoints")
    if config.get("dataset_path") is not None:
        raise ValueError("AV1 transition proxy refuses real datasets")
    configured_spec = config.get("spec")
    if configured_spec is None:
        spec = AV1TransitionProxySpec()
    elif isinstance(configured_spec, AV1TransitionProxySpec):
        spec = configured_spec
    else:
        raise TypeError("config['spec'] must be an AV1TransitionProxySpec")

    if "steps" in config:
        steps = config["steps"]
    else:
        budget = config.get("budget")
        if isinstance(budget, Mapping):
            steps = budget.get("optimizer_steps_per_arm", 200)
        else:
            steps = 200
    if type(steps) is not int or not (1 <= steps <= 300):
        raise ValueError("optimizer steps must be a plain integer in [1, 300]")

    runtime = config.get("runtime")
    if isinstance(runtime, Mapping):
        if runtime.get("device", "cpu") != "cpu":
            raise ValueError("AV1 transition proxy refuses non-CPU runtime")
        if runtime.get("dtype", "float32") != "float32":
            raise ValueError("AV1 transition proxy refuses non-float32 runtime")
        if bool(runtime.get("allow_cuda_initialization", False)):
            raise ValueError("AV1 transition proxy refuses CUDA initialization")
        if bool(runtime.get("allow_compile_or_jit", False)):
            raise ValueError("AV1 transition proxy refuses compile/JIT")
        if bool(runtime.get("allow_checkpoint", False)):
            raise ValueError("AV1 transition proxy refuses checkpoints")
        if bool(runtime.get("allow_real_data", False)):
            raise ValueError("AV1 transition proxy refuses real data")

    identity = config.get("identity")
    if isinstance(identity, Mapping):
        card_sha = identity.get("run_card_sha256", AV0_CARD_SHA256)
        if card_sha != AV0_CARD_SHA256:
            raise ValueError("run-card SHA256 differs from the AV1 source pin")

    topology = config.get("topology")
    if isinstance(topology, Mapping):
        topology_contract = {
            "batch_size": spec.batch_size,
            "latent_channels": spec.latent_channels,
            "hidden_dim": spec.hidden_dim,
            "context_dim": spec.context_dim,
            "action_dim": spec.action_dim,
            "gdn_depth": spec.gdn_depth,
            "frame_chunk_size": spec.frame_chunk_size,
            "temporal_compression": spec.temporal_compression,
            "video_stride": spec.video_stride,
            "valid_raw_count": spec.valid_raw_count,
            "valid_latent_count": spec.valid_latent_count,
            "valid_action_count": spec.valid_action_count,
            "chunk_count": spec.chunk_count,
        }
        for name, required in topology_contract.items():
            if topology.get(name, required) != required:
                raise ValueError(
                    f"config topology {name} differs from frozen AV0 value"
                )

    registered_budget = (
        steps == 200
        and isinstance(config.get("budget"), Mapping)
        and config.get("schema") == "cach.av1.transition_proxy.config.v1"
    )
    return spec, steps, registered_budget


def run_av1_screen(
    config: Mapping[str, object],
    progress_callback: Callable[[Mapping[str, object]], object] | None = None,
) -> dict[str, object]:
    """Execute the CPU synthetic screen without writing files or creating roots."""

    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    start_time = time.perf_counter()
    spec, steps, registered_budget = _read_steps_and_spec(config)

    # Task tensor digests are complete before either model is constructed.
    task = build_av1_synthetic_task(spec)
    pair = build_av1_pair(spec, task=task)
    theta0_manifests = {
        "reference": pair.reference_theta0_manifest,
        "candidate": pair.candidate_theta0_manifest,
    }
    if progress_callback is not None:
        # This must remain the first callback and must precede every forward.
        progress_callback(
            {
                "event": "theta0_manifests_ready",
                "payload": theta0_manifests,
            }
        )

    reasons: list[str] = []
    data_valid, data_reasons, data_diagnostics = _validate_synthetic_data(task)
    reasons.extend(data_reasons)

    implementation_valid = True
    common = _common_trainable_names(spec.gdn_depth)
    candidate_only = _candidate_only_trainable_names(spec.gdn_depth)
    if pair.reference.trainable_name_set != common:
        implementation_valid = False
        reasons.append("IMPLEMENTATION_REFERENCE_ALLOWLIST_MISMATCH")
    if pair.candidate.trainable_name_set != common | candidate_only:
        implementation_valid = False
        reasons.append("IMPLEMENTATION_CANDIDATE_ALLOWLIST_MISMATCH")
    if pair.reference.action_conditioner is not None:
        implementation_valid = False
        reasons.append("IMPLEMENTATION_REFERENCE_ACTION_PATH_PRESENT")
    if pair.candidate.action_conditioner is None:
        implementation_valid = False
        reasons.append("IMPLEMENTATION_CANDIDATE_ACTION_PATH_ABSENT")
    if not isinstance(pair.reference.action_backbone, ActionDiT) or not isinstance(
        pair.candidate.action_backbone,
        ActionDiT,
    ):
        implementation_valid = False
        reasons.append("IMPLEMENTATION_REAL_ACTIONDIT_ABSENT")
    if any(
        parameter.requires_grad
        for arm in (pair.reference, pair.candidate)
        for parameter in arm.action_backbone.parameters()
    ):
        implementation_valid = False
        reasons.append("IMPLEMENTATION_ACTIONDIT_NOT_FROZEN")

    theta0_weights = {
        name: dict(pair.candidate.named_parameters())[name].detach().clone()
        for name in _seam_weight_names(pair.candidate)
    }
    initial_outputs: dict[str, dict[str, AV1SequenceOutput]] = {
        "reference": {},
        "candidate": {},
    }
    finite_all = True
    for arm_key, arm in (
        ("reference", pair.reference),
        ("candidate", pair.candidate),
    ):
        for mode in ("correct", "shuffle", "no_action"):
            output = arm.run_sequence(
                task,
                mode=mode,
                run_action_diagnostic=(mode == "correct"),
            )
            initial_outputs[arm_key][mode] = output
            finite_all = finite_all and _output_is_finite(output)

    theta0_exact_identity = True
    for mode in ("correct", "shuffle", "no_action"):
        reference_video = initial_outputs["reference"][mode].video_prediction
        candidate_video = initial_outputs["candidate"][mode].video_prediction
        if not torch.equal(reference_video, candidate_video):
            theta0_exact_identity = False
            reasons.append(f"THETA0_PAIR_OUTPUT_MISMATCH_{mode.upper()}")
    for arm_key in ("reference", "candidate"):
        correct_video = initial_outputs[arm_key]["correct"].video_prediction
        for mode in ("shuffle", "no_action"):
            if not torch.equal(
                correct_video,
                initial_outputs[arm_key][mode].video_prediction,
            ):
                theta0_exact_identity = False
                reasons.append(
                    f"THETA0_{arm_key.upper()}_MODE_MISMATCH_{mode.upper()}"
                )
    if not _states_numerically_equal(
        initial_outputs["reference"]["correct"].final_state,
        initial_outputs["candidate"]["correct"].final_state,
    ):
        theta0_exact_identity = False
        reasons.append("THETA0_PAIR_STATE_NUMERICAL_MISMATCH")

    reference_empty = pair.reference.empty_state()
    candidate_empty = pair.candidate.empty_state()
    empty_state_zero_equal = _states_numerically_equal(
        reference_empty,
        candidate_empty,
    )
    if not empty_state_zero_equal:
        implementation_valid = False
        reasons.append("IMPLEMENTATION_CROSS_ARM_EMPTY_STATE_MISMATCH")
    empty_state_digests = {
        "reference": _state_digest(reference_empty),
        "candidate": _state_digest(candidate_empty),
    }
    if empty_state_digests["reference"] == empty_state_digests["candidate"]:
        implementation_valid = False
        reasons.append("IMPLEMENTATION_CROSS_ARM_IDENTITY_NOT_DISTINCT")

    seam_directional_jvp_rms_theta0 = _candidate_seam_directional_jvp(
        pair.candidate,
        task,
    )
    if not math.isfinite(seam_directional_jvp_rms_theta0):
        finite_all = False
        reasons.append("NUMERICS_NONFINITE_SEAM_JVP")
    if seam_directional_jvp_rms_theta0 <= 0.0:
        implementation_valid = False
        reasons.append("IMPLEMENTATION_ZERO_SEAM_DIRECTIONAL_JVP")

    reference_loss_trace, _, reference_finite = _train_arm(
        pair.reference,
        task,
        steps=steps,
        progress_callback=progress_callback,
    )
    candidate_loss_trace, seam_grad_rms_step0, candidate_finite = _train_arm(
        pair.candidate,
        task,
        steps=steps,
        progress_callback=progress_callback,
    )
    finite_all = finite_all and reference_finite and candidate_finite
    if seam_grad_rms_step0 is None:
        raise AssertionError("candidate seam gradient was not captured")

    final_outputs: dict[str, dict[str, AV1SequenceOutput]] = {
        "reference": {},
        "candidate": {},
    }
    for arm_key, arm in (
        ("reference", pair.reference),
        ("candidate", pair.candidate),
    ):
        for mode in ("correct", "shuffle", "no_action"):
            output = arm.run_sequence(
                task,
                mode=mode,
                run_action_diagnostic=(mode == "correct"),
            )
            final_outputs[arm_key][mode] = output
            finite_all = finite_all and _output_is_finite(output)

    for mode in ("shuffle", "no_action"):
        if not torch.equal(
            final_outputs["reference"]["correct"].video_prediction,
            final_outputs["reference"][mode].video_prediction,
        ):
            implementation_valid = False
            reasons.append(
                f"IMPLEMENTATION_TRAINED_REFERENCE_MODE_MISMATCH_{mode.upper()}"
            )
    reference_video = final_outputs["reference"]["correct"].video_prediction
    for pair_id in range(spec.counterfactual_pair_count):
        if not torch.equal(
            reference_video[2 * pair_id],
            reference_video[2 * pair_id + 1],
        ):
            implementation_valid = False
            reasons.append(
                f"IMPLEMENTATION_REFERENCE_PAIR_{pair_id}_DIFFERS"
            )
    candidate_no_action = final_outputs["candidate"][
        "no_action"
    ].video_prediction
    for pair_id in range(spec.counterfactual_pair_count):
        if not torch.equal(
            candidate_no_action[2 * pair_id],
            candidate_no_action[2 * pair_id + 1],
        ):
            implementation_valid = False
            reasons.append(
                f"IMPLEMENTATION_CANDIDATE_NO_ACTION_PAIR_{pair_id}_DIFFERS"
            )

    # Re-run after a different mode order.  Every call owns a fresh state.
    reordered_correct = pair.candidate.run_sequence(
        task,
        mode="correct",
        run_action_diagnostic=False,
    )
    state_order_independent = (
        torch.equal(
            reordered_correct.video_prediction,
            final_outputs["candidate"]["correct"].video_prediction,
        )
        and _states_numerically_equal(
            reordered_correct.final_state,
            final_outputs["candidate"]["correct"].final_state,
        )
    )
    if not state_order_independent:
        implementation_valid = False
        reasons.append("IMPLEMENTATION_MODE_ORDER_OR_STATE_REUSE_DETECTED")

    target_leakage = _target_leakage_diagnostics(pair.candidate, task)
    target_leakage_valid = (
        not bool(target_leakage["target_perturbation_changes_prediction"])
        and not bool(target_leakage["target_perturbation_changes_state"])
        and not bool(target_leakage["target_reverse_gradient_nonzero"])
        and bool(target_leakage["target_jvp_exact_zero"])
    )
    if not target_leakage_valid:
        implementation_valid = False
        reasons.append("IMPLEMENTATION_TARGET_LEAKAGE_DETECTED")

    no_action_frozen = (
        not pair.reference.cach_core.no_action_slot.requires_grad
        and not pair.candidate.cach_core.no_action_slot.requires_grad
        and not bool(
            pair.reference.cach_core.no_action_slot.detach().count_nonzero()
        )
        and not bool(
            pair.candidate.cach_core.no_action_slot.detach().count_nonzero()
        )
    )
    if not no_action_frozen:
        implementation_valid = False
        reasons.append("IMPLEMENTATION_NO_ACTION_NOT_FROZEN_ZERO")

    candidate_mse = {
        mode: _mse_by_horizon(
            final_outputs["candidate"][mode].video_prediction,
            task.video_target,
        )
        for mode in ("correct", "shuffle", "no_action")
    }
    reference_mse = _mse_by_horizon(
        final_outputs["reference"]["correct"].video_prediction,
        task.video_target,
    )
    candidate_primary = {
        mode: _primary_metric(candidate_mse[mode])
        for mode in ("correct", "shuffle", "no_action")
    }
    reference_primary = _primary_metric(reference_mse)
    shuffle_gap_by_horizon = _relative_gap_by_horizon(
        candidate_mse["shuffle"],
        candidate_mse["correct"],
    )
    no_action_gap_by_horizon = _relative_gap_by_horizon(
        candidate_mse["no_action"],
        candidate_mse["correct"],
    )
    reference_delta_by_horizon = (
        _counterfactual_delta_nmse_by_horizon(
            final_outputs["reference"]["correct"].video_prediction,
            task.video_target,
        )
    )
    candidate_delta_by_horizon = (
        _counterfactual_delta_nmse_by_horizon(
            final_outputs["candidate"]["correct"].video_prediction,
            task.video_target,
        )
    )
    seam_update_rms_final = _seam_update_rms(
        pair.candidate,
        theta0_weights,
    )
    metrics: dict[str, object] = {
        "loss_drop_rel": (
            candidate_loss_trace[0] - candidate_loss_trace[-1]
        )
        / max(candidate_loss_trace[0], _EPSILON),
        "seam_grad_rms_step0": seam_grad_rms_step0,
        "seam_update_rms_final": seam_update_rms_final,
        "counterfactual_delta_nmse_correct": _counterfactual_delta_nmse(
            final_outputs["candidate"]["correct"].video_prediction,
            task.video_target,
        ),
        "counterfactual_delta_nmse_correct_by_horizon": (
            candidate_delta_by_horizon
        ),
        "shuffle_gap_rel": (
            candidate_primary["shuffle"] - candidate_primary["correct"]
        )
        / max(candidate_primary["shuffle"], _EPSILON),
        "shuffle_gap_rel_by_horizon": shuffle_gap_by_horizon,
        "no_action_gap_rel": (
            candidate_primary["no_action"] - candidate_primary["correct"]
        )
        / max(candidate_primary["no_action"], _EPSILON),
        "no_action_gap_rel_by_horizon": no_action_gap_by_horizon,
        "candidate_gain_vs_reference": (
            reference_primary - candidate_primary["correct"]
        )
        / max(reference_primary, _EPSILON),
        "reference_delta_nmse_by_horizon": reference_delta_by_horizon,
        "candidate_mse_by_mode_and_horizon": candidate_mse,
        "reference_mse_by_horizon": reference_mse,
        "candidate_primary_by_mode": candidate_primary,
        "reference_primary": reference_primary,
    }
    finite_all = finite_all and _all_finite_scalars(metrics)
    if not finite_all:
        reasons.append("NUMERICS_NONFINITE")

    numerics_valid = (
        finite_all
        and no_action_frozen
        and target_leakage_valid
        and math.isfinite(seam_grad_rms_step0)
        and math.isfinite(seam_update_rms_final)
    )
    final_parameter_digests = {
        "reference": _parameter_digest_manifest(pair.reference),
        "candidate": _parameter_digest_manifest(pair.candidate),
    }
    elapsed_seconds = time.perf_counter() - start_time
    max_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

    diagnostics: dict[str, object] = {
        "test_only": not registered_budget,
        "registered_budget": registered_budget,
        "registered_200_step_budget_executed": (
            registered_budget and steps == 200
        ),
        "device": "cpu",
        "dtype": "float32",
        "operator_level": "TRANSITION_PROXY",
        "integration_path": "EXPERIMENTAL_PATH",
        "result_scope": "PROXY_ONLY_ACTUAL_ARCHITECTURE_NOT_ASSESSED",
        "av0_card_sha256": AV0_CARD_SHA256,
        "task": data_diagnostics,
        "trainable_names": {
            "reference": sorted(pair.reference.trainable_name_set),
            "candidate": sorted(pair.candidate.trainable_name_set),
            "candidate_only": sorted(candidate_only),
        },
        "empty_state_digests": empty_state_digests,
        "empty_state_zero_tensor_equality": empty_state_zero_equal,
        "state_order_independent": state_order_independent,
        "target_leakage": target_leakage,
        "no_action_frozen_exact_zero": no_action_frozen,
        "seam_directional_jvp_rms_theta0": (
            seam_directional_jvp_rms_theta0
        ),
        "action_diagnostic": {
            "class": type(pair.candidate.action_backbone).__name__,
            "depth": spec.gdn_depth,
            "reference_theta0_prediction_digests": [
                tensor_digest(value)
                for value in initial_outputs["reference"][
                    "correct"
                ].action_predictions
            ],
            "candidate_theta0_prediction_digests": [
                tensor_digest(value)
                for value in initial_outputs["candidate"][
                    "correct"
                ].action_predictions
            ],
            "reference_final_prediction_digests": [
                tensor_digest(value)
                for value in final_outputs["reference"][
                    "correct"
                ].action_predictions
            ],
            "candidate_final_prediction_digests": [
                tensor_digest(value)
                for value in final_outputs["candidate"][
                    "correct"
                ].action_predictions
            ],
            "loss_weight": 0.0,
            "frozen": True,
        },
        "optimizer_steps_completed_by_arm": {
            "REF-GDN-CORRECTED": steps,
            "CACH-A": steps,
        },
        "commit_diagnostic": {
            "update_free": True,
            "target_bytes_read": False,
            "reference_final_chunk_id": final_outputs["reference"][
                "correct"
            ].final_state.next_chunk_id,
            "candidate_final_chunk_id": final_outputs["candidate"][
                "correct"
            ].final_state.next_chunk_id,
            "reference_action_cursor": final_outputs["reference"][
                "correct"
            ].final_state.action_cursor,
            "candidate_action_cursor": final_outputs["candidate"][
                "correct"
            ].final_state.action_cursor,
        },
    }
    validity = {
        "finite_all": finite_all,
        "theta0_exact_identity": theta0_exact_identity,
        "implementation_valid": implementation_valid,
        "data_valid": data_valid,
        "numerics_valid": numerics_valid,
        "reasons": sorted(set(reasons)),
    }
    result: dict[str, object] = {
        "schema": "cach.av1.transition_proxy.screen.v1",
        "validity": validity,
        "metrics": metrics,
        "loss_trace": {
            "reference": reference_loss_trace,
            "candidate": candidate_loss_trace,
        },
        "theta0_manifests": theta0_manifests,
        "final_parameter_digests": final_parameter_digests,
        "diagnostics": diagnostics,
        "resource_usage": {
            "device": "cpu",
            "dtype": "float32",
            "wall_seconds": elapsed_seconds,
            "max_process_rss_bytes": max_rss_bytes,
            "optimizer_steps_per_arm": steps,
            "optimizer_steps_total": steps * 2,
            "filesystem_writes": 0,
            "run_root_created": False,
            "gpu_used": False,
            "checkpoint_loaded": False,
            "checkpoint_saved": False,
            "real_data_used": False,
        },
    }
    if progress_callback is not None:
        progress_callback(
            {
                "event": "screen_complete",
                "optimizer_steps_completed_by_arm": diagnostics[
                    "optimizer_steps_completed_by_arm"
                ],
                "validity": validity,
                "metrics": metrics,
            }
        )
    return result


__all__ = [
    "AV0_CARD_SHA256",
    "AV1SequenceOutput",
    "AV1SyntheticTask",
    "AV1TaskChunk",
    "AV1TransitionProxyArm",
    "AV1TransitionProxyPair",
    "AV1TransitionProxySpec",
    "AV1TransitionState",
    "build_av1_pair",
    "build_av1_synthetic_task",
    "run_av1_screen",
]
