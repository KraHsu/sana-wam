"""Authorized, update-free CUDA smoke for the synthetic REF/CACH mini pair.

This module is an isolated device-orchestration surface around the reviewed
CPU canonical pair in :mod:`sana_wam.model.cach_minimal_cpu`.  The canonical
modules and their registered parameters remain on CPU.  Independent FP32 CUDA
tensor shadows are supplied with :func:`torch.func.functional_call`; no model
is moved, no parameter is updated, and no optimizer is constructible through
the public API.

The single public entry point has a fixed synthetic shape and a fixed GPU UUID.
It is not a trainer, evaluator, checkpoint loader, real-data path, vendor GDN
parity claim, full-model admission, or scientific result.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import os
import subprocess
import sys
import time
from typing import Final, TypeAlias, cast
from unittest.mock import patch

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.utils.data import DataLoader

from sana_wam.cach.action_conditioning import (
    reduce_end_of_bin_action_condition,
)
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_chunk_layout import (
    ChunkActionLayout,
    build_synthetic_chunk_action_layout_for_tests,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)
from sana_wam.model.cach_minimal_cpu import (
    MinimalCACHContractError,
    MinimalCACHPair,
    MinimalCACHSpec,
    MinimalCorrectedGDNArchitecture,
    MinimalLayerState,
    PureTorchChunkCausalGDNProxy,
    build_minimal_ref_cach_pair,
)
from sana_wam.model.video_backbone.sana.hybrid_cache import tensor_digest


JSONScalar: TypeAlias = str | int | float | bool | None

_SCHEMA: Final = "cach.authorized_cuda_synthetic_smoke.v1"
_EXPECTED_GPU_UUID: Final = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
_EXPECTED_CAPABILITY: Final = (9, 0)
_MEMORY_LIMIT_BYTES: Final = 3 * 1024**3
_NVIDIA_SMI_MEMORY_LIMIT_MIB: Final = 4096
_MINIMUM_FREE_MEMORY_MIB: Final = 8 * 1024
_SOFT_RUNTIME_LIMIT_SECONDS: Final = 270.0
_BATCH_SIZE: Final = 1
_VALID_LATENT_COUNT: Final = 6
_DATA_SEED: Final = 20260804


@dataclass(frozen=True)
class _TensorSnapshot:
    digest: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    is_leaf: bool
    object_id: int
    storage_data_ptr: int
    storage_nbytes: int
    tensor_nbytes: int
    storage_offset: int
    stride: tuple[int, ...]
    version: int
    grad_is_none: bool


@dataclass(frozen=True)
class _ArmRuntime:
    model: MinimalCorrectedGDNArchitecture
    parameters: Mapping[str, Tensor]
    block_runners: tuple[_FunctionalBlockRunner, ...]


@dataclass(frozen=True)
class _ArmResult:
    video_outputs: tuple[Tensor, ...]
    final_layers: tuple[MinimalLayerState, ...]

    @property
    def video_prediction(self) -> Tensor:
        return torch.cat(self.video_outputs, dim=1)


class _FunctionalBlockRunner(nn.Module):
    """Expose ``scan_chunk`` as a functional-call-compatible ``forward``."""

    def __init__(self, block: PureTorchChunkCausalGDNProxy) -> None:
        super().__init__()
        self.block = block

    def forward(
        self,
        hidden: Tensor,
        main_s_kv: Tensor,
        main_s_z: Tensor,
        frame_valid_mask: Tensor,
        action_embedding: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output, next_state = self.block.scan_chunk(
            hidden,
            MinimalLayerState(main_s_kv=main_s_kv, main_s_z=main_s_z),
            frame_valid_mask=frame_valid_mask,
            action_embedding=action_embedding,
        )
        return output, next_state.main_s_kv, next_state.main_s_z


def _fail(message: str) -> None:
    raise MinimalCACHContractError(f"CUDA synthetic smoke refused: {message}")


def _require_exact_environment() -> None:
    required = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_CACHE_DISABLE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": _EXPECTED_GPU_UUID,
        "GDN_DISABLE_COMPILE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TORCHDYNAMO_DISABLE": "1",
    }
    mismatches = {
        name: os.environ.get(name)
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        _fail(f"required environment mismatch: {mismatches}")
    singleton_values = {
        "LOCAL_WORLD_SIZE": "1",
        "WORLD_SIZE": "1",
    }
    zero_rank_values = {
        "LOCAL_RANK": "0",
        "RANK": "0",
    }
    for name, expected in singleton_values.items():
        if os.environ.get(name) not in {None, expected}:
            _fail(f"{name} must be unset or {expected}")
    for name, expected in zero_rank_values.items():
        if os.environ.get(name) not in {None, expected}:
            _fail(f"{name} must be unset or {expected}")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _fail("torch.distributed must remain uninitialized")


def _run_nvidia_smi(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("nvidia-smi", *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        _fail(f"nvidia-smi failed: {error}")
    return completed.stdout.strip()


def _target_gpu_inventory() -> tuple[str, int]:
    output = _run_nvidia_smi(
        "--query-gpu=uuid,name,memory.free",
        "--format=csv,noheader,nounits",
    )
    matches: list[tuple[str, int]] = []
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3 or fields[0] != _EXPECTED_GPU_UUID:
            continue
        try:
            free_memory_mib = int(fields[2])
        except ValueError:
            _fail(f"invalid nvidia-smi free-memory value: {fields[2]!r}")
        matches.append((fields[1], free_memory_mib))
    if len(matches) != 1:
        _fail("authorized GPU UUID was not found exactly once")
    name, free_memory_mib = matches[0]
    if "H200" not in name:
        _fail(f"authorized UUID is not an H200: {name!r}")
    if free_memory_mib < _MINIMUM_FREE_MEMORY_MIB:
        _fail(f"authorized GPU has insufficient free memory: {free_memory_mib} MiB")
    return name, free_memory_mib


def _target_compute_apps() -> tuple[tuple[int, int], ...]:
    output = _run_nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    applications: list[tuple[int, int]] = []
    if not output:
        return ()
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3 or fields[0] != _EXPECTED_GPU_UUID:
            continue
        try:
            applications.append((int(fields[1]), int(fields[2])))
        except ValueError:
            _fail(f"invalid nvidia-smi compute-app row: {line!r}")
    return tuple(sorted(applications))


def _require_no_target_compute_apps() -> None:
    applications = _target_compute_apps()
    if applications:
        _fail(f"authorized GPU already has compute applications: {applications}")


def _require_exclusive_current_process() -> int:
    current_pid = os.getpid()
    deadline = time.monotonic() + 3.0
    while True:
        applications = _target_compute_apps()
        foreign = tuple(item for item in applications if item[0] != current_pid)
        if foreign:
            _fail(f"foreign compute process appeared on authorized GPU: {foreign}")
        own = tuple(item for item in applications if item[0] == current_pid)
        if len(own) == 1:
            if own[0][1] >= _NVIDIA_SMI_MEMORY_LIMIT_MIB:
                _fail(f"nvidia-smi process memory exceeded limit: {own[0][1]} MiB")
            return own[0][1]
        if time.monotonic() >= deadline:
            _fail("current CUDA process did not appear exactly once in nvidia-smi")
        time.sleep(0.1)


def _assert_no_forbidden_acceleration_modules() -> None:
    forbidden_roots = ("triton", "torch._inductor")
    loaded = tuple(
        sorted(
            name
            for name in sys.modules
            if any(
                name == root or name.startswith(f"{root}.") for root in forbidden_roots
            )
        )
    )
    if loaded:
        _fail(f"forbidden acceleration modules were loaded: {loaded}")


@contextmanager
def _forbidden_operation_guard() -> Iterator[None]:
    def blocked(*_args: object, **_kwargs: object) -> None:
        _fail("a prohibited training/loading/compilation operation was attempted")

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor, "backward", blocked))
        stack.enter_context(patch.object(torch.autograd, "backward", blocked))
        stack.enter_context(patch.object(torch, "compile", blocked))
        stack.enter_context(patch.object(torch, "load", blocked))
        stack.enter_context(patch.object(torch, "save", blocked))
        stack.enter_context(patch.object(torch.jit, "load", blocked))
        stack.enter_context(patch.object(torch.jit, "script", blocked))
        stack.enter_context(patch.object(torch.jit, "trace", blocked))
        stack.enter_context(patch.object(nn.Module, "load_state_dict", blocked))
        stack.enter_context(patch.object(torch.optim.Optimizer, "__init__", blocked))
        stack.enter_context(patch.object(DataLoader, "__init__", blocked))
        yield


def _synthetic_layout() -> ChunkActionLayout:
    spec = synthetic_layout_spec(
        frame_chunk_size=2,
        temporal_compression=1,
        video_stride=1,
        action_dim=20,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=_VALID_LATENT_COUNT,
        video_stride=1,
        source_row_label="authorized-cuda-smoke-row",
        episode_label="authorized-cuda-smoke-episode",
    )
    return build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=_VALID_LATENT_COUNT,
        video_valid_mask=(True,) * _VALID_LATENT_COUNT,
        action_valid_mask=(True,) * (_VALID_LATENT_COUNT - 1),
    )


def _synthetic_pair(layout: ChunkActionLayout) -> MinimalCACHPair:
    pair = build_minimal_ref_cach_pair(
        spec=MinimalCACHSpec(
            latent_dim=6,
            hidden_dim=8,
            action_dim=20,
            depth=20,
            chunk_size=2,
            shared_init_seed=20260802,
            operator_init_seed=20260803,
        ),
        layout=layout,
    )
    pair.assert_integrity()
    return pair


def _named_registered_tensors(pair: MinimalCACHPair) -> dict[str, Tensor]:
    values: dict[str, Tensor] = {}
    for arm_name, model in (
        ("reference", pair.reference),
        ("candidate", pair.candidate),
    ):
        values.update(
            {
                f"{arm_name}.parameter.{name}": parameter
                for name, parameter in model.named_parameters()
            }
        )
        values.update(
            {
                f"{arm_name}.buffer.{name}": buffer
                for name, buffer in model.named_buffers()
            }
        )
    return values


def _tensor_snapshot(value: Tensor) -> _TensorSnapshot:
    storage = value.untyped_storage()
    return _TensorSnapshot(
        digest=tensor_digest(value),
        shape=tuple(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        requires_grad=value.requires_grad,
        is_leaf=value.is_leaf,
        object_id=id(value),
        storage_data_ptr=storage.data_ptr(),
        storage_nbytes=storage.nbytes(),
        tensor_nbytes=value.numel() * value.element_size(),
        storage_offset=value.storage_offset(),
        stride=tuple(value.stride()),
        version=value._version,
        grad_is_none=value.grad is None,
    )


def _snapshot(values: Mapping[str, Tensor]) -> dict[str, _TensorSnapshot]:
    return {name: _tensor_snapshot(values[name]) for name in sorted(values)}


def _assert_snapshot_exact(
    before: Mapping[str, _TensorSnapshot],
    after: Mapping[str, _TensorSnapshot],
    *,
    label: str,
) -> None:
    if before != after:
        changed = tuple(
            sorted(
                name
                for name in before.keys() | after.keys()
                if before.get(name) != after.get(name)
            )
        )
        _fail(f"{label} tensor snapshot changed: {changed}")


def _tensor_bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return (
        tuple(left.shape) == tuple(right.shape)
        and left.dtype is right.dtype
        and left.device == right.device
        and left.layout is right.layout
        and tensor_digest(left) == tensor_digest(right)
    )


def _cuda_parameter_shadows(
    model: MinimalCorrectedGDNArchitecture,
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    if tuple(model.named_buffers()):
        _fail("CUDA shadow runner does not admit registered buffers")
    shadows: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        shadow = parameter.detach().to(
            device=device,
            dtype=torch.float32,
            copy=True,
        )
        shadow.requires_grad_(parameter.requires_grad)
        if not shadow.is_leaf:
            _fail(f"CUDA parameter shadow is not a leaf: {name}")
        shadows[name] = shadow
    if not shadows:
        _fail("CUDA parameter shadow inventory is empty")
    return shadows


def _assert_shadow_pair(
    pair: MinimalCACHPair,
    reference: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
) -> None:
    reference_names = set(reference)
    candidate_names = set(candidate)
    expected_shared = {entry.name for entry in pair.shared_parameter_inventory}
    expected_candidate_only = {
        entry.name for entry in pair.candidate_only_parameter_inventory
    }
    if reference_names != expected_shared:
        _fail("reference CUDA inventory differs from canonical shared inventory")
    if candidate_names != expected_shared | expected_candidate_only:
        _fail("candidate CUDA inventory differs from canonical inventory")
    for name in sorted(expected_shared):
        if not _tensor_bitwise_equal(reference[name], candidate[name]):
            _fail(f"shared CUDA parameter bytes differ: {name}")
        if reference[name].untyped_storage().data_ptr() == (
            candidate[name].untyped_storage().data_ptr()
        ):
            _fail(f"shared CUDA parameter storage aliases across arms: {name}")
    reference_pointers = tuple(
        value.untyped_storage().data_ptr() for value in reference.values()
    )
    candidate_pointers = tuple(
        value.untyped_storage().data_ptr() for value in candidate.values()
    )
    if (
        len(reference_pointers) != len(set(reference_pointers))
        or len(candidate_pointers) != len(set(candidate_pointers))
        or not set(reference_pointers).isdisjoint(candidate_pointers)
    ):
        _fail("CUDA shadow storage topology is not arm-independent")
    zero_seam = tuple(
        name
        for name in sorted(expected_candidate_only)
        if name == "no_action_slot" or ".action_output_projection." in name
    )
    if len(zero_seam) != 41:
        _fail(f"expected 41 exact-zero seam tensors, found {len(zero_seam)}")
    for name in zero_seam:
        if bool(candidate[name].detach().count_nonzero()):
            _fail(f"candidate CUDA output seam is not exact zero: {name}")


def _runtime(
    model: MinimalCorrectedGDNArchitecture,
    parameters: Mapping[str, Tensor],
) -> _ArmRuntime:
    runners = tuple(_FunctionalBlockRunner(block) for block in model.blocks)
    if len(runners) != model.spec.depth:
        _fail("functional block-runner depth mismatch")
    return _ArmRuntime(
        model=model,
        parameters=parameters,
        block_runners=runners,
    )


def _subparameters(
    parameters: Mapping[str, Tensor],
    prefix: str,
    *,
    wrapper_prefix: str = "",
) -> dict[str, Tensor]:
    marker = f"{prefix}."
    selected = {
        f"{wrapper_prefix}{name[len(marker) :]}": value
        for name, value in parameters.items()
        if name.startswith(marker)
    }
    if not selected:
        _fail(f"functional parameter subset is empty: {prefix}")
    return selected


def _linear_module_call(
    runtime: _ArmRuntime,
    *,
    prefix: str,
    value: Tensor,
) -> Tensor:
    module = cast(nn.Module, getattr(runtime.model, prefix))
    output = functional_call(
        module,
        _subparameters(runtime.parameters, prefix),
        (value,),
        strict=True,
    )
    if not isinstance(output, Tensor):
        _fail(f"functional module returned a non-tensor: {prefix}")
    return output


def _action_embeddings(
    runtime: _ArmRuntime,
    *,
    actions: Sequence[Tensor],
    frame_valid_masks: Sequence[Tensor],
    layout: ChunkActionLayout,
) -> tuple[Tensor | None, ...]:
    if runtime.model.staging_variant is CACHStagingVariant.REF_GDN_CORRECTED:
        if runtime.model.action_conditioner is not None:
            _fail("reference action conditioner is not structurally absent")
        return (None,) * len(layout.chunks)
    if runtime.model.staging_variant is not CACHStagingVariant.CACH_A:
        _fail("unexpected staging variant")
    conditioner = runtime.model.action_conditioner
    if conditioner is None or "no_action_slot" not in runtime.parameters:
        _fail("candidate action-conditioning seam is incomplete")
    committed = torch.empty(
        _BATCH_SIZE,
        0,
        runtime.model.spec.action_dim,
        device=actions[0].device,
        dtype=torch.float32,
    )
    embeddings: list[Tensor] = []
    for chunk, noisy_actions, frame_valid_mask in zip(
        layout.chunks,
        actions,
        frame_valid_masks,
        strict=True,
    ):
        reduced = reduce_end_of_bin_action_condition(
            noisy_actions,
            committed_actions=None if committed.shape[1] == 0 else committed,
            chunk=chunk,
            no_action_slot=runtime.parameters["no_action_slot"],
        )
        if not torch.equal(reduced.latent_valid_mask, frame_valid_mask):
            _fail("CUDA reducer/layout frame masks differ")
        embedded = functional_call(
            conditioner,
            _subparameters(runtime.parameters, "action_conditioner"),
            (reduced.condition,),
            strict=True,
        )
        if not isinstance(embedded, Tensor):
            _fail("candidate action conditioner returned a non-tensor")
        embeddings.append(
            torch.where(
                frame_valid_mask.unsqueeze(-1),
                embedded,
                torch.zeros_like(embedded),
            )
        )
        committed = torch.cat(
            [committed, noisy_actions[:, : chunk.valid_action_count]],
            dim=1,
        )
    return tuple(embeddings)


def _video_embedding(
    runtime: _ArmRuntime,
    video: Tensor,
    frame_valid_mask: Tensor,
) -> Tensor:
    timestep = torch.zeros(
        _BATCH_SIZE,
        1,
        1,
        device=video.device,
        dtype=torch.float32,
    )
    hidden = _linear_module_call(
        runtime,
        prefix="video_input_projection",
        value=video,
    )
    hidden = hidden + _linear_module_call(
        runtime,
        prefix="video_timestep_projection",
        value=timestep,
    )
    return torch.where(
        frame_valid_mask.unsqueeze(-1),
        hidden,
        torch.zeros_like(hidden),
    )


def _empty_layer_state(runtime: _ArmRuntime, device: torch.device) -> MinimalLayerState:
    hidden_dim = runtime.model.spec.hidden_dim
    return MinimalLayerState(
        main_s_kv=torch.zeros(
            _BATCH_SIZE,
            hidden_dim,
            hidden_dim,
            device=device,
            dtype=torch.float32,
        ),
        main_s_z=torch.zeros(
            _BATCH_SIZE,
            hidden_dim,
            device=device,
            dtype=torch.float32,
        ),
    )


def _block_call(
    runtime: _ArmRuntime,
    *,
    layer_index: int,
    hidden: Tensor,
    state: MinimalLayerState,
    frame_valid_mask: Tensor,
    action_embedding: Tensor | None,
) -> tuple[Tensor, MinimalLayerState]:
    prefix = f"blocks.{layer_index}"
    output = functional_call(
        runtime.block_runners[layer_index],
        _subparameters(runtime.parameters, prefix, wrapper_prefix="block."),
        (
            hidden,
            state.main_s_kv,
            state.main_s_z,
            frame_valid_mask,
            action_embedding,
        ),
        strict=True,
    )
    if not isinstance(output, tuple) or len(output) != 3:
        _fail(f"functional block {layer_index} returned an invalid result")
    next_hidden, next_s_kv, next_s_z = output
    if not all(isinstance(value, Tensor) for value in output):
        _fail(f"functional block {layer_index} returned a non-tensor")
    return next_hidden, MinimalLayerState(next_s_kv, next_s_z)


def _video_prediction(
    runtime: _ArmRuntime,
    hidden: Tensor,
    frame_valid_mask: Tensor,
) -> Tensor:
    prediction = _linear_module_call(
        runtime,
        prefix="video_output_projection",
        value=hidden,
    )
    return torch.where(
        frame_valid_mask.unsqueeze(-1),
        prediction,
        torch.zeros_like(prediction),
    )


def _run_layer_major(
    runtime: _ArmRuntime,
    *,
    videos: Sequence[Tensor],
    actions: Sequence[Tensor],
    frame_valid_masks: Sequence[Tensor],
    layout: ChunkActionLayout,
) -> _ArmResult:
    action_embeddings = _action_embeddings(
        runtime,
        actions=actions,
        frame_valid_masks=frame_valid_masks,
        layout=layout,
    )
    hidden_chunks = [
        _video_embedding(runtime, video, frame_valid_mask)
        for video, frame_valid_mask in zip(
            videos,
            frame_valid_masks,
            strict=True,
        )
    ]
    final_layers: list[MinimalLayerState] = []
    for layer_index in range(runtime.model.spec.depth):
        layer_state = _empty_layer_state(runtime, videos[0].device)
        next_hidden_chunks: list[Tensor] = []
        for hidden, frame_valid_mask, action_embedding in zip(
            hidden_chunks,
            frame_valid_masks,
            action_embeddings,
            strict=True,
        ):
            hidden, layer_state = _block_call(
                runtime,
                layer_index=layer_index,
                hidden=hidden,
                state=layer_state,
                frame_valid_mask=frame_valid_mask,
                action_embedding=action_embedding,
            )
            next_hidden_chunks.append(hidden)
        hidden_chunks = next_hidden_chunks
        final_layers.append(layer_state)
    outputs = tuple(
        _video_prediction(runtime, hidden, frame_valid_mask)
        for hidden, frame_valid_mask in zip(
            hidden_chunks,
            frame_valid_masks,
            strict=True,
        )
    )
    return _ArmResult(
        video_outputs=outputs,
        final_layers=tuple(final_layers),
    )


def _run_chunk_major(
    runtime: _ArmRuntime,
    *,
    videos: Sequence[Tensor],
    actions: Sequence[Tensor],
    frame_valid_masks: Sequence[Tensor],
    layout: ChunkActionLayout,
) -> _ArmResult:
    action_embeddings = _action_embeddings(
        runtime,
        actions=actions,
        frame_valid_masks=frame_valid_masks,
        layout=layout,
    )
    layer_states = [
        _empty_layer_state(runtime, videos[0].device)
        for _ in range(runtime.model.spec.depth)
    ]
    outputs: list[Tensor] = []
    for video, frame_valid_mask, action_embedding in zip(
        videos,
        frame_valid_masks,
        action_embeddings,
        strict=True,
    ):
        hidden = _video_embedding(runtime, video, frame_valid_mask)
        for layer_index, state in enumerate(layer_states):
            hidden, next_state = _block_call(
                runtime,
                layer_index=layer_index,
                hidden=hidden,
                state=state,
                frame_valid_mask=frame_valid_mask,
                action_embedding=action_embedding,
            )
            layer_states[layer_index] = next_state
        outputs.append(_video_prediction(runtime, hidden, frame_valid_mask))
    return _ArmResult(
        video_outputs=tuple(outputs),
        final_layers=tuple(layer_states),
    )


def _assert_result_exact(left: _ArmResult, right: _ArmResult, *, label: str) -> None:
    if len(left.video_outputs) != len(right.video_outputs):
        _fail(f"{label} output count differs")
    for index, (left_output, right_output) in enumerate(
        zip(left.video_outputs, right.video_outputs, strict=True)
    ):
        if not _tensor_bitwise_equal(left_output, right_output):
            _fail(f"{label} video output differs at chunk {index}")
    if len(left.final_layers) != len(right.final_layers):
        _fail(f"{label} layer-state count differs")
    for index, (left_state, right_state) in enumerate(
        zip(left.final_layers, right.final_layers, strict=True)
    ):
        if not _tensor_bitwise_equal(left_state.main_s_kv, right_state.main_s_kv):
            _fail(f"{label} S_kv differs at layer {index}")
        if not _tensor_bitwise_equal(left_state.main_s_z, right_state.main_s_z):
            _fail(f"{label} S_z differs at layer {index}")


def _assert_result_fp32_cuda(result: _ArmResult, *, label: str) -> None:
    tensors = list(result.video_outputs)
    tensors.extend(
        tensor
        for layer in result.final_layers
        for tensor in (layer.main_s_kv, layer.main_s_z)
    )
    for index, value in enumerate(tensors):
        if value.device.type != "cuda" or value.device.index != 0:
            _fail(f"{label} tensor {index} escaped logical cuda:0")
        if value.dtype is not torch.float32:
            _fail(f"{label} tensor {index} escaped FP32")
        if not bool(torch.isfinite(value.detach()).all()):
            _fail(f"{label} tensor {index} is non-finite")


def _synthetic_inputs(
    layout: ChunkActionLayout,
    *,
    device: torch.device,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    generator = torch.Generator(device=device).manual_seed(_DATA_SEED)
    videos: list[Tensor] = []
    actions: list[Tensor] = []
    masks: list[Tensor] = []
    for chunk in layout.chunks:
        frame_valid_mask = (
            torch.tensor(
                chunk.latent_valid_mask,
                device=device,
                dtype=torch.bool,
            )
            .view(1, -1)
            .clone()
        )
        video = torch.zeros(
            _BATCH_SIZE,
            layout.frame_chunk_size,
            6,
            device=device,
            dtype=torch.float32,
        )
        video[:, : chunk.valid_latent_count] = torch.randn(
            _BATCH_SIZE,
            chunk.valid_latent_count,
            6,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        noisy_actions = torch.zeros(
            _BATCH_SIZE,
            chunk.action_slot_capacity,
            20,
            device=device,
            dtype=torch.float32,
        )
        noisy_actions[:, : chunk.valid_action_count] = torch.randn(
            _BATCH_SIZE,
            chunk.valid_action_count,
            20,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        videos.append(video)
        actions.append(noisy_actions)
        masks.append(frame_valid_mask)
    return tuple(videos), tuple(actions), tuple(masks)


def _perturb_valid_actions(
    actions: Sequence[Tensor],
    layout: ChunkActionLayout,
    *,
    chunk_index: int = 1,
    amount: float = 3.0,
) -> tuple[Tensor, ...]:
    changed = [value.detach().clone() for value in actions]
    valid_count = layout.chunks[chunk_index].valid_action_count
    changed[chunk_index][:, :valid_count] += amount
    return tuple(changed)


def _perturb_future_video(videos: Sequence[Tensor]) -> tuple[Tensor, ...]:
    changed = [value.detach().clone() for value in videos]
    changed[-1] = changed[-1] + 7.0
    return tuple(changed)


def _adapter_inventory(
    runtime: _ArmRuntime,
) -> tuple[tuple[str, ...], tuple[Tensor, ...]]:
    names = tuple(
        name
        for name in sorted(runtime.parameters)
        if ".action_output_projection." in name
    )
    if len(names) != 40:
        _fail(f"expected 40 adapter tensors, found {len(names)}")
    return names, tuple(runtime.parameters[name] for name in names)


def _result_probe(
    result: _ArmResult,
    *,
    output_indices: Sequence[int],
) -> Tensor:
    selected = tuple(result.video_outputs[index] for index in output_indices)
    if not selected:
        _fail("adapter gradient probe selected no video outputs")
    video = torch.cat(selected, dim=1)
    weights = torch.linspace(
        -1.0,
        1.0,
        video.numel(),
        device=video.device,
        dtype=torch.float32,
    ).reshape_as(video)
    return (video * weights).sum() + 0.125 * video.square().sum()


def _adapter_gradient_tuple(
    runtime: _ArmRuntime,
    result: _ArmResult,
    *,
    output_indices: Sequence[int],
    retain_graph: bool,
) -> tuple[tuple[str, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    adapter_names, adapter_parameters = _adapter_inventory(runtime)
    gradients = torch.autograd.grad(
        _result_probe(result, output_indices=output_indices),
        adapter_parameters,
        allow_unused=False,
        materialize_grads=False,
        retain_graph=retain_graph,
    )
    for name, parameter, gradient in zip(
        adapter_names,
        adapter_parameters,
        gradients,
        strict=True,
    ):
        if gradient.shape != parameter.shape:
            _fail(f"adapter gradient shape differs from parameter: {name}")
        if gradient.device != parameter.device or gradient.device != torch.device(
            "cuda", 0
        ):
            _fail(f"adapter gradient escaped logical cuda:0: {name}")
        if gradient.dtype is not torch.float32:
            _fail(f"adapter gradient escaped FP32: {name}")
        if not bool(torch.isfinite(gradient).all()):
            _fail(f"adapter gradient is non-finite: {name}")
        if bool(parameter.detach().count_nonzero()):
            _fail(f"zero adapter parameter changed: {name}")
        if parameter.grad is not None:
            _fail(f"autograd.grad populated .grad: {name}")
    return adapter_names, adapter_parameters, gradients


def _future_action_gradient_separation(
    runtime: _ArmRuntime,
    *,
    videos: Sequence[Tensor],
    actions: Sequence[Tensor],
    frame_valid_masks: Sequence[Tensor],
    layout: ChunkActionLayout,
) -> _ArmResult:
    future_actions = _perturb_valid_actions(
        actions,
        layout,
        chunk_index=len(layout.chunks) - 1,
        amount=-5.0,
    )
    baseline = _run_layer_major(
        runtime,
        videos=videos,
        actions=actions,
        frame_valid_masks=frame_valid_masks,
        layout=layout,
    )
    comparison = _run_layer_major(
        runtime,
        videos=videos,
        actions=future_actions,
        frame_valid_masks=frame_valid_masks,
        layout=layout,
    )
    _assert_result_exact(
        baseline,
        comparison,
        label="zero candidate future-action output identity",
    )
    prefix_indices = tuple(range(len(layout.chunks) - 1))
    _, _, baseline_prefix_gradients = _adapter_gradient_tuple(
        runtime,
        baseline,
        output_indices=prefix_indices,
        retain_graph=True,
    )
    _, _, comparison_prefix_gradients = _adapter_gradient_tuple(
        runtime,
        comparison,
        output_indices=prefix_indices,
        retain_graph=True,
    )
    for index, (baseline_gradient, comparison_gradient) in enumerate(
        zip(
            baseline_prefix_gradients,
            comparison_prefix_gradients,
            strict=True,
        )
    ):
        if not _tensor_bitwise_equal(baseline_gradient, comparison_gradient):
            _fail(f"future action changed prefix adapter gradient {index}")

    future_index = (len(layout.chunks) - 1,)
    _, _, baseline_future_gradients = _adapter_gradient_tuple(
        runtime,
        baseline,
        output_indices=future_index,
        retain_graph=True,
    )
    _, _, comparison_future_gradients = _adapter_gradient_tuple(
        runtime,
        comparison,
        output_indices=future_index,
        retain_graph=False,
    )
    if all(
        _tensor_bitwise_equal(baseline_gradient, comparison_gradient)
        for baseline_gradient, comparison_gradient in zip(
            baseline_future_gradients,
            comparison_future_gradients,
            strict=True,
        )
    ):
        _fail("future action did not change any future adapter gradient")
    return baseline


def _adapter_gradients(
    runtime: _ArmRuntime,
    result: _ArmResult,
) -> tuple[Tensor, ...]:
    _, _, gradients = _adapter_gradient_tuple(
        runtime,
        result,
        output_indices=tuple(range(len(result.video_outputs))),
        retain_graph=False,
    )
    for index, gradient in enumerate(gradients):
        if not bool(gradient.count_nonzero()):
            _fail(f"adapter gradient is exact zero: tensor {index}")
    return gradients


def run_authorized_cach_minimal_cuda_smoke() -> dict[str, JSONScalar]:
    """Run the one fixed, authorized H200 synthetic-mini CUDA smoke.

    The function deliberately accepts no arguments.  It fail-closes unless the
    exact UUID, singleton-device environment, deterministic FP32 mode, empty
    target GPU, memory limit, and update-free tensor snapshots all hold.
    """

    started_at = time.monotonic()
    checks: list[str] = []
    _require_exact_environment()
    _assert_no_forbidden_acceleration_modules()
    gpu_name, preflight_free_memory_mib = _target_gpu_inventory()
    _require_no_target_compute_apps()
    checks.append("preflight")

    layout = _synthetic_layout()
    pair = _synthetic_pair(layout)
    cpu_tensors = _named_registered_tensors(pair)
    cpu_before = _snapshot(cpu_tensors)
    pair_digest = pair.pair_digest
    checks.append("canonical_cpu_pair")

    if torch.cuda.device_count() != 1:
        _fail(f"expected one logical CUDA device, found {torch.cuda.device_count()}")
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    capability = (properties.major, properties.minor)
    if capability != _EXPECTED_CAPABILITY:
        _fail(f"unexpected CUDA capability: {capability}")
    if "H200" not in properties.name:
        _fail(f"logical cuda:0 is not an H200: {properties.name!r}")
    torch.cuda.set_per_process_memory_fraction(
        _MEMORY_LIMIT_BYTES / properties.total_memory,
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    cuda_context_sentinel = torch.empty(1, device=device, dtype=torch.float32)
    torch.cuda.synchronize(device)
    del cuda_context_sentinel
    sampled_process_memory_mib = _require_exclusive_current_process()
    checks.append("singleton_cuda")

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_deterministic_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with (
            _forbidden_operation_guard(),
            torch.autocast(
                device_type="cuda",
                enabled=False,
            ),
        ):
            reference_parameters = _cuda_parameter_shadows(
                pair.reference,
                device=device,
            )
            candidate_parameters = _cuda_parameter_shadows(
                pair.candidate,
                device=device,
            )
            _assert_shadow_pair(pair, reference_parameters, candidate_parameters)
            reference_runtime = _runtime(pair.reference, reference_parameters)
            candidate_runtime = _runtime(pair.candidate, candidate_parameters)
            cuda_tensors = {
                **{
                    f"reference.{name}": value
                    for name, value in reference_parameters.items()
                },
                **{
                    f"candidate.{name}": value
                    for name, value in candidate_parameters.items()
                },
            }
            cuda_before = _snapshot(cuda_tensors)
            checks.append("independent_cuda_shadows")

            videos, actions, frame_valid_masks = _synthetic_inputs(
                layout,
                device=device,
            )
            input_tensors = {
                **{f"video.{index}": value for index, value in enumerate(videos)},
                **{f"action.{index}": value for index, value in enumerate(actions)},
                **{
                    f"frame_valid_mask.{index}": value
                    for index, value in enumerate(frame_valid_masks)
                },
            }
            inputs_before = _snapshot(input_tensors)
            with torch.no_grad():
                reference_full = _run_layer_major(
                    reference_runtime,
                    videos=videos,
                    actions=actions,
                    frame_valid_masks=frame_valid_masks,
                    layout=layout,
                )
                reference_chunk = _run_chunk_major(
                    reference_runtime,
                    videos=videos,
                    actions=actions,
                    frame_valid_masks=frame_valid_masks,
                    layout=layout,
                )
                _assert_result_exact(
                    reference_full,
                    reference_chunk,
                    label="reference full/chunk",
                )
                checks.append("reference_full_chunk_exact")
                sampled_process_memory_mib = max(
                    sampled_process_memory_mib,
                    _require_exclusive_current_process(),
                )

                candidate_full = _run_layer_major(
                    candidate_runtime,
                    videos=videos,
                    actions=actions,
                    frame_valid_masks=frame_valid_masks,
                    layout=layout,
                )
                candidate_chunk = _run_chunk_major(
                    candidate_runtime,
                    videos=videos,
                    actions=actions,
                    frame_valid_masks=frame_valid_masks,
                    layout=layout,
                )
                _assert_result_exact(
                    candidate_full,
                    candidate_chunk,
                    label="candidate full/chunk",
                )
                checks.append("candidate_full_chunk_exact")

                _assert_result_exact(
                    reference_full,
                    candidate_full,
                    label="zero candidate/reference",
                )
                checks.append("zero_identity_exact")
                _assert_result_fp32_cuda(reference_full, label="reference")
                _assert_result_fp32_cuda(candidate_full, label="candidate")
                checks.append("fp32_cuda_finite")
                sampled_process_memory_mib = max(
                    sampled_process_memory_mib,
                    _require_exclusive_current_process(),
                )

                changed_actions = _perturb_valid_actions(actions, layout)
                reference_changed_actions = _run_layer_major(
                    reference_runtime,
                    videos=videos,
                    actions=changed_actions,
                    frame_valid_masks=frame_valid_masks,
                    layout=layout,
                )
                candidate_changed_actions = _run_layer_major(
                    candidate_runtime,
                    videos=videos,
                    actions=changed_actions,
                    frame_valid_masks=frame_valid_masks,
                    layout=layout,
                )
                _assert_result_exact(
                    reference_full,
                    reference_changed_actions,
                    label="reference action bypass",
                )
                _assert_result_exact(
                    candidate_full,
                    candidate_changed_actions,
                    label="zero candidate action identity",
                )
                checks.append("action_bypass_and_zero_seam")
                sampled_process_memory_mib = max(
                    sampled_process_memory_mib,
                    _require_exclusive_current_process(),
                )

                future_videos = _perturb_future_video(videos)
                candidate_future = _run_layer_major(
                    candidate_runtime,
                    videos=future_videos,
                    actions=actions,
                    frame_valid_masks=frame_valid_masks,
                    layout=layout,
                )
                for index in range(len(candidate_full.video_outputs) - 1):
                    if not _tensor_bitwise_equal(
                        candidate_full.video_outputs[index],
                        candidate_future.video_outputs[index],
                    ):
                        _fail(f"future video perturbation changed prefix chunk {index}")
                if _tensor_bitwise_equal(
                    candidate_full.video_outputs[-1],
                    candidate_future.video_outputs[-1],
                ):
                    _fail("future video perturbation did not change its own chunk")
                checks.append("future_video_prefix_separation")
                sampled_process_memory_mib = max(
                    sampled_process_memory_mib,
                    _require_exclusive_current_process(),
                )

            candidate_gradient_result = _future_action_gradient_separation(
                candidate_runtime,
                videos=videos,
                actions=actions,
                frame_valid_masks=frame_valid_masks,
                layout=layout,
            )
            checks.append("future_action_gradient_separation")
            gradients = _adapter_gradients(
                candidate_runtime,
                candidate_gradient_result,
            )
            checks.append("zero_adapter_autograd_wiring")
            sampled_process_memory_mib = max(
                sampled_process_memory_mib,
                _require_exclusive_current_process(),
            )

            torch.cuda.synchronize(device)
            _assert_snapshot_exact(
                cuda_before,
                _snapshot(cuda_tensors),
                label="CUDA shadow",
            )
            checks.append("cuda_update_free_snapshot")
            _assert_snapshot_exact(
                inputs_before,
                _snapshot(input_tensors),
                label="synthetic input",
            )
            checks.append("synthetic_inputs_read_only")
            pair.assert_integrity()
            _assert_snapshot_exact(
                cpu_before,
                _snapshot(cpu_tensors),
                label="canonical CPU pair",
            )
            if pair.pair_digest != pair_digest:
                _fail("canonical CPU pair digest changed")
            checks.append("cpu_pair_untouched")
            _assert_no_forbidden_acceleration_modules()
            checks.append("no_compile_jit_triton")

            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            if peak_allocated > _MEMORY_LIMIT_BYTES:
                _fail(f"peak allocated bytes exceeded limit: {peak_allocated}")
            if peak_reserved > _MEMORY_LIMIT_BYTES:
                _fail(f"peak reserved bytes exceeded limit: {peak_reserved}")
            process_memory_mib = _require_exclusive_current_process()
            sampled_process_memory_mib = max(
                sampled_process_memory_mib,
                process_memory_mib,
            )
            checks.append("resource_limits")
            elapsed_seconds = time.monotonic() - started_at
            if elapsed_seconds >= _SOFT_RUNTIME_LIMIT_SECONDS:
                _fail(f"soft runtime limit exceeded: {elapsed_seconds:.3f}s")
            checks.append("runtime_limit")

            reference_prediction_sha256 = tensor_digest(reference_full.video_prediction)
            candidate_prediction_sha256 = tensor_digest(candidate_full.video_prediction)
            if reference_prediction_sha256 != candidate_prediction_sha256:
                _fail("reference/candidate raw prediction digests differ")

            return {
                "adapter_gradient_tensor_count": len(gradients),
                "batch_size": _BATCH_SIZE,
                "candidate_only_parameter_count": len(
                    pair.candidate_only_parameter_inventory
                ),
                "candidate_prediction_sha256": candidate_prediction_sha256,
                "checks_passed": len(checks),
                "checkpoint_loaded": False,
                "chunk_count": len(layout.chunks),
                "chunk_size": pair.candidate.spec.chunk_size,
                "complete_2b": False,
                "cuda_shadow_parameter_count": len(cuda_tensors),
                "depth": pair.candidate.spec.depth,
                "dtype": "float32",
                "elapsed_seconds": round(elapsed_seconds, 6),
                "formal_admission": False,
                "gpu_capability": (f"{properties.major}.{properties.minor}"),
                "gpu_name": gpu_name,
                "gpu_uuid": _EXPECTED_GPU_UUID,
                "hidden_dim": pair.candidate.spec.hidden_dim,
                "latent_dim": pair.candidate.spec.latent_dim,
                "logical_cuda_device_count": torch.cuda.device_count(),
                "max_memory_allocated_bytes": peak_allocated,
                "max_memory_reserved_bytes": peak_reserved,
                "nvidia_smi_process_memory_mib": process_memory_mib,
                "nvidia_smi_max_sampled_process_memory_mib": (
                    sampled_process_memory_mib
                ),
                "pair_digest": pair_digest,
                "parameter_update_count": 0,
                "preflight_free_memory_mib": preflight_free_memory_mib,
                "real_data_loaded": False,
                "reference_prediction_sha256": reference_prediction_sha256,
                "schema": _SCHEMA,
                "scientific_result": False,
                "smoke_scope": "video_conditioning_seam_only",
                "status": "passed",
                "training_step_count": 0,
                "update_free": True,
                "vendor_gdn_numerical_parity": False,
            }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_deterministic_warn_only,
        )


__all__ = ["run_authorized_cach_minimal_cuda_smoke"]
