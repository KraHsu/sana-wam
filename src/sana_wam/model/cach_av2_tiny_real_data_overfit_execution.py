"""Additive AV-2 tiny real-data execution core for frozen CACH-A4.

This module binds the already-frozen A4 vendor bridge to one deliberately
small RoboTwin slice.  It does not create run roots, write results, load or
save checkpoints, or select a checkpoint.  Dataset access is explicit and
all model forward tasks contain NaN sentinels in their legacy target fields;
the real target stays in the execution batch and is passed only to the loss.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from types import MappingProxyType
from typing import Any, Final, Literal

import torch
from torch import Tensor, nn

from sana_wam.dataloader.transforms.rotation import quat_xyzw_to_rotation_6d
from sana_wam.model import cach_av1b_a4_state_conditioned_causal_odd_stream_r2 as a4
from sana_wam.model import cach_av2_tiny_real_data_overfit as av2_scaffold


AV2_ARCHITECTURE_ID: Final = a4.CACH_A4_ARCHITECTURE_ID
AV2_EXECUTION_SCHEMA: Final = (
    "cach.cach_a4.av2_tiny_real_data_overfit.execution.v1"
)
AV2_EXECUTION_RESULT_SCHEMA: Final = (
    "cach.cach_a4.av2_tiny_real_data_overfit.execution_result.v1"
)
AV2_TASK_NAME: Final = "adjust_bottle"
AV2_VARIANT: Final = "aloha-agilex_clean_50"
AV2_CAMERA: Final = "head_camera"
AV2_TRAIN_EPISODES: Final = tuple(range(0, 8))
AV2_HELDOUT_EPISODES: Final = tuple(range(8, 16))
AV2_RAW_START: Final = 0
AV2_RAW_END_EXCLUSIVE: Final = 33
AV2_ACTION_START: Final = 1
AV2_ACTION_END_EXCLUSIVE: Final = 33
AV2_FRAME_INDICES: Final = (0, 8, 16, 24, 32)
AV2_SHUFFLE_PERMUTATION: Final = (1, 0, 3, 2, 5, 4, 7, 6)
AV2_SEED: Final = 2026080601
AV2_STEPS_PER_ARM: Final = 1000
AV2_MAX_WALL_CLOCK_MINUTES: Final = 60
AV2_LR_START: Final = 3.0e-3
AV2_LR_END: Final = 3.0e-5
AV2_GRAD_CLIP_NORM: Final = 1.0

_DEFAULT_THRESHOLDS: Final = MappingProxyType(
    {
        "candidate_heldout_decrease_min": 0.05,
        "candidate_vs_reference_noninferiority_ratio_max": 1.0,
        "correct_vs_no_action_improvement_min": 0.05,
        "correct_vs_shuffle_improvement_min": 0.05,
        "min_action_variance": 1.0e-8,
        "min_motion_mse": 1.0e-8,
        "min_shuffle_mse": 1.0e-8,
    }
)


class AV2ExecutionError(RuntimeError):
    """The AV-2 execution contract or runtime validity check failed."""


@dataclass(frozen=True)
class AV2RealSplit:
    """One fixed eight-window split; target never becomes a model input."""

    name: Literal["train", "heldout"]
    windows: tuple[av2_scaffold.WindowIdentity, ...]
    actions: Tensor = field(repr=False, compare=False)
    noisy_video: Tensor = field(repr=False, compare=False)
    target: Tensor = field(repr=False, compare=False)
    proprio: Tensor = field(repr=False, compare=False)
    action_valid_mask: Tensor = field(repr=False, compare=False)
    frame_valid_mask: Tensor = field(repr=False, compare=False)
    episode_file_sha256: Mapping[str, str]
    tensor_sha256: Mapping[str, str]


@dataclass(frozen=True)
class AV2RealDataBundle:
    dataset_root: str
    train: AV2RealSplit
    heldout: AV2RealSplit
    selection_sha256: str


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(detached.view(torch.uint8).numpy().tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_cpu_float32(value: Tensor, *, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise AV2ExecutionError(f"{name} must have shape {shape}")
    if value.device.type != "cpu" or value.dtype is not torch.float32:
        raise AV2ExecutionError(f"{name} must be CPU float32")
    if not bool(torch.isfinite(value).all().item()):
        raise AV2ExecutionError(f"{name} must be finite")


def _validate_real_split(split: AV2RealSplit) -> None:
    _require_cpu_float32(split.actions, name="actions", shape=(8, 32, 20))
    _require_cpu_float32(
        split.noisy_video, name="noisy_video", shape=(8, 5, 3, 1, 1)
    )
    _require_cpu_float32(split.target, name="target", shape=(8, 5, 3, 1, 1))
    _require_cpu_float32(split.proprio, name="proprio", shape=(8, 2, 20))
    if (
        tuple(split.action_valid_mask.shape) != (8, 32)
        or split.action_valid_mask.device.type != "cpu"
        or split.action_valid_mask.dtype is not torch.bool
        or not bool(split.action_valid_mask.all().item())
    ):
        raise AV2ExecutionError("action_valid_mask must be all-true CPU bool [8,32]")
    if (
        tuple(split.frame_valid_mask.shape) != (8, 5)
        or split.frame_valid_mask.device.type != "cpu"
        or split.frame_valid_mask.dtype is not torch.bool
        or not bool(split.frame_valid_mask.all().item())
    ):
        raise AV2ExecutionError("frame_valid_mask must be all-true CPU bool [8,5]")
    if not torch.equal(split.noisy_video, split.noisy_video[:, :1].expand_as(split.noisy_video)):
        raise AV2ExecutionError("noisy_video must repeat the absolute frame-0 mean")
    if bool(split.target[:, 0].count_nonzero().item()):
        raise AV2ExecutionError("relative target frame 0 must be exact zero")
    if not torch.equal(split.proprio[:, 0], split.proprio[:, 1]):
        raise AV2ExecutionError("row-0 proprio must be repeated over both chunks")


def _episode_path(dataset_root: Path, episode_id: int) -> Path:
    return (
        dataset_root
        / AV2_TASK_NAME
        / AV2_VARIANT
        / "data"
        / f"episode{episode_id}.hdf5"
    )


def _eef20(handle: Any, start: int, end: int) -> Any:
    """Read absolute RoboTwin EEF vectors without normalization."""

    import numpy as np

    left_pose = np.asarray(handle["endpose/left_endpose"][start:end])
    right_pose = np.asarray(handle["endpose/right_endpose"][start:end])
    left_grip = np.asarray(
        handle["endpose/left_gripper"][start:end], dtype=np.float64
    ).reshape(end - start, -1)
    right_grip = np.asarray(
        handle["endpose/right_gripper"][start:end], dtype=np.float64
    ).reshape(end - start, -1)
    if (
        left_pose.shape != (end - start, 7)
        or right_pose.shape != (end - start, 7)
        or left_grip.shape != (end - start, 1)
        or right_grip.shape != (end - start, 1)
    ):
        raise AV2ExecutionError("RoboTwin endpose/gripper shape differs")
    left = np.concatenate(
        (
            left_pose[:, :3],
            quat_xyzw_to_rotation_6d(left_pose[:, 3:]),
            left_grip,
        ),
        axis=-1,
    )
    right = np.concatenate(
        (
            right_pose[:, :3],
            quat_xyzw_to_rotation_6d(right_pose[:, 3:]),
            right_grip,
        ),
        axis=-1,
    )
    result = np.concatenate((left, right), axis=-1).astype(np.float32)
    if result.shape != (end - start, 20) or not np.isfinite(result).all():
        raise AV2ExecutionError("absolute EEF20 extraction failed")
    return result


def _decode_channel_mean(encoded: Any) -> Tensor:
    """Decode JPEG with OpenCV and retain its native BGR channel ordering."""

    import cv2
    import numpy as np

    if isinstance(encoded, (bytes, bytearray, memoryview)):
        byte_array = np.frombuffer(encoded, dtype=np.uint8)
    else:
        byte_array = np.asarray(encoded, dtype=np.uint8).reshape(-1)
    decoded = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise AV2ExecutionError("cv2.imdecode failed for head_camera JPEG")
    # No cvtColor is permitted: the BGR bytes are the frozen observation basis.
    # Keep the frozen adapter's float32 accumulation/order: uint8 -> float32,
    # spatial mean in float32, then divide by 255 in float32.
    mean = decoded.astype(np.float32).mean(axis=(0, 1)) / np.float32(255.0)
    if not np.isfinite(mean).all():
        raise AV2ExecutionError("decoded channel mean is non-finite")
    return torch.from_numpy(mean).view(3, 1, 1)


def _load_split(
    dataset_root: Path,
    *,
    name: Literal["train", "heldout"],
    episode_ids: Sequence[int],
) -> AV2RealSplit:
    import h5py

    actions: list[Tensor] = []
    noisy: list[Tensor] = []
    targets: list[Tensor] = []
    proprio: list[Tensor] = []
    windows: list[av2_scaffold.WindowIdentity] = []
    source_sha: dict[str, str] = {}
    for episode_id in episode_ids:
        path = _episode_path(dataset_root, episode_id)
        if not path.is_file():
            raise AV2ExecutionError(f"required episode is absent: {path}")
        source_sha[str(path)] = _file_sha256(path)
        with h5py.File(path, "r") as handle:
            rgb_key = f"observation/{AV2_CAMERA}/rgb"
            required = (
                rgb_key,
                "endpose/left_endpose",
                "endpose/right_endpose",
                "endpose/left_gripper",
                "endpose/right_gripper",
            )
            if any(key not in handle for key in required):
                raise AV2ExecutionError(f"required HDF5 key missing in {path}")
            if any(handle[key].shape[0] < AV2_RAW_END_EXCLUSIVE for key in required):
                raise AV2ExecutionError(f"episode has fewer than 33 rows: {path}")

            absolute_eef = torch.from_numpy(
                _eef20(handle, AV2_RAW_START, AV2_RAW_END_EXCLUSIVE)
            )
            actions.append(absolute_eef[AV2_ACTION_START:AV2_ACTION_END_EXCLUSIVE])
            row0 = absolute_eef[0].clone()
            proprio.append(row0.view(1, 20).repeat(2, 1))

            absolute_video = torch.stack(
                [_decode_channel_mean(handle[rgb_key][index]) for index in AV2_FRAME_INDICES]
            )
            noisy.append(absolute_video[:1].repeat(5, 1, 1, 1))
            targets.append(absolute_video - absolute_video[:1])
        windows.append(
            av2_scaffold.WindowIdentity(
                task_name=AV2_TASK_NAME,
                episode_id=str(episode_id),
                raw_start=AV2_RAW_START,
                raw_end_exclusive=AV2_RAW_END_EXCLUSIVE,
            )
        )

    split = AV2RealSplit(
        name=name,
        windows=tuple(windows),
        actions=torch.stack(actions).contiguous(),
        noisy_video=torch.stack(noisy).contiguous(),
        target=torch.stack(targets).contiguous(),
        proprio=torch.stack(proprio).contiguous(),
        action_valid_mask=torch.ones((8, 32), dtype=torch.bool),
        frame_valid_mask=torch.ones((8, 5), dtype=torch.bool),
        episode_file_sha256=MappingProxyType(source_sha),
        tensor_sha256=MappingProxyType({}),
    )
    tensor_sha = {
        "actions": _tensor_sha256(split.actions),
        "noisy_video": _tensor_sha256(split.noisy_video),
        "proprio": _tensor_sha256(split.proprio),
        "target": _tensor_sha256(split.target),
    }
    split = replace(split, tensor_sha256=MappingProxyType(tensor_sha))
    _validate_real_split(split)
    return split


def load_av2_real_data(dataset_root: str | os.PathLike[str]) -> AV2RealDataBundle:
    """Load exactly episodes 0..7 for train and 8..15 for held-out.

    The call is read-only.  It performs no fallback selection, augmentation,
    random crop, normalization, color conversion, or data-dependent filtering.
    """

    root = Path(dataset_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise AV2ExecutionError("dataset_root must resolve to a directory")
    train = _load_split(root, name="train", episode_ids=AV2_TRAIN_EPISODES)
    heldout = _load_split(root, name="heldout", episode_ids=AV2_HELDOUT_EPISODES)
    av2_scaffold.assert_train_holdout_disjoint(train.windows, heldout.windows)
    selection = {
        "action_rows": [AV2_ACTION_START, AV2_ACTION_END_EXCLUSIVE],
        "camera": AV2_CAMERA,
        "frame_indices": list(AV2_FRAME_INDICES),
        "heldout_episode_ids": list(AV2_HELDOUT_EPISODES),
        "normalize": None,
        "raw_interval": [AV2_RAW_START, AV2_RAW_END_EXCLUSIVE],
        "target": "channel_mean_div_255_relative_to_frame0",
        "task": AV2_TASK_NAME,
        "train_episode_ids": list(AV2_TRAIN_EPISODES),
        "variant": AV2_VARIANT,
    }
    return AV2RealDataBundle(
        dataset_root=str(root),
        train=train,
        heldout=heldout,
        selection_sha256=_canonical_sha256(selection),
    )


def _thresholds(value: Mapping[str, object] | None) -> Mapping[str, float]:
    supplied = {} if value is None else dict(value)
    result: dict[str, float] = {}
    for name, default in _DEFAULT_THRESHOLDS.items():
        raw = supplied.get(name, default)
        if type(raw) not in (int, float) or not math.isfinite(float(raw)):
            raise AV2ExecutionError(f"threshold {name} must be finite")
        result[name] = float(raw)
    return MappingProxyType(result)


def assess_data_adequacy(
    bundle: AV2RealDataBundle,
    thresholds: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Apply the frozen positive-variance/motion/shuffle predicates."""

    resolved = _thresholds(thresholds)
    reports: dict[str, object] = {}
    for split in (bundle.train, bundle.heldout):
        shuffled = split.actions.index_select(
            0, torch.tensor(AV2_SHUFFLE_PERMUTATION, dtype=torch.long)
        )
        report = av2_scaffold.evaluate_data_adequacy(
            actions=split.actions,
            action_valid_mask=split.action_valid_mask,
            target=split.target,
            frame_valid_mask=split.frame_valid_mask,
            shuffled_actions=shuffled,
            min_action_variance=resolved["min_action_variance"],
            min_shuffle_mse=resolved["min_shuffle_mse"],
            min_motion_mse=resolved["min_motion_mse"],
        )
        reports[split.name] = {
            "action_mask_nonempty": report.action_mask_nonempty,
            "frame_mask_nonempty": report.frame_mask_nonempty,
            "action_variance": report.action_variance,
            "shuffle_mse": report.shuffle_mse,
            "motion_mse": report.motion_mse,
            "finite": report.finite,
            "adequate": report.adequate,
        }
    reports["all_adequate"] = bool(
        reports["train"]["adequate"] and reports["heldout"]["adequate"]  # type: ignore[index]
    )
    return MappingProxyType(reports)


def _device_copy(value: Tensor, device: torch.device) -> Tensor:
    return value.detach().to(device=device, dtype=value.dtype).contiguous()


def materialize_a4_task(
    split: AV2RealSplit,
    *,
    device: torch.device | str,
    spec: a4.A4BridgeSpec | None = None,
) -> a4.A4Task:
    """Materialize only forward inputs; target fields become NaN sentinels."""

    _validate_real_split(split)
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index != 0:
        raise AV2ExecutionError("AV-2 model materialization requires logical cuda:0")
    resolved_spec = a4.A4BridgeSpec() if spec is None else spec
    template = a4.build_a4_synthetic_task(resolved_spec, device=resolved)
    actions = _device_copy(split.actions, resolved)
    chunks = []
    for template_chunk in template.chunks:
        layout = template_chunk.layout
        valid = actions[:, layout.action_start : layout.action_end]
        padding = layout.action_slot_capacity - layout.valid_action_count
        if padding:
            valid = torch.cat(
                (
                    valid,
                    torch.zeros(
                        (8, padding, 20), dtype=torch.float32, device=resolved
                    ),
                ),
                dim=1,
            )
        chunks.append(replace(template_chunk, actions=valid.contiguous()))
    sentinel = torch.full(
        (8, 5, 3, 1, 1), float("nan"), dtype=torch.float32, device=resolved
    )
    task = replace(
        template,
        global_actions=actions,
        noisy_video=_device_copy(split.noisy_video, resolved),
        base_target=sentinel.clone(),
        video_target=sentinel.clone(),
        context=torch.zeros((8, 2, 2, 4), dtype=torch.float32, device=resolved),
        context_mask=torch.ones((8, 2, 2), dtype=torch.bool, device=resolved),
        proprio=_device_copy(split.proprio, resolved),
        video_timestep=torch.zeros((8, 2), dtype=torch.float32, device=resolved),
        action_timestep=torch.zeros((8, 2), dtype=torch.float32, device=resolved),
        chunks=tuple(chunks),
        score_mask=_device_copy(split.frame_valid_mask[:, 1:], resolved),
        shuffle_permutation=AV2_SHUFFLE_PERMUTATION,
        # Keep the frozen layout recipe identity.  The real input identities
        # live in source_tensor_digests and never masquerade as the AV-1 bytes.
        recipe_digest=template.recipe_digest,
        source_tensor_digests={
            name: digest
            for name, digest in split.tensor_sha256.items()
            if name != "target"
        },
    )
    # The inherited validator accepts the frozen recipe/layout while target
    # payloads remain inaccessible sentinels (it validates dtype/shape only).
    a4.a3.a2r1._validate_vendor_task(task)
    assert_target_sentinels(task)
    return task


def assert_target_sentinels(task: a4.A4Task) -> None:
    """Fail if a model task carries real target bytes."""

    for name in ("base_target", "video_target"):
        value = getattr(task, name)
        if tuple(value.shape) != (8, 5, 3, 1, 1) or not bool(torch.isnan(value).all()):
            raise AV2ExecutionError(f"forward task {name} is not the NaN sentinel")


def masked_future_mse(
    prediction: Tensor,
    target: Tensor,
    frame_valid_mask: Tensor | None = None,
) -> Tensor:
    """Direct positive MSE on valid future frames 1..4."""

    if tuple(prediction.shape) != (8, 5, 3, 1, 1):
        raise AV2ExecutionError("prediction must be [8,5,3,1,1]")
    if tuple(target.shape) != tuple(prediction.shape):
        raise AV2ExecutionError("target and prediction shapes differ")
    if target.device != prediction.device or target.dtype != prediction.dtype:
        raise AV2ExecutionError("target and prediction dtype/device differ")
    mask = (
        torch.ones((8, 5), dtype=torch.bool, device=prediction.device)
        if frame_valid_mask is None
        else frame_valid_mask
    )
    if tuple(mask.shape) != (8, 5) or mask.dtype is not torch.bool:
        raise AV2ExecutionError("frame_valid_mask must be bool [8,5]")
    expanded = mask[:, 1:].view(8, 4, 1, 1, 1).expand(8, 4, 3, 1, 1)
    residual = (prediction[:, 1:] - target[:, 1:]).masked_select(expanded)
    if residual.numel() == 0:
        raise AV2ExecutionError("future loss mask is empty")
    return residual.square().mean()


def _metric(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    value = float(masked_future_mse(prediction, target, mask).detach().double().item())
    if not math.isfinite(value):
        raise AV2ExecutionError("AV-2 MSE is non-finite")
    return value


def _evaluate_arm(
    arm: a4.A4BridgeArm,
    task: a4.A4Task,
    target: Tensor,
    mask: Tensor,
    *,
    modes: Sequence[a4.ActionMode],
) -> tuple[Mapping[str, float], Mapping[str, a4.A4SequenceOutput]]:
    assert_target_sentinels(task)
    metrics: dict[str, float] = {}
    outputs: dict[str, a4.A4SequenceOutput] = {}
    arm.train(False)
    with torch.no_grad():
        for mode in modes:
            output = arm(task, mode=mode)
            if not bool(torch.isfinite(output.video_prediction).all().item()):
                raise AV2ExecutionError(
                    "non-finite forward output indicates target-sentinel leakage or instability"
                )
            metrics[str(mode)] = _metric(output.video_prediction, target, mask)
            outputs[str(mode)] = output
    arm.train(True)
    return MappingProxyType(metrics), MappingProxyType(outputs)


def _target_sentinel_perturbation_bitwise_invariant(
    arm: a4.A4BridgeArm,
    task: a4.A4Task,
    baseline: a4.A4SequenceOutput,
) -> bool:
    """Prove that changing only the NaN sentinel payload cannot affect forward."""

    assert_target_sentinels(task)
    perturbed = replace(
        task,
        base_target=-task.base_target,
        video_target=-task.video_target,
    )
    assert_target_sentinels(perturbed)
    arm.train(False)
    with torch.no_grad():
        alternate = arm(perturbed, mode="correct")
    arm.train(True)
    return all(
        torch.equal(left, right)
        for left, right in (
            (baseline.video_prediction, alternate.video_prediction),
            (baseline.common_prediction, alternate.common_prediction),
            (baseline.action_delta, alternate.action_delta),
            (baseline.final_common_hidden, alternate.final_common_hidden),
        )
    )


def _set_cosine_lr(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    start: float,
    end: float,
) -> float:
    if not 0 <= step < total_steps:
        raise AV2ExecutionError("optimizer step is outside the frozen schedule")
    fraction = float(step) / float(max(total_steps - 1, 1))
    lr = end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * fraction))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _named_parameters(
    module: nn.Module, names: Sequence[str]
) -> tuple[list[str], list[nn.Parameter]]:
    table = dict(module.named_parameters())
    ordered = sorted(names, key=lambda value: value.encode("utf-8"))
    if len(ordered) != len(set(ordered)) or any(name not in table for name in ordered):
        raise AV2ExecutionError("optimizer parameter scope is malformed")
    parameters = [table[name] for name in ordered]
    if any(not parameter.requires_grad for parameter in parameters):
        raise AV2ExecutionError("optimizer scope contains frozen parameters")
    return ordered, parameters


def _snapshot(names: Sequence[str], parameters: Sequence[nn.Parameter]) -> Mapping[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in zip(names, parameters, strict=True)
    }


def _rms(values: Sequence[Tensor]) -> float:
    total = sum(value.numel() for value in values)
    if total == 0:
        return 0.0
    square_sum = sum(
        float(value.detach().double().square().sum().item()) for value in values
    )
    return math.sqrt(square_sum / float(total))


def _gradient_rms(parameters: Sequence[nn.Parameter]) -> float:
    values = [
        parameter.grad
        if parameter.grad is not None
        else torch.zeros_like(parameter)
        for parameter in parameters
    ]
    return _rms(values)


def _update_rms(
    names: Sequence[str],
    parameters: Sequence[nn.Parameter],
    initial: Mapping[str, Tensor],
) -> float:
    return _rms(
        [
            parameter.detach() - initial[name]
            for name, parameter in zip(names, parameters, strict=True)
        ]
    )


def _parameter_digest_manifest(
    names: Sequence[str], parameters: Sequence[nn.Parameter]
) -> Mapping[str, object]:
    entries = {
        name: {
            "dtype": str(parameter.dtype),
            "shape": list(parameter.shape),
            "sha256": _tensor_sha256(parameter),
        }
        for name, parameter in zip(names, parameters, strict=True)
    }
    return MappingProxyType(
        {
            "combined_sha256": _canonical_sha256(entries),
            "entries": entries,
        }
    )


def _train_serial_arm(
    arm: a4.A4BridgeArm,
    *,
    names: Sequence[str],
    parameters: Sequence[nn.Parameter],
    task: a4.A4Task,
    target: Tensor,
    mask: Tensor,
    steps: int,
    lr_start: float,
    lr_end: float,
    deadline: float,
    label: str,
    progress_callback: Callable[[Mapping[str, object]], None] | None,
    diagnostic_parameter_groups: Mapping[str, Sequence[nn.Parameter]] | None = None,
) -> Mapping[str, object]:
    optimizer = torch.optim.AdamW(
        parameters,
        lr=lr_start,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    trace: list[float] = []
    lr_trace: list[float] = []
    step0_gradient_rms = 0.0
    step0_gradient_rms_by_scope: dict[str, float] = {}
    clip_norm_max = 0.0
    arm.train(True)
    for step in range(steps):
        if time.monotonic() >= deadline:
            raise AV2ExecutionError("AV-2 exceeded the 60-minute total wall-clock budget")
        lr_trace.append(
            _set_cosine_lr(
                optimizer,
                step=step,
                total_steps=steps,
                start=lr_start,
                end=lr_end,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        assert_target_sentinels(task)
        output = arm(task, mode="correct")
        if not bool(torch.isfinite(output.video_prediction).all().item()):
            raise AV2ExecutionError("training output is non-finite")
        loss = masked_future_mse(output.video_prediction, target, mask)
        if not bool(torch.isfinite(loss).item()):
            raise AV2ExecutionError("training loss is non-finite")
        trace.append(float(loss.detach().item()))
        loss.backward()
        if step == 0:
            step0_gradient_rms = _gradient_rms(parameters)
            if diagnostic_parameter_groups is not None:
                step0_gradient_rms_by_scope = {
                    name: _gradient_rms(scope)
                    for name, scope in diagnostic_parameter_groups.items()
                }
        clip = float(
            torch.nn.utils.clip_grad_norm_(parameters, AV2_GRAD_CLIP_NORM)
            .detach()
            .item()
        )
        if not math.isfinite(clip):
            raise AV2ExecutionError("gradient norm is non-finite")
        clip_norm_max = max(clip_norm_max, clip)
        optimizer.step()
        if progress_callback is not None and (
            step == 0 or (step + 1) % 100 == 0 or step + 1 == steps
        ):
            progress_callback(
                {
                    "arm": label,
                    "loss": trace[-1],
                    "lr": lr_trace[-1],
                    "step": step + 1,
                    "steps": steps,
                }
            )
    return MappingProxyType(
        {
            "clip_norm_max": clip_norm_max,
            "final_loss_pre_update": trace[-1],
            "initial_loss_pre_update": trace[0],
            "loss_trace": tuple(trace),
            "lr_end_observed": lr_trace[-1],
            "lr_start_observed": lr_trace[0],
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "parameter_names": tuple(names),
            "step0_gradient_rms": step0_gradient_rms,
            "step0_gradient_rms_by_scope": step0_gradient_rms_by_scope,
            "steps_completed": len(trace),
        }
    )


def classify_av2(
    *,
    all_validity: bool,
    metrics: Mapping[str, object],
    thresholds: Mapping[str, object] | None = None,
) -> tuple[str, str | None, Mapping[str, bool]]:
    """Typed AV-2 verdict; only GO unlocks AV-3."""

    resolved = _thresholds(thresholds)
    required = (
        "candidate_heldout_decrease",
        "candidate_vs_reference_ratio",
        "correct_vs_no_action_improvement",
        "correct_vs_shuffle_improvement",
    )
    if any(name not in metrics for name in required):
        raise AV2ExecutionError("classification metrics are incomplete")
    values = {name: float(metrics[name]) for name in required}
    if not all(math.isfinite(value) for value in values.values()):
        return (
            "AV2_INVALID_FAIL_CLOSED",
            "NONFINITE_DECISION_METRIC",
            MappingProxyType({}),
        )
    checks = {
        "candidate_heldout_decrease": values["candidate_heldout_decrease"]
        >= resolved["candidate_heldout_decrease_min"],
        "candidate_vs_reference_noninferiority": values[
            "candidate_vs_reference_ratio"
        ]
        <= resolved["candidate_vs_reference_noninferiority_ratio_max"],
        "correct_vs_no_action": values["correct_vs_no_action_improvement"]
        >= resolved["correct_vs_no_action_improvement_min"],
        "correct_vs_shuffle": values["correct_vs_shuffle_improvement"]
        >= resolved["correct_vs_shuffle_improvement_min"],
    }
    if not all_validity:
        return (
            "AV2_INVALID_FAIL_CLOSED",
            "AV2_VALIDITY_FAILURE",
            MappingProxyType(checks),
        )
    if all(checks.values()):
        return "AV2_GO", None, MappingProxyType(checks)
    action_unsupported = (
        values["candidate_vs_reference_ratio"] >= 1.0
        and values["correct_vs_no_action_improvement"] <= 0.0
        and values["correct_vs_shuffle_improvement"] <= 0.0
    )
    if action_unsupported:
        return (
            "REDUCED_ARCH_STOP",
            "AV2_ACTION_CONDITIONING_UNSUPPORTED",
            MappingProxyType(checks),
        )
    return (
        "REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED",
        "AV2_REVIEW_BAND_NO_BUG_BUDGET",
        MappingProxyType(checks),
    )


def _mapping(config: Mapping[str, object], canonical: str, alias: str) -> Mapping[str, object]:
    value = config.get(canonical, config.get(alias))
    if not isinstance(value, Mapping):
        raise AV2ExecutionError(f"config.{canonical} must be a mapping")
    return value


def _require_execution_config(
    config: Mapping[str, object],
) -> tuple[str, int, int, float, float, Mapping[str, float]]:
    if not isinstance(config, Mapping):
        raise AV2ExecutionError("config must be a mapping")
    if config.get("schema") != AV2_EXECUTION_SCHEMA:
        raise AV2ExecutionError("AV-2 execution config schema differs")
    if config.get("architecture_id") != AV2_ARCHITECTURE_ID:
        raise AV2ExecutionError("AV-2 architecture id differs")
    data = _mapping(config, "data_contract", "data")
    training = _mapping(config, "training_contract", "training")
    dataset_root = data.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise AV2ExecutionError("data_contract.dataset_root must be a path string")
    exact_data = {
        "action_dim": 20,
        "action_mode": "ABSOLUTE_EEF20",
        "action_rows": [AV2_ACTION_START, AV2_ACTION_END_EXCLUSIVE],
        "batch_size": 8,
        "camera": AV2_CAMERA,
        "frame_indices": list(AV2_FRAME_INDICES),
        "heldout_episode_ids": list(AV2_HELDOUT_EPISODES),
        "noisy_video_recipe": "ANCHOR_RGB_MEAN_DIV_255_REPEAT_5",
        "normalize": None,
        "observation_decoder": "CV2_IMDECODE_COLOR_NO_EXTRA_CONVERSION",
        "observation_projection": "SPATIAL_RGB_MEAN_DIV_255",
        "proprio_recipe": "ABSOLUTE_EEF20_RAW_ROW_0_REPEAT_2_NORMALIZATION_NONE",
        "raw_end_exclusive": AV2_RAW_END_EXCLUSIVE,
        "raw_start": AV2_RAW_START,
        "shuffle_permutation": list(AV2_SHUFFLE_PERMUTATION),
        "target_forward_access": False,
        "target_recipe": "RGB_MEAN_DIV_255_MINUS_ANCHOR_FRAME_0",
        "task": AV2_TASK_NAME,
        "train_episode_ids": list(AV2_TRAIN_EPISODES),
        "variant": AV2_VARIANT,
        "window_count_per_split": 8,
    }
    expected_data_keys = frozenset(exact_data) | {"dataset_root"}
    if frozenset(data) != expected_data_keys:
        raise AV2ExecutionError("data_contract keys differ from AV-2")
    for name, expected in exact_data.items():
        if data.get(name) != expected or type(data.get(name)) is not type(expected):
            raise AV2ExecutionError(f"data_contract.{name} differs from AV-2")
    exact_training = {
        "checkpoint_load": False,
        "checkpoint_save": False,
        "final_step_only": True,
        "fresh_initialization": True,
        "gradient_clip_norm": AV2_GRAD_CLIP_NORM,
        "loss": "POSITIVE_MASKED_FUTURE_MSE_NO_SUBTRACTION",
        "lr_end": AV2_LR_END,
        "lr_schedule": "COSINE",
        "lr_start": AV2_LR_START,
        "max_steps_per_arm": AV2_STEPS_PER_ARM,
        "max_wall_clock_minutes": AV2_MAX_WALL_CLOCK_MINUTES,
        "optimizer": "AdamW",
        "save_final_screen_state": False,
        "seed": AV2_SEED,
        "serial_arm_order": ["reference", "candidate"],
        "serial_arms": True,
        "weight_decay": 0.0,
    }
    if frozenset(training) != frozenset(exact_training):
        raise AV2ExecutionError("training_contract keys differ from AV-2")
    for name, expected in exact_training.items():
        if training.get(name) != expected or type(training.get(name)) is not type(expected):
            raise AV2ExecutionError(f"training_contract.{name} differs from AV-2")
    threshold_config = config.get("thresholds")
    if not isinstance(threshold_config, Mapping) or frozenset(
        threshold_config
    ) != frozenset(_DEFAULT_THRESHOLDS):
        raise AV2ExecutionError("threshold keys differ from AV-2")
    lr_start = float(training["lr_start"])
    lr_end = float(training["lr_end"])
    if not (math.isfinite(lr_start) and math.isfinite(lr_end) and 0.0 < lr_end <= lr_start):
        raise AV2ExecutionError("AV-2 learning-rate schedule is malformed")
    return (
        dataset_root,
        int(training["max_steps_per_arm"]),
        int(training["max_wall_clock_minutes"]),
        lr_start,
        lr_end,
        _thresholds(threshold_config),
    )


def run_av2_tiny_real_data_overfit(
    config: Mapping[str, object],
    *,
    expected_gpu_uuid: str,
    source_and_predecessor_pins_match: bool,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    preloaded_bundle: AV2RealDataBundle | None = None,
) -> Mapping[str, object]:
    """Execute the single-GPU fresh-init, serial 1000-step-per-arm screen."""

    started_monotonic = time.monotonic()
    (
        dataset_root,
        steps,
        wall_minutes,
        lr_start,
        lr_end,
        thresholds,
    ) = _require_execution_config(config)
    if type(source_and_predecessor_pins_match) is not bool:
        raise AV2ExecutionError("predecessor pin proof must be an exact bool")
    if not source_and_predecessor_pins_match:
        raise AV2ExecutionError("source or predecessor pin mismatch")
    if not isinstance(expected_gpu_uuid, str) or not expected_gpu_uuid.startswith("GPU-"):
        raise AV2ExecutionError("expected_gpu_uuid is malformed")
    hardware = config.get("hardware_contract")
    if not isinstance(hardware, Mapping):
        raise AV2ExecutionError("hardware_contract is required")
    expected_hardware = {
        "cuda_visible_devices": expected_gpu_uuid,
        "gpu_count": 1,
        "gpu_uuid": expected_gpu_uuid,
        "logical_device": "cuda:0",
        "physical_gpu_index": 0,
        "reservation": "EXCLUSIVE_AND_IDLE_RECHECK_IMMEDIATELY_BEFORE_ROOT",
    }
    if any(
        hardware.get(name) != value or type(hardware.get(name)) is not type(value)
        for name, value in expected_hardware.items()
    ):
        raise AV2ExecutionError("hardware_contract differs from AV-2 binding")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu_uuid:
        raise AV2ExecutionError("CUDA_VISIBLE_DEVICES differs from AV-2 binding")
    if os.environ.get("FUSED_GDN_PRECISION") != "0":
        raise AV2ExecutionError("FUSED_GDN_PRECISION must equal 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AV2ExecutionError("AV-2 requires exactly one visible CUDA device")
    torch.manual_seed(AV2_SEED)
    torch.cuda.manual_seed_all(AV2_SEED)
    device = torch.device("cuda:0")
    if preloaded_bundle is None:
        bundle = load_av2_real_data(dataset_root)
    else:
        if not isinstance(preloaded_bundle, AV2RealDataBundle):
            raise AV2ExecutionError("preloaded_bundle has the wrong type")
        configured_root = str(Path(dataset_root).expanduser().resolve(strict=True))
        if preloaded_bundle.dataset_root != configured_root:
            raise AV2ExecutionError("preloaded bundle dataset root differs from config")
        _validate_real_split(preloaded_bundle.train)
        _validate_real_split(preloaded_bundle.heldout)
        av2_scaffold.assert_train_holdout_disjoint(
            preloaded_bundle.train.windows, preloaded_bundle.heldout.windows
        )
        bundle = preloaded_bundle
    adequacy = assess_data_adequacy(bundle, thresholds)
    if not bool(adequacy["all_adequate"]):
        raise AV2ExecutionError("fixed AV-2 data slices fail adequacy")
    train_task = materialize_a4_task(bundle.train, device=device)
    heldout_task = materialize_a4_task(bundle.heldout, device=device)
    train_target = _device_copy(bundle.train.target, device)
    heldout_target = _device_copy(bundle.heldout.target, device)
    train_mask = _device_copy(bundle.train.frame_valid_mask, device)
    heldout_mask = _device_copy(bundle.heldout.frame_valid_mask, device)
    pair = a4.build_cach_a4_pair(device=device)

    reference_names, reference_parameters = _named_parameters(
        pair.reference, pair.reference_common_trainable_names
    )
    candidate_names, candidate_parameters = _named_parameters(
        pair.candidate,
        pair.candidate_common_trainable_names | pair.candidate_action_trainable_names,
    )
    candidate_common_names, candidate_common_parameters = _named_parameters(
        pair.candidate, pair.candidate_common_trainable_names
    )
    candidate_action_names, candidate_action_parameters = _named_parameters(
        pair.candidate, pair.candidate_action_trainable_names
    )
    if set(reference_names) != set(pair.reference_common_trainable_names):
        raise AV2ExecutionError("reference optimizer scope drifted")
    if set(candidate_names) != set(
        pair.candidate_common_trainable_names | pair.candidate_action_trainable_names
    ):
        raise AV2ExecutionError("candidate optimizer scope drifted")
    reference_initial = _snapshot(reference_names, reference_parameters)
    candidate_initial = _snapshot(candidate_names, candidate_parameters)
    candidate_action_initial = _snapshot(
        candidate_action_names, candidate_action_parameters
    )

    theta0_reference_train, theta0_reference_outputs = _evaluate_arm(
        pair.reference,
        train_task,
        train_target,
        train_mask,
        modes=("correct",),
    )
    theta0_candidate_train, theta0_candidate_train_outputs = _evaluate_arm(
        pair.candidate,
        train_task,
        train_target,
        train_mask,
        modes=("correct",),
    )
    theta0_reference_heldout, theta0_reference_heldout_outputs = _evaluate_arm(
        pair.reference,
        heldout_task,
        heldout_target,
        heldout_mask,
        modes=("correct",),
    )
    theta0_candidate_heldout, theta0_candidate_heldout_outputs = _evaluate_arm(
        pair.candidate,
        heldout_task,
        heldout_target,
        heldout_mask,
        modes=("correct", "shuffle", "no_action"),
    )
    theta0_common_equal = torch.equal(
        theta0_reference_outputs["correct"].common_prediction,
        theta0_candidate_train_outputs["correct"].common_prediction,
    )
    theta0_candidate_identity = torch.equal(
        theta0_candidate_train_outputs["correct"].video_prediction,
        theta0_candidate_train_outputs["correct"].common_prediction,
    )
    del theta0_reference_heldout_outputs, theta0_candidate_heldout_outputs

    deadline = started_monotonic + float(wall_minutes * 60)
    reference_training = _train_serial_arm(
        pair.reference,
        names=reference_names,
        parameters=reference_parameters,
        task=train_task,
        target=train_target,
        mask=train_mask,
        steps=steps,
        lr_start=lr_start,
        lr_end=lr_end,
        deadline=deadline,
        label="reference",
        progress_callback=progress_callback,
    )
    candidate_training = _train_serial_arm(
        pair.candidate,
        names=candidate_names,
        parameters=candidate_parameters,
        task=train_task,
        target=train_target,
        mask=train_mask,
        steps=steps,
        lr_start=lr_start,
        lr_end=lr_end,
        deadline=deadline,
        label="candidate",
        progress_callback=progress_callback,
        diagnostic_parameter_groups={
            "candidate_action": candidate_action_parameters,
        },
    )

    final_reference_train, _ = _evaluate_arm(
        pair.reference, train_task, train_target, train_mask, modes=("correct",)
    )
    final_candidate_train, _ = _evaluate_arm(
        pair.candidate, train_task, train_target, train_mask, modes=("correct",)
    )
    final_reference_heldout, final_reference_outputs = _evaluate_arm(
        pair.reference, heldout_task, heldout_target, heldout_mask, modes=("correct",)
    )
    final_candidate_heldout, final_candidate_outputs = _evaluate_arm(
        pair.candidate,
        heldout_task,
        heldout_target,
        heldout_mask,
        modes=("correct", "shuffle", "no_action", "seam_disabled"),
    )
    epsilon = 1.0e-30
    decision_metrics: dict[str, float] = {
        "candidate_heldout_decrease": 1.0
        - final_candidate_heldout["correct"]
        / max(theta0_candidate_heldout["correct"], epsilon),
        "candidate_vs_reference_ratio": final_candidate_heldout["correct"]
        / max(final_reference_heldout["correct"], epsilon),
        "correct_vs_no_action_improvement": 1.0
        - final_candidate_heldout["correct"]
        / max(final_candidate_heldout["no_action"], epsilon),
        "correct_vs_shuffle_improvement": 1.0
        - final_candidate_heldout["correct"]
        / max(final_candidate_heldout["shuffle"], epsilon),
    }
    reference_update_rms = _update_rms(
        reference_names, reference_parameters, reference_initial
    )
    candidate_update_rms = _update_rms(
        candidate_names, candidate_parameters, candidate_initial
    )
    candidate_action_update_rms = _update_rms(
        candidate_action_names,
        candidate_action_parameters,
        candidate_action_initial,
    )
    final_correct = final_candidate_outputs["correct"]
    final_no_action = final_candidate_outputs["no_action"]
    final_seam_disabled = final_candidate_outputs["seam_disabled"]
    if (
        pair.candidate.action_stream is None
        or final_correct.raw_action is None
        or final_correct.action_present_mask is None
    ):
        raise AV2ExecutionError("final candidate action stream diagnostics are absent")
    stream_diagnostics = dict(
        a4._state_stream_diagnostics(
            pair.candidate.action_stream,
            final_correct.raw_action,
            final_correct.final_common_hidden.detach(),
            final_correct.action_present_mask,
        )
    )
    candidate_target_sentinel_invariant = (
        _target_sentinel_perturbation_bitwise_invariant(
            pair.candidate,
            heldout_task,
            final_correct,
        )
    )
    reference_target_sentinel_invariant = (
        _target_sentinel_perturbation_bitwise_invariant(
            pair.reference,
            heldout_task,
            final_reference_outputs["correct"],
        )
    )
    target_sentinel_invariant = (
        candidate_target_sentinel_invariant
        and reference_target_sentinel_invariant
    )
    no_action_bypass = (
        torch.equal(final_no_action.video_prediction, final_no_action.common_prediction)
        and not bool(final_no_action.action_delta.count_nonzero().item())
        and final_no_action.reducer_calls_this_forward == 0
        and final_no_action.action_stream_calls_this_forward == 0
    )
    seam_disabled_bypass = (
        torch.equal(
            final_seam_disabled.video_prediction,
            final_seam_disabled.common_prediction,
        )
        and not bool(final_seam_disabled.action_delta.count_nonzero().item())
        and final_seam_disabled.reducer_calls_this_forward == 0
        and final_seam_disabled.action_stream_calls_this_forward == 0
    )
    bypass_modes_bitwise_equal = torch.equal(
        final_no_action.video_prediction,
        final_seam_disabled.video_prediction,
    )
    vendor_calls_positive = all(
        output.vendor_calls_this_forward > 0
        for output in (
            final_reference_outputs["correct"],
            final_correct,
            final_candidate_outputs["shuffle"],
            final_no_action,
            final_seam_disabled,
        )
    )
    candidate_action_step0_gradient_rms = float(
        candidate_training["step0_gradient_rms_by_scope"]["candidate_action"]  # type: ignore[index]
    )
    final_parameter_digests = {
        "reference_common": dict(
            _parameter_digest_manifest(reference_names, reference_parameters)
        ),
        "candidate_common": dict(
            _parameter_digest_manifest(
                candidate_common_names, candidate_common_parameters
            )
        ),
        "candidate_action": dict(
            _parameter_digest_manifest(
                candidate_action_names, candidate_action_parameters
            )
        ),
    }
    target_sha_absent_from_forward_task = all(
        digest not in train_task.source_tensor_digests.values()
        and digest not in heldout_task.source_tensor_digests.values()
        for digest in (
            bundle.train.tensor_sha256["target"],
            bundle.heldout.tensor_sha256["target"],
        )
    )
    validity: dict[str, bool] = {
        "candidate_action_only_step0_gradient_positive": (
            candidate_action_step0_gradient_rms > 0.0
        ),
        "candidate_action_only_updated": candidate_action_update_rms > 0.0,
        "candidate_optimizer_updated": candidate_update_rms > 0.0,
        "candidate_action_delta_nonzero": bool(
            final_correct.action_delta.count_nonzero().item()
        ),
        "candidate_train_loss_declined": final_candidate_train["correct"]
        < theta0_candidate_train["correct"],
        "data_adequate": bool(adequacy["all_adequate"]),
        "decay_action_blind_bitwise": bool(
            stream_diagnostics["decay_action_blind_bitwise"]
        ),
        "delta_state_to_future_jvp_positive": float(
            stream_diagnostics["delta_state_to_future_output_jvp_rms"]
        )
        > 0.0,
        "deadline_respected": time.monotonic() < deadline,
        "finite_decision_metrics": all(
            math.isfinite(value) for value in decision_metrics.values()
        ),
        "future_to_prefix_exact_zero": float(
            stream_diagnostics["future_to_prefix_max_abs"]
        )
        == 0.0,
        "no_action_bypass_exact": no_action_bypass,
        "no_action_and_seam_bypass_bitwise_equal": bypass_modes_bitwise_equal,
        "oddness_exact_zero": float(stream_diagnostics["oddness_max_abs"])
        == 0.0,
        "action_jvp_positive": float(stream_diagnostics["action_jvp_rms"]) > 0.0,
        "past_action_to_future_jvp_positive": float(
            stream_diagnostics["past_action_to_future_output_jvp_rms"]
        )
        > 0.0,
        "predecessor_pins_match": source_and_predecessor_pins_match,
        "reference_optimizer_updated": reference_update_rms > 0.0,
        "reference_train_loss_declined": final_reference_train["correct"]
        < theta0_reference_train["correct"],
        "seam_disabled_bypass_exact": seam_disabled_bypass,
        "shuffle_changes_prediction": not torch.equal(
            final_correct.video_prediction,
            final_candidate_outputs["shuffle"].video_prediction,
        ),
        "state_jvp_positive": float(
            stream_diagnostics["common_state_to_output_jvp_rms"]
        )
        > 0.0,
        "target_sentinel_perturbation_bitwise_invariant": target_sentinel_invariant,
        "target_fields_are_sentinel": True,
        "target_sha_absent_from_forward_task": target_sha_absent_from_forward_task,
        "theta0_candidate_identity": theta0_candidate_identity,
        "theta0_common_equal": theta0_common_equal,
        "vendor_calls_positive": vendor_calls_positive,
        "zero_origin_exact_zero": float(stream_diagnostics["zero_origin_max_abs"])
        == 0.0,
    }
    validity["run_valid"] = all(validity.values())
    verdict, reason_code, verdict_checks = classify_av2(
        all_validity=validity["run_valid"],
        metrics=decision_metrics,
        thresholds=thresholds,
    )
    elapsed = time.monotonic() - started_monotonic
    return MappingProxyType(
        {
            "schema": AV2_EXECUTION_RESULT_SCHEMA,
            "architecture_id": AV2_ARCHITECTURE_ID,
            "data": {
                "adequacy": dict(adequacy),
                "dataset_root": bundle.dataset_root,
                "heldout_episode_file_sha256": dict(
                    bundle.heldout.episode_file_sha256
                ),
                "heldout_tensor_sha256": dict(bundle.heldout.tensor_sha256),
                "selection_sha256": bundle.selection_sha256,
                "train_episode_file_sha256": dict(bundle.train.episode_file_sha256),
                "train_tensor_sha256": dict(bundle.train.tensor_sha256),
            },
            "theta0": {
                "candidate_heldout_mse": dict(theta0_candidate_heldout),
                "candidate_train_mse": dict(theta0_candidate_train),
                "candidate_identity": theta0_candidate_identity,
                "common_equal": theta0_common_equal,
                "manifests": pair.theta0_manifests,
                "reference_heldout_mse": dict(theta0_reference_heldout),
                "reference_train_mse": dict(theta0_reference_train),
            },
            "training": {
                "candidate": dict(candidate_training),
                "candidate_action_update_rms": candidate_action_update_rms,
                "candidate_update_rms": candidate_update_rms,
                "final_parameter_digests": final_parameter_digests,
                "reference": dict(reference_training),
                "reference_update_rms": reference_update_rms,
                "serial_arms": True,
            },
            "evaluation": {
                "candidate_heldout_mse": dict(final_candidate_heldout),
                "candidate_train_mse": dict(final_candidate_train),
                "decision_metrics": decision_metrics,
                "reference_heldout_mse": dict(final_reference_heldout),
                "reference_train_mse": dict(final_reference_train),
                "thresholds": dict(thresholds),
            },
            "diagnostics": {
                "action_delta_rms": _rms(
                    [final_candidate_outputs["correct"].action_delta]
                ),
                "no_action_delta_exact_zero": not bool(
                    final_candidate_outputs["no_action"].action_delta.count_nonzero()
                ),
                "no_action_bypass_exact": no_action_bypass,
                "seam_disabled_bypass_exact": seam_disabled_bypass,
                "state_stream": stream_diagnostics,
                "shuffle_changes_prediction": not torch.equal(
                    final_candidate_outputs["correct"].video_prediction,
                    final_candidate_outputs["shuffle"].video_prediction,
                ),
                "target_sha_absent_from_forward_task": (
                    target_sha_absent_from_forward_task
                ),
                "target_sentinel_perturbation_bitwise_invariant": (
                    target_sentinel_invariant
                ),
                "candidate_target_sentinel_perturbation_bitwise_invariant": (
                    candidate_target_sentinel_invariant
                ),
                "reference_target_sentinel_perturbation_bitwise_invariant": (
                    reference_target_sentinel_invariant
                ),
                "vendor_calls_positive": vendor_calls_positive,
            },
            "validity": validity,
            "verdict": verdict,
            "reason_code": reason_code,
            "verdict_checks": dict(verdict_checks),
            "runtime": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "elapsed_seconds": elapsed,
                "expected_gpu_uuid": expected_gpu_uuid,
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "visible_gpu_name": torch.cuda.get_device_name(0),
            },
        }
    )


__all__ = [
    "AV2_ARCHITECTURE_ID",
    "AV2_CAMERA",
    "AV2_EXECUTION_RESULT_SCHEMA",
    "AV2_EXECUTION_SCHEMA",
    "AV2_FRAME_INDICES",
    "AV2_HELDOUT_EPISODES",
    "AV2RealDataBundle",
    "AV2RealSplit",
    "AV2ExecutionError",
    "AV2_SEED",
    "AV2_STEPS_PER_ARM",
    "AV2_TASK_NAME",
    "AV2_TRAIN_EPISODES",
    "AV2_VARIANT",
    "assert_target_sentinels",
    "assess_data_adequacy",
    "classify_av2",
    "load_av2_real_data",
    "masked_future_mse",
    "materialize_a4_task",
    "run_av2_tiny_real_data_overfit",
]
