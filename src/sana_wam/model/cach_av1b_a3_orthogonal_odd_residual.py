"""CACH-A3 normalized-common, exact-odd output-residual screen.

This module is an experimental synthetic architecture screen only.  It reuses
the frozen CACH-A2-R1 task and the real vendor-GDN reference arm, but keeps the
entire common video trunk action blind.  The common prediction is produced
from a parameter-free RMS-normalized final hidden.  A candidate-only action
core is exactly odd-symmetrized and added once, in video-output space; no
vendor block, normalization, or FFN follows that addition.

No target tensor is accepted by any forward method.  Targets are consumed by
the screen's positive common and delta losses only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import platform
import resource
import time
from typing import Final, Literal

import torch
from torch import Tensor, nn

from sana_wam.cach.action_conditioning import (
    reduce_end_of_bin_action_condition,
)
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model import (
    cach_av1b_a2_zero_anchor_vendor_gdn_bridge_r1 as a2r1,
)
from sana_wam.model.video_backbone.sana.hybrid_cache import tensor_digest


CACH_A3_ARCHITECTURE_ID: Final = (
    "CACH-A3-NORMALIZED-COMMON-EXACT-ODD-OUTPUT-RESIDUAL-v1"
)
A3_ARCHITECTURE_ID: Final = CACH_A3_ARCHITECTURE_ID
A3_CONFIG_SCHEMA: Final = (
    "cach.cach_a3.normalized_common_exact_odd_output_residual.config.v1"
)
A3_SCREEN_SCHEMA: Final = (
    "cach.cach_a3.normalized_common_exact_odd_output_residual.screen.v1"
)
A3_INITIALIZER_REVISION: Final = "cach-a3-orthogonal-odd-residual-v1"
A3_CANDIDATE_ONLY_SEED: Final = 2026080521
A3_TRAIN_STEPS: Final = 200
A3_LR_START: Final = 3.0e-3
A3_LR_END: Final = 3.0e-5
A3_SHUFFLE_PERMUTATION: Final = a2r1.AV1B_SHUFFLE_PERMUTATION

ActionMode = Literal["correct", "shuffle", "no_action", "seam_disabled"]
A3BridgeSpec = a2r1.AV1BVendorGDNBridgeSpec
A3Task = a2r1.AV1BVendorTask


@dataclass(frozen=True)
class A3SequenceOutput:
    """Orthogonally decomposed common and action predictions."""

    video_prediction: Tensor = field(repr=False, compare=False)
    common_prediction: Tensor = field(repr=False, compare=False)
    action_delta: Tensor = field(repr=False, compare=False)
    final_common_hidden: Tensor = field(repr=False, compare=False)
    raw_action: Tensor | None = field(repr=False, compare=False)
    action_present_mask: Tensor | None = field(repr=False, compare=False)
    vendor_calls_this_forward: int
    action_conditioner_calls_this_forward: int
    action_output_calls_this_forward: int


@dataclass(frozen=True)
class A3BridgePair:
    spec: A3BridgeSpec
    reference: "A3BridgeArm"
    candidate: "A3BridgeArm"
    reference_common_trainable_names: frozenset[str]
    candidate_common_trainable_names: frozenset[str]
    candidate_action_trainable_names: frozenset[str]
    theta0_manifests: Mapping[str, object]


def _parameter_free_rms_norm(value: Tensor, eps: float = 1.0e-6) -> Tensor:
    return value * torch.rsqrt(
        value.float().square().mean(dim=-1, keepdim=True) + eps
    ).to(dtype=value.dtype)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_manifest(module: nn.Module) -> Mapping[str, object]:
    parameters: dict[str, object] = {}
    buffers: dict[str, object] = {}
    for name, value in sorted(
        module.named_parameters(), key=lambda item: item[0].encode("utf-8")
    ):
        detached = value.detach()
        parameters[name] = {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "requires_grad": value.requires_grad,
            "sha256": tensor_digest(detached),
        }
    for name, value in sorted(
        module.named_buffers(), key=lambda item: item[0].encode("utf-8")
    ):
        detached = value.detach()
        buffers[name] = {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "sha256": tensor_digest(detached),
        }
    entries = {"parameters": parameters, "buffers": buffers}
    return {"entries": entries, "combined_sha256": _canonical_digest(entries)}


def _active_indices(mask: Tensor) -> Tensor:
    if mask.dtype is not torch.bool:
        raise TypeError("action_present_mask must be bool")
    if mask.ndim != 3 or mask.shape[-1] != 1:
        raise ValueError("action_present_mask must be [batch,time,1]")
    return mask.squeeze(-1).reshape(-1).nonzero(as_tuple=False).flatten()


class ExactOddCausalActionResidual(nn.Module):
    """Bias-free 20/64/20 core and zero-init 20/3 output projection.

    Exact oddness is imposed on the whole nonlinear core as
    ``0.5 * (g(a) - g(-a))``.  Only typed-active rows are gathered and
    evaluated.  Inactive rows are materialized by scattering exact zeros.
    """

    def __init__(self, spec: A3BridgeSpec) -> None:
        super().__init__()
        self.spec = spec
        self.conditioner = nn.Sequential(
            nn.Linear(
                spec.action_dim,
                spec.hidden_dim,
                bias=False,
                device="cpu",
                dtype=torch.float32,
            ),
            nn.SiLU(),
            nn.Linear(
                spec.hidden_dim,
                spec.action_dim,
                bias=False,
                device="cpu",
                dtype=torch.float32,
            ),
        )
        self.output_projection = nn.Linear(
            spec.action_dim,
            spec.latent_channels,
            bias=False,
            device="cpu",
            dtype=torch.float32,
        )
        self.conditioner_call_count = 0
        self.output_call_count = 0
        self._initialize()

    def _initialize(self) -> None:
        with torch.no_grad():
            for name, parameter in sorted(
                self.conditioner.named_parameters(),
                key=lambda item: item[0].encode("utf-8"),
            ):
                material = (
                    f"{A3_INITIALIZER_REVISION}:{A3_CANDIDATE_ONLY_SEED}:"
                    f"{name}"
                ).encode("utf-8")
                seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                bound = 0.25 / math.sqrt(float(parameter.shape[-1]))
                parameter.uniform_(-bound, bound, generator=generator)
            self.output_projection.weight.zero_()

    def forward(
        self,
        raw_action: Tensor,
        action_present_mask: Tensor,
        *,
        count_calls: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if raw_action.ndim != 3 or raw_action.shape[-1] != self.spec.action_dim:
            raise ValueError("raw_action must be [batch,time,20]")
        if tuple(action_present_mask.shape) != (*raw_action.shape[:-1], 1):
            raise ValueError("raw action and typed mask axes differ")
        active_index = _active_indices(action_present_mask)
        delta_flat = torch.zeros(
            (raw_action.shape[0] * raw_action.shape[1], self.spec.latent_channels),
            dtype=raw_action.dtype,
            device=raw_action.device,
        )
        odd_flat = torch.zeros_like(raw_action.reshape(-1, self.spec.action_dim))
        if not active_index.numel():
            return (
                delta_flat.reshape(
                    raw_action.shape[0], raw_action.shape[1], self.spec.latent_channels
                ),
                odd_flat.reshape_as(raw_action),
            )
        active = raw_action.reshape(-1, self.spec.action_dim).index_select(
            0, active_index
        )
        positive = self.conditioner(active)
        negative = self.conditioner(-active)
        odd = 0.5 * (positive - negative)
        active_delta = self.output_projection(odd)
        odd_flat = odd_flat.index_copy(0, active_index, odd)
        delta_flat = delta_flat.index_copy(0, active_index, active_delta)
        if count_calls:
            self.conditioner_call_count += 2
            self.output_call_count += 1
        return (
            delta_flat.reshape(
                raw_action.shape[0], raw_action.shape[1], self.spec.latent_channels
            ),
            odd_flat.reshape_as(raw_action),
        )


class A3BridgeArm(nn.Module):
    """Action-blind R1 common trunk plus an optional parallel odd head."""

    def __init__(
        self,
        *,
        common: a2r1.AV1BVendorGDNBridgeArm,
        candidate: bool,
    ) -> None:
        super().__init__()
        if common.staging_variant is not CACHStagingVariant.REF_GDN_CORRECTED:
            raise ValueError("A3 common trunk must be the structural R1 reference")
        self.common = common
        self.spec = common.spec
        self.candidate = bool(candidate)
        self.action_residual = (
            ExactOddCausalActionResidual(self.spec) if self.candidate else None
        )
        anchor = (
            torch.zeros(self.spec.action_dim, dtype=torch.float32, device="cpu")
            if self.candidate
            else None
        )
        self.register_buffer("no_action_slot", anchor, persistent=True)

    @property
    def arm_name(self) -> str:
        return "CACH-A3" if self.candidate else "REF-GDN-CORRECTED-A3"

    @property
    def common_trainable_names(self) -> frozenset[str]:
        return frozenset(
            f"common.{name}"
            for name, parameter in self.common.named_parameters()
            if parameter.requires_grad
        )

    @property
    def action_trainable_names(self) -> frozenset[str]:
        if self.action_residual is None:
            return frozenset()
        return frozenset(
            f"action_residual.{name}"
            for name, parameter in self.action_residual.named_parameters()
            if parameter.requires_grad
        )

    def _actions_for_mode(self, task: A3Task, mode: ActionMode) -> Tensor:
        if mode == "correct":
            return task.global_actions
        if mode == "shuffle":
            permutation = torch.tensor(
                A3_SHUFFLE_PERMUTATION,
                dtype=torch.long,
                device=task.global_actions.device,
            )
            return task.global_actions.index_select(0, permutation)
        raise RuntimeError("inactive modes must not read action values")

    @staticmethod
    def _fixed_capacity_actions(
        global_actions: Tensor,
        task_chunk: a2r1.AV1BVendorTaskChunk,
    ) -> Tensor:
        chunk = task_chunk.layout
        valid = global_actions[:, chunk.action_start : chunk.action_end]
        padding = chunk.action_slot_capacity - chunk.valid_action_count
        if not padding:
            return valid
        return torch.cat(
            (
                valid,
                torch.zeros(
                    (valid.shape[0], padding, valid.shape[2]),
                    dtype=valid.dtype,
                    device=valid.device,
                ),
            ),
            dim=1,
        )

    def _reduced_action_and_mask(
        self,
        task: A3Task,
        mode: ActionMode,
    ) -> tuple[Tensor, Tensor]:
        if mode in ("no_action", "seam_disabled"):
            return (
                torch.zeros(
                    (self.spec.batch_size, 5, self.spec.action_dim),
                    dtype=task.noisy_video.dtype,
                    device=task.noisy_video.device,
                ),
                torch.zeros(
                    (self.spec.batch_size, 5, 1),
                    dtype=torch.bool,
                    device=task.noisy_video.device,
                ),
            )
        if self.no_action_slot is None:
            raise RuntimeError("candidate no-action anchor is absent")
        global_actions = self._actions_for_mode(task, mode)
        reduced_parts: list[Tensor] = []
        mask_parts: list[Tensor] = []
        for task_chunk in task.chunks:
            local = self._fixed_capacity_actions(global_actions, task_chunk)
            committed = (
                None
                if task_chunk.layout.action_start == 0
                else global_actions[:, : task_chunk.layout.action_start]
            )
            reduced = reduce_end_of_bin_action_condition(
                local,
                committed_actions=committed,
                chunk=task_chunk.layout,
                no_action_slot=self.no_action_slot,
            )
            valid_count = task_chunk.layout.valid_latent_count
            reduced_parts.append(reduced.condition[:, :valid_count])
            local_mask = torch.tensor(
                tuple(
                    bool(task_chunk.layout.latent_valid_mask[index])
                    and not span.anchor_no_action_slot
                    for index, span in enumerate(
                        task_chunk.layout.latent_action_spans
                    )
                ),
                dtype=torch.bool,
                device=global_actions.device,
            ).view(1, -1).expand(self.spec.batch_size, -1)
            mask_parts.append(local_mask[:, :valid_count])
        raw = torch.cat(reduced_parts, dim=1)
        mask = torch.cat(mask_parts, dim=1).unsqueeze(-1)
        expected = torch.tensor(
            (False, True, True, True, True),
            dtype=torch.bool,
            device=mask.device,
        ).view(1, 5, 1).expand_as(mask)
        if tuple(raw.shape) != (8, 5, 20) or not torch.equal(mask, expected):
            raise RuntimeError("A3 reducer or typed-mask contract drifted")
        return raw, mask

    def common_prediction(
        self, task: A3Task, *, capture_diagnostics: bool = True
    ) -> tuple[Tensor, Tensor, int]:
        """Run the structural reference path; no action value is read."""

        output = self.common(
            task,
            mode="no_action",
            target_override=None,
            capture_diagnostics=True,
            run_action_diagnostic=False,
        )
        if not output.block_bridges:
            raise RuntimeError("R1 reference did not expose final common hidden")
        final_hidden = output.block_bridges[-1]
        prediction = self.common.video_output_projection(
            _parameter_free_rms_norm(final_hidden)
        ).unsqueeze(-1).unsqueeze(-1)
        if not capture_diagnostics:
            final_hidden = final_hidden.detach()
        return prediction, final_hidden, output.vendor_calls_this_forward

    def action_delta(
        self,
        task: A3Task,
        mode: ActionMode,
        *,
        count_calls: bool = True,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if mode not in ("correct", "shuffle", "no_action", "seam_disabled"):
            raise ValueError("unsupported A3 action mode")
        if self.action_residual is None:
            zero = torch.zeros(
                (self.spec.batch_size, 5, self.spec.latent_channels, 1, 1),
                dtype=task.noisy_video.dtype,
                device=task.noisy_video.device,
            )
            # Structural reference bypass: do not call the reducer and do not
            # read task.global_actions in any mode.
            return zero, None, None
        raw, mask = self._reduced_action_and_mask(task, mode)
        if mode in ("no_action", "seam_disabled"):
            zero = torch.zeros(
                (self.spec.batch_size, 5, self.spec.latent_channels, 1, 1),
                dtype=task.noisy_video.dtype,
                device=task.noisy_video.device,
            )
            return zero, raw, mask
        delta, _ = self.action_residual(raw, mask, count_calls=count_calls)
        return delta.unsqueeze(-1).unsqueeze(-1), raw, mask

    def forward(
        self,
        task: A3Task,
        mode: ActionMode = "correct",
        *,
        capture_diagnostics: bool = True,
    ) -> A3SequenceOutput:
        before_conditioner = (
            0
            if self.action_residual is None
            else self.action_residual.conditioner_call_count
        )
        before_output = (
            0
            if self.action_residual is None
            else self.action_residual.output_call_count
        )
        common, hidden, vendor_calls = self.common_prediction(
            task, capture_diagnostics=capture_diagnostics
        )
        delta, raw, mask = self.action_delta(task, mode)
        prediction = common + delta
        after_conditioner = (
            0
            if self.action_residual is None
            else self.action_residual.conditioner_call_count
        )
        after_output = (
            0
            if self.action_residual is None
            else self.action_residual.output_call_count
        )
        return A3SequenceOutput(
            video_prediction=prediction,
            common_prediction=common,
            action_delta=delta,
            final_common_hidden=hidden,
            raw_action=raw,
            action_present_mask=mask,
            vendor_calls_this_forward=vendor_calls,
            action_conditioner_calls_this_forward=(
                after_conditioner - before_conditioner
            ),
            action_output_calls_this_forward=after_output - before_output,
        )


def build_a3_synthetic_task(
    spec: A3BridgeSpec | None = None,
    *,
    device: torch.device | str = torch.device("cuda:0"),
) -> A3Task:
    return a2r1.build_av1b_vendor_task(spec, device=device)


def build_cach_a3_pair(
    spec: A3BridgeSpec | None = None,
    *,
    device: torch.device | str = torch.device("cuda:0"),
) -> A3BridgePair:
    spec = A3BridgeSpec() if spec is None else spec
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index != 0:
        raise ValueError("A3 is frozen to logical cuda:0")
    torch.manual_seed(a2r1.AV1B_SHARED_NAMED_SEED)
    torch.cuda.manual_seed_all(a2r1.AV1B_SHARED_NAMED_SEED)
    with torch.device("cpu"):
        common_template = a2r1.AV1BVendorGDNBridgeArm(
            spec=spec,
            staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
        )
    reference = A3BridgeArm(common=common_template, candidate=False)
    candidate = A3BridgeArm(
        common=copy.deepcopy(common_template), candidate=True
    )
    reference = reference.to(device=resolved, dtype=torch.float32)
    candidate = candidate.to(device=resolved, dtype=torch.float32)
    reference.train(True)
    candidate.train(True)
    reference_common = reference.common_trainable_names
    candidate_common = candidate.common_trainable_names
    candidate_action = candidate.action_trainable_names
    if reference_common != candidate_common:
        raise RuntimeError("A3 common trainable names differ across arms")
    if reference_common & candidate_action:
        raise RuntimeError("A3 common/action optimizer scopes overlap")
    ref_state = reference.common.state_dict()
    cand_state = candidate.common.state_dict()
    if ref_state.keys() != cand_state.keys() or any(
        not torch.equal(ref_state[name], cand_state[name]) for name in ref_state
    ):
        raise RuntimeError("A3 common theta0 bytes differ")
    if candidate.action_residual is None:
        raise RuntimeError("A3 candidate action residual is absent")
    if bool(candidate.action_residual.output_projection.weight.count_nonzero()):
        raise RuntimeError("A3 output action projection must be zero at theta0")
    theta0 = {
        "reference_common": _state_manifest(reference.common),
        "candidate_common": _state_manifest(candidate.common),
        "candidate_action": _state_manifest(candidate.action_residual),
    }
    return A3BridgePair(
        spec=spec,
        reference=reference,
        candidate=candidate,
        reference_common_trainable_names=reference_common,
        candidate_common_trainable_names=candidate_common,
        candidate_action_trainable_names=candidate_action,
        theta0_manifests=theta0,
    )


build_a3_pair = build_cach_a3_pair
ExactOddOutputResidual = ExactOddCausalActionResidual


def _require_config(config: Mapping[str, object]) -> None:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if config.get("schema") != A3_CONFIG_SCHEMA:
        raise ValueError("A3 config schema differs from the frozen interface")
    declared = config.get("architecture_id")
    architecture = config.get("architecture")
    if declared is None and isinstance(architecture, Mapping):
        declared = architecture.get("architecture_id")
    if declared is not None and declared != CACH_A3_ARCHITECTURE_ID:
        raise ValueError("A3 architecture id differs from the frozen interface")


def _paired_common(value: Tensor) -> Tensor:
    return 0.5 * (value[0::2] + value[1::2])


def _paired_half_delta(value: Tensor) -> Tensor:
    return 0.5 * (value[0::2] - value[1::2])


def _positive_mse(predicted: Tensor, expected: Tensor) -> Tensor:
    return (predicted - expected).square().mean()


def _nmse(predicted: Tensor, expected: Tensor) -> float:
    residual = (predicted.detach().double() - expected.detach().double()).square().sum()
    denominator = expected.detach().double().square().sum().clamp_min(1.0e-12)
    return float((residual / denominator).item())


def _energy_ratio(predicted: Tensor, expected: Tensor) -> float:
    numerator = predicted.detach().double().square().sum()
    denominator = expected.detach().double().square().sum().clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def _alignment_cosine(predicted: Tensor, expected: Tensor) -> float:
    left = predicted.detach().double()
    right = expected.detach().double()
    numerator = (left * right).sum()
    denominator = (
        left.square().sum().sqrt() * right.square().sum().sqrt()
    ).clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def _action_metric_slice(
    correct: Tensor,
    no_action: Tensor,
    shuffle: Tensor,
    target: Tensor,
) -> Mapping[str, float]:
    """Compute one registered action-only metric slice in float64."""

    correct64 = correct.detach().double()
    no_action64 = no_action.detach().double()
    shuffle64 = shuffle.detach().double()
    target64 = target.detach().double()
    predicted_delta = _paired_half_delta(correct64)
    target_delta = _paired_half_delta(target64)
    target_half_delta_energy = float(target_delta.square().mean().item())
    denominator = max(target_half_delta_energy, 1.0e-12)
    delta_mse = float((predicted_delta - target_delta).square().mean().item())
    predicted_energy = float(predicted_delta.square().mean().item())
    delta_dot = float((predicted_delta * target_delta).sum().item())
    predicted_sum = float(predicted_delta.square().sum().item())
    target_sum = float(target_delta.square().sum().item())
    correct_mse = float((correct64 - target64).square().mean().item())
    no_action_mse = float((no_action64 - target64).square().mean().item())
    shuffle_mse = float((shuffle64 - target64).square().mean().item())
    return {
        "target_half_delta_energy": target_half_delta_energy,
        "delta_mse": delta_mse,
        "delta_nmse": delta_mse / denominator,
        "delta_energy_ratio": predicted_energy / denominator,
        "delta_alignment_cosine": delta_dot
        / max(math.sqrt(predicted_sum * target_sum), 1.0e-12),
        "action_explained_fraction": (
            target_half_delta_energy - delta_mse
        )
        / denominator,
        "correct_mse": correct_mse,
        "no_action_mse": no_action_mse,
        "shuffle_mse": shuffle_mse,
        "no_action_recovery": (no_action_mse - correct_mse) / denominator,
        "shuffle_penalty": (shuffle_mse - correct_mse) / denominator,
    }


def _registered_action_metrics(
    correct: Tensor,
    no_action: Tensor,
    shuffle: Tensor,
    target: Tensor,
) -> Mapping[str, object]:
    return {
        "aggregate": _action_metric_slice(
            correct[:, 1:5],
            no_action[:, 1:5],
            shuffle[:, 1:5],
            target[:, 1:5],
        ),
        "by_horizon": {
            str(horizon): _action_metric_slice(
                correct[:, horizon : horizon + 1],
                no_action[:, horizon : horizon + 1],
                shuffle[:, horizon : horizon + 1],
                target[:, horizon : horizon + 1],
            )
            for horizon in (1, 2, 3, 4)
        },
    }


def _mse_by_horizon(predicted: Tensor, expected: Tensor) -> Mapping[str, float]:
    return {
        str(horizon): float(
            (predicted[:, horizon] - expected[:, horizon])
            .detach()
            .double()
            .square()
            .mean()
            .item()
        )
        for horizon in (1, 2, 3, 4)
    }


def _primary(mse: Mapping[str, float]) -> float:
    return 0.20 * mse["1"] + 0.30 * mse["2"] + 0.50 * mse["4"]


def _gap(altered: float, correct: float) -> float:
    return (altered - correct) / max(altered, 1.0e-12)


def _rms(values: list[Tensor], *, missing_numel: int = 0) -> float:
    square_sum = 0.0
    count = int(missing_numel)
    for value in values:
        detached = value.detach().double()
        square_sum += float(detached.square().sum().item())
        count += detached.numel()
    return math.sqrt(square_sum / count) if count else 0.0


def _numeric_leaves(value: object) -> tuple[float, ...]:
    leaves: list[float] = []

    def visit(item: object) -> None:
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float)):
            leaves.append(float(item))
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(leaves)


def _parameters(
    module: nn.Module, names: frozenset[str]
) -> tuple[list[str], list[nn.Parameter]]:
    named = dict(module.named_parameters())
    ordered = sorted(names, key=lambda value: value.encode("utf-8"))
    if any(name not in named for name in ordered):
        raise RuntimeError("registered optimizer parameter is absent")
    return ordered, [named[name] for name in ordered]


def _snapshot(
    names: list[str], parameters: list[nn.Parameter]
) -> Mapping[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in zip(names, parameters, strict=True)
    }


def _update_rms(
    names: list[str], parameters: list[nn.Parameter], initial: Mapping[str, Tensor]
) -> float:
    return _rms(
        [
            parameter.detach() - initial[name]
            for name, parameter in zip(names, parameters, strict=True)
        ]
    )


def _gradient_rms(parameters: list[nn.Parameter]) -> float:
    present = [parameter.grad for parameter in parameters if parameter.grad is not None]
    missing = sum(
        parameter.numel() for parameter in parameters if parameter.grad is None
    )
    return _rms([value for value in present if value is not None], missing_numel=missing)


def _all_none(gradients: tuple[Tensor | None, ...]) -> bool:
    return all(value is None for value in gradients)


def _set_cosine_lr(optimizer: torch.optim.Optimizer, step: int) -> float:
    if not 0 <= step < A3_TRAIN_STEPS:
        raise ValueError("cosine step is outside the frozen schedule")
    fraction = float(step) / float(A3_TRAIN_STEPS - 1)
    lr = A3_LR_END + 0.5 * (A3_LR_START - A3_LR_END) * (
        1.0 + math.cos(math.pi * fraction)
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _loss_statistics(trace: list[float]) -> Mapping[str, object]:
    if len(trace) != A3_TRAIN_STEPS + 1:
        raise ValueError("A3 loss trace must contain steps 0 through 200")

    def window(size: int) -> Mapping[str, float]:
        values = trace[-size:]
        ordered = sorted(values)
        middle = size // 2
        median = (
            ordered[middle]
            if size % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        return {
            "mean": sum(values) / float(size),
            "median": median,
            "p90_nearest_rank": ordered[math.ceil(0.9 * size) - 1],
            "min": ordered[0],
            "max": ordered[-1],
        }

    return {
        "last25": window(25),
        "last50": window(50),
        "trace_min": min(trace),
        "trace_min_step": trace.index(min(trace)),
        "final_over_trace_min_ratio": trace[-1] / max(min(trace), 1.0e-12),
        "last50_min": min(trace[-50:]),
        "final_over_last50_min_ratio": trace[-1]
        / max(min(trace[-50:]), 1.0e-12),
    }


def _raw_action_jvp(
    module: ExactOddCausalActionResidual,
    raw: Tensor,
    mask: Tensor,
) -> float:
    direction = torch.ones_like(raw)
    direction = direction * mask.to(dtype=raw.dtype)
    denominator = direction.double().square().sum().sqrt().clamp_min(1.0)
    direction = direction / denominator.to(dtype=direction.dtype)

    def function(value: Tensor) -> Tensor:
        return module(value, mask, count_calls=False)[0]

    _, tangent = torch.func.jvp(function, (raw,), (direction,))
    return float(tangent.detach().double().square().mean().sqrt().item())


def _emit(
    callback: Callable[[Mapping[str, object]], None] | None,
    event: str,
    payload: Mapping[str, object],
) -> None:
    if callback is not None:
        callback({"event": event, **dict(payload)})


def classify_cach_a3(
    config: Mapping[str, object],
    *,
    all_validity: bool,
    action_metrics: Mapping[str, object],
    common_stable: bool,
) -> tuple[str, str | None, Mapping[str, object]]:
    """Apply only the threshold and verdict values frozen in ``config``."""

    thresholds = config.get("thresholds")
    verdict = config.get("verdict")
    if not isinstance(thresholds, Mapping) or not isinstance(verdict, Mapping):
        raise ValueError("A3 config thresholds/verdict mappings are required")
    go = thresholds.get("operator_go")
    stop = thresholds.get("action_weak_stop")
    aggregate = action_metrics.get("aggregate")
    by_horizon = action_metrics.get("by_horizon")
    if not all(
        isinstance(value, Mapping)
        for value in (go, stop, aggregate, by_horizon)
    ):
        raise ValueError("A3 action threshold or metric mapping is malformed")
    assert isinstance(go, Mapping)
    assert isinstance(stop, Mapping)
    assert isinstance(aggregate, Mapping)
    assert isinstance(by_horizon, Mapping)

    aggregate_go_checks = {
        "delta_nmse": float(aggregate["delta_nmse"])
        <= float(go["delta_nmse_max"]),
        "delta_energy_ratio": float(go["delta_energy_ratio_min"])
        <= float(aggregate["delta_energy_ratio"])
        <= float(go["delta_energy_ratio_max"]),
        "delta_alignment_cosine": float(
            aggregate["delta_alignment_cosine"]
        )
        >= float(go["delta_alignment_cosine_min"]),
        "action_explained_fraction": float(
            aggregate["action_explained_fraction"]
        )
        >= float(go["action_explained_fraction_min"]),
        "no_action_recovery": float(aggregate["no_action_recovery"])
        >= float(go["no_action_recovery_min"]),
        "shuffle_penalty": float(aggregate["shuffle_penalty"])
        >= float(go["shuffle_penalty_min"]),
    }
    primary_go_checks: dict[str, Mapping[str, bool]] = {}
    for horizon in ("1", "2", "4"):
        one = by_horizon.get(horizon)
        if not isinstance(one, Mapping):
            raise ValueError(f"A3 horizon {horizon} action metrics are absent")
        primary_go_checks[horizon] = {
            "delta_nmse": float(one["delta_nmse"])
            <= float(go["delta_nmse_primary_horizon_max_each"]),
            "delta_energy_ratio": float(
                go["delta_energy_ratio_primary_horizon_min_each"]
            )
            <= float(one["delta_energy_ratio"])
            <= float(go["delta_energy_ratio_primary_horizon_max_each"]),
            "delta_alignment_cosine": float(
                one["delta_alignment_cosine"]
            )
            >= float(go["delta_alignment_cosine_primary_horizon_min_each"]),
            "action_explained_fraction": float(
                one["action_explained_fraction"]
            )
            >= float(
                go["action_explained_fraction_primary_horizon_min_each"]
            ),
            "no_action_recovery": float(one["no_action_recovery"])
            >= float(go["no_action_recovery_primary_horizon_min_each"]),
            "shuffle_penalty": float(one["shuffle_penalty"])
            >= float(go["shuffle_penalty_primary_horizon_min_each"]),
        }
    action_go = all(aggregate_go_checks.values()) and all(
        all(checks.values()) for checks in primary_go_checks.values()
    )
    stop_checks = {
        "delta_nmse": float(aggregate["delta_nmse"])
        >= float(stop["delta_nmse_min"]),
        "delta_alignment_cosine": float(
            aggregate["delta_alignment_cosine"]
        )
        <= float(stop["delta_alignment_cosine_max"]),
        "action_explained_fraction": float(
            aggregate["action_explained_fraction"]
        )
        <= float(stop["action_explained_fraction_max"]),
        "shuffle_penalty": float(aggregate["shuffle_penalty"])
        <= float(stop["shuffle_penalty_max"]),
    }
    action_weak_stop = bool(stop.get("all_required")) and all(
        stop_checks.values()
    )
    inputs: Mapping[str, object] = {
        "all_validity": all_validity,
        "aggregate_go_checks": aggregate_go_checks,
        "primary_horizon_go_checks": primary_go_checks,
        "action_go": action_go,
        "action_weak_stop_checks": stop_checks,
        "action_weak_stop": action_weak_stop,
        "common_stable": common_stable,
        "operator_go_common_stable": (
            all_validity and action_go and common_stable
        ),
        "operator_go_common_blocked": (
            all_validity and action_go and not common_stable
        ),
    }
    if not all_validity:
        return str(verdict["invalid"]), "A3_VALIDITY_FAILURE", inputs
    if action_go and common_stable:
        return str(verdict["valid_go_common_stable"]), None, inputs
    if action_go:
        return (
            str(verdict["valid_go_common_blocked"]),
            "A3_ACTION_GO_COMMON_BLOCKED",
            inputs,
        )
    if action_weak_stop:
        return (
            str(verdict["valid_action_weak_stop"]),
            "A3_ACTION_WEAK",
            inputs,
        )
    return (
        str(verdict["valid_other"]),
        "A3_ACTION_NEITHER_GO_NOR_WEAK_STOP",
        inputs,
    )


def run_cach_a3_screen(
    config: Mapping[str, object],
    *,
    expected_gpu_uuid: str,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> Mapping[str, object]:
    """Run the 200-step separated-positive-loss A3 synthetic screen."""

    started = time.monotonic()
    _require_config(config)
    if not isinstance(expected_gpu_uuid, str) or not expected_gpu_uuid.startswith(
        "GPU-"
    ):
        raise ValueError("expected_gpu_uuid must be a physical GPU UUID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from execution binding")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A3 requires exactly one visible CUDA device")
    if os.environ.get("FUSED_GDN_PRECISION") != "0":
        raise RuntimeError("A3 requires FUSED_GDN_PRECISION=0")

    device = torch.device("cuda:0")
    spec = A3BridgeSpec()
    task = build_a3_synthetic_task(spec, device=device)
    pair = build_cach_a3_pair(spec, device=device)
    target = task.video_target.detach()
    common_target = _paired_common(target)
    delta_target = _paired_half_delta(target)

    ref_names, ref_parameters = _parameters(
        pair.reference, pair.reference_common_trainable_names
    )
    cand_common_names, cand_common_parameters = _parameters(
        pair.candidate, pair.candidate_common_trainable_names
    )
    action_names, action_parameters = _parameters(
        pair.candidate, pair.candidate_action_trainable_names
    )
    if set(ref_names) != set(cand_common_names):
        raise RuntimeError("reference/candidate common optimizer names differ")
    if set(cand_common_names) & set(action_names):
        raise RuntimeError("common and action optimizer scopes overlap")

    ref_initial = _snapshot(ref_names, ref_parameters)
    cand_common_initial = _snapshot(cand_common_names, cand_common_parameters)
    action_initial = _snapshot(action_names, action_parameters)

    with torch.no_grad():
        theta0_reference = pair.reference(task, mode="correct")
        theta0_candidate = pair.candidate(task, mode="correct")
        theta0_no_action = pair.candidate(task, mode="no_action")
    theta0_common_equal = torch.equal(
        theta0_reference.common_prediction,
        theta0_candidate.common_prediction,
    )
    theta0_candidate_identity = torch.equal(
        theta0_candidate.video_prediction,
        theta0_candidate.common_prediction,
    )
    if pair.candidate.action_residual is None:
        raise RuntimeError("candidate action residual is absent")
    theta0_raw = theta0_candidate.raw_action
    theta0_mask = theta0_candidate.action_present_mask
    if theta0_raw is None or theta0_mask is None:
        raise RuntimeError("candidate theta0 action diagnostic is absent")
    theta0_action_jvp = _raw_action_jvp(
        pair.candidate.action_residual, theta0_raw, theta0_mask
    )

    # Cross-scope gradients are checked before either optimizer updates.
    candidate_common_for_cross, _, _ = pair.candidate.common_prediction(task)
    cross_common_loss = _positive_mse(
        _paired_common(candidate_common_for_cross), common_target
    )
    common_to_action = torch.autograd.grad(
        cross_common_loss,
        action_parameters,
        allow_unused=True,
        materialize_grads=False,
    )
    candidate_delta_for_cross, _, _ = pair.candidate.action_delta(
        task, "correct", count_calls=False
    )
    cross_delta_loss = _positive_mse(
        _paired_half_delta(candidate_delta_for_cross), delta_target
    )
    delta_to_common = torch.autograd.grad(
        cross_delta_loss,
        cand_common_parameters,
        allow_unused=True,
        materialize_grads=False,
    )
    separated_gradient_scopes = _all_none(common_to_action) and _all_none(
        delta_to_common
    )

    reference_optimizer = torch.optim.AdamW(
        ref_parameters,
        lr=A3_LR_START,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    candidate_common_optimizer = torch.optim.AdamW(
        cand_common_parameters,
        lr=A3_LR_START,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    action_optimizer = torch.optim.AdamW(
        action_parameters,
        lr=A3_LR_START,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    traces: dict[str, list[float]] = {
        "reference_common": [],
        "candidate_common": [],
        "candidate_delta": [],
    }
    step0_gradients: dict[str, float] = {}
    lr_trace: list[float] = []

    for step in range(A3_TRAIN_STEPS):
        lr = _set_cosine_lr(reference_optimizer, step)
        _set_cosine_lr(candidate_common_optimizer, step)
        _set_cosine_lr(action_optimizer, step)
        lr_trace.append(lr)

        reference_optimizer.zero_grad(set_to_none=True)
        reference_common, _, _ = pair.reference.common_prediction(task)
        reference_loss = _positive_mse(
            _paired_common(reference_common), common_target
        )
        traces["reference_common"].append(float(reference_loss.detach().item()))
        reference_loss.backward()
        if step == 0:
            step0_gradients["reference_common"] = _gradient_rms(ref_parameters)
        ref_norm = torch.nn.utils.clip_grad_norm_(ref_parameters, 1.0)
        if not bool(torch.isfinite(ref_norm)):
            raise FloatingPointError("reference common gradient is non-finite")
        reference_optimizer.step()

        candidate_common_optimizer.zero_grad(set_to_none=True)
        candidate_common, _, _ = pair.candidate.common_prediction(task)
        candidate_common_loss = _positive_mse(
            _paired_common(candidate_common), common_target
        )
        traces["candidate_common"].append(
            float(candidate_common_loss.detach().item())
        )
        candidate_common_loss.backward()
        if step == 0:
            step0_gradients["candidate_common"] = _gradient_rms(
                cand_common_parameters
            )
        common_norm = torch.nn.utils.clip_grad_norm_(cand_common_parameters, 1.0)
        if not bool(torch.isfinite(common_norm)):
            raise FloatingPointError("candidate common gradient is non-finite")
        candidate_common_optimizer.step()

        action_optimizer.zero_grad(set_to_none=True)
        candidate_delta, _, _ = pair.candidate.action_delta(task, "correct")
        candidate_delta_loss = _positive_mse(
            _paired_half_delta(candidate_delta), delta_target
        )
        traces["candidate_delta"].append(
            float(candidate_delta_loss.detach().item())
        )
        candidate_delta_loss.backward()
        if step == 0:
            step0_gradients["candidate_action"] = _gradient_rms(
                action_parameters
            )
        action_norm = torch.nn.utils.clip_grad_norm_(action_parameters, 1.0)
        if not bool(torch.isfinite(action_norm)):
            raise FloatingPointError("candidate action gradient is non-finite")
        action_optimizer.step()

        if step in (0, 49, 99, 149, 199):
            _emit(
                progress_callback,
                "optimizer_progress",
                {
                    "optimizer_steps_completed": step + 1,
                    "lr": lr,
                    "reference_common_loss": traces["reference_common"][-1],
                    "candidate_common_loss": traces["candidate_common"][-1],
                    "candidate_delta_loss": traces["candidate_delta"][-1],
                },
            )

    # The final point is post-step-200 and is never used for selection.
    final_reference_common, _, _ = pair.reference.common_prediction(task)
    final_candidate_common, _, _ = pair.candidate.common_prediction(task)
    final_candidate_delta, _, _ = pair.candidate.action_delta(
        task, "correct", count_calls=False
    )
    traces["reference_common"].append(
        float(
            _positive_mse(
                _paired_common(final_reference_common), common_target
            ).detach().item()
        )
    )
    traces["candidate_common"].append(
        float(
            _positive_mse(
                _paired_common(final_candidate_common), common_target
            ).detach().item()
        )
    )
    traces["candidate_delta"].append(
        float(
            _positive_mse(
                _paired_half_delta(final_candidate_delta), delta_target
            ).detach().item()
        )
    )

    with torch.no_grad():
        reference_correct = pair.reference(task, mode="correct")
        candidate_correct = pair.candidate(task, mode="correct")
        candidate_shuffle = pair.candidate(task, mode="shuffle")
        candidate_no_action = pair.candidate(task, mode="no_action")
        candidate_disabled = pair.candidate(task, mode="seam_disabled")
        raw = candidate_correct.raw_action
        mask = candidate_correct.action_present_mask
        if raw is None or mask is None or pair.candidate.action_residual is None:
            raise RuntimeError("final A3 action diagnostic is absent")
        direct_positive, _ = pair.candidate.action_residual(
            raw, mask, count_calls=False
        )
        direct_negative, _ = pair.candidate.action_residual(
            -raw, mask, count_calls=False
        )
        direct_zero, _ = pair.candidate.action_residual(
            torch.zeros_like(raw), mask, count_calls=False
        )

    final_action_jvp = _raw_action_jvp(
        pair.candidate.action_residual, raw, mask
    )
    odd_sum = direct_positive + direct_negative
    oddness_max_abs = float(odd_sum.abs().max().item())
    oddness_rms = float(odd_sum.double().square().mean().sqrt().item())
    zero_origin_exact = not bool(direct_zero.count_nonzero())
    common_parity = (
        torch.equal(
            candidate_correct.common_prediction,
            candidate_shuffle.common_prediction,
        )
        and torch.equal(
            candidate_correct.common_prediction,
            candidate_no_action.common_prediction,
        )
        and torch.equal(
            candidate_correct.common_prediction,
            candidate_disabled.common_prediction,
        )
    )
    no_action_bypass = torch.equal(
        candidate_no_action.video_prediction,
        candidate_disabled.video_prediction,
    ) and torch.equal(
        candidate_no_action.video_prediction,
        candidate_no_action.common_prediction,
    )
    pair_common_reconstruction_error = (
        _paired_common(candidate_correct.video_prediction)
        - _paired_common(candidate_correct.common_prediction)
    )
    pair_common_reconstruction_max_abs = float(
        pair_common_reconstruction_error.abs().max().item()
    )

    final_common_prediction_difference_max_abs = float(
        (final_reference_common.detach() - final_candidate_common.detach())
        .abs()
        .max()
        .item()
    )
    reference_named = dict(pair.reference.named_parameters())
    candidate_named = dict(pair.candidate.named_parameters())
    common_parameter_differences = [
        (
            reference_named[name].detach()
            - candidate_named[name].detach()
        )
        for name in ref_names
    ]
    final_common_parameters_bitwise_equal = all(
        torch.equal(
            reference_named[name].detach(),
            candidate_named[name].detach(),
        )
        for name in ref_names
    )
    final_common_parameter_difference_max_abs = max(
        (
            float(value.abs().max().item())
            for value in common_parameter_differences
            if value.numel()
        ),
        default=0.0,
    )

    action_metrics = _registered_action_metrics(
        candidate_correct.video_prediction,
        candidate_no_action.video_prediction,
        candidate_shuffle.video_prediction,
        target,
    )
    action_aggregate = action_metrics["aggregate"]
    if not isinstance(action_aggregate, Mapping):
        raise RuntimeError("aggregate A3 action metrics are malformed")
    delta_nmse = float(action_aggregate["delta_nmse"])
    delta_energy = float(action_aggregate["delta_energy_ratio"])
    delta_cosine = float(action_aggregate["delta_alignment_cosine"])
    predicted_common = _paired_common(candidate_correct.video_prediction)
    common_nmse = _nmse(predicted_common[:, 1:5], common_target[:, 1:5])
    common_mse = float(
        _positive_mse(
            predicted_common[:, 1:5], common_target[:, 1:5]
        ).detach().item()
    )

    predictions = {
        "reference": reference_correct.video_prediction.detach(),
        "correct": candidate_correct.video_prediction.detach(),
        "shuffle": candidate_shuffle.video_prediction.detach(),
        "no_action": candidate_no_action.video_prediction.detach(),
    }
    mse = {name: _mse_by_horizon(value, target) for name, value in predictions.items()}
    primary = {name: _primary(value) for name, value in mse.items()}
    target_energy = float(target[:, 1:5].double().square().mean().item())
    normalized_explained = {
        name: 1.0
        - float(
            (value[:, 1:5] - target[:, 1:5])
            .double()
            .square()
            .mean()
            .item()
        )
        / max(target_energy, 1.0e-12)
        for name, value in predictions.items()
    }

    loss_stability = {
        name: _loss_statistics(trace) for name, trace in traces.items()
    }
    final_common_mse = max(
        traces["reference_common"][-1],
        traces["candidate_common"][-1],
    )
    final_common_over_last50_min = max(
        float(
            loss_stability["reference_common"][
                "final_over_last50_min_ratio"
            ]
        ),
        float(
            loss_stability["candidate_common"][
                "final_over_last50_min_ratio"
            ]
        ),
    )
    threshold_root = config.get("thresholds")
    if not isinstance(threshold_root, Mapping):
        raise ValueError("A3 threshold mapping is required")
    validity_thresholds = threshold_root.get("validity")
    common_stability_thresholds = threshold_root.get("common_stability")
    if not isinstance(validity_thresholds, Mapping) or not isinstance(
        common_stability_thresholds, Mapping
    ):
        raise ValueError("A3 validity/common-stability thresholds are required")
    common_stable = (
        final_common_mse
        <= float(common_stability_thresholds["final_common_mse_max"])
        and final_common_over_last50_min
        <= float(
            common_stability_thresholds["final_over_last50_min_max"]
        )
    )
    zero_origin_max_abs = float(direct_zero.abs().max().item())
    inactive_delta_max_abs = float(
        candidate_no_action.action_delta.detach().abs().max().item()
    )
    no_action_jvp = 0.0
    action_update_rms = _update_rms(
        action_names, action_parameters, action_initial
    )
    finite = all(
        bool(torch.isfinite(value).all()) for value in predictions.values()
    ) and all(
        math.isfinite(value)
        for value in (
            common_nmse,
            common_mse,
            oddness_max_abs,
            pair_common_reconstruction_max_abs,
            final_common_prediction_difference_max_abs,
            final_common_parameter_difference_max_abs,
            final_common_mse,
            final_common_over_last50_min,
            theta0_action_jvp,
            final_action_jvp,
            *_numeric_leaves(action_metrics),
            *[item for trace in traces.values() for item in trace],
        )
    )
    checks: dict[str, bool] = {
        "architecture_id_exact": True,
        "reference_structural_action_bypass": (
            reference_correct.raw_action is None
            and reference_correct.action_present_mask is None
            and reference_correct.action_conditioner_calls_this_forward == 0
            and reference_correct.action_output_calls_this_forward == 0
        ),
        "common_trunk_mode_parity_bitwise": common_parity,
        "final_reference_candidate_common_prediction_parity": (
            final_common_prediction_difference_max_abs
            <= float(
                validity_thresholds[
                    "common_mode_prediction_difference_max_abs"
                ]
            )
        ),
        "final_reference_candidate_common_parameter_parity": (
            final_common_parameter_difference_max_abs
            <= float(
                validity_thresholds[
                    "common_mode_prediction_difference_max_abs"
                ]
            )
        ),
        "theta0_common_bytes_and_outputs_equal": theta0_common_equal,
        "theta0_candidate_output_equals_normalized_common": (
            theta0_candidate_identity
        ),
        "exact_odd_direct_output_within_fp32_tolerance": (
            oddness_max_abs
            <= float(validity_thresholds["oddness_max_abs"])
        ),
        "pair_common_reconstruction_within_fp32_tolerance": (
            pair_common_reconstruction_max_abs
            <= float(
                validity_thresholds[
                    "common_mode_prediction_difference_max_abs"
                ]
            )
        ),
        "zero_origin_action_output_exact_zero": (
            zero_origin_exact
            and zero_origin_max_abs
            <= float(validity_thresholds["zero_origin_max_abs"])
        ),
        "inactive_action_delta_exact_zero": (
            inactive_delta_max_abs
            <= float(validity_thresholds["inactive_delta_max_abs"])
        ),
        "no_action_and_seam_disabled_direct_bypass_bitwise": no_action_bypass,
        "no_action_action_module_calls_zero": (
            candidate_no_action.action_conditioner_calls_this_forward == 0
            and candidate_no_action.action_output_calls_this_forward == 0
            and candidate_disabled.action_conditioner_calls_this_forward == 0
            and candidate_disabled.action_output_calls_this_forward == 0
        ),
        "common_and_action_gradient_scopes_disjoint": separated_gradient_scopes,
        "three_optimizers_disjoint_and_exact_200_steps": (
            len(lr_trace)
            == int(
                validity_thresholds[
                    "exact_macrosteps_candidate_action"
                ]
            )
            == int(
                validity_thresholds[
                    "exact_macrosteps_each_common_arm"
                ]
            )
            and abs(lr_trace[0] - A3_LR_START) <= 1.0e-15
            and abs(lr_trace[-1] - A3_LR_END) <= 1.0e-15
        ),
        "action_parameter_update_live": (
            action_update_rms
            > float(
                validity_thresholds[
                    "seam_update_rms_final_strictly_greater_than"
                ]
            )
        ),
        "action_gradient_step0_live": (
            step0_gradients["candidate_action"]
            > float(
                validity_thresholds[
                    "seam_gradient_rms_step0_strictly_greater_than"
                ]
            )
        ),
        "action_raw_jvp_final_live": (
            final_action_jvp
            > float(
                validity_thresholds[
                    "action_jvp_rms_final_strictly_greater_than"
                ]
            )
        ),
        "no_action_raw_jvp_exact_zero": (
            no_action_jvp
            == float(validity_thresholds["no_action_jvp_rms"])
        ),
        "all_outputs_losses_metrics_finite": finite,
        "target_forward_leakage_absent_by_signature": True,
        "positive_separated_losses_only": True,
        "checkpoint_real_data_loss_subtraction_absent": True,
    }
    reasons = sorted(name for name, passed in checks.items() if not passed)
    all_validity = not reasons
    typed_verdict, diagnostic_subclassification, classification_inputs = (
        classify_cach_a3(
            config,
            all_validity=all_validity,
            action_metrics=action_metrics,
            common_stable=common_stable,
        )
    )
    metrics: Mapping[str, object] = {
        "action": action_metrics,
        "counterfactual_delta_nmse": float(
            action_aggregate["delta_nmse"]
        ),
        "target_half_delta_energy": float(
            action_aggregate["target_half_delta_energy"]
        ),
        "delta_energy_ratio": float(
            action_aggregate["delta_energy_ratio"]
        ),
        "delta_alignment_cosine": float(
            action_aggregate["delta_alignment_cosine"]
        ),
        "action_explained_fraction": float(
            action_aggregate["action_explained_fraction"]
        ),
        "no_action_recovery": float(
            action_aggregate["no_action_recovery"]
        ),
        "shuffle_penalty": float(action_aggregate["shuffle_penalty"]),
        "action_by_horizon": action_metrics["by_horizon"],
        "mse_by_horizon": mse,
        "primary": primary,
        "candidate_gain_vs_reference": (
            (primary["reference"] - primary["correct"])
            / max(primary["reference"], 1.0e-12)
        ),
        "legacy_total_relative_gaps_report_only": {
            "candidate_gain_vs_reference": (
                (primary["reference"] - primary["correct"])
                / max(primary["reference"], 1.0e-12)
            ),
            "shuffle_gap_rel": _gap(
                primary["shuffle"], primary["correct"]
            ),
            "no_action_gap_rel": _gap(
                primary["no_action"], primary["correct"]
            ),
            "used_for_classification": False,
        },
        "shuffle_gap_rel": _gap(primary["shuffle"], primary["correct"]),
        "no_action_gap_rel": _gap(primary["no_action"], primary["correct"]),
        "normalized_explained": normalized_explained,
        "paired_common_mse": common_mse,
        "paired_common_nmse": common_nmse,
        "paired_half_delta_nmse": delta_nmse,
        "paired_half_delta_energy_ratio": delta_energy,
        "paired_half_delta_alignment_cosine": delta_cosine,
        "oddness_max_abs": oddness_max_abs,
        "oddness_rms": oddness_rms,
        "pair_common_reconstruction_max_abs": pair_common_reconstruction_max_abs,
        "final_reference_candidate_common_prediction_difference_max_abs": (
            final_common_prediction_difference_max_abs
        ),
        "final_reference_candidate_common_parameter_difference_max_abs": (
            final_common_parameter_difference_max_abs
        ),
        "final_reference_candidate_common_parameters_bitwise_equal": (
            final_common_parameters_bitwise_equal
        ),
        "final_common_mse_for_stability": final_common_mse,
        "final_common_over_last50_min_ratio": (
            final_common_over_last50_min
        ),
        "common_stable": common_stable,
        "action_raw_jvp_rms_theta0": theta0_action_jvp,
        "action_raw_jvp_rms_step200": final_action_jvp,
        "gradient_rms_step0": step0_gradients,
        "update_rms_final": {
            "reference_common": _update_rms(
                ref_names, ref_parameters, ref_initial
            ),
            "candidate_common": _update_rms(
                cand_common_names, cand_common_parameters, cand_common_initial
            ),
            "candidate_action": action_update_rms,
        },
        "common_stability": loss_stability,
        "cosine_lr": {
            "start": lr_trace[0],
            "final": lr_trace[-1],
            "steps": len(lr_trace),
        },
    }
    jit_inventory = a2r1._screen_jit_inventory(
        pair.reference.common, pair.candidate.common
    )
    diagnostics: Mapping[str, object] = {
        "architecture_id": CACH_A3_ARCHITECTURE_ID,
        "common_path": (
            "R1_STRUCTURAL_REFERENCE_FINAL_HIDDEN_PARAMETER_FREE_RMSNORM_"
            "VIDEO_OUTPUT_PROJECTION"
        ),
        "action_path": (
            "TYPED_END_OF_BIN_20_64_20_EXACT_ODD_20_3_OUTPUT_ONLY"
        ),
        "post_output_add_operation": None,
        "reference_action_tensor_read": False,
        "common_mode_reads_action": False,
        "target_is_forward_argument": False,
        "losses": ["POSITIVE_L_COMMON", "POSITIVE_L_DELTA"],
        "optimizer_scopes": {
            "reference_common": ref_names,
            "candidate_common": cand_common_names,
            "candidate_action": action_names,
            "pairwise_disjoint": True,
        },
        "theta0_action_output_projection_exact_zero": theta0_candidate_identity,
        "no_action_direct_bypass": no_action_bypass,
        "common_parity": common_parity,
        "final_common_parity": {
            "prediction_difference_max_abs": (
                final_common_prediction_difference_max_abs
            ),
            "parameters_bitwise_equal": (
                final_common_parameters_bitwise_equal
            ),
            "parameter_difference_max_abs": (
                final_common_parameter_difference_max_abs
            ),
        },
        "common_stability": {
            "final_common_mse": final_common_mse,
            "final_over_last50_min_ratio": (
                final_common_over_last50_min
            ),
            "stable": common_stable,
            "affects_only_go_subtype": True,
        },
        "classification_threshold_source": "config.thresholds",
        "legacy_total_relative_gaps_used_for_classification": False,
        "oddness": {
            "formula": "0.5*(g(a)-g(-a))",
            "max_abs_delta_plus_delta_negated": oddness_max_abs,
            "zero_origin_exact": zero_origin_exact,
        },
    }
    final_parameter_digests: Mapping[str, object] = {
        "reference_common": _state_manifest(pair.reference.common),
        "candidate_common": _state_manifest(pair.candidate.common),
        "candidate_action": _state_manifest(pair.candidate.action_residual),
    }
    elapsed = time.monotonic() - started
    resource_usage: Mapping[str, object] = {
        "elapsed_seconds_screen": elapsed,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if platform.system() == "Darwin" else 1024),
        "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "checkpoint_loaded": False,
        "checkpoint_saved": False,
        "real_data_used": False,
        "loss_subtraction_used": False,
    }
    return {
        "schema": A3_SCREEN_SCHEMA,
        "architecture_id": CACH_A3_ARCHITECTURE_ID,
        "metrics": metrics,
        "evidence": diagnostics,
        "validity": {
            "all_card_validity_checks_pass": all_validity,
            "checks": checks,
            "reasons": reasons,
        },
        "diagnostics": diagnostics,
        "classification_inputs": classification_inputs,
        "typed_verdict": typed_verdict,
        "diagnostic_subclassification": diagnostic_subclassification,
        "theta0_manifests": pair.theta0_manifests,
        "loss_trace": traces,
        "final_parameter_digests": final_parameter_digests,
        "jit_inventory": jit_inventory,
        "resource_usage": resource_usage,
        "elapsed_seconds_total": elapsed,
    }


__all__ = [
    "A3_ARCHITECTURE_ID",
    "A3_CONFIG_SCHEMA",
    "A3_SCREEN_SCHEMA",
    "A3BridgeArm",
    "A3BridgePair",
    "A3BridgeSpec",
    "A3SequenceOutput",
    "A3Task",
    "CACH_A3_ARCHITECTURE_ID",
    "ExactOddCausalActionResidual",
    "ExactOddOutputResidual",
    "build_a3_pair",
    "build_cach_a3_pair",
    "build_a3_synthetic_task",
    "classify_cach_a3",
    "run_cach_a3_screen",
]
