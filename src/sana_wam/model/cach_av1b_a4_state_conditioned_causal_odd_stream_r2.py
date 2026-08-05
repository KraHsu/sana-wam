"""CACH-A4 state-conditioned causal exact-odd delta stream.

The frozen A3 common/task implementation is reused without modification.
Both arms run the A3 normalized, action-blind common path.  The candidate
adds one parallel action stream which reads a detached common hidden state,
normalizes it without parameters, and computes an exact-odd causal delta.
The delta is added only after the common video output; no later model
operator is permitted.

This remains a synthetic reduced architecture screen.  No forward method
accepts a target, checkpoint, real-data handle, or filesystem path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass, field
import hashlib
import math
import os
import platform
import resource
import time
from typing import Final, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from sana_wam.cach.action_conditioning import reduce_end_of_bin_action_condition
from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model import cach_av1b_a3_orthogonal_odd_residual as a3


CACH_A4_ARCHITECTURE_ID: Final = (
    "CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1"
)
A4_CONFIG_SCHEMA: Final = "cach.cach_a4.state_conditioned_causal_odd_stream.config.v1"
A4_SCREEN_SCHEMA: Final = "cach.cach_a4.state_conditioned_causal_odd_stream.screen.v1"
A4_INITIALIZER_REVISION: Final = "cach-a4-state-conditioned-causal-odd-v1"
A4_CANDIDATE_ONLY_SEED: Final = 2026080522
A4_TRAIN_STEPS: Final = 200
A4_LR_START: Final = 3.0e-3
A4_LR_END: Final = 3.0e-5
A4_SHUFFLE_PERMUTATION: Final = a3.A3_SHUFFLE_PERMUTATION

ActionMode = Literal["correct", "shuffle", "no_action", "seam_disabled"]
A4BridgeSpec = a3.A3BridgeSpec
A4Task = a3.A3Task


@dataclass(frozen=True)
class A4ActionStreamOutput:
    delta: Tensor = field(repr=False, compare=False)
    odd_input: Tensor = field(repr=False, compare=False)
    write: Tensor = field(repr=False, compare=False)
    recurrent_state: Tensor = field(repr=False, compare=False)
    decay: Tensor = field(repr=False, compare=False)


@dataclass(frozen=True)
class A4SequenceOutput:
    video_prediction: Tensor = field(repr=False, compare=False)
    common_prediction: Tensor = field(repr=False, compare=False)
    action_delta: Tensor = field(repr=False, compare=False)
    final_common_hidden: Tensor = field(repr=False, compare=False)
    normalized_detached_common_state: Tensor = field(repr=False, compare=False)
    raw_action: Tensor | None = field(repr=False, compare=False)
    action_present_mask: Tensor | None = field(repr=False, compare=False)
    vendor_calls_this_forward: int
    reducer_calls_this_forward: int
    action_stream_calls_this_forward: int


@dataclass(frozen=True)
class A4BridgePair:
    spec: A4BridgeSpec
    reference: "A4BridgeArm"
    candidate: "A4BridgeArm"
    reference_common_trainable_names: frozenset[str]
    candidate_common_trainable_names: frozenset[str]
    candidate_action_trainable_names: frozenset[str]
    theta0_manifests: Mapping[str, object]


def _seed_for_name(name: str) -> int:
    material = (f"{A4_INITIALIZER_REVISION}:{A4_CANDIDATE_ONLY_SEED}:{name}").encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


class StateConditionedCausalOddActionStream(nn.Module):
    """Strictly causal linear recurrence with state-only sigmoid decay.

    ``g(s, a)`` is constructed from bias-free action and common-state
    projections.  The injected feature is exactly
    ``0.5 * (g(s, a) - g(s, -a))``.  The recurrence

    ``carry[t] = decay(s[t]) * carry[t-1] + mask[t] * W_write(odd[t])``

    is linear in the prior carry and odd action feature.  The decay reads no
    action tensor.  The caller is responsible for detaching the common state;
    normalization is performed here so standalone local state JVPs can still
    verify that the functional state dependence is live.
    """

    def __init__(self, spec: A4BridgeSpec | None = None) -> None:
        super().__init__()
        self.spec = A4BridgeSpec() if spec is None else spec
        self.action_projection = nn.Linear(
            self.spec.action_dim,
            self.spec.hidden_dim,
            bias=False,
            device="cpu",
            dtype=torch.float32,
        )
        self.state_projection = nn.Linear(
            self.spec.hidden_dim,
            self.spec.hidden_dim,
            bias=False,
            device="cpu",
            dtype=torch.float32,
        )
        self.decay_projection = nn.Linear(
            self.spec.hidden_dim,
            self.spec.hidden_dim,
            bias=False,
            device="cpu",
            dtype=torch.float32,
        )
        self.write_projection = nn.Linear(
            self.spec.hidden_dim,
            self.spec.hidden_dim,
            bias=False,
            device="cpu",
            dtype=torch.float32,
        )
        self.output_projection = nn.Linear(
            self.spec.hidden_dim,
            self.spec.latent_channels,
            bias=False,
            device="cpu",
            dtype=torch.float32,
        )
        self.call_count = 0
        self._initialize()

    def _initialize(self) -> None:
        with torch.no_grad():
            for name, parameter in sorted(
                self.named_parameters(),
                key=lambda item: item[0].encode("utf-8"),
            ):
                if name in {
                    "decay_projection.weight",
                    "output_projection.weight",
                }:
                    parameter.zero_()
                    continue
                generator = torch.Generator(device="cpu")
                generator.manual_seed(_seed_for_name(name))
                bound = 0.25 / math.sqrt(float(parameter.shape[-1]))
                parameter.uniform_(-bound, bound, generator=generator)

    def forward(
        self,
        raw_action: Tensor,
        common_state: Tensor,
        action_present_mask: Tensor,
        *,
        count_calls: bool = True,
    ) -> A4ActionStreamOutput:
        if raw_action.ndim != 3 or raw_action.shape[-1] != self.spec.action_dim:
            raise ValueError("raw_action must be [batch,time,20]")
        if common_state.ndim != 3 or common_state.shape[-1] != self.spec.hidden_dim:
            raise ValueError("common_state must be [batch,time,64]")
        if tuple(common_state.shape[:2]) != tuple(raw_action.shape[:2]):
            raise ValueError("action and common-state axes differ")
        if tuple(action_present_mask.shape) != (*raw_action.shape[:2], 1):
            raise ValueError("typed mask must be [batch,time,1]")
        if action_present_mask.dtype is not torch.bool:
            raise TypeError("typed mask must be bool")
        if raw_action.dtype != common_state.dtype:
            raise TypeError("action and common state dtypes differ")

        state = a3._parameter_free_rms_norm(common_state)
        decay = torch.sigmoid(self.decay_projection(state))

        carry = torch.zeros(
            (raw_action.shape[0], self.spec.hidden_dim),
            dtype=raw_action.dtype,
            device=raw_action.device,
        )
        odd_inputs: list[Tensor] = []
        writes: list[Tensor] = []
        states: list[Tensor] = []
        deltas: list[Tensor] = []
        for time_index in range(raw_action.shape[1]):
            active_indices = torch.nonzero(
                action_present_mask[:, time_index, 0], as_tuple=False
            ).flatten()
            odd_at_time = torch.zeros_like(carry)
            write_at_time = torch.zeros_like(carry)
            if active_indices.numel() != 0:
                active_action = raw_action[:, time_index].index_select(
                    0, active_indices
                )
                active_state = state[:, time_index].index_select(0, active_indices)
                state_term = self.state_projection(active_state)
                positive = F.silu(self.action_projection(active_action) + state_term)
                negative = F.silu(self.action_projection(-active_action) + state_term)
                active_odd = 0.5 * (positive - negative)
                active_write = self.write_projection(active_odd)
                odd_at_time = odd_at_time.index_copy(0, active_indices, active_odd)
                write_at_time = write_at_time.index_copy(
                    0, active_indices, active_write
                )
            carry = decay[:, time_index] * carry + write_at_time
            odd_inputs.append(odd_at_time)
            writes.append(write_at_time)
            states.append(carry)
            projected = self.output_projection(carry)
            deltas.append(
                torch.where(
                    action_present_mask[:, time_index],
                    projected,
                    torch.zeros_like(projected),
                )
            )
        if count_calls:
            self.call_count += 1
        return A4ActionStreamOutput(
            delta=torch.stack(deltas, dim=1),
            odd_input=torch.stack(odd_inputs, dim=1),
            write=torch.stack(writes, dim=1),
            recurrent_state=torch.stack(states, dim=1),
            decay=decay,
        )


class A4BridgeArm(nn.Module):
    """Frozen-A3 common arm plus an optional A4 delta stream."""

    def __init__(
        self,
        *,
        common: nn.Module,
        candidate: bool,
        spec: A4BridgeSpec | None = None,
    ) -> None:
        super().__init__()
        resolved_spec = getattr(common, "spec", None) if spec is None else spec
        if not isinstance(resolved_spec, A4BridgeSpec):
            raise TypeError("A4 common arm must expose the frozen A3 spec")
        self.spec = resolved_spec
        self.common = common
        self.candidate = bool(candidate)
        self.action_stream = (
            StateConditionedCausalOddActionStream(self.spec) if self.candidate else None
        )
        anchor = (
            torch.zeros(self.spec.action_dim, dtype=torch.float32)
            if self.candidate
            else None
        )
        self.register_buffer("no_action_slot", anchor, persistent=True)
        self.reducer_call_count = 0

    @property
    def arm_name(self) -> str:
        return "CACH-A4" if self.candidate else "REF-NORMALIZED-A4"

    @property
    def common_trainable_names(self) -> frozenset[str]:
        return frozenset(
            f"common.{name}"
            for name, parameter in self.common.named_parameters()
            if parameter.requires_grad
        )

    @property
    def action_trainable_names(self) -> frozenset[str]:
        if self.action_stream is None:
            return frozenset()
        return frozenset(
            f"action_stream.{name}"
            for name, parameter in self.action_stream.named_parameters()
            if parameter.requires_grad
        )

    def _actions_for_mode(self, task: A4Task, mode: ActionMode) -> Tensor:
        if mode == "correct":
            return task.global_actions
        if mode == "shuffle":
            permutation = torch.tensor(
                A4_SHUFFLE_PERMUTATION,
                dtype=torch.long,
                device=task.global_actions.device,
            )
            return task.global_actions.index_select(0, permutation)
        raise RuntimeError("inactive mode must bypass action read")

    @staticmethod
    def _fixed_capacity_actions(global_actions: Tensor, task_chunk: object) -> Tensor:
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

    def _reduce_active_actions(
        self, task: A4Task, mode: ActionMode
    ) -> tuple[Tensor, Tensor]:
        if mode not in ("correct", "shuffle"):
            raise RuntimeError("only active modes may call the A4 reducer")
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
            local_mask = (
                torch.tensor(
                    tuple(
                        bool(task_chunk.layout.latent_valid_mask[index])
                        and not span.anchor_no_action_slot
                        for index, span in enumerate(
                            task_chunk.layout.latent_action_spans
                        )
                    ),
                    dtype=torch.bool,
                    device=global_actions.device,
                )
                .view(1, -1)
                .expand(self.spec.batch_size, -1)
            )
            mask_parts.append(local_mask[:, :valid_count])
        raw = torch.cat(reduced_parts, dim=1)
        mask = torch.cat(mask_parts, dim=1).unsqueeze(-1)
        expected = (
            torch.tensor(
                (False, True, True, True, True),
                dtype=torch.bool,
                device=mask.device,
            )
            .view(1, 5, 1)
            .expand_as(mask)
        )
        if tuple(raw.shape) != (8, 5, 20) or not torch.equal(mask, expected):
            raise RuntimeError("A4 reducer or typed mask drifted")
        self.reducer_call_count += 1
        return raw, mask

    def _common_forward(self, task: A4Task) -> tuple[Tensor, Tensor, int]:
        output = self.common(task, mode="no_action", capture_diagnostics=True)
        return (
            output.common_prediction,
            output.final_common_hidden,
            output.vendor_calls_this_forward,
        )

    def action_delta_from_state(
        self,
        task: A4Task,
        mode: ActionMode,
        common_hidden: Tensor,
        *,
        count_calls: bool = True,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        detached_common = common_hidden.detach()
        normalized_detached = a3._parameter_free_rms_norm(detached_common)
        if self.action_stream is None or mode in ("no_action", "seam_disabled"):
            zero = torch.zeros(
                (
                    common_hidden.shape[0],
                    common_hidden.shape[1],
                    self.spec.latent_channels,
                    1,
                    1,
                ),
                dtype=common_hidden.dtype,
                device=common_hidden.device,
            )
            # Structural bypass occurs before reducer and stream invocation.
            return zero, None, None, normalized_detached
        raw, mask = self._reduce_active_actions(task, mode)
        stream_output = self.action_stream(
            raw,
            detached_common,
            mask,
            count_calls=count_calls,
        )
        return (
            stream_output.delta.unsqueeze(-1).unsqueeze(-1),
            raw,
            mask,
            normalized_detached,
        )

    def forward(
        self,
        task: A4Task,
        mode: ActionMode = "correct",
        *,
        capture_diagnostics: bool = True,
    ) -> A4SequenceOutput:
        if mode not in ("correct", "shuffle", "no_action", "seam_disabled"):
            raise ValueError("unsupported A4 mode")
        reducer_before = self.reducer_call_count
        stream_before = (
            0 if self.action_stream is None else self.action_stream.call_count
        )
        common, hidden, vendor_calls = self._common_forward(task)
        delta, raw, mask, normalized_state = self.action_delta_from_state(
            task, mode, hidden
        )
        prediction = common + delta
        stream_after = (
            0 if self.action_stream is None else self.action_stream.call_count
        )
        return A4SequenceOutput(
            video_prediction=prediction,
            common_prediction=common,
            action_delta=delta,
            final_common_hidden=hidden if capture_diagnostics else hidden.detach(),
            normalized_detached_common_state=normalized_state,
            raw_action=raw,
            action_present_mask=mask,
            vendor_calls_this_forward=vendor_calls,
            reducer_calls_this_forward=self.reducer_call_count - reducer_before,
            action_stream_calls_this_forward=stream_after - stream_before,
        )


def build_a4_synthetic_task(
    spec: A4BridgeSpec | None = None,
    *,
    device: torch.device | str = torch.device("cuda:0"),
) -> A4Task:
    return a3.build_a3_synthetic_task(spec, device=device)


def build_cach_a4_pair(
    spec: A4BridgeSpec | None = None,
    *,
    device: torch.device | str = torch.device("cuda:0"),
) -> A4BridgePair:
    spec = A4BridgeSpec() if spec is None else spec
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index != 0:
        raise ValueError("A4 is frozen to logical cuda:0")
    torch.manual_seed(a3.a2r1.AV1B_SHARED_NAMED_SEED)
    torch.cuda.manual_seed_all(a3.a2r1.AV1B_SHARED_NAMED_SEED)
    with torch.device("cpu"):
        r1_common = a3.a2r1.AV1BVendorGDNBridgeArm(
            spec=spec,
            staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
        )
        a3_common = a3.A3BridgeArm(common=r1_common, candidate=False)
    reference = A4BridgeArm(common=a3_common, candidate=False)
    candidate = A4BridgeArm(common=copy.deepcopy(a3_common), candidate=True)
    reference = reference.to(device=resolved, dtype=torch.float32)
    candidate = candidate.to(device=resolved, dtype=torch.float32)
    reference.train(True)
    candidate.train(True)
    reference_common = reference.common_trainable_names
    candidate_common = candidate.common_trainable_names
    candidate_action = candidate.action_trainable_names
    if reference_common != candidate_common or reference_common & candidate_action:
        raise RuntimeError("A4 optimizer scopes or common names drifted")
    ref_state = reference.common.state_dict()
    cand_state = candidate.common.state_dict()
    if ref_state.keys() != cand_state.keys() or any(
        not torch.equal(ref_state[name], cand_state[name]) for name in ref_state
    ):
        raise RuntimeError("A4 common theta0 bytes differ")
    if candidate.action_stream is None:
        raise RuntimeError("A4 candidate stream is absent")
    if bool(candidate.action_stream.output_projection.weight.count_nonzero()):
        raise RuntimeError("A4 output projection must be zero at theta0")
    theta0 = {
        "reference_common": a3._state_manifest(reference.common),
        "candidate_common": a3._state_manifest(candidate.common),
        "candidate_action": a3._state_manifest(candidate.action_stream),
    }
    return A4BridgePair(
        spec=spec,
        reference=reference,
        candidate=candidate,
        reference_common_trainable_names=reference_common,
        candidate_common_trainable_names=candidate_common,
        candidate_action_trainable_names=candidate_action,
        theta0_manifests=theta0,
    )


build_a4_pair = build_cach_a4_pair


def _require_config(config: Mapping[str, object]) -> None:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if config.get("schema") != A4_CONFIG_SCHEMA:
        raise ValueError("A4 config schema differs")
    if config.get("architecture_id") != CACH_A4_ARCHITECTURE_ID:
        raise ValueError("A4 architecture id differs")
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("A4 training mapping is required")
    if int(training.get("macrosteps_per_arm", -1)) != A4_TRAIN_STEPS:
        raise ValueError("A4 is frozen to 200 macrosteps")


def _set_lr(optimizer: torch.optim.Optimizer, step: int) -> float:
    fraction = float(step) / float(A4_TRAIN_STEPS - 1)
    value = A4_LR_END + 0.5 * (A4_LR_START - A4_LR_END) * (
        1.0 + math.cos(math.pi * fraction)
    )
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def _stream_jvp(
    stream: StateConditionedCausalOddActionStream,
    raw: Tensor,
    state: Tensor,
    mask: Tensor,
    *,
    with_respect_to: Literal["action", "state"],
) -> float:
    if with_respect_to == "action":
        direction = torch.ones_like(raw) * mask.to(dtype=raw.dtype)

        def function(value: Tensor) -> Tensor:
            return stream(value, state, mask, count_calls=False).delta

        _, tangent = torch.func.jvp(function, (raw,), (direction,))
    else:
        direction = torch.linspace(
            -0.5,
            0.5,
            state.numel(),
            dtype=state.dtype,
            device=state.device,
        ).reshape_as(state)

        def function(value: Tensor) -> Tensor:
            return stream(raw, value, mask, count_calls=False).delta

        _, tangent = torch.func.jvp(function, (state,), (direction,))
    return float(tangent.detach().double().square().mean().sqrt().item())


def _past_action_to_future_output_jvp(
    stream: StateConditionedCausalOddActionStream,
    raw: Tensor,
    state: Tensor,
    mask: Tensor,
) -> float:
    if raw.shape[1] < 3:
        raise ValueError("A4 causal JVP requires at least three positions")
    direction = torch.zeros_like(raw)
    direction[:, 1] = torch.linspace(
        -0.75,
        0.75,
        raw.shape[0] * raw.shape[2],
        dtype=raw.dtype,
        device=raw.device,
    ).reshape(raw.shape[0], raw.shape[2])
    direction[:, 1] *= mask[:, 1].to(dtype=raw.dtype)

    def function(value: Tensor) -> Tensor:
        return stream(value, state, mask, count_calls=False).delta[:, -1]

    _, tangent = torch.func.jvp(function, (raw,), (direction,))
    return float(tangent.detach().double().square().mean().sqrt().item())


def _delta_state_to_future_output_jvp(
    stream: StateConditionedCausalOddActionStream,
    output: A4ActionStreamOutput,
) -> float:
    if output.recurrent_state.shape[1] < 3:
        raise ValueError("A4 recurrent-state JVP requires at least three positions")
    initial = torch.zeros_like(output.recurrent_state[:, 1])
    direction = torch.linspace(
        -0.5,
        0.5,
        initial.numel(),
        dtype=initial.dtype,
        device=initial.device,
    ).reshape_as(initial)
    decay = output.decay.detach()
    write = output.write.detach()

    def function(value: Tensor) -> Tensor:
        carry = value
        for time_index in range(2, output.recurrent_state.shape[1]):
            carry = decay[:, time_index] * carry + write[:, time_index]
        return stream.output_projection(carry)

    _, tangent = torch.func.jvp(function, (initial,), (direction,))
    return float(tangent.detach().double().square().mean().sqrt().item())


def _state_stream_diagnostics(
    stream: StateConditionedCausalOddActionStream,
    raw: Tensor,
    state: Tensor,
    mask: Tensor,
) -> Mapping[str, object]:
    with torch.no_grad():
        positive = stream(raw, state, mask, count_calls=False)
        negative = stream(-raw, state, mask, count_calls=False)
        zero = stream(torch.zeros_like(raw), state, mask, count_calls=False)
        future_raw = raw.clone()
        future_state = state.clone()
        future_raw[:, 3:] = future_raw[:, 3:] * -3.0 + 0.75
        future_state[:, 3:] = future_state[:, 3:] + 1.25
        future_changed = stream(future_raw, future_state, mask, count_calls=False)
        past_raw = raw.clone()
        past_raw[:, 1] = past_raw[:, 1] + 0.5
        past_changed = stream(past_raw, state, mask, count_calls=False)
        prefix_difference = (future_changed.delta[:, :3] - positive.delta[:, :3]).abs()
        prefix_denominator = max(
            float(positive.delta[:, :3].abs().max().item()),
            float(torch.finfo(positive.delta.dtype).eps),
        )
    common_state_to_output_jvp_rms = _stream_jvp(
        stream, raw, state, mask, with_respect_to="state"
    )
    return {
        "oddness_max_abs": float((positive.delta + negative.delta).abs().max().item()),
        "zero_origin_max_abs": float(zero.delta.abs().max().item()),
        "decay_action_blind_bitwise": torch.equal(positive.decay, negative.decay),
        "future_to_prefix_max_abs": float(prefix_difference.max().item()),
        "future_to_prefix_max_rel": float(
            prefix_difference.max().item() / prefix_denominator
        ),
        "past_to_future_max_abs": float(
            (past_changed.delta[:, 4] - positive.delta[:, 4]).abs().max().item()
        ),
        "action_jvp_rms": _stream_jvp(
            stream, raw, state, mask, with_respect_to="action"
        ),
        "past_action_to_future_output_jvp_rms": (
            _past_action_to_future_output_jvp(stream, raw, state, mask)
        ),
        "delta_state_to_future_output_jvp_rms": (
            _delta_state_to_future_output_jvp(stream, positive)
        ),
        "common_state_to_output_jvp_rms": common_state_to_output_jvp_rms,
        "common_state_jvp_rms": common_state_to_output_jvp_rms,
        "decay_mean": float(positive.decay.mean().item()),
        "decay_min": float(positive.decay.min().item()),
        "decay_max": float(positive.decay.max().item()),
        "decay_finite": bool(torch.isfinite(positive.decay).all().item()),
        "decay_strictly_between_zero_and_one": bool(
            ((positive.decay > 0.0) & (positive.decay < 1.0)).all().item()
        ),
        "state_rms": float(
            positive.recurrent_state.detach().double().square().mean().sqrt().item()
        ),
    }


def classify_cach_a4(
    config: Mapping[str, object],
    *,
    all_validity: bool,
    action_metrics: Mapping[str, object],
    common_stable: bool,
) -> tuple[str, str | None, Mapping[str, object]]:
    thresholds = config.get("thresholds")
    verdicts = config.get("verdict")
    if not isinstance(thresholds, Mapping) or not isinstance(verdicts, Mapping):
        raise ValueError("A4 thresholds and verdict are required")
    go = thresholds.get("operator_go")
    stop = thresholds.get("causal_stream_weak_stop")
    aggregate = action_metrics.get("aggregate")
    by_horizon = action_metrics.get("by_horizon")
    if not all(
        isinstance(value, Mapping) for value in (go, stop, aggregate, by_horizon)
    ):
        raise ValueError("A4 action thresholds or metrics are malformed")
    assert isinstance(go, Mapping)
    assert isinstance(stop, Mapping)
    assert isinstance(aggregate, Mapping)
    assert isinstance(by_horizon, Mapping)

    aggregate_checks = {
        "delta_nmse": float(aggregate["delta_nmse"]) <= float(go["delta_nmse_max"]),
        "energy": float(go["delta_energy_ratio_min"])
        <= float(aggregate["delta_energy_ratio"])
        <= float(go["delta_energy_ratio_max"]),
        "cosine": float(aggregate["delta_alignment_cosine"])
        >= float(go["delta_alignment_cosine_min"]),
        "explained": float(aggregate["action_explained_fraction"])
        >= float(go["action_explained_fraction_min"]),
        "no_action": float(aggregate["no_action_recovery"])
        >= float(go["no_action_recovery_min"]),
        "shuffle": float(aggregate["shuffle_penalty"])
        >= float(go["shuffle_penalty_min"]),
    }
    horizon_checks: dict[str, Mapping[str, bool]] = {}
    for horizon in ("1", "2", "4"):
        one = by_horizon[horizon]
        assert isinstance(one, Mapping)
        horizon_checks[horizon] = {
            "delta_nmse": float(one["delta_nmse"])
            <= float(go["delta_nmse_primary_horizon_max_each"]),
            "energy": float(go["delta_energy_ratio_primary_horizon_min_each"])
            <= float(one["delta_energy_ratio"])
            <= float(go["delta_energy_ratio_primary_horizon_max_each"]),
            "cosine": float(one["delta_alignment_cosine"])
            >= float(go["delta_alignment_cosine_primary_horizon_min_each"]),
            "explained": float(one["action_explained_fraction"])
            >= float(go["action_explained_fraction_primary_horizon_min_each"]),
            "no_action": float(one["no_action_recovery"])
            >= float(go["no_action_recovery_primary_horizon_min_each"]),
            "shuffle": float(one["shuffle_penalty"])
            >= float(go["shuffle_penalty_primary_horizon_min_each"]),
        }
    action_go = all(aggregate_checks.values()) and all(
        all(value.values()) for value in horizon_checks.values()
    )
    stop_checks = {
        "delta_nmse": float(aggregate["delta_nmse"]) >= float(stop["delta_nmse_min"]),
        "cosine": float(aggregate["delta_alignment_cosine"])
        <= float(stop["delta_alignment_cosine_max"]),
        "explained": float(aggregate["action_explained_fraction"])
        <= float(stop["action_explained_fraction_max"]),
        "shuffle": float(aggregate["shuffle_penalty"])
        <= float(stop["shuffle_penalty_max"]),
    }
    if "all_required" not in stop or not isinstance(stop["all_required"], bool):
        raise ValueError("A4 causal_stream_weak_stop.all_required is required")
    action_stop = bool(stop["all_required"]) and all(stop_checks.values())
    inputs: Mapping[str, object] = {
        "all_validity": all_validity,
        "aggregate_go_checks": aggregate_checks,
        "primary_horizon_go_checks": horizon_checks,
        "action_go": action_go,
        "causal_stream_weak_stop_checks": stop_checks,
        "causal_stream_weak_stop": action_stop,
        "common_stable": common_stable,
    }

    def verdict(name: str) -> str:
        if name not in verdicts:
            raise ValueError(f"A4 verdict key is required: {name}")
        return str(verdicts[name])

    if not all_validity:
        return verdict("invalid"), "A4_VALIDITY_FAILURE", inputs
    if action_go and common_stable:
        return (
            verdict("valid_go_common_stable"),
            None,
            inputs,
        )
    if action_go:
        return (
            verdict("valid_go_common_blocked"),
            "A4_ACTION_GO_COMMON_BLOCKED",
            inputs,
        )
    if action_stop:
        return (
            verdict("valid_all_weak_stop"),
            "A4_CAUSAL_STREAM_WEAK",
            inputs,
        )
    return (
        verdict("valid_other"),
        "A4_ACTION_NEITHER_GO_NOR_WEAK_STOP",
        inputs,
    )


def run_cach_a4_screen(
    config: Mapping[str, object],
    *,
    expected_gpu_uuid: str,
    source_six_and_predecessor_pins_match: bool,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> Mapping[str, object]:
    """Run the 200-step separated positive common/delta A4 screen."""

    started = time.monotonic()
    _require_config(config)
    if not isinstance(source_six_and_predecessor_pins_match, bool):
        raise TypeError("A4 source/predecessor pin proof must be bool")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from A4 binding")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A4 requires exactly one visible CUDA device")
    if os.environ.get("FUSED_GDN_PRECISION") != "0":
        raise RuntimeError("A4 requires FUSED_GDN_PRECISION=0")
    device = torch.device("cuda:0")
    spec = A4BridgeSpec()
    task = build_a4_synthetic_task(spec, device=device)
    pair = build_cach_a4_pair(spec, device=device)
    target = task.video_target.detach()
    common_target = a3._paired_common(target)
    delta_target = a3._paired_half_delta(target)

    ref_names, ref_parameters = a3._parameters(
        pair.reference, pair.reference_common_trainable_names
    )
    cand_names, cand_parameters = a3._parameters(
        pair.candidate, pair.candidate_common_trainable_names
    )
    action_names, action_parameters = a3._parameters(
        pair.candidate, pair.candidate_action_trainable_names
    )
    ref_initial = a3._snapshot(ref_names, ref_parameters)
    cand_initial = a3._snapshot(cand_names, cand_parameters)
    action_initial = a3._snapshot(action_names, action_parameters)

    reference_optimizer = torch.optim.AdamW(
        ref_parameters, lr=A4_LR_START, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=0.0
    )
    candidate_optimizer = torch.optim.AdamW(
        cand_parameters, lr=A4_LR_START, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=0.0
    )
    action_optimizer = torch.optim.AdamW(
        action_parameters,
        lr=A4_LR_START,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    traces: dict[str, list[float]] = {
        "reference_common": [],
        "candidate_common": [],
        "candidate_delta": [],
    }
    step0_gradient_rms: dict[str, float] = {}
    gradient_clip_finite = True
    gradient_clip_norm_max = {
        "reference_common": 0.0,
        "candidate_common": 0.0,
        "candidate_action": 0.0,
    }
    lr_trace: list[float] = []

    with torch.no_grad():
        theta0_reference = pair.reference(task, mode="correct")
        theta0_candidate = pair.candidate(task, mode="correct")
    theta0_common_equal = torch.equal(
        theta0_reference.common_prediction, theta0_candidate.common_prediction
    )
    theta0_candidate_identity = torch.equal(
        theta0_candidate.video_prediction, theta0_candidate.common_prediction
    )

    for step in range(A4_TRAIN_STEPS):
        lr_trace.append(_set_lr(reference_optimizer, step))
        _set_lr(candidate_optimizer, step)
        _set_lr(action_optimizer, step)

        reference_optimizer.zero_grad(set_to_none=True)
        reference_common, _, _ = pair.reference._common_forward(task)
        reference_loss = a3._positive_mse(
            a3._paired_common(reference_common), common_target
        )
        traces["reference_common"].append(float(reference_loss.detach().item()))
        reference_loss.backward()
        if step == 0:
            step0_gradient_rms["reference_common"] = a3._gradient_rms(ref_parameters)
        ref_clip = float(
            torch.nn.utils.clip_grad_norm_(ref_parameters, 1.0).detach().item()
        )
        gradient_clip_finite = gradient_clip_finite and math.isfinite(ref_clip)
        gradient_clip_norm_max["reference_common"] = max(
            gradient_clip_norm_max["reference_common"], ref_clip
        )
        reference_optimizer.step()

        candidate_optimizer.zero_grad(set_to_none=True)
        candidate_common, _, _ = pair.candidate._common_forward(task)
        candidate_loss = a3._positive_mse(
            a3._paired_common(candidate_common), common_target
        )
        traces["candidate_common"].append(float(candidate_loss.detach().item()))
        candidate_loss.backward()
        if step == 0:
            step0_gradient_rms["candidate_common"] = a3._gradient_rms(cand_parameters)
        cand_clip = float(
            torch.nn.utils.clip_grad_norm_(cand_parameters, 1.0).detach().item()
        )
        gradient_clip_finite = gradient_clip_finite and math.isfinite(cand_clip)
        gradient_clip_norm_max["candidate_common"] = max(
            gradient_clip_norm_max["candidate_common"], cand_clip
        )
        candidate_optimizer.step()

        action_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            _, action_common_hidden, _ = pair.candidate._common_forward(task)
        action_delta, _, _, _ = pair.candidate.action_delta_from_state(
            task, "correct", action_common_hidden
        )
        action_loss = a3._positive_mse(
            a3._paired_half_delta(action_delta), delta_target
        )
        traces["candidate_delta"].append(float(action_loss.detach().item()))
        action_loss.backward()
        if step == 0:
            step0_gradient_rms["candidate_action"] = a3._gradient_rms(action_parameters)
        action_clip = float(
            torch.nn.utils.clip_grad_norm_(action_parameters, 1.0).detach().item()
        )
        gradient_clip_finite = gradient_clip_finite and math.isfinite(action_clip)
        gradient_clip_norm_max["candidate_action"] = max(
            gradient_clip_norm_max["candidate_action"], action_clip
        )
        action_optimizer.step()

        if progress_callback is not None and step in (0, 49, 99, 149, 199):
            progress_callback(
                {
                    "event": "optimizer_progress",
                    "optimizer_steps_completed": step + 1,
                    "lr": lr_trace[-1],
                    "reference_common_loss": traces["reference_common"][-1],
                    "candidate_common_loss": traces["candidate_common"][-1],
                    "candidate_delta_loss": traces["candidate_delta"][-1],
                }
            )

    final_reference_common, _, _ = pair.reference._common_forward(task)
    final_candidate_common, final_state, _ = pair.candidate._common_forward(task)
    final_delta, _, _, _ = pair.candidate.action_delta_from_state(
        task, "correct", final_state, count_calls=False
    )
    traces["reference_common"].append(
        float(
            a3._positive_mse(a3._paired_common(final_reference_common), common_target)
            .detach()
            .item()
        )
    )
    traces["candidate_common"].append(
        float(
            a3._positive_mse(a3._paired_common(final_candidate_common), common_target)
            .detach()
            .item()
        )
    )
    traces["candidate_delta"].append(
        float(
            a3._positive_mse(a3._paired_half_delta(final_delta), delta_target)
            .detach()
            .item()
        )
    )

    with torch.no_grad():
        reference = pair.reference(task, mode="correct")
        correct = pair.candidate(task, mode="correct")
        shuffle = pair.candidate(task, mode="shuffle")
        no_action = pair.candidate(task, mode="no_action")
        disabled = pair.candidate(task, mode="seam_disabled")
    if (
        pair.candidate.action_stream is None
        or correct.raw_action is None
        or correct.action_present_mask is None
    ):
        raise RuntimeError("A4 final action diagnostics are absent")
    stream_diagnostics = _state_stream_diagnostics(
        pair.candidate.action_stream,
        correct.raw_action,
        correct.final_common_hidden.detach(),
        correct.action_present_mask,
    )
    action_metrics = a3._registered_action_metrics(
        correct.video_prediction,
        no_action.video_prediction,
        shuffle.video_prediction,
        target,
    )
    mse = {
        "reference": a3._mse_by_horizon(reference.video_prediction, target),
        "correct": a3._mse_by_horizon(correct.video_prediction, target),
        "shuffle": a3._mse_by_horizon(shuffle.video_prediction, target),
        "no_action": a3._mse_by_horizon(no_action.video_prediction, target),
    }
    primary = {name: a3._primary(value) for name, value in mse.items()}
    stability = {name: a3._loss_statistics(trace) for name, trace in traces.items()}
    final_common_mse = max(
        traces["reference_common"][-1], traces["candidate_common"][-1]
    )
    final_over_last50 = max(
        float(stability["reference_common"]["final_over_last50_min_ratio"]),
        float(stability["candidate_common"]["final_over_last50_min_ratio"]),
    )
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("A4 threshold mapping is required")
    common_t = thresholds.get("common_stability")
    validity_t = thresholds.get("validity")
    if not isinstance(common_t, Mapping) or not isinstance(validity_t, Mapping):
        raise ValueError("A4 common/validity thresholds are required")
    common_stable = final_common_mse <= float(
        common_t["final_common_mse_max"]
    ) and final_over_last50 <= float(common_t["final_over_last50_min_max"])

    ref_named = dict(pair.reference.named_parameters())
    cand_named = dict(pair.candidate.named_parameters())
    common_parameter_parity = all(
        torch.equal(ref_named[name].detach(), cand_named[name].detach())
        for name in ref_names
    )
    common_prediction_difference = float(
        (final_reference_common.detach() - final_candidate_common.detach())
        .abs()
        .max()
        .item()
    )
    no_action_bypass = (
        torch.equal(no_action.video_prediction, no_action.common_prediction)
        and torch.equal(disabled.video_prediction, disabled.common_prediction)
        and no_action.reducer_calls_this_forward == 0
        and disabled.reducer_calls_this_forward == 0
        and no_action.action_stream_calls_this_forward == 0
        and disabled.action_stream_calls_this_forward == 0
    )
    action_update = a3._update_rms(action_names, action_parameters, action_initial)
    inactive_mask = ~correct.action_present_mask.unsqueeze(-1).unsqueeze(-1).expand_as(
        correct.action_delta
    )
    inactive_delta_max_abs = float(
        correct.action_delta.masked_select(inactive_mask).abs().max().item()
    )
    inactive_and_no_action_delta_max_abs = max(
        inactive_delta_max_abs,
        float(no_action.action_delta.abs().max().item()),
        float(disabled.action_delta.abs().max().item()),
    )
    finite = all(
        math.isfinite(value)
        for value in (
            *a3._numeric_leaves(action_metrics),
            *a3._numeric_leaves(stream_diagnostics),
            *[item for values in traces.values() for item in values],
            *gradient_clip_norm_max.values(),
            common_prediction_difference,
            final_common_mse,
            final_over_last50,
        )
    )

    def validity_float(name: str) -> float:
        if name not in validity_t:
            raise ValueError(f"A4 validity threshold is required: {name}")
        return float(validity_t[name])

    def validity_bool(name: str) -> bool:
        if name not in validity_t or not isinstance(validity_t[name], bool):
            raise ValueError(f"A4 bool validity threshold is required: {name}")
        return bool(validity_t[name])

    def validity_int(name: str) -> int:
        if name not in validity_t or isinstance(validity_t[name], bool):
            raise ValueError(f"A4 integer validity threshold is required: {name}")
        return int(validity_t[name])

    theta0_common_bytes_equal = (
        pair.theta0_manifests["reference_common"]
        == pair.theta0_manifests["candidate_common"]
    )

    checks = {
        "source_six_and_predecessor_pins_match": (
            source_six_and_predecessor_pins_match
            is validity_bool("source_six_and_predecessor_pins_match")
        ),
        "theta0_common_bytes_equal": (
            theta0_common_bytes_equal is validity_bool("theta0_common_bytes_equal")
        ),
        "common_trajectory_max_abs": common_prediction_difference
        <= validity_float("common_trajectory_max_abs"),
        "oddness_max_abs": float(stream_diagnostics["oddness_max_abs"])
        <= validity_float("oddness_max_abs"),
        "zero_origin_max_abs": float(stream_diagnostics["zero_origin_max_abs"])
        <= validity_float("zero_origin_max_abs"),
        "inactive_and_no_action_delta_max_abs": (
            inactive_and_no_action_delta_max_abs
            <= validity_float("inactive_and_no_action_delta_max_abs")
        ),
        "finite_all": finite is validity_bool("finite_all"),
        "exact_macrosteps_each_arm": (
            len(lr_trace) == validity_int("exact_macrosteps_each_arm")
            and abs(lr_trace[0] - A4_LR_START) <= 1.0e-15
            and abs(lr_trace[-1] - A4_LR_END) <= 1.0e-15
        ),
        "seam_gradient_rms_step0_gt": (
            step0_gradient_rms["candidate_action"]
            > validity_float("seam_gradient_rms_step0_gt")
        ),
        "seam_update_rms_final_gt": action_update
        > validity_float("seam_update_rms_final_gt"),
        "action_jvp_rms_final_gt": float(stream_diagnostics["action_jvp_rms"])
        > validity_float("action_jvp_rms_final_gt"),
        "prefix_future_leakage_max_abs": float(
            stream_diagnostics["future_to_prefix_max_abs"]
        )
        <= validity_float("prefix_future_leakage_max_abs"),
        "prefix_future_leakage_max_rel": float(
            stream_diagnostics["future_to_prefix_max_rel"]
        )
        <= validity_float("prefix_future_leakage_max_rel"),
        "past_action_to_future_output_jvp_rms_gt": float(
            stream_diagnostics["past_action_to_future_output_jvp_rms"]
        )
        > validity_float("past_action_to_future_output_jvp_rms_gt"),
        "delta_state_to_future_output_jvp_rms_gt": float(
            stream_diagnostics["delta_state_to_future_output_jvp_rms"]
        )
        > validity_float("delta_state_to_future_output_jvp_rms_gt"),
        "theta0_common_output_equal": theta0_common_equal,
        "theta0_candidate_delta_exact_zero": theta0_candidate_identity,
        "reference_structural_action_bypass": (
            reference.raw_action is None
            and reference.reducer_calls_this_forward == 0
            and reference.action_stream_calls_this_forward == 0
        ),
        "inactive_structural_direct_bypass": no_action_bypass,
        "common_state_to_output_jvp_rms_gt": float(
            stream_diagnostics["common_state_to_output_jvp_rms"]
        )
        > validity_float("common_state_to_output_jvp_rms_gt"),
        "state_decay_action_blind": bool(
            stream_diagnostics["decay_action_blind_bitwise"]
        ),
        "state_decay_finite_and_in_range": bool(stream_diagnostics["decay_finite"])
        and bool(stream_diagnostics["decay_strictly_between_zero_and_one"]),
        "common_state_detached_at_integration": (
            not correct.normalized_detached_common_state.requires_grad
        ),
        "common_parameter_parity": common_parameter_parity,
        "gradient_clip_norms_finite": gradient_clip_finite,
        "positive_separated_losses_no_subtraction": True,
        "target_forward_leakage_absent": True,
    }
    reasons = sorted(name for name, passed in checks.items() if not passed)
    verdict, subclassification, classification_inputs = classify_cach_a4(
        config,
        all_validity=not reasons,
        action_metrics=action_metrics,
        common_stable=common_stable,
    )
    aggregate = action_metrics["aggregate"]
    assert isinstance(aggregate, Mapping)
    metrics: Mapping[str, object] = {
        "action": action_metrics,
        "counterfactual_delta_nmse": float(aggregate["delta_nmse"]),
        "target_half_delta_energy": float(aggregate["target_half_delta_energy"]),
        "delta_energy_ratio": float(aggregate["delta_energy_ratio"]),
        "delta_alignment_cosine": float(aggregate["delta_alignment_cosine"]),
        "action_explained_fraction": float(aggregate["action_explained_fraction"]),
        "no_action_recovery": float(aggregate["no_action_recovery"]),
        "shuffle_penalty": float(aggregate["shuffle_penalty"]),
        "mse_by_horizon": mse,
        "primary": primary,
        "legacy_total_relative_gaps_report_only": {
            "candidate_gain_vs_reference": (
                (primary["reference"] - primary["correct"])
                / max(primary["reference"], 1.0e-12)
            ),
            "shuffle_gap_rel": a3._gap(primary["shuffle"], primary["correct"]),
            "no_action_gap_rel": a3._gap(primary["no_action"], primary["correct"]),
            "used_for_classification": False,
        },
        "state_stream": stream_diagnostics,
        "common_stability": stability,
        "final_common_mse": final_common_mse,
        "final_common_over_last50_min_ratio": final_over_last50,
        "common_stable": common_stable,
        "gradient_rms_step0": step0_gradient_rms,
        "gradient_clip_norm_max": gradient_clip_norm_max,
        "inactive_and_no_action_delta_max_abs": (inactive_and_no_action_delta_max_abs),
        "update_rms_final": {
            "reference_common": a3._update_rms(ref_names, ref_parameters, ref_initial),
            "candidate_common": a3._update_rms(
                cand_names, cand_parameters, cand_initial
            ),
            "candidate_action": action_update,
        },
    }
    diagnostics: Mapping[str, object] = {
        "architecture_id": CACH_A4_ARCHITECTURE_ID,
        "common_source": "FROZEN_A3_NORMALIZED_ACTION_BLIND_COMMON",
        "action_stream_reads_detached_common_state": True,
        "state_normalization": "PARAMETER_FREE_RMS_NORM",
        "odd_formula": "0.5*(g(s,a)-g(s,-a))",
        "decay": "SIGMOID_BIAS_FREE_STATE_ONLY_ACTION_BLIND",
        "recurrence": "STRICT_TEMPORAL_CAUSAL_LINEAR",
        "output_addition": "AFTER_COMMON_VIDEO_OUTPUT_NO_LATER_OPERATOR",
        "optimizer_scopes": {
            "reference_common": ref_names,
            "candidate_common": cand_names,
            "candidate_action": action_names,
            "disjoint": True,
        },
        "common_parameter_parity": common_parameter_parity,
        "common_prediction_difference_max_abs": common_prediction_difference,
        "optimizer_steps_completed_by_arm": {
            "reference": len(lr_trace),
            "candidate": len(lr_trace),
        },
        "optimizer_updates_completed_by_scope": {
            "reference_common": len(lr_trace),
            "candidate_common": len(lr_trace),
            "candidate_action": len(lr_trace),
        },
        "gradient_clip_norms_finite": gradient_clip_finite,
        "gradient_clip_norm_max": gradient_clip_norm_max,
        "checkpoint_loaded_or_saved": False,
        "real_data_used": False,
        "loss_subtraction_used": False,
    }
    final_parameter_digests = {
        "reference_common": a3._state_manifest(pair.reference.common),
        "candidate_common": a3._state_manifest(pair.candidate.common),
        "candidate_action": a3._state_manifest(pair.candidate.action_stream),
    }
    jit_inventory = a3.a2r1._screen_jit_inventory(
        pair.reference.common.common,
        pair.candidate.common.common,
    )
    elapsed = time.monotonic() - started
    return {
        "schema": A4_SCREEN_SCHEMA,
        "architecture_id": CACH_A4_ARCHITECTURE_ID,
        "metrics": metrics,
        "validity": {
            "all_card_validity_checks_pass": not reasons,
            "checks": checks,
            "reasons": reasons,
        },
        "diagnostics": diagnostics,
        "evidence": diagnostics,
        "classification_inputs": classification_inputs,
        "typed_verdict": verdict,
        "diagnostic_subclassification": subclassification,
        "theta0_manifests": pair.theta0_manifests,
        "loss_trace": traces,
        "final_parameter_digests": final_parameter_digests,
        "jit_inventory": jit_inventory,
        "resource_usage": {
            "elapsed_seconds_screen": elapsed,
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            * (1 if platform.system() == "Darwin" else 1024),
            "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "checkpoint_loaded": False,
            "checkpoint_saved": False,
            "real_data_used": False,
        },
        "elapsed_seconds_total": elapsed,
    }


__all__ = [
    "A4ActionStreamOutput",
    "A4BridgeArm",
    "A4BridgePair",
    "A4BridgeSpec",
    "A4SequenceOutput",
    "A4Task",
    "A4_CONFIG_SCHEMA",
    "A4_SCREEN_SCHEMA",
    "CACH_A4_ARCHITECTURE_ID",
    "StateConditionedCausalOddActionStream",
    "build_a4_pair",
    "build_a4_synthetic_task",
    "build_cach_a4_pair",
    "classify_cach_a4",
    "run_cach_a4_screen",
]
