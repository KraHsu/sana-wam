from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from sana_wam.model.cach_av1b_a4_state_conditioned_causal_odd_stream import (
    A4BridgeArm,
    A4BridgeSpec,
    CACH_A4_ARCHITECTURE_ID,
    StateConditionedCausalOddActionStream,
    _require_config,
    _state_stream_diagnostics,
    classify_cach_a4,
)


def _mask(batch: int = 2) -> torch.Tensor:
    return (
        torch.tensor((False, True, True, True, True), dtype=torch.bool)
        .view(1, 5, 1)
        .expand(batch, 5, 1)
        .clone()
    )


def _live_stream() -> StateConditionedCausalOddActionStream:
    stream = StateConditionedCausalOddActionStream(A4BridgeSpec())
    with torch.no_grad():
        stream.action_projection.weight.zero_()
        stream.state_projection.weight.zero_()
        stream.decay_projection.weight.zero_()
        stream.write_projection.weight.zero_()
        stream.output_projection.weight.zero_()
        for index in range(20):
            stream.action_projection.weight[index, index] = 0.75
        for index in range(64):
            stream.state_projection.weight[index, index] = 0.20
            stream.decay_projection.weight[index, index] = 0.10
            stream.write_projection.weight[index, index] = 1.00
        for channel in range(3):
            stream.output_projection.weight[channel, channel] = 0.50
            stream.output_projection.weight[channel, channel + 8] = -0.25
    return stream


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.linspace(-1.0, 1.0, 2 * 5 * 20).reshape(2, 5, 20)
    state = torch.linspace(0.2, 2.2, 2 * 5 * 64).reshape(2, 5, 64)
    return raw, state, _mask()


def test_a4_identity_and_bias_contract() -> None:
    assert CACH_A4_ARCHITECTURE_ID == (
        "CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1"
    )
    stream = StateConditionedCausalOddActionStream(A4BridgeSpec())
    assert stream.action_projection.bias is None
    assert stream.state_projection.bias is None
    assert stream.decay_projection.bias is None
    assert stream.write_projection.bias is None
    assert stream.output_projection.bias is None
    assert torch.count_nonzero(stream.action_projection.weight).item() > 0
    assert torch.count_nonzero(stream.state_projection.weight).item() > 0
    assert torch.count_nonzero(stream.write_projection.weight).item() > 0
    assert torch.count_nonzero(stream.decay_projection.weight).item() == 0
    assert torch.count_nonzero(stream.output_projection.weight).item() == 0


def test_zero_origin_exact_oddness_and_typed_inactive_zero() -> None:
    stream = _live_stream()
    raw, state, mask = _inputs()
    positive = stream(raw, state, mask, count_calls=False)
    negative = stream(-raw, state, mask, count_calls=False)
    zero = stream(torch.zeros_like(raw), state, mask, count_calls=False)
    inactive = stream(raw, state, torch.zeros_like(mask), count_calls=False)

    assert float((positive.delta + negative.delta).abs().max()) <= 1.0e-6
    assert torch.count_nonzero(zero.delta).item() == 0
    assert torch.count_nonzero(inactive.delta).item() == 0
    assert torch.count_nonzero(positive.delta[:, 0]).item() == 0
    assert torch.equal(positive.decay, negative.decay)


def test_typed_inactive_rows_never_enter_g_or_write() -> None:
    stream = _live_stream()
    raw, state, _ = _inputs()
    mask = torch.tensor(
        (
            (False, True, False, True, False),
            (False, False, True, True, False),
        ),
        dtype=torch.bool,
    ).unsqueeze(-1)
    raw = raw.clone()
    raw[~mask.expand_as(raw)] = float("nan")

    rows: dict[str, list[int]] = {
        "action": [],
        "state": [],
        "write": [],
    }

    def capture(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            assert torch.isfinite(inputs[0]).all()
            rows[name].append(int(inputs[0].shape[0]))

        return hook

    handles = (
        stream.action_projection.register_forward_pre_hook(capture("action")),
        stream.state_projection.register_forward_pre_hook(capture("state")),
        stream.write_projection.register_forward_pre_hook(capture("write")),
    )
    try:
        output = stream(raw, state, mask, count_calls=False)
    finally:
        for handle in handles:
            handle.remove()

    active_rows = int(mask.sum().item())
    assert sum(rows["action"]) == 2 * active_rows
    assert sum(rows["state"]) == active_rows
    assert sum(rows["write"]) == active_rows
    assert len(rows["write"]) == 3
    assert torch.isfinite(output.delta).all()
    assert torch.count_nonzero(output.delta[~mask.expand_as(output.delta)]).item() == 0


class _FakeCommon(nn.Module):
    def __init__(self, spec: A4BridgeSpec, state: torch.Tensor) -> None:
        super().__init__()
        self.spec = spec
        self.register_buffer("state", state)

    def forward(
        self,
        task: object,
        mode: str,
        *,
        capture_diagnostics: bool,
    ) -> SimpleNamespace:
        assert mode == "no_action"
        batch, frames, _ = self.state.shape
        common = torch.zeros(batch, frames, self.spec.latent_channels, 1, 1)
        return SimpleNamespace(
            common_prediction=common,
            final_common_hidden=self.state,
            vendor_calls_this_forward=0,
        )


class _NoActionTrapTask:
    @property
    def global_actions(self) -> torch.Tensor:
        raise AssertionError("inactive A4 mode read global_actions")


def test_full_no_action_and_seam_disabled_are_structural_bypasses() -> None:
    spec = A4BridgeSpec()
    state = torch.linspace(0.1, 1.1, 2 * 5 * 64).reshape(2, 5, 64)
    arm = A4BridgeArm(common=_FakeCommon(spec, state), candidate=True, spec=spec)
    assert arm.action_stream is not None
    task = _NoActionTrapTask()

    for mode in ("no_action", "seam_disabled"):
        reducer_before = arm.reducer_call_count
        stream_before = arm.action_stream.call_count
        output = arm(task, mode=mode)
        assert output.raw_action is None
        assert output.action_present_mask is None
        assert output.reducer_calls_this_forward == 0
        assert output.action_stream_calls_this_forward == 0
        assert arm.reducer_call_count == reducer_before
        assert arm.action_stream.call_count == stream_before
        assert torch.equal(output.video_prediction, output.common_prediction)
        assert torch.count_nonzero(output.action_delta).item() == 0


def test_future_changes_cannot_change_prefix() -> None:
    stream = _live_stream()
    raw, state, mask = _inputs()
    baseline = stream(raw, state, mask, count_calls=False).delta
    changed_raw = raw.clone()
    changed_state = state.clone()
    changed_raw[:, 3:] = changed_raw[:, 3:] * -3.0 + 1.25
    changed_state[:, 3:] = changed_state[:, 3:] + 9.0
    changed = stream(changed_raw, changed_state, mask, count_calls=False).delta
    assert torch.equal(baseline[:, :3], changed[:, :3])


def test_past_action_has_live_future_influence() -> None:
    stream = _live_stream()
    raw, state, mask = _inputs()
    changed_raw = raw.clone()
    changed_raw[:, 1] = changed_raw[:, 1] + 0.75
    baseline = stream(raw, state, mask, count_calls=False).delta
    changed = stream(changed_raw, state, mask, count_calls=False).delta
    assert float((changed[:, 4] - baseline[:, 4]).abs().max()) > 1.0e-6


def test_common_state_jvp_and_state_conditioned_decay_are_live() -> None:
    stream = _live_stream()
    raw, state, mask = _inputs()
    direction = torch.linspace(-0.5, 0.5, state.numel()).reshape_as(state)

    def function(value: torch.Tensor) -> torch.Tensor:
        return stream(raw, value, mask, count_calls=False).delta

    _, tangent = torch.func.jvp(function, (state,), (direction,))
    assert bool(torch.isfinite(tangent).all())
    assert float(tangent.abs().max()) > 1.0e-8

    first = stream(raw, state, mask, count_calls=False).decay
    second = stream(raw, state + 0.125, mask, count_calls=False).decay
    assert not torch.equal(first, second)


def test_registered_causal_jvps_and_state_diagnostics_are_live() -> None:
    stream = _live_stream()
    raw, state, mask = _inputs()
    diagnostics = _state_stream_diagnostics(stream, raw, state, mask)

    assert diagnostics["future_to_prefix_max_abs"] == 0.0
    assert diagnostics["future_to_prefix_max_rel"] == 0.0
    assert diagnostics["past_action_to_future_output_jvp_rms"] > 1.0e-8
    assert diagnostics["delta_state_to_future_output_jvp_rms"] > 1.0e-8
    assert diagnostics["common_state_to_output_jvp_rms"] > 1.0e-8
    assert diagnostics["common_state_jvp_rms"] > 1.0e-8
    assert (
        diagnostics["common_state_to_output_jvp_rms"]
        == diagnostics["common_state_jvp_rms"]
    )
    assert diagnostics["decay_finite"] is True
    assert diagnostics["decay_strictly_between_zero_and_one"] is True
    assert 0.0 < diagnostics["decay_mean"] < 1.0
    assert diagnostics["state_rms"] > 0.0


def test_frozen_config_and_causal_stream_weak_stop_keys_match() -> None:
    config = json.loads(
        Path(
            "configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream.yaml"
        ).read_text(encoding="utf-8")
    )
    _require_config(config)
    weak = {
        "delta_nmse": 1.0,
        "delta_energy_ratio": 0.0,
        "delta_alignment_cosine": 0.0,
        "action_explained_fraction": 0.0,
        "no_action_recovery": 0.0,
        "shuffle_penalty": 0.0,
    }
    verdict, subclassification, inputs = classify_cach_a4(
        config,
        all_validity=True,
        action_metrics={
            "aggregate": weak,
            "by_horizon": {key: weak for key in ("1", "2", "4")},
        },
        common_stable=False,
    )
    assert verdict == "OPERATOR_STOP_CAUSAL_STREAM_WEAK"
    assert subclassification == "A4_CAUSAL_STREAM_WEAK"
    assert inputs["causal_stream_weak_stop"] is True
