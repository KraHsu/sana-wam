"""CPU synthetic checks for the isolated 20-layer REF/CACH mini pair.

No test in this file constructs an optimizer, reads a checkpoint or dataset,
or initializes CUDA.  The pure-Torch proxy is non-scientific and is not the
complete 2B/vendor-GDN admission run.
"""

from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

from sana_wam.cach.staging_variant import CACHStagingVariant
from sana_wam.model.action_chunk_layout import (
    SelectedProprioBinding,
    build_synthetic_chunk_action_layout_for_tests,
    canonical_proprio_row_sha256,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)
from sana_wam.model.cach_minimal_cpu import (
    MinimalCACHContractError,
    MinimalCACHSpec,
    MinimalChunkBatch,
    MinimalCorrectedGDNArchitecture,
    MinimalEphemeralStateOwner,
    MinimalGDNState,
    MinimalLayerState,
    build_minimal_ref_cach_pair,
)
from sana_wam.model.video_backbone.sana.hybrid_cache import (
    LayerKind,
    tensor_digest,
)


SEED = 20260802
BATCH_SIZE = 2


def _layout(*, valid_latent_count: int = 6):
    spec = synthetic_layout_spec(
        frame_chunk_size=2,
        temporal_compression=1,
        video_stride=1,
        action_dim=20,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=valid_latent_count,
        video_stride=1,
        source_row_label=f"minimal-row-{valid_latent_count}",
        episode_label=f"minimal-episode-{valid_latent_count}",
    )
    return build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=valid_latent_count,
        video_valid_mask=(True,) * valid_latent_count,
        action_valid_mask=(True,) * (valid_latent_count - 1),
    )


def _spec() -> MinimalCACHSpec:
    return MinimalCACHSpec(
        latent_dim=6,
        hidden_dim=8,
        action_dim=20,
        depth=20,
        chunk_size=2,
        shared_init_seed=SEED,
        operator_init_seed=SEED + 1,
    )


def _batches(
    layout,
    *,
    seed: int = SEED + 1,
    requires_grad: bool = False,
) -> tuple[MinimalChunkBatch, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches = []
    for chunk in layout.chunks:
        video = torch.zeros(BATCH_SIZE, 2, 6, dtype=torch.float32)
        video[:, : chunk.valid_latent_count] = torch.randn(
            BATCH_SIZE,
            chunk.valid_latent_count,
            6,
            generator=generator,
        )
        actions = torch.zeros(
            BATCH_SIZE,
            chunk.action_slot_capacity,
            20,
            dtype=torch.float32,
        )
        actions[:, : chunk.valid_action_count] = torch.randn(
            BATCH_SIZE,
            chunk.valid_action_count,
            20,
            generator=generator,
        )
        proprio = torch.randn(
            BATCH_SIZE,
            20,
            generator=generator,
            dtype=torch.float32,
        )
        bindings = tuple(
            SelectedProprioBinding(
                layout_instance_digest=layout.layout_instance_digest,
                chunk_id=chunk.chunk_id,
                raw_index=chunk.proprio_raw_index,
                timestamp=chunk.proprio_timestamp,
                source_receipt_digest=chunk.proprio_source_receipt_digest,
                proprio_value_sha256=canonical_proprio_row_sha256(proprio[index]),
            )
            for index in range(BATCH_SIZE)
        )
        video.requires_grad_(requires_grad)
        actions.requires_grad_(requires_grad)
        batches.append(
            MinimalChunkBatch(
                layout_instance_digest=layout.layout_instance_digest,
                chunk_id=chunk.chunk_id,
                video_latents=video,
                frame_valid_mask=torch.tensor(
                    chunk.latent_valid_mask,
                    dtype=torch.bool,
                )
                .view(1, -1)
                .expand(BATCH_SIZE, -1)
                .clone(),
                noisy_actions=actions,
                action_valid_mask=torch.tensor(
                    chunk.action_valid_mask,
                    dtype=torch.bool,
                )
                .view(1, -1)
                .expand(BATCH_SIZE, -1)
                .clone(),
                proprio_state=proprio,
                proprio_bindings=bindings,
                video_timestep=torch.zeros(BATCH_SIZE, dtype=torch.float32),
                action_timestep=torch.zeros(BATCH_SIZE, dtype=torch.float32),
            )
        )
    return tuple(batches)


def _assert_tensor_exact(left: torch.Tensor, right: torch.Tensor) -> None:
    torch.testing.assert_close(left, right, atol=0, rtol=0)


def _assert_layer_state_exact(
    left: tuple[MinimalLayerState, ...],
    right: tuple[MinimalLayerState, ...],
) -> None:
    assert len(left) == len(right) == 20
    for left_layer, right_layer in zip(left, right):
        _assert_tensor_exact(left_layer.main_s_kv, right_layer.main_s_kv)
        _assert_tensor_exact(left_layer.main_s_z, right_layer.main_s_z)


def _assert_output_exact(left, right) -> None:
    assert left.chunk_id == right.chunk_id
    _assert_tensor_exact(left.video_prediction, right.video_prediction)
    _assert_tensor_exact(left.action_prediction, right.action_prediction)


def _parameter_digests(model) -> dict[str, str]:
    values = {
        name: tensor_digest(parameter) for name, parameter in model.named_parameters()
    }
    values.update(
        {
            f"buffer:{name}": tensor_digest(buffer)
            for name, buffer in model.named_buffers()
        }
    )
    return values


def _install_nonzero_diagnostic_seam(model) -> None:
    assert model.staging_variant is CACHStagingVariant.CACH_A
    model.install_synthetic_nonzero_action_seam_for_tests()


def _perturb_actions(
    batches: tuple[MinimalChunkBatch, ...],
    *,
    chunk_id: int,
    amount: float,
    requires_grad: bool | None = None,
) -> tuple[MinimalChunkBatch, ...]:
    changed = list(batches)
    batch = changed[chunk_id]
    actions = batch.noisy_actions.detach().clone()
    valid = batch.action_valid_mask.unsqueeze(-1)
    actions = torch.where(valid, actions + amount, actions)
    actions.requires_grad_(
        batch.noisy_actions.requires_grad if requires_grad is None else requires_grad
    )
    changed[chunk_id] = replace(batch, noisy_actions=actions)
    return tuple(changed)


def _perturb_video(
    batches: tuple[MinimalChunkBatch, ...],
    *,
    chunk_id: int,
    amount: float,
    requires_grad: bool | None = None,
) -> tuple[MinimalChunkBatch, ...]:
    changed = list(batches)
    batch = changed[chunk_id]
    video = batch.video_latents.detach().clone()
    valid = batch.frame_valid_mask.unsqueeze(-1)
    video = torch.where(valid, video + amount, video)
    video.requires_grad_(
        batch.video_latents.requires_grad if requires_grad is None else requires_grad
    )
    changed[chunk_id] = replace(batch, video_latents=video)
    return tuple(changed)


def test_pair_has_20_layer_cpu_topology_named_init_and_unique_delta() -> None:
    pair = build_minimal_ref_cach_pair(spec=_spec(), layout=_layout())

    assert pair.reference is not pair.candidate
    assert len(pair.reference.blocks) == len(pair.candidate.blocks) == 20
    assert all(
        layer.kind is LayerKind.GDN_FULL_HISTORY
        for layer in pair.reference.layer_registry.layers
    )
    assert pair.reference.layer_registry.manifest_digest == (
        pair.candidate.layer_registry.manifest_digest
    )
    for model in (pair.reference, pair.candidate):
        manifest = model.graph_manifest()
        assert manifest["operator_contract"] == ("pure_torch_chunk_causal_gdn_proxy_v1")
        assert manifest["synthetic_test_only"] is True
        assert manifest["complete_2b"] is False
        assert manifest["full_model_admission"] is False
        assert manifest["scientific_result"] is False
        assert manifest["vendor_gdn_numerical_parity"] is False
        assert set(model.disabled_components) == {
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
        }
        pointers = [
            parameter.untyped_storage().data_ptr() for parameter in model.parameters()
        ]
        assert len(pointers) == len(set(pointers))
        assert all(
            parameter.device.type == "cpu" and parameter.dtype is torch.float32
            for parameter in model.parameters()
        )

    pair.assert_integrity()
    assert pair.shared_parameter_digests
    shared_names = {name for name, _digest in pair.shared_parameter_digests}
    assert shared_names.isdisjoint(pair.candidate_only_parameters)
    assert set(pair.reference.state_dict()).isdisjoint(
        set(pair.candidate_only_parameters)
    )
    assert "no_action_slot" in pair.candidate_only_parameters
    assert any(
        name.startswith("action_conditioner.")
        for name in pair.candidate_only_parameters
    )
    assert (
        sum(
            ".action_output_projection." in name
            for name in pair.candidate_only_parameters
        )
        == 40
    )
    candidate_parameters = dict(pair.candidate.named_parameters())
    assert not bool(candidate_parameters["no_action_slot"].count_nonzero())
    for name in pair.candidate_only_parameters:
        if ".action_output_projection." in name:
            assert not bool(candidate_parameters[name].count_nonzero())
    assert any(
        bool(candidate_parameters[name].count_nonzero())
        for name in pair.candidate_only_parameters
        if name.startswith("action_conditioner.") and name.endswith(".weight")
    )
    assert all(
        entry.initializer_role.startswith("candidate_")
        for entry in pair.candidate_only_parameter_inventory
    )
    assert all(
        entry.requires_grad
        for entry in (
            pair.shared_parameter_inventory + pair.candidate_only_parameter_inventory
        )
    )

    rebuilt = build_minimal_ref_cach_pair(spec=_spec(), layout=_layout())
    assert rebuilt.pair_digest == pair.pair_digest
    assert rebuilt.shared_parameter_inventory == pair.shared_parameter_inventory
    assert rebuilt.candidate_only_parameter_inventory == (
        pair.candidate_only_parameter_inventory
    )
    assert rebuilt.reference.architecture_instance_token != (
        pair.reference.architecture_instance_token
    )


def test_operator_seed_changes_only_candidate_operator_inventory() -> None:
    layout = _layout()
    baseline = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    changed_spec = replace(_spec(), operator_init_seed=SEED + 2)
    changed = build_minimal_ref_cach_pair(spec=changed_spec, layout=layout)

    assert baseline.shared_parameter_inventory == changed.shared_parameter_inventory
    baseline_operator = tuple(
        entry.digest
        for entry in baseline.candidate_only_parameter_inventory
        if entry.initializer_role == "candidate_operator_named_uniform_v1"
    )
    changed_operator = tuple(
        entry.digest
        for entry in changed.candidate_only_parameter_inventory
        if entry.initializer_role == "candidate_operator_named_uniform_v1"
    )
    assert baseline_operator
    assert baseline_operator != changed_operator
    assert baseline.pair_digest != changed.pair_digest


def test_zero_candidate_is_exact_reference_and_reference_video_bypasses_actions() -> (
    None
):
    layout = _layout()
    pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    batches = _batches(layout)

    reference = pair.reference.forward_sequence(batches)
    candidate = pair.candidate.forward_sequence(batches)
    assert len(reference.outputs) == len(candidate.outputs) == 3
    for reference_output, candidate_output in zip(
        reference.outputs,
        candidate.outputs,
    ):
        _assert_output_exact(reference_output, candidate_output)
    _assert_layer_state_exact(
        reference.final_state.layers, candidate.final_state.layers
    )
    _assert_tensor_exact(
        reference.final_state.committed_actions,
        candidate.final_state.committed_actions,
    )

    changed_actions = _perturb_actions(batches, chunk_id=0, amount=3.0)
    changed_reference = pair.reference.forward_sequence(changed_actions)
    _assert_tensor_exact(
        reference.video_prediction,
        changed_reference.video_prediction,
    )
    assert not torch.equal(
        reference.action_prediction,
        changed_reference.action_prediction,
    )
    changed_zero_candidate = pair.candidate.forward_sequence(changed_actions)
    _assert_tensor_exact(
        candidate.video_prediction,
        changed_zero_candidate.video_prediction,
    )
    assert not torch.equal(
        candidate.action_prediction,
        changed_zero_candidate.action_prediction,
    )


@pytest.mark.parametrize("arm", ["reference", "candidate", "candidate_nonzero"])
def test_full_sequence_matches_chunk_major_stage_and_commit_exactly(arm: str) -> None:
    layout = _layout()
    pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    if arm == "candidate_nonzero":
        model = pair.candidate
        _install_nonzero_diagnostic_seam(model)
    else:
        model = getattr(pair, arm)
    assert isinstance(model, MinimalCorrectedGDNArchitecture)
    before_parameters = _parameter_digests(model)
    batches = _batches(layout)
    full = model.forward_sequence(batches)
    owner = MinimalEphemeralStateOwner(
        model,
        batch_size=BATCH_SIZE,
        owner_id=f"full-chunk-{arm}",
    )

    chunk_outputs = []
    for batch in batches:
        source = owner.snapshot()
        source_digest = source.state_digest
        denoise = owner.run_denoise_chunk(batch)
        assert owner.snapshot().state_digest == source_digest
        proposal = owner.stage_paired_t0(batch)
        assert owner.snapshot().state_digest == source_digest
        _assert_output_exact(denoise, proposal.output)
        committed = owner.commit_paired(proposal)
        assert committed.revision == batch.chunk_id + 1
        assert committed.next_chunk_id == batch.chunk_id + 1
        chunk_outputs.append(proposal.output)

    assert owner.revision == len(batches)
    for full_output, chunk_output in zip(full.outputs, chunk_outputs):
        _assert_output_exact(full_output, chunk_output)
    streamed = owner.snapshot()
    _assert_layer_state_exact(full.final_state.layers, streamed.layers)
    _assert_tensor_exact(
        full.final_state.committed_actions,
        streamed.committed_actions,
    )
    assert full.final_state.action_cursor == streamed.action_cursor
    assert _parameter_digests(model) == before_parameters
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nonzero_diagnostic_seam_changes_candidate_only() -> None:
    layout = _layout()
    pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    _install_nonzero_diagnostic_seam(pair.candidate)
    batches = _batches(layout)
    changed = _perturb_actions(batches, chunk_id=1, amount=2.0)

    reference = pair.reference.forward_sequence(batches)
    changed_reference = pair.reference.forward_sequence(changed)
    _assert_tensor_exact(
        reference.video_prediction,
        changed_reference.video_prediction,
    )

    candidate = pair.candidate.forward_sequence(batches)
    changed_candidate = pair.candidate.forward_sequence(changed)
    assert not torch.equal(
        candidate.outputs[1].video_prediction,
        changed_candidate.outputs[1].video_prediction,
    )
    assert not torch.equal(
        candidate.outputs[2].video_prediction,
        changed_candidate.outputs[2].video_prediction,
    )
    assert any(
        not torch.equal(left.main_s_kv, right.main_s_kv)
        for left, right in zip(
            candidate.final_state.layers,
            changed_candidate.final_state.layers,
        )
    )
    assert bool(torch.isfinite(changed_candidate.video_prediction).all())


@pytest.mark.parametrize("future_kind", ["video", "action"])
def test_future_chunk_perturbation_leaves_prefix_outputs_and_gradients_exact(
    future_kind: str,
) -> None:
    layout = _layout()
    pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    model = pair.candidate
    _install_nonzero_diagnostic_seam(model)
    baseline_batches = _batches(layout, requires_grad=True)
    comparison_batches = _batches(layout, requires_grad=True)
    if future_kind == "video":
        comparison_batches = _perturb_video(
            comparison_batches,
            chunk_id=2,
            amount=7.0,
            requires_grad=True,
        )
    else:
        comparison_batches = _perturb_actions(
            comparison_batches,
            chunk_id=2,
            amount=-5.0,
            requires_grad=True,
        )

    baseline = model.forward_sequence(baseline_batches)
    comparison = model.forward_sequence(comparison_batches)
    if future_kind == "video":
        assert not torch.equal(
            baseline_batches[2].video_latents,
            comparison_batches[2].video_latents,
        )
    else:
        assert not torch.equal(
            baseline_batches[2].noisy_actions,
            comparison_batches[2].noisy_actions,
        )
    assert not torch.equal(
        baseline.outputs[2].video_prediction,
        comparison.outputs[2].video_prediction,
    )
    for index in (0, 1):
        _assert_output_exact(baseline.outputs[index], comparison.outputs[index])

    def prefix_probe(result) -> torch.Tensor:
        pieces = []
        for output in result.outputs[:2]:
            video_weight = torch.linspace(
                0.25,
                1.25,
                output.video_prediction.numel(),
            ).reshape_as(output.video_prediction)
            action_weight = torch.linspace(
                -0.75,
                0.5,
                output.action_prediction.numel(),
            ).reshape_as(output.action_prediction)
            pieces.append((output.video_prediction * video_weight).sum())
            pieces.append((output.action_prediction * action_weight).sum())
        return torch.stack(pieces).sum()

    parameters = tuple(model.parameters())
    baseline_inputs = tuple(
        tensor
        for batch in baseline_batches
        for tensor in (batch.video_latents, batch.noisy_actions)
    )
    comparison_inputs = tuple(
        tensor
        for batch in comparison_batches
        for tensor in (batch.video_latents, batch.noisy_actions)
    )
    baseline_gradients = torch.autograd.grad(
        prefix_probe(baseline),
        parameters + baseline_inputs,
        allow_unused=True,
    )
    comparison_gradients = torch.autograd.grad(
        prefix_probe(comparison),
        parameters + comparison_inputs,
        allow_unused=True,
    )
    for left, right in zip(
        baseline_gradients[: len(parameters)],
        comparison_gradients[: len(parameters)],
    ):
        if left is None or right is None:
            assert left is right
        else:
            _assert_tensor_exact(left, right)
    baseline_input_gradients = baseline_gradients[len(parameters) :]
    comparison_input_gradients = comparison_gradients[len(parameters) :]
    for chunk_index in range(len(baseline_batches)):
        start = chunk_index * 2
        for left, right in zip(
            baseline_input_gradients[start : start + 2],
            comparison_input_gradients[start : start + 2],
        ):
            if chunk_index < 2:
                if left is None or right is None:
                    assert left is right
                else:
                    _assert_tensor_exact(left, right)
            else:
                assert left is None or not bool(left.count_nonzero())
                assert right is None or not bool(right.count_nonzero())


def test_zero_seam_has_nonzero_gradients_without_parameter_updates() -> None:
    layout = _layout()
    pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    model = pair.candidate
    before = _parameter_digests(model)
    result = model.forward_sequence(_batches(layout))
    video_weight = torch.linspace(
        -1.0,
        1.0,
        result.video_prediction.numel(),
    ).reshape_as(result.video_prediction)
    action_weight = torch.linspace(
        0.5,
        1.5,
        result.action_prediction.numel(),
    ).reshape_as(result.action_prediction)
    probe = (result.video_prediction * video_weight).sum()
    probe = probe + (result.action_prediction * action_weight).sum()
    adapter_parameters = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if ".action_output_projection." in name
    )
    assert len(adapter_parameters) == 40
    gradients = torch.autograd.grad(probe, adapter_parameters)
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert all(bool(gradient.count_nonzero()) for gradient in gradients)
    assert all(
        not bool(parameter.detach().count_nonzero()) for parameter in adapter_parameters
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert _parameter_digests(model) == before


def test_ephemeral_commit_is_exactly_once_and_fail_closed() -> None:
    layout = _layout()
    pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    batches = _batches(layout)
    before_parameters = _parameter_digests(pair.candidate)
    owner = MinimalEphemeralStateOwner(
        pair.candidate,
        batch_size=BATCH_SIZE,
        owner_id="candidate-owner",
    )
    other_owner = MinimalEphemeralStateOwner(
        pair.candidate,
        batch_size=BATCH_SIZE,
        owner_id="other-owner",
    )
    reference_owner = MinimalEphemeralStateOwner(
        pair.reference,
        batch_size=BATCH_SIZE,
        owner_id="candidate-owner",
    )
    rebuilt_pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    rebuilt_owner = MinimalEphemeralStateOwner(
        rebuilt_pair.candidate,
        batch_size=BATCH_SIZE,
        owner_id="candidate-owner",
    )

    original_digest = owner.snapshot().state_digest
    proposal = owner.stage_paired_t0(batches[0])
    alternative = owner.stage_paired_t0(
        _perturb_actions(batches, chunk_id=0, amount=1.0)[0]
    )
    assert owner.snapshot().state_digest == original_digest
    with pytest.raises(MinimalCACHContractError, match="another owner"):
        other_owner.commit_paired(proposal)
    with pytest.raises(MinimalCACHContractError, match="not issued"):
        reference_owner.commit_paired(proposal)
    forged_copy = replace(proposal)
    with pytest.raises(MinimalCACHContractError, match="not issued"):
        owner.commit_paired(forged_copy)
    rebuilt_proposal = rebuilt_owner.stage_paired_t0(batches[0])
    with pytest.raises(MinimalCACHContractError, match="not issued"):
        owner.commit_paired(rebuilt_proposal)
    assert other_owner.revision == reference_owner.revision == 0
    assert owner.snapshot().state_digest == original_digest

    first_state = owner.commit_paired(proposal)
    assert first_state.revision == first_state.next_chunk_id == 1
    after_first = owner.snapshot().state_digest
    with pytest.raises(MinimalCACHContractError, match="already committed"):
        owner.commit_paired(proposal)
    assert owner.snapshot().state_digest == after_first
    with pytest.raises(MinimalCACHContractError, match="stale or out of order"):
        owner.commit_paired(alternative)
    assert owner.snapshot().state_digest == after_first

    second = owner.stage_paired_t0(batches[1])
    second_state = owner.commit_paired(second)
    assert second_state.revision == second_state.next_chunk_id == 2
    expected_actions = torch.cat(
        [
            batches[0].noisy_actions[:, : layout.chunks[0].valid_action_count],
            batches[1].noisy_actions[:, : layout.chunks[1].valid_action_count],
        ],
        dim=1,
    )
    _assert_tensor_exact(second_state.committed_actions, expected_actions)

    stale_across_reset = owner.stage_paired_t0(batches[2])
    reset_state = owner.reset()
    assert reset_state.revision == reset_state.next_chunk_id == 0
    assert reset_state.action_cursor == 0
    assert not bool(reset_state.committed_actions.numel())
    with pytest.raises(MinimalCACHContractError, match="owner/epoch"):
        owner.commit_paired(stale_across_reset)
    assert owner.revision == owner.next_chunk_id == 0
    assert _parameter_digests(pair.candidate) == before_parameters
    assert all(parameter.grad is None for parameter in pair.candidate.parameters())


@pytest.mark.parametrize(
    ("drift_kind", "error_match"),
    [
        ("bytes", "snapshot changed"),
        ("requires_grad", "trainable classification"),
        ("storage", "storage topology"),
    ],
)
def test_staged_proposal_rejects_unregistered_parameter_drift(
    drift_kind: str,
    error_match: str,
) -> None:
    layout = _layout()
    pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    owner = MinimalEphemeralStateOwner(
        pair.candidate,
        batch_size=BATCH_SIZE,
        owner_id="parameter-drift-owner",
    )
    proposal = owner.stage_paired_t0(_batches(layout)[0])
    source_digest = owner.snapshot().state_digest
    parameter = pair.candidate.video_output_projection.weight
    if drift_kind == "bytes":
        with torch.no_grad():
            parameter[0, 0].add_(1.0)
    elif drift_kind == "requires_grad":
        parameter.requires_grad_(False)
    else:
        with torch.no_grad():
            parameter.data = parameter.detach().clone()
    with pytest.raises(MinimalCACHContractError, match=error_match):
        owner.commit_paired(proposal)
    assert owner.snapshot().state_digest == source_digest


def test_staged_proposal_rejects_in_place_output_mutation() -> None:
    layout = _layout()
    pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    owner = MinimalEphemeralStateOwner(
        pair.candidate,
        batch_size=BATCH_SIZE,
        owner_id="proposal-mutation-owner",
    )
    proposal = owner.stage_paired_t0(_batches(layout)[0])
    source_digest = owner.snapshot().state_digest
    with torch.no_grad():
        proposal.output.video_prediction[0, 0, 0].add_(1.0)
    with pytest.raises(MinimalCACHContractError, match="proposal was mutated"):
        owner.commit_paired(proposal)
    assert owner.snapshot().state_digest == source_digest


def test_partial_tail_and_bad_padding_fail_closed() -> None:
    layout = _layout(valid_latent_count=5)
    pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
    batches = _batches(layout)
    assert layout.chunks[-1].latent_valid_mask == (True, False)
    result = pair.candidate.forward_sequence(batches)
    assert not bool(result.outputs[-1].video_prediction[:, 1:].count_nonzero())
    last_chunk = layout.chunks[-1]
    assert last_chunk.valid_action_count < last_chunk.action_slot_capacity
    assert not bool(
        result.outputs[-1]
        .action_prediction[:, last_chunk.valid_action_count :]
        .count_nonzero()
    )

    before_parameters = _parameter_digests(pair.candidate)
    partial_owner = MinimalEphemeralStateOwner(
        pair.candidate,
        batch_size=BATCH_SIZE,
        owner_id="partial-tail-owner",
    )
    for batch in batches:
        partial_owner.commit_paired(partial_owner.stage_paired_t0(batch))
    partial_state = partial_owner.snapshot()
    assert partial_state.revision == len(layout.chunks)
    expected_actions = torch.cat(
        tuple(
            batch.noisy_actions[:, : chunk.valid_action_count]
            for batch, chunk in zip(batches, layout.chunks)
        ),
        dim=1,
    )
    assert expected_actions.shape[1] == layout.valid_action_count
    _assert_tensor_exact(partial_state.committed_actions, expected_actions)
    assert _parameter_digests(pair.candidate) == before_parameters

    poisoned = batches[-1].video_latents.detach().clone()
    poisoned[:, 1] = 1.0
    bad_batches = batches[:-1] + (replace(batches[-1], video_latents=poisoned),)
    with pytest.raises(MinimalCACHContractError, match="padded video"):
        pair.candidate.forward_sequence(bad_batches)

    poisoned_actions = batches[-1].noisy_actions.detach().clone()
    poisoned_actions[:, last_chunk.valid_action_count :] = 1.0
    bad_action_batches = batches[:-1] + (
        replace(batches[-1], noisy_actions=poisoned_actions),
    )
    with pytest.raises(MinimalCACHContractError, match="padded actions"):
        pair.candidate.forward_sequence(bad_action_batches)

    owner = MinimalEphemeralStateOwner(
        pair.candidate,
        batch_size=BATCH_SIZE,
        owner_id="timestep-owner",
    )
    nonzero_t = replace(
        batches[0],
        video_timestep=torch.ones(BATCH_SIZE, dtype=torch.float32),
    )
    owner.run_denoise_chunk(nonzero_t)
    with pytest.raises(MinimalCACHContractError, match="exact video/action t=0"):
        owner.stage_paired_t0(nonzero_t)
    assert owner.revision == 0


def test_public_surface_has_no_loader_target_or_optimizer_arguments() -> None:
    forbidden = {
        "checkpoint",
        "clean_action",
        "clean_video",
        "dataset",
        "evaluator",
        "future",
        "loader",
        "optimizer",
        "target",
        "trainer",
    }
    surfaces = (
        MinimalCACHSpec,
        MinimalChunkBatch,
        MinimalCorrectedGDNArchitecture.forward,
        MinimalCorrectedGDNArchitecture.forward_sequence,
        MinimalCorrectedGDNArchitecture.run_denoise_chunk,
        MinimalCorrectedGDNArchitecture.stage_paired_t0,
    )
    for surface in surfaces:
        names = set(inspect.signature(surface).parameters)
        assert names.isdisjoint(forbidden)

    layout = _layout()
    model = build_minimal_ref_cach_pair(spec=_spec(), layout=layout).candidate
    bad_model = build_minimal_ref_cach_pair(spec=_spec(), layout=layout).candidate
    state: MinimalGDNState = bad_model.empty_state(batch_size=BATCH_SIZE)
    bad_model.to(dtype=torch.float64)
    with pytest.raises(MinimalCACHContractError, match="CPU/FP32"):
        bad_model.run_denoise_chunk(_batches(layout)[0], state)
    assert model.graph_manifest()["scientific_result"] is False


def test_internal_allocations_ignore_a_non_cpu_global_default_device() -> None:
    previous_default = torch.get_default_device()
    try:
        # ``meta`` exercises the same default-device hazard without touching
        # or initializing CUDA during this CPU-only test.
        torch.set_default_device("meta")
        layout = _layout()
        pair = build_minimal_ref_cach_pair(spec=_spec(), layout=layout)
        for model in (pair.reference, pair.candidate):
            assert all(
                parameter.device.type == "cpu" for parameter in model.parameters()
            )
            state = model.empty_state(batch_size=BATCH_SIZE)
            assert state.committed_actions.device.type == "cpu"
            assert all(layer.main_s_kv.device.type == "cpu" for layer in state.layers)
    finally:
        torch.set_default_device(previous_default)
