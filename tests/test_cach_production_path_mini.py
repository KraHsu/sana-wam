"""Hard CPU/synthetic C0--C8 checks for the production-shaped Phase-C mini."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import stat
import threading
import time

import pytest
import torch
from torch import Tensor

from sana_wam.cach.authority import CACHAuthorityError
from sana_wam.cach.stage2_failure_evidence import (
    Stage2FailureInjectionReceipt,
    freeze_stage2_failure_evidence,
)
from sana_wam.model import build_architecture
from sana_wam.model.cach_production_path_mini import (
    ProductionPathMiniArm,
    ProductionPathMiniContractError,
    ProductionPathMiniDispatcher,
    ProductionPathMiniEpisode,
    ProductionPathMiniSpec,
    ProductionPathMiniVideoBackbone,
    build_production_path_mini_episode_for_tests,
    build_production_path_mini_layout_for_tests,
    build_production_path_mini_pair_for_tests,
)
from sana_wam.model.video_backbone.sana import hybrid_cache as hc
from sana_wam.model.video_backbone.sana import hybrid_cache_stage2b as h2


@pytest.fixture(scope="module")
def pair():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    return build_production_path_mini_pair_for_tests()


@pytest.fixture(scope="module")
def episode(pair):
    return build_production_path_mini_episode_for_tests(pair.spec, pair.layout)


def _assert_raw_equal(left: Tensor, right: Tensor) -> None:
    assert left.dtype == right.dtype
    assert left.device == right.device
    assert tuple(left.shape) == tuple(right.shape)
    assert hc.tensor_digest(left) == hc.tensor_digest(right)


def _assert_optional_grad_equal(left: Tensor | None, right: Tensor | None) -> None:
    assert (left is None) == (right is None)
    if left is not None:
        assert right is not None
        _assert_raw_equal(left, right)


def _state_tensor_map(state: h2.Stage2BTemporalState, layer_index: int):
    return {
        name: snapshot.clone_tensor()
        for name, snapshot in state.layer_states[layer_index].tensors
    }


def _history_tensor(state: h2.Stage2BTemporalState) -> Tensor:
    history = state.committed_action_history
    assert history is not None
    return torch.cat([span.clone_actions() for span in history.spans], dim=1)


def _objective_prefix(result, through_chunk: int) -> Tensor:
    terms = []
    for index in range(through_chunk + 1):
        terms.append(result.video_predictions[index].square().sum())
        terms.append(result.action_predictions[index].square().sum())
    return torch.stack(terms).sum()


def _module_tensor_fingerprint(module) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (name, tensor._version, hc.tensor_digest(tensor))
        for name, tensor in (
            list(module.named_parameters()) + list(module.named_buffers())
        )
    )


def _requires_grad_episode(source: ProductionPathMiniEpisode):
    cloned = source.owned_clone()
    for chunk in cloned.chunks:
        chunk.noisy_video.requires_grad_(True)
        chunk.noisy_actions.requires_grad_(True)
        chunk.clean_video_target.requires_grad_(True)
        chunk.clean_action_target.requires_grad_(True)
    return cloned


def _perturb_valid(value: Tensor, mask: Tensor, amount: float) -> Tensor:
    result = value.detach().clone()
    if value.ndim == 5:
        result = result + amount * mask[:, None, :, None, None]
    elif value.ndim == 3:
        result = result + amount * mask.unsqueeze(-1)
    else:  # pragma: no cover - helper has only two registered tensor shapes.
        raise AssertionError("unregistered perturbation tensor rank")
    result.requires_grad_(True)
    return result


def test_phase_c_c4_builder_rejects_drift_before_model_allocation(monkeypatch) -> None:
    import sana_wam.model.cach_production_path_mini as mini

    calls = 0
    original = mini._ProductionPathMiniDiT.__init__

    def forbidden_allocation(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("graph allocation occurred before config rejection")

    monkeypatch.setattr(mini._ProductionPathMiniDiT, "__init__", forbidden_allocation)
    expected_forbidden_keys = frozenset(
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
    assert mini._FORBIDDEN_CONFIG_KEYS == expected_forbidden_keys
    rejected = tuple({key: object()} for key in sorted(expected_forbidden_keys)) + (
        {"observed_prefix_chunks": 1},
        {"episode_bootstrap": "legacy_clean_prefix"},
        {"device": "cuda"},
        {"dtype": "bfloat16"},
        {"unknown_phase_c_key": True},
    )
    for config in rejected:
        with pytest.raises(ProductionPathMiniContractError):
            build_production_path_mini_pair_for_tests(requested_config=config)
    assert calls == 0
    monkeypatch.setattr(mini._ProductionPathMiniDiT, "__init__", original)

    with pytest.raises(CACHAuthorityError, match="Stage 2 authority"):
        build_architecture({"variant": "cach_sana_wam_v0"})
    with pytest.raises(ProductionPathMiniContractError, match="immutable"):
        ProductionPathMiniSpec(hidden_dim=1024)
    with pytest.raises(ProductionPathMiniContractError, match="three fixed"):
        build_production_path_mini_layout_for_tests(
            ProductionPathMiniSpec(), valid_raw_count=1_000_000
        )
    with pytest.raises(ProductionPathMiniContractError, match="checkpoint"):
        ProductionPathMiniVideoBackbone.from_pretrained("/forbidden")
    with pytest.raises(ProductionPathMiniContractError, match="sealed"):
        ProductionPathMiniVideoBackbone.from_mini_config(hidden_size=1024)

    sealed_pair = build_production_path_mini_pair_for_tests()
    assert not torch.cuda.is_initialized()
    with pytest.raises(ProductionPathMiniContractError, match="transforms"):
        sealed_pair.candidate.to(dtype=torch.float64)
    with pytest.raises(ProductionPathMiniContractError, match="transforms"):
        sealed_pair.candidate.cuda()
    with pytest.raises(ProductionPathMiniContractError, match="state loading"):
        sealed_pair.candidate.load_state_dict({})
    with pytest.raises(ProductionPathMiniContractError, match="eval-only"):
        sealed_pair.candidate.train()
    alternate_layout = build_production_path_mini_layout_for_tests(
        sealed_pair.spec, valid_raw_count=17
    )
    with pytest.raises(ProductionPathMiniContractError, match="canonical fixed"):
        ProductionPathMiniArm(
            spec=sealed_pair.spec,
            layout=alternate_layout,
            staging_variant=sealed_pair.candidate.staging_variant,
        )
    assert not torch.cuda.is_initialized()

    previous_default = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        direct_cpu_video = ProductionPathMiniVideoBackbone(
            spec=sealed_pair.spec,
            staging_variant=sealed_pair.candidate.staging_variant,
        )
        cpu_forced_pair = build_production_path_mini_pair_for_tests()
        cpu_forced_episode = build_production_path_mini_episode_for_tests(
            cpu_forced_pair.spec, cpu_forced_pair.layout
        )
    finally:
        torch.set_default_device(previous_default)
    assert all(
        tensor.device.type == "cpu"
        for arm in (cpu_forced_pair.reference, cpu_forced_pair.candidate)
        for tensor in list(arm.parameters()) + list(arm.buffers())
    )
    assert all(
        chunk.noisy_video.device.type == "cpu"
        and chunk.clean_video_target.device.type == "cpu"
        and chunk.noisy_actions.device.type == "cpu"
        and chunk.clean_action_target.device.type == "cpu"
        for chunk in cpu_forced_episode.chunks
    )
    assert all(
        tensor.device.type == "cpu"
        for tensor in list(direct_cpu_video.parameters())
        + list(direct_cpu_video.buffers())
    )
    with pytest.raises(ProductionPathMiniContractError, match="eval-only"):
        direct_cpu_video.train()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ProductionPathMiniContractError, match="default dtype"):
            build_production_path_mini_pair_for_tests()
        with pytest.raises(ProductionPathMiniContractError, match="default dtype"):
            ProductionPathMiniVideoBackbone(
                spec=sealed_pair.spec,
                staging_variant=sealed_pair.candidate.staging_variant,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    assert not torch.cuda.is_initialized()


def test_phase_c_c1_c4_c8_graph_inventory_and_named_init(pair) -> None:
    pair.assert_integrity()
    assert pair.layout.synthetic_test_only is True
    bootstrap_time = hc.ContentTime.from_layout(
        episode_id="phase-c-bootstrap-contract",
        episode_epoch=0,
        layout=pair.layout,
        chunk=pair.layout.chunks[0],
    )
    assert bootstrap_time.bootstrap_mode == "first_frame_pinned"
    assert bootstrap_time.observed_prefix_chunks == 0
    assert pair.reference.layout is pair.layout
    assert pair.candidate.layout is pair.layout
    assert len(pair.reference.registry.layers) == 20
    assert pair.reference.registry.manifest_digest == pair.candidate.registry.manifest_digest
    assert all(layer.kind is hc.LayerKind.GDN_FULL_HISTORY for layer in pair.reference.registry.layers)
    assert all(layer.camera_enabled is False for layer in pair.reference.registry.layers)
    assert all(layer.main_shortconv_enabled is True for layer in pair.reference.registry.layers)
    assert all(layer.ffn_tconv_enabled is True for layer in pair.reference.registry.layers)
    assert len(pair.zero_adapter_parameter_names) == 40
    assert not bool(pair.reference.cach_core.no_action_slot.detach().count_nonzero())
    assert not bool(pair.candidate.cach_core.no_action_slot.detach().count_nonzero())

    candidate_parameters = dict(pair.candidate.named_parameters())
    assert all(
        not bool(candidate_parameters[name].detach().count_nonzero())
        for name in pair.zero_adapter_parameter_names
    )
    candidate_only = tuple(entry.name for entry in pair.candidate_only_inventory)
    assert len(candidate_only) == 44
    assert all(
        name.startswith("video_backbone.dit.delta_pose_embedder.")
        or ".operator.action_output_projection." in name
        for name in candidate_only
    )
    assert not set(dict(pair.reference.named_parameters())) - set(
        dict(pair.candidate.named_parameters())
    )

    for arm in (pair.reference, pair.candidate):
        graph = arm.graph_manifest()
        assert graph["afcc_reachable"] is False
        assert graph["anchors"] == 0
        assert graph["attnres_modules"] == 0
        assert graph["self_forcing_modules"] == 0
        assert graph["complete_2b"] is False
        assert graph["formal_admission"] is False
        assert graph["vendor_gdn_numerical_parity"] is False
        names = tuple(name.casefold() for name, _ in arm.named_modules())
        for forbidden in ("afcc", "anchor", "attnres", "self_forcing"):
            assert all(forbidden not in name for name in names)


def test_phase_c_c1_exact_reference_bypass_and_zero_candidate_identity(pair, episode) -> None:
    reference = pair.reference.run_full_sequence(episode)
    candidate = pair.candidate.run_full_sequence(episode)
    expected_chunks = len(episode.chunks)
    assert len(reference.video_predictions) == expected_chunks
    assert len(candidate.video_predictions) == expected_chunks
    assert len(reference.action_predictions) == expected_chunks
    assert len(candidate.action_predictions) == expected_chunks
    assert len(reference.final_layer_payloads) == 20
    assert len(candidate.final_layer_payloads) == 20
    assert tuple(
        payload.layer_index for payload in reference.final_layer_payloads
    ) == tuple(range(20))
    assert tuple(
        payload.layer_index for payload in candidate.final_layer_payloads
    ) == tuple(range(20))
    for left, right in zip(reference.video_predictions, candidate.video_predictions):
        _assert_raw_equal(left, right)
    for left, right in zip(reference.action_predictions, candidate.action_predictions):
        _assert_raw_equal(left, right)
    for left, right in zip(
        reference.final_layer_payloads, candidate.final_layer_payloads
    ):
        assert left.layer_index == right.layer_index
        assert set(left.tensors) == set(right.tensors)
        for name in left.tensors:
            _assert_raw_equal(left.tensors[name], right.tensors[name])

    changed_actions = episode.chunks[1].noisy_actions.detach().clone()
    changed_actions[:, : pair.layout.chunks[1].valid_action_count] += 3.0
    perturbed = episode.owned_clone(
        chunk_updates={1: {"noisy_actions": changed_actions}}
    )
    reference_changed = pair.reference.run_full_sequence(perturbed)
    candidate_changed = pair.candidate.run_full_sequence(perturbed)
    assert len(reference_changed.video_predictions) == expected_chunks
    assert len(candidate_changed.video_predictions) == expected_chunks
    assert len(reference_changed.action_predictions) == expected_chunks
    assert len(candidate_changed.action_predictions) == expected_chunks
    assert len(reference_changed.final_layer_payloads) == 20
    assert len(candidate_changed.final_layer_payloads) == 20
    for baseline, changed in zip(
        reference.video_predictions, reference_changed.video_predictions
    ):
        _assert_raw_equal(baseline, changed)
    for baseline, changed in zip(
        candidate.video_predictions, candidate_changed.video_predictions
    ):
        _assert_raw_equal(baseline, changed)
    for baseline, changed in zip(
        reference.final_layer_payloads, reference_changed.final_layer_payloads
    ):
        for name in baseline.tensors:
            _assert_raw_equal(baseline.tensors[name], changed.tensors[name])
    for baseline, changed in zip(
        candidate.final_layer_payloads, candidate_changed.final_layer_payloads
    ):
        for name in baseline.tensors:
            _assert_raw_equal(baseline.tensors[name], changed.tensors[name])


def test_phase_c_c0_future_and_target_separation_reverse_grad_and_jvp(pair, episode) -> None:
    base = _requires_grad_episode(episode)
    changed = _requires_grad_episode(episode)
    probe = 1
    future = 2
    probe_batch = changed.chunks[probe]
    future_batch = changed.chunks[future]
    changed_chunks = list(changed.chunks)
    changed_chunks[probe] = replace(
        probe_batch,
        clean_video_target=_perturb_valid(
            probe_batch.clean_video_target,
            probe_batch.frame_valid_mask,
            5.0,
        ),
        clean_action_target=_perturb_valid(
            probe_batch.clean_action_target,
            probe_batch.action_valid_mask,
            7.0,
        ),
    )
    changed_chunks[future] = replace(
        future_batch,
        noisy_video=_perturb_valid(
            future_batch.noisy_video,
            future_batch.frame_valid_mask,
            11.0,
        ),
        noisy_actions=_perturb_valid(
            future_batch.noisy_actions,
            future_batch.action_valid_mask,
            13.0,
        ),
        clean_video_target=_perturb_valid(
            future_batch.clean_video_target,
            future_batch.frame_valid_mask,
            17.0,
        ),
        clean_action_target=_perturb_valid(
            future_batch.clean_action_target,
            future_batch.action_valid_mask,
            19.0,
        ),
    )
    changed = ProductionPathMiniEpisode(
        layout=changed.layout,
        chunks=tuple(changed_chunks),
        recipe_digest=changed.recipe_digest,
    )

    base_result = pair.candidate.run_full_sequence(base)
    changed_result = pair.candidate.run_full_sequence(changed)
    for index in range(probe + 1):
        _assert_raw_equal(
            base_result.video_predictions[index],
            changed_result.video_predictions[index],
        )
        _assert_raw_equal(
            base_result.action_predictions[index],
            changed_result.action_predictions[index],
        )

    parameters = tuple(pair.candidate.parameters())
    base_inputs = tuple(
        tensor
        for chunk in base.chunks
        for tensor in (
            chunk.noisy_video,
            chunk.noisy_actions,
            chunk.clean_video_target,
            chunk.clean_action_target,
        )
    )
    changed_inputs = tuple(
        tensor
        for chunk in changed.chunks
        for tensor in (
            chunk.noisy_video,
            chunk.noisy_actions,
            chunk.clean_video_target,
            chunk.clean_action_target,
        )
    )
    base_grads = torch.autograd.grad(
        _objective_prefix(base_result, probe),
        parameters + base_inputs,
        allow_unused=True,
    )
    changed_grads = torch.autograd.grad(
        _objective_prefix(changed_result, probe),
        parameters + changed_inputs,
        allow_unused=True,
    )
    for left, right in zip(base_grads, changed_grads):
        _assert_optional_grad_equal(left, right)

    input_offset = len(parameters)
    future_offset = input_offset + future * 4
    assert base_grads[future_offset] is None
    assert base_grads[future_offset + 1] is None
    probe_clean_offset = input_offset + probe * 4 + 2
    assert base_grads[probe_clean_offset] is None
    assert base_grads[probe_clean_offset + 1] is None

    future_video = episode.chunks[future].noisy_video.detach().clone()

    def prefix_from_future(value: Tensor) -> Tensor:
        chunks = list(episode.chunks)
        chunks[future] = replace(chunks[future], noisy_video=value)
        bound = ProductionPathMiniEpisode(
            layout=episode.layout,
            chunks=tuple(chunks),
            recipe_digest=episode.recipe_digest,
        )
        result = pair.candidate.run_full_sequence(bound)
        return torch.cat(
            [
                result.video_predictions[index].flatten()
                for index in range(probe + 1)
            ]
            + [
                result.action_predictions[index].flatten()
                for index in range(probe + 1)
            ]
        )

    _, tangent = torch.autograd.functional.jvp(
        prefix_from_future,
        future_video,
        torch.ones_like(future_video),
        create_graph=False,
        strict=False,
    )
    assert not bool(tangent.count_nonzero())

    clean_probe_video = _perturb_valid(
        episode.chunks[probe].clean_video_target,
        episode.chunks[probe].frame_valid_mask,
        2.5,
    ).detach()
    clean_probe_only = episode.owned_clone(
        chunk_updates={probe: {"clean_video_target": clean_probe_video}}
    )
    clean_probe_result = pair.candidate.run_full_sequence(clean_probe_only)
    baseline_result = pair.candidate.run_full_sequence(episode)
    for index in range(probe + 1):
        _assert_raw_equal(
            baseline_result.video_predictions[index],
            clean_probe_result.video_predictions[index],
        )
        _assert_raw_equal(
            baseline_result.action_predictions[index],
            clean_probe_result.action_predictions[index],
        )
    assert hc.tensor_digest(baseline_result.video_predictions[future]) != (
        hc.tensor_digest(clean_probe_result.video_predictions[future])
    )

    future_video = _perturb_valid(
        episode.chunks[future].noisy_video,
        episode.chunks[future].frame_valid_mask,
        1.75,
    ).detach()
    future_video_only = episode.owned_clone(
        chunk_updates={future: {"noisy_video": future_video}}
    )
    future_action = _perturb_valid(
        episode.chunks[future].noisy_actions,
        episode.chunks[future].action_valid_mask,
        1.25,
    ).detach()
    future_action_only = episode.owned_clone(
        chunk_updates={future: {"noisy_actions": future_action}}
    )
    for arm in (pair.reference, pair.candidate):
        baseline_dispatcher = ProductionPathMiniDispatcher(
            arm=arm,
            episode=episode,
            episode_id=f"phase-c-c0-base-{arm.staging_variant.value}",
        )
        video_dispatcher = ProductionPathMiniDispatcher(
            arm=arm,
            episode=future_video_only,
            episode_id=f"phase-c-c0-video-{arm.staging_variant.value}",
        )
        action_dispatcher = ProductionPathMiniDispatcher(
            arm=arm,
            episode=future_action_only,
            episode_id=f"phase-c-c0-action-{arm.staging_variant.value}",
        )
        for chunk_id in range(future):
            baseline_read = baseline_dispatcher.run_readonly_chunk(
                chunk_id=chunk_id
            )
            video_read = video_dispatcher.run_readonly_chunk(chunk_id=chunk_id)
            action_read = action_dispatcher.run_readonly_chunk(chunk_id=chunk_id)
            _assert_raw_equal(
                baseline_read.video_prediction, video_read.video_prediction
            )
            _assert_raw_equal(
                baseline_read.video_prediction, action_read.video_prediction
            )
            _assert_raw_equal(
                baseline_read.action_prediction, video_read.action_prediction
            )
            _assert_raw_equal(
                baseline_read.action_prediction, action_read.action_prediction
            )
            baseline_dispatcher.commit_paired(chunk_id=chunk_id)
            video_dispatcher.commit_paired(chunk_id=chunk_id)
            action_dispatcher.commit_paired(chunk_id=chunk_id)
        baseline_future = baseline_dispatcher.run_readonly_chunk(chunk_id=future)
        video_future = video_dispatcher.run_readonly_chunk(chunk_id=future)
        action_future = action_dispatcher.run_readonly_chunk(chunk_id=future)
        assert hc.tensor_digest(baseline_future.video_prediction) != (
            hc.tensor_digest(video_future.video_prediction)
        )
        assert hc.tensor_digest(baseline_future.action_prediction) != (
            hc.tensor_digest(video_future.action_prediction)
        )
        _assert_raw_equal(
            baseline_future.video_prediction, action_future.video_prediction
        )
        assert hc.tensor_digest(baseline_future.action_prediction) != (
            hc.tensor_digest(action_future.action_prediction)
        )


def test_phase_c_c1_zero_adapter_reverse_grad_and_parameter_jvp(pair, episode) -> None:
    reference_before = _module_tensor_fingerprint(pair.reference)
    candidate_before = _module_tensor_fingerprint(pair.candidate)
    names = pair.zero_adapter_parameter_names
    parameters = dict(pair.candidate.named_parameters())
    tracked = tuple(parameters[name] for name in names)
    versions = tuple(parameter._version for parameter in tracked)
    digests = tuple(hc.tensor_digest(parameter) for parameter in tracked)

    objective = pair.candidate(episode).square().sum()
    gradients = torch.autograd.grad(objective, tracked)
    assert len(gradients) == 40
    for gradient in gradients:
        assert bool(torch.isfinite(gradient).all())
        assert bool(gradient.count_nonzero())

    base_parameters = dict(pair.candidate.named_parameters())
    primals = tuple(
        base_parameters[name].detach().clone().requires_grad_(True)
        for name in names
    )
    tangents = tuple(
        torch.full_like(value, (index + 1) / 1000.0)
        for index, value in enumerate(primals)
    )

    def parameter_direction(*values: Tensor) -> Tensor:
        mapping = dict(base_parameters)
        mapping.update(dict(zip(names, values)))
        return torch.func.functional_call(
            pair.candidate,
            mapping,
            (episode,),
            strict=False,
        )

    _, parameter_jvp = torch.func.jvp(
        parameter_direction,
        primals,
        tangents,
    )
    assert bool(torch.isfinite(parameter_jvp).all())
    assert bool(parameter_jvp.count_nonzero())
    assert tuple(parameter._version for parameter in tracked) == versions
    assert tuple(hc.tensor_digest(parameter) for parameter in tracked) == digests
    assert _module_tensor_fingerprint(pair.reference) == reference_before
    assert _module_tensor_fingerprint(pair.candidate) == candidate_before
    assert all(parameter.grad is None for parameter in pair.reference.parameters())
    assert all(parameter.grad is None for parameter in pair.candidate.parameters())


def test_phase_c_c2_c7_c8_full_vs_chunk_and_atomic_growth(pair, episode) -> None:
    for batch in episode.chunks:
        assert hc.tensor_digest(batch.noisy_video) != hc.tensor_digest(
            batch.clean_video_target
        )
        assert hc.tensor_digest(batch.noisy_actions) != hc.tensor_digest(
            batch.clean_action_target
        )
        assert bool(batch.video_timestep.count_nonzero())
        assert bool(batch.action_timestep.count_nonzero())
    for arm in (pair.reference, pair.candidate):
        full = arm.run_full_sequence(episode)
        assert len(full.video_predictions) == len(episode.chunks)
        assert len(full.action_predictions) == len(episode.chunks)
        assert len(full.final_layer_payloads) == 20
        assert tuple(
            payload.layer_index for payload in full.final_layer_payloads
        ) == tuple(range(20))
        assert all(
            set(payload.tensors)
            == {
                "ffn_tconv_left_context",
                "main_s_kv",
                "main_s_z",
                "main_shortconv_left_context",
            }
            for payload in full.final_layer_payloads
        )
        dispatcher = ProductionPathMiniDispatcher(
            arm=arm,
            episode=episode,
            episode_id=f"phase-c-equivalence-{arm.staging_variant.value}",
        )
        for chunk_id in range(len(pair.layout.chunks)):
            before = dispatcher.state_owner.state
            pointer_before = dispatcher.state_owner.live_pointer_identity_for_tests
            swaps_before = dispatcher.state_owner.pointer_swap_count
            readonly = dispatcher.run_readonly_chunk(chunk_id=chunk_id)
            readonly_repeat = dispatcher.run_readonly_chunk(chunk_id=chunk_id)
            after_denoise = dispatcher.state_owner.state
            assert dispatcher.state_owner.live_pointer_identity_for_tests == pointer_before
            assert dispatcher.state_owner.pointer_swap_count == swaps_before
            assert after_denoise.state_manifest_digest == before.state_manifest_digest
            assert after_denoise.revision == before.revision
            _assert_raw_equal(
                readonly_repeat.video_prediction, readonly.video_prediction
            )
            _assert_raw_equal(
                readonly_repeat.action_prediction, readonly.action_prediction
            )
            _assert_raw_equal(
                readonly.video_prediction, full.video_predictions[chunk_id]
            )
            _assert_raw_equal(
                readonly.action_prediction, full.action_predictions[chunk_id]
            )

            receipt = dispatcher.commit_paired(chunk_id=chunk_id)
            after_commit = dispatcher.state_owner.state
            assert receipt.ephemeral is True
            assert receipt.durable_publication is False
            assert receipt.controller_commit is False
            assert receipt.revision_after == receipt.revision_before + 1
            assert after_commit.revision == chunk_id + 1
            assert after_commit.next_chunk_id == chunk_id + 1
            assert after_commit.action_cursor == pair.layout.chunks[chunk_id].action_end
            assert dispatcher.state_owner.pointer_swap_count == swaps_before + 1
            assert dispatcher.state_owner.live_pointer_identity_for_tests != pointer_before
            for layer in after_commit.layer_states:
                assert layer.through == receipt_to_content_time(
                    dispatcher, chunk_id
                )
                assert set(name for name, _ in layer.tensors) == {
                    "ffn_tconv_left_context",
                    "main_s_kv",
                    "main_s_z",
                    "main_shortconv_left_context",
                }
                tensors = {
                    name: snapshot.clone_tensor()
                    for name, snapshot in layer.tensors
                }
                assert tuple(tensors["main_s_kv"].shape) == (1, 1, 4, 4)
                assert tuple(tensors["main_s_z"].shape) == (1, 1, 4, 1)
                assert tuple(
                    tensors["main_shortconv_left_context"].shape
                ) == (1, 3, 4)
                assert tuple(tensors["ffn_tconv_left_context"].shape) == (
                    1,
                    4,
                    1,
                    1,
                )

        final_state = dispatcher.state_owner.state
        _assert_raw_equal(_history_tensor(final_state), full.committed_actions)
        for payload in full.final_layer_payloads:
            observed = _state_tensor_map(final_state, payload.layer_index)
            for name, value in payload.tensors.items():
                _assert_raw_equal(observed[name], value)


def receipt_to_content_time(
    dispatcher: ProductionPathMiniDispatcher, chunk_id: int
) -> hc.ContentTime:
    state = dispatcher.state_owner.state
    return hc.ContentTime.from_layout(
        episode_id=state.episode_id,
        episode_epoch=state.episode_epoch,
        layout=dispatcher.layout,
        chunk=dispatcher.layout.chunks[chunk_id],
    )


def test_phase_c_c3_one_layout_binding_and_pre_operator_rejections(
    pair, episode, monkeypatch
) -> None:
    dispatcher = ProductionPathMiniDispatcher(
        arm=pair.candidate,
        episode=episode,
        episode_id="phase-c-layout-binding",
    )
    assert dispatcher.layout is pair.layout
    assert dispatcher.state_owner._layout is pair.layout
    assert dispatcher.state_owner.state.layout_instance_digest == pair.layout.layout_instance_digest
    forward_calls = pair.candidate.video_backbone._dit.forward_long_call_count

    bad_mask = episode.chunks[0].frame_valid_mask.detach().clone()
    bad_mask[:, -1] = False
    bad_chunk = replace(episode.chunks[0], frame_valid_mask=bad_mask)
    with pytest.raises(ValueError, match="frame_valid_mask"):
        ProductionPathMiniEpisode(
            layout=episode.layout,
            chunks=(bad_chunk,) + episode.chunks[1:],
            recipe_digest=episode.recipe_digest,
        )
    assert pair.candidate.video_backbone._dit.forward_long_call_count == forward_calls

    oversized_context = episode.chunks[0].context.repeat(2, 1, 1)
    with pytest.raises(ValueError, match="context must have exact"):
        ProductionPathMiniEpisode(
            layout=episode.layout,
            chunks=(
                replace(episode.chunks[0], context=oversized_context),
            )
            + episode.chunks[1:],
            recipe_digest=episode.recipe_digest,
        )
    wrong_timestep = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="fixed Phase-C recipe"):
        ProductionPathMiniEpisode(
            layout=episode.layout,
            chunks=(
                replace(episode.chunks[0], video_timestep=wrong_timestep),
            )
            + episode.chunks[1:],
            recipe_digest=episode.recipe_digest,
        )
    assert pair.candidate.video_backbone._dit.forward_long_call_count == forward_calls

    state = dispatcher.state_owner.state
    conditioning = dispatcher._conditioning(
        chunk_id=0, source_state_manifest_digest=state.state_manifest_digest
    )
    request = dispatcher._teacher_request(
        chunk_id=0, conditioning_digest=conditioning.conditioning_digest
    )
    callback_calls = 0

    def forbidden_callback(_context):
        nonlocal callback_calls
        callback_calls += 1
        raise AssertionError("operator callback must not run")

    wrong_time = replace(
        request.content_time,
        action_token_end_exclusive=request.content_time.action_token_end_exclusive + 1,
        action_rope_end_exclusive=request.content_time.action_rope_end_exclusive + 1,
    )
    wrong_request = replace(request, content_time=wrong_time)
    with pytest.raises(hc.CacheContractError) as wrong_interval:
        dispatcher.state_owner.commit_paired(
            request=wrong_request, stage_callback=forbidden_callback
        )
    assert wrong_interval.value.code == "CACHE_CONTENT_TIME_MISMATCH"
    assert callback_calls == 0

    wrong_digest_time = replace(
        request.content_time,
        layout_instance_digest="d" * 64,
    )
    with pytest.raises(hc.CacheContractError) as wrong_digest:
        dispatcher.state_owner.commit_paired(
            request=replace(request, content_time=wrong_digest_time),
            stage_callback=forbidden_callback,
        )
    assert wrong_digest.value.code == "CACHE_CONTENT_TIME_MISMATCH"
    assert callback_calls == 0

    for forged_time in (
        replace(request.content_time, episode_id="forged-phase-c-episode"),
        replace(request.content_time, episode_epoch=1),
    ):
        with pytest.raises(hc.CacheContractError) as forged:
            dispatcher.state_owner.commit_paired(
                request=replace(request, content_time=forged_time),
                stage_callback=forbidden_callback,
            )
        assert forged.value.code == "CACHE_CONTENT_TIME_MISMATCH"
        assert callback_calls == 0

    wrong_action_mask = request.action_valid_mask.detach().clone()
    wrong_action_mask[:, 0] = False
    with pytest.raises(hc.CacheContractError) as wrong_mask:
        dispatcher.state_owner.commit_paired(
            request=replace(request, action_valid_mask=wrong_action_mask),
            stage_callback=forbidden_callback,
        )
    assert wrong_mask.value.code == "COMMIT_TEACHER_ROW_IDENTITY_MISMATCH"
    assert callback_calls == 0
    assert dispatcher.state_owner.state.state_manifest_digest == state.state_manifest_digest

    float64_video = request.video.to(dtype=torch.float64)
    float64_actions = request.actions.to(dtype=torch.float64)
    float64_proof = replace(
        request.proof,
        video_tensor_digest=hc.tensor_digest(float64_video),
        action_tensor_digest=hc.tensor_digest(float64_actions),
    )
    float64_request = replace(
        request,
        video=float64_video,
        actions=float64_actions,
        proof=float64_proof,
    )
    with pytest.raises(ProductionPathMiniContractError, match="CPU/FP32"):
        dispatcher.state_owner.commit_paired(
            request=float64_request,
            stage_callback=forbidden_callback,
        )
    assert callback_calls == 0

    malformed_callback_calls = 0

    def malformed_payload_callback(_context, *, wrong_shape: bool):
        nonlocal malformed_callback_calls
        malformed_callback_calls += 1
        shapes = {
            "ffn_tconv_left_context": (1, 4, 1, 1),
            "main_s_kv": (1, 1, 4, 4),
            "main_s_z": (1, 1, 4, 1),
            "main_shortconv_left_context": (1, 3, 4),
        }
        if wrong_shape:
            shapes["main_shortconv_left_context"] = (1, 2, 4)
        return tuple(
            hc.StagedLayerPayload(
                layer_index=index,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                tensors={
                    name: torch.zeros(
                        shape,
                        dtype=(torch.float32 if wrong_shape else torch.float64),
                        device="cpu",
                    )
                    for name, shape in shapes.items()
                },
            )
            for index in range(20)
        )

    for wrong_shape in (False, True):
        with pytest.raises(hc.CacheContractError) as malformed:
            dispatcher.state_owner.commit_paired(
                request=request,
                stage_callback=lambda context, wrong_shape=wrong_shape: (
                    malformed_payload_callback(
                        context, wrong_shape=wrong_shape
                    )
                ),
            )
        assert malformed.value.code == "COMMIT_STAGING_FAILED"
        assert dispatcher.state_owner.state.state_manifest_digest == (
            state.state_manifest_digest
        )
    assert malformed_callback_calls == 2

    import sana_wam.model.cach_numerical_core as numerical_core

    start_end_calls: list[tuple[int, int]] = []
    rope_calls: list[tuple[int, ...]] = []
    reducer_calls: list[tuple[int, int, int]] = []
    action_input_digests: list[str] = []
    original_run_chunk = pair.candidate.video_backbone.run_chunk
    original_rope = pair.candidate.action_backbone._get_rope_freqs_at
    original_action_forward = (
        pair.candidate.action_backbone.forward_with_bridge_tuple
    )
    original_reducer = numerical_core.reduce_end_of_bin_action_condition

    def captured_run_chunk(*args, **kwargs):
        start_end_calls.append((kwargs["start_f"], kwargs["end_f"]))
        return original_run_chunk(*args, **kwargs)

    def captured_rope(positions):
        rope_calls.append(tuple(int(value) for value in positions.tolist()))
        return original_rope(positions)

    def captured_action_forward(actions, *args, **kwargs):
        action_input_digests.append(hc.tensor_digest(actions))
        return original_action_forward(actions, *args, **kwargs)

    def captured_reducer(noisy_actions, *, committed_actions, chunk, no_action_slot):
        reducer_calls.append(
            (
                chunk.chunk_id,
                0 if committed_actions is None else committed_actions.shape[1],
                noisy_actions.shape[1],
            )
        )
        reduced = original_reducer(
            noisy_actions,
            committed_actions=committed_actions,
            chunk=chunk,
            no_action_slot=no_action_slot,
        )
        for local_latent, span in enumerate(chunk.latent_action_spans):
            expected = (
                no_action_slot.view(1, -1)
                if span.anchor_no_action_slot
                else noisy_actions[
                    :,
                    span.action_end - chunk.action_start - 1,
                    :,
                ]
            )
            _assert_raw_equal(reduced.condition[:, local_latent], expected)
        return reduced

    monkeypatch.setattr(
        pair.candidate.video_backbone, "run_chunk", captured_run_chunk
    )
    monkeypatch.setattr(
        pair.candidate.action_backbone, "_get_rope_freqs_at", captured_rope
    )
    monkeypatch.setattr(
        pair.candidate.action_backbone,
        "forward_with_bridge_tuple",
        captured_action_forward,
    )
    monkeypatch.setattr(
        numerical_core,
        "reduce_end_of_bin_action_condition",
        captured_reducer,
    )
    traced = ProductionPathMiniDispatcher(
        arm=pair.candidate,
        episode=episode,
        episode_id="phase-c-runtime-layout-trace",
    )
    for chunk_id in range(len(pair.layout.chunks)):
        traced.run_readonly_chunk(chunk_id=chunk_id)
        traced.commit_paired(chunk_id=chunk_id)
    assert start_end_calls == [
        (0, 3),
        (0, 3),
        (3, 6),
        (3, 6),
        (6, 8),
        (6, 8),
    ]
    expected_rope = []
    for chunk in pair.layout.chunks:
        positions = tuple(range(chunk.action_rope_start, chunk.action_rope_end))
        expected_rope.extend((positions, positions))
    assert rope_calls == expected_rope
    expected_action_input_digests = []
    for batch, chunk in zip(episode.chunks, pair.layout.chunks):
        expected_action_input_digests.extend(
            (
                hc.tensor_digest(
                    batch.noisy_actions[:, : chunk.valid_action_count]
                ),
                hc.tensor_digest(
                    batch.clean_action_target[:, : chunk.valid_action_count]
                ),
            )
        )
    assert action_input_digests == expected_action_input_digests
    assert reducer_calls == [
        (0, 0, 16),
        (0, 0, 16),
        (1, 16, 24),
        (1, 16, 24),
        (2, 40, 24),
        (2, 40, 24),
    ]


def test_phase_c_c4_c6_bootstrap_continuation_tail_exact_action_coverage(pair, episode) -> None:
    chunks = pair.layout.chunks
    assert [chunk.valid_latent_count for chunk in chunks] == [3, 3, 2]
    assert [chunk.valid_action_count for chunk in chunks] == [16, 24, 16]
    assert [(chunk.action_start, chunk.action_end) for chunk in chunks] == [
        (0, 16),
        (16, 40),
        (40, 56),
    ]
    assert chunks[0].is_bootstrap is True
    assert chunks[0].anchor_no_action_slot is True
    assert chunks[1].is_bootstrap is False
    assert chunks[2].is_partial_tail is True
    anchors = [
        span.latent_index
        for chunk in chunks
        for span in chunk.latent_action_spans
        if span.anchor_no_action_slot
    ]
    assert anchors == [0]

    owned = [
        action_index
        for chunk in chunks
        for span in chunk.latent_action_spans
        for action_index in range(span.action_start, span.action_end)
    ]
    assert owned == list(range(56))
    assert len(owned) == len(set(owned))
    assert tuple(
        position for chunk in chunks for position in chunk.action_rope_positions
    ) == tuple(range(56))
    for batch, chunk in zip(episode.chunks, chunks):
        assert int(batch.action_valid_mask.sum()) == chunk.valid_action_count
        assert not bool(
            batch.noisy_actions.masked_select(
                ~batch.action_valid_mask.unsqueeze(-1)
            ).count_nonzero()
        )

    exact_k = build_production_path_mini_layout_for_tests(
        ProductionPathMiniSpec(), valid_raw_count=17
    )
    bootstrap_tail = build_production_path_mini_layout_for_tests(
        ProductionPathMiniSpec(), valid_raw_count=9
    )
    assert len(exact_k.chunks) == 1
    assert exact_k.chunks[0].valid_latent_count == 3
    assert exact_k.chunks[0].is_partial_tail is False
    assert len(bootstrap_tail.chunks) == 1
    assert bootstrap_tail.chunks[0].valid_latent_count == 2
    assert bootstrap_tail.chunks[0].is_partial_tail is True

    captured: list[Tensor] = []
    embedder = pair.candidate.video_backbone._dit.delta_pose_embedder
    assert embedder is not None
    hook = embedder.register_forward_pre_hook(
        lambda _module, values: captured.append(values[0].detach().clone())
    )
    try:
        dispatcher = ProductionPathMiniDispatcher(
            arm=pair.candidate,
            episode=episode,
            episode_id="phase-c-bootstrap-run",
        )
        dispatcher.run_readonly_chunk(chunk_id=0)
    finally:
        hook.remove()
    assert captured
    _assert_raw_equal(
        captured[0][:, 0],
        pair.candidate.cach_core.no_action_slot.detach().view(1, -1),
    )
    assert pair.reference.video_backbone._dit.delta_pose_embedder is None


def test_phase_c_c7_failure_duplicate_reset_and_stale_view(pair, episode) -> None:
    injection_points = (
        ("inject_failure_before_vendor_for_tests", 0),
        ("inject_failure_after_vendor_for_tests", 1),
        ("inject_failure_after_materialize_for_tests", 1),
    )
    for injection_name, expected_forward_delta in injection_points:
        injected_dispatcher = ProductionPathMiniDispatcher(
            arm=pair.candidate,
            episode=episode,
            episode_id=f"phase-c-atomicity-{injection_name}",
        )
        injected_before = injected_dispatcher.state_owner.state
        injected_pointer = (
            injected_dispatcher.state_owner.live_pointer_identity_for_tests
        )
        injected_swaps = injected_dispatcher.state_owner.pointer_swap_count
        forward_before = pair.candidate.video_backbone._dit.forward_long_call_count
        with pytest.raises(hc.CacheContractError) as injected:
            injected_dispatcher.commit_paired(
                chunk_id=0, **{injection_name: True}
            )
        assert injected.value.code == "COMMIT_STAGING_FAILED"
        assert pair.candidate.video_backbone._dit.forward_long_call_count == (
            forward_before + expected_forward_delta
        )
        assert (
            injected_dispatcher.state_owner.live_pointer_identity_for_tests
            == injected_pointer
        )
        assert (
            injected_dispatcher.state_owner.pointer_swap_count == injected_swaps
        )
        assert (
            injected_dispatcher.state_owner.state.state_manifest_digest
            == injected_before.state_manifest_digest
        )
        recovery = injected_dispatcher.commit_paired(chunk_id=0)
        assert recovery.revision_before == 0
        assert recovery.revision_after == 1
        assert injected_dispatcher.state_owner.state.next_chunk_id == 1

    pending_dispatcher = ProductionPathMiniDispatcher(
        arm=pair.candidate,
        episode=episode,
        episode_id="phase-c-pending-reservation",
    )
    pending_state = pending_dispatcher.state_owner.state
    pending_conditioning = pending_dispatcher._conditioning(
        chunk_id=0,
        source_state_manifest_digest=pending_state.state_manifest_digest,
    )
    pending_request = pending_dispatcher._teacher_request(
        chunk_id=0,
        conditioning_digest=pending_conditioning.conditioning_digest,
    )
    entered = threading.Event()
    release = threading.Event()
    worker_codes: list[str] = []

    def blocking_callback(_context):
        entered.set()
        if not release.wait(timeout=5.0):
            raise RuntimeError("Phase-C pending test timed out")
        raise RuntimeError("registered Phase-C pending-owner failure")

    def run_pending_worker() -> None:
        try:
            pending_dispatcher.state_owner.commit_paired(
                request=pending_request,
                stage_callback=blocking_callback,
            )
        except hc.CacheContractError as exc:
            worker_codes.append(exc.code)

    worker = threading.Thread(target=run_pending_worker, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=5.0)
        second_callback_calls = 0

        def second_callback(_context):
            nonlocal second_callback_calls
            second_callback_calls += 1
            raise AssertionError("second operator callback must not run")

        with pytest.raises(hc.CacheContractError) as pending:
            pending_dispatcher.state_owner.commit_paired(
                request=pending_request,
                stage_callback=second_callback,
            )
        assert pending.value.code == "COMMIT_DUPLICATE_PENDING"
        assert second_callback_calls == 0
    finally:
        release.set()
        worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert worker_codes == ["COMMIT_STAGING_FAILED"]
    assert (
        pending_dispatcher.state_owner.state.state_manifest_digest
        == pending_state.state_manifest_digest
    )

    dispatcher = ProductionPathMiniDispatcher(
        arm=pair.candidate,
        episode=episode,
        episode_id="phase-c-atomicity",
    )
    before = dispatcher.state_owner.state

    view = dispatcher.state_owner.snapshot_for_denoise(
        expected_episode_id=before.episode_id,
        expected_episode_epoch=before.episode_epoch,
        expected_revision=0,
        expected_next_chunk_id=0,
    )
    scratch = view.export_scratch()
    dispatcher.commit_paired(chunk_id=0)
    with pytest.raises(hc.CacheContractError) as stale:
        dispatcher.state_owner.finish_denoise(read_view=view, scratch=scratch)
    assert stale.value.code == "COMMIT_STALE_REVISION"

    state_after_one = dispatcher.state_owner.state
    with pytest.raises(hc.CacheContractError) as duplicate:
        dispatcher.commit_paired(chunk_id=0)
    assert duplicate.value.code == "COMMIT_DUPLICATE_CONFLICT"
    assert dispatcher.state_owner.state.state_manifest_digest == state_after_one.state_manifest_digest

    reset = dispatcher.reset_episode_for_tests(
        reset_id="phase-c-reset",
        new_episode_id="phase-c-after-reset",
    )
    assert reset.ephemeral is True
    assert reset.durable_publication is False
    state = dispatcher.state_owner.state
    assert state.revision == 0
    assert state.action_cursor == 0
    assert state.committed_action_history is None
    assert all(layer.is_empty for layer in state.layer_states)
    readonly_after_reset = dispatcher.run_readonly_chunk(chunk_id=0)
    assert tuple(readonly_after_reset.video_prediction.shape) == (1, 3, 3, 1, 1)
    receipt_after_reset = dispatcher.commit_paired(chunk_id=0)
    assert receipt_after_reset.revision_before == 0
    assert receipt_after_reset.revision_after == 1
    recommitted = dispatcher.state_owner.state
    assert recommitted.episode_id == "phase-c-after-reset"
    assert recommitted.episode_epoch == 1
    assert recommitted.next_chunk_id == 1


def test_phase_c_failure_receipt_is_exclusive_immutable_tmp_only(
    pair, episode, tmp_path, request
) -> None:
    dispatcher = ProductionPathMiniDispatcher(
        arm=pair.reference,
        episode=episode,
        episode_id="phase-c-failure-receipt",
    )
    before = dispatcher.state_owner.state
    conditioning = dispatcher._conditioning(
        chunk_id=0, source_state_manifest_digest=before.state_manifest_digest
    )
    teacher_request = dispatcher._teacher_request(
        chunk_id=0, conditioning_digest=conditioning.conditioning_digest
    )
    with pytest.raises(hc.CacheContractError) as caught:
        dispatcher.commit_paired(
            chunk_id=0, inject_failure_before_vendor_for_tests=True
        )
    after = dispatcher.state_owner.state
    assert caught.value.code == "COMMIT_STAGING_FAILED"
    receipt = Stage2FailureInjectionReceipt(
        receipt_id="phase-c-injected-failure",
        commit_id=teacher_request.commit_id,
        transaction_nonce=teacher_request.transaction_nonce,
        episode_id=before.episode_id,
        episode_epoch=before.episode_epoch,
        chunk_id=0,
        layout_instance_digest=pair.layout.layout_instance_digest,
        source_proof_digest=teacher_request.proof.source_proof_digest,
        failure_code=caught.value.code,
        injection_point="stage_callback_before_vendor_forward",
        state_manifest_before=before.state_manifest_digest,
        state_manifest_after=after.state_manifest_digest,
        revision_before=before.revision,
        revision_after=after.revision,
        action_cursor_before=before.action_cursor,
        action_cursor_after=after.action_cursor,
        recorded_at_monotonic_ns=time.monotonic_ns(),
    )
    root = (tmp_path / "phase-c-nonformal-failure").resolve()
    assert str(root).startswith("/tmp/")

    def remove_ephemeral_failure_root() -> None:
        if not root.exists():
            return
        os.chmod(root, 0o700)
        for child in root.iterdir():
            os.chmod(child, 0o600)
            child.unlink()
        root.rmdir()

    request.addfinalizer(remove_ephemeral_failure_root)
    frozen = freeze_stage2_failure_evidence(root, receipt)
    assert frozen.receipt_sha256 == hashlib.sha256(receipt.canonical_bytes).hexdigest()
    assert stat.S_IMODE(os.stat(frozen.root).st_mode) == 0o500
    assert stat.S_IMODE(os.stat(frozen.receipt_path).st_mode) == 0o400
    assert frozen.receipt_path.read_bytes() == receipt.canonical_bytes
    with pytest.raises(FileExistsError):
        freeze_stage2_failure_evidence(frozen.root, receipt)


def test_phase_c_cpu_only_and_no_prohibited_runtime_side_effects(pair) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    assert not hasattr(pair.reference, "optimizer")
    assert not hasattr(pair.candidate, "optimizer")
    assert not hasattr(pair.reference, "trainer")
    assert not hasattr(pair.candidate, "trainer")
    assert not hasattr(pair.reference, "checkpoint")
    assert not hasattr(pair.candidate, "checkpoint")
    assert not hasattr(pair.reference, "dataset")
    assert not hasattr(pair.candidate, "dataset")
