"""Stage-2 CUDA integration of paired t=0 commits with a real mini GDN."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

from sana_wam.cach.action_conditioning import reduce_end_of_bin_action_condition
from sana_wam.cach.prefix_compaction import FixedKPrefixPlan
from sana_wam.model.action_chunk_layout import (
    build_synthetic_chunk_action_layout_for_tests,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)
from sana_wam.model.video_backbone.sana import hybrid_cache as hc
from sana_wam.model.video_backbone.sana.hybrid_cache_codec import (
    scratch_to_vendor_cache,
    vendor_cache_to_staged_payloads,
)


os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cached mini GDN uses CUDA/Triton",
)

DATASET_SHA = "e" * 64
ROW_ORDER_SHA = "f" * 64


class _DurablePublisher:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True)
        self.calls: list[str] = []

    def publish_exclusive(self, *, receipt_id, payload, expected_sha256):
        path = self.root / (hashlib.sha256(receipt_id.encode()).hexdigest() + ".json")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise OSError("short receipt write")
                view = view[count:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        readback = path.read_bytes()
        directory = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        observed = hashlib.sha256(readback).hexdigest()
        assert readback == payload
        assert observed == expected_sha256
        self.calls.append(receipt_id)
        return hc.ReceiptPublication(
            receipt_id=receipt_id,
            sha256=observed,
            durable=True,
            readback_verified=True,
        )


def _layout(*, episode: str):
    # L8 gives all three registered ownership cases in one episode:
    # bootstrap (16 actions), full continuation (24), partial tail (16).
    valid_latent_count = 8
    valid_raw_count = (valid_latent_count - 1) * 8 + 1
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=valid_raw_count,
        video_stride=1,
        source_row_label=f"{episode}-row",
        episode_label=episode,
    )
    return build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=valid_raw_count,
        video_valid_mask=(True,) * valid_raw_count,
        action_valid_mask=(True,) * (valid_raw_count - 1),
    )


def _registry() -> hc.LayerRegistry:
    return hc.LayerRegistry(
        layers=(
            hc.LayerSpec(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="CachedChunkCausalGDN",
                main_shortconv_enabled=True,
                ffn_tconv_enabled=True,
            ),
        )
    )


def _build_backbone():
    from sana_wam.model.video_backbone.sana.adapter import SanaVideoBackbone

    backbone = SanaVideoBackbone.from_mini_config(
        depth=1,
        hidden_size=224,
        num_heads=2,
        linear_head_dim=112,
        f=3,
        h=4,
        w=4,
        device="cuda",
        dtype=torch.bfloat16,
        attn_kernel="gdn",
        chunk_size=3,
        use_delta_pose_additive=True,
        delta_pose_additive_dim=20,
    )
    backbone._dit.eval()
    return backbone


def _requests(
    layout,
    *,
    future_video_delta: float = 0.0,
    future_action_delta: float = 0.0,
):
    generator = torch.Generator(device="cuda").manual_seed(20260731)
    requests = []
    for chunk in layout.chunks:
        fixed_k = len(chunk.latent_valid_mask)
        video = torch.zeros(1, 16, fixed_k, 4, 4, device="cuda", dtype=torch.bfloat16)
        video[:, :, : chunk.valid_latent_count] = torch.randn(
            1,
            16,
            chunk.valid_latent_count,
            4,
            4,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if chunk.chunk_id == 1 and future_video_delta:
            video[:, :, : chunk.valid_latent_count].add_(future_video_delta)
        frame_mask = torch.tensor(
            chunk.latent_valid_mask, device="cuda", dtype=torch.bool
        ).view(1, -1)
        actions = torch.zeros(
            1,
            chunk.action_slot_capacity,
            20,
            device="cuda",
            dtype=torch.bfloat16,
        )
        actions[:, : chunk.valid_action_count] = torch.randn(
            1,
            chunk.valid_action_count,
            20,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if chunk.chunk_id == 1 and future_action_delta:
            actions[:, : chunk.valid_action_count].add_(future_action_delta)
        action_mask = torch.tensor(
            chunk.action_valid_mask, device="cuda", dtype=torch.bool
        ).view(1, -1)
        content_time = hc.ContentTime.from_layout(
            episode_id="episode-0",
            episode_epoch=0,
            layout=layout,
            chunk=chunk,
        )
        proof = hc.TeacherForcingDatasetPairProof(
            dataset_manifest_sha256=DATASET_SHA,
            dataset_episode_id="episode-0",
            dataset_row_identity=f"episode-0:chunk-{chunk.chunk_id}",
            dataset_row_start=0,
            dataset_row_end_exclusive=layout.valid_raw_count,
            layout_spec_sha256=layout.layout_spec_sha256,
            layout_instance_digest=layout.layout_instance_digest,
            video_tensor_digest=hc.tensor_digest(video),
            frame_valid_mask_digest=hc.tensor_digest(frame_mask),
            action_tensor_digest=hc.tensor_digest(actions),
            action_mask_digest=hc.tensor_digest(action_mask),
            row_order_manifest_sha256=ROW_ORDER_SHA,
        )
        requests.append(
            hc.TeacherForcingCommitRequest(
                commit_id=f"commit-{chunk.chunk_id}",
                transaction_nonce=f"nonce-{chunk.chunk_id}",
                expected_episode_id="episode-0",
                expected_episode_epoch=0,
                expected_revision=chunk.chunk_id,
                content_time=content_time,
                video=video,
                frame_valid_mask=frame_mask,
                actions=actions,
                action_valid_mask=action_mask,
                proof=proof,
            )
        )
    return tuple(requests)


class _MiniPairedStager:
    def __init__(self, *, backbone, layout, requests):
        self.backbone = backbone
        self.layout = layout
        self.requests = requests
        self.calls: list[int] = []
        self.outputs: dict[int, torch.Tensor] = {}
        self.action_outputs: dict[int, torch.Tensor] = {}
        self.action_rope_ranges: dict[int, tuple[int, int]] = {}
        self.context = torch.randn(
            1,
            8,
            64,
            generator=torch.Generator(device="cuda").manual_seed(7),
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.seq_lens = torch.tensor([8], device="cuda", dtype=torch.long)
        from sana_wam.model.action_backbone.joint_action_dit import ActionDiT

        torch.manual_seed(424242)
        self.action_backbone = ActionDiT(
            action_dim=20,
            dim=64,
            ffn_dim=128,
            num_heads=2,
            num_layers=1,
            video_dim=224,
            bridge_layers=(0,),
            variant="joint_cross_attn",
            attn_head_dim=32,
            text_dim=64,
            freq_dim=64,
            max_action_len=128,
        ).to(device="cuda", dtype=torch.bfloat16)
        self.action_backbone.eval()

    def __call__(self, context: hc.PairedStagingContext):
        assert context.video_timestep == context.action_timestep == 0
        chunk_id = context.content_time.chunk_id
        self.calls.append(chunk_id)
        chunk = self.layout.chunks[chunk_id]
        if chunk_id == 0:
            committed = None
        else:
            earlier = self.requests[:chunk_id]
            committed = torch.cat(
                [
                    request.actions[:, : prior.valid_action_count]
                    for request, prior in zip(earlier, self.layout.chunks[:chunk_id])
                ],
                dim=1,
            )
            assert committed.shape[1] == chunk.action_start
        condition = reduce_end_of_bin_action_condition(
            context.actions,
            committed_actions=committed,
            chunk=chunk,
            no_action_slot=torch.zeros(20, device="cuda", dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            condition.latent_valid_mask,
            context.frame_valid_mask,
            atol=0,
            rtol=0,
        )
        for local_latent, span in enumerate(chunk.latent_action_spans):
            if span.anchor_no_action_slot:
                assert local_latent == 0 and chunk_id == 0
                assert not bool(condition.condition[:, local_latent].count_nonzero())
            else:
                local_action = span.action_end - chunk.action_start - 1
                torch.testing.assert_close(
                    condition.condition[:, local_latent],
                    context.actions[:, local_action],
                    atol=0,
                    rtol=0,
                )
        vendor_cache = scratch_to_vendor_cache(
            context.previous_scratch,
            _registry(),
        )
        with torch.no_grad():
            output, bridges, vendor_cache = self.backbone.run_chunk(
                context.video,
                torch.tensor([0.0], device="cuda"),
                context=self.context,
                seq_lens=self.seq_lens,
                kv_cache=vendor_cache,
                start_f=context.content_time.latent_start,
                end_f=context.content_time.latent_end_exclusive,
                save_kv_cache=True,
                bridge_layers=(0,),
                action_condition=condition.condition,
                frame_valid_mask=condition.latent_valid_mask,
            )
            assert bool(torch.isfinite(output).all())
            assert bridges and all(
                bool(torch.isfinite(value).all()) for value in bridges.values()
            )
            action_plan = FixedKPrefixPlan.from_mask(context.action_valid_mask)
            compact_actions = action_plan.compact_condition(context.actions)
            rope_start = context.content_time.action_rope_start
            rope_end = context.content_time.action_rope_end_exclusive
            assert rope_end - rope_start == action_plan.valid_slots
            rope_positions = torch.arange(rope_start, rope_end, dtype=torch.long)
            action_freqs = self.action_backbone._get_rope_freqs_at(rope_positions)
            video_plan = FixedKPrefixPlan.from_mask(context.frame_valid_mask)
            compact_bridge = video_plan.compact_token_sequence(bridges[0])
            compact_action_output = self.action_backbone.forward_with_bridge_tuple(
                compact_actions,
                (compact_bridge,),
                torch.tensor([0.0], device="cuda", dtype=torch.bfloat16),
                context=self.context,
                context_mask=torch.ones(1, 8, device="cuda", dtype=torch.bool),
                action_freqs=action_freqs,
            )
            action_output = action_plan.restore_token_sequence(compact_action_output)
        self.outputs[chunk_id] = output.detach().clone()
        self.action_outputs[chunk_id] = action_output.detach().clone()
        self.action_rope_ranges[chunk_id] = (rope_start, rope_end)
        assert tuple(action_output.shape) == tuple(context.actions.shape)
        assert bool(torch.isfinite(action_output).all())
        assert bool(action_output[:, : action_plan.valid_slots].count_nonzero())
        if action_plan.valid_slots < action_plan.fixed_slots:
            assert not bool(action_output[:, action_plan.valid_slots :].count_nonzero())
        return vendor_cache_to_staged_payloads(vendor_cache, _registry())


def _run_chunks(*, backbone, layout, requests, receipt_root):
    publisher = _DurablePublisher(receipt_root)
    manager = hc.HybridCacheManager.for_synthetic_tests(
        registry=_registry(),
        episode_id="episode-0",
        layout=layout,
        receipt_publisher=publisher,
    )
    stager = _MiniPairedStager(backbone=backbone, layout=layout, requests=requests)
    states = []
    for request in requests:
        receipt = manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=stager,
        )
        assert receipt.revision_after == request.expected_revision + 1
        states.append(manager.state)
    return manager, publisher, stager, tuple(states)


def _state_tensors(state):
    return {
        name: snapshot.clone_tensor()
        for name, snapshot in state.layer_states[0].tensors
    }


def _assert_tensor_maps_equal(left, right) -> None:
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)


@requires_cuda
def test_three_chunk_paired_t0_commit_cache_progression_and_reset(tmp_path) -> None:
    torch.manual_seed(20260731)
    layout = _layout(episode="episode-0")
    requests = _requests(layout)
    manager, publisher, stager, states = _run_chunks(
        backbone=_build_backbone(),
        layout=layout,
        requests=requests,
        receipt_root=tmp_path / "receipts",
    )

    assert [chunk.valid_latent_count for chunk in layout.chunks] == [3, 3, 2]
    assert [chunk.action_slot_capacity for chunk in layout.chunks] == [16, 24, 24]
    assert [chunk.valid_action_count for chunk in layout.chunks] == [16, 24, 16]
    assert stager.calls == [0, 1, 2]
    assert stager.action_rope_ranges == {0: (0, 16), 1: (16, 40), 2: (40, 56)}
    assert publisher.calls == ["commit-0", "commit-1", "commit-2"]
    assert states[0].revision == 1 and states[0].next_chunk_id == 1
    assert states[1].revision == 2 and states[1].next_chunk_id == 2
    assert states[2].revision == 3 and states[2].next_chunk_id == 3
    assert [state.action_cursor for state in states] == [16, 40, 56]
    assert states[0].state_manifest_digest != states[1].state_manifest_digest
    assert states[1].state_manifest_digest != states[2].state_manifest_digest
    assert states[0].layer_states[0].through.chunk_id == 0
    assert states[1].layer_states[0].through.chunk_id == 1
    assert states[2].layer_states[0].through.chunk_id == 2
    assert not bool(stager.outputs[2][:, :, 2:].count_nonzero())
    assert not bool(stager.action_outputs[2][:, 16:].count_nonzero())

    retry = manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=requests[0],
        stage_callback=stager,
    )
    assert retry.commit_id == "commit-0"
    assert stager.calls == [0, 1, 2]
    assert publisher.calls == ["commit-0", "commit-1", "commit-2"]
    assert manager.state.revision == 3

    reset_layout = _layout(episode="episode-1")
    manager.reset(
        reset_id="reset-after-three-chunks",
        transaction_nonce="reset-nonce",
        new_episode_id="episode-1",
        layout=reset_layout,
    )
    view = manager.snapshot_for_denoise(
        expected_episode_id="episode-1",
        expected_episode_epoch=1,
        expected_revision=0,
        expected_next_chunk_id=0,
    )
    assert scratch_to_vendor_cache(view.export_scratch(), _registry()) == [[None] * 10]
    assert publisher.calls[-1] == "reset-after-three-chunks"


@requires_cuda
def test_future_chunk_perturbation_cannot_change_prior_committed_state(tmp_path) -> None:
    torch.manual_seed(20260731)
    layout = _layout(episode="episode-0")
    backbone = _build_backbone()
    parameter_snapshot = {
        name: value.detach().clone()
        for name, value in backbone._dit.state_dict().items()
    }
    baseline = _requests(layout, future_video_delta=0.0)
    perturbed = _requests(layout, future_video_delta=4.0)

    _, _, baseline_stager, baseline_states = _run_chunks(
        backbone=backbone,
        layout=layout,
        requests=baseline,
        receipt_root=tmp_path / "baseline",
    )
    _, _, perturbed_stager, perturbed_states = _run_chunks(
        backbone=backbone,
        layout=layout,
        requests=perturbed,
        receipt_root=tmp_path / "perturbed",
    )

    assert baseline_states[0].state_manifest_digest == perturbed_states[0].state_manifest_digest
    _assert_tensor_maps_equal(
        _state_tensors(baseline_states[0]),
        _state_tensors(perturbed_states[0]),
    )
    torch.testing.assert_close(
        baseline_stager.outputs[0],
        perturbed_stager.outputs[0],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        baseline_stager.action_outputs[0],
        perturbed_stager.action_outputs[0],
        atol=0,
        rtol=0,
    )
    baseline_future_cache = _state_tensors(baseline_states[1])
    perturbed_future_cache = _state_tensors(perturbed_states[1])
    assert any(
        not torch.equal(baseline_future_cache[name], perturbed_future_cache[name])
        for name in ("main_s_kv", "main_s_z")
    )
    assert not torch.equal(
        baseline_stager.action_outputs[1],
        perturbed_stager.action_outputs[1],
    )
    _assert_tensor_maps_equal(
        parameter_snapshot,
        {name: value.detach() for name, value in backbone._dit.state_dict().items()},
    )


@requires_cuda
def test_future_action_perturbation_respects_zero_init_and_prior_causality(
    tmp_path,
) -> None:
    torch.manual_seed(20260731)
    layout = _layout(episode="episode-0")
    backbone = _build_backbone()
    parameter_snapshot = {
        name: value.detach().clone()
        for name, value in backbone._dit.state_dict().items()
    }
    baseline = _requests(layout, future_action_delta=0.0)
    perturbed = _requests(layout, future_action_delta=4.0)

    _, _, baseline_stager, baseline_states = _run_chunks(
        backbone=backbone,
        layout=layout,
        requests=baseline,
        receipt_root=tmp_path / "baseline-action",
    )
    _, _, perturbed_stager, perturbed_states = _run_chunks(
        backbone=backbone,
        layout=layout,
        requests=perturbed,
        receipt_root=tmp_path / "perturbed-action",
    )

    _assert_tensor_maps_equal(
        _state_tensors(baseline_states[0]),
        _state_tensors(perturbed_states[0]),
    )
    torch.testing.assert_close(
        baseline_stager.outputs[0],
        perturbed_stager.outputs[0],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        baseline_stager.action_outputs[0],
        perturbed_stager.action_outputs[0],
        atol=0,
        rtol=0,
    )

    # The video seam is an exact output bypass at theta0, so an action-only
    # perturbation does not yet alter video output/cache. The separate
    # parameter-gradient test proves that the zero projection remains wired.
    _assert_tensor_maps_equal(
        _state_tensors(baseline_states[1]),
        _state_tensors(perturbed_states[1]),
    )
    torch.testing.assert_close(
        baseline_stager.outputs[1],
        perturbed_stager.outputs[1],
        atol=0,
        rtol=0,
    )
    assert not torch.equal(
        baseline_stager.action_outputs[1],
        perturbed_stager.action_outputs[1],
    )
    _assert_tensor_maps_equal(
        parameter_snapshot,
        {name: value.detach() for name, value in backbone._dit.state_dict().items()},
    )
