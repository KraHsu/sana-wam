"""CPU/static admission for the Stage-2B production numerical scaffolding."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sana_wam.cach.action_conditioning import reduce_end_of_bin_action_condition
from sana_wam.cach.staging_variant import (
    CACHStagingVariant,
    history_summary_operator_for_variant,
    require_cach_staging_variant,
)
from sana_wam.model.action_chunk_layout import (
    SelectedProprioBinding,
    build_synthetic_chunk_action_layout_for_tests,
    canonical_proprio_row_sha256,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)
from sana_wam.model.cach_paired_stager import (
    CACHChunkConditioning,
    CACHTeacherForcingPairedStager,
    build_layer_registry_from_backbone,
)
from sana_wam.model.cach_numerical_core import CACHNumericalCore
from sana_wam.model.video_backbone.sana import hybrid_cache as hc
from sana_wam.model.video_backbone.sana import hybrid_cache_stage2b as h2


ROOT = Path(__file__).resolve().parents[1]


def _layout(*, valid_raw_count: int = 9):
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=valid_raw_count,
        video_stride=1,
        source_row_label="stage2b-row",
        episode_label="stage2b-episode",
    )
    return build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=valid_raw_count,
        video_valid_mask=(True,) * valid_raw_count,
        action_valid_mask=(True,) * (valid_raw_count - 1),
    )


def _conditioning_for(layout, *, chunk_id: int, source_digest: str):
    chunk = layout.chunks[chunk_id]
    context = torch.randn(1, 4, 20)
    mask = torch.ones(1, 4, dtype=torch.bool)
    seq_lens = torch.tensor([4], dtype=torch.long)
    proprio = torch.randn(1, 20, dtype=torch.float32)
    binding = SelectedProprioBinding(
        layout_instance_digest=layout.layout_instance_digest,
        chunk_id=chunk_id,
        raw_index=chunk.proprio_raw_index,
        timestamp=chunk.proprio_timestamp,
        source_receipt_digest=chunk.proprio_source_receipt_digest,
        proprio_value_sha256=canonical_proprio_row_sha256(proprio[0]),
    )
    return CACHChunkConditioning.create(
        layout_instance_digest=layout.layout_instance_digest,
        chunk_id=chunk_id,
        source_state_manifest_digest=source_digest,
        context=context,
        context_mask=mask,
        seq_lens=seq_lens,
        proprio_state=proprio,
        proprio_bindings=(binding,),
    )


def _conditioning(*, source_digest: str = "a" * 64):
    layout = _layout()
    return layout, _conditioning_for(
        layout,
        chunk_id=0,
        source_digest=source_digest,
    )


def test_numerical_core_owns_only_exact_zero_no_action_parameter() -> None:
    core = CACHNumericalCore(action_dim=20)

    assert tuple(core.state_dict()) == ("no_action_slot",)
    assert tuple(core.named_parameters())[0][0] == "no_action_slot"
    assert not tuple(core.named_children())
    assert core.no_action_slot.requires_grad
    assert core.no_action_slot.shape == (20,)
    assert not bool(core.no_action_slot.detach().count_nonzero())


def test_staging_variants_have_distinct_strict_history_summary_identities() -> None:
    assert history_summary_operator_for_variant(CACHStagingVariant.CACH_A) == (
        "gdn_recurrent_paired_commit_v1"
    )
    assert history_summary_operator_for_variant(
        CACHStagingVariant.REF_GDN_CORRECTED
    ) == "gdn_recurrent_video_only_paired_commit_v1"
    with pytest.raises(TypeError, match="CACHStagingVariant"):
        require_cach_staging_variant("cach_a")


def test_stage2b_is_additive_over_exact_frozen_stage1_sources() -> None:
    pins = {
        "src/sana_wam/model/causal_action_hybrid.py": (
            "e714ea6ade41223ed1b93e643cbd179fa63ff7ab48d568f36dc95e5eaab17560"
        ),
        "src/sana_wam/model/video_backbone/sana/hybrid_cache.py": (
            "5b61ab6492670e8021b3603f965e9f3e8921e9d7f0ea77b49e40d92a77f78a74"
        ),
    }
    for relative, expected in pins.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_chunk_conditioning_digest_binds_every_tensor_and_source_state() -> None:
    layout, conditioning = _conditioning()
    digest = conditioning.conditioning_digest

    conditioning.verify_binding(
        layout=layout,
        chunk_id=0,
        source_state_manifest_digest="a" * 64,
    )
    with pytest.raises(ValueError, match="conditioning binding"):
        conditioning.verify_binding(
            layout=layout,
            chunk_id=0,
            source_state_manifest_digest="b" * 64,
        )

    conditioning.context.add_(1.0)
    assert conditioning.conditioning_digest == digest
    with pytest.raises(RuntimeError, match="mutated"):
        conditioning.verify_binding(
            layout=layout,
            chunk_id=0,
            source_state_manifest_digest="a" * 64,
        )


class _Topology(nn.Module):
    def __init__(self, attention: nn.Module, feed_forward: nn.Module):
        super().__init__()
        block = nn.Module()
        block.add_module("attn", attention)
        block.add_module("mlp", feed_forward)
        dit = nn.Module()
        dit.add_module("blocks", nn.ModuleList([block]))
        self.add_module("_dit", dit)
        self.cached_streaming_enabled = True


def _uninitialized_module(module_type: type[nn.Module]) -> nn.Module:
    value = module_type.__new__(module_type)
    nn.Module.__init__(value)
    return value


def test_registry_uses_exact_executable_operator_classes_for_synthetic_mini() -> None:
    from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
    from diffusion.model.nets.sana_gdn_blocks import CachedChunkCausalGDN

    attention = _uninitialized_module(CachedChunkCausalGDN)
    attention.conv_k = None
    feed_forward = _uninitialized_module(CachedGLUMBConvTemp)
    registry = build_layer_registry_from_backbone(
        _Topology(attention, feed_forward),
        synthetic_test_only=True,
    )

    assert len(registry.layers) == 1
    spec = registry.layers[0]
    assert spec.operator_class == (
        "diffusion.model.nets.sana_gdn_blocks.CachedChunkCausalGDN"
    )
    assert spec.camera_enabled is False
    assert spec.main_shortconv_enabled is False
    assert spec.ffn_tconv_enabled is True


def test_registry_rejects_non_cached_attention_before_cache_slot_inference() -> None:
    from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp

    feed_forward = _uninitialized_module(CachedGLUMBConvTemp)
    with pytest.raises(RuntimeError, match="non-CachedChunkCausalGDN"):
        build_layer_registry_from_backbone(
            _Topology(nn.Identity(), feed_forward),
            synthetic_test_only=True,
        )


def test_production_stager_api_has_no_manager_or_future_request_list() -> None:
    parameters = inspect.signature(CACHTeacherForcingPairedStager).parameters
    assert "manager" not in parameters
    assert "requests" not in parameters
    assert "future_requests" not in parameters
    call_annotation = inspect.signature(
        CACHTeacherForcingPairedStager.__call__
    ).return_annotation
    assert call_annotation != inspect.Signature.empty


class _Publisher:
    def publish_exclusive(self, *, receipt_id, payload, expected_sha256):
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        return hc.ReceiptPublication(
            receipt_id=receipt_id,
            sha256=expected_sha256,
            durable=True,
            readback_verified=True,
        )


class _FakeVideoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_action_condition = None
        self.last_frame_valid_mask = None
        self.incoming_main_s_kv = []
        self.action_conditions = []

    def run_chunk(
        self,
        video,
        _timestep,
        *,
        kv_cache,
        bridge_layers,
        action_condition,
        frame_valid_mask,
        **_kwargs,
    ):
        self.last_action_condition = (
            None if action_condition is None else action_condition.detach().clone()
        )
        self.last_frame_valid_mask = frame_valid_mask.detach().clone()
        self.action_conditions.append(
            None if action_condition is None else action_condition.detach().clone()
        )
        batch, _, fixed_k = video.shape[:3]
        bridge = video.mean(dim=(1, 3, 4)).unsqueeze(-1).expand(batch, fixed_k, 4)
        slots = kv_cache[0]
        self.incoming_main_s_kv.append(
            None if slots[0] is None else slots[0].detach().clone()
        )
        value = float(len(self.incoming_main_s_kv))
        slots[0] = video.new_full((batch, 1, 2, 2), value)
        slots[1] = video.new_full((batch, 1, 2, 1), value)
        slots[6] = 1.0
        slots[9] = video.new_ones((batch, 1, 1, 2))
        return video.detach().clone(), {bridge_layers[0]: bridge}, kv_cache


class _FakeActionBackbone(nn.Module):
    bridge_layers = (0,)

    def __init__(self):
        super().__init__()
        self.last_actions = None
        self.last_rope_positions = None
        self.rope_positions = []

    def _get_rope_freqs_at(self, positions):
        self.last_rope_positions = positions.detach().clone()
        self.rope_positions.append(positions.detach().clone())
        return positions.to(dtype=torch.float32).view(-1, 1)

    def bridge_tuple_from_dict(self, bridges):
        return (bridges[0],)

    def forward_with_bridge_tuple(
        self,
        actions,
        _bridges,
        _timestep,
        **_kwargs,
    ):
        self.last_actions = actions.detach().clone()
        return actions + 1


def _teacher_request(layout, conditioning_digest, *, chunk_id: int = 0):
    chunk = layout.chunks[chunk_id]
    video = torch.arange(
        3 * len(chunk.latent_valid_mask) * 4,
        dtype=torch.float32,
    ).reshape(1, 3, len(chunk.latent_valid_mask), 2, 2)
    frame_mask = torch.tensor(chunk.latent_valid_mask, dtype=torch.bool).view(1, -1)
    video = video.masked_fill((~frame_mask)[:, None, :, None, None], 0)
    actions = torch.arange(
        chunk.action_slot_capacity * 20,
        dtype=torch.float32,
    ).reshape(1, chunk.action_slot_capacity, 20)
    action_mask = torch.tensor(chunk.action_valid_mask, dtype=torch.bool).view(1, -1)
    actions = actions.masked_fill((~action_mask)[:, :, None], 0)
    proof = hc.TeacherForcingDatasetPairProof(
        dataset_manifest_sha256="b" * 64,
        dataset_episode_id="stage2b-episode",
        dataset_row_identity=f"stage2b-episode:chunk-{chunk_id}",
        dataset_row_start=0,
        dataset_row_end_exclusive=layout.valid_raw_count,
        layout_spec_sha256=layout.layout_spec_sha256,
        layout_instance_digest=layout.layout_instance_digest,
        video_tensor_digest=hc.tensor_digest(video),
        frame_valid_mask_digest=hc.tensor_digest(frame_mask),
        action_tensor_digest=hc.tensor_digest(actions),
        action_mask_digest=hc.tensor_digest(action_mask),
        row_order_manifest_sha256="c" * 64,
    )
    return h2.Stage2BTeacherForcingCommitRequest(
        commit_id=f"stage2b-production-core-{chunk_id}",
        transaction_nonce=f"stage2b-production-core-nonce-{chunk_id}",
        expected_episode_id="stage2b-episode",
        expected_episode_epoch=0,
        expected_revision=chunk_id,
        content_time=hc.ContentTime.from_layout(
            episode_id="stage2b-episode",
            episode_epoch=0,
            layout=layout,
            chunk=chunk,
        ),
        video=video,
        frame_valid_mask=frame_mask,
        actions=actions,
        action_valid_mask=action_mask,
        proof=proof,
        conditioning_digest=conditioning_digest,
    )


def test_manager_drives_production_stager_without_request_history_or_literal_no_action() -> None:
    layout = _layout()
    chunk = layout.chunks[0]
    registry = hc.LayerRegistry(
        layers=(
            hc.LayerSpec(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="synthetic.Stage2BProductionGDN",
                main_shortconv_enabled=False,
                ffn_tconv_enabled=True,
            ),
        )
    )
    manager = h2.Stage2BHybridCacheManager.for_synthetic_tests(
        registry=registry,
        episode_id="stage2b-episode",
        layout=layout,
        receipt_publisher=_Publisher(),
        staging_variant=h2.CACHStagingVariant.CACH_A,
    )
    source_digest = manager.state.state_manifest_digest
    context = torch.randn(1, 4, 4)
    context_mask = torch.ones(1, 4, dtype=torch.bool)
    seq_lens = torch.tensor([4], dtype=torch.long)
    proprio = torch.randn(1, 20, dtype=torch.float32)
    binding = SelectedProprioBinding(
        layout_instance_digest=layout.layout_instance_digest,
        chunk_id=0,
        raw_index=chunk.proprio_raw_index,
        timestamp=chunk.proprio_timestamp,
        source_receipt_digest=chunk.proprio_source_receipt_digest,
        proprio_value_sha256=canonical_proprio_row_sha256(proprio[0]),
    )
    conditioning = CACHChunkConditioning.create(
        layout_instance_digest=layout.layout_instance_digest,
        chunk_id=0,
        source_state_manifest_digest=source_digest,
        context=context,
        context_mask=context_mask,
        seq_lens=seq_lens,
        proprio_state=proprio,
        proprio_bindings=(binding,),
    )
    core = CACHNumericalCore(action_dim=20)
    with torch.no_grad():
        core.no_action_slot.fill_(0.25)
    video_backbone = _FakeVideoBackbone()
    action_backbone = _FakeActionBackbone()
    stager = CACHTeacherForcingPairedStager(
        core=core,
        video_backbone=video_backbone,
        action_backbone=action_backbone,
        proprio_context_encoder=nn.Linear(20, 4),
        registry=registry,
        layout=layout,
        conditioning=conditioning,
        staging_variant=h2.CACHStagingVariant.CACH_A,
        synthetic_test_only=True,
    )
    request = _teacher_request(layout, conditioning.conditioning_digest)

    receipt = manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=stager,
    )

    assert receipt.conditioning_digest == conditioning.conditioning_digest
    assert receipt.staging_variant is CACHStagingVariant.CACH_A
    assert manager.state.revision == 1
    assert manager.state.staging_variant is CACHStagingVariant.CACH_A
    expected_condition = reduce_end_of_bin_action_condition(
        request.actions,
        committed_actions=None,
        chunk=chunk,
        no_action_slot=core.no_action_slot,
    )
    torch.testing.assert_close(
        video_backbone.last_action_condition,
        expected_condition.condition,
        atol=0,
        rtol=0,
    )
    assert manager.state.committed_action_history is not None
    torch.testing.assert_close(
        video_backbone.last_action_condition[:, 0],
        core.no_action_slot.detach().view(1, -1),
        atol=0,
        rtol=0,
    )
    assert not bool(
        video_backbone.last_action_condition[
            ~video_backbone.last_frame_valid_mask
        ].count_nonzero()
    )
    assert action_backbone.last_actions.shape[1] == chunk.valid_action_count
    assert tuple(action_backbone.last_rope_positions.tolist()) == tuple(
        range(chunk.action_rope_start, chunk.action_rope_end)
    )


def test_production_stager_continuation_consumes_prior_cache_and_exact_history() -> None:
    layout = _layout(valid_raw_count=33)
    registry = hc.LayerRegistry(
        layers=(
            hc.LayerSpec(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="synthetic.Stage2BProductionGDN",
                main_shortconv_enabled=False,
                ffn_tconv_enabled=True,
            ),
        )
    )
    manager = h2.Stage2BHybridCacheManager.for_synthetic_tests(
        registry=registry,
        episode_id="stage2b-episode",
        layout=layout,
        receipt_publisher=_Publisher(),
        staging_variant=h2.CACHStagingVariant.CACH_A,
    )
    core = CACHNumericalCore(action_dim=20)
    video_backbone = _FakeVideoBackbone()
    action_backbone = _FakeActionBackbone()
    proprio_encoder = nn.Linear(20, 20)
    receipts = []
    requests = []

    for chunk_id in (0, 1):
        source_digest = manager.state.state_manifest_digest
        conditioning = _conditioning_for(
            layout,
            chunk_id=chunk_id,
            source_digest=source_digest,
        )
        stager = CACHTeacherForcingPairedStager(
            core=core,
            video_backbone=video_backbone,
            action_backbone=action_backbone,
            proprio_context_encoder=proprio_encoder,
            registry=registry,
            layout=layout,
            conditioning=conditioning,
            staging_variant=h2.CACHStagingVariant.CACH_A,
            synthetic_test_only=True,
        )
        request = _teacher_request(
            layout,
            conditioning.conditioning_digest,
            chunk_id=chunk_id,
        )
        requests.append(request)
        receipts.append(
            manager.commit_paired(
                commit_source=hc.CommitSource.TEACHER_FORCING,
                request=request,
                stage_callback=stager,
            )
        )

    assert video_backbone.incoming_main_s_kv[0] is None
    torch.testing.assert_close(
        video_backbone.incoming_main_s_kv[1],
        torch.ones_like(video_backbone.incoming_main_s_kv[1]),
        atol=0,
        rtol=0,
    )
    assert receipts[1].action_history_digest_before == (
        receipts[0].action_history_digest_after
    )
    assert receipts[1].action_cursor_before == layout.chunks[1].action_start
    assert manager.state.action_cursor == layout.chunks[1].action_end
    assert tuple(action_backbone.rope_positions[1].tolist()) == tuple(
        range(
            layout.chunks[1].action_rope_start,
            layout.chunks[1].action_rope_end,
        )
    )
    assert not torch.equal(
        video_backbone.action_conditions[1][:, 0],
        core.no_action_slot.detach().view(1, -1),
    )
    committed_prefix = requests[0].actions[
        :, : layout.chunks[0].valid_action_count
    ]
    expected_continuation = reduce_end_of_bin_action_condition(
        requests[1].actions,
        committed_actions=committed_prefix,
        chunk=layout.chunks[1],
        no_action_slot=core.no_action_slot,
    )
    torch.testing.assert_close(
        video_backbone.action_conditions[1],
        expected_continuation.condition,
        atol=0,
        rtol=0,
    )


def test_ref_gdn_corrected_is_explicit_action_to_video_identity_bypass() -> None:
    layout = _layout()
    registry = hc.LayerRegistry(
        layers=(
            hc.LayerSpec(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="synthetic.Stage2BReferenceGDN",
                main_shortconv_enabled=False,
                ffn_tconv_enabled=True,
            ),
        )
    )
    manager = h2.Stage2BHybridCacheManager.for_synthetic_tests(
        registry=registry,
        episode_id="stage2b-episode",
        layout=layout,
        receipt_publisher=_Publisher(),
        staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
    )
    conditioning = _conditioning_for(
        layout,
        chunk_id=0,
        source_digest=manager.state.state_manifest_digest,
    )
    core = CACHNumericalCore(action_dim=20)
    video_backbone = _FakeVideoBackbone()
    action_backbone = _FakeActionBackbone()
    request = _teacher_request(layout, conditioning.conditioning_digest)

    wrong_variant = CACHTeacherForcingPairedStager(
        core=core,
        video_backbone=video_backbone,
        action_backbone=action_backbone,
        proprio_context_encoder=nn.Linear(20, 20),
        registry=registry,
        layout=layout,
        conditioning=conditioning,
        staging_variant=CACHStagingVariant.CACH_A,
        synthetic_test_only=True,
    )
    with pytest.raises(hc.CacheContractError) as mixed:
        manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=request,
            stage_callback=wrong_variant,
        )
    assert mixed.value.code == "COMMIT_STAGING_FAILED"
    assert isinstance(mixed.value.__cause__, ValueError)
    assert "variants differ" in str(mixed.value.__cause__)
    assert manager.state.revision == 0

    with torch.no_grad():
        core.no_action_slot.fill_(float("nan"))
    reference_stager = CACHTeacherForcingPairedStager(
        core=core,
        video_backbone=video_backbone,
        action_backbone=action_backbone,
        proprio_context_encoder=nn.Linear(20, 20),
        registry=registry,
        layout=layout,
        conditioning=conditioning,
        staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
        synthetic_test_only=True,
    )
    receipt = manager.commit_paired(
        commit_source=hc.CommitSource.TEACHER_FORCING,
        request=request,
        stage_callback=reference_stager,
    )

    assert receipt.staging_variant is CACHStagingVariant.REF_GDN_CORRECTED
    assert receipt.to_manifest()["staging_variant"] == "ref_gdn_corrected"
    assert b'"staging_variant":"ref_gdn_corrected"' in receipt.canonical_bytes
    assert hashlib.sha256(receipt.canonical_bytes).hexdigest() == (
        receipt.receipt_sha256
    )
    assert video_backbone.last_action_condition is None
    assert action_backbone.last_actions is not None
    state = manager.state
    assert state.staging_variant is CACHStagingVariant.REF_GDN_CORRECTED
    assert state.to_manifest()["history_summary_operator"] == (
        "gdn_recurrent_video_only_paired_commit_v1"
    )
    expected_transaction = hc._sha256_bytes(
        hc._canonical_json_bytes(
            {
                "conditioning_digest": request.conditioning_digest,
                "schema": "cach.stage2b.paired_payload.v1",
                "staging_variant": CACHStagingVariant.REF_GDN_CORRECTED.value,
                "teacher_pair_payload_digest": receipt.teacher_pair_payload_digest,
            }
        )
    )
    alternate_transaction = hc._sha256_bytes(
        hc._canonical_json_bytes(
            {
                "conditioning_digest": request.conditioning_digest,
                "schema": "cach.stage2b.paired_payload.v1",
                "staging_variant": CACHStagingVariant.CACH_A.value,
                "teacher_pair_payload_digest": receipt.teacher_pair_payload_digest,
            }
        )
    )
    assert receipt.paired_payload_digest == expected_transaction
    assert alternate_transaction != receipt.paired_payload_digest

    next_layout = _layout()
    reset = manager.reset(
        reset_id="stage2b-reference-reset",
        transaction_nonce="stage2b-reference-reset-nonce",
        new_episode_id="stage2b-reference-reset-episode",
        layout=next_layout,
    )
    assert reset.staging_variant is CACHStagingVariant.REF_GDN_CORRECTED
    assert reset.to_manifest()["staging_variant"] == "ref_gdn_corrected"
    assert b'"staging_variant":"ref_gdn_corrected"' in reset.canonical_bytes
    assert hashlib.sha256(reset.canonical_bytes).hexdigest() == reset.receipt_sha256
    assert manager.state.staging_variant is CACHStagingVariant.REF_GDN_CORRECTED


def test_non_synthetic_manager_rejects_missing_conditioning_and_callback_bypass() -> None:
    layout = replace(_layout(), synthetic_test_only=False)
    registry = hc.LayerRegistry(
        layers=tuple(
            hc.LayerSpec(
                layer_index=index,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="synthetic.Stage2BProductionGuard",
                main_shortconv_enabled=False,
                ffn_tconv_enabled=True,
            )
            for index in range(20)
        )
    )
    manager = h2.Stage2BHybridCacheManager(
        registry=registry,
        episode_id="stage2b-episode",
        layout=layout,
        receipt_publisher=_Publisher(),
        staging_variant=h2.CACHStagingVariant.CACH_A,
    )
    missing = _teacher_request(layout, None)
    with pytest.raises(hc.CacheContractError) as no_conditioning:
        manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=missing,
            stage_callback=lambda context: (),
        )
    assert no_conditioning.value.code == "COMMIT_CONDITIONING_IDENTITY_MISSING"

    bypass = replace(missing, conditioning_digest="d" * 64)
    with pytest.raises(hc.CacheContractError) as unauthorized:
        manager.commit_paired(
            commit_source=hc.CommitSource.TEACHER_FORCING,
            request=bypass,
            stage_callback=lambda context: (),
        )
    assert unauthorized.value.code == "COMMIT_STAGING_CALLBACK_UNAUTHORIZED"
    assert manager.state.revision == 0


def test_production_stager_binds_variant_to_explicit_adapter_action_seam() -> None:
    layout = replace(_layout(), synthetic_test_only=False)
    conditioning = _conditioning_for(
        layout,
        chunk_id=0,
        source_digest="a" * 64,
    )
    registry = hc.LayerRegistry(
        layers=tuple(
            hc.LayerSpec(
                layer_index=index,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="synthetic.Stage2BProductionGuard",
                main_shortconv_enabled=False,
                ffn_tconv_enabled=True,
            )
            for index in range(20)
        )
    )

    def video_backbone(*, enabled: bool):
        video = _FakeVideoBackbone()
        video._pipe = SimpleNamespace(
            config={
                "use_delta_pose_additive": enabled,
                "delta_pose_additive_dim": 20,
            }
        )
        video._dit = nn.Module()
        video._dit._cach_delta_pose_output_zero_verified = True
        return video

    common = {
        "core": CACHNumericalCore(action_dim=20),
        "action_backbone": _FakeActionBackbone(),
        "proprio_context_encoder": nn.Linear(20, 20),
        "registry": registry,
        "layout": layout,
        "conditioning": conditioning,
    }
    candidate = CACHTeacherForcingPairedStager(
        **common,
        video_backbone=video_backbone(enabled=True),
        staging_variant=CACHStagingVariant.CACH_A,
    )
    reference = CACHTeacherForcingPairedStager(
        **common,
        video_backbone=video_backbone(enabled=False),
        staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
    )
    assert candidate._staging_variant is CACHStagingVariant.CACH_A
    assert reference._staging_variant is CACHStagingVariant.REF_GDN_CORRECTED

    with pytest.raises(RuntimeError, match="CACH-A requires"):
        CACHTeacherForcingPairedStager(
            **common,
            video_backbone=video_backbone(enabled=False),
            staging_variant=CACHStagingVariant.CACH_A,
        )
    with pytest.raises(RuntimeError, match="identity bypass"):
        CACHTeacherForcingPairedStager(
            **common,
            video_backbone=video_backbone(enabled=True),
            staging_variant=CACHStagingVariant.REF_GDN_CORRECTED,
        )
