from __future__ import annotations

import importlib.util
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from sana_wam.model.action_chunk_layout import (
    build_synthetic_chunk_action_layout_for_tests,
    synthetic_equal_rate_proof,
    synthetic_layout_spec,
)


# Load the pure contract without executing the SANA package's dependency-heavy
# __init__.py.  Stage 1 must remain testable without a vendored cache/model codec.
_MODULE_NAME = "_cach_hybrid_cache_contract_under_test"
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/sana_wam/model/video_backbone/sana/hybrid_cache.py"
)
if _MODULE_NAME not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    assert _SPEC is not None and _SPEC.loader is not None
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_MODULE_NAME] = _MODULE
    _SPEC.loader.exec_module(_MODULE)
hc = sys.modules[_MODULE_NAME]


def _layout_for_latent_count(valid_latent_count: int):
    valid_raw_count = (valid_latent_count - 1) * 8 + 1
    spec = synthetic_layout_spec(
        frame_chunk_size=3,
        temporal_compression=8,
        video_stride=1,
    )
    proof = synthetic_equal_rate_proof(
        valid_raw_count=valid_raw_count,
        video_stride=1,
    )
    return build_synthetic_chunk_action_layout_for_tests(
        spec=spec,
        proof=proof,
        row_start_raw_index=0,
        valid_raw_count=valid_raw_count,
        video_valid_mask=(True,) * valid_raw_count,
        action_valid_mask=(True,) * (valid_raw_count - 1),
    )


L5_LAYOUT = _layout_for_latent_count(5)
LAYOUT_SHA = L5_LAYOUT.layout_spec_sha256
INSTANCE_SHA = L5_LAYOUT.layout_instance_digest


def _content_time(**overrides):
    content_time = hc.ContentTime.from_layout(
        episode_id="episode-0",
        episode_epoch=0,
        layout=L5_LAYOUT,
        chunk=L5_LAYOUT.chunks[0],
    )
    return replace(content_time, **overrides) if overrides else content_time


def _registry():
    return hc.LayerRegistry(
        layers=(
            hc.LayerSpec(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="GatedDeltaNet",
                main_shortconv_enabled=True,
            ),
            hc.LayerSpec(
                layer_index=1,
                kind=hc.LayerKind.SOFTMAX_PREVIOUS_CHUNK,
                operator_class="MultiHeadCrossAttention",
                tokens_per_latent=2,
            ),
        )
    )


def test_content_time_is_frozen_and_manifest_is_exact() -> None:
    content_time = _content_time()
    assert content_time.to_manifest() == {
        "action_rope_end_exclusive": 16,
        "action_rope_start": 0,
        "action_token_end_exclusive": 16,
        "action_token_start": 0,
        "bootstrap_mode": "first_frame_pinned",
        "chunk_id": 0,
        "episode_epoch": 0,
        "episode_id": "episode-0",
        "latent_end_exclusive": 3,
        "latent_start": 0,
        "layout_instance_digest": INSTANCE_SHA,
        "layout_spec_sha256": LAYOUT_SHA,
        "observed_prefix_chunks": 0,
        "raw_observation_end_exclusive": 17,
        "raw_observation_start": 0,
    }
    with pytest.raises(FrozenInstanceError):
        content_time.chunk_id = 1


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"chunk_id": True}, "CACHE_SCHEMA_MISMATCH"),
        ({"latent_end_exclusive": 0}, "CACHE_CONTENT_TIME_MISMATCH"),
        (
            {"action_token_start": 2, "action_token_end_exclusive": 1},
            "CACHE_CONTENT_TIME_MISMATCH",
        ),
        ({"action_rope_start": 1}, "CACHE_CONTENT_TIME_MISMATCH"),
        ({"bootstrap_mode": "legacy_prefix"}, "BOOTSTRAP_CONTRACT_MISMATCH"),
        ({"observed_prefix_chunks": 1}, "BOOTSTRAP_CONTRACT_MISMATCH"),
        ({"latent_start": 1}, "CACHE_CONTENT_TIME_MISMATCH"),
        ({"action_token_start": 1}, "CACHE_CONTENT_TIME_MISMATCH"),
    ],
)
def test_content_time_rejects_ambiguous_or_legacy_identity(
    overrides, code
) -> None:
    with pytest.raises(hc.CacheContractError) as exc_info:
        _content_time(**overrides)
    assert exc_info.value.code == code


@pytest.mark.parametrize(
    ("valid_latent_count", "expected"),
    [
        (
            5,
            [
                (0, 0, 3, 0, 17, 0, 16, 0, 16),
                (1, 3, 5, 17, 33, 16, 32, 16, 32),
            ],
        ),
        (
            8,
            [
                (0, 0, 3, 0, 17, 0, 16, 0, 16),
                (1, 3, 6, 17, 41, 16, 40, 16, 40),
                (2, 6, 8, 41, 57, 40, 56, 40, 56),
            ],
        ),
    ],
)
def test_content_time_is_derived_from_exact_l5_l8_synthetic_layout_chunks(
    valid_latent_count, expected
) -> None:
    layout = _layout_for_latent_count(valid_latent_count)
    content_times = tuple(
        hc.ContentTime.from_layout(
            episode_id="episode-0",
            episode_epoch=0,
            layout=layout,
            chunk=chunk,
        )
        for chunk in layout.chunks
    )
    assert [
        (
            content_time.chunk_id,
            content_time.latent_start,
            content_time.latent_end_exclusive,
            content_time.raw_observation_start,
            content_time.raw_observation_end_exclusive,
            content_time.action_token_start,
            content_time.action_token_end_exclusive,
            content_time.action_rope_start,
            content_time.action_rope_end_exclusive,
        )
        for content_time in content_times
    ] == expected
    assert all(
        left.latent_end_exclusive == right.latent_start
        and left.raw_observation_end_exclusive
        == right.raw_observation_start
        and left.action_token_end_exclusive == right.action_token_start
        and left.action_rope_end_exclusive == right.action_rope_start
        for left, right in zip(content_times, content_times[1:])
    )
    assert content_times[-1].raw_observation_end_exclusive == (
        layout.valid_raw_count
    )


def test_content_time_rejects_unowned_chunk_and_forged_layout_geometry() -> None:
    with pytest.raises(hc.CacheContractError) as exc_info:
        hc.ContentTime.from_layout(
            episode_id="episode-0",
            episode_epoch=0,
            layout=L5_LAYOUT,
            chunk=replace(L5_LAYOUT.chunks[0]),
        )
    assert exc_info.value.code == "CACHE_CONTENT_TIME_MISMATCH"

    forged = replace(
        _content_time(),
        raw_observation_end_exclusive=16,
    )
    with pytest.raises(hc.CacheContractError) as exc_info:
        forged.verify_layout_binding(
            layout=L5_LAYOUT,
            chunk=L5_LAYOUT.chunks[0],
        )
    assert exc_info.value.code == "CACHE_CONTENT_TIME_MISMATCH"


def test_layer_registry_is_typed_contiguous_and_digest_stable() -> None:
    first = _registry()
    second = _registry()
    assert first.manifest_digest == second.manifest_digest
    assert first.layers[0].required_tensor_fields == frozenset(
        {
            "main_s_kv",
            "main_s_z",
            "main_shortconv_left_context",
            "ffn_tconv_left_context",
        }
    )
    assert first.layers[1].required_tensor_fields == frozenset(
        {"main_k_post_rope", "main_v", "ffn_tconv_left_context"}
    )

    with pytest.raises(hc.CacheContractError) as exc_info:
        hc.LayerRegistry(
            layers=(
                hc.LayerSpec(
                    layer_index=1,
                    kind=hc.LayerKind.GDN_FULL_HISTORY,
                    operator_class="GatedDeltaNet",
                ),
            )
        )
    assert exc_info.value.code == "CACHE_SCHEMA_MISMATCH"


def test_cach_a_registry_is_exactly_twenty_gdn_and_zero_softmax() -> None:
    valid = hc.LayerRegistry(
        layers=tuple(
            hc.LayerSpec(
                layer_index=index,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="GatedDeltaNet",
            )
            for index in range(20)
        )
    )
    hc.validate_cach_a_registry(valid)

    with pytest.raises(hc.CacheContractError) as exc_info:
        hc.validate_cach_a_registry(_registry())
    assert exc_info.value.code == "CACHE_CACH_A_TOPOLOGY_MISMATCH"


def test_registry_rejects_attnres_and_ambiguous_softmax_history() -> None:
    with pytest.raises(hc.CacheContractError) as exc_info:
        hc.LayerSpec(
            layer_index=0,
            kind=hc.LayerKind.GDN_FULL_HISTORY,
            operator_class="SanaAttnResBlock",
        )
    assert exc_info.value.code == "CACHE_ATTNRES_PERSISTED"

    with pytest.raises(hc.CacheContractError) as exc_info:
        hc.LayerSpec(
            layer_index=0,
            kind=hc.LayerKind.SOFTMAX_PREVIOUS_CHUNK,
            operator_class="MultiHeadCrossAttention",
        )
    assert exc_info.value.code == "CACHE_SCHEMA_MISMATCH"


def test_tensor_snapshot_clones_source_and_portable_manifest_has_no_pointer() -> None:
    source = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    snapshot = hc.TensorSnapshot.from_tensor("layer/0/main_s_kv", source)
    before = snapshot.to_manifest()
    source.add_(100)
    after = snapshot.to_manifest()

    assert before == after
    assert "data_ptr" not in repr(after)
    exported = snapshot.clone_tensor()
    exported.zero_()
    assert snapshot.to_manifest() == before


def test_empty_state_manifest_binds_registry_and_is_deterministic() -> None:
    registry = _registry()
    first = hc.HybridTemporalState.empty(
        episode_id="episode-0",
        episode_epoch=0,
        layout_spec_sha256=LAYOUT_SHA,
        layout_instance_digest=INSTANCE_SHA,
        registry=registry,
    )
    second = hc.HybridTemporalState.empty(
        episode_id="episode-0",
        episode_epoch=0,
        layout_spec_sha256=LAYOUT_SHA,
        layout_instance_digest=INSTANCE_SHA,
        registry=registry,
    )

    assert first.state_manifest_digest == second.state_manifest_digest
    assert first.layer_registry_digest == registry.manifest_digest
    assert first.revision == first.next_chunk_id == first.action_cursor == 0
    assert first.committed_through_chunk is None
    assert all(layer.is_empty for layer in first.layer_states)
    assert first.to_manifest()["pair_evidence_mode"] == "none"
