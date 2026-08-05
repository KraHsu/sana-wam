"""CPU contract tests for the strict legacy-cache boundary."""

import pytest
import torch

from sana_wam.model.video_backbone.sana import hybrid_cache as hc
from sana_wam.model.video_backbone.sana.hybrid_cache_codec import (
    scratch_to_vendor_cache,
    vendor_cache_to_staged_payloads,
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


def _vendor_cache() -> list[list[object]]:
    slots: list[object] = [None] * 10
    slots[0] = torch.randn(1, 2, 4, 4)
    slots[1] = torch.randn(1, 2, 4, 1)
    slots[4] = torch.randn(3, 2, 8)
    slots[6] = 1.0
    slots[9] = torch.randn(1, 8, 2, 4)
    return [slots]


def _scratch(payload: hc.StagedLayerPayload) -> hc.CacheScratch:
    return hc.CacheScratch(
        layers=(
            hc.LayerScratch(
                layer_index=payload.layer_index,
                kind=payload.kind,
                tensors=hc._TrackedTensorMap(payload.tensors),
            ),
        ),
        source_state_manifest_digest="0" * 64,
        _source_state_identity=1,
        _baseline_fingerprint=(),
    )


def test_gdn_codec_round_trip_uses_executable_slot_9_contract() -> None:
    registry = _registry()
    source = _vendor_cache()
    payload = vendor_cache_to_staged_payloads(source, registry)[0]

    assert frozenset(payload.tensors) == registry.layers[0].required_tensor_fields
    rebuilt = scratch_to_vendor_cache(_scratch(payload), registry)
    assert rebuilt[0][5] is None
    assert rebuilt[0][6] == 1.0
    for slot in (0, 1, 4, 9):
        torch.testing.assert_close(rebuilt[0][slot], source[0][slot], atol=0, rtol=0)


def test_decode_clones_vendor_tensors() -> None:
    source = _vendor_cache()
    payload = vendor_cache_to_staged_payloads(source, _registry())[0]
    original = payload.tensors["main_s_kv"].clone()
    source[0][0].zero_()
    torch.testing.assert_close(payload.tensors["main_s_kv"], original, atol=0, rtol=0)


@pytest.mark.parametrize("slot,value", [(5, torch.ones(1)), (7, torch.ones(1)), (8, 1.0)])
def test_decode_rejects_nonempty_unowned_slots(slot: int, value: object) -> None:
    source = _vendor_cache()
    source[0][slot] = value
    with pytest.raises(hc.CacheContractError, match="CACHE_LEGACY_SLOT_MISMATCH"):
        vendor_cache_to_staged_payloads(source, _registry())


def test_decode_rejects_non_float_or_wrong_type_flag() -> None:
    for bad_flag in (torch.tensor(1.0), 0.0, 1):
        source = _vendor_cache()
        source[0][6] = bad_flag
        with pytest.raises(
            hc.CacheContractError,
            match="CACHE_LEGACY_TYPE_FLAG_MISMATCH",
        ):
            vendor_cache_to_staged_payloads(source, _registry())


def test_empty_scratch_encodes_as_pristine_vendor_cache() -> None:
    empty_payload = hc.StagedLayerPayload(
        layer_index=0,
        kind=hc.LayerKind.GDN_FULL_HISTORY,
        tensors={},
    )
    rebuilt = scratch_to_vendor_cache(_scratch(empty_payload), _registry())
    assert rebuilt == [[None] * 10]


def test_decode_requires_exact_mutable_legacy_list_shape() -> None:
    source = _vendor_cache()
    with pytest.raises(hc.CacheContractError, match="CACHE_LEGACY_SLOT_MISMATCH"):
        vendor_cache_to_staged_payloads(tuple(source), _registry())
    with pytest.raises(hc.CacheContractError, match="CACHE_LEGACY_SLOT_MISMATCH"):
        vendor_cache_to_staged_payloads([tuple(source[0])], _registry())


def test_encode_rejects_mixed_empty_and_populated_layers() -> None:
    first_spec = _registry().layers[0]
    registry = hc.LayerRegistry(
        layers=(
            first_spec,
            hc.LayerSpec(
                layer_index=1,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                operator_class="CachedChunkCausalGDN",
                main_shortconv_enabled=True,
                ffn_tconv_enabled=True,
            ),
        )
    )
    payload = vendor_cache_to_staged_payloads(_vendor_cache(), _registry())[0]
    scratch = hc.CacheScratch(
        layers=(
            hc.LayerScratch(
                layer_index=0,
                kind=payload.kind,
                tensors=hc._TrackedTensorMap(payload.tensors),
            ),
            hc.LayerScratch(
                layer_index=1,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                tensors=hc._TrackedTensorMap({}),
            ),
        ),
        source_state_manifest_digest="0" * 64,
        _source_state_identity=1,
        _baseline_fingerprint=(),
    )
    with pytest.raises(hc.CacheContractError, match="CACHE_SCHEMA_MISMATCH"):
        scratch_to_vendor_cache(scratch, registry)
