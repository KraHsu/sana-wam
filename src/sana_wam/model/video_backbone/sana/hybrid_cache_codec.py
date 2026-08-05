"""Strict Stage-2 codec for the pinned SANA legacy ``list[10]`` cache.

The vendored cache is transaction-local only.  It must be decoded into typed
payloads before publication and reconstructed from :class:`CacheScratch`
before another vendor call.  No live typed state is ever exposed to SANA.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from .hybrid_cache import (
    CacheScratch,
    LayerKind,
    LayerRegistry,
    StagedLayerPayload,
    _fail,
)


_CACHE_SLOTS = 10
_SLOT_MAIN_KV = 0
_SLOT_MAIN_Z_OR_V = 1
_SLOT_CAMERA = 2
_SLOT_CAMERA_AUX = 3
_SLOT_MAIN_SHORTCONV = 4
_SLOT_DOCUMENTED_TCONV_UNUSED = 5
_SLOT_TYPE_FLAG = 6
_SLOT_RESERVED_7 = 7
_SLOT_RESERVED_8 = 8
# Pinned CachedGLUMBConvTemp reads/writes kv_cache[-1], i.e. slot 9.  The
# vendor table's slot-5 statement is not the executable contract.
_SLOT_FFN_TCONV_ACTUAL = 9


def _require_tensor(value: object, *, layer: int, slot: int) -> Tensor:
    if not isinstance(value, Tensor):
        _fail(
            "CACHE_LEGACY_SLOT_MISMATCH",
            f"layer {layer} slot {slot} must contain a tensor",
        )
    return value


def _require_none(value: object, *, layer: int, slot: int) -> None:
    if value is not None:
        _fail(
            "CACHE_LEGACY_SLOT_MISMATCH",
            f"layer {layer} unowned slot {slot} must be None",
        )


def _require_type_flag(value: object, *, layer: int, expected: float) -> None:
    if type(value) is not float or value != expected:
        _fail(
            "CACHE_LEGACY_TYPE_FLAG_MISMATCH",
            f"layer {layer} slot 6 must be the float {expected}",
        )


def _owned_tensor(
    slots: Sequence[object],
    *,
    layer: int,
    slot: int,
    enabled: bool,
) -> Tensor | None:
    if enabled:
        return _require_tensor(slots[slot], layer=layer, slot=slot)
    _require_none(slots[slot], layer=layer, slot=slot)
    return None


def vendor_cache_to_staged_payloads(
    cache: Sequence[Sequence[object]],
    registry: LayerRegistry,
) -> tuple[StagedLayerPayload, ...]:
    """Decode a populated vendor cache into detached typed payloads."""

    if not isinstance(registry, LayerRegistry):
        _fail("CACHE_SCHEMA_MISMATCH", "registry must be a LayerRegistry")
    if type(cache) is not list or len(cache) != len(registry.layers):
        _fail(
            "CACHE_LEGACY_SLOT_MISMATCH",
            "vendor cache must contain exactly one entry per registry layer",
        )

    payloads = []
    for spec, slots in zip(registry.layers, cache):
        if type(slots) is not list or len(slots) != _CACHE_SLOTS:
            _fail(
                "CACHE_LEGACY_SLOT_MISMATCH",
                f"layer {spec.layer_index} vendor cache must have exactly 10 slots",
            )
        tensors: dict[str, Tensor] = {}
        if spec.kind is LayerKind.GDN_FULL_HISTORY:
            _require_type_flag(slots[_SLOT_TYPE_FLAG], layer=spec.layer_index, expected=1.0)
            tensors["main_s_kv"] = _require_tensor(
                slots[_SLOT_MAIN_KV], layer=spec.layer_index, slot=_SLOT_MAIN_KV
            )
            tensors["main_s_z"] = _require_tensor(
                slots[_SLOT_MAIN_Z_OR_V],
                layer=spec.layer_index,
                slot=_SLOT_MAIN_Z_OR_V,
            )
            camera = _owned_tensor(
                slots,
                layer=spec.layer_index,
                slot=_SLOT_CAMERA,
                enabled=spec.camera_enabled,
            )
            if camera is not None:
                tensors["camera_s_kv"] = camera
            _require_none(slots[_SLOT_CAMERA_AUX], layer=spec.layer_index, slot=_SLOT_CAMERA_AUX)
            shortconv = _owned_tensor(
                slots,
                layer=spec.layer_index,
                slot=_SLOT_MAIN_SHORTCONV,
                enabled=spec.main_shortconv_enabled,
            )
            if shortconv is not None:
                tensors["main_shortconv_left_context"] = shortconv
        else:
            _require_type_flag(slots[_SLOT_TYPE_FLAG], layer=spec.layer_index, expected=0.0)
            tensors["main_k_post_rope"] = _require_tensor(
                slots[_SLOT_MAIN_KV], layer=spec.layer_index, slot=_SLOT_MAIN_KV
            )
            tensors["main_v"] = _require_tensor(
                slots[_SLOT_MAIN_Z_OR_V],
                layer=spec.layer_index,
                slot=_SLOT_MAIN_Z_OR_V,
            )
            camera_k = _owned_tensor(
                slots,
                layer=spec.layer_index,
                slot=_SLOT_CAMERA,
                enabled=spec.camera_enabled,
            )
            camera_v = _owned_tensor(
                slots,
                layer=spec.layer_index,
                slot=_SLOT_CAMERA_AUX,
                enabled=spec.camera_enabled,
            )
            if camera_k is not None and camera_v is not None:
                tensors["camera_k_post_ucpe"] = camera_k
                tensors["camera_v_post_ucpe"] = camera_v
            _require_none(
                slots[_SLOT_MAIN_SHORTCONV],
                layer=spec.layer_index,
                slot=_SLOT_MAIN_SHORTCONV,
            )

        _require_none(
            slots[_SLOT_DOCUMENTED_TCONV_UNUSED],
            layer=spec.layer_index,
            slot=_SLOT_DOCUMENTED_TCONV_UNUSED,
        )
        _require_none(slots[_SLOT_RESERVED_7], layer=spec.layer_index, slot=_SLOT_RESERVED_7)
        _require_none(slots[_SLOT_RESERVED_8], layer=spec.layer_index, slot=_SLOT_RESERVED_8)
        ffn = _owned_tensor(
            slots,
            layer=spec.layer_index,
            slot=_SLOT_FFN_TCONV_ACTUAL,
            enabled=spec.ffn_tconv_enabled,
        )
        if ffn is not None:
            tensors["ffn_tconv_left_context"] = ffn

        # Detach and clone at the boundary so subsequent vendor mutation cannot
        # alter the staged transaction.
        payloads.append(
            StagedLayerPayload(
                layer_index=spec.layer_index,
                kind=spec.kind,
                tensors={name: value.detach().clone() for name, value in tensors.items()},
            )
        )
    return tuple(payloads)


def scratch_to_vendor_cache(
    scratch: CacheScratch,
    registry: LayerRegistry,
) -> list[list[object]]:
    """Encode typed scratch into a fresh vendor cache, or all-None if empty."""

    if not isinstance(scratch, CacheScratch):
        _fail("CACHE_SCHEMA_MISMATCH", "scratch must be CacheScratch")
    if not isinstance(registry, LayerRegistry) or len(scratch.layers) != len(registry.layers):
        _fail("CACHE_SCHEMA_MISMATCH", "scratch and registry layer counts differ")
    empty_layers = tuple(not layer.tensors for layer in scratch.layers)
    if any(empty_layers) and not all(empty_layers):
        _fail(
            "CACHE_SCHEMA_MISMATCH",
            "scratch cannot mix empty and populated temporal layers",
        )

    cache: list[list[object]] = []
    for layer, spec in zip(scratch.layers, registry.layers):
        if layer.layer_index != spec.layer_index or layer.kind is not spec.kind:
            _fail("CACHE_LAYER_KIND_MISMATCH", "scratch differs from registry")
        observed = frozenset(layer.tensors)
        if not observed:
            cache.append([None] * _CACHE_SLOTS)
            continue
        if observed != spec.required_tensor_fields:
            _fail(
                "CACHE_SCHEMA_MISMATCH",
                f"layer {spec.layer_index} scratch fields differ from registry",
            )
        slots: list[object] = [None] * _CACHE_SLOTS
        if spec.kind is LayerKind.GDN_FULL_HISTORY:
            slots[_SLOT_MAIN_KV] = layer.tensors["main_s_kv"].detach().clone()
            slots[_SLOT_MAIN_Z_OR_V] = layer.tensors["main_s_z"].detach().clone()
            if spec.camera_enabled:
                slots[_SLOT_CAMERA] = layer.tensors["camera_s_kv"].detach().clone()
            if spec.main_shortconv_enabled:
                slots[_SLOT_MAIN_SHORTCONV] = layer.tensors[
                    "main_shortconv_left_context"
                ].detach().clone()
            slots[_SLOT_TYPE_FLAG] = 1.0
        else:
            slots[_SLOT_MAIN_KV] = layer.tensors["main_k_post_rope"].detach().clone()
            slots[_SLOT_MAIN_Z_OR_V] = layer.tensors["main_v"].detach().clone()
            if spec.camera_enabled:
                slots[_SLOT_CAMERA] = layer.tensors["camera_k_post_ucpe"].detach().clone()
                slots[_SLOT_CAMERA_AUX] = layer.tensors[
                    "camera_v_post_ucpe"
                ].detach().clone()
            slots[_SLOT_TYPE_FLAG] = 0.0
        if spec.ffn_tconv_enabled:
            slots[_SLOT_FFN_TCONV_ACTUAL] = layer.tensors[
                "ffn_tconv_left_context"
            ].detach().clone()
        cache.append(slots)
    return cache


__all__ = ["scratch_to_vendor_cache", "vendor_cache_to_staged_payloads"]
