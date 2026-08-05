"""Stage-2 mini GDN numerical checks; no data, checkpoint, or optimizer step."""

from __future__ import annotations

import os

import pytest
import torch


os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cached mini GDN uses CUDA/Triton",
)


def _build_backbone():
    from sana_wam.model.video_backbone.sana.adapter import SanaVideoBackbone

    return SanaVideoBackbone.from_mini_config(
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


def _inputs(*, fixed_k: int, valid_k: int, seed: int = 20260731):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    video = torch.zeros(
        1,
        16,
        fixed_k,
        4,
        4,
        device="cuda",
        dtype=torch.bfloat16,
    )
    video[:, :, :valid_k] = torch.randn(
        1,
        16,
        valid_k,
        4,
        4,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    action = torch.zeros(
        1,
        fixed_k,
        20,
        device="cuda",
        dtype=torch.bfloat16,
    )
    action[:, :valid_k] = torch.randn(
        1,
        valid_k,
        20,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    frame_mask = torch.zeros(1, fixed_k, device="cuda", dtype=torch.bool)
    frame_mask[:, :valid_k] = True
    return {
        "video": video,
        "action": action,
        "frame_mask": frame_mask,
        "timestep": torch.tensor([500.0], device="cuda"),
        "context": torch.randn(
            1,
            8,
            64,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        "seq_lens": torch.tensor([8], device="cuda", dtype=torch.long),
    }


def _run(backbone, values, *, end_f: int, start_f: int = 0, cache=None):
    if cache is None:
        cache = backbone.empty_kv_cache(backbone.num_layers)
    output, bridges, cache = backbone.run_chunk(
        values["video"],
        values["timestep"],
        context=values["context"],
        seq_lens=values["seq_lens"],
        kv_cache=cache,
        start_f=start_f,
        end_f=end_f,
        save_kv_cache=True,
        bridge_layers=(0,),
        action_condition=values["action"],
        frame_valid_mask=values["frame_mask"],
    )
    return output, bridges, cache


def _assert_cache_equal(left, right) -> None:
    assert len(left) == len(right)
    for left_layer, right_layer in zip(left, right):
        assert len(left_layer) == len(right_layer) == 10
        for left_slot, right_slot in zip(left_layer, right_layer):
            if left_slot is None or right_slot is None:
                assert left_slot is right_slot
            elif isinstance(left_slot, torch.Tensor):
                assert isinstance(right_slot, torch.Tensor)
                torch.testing.assert_close(
                    left_slot,
                    right_slot,
                    atol=0,
                    rtol=0,
                )
            else:
                assert left_slot == right_slot


def _clone_cache(cache):
    return [
        [
            value.detach().clone() if isinstance(value, torch.Tensor) else value
            for value in layer
        ]
        for layer in cache
    ]


@requires_cuda
def test_partial_tail_matches_direct_valid_prefix_and_restores_fixed_k() -> None:
    torch.manual_seed(20260731)
    backbone = _build_backbone()
    partial = _inputs(fixed_k=3, valid_k=2)
    direct = {
        **partial,
        "video": partial["video"][:, :, :2].clone(),
        "action": partial["action"][:, :2].clone(),
        "frame_mask": partial["frame_mask"][:, :2].clone(),
    }

    partial_output, partial_bridges, partial_cache = _run(
        backbone,
        partial,
        end_f=2,
    )
    direct_output, direct_bridges, direct_cache = _run(
        backbone,
        direct,
        end_f=2,
    )

    assert partial_output.shape == (1, 16, 3, 4, 4)
    assert direct_output.shape == (1, 16, 2, 4, 4)
    torch.testing.assert_close(
        partial_output[:, :, :2],
        direct_output,
        atol=0,
        rtol=0,
    )
    assert not bool(partial_output[:, :, 2:].count_nonzero())

    direct_tokens = direct_bridges[0].shape[1]
    assert partial_bridges[0].shape[1] * 2 == direct_tokens * 3
    torch.testing.assert_close(
        partial_bridges[0][:, :direct_tokens],
        direct_bridges[0],
        atol=0,
        rtol=0,
    )
    assert not bool(partial_bridges[0][:, direct_tokens:].count_nonzero())
    _assert_cache_equal(partial_cache, direct_cache)


@requires_cuda
def test_zero_initialized_action_adapter_is_exact_bypass() -> None:
    torch.manual_seed(20260731)
    backbone = _build_backbone()
    conditioned = _inputs(fixed_k=3, valid_k=3)
    bypass = {**conditioned, "action": torch.zeros_like(conditioned["action"])}

    conditioned_output, conditioned_bridges, conditioned_cache = _run(
        backbone,
        conditioned,
        end_f=3,
    )
    bypass_output, bypass_bridges, bypass_cache = _run(
        backbone,
        bypass,
        end_f=3,
    )

    torch.testing.assert_close(
        conditioned_output,
        bypass_output,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        conditioned_bridges[0],
        bypass_bridges[0],
        atol=0,
        rtol=0,
    )
    _assert_cache_equal(conditioned_cache, bypass_cache)


@requires_cuda
def test_zero_initialized_action_adapter_has_finite_nonzero_parameter_gradient() -> None:
    """Prove the exact-zero output seam is wired, without changing its weights."""

    torch.manual_seed(20260731)
    backbone = _build_backbone()
    values = _inputs(fixed_k=3, valid_k=3)
    projection = backbone._dit.blocks[0].delta_pose_proj

    assert not bool(projection.weight.count_nonzero())
    assert projection.bias is not None
    assert not bool(projection.bias.count_nonzero())

    _, bridges, _ = _run(backbone, values, end_f=3)
    bridges[0].float().square().mean().backward()

    assert projection.weight.grad is not None
    assert projection.bias.grad is not None
    assert bool(torch.isfinite(projection.weight.grad).all())
    assert bool(torch.isfinite(projection.bias.grad).all())
    assert bool(projection.weight.grad.count_nonzero())
    assert bool(projection.bias.grad.count_nonzero())
    assert not bool(projection.weight.count_nonzero())
    assert not bool(projection.bias.count_nonzero())


@requires_cuda
def test_partial_tail_rejects_bad_window_before_vendor_forward() -> None:
    backbone = _build_backbone()
    partial = _inputs(fixed_k=3, valid_k=2)
    with pytest.raises(ValueError, match="window must equal the valid prefix"):
        _run(backbone, partial, end_f=3)


@requires_cuda
def test_partial_tail_rejects_nonzero_padding_before_dtype_conversion() -> None:
    backbone = _build_backbone()
    partial = _inputs(fixed_k=3, valid_k=2)
    partial["video"] = partial["video"].float()
    # This float32 subnormal becomes zero in BF16. The source tensor must still
    # fail before the adapter performs its legacy dtype conversion.
    partial["video"][:, :, 2:] = torch.finfo(torch.float32).tiny / 2
    with pytest.raises(ValueError, match="video slots must be exact zero"):
        _run(backbone, partial, end_f=2)


@requires_cuda
def test_real_mini_gdn_cache_obeys_typed_codec_contract() -> None:
    from sana_wam.model.video_backbone.sana import hybrid_cache as hc
    from sana_wam.model.video_backbone.sana.hybrid_cache_codec import (
        scratch_to_vendor_cache,
        vendor_cache_to_staged_payloads,
    )

    torch.manual_seed(20260731)
    backbone = _build_backbone()
    values = _inputs(fixed_k=3, valid_k=2)
    _, _, vendor_cache = _run(backbone, values, end_f=2)
    registry = hc.LayerRegistry(
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

    payload = vendor_cache_to_staged_payloads(vendor_cache, registry)[0]
    assert frozenset(payload.tensors) == registry.layers[0].required_tensor_fields
    scratch = hc.CacheScratch(
        layers=(
            hc.LayerScratch(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                tensors=hc._TrackedTensorMap(payload.tensors),
            ),
        ),
        source_state_manifest_digest="0" * 64,
        _source_state_identity=1,
        _baseline_fingerprint=(),
    )
    rebuilt = scratch_to_vendor_cache(scratch, registry)
    _assert_cache_equal(rebuilt, vendor_cache)


@requires_cuda
def test_codec_continuation_equivalence_state_advance_and_fresh_reset() -> None:
    from sana_wam.model.video_backbone.sana import hybrid_cache as hc
    from sana_wam.model.video_backbone.sana.hybrid_cache_codec import (
        scratch_to_vendor_cache,
        vendor_cache_to_staged_payloads,
    )

    torch.manual_seed(20260731)
    backbone = _build_backbone()
    chunk0 = _inputs(fixed_k=3, valid_k=3, seed=11)
    chunk1 = _inputs(fixed_k=3, valid_k=3, seed=22)
    output0, bridges0, raw_cache0 = _run(backbone, chunk0, start_f=0, end_f=3)
    frozen_cache0 = _clone_cache(raw_cache0)
    registry = hc.LayerRegistry(
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
    payload = vendor_cache_to_staged_payloads(frozen_cache0, registry)[0]
    frozen_tensors = {
        name: value.detach().clone() for name, value in payload.tensors.items()
    }
    scratch = hc.CacheScratch(
        layers=(
            hc.LayerScratch(
                layer_index=0,
                kind=hc.LayerKind.GDN_FULL_HISTORY,
                tensors=hc._TrackedTensorMap(payload.tensors),
            ),
        ),
        source_state_manifest_digest="0" * 64,
        _source_state_identity=1,
        _baseline_fingerprint=(),
    )

    direct_cache = _clone_cache(frozen_cache0)
    roundtrip_cache = scratch_to_vendor_cache(scratch, registry)
    direct_output, direct_bridges, direct_cache = _run(
        backbone,
        chunk1,
        cache=direct_cache,
        start_f=3,
        end_f=6,
    )
    roundtrip_output, roundtrip_bridges, roundtrip_cache = _run(
        backbone,
        chunk1,
        cache=roundtrip_cache,
        start_f=3,
        end_f=6,
    )
    torch.testing.assert_close(direct_output, roundtrip_output, atol=0, rtol=0)
    torch.testing.assert_close(
        direct_bridges[0], roundtrip_bridges[0], atol=0, rtol=0
    )
    _assert_cache_equal(direct_cache, roundtrip_cache)
    assert not torch.equal(frozen_cache0[0][0], direct_cache[0][0])
    assert not torch.equal(frozen_cache0[0][1], direct_cache[0][1])
    for name, value in frozen_tensors.items():
        torch.testing.assert_close(payload.tensors[name], value, atol=0, rtol=0)

    replay_output, replay_bridges, replay_cache = _run(
        backbone,
        chunk0,
        start_f=0,
        end_f=3,
    )
    torch.testing.assert_close(output0, replay_output, atol=0, rtol=0)
    torch.testing.assert_close(bridges0[0], replay_bridges[0], atol=0, rtol=0)
    _assert_cache_equal(frozen_cache0, replay_cache)
