from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sana_wam.model.action_video_memory_adapter import ActionVideoMemoryAdapter
from sana_wam.model.architecture import DualSystemARArchitecture
from sana_wam.model.ar.sana_ar_inference import (
    ARLinearStateCache,
    ar_inference_attn,
    clean_state_from_tokens,
)
from sana_wam.model.ar.sana_ar_linear_attn import (
    _ar_chunked_linear_attn,
    build_ar_seq_meta,
)
from sana_wam.model.ar.sana_ar_mot_driver import SanaARMoTJointDriver


def _nonidentity_adapter(*, layers=1, heads=2, dim=4, rank=2):
    adapter = ActionVideoMemoryAdapter(
        num_layers=layers,
        num_heads=heads,
        head_dim=dim,
        rank=rank,
        init_seed=3,
    )
    with torch.no_grad():
        adapter.k_right.fill_(0.07)
        adapter.v_right.fill_(-0.04)
        adapter.z_logits.fill_(0.2)
    return adapter


def _attention_inputs(meta, *, batch=2, heads=2, dim=4, seed=5):
    generator = torch.Generator().manual_seed(seed)
    tokens = meta.frame_ids.numel()
    q = torch.randn(batch, heads, tokens, dim, generator=generator, dtype=torch.float64)
    k = torch.randn(batch, heads, tokens, dim, generator=generator, dtype=torch.float64)
    v = torch.randn(batch, heads, tokens, dim, generator=generator, dtype=torch.float64)
    phi_q = torch.rand(batch, heads, tokens, dim, generator=generator, dtype=torch.float64)
    phi_k = torch.rand(batch, heads, tokens, dim, generator=generator, dtype=torch.float64)
    return q, k, v, phi_q, phi_k


def _cache_from_clean(meta, k, v, phi_k, *, window=10_000):
    cache = ARLinearStateCache(num_layers=1, window=window)
    for frame_id in range(meta.num_frames):
        selector = (meta.noise_ids == 1) & (meta.frame_ids == frame_id)
        if not bool(selector.any()):
            continue
        index = selector.nonzero(as_tuple=False).squeeze(-1)
        S, z = clean_state_from_tokens(
            k[:, :, index, :], v[:, :, index, :], phi_k[:, :, index, :]
        )
        cache.update(0, frame_id, S, z, is_pred=False)
    return cache


def test_even_video_window_is_read_only_and_preserves_schema():
    cache = ARLinearStateCache(num_layers=1, window=4)
    S = torch.ones(1, 1, 2, 2)
    z = torch.ones(1, 1, 1, 2)
    cache.update(0, 0, S, z, is_pred=False)
    cache.update(0, 1, S * 10, z * 10, is_pred=False)
    cache.update(0, 2, S * 2, z * 2, is_pred=True)

    before = cache.snapshot()
    canonical = cache.windowed_state(0, 3, hi_inclusive=2)
    video = cache.windowed_video_state(0, 3, hi_inclusive=2)
    assert canonical is not None and video is not None
    torch.testing.assert_close(canonical[0], S * 13)
    torch.testing.assert_close(video[0], S * 3)
    assert cache.snapshot() == before
    assert all(
        set(entry) == {"frame_id", "S", "z", "is_pred"}
        for entry in cache.snapshot()[0]
    )

    cache.clear_pred()
    video_after_clear = cache.windowed_video_state(0, 3, hi_inclusive=2)
    assert video_after_clear is not None
    torch.testing.assert_close(video_after_clear[0], S)
    assert cache.info()["frame_ids"] == [0, 1]


def test_identity_training_seam_is_bitwise_for_every_row():
    meta = build_ar_seq_meta(
        num_chunks=3,
        video_tokens_per_chunk=2,
        action_tokens_per_chunk=2,
        window=4,
    )
    inputs = _attention_inputs(meta)
    baseline = _ar_chunked_linear_attn(*inputs, meta, eps=1e-12)
    identity = ActionVideoMemoryAdapter(
        num_layers=1, num_heads=2, head_dim=4, rank=2
    )
    adapted = _ar_chunked_linear_attn(
        *inputs,
        meta,
        eps=1e-12,
        action_video_memory_adapter=identity,
        layer_id=0,
    )
    assert torch.equal(adapted, baseline)


def test_training_adapter_changes_only_odd_noisy_action_rows():
    meta = build_ar_seq_meta(
        num_chunks=3,
        video_tokens_per_chunk=2,
        action_tokens_per_chunk=2,
        window=4,
    )
    inputs = _attention_inputs(meta)
    baseline = _ar_chunked_linear_attn(*inputs, meta, eps=1e-12)
    adapted = _ar_chunked_linear_attn(
        *inputs,
        meta,
        eps=1e-12,
        action_video_memory_adapter=_nonidentity_adapter(),
        layer_id=0,
    )
    action_noisy = (meta.frame_ids % 2 == 1) & (meta.noise_ids == 0)
    assert not torch.equal(adapted[:, :, action_noisy], baseline[:, :, action_noisy])
    assert torch.equal(adapted[:, :, ~action_noisy], baseline[:, :, ~action_noisy])


def test_training_and_cache_query_match_with_adapter():
    window = 3
    meta = build_ar_seq_meta(
        num_chunks=4,
        video_tokens_per_chunk=2,
        action_tokens_per_chunk=1,
        window=window,
    )
    q, k, v, phi_q, phi_k = _attention_inputs(meta, seed=11)
    adapter = _nonidentity_adapter()
    full = _ar_chunked_linear_attn(
        q,
        k,
        v,
        phi_q,
        phi_k,
        meta,
        eps=1e-12,
        action_video_memory_adapter=adapter,
        layer_id=0,
    )
    cache = _cache_from_clean(meta, k, v, phi_k)

    for frame_id in range(meta.num_frames):
        selector = (meta.noise_ids == 0) & (meta.frame_ids == frame_id)
        if not bool(selector.any()):
            continue
        index = selector.nonzero(as_tuple=False).squeeze(-1)
        win = cache.windowed_state(
            0, frame_id, hi_inclusive=frame_id - 1, window=window
        )
        if frame_id % 2 == 1:
            video_win = cache.windowed_video_state(
                0, frame_id, hi_inclusive=frame_id - 1, window=window
            )
            if video_win is not None:
                assert win is not None
                delta_s, delta_z = adapter.forward_delta(0, *video_win)
                win = (win[0] + delta_s, win[1] + delta_z)
        got = ar_inference_attn(
            q[:, :, index],
            phi_q[:, :, index],
            win,
            k[:, :, index],
            phi_k[:, :, index],
            v[:, :, index],
            eps=1e-12,
        )
        torch.testing.assert_close(got, full[:, :, index], atol=1e-9, rtol=1e-7)


class _FakeBackbone(nn.Module):
    video_attention_mask_mode = "bidirectional"

    def __init__(self, *, heads=2, dim=3, layers=2):
        super().__init__()
        self._heads, self._dim, self._layers = heads, dim, layers
        self.action_dim = heads * dim

    @property
    def num_heads(self):
        return self._heads

    @property
    def head_dim(self):
        return self._dim

    @property
    def num_layers(self):
        return self._layers

    @property
    def attn_kernel(self):
        return "linear_relu"

    def pre_attn_at_layer(self, layer_id, state):  # noqa: ARG002
        phi = state.x.abs() + 1.0
        post = {"uses_linear_attn": True, "q_unrot": phi, "k_unrot": phi}
        return state.x, state.x, state.x, post

    def post_attn_at_layer(self, layer_id, state, out, post):  # noqa: ARG002
        state.x = out
        return state


class _FakeJointActionBackbone(_FakeBackbone):
    def pre_attn_at_layer(self, layer_id, state):  # noqa: ARG002
        x = state.payload.x_action
        phi = x.abs() + 1.0
        post = {"uses_linear_attn": True, "q_unrot": phi, "k_unrot": phi}
        return x, x, x, post

    def post_attn_at_layer(self, layer_id, state, out, post):  # noqa: ARG002
        state.payload.x_action = out
        return state


def _driver_pair(adapter):
    video = _FakeBackbone()
    action = _FakeBackbone()
    driver = SanaARMoTJointDriver(
        video,
        action,
        mot_checkpoint_mixed_attn=False,
        action_video_memory_adapter=adapter,
    )
    return video, action, driver


def _history_cache(*, layers=2, heads=2, dim=3):
    generator = torch.Generator().manual_seed(31)
    cache = ARLinearStateCache(num_layers=layers, window=8)
    for layer in range(layers):
        S = torch.randn(1, heads, dim, dim, generator=generator)
        z = torch.rand(1, heads, 1, dim, generator=generator) + 0.5
        cache.update(layer, 0, S, z, is_pred=False)
    return cache


def _clone_cache(cache):
    clone = ARLinearStateCache(cache.num_layers, cache.window)
    clone.restore_snapshot(cache.snapshot())
    return clone


def test_driver_adapter_is_action_query_only_and_clean_writer_is_canonical():
    generator = torch.Generator().manual_seed(41)
    initial = torch.randn(1, 2, 6, generator=generator)
    adapter = _nonidentity_adapter(layers=2, heads=2, dim=3)
    video_base, action_base, baseline = _driver_pair(None)
    video_adapted, action_adapted, treatment = _driver_pair(adapter)
    history = _history_cache()

    def run(driver, backbone, *, frame_id, store_clean):
        state = SimpleNamespace(x=initial.clone())
        cache = _clone_cache(history)
        driver.run_ar_chunk_through_backbone(
            backbone,
            state,
            cache,
            frame_id,
            store_clean=store_clean,
            is_pred=False,
        )
        return state.x, cache

    base_action, _ = run(baseline, action_base, frame_id=1, store_clean=False)
    adapted_action, _ = run(
        treatment, action_adapted, frame_id=1, store_clean=False
    )
    assert not torch.equal(adapted_action, base_action)

    base_video, _ = run(baseline, video_base, frame_id=2, store_clean=False)
    adapted_video, _ = run(
        treatment, video_adapted, frame_id=2, store_clean=False
    )
    assert torch.equal(adapted_video, base_video)

    base_clean, base_cache = run(
        baseline, action_base, frame_id=1, store_clean=True
    )
    adapted_clean, adapted_cache = run(
        treatment, action_adapted, frame_id=1, store_clean=True
    )
    assert torch.equal(adapted_clean, base_clean)
    for base_layer, adapted_layer in zip(
        base_cache.snapshot(), adapted_cache.snapshot()
    ):
        base_entry = next(entry for entry in base_layer if entry["frame_id"] == 1)
        adapted_entry = next(
            entry for entry in adapted_layer if entry["frame_id"] == 1
        )
        assert torch.equal(adapted_entry["S"], base_entry["S"])
        assert torch.equal(adapted_entry["z"], base_entry["z"])


def test_training_kernel_requires_layer_id_when_adapter_is_present():
    meta = build_ar_seq_meta(
        num_chunks=1,
        video_tokens_per_chunk=1,
        action_tokens_per_chunk=1,
        window=2,
    )
    inputs = _attention_inputs(meta)
    with pytest.raises(ValueError, match="requires layer_id"):
        _ar_chunked_linear_attn(
            *inputs,
            meta,
            action_video_memory_adapter=_nonidentity_adapter(),
        )


def test_gradient_checkpoint_backward_retains_adapter_descriptor():
    meta = build_ar_seq_meta(
        num_chunks=2,
        video_tokens_per_chunk=1,
        action_tokens_per_chunk=1,
        window=4,
    )
    generator = torch.Generator().manual_seed(71)
    video_input = torch.randn(1, 4, 6, generator=generator)
    action_input = torch.randn(1, 4, 6, generator=generator)

    def run(use_checkpointing):
        video = _FakeBackbone(heads=2, dim=3, layers=1)
        action = _FakeJointActionBackbone(heads=2, dim=3, layers=1)
        adapter = _nonidentity_adapter(layers=1, heads=2, dim=3)
        driver = SanaARMoTJointDriver(
            video,
            action,
            mot_checkpoint_mixed_attn=True,
            action_video_memory_adapter=adapter,
        )
        vstate = SimpleNamespace(x=video_input.clone().requires_grad_(True))
        payload = SimpleNamespace(
            x_action=action_input.clone().requires_grad_(True)
        )
        astate = SimpleNamespace(payload=payload)
        driver.run_ar_joint_loop(
            vstate,
            astate,
            ar_meta=meta,
            use_gradient_checkpointing=use_checkpointing,
        )
        loss = vstate.x.square().sum() + payload.x_action.square().sum()
        loss.backward()
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in adapter.named_parameters()
        }
        return vstate.x.detach(), payload.x_action.detach(), grads

    eager_video, eager_action, eager_grads = run(False)
    ckpt_video, ckpt_action, ckpt_grads = run(True)
    assert torch.equal(ckpt_video, eager_video)
    assert torch.equal(ckpt_action, eager_action)
    for name in eager_grads:
        torch.testing.assert_close(ckpt_grads[name], eager_grads[name])


def test_architecture_registers_adapter_as_top_level_driver_dependency():
    architecture = DualSystemARArchitecture(cfg=None)
    architecture.video_backbone = _FakeBackbone(heads=2, dim=8, layers=2)
    architecture.action_backbone = _FakeBackbone(heads=2, dim=8, layers=2)
    architecture._mot_driver_kwargs = {
        "mot_checkpoint_mixed_attn": False,
        "attention_mask_mode": "joint",
        "video_attention_mask_mode": "first_frame_causal",
    }
    architecture._action_video_memory_adapter_cfg = {
        "enabled": True,
        "rank": 8,
        "z_scale_min": 0.5,
        "z_scale_max": 2.0,
        "init_seed": 19,
    }
    architecture.action_video_memory_adapter = None

    driver = architecture.build_mot_driver()
    adapter = architecture.action_video_memory_adapter
    assert isinstance(adapter, ActionVideoMemoryAdapter)
    assert driver.action_video_memory_adapter is adapter
    adapter_keys = [
        key
        for key in architecture.state_dict()
        if key.startswith("action_video_memory_adapter.")
    ]
    assert len(adapter_keys) == 5


def test_disabled_architecture_has_no_adapter_checkpoint_keys():
    architecture = DualSystemARArchitecture(cfg=None)
    assert architecture.action_video_memory_adapter is None
    assert not any(
        key.startswith("action_video_memory_adapter.")
        for key in architecture.state_dict()
    )
