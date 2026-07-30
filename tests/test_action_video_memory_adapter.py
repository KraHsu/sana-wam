from __future__ import annotations

import pytest
import torch

from sana_wam.model.action_video_memory_adapter import ActionVideoMemoryAdapter


def _adapter(**overrides) -> ActionVideoMemoryAdapter:
    kwargs = {
        "num_layers": 2,
        "num_heads": 3,
        "head_dim": 5,
        "rank": 2,
        "init_seed": 17,
    }
    kwargs.update(overrides)
    return ActionVideoMemoryAdapter(**kwargs)


def _state(*, batch=2, heads=3, dim=5, dtype=torch.float64):
    generator = torch.Generator().manual_seed(9)
    S = torch.randn(batch, heads, dim, dim, generator=generator, dtype=dtype)
    z = torch.rand(batch, heads, 1, dim, generator=generator, dtype=dtype)
    return S, z


def test_identity_init_is_exact_and_deterministic():
    left, right = _adapter(), _adapter()
    assert all(
        torch.equal(a, b)
        for a, b in zip(left.state_dict().values(), right.state_dict().values())
    )
    S, z = _state()
    delta_s, delta_z = left.forward_delta(1, S, z)
    assert torch.count_nonzero(delta_s) == 0
    assert torch.count_nonzero(delta_z) == 0
    adapted_s, adapted_z = left(1, S, z)
    assert torch.equal(adapted_s, S)
    assert torch.equal(adapted_z, z)


def test_state_projection_matches_explicit_token_projection():
    adapter = _adapter(num_layers=1)
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        adapter.k_right.normal_(generator=generator, std=0.08)
        adapter.v_right.normal_(generator=generator, std=0.08)
        adapter.z_logits.uniform_(-0.7, 0.7, generator=generator)

    batch, heads, tokens, dim = 2, 3, 7, 5
    K = torch.randn(batch, heads, tokens, dim, generator=generator)
    V = torch.randn(batch, heads, tokens, dim, generator=generator)
    phi_k = torch.rand(batch, heads, tokens, dim, generator=generator)
    S = V.transpose(-1, -2) @ K
    z = phi_k.sum(dim=-2, keepdim=True)

    eye = torch.eye(dim).expand(heads, dim, dim)
    P_k = eye + adapter.k_left[0] @ adapter.k_right[0].transpose(-1, -2)
    P_v = eye + adapter.v_left[0] @ adapter.v_right[0].transpose(-1, -2)
    projected_K = torch.einsum("bhnd,hdk->bhnk", K, P_k)
    projected_V = torch.einsum("bhnd,hdk->bhnk", V, P_v)
    expected_s = projected_V.transpose(-1, -2) @ projected_K
    scale = adapter.normalizer_scale(0, dtype=z.dtype, device=z.device)
    expected_z = z * scale.unsqueeze(0).unsqueeze(2)

    actual_s, actual_z = adapter(0, S, z)
    torch.testing.assert_close(actual_s, expected_s, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_z, expected_z)


def test_normalizer_scale_is_positive_bounded_and_identity_at_zero():
    adapter = _adapter(num_layers=1)
    identity = adapter.normalizer_scale(
        0, dtype=torch.float64, device=torch.device("cpu")
    )
    assert torch.equal(identity, torch.ones_like(identity))
    with torch.no_grad():
        adapter.z_logits[0, 0].fill_(-100.0)
        adapter.z_logits[0, 1].fill_(100.0)
    scale = adapter.normalizer_scale(
        0, dtype=torch.float64, device=torch.device("cpu")
    )
    assert bool((scale >= 0.5).all())
    assert bool((scale <= 2.0).all())
    torch.testing.assert_close(scale[0], torch.full_like(scale[0], 0.5))
    torch.testing.assert_close(scale[1], torch.full_like(scale[1], 2.0))


def test_identity_init_has_live_finite_gradients():
    adapter = _adapter(num_layers=1)
    S, z = _state(dtype=torch.float32)
    delta_s, delta_z = adapter.forward_delta(0, S, z)
    loss = delta_s.sum() + delta_z.sum()
    loss.backward()
    for parameter in adapter.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(adapter.k_right.grad) > 0
    assert torch.count_nonzero(adapter.v_right.grad) > 0
    assert torch.count_nonzero(adapter.z_logits.grad) > 0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"rank": 6}, "cannot exceed"),
        ({"z_scale_min": 0.0}, "0 < min"),
        ({"z_scale_max": 1.0}, "0 < min"),
    ],
)
def test_constructor_rejects_invalid_contract(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _adapter(head_dim=5, **kwargs)


def test_input_contract_fails_closed():
    adapter = _adapter(num_layers=1)
    S, z = _state()
    with pytest.raises(IndexError, match="outside"):
        adapter.forward_delta(1, S, z)
    with pytest.raises(ValueError, match="S_video"):
        adapter.forward_delta(0, S[:, :, :-1], z)
    with pytest.raises(ValueError, match="same dtype"):
        adapter.forward_delta(0, S, z.float())
    with pytest.raises(ValueError, match="same device"):
        # Meta tensors exercise the device guard without requiring CUDA.
        adapter.forward_delta(0, S, torch.empty_like(z, device="meta"))


def test_parameter_count_matches_frozen_formula():
    layers, heads, dim, rank = 20, 20, 112, 8
    adapter = ActionVideoMemoryAdapter(
        num_layers=layers,
        num_heads=heads,
        head_dim=dim,
        rank=rank,
    )
    expected = 4 * layers * heads * dim * rank + layers * heads * dim
    assert sum(parameter.numel() for parameter in adapter.parameters()) == expected
    assert expected == 1_478_400
