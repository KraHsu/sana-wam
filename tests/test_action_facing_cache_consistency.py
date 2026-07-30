from __future__ import annotations

import pytest
import torch

from sana_wam.model.action_facing_cache_consistency import (
    action_facing_cache_consistency_loss,
    normalized_action_facing_numerator_loss,
    validate_afcc_reference,
    validate_afcc_weight,
)
from sana_wam.model.ar.sana_ar_linear_attn import (
    _ar_chunked_linear_attn,
    build_ar_seq_meta,
)


def _inputs(*, batch: int, heads: int, tokens: int, dim: int):
    generator = torch.Generator().manual_seed(2607)
    q = torch.randn(batch, heads, tokens, dim, generator=generator)
    k = torch.randn(batch, heads, tokens, dim, generator=generator)
    value = torch.randn(batch, heads, tokens, dim, generator=generator)
    phi_q = torch.rand(batch, heads, tokens, dim, generator=generator) + 0.2
    phi_k = torch.rand(batch, heads, tokens, dim, generator=generator) + 0.2
    return q, k, value, phi_q, phi_k


def test_kernel_capture_is_video_only_production_numerator():
    chunks, video_width, action_width = 3, 2, 3
    tokens = 2 * chunks * (video_width + action_width)
    values = _inputs(batch=2, heads=2, tokens=tokens, dim=4)
    meta = build_ar_seq_meta(
        num_chunks=chunks,
        video_tokens_per_chunk=video_width,
        action_tokens_per_chunk=action_width,
        window=72,
    )

    baseline = _ar_chunked_linear_attn(*values, meta)
    captured_output, captured = _ar_chunked_linear_attn(
        *values, meta, return_action_video_numerators=True
    )
    assert torch.equal(captured_output, baseline)
    assert captured.shape == (2, chunks, 2, action_width, 4)

    q, k, value, _, _ = values
    video_clean_start = chunks * video_width
    action_noisy_start = 2 * chunks * video_width
    per_frame_state = torch.zeros(
        2 * chunks,
        q.shape[0],
        q.shape[1],
        q.shape[3],
        q.shape[3],
    )
    for chunk in range(chunks):
        video_indices = torch.arange(
            video_clean_start + chunk * video_width,
            video_clean_start + (chunk + 1) * video_width,
        )
        per_frame_state[2 * chunk] = (
            value[:, :, video_indices].transpose(-1, -2)
            @ k[:, :, video_indices]
        )
    prefix_state = torch.zeros(
        2 * chunks + 1,
        q.shape[0],
        q.shape[1],
        q.shape[3],
        q.shape[3],
    )
    prefix_state[1:] = torch.cumsum(per_frame_state, dim=0)
    for chunk in range(chunks):
        action_indices = torch.arange(
            action_noisy_start + chunk * action_width,
            action_noisy_start + (chunk + 1) * action_width,
        )
        video_state = prefix_state[2 * chunk + 1] - prefix_state[0]
        expected = q[:, :, action_indices] @ video_state.transpose(-1, -2)
        torch.testing.assert_close(captured[:, chunk], expected, atol=0, rtol=0)


def test_afcc_loss_is_zero_for_self_and_reference_is_detached():
    student = torch.randn(2, 3, 4, 5, requires_grad=True)
    reference = student.detach().clone()
    valid = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )
    loss = normalized_action_facing_numerator_loss(student, reference, valid)
    assert loss.dtype == torch.float32
    assert loss.item() == 0.0
    loss.backward()
    assert reference.grad is None
    assert torch.count_nonzero(student.grad) == 0


def test_afcc_full_loss_excludes_padded_and_empty_chunks():
    student = torch.zeros(1, 2, 3, 1, 4, 2, requires_grad=True)
    reference = torch.ones_like(student, requires_grad=False)
    valid = torch.tensor([[[True, True, False, False], [True] * 4, [False] * 4]])
    baseline = action_facing_cache_consistency_loss(student, reference, valid)

    changed = student.detach().clone()
    changed[:, :, 0, :, 2:] = 1000.0
    changed[:, :, 2] = 1000.0
    changed.requires_grad_()
    observed = action_facing_cache_consistency_loss(changed, reference, valid)
    torch.testing.assert_close(observed, baseline, atol=0, rtol=0)
    observed.backward()
    assert torch.count_nonzero(changed.grad[:, :, 0, :, 2:]) == 0
    assert torch.count_nonzero(changed.grad[:, :, 2]) == 0
    assert torch.count_nonzero(changed.grad[:, :, :2]) > 0


def test_afcc_reference_contract_requires_complete_detached_tensor():
    reference = torch.zeros(1, 2, 3, 4, 5, 6)
    observed = validate_afcc_reference(
        reference,
        batch_size=1,
        num_layers=2,
        num_chunks=3,
        num_heads=4,
        action_tokens_per_chunk=5,
        head_dim=6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(observed, reference)
    assert observed.requires_grad is False
    with pytest.raises(ValueError, match="shape"):
        validate_afcc_reference(
            reference[:, :-1],
            batch_size=1,
            num_layers=2,
            num_chunks=3,
            num_heads=4,
            action_tokens_per_chunk=5,
            head_dim=6,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="detached"):
        validate_afcc_reference(
            reference.requires_grad_(),
            batch_size=1,
            num_layers=2,
            num_chunks=3,
            num_heads=4,
            action_tokens_per_chunk=5,
            head_dim=6,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


@pytest.mark.parametrize("value", [-1, 0.5, 2, True, float("nan")])
def test_afcc_weight_rejects_non_binary_values(value):
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        validate_afcc_weight(value)
