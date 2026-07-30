from __future__ import annotations

from einops import rearrange
import torch

from sana_wam.model.ar.attention_diagnostics import ARAttentionCapture
from sana_wam.model.ar.sana_ar_linear_attn import (
    _ar_chunked_linear_attn,
    build_ar_seq_meta,
)


class _FakeDriver:
    def __init__(self):
        self.num_heads = 2
        self.eps = 1e-8
        self._ar_meta = build_ar_seq_meta(
            num_chunks=2,
            video_tokens_per_chunk=2,
            action_tokens_per_chunk=1,
            window=10,
        )

    def _step_impl(self, layer_id, value):
        return layer_id + value

    def _mixed_attention(
        self,
        q_cat,
        k_cat,
        v_cat,
        attn_mask,
        *,
        phi_q,
        phi_k,
        **_kwargs,
    ):
        assert attn_mask is None
        q = rearrange(q_cat, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k_cat, "b s (h d) -> b h s d", h=self.num_heads)
        value = rearrange(v_cat, "b s (h d) -> b h s d", h=self.num_heads)
        pq = rearrange(phi_q, "b s (h d) -> b h s d", h=self.num_heads)
        pk = rearrange(phi_k, "b s (h d) -> b h s d", h=self.num_heads)
        output = _ar_chunked_linear_attn(
            q, k, value, pq, pk, self._ar_meta, eps=self.eps
        )
        return rearrange(output, "b h s d -> b s (h d)")


def test_capture_records_components_without_changing_output():
    torch.manual_seed(7)
    driver = _FakeDriver()
    tokens = len(driver._ar_meta.frame_ids)
    q = torch.rand(1, tokens, 8)
    k = torch.rand(1, tokens, 8)
    value = torch.randn(1, tokens, 8)
    phi_q = torch.relu(q)
    phi_k = torch.relu(k)
    expected = driver._mixed_attention(
        q, k, value, None, phi_q=phi_q, phi_k=phi_k
    )

    original_mixed = driver._mixed_attention
    original_step = driver._step_impl
    with ARAttentionCapture(driver, capture_video_branches=False) as capture:
        assert driver._step_impl(3, 4) == 7
        actual = driver._mixed_attention(
            q, k, value, None, phi_q=phi_q, phi_k=phi_k
        )
        assert torch.equal(actual, expected)
        assert len(capture.records) == 2 * len(capture.COMPONENTS)
        assert {row["component"] for row in capture.records} == set(capture.COMPONENTS)
        assert max(row["reconstruction_rmse"] for row in capture.records) < 1e-6

    assert driver._mixed_attention == original_mixed
    assert driver._step_impl == original_step
    assert driver._ar_attention_capture_active is False
