"""Regression: AR rollout must cache the executed action as confirmed history.

The training kernel puts the action ``clean`` copy (frame ``2c+1``) in every
noisy query's visible set (``noise2clean``, ``fj < fi``), so the world model is
trained to condition future video/action on past actions. An earlier rollout
implementation denoised the action chunk but **never wrote it into the
linear-state cache** — at inference the model became blind to executed actions, a
silent train/inference distribution shift that the behavioural rollout tests
(obs→action, window isolation) could not catch.

This test pins the fix at the exact seam: ``DualSystemARArchitecture
._denoise_action_chunk`` must ingest the denoised chunk as a **confirmed**
(``is_pred=False``) clean state at its modality-parity frame id, so it (a)
survives ``clear_pred`` (unlike the predicted video chunk, which a real obs later
replaces) and (b) is visible to in-window future frames but evicted past the
window.

Pure CPU — fake linear backbones implementing only the
``run_ar_chunk_through_backbone`` contract; no SANA dependency, so a SANA
pin-bump cannot mask a regression here.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from sana_wam.model.architecture import DualSystemARArchitecture
from sana_wam.model.ar.sana_ar_inference import ARLinearStateCache


def _phi(t: torch.Tensor) -> torch.Tensor:
    """Strictly-positive un-rotated ReLU-track stand-in.

    The real SANA kernel uses ``relu(q)/relu(k)``; here we use ``abs(x)+1`` so the
    linear-attention denominator is always ``> 0`` (a degenerate fake with whole
    relu'd heads zeroed would divide by ~eps and produce NaNs unrelated to the
    behaviour under test).
    """
    return t.abs() + 1.0


class _FakeLinearBackbone(nn.Module):
    """Minimal linear_relu backbone honouring the AR chunk-pass contract.

    Exposes the head metadata the driver needs plus the action-side
    ``prepare_state`` / ``extract_prediction`` and the per-layer
    ``pre_attn_at_layer`` / ``post_attn_at_layer`` hooks that
    ``run_ar_chunk_through_backbone`` drives. ``action_dim == num_heads *
    head_dim`` so q/k/v are just the (relu-tracked) tokens — enough to populate a
    real ``(S, z)`` clean state in the cache.
    """

    video_attention_mask_mode = "bidirectional"

    def __init__(self, num_heads: int, head_dim: int, num_layers: int):
        super().__init__()
        self._nh, self._hd, self._nl = num_heads, head_dim, num_layers
        self.action_dim = num_heads * head_dim
        self.dim = num_heads * head_dim

    @property
    def num_heads(self) -> int:
        return self._nh

    @property
    def head_dim(self) -> int:
        return self._hd

    @property
    def num_layers(self) -> int:
        return self._nl

    @property
    def attn_kernel(self) -> str:
        return "linear_relu"

    # --- action-backbone surface used by _denoise_action_chunk ---
    def prepare_state(self, x, timestep, *, context=None, context_mask=None,
                      token_timesteps=None, frame_ids=None, rope_positions=None):
        return SimpleNamespace(x=x)

    def extract_prediction(self, state):
        return state.x

    # --- per-layer hooks used by run_ar_chunk_through_backbone ---
    def pre_attn_at_layer(self, layer_id, state):
        x = state.x
        post = {"uses_linear_attn": True, "q_unrot": _phi(x), "k_unrot": _phi(x)}
        return x, x, x, post

    def post_attn_at_layer(self, layer_id, state, out, post):
        state.x = out
        return state


def _build_arch_with_fakes(num_heads=2, head_dim=4, num_layers=2):
    vb = _FakeLinearBackbone(num_heads, head_dim, num_layers)
    ab = _FakeLinearBackbone(num_heads, head_dim, num_layers)
    arch = DualSystemARArchitecture(cfg=None)
    arch.video_backbone, arch.action_backbone = vb, ab
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    arch._mot_driver_kwargs = {
        "attention_mask_mode": "joint",
        "video_attention_mask_mode": "first_frame_causal",
        "mot_checkpoint_mixed_attn": False,
    }
    driver = arch.build_mot_driver()
    return arch, driver


def test_denoise_action_chunk_caches_confirmed_history():
    arch, driver = _build_arch_with_fakes(num_layers=2)
    window = 4
    cache = ARLinearStateCache(num_layers=arch.action_backbone.num_layers, window=window)
    gen = torch.Generator().manual_seed(0)

    frame_id = 3  # action chunk 1 (modality parity 2c+1)
    action_tokens = 2
    a_sigmas = [1.0, 0.5, 0.0]  # 2 denoise steps
    a_ts = [1000.0, 500.0]

    x = arch._denoise_action_chunk(
        driver, cache, frame_id=frame_id, batch=1, action_tokens=action_tokens,
        a_sigmas=a_sigmas, a_ts=a_ts, context=None, context_mask=None, gen=gen,
    )
    assert x.shape == (1, action_tokens, arch.action_backbone.action_dim)
    assert torch.isfinite(x).all()

    # (1) Every layer cached a state AT this action frame — the core regression:
    #     before the fix the cache stayed empty after action denoising.
    for layer in range(cache.num_layers):
        assert any(e["frame_id"] == frame_id for e in cache._entries[layer]), (
            f"layer {layer} has no cached action state at frame {frame_id}"
        )

    # (2) It is CONFIRMED (is_pred=False) and visible to an in-window future query.
    in_window = cache.windowed_state(0, query_frame=frame_id + 1, hi_inclusive=frame_id, window=window)
    assert in_window is not None
    S_win, z_win = in_window
    assert torch.isfinite(S_win).all() and torch.isfinite(z_win).all()

    # (3) Confirmed history must survive clear_pred (a real obs only replaces the
    #     predicted VIDEO chunk; there is no real-action obs to swap in).
    cache.clear_pred()
    assert cache.windowed_state(0, query_frame=frame_id + 1, hi_inclusive=frame_id, window=window) is not None
    assert all(not e["is_pred"] for layer in cache._entries for e in layer)

    # (4) Sliding window still bounds visibility: far-future frame can't see it.
    far = frame_id + window + 1
    assert cache.windowed_state(0, query_frame=far, hi_inclusive=far - 1, window=window) is None


def test_cached_action_state_matches_direct_clean_state():
    """The cached ``(S, z)`` must equal the dual-track clean state of the final
    denoised action tokens (the same quantity the training kernel sums over the
    ``a_clean`` copy) — proving the cache holds the *right* history, not just
    *some* entry."""
    from einops import rearrange

    from sana_wam.model.ar.sana_ar_inference import clean_state_from_tokens

    arch, driver = _build_arch_with_fakes(num_layers=1)
    cache = ARLinearStateCache(num_layers=1, window=99)
    gen = torch.Generator().manual_seed(3)
    frame_id, action_tokens = 5, 3
    a_sigmas = [1.0, 0.0]  # single denoise step -> final x is exactly the predicted chunk
    a_ts = [1000.0]

    x = arch._denoise_action_chunk(
        driver, cache, frame_id=frame_id, batch=1, action_tokens=action_tokens,
        a_sigmas=a_sigmas, a_ts=a_ts, context=None, context_mask=None, gen=gen,
    )

    # Reconstruct the clean state from the final tokens (fake backbone: q=k=v=x,
    # phi=_phi(x)) and compare to the single cached entry.
    n = driver.num_heads
    tilde_k = rearrange(x, "b s (n d) -> b n s d", n=n)
    v_h = tilde_k
    phi_k = rearrange(_phi(x), "b s (n d) -> b n s d", n=n)
    S_exp, z_exp = clean_state_from_tokens(tilde_k, v_h, phi_k)

    entries = cache._entries[0]
    assert len(entries) == 1 and entries[0]["frame_id"] == frame_id
    torch.testing.assert_close(entries[0]["S"], S_exp)
    torch.testing.assert_close(entries[0]["z"], z_exp)
