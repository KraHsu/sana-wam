"""VAE latent cache: round-trip + prepare_inputs collation.

Pins the opt-in latent cache (upstream Sana-wm ``load_vae_feat`` analogue):

1. ``save_latent`` → ``VAELatentCacheTransform.apply`` round-trips a latent and
   attaches it as ``pre_encoded_latents`` only for the matching clip key.
2. A geometry-mismatched / missing entry is a clean miss (sample untouched), so
   live encoding stays the fallback.
3. The collation contract that ``base.prepare_inputs`` relies on: per-sample
   ``(C, T, H, W)`` latents stack to a ``(B, C, T, H, W)`` ``input_latents`` and a
   mixed batch (some cached, some not) is rejected.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sana_wam.dataloader.transforms.vae_latent_cache import (
    VAELatentCacheTransform,
    cache_path_for_sha,
    latent_cache_key,
    save_latent,
)


def _sample(ep="ep0.hdf5", start=4):
    return {"episode_path": ep, "start_frame": start, "video": ["frame"]}


def test_save_then_load_attaches_latent(tmp_path):
    geom = "nf=49;vs=4;h=384;w=320"
    latent = torch.randn(16, 4, 12, 10)
    sha = latent_cache_key("ep0.hdf5", 4, geom)
    save_latent(str(tmp_path), sha, latent)

    t = VAELatentCacheTransform(cache_dir=str(tmp_path), geometry=geom)
    out = t.apply(_sample())
    assert "pre_encoded_latents" in out
    torch.testing.assert_close(out["pre_encoded_latents"], latent)


def test_geometry_mismatch_is_a_miss(tmp_path):
    latent = torch.randn(16, 4, 12, 10)
    sha = latent_cache_key("ep0.hdf5", 4, "nf=49;vs=4;h=384;w=320")
    save_latent(str(tmp_path), sha, latent)

    # Different geometry → different key → miss, sample untouched.
    t = VAELatentCacheTransform(cache_dir=str(tmp_path), geometry="nf=33;vs=1;h=480;w=832")
    out = t.apply(_sample())
    assert "pre_encoded_latents" not in out


def test_missing_episode_fields_is_a_miss(tmp_path):
    t = VAELatentCacheTransform(cache_dir=str(tmp_path), geometry="g")
    out = t.apply({"video": ["f"]})  # no episode_path / start_frame
    assert "pre_encoded_latents" not in out


def test_save_latent_squeezes_batch_dim(tmp_path):
    latent_5d = torch.randn(1, 16, 4, 8, 8)
    sha = "deadbeef" + "0" * 56
    path = save_latent(str(tmp_path), sha, latent_5d)
    assert path == cache_path_for_sha(str(tmp_path), sha)
    from safetensors.torch import load_file

    loaded = load_file(path)["latent"]
    assert loaded.shape == (16, 4, 8, 8)


def test_nonexistent_cache_dir_raises():
    with pytest.raises(NotADirectoryError):
        VAELatentCacheTransform(cache_dir="/no/such/dir", geometry="g")


# --- collation contract used by base.prepare_inputs -------------------------

def _collate_latents(samples, dtype=torch.float32, device="cpu"):
    """The exact stacking/validation logic prepare_inputs applies."""
    all_lat = [s.get("pre_encoded_latents") for s in samples]
    flags = [t is not None for t in all_lat]
    if not any(flags):
        return None
    if not all(flags):
        raise ValueError("Mixed pre_encoded_latents in batch")
    tensors = []
    for t in all_lat:
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        if t.dim() == 5 and t.shape[0] == 1:
            t = t[0]
        if t.dim() != 4:
            raise ValueError("bad shape")
        tensors.append(t)
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) > 1:
        raise ValueError("inconsistent shapes")
    return torch.stack([t.to(dtype=dtype, device=device) for t in tensors], dim=0)


def test_collation_stacks_uniform_latents():
    samples = [{"pre_encoded_latents": torch.randn(16, 4, 8, 8)} for _ in range(3)]
    out = _collate_latents(samples)
    assert out.shape == (3, 16, 4, 8, 8)


def test_collation_rejects_mixed_batch():
    samples = [{"pre_encoded_latents": torch.randn(16, 4, 8, 8)}, {}]
    with pytest.raises(ValueError, match="Mixed"):
        _collate_latents(samples)


def test_collation_rejects_inconsistent_shapes():
    samples = [
        {"pre_encoded_latents": torch.randn(16, 4, 8, 8)},
        {"pre_encoded_latents": torch.randn(16, 4, 6, 8)},
    ]
    with pytest.raises(ValueError, match="inconsistent"):
        _collate_latents(samples)


def test_collation_none_when_no_cache():
    assert _collate_latents([{}, {}]) is None
