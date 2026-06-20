"""Opt-in VAE latent cache for the RoboTwin dataset.

Mirrors upstream Sana-wm's ``load_vae_feat`` / ``SanaWMZipLatentDataset``
pattern (``../Sana/diffusion/data/datasets/video/sana_wm_zip_latent_data.py``):
precompute the VAE-encoded video latents once, then read them at training time
so the per-step VAE encode (the throughput bottleneck in ``preprocess_input``)
is skipped.

This is the latent analogue of :mod:`text_embedding_cache`. A sample's clip is
keyed by ``(episode_path, start_frame, video-geometry)`` — every input that
determines the encoded pixels — so a cache entry can only ever be reused for a
byte-identical clip. The geometry component guards against silently reusing a
cache built at a different resolution / stride / camera layout.

Cache layout (``<cache_dir>/``):

    <sha[:2]>/<sha>.safetensors   # key "latent" -> (C, T_lat, H_lat, W_lat)

Misses return the sample untouched (live encode stays the fallback), so a
partial cache degrades gracefully rather than erroring.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

from sana_wam.dataloader.transforms.base import ModalityTransform

BUCKET_PREFIX_LEN = 2

_PRECOMPUTE_HINT = (
    "Build the latent cache first with scripts/precompute_vae_latents.py, "
    "or unset dataloader.vae_cache_dir to encode live."
)


def latent_cache_key(episode_path: str, start_frame: int, geometry: str) -> str:
    """Deterministic SHA over the clip identity (path + window + geometry)."""
    payload = f"{os.path.abspath(episode_path)}|{int(start_frame)}|{geometry}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path_for_sha(cache_dir: str, sha: str) -> str:
    return os.path.join(cache_dir, sha[:BUCKET_PREFIX_LEN], f"{sha}.safetensors")


class VAELatentCacheTransform(ModalityTransform):
    """Attach a cached ``pre_encoded_latents`` tensor to a sample when present.

    Args:
        cache_dir: Directory holding ``<sha[:2]>/<sha>.safetensors`` entries.
        geometry: A string fingerprint of the video geometry that produced the
            latents (resolution, stride, frame count, multiview layout). Two
            datasets with different geometry must pass different fingerprints so
            their caches never collide.
    """

    def __init__(self, cache_dir: str, geometry: str):
        super().__init__()
        if not os.path.isdir(cache_dir):
            raise NotADirectoryError(
                f"vae_cache_dir={cache_dir!r} is not a directory.\n{_PRECOMPUTE_HINT}"
            )
        self.cache_dir = cache_dir
        self.geometry = str(geometry)

    def key_for(self, sample: dict) -> Optional[str]:
        ep = sample.get("episode_path")
        start = sample.get("start_frame")
        if ep is None or start is None:
            return None
        return latent_cache_key(ep, int(start), self.geometry)

    def apply(self, data: dict) -> dict:
        sha = self.key_for(data)
        if sha is None:
            return data
        path = cache_path_for_sha(self.cache_dir, sha)
        if not os.path.isfile(path):
            return data  # miss → live encode fallback
        tensors = load_file(path)
        latent = tensors.get("latent")
        if latent is not None:
            data["pre_encoded_latents"] = latent
        return data


def save_latent(cache_dir: str, sha: str, latent: torch.Tensor) -> str:
    """Write one ``(C, T, H, W)`` latent to the bucketed cache. Returns the path."""
    if latent.dim() == 5 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.dim() != 4:
        raise ValueError(f"latent must be (C, T, H, W) or (1, C, T, H, W); got {tuple(latent.shape)}")
    path = cache_path_for_sha(cache_dir, sha)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({"latent": latent.contiguous().cpu()}, path)
    return path
