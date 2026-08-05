#!/usr/bin/env python
"""Precompute VAE latents for the RoboTwin dataset (opt-in training speedup).

Mirrors upstream Sana-wm's latent-cache workflow (``load_vae_feat`` /
``SanaWMZipLatentDataset``): encode each training clip once with the Wan VAE and
store ``(C, T_lat, H_lat, W_lat)`` to a bucketed safetensors cache. Training then
reads the cache (``dataloader.vae_cache_dir=...``) and skips the per-step VAE
encode — the throughput bottleneck in ``preprocess_input``.

Usage (single GPU):
    python scripts/precompute_vae_latents.py --config configs/train_ar_sana.yaml \
        --cache-dir /DATA/cache/robotwin_latents \
        model.video_backbone.model_path=/DATA/share/SANA-Video_2B_480p

The cache key encodes (episode_path, start_frame, geometry); a cache built at one
resolution/stride/layout will simply miss for a different geometry (safe — falls
back to live encode), so re-run this script after changing video geometry.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party" / "Sana"))

from sana_wam.train.cach_stage0_guard import (  # noqa: E402
    reject_cach_stage0_base_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompute_vae_latents")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True, help="Output dir for the latent cache.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="Cap on #clips (debug).")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    reject_cach_stage0_base_config(
        cfg, entrypoint="scripts/precompute_vae_latents.py"
    )
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    # Keep torch/model imports after the unmerged Stage-0 marker denial.
    import torch
    from sana_wam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset
    from sana_wam.dataloader.transforms.vae_latent_cache import latent_cache_key, save_latent
    from sana_wam.model.video_backbone.sana.adapter import SanaVideoBackbone, _pil_video_to_tensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    os.makedirs(args.cache_dir, exist_ok=True)

    # Build the dataset WITHOUT a cache (we are creating it). Force live frames.
    dl_cfg = OmegaConf.merge(cfg.dataloader, OmegaConf.create({"vae_cache_dir": None, "split": args.split}))
    dataset = MultiTaskRoboTwinDataset.from_config(dl_cfg, split=args.split)
    geometry = dataset._sub_datasets[0].latent_cache_geometry()
    logger.info("Latent geometry fingerprint: %s", geometry)

    # Load only the VAE side of the backbone.
    vb = SanaVideoBackbone.from_config(cfg.model.video_backbone, device=device, dtype=dtype)
    if vb._pipe.vae is None:
        raise RuntimeError("VAE not loaded; set model.video_backbone.model_path to a bundle with vae/Wan2.1_VAE.pth")

    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    logger.info("Encoding %d clips → %s", n, args.cache_dir)

    written = skipped = 0
    for i in range(n):
        sample = dataset[i]
        ep, start = sample.get("episode_path"), sample.get("start_frame")
        if ep is None or start is None:
            continue
        sha = latent_cache_key(ep, int(start), geometry)
        from sana_wam.dataloader.transforms.vae_latent_cache import cache_path_for_sha

        if os.path.isfile(cache_path_for_sha(args.cache_dir, sha)):
            skipped += 1
            continue
        video = _pil_video_to_tensor([sample["video"]]).to(device=device, dtype=dtype)
        with torch.no_grad():
            latent = vb._pipe.vae.encode([video[0]], device=device, tiled=True)
        save_latent(args.cache_dir, sha, latent.to(torch.float32))
        written += 1
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d (written=%d skipped=%d)", i + 1, n, written, skipped)

    logger.info("Done: written=%d skipped=%d total=%d", written, skipped, n)


if __name__ == "__main__":
    main()
