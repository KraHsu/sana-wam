#!/usr/bin/env python
"""torchrun entry point for SANA block-AR training.

Usage (single GPU):
    python scripts/train.py --config configs/train_ar_sana.yaml [key=value ...]

Multi-GPU:
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/train_ar_sana.yaml dataloader.train_tasks=[adjust_bottle,lift_pot]

CLI overrides use OmegaConf dotlist syntax, e.g.
    model.video_backbone.model_path=/DATA/share/SANA-Video_2B_480p training.max_steps=100
"""

from __future__ import annotations

# NVLS/NVLink-SHARP eager-connect hangs for >2 GPUs on this H200 box — must be
# set BEFORE NCCL initialises. Harmless on single GPU / other hardware.
import os

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

import argparse
import logging
import sys
from pathlib import Path

import torch.distributed as dist
from omegaconf import OmegaConf

# Make ``sana_wam`` and the SANA submodule importable without an install.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party" / "Sana"))

from sana_wam.train.trainer import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, overrides = ap.parse_known_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank() if dist.is_initialized() else 0
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    trainer = Trainer(cfg)
    out = trainer.train()
    if rank == 0:
        logging.getLogger(__name__).info("Training done. Checkpoints in %s", out)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
