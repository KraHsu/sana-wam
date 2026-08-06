#!/usr/bin/env python
"""Dedicated non-formal launcher for a materialized LIBERO training config."""

from __future__ import annotations

import os

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party" / "Sana"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error("LIBERO launcher forbids unpinned dotlist overrides")

    cfg = OmegaConf.load(args.config)
    from sana_wam.train.libero_contract import validate_libero_training_config

    validate_libero_training_config(cfg, require_materialized_stats=True)
    output_dir = str(OmegaConf.select(cfg, "training.output_dir", default=""))
    if not output_dir or "UNAUTHORIZED_TEMPLATE" in output_dir:
        raise ValueError(
            "materialize a unique LIBERO training.output_dir before launch"
        )

    import torch.distributed as dist

    from sana_wam.train.trainer import Trainer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    trainer = Trainer(cfg)
    validate_libero_training_config(cfg, dataset=trainer.dataset)
    output = trainer.train()
    if rank == 0:
        logging.getLogger(__name__).info("Training done. Checkpoints in %s", output)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
