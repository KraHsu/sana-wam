#!/usr/bin/env python
"""torchrun entry point for SANA block-AR training.

Usage (single GPU):
    python scripts/train.py --config configs/train_ar_sana.yaml [key=value ...]

Formal Phase-6 accepts no dotlist overrides and requires an immutable launch ticket.

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

# GDN scan helpers are wrapped in ``@torch.compile`` (third_party/Sana
# sana_gdn_blocks.py). torch 2.7.1's inductor calls ``triton.compiler.triton_key``,
# which triton 3.5.1 (required by the GDN bare-@triton.jit kernels) removed —
# an ABI break. The kernels themselves don't need inductor, so disable the
# torch.compile wrapper. No-op for the linear_relu path. Must precede any
# ``diffusion.*`` import.
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Make ``sana_wam`` and the SANA submodule importable without an install.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party" / "Sana"))



def _is_formal_phase6(cfg) -> bool:
    training = cfg.get("training", {})
    return any(
        training.get(key, None) not in (None, "")
        for key in (
            "phase6_arm",
            "phase6_registry",
            "phase6_plan_artifact",
            "phase6_dataset_contract_artifact",
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase6-launch-ticket")
    args, overrides = ap.parse_known_args()

    base_cfg = OmegaConf.load(args.config)
    is_formal_phase6 = _is_formal_phase6(base_cfg)
    launch_context = None
    if is_formal_phase6:
        if overrides:
            raise RuntimeError("formal Phase-6 forbids every CLI config override")
        if not args.phase6_launch_ticket:
            raise RuntimeError("formal Phase-6 requires --phase6-launch-ticket")
        from sana_wam.train.phase6_launch_manifest import (
            authorize_formal_phase6_invocation,
        )

        launch_context = authorize_formal_phase6_invocation(
            ticket_path=args.phase6_launch_ticket,
            config_path=args.config,
            argv=[sys.executable, *sys.argv],
            environment=os.environ,
            working_directory=os.getcwd(),
        )
    elif args.phase6_launch_ticket:
        raise RuntimeError("Phase-6 launch tickets are forbidden for legacy training")
    cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(overrides))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if is_formal_phase6 and world_size != 1:
        raise RuntimeError(
            "formal Phase-6 preflight requires WORLD_SIZE=1 before any torch/NCCL import"
        )

    # Formal Phase-6 rejects an invalid world size before importing torch or
    # Trainer. Trainer then runs the full preflight as its first operation.
    import torch.distributed as dist
    from sana_wam.train.trainer import Trainer

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank() if dist.is_initialized() else 0
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    trainer = Trainer(cfg, launch_context=launch_context)
    out = trainer.train()
    if rank == 0:
        logging.getLogger(__name__).info("Training done. Checkpoints in %s", out)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
