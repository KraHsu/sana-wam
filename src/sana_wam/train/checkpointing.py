"""Checkpoint dir helpers — the train→deploy on-disk contract.

A checkpoint directory written by training must contain (so
``sana_wam.deploy.model_loader.load_from_checkpoint_dir`` can reconstruct):
  - ``config.yaml``                  — the (flat-enough) training config
  - ``checkpoint_step_<N>.safetensors`` — architecture ``state_dict``
  - ``action_stats.npy``             — for deploy-time (de)normalization

Verbatim-ported from openwam's train/utils/checkpointing.py (self-contained).
"""

from __future__ import annotations

import glob as _glob
import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)


def save_config(output_dir: str, cfg) -> None:
    """Serialize the OmegaConf config to ``config.yaml`` (write-once)."""
    config_path = os.path.join(output_dir, "config.yaml")
    if os.path.exists(config_path):
        return
    os.makedirs(output_dir, exist_ok=True)
    from omegaconf import OmegaConf

    OmegaConf.save(cfg, config_path)
    logger.info("Saved config to %s", config_path)


def save_action_stats(output_dir: str, dataset) -> None:
    """Copy the dataset's resolved action-stats ``.npy`` into the ckpt dir (write-once).

    Preserves the nested ``{"joint": ..., "eef": ...}`` schema so deploy can
    pick the sub-dict matching the saved config's ``action_mode``.
    """
    dst = os.path.join(output_dir, "action_stats.npy")
    if os.path.exists(dst):
        return
    src = getattr(dataset, "action_stats_path", None)
    if not src or not os.path.exists(src):
        logger.warning("[normalizer] no action_stats_path on dataset (src=%s); nothing copied.", src)
        return
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(src, dst)
    logger.info("[normalizer] Copied action stats:\n  src: %s\n  dst: %s", src, dst)


def manage_checkpoints(output_dir: str, keep_last_k: int) -> None:
    """Keep only the most recent ``keep_last_k`` ``checkpoint_step_*`` files."""
    files = _glob.glob(os.path.join(output_dir, "checkpoint_step_*"))

    def _step_num(path):
        m = re.search(r"checkpoint_step_(\d+)", path)
        return int(m.group(1)) if m else 0

    files.sort(key=_step_num)
    while len(files) > keep_last_k:
        old = files.pop(0)
        if os.path.isfile(old):
            os.remove(old)
            logger.info("Removed old checkpoint: %s", old)


__all__ = ["save_config", "save_action_stats", "manage_checkpoints"]
