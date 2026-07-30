"""Load a SANA block-AR architecture from a training checkpoint directory.

There is exactly one architecture and one video backbone, so there is no
registry resolution and no multi-backbone deploy branches. The on-disk contract:
  - ``config.yaml``                     — saved training config
  - ``checkpoint_step_<N>.safetensors`` — architecture state_dict
  - ``action_stats.npy``                — deploy (de)normalization

Construction loads the SANA bundle (``model.video_backbone.model_path``) then
``load_checkpoint`` overwrites every weight (incl. the frozen VAE/text encoder,
which are in the saved state_dict) with the trained values.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from sana_wam.config import flatten_model_cfg
from sana_wam.model import build_architecture
from sana_wam.model.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}


def _find_latest_checkpoint(ckpt_dir: str) -> str:
    """Return the highest-step ``checkpoint_step_N.safetensors`` in *ckpt_dir*."""
    files = glob.glob(os.path.join(ckpt_dir, "checkpoint_step_*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No checkpoint_step_*.safetensors found in {ckpt_dir}")
    step_re = re.compile(r"checkpoint_step_(\d+)\.safetensors$")
    numbered = [(int(m.group(1)), f) for f in files if (m := step_re.search(os.path.basename(f)))]
    if not numbered:
        raise FileNotFoundError(f"No checkpoint_step_<int>.safetensors in {ckpt_dir}")
    numbered.sort(key=lambda p: p[0])
    return numbered[-1][1]


def load_from_checkpoint_dir(
    ckpt_dir: str,
    device: str = "cuda",
    ckpt_name: Optional[str] = None,
) -> Tuple[DictConfig, BaseWAMArchitecture]:
    """Reconstruct ``(cfg, architecture)`` from a self-contained checkpoint dir."""
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in {ckpt_dir}")
    cfg = OmegaConf.load(config_path)

    ckpt_path = os.path.join(ckpt_dir, ckpt_name) if ckpt_name else _find_latest_checkpoint(ckpt_dir)
    logger.info("Loading checkpoint: %s", ckpt_path)

    flat_cfg = flatten_model_cfg(cfg.model)
    # Component construction already places the DiT, VAE, and text encoder;
    # honor the loader's target instead of implicitly staging them on cuda:0.
    OmegaConf.update(flat_cfg, "video_backbone._device", device, merge=False)
    OmegaConf.update(flat_cfg, "video_backbone._ckpt_dir", ckpt_dir, merge=False)
    architecture = build_architecture(flat_cfg)

    mp = OmegaConf.select(cfg, "accelerate.mixed_precision", default="bf16")
    model_dtype = _DTYPE_MAP.get(str(mp).strip().lower(), torch.bfloat16)
    architecture.set_dtype_device(model_dtype, torch.device(device))
    # ``cross_attn.norm_kv.*`` was added to the shared BridgeCrossAttention for the
    # frozen linear-attn SANA backbone; older GDN cross-attn / GDN-AR checkpoints
    # predate it. Tolerate ONLY those keys being absent (kept at default LayerNorm
    # init, logged) so deploying a pre-norm_kv checkpoint no longer dies on strict
    # load; every other missing/unexpected key still fails loudly.
    architecture.load_checkpoint(ckpt_path, strict=True, allow_missing_patterns=("norm_kv",))
    architecture.eval()

    architecture.attach_action_normalizer(_build_action_normalizer(cfg, ckpt_dir))
    logger.info("Model loaded on %s", device)
    return cfg, architecture


def _build_action_normalizer(cfg: DictConfig, ckpt_dir: str):
    """Build the deploy action normalizer from ``action_stats.npy`` + saved config.

    Returns ``None`` when normalization was disabled at train time.
    """
    norm_mode = OmegaConf.select(cfg, "dataloader.normalize_mode", default=None)
    action_mode = OmegaConf.select(cfg, "dataloader.action_mode", default="joint")
    if OmegaConf.select(cfg, "dataloader", default=None) is None or norm_mode in (None, "", "none", "null"):
        logger.info("[normalizer] normalize_mode=%r disabled; normalizer INACTIVE.", norm_mode)
        return None

    stats_path = os.path.join(ckpt_dir, "action_stats.npy")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Missing required action_stats.npy in checkpoint dir: {stats_path}. "
            "Checkpoints with active action normalization must include it."
        )

    from sana_wam.dataloader.transforms.normalize import YAML_TO_NORM_MODE, ActionNormalizer, load_mode_stats

    if norm_mode not in YAML_TO_NORM_MODE:
        logger.warning("[normalizer] Unknown normalize_mode %r; DISABLED.", norm_mode)
        return None
    mode_stats = load_mode_stats(stats_path, action_mode)
    if mode_stats is None:
        logger.warning("[normalizer] No '%s' entry in %s; DISABLED.", action_mode, stats_path)
        return None
    normalizer = ActionNormalizer(mode=YAML_TO_NORM_MODE[norm_mode], stats=mode_stats)
    logger.info("[normalizer] Active: mode=%s action_mode=%s dim=%d", norm_mode, action_mode, len(mode_stats["mean"]))
    return normalizer


__all__ = ["load_from_checkpoint_dir"]
