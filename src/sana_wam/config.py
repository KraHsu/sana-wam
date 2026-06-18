"""Config flattening — replaces openwam's architecture registry resolution.

The architecture ``__init__`` consumes a single flat ``DictConfig`` (reads keys
via ``cfg.get(...)``). The training/deploy YAML keeps the readable nested shape
(``model.architecture`` / ``model.action_backbone`` / ``model.video_backbone``);
``flatten_model_cfg`` merges them into the flat dict the architecture expects.

This is the (registry-free) equivalent of openwam's
``resolve_architecture_config`` flattening (registry.py:141-153), minus the
framework/variant registry lookup — there is exactly one architecture here.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def flatten_model_cfg(model_cfg: Any) -> DictConfig:
    """Flatten ``{architecture, action_backbone, video_backbone}`` into one cfg.

    - architecture items (minus framework/variant) become top-level keys
    - framework/variant are preserved (the deploy dispatch reads them)
    - action_backbone items are merged in at top level (the action DiT reads
      ``dim``/``ffn_dim``/``attn_kernel`` from the flat cfg)
    - the ``video_backbone`` subdict is kept nested under ``video_backbone``
    """
    arch = _get(model_cfg, "architecture", {}) or {}
    action = _get(model_cfg, "action_backbone", {}) or {}
    arch_items = OmegaConf.to_container(arch, resolve=True) if OmegaConf.is_config(arch) else dict(arch)
    action_items = OmegaConf.to_container(action, resolve=True) if OmegaConf.is_config(action) else dict(action)

    params = {k: v for k, v in arch_items.items() if k not in ("framework", "variant")}
    params["framework"] = arch_items.get("framework")
    params["variant"] = arch_items.get("variant")
    for k, v in action_items.items():
        params[k] = v

    vb = _get(model_cfg, "video_backbone", None)
    if vb is not None:
        params["video_backbone"] = OmegaConf.to_container(vb, resolve=True) if OmegaConf.is_config(vb) else vb

    return OmegaConf.create(params)


__all__ = ["flatten_model_cfg"]
