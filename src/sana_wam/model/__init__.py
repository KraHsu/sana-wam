"""Architecture dispatch by ``architecture.variant``.

The flattened model cfg preserves ``framework`` / ``variant`` (see
``sana_wam.config.flatten_model_cfg``). ``build_architecture`` reads ``variant``
and constructs the matching class:

- ``autoregressive`` → :class:`DualSystemARArchitecture` (SANA linear_relu, MoT
  block-causal duplicated-sequence path).
- ``joint_cross_attn`` → :class:`DualSystemCrossAttnArchitecture` (GDN video
  backbone + cross-attention action bridge).
- ``joint_self_attn`` → :class:`DualSystemSelfAttnArchitecture` (non-AR MoT joint
  self-attention).

Trainer / deploy call this instead of hardcoding a class.
"""

from __future__ import annotations

from typing import Any

_VARIANT_AUTOREGRESSIVE = "autoregressive"
_VARIANT_CROSS_ATTN = "joint_cross_attn"
_VARIANT_SELF_ATTN = "joint_self_attn"


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        v = getattr(cfg, key)
        return default if v is None else v
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def build_architecture(flat_cfg: Any):
    """Construct the architecture for ``flat_cfg`` (a flattened model cfg).

    ``flat_cfg`` is the output of ``flatten_model_cfg`` — it carries ``variant``
    at the top level. Raises on an unknown variant rather than silently picking
    a default (a config typo would otherwise train the wrong model).
    """
    variant = _cfg_get(flat_cfg, "variant", _VARIANT_AUTOREGRESSIVE)

    if variant in (_VARIANT_AUTOREGRESSIVE, None):
        from sana_wam.model.architecture import DualSystemARArchitecture

        return DualSystemARArchitecture(flat_cfg)
    if variant == _VARIANT_CROSS_ATTN:
        from sana_wam.model.cross_attn import DualSystemCrossAttnArchitecture

        return DualSystemCrossAttnArchitecture(flat_cfg)
    if variant == _VARIANT_SELF_ATTN:
        from sana_wam.model.joint_self_attn import DualSystemSelfAttnArchitecture

        return DualSystemSelfAttnArchitecture(flat_cfg)
    raise ValueError(
        f"Unknown architecture.variant={variant!r}. Choose from: "
        f"{_VARIANT_AUTOREGRESSIVE}, {_VARIANT_CROSS_ATTN}, {_VARIANT_SELF_ATTN}."
    )


__all__ = ["build_architecture"]
