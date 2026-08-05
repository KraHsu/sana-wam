"""Stage-1 execution boundary for CACH-SANA-WAM.

Stage 1 is an implementation-and-contract-test revision.  There is no
executable training/deploy authority and, consequently, no capability token
that can be minted by this module.  Canonical entry points call these helpers
before CUDA, datasets, model construction, checkpoint selection, or roots.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CACH_VARIANT = "cach_sana_wam_v0"
STAGE1_MARKER = "cach_stage1"


class CACHAuthorityError(RuntimeError):
    """A CACH operation was requested without an eligible authority."""


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    try:
        return value.get(key, default)
    except (AttributeError, TypeError):
        return getattr(value, key, default)


def _has_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value
    try:
        return key in value.keys()
    except (AttributeError, TypeError):
        return hasattr(value, key)


def _variant_from_config(cfg: Any) -> Any:
    model = _get(cfg, "model", None)
    architecture = _get(model, "architecture", None)
    return _get(architecture, "variant", None)


def is_cach_variant(cfg_or_flat: Any) -> bool:
    """Return whether a config selects CACH or carries its reserved marker.

    The marker is independently authoritative.  A caller cannot retain a
    ``cach_stage1`` document and overwrite only ``variant`` to route around a
    pre-runtime denial.
    """

    direct = _get(cfg_or_flat, "variant", None)
    nested = _variant_from_config(cfg_or_flat)
    # Treat the two coordinates independently.  A legacy-looking top-level
    # value must never hide a nested CACH selection (or vice versa) before
    # Trainer/deploy reaches CUDA, data, roots, or model dispatch.
    return (
        _has_key(cfg_or_flat, STAGE1_MARKER)
        or direct == CACH_VARIANT
        or nested == CACH_VARIANT
    )


def reject_cach_model_build(cfg_or_flat: Any, capability: Any = None) -> None:
    """Reject every Stage-1 CACH model construction attempt.

    ``capability`` is accepted only to keep the future Stage-2 seam explicit;
    Stage 1 intentionally defines no accepted token or permissive fallback.
    """

    if not is_cach_variant(cfg_or_flat):
        return
    del capability
    raise CACHAuthorityError(
        "CACH Stage 1 authorizes pure contract implementation/tests only; "
        "model construction requires a separate Stage 2 authority"
    )


def reject_cach_training_before_runtime(
    cfg: Any,
    launch_context: Any = None,
) -> None:
    """Reject CACH before Trainer touches CUDA, datasets, models, or roots."""

    if not is_cach_variant(cfg):
        return
    del launch_context
    raise CACHAuthorityError(
        "CACH training is not authorized: Stage 1 permits contract code and "
        "lightweight tests only"
    )


def reject_cach_deploy_before_runtime(cfg: Any) -> None:
    """Reject CACH before checkpoint discovery, torch import, or engine build."""

    if not is_cach_variant(cfg):
        return
    raise CACHAuthorityError(
        "CACH deploy is not authorized in Stage 1; the exact CACH checkpoint "
        "loader and applied-action authority are not commissioned"
    )


__all__ = [
    "CACHAuthorityError",
    "CACH_VARIANT",
    "STAGE1_MARKER",
    "is_cach_variant",
    "reject_cach_deploy_before_runtime",
    "reject_cach_model_build",
    "reject_cach_training_before_runtime",
]
