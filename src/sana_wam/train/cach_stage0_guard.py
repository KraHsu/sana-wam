"""Torch-free guard for the reserved, non-executable CACH Stage-0 config."""

from __future__ import annotations

from typing import Any


def reject_cach_stage0_base_config(base_config: Any, *, entrypoint: str) -> None:
    """Reject a raw base config carrying the reserved ``cach_stage0`` marker.

    Callers must invoke this on the unmerged base document before importing
    torch, constructing a model, selecting a device, creating a root, or
    applying CLI overrides.  Checking the raw document makes
    ``cach_stage0=null`` incapable of removing the denial.
    """

    try:
        keys = base_config.keys()
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(f"{entrypoint}: base config must be a mapping") from exc
    reserved = "cach_stage0" in keys
    if reserved:
        raise RuntimeError(
            f"{entrypoint}: CACH Stage-0 configs are non-executable; use the "
            "reviewed CACH launcher only after a separately registered "
            "execution authority exists"
        )


__all__ = ["reject_cach_stage0_base_config"]
