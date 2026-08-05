"""Lazy runtime-guard wrapper for the frozen CACH-A4 synthetic screen.

This R1 module changes no model, parameter, optimizer, task, metric, or
classification semantics.  Its only purpose is to require the two process
environment guards needed by the frozen vendor-GDN dependency before any
base A4 or vendor module can be imported.  The base implementation is loaded
only inside :func:`run_cach_a4_screen` after both values match exactly.

The module intentionally imports only Python standard-library modules at
module load time.  Importing this wrapper therefore cannot initialize Torch,
CUDA, Triton, the vendor tree, or a model.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import os
from typing import Any, Final


BASE_A4_MODULE: Final = (
    "sana_wam.model.cach_av1b_a4_state_conditioned_causal_odd_stream"
)
REQUIRED_RUNTIME_ENV: Final = {
    "GDN_DISABLE_COMPILE": "1",
    "TORCHDYNAMO_DISABLE": "1",
}

# These identities are copied literally from the frozen base A4 source.  R1
# is a launcher/environment repair, not a new architecture or screen schema.
CACH_A4_ARCHITECTURE_ID: Final = (
    "CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1"
)
A4_CONFIG_SCHEMA: Final = "cach.cach_a4.state_conditioned_causal_odd_stream.config.v1"
A4_SCREEN_SCHEMA: Final = "cach.cach_a4.state_conditioned_causal_odd_stream.screen.v1"


class A4R1EnvironmentError(RuntimeError):
    """Raised before the frozen base A4 module is imported."""


def require_runtime_environment(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Require exact vendor/Dynamo guards without importing the base model."""

    observed = os.environ if environ is None else environ
    mismatches = [
        f"{name}={observed.get(name)!r} (required '1')"
        for name in sorted(REQUIRED_RUNTIME_ENV)
        if observed.get(name) != REQUIRED_RUNTIME_ENV[name]
    ]
    if mismatches:
        raise A4R1EnvironmentError(
            "A4-R1 runtime environment rejected before base import: "
            + "; ".join(mismatches)
        )


def run_cach_a4_screen(*args: Any, **kwargs: Any) -> Any:
    """Validate the R1 environment and delegate to frozen base A4 exactly."""

    require_runtime_environment()
    base_a4 = importlib.import_module(BASE_A4_MODULE)
    delegate = getattr(base_a4, "run_cach_a4_screen", None)
    if not callable(delegate):
        raise RuntimeError("frozen base A4 does not expose callable run_cach_a4_screen")
    return delegate(*args, **kwargs)


__all__ = [
    "A4_CONFIG_SCHEMA",
    "A4_SCREEN_SCHEMA",
    "A4R1EnvironmentError",
    "BASE_A4_MODULE",
    "CACH_A4_ARCHITECTURE_ID",
    "REQUIRED_RUNTIME_ENV",
    "require_runtime_environment",
    "run_cach_a4_screen",
]
