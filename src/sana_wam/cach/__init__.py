"""Fail-closed contracts for Causal Action-Conditioned Hybrid SANA-WAM.

The package is deliberately standard-library-only at import time.  Stage 1
implements schemas and validators; it does not authorize model construction,
training, deployment, evaluation, GPU use, or run-root creation.
"""

from sana_wam.cach.authority import (
    CACHAuthorityError,
    is_cach_variant,
    reject_cach_deploy_before_runtime,
    reject_cach_model_build,
    reject_cach_training_before_runtime,
)

__all__ = [
    "CACHAuthorityError",
    "is_cach_variant",
    "reject_cach_deploy_before_runtime",
    "reject_cach_model_build",
    "reject_cach_training_before_runtime",
]
