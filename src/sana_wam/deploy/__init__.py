"""Deploy package — single-architecture (block-AR) project.

There is exactly one engine (``ARInferenceEngine``), so ``build_engine`` does
not dispatch on a registry — it always returns the AR engine. ``model_loader`` /
``policy`` / ``policy_server`` are added in the deploy phase.
"""

from sana_wam.deploy.base import BaseInferenceEngine


def build_engine(cfg, architecture, training_cfg=None):
    """Return the block-AR closed-loop inference engine (one AR step per generate)."""
    from sana_wam.deploy.ar_engine import ARInferenceEngine

    return ARInferenceEngine(cfg=cfg, architecture=architecture)


__all__ = ["BaseInferenceEngine", "build_engine"]
