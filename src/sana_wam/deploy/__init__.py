"""Deploy package — engine dispatch by architecture type.

``build_engine`` picks the inference engine matching the architecture:

- :class:`DualSystemARArchitecture` → :class:`ARInferenceEngine` (stateful block-AR,
  one chunk per ``generate``, linear_relu KV cache).
- :class:`DualSystemGDNARArchitecture` → :class:`GDNARInferenceEngine` (stateful
  cached-GDN streaming, rolling GDN state cache, one chunk per ``generate``).
- :class:`DualSystemCrossAttnArchitecture` → :class:`CrossAttnInferenceEngine`
  (stateless one-shot TI2V denoise, GDN video backbone + cross-attn action).
"""


def __getattr__(name):
    """Keep package import torch-free until a guarded engine build is requested."""

    if name == "BaseInferenceEngine":
        from sana_wam.deploy.base import BaseInferenceEngine

        return BaseInferenceEngine
    raise AttributeError(name)


def build_engine(cfg, architecture, training_cfg=None):
    """Return the inference engine matching ``architecture``'s type."""
    from sana_wam.train.cach_stage0_guard import reject_cach_stage0_base_config

    reject_cach_stage0_base_config(
        cfg, entrypoint="sana_wam.deploy.build_engine"
    )
    from sana_wam.cach.authority import (
        is_cach_variant,
        reject_cach_deploy_before_runtime,
    )

    # CACH is a subclass of the legacy GDN-AR architecture.  Reject it before
    # the isinstance chain so it cannot inherit fixed-ATC/command-only deploy
    # semantics by fallback.
    if is_cach_variant(cfg) or type(architecture).__name__ == (
        "CausalActionHybridArchitecture"
    ):
        reject_cach_deploy_before_runtime(cfg)
        raise RuntimeError("CACH deploy engine is not implemented in Stage 1")
    from sana_wam.model.cross_attn import DualSystemCrossAttnArchitecture
    from sana_wam.model.gdn_ar import DualSystemGDNARArchitecture

    # Check the GDN-AR subclass BEFORE its cross-attn parent (isinstance would
    # otherwise misroute the streaming arch to the one-shot cross-attn engine).
    if isinstance(architecture, DualSystemGDNARArchitecture):
        from sana_wam.deploy.gdn_ar_engine import GDNARInferenceEngine

        return GDNARInferenceEngine(cfg=cfg, architecture=architecture)

    if isinstance(architecture, DualSystemCrossAttnArchitecture):
        from sana_wam.deploy.cross_attn_engine import CrossAttnInferenceEngine

        return CrossAttnInferenceEngine(cfg=cfg, architecture=architecture)

    from sana_wam.deploy.ar_engine import ARInferenceEngine

    return ARInferenceEngine(cfg=cfg, architecture=architecture)


__all__ = ["BaseInferenceEngine", "build_engine"]
