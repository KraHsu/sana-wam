"""Deploy package — engine dispatch by architecture type.

``build_engine`` picks the inference engine matching the architecture:

- :class:`DualSystemARArchitecture` → :class:`ARInferenceEngine` (stateful block-AR,
  one chunk per ``generate``, linear_relu KV cache).
- :class:`DualSystemGDNARArchitecture` → :class:`GDNARInferenceEngine` (stateful
  cached-GDN streaming, rolling GDN state cache, one chunk per ``generate``).
- :class:`DualSystemCrossAttnArchitecture` → :class:`CrossAttnInferenceEngine`
  (stateless one-shot TI2V denoise, GDN video backbone + cross-attn action).
"""

from sana_wam.deploy.base import BaseInferenceEngine


def build_engine(cfg, architecture, training_cfg=None):
    """Return the inference engine matching ``architecture``'s type."""
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
