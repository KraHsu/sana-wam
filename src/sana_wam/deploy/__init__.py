"""Deploy package — engine dispatch by architecture type.

``build_engine`` picks the inference engine matching the architecture:

- :class:`DualSystemARArchitecture` → :class:`ARInferenceEngine` (stateful block-AR,
  one chunk per ``generate``, linear_relu KV cache).
- :class:`DualSystemCrossAttnArchitecture` → :class:`CrossAttnInferenceEngine`
  (stateless one-shot TI2V denoise, GDN video backbone + cross-attn action).
"""

from sana_wam.deploy.base import BaseInferenceEngine


def build_engine(cfg, architecture, training_cfg=None):
    """Return the inference engine matching ``architecture``'s type."""
    from sana_wam.model.cross_attn import DualSystemCrossAttnArchitecture

    if isinstance(architecture, DualSystemCrossAttnArchitecture):
        from sana_wam.deploy.cross_attn_engine import CrossAttnInferenceEngine

        return CrossAttnInferenceEngine(cfg=cfg, architecture=architecture)

    from sana_wam.deploy.ar_engine import ARInferenceEngine

    return ARInferenceEngine(cfg=cfg, architecture=architecture)


__all__ = ["BaseInferenceEngine", "build_engine"]
