"""Strictly-positive, checkpoint-compatible feature maps for SANA linear attention."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class StrictlyPositiveLearnableFeatureMap(nn.Module):
    """Drop-in replacement for upstream ``LearnableFeatureMap``.

    Parameter names and shapes stay exactly ``alpha``, ``w1.*``, and ``w2.*`` so
    an existing learnable-ReLU checkpoint remains strict-load compatible.
    """

    def __init__(
        self,
        dim: int,
        hidden: int | None = None,
        *,
        beta: float = 16.0,
        delta: float = 1e-4,
    ) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        hidden = hidden or dim
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = float(beta)
        self.delta = float(delta)

    @classmethod
    def from_learnable(
        cls,
        source: nn.Module,
        *,
        beta: float,
        delta: float,
    ) -> "StrictlyPositiveLearnableFeatureMap":
        for name in ("w1", "w2", "alpha"):
            if not hasattr(source, name):
                raise TypeError(
                    "strictly-positive warm start requires an upstream "
                    f"LearnableFeatureMap with {name!r}"
                )
        replacement = cls(
            int(source.w1.in_features),
            int(source.w1.out_features),
            beta=beta,
            delta=delta,
        )
        # Preserve the exact Parameter objects, state-dict keys, dtype, and device.
        replacement.w1 = source.w1
        replacement.w2 = source.w2
        replacement.alpha = source.alpha
        return replacement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_dtype = self.w1.weight.dtype
        features = x.transpose(-1, -2).to(weight_dtype)
        residual = self.w2(F.gelu(self.w1(features)))
        residual = residual.to(x.dtype).transpose(-1, -2)
        preactivation = x + torch.tanh(self.alpha).to(x.dtype) * residual
        return F.softplus(preactivation, beta=self.beta) + self.delta


def install_strictly_positive_feature_maps(
    dit: nn.Module,
    config: Mapping | None,
) -> int:
    """Replace every learnable SANA feature map in ``dit`` in place."""
    if config is None:
        return 0
    if not isinstance(config, Mapping):
        raise TypeError(
            "strictly_positive_feature_map must be a mapping with beta/delta"
        )
    unknown = sorted(set(config) - {"beta", "delta"})
    if unknown:
        raise ValueError(
            f"unknown strictly_positive_feature_map fields: {unknown}"
        )
    beta = float(config.get("beta", 16.0))
    delta = float(config.get("delta", 1e-4))
    count = 0
    for block in getattr(dit, "blocks", ()):
        attention = getattr(block, "attn", None)
        source = getattr(attention, "kernel_func", None)
        if source is None:
            continue
        if isinstance(source, StrictlyPositiveLearnableFeatureMap):
            raise RuntimeError("strictly-positive feature map was installed twice")
        if not all(hasattr(source, name) for name in ("w1", "w2", "alpha")):
            raise TypeError(
                "strictly_positive_feature_map requires linear_feature_map=learnable"
            )
        attention.kernel_func = StrictlyPositiveLearnableFeatureMap.from_learnable(
            source,
            beta=beta,
            delta=delta,
        )
        count += 1
    if count == 0:
        raise RuntimeError("no learnable SANA feature maps were found to replace")
    return count


__all__ = [
    "StrictlyPositiveLearnableFeatureMap",
    "install_strictly_positive_feature_maps",
]
