"""Strict Stage 2B staging-variant identities.

The variant is part of temporal-state/receipt identity, not a convenience
flag.  Callers must therefore pass an enum member explicitly; accepting raw
strings here would allow an unvalidated configuration value to select a
different numerical path.
"""

from __future__ import annotations

from enum import Enum


class CACHStagingVariant(str, Enum):
    """Closed set of Stage 2B offline paired-staging variants."""

    REF_GDN_CORRECTED = "ref_gdn_corrected"
    CACH_A = "cach_a"

    @property
    def history_summary_operator(self) -> str:
        """Return the exact cache-history interpretation for this variant."""

        return history_summary_operator_for_variant(self)


def require_cach_staging_variant(value: object) -> CACHStagingVariant:
    """Require a typed variant without implicitly coercing a raw string."""

    if not isinstance(value, CACHStagingVariant):
        raise TypeError("staging_variant must be a CACHStagingVariant")
    return value


def history_summary_operator_for_variant(
    staging_variant: CACHStagingVariant,
) -> str:
    """Map a typed variant to its immutable history-summary operator.

    CACH-A retains the frozen action-conditioned operator identity.  The
    corrected reference explicitly identifies a recurrent video-only summary
    and must never be reinterpreted as action-conditioned history.
    """

    variant = require_cach_staging_variant(staging_variant)
    if variant is CACHStagingVariant.CACH_A:
        return "gdn_recurrent_paired_commit_v1"
    if variant is CACHStagingVariant.REF_GDN_CORRECTED:
        return "gdn_recurrent_video_only_paired_commit_v1"
    raise AssertionError("unreachable CACH staging variant")


__all__ = [
    "CACHStagingVariant",
    "history_summary_operator_for_variant",
    "require_cach_staging_variant",
]
