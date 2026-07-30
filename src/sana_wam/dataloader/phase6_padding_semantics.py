"""Torch-free provenance for the frozen Phase-6 padding semantics.

The dataset contract carries this value so every downstream row is bound to
the same interpretation of ordinary RoboTwin padding.  This module describes
provenance only; runtime mask construction remains in the dataset/model path.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from typing import Any


PADDING_SEMANTICS_AMENDMENT_SHA256 = (
    "cf9bda2b5626c4c2a5b967c2d11673c0885e4e9aa2e54e81dafc5504e90bed5a"
)
PADDING_SEMANTICS_CONTRACT_SHA256 = (
    "6215609e0e978d30e8ba4d34f763a3ebc55c7d1783a162c092113e9e302d153a"
)
PADDING_REQUIRED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256 = (
    "8632450705ee3ba35855f07ccb48ad300aac5abf711abf0307808b52fb613081"
)
PADDING_REQUIRED_EXPANSION_SUPPORT_CONTRACT_SHA256 = (
    "a659b602fbba11223b8d5129a82e8699bfd99b2cc42b262d154cd9b11504b116"
)

PADDING_SEMANTICS_CONTRACT = {
    "action_key_rule": (
        "action token is a usable structured joint-AR key iff action_is_pad is "
        "false; apply to both noisy and clean copies at every layer"
    ),
    "action_output_rule": (
        "action losses and non-regression remain evaluated only on non-padded "
        "action tokens"
    ),
    "attention_rule": (
        "all clean-history states, adapter video-memory states, and within-frame "
        "noisy-key states exclude padded keys before numerator and denominator "
        "accumulation"
    ),
    "backward_compatibility_rule": (
        "outside a Phase-6 bound row, an absent modality mask means all keys of "
        "that modality are valid"
    ),
    "batch_rule": (
        "key validity is per sample and must not be collapsed into a batch-common "
        "mask"
    ),
    "phase6_mask_requirement": (
        "every Phase-6 plan row must provide boolean video_is_pad [B,T] and "
        "action_is_pad [B,Ta] with exact shapes; missing, stale, or non-boolean "
        "masks fail closed"
    ),
    "query_rule": (
        "padded query rows may be computed but are never usable as structured "
        "joint-AR keys in the same or any later layer"
    ),
    "schema_version": "sana-phase6-padding-semantics-contract-v1",
    "sequence_layout": "[video_noisy,video_clean,action_noisy,action_clean]",
    "temporal_mlp_kernel": (
        "GLUMBConvTemp t_kernel_size=3 within each frame_chunk_size=2 physical chunk"
    ),
    "temporal_mlp_rule": (
        "after per-frame spatial GLU and before temporal convolution, zero "
        "padded-frame states; apply the temporal convolution, then zero "
        "padded-frame outputs so valid states are invariant to padded latent values"
    ),
    "track_duplication_rule": (
        "the same per-modality validity mask applies to noisy and clean copies"
    ),
    "video_key_expansion_rule": (
        "repeat each latent-frame validity value over all h*w patch tokens before "
        "sequence concatenation"
    ),
    "video_key_rule": (
        "video patch token is a usable structured joint-AR key iff its source "
        "latent frame is not padded"
    ),
    "video_loss_rule": (
        "on-path video MSE is reduced only over non-padded latent frames and also "
        "excludes bootstrap latent0 when bootstrap_clean_prefix is enabled"
    ),
    "window_attention_rule": (
        "under frozen frame_chunk_size=2 and window_count=(1,1,4), each real "
        "latent frame occupies a distinct temporal window; validation must fail "
        "closed if a valid and padded latent frame could share a window"
    ),
}


class PaddingSemanticsError(ValueError):
    """Raised when frozen padding provenance is internally inconsistent."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PaddingSemanticsError(
            f"padding provenance is not canonical-JSON serializable: {exc}"
        ) from exc
    return payload + b"\n"


def padding_semantics_contract() -> dict[str, str]:
    """Return a defensive copy after authenticating the registered value."""

    return validate_padding_semantics_contract(PADDING_SEMANTICS_CONTRACT)


def validate_padding_semantics_contract(value: Any) -> dict[str, str]:
    """Validate the exact nested schema, values, types, and canonical hash."""

    if not isinstance(value, Mapping) or set(value) != set(
        PADDING_SEMANTICS_CONTRACT
    ):
        raise PaddingSemanticsError(
            "padding-semantics contract has an unexpected schema"
        )
    copied = deepcopy(dict(value))
    if any(
        type(copied[key]) is not type(expected) or copied[key] != expected
        for key, expected in PADDING_SEMANTICS_CONTRACT.items()
    ):
        raise PaddingSemanticsError(
            "padding-semantics contract value or type differs"
        )
    observed = sha256(_canonical_json_bytes(copied)).hexdigest()
    if observed != PADDING_SEMANTICS_CONTRACT_SHA256:
        raise PaddingSemanticsError(
            "in-process padding-semantics contract differs from its pinned SHA256"
        )
    return copied


def padding_eligibility_dependency() -> dict[str, str]:
    """Return the eligibility provenance required by the padding amendment."""

    return {
        "amendment_sha256": (
            PADDING_REQUIRED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        ),
        "support_contract_sha256": (
            PADDING_REQUIRED_EXPANSION_SUPPORT_CONTRACT_SHA256
        ),
    }


def validate_padding_eligibility_dependency(
    *,
    expansion_eligibility_amendment_sha256: Any,
    expansion_support_contract_sha256: Any,
) -> None:
    """Fail unless expansion provenance satisfies the padding amendment."""

    if (
        type(expansion_eligibility_amendment_sha256) is not str
        or expansion_eligibility_amendment_sha256
        != PADDING_REQUIRED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        or type(expansion_support_contract_sha256) is not str
        or expansion_support_contract_sha256
        != PADDING_REQUIRED_EXPANSION_SUPPORT_CONTRACT_SHA256
    ):
        raise PaddingSemanticsError(
            "padding-semantics eligibility dependency differs from its amendment"
        )


# Authenticate the literal and its registered dependency at import time.
padding_semantics_contract()
validate_padding_eligibility_dependency(
    expansion_eligibility_amendment_sha256=(
        PADDING_REQUIRED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
    ),
    expansion_support_contract_sha256=(
        PADDING_REQUIRED_EXPANSION_SUPPORT_CONTRACT_SHA256
    ),
)
