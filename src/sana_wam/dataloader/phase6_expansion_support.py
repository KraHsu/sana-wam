"""Torch-free structural eligibility for Phase-6 local expansion.

The eligibility decision is deliberately limited to the frozen temporal
geometry and padding mask.  It must never inspect actions, outcomes, losses,
model outputs, tangents, or chords.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from typing import Any


EXPANSION_SUPPORT_CONTRACT_SHA256 = (
    "a659b602fbba11223b8d5129a82e8699bfd99b2cc42b262d154cd9b11504b116"
)
EXPANSION_ELIGIBILITY_AMENDMENT_SHA256 = (
    "8632450705ee3ba35855f07ccb48ad300aac5abf711abf0307808b52fb613081"
)

EXPANSION_SUPPORT_CONTRACT = {
    "actual_valid_raw_rule": (
        "min(window_logical_length,max(0,episode_length-start_frame))"
    ),
    "bootstrap_clean_prefix": True,
    "bootstrap_rule": "exclude latent0 only",
    "causal_latent_validity_rule": (
        "latent0=sampled0; tail sampled frames grouped by temporal_compression "
        "with out-of-range entries padded; latent valid iff any grouped sampled "
        "frame is valid"
    ),
    "causal_temporal": True,
    "chunk_validity_reduction": "any",
    "content_dependent_filtering_forbidden": [
        "action_values",
        "outcomes",
        "success_labels",
        "task_specific_thresholds",
        "model_outputs",
        "tangent_values",
        "chord_values",
        "loss_values",
    ],
    "eligibility_rule": (
        "at least one chunk contains at least one non-bootstrap non-padded latent frame"
    ),
    "frame_chunk_size": 2,
    "latent_frame_count": 8,
    "minimum_actual_valid_raw_frames": 5,
    "minimum_valid_sampled_video_frames": 2,
    "raw_window_frames": 113,
    "sample_validity_rule": "video_sample_index < actual_valid_raw_frames",
    "schema_version": "sana-phase6-expansion-support-contract-v1",
    "temporal_compression": 4,
    "video_sample_indices": list(range(0, 113, 4)),
    "video_stride": 4,
}

EXPANSION_SUPPORT_KEYS = frozenset(
    {
        "actual_valid_raw_frames",
        "valid_sampled_video_frames",
        "valid_latent_frames",
        "nonbootstrap_valid_latent_frames",
        "eligible_chunk_count",
        "eligible",
    }
)
EXPANSION_SUPPORT_COUNT_KEYS = (
    "actual_valid_raw_frames",
    "valid_sampled_video_frames",
    "valid_latent_frames",
    "nonbootstrap_valid_latent_frames",
    "eligible_chunk_count",
)
_RAW_WINDOW_FRAMES = 113
_VIDEO_STRIDE = 4
_VIDEO_SAMPLE_INDICES = tuple(range(0, 113, 4))
_CAUSAL_TEMPORAL = True
_TEMPORAL_COMPRESSION = 4
_LATENT_FRAME_COUNT = 8
_BOOTSTRAP_CLEAN_PREFIX = True
_FRAME_CHUNK_SIZE = 2


class ExpansionSupportError(ValueError):
    """Raised when support metadata or frozen geometry is malformed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def expansion_support_contract() -> dict[str, Any]:
    """Return an isolated copy of the frozen contract after authenticating it."""

    observed = sha256(_canonical_json_bytes(EXPANSION_SUPPORT_CONTRACT)).hexdigest()
    if observed != EXPANSION_SUPPORT_CONTRACT_SHA256:
        raise ExpansionSupportError(
            "in-process expansion-support contract differs from its pinned SHA256"
        )
    return deepcopy(EXPANSION_SUPPORT_CONTRACT)


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ExpansionSupportError(f"{name} must be an integer >= {minimum}")
    return value


def validate_expansion_geometry(
    *,
    raw_window_frames: Any,
    video_stride: Any,
    video_sample_indices: Any,
    causal_temporal: Any,
    temporal_compression: Any,
) -> None:
    """Fail unless dataset metadata matches the frozen support geometry."""

    if type(raw_window_frames) is not int or raw_window_frames != _RAW_WINDOW_FRAMES:
        raise ExpansionSupportError("raw-window geometry differs from support contract")
    if type(video_stride) is not int or video_stride != _VIDEO_STRIDE:
        raise ExpansionSupportError("video stride differs from support contract")
    try:
        observed_sample_indices = list(video_sample_indices)
    except TypeError as exc:
        raise ExpansionSupportError(
            "video sample indices are not an iterable of frozen indices"
        ) from exc
    if any(type(index) is not int for index in observed_sample_indices) or (
        observed_sample_indices != list(_VIDEO_SAMPLE_INDICES)
    ):
        raise ExpansionSupportError("video sample indices differ from support contract")
    if type(causal_temporal) is not bool or causal_temporal is not _CAUSAL_TEMPORAL:
        raise ExpansionSupportError("causal-temporal mode differs from support contract")
    if (
        type(temporal_compression) is not int
        or temporal_compression != _TEMPORAL_COMPRESSION
    ):
        raise ExpansionSupportError(
            "temporal compression differs from support contract"
        )


def _support_from_actual_valid_raw_frames(
    actual_valid_raw_frames: int,
) -> dict[str, int | bool]:
    actual_valid = _require_int(
        actual_valid_raw_frames,
        "actual_valid_raw_frames",
    )
    if actual_valid > _RAW_WINDOW_FRAMES:
        raise ExpansionSupportError(
            "actual_valid_raw_frames exceeds the frozen raw window"
        )

    sampled_valid = [
        index < actual_valid for index in _VIDEO_SAMPLE_INDICES
    ]
    latent_valid = [sampled_valid[0]]
    tail = sampled_valid[1:]
    if len(tail) % _TEMPORAL_COMPRESSION != 0:
        raise ExpansionSupportError("frozen causal temporal geometry is not divisible")
    latent_valid.extend(
        any(tail[offset : offset + _TEMPORAL_COMPRESSION])
        for offset in range(0, len(tail), _TEMPORAL_COMPRESSION)
    )
    if len(latent_valid) != _LATENT_FRAME_COUNT:
        raise ExpansionSupportError("derived latent-frame count differs from contract")

    nonbootstrap_valid = list(latent_valid)
    if _BOOTSTRAP_CLEAN_PREFIX:
        nonbootstrap_valid[0] = False
    if len(nonbootstrap_valid) % _FRAME_CHUNK_SIZE != 0:
        raise ExpansionSupportError("latent frames are not divisible into frozen chunks")
    chunk_valid = [
        any(nonbootstrap_valid[offset : offset + _FRAME_CHUNK_SIZE])
        for offset in range(0, len(nonbootstrap_valid), _FRAME_CHUNK_SIZE)
    ]
    eligible_chunk_count = sum(chunk_valid)
    return {
        "actual_valid_raw_frames": actual_valid,
        "valid_sampled_video_frames": sum(sampled_valid),
        "valid_latent_frames": sum(latent_valid),
        "nonbootstrap_valid_latent_frames": sum(nonbootstrap_valid),
        "eligible_chunk_count": eligible_chunk_count,
        "eligible": eligible_chunk_count >= 1,
    }


def expansion_support_for_window(
    *,
    episode_length: Any,
    start_frame: Any,
    window_logical_length: Any,
) -> dict[str, int | bool]:
    """Compute the structural expansion support for one ordinary window."""

    episode_length = _require_int(episode_length, "episode_length", minimum=1)
    start_frame = _require_int(start_frame, "start_frame")
    window_logical_length = _require_int(
        window_logical_length,
        "window_logical_length",
    )
    if window_logical_length > _RAW_WINDOW_FRAMES:
        raise ExpansionSupportError(
            "window_logical_length exceeds the frozen raw window"
        )
    actual_valid = min(
        window_logical_length,
        max(0, episode_length - start_frame),
    )
    return _support_from_actual_valid_raw_frames(actual_valid)


def validate_expansion_support(
    value: Any,
    *,
    require_eligible: bool,
) -> dict[str, int | bool]:
    """Validate a serialized support dictionary and return an isolated copy."""

    if not isinstance(value, Mapping) or set(value) != EXPANSION_SUPPORT_KEYS:
        raise ExpansionSupportError("expansion support has an unexpected schema")
    counts = {
        key: _require_int(value[key], key) for key in EXPANSION_SUPPORT_COUNT_KEYS
    }
    if type(value["eligible"]) is not bool:
        raise ExpansionSupportError("eligible must be a boolean")
    actual_valid = counts["actual_valid_raw_frames"]
    expected = _support_from_actual_valid_raw_frames(actual_valid)
    if dict(value) != expected:
        raise ExpansionSupportError(
            "expansion support differs from the frozen structural derivation"
        )
    if require_eligible and not expected["eligible"]:
        raise ExpansionSupportError(
            "no non-bootstrap, non-padded expansion chunk exists"
        )
    return dict(expected)


def expansion_support_for_child_window(
    child: Any,
    *,
    episode_index: Any,
    start_frame: Any,
    window_logical_length: Any,
) -> dict[str, int | bool]:
    """Compute support from one RoboTwin child's metadata without materializing it."""

    validate_expansion_geometry(
        raw_window_frames=getattr(child, "_raw_window_len", None),
        video_stride=getattr(child, "video_stride", None),
        video_sample_indices=getattr(child, "_video_sample_indices", ()),
        causal_temporal=getattr(child, "causal_temporal", None),
        temporal_compression=getattr(child, "temporal_compression", None),
    )
    episode_index = _require_int(episode_index, "episode_index")
    episode_lengths = getattr(child, "_episode_lengths", None)
    if not isinstance(episode_lengths, (list, tuple)):
        raise ExpansionSupportError("runtime child has no episode-length metadata")
    if episode_index >= len(episode_lengths):
        raise ExpansionSupportError("episode_index is outside episode-length metadata")
    episode_length = _require_int(
        episode_lengths[episode_index],
        "episode_length",
        minimum=1,
    )
    return expansion_support_for_window(
        episode_length=episode_length,
        start_frame=start_frame,
        window_logical_length=window_logical_length,
    )


def _authenticate_implementation() -> None:
    contract = expansion_support_contract()
    implementation_geometry = {
        "raw_window_frames": _RAW_WINDOW_FRAMES,
        "video_stride": _VIDEO_STRIDE,
        "video_sample_indices": list(_VIDEO_SAMPLE_INDICES),
        "causal_temporal": _CAUSAL_TEMPORAL,
        "temporal_compression": _TEMPORAL_COMPRESSION,
        "latent_frame_count": _LATENT_FRAME_COUNT,
        "bootstrap_clean_prefix": _BOOTSTRAP_CLEAN_PREFIX,
        "frame_chunk_size": _FRAME_CHUNK_SIZE,
    }
    if any(contract[key] != value for key, value in implementation_geometry.items()):
        raise ExpansionSupportError(
            "expansion-support implementation geometry differs from its contract"
        )
    if (
        _support_from_actual_valid_raw_frames(4)["eligible"]
        or not _support_from_actual_valid_raw_frames(5)["eligible"]
    ):
        raise ExpansionSupportError(
            "expansion-support implementation minimum differs from its contract"
        )


# Authenticate once at import; per-window enumeration remains an O(geometry) mask pass.
_authenticate_implementation()
