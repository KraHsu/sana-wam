"""Complete-random initialization and pair-parity checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

from sana_wam.cach.model_inventory import CACHInventoryError, InventoryEntry


def validate_fresh_initialization_fields(
    *,
    initialization_mode: object,
    model_path: object,
    init_dit_from: object,
    init_checkpoint: object,
    init_checkpoint_sha256: object,
    freeze: object,
) -> None:
    """Validate the fields before any builder/checkpoint seam is reachable."""

    if initialization_mode != "complete_random_v1":
        raise CACHInventoryError("initialization mode must be complete_random_v1")
    for name, value in {
        "model_path": model_path,
        "init_dit_from": init_dit_from,
        "init_checkpoint": init_checkpoint,
        "init_checkpoint_sha256": init_checkpoint_sha256,
    }.items():
        if value is not None:
            raise CACHInventoryError(f"fresh CACH forbids {name}")
    if type(freeze) is not list or freeze != [
        "video_backbone.vae",
        "video_backbone.text_encoder",
    ]:
        raise CACHInventoryError("fresh CACH freeze allowlist differs")


def compare_shared_initialization(
    reference: Iterable[InventoryEntry],
    candidate: Iterable[InventoryEntry],
) -> None:
    """Require complete inventory parity for every shared tensor.

    Initialization parity is not only tensor-byte parity.  A shared tensor
    cannot silently change architecture role, trainability, optimizer
    ownership, source, or any other inventory metadata between pair arms.
    Comparing the canonical documents also makes newly added inventory fields
    fail closed instead of requiring this checker to maintain a second,
    potentially incomplete field allowlist.
    """

    def shared(entries: Iterable[InventoryEntry]) -> Mapping[str, InventoryEntry]:
        result = {}
        for entry in entries:
            if entry.shared_or_operator_specific != "shared":
                continue
            if entry.fully_qualified_name in result:
                raise CACHInventoryError("duplicate shared inventory name")
            result[entry.fully_qualified_name] = entry
        return result

    left = shared(reference)
    right = shared(candidate)
    if set(left) != set(right):
        raise CACHInventoryError("reference/candidate shared name set differs")
    for name in sorted(left):
        ref = left[name]
        cand = right[name]
        ref_metadata = ref.canonical_dict()
        cand_metadata = cand.canonical_dict()
        if ref_metadata != cand_metadata:
            differing_fields = sorted(
                key
                for key in ref_metadata
                if ref_metadata[key] != cand_metadata.get(key)
            )
            raise CACHInventoryError(
                "reference/candidate shared initialization differs: "
                f"{name}; metadata_fields={differing_fields}"
            )


def validate_operator_specific_delta(
    reference: Iterable[InventoryEntry],
    candidate: Iterable[InventoryEntry],
    *,
    expected_candidate_names: set[str],
) -> None:
    """Require CACH-A's only candidate-only tensors to match its frozen delta."""

    ref_names = {
        item.fully_qualified_name
        for item in reference
        if item.shared_or_operator_specific == "operator_specific"
    }
    cand_names = {
        item.fully_qualified_name
        for item in candidate
        if item.shared_or_operator_specific == "operator_specific"
    }
    if ref_names:
        raise CACHInventoryError("CACH-A reference has operator-specific tensors")
    if cand_names != expected_candidate_names:
        raise CACHInventoryError("CACH-A operator-specific tensor set differs")


__all__ = [
    "compare_shared_initialization",
    "validate_fresh_initialization_fields",
    "validate_operator_specific_delta",
]
