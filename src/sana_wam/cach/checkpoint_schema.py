"""Exact CACH checkpoint metadata contract.

This module does not discover checkpoints, construct a model, or import torch.
It validates a caller-supplied exact state mapping against a frozen schema.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from sana_wam.cach.model_inventory import (
    CACHInventoryError,
    InventoryEntry,
    canonical_json_bytes,
)


class CACHCheckpointError(ValueError):
    """Checkpoint keys or per-tensor metadata differ from the frozen schema."""


@dataclass(frozen=True, slots=True)
class CheckpointEntry:
    name: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    required: bool = True

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise CACHCheckpointError("checkpoint key must be non-empty")
        if type(self.role) is not str or not self.role:
            raise CACHCheckpointError(f"checkpoint role missing for {self.name}")
        if (
            type(self.shape) is not tuple
            or not all(type(item) is int and item >= 0 for item in self.shape)
        ):
            raise CACHCheckpointError(f"checkpoint shape differs for {self.name}")
        expected = 1
        for item in self.shape:
            expected *= item
        if type(self.numel) is not int or self.numel != expected:
            raise CACHCheckpointError(f"checkpoint numel differs for {self.name}")
        if self.required is not True:
            raise CACHCheckpointError("every CACH checkpoint entry must be required")

    def canonical_dict(self) -> dict:
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value


@dataclass(frozen=True, slots=True)
class CheckpointSchema:
    candidate_revision: str
    entries: tuple[CheckpointEntry, ...]
    schema: str = "cach.checkpoint_schema.v1"

    def __post_init__(self) -> None:
        if type(self.candidate_revision) is not str or not self.candidate_revision:
            raise CACHCheckpointError("candidate revision must be non-empty")
        names = [entry.name for entry in self.entries]
        if names != sorted(names, key=lambda item: item.encode("utf-8")):
            raise CACHCheckpointError("checkpoint entries must use canonical name order")
        if len(names) != len(set(names)):
            raise CACHCheckpointError("checkpoint entries must be unique")

    def canonical_dict(self) -> dict:
        return {
            "schema": self.schema,
            "candidate_revision": self.candidate_revision,
            "entries": [entry.canonical_dict() for entry in self.entries],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.canonical_dict())).hexdigest()

    @property
    def key_set_sha256(self) -> str:
        names = [entry.name for entry in self.entries]
        return hashlib.sha256(canonical_json_bytes(names)).hexdigest()


def schema_from_inventory(
    entries: Iterable[InventoryEntry],
    *,
    candidate_revision: str,
) -> CheckpointSchema:
    """Create a design-time schema, excluding frozen VAE/text state exactly."""

    checkpoint_entries = []
    for entry in entries:
        if entry.architecture_role in {"vae", "text_encoder"}:
            continue
        checkpoint_entries.append(
            CheckpointEntry(
                name=entry.fully_qualified_name,
                role=entry.architecture_role,
                shape=entry.shape,
                dtype=entry.dtype,
                numel=entry.numel,
            )
        )
    checkpoint_entries.sort(key=lambda item: item.name.encode("utf-8"))
    return CheckpointSchema(
        candidate_revision=candidate_revision,
        entries=tuple(checkpoint_entries),
    )


def _tensor_metadata(value: Any, name: str) -> tuple[tuple[int, ...], str, int]:
    try:
        shape = tuple(int(item) for item in value.shape)
        dtype = str(value.dtype)
        if dtype.startswith("torch."):
            dtype = dtype[len("torch.") :]
        raw_numel = value.numel()
        numel = int(raw_numel)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CACHCheckpointError(
            f"checkpoint value for {name} lacks tensor metadata"
        ) from exc
    return shape, dtype, numel


def validate_checkpoint_state(
    state: Mapping[str, Any],
    schema: CheckpointSchema,
) -> None:
    if not isinstance(state, Mapping):
        raise CACHCheckpointError("checkpoint state must be a mapping")
    if not all(type(key) is str for key in state):
        raise CACHCheckpointError("checkpoint keys must be strings")
    expected = {entry.name: entry for entry in schema.entries}
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise CACHCheckpointError(
            f"checkpoint key set differs: missing={missing}, unexpected={unexpected}"
        )
    for name, entry in expected.items():
        observed = _tensor_metadata(state[name], name)
        required = (entry.shape, entry.dtype, entry.numel)
        if observed != required:
            raise CACHCheckpointError(
                f"checkpoint metadata differs for {name}: "
                f"expected={required}, got={observed}"
            )


def load_exact_checkpoint(
    module: Any,
    state: Mapping[str, Any],
    schema: CheckpointSchema,
) -> None:
    """Validate exact metadata, then invoke only a strict model load."""

    validate_checkpoint_state(state, schema)
    try:
        result = module.load_state_dict(state, strict=True)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise CACHCheckpointError(f"strict checkpoint load failed: {exc}") from exc
    missing = tuple(getattr(result, "missing_keys", ()))
    unexpected = tuple(getattr(result, "unexpected_keys", ()))
    if missing or unexpected:
        raise CACHCheckpointError(
            f"strict loader returned mismatches: missing={missing}, "
            f"unexpected={unexpected}"
        )


def validate_checkpoint_schema_against_inventory(
    schema: CheckpointSchema,
    inventory: Iterable[InventoryEntry],
) -> None:
    expected = schema_from_inventory(
        inventory,
        candidate_revision=schema.candidate_revision,
    )
    if schema.canonical_dict() != expected.canonical_dict():
        raise CACHInventoryError("checkpoint schema differs from model inventory")


__all__ = [
    "CACHCheckpointError",
    "CheckpointEntry",
    "CheckpointSchema",
    "load_exact_checkpoint",
    "schema_from_inventory",
    "validate_checkpoint_schema_against_inventory",
    "validate_checkpoint_state",
]
