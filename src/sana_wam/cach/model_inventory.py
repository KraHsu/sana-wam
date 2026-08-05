"""Canonical, framework-neutral CACH parameter/buffer inventory."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROLES = {
    "action_backbone",
    "action_conditioner",
    "buffer",
    "gdn",
    "proprio_encoder",
    "text_encoder",
    "vae",
    "video_dit",
}
_OWNERSHIP = {"operator_specific", "shared"}
_SOURCES = {"random", "source_external"}


class CACHInventoryError(ValueError):
    """An inventory is ambiguous, incomplete, or violates freeze policy."""


def _digest(value: str, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise CACHInventoryError(f"{name} must be one lowercase SHA256")
    return value


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    fully_qualified_name: str
    kind: str
    architecture_role: str
    shared_or_operator_specific: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    requires_grad: bool
    optimizer_group_or_null: str | None
    init_tensor_sha256: str
    source_external_or_random: str

    def __post_init__(self) -> None:
        if (
            type(self.fully_qualified_name) is not str
            or not self.fully_qualified_name
            or self.fully_qualified_name.strip() != self.fully_qualified_name
        ):
            raise CACHInventoryError("inventory name must be non-empty and canonical")
        if self.kind not in {"buffer", "parameter"}:
            raise CACHInventoryError(f"invalid inventory kind for {self.fully_qualified_name}")
        if self.architecture_role not in _ROLES:
            raise CACHInventoryError(
                f"invalid architecture role for {self.fully_qualified_name}"
            )
        if self.shared_or_operator_specific not in _OWNERSHIP:
            raise CACHInventoryError(
                f"invalid ownership for {self.fully_qualified_name}"
            )
        if (
            type(self.shape) is not tuple
            or not all(type(item) is int and item >= 0 for item in self.shape)
        ):
            raise CACHInventoryError(f"invalid shape for {self.fully_qualified_name}")
        expected_numel = 1
        for item in self.shape:
            expected_numel *= item
        if type(self.numel) is not int or self.numel != expected_numel:
            raise CACHInventoryError(f"numel differs for {self.fully_qualified_name}")
        if type(self.requires_grad) is not bool:
            raise CACHInventoryError(
                f"requires_grad must be boolean for {self.fully_qualified_name}"
            )
        if self.kind == "buffer" and self.requires_grad:
            raise CACHInventoryError(f"buffer cannot require grad: {self.fully_qualified_name}")
        if self.kind == "buffer" and self.optimizer_group_or_null is not None:
            raise CACHInventoryError(
                f"buffer cannot enter an optimizer: {self.fully_qualified_name}"
            )
        if self.optimizer_group_or_null is not None and (
            type(self.optimizer_group_or_null) is not str
            or not self.optimizer_group_or_null
        ):
            raise CACHInventoryError(
                f"optimizer group differs for {self.fully_qualified_name}"
            )
        _digest(self.init_tensor_sha256, f"{self.fully_qualified_name} init digest")
        if self.source_external_or_random not in _SOURCES:
            raise CACHInventoryError(
                f"invalid source for {self.fully_qualified_name}"
            )

    def canonical_dict(self) -> dict:
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def name_set_sha256(names: Iterable[str]) -> str:
    canonical = sorted(names, key=lambda item: item.encode("utf-8"))
    if len(canonical) != len(set(canonical)):
        raise CACHInventoryError("inventory names must be unique")
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def validate_inventory(
    entries: Iterable[InventoryEntry],
    *,
    expected_names: set[str] | None = None,
) -> tuple[InventoryEntry, ...]:
    """Validate freeze/optimizer ownership and return canonical name order."""

    ordered = tuple(
        sorted(entries, key=lambda entry: entry.fully_qualified_name.encode("utf-8"))
    )
    names = [entry.fully_qualified_name for entry in ordered]
    if len(names) != len(set(names)):
        raise CACHInventoryError("inventory names must be unique")
    if expected_names is not None and set(names) != expected_names:
        raise CACHInventoryError("observed inventory name set differs")

    for entry in ordered:
        external = entry.architecture_role in {"vae", "text_encoder"}
        if external:
            if entry.source_external_or_random != "source_external":
                raise CACHInventoryError(
                    f"external encoder source differs: {entry.fully_qualified_name}"
                )
            if entry.kind == "parameter" and entry.requires_grad:
                raise CACHInventoryError(
                    f"external encoder is trainable: {entry.fully_qualified_name}"
                )
            if entry.optimizer_group_or_null is not None:
                raise CACHInventoryError(
                    f"external encoder entered optimizer: {entry.fully_qualified_name}"
                )
            continue

        if entry.source_external_or_random != "random":
            raise CACHInventoryError(
                f"complete-random component has external source: "
                f"{entry.fully_qualified_name}"
            )
        if entry.kind == "parameter":
            if not entry.requires_grad:
                raise CACHInventoryError(
                    f"non-external parameter is frozen: {entry.fully_qualified_name}"
                )
            if entry.optimizer_group_or_null is None:
                raise CACHInventoryError(
                    f"trainable parameter is absent from optimizer: "
                    f"{entry.fully_qualified_name}"
                )
    return ordered


def build_inventory_document(
    entries: Iterable[InventoryEntry],
    *,
    candidate_revision: str,
    resolved_config_sha256: str,
    source_manifest_sha256: str,
    expected_key_schema_sha256: str,
) -> dict:
    ordered = validate_inventory(entries)
    for name, value in {
        "resolved_config_sha256": resolved_config_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "expected_key_schema_sha256": expected_key_schema_sha256,
    }.items():
        _digest(value, name)
    if type(candidate_revision) is not str or not candidate_revision:
        raise CACHInventoryError("candidate_revision must be non-empty")

    trainable = [
        item.fully_qualified_name
        for item in ordered
        if item.kind == "parameter" and item.requires_grad
    ]
    frozen = [
        item.fully_qualified_name
        for item in ordered
        if item.kind == "parameter" and not item.requires_grad
    ]
    buffers = [
        item.fully_qualified_name for item in ordered if item.kind == "buffer"
    ]
    return {
        "schema": "cach.model_inventory.v1",
        "candidate_revision": candidate_revision,
        "resolved_config_sha256": resolved_config_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "expected_key_schema_sha256": expected_key_schema_sha256,
        "trainable_name_set_sha256": name_set_sha256(trainable),
        "frozen_name_set_sha256": name_set_sha256(frozen),
        "buffer_name_set_sha256": name_set_sha256(buffers),
        "trainable_numel": sum(
            item.numel
            for item in ordered
            if item.kind == "parameter" and item.requires_grad
        ),
        "frozen_numel": sum(
            item.numel
            for item in ordered
            if item.kind == "parameter" and not item.requires_grad
        ),
        "entries": [item.canonical_dict() for item in ordered],
    }


def inventory_document_sha256(value: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "CACHInventoryError",
    "InventoryEntry",
    "build_inventory_document",
    "canonical_json_bytes",
    "inventory_document_sha256",
    "name_set_sha256",
    "validate_inventory",
]
