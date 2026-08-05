"""Pure metadata tests for CACH trainable/frozen/optimizer closure."""

from __future__ import annotations

from dataclasses import replace

import pytest

from sana_wam.cach.initialization import (
    compare_shared_initialization,
    validate_fresh_initialization_fields,
)
from sana_wam.cach.model_inventory import (
    CACHInventoryError,
    InventoryEntry,
    name_set_sha256,
    validate_inventory,
)
from sana_wam.cach.verifier import (
    validate_inventory_pair,
    validate_optimizer_membership,
)

_SHARED_DIGEST = "1" * 64
_OPERATOR_DIGEST = "2" * 64


def _entry(
    name: str,
    role: str,
    *,
    ownership: str = "shared",
    kind: str = "parameter",
    requires_grad: bool = True,
    optimizer_group: str | None = "main",
    source: str = "random",
    digest: str = _SHARED_DIGEST,
) -> InventoryEntry:
    return InventoryEntry(
        fully_qualified_name=name,
        kind=kind,
        architecture_role=role,
        shared_or_operator_specific=ownership,
        shape=(2, 2),
        dtype="float32",
        numel=4,
        requires_grad=requires_grad,
        optimizer_group_or_null=optimizer_group,
        init_tensor_sha256=digest,
        source_external_or_random=source,
    )


def _shared_entries() -> tuple[InventoryEntry, ...]:
    return (
        _entry("action_backbone.weight", "action_backbone"),
        _entry("proprio_encoder.weight", "proprio_encoder"),
        _entry("video_backbone.dit.block.gdn.weight", "gdn"),
        _entry("video_backbone.dit.weight", "video_dit"),
        _entry(
            "video_backbone.dit.position_ids",
            "video_dit",
            kind="buffer",
            requires_grad=False,
            optimizer_group=None,
        ),
        _entry(
            "video_backbone.text_encoder.weight",
            "text_encoder",
            requires_grad=False,
            optimizer_group=None,
            source="source_external",
        ),
        _entry(
            "video_backbone.vae.weight",
            "vae",
            requires_grad=False,
            optimizer_group=None,
            source="source_external",
        ),
    )


def _candidate_entries() -> tuple[InventoryEntry, ...]:
    return _shared_entries() + (
        _entry(
            "video_backbone.dit.action_conditioner.proj.weight",
            "action_conditioner",
            ownership="operator_specific",
            optimizer_group="conditioner",
            digest=_OPERATOR_DIGEST,
        ),
    )


def _groups(entries: tuple[InventoryEntry, ...]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for entry in entries:
        if entry.optimizer_group_or_null is None:
            continue
        result.setdefault(entry.optimizer_group_or_null, []).append(
            entry.fully_qualified_name
        )
    return result


def test_only_vae_and_text_are_frozen_and_every_hybrid_parameter_is_trainable() -> None:
    ordered = validate_inventory(_candidate_entries())
    frozen = {
        entry.architecture_role
        for entry in ordered
        if entry.kind == "parameter" and not entry.requires_grad
    }
    trainable = {
        entry.architecture_role
        for entry in ordered
        if entry.kind == "parameter" and entry.requires_grad
    }
    assert frozen == {"vae", "text_encoder"}
    assert {
        "action_backbone",
        "action_conditioner",
        "gdn",
        "proprio_encoder",
        "video_dit",
    } <= trainable


@pytest.mark.parametrize(
    "bad_entry",
    [
        replace(
            _shared_entries()[0],
            requires_grad=False,
            optimizer_group_or_null=None,
        ),
        replace(
            _shared_entries()[-1],
            requires_grad=True,
            optimizer_group_or_null="main",
        ),
    ],
)
def test_frozen_allowlist_cannot_expand_or_external_encoders_become_trainable(
    bad_entry: InventoryEntry,
) -> None:
    entries = list(_shared_entries())
    entries[
        next(
            index
            for index, entry in enumerate(entries)
            if entry.fully_qualified_name == bad_entry.fully_qualified_name
        )
    ] = bad_entry
    with pytest.raises(CACHInventoryError):
        validate_inventory(entries)


def test_optimizer_parameter_membership_is_exact_and_single_group() -> None:
    entries = _candidate_entries()
    ordered = validate_optimizer_membership(entries, _groups(entries))
    expected = {
        entry.fully_qualified_name
        for entry in ordered
        if entry.kind == "parameter" and entry.requires_grad
    }
    assert expected == set().union(*_groups(entries).values())

    groups = _groups(entries)
    groups["main"].append("not.in.inventory")
    with pytest.raises(CACHInventoryError, match="parameter set differs"):
        validate_optimizer_membership(entries, groups)

    groups = _groups(entries)
    duplicate = groups["main"][0]
    groups["conditioner"].append(duplicate)
    with pytest.raises(CACHInventoryError, match="multiple optimizer groups"):
        validate_optimizer_membership(entries, groups)


def test_shared_initialization_and_operator_delta_are_pair_exact() -> None:
    reference = _shared_entries()
    candidate = _candidate_entries()
    reference_ordered, candidate_ordered = validate_inventory_pair(
        reference,
        candidate,
        expected_candidate_operator_names={
            "video_backbone.dit.action_conditioner.proj.weight"
        },
        reference_optimizer_groups=_groups(reference),
        candidate_optimizer_groups=_groups(candidate),
    )
    reference_shared = {
        entry.fully_qualified_name: entry.init_tensor_sha256
        for entry in reference_ordered
        if entry.shared_or_operator_specific == "shared"
    }
    candidate_shared = {
        entry.fully_qualified_name: entry.init_tensor_sha256
        for entry in candidate_ordered
        if entry.shared_or_operator_specific == "shared"
    }
    assert reference_shared == candidate_shared

    changed = tuple(
        replace(entry, init_tensor_sha256="3" * 64)
        if entry.fully_qualified_name == "video_backbone.dit.weight"
        else entry
        for entry in candidate
    )
    with pytest.raises(CACHInventoryError, match="initialization differs"):
        validate_inventory_pair(
            reference,
            changed,
            expected_candidate_operator_names={
                "video_backbone.dit.action_conditioner.proj.weight"
            },
            reference_optimizer_groups=_groups(reference),
            candidate_optimizer_groups=_groups(changed),
        )


@pytest.mark.parametrize(
    "metadata_update",
    [
        {"architecture_role": "proprio_encoder"},
        {"requires_grad": False},
        {"optimizer_group_or_null": "alternate"},
        {"source_external_or_random": "source_external"},
    ],
)
def test_shared_parity_covers_all_inventory_metadata(
    metadata_update: dict,
) -> None:
    reference = _shared_entries()
    candidate = tuple(
        replace(entry, **metadata_update)
        if entry.fully_qualified_name == "action_backbone.weight"
        else entry
        for entry in reference
    )
    with pytest.raises(CACHInventoryError, match="shared initialization differs"):
        compare_shared_initialization(reference, candidate)


def test_fresh_initialization_freeze_allowlist_is_literal() -> None:
    validate_fresh_initialization_fields(
        initialization_mode="complete_random_v1",
        model_path=None,
        init_dit_from=None,
        init_checkpoint=None,
        init_checkpoint_sha256=None,
        freeze=["video_backbone.vae", "video_backbone.text_encoder"],
    )
    with pytest.raises(CACHInventoryError, match="freeze allowlist"):
        validate_fresh_initialization_fields(
            initialization_mode="complete_random_v1",
            model_path=None,
            init_dit_from=None,
            init_checkpoint=None,
            init_checkpoint_sha256=None,
            freeze=[
                "video_backbone.dit",
                "video_backbone.vae",
                "video_backbone.text_encoder",
            ],
        )


def test_trainable_name_digest_is_order_independent_but_exact() -> None:
    names = [
        entry.fully_qualified_name
        for entry in _candidate_entries()
        if entry.kind == "parameter" and entry.requires_grad
    ]
    assert name_set_sha256(names) == name_set_sha256(reversed(names))
    assert name_set_sha256(names) != name_set_sha256(names[:-1])
