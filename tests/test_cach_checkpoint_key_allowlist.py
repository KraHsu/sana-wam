"""Pure fake-tensor tests for the exact CACH checkpoint contract."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sana_wam.cach.checkpoint_schema import (
    CACHCheckpointError,
    load_exact_checkpoint,
    schema_from_inventory,
    validate_checkpoint_state,
)
from sana_wam.cach.model_inventory import InventoryEntry
from sana_wam.cach.verifier import (
    validate_exact_checkpoint_locator,
    verify_checkpoint_contract,
)
from sana_wam.deploy.cach_model_loader import CACHDeployCheckpointDescriptor

_ZERO = "0" * 64
_ONE = "1" * 64


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str):
        self.shape = shape
        self.dtype = dtype

    def numel(self) -> int:
        result = 1
        for item in self.shape:
            result *= item
        return result


@dataclass
class FakeLoadResult:
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()


class FakeModule:
    def __init__(self, result: FakeLoadResult | None = None):
        self.result = result or FakeLoadResult()
        self.strict_values: list[bool] = []

    def load_state_dict(self, state, *, strict: bool):
        self.strict_values.append(strict)
        return self.result


def _entry(
    name: str,
    role: str,
    *,
    kind: str = "parameter",
    shape: tuple[int, ...] = (2, 3),
    requires_grad: bool = True,
    optimizer_group: str | None = "main",
    source: str = "random",
) -> InventoryEntry:
    numel = 1
    for item in shape:
        numel *= item
    return InventoryEntry(
        fully_qualified_name=name,
        kind=kind,
        architecture_role=role,
        shared_or_operator_specific="shared",
        shape=shape,
        dtype="float32",
        numel=numel,
        requires_grad=requires_grad,
        optimizer_group_or_null=optimizer_group,
        init_tensor_sha256=_ZERO,
        source_external_or_random=source,
    )


def _inventory() -> tuple[InventoryEntry, ...]:
    return (
        _entry("action_backbone.weight", "action_backbone"),
        _entry(
            "video_backbone.dit.position_ids",
            "video_dit",
            kind="buffer",
            shape=(3,),
            requires_grad=False,
            optimizer_group=None,
        ),
        _entry("video_backbone.dit.weight", "video_dit"),
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


def _state(schema) -> dict[str, FakeTensor]:
    return {
        entry.name: FakeTensor(entry.shape, entry.dtype)
        for entry in schema.entries
    }


def _locator(schema, **updates) -> dict:
    value = {
        "candidate_revision": schema.candidate_revision,
        "checkpoint_path": "/DATA/cach/campaign/checkpoint_step_12.safetensors",
        "checkpoint_raw_sha256": _ONE,
        "checkpoint_schema_sha256": schema.sha256,
        "endpoint_step": 12,
    }
    value.update(updates)
    return value


def test_schema_is_an_exact_sorted_allowlist_and_excludes_external_state() -> None:
    schema = schema_from_inventory(_inventory(), candidate_revision="candidate-v1")
    names = tuple(entry.name for entry in schema.entries)
    assert names == tuple(sorted(names))
    assert names == (
        "action_backbone.weight",
        "video_backbone.dit.position_ids",
        "video_backbone.dit.weight",
    )
    assert all("vae" not in name and "text_encoder" not in name for name in names)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unexpected",
        "substring_lookalike",
    ],
)
def test_missing_unexpected_and_substring_lookalikes_are_rejected(
    mutation: str,
) -> None:
    schema = schema_from_inventory(_inventory(), candidate_revision="candidate-v1")
    state = _state(schema)
    first = schema.entries[0].name
    if mutation == "missing":
        del state[first]
    elif mutation == "unexpected":
        state["legacy.extra.weight"] = FakeTensor((2, 3), "float32")
    else:
        state[first + ".adapter"] = state.pop(first)
    with pytest.raises(CACHCheckpointError, match="key set differs"):
        validate_checkpoint_state(state, schema)


@pytest.mark.parametrize(
    "replacement",
    [
        FakeTensor((3, 2), "float32"),
        FakeTensor((2, 3), "float16"),
    ],
)
def test_per_key_shape_and_dtype_must_match(replacement: FakeTensor) -> None:
    schema = schema_from_inventory(_inventory(), candidate_revision="candidate-v1")
    state = _state(schema)
    state[schema.entries[0].name] = replacement
    with pytest.raises(CACHCheckpointError, match="metadata differs"):
        validate_checkpoint_state(state, schema)


def test_loader_invokes_only_strict_state_loading() -> None:
    schema = schema_from_inventory(_inventory(), candidate_revision="candidate-v1")
    module = FakeModule()
    load_exact_checkpoint(module, _state(schema), schema)
    assert module.strict_values == [True]

    bad = FakeModule(FakeLoadResult(missing_keys=("dit.weight",)))
    with pytest.raises(CACHCheckpointError, match="returned mismatches"):
        load_exact_checkpoint(bad, _state(schema), schema)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"candidate_revision": "other"}, "revision differs"),
        ({"checkpoint_schema_sha256": _ZERO}, "schema digest differs"),
        ({"endpoint_step": 13}, "path endpoint step differs"),
        (
            {
                "checkpoint_path": (
                    "/DATA/cach/campaign/checkpoint_step_13.safetensors"
                )
            },
            "path endpoint step differs",
        ),
        (
            {"checkpoint_path": "/DATA/cach/latest/checkpoint.safetensors"},
            "cannot select latest",
        ),
        (
            {"checkpoint_path": "/DATA/cach/checkpoint_*.safetensors"},
            "glob syntax",
        ),
        ({"checkpoint_path": "relative/checkpoint.safetensors"}, "absolute path"),
        (
            {
                "checkpoint_path": (
                    "/DATA//cach/campaign/checkpoint_step_12.safetensors"
                )
            },
            "double slash",
        ),
        (
            {
                "checkpoint_path": (
                    "//DATA/cach/campaign/checkpoint_step_12.safetensors"
                )
            },
            "double slash",
        ),
        (
            {"checkpoint_path": "/DATA/cach/campaign/endpoint.safetensors"},
            "basename must be",
        ),
    ],
)
def test_checkpoint_locator_is_literal_and_revision_bound(
    updates: dict,
    message: str,
) -> None:
    schema = schema_from_inventory(_inventory(), candidate_revision="candidate-v1")
    with pytest.raises(CACHCheckpointError, match=message):
        validate_exact_checkpoint_locator(_locator(schema, **updates), schema)


def test_combined_verifier_closes_schema_inventory_state_and_locator() -> None:
    inventory = _inventory()
    schema = schema_from_inventory(inventory, candidate_revision="candidate-v1")
    report = verify_checkpoint_contract(
        _state(schema),
        schema,
        inventory,
        locator=_locator(schema),
    )
    assert report["checkpoint_state_exact"] is True
    assert report["external_vae_text_state_excluded"] is True
    assert report["checkpoint_schema_sha256"] == schema.sha256


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "relative/checkpoint.safetensors",
        "/DATA/cach/latest/checkpoint.safetensors",
        "/DATA/cach/checkpoint_*.safetensors",
        "/DATA/cach/../checkpoint.safetensors",
    ],
)
def test_stage1_deploy_descriptor_rejects_ambiguous_paths(path: str) -> None:
    with pytest.raises(ValueError):
        CACHDeployCheckpointDescriptor(
            candidate_revision="candidate-v1",
            endpoint_step=12,
            checkpoint_absolute_path=path,
            checkpoint_raw_sha256=_ZERO,
            checkpoint_schema_sha256=_ZERO,
            resolved_config_sha256=_ZERO,
            source_manifest_sha256=_ZERO,
            engine_schema_sha256=_ZERO,
        )


def test_stage1_deploy_descriptor_accepts_exact_path_step_binding() -> None:
    descriptor = CACHDeployCheckpointDescriptor(
        candidate_revision="candidate-v1",
        endpoint_step=12,
        checkpoint_absolute_path=(
            "/DATA/cach/campaign/checkpoint_step_12.safetensors"
        ),
        checkpoint_raw_sha256=_ZERO,
        checkpoint_schema_sha256=_ZERO,
        resolved_config_sha256=_ZERO,
        source_manifest_sha256=_ZERO,
        engine_schema_sha256=_ZERO,
    )
    assert descriptor.endpoint_step == 12


@pytest.mark.parametrize(
    ("endpoint_step", "path"),
    [
        (13, "/DATA/cach/campaign/checkpoint_step_12.safetensors"),
        (12, "/DATA/cach/campaign/checkpoint_step_13.safetensors"),
        (12, "/DATA//cach/campaign/checkpoint_step_12.safetensors"),
        (12, "//DATA/cach/campaign/checkpoint_step_12.safetensors"),
        (12, "/DATA/cach/campaign/endpoint.safetensors"),
    ],
)
def test_stage1_deploy_descriptor_binds_path_to_endpoint_step(
    endpoint_step: int,
    path: str,
) -> None:
    with pytest.raises(ValueError):
        CACHDeployCheckpointDescriptor(
            candidate_revision="candidate-v1",
            endpoint_step=endpoint_step,
            checkpoint_absolute_path=path,
            checkpoint_raw_sha256=_ZERO,
            checkpoint_schema_sha256=_ZERO,
            resolved_config_sha256=_ZERO,
            source_manifest_sha256=_ZERO,
            engine_schema_sha256=_ZERO,
        )
