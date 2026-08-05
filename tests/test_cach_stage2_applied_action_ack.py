"""CPU tests for the independent APPLIED_ACTION_ACK v1 verifier."""

from __future__ import annotations

import base64
import ast
import hashlib
import json
import os
import struct
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sana_wam.cach.applied_action_ack import (
    AppliedActionAckContext,
    AppliedActionAckError,
    AppliedActionAckReplayRegistry,
    ImmutableAuthorityRegistry,
    ImmutableObjectAuthority,
    compute_ack_payload_digest,
    verify_applied_action_ack,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


LAYOUT_DIGEST = _digest("layout")
TRANSFORM_DIGEST = _digest("transform")
OBSERVATION_DIGEST = _digest("observation-interval")
REPRESENTATION_ID = "robotwin.eef16.canonical.v1"
UNITS_ID = "robotwin.eef16.si.v1"


def _raw(values: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _context() -> AppliedActionAckContext:
    return AppliedActionAckContext(
        episode_id="episode-7",
        episode_epoch=7,
        pending_command_id="command-3",
        next_chunk_id=3,
        layout_instance_digest=LAYOUT_DIGEST,
        applied_count=2,
        action_dim=2,
        action_representation_id=REPRESENTATION_ID,
        action_units_id=UNITS_ID,
        controller_sequence_start=40,
        controller_transform_allowlist=frozenset({TRANSFORM_DIGEST}),
        observation_interval_digest=OBSERVATION_DIGEST,
        commanded_tensor=torch.tensor(
            [[91.0, 92.0], [93.0, 94.0]], dtype=torch.float32
        ),
    )


def _redigest(payload: dict) -> dict:
    payload["ack_payload_digest"] = compute_ack_payload_digest(payload)
    return payload


def _payload(
    raw: bytes | None = None,
    *,
    source: dict | None = None,
) -> dict:
    raw = _raw() if raw is None else raw
    if source is None:
        source = {
            "kind": "inline_tensor",
            "data_base64": base64.b64encode(raw).decode("ascii"),
        }
    return _redigest(
        {
            "schema": "cach.applied_action_ack.v1",
            "ack_id": "ack-3",
            "episode_id": "episode-7",
            "episode_epoch": 7,
            "command_id": "command-3",
            "chunk_id": 3,
            "layout_instance_digest": LAYOUT_DIGEST,
            "canonical_applied_tensor_or_immutable_ref": source,
            "dtype": "float32_le",
            "shape": [2, 2],
            "token_order": "layout_action_token_ascending_v1",
            "action_representation_id": REPRESENTATION_ID,
            "action_units_id": UNITS_ID,
            "action_values_digest": hashlib.sha256(raw).hexdigest(),
            "applied_count": 2,
            "controller_transform_digest": TRANSFORM_DIGEST,
            "observation_interval_digest": OBSERVATION_DIGEST,
            "controller_sequence_start": 40,
            "controller_sequence_end_exclusive": 42,
            "controller_monotonic_start_ns": 1_000,
            "controller_monotonic_end_ns": 2_000,
            "completion_status": "fully_applied",
            "ack_payload_digest": "",
        }
    )


def _verify(
    payload: dict,
    *,
    context: AppliedActionAckContext | None = None,
    replay: AppliedActionAckReplayRegistry | None = None,
    immutable: ImmutableAuthorityRegistry | None = None,
):
    return verify_applied_action_ack(
        payload,
        context=_context() if context is None else context,
        replay_registry=AppliedActionAckReplayRegistry() if replay is None else replay,
        immutable_registry=immutable,
    )


def _assert_code(code: str, payload: dict, **kwargs) -> None:
    with pytest.raises(AppliedActionAckError) as caught:
        _verify(payload, **kwargs)
    assert caught.value.code == code
    assert str(caught.value).startswith(f"{code}:")


def test_inline_applied_bytes_win_when_command_differs() -> None:
    payload = _payload()
    result = _verify(payload)

    torch.testing.assert_close(
        result.canonical_applied_tensor,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        atol=0,
        rtol=0,
    )
    assert result.commanded_differs_from_applied is True
    assert result.canonical_applied_bytes == _raw()
    assert result.canonical_applied_bytes != _raw((91.0, 92.0, 93.0, 94.0))
    assert result.action_values_digest == hashlib.sha256(_raw()).hexdigest()
    assert result.is_duplicate is False


def test_ack_payload_digest_has_exact_documented_canonical_json_definition() -> None:
    payload = _payload()
    without_digest = dict(payload)
    del without_digest["ack_payload_digest"]
    expected = json.dumps(
        without_digest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert not expected.endswith(b"\n")
    assert compute_ack_payload_digest(payload) == hashlib.sha256(expected).hexdigest()


def test_command_only_and_digest_only_cannot_substitute_for_applied_bytes() -> None:
    command_only = _payload()
    del command_only["canonical_applied_tensor_or_immutable_ref"]
    _redigest(command_only)
    _assert_code("ACTION_EVIDENCE_COMMANDED_ONLY", command_only)

    digest_only = _payload(source={"kind": "digest_only"})
    _assert_code("ACK_CANONICAL_BYTES_REQUIRED", digest_only)


def test_json_numeric_tensor_and_bad_base64_are_rejected() -> None:
    numeric = _payload(
        source={"kind": "inline_tensor", "data_base64": [1.0, 2.0, 3.0, 4.0]}
    )
    _assert_code("ACK_JSON_NUMERIC_TENSOR_REJECTED", numeric)

    invalid = _payload(
        source={"kind": "inline_tensor", "data_base64": "not@@base64"}
    )
    _assert_code("ACK_INLINE_BASE64_INVALID", invalid)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (struct.pack("<4f", float("nan"), 2.0, 3.0, 4.0), "ACK_ACTION_NONFINITE"),
        (struct.pack("<4f", float("inf"), 2.0, 3.0, 4.0), "ACK_ACTION_NONFINITE"),
        (struct.pack("<I3f", 0x80000000, 2.0, 3.0, 4.0), "ACK_ACTION_NEGATIVE_ZERO"),
    ],
)
def test_nonfinite_and_negative_zero_canonical_bytes_are_rejected(
    raw: bytes, code: str
) -> None:
    _assert_code(code, _payload(raw))


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("schema", "cach.applied_action_ack.v2", "ACK_SCHEMA_VERSION_MISMATCH"),
        ("episode_id", "episode-old", "ACK_EPISODE_MISMATCH"),
        ("episode_epoch", 6, "ACK_EPISODE_MISMATCH"),
        ("command_id", "command-stale", "ACK_COMMAND_UNKNOWN_OR_STALE"),
        ("chunk_id", 4, "ACK_CHUNK_MISMATCH"),
        ("layout_instance_digest", _digest("other-layout"), "ACK_LAYOUT_MISMATCH"),
        ("completion_status", "partially_applied", "ACK_COMPLETION_INCOMPLETE"),
        ("dtype", "float32_be", "ACK_DTYPE_MISMATCH"),
        ("shape", [4], "ACK_SHAPE_MISMATCH"),
        ("token_order", "controller_order", "ACK_TOKEN_ORDER_MISMATCH"),
        ("action_representation_id", "other-representation", "ACK_REPRESENTATION_MISMATCH"),
        ("action_units_id", "other-units", "ACK_UNITS_MISMATCH"),
        ("action_values_digest", _digest("wrong-values"), "ACK_ACTION_VALUES_DIGEST_MISMATCH"),
        ("applied_count", 1, "ACK_APPLIED_COUNT_MISMATCH"),
        ("controller_sequence_start", 39, "ACK_CONTROLLER_SEQUENCE_NONCONTIGUOUS"),
        ("controller_sequence_end_exclusive", 43, "ACK_CONTROLLER_SEQUENCE_NONCONTIGUOUS"),
        ("controller_transform_digest", _digest("unregistered-transform"), "ACK_CONTROLLER_TRANSFORM_UNREGISTERED"),
        ("observation_interval_digest", _digest("other-observation"), "ACK_OBSERVATION_INTERVAL_MISMATCH"),
    ],
)
def test_identity_layout_tensor_and_interval_contracts_fail_closed(
    field: str, value: object, code: str
) -> None:
    payload = _payload()
    payload[field] = value
    _redigest(payload)
    _assert_code(code, payload)


def test_root_and_branch_keys_are_exact() -> None:
    root_extra = _payload()
    root_extra["diagnostic"] = True
    _redigest(root_extra)
    _assert_code("ACK_SCHEMA_KEYS_MISMATCH", root_extra)

    branch_extra = _payload()
    branch_extra["canonical_applied_tensor_or_immutable_ref"]["object_sha256"] = _digest(
        "not-an-inline-field"
    )
    _redigest(branch_extra)
    _assert_code("ACK_INLINE_SCHEMA_MISMATCH", branch_extra)


def test_payload_digest_is_recomputed_after_all_content_checks() -> None:
    payload = _payload()
    payload["ack_payload_digest"] = "0" * 64
    _assert_code("ACK_PAYLOAD_DIGEST_MISMATCH", payload)


@pytest.mark.parametrize("invalid_end", [999, 1_000])
def test_controller_sequence_and_monotonic_interval_are_contiguous(
    invalid_end: int,
) -> None:
    payload = _payload()
    payload["controller_monotonic_end_ns"] = invalid_end
    _redigest(payload)
    _assert_code("ACK_CONTROLLER_MONOTONIC_INTERVAL_INVALID", payload)


def test_identical_duplicate_is_idempotent_but_conflicting_ack_id_fails() -> None:
    replay = AppliedActionAckReplayRegistry()
    first = _verify(_payload(), replay=replay)
    duplicate = _verify(_payload(), replay=replay)
    assert first.is_duplicate is False
    assert duplicate.is_duplicate is True

    conflict = _payload()
    conflict["controller_monotonic_end_ns"] = 2_001
    _redigest(conflict)
    _assert_code("ACK_ID_CONFLICT", conflict, replay=replay)


def _authority(
    uri: str,
    path: Path,
    *,
    object_size: int | None = None,
    object_sha256: str | None = None,
) -> ImmutableAuthorityRegistry:
    data = path.read_bytes() if path.is_file() else b""
    return ImmutableAuthorityRegistry(
        entries={
            uri: ImmutableObjectAuthority(
                path=path.absolute(),
                object_size=len(data) if object_size is None else object_size,
                object_sha256=(
                    hashlib.sha256(data).hexdigest()
                    if object_sha256 is None
                    else object_sha256
                ),
            )
        }
    )


def _ref_payload(uri: str, whole: bytes, offset: int = 0) -> dict:
    raw = _raw()
    return _payload(
        raw,
        source={
            "kind": "immutable_ref",
            "uri": uri,
            "byte_offset": offset,
            "byte_length": len(raw),
            "object_sha256": hashlib.sha256(whole).hexdigest(),
        },
    )


def test_registered_immutable_ref_uses_stable_exact_range(tmp_path: Path) -> None:
    prefix = b"hdr"
    whole = prefix + _raw() + b"tail"
    path = tmp_path / "actions.bin"
    path.write_bytes(whole)
    uri = "authority://stage2/actions-3"
    result = _verify(
        _ref_payload(uri, whole, len(prefix)),
        immutable=_authority(uri, path),
    )
    assert result.canonical_applied_bytes == _raw()
    torch.testing.assert_close(
        result.canonical_applied_tensor,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        atol=0,
        rtol=0,
    )


def test_immutable_ref_requires_registered_uri_and_object_digest(tmp_path: Path) -> None:
    path = tmp_path / "actions.bin"
    path.write_bytes(_raw())
    uri = "authority://stage2/actions-3"
    _assert_code(
        "ACK_IMMUTABLE_URI_UNREGISTERED",
        _ref_payload(uri, _raw()),
        immutable=ImmutableAuthorityRegistry.empty(),
    )

    missing_digest = _ref_payload(uri, _raw())
    del missing_digest["canonical_applied_tensor_or_immutable_ref"]["object_sha256"]
    _redigest(missing_digest)
    _assert_code(
        "ACK_IMMUTABLE_SCHEMA_MISMATCH",
        missing_digest,
        immutable=_authority(uri, path),
    )


def test_immutable_ref_rejects_symlink_changed_object_and_short_range(
    tmp_path: Path,
) -> None:
    uri = "authority://stage2/actions-3"
    real = tmp_path / "real.bin"
    real.write_bytes(_raw())
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    _assert_code(
        "ACK_IMMUTABLE_REF_SYMLINK",
        _ref_payload(uri, _raw()),
        immutable=_authority(uri, link),
    )

    changed = tmp_path / "changed.bin"
    changed.write_bytes(_raw())
    changed_registry = _authority(uri, changed)
    changed.write_bytes(_raw((5.0, 6.0, 7.0, 8.0)))
    _assert_code(
        "ACK_IMMUTABLE_OBJECT_MISMATCH",
        _ref_payload(uri, _raw()),
        immutable=changed_registry,
    )

    short = tmp_path / "short.bin"
    short.write_bytes(_raw()[:4])
    short_data = short.read_bytes()
    short_payload = _payload(
        source={
            "kind": "immutable_ref",
            "uri": uri,
            "byte_offset": 0,
            "byte_length": len(_raw()),
            "object_sha256": hashlib.sha256(short_data).hexdigest(),
        }
    )
    _assert_code(
        "ACK_IMMUTABLE_SHORT_READ",
        short_payload,
        immutable=_authority(uri, short),
    )


def test_context_binds_layout_coverage_and_pending_command_tensor() -> None:
    context = _context()
    with pytest.raises(AppliedActionAckError) as caught:
        replace(context, commanded_tensor=torch.ones(1, 4, dtype=torch.float32))
    assert caught.value.code == "ACK_CONTEXT_INVALID"

    with pytest.raises(AppliedActionAckError) as caught:
        replace(context, controller_transform_allowlist=frozenset())
    assert caught.value.code == "ACK_CONTEXT_INVALID"


def test_source_has_no_hybrid_cache_manager_or_deploy_commit_edge() -> None:
    source = Path(__file__).parents[1] / "src/sana_wam/cach/applied_action_ack.py"
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any("hybrid_cache" in name for name in imported_modules)
    assert not any(name.startswith("sana_wam.deploy") for name in imported_modules)
