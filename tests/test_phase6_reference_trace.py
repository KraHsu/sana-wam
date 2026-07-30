from __future__ import annotations

import hashlib

import pytest
import torch

from sana_wam.model.phase6_reference_trace import tensor_descriptor, trace_sha256


def test_bfloat16_descriptor_hashes_original_raw_bits_without_float32_cast():
    value = torch.tensor([1.0, -2.5, 0.0], dtype=torch.bfloat16)
    raw = value.contiguous().view(torch.uint8).numpy().tobytes(order="C")

    descriptor = tensor_descriptor(value)

    assert descriptor == {
        "device_class": "cpu",
        "dtype": "torch.bfloat16",
        "layout": "torch.strided",
        "numel": 3,
        "raw_bits_sha256": hashlib.sha256(raw).hexdigest(),
        "shape": [3],
        "stride": [1],
    }
    assert len(raw) == value.numel() * 2


def test_descriptor_binds_original_stride_but_hashes_logical_tensor_order():
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    noncontiguous = base[:, ::2]
    contiguous = noncontiguous.clone()

    left = tensor_descriptor(noncontiguous)
    right = tensor_descriptor(contiguous)

    assert left["raw_bits_sha256"] == right["raw_bits_sha256"]
    assert left["shape"] == right["shape"] == [3, 2]
    assert left["stride"] != right["stride"]
    assert trace_sha256("test", {"value": noncontiguous}) != trace_sha256(
        "test", {"value": contiguous}
    )


def test_trace_is_key_order_independent_and_metadata_sensitive():
    first = torch.tensor([1, 2], dtype=torch.int64)
    second = torch.tensor([True, False])
    left = trace_sha256(
        "paired-input",
        {"z": second, "a": first},
        metadata={"domain_seeds": {"video-noise": 7}},
    )
    right = trace_sha256(
        "paired-input",
        {"a": first, "z": second},
        metadata={"domain_seeds": {"video-noise": 7}},
    )
    changed = trace_sha256(
        "paired-input",
        {"a": first, "z": second},
        metadata={"domain_seeds": {"video-noise": 8}},
    )
    assert left == right
    assert left != changed


def test_trace_accepts_only_canonical_metadata_and_is_type_sensitive():
    metadata = {
        "boolean": False,
        "float": 0.0,
        "integer": 0,
        "list": [None, "value"],
        "nested": {"seed": 7},
    }

    first = trace_sha256("metadata", {}, metadata=metadata)
    second = trace_sha256("metadata", {}, metadata=dict(reversed(metadata.items())))

    assert first == second
    assert trace_sha256("metadata", {}, metadata={"value": False}) != trace_sha256(
        "metadata", {}, metadata={"value": 0}
    )
    assert trace_sha256("metadata", {}, metadata={"value": 0}) != trace_sha256(
        "metadata", {}, metadata={"value": 0.0}
    )


@pytest.mark.parametrize(
    ("metadata", "error", "match"),
    (
        ({"bad": torch.tensor(1)}, TypeError, "metadata.bad.*Tensor"),
        ({"bad": (1, 2)}, TypeError, "metadata.bad.*tuple"),
        ({"bad": {1, 2}}, TypeError, "metadata.bad.*set"),
        ({"bad": b"value"}, TypeError, "metadata.bad.*bytes"),
        ({1: "value"}, TypeError, "keys must be non-empty strings"),
        ({"": "value"}, TypeError, "keys must be non-empty strings"),
        ({"bad": float("nan")}, ValueError, "metadata.bad.*finite"),
        ({"bad": float("inf")}, ValueError, "metadata.bad.*finite"),
    ),
)
def test_trace_rejects_noncanonical_metadata(metadata, error, match):
    with pytest.raises(error, match=match):
        trace_sha256("metadata", {}, metadata=metadata)


def test_descriptor_rejects_non_tensor_and_sparse_layout():
    with pytest.raises(TypeError, match="tensors or None"):
        tensor_descriptor("not-a-tensor")
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2)
    )
    with pytest.raises(ValueError, match="strided"):
        tensor_descriptor(sparse)


def test_trace_non_tensor_error_names_the_offending_key_and_type():
    with pytest.raises(TypeError, match="bad_value.*str.*tensors or None"):
        trace_sha256("test", {"good": None, "bad_value": "not-a-tensor"})
