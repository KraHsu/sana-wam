from __future__ import annotations

import hashlib
import os

import pytest
import torch
from safetensors.torch import save_file

from sana_wam.train.action_facing_cache_reference import (
    AFCC_REFERENCE_SCHEMA_VERSION,
    AFCC_REFERENCE_TENSOR_KEY,
    ActionFacingCacheReferenceError,
    ActionFacingCacheReferenceStore,
    canonical_json_bytes,
    tensor_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _build_store(tmp_path, *, corrupt_manifest_sha: bool = False):
    shape = (2, 3, 4, 5, 6)
    rows = []
    expected_tensors = []
    for step in (1, 2):
        tensor = torch.arange(
            torch.tensor(shape).prod().item(), dtype=torch.float32
        ).reshape(shape).to(torch.bfloat16)
        tensor = tensor + step
        path = tmp_path / f"row_{step:03d}.safetensors"
        save_file({AFCC_REFERENCE_TENSOR_KEY: tensor}, path)
        os.chmod(path, 0o400)
        payload = path.read_bytes()
        rows.append(
            {
                "action_sigma": 1.0 if step == 1 else 0.9,
                "common_input_trace_sha256": _sha(f"trace-{step}"),
                "dataset_contract_row_sha256": _sha(f"dataset-{step}"),
                "dataset_index": step - 1,
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "global_step": step,
                "path": str(path.resolve()),
                "plan_row_sha256": _sha(f"plan-row-{step}"),
                "size_bytes": len(payload),
                "task_name": f"task-{step}",
                "tensor_sha256": tensor_sha256(tensor),
                "t0_reference_forward_trace_sha256": _sha(f"t0-{step}"),
            }
        )
        expected_tensors.append(tensor)
    manifest = {
        "action_stats_sha256": _sha("stats"),
        "dataset_contract_artifact_sha256": _sha("dataset"),
        "identity_sha256": _sha("identity"),
        "plan_sha256": _sha("plan"),
        "reference_checkpoint_sha256": _sha("checkpoint"),
        "row_count": 2,
        "rows": rows,
        "scalar_reference_artifact_sha256": _sha("scalar-reference"),
        "schema_version": AFCC_REFERENCE_SCHEMA_VERSION,
        "source_manifest_sha256": _sha("source"),
        "precompute_config_file_sha256": _sha("config"),
        "precompute_gpu_runtime": {"device": "test"},
        "reference_execution_authority_sha256": _sha("execution-authority"),
        "tensor_contract": {
            "dtype": "BF16",
            "key": AFCC_REFERENCE_TENSOR_KEY,
            "shape": list(shape),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    os.chmod(manifest_path, 0o400)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if corrupt_manifest_sha:
        manifest_sha = "0" * 64
    store = ActionFacingCacheReferenceStore(
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha,
        expected_plan_sha256=_sha("plan"),
        expected_identity_sha256=_sha("identity"),
        expected_dataset_contract_sha256=_sha("dataset"),
        expected_reference_checkpoint_sha256=_sha("checkpoint"),
        expected_action_stats_sha256=_sha("stats"),
        expected_source_manifest_sha256=_sha("source"),
        expected_row_count=2,
        expected_shape=shape,
    )
    return store, expected_tensors


def test_reference_store_lazily_loads_exact_bf16_rows(tmp_path):
    store, expected = _build_store(tmp_path)
    assert store.common_trace_for_global_step(2) == _sha("trace-2")
    assert store.audit_metadata_for_global_step(2) == {
        "action_sigma": 0.9,
        "plan_row_sha256": _sha("plan-row-2"),
        "task_name": "task-2",
    }
    for step, tensor in enumerate(expected, start=1):
        observed = store.tensor_for_global_step(step)
        assert observed.dtype == torch.bfloat16
        assert observed.requires_grad is False
        assert torch.equal(observed, tensor)


def test_reference_store_rejects_manifest_sha_drift(tmp_path):
    with pytest.raises(ActionFacingCacheReferenceError, match="SHA256 differs"):
        _build_store(tmp_path, corrupt_manifest_sha=True)
