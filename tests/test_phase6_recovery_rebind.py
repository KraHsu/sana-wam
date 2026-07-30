from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import ast
import importlib.util
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "phase6_recovery_rebind_under_test",
    HERE.parent / "scripts" / "build_phase6_recovery_rebind.py",
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "0")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.delitem(sys.modules, "torch", raising=False)


def pair(
    purpose: str, *, historical: bool, label: str
) -> tuple[dict, dict, bytes, bytes]:
    lifecycle = builder._pair_lifecycle(purpose, historical=historical)
    registration = (
        "registered_before_recovery_cohort_started"
        if historical
        else "registered_before_phase6_training_results"
    )
    request = {
        "artifacts": {},
        "authorization": {"operation": purpose},
        **lifecycle,
        registration: True,
        "purpose": purpose,
        "schema_version": (
            builder.NEW_REQUEST_SCHEMA if historical else builder.OLD_REQUEST_SCHEMA
        ),
    }
    request_bytes = builder.canonical_json_bytes(request)
    report = {
        "artifacts": {},
        "authorization": deepcopy(request["authorization"]),
        **lifecycle,
        registration: True,
        "purpose": purpose,
        "request_sha256": sha256(request_bytes).hexdigest(),
        "schema_version": (
            builder.NEW_REPORT_SCHEMA if historical else builder.OLD_REPORT_SCHEMA
        ),
        "status": "pass",
        "synthetic_label": label,
    }
    return request, report, request_bytes, builder.canonical_json_bytes(report)


def test_legacy_pair_lifecycle_matches_registered_v5_shapes():
    reference = builder._pair_lifecycle("reference_precompute", historical=False)
    smoke = builder._pair_lifecycle("smoke", historical=False)

    assert reference == {
        "reference_gpu_started": False,
        "training_started": False,
    }
    assert "closed_loop_started" not in reference
    assert smoke == {
        "closed_loop_started": False,
        "formal_training_started": False,
        "reference_precompute_completed": True,
        "smoke_gpu_started": False,
    }


def synthetic_pin(path: Path, data: bytes, mode: int = 0o400) -> builder.Pin:
    return builder.Pin(str(path.resolve()), sha256(data).hexdigest(), len(data), mode)


def write_registered(pin: builder.Pin, data: bytes) -> None:
    Path(pin.path).write_bytes(data)
    os.chmod(pin.path, pin.mode)


def replace_registered(path: str, data: bytes, mode: int) -> builder.Pin:
    if Path(path).exists():
        os.chmod(path, 0o600)
    Path(path).write_bytes(data)
    os.chmod(path, mode)
    return synthetic_pin(Path(path), data, mode)


def old_reference_fixture(tmp_path: Path) -> dict:
    old_request, old_report, old_request_bytes, old_report_bytes = pair(
        "reference_precompute", historical=False, label="old-reference"
    )
    source_v4 = digest("source-v4")
    config = {
        "dataloader": {"batch_size": 1},
        "execution": {"forward_only": True, "training_started": False},
        "model": {"model": "real-2b"},
        "schema_version": "sana-phase6-action-reference-precompute-config-v1",
        "training": {
            "phase6_code_source_manifest": "/formal/source-v4.json",
            "phase6_code_source_manifest_sha256": source_v4,
        },
    }
    reference = {
        "action_stats_sha256": digest("action-stats"),
        "dataset_contract_artifact_sha256": digest("dataset"),
        "identity_sha256": digest("identity"),
        "plan_sha256": digest("plan"),
        "precompute_config": config,
        "precompute_config_file_sha256": digest("old-config-file"),
        "precompute_config_sha256": sha256(
            builder.canonical_json_bytes(config)
        ).hexdigest(),
        "precompute_gpu_runtime": {"device": "H200", "world_size": 1},
        "precompute_gpu_runtime_sha256": "",
        "precompute_preflight_report_sha256": sha256(old_report_bytes).hexdigest(),
        "precompute_preflight_request_sha256": sha256(old_request_bytes).hexdigest(),
        "precompute_source_manifest_sha256": source_v4,
        "reference_checkpoint_sha256": digest("phase1-checkpoint"),
        "rows": [],
        "schema_version": builder.REFERENCE_SCHEMA,
    }
    reference["precompute_gpu_runtime_sha256"] = sha256(
        builder.canonical_json_bytes(reference["precompute_gpu_runtime"])
    ).hexdigest()
    for index in range(504):
        row = {
            "action_sigma": [0.5, 0.9, 1.0][index % 3],
            "common_input_trace_sha256": digest(f"common:{index}"),
            "dataset_contract_row_sha256": digest(f"dataset-row:{index}"),
            "dataset_index": index,
            "error_float32_bits": f"{index:08x}",
            "global_step": index + 1,
            "plan_row_sha256": digest(f"plan-row:{index}"),
            "reference_forward_context_sha256": "",
            "t0_reference_forward_trace_sha256": digest(f"trace:{index}"),
            "task_name": f"task-{index % 42:02d}",
        }
        row["reference_forward_context_sha256"] = builder._new_context_sha(
            row, reference
        )
        reference["rows"].append(row)
    reference_bytes = builder.canonical_json_bytes(reference)
    reference_sha = sha256(reference_bytes).hexdigest()
    build = {
        "action_stats_sha256": reference["action_stats_sha256"],
        "closed_loop": False,
        "dataset_contract_artifact_sha256": reference[
            "dataset_contract_artifact_sha256"
        ],
        "gpu_forward_executed": True,
        "precompute_config_file_sha256": reference["precompute_config_file_sha256"],
        "precompute_config_path": "/formal/old-reference.yaml",
        "precompute_config_sha256": reference["precompute_config_sha256"],
        "precompute_gpu_runtime": reference["precompute_gpu_runtime"],
        "precompute_gpu_runtime_sha256": reference["precompute_gpu_runtime_sha256"],
        "precompute_preflight_report_sha256": reference[
            "precompute_preflight_report_sha256"
        ],
        "precompute_preflight_request_sha256": reference[
            "precompute_preflight_request_sha256"
        ],
        "precompute_source_manifest_sha256": source_v4,
        "reference_artifact_path": "/formal/reference-v4.json",
        "reference_artifact_sha256": reference_sha,
        "reference_checkpoint_sha256": reference["reference_checkpoint_sha256"],
        "row_count": 504,
        "schema_version": builder.REFERENCE_BUILD_SCHEMA,
        "training_started": False,
    }
    build_bytes = builder.canonical_json_bytes(build)
    spot_rows = []
    for step in builder.SPOT_STEPS:
        row = reference["rows"][step - 1]
        spot_rows.append(
            {
                "common_input_trace_sha256": row["common_input_trace_sha256"],
                "dataset_contract_row_sha256": row["dataset_contract_row_sha256"],
                "exact_match": True,
                "expected_error_float32_bits": row["error_float32_bits"],
                "global_step": step,
                "observed_error_float32_bits": row["error_float32_bits"],
                "plan_row_sha256": row["plan_row_sha256"],
                "reference_forward_context_sha256": row[
                    "reference_forward_context_sha256"
                ],
                "t0_reference_forward_trace_sha256": row[
                    "t0_reference_forward_trace_sha256"
                ],
            }
        )
    spot = {
        "action_stats_sha256": reference["action_stats_sha256"],
        "closed_loop": False,
        "dataset_contract_artifact_sha256": reference[
            "dataset_contract_artifact_sha256"
        ],
        "global_steps": list(builder.SPOT_STEPS),
        "gpu_forward_executed": True,
        "identity_sha256": reference["identity_sha256"],
        "plan_sha256": reference["plan_sha256"],
        "precompute_config_file_sha256": reference["precompute_config_file_sha256"],
        "precompute_config_sha256": reference["precompute_config_sha256"],
        "precompute_gpu_runtime_sha256": reference["precompute_gpu_runtime_sha256"],
        "precompute_preflight_report_sha256": reference[
            "precompute_preflight_report_sha256"
        ],
        "precompute_preflight_request_sha256": reference[
            "precompute_preflight_request_sha256"
        ],
        "precompute_source_manifest_sha256": source_v4,
        "reference_artifact_sha256": reference_sha,
        "reference_checkpoint_sha256": reference["reference_checkpoint_sha256"],
        "rows": spot_rows,
        "schema_version": builder.REFERENCE_SPOT_SCHEMA,
        "spot_gpu_runtime": {"device": "H200", "world_size": 1},
        "spot_gpu_runtime_sha256": digest("spot-runtime"),
        "spot_preflight_report_sha256": sha256(old_report_bytes).hexdigest(),
        "spot_preflight_request_sha256": sha256(old_request_bytes).hexdigest(),
        "training_started": False,
        "verification_passed": True,
    }
    spot_bytes = builder.canonical_json_bytes(spot)
    numeric = builder.project_reference_numeric(reference)
    spot_numeric = builder.project_reference_spot_numeric(spot)
    values = {
        "old_request": old_request,
        "old_request_bytes": old_request_bytes,
        "old_report": old_report,
        "old_report_bytes": old_report_bytes,
        "reference": reference,
        "reference_bytes": reference_bytes,
        "build": build,
        "build_bytes": build_bytes,
        "spot": spot,
        "spot_bytes": spot_bytes,
        "numeric": numeric,
        "spot_numeric": spot_numeric,
    }
    values["pins"] = {
        "reference_request": synthetic_pin(
            tmp_path / "old-ref-request", old_request_bytes, 0o444
        ),
        "reference_report": synthetic_pin(
            tmp_path / "old-ref-report", old_report_bytes, 0o600
        ),
        "reference": synthetic_pin(tmp_path / "old-reference", reference_bytes),
        "reference_build": synthetic_pin(tmp_path / "old-build", build_bytes),
        "reference_spot": synthetic_pin(tmp_path / "old-spot", spot_bytes),
        "reference_numeric_payload": synthetic_pin(tmp_path / "numeric", numeric),
        "reference_spot_numeric_payload": synthetic_pin(
            tmp_path / "spot-numeric", spot_numeric
        ),
    }
    return values


def new_pair(
    purpose: str, source_pin: builder.Pin, authority_pin: builder.Pin
) -> tuple[dict, dict]:
    request, report, _request_bytes, _report_bytes = pair(
        purpose, historical=True, label=f"new-{purpose}"
    )
    request["source_manifest"] = {
        "path": source_pin.path,
        "sha256": source_pin.sha256,
    }
    request["artifacts"] = {
        "recovery_authority": {
            "path": authority_pin.path,
            "sha256": authority_pin.sha256,
        }
    }
    report["artifacts"] = deepcopy(request["artifacts"])
    report["request_sha256"] = sha256(builder.canonical_json_bytes(request)).hexdigest()
    return request, report


def test_reference_rebind_preserves_504_and_spot_numeric_bytes(tmp_path: Path) -> None:
    old = old_reference_fixture(tmp_path)
    source = builder.Pin(
        str((tmp_path / "source-v6").resolve()), digest("source-v6"), 99, 0o400
    )
    authority = builder.Pin(
        str((tmp_path / "authority").resolve()), digest("authority"), 101, 0o400
    )
    request, report = new_pair("reference_precompute", source, authority)
    request["input_config"] = {
        "path": "/formal/reference-v8.yaml",
        "sha256": digest("new-config-file"),
    }
    report["request_sha256"] = sha256(builder.canonical_json_bytes(request)).hexdigest()
    config = {
        "dataloader": deepcopy(old["reference"]["precompute_config"]["dataloader"]),
        "model": deepcopy(old["reference"]["precompute_config"]["model"]),
        "training": {
            "phase6_code_source_manifest": source.path,
            "phase6_code_source_manifest_sha256": source.sha256,
        },
    }
    outputs = {
        "request": str((tmp_path / "reference-request-v7").resolve()),
        "report": str((tmp_path / "reference-report-v7").resolve()),
        "reference": str((tmp_path / "reference-v8").resolve()),
        "build": str((tmp_path / "build-v8").resolve()),
        "spot": str((tmp_path / "spot-v8").resolve()),
        "receipt": str((tmp_path / "reference-receipt").resolve()),
    }
    payloads = builder.build_reference_payloads(
        old_request_bytes=old["old_request_bytes"],
        old_report_bytes=old["old_report_bytes"],
        old_reference_bytes=old["reference_bytes"],
        old_build_bytes=old["build_bytes"],
        old_spot_bytes=old["spot_bytes"],
        numeric_payload_bytes=old["numeric"],
        spot_numeric_payload_bytes=old["spot_numeric"],
        new_request=request,
        new_report=report,
        reference_config=config,
        source_pin=source,
        authority_pin=authority,
        created_at_utc=now(),
        output_paths=outputs,
        old_pins=old["pins"],
    )
    new_reference = builder.strict_json(payloads[outputs["reference"]], "new reference")
    new_spot = builder.strict_json(payloads[outputs["spot"]], "new spot")
    assert builder.project_reference_numeric(new_reference) == old["numeric"]
    assert builder.project_reference_spot_numeric(new_spot) == old["spot_numeric"]
    assert all(
        left["reference_forward_context_sha256"]
        != right["reference_forward_context_sha256"]
        for left, right in zip(
            old["reference"]["rows"], new_reference["rows"], strict=True
        )
    )
    receipt = builder.strict_json(payloads[outputs["receipt"]], "receipt")
    assert {key: receipt[key] for key in builder.recovery_lifecycle()} == (
        builder.recovery_lifecycle()
    )
    assert receipt["projections"]["reference_504_rows"]["byte_identical"] is True
    assert receipt["projections"]["spot_3_rows"]["byte_identical"] is True


def published_reference_replay_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    old = old_reference_fixture(tmp_path)
    old_bytes = {
        "reference_request": old["old_request_bytes"],
        "reference_report": old["old_report_bytes"],
        "reference": old["reference_bytes"],
        "reference_build": old["build_bytes"],
        "reference_spot": old["spot_bytes"],
        "reference_numeric_payload": old["numeric"],
        "reference_spot_numeric_payload": old["spot_numeric"],
    }
    for role, data in old_bytes.items():
        write_registered(old["pins"][role], data)
    monkeypatch.setattr(builder, "FORMAL_OLD_PINS", dict(old["pins"]))

    source_data = b"source-v6"
    authority_data = b"recovery-authority"
    source = synthetic_pin(tmp_path / "source-v6", source_data)
    authority = synthetic_pin(tmp_path / "authority", authority_data)
    write_registered(source, source_data)
    write_registered(authority, authority_data)
    request, report = new_pair("reference_precompute", source, authority)
    request["input_config"] = {
        "path": str((tmp_path / "reference-v8.yaml").resolve()),
        "sha256": digest("new-config-file"),
    }
    report["request_sha256"] = sha256(builder.canonical_json_bytes(request)).hexdigest()
    config = {
        "dataloader": deepcopy(old["reference"]["precompute_config"]["dataloader"]),
        "model": deepcopy(old["reference"]["precompute_config"]["model"]),
        "training": {
            "phase6_code_source_manifest": source.path,
            "phase6_code_source_manifest_sha256": source.sha256,
        },
    }
    outputs = {
        role: str((tmp_path / f"new-{role}").resolve())
        for role in ("request", "report", "reference", "build", "spot", "receipt")
    }
    payloads = builder.build_reference_payloads(
        old_request_bytes=old["old_request_bytes"],
        old_report_bytes=old["old_report_bytes"],
        old_reference_bytes=old["reference_bytes"],
        old_build_bytes=old["build_bytes"],
        old_spot_bytes=old["spot_bytes"],
        numeric_payload_bytes=old["numeric"],
        spot_numeric_payload_bytes=old["spot_numeric"],
        new_request=request,
        new_report=report,
        reference_config=config,
        source_pin=source,
        authority_pin=authority,
        created_at_utc=now(),
        output_paths=outputs,
        old_pins=old["pins"],
    )
    modes = {
        "request": 0o444,
        "report": 0o600,
        "reference": 0o400,
        "build": 0o400,
        "spot": 0o400,
        "receipt": 0o400,
    }
    for role, mode in modes.items():
        replace_registered(outputs[role], payloads[outputs[role]], mode)
    monkeypatch.setattr(builder, "NEW_REFERENCE_REQUEST", outputs["request"])
    monkeypatch.setattr(builder, "NEW_REFERENCE_REPORT", outputs["report"])
    monkeypatch.setattr(builder, "NEW_REFERENCE", outputs["reference"])
    monkeypatch.setattr(builder, "NEW_REFERENCE_BUILD", outputs["build"])
    monkeypatch.setattr(builder, "NEW_REFERENCE_SPOT", outputs["spot"])
    receipt_pin = synthetic_pin(Path(outputs["receipt"]), payloads[outputs["receipt"]])
    builder.validate_reference_rebind_receipt(
        receipt_pin, source_pin=source, authority_pin=authority
    )
    return {
        "authority": authority,
        "outputs": outputs,
        "payloads": payloads,
        "source": source,
    }


def test_reference_receipt_replay_is_safe_in_gpu_parent_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = published_reference_replay_fixture(tmp_path, monkeypatch)
    receipt_pin = synthetic_pin(
        Path(replay["outputs"]["receipt"]),
        replay["payloads"][replay["outputs"]["receipt"]],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setitem(sys.modules, "torch", object())

    with pytest.raises(builder.RebindError, match="CUDA_VISIBLE_DEVICES"):
        builder.build_reference_payloads()
    builder.validate_reference_rebind_receipt(
        receipt_pin,
        source_pin=replay["source"],
        authority_pin=replay["authority"],
    )


@pytest.mark.parametrize(
    ("target", "receipt_role", "mutation"),
    [
        ("build", "build_v9", "build_backlink"),
        ("reference", "reference_v9", "reference_task"),
        ("spot", "spot_v9", "spot_exact_match"),
    ],
)
def test_reference_receipt_replay_rejects_non_numeric_and_backlink_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    receipt_role: str,
    mutation: str,
) -> None:
    replay = published_reference_replay_fixture(tmp_path, monkeypatch)
    outputs = replay["outputs"]
    forged = builder.strict_json(Path(outputs[target]).read_bytes(), target)
    if mutation == "build_backlink":
        forged["reference_artifact_sha256"] = "0" * 64
    elif mutation == "reference_task":
        forged["rows"][0]["task_name"] = "forged-task"
    else:
        forged["rows"][0]["exact_match"] = False
    forged_bytes = builder.canonical_json_bytes(forged)
    forged_pin = replace_registered(outputs[target], forged_bytes, 0o400)

    receipt = builder.strict_json(
        replay["payloads"][outputs["receipt"]], "reference receipt"
    )
    receipt["new"][receipt_role] = forged_pin.value()
    receipt_bytes = builder.canonical_json_bytes(receipt)
    receipt_pin = replace_registered(outputs["receipt"], receipt_bytes, 0o400)
    with pytest.raises(builder.RebindError, match="deterministic output replay"):
        builder.validate_reference_rebind_receipt(
            receipt_pin,
            source_pin=replay["source"],
            authority_pin=replay["authority"],
        )


def test_reference_numeric_drift_fails_before_any_publication(tmp_path: Path) -> None:
    old = old_reference_fixture(tmp_path)
    source = builder.Pin(
        str((tmp_path / "source-v6").resolve()), digest("source-v6"), 99, 0o400
    )
    authority = builder.Pin(
        str((tmp_path / "authority").resolve()), digest("authority"), 99, 0o400
    )
    request, report = new_pair("reference_precompute", source, authority)
    request["input_config"] = {"path": "/formal/new.yaml", "sha256": digest("config")}
    report["request_sha256"] = sha256(builder.canonical_json_bytes(request)).hexdigest()
    config = deepcopy(old["reference"]["precompute_config"])
    del config["execution"]
    del config["schema_version"]
    config["training"]["phase6_code_source_manifest"] = source.path
    config["training"]["phase6_code_source_manifest_sha256"] = source.sha256
    drift = builder.strict_json(old["numeric"], "numeric")
    drift["rows"][0]["error_float32_bits"] = "deadbeef"
    outputs = {
        key: str((tmp_path / key).resolve())
        for key in ("request", "report", "reference", "build", "spot", "receipt")
    }
    with pytest.raises(builder.RebindError, match="historical reference projection"):
        builder.build_reference_payloads(
            old_request_bytes=old["old_request_bytes"],
            old_report_bytes=old["old_report_bytes"],
            old_reference_bytes=old["reference_bytes"],
            old_build_bytes=old["build_bytes"],
            old_spot_bytes=old["spot_bytes"],
            numeric_payload_bytes=builder.canonical_json_bytes(drift),
            spot_numeric_payload_bytes=old["spot_numeric"],
            new_request=request,
            new_report=report,
            reference_config=config,
            source_pin=source,
            authority_pin=authority,
            created_at_utc=now(),
            output_paths=outputs,
            old_pins=old["pins"],
        )
    assert not any(Path(path).exists() for path in outputs.values())


def smoke_fixture(tmp_path: Path) -> dict:
    old_request, old_report, old_request_bytes, old_report_bytes = pair(
        "smoke", historical=False, label="old-smoke"
    )
    binding_keys = (
        "action_stats_sha256",
        "arm_projection_manifest_sha256",
        "dataset_contract_sha256",
        "identity_sha256",
        "phase1_checkpoint_sha256",
        "plan_sha256",
        "preflight_report_sha256",
        "preflight_request_sha256",
        "reference_artifact_sha256",
        "runtime_support_manifest_sha256",
        "smoke_runtime_config_sha256",
        "source_manifest_sha256",
        "spot_artifact_sha256",
        "student_checkpoint_sha256",
    )
    gate = {
        "artifact_role": "real_2b_smoke_gate",
        "authorization": {"operation": "real-2b-smoke"},
        "bindings": {key: digest(f"old:{key}") for key in binding_keys},
        "cases": [
            {
                "case_id": arm,
                "config_projection_sha256": digest(f"legacy-projection:{arm}"),
                "formal_training_started": False,
                "gpu_forward_executed": True,
                "numeric_trace_sha256": digest(f"trace:{arm}"),
            }
            for arm in builder.ARM_NAMES
        ],
        "closed_loop_started": False,
        "completed_at_utc": "2026-07-24T04:05:06Z",
        "formal_training_started": False,
        "gpu_smoke_executed": True,
        "schema_version": builder.OLD_SMOKE_SCHEMA,
        "status": "pass",
    }
    gate_bytes = builder.canonical_json_bytes(gate)
    nonbinding = builder.project_smoke_legacy_nonbinding(gate)
    pins = {
        "smoke_request": synthetic_pin(
            tmp_path / "old-smoke-request", old_request_bytes, 0o444
        ),
        "smoke_report": synthetic_pin(
            tmp_path / "old-smoke-report", old_report_bytes, 0o600
        ),
        "smoke_gate": synthetic_pin(tmp_path / "old-gate", gate_bytes),
        "smoke_nonbinding_payload": synthetic_pin(
            tmp_path / "old-nonbinding", nonbinding
        ),
    }
    return {
        "request_bytes": old_request_bytes,
        "report_bytes": old_report_bytes,
        "gate": gate,
        "gate_bytes": gate_bytes,
        "nonbinding": nonbinding,
        "pins": pins,
    }


def test_smoke_rebind_preserves_science_and_rebinds_operational_leaves(
    tmp_path: Path,
) -> None:
    old = smoke_fixture(tmp_path)
    source = builder.Pin(
        str((tmp_path / "source-v6").resolve()), digest("source-v6"), 99, 0o400
    )
    authority = builder.Pin(
        str((tmp_path / "authority").resolve()), digest("authority"), 99, 0o400
    )
    reference_receipt = builder.Pin(
        str((tmp_path / "reference-receipt").resolve()),
        digest("reference-receipt"),
        222,
        0o400,
    )
    request, report = new_pair("smoke", source, authority)
    request["input_config"] = {
        "path": "/formal/smoke-v7.json",
        "sha256": digest("smoke-config"),
    }
    request["runtime_files"] = {
        "action_stats": {"path": "/runtime/action", "sha256": digest("action-stats")},
        "phase1_checkpoint": {"path": "/runtime/phase1", "sha256": digest("phase1")},
        "student_checkpoint": {"path": "/runtime/student", "sha256": digest("student")},
    }
    projection_manifest = {
        "arms": [
            {"arm": arm, "projection_sha256": digest(f"recovery-projection:{arm}")}
            for arm in builder.ARM_NAMES
        ],
        **builder.recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "schema_version": "synthetic-projection-manifest-v2",
    }
    projection_bytes = builder.canonical_json_bytes(projection_manifest)
    request["artifacts"].update(
        {
            "action_reference": {
                "path": "/formal/reference-v8",
                "sha256": digest("reference-v8"),
            },
            "action_reference_spot_check": {
                "path": "/formal/spot-v8",
                "sha256": digest("spot-v8"),
            },
            "arm_projection_manifest": {
                "path": "/formal/projection-v2",
                "sha256": sha256(projection_bytes).hexdigest(),
            },
            "dataset_contract": {
                "path": "/formal/dataset",
                "sha256": digest("dataset"),
            },
            "projection_equivalence_receipt": {
                "path": "/formal/equivalence",
                "sha256": digest("equivalence"),
            },
            "runtime_support_manifest": {
                "path": "/formal/runtime-support",
                "sha256": digest("runtime-support"),
            },
        }
    )
    report["artifacts"] = deepcopy(request["artifacts"])
    report["request_sha256"] = sha256(builder.canonical_json_bytes(request)).hexdigest()
    outputs = {
        "request": str((tmp_path / "smoke-request-v7").resolve()),
        "report": str((tmp_path / "smoke-report-v7").resolve()),
        "gate": str((tmp_path / "gate-v7").resolve()),
        "smoke_receipt": str((tmp_path / "smoke-receipt").resolve()),
        "evidence_receipt": str((tmp_path / "evidence-receipt").resolve()),
    }
    validator_calls = []

    def validator(data: bytes, bindings: dict) -> None:
        validator_calls.append((data, bindings))

    payloads = builder.build_smoke_payloads(
        old_request_bytes=old["request_bytes"],
        old_report_bytes=old["report_bytes"],
        old_gate_bytes=old["gate_bytes"],
        old_nonbinding_payload_bytes=old["nonbinding"],
        new_request=request,
        new_report=report,
        projection_manifest=projection_manifest,
        projection_manifest_bytes=projection_bytes,
        source_pin=source,
        authority_pin=authority,
        reference_receipt_pin=reference_receipt,
        created_at_utc=now(),
        output_paths=outputs,
        old_pins=old["pins"],
        production_validator=validator,
    )
    new_gate = builder.strict_json(payloads[outputs["gate"]], "new gate")
    assert len(validator_calls) == 1
    assert builder.project_smoke_scientific(
        old["gate"]
    ) == builder.project_smoke_scientific(new_gate)
    assert "formal_training_started" not in new_gate
    assert new_gate["schema_version"] == builder.NEW_SMOKE_SCHEMA
    assert new_gate["historical_formal_training_started"] is True
    assert new_gate["recovery_cohort_started"] is False
    assert builder.canonical_json_bytes(new_gate["cases"]) == (
        builder.canonical_json_bytes(old["gate"]["cases"])
    )
    assert [case["config_projection_sha256"] for case in new_gate["cases"]] == [
        digest(f"legacy-projection:{arm}") for arm in builder.ARM_NAMES
    ]
    evidence = builder.strict_json(payloads[outputs["evidence_receipt"]], "evidence")
    assert {key: evidence[key] for key in builder.recovery_lifecycle()} == (
        builder.recovery_lifecycle()
    )
    assert evidence["gpu_reexecution_performed"] is False
    assert all(evidence["validation"].values())


def test_smoke_rebind_passes_production_gate_with_byte_identical_cases(
    tmp_path: Path,
) -> None:
    from sana_wam.train import phase6_arm_config
    from sana_wam.train import phase6_smoke_gate as production_gate
    from tests import test_phase6_smoke_gate as smoke_fixture_module

    # Convert a production-valid v4 fixture back to the registered v3 wrapper.
    # Its five nested GPU case payloads already carry the immutable legacy pins.
    old_gate = deepcopy(smoke_fixture_module._artifact())
    for key, expected in builder.recovery_lifecycle().items():
        assert old_gate.pop(key) == expected
    assert old_gate.pop("registered_before_recovery_cohort_started") is True
    old_gate["closed_loop_started"] = False
    old_gate["formal_training_started"] = False
    old_gate["schema_version"] = builder.OLD_SMOKE_SCHEMA
    old_gate_bytes = builder.canonical_json_bytes(old_gate)
    old_nonbinding = builder.project_smoke_legacy_nonbinding(old_gate)
    old_request, old_report, old_request_bytes, old_report_bytes = pair(
        "smoke", historical=False, label="production-old-smoke"
    )
    old_pins = {
        "smoke_request": synthetic_pin(
            tmp_path / "production-old-request", old_request_bytes, 0o444
        ),
        "smoke_report": synthetic_pin(
            tmp_path / "production-old-report", old_report_bytes, 0o600
        ),
        "smoke_gate": synthetic_pin(tmp_path / "production-old-gate", old_gate_bytes),
        "smoke_nonbinding_payload": synthetic_pin(
            tmp_path / "production-old-nonbinding", old_nonbinding
        ),
    }

    source = builder.Pin(
        str((tmp_path / "production-source-v6").resolve()),
        digest("production-source-v6"),
        99,
        0o400,
    )
    authority = builder.Pin(
        str((tmp_path / "production-authority").resolve()),
        digest("production-authority"),
        101,
        0o400,
    )
    reference_receipt = builder.Pin(
        str((tmp_path / "production-reference-receipt").resolve()),
        digest("production-reference-receipt"),
        103,
        0o400,
    )
    request, report = new_pair("smoke", source, authority)
    request["input_config"] = {
        "path": "/formal/production-smoke-runtime-v5.json",
        "sha256": digest("production-smoke-runtime"),
    }
    request["runtime_files"] = {
        "action_stats": {
            "path": "/runtime/production-action",
            "sha256": old_gate["bindings"]["action_stats_sha256"],
        },
        "phase1_checkpoint": {
            "path": "/runtime/production-phase1",
            "sha256": old_gate["bindings"]["phase1_checkpoint_sha256"],
        },
        "student_checkpoint": {
            "path": "/runtime/production-student",
            "sha256": old_gate["bindings"]["student_checkpoint_sha256"],
        },
    }
    projection_manifest = phase6_arm_config.build_phase6_arm_projection_manifest()
    projection_bytes = builder.canonical_json_bytes(projection_manifest)
    request["artifacts"].update(
        {
            "action_reference": {
                "path": "/formal/production-reference-v8",
                "sha256": digest("production-reference-v8"),
            },
            "action_reference_spot_check": {
                "path": "/formal/production-spot-v8",
                "sha256": digest("production-spot-v8"),
            },
            "arm_projection_manifest": {
                "path": "/formal/production-projection-v2",
                "sha256": sha256(projection_bytes).hexdigest(),
            },
            "dataset_contract": {
                "path": "/formal/production-dataset",
                "sha256": digest("production-dataset"),
            },
            "projection_equivalence_receipt": {
                "path": "/formal/production-equivalence",
                "sha256": digest("production-equivalence"),
            },
            "runtime_support_manifest": {
                "path": "/formal/production-runtime-support",
                "sha256": digest("production-runtime-support"),
            },
        }
    )
    report["artifacts"] = deepcopy(request["artifacts"])
    report["request_sha256"] = sha256(builder.canonical_json_bytes(request)).hexdigest()
    outputs = {
        role: str((tmp_path / f"production-{role}").resolve())
        for role in ("request", "report", "gate", "smoke_receipt", "evidence_receipt")
    }
    validated: list[dict] = []

    def validator(data: bytes, bindings: dict[str, str]) -> None:
        value = builder.strict_json(data, "production smoke gate")
        assert value["bindings"] == bindings
        kwargs = {
            "expected_action_stats_sha256": bindings["action_stats_sha256"],
            "expected_arm_projection_manifest_sha256": bindings[
                "arm_projection_manifest_sha256"
            ],
            "expected_artifact_sha256": sha256(data).hexdigest(),
            "expected_dataset_contract_sha256": bindings["dataset_contract_sha256"],
            "expected_identity_sha256": bindings["identity_sha256"],
            "expected_phase1_checkpoint_sha256": bindings["phase1_checkpoint_sha256"],
            "expected_plan_sha256": bindings["plan_sha256"],
            "expected_preflight_report_sha256": bindings["preflight_report_sha256"],
            "expected_preflight_request_sha256": bindings["preflight_request_sha256"],
            "expected_reference_artifact_sha256": bindings["reference_artifact_sha256"],
            "expected_runtime_support_manifest_sha256": bindings[
                "runtime_support_manifest_sha256"
            ],
            "expected_smoke_runtime_config_sha256": bindings[
                "smoke_runtime_config_sha256"
            ],
            "expected_source_manifest_sha256": bindings["source_manifest_sha256"],
            "expected_spot_artifact_sha256": bindings["spot_artifact_sha256"],
            "expected_student_checkpoint_sha256": bindings["student_checkpoint_sha256"],
        }
        validated.append(
            production_gate.validate_phase6_smoke_artifact_bytes(data, **kwargs)
        )

    payloads = builder.build_smoke_payloads(
        old_request_bytes=old_request_bytes,
        old_report_bytes=old_report_bytes,
        old_gate_bytes=old_gate_bytes,
        old_nonbinding_payload_bytes=old_nonbinding,
        new_request=request,
        new_report=report,
        projection_manifest=projection_manifest,
        projection_manifest_bytes=projection_bytes,
        source_pin=source,
        authority_pin=authority,
        reference_receipt_pin=reference_receipt,
        created_at_utc=now(),
        output_paths=outputs,
        old_pins=old_pins,
        production_validator=validator,
    )
    new_gate = builder.strict_json(payloads[outputs["gate"]], "production new gate")
    assert len(validated) == 1
    assert [case["case_id"] for case in validated[0]["cases"]] == list(
        builder.ARM_NAMES
    )
    assert builder.canonical_json_bytes(new_gate["cases"]) == (
        builder.canonical_json_bytes(old_gate["cases"])
    )
    assert builder.project_smoke_scientific(new_gate) == (
        builder.project_smoke_scientific(old_gate)
    )
    smoke_receipt = builder.strict_json(
        payloads[outputs["smoke_receipt"]], "production smoke receipt"
    )
    evidence_receipt = builder.strict_json(
        payloads[outputs["evidence_receipt"]], "production evidence receipt"
    )
    assert smoke_receipt["created_at_utc"] == evidence_receipt["created_at_utc"]


def test_smoke_cli_rejects_inverse_chronology_before_any_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_data = builder.canonical_json_bytes({"status": "synthetic-reference"})
    reference_path = tmp_path / "reference-rebind-receipt.json"
    reference_path.write_bytes(reference_data)
    os.chmod(reference_path, 0o400)
    source_pin = builder.Pin(
        str((tmp_path / "source-v6.json").resolve()), digest("source-v6"), 11, 0o400
    )
    authority_pin = builder.Pin(
        str((tmp_path / "authority.json").resolve()), digest("authority"), 13, 0o400
    )
    output_paths = [
        tmp_path / name
        for name in (
            "smoke-request-v7.json",
            "smoke-report-v7.json",
            "smoke-gate-v7.json",
            "smoke-receipt.json",
            "evidence-receipt.json",
        )
    ]
    for name, output in zip(
        (
            "NEW_SMOKE_REQUEST",
            "NEW_SMOKE_REPORT",
            "NEW_SMOKE_GATE",
            "SMOKE_RECEIPT",
            "EVIDENCE_REBINDING_RECEIPT",
        ),
        output_paths,
        strict=True,
    ):
        monkeypatch.setattr(builder, name, str(output.resolve()))
    monkeypatch.setattr(builder, "REFERENCE_RECEIPT", str(reference_path.resolve()))

    def cli_pin(_path, _sha, _size, _mode, label):
        return source_pin if label == "source manifest-v9" else authority_pin

    validated = []

    def validate_reference(pin, *, source_pin, authority_pin):
        assert builder.stable_read(pin) == reference_data
        validated.append((source_pin, authority_pin))
        return {"created_at_utc": "2026-07-25T00:00:02Z"}

    monkeypatch.setattr(builder, "_cli_pin", cli_pin)
    monkeypatch.setattr(
        builder, "validate_reference_rebind_receipt", validate_reference
    )
    monkeypatch.setattr(
        builder,
        "_read_formal_roles",
        lambda *_args, **_kwargs: pytest.fail(
            "historical inputs were read after chronology should have failed"
        ),
    )
    monkeypatch.setattr(
        builder,
        "_production_smoke_inputs",
        lambda *_args, **_kwargs: pytest.fail(
            "production inputs were built after chronology should have failed"
        ),
    )
    args = SimpleNamespace(
        source_manifest_v9=source_pin.path,
        source_manifest_v9_sha256=source_pin.sha256,
        source_manifest_v9_size=source_pin.size_bytes,
        recovery_authority=authority_pin.path,
        recovery_authority_sha256=authority_pin.sha256,
        recovery_authority_size=authority_pin.size_bytes,
        reference_receipt_sha256=sha256(reference_data).hexdigest(),
        created_at_utc="2026-07-25T00:00:01Z",
        publish=True,
    )
    with pytest.raises(builder.RebindError, match="creation chronology"):
        builder._smoke_cli(args)
    assert len(validated) == 1
    assert not any(path.exists() for path in output_paths)


def published_smoke_replay_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    old = smoke_fixture(tmp_path)
    old_bytes = {
        "smoke_request": old["request_bytes"],
        "smoke_report": old["report_bytes"],
        "smoke_gate": old["gate_bytes"],
        "smoke_nonbinding_payload": old["nonbinding"],
    }
    for role, data in old_bytes.items():
        write_registered(old["pins"][role], data)
    monkeypatch.setattr(builder, "FORMAL_OLD_PINS", dict(old["pins"]))

    source_data = b"source-v6"
    authority_data = b"recovery-authority"
    reference_receipt_data = b"reference-rebind-receipt"
    source = synthetic_pin(tmp_path / "source-v6", source_data)
    authority = synthetic_pin(tmp_path / "authority", authority_data)
    reference_receipt = synthetic_pin(
        tmp_path / "reference-receipt", reference_receipt_data
    )
    for pin, data in (
        (source, source_data),
        (authority, authority_data),
        (reference_receipt, reference_receipt_data),
    ):
        write_registered(pin, data)

    projection_manifest = {
        "arms": [
            {"arm": arm, "projection_sha256": digest(f"recovery-projection:{arm}")}
            for arm in builder.ARM_NAMES
        ],
        **builder.recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "schema_version": "synthetic-projection-manifest-v2",
    }
    projection_bytes = builder.canonical_json_bytes(projection_manifest)
    projection_pin = synthetic_pin(tmp_path / "projection-manifest", projection_bytes)
    write_registered(projection_pin, projection_bytes)

    request, report = new_pair("smoke", source, authority)
    request["input_config"] = {
        "path": str((tmp_path / "smoke-v7.json").resolve()),
        "sha256": digest("smoke-config"),
    }
    request["runtime_files"] = {
        "action_stats": {"path": "/runtime/action", "sha256": digest("action-stats")},
        "phase1_checkpoint": {"path": "/runtime/phase1", "sha256": digest("phase1")},
        "student_checkpoint": {"path": "/runtime/student", "sha256": digest("student")},
    }
    request["artifacts"].update(
        {
            "action_reference": {
                "path": "/formal/reference-v8",
                "sha256": digest("reference-v8"),
            },
            "action_reference_spot_check": {
                "path": "/formal/spot-v8",
                "sha256": digest("spot-v8"),
            },
            "arm_projection_manifest": {
                "path": projection_pin.path,
                "sha256": projection_pin.sha256,
            },
            "dataset_contract": {
                "path": "/formal/dataset",
                "sha256": digest("dataset"),
            },
            "projection_equivalence_receipt": {
                "path": "/formal/equivalence",
                "sha256": digest("equivalence"),
            },
            "runtime_support_manifest": {
                "path": "/formal/runtime-support",
                "sha256": digest("runtime-support"),
            },
        }
    )
    report["artifacts"] = deepcopy(request["artifacts"])
    report["request_sha256"] = sha256(builder.canonical_json_bytes(request)).hexdigest()
    outputs = {
        role: str((tmp_path / f"new-{role}").resolve())
        for role in ("request", "report", "gate", "smoke_receipt", "evidence_receipt")
    }
    payloads = builder.build_smoke_payloads(
        old_request_bytes=old["request_bytes"],
        old_report_bytes=old["report_bytes"],
        old_gate_bytes=old["gate_bytes"],
        old_nonbinding_payload_bytes=old["nonbinding"],
        new_request=request,
        new_report=report,
        projection_manifest=projection_manifest,
        projection_manifest_bytes=projection_bytes,
        source_pin=source,
        authority_pin=authority,
        reference_receipt_pin=reference_receipt,
        created_at_utc=now(),
        output_paths=outputs,
        old_pins=old["pins"],
        production_validator=None,
    )
    modes = {
        "request": 0o444,
        "report": 0o600,
        "gate": 0o400,
        "smoke_receipt": 0o400,
    }
    for role, mode in modes.items():
        replace_registered(outputs[role], payloads[outputs[role]], mode)
    monkeypatch.setattr(builder, "NEW_SMOKE_REQUEST", outputs["request"])
    monkeypatch.setattr(builder, "NEW_SMOKE_REPORT", outputs["report"])
    monkeypatch.setattr(builder, "NEW_SMOKE_GATE", outputs["gate"])
    monkeypatch.setattr(
        builder, "EVIDENCE_REBINDING_RECEIPT", outputs["evidence_receipt"]
    )
    receipt_pin = synthetic_pin(
        Path(outputs["smoke_receipt"]), payloads[outputs["smoke_receipt"]]
    )
    builder.validate_smoke_rebind_receipt(
        receipt_pin,
        source_pin=source,
        authority_pin=authority,
        reference_receipt_pin=reference_receipt,
    )
    return {
        "authority": authority,
        "outputs": outputs,
        "payloads": payloads,
        "reference_receipt": reference_receipt,
        "source": source,
    }


def test_smoke_receipt_replay_is_safe_in_gpu_parent_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = published_smoke_replay_fixture(tmp_path, monkeypatch)
    receipt_pin = synthetic_pin(
        Path(replay["outputs"]["smoke_receipt"]),
        replay["payloads"][replay["outputs"]["smoke_receipt"]],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setitem(sys.modules, "torch", object())

    with pytest.raises(builder.RebindError, match="CUDA_VISIBLE_DEVICES"):
        builder.build_smoke_payloads()
    builder.validate_smoke_rebind_receipt(
        receipt_pin,
        source_pin=replay["source"],
        authority_pin=replay["authority"],
        reference_receipt_pin=replay["reference_receipt"],
    )


@pytest.mark.parametrize("mutation", ["binding", "case_projection"])
def test_smoke_receipt_replay_rejects_forged_operational_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    replay = published_smoke_replay_fixture(tmp_path, monkeypatch)
    outputs = replay["outputs"]
    gate = builder.strict_json(Path(outputs["gate"]).read_bytes(), "smoke gate")
    if mutation == "binding":
        gate["bindings"]["plan_sha256"] = "0" * 64
    else:
        gate["cases"][0]["config_projection_sha256"] = "f" * 64
    gate_bytes = builder.canonical_json_bytes(gate)
    gate_pin = replace_registered(outputs["gate"], gate_bytes, 0o400)

    receipt = builder.strict_json(
        replay["payloads"][outputs["smoke_receipt"]], "smoke receipt"
    )
    receipt["new"]["gate_v8"] = gate_pin.value()
    receipt_bytes = builder.canonical_json_bytes(receipt)
    receipt_pin = replace_registered(outputs["smoke_receipt"], receipt_bytes, 0o400)
    with pytest.raises(
        builder.RebindError,
        match="projection replay|deterministic output replay",
    ):
        builder.validate_smoke_rebind_receipt(
            receipt_pin,
            source_pin=replay["source"],
            authority_pin=replay["authority"],
            reference_receipt_pin=replay["reference_receipt"],
        )


def test_wrong_source_or_authority_cli_binding_fails_closed(tmp_path: Path) -> None:
    source = builder.Pin(
        str((tmp_path / "source").resolve()), digest("source"), 1, 0o400
    )
    authority = builder.Pin(
        str((tmp_path / "authority").resolve()), digest("authority"), 1, 0o400
    )
    request, _report = new_pair("reference_precompute", source, authority)
    request["source_manifest"]["sha256"] = digest("wrong")
    with pytest.raises(builder.RebindError, match="source-v6 CLI binding"):
        builder.validate_recovery_bindings(
            request, source_pin=source, authority_pin=authority
        )
    request, _report = new_pair("reference_precompute", source, authority)
    request["historical_optimizer_step_calls_per_arm"] = True
    with pytest.raises(builder.RebindError, match="recovery lifecycle"):
        builder.validate_recovery_bindings(
            request, source_pin=source, authority_pin=authority
        )
    request, _report = new_pair("reference_precompute", source, authority)
    request["historical_scheduled_lr_scale_at_step0"] = 0
    with pytest.raises(builder.RebindError, match="recovery lifecycle"):
        builder.validate_recovery_bindings(
            request, source_pin=source, authority_pin=authority
        )


def test_stable_read_rejects_symlink_and_sha_drift(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"payload")
    os.chmod(target, 0o400)
    observed_mode = stat.S_IMODE(os.lstat(target).st_mode)
    mode = 0o400 if builder._mode_matches(observed_mode, 0o400) else observed_mode
    pin = builder.Pin(str(target.resolve()), sha256(b"payload").hexdigest(), 7, mode)
    assert builder.stable_read(pin) == b"payload"
    wrong = builder.Pin(str(target.resolve()), digest("wrong"), 7, mode)
    with pytest.raises(builder.RebindError, match="SHA256 differs"):
        builder.stable_read(wrong)
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    link_pin = builder.Pin(str(link.absolute()), pin.sha256, 7, mode)
    with pytest.raises(builder.RebindError, match="registration differs"):
        builder.stable_read(link_pin)


def test_o_excl_receipt_last_publication_is_idempotent_but_never_overwrites(
    tmp_path: Path,
) -> None:
    support = str((tmp_path / "support").resolve())
    receipt = str((tmp_path / "receipt").resolve())
    payloads = {support: b"support", receipt: b"receipt"}
    modes = {support: 0o400, receipt: 0o400}
    first = builder.publish_bundle(payloads, modes=modes, commit_path=receipt)
    assert first[support] == "created_this_invocation"
    assert first[receipt] == "created_this_invocation"
    second = builder.publish_bundle(payloads, modes=modes, commit_path=receipt)
    assert set(second.values()) == {"recovered_exact_existing"}
    with pytest.raises(builder.RebindError, match="differs"):
        builder.publish_bundle(
            {support: b"changed", receipt: b"receipt"},
            modes=modes,
            commit_path=receipt,
        )
    assert Path(support).read_bytes() == b"support"


@pytest.mark.parametrize("failure", ["short_write", "fsync", "directory_fsync"])
def test_atomic_exclusive_write_failure_leaves_no_final_or_owned_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    if failure == "directory_fsync" and builder.os.name != "posix":
        pytest.skip("directory fsync is a POSIX publication contract")
    destination = str((tmp_path / "support").resolve())
    payload = b"registered-payload"
    temporary = Path(builder._temporary_output_path(destination, payload))
    if failure == "short_write":
        real_write = builder.os.write
        calls = 0

        def fail_after_partial(descriptor: int, data) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_write(descriptor, bytes(data[:1]))
            raise OSError("injected write failure")

        monkeypatch.setattr(builder.os, "write", fail_after_partial)
    else:
        real_fsync = builder.os.fsync
        calls = 0

        failure_call = 1 if failure == "fsync" else 3

        def fail_selected_fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == failure_call:
                raise OSError("injected fsync failure")
            real_fsync(descriptor)

        monkeypatch.setattr(builder.os, "fsync", fail_selected_fsync)
    with pytest.raises(OSError, match="injected"):
        builder._write_exclusive(destination, payload, 0o400)
    assert not Path(destination).exists()
    assert not temporary.exists()


def test_deterministic_temp_recovers_crash_after_link_before_unlink(
    tmp_path: Path,
) -> None:
    destination = str((tmp_path / "support").resolve())
    payload = b"registered-payload"
    temporary = Path(builder._temporary_output_path(destination, payload))
    temporary.write_bytes(payload)
    os.chmod(temporary, 0o400)
    os.link(temporary, destination, follow_symlinks=False)
    assert os.lstat(temporary).st_nlink == 2

    states = builder.publish_bundle(
        {destination: payload},
        modes={destination: 0o400},
        commit_path=destination,
    )
    assert states[destination] == "recovered_exact_existing"
    assert not temporary.exists()
    assert os.lstat(destination).st_nlink == 1
    assert Path(destination).read_bytes() == payload


def test_preexisting_temp_rejects_racing_different_final_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = str((tmp_path / "support").resolve())
    payload = b"registered-payload"
    temporary = Path(builder._temporary_output_path(destination, payload))
    temporary.write_bytes(payload)
    os.chmod(temporary, 0o400)

    def publish_different_inode(*_args, **_kwargs) -> None:
        Path(destination).write_bytes(payload)
        os.chmod(destination, 0o400)
        raise FileExistsError(destination)

    monkeypatch.setattr(builder.os, "link", publish_different_inode)
    with pytest.raises(builder.RebindError, match="raced with final"):
        builder._write_exclusive(destination, payload, 0o400)
    assert temporary.exists()
    assert Path(destination).exists()
    assert os.lstat(temporary).st_ino != os.lstat(destination).st_ino


def test_receipt_visible_with_missing_support_is_not_repaired_in_place(
    tmp_path: Path,
) -> None:
    support = str((tmp_path / "support").resolve())
    receipt = tmp_path / "receipt"
    receipt.write_bytes(b"receipt")
    os.chmod(receipt, 0o400)
    payloads = {support: b"support", str(receipt.resolve()): b"receipt"}
    with pytest.raises(builder.RebindError, match="receipt is visible"):
        builder.publish_bundle(
            payloads,
            modes={path: 0o400 for path in payloads},
            commit_path=str(receipt.resolve()),
        )
    assert not Path(support).exists()


def test_cpu_environment_is_explicit_and_builder_has_no_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(builder.RebindError, match="explicitly ''"):
        builder.require_cpu_environment()
    source = (HERE.parent / "scripts" / "build_phase6_recovery_rebind.py").read_text(
        encoding="ascii"
    )
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(ast.parse(source))
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [node]
            if isinstance(node, ast.ImportFrom)
            else []
        )
        if isinstance(alias, ast.alias)
    } | {
        node.module.split(".", 1)[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "torch" not in imported
    assert "O_EXCL" in source
    assert "stable_read" in source
    assert "--source-manifest-v9-sha256" in source
    assert "--recovery-authority-sha256" in source
