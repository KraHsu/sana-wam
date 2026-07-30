from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


CANDIDATE = Path(__file__).resolve().parents[1]
MODULE_PATH = CANDIDATE / "src/sana_wam/train/phase6_artifact_dag.py"
CLI_PATH = CANDIDATE / "scripts/build_phase6_artifact_dag.py"
PREFLIGHT_CLI_PATH = CANDIDATE / "scripts/validate_phase6_artifact_preflight.py"
PHASE6_ROOT = CANDIDATE.parent
TEST_DATASET_SHA256 = sha256(b"synthetic-dataset-contract-v2").hexdigest()
_TORCH_MISSING = object()
_DAG_TORCH_BEFORE = _TORCH_MISSING


def _assert_torch_module_unchanged():
    if _DAG_TORCH_BEFORE is _TORCH_MISSING:
        assert "torch" not in sys.modules
    else:
        assert sys.modules.get("torch") is _DAG_TORCH_BEFORE


def _owned_module_name(name: str) -> bool:
    return (
        name == "sana_wam"
        or name.startswith("sana_wam.")
        or name == "yaml"
        or name.startswith("yaml.")
        or name == "phase6_independent_preflight_cli_under_test"
    )


def _restore_owned_modules(snapshot):
    for name in tuple(sys.modules):
        if _owned_module_name(name):
            sys.modules.pop(name, None)
    sys.modules.update(snapshot)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dag():
    global _DAG_TORCH_BEFORE
    _DAG_TORCH_BEFORE = sys.modules.get("torch", _TORCH_MISSING)
    module_snapshot = {
        name: module
        for name, module in tuple(sys.modules.items())
        if _owned_module_name(name)
    }
    try:
        production_package = CANDIDATE / "src/sana_wam"
        overlay_package = CANDIDATE / "test_overlay/sana_wam"
        sana_wam = types.ModuleType("sana_wam")
        sana_wam.__path__ = [str(production_package), str(overlay_package)]
        train = types.ModuleType("sana_wam.train")
        train.__path__ = [
            str(production_package / "train"),
            str(overlay_package / "train"),
        ]
        sys.modules["sana_wam"] = sana_wam
        sys.modules["sana_wam.train"] = train
        _load(
            "sana_wam.train.phase6_downstream_pins",
            production_package / "train/phase6_downstream_pins.py",
        )
        module = _load("sana_wam.train.phase6_artifact_dag", MODULE_PATH)
        if not module._SHA256_RE.fullmatch(module.FROZEN_SHA256["recovery_authority"]):
            module.FROZEN_SHA256["recovery_authority"] = sha256(
                b"synthetic recovery authority"
            ).hexdigest()
        _assert_torch_module_unchanged()
        yield module
    finally:
        _restore_owned_modules(module_snapshot)
        _assert_torch_module_unchanged()


def _install_dependency_modules(dag):
    production_package = CANDIDATE / "src/sana_wam"
    if (production_package / "train/phase6_arm_config.py").is_file():
        package_paths = [production_package]
        arm_path = production_package / "train/phase6_arm_config.py"
        reference_path = production_package / "train/action_reference_table.py"
        production_support = production_package / "dataloader"
        support_names = (
            "phase6_expansion_support.py",
            "phase6_padding_semantics.py",
            "task_sample_plan.py",
        )
        production_presence = tuple(
            (production_support / name).is_file() for name in support_names
        )
        overlay_support = CANDIDATE / "test_overlay/sana_wam/dataloader"
        if all(production_presence):
            support_root = production_support
        elif not any(production_presence) and all(
            (overlay_support / name).is_file() for name in support_names
        ):
            support_root = overlay_support
        else:
            raise AssertionError("Phase-6 support module set is incomplete")
    else:
        package_paths = [
            PHASE6_ROOT / "materialization_order_fix/src/sana_wam",
            PHASE6_ROOT / "action_reference_integration_fixes/src/sana_wam",
        ]
        arm_path = package_paths[0] / "train/phase6_arm_config.py"
        reference_path = package_paths[1] / "train/action_reference_table.py"
        support_root = PHASE6_ROOT / "_test_support/sana_wam/dataloader"
    train_paths = [path / "train" for path in package_paths]
    sana_wam = types.ModuleType("sana_wam")
    sana_wam.__path__ = [str(path) for path in package_paths]
    train = types.ModuleType("sana_wam.train")
    train.__path__ = [str(path) for path in train_paths]
    sys.modules["sana_wam"] = sana_wam
    sys.modules["sana_wam.train"] = train
    dataloader = types.ModuleType("sana_wam.dataloader")
    dataloader.__path__ = [str(support_root)]
    sys.modules["sana_wam.dataloader"] = dataloader
    _load(
        "sana_wam.dataloader.phase6_expansion_support",
        support_root / "phase6_expansion_support.py",
    )
    _load(
        "sana_wam.dataloader.phase6_padding_semantics",
        support_root / "phase6_padding_semantics.py",
    )
    _load(
        "sana_wam.dataloader.task_sample_plan",
        support_root / "task_sample_plan.py",
    )
    # The local harness intentionally lacks PyYAML. The projection builder only
    # needs these import-time loader symbols; no YAML parsing is used here.
    yaml = types.ModuleType("yaml")

    class SafeLoader:
        @classmethod
        def add_constructor(cls, *_args, **_kwargs):
            return None

    yaml.SafeLoader = SafeLoader
    yaml.YAMLError = Exception
    yaml.resolver = types.SimpleNamespace(
        BaseResolver=types.SimpleNamespace(DEFAULT_MAPPING_TAG="mapping")
    )
    sys.modules["yaml"] = yaml
    _load(
        "sana_wam.train.phase6_downstream_pins",
        production_package / "train/phase6_downstream_pins.py",
    )
    arm = _load(
        "sana_wam.train.phase6_arm_config",
        arm_path,
    )
    reference = _load(
        "sana_wam.train.action_reference_table",
        reference_path,
    )
    return arm, reference


def _fake_pin(dag, role: str, *, digest: str | None = None):
    return {
        "path": f"/fake/{role}",
        "sha256": digest or sha256(role.encode()).hexdigest(),
    }


def _shape_request(dag, purpose: str):
    # Request-shape tests use an isolated digest so frozen-pin drift is explicit.
    artifacts = {
        role: _fake_pin(
            dag,
            role,
            digest=(
                TEST_DATASET_SHA256
                if role == "dataset_contract"
                else dag.FROZEN_SHA256.get(role)
            ),
        )
        for role in dag.FIXED_ARTIFACT_ROLES
    }
    if purpose in ("smoke", "training"):
        artifacts.update(
            {role: _fake_pin(dag, role) for role in dag.SMOKE_ADDITIONAL_ROLES}
        )
    if purpose == "training":
        artifacts.update(
            {role: _fake_pin(dag, role) for role in dag.TRAINING_ADDITIONAL_ROLES}
        )
    runtime = {
        role: _fake_pin(dag, role, digest=dag.FROZEN_SHA256[role])
        for role in dag.RUNTIME_ROLES
    }
    projection = "1" * 64
    if purpose == "reference_precompute":
        authorization = deepcopy(dag.REFERENCE_AUTHORIZATION)
        lifecycle = {
            **dag.recovery_lifecycle(),
            "reference_gpu_started": False,
        }
    elif purpose == "smoke":
        authorization = deepcopy(dag.SMOKE_AUTHORIZATION)
        lifecycle = {
            **dag.recovery_lifecycle(),
            "reference_precompute_completed": True,
            "smoke_gpu_started": False,
        }
    else:
        authorization = dag.training_authorization("T1_E1A1", projection)
        lifecycle = {
            **dag.recovery_lifecycle(),
            "reference_precompute_completed": True,
            "smoke_completed": True,
        }
    return {
        "authorization": authorization,
        "artifacts": artifacts,
        "effective_output_root": dag.OPERATIONAL_ROOT,
        "input_config": {"path": "/fake/config", "sha256": projection},
        **lifecycle,
        "purpose": purpose,
        "registered_before_recovery_cohort_started": True,
        "repository_root": dag.REPOSITORY_ROOT,
        "runtime_files": runtime,
        "schema_version": dag.REQUEST_SCHEMA_VERSION,
        "source_manifest": _fake_pin(dag, "source"),
    }


def test_plan_is_topological_acyclic_and_non_training(dag):
    plan = dag.build_artifact_dag_plan()
    validated = dag.validate_artifact_dag_plan(plan)
    assert validated == plan
    ids = [node["id"] for node in plan["nodes"]]
    assert ids == [
        "recovery_authority",
        "source_manifest",
        "after_source_freeze",
        "reference_inputs",
        "reference_numeric_cpu_rebind",
        "smoke_request",
        "smoke_report",
        "smoke_numeric_cpu_rebind",
        "training_requests",
        "training_reports",
        "materialization_pins",
        "final_sink",
    ]
    assert all(node["training"] is False for node in plan["nodes"])
    assert [node["id"] for node in plan["nodes"] if node["gpu"]] == []
    assert plan["source_manifest_generated_by_this_cli"] is False
    commands = [
        command
        for node in plan["nodes"]
        for command in (
            [node["command"]] if "command" in node else node.get("commands", [])
        )
    ]
    assert commands
    assert all(
        (
            command[1] == "-B"
            and command[2].endswith("/scripts/validate_phase6_artifact_preflight.py")
            and "--purpose" in command
        )
        or (
            command[1] == "-B"
            and command[2].endswith("/scripts/build_phase6_recovery_rebind.py")
            and command[3] in {"reference", "smoke"}
            and "--publish" in command
            and "--source-manifest-v9-sha256" in command
            and "--recovery-authority-sha256" in command
        )
        for command in commands
    )
    smoke_rebind = next(
        node for node in plan["nodes"] if node["id"] == "smoke_numeric_cpu_rebind"
    )
    assert smoke_rebind["environment"] == {
        "CUDA_VISIBLE_DEVICES": "",
        "GIT_OPTIONAL_LOCKS": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    assert smoke_rebind["outputs"][-1].endswith("/evidence_rebinding_receipt_v6.json")
    training = next(node for node in plan["nodes"] if node["id"] == "training_requests")
    assert training["inputs"] == [smoke_rebind["outputs"][-1]]
    assert training["requires_evidence_rebinding_receipt_sha256"] is True
    assert all(
        "report" not in node.get("producer", "").lower()
        or "preflight" in node.get("producer", "").lower()
        or node["id"] == "reference_numeric_cpu_rebind"
        for node in plan["nodes"]
    )


def test_plan_rejects_cycle_future_edge_and_duplicate_output(dag):
    plan = dag.build_artifact_dag_plan()
    cyclic = deepcopy(plan)
    cyclic["nodes"][0]["depends_on"] = ["final_sink"]
    with pytest.raises(dag.Phase6ArtifactDagError, match="cyclic"):
        dag.validate_artifact_dag_plan(cyclic)
    duplicate = deepcopy(plan)
    duplicate["nodes"][1]["outputs"][0] = duplicate["nodes"][0]["outputs"][0]
    with pytest.raises(dag.Phase6ArtifactDagError, match="duplicate"):
        dag.validate_artifact_dag_plan(duplicate)


@pytest.mark.parametrize("tamper", ["allowed_output", "dependency", "producer"])
def test_plan_rejects_exact_node_contract_tampering(dag, tamper):
    plan = dag.build_artifact_dag_plan()
    changed = deepcopy(plan)
    if tamper == "allowed_output":
        changed["nodes"][4]["outputs"][0] = (
            f"{dag.OPERATIONAL_ROOT}/arbitrary-unregistered.json"
        )
    elif tamper == "dependency":
        changed["nodes"][-1]["depends_on"] = []
    else:
        changed["nodes"][4]["producer"] = "unregistered producer"
    with pytest.raises(dag.Phase6ArtifactDagError, match="artifact DAG node"):
        dag.validate_artifact_dag_plan(changed)


def _evidence_consumer_fixture(
    dag,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    evidence_created: str = "2026-07-25T00:00:01Z",
    reference_created: str = "2026-07-25T00:00:00Z",
    smoke_created: str = "2026-07-25T00:00:01Z",
    wrong_builder_manifest_sha: bool = False,
):
    if os.name != "posix":
        pytest.skip("formal receipt metadata modes are a POSIX contract")

    def publish(name: str, data: bytes, mode: int = 0o400):
        path = (tmp_path / name).resolve()
        path.write_bytes(data)
        os.chmod(path, mode)
        return {
            "mode": f"{mode:04o}",
            "path": str(path),
            "sha256": sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    builder_source = f"""\
def pin_from_value(value, _label):
    return value
def validate_reference_rebind_receipt(*_args, **_kwargs):
    return {{"created_at_utc": {reference_created!r}}}
def validate_smoke_rebind_receipt(*_args, **_kwargs):
    return {{"created_at_utc": {smoke_created!r}}}
""".encode("ascii")
    builder_pin = publish("build_phase6_recovery_rebind.py", builder_source, 0o400)
    builder_sha = "f" * 64 if wrong_builder_manifest_sha else builder_pin["sha256"]
    source_manifest = {
        "files": [
            {
                "path": "scripts/build_phase6_recovery_rebind.py",
                "realpath": builder_pin["path"],
                "sha256": builder_sha,
                "size_bytes": builder_pin["size_bytes"],
            }
        ]
    }
    source_pin = publish(
        "phase6_source_runtime_manifest_v9.json",
        dag.canonical_json_bytes(source_manifest),
    )
    authority_pin = publish("recovery_authority.json", b"authority\n")
    reference_pin = publish("reference_rebind_receipt.json", b"reference\n")
    smoke_pin = publish("smoke_rebind_receipt.json", b"smoke\n")
    gate_pin = publish("smoke_gate.json", b"gate\n")
    report_pin = publish("smoke_report.json", b"report\n", 0o600)
    request_pin = publish("smoke_request.json", b"request\n", 0o444)
    artifacts = {
        "source_manifest": source_pin["path"],
        "recovery_authority": authority_pin["path"],
        "reference_rebind_receipt": reference_pin["path"],
        "smoke_rebind_receipt": smoke_pin["path"],
        "evidence_rebinding_receipt": str(
            (tmp_path / "evidence_rebinding_receipt.json").resolve()
        ),
        "real_2b_smoke_gate": gate_pin["path"],
        "smoke_preflight_report": report_pin["path"],
        "smoke_preflight_request": request_pin["path"],
    }
    evidence = {
        "artifact_role": "phase6_planrow_recovery_historical_gpu_evidence_rebinding",
        "created_at_utc": evidence_created,
        "gpu_reexecution_performed": False,
        "historical_gpu_evidence_reused": True,
        **dag.recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "reference_rebind_receipt": reference_pin,
        "recovery_authority": authority_pin,
        "schema_version": "sana-phase6-evidence-rebinding-receipt-v6",
        "smoke_rebind_receipt": smoke_pin,
        "source_manifest_v9": source_pin,
        "status": "pass",
        "torch_imported": False,
        "validation": {
            "reference_504_numeric_projection_byte_identical": True,
            "reference_spot_3_numeric_projection_byte_identical": True,
            "smoke_scientific_projection_byte_identical": True,
        },
        "versioned_outputs": {
            "gate_v8": gate_pin,
            "reference_receipt": reference_pin,
            "report_v10": report_pin,
            "request_v10": request_pin,
            "smoke_receipt": smoke_pin,
        },
    }
    evidence_pin = publish(
        "evidence_rebinding_receipt.json", dag.canonical_json_bytes(evidence)
    )
    paths = types.SimpleNamespace(artifacts=artifacts)
    monkeypatch.setattr(dag, "RECOVERY_REBIND_BUILDER", builder_pin["path"])
    monkeypatch.setitem(
        dag.FROZEN_SHA256, "recovery_authority", authority_pin["sha256"]
    )
    return lambda: dag.validate_evidence_rebinding_receipt(
        evidence_pin["sha256"], paths=paths
    )


def test_evidence_consumer_executes_only_manifest_captured_builder_bytes(
    dag, tmp_path, monkeypatch
):
    validate = _evidence_consumer_fixture(dag, tmp_path, monkeypatch)
    assert validate()["sha256"]


def test_evidence_consumer_rejects_builder_manifest_pin_drift(
    dag, tmp_path, monkeypatch
):
    validate = _evidence_consumer_fixture(
        dag, tmp_path, monkeypatch, wrong_builder_manifest_sha=True
    )
    with pytest.raises(dag.Phase6ArtifactDagError, match="SHA256"):
        validate()


@pytest.mark.parametrize(
    ("evidence_created", "reference_created", "smoke_created", "message"),
    [
        (
            "2999-01-01T00:00:00Z",
            "2026-07-25T00:00:00Z",
            "2026-07-25T00:00:01Z",
            "future-dated",
        ),
        (
            "2026-07-25T00:00:02Z",
            "2026-07-25T00:00:00Z",
            "2026-07-25T00:00:01Z",
            "chronology",
        ),
        (
            "2026-07-25T00:00:01Z",
            "2026-07-25T00:00:02Z",
            "2026-07-25T00:00:01Z",
            "chronology",
        ),
    ],
)
def test_evidence_consumer_rejects_future_or_mismatched_creation_time(
    dag,
    tmp_path,
    monkeypatch,
    evidence_created,
    reference_created,
    smoke_created,
    message,
):
    validate = _evidence_consumer_fixture(
        dag,
        tmp_path,
        monkeypatch,
        evidence_created=evidence_created,
        reference_created=reference_created,
        smoke_created=smoke_created,
    )
    with pytest.raises(dag.Phase6ArtifactDagError, match=message):
        validate()


@pytest.mark.parametrize("purpose", ["reference_precompute", "smoke", "training"])
def test_request_shape_accepts_only_exact_purpose_contract(dag, purpose, monkeypatch):
    monkeypatch.setitem(dag.FROZEN_SHA256, "dataset_contract", TEST_DATASET_SHA256)
    request = _shape_request(dag, purpose)
    assert (
        dag.validate_preflight_request_shape(request, expected_purpose=purpose)
        == request
    )
    request["recovery_cohort_started"] = True
    with pytest.raises(dag.Phase6ArtifactDagError, match="lifecycle"):
        dag.validate_preflight_request_shape(request, expected_purpose=purpose)


def test_request_rejects_dataset_digest_drift(dag):
    assert dag.FROZEN_SHA256["dataset_contract"] == (
        "29a66c9e408ae68faa10705640ee414fbe9de96e28ab0841b1df60abbe6466ab"
    )
    request = _shape_request(dag, "reference_precompute")
    with pytest.raises(dag.Phase6ArtifactDagError, match="frozen artifact"):
        dag.validate_preflight_request_shape(
            request, expected_purpose="reference_precompute"
        )


def test_training_authorization_is_fixed_and_non_resumable(dag):
    authorization = dag.training_authorization("T1_E0A1", "a" * 64)
    assert authorization["factors"] == {"A": True, "E": False, "T": True}
    assert authorization["run_id"] == "phase6_20260725_T1_E0A1_planrowfix_r2"
    assert authorization["run_directory"].endswith(
        "/arms/T1_E0A1/phase6_20260725_T1_E0A1_planrowfix_r2"
    )
    assert authorization["resume_allowed"] is False
    assert authorization["closed_loop_allowed"] is False
    with pytest.raises(dag.Phase6ArtifactDagError, match="unknown"):
        dag.training_authorization("T0_E1A0", "a" * 64)


def test_request_bytes_are_canonical_and_hash_bound(dag, monkeypatch):
    monkeypatch.setitem(dag.FROZEN_SHA256, "dataset_contract", TEST_DATASET_SHA256)
    request = _shape_request(dag, "training")
    data = dag.canonical_json_bytes(request)
    digest = sha256(data).hexdigest()
    assert (
        dag.validate_preflight_request_bytes(
            data,
            expected_sha256=digest,
            expected_purpose="training",
        )
        == request
    )
    pretty = json.dumps(request, indent=2).encode() + b"\n"
    with pytest.raises(dag.Phase6ArtifactDagError, match="canonical"):
        dag.validate_preflight_request_bytes(
            pretty,
            expected_sha256=sha256(pretty).hexdigest(),
            expected_purpose="training",
        )
    duplicate = b'{"schema_version":"x","schema_version":"y"}\n'
    with pytest.raises(dag.Phase6ArtifactDagError, match="duplicate"):
        dag.validate_preflight_request_bytes(
            duplicate,
            expected_sha256=sha256(duplicate).hexdigest(),
            expected_purpose="training",
        )


def test_stable_file_hash_and_exclusive_write(dag, tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b"payload")
    digest = sha256(b"payload").hexdigest()
    assert dag.stable_file_sha256(source, expected_sha256=digest) == digest
    with pytest.raises(dag.Phase6ArtifactDagError, match="mismatch"):
        dag.stable_file_sha256(source, expected_sha256="a" * 64)

    destination = tmp_path / "out.json"
    monkeypatch.setattr(
        dag, "_validate_output_parent", lambda path, **_kwargs: Path(path)
    )
    assert (
        dag._exclusive_write(str(destination), b"first", paths=dag.FROZEN_PATHS)
        == sha256(b"first").hexdigest()
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        dag._exclusive_write(str(destination), b"second", paths=dag.FROZEN_PATHS)
    assert destination.read_bytes() == b"first"


def test_generic_file_hash_rejects_regular_path_replacement(dag, tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    replacement = tmp_path / "replacement.json"
    source.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    real_lstat = dag.os.lstat
    source_lstats = 0

    def replace_before_final_lstat(path, *args, **kwargs):
        nonlocal source_lstats
        if Path(path) == source:
            source_lstats += 1
            if source_lstats == 2:
                replacement.replace(source)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(dag.os, "lstat", replace_before_final_lstat)
    with pytest.raises(dag.Phase6ArtifactDagError, match="changed while hashing"):
        dag.stable_file_sha256(source)
    assert source.read_bytes() == b"replacement"


def _file_symlink_or_skip(link: Path, target: str | Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"real file symlinks are unavailable: {exc}")
    assert link.is_symlink()


def test_runtime_file_hash_accepts_a_real_final_symlink(dag, tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.safetensors"
    target.write_bytes(b"checkpoint-payload")
    link = tmp_path / "checkpoint-final.safetensors"
    _file_symlink_or_skip(link, target.name)
    digest = sha256(target.read_bytes()).hexdigest()

    with pytest.raises(dag.Phase6ArtifactDagError, match="final symlink"):
        dag.stable_file_sha256(link, expected_sha256=digest)
    assert dag.stable_runtime_file_sha256(link, expected_sha256=digest) == digest

    monkeypatch.setattr(dag, "_MAX_METADATA_BYTES", target.stat().st_size - 1)
    with pytest.raises(dag.Phase6ArtifactDagError, match="metadata file is too large"):
        dag.stable_file_sha256(target, metadata_only=True)

    directory_link = tmp_path / "directory-final"
    directory_target = tmp_path / "directory-target"
    directory_target.mkdir()
    _file_symlink_or_skip(directory_link, directory_target.name)
    with pytest.raises(dag.Phase6ArtifactDagError, match="target is not regular"):
        dag.stable_runtime_file_sha256(directory_link)

    broken_link = tmp_path / "broken-final"
    _file_symlink_or_skip(broken_link, "missing-target")
    with pytest.raises(dag.Phase6ArtifactDagError, match="cannot inspect"):
        dag.stable_runtime_file_sha256(broken_link)


def test_runtime_file_hash_rejects_final_symlink_retarget(dag, tmp_path, monkeypatch):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.write_bytes(b"first-checkpoint")
    second.write_bytes(b"second-checkpoint")
    link = tmp_path / "checkpoint.safetensors"
    _file_symlink_or_skip(link, first.name)
    real_read = dag.os.read
    retargeted = False

    def retarget_after_read(descriptor, size):
        nonlocal retargeted
        data = real_read(descriptor, size)
        if not retargeted:
            retargeted = True
            link.unlink()
            link.symlink_to(second.name)
        return data

    monkeypatch.setattr(dag.os, "read", retarget_after_read)
    with pytest.raises(dag.Phase6ArtifactDagError, match="changed while hashing"):
        dag.stable_runtime_file_sha256(link)


def test_runtime_file_hash_rejects_final_symlink_entry_instability(
    dag, tmp_path, monkeypatch
):
    target = tmp_path / "checkpoint.safetensors"
    target.write_bytes(b"checkpoint")
    link = tmp_path / "checkpoint-final.safetensors"
    _file_symlink_or_skip(link, target.name)
    real_read = dag.os.read
    replaced = False

    def replace_link_after_read(descriptor, size):
        nonlocal replaced
        data = real_read(descriptor, size)
        if not replaced:
            replaced = True
            link.unlink()
            # A different link payload reaches the same target and realpath.
            link.symlink_to(target.resolve())
        return data

    monkeypatch.setattr(dag.os, "read", replace_link_after_read)
    with pytest.raises(
        dag.Phase6ArtifactDagError, match="declared runtime path changed"
    ):
        dag.stable_runtime_file_sha256(link)


def test_runtime_roles_use_only_the_dedicated_runtime_pin_reader(dag, monkeypatch):
    seen = []

    def runtime_pin(path, *, expected_sha256):
        seen.append((path, expected_sha256))
        return {"path": path, "sha256": expected_sha256}

    def generic_pin_forbidden(*_args, **_kwargs):
        raise AssertionError("runtime roles must not use the generic pin reader")

    paths = types.SimpleNamespace(
        runtime_files={role: f"/runtime/{role}" for role in dag.RUNTIME_ROLES}
    )
    monkeypatch.setattr(dag, "pin_runtime_file", runtime_pin)
    monkeypatch.setattr(dag, "pin_file", generic_pin_forbidden)

    observed = dag._runtime_pins(paths=paths)
    assert tuple(observed) == dag.RUNTIME_ROLES
    assert seen == [
        (f"/runtime/{role}", dag.FROZEN_SHA256[role]) for role in dag.RUNTIME_ROLES
    ]


def test_stage_publication_rolls_back_and_exactly_recovers(dag, tmp_path, monkeypatch):
    monkeypatch.setattr(
        dag, "_validate_output_parent", lambda path, **_kwargs: Path(path)
    )
    first = str(tmp_path / "first.json")
    second = str(tmp_path / "second.json")
    payloads = {first: b"first", second: b"second"}
    real_write = dag._exclusive_write
    calls = 0

    def fail_second(path, payload, *, paths):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fault injection")
        return real_write(path, payload, paths=paths)

    monkeypatch.setattr(dag, "_exclusive_write", fail_second)
    with pytest.raises(OSError, match="fault injection"):
        dag._publish_payload_set(payloads, paths=dag.FROZEN_PATHS)
    assert not Path(first).exists()
    assert not Path(second).exists()

    monkeypatch.setattr(dag, "_exclusive_write", real_write)
    real_write(first, b"first", paths=dag.FROZEN_PATHS)
    observed = dag._publish_payload_set(payloads, paths=dag.FROZEN_PATHS)
    assert observed == {
        first: sha256(b"first").hexdigest(),
        second: sha256(b"second").hexdigest(),
    }
    assert Path(first).read_bytes() == b"first"
    assert Path(second).read_bytes() == b"second"


def test_stage_publication_rejects_wrong_partial_before_new_write(
    dag, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        dag, "_validate_output_parent", lambda path, **_kwargs: Path(path)
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"wrong")
    with pytest.raises(dag.Phase6ArtifactDagError, match="mismatch"):
        dag._publish_payload_set(
            {str(first): b"expected", str(second): b"second"},
            paths=dag.FROZEN_PATHS,
        )
    assert not second.exists()


def _stub_reference_transaction(dag, tmp_path, monkeypatch):
    config_path = tmp_path / "phase1_action_reference_precompute.yaml"
    request_path = tmp_path / "reference-preflight-request.json"
    paths = types.SimpleNamespace(
        artifacts={"reference_preflight_request": str(request_path)},
        operational_root="/frozen/operational-root",
        reference_config=str(config_path),
        repository_root="/frozen/repository-root",
    )
    monkeypatch.setattr(
        dag, "_validate_output_parent", lambda path, **_kwargs: Path(path)
    )
    monkeypatch.setattr(
        dag, "build_reference_precompute_config", lambda **_kwargs: {"config": True}
    )
    monkeypatch.setattr(dag, "_canonical_yaml_bytes", lambda _value: b"config\n")
    monkeypatch.setattr(
        dag,
        "_common_fixed_artifact_pins",
        lambda **_kwargs: {"registry": {"path": "/registry", "sha256": "1" * 64}},
    )
    monkeypatch.setattr(
        dag,
        "_runtime_pins",
        lambda **_kwargs: {
            "phase1_checkpoint": {"path": "/checkpoint", "sha256": "2" * 64}
        },
    )
    monkeypatch.setattr(
        dag,
        "_source_pin",
        lambda **_kwargs: {"path": "/source", "sha256": "3" * 64},
    )
    monkeypatch.setattr(
        dag,
        "validate_preflight_request_shape",
        lambda request, **_kwargs: deepcopy(dict(request)),
    )
    monkeypatch.setattr(
        dag,
        "validate_preflight_request_bytes",
        lambda _data, **_kwargs: {},
    )
    return paths, config_path, request_path


def test_reference_runtime_pin_failure_precedes_any_output(dag, tmp_path, monkeypatch):
    paths, config_path, request_path = _stub_reference_transaction(
        dag, tmp_path, monkeypatch
    )

    def fail_runtime_pins(**_kwargs):
        raise dag.Phase6ArtifactDagError("checkpoint pin failure")

    monkeypatch.setattr(dag, "_runtime_pins", fail_runtime_pins)
    with pytest.raises(dag.Phase6ArtifactDagError, match="checkpoint pin failure"):
        dag.materialize_reference_inputs(paths=paths)
    assert not config_path.exists()
    assert not request_path.exists()


@pytest.mark.parametrize(
    "failure_stage", ["fresh-validation", "publication", "published-validation"]
)
def test_reference_request_failure_rolls_back_new_config(
    dag, tmp_path, monkeypatch, failure_stage
):
    paths, config_path, request_path = _stub_reference_transaction(
        dag, tmp_path, monkeypatch
    )
    if failure_stage == "fresh-validation":
        monkeypatch.setattr(
            dag,
            "_fresh_preflight_report",
            lambda _request: {"purpose": "reference_precompute", "status": "fail"},
        )
        expected_error = dag.Phase6ArtifactDagError
    else:
        monkeypatch.setattr(
            dag,
            "_fresh_preflight_report",
            lambda _request: {"purpose": "reference_precompute", "status": "pass"},
        )
        if failure_stage == "publication":
            real_write = dag._exclusive_write

            def fail_request_publication(path, payload, *, paths):
                if path == str(request_path):
                    raise OSError("request publication failure")
                return real_write(path, payload, paths=paths)

            monkeypatch.setattr(dag, "_exclusive_write", fail_request_publication)
            expected_error = OSError
        else:

            def fail_published_validation(_data, **_kwargs):
                raise dag.Phase6ArtifactDagError("published validation failure")

            monkeypatch.setattr(
                dag, "validate_preflight_request_bytes", fail_published_validation
            )
            expected_error = dag.Phase6ArtifactDagError

    with pytest.raises(expected_error):
        dag.materialize_reference_inputs(paths=paths)
    assert not config_path.exists()
    assert not request_path.exists()


def test_reference_request_failure_preserves_exact_existing_config(
    dag, tmp_path, monkeypatch
):
    paths, config_path, request_path = _stub_reference_transaction(
        dag, tmp_path, monkeypatch
    )
    config_path.write_bytes(b"config\n")
    before = config_path.stat()
    monkeypatch.setattr(
        dag,
        "_fresh_preflight_report",
        lambda _request: {"purpose": "reference_precompute", "status": "fail"},
    )

    with pytest.raises(dag.Phase6ArtifactDagError, match="validation did not pass"):
        dag.materialize_reference_inputs(paths=paths)
    after = config_path.stat()
    assert config_path.read_bytes() == b"config\n"
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert not request_path.exists()


def test_reference_config_is_full_t0_projection_but_not_a_training_entrypoint(
    dag, monkeypatch
):
    arm, _reference = _install_dependency_modules(dag)
    preflight_path = CANDIDATE / "src/sana_wam/train/phase6_preflight.py"
    if not preflight_path.is_file():
        preflight_path = (
            PHASE6_ROOT
            / "action_reference_integration_fixes/src/sana_wam/train/phase6_preflight.py"
        )
    preflight = _load(
        "sana_wam.train.phase6_preflight",
        preflight_path,
    )
    assert tuple(arm.ARM_NAMES) == dag.ARM_NAMES
    assert preflight.REQUEST_SCHEMA_VERSION == dag.REQUEST_SCHEMA_VERSION
    assert preflight._REFERENCE_PRECOMPUTE_AUTHORIZATION == dag.REFERENCE_AUTHORIZATION
    assert preflight._SMOKE_AUTHORIZATION == dag.SMOKE_AUTHORIZATION
    assert preflight._REFERENCE_PRECOMPUTE_ARTIFACT_ROLES == dag.FIXED_ARTIFACT_ROLES
    assert preflight._SMOKE_ARTIFACT_ROLES == (
        *dag.FIXED_ARTIFACT_ROLES,
        *dag.SMOKE_ADDITIONAL_ROLES,
    )
    assert preflight._TRAINING_ARTIFACT_ROLES == (
        *dag.FIXED_ARTIFACT_ROLES,
        *dag.SMOKE_ADDITIONAL_ROLES,
        *dag.TRAINING_ADDITIONAL_ROLES,
    )
    monkeypatch.setitem(dag.FROZEN_SHA256, "dataset_contract", "d" * 64)
    monkeypatch.setattr(dag, "stable_file_sha256", lambda *_args, **_kwargs: "9" * 64)
    config = dag.build_reference_precompute_config()
    projection = arm.build_phase6_arm_projection("T0_E0A0")
    assert config["model"] == projection["model"]
    assert config["dataloader"] == projection["dataloader"]
    assert (
        config["model"]["video_backbone"]["continuous_timestep_conditioning"] is False
    )
    assert config["model"]["architecture"]["video_local_expansion_weight"] == 0.0
    assert (
        config["model"]["architecture"]["action_video_memory_adapter"]["enabled"]
        is False
    )
    training = config["training"]
    assert training["phase1_action_reference_precompute_mode"] is True
    assert training["init_checkpoint_sha256"] == dag.FROZEN_SHA256["phase1_checkpoint"]
    for key in (
        "phase6_expansion_eligibility_amendment",
        "phase6_expansion_eligibility_amendment_sha256",
        "phase6_expansion_support_contract_sha256",
        "phase6_padding_semantics_amendment",
        "phase6_padding_semantics_amendment_sha256",
        "phase6_padding_semantics_contract_sha256",
    ):
        assert training[key] == arm.KNOWN_TRAINING_PINS[key]
    forbidden = {
        "phase6_arm",
        "phase6_run_id",
        "output_dir",
        "phase6_launch_required",
        "save_steps",
        "optimizer_master_weights",
        "video_lr",
        "warmup_steps",
        "lr_schedule_steps",
        "trainable_parameter_patterns",
    }
    assert forbidden.isdisjoint(training)
    assert all(
        not key.startswith("phase1_action_reference_artifact") for key in training
    )
    _assert_torch_module_unchanged()


def test_materialization_pins_canonical_round_trip_with_final_dependency(
    dag, tmp_path, monkeypatch
):
    arm, _reference = _install_dependency_modules(dag)

    def publish(name: str, payload: bytes) -> dict[str, str]:
        path = (tmp_path / name).resolve()
        path.write_bytes(payload)
        return {"path": str(path), "sha256": sha256(payload).hexdigest()}

    authority_payload = arm.canonical_json_bytes({"status": "synthetic-test"})
    authority = publish("recovery-authority.json", authority_payload)
    monkeypatch.setattr(arm, "PLANROW_RECOVERY_AUTHORITY_PATH", authority["path"])
    monkeypatch.setattr(arm, "PLANROW_RECOVERY_AUTHORITY_SHA256", authority["sha256"])
    monkeypatch.setattr(
        arm,
        "validate_phase6_planrow_recovery_authority",
        lambda **_kwargs: {"status": "synthetic-test"},
    )

    common = {}
    for role, _base in arm.COMMON_DYNAMIC_PIN_FIELDS:
        if role == "arm_projection_manifest":
            payload = arm.canonical_json_bytes(
                arm.build_phase6_arm_projection_manifest()
            )
        elif role == "projection_equivalence_receipt":
            payload = arm.canonical_json_bytes(
                arm.build_phase6_projection_equivalence_receipt()
            )
        elif role == "recovery_authority":
            common[role] = authority
            continue
        else:
            payload = f"common:{role}".encode()
        common[role] = publish(f"common-{role}.bin", payload)
    arms = {}
    for arm_name in arm.ARM_NAMES:
        arm_pins = {}
        for role, _base in arm.ARM_DYNAMIC_PIN_FIELDS:
            payload = (
                arm.canonical_json_bytes(arm.build_phase6_arm_projection(arm_name))
                if role == "scientific_projection"
                else f"{arm_name}:{role}".encode()
            )
            arm_pins[role] = publish(f"{arm_name}-{role}.bin", payload)
        arms[arm_name] = arm_pins
    value = {
        "arms": arms,
        "common": common,
        **arm.recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "schema_version": arm.ARM_MATERIALIZATION_PINS_SCHEMA_VERSION,
    }
    data = arm.canonical_json_bytes(value)
    parsed = json.loads(data)
    assert tuple(parsed["arms"]) != tuple(arm.ARM_NAMES)
    arm.validate_phase6_materialization_pins(parsed, verify_files=True)
    pins_path = tmp_path / "materialization-pins.json"
    pins_path.write_bytes(data)
    loaded = arm.load_phase6_materialization_pins(
        pins_path, expected_sha256=sha256(data).hexdigest()
    )
    assert loaded == parsed


def test_materialization_pins_require_real_revalidated_reports(dag, monkeypatch):
    def report_pair(request, report, *, expected_purpose):
        assert expected_purpose in {"reference_precompute", "smoke", "training"}
        return (
            {"path": request, "sha256": sha256(request.encode()).hexdigest()},
            {"path": report, "sha256": sha256(report.encode()).hexdigest()},
        )

    def fake_pin(path, **_kwargs):
        return {"path": path, "sha256": sha256(path.encode()).hexdigest()}

    captured = []

    def validate(value, *, verify_files):
        captured.append(deepcopy(value))
        assert verify_files is True

    fake_arm = types.ModuleType("sana_wam.train.phase6_arm_config")
    fake_arm.validate_phase6_materialization_pins = validate
    monkeypatch.setitem(sys.modules, "sana_wam.train.phase6_arm_config", fake_arm)
    monkeypatch.setattr(dag, "_load_revalidated_report_pin", report_pair)
    monkeypatch.setattr(dag, "pin_file", fake_pin)
    value = dag.build_materialization_pins()
    assert captured[0] == value
    assert captured[1] == json.loads(dag.canonical_json_bytes(value))
    assert tuple(value["arms"]) == dag.ARM_NAMES
    assert set(value["common"]) == {
        "dataset_contract",
        "source_manifest",
        "runtime_support_manifest",
        "projection_equivalence_receipt",
        "recovery_authority",
        "reference_preflight_request",
        "reference_preflight_report",
        "reference_rebind_receipt",
        "action_reference",
        "action_reference_spot_check",
        "arm_projection_manifest",
        "smoke_runtime_config",
        "smoke_preflight_request",
        "smoke_preflight_report",
        "smoke_rebind_receipt",
        "evidence_rebinding_receipt",
        "real_2b_smoke_gate",
    }
    for arm_name in dag.ARM_NAMES:
        assert set(value["arms"][arm_name]) == {
            "scientific_projection",
            "training_preflight_request",
            "training_preflight_report",
        }


def test_materialization_order_incompatibility_fails_before_publication(
    dag, monkeypatch
):
    def report_pair(request, report, *, expected_purpose):
        return (
            {"path": request, "sha256": sha256(request.encode()).hexdigest()},
            {"path": report, "sha256": sha256(report.encode()).hexdigest()},
        )

    def fake_pin(path, **_kwargs):
        return {"path": path, "sha256": sha256(path.encode()).hexdigest()}

    def order_sensitive_validator(value, *, verify_files):
        assert verify_files is True
        if tuple(value["arms"]) != dag.ARM_NAMES:
            raise ValueError("arm order differs after canonical serialization")

    fake_arm = types.ModuleType("sana_wam.train.phase6_arm_config")
    fake_arm.validate_phase6_materialization_pins = order_sensitive_validator
    monkeypatch.setitem(sys.modules, "sana_wam.train.phase6_arm_config", fake_arm)
    monkeypatch.setattr(dag, "_load_revalidated_report_pin", report_pair)
    monkeypatch.setattr(dag, "pin_file", fake_pin)
    wrote = False

    def forbidden_write(*_args, **_kwargs):
        nonlocal wrote
        wrote = True
        raise AssertionError("publication must not be attempted")

    monkeypatch.setattr(dag, "_publish_payload_set", forbidden_write)
    with pytest.raises(dag.Phase6ArtifactDagError, match="materialization-pins"):
        dag.materialize_training_pins()
    assert wrote is False


def test_cli_plan_and_each_dry_run_write_nothing(tmp_path):
    before = {path.relative_to(CANDIDATE) for path in CANDIDATE.rglob("*")}
    plan_run = subprocess.run(
        [sys.executable, str(CLI_PATH), "plan"],
        check=True,
        capture_output=True,
    )
    plan = json.loads(plan_run.stdout)
    assert plan["schema_version"] == "sana-phase6-artifact-dag-plan-v2"
    assert plan_run.stdout.endswith(b"\n")
    for command in (
        "after-source-freeze",
        "reference-inputs",
        "smoke-request",
        "training-requests",
        "training-pins",
        "final-sink",
    ):
        completed = subprocess.run(
            [sys.executable, str(CLI_PATH), command, "--dry-run"],
            check=True,
            capture_output=True,
            cwd=tmp_path,
        )
        result = json.loads(completed.stdout)
        assert result["dry_run"] is True
        assert result["writes_performed"] is False
    after = {path.relative_to(CANDIDATE) for path in CANDIDATE.rglob("*")}
    # Python may create bytecode caches; no protocol artifact may appear.
    assert {path for path in after - before if "__pycache__" not in path.parts} == set()


def test_independent_report_cli_uses_real_preflight_entrypoints(
    dag, monkeypatch, capsys, tmp_path
):
    request_path = tmp_path / "request.json"
    request_path.write_bytes(b"{}\n")
    calls = []
    preflight = types.ModuleType("sana_wam.train.phase6_preflight")

    class PreflightValidationError(ValueError):
        pass

    preflight.PreflightValidationError = PreflightValidationError
    preflight.canonical_json_bytes = dag.canonical_json_bytes

    def load_request(path, *, expected_sha256):
        calls.append(("load", path, expected_sha256))
        return {"purpose": "training"}

    def validate(request):
        calls.append(("validate", request))
        return {"purpose": "training", "status": "pass"}

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("check-only must not write a report")

    preflight.load_canonical_preflight_request = load_request
    preflight.validate_phase6_preflight = validate
    preflight.run_phase6_preflight = forbidden_run
    if "sana_wam" not in sys.modules:
        package = types.ModuleType("sana_wam")
        package.__path__ = []
        monkeypatch.setitem(sys.modules, "sana_wam", package)
    if "sana_wam.train" not in sys.modules:
        train = types.ModuleType("sana_wam.train")
        train.__path__ = []
        monkeypatch.setitem(sys.modules, "sana_wam.train", train)
    monkeypatch.setitem(sys.modules, "sana_wam.train.phase6_preflight", preflight)
    cli = _load("phase6_independent_preflight_cli_under_test", PREFLIGHT_CLI_PATH)
    assert (
        cli.main(
            [
                "--request",
                str(request_path),
                "--request-sha256",
                "a" * 64,
                "--purpose",
                "training",
                "--check-only",
            ]
        )
        == 0
    )
    assert [call[0] for call in calls] == ["load", "validate"]
    assert "PASS check_only=true" in capsys.readouterr().out


def test_module_source_contains_no_gpu_or_training_execution_import(dag):
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imports
    assert "torch.cuda" not in source
    assert "subprocess" not in source
    assert "optimizer.step" not in source
    assert 'closed_loop_allowed": True' not in source
    preflight_source = PREFLIGHT_CLI_PATH.read_text(encoding="utf-8")
    preflight_tree = ast.parse(preflight_source)
    preflight_imports = {
        alias.name
        for node in ast.walk(preflight_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in preflight_imports
    assert "run_phase6_preflight" in preflight_source
