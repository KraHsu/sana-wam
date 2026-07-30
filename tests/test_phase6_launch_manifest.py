from __future__ import annotations

import json
import os
import shutil
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from sana_wam.train import phase6_arm_config as arm_config
from sana_wam.train import phase6_launch_manifest as launch
from sana_wam.train import phase6_artifact_dag as artifact_dag
from sana_wam.train import phase6_run_integrity as integrity
from sana_wam.train.phase6_arm_config import (
    ARM_NAMES,
    KNOWN_TRAINING_PINS,
    PRIMARY_RUN_DIRECTORIES,
    PRIMARY_RUN_IDS,
    canonical_json_bytes,
    materialize_arm_configs,
    write_phase6_arm_projection_bundle,
    write_phase6_smoke_runtime_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "configs" / "phase6" / "templates"
EXPECTED_SOURCE_REPAIR_AMENDMENT_SHA256 = str(
    "623390e935dcca25d62cdc532109246c589170186ff39d111803ebfc9b674d41",
)
EXPECTED_RUNTIME_ROLES = (
    "student_checkpoint",
    "phase1_checkpoint",
    "action_stats",
)


@pytest.fixture(autouse=True)
def _synthetic_recovery_authority(tmp_path, monkeypatch):
    authority = tmp_path / "recovery_authority.json"
    authority.write_bytes(canonical_json_bytes({"status": "synthetic-test"}))
    digest = sha256(authority.read_bytes()).hexdigest()
    for module in (arm_config, launch):
        monkeypatch.setattr(
            module, "PLANROW_RECOVERY_AUTHORITY_PATH", str(authority.resolve())
        )
        monkeypatch.setattr(module, "PLANROW_RECOVERY_AUTHORITY_SHA256", digest)
        monkeypatch.setattr(
            module,
            "validate_phase6_planrow_recovery_authority",
            lambda **_kwargs: {"status": "synthetic-test"},
        )


def _write_json(path: Path, value) -> dict[str, str]:
    path.write_bytes(canonical_json_bytes(value))
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _fixed_common() -> dict[str, dict[str, str]]:
    training_fields = {
        "protocol": ("phase6_protocol_document", "phase6_protocol_document_sha256"),
        "registry": ("phase6_registry", "phase6_registry_sha256"),
        "storage_amendment": (
            "phase6_operational_storage_amendment",
            "phase6_operational_storage_amendment_sha256",
        ),
        "external_components_manifest": (
            "phase6_external_components_manifest",
            "phase6_external_components_manifest_sha256",
        ),
        "expansion_eligibility_amendment": (
            "phase6_expansion_eligibility_amendment",
            "phase6_expansion_eligibility_amendment_sha256",
        ),
        "padding_semantics_amendment": (
            "phase6_padding_semantics_amendment",
            "phase6_padding_semantics_amendment_sha256",
        ),
        "task_plan": ("phase6_plan_artifact", "phase6_plan_artifact_sha256"),
        "student_checkpoint": ("init_checkpoint", "init_checkpoint_sha256"),
        "phase1_checkpoint": (
            "phase1_reference_checkpoint",
            "phase1_reference_checkpoint_sha256",
        ),
    }
    result = {
        role: {
            "path": KNOWN_TRAINING_PINS[path_key],
            "sha256": KNOWN_TRAINING_PINS[digest_key],
        }
        for role, (path_key, digest_key) in training_fields.items()
    }
    result["action_stats"] = {
        "path": (
            "/home/zch/workspace/sana-wam/logs/sana_principles_audit_20260722/"
            "trajectory_mixed_step0_run/action_stats.npy"
        ),
        "sha256": KNOWN_TRAINING_PINS["action_stats_sha256"],
    }
    return result


def _case(tmp_path, *, verify_artifact_files=False):
    repository = tmp_path / "repo"
    repository.mkdir()
    templates = repository / "configs" / "phase6" / "templates"
    templates.mkdir(parents=True)
    for source in TEMPLATE_DIR.glob("*.yaml.in"):
        (templates / source.name).write_bytes(source.read_bytes())
    projection_dir = repository / "configs" / "phase6" / "projections"
    projection_dir.mkdir()
    projection_manifest = tmp_path / "projection_manifest.json"
    projection_paths = {
        arm: projection_dir / f"projection_{arm}.json" for arm in ARM_NAMES
    }
    write_phase6_arm_projection_bundle(projection_manifest, projection_paths)

    smoke_runtime_dir = repository / "configs" / "phase6" / "smoke_runtime"
    smoke_runtime_dir.mkdir()
    smoke_runtime_bundle = repository / "configs" / "phase6" / "smoke_runtime.json"
    smoke_runtime_digests = write_phase6_smoke_runtime_bundle(
        bundle_path=smoke_runtime_bundle,
        projection_manifest_path=str(projection_manifest.resolve()),
        projection_manifest_sha256=sha256(projection_manifest.read_bytes()).hexdigest(),
        template_dir=templates,
        runtime_config_paths={
            arm: smoke_runtime_dir / f"runtime_{arm}.json" for arm in ARM_NAMES
        },
    )

    common = {}
    for role in (
        "dataset_contract",
        "source_manifest",
        "runtime_support_manifest",
        "reference_preflight_request",
        "reference_preflight_report",
        "reference_rebind_receipt",
        "action_reference",
        "action_reference_spot_check",
        "smoke_rebind_receipt",
        "evidence_rebinding_receipt",
    ):
        common[role] = _write_json(
            tmp_path / f"{role}.json", {"role": role, "registered": True}
        )
    common["arm_projection_manifest"] = {
        "path": str(projection_manifest.resolve()),
        "sha256": sha256(projection_manifest.read_bytes()).hexdigest(),
    }
    common["projection_equivalence_receipt"] = _write_json(
        tmp_path / "projection_equivalence_receipt.json",
        arm_config.build_phase6_projection_equivalence_receipt(),
    )
    authority = Path(arm_config.PLANROW_RECOVERY_AUTHORITY_PATH)
    common["recovery_authority"] = {
        "path": str(authority),
        "sha256": sha256(authority.read_bytes()).hexdigest(),
    }
    common["smoke_runtime_config"] = {
        "path": str(smoke_runtime_bundle.resolve()),
        "sha256": smoke_runtime_digests["bundle"],
    }
    smoke_request_value = {
        "input_config": common["smoke_runtime_config"],
        "purpose": "smoke",
        "schema_version": launch.TRAINING_REQUEST_SCHEMA_VERSION,
    }
    common["smoke_preflight_request"] = _write_json(
        tmp_path / "smoke_request.json", smoke_request_value
    )
    common["smoke_preflight_report"] = _write_json(
        tmp_path / "smoke_report.json",
        {
            **arm_config.recovery_lifecycle(),
            "purpose": "smoke",
            "request_sha256": common["smoke_preflight_request"]["sha256"],
            "status": "pass",
        },
    )
    common["real_2b_smoke_gate"] = _write_json(
        tmp_path / "smoke_gate.json",
        {
            **arm_config.recovery_lifecycle(),
            "registered_before_recovery_cohort_started": True,
            "status": "pass",
        },
    )

    all_common = {**_fixed_common(), **common}
    arm_pins = {}
    for arm in ARM_NAMES:
        projection_pin = {
            "path": str(projection_paths[arm].resolve()),
            "sha256": sha256(projection_paths[arm].read_bytes()).hexdigest(),
        }
        authorization = launch._authorization(arm, projection_pin["sha256"])
        request = {
            "artifacts": {
                role: all_common[role] for role in launch._PREFLIGHT_ARTIFACT_ROLES
            },
            "authorization": authorization,
            "effective_output_root": "/DATA/share/sana_phase6_principled_constraints_20260724",
            "input_config": projection_pin,
            **arm_config.recovery_lifecycle(),
            "purpose": "training",
            "reference_precompute_completed": True,
            "registered_before_recovery_cohort_started": True,
            "repository_root": str(repository.resolve()),
            "runtime_files": {
                role: all_common[role] for role in launch._PREFLIGHT_RUNTIME_ROLES
            },
            "schema_version": launch.TRAINING_REQUEST_SCHEMA_VERSION,
            "smoke_completed": True,
            "source_manifest": all_common["source_manifest"],
        }
        request_pin = _write_json(tmp_path / f"request_{arm}.json", request)
        report_pin = _write_json(
            tmp_path / f"report_{arm}.json",
            {
                "authorization": authorization,
                "input_config": {"sha256": projection_pin["sha256"]},
                **arm_config.recovery_lifecycle(),
                "purpose": "training",
                "reference_precompute_completed": True,
                "registered_before_recovery_cohort_started": True,
                "request_sha256": request_pin["sha256"],
                "schema_version": launch.TRAINING_REPORT_SCHEMA_VERSION,
                "smoke_completed": True,
                "status": "pass",
            },
        )
        arm_pins[arm] = {
            "scientific_projection": projection_pin,
            "training_preflight_request": request_pin,
            "training_preflight_report": report_pin,
        }

    materialization = {
        "arms": arm_pins,
        "common": common,
        **arm_config.recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "schema_version": arm_config.ARM_MATERIALIZATION_PINS_SCHEMA_VERSION,
    }
    config_dir = repository / "configs" / "phase6" / "final"
    materialize_arm_configs(
        template_dir=templates,
        output_dir=config_dir,
        materialization_pins=materialization,
    )
    train_script = repository / "scripts" / "train.py"
    train_script.parent.mkdir()
    train_script.write_text("# launch test entry\n", encoding="utf-8")
    python_executable = repository / ".venv" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    shutil.copy2(sys.executable, python_executable)
    ticket_dir = tmp_path / "tickets"
    ticket_dir.mkdir()
    manifest = launch.build_phase6_launch_manifest(
        config_dir=config_dir,
        repository_root=repository,
        python_executable=python_executable,
        train_script=train_script,
        ticket_dir=ticket_dir,
        verify_artifact_files=verify_artifact_files,
    )
    manifest_path = tmp_path / "launch_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    manifest_sha = sha256(manifest_path.read_bytes()).hexdigest()
    return repository, config_dir, manifest, manifest_path, manifest_sha


def test_launch_manifest_is_exact_five_gpu_dag_sink(tmp_path):
    _repo, _configs, manifest, _path, digest = _case(tmp_path)
    data = canonical_json_bytes(manifest)
    loaded = launch.validate_phase6_launch_manifest_bytes(
        data,
        expected_manifest_sha256=digest,
        verify_files=False,
    )
    assert [entry["gpu_index"] for entry in loaded["arms"]] == [0, 1, 2, 3, 4]
    assert [entry["run_id"] for entry in loaded["arms"]] == [
        PRIMARY_RUN_IDS[arm] for arm in ARM_NAMES
    ]
    assert "common:smoke_runtime_config" in loaded["dag"]["direct_sink_inputs"]
    assert {
        key: loaded[key] for key in arm_config.recovery_lifecycle()
    } == arm_config.recovery_lifecycle()
    assert "closed_loop_started" not in loaded
    for role in (
        "reference_rebind_receipt",
        "smoke_rebind_receipt",
        "evidence_rebinding_receipt",
    ):
        assert role in loaded["common_artifacts"]
    for entry in loaded["arms"]:
        arm = entry["arm"]
        assert entry["output_directory"] == PRIMARY_RUN_DIRECTORIES[arm]
        assert entry["formal_cli_overrides"] == []
        assert entry["argv"][2::2] == ["--config", "--phase6-launch-ticket"]
        for role in (
            "raw_yaml",
            "scientific_projection",
            "training_preflight_request",
            "training_preflight_report",
        ):
            assert f"arm:{arm}:{role}" in loaded["dag"]["direct_sink_inputs"]


def test_launch_manifest_rejects_raw_yaml_or_request_drift(tmp_path):
    _repo, _configs, manifest, _path, _digest = _case(tmp_path)
    changed = deepcopy(manifest)
    changed["arms"][0]["artifacts"]["raw_yaml"]["sha256"] = "e" * 64
    data = canonical_json_bytes(changed)
    with pytest.raises(launch.Phase6LaunchError, match="raw YAML"):
        launch.validate_phase6_launch_manifest_bytes(
            data,
            expected_manifest_sha256=sha256(data).hexdigest(),
            verify_files=False,
        )
    changed = deepcopy(manifest)
    changed["arms"][1]["authorization"]["run_id"] = "override"
    data = canonical_json_bytes(changed)
    with pytest.raises(launch.Phase6LaunchError, match="authorization"):
        launch.validate_phase6_launch_manifest_bytes(
            data,
            expected_manifest_sha256=sha256(data).hexdigest(),
            verify_files=False,
        )


@pytest.mark.parametrize(
    ("key", "alias"),
    (
        ("historical_checkpoint_count", False),
        ("historical_optimizer_step_calls_per_arm", True),
        ("historical_scheduled_lr_scale_at_step0", 0),
    ),
)
def test_launch_manifest_rejects_lifecycle_type_aliases(tmp_path, key, alias):
    _repo, _configs, manifest, _path, _digest = _case(tmp_path)
    manifest[key] = alias
    data = canonical_json_bytes(manifest)
    with pytest.raises(launch.Phase6LaunchError, match=key):
        launch.validate_phase6_launch_manifest_bytes(
            data,
            expected_manifest_sha256=sha256(data).hexdigest(),
            verify_files=False,
        )


_AUTHORIZATION_BOOL_ALIASES = (
    ("closed_loop_allowed", 0),
    ("optimizer_training_allowed", 1),
    ("real_2b_smoke_completed", 1),
    ("reference_gpu_forward_completed", 1),
    ("resume_allowed", 0),
)


@pytest.mark.parametrize(("key", "alias"), _AUTHORIZATION_BOOL_ALIASES)
def test_launch_manifest_rejects_authorization_bool_aliases(tmp_path, key, alias):
    _repo, _configs, manifest, _path, _digest = _case(tmp_path)
    changed = deepcopy(manifest)
    changed["arms"][0]["authorization"][key] = alias
    data = canonical_json_bytes(changed)
    with pytest.raises(launch.Phase6LaunchError, match=key):
        launch.validate_phase6_launch_manifest_bytes(
            data,
            expected_manifest_sha256=sha256(data).hexdigest(),
            verify_files=False,
        )


@pytest.mark.parametrize(("key", "alias"), _AUTHORIZATION_BOOL_ALIASES)
def test_launch_ticket_rejects_authorization_bool_aliases(tmp_path, key, alias):
    repository, _configs, manifest, manifest_path, manifest_sha = _case(tmp_path)
    tickets = launch.issue_phase6_launch_tickets(
        manifest_path, expected_manifest_sha256=manifest_sha, verify_files=False
    )
    ticket_path = tickets[0]
    ticket = json.loads(ticket_path.read_bytes())
    ticket["authorization"][key] = alias
    os.chmod(ticket_path, 0o600)
    ticket_path.write_bytes(canonical_json_bytes(ticket))
    os.chmod(ticket_path, 0o444)
    first = manifest["arms"][0]
    with pytest.raises(launch.Phase6LaunchError, match=key):
        launch.authorize_formal_phase6_invocation(
            ticket_path=ticket_path,
            config_path=first["artifacts"]["raw_yaml"]["path"],
            argv=first["argv"],
            environment=first["environment"],
            working_directory=repository,
            verify_files=False,
        )


@pytest.mark.parametrize(("key", "alias"), _AUTHORIZATION_BOOL_ALIASES)
def test_validated_launch_context_rejects_authorization_bool_aliases(
    tmp_path, key, alias
):
    arm = ARM_NAMES[0]
    authorization = launch._authorization(arm, "a" * 64)
    authorization[key] = alias
    value = {
        "arm": arm,
        "authorization": authorization,
        "config_path": str((tmp_path / "arm.yaml").resolve()),
        "config_sha256": "b" * 64,
        "launch_manifest_path": str((tmp_path / "manifest.json").resolve()),
        "launch_manifest_sha256": "c" * 64,
        "output_directory": PRIMARY_RUN_DIRECTORIES[arm],
        "run_id": PRIMARY_RUN_IDS[arm],
        "ticket_path": str((tmp_path / "ticket.json").resolve()),
        "ticket_sha256": "d" * 64,
    }
    with pytest.raises(integrity.RunIntegrityError, match=key):
        integrity.Phase6ValidatedLaunchContext.from_mapping(value)


def test_validate_only_ticket_and_exact_formal_cli(tmp_path):
    repository, _configs, manifest, manifest_path, manifest_sha = _case(tmp_path)
    tickets = launch.issue_phase6_launch_tickets(
        manifest_path, expected_manifest_sha256=manifest_sha, verify_files=False
    )
    assert len(tickets) == 5
    first = manifest["arms"][0]
    context = launch.authorize_formal_phase6_invocation(
        ticket_path=tickets[0],
        config_path=first["artifacts"]["raw_yaml"]["path"],
        argv=first["argv"],
        environment=first["environment"],
        working_directory=repository,
        verify_files=False,
    )
    assert set(context) == {
        "arm",
        "authorization",
        "config_path",
        "config_sha256",
        "launch_manifest_path",
        "launch_manifest_sha256",
        "output_directory",
        "run_id",
        "ticket_path",
        "ticket_sha256",
    }
    assert context["arm"] == ARM_NAMES[0]
    assert context["authorization"] == first["authorization"]
    assert context["config_path"] == first["artifacts"]["raw_yaml"]["path"]
    assert context["config_sha256"] == first["artifacts"]["raw_yaml"]["sha256"]
    assert context["output_directory"] == first["output_directory"]
    assert context["run_id"] == first["run_id"]
    assert context["launch_manifest_sha256"] == manifest_sha
    assert context["ticket_sha256"] == sha256(tickets[0].read_bytes()).hexdigest()
    tampered_ticket = launch._strict_json(tickets[1].read_bytes(), "test ticket")
    tampered_ticket["historical_scheduled_lr_scale_at_step0"] = 0
    os.chmod(tickets[1], 0o600)
    tickets[1].write_bytes(canonical_json_bytes(tampered_ticket))
    os.chmod(tickets[1], 0o400)
    with pytest.raises(launch.Phase6LaunchError, match="scheduled_lr"):
        second = manifest["arms"][1]
        launch.authorize_formal_phase6_invocation(
            ticket_path=tickets[1],
            config_path=second["artifacts"]["raw_yaml"]["path"],
            argv=second["argv"],
            environment=second["environment"],
            working_directory=repository,
            verify_files=False,
        )
    with pytest.raises(launch.Phase6LaunchError, match="argv differs"):
        launch.authorize_formal_phase6_invocation(
            ticket_path=tickets[0],
            config_path=first["artifacts"]["raw_yaml"]["path"],
            argv=[*first["argv"], "training.max_steps=1"],
            environment=first["environment"],
            working_directory=repository,
            verify_files=False,
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        launch.issue_phase6_launch_tickets(
            manifest_path, expected_manifest_sha256=manifest_sha, verify_files=False
        )


def test_later_arm_authorization_allows_other_run_directory_to_exist(
    tmp_path, monkeypatch
):
    repository, _configs, manifest, manifest_path, manifest_sha = _case(tmp_path)
    tickets = launch.issue_phase6_launch_tickets(
        manifest_path, expected_manifest_sha256=manifest_sha, verify_files=False
    )
    second = manifest["arms"][1]
    original_exists = Path.exists
    other_run = PRIMARY_RUN_DIRECTORIES[ARM_NAMES[0]]

    def selective_exists(path):
        if str(path) == other_run:
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", selective_exists)
    context = launch.authorize_formal_phase6_invocation(
        ticket_path=tickets[1],
        config_path=second["artifacts"]["raw_yaml"]["path"],
        argv=second["argv"],
        environment=second["environment"],
        working_directory=repository,
        verify_files=False,
    )
    assert context["arm"] == ARM_NAMES[1]


def test_ticket_and_manifest_symlinks_are_rejected(tmp_path):
    if os.name == "nt":
        pytest.skip("unprivileged Windows symlink creation is not reliable")
    _repository, _configs, _manifest, manifest_path, manifest_sha = _case(tmp_path)
    link = tmp_path / "manifest_link.json"
    link.symlink_to(manifest_path)
    with pytest.raises(launch.Phase6LaunchError, match="non-symlink"):
        launch.issue_phase6_launch_tickets(
            link, expected_manifest_sha256=manifest_sha, verify_files=False
        )


def _file_symlink_or_skip(link: Path, target: str | Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"real file symlinks are unavailable: {exc}")
    assert link.is_symlink()


def test_source_repair_backlink_remains_freshly_validated():
    assert (
        launch.SOURCE_REPAIR_AMENDMENT_SHA256 == EXPECTED_SOURCE_REPAIR_AMENDMENT_SHA256
    )
    launch._validate_source_repair_amendment()
    assert launch._PREFLIGHT_RUNTIME_ROLES == EXPECTED_RUNTIME_ROLES


@pytest.mark.parametrize(
    "role", ("student_checkpoint", "phase1_checkpoint", "action_stats")
)
def test_only_runtime_roles_accept_a_stable_final_symlink(tmp_path, role):
    target = tmp_path / f"{role}.payload"
    target.write_bytes(f"registered-{role}".encode("ascii"))
    link = tmp_path / f"{role}.link"
    _file_symlink_or_skip(link, target.name)
    digest = sha256(target.read_bytes()).hexdigest()

    assert launch._pin(
        str(link), digest, role, verify_file=True, runtime_file=True
    ) == {"path": str(link), "sha256": digest}
    with pytest.raises(launch.Phase6LaunchError, match="non-symlink"):
        launch._pin(
            str(link),
            digest,
            "metadata artifact",
            verify_file=True,
            runtime_file=False,
        )


def test_runtime_role_rejects_digest_broken_link_and_directory(tmp_path):
    target = tmp_path / "checkpoint.safetensors"
    target.write_bytes(b"checkpoint")
    link = tmp_path / "checkpoint.link"
    _file_symlink_or_skip(link, target.name)
    with pytest.raises(launch.Phase6LaunchError, match="SHA256 mismatch"):
        launch._pin(
            str(link),
            "f" * 64,
            "student_checkpoint",
            verify_file=True,
            runtime_file=True,
        )

    broken = tmp_path / "broken.link"
    _file_symlink_or_skip(broken, "missing.safetensors")
    with pytest.raises(launch.Phase6LaunchError, match="runtime-file validation"):
        launch._pin(
            str(broken),
            "e" * 64,
            "student_checkpoint",
            verify_file=True,
            runtime_file=True,
        )

    directory = tmp_path / "directory"
    directory.mkdir()
    directory_link = tmp_path / "directory.link"
    _file_symlink_or_skip(directory_link, directory.name)
    with pytest.raises(launch.Phase6LaunchError, match="target is not regular"):
        launch._pin(
            str(directory_link),
            "d" * 64,
            "action_stats",
            verify_file=True,
            runtime_file=True,
        )


def test_runtime_role_retargets_fail_closed_and_translate_helper_error(
    tmp_path, monkeypatch
):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.write_bytes(b"first-checkpoint")
    second.write_bytes(b"second-checkpoint")
    link = tmp_path / "checkpoint.link"
    _file_symlink_or_skip(link, first.name)
    real_read = artifact_dag.os.read
    retargeted = False

    def retarget_after_read(descriptor, size):
        nonlocal retargeted
        data = real_read(descriptor, size)
        if not retargeted:
            retargeted = True
            link.unlink()
            link.symlink_to(second.name)
        return data

    monkeypatch.setattr(artifact_dag.os, "read", retarget_after_read)
    with pytest.raises(launch.Phase6LaunchError, match="changed while hashing"):
        launch._pin(
            str(link),
            sha256(first.read_bytes()).hexdigest(),
            "phase1_checkpoint",
            verify_file=True,
            runtime_file=True,
        )

    def helper_failure(*_args, **_kwargs):
        raise artifact_dag.Phase6ArtifactDagError("synthetic helper failure")

    monkeypatch.setattr(artifact_dag, "stable_runtime_file_sha256", helper_failure)
    with pytest.raises(launch.Phase6LaunchError, match="synthetic helper failure"):
        launch._pin(
            str(first),
            sha256(first.read_bytes()).hexdigest(),
            "phase1_checkpoint",
            verify_file=True,
            runtime_file=True,
        )


def test_runtime_role_rejects_final_symlink_entry_replacement(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.safetensors"
    target.write_bytes(b"checkpoint")
    link = tmp_path / "checkpoint.link"
    _file_symlink_or_skip(link, target.name)
    real_read = artifact_dag.os.read
    replaced = False

    def replace_link_after_read(descriptor, size):
        nonlocal replaced
        data = real_read(descriptor, size)
        if not replaced:
            replaced = True
            link.unlink()
            link.symlink_to(target.resolve())
        return data

    monkeypatch.setattr(artifact_dag.os, "read", replace_link_after_read)
    with pytest.raises(launch.Phase6LaunchError, match="declared runtime path changed"):
        launch._pin(
            str(link),
            sha256(target.read_bytes()).hexdigest(),
            "student_checkpoint",
            verify_file=True,
            runtime_file=True,
        )


def test_build_and_validator_route_only_runtime_common_roles_to_helper(
    tmp_path, monkeypatch
):
    seen: list[tuple[str, str]] = []

    def runtime_reader(path, *, expected_sha256):
        seen.append((path, expected_sha256))
        return expected_sha256

    monkeypatch.setattr(artifact_dag, "stable_runtime_file_sha256", runtime_reader)
    _repository, _configs, manifest, _path, digest = _case(
        tmp_path, verify_artifact_files=True
    )
    expected_runtime = _fixed_common()
    assert seen == [
        (expected_runtime[role]["path"], expected_runtime[role]["sha256"])
        for role in EXPECTED_RUNTIME_ROLES
    ]

    seen.clear()
    monkeypatch.setattr(
        launch, "_validate_smoke_with_available_contract", lambda *_args: None
    )
    launch.validate_phase6_launch_manifest_bytes(
        canonical_json_bytes(manifest),
        expected_manifest_sha256=digest,
        verify_files=True,
        require_output_directories_absent=False,
    )
    assert seen == [
        (expected_runtime[role]["path"], expected_runtime[role]["sha256"])
        for role in EXPECTED_RUNTIME_ROLES
    ]


def test_build_and_validator_reject_a_metadata_final_symlink(tmp_path, monkeypatch):
    repository, config_dir, manifest, _path, _digest = _case(tmp_path)
    target = tmp_path / "protocol.json"
    target.write_bytes(b"registered protocol")
    link = tmp_path / "protocol.link"
    _file_symlink_or_skip(link, target.name)
    digest = sha256(target.read_bytes()).hexdigest()
    real_common = launch._config_common_artifacts

    def common_with_protocol_symlink(config):
        common = real_common(config)
        common["protocol"] = {"path": str(link), "sha256": digest}
        return common

    monkeypatch.setattr(
        launch, "_config_common_artifacts", common_with_protocol_symlink
    )
    with pytest.raises(launch.Phase6LaunchError, match="non-symlink"):
        launch.build_phase6_launch_manifest(
            config_dir=config_dir,
            repository_root=repository,
            python_executable=repository / ".venv" / "bin" / "python",
            train_script=repository / "scripts" / "train.py",
            ticket_dir=tmp_path / "tickets",
            verify_artifact_files=True,
        )

    changed = deepcopy(manifest)
    changed["common_artifacts"]["protocol"] = {
        "path": str(link),
        "sha256": digest,
    }
    data = canonical_json_bytes(changed)
    with pytest.raises(launch.Phase6LaunchError, match="non-symlink"):
        launch.validate_phase6_launch_manifest_bytes(
            data,
            expected_manifest_sha256=sha256(data).hexdigest(),
            verify_files=True,
            require_output_directories_absent=False,
        )


@pytest.mark.parametrize(
    "target_path",
    (
        launch.SOURCE_REPAIR_AMENDMENT_PATH,
        launch._SOURCE_REPAIR_REVIEW_SUBJECT_PATH,
        launch._SOURCE_REPAIR_BOOTSTRAP_RECEIPT_PATH,
        launch._SOURCE_REPAIR_REVIEW_RECEIPT_PATHS["A"],
        launch._SOURCE_REPAIR_REVIEW_RECEIPT_PATHS["B"],
    ),
)
def test_source_repair_authority_is_freshly_rehashed(target_path, monkeypatch):
    launch._validate_source_repair_amendment()
    real_reader = launch._stable_read_regular_with_stat

    def tampered_digest(path, name, *, capture):
        digest, data, info = real_reader(path, name, capture=capture)
        if Path(path) == Path(target_path):
            digest = "0" * 64
        return digest, data, info

    monkeypatch.setattr(launch, "_stable_read_regular_with_stat", tampered_digest)
    with pytest.raises(launch.Phase6LaunchError):
        launch._validate_source_repair_amendment()
