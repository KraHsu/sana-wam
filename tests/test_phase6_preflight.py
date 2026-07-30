from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    expansion_support_contract,
    expansion_support_for_window,
)
from sana_wam.dataloader.phase6_padding_semantics import (
    PADDING_SEMANTICS_AMENDMENT_SHA256,
    PADDING_SEMANTICS_CONTRACT_SHA256,
    padding_semantics_contract,
)
from sana_wam.train import phase6_preflight as preflight
from sana_wam.train import phase6_arm_config as arm_config


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    return _digest(path.read_bytes())


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.resolve()


def _compact(path: Path, value: object, *, ensure_ascii: bool = False) -> Path:
    return _write(
        path,
        preflight.canonical_json_bytes(value, ensure_ascii=ensure_ascii),
    )


def _pretty(path: Path, value: object) -> Path:
    return _write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


@dataclass
class _Case:
    root: Path
    repository: Path
    output_root: Path
    artifacts: dict[str, Path]
    runtime: dict[str, Path]
    support: list[Path]
    source_manifest: Path
    input_config: Path
    request_path: Path
    request: dict
    policy: preflight._FrozenPolicy

    def repin_artifact(self, role: str) -> None:
        digest = _file_digest(self.artifacts[role])
        self.request["artifacts"][role]["sha256"] = digest
        self.policy = replace(
            self.policy,
            fixed_artifact_sha256=tuple(
                (name, digest if name == role else pinned)
                for name, pinned in self.policy.fixed_artifact_sha256
            ),
        )

    def write_request(self) -> None:
        _compact(self.request_path, self.request)


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Case:
    runtime_environment = {
        "abi_cache_tag": "cpython-test",
        "distributions": [{"name": "synthetic", "version": "1"}],
        "implementation": "cpython",
        "libc": {"name": "glibc", "version": "test"},
        "platform_machine": "test-machine",
        "platform_system": "TestOS",
        "python_executable_realpath": "/test/python",
        "python_version": "3.12.0",
        "soabi": "cpython-test-soabi",
    }
    vcs = {
        "repository_head": "1" * 40,
        "root_source_status_lines": [],
        "root_source_status_sha256": _digest(b"[]\n"),
        "sana_gitlink_head": "2" * 40,
        "sana_source_status_lines": [],
        "sana_source_status_sha256": _digest(b"[]\n"),
        "sana_worktree_head": "2" * 40,
    }
    monkeypatch.setattr(
        preflight, "_runtime_environment_fingerprint", lambda: runtime_environment
    )
    monkeypatch.setattr(preflight, "_repository_vcs_fingerprint", lambda root: vcs)

    recovery_authority = _compact(
        tmp_path / "artifacts" / "recovery-authority.json",
        {"status": "synthetic-test"},
    )
    recovery_authority_sha = _file_digest(recovery_authority)
    monkeypatch.setattr(
        preflight,
        "PLANROW_RECOVERY_AUTHORITY_PATH",
        str(recovery_authority),
    )
    monkeypatch.setattr(
        preflight,
        "PLANROW_RECOVERY_AUTHORITY_SHA256",
        recovery_authority_sha,
    )
    monkeypatch.setattr(
        preflight,
        "validate_phase6_planrow_recovery_authority",
        lambda **_kwargs: {"status": "synthetic-test"},
    )
    monkeypatch.setattr(
        arm_config,
        "validate_phase6_projection_equivalence_receipt_bytes",
        lambda _data, **_kwargs: {"status": "synthetic-test"},
    )

    repository = (tmp_path / "repo").resolve()
    _write(repository / ".python-version", b"3.12\n")
    _write(repository / "pyproject.toml", b"[project]\nname='test'\n")
    _write(repository / "uv.lock", b"version = 1\n")
    _write(repository / "scripts" / "train.py", b"print('train')\n")
    _write(repository / "src" / "package" / "model.py", b"VALUE = 1\n")
    _write(repository / "tests" / "test_model.py", b"def test_model():\n    pass\n")
    template_source = (
        Path(__file__).resolve().parents[1] / "configs" / "phase6" / "templates"
    )
    for template in template_source.glob("*.yaml.in"):
        _write(
            repository / "configs" / "phase6" / "templates" / template.name,
            template.read_bytes(),
        )
    _write(
        repository / "third_party" / "Sana" / "diffusion" / "model.py",
        b"UPSTREAM = 1\n",
    )
    input_config = _write(
        repository / "configs" / "reference.yaml", b"training:\n  seed: 20260724\n"
    )

    output_root = (tmp_path / "output").resolve()
    output_root.mkdir()
    original_output = (tmp_path / "forbidden-output").resolve()

    runtime = {
        "student_checkpoint": _write(tmp_path / "runtime" / "student", b"student"),
        "phase1_checkpoint": _write(tmp_path / "runtime" / "phase1", b"phase1"),
        "action_stats": _write(tmp_path / "runtime" / "stats", b"stats"),
    }
    runtime_sha = tuple((role, _file_digest(path)) for role, path in runtime.items())
    external = [
        _write(tmp_path / "external" / f"file-{index}", f"ext-{index}".encode())
        for index in range(2)
    ]
    external_policy = tuple(
        (str(path), path.stat().st_size, _file_digest(path)) for path in external
    )
    support = [
        _write(tmp_path / "support" / f"config-{index}.json", b"{}\n")
        for index in range(2)
    ]
    support_policy = tuple(
        (str(path), path.stat().st_size, _file_digest(path)) for path in support
    )

    plan_rows = []
    for index, sigma in enumerate((1.0, 0.9), start=1):
        plan_rows.append(
            {
                "action_sigma": sigma,
                "cycle": 0,
                "domain_seeds": {"video-noise": index},
                "global_step": index,
                "identity": {
                    "dataset_index": 100 + index,
                    "episode_index": index - 1,
                    "episode_path": str(tmp_path / f"episode-{index}.hdf5"),
                    "prompt": f"prompt-{index}",
                    "source_dataset": "RoboTwin",
                    "source_kind": "ordinary_expert",
                    "source_variant": "clean_50",
                    "start_frame": index,
                    "task_name": f"task-{index}",
                },
                "position_in_cycle": index - 1,
            }
        )
    plan = {
        "action_sigmas": [1.0, 0.9, 0.5],
        "holdout_task_sha256": "2" * 64,
        "holdout_tasks": ["holdout"],
        "optimizer_steps": 2,
        "protocol_seed": 20260724,
        "rows": plan_rows,
        "schema_version": "sana-phase6-task-plan-v1",
        "seed_domains": ["video-noise"],
        "sigma_exposures_per_task": 1,
        "source_contract": {
            "dataset": "RoboTwin",
            "kind": "ordinary_expert",
            "variant": "clean_50",
        },
        "task_cycles": 1,
        "train_task_sha256": "1" * 64,
        "train_tasks": ["task-1", "task-2"],
    }
    plan_sha = _digest(preflight.canonical_json_bytes(plan))
    identity_sha = preflight._task_identity_sha256(plan)
    runtime_pins = dict(runtime_sha)

    registry = {
        "action_interface": {},
        "action_non_regression": {},
        "action_stats_sha256": runtime_pins["action_stats"],
        "arms": [],
        "common_training": {},
        "constraints": {},
        "data": {"optimizer_steps": 2},
        "deployment_contract": {},
        "expansion_constraint": {},
        "initial_checkpoint": {
            "path": str(runtime["student_checkpoint"]),
            "role": "student initialization",
            "sha256": runtime_pins["student_checkpoint"],
        },
        "output_root": str(original_output),
        "primary_gates": {},
        "protocol": "synthetic",
        "registered_before_phase6_training_results": True,
        "risk_reference_checkpoint": {
            "path": str(runtime["phase1_checkpoint"]),
            "role": "one-sided action-function reference",
            "sha256": runtime_pins["phase1_checkpoint"],
        },
        "seed": 20260724,
        "time_conditioning": {},
    }
    artifacts: dict[str, Path] = {}
    artifacts["registry"] = _compact(tmp_path / "artifacts" / "registry.json", registry)
    registry_sha = _file_digest(artifacts["registry"])
    artifacts["protocol"] = _write(
        tmp_path / "artifacts" / "protocol.md", b"# protocol\n"
    )
    storage = {
        "amendment": "test",
        "effective_output_root": str(output_root),
        "original_output_root": str(original_output),
        "reason": "test",
        "registered_before_phase6_training_results": True,
        "registered_protocol_sha256": registry_sha,
        "scientific_contract_changed": False,
        "storage_checks": {
            "effective_root_parent_writable": True,
            "filesystem_available_bytes_class": "test",
            "original_root_parent_writable": False,
        },
    }
    artifacts["storage_amendment"] = _pretty(
        tmp_path / "artifacts" / "storage.json", storage
    )
    external_manifest = {
        "components": [
            {"path": path, "sha256": digest, "size_bytes": size}
            for path, size, digest in external_policy
        ],
        "registered_before_phase6_training_results": True,
        "schema_version": "sana-phase6-external-components-v1",
        "training_or_gpu_started": False,
    }
    artifacts["external_components_manifest"] = _pretty(
        tmp_path / "artifacts" / "external.json", external_manifest
    )
    registered_pins = {
        "external_components_sha256": _file_digest(
            artifacts["external_components_manifest"]
        ),
        "protocol_sha256": _file_digest(artifacts["protocol"]),
        "registry_sha256": registry_sha,
        "storage_amendment_sha256": _file_digest(artifacts["storage_amendment"]),
    }
    support_contract = expansion_support_contract()
    expansion_amendment = {
        "candidate_pool_policy": {
            "action_sigma_schedule_changed": False,
            "eligible_candidate_count": 460_738,
            "excluded_candidate_count": 6_300,
            "excluded_per_task": 150,
            "ordinary_dataset_global_indices_changed": False,
            "protocol_seed_changed": False,
            "raw_candidate_count": 467_038,
            "runtime_skip_allowed": False,
            "slot_specific_backfill_allowed": False,
            "strategy": "structural filter before hash",
            "task_count": 42,
            "task_order_changed": False,
        },
        "constraints": {
            "best_of_n": False,
            "closed_loop": False,
            "critics": False,
            "dagger": False,
            "outcome_or_model_based_sampler_selection": False,
            "recovery_annotations": False,
            "reranking": False,
            "task_specific_filter": False,
        },
        "execution_state": {
            "closed_loop_started": False,
            "formal_training_started": False,
            "optimizer_steps": 0,
            "training_started": False,
        },
        "expansion_support_contract": support_contract,
        "expansion_support_contract_sha256": EXPANSION_SUPPORT_CONTRACT_SHA256,
        "incident": {},
        "old_plan_structural_audit": {},
        "reason": "test",
        "registered_before_phase6_training_results": True,
        "registered_contract_pins": registered_pins,
        "schema_version": "sana-phase6-expansion-eligibility-amendment-v1",
        "scientific_factor_changed": False,
        "superseded_plan_pins": {},
    }
    artifacts["expansion_eligibility_amendment"] = _compact(
        tmp_path / "artifacts" / "expansion-amendment.json",
        expansion_amendment,
        ensure_ascii=True,
    )
    padding_contract = padding_semantics_contract()
    padding_amendment = {
        "constraints": {
            "best_of_n": False,
            "closed_loop": False,
            "critics": False,
            "dagger": False,
            "recovery_annotations": False,
            "reranking": False,
            "task_specific_sampler": False,
        },
        "discovery": {},
        "eligibility_dependency": {
            "amendment_sha256": EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
            "support_contract_sha256": EXPANSION_SUPPORT_CONTRACT_SHA256,
        },
        "execution_state": {
            "closed_loop_started": False,
            "formal_training_started": False,
            "optimizer_steps": 0,
            "training_started": False,
        },
        "implementation_requirements": {"padding_semantics": True},
        "padding_semantics_contract": padding_contract,
        "padding_semantics_contract_hash_serialization": "canonical JSON plus LF",
        "padding_semantics_contract_sha256": PADDING_SEMANTICS_CONTRACT_SHA256,
        "reason": "test",
        "registered_before_phase6_training_results": True,
        "registered_contract_pins": registered_pins,
        "schema_version": "sana-phase6-padding-semantics-amendment-v1",
        "scientific_factor_changed": False,
        "source_manifest_at_discovery_sha256": "3" * 64,
        "uniformity": {},
    }
    artifacts["padding_semantics_amendment"] = _compact(
        tmp_path / "artifacts" / "padding-amendment.json",
        padding_amendment,
        ensure_ascii=True,
    )
    task_artifact = {
        "artifact_schema_version": "sana-phase6-task-plan-artifact-v2",
        "expansion_eligibility_amendment_sha256": (
            EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        ),
        "expansion_support_contract": support_contract,
        "expansion_support_contract_sha256": EXPANSION_SUPPORT_CONTRACT_SHA256,
        "identity_sha256": identity_sha,
        "plan": plan,
        "plan_sha256": plan_sha,
    }
    artifacts["task_plan"] = _compact(
        tmp_path / "artifacts" / "task-plan.json", task_artifact
    )
    dataset_rows = []
    for row in plan_rows:
        identity = row["identity"]
        support_fields = expansion_support_for_window(
            episode_length=113,
            start_frame=0,
            window_logical_length=113,
        )
        dataset_rows.append(
            {
                "dataset_index": identity["dataset_index"],
                "episode_index": identity["episode_index"],
                "episode_path": identity["episode_path"],
                "episode_sha256": _digest(b"episode"),
                "global_step": row["global_step"],
                "instruction_source_sha256": _digest(b"instruction"),
                "start_frame": identity["start_frame"],
                "task_name": identity["task_name"],
                "window_logical_length": 113,
                **support_fields,
            }
        )
    dataset = {
        "action_stats_sha256": runtime_pins["action_stats"],
        "episodes": [{"test": True}],
        "expansion_eligibility_amendment_sha256": (
            EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        ),
        "expansion_support_contract": support_contract,
        "expansion_support_contract_sha256": EXPANSION_SUPPORT_CONTRACT_SHA256,
        "identity_sha256": identity_sha,
        "padding_semantics_amendment_sha256": PADDING_SEMANTICS_AMENDMENT_SHA256,
        "padding_semantics_contract": padding_contract,
        "padding_semantics_contract_sha256": PADDING_SEMANTICS_CONTRACT_SHA256,
        "plan_sha256": plan_sha,
        "preprocessing_contract": {"action_stats_sha256": runtime_pins["action_stats"]},
        "rows": dataset_rows,
        "schema_version": "sana-phase6-dataset-contract-v2",
    }
    artifacts["dataset_contract"] = _compact(
        tmp_path / "artifacts" / "dataset.json", dataset, ensure_ascii=True
    )
    runtime_support = {
        "files": [
            {"path": path, "sha256": digest, "size_bytes": size}
            for path, size, digest in support_policy
        ],
        "registered_before_phase6_training_results": True,
        "schema_version": preflight.RUNTIME_SUPPORT_SCHEMA_VERSION,
        "training_or_gpu_started": False,
    }
    artifacts["runtime_support_manifest"] = _compact(
        tmp_path / "artifacts" / "runtime-support.json", runtime_support
    )
    artifacts["projection_equivalence_receipt"] = _compact(
        tmp_path / "artifacts" / "projection-equivalence-receipt.json",
        {"status": "synthetic-test"},
    )
    artifacts["recovery_authority"] = recovery_authority

    fixed_artifacts = tuple(
        (role, _file_digest(path))
        for role, path in artifacts.items()
        if role != "runtime_support_manifest"
    )
    policy = preflight._FrozenPolicy(
        fixed_artifact_sha256=fixed_artifacts,
        runtime_sha256=runtime_sha,
        external_components=external_policy,
        effective_output_root=str(output_root),
        original_output_root=str(original_output),
        plan_sha256=plan_sha,
        identity_sha256=identity_sha,
        optimizer_steps=2,
        seed=20260724,
        runtime_support_files=support_policy,
    )
    source_value = preflight.build_phase6_source_manifest(repository)
    source_manifest = _compact(
        tmp_path / "artifacts" / "source-manifest.json", source_value
    )
    request = {
        "authorization": dict(preflight._REFERENCE_PRECOMPUTE_AUTHORIZATION),
        "artifacts": {
            role: {"path": str(path), "sha256": _file_digest(path)}
            for role, path in artifacts.items()
        },
        "effective_output_root": str(output_root),
        "input_config": {
            "path": str(input_config),
            "sha256": _file_digest(input_config),
        },
        **preflight.recovery_lifecycle(),
        "purpose": "reference_precompute",
        "reference_gpu_started": False,
        "registered_before_recovery_cohort_started": True,
        "repository_root": str(repository),
        "runtime_files": {
            role: {"path": str(path), "sha256": _file_digest(path)}
            for role, path in runtime.items()
        },
        "schema_version": preflight.REQUEST_SCHEMA_VERSION,
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": _file_digest(source_manifest),
        },
    }
    request_path = _compact(output_root / "request.json", request)
    return _Case(
        root=tmp_path,
        repository=repository,
        output_root=output_root,
        artifacts=artifacts,
        runtime=runtime,
        support=support,
        source_manifest=source_manifest,
        input_config=input_config,
        request_path=request_path,
        request=request,
        policy=policy,
    )


def _validate(case: _Case, request: dict | None = None) -> dict:
    return preflight._validate_with_policy(
        case.request if request is None else request,
        policy=case.policy,
    )


def _artifact_value(case: _Case, role: str) -> dict:
    value = json.loads(case.artifacts[role].read_bytes())
    assert isinstance(value, dict)
    return value


def _rewrite_artifact(
    case: _Case,
    role: str,
    value: dict,
    *,
    canonical: bool = True,
) -> None:
    if canonical:
        _compact(
            case.artifacts[role],
            value,
            ensure_ascii=role
            in {
                "dataset_contract",
                "expansion_eligibility_amendment",
                "padding_semantics_amendment",
            },
        )
    else:
        _write(
            case.artifacts[role],
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
        )
    case.repin_artifact(role)


def test_reference_precompute_happy_path_binds_config_source_and_runtime(case: _Case):
    first = _validate(case)
    second = _validate(case)
    assert first == second
    assert first["purpose"] == "reference_precompute"
    assert first["status"] == "pass"
    assert first["pins"]["observed_input_config_sha256"] == _file_digest(
        case.input_config
    )
    assert first["pins"]["observed_runtime_support_manifest_sha256"] == _file_digest(
        case.artifacts["runtime_support_manifest"]
    )
    assert first["input_config"]["sha256"] == _file_digest(case.input_config)
    assert len(first["runtime_support_files"]) == 2


@pytest.mark.parametrize(
    "role",
    ["expansion_eligibility_amendment", "padding_semantics_amendment"],
)
def test_amendments_must_use_canonical_bytes(case: _Case, role: str):
    _rewrite_artifact(case, role, _artifact_value(case, role), canonical=False)
    with pytest.raises(preflight.PreflightValidationError, match="not canonical JSON"):
        _validate(case)


def test_old_task_outer_v1_shape_fails_closed(case: _Case):
    value = _artifact_value(case, "task_plan")
    value["artifact_schema_version"] = "sana-phase6-task-plan-artifact-v1"
    for key in (
        "expansion_eligibility_amendment_sha256",
        "expansion_support_contract",
        "expansion_support_contract_sha256",
    ):
        value.pop(key)
    _rewrite_artifact(case, "task_plan", value)

    with pytest.raises(
        preflight.PreflightValidationError,
        match="task-plan artifact keys differ",
    ):
        _validate(case)


def test_old_dataset_contract_v1_shape_fails_closed(case: _Case):
    value = _artifact_value(case, "dataset_contract")
    value["schema_version"] = "sana-phase6-dataset-contract-v1"
    for key in (
        "expansion_eligibility_amendment_sha256",
        "expansion_support_contract",
        "expansion_support_contract_sha256",
        "padding_semantics_amendment_sha256",
        "padding_semantics_contract",
        "padding_semantics_contract_sha256",
    ):
        value.pop(key)
    _rewrite_artifact(case, "dataset_contract", value)

    with pytest.raises(
        preflight.PreflightValidationError,
        match="dataset contract keys differ",
    ):
        _validate(case)


@pytest.mark.parametrize(
    ("role", "hash_key"),
    [
        (
            "expansion_eligibility_amendment",
            "expansion_support_contract_sha256",
        ),
        ("padding_semantics_amendment", "padding_semantics_contract_sha256"),
    ],
)
def test_amendment_nested_hash_drift_fails_closed(
    case: _Case,
    role: str,
    hash_key: str,
):
    value = _artifact_value(case, role)
    value[hash_key] = _digest(f"{role}-drift".encode())
    _rewrite_artifact(case, role, value)

    with pytest.raises(preflight.PreflightValidationError, match="nested contract SHA"):
        _validate(case)


@pytest.mark.parametrize(
    ("role", "contract_key", "field", "replacement"),
    [
        (
            "expansion_eligibility_amendment",
            "expansion_support_contract",
            "raw_window_frames",
            114,
        ),
        (
            "padding_semantics_amendment",
            "padding_semantics_contract",
            "schema_version",
            "sana-phase6-padding-semantics-contract-v0",
        ),
    ],
)
def test_amendment_nested_contract_value_drift_fails_closed(
    case: _Case,
    role: str,
    contract_key: str,
    field: str,
    replacement: object,
):
    value = _artifact_value(case, role)
    value[contract_key][field] = replacement
    _rewrite_artifact(case, role, value)

    with pytest.raises(preflight.PreflightValidationError, match="contract"):
        _validate(case)


@pytest.mark.parametrize(
    "dependency_key",
    ["amendment_sha256", "support_contract_sha256"],
)
def test_padding_eligibility_dependency_drift_fails_closed(
    case: _Case,
    dependency_key: str,
):
    role = "padding_semantics_amendment"
    value = _artifact_value(case, role)
    value["eligibility_dependency"][dependency_key] = _digest(
        f"{dependency_key}-drift".encode()
    )
    _rewrite_artifact(case, role, value)

    with pytest.raises(preflight.PreflightValidationError, match="dependency differs"):
        _validate(case)


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_expansion_candidate_pool_policy_requires_exact_keys(
    case: _Case,
    mutation: str,
):
    role = "expansion_eligibility_amendment"
    value = _artifact_value(case, role)
    pool = value["candidate_pool_policy"]
    if mutation == "extra":
        pool["unexpected"] = False
    else:
        pool.pop("strategy")
    _rewrite_artifact(case, role, value)

    with pytest.raises(
        preflight.PreflightValidationError,
        match="candidate_pool_policy keys differ",
    ):
        _validate(case)


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_padding_constraints_require_exact_keys(case: _Case, mutation: str):
    role = "padding_semantics_amendment"
    value = _artifact_value(case, role)
    constraints = value["constraints"]
    if mutation == "extra":
        constraints["unexpected"] = False
    else:
        constraints.pop("task_specific_sampler")
    _rewrite_artifact(case, role, value)

    with pytest.raises(
        preflight.PreflightValidationError,
        match="constraints keys differ",
    ):
        _validate(case)


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_padding_dependency_requires_exact_keys(case: _Case, mutation: str):
    role = "padding_semantics_amendment"
    value = _artifact_value(case, role)
    dependency = value["eligibility_dependency"]
    if mutation == "extra":
        dependency["unexpected"] = _digest(b"unexpected")
    else:
        dependency.pop("support_contract_sha256")
    _rewrite_artifact(case, role, value)

    with pytest.raises(
        preflight.PreflightValidationError,
        match="eligibility_dependency keys differ",
    ):
        _validate(case)


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("raw_candidate_count", True),
        ("excluded_per_task", 150.0),
        ("runtime_skip_allowed", 0),
        ("action_sigma_schedule_changed", True),
    ],
)
def test_expansion_candidate_pool_rejects_numeric_aliases(
    case: _Case,
    field: str,
    alias: object,
):
    role = "expansion_eligibility_amendment"
    value = _artifact_value(case, role)
    value["candidate_pool_policy"][field] = alias
    _rewrite_artifact(case, role, value)

    with pytest.raises(preflight.PreflightValidationError):
        _validate(case)


@pytest.mark.parametrize("alias", [True, 0])
def test_padding_constraints_must_be_exact_false_booleans(
    case: _Case,
    alias: object,
):
    role = "padding_semantics_amendment"
    value = _artifact_value(case, role)
    value["constraints"]["best_of_n"] = alias
    _rewrite_artifact(case, role, value)

    with pytest.raises(
        preflight.PreflightValidationError,
        match="forbidden constraints differ",
    ):
        _validate(case)


@pytest.mark.parametrize(
    ("field", "alias"),
    [("optimizer_steps", True), ("protocol_seed", 20260724.0)],
)
def test_task_plan_integer_fields_reject_numeric_aliases(
    case: _Case,
    field: str,
    alias: object,
):
    role = "task_plan"
    value = _artifact_value(case, role)
    value["plan"][field] = alias
    _rewrite_artifact(case, role, value)

    with pytest.raises(preflight.PreflightValidationError, match="must be an integer"):
        _validate(case)


def test_task_plan_action_sigmas_reject_integer_alias(case: _Case):
    role = "task_plan"
    value = _artifact_value(case, role)
    value["plan"]["action_sigmas"][0] = 1
    _rewrite_artifact(case, role, value)

    with pytest.raises(
        preflight.PreflightValidationError, match="action_sigmas differ"
    ):
        _validate(case)


@pytest.mark.parametrize(
    ("field", "alias"),
    [("global_step", True), ("window_logical_length", 113.0)],
)
def test_dataset_row_integer_fields_reject_numeric_aliases(
    case: _Case,
    field: str,
    alias: object,
):
    role = "dataset_contract"
    value = _artifact_value(case, role)
    value["rows"][0][field] = alias
    _rewrite_artifact(case, role, value)

    with pytest.raises(preflight.PreflightValidationError, match="must be an integer"):
        _validate(case)


def test_input_config_and_runtime_support_drift_fail_closed(case: _Case):
    case.input_config.write_bytes(b"changed: true\n")
    with pytest.raises(preflight.PreflightValidationError, match="input-config SHA"):
        _validate(case)

    case = deepcopy(case)
    case.request["input_config"]["sha256"] = _file_digest(case.input_config)
    case.support[0].write_bytes(b"tampered\n")
    with pytest.raises(preflight.PreflightValidationError, match="runtime-support"):
        _validate(case)


def test_purpose_cannot_cross_authorize_smoke_or_training(case: _Case):
    smoke = deepcopy(case.request)
    smoke["purpose"] = "smoke"
    smoke["authorization"] = dict(preflight._SMOKE_AUTHORIZATION)
    smoke.pop("reference_gpu_started")
    smoke.update(
        {
            "reference_precompute_completed": True,
            "smoke_gpu_started": False,
        }
    )
    with pytest.raises(
        preflight.PreflightValidationError, match="artifacts keys differ"
    ):
        _validate(case, smoke)

    training = deepcopy(case.request)
    training["purpose"] = "training"
    training["authorization"] = {
        "operation": "phase6_training",
        "arm": "T1_E0A0",
        "factors": {"A": False, "E": False, "T": True},
        "arm_config_projection_sha256": _digest(b"projection"),
        "run_id": "run",
        "run_directory": str(case.output_root / "run"),
        "reference_gpu_forward_completed": True,
        "real_2b_smoke_completed": True,
        "optimizer_training_allowed": True,
        "closed_loop_allowed": False,
        "resume_allowed": False,
    }
    training.pop("reference_gpu_started")
    training.update(
        {
            "reference_precompute_completed": True,
            "smoke_completed": True,
        }
    )
    with pytest.raises(
        preflight.PreflightValidationError, match="artifacts keys differ"
    ):
        _validate(case, training)


@pytest.mark.parametrize(
    ("purpose", "field", "alias"),
    [
        ("reference_precompute", "continuous_time", 0),
        ("reference_precompute", "lambda_action", 0),
        ("smoke", "paired_global_step", True),
        ("smoke", "optimizer_creation_allowed", 0),
    ],
)
def test_reference_and_smoke_authorization_reject_numeric_type_aliases(
    case: _Case,
    purpose: str,
    field: str,
    alias: object,
):
    request = deepcopy(case.request)
    if purpose == "smoke":
        request["purpose"] = purpose
        request["authorization"] = dict(preflight._SMOKE_AUTHORIZATION)
        request.pop("reference_gpu_started")
        request.update(
            {
                "reference_precompute_completed": True,
                "smoke_gpu_started": False,
            }
        )
    request["authorization"][field] = alias

    with pytest.raises(preflight.PreflightValidationError, match="inexact JSON type"):
        _validate(case, request)


@pytest.mark.parametrize(
    ("key", "alias"),
    (
        ("historical_checkpoint_count", False),
        ("historical_optimizer_step_calls_per_arm", True),
        ("historical_scheduled_lr_scale_at_step0", 0),
    ),
)
def test_recovery_lifecycle_rejects_json_numeric_aliases(case: _Case, key, alias):
    request = deepcopy(case.request)
    request[key] = alias
    with pytest.raises(preflight.PreflightValidationError, match=key):
        _validate(case, request)


def test_source_manifest_covers_lockfiles_upstream_and_runtime_fingerprint(case: _Case):
    value = json.loads(case.source_manifest.read_bytes())
    paths = [item["path"] for item in value["files"]]
    assert paths == sorted(paths)
    assert ".python-version" in paths
    assert "pyproject.toml" in paths
    assert "uv.lock" in paths
    assert "third_party/Sana/diffusion/model.py" in paths
    assert value["runtime_environment"]["soabi"] == "cpython-test-soabi"
    assert value["repository_vcs"]["repository_head"] == "1" * 40


def _synthetic_rebind_builder_manifest(builder: Path) -> dict:
    payload = builder.read_bytes()
    return {
        "files": [
            {
                "path": "scripts/build_phase6_recovery_rebind.py",
                "realpath": str(builder.resolve()),
                "sha256": _digest(payload),
                "size_bytes": len(payload),
            }
        ]
    }


def test_rebind_builder_capture_requires_exact_source_manifest_entry(tmp_path: Path):
    repository = tmp_path / "repo"
    builder = _write(
        repository / "scripts" / "build_phase6_recovery_rebind.py", b"VALUE = 1\n"
    )
    manifest = _synthetic_rebind_builder_manifest(builder)
    manifest["files"][0]["unexpected"] = True
    with pytest.raises(preflight.PreflightValidationError, match="entry keys differ"):
        preflight._capture_rebind_builder_bytes(
            manifest,
            repository_root=str(repository.resolve()),
            expected_builder_path=str(builder),
        )


def test_rebind_builder_exec_input_never_reopens_after_stable_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository = tmp_path / "repo"
    original = b"VALUE = 1\n"
    changed = b"VALUE = 2\n"
    builder = _write(
        repository / "scripts" / "build_phase6_recovery_rebind.py", original
    )
    manifest = _synthetic_rebind_builder_manifest(builder)
    observe = preflight._observe_file

    def capture_then_drift(*args, **kwargs):
        result = observe(*args, **kwargs)
        builder.write_bytes(changed)
        return result

    monkeypatch.setattr(preflight, "_observe_file", capture_then_drift)
    captured = preflight._capture_rebind_builder_bytes(
        manifest,
        repository_root=str(repository.resolve()),
        expected_builder_path=str(builder),
    )
    assert captured == original
    assert builder.read_bytes() == changed


def test_runtime_fingerprint_rejects_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
):
    def missing(name: str):  # noqa: ARG001
        raise preflight.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(preflight.importlib_metadata, "version", missing)
    with pytest.raises(
        preflight.PreflightValidationError, match="distributions are missing"
    ):
        preflight._runtime_environment_fingerprint()


def test_report_and_source_writers_are_exclusive(case: _Case):
    report = _validate(case)
    report_path = case.root / "report.json"
    digest = preflight.write_canonical_report(report_path, report)
    assert digest == _file_digest(report_path)
    with pytest.raises(
        preflight.PreflightValidationError, match="refusing to overwrite"
    ):
        preflight.write_canonical_report(report_path, report)
    assert not list(case.root.glob(".report.json.*.tmp"))

    source_path = case.root / "source-copy.json"
    source_value = json.loads(case.source_manifest.read_bytes())
    preflight.write_phase6_source_manifest(source_path, source_value)
    with pytest.raises(
        preflight.PreflightValidationError, match="refusing to overwrite"
    ):
        preflight.write_phase6_source_manifest(source_path, source_value)


def test_run_preflight_is_exclusive_and_report_loader_is_sha_pinned(
    case: _Case, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(preflight, "_FROZEN_POLICY", case.policy)
    report_path = case.output_root / "run-report.json"
    report = preflight.run_phase6_preflight(
        case.request_path,
        report_path,
        expected_request_sha256=_file_digest(case.request_path),
        expected_purpose="reference_precompute",
    )
    loaded = preflight.load_canonical_preflight_report(
        report_path, expected_sha256=_file_digest(report_path)
    )
    assert loaded == report
    reproduced = preflight.load_and_revalidate_phase6_preflight_report(
        case.request_path,
        report_path,
        expected_request_sha256=_file_digest(case.request_path),
        expected_report_sha256=_file_digest(report_path),
        expected_purpose="reference_precompute",
    )
    assert reproduced == report
    with pytest.raises(preflight.PreflightValidationError, match="overwrite"):
        preflight.run_phase6_preflight(
            case.request_path,
            report_path,
            expected_request_sha256=_file_digest(case.request_path),
            expected_purpose="reference_precompute",
        )


def test_prior_authorization_pair_binds_request_bytes(case: _Case):
    report = _validate(case)
    request_bytes = preflight.canonical_json_bytes(case.request)
    preflight._validate_prior_authorization_pair(
        case.request,
        report,
        request_bytes,
        preflight.canonical_json_bytes(report),
        expected_purpose="reference_precompute",
        policy=case.policy,
        file_hasher=None,
    )
    changed = deepcopy(report)
    changed["request_sha256"] = "4" * 64
    with pytest.raises(preflight.PreflightValidationError, match="request SHA"):
        preflight._validate_prior_authorization_pair(
            case.request,
            changed,
            request_bytes,
            preflight.canonical_json_bytes(changed),
            expected_purpose="reference_precompute",
            policy=case.policy,
            file_hasher=None,
        )

    aliased_authorization = deepcopy(report)
    aliased_authorization["authorization"]["continuous_time"] = 0
    with pytest.raises(preflight.PreflightValidationError, match="inexact JSON type"):
        preflight._validate_prior_authorization_pair(
            case.request,
            aliased_authorization,
            request_bytes,
            preflight.canonical_json_bytes(aliased_authorization),
            expected_purpose="reference_precompute",
            policy=case.policy,
            file_hasher=None,
        )

    partial = deepcopy(report)
    partial["pins"].pop("observed_source_manifest_sha256")
    with pytest.raises(preflight.PreflightValidationError, match="does not reproduce"):
        preflight._validate_prior_authorization_pair(
            case.request,
            partial,
            request_bytes,
            preflight.canonical_json_bytes(partial),
            expected_purpose="reference_precompute",
            policy=case.policy,
            file_hasher=None,
        )


def test_smoke_and_training_purposes_close_prior_authorization_dag(
    case: _Case, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(preflight, "_validate_action_reference", lambda *a, **k: None)
    monkeypatch.setattr(
        preflight, "_validate_action_reference_spot_check", lambda *a, **k: None
    )
    full_replays = []
    monkeypatch.setattr(
        preflight,
        "_validate_full_rebind_receipts",
        lambda **kwargs: full_replays.append(kwargs["purpose"]),
    )

    arms = []
    factor_rows = {
        "T0_E0A0": {"A": False, "E": False, "T": False},
        "T1_E0A0": {"A": False, "E": False, "T": True},
        "T1_E1A0": {"A": False, "E": True, "T": True},
        "T1_E0A1": {"A": True, "E": False, "T": True},
        "T1_E1A1": {"A": True, "E": True, "T": True},
    }
    for arm, factors in factor_rows.items():
        projection = {"arm": arm, "factors": factors}
        arms.append(
            {
                "arm": arm,
                "factors": factors,
                "projection": projection,
                "projection_sha256": _digest(
                    preflight.canonical_json_bytes(projection)
                ),
                "legacy_projection_sha256": _digest(
                    f"legacy-projection:{arm}".encode("ascii")
                ),
                "run_directory": str(case.output_root / arm / "run"),
                "run_id": f"run-{arm}",
            }
        )
    arm_manifest_value = {"arms": arms}
    arm_module = ModuleType("sana_wam.train.phase6_arm_config")

    def validate_arm(data: bytes, **kwargs):  # noqa: ARG001
        return deepcopy(arm_manifest_value)

    arm_module.validate_phase6_arm_projection_manifest_bytes = validate_arm
    arm_module.validate_phase6_smoke_runtime_bundle_bytes = lambda data, **kwargs: {
        "status": "valid"
    }
    arm_module.validate_phase6_projection_equivalence_receipt_bytes = (
        lambda data, **kwargs: {"status": "valid"}
    )
    monkeypatch.setitem(sys.modules, arm_module.__name__, arm_module)
    _compact(
        case.artifacts["projection_equivalence_receipt"],
        {
            "arms": [
                {
                    "arm": item["arm"],
                    "legacy_projection_sha256": item["legacy_projection_sha256"],
                    "recovery_projection_sha256": item["projection_sha256"],
                }
                for item in arms
            ],
            "scientific_equivalence": True,
        },
    )
    case.repin_artifact("projection_equivalence_receipt")
    case.write_request()

    reference_report_value = _validate(case)
    reference_report_path = _compact(
        case.output_root / "reference-report.json", reference_report_value
    )
    reference_row = {
        "action_sigma": 1.0,
        "common_input_trace_sha256": _digest(b"common"),
        "dataset_contract_row_sha256": _digest(b"dataset-row"),
        "dataset_index": 101,
        "global_step": 1,
        "plan_row_sha256": _digest(b"plan-row"),
        "task_name": "task-1",
    }
    extra_values = {
        "reference_preflight_request": case.request,
        "reference_preflight_report": reference_report_value,
        "action_reference": {"rows": [reference_row]},
        "action_reference_spot_check": {"verified": True},
        "arm_projection_manifest": arm_manifest_value,
    }
    extra_paths = {
        role: _compact(case.root / "artifacts" / f"{role}.json", value)
        for role, value in extra_values.items()
    }
    extra_paths["reference_preflight_request"] = case.request_path
    extra_paths["reference_preflight_report"] = reference_report_path
    reference_receipt_value = {
        "artifact_role": "phase6_reference_v9_cpu_byte_identical_rebind",
        "authorization": {
            "recovery_authority": case.request["artifacts"]["recovery_authority"],
            "source_manifest_v9": case.request["source_manifest"],
        },
        **preflight.recovery_lifecycle(),
        "new": {
            "request_v10": {
                "path": str(case.request_path),
                "sha256": _file_digest(case.request_path),
            },
            "report_v10": {
                "path": str(reference_report_path),
                "sha256": _file_digest(reference_report_path),
            },
        },
        "registered_before_recovery_cohort_started": True,
        "schema_version": "sana-phase6-reference-v9-cpu-rebind-receipt-v1",
        "status": "pass",
    }
    extra_paths["reference_rebind_receipt"] = _compact(
        case.root / "artifacts" / "reference_rebind_receipt.json",
        reference_receipt_value,
    )

    smoke_request = deepcopy(case.request)
    smoke_request["purpose"] = "smoke"
    smoke_request["authorization"] = dict(preflight._SMOKE_AUTHORIZATION)
    smoke_request.pop("reference_gpu_started")
    smoke_request.update(
        {
            "reference_precompute_completed": True,
            "smoke_gpu_started": False,
        }
    )
    smoke_bundle_path = _compact(
        case.repository / "configs" / "smoke-runtime-bundle.json",
        {"schema_version": "synthetic-smoke-runtime-bundle"},
    )
    smoke_request["input_config"] = {
        "path": str(smoke_bundle_path),
        "sha256": _file_digest(smoke_bundle_path),
    }
    smoke_request["artifacts"].update(
        {
            role: {"path": str(path), "sha256": _file_digest(path)}
            for role, path in extra_paths.items()
        }
    )
    smoke_request_path = _compact(
        case.output_root / "smoke-request.json", smoke_request
    )
    smoke_report_value = _validate(case, smoke_request)
    assert smoke_report_value["purpose"] == "smoke"
    assert smoke_report_value["pins"]["observed_reference_preflight_report_sha256"] == (
        _file_digest(reference_report_path)
    )
    smoke_report_path = _compact(
        case.output_root / "smoke-report.json", smoke_report_value
    )

    selected = arms[1]
    projection_path = _compact(
        case.repository / "configs" / "T1_E0A0.projection.json",
        selected["projection"],
    )
    smoke_artifact_value = {
        "cases": [
            {
                "case_id": item["arm"],
                "config_projection_sha256": item["legacy_projection_sha256"],
            }
            for item in arms
        ],
        **preflight.recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "paired_fixed_row": reference_row,
        "status": "pass",
    }
    smoke_artifact_path = _compact(
        case.root / "artifacts" / "real-smoke.json", smoke_artifact_value
    )

    def artifact_pin(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": _file_digest(path)}

    reference_receipt_pin = artifact_pin(extra_paths["reference_rebind_receipt"])
    smoke_receipt_value = {
        "artifact_role": "phase6_smoke_v8_cpu_byte_identical_rebind",
        "authorization": {
            "recovery_authority": case.request["artifacts"]["recovery_authority"],
            "source_manifest_v9": case.request["source_manifest"],
        },
        **preflight.recovery_lifecycle(),
        "new": {
            "request_v10": artifact_pin(smoke_request_path),
            "report_v10": artifact_pin(smoke_report_path),
            "gate_v8": artifact_pin(smoke_artifact_path),
        },
        "reference_rebind_receipt": reference_receipt_pin,
        "registered_before_recovery_cohort_started": True,
        "schema_version": "sana-phase6-smoke-v8-cpu-rebind-receipt-v1",
        "status": "pass",
    }
    smoke_receipt_path = _compact(
        case.root / "artifacts" / "smoke_rebind_receipt.json",
        smoke_receipt_value,
    )
    smoke_receipt_pin = artifact_pin(smoke_receipt_path)
    evidence_receipt_value = {
        "artifact_role": "phase6_planrow_recovery_historical_gpu_evidence_rebinding",
        **preflight.recovery_lifecycle(),
        "recovery_authority": case.request["artifacts"]["recovery_authority"],
        "reference_rebind_receipt": reference_receipt_pin,
        "smoke_rebind_receipt": smoke_receipt_pin,
        "source_manifest_v9": case.request["source_manifest"],
        "schema_version": "sana-phase6-evidence-rebinding-receipt-v6",
        "status": "pass",
    }
    evidence_receipt_path = _compact(
        case.root / "artifacts" / "evidence_rebinding_receipt.json",
        evidence_receipt_value,
    )
    smoke_module = ModuleType("sana_wam.train.phase6_smoke_gate")

    def validate_smoke(data: bytes, **kwargs):  # noqa: ARG001
        return deepcopy(smoke_artifact_value)

    smoke_module.validate_phase6_smoke_artifact_bytes = validate_smoke
    monkeypatch.setitem(sys.modules, smoke_module.__name__, smoke_module)

    training_request = deepcopy(smoke_request)
    training_request["purpose"] = "training"
    training_request["authorization"] = {
        "operation": "phase6_training",
        "arm": selected["arm"],
        "factors": selected["factors"],
        "arm_config_projection_sha256": selected["projection_sha256"],
        "run_id": selected["run_id"],
        "run_directory": selected["run_directory"],
        "reference_gpu_forward_completed": True,
        "real_2b_smoke_completed": True,
        "optimizer_training_allowed": True,
        "closed_loop_allowed": False,
        "resume_allowed": False,
    }
    for key in ("smoke_gpu_started",):
        training_request.pop(key)
    training_request.update(
        {
            "smoke_completed": True,
        }
    )
    training_request["input_config"] = {
        "path": str(projection_path),
        "sha256": _file_digest(projection_path),
    }
    training_extras = {
        "smoke_preflight_request": smoke_request_path,
        "smoke_preflight_report": smoke_report_path,
        "smoke_rebind_receipt": smoke_receipt_path,
        "evidence_rebinding_receipt": evidence_receipt_path,
        "real_2b_smoke_gate": smoke_artifact_path,
    }
    training_request["artifacts"].update(
        {
            role: {"path": str(path), "sha256": _file_digest(path)}
            for role, path in training_extras.items()
        }
    )
    training_report = _validate(case, training_request)
    assert training_report["purpose"] == "training"
    assert "smoke" in full_replays
    assert "training" in full_replays
    assert training_report["input_config_role"] == ("arm_scientific_projection_sidecar")
    assert training_report["pins"]["observed_real_2b_smoke_gate_sha256"] == (
        _file_digest(smoke_artifact_path)
    )

    wrong_projection_smoke = deepcopy(smoke_artifact_value)
    wrong_projection_smoke["cases"][0]["config_projection_sha256"] = arms[0][
        "projection_sha256"
    ]
    with monkeypatch.context() as fault:
        fault.setattr(
            smoke_module,
            "validate_phase6_smoke_artifact_bytes",
            lambda data, **kwargs: deepcopy(wrong_projection_smoke),
        )
        with pytest.raises(
            preflight.PreflightValidationError,
            match="smoke legacy projection differs from equivalence receipt",
        ):
            _validate(case, training_request)

    aliased_training_request = deepcopy(training_request)
    aliased_training_request["authorization"]["factors"]["A"] = 0
    with pytest.raises(
        preflight.PreflightValidationError, match="must be booleans|inexact JSON type"
    ):
        _validate(case, aliased_training_request)

    bad_evidence = deepcopy(evidence_receipt_value)
    bad_evidence["smoke_rebind_receipt"] = {
        **bad_evidence["smoke_rebind_receipt"],
        "sha256": "e" * 64,
    }
    bad_evidence_path = _compact(
        case.root / "artifacts" / "bad-evidence-rebinding-receipt.json",
        bad_evidence,
    )
    bad_training_request = deepcopy(training_request)
    bad_training_request["artifacts"]["evidence_rebinding_receipt"] = artifact_pin(
        bad_evidence_path
    )
    with pytest.raises(
        preflight.PreflightValidationError,
        match="evidence receipt smoke_rebind_receipt backlink",
    ):
        _validate(case, bad_training_request)


def test_import_remains_torch_free():
    source_root = Path(__file__).parents[1] / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        "import sana_wam.train.phase6_preflight; "
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
