from __future__ import annotations

from copy import deepcopy
import errno
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import shutil
import stat
from types import SimpleNamespace
import subprocess
import sys
from typing import Any

import pytest

from sana_wam.train import phase6_recovery as recovery


TIMES = {
    "baseline": "2026-01-01T00:00:00Z",
    "proposal": "2026-01-01T00:01:00Z",
    "subject": "2026-01-01T00:02:00Z",
    "A": "2026-01-01T00:03:00Z",
    "B": "2026-01-01T00:04:00Z",
    "authority": "2026-01-01T00:05:00Z",
}


def _write(path: Path, data: bytes, *, registered: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if registered:
        os.chmod(path, 0o400)
    return {
        "mode": "0400",
        "path": os.fspath(path),
        "sha256": sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _rewrite_registered(path: Path, data: bytes) -> dict[str, Any]:
    os.chmod(path, 0o600)
    return _write(path, data, registered=True)


def _full_pin(pin: dict[str, Any]) -> dict[str, Any]:
    observed = os.lstat(pin["path"])
    return {
        "device": observed.st_dev,
        "gid": getattr(observed, "st_gid", 0),
        "inode": observed.st_ino,
        "mode": "0400",
        "nlink": observed.st_nlink,
        "path": pin["path"],
        "sha256": pin["sha256"],
        "size_bytes": pin["size_bytes"],
        "uid": getattr(observed, "st_uid", 0),
    }


def _canonical(value: Any) -> bytes:
    return recovery.canonical_json_bytes(value)


def test_prior_recovery_source_manifest_v8_anchors_hotfix_baseline():
    data = b"VALUE = 1\n"
    manifest = {
        "files": [
            {
                "path": "src/example.py",
                "realpath": f"{recovery.REPOSITORY_ROOT}/src/example.py",
                "sha256": sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        ],
        **recovery.recovery_lifecycle(),
        "inventory_globs": list(recovery.SOURCE_INVENTORY_GLOBS),
        "recovery_authority": deepcopy(recovery.PRIOR_RECOVERY_AUTHORITY),
        "registered_before_recovery_cohort_started": True,
        "repository_root": recovery.REPOSITORY_ROOT,
        "repository_vcs": {},
        "runtime_environment": {},
        "schema_version": "sana-phase6-source-runtime-manifest-v3",
    }

    assert recovery._validate_prior_source_manifest(_canonical(manifest)) == {
        "src/example.py": {
            "sha256": sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    }
    manifest["recovery_authority"]["sha256"] = "0" * 64
    with pytest.raises(
        recovery.Phase6RecoveryAuthorityError,
        match="prior source manifest identity differs",
    ):
        recovery._validate_prior_source_manifest(_canonical(manifest))


def test_prior_source_execution_receipt_authenticates_mixed_modes(governance):
    modes = recovery.prior_recovery_source_modes()
    assert modes == {
        "scripts/train.py": 0o644,
        **{
            f"configs/phase6/templates/train_phase6_{arm}.yaml.in": 0o664
            for arm in recovery.ARM_NAMES
        },
    }


def test_prior_source_execution_receipt_rejects_mode_tamper(
    governance, monkeypatch: pytest.MonkeyPatch
):
    path = Path(recovery.PRIOR_SOURCE_EXECUTION_RECEIPT_PIN["path"])
    value = recovery._strict_json(path.read_bytes(), "prior source receipt fixture")
    value["files"][0]["rendered_mode"] = "0664"
    value["files"][0]["tokenized_mode"] = "0664"
    tampered_pin = _rewrite_registered(path, _canonical(value))
    monkeypatch.setattr(recovery, "PRIOR_SOURCE_EXECUTION_RECEIPT_PIN", tampered_pin)
    with pytest.raises(
        recovery.Phase6RecoveryAuthorityError, match="added mode differs"
    ):
        recovery.prior_recovery_source_modes()


def _minimal_repo(root: Path) -> None:
    files = {
        ".python-version": b"3.11\n",
        "pyproject.toml": b"[project]\nname='fixture'\n",
        "uv.lock": b"fixture-lock\n",
        "scripts/train.py": b"print('before')\n",
        "src/sana_wam/train/old.py": b"VALUE = 'before'\n",
        "tests/test_existing.py": b"def test_before(): assert True\n",
    }
    for arm in recovery.ARM_NAMES:
        files[f"configs/phase6/templates/train_phase6_{arm}.yaml.in"] = (
            f"training:\n  arm: {arm}\n".encode("ascii")
        )
    for relative, data in files.items():
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.chmod(path, 0o664)


def _patch_layout(monkeypatch: pytest.MonkeyPatch, base: Path, target: Path) -> Path:
    governance = base / "governance"
    operational = base / "operational"
    monkeypatch.setattr(recovery, "REPOSITORY_ROOT", os.fspath(target))
    monkeypatch.setattr(recovery, "OPERATIONAL_ROOT", os.fspath(operational))
    monkeypatch.setattr(recovery, "RECOVERY_ROOT", os.fspath(governance))
    monkeypatch.setattr(
        recovery,
        "RECOVERY_BASELINE_SNAPSHOT_PATH",
        os.fspath(governance / "baseline.json"),
    )
    monkeypatch.setattr(
        recovery, "RECOVERY_PROPOSAL_PATH", os.fspath(governance / "proposal.json")
    )
    monkeypatch.setattr(
        recovery, "RECOVERY_REVIEW_SUBJECT_PATH", os.fspath(governance / "subject.json")
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_REVIEW_RECEIPT_PATHS",
        {
            "A": os.fspath(governance / "review_A.json"),
            "B": os.fspath(governance / "review_B.json"),
        },
    )
    monkeypatch.setattr(
        recovery,
        "PLANROW_RECOVERY_AUTHORITY_PATH",
        os.fspath(governance / "authority.json"),
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_RENDER_PLAN_PATH",
        os.fspath(governance / "source_render_plan.json"),
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH",
        os.fspath(governance / "source_execution_receipt.json"),
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_RUN_DIRECTORIES",
        {
            arm: os.fspath(operational / "arms" / arm / recovery.RECOVERY_RUN_IDS[arm])
            for arm in recovery.ARM_NAMES
        },
    )
    monkeypatch.setattr(
        recovery,
        "RECOVERY_VERSIONED_OUTPUTS",
        {
            "evidence_rebinding_receipt": os.fspath(
                governance / "evidence_rebind.json"
            ),
            "reference_rebind_receipt": os.fspath(governance / "reference_rebind.json"),
            "smoke_rebind_receipt": os.fspath(governance / "smoke_rebind.json"),
            "source_execution_receipt": os.fspath(
                governance / "source_execution_receipt.json"
            ),
            "source_manifest": os.fspath(operational / "source_v5.json"),
            "source_render_plan": os.fspath(governance / "source_render_plan.json"),
            "ticket_root": os.fspath(operational / "tickets_v4"),
        },
    )
    return governance


def _prior_artifacts(
    monkeypatch: pytest.MonkeyPatch, governance: Path, target: Path
) -> None:
    prior_files = []
    prior_paths = sorted(
        [
            ".python-version",
            "pyproject.toml",
            "scripts/train.py",
            "src/sana_wam/train/old.py",
            "uv.lock",
            *(
                f"configs/phase6/templates/train_phase6_{arm}.yaml.in"
                for arm in recovery.ARM_NAMES
            ),
        ]
    )
    for relative in prior_paths:
        data = target.joinpath(*relative.split("/")).read_bytes()
        prior_files.append(
            {
                "path": relative,
                "realpath": os.fspath(target.joinpath(*relative.split("/")).resolve()),
                "sha256": sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    manifest = {
        "files": prior_files,
        "inventory_globs": [
            ".python-version",
            "pyproject.toml",
            "scripts/**/*.py",
            "src/**/*.py",
            "third_party/Sana/**/*.py",
            "uv.lock",
        ],
        "registered_before_phase6_training_results": True,
        "repository_root": os.fspath(target),
        "repository_vcs": {},
        "runtime_environment": {},
        "schema_version": "sana-phase6-source-runtime-manifest-v2",
        "training_or_gpu_started": False,
    }
    manifest_pin = _write(
        governance / "source_v4.json", _canonical(manifest), registered=True
    )
    repair_pin = _write(
        governance / "repair.json",
        _canonical({"status": "registered"}),
        registered=True,
    )
    launch_failure_pin = _write(
        governance / "prior_launch_failure.json",
        _canonical({"status": "registered_fail_closed"}),
        registered=True,
    )
    prior_by_path = {entry["path"]: entry for entry in prior_files}
    template_mode_path = governance / "prior_template_modes.json"
    template_mode_value = {
        "artifact_path": os.fspath(template_mode_path),
        "artifact_role": "phase6_launch_retry_template_mode_provenance",
        "files": [
            {
                "before": {
                    "sha256": prior_by_path[relative]["sha256"],
                    "size_bytes": prior_by_path[relative]["size_bytes"],
                },
                "mode": "0664",
                "path": relative,
                "realpath": os.fspath(target.joinpath(*relative.split("/")).resolve()),
            }
            for relative in sorted(
                path
                for path in prior_paths
                if path.startswith("configs/phase6/templates/")
            )
        ],
        "launch_preflight_failure": launch_failure_pin,
        "registered_at_utc": TIMES["baseline"],
        "registered_before_planrowfix_r2_source_change": True,
        "schema_version": "sana-phase6-launch-retry-template-mode-receipt-v1",
        "source_manifest_v8": manifest_pin,
        "status": "registered",
    }
    template_mode_pin = _write(
        template_mode_path, _canonical(template_mode_value), registered=True
    )
    prior_authority_pin = _write(
        governance / "prior_authority.json", b"prior-authority\n", registered=True
    )
    prior_proposal_pin = _write(
        governance / "prior_proposal.json", b"prior-proposal\n", registered=True
    )
    prior_render_plan_pin = _write(
        governance / "prior_render_plan.json", b"prior-render-plan\n", registered=True
    )
    prior_candidate_root = "/fixture/prior-recovery-candidate"
    prior_changed = next(
        {
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
        }
        for entry in prior_files
        if entry["path"] == "scripts/train.py"
    )
    prior_delta = {
        "before": None,
        "candidate": deepcopy(prior_changed),
        "change": "added",
        "path": "scripts/train.py",
    }
    prior_receipt_path = governance / "prior_source_execution_receipt.json"
    prior_receipt = {
        "artifact_path": os.fspath(prior_receipt_path),
        "artifact_role": "phase6_recovery_source_execution_receipt",
        "authority": prior_authority_pin,
        "authorized_source_delta": [prior_delta],
        "candidate_source_root": prior_candidate_root,
        "files": [
            {
                "before": None,
                "before_mode": None,
                "change": "added",
                "path": "scripts/train.py",
                "rendered": deepcopy(prior_changed),
                "rendered_mode": "0644",
                "state": "rendered_final",
                "tokenized": deepcopy(prior_changed),
                "tokenized_mode": "0644",
            }
        ],
        "gpu_executed": False,
        "lifecycle": recovery.recovery_lifecycle(),
        "proposal": prior_proposal_pin,
        "receipt_is_commit_marker": True,
        "registered_at_utc": TIMES["baseline"],
        "rendered_source_inventory": {
            "files": [
                {
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "size_bytes": entry["size_bytes"],
                }
                for entry in prior_files
            ],
            "inventory_globs": list(recovery.SOURCE_INVENTORY_GLOBS),
            "representation": "raw-rendered",
            "schema_version": "sana-phase6-raw-rendered-source-inventory-v1",
        },
        "schema_version": "sana-phase6-recovery-source-execution-receipt-v1",
        "source_render_plan": prior_render_plan_pin,
        "source_write_completed": True,
        "status": "registered",
        "target_repository_root": os.fspath(target),
    }
    prior_receipt_pin = _write(
        prior_receipt_path, _canonical(prior_receipt), registered=True
    )
    monkeypatch.setattr(recovery, "PRIOR_RECOVERY_AUTHORITY_PIN", prior_authority_pin)
    monkeypatch.setattr(recovery, "PRIOR_RECOVERY_PROPOSAL_PIN", prior_proposal_pin)
    monkeypatch.setattr(
        recovery, "PRIOR_RECOVERY_RENDER_PLAN_PIN", prior_render_plan_pin
    )
    monkeypatch.setattr(recovery, "PRIOR_RECOVERY_CANDIDATE_ROOT", prior_candidate_root)
    monkeypatch.setattr(
        recovery, "PRIOR_SOURCE_EXECUTION_RECEIPT_PIN", prior_receipt_pin
    )
    monkeypatch.setattr(
        recovery, "PRIOR_LAUNCH_FAILURE_RECEIPT_PIN", launch_failure_pin
    )
    monkeypatch.setattr(recovery, "PRIOR_TEMPLATE_MODE_RECEIPT_PIN", template_mode_pin)
    monkeypatch.setattr(recovery, "PRIOR_SOURCE_MANIFEST_PATH", manifest_pin["path"])
    monkeypatch.setattr(
        recovery, "PRIOR_SOURCE_MANIFEST_SHA256", manifest_pin["sha256"]
    )
    monkeypatch.setattr(
        recovery, "PRIOR_SOURCE_MANIFEST_SIZE", manifest_pin["size_bytes"]
    )
    monkeypatch.setattr(
        recovery, "PRIOR_SOURCE_REPAIR_AMENDMENT_PATH", repair_pin["path"]
    )
    monkeypatch.setattr(
        recovery, "PRIOR_SOURCE_REPAIR_AMENDMENT_SHA256", repair_pin["sha256"]
    )
    monkeypatch.setattr(
        recovery, "PRIOR_SOURCE_REPAIR_AMENDMENT_SIZE", repair_pin["size_bytes"]
    )


def _support_artifacts(monkeypatch: pytest.MonkeyPatch, governance: Path) -> None:
    aggregate = {
        "affected_arm_count": 5,
        "checkpoint_count_total": 0,
        "completion_manifest_count_total": 0,
        "durable_metrics_rows_total": 0,
        "durable_optimizer_state_available": False,
        "formal_training_started": True,
        "old_cohort_completed": False,
        "old_cohort_resumable": False,
        "old_cohort_retryable_in_place": False,
        "optimizer_step_calls_per_arm": 1,
        "optimizer_step_calls_total": 5,
        "scientific_results_available": False,
        "summary_count_total": 0,
    }
    arms = [
        {
            "arm": arm,
            "nonzero_weight_update_proven": False,
            "optimizer_state_not_durable": True,
            "optimizer_step_calls": 1,
            "resume_allowed": False,
            "run_id": f"phase6_20260724_{arm}_primary",
            "scientific_result_available": False,
        }
        for arm in recovery.ARM_NAMES
    ]
    inventory = {
        "absence_contract": {},
        "aggregate_facts": aggregate,
        "arms": arms,
        "evidence_preservation": {},
        "failed_launch": {},
        "failure_signature": {},
        "host": "fixture",
        "observed_at_utc": TIMES["baseline"],
        "old_source_pins": [],
        "operational_root": recovery.OPERATIONAL_ROOT,
        "quiescence_observation": {},
        "recovery_candidate_pins": {},
        "schema_version": "phase6-failed-run-readonly-inventory-v1",
        "task_plan": {},
    }
    inventory_pin = _write(
        governance / "inventory.json", _canonical(inventory), registered=True
    )

    failed_subject = {
        "artifact_role": "phase6_failed_run_inventory_review_subject",
        "builder": {},
        "deprecated_candidates": [],
        "evidence_root": "fixture",
        "historical_claims": {},
        "inventory": inventory_pin,
        "review_requirements": {},
        "schema_version": "sana-phase6-failed-run-inventory-review-subject-v1",
    }
    failed_subject_pin = _write(
        governance / "failed_subject.json", _canonical(failed_subject), registered=True
    )
    failed_review_pins = {}
    for slot in ("A", "B"):
        receipt = {
            "artifact_role": "phase6_failed_run_inventory_independent_review_receipt",
            "attestations": {},
            "decision": "approve",
            "dynamic_replay": {},
            "registered_at_utc": TIMES[slot],
            "review_slot": slot,
            "review_subject": {
                key: failed_subject_pin[key] for key in ("path", "sha256", "size_bytes")
            },
            "reviewed_candidate_subject": {},
            "reviewer": {},
            "schema_version": "sana-phase6-failed-run-inventory-review-receipt-v1",
            "status": "registered",
        }
        failed_review_pins[slot] = _write(
            governance / f"failed_review_{slot}.json",
            _canonical(receipt),
            registered=True,
        )

    rowfix_subject = {
        "artifact_role": "phase6_plan_row_envelope_rowfix_review_subject",
        "before": {},
        "candidate": {},
        "deprecated_test_candidates": [],
        "production_dependencies": {},
        "required_contract": {},
        "required_tests": {},
        "schema_version": "sana-phase6-plan-row-envelope-rowfix-review-subject-v1",
    }
    rowfix_subject_pin = _write(
        governance / "rowfix_subject.json", _canonical(rowfix_subject), registered=True
    )
    rowfix_review_pins = {}
    for role in ("code", "test"):
        receipt = {
            "artifact_role": "phase6_plan_row_envelope_rowfix_independent_review_receipt",
            "attestations": {},
            "candidate": {},
            "decision": "approve",
            "dynamic_evidence": {},
            "registered_at_utc": TIMES["A" if role == "code" else "B"],
            "review_role": role,
            "review_subject": {
                key: rowfix_subject_pin[key] for key in ("path", "sha256", "size_bytes")
            },
            "reviewed_candidate_subject": {},
            "reviewer": {},
            "schema_version": "sana-phase6-plan-row-envelope-rowfix-review-receipt-v1",
            "status": "registered",
        }
        rowfix_review_pins[role] = _write(
            governance / f"rowfix_{role}.json", _canonical(receipt), registered=True
        )

    failed_bootstrap = {
        "artifact_role": "phase6_failed_inventory_bootstrap_publication_receipt",
        "builder_input": {},
        "candidate_inputs": {},
        "lifecycle": {
            "closed_loop_started": False,
            "durable_optimizer_state_available": False,
            "historical_formal_training_started": True,
            "recovery_cohort_started": False,
        },
        "outputs": {
            "bootstrap_builder": {},
            "inventory": _full_pin(inventory_pin),
            "inventory_builder": {},
            "review_subject": _full_pin(failed_subject_pin),
            "reviewer_A": _full_pin(failed_review_pins["A"]),
            "reviewer_B": _full_pin(failed_review_pins["B"]),
        },
        "registered_at_utc": TIMES["B"],
        "schema_version": "sana-phase6-failed-inventory-bootstrap-receipt-v1",
        "status": "registered",
    }
    failed_bootstrap_pin = _write(
        governance / "failed_bootstrap.json",
        _canonical(failed_bootstrap),
        registered=True,
    )
    rowfix_bootstrap = {
        "artifact_role": "phase6_planrow_rowfix_review_bootstrap_receipt",
        "builder_input": {},
        "candidate_inputs": {},
        "failed_run_inventory": _full_pin(inventory_pin),
        "outputs": {
            "builder": {},
            "code": _full_pin(rowfix_review_pins["code"]),
            "subject": _full_pin(rowfix_subject_pin),
            "test": _full_pin(rowfix_review_pins["test"]),
        },
        "registered_at_utc": TIMES["B"],
        "rowfix": {
            "source_sha256": recovery.PUBLISHED_ROWFIX_SOURCE_SHA256,
            "test_sha256": recovery.PUBLISHED_ROWFIX_TEST_SHA256,
        },
        "schema_version": "sana-phase6-planrow-rowfix-review-bootstrap-receipt-v1",
        "status": "registered",
    }
    rowfix_bootstrap_pin = _write(
        governance / "rowfix_bootstrap.json",
        _canonical(rowfix_bootstrap),
        registered=True,
    )

    values = {
        "FAILED_RUN_INVENTORY": inventory_pin,
        "FAILED_INVENTORY_SUPPORT_BUNDLE": failed_bootstrap_pin,
        "FAILED_INVENTORY_REVIEW_SUBJECT": failed_subject_pin,
        "ROWFIX_SUPPORT_BUNDLE": rowfix_bootstrap_pin,
        "ROWFIX_REVIEW_SUBJECT": rowfix_subject_pin,
    }
    for prefix, pin in values.items():
        monkeypatch.setattr(recovery, f"{prefix}_PATH", pin["path"])
        monkeypatch.setattr(recovery, f"{prefix}_SHA256", pin["sha256"])
        monkeypatch.setattr(recovery, f"{prefix}_SIZE", pin["size_bytes"])
    monkeypatch.setattr(
        recovery,
        "FAILED_INVENTORY_REVIEW_RECEIPT_PINS",
        {
            slot: {key: pin[key] for key in ("path", "sha256", "size_bytes")}
            for slot, pin in failed_review_pins.items()
        },
    )
    monkeypatch.setattr(
        recovery,
        "ROWFIX_REVIEW_RECEIPT_PINS",
        {
            role: {key: pin[key] for key in ("path", "sha256", "size_bytes")}
            for role, pin in rowfix_review_pins.items()
        },
    )


def _copy_candidate(source_module: Path, baseline: Path, candidate: Path) -> None:
    shutil.copytree(baseline, candidate)
    candidate.joinpath("src/sana_wam/train/old.py").write_bytes(
        b"VALUE = 'candidate'\n"
    )
    candidate.joinpath("tests/test_existing.py").write_bytes(
        b"def test_after(): assert True\n"
    )
    candidate.joinpath(
        "configs/phase6/templates/train_phase6_T0_E0A0.yaml.in"
    ).write_bytes(b"training:\n  arm: T0_E0A0\n  recovery: true\n")
    recovery_target = candidate / recovery.RECOVERY_SOURCE_PATH
    recovery_target.parent.mkdir(parents=True, exist_ok=True)
    recovery_source = source_module.read_bytes()
    authority_token = recovery._PLANROW_RECOVERY_AUTHORITY_SHA256_TOKEN.encode("ascii")
    token_count = recovery_source.count(authority_token)
    if token_count == 1:
        recovery_source = recovery._normalize_recovery_source(recovery_source, None)
    elif token_count == 0:
        recovery_source = recovery._normalize_recovery_source(
            recovery_source,
            recovery.PLANROW_RECOVERY_AUTHORITY_SHA256,
        )
    else:
        raise recovery.Phase6RecoveryAuthorityError(
            "candidate recovery source authority token count differs"
        )
    recovery_target.write_bytes(recovery_source)
    for relative in (
        "scripts/build_phase6_recovery_rebind.py",
        "tests/test_phase6_recovery_rebind.py",
    ):
        path = candidate.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# recovery rebind fixture\n")
    (candidate / "src/sana_wam/empty.py").write_bytes(b"")


def _deploy_rendered(candidate: Path, target: Path, authority_sha: str) -> None:
    for path in candidate.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(candidate)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        if relative.as_posix() == recovery.RECOVERY_SOURCE_PATH:
            data = recovery.render_phase6_recovery_source(data, authority_sha)
        destination.write_bytes(data)


def _restore_tree(snapshot: Path, target: Path) -> None:
    shutil.rmtree(target)
    shutil.copytree(snapshot, target)


@pytest.fixture
def governance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "target"
    candidate = tmp_path / "candidate"
    baseline_backup = tmp_path / "baseline_backup"
    _minimal_repo(target)
    governance_root = _patch_layout(monkeypatch, tmp_path, target)
    _prior_artifacts(monkeypatch, governance_root, target)
    _support_artifacts(monkeypatch, governance_root)

    baseline = recovery.build_phase6_recovery_baseline_snapshot(
        repository_root=os.fspath(target), registered_at_utc=TIMES["baseline"]
    )
    baseline_pin = _write(
        Path(recovery.RECOVERY_BASELINE_SNAPSHOT_PATH),
        _canonical(baseline),
        registered=True,
    )
    _copy_candidate(Path(recovery.__file__), target, candidate)
    proposal = recovery.build_phase6_recovery_proposal(
        baseline_source_snapshot_pin=baseline_pin,
        candidate_source_root=os.fspath(candidate),
        target_repository_root=os.fspath(target),
        registered_at_utc=TIMES["proposal"],
    )
    proposal_pin = _write(
        Path(recovery.RECOVERY_PROPOSAL_PATH), _canonical(proposal), registered=True
    )
    subject = recovery.build_phase6_recovery_review_subject(
        proposal_pin=proposal_pin,
        candidate_source_root=os.fspath(candidate),
        registered_at_utc=TIMES["subject"],
    )
    subject_pin = _write(
        Path(recovery.RECOVERY_REVIEW_SUBJECT_PATH),
        _canonical(subject),
        registered=True,
    )
    reviews = {}
    review_pins = {}
    for slot in ("A", "B"):
        reviews[slot] = recovery.build_phase6_recovery_review_receipt(
            slot=slot,
            review_subject_pin=subject_pin,
            candidate_source_root=os.fspath(candidate),
            reviewer_identity_claim={
                "identity": f"/root/reviewer_{slot.lower()}",
                "identity_namespace": "codex-collaboration-task",
            },
            session_or_task_id=f"phase6-review-{slot}",
            registered_at_utc=TIMES[slot],
        )
        review_pins[slot] = _write(
            Path(recovery.RECOVERY_REVIEW_RECEIPT_PATHS[slot]),
            _canonical(reviews[slot]),
            registered=True,
        )
    authority = recovery.build_phase6_planrow_recovery_authority(
        review_subject_pin=subject_pin,
        review_pins=review_pins,
        candidate_source_root=os.fspath(candidate),
        registered_at_utc=TIMES["authority"],
    )
    authority_data = _canonical(authority)
    authority_sha = sha256(authority_data).hexdigest()
    authority_pin = _write(
        Path(recovery.PLANROW_RECOVERY_AUTHORITY_PATH), authority_data, registered=True
    )
    assert authority_pin["sha256"] == authority_sha
    recovery.validate_phase6_planrow_recovery_authority_prerender_bytes(
        authority_data,
        expected_authority_sha256=authority_sha,
        candidate_source_root=os.fspath(candidate),
    )
    render_plan = recovery.build_phase6_recovery_source_render_plan(
        authority_pin=authority_pin,
        candidate_source_root=os.fspath(candidate),
        target_repository_root=os.fspath(target),
        registered_at_utc="2026-01-01T00:06:00Z",
    )
    render_plan_data = _canonical(render_plan)
    render_plan_pin = _write(
        Path(recovery.RECOVERY_RENDER_PLAN_PATH), render_plan_data, registered=True
    )
    recovery.validate_phase6_recovery_source_render_plan_bytes(
        render_plan_data,
        candidate_source_root=os.fspath(candidate),
        expected_authority_sha256=authority_sha,
    )
    shutil.copytree(target, baseline_backup)
    _deploy_rendered(candidate, target, authority_sha)
    monkeypatch.setattr(recovery, "PLANROW_RECOVERY_AUTHORITY_SHA256", authority_sha)

    state = {
        "authority": authority,
        "authority_data": authority_data,
        "authority_sha": authority_sha,
        "baseline": baseline,
        "baseline_backup": baseline_backup,
        "baseline_pin": baseline_pin,
        "candidate": candidate,
        "proposal": proposal,
        "proposal_pin": proposal_pin,
        "render_plan": render_plan,
        "render_plan_data": render_plan_data,
        "render_plan_pin": render_plan_pin,
        "review_pins": review_pins,
        "reviews": reviews,
        "subject": subject,
        "subject_pin": subject_pin,
        "target": target,
    }
    try:
        yield state
    finally:
        for path in tmp_path.rglob("*"):
            if path.is_file():
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass


def test_end_to_end_acyclic_authority_and_complete_delta(governance):
    state = governance
    validated = recovery.validate_phase6_planrow_recovery_authority_bytes(
        state["authority_data"],
        expected_authority_sha256=state["authority_sha"],
        target_repository_root=os.fspath(state["target"]),
    )
    assert validated == state["authority"]
    assert (
        recovery.validate_phase6_planrow_recovery_authority(
            repository_root=os.fspath(state["target"])
        )
        == state["authority"]
    )
    delta = {
        entry["path"]: entry for entry in state["proposal"]["authorized_source_delta"]
    }
    assert delta["tests/test_existing.py"]["before"] is not None
    assert (
        delta["configs/phase6/templates/train_phase6_T0_E0A0.yaml.in"]["before"]
        is not None
    )
    assert delta[recovery.RECOVERY_SOURCE_PATH]["before"] is None
    assert "scripts/build_phase6_recovery_rebind.py" in delta
    assert "tests/test_phase6_recovery_rebind.py" in delta
    inventory = state["proposal"]["source_inventory"]["files"]
    empty = next(item for item in inventory if item["path"] == "src/sana_wam/empty.py")
    assert empty["size_bytes"] == 0
    assert empty["sha256"] == sha256(b"").hexdigest()
    assert len(state["proposal"]["supporting_bundles"]) == 9
    assert (
        state["proposal"]["versioned_outputs"]["source_execution_receipt"]
        == recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="formal source apply is Linux-only"
)
def test_source_apply_cli_integrates_with_real_authority_graph(
    governance, monkeypatch: pytest.MonkeyPatch
):
    state = governance
    _restore_tree(state["baseline_backup"], state["target"])
    script = (
        Path(__file__).resolve().parents[1] / "scripts/apply_phase6_recovery_source.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase6_recovery_source_apply_integration", script
    )
    assert spec is not None and spec.loader is not None
    apply = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(apply)
    monkeypatch.setattr(
        apply,
        "sys",
        SimpleNamespace(
            dont_write_bytecode=sys.dont_write_bytecode,
            modules={},
            platform=sys.platform,
            stderr=sys.stderr,
        ),
    )
    for key, value in apply._REQUIRED_CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    integration_before_modes = {
        entry["path"]: stat.S_IMODE(
            state["target"].joinpath(*entry["path"].split("/")).stat().st_mode
        )
        for entry in state["authority"]["authorized_source_delta"]
        if entry["change"] == "modified"
    }
    monkeypatch.setattr(
        recovery,
        "prior_recovery_source_modes",
        lambda: deepcopy(integration_before_modes),
    )
    args = SimpleNamespace(
        authority_sha256=state["authority_sha"],
        authority_size_bytes=len(state["authority_data"]),
        candidate_source_root=os.fspath(state["candidate"]),
        output=recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH,
        registered_at_utc="2026-01-01T00:07:00Z",
        render_plan_sha256=state["render_plan_pin"]["sha256"],
        render_plan_size_bytes=state["render_plan_pin"]["size_bytes"],
        target_repository_root=os.fspath(state["target"]),
    )
    digest, size = apply._execute(args)
    receipt = Path(recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH)
    data = receipt.read_bytes()
    assert (digest, size) == (sha256(data).hexdigest(), len(data))
    recovery.validate_phase6_planrow_recovery_authority_bytes(
        state["authority_data"],
        expected_authority_sha256=state["authority_sha"],
        target_repository_root=os.fspath(state["target"]),
    )


def test_proposal_exact_keys_and_delta_omission_fail(governance):
    state = governance
    proposal = deepcopy(state["proposal"])
    proposal["unexpected"] = True
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="keys differ"):
        recovery.validate_phase6_recovery_proposal_bytes(
            _canonical(proposal), source_root=os.fspath(state["candidate"])
        )
    proposal = deepcopy(state["proposal"])
    proposal["authorized_source_delta"] = proposal["authorized_source_delta"][1:]
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="incomplete"):
        recovery.validate_phase6_recovery_proposal_bytes(
            _canonical(proposal),
            source_root=os.fspath(state["target"]),
            authority_sha256=state["authority_sha"],
        )


def test_before_pin_for_test_and_template_is_enforced(governance):
    state = governance
    proposal = deepcopy(state["proposal"])
    entry = next(
        item
        for item in proposal["authorized_source_delta"]
        if item["path"] == "tests/test_existing.py"
    )
    entry["before"] = None
    entry["change"] = "added"
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="incomplete"):
        recovery.validate_phase6_recovery_proposal_bytes(
            _canonical(proposal),
            source_root=os.fspath(state["target"]),
            authority_sha256=state["authority_sha"],
        )


def test_baseline_snapshot_is_anchored_to_source_manifest_v8(governance):
    baseline = deepcopy(governance["baseline"])
    entry = next(
        item
        for item in baseline["files"]
        if item["path"] == "src/sana_wam/train/old.py"
    )
    entry["sha256"] = "0" * 64
    with pytest.raises(
        recovery.Phase6RecoveryAuthorityError, match="source-manifest-v8"
    ):
        recovery.validate_phase6_recovery_baseline_snapshot_bytes(_canonical(baseline))


def test_prerender_proposal_rejects_unanchored_fake_baseline(governance):
    state = governance
    _restore_tree(state["baseline_backup"], state["target"])
    baseline = deepcopy(state["baseline"])
    entry = next(
        item for item in baseline["files"] if item["path"] == "tests/test_existing.py"
    )
    entry["sha256"] = "0" * 64
    fake_pin = _rewrite_registered(
        Path(recovery.RECOVERY_BASELINE_SNAPSHOT_PATH), _canonical(baseline)
    )
    proposal = deepcopy(state["proposal"])
    proposal["baseline_source_snapshot"] = fake_pin
    with pytest.raises(
        recovery.Phase6RecoveryAuthorityError, match="live pre-render formal repository"
    ):
        recovery.validate_phase6_recovery_proposal_bytes(
            _canonical(proposal), source_root=os.fspath(state["candidate"])
        )


def test_candidate_and_target_source_phases_are_distinct(governance):
    state = governance
    with pytest.raises(
        recovery.Phase6RecoveryAuthorityError, match="live pre-render formal repository"
    ):
        recovery.validate_phase6_recovery_proposal_bytes(
            _canonical(state["proposal"]), source_root=os.fspath(state["candidate"])
        )
    recovery.validate_phase6_recovery_proposal_bytes(
        _canonical(state["proposal"]),
        source_root=os.fspath(state["target"]),
        authority_sha256=state["authority_sha"],
    )
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="source-root"):
        recovery.validate_phase6_recovery_proposal_bytes(
            _canonical(state["proposal"]),
            source_root=os.fspath(state["candidate"]),
            authority_sha256=state["authority_sha"],
        )
    assert (
        recovery.validate_phase6_planrow_recovery_authority_apply_bytes(
            state["authority_data"],
            expected_authority_sha256=state["authority_sha"],
            candidate_source_root=os.fspath(state["candidate"]),
        )
        == state["authority"]
    )
    assert (
        recovery.validate_phase6_recovery_source_render_plan_bytes(
            state["render_plan_data"],
            candidate_source_root=os.fspath(state["candidate"]),
            expected_authority_sha256=state["authority_sha"],
        )
        == state["render_plan"]
    )


def test_render_plan_recomputes_tokenized_and_rendered_pins(governance):
    state = governance
    plan = deepcopy(state["render_plan"])
    plan["rendered_source"]["sha256"] = "0" * 64
    with pytest.raises(
        recovery.Phase6RecoveryAuthorityError, match="source pins differ"
    ):
        recovery.validate_phase6_recovery_source_render_plan_bytes(
            _canonical(plan),
            candidate_source_root=os.fspath(state["candidate"]),
            expected_authority_sha256=state["authority_sha"],
        )


def test_lifecycle_rejects_bool_int_and_old_keys():
    expected = recovery.recovery_lifecycle()
    assert len(expected) == 16
    recovery.validate_recovery_lifecycle(expected)
    wrong = deepcopy(expected)
    wrong["historical_durable_metrics_rows_total"] = False
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="type differs"):
        recovery.validate_recovery_lifecycle(wrong)
    old = deepcopy(expected)
    old["closed_loop_started"] = old.pop("recovery_closed_loop_started")
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="keys differ"):
        recovery.validate_recovery_lifecycle(old)
    extended = {**expected, "artifact_field": "allowed"}
    recovery.validate_recovery_lifecycle(extended, exact=False)


def test_token_render_is_exact_single_replacement():
    token = recovery._PLANROW_RECOVERY_AUTHORITY_SHA256_TOKEN.encode("ascii")
    digest = "a" * 64
    assert recovery.render_phase6_recovery_source(
        b"left" + token + b"right", digest
    ) == (b"left" + digest.encode("ascii") + b"right")
    for invalid in (b"missing", token + token):
        with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="token count"):
            recovery.render_phase6_recovery_source(invalid, digest)


def test_review_identity_hash_and_exact_keys_fail(governance):
    state = governance
    review = deepcopy(state["reviews"]["A"])
    review["reviewer"]["identity_sha256"] = "0" * 64
    with pytest.raises(
        recovery.Phase6RecoveryAuthorityError, match="identity claim hash"
    ):
        recovery.validate_phase6_recovery_review_receipt_bytes(
            _canonical(review),
            slot="A",
            expected_review_subject_pin=state["subject_pin"],
            review_subject=state["subject"],
            proposal=state["proposal"],
        )
    review = deepcopy(state["reviews"]["A"])
    review["unexpected"] = None
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="keys differ"):
        recovery.validate_phase6_recovery_review_receipt_bytes(
            _canonical(review),
            slot="A",
            expected_review_subject_pin=state["subject_pin"],
            review_subject=state["subject"],
            proposal=state["proposal"],
        )


def test_final_builder_requires_identity_and_session_independence(governance):
    state = governance
    _restore_tree(state["baseline_backup"], state["target"])
    original_b = deepcopy(state["reviews"]["B"])
    duplicate_identity = deepcopy(original_b)
    duplicate_identity["reviewer"]["identity_claim"] = deepcopy(
        state["reviews"]["A"]["reviewer"]["identity_claim"]
    )
    duplicate_identity["reviewer"]["identity_sha256"] = state["reviews"]["A"][
        "reviewer"
    ]["identity_sha256"]
    pin_b = _rewrite_registered(
        Path(recovery.RECOVERY_REVIEW_RECEIPT_PATHS["B"]),
        _canonical(duplicate_identity),
    )
    pins = {**state["review_pins"], "B": pin_b}
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="not independent"):
        recovery.build_phase6_planrow_recovery_authority(
            review_subject_pin=state["subject_pin"],
            review_pins=pins,
            candidate_source_root=os.fspath(state["candidate"]),
            registered_at_utc=TIMES["authority"],
        )

    duplicate_session = deepcopy(original_b)
    duplicate_session["reviewer"]["session_or_task_id"] = state["reviews"]["A"][
        "reviewer"
    ]["session_or_task_id"]
    pin_b = _rewrite_registered(
        Path(recovery.RECOVERY_REVIEW_RECEIPT_PATHS["B"]), _canonical(duplicate_session)
    )
    pins["B"] = pin_b
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="not independent"):
        recovery.build_phase6_planrow_recovery_authority(
            review_subject_pin=state["subject_pin"],
            review_pins=pins,
            candidate_source_root=os.fspath(state["candidate"]),
            registered_at_utc=TIMES["authority"],
        )


def test_subject_and_final_authority_reject_extra_keys(governance):
    state = governance
    subject = deepcopy(state["subject"])
    subject["unexpected"] = 1
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="keys differ"):
        recovery.validate_phase6_recovery_review_subject_bytes(
            _canonical(subject), source_root=os.fspath(state["candidate"])
        )
    authority = deepcopy(state["authority"])
    authority["unexpected"] = 1
    data = _canonical(authority)
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="keys differ"):
        recovery.validate_phase6_planrow_recovery_authority_bytes(
            data,
            expected_authority_sha256=sha256(data).hexdigest(),
            target_repository_root=os.fspath(state["target"]),
        )


def test_rendered_repository_tamper_fails_final_validation(governance):
    state = governance
    path = state["target"] / "src/sana_wam/train/old.py"
    path.write_bytes(b"VALUE = 'tampered'\n")
    with pytest.raises(
        recovery.Phase6RecoveryAuthorityError, match="inventory differs"
    ):
        recovery.validate_phase6_planrow_recovery_authority_bytes(
            state["authority_data"],
            expected_authority_sha256=state["authority_sha"],
            target_repository_root=os.fspath(state["target"]),
        )


def test_registered_read_uses_nofollow_and_rejects_multilink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "registered.json"
    pin = _write(path, b"{}\n", registered=True)
    original_open = os.open
    seen = []

    def recording_open(file, flags, *args, **kwargs):
        seen.append(flags)
        return original_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(recovery.os, "open", recording_open)
    recovery._stable_registered_pin(pin, "registered fixture")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        assert seen and seen[0] & nofollow

    os.chmod(path, 0o600)
    linked = tmp_path / "linked.json"
    try:
        os.link(path, linked)
    except OSError:
        pytest.skip("hard links are unavailable")
    os.chmod(path, 0o400)
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="link count"):
        recovery._stable_registered_pin(pin, "registered fixture")
    os.chmod(path, 0o600)
    os.chmod(linked, 0o600)


def test_registered_read_rejects_symlink(tmp_path: Path):
    target = tmp_path / "target.json"
    pin = _write(target, b"{}\n", registered=True)
    link = tmp_path / "link.json"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlinks are unavailable")
    symlink_pin = {**pin, "path": os.fspath(link)}
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="non-symlink"):
        recovery._stable_registered_pin(symlink_pin, "symlink fixture")
    os.chmod(target, 0o600)


def test_governance_cli_publication_is_exclusive_and_durable(tmp_path: Path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/build_phase6_recovery_governance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase6_recovery_governance_cli", script
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "artifact.json"
    digest = cli._publish_exact(os.fspath(output), os.fspath(output), b"{}\n")
    assert digest == sha256(b"{}\n").hexdigest()
    assert output.read_bytes() == b"{}\n"
    assert os.lstat(output).st_nlink == 1
    assert not cli._publication_temp_path(output, digest).exists()
    assert cli._publish_exact(os.fspath(output), os.fspath(output), b"{}\n") == digest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli._publish_exact(os.fspath(output), os.fspath(output), b"changed\n")
    os.chmod(output, 0o600)


def test_governance_cli_recovers_sigkill_publication_states(tmp_path: Path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/build_phase6_recovery_governance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase6_recovery_governance_crash", script
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    data = b'{"state":"complete"}\n'
    digest = sha256(data).hexdigest()
    outputs = [
        tmp_path / name for name in ("partial.json", "staged.json", "linked.json")
    ]

    partial_temp = cli._publication_temp_path(outputs[0], digest)
    partial_temp.write_bytes(data[:5])
    os.chmod(partial_temp, 0o400)
    assert (
        cli._publish_exact(os.fspath(outputs[0]), os.fspath(outputs[0]), data) == digest
    )
    assert not partial_temp.exists()

    staged_temp = cli._publication_temp_path(outputs[1], digest)
    staged_temp.write_bytes(data)
    os.chmod(staged_temp, 0o400)
    assert (
        cli._publish_exact(os.fspath(outputs[1]), os.fspath(outputs[1]), data) == digest
    )
    assert not staged_temp.exists()

    linked_temp = cli._publication_temp_path(outputs[2], digest)
    linked_temp.write_bytes(data)
    os.chmod(linked_temp, 0o400)
    os.link(linked_temp, outputs[2])
    assert os.lstat(linked_temp).st_nlink == 2
    assert (
        cli._publish_exact(os.fspath(outputs[2]), os.fspath(outputs[2]), data) == digest
    )
    assert not linked_temp.exists()
    assert os.lstat(outputs[2]).st_nlink == 1
    for output in outputs:
        os.chmod(output, 0o600)


def test_governance_cli_prelink_faults_leave_no_final_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/build_phase6_recovery_governance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase6_recovery_governance_faults", script
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    data = b'{"fault":"recoverable"}\n'
    digest = sha256(data).hexdigest()

    write_output = tmp_path / "write_fault.json"
    write_temp = cli._publication_temp_path(write_output, digest)
    with monkeypatch.context() as fault:
        fault.setattr(
            cli.os,
            "write",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError(errno.ENOSPC, "injected write ENOSPC")
            ),
        )
        with pytest.raises(OSError, match="injected write ENOSPC"):
            cli._publish_exact(os.fspath(write_output), os.fspath(write_output), data)
    assert not write_output.exists()
    assert not write_temp.exists()
    assert (
        cli._publish_exact(os.fspath(write_output), os.fspath(write_output), data)
        == digest
    )

    link_output = tmp_path / "link_fault.json"
    link_temp = cli._publication_temp_path(link_output, digest)
    with monkeypatch.context() as fault:
        fault.setattr(
            cli.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError(errno.ENOSPC, "injected link ENOSPC")
            ),
        )
        with pytest.raises(OSError, match="injected link ENOSPC"):
            cli._publish_exact(os.fspath(link_output), os.fspath(link_output), data)
    assert not link_output.exists()
    assert not link_temp.exists()
    assert (
        cli._publish_exact(os.fspath(link_output), os.fspath(link_output), data)
        == digest
    )

    unlink_output = tmp_path / "unlink_fault.json"
    unlink_temp = cli._publication_temp_path(unlink_output, digest)
    real_unlink = cli.os.unlink

    def fail_temp_unlink(path, *args, **kwargs):
        if Path(path) == unlink_temp:
            raise OSError(errno.EIO, "injected post-link unlink failure")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(cli.os, "unlink", fail_temp_unlink)
        with pytest.raises(OSError, match="injected post-link unlink failure"):
            cli._publish_exact(os.fspath(unlink_output), os.fspath(unlink_output), data)
    assert unlink_output.exists()
    assert unlink_temp.exists()
    assert os.lstat(unlink_output).st_nlink == 2
    assert (
        cli._publish_exact(os.fspath(unlink_output), os.fspath(unlink_output), data)
        == digest
    )
    assert not unlink_temp.exists()
    assert os.lstat(unlink_output).st_nlink == 1

    for output in (write_output, link_output, unlink_output):
        os.chmod(output, 0o600)


def test_governance_cli_requires_fixed_cpu_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/build_phase6_recovery_governance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase6_recovery_governance_env", script
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    environment = os.environ.copy()
    environment.update(cli._REQUIRED_CPU_ENVIRONMENT)
    completed = subprocess.run(
        [sys.executable, "-B", os.fspath(script), "--help"],
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    for key, value in cli._REQUIRED_CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(cli.GovernanceCliError, match="CUDA_VISIBLE_DEVICES"):
        cli._assert_cpu_only_runtime()
