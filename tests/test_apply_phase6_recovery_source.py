from __future__ import annotations

from copy import deepcopy
import errno
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from sana_wam.train import phase6_recovery as recovery


TOKEN = recovery._PLANROW_RECOVERY_AUTHORITY_SHA256_TOKEN.encode("ascii")


def _canonical(value: Any) -> bytes:
    return recovery.canonical_json_bytes(value)


def _pin(data: bytes) -> dict[str, Any]:
    return {"sha256": sha256(data).hexdigest(), "size_bytes": len(data)}


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


def _load_apply_module(name: str):
    script = (
        Path(__file__).resolve().parents[1] / "scripts/apply_phase6_recovery_source.py"
    )
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_exchange(source: Path, destination: Path, _parent_fd=None) -> bool:
    """Exercise exchange recovery logic on non-Linux test hosts only."""
    temporary = destination.with_name(f".{destination.name}.pytest-exchange")
    assert not temporary.exists()
    os.replace(destination, temporary)
    try:
        os.replace(source, destination)
        os.replace(temporary, source)
    except BaseException:
        if temporary.exists() and not destination.exists():
            os.replace(temporary, destination)
        raise
    return True


@pytest.fixture
def source_apply_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "target"
    candidate = tmp_path / "candidate"
    governance = tmp_path / "governance"
    for root in (target, candidate):
        (root / "scripts").mkdir(parents=True)
        (root / "src/sana_wam/train").mkdir(parents=True)

    unchanged = b"UNCHANGED = True\n"
    modified_before = b"VALUE = 'before'\n"
    modified_after = b"VALUE = 'after'\n"
    added = b"ADDED = True\n"
    tokenized_recovery = b"AUTHORITY = b'" + TOKEN + b"'\n"
    _write(target / "scripts/modified.py", modified_before)
    _write(target / "src/unchanged.py", unchanged)
    _write(candidate / "scripts/modified.py", modified_after)
    _write(candidate / "scripts/new.py", added)
    _write(candidate / "src/unchanged.py", unchanged)
    _write(candidate / "src/sana_wam/train/phase6_recovery.py", tokenized_recovery)
    os.chmod(target / "scripts/modified.py", 0o664)
    os.chmod(candidate / "scripts/modified.py", 0o664)

    baseline = {
        "files": [
            {"path": "scripts/modified.py", **_pin(modified_before)},
            {"path": "src/unchanged.py", **_pin(unchanged)},
        ]
    }
    proposal = {
        "source_inventory": {
            "files": [
                {"path": "scripts/modified.py", **_pin(modified_after)},
                {"path": "scripts/new.py", **_pin(added)},
                {
                    "path": recovery.RECOVERY_SOURCE_PATH,
                    **_pin(tokenized_recovery),
                },
                {"path": "src/unchanged.py", **_pin(unchanged)},
            ]
        }
    }
    baseline_pin = _write(
        governance / "baseline.json", _canonical(baseline), registered=True
    )
    proposal_pin = _write(
        governance / "proposal.json", _canonical(proposal), registered=True
    )
    delta = [
        {
            "before": _pin(modified_before),
            "candidate": _pin(modified_after),
            "change": "modified",
            "path": "scripts/modified.py",
        },
        {
            "before": None,
            "candidate": _pin(added),
            "change": "added",
            "path": "scripts/new.py",
        },
        {
            "before": None,
            "candidate": _pin(tokenized_recovery),
            "change": "added",
            "path": recovery.RECOVERY_SOURCE_PATH,
        },
    ]
    authority = {
        "authorized_source_delta": delta,
        "baseline_source_snapshot": baseline_pin,
        "proposal": proposal_pin,
        "registered_at_utc": "2026-01-01T00:00:00Z",
    }
    authority_data = _canonical(authority)
    authority_pin = _write(
        governance / "authority.json", authority_data, registered=True
    )
    rendered_recovery = recovery.render_phase6_recovery_source(
        tokenized_recovery, authority_pin["sha256"]
    )
    render_plan = {
        "artifact_path": os.fspath(governance / "render_plan.json"),
        "artifact_role": "phase6_recovery_source_render_plan",
        "authority": authority_pin,
        "candidate_source_root": os.fspath(candidate),
        "registered_at_utc": "2026-01-01T00:00:30Z",
        "rendered_source": {
            "path": recovery.RECOVERY_SOURCE_PATH,
            **_pin(rendered_recovery),
        },
        "schema_version": recovery.RECOVERY_RENDER_PLAN_SCHEMA_VERSION,
        "source_write_performed": False,
        "status": "ready",
        "target_repository_root": os.fspath(target),
        "tokenized_source": {
            "path": recovery.RECOVERY_SOURCE_PATH,
            **_pin(tokenized_recovery),
        },
    }
    render_plan_pin = _write(
        governance / "render_plan.json", _canonical(render_plan), registered=True
    )
    receipt_path = governance / "execution_receipt.json"

    monkeypatch.setattr(recovery, "REPOSITORY_ROOT", os.fspath(target))
    monkeypatch.setattr(
        recovery, "PLANROW_RECOVERY_AUTHORITY_PATH", authority_pin["path"]
    )
    monkeypatch.setattr(recovery, "RECOVERY_RENDER_PLAN_PATH", render_plan_pin["path"])
    monkeypatch.setattr(
        recovery, "RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH", os.fspath(receipt_path)
    )
    prior_modes = {"scripts/modified.py": 0o664}
    monkeypatch.setattr(
        recovery, "prior_recovery_source_modes", lambda: deepcopy(prior_modes)
    )
    monkeypatch.setattr(
        recovery, "SOURCE_INVENTORY_GLOBS", ("scripts/**/*.py", "src/**/*.py")
    )

    final_calls: list[str] = []
    prerender_calls: list[str] = []
    final_failure = {"enabled": False}

    def validate_apply(
        data: bytes,
        *,
        expected_authority_sha256: str,
        candidate_source_root: str,
    ):
        assert data == authority_data
        assert expected_authority_sha256 == authority_pin["sha256"]
        assert candidate_source_root == os.fspath(candidate)
        return deepcopy(authority)

    def validate_proposal(data: bytes, *, candidate_source_root: str):
        assert isinstance(data, bytes)
        assert candidate_source_root == os.fspath(candidate)
        return deepcopy(proposal)

    def validate_baseline(data: bytes):
        assert data == _canonical(baseline)
        return deepcopy(baseline)

    def validate_render_plan(
        data: bytes,
        *,
        candidate_source_root: str,
        expected_authority_sha256: str,
    ):
        assert data == _canonical(render_plan)
        assert candidate_source_root == os.fspath(candidate)
        assert expected_authority_sha256 == authority_pin["sha256"]
        return deepcopy(render_plan)

    def validate_prerender(
        data: bytes,
        *,
        expected_authority_sha256: str,
        candidate_source_root: str,
    ):
        prerender_calls.append("called")
        assert data == authority_data
        assert expected_authority_sha256 == authority_pin["sha256"]
        assert candidate_source_root == os.fspath(candidate)
        return deepcopy(authority)

    def validate_final(
        data: bytes,
        *,
        expected_authority_sha256: str,
        target_repository_root: str,
    ):
        final_calls.append("called")
        if final_failure["enabled"]:
            raise recovery.Phase6RecoveryAuthorityError("injected final validation")
        assert data == authority_data
        assert expected_authority_sha256 == authority_pin["sha256"]
        assert target_repository_root == os.fspath(target)
        assert (target / "scripts/modified.py").read_bytes() == modified_after
        assert (target / "scripts/new.py").read_bytes() == added
        assert (
            target / "src/sana_wam/train/phase6_recovery.py"
        ).read_bytes() == rendered_recovery
        return deepcopy(authority)

    monkeypatch.setattr(
        recovery,
        "validate_phase6_planrow_recovery_authority_apply_bytes",
        validate_apply,
    )
    monkeypatch.setattr(
        recovery, "validate_phase6_recovery_proposal_apply_bytes", validate_proposal
    )
    monkeypatch.setattr(
        recovery, "validate_phase6_recovery_baseline_snapshot_bytes", validate_baseline
    )
    monkeypatch.setattr(
        recovery,
        "validate_phase6_recovery_source_render_plan_bytes",
        validate_render_plan,
    )
    monkeypatch.setattr(
        recovery,
        "validate_phase6_planrow_recovery_authority_prerender_bytes",
        validate_prerender,
    )
    monkeypatch.setattr(
        recovery, "validate_phase6_planrow_recovery_authority_bytes", validate_final
    )
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "GIT_OPTIONAL_LOCKS": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }.items():
        monkeypatch.setenv(key, value)

    apply = _load_apply_module("phase6_recovery_source_apply_fixture")
    test_platform = sys.platform
    if not sys.platform.startswith("linux"):
        test_platform = "linux"
        monkeypatch.setattr(apply, "_rename_exchange", _test_exchange)
    monkeypatch.setattr(
        apply,
        "sys",
        SimpleNamespace(
            dont_write_bytecode=sys.dont_write_bytecode,
            modules={},
            platform=test_platform,
            stderr=sys.stderr,
        ),
    )
    args = SimpleNamespace(
        authority_sha256=authority_pin["sha256"],
        authority_size_bytes=authority_pin["size_bytes"],
        candidate_source_root=os.fspath(candidate),
        output=os.fspath(receipt_path),
        render_plan_sha256=render_plan_pin["sha256"],
        render_plan_size_bytes=render_plan_pin["size_bytes"],
        registered_at_utc="2026-01-01T00:01:00Z",
        target_repository_root=os.fspath(target),
    )
    state = {
        "added": added,
        "apply": apply,
        "args": args,
        "authority": authority,
        "authority_pin": authority_pin,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "final_calls": final_calls,
        "final_failure": final_failure,
        "modified_after": modified_after,
        "modified_before": modified_before,
        "proposal": proposal,
        "prior_modes": prior_modes,
        "prerender_calls": prerender_calls,
        "receipt": receipt_path,
        "render_plan": render_plan,
        "render_plan_pin": render_plan_pin,
        "rendered_recovery": rendered_recovery,
        "target": target,
        "tokenized_recovery": tokenized_recovery,
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


def test_source_apply_end_to_end_receipt_last_and_idempotent(source_apply_case):
    state = source_apply_case
    apply = state["apply"]
    digest, size = apply._execute(state["args"])
    receipt_data = state["receipt"].read_bytes()
    assert digest == sha256(receipt_data).hexdigest()
    assert size == len(receipt_data)
    receipt = recovery._strict_json(receipt_data, "source execution receipt fixture")
    assert receipt["source_write_completed"] is True
    assert receipt["receipt_is_commit_marker"] is True
    assert receipt["gpu_executed"] is False
    assert receipt["proposal"] == state["authority"]["proposal"]
    assert receipt["source_render_plan"] == state["render_plan_pin"]
    assert receipt["authorized_source_delta"] == state["delta"]
    assert receipt["lifecycle"] == recovery.recovery_lifecycle()
    assert len(receipt["lifecycle"]) == 16
    assert [entry["state"] for entry in receipt["files"]] == ["rendered_final"] * 3
    rendered_files = {
        entry["path"]: entry for entry in receipt["rendered_source_inventory"]["files"]
    }
    assert rendered_files[recovery.RECOVERY_SOURCE_PATH] == {
        "path": recovery.RECOVERY_SOURCE_PATH,
        **_pin(state["rendered_recovery"]),
    }
    assert receipt["rendered_source_inventory"]["representation"] == "raw-rendered"
    assert state["prerender_calls"] == ["called"]
    assert state["final_calls"] == ["called", "called", "called"]
    assert os.lstat(state["receipt"]).st_nlink == 1
    if os.name == "posix":
        assert (
            stat.S_IMODE(os.lstat(state["target"] / "scripts/modified.py").st_mode)
            == 0o664
        )
        assert (
            stat.S_IMODE(os.lstat(state["target"] / "scripts/new.py").st_mode) == 0o644
        )

    state["args"].registered_at_utc = "2026-01-01T00:02:00Z"
    digest_again, size_again = apply._execute(state["args"])
    assert (digest_again, size_again) == (digest, size)
    assert state["receipt"].read_bytes() == receipt_data
    assert state["final_calls"] == ["called", "called", "called", "called"]


def test_source_apply_partial_commit_restarts_from_before_final_mix(
    source_apply_case, monkeypatch: pytest.MonkeyPatch
):
    state = source_apply_case
    apply = state["apply"]
    real_link = apply.os.link

    def fail_first_add(source, destination, *args, **kwargs):
        if Path(destination).name == "new.py":
            raise OSError(errno.EIO, "injected partial commit")
        return real_link(source, destination, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(apply.os, "link", fail_first_add)
        with pytest.raises(OSError, match="injected partial commit"):
            apply._execute(state["args"])
    assert (state["target"] / "scripts/modified.py").read_bytes() == state[
        "modified_after"
    ]
    assert not (state["target"] / "scripts/new.py").exists()
    assert not state["receipt"].exists()

    apply._execute(state["args"])
    assert (state["target"] / "scripts/new.py").read_bytes() == state["added"]
    assert state["receipt"].exists()
    assert state["prerender_calls"] == ["called"]


def test_source_apply_recovers_modified_replace_crash_window(
    source_apply_case, monkeypatch: pytest.MonkeyPatch
):
    state = source_apply_case
    apply = state["apply"]
    real_replace = apply._replace_modified

    def fail_after_replace(item):
        apply._rename_exchange(
            item["stage_path"], item["destination"], item["parent_fd"]
        )
        raise OSError(errno.EIO, "injected replace crash-window")

    with monkeypatch.context() as fault:
        fault.setattr(apply, "_replace_modified", fail_after_replace)
        with pytest.raises(OSError, match="replace crash-window"):
            apply._execute(state["args"])
    assert (state["target"] / "scripts/modified.py").read_bytes() == state[
        "modified_after"
    ]
    assert not state["receipt"].exists()

    assert apply._replace_modified is real_replace
    apply._execute(state["args"])
    assert state["receipt"].exists()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="renameat2 exchange is Linux-only"
)
def test_source_apply_rejects_unwitnessed_displaced_third_state(
    source_apply_case, monkeypatch: pytest.MonkeyPatch
):
    state = source_apply_case
    apply = state["apply"]
    destination = state["target"] / "scripts/modified.py"
    third_state = b"VALUE = 'concurrent third state'\n"
    expected_stage = apply._stage_path(
        destination,
        "scripts/modified.py",
        _pin(state["modified_after"]),
        state["authority_pin"]["sha256"],
    )

    def displace_third_state_then_crash(item):
        item["destination"].write_bytes(third_state)
        os.chmod(item["destination"], 0o664)
        apply._rename_exchange(
            item["stage_path"], item["destination"], item["parent_fd"]
        )
        apply._fsync_parent_at(item["destination"], item["parent_fd"])
        raise OSError(errno.EIO, "injected displaced-third-state crash")

    with monkeypatch.context() as fault:
        fault.setattr(apply, "_replace_modified", displace_third_state_then_crash)
        with pytest.raises(OSError, match="displaced-third-state crash"):
            apply._execute(state["args"])
    assert destination.read_bytes() == state["modified_after"]
    assert expected_stage.read_bytes() == third_state
    assert not state["receipt"].exists()

    with pytest.raises(apply.SourceApplyError, match="unwitnessed staged source"):
        apply._execute(state["args"])
    assert destination.read_bytes() == state["modified_after"]
    assert expected_stage.read_bytes() == third_state
    assert not state["receipt"].exists()

    with pytest.raises(apply.SourceApplyError, match="unwitnessed staged source"):
        apply._execute(state["args"])
    assert destination.read_bytes() == state["modified_after"]
    assert expected_stage.read_bytes() == third_state
    assert not state["receipt"].exists()


def test_source_apply_rejects_unknown_stage_without_deleting_it(source_apply_case):
    state = source_apply_case
    apply = state["apply"]
    destination = state["target"] / "scripts/modified.py"
    stage = apply._stage_path(
        destination,
        "scripts/modified.py",
        _pin(state["modified_after"]),
        state["authority_pin"]["sha256"],
    )
    unknown = b"VALUE = 'unknown deterministic stage'\n"
    stage.write_bytes(unknown)
    if os.name == "posix":
        os.chmod(stage, 0o664)

    with pytest.raises(apply.SourceApplyError, match="unwitnessed staged source"):
        apply._execute(state["args"])

    assert destination.read_bytes() == state["modified_before"]
    assert stage.read_bytes() == unknown
    assert not state["receipt"].exists()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="renameat2 exchange is Linux-only"
)
def test_source_apply_rollback_refuses_replaced_installed_inode(
    source_apply_case, monkeypatch: pytest.MonkeyPatch
):
    state = source_apply_case
    apply = state["apply"]
    destination = state["target"] / "scripts/modified.py"
    real_stable = apply._stable_source
    injected = {"done": False}
    third = b"VALUE = 'post-exchange concurrent replacement'\n"

    def replace_before_rollback(path, relative, expected, **kwargs):
        if (
            not injected["done"]
            and Path(path).name.startswith(".phase6-source-")
            and expected == _pin(state["modified_before"])
        ):
            injected["done"] = True
            replacement = destination.with_name(".concurrent-replacement.py")
            replacement.write_bytes(third)
            os.chmod(replacement, 0o664)
            os.replace(replacement, destination)
            raise apply.SourceApplyError("injected validation failure")
        return real_stable(path, relative, expected, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(apply, "_stable_source", replace_before_rollback)
        with pytest.raises(apply.SourceApplyError, match="rollback failed"):
            apply._execute(state["args"])

    assert injected["done"] is True
    assert destination.read_bytes() == third
    assert not state["receipt"].exists()


def test_source_apply_stage_enospc_leaves_formal_tree_unmodified(
    source_apply_case, monkeypatch: pytest.MonkeyPatch
):
    state = source_apply_case
    apply = state["apply"]
    before = (state["target"] / "scripts/modified.py").read_bytes()
    with apply._source_apply_lock():
        pass
    with monkeypatch.context() as fault:
        fault.setattr(
            apply.os,
            "write",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError(errno.ENOSPC, "injected stage ENOSPC")
            ),
        )
        with pytest.raises(OSError, match="injected stage ENOSPC"):
            apply._execute(state["args"])
    assert (state["target"] / "scripts/modified.py").read_bytes() == before
    assert not (state["target"] / "scripts/new.py").exists()
    assert not state["receipt"].exists()


def test_source_apply_rejects_non_linux_before_staging(
    source_apply_case, monkeypatch: pytest.MonkeyPatch
):
    state = source_apply_case
    apply = state["apply"]
    before = (state["target"] / "scripts/modified.py").read_bytes()
    with monkeypatch.context() as fault:
        fault.setattr(apply.sys, "platform", "win32")
        with pytest.raises(apply.SourceApplyError, match="requires Linux"):
            apply._execute(state["args"])
    assert (state["target"] / "scripts/modified.py").read_bytes() == before
    assert not list(state["target"].rglob(".phase6-source-*"))
    assert not state["receipt"].exists()


def test_source_apply_late_stage_failure_has_zero_target_commits(
    source_apply_case, monkeypatch: pytest.MonkeyPatch
):
    state = source_apply_case
    apply = state["apply"]
    real_prepare = apply._prepare_stage
    calls = {"count": 0}

    def fail_third_stage(item):
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError(errno.ENOSPC, "injected late-stage ENOSPC")
        return real_prepare(item)

    before = (state["target"] / "scripts/modified.py").read_bytes()
    with monkeypatch.context() as fault:
        fault.setattr(apply, "_prepare_stage", fail_third_stage)
        with pytest.raises(OSError, match="late-stage ENOSPC"):
            apply._execute(state["args"])
    assert calls["count"] == 3
    assert (state["target"] / "scripts/modified.py").read_bytes() == before
    assert not (state["target"] / "scripts/new.py").exists()
    assert not state["receipt"].exists()

    apply._execute(state["args"])
    assert state["receipt"].exists()


def test_source_apply_invalid_late_python_has_zero_stages_and_writes(
    source_apply_case,
):
    state = source_apply_case
    invalid = b"def broken(:\n"
    candidate = state["candidate"] / "scripts/new.py"
    candidate.write_bytes(invalid)
    pin = _pin(invalid)
    proposal_entry = next(
        entry
        for entry in state["proposal"]["source_inventory"]["files"]
        if entry["path"] == "scripts/new.py"
    )
    proposal_entry.update(pin)
    delta_entry = next(
        entry
        for entry in state["authority"]["authorized_source_delta"]
        if entry["path"] == "scripts/new.py"
    )
    delta_entry["candidate"] = pin
    before = (state["target"] / "scripts/modified.py").read_bytes()
    with pytest.raises(state["apply"].SourceApplyError, match="Python syntax"):
        state["apply"]._execute(state["args"])
    assert (state["target"] / "scripts/modified.py").read_bytes() == before
    assert not list(state["target"].rglob(".phase6-source-*.tmp"))
    assert not state["receipt"].exists()


def test_source_apply_preserves_prior_receipt_authorized_0644_mode(source_apply_case):
    if os.name != "posix":
        pytest.skip("POSIX mode distinctions are required")
    state = source_apply_case
    target = state["target"] / "scripts/modified.py"
    candidate = state["candidate"] / "scripts/modified.py"
    os.chmod(target, 0o644)
    os.chmod(candidate, 0o644)
    state["prior_modes"]["scripts/modified.py"] = 0o644

    state["apply"]._execute(state["args"])

    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    receipt = recovery._strict_json(
        state["receipt"].read_bytes(), "source execution receipt fixture"
    )
    modified = next(
        entry for entry in receipt["files"] if entry["path"] == "scripts/modified.py"
    )
    assert modified["before_mode"] == "0644"
    assert modified["rendered_mode"] == "0644"


def test_source_apply_rejects_modified_mode_drift(source_apply_case):
    if os.name != "posix":
        pytest.skip("POSIX mode distinctions are required")
    state = source_apply_case
    target = state["target"] / "scripts/modified.py"
    os.chmod(target, 0o644)
    with pytest.raises(state["apply"].SourceApplyError, match="outside before/final"):
        state["apply"]._execute(state["args"])
    assert not (state["target"] / "scripts/new.py").exists()
    assert not state["receipt"].exists()


def test_source_apply_recovers_added_hardlink_window(
    source_apply_case, monkeypatch: pytest.MonkeyPatch
):
    state = source_apply_case
    apply = state["apply"]
    destination = state["target"] / "scripts/new.py"
    expected_stage = apply._stage_path(
        destination,
        "scripts/new.py",
        _pin(state["added"]),
        state["authority_pin"]["sha256"],
    )
    real_unlink = apply.os.unlink

    def fail_link_window_unlink(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.name == expected_stage.name:
            raise OSError(errno.EIO, "injected source link-window")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(apply.os, "unlink", fail_link_window_unlink)
        with pytest.raises(OSError, match="injected source link-window"):
            apply._execute(state["args"])
    stage = expected_stage
    assert destination.exists()
    assert os.lstat(destination).st_nlink == 2
    assert os.lstat(stage).st_ino == os.lstat(destination).st_ino
    assert not state["receipt"].exists()

    apply._execute(state["args"])
    assert not stage.exists()
    assert os.lstat(destination).st_nlink == 1
    assert state["receipt"].exists()


def test_source_apply_rejects_target_drift_before_any_write(source_apply_case):
    state = source_apply_case
    (state["target"] / "src/unchanged.py").write_bytes(b"DRIFT = True\n")
    with pytest.raises(
        (state["apply"].SourceApplyError, recovery.Phase6RecoveryAuthorityError)
    ):
        state["apply"]._execute(state["args"])
    assert (state["target"] / "scripts/modified.py").read_bytes() != state[
        "modified_after"
    ]
    assert not (state["target"] / "scripts/new.py").exists()
    assert not state["receipt"].exists()


def test_source_apply_rejects_extra_path_before_any_write(source_apply_case):
    state = source_apply_case
    _write(state["target"] / "scripts/extra.py", b"EXTRA = True\n")
    with pytest.raises(state["apply"].SourceApplyError, match="extra source paths"):
        state["apply"]._execute(state["args"])
    assert (state["target"] / "scripts/modified.py").read_bytes() != state[
        "modified_after"
    ]
    assert not state["receipt"].exists()


def test_source_apply_rejects_candidate_multilink(source_apply_case, tmp_path: Path):
    state = source_apply_case
    source = state["candidate"] / "scripts/modified.py"
    link = tmp_path / "candidate-hardlink.py"
    try:
        os.link(source, link)
    except OSError:
        pytest.skip("hard links are unavailable")
    with pytest.raises(
        (state["apply"].SourceApplyError, recovery.Phase6RecoveryAuthorityError),
        match="descriptor precondition|link count",
    ):
        state["apply"]._execute(state["args"])
    assert not state["receipt"].exists()


def test_source_apply_rejects_target_symlink(source_apply_case, tmp_path: Path):
    state = source_apply_case
    outside = tmp_path / "outside.py"
    outside.write_bytes(state["added"])
    destination = state["target"] / "scripts/new.py"
    try:
        os.symlink(outside, destination)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(state["apply"].SourceApplyError, match="regular file"):
        state["apply"]._execute(state["args"])
    assert outside.read_bytes() == state["added"]
    assert not state["receipt"].exists()


def test_source_apply_rejects_parent_escape(source_apply_case, tmp_path: Path):
    state = source_apply_case
    parent = state["target"] / "src/sana_wam/train"
    parent.rmdir()
    outside = tmp_path / "outside-parent"
    outside.mkdir()
    try:
        os.symlink(outside, parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(state["apply"].SourceApplyError, match="escapes repository"):
        state["apply"]._execute(state["args"])
    assert not list(outside.iterdir())
    assert not state["receipt"].exists()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="directory-fd binding is Linux-only"
)
def test_source_apply_rejects_parent_rebind_before_mutation(
    source_apply_case, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    state = source_apply_case
    apply = state["apply"]
    target_parent = state["target"] / "scripts"
    detached = tmp_path / "detached-scripts"
    real_apply = apply._apply_plan

    def rebind_then_apply(plan, states):
        os.rename(target_parent, detached)
        target_parent.mkdir()
        _write(target_parent / "modified.py", state["modified_before"])
        return real_apply(plan, states)

    with monkeypatch.context() as fault:
        fault.setattr(apply, "_apply_plan", rebind_then_apply)
        with pytest.raises(apply.SourceApplyError, match="parent path was rebound"):
            apply._execute(state["args"])

    assert (detached / "modified.py").read_bytes() == state["modified_before"]
    assert not (detached / "new.py").exists()
    assert (target_parent / "modified.py").read_bytes() == state["modified_before"]
    assert not (target_parent / "new.py").exists()
    assert not state["receipt"].exists()


@pytest.mark.parametrize("tokenized", [b"AUTHORITY = b'missing'\n", TOKEN + TOKEN])
def test_source_apply_rejects_non_single_recovery_token(source_apply_case, tokenized):
    state = source_apply_case
    recovery_path = state["candidate"] / recovery.RECOVERY_SOURCE_PATH
    recovery_path.write_bytes(tokenized)
    pin = _pin(tokenized)
    recovery_entry = next(
        entry
        for entry in state["proposal"]["source_inventory"]["files"]
        if entry["path"] == recovery.RECOVERY_SOURCE_PATH
    )
    recovery_entry.update(pin)
    delta_entry = next(
        entry
        for entry in state["authority"]["authorized_source_delta"]
        if entry["path"] == recovery.RECOVERY_SOURCE_PATH
    )
    delta_entry["candidate"] = pin
    with pytest.raises(recovery.Phase6RecoveryAuthorityError, match="token count"):
        state["apply"]._execute(state["args"])
    assert not state["receipt"].exists()


def test_source_apply_final_validator_failure_never_publishes_receipt(
    source_apply_case,
):
    state = source_apply_case
    state["final_failure"]["enabled"] = True
    with pytest.raises(
        recovery.Phase6RecoveryAuthorityError, match="injected final validation"
    ):
        state["apply"]._execute(state["args"])
    assert (state["target"] / "scripts/new.py").exists()
    assert not state["receipt"].exists()

    state["final_failure"]["enabled"] = False
    state["apply"]._execute(state["args"])
    assert state["receipt"].exists()


def test_source_apply_receipt_publication_failure_is_last_and_retryable(
    source_apply_case, monkeypatch: pytest.MonkeyPatch
):
    state = source_apply_case
    apply = state["apply"]
    with monkeypatch.context() as fault:
        fault.setattr(
            apply.publication,
            "_publish_exact",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError(errno.ENOSPC, "injected receipt publication")
            ),
        )
        with pytest.raises(OSError, match="injected receipt publication"):
            apply._execute(state["args"])
    assert (state["target"] / "scripts/new.py").exists()
    assert not state["receipt"].exists()

    apply._execute(state["args"])
    assert state["receipt"].exists()


def test_source_apply_removes_receipt_if_tree_drifts_during_publication(
    source_apply_case, monkeypatch: pytest.MonkeyPatch
):
    state = source_apply_case
    apply = state["apply"]
    target = state["target"] / "scripts/modified.py"
    tampered = b"VALUE = 'tampered during receipt publication'\n"
    real_publish = apply.publication._publish_exact

    def tamper_then_publish(*args, **kwargs):
        target.write_bytes(tampered)
        if os.name == "posix":
            os.chmod(target, 0o664)
        return real_publish(*args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(apply.publication, "_publish_exact", tamper_then_publish)
        with pytest.raises(apply.SourceApplyError, match="outside before/final"):
            apply._execute(state["args"])

    assert target.read_bytes() == tampered
    assert not state["receipt"].exists()


def test_source_apply_valid_receipt_does_not_mask_target_tamper(source_apply_case):
    state = source_apply_case
    state["apply"]._execute(state["args"])
    receipt_data = state["receipt"].read_bytes()
    target = state["target"] / "scripts/modified.py"
    target.write_bytes(b"VALUE = 'tampered after receipt'\n")
    with pytest.raises(state["apply"].SourceApplyError, match="outside before/final"):
        state["apply"]._execute(state["args"])
    assert target.read_bytes() == b"VALUE = 'tampered after receipt'\n"
    assert state["receipt"].read_bytes() == receipt_data


def test_source_apply_valid_receipt_forbids_exact_baseline_rollback(
    source_apply_case,
):
    state = source_apply_case
    state["apply"]._execute(state["args"])
    target = state["target"] / "scripts/modified.py"
    target.write_bytes(state["modified_before"])
    if os.name == "posix":
        os.chmod(target, 0o664)
    with pytest.raises(
        state["apply"].SourceApplyError, match="receipt requires an exact final tree"
    ):
        state["apply"]._execute(state["args"])
    assert target.read_bytes() == state["modified_before"]
    assert (state["target"] / "scripts/new.py").read_bytes() == state["added"]


def test_source_apply_valid_receipt_forbids_added_file_removal(source_apply_case):
    state = source_apply_case
    state["apply"]._execute(state["args"])
    added = state["target"] / "scripts/new.py"
    os.unlink(added)
    with pytest.raises(
        state["apply"].SourceApplyError, match="receipt requires an exact final tree"
    ):
        state["apply"]._execute(state["args"])
    assert not added.exists()
    assert (state["target"] / "scripts/modified.py").read_bytes() == state[
        "modified_after"
    ]


def test_source_apply_valid_receipt_does_not_repair_source_link_window(
    source_apply_case,
):
    state = source_apply_case
    apply = state["apply"]
    apply._execute(state["args"])
    destination = state["target"] / "scripts/new.py"
    stage = apply._stage_path(
        destination,
        "scripts/new.py",
        _pin(state["added"]),
        state["authority_pin"]["sha256"],
    )
    os.link(destination, stage)
    with pytest.raises(
        apply.SourceApplyError, match="receipt requires an exact final tree"
    ):
        apply._execute(state["args"])
    assert stage.exists()
    assert os.lstat(stage).st_nlink == 2
    assert os.lstat(destination).st_nlink == 2


def test_source_apply_receipt_rejects_bool_int_alias(source_apply_case):
    state = source_apply_case
    state["apply"]._execute(state["args"])
    receipt = recovery._strict_json(
        state["receipt"].read_bytes(), "source execution receipt fixture"
    )
    receipt["gpu_executed"] = 0
    os.chmod(state["receipt"], 0o600)
    state["receipt"].write_bytes(_canonical(receipt))
    os.chmod(state["receipt"], 0o400)
    with pytest.raises(state["apply"].SourceApplyError, match="type differs"):
        state["apply"]._execute(state["args"])


def test_source_apply_rejects_authorized_deletion(source_apply_case):
    state = source_apply_case
    state["authority"]["authorized_source_delta"][0] = {
        "before": state["authority"]["authorized_source_delta"][0]["before"],
        "candidate": None,
        "change": "deleted",
        "path": "scripts/modified.py",
    }
    with pytest.raises(state["apply"].SourceApplyError, match="deletion is forbidden"):
        state["apply"]._execute(state["args"])
    assert not state["receipt"].exists()
