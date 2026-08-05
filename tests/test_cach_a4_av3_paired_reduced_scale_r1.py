from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale_r1.py"
BASE_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale.py"
CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale_r1.yaml"
CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_EXECUTION_R1_RUN_CARD.json"
)
BASE_RUNNER_SHA256 = "7f20aee0662cbabf00e2138506ddafb2fbfe75965cdc32ec973e74813b8b6c27"
CONFIG_SHA256 = "335a559dd79cfaccea36df32212de10c2b604427701e2609321c17ebfc259a90"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_runner() -> object:
    name = "_cach_a4_av3_paired_reduced_scale_r1_test_target"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_r1_overlay_and_frozen_base_identity() -> None:
    payload = CONFIG_PATH.read_bytes()
    overlay = json.loads(payload)
    assert payload == _canonical(overlay)
    assert hashlib.sha256(payload).hexdigest() == CONFIG_SHA256
    assert overlay["fix_contract"] == {
        "new_expression": "zip(offsets[:-1], offsets[1:], strict=True)",
        "old_expression": "zip(offsets, offsets[1:], strict=True)",
        "only_semantic_delta": "RESOURCE_OBSERVATION_ADJACENT_PAIR_LENGTH_ALIGNMENT",
    }
    assert overlay["authority"]["statement"] == "授权 AV3-R1"
    assert overlay["execution_override"]["automatic_rerun"] is False
    assert overlay["execution_override"]["root_reuse"] is False

    metadata = BASE_RUNNER_PATH.lstat()
    assert stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o444
    assert hashlib.sha256(BASE_RUNNER_PATH.read_bytes()).hexdigest() == BASE_RUNNER_SHA256


def test_r1_card_binds_failure_record_and_materialized_files() -> None:
    payload = CARD_PATH.read_bytes()
    card = json.loads(payload)
    assert payload == _canonical(card)
    failure = card["failure_predecessor"]
    terminal = failure["terminal_record"]
    assert failure["authorized_base_card_sha256"] == (
        "72124b63e278490ff39262aaa5bcf5a71121a533df6f55aff7147d98b11e06f9"
    )
    assert failure["base_namespace_absent_after_failure"] is True
    assert terminal["state"] == "ENV_BLOCKED"
    assert terminal["phase"] == "resource_observation"
    assert terminal["root_created"] is False
    assert hashlib.sha256(_canonical(terminal)).hexdigest() == (
        failure["terminal_record_sha256"]
    )
    for role, path in {
        "config": CONFIG_PATH,
        "runner": RUNNER_PATH,
        "test": Path(__file__),
    }.items():
        assert card["materialized_pins"][role] == {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_r1_observation_uses_equal_length_adjacent_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner._base, "_filesystem_snapshot", lambda: {"kind": "fake-fs"})
    monkeypatch.setattr(
        runner._base,
        "_related_process_snapshot",
        lambda: {"kind": "fake-processes"},
    )
    now = [0.0]
    sleeps: list[float] = []
    snapshots: list[float] = []

    def monotonic() -> float:
        return now[0]

    def sleeper(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    def snapshot() -> dict[str, object]:
        snapshots.append(now[0])
        return {
            "captured_unix_ns": len(snapshots),
            "compute_apps": [],
            "gpu_uuid": runner.EXPECTED_GPU_UUID,
            "index": runner.EXPECTED_GPU_INDEX,
            "memory_used_mib": 0,
            "utilization_gpu_percent": 0,
        }

    result = runner._observe_resources(
        seconds=120,
        interval=5,
        monotonic=monotonic,
        sleeper=sleeper,
        snapshot=snapshot,
    )
    expected = [float(value) for value in range(0, 121, 5)]
    assert [
        sample["monotonic_offset_seconds"] for sample in result["gpu_samples"]
    ] == expected
    assert result["duration_observed_seconds"] == 120.0
    assert result["sample_count"] == 25
    assert result["maximum_sample_gap_seconds"] == 5.0
    assert snapshots == expected
    assert sleeps == [5.0] * 24


def test_r1_observation_rejects_oversized_sample_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner._base, "_filesystem_snapshot", lambda: {})
    monkeypatch.setattr(runner._base, "_related_process_snapshot", lambda: {})
    now = [0.0]

    def sleeper(_delay: float) -> None:
        now[0] += 8.0

    with pytest.raises(RuntimeError, match="sample count/gap differs"):
        runner._observe_resources(
            seconds=16,
            interval=5,
            monotonic=lambda: now[0],
            sleeper=sleeper,
            snapshot=lambda: {},
        )


def test_r1_attempt_receipt_is_single_use_and_frozen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    attempt_dir = tmp_path / "attempt"
    monkeypatch.setattr(runner, "ATTEMPT_DIR", attempt_dir)
    card_sha = "a" * 64
    try:
        runner._consume_attempt(card_sha)
        receipt_path = attempt_dir / "ATTEMPT.json"
        assert stat.S_IMODE(attempt_dir.lstat().st_mode) == 0o555
        assert stat.S_IMODE(receipt_path.lstat().st_mode) == 0o444
        receipt = json.loads(receipt_path.read_bytes())
        assert receipt["authorized_execution_card_sha256"] == card_sha
        assert receipt["automatic_rerun_allowed"] is False
        assert receipt["state"] == "R1_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY"
        with pytest.raises(RuntimeError, match="one-shot attempt receipt"):
            runner._consume_attempt(card_sha)
    finally:
        receipt_path = attempt_dir / "ATTEMPT.json"
        if receipt_path.exists():
            receipt_path.chmod(0o600)
        if attempt_dir.exists():
            attempt_dir.chmod(0o700)


def test_r1_static_flag_rejects_abbreviation_without_consuming_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    attempt_dir = tmp_path / "must-not-exist"
    monkeypatch.setattr(runner, "ATTEMPT_DIR", attempt_dir)
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--static-pref",
                "--execution-card-sha256",
                "a" * 64,
            ]
        )
    assert not attempt_dir.exists()


def test_r1_parallel_base_process_detection_uses_basename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner._base,
        "_related_process_snapshot",
        lambda: {
            "matching_rows": [
                "1 python scripts/run_cach_a4_av3_paired_reduced_scale_r1.py"
            ]
        },
    )
    runner._reject_parallel_base_execution()
    monkeypatch.setattr(
        runner._base,
        "_related_process_snapshot",
        lambda: {
            "matching_rows": [
                "2 python run_cach_a4_av3_paired_reduced_scale.py"
            ]
        },
    )
    with pytest.raises(RuntimeError, match="parallel base AV3 execution"):
        runner._reject_parallel_base_execution()


def test_r1_source_contains_only_the_registered_pairing_expression() -> None:
    tree = ast.parse(RUNNER_PATH.read_bytes(), filename=str(RUNNER_PATH))
    imported_roots: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module is not None:
            imported_roots.add(statement.module.split(".", 1)[0])
    assert "torch" not in imported_roots
    assert "sana_wam" not in imported_roots

    base_source = BASE_RUNNER_PATH.read_text(encoding="utf-8")
    observe = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef)
        and statement.name == "_observe_resources"
    )
    zip_calls = [
        ast.unparse(node)
        for node in ast.walk(observe)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "zip"
    ]
    assert zip_calls == ["zip(offsets[:-1], offsets[1:], strict=True)"]
    assert "zip(offsets, offsets[1:], strict=True)" in base_source

    lifecycle = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.ClassDef)
        and statement.name == "R1RootLifecycle"
    )
    create = next(
        statement
        for statement in lifecycle.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "create"
    )
    calls = [node for node in ast.walk(create) if isinstance(node, ast.Call)]
    absence_line = min(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "_require_paths_absent"
    )
    create_line = min(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "create"
    )
    assert absence_line < create_line
