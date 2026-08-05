from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import stat
import sys

import pytest


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale_r2.py"
BASE_SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale_r2.py"
CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale_r2.yaml"
CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_EXECUTION_R2_RUN_CARD.json"
)
MANIFEST_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_DATA_SELECTION.json"
)
BASE_CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale.yaml"
SOURCE_SHA256 = "d069c3af8a324842cdb8b9f3bdc1cc6d052c3e16e7398de5cd5290fcb8852d75"
CONFIG_SHA256 = "88ab7cfbd526a436df16a12f696fb5fb984b73c12187f1896f60998042cdb0ae"
MANIFEST_SHA256 = "fecd1a9fb2f4371011d9c3c86469fed8f24f7f6aa5b3f1be9811fd447fb2ab95"
SELECTION_SHA256 = "49a56dc087d693a4f25005095f390996dab8639eb4e387bf8c70b872dfeb50d9"


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


def _load(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source() -> object:
    return _load(SOURCE_PATH, "_cach_a4_av3_r2_test_source")


def _runner() -> object:
    return _load(RUNNER_PATH, "_cach_a4_av3_r2_test_runner")


def _window(source: object, episode: int, *, start: int = 0) -> object:
    return source._base.av2.av2_scaffold.WindowIdentity(
        task_name="adjust_bottle",
        episode_id=str(episode),
        raw_start=start,
        raw_end_exclusive=start + 33,
    )


class _NormalizeLoader(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        node = self.generic_visit(node)
        assert isinstance(node, ast.Attribute)
        if isinstance(node.value, ast.Name) and node.value.id == "_base":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        node = self.generic_visit(node)
        assert isinstance(node, ast.Call)
        is_r2 = isinstance(node.func, ast.Name) and (
            node.func.id == "assert_av3_train_holdout_disjoint"
        )
        is_base = isinstance(node.func, ast.Attribute) and (
            node.func.attr == "assert_train_holdout_disjoint"
        )
        if is_r2 or is_base:
            node.func = ast.Name(id="_REGISTERED_DISJOINT_ASSERTION", ctx=ast.Load())
        return node


def _normalized_loader(function: object) -> str:
    node = ast.parse(inspect.getsource(function)).body[0]
    assert isinstance(node, ast.FunctionDef)
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        node.body.pop(0)
    normalized = _NormalizeLoader().visit(node)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def test_r2_loader_is_frozen_loader_plus_only_registered_assertion() -> None:
    source = _source()
    assert hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() == SOURCE_SHA256
    assert _normalized_loader(source.load_av3_real_data) == _normalized_loader(
        source._base.load_av3_real_data
    )
    assert source.AV3RealDataBundle is source._base.AV3RealDataBundle
    assert source.compute_av3_metrics is source._base.compute_av3_metrics
    assert source.classify_av3 is source._base.classify_av3


def test_r2_native_16_by_16_validator_preserves_frozen_overlap_semantics() -> None:
    source = _source()
    train = tuple(_window(source, episode) for episode in range(16, 32))
    heldout = tuple(_window(source, episode) for episode in range(32, 48))
    source.assert_av3_train_holdout_disjoint(train, heldout)
    with pytest.raises(Exception, match="exactly 8 train and 8 held-out"):
        source._base.av2.av2_scaffold.assert_train_holdout_disjoint(train, heldout)
    with pytest.raises(source.AV3ExecutionError, match="exactly 16 train"):
        source.assert_av3_train_holdout_disjoint(train[:-1], heldout)
    with pytest.raises(source.AV3ExecutionError, match="duplicate"):
        source.assert_av3_train_holdout_disjoint(train[:-1] + train[:1], heldout)
    overlapping = heldout[:-1] + (_window(source, 16, start=1),)
    with pytest.raises(Exception, match="raw intervals overlap"):
        source.assert_av3_train_holdout_disjoint(train, overlapping)


def test_r2_no_preload_validates_config_before_data_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    events: list[str] = []
    bundle = object()
    monkeypatch.setattr(
        source._base,
        "validate_av3_config",
        lambda _config: events.append("validate"),
    )
    monkeypatch.setattr(
        source._base,
        "_config_mapping",
        lambda _config, _name: {"dataset_root": "/unused", "manifest_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        source,
        "load_av3_real_data",
        lambda *_args, **_kwargs: events.append("data") or bundle,
    )
    monkeypatch.setattr(
        source._base,
        "run_av3_paired_reduced_scale",
        lambda *_args, **kwargs: events.append("delegate")
        or {"same_bundle": kwargs["preloaded_bundle"] is bundle},
    )
    result = source.run_av3_paired_reduced_scale(
        {},
        expected_gpu_uuid="unused",
        total_deadline_monotonic=1.0,
        source_data_and_predecessor_pins_match=True,
        manifest={},
        manifest_path="/unused",
        theta0_callback=lambda _value: None,
    )
    assert events == ["validate", "data", "delegate"]
    assert result == {"same_bundle": True}


def test_r2_actual_frozen_cohort_load_and_adequacy() -> None:
    source = _source()
    manifest_payload = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_payload)
    assert manifest_payload == _canonical(manifest)
    assert hashlib.sha256(manifest_payload).hexdigest() == MANIFEST_SHA256
    bundle = source.load_av3_real_data(
        "/DATA/share/RoboTwin2.0/dataset",
        manifest,
        manifest_path=MANIFEST_PATH,
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    adequacy = source.assess_av3_data_adequacy(bundle)
    assert isinstance(bundle, source.AV3RealDataBundle)
    assert bundle.selection_sha256 == SELECTION_SHA256
    assert bundle.manifest_sha256 == MANIFEST_SHA256
    assert [[int(window.episode_id) for window in split.windows] for split in bundle.train_microbatches] == [
        list(range(16, 24)),
        list(range(24, 32)),
    ]
    assert [[int(window.episode_id) for window in split.windows] for split in bundle.heldout_microbatches] == [
        list(range(32, 40)),
        list(range(40, 48)),
    ]
    assert adequacy["all_adequate"] is True


def test_r2_overlay_and_card_bind_exact_materialization() -> None:
    overlay_payload = CONFIG_PATH.read_bytes()
    overlay = json.loads(overlay_payload)
    assert overlay_payload == _canonical(overlay)
    assert hashlib.sha256(overlay_payload).hexdigest() == CONFIG_SHA256
    assert overlay["authority"]["statement"] == "授权 AV3-R2"
    assert overlay["execution_override"]["automatic_rerun"] is False
    assert overlay["execution_override"]["root_reuse"] is False

    card_payload = CARD_PATH.read_bytes()
    card = json.loads(card_payload)
    assert card_payload == _canonical(card)
    assert card["predecessor_revision"]["r1_terminal_record"]["record"]["phase"] == (
        "cpu_data_load_and_adequacy"
    )
    terminal = card["predecessor_revision"]["r1_terminal_record"]
    assert hashlib.sha256(_canonical(terminal["record"])).hexdigest() == (
        terminal["canonical_json_plus_lf_sha256"]
    )
    for role, path in {
        "config": CONFIG_PATH,
        "runner": RUNNER_PATH,
        "source": SOURCE_PATH,
        "test": Path(__file__),
    }.items():
        assert card["materialized_pins"][role] == {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    assert set(card["inherited_pins"]) == {
        "base_config",
        "base_source",
        "data_manifest",
        "r1_attempt_receipt",
        "r1_config",
        "r1_execution_card",
        "r1_runner",
        "r1_test",
    }


def test_r2_runner_is_cpu_static_at_import_and_lifecycle_rechecks() -> None:
    tree = ast.parse(RUNNER_PATH.read_bytes(), filename=str(RUNNER_PATH))
    imported_roots: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module is not None:
            imported_roots.add(statement.module.split(".", 1)[0])
    assert "torch" not in imported_roots
    assert "sana_wam" not in imported_roots

    lifecycle = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "R2RootLifecycle"
    )
    create = next(
        node
        for node in lifecycle.body
        if isinstance(node, ast.FunctionDef) and node.name == "create"
    )
    call_lines: dict[str, int] = {}
    for node in ast.walk(create):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_lines[node.func.id] = node.lineno
        elif isinstance(node.func, ast.Attribute):
            call_lines[node.func.attr] = node.lineno
    assert call_lines["_verify_predecessor_runtime_state"] < call_lines[
        "_reject_parallel_predecessor_execution"
    ] < call_lines["create"]


def test_r2_attempt_receipt_is_single_use_and_frozen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
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
        assert receipt["state"] == "R2_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY"
        with pytest.raises(RuntimeError, match="one-shot attempt receipt"):
            runner._consume_attempt(card_sha)
    finally:
        receipt_path = attempt_dir / "ATTEMPT.json"
        if receipt_path.exists():
            receipt_path.chmod(0o600)
        if attempt_dir.exists():
            attempt_dir.chmod(0o700)


def test_r2_static_abbreviation_is_rejected_without_consuming_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    attempt_dir = tmp_path / "must-not-exist"
    monkeypatch.setattr(runner, "ATTEMPT_DIR", attempt_dir)
    with pytest.raises(SystemExit):
        runner.main(["--static-pref", "--execution-card-sha256", "a" * 64])
    assert not attempt_dir.exists()
