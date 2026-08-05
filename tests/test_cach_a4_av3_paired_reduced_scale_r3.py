from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import pytest


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale_r3.py"
R2_SOURCE_PATH = REPO_ROOT / "src/sana_wam/model/cach_a4_av3_paired_reduced_scale_r2.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_cach_a4_av3_paired_reduced_scale_r3.py"
CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_a4_av3_paired_reduced_scale_r3.yaml"
CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_EXECUTION_R3_RUN_CARD.json"
)
MANIFEST_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_DATA_SELECTION.json"
)
SOURCE_SHA256 = "750c02f03cf571c5a532eb6316d912c64d886b036b11eb4fe344a68125292030"
CONFIG_SHA256 = "0f25f57b757a21161016824b329104d2b1b8f11de6f23450b8f92b5c830f9a81"
R2_SOURCE_SHA256 = "d069c3af8a324842cdb8b9f3bdc1cc6d052c3e16e7398de5cd5290fcb8852d75"
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
    return _load(SOURCE_PATH, "_cach_a4_av3_r3_test_source")


def _runner() -> object:
    return _load(RUNNER_PATH, "_cach_a4_av3_r3_test_runner")


class _FakeCuda:
    def __init__(self, events: list[str], *, fail_at: str | None = None) -> None:
        self.events = events
        self.fail_at = fail_at

    def init(self) -> None:
        self.events.append("cuda_init")
        if self.fail_at == "init":
            raise RuntimeError("fake init failure")

    def reset_peak_memory_stats(self, _device: object = None) -> None:
        self.events.append("original_reset")
        if self.fail_at == "reset":
            raise RuntimeError("fake reset failure")


class _FakeTorch:
    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda


def _install_fake_base(
    source: object,
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[str],
    fail_before_reset: bool = False,
    fail_after_reset: bool = False,
    double_reset: bool = False,
    cuda_fail_at: str | None = None,
    nested_reset: bool = False,
    skip_reset: bool = False,
) -> _FakeTorch:
    fake_torch = _FakeTorch(_FakeCuda(events, fail_at=cuda_fail_at))

    def fake_base_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("delegate_enter")
        if fail_before_reset:
            raise RuntimeError("fake pre-reset rejection")
        if nested_reset:
            def nested() -> None:
                source._r2._base.torch.cuda.reset_peak_memory_stats("cuda:0")

            nested()
        elif not skip_reset:
            source._r2._base.torch.cuda.reset_peak_memory_stats("cuda:0")
        if double_reset:
            source._r2._base.torch.cuda.reset_peak_memory_stats("cuda:0")
        if fail_after_reset:
            raise RuntimeError("fake post-reset failure")
        events.append("delegate_return")
        return {"fake": True}

    monkeypatch.setattr(source, "_REAL_TORCH", fake_torch)
    monkeypatch.setattr(source._r2._base, "torch", fake_torch)
    monkeypatch.setattr(
        source._r2._base,
        "run_av3_paired_reduced_scale",
        fake_base_run,
    )
    return fake_torch


def _invoke_source(source: object) -> object:
    return source.run_av3_paired_reduced_scale(
        {},
        expected_gpu_uuid="unused",
        total_deadline_monotonic=1.0,
        source_data_and_predecessor_pins_match=True,
        manifest={},
        manifest_path="/unused",
        theta0_callback=lambda _value: None,
        preloaded_bundle=object(),
    )


def test_r3_scoped_proxy_orders_init_then_original_reset_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    events: list[str] = []
    global_torch = sys.modules["torch"]
    global_reset = global_torch.cuda.reset_peak_memory_stats
    fake_torch = _install_fake_base(source, monkeypatch, events=events)
    result = _invoke_source(source)
    assert result == {"fake": True}
    assert events == [
        "delegate_enter",
        "cuda_init",
        "original_reset",
        "delegate_return",
    ]
    assert source._r2._base.torch is fake_torch
    assert sys.modules["torch"] is global_torch
    assert global_torch.cuda.reset_peak_memory_stats is global_reset


@pytest.mark.parametrize(
    ("fail_before_reset", "cuda_fail_at", "message", "events"),
    [
        (True, None, "pre-reset", ["delegate_enter"]),
        (False, "init", "init failure", ["delegate_enter", "cuda_init"]),
        (
            False,
            "reset",
            "reset failure",
            ["delegate_enter", "cuda_init", "original_reset"],
        ),
    ],
)
def test_r3_scoped_proxy_restores_on_every_failure_boundary(
    monkeypatch: pytest.MonkeyPatch,
    fail_before_reset: bool,
    cuda_fail_at: str | None,
    message: str,
    events: list[str],
) -> None:
    source = _source()
    observed: list[str] = []
    fake_torch = _install_fake_base(
        source,
        monkeypatch,
        events=observed,
        fail_before_reset=fail_before_reset,
        cuda_fail_at=cuda_fail_at,
    )
    with pytest.raises(RuntimeError, match=message):
        _invoke_source(source)
    assert observed == events
    assert source._r2._base.torch is fake_torch


def test_r3_scoped_proxy_rejects_second_reset_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    events: list[str] = []
    fake_torch = _install_fake_base(
        source,
        monkeypatch,
        events=events,
        double_reset=True,
    )
    with pytest.raises(source.R3SourceContractError, match="more than once"):
        _invoke_source(source)
    assert events == ["delegate_enter", "cuda_init", "original_reset"]
    assert source._r2._base.torch is fake_torch


def test_r3_scoped_proxy_rejects_unregistered_nested_caller_before_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    events: list[str] = []
    fake_torch = _install_fake_base(
        source,
        monkeypatch,
        events=events,
        nested_reset=True,
    )
    with pytest.raises(source.R3SourceContractError, match="unregistered caller"):
        _invoke_source(source)
    assert events == ["delegate_enter"]
    assert source._r2._base.torch is fake_torch


def test_r3_scoped_proxy_restores_after_post_reset_delegate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    events: list[str] = []
    fake_torch = _install_fake_base(
        source,
        monkeypatch,
        events=events,
        fail_after_reset=True,
    )
    with pytest.raises(RuntimeError, match="post-reset failure"):
        _invoke_source(source)
    assert events == ["delegate_enter", "cuda_init", "original_reset"]
    assert source._r2._base.torch is fake_torch


def test_r3_scoped_proxy_requires_registered_reset_on_normal_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    events: list[str] = []
    fake_torch = _install_fake_base(
        source,
        monkeypatch,
        events=events,
        skip_reset=True,
    )
    with pytest.raises(source.R3SourceContractError, match="did not cross"):
        _invoke_source(source)
    assert events == ["delegate_enter", "delegate_return"]
    assert source._r2._base.torch is fake_torch


def test_r3_source_is_thin_and_preserves_r2_scientific_surface() -> None:
    source = _source()
    assert hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() == SOURCE_SHA256
    assert hashlib.sha256(R2_SOURCE_PATH.read_bytes()).hexdigest() == R2_SOURCE_SHA256
    assert source.AV3RealDataBundle is source._r2.AV3RealDataBundle
    assert source.load_av3_real_data is source._r2.load_av3_real_data
    assert source.assess_av3_data_adequacy is source._r2.assess_av3_data_adequacy
    assert source.compute_av3_metrics is source._r2.compute_av3_metrics
    assert source.classify_av3 is source._r2.classify_av3

    tree = ast.parse(SOURCE_PATH.read_bytes(), filename=str(SOURCE_PATH))
    imported_roots: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module is not None:
            imported_roots.add(statement.module.split(".", 1)[0])
    assert "torch" not in imported_roots
    assert "sana_wam" not in imported_roots

    reset_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "reset_peak_memory_stats"
    )
    calls = [ast.unparse(node) for node in ast.walk(reset_method) if isinstance(node, ast.Call)]
    init_index = next(i for i, call in enumerate(calls) if call == "self._real_cuda.init()")
    reset_index = next(
        i
        for i, call in enumerate(calls)
        if call.startswith("self._real_cuda.reset_peak_memory_stats(")
    )
    assert init_index < reset_index

    run_method = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_av3_paired_reduced_scale"
    )
    guarded_try = next(node for node in ast.walk(run_method) if isinstance(node, ast.Try))
    assignment = next(
        node
        for node in guarded_try.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "torch"
            for target in node.targets
        )
    )
    assert assignment.lineno < guarded_try.finalbody[0].lineno

    base_run = ast.parse(
        R2_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(R2_SOURCE_PATH)
    )
    assert sum(
        1
        for node in ast.walk(base_run)
        if isinstance(node, ast.Attribute) and node.attr == "reset_peak_memory_stats"
    ) == 0
    frozen_base = source._r2._base
    frozen_tree = ast.parse(
        Path(frozen_base.__file__).read_text(encoding="utf-8"),
        filename=str(frozen_base.__file__),
    )
    run_node = next(
        node
        for node in frozen_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_av3_paired_reduced_scale"
    )
    assert sum(
        1
        for node in ast.walk(run_node)
        if isinstance(node, ast.Attribute) and node.attr == "reset_peak_memory_stats"
    ) == 1


def test_r3_actual_frozen_cohort_loader_is_unchanged() -> None:
    source = _source()
    payload = MANIFEST_PATH.read_bytes()
    manifest = json.loads(payload)
    assert payload == _canonical(manifest)
    assert hashlib.sha256(payload).hexdigest() == MANIFEST_SHA256
    bundle = source.load_av3_real_data(
        "/DATA/share/RoboTwin2.0/dataset",
        manifest,
        manifest_path=MANIFEST_PATH,
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    adequacy = source.assess_av3_data_adequacy(bundle)
    assert bundle.selection_sha256 == SELECTION_SHA256
    assert bundle.manifest_sha256 == MANIFEST_SHA256
    assert adequacy["all_adequate"] is True


def test_r3_overlay_and_card_bind_exact_materialization() -> None:
    overlay_payload = CONFIG_PATH.read_bytes()
    overlay = json.loads(overlay_payload)
    assert overlay_payload == _canonical(overlay)
    assert hashlib.sha256(overlay_payload).hexdigest() == CONFIG_SHA256
    assert overlay["authority"]["statement"] == "授权 AV3-R3"
    assert overlay["fix_contract"]["only_semantic_delta"] == (
        "CUDA_ALLOCATOR_INITIALIZATION_BEFORE_PEAK_MEMORY_RESET"
    )

    card_payload = CARD_PATH.read_bytes()
    card = json.loads(card_payload)
    assert card_payload == _canonical(card)
    terminal = card["predecessor_revision"]["r2_terminal_evidence"]
    assert terminal["result"]["expected_content"]["phase"] == "model_execution"
    assert terminal["result"]["expected_content"]["valid_result"] is False
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


def test_r3_runner_revalidates_r2_frozen_failure_and_original_lifecycle() -> None:
    runner = _runner()
    runner._verify_r2_frozen_failure()
    assert runner._BaseRootLifecycle is runner._r2._BaseRootLifecycle
    assert issubclass(runner.R3RootLifecycle, runner._r2._BaseRootLifecycle)
    assert runner._base.RootLifecycle is runner.R3RootLifecycle

    tree = ast.parse(RUNNER_PATH.read_bytes(), filename=str(RUNNER_PATH))
    imported_roots: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module is not None:
            imported_roots.add(statement.module.split(".", 1)[0])
    assert "torch" not in imported_roots
    assert "sana_wam" not in imported_roots


def test_r3_attempt_receipt_is_single_use_and_frozen(
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
        assert receipt["state"] == "R3_EXECUTION_ATTEMPT_CONSUMED_PRE_CAPABILITY"
        with pytest.raises(RuntimeError, match="one-shot attempt receipt"):
            runner._consume_attempt(card_sha)
    finally:
        receipt_path = attempt_dir / "ATTEMPT.json"
        if receipt_path.exists():
            receipt_path.chmod(0o600)
        if attempt_dir.exists():
            attempt_dir.chmod(0o700)


def test_r3_static_abbreviation_is_rejected_without_consuming_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    attempt_dir = tmp_path / "must-not-exist"
    monkeypatch.setattr(runner, "ATTEMPT_DIR", attempt_dir)
    with pytest.raises(SystemExit):
        runner.main(["--static-pref", "--execution-card-sha256", "a" * 64])
    assert not attempt_dir.exists()
