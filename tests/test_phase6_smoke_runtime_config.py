from __future__ import annotations

import importlib.util
import json
import sys
import types
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from sana_wam.train.phase6_arm_config import (
    ARM_NAMES,
    ARM_PROJECTION_MANIFEST_SCHEMA_VERSION,
    SMOKE_RUNTIME_ARM_SCHEMA_VERSION,
    SMOKE_RUNTIME_BUNDLE_SCHEMA_VERSION,
    Phase6ArmConfigError,
    canonical_json_bytes,
    validate_phase6_smoke_runtime_bundle_bytes,
    write_phase6_arm_projection_bundle,
    write_phase6_smoke_runtime_bundle,
)
from sana_wam.train.phase6_recovery import recovery_lifecycle

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "configs" / "phase6" / "templates"


def _bundle(tmp_path):
    projection_manifest = tmp_path / "projection.json"
    projection_paths = {arm: tmp_path / f"projection_{arm}.json" for arm in ARM_NAMES}
    write_phase6_arm_projection_bundle(projection_manifest, projection_paths)
    runtime_paths = {arm: tmp_path / f"runtime_{arm}.json" for arm in ARM_NAMES}
    bundle_path = tmp_path / "smoke_runtime_bundle.json"
    digests = write_phase6_smoke_runtime_bundle(
        bundle_path=bundle_path,
        projection_manifest_path=str(projection_manifest.resolve()),
        projection_manifest_sha256=sha256(projection_manifest.read_bytes()).hexdigest(),
        template_dir=TEMPLATE_DIR,
        runtime_config_paths=runtime_paths,
    )
    return projection_manifest, runtime_paths, bundle_path, digests


def test_smoke_runtime_bundle_is_full_acyclic_five_arm_contract(tmp_path):
    projection, runtime_paths, bundle_path, digests = _bundle(tmp_path)
    value = validate_phase6_smoke_runtime_bundle_bytes(
        bundle_path.read_bytes(),
        expected_artifact_sha256=digests["bundle"],
        expected_projection_manifest_sha256=sha256(projection.read_bytes()).hexdigest(),
    )
    assert value["schema_version"] == SMOKE_RUNTIME_BUNDLE_SCHEMA_VERSION
    assert json.loads(projection.read_bytes())["schema_version"] == (
        ARM_PROJECTION_MANIFEST_SCHEMA_VERSION
    )
    assert {key: value[key] for key in recovery_lifecycle()} == recovery_lifecycle()
    assert value["registered_before_recovery_cohort_started"] is True
    assert [entry["arm"] for entry in value["arms"]] == list(ARM_NAMES)
    for entry in value["arms"]:
        runtime = json.loads(runtime_paths[entry["arm"]].read_text(encoding="utf-8"))
        assert runtime["schema_version"] == SMOKE_RUNTIME_ARM_SCHEMA_VERSION
        assert set(runtime) == {
            "arm",
            "construction_contract",
            "factors",
            "input_roles",
            "projection",
            "projection_sha256",
            *recovery_lifecycle(),
            "schema_version",
        }
        assert {
            key: runtime[key] for key in recovery_lifecycle()
        } == recovery_lifecycle()
        construction = runtime["construction_contract"]
        assert construction["optimizer_creation_allowed"] is False
        assert construction["optimizer_step_allowed"] is False
        assert construction["closed_loop_allowed"] is False
        assert "model" in runtime["projection"]
        assert "dataloader" in runtime["projection"]
        assert "training" in runtime["projection"]
        text = runtime_paths[entry["arm"]].read_text(encoding="utf-8").lower()
        for forbidden in (
            "preflight_request",
            "preflight_report",
            "smoke_gate",
            "launch_manifest",
        ):
            assert forbidden not in text


def test_smoke_runtime_bundle_rejects_projection_or_arm_drift(tmp_path):
    projection, _runtime_paths, bundle_path, _digests = _bundle(tmp_path)
    value = json.loads(bundle_path.read_text(encoding="utf-8"))
    changed = deepcopy(value)
    changed["arms"][0]["projection_sha256"] = "e" * 64
    data = canonical_json_bytes(changed)
    with pytest.raises(Phase6ArmConfigError, match="projection differs"):
        validate_phase6_smoke_runtime_bundle_bytes(
            data,
            expected_artifact_sha256=sha256(data).hexdigest(),
            expected_projection_manifest_sha256=sha256(
                projection.read_bytes()
            ).hexdigest(),
        )
    with pytest.raises(Phase6ArmConfigError, match="file SHA256 mismatch"):
        validate_phase6_smoke_runtime_bundle_bytes(
            bundle_path.read_bytes(),
            expected_artifact_sha256="e" * 64,
            expected_projection_manifest_sha256=sha256(
                projection.read_bytes()
            ).hexdigest(),
        )


@pytest.mark.parametrize(
    ("key", "alias"),
    (
        ("historical_checkpoint_count", False),
        ("historical_optimizer_step_calls_per_arm", True),
        ("historical_scheduled_lr_scale_at_step0", 0),
    ),
)
def test_smoke_runtime_bundle_rejects_lifecycle_type_aliases(tmp_path, key, alias):
    projection, _runtime_paths, bundle_path, _digests = _bundle(tmp_path)
    value = json.loads(bundle_path.read_text(encoding="utf-8"))
    value[key] = alias
    data = canonical_json_bytes(value)
    with pytest.raises(Phase6ArmConfigError, match=key):
        validate_phase6_smoke_runtime_bundle_bytes(
            data,
            expected_artifact_sha256=sha256(data).hexdigest(),
            expected_projection_manifest_sha256=sha256(
                projection.read_bytes()
            ).hexdigest(),
        )


def test_smoke_runtime_publication_is_exclusive(tmp_path):
    projection, runtime_paths, bundle_path, _digests = _bundle(tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_phase6_smoke_runtime_bundle(
            bundle_path=bundle_path,
            projection_manifest_path=str(projection.resolve()),
            projection_manifest_sha256=sha256(projection.read_bytes()).hexdigest(),
            template_dir=TEMPLATE_DIR,
            runtime_config_paths=runtime_paths,
        )


def test_smoke_runtime_generator_rejects_base_template_drift(tmp_path):
    templates = tmp_path / "templates"
    templates.mkdir()
    for source in TEMPLATE_DIR.glob("*.yaml.in"):
        (templates / source.name).write_bytes(source.read_bytes())
    target = templates / "train_phase6_T0_E0A0.yaml.in"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            "action_dim: 20", "action_dim: 19", 1
        ),
        encoding="utf-8",
    )
    projection = tmp_path / "projection.json"
    write_phase6_arm_projection_bundle(
        projection, {arm: tmp_path / f"projection_{arm}.json" for arm in ARM_NAMES}
    )
    with pytest.raises(Phase6ArmConfigError, match="template model differs"):
        write_phase6_smoke_runtime_bundle(
            bundle_path=tmp_path / "bundle.json",
            projection_manifest_path=str(projection.resolve()),
            projection_manifest_sha256=sha256(projection.read_bytes()).hexdigest(),
            template_dir=templates,
            runtime_config_paths={
                arm: tmp_path / f"runtime_{arm}.json" for arm in ARM_NAMES
            },
        )


def test_producer_bytes_are_accepted_by_production_runtime_loader(
    tmp_path, monkeypatch
):
    projection, _runtime_paths, bundle_path, digests = _bundle(tmp_path)
    sidecar = json.loads(bundle_path.read_text(encoding="utf-8"))
    fake_torch = types.ModuleType("torch")
    fake_gate = types.ModuleType("sana_wam.train.phase6_smoke_gate")
    fake_gate.RUNTIME_SUPPORT_API_VERSION = sidecar["runtime_api_version"]
    fake_gate.SMOKE_AUTHORIZATION = sidecar["authorization"]
    fake_gate.canonical_json_bytes = canonical_json_bytes
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sana_wam.train.phase6_smoke_gate", fake_gate)

    runtime_path = ROOT / "src/sana_wam/train/phase6_smoke_runtime.py"
    if not runtime_path.is_file():
        runtime_path = (
            ROOT.parent
            / "smoke_gate_candidate"
            / "src/sana_wam/train/phase6_smoke_runtime.py"
        )
    spec = importlib.util.spec_from_file_location(
        "phase6_smoke_runtime_contract_probe", runtime_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SIDECAR_SCHEMA_VERSION == SMOKE_RUNTIME_BUNDLE_SCHEMA_VERSION
    assert module.ARM_RUNTIME_CONFIG_SCHEMA_VERSION == SMOKE_RUNTIME_ARM_SCHEMA_VERSION
    loaded = module._load_runtime_configs(
        {
            "artifacts": {
                "arm_projection_manifest": {
                    "path": str(projection.resolve()),
                    "sha256": sha256(projection.read_bytes()).hexdigest(),
                }
            },
            "input_config": {
                "path": str(bundle_path.resolve()),
                "sha256": digests["bundle"],
            },
        }
    )
    assert tuple(loaded) == ARM_NAMES
    assert all(loaded[arm]["projection"]["arm"] == arm for arm in ARM_NAMES)

    arm = ARM_NAMES[0]
    legacy = deepcopy(loaded[arm])
    legacy["formal_training_started"] = False
    legacy["closed_loop_started"] = False
    for key in recovery_lifecycle():
        legacy.pop(key)
    with pytest.raises(
        module.Phase6SmokeRuntimeError, match="runtime config keys differ"
    ):
        module._validate_arm_config(
            legacy,
            arm=arm,
            projection_entry={
                "projection": loaded[arm]["projection"],
                "projection_sha256": loaded[arm]["projection_sha256"],
            },
        )


def test_smoke_trainer_mixin_uses_smoke_authority_without_training_context():
    from sana_wam.train import phase6_smoke_runtime as runtime
    from sana_wam.train.trainer import Trainer

    report = {
        "authorization": runtime.SMOKE_AUTHORIZATION,
        **recovery_lifecycle(),
        "purpose": "smoke",
        "reference_precompute_completed": True,
        "registered_before_recovery_cohort_started": True,
        "smoke_gpu_started": False,
        "status": "pass",
    }

    class _ReachedWorldSizeGate(Exception):
        pass

    class _SmokeProbe(runtime._SmokeTrainerMixin, Trainer):
        def _validate_world_size_contract(self):
            raise _ReachedWorldSizeGate

    cfg = types.SimpleNamespace(training={})
    with pytest.raises(_ReachedWorldSizeGate):
        _SmokeProbe(cfg, smoke_preflight_report=report)

    invalid_reports = (
        (dict(report, purpose="training"), "passing smoke report"),
        (dict(report, status="fail"), "passing smoke report"),
        (dict(report, authorization={}), "smoke authorization differs"),
    )
    for invalid_report, message in invalid_reports:
        with pytest.raises(runtime.Phase6SmokeRuntimeError, match=message):
            _SmokeProbe(cfg, smoke_preflight_report=invalid_report)

    for bad_lifecycle in (
        dict(report, smoke_gpu_started=True),
        dict(report, historical_formal_training_started=False),
        dict(report, recovery_cohort_started=True),
        dict(report, registered_before_recovery_cohort_started=False),
        dict(report, historical_checkpoint_count=False),
        dict(report, historical_optimizer_step_calls_per_arm=True),
        dict(report, historical_scheduled_lr_scale_at_step0=0),
    ):
        with pytest.raises(
            ValueError,
            match="smoke preflight lifecycle differs",
        ):
            _SmokeProbe(cfg, smoke_preflight_report=bad_lifecycle)

    probe = object.__new__(_SmokeProbe)
    probe._phase6_launch_context = object()
    with pytest.raises(
        ValueError,
        match="forbids a formal-training launch_context",
    ):
        probe._validate_phase6_launch_context_early()

    probe._phase6_launch_context = None
    probe._authorized_smoke_report = report
    probe._phase6_preflight_report = report
    probe._phase6_preflight_report_bytes = b"changed"
    probe._phase6_preflight_report_sha256 = "0" * 64
    with pytest.raises(ValueError, match="authority changed during construction"):
        probe._validate_phase6_launch_context_early()

    formal = object.__new__(Trainer)
    formal._phase6_preflight_report = {"purpose": "training"}
    formal._phase6_launch_context = None
    with pytest.raises(
        ValueError,
        match="formal Phase-6 requires a validated launch_context",
    ):
        Trainer._validate_phase6_launch_context_early(formal)


@pytest.mark.parametrize(
    ("key", "alias"),
    (
        ("historical_checkpoint_count", False),
        ("historical_optimizer_step_calls_per_arm", True),
        ("historical_scheduled_lr_scale_at_step0", 0),
    ),
)
def test_formal_trainer_rejects_recovery_lifecycle_type_aliases(key, alias):
    from sana_wam.train.trainer import Trainer

    authorization = {"operation": "synthetic-training"}
    context = types.SimpleNamespace(
        authorization=authorization,
        revalidate_pinned_files=lambda: None,
    )
    trainer = object.__new__(Trainer)
    trainer._phase6_launch_context = context
    trainer._phase6_preflight_report = {
        "authorization": authorization,
        **recovery_lifecycle(),
        "reference_precompute_completed": True,
        "registered_before_recovery_cohort_started": True,
        "smoke_completed": True,
    }
    trainer._phase6_preflight_report[key] = alias
    with pytest.raises(RuntimeError, match=key):
        trainer._validate_phase6_launch_context_early()


def test_source_repair_amendment_is_stable_hashed(tmp_path, monkeypatch):
    from sana_wam.train import phase6_smoke_runtime as runtime

    path = tmp_path / "source_repair_amendment.json"
    value = {
        "active_chain_backlink": {
            "required_amendment_path_constant": str(path),
            "required_runtime_factory_stable_hash_check": True,
        },
        "schema_version": "sana-phase6-source-repair-amendment-v2",
    }
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    monkeypatch.setattr(runtime, "SOURCE_REPAIR_AMENDMENT_PATH", path)
    monkeypatch.setattr(
        runtime,
        "SOURCE_REPAIR_AMENDMENT_SHA256",
        sha256(payload).hexdigest(),
    )
    runtime._validate_source_repair_amendment()

    path.write_bytes(canonical_json_bytes(dict(value, changed=True)))
    with pytest.raises(runtime.Phase6SmokeRuntimeError, match="SHA256 differs"):
        runtime._validate_source_repair_amendment()
