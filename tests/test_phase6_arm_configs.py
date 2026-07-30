from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from sana_wam.train import phase6_arm_config as arm_config
from sana_wam.train.phase6_arm_config import (
    ADAPTER_PATTERN,
    ARM_FACTORS,
    ARM_NAMES,
    HOLDOUT_TASKS,
    TRAIN_TASKS,
    VIDEO_PATTERNS,
    Phase6ArmConfigError,
    canonical_json_bytes,
    load_arm_config,
    load_phase6_materialization_pins,
    materialize_arm_configs,
    validate_arm_config,
    validate_arm_set,
    write_phase6_arm_projection_bundle,
)
from sana_wam.train.phase6_downstream_pins import FIXED_AMENDMENT_CONFIG_PINS

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "configs" / "phase6" / "templates"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _synthetic_recovery_authority(tmp_path, monkeypatch):
    authority = tmp_path / "recovery_authority.json"
    authority.write_bytes(canonical_json_bytes({"status": "synthetic-test"}))
    monkeypatch.setattr(
        arm_config, "PLANROW_RECOVERY_AUTHORITY_PATH", str(authority.resolve())
    )
    monkeypatch.setattr(
        arm_config, "PLANROW_RECOVERY_AUTHORITY_SHA256", _digest(authority)
    )
    monkeypatch.setattr(
        arm_config,
        "validate_phase6_planrow_recovery_authority",
        lambda **_kwargs: {"status": "synthetic-test"},
    )


def _materialize(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    projection_manifest = tmp_path / "projection_manifest.json"
    projection_paths = {arm: tmp_path / f"projection_{arm}.json" for arm in ARM_NAMES}
    write_phase6_arm_projection_bundle(projection_manifest, projection_paths)
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
        "smoke_runtime_config",
        "smoke_preflight_request",
        "smoke_preflight_report",
        "smoke_rebind_receipt",
        "evidence_rebinding_receipt",
        "real_2b_smoke_gate",
    ):
        path = tmp_path / f"{role}.json"
        path.write_bytes(f"frozen {role}\n".encode())
        common[role] = {"path": str(path.resolve()), "sha256": _digest(path)}
    common["arm_projection_manifest"] = {
        "path": str(projection_manifest.resolve()),
        "sha256": _digest(projection_manifest),
    }
    equivalence = tmp_path / "projection_equivalence_receipt.json"
    equivalence.write_bytes(
        canonical_json_bytes(arm_config.build_phase6_projection_equivalence_receipt())
    )
    common["projection_equivalence_receipt"] = {
        "path": str(equivalence.resolve()),
        "sha256": _digest(equivalence),
    }
    authority = Path(arm_config.PLANROW_RECOVERY_AUTHORITY_PATH)
    common["recovery_authority"] = {
        "path": str(authority),
        "sha256": _digest(authority),
    }
    arm_pins = {}
    for arm in ARM_NAMES:
        request = tmp_path / f"training_request_{arm}.json"
        report = tmp_path / f"training_report_{arm}.json"
        request.write_bytes(f"request {arm}\n".encode())
        report.write_bytes(f"report {arm}\n".encode())
        arm_pins[arm] = {
            "scientific_projection": {
                "path": str(projection_paths[arm].resolve()),
                "sha256": _digest(projection_paths[arm]),
            },
            "training_preflight_request": {
                "path": str(request.resolve()),
                "sha256": _digest(request),
            },
            "training_preflight_report": {
                "path": str(report.resolve()),
                "sha256": _digest(report),
            },
        }
    pins = {
        "arms": arm_pins,
        "common": common,
        **arm_config.recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "schema_version": arm_config.ARM_MATERIALIZATION_PINS_SCHEMA_VERSION,
    }
    output = tmp_path / "configs"
    paths = materialize_arm_configs(
        template_dir=TEMPLATE_DIR,
        output_dir=output,
        materialization_pins=pins,
    )
    configs = [load_arm_config(path) for path in paths]
    return paths, configs, pins


def test_materialized_five_arm_matrix_is_exact_and_contains_no_tricks(tmp_path):
    paths, configs, pins = _materialize(tmp_path)
    validate_arm_set(configs)
    by_arm = {config["training"]["phase6_arm"]: config for config in configs}

    assert tuple(by_arm) == ARM_NAMES
    assert len(paths) == 5
    for arm, (continuous, expansion, adapter) in ARM_FACTORS.items():
        config = by_arm[arm]
        architecture = config["model"]["architecture"]
        training = config["training"]
        assert (
            config["model"]["video_backbone"]["continuous_timestep_conditioning"]
            is continuous
        )
        assert architecture["video_local_expansion_weight"] == float(expansion)
        assert architecture["action_non_regression_weight"] == float(adapter)
        assert architecture["action_video_memory_adapter"]["enabled"] is adapter
        expected_patterns = [*VIDEO_PATTERNS]
        if adapter:
            expected_patterns.append(ADAPTER_PATTERN)
        assert training["trainable_parameter_patterns"] == expected_patterns
        assert not any(
            pattern.startswith("action_backbone") for pattern in expected_patterns
        )
        if adapter:
            assert training["action_memory_lr"] == 1.0e-4
            assert training["phase1_action_reference_artifact"] == str(
                Path(pins["common"]["action_reference"]["path"])
            )
            assert training["phase1_action_reference_live_spot_check_required"] is True
            assert (
                training["phase1_action_reference_checkpoint_sha256"]
                == training["phase1_reference_checkpoint_sha256"]
            )
        else:
            assert not any(
                key.startswith("phase1_action_reference") for key in training
            )
            assert "action_memory_lr" not in training
        assert training["phase6_dataset_contract_artifact"] == str(
            Path(pins["common"]["dataset_contract"]["path"])
        )
        assert training["phase6_code_source_manifest"] == str(
            Path(pins["common"]["source_manifest"]["path"])
        )
        assert training["phase1_reference_checkpoint_sha256"] == (
            "ba7bb59fe7e44f6efd6a4cef79f66aff7ccf5dbe82a5a166634e35055e94f089"
        )
        assert training["max_steps"] == training["lr_schedule_steps"] == 504
        assert training["expected_world_size"] == 1
        assert training["save_steps"] == 0
        assert training["save_at_steps"] == [504]
        assert training["lambda_action"] == 0.0

    forbidden = (
        "recovery_annotations",
        "dagger",
        "critic",
        "rerank",
        "best_of_n",
        "closed_loop",
    )
    combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    assert all(token not in combined for token in forbidden)


def test_dataset_cohorts_and_source_contract_are_frozen(tmp_path):
    _, configs, _ = _materialize(tmp_path)
    dataloader = configs[0]["dataloader"]
    assert tuple(dataloader["train_tasks"]) == TRAIN_TASKS
    assert tuple(dataloader["holdout_tasks"]) == HOLDOUT_TASKS
    assert sha256(
        "".join(f"{task}\n" for task in sorted(TRAIN_TASKS)).encode()
    ).hexdigest() == (
        "68293563c113d91e4878844e7cb5ef1b70ad14656da0a7c58a54a6069cc1e67d"
    )
    assert sha256(
        "".join(f"{task}\n" for task in sorted(HOLDOUT_TASKS)).encode()
    ).hexdigest() == (
        "689f9d1509e30fc446b995edf72d4d122a78b4f0f708efece2d63791c5362f6e"
    )
    assert dataloader["type"] == "robotwin"
    assert dataloader["variant"] == "clean_50"
    assert dataloader["filter_static_segments"] is False
    assert dataloader["text_embedding_cache_dir"] is None
    assert dataloader["vae_cache_dir"] is None
    assert dataloader["num_frames"] == 113
    assert dataloader["val_ratio"] == 0.0
    assert dataloader["seed"] == 42
    assert configs[0]["training"]["seed"] == 20260724


def test_registry_protocol_plan_and_external_pins_are_literal(tmp_path):
    _, configs, _ = _materialize(tmp_path)
    training = configs[0]["training"]
    assert training["phase6_protocol_document_sha256"] == (
        "765fcc884cf4cb168fac676de55801667b561fde2cc0cf98a82007ef5469d879"
    )
    assert training["phase6_registry_sha256"] == (
        "9166e1ecde74f14d7d69ae2f5072b10732b01b9f3b0004c438d0e97ed94b00b1"
    )
    assert training["phase6_operational_storage_amendment_sha256"] == (
        "e08dd7ef33f381a34808cc166e5eae90d7ec20fd9956b481a7bf984b5a8e5c8f"
    )
    assert training["phase6_external_components_manifest_sha256"] == (
        "6f886e00843527ec1933e2f4ebd03dd1e7bf0069e5ce0892bd834b893a6a1b2a"
    )
    assert training["phase6_plan_artifact_sha256"] == (
        "7a1063df1d97fa0b8859dc86bf22f1ba3ad23703b6f6e449388dc950b8dc3ce6"
    )
    assert training["phase6_plan_sha256"] == (
        "701d1436c9804df960d190586c264b80af42e8ce107751693e4c5adbe1089411"
    )
    assert training["phase6_identity_sha256"] == (
        "08b9fcf418b0c4aabf7ea5e494cc8f603e63b797658c2c3890c2a06599a965dd"
    )
    for key, expected in FIXED_AMENDMENT_CONFIG_PINS.items():
        assert training[key] == expected
    assert training["init_checkpoint_sha256"] == (
        "e9549aff484da56eada21972f00fb3aa2f95f8a65397a55cf0f38c79e6254137"
    )
    assert training["action_stats_sha256"] == (
        "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"
    )
    assert training["phase1_reference_checkpoint"] == (
        "/home/zch/workspace/sana-wam/logs/sana_principles_audit_20260722/"
        "phase1_weights_chunkwise_arch_run/checkpoint_step_0.safetensors"
    )
    assert training["phase1_reference_checkpoint_sha256"] == (
        "ba7bb59fe7e44f6efd6a4cef79f66aff7ccf5dbe82a5a166634e35055e94f089"
    )


def test_validator_rejects_schedule_data_factor_and_trainable_drift(tmp_path):
    _, configs, _ = _materialize(tmp_path)
    base = configs[1]  # T1_E0A0

    mutations = []
    for path, value in (
        (("training", "max_steps"), 503),
        (("training", "lr_schedule_steps"), 505),
        (("training", "expected_world_size"), 2),
        (("training", "lambda_action"), 1.0),
        (("training", "save_at_steps"), [100, 504]),
        (("training", "debug"), True),
        (("dataloader", "type"), "mixture"),
        (("dataloader", "filter_static_segments"), True),
        (("model", "video_backbone", "continuous_timestep_conditioning"), False),
        (("model", "architecture", "video_local_expansion_weight"), 1.0),
        (("model", "architecture", "action_non_regression_weight"), 1.0),
        (
            ("training", "trainable_parameter_patterns"),
            [*VIDEO_PATTERNS, "action_backbone.*"],
        ),
    ):
        changed = deepcopy(base)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        mutations.append(changed)

    changed = deepcopy(base)
    changed["training"]["critic"] = {"enabled": True}
    mutations.append(changed)
    changed = deepcopy(base)
    changed["dataloader"]["train_tasks"][0:2] = reversed(
        changed["dataloader"]["train_tasks"][0:2]
    )
    mutations.append(changed)

    for changed in mutations:
        with pytest.raises(Phase6ArmConfigError):
            validate_arm_config(changed)


def test_a0_rejects_reference_and_a1_requires_complete_reference(tmp_path):
    _, configs, _ = _materialize(tmp_path)
    by_arm = {config["training"]["phase6_arm"]: config for config in configs}

    a0 = deepcopy(by_arm["T1_E0A0"])
    a0["training"]["phase1_action_reference_artifact"] = "/tmp/forbidden.json"
    a0["training"]["phase1_action_reference_artifact_sha256"] = "a" * 64
    with pytest.raises(Phase6ArmConfigError, match="keys differ"):
        validate_arm_config(a0)

    for key in (
        "phase1_action_reference_artifact",
        "phase1_action_reference_artifact_sha256",
    ):
        a1 = deepcopy(by_arm["T1_E0A1"])
        del a1["training"][key]
        with pytest.raises(Phase6ArmConfigError):
            validate_arm_config(a1)


def test_templates_and_fake_pins_fail_closed(tmp_path):
    template = load_arm_config(TEMPLATE_DIR / "train_phase6_T0_E0A0.yaml.in")
    with pytest.raises(Phase6ArmConfigError):
        validate_arm_config(template)

    _, _, pins = _materialize(tmp_path / "valid")
    output = tmp_path / "output"
    bad = deepcopy(pins)
    bad["common"]["dataset_contract"]["sha256"] = "a" * 64
    with pytest.raises(Phase6ArmConfigError, match="SHA256 mismatch"):
        materialize_arm_configs(
            template_dir=TEMPLATE_DIR,
            output_dir=output,
            materialization_pins=bad,
        )
    assert not output.exists()

    bad = deepcopy(pins)
    bad["common"]["dataset_contract"]["sha256"] = "__PHASE6_DATASET_CONTRACT_SHA256__"
    with pytest.raises(Phase6ArmConfigError, match="lowercase SHA256"):
        materialize_arm_configs(
            template_dir=TEMPLATE_DIR,
            output_dir=output,
            materialization_pins=bad,
        )


def test_materializer_refuses_overwrite(tmp_path):
    paths, _, pins = _materialize(tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_arm_configs(
            template_dir=TEMPLATE_DIR,
            output_dir=paths[0].parent,
            materialization_pins=pins,
        )


def test_materialization_pins_survive_canonical_json_round_trip(tmp_path):
    _, _, pins = _materialize(tmp_path / "valid")
    data = canonical_json_bytes(pins)
    path = tmp_path / "materialization_pins.json"
    path.write_bytes(data)

    # Canonical JSON sorts object keys, while scientific arm processing retains
    # the frozen ARM_NAMES order.
    assert tuple(pins["arms"]) == ARM_NAMES
    loaded = load_phase6_materialization_pins(
        path, expected_sha256=sha256(data).hexdigest()
    )
    assert tuple(loaded["arms"]) != ARM_NAMES
    round_trip_paths = materialize_arm_configs(
        template_dir=TEMPLATE_DIR,
        output_dir=tmp_path / "round_trip_configs",
        materialization_pins=loaded,
    )
    assert (
        tuple(
            load_arm_config(config)["training"]["phase6_arm"]
            for config in round_trip_paths
        )
        == ARM_NAMES
    )


@pytest.mark.parametrize(
    ("key", "alias"),
    (
        ("historical_checkpoint_count", False),
        ("historical_optimizer_step_calls_per_arm", True),
        ("historical_scheduled_lr_scale_at_step0", 0),
    ),
)
def test_materialization_lifecycle_rejects_json_numeric_aliases(tmp_path, key, alias):
    _, _, pins = _materialize(tmp_path / "valid")
    pins[key] = alias
    with pytest.raises(Phase6ArmConfigError, match=key):
        arm_config.validate_phase6_materialization_pins(pins, verify_files=False)


def test_arm_set_requires_common_dataset_and_reference_pins(tmp_path):
    _, configs, _ = _materialize(tmp_path)

    with pytest.raises(Phase6ArmConfigError, match="arm set differs"):
        validate_arm_set(configs[:-1])

    changed = deepcopy(configs)
    changed[0]["training"]["phase6_dataset_contract_artifact"] = str(
        (tmp_path / "other.json").resolve()
    )
    validate_arm_config(changed[0])
    with pytest.raises(Phase6ArmConfigError, match="share one dataset"):
        validate_arm_set(changed)

    changed = deepcopy(configs)
    changed[0]["training"]["phase6_code_source_manifest"] = str(
        (tmp_path / "other_source.json").resolve()
    )
    validate_arm_config(changed[0])
    with pytest.raises(Phase6ArmConfigError, match="source-manifest"):
        validate_arm_set(changed)

    changed = deepcopy(configs)
    a1_indices = [
        index
        for index, config in enumerate(changed)
        if ARM_FACTORS[config["training"]["phase6_arm"]][2]
    ]
    for key in (
        "phase1_action_reference_artifact",
        "phase6_action_reference_provenance_artifact",
    ):
        changed[a1_indices[0]]["training"][key] = str(
            (tmp_path / "other_reference.json").resolve()
        )
    validate_arm_config(changed[a1_indices[0]])
    with pytest.raises(Phase6ArmConfigError, match="share one action-reference"):
        validate_arm_set(changed)


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    config = tmp_path / "duplicate.yaml"
    config.write_text("model: {}\nmodel: {}\n", encoding="utf-8")
    with pytest.raises(Phase6ArmConfigError, match="duplicate YAML key"):
        load_arm_config(config)
