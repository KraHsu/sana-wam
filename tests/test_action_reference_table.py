from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from sana_wam.dataloader.task_sample_plan import PlannedSample, SampleIdentity
from sana_wam.train.action_reference_table import (
    ActionReferenceTable,
    ActionReferenceValidationError,
    EXPECTED_ACTION_STATS_SHA256,
    EXPECTED_REFERENCE_CHECKPOINT_SHA256,
    PRECOMPUTE_CONFIG_SCHEMA_VERSION,
    SPOT_CHECK_GLOBAL_STEPS,
    SPOT_CHECK_SCHEMA_VERSION,
    canonical_json_bytes,
    validate_live_spot_artifact_bytes,
    validate_reference_artifact_envelope_bytes,
)
from sana_wam.train.phase6_downstream_pins import (
    DATASET_CONTRACT_ARTIFACT_SHA256,
    FIXED_AMENDMENT_CONFIG_PINS,
    TASK_PLAN_ARTIFACT_SHA256,
)


TEST_DATASET_CONTRACT_SHA256 = DATASET_CONTRACT_ARTIFACT_SHA256


def _row(step: int) -> PlannedSample:
    return PlannedSample(
        global_step=step,
        cycle=(step - 1) // 42,
        position_in_cycle=(step - 1) % 42,
        identity=SampleIdentity(
            task_name=f"task_{(step - 1) % 42:02d}",
            episode_index=(step - 1) % 50,
            episode_path=f"/data/task_{(step - 1) % 42:02d}/episode{step}.hdf5",
            start_frame=step,
            prompt=f"prompt {step}",
            dataset_index=step + 1000,
        ),
        action_sigma=(1.0, 0.9, 0.5)[step % 3],
        domain_seeds=(
            ("video-noise", step),
            ("action-noise", step + 1),
            ("expansion-noise", step + 2),
            ("expansion-direction", step + 3),
            ("reference-query", step + 4),
            ("prompt-choice", step + 5),
        ),
    )


class _Plan:
    plan_sha256 = "a" * 64
    identity_sha256 = "b" * 64
    rows = tuple(_row(step) for step in range(1, 505))

    def validate(self):
        return None


def _dataset_rows():
    return tuple(
        {
            "global_step": row.global_step,
            "dataset_index": row.identity.dataset_index,
            "task_name": row.identity.task_name,
            "episode_index": row.identity.episode_index,
            "episode_path": row.identity.episode_path,
            "start_frame": row.identity.start_frame,
            "window_logical_length": 113,
            "episode_sha256": hashlib.sha256(
                row.identity.episode_path.encode()
            ).hexdigest(),
            "instruction_source_sha256": hashlib.sha256(
                row.identity.prompt.encode()
            ).hexdigest(),
        }
        for row in _Plan.rows
    )


def _precompute_config():
    return {
        "schema_version": PRECOMPUTE_CONFIG_SCHEMA_VERSION,
        "model": {
            "architecture": {
                "video_local_expansion_weight": 0.0,
                "action_non_regression_weight": 0.0,
                "action_video_memory_adapter": {"enabled": False},
            },
            "video_backbone": {"continuous_timestep_conditioning": False},
        },
        "dataloader": {
            "type": "robotwin",
            "filter_static_segments": False,
            "growing_history": False,
            "text_embedding_cache_dir": None,
            "text_embedding_dropout": 0.0,
            "vae_cache_dir": None,
        },
        "training": {
            "phase1_action_reference_precompute_mode": True,
            "init_checkpoint_sha256": EXPECTED_REFERENCE_CHECKPOINT_SHA256,
            "phase1_reference_checkpoint_sha256": (
                EXPECTED_REFERENCE_CHECKPOINT_SHA256
            ),
            "action_stats_sha256": EXPECTED_ACTION_STATS_SHA256,
            "phase6_plan_sha256": _Plan.plan_sha256,
            "phase6_identity_sha256": _Plan.identity_sha256,
            "phase6_plan_artifact_sha256": TASK_PLAN_ARTIFACT_SHA256,
            "phase6_dataset_contract_artifact_sha256": (
                TEST_DATASET_CONTRACT_SHA256
            ),
            **FIXED_AMENDMENT_CONFIG_PINS,
            "lambda_video": 1.0,
            "lambda_action": 0.0,
            "expected_world_size": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 504,
            "seed": 20260724,
            "num_workers": 0,
        },
        "execution": {
            "autograd_context": "torch.no_grad",
            "autograd_enabled": False,
            "backward_called": False,
            "closed_loop": False,
            "forward_only": True,
            "gradient_checkpointing": False,
            "model_mode": "eval",
            "model_result_key": "phase6_action_unweighted_mse",
            "optimizer_created": False,
            "phase6_plan_rows_required": True,
            "reference_input_key": None,
            "resume_allowed": False,
            "row_context_contract": (
                "plan-row prompt plus dataset-contract logical-length/episode/instruction"
            ),
            "row_noise_contract": "phase6_plan_row.domain_seeds",
            "spot_check_global_steps": [1, 253, 504],
            "training_started": False,
        },
    }


def _common_traces():
    return tuple(hashlib.sha256(f"common-{step}".encode()).hexdigest() for step in range(1, 505))


def _t0_traces():
    return tuple(hashlib.sha256(f"t0-{step}".encode()).hexdigest() for step in range(1, 505))


def _gpu_runtime():
    return {
        "compute_capability": [9, 0],
        "cuda_device_count": 8,
        "cuda_device_index": 0,
        "cuda_device_name": "Synthetic H200",
        "cuda_device_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "cuda_driver_version": "570.86",
        "device_type": "cuda",
        "multi_processor_count": 132,
        "total_memory_bytes": 141_000_000_000,
        "torch_cuda_version": "12.8",
        "torch_version": "2.7.1+cu128",
        "world_size": 1,
    }


def _table(config=None) -> ActionReferenceTable:
    return ActionReferenceTable.build(
        _Plan(),
        _dataset_rows(),
        [step / 1000 for step in range(1, 505)],
        _common_traces(),
        _t0_traces(),
        dataset_contract_artifact_sha256=(
            TEST_DATASET_CONTRACT_SHA256
        ),
        reference_checkpoint_sha256=EXPECTED_REFERENCE_CHECKPOINT_SHA256,
        action_stats_sha256=EXPECTED_ACTION_STATS_SHA256,
        precompute_config_file_sha256="c" * 64,
        precompute_config=config or _precompute_config(),
        precompute_gpu_runtime=_gpu_runtime(),
        precompute_preflight_request_sha256="e" * 64,
        precompute_preflight_report_sha256="f" * 64,
        precompute_source_manifest_sha256="d" * 64,
    )


def _load(artifact: bytes, **overrides):
    kwargs = {
        "expected_artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "expected_dataset_contract_artifact_sha256": (
            TEST_DATASET_CONTRACT_SHA256
        ),
        "expected_reference_checkpoint_sha256": (
            EXPECTED_REFERENCE_CHECKPOINT_SHA256
        ),
        "expected_action_stats_sha256": EXPECTED_ACTION_STATS_SHA256,
        "expected_precompute_source_manifest_sha256": "d" * 64,
    }
    kwargs.update(overrides)
    return ActionReferenceTable.from_artifact_bytes(
        artifact, _Plan(), _dataset_rows(), **kwargs
    )


def _spot_bytes(table: ActionReferenceTable, reference_sha256: str) -> bytes:
    rows = []
    for global_step in SPOT_CHECK_GLOBAL_STEPS:
        index = global_step - 1
        rows.append(
            {
                "global_step": global_step,
                "expected_error_float32_bits": table.error_bits[index],
                "observed_error_float32_bits": table.error_bits[index],
                "exact_match": True,
                "common_input_trace_sha256": (
                    table.common_input_trace_sha256[index]
                ),
                "plan_row_sha256": table.plan_row_sha256[index],
                "dataset_contract_row_sha256": (
                    table.dataset_contract_row_sha256[index]
                ),
                "reference_forward_context_sha256": (
                    table.reference_forward_context_sha256[index]
                ),
                "t0_reference_forward_trace_sha256": (
                    table.t0_reference_forward_trace_sha256[index]
                ),
            }
        )
    return canonical_json_bytes(
        {
            "schema_version": SPOT_CHECK_SCHEMA_VERSION,
            "verification_passed": True,
            "gpu_forward_executed": True,
            "training_started": False,
            "closed_loop": False,
            "reference_artifact_sha256": reference_sha256,
            "plan_sha256": table.plan_sha256,
            "identity_sha256": table.identity_sha256,
            "dataset_contract_artifact_sha256": (
                table.dataset_contract_artifact_sha256
            ),
            "reference_checkpoint_sha256": table.reference_checkpoint_sha256,
            "action_stats_sha256": table.action_stats_sha256,
            "precompute_config_file_sha256": (
                table.precompute_config_file_sha256
            ),
            "precompute_config_sha256": table.precompute_config_sha256,
            "precompute_gpu_runtime_sha256": (
                table.precompute_gpu_runtime_sha256
            ),
            "precompute_preflight_report_sha256": (
                table.precompute_preflight_report_sha256
            ),
            "precompute_preflight_request_sha256": (
                table.precompute_preflight_request_sha256
            ),
            "precompute_source_manifest_sha256": (
                table.precompute_source_manifest_sha256
            ),
            "spot_gpu_runtime": table.precompute_gpu_runtime,
            "spot_gpu_runtime_sha256": table.precompute_gpu_runtime_sha256,
            "spot_preflight_report_sha256": (
                table.precompute_preflight_report_sha256
            ),
            "spot_preflight_request_sha256": (
                table.precompute_preflight_request_sha256
            ),
            "global_steps": list(SPOT_CHECK_GLOBAL_STEPS),
            "rows": rows,
        }
    )


def test_reference_v2_roundtrip_preserves_float32_bits_and_all_504_rows():
    table = _table()
    artifact = table.to_artifact_bytes(_Plan(), _dataset_rows())
    loaded = _load(artifact)
    assert len(loaded.error_bits) == 504
    assert loaded.error_bits == table.error_bits
    assert loaded.errors == table.errors
    assert loaded.error_for_global_step(504) == table.errors[-1]
    assert loaded.artifact_sha256(_Plan(), _dataset_rows()) == hashlib.sha256(
        artifact
    ).hexdigest()
    assert loaded.precompute_config == _precompute_config()


def test_reference_rows_bind_plan_noise_dataset_padding_and_instruction_context():
    table = _table()
    artifact = table.to_artifact_bytes(_Plan(), _dataset_rows())
    dataset_rows = list(_dataset_rows())
    dataset_rows[252] = {
        **dataset_rows[252],
        "window_logical_length": 112,
    }
    with pytest.raises(ActionReferenceValidationError, match="provenance drift"):
        ActionReferenceTable.from_artifact_bytes(
            artifact,
            _Plan(),
            dataset_rows,
            expected_artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            expected_dataset_contract_artifact_sha256=(
                TEST_DATASET_CONTRACT_SHA256
            ),
            expected_reference_checkpoint_sha256=(
                EXPECTED_REFERENCE_CHECKPOINT_SHA256
            ),
            expected_action_stats_sha256=EXPECTED_ACTION_STATS_SHA256,
            expected_precompute_source_manifest_sha256="d" * 64,
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("model", "video_backbone", "continuous_timestep_conditioning"),
            True,
            "T0",
        ),
        (("model", "architecture", "video_local_expansion_weight"), 1.0, "0.0"),
        (("model", "architecture", "action_non_regression_weight"), 1.0, "0.0"),
        (
            ("model", "architecture", "action_video_memory_adapter", "enabled"),
            True,
            "disabled",
        ),
        (("training", "lambda_action"), 1.0, "lambda_action"),
        (("training", "num_workers"), 4, "num_workers"),
        (("execution", "forward_only"), False, "execution contract"),
        (("execution", "closed_loop"), True, "execution contract"),
    ],
)
def test_precompute_config_is_exact_t0_forward_only(path, value, message):
    config = _precompute_config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ActionReferenceValidationError, match=message):
        _table(config)


@pytest.mark.parametrize("field", sorted(FIXED_AMENDMENT_CONFIG_PINS))
def test_precompute_config_rejects_each_fixed_amendment_pin_mutation(field):
    config = _precompute_config()
    expected = config["training"][field]
    config["training"][field] = (
        "f" * 64 if field.endswith("_sha256") else f"{expected}.drift"
    )
    with pytest.raises(ActionReferenceValidationError, match=field):
        _table(config)


def test_precompute_may_not_consume_an_existing_reference_table():
    config = _precompute_config()
    config["training"]["phase1_action_reference_artifact"] = "/old/table.json"
    with pytest.raises(ActionReferenceValidationError, match="may not consume"):
        _table(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_artifact_sha256", "e" * 64),
        ("expected_dataset_contract_artifact_sha256", "e" * 64),
        ("expected_reference_checkpoint_sha256", "e" * 64),
        ("expected_action_stats_sha256", "e" * 64),
        ("expected_precompute_source_manifest_sha256", "e" * 64),
    ],
)
def test_external_artifact_and_provenance_pins_fail_closed(field, value):
    artifact = _table().to_artifact_bytes(_Plan(), _dataset_rows())
    with pytest.raises(ActionReferenceValidationError):
        _load(artifact, **{field: value})


def test_reference_rejects_v1_noncanonical_duplicate_and_row_drift():
    artifact = _table().to_artifact_bytes(_Plan(), _dataset_rows())
    payload = json.loads(artifact)

    payload["schema_version"] = "sana-phase6-action-reference-v1"
    v1 = canonical_json_bytes(payload)
    with pytest.raises(ActionReferenceValidationError, match="requires v2"):
        _load(v1)

    with pytest.raises(ActionReferenceValidationError, match="canonical"):
        _load(b"  " + artifact)

    duplicate = artifact.replace(
        b'{"action_stats_sha256":',
        b'{"schema_version":"duplicate","action_stats_sha256":',
        1,
    )
    with pytest.raises(ActionReferenceValidationError, match="duplicate"):
        _load(duplicate)

    payload = json.loads(artifact)
    payload["rows"][0]["dataset_index"] += 1
    drift = canonical_json_bytes(payload)
    with pytest.raises(ActionReferenceValidationError, match="identity drift"):
        _load(drift)


def test_reference_rejects_nonfinite_negative_and_negative_zero_values():
    for value in (float("nan"), float("inf"), -1.0, -0.0):
        errors = [0.1] * 504
        errors[0] = value
        with pytest.raises(ActionReferenceValidationError, match="finite|non-negative"):
            ActionReferenceTable.build(
                _Plan(),
                _dataset_rows(),
                errors,
                _common_traces(),
                _t0_traces(),
                dataset_contract_artifact_sha256=(
                    TEST_DATASET_CONTRACT_SHA256
                ),
                reference_checkpoint_sha256=(
                    EXPECTED_REFERENCE_CHECKPOINT_SHA256
                ),
                action_stats_sha256=EXPECTED_ACTION_STATS_SHA256,
                precompute_config_file_sha256="c" * 64,
                precompute_config=_precompute_config(),
                precompute_gpu_runtime=_gpu_runtime(),
                precompute_preflight_request_sha256="e" * 64,
                precompute_preflight_report_sha256="f" * 64,
                precompute_source_manifest_sha256="d" * 64,
            )


def test_precompute_config_property_is_defensive_copy():
    table = _table()
    observed = table.precompute_config
    observed["training"]["lambda_action"] = 99
    assert table.precompute_config["training"]["lambda_action"] == 0.0


def test_precompute_gpu_runtime_property_is_defensive_and_sha_bound():
    table = _table()
    observed = table.precompute_gpu_runtime
    observed["cuda_device_name"] = "different"
    assert table.precompute_gpu_runtime["cuda_device_name"] == "Synthetic H200"
    assert table.precompute_gpu_runtime_sha256 == hashlib.sha256(
        canonical_json_bytes(_gpu_runtime())
    ).hexdigest()


def test_reference_envelope_is_torch_free_and_exposes_preflight_pins():
    table = _table()
    artifact = table.to_artifact_bytes(_Plan(), _dataset_rows())
    loaded = validate_reference_artifact_envelope_bytes(
        artifact,
        expected_artifact_sha256=hashlib.sha256(artifact).hexdigest(),
    )
    assert loaded["precompute_preflight_request_sha256"] == "e" * 64
    payload = json.loads(artifact)
    payload["precompute_preflight_report_sha256"] = "a" * 64
    changed = canonical_json_bytes(payload)
    loaded = validate_reference_artifact_envelope_bytes(
        changed,
        expected_artifact_sha256=hashlib.sha256(changed).hexdigest(),
    )
    assert loaded["precompute_preflight_report_sha256"] == "a" * 64


def test_precompute_config_sha_binds_every_exact_config_field():
    first = _table()
    changed = deepcopy(_precompute_config())
    changed["training"]["output_dir"] = "/different-but-semantic-noop-path"
    second = _table(changed)
    assert first.precompute_config_sha256 != second.precompute_config_sha256


def test_live_spot_roundtrip_binds_exact_table_bits_and_both_trace_classes():
    table = _table()
    reference_bytes = table.to_artifact_bytes(_Plan(), _dataset_rows())
    reference_sha256 = hashlib.sha256(reference_bytes).hexdigest()
    spot = _spot_bytes(table, reference_sha256)

    loaded = validate_live_spot_artifact_bytes(
        spot,
        table,
        _Plan(),
        _dataset_rows(),
        expected_artifact_sha256=hashlib.sha256(spot).hexdigest(),
        expected_reference_artifact_sha256=reference_sha256,
    )

    assert loaded["global_steps"] == [1, 253, 504]
    assert loaded["verification_passed"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_error_float32_bits", "00000000"),
        ("common_input_trace_sha256", "e" * 64),
        ("t0_reference_forward_trace_sha256", "e" * 64),
        ("reference_forward_context_sha256", "e" * 64),
        ("exact_match", False),
    ],
)
def test_live_spot_rejects_error_or_trace_drift(field, value):
    table = _table()
    reference_sha256 = table.artifact_sha256(_Plan(), _dataset_rows())
    payload = json.loads(_spot_bytes(table, reference_sha256))
    payload["rows"][1][field] = value
    spot = canonical_json_bytes(payload)

    with pytest.raises(ActionReferenceValidationError, match="step 253"):
        validate_live_spot_artifact_bytes(
            spot,
            table,
            _Plan(),
            _dataset_rows(),
            expected_artifact_sha256=hashlib.sha256(spot).hexdigest(),
            expected_reference_artifact_sha256=reference_sha256,
        )


def test_live_spot_rejects_wrong_step_set_and_wrong_reference_identity():
    table = _table()
    reference_sha256 = table.artifact_sha256(_Plan(), _dataset_rows())
    payload = json.loads(_spot_bytes(table, reference_sha256))
    payload["global_steps"] = [1, 252, 504]
    wrong_steps = canonical_json_bytes(payload)
    with pytest.raises(ActionReferenceValidationError, match="global steps"):
        validate_live_spot_artifact_bytes(
            wrong_steps,
            table,
            _Plan(),
            _dataset_rows(),
            expected_artifact_sha256=hashlib.sha256(wrong_steps).hexdigest(),
            expected_reference_artifact_sha256=reference_sha256,
        )

    spot = _spot_bytes(table, reference_sha256)
    with pytest.raises(ActionReferenceValidationError, match="identify the table"):
        validate_live_spot_artifact_bytes(
            spot,
            table,
            _Plan(),
            _dataset_rows(),
            expected_artifact_sha256=hashlib.sha256(spot).hexdigest(),
            expected_reference_artifact_sha256="f" * 64,
        )
