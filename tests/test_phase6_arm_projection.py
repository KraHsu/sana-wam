from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest
from sana_wam.train.phase6_arm_config import (
    ARM_FACTORS,
    ARM_NAMES,
    PRIMARY_RUN_DIRECTORIES,
    PRIMARY_RUN_IDS,
    Phase6ArmConfigError,
    build_phase6_arm_projection_manifest,
    canonical_json_bytes,
    validate_phase6_arm_projection_manifest_bytes,
    write_phase6_arm_projection_manifest,
)
from sana_wam.train.phase6_recovery import recovery_lifecycle


def _artifact():
    value = build_phase6_arm_projection_manifest()
    data = canonical_json_bytes(value)
    return value, data, sha256(data).hexdigest()


def test_projection_manifest_is_exact_five_arm_matrix():
    value, data, digest = _artifact()
    loaded = validate_phase6_arm_projection_manifest_bytes(
        data, expected_artifact_sha256=digest
    )
    assert [entry["arm"] for entry in loaded["arms"]] == list(ARM_NAMES)
    for entry in loaded["arms"]:
        arm = entry["arm"]
        continuous, expansion, adapter = ARM_FACTORS[arm]
        assert entry["factors"] == {"A": adapter, "E": expansion, "T": continuous}
        assert entry["run_id"] == PRIMARY_RUN_IDS[arm]
        assert entry["run_directory"] == PRIMARY_RUN_DIRECTORIES[arm]
        assert entry["projection"]["training"]["output_dir"] == entry["run_directory"]
        assert entry["projection"]["training"]["phase6_run_id"] == entry["run_id"]
        assert (
            entry["projection_sha256"]
            == sha256(canonical_json_bytes(entry["projection"])).hexdigest()
        )
    assert value == loaded
    assert {key: loaded[key] for key in recovery_lifecycle()} == recovery_lifecycle()
    assert "closed_loop_started" not in loaded


def test_selected_arm_and_projection_sha_are_bound():
    value, data, digest = _artifact()
    entry = value["arms"][3]
    validate_phase6_arm_projection_manifest_bytes(
        data,
        expected_artifact_sha256=digest,
        expected_arm=entry["arm"],
        expected_projection_sha256=entry["projection_sha256"],
    )
    with pytest.raises(Phase6ArmConfigError, match="projection SHA256 mismatch"):
        validate_phase6_arm_projection_manifest_bytes(
            data,
            expected_artifact_sha256=digest,
            expected_arm=entry["arm"],
            expected_projection_sha256="e" * 64,
        )
    with pytest.raises(Phase6ArmConfigError, match="requires expected_arm"):
        validate_phase6_arm_projection_manifest_bytes(
            data,
            expected_artifact_sha256=digest,
            expected_projection_sha256=entry["projection_sha256"],
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["arms"].reverse(), "differs from frozen design"),
        (
            lambda value: value["arms"][0]["projection"]["training"].__setitem__(
                "max_steps", 503
            ),
            "differs from frozen design",
        ),
        (
            lambda value: value["arms"][1].__setitem__(
                "run_directory", "/tmp/unregistered"
            ),
            "differs from frozen design",
        ),
    ],
)
def test_any_scientific_or_run_binding_drift_fails(mutator, message):
    value, _data, _digest = _artifact()
    changed = deepcopy(value)
    mutator(changed)
    changed_data = canonical_json_bytes(changed)
    with pytest.raises(Phase6ArmConfigError, match=message):
        validate_phase6_arm_projection_manifest_bytes(
            changed_data,
            expected_artifact_sha256=sha256(changed_data).hexdigest(),
        )


def test_noncanonical_duplicate_and_wrong_file_sha_fail_closed():
    _value, data, _digest = _artifact()
    with pytest.raises(Phase6ArmConfigError, match="file SHA256 mismatch"):
        validate_phase6_arm_projection_manifest_bytes(
            data, expected_artifact_sha256="e" * 64
        )
    noncanonical = data.replace(b'"arms":', b'"arms" :', 1)
    with pytest.raises(Phase6ArmConfigError, match="canonical JSON"):
        validate_phase6_arm_projection_manifest_bytes(
            noncanonical,
            expected_artifact_sha256=sha256(noncanonical).hexdigest(),
        )
    duplicate = data.replace(b'{"arms":', b'{"arms":[],"arms":', 1)
    with pytest.raises(Phase6ArmConfigError, match="duplicate JSON key"):
        validate_phase6_arm_projection_manifest_bytes(
            duplicate,
            expected_artifact_sha256=sha256(duplicate).hexdigest(),
        )


@pytest.mark.parametrize(
    ("key", "alias"),
    (
        ("historical_checkpoint_count", False),
        ("historical_optimizer_step_calls_per_arm", True),
        ("historical_scheduled_lr_scale_at_step0", 0),
    ),
)
def test_projection_manifest_rejects_lifecycle_type_aliases(key, alias):
    value, _data, _digest = _artifact()
    value[key] = alias
    data = canonical_json_bytes(value)
    with pytest.raises(Phase6ArmConfigError, match=key):
        validate_phase6_arm_projection_manifest_bytes(
            data, expected_artifact_sha256=sha256(data).hexdigest()
        )


def test_projection_manifest_publish_is_exclusive(tmp_path):
    value, data, digest = _artifact()
    path = tmp_path / "projection.json"
    assert write_phase6_arm_projection_manifest(path, value) == digest
    assert path.read_bytes() == data
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_phase6_arm_projection_manifest(path, value)
    assert path.read_bytes() == data
