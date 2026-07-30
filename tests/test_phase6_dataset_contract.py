from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import sana_wam.dataloader.phase6_dataset_contract as contract_module
from sana_wam.dataloader.phase6_dataset_contract import (
    DatasetContractError,
    Phase6DatasetContract,
    runtime_window_row,
    sha256_file,
)
from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    expansion_support_for_window,
)
from sana_wam.dataloader.phase6_padding_semantics import (
    PADDING_SEMANTICS_AMENDMENT_SHA256,
    PADDING_SEMANTICS_CONTRACT_SHA256,
    padding_semantics_contract,
)


@dataclass(frozen=True)
class FakeIdentity:
    task_name: str
    episode_index: int
    episode_path: str
    start_frame: int
    dataset_index: int


@dataclass(frozen=True)
class FakePlanRow:
    global_step: int
    identity: FakeIdentity


class FakePlan:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.plan_sha256 = "1" * 64
        self.identity_sha256 = "2" * 64

    def to_artifact_bytes(self) -> bytes:
        value = [
            {
                "global_step": row.global_step,
                "dataset_index": row.identity.dataset_index,
                "episode_path": row.identity.episode_path,
                "start_frame": row.identity.start_frame,
            }
            for row in self.rows
        ]
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class FakeChild:
    def __init__(self, task_name: str, data_root: Path, episode_path: Path):
        self.task_name = task_name
        self.prompt_task_name = task_name.replace("_", " ")
        self.data_root = str(data_root)
        self._episode_files = [str(episode_path)]
        self._episode_lengths = [200]
        self._window_index = [(0, start, 113) for start in range(12)]
        self._val_samples = None
        self._instructions = {}

        self.action_mode = "eef"
        self.causal_temporal = True
        self.delta_action = False
        self._filter_static_segments = False
        self.growing_history = False
        self.height = 384
        self.multiview = True
        self.normalize_mode = "min-max"
        self.num_frames = 113
        self.repeat = 1
        self.robot = "aloha-agilex"
        self.split = "train"
        self.temporal_compression = 4
        self.variant = "clean_50"
        self.video_stride = 4
        self.width = 320
        self.window_stride = 1
        self.camera_layout = ["head_camera", "left_camera", "right_camera"]
        self.target_camera = "head_camera"
        self.vae_type = "wan"
        self._raw_window_len = 113
        self.num_video_frames = 29
        self._video_sample_indices = list(range(0, 113, 4))
        self.history_min_frames = 2
        self.history_stride = 1
        self.gdn_chunk_size = 1
        self._static_segment_threshold = 1.0e-5
        self._max_static_retry = 3
        self._text_embedding_transform = None
        self._vae_latent_transform = None

    def __len__(self):
        return len(self._window_index)

    def _resolve_local_index(self, index: int):
        return self._window_index[index]


class FakeDataset:
    def __init__(self, children, action_stats_path: Path):
        self._sub_datasets = list(children)
        self._cumulative_lengths = []
        running = 0
        for child in children:
            running += len(child)
            self._cumulative_lengths.append(running)
        self._total_length = running
        self.variant = "clean_50"
        self.dataset_type = "robotwin"
        self.action_stats_path = str(action_stats_path)

    @property
    def task_names(self):
        return tuple(child.task_name for child in self._sub_datasets)

    def resolve_global_index(self, index: int):
        if type(index) is not int or index < 0 or index >= self._total_length:
            raise IndexError(index)
        for child_index, cumulative in enumerate(self._cumulative_lengths):
            if index < cumulative:
                prior = (
                    0 if child_index == 0 else self._cumulative_lengths[child_index - 1]
                )
                return child_index, index - prior
        raise AssertionError("unreachable")


@pytest.fixture()
def synthetic_runtime(tmp_path):
    sources = {}
    for source_id in (
        "sana_wam.dataloader.robotwin_dataset",
        "sana_wam.dataloader.transforms.multiview",
        "sana_wam.dataloader.transforms.normalize",
        "sana_wam.dataloader.transforms.rotation",
    ):
        source = tmp_path / f"{source_id.rsplit('.', 1)[-1]}.py"
        source.write_bytes(f"# frozen synthetic source: {source_id}\n".encode())
        sources[source_id] = source
    stats = tmp_path / "stats.npy"
    stats.write_bytes(b"synthetic-action-stats\n")
    children = []
    rows = []
    tasks = tuple(f"task_{index:02d}" for index in range(42))
    global_step = 1
    for task_index, task in enumerate(tasks):
        data_root = tmp_path / task / "data"
        instruction_root = data_root.parent / "instructions"
        data_root.mkdir(parents=True)
        episode = data_root / "episode0.hdf5"
        episode.write_bytes(f"episode bytes for {task}\n".encode())
        child = FakeChild(task, data_root, episode)
        if task_index % 2 == 0:
            instruction_root.mkdir()
            instruction = instruction_root / "episode0.json"
            instruction.write_text(
                json.dumps({"seen": [f"perform {task}"]}), encoding="utf-8"
            )
            child._instructions["episode0.json"] = {"seen": [f"perform {task}"]}
        children.append(child)
        for local_index in range(12):
            dataset_index = task_index * 12 + local_index
            rows.append(
                FakePlanRow(
                    global_step=global_step,
                    identity=FakeIdentity(
                        task_name=task,
                        episode_index=0,
                        episode_path=str(episode),
                        start_frame=local_index,
                        dataset_index=dataset_index,
                    ),
                )
            )
            global_step += 1
    dataset = FakeDataset(children, stats)
    plan = FakePlan(rows)
    config = {
        "dataset_dir": str(tmp_path),
        "text_embedding_cache_dir": None,
        "vae_cache_dir": None,
        "text_embedding_dropout": 0.0,
        "val_ratio": 0.0,
        "seed": 42,
        "num_val_samples": 4,
        "backbone": None,
    }
    action_stats_sha256 = sha256_file(stats)
    contract = Phase6DatasetContract.build(
        dataset,
        plan,
        dataloader_config=config,
        action_stats_sha256=action_stats_sha256,
        preprocessing_source_paths=sources,
        train_tasks=tasks,
    )
    return SimpleNamespace(
        sources=sources,
        stats=stats,
        children=children,
        tasks=tasks,
        dataset=dataset,
        plan=plan,
        config=config,
        action_stats_sha256=action_stats_sha256,
        contract=contract,
    )


def validate(runtime, **overrides):
    runtime.contract.validate_runtime(
        overrides.get("dataset", runtime.dataset),
        overrides.get("plan", runtime.plan),
        dataloader_config=overrides.get("config", runtime.config),
        action_stats_sha256=overrides.get(
            "action_stats_sha256", runtime.action_stats_sha256
        ),
        preprocessing_source_paths=overrides.get("sources", runtime.sources),
        train_tasks=overrides.get("tasks", runtime.tasks),
        file_hasher=overrides.get("file_hasher", sha256_file),
    )


def canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def test_build_and_validate_all_504_rows(synthetic_runtime):
    runtime = synthetic_runtime
    assert len(runtime.contract.rows) == 504
    assert len(runtime.contract.episodes) == 42
    assert [row["global_step"] for row in runtime.contract.rows] == list(range(1, 505))
    validate(runtime)


def test_dataset_contract_v2_binds_support_at_top_level_and_per_row(
    synthetic_runtime,
):
    contract = synthetic_runtime.contract
    assert contract.schema_version == "sana-phase6-dataset-contract-v2"
    value = contract.to_dict()
    assert (
        value["expansion_support_contract_sha256"]
        == EXPANSION_SUPPORT_CONTRACT_SHA256
    )
    assert (
        value["expansion_eligibility_amendment_sha256"]
        == EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
    )
    assert (
        contract.padding_semantics_amendment_sha256
        == PADDING_SEMANTICS_AMENDMENT_SHA256
    )
    assert (
        contract.padding_semantics_contract_sha256
        == PADDING_SEMANTICS_CONTRACT_SHA256
    )
    assert contract.padding_semantics_contract == padding_semantics_contract()
    assert all(row["eligible"] is True for row in value["rows"])
    assert min(row["eligible_chunk_count"] for row in value["rows"]) >= 1


def test_padding_contract_accessors_are_defensive_copies(synthetic_runtime):
    contract = synthetic_runtime.contract
    property_value = contract.padding_semantics_contract
    property_value["schema_version"] = "mutated"
    assert contract.padding_semantics_contract == padding_semantics_contract()

    artifact_value = contract.to_dict()
    artifact_value["padding_semantics_contract"]["schema_version"] = "mutated"
    assert contract.padding_semantics_contract == padding_semantics_contract()


@pytest.mark.parametrize(
    "mutation",
    ("nested_value", "nested_hash", "amendment_hash", "nested_extra"),
)
def test_padding_provenance_tampering_fails_closed(synthetic_runtime, mutation):
    value = synthetic_runtime.contract.to_dict()
    if mutation == "nested_value":
        value["padding_semantics_contract"]["schema_version"] = "mutated"
        value["padding_semantics_contract_sha256"] = hashlib.sha256(
            canonical_bytes(value["padding_semantics_contract"])
        ).hexdigest()
    elif mutation == "nested_hash":
        value["padding_semantics_contract_sha256"] = "0" * 64
    elif mutation == "amendment_hash":
        value["padding_semantics_amendment_sha256"] = "0" * 64
    else:
        value["padding_semantics_contract"]["extra"] = "forbidden"
        value["padding_semantics_contract_sha256"] = hashlib.sha256(
            canonical_bytes(value["padding_semantics_contract"])
        ).hexdigest()
    with pytest.raises(DatasetContractError, match="padding-semantics"):
        Phase6DatasetContract.from_artifact_bytes(canonical_bytes(value))


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema_version", 1),
        ("sequence_layout", True),
        ("video_key_rule", 1.0),
    ),
)
def test_padding_contract_rejects_numeric_type_substitutions(
    synthetic_runtime,
    field,
    replacement,
):
    value = synthetic_runtime.contract.to_dict()
    value["padding_semantics_contract"][field] = replacement
    value["padding_semantics_contract_sha256"] = hashlib.sha256(
        canonical_bytes(value["padding_semantics_contract"])
    ).hexdigest()
    with pytest.raises(DatasetContractError, match="padding-semantics"):
        Phase6DatasetContract.from_artifact_bytes(canonical_bytes(value))


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing", "keys differ"),
        ("extra", "keys differ"),
        ("numeric_sha", "lowercase SHA256"),
    ),
)
def test_padding_top_level_schema_and_types_are_exact(
    synthetic_runtime,
    mutation,
    match,
):
    value = synthetic_runtime.contract.to_dict()
    if mutation == "missing":
        del value["padding_semantics_contract"]
    elif mutation == "extra":
        value["padding_semantics_contract_alias"] = value[
            "padding_semantics_contract"
        ]
    else:
        value["padding_semantics_contract_sha256"] = 1
    with pytest.raises(DatasetContractError, match=match):
        Phase6DatasetContract.from_artifact_bytes(canonical_bytes(value))


def test_serialized_ineligible_row_and_support_pin_tampering_fail_closed(
    synthetic_runtime,
):
    value = synthetic_runtime.contract.to_dict()
    ineligible = expansion_support_for_window(
        episode_length=4,
        start_frame=0,
        window_logical_length=113,
    )
    value["rows"][0].update(ineligible)
    with pytest.raises(DatasetContractError, match="no non-bootstrap"):
        Phase6DatasetContract.from_artifact_bytes(canonical_bytes(value))

    value = synthetic_runtime.contract.to_dict()
    value["expansion_support_contract_sha256"] = "0" * 64
    with pytest.raises(DatasetContractError, match="support contract"):
        Phase6DatasetContract.from_artifact_bytes(canonical_bytes(value))


@pytest.mark.parametrize(
    "field",
    (
        "actual_valid_raw_frames",
        "valid_sampled_video_frames",
        "valid_latent_frames",
        "nonbootstrap_valid_latent_frames",
        "eligible_chunk_count",
    ),
)
def test_dataset_contract_rejects_numerically_equal_float_support_counts(
    synthetic_runtime,
    field,
):
    value = synthetic_runtime.contract.to_dict()
    value["rows"][0][field] = float(value["rows"][0][field])
    with pytest.raises(DatasetContractError, match="integer"):
        Phase6DatasetContract.from_artifact_bytes(canonical_bytes(value))


def test_dataset_contract_rejects_integer_eligible_alias(synthetic_runtime):
    value = synthetic_runtime.contract.to_dict()
    value["rows"][0]["eligible"] = 1
    with pytest.raises(DatasetContractError, match="boolean"):
        Phase6DatasetContract.from_artifact_bytes(canonical_bytes(value))


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("raw_window_frames", 113.0),
        ("bootstrap_clean_prefix", 1),
        (
            "video_sample_indices",
            [float(index) for index in range(0, 113, 4)],
        ),
    ),
)
def test_dataset_contract_rehashes_embedded_support_contract_types(
    synthetic_runtime,
    field,
    replacement,
):
    value = synthetic_runtime.contract.to_dict()
    value["expansion_support_contract"][field] = replacement
    with pytest.raises(DatasetContractError, match="expansion-support contract"):
        Phase6DatasetContract.from_artifact_bytes(canonical_bytes(value))


def test_episode_length_drift_changes_support_without_changing_old_identity(
    synthetic_runtime,
):
    runtime = synthetic_runtime
    identity_before = runtime.plan.rows[0].identity
    runtime.children[0]._episode_lengths[0] = 4
    assert runtime.plan.rows[0].identity == identity_before
    with pytest.raises(DatasetContractError, match="runtime row drift"):
        validate(runtime)


def test_builder_does_not_mutate_task_plan_bytes_or_hashes(synthetic_runtime):
    runtime = synthetic_runtime
    before = runtime.plan.to_artifact_bytes()
    pins = (runtime.plan.plan_sha256, runtime.plan.identity_sha256)
    rebuilt = Phase6DatasetContract.build(
        runtime.dataset,
        runtime.plan,
        dataloader_config=runtime.config,
        action_stats_sha256=runtime.action_stats_sha256,
        preprocessing_source_paths=runtime.sources,
        train_tasks=runtime.tasks,
    )
    assert runtime.plan.to_artifact_bytes() == before
    assert (runtime.plan.plan_sha256, runtime.plan.identity_sha256) == pins
    assert rebuilt.to_artifact_bytes() == runtime.contract.to_artifact_bytes()


def test_hashing_is_deduplicated_by_episode(synthetic_runtime):
    runtime = synthetic_runtime
    calls = []

    def counting_hasher(path):
        calls.append(os.path.abspath(path))
        return sha256_file(path)

    validate(runtime, file_hasher=counting_hasher)
    hdf5_calls = [path for path in calls if path.endswith(".hdf5")]
    assert len(hdf5_calls) == 42
    assert len(set(hdf5_calls)) == 42


def test_logical_length_drift_fails_although_old_plan_identity_is_unchanged(
    synthetic_runtime,
):
    runtime = synthetic_runtime
    identity_before = runtime.plan.rows[0].identity
    runtime.children[0]._window_index[0] = (0, 0, 112)
    assert runtime.plan.rows[0].identity == identity_before
    with pytest.raises(DatasetContractError, match="runtime row drift"):
        validate(runtime)


@pytest.mark.parametrize("mutation", ["same_size", "size_change"])
def test_episode_byte_or_size_mutation_fails(synthetic_runtime, mutation):
    runtime = synthetic_runtime
    episode = Path(runtime.children[0]._episode_files[0])
    original = episode.read_bytes()
    if mutation == "same_size":
        episode.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    else:
        episode.write_bytes(original + b"x")
    with pytest.raises(DatasetContractError, match="episode/instruction"):
        validate(runtime)


def test_episode_path_change_fails(synthetic_runtime):
    runtime = synthetic_runtime
    old = Path(runtime.children[0]._episode_files[0])
    replacement = old.with_name("episode1.hdf5")
    replacement.write_bytes(old.read_bytes())
    runtime.children[0]._episode_files[0] = str(replacement)
    with pytest.raises(DatasetContractError, match="runtime row drift"):
        validate(runtime)


def test_episode_symlink_retarget_fails_even_for_identical_bytes(synthetic_runtime):
    runtime = synthetic_runtime
    child = runtime.children[0]
    original = Path(child._episode_files[0])
    target_a = original.with_name("target-a.hdf5")
    target_b = original.with_name("target-b.hdf5")
    target_a.write_bytes(original.read_bytes())
    target_b.write_bytes(original.read_bytes())
    link = original.with_name("episode-link.hdf5")
    try:
        os.symlink(target_a, link)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    child._episode_files[0] = str(link)
    for row_index in range(12):
        old_row = runtime.plan.rows[row_index]
        runtime.plan.rows = (
            *runtime.plan.rows[:row_index],
            FakePlanRow(
                old_row.global_step,
                FakeIdentity(
                    old_row.identity.task_name,
                    old_row.identity.episode_index,
                    str(link),
                    old_row.identity.start_frame,
                    old_row.identity.dataset_index,
                ),
            ),
            *runtime.plan.rows[row_index + 1 :],
        )
    contract = Phase6DatasetContract.build(
        runtime.dataset,
        runtime.plan,
        dataloader_config=runtime.config,
        action_stats_sha256=runtime.action_stats_sha256,
        preprocessing_source_paths=runtime.sources,
        train_tasks=runtime.tasks,
    )
    link.unlink()
    os.symlink(target_b, link)
    runtime.contract = contract
    with pytest.raises(DatasetContractError, match="episode/instruction"):
        validate(runtime)


def test_realpath_retarget_is_always_exercised_without_platform_symlink_support(
    synthetic_runtime, monkeypatch
):
    runtime = synthetic_runtime
    episode = os.path.abspath(runtime.children[0]._episode_files[0])
    original_realpath = contract_module._resolved_path

    def retargeted_realpath(path):
        observed = original_realpath(path)
        return (
            observed + ".retargeted" if os.path.abspath(path) == episode else observed
        )

    monkeypatch.setattr(contract_module, "_resolved_path", retargeted_realpath)
    with pytest.raises(DatasetContractError, match="episode/instruction"):
        validate(runtime)


def test_instruction_file_mutation_fails(synthetic_runtime):
    runtime = synthetic_runtime
    instruction = (
        Path(runtime.children[0].data_root).parent / "instructions" / "episode0.json"
    )
    instruction.write_text(json.dumps({"seen": ["changed"]}), encoding="utf-8")
    with pytest.raises(DatasetContractError, match="episode/instruction"):
        validate(runtime)


def test_instruction_fallback_descriptor_change_fails(synthetic_runtime):
    runtime = synthetic_runtime
    fallback_child = runtime.children[1]
    fallback_child.prompt_task_name = "changed fallback"
    with pytest.raises(DatasetContractError, match="episode/instruction"):
        validate(runtime)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("height", 416),
        ("camera_layout", ["head_camera", "right_camera", "left_camera"]),
        ("action_mode", "joint"),
        ("normalize_mode", None),
        ("video_stride", 2),
        ("delta_action", True),
    ],
)
def test_runtime_preprocessing_change_fails(synthetic_runtime, attribute, value):
    runtime = synthetic_runtime
    for child in runtime.children:
        setattr(child, attribute, deepcopy(value))
    with pytest.raises(DatasetContractError):
        validate(runtime)


def test_config_only_preprocessing_change_fails(synthetic_runtime):
    runtime = synthetic_runtime
    config = dict(runtime.config)
    config["val_ratio"] = 0.1
    with pytest.raises(DatasetContractError, match="preprocessing"):
        validate(runtime, config=config)


def test_source_file_mutation_fails(synthetic_runtime):
    runtime = synthetic_runtime
    runtime.sources["sana_wam.dataloader.transforms.rotation"].write_bytes(
        b"# changed source\n"
    )
    with pytest.raises(DatasetContractError, match="preprocessing"):
        validate(runtime)


def test_external_and_embedded_sha_pins_fail_closed(synthetic_runtime):
    runtime = synthetic_runtime
    data = runtime.contract.to_artifact_bytes()
    digest = hashlib.sha256(data).hexdigest()
    Phase6DatasetContract.from_artifact_bytes(
        data,
        expected_artifact_sha256=digest,
        expected_plan_sha256=runtime.plan.plan_sha256,
        expected_identity_sha256=runtime.plan.identity_sha256,
        expected_action_stats_sha256=runtime.action_stats_sha256,
    )
    for field, value in (
        ("expected_artifact_sha256", "a" * 64),
        ("expected_plan_sha256", "a" * 64),
        ("expected_identity_sha256", "a" * 64),
        ("expected_action_stats_sha256", "a" * 64),
    ):
        kwargs = {
            "expected_artifact_sha256": digest,
            "expected_plan_sha256": runtime.plan.plan_sha256,
            "expected_identity_sha256": runtime.plan.identity_sha256,
            "expected_action_stats_sha256": runtime.action_stats_sha256,
        }
        kwargs[field] = value
        with pytest.raises(DatasetContractError):
            Phase6DatasetContract.from_artifact_bytes(data, **kwargs)


def test_action_stats_runtime_pin_mismatch_fails(synthetic_runtime):
    runtime = synthetic_runtime
    with pytest.raises(DatasetContractError, match="action-stats"):
        validate(runtime, action_stats_sha256="a" * 64)


def test_duplicate_key_noncanonical_and_bool_as_int_are_rejected(synthetic_runtime):
    runtime = synthetic_runtime
    data = runtime.contract.to_artifact_bytes()
    duplicate = data.replace(
        b'{"action_stats_sha256":',
        b'{"schema_version":"duplicate","action_stats_sha256":',
        1,
    )
    with pytest.raises(DatasetContractError):
        Phase6DatasetContract.from_artifact_bytes(duplicate)
    with pytest.raises(DatasetContractError, match="canonical"):
        Phase6DatasetContract.from_artifact_bytes(b"  " + data)

    value = runtime.contract.to_dict()
    value["rows"][0]["dataset_index"] = True
    with pytest.raises(DatasetContractError, match="integer"):
        Phase6DatasetContract.from_artifact_bytes(canonical_bytes(value))

    value = runtime.contract.to_dict()
    value["rows"][0]["window_logical_length"] = 114
    with pytest.raises(DatasetContractError, match="exceeds"):
        Phase6DatasetContract.from_artifact_bytes(canonical_bytes(value))


def test_runtime_window_row_is_the_per_access_logical_guard(synthetic_runtime):
    runtime = synthetic_runtime
    observed = runtime_window_row(runtime.dataset, global_step=1, dataset_index=0)
    expected = runtime.contract.row_for_dataset_index(0)
    assert all(expected[key] == value for key, value in observed.items())
    runtime.children[0]._window_index[0] = (0, 0, 99)
    observed = runtime_window_row(runtime.dataset, global_step=1, dataset_index=0)
    assert observed["window_logical_length"] == 99
    assert expected["window_logical_length"] == 113
