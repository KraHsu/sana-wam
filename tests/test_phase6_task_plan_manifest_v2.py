from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "verify_phase6_task_plan.py"
BUILDER_PATH = ROOT / "scripts" / "build_phase6_task_plan.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "phase6_task_plan_verifier_candidate",
        VERIFIER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _count_fixture(module):
    tasks = tuple(f"task_{index:02d}" for index in range(42))
    quotient, remainder = divmod(module.EXPECTED_ELIGIBLE_CANDIDATE_TOTAL, 42)
    eligible = {
        task: quotient + int(index < remainder) for index, task in enumerate(tasks)
    }
    excluded = {task: module.EXPECTED_EXCLUDED_PER_TASK for task in tasks}
    raw = {task: eligible[task] + excluded[task] for task in tasks}
    return tasks, raw, eligible, excluded


def _valid_manifest(module, tmp_path: Path):
    tasks, raw, eligible, excluded = _count_fixture(module)
    paths = {}
    for name in ("registry", "action_stats", "artifact", "jsonl"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths[name] = path.resolve()
    manifest = {
        "action_stats_path": str(paths["action_stats"]),
        "action_stats_sha256": module.EXPECTED_ACTION_STATS_SHA256,
        "artifact_path": str(paths["artifact"]),
        "artifact_sha256": "1" * 64,
        "dataset_config": dict(module.EXPECTED_DATASET_CONFIG),
        "dataset_global_window_count": module.EXPECTED_RAW_CANDIDATE_TOTAL,
        "eligible_candidate_pool_counts": eligible,
        "eligible_candidate_total": module.EXPECTED_ELIGIBLE_CANDIDATE_TOTAL,
        "excluded_candidate_pool_counts": excluded,
        "excluded_candidate_total": module.EXPECTED_EXCLUDED_CANDIDATE_TOTAL,
        "expansion_eligibility_amendment_sha256": (
            module.EXPECTED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        ),
        "expansion_support_contract_sha256": (
            module.EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256
        ),
        "identity_sha256": "2" * 64,
        "jsonl_path": str(paths["jsonl"]),
        "jsonl_sha256": "3" * 64,
        "optimizer_steps": 0,
        "plan_sha256": "4" * 64,
        "raw_candidate_pool_counts": raw,
        "raw_candidate_total": module.EXPECTED_RAW_CANDIDATE_TOTAL,
        "registry_path": str(paths["registry"]),
        "registry_sha256": module.EXPECTED_REGISTRY_SHA256,
        "row_count": module.EXPECTED_ROW_COUNT,
        "schema_version": "sana-phase6-task-plan-build-manifest-v2",
        "selected_minimum_support": {
            "actual_valid_raw_frames": 5,
            "valid_sampled_video_frames": 2,
            "valid_latent_frames": 2,
            "nonbootstrap_valid_latent_frames": 1,
            "eligible_chunk_count": 1,
            "eligible": True,
        },
        "training_or_gpu_started": False,
    }
    assert len(manifest) == 25
    return tasks, manifest, paths


def _validate(module, manifest, tasks, paths):
    module._validate_manifest_provenance(
        manifest,
        tasks=tasks,
        registry_path=paths["registry"],
        action_stats_path=paths["action_stats"],
        artifact_path=paths["artifact"],
        jsonl_path=paths["jsonl"],
    )


def test_manifest_parser_requires_canonical_exact_25_key_schema(tmp_path) -> None:
    module = _load_verifier()
    tasks, manifest, paths = _valid_manifest(module, tmp_path)
    artifact = tmp_path / "manifest.json"
    artifact.write_bytes(module._canonical(manifest))
    parsed = module._parse_manifest(artifact)
    assert parsed == manifest
    _validate(module, parsed, tasks, paths)

    for alias_value in (
        manifest["eligible_candidate_pool_counts"],
        {"divergent": 1},
    ):
        with_alias = dict(manifest)
        with_alias["candidate_pool_counts"] = alias_value
        artifact.write_bytes(module._canonical(with_alias))
        with pytest.raises(ValueError, match="exact v2 key set"):
            module._parse_manifest(artifact)


def test_builder_and_verifier_share_the_exact_25_key_manifest_schema() -> None:
    verifier = _load_verifier()
    tree = ast.parse(BUILDER_PATH.read_text(encoding="utf-8"))
    manifest_dict = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "manifest"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    )
    builder_keys = frozenset(key.value for key in manifest_dict.keys)
    assert len(builder_keys) == 25
    assert builder_keys == verifier.MANIFEST_KEYS
    assert "candidate_pool_counts" not in builder_keys


def test_manifest_parser_rejects_duplicate_nonfinite_and_noncanonical_json(
    tmp_path,
) -> None:
    module = _load_verifier()
    _tasks, manifest, _paths = _valid_manifest(module, tmp_path)
    artifact = tmp_path / "manifest.json"
    canonical = module._canonical(manifest)

    duplicate = canonical.rstrip()[:-1] + b',"row_count":504}\n'
    artifact.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        module._parse_manifest(artifact)

    artifact.write_bytes(canonical.replace(b'"row_count":504', b'"row_count":NaN'))
    with pytest.raises(ValueError, match="non-finite"):
        module._parse_manifest(artifact)

    artifact.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        module._parse_manifest(artifact)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("raw_candidate_total", 467_038.0),
        ("eligible_candidate_total", 460_738.0),
        ("excluded_candidate_total", 6_300.0),
        ("dataset_global_window_count", 467_038.0),
        ("row_count", 504.0),
        ("optimizer_steps", False),
        ("training_or_gpu_started", 0),
    ),
)
def test_manifest_provenance_rejects_same_value_float_bool_aliases(
    tmp_path,
    field,
    replacement,
) -> None:
    module = _load_verifier()
    tasks, manifest, paths = _valid_manifest(module, tmp_path)
    manifest[field] = replacement
    with pytest.raises(ValueError):
        _validate(module, manifest, tasks, paths)


def test_manifest_rejects_nested_same_value_numeric_aliases(tmp_path) -> None:
    module = _load_verifier()
    tasks, manifest, paths = _valid_manifest(module, tmp_path)
    task = tasks[0]
    mutations = (
        ("raw_candidate_pool_counts", task, float(manifest["raw_candidate_pool_counts"][task])),
        ("eligible_candidate_pool_counts", task, True),
        ("excluded_candidate_pool_counts", task, 150.0),
        ("selected_minimum_support", "eligible_chunk_count", True),
        ("selected_minimum_support", "valid_latent_frames", 2.0),
        ("dataset_config", "num_frames", 113.0),
        ("dataset_config", "causal_temporal", 1),
    )
    for mapping_name, key, replacement in mutations:
        mutated = dict(manifest)
        mutated[mapping_name] = dict(manifest[mapping_name])
        mutated[mapping_name][key] = replacement
        with pytest.raises(ValueError):
            _validate(module, mutated, tasks, paths)


@pytest.mark.parametrize("delta", (-1, 1))
def test_unselected_candidate_add_remove_cannot_hide_behind_same_top12(
    tmp_path,
    delta,
) -> None:
    module = _load_verifier()
    tasks, manifest, paths = _valid_manifest(module, tmp_path)
    task = tasks[-1]
    manifest["eligible_candidate_pool_counts"][task] += delta
    with pytest.raises(ValueError, match="eligibility amendment"):
        _validate(module, manifest, tasks, paths)


def test_manifest_binds_each_path_hash_and_execution_field(tmp_path) -> None:
    module = _load_verifier()
    tasks, manifest, paths = _valid_manifest(module, tmp_path)
    mutations = {
        "registry_path": str(tmp_path / "other-registry"),
        "action_stats_path": str(tmp_path / "other-stats"),
        "artifact_path": str(tmp_path / "other-artifact"),
        "jsonl_path": str(tmp_path / "other-jsonl"),
        "registry_sha256": "0" * 64,
        "action_stats_sha256": "0" * 64,
        "artifact_sha256": 1,
        "jsonl_sha256": True,
        "plan_sha256": "not-a-sha",
        "identity_sha256": "A" * 64,
    }
    for key, replacement in mutations.items():
        mutated = dict(manifest)
        mutated[key] = replacement
        with pytest.raises(ValueError):
            _validate(module, mutated, tasks, paths)


def _load_builder_count_guard():
    tree = ast.parse(BUILDER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validated_candidate_pool_counts"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "EXPECTED_TASK_COUNT": 42,
        "EXPECTED_RAW_CANDIDATE_TOTAL": 467_038,
        "EXPECTED_ELIGIBLE_CANDIDATE_TOTAL": 460_738,
        "EXPECTED_EXCLUDED_CANDIDATE_TOTAL": 6_300,
        "EXPECTED_EXCLUDED_PER_TASK": 150,
    }
    exec(compile(module, str(BUILDER_PATH), "exec"), namespace)
    return namespace["_validated_candidate_pool_counts"]


class _LengthOnly:
    def __init__(self, length: int):
        self.length = length

    def __len__(self):
        return self.length


class _FakeDataset:
    def __init__(self, children):
        self._sub_datasets = children

    def __len__(self):
        return sum(len(item) for item in self._sub_datasets)


def test_builder_authenticates_full_pool_before_hash_ordering() -> None:
    verifier = _load_verifier()
    tasks, raw, eligible, _excluded = _count_fixture(verifier)
    dataset = _FakeDataset([_LengthOnly(raw[task]) for task in tasks])
    contract = SimpleNamespace(train_tasks=tasks)
    pools = {task: _LengthOnly(eligible[task]) for task in tasks}
    guard = _load_builder_count_guard()
    guard(dataset, contract, pools)

    for delta in (-1, 1):
        mutated = dict(pools)
        mutated[tasks[0]] = _LengthOnly(eligible[tasks[0]] + delta)
        with pytest.raises(ValueError, match="eligibility amendment"):
            guard(dataset, contract, mutated)

    source = BUILDER_PATH.read_text(encoding="utf-8")
    assert source.index("_validated_candidate_pool_counts(dataset, contract, pools)") < (
        source.index("TaskRoundRobinPlan.build(contract, pools)")
    )
