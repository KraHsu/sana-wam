#!/usr/bin/env python3
"""Independently rebind and spot-materialize the frozen Phase-6 task plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256 = (
    "a659b602fbba11223b8d5129a82e8699bfd99b2cc42b262d154cd9b11504b116"
)
EXPECTED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256 = (
    "8632450705ee3ba35855f07ccb48ad300aac5abf711abf0307808b52fb613081"
)
EXPECTED_REGISTRY_SHA256 = (
    "9166e1ecde74f14d7d69ae2f5072b10732b01b9f3b0004c438d0e97ed94b00b1"
)
EXPECTED_ACTION_STATS_SHA256 = (
    "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"
)
EXPECTED_TASK_COUNT = 42
EXPECTED_RAW_CANDIDATE_TOTAL = 467_038
EXPECTED_ELIGIBLE_CANDIDATE_TOTAL = 460_738
EXPECTED_EXCLUDED_CANDIDATE_TOTAL = 6_300
EXPECTED_EXCLUDED_PER_TASK = 150
EXPECTED_ROW_COUNT = 504
SUPPORT_COUNT_KEYS = (
    "actual_valid_raw_frames",
    "valid_sampled_video_frames",
    "valid_latent_frames",
    "nonbootstrap_valid_latent_frames",
    "eligible_chunk_count",
)
MANIFEST_KEYS = frozenset(
    {
        "action_stats_path",
        "action_stats_sha256",
        "artifact_path",
        "artifact_sha256",
        "dataset_config",
        "dataset_global_window_count",
        "eligible_candidate_pool_counts",
        "eligible_candidate_total",
        "excluded_candidate_pool_counts",
        "excluded_candidate_total",
        "expansion_eligibility_amendment_sha256",
        "expansion_support_contract_sha256",
        "identity_sha256",
        "jsonl_path",
        "jsonl_sha256",
        "optimizer_steps",
        "plan_sha256",
        "raw_candidate_pool_counts",
        "raw_candidate_total",
        "registry_path",
        "registry_sha256",
        "row_count",
        "schema_version",
        "selected_minimum_support",
        "training_or_gpu_started",
    }
)
EXPECTED_DATASET_CONFIG = {
    "action_mode": "eef",
    "causal_temporal": True,
    "delta_action": False,
    "filter_static_segments": False,
    "growing_history": False,
    "height": 384,
    "multiview": True,
    "normalize_mode": "min-max",
    "num_frames": 113,
    "repeat": 1,
    "robot": "aloha-agilex",
    "split": "train",
    "temporal_compression": 4,
    "text_embedding_dropout": 0.0,
    "val_ratio": 0.0,
    "variant": "clean_50",
    "video_stride": 4,
    "width": 320,
    "window_stride": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(token: str):
    raise ValueError(f"non-finite JSON token: {token}")


def _parse_manifest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    try:
        value = json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeError, TypeError) as exc:
        raise ValueError(f"invalid task-plan build manifest: {exc}") from exc
    if not isinstance(value, dict) or set(value) != MANIFEST_KEYS:
        raise ValueError("task-plan build manifest must have the exact v2 key set")
    if data != _canonical(value):
        raise ValueError("task-plan build manifest is not canonical JSON")
    return value


def _require_sha(value: Any, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _require_int(value: Any, name: str, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} must be the integer {expected}")
    return value


def _require_false(value: Any, name: str) -> None:
    if type(value) is not bool or value is not False:
        raise ValueError(f"{name} must be boolean false")


def _require_exact_config(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(EXPECTED_DATASET_CONFIG):
        raise ValueError("dataset_config has unexpected keys")
    for key, expected in EXPECTED_DATASET_CONFIG.items():
        observed = value[key]
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(f"dataset_config.{key} differs from frozen value")


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _require_support_types(value: Any, name: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        *SUPPORT_COUNT_KEYS,
        "eligible",
    }:
        raise ValueError(f"{name} has unexpected keys")
    for key in SUPPORT_COUNT_KEYS:
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"{name}.{key} must be a non-negative integer")
    if type(value["eligible"]) is not bool:
        raise ValueError(f"{name}.eligible must be a boolean")


def _require_count_map(value: Any, tasks: tuple[str, ...], name: str) -> None:
    if not isinstance(value, dict) or set(value) != set(tasks):
        raise ValueError(f"{name} task keys differ from the frozen cohort")
    for task in tasks:
        count = value[task]
        if type(count) is not int or count < 0:
            raise ValueError(f"{name}[{task!r}] must be a non-negative integer")


def _validate_candidate_counts(
    *,
    raw: Any,
    eligible: Any,
    excluded: Any,
    raw_total: Any,
    eligible_total: Any,
    excluded_total: Any,
    dataset_count: Any,
    tasks: tuple[str, ...],
    name: str,
) -> None:
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ValueError(f"{name} must bind exactly 42 tasks")
    _require_count_map(raw, tasks, f"{name}.raw")
    _require_count_map(eligible, tasks, f"{name}.eligible")
    _require_count_map(excluded, tasks, f"{name}.excluded")
    _require_int(raw_total, f"{name}.raw_total", EXPECTED_RAW_CANDIDATE_TOTAL)
    _require_int(
        eligible_total,
        f"{name}.eligible_total",
        EXPECTED_ELIGIBLE_CANDIDATE_TOTAL,
    )
    _require_int(
        excluded_total,
        f"{name}.excluded_total",
        EXPECTED_EXCLUDED_CANDIDATE_TOTAL,
    )
    _require_int(
        dataset_count,
        f"{name}.dataset_count",
        EXPECTED_RAW_CANDIDATE_TOTAL,
    )
    if (
        sum(raw.values()) != EXPECTED_RAW_CANDIDATE_TOTAL
        or sum(eligible.values()) != EXPECTED_ELIGIBLE_CANDIDATE_TOTAL
        or sum(excluded.values()) != EXPECTED_EXCLUDED_CANDIDATE_TOTAL
    ):
        raise ValueError(f"{name} totals differ from the frozen eligibility amendment")
    if any(
        excluded[task] != EXPECTED_EXCLUDED_PER_TASK
        or raw[task] != eligible[task] + excluded[task]
        for task in tasks
    ):
        raise ValueError(
            f"{name} per-task counts differ from the frozen eligibility amendment"
        )


def _require_frozen_expansion_geometry(child: Any) -> None:
    sample_indices = child._video_sample_indices
    if (
        type(child._raw_window_len) is not int
        or child._raw_window_len != 113
        or type(child.video_stride) is not int
        or child.video_stride != 4
        or type(sample_indices) is not list
        or any(type(index) is not int for index in sample_indices)
        or sample_indices != list(range(0, 113, 4))
        or type(child.causal_temporal) is not bool
        or child.causal_temporal is not True
        or type(child.temporal_compression) is not int
        or child.temporal_compression != 4
    ):
        raise ValueError("runtime dataset differs from frozen expansion geometry")


def _independent_support(child: Any, local_index: int) -> dict[str, int | bool]:
    _require_frozen_expansion_geometry(child)
    episode_index, start_frame, logical_length = child._resolve_local_index(local_index)
    if any(
        type(value) is not int or value < 0
        for value in (episode_index, start_frame, logical_length)
    ) or logical_length > 113:
        raise ValueError("runtime window metadata differs from frozen integer bounds")
    episode_lengths = child._episode_lengths
    if episode_index >= len(episode_lengths):
        raise ValueError("runtime episode index exceeds episode-length metadata")
    episode_length = episode_lengths[episode_index]
    if type(episode_length) is not int or episode_length < 1:
        raise ValueError("runtime episode length must be a positive integer")
    actual_valid = min(logical_length, max(0, episode_length - start_frame))
    sampled = [index < actual_valid for index in range(0, 113, 4)]
    latent = [sampled[0]] + [
        any(sampled[offset : offset + 4]) for offset in range(1, 29, 4)
    ]
    nonbootstrap = list(latent)
    nonbootstrap[0] = False
    chunks = [any(nonbootstrap[offset : offset + 2]) for offset in range(0, 8, 2)]
    eligible_chunk_count = sum(chunks)
    return {
        "actual_valid_raw_frames": actual_valid,
        "valid_sampled_video_frames": sum(sampled),
        "valid_latent_frames": sum(latent),
        "nonbootstrap_valid_latent_frames": sum(nonbootstrap),
        "eligible_chunk_count": eligible_chunk_count,
        "eligible": eligible_chunk_count >= 1,
    }


def _independent_candidate_pools(dataset: Any, contract: Any):
    pools = {task: [] for task in contract.train_tasks}
    raw_counts = {task: 0 for task in contract.train_tasks}
    excluded_counts = {task: 0 for task in contract.train_tasks}
    for dataset_index in range(len(dataset)):
        child_index, local_index = dataset.resolve_global_index(dataset_index)
        child = dataset._sub_datasets[child_index]
        task = child.task_name
        raw_counts[task] += 1
        support = _independent_support(child, local_index)
        if not support["eligible"]:
            excluded_counts[task] += 1
            continue
        pools[task].append(
            dataset.phase6_identity_at(
                dataset_index,
                protocol_seed=contract.protocol_seed,
            )
        )
    return pools, raw_counts, excluded_counts


def _validate_manifest_provenance(
    manifest: dict[str, Any],
    *,
    tasks: tuple[str, ...],
    registry_path: Path,
    action_stats_path: Path,
    artifact_path: Path,
    jsonl_path: Path,
) -> None:
    if manifest["schema_version"] != "sana-phase6-task-plan-build-manifest-v2":
        raise ValueError("task-plan build-manifest schema mismatch")
    _require_exact_config(manifest["dataset_config"])
    _require_support_types(
        manifest["selected_minimum_support"],
        "selected_minimum_support",
    )
    if (
        manifest["selected_minimum_support"]["eligible"] is not True
        or manifest["selected_minimum_support"]["eligible_chunk_count"] < 1
    ):
        raise ValueError("selected_minimum_support must be structurally eligible")
    _require_int(manifest["optimizer_steps"], "optimizer_steps", 0)
    _require_int(manifest["row_count"], "row_count", EXPECTED_ROW_COUNT)
    _require_false(
        manifest["training_or_gpu_started"],
        "training_or_gpu_started",
    )
    _validate_candidate_counts(
        raw=manifest["raw_candidate_pool_counts"],
        eligible=manifest["eligible_candidate_pool_counts"],
        excluded=manifest["excluded_candidate_pool_counts"],
        raw_total=manifest["raw_candidate_total"],
        eligible_total=manifest["eligible_candidate_total"],
        excluded_total=manifest["excluded_candidate_total"],
        dataset_count=manifest["dataset_global_window_count"],
        tasks=tasks,
        name="manifest candidate pools",
    )

    expected_paths = {
        "registry_path": registry_path,
        "action_stats_path": action_stats_path,
        "artifact_path": artifact_path,
        "jsonl_path": jsonl_path,
    }
    for key, expected_path in expected_paths.items():
        if type(manifest[key]) is not str or manifest[key] != str(expected_path):
            raise ValueError(f"{key} differs from the independently resolved path")

    for key in (
        "registry_sha256",
        "action_stats_sha256",
        "artifact_sha256",
        "jsonl_sha256",
        "plan_sha256",
        "identity_sha256",
        "expansion_support_contract_sha256",
        "expansion_eligibility_amendment_sha256",
    ):
        _require_sha(manifest[key], key)
    if manifest["registry_sha256"] != EXPECTED_REGISTRY_SHA256:
        raise ValueError("registry_sha256 differs from the frozen registry")
    if manifest["action_stats_sha256"] != EXPECTED_ACTION_STATS_SHA256:
        raise ValueError("action_stats_sha256 differs from the frozen action stats")
    if (
        manifest["expansion_support_contract_sha256"]
        != EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256
        or manifest["expansion_eligibility_amendment_sha256"]
        != EXPECTED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
    ):
        raise ValueError("expansion-support provenance differs from frozen pins")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from sana_wam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset
    from sana_wam.dataloader.task_sample_plan import (
        PlanSampler,
        SamplerContract,
        TaskRoundRobinPlan,
    )

    args = parse_args()
    registry_path = args.registry.expanduser().resolve(strict=True)
    action_stats_path = args.action_stats.expanduser().resolve(strict=True)
    artifact_path = args.plan.expanduser().resolve(strict=True)
    jsonl_path = args.jsonl.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    if _sha256(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("registry file differs from the frozen SHA256")
    if _sha256(action_stats_path) != EXPECTED_ACTION_STATS_SHA256:
        raise ValueError("action-stats file differs from the frozen SHA256")
    contract = SamplerContract.from_registry(registry_path)
    manifest = _parse_manifest(manifest_path)
    _validate_manifest_provenance(
        manifest,
        tasks=contract.train_tasks,
        registry_path=registry_path,
        action_stats_path=action_stats_path,
        artifact_path=artifact_path,
        jsonl_path=jsonl_path,
    )

    plan_bytes = artifact_path.read_bytes()
    if _sha256(artifact_path) != manifest["artifact_sha256"]:
        raise ValueError("plan artifact file SHA256 differs from build manifest")
    plan = TaskRoundRobinPlan.from_artifact_bytes(
        plan_bytes,
        expected_plan_sha256=manifest["plan_sha256"],
        expected_identity_sha256=manifest["identity_sha256"],
    )
    if len(plan.rows) != manifest["row_count"]:
        raise ValueError("plan row count differs from build manifest")
    jsonl_bytes = jsonl_path.read_bytes()
    if _sha256(jsonl_path) != manifest["jsonl_sha256"]:
        raise ValueError("plan JSONL SHA256 differs from build manifest")
    if jsonl_bytes != plan.canonical_jsonl_bytes():
        raise ValueError("plan JSONL bytes differ from the reconstructed plan")
    config = manifest["dataset_config"]
    dataset = MultiTaskRoboTwinDataset(
        dataset_dir=str(args.dataset_dir.expanduser().resolve(strict=True)),
        robot=config["robot"],
        variant=config["variant"],
        tasks=list(contract.train_tasks),
        task_name=None,
        action_stats_path=str(action_stats_path),
        action_mode=config["action_mode"],
        num_frames=config["num_frames"],
        video_stride=config["video_stride"],
        height=config["height"],
        width=config["width"],
        split=config["split"],
        val_ratio=config["val_ratio"],
        repeat=config["repeat"],
        window_stride=config["window_stride"],
        multiview=config["multiview"],
        camera_layout=["head_camera", "left_camera", "right_camera"],
        target_camera="head_camera",
        normalize_mode=config["normalize_mode"],
        filter_static_segments=config["filter_static_segments"],
        text_embedding_cache_dir=None,
        text_embedding_dropout=config["text_embedding_dropout"],
        vae_cache_dir=None,
        temporal_compression=config["temporal_compression"],
        causal_temporal=config["causal_temporal"],
        growing_history=config["growing_history"],
        delta_action=config["delta_action"],
    )
    pools, raw_counts, excluded_counts = _independent_candidate_pools(
        dataset, contract
    )
    eligible_counts = {task: len(pools[task]) for task in contract.train_tasks}
    _validate_candidate_counts(
        raw=raw_counts,
        eligible=eligible_counts,
        excluded=excluded_counts,
        raw_total=sum(raw_counts.values()),
        eligible_total=sum(eligible_counts.values()),
        excluded_total=sum(excluded_counts.values()),
        dataset_count=len(dataset),
        tasks=contract.train_tasks,
        name="independently observed candidate pools",
    )
    if (
        raw_counts != manifest["raw_candidate_pool_counts"]
        or eligible_counts != manifest["eligible_candidate_pool_counts"]
        or excluded_counts != manifest["excluded_candidate_pool_counts"]
        or sum(raw_counts.values()) != manifest["raw_candidate_total"]
        or sum(eligible_counts.values()) != manifest["eligible_candidate_total"]
        or sum(excluded_counts.values()) != manifest["excluded_candidate_total"]
    ):
        raise RuntimeError("candidate-pool eligibility counts differ from build manifest")
    rebuilt = TaskRoundRobinPlan.build(contract, pools)
    if rebuilt.to_payload() != plan.to_payload():
        raise RuntimeError("clean eligible-pool rebuild differs from published plan")
    sampler = PlanSampler(
        plan,
        expected_plan_sha256=plan.plan_sha256,
        expected_identity_sha256=plan.identity_sha256,
    )
    indices = tuple(sampler)
    if len(indices) != 504 or len(set(indices)) != 504:
        raise RuntimeError("sampler does not contain 504 unique indices")

    materialized = []
    selected_support = []
    for row in plan.rows:
        child_index, local_index = dataset.resolve_global_index(
            row.identity.dataset_index
        )
        support = _independent_support(dataset._sub_datasets[child_index], local_index)
        if not support["eligible"]:
            raise RuntimeError("plan contains a structurally ineligible row")
        selected_support.append(support)
    selected_minimum_support = {
        key: min(int(support[key]) for support in selected_support)
        for key in SUPPORT_COUNT_KEYS
    }
    selected_minimum_support["eligible"] = True
    if selected_minimum_support != manifest["selected_minimum_support"]:
        raise RuntimeError("selected minimum support differs from build manifest")
    for row_index in (0, len(plan.rows) // 2, len(plan.rows) - 1):
        row = plan.rows[row_index]
        sample = dataset.phase6_get_planned_item(
            row.identity.dataset_index,
            planned_row=row,
            protocol_seed=plan.protocol_seed,
            plan_sha256=plan.plan_sha256,
            identity_sha256=plan.identity_sha256,
        )
        payload = sample["phase6_plan_row"]
        if payload["global_step"] != row.global_step:
            raise RuntimeError("materialized global step changed")
        materialized.append(
            {
                "dataset_index": row.identity.dataset_index,
                "episode_path": sample["episode_path"],
                "global_step": row.global_step,
                "prompt": sample["prompt"],
                "start_frame": sample["start_frame"],
                "task_name": sample["task_name"],
            }
        )
    result = {
        "artifact_sha256": _sha256(args.plan),
        "dataset_global_window_count": len(dataset),
        "eligible_candidate_pool_counts": eligible_counts,
        "eligible_candidate_total": sum(eligible_counts.values()),
        "excluded_candidate_pool_counts": excluded_counts,
        "excluded_candidate_total": sum(excluded_counts.values()),
        "expansion_eligibility_amendment_sha256": (
            EXPECTED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        ),
        "expansion_support_contract_sha256": (
            EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256
        ),
        "identity_sha256": plan.identity_sha256,
        "materialized_spot_checks": materialized,
        "plan_sha256": plan.plan_sha256,
        "registry_sha256": _sha256(args.registry),
        "sampler_unique_indices": len(set(indices)),
        "raw_candidate_pool_counts": raw_counts,
        "raw_candidate_total": sum(raw_counts.values()),
        "schema_version": "sana-phase6-task-plan-verification-v2",
        "selected_minimum_support": selected_minimum_support,
        "training_or_gpu_started": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive(output, _canonical(result))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
