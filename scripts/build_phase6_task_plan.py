#!/usr/bin/env python3
"""Build the frozen Phase-6 504-row plan from RoboTwin metadata only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sana_wam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset
from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    validate_expansion_support,
)
from sana_wam.dataloader.robotwin_plan_binding import phase6_expansion_support_at
from sana_wam.dataloader.task_sample_plan import (
    SamplerContract,
    TaskRoundRobinPlan,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _canonical_json_bytes(value: Any) -> bytes:
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


def _validated_candidate_pool_counts(dataset, contract, pools):
    tasks = contract.train_tasks
    children = dataset._sub_datasets
    if len(tasks) != EXPECTED_TASK_COUNT or len(children) != EXPECTED_TASK_COUNT:
        raise ValueError("candidate-pool cohort must contain exactly 42 tasks")
    if set(pools) != set(tasks):
        raise ValueError("eligible candidate pools differ from the frozen task cohort")
    raw = {task: len(child) for task, child in zip(tasks, children)}
    eligible = {task: len(pools[task]) for task in tasks}
    excluded = {task: raw[task] - eligible[task] for task in tasks}
    if any(
        raw[task] != eligible[task] + excluded[task]
        or excluded[task] != EXPECTED_EXCLUDED_PER_TASK
        for task in tasks
    ):
        raise ValueError(
            "per-task candidate counts differ from the frozen eligibility amendment"
        )
    if (
        len(dataset) != EXPECTED_RAW_CANDIDATE_TOTAL
        or sum(raw.values()) != EXPECTED_RAW_CANDIDATE_TOTAL
        or sum(eligible.values()) != EXPECTED_ELIGIBLE_CANDIDATE_TOTAL
        or sum(excluded.values()) != EXPECTED_EXCLUDED_CANDIDATE_TOTAL
    ):
        raise ValueError(
            "candidate totals differ from the frozen eligibility amendment"
        )
    return raw, eligible, excluded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=113)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.num_frames,
        args.video_stride,
        args.height,
        args.width,
    ) != (113, 4, 384, 320):
        raise ValueError("task-plan builder geometry differs from the frozen config")
    registry = args.registry.resolve()
    action_stats = args.action_stats.resolve()
    if _sha256_file(registry) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("registry SHA256 differs from the frozen Phase-6 registry")
    if _sha256_file(action_stats) != EXPECTED_ACTION_STATS_SHA256:
        raise ValueError("action-stats SHA256 differs from the frozen contract")
    contract = SamplerContract.from_registry(registry)

    dataset_config = {
        "action_mode": "eef",
        "causal_temporal": True,
        "delta_action": False,
        "filter_static_segments": False,
        "growing_history": False,
        "height": args.height,
        "multiview": True,
        "normalize_mode": "min-max",
        "num_frames": args.num_frames,
        "repeat": 1,
        "robot": "aloha-agilex",
        "split": "train",
        "temporal_compression": 4,
        "text_embedding_dropout": 0.0,
        "val_ratio": 0.0,
        "variant": "clean_50",
        "video_stride": args.video_stride,
        "width": args.width,
        "window_stride": 1,
    }
    dataset = MultiTaskRoboTwinDataset(
        dataset_dir=str(args.dataset_dir.resolve()),
        robot=dataset_config["robot"],
        variant=dataset_config["variant"],
        tasks=list(contract.train_tasks),
        task_name=None,
        action_stats_path=str(action_stats),
        action_mode=dataset_config["action_mode"],
        num_frames=dataset_config["num_frames"],
        video_stride=dataset_config["video_stride"],
        height=dataset_config["height"],
        width=dataset_config["width"],
        split=dataset_config["split"],
        val_ratio=dataset_config["val_ratio"],
        repeat=dataset_config["repeat"],
        window_stride=dataset_config["window_stride"],
        multiview=dataset_config["multiview"],
        camera_layout=["head_camera", "left_camera", "right_camera"],
        target_camera="head_camera",
        normalize_mode=dataset_config["normalize_mode"],
        filter_static_segments=dataset_config["filter_static_segments"],
        text_embedding_cache_dir=None,
        text_embedding_dropout=dataset_config["text_embedding_dropout"],
        vae_cache_dir=None,
        temporal_compression=dataset_config["temporal_compression"],
        causal_temporal=dataset_config["causal_temporal"],
        growing_history=dataset_config["growing_history"],
        delta_action=dataset_config["delta_action"],
    )
    pools = dataset.phase6_candidate_pools(contract, dataset_type="robotwin")
    raw_pool_counts, eligible_pool_counts, excluded_pool_counts = (
        _validated_candidate_pool_counts(dataset, contract, pools)
    )
    # Hash ordering is allowed only after the complete shared pool is authenticated.
    plan = TaskRoundRobinPlan.build(contract, pools)
    selected_support = [
        phase6_expansion_support_at(dataset, row.identity.dataset_index)
        for row in plan.rows
    ]
    for support in selected_support:
        validate_expansion_support(support, require_eligible=True)
    support_count_keys = (
        "actual_valid_raw_frames",
        "valid_sampled_video_frames",
        "valid_latent_frames",
        "nonbootstrap_valid_latent_frames",
        "eligible_chunk_count",
    )
    selected_minimum_support = {
        key: min(support[key] for support in selected_support)
        for key in support_count_keys
    }
    selected_minimum_support["eligible"] = all(
        support["eligible"] is True for support in selected_support
    )
    artifact_bytes = plan.to_artifact_bytes()
    jsonl_bytes = plan.canonical_jsonl_bytes()

    # Reparse before publishing so serialization and embedded hashes fail closed.
    TaskRoundRobinPlan.from_artifact_bytes(
        artifact_bytes,
        expected_plan_sha256=plan.plan_sha256,
        expected_identity_sha256=plan.identity_sha256,
    )
    artifact_path = args.output.resolve()
    jsonl_path = args.jsonl.resolve()
    manifest_path = args.manifest.resolve()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    jsonl_sha256 = hashlib.sha256(jsonl_bytes).hexdigest()
    manifest = {
        "action_stats_path": str(action_stats),
        "action_stats_sha256": EXPECTED_ACTION_STATS_SHA256,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "dataset_config": dataset_config,
        "dataset_global_window_count": len(dataset),
        "eligible_candidate_pool_counts": eligible_pool_counts,
        "eligible_candidate_total": sum(eligible_pool_counts.values()),
        "excluded_candidate_pool_counts": excluded_pool_counts,
        "excluded_candidate_total": sum(excluded_pool_counts.values()),
        "expansion_eligibility_amendment_sha256": (
            EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        ),
        "expansion_support_contract_sha256": EXPANSION_SUPPORT_CONTRACT_SHA256,
        "identity_sha256": plan.identity_sha256,
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": jsonl_sha256,
        "optimizer_steps": 0,
        "plan_sha256": plan.plan_sha256,
        "registry_path": str(registry),
        "registry_sha256": EXPECTED_REGISTRY_SHA256,
        "raw_candidate_pool_counts": raw_pool_counts,
        "raw_candidate_total": sum(raw_pool_counts.values()),
        "row_count": len(plan.rows),
        "schema_version": "sana-phase6-task-plan-build-manifest-v2",
        "selected_minimum_support": selected_minimum_support,
        "training_or_gpu_started": False,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    _exclusive_write(artifact_path, artifact_bytes)
    _exclusive_write(jsonl_path, jsonl_bytes)
    _exclusive_write(manifest_path, manifest_bytes)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
