#!/usr/bin/env python3
"""Build the second, byte-level artifact for the frozen Phase-6 plan.

This is a CPU-only metadata and hashing pass.  It never constructs the model,
loads a checkpoint, starts training, or materializes video/action tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import secrets
from typing import Any

from sana_wam.dataloader import robotwin_dataset as robotwin_dataset_module
from sana_wam.dataloader.phase6_dataset_contract import Phase6DatasetContract
from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    expansion_support_contract,
)
from sana_wam.dataloader.phase6_padding_semantics import (
    PADDING_SEMANTICS_AMENDMENT_SHA256,
    PADDING_SEMANTICS_CONTRACT_SHA256,
    padding_eligibility_dependency,
    padding_semantics_contract,
)
from sana_wam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset
from sana_wam.dataloader.transforms import multiview as multiview_module
from sana_wam.dataloader.transforms import normalize as normalize_module
from sana_wam.dataloader.transforms import rotation as rotation_module
from sana_wam.dataloader.task_sample_plan import (
    SamplerContract,
    TaskRoundRobinPlan,
)


EXPECTED_REGISTRY_SHA256 = (
    "9166e1ecde74f14d7d69ae2f5072b10732b01b9f3b0004c438d0e97ed94b00b1"
)
# Independently verified publication pins for the eligible 504-row rebuild.
EXPECTED_PLAN_ARTIFACT_SHA256 = (
    "7a1063df1d97fa0b8859dc86bf22f1ba3ad23703b6f6e449388dc950b8dc3ce6"
)
EXPECTED_PLAN_SHA256 = (
    "701d1436c9804df960d190586c264b80af42e8ce107751693e4c5adbe1089411"
)
EXPECTED_IDENTITY_SHA256 = (
    "08b9fcf418b0c4aabf7ea5e494cc8f603e63b797658c2c3890c2a06599a965dd"
)
EXPECTED_ACTION_STATS_SHA256 = (
    "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"
)
EXPECTED_PREPROCESSING_SOURCE_SHA256 = {
    "sana_wam.dataloader.robotwin_dataset": (
        "a7db73364f260b693e376f16783576d383ad88c3ee9f8258135acd91e81c1445"
    ),
    "sana_wam.dataloader.transforms.multiview": (
        "3064a85fde72971a804148402ac9228a0021c3b653528f9a890f52cc755e073e"
    ),
    "sana_wam.dataloader.transforms.normalize": (
        "91e8cb7988f090f1ff344dc4f899f0c8c2b27377168aec1ccb6037c2620b5651"
    ),
    "sana_wam.dataloader.transforms.rotation": (
        "f928676154ba81d760074e586e853e7e59e91ba6ad58637bcd4970e1d42c624b"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
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


def _atomic_exclusive_write(path: Path, data: bytes) -> None:
    """Publish complete bytes atomically and refuse an existing destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite {path}") from exc
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--plan-artifact", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _verify_frozen_file(path: Path, expected: str, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    actual = _sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{name} SHA256 mismatch: expected {expected}, got {actual}")
    return resolved


def main() -> None:
    args = parse_args()
    registry_path = _verify_frozen_file(
        args.registry, EXPECTED_REGISTRY_SHA256, "registry"
    )
    plan_path = _verify_frozen_file(
        args.plan_artifact, EXPECTED_PLAN_ARTIFACT_SHA256, "task-plan artifact"
    )
    action_stats_path = _verify_frozen_file(
        args.action_stats, EXPECTED_ACTION_STATS_SHA256, "action stats"
    )
    dataset_dir = args.dataset_dir.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve(strict=False)
    manifest_path = args.manifest.expanduser().resolve(strict=False)
    if output_path == manifest_path:
        raise ValueError("--output and --manifest must be different paths")
    for destination in (output_path, manifest_path):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")

    preprocessing_modules = {
        "sana_wam.dataloader.robotwin_dataset": robotwin_dataset_module,
        "sana_wam.dataloader.transforms.multiview": multiview_module,
        "sana_wam.dataloader.transforms.normalize": normalize_module,
        "sana_wam.dataloader.transforms.rotation": rotation_module,
    }
    preprocessing_source_paths = {}
    for source_id, module in preprocessing_modules.items():
        source = inspect.getsourcefile(module)
        if not source:
            raise RuntimeError(
                f"cannot locate imported preprocessing source {source_id}"
            )
        source_path = Path(source).resolve(strict=True)
        actual = _sha256_file(source_path)
        expected = EXPECTED_PREPROCESSING_SOURCE_SHA256[source_id]
        if actual != expected:
            raise ValueError(
                f"{source_id} SHA256 mismatch: expected {expected}, got {actual}"
            )
        preprocessing_source_paths[source_id] = str(source_path)

    sampler_contract = SamplerContract.from_registry(registry_path)
    plan_bytes = plan_path.read_bytes()
    plan = TaskRoundRobinPlan.from_artifact_bytes(
        plan_bytes,
        expected_plan_sha256=EXPECTED_PLAN_SHA256,
        expected_identity_sha256=EXPECTED_IDENTITY_SHA256,
    )
    dataloader_config = {
        "type": "robotwin",
        "dataset_dir": str(dataset_dir),
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
        "temporal_compression": 4,
        "text_embedding_dropout": 0.0,
        "val_ratio": 0.0,
        "variant": "clean_50",
        "video_stride": 4,
        "width": 320,
        "window_stride": 1,
        "camera_layout": ["head_camera", "left_camera", "right_camera"],
        "target_camera": "head_camera",
        "action_stats_path": str(action_stats_path),
        "vae_type": "wan",
        "history_min_frames": 1,
        "history_stride": 1,
        "gdn_chunk_size": 1,
        "static_segment_threshold": 1.0e-5,
        "max_static_retry": 3,
        "text_embedding_cache_dir": None,
        "vae_cache_dir": None,
        "seed": 42,
        "num_val_samples": 4,
        "backbone": None,
    }
    dataset = MultiTaskRoboTwinDataset(
        dataset_dir=str(dataset_dir),
        robot=dataloader_config["robot"],
        variant=dataloader_config["variant"],
        tasks=list(sampler_contract.train_tasks),
        task_name=None,
        action_stats_path=str(action_stats_path),
        action_mode=dataloader_config["action_mode"],
        normalize_mode=dataloader_config["normalize_mode"],
        num_frames=dataloader_config["num_frames"],
        height=dataloader_config["height"],
        width=dataloader_config["width"],
        split="train",
        val_ratio=dataloader_config["val_ratio"],
        repeat=dataloader_config["repeat"],
        target_camera=dataloader_config["target_camera"],
        window_stride=dataloader_config["window_stride"],
        video_stride=dataloader_config["video_stride"],
        multiview=dataloader_config["multiview"],
        camera_layout=dataloader_config["camera_layout"],
        backbone=dataloader_config["backbone"],
        filter_static_segments=dataloader_config["filter_static_segments"],
        static_segment_threshold=dataloader_config["static_segment_threshold"],
        max_static_retry=dataloader_config["max_static_retry"],
        text_embedding_cache_dir=None,
        text_embedding_dropout=dataloader_config["text_embedding_dropout"],
        vae_cache_dir=None,
        temporal_compression=dataloader_config["temporal_compression"],
        causal_temporal=dataloader_config["causal_temporal"],
        vae_type=dataloader_config["vae_type"],
        growing_history=dataloader_config["growing_history"],
        history_min_frames=dataloader_config["history_min_frames"],
        history_stride=dataloader_config["history_stride"],
        gdn_chunk_size=dataloader_config["gdn_chunk_size"],
        delta_action=dataloader_config["delta_action"],
        seed=dataloader_config["seed"],
        num_val_samples=dataloader_config["num_val_samples"],
    )
    dataset_contract = Phase6DatasetContract.build(
        dataset,
        plan,
        dataloader_config=dataloader_config,
        action_stats_sha256=EXPECTED_ACTION_STATS_SHA256,
        preprocessing_source_paths=preprocessing_source_paths,
        train_tasks=sampler_contract.train_tasks,
    )
    artifact_bytes = dataset_contract.to_artifact_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    reparsed = Phase6DatasetContract.from_artifact_bytes(
        artifact_bytes,
        expected_artifact_sha256=artifact_sha256,
        expected_plan_sha256=EXPECTED_PLAN_SHA256,
        expected_identity_sha256=EXPECTED_IDENTITY_SHA256,
        expected_action_stats_sha256=EXPECTED_ACTION_STATS_SHA256,
    )
    reparsed.validate_runtime(
        dataset,
        plan,
        dataloader_config=dataloader_config,
        action_stats_sha256=EXPECTED_ACTION_STATS_SHA256,
        preprocessing_source_paths=preprocessing_source_paths,
        train_tasks=sampler_contract.train_tasks,
    )
    support_count_keys = (
        "actual_valid_raw_frames",
        "valid_sampled_video_frames",
        "valid_latent_frames",
        "nonbootstrap_valid_latent_frames",
        "eligible_chunk_count",
    )
    selected_minimum_support = {
        key: min(int(row[key]) for row in reparsed.rows)
        for key in support_count_keys
    }
    selected_minimum_support["eligible"] = all(
        bool(row["eligible"]) for row in reparsed.rows
    )

    manifest = {
        "schema_version": "sana-phase6-dataset-contract-build-manifest-v2",
        "artifact_path": str(output_path),
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": len(artifact_bytes),
        "plan_artifact_path": str(plan_path),
        "plan_artifact_sha256": EXPECTED_PLAN_ARTIFACT_SHA256,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "identity_sha256": EXPECTED_IDENTITY_SHA256,
        "action_stats_path": str(action_stats_path),
        "action_stats_sha256": EXPECTED_ACTION_STATS_SHA256,
        "expansion_eligibility_amendment_sha256": (
            EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        ),
        "expansion_support_contract": expansion_support_contract(),
        "expansion_support_contract_sha256": EXPANSION_SUPPORT_CONTRACT_SHA256,
        "padding_semantics_amendment_sha256": (
            PADDING_SEMANTICS_AMENDMENT_SHA256
        ),
        "padding_semantics_contract": padding_semantics_contract(),
        "padding_semantics_contract_sha256": PADDING_SEMANTICS_CONTRACT_SHA256,
        "padding_eligibility_dependency": padding_eligibility_dependency(),
        "registry_path": str(registry_path),
        "registry_sha256": EXPECTED_REGISTRY_SHA256,
        "preprocessing_source_manifest": reparsed.preprocessing_contract[
            "preprocessing_source_manifest"
        ],
        "preprocessing_source_manifest_sha256": reparsed.preprocessing_contract[
            "preprocessing_source_manifest_sha256"
        ],
        "row_count": len(reparsed.rows),
        "deduplicated_episode_count": len(reparsed.episodes),
        "preprocessing_contract": reparsed.preprocessing_contract,
        "selected_minimum_support": selected_minimum_support,
        "training_or_gpu_started": False,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    _atomic_exclusive_write(output_path, artifact_bytes)
    try:
        _atomic_exclusive_write(manifest_path, manifest_bytes)
    except BaseException:
        # The artifact remains a valid immutable output; never silently overwrite
        # or delete a successfully published audit file.
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
