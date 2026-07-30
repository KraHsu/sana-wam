#!/usr/bin/env python3
"""Independent CPU verifier for the real Phase-6 dataset-contract artifact.

This script intentionally does not import ``phase6_dataset_contract`` or its
builder. It reconstructs the dataset, parses the artifact itself, rehashes all
selected sources, and publishes a separate canonical verification manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any

from sana_wam.dataloader import robotwin_dataset as robotwin_dataset_module
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
EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256 = (
    "a659b602fbba11223b8d5129a82e8699bfd99b2cc42b262d154cd9b11504b116"
)
EXPECTED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256 = (
    "8632450705ee3ba35855f07ccb48ad300aac5abf711abf0307808b52fb613081"
)
EXPECTED_PADDING_SEMANTICS_AMENDMENT_SHA256 = (
    "cf9bda2b5626c4c2a5b967c2d11673c0885e4e9aa2e54e81dafc5504e90bed5a"
)
EXPECTED_PADDING_SEMANTICS_CONTRACT_SHA256 = (
    "6215609e0e978d30e8ba4d34f763a3ebc55c7d1783a162c092113e9e302d153a"
)
EXPECTED_PADDING_ELIGIBILITY_DEPENDENCY = {
    "amendment_sha256": (
        "8632450705ee3ba35855f07ccb48ad300aac5abf711abf0307808b52fb613081"
    ),
    "support_contract_sha256": (
        "a659b602fbba11223b8d5129a82e8699bfd99b2cc42b262d154cd9b11504b116"
    ),
}
EXPECTED_PADDING_SEMANTICS_CONTRACT = {
    "action_key_rule": (
        "action token is a usable structured joint-AR key iff action_is_pad is "
        "false; apply to both noisy and clean copies at every layer"
    ),
    "action_output_rule": (
        "action losses and non-regression remain evaluated only on non-padded "
        "action tokens"
    ),
    "attention_rule": (
        "all clean-history states, adapter video-memory states, and within-frame "
        "noisy-key states exclude padded keys before numerator and denominator "
        "accumulation"
    ),
    "backward_compatibility_rule": (
        "outside a Phase-6 bound row, an absent modality mask means all keys of "
        "that modality are valid"
    ),
    "batch_rule": (
        "key validity is per sample and must not be collapsed into a batch-common "
        "mask"
    ),
    "phase6_mask_requirement": (
        "every Phase-6 plan row must provide boolean video_is_pad [B,T] and "
        "action_is_pad [B,Ta] with exact shapes; missing, stale, or non-boolean "
        "masks fail closed"
    ),
    "query_rule": (
        "padded query rows may be computed but are never usable as structured "
        "joint-AR keys in the same or any later layer"
    ),
    "schema_version": "sana-phase6-padding-semantics-contract-v1",
    "sequence_layout": "[video_noisy,video_clean,action_noisy,action_clean]",
    "temporal_mlp_kernel": (
        "GLUMBConvTemp t_kernel_size=3 within each frame_chunk_size=2 physical chunk"
    ),
    "temporal_mlp_rule": (
        "after per-frame spatial GLU and before temporal convolution, zero "
        "padded-frame states; apply the temporal convolution, then zero "
        "padded-frame outputs so valid states are invariant to padded latent values"
    ),
    "track_duplication_rule": (
        "the same per-modality validity mask applies to noisy and clean copies"
    ),
    "video_key_expansion_rule": (
        "repeat each latent-frame validity value over all h*w patch tokens before "
        "sequence concatenation"
    ),
    "video_key_rule": (
        "video patch token is a usable structured joint-AR key iff its source "
        "latent frame is not padded"
    ),
    "video_loss_rule": (
        "on-path video MSE is reduced only over non-padded latent frames and also "
        "excludes bootstrap latent0 when bootstrap_clean_prefix is enabled"
    ),
    "window_attention_rule": (
        "under frozen frame_chunk_size=2 and window_count=(1,1,4), each real "
        "latent frame occupies a distinct temporal window; validation must fail "
        "closed if a valid and padded latent frame could share a window"
    ),
}
EXPECTED_SOURCE_SHA256 = {
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
TOP_KEYS = {
    "schema_version",
    "plan_sha256",
    "identity_sha256",
    "action_stats_sha256",
    "expansion_eligibility_amendment_sha256",
    "expansion_support_contract",
    "expansion_support_contract_sha256",
    "padding_semantics_amendment_sha256",
    "padding_semantics_contract",
    "padding_semantics_contract_sha256",
    "preprocessing_contract",
    "rows",
    "episodes",
}
ROW_KEYS = {
    "global_step",
    "dataset_index",
    "task_name",
    "episode_index",
    "episode_path",
    "start_frame",
    "window_logical_length",
    "episode_sha256",
    "instruction_source_sha256",
    "actual_valid_raw_frames",
    "valid_sampled_video_frames",
    "valid_latent_frames",
    "nonbootstrap_valid_latent_frames",
    "eligible_chunk_count",
    "eligible",
}
EPISODE_KEYS = {
    "task_name",
    "episode_index",
    "episode_path",
    "resolved_path",
    "size_bytes",
    "sha256",
    "instruction_source",
}
INSTRUCTION_KEYS = {
    "kind",
    "path",
    "resolved_path",
    "size_bytes",
    "sha256",
    "fallback_descriptor",
}
SUPPORT_COUNT_KEYS = (
    "actual_valid_raw_frames",
    "valid_sampled_video_frames",
    "valid_latent_frames",
    "nonbootstrap_valid_latent_frames",
    "eligible_chunk_count",
)


def canonical_json_bytes(value: Any) -> bytes:
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


def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def parse_canonical_json(data: bytes, name: str):
    try:
        value = json.loads(
            data,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (json.JSONDecodeError, UnicodeError, TypeError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc
    if canonical_json_bytes(value) != data:
        raise ValueError(f"{name} is not canonical JSON")
    return value


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} is not a lowercase SHA256")
    return value


def require_int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def require_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has unexpected keys")


def verify_padding_provenance(value: Any) -> None:
    """Independently authenticate padding provenance and its dependency."""

    required_keys = {
        "expansion_eligibility_amendment_sha256",
        "expansion_support_contract_sha256",
        "padding_semantics_amendment_sha256",
        "padding_semantics_contract",
        "padding_semantics_contract_sha256",
    }
    if not isinstance(value, dict) or not required_keys.issubset(value):
        raise ValueError("padding provenance fields are missing")
    for key in required_keys - {"padding_semantics_contract"}:
        require_sha(value[key], key)

    expected_contract_sha = hashlib.sha256(
        canonical_json_bytes(EXPECTED_PADDING_SEMANTICS_CONTRACT)
    ).hexdigest()
    if expected_contract_sha != EXPECTED_PADDING_SEMANTICS_CONTRACT_SHA256:
        raise RuntimeError("independent padding-semantics literal has drifted")
    expected_dependency = {
        "amendment_sha256": EXPECTED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
        "support_contract_sha256": EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256,
    }
    if EXPECTED_PADDING_ELIGIBILITY_DEPENDENCY != expected_dependency:
        raise RuntimeError("independent padding/eligibility dependency has drifted")

    contract = value["padding_semantics_contract"]
    require_keys(
        contract,
        set(EXPECTED_PADDING_SEMANTICS_CONTRACT),
        "padding_semantics_contract",
    )
    if any(
        type(contract[key]) is not type(expected) or contract[key] != expected
        for key, expected in EXPECTED_PADDING_SEMANTICS_CONTRACT.items()
    ):
        raise ValueError("padding-semantics contract value or type mismatch")
    observed_contract_sha = hashlib.sha256(
        canonical_json_bytes(contract)
    ).hexdigest()
    if (
        observed_contract_sha != EXPECTED_PADDING_SEMANTICS_CONTRACT_SHA256
        or value["padding_semantics_contract_sha256"]
        != EXPECTED_PADDING_SEMANTICS_CONTRACT_SHA256
        or value["padding_semantics_amendment_sha256"]
        != EXPECTED_PADDING_SEMANTICS_AMENDMENT_SHA256
        or value["expansion_eligibility_amendment_sha256"]
        != EXPECTED_PADDING_ELIGIBILITY_DEPENDENCY["amendment_sha256"]
        or value["expansion_support_contract_sha256"]
        != EXPECTED_PADDING_ELIGIBILITY_DEPENDENCY["support_contract_sha256"]
    ):
        raise ValueError("padding-semantics provenance or dependency pin mismatch")


def require_support_types(value: Any, name: str) -> None:
    for key in SUPPORT_COUNT_KEYS:
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"{name}.{key} must be a non-negative integer")
    if type(value["eligible"]) is not bool:
        raise ValueError(f"{name}.eligible must be a boolean")


def require_frozen_expansion_geometry(child: Any) -> None:
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


def stable_file_descriptor(path: Path | str) -> tuple[str, int, str]:
    absolute = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(absolute):
        raise FileNotFoundError(absolute)
    resolved_before = os.path.realpath(absolute)
    before = os.stat(absolute)
    digest = sha256_file(absolute)
    after = os.stat(absolute)
    resolved_after = os.path.realpath(absolute)
    if (
        resolved_before,
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        resolved_after,
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"file changed while hashing: {absolute}")
    return resolved_before, before.st_size, digest


def verify_frozen_file(path: Path, digest: str, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    actual = sha256_file(resolved)
    if actual != digest:
        raise ValueError(f"{name} SHA mismatch: expected {digest}, got {actual}")
    return resolved


def atomic_exclusive_write(path: Path, data: bytes) -> None:
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
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def source_paths_and_manifest():
    modules = {
        "sana_wam.dataloader.robotwin_dataset": robotwin_dataset_module,
        "sana_wam.dataloader.transforms.multiview": multiview_module,
        "sana_wam.dataloader.transforms.normalize": normalize_module,
        "sana_wam.dataloader.transforms.rotation": rotation_module,
    }
    paths = {}
    manifest = []
    for source_id in sorted(modules):
        raw_path = inspect.getsourcefile(modules[source_id])
        if not raw_path:
            raise RuntimeError(f"cannot locate {source_id}")
        path = Path(raw_path).resolve(strict=True)
        resolved, size_bytes, digest = stable_file_descriptor(path)
        if digest != EXPECTED_SOURCE_SHA256[source_id]:
            raise ValueError(f"frozen source drift: {source_id}")
        paths[source_id] = str(path)
        manifest.append(
            {
                "source_id": source_id,
                "path": str(path),
                "resolved_path": resolved,
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )
    return paths, manifest


def make_dataset(dataset_dir: Path, action_stats: Path, tasks: tuple[str, ...]):
    config = {
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
        "action_stats_path": str(action_stats),
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
        robot="aloha-agilex",
        variant="clean_50",
        tasks=list(tasks),
        task_name=None,
        action_stats_path=str(action_stats),
        action_mode="eef",
        normalize_mode="min-max",
        num_frames=113,
        height=384,
        width=320,
        split="train",
        val_ratio=0.0,
        repeat=1,
        target_camera="head_camera",
        window_stride=1,
        video_stride=4,
        multiview=True,
        camera_layout=["head_camera", "left_camera", "right_camera"],
        backbone=None,
        filter_static_segments=False,
        static_segment_threshold=1.0e-5,
        max_static_retry=3,
        text_embedding_cache_dir=None,
        text_embedding_dropout=0.0,
        vae_cache_dir=None,
        temporal_compression=4,
        causal_temporal=True,
        vae_type="wan",
        growing_history=False,
        history_min_frames=1,
        history_stride=1,
        gdn_chunk_size=1,
        delta_action=False,
        seed=42,
        num_val_samples=4,
    )
    return dataset, config


def uniform(children, name):
    values = [getattr(child, name) for child in children]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"child preprocessing drift: {name}")
    return values[0]


def expected_preprocessing(dataset, config, tasks, source_manifest):
    children = dataset._sub_datasets
    task_list = list(tasks)
    source_manifest_sha = hashlib.sha256(
        canonical_json_bytes(source_manifest)
    ).hexdigest()
    task_set_sha = hashlib.sha256(
        "".join(f"{task}\n" for task in sorted(tasks)).encode()
    ).hexdigest()
    return {
        "action_mode": uniform(children, "action_mode"),
        "causal_temporal": uniform(children, "causal_temporal"),
        "delta_action": uniform(children, "delta_action"),
        "filter_static_segments": uniform(children, "_filter_static_segments"),
        "growing_history": uniform(children, "growing_history"),
        "height": uniform(children, "height"),
        "multiview": uniform(children, "multiview"),
        "normalize_mode": uniform(children, "normalize_mode"),
        "num_frames": uniform(children, "num_frames"),
        "repeat": uniform(children, "repeat"),
        "robot": uniform(children, "robot"),
        "split": uniform(children, "split"),
        "temporal_compression": uniform(children, "temporal_compression"),
        "text_embedding_dropout": float(config["text_embedding_dropout"]),
        "val_ratio": float(config["val_ratio"]),
        "variant": dataset.variant,
        "video_stride": uniform(children, "video_stride"),
        "width": uniform(children, "width"),
        "window_stride": uniform(children, "window_stride"),
        "camera_layout": list(uniform(children, "camera_layout")),
        "target_camera": uniform(children, "target_camera"),
        "action_stats_path": os.path.realpath(dataset.action_stats_path),
        "action_stats_sha256": EXPECTED_ACTION_STATS_SHA256,
        "vae_type": uniform(children, "vae_type"),
        "raw_window_len": uniform(children, "_raw_window_len"),
        "num_video_frames": uniform(children, "num_video_frames"),
        "video_sample_indices": list(uniform(children, "_video_sample_indices")),
        "history_min_frames": uniform(children, "history_min_frames"),
        "history_stride": uniform(children, "history_stride"),
        "gdn_chunk_size": uniform(children, "gdn_chunk_size"),
        "static_segment_threshold": uniform(children, "_static_segment_threshold"),
        "max_static_retry": uniform(children, "_max_static_retry"),
        "text_embedding_cache_dir": None,
        "vae_cache_dir": None,
        "dataset_dir": os.path.realpath(config["dataset_dir"]),
        "seed": 42,
        "num_val_samples": 4,
        "backbone": None,
        "robotwin_dataset_source_sha256": EXPECTED_SOURCE_SHA256[
            "sana_wam.dataloader.robotwin_dataset"
        ],
        "preprocessing_source_manifest": source_manifest,
        "preprocessing_source_manifest_sha256": source_manifest_sha,
        "canonical_train_tasks": task_list,
        "canonical_train_tasks_sha256": task_set_sha,
        "canonical_train_task_order_sha256": hashlib.sha256(
            canonical_json_bytes(task_list)
        ).hexdigest(),
    }


def resolve_window(dataset, dataset_index: int):
    child_index, local_index = dataset.resolve_global_index(dataset_index)
    child = dataset._sub_datasets[child_index]
    require_frozen_expansion_geometry(child)
    episode_index, start_frame, logical_length = child._resolve_local_index(local_index)
    support = independent_support(
        episode_length=child._episode_lengths[episode_index],
        start_frame=start_frame,
        window_logical_length=logical_length,
    )
    return child, {
        "dataset_index": dataset_index,
        "task_name": child.task_name,
        "episode_index": episode_index,
        "episode_path": child._episode_files[episode_index],
        "start_frame": start_frame,
        "window_logical_length": logical_length,
        **support,
    }


def independent_support(*, episode_length, start_frame, window_logical_length):
    if any(
        type(value) is not int or value < 0
        for value in (episode_length, start_frame, window_logical_length)
    ):
        raise ValueError("support inputs must be non-negative integers")
    if episode_length < 1 or window_logical_length > 113:
        raise ValueError("support inputs differ from frozen geometry")
    actual_valid = min(
        window_logical_length,
        max(0, episode_length - start_frame),
    )
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


def instruction_key(episode_path: str) -> str:
    basename = os.path.basename(episode_path)
    number = basename.replace("episode", "").replace(".hdf5", "")
    return f"episode{number}.json"


def expected_instruction(child, episode_path: str, file_cache):
    key = instruction_key(episode_path)
    if key in child._instructions:
        path = os.path.abspath(
            os.path.join(os.path.dirname(child.data_root), "instructions", key)
        )
        if path not in file_cache:
            file_cache[path] = stable_file_descriptor(path)
        resolved, size_bytes, digest = file_cache[path]
        return {
            "kind": "json_file",
            "path": path,
            "resolved_path": resolved,
            "size_bytes": size_bytes,
            "sha256": digest,
            "fallback_descriptor": None,
        }
    task_name = child.prompt_task_name
    descriptor = {
        "schema_version": "sana-phase6-instruction-fallback-descriptor-v1",
        "task_name": task_name,
        "episode_basename": os.path.basename(episode_path),
        "base_prompt": f"The bimanual robot is performing a {task_name} task.",
    }
    payload = canonical_json_bytes(descriptor)
    return {
        "kind": "deterministic_fallback",
        "path": f"fallback://robotwin/{task_name}/{key}",
        "resolved_path": None,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "fallback_descriptor": descriptor,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--plan-artifact", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--dataset-contract-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = verify_frozen_file(args.registry, EXPECTED_REGISTRY_SHA256, "registry")
    plan_path = verify_frozen_file(
        args.plan_artifact, EXPECTED_PLAN_ARTIFACT_SHA256, "plan artifact"
    )
    action_stats = verify_frozen_file(
        args.action_stats, EXPECTED_ACTION_STATS_SHA256, "action stats"
    )
    dataset_dir = args.dataset_dir.expanduser().resolve(strict=True)
    artifact_path = args.dataset_contract.expanduser().resolve(strict=True)
    expected_artifact_sha = require_sha(
        args.dataset_contract_sha256, "dataset-contract SHA"
    )
    artifact_bytes = artifact_path.read_bytes()
    if hashlib.sha256(artifact_bytes).hexdigest() != expected_artifact_sha:
        raise ValueError("external dataset-contract SHA mismatch")
    artifact = parse_canonical_json(artifact_bytes, "dataset contract")
    require_keys(artifact, TOP_KEYS, "dataset contract")
    if artifact["schema_version"] != "sana-phase6-dataset-contract-v2":
        raise ValueError("dataset-contract schema mismatch")
    if (
        artifact["expansion_support_contract_sha256"]
        != EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256
        or hashlib.sha256(
            canonical_json_bytes(artifact["expansion_support_contract"])
        ).hexdigest()
        != EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256
        or artifact["expansion_eligibility_amendment_sha256"]
        != EXPECTED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
    ):
        raise ValueError("expansion-support provenance pin mismatch")
    verify_padding_provenance(artifact)
    if (
        artifact["plan_sha256"] != EXPECTED_PLAN_SHA256
        or artifact["identity_sha256"] != EXPECTED_IDENTITY_SHA256
        or artifact["action_stats_sha256"] != EXPECTED_ACTION_STATS_SHA256
    ):
        raise ValueError("embedded provenance pin mismatch")

    sampler_contract = SamplerContract.from_registry(registry)
    plan = TaskRoundRobinPlan.from_artifact_bytes(
        plan_path.read_bytes(),
        expected_plan_sha256=EXPECTED_PLAN_SHA256,
        expected_identity_sha256=EXPECTED_IDENTITY_SHA256,
    )
    _source_paths, source_manifest = source_paths_and_manifest()
    dataset, config = make_dataset(
        dataset_dir, action_stats, sampler_contract.train_tasks
    )
    observed_preprocessing = expected_preprocessing(
        dataset, config, sampler_contract.train_tasks, source_manifest
    )
    if artifact["preprocessing_contract"] != observed_preprocessing:
        raise ValueError("preprocessing contract differs from independent observation")

    rows = artifact["rows"]
    episodes = artifact["episodes"]
    if not isinstance(rows, list) or len(rows) != 504:
        raise ValueError("artifact must contain 504 rows")
    if not isinstance(episodes, list) or not episodes or len(episodes) > len(rows):
        raise ValueError("artifact has an invalid deduplicated episode count")
    if [item.get("episode_path") for item in episodes] != sorted(
        item.get("episode_path") for item in episodes
    ):
        raise ValueError("episode descriptors are not sorted")
    episode_map = {}
    for index, episode in enumerate(episodes):
        require_keys(episode, EPISODE_KEYS, f"episodes[{index}]")
        require_sha(episode["sha256"], f"episodes[{index}].sha256")
        require_keys(
            episode["instruction_source"],
            INSTRUCTION_KEYS,
            f"episodes[{index}].instruction_source",
        )
        path = episode["episode_path"]
        if not isinstance(path, str) or not path or path in episode_map:
            raise ValueError("duplicate or invalid episode path")
        episode_map[path] = episode

    contexts = {}
    seen_indices = set()
    verified_row_projection = []
    for offset, (row, plan_row) in enumerate(zip(rows, plan.rows), start=1):
        require_keys(row, ROW_KEYS, f"rows[{offset - 1}]")
        require_support_types(row, f"rows[{offset - 1}]")
        if require_int(row["global_step"], "global_step", 1) != offset:
            raise ValueError("global-step order drift")
        dataset_index = require_int(row["dataset_index"], "dataset_index")
        if dataset_index in seen_indices:
            raise ValueError("reused dataset index")
        seen_indices.add(dataset_index)
        identity = plan_row.identity
        plan_projection = {
            "global_step": plan_row.global_step,
            "dataset_index": identity.dataset_index,
            "task_name": identity.task_name,
            "episode_index": identity.episode_index,
            "episode_path": identity.episode_path,
            "start_frame": identity.start_frame,
        }
        if any(row[key] != value for key, value in plan_projection.items()):
            raise ValueError(f"plan binding drift at global step {offset}")
        child, runtime = resolve_window(dataset, dataset_index)
        if any(row[key] != value for key, value in runtime.items()):
            raise ValueError(f"runtime window drift at global step {offset}")
        if row["eligible"] is not True or row["eligible_chunk_count"] < 1:
            raise ValueError("artifact contains a structurally ineligible row")
        if (
            type(row["window_logical_length"]) is not int
            or row["window_logical_length"] < 2
            or row["window_logical_length"] > observed_preprocessing["num_frames"]
        ):
            raise ValueError("invalid logical window length")
        episode = episode_map.get(row["episode_path"])
        if episode is None:
            raise ValueError("row references absent episode")
        if (
            row["episode_sha256"] != episode["sha256"]
            or row["instruction_source_sha256"]
            != episode["instruction_source"]["sha256"]
            or row["task_name"] != episode["task_name"]
            or row["episode_index"] != episode["episode_index"]
        ):
            raise ValueError("row/episode descriptor mismatch")
        contexts.setdefault(row["episode_path"], (child, runtime))
        verified_row_projection.append(
            {
                key: row[key]
                for key in (
                    "global_step",
                    "dataset_index",
                    "episode_path",
                    "start_frame",
                    "window_logical_length",
                    "actual_valid_raw_frames",
                    "valid_sampled_video_frames",
                    "valid_latent_frames",
                    "nonbootstrap_valid_latent_frames",
                    "eligible_chunk_count",
                    "eligible",
                    "episode_sha256",
                    "instruction_source_sha256",
                )
            }
        )
    if set(contexts) != set(episode_map):
        raise ValueError("unreferenced or missing episode descriptor")

    file_cache = {}
    instruction_file_count = 0
    fallback_count = 0
    for episode in episodes:
        child, runtime = contexts[episode["episode_path"]]
        path = os.path.abspath(episode["episode_path"])
        if path not in file_cache:
            file_cache[path] = stable_file_descriptor(path)
        resolved, size_bytes, digest = file_cache[path]
        if (
            episode["resolved_path"] != resolved
            or episode["size_bytes"] != size_bytes
            or episode["sha256"] != digest
            or episode["task_name"] != runtime["task_name"]
            or episode["episode_index"] != runtime["episode_index"]
        ):
            raise ValueError(f"episode bytes/path drift: {path}")
        instruction = expected_instruction(child, episode["episode_path"], file_cache)
        if instruction != episode["instruction_source"]:
            raise ValueError(f"instruction source drift: {path}")
        if instruction["kind"] == "json_file":
            instruction_file_count += 1
        else:
            fallback_count += 1

    support_count_keys = (
        "actual_valid_raw_frames",
        "valid_sampled_video_frames",
        "valid_latent_frames",
        "nonbootstrap_valid_latent_frames",
        "eligible_chunk_count",
    )
    selected_minimum_support = {
        key: min(int(row[key]) for row in rows) for key in support_count_keys
    }
    selected_minimum_support["eligible"] = all(row["eligible"] is True for row in rows)
    verification_manifest = {
        "schema_version": "sana-phase6-dataset-contract-verification-v2",
        "verification_passed": True,
        "dataset_contract_path": str(artifact_path),
        "dataset_contract_sha256": expected_artifact_sha,
        "plan_artifact_sha256": EXPECTED_PLAN_ARTIFACT_SHA256,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "identity_sha256": EXPECTED_IDENTITY_SHA256,
        "action_stats_sha256": EXPECTED_ACTION_STATS_SHA256,
        "expansion_eligibility_amendment_sha256": (
            EXPECTED_EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        ),
        "expansion_support_contract_sha256": (
            EXPECTED_EXPANSION_SUPPORT_CONTRACT_SHA256
        ),
        "padding_semantics_amendment_sha256": (
            EXPECTED_PADDING_SEMANTICS_AMENDMENT_SHA256
        ),
        "padding_semantics_contract": dict(EXPECTED_PADDING_SEMANTICS_CONTRACT),
        "padding_semantics_contract_sha256": (
            EXPECTED_PADDING_SEMANTICS_CONTRACT_SHA256
        ),
        "padding_eligibility_dependency": dict(
            EXPECTED_PADDING_ELIGIBILITY_DEPENDENCY
        ),
        "row_count": len(rows),
        "row_binding_sequence_sha256": hashlib.sha256(
            canonical_json_bytes(verified_row_projection)
        ).hexdigest(),
        "deduplicated_episode_count": len(episodes),
        "episode_descriptor_sequence_sha256": hashlib.sha256(
            canonical_json_bytes(episodes)
        ).hexdigest(),
        "instruction_file_count": instruction_file_count,
        "instruction_fallback_count": fallback_count,
        "preprocessing_source_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(source_manifest)
        ).hexdigest(),
        "selected_minimum_support": selected_minimum_support,
        "training_or_gpu_started": False,
    }
    manifest_bytes = canonical_json_bytes(verification_manifest)
    manifest_path = args.output_manifest.expanduser().resolve(strict=False)
    atomic_exclusive_write(manifest_path, manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    print(json.dumps(verification_manifest, indent=2, sort_keys=True))
    print(f"verification_manifest_sha256={manifest_sha}")


if __name__ == "__main__":
    main()
