#!/usr/bin/env python
"""Run the bounded LIBERO T5 same-suite cross-task transfer screen."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

from smoke_libero_ar_real_gpu import (  # noqa: E402
    DEFAULT_CONFIG,
    EXPECTED_BASE_PARTIAL_LOAD_PREFIX,
    EXPECTED_BASE_PARTIAL_LOAD_SAMPLES,
    PARQUET_SHA256,
    SPATIAL_NAME,
    _assert_idle_gpu,
    _find_fixed_sample,
    _query_gpu,
    _repo_commit,
    _sha256_file,
)
from smoke_libero_ar_real_update_gpu import (  # noqa: E402
    EXPECTED_SPATIAL_SAMPLE_ASSET_SHA256,
    EXPECTED_PRESERVE_FROZEN_INPUT_GRAD_MODULES,
    EXPECTED_SANA_COMMIT,
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    EXPECTED_TRAINABLE_ROOTS,
    EXPECTED_TRAINABLE_TENSOR_COUNT,
    _assert_source_tree_clean,
    _capture_update_probes,
    _create_run_root,
    _freeze_run_root,
    _optimizer_state_is_finite,
    _parameter_root,
    _sha256_json,
    _summarize_updates,
    _tensor_scalar,
    _verify_external_assets,
    _verify_pinned_asset,
)


T5_UPDATE_STEPS = 20
T5_LEARNABILITY_RATIO = 0.95
T5_INITIALIZATION_SEED = 20260806
T5_LOSS_RECIPE_SEED = 20260826
T5_CROSS_TASK_MEDIAN_RATIO = 0.95
T5_CROSS_TASK_MIN_IMPROVED = 2
T5_TRAIN_EPISODE_LENGTH = 110
T5_TRAIN_TASK_INDEX = 0
T5_TRAIN_TASK_NAME = (
    "pick up the black bowl next to the cookie box and place it on the plate"
)
T5_CROSS_TASK_SAMPLES = (
    {
        "assets": {
            "data/chunk-000/episode_000036.parquet": (
                "09d6f24419a6d70c766c42035c2cf993a59663070f1989a9ae54fa4cc83bc8f4"
            ),
            "videos/chunk-000/observation.images.image/episode_000036.mp4": (
                "a84cc4614ed6d7d08c9945567e7d3675e9ff7afa19113b9eaa5f68e3fe8af419"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000036.mp4": (
                "5aca422da74b6d059a8faac3c7e7ce1fa5e853c1c98486b106ba94c092a22114"
            ),
        },
        "episode_index": 36,
        "episode_length": 116,
        "episode_selection_payload_sha256": (
            "089747625a57d2d60e853d5d5c1c0c43ad6a01b477dbede629dda40933defcc9"
        ),
        "label": "H1",
        "start_frame": 0,
        "task": (
            "pick up the black bowl next to the ramekin and place it on the plate"
        ),
        "task_index": 7,
        "task_selection_payload_sha256": (
            "0a028e007220594e3d743dbd11db14944ca281ae37db8248ac7cf257cb742415"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000325.parquet": (
                "a7f51cc301d4e32a15717d6fea1384f45416d5fc5ab8e4183f7dee7c57b291d3"
            ),
            "videos/chunk-000/observation.images.image/episode_000325.mp4": (
                "edd51702dfcf53b60db9e12be38d18acad88678657e4cf063ec01da7517ac8d6"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000325.mp4": (
                "8511a776845f75aefeb1a2b904d00d668a6c5ddc25d03f7480474c6436ba2e58"
            ),
        },
        "episode_index": 325,
        "episode_length": 131,
        "episode_selection_payload_sha256": (
            "0a92dadc9f71161d6bc9e40476fbceed1d9ebc8bb297f111b402d17c8b9a50ca"
        ),
        "label": "H2",
        "start_frame": 0,
        "task": (
            "pick up the black bowl in the top drawer of the wooden cabinet and "
            "place it on the plate"
        ),
        "task_index": 1,
        "task_selection_payload_sha256": (
            "0ed2d254a2b9424e0c4a5dadc44532e66081c215edb6d651949c50c6e70d4432"
        ),
    },
    {
        "assets": {
            "data/chunk-000/episode_000011.parquet": (
                "5659083b099c426d2e9f4dde654fe3e556bde369c5c99f40eb0e521373b99c7f"
            ),
            "videos/chunk-000/observation.images.image/episode_000011.mp4": (
                "be57d261c1201b7258599f2b7f3b597e0464616daa8c92b8ffc4a5ef64c39a9a"
            ),
            "videos/chunk-000/observation.images.wrist_image/episode_000011.mp4": (
                "a8ad057219920267213f88a801dc7aecf7f864c05b2beb327ea7820031167102"
            ),
        },
        "episode_index": 11,
        "episode_length": 84,
        "episode_selection_payload_sha256": (
            "02dbc353aaa077f46f0ee7cb4513417a6343c20146d2abfbf197279f35bbd13e"
        ),
        "label": "H3",
        "start_frame": 0,
        "task": (
            "pick up the black bowl between the plate and the ramekin and place it "
            "on the plate"
        ),
        "task_index": 4,
        "task_selection_payload_sha256": (
            "34e97146b609f1ff31d2b9cf44a2e86c3fc84205383535f7bdb0d2280b8b7ff2"
        ),
    },
)
T5_CROSS_TASK_EPISODE_INDICES = (36, 325, 11)
T5_CROSS_TASK_TASK_INDICES = (7, 1, 4)
T5_ELIGIBLE_MANIFEST_SHA256 = (
    "7b15d57b58ef398e7f0c3cd9371db03bd3312e4a4f3c3168f356df0f612bba45"
)
T5_SELECTION_MANIFEST_SHA256 = (
    "e60eee38c8b4ba9f9e0ffe49eac3d102ced497981404ad3613ac30fa0f324886"
)
T5_EPISODES_METADATA_SHA256 = (
    "690349688e96d984007bf2cca3ccf2683dfc809f03dab55e1159e4cc71f150b7"
)
T5_TASKS_METADATA_SHA256 = (
    "399841cf8e861052d3fb7214ed619b5a10b4eded2d6f107e7fed338a233dd0e1"
)
T5_CROSS_TASK_SAMPLE_ASSET_SHA256 = {
    row["episode_index"]: row["assets"] for row in T5_CROSS_TASK_SAMPLES
}
T5_CROSS_TASK_EPISODE_LENGTHS = {
    row["episode_index"]: row["episode_length"] for row in T5_CROSS_TASK_SAMPLES
}
T5_FIXED_RECIPE_SIGNATURE_SHA256 = (
    "ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad"
)
T5_TRAINING_CORE_SHA256 = (
    "e34a2dd0ae2dfe28303dcd9baf64ffc5800d002b0b7646b89dde2aafa132c352"
)
SELECTED_STATS_SHA256 = (
    "e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7"
)
SELECTED_POPULATION_SHA256 = (
    "7ed9772facf261299022e55169bcdaaa49fe3a7a20e0057e08cf419b5e584146"
)
T1_PREDECESSOR_RESULT_SHA256 = (
    "9d9c139fd67baa8c131ad9e6537862536b8262b732c2fda668a7c8b8b6a8621a"
)
T1_PREDECESSOR_RESULT_PATH = Path(
    "/tmp/sana-wam-libero-t1-one-update-fp32master-0337e28-20260806-a3/RESULT.json"
)
T2_PREDECESSOR_RESULT_SHA256 = (
    "67d250cdb13470bea9e9fa53531d3c76c65145e5040dc7a45079d84ac4960a57"
)
T2_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t2/2dc1ce730df3/"
    "libero-t2-fixed20-20260806-a1/RESULT.json"
)
T2_PREDECESSOR_SOURCE_COMMIT = "2dc1ce730df37bd9a2a71e4946f153380a8d4649"
T2_PREDECESSOR_RUNNER_SHA256 = (
    "f5e5e3e80d6b5afd586e54e83a7e63dcf43280192363c70580aa03145267e9a6"
)
T2_PREDECESSOR_RUNNER_PATH = ROOT / (
    "scripts/smoke_libero_ar_t2_fixed_sample_20step_gpu.py"
)
T3_PREDECESSOR_RESULT_SHA256 = (
    "c883608f2a47b6258f824d4d97a94f8a390d03bab671a592fb758eea61b3a01e"
)
T3_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t3/ac431f8727ac/"
    "libero-t3-heldout3-fixed20-220afb60725d0cfd591bc4fe225cdd21/RESULT.json"
)
T3_PREDECESSOR_SOURCE_COMMIT = "ac431f8727ac345c3bd0ad4a442e53801310fd81"
T3_PREDECESSOR_RUNNER_SHA256 = (
    "0db0bfe086bc4b5ea92462e4832f0c448408656cca819796ec253f6226b4c017"
)
T3_PREDECESSOR_RUNNER_PATH = ROOT / (
    "scripts/smoke_libero_ar_t3_heldout_recipe_transfer_gpu.py"
)
T4_PREDECESSOR_RESULT_SHA256 = (
    "4c33aaff6d202068d77efc0ac406c74198c56e72527cfabdde046fc9a3a4b6f4"
)
T4_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t4/7db8182ef46f/"
    "libero-t4-heldout3-fixed20-36120c596971575d286d378df42ac564/RESULT.json"
)
T4_PREDECESSOR_SOURCE_COMMIT = "7db8182ef46fce367f04f66bfbc34690fa86592b"
T4_PREDECESSOR_RUNNER_SHA256 = (
    "3d449761861b7fe75816d4d186b2e82402c160422eeb5f490fbb61ed274f088c"
)
T4_PREDECESSOR_RUNNER_PATH = ROOT / (
    "scripts/smoke_libero_ar_t4_heldout_sample_transfer_gpu.py"
)
SPATIAL_STATS_GR00T_SHA256 = (
    "0a4b08f5afcdcbe186ec70ea6ff2569233bab706a4d99ed603b23a7688bc33bb"
)
T5_RUN_NAMESPACE = Path("/DATA/share/sana_wam_libero_nonformal_screens/t5")
_ACTIVE_RUN_ROOT: Path | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"T5 metadata must be a regular non-symlink file: {path}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            raise RuntimeError(
                f"T5 metadata contains a blank row: {path}:{line_number}"
            )
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError(
                f"T5 metadata row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _build_eligible_sample_manifest(
    task_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if any(set(row) != {"task_index", "task"} for row in task_rows):
        raise RuntimeError("T5 tasks metadata schema differs")
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    if (
        sorted(task_by_index) != list(range(10))
        or len(task_by_index) != len(task_rows)
        or len(set(task_by_index.values())) != len(task_by_index)
        or task_by_index[T5_TRAIN_TASK_INDEX] != T5_TRAIN_TASK_NAME
    ):
        raise RuntimeError("T5 Spatial task registry differs")
    task_index_by_name = {name: index for index, name in task_by_index.items()}
    episodes_by_task: dict[int, list[dict[str, int]]] = {
        task_index: [] for task_index in range(1, 10)
    }
    observed_episode_indices: set[int] = set()
    for row in episode_rows:
        if set(row) != {"episode_index", "tasks", "length"}:
            raise RuntimeError("T5 episodes metadata schema differs")
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        tasks = row["tasks"]
        if (
            episode_index in observed_episode_indices
            or not isinstance(tasks, list)
            or len(tasks) != 1
            or tasks[0] not in task_index_by_name
            or length <= 0
        ):
            raise RuntimeError("T5 episodes metadata identity differs")
        observed_episode_indices.add(episode_index)
        task_index = task_index_by_name[tasks[0]]
        if task_index != T5_TRAIN_TASK_INDEX:
            episodes_by_task[task_index].append(
                {"episode_index": episode_index, "length": length}
            )
    tasks = [
        {
            "episode_count": len(episodes_by_task[task_index]),
            "episodes": sorted(
                episodes_by_task[task_index], key=lambda row: row["episode_index"]
            ),
            "task": task_by_index[task_index],
            "task_index": task_index,
        }
        for task_index in range(1, 10)
    ]
    manifest = {
        "candidate_episode_count": sum(row["episode_count"] for row in tasks),
        "candidate_task_count": len(tasks),
        "dataset": SPATIAL_NAME,
        "episodes_jsonl_sha256": T5_EPISODES_METADATA_SHA256,
        "excluded_task_index": T5_TRAIN_TASK_INDEX,
        "schema_version": "sana-wam-libero-t5-cross-task-eligible-v1",
        "start_frame": 0,
        "t4_predecessor_result_sha256": T4_PREDECESSOR_RESULT_SHA256,
        "tasks": tasks,
        "tasks_jsonl_sha256": T5_TASKS_METADATA_SHA256,
    }
    if (
        manifest["candidate_task_count"] != 9
        or manifest["candidate_episode_count"] != 386
        or _sha256_json(manifest) != T5_ELIGIBLE_MANIFEST_SHA256
    ):
        raise RuntimeError("T5 eligible cross-task manifest differs")
    return manifest


def _eligible_sample_manifest(spatial_root: Path) -> dict[str, Any]:
    tasks_path = spatial_root / "meta/tasks.jsonl"
    episodes_path = spatial_root / "meta/episodes.jsonl"
    if _sha256_file(tasks_path) != T5_TASKS_METADATA_SHA256:
        raise RuntimeError("T5 tasks metadata SHA differs")
    if _sha256_file(episodes_path) != T5_EPISODES_METADATA_SHA256:
        raise RuntimeError("T5 episodes metadata SHA differs")
    return _build_eligible_sample_manifest(
        _read_jsonl(tasks_path),
        _read_jsonl(episodes_path),
    )


def _task_selection_payload_sha256(task_index: int) -> str:
    payload = (
        "SANA-WAM/LIBERO/T5_TASK_SELECTION_V1\n"
        f"T4_RESULT_SHA256={T4_PREDECESSOR_RESULT_SHA256}\n"
        f"ELIGIBLE_MANIFEST_SHA256={T5_ELIGIBLE_MANIFEST_SHA256}\n"
        f"DATASET={SPATIAL_NAME}\n"
        f"TASK_INDEX={task_index}\n"
        "START_FRAME=0\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _episode_selection_payload_sha256(task_index: int, episode_index: int) -> str:
    payload = (
        "SANA-WAM/LIBERO/T5_EPISODE_SELECTION_V1\n"
        f"T4_RESULT_SHA256={T4_PREDECESSOR_RESULT_SHA256}\n"
        f"ELIGIBLE_MANIFEST_SHA256={T5_ELIGIBLE_MANIFEST_SHA256}\n"
        f"DATASET={SPATIAL_NAME}\n"
        f"TASK_INDEX={task_index}\n"
        f"EPISODE_INDEX={episode_index}\n"
        "START_FRAME=0\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _selection_manifest(eligible_manifest: dict[str, Any]) -> dict[str, Any]:
    ranked_tasks = sorted(
        (
            _task_selection_payload_sha256(int(task_row["task_index"])),
            task_row,
        )
        for task_row in eligible_manifest["tasks"]
    )
    samples = []
    for rank, (task_sha256, task_row) in enumerate(ranked_tasks[:3], start=1):
        task_index = int(task_row["task_index"])
        ranked_episodes = sorted(
            (
                _episode_selection_payload_sha256(
                    task_index, int(episode_row["episode_index"])
                ),
                episode_row,
            )
            for episode_row in task_row["episodes"]
        )
        episode_sha256, episode_row = ranked_episodes[0]
        expected = T5_CROSS_TASK_SAMPLES[rank - 1]
        sample = {
            "assets": expected["assets"],
            "episode_index": int(episode_row["episode_index"]),
            "episode_length": int(episode_row["length"]),
            "episode_selection_payload_sha256": episode_sha256,
            "label": f"H{rank}",
            "start_frame": 0,
            "task": task_row["task"],
            "task_index": task_index,
            "task_selection_payload_sha256": task_sha256,
        }
        if sample != expected:
            raise RuntimeError(f"T5 mechanically selected sample H{rank} differs")
        samples.append(sample)
    manifest = {
        "eligible_manifest_sha256": T5_ELIGIBLE_MANIFEST_SHA256,
        "samples": samples,
        "schema_version": "sana-wam-libero-t5-cross-task-selection-v1",
        "selection_rule": (
            "rank tasks by task payload SHA256; take first 3; within each selected "
            "task rank episodes by episode payload SHA256 and take first"
        ),
        "t4_predecessor_result_sha256": T4_PREDECESSOR_RESULT_SHA256,
    }
    if _sha256_json(manifest) != T5_SELECTION_MANIFEST_SHA256:
        raise RuntimeError("T5 selected cross-task manifest differs")
    return manifest


def _select_t5_samples(
    dataset: Any,
    eligible_manifest: dict[str, Any],
    selection_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    live_task_rows: dict[int, dict[str, Any]] = {}
    live_episode_rows: dict[int, dict[str, Any]] = {}
    for episode in dataset._episodes:
        if episode.dataset != SPATIAL_NAME:
            continue
        task_row = {"task_index": episode.task_index, "task": episode.task}
        previous_task = live_task_rows.setdefault(episode.task_index, task_row)
        if previous_task != task_row or episode.episode_index in live_episode_rows:
            raise RuntimeError("T5 live Spatial registry identity is ambiguous")
        live_episode_rows[episode.episode_index] = {
            "episode_index": episode.episode_index,
            "tasks": [episode.task],
            "length": episode.length,
        }
    live_manifest = _build_eligible_sample_manifest(
        [live_task_rows[index] for index in sorted(live_task_rows)],
        [live_episode_rows[index] for index in sorted(live_episode_rows)],
    )
    if live_manifest != eligible_manifest:
        raise RuntimeError("T5 live eligible cross-task population differs")

    expected_keys = {
        (int(row["task_index"]), int(row["episode_index"]))
        for row in selection_manifest["samples"]
    }
    start_zero: dict[tuple[int, int], list[tuple[int, Any]]] = {
        key: [] for key in expected_keys
    }
    for dataset_index, (episode_position, start) in enumerate(dataset._windows):
        episode = dataset._episodes[episode_position]
        key = (episode.task_index, episode.episode_index)
        if episode.dataset == SPATIAL_NAME and start == 0 and key in start_zero:
            start_zero[key].append((dataset_index, episode))

    selected = []
    for row in selection_manifest["samples"]:
        key = (int(row["task_index"]), int(row["episode_index"]))
        matches = start_zero[key]
        if len(matches) != 1:
            raise RuntimeError(
                "T5 cross-task episode/start must occur exactly once: "
                f"task={key[0]} episode={key[1]} matches={len(matches)}"
            )
        dataset_index, episode = matches[0]
        selected.append({**row, "dataset_index": dataset_index, "episode": episode})
    return selected


@contextlib.contextmanager
def _fixed_rng_recipe(torch_module: Any, *, seed: int, cuda_device: int | None):
    """Reset one loss recipe while restoring the caller's RNG states."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [] if cuda_device is None else [cuda_device]
    try:
        with torch_module.random.fork_rng(devices=devices, enabled=True):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch_module.manual_seed(seed)
            if cuda_device is not None:
                torch_module.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _learnability_summary(
    initial_action_loss: float,
    pre_update_action_losses: list[float],
    post_update_action_loss: float,
) -> dict[str, Any]:
    if len(pre_update_action_losses) != T5_UPDATE_STEPS:
        raise ValueError(f"T5 requires exactly {T5_UPDATE_STEPS} pre-update losses")
    values = [initial_action_loss, *pre_update_action_losses, post_update_action_loss]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("T5 losses must be finite and non-negative")
    if initial_action_loss <= 0.0:
        raise ValueError("T5 initial action loss must be strictly positive")
    first_five_median = statistics.median(pre_update_action_losses[:5])
    last_five_median = statistics.median(pre_update_action_losses[-5:])
    ratio = post_update_action_loss / initial_action_loss
    return {
        "first_five_median": first_five_median,
        "initial_measurement": initial_action_loss,
        "initial_pre_update": pre_update_action_losses[0],
        "last_five_median": last_five_median,
        "post_update": post_update_action_loss,
        "post_to_initial_ratio": ratio,
        "relative_drop": 1.0 - ratio,
        "threshold_ratio": T5_LEARNABILITY_RATIO,
        "loss_gate": (
            post_update_action_loss <= T5_LEARNABILITY_RATIO * initial_action_loss
            and last_five_median <= T5_LEARNABILITY_RATIO * first_five_median
        ),
    }


def _cross_task_transfer_summary(
    before_action_losses: dict[int, float],
    after_action_losses: dict[int, float],
) -> dict[str, Any]:
    expected_episodes = set(T5_CROSS_TASK_EPISODE_INDICES)
    if (
        set(before_action_losses) != expected_episodes
        or set(after_action_losses) != expected_episodes
    ):
        raise ValueError("T5 cross-task loss episodes differ from the frozen set")

    rows = []
    for episode_index in T5_CROSS_TASK_EPISODE_INDICES:
        before = before_action_losses[episode_index]
        after = after_action_losses[episode_index]
        if (
            not math.isfinite(before)
            or not math.isfinite(after)
            or before <= 0.0
            or after < 0.0
        ):
            raise ValueError(
                "T5 cross-task losses must be finite with positive pre-loss"
            )
        ratio = after / before
        rows.append(
            {
                "after_action_loss": after,
                "absolute_delta": after - before,
                "before_action_loss": before,
                "episode_index": episode_index,
                "improved": after < before,
                "post_to_pre_ratio": ratio,
                "relative_drop": 1.0 - ratio,
            }
        )

    median_ratio = statistics.median(row["post_to_pre_ratio"] for row in rows)
    improved_count = sum(row["improved"] for row in rows)
    return {
        "heldout_sample_count": len(rows),
        "heldout_task_count": len(rows),
        "improved_count": improved_count,
        "median_post_to_pre_ratio": median_ratio,
        "minimum_improved_count": T5_CROSS_TASK_MIN_IMPROVED,
        "per_sample": rows,
        "sorted_post_to_pre_ratios": sorted(row["post_to_pre_ratio"] for row in rows),
        "tasks_distinct": True,
        "threshold_median_ratio": T5_CROSS_TASK_MEDIAN_RATIO,
        "transfer_gate": (
            median_ratio <= T5_CROSS_TASK_MEDIAN_RATIO
            and improved_count >= T5_CROSS_TASK_MIN_IMPROVED
        ),
    }


_CORE_OPTIMIZER_KEYS = (
    "adam_step_values",
    "betas",
    "constant_learning_rates",
    "cumulative_master_update",
    "cumulative_projected_bf16_update",
    "first_master_update",
    "first_projected_bf16_update",
    "master_object_identity_persistent",
    "optimizer_master_parameter_count",
    "optimizer_master_parameter_elements",
    "optimizer_state_finite",
    "production_accumulation_equivalent",
    "production_schedule_equivalent",
    "projection_exact_after_steps",
    "update_sentinel_count",
    "update_sentinel_names",
    "weight_decay",
)
_CORE_RANDOMNESS_KEYS = (
    "initialization_seed",
    "recipe_reset",
    "rng_state_restored_after_every_forward",
    "training_loss_recipe_seed",
    "training_recipe_signature_sha256",
    "training_recipe_signatures",
    "unique_training_recipe_signatures",
)
_CORE_SCOPE_KEYS = (
    "architecture_training_forwards",
    "backward_calls",
    "optimizer_steps",
    "sana_base_pretrained_checkpoint_loaded",
    "sana_wam_training_checkpoint_loaded",
    "sana_wam_training_checkpoint_saved",
)


def _training_core_projection(result: dict[str, Any]) -> dict[str, Any]:
    learnability = {
        key: value for key, value in result["learnability"].items() if key != "role"
    }
    per_step = [
        {
            key: value
            for key, value in row.items()
            if key not in {"peak_reserved_bytes", "seconds"}
        }
        for row in result["per_step"]
    ]
    if len(per_step) != T5_UPDATE_STEPS:
        raise ValueError("T5/T3 training-core view requires exactly 20 step rows")
    return {
        "architecture": result["architecture"],
        "assets": {
            key: result["assets"][key]
            for key in (
                "config_sha256",
                "spatial_files_post_sample_manifest_sha256",
                "stats_population_sha256",
                "stats_sha256",
                "stats_validation",
            )
        },
        "identity": {
            key: result["identity"][key]
            for key in (
                "config_sha256",
                "sana_commit",
                "spatial_assets_post_sample_sha256",
                "stats_population_sha256",
                "stats_sha256",
            )
        },
        "inputs": result["inputs"],
        "learnability": learnability,
        "optimizer": {key: result["optimizer"][key] for key in _CORE_OPTIMIZER_KEYS},
        "per_step": per_step,
        "randomness": {key: result["randomness"][key] for key in _CORE_RANDOMNESS_KEYS},
        "sample": result["sample"],
        "scope": {key: result["scope"][key] for key in _CORE_SCOPE_KEYS},
    }


def _write_terminal_report(path: Path, report: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            report,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def _tensor_observation(tensor: Any) -> dict[str, Any]:
    values = tensor.detach().float().reshape(-1)
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "values": [float(value) for value in values.tolist()],
    }


def _terminalize_failure(root: Path, error: BaseException) -> None:
    marker = root / "FAILED.json"
    result_marker = root / "RESULT.json"
    if (
        not result_marker.exists()
        and not result_marker.is_symlink()
        and not marker.exists()
        and not marker.is_symlink()
    ):
        _write_terminal_report(
            marker,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "error_type": type(error).__name__,
                "result": "FAIL",
                "schema_version": "sana-wam-libero-t5-cross-task-failure-v1",
            },
        )
    _freeze_run_root(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--steps", type=int, default=T5_UPDATE_STEPS)
    parser.add_argument(
        "--initialization-seed", type=int, default=T5_INITIALIZATION_SEED
    )
    parser.add_argument("--loss-recipe-seed", type=int, default=T5_LOSS_RECIPE_SEED)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

    if args.steps != T5_UPDATE_STEPS:
        raise ValueError(f"T5 steps must be exactly {T5_UPDATE_STEPS}")
    if args.initialization_seed != T5_INITIALIZATION_SEED:
        raise ValueError("T5 initialization seed differs")
    if args.loss_recipe_seed != T5_LOSS_RECIPE_SEED:
        raise ValueError("T5 loss recipe seed differs")
    if len(args.nonce) != 32 or any(
        character not in "0123456789abcdef" for character in args.nonce
    ):
        raise ValueError("T5 nonce must be exactly 32 lowercase hex characters")
    expected_root_name = f"libero-t5-crosstask3-fixed20-{args.nonce}"
    if args.run_root.name != expected_root_name:
        raise ValueError("T5 run-root basename does not bind the nonce")
    if args.run_root.parent.name != args.expected_repo_commit[:12]:
        raise ValueError("T5 run-root parent does not bind the source commit")
    expected_run_root = (
        T5_RUN_NAMESPACE / args.expected_repo_commit[:12] / expected_root_name
    )
    if args.run_root.expanduser() != expected_run_root:
        raise ValueError(
            "T5 run root differs from the frozen non-formal namespace: "
            f"expected={expected_run_root} got={args.run_root.expanduser()}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {str(args.physical_gpu), args.expected_gpu_uuid}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the selected physical GPU "
            f"({args.physical_gpu} or {args.expected_gpu_uuid}), got {visible!r}"
        )

    config_candidate = args.config.expanduser()
    if config_candidate.is_symlink() or not config_candidate.is_file():
        raise ValueError(
            f"config must be a regular non-symlink file: {config_candidate}"
        )
    config_path = config_candidate.resolve()
    runner_path = Path(__file__).resolve()
    run_root = _create_run_root(args.run_root.expanduser())
    global _ACTIVE_RUN_ROOT
    _ACTIVE_RUN_ROOT = run_root

    actual_commit = _repo_commit()
    actual_config_sha256 = _sha256_file(config_path)
    actual_runner_sha256 = _sha256_file(runner_path)
    expected_identity = {
        "config_sha256": args.expected_config_sha256.lower(),
        "repo_commit": args.expected_repo_commit.lower(),
        "runner_sha256": args.expected_runner_sha256.lower(),
        "sana_commit": EXPECTED_SANA_COMMIT,
    }
    actual_identity = {
        "config_sha256": actual_config_sha256,
        "repo_commit": actual_commit,
        "runner_sha256": actual_runner_sha256,
        "sana_commit": EXPECTED_SANA_COMMIT,
    }
    if actual_identity != expected_identity:
        raise RuntimeError(
            "T5 source identity differs: "
            f"expected={expected_identity!r} actual={actual_identity!r}"
        )
    actual_identity["sana_commit"] = _assert_source_tree_clean(actual_commit)

    cfg = OmegaConf.load(config_path)
    if cfg.model.architecture.variant != "autoregressive":
        raise RuntimeError("T5 requires the AR architecture")
    if cfg.model.video_backbone.continuous_timestep_conditioning is not True:
        raise RuntimeError("T5 requires continuous FP32 video timesteps")
    if cfg.training.use_gradient_checkpointing is not True:
        raise RuntimeError("T5 requires gradient checkpointing")
    if int(cfg.training.seed) != args.initialization_seed:
        raise RuntimeError("T5 CLI initialization seed must equal training.seed")
    if tuple(cfg.training.trainable_modules) != EXPECTED_TRAINABLE_ROOTS:
        raise RuntimeError("T5 trainable module allowlist differs")
    if (
        tuple(cfg.training.preserve_frozen_input_grad_modules)
        != EXPECTED_PRESERVE_FROZEN_INPUT_GRAD_MODULES
    ):
        raise RuntimeError("T5 frozen input-gradient preservation differs")
    if cfg.training.optimizer_master_weights is not True:
        raise RuntimeError("T5 requires persistent FP32 optimizer masters")
    if (
        float(cfg.training.lambda_video) != 0.0
        or float(cfg.training.lambda_action) != 1.0
    ):
        raise RuntimeError("T5 requires action-only loss weighting")
    if float(cfg.training.weight_decay) != 0.0:
        raise RuntimeError("T5 requires weight_decay=0")
    if (
        float(cfg.training.action_lr) != 1.0e-4
        or float(cfg.training.video_lr) != 1.0e-4
    ):
        raise RuntimeError("T5 requires fixed base learning rates of 1e-4")
    if float(cfg.training.grad_clip) != 1.0:
        raise RuntimeError("T5 requires the fixed gradient clip bound 1.0")
    forbidden_checkpoint_fields = (
        "training.init_checkpoint",
        "training.resume_checkpoint",
        "training.resume_from_checkpoint",
        "training.resume_manifest",
        "model.video_backbone.init_dit_from",
    )
    for path in forbidden_checkpoint_fields:
        if OmegaConf.select(cfg, path, default=None) is not None:
            raise RuntimeError(f"T5 forbids checkpoint field {path}")
    if str(cfg.training.action_stats_sha256) != SELECTED_STATS_SHA256:
        raise RuntimeError("T5 selected-row stats SHA pin differs")
    if str(cfg.training.action_stats_population_sha256) != SELECTED_POPULATION_SHA256:
        raise RuntimeError("T5 selected population SHA pin differs")
    stats_path = Path(str(cfg.dataloader.action_stats_path)).expanduser()
    if stats_path.is_symlink() or not stats_path.is_file():
        raise RuntimeError("T5 stats artifact must be a regular non-symlink file")
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T5 selected-row stats artifact SHA differs")

    from sana_wam.train.libero_contract import validate_libero_training_config

    # This is the only full numeric live-source preflight. It runs before any
    # model construction or CUDA allocation.
    validate_libero_training_config(cfg, require_materialized_stats=True)
    external_assets = _verify_external_assets(cfg)
    spatial_root = Path(str(cfg.dataloader.dataset_roots[0]))
    eligible_sample_manifest = _eligible_sample_manifest(spatial_root)
    cross_task_selection_manifest = _selection_manifest(eligible_sample_manifest)
    external_assets["spatial_sample/meta/stats_gr00t.json"] = _verify_pinned_asset(
        spatial_root / "meta/stats_gr00t.json",
        SPATIAL_STATS_GR00T_SHA256,
    )
    external_assets["t1_predecessor/RESULT.json"] = _verify_pinned_asset(
        T1_PREDECESSOR_RESULT_PATH,
        T1_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t2_predecessor/RESULT.json"] = _verify_pinned_asset(
        T2_PREDECESSOR_RESULT_PATH,
        T2_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t2_predecessor/runner.py"] = _verify_pinned_asset(
        T2_PREDECESSOR_RUNNER_PATH,
        T2_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t3_predecessor/RESULT.json"] = _verify_pinned_asset(
        T3_PREDECESSOR_RESULT_PATH,
        T3_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t3_predecessor/runner.py"] = _verify_pinned_asset(
        T3_PREDECESSOR_RUNNER_PATH,
        T3_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["t4_predecessor/RESULT.json"] = _verify_pinned_asset(
        T4_PREDECESSOR_RESULT_PATH,
        T4_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t4_predecessor/runner.py"] = _verify_pinned_asset(
        T4_PREDECESSOR_RUNNER_PATH,
        T4_PREDECESSOR_RUNNER_SHA256,
    )
    external_assets["cross_task_metadata/meta/episodes.jsonl"] = _verify_pinned_asset(
        spatial_root / "meta/episodes.jsonl",
        T5_EPISODES_METADATA_SHA256,
    )
    external_assets["cross_task_metadata/meta/tasks.jsonl"] = _verify_pinned_asset(
        spatial_root / "meta/tasks.jsonl",
        T5_TASKS_METADATA_SHA256,
    )
    for episode_index, assets in T5_CROSS_TASK_SAMPLE_ASSET_SHA256.items():
        for relative_path, expected_sha256 in sorted(assets.items()):
            external_assets[
                f"cross_task_sample/episode_{episode_index:06d}/{relative_path}"
            ] = _verify_pinned_asset(
                spatial_root / relative_path,
                expected_sha256,
            )
    t2_predecessor = json.loads(T2_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t2_predecessor.get("valid_run") is not True
        or t2_predecessor.get("scientific_verdict") != "T2_FIXED_SAMPLE_LEARNABILITY_GO"
        or t2_predecessor.get("identity", {}).get("repo_commit")
        != T2_PREDECESSOR_SOURCE_COMMIT
        or t2_predecessor.get("identity", {}).get("runner_sha256")
        != T2_PREDECESSOR_RUNNER_SHA256
    ):
        raise RuntimeError("T5 predecessor T2 evidence semantics differ")
    t3_predecessor = json.loads(T3_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t3_predecessor.get("valid_run") is not True
        or t3_predecessor.get("scientific_verdict") != "T3_HELDOUT_RECIPE_TRANSFER_GO"
        or t3_predecessor.get("identity", {}).get("repo_commit")
        != T3_PREDECESSOR_SOURCE_COMMIT
        or t3_predecessor.get("identity", {}).get("runner_sha256")
        != T3_PREDECESSOR_RUNNER_SHA256
        or t3_predecessor.get("scope", {}).get("architecture_forwards_total") != 28
        or t3_predecessor.get("scope", {}).get("backward_calls") != T5_UPDATE_STEPS
        or t3_predecessor.get("scope", {}).get("optimizer_steps") != T5_UPDATE_STEPS
    ):
        raise RuntimeError("T5 predecessor T3 evidence semantics differ")
    t3_training_core = _training_core_projection(t3_predecessor)
    t4_predecessor = json.loads(T4_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t4_predecessor.get("valid_run") is not True
        or t4_predecessor.get("scientific_verdict") != "T4_HELDOUT_SAMPLE_TRANSFER_GO"
        or t4_predecessor.get("identity", {}).get("repo_commit")
        != T4_PREDECESSOR_SOURCE_COMMIT
        or t4_predecessor.get("identity", {}).get("runner_sha256")
        != T4_PREDECESSOR_RUNNER_SHA256
        or t4_predecessor.get("scope", {}).get("architecture_forwards_total") != 28
        or t4_predecessor.get("scope", {}).get("backward_calls") != T5_UPDATE_STEPS
        or t4_predecessor.get("scope", {}).get("optimizer_steps") != T5_UPDATE_STEPS
    ):
        raise RuntimeError("T5 predecessor T4 evidence semantics differ")
    t4_training_core = _training_core_projection(t4_predecessor)
    if (
        t3_training_core != t4_training_core
        or _sha256_json(t3_training_core) != T5_TRAINING_CORE_SHA256
        or _sha256_json(t4_training_core) != T5_TRAINING_CORE_SHA256
    ):
        raise RuntimeError("T5 frozen T3/T4 training-core evidence differs")
    external_assets = dict(sorted(external_assets.items()))
    actual_identity.update(
        {
            "external_assets_sha256": _sha256_json(external_assets),
            "stats_population_sha256": SELECTED_POPULATION_SHA256,
            "stats_sha256": SELECTED_STATS_SHA256,
            "t1_predecessor_result_sha256": T1_PREDECESSOR_RESULT_SHA256,
            "t2_predecessor_result_sha256": T2_PREDECESSOR_RESULT_SHA256,
            "t2_predecessor_runner_sha256": T2_PREDECESSOR_RUNNER_SHA256,
            "t2_predecessor_source_commit": T2_PREDECESSOR_SOURCE_COMMIT,
            "t3_predecessor_result_sha256": T3_PREDECESSOR_RESULT_SHA256,
            "t3_predecessor_runner_sha256": T3_PREDECESSOR_RUNNER_SHA256,
            "t3_predecessor_source_commit": T3_PREDECESSOR_SOURCE_COMMIT,
            "t4_predecessor_result_sha256": T4_PREDECESSOR_RESULT_SHA256,
            "t4_predecessor_runner_sha256": T4_PREDECESSOR_RUNNER_SHA256,
            "t4_predecessor_source_commit": T4_PREDECESSOR_SOURCE_COMMIT,
            "eligible_manifest_sha256": T5_ELIGIBLE_MANIFEST_SHA256,
            "selection_manifest_sha256": T5_SELECTION_MANIFEST_SHA256,
        }
    )

    import fcntl

    lock_path = Path(f"/tmp/sana-wam-{args.expected_gpu_uuid}.lock")
    gpu_lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"selected GPU lock is already held: {lock_path}") from exc
    gpu_before = _assert_idle_gpu(args.physical_gpu, args.expected_gpu_uuid)
    if "H200" not in gpu_before["name"].upper():
        raise RuntimeError(f"T5 selected GPU is not an H200: {gpu_before['name']}")

    import torch

    from sana_wam.dataloader.transforms.multiview import format_prompt_for_inference
    from sana_wam.train.trainer import Trainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "T5 requires exactly one visible CUDA GPU; "
            f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.initialization_seed)
    np.random.seed(args.initialization_seed % (2**32))
    torch.manual_seed(args.initialization_seed)
    torch.cuda.manual_seed_all(args.initialization_seed)

    warning_messages: list[str] = []

    class WarningCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warning_messages.append(record.getMessage())

    warning_capture = WarningCapture(level=logging.WARNING)
    builder_logger = logging.getLogger(
        "sana_wam.model.video_backbone.sana.pipeline_builder"
    )
    builder_logger.addHandler(warning_capture)
    gpu_preconstruction = _assert_idle_gpu(args.physical_gpu, args.expected_gpu_uuid)
    torch.cuda.reset_peak_memory_stats(device)
    build_started = time.perf_counter()
    try:
        trainer = Trainer(cfg)
    finally:
        builder_logger.removeHandler(warning_capture)
    torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - build_started
    build_memory = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
    }
    partial_load_warnings = [
        message
        for message in warning_messages
        if "ckpt load partial" in message.lower()
    ]
    if not (
        len(partial_load_warnings) == 1
        and partial_load_warnings[0].startswith(EXPECTED_BASE_PARTIAL_LOAD_PREFIX)
        and all(
            sample in partial_load_warnings[0]
            for sample in EXPECTED_BASE_PARTIAL_LOAD_SAMPLES
        )
    ):
        raise RuntimeError(
            "SANA base checkpoint partial-load identity differs from the known "
            f"fresh AR additions: {warning_messages!r}"
        )

    architecture = trainer.architecture
    video_backbone = architecture.video_backbone
    if any(parameter.requires_grad for parameter in video_backbone.parameters()):
        raise RuntimeError("T5 video-backbone parameters are not frozen")
    named_trainables = [
        (name, parameter)
        for name, parameter in architecture.named_parameters()
        if parameter.requires_grad
    ]
    named_frozen = [
        (name, parameter)
        for name, parameter in architecture.named_parameters()
        if not parameter.requires_grad
    ]
    observed_roots = tuple(
        sorted({_parameter_root(name) for name, _parameter in named_trainables})
    )
    if observed_roots != tuple(sorted(EXPECTED_TRAINABLE_ROOTS)):
        raise RuntimeError(f"T5 trainable parameter roots differ: {observed_roots!r}")
    if len(named_trainables) != EXPECTED_TRAINABLE_TENSOR_COUNT:
        raise RuntimeError("T5 trainable tensor count differs")
    if sum(parameter.numel() for _name, parameter in named_trainables) != (
        EXPECTED_TRAINABLE_PARAMETER_COUNT
    ):
        raise RuntimeError("T5 trainable parameter count differs")
    frozen_versions = {name: parameter._version for name, parameter in named_frozen}

    dataset_index, episode = _find_fixed_sample(trainer.dataset)
    heldout_selections = _select_t5_samples(
        trainer.dataset,
        eligible_sample_manifest,
        cross_task_selection_manifest,
    )
    if (
        episode.dataset != SPATIAL_NAME
        or _sha256_file(episode.data_path()) != PARQUET_SHA256
    ):
        raise RuntimeError("T5 fixed real LIBERO sample identity differs")
    if (
        episode.episode_index != 0
        or episode.length != T5_TRAIN_EPISODE_LENGTH
        or episode.task_index != T5_TRAIN_TASK_INDEX
        or episode.task != T5_TRAIN_TASK_NAME
    ):
        raise RuntimeError("T5 fixed LIBERO episode metadata differs")
    sample_started = time.perf_counter()
    training_sample = trainer.dataset[dataset_index]
    sample_seconds = time.perf_counter() - sample_started
    if (
        training_sample["action_alignment"] != "observation_t_to_action_t"
        or training_sample["episode_index"] != 0
        or training_sample["episode_length"] != T5_TRAIN_EPISODE_LENGTH
        or training_sample["start_frame"] != 0
        or training_sample["task_index"] != T5_TRAIN_TASK_INDEX
        or training_sample["task_name"] != T5_TRAIN_TASK_NAME
        or training_sample["prompt"] != format_prompt_for_inference(T5_TRAIN_TASK_NAME)
    ):
        raise RuntimeError("T5 fixed LIBERO sample metadata differs")
    heldout_sample_seconds: dict[int, float] = {}
    spatial_assets_after_sample = {
        f"spatial_sample/{relative_path}": _verify_pinned_asset(
            spatial_root / relative_path, expected_sha256
        )
        for relative_path, expected_sha256 in sorted(
            EXPECTED_SPATIAL_SAMPLE_ASSET_SHA256.items()
        )
    }
    for key, observed in spatial_assets_after_sample.items():
        if external_assets[key] != observed:
            raise RuntimeError(f"T5 Spatial asset changed while loading: {key}")
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T5 selected-row stats changed while loading")
    actual_identity["spatial_assets_post_sample_sha256"] = _sha256_json(
        spatial_assets_after_sample
    )

    trainer._set_training_mode()
    if any(module.training for module in video_backbone.modules()):
        raise RuntimeError("T5 preserved video backbone left eval mode")
    prepare_inputs_calls = 0
    prepare_measurements: dict[str, dict[str, Any]] = {}

    def prepare_one(label: str, raw_sample: dict[str, Any]) -> dict[str, Any]:
        nonlocal prepare_inputs_calls
        torch.cuda.reset_peak_memory_stats(device)
        prepare_started = time.perf_counter()
        with torch.no_grad():
            prepared = architecture.prepare_inputs([raw_sample])
        prepare_inputs_calls += 1
        torch.cuda.synchronize(device)
        prepare_measurements[label] = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "seconds": time.perf_counter() - prepare_started,
        }
        if prepared.get("use_gradient_checkpointing") is not True:
            raise RuntimeError(f"T5 {label} inputs dropped gradient checkpointing")
        if prepared.get("use_gradient_checkpointing_offload") is not False:
            raise RuntimeError("T5 requires checkpoint offload=false")
        invalid_tensors = [
            key
            for key, value in prepared.items()
            if isinstance(value, torch.Tensor)
            and (value.requires_grad or value.grad_fn is not None)
        ]
        if invalid_tensors:
            raise RuntimeError(
                f"T5 {label} prepared tensors retain autograd state: "
                f"{invalid_tensors!r}"
            )
        return prepared

    # Preserve the T2/T3/T4 training-sample preparation path before introducing
    # any cross-task sample decode or encoding into the forward schedule.
    training_inputs = prepare_one("training", training_sample)
    heldout_inputs: dict[int, dict[str, Any]] = {}
    heldout_assets_after_sample = {}
    for selected in heldout_selections:
        episode_index = selected["episode_index"]
        sample_started = time.perf_counter()
        heldout_sample = trainer.dataset[selected["dataset_index"]]
        heldout_sample_seconds[episode_index] = time.perf_counter() - sample_started
        selected_episode = selected["episode"]
        if (
            selected_episode.dataset != SPATIAL_NAME
            or selected_episode.episode_index != episode_index
            or selected_episode.length != selected["episode_length"]
            or selected_episode.task_index != selected["task_index"]
            or selected_episode.task != selected["task"]
            or heldout_sample["action_alignment"] != "observation_t_to_action_t"
            or heldout_sample["episode_index"] != episode_index
            or heldout_sample["episode_length"] != selected["episode_length"]
            or heldout_sample["start_frame"] != 0
            or heldout_sample["task_index"] != selected["task_index"]
            or heldout_sample["task_name"] != selected["task"]
            or heldout_sample["prompt"] != format_prompt_for_inference(selected["task"])
        ):
            raise RuntimeError(
                f"T5 cross-task LIBERO sample metadata differs: {episode_index}"
            )
        heldout_inputs[episode_index] = prepare_one(
            f"heldout_episode_{episode_index}", heldout_sample
        )
        del heldout_sample
        for relative_path, expected_sha256 in sorted(
            T5_CROSS_TASK_SAMPLE_ASSET_SHA256[episode_index].items()
        ):
            key = f"cross_task_sample/episode_{episode_index:06d}/{relative_path}"
            observed = _verify_pinned_asset(
                spatial_root / relative_path,
                expected_sha256,
            )
            if external_assets[key] != observed:
                raise RuntimeError(f"T5 cross-task asset changed while loading: {key}")
            heldout_assets_after_sample[key] = observed
    actual_identity["cross_task_assets_post_sample_sha256"] = _sha256_json(
        heldout_assets_after_sample
    )
    cross_task_metadata_after_sample = {
        "meta/episodes.jsonl": _verify_pinned_asset(
            spatial_root / "meta/episodes.jsonl", T5_EPISODES_METADATA_SHA256
        ),
        "meta/tasks.jsonl": _verify_pinned_asset(
            spatial_root / "meta/tasks.jsonl", T5_TASKS_METADATA_SHA256
        ),
    }
    for relative_path, observed in cross_task_metadata_after_sample.items():
        key = f"cross_task_metadata/{relative_path}"
        if external_assets[key] != observed:
            raise RuntimeError(f"T5 metadata changed while loading: {key}")
    actual_identity["cross_task_metadata_post_sample_sha256"] = _sha256_json(
        cross_task_metadata_after_sample
    )
    if prepare_inputs_calls != 4:
        raise RuntimeError(
            f"T5 prepare_inputs call count differs: {prepare_inputs_calls}"
        )

    prepared_inputs = {"training": training_inputs}
    prepared_inputs.update(
        {
            f"heldout_episode_{episode_index}": heldout_inputs[episode_index]
            for episode_index in T5_CROSS_TASK_EPISODE_INDICES
        }
    )
    observed_storage: dict[tuple[str, int], tuple[str, str]] = {}
    for label, prepared in prepared_inputs.items():
        for key, value in prepared.items():
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                continue
            storage_key = (str(value.device), value.untyped_storage().data_ptr())
            previous = observed_storage.get(storage_key)
            if previous is not None and previous[0] != label:
                raise RuntimeError(
                    "T5 prepared tensors alias across samples: "
                    f"{previous!r} and {(label, key)!r}"
                )
            observed_storage[storage_key] = (label, key)

    def prepared_input_snapshot() -> list[tuple[Any, ...]]:
        return [
            (
                label,
                key,
                id(value),
                value.data_ptr(),
                value._version,
                str(value.dtype),
                tuple(value.shape),
            )
            for label, prepared in prepared_inputs.items()
            for key, value in sorted(prepared.items())
            if isinstance(value, torch.Tensor)
        ]

    initial_prepared_input_snapshot = prepared_input_snapshot()

    def prepared_context_report(
        prepared: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        context = prepared["context"]
        context_bytes = (
            context.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        )
        seq_lens = prepared.get("seq_lens")
        if isinstance(seq_lens, torch.Tensor):
            seq_lens_report: Any = [
                int(value) for value in seq_lens.detach().cpu().reshape(-1).tolist()
            ]
        elif isinstance(seq_lens, (list, tuple)):
            seq_lens_report = [int(value) for value in seq_lens]
        elif seq_lens is None:
            seq_lens_report = None
        else:
            seq_lens_report = int(seq_lens)
        return {
            "context_dtype": str(context.dtype),
            "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
            "context_shape": list(context.shape),
            "prompt": prompt,
            "seq_lens": seq_lens_report,
        }

    cross_task_context = {
        "training": prepared_context_report(training_inputs, training_sample["prompt"])
    }
    cross_task_context["training"].update(
        {
            "episode_index": 0,
            "task": T5_TRAIN_TASK_NAME,
            "task_index": T5_TRAIN_TASK_INDEX,
        }
    )
    for selected in heldout_selections:
        episode_index = selected["episode_index"]
        cross_task_context[selected["label"]] = prepared_context_report(
            heldout_inputs[episode_index],
            format_prompt_for_inference(selected["task"]),
        )
        cross_task_context[selected["label"]].update(
            {
                "episode_index": episode_index,
                "task": selected["task"],
                "task_index": selected["task_index"],
            }
        )
    context_digests = {row["context_sha256"] for row in cross_task_context.values()}
    if len(cross_task_context) != 4 or len(context_digests) != 4:
        raise RuntimeError("T5 prepared cross-task contexts are not pairwise distinct")

    model_groups = trainer._param_groups()
    if len(model_groups) != 2 or any(
        float(group["lr"]) != 1.0e-4 for group in model_groups
    ):
        raise RuntimeError("T5 optimizer group/LR structure differs")
    params = [parameter for group in model_groups for parameter in group["params"]]
    parameter_ids = [id(parameter) for parameter in params]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != {
        id(parameter) for _name, parameter in named_trainables
    }:
        raise RuntimeError("T5 optimizer parameters differ from trainables")
    optimizer_groups, master_pairs = trainer._optimizer_param_groups(model_groups)
    if len(master_pairs) != len(params):
        raise RuntimeError("T5 FP32 master coverage differs")
    if [id(model) for model, _master in master_pairs] != parameter_ids:
        raise RuntimeError("T5 FP32 master/model ordering differs")
    masters = [master for _model, master in master_pairs]
    initial_master_ids = [id(master) for master in masters]
    if (
        len(initial_master_ids) != len(set(initial_master_ids))
        or any(model.dtype != torch.bfloat16 for model, _master in master_pairs)
        or any(master.dtype != torch.float32 for _model, master in master_pairs)
        or any(not master.requires_grad for _model, master in master_pairs)
        or any(model.shape != master.shape for model, master in master_pairs)
        or any(model.device != master.device for model, master in master_pairs)
    ):
        raise RuntimeError("T5 FP32 master identity/dtype differs")
    if [
        id(parameter) for group in optimizer_groups for parameter in group["params"]
    ] != initial_master_ids:
        raise RuntimeError("T5 optimizer groups do not contain the FP32 masters")
    if [len(group["params"]) for group in optimizer_groups] != [
        len(group["params"]) for group in model_groups
    ] or [float(group["lr"]) for group in optimizer_groups] != [
        float(group["lr"]) for group in model_groups
    ]:
        raise RuntimeError("T5 FP32 master optimizer group structure differs")
    trainable_name_by_id = {id(parameter): name for name, parameter in named_trainables}
    named_masters = [
        (trainable_name_by_id[id(model)], master) for model, master in master_pairs
    ]
    model_by_name = dict(named_trainables)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(cfg.training.weight_decay),
        betas=(0.9, 0.95),
    )

    def probe_mutation_snapshot() -> dict[str, Any]:
        if any(parameter.grad is not None for _name, parameter in named_trainables):
            raise RuntimeError("T5 update-free probe observed model gradients")
        if any(parameter.grad is not None for _name, parameter in named_frozen):
            raise RuntimeError("T5 update-free probe observed frozen gradients")
        if any(master.grad is not None for _name, master in named_masters):
            raise RuntimeError("T5 update-free probe observed master gradients")
        optimizer_state = []
        for name, master in named_masters:
            for key, value in sorted(optimizer.state.get(master, {}).items()):
                if isinstance(value, torch.Tensor):
                    observed = (
                        "tensor",
                        id(value),
                        value.data_ptr(),
                        value._version,
                    )
                else:
                    observed = ("value", repr(value))
                optimizer_state.append((name, key, observed))
        return {
            "architecture_buffers": [
                (name, id(buffer), buffer.data_ptr(), buffer._version)
                for name, buffer in architecture.named_buffers()
            ],
            "architecture_module_modes": [
                (name, module.training) for name, module in architecture.named_modules()
            ],
            "architecture_parameters": [
                (name, id(parameter), parameter.data_ptr(), parameter._version)
                for name, parameter in architecture.named_parameters()
            ],
            "master_parameters": [
                (name, id(master), master.data_ptr(), master._version)
                for name, master in named_masters
            ],
            "optimizer_state": optimizer_state,
            "prepared_inputs": prepared_input_snapshot(),
        }

    recipe_capture: dict[str, list[dict[str, Any]]] = {
        "action": [],
        "video": [],
    }
    architecture_forward_calls = 0
    backward_calls = 0
    optimizer_step_calls = 0
    video_embedder = video_backbone._dit.t_embedder
    original_video_embedder_forward = video_embedder.forward
    original_action_prepare_state = architecture.action_backbone.prepare_state
    original_architecture_forward = architecture.forward

    def capture_video_timestep(timestep):
        recipe_capture["video"].append(_tensor_observation(timestep))
        return original_video_embedder_forward(timestep)

    def capture_action_timestep(*positional, **keyword):
        if len(positional) >= 2 and isinstance(positional[1], torch.Tensor):
            recipe_capture["action"].append(_tensor_observation(positional[1]))
        token_timestep = keyword.get("token_timesteps")
        if isinstance(token_timestep, torch.Tensor):
            recipe_capture["action"].append(_tensor_observation(token_timestep))
        return original_action_prepare_state(*positional, **keyword)

    def capture_architecture_forward(*positional, **keyword):
        nonlocal architecture_forward_calls
        architecture_forward_calls += 1
        return original_architecture_forward(*positional, **keyword)

    video_embedder.forward = capture_video_timestep
    architecture.action_backbone.prepare_state = capture_action_timestep
    architecture.forward = capture_architecture_forward

    def fixed_forward(
        *, backward: bool, forward_inputs: dict[str, Any], recipe_seed: int
    ) -> tuple[dict[str, float], str]:
        nonlocal backward_calls
        recipe_capture["action"].clear()
        recipe_capture["video"].clear()
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()
        torch_cpu_rng_before = torch.random.get_rng_state().clone()
        torch_cuda_rng_before = torch.cuda.get_rng_state().clone()
        with _fixed_rng_recipe(
            torch,
            seed=recipe_seed,
            cuda_device=torch.cuda.current_device(),
        ):
            result = architecture.compute_loss(
                lambda_video=trainer.lambda_video,
                lambda_action=trainer.lambda_action,
                **forward_inputs,
            )
            loss = result["loss"]
            loss_action = result["loss_action"]
            if (
                loss.ndim != 0
                or loss_action.ndim != 0
                or not bool(torch.isfinite(loss).item())
                or not bool(torch.isfinite(loss_action).item())
            ):
                raise RuntimeError("T5 produced a non-finite scalar loss")
            scalars = {
                "loss": _tensor_scalar(loss),
                "loss_action": _tensor_scalar(loss_action),
            }
            tolerance = 1.0e-6 + 1.0e-6 * abs(scalars["loss_action"])
            if abs(scalars["loss"] - scalars["loss_action"]) > tolerance:
                raise RuntimeError("T5 total/action loss identity differs")
            signature = _sha256_json(recipe_capture)
            if not recipe_capture["action"] or not recipe_capture["video"]:
                raise RuntimeError("T5 stochastic recipe capture is incomplete")
            if backward:
                loss.backward()
                backward_calls += 1
        numpy_rng_after = np.random.get_state()
        if (
            random.getstate() != python_rng_before
            or numpy_rng_after[0] != numpy_rng_before[0]
            or not np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
            or numpy_rng_after[2:] != numpy_rng_before[2:]
            or not torch.equal(torch.random.get_rng_state(), torch_cpu_rng_before)
            or not torch.equal(torch.cuda.get_rng_state(), torch_cuda_rng_before)
        ):
            raise RuntimeError("T5 fixed recipe did not restore caller RNG state")
        del loss, loss_action, result
        return scalars, signature

    per_step: list[dict[str, Any]] = []
    training_recipe_signatures: list[str] = []
    heldout_before: dict[int, dict[str, float]] = {}
    heldout_after: dict[int, dict[str, float]] = {}
    heldout_sample_signatures: dict[int, list[str]] = {
        episode_index: [] for episode_index in T5_CROSS_TASK_EPISODE_INDICES
    }
    initial_probe: dict[str, float]
    final_probe: dict[str, float]
    master_probes = None
    model_probes = None
    first_master_update = None
    first_projected_update = None
    update_sentinels: list[dict[str, Any]] = []
    update_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for parameter in params:
            parameter.grad = None
        optimizer.zero_grad(set_to_none=True)
        if optimizer.state:
            raise RuntimeError("T5 optimizer state exists before held-out pre probes")
        heldout_pre_state = probe_mutation_snapshot()
        for episode_index in T5_CROSS_TASK_EPISODE_INDICES:
            scalars, signature = fixed_forward(
                backward=False,
                forward_inputs=heldout_inputs[episode_index],
                recipe_seed=args.loss_recipe_seed,
            )
            heldout_before[episode_index] = scalars
            heldout_sample_signatures[episode_index].append(signature)
        if probe_mutation_snapshot() != heldout_pre_state or optimizer.state:
            raise RuntimeError("T5 held-out pre probes mutated training state")

        initial_probe, initial_signature = fixed_forward(
            backward=False,
            forward_inputs=training_inputs,
            recipe_seed=args.loss_recipe_seed,
        )
        training_recipe_signatures.append(initial_signature)

        for step_index in range(T5_UPDATE_STEPS):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for parameter in params:
                parameter.grad = None
            step_scalars, recipe_signature = fixed_forward(
                backward=True,
                forward_inputs=training_inputs,
                recipe_seed=args.loss_recipe_seed,
            )
            training_recipe_signatures.append(recipe_signature)
            if step_index == 0:
                tolerance = 1.0e-6 + 1.0e-6 * abs(initial_probe["loss_action"])
                if (
                    abs(step_scalars["loss_action"] - initial_probe["loss_action"])
                    > tolerance
                ):
                    raise RuntimeError(
                        "T5 fixed initial loss recipe is not reproducible"
                    )

            missing_gradients = []
            nonzero_gradient_tensors = 0
            nonzero_gradient_roots: set[str] = set()
            max_gradient_abs = 0.0
            for name, parameter in named_trainables:
                gradient = parameter.grad
                if gradient is None:
                    missing_gradients.append(name)
                    continue
                if not bool(torch.isfinite(gradient).all().item()):
                    raise RuntimeError(
                        f"T5 non-finite gradient at step {step_index + 1}: {name}"
                    )
                maximum = float(gradient.detach().abs().max().float().item())
                max_gradient_abs = max(max_gradient_abs, maximum)
                if maximum > 0.0:
                    nonzero_gradient_tensors += 1
                    nonzero_gradient_roots.add(_parameter_root(name))
            if missing_gradients:
                raise RuntimeError(
                    f"T5 missing gradients at step {step_index + 1}: "
                    f"{missing_gradients[:8]!r}"
                )
            if nonzero_gradient_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    f"T5 nonzero-gradient roots differ at step {step_index + 1}: "
                    f"{sorted(nonzero_gradient_roots)!r}"
                )
            if any(parameter.grad is not None for _name, parameter in named_frozen):
                raise RuntimeError("T5 frozen parameters received gradients")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, float(cfg.training.grad_clip)
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError("T5 gradient norm is non-finite")
            trainer._sync_master_gradients(master_pairs)
            if step_index == 0:
                invalid_master_gradients = [
                    name
                    for name, master in named_masters
                    if master.grad is None
                    or master.grad.dtype != torch.float32
                    or not bool(torch.isfinite(master.grad).all().item())
                    or not bool(torch.count_nonzero(master.grad).item())
                ]
                if invalid_master_gradients:
                    raise RuntimeError(
                        "T5 first-step masters lack finite nonzero FP32 gradients: "
                        f"{invalid_master_gradients[:8]!r}"
                    )
                master_probes = _capture_update_probes(named_masters)
                model_probes = _capture_update_probes(named_trainables)
                if {row["name"] for row in master_probes} != {
                    name for name, _master in named_masters
                }:
                    raise RuntimeError("T5 first-step master probe coverage differs")

                for root in EXPECTED_TRAINABLE_ROOTS:
                    candidates = sorted(
                        (
                            probe
                            for probe in master_probes
                            if _parameter_root(probe["name"]) == root
                        ),
                        key=lambda probe: abs(probe["gradient"]),
                        reverse=True,
                    )[:4]
                    if len(candidates) != 4:
                        raise RuntimeError(f"T5 lacks four update sentinels for {root}")
                    update_sentinels.extend(candidates)

            sentinel_before = [
                {
                    "before_master": _tensor_scalar(
                        probe["parameter"].detach().reshape(-1)[probe["index"]]
                    ),
                    "before_model": _tensor_scalar(
                        model_by_name[probe["name"]]
                        .detach()
                        .reshape(-1)[probe["index"]]
                    ),
                    "gradient": _tensor_scalar(
                        probe["parameter"].grad.detach().reshape(-1)[probe["index"]]
                    ),
                    "index": probe["index"],
                    "name": probe["name"],
                }
                for probe in update_sentinels
            ]

            optimizer.step()
            optimizer_step_calls += 1
            if step_index == 0:
                first_master_update, changed_master_ids = _summarize_updates(
                    master_probes
                )
                if len(changed_master_ids) != len(named_masters):
                    raise RuntimeError("T5 first step did not update every FP32 master")
            trainer._copy_master_parameters_to_model(master_pairs)
            sentinel_updates = []
            for before, probe in zip(sentinel_before, update_sentinels, strict=True):
                after_master = _tensor_scalar(
                    probe["parameter"].detach().reshape(-1)[probe["index"]]
                )
                after_model = _tensor_scalar(
                    model_by_name[probe["name"]].detach().reshape(-1)[probe["index"]]
                )
                row = {
                    **before,
                    "after_master": after_master,
                    "after_model": after_model,
                    "master_delta": after_master - before["before_master"],
                    "model_delta": after_model - before["before_model"],
                    "root": _parameter_root(probe["name"]),
                }
                if not all(
                    math.isfinite(row[key])
                    for key in (
                        "after_master",
                        "after_model",
                        "master_delta",
                        "model_delta",
                    )
                ):
                    raise RuntimeError("T5 update sentinel became non-finite")
                sentinel_updates.append(row)
            changed_master_sentinels = [
                row for row in sentinel_updates if row["master_delta"] != 0.0
            ]
            changed_master_sentinel_roots = {
                row["root"] for row in changed_master_sentinels
            }
            if changed_master_sentinel_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    "T5 sampled FP32 master update roots differ at step "
                    f"{step_index + 1}: "
                    f"{sorted(changed_master_sentinel_roots)!r}"
                )
            sentinel_adam_steps = {
                _tensor_scalar(optimizer.state[probe["parameter"]]["step"])
                for probe in update_sentinels
            }
            if sentinel_adam_steps != {float(step_index + 1)}:
                raise RuntimeError(
                    "T5 sampled AdamW step counters differ at step "
                    f"{step_index + 1}: {sorted(sentinel_adam_steps)!r}"
                )
            if step_index in {0, T5_UPDATE_STEPS - 1}:
                projection_mismatches = [
                    trainable_name_by_id[id(model)]
                    for model, master in master_pairs
                    if not torch.equal(
                        model.detach(), master.detach().to(dtype=model.dtype)
                    )
                ]
                if projection_mismatches:
                    raise RuntimeError(
                        f"T5 BF16 projection differs: {projection_mismatches[:8]!r}"
                    )
                if not _optimizer_state_is_finite(optimizer):
                    raise RuntimeError("T5 AdamW state is non-finite")
            if step_index == 0:
                first_projected_update, _changed_model_ids = _summarize_updates(
                    model_probes
                )
            torch.cuda.synchronize(device)
            per_step.append(
                {
                    "gradient_nonzero_roots": sorted(nonzero_gradient_roots),
                    "gradient_nonzero_tensor_count": nonzero_gradient_tensors,
                    "loss": step_scalars["loss"],
                    "loss_action": step_scalars["loss_action"],
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "pre_clip_global_norm": _tensor_scalar(grad_norm),
                    "pre_clip_max_abs": max_gradient_abs,
                    "recipe_signature_sha256": recipe_signature,
                    "sampled_bf16_update_changed_count": sum(
                        row["model_delta"] != 0.0 for row in sentinel_updates
                    ),
                    "sampled_bf16_update_changed_roots": sorted(
                        {
                            row["root"]
                            for row in sentinel_updates
                            if row["model_delta"] != 0.0
                        }
                    ),
                    "sampled_bf16_update_l2": math.sqrt(
                        sum(row["model_delta"] ** 2 for row in sentinel_updates)
                    ),
                    "sampled_bf16_update_max_abs": max(
                        abs(row["model_delta"]) for row in sentinel_updates
                    ),
                    "sampled_master_update_changed_count": len(
                        changed_master_sentinels
                    ),
                    "sampled_master_update_changed_roots": sorted(
                        changed_master_sentinel_roots
                    ),
                    "sampled_master_update_l2": math.sqrt(
                        sum(row["master_delta"] ** 2 for row in sentinel_updates)
                    ),
                    "sampled_master_update_max_abs": max(
                        abs(row["master_delta"]) for row in sentinel_updates
                    ),
                    "sampled_optimizer_step_values": sorted(sentinel_adam_steps),
                    "seconds": time.perf_counter() - step_started,
                    "step": step_index + 1,
                }
            )

        optimizer.zero_grad(set_to_none=True)
        for parameter in params:
            parameter.grad = None
        heldout_post_state = probe_mutation_snapshot()
        final_probe, final_signature = fixed_forward(
            backward=False,
            forward_inputs=training_inputs,
            recipe_seed=args.loss_recipe_seed,
        )
        training_recipe_signatures.append(final_signature)
        for episode_index in T5_CROSS_TASK_EPISODE_INDICES:
            scalars, signature = fixed_forward(
                backward=False,
                forward_inputs=heldout_inputs[episode_index],
                recipe_seed=args.loss_recipe_seed,
            )
            heldout_after[episode_index] = scalars
            heldout_sample_signatures[episode_index].append(signature)
        if probe_mutation_snapshot() != heldout_post_state:
            raise RuntimeError("T5 held-out post probes mutated training state")
    finally:
        video_embedder.forward = original_video_embedder_forward
        architecture.action_backbone.prepare_state = original_action_prepare_state
        architecture.forward = original_architecture_forward

    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - update_started
    if architecture_forward_calls != T5_UPDATE_STEPS + 8:
        raise RuntimeError(
            f"T5 architecture forward count differs: {architecture_forward_calls}"
        )
    if backward_calls != T5_UPDATE_STEPS:
        raise RuntimeError(f"T5 backward call count differs: {backward_calls}")
    if optimizer_step_calls != T5_UPDATE_STEPS:
        raise RuntimeError(
            f"T5 optimizer step call count differs: {optimizer_step_calls}"
        )
    if (
        len(training_recipe_signatures) != T5_UPDATE_STEPS + 2
        or len(set(training_recipe_signatures)) != 1
    ):
        raise RuntimeError("T5 training RNG recipe signatures differ across forwards")
    invalid_heldout_signatures = {
        episode_index: signatures
        for episode_index, signatures in heldout_sample_signatures.items()
        if len(signatures) != 2 or len(set(signatures)) != 1
    }
    if invalid_heldout_signatures:
        raise RuntimeError(
            f"T5 held-out RNG recipe signatures differ: {invalid_heldout_signatures!r}"
        )
    all_recipe_signatures = training_recipe_signatures + [
        signature
        for episode_index in T5_CROSS_TASK_EPISODE_INDICES
        for signature in heldout_sample_signatures[episode_index]
    ]
    if len(all_recipe_signatures) != T5_UPDATE_STEPS + 8 or set(
        all_recipe_signatures
    ) != {T5_FIXED_RECIPE_SIGNATURE_SHA256}:
        raise RuntimeError("T5 fixed loss recipe signature differs across forwards")
    final_optimizer_master_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if final_optimizer_master_ids != initial_master_ids:
        raise RuntimeError("T5 FP32 master objects changed during the loop")
    if prepared_input_snapshot() != initial_prepared_input_snapshot:
        raise RuntimeError("T5 prepared input identities or versions changed")
    frozen_version_changes = [
        name
        for name, parameter in named_frozen
        if parameter._version != frozen_versions[name]
    ]
    if frozen_version_changes:
        raise RuntimeError(
            f"T5 frozen parameters changed: {frozen_version_changes[:8]!r}"
        )
    if any(not bool(torch.isfinite(master).all().item()) for master in masters):
        raise RuntimeError("T5 FP32 masters became non-finite")
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in params):
        raise RuntimeError("T5 BF16 trainables became non-finite")
    if not _optimizer_state_is_finite(optimizer):
        raise RuntimeError("T5 final AdamW state is non-finite")
    adam_steps: list[float] = []
    for master in masters:
        state = optimizer.state.get(master)
        if not state or "step" not in state:
            raise RuntimeError("T5 AdamW state is missing for an FP32 master")
        step_value = _tensor_scalar(state["step"])
        if not math.isfinite(step_value):
            raise RuntimeError("T5 AdamW step counter is non-finite")
        adam_steps.append(step_value)
    if set(adam_steps) != {float(T5_UPDATE_STEPS)}:
        raise RuntimeError(
            f"T5 AdamW step counters differ: {sorted(set(adam_steps))!r}"
        )

    cumulative_master_update, cumulative_master_ids = _summarize_updates(master_probes)
    cumulative_projected_update, _cumulative_model_ids = _summarize_updates(
        model_probes
    )
    if len(cumulative_master_ids) != len(named_masters) or set(
        cumulative_master_update["changed_trainable_roots"]
    ) != set(EXPECTED_TRAINABLE_ROOTS):
        raise RuntimeError("T5 cumulative FP32 master update coverage differs")

    learnability = _learnability_summary(
        initial_probe["loss_action"],
        [row["loss_action"] for row in per_step],
        final_probe["loss_action"],
    )
    cross_task_transfer = _cross_task_transfer_summary(
        {
            episode_index: heldout_before[episode_index]["loss_action"]
            for episode_index in T5_CROSS_TASK_EPISODE_INDICES
        },
        {
            episode_index: heldout_after[episode_index]["loss_action"]
            for episode_index in T5_CROSS_TASK_EPISODE_INDICES
        },
    )
    selection_by_episode = {
        row["episode_index"]: row for row in cross_task_selection_manifest["samples"]
    }
    cross_task_transfer_report = {
        **cross_task_transfer,
        "per_sample": [
            {
                **row,
                "after_total_loss": heldout_after[row["episode_index"]]["loss"],
                "assets": selection_by_episode[row["episode_index"]]["assets"],
                "before_total_loss": heldout_before[row["episode_index"]]["loss"],
                "episode_length": selection_by_episode[row["episode_index"]][
                    "episode_length"
                ],
                "episode_selection_payload_sha256": selection_by_episode[
                    row["episode_index"]
                ]["episode_selection_payload_sha256"],
                "label": selection_by_episode[row["episode_index"]]["label"],
                "post_recipe_signature_sha256": heldout_sample_signatures[
                    row["episode_index"]
                ][1],
                "pre_recipe_signature_sha256": heldout_sample_signatures[
                    row["episode_index"]
                ][0],
                "task": selection_by_episode[row["episode_index"]]["task"],
                "task_index": selection_by_episode[row["episode_index"]]["task_index"],
                "task_selection_payload_sha256": selection_by_episode[
                    row["episode_index"]
                ]["task_selection_payload_sha256"],
                "start_frame": 0,
            }
            for row in cross_task_transfer["per_sample"]
        ],
        "loss_recipe_seed": args.loss_recipe_seed,
        "probe_state_unchanged": True,
        "samples_never_entered_backward_or_update": True,
    }
    projected_roots = set(cumulative_projected_update["changed_trainable_roots"])
    secondary_diagnostic_warnings = []
    if not learnability["loss_gate"]:
        secondary_diagnostic_warnings.append(
            "training-recipe T2 learnability gate did not reproduce; "
            "this is not part of the frozen T5 primary transfer gate"
        )
    if projected_roots != set(EXPECTED_TRAINABLE_ROOTS):
        secondary_diagnostic_warnings.append(
            "20-step BF16 projection did not visibly change every trainable root; "
            "this is not part of the frozen T5 primary transfer gate"
        )
    verdict_reasons = []
    if not cross_task_transfer["transfer_gate"]:
        verdict_reasons.append(
            "same-suite cross-task transfer did not clear the median/2-of-3 gate"
        )
    verdict = (
        "T5_CROSS_TASK_TRANSFER_GO"
        if not verdict_reasons
        else "T5_CROSS_TASK_TRANSFER_INCONCLUSIVE"
    )
    gpu_after = _query_gpu(args.physical_gpu)
    architecture_report = {
        "action_dim": architecture.action_dim,
        "class": type(architecture).__name__,
        "continuous_timestep_conditioning": True,
        "parameter_count": sum(
            parameter.numel() for parameter in architecture.parameters()
        ),
        "proprio_dim": architecture.proprio_dim,
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
        "trainable_parameter_tensor_count": EXPECTED_TRAINABLE_TENSOR_COUNT,
        "trainable_roots": list(EXPECTED_TRAINABLE_ROOTS),
        "video_backbone_frozen": True,
    }
    assets_report = {
        "config_path": str(config_path),
        "config_sha256": actual_config_sha256,
        "external_files": external_assets,
        "external_files_manifest_sha256": actual_identity["external_assets_sha256"],
        "cross_task_files_post_sample": heldout_assets_after_sample,
        "cross_task_files_post_sample_manifest_sha256": actual_identity[
            "cross_task_assets_post_sample_sha256"
        ],
        "cross_task_metadata_post_sample": cross_task_metadata_after_sample,
        "cross_task_metadata_post_sample_manifest_sha256": actual_identity[
            "cross_task_metadata_post_sample_sha256"
        ],
        "spatial_files_post_sample": spatial_assets_after_sample,
        "spatial_files_post_sample_manifest_sha256": actual_identity[
            "spatial_assets_post_sample_sha256"
        ],
        "stats_path": str(stats_path.resolve()),
        "stats_population_sha256": SELECTED_POPULATION_SHA256,
        "stats_sha256": SELECTED_STATS_SHA256,
        "stats_validation": {
            "formal_training_admission_granted": False,
            "live_numeric_pretrainer_validation_executed": True,
            "materialized_selected_row_v2_required": True,
        },
    }
    identity_report = {
        **actual_identity,
        "identity_sha256": _sha256_json(actual_identity),
    }

    def prepared_shape_report(prepared: dict[str, Any]) -> dict[str, Any]:
        return {
            "action_is_pad_true": int(prepared["action_is_pad"].sum().item()),
            "actions": list(prepared["actions"].shape),
            "context": list(prepared["context"].shape),
            "input_latents": list(prepared["input_latents"].shape),
            "proprio_seq": list(prepared["proprio_seq"].shape),
            "proprio_state": list(prepared["proprio_state"].shape),
            "video_is_pad_true": int(prepared["video_is_pad"].sum().item()),
        }

    training_inputs_report = prepared_shape_report(training_inputs)
    heldout_inputs_report = {
        selection_by_episode[episode_index]["label"]: prepared_shape_report(
            heldout_inputs[episode_index]
        )
        for episode_index in T5_CROSS_TASK_EPISODE_INDICES
    }
    learnability_report = {
        **learnability,
        "final_probe_total_loss": final_probe["loss"],
        "initial_probe_action_loss": initial_probe["loss_action"],
        "initial_probe_total_loss": initial_probe["loss"],
        "pre_update_action_loss_curve": [row["loss_action"] for row in per_step],
        "pre_update_total_loss_curve": [row["loss"] for row in per_step],
        "role": "secondary_diagnostic_not_t5_primary_gate",
    }
    optimizer_report = {
        "adam_step_values": sorted(set(adam_steps)),
        "betas": [0.9, 0.95],
        "constant_learning_rates": [
            float(group["lr"]) for group in optimizer.param_groups
        ],
        "cumulative_master_update": cumulative_master_update,
        "cumulative_projected_bf16_update": cumulative_projected_update,
        "first_master_update": first_master_update,
        "first_projected_bf16_update": first_projected_update,
        "master_object_identity_persistent": True,
        "optimizer_master_parameter_count": len(master_pairs),
        "optimizer_master_parameter_elements": sum(
            master.numel() for master in masters
        ),
        "optimizer_state_finite": True,
        "production_accumulation_equivalent": False,
        "production_schedule_equivalent": False,
        "projection_exact_after_steps": [1, T5_UPDATE_STEPS],
        "update_sentinel_count": len(update_sentinels),
        "update_sentinel_names": [
            {
                "index": probe["index"],
                "name": probe["name"],
                "root": _parameter_root(probe["name"]),
            }
            for probe in update_sentinels
        ],
        "weight_decay": float(cfg.training.weight_decay),
    }
    randomness_report = {
        "eligible_cross_task_manifest": eligible_sample_manifest,
        "eligible_cross_task_manifest_sha256": T5_ELIGIBLE_MANIFEST_SHA256,
        "heldout_signatures_per_label": {
            selection_by_episode[episode_index]["label"]: (
                heldout_sample_signatures[episode_index]
            )
            for episode_index in T5_CROSS_TASK_EPISODE_INDICES
        },
        "initialization_seed": args.initialization_seed,
        "training_loss_recipe_seed": args.loss_recipe_seed,
        "recipe_reset": "python+numpy+torch-cpu+torch-cuda before every forward",
        "rng_state_restored_after_every_forward": True,
        "selected_cross_task_manifest": cross_task_selection_manifest,
        "selected_cross_task_manifest_sha256": T5_SELECTION_MANIFEST_SHA256,
        "total_recipe_signatures": len(all_recipe_signatures),
        "training_recipe_signature_sha256": training_recipe_signatures[0],
        "training_recipe_signatures": len(training_recipe_signatures),
        "unique_loss_recipe_signature_count": len(set(all_recipe_signatures)),
        "unique_training_recipe_signatures": len(set(training_recipe_signatures)),
    }
    training_sample_report = {
        "action_alignment": training_sample["action_alignment"],
        "dataset": episode.dataset,
        "episode_index": episode.episode_index,
        "episode_length": episode.length,
        "parquet_sha256": PARQUET_SHA256,
        "start_frame": training_sample["start_frame"],
        "task": training_sample["task_name"],
        "task_index": training_sample["task_index"],
    }
    scope_report = {
        "architecture_forwards_total": T5_UPDATE_STEPS + 8,
        "architecture_measurement_forwards": 8,
        "architecture_training_forwards": T5_UPDATE_STEPS,
        "backward_calls": backward_calls,
        "benchmark_evaluation_executed": False,
        "formal_training_executed": False,
        "cross_task_measurement_forwards": 6,
        "cross_task_samples_in_backward_or_update": 0,
        "main_recipe_measurement_forwards": 2,
        "optimizer_steps": optimizer_step_calls,
        "post_probe_reprepare_calls": 0,
        "prepare_inputs_calls": prepare_inputs_calls,
        "probe_order": [
            "cross_task_pre_h1",
            "cross_task_pre_h2",
            "cross_task_pre_h3",
            "training_recipe_pre",
            "training_recipe_update_1_through_20",
            "training_recipe_post",
            "cross_task_post_h1",
            "cross_task_post_h2",
            "cross_task_post_h3",
        ],
        "sana_base_pretrained_checkpoint_loaded": True,
        "sana_wam_training_checkpoint_loaded": False,
        "sana_wam_training_checkpoint_saved": False,
        "simulator_executed": False,
    }
    current_training_core = _training_core_projection(
        {
            "architecture": architecture_report,
            "assets": assets_report,
            "identity": identity_report,
            "inputs": training_inputs_report,
            "learnability": learnability_report,
            "optimizer": optimizer_report,
            "per_step": per_step,
            "randomness": randomness_report,
            "sample": training_sample_report,
            "scope": scope_report,
        }
    )
    expected_training_core_sha256 = _sha256_json(t4_training_core)
    observed_training_core_sha256 = _sha256_json(current_training_core)
    if (
        expected_training_core_sha256 != T5_TRAINING_CORE_SHA256
        or observed_training_core_sha256 != T5_TRAINING_CORE_SHA256
        or current_training_core != t4_training_core
    ):
        raise RuntimeError(
            "T5 training core drifted from frozen T4: "
            f"expected_sha256={expected_training_core_sha256} "
            f"observed_sha256={observed_training_core_sha256}"
        )
    training_core_reproduction = {
        "equal_to_frozen_t4": True,
        "expected_projection_sha256": expected_training_core_sha256,
        "observed_projection_sha256": observed_training_core_sha256,
        "predecessor_result_sha256": T4_PREDECESSOR_RESULT_SHA256,
    }
    report = {
        "architecture": architecture_report,
        "assets": assets_report,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "after": gpu_after,
            "before": gpu_before,
            "cuda_visible_devices": visible,
            "lock_path": str(lock_path),
            "preconstruction": gpu_preconstruction,
            "torch_cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "visible_ordinal": 0,
        },
        "cross_task_context": cross_task_context,
        "cross_task_inputs": heldout_inputs_report,
        "cross_task_transfer": cross_task_transfer_report,
        "identity": identity_report,
        "inputs": training_inputs_report,
        "interpretation_limits": {
            "benchmark_success_claimed": False,
            "cross_suite_transfer_claimed": False,
            "heldout_axis": (
                "same-suite distinct-task plus distinct-episode optimizer-update "
                "exposure"
            ),
            "normalization_population_includes_probe_samples": True,
            "strict_dataset_holdout": False,
            "task_text_only_isolation": False,
        },
        "learnability": learnability_report,
        "memory": {
            "build": build_memory,
            "prepare": prepare_measurements,
            "update_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "update_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "optimizer": optimizer_report,
        "per_step": per_step,
        "randomness": randomness_report,
        "execution_result": "PASS",
        "harness_result": "PASS",
        "result": verdict,
        "run": {
            "nonce": args.nonce,
            "root": str(run_root),
        },
        "sample": training_sample_report,
        "schema_version": "sana-wam-libero-t5-cross-task-transfer-v1",
        "scope": scope_report,
        "timings_seconds": {
            "model_and_dataset_build": build_seconds,
            "prepare_real_samples": {
                label: measurement["seconds"]
                for label, measurement in prepare_measurements.items()
            },
            "sample_load": sample_seconds,
            "cross_task_sample_load": {
                str(key): value for key, value in heldout_sample_seconds.items()
            },
            "twenty_updates_and_eight_measurements": update_seconds,
        },
        "training_core_reproduction": training_core_reproduction,
        "valid_run": True,
        "scientific_verdict": verdict,
        "secondary_diagnostic_warnings": secondary_diagnostic_warnings,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "warnings": warning_messages,
    }
    if any(run_root.iterdir()):
        raise RuntimeError("T5 run root was not empty before terminal write")
    _write_terminal_report(run_root / "RESULT.json", report)
    _freeze_run_root(run_root)
    _ACTIVE_RUN_ROOT = None
    gpu_lock.close()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as error:
        if _ACTIVE_RUN_ROOT is not None:
            try:
                _terminalize_failure(_ACTIVE_RUN_ROOT, error)
            except BaseException as freeze_error:
                print(
                    "failed to terminalize T5 root: "
                    f"{type(freeze_error).__name__}: {freeze_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    raise SystemExit(exit_code)
