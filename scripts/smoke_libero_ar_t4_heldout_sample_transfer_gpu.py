#!/usr/bin/env python
"""Run the bounded LIBERO T4 held-out-sample transfer screen."""

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


T4_UPDATE_STEPS = 20
T4_LEARNABILITY_RATIO = 0.95
T4_INITIALIZATION_SEED = 20260806
T4_LOSS_RECIPE_SEED = 20260826
T4_HELDOUT_MEDIAN_RATIO = 0.95
T4_HELDOUT_MIN_IMPROVED = 2
T4_TRAIN_EPISODE_LENGTH = 110
T4_TASK_INDEX = 0
T4_TASK_NAME = "pick up the black bowl next to the cookie box and place it on the plate"
T4_HELDOUT_EPISODE_INDICES = (16, 405, 40)
T4_HELDOUT_EPISODE_LENGTHS = {16: 111, 405: 130, 40: 141}
T4_HELDOUT_SELECTION_SHA256 = {
    16: "02938329ed205ce9032b9bc7c54d0e3f1f1716d86b8526bdc3e220bd784b6c6e",
    405: "09d59c3d6bec297af4f0d04d034d0f8ba65b2db2552f93cf8325946b1e6e3c72",
    40: "0d8d7c4a8f60985b3b0d2259bffae34b99c911bce2757b7559294e02e229fef8",
}
T4_ELIGIBLE_EPISODES = (
    (13, 113),
    (16, 111),
    (30, 125),
    (31, 157),
    (40, 141),
    (43, 113),
    (48, 115),
    (53, 117),
    (68, 124),
    (72, 141),
    (75, 135),
    (76, 133),
    (77, 125),
    (79, 122),
    (110, 122),
    (113, 124),
    (154, 133),
    (155, 133),
    (157, 152),
    (160, 118),
    (168, 123),
    (185, 106),
    (187, 119),
    (195, 129),
    (211, 122),
    (212, 124),
    (224, 123),
    (241, 146),
    (265, 130),
    (273, 127),
    (278, 122),
    (283, 121),
    (303, 114),
    (314, 133),
    (343, 120),
    (345, 113),
    (346, 137),
    (350, 123),
    (351, 124),
    (352, 108),
    (363, 108),
    (384, 155),
    (394, 124),
    (400, 130),
    (405, 130),
)
T4_ELIGIBLE_MANIFEST_SHA256 = (
    "e77462aa3380b1b14bee646b0965eb465baca76bff3d0805f9fc7d54d61419ef"
)
T4_EPISODES_METADATA_SHA256 = (
    "690349688e96d984007bf2cca3ccf2683dfc809f03dab55e1159e4cc71f150b7"
)
T4_TASKS_METADATA_SHA256 = (
    "399841cf8e861052d3fb7214ed619b5a10b4eded2d6f107e7fed338a233dd0e1"
)
T4_HELDOUT_SAMPLE_ASSET_SHA256 = {
    16: {
        "data/chunk-000/episode_000016.parquet": (
            "6e440ab8884337435fae9b0868a0b3911fe8dca6be00ba1061f343e666d130db"
        ),
        "videos/chunk-000/observation.images.image/episode_000016.mp4": (
            "cef539255b3e7322e6b4e37ef4a3fc5c9d9946685460c3e5f7bde772f1e1e544"
        ),
        "videos/chunk-000/observation.images.wrist_image/episode_000016.mp4": (
            "9f6cf8979f6ab33a87084443218e86d18f27d29118d8880a338d4f4b8d60d6a1"
        ),
    },
    405: {
        "data/chunk-000/episode_000405.parquet": (
            "0887ba7d42fdfbe26bc2402055ab75073a3bc860408f3306dabc896f9d2a1611"
        ),
        "videos/chunk-000/observation.images.image/episode_000405.mp4": (
            "46be6fe35ecf534cb64a259ff33b1bc19756180628c6859cd1e9272d1e92daf6"
        ),
        "videos/chunk-000/observation.images.wrist_image/episode_000405.mp4": (
            "e6be28d3d32d836530643d92919340afcffd02747bcfb743511898cee7321f3e"
        ),
    },
    40: {
        "data/chunk-000/episode_000040.parquet": (
            "4b2f24a8818e18c489b041e4292553c0f76ca6f121de18bd695b0dd440806617"
        ),
        "videos/chunk-000/observation.images.image/episode_000040.mp4": (
            "9d4d5a9f2082020193d7576f2174314fd831169b8086de59d125b11f2ff70650"
        ),
        "videos/chunk-000/observation.images.wrist_image/episode_000040.mp4": (
            "b02619c14008af3dae2c39a8762d197cd19d3599d7a80d51553c5550713c7b9d"
        ),
    },
}
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
SPATIAL_STATS_GR00T_SHA256 = (
    "0a4b08f5afcdcbe186ec70ea6ff2569233bab706a4d99ed603b23a7688bc33bb"
)
T4_RUN_NAMESPACE = Path("/DATA/share/sana_wam_libero_nonformal_screens/t4")
_ACTIVE_RUN_ROOT: Path | None = None


def _eligible_sample_manifest() -> dict[str, Any]:
    manifest = {
        "dataset": SPATIAL_NAME,
        "eligible_episode_count": len(T4_ELIGIBLE_EPISODES),
        "episodes": [
            {"episode_index": episode_index, "length": length}
            for episode_index, length in T4_ELIGIBLE_EPISODES
        ],
        "episodes_jsonl_sha256": T4_EPISODES_METADATA_SHA256,
        "excluded_training_episode_index": 0,
        "schema_version": "sana-wam-libero-t4-same-task-eligible-v1",
        "start_frame": 0,
        "task": T4_TASK_NAME,
        "task_index": T4_TASK_INDEX,
        "tasks_jsonl_sha256": T4_TASKS_METADATA_SHA256,
    }
    if _sha256_json(manifest) != T4_ELIGIBLE_MANIFEST_SHA256:
        raise RuntimeError("T4 eligible-sample manifest SHA differs")
    return manifest


def _selection_payload_sha256(episode_index: int) -> str:
    payload = (
        "SANA-WAM/LIBERO/T4_HELDOUT_SAMPLE_V1\n"
        f"T3_RESULT_SHA256={T3_PREDECESSOR_RESULT_SHA256}\n"
        f"DATASET={SPATIAL_NAME}\n"
        f"TASK_INDEX={T4_TASK_INDEX}\n"
        f"EPISODE_INDEX={episode_index}\n"
        "START_FRAME=0\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _heldout_sample_manifest() -> list[dict[str, Any]]:
    ranked = sorted(
        (
            _selection_payload_sha256(episode_index),
            episode_index,
            length,
        )
        for episode_index, length in T4_ELIGIBLE_EPISODES
    )
    rows = [
        {
            "episode_index": episode_index,
            "episode_length": length,
            "label": f"H{rank}",
            "selection_payload_sha256": payload_sha256,
            "start_frame": 0,
        }
        for rank, (payload_sha256, episode_index, length) in enumerate(
            ranked[:3], start=1
        )
    ]
    if tuple(row["episode_index"] for row in rows) != T4_HELDOUT_EPISODE_INDICES:
        raise RuntimeError("T4 mechanically selected episode set differs")
    for row in rows:
        episode_index = row["episode_index"]
        if (
            row["episode_length"] != T4_HELDOUT_EPISODE_LENGTHS[episode_index]
            or row["selection_payload_sha256"]
            != T4_HELDOUT_SELECTION_SHA256[episode_index]
        ):
            raise RuntimeError("T4 held-out sample derivation differs")
    return rows


def _select_t4_samples(dataset: Any) -> list[dict[str, Any]]:
    start_zero: dict[int, list[tuple[int, Any]]] = {}
    observed_eligible: dict[int, int] = {}
    for dataset_index, (episode_position, start) in enumerate(dataset._windows):
        episode = dataset._episodes[episode_position]
        if (
            episode.dataset != SPATIAL_NAME
            or episode.task_index != T4_TASK_INDEX
            or episode.task != T4_TASK_NAME
            or start != 0
        ):
            continue
        if episode.episode_index != 0:
            observed_eligible[episode.episode_index] = episode.length
        start_zero.setdefault(episode.episode_index, []).append(
            (dataset_index, episode)
        )
    if tuple(sorted(observed_eligible.items())) != T4_ELIGIBLE_EPISODES:
        raise RuntimeError("T4 live same-task eligible population differs")
    invalid_start_zero_counts = {
        episode_index: len(start_zero.get(episode_index, []))
        for episode_index in (0, *(row[0] for row in T4_ELIGIBLE_EPISODES))
        if len(start_zero.get(episode_index, [])) != 1
    }
    if invalid_start_zero_counts:
        raise RuntimeError(
            "T4 same-task episode/start identities are not unique: "
            f"{invalid_start_zero_counts!r}"
        )
    selected = []
    for row in _heldout_sample_manifest():
        matches = start_zero.get(row["episode_index"], [])
        if len(matches) != 1:
            raise RuntimeError(
                "T4 held-out episode/start must occur exactly once: "
                f"episode={row['episode_index']} matches={len(matches)}"
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
    if len(pre_update_action_losses) != T4_UPDATE_STEPS:
        raise ValueError(f"T4 requires exactly {T4_UPDATE_STEPS} pre-update losses")
    values = [initial_action_loss, *pre_update_action_losses, post_update_action_loss]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("T4 losses must be finite and non-negative")
    if initial_action_loss <= 0.0:
        raise ValueError("T4 initial action loss must be strictly positive")
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
        "threshold_ratio": T4_LEARNABILITY_RATIO,
        "loss_gate": (
            post_update_action_loss <= T4_LEARNABILITY_RATIO * initial_action_loss
            and last_five_median <= T4_LEARNABILITY_RATIO * first_five_median
        ),
    }


def _heldout_sample_transfer_summary(
    before_action_losses: dict[int, float],
    after_action_losses: dict[int, float],
) -> dict[str, Any]:
    expected_episodes = set(T4_HELDOUT_EPISODE_INDICES)
    if (
        set(before_action_losses) != expected_episodes
        or set(after_action_losses) != expected_episodes
    ):
        raise ValueError("T4 held-out loss episodes differ from the frozen set")

    rows = []
    for episode_index in T4_HELDOUT_EPISODE_INDICES:
        before = before_action_losses[episode_index]
        after = after_action_losses[episode_index]
        if (
            not math.isfinite(before)
            or not math.isfinite(after)
            or before <= 0.0
            or after < 0.0
        ):
            raise ValueError("T4 held-out losses must be finite with positive pre-loss")
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
        "improved_count": improved_count,
        "median_post_to_pre_ratio": median_ratio,
        "minimum_improved_count": T4_HELDOUT_MIN_IMPROVED,
        "per_episode": rows,
        "sorted_post_to_pre_ratios": sorted(row["post_to_pre_ratio"] for row in rows),
        "threshold_median_ratio": T4_HELDOUT_MEDIAN_RATIO,
        "transfer_gate": (
            median_ratio <= T4_HELDOUT_MEDIAN_RATIO
            and improved_count >= T4_HELDOUT_MIN_IMPROVED
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
    if len(per_step) != T4_UPDATE_STEPS:
        raise ValueError("T4/T3 training-core view requires exactly 20 step rows")
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
                "schema_version": "sana-wam-libero-t4-fixed-sample-failure-v1",
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
    parser.add_argument("--steps", type=int, default=T4_UPDATE_STEPS)
    parser.add_argument(
        "--initialization-seed", type=int, default=T4_INITIALIZATION_SEED
    )
    parser.add_argument("--loss-recipe-seed", type=int, default=T4_LOSS_RECIPE_SEED)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

    if args.steps != T4_UPDATE_STEPS:
        raise ValueError(f"T4 steps must be exactly {T4_UPDATE_STEPS}")
    if args.initialization_seed != T4_INITIALIZATION_SEED:
        raise ValueError("T4 initialization seed differs")
    if args.loss_recipe_seed != T4_LOSS_RECIPE_SEED:
        raise ValueError("T4 loss recipe seed differs")
    if len(args.nonce) != 32 or any(
        character not in "0123456789abcdef" for character in args.nonce
    ):
        raise ValueError("T4 nonce must be exactly 32 lowercase hex characters")
    expected_root_name = f"libero-t4-heldout3-fixed20-{args.nonce}"
    if args.run_root.name != expected_root_name:
        raise ValueError("T4 run-root basename does not bind the nonce")
    if args.run_root.parent.name != args.expected_repo_commit[:12]:
        raise ValueError("T4 run-root parent does not bind the source commit")
    expected_run_root = (
        T4_RUN_NAMESPACE / args.expected_repo_commit[:12] / expected_root_name
    )
    if args.run_root.expanduser() != expected_run_root:
        raise ValueError(
            "T4 run root differs from the frozen non-formal namespace: "
            f"expected={expected_run_root} got={args.run_root.expanduser()}"
        )
    eligible_sample_manifest = _eligible_sample_manifest()
    heldout_sample_manifest = _heldout_sample_manifest()
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
            "T4 source identity differs: "
            f"expected={expected_identity!r} actual={actual_identity!r}"
        )
    actual_identity["sana_commit"] = _assert_source_tree_clean(actual_commit)

    cfg = OmegaConf.load(config_path)
    if cfg.model.architecture.variant != "autoregressive":
        raise RuntimeError("T4 requires the AR architecture")
    if cfg.model.video_backbone.continuous_timestep_conditioning is not True:
        raise RuntimeError("T4 requires continuous FP32 video timesteps")
    if cfg.training.use_gradient_checkpointing is not True:
        raise RuntimeError("T4 requires gradient checkpointing")
    if int(cfg.training.seed) != args.initialization_seed:
        raise RuntimeError("T4 CLI initialization seed must equal training.seed")
    if tuple(cfg.training.trainable_modules) != EXPECTED_TRAINABLE_ROOTS:
        raise RuntimeError("T4 trainable module allowlist differs")
    if (
        tuple(cfg.training.preserve_frozen_input_grad_modules)
        != EXPECTED_PRESERVE_FROZEN_INPUT_GRAD_MODULES
    ):
        raise RuntimeError("T4 frozen input-gradient preservation differs")
    if cfg.training.optimizer_master_weights is not True:
        raise RuntimeError("T4 requires persistent FP32 optimizer masters")
    if (
        float(cfg.training.lambda_video) != 0.0
        or float(cfg.training.lambda_action) != 1.0
    ):
        raise RuntimeError("T4 requires action-only loss weighting")
    if float(cfg.training.weight_decay) != 0.0:
        raise RuntimeError("T4 requires weight_decay=0")
    if (
        float(cfg.training.action_lr) != 1.0e-4
        or float(cfg.training.video_lr) != 1.0e-4
    ):
        raise RuntimeError("T4 requires fixed base learning rates of 1e-4")
    if float(cfg.training.grad_clip) != 1.0:
        raise RuntimeError("T4 requires the fixed gradient clip bound 1.0")
    forbidden_checkpoint_fields = (
        "training.init_checkpoint",
        "training.resume_checkpoint",
        "training.resume_from_checkpoint",
        "training.resume_manifest",
        "model.video_backbone.init_dit_from",
    )
    for path in forbidden_checkpoint_fields:
        if OmegaConf.select(cfg, path, default=None) is not None:
            raise RuntimeError(f"T4 forbids checkpoint field {path}")
    if str(cfg.training.action_stats_sha256) != SELECTED_STATS_SHA256:
        raise RuntimeError("T4 selected-row stats SHA pin differs")
    if str(cfg.training.action_stats_population_sha256) != SELECTED_POPULATION_SHA256:
        raise RuntimeError("T4 selected population SHA pin differs")
    stats_path = Path(str(cfg.dataloader.action_stats_path)).expanduser()
    if stats_path.is_symlink() or not stats_path.is_file():
        raise RuntimeError("T4 stats artifact must be a regular non-symlink file")
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T4 selected-row stats artifact SHA differs")

    from sana_wam.train.libero_contract import validate_libero_training_config

    # This is the only full numeric live-source preflight. It runs before any
    # model construction or CUDA allocation.
    validate_libero_training_config(cfg, require_materialized_stats=True)
    external_assets = _verify_external_assets(cfg)
    spatial_root = Path(str(cfg.dataloader.dataset_roots[0]))
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
    for episode_index, assets in T4_HELDOUT_SAMPLE_ASSET_SHA256.items():
        for relative_path, expected_sha256 in sorted(assets.items()):
            external_assets[
                f"heldout_sample/episode_{episode_index:06d}/{relative_path}"
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
        raise RuntimeError("T4 predecessor T2 evidence semantics differ")
    t3_predecessor = json.loads(T3_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t3_predecessor.get("valid_run") is not True
        or t3_predecessor.get("scientific_verdict") != "T3_HELDOUT_RECIPE_TRANSFER_GO"
        or t3_predecessor.get("identity", {}).get("repo_commit")
        != T3_PREDECESSOR_SOURCE_COMMIT
        or t3_predecessor.get("identity", {}).get("runner_sha256")
        != T3_PREDECESSOR_RUNNER_SHA256
        or t3_predecessor.get("scope", {}).get("architecture_forwards_total") != 28
        or t3_predecessor.get("scope", {}).get("backward_calls") != T4_UPDATE_STEPS
        or t3_predecessor.get("scope", {}).get("optimizer_steps") != T4_UPDATE_STEPS
    ):
        raise RuntimeError("T4 predecessor T3 evidence semantics differ")
    t3_training_core = _training_core_projection(t3_predecessor)
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
        raise RuntimeError(f"T4 selected GPU is not an H200: {gpu_before['name']}")

    import torch

    from sana_wam.train.trainer import Trainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "T4 requires exactly one visible CUDA GPU; "
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
        raise RuntimeError("T4 video-backbone parameters are not frozen")
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
        raise RuntimeError(f"T4 trainable parameter roots differ: {observed_roots!r}")
    if len(named_trainables) != EXPECTED_TRAINABLE_TENSOR_COUNT:
        raise RuntimeError("T4 trainable tensor count differs")
    if sum(parameter.numel() for _name, parameter in named_trainables) != (
        EXPECTED_TRAINABLE_PARAMETER_COUNT
    ):
        raise RuntimeError("T4 trainable parameter count differs")
    frozen_versions = {name: parameter._version for name, parameter in named_frozen}

    dataset_index, episode = _find_fixed_sample(trainer.dataset)
    heldout_selections = _select_t4_samples(trainer.dataset)
    if (
        episode.dataset != SPATIAL_NAME
        or _sha256_file(episode.data_path()) != PARQUET_SHA256
    ):
        raise RuntimeError("T4 fixed real LIBERO sample identity differs")
    if (
        episode.episode_index != 0
        or episode.length != T4_TRAIN_EPISODE_LENGTH
        or episode.task_index != T4_TASK_INDEX
        or episode.task != T4_TASK_NAME
    ):
        raise RuntimeError("T4 fixed LIBERO episode metadata differs")
    sample_started = time.perf_counter()
    training_sample = trainer.dataset[dataset_index]
    sample_seconds = time.perf_counter() - sample_started
    if (
        training_sample["action_alignment"] != "observation_t_to_action_t"
        or training_sample["episode_index"] != 0
        or training_sample["episode_length"] != T4_TRAIN_EPISODE_LENGTH
        or training_sample["start_frame"] != 0
        or training_sample["task_index"] != T4_TASK_INDEX
        or training_sample["task_name"] != T4_TASK_NAME
    ):
        raise RuntimeError("T4 fixed LIBERO sample metadata differs")
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
            raise RuntimeError(f"T4 Spatial asset changed while loading: {key}")
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T4 selected-row stats changed while loading")
    actual_identity["spatial_assets_post_sample_sha256"] = _sha256_json(
        spatial_assets_after_sample
    )

    trainer._set_training_mode()
    if any(module.training for module in video_backbone.modules()):
        raise RuntimeError("T4 preserved video backbone left eval mode")
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
            raise RuntimeError(f"T4 {label} inputs dropped gradient checkpointing")
        if prepared.get("use_gradient_checkpointing_offload") is not False:
            raise RuntimeError("T4 requires checkpoint offload=false")
        invalid_tensors = [
            key
            for key, value in prepared.items()
            if isinstance(value, torch.Tensor)
            and (value.requires_grad or value.grad_fn is not None)
        ]
        if invalid_tensors:
            raise RuntimeError(
                f"T4 {label} prepared tensors retain autograd state: "
                f"{invalid_tensors!r}"
            )
        return prepared

    # Preserve the T2/T3 training-sample preparation path before introducing
    # any held-out sample decode or encoding into the forward schedule.
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
            or selected_episode.task_index != T4_TASK_INDEX
            or selected_episode.task != T4_TASK_NAME
            or heldout_sample["action_alignment"] != "observation_t_to_action_t"
            or heldout_sample["episode_index"] != episode_index
            or heldout_sample["episode_length"] != selected["episode_length"]
            or heldout_sample["start_frame"] != 0
            or heldout_sample["task_index"] != T4_TASK_INDEX
            or heldout_sample["task_name"] != T4_TASK_NAME
        ):
            raise RuntimeError(
                f"T4 held-out LIBERO sample metadata differs: {episode_index}"
            )
        heldout_inputs[episode_index] = prepare_one(
            f"heldout_episode_{episode_index}", heldout_sample
        )
        del heldout_sample
        for relative_path, expected_sha256 in sorted(
            T4_HELDOUT_SAMPLE_ASSET_SHA256[episode_index].items()
        ):
            key = f"heldout_sample/episode_{episode_index:06d}/{relative_path}"
            observed = _verify_pinned_asset(
                spatial_root / relative_path,
                expected_sha256,
            )
            if external_assets[key] != observed:
                raise RuntimeError(f"T4 held-out asset changed while loading: {key}")
            heldout_assets_after_sample[key] = observed
    actual_identity["heldout_assets_post_sample_sha256"] = _sha256_json(
        heldout_assets_after_sample
    )
    if prepare_inputs_calls != 4:
        raise RuntimeError(
            f"T4 prepare_inputs call count differs: {prepare_inputs_calls}"
        )

    prepared_inputs = {"training": training_inputs}
    prepared_inputs.update(
        {
            f"heldout_episode_{episode_index}": heldout_inputs[episode_index]
            for episode_index in T4_HELDOUT_EPISODE_INDICES
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
                    "T4 prepared tensors alias across samples: "
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

    model_groups = trainer._param_groups()
    if len(model_groups) != 2 or any(
        float(group["lr"]) != 1.0e-4 for group in model_groups
    ):
        raise RuntimeError("T4 optimizer group/LR structure differs")
    params = [parameter for group in model_groups for parameter in group["params"]]
    parameter_ids = [id(parameter) for parameter in params]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != {
        id(parameter) for _name, parameter in named_trainables
    }:
        raise RuntimeError("T4 optimizer parameters differ from trainables")
    optimizer_groups, master_pairs = trainer._optimizer_param_groups(model_groups)
    if len(master_pairs) != len(params):
        raise RuntimeError("T4 FP32 master coverage differs")
    if [id(model) for model, _master in master_pairs] != parameter_ids:
        raise RuntimeError("T4 FP32 master/model ordering differs")
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
        raise RuntimeError("T4 FP32 master identity/dtype differs")
    if [
        id(parameter) for group in optimizer_groups for parameter in group["params"]
    ] != initial_master_ids:
        raise RuntimeError("T4 optimizer groups do not contain the FP32 masters")
    if [len(group["params"]) for group in optimizer_groups] != [
        len(group["params"]) for group in model_groups
    ] or [float(group["lr"]) for group in optimizer_groups] != [
        float(group["lr"]) for group in model_groups
    ]:
        raise RuntimeError("T4 FP32 master optimizer group structure differs")
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
            raise RuntimeError("T4 update-free probe observed model gradients")
        if any(parameter.grad is not None for _name, parameter in named_frozen):
            raise RuntimeError("T4 update-free probe observed frozen gradients")
        if any(master.grad is not None for _name, master in named_masters):
            raise RuntimeError("T4 update-free probe observed master gradients")
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
                raise RuntimeError("T4 produced a non-finite scalar loss")
            scalars = {
                "loss": _tensor_scalar(loss),
                "loss_action": _tensor_scalar(loss_action),
            }
            tolerance = 1.0e-6 + 1.0e-6 * abs(scalars["loss_action"])
            if abs(scalars["loss"] - scalars["loss_action"]) > tolerance:
                raise RuntimeError("T4 total/action loss identity differs")
            signature = _sha256_json(recipe_capture)
            if not recipe_capture["action"] or not recipe_capture["video"]:
                raise RuntimeError("T4 stochastic recipe capture is incomplete")
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
            raise RuntimeError("T4 fixed recipe did not restore caller RNG state")
        del loss, loss_action, result
        return scalars, signature

    per_step: list[dict[str, Any]] = []
    training_recipe_signatures: list[str] = []
    heldout_before: dict[int, dict[str, float]] = {}
    heldout_after: dict[int, dict[str, float]] = {}
    heldout_sample_signatures: dict[int, list[str]] = {
        episode_index: [] for episode_index in T4_HELDOUT_EPISODE_INDICES
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
            raise RuntimeError("T4 optimizer state exists before held-out pre probes")
        heldout_pre_state = probe_mutation_snapshot()
        for episode_index in T4_HELDOUT_EPISODE_INDICES:
            scalars, signature = fixed_forward(
                backward=False,
                forward_inputs=heldout_inputs[episode_index],
                recipe_seed=args.loss_recipe_seed,
            )
            heldout_before[episode_index] = scalars
            heldout_sample_signatures[episode_index].append(signature)
        if probe_mutation_snapshot() != heldout_pre_state or optimizer.state:
            raise RuntimeError("T4 held-out pre probes mutated training state")

        initial_probe, initial_signature = fixed_forward(
            backward=False,
            forward_inputs=training_inputs,
            recipe_seed=args.loss_recipe_seed,
        )
        training_recipe_signatures.append(initial_signature)

        for step_index in range(T4_UPDATE_STEPS):
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
                        "T4 fixed initial loss recipe is not reproducible"
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
                        f"T4 non-finite gradient at step {step_index + 1}: {name}"
                    )
                maximum = float(gradient.detach().abs().max().float().item())
                max_gradient_abs = max(max_gradient_abs, maximum)
                if maximum > 0.0:
                    nonzero_gradient_tensors += 1
                    nonzero_gradient_roots.add(_parameter_root(name))
            if missing_gradients:
                raise RuntimeError(
                    f"T4 missing gradients at step {step_index + 1}: "
                    f"{missing_gradients[:8]!r}"
                )
            if nonzero_gradient_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    f"T4 nonzero-gradient roots differ at step {step_index + 1}: "
                    f"{sorted(nonzero_gradient_roots)!r}"
                )
            if any(parameter.grad is not None for _name, parameter in named_frozen):
                raise RuntimeError("T4 frozen parameters received gradients")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, float(cfg.training.grad_clip)
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError("T4 gradient norm is non-finite")
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
                        "T4 first-step masters lack finite nonzero FP32 gradients: "
                        f"{invalid_master_gradients[:8]!r}"
                    )
                master_probes = _capture_update_probes(named_masters)
                model_probes = _capture_update_probes(named_trainables)
                if {row["name"] for row in master_probes} != {
                    name for name, _master in named_masters
                }:
                    raise RuntimeError("T4 first-step master probe coverage differs")

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
                        raise RuntimeError(f"T4 lacks four update sentinels for {root}")
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
                    raise RuntimeError("T4 first step did not update every FP32 master")
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
                    raise RuntimeError("T4 update sentinel became non-finite")
                sentinel_updates.append(row)
            changed_master_sentinels = [
                row for row in sentinel_updates if row["master_delta"] != 0.0
            ]
            changed_master_sentinel_roots = {
                row["root"] for row in changed_master_sentinels
            }
            if changed_master_sentinel_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    "T4 sampled FP32 master update roots differ at step "
                    f"{step_index + 1}: "
                    f"{sorted(changed_master_sentinel_roots)!r}"
                )
            sentinel_adam_steps = {
                _tensor_scalar(optimizer.state[probe["parameter"]]["step"])
                for probe in update_sentinels
            }
            if sentinel_adam_steps != {float(step_index + 1)}:
                raise RuntimeError(
                    "T4 sampled AdamW step counters differ at step "
                    f"{step_index + 1}: {sorted(sentinel_adam_steps)!r}"
                )
            if step_index in {0, T4_UPDATE_STEPS - 1}:
                projection_mismatches = [
                    trainable_name_by_id[id(model)]
                    for model, master in master_pairs
                    if not torch.equal(
                        model.detach(), master.detach().to(dtype=model.dtype)
                    )
                ]
                if projection_mismatches:
                    raise RuntimeError(
                        f"T4 BF16 projection differs: {projection_mismatches[:8]!r}"
                    )
                if not _optimizer_state_is_finite(optimizer):
                    raise RuntimeError("T4 AdamW state is non-finite")
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
        for episode_index in T4_HELDOUT_EPISODE_INDICES:
            scalars, signature = fixed_forward(
                backward=False,
                forward_inputs=heldout_inputs[episode_index],
                recipe_seed=args.loss_recipe_seed,
            )
            heldout_after[episode_index] = scalars
            heldout_sample_signatures[episode_index].append(signature)
        if probe_mutation_snapshot() != heldout_post_state:
            raise RuntimeError("T4 held-out post probes mutated training state")
    finally:
        video_embedder.forward = original_video_embedder_forward
        architecture.action_backbone.prepare_state = original_action_prepare_state
        architecture.forward = original_architecture_forward

    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - update_started
    if architecture_forward_calls != T4_UPDATE_STEPS + 8:
        raise RuntimeError(
            f"T4 architecture forward count differs: {architecture_forward_calls}"
        )
    if backward_calls != T4_UPDATE_STEPS:
        raise RuntimeError(f"T4 backward call count differs: {backward_calls}")
    if optimizer_step_calls != T4_UPDATE_STEPS:
        raise RuntimeError(
            f"T4 optimizer step call count differs: {optimizer_step_calls}"
        )
    if (
        len(training_recipe_signatures) != T4_UPDATE_STEPS + 2
        or len(set(training_recipe_signatures)) != 1
    ):
        raise RuntimeError("T4 training RNG recipe signatures differ across forwards")
    invalid_heldout_signatures = {
        episode_index: signatures
        for episode_index, signatures in heldout_sample_signatures.items()
        if len(signatures) != 2 or len(set(signatures)) != 1
    }
    if invalid_heldout_signatures:
        raise RuntimeError(
            f"T4 held-out RNG recipe signatures differ: {invalid_heldout_signatures!r}"
        )
    final_optimizer_master_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if final_optimizer_master_ids != initial_master_ids:
        raise RuntimeError("T4 FP32 master objects changed during the loop")
    if prepared_input_snapshot() != initial_prepared_input_snapshot:
        raise RuntimeError("T4 prepared input identities or versions changed")
    frozen_version_changes = [
        name
        for name, parameter in named_frozen
        if parameter._version != frozen_versions[name]
    ]
    if frozen_version_changes:
        raise RuntimeError(
            f"T4 frozen parameters changed: {frozen_version_changes[:8]!r}"
        )
    if any(not bool(torch.isfinite(master).all().item()) for master in masters):
        raise RuntimeError("T4 FP32 masters became non-finite")
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in params):
        raise RuntimeError("T4 BF16 trainables became non-finite")
    if not _optimizer_state_is_finite(optimizer):
        raise RuntimeError("T4 final AdamW state is non-finite")
    adam_steps: list[float] = []
    for master in masters:
        state = optimizer.state.get(master)
        if not state or "step" not in state:
            raise RuntimeError("T4 AdamW state is missing for an FP32 master")
        step_value = _tensor_scalar(state["step"])
        if not math.isfinite(step_value):
            raise RuntimeError("T4 AdamW step counter is non-finite")
        adam_steps.append(step_value)
    if set(adam_steps) != {float(T4_UPDATE_STEPS)}:
        raise RuntimeError(
            f"T4 AdamW step counters differ: {sorted(set(adam_steps))!r}"
        )

    cumulative_master_update, cumulative_master_ids = _summarize_updates(master_probes)
    cumulative_projected_update, _cumulative_model_ids = _summarize_updates(
        model_probes
    )
    if len(cumulative_master_ids) != len(named_masters) or set(
        cumulative_master_update["changed_trainable_roots"]
    ) != set(EXPECTED_TRAINABLE_ROOTS):
        raise RuntimeError("T4 cumulative FP32 master update coverage differs")

    learnability = _learnability_summary(
        initial_probe["loss_action"],
        [row["loss_action"] for row in per_step],
        final_probe["loss_action"],
    )
    heldout_transfer = _heldout_sample_transfer_summary(
        {
            episode_index: heldout_before[episode_index]["loss_action"]
            for episode_index in T4_HELDOUT_EPISODE_INDICES
        },
        {
            episode_index: heldout_after[episode_index]["loss_action"]
            for episode_index in T4_HELDOUT_EPISODE_INDICES
        },
    )
    heldout_transfer_report = {
        **heldout_transfer,
        "per_episode": [
            {
                **row,
                "after_total_loss": heldout_after[row["episode_index"]]["loss"],
                "assets": T4_HELDOUT_SAMPLE_ASSET_SHA256[row["episode_index"]],
                "before_total_loss": heldout_before[row["episode_index"]]["loss"],
                "episode_length": T4_HELDOUT_EPISODE_LENGTHS[row["episode_index"]],
                "label": heldout_sample_manifest[
                    T4_HELDOUT_EPISODE_INDICES.index(row["episode_index"])
                ]["label"],
                "post_recipe_signature_sha256": heldout_sample_signatures[
                    row["episode_index"]
                ][1],
                "pre_recipe_signature_sha256": heldout_sample_signatures[
                    row["episode_index"]
                ][0],
                "selection_payload_sha256": T4_HELDOUT_SELECTION_SHA256[
                    row["episode_index"]
                ],
                "start_frame": 0,
            }
            for row in heldout_transfer["per_episode"]
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
            "this is not part of the frozen T4 primary transfer gate"
        )
    if projected_roots != set(EXPECTED_TRAINABLE_ROOTS):
        secondary_diagnostic_warnings.append(
            "20-step BF16 projection did not visibly change every trainable root; "
            "this is not part of the frozen T4 primary transfer gate"
        )
    verdict_reasons = []
    if not heldout_transfer["transfer_gate"]:
        verdict_reasons.append(
            "held-out sample transfer did not clear the median/2-of-3 gate"
        )
    verdict = (
        "T4_HELDOUT_SAMPLE_TRANSFER_GO"
        if not verdict_reasons
        else "T4_HELDOUT_SAMPLE_TRANSFER_INCONCLUSIVE"
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
        "heldout_files_post_sample": heldout_assets_after_sample,
        "heldout_files_post_sample_manifest_sha256": actual_identity[
            "heldout_assets_post_sample_sha256"
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
        str(episode_index): prepared_shape_report(heldout_inputs[episode_index])
        for episode_index in T4_HELDOUT_EPISODE_INDICES
    }
    learnability_report = {
        **learnability,
        "final_probe_total_loss": final_probe["loss"],
        "initial_probe_action_loss": initial_probe["loss_action"],
        "initial_probe_total_loss": initial_probe["loss"],
        "pre_update_action_loss_curve": [row["loss_action"] for row in per_step],
        "pre_update_total_loss_curve": [row["loss"] for row in per_step],
        "role": "secondary_diagnostic_not_t4_primary_gate",
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
        "projection_exact_after_steps": [1, T4_UPDATE_STEPS],
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
    all_recipe_signatures = training_recipe_signatures + [
        signature
        for episode_index in T4_HELDOUT_EPISODE_INDICES
        for signature in heldout_sample_signatures[episode_index]
    ]
    randomness_report = {
        "eligible_sample_manifest": eligible_sample_manifest,
        "eligible_sample_manifest_sha256": T4_ELIGIBLE_MANIFEST_SHA256,
        "heldout_sample_manifest": heldout_sample_manifest,
        "heldout_signatures_per_episode": {
            str(episode_index): heldout_sample_signatures[episode_index]
            for episode_index in T4_HELDOUT_EPISODE_INDICES
        },
        "initialization_seed": args.initialization_seed,
        "training_loss_recipe_seed": args.loss_recipe_seed,
        "recipe_reset": "python+numpy+torch-cpu+torch-cuda before every forward",
        "rng_state_restored_after_every_forward": True,
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
        "architecture_forwards_total": T4_UPDATE_STEPS + 8,
        "architecture_measurement_forwards": 8,
        "architecture_training_forwards": T4_UPDATE_STEPS,
        "backward_calls": backward_calls,
        "benchmark_evaluation_executed": False,
        "formal_training_executed": False,
        "heldout_measurement_forwards": 6,
        "heldout_samples_in_backward_or_update": 0,
        "main_recipe_measurement_forwards": 2,
        "optimizer_steps": optimizer_step_calls,
        "post_probe_reprepare_calls": 0,
        "prepare_inputs_calls": prepare_inputs_calls,
        "probe_order": [
            "heldout_pre_h1",
            "heldout_pre_h2",
            "heldout_pre_h3",
            "training_recipe_pre",
            "training_recipe_update_1_through_20",
            "training_recipe_post",
            "heldout_post_h1",
            "heldout_post_h2",
            "heldout_post_h3",
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
    expected_training_core_sha256 = _sha256_json(t3_training_core)
    observed_training_core_sha256 = _sha256_json(current_training_core)
    if current_training_core != t3_training_core:
        raise RuntimeError(
            "T4 training core drifted from frozen T3: "
            f"expected_sha256={expected_training_core_sha256} "
            f"observed_sha256={observed_training_core_sha256}"
        )
    training_core_reproduction = {
        "equal_to_frozen_t3": True,
        "expected_projection_sha256": expected_training_core_sha256,
        "observed_projection_sha256": observed_training_core_sha256,
        "predecessor_result_sha256": T3_PREDECESSOR_RESULT_SHA256,
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
        "heldout_inputs": heldout_inputs_report,
        "heldout_sample_transfer": heldout_transfer_report,
        "identity": identity_report,
        "inputs": training_inputs_report,
        "interpretation_limits": {
            "benchmark_success_claimed": False,
            "cross_suite_or_task_transfer_claimed": False,
            "heldout_axis": "optimizer-update exposure only",
            "normalization_population_includes_probe_episodes": True,
            "strict_dataset_holdout": False,
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
        "schema_version": "sana-wam-libero-t4-heldout-sample-transfer-v1",
        "scope": scope_report,
        "timings_seconds": {
            "model_and_dataset_build": build_seconds,
            "prepare_real_samples": {
                label: measurement["seconds"]
                for label, measurement in prepare_measurements.items()
            },
            "sample_load": sample_seconds,
            "heldout_sample_load": {
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
        raise RuntimeError("T4 run root was not empty before terminal write")
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
                    "failed to terminalize T4 root: "
                    f"{type(freeze_error).__name__}: {freeze_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    raise SystemExit(exit_code)
