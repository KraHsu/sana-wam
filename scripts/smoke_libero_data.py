#!/usr/bin/env python
"""Run the fixed CPU-only LIBERO real-data alignment smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

from sana_wam.dataloader.libero_dataset import LiberoLeRobotDataset  # noqa: E402
from sana_wam.dataloader.libero_stats import sha256_file  # noqa: E402


SPATIAL_NAME = "libero_spatial_no_noops_1.0.0_lerobot"
PARQUET_SHA256 = "3f875604fad478765549128759edfb33a64b69b7b82decebc9e4f38155f20a8c"
PRIMARY_RGB_SHA256 = "0b03b94b96208846f7fdab3a5dcacd0ef8e470d3823f86101d0b7bbb84424f16"
WRIST_RGB_SHA256 = "10b9118f9a99d19fcfba4c80a53101975b136fe8453b001fa5f32f2a762429df"
EXPECTED_STATE0 = np.asarray(
    [
        -0.2048747689,
        -0.0100815259,
        1.1746579409,
        3.1396350861,
        0.0001441165,
        -0.0880179554,
        0.0387861729,
        -0.0387897342,
    ],
    dtype=np.float32,
)
EXPECTED_ACTION0 = np.asarray(
    [
        0.1312499940,
        -0.0401785709,
        0.0,
        0.0,
        -0.0492857136,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)


def _rgb_sha256(image) -> str:
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _repo_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CPU smoke requires CUDA_VISIBLE_DEVICES to be empty or -1")
    stats_path = Path(args.stats).expanduser()
    report_candidate = Path(args.report).expanduser()
    if stats_path.is_symlink() or report_candidate.is_symlink():
        raise ValueError("LIBERO smoke paths must not be symlinks")
    stats_path = stats_path.resolve()
    report_path = report_candidate.resolve()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite smoke report: {report_path}")

    cfg = OmegaConf.load(
        ROOT / "configs/benchmarks/libero/train_libero_ar_baseline.yaml"
    )
    cfg.dataloader.action_stats_path = str(stats_path)
    dataset = LiberoLeRobotDataset.from_config(cfg.dataloader, split="train")

    selected = None
    for dataset_index, (episode_position, start) in enumerate(dataset._windows):
        episode = dataset._episodes[episode_position]
        if (
            episode.dataset == SPATIAL_NAME
            and episode.episode_index == 0
            and start == 0
        ):
            selected = (dataset_index, episode)
            break
    if selected is None:
        raise RuntimeError("fixed LIBERO Spatial episode 0 window was not found")
    dataset_index, episode = selected

    parquet_sha256 = sha256_file(episode.data_path())
    if parquet_sha256 != PARQUET_SHA256:
        raise RuntimeError(f"Spatial episode 0 Parquet SHA differs: {parquet_sha256}")
    states, actions, task_indices = dataset._read_episode_arrays(episode)
    np.testing.assert_allclose(states[0], EXPECTED_STATE0, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(actions[0], EXPECTED_ACTION0, rtol=0.0, atol=1e-7)
    if not np.all(task_indices == 0):
        raise RuntimeError("Spatial episode 0 task_index differs")

    primary_frame = dataset._read_video_indices(
        episode.video_path("observation.images.image"), [0]
    )[0]
    wrist_frame = dataset._read_video_indices(
        episode.video_path("observation.images.wrist_image"), [0]
    )[0]
    primary_sha256 = _rgb_sha256(primary_frame)
    wrist_sha256 = _rgb_sha256(wrist_frame)
    if primary_sha256 != PRIMARY_RGB_SHA256:
        raise RuntimeError(f"primary RGB frame-0 SHA differs: {primary_sha256}")
    if wrist_sha256 != WRIST_RGB_SHA256:
        raise RuntimeError(f"wrist RGB frame-0 SHA differs: {wrist_sha256}")

    sample = dataset[dataset_index]
    if sample["action_alignment"] != "observation_t_to_action_t":
        raise RuntimeError("sample action alignment label differs")
    if tuple(sample["action"].shape) != (112, 7):
        raise RuntimeError(f"action shape differs: {tuple(sample['action'].shape)}")
    if tuple(sample["proprio_seq"].shape) != (113, 8):
        raise RuntimeError(
            f"proprio_seq shape differs: {tuple(sample['proprio_seq'].shape)}"
        )
    if len(sample["video"]) != 29:
        raise RuntimeError(f"video sample count differs: {len(sample['video'])}")
    video_shape = tuple(np.asarray(sample["video"][0]).shape)
    if video_shape != (384, 320, 3):
        raise RuntimeError(f"composite training frame shape differs: {video_shape}")
    if int(sample["action_mask"].sum()) != 109:
        raise RuntimeError("action padding mask count differs")
    if int(sample["video_mask"].sum()) != 28:
        raise RuntimeError("video padding mask count differs")

    expected_action_normalized = dataset._action_normalizer.normalize(actions[:1])[0]
    expected_state_normalized = dataset._state_normalizer.normalize(states[:1])[0]
    np.testing.assert_allclose(
        sample["action"][0].numpy(), expected_action_normalized, rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        sample["proprio"][0].numpy(), expected_state_normalized, rtol=0.0, atol=1e-6
    )

    report = {
        "action0": actions[0].tolist(),
        "action_mask_true": int(sample["action_mask"].sum()),
        "action_shape": list(sample["action"].shape),
        "alignment": sample["action_alignment"],
        "cpu_only": not torch.cuda.is_available(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_episode_count": len(dataset._episodes),
        "dataset_window_count": len(dataset._windows),
        "excluded_episode_count": len(dataset._observed_exclusions),
        "parquet_sha256": parquet_sha256,
        "primary_rgb_frame0_sha256": primary_sha256,
        "proprio_seq_shape": list(sample["proprio_seq"].shape),
        "pyarrow_version": pyarrow.__version__,
        "repo_commit": _repo_commit(),
        "schema_version": "sana-wam-libero-real-data-smoke-v1",
        "selected_dataset": episode.dataset,
        "selected_episode": episode.episode_index,
        "selected_episode_length": episode.length,
        "state0": states[0].tolist(),
        "stats_sha256": sha256_file(stats_path),
        "video_mask_true": int(sample["video_mask"].sum()),
        "video_shape": list(video_shape),
        "wrist_rgb_frame0_sha256": wrist_sha256,
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    report_path.chmod(0o444)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
