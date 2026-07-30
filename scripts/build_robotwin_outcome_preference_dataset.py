#!/usr/bin/env python
"""Build grouped generation-zero outcome preferences from verified rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

try:
    from scripts.verify_paired_noise_telemetry import load_client_run, load_run
except ModuleNotFoundError:  # Direct script execution.
    from verify_paired_noise_telemetry import load_client_run, load_run


_RESULT_PATTERN = re.compile(r"Data has been saved to (.+?)/_result\.txt")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_run_bundle(path: Path, expected_episodes: int, cache: dict) -> tuple:
    resolved = path.resolve()
    if resolved not in cache:
        server = load_run(
            resolved,
            expected_mode="predicted",
            expected_episodes=expected_episodes,
        )
        client = load_client_run(resolved, expected_episodes=expected_episodes)
        matches = _RESULT_PATTERN.findall(
            (resolved / "eval.log").read_text(errors="replace")
        )
        if len(matches) != 1:
            raise ValueError(f"cannot identify one eval-result directory in {resolved}")
        video_dir = resolved / "robotwin_runtime" / Path(matches[0])
        if not video_dir.is_dir():
            raise ValueError(f"video directory is missing: {video_dir}")
        cache[resolved] = server, client, video_dir
    return cache[resolved]


def _video_frame_zero(video_path: Path, expected_frames: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        ok, frame_bgr = capture.read()
    finally:
        capture.release()
    if not ok or frame_bgr is None:
        raise ValueError(f"cannot decode first frame: {video_path}")
    if reported_frames != expected_frames:
        raise ValueError(
            f"video frame count mismatch for {video_path}: "
            f"{reported_frames} != {expected_frames}"
        )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_dataset(plan_path: Path, verification_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    plan = json.loads(plan_path.read_text())
    verification = json.loads(verification_path.read_text())
    if plan.get("kind") != "outcome_groups_g5_plan":
        raise ValueError("invalid outcome-group plan kind")
    if verification.get("scenes") != 30 or verification.get("group_size") != 5:
        raise ValueError("verification report does not describe 30 groups of 5")
    selected = plan["selected_scenes"]
    replicas = plan["replicas"]
    expected_outcomes = verification["outcomes_by_scene"]
    output_dir.mkdir(parents=True)
    image_dir = output_dir / "images"
    array_dir = output_dir / "groups"
    image_dir.mkdir()
    array_dir.mkdir()
    cache = {}
    groups = []
    total_candidates = 0
    total_positive = 0
    total_video_bytes = 0
    pixel_difference_maxima = []

    replica_bundles = [
        _load_run_bundle(Path(replica["run_dir"]), 30, cache)
        for replica in replicas
    ]
    for group_index, row in enumerate(selected):
        seed = row["environment_seed"]
        source_path = Path(row["source_run"])
        source_server, source_client, source_video_dir = _load_run_bundle(
            source_path, 20, cache
        )
        source_matches = [
            index
            for index, pair in enumerate(source_server["pairs"])
            if pair[2] == seed
        ]
        if len(source_matches) != 1:
            raise ValueError(f"cannot identify source episode for seed {seed}")
        source_index = source_matches[0]
        bundles = [
            (
                "baseline",
                source_path,
                source_server,
                source_client,
                source_video_dir,
                source_index,
            )
        ]
        for replica_index, replica in enumerate(replicas):
            server, client, video_dir = replica_bundles[replica_index]
            bundles.append(
                (
                    f"replica_{replica_index}",
                    Path(replica["run_dir"]),
                    server,
                    client,
                    video_dir,
                    group_index,
                )
            )

        raw_actions = []
        normalized_actions = []
        proprios = []
        outcomes = []
        model_noise_seeds = []
        candidates = []
        decoded_frames = []
        for candidate_index, (
            role,
            run_path,
            server,
            client,
            video_dir,
            episode_index,
        ) in enumerate(bundles):
            pair = server["pairs"][episode_index]
            generation_zero = server["generation_zero"][episode_index]
            end = client["ends"][episode_index]
            if pair[2] != seed:
                raise ValueError(
                    f"candidate scene mismatch at group {group_index} candidate {candidate_index}"
                )
            success = bool(end["success"])
            expected_success = bool(expected_outcomes[str(seed)][candidate_index])
            if success is not expected_success:
                raise ValueError(
                    f"candidate outcome mismatch at group {group_index} candidate {candidate_index}"
                )
            raw = generation_zero["predicted_actions"]
            normalized = generation_zero["predicted_actions_normalized"]
            proprio = generation_zero["observed_proprio"]
            if raw.shape != (28, 20) or normalized.shape != (28, 20):
                raise ValueError(f"invalid action chunk shape for seed {seed}")
            if proprio.shape != (20,):
                raise ValueError(f"invalid generation-zero proprio for seed {seed}")
            video_path = video_dir / f"episode{episode_index}.mp4"
            expected_frames = int(end["step_count"]) + int(success)
            frame_rgb = _video_frame_zero(video_path, expected_frames)
            image_path = image_dir / f"group{group_index:03d}_candidate{candidate_index}.png"
            Image.fromarray(frame_rgb).save(image_path)
            video_size = video_path.stat().st_size
            total_video_bytes += video_size
            decoded_frames.append(frame_rgb)
            raw_actions.append(raw)
            normalized_actions.append(normalized)
            proprios.append(proprio)
            outcomes.append(success)
            model_noise_seeds.append(pair[4])
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "role": role,
                    "run_path": str(run_path),
                    "episode_index": episode_index,
                    "episode_key": server["starts"][episode_index]["episode_key"],
                    "model_noise_seed": pair[4],
                    "success": success,
                    "step_count": int(end["step_count"]),
                    "video_frame_count": expected_frames,
                    "source_video": {
                        "path": str(video_path),
                        "size_bytes": video_size,
                        "sha256": _sha256(video_path),
                    },
                    "initial_head_image": {
                        "path": str(image_path.relative_to(output_dir)),
                        "size_bytes": image_path.stat().st_size,
                        "sha256": _sha256(image_path),
                    },
                }
            )

        proprio_array = np.stack(proprios).astype(np.float32)
        if not all(np.array_equal(proprio_array[0], value) for value in proprio_array[1:]):
            raise ValueError(f"generation-zero proprio differs within seed {seed}")
        reference_frame = decoded_frames[0].astype(np.int16)
        pixel_difference_maxima.extend(
            int(np.abs(frame.astype(np.int16) - reference_frame).max())
            for frame in decoded_frames[1:]
        )
        group_path = array_dir / f"group{group_index:03d}.npz"
        np.savez_compressed(
            group_path,
            schema_version=np.asarray(1, dtype=np.int16),
            environment_seed=np.asarray(seed, dtype=np.int64),
            predicted_actions=np.stack(raw_actions).astype(np.float32),
            predicted_actions_normalized=np.stack(normalized_actions).astype(
                np.float32
            ),
            observed_proprio=proprio_array,
            success=np.asarray(outcomes, dtype=np.bool_),
            model_noise_seed=np.asarray(model_noise_seeds, dtype=np.int64),
        )
        positive_count = sum(outcomes)
        total_candidates += len(candidates)
        total_positive += positive_count
        groups.append(
            {
                "group_index": group_index,
                "environment_seed": seed,
                "task_name": row["task_name"],
                "task_config": row["task_config"],
                "prompt": row["prompt"],
                "prompt_sha256": row["prompt_sha256"],
                "prompt_group": row["prompt_group"],
                "positive_count": positive_count,
                "negative_count": 5 - positive_count,
                "mixed": 0 < positive_count < 5,
                "group_array": {
                    "path": str(group_path.relative_to(output_dir)),
                    "size_bytes": group_path.stat().st_size,
                    "sha256": _sha256(group_path),
                },
                "candidates": candidates,
            }
        )

    report = {
        "schema_version": 1,
        "kind": "robotwin_generation_zero_outcome_preference_dataset",
        "complete": True,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "verification_path": str(verification_path),
        "verification_sha256": _sha256(verification_path),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "observation_view": "head_camera_rgb_from_pre_action_eval_video_frame_0",
        "group_count": len(groups),
        "mixed_group_count": sum(group["mixed"] for group in groups),
        "candidate_count": total_candidates,
        "positive_candidates": total_positive,
        "negative_candidates": total_candidates - total_positive,
        "source_video_bytes": total_video_bytes,
        "max_initial_pixel_delta_across_replays": max(pixel_difference_maxima),
        "groups": groups,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "dataset_manifest.sha256").write_text(
        f"{_sha256(manifest_path)}  {manifest_path}\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("verification", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = build_dataset(args.plan, args.verification, args.output_dir)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "complete",
                    "group_count",
                    "mixed_group_count",
                    "candidate_count",
                    "positive_candidates",
                    "negative_candidates",
                    "max_initial_pixel_delta_across_replays",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())