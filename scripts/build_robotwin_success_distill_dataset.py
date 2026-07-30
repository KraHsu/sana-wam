#!/usr/bin/env python
"""Export successful baseline rollouts into standard RoboTwin episode HDF5 files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

try:
    from benchmarks.utils.action_conversion import rot6d_to_quat_xyzw
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmarks.utils.action_conversion import rot6d_to_quat_xyzw


CAMERAS = ("head", "left", "right")
ROBOTWIN_CAMERAS = {
    "head": "head_camera",
    "left": "left_camera",
    "right": "right_camera",
}
ACTION_DIM = 20
BASELINE_CHECKPOINT_SHA256 = (
    "aea6b58c9107aeeeffba561f39ffedfc3f8fd8420a4abfcab9038c8cdd4a129d"
)
BASELINE_ACTION_STATS_SHA256 = (
    "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, root: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} contains a non-object row")
    return rows


def _jpeg(path: Path) -> bytes:
    payload = path.read_bytes()
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise ValueError(f"not a complete JPEG: {path}")
    with Image.open(path) as image:
        image.verify()
    return payload


def _eef20_to_endpose(states: np.ndarray) -> dict[str, np.ndarray]:
    if states.ndim != 2 or states.shape[1] != ACTION_DIM:
        raise ValueError(f"states must have shape (T, 20), got {states.shape}")
    left_quat = np.stack([rot6d_to_quat_xyzw(row[3:9]) for row in states])
    right_quat = np.stack([rot6d_to_quat_xyzw(row[13:19]) for row in states])
    return {
        "left_endpose": np.concatenate([states[:, 0:3], left_quat], axis=1).astype(
            np.float32
        ),
        "left_gripper": states[:, 9].astype(np.float32),
        "right_endpose": np.concatenate([states[:, 10:13], right_quat], axis=1).astype(
            np.float32
        ),
        "right_gripper": states[:, 19].astype(np.float32),
    }


def _write_episode(
    path: Path,
    *,
    states: np.ndarray,
    camera_payloads: dict[str, list[bytes]],
    attrs: dict[str, Any],
) -> None:
    endpose = _eef20_to_endpose(states)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            for key, value in attrs.items():
                handle.attrs[key] = value
            observation = handle.create_group("observation")
            jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
            for camera in CAMERAS:
                group = observation.create_group(ROBOTWIN_CAMERAS[camera])
                dataset = group.create_dataset(
                    "rgb", shape=(len(states),), dtype=jpeg_dtype
                )
                payloads = camera_payloads[camera]
                if len(payloads) != len(states):
                    raise ValueError(
                        f"{camera} image count {len(payloads)} != state count {len(states)}"
                    )
                for index, payload in enumerate(payloads):
                    dataset[index] = np.frombuffer(payload, dtype=np.uint8)
            endpose_group = handle.create_group("endpose")
            for key, value in endpose.items():
                endpose_group.create_dataset(key, data=value)
            handle.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(run_dir: Path, output_root: Path, action_stats_path: Path) -> dict:
    run_dir = run_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root must be absent or empty: {output_root}")
    if (run_dir / "eval.exit").read_text().strip() != "0":
        raise ValueError("source rollout did not exit successfully")
    if _sha256(action_stats_path) != BASELINE_ACTION_STATS_SHA256:
        raise ValueError("action stats do not match the baseline digest")
    checkpoint_line = (run_dir / "checkpoint.sha256").read_text().split()
    if not checkpoint_line or checkpoint_line[0] != BASELINE_CHECKPOINT_SHA256:
        raise ValueError("source run is not the baseline checkpoint")

    episode_rows = _load_jsonl(run_dir / "telemetry/client/episodes.jsonl")
    step_rows = _load_jsonl(run_dir / "telemetry/client/steps.jsonl")
    starts = [row for row in episode_rows if row.get("event") == "episode_start"]
    ends = {
        row["episode_key"]: row
        for row in episode_rows
        if row.get("event") == "episode_end"
    }
    grouped_steps: dict[str, list[dict]] = {}
    for row in step_rows:
        grouped_steps.setdefault(row["episode_key"], []).append(row)

    data_dir = (
        output_root / "dataset" / "adjust_bottle" / "aloha-agilex_self_distill" / "data"
    )
    instructions_dir = data_dir.parent / "instructions"
    data_dir.mkdir(parents=True)
    instructions_dir.mkdir()
    stats = np.load(action_stats_path, allow_pickle=True).item()["eef"]
    stats_min = np.asarray(stats["min"], dtype=np.float64)
    stats_max = np.asarray(stats["max"], dtype=np.float64)

    exported = []
    scene_info = {}
    rejected = []
    debug_root = run_dir / "debug_images"
    for source_ordinal, start in enumerate(starts):
        context = start.get("context", {})
        episode_key = context.get("episode_key")
        end = ends.get(episode_key)
        if end is None:
            raise ValueError(f"episode {episode_key!r} has no end record")
        if not bool(end.get("success")):
            continue
        steps = sorted(
            grouped_steps.get(episode_key, []), key=lambda row: row["env_step"]
        )
        expected_steps = list(range(len(steps)))
        if [row.get("env_step") for row in steps] != expected_steps:
            raise ValueError(f"episode {episode_key!r} has a telemetry step gap")
        if len(steps) != end.get("step_count"):
            raise ValueError(f"episode {episode_key!r} step count mismatch")
        if len(steps) < 2:
            rejected.append({"episode_key": episode_key, "reason": "too_short"})
            continue
        pre_states = np.asarray([row["pre_proprio"] for row in steps], dtype=np.float32)
        if (
            pre_states.shape != (len(steps), ACTION_DIM)
            or not np.isfinite(pre_states).all()
        ):
            raise ValueError(f"episode {episode_key!r} has invalid EEF20 states")

        episode_debug = debug_root / f"ep{source_ordinal:04d}"
        terminal_meta_path = episode_debug / "terminal" / "meta.json"
        if not terminal_meta_path.is_file():
            raise FileNotFoundError(
                f"successful episode {episode_key!r} has no terminal debug state"
            )
        terminal_meta = json.loads(terminal_meta_path.read_text(encoding="utf-8"))
        terminal_state = np.asarray(terminal_meta.get("state"), dtype=np.float32)
        if (
            terminal_state.shape != (ACTION_DIM,)
            or not np.isfinite(terminal_state).all()
        ):
            raise ValueError(f"episode {episode_key!r} terminal state is invalid")
        final_post = np.asarray(steps[-1].get("post_proprio"), dtype=np.float32)
        if final_post.shape != (ACTION_DIM,) or not np.allclose(
            terminal_state, final_post, atol=1e-5, rtol=1e-5
        ):
            raise ValueError(f"episode {episode_key!r} terminal state mismatch")

        states = np.concatenate([pre_states, terminal_state[None, :]], axis=0)
        payloads = {camera: [] for camera in CAMERAS}
        for step in steps:
            debug_step = episode_debug / f"step_{int(step['env_step']) + 1:04d}"
            meta = json.loads((debug_step / "meta.json").read_text(encoding="utf-8"))
            if meta.get("prompt") != step.get("prompt"):
                raise ValueError(f"episode {episode_key!r} debug prompt mismatch")
            for camera in CAMERAS:
                payloads[camera].append(_jpeg(debug_step / f"{camera}.jpg"))
        for camera in CAMERAS:
            payloads[camera].append(_jpeg(episode_debug / "terminal" / f"{camera}.jpg"))

        prompt = steps[0].get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"episode {episode_key!r} prompt is invalid")
        if any(step.get("prompt") != prompt for step in steps):
            raise ValueError(f"episode {episode_key!r} prompt changed")
        output_index = len(exported)
        episode_path = data_dir / f"episode{output_index}.hdf5"
        _write_episode(
            episode_path,
            states=states,
            camera_payloads=payloads,
            attrs={
                "schema_version": 1,
                "source_kind": "successful_policy_rollout",
                "source_episode_key": episode_key,
                "environment_seed": int(end["environment_seed"]),
                "source_step_count": len(steps),
                "success": True,
            },
        )
        instruction_path = instructions_dir / f"episode{output_index}.json"
        instruction_path.write_text(
            json.dumps({"seen": [prompt], "unseen": [prompt]}, indent=2) + "\n",
            encoding="utf-8",
        )
        normalized = 2.0 * (states - stats_min) / (stats_max - stats_min) - 1.0
        artifact = _artifact(episode_path, output_root)
        record = {
            "output_episode_index": output_index,
            "source_episode_ordinal": source_ordinal,
            "source_episode_key": episode_key,
            "environment_seed": int(end["environment_seed"]),
            "prompt": prompt,
            "step_count": len(steps),
            "frame_count": len(states),
            "normalized_min": float(normalized.min()),
            "normalized_max": float(normalized.max()),
            "normalized_out_of_range_fraction": float((np.abs(normalized) > 1).mean()),
            "artifact": artifact,
        }
        exported.append(record)
        scene_info[f"episode_{output_index}"] = {
            "info": {
                "active_arm": "both",
                "source_kind": "successful_policy_rollout",
                "environment_seed": int(end["environment_seed"]),
            }
        }

    if not exported:
        raise ValueError("source run contains no exportable successful episodes")
    scene_info_path = data_dir.parent / "scene_info.json"
    scene_info_path.write_text(
        json.dumps(scene_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "kind": "robotwin_success_distillation_dataset",
        "complete": True,
        "source_run": str(run_dir),
        "source_checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
        "action_stats_sha256": BASELINE_ACTION_STATS_SHA256,
        "successful_source_episode_count": sum(
            bool(row.get("success")) for row in ends.values()
        ),
        "exported_episode_count": len(exported),
        "rejected": rejected,
        "episodes": exported,
        "source_artifacts": {
            "prompt_manifest": _artifact(run_dir / "prompt_manifest.json", run_dir),
            "client_episodes": _artifact(
                run_dir / "telemetry/client/episodes.jsonl", run_dir
            ),
            "client_steps": _artifact(
                run_dir / "telemetry/client/steps.jsonl", run_dir
            ),
        },
        "dataset": {
            "dataset_dir": str(output_root / "dataset"),
            "task_name": "adjust_bottle",
            "robot": "aloha-agilex",
            "variant": "self_distill",
            "data_root": str(data_dir),
            "scene_info_artifact": _artifact(scene_info_path, output_root),
        },
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = _sha256(manifest_path)
    (output_root / "dataset_manifest.sha256").write_text(
        f"{digest}  dataset_manifest.json\n", encoding="ascii"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--action-stats-path",
        type=Path,
        default=Path(
            "/home/zch/wuji-openwam-dev/sandbox/sana_ar_graft_AC_lownoise_only/"
            "2026-07-04_13-03-25/action_stats.npy"
        ),
    )
    args = parser.parse_args()
    manifest = build(
        args.run_dir,
        args.output_root,
        args.action_stats_path.expanduser().resolve(),
    )
    print(
        "success_distill_export: PASS "
        f"episodes={manifest['exported_episode_count']} "
        f"output={args.output_root.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
