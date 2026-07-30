#!/usr/bin/env python
"""Extract expert-only standard episodes from full-suffix takeover artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import h5py
import numpy as np


CAMERAS = ("head_camera", "left_camera", "right_camera")


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


def _write_episode(source: Path, target: Path, start: int) -> int:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
            total = src["endpose/left_gripper"].shape[0]
            if not 0 <= start < total - 1:
                raise ValueError(f"invalid expert suffix start {start} for {source}")
            length = total - start
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["source_kind"] = "full_expert_suffix_expert_only"
            dst.attrs["source_artifact_sha256"] = _sha256(source)
            dst.attrs["source_frame_start"] = start

            observation = dst.create_group("observation")
            jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
            for camera in CAMERAS:
                raw = src[f"observation/{camera}/rgb"]
                group = observation.create_group(camera)
                dataset = group.create_dataset("rgb", shape=(length,), dtype=jpeg_dtype)
                for out_index, source_index in enumerate(range(start, total)):
                    dataset[out_index] = np.asarray(raw[source_index], dtype=np.uint8)

            endpose = dst.create_group("endpose")
            for key in (
                "left_endpose",
                "left_gripper",
                "right_endpose",
                "right_gripper",
            ):
                endpose.create_dataset(key, data=src[f"endpose/{key}"][start:])
            dst.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return length


def build(source_root: Path, output_root: Path) -> dict:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root must be absent or empty: {output_root}")
    source_manifest_path = source_root / "dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    samples = source_manifest.get("samples")
    if source_manifest.get("complete") is not True or not isinstance(samples, list):
        raise ValueError("source full-suffix manifest is incomplete")

    variant_root = (
        output_root / "dataset" / "adjust_bottle" / "aloha-agilex_full_expert_suffix"
    )
    data_dir = variant_root / "data"
    instructions_dir = variant_root / "instructions"
    data_dir.mkdir(parents=True)
    instructions_dir.mkdir()
    exported = []
    scene_info = {}
    for output_index, sample in enumerate(samples):
        source_path = source_root / sample["artifact"]["path"]
        source_total = int(sample["frame_count"])
        expert_count = int(sample["expert_frame_count"])
        start = source_total - expert_count
        target_path = data_dir / f"episode{output_index}.hdf5"
        frame_count = _write_episode(source_path, target_path, start)
        if frame_count != expert_count:
            raise ValueError("expert suffix frame count changed during extraction")
        prompt = sample["prompt"]
        (instructions_dir / f"episode{output_index}.json").write_text(
            json.dumps({"seen": [prompt], "unseen": [prompt]}, indent=2) + "\n",
            encoding="utf-8",
        )
        scene_info[f"episode_{output_index}"] = {
            "info": {
                "active_arm": sample["active_arm"],
                "source_kind": "full_expert_suffix",
                "environment_seed": sample["environment_seed"],
                "anchor_env_step": sample["anchor_env_step"],
            }
        }
        exported.append(
            {
                **{
                    key: sample[key]
                    for key in (
                        "anchor_id",
                        "environment_seed",
                        "anchor_env_step",
                        "prompt",
                        "active_arm",
                        "expert_frame_count",
                    )
                },
                "output_episode_index": output_index,
                "source_frame_start": start,
                "frame_count": frame_count,
                "artifact": _artifact(target_path, output_root),
            }
        )

    if not exported:
        raise ValueError("source manifest has no full expert suffix samples")
    scene_info_path = variant_root / "scene_info.json"
    scene_info_path.write_text(
        json.dumps(scene_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "kind": "robotwin_full_expert_suffix_dataset",
        "complete": True,
        "source_manifest_artifact": {
            "path": str(source_manifest_path),
            "size_bytes": source_manifest_path.stat().st_size,
            "sha256": _sha256(source_manifest_path),
        },
        "sample_count": len(exported),
        "samples": exported,
        "dataset": {
            "dataset_dir": str(output_root / "dataset"),
            "task_name": "adjust_bottle",
            "robot": "aloha-agilex",
            "variant": "full_expert_suffix",
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
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    manifest = build(args.source_root, args.output_root)
    print(
        "expert_suffix_export: PASS "
        f"episodes={manifest['sample_count']} "
        f"output={args.output_root.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
