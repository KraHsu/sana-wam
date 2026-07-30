#!/usr/bin/env python
"""Prepare one deterministic evaluation context from every RoboTwin episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from probe_sana_action_denoising import (  # noqa: E402
    build_dataset,
    load_or_prepare_pair,
)
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--task-name")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SANA distributional evaluation contexts",
        "",
        f"- Episodes/contexts: `{report['context_count']}`",
        f"- Prepared batches: `{report['pair_count']}`",
        f"- Sampling seed: `{report['seed']}`",
        f"- Task: `{report['task_name']}`",
        f"- Checkpoint used only for frozen preprocessing: "
        f"`{report['preprocessing_checkpoint']}`",
        f"- Checkpoint SHA256: `{report['preprocessing_checkpoint_sha256']}`",
        "- Protocol: one uniformly sampled valid window from every training "
        "episode, followed by a deterministic episode shuffle and disjoint "
        "two-context batching.",
        "",
        "| pair | item | episode | start | available windows | relative position |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in report["pairs"]:
        for item_index, item in enumerate(pair["samples"]):
            lines.append(
                f"| {pair['pair_index']} | {item_index} | "
                f"{item['episode_index']} | {item['start_frame']} | "
                f"{item['episode_window_count']} | "
                f"{item['window_position_fraction']:.6f} |"
            )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    cfg, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    if args.task_name:
        cfg.dataloader.task_name = args.task_name
    _, dataset = build_dataset(cfg)

    windows_by_episode: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for local_index, item in enumerate(dataset._window_index):
        episode_index, start, _ = item
        windows_by_episode[int(episode_index)].append((local_index, int(start)))
    episode_indices = sorted(windows_by_episode)
    if len(episode_indices) < 2 or len(episode_indices) % 2:
        raise RuntimeError(
            "distribution context preparation requires an even episode count"
        )

    rng = np.random.default_rng(args.seed)
    descriptors = []
    for episode_index in episode_indices:
        windows = windows_by_episode[episode_index]
        selected_offset = int(rng.integers(0, len(windows)))
        local_index, start = windows[selected_offset]
        descriptors.append(
            {
                "local_index": local_index,
                "episode_index": episode_index,
                "episode_path": dataset._episode_files[episode_index],
                "start_frame": start,
                "active_arm": "unknown",
                "episode_window_count": len(windows),
                "window_position_fraction": (
                    0.0
                    if len(windows) == 1
                    else selected_offset / (len(windows) - 1)
                ),
            }
        )
    permutation = rng.permutation(len(descriptors)).tolist()
    descriptors = [descriptors[index] for index in permutation]
    pairs = [descriptors[index : index + 2] for index in range(0, len(descriptors), 2)]

    pair_reports = []
    seen_episodes = set()
    for pair_index, pair in enumerate(pairs):
        cache_path = args.output_dir / f"prepared_pair_{pair_index:02d}.pt"
        tensors, samples = load_or_prepare_pair(
            architecture,
            dataset,
            pair,
            cache_path,
            rebuild=True,
        )
        for descriptor, sample in zip(pair, samples):
            sample["episode_window_count"] = descriptor["episode_window_count"]
            sample["window_position_fraction"] = descriptor[
                "window_position_fraction"
            ]
            episode_index = int(sample["episode_index"])
            if episode_index in seen_episodes:
                raise RuntimeError(f"duplicate episode {episode_index}")
            seen_episodes.add(episode_index)
        torch.save({"tensors": tensors, "samples": samples}, cache_path)
        pair_reports.append(
            {
                "pair_index": pair_index,
                "path": str(cache_path),
                "sha256": sha256_file(cache_path),
                "tensor_shapes": {
                    name: list(value.shape) for name, value in tensors.items()
                },
                "samples": samples,
            }
        )
        print(
            f"prepared pair {pair_index + 1}/{len(pairs)} "
            f"episodes={[item['episode_index'] for item in samples]}",
            flush=True,
        )

    if seen_episodes != set(episode_indices):
        raise RuntimeError("prepared contexts do not cover every episode exactly once")
    checkpoint_path = args.run_dir / args.checkpoint
    report = {
        "preprocessing_checkpoint": str(checkpoint_path),
        "preprocessing_checkpoint_sha256": sha256_file(checkpoint_path),
        "seed": args.seed,
        "task_name": str(cfg.dataloader.task_name),
        "episode_count": len(episode_indices),
        "context_count": len(descriptors),
        "pair_count": len(pairs),
        "episode_indices": episode_indices,
        "relative_position_min": min(
            item["window_position_fraction"] for item in descriptors
        ),
        "relative_position_max": max(
            item["window_position_fraction"] for item in descriptors
        ),
        "relative_position_mean": float(
            np.mean([item["window_position_fraction"] for item in descriptors])
        ),
        "pairs": pair_reports,
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "prepared_context_manifest.json"
    markdown_path = args.output_dir / "prepared_context_manifest.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
