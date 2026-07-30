#!/usr/bin/env python
"""Measure same-seed video Euler convergence independently of paired target RMSE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from probe_sana_action_denoising import to_device  # noqa: E402
from probe_sana_ar_decomposition import build_expert_history  # noqa: E402
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--steps", default="1,2,4,10,20,50")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


@dataclass
class RMS:
    square_sum: float = 0.0
    count: int = 0

    def add(self, value: torch.Tensor) -> None:
        value = value.double()
        self.square_sum += float(value.square().sum().item())
        self.count += value.numel()

    def result(self) -> float:
        return (self.square_sum / self.count) ** 0.5


def sample_rms(value: torch.Tensor) -> list[float]:
    """Return one RMS per batch item without hiding context-level outliers."""
    flattened = value.double().reshape(value.shape[0], -1)
    return torch.sqrt(flattened.square().mean(dim=1)).cpu().tolist()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Video Euler solver convergence",
        "",
        "Every solution uses the same Gaussian seed, expert cache, and context. "
        "Output deltas test numerical convergence without treating a paired "
        "expert future as the unique valid generative sample.",
        "",
        "| Euler steps | paired expert RMSE | delta from finest solution |",
        "| ---: | ---: | ---: |",
    ]
    for row in report["solutions"]:
        delta = row["delta_from_finest_rms"]
        lines.append(
            f"| {row['steps']} | {row['paired_expert_rmse']:.6f} | "
            f"{'-' if delta is None else f'{delta:.6f}'} |"
        )
    lines.extend(
        [
            "",
            "| Coarse steps | Fine steps | same-seed output delta RMS |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in report["adjacent_deltas"]:
        lines.append(
            f"| {row['coarse_steps']} | {row['fine_steps']} | "
            f"{row['output_delta_rms']:.6f} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    step_counts = [int(value) for value in args.steps.split(",")]
    if len(step_counts) < 3 or any(value <= 0 for value in step_counts):
        raise ValueError("at least three positive step counts are required")
    if step_counts != sorted(set(step_counts)):
        raise ValueError("step counts must be unique and strictly increasing")
    if args.pairs <= 0 or args.chunk <= 0:
        raise ValueError("pairs and continuation chunk must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    device = torch.device(args.device)
    expert_errors = {steps: RMS() for steps in step_counts}
    finest_deltas = {steps: RMS() for steps in step_counts[:-1]}
    adjacent_deltas: dict[tuple[int, int], RMS] = defaultdict(RMS)
    context_metrics = []

    for pair_index in range(args.pairs):
        saved = torch.load(
            args.prepared_dir / f"prepared_pair_{pair_index:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        metadata = saved["samples"]
        payload = to_device(saved["tensors"], device, architecture.dtype)
        actions = payload["actions"]
        latents = payload["input_latents"]
        frame_chunk_size = int(architecture._ar_frame_chunk_size)
        chunks = latents.shape[2] // frame_chunk_size
        if args.chunk >= chunks:
            raise ValueError(f"chunk {args.chunk} is outside 0..{chunks - 1}")
        tokens_per_chunk = actions.shape[1] // chunks
        video_chunks = list(latents.split(frame_chunk_size, dim=2))
        action_chunks = list(actions.split(tokens_per_chunk, dim=1))
        proprio_chunks = [
            payload["proprio_seq"][:, chunk * tokens_per_chunk]
            for chunk in range(chunks)
        ]
        context, context_mask = architecture._rollout_step_context(
            payload["context"], payload["seq_lens"], payload["proprio_state"]
        )
        cache = build_expert_history(
            architecture,
            driver,
            video_chunks,
            action_chunks,
            proprio_chunks,
            stop_chunk=args.chunk,
            context=context,
            context_mask=context_mask,
        )
        video_proprio, _ = architecture._rollout_proprio_deltas(
            proprio_chunks[args.chunk]
        )
        outputs = {}
        for steps in step_counts:
            architecture.video_backbone.scheduler.set_timesteps(steps)
            sigmas = [
                float(value)
                for value in architecture.video_backbone.scheduler.sigmas.tolist()
            ] + [0.0]
            timesteps = architecture.video_backbone.scheduler.timesteps
            generator = torch.Generator(device=device).manual_seed(
                args.seed + pair_index * 1009 + args.chunk * 100_003
            )
            prediction = architecture._denoise_video_chunk(
                driver,
                cache,
                frame_id=2 * args.chunk,
                like=video_chunks[args.chunk],
                v_sigmas=sigmas,
                v_ts=timesteps,
                context=context,
                context_mask=context_mask,
                gen=generator,
                v_proprio=video_proprio,
            )
            removed = cache.pop_frame(2 * args.chunk)
            if any(len(layer) != 1 for layer in removed):
                raise RuntimeError("predicted video cache cleanup was incomplete")
            outputs[steps] = prediction.float()
            expert_errors[steps].add(prediction.float() - video_chunks[args.chunk])

        finest = outputs[step_counts[-1]]
        for steps in step_counts[:-1]:
            finest_deltas[steps].add(outputs[steps] - finest)
        for coarse, fine in zip(step_counts, step_counts[1:]):
            adjacent_deltas[(coarse, fine)].add(outputs[coarse] - outputs[fine])

        if len(metadata) != finest.shape[0]:
            raise RuntimeError("prepared metadata and tensor batch sizes differ")
        sample_expert_errors = {
            steps: sample_rms(outputs[steps] - video_chunks[args.chunk])
            for steps in step_counts
        }
        sample_finest_deltas = {
            steps: sample_rms(outputs[steps] - finest)
            for steps in step_counts[:-1]
        }
        sample_adjacent_deltas = {
            (coarse, fine): sample_rms(outputs[coarse] - outputs[fine])
            for coarse, fine in zip(step_counts, step_counts[1:])
        }
        for context_index, item in enumerate(metadata):
            distances = [
                sample_finest_deltas[steps][context_index]
                for steps in step_counts[:-1]
            ]
            adjacent = [
                sample_adjacent_deltas[(coarse, fine)][context_index]
                for coarse, fine in zip(step_counts, step_counts[1:])
            ]
            values = (
                [sample_expert_errors[steps][context_index] for steps in step_counts]
                + distances
                + adjacent
            )
            context_metrics.append(
                {
                    "pair_index": pair_index,
                    "context_index": context_index,
                    "episode_index": int(item["episode_index"]),
                    "episode_path": item.get("episode_path"),
                    "start_frame": int(item["start_frame"]),
                    "solutions": [
                        {
                            "steps": steps,
                            "paired_expert_rmse": sample_expert_errors[steps][
                                context_index
                            ],
                            "delta_from_finest_rms": (
                                None
                                if steps == step_counts[-1]
                                else sample_finest_deltas[steps][context_index]
                            ),
                        }
                        for steps in step_counts
                    ],
                    "adjacent_deltas": [
                        {
                            "coarse_steps": coarse,
                            "fine_steps": fine,
                            "output_delta_rms": sample_adjacent_deltas[
                                (coarse, fine)
                            ][context_index],
                        }
                        for coarse, fine in zip(step_counts, step_counts[1:])
                    ],
                    "distance_to_finest_strictly_decreases": all(
                        coarse > fine > 0.0
                        for coarse, fine in zip(distances, distances[1:])
                    ),
                    "adjacent_delta_strictly_decreases": all(
                        coarse > fine > 0.0
                        for coarse, fine in zip(adjacent, adjacent[1:])
                    ),
                    "all_finite": all(math.isfinite(value) for value in values),
                }
            )

    solutions = []
    for steps in step_counts:
        solutions.append(
            {
                "steps": steps,
                "paired_expert_rmse": expert_errors[steps].result(),
                "delta_from_finest_rms": (
                    None if steps == step_counts[-1] else finest_deltas[steps].result()
                ),
            }
        )
    deltas = []
    for coarse, fine in zip(step_counts, step_counts[1:]):
        deltas.append(
            {
                "coarse_steps": coarse,
                "fine_steps": fine,
                "output_delta_rms": adjacent_deltas[(coarse, fine)].result(),
            }
        )
    manifest_path = args.prepared_dir / "prepared_context_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "task_name": manifest.get("task_name"),
        "pair_count": args.pairs,
        "context_count": len(context_metrics),
        "chunk": args.chunk,
        "step_counts": step_counts,
        "seed": args.seed,
        "solutions": solutions,
        "adjacent_deltas": deltas,
        "context_metrics": context_metrics,
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "video_solver_convergence.json"
    markdown_path = args.output_dir / "video_solver_convergence.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
