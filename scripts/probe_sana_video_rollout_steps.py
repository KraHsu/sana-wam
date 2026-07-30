#!/usr/bin/env python
"""Measure predicted-video latent error as a function of Euler step count."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
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
from probe_sana_ar_decomposition import build_expert_history, cosine  # noqa: E402
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
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


class MetricSums:
    def __init__(self) -> None:
        self.square_sum = 0.0
        self.element_count = 0
        self.cosine_sum = 0.0
        self.sample_count = 0

    def add(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        error = prediction.float() - target.float()
        self.square_sum += float(error.double().square().sum().item())
        self.element_count += error.numel()
        self.cosine_sum += cosine(prediction.float(), target.float())
        self.sample_count += 1

    def final(self) -> tuple[float, float]:
        return (
            (self.square_sum / self.element_count) ** 0.5,
            self.cosine_sum / self.sample_count,
        )


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Predicted-video rollout step sweep",
        "",
        "Every step count starts from identical Gaussian noise and identical "
        "expert video/action cache history.",
        "",
        "| Euler steps | chunk | latent RMSE | latent cosine |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in report["metrics"]:
        lines.append(
            f"| {row['steps']} | {row['chunk']} | "
            f"{row['latent_rmse']:.6f} | {row['latent_cosine']:.6f} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    step_counts = [int(value) for value in args.steps.split(",")]
    if not step_counts or any(value <= 0 for value in step_counts):
        raise ValueError("steps must be positive integers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    device = torch.device(args.device)
    sums: dict[tuple[int, int], MetricSums] = defaultdict(MetricSums)

    for pair_index in range(args.pairs):
        saved = torch.load(
            args.prepared_dir / f"prepared_pair_{pair_index:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        payload = to_device(saved["tensors"], device, architecture.dtype)
        actions = payload["actions"]
        latents = payload["input_latents"]
        frame_chunk_size = int(architecture._ar_frame_chunk_size)
        chunks = latents.shape[2] // frame_chunk_size
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

        for chunk in range(1, chunks):
            cache = build_expert_history(
                architecture,
                driver,
                video_chunks,
                action_chunks,
                proprio_chunks,
                stop_chunk=chunk,
                context=context,
                context_mask=context_mask,
            )
            video_proprio, _ = architecture._rollout_proprio_deltas(
                proprio_chunks[chunk]
            )
            for steps in step_counts:
                architecture.video_backbone.scheduler.set_timesteps(steps)
                sigmas = [
                    float(value)
                    for value in architecture.video_backbone.scheduler.sigmas.tolist()
                ] + [0.0]
                timesteps = architecture.video_backbone.scheduler.timesteps
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + pair_index * 1009 + chunk * 100_003
                )
                prediction = architecture._denoise_video_chunk(
                    driver,
                    cache,
                    frame_id=2 * chunk,
                    like=video_chunks[chunk],
                    v_sigmas=sigmas,
                    v_ts=timesteps,
                    context=context,
                    context_mask=context_mask,
                    gen=generator,
                    v_proprio=video_proprio,
                )
                removed = cache.pop_frame(2 * chunk)
                if any(len(layer) != 1 for layer in removed):
                    raise RuntimeError("predicted video cache cleanup was incomplete")
                sums[(steps, chunk)].add(prediction, video_chunks[chunk])

    metrics = []
    for steps in step_counts:
        chunks = sorted(chunk for count, chunk in sums if count == steps)
        for chunk in chunks:
            latent_rmse, latent_cosine = sums[(steps, chunk)].final()
            metrics.append(
                {
                    "steps": steps,
                    "chunk": chunk,
                    "latent_rmse": latent_rmse,
                    "latent_cosine": latent_cosine,
                }
            )
    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "pair_count": args.pairs,
        "step_counts": step_counts,
        "metrics": metrics,
        "total_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "video_rollout_steps.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output_dir / "video_rollout_steps.md").write_text(markdown(report))
    print(json.dumps({"output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
