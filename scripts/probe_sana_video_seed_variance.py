#!/usr/bin/env python
"""Decompose paired rollout MSE into conditional-mean error and seed variance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--steps", default="1,10")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


@dataclass
class Decomposition:
    total_error_square_sum: float = 0.0
    total_error_count: int = 0
    mean_error_square_sum: float = 0.0
    mean_error_count: int = 0
    variance_square_sum: float = 0.0
    variance_count: int = 0
    pairwise_mse_sum: float = 0.0
    pairwise_count: int = 0
    target_distance_sum: float = 0.0
    target_distance_count: int = 0
    pairwise_distance_sum: float = 0.0

    def add(self, predictions: torch.Tensor, target: torch.Tensor) -> None:
        predictions = predictions.double()
        target = target.double()
        mean = predictions.mean(dim=0)
        total_error = predictions - target.unsqueeze(0)
        centered = predictions - mean.unsqueeze(0)
        mean_error = mean - target
        self.total_error_square_sum += float(total_error.square().sum().item())
        self.total_error_count += total_error.numel()
        self.mean_error_square_sum += float(mean_error.square().sum().item())
        self.mean_error_count += mean_error.numel()
        self.variance_square_sum += float(centered.square().sum().item())
        self.variance_count += centered.numel()

        flattened = predictions.flatten(start_dim=1)
        elements = flattened.shape[1]
        target_flat = target.flatten().unsqueeze(0)
        target_distances = (
            torch.linalg.vector_norm(flattened - target_flat, dim=1)
            / elements**0.5
        )
        pairwise_distances = torch.pdist(flattened, p=2) / elements**0.5
        self.target_distance_sum += float(target_distances.sum().item())
        self.target_distance_count += target_distances.numel()
        self.pairwise_distance_sum += float(pairwise_distances.sum().item())
        self.pairwise_mse_sum += float(pairwise_distances.square().sum().item())
        self.pairwise_count += pairwise_distances.numel()

    def result(self) -> dict[str, float]:
        total_mse = self.total_error_square_sum / self.total_error_count
        mean_mse = self.mean_error_square_sum / self.mean_error_count
        variance = self.variance_square_sum / self.variance_count
        pairwise_mse = self.pairwise_mse_sum / self.pairwise_count
        seed_count = self.total_error_count / self.mean_error_count
        corrected_mean_mse = mean_mse - variance / (seed_count - 1.0)
        unbiased_variance = variance * seed_count / (seed_count - 1.0)
        mean_target_distance = (
            self.target_distance_sum / self.target_distance_count
        )
        mean_pairwise_distance = (
            self.pairwise_distance_sum / self.pairwise_count
        )
        return {
            "paired_mse": total_mse,
            "paired_rmse": total_mse**0.5,
            "ensemble_mean_mse": mean_mse,
            "ensemble_mean_rmse": mean_mse**0.5,
            "within_seed_variance": variance,
            "within_seed_rms": variance**0.5,
            "pairwise_seed_mse": pairwise_mse,
            "pairwise_seed_rmse": pairwise_mse**0.5,
            "mse_decomposition_residual": total_mse - mean_mse - variance,
            "monte_carlo_corrected_mean_mse": corrected_mean_mse,
            "monte_carlo_corrected_mean_rmse": corrected_mean_mse**0.5,
            "unbiased_within_seed_variance": unbiased_variance,
            "mean_target_distance": mean_target_distance,
            "mean_pairwise_seed_distance": mean_pairwise_distance,
            "energy_score": mean_target_distance - 0.5 * mean_pairwise_distance,
        }


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


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Video seed-variance decomposition",
        "",
        "For each expert context, multiple outputs are generated from independent "
        "Gaussian seeds. The paired MSE identity is exact:",
        "",
        "```text",
        "E_seed ||prediction - expert||^2",
        "  = ||E_seed prediction - expert||^2 + E_seed ||prediction - mean||^2.",
        "```",
        "",
        "| Euler steps | paired RMSE | ensemble-mean RMSE | corrected-mean RMSE | seed RMS | energy score | residual |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["metrics"]:
        lines.append(
            f"| {row['steps']} | {row['paired_rmse']:.6f} | "
            f"{row['ensemble_mean_rmse']:.6f} | "
            f"{row['monte_carlo_corrected_mean_rmse']:.6f} | "
            f"{row['within_seed_rms']:.6f} | "
            f"{row['energy_score']:.6f} | "
            f"{row['mse_decomposition_residual']:.3e} |"
        )
    comparison = report["comparison"]
    lines.extend(
        [
            "",
            f"- Paired MSE gap ({comparison['to_steps']} - "
            f"{comparison['from_steps']} steps): "
            f"`{comparison['paired_mse_gap']:+.6f}`",
            f"- Ensemble-mean MSE contribution: "
            f"`{comparison['ensemble_mean_mse_gap']:+.6f}`",
            f"- Monte-Carlo-corrected mean MSE contribution: "
            f"`{comparison['monte_carlo_corrected_mean_mse_gap']:+.6f}`",
            f"- Seed-variance contribution: "
            f"`{comparison['within_seed_variance_gap']:+.6f}`",
            f"- Variance fraction of paired-MSE gap: "
            f"`{comparison['variance_fraction_of_gap']:.6f}`",
            f"- Matched-seed output delta RMS: "
            f"`{comparison['matched_seed_output_delta_rms']:.6f}`",
            f"- Ensemble-mean output delta RMS: "
            f"`{comparison['ensemble_mean_output_delta_rms']:.6f}`",
            f"- Energy-score gap: `{comparison['energy_score_gap']:+.6f}`",
        ]
    )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    step_counts = [int(value) for value in args.steps.split(",")]
    if len(step_counts) != 2 or any(value <= 0 for value in step_counts):
        raise ValueError("exactly two positive step counts are required")
    if step_counts[0] >= step_counts[1]:
        raise ValueError("step counts must be strictly increasing")
    if args.pairs <= 0 or args.seeds < 2:
        raise ValueError("pairs must be positive and seeds must be at least 2")
    if args.chunk <= 0:
        raise ValueError("chunk must be a continuation chunk")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    device = torch.device(args.device)
    sums = {steps: Decomposition() for steps in step_counts}
    matched_seed_delta = RMS()
    ensemble_mean_delta = RMS()

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
        outputs: dict[int, torch.Tensor] = {}
        for steps in step_counts:
            architecture.video_backbone.scheduler.set_timesteps(steps)
            sigmas = [
                float(value)
                for value in architecture.video_backbone.scheduler.sigmas.tolist()
            ] + [0.0]
            timesteps = architecture.video_backbone.scheduler.timesteps
            predictions = []
            for seed_index in range(args.seeds):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed
                    + pair_index * 1009
                    + args.chunk * 100_003
                    + seed_index * 10_000_019
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
                predictions.append(prediction.float())
            stacked = torch.stack(predictions)
            outputs[steps] = stacked
            sums[steps].add(stacked, video_chunks[args.chunk].float())

        first = outputs[step_counts[0]]
        second = outputs[step_counts[1]]
        matched_seed_delta.add(second - first)
        ensemble_mean_delta.add(second.mean(dim=0) - first.mean(dim=0))

    metrics = []
    by_steps = {}
    for steps in step_counts:
        row = {"steps": steps, **sums[steps].result()}
        metrics.append(row)
        by_steps[steps] = row
    first = by_steps[step_counts[0]]
    second = by_steps[step_counts[1]]
    paired_gap = second["paired_mse"] - first["paired_mse"]
    variance_gap = (
        second["within_seed_variance"] - first["within_seed_variance"]
    )
    comparison = {
        "from_steps": step_counts[0],
        "to_steps": step_counts[1],
        "paired_mse_gap": paired_gap,
        "ensemble_mean_mse_gap": (
            second["ensemble_mean_mse"] - first["ensemble_mean_mse"]
        ),
        "monte_carlo_corrected_mean_mse_gap": (
            second["monte_carlo_corrected_mean_mse"]
            - first["monte_carlo_corrected_mean_mse"]
        ),
        "within_seed_variance_gap": variance_gap,
        "variance_fraction_of_gap": variance_gap / paired_gap,
        "matched_seed_output_delta_rms": matched_seed_delta.result(),
        "ensemble_mean_output_delta_rms": ensemble_mean_delta.result(),
        "energy_score_gap": second["energy_score"] - first["energy_score"],
    }
    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "pair_count": args.pairs,
        "seed_count": args.seeds,
        "chunk": args.chunk,
        "step_counts": step_counts,
        "seed": args.seed,
        "metrics": metrics,
        "comparison": comparison,
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "video_seed_variance.json"
    markdown_path = args.output_dir / "video_seed_variance.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
