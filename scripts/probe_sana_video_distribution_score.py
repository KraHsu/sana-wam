#!/usr/bin/env python
"""Score SANA video samples with proper latent and fixed-feature energy scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", default="1,10")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_coarse_motion_features(value: torch.Tensor) -> torch.Tensor:
    """Apply a checkpoint-independent low-frequency motion representation."""
    leading = value.shape[:-4]
    frames = value.shape[-3]
    flattened = value.float().reshape(-1, *value.shape[-4:])
    coarse = F.adaptive_avg_pool3d(flattened, (frames, 12, 10))
    temporal = coarse[:, :, 1:] - coarse[:, :, :-1]
    horizontal = coarse[..., 1:] - coarse[..., :-1]
    vertical = coarse[..., 1:, :] - coarse[..., :-1, :]
    spatial_mean = flattened.mean(dim=(-1, -2))
    spatial_std = flattened.std(dim=(-1, -2), unbiased=False)
    parts = [
        coarse.flatten(start_dim=1),
        temporal.flatten(start_dim=1),
        horizontal.flatten(start_dim=1),
        vertical.flatten(start_dim=1),
        spatial_mean.flatten(start_dim=1),
        spatial_std.flatten(start_dim=1),
    ]
    features = torch.cat(parts, dim=1)
    return features.reshape(*leading, features.shape[-1])


def representation_vectors(
    predictions: torch.Tensor,
    target: torch.Tensor,
    representation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if representation == "latent":
        return predictions.flatten(start_dim=2), target.flatten(start_dim=1)
    if representation == "fixed_coarse_motion":
        return (
            fixed_coarse_motion_features(predictions),
            fixed_coarse_motion_features(target),
        )
    raise ValueError(representation)


def context_score(predictions: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Score one context represented as seed-by-feature and feature tensors."""
    predictions = predictions.double()
    target = target.double()
    seed_count, dimensions = predictions.shape
    mean = predictions.mean(dim=0)
    centered = predictions - mean.unsqueeze(0)
    errors = predictions - target.unsqueeze(0)
    paired_mse = float(errors.square().mean().item())
    mean_mse = float((mean - target).square().mean().item())
    variance = float(centered.square().mean().item())
    corrected_mean_mse = mean_mse - variance / (seed_count - 1)
    target_distances = torch.linalg.vector_norm(errors, dim=1) / dimensions**0.5
    pairwise_distances = torch.pdist(predictions, p=2) / dimensions**0.5
    mean_target_distance = float(target_distances.mean().item())
    mean_pairwise_distance = float(pairwise_distances.mean().item())
    return {
        "paired_mse": paired_mse,
        "ensemble_mean_mse": mean_mse,
        "monte_carlo_corrected_mean_mse": corrected_mean_mse,
        "within_seed_variance": variance,
        "mean_target_distance": mean_target_distance,
        "mean_pairwise_seed_distance": mean_pairwise_distance,
        "energy_score": mean_target_distance - 0.5 * mean_pairwise_distance,
        "mse_decomposition_residual": paired_mse - mean_mse - variance,
    }


def bootstrap_mean_ci(
    values: list[float], *, seed: int, samples: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def aggregate_contexts(
    rows: list[dict[str, Any]], *, bootstrap_seed: int, bootstrap_samples: int
) -> dict[str, float]:
    paired_mse = float(np.mean([row["paired_mse"] for row in rows]))
    mean_mse = float(np.mean([row["ensemble_mean_mse"] for row in rows]))
    corrected_mean_mse = float(
        np.mean([row["monte_carlo_corrected_mean_mse"] for row in rows])
    )
    variance = float(np.mean([row["within_seed_variance"] for row in rows]))
    energies = [row["energy_score"] for row in rows]
    ci_low, ci_high = bootstrap_mean_ci(
        energies, seed=bootstrap_seed, samples=bootstrap_samples
    )
    return {
        "context_count": len(rows),
        "paired_mse": paired_mse,
        "paired_rmse": paired_mse**0.5,
        "ensemble_mean_mse": mean_mse,
        "ensemble_mean_rmse": mean_mse**0.5,
        "monte_carlo_corrected_mean_mse": corrected_mean_mse,
        "monte_carlo_corrected_mean_rmse": corrected_mean_mse**0.5,
        "within_seed_variance": variance,
        "within_seed_rms": variance**0.5,
        "energy_score": float(np.mean(energies)),
        "energy_score_ci95_low": ci_low,
        "energy_score_ci95_high": ci_high,
        "max_abs_mse_decomposition_residual": max(
            abs(row["mse_decomposition_residual"]) for row in rows
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SANA video distribution score",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Contexts: `{report['context_count']}` from `{report['pair_count']}` batches",
        f"- Seeds per context: `{report['seed_count']}`",
        f"- Prepared manifest SHA256: `{report['prepared_manifest_sha256']}`",
        "- Energy score is lower-is-better and balances target distance against "
        "sample dispersion.",
        "",
    ]
    for representation in report["representations"]:
        lines.extend(
            [
                f"## {representation['name']}",
                "",
                "| steps | paired RMSE | corrected mean RMSE | seed RMS | energy score | 95% context CI |",
                "| ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in representation["metrics"]:
            lines.append(
                f"| {row['steps']} | {row['paired_rmse']:.6f} | "
                f"{row['monte_carlo_corrected_mean_rmse']:.6f} | "
                f"{row['within_seed_rms']:.6f} | "
                f"{row['energy_score']:.6f} | "
                f"[{row['energy_score_ci95_low']:.6f}, "
                f"{row['energy_score_ci95_high']:.6f}] |"
            )
        comparison = representation["comparison"]
        lines.extend(
            [
                "",
                f"Paired 10-minus-1-step energy difference: "
                f"`{comparison['energy_score_gap']:+.6f}` "
                f"(95% paired-context CI "
                f"`[{comparison['energy_score_gap_ci95_low']:+.6f}, "
                f"{comparison['energy_score_gap_ci95_high']:+.6f}]`).",
                "",
            ]
        )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    step_counts = [int(value) for value in args.steps.split(",")]
    if len(step_counts) != 2 or step_counts != sorted(set(step_counts)):
        raise ValueError("exactly two unique increasing step counts are required")
    if args.pairs <= 0 or args.seeds < 2 or args.bootstrap_samples <= 0:
        raise ValueError("pairs, seeds, and bootstrap samples must be positive")
    if args.chunk <= 0:
        raise ValueError("chunk must be a continuation chunk")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest_path = args.prepared_dir / "prepared_context_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if args.pairs > int(manifest["pair_count"]):
        raise ValueError("requested pairs exceed the prepared manifest")
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    device = torch.device(args.device)
    representation_names = ("latent", "fixed_coarse_motion")
    context_rows: dict[tuple[str, int], list[dict[str, Any]]] = {
        (name, steps): []
        for name in representation_names
        for steps in step_counts
    }

    for pair_index in range(args.pairs):
        saved = torch.load(
            args.prepared_dir / f"prepared_pair_{pair_index:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        payload = to_device(saved["tensors"], device, architecture.dtype)
        metadata = saved["samples"]
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
        target = video_chunks[args.chunk].float()

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
            for representation in representation_names:
                prediction_vectors, target_vectors = representation_vectors(
                    stacked, target, representation
                )
                for context_index, item in enumerate(metadata):
                    row = {
                        "pair_index": pair_index,
                        "context_index": context_index,
                        "episode_index": int(item["episode_index"]),
                        "start_frame": int(item["start_frame"]),
                        "steps": steps,
                        **context_score(
                            prediction_vectors[:, context_index],
                            target_vectors[context_index],
                        ),
                    }
                    context_rows[(representation, steps)].append(row)
        print(f"pair {pair_index + 1}/{args.pairs} complete", flush=True)

    representations = []
    for representation_index, representation in enumerate(representation_names):
        metrics = []
        for steps in step_counts:
            rows = context_rows[(representation, steps)]
            metric = {
                "steps": steps,
                **aggregate_contexts(
                    rows,
                    bootstrap_seed=args.seed
                    + representation_index * 100_003
                    + steps,
                    bootstrap_samples=args.bootstrap_samples,
                ),
            }
            metrics.append(metric)
        first_rows = context_rows[(representation, step_counts[0])]
        second_rows = context_rows[(representation, step_counts[1])]
        first_by_episode = {row["episode_index"]: row for row in first_rows}
        second_by_episode = {row["episode_index"]: row for row in second_rows}
        if first_by_episode.keys() != second_by_episode.keys():
            raise RuntimeError("step comparisons have different context identities")
        gaps = [
            second_by_episode[episode]["energy_score"]
            - first_by_episode[episode]["energy_score"]
            for episode in sorted(first_by_episode)
        ]
        gap_low, gap_high = bootstrap_mean_ci(
            gaps,
            seed=args.seed + representation_index * 1_000_003 + 71,
            samples=args.bootstrap_samples,
        )
        representations.append(
            {
                "name": representation,
                "metrics": metrics,
                "comparison": {
                    "from_steps": step_counts[0],
                    "to_steps": step_counts[1],
                    "energy_score_gap": float(np.mean(gaps)),
                    "energy_score_gap_ci95_low": gap_low,
                    "energy_score_gap_ci95_high": gap_high,
                },
                "context_metrics": {
                    str(steps): context_rows[(representation, steps)]
                    for steps in step_counts
                },
            }
        )

    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "pair_count": args.pairs,
        "context_count": args.pairs * 2,
        "seed_count": args.seeds,
        "chunk": args.chunk,
        "step_counts": step_counts,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "representations": representations,
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "video_distribution_score.json"
    markdown_path = args.output_dir / "video_distribution_score.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
