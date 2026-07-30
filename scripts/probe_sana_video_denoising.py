#!/usr/bin/env python
"""Measure SANA's cache-conditioned video vector field at fixed sigmas."""

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

from probe_sana_action_denoising import parse_sigmas, to_device  # noqa: E402
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
    parser.add_argument("--sigmas", default="1.0,0.9,0.5")
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


class ErrorSums:
    def __init__(self) -> None:
        self.square_sum = 0.0
        self.count = 0

    def add(self, value: torch.Tensor) -> None:
        self.square_sum += float(value.double().square().sum().item())
        self.count += value.numel()

    def rms(self) -> float:
        return (self.square_sum / self.count) ** 0.5


def scheduler_point(scheduler, requested_sigma: float) -> tuple[float, float]:
    sigmas = scheduler.sigmas.float()
    index = int((sigmas - requested_sigma).abs().argmin().item())
    return float(sigmas[index].item()), float(scheduler.timesteps[index].item())


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fixed-sigma cache-conditioned video denoising",
        "",
        "Each current video chunk is noised from the expert latent and queried "
        "against identical expert video/action cache history. Errors are measured "
        "against the flow target `noise - clean_latent`.",
        "",
        "| requested sigma | actual sigma | chunk | velocity RMSE | reconstructed x0 RMSE |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["metrics"]:
        lines.append(
            f"| {row['requested_sigma']:.3f} | {row['actual_sigma']:.6f} | "
            f"{row['chunk']} | {row['velocity_rmse']:.6f} | "
            f"{row['x0_rmse']:.6f} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    requested_sigmas = parse_sigmas(args.sigmas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    device = torch.device(args.device)
    scheduler = architecture.video_backbone.scheduler
    scheduler.set_timesteps(scheduler.num_train_timesteps, training=True)
    schedule = {
        requested: scheduler_point(scheduler, requested)
        for requested in requested_sigmas
    }
    sums: dict[tuple[float, int, str], ErrorSums] = defaultdict(ErrorSums)

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
        generator = torch.Generator(device=device).manual_seed(
            args.seed + pair_index * 1009
        )
        noise_chunks = list(
            torch.randn(
                latents.shape,
                generator=generator,
                device=device,
                dtype=architecture.dtype,
            ).split(frame_chunk_size, dim=2)
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
            clean = video_chunks[chunk]
            noise = noise_chunks[chunk]
            batch, n_frames = clean.shape[0], clean.shape[2]
            start_frame = chunk * n_frames
            rope_index = torch.arange(
                start_frame, start_frame + n_frames, device=device
            )
            frame_proprio = (
                None
                if video_proprio is None
                else video_proprio[:, None, :].expand(batch, n_frames, -1)
            )

            for requested_sigma in requested_sigmas:
                sigma, timestep = schedule[requested_sigma]
                noisy = (1.0 - sigma) * clean + sigma * noise
                extra = (
                    {}
                    if frame_proprio is None
                    else {"frame_proprio_emb": frame_proprio}
                )
                state = architecture.video_backbone.prepare(
                    latents=noisy,
                    timestep=torch.full(
                        (batch,), timestep, device=device, dtype=architecture.dtype
                    ),
                    frame_timesteps=torch.full(
                        (batch, n_frames),
                        timestep,
                        device=device,
                        dtype=architecture.dtype,
                    ),
                    rope_frame_index=rope_index,
                    context=context,
                    context_mask=context_mask,
                    **extra,
                )
                driver.run_ar_chunk_through_backbone(
                    architecture.video_backbone,
                    state,
                    cache,
                    frame_id=2 * chunk,
                    store_clean=False,
                )
                prediction = architecture.video_backbone.finalize(state)
                target = noise - clean
                velocity_error = prediction.float() - target.float()
                x0_hat = noisy.float() - sigma * prediction.float()
                x0_error = x0_hat - clean.float()
                sums[(requested_sigma, chunk, "velocity")].add(velocity_error)
                sums[(requested_sigma, chunk, "x0")].add(x0_error)

    metrics = []
    for requested_sigma in requested_sigmas:
        actual_sigma, timestep = schedule[requested_sigma]
        chunks = sorted(
            chunk
            for sigma, chunk, kind in sums
            if sigma == requested_sigma and kind == "velocity"
        )
        for chunk in chunks:
            metrics.append(
                {
                    "requested_sigma": requested_sigma,
                    "actual_sigma": actual_sigma,
                    "timestep": timestep,
                    "chunk": chunk,
                    "velocity_rmse": sums[
                        (requested_sigma, chunk, "velocity")
                    ].rms(),
                    "x0_rmse": sums[(requested_sigma, chunk, "x0")].rms(),
                }
            )

    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "pair_count": args.pairs,
        "requested_sigmas": requested_sigmas,
        "metrics": metrics,
        "total_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "video_sigma_probe.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output_dir / "video_sigma_probe.md").write_text(markdown(report))
    print(json.dumps({"output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
