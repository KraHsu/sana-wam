#!/usr/bin/env python
"""Profile endpoint error and drift along an actual video Euler trajectory."""

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
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


@dataclass
class Sums:
    count: int = 0
    square_sum: float = 0.0

    def add(self, value: torch.Tensor) -> None:
        self.count += value.numel()
        self.square_sum += float(value.double().square().sum().item())

    def rms(self) -> float:
        return (self.square_sum / self.count) ** 0.5


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Video Euler trajectory endpoint profile",
        "",
        "At every Euler query, `x0_hat = x_sigma - sigma * v_theta` is compared "
        "with the exact expert latent endpoint.",
        "",
        "| step | sigma | next sigma | state RMSE | endpoint RMSE | endpoint drift | next-state RMSE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["metrics"]:
        drift = row.get("endpoint_drift_rms")
        drift_text = "-" if drift is None else f"{drift:.6f}"
        lines.append(
            f"| {row['step']} | {row['sigma']:.6f} | "
            f"{row['next_sigma']:.6f} | {row['state_rmse']:.6f} | "
            f"{row['endpoint_rmse']:.6f} | {drift_text} | "
            f"{row['next_state_rmse']:.6f} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
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
    architecture.video_backbone.scheduler.set_timesteps(args.steps)
    sigmas = [
        float(value)
        for value in architecture.video_backbone.scheduler.sigmas.tolist()
    ] + [0.0]
    timesteps = architecture.video_backbone.scheduler.timesteps
    accumulators = [
        {
            "state": Sums(),
            "endpoint": Sums(),
            "drift": Sums(),
            "next_state": Sums(),
        }
        for _ in range(args.steps)
    ]

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
        clean = video_chunks[args.chunk]
        video_proprio, _ = architecture._rollout_proprio_deltas(
            proprio_chunks[args.chunk]
        )
        batch, n_frames = clean.shape[0], clean.shape[2]
        start_frame = args.chunk * n_frames
        rope_index = torch.arange(
            start_frame, start_frame + n_frames, device=device
        )
        frame_proprio = (
            None
            if video_proprio is None
            else video_proprio[:, None, :].expand(batch, n_frames, -1)
        )
        extra = (
            {}
            if frame_proprio is None
            else {"frame_proprio_emb": frame_proprio}
        )
        generator = torch.Generator(device=device).manual_seed(
            args.seed + pair_index * 1009 + args.chunk * 100_003
        )
        state = torch.randn(
            clean.shape,
            generator=generator,
            device=device,
            dtype=architecture.dtype,
        )
        previous_endpoint = None
        for index in range(args.steps):
            sigma = sigmas[index]
            next_sigma = sigmas[index + 1]
            timestep = float(timesteps[index])
            frame_timesteps = torch.full(
                (batch, n_frames),
                timestep,
                device=device,
                dtype=architecture.dtype,
            )
            video_state = architecture.video_backbone.prepare(
                latents=state,
                timestep=torch.full(
                    (batch,), timestep, device=device, dtype=architecture.dtype
                ),
                frame_timesteps=frame_timesteps,
                rope_frame_index=rope_index,
                context=context,
                context_mask=context_mask,
                **extra,
            )
            driver.run_ar_chunk_through_backbone(
                architecture.video_backbone,
                video_state,
                cache,
                frame_id=2 * args.chunk,
                store_clean=False,
            )
            velocity = architecture.video_backbone.finalize(video_state)
            endpoint = state.float() - sigma * velocity.float()
            next_state = state + velocity * (next_sigma - sigma)
            sums = accumulators[index]
            sums["state"].add(state.float() - clean.float())
            sums["endpoint"].add(endpoint - clean.float())
            if previous_endpoint is not None:
                sums["drift"].add(endpoint - previous_endpoint)
            sums["next_state"].add(next_state.float() - clean.float())
            previous_endpoint = endpoint
            state = next_state

    metrics = []
    for index, sums in enumerate(accumulators):
        metrics.append(
            {
                "step": index,
                "sigma": sigmas[index],
                "next_sigma": sigmas[index + 1],
                "timestep": float(timesteps[index]),
                "state_rmse": sums["state"].rms(),
                "endpoint_rmse": sums["endpoint"].rms(),
                "endpoint_drift_rms": (
                    None if index == 0 else sums["drift"].rms()
                ),
                "next_state_rmse": sums["next_state"].rms(),
            }
        )
    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "pair_count": args.pairs,
        "chunk": args.chunk,
        "steps": args.steps,
        "seed": args.seed,
        "metrics": metrics,
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "video_trajectory_profile.json"
    markdown_path = args.output_dir / "video_trajectory_profile.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
