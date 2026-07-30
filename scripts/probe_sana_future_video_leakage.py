#!/usr/bin/env python
"""Measure whether current action predictions depend on future video chunks."""

from __future__ import annotations

import argparse
import json
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

from probe_sana_action_denoising import (  # noqa: E402
    parse_sigmas,
    predict_velocity,
    to_device,
)
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
    parser.add_argument("--video-sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


@dataclass
class DeltaSums:
    count: int = 0
    square_sum: float = 0.0
    abs_max: float = 0.0

    def add(self, value: torch.Tensor, mask: torch.Tensor) -> None:
        selected = value[mask]
        if not selected.numel():
            return
        self.count += selected.numel()
        self.square_sum += float(selected.double().square().sum().item())
        self.abs_max = max(self.abs_max, float(selected.abs().max().item()))

    def final(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "rms": (self.square_sum / self.count) ** 0.5 if self.count else 0.0,
            "abs_max": self.abs_max,
        }


@dataclass
class ErrorSums:
    count: int = 0
    square_sum: float = 0.0

    def add(self, value: torch.Tensor, mask: torch.Tensor) -> None:
        selected = value[mask]
        if not selected.numel():
            return
        self.count += selected.numel()
        self.square_sum += float(selected.double().square().sum().item())

    def final(self) -> float:
        return (self.square_sum / self.count) ** 0.5


def replace_video_chunks(
    payload: dict[str, torch.Tensor],
    *,
    frame_start: int,
    frame_end: int,
) -> dict[str, torch.Tensor]:
    result = dict(payload)
    latents = payload["input_latents"].clone()
    permuted = payload["input_latents"].roll(shifts=1, dims=0)
    latents[:, :, frame_start:frame_end] = permuted[:, :, frame_start:frame_end]
    result["input_latents"] = latents
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Future-video leakage into current action predictions",
        "",
        "Future video chunks are permuted across the two samples while current "
        "and past video, all actions, prompt, proprioception, and diffusion noise "
        "remain fixed.",
        "",
        "| Layout | sigma | action chunk | future-video delta RMS | max abs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["future_video_sensitivity"]:
        lines.append(
            f"| {row['layout']} | {row['sigma']:.3f} | {row['chunk']} | "
            f"{row['rms']:.9f} | {row['abs_max']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Current-video positive control",
            "",
            "| Layout | sigma | action chunk | current-video delta RMS | max abs |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["current_video_positive_control"]:
        lines.append(
            f"| {row['layout']} | {row['sigma']:.3f} | {row['chunk']} | "
            f"{row['rms']:.9f} | {row['abs_max']:.9f} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sigmas = parse_sigmas(args.sigmas)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    device = torch.device(args.device)
    frame_chunk_size = int(architecture._ar_frame_chunk_size)
    future_sums: dict[tuple[str, float, int], DeltaSums] = defaultdict(DeltaSums)
    current_sums: dict[tuple[str, float, int], DeltaSums] = defaultdict(DeltaSums)
    error_sums: dict[tuple[str, float], ErrorSums] = defaultdict(ErrorSums)

    layouts = (("legacy_full", None), ("chunkwise", frame_chunk_size))
    for layout, temporal_chunk_frames in layouts:
        architecture.video_backbone._ar_temporal_chunk_frames = (
            temporal_chunk_frames
        )
        architecture._ar_chunkwise_temporal_ops = temporal_chunk_frames is not None
        for pair_index in range(args.pairs):
            saved = torch.load(
                args.prepared_dir / f"prepared_pair_{pair_index:02d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            payload = to_device(saved["tensors"], device, architecture.dtype)
            actions = payload["actions"]
            latent_frames = payload["input_latents"].shape[2]
            chunks = latent_frames // frame_chunk_size
            tokens_per_chunk = actions.shape[1] // chunks
            chunk_indices = [chunk * tokens_per_chunk for chunk in range(chunks)]
            proprio_per_chunk = payload["proprio_seq"][:, chunk_indices]
            generator = torch.Generator(device=device).manual_seed(
                args.seed + pair_index * 1009
            )
            action_noise = torch.randn(
                actions.shape,
                generator=generator,
                device=device,
                dtype=architecture.dtype,
            )
            video_noise = torch.randn(
                payload["input_latents"].shape,
                generator=generator,
                device=device,
                dtype=architecture.dtype,
            )
            valid = torch.ones_like(actions, dtype=torch.bool)
            if "action_is_pad" in payload:
                valid &= (~payload["action_is_pad"]).unsqueeze(-1)

            for sigma in sigmas:
                baseline, target, _ = predict_velocity(
                    architecture,
                    payload,
                    condition="correct",
                    sigma=sigma,
                    video_sigma=args.video_sigma,
                    action_noise=action_noise,
                    video_noise=video_noise,
                    proprio_per_chunk=proprio_per_chunk,
                )
                error_sums[(layout, sigma)].add(baseline - target, valid)
                for chunk in range(chunks - 1):
                    future_start = (chunk + 1) * frame_chunk_size
                    future_payload = replace_video_chunks(
                        payload,
                        frame_start=future_start,
                        frame_end=latent_frames,
                    )
                    future_prediction, future_target, _ = predict_velocity(
                        architecture,
                        future_payload,
                        condition="correct",
                        sigma=sigma,
                        video_sigma=args.video_sigma,
                        action_noise=action_noise,
                        video_noise=video_noise,
                        proprio_per_chunk=proprio_per_chunk,
                    )
                    if not torch.equal(target, future_target):
                        raise RuntimeError("future-video intervention changed target")
                    start = chunk * tokens_per_chunk
                    end = start + tokens_per_chunk
                    future_sums[(layout, sigma, chunk)].add(
                        future_prediction[:, start:end] - baseline[:, start:end],
                        valid[:, start:end],
                    )

                current_payload = replace_video_chunks(
                    payload,
                    frame_start=0,
                    frame_end=frame_chunk_size,
                )
                current_prediction, current_target, _ = predict_velocity(
                    architecture,
                    current_payload,
                    condition="correct",
                    sigma=sigma,
                    video_sigma=args.video_sigma,
                    action_noise=action_noise,
                    video_noise=video_noise,
                    proprio_per_chunk=proprio_per_chunk,
                )
                if not torch.equal(target, current_target):
                    raise RuntimeError("current-video intervention changed target")
                current_sums[(layout, sigma, 0)].add(
                    current_prediction[:, :tokens_per_chunk]
                    - baseline[:, :tokens_per_chunk],
                    valid[:, :tokens_per_chunk],
                )
            del payload, action_noise, video_noise
        print(f"layout={layout} complete", flush=True)

    future_metrics = [
        {"layout": layout, "sigma": sigma, "chunk": chunk, **sums.final()}
        for (layout, sigma, chunk), sums in sorted(future_sums.items())
    ]
    current_metrics = [
        {"layout": layout, "sigma": sigma, "chunk": chunk, **sums.final()}
        for (layout, sigma, chunk), sums in sorted(current_sums.items())
    ]
    baseline_metrics = [
        {"layout": layout, "sigma": sigma, "velocity_rmse": sums.final()}
        for (layout, sigma), sums in sorted(error_sums.items())
    ]
    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "pair_count": args.pairs,
        "sigmas": sigmas,
        "video_sigma": args.video_sigma,
        "seed": args.seed,
        "intervention": {
            "future_video_permuted_across_batch": True,
            "actions_prompt_proprio_and_noise_fixed": True,
            "current_video_positive_control_chunk": 0,
        },
        "baseline_action_metrics": baseline_metrics,
        "future_video_sensitivity": future_metrics,
        "current_video_positive_control": current_metrics,
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "future_video_leakage.json"
    markdown_path = args.output_dir / "future_video_leakage.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
