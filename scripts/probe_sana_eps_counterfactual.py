#!/usr/bin/env python
"""Test whether SANA's near-zero AR denominator is causally tied to action error."""

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

from probe_sana_action_denoising import parse_sigmas, predict_velocity, to_device  # noqa: E402
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


DEFAULT_RUN = Path(
    "/home/zch/wuji-openwam-dev/sandbox/"
    "sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25"
)
DEFAULT_PARENT = ROOT / "logs" / "sana_principles_audit_20260722"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", default="checkpoint_step_12000.safetensors")
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PARENT / "action_sigma_probe")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PARENT / "eps_counterfactual")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--sigmas", default="1.0,0.9,0.5")
    parser.add_argument("--eps-values", default="1e-15,1e-12,1e-9,1e-6,1e-4,1e-2,1e0")
    parser.add_argument("--video-sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


class ErrorSums:
    def __init__(self) -> None:
        self.squared_error = 0.0
        self.count = 0

    def add(self, error: torch.Tensor) -> None:
        self.squared_error += float(error.double().square().sum().item())
        self.count += error.numel()

    def rmse(self) -> float:
        return (self.squared_error / self.count) ** 0.5


def markdown(report: dict[str, Any]) -> str:
    rows = report["metrics"]

    def find(eps: float, sigma: float, scope: str) -> float:
        return next(
            row["velocity_rmse"]
            for row in rows
            if row["eps"] == eps and row["sigma"] == sigma and row["scope"] == scope
        )

    lines = [
        "# AR denominator-epsilon counterfactual",
        "",
        "This is an offline numerical intervention on the unchanged checkpoint, not a success-rate trick: only the denominator additive epsilon in the production SANA AR kernel changes.",
        "",
        "| eps | sigma | all velocity RMSE | bootstrap | continuation | delta vs checkpoint |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline_eps = report["eps_values"][0]
    for eps in report["eps_values"]:
        for sigma in report["sigmas"]:
            all_error = find(eps, sigma, "all")
            baseline = find(baseline_eps, sigma, "all")
            lines.append(
                f"| {eps:.1e} | {sigma:.3f} | {all_error:.6f} | "
                f"{find(eps, sigma, 'bootstrap'):.6f} | "
                f"{find(eps, sigma, 'continuation'):.6f} | {all_error - baseline:+.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sigmas = parse_sigmas(args.sigmas)
    eps_values = [float(value) for value in args.eps_values.split(",")]
    if not eps_values or any(value <= 0 for value in eps_values):
        raise ValueError(f"eps values must be positive, got {eps_values}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    checkpoint_eps = float(driver.eps)
    if checkpoint_eps != eps_values[0]:
        raise RuntimeError(
            f"first eps must be checkpoint value {checkpoint_eps}, got {eps_values[0]}"
        )
    device = torch.device(args.device)
    sums: dict[tuple[float, float, str], ErrorSums] = defaultdict(ErrorSums)

    for pair_index in range(args.pairs):
        prepared_path = args.prepared_dir / f"prepared_pair_{pair_index:02d}.pt"
        saved = torch.load(prepared_path, map_location="cpu", weights_only=False)
        payload = to_device(saved["tensors"], device, architecture.dtype)
        actions = payload["actions"]
        chunks = payload["input_latents"].shape[2] // int(
            architecture._ar_frame_chunk_size
        )
        tokens_per_chunk = actions.shape[1] // chunks
        proprio_per_chunk = payload["proprio_seq"][
            :, [chunk * tokens_per_chunk for chunk in range(chunks)]
        ]
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
        for eps in eps_values:
            driver.eps = eps
            for sigma in sigmas:
                prediction, target, _ = predict_velocity(
                    architecture,
                    payload,
                    condition="correct",
                    sigma=sigma,
                    video_sigma=args.video_sigma,
                    action_noise=action_noise,
                    video_noise=video_noise,
                    proprio_per_chunk=proprio_per_chunk,
                )
                error = prediction - target
                sums[(eps, sigma, "all")].add(error)
                sums[(eps, sigma, "bootstrap")].add(
                    error[:, :tokens_per_chunk]
                )
                sums[(eps, sigma, "continuation")].add(
                    error[:, tokens_per_chunk:]
                )
            print(f"pair={pair_index} eps={eps:.1e} complete", flush=True)
        del payload, action_noise, video_noise
        torch.cuda.empty_cache()
    driver.eps = checkpoint_eps

    metrics = [
        {
            "eps": eps,
            "sigma": sigma,
            "scope": scope,
            "count": value.count,
            "velocity_rmse": value.rmse(),
        }
        for (eps, sigma, scope), value in sorted(sums.items())
    ]
    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "pair_count": args.pairs,
        "sigmas": sigmas,
        "eps_values": eps_values,
        "video_sigma": args.video_sigma,
        "fixed_noise_across_eps": True,
        "total_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    json_path = args.output_dir / "eps_counterfactual.json"
    markdown_path = args.output_dir / "eps_counterfactual.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
