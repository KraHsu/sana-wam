#!/usr/bin/env python
"""Decompose teacher-forced versus predicted-video SANA AR action error offline."""

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
from sana_wam.model.ar.sana_ar_inference import ARLinearStateCache  # noqa: E402


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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PARENT / "ar_decomposition")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--sigmas", default="1.0,0.9")
    parser.add_argument("--video-steps", type=int, default=10)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--video-sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


@torch.inference_mode()
def ingest_clean_action(
    architecture,
    driver,
    cache,
    action: torch.Tensor,
    *,
    frame_id: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_proprio: torch.Tensor | None,
) -> None:
    batch, tokens, _ = action.shape
    frame_ids = torch.full(
        (tokens,), frame_id, dtype=torch.long, device=action.device
    )
    extra = (
        {}
        if action_proprio is None
        else {"token_proprio_emb": action_proprio[:, None, :]}
    )
    state = architecture.action_backbone.prepare_state(
        action,
        torch.zeros(batch, device=action.device, dtype=action.dtype),
        context=context,
        context_mask=context_mask,
        token_timesteps=torch.zeros(
            batch, tokens, device=action.device, dtype=action.dtype
        ),
        frame_ids=frame_ids,
        rope_positions=frame_ids,
        **extra,
    )
    driver.run_ar_chunk_through_backbone(
        architecture.action_backbone,
        state,
        cache,
        frame_id,
        store_clean=True,
        is_pred=False,
    )


@torch.inference_mode()
def query_action(
    architecture,
    driver,
    cache,
    noisy_action: torch.Tensor,
    *,
    sigma: float,
    frame_id: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_proprio: torch.Tensor | None,
) -> torch.Tensor:
    batch, tokens, _ = noisy_action.shape
    frame_ids = torch.full(
        (tokens,), frame_id, dtype=torch.long, device=noisy_action.device
    )
    extra = (
        {}
        if action_proprio is None
        else {"token_proprio_emb": action_proprio[:, None, :]}
    )
    token_timesteps = torch.full(
        (batch, tokens),
        sigma * 1000.0,
        device=noisy_action.device,
        dtype=noisy_action.dtype,
    )
    state = architecture.action_backbone.prepare_state(
        noisy_action,
        torch.full(
            (batch,),
            sigma * 1000.0,
            device=noisy_action.device,
            dtype=noisy_action.dtype,
        ),
        context=context,
        context_mask=context_mask,
        token_timesteps=token_timesteps,
        frame_ids=frame_ids,
        rope_positions=frame_ids,
        **extra,
    )
    driver.run_ar_chunk_through_backbone(
        architecture.action_backbone,
        state,
        cache,
        frame_id,
        store_clean=False,
    )
    return architecture.action_backbone.extract_prediction(state).float()


def build_expert_history(
    architecture,
    driver,
    video_chunks: list[torch.Tensor],
    action_chunks: list[torch.Tensor],
    proprio_chunks: list[torch.Tensor],
    *,
    stop_chunk: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
) -> ARLinearStateCache:
    cache = ARLinearStateCache(
        num_layers=architecture.video_backbone.num_layers,
        window=int(architecture._ar_attn_window),
    )
    for chunk in range(stop_chunk):
        video_proprio, action_proprio = architecture._rollout_proprio_deltas(
            proprio_chunks[chunk]
        )
        architecture._ingest_clean_video(
            driver,
            cache,
            video_chunks[chunk],
            frame_id=2 * chunk,
            context=context,
            context_mask=context_mask,
            v_proprio=video_proprio,
        )
        ingest_clean_action(
            architecture,
            driver,
            cache,
            action_chunks[chunk],
            frame_id=2 * chunk + 1,
            context=context,
            context_mask=context_mask,
            action_proprio=action_proprio,
        )
    return cache


def build_predicted_action_history(
    architecture,
    driver,
    video_chunks: list[torch.Tensor],
    proprio_chunks: list[torch.Tensor],
    *,
    stop_chunk: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_tokens: int,
    action_sigmas: list[float],
    action_timesteps: torch.Tensor,
    generator: torch.Generator,
) -> ARLinearStateCache:
    cache = ARLinearStateCache(
        num_layers=architecture.video_backbone.num_layers,
        window=int(architecture._ar_attn_window),
    )
    for chunk in range(stop_chunk):
        video_proprio, action_proprio = architecture._rollout_proprio_deltas(
            proprio_chunks[chunk]
        )
        architecture._ingest_clean_video(
            driver,
            cache,
            video_chunks[chunk],
            frame_id=2 * chunk,
            context=context,
            context_mask=context_mask,
            v_proprio=video_proprio,
        )
        architecture._denoise_action_chunk(
            driver,
            cache,
            frame_id=2 * chunk + 1,
            batch=video_chunks[chunk].shape[0],
            action_tokens=action_tokens,
            a_sigmas=action_sigmas,
            a_ts=action_timesteps,
            context=context,
            context_mask=context_mask,
            gen=generator,
            a_proprio=action_proprio,
        )
    return cache


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.double().flatten()
    right_flat = right.double().flatten()
    return float(
        torch.dot(left_flat, right_flat)
        / (torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)).clamp_min(1e-30)
    )


class MetricSums:
    def __init__(self) -> None:
        self.values: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add_rms(self, name: str, value: torch.Tensor) -> None:
        self.values[name] += float(value.double().square().sum().item())
        self.counts[name] += value.numel()

    def add_scalar(self, name: str, value: float) -> None:
        self.values[name] += value
        self.counts[name] += 1

    def final(self) -> dict[str, float]:
        result = {}
        for name, total in self.values.items():
            if name.endswith("_cosine"):
                result[name] = total / self.counts[name]
            else:
                result[name] = (total / self.counts[name]) ** 0.5
        return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Teacher-forced versus predicted-video AR decomposition",
        "",
        "Each predicted-video comparison starts from identical expert video/action cache history. The only changed input to the fixed-sigma action query is the current chunk: encoded expert latent versus SANA's 10-step predicted latent.",
        "",
        "| sigma | chunk | full-train RMSE | cache-teacher RMSE | predicted-video RMSE | video delta | AR-action-history RMSE | history delta | video sensitivity | history sensitivity | latent RMSE | latent cosine |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["metrics"]:
        lines.append(
            f"| {row['sigma']:.3f} | {row['chunk']} | "
            f"{row['full_train_velocity_rmse']:.6f} | "
            f"{row['cache_teacher_velocity_rmse']:.6f} | "
            f"{row['predicted_video_velocity_rmse']:.6f} | "
            f"{row['predicted_video_velocity_rmse'] - row['cache_teacher_velocity_rmse']:+.6f} | "
            f"{row['autoregressive_action_history_velocity_rmse']:.6f} | "
            f"{row['autoregressive_action_history_velocity_rmse'] - row['cache_teacher_velocity_rmse']:+.6f} | "
            f"{row['action_prediction_sensitivity_rms']:.6f} | "
            f"{row['action_history_sensitivity_rms']:.6f} | "
            f"{row['predicted_latent_rmse']:.6f} | {row['predicted_latent_cosine']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Full duplicated-forward versus cache teacher equivalence max delta: `{report['cache_equivalence_max_abs_delta']:.6g}`.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sigmas = parse_sigmas(args.sigmas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    device = torch.device(args.device)
    architecture.video_backbone.scheduler.set_timesteps(args.video_steps)
    video_sigmas = [
        float(value) for value in architecture.video_backbone.scheduler.sigmas.tolist()
    ] + [0.0]
    video_timesteps = architecture.video_backbone.scheduler.timesteps
    architecture.action_backbone.scheduler.set_timesteps(args.action_steps)
    action_sigmas = [
        float(value) for value in architecture.action_backbone.scheduler.sigmas.tolist()
    ] + [0.0]
    action_timesteps = architecture.action_backbone.scheduler.timesteps
    sums: dict[tuple[float, int], MetricSums] = defaultdict(MetricSums)
    max_cache_delta = 0.0

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
        action_noise = torch.randn(
            actions.shape,
            generator=generator,
            device=device,
            dtype=architecture.dtype,
        )
        video_noise = torch.randn(
            latents.shape,
            generator=generator,
            device=device,
            dtype=architecture.dtype,
        )

        full_predictions = {}
        for sigma in sigmas:
            prediction, target, _ = predict_velocity(
                architecture,
                payload,
                condition="correct",
                sigma=sigma,
                video_sigma=args.video_sigma,
                action_noise=action_noise,
                video_noise=video_noise,
                proprio_per_chunk=torch.stack(proprio_chunks, dim=1),
            )
            full_predictions[sigma] = (prediction, target)

        for chunk in range(1, chunks):
            video_proprio, action_proprio = architecture._rollout_proprio_deltas(
                proprio_chunks[chunk]
            )
            teacher_cache = build_expert_history(
                architecture,
                driver,
                video_chunks,
                action_chunks,
                proprio_chunks,
                stop_chunk=chunk,
                context=context,
                context_mask=context_mask,
            )
            architecture._ingest_clean_video(
                driver,
                teacher_cache,
                video_chunks[chunk],
                frame_id=2 * chunk,
                context=context,
                context_mask=context_mask,
                v_proprio=video_proprio,
            )

            predicted_cache = build_expert_history(
                architecture,
                driver,
                video_chunks,
                action_chunks,
                proprio_chunks,
                stop_chunk=chunk,
                context=context,
                context_mask=context_mask,
            )
            video_generator = torch.Generator(device=device).manual_seed(
                args.seed + pair_index * 1009 + chunk * 100_003
            )
            predicted_video = architecture._denoise_video_chunk(
                driver,
                predicted_cache,
                frame_id=2 * chunk,
                like=video_chunks[chunk],
                v_sigmas=video_sigmas,
                v_ts=video_timesteps,
                context=context,
                context_mask=context_mask,
                gen=video_generator,
                v_proprio=video_proprio,
            )

            action_history_generator = torch.Generator(device=device).manual_seed(
                args.seed + pair_index * 1009 + chunk * 200_003
            )
            predicted_action_cache = build_predicted_action_history(
                architecture,
                driver,
                video_chunks,
                proprio_chunks,
                stop_chunk=chunk,
                context=context,
                context_mask=context_mask,
                action_tokens=tokens_per_chunk,
                action_sigmas=action_sigmas,
                action_timesteps=action_timesteps,
                generator=action_history_generator,
            )
            architecture._ingest_clean_video(
                driver,
                predicted_action_cache,
                video_chunks[chunk],
                frame_id=2 * chunk,
                context=context,
                context_mask=context_mask,
                v_proprio=video_proprio,
            )

            latent_error = predicted_video.float() - video_chunks[chunk].float()
            for sigma in sigmas:
                start = chunk * tokens_per_chunk
                end = start + tokens_per_chunk
                noise_chunk = action_noise[:, start:end]
                action_chunk = action_chunks[chunk]
                noisy_action = (1.0 - sigma) * action_chunk + sigma * noise_chunk
                teacher_prediction = query_action(
                    architecture,
                    driver,
                    teacher_cache,
                    noisy_action,
                    sigma=sigma,
                    frame_id=2 * chunk + 1,
                    context=context,
                    context_mask=context_mask,
                    action_proprio=action_proprio,
                )
                predicted_video_prediction = query_action(
                    architecture,
                    driver,
                    predicted_cache,
                    noisy_action,
                    sigma=sigma,
                    frame_id=2 * chunk + 1,
                    context=context,
                    context_mask=context_mask,
                    action_proprio=action_proprio,
                )
                predicted_action_history_prediction = query_action(
                    architecture,
                    driver,
                    predicted_action_cache,
                    noisy_action,
                    sigma=sigma,
                    frame_id=2 * chunk + 1,
                    context=context,
                    context_mask=context_mask,
                    action_proprio=action_proprio,
                )
                full_prediction, full_target = full_predictions[sigma]
                full_chunk = full_prediction[:, start:end]
                target_chunk = full_target[:, start:end]
                max_cache_delta = max(
                    max_cache_delta,
                    float((teacher_prediction - full_chunk).abs().max().item()),
                )
                metric = sums[(sigma, chunk)]
                metric.add_rms(
                    "full_train_velocity_rmse", full_chunk - target_chunk
                )
                metric.add_rms(
                    "cache_teacher_velocity_rmse",
                    teacher_prediction - target_chunk,
                )
                metric.add_rms(
                    "predicted_video_velocity_rmse",
                    predicted_video_prediction - target_chunk,
                )
                metric.add_rms(
                    "autoregressive_action_history_velocity_rmse",
                    predicted_action_history_prediction - target_chunk,
                )
                metric.add_rms(
                    "action_prediction_sensitivity_rms",
                    predicted_video_prediction - teacher_prediction,
                )
                metric.add_rms(
                    "action_history_sensitivity_rms",
                    predicted_action_history_prediction - teacher_prediction,
                )
                metric.add_rms("predicted_latent_rmse", latent_error)
                metric.add_scalar(
                    "predicted_latent_cosine",
                    cosine(predicted_video, video_chunks[chunk]),
                )
            print(f"pair={pair_index} chunk={chunk} complete", flush=True)
        del payload, action_noise, video_noise
        torch.cuda.empty_cache()

    metrics = [
        {"sigma": sigma, "chunk": chunk, **value.final()}
        for (sigma, chunk), value in sorted(sums.items())
    ]
    report = {
        "checkpoint": str(args.run_dir / args.checkpoint),
        "pair_count": args.pairs,
        "sigmas": sigmas,
        "video_steps": args.video_steps,
        "action_steps": args.action_steps,
        "expert_history_for_each_comparison": True,
        "oracle_current_chunk_proprio_for_video_prediction": True,
        "cache_equivalence_max_abs_delta": max_cache_delta,
        "total_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    json_path = args.output_dir / "ar_decomposition.json"
    markdown_path = args.output_dir / "ar_decomposition.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
