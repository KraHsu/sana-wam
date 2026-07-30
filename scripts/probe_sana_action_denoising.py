#!/usr/bin/env python
"""Measure fixed-sigma SANA action denoising error and conditioning sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

DEFAULT_RUN = Path(
    "/home/zch/wuji-openwam-dev/sandbox/"
    "sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25"
)
DEFAULT_OUTPUT = ROOT / "logs" / "sana_principles_audit_20260722" / "action_sigma_probe"
CONDITIONS = (
    "correct",
    "shuffled_observation",
    "shuffled_prompt",
    "shuffled_proprio",
    "zero_proprio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", default="checkpoint_step_12000.safetensors")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--sigmas", default="1.0,0.9,0.5,0.1")
    parser.add_argument("--video-sigma", type=float, default=0.5)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-candidate-episodes", type=int, default=30)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sigmas(raw: str) -> list[float]:
    values = [float(item) for item in raw.split(",")]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"sigmas must be in [0,1], got {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"sigmas must be unique, got {values}")
    return values


def active_arm_from_scene(dataset, episode_index: int) -> str:
    scene = dataset._scene_info.get(f"episode_{episode_index}", {})
    info = scene.get("info", scene)
    arm = str(info.get("active_arm", "unknown")).lower()
    return arm if arm in {"left", "right", "both"} else "unknown"


def candidate_windows(dataset, limit: int) -> list[dict[str, Any]]:
    by_episode: dict[int, tuple[int, int]] = {}
    for local_index, item in enumerate(dataset._window_index):
        episode_index, start, _ = item
        current = by_episode.get(int(episode_index))
        if current is None or int(start) < current[1]:
            by_episode[int(episode_index)] = (local_index, int(start))
    candidates = []
    for episode_index, (local_index, start) in sorted(by_episode.items())[:limit]:
        candidates.append(
            {
                "local_index": local_index,
                "episode_index": episode_index,
                "episode_path": dataset._episode_files[episode_index],
                "start_frame": start,
                "active_arm": active_arm_from_scene(dataset, episode_index),
            }
        )
    return candidates


def select_pairs(candidates: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    unused = list(candidates)
    pairs: list[list[dict[str, Any]]] = []
    while len(pairs) < count and len(unused) >= 2:
        first = unused.pop(0)
        preferred = next(
            (
                index
                for index, item in enumerate(unused)
                if {first["active_arm"], item["active_arm"]} == {"left", "right"}
            ),
            None,
        )
        second = unused.pop(0 if preferred is None else preferred)
        pairs.append([first, second])
    if len(pairs) != count:
        raise RuntimeError(f"requested {count} pairs but only selected {len(pairs)}")
    return pairs


def build_dataset(cfg):
    from sana_wam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset

    dataloader_cfg = OmegaConf.merge(
        cfg.dataloader,
        OmegaConf.create(
            {
                "split": "train",
                "filter_static_segments": False,
                "text_embedding_dropout": 0.0,
            }
        ),
    )
    dataset = MultiTaskRoboTwinDataset.from_config(dataloader_cfg, split="train")
    if len(dataset._sub_datasets) != 1:
        raise RuntimeError("probe expects the checkpoint's single-task RoboTwin dataset")
    return dataset, dataset._sub_datasets[0]


def cache_payload(inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    names = (
        "input_latents",
        "context",
        "seq_lens",
        "actions",
        "proprio_state",
        "proprio_seq",
        "action_is_pad",
    )
    return {
        name: inputs[name].detach().cpu()
        for name in names
        if name in inputs and isinstance(inputs[name], torch.Tensor)
    }


def load_or_prepare_pair(
    architecture,
    dataset,
    descriptors: list[dict[str, Any]],
    cache_path: Path,
    *,
    rebuild: bool,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if cache_path.is_file() and not rebuild:
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        return saved["tensors"], saved["samples"]

    samples = []
    metadata = []
    for descriptor in descriptors:
        sample = dataset[int(descriptor["local_index"])]
        samples.append(sample)
        metadata.append(
            {
                **descriptor,
                "episode_path": str(sample["episode_path"]),
                "start_frame": int(sample["start_frame"]),
                "end_frame": int(sample["end_frame"]),
                "prompt": sample["prompt"],
                "active_arm": str(sample.get("active_arm", descriptor["active_arm"])),
            }
        )
    prepared = architecture.prepare_inputs(samples)
    payload = cache_payload(prepared)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tensors": payload, "samples": metadata}, cache_path)
    return payload, metadata


def to_device(payload: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in payload.items():
        target_dtype = dtype if value.is_floating_point() else value.dtype
        result[name] = value.to(device=device, dtype=target_dtype)
    return result


def permute_batch(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] < 2:
        raise ValueError("conditioning shuffle requires batch size >= 2")
    return value.roll(shifts=1, dims=0)


def condition_inputs(
    payload: dict[str, torch.Tensor],
    condition: str,
    proprio_per_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    latents = payload["input_latents"]
    context = payload["context"]
    seq_lens = payload["seq_lens"]
    proprio = payload["proprio_state"]
    per_chunk = proprio_per_chunk
    if condition == "shuffled_observation":
        latents = permute_batch(latents)
    elif condition == "shuffled_prompt":
        context = permute_batch(context)
        seq_lens = permute_batch(seq_lens)
    elif condition == "shuffled_proprio":
        proprio = permute_batch(proprio)
        per_chunk = permute_batch(per_chunk)
    elif condition == "zero_proprio":
        proprio = torch.zeros_like(proprio)
        per_chunk = torch.zeros_like(per_chunk)
    elif condition != "correct":
        raise ValueError(condition)
    return latents, context, seq_lens, proprio, per_chunk


@torch.inference_mode()
def predict_velocity(
    architecture,
    payload: dict[str, torch.Tensor],
    *,
    condition: str,
    sigma: float,
    video_sigma: float,
    action_noise: torch.Tensor,
    video_noise: torch.Tensor,
    proprio_per_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = payload["actions"]
    latents, context, seq_lens, proprio, per_chunk = condition_inputs(
        payload, condition, proprio_per_chunk
    )
    batch, _, latent_frames, _, _ = latents.shape
    frame_chunk_size = int(architecture._ar_frame_chunk_size)
    action_tokens = actions.shape[1]

    noisy_actions = (1.0 - sigma) * actions + sigma * action_noise
    action_ts = torch.full(
        (batch, action_tokens), sigma * 1000.0, device=actions.device, dtype=actions.dtype
    )
    video_ts = torch.full(
        (batch, latent_frames),
        video_sigma * 1000.0,
        device=latents.device,
        dtype=latents.dtype,
    )
    noisy_video = (1.0 - video_sigma) * latents + video_sigma * video_noise
    if architecture._ar_bootstrap_clean_prefix:
        noisy_video[:, :, 0] = latents[:, :, 0]
        video_ts[:, 0] = 0.0

    _, prediction = architecture.forward(
        noisy_actions,
        None,
        proprio_state=proprio,
        proprio_per_chunk=per_chunk,
        latents=noisy_video,
        ar_clean_latents=latents,
        ar_clean_actions=actions,
        ar_video_frame_timesteps=video_ts,
        ar_clean_video_frame_timesteps=torch.zeros_like(video_ts),
        ar_action_token_timesteps=action_ts,
        ar_clean_action_token_timesteps=torch.zeros_like(action_ts),
        ar_frame_chunk_size=frame_chunk_size,
        ar_attn_window=int(architecture._ar_attn_window),
        timestep=video_ts.mean(dim=1),
        context=context,
        seq_lens=seq_lens,
    )
    target = action_noise - actions
    x0_hat = noisy_actions - sigma * prediction
    return prediction.float(), target.float(), x0_hat.float()


def scope_masks(
    payload: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    chunks: int,
) -> dict[str, torch.Tensor]:
    actions = payload["actions"]
    batch, tokens, dims = actions.shape
    if dims != 20 or tokens % chunks:
        raise ValueError(f"expected 20D actions divisible across chunks, got {tuple(actions.shape)}")
    tokens_per_chunk = tokens // chunks
    valid = torch.ones((batch, tokens, dims), dtype=torch.bool, device=actions.device)
    if "action_is_pad" in payload:
        valid &= (~payload["action_is_pad"].bool()).unsqueeze(-1)

    def token_mask(start: int, end: int) -> torch.Tensor:
        mask = torch.zeros_like(valid)
        mask[:, start:end, :] = True
        return mask & valid

    def dim_mask(indices: list[int]) -> torch.Tensor:
        mask = torch.zeros_like(valid)
        mask[:, :, indices] = True
        return mask & valid

    phase_tokens = {
        "bootstrap": (0, tokens_per_chunk),
        "continuation": (tokens_per_chunk, tokens),
    }
    component_dims = {
        "xyz": [0, 1, 2, 10, 11, 12],
        "rotation6d": [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18],
        "gripper": [9, 19],
    }
    arm_dims = {"left_arm": list(range(0, 10)), "right_arm": list(range(10, 20))}
    masks = {"all": valid, **{name: token_mask(*span) for name, span in phase_tokens.items()}}
    for chunk in range(chunks):
        masks[f"chunk_{chunk}"] = token_mask(
            chunk * tokens_per_chunk, (chunk + 1) * tokens_per_chunk
        )
    for name, indices in {**component_dims, **arm_dims}.items():
        masks[name] = dim_mask(indices)
    for phase, span in phase_tokens.items():
        for component, indices in component_dims.items():
            masks[f"{phase}_{component}"] = token_mask(*span) & dim_mask(indices)
        for arm, indices in arm_dims.items():
            masks[f"{phase}_{arm}"] = token_mask(*span) & dim_mask(indices)

    active = torch.zeros_like(valid)
    inactive = torch.zeros_like(valid)
    for sample_index, item in enumerate(metadata):
        arm = str(item.get("active_arm", "unknown")).lower()
        if arm == "left":
            active[sample_index, :, :10] = True
            inactive[sample_index, :, 10:] = True
        elif arm == "right":
            active[sample_index, :, 10:] = True
            inactive[sample_index, :, :10] = True
        elif arm == "both":
            active[sample_index] = True
    if bool(active.any()):
        masks["active_arm"] = active & valid
    if bool(inactive.any()):
        masks["inactive_arm"] = inactive & valid
    return masks


@dataclass
class Sums:
    count: int = 0
    velocity_error_sq: float = 0.0
    x0_error_sq: float = 0.0
    velocity_delta_sq: float = 0.0
    x0_delta_sq: float = 0.0
    prediction_sq: float = 0.0
    action_sq: float = 0.0
    target_sq: float = 0.0

    def add(
        self,
        mask: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        x0_hat: torch.Tensor,
        action: torch.Tensor,
        baseline_prediction: torch.Tensor,
        baseline_x0: torch.Tensor,
    ) -> None:
        count = int(mask.sum())
        if count == 0:
            return
        self.count += count

        def square_sum(value: torch.Tensor) -> float:
            return float(value[mask].double().square().sum().item())

        self.velocity_error_sq += square_sum(prediction - target)
        self.x0_error_sq += square_sum(x0_hat - action.float())
        self.velocity_delta_sq += square_sum(prediction - baseline_prediction)
        self.x0_delta_sq += square_sum(x0_hat - baseline_x0)
        self.prediction_sq += square_sum(baseline_prediction)
        self.action_sq += square_sum(action.float())
        self.target_sq += square_sum(target)

    def final(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0}

        def rms(total: float) -> float:
            return (total / self.count) ** 0.5

        prediction_rms = rms(self.prediction_sq)
        action_rms = rms(self.action_sq)
        return {
            "count": self.count,
            "velocity_rmse": rms(self.velocity_error_sq),
            "x0_rmse": rms(self.x0_error_sq),
            "velocity_sensitivity_rms": rms(self.velocity_delta_sq),
            "x0_sensitivity_rms": rms(self.x0_delta_sq),
            "velocity_sensitivity_over_prediction_rms": rms(self.velocity_delta_sq)
            / max(prediction_rms, 1e-12),
            "x0_sensitivity_over_action_rms": rms(self.x0_delta_sq) / max(action_rms, 1e-12),
            "baseline_prediction_rms": prediction_rms,
            "action_rms": action_rms,
            "target_rms": rms(self.target_sq),
        }


def markdown(report: dict[str, Any]) -> str:
    rows = report["metrics"]

    def find(sigma: float, condition: str, scope: str) -> dict[str, Any]:
        return next(
            row
            for row in rows
            if row["sigma"] == sigma and row["condition"] == condition and row["scope"] == scope
        )

    lines = [
        "# Fixed-sigma action denoising probe",
        "",
        f"- Checkpoint: `{report['checkpoint']['path']}`",
        f"- Expert clips: {report['sample_count']} clips in {report['pair_count']} disjoint episode pairs.",
        f"- Conditions: `{', '.join(report['conditions'])}`.",
        "- Errors are measured in the checkpoint's normalized 20D EEF action coordinates.",
        "- Every condition uses identical clean expert actions and identical action/video noise; only the named conditioning source changes.",
        "",
        "## Correct-condition denoising error",
        "",
        "| sigma | all velocity RMSE | bootstrap | continuation | xyz | rotation6d | gripper | left arm | right arm |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sigma in report["sigmas"]:
        values = [find(sigma, "correct", scope)["velocity_rmse"] for scope in (
            "all", "bootstrap", "continuation", "xyz", "rotation6d", "gripper", "left_arm", "right_arm"
        )]
        lines.append("| " + f"{sigma:.3f} | " + " | ".join(f"{value:.6f}" for value in values) + " |")

    lines.extend(
        [
            "",
            "## Conditioning sensitivity",
            "",
            "Each value is RMS change in the predicted velocity relative to the correct-condition output.",
            "",
            "| sigma | condition | all | bootstrap | continuation | error change (all) |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for sigma in report["sigmas"]:
        correct = find(sigma, "correct", "all")
        for condition in report["conditions"]:
            if condition == "correct":
                continue
            all_row = find(sigma, condition, "all")
            bootstrap = find(sigma, condition, "bootstrap")
            continuation = find(sigma, condition, "continuation")
            lines.append(
                f"| {sigma:.3f} | {condition} | {all_row['velocity_sensitivity_rms']:.6f} | "
                f"{bootstrap['velocity_sensitivity_rms']:.6f} | {continuation['velocity_sensitivity_rms']:.6f} | "
                f"{all_row['velocity_rmse'] - correct['velocity_rmse']:+.6f} |"
            )

    lines.extend(["", "## Samples", ""])
    for index, sample in enumerate(report["samples"]):
        lines.append(
            f"- `{index}` arm={sample['active_arm']} episode=`{sample['episode_path']}` "
            f"start={sample['start_frame']} prompt={json.dumps(sample['prompt'])}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sigmas = parse_sigmas(args.sigmas)
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    torch.manual_seed(args.seed)

    from sana_wam.deploy.model_loader import load_from_checkpoint_dir

    started = time.perf_counter()
    cfg, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    device = torch.device(args.device)
    dataset, subdataset = build_dataset(cfg)
    pairs = select_pairs(
        candidate_windows(subdataset, args.max_candidate_episodes), args.pairs
    )

    accumulators: dict[tuple[float, str, str], Sums] = defaultdict(Sums)
    per_pair_metrics: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []
    pair_timings = []
    for pair_index, descriptors in enumerate(pairs):
        pair_started = time.perf_counter()
        pair_accumulators: dict[tuple[float, str, str], Sums] = defaultdict(Sums)
        cache_path = args.output_dir / f"prepared_pair_{pair_index:02d}.pt"
        cpu_payload, metadata = load_or_prepare_pair(
            architecture,
            subdataset,
            descriptors,
            cache_path,
            rebuild=args.rebuild_cache,
        )
        all_metadata.extend(metadata)
        payload = to_device(cpu_payload, device, architecture.dtype)
        actions = payload["actions"]
        latent_frames = payload["input_latents"].shape[2]
        chunks = latent_frames // int(architecture._ar_frame_chunk_size)
        tokens_per_chunk = actions.shape[1] // chunks
        chunk_indices = [chunk * tokens_per_chunk for chunk in range(chunks)]
        proprio_per_chunk = payload["proprio_seq"][:, chunk_indices]
        masks = scope_masks(payload, metadata, chunks)

        generator = torch.Generator(device=device).manual_seed(args.seed + pair_index * 1009)
        action_noise = torch.randn(
            actions.shape, generator=generator, device=device, dtype=architecture.dtype
        )
        video_noise = torch.randn(
            payload["input_latents"].shape,
            generator=generator,
            device=device,
            dtype=architecture.dtype,
        )

        for sigma in sigmas:
            baseline_prediction, target, baseline_x0 = predict_velocity(
                architecture,
                payload,
                condition="correct",
                sigma=sigma,
                video_sigma=args.video_sigma,
                action_noise=action_noise,
                video_noise=video_noise,
                proprio_per_chunk=proprio_per_chunk,
            )
            for condition in CONDITIONS:
                if condition == "correct":
                    prediction, x0_hat = baseline_prediction, baseline_x0
                else:
                    prediction, condition_target, x0_hat = predict_velocity(
                        architecture,
                        payload,
                        condition=condition,
                        sigma=sigma,
                        video_sigma=args.video_sigma,
                        action_noise=action_noise,
                        video_noise=video_noise,
                        proprio_per_chunk=proprio_per_chunk,
                    )
                    if not torch.equal(target, condition_target):
                        raise RuntimeError("fixed-noise target changed across conditions")
                for scope, mask in masks.items():
                    for accumulator in (
                        accumulators[(sigma, condition, scope)],
                        pair_accumulators[(sigma, condition, scope)],
                    ):
                        accumulator.add(
                            mask,
                            prediction,
                            target,
                            x0_hat,
                            actions,
                            baseline_prediction,
                            baseline_x0,
                        )
            print(
                f"pair={pair_index} sigma={sigma:.3f} complete "
                f"elapsed={time.perf_counter() - pair_started:.1f}s",
                flush=True,
            )
        pair_timings.append(time.perf_counter() - pair_started)
        per_pair_metrics.extend(
            {
                "pair_index": pair_index,
                "sigma": sigma,
                "condition": condition,
                "scope": scope,
                **sums.final(),
            }
            for (sigma, condition, scope), sums in sorted(
                pair_accumulators.items(),
                key=lambda item: (item[0][0], item[0][1], item[0][2]),
            )
        )
        del payload, action_noise, video_noise
        torch.cuda.empty_cache()

    metrics = []
    for (sigma, condition, scope), sums in sorted(
        accumulators.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        metrics.append(
            {"sigma": sigma, "condition": condition, "scope": scope, **sums.final()}
        )
    checkpoint_path = args.run_dir / args.checkpoint
    contract_path = args.output_dir.parent / "checkpoint_contract_audit.json"
    checkpoint_sha = None
    if contract_path.is_file():
        checkpoint_sha = json.loads(contract_path.read_text())["provenance"].get(
            "checkpoint_sha256"
        )
    if not checkpoint_sha:
        checkpoint_sha = sha256_file(checkpoint_path)
    report = {
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "config_path": str(args.run_dir / "config.yaml"),
        "seed": args.seed,
        "sigmas": sigmas,
        "video_sigma": args.video_sigma,
        "conditions": list(CONDITIONS),
        "pair_count": args.pairs,
        "sample_count": len(all_metadata),
        "samples": all_metadata,
        "pair_seconds": pair_timings,
        "total_seconds": time.perf_counter() - started,
        "fixed_noise_contract": {
            "same_action_noise_across_sigmas_within_pair": True,
            "same_action_and_video_noise_across_conditions": True,
            "clean_expert_action_history": True,
            "clean_expert_video_copy": True,
            "model_mode": "eval/inference_mode",
            "closed_loop_simulation": False,
        },
        "metrics": metrics,
        "per_pair_metrics": per_pair_metrics,
    }
    json_path = args.output_dir / "action_sigma_probe.json"
    markdown_path = args.output_dir / "action_sigma_probe.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
