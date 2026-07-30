#!/usr/bin/env python
"""Audit the exact architecture and training/inference contract of a SANA AR checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

DEFAULT_RUN = Path(
    "/home/zch/wuji-openwam-dev/sandbox/"
    "sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25"
)
DEFAULT_CHECKPOINT = "checkpoint_step_12000.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs" / "sana_principles_audit_20260722")
    parser.add_argument("--visibility-chunks", type=int, default=3)
    parser.add_argument("--verify-model-load", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-checkpoint-sha256", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def shifted_schedule(steps: int, shift: float) -> torch.Tensor:
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
    return shift * base / (1.0 + (shift - 1.0) * base)


def action_low_noise_weights(sigmas: torch.Tensor) -> torch.Tensor:
    snr = ((1.0 - sigmas.clamp(1e-3, 1.0)) / sigmas.clamp(1e-3, 1.0)).square()
    raw = snr.clamp(max=2.0)
    return raw * (len(raw) / raw.sum())


def schedule_report() -> dict[str, Any]:
    action_train = shifted_schedule(1000, 5.0)
    video_train = shifted_schedule(1000, 3.0)
    action_weights = action_low_noise_weights(action_train)
    bins = {
        "high_sigma_0.8_1.0": (0.8, 1.0),
        "mid_sigma_0.35_0.65": (0.35, 0.65),
        "low_sigma_0.0_0.2": (0.0, 0.2),
    }

    def masses(sigmas: torch.Tensor, weights: torch.Tensor | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, (lo, hi) in bins.items():
            mask = (sigmas >= lo) & (sigmas <= hi)
            item = {
                "count": int(mask.sum()),
                "sample_fraction": float(mask.double().mean()),
            }
            if weights is not None:
                item["normalized_loss_mass"] = float(weights[mask].sum() / weights.sum())
                item["mean_training_weight"] = float(weights[mask].mean()) if bool(mask.any()) else None
            result[name] = item
        return result

    return {
        "training": {
            "num_discrete_ids": 1000,
            "timestep_id_sampling": "uniform independently per chunk",
            "video_shift": 3.0,
            "video_loss_weighting": "uniform",
            "video_sigma_bins": masses(video_train),
            "action_shift": 5.0,
            "action_loss_weighting": "low_noise: normalized clamp(((1-sigma)/sigma)^2, max=2)",
            "action_sigma_bins": masses(action_train, action_weights),
            "action_weight_summary": {
                "min": float(action_weights.min()),
                "median": float(action_weights.median()),
                "max": float(action_weights.max()),
                "sigma_1_weight": float(action_weights[0]),
            },
        },
        "inference": {
            "steps": 10,
            "video_shift": 3.0,
            "video_sigmas_including_terminal_zero": [
                float(value) for value in shifted_schedule(10, 3.0)
            ]
            + [0.0],
            "action_shift": 5.0,
            "action_sigmas_including_terminal_zero": [
                float(value) for value in shifted_schedule(10, 5.0)
            ]
            + [0.0],
            "integration": "explicit Euler rectified-flow step x_next=x+v*(sigma_next-sigma)",
        },
    }


def geometry_report(config: dict[str, Any]) -> dict[str, Any]:
    dataloader = config["dataloader"]
    architecture = config["model"]["architecture"]
    raw_frames = int(dataloader["num_frames"])
    video_stride = int(dataloader["video_stride"])
    sampled_video_frames = len(range(0, raw_frames, video_stride))
    temporal_compression = int(dataloader["temporal_compression"])
    latent_frames = 1 + (sampled_video_frames - 1) // temporal_compression
    spatial_compression = 8
    latent_h = int(dataloader["height"]) // spatial_compression
    latent_w = int(dataloader["width"]) // spatial_compression
    patch_t, patch_h, patch_w = (1, 2, 2)
    patch_grid = (latent_frames // patch_t, latent_h // patch_h, latent_w // patch_w)
    frame_chunk_size = int(architecture["ar_frame_chunk_size"])
    num_chunks = latent_frames // frame_chunk_size
    action_tokens = raw_frames - 1
    if latent_frames % frame_chunk_size:
        raise RuntimeError(
            f"latent frames {latent_frames} not divisible by chunk size {frame_chunk_size}"
        )
    if action_tokens % num_chunks:
        raise RuntimeError(
            f"action tokens {action_tokens} not divisible by chunks {num_chunks}"
        )
    video_tokens_per_chunk = (
        frame_chunk_size * patch_grid[1] * patch_grid[2]
    )
    action_tokens_per_chunk = action_tokens // num_chunks
    total_duplicated_tokens = 2 * (
        video_tokens_per_chunk * num_chunks + action_tokens
    )
    return {
        "raw_robot_steps": raw_frames,
        "raw_action_targets": action_tokens,
        "video_stride": video_stride,
        "sampled_rgb_frames": sampled_video_frames,
        "wan_vae": {
            "latent_channels": 16,
            "temporal_compression": temporal_compression,
            "spatial_compression": spatial_compression,
            "causal": bool(dataloader["causal_temporal"]),
            "latent_shape_without_batch": [16, latent_frames, latent_h, latent_w],
        },
        "dit_patch_size": [patch_t, patch_h, patch_w],
        "dit_patch_grid": list(patch_grid),
        "ar_frame_chunk_size_latents": frame_chunk_size,
        "num_ar_chunks": num_chunks,
        "video_tokens_per_chunk": video_tokens_per_chunk,
        "action_tokens_per_chunk": action_tokens_per_chunk,
        "canonical_duplicated_sequence_tokens": total_duplicated_tokens,
        "ar_frame_axis": [
            {"chunk": chunk, "video_frame_id": 2 * chunk, "action_frame_id": 2 * chunk + 1}
            for chunk in range(num_chunks)
        ],
    }


def visibility_report(chunks: int, window: int) -> dict[str, Any]:
    tokens: list[dict[str, Any]] = []
    for chunk in range(chunks):
        for modality, frame in (("video", 2 * chunk), ("action", 2 * chunk + 1)):
            for copy, noise_id in (("noisy", 0), ("clean", 1)):
                tokens.append(
                    {
                        "label": f"{modality[0].upper()}{copy[0].upper()}{chunk}",
                        "chunk": chunk,
                        "modality": modality,
                        "copy": copy,
                        "frame_id": frame,
                        "noise_id": noise_id,
                    }
                )

    def visible(query: dict[str, Any], key: dict[str, Any]) -> bool:
        fi, fj = query["frame_id"], key["frame_id"]
        ni, nj = query["noise_id"], key["noise_id"]
        clean2clean = ni == 1 and nj == 1 and fj <= fi
        noise2clean = ni == 0 and nj == 1 and fj < fi
        noise2noise = ni == 0 and nj == 0 and fj == fi
        return (clean2clean or noise2clean or noise2noise) and abs(fi - fj) <= window

    matrix = [
        [1 if visible(query, key) else 0 for key in tokens]
        for query in tokens
    ]
    return {
        "legend": "row=query, column=key; 1=visible; V/A=modality, N/C=noisy/clean copy",
        "token_order": [item["label"] for item in tokens],
        "token_metadata": tokens,
        "matrix": matrix,
        "rule": (
            "(clean_q&clean_k&f_k<=f_q) OR (noisy_q&clean_k&f_k<f_q) OR "
            "(noisy_q&noisy_k&f_k==f_q), then |f_q-f_k|<=window"
        ),
    }


def component_name(key: str) -> str:
    if key.startswith("video_backbone.dit."):
        return "video_dit"
    if key.startswith("video_backbone.vae."):
        return "wan_vae"
    if key.startswith("video_backbone.text_encoder."):
        return "gemma_text_encoder"
    if key.startswith("action_backbone."):
        return "action_backbone"
    if key.startswith(("proprio_video_embed.", "proprio_action_embed.")):
        return "per_chunk_proprio_encoders"
    if key.startswith("proprio_encoder.") or "proprio" in key:
        return "clip_proprio_context"
    return "other_architecture"


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "mean": float(value.mean()),
        "std": float(value.std()) if value.numel() > 1 else 0.0,
        "min": float(value.min()),
        "max": float(value.max()),
        "l2": float(torch.linalg.vector_norm(value)),
        "nonzero_fraction": float((value != 0).float().mean()),
    }


def checkpoint_report(path: Path) -> dict[str, Any]:
    component_tensors: dict[str, int] = defaultdict(int)
    component_numel: dict[str, int] = defaultdict(int)
    checkpoint_dtypes: dict[str, int] = defaultdict(int)
    block_patterns = {
        "video": re.compile(r"^video_backbone\.dit\.blocks\.(\d+)\."),
        "action": re.compile(r"^action_backbone\.blocks\.(\d+)\."),
    }
    block_ids: dict[str, set[int]] = {"video": set(), "action": set()}
    layers: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    all_keys: list[str]

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        all_keys = list(checkpoint.keys())
        for key in all_keys:
            shape = tuple(checkpoint.get_slice(key).get_shape())
            numel = math.prod(shape)
            component = component_name(key)
            component_tensors[component] += 1
            component_numel[component] += numel
            checkpoint_dtypes[str(checkpoint.get_slice(key).get_dtype())] += numel
            for name, pattern in block_patterns.items():
                match = pattern.match(key)
                if match:
                    block_ids[name].add(int(match.group(1)))

        for layer in sorted(block_ids["video"]):
            stem = f"video_backbone.dit.blocks.{layer}"
            scale_key = f"{stem}.learnable_fa_scale"
            alpha_key = f"{stem}.attn.kernel_func.alpha"
            w1_key = f"{stem}.attn.kernel_func.w1.weight"
            layers.append(
                {
                    "layer": layer,
                    "learnable_fa_scale": float(checkpoint.get_tensor(scale_key).float().item()),
                    "feature_map_alpha": float(checkpoint.get_tensor(alpha_key).float().item()),
                    "feature_map_w1_l2": float(
                        torch.linalg.vector_norm(checkpoint.get_tensor(w1_key).float())
                    ),
                }
            )

        for key in (
            "proprio_video_embed.weight",
            "proprio_video_embed.bias",
            "proprio_action_embed.weight",
            "proprio_action_embed.bias",
            "video_backbone.dit.x_embedder.proj.weight",
            "video_backbone.dit.blocks.0.attn.qkv.weight",
            "action_backbone.blocks.0.self_attn.q.weight",
        ):
            if key in all_keys:
                selected[key] = tensor_stats(checkpoint.get_tensor(key))

    component_rows = {
        name: {
            "tensor_count": component_tensors[name],
            "numel": component_numel[name],
            "billion_values": component_numel[name] / 1e9,
        }
        for name in sorted(component_tensors)
    }
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "tensor_count": len(all_keys),
        "stored_values_by_dtype": dict(sorted(checkpoint_dtypes.items())),
        "components": component_rows,
        "video_block_ids": sorted(block_ids["video"]),
        "action_block_ids": sorted(block_ids["action"]),
        "layerwise_linear_attention": layers,
        "selected_tensor_stats": selected,
        "contains_action_self_rebalance_parameters": any(
            "action_self" in key for key in all_keys
        ),
        "key_fingerprint_sha256": hashlib.sha256("\n".join(all_keys).encode()).hexdigest(),
        "keys": all_keys,
    }


def verify_model(report: dict[str, Any], run_dir: Path, checkpoint: str, device: str) -> None:
    from sana_wam.deploy.model_loader import load_from_checkpoint_dir

    _, architecture = load_from_checkpoint_dir(
        str(run_dir), device=device, ckpt_name=checkpoint
    )
    driver = architecture._mot_driver or architecture.build_mot_driver()
    checkpoint_keys = set(report["checkpoint"]["keys"])
    state = architecture.state_dict()
    state_keys = set(state)
    first_video_block = architecture.video_backbone._dit.blocks[0]
    first_action_block = architecture.action_backbone.blocks[0]
    report["live_model_verification"] = {
        "strict_checkpoint_load": "PASS",
        "architecture_class": type(architecture).__name__,
        "driver_class": type(driver).__name__,
        "video_backbone_class": type(architecture.video_backbone).__name__,
        "action_backbone_class": type(architecture.action_backbone).__name__,
        "video_attention_class": type(first_video_block.attn).__name__,
        "video_feature_map_class": type(first_video_block.attn.kernel_func).__name__,
        "video_additional_attention_class": type(first_video_block.flash_attn_additional).__name__,
        "action_attention_class": type(first_action_block.self_attn).__name__,
        "state_dict_tensor_count": len(state),
        "checkpoint_tensor_count": len(checkpoint_keys),
        "missing_from_checkpoint": sorted(state_keys - checkpoint_keys),
        "unexpected_in_checkpoint": sorted(checkpoint_keys - state_keys),
        "parameter_numel": sum(parameter.numel() for parameter in architecture.parameters()),
        "trainable_parameter_numel_in_eval_object": sum(
            parameter.numel() for parameter in architecture.parameters() if parameter.requires_grad
        ),
        "video": {
            "layers": architecture.video_backbone.num_layers,
            "dim": architecture.video_backbone.dim,
            "heads": architecture.video_backbone.num_heads,
            "head_dim": architecture.video_backbone.head_dim,
            "context_dim": architecture.video_backbone.context_dim,
            "dit_patch_size": list(architecture.video_backbone.dit_patch_size),
            "temporal_compression": architecture.video_backbone.temporal_compression,
            "spatial_compression": architecture.video_backbone._spatial_compression,
            "causal_temporal": architecture.video_backbone.causal_temporal,
        },
        "action": {
            "layers": len(architecture.action_backbone.blocks),
            "dim": architecture.action_backbone.dim,
            "action_dim": architecture.action_backbone.action_dim,
            "attention_heads": first_action_block.self_attn.num_heads,
            "attention_head_dim": first_action_block.self_attn.attn_head_dim,
        },
        "ar": {
            "frame_chunk_size": architecture._ar_frame_chunk_size,
            "attention_window": architecture._ar_attn_window,
            "bootstrap_clean_prefix": architecture._ar_bootstrap_clean_prefix,
            "proprio_per_chunk": architecture._proprio_per_chunk,
            "proprio_action_dropout_prob": architecture._proprio_action_dropout_prob,
            "driver_has_action_self_attn_weight": hasattr(driver, "action_self_attn_weight"),
            "driver_action_self_attn_weight": getattr(driver, "action_self_attn_weight", None),
            "dense_action_row_rebalance_active": False,
        },
    }


def markdown(report: dict[str, Any]) -> str:
    checkpoint = report["checkpoint"]
    geometry = report["geometry"]
    schedule = report["schedules"]
    action_bins = schedule["training"]["action_sigma_bins"]
    visibility = report["visibility"]
    lines = [
        "# SANA AR checkpoint contract audit",
        "",
        f"- Checkpoint: `{checkpoint['file']}`",
        f"- SHA256: `{report['provenance'].get('checkpoint_sha256', 'skipped')}`",
        "- Architecture: `DualSystemARArchitecture` -> `SanaARMoTJointDriver`",
        f"- Checkpoint tensors: {checkpoint['tensor_count']}; video/action blocks: "
        f"{len(checkpoint['video_block_ids'])}/{len(checkpoint['action_block_ids'])}",
        "",
        "## Geometry",
        "",
        f"- Raw window {geometry['raw_robot_steps']} states -> {geometry['sampled_rgb_frames']} RGB frames "
        f"(stride {geometry['video_stride']}) -> {geometry['wan_vae']['latent_shape_without_batch'][1]} Wan latent frames.",
        f"- Latent `{geometry['wan_vae']['latent_shape_without_batch']}` -> patch grid "
        f"`{geometry['dit_patch_grid']}` -> {geometry['num_ar_chunks']} AR chunks.",
        f"- Each chunk contains {geometry['video_tokens_per_chunk']} video tokens and "
        f"{geometry['action_tokens_per_chunk']} action tokens; duplicated train sequence has "
        f"{geometry['canonical_duplicated_sequence_tokens']} tokens.",
        "",
        "## Action objective",
        "",
    ]
    for name, item in action_bins.items():
        lines.append(
            f"- `{name}`: sampled {100 * item['sample_fraction']:.2f}% of timestep IDs, "
            f"receives {100 * item['normalized_loss_mass']:.4f}% of normalized action-loss mass "
            f"(mean weight {item['mean_training_weight']:.6g})."
        )
    lines.extend(
        [
            f"- Ten-step action inference sigmas: `{schedule['inference']['action_sigmas_including_terminal_zero']}`",
            "",
            "## AR visibility",
            "",
            f"Token order: `{' '.join(visibility['token_order'])}`",
            "",
            "```text",
            "query\\key " + " ".join(f"{label:>3}" for label in visibility["token_order"]),
        ]
    )
    for label, row in zip(visibility["token_order"], visibility["matrix"]):
        lines.append(f"{label:>9} " + " ".join(f"{value:>3}" for value in row))
    lines.extend(
        [
            "```",
            "",
            "## Linear-attention parameters",
            "",
            "| layer | flash scale | feature-map alpha | w1 L2 |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for item in checkpoint["layerwise_linear_attention"]:
        lines.append(
            f"| {item['layer']} | {item['learnable_fa_scale']:.6g} | "
            f"{item['feature_map_alpha']:.6g} | {item['feature_map_w1_l2']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Contract checks",
            "",
            f"- Checkpoint contains action-row rebalance parameters: "
            f"`{checkpoint['contains_action_self_rebalance_parameters']}`.",
            "- `DualSystemARArchitecture.build_mot_driver()` removes the dense-mask "
            "`action_self_attn_*` and `action_video_attn_mode` knobs before constructing the AR driver.",
            "- The current AR kernel uses structured clean-history plus within-frame noisy attention; "
            "dense MoT action-row rebalance is inactive.",
        ]
    )
    live = report.get("live_model_verification")
    if live:
        lines.extend(
            [
                "",
                "## Live strict-load verification",
                "",
                f"- Strict load: `{live['strict_checkpoint_load']}`; architecture/driver: "
                f"`{live['architecture_class']}` / `{live['driver_class']}`.",
                f"- State/checkpoint keys: {live['state_dict_tensor_count']} / "
                f"{live['checkpoint_tensor_count']}; missing={len(live['missing_from_checkpoint'])}, "
                f"unexpected={len(live['unexpected_in_checkpoint'])}.",
                f"- Video: {live['video']['layers']} layers, dim {live['video']['dim']}, "
                f"{live['video']['heads']}x{live['video']['head_dim']} attention.",
                f"- Action: {live['action']['layers']} layers, dim {live['action']['dim']}, "
                f"{live['action']['attention_heads']}x{live['action']['attention_head_dim']} attention.",
            ]
        )
    lines.extend(
        [
            "",
            "## Immediate falsifiable hypothesis",
            "",
            "The low-noise action weighting may train local denoising/refinement while supplying too "
            "little gradient to high-noise global mode formation. This report establishes the objective "
            "imbalance but does not infer behavior from it. The next probe must measure fixed-noise expert "
            "velocity error and conditioning sensitivity at high/mid/low sigma.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.run_dir / "config.yaml"
    checkpoint_path = args.run_dir / args.checkpoint
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(cfg, dict)
    report: dict[str, Any] = {
        "provenance": {
            "repo_root": str(ROOT),
            "git_head": git_value("rev-parse", "HEAD"),
            "git_status_porcelain": git_value("status", "--porcelain"),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint_sha256": (
                None if args.skip_checkpoint_sha256 else sha256_file(checkpoint_path)
            ),
        },
        "resolved_config": cfg,
        "architecture_contract": {
            "architecture_class": "DualSystemARArchitecture",
            "driver_class": "SanaARMoTJointDriver",
            "video_backbone": "SanaVideoBackbone (SANA-Video-2B-480p)",
            "action_backbone": "ActionDiT joint_self_attn expert",
            "attention": "AR structured MoT linear-ReLU + video-only window-flash branch",
            "action_row_rebalance": "inactive in AR driver",
        },
        "geometry": geometry_report(cfg),
        "schedules": schedule_report(),
        "visibility": visibility_report(
            args.visibility_chunks, int(cfg["model"]["architecture"]["ar_attn_window"])
        ),
        "checkpoint": checkpoint_report(checkpoint_path),
    }
    if args.verify_model_load:
        verify_model(report, args.run_dir, args.checkpoint, args.device)

    # The full key list is useful for coverage computation but makes the artifact noisy.
    report["checkpoint"].pop("keys", None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "checkpoint_contract_audit.json"
    markdown_path = args.output_dir / "checkpoint_contract_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
