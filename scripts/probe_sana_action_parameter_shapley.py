#!/usr/bin/env python
"""Factor video parameter groups into cache-teacher action MSE by Shapley value."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT if (ROOT / "src").is_dir() else ROOT.parent / "phase4"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Sana"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from probe_sana_action_denoising import parse_sigmas, to_device  # noqa: E402
from probe_sana_action_weight_overlay import (  # noqa: E402
    GROUP_PATTERNS,
    apply_groups,
    cache_group_tensors,
    matching_keys,
)
from probe_sana_ar_decomposition import build_expert_history, query_action  # noqa: E402
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


GROUPS = tuple(GROUP_PATTERNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--overlay-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--sigmas", default="1.0,0.9,0.5")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def tensor_sha256(values: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def bootstrap_ci(
    values: np.ndarray, *, seed: int, label: str, samples: int
) -> tuple[float, float]:
    derived = int.from_bytes(
        hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(derived)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def coalition_groups(mask: int) -> tuple[str, ...]:
    return tuple(group for index, group in enumerate(GROUPS) if mask & (1 << index))


def mobius(values: dict[int, float]) -> dict[int, float]:
    result = {}
    for mask in range(1 << len(GROUPS)):
        result[mask] = values[mask] - sum(
            result[submask]
            for submask in range(mask)
            if submask & mask == submask
        )
    return result


def shapley(values: dict[int, float]) -> dict[str, float]:
    count = len(GROUPS)
    result = {}
    for index, group in enumerate(GROUPS):
        bit = 1 << index
        contribution = 0.0
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(count - size - 1)
                / math.factorial(count)
            )
            contribution += weight * (values[mask | bit] - values[mask])
        result[group] = contribution
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Video-parameter Shapley action decomposition",
        "",
        f"- Task: `{report['task']}`",
        f"- Contexts: `{report['context_count']}`",
        f"- Endpoint validation: `{report['validation']['result']}`",
        "",
        "## Integrated Shapley contributions",
        "",
        "Positive values increase cache-teacher action MSE.",
        "",
        "| Group | Mean MSE contribution | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for row in report["integrated"]["shapley"]:
        lines.append(
            f"| {row['group']} | {row['mean']:+.6f} | "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            f"Full mixed-minus-Phase1 MSE: "
            f"`{report['integrated']['total_delta_mean']:+.6f}`.",
            "",
            "The JSON contains all eight coalition outcomes, context-level "
            "Mobius interactions, and exact Shapley efficiency residuals.",
        ]
    )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sigmas = parse_sigmas(args.sigmas)
    if args.pairs <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("pairs and bootstrap-samples must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    _, architecture = load_from_checkpoint_dir(
        str(args.base_run_dir),
        device=args.device,
        ckpt_name=args.base_checkpoint,
    )
    architecture.eval()
    device = torch.device(args.device)
    driver = architecture._mot_driver or architecture.build_mot_driver()
    model_state = architecture.state_dict(keep_vars=True)
    state_keys = list(model_state)
    group_keys = {
        group: matching_keys(state_keys, pattern)
        for group, pattern in GROUP_PATTERNS.items()
    }
    base_checkpoint = args.base_run_dir / args.base_checkpoint
    base_tensors = cache_group_tensors(base_checkpoint, group_keys)
    overlay_tensors = cache_group_tensors(args.overlay_checkpoint, group_keys)
    group_contract = {}
    seen = set()
    for group, keys in group_keys.items():
        overlap = seen.intersection(keys)
        if overlap or not keys:
            raise RuntimeError(f"invalid parameter group {group}: overlap={overlap}")
        seen.update(keys)
        elements = sum(model_state[key].numel() for key in keys)
        changed = sum(
            int(torch.count_nonzero(base_tensors[group][key] != overlay_tensors[group][key]))
            for key in keys
        )
        group_contract[group] = {
            "pattern": GROUP_PATTERNS[group],
            "tensor_count": len(keys),
            "parameter_count": elements,
            "changed_element_count": changed,
        }

    prepared = [
        torch.load(
            args.prepared_dir / f"prepared_pair_{pair_index:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for pair_index in range(args.pairs)
    ]
    noise_by_pair = []
    for pair_index, saved in enumerate(prepared):
        actions = saved["tensors"]["actions"]
        generator = torch.Generator(device=device).manual_seed(
            args.seed + pair_index * 1009
        )
        noise_by_pair.append(
            torch.randn(
                actions.shape,
                generator=generator,
                device=device,
                dtype=architecture.dtype,
            ).cpu()
        )
    noise_sha = tensor_sha256(noise_by_pair)

    square_sums: dict[int, dict[float, dict[int, float]]] = {
        mask: {sigma: {} for sigma in sigmas} for mask in range(1 << len(GROUPS))
    }
    counts: dict[int, dict[float, dict[int, int]]] = {
        mask: {sigma: {} for sigma in sigmas} for mask in range(1 << len(GROUPS))
    }
    condition_seconds = {}
    for mask in range(1 << len(GROUPS)):
        condition_started = time.perf_counter()
        enabled = coalition_groups(mask)
        apply_groups(model_state, base_tensors, overlay_tensors, enabled)
        for pair_index, saved in enumerate(prepared):
            payload = to_device(saved["tensors"], device, architecture.dtype)
            metadata = saved["samples"]
            actions = payload["actions"]
            latent_frames = payload["input_latents"].shape[2]
            chunks = latent_frames // int(architecture._ar_frame_chunk_size)
            tokens_per_chunk = actions.shape[1] // chunks
            video_chunks = list(
                payload["input_latents"].split(
                    int(architecture._ar_frame_chunk_size), dim=2
                )
            )
            action_chunks = list(actions.split(tokens_per_chunk, dim=1))
            proprio_chunks = [
                payload["proprio_seq"][:, chunk * tokens_per_chunk]
                for chunk in range(chunks)
            ]
            context, context_mask = architecture._rollout_step_context(
                payload["context"], payload["seq_lens"], payload["proprio_state"]
            )
            action_noise = noise_by_pair[pair_index].to(device)
            pair_sums = {
                sigma: torch.zeros(actions.shape[0], dtype=torch.float64)
                for sigma in sigmas
            }
            pair_counts = {
                sigma: torch.zeros(actions.shape[0], dtype=torch.int64)
                for sigma in sigmas
            }
            for chunk in range(chunks):
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
                start = chunk * tokens_per_chunk
                end = start + tokens_per_chunk
                action_chunk = action_chunks[chunk]
                noise_chunk = action_noise[:, start:end]
                valid = torch.ones_like(action_chunk, dtype=torch.bool)
                if "action_is_pad" in payload:
                    valid &= (~payload["action_is_pad"][:, start:end]).unsqueeze(-1)
                for sigma in sigmas:
                    noisy_action = (1.0 - sigma) * action_chunk + sigma * noise_chunk
                    prediction = query_action(
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
                    target = noise_chunk.float() - action_chunk.float()
                    error = (prediction - target).double().square()
                    for item_index in range(actions.shape[0]):
                        item_mask = valid[item_index]
                        pair_sums[sigma][item_index] += error[item_index][
                            item_mask
                        ].sum().cpu()
                        pair_counts[sigma][item_index] += int(item_mask.sum())
            for item_index, item in enumerate(metadata):
                episode = int(item["episode_index"])
                for sigma in sigmas:
                    square_sums[mask][sigma][episode] = float(
                        pair_sums[sigma][item_index]
                    )
                    counts[mask][sigma][episode] = int(
                        pair_counts[sigma][item_index]
                    )
            print(
                f"condition={mask:03b} pair={pair_index + 1}/{args.pairs}",
                flush=True,
            )
        condition_seconds[f"{mask:03b}"] = time.perf_counter() - condition_started
        torch.cuda.empty_cache()

    context_mse: dict[int, dict[float, dict[int, float]]] = {
        mask: {
            sigma: {
                episode: square_sums[mask][sigma][episode]
                / counts[mask][sigma][episode]
                for episode in square_sums[mask][sigma]
            }
            for sigma in sigmas
        }
        for mask in range(1 << len(GROUPS))
    }
    episodes = sorted(context_mse[0][sigmas[0]])
    for mask in context_mse:
        for sigma in sigmas:
            if sorted(context_mse[mask][sigma]) != episodes:
                raise RuntimeError("coalition context identities differ")

    def summarize_outcomes(
        values_by_episode: dict[int, dict[int, float]], label: str
    ) -> dict[str, Any]:
        context_rows = []
        shapley_arrays = {group: [] for group in GROUPS}
        mobius_arrays = {mask: [] for mask in range(1 << len(GROUPS))}
        total_deltas = []
        efficiency = []
        for episode in episodes:
            values = {
                mask: values_by_episode[mask][episode]
                for mask in range(1 << len(GROUPS))
            }
            mobius_values = mobius(values)
            shapley_values = shapley(values)
            total = values[(1 << len(GROUPS)) - 1] - values[0]
            residual = sum(shapley_values.values()) - total
            total_deltas.append(total)
            efficiency.append(residual)
            for group in GROUPS:
                shapley_arrays[group].append(shapley_values[group])
            for mask in mobius_arrays:
                mobius_arrays[mask].append(mobius_values[mask])
            context_rows.append(
                {
                    "episode_index": episode,
                    "coalition_mse": {
                        f"{mask:03b}": values[mask]
                        for mask in range(1 << len(GROUPS))
                    },
                    "mobius": {
                        f"{mask:03b}": mobius_values[mask]
                        for mask in range(1 << len(GROUPS))
                    },
                    "shapley": shapley_values,
                    "total_delta": total,
                    "efficiency_residual": residual,
                }
            )
        shapley_rows = []
        for group in GROUPS:
            values = np.asarray(shapley_arrays[group], dtype=np.float64)
            low, high = bootstrap_ci(
                values,
                seed=args.seed,
                label=f"{label}:shapley:{group}",
                samples=args.bootstrap_samples,
            )
            shapley_rows.append(
                {
                    "group": group,
                    "mean": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        mobius_rows = []
        for mask in range(1 << len(GROUPS)):
            values = np.asarray(mobius_arrays[mask], dtype=np.float64)
            low, high = bootstrap_ci(
                values,
                seed=args.seed,
                label=f"{label}:mobius:{mask}",
                samples=args.bootstrap_samples,
            )
            mobius_rows.append(
                {
                    "mask": f"{mask:03b}",
                    "groups": list(coalition_groups(mask)),
                    "mean": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        total_values = np.asarray(total_deltas)
        total_low, total_high = bootstrap_ci(
            total_values,
            seed=args.seed,
            label=f"{label}:total",
            samples=args.bootstrap_samples,
        )
        return {
            "coalition_means": {
                f"{mask:03b}": float(
                    np.mean([values_by_episode[mask][episode] for episode in episodes])
                )
                for mask in range(1 << len(GROUPS))
            },
            "total_delta_mean": float(total_values.mean()),
            "total_delta_ci95_low": total_low,
            "total_delta_ci95_high": total_high,
            "shapley": shapley_rows,
            "mobius": mobius_rows,
            "max_abs_efficiency_residual": float(
                np.max(np.abs(np.asarray(efficiency)))
            ),
            "contexts": context_rows,
        }

    sigma_reports = {}
    for sigma in sigmas:
        values = {
            mask: context_mse[mask][sigma]
            for mask in range(1 << len(GROUPS))
        }
        sigma_reports[str(sigma)] = summarize_outcomes(values, f"sigma:{sigma}")
    integrated_values = {
        mask: {
            episode: float(
                np.mean([context_mse[mask][sigma][episode] for sigma in sigmas])
            )
            for episode in episodes
        }
        for mask in range(1 << len(GROUPS))
    }
    integrated = summarize_outcomes(integrated_values, "integrated")

    reference_validation = None
    if args.reference_json is not None:
        reference = json.loads(args.reference_json.read_text())
        differences = []
        for sigma in sigmas:
            for condition, mask in (("phase1_video", 0), ("selected_video", 7)):
                expected = reference["context_metrics"][condition][str(sigma)]
                for episode in episodes:
                    differences.append(
                        abs(
                            context_mse[mask][sigma][episode]
                            - float(expected[str(episode)])
                        )
                    )
        reference_validation = {
            "path": str(args.reference_json),
            "max_abs_context_mse_delta": max(differences),
        }
    validation_pass = (
        integrated["max_abs_efficiency_residual"] <= 1e-12
        and all(
            report["max_abs_efficiency_residual"] <= 1e-12
            for report in sigma_reports.values()
        )
        and (
            reference_validation is None
            or reference_validation["max_abs_context_mse_delta"] <= 1e-12
        )
        and all(
            math.isfinite(value)
            for mask in context_mse.values()
            for by_episode in mask.values()
            for value in by_episode.values()
        )
    )
    report = {
        "task": args.prepared_dir.name.removesuffix("_contexts_seed20260724"),
        "base_checkpoint": str(base_checkpoint),
        "overlay_checkpoint": str(args.overlay_checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "pair_count": args.pairs,
        "context_count": len(episodes),
        "sigmas": sigmas,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "groups": list(GROUPS),
        "group_contract": group_contract,
        "coalitions": [
            {"mask": f"{mask:03b}", "groups": list(coalition_groups(mask))}
            for mask in range(1 << len(GROUPS))
        ],
        "action_noise_sha256": noise_sha,
        "validation": {
            "result": "PASS" if validation_pass else "FAIL",
            "reference": reference_validation,
        },
        "by_sigma": sigma_reports,
        "integrated": integrated,
        "condition_seconds": condition_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "action_parameter_shapley.json"
    markdown_path = args.output_dir / "action_parameter_shapley.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))
    if not validation_pass:
        raise RuntimeError("parameter Shapley validation failed")


if __name__ == "__main__":
    main()
