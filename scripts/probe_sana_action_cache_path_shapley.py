#!/usr/bin/env python
"""Decompose video-to-action effects through the deployed SANA cache states."""

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
from sana_wam.model.ar.sana_ar_inference import ARLinearStateCache  # noqa: E402


CACHE_CATEGORIES = ("current_video", "prior_video", "action_history")
INTERFACE_PLAYERS = ("video_S_numerator", "video_z_normalizer", "action_history")


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


def coalition_names(mask: int, players: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        player for index, player in enumerate(players) if mask & (1 << index)
    )


def mobius(values: dict[int, float], player_count: int) -> dict[int, float]:
    result = {}
    for mask in range(1 << player_count):
        result[mask] = values[mask] - sum(
            result[submask]
            for submask in range(mask)
            if submask & mask == submask
        )
    return result


def shapley(values: dict[int, float], players: tuple[str, ...]) -> dict[str, float]:
    count = len(players)
    result = {}
    for index, player in enumerate(players):
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
        result[player] = contribution
    return result


def entry_category(frame_id: int, current_video_frame: int) -> str:
    if frame_id == current_video_frame:
        return "current_video"
    if frame_id % 2 == 0:
        return "prior_video"
    return "action_history"


def aligned_entries(base_cache, selected_cache) -> list[list[tuple[dict, dict]]]:
    base_snapshot = base_cache.snapshot()
    selected_snapshot = selected_cache.snapshot()
    if len(base_snapshot) != len(selected_snapshot):
        raise RuntimeError("base/selected cache layer counts differ")
    result = []
    for base_layer, selected_layer in zip(base_snapshot, selected_snapshot):
        base_map = {int(entry["frame_id"]): entry for entry in base_layer}
        selected_map = {int(entry["frame_id"]): entry for entry in selected_layer}
        if base_map.keys() != selected_map.keys():
            raise RuntimeError("base/selected cache frame identities differ")
        layer = []
        for frame_id in sorted(base_map):
            base_entry = base_map[frame_id]
            selected_entry = selected_map[frame_id]
            if bool(base_entry["is_pred"]) or bool(selected_entry["is_pred"]):
                raise RuntimeError("cache-teacher decomposition requires confirmed entries")
            for field in ("S", "z"):
                if (
                    base_entry[field].shape != selected_entry[field].shape
                    or base_entry[field].dtype != selected_entry[field].dtype
                ):
                    raise RuntimeError(f"cache entry contract differs for {field}")
            layer.append((base_entry, selected_entry))
        result.append(layer)
    return result


def build_category_cache(
    base_cache,
    selected_cache,
    *,
    mask: int,
    current_video_frame: int,
) -> ARLinearStateCache:
    layers = aligned_entries(base_cache, selected_cache)
    snapshot = []
    for layer in layers:
        entries = []
        for base_entry, selected_entry in layer:
            category = entry_category(
                int(base_entry["frame_id"]), current_video_frame
            )
            bit = 1 << CACHE_CATEGORIES.index(category)
            source = selected_entry if mask & bit else base_entry
            entries.append(source.copy())
        snapshot.append(entries)
    cache = ARLinearStateCache(base_cache.num_layers, base_cache.window)
    cache.restore_snapshot(snapshot)
    return cache


def build_interface_cache(
    base_cache,
    selected_cache,
    *,
    mask: int,
) -> ARLinearStateCache:
    layers = aligned_entries(base_cache, selected_cache)
    use_selected_s = bool(mask & 1)
    use_selected_z = bool(mask & 2)
    use_selected_history = bool(mask & 4)
    snapshot = []
    for layer in layers:
        entries = []
        for base_entry, selected_entry in layer:
            frame_id = int(base_entry["frame_id"])
            if frame_id % 2 == 1:
                source = selected_entry if use_selected_history else base_entry
                entries.append(source.copy())
                continue
            entries.append(
                {
                    "frame_id": frame_id,
                    "S": selected_entry["S"] if use_selected_s else base_entry["S"],
                    "z": selected_entry["z"] if use_selected_z else base_entry["z"],
                    "is_pred": False,
                }
            )
        snapshot.append(entries)
    cache = ARLinearStateCache(base_cache.num_layers, base_cache.window)
    cache.restore_snapshot(snapshot)
    return cache


def cache_signature(cache) -> list[list[tuple[int, int, int, float, float]]]:
    result = []
    for layer in cache.snapshot():
        result.append(
            [
                (
                    int(entry["frame_id"]),
                    int(entry["S"].data_ptr()),
                    int(entry["z"].data_ptr()),
                    float(entry["S"].double().sum().item()),
                    float(entry["z"].double().sum().item()),
                )
                for entry in layer
            ]
        )
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SANA cache-path action decomposition",
        "",
        f"- Task: `{report['task']}`",
        f"- Contexts: `{report['context_count']}`",
        f"- Validation: `{report['validation']['result']}`",
        "",
        "## Integrated category Shapley",
        "",
        "| Cache path | Mean MSE contribution | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for row in report["category"]["integrated"]["shapley"]:
        lines.append(
            f"| {row['player']} | {row['mean']:+.6f} | "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Integrated deployed-interface Shapley",
            "",
            "| Interface state | Mean MSE contribution | 95% CI |",
            "| --- | ---: | --- |",
        ]
    )
    for row in report["interface"]["integrated"]["shapley"]:
        lines.append(
            f"| {row['player']} | {row['mean']:+.6f} | "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |"
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
    group_keys = {
        group: matching_keys(list(model_state), pattern)
        for group, pattern in GROUP_PATTERNS.items()
    }
    base_checkpoint = args.base_run_dir / args.base_checkpoint
    base_tensors = cache_group_tensors(base_checkpoint, group_keys)
    overlay_tensors = cache_group_tensors(args.overlay_checkpoint, group_keys)
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
        generator = torch.Generator(device=device).manual_seed(
            args.seed + pair_index * 1009
        )
        noise_by_pair.append(
            torch.randn(
                saved["tensors"]["actions"].shape,
                generator=generator,
                device=device,
                dtype=architecture.dtype,
            ).cpu()
        )
    noise_sha = tensor_sha256(noise_by_pair)

    families = {
        "category": CACHE_CATEGORIES,
        "interface": INTERFACE_PLAYERS,
    }
    square_sums = {
        family: {
            mask: {sigma: {} for sigma in sigmas}
            for mask in range(1 << len(players))
        }
        for family, players in families.items()
    }
    counts = {
        family: {
            mask: {sigma: {} for sigma in sigmas}
            for mask in range(1 << len(players))
        }
        for family, players in families.items()
    }
    chunk_mse = {
        family: {
            mask: {sigma: {} for sigma in sigmas}
            for mask in range(1 << len(players))
        }
        for family, players in families.items()
    }
    maximum_duplicate_output_delta = 0.0
    all_cache_unchanged = True

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
        for chunk in range(chunks):
            video_proprio, action_proprio = architecture._rollout_proprio_deltas(
                proprio_chunks[chunk]
            )
            apply_groups(model_state, base_tensors, overlay_tensors, ())
            base_cache = build_expert_history(
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
                base_cache,
                video_chunks[chunk],
                frame_id=2 * chunk,
                context=context,
                context_mask=context_mask,
                v_proprio=video_proprio,
            )
            apply_groups(
                model_state,
                base_tensors,
                overlay_tensors,
                tuple(GROUP_PATTERNS),
            )
            selected_cache = build_expert_history(
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
                selected_cache,
                video_chunks[chunk],
                frame_id=2 * chunk,
                context=context,
                context_mask=context_mask,
                v_proprio=video_proprio,
            )
            aligned_entries(base_cache, selected_cache)
            start = chunk * tokens_per_chunk
            end = start + tokens_per_chunk
            action_chunk = action_chunks[chunk]
            noise_chunk = action_noise[:, start:end]
            valid = torch.ones_like(action_chunk, dtype=torch.bool)
            if "action_is_pad" in payload:
                valid &= (~payload["action_is_pad"][:, start:end]).unsqueeze(-1)
            for sigma in sigmas:
                noisy_action = (1.0 - sigma) * action_chunk + sigma * noise_chunk
                target = noise_chunk.float() - action_chunk.float()
                outputs: dict[str, dict[int, torch.Tensor]] = {
                    family: {} for family in families
                }
                for mask in range(8):
                    cache = build_category_cache(
                        base_cache,
                        selected_cache,
                        mask=mask,
                        current_video_frame=2 * chunk,
                    )
                    before = cache_signature(cache)
                    outputs["category"][mask] = query_action(
                        architecture,
                        driver,
                        cache,
                        noisy_action,
                        sigma=sigma,
                        frame_id=2 * chunk + 1,
                        context=context,
                        context_mask=context_mask,
                        action_proprio=action_proprio,
                    )
                    all_cache_unchanged &= before == cache_signature(cache)
                for mask in range(8):
                    cache = build_interface_cache(
                        base_cache, selected_cache, mask=mask
                    )
                    before = cache_signature(cache)
                    outputs["interface"][mask] = query_action(
                        architecture,
                        driver,
                        cache,
                        noisy_action,
                        sigma=sigma,
                        frame_id=2 * chunk + 1,
                        context=context,
                        context_mask=context_mask,
                        action_proprio=action_proprio,
                    )
                    all_cache_unchanged &= before == cache_signature(cache)
                for endpoint in (0, 7):
                    maximum_duplicate_output_delta = max(
                        maximum_duplicate_output_delta,
                        float(
                            (
                                outputs["category"][endpoint]
                                - outputs["interface"][endpoint]
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                for family, players in families.items():
                    for mask in range(1 << len(players)):
                        error = (outputs[family][mask] - target).double().square()
                        for item_index, item in enumerate(metadata):
                            episode = int(item["episode_index"])
                            item_mask = valid[item_index]
                            total = float(error[item_index][item_mask].sum().cpu())
                            count = int(item_mask.sum())
                            square_sums[family][mask][sigma][episode] = (
                                square_sums[family][mask][sigma].get(episode, 0.0)
                                + total
                            )
                            counts[family][mask][sigma][episode] = (
                                counts[family][mask][sigma].get(episode, 0) + count
                            )
                            if count:
                                chunk_mse[family][mask][sigma].setdefault(
                                    episode, {}
                                )[chunk] = total / count
            print(
                f"pair={pair_index + 1}/{args.pairs} chunk={chunk + 1}/{chunks}",
                flush=True,
            )

    context_mse = {
        family: {
            mask: {
                sigma: {
                    episode: square_sums[family][mask][sigma][episode]
                    / counts[family][mask][sigma][episode]
                    for episode in square_sums[family][mask][sigma]
                }
                for sigma in sigmas
            }
            for mask in range(1 << len(players))
        }
        for family, players in families.items()
    }
    episodes = sorted(context_mse["category"][0][sigmas[0]])

    def summarize_values(
        values_by_mask: dict[int, dict[int, float]],
        players: tuple[str, ...],
        label: str,
        episode_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        summary_episodes = episodes if episode_ids is None else episode_ids
        player_values = {player: [] for player in players}
        mobius_values = {mask: [] for mask in range(1 << len(players))}
        context_rows = []
        total_values = []
        residuals = []
        for episode in summary_episodes:
            values = {
                mask: values_by_mask[mask][episode]
                for mask in range(1 << len(players))
            }
            shapley_row = shapley(values, players)
            mobius_row = mobius(values, len(players))
            total = values[(1 << len(players)) - 1] - values[0]
            residual = sum(shapley_row.values()) - total
            total_values.append(total)
            residuals.append(residual)
            for player in players:
                player_values[player].append(shapley_row[player])
            for mask in mobius_values:
                mobius_values[mask].append(mobius_row[mask])
            context_rows.append(
                {
                    "episode_index": episode,
                    "coalition_mse": {
                        f"{mask:03b}": values[mask]
                        for mask in range(1 << len(players))
                    },
                    "shapley": shapley_row,
                    "mobius": {
                        f"{mask:03b}": mobius_row[mask]
                        for mask in range(1 << len(players))
                    },
                    "total_delta": total,
                    "efficiency_residual": residual,
                }
            )
        shapley_rows = []
        for player in players:
            values = np.asarray(player_values[player], dtype=np.float64)
            low, high = bootstrap_ci(
                values,
                seed=args.seed,
                label=f"{label}:shapley:{player}",
                samples=args.bootstrap_samples,
            )
            shapley_rows.append(
                {
                    "player": player,
                    "mean": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        mobius_rows = []
        for mask in range(1 << len(players)):
            values = np.asarray(mobius_values[mask], dtype=np.float64)
            low, high = bootstrap_ci(
                values,
                seed=args.seed,
                label=f"{label}:mobius:{mask}",
                samples=args.bootstrap_samples,
            )
            mobius_rows.append(
                {
                    "mask": f"{mask:03b}",
                    "players": list(coalition_names(mask, players)),
                    "mean": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        total = np.asarray(total_values, dtype=np.float64)
        total_low, total_high = bootstrap_ci(
            total,
            seed=args.seed,
            label=f"{label}:total",
            samples=args.bootstrap_samples,
        )
        return {
            "coalition_means": {
                f"{mask:03b}": float(
                    np.mean(
                        [
                            values_by_mask[mask][episode]
                            for episode in summary_episodes
                        ]
                    )
                )
                for mask in range(1 << len(players))
            },
            "total_delta_mean": float(total.mean()),
            "total_delta_ci95_low": total_low,
            "total_delta_ci95_high": total_high,
            "shapley": shapley_rows,
            "mobius": mobius_rows,
            "max_abs_efficiency_residual": float(
                np.max(np.abs(np.asarray(residuals)))
            ),
            "contexts": context_rows,
        }

    family_reports = {}
    for family, players in families.items():
        by_sigma = {}
        for sigma in sigmas:
            by_sigma[str(sigma)] = summarize_values(
                {
                    mask: context_mse[family][mask][sigma]
                    for mask in range(1 << len(players))
                },
                players,
                f"{family}:sigma:{sigma}",
            )
        integrated_values = {
            mask: {
                episode: float(
                    np.mean(
                        [context_mse[family][mask][sigma][episode] for sigma in sigmas]
                    )
                )
                for episode in episodes
            }
            for mask in range(1 << len(players))
        }
        chunk_reports = {}
        chunk_ids = sorted(
            {
                chunk
                for episode_chunks in chunk_mse[family][0][sigmas[0]].values()
                for chunk in episode_chunks
            }
        )
        for chunk in chunk_ids:
            sigma_reports = {}
            for sigma in sigmas:
                chunk_episodes = [
                    episode
                    for episode in episodes
                    if all(
                        chunk
                        in chunk_mse[family][mask][sigma].get(episode, {})
                        for mask in range(1 << len(players))
                    )
                ]
                sigma_reports[str(sigma)] = summarize_values(
                    {
                        mask: {
                            episode: chunk_mse[family][mask][sigma][episode][chunk]
                            for episode in chunk_episodes
                        }
                        for mask in range(1 << len(players))
                    },
                    players,
                    f"{family}:chunk:{chunk}:sigma:{sigma}",
                    chunk_episodes,
                )
            chunk_reports[str(chunk)] = sigma_reports
        family_reports[family] = {
            "players": list(players),
            "by_sigma": by_sigma,
            "integrated": summarize_values(
                integrated_values, players, f"{family}:integrated"
            ),
            "by_chunk_sigma": chunk_reports,
        }

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
                            context_mse["category"][mask][sigma][episode]
                            - float(expected[str(episode)])
                        )
                    )
        reference_validation = {
            "path": str(args.reference_json),
            "max_abs_context_mse_delta": max(differences),
        }
    efficiency_max = max(
        family_reports[family]["integrated"]["max_abs_efficiency_residual"]
        for family in families
    )
    validation_pass = (
        all_cache_unchanged
        and maximum_duplicate_output_delta == 0.0
        and efficiency_max <= 1e-12
        and (
            reference_validation is None
            or reference_validation["max_abs_context_mse_delta"] <= 1e-12
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
        "action_noise_sha256": noise_sha,
        "cache_contract": {
            "expert_endpoints_fixed": True,
            "cache_reencoded_per_video_weight_condition": True,
            "cache_entry_states": ["S_numerator", "z_normalizer"],
        },
        "validation": {
            "result": "PASS" if validation_pass else "FAIL",
            "all_query_caches_unchanged": all_cache_unchanged,
            "max_duplicate_endpoint_output_delta": maximum_duplicate_output_delta,
            "max_integrated_shapley_efficiency_residual": efficiency_max,
            "reference": reference_validation,
        },
        "category": family_reports["category"],
        "interface": family_reports["interface"],
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "action_cache_path_shapley.json"
    markdown_path = args.output_dir / "action_cache_path_shapley.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))
    if not validation_pass:
        raise RuntimeError("cache-path Shapley validation failed")


if __name__ == "__main__":
    main()
