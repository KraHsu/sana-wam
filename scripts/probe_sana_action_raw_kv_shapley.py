#!/usr/bin/env python
"""Factor video cache effects into raw K, V, and action-history paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT if (ROOT / "src").is_dir() else ROOT.parent / "phase4"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Sana"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from einops import rearrange  # noqa: E402
from probe_sana_action_denoising import parse_sigmas, to_device  # noqa: E402
from probe_sana_action_weight_overlay import (  # noqa: E402
    GROUP_PATTERNS,
    apply_groups,
    cache_group_tensors,
    matching_keys,
)
from probe_sana_ar_decomposition import (  # noqa: E402
    ingest_clean_action,
    query_action,
)
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402
from sana_wam.model.ar.sana_ar_inference import (  # noqa: E402
    ARLinearStateCache,
    clean_state_from_tokens,
)


PLAYERS = (
    "video_rotated_K_plus_phiK",
    "video_V",
    "action_history",
)
KV_PLAYERS = PLAYERS[:2]
RawTriple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
StatePair = tuple[torch.Tensor, torch.Tensor]


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
    parser.add_argument(
        "--video-scope",
        choices=("all", "current", "prior"),
        default="all",
        help=(
            "Video entries whose K/V players are switched. Entries outside a "
            "current/prior scope remain at the Phase1 reconstruction."
        ),
    )
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


def tensors_finite(values: tuple[torch.Tensor, ...]) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in values)


def to_heads(value: torch.Tensor, heads: int) -> torch.Tensor:
    return rearrange(value, "b s (n d) -> b n s d", n=heads)


def _ingest_video_with_hook(
    architecture,
    driver,
    cache,
    clean: torch.Tensor,
    *,
    frame_id: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    video_proprio: torch.Tensor | None,
    on_layer: Callable[[int, RawTriple], None],
) -> None:
    """Observe production pre-attention tensors without replacing ingest math."""
    backbone = architecture.video_backbone
    original = backbone.pre_attn_at_layer
    had_override = "pre_attn_at_layer" in backbone.__dict__
    old_override = backbone.__dict__.get("pre_attn_at_layer")
    seen: set[int] = set()

    def hooked(_self, layer_id: int, state):
        if layer_id in seen:
            raise RuntimeError(f"video pre-attention layer {layer_id} ran twice")
        q, k, v, post = original(layer_id, state)
        if "k_unrot" not in post or not post.get("uses_linear_attn", False):
            raise RuntimeError("video pre-attention did not publish linear K tracks")
        on_layer(
            layer_id,
            (
                to_heads(k, driver.num_heads).detach(),
                to_heads(v, driver.num_heads).detach(),
                to_heads(post["k_unrot"], driver.num_heads).detach(),
            ),
        )
        seen.add(layer_id)
        return q, k, v, post

    object.__setattr__(
        backbone,
        "pre_attn_at_layer",
        types.MethodType(hooked, backbone),
    )
    try:
        architecture._ingest_clean_video(
            driver,
            cache,
            clean,
            frame_id=frame_id,
            context=context,
            context_mask=context_mask,
            v_proprio=video_proprio,
        )
    finally:
        if had_override:
            object.__setattr__(backbone, "pre_attn_at_layer", old_override)
        else:
            object.__delattr__(backbone, "pre_attn_at_layer")
    expected = set(range(backbone.num_layers))
    if seen != expected:
        raise RuntimeError(
            f"captured video layers {sorted(seen)}, expected {sorted(expected)}"
        )


def ingest_base_video_with_raw(
    architecture,
    driver,
    cache,
    clean: torch.Tensor,
    *,
    frame_id: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    video_proprio: torch.Tensor | None,
    finite_status: dict[str, bool],
) -> dict[int, RawTriple]:
    raw: dict[int, RawTriple] = {}

    def capture(layer_id: int, tensors: RawTriple) -> None:
        finite_status["raw_inputs"] &= tensors_finite(tensors)
        raw[layer_id] = tensors

    _ingest_video_with_hook(
        architecture,
        driver,
        cache,
        clean,
        frame_id=frame_id,
        context=context,
        context_mask=context_mask,
        video_proprio=video_proprio,
        on_layer=capture,
    )
    return raw


def ingest_selected_video_with_factorial(
    architecture,
    driver,
    cache,
    clean: torch.Tensor,
    *,
    frame_id: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    video_proprio: torch.Tensor | None,
    base_raw: dict[int, RawTriple],
    finite_status: dict[str, bool],
) -> dict[int, dict[int, StatePair]]:
    """Build all K/V states with the production state-construction helper."""
    factorial: dict[int, dict[int, StatePair]] = {}

    def combine(layer_id: int, selected: RawTriple) -> None:
        if layer_id not in base_raw:
            raise RuntimeError(f"missing Phase1 raw tensors for layer {layer_id}")
        base_k, base_v, base_phi_k = base_raw.pop(layer_id)
        selected_k, selected_v, selected_phi_k = selected
        finite_status["raw_inputs"] &= tensors_finite(selected)

        # Bit 0 switches both numerator K and denominator phi(K); bit 1 switches V.
        states = {
            0: clean_state_from_tokens(base_k, base_v, base_phi_k),
            1: clean_state_from_tokens(selected_k, base_v, selected_phi_k),
            2: clean_state_from_tokens(base_k, selected_v, base_phi_k),
            3: clean_state_from_tokens(selected_k, selected_v, selected_phi_k),
        }
        finite_status["reconstructed_states"] &= all(
            tensors_finite(state) for state in states.values()
        )
        factorial[layer_id] = states

    _ingest_video_with_hook(
        architecture,
        driver,
        cache,
        clean,
        frame_id=frame_id,
        context=context,
        context_mask=context_mask,
        video_proprio=video_proprio,
        on_layer=combine,
    )
    if base_raw:
        raise RuntimeError(f"unused Phase1 raw layers: {sorted(base_raw)}")
    return factorial


def frame_entry(cache, layer_id: int, frame_id: int) -> dict:
    matches = [
        entry
        for entry in cache.snapshot()[layer_id]
        if int(entry["frame_id"]) == frame_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one cache entry at layer={layer_id}, frame={frame_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def validate_reconstruction(
    base_cache,
    selected_cache,
    factorial: dict[int, dict[int, StatePair]],
    *,
    frame_id: int,
    stats: dict[str, Any],
) -> None:
    for layer_id, states in factorial.items():
        base_entry = frame_entry(base_cache, layer_id, frame_id)
        selected_entry = frame_entry(selected_cache, layer_id, frame_id)
        for condition, state, entry in (
            ("base", states[0], base_entry),
            ("selected", states[3], selected_entry),
        ):
            for index, field in enumerate(("S", "z")):
                exact = torch.equal(state[index], entry[field])
                stats[condition][f"{field}_exact_count"] += int(exact)
                stats[condition][f"{field}_max_abs_delta"] = max(
                    stats[condition][f"{field}_max_abs_delta"],
                    max_abs_delta(state[index], entry[field]),
                )
        stats["entry_count_per_condition"] += 1


def aligned_cache_entries(base_cache, selected_cache) -> list[list[tuple[dict, dict]]]:
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
                raise RuntimeError("raw K/V probe requires confirmed cache entries")
            for field in ("S", "z"):
                if (
                    base_entry[field].shape != selected_entry[field].shape
                    or base_entry[field].dtype != selected_entry[field].dtype
                ):
                    raise RuntimeError(f"cache entry contract differs for {field}")
            layer.append((base_entry, selected_entry))
        result.append(layer)
    return result


def frame_in_scope(frame_id: int, current_video_frame: int, scope: str) -> bool:
    if scope == "all":
        return frame_id % 2 == 0
    if scope == "current":
        return frame_id == current_video_frame
    if scope == "prior":
        return frame_id % 2 == 0 and frame_id < current_video_frame
    raise ValueError(f"unknown video scope {scope!r}")


def build_raw_factorial_cache(
    base_cache,
    selected_cache,
    raw_states: dict[int, dict[int, dict[int, StatePair]]],
    *,
    mask: int,
    current_video_frame: int,
    scope: str,
) -> ARLinearStateCache:
    layers = aligned_cache_entries(base_cache, selected_cache)
    kv_mask = mask & 3
    use_selected_history = bool(mask & 4)
    snapshot = []
    for layer_id, layer in enumerate(layers):
        entries = []
        for base_entry, selected_entry in layer:
            frame_id = int(base_entry["frame_id"])
            if frame_id % 2 == 1:
                source = selected_entry if use_selected_history else base_entry
                entries.append(source.copy())
                continue
            active_kv_mask = (
                kv_mask
                if frame_in_scope(frame_id, current_video_frame, scope)
                else 0
            )
            try:
                S, z = raw_states[frame_id][layer_id][active_kv_mask]
            except KeyError as error:
                raise RuntimeError(
                    f"missing raw state for frame={frame_id}, layer={layer_id}, "
                    f"kv_mask={active_kv_mask}"
                ) from error
            entries.append(
                {
                    "frame_id": frame_id,
                    "S": S,
                    "z": z,
                    "is_pred": False,
                }
            )
        snapshot.append(entries)
    cache = ARLinearStateCache(base_cache.num_layers, base_cache.window)
    cache.restore_snapshot(snapshot)
    return cache


def cache_signature(cache) -> list[list[tuple[Any, ...]]]:
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
        "# SANA raw video K/V controlled decomposition",
        "",
        f"- Task: `{report['task']}`",
        f"- Video scope: `{report['video_scope']}`",
        f"- Contexts: `{report['context_count']}`",
        f"- Validation: `{report['validation']['result']}`",
        "",
        "Hybrid K/V coalitions are internal off-manifold controlled effects, not "
        "deployable model variants. Switching K also switches phi(K) and z.",
        "",
        "## Integrated K/V/history Shapley",
        "",
        "| Player | Mean MSE contribution | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for row in report["full_factorial"]["integrated"]["shapley"]:
        lines.append(
            f"| {row['player']} | {row['mean']:+.6f} | "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |"
        )
    for background in ("base", "selected"):
        lines.extend(
            [
                "",
                f"## K/V factorial with {background} action history",
                "",
                "| Player | Mean MSE contribution | 95% CI |",
                "| --- | ---: | --- |",
            ]
        )
        rows = report["kv_conditioned_on_action_history"][background][
            "integrated"
        ]["shapley"]
        for row in rows:
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
    state_keys = list(model_state)
    group_keys = {
        group: matching_keys(state_keys, pattern)
        for group, pattern in GROUP_PATTERNS.items()
    }
    base_checkpoint = args.base_run_dir / args.base_checkpoint
    base_tensors = cache_group_tensors(base_checkpoint, group_keys)
    overlay_tensors = cache_group_tensors(args.overlay_checkpoint, group_keys)
    group_contract = {}
    seen_keys: set[str] = set()
    for group, keys in group_keys.items():
        overlap = seen_keys.intersection(keys)
        if overlap or not keys:
            raise RuntimeError(f"invalid parameter group {group}: overlap={overlap}")
        seen_keys.update(keys)
        group_contract[group] = {
            "pattern": GROUP_PATTERNS[group],
            "tensor_count": len(keys),
            "parameter_count": sum(model_state[key].numel() for key in keys),
            "changed_element_count": sum(
                int(
                    torch.count_nonzero(
                        base_tensors[group][key] != overlay_tensors[group][key]
                    )
                )
                for key in keys
            ),
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
    noise_sha_before = tensor_sha256(noise_by_pair)

    square_sums = {
        mask: {sigma: {} for sigma in sigmas} for mask in range(8)
    }
    counts = {mask: {sigma: {} for sigma in sigmas} for mask in range(8)}
    chunk_mse = {
        mask: {sigma: {} for sigma in sigmas} for mask in range(8)
    }
    finite_status = {
        "raw_inputs": True,
        "reconstructed_states": True,
        "action_outputs": True,
    }
    reconstruction_stats: dict[str, Any] = {
        "entry_count_per_condition": 0,
        "base": {
            "S_exact_count": 0,
            "z_exact_count": 0,
            "S_max_abs_delta": 0.0,
            "z_max_abs_delta": 0.0,
        },
        "selected": {
            "S_exact_count": 0,
            "z_exact_count": 0,
            "S_max_abs_delta": 0.0,
            "z_max_abs_delta": 0.0,
        },
    }
    all_query_caches_unchanged = True
    maximum_base_endpoint_output_delta = 0.0
    maximum_selected_endpoint_output_delta = 0.0

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
        base_cache = ARLinearStateCache(
            architecture.video_backbone.num_layers,
            int(architecture._ar_attn_window),
        )
        selected_cache = ARLinearStateCache(
            architecture.video_backbone.num_layers,
            int(architecture._ar_attn_window),
        )
        raw_states: dict[int, dict[int, dict[int, StatePair]]] = {}

        for chunk in range(chunks):
            current_video_frame = 2 * chunk
            video_proprio, action_proprio = architecture._rollout_proprio_deltas(
                proprio_chunks[chunk]
            )
            apply_groups(model_state, base_tensors, overlay_tensors, ())
            base_raw = ingest_base_video_with_raw(
                architecture,
                driver,
                base_cache,
                video_chunks[chunk],
                frame_id=current_video_frame,
                context=context,
                context_mask=context_mask,
                video_proprio=video_proprio,
                finite_status=finite_status,
            )
            apply_groups(
                model_state,
                base_tensors,
                overlay_tensors,
                tuple(GROUP_PATTERNS),
            )
            factorial = ingest_selected_video_with_factorial(
                architecture,
                driver,
                selected_cache,
                video_chunks[chunk],
                frame_id=current_video_frame,
                context=context,
                context_mask=context_mask,
                video_proprio=video_proprio,
                base_raw=base_raw,
                finite_status=finite_status,
            )
            raw_states[current_video_frame] = factorial
            validate_reconstruction(
                base_cache,
                selected_cache,
                factorial,
                frame_id=current_video_frame,
                stats=reconstruction_stats,
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
                target = noise_chunk.float() - action_chunk.float()
                outputs: dict[int, torch.Tensor] = {}
                for mask in range(8):
                    cache = build_raw_factorial_cache(
                        base_cache,
                        selected_cache,
                        raw_states,
                        mask=mask,
                        current_video_frame=current_video_frame,
                        scope=args.video_scope,
                    )
                    before = cache_signature(cache)
                    output = query_action(
                        architecture,
                        driver,
                        cache,
                        noisy_action,
                        sigma=sigma,
                        frame_id=current_video_frame + 1,
                        context=context,
                        context_mask=context_mask,
                        action_proprio=action_proprio,
                    )
                    all_query_caches_unchanged &= before == cache_signature(cache)
                    finite_status["action_outputs"] &= tensors_finite((output,))
                    outputs[mask] = output

                before = cache_signature(base_cache)
                base_direct = query_action(
                    architecture,
                    driver,
                    base_cache,
                    noisy_action,
                    sigma=sigma,
                    frame_id=current_video_frame + 1,
                    context=context,
                    context_mask=context_mask,
                    action_proprio=action_proprio,
                )
                all_query_caches_unchanged &= before == cache_signature(base_cache)
                maximum_base_endpoint_output_delta = max(
                    maximum_base_endpoint_output_delta,
                    max_abs_delta(outputs[0], base_direct),
                )

                before = cache_signature(selected_cache)
                selected_direct = query_action(
                    architecture,
                    driver,
                    selected_cache,
                    noisy_action,
                    sigma=sigma,
                    frame_id=current_video_frame + 1,
                    context=context,
                    context_mask=context_mask,
                    action_proprio=action_proprio,
                )
                all_query_caches_unchanged &= before == cache_signature(
                    selected_cache
                )
                maximum_selected_endpoint_output_delta = max(
                    maximum_selected_endpoint_output_delta,
                    max_abs_delta(outputs[7], selected_direct),
                )
                finite_status["action_outputs"] &= tensors_finite(
                    (base_direct, selected_direct)
                )

                for mask, output in outputs.items():
                    error = (output - target).double().square()
                    for item_index, item in enumerate(metadata):
                        episode = int(item["episode_index"])
                        item_mask = valid[item_index]
                        total = float(error[item_index][item_mask].sum().cpu())
                        count = int(item_mask.sum())
                        square_sums[mask][sigma][episode] = (
                            square_sums[mask][sigma].get(episode, 0.0) + total
                        )
                        counts[mask][sigma][episode] = (
                            counts[mask][sigma].get(episode, 0) + count
                        )
                        if count:
                            chunk_mse[mask][sigma].setdefault(episode, {})[
                                chunk
                            ] = total / count

            if chunk + 1 < chunks:
                apply_groups(model_state, base_tensors, overlay_tensors, ())
                ingest_clean_action(
                    architecture,
                    driver,
                    base_cache,
                    action_chunk,
                    frame_id=current_video_frame + 1,
                    context=context,
                    context_mask=context_mask,
                    action_proprio=action_proprio,
                )
                apply_groups(
                    model_state,
                    base_tensors,
                    overlay_tensors,
                    tuple(GROUP_PATTERNS),
                )
                ingest_clean_action(
                    architecture,
                    driver,
                    selected_cache,
                    action_chunk,
                    frame_id=current_video_frame + 1,
                    context=context,
                    context_mask=context_mask,
                    action_proprio=action_proprio,
                )
            print(
                f"pair={pair_index + 1}/{args.pairs} chunk={chunk + 1}/{chunks}",
                flush=True,
            )

    context_mse = {
        mask: {
            sigma: {
                episode: square_sums[mask][sigma][episode]
                / counts[mask][sigma][episode]
                for episode in square_sums[mask][sigma]
            }
            for sigma in sigmas
        }
        for mask in range(8)
    }
    episodes = sorted(context_mse[0][sigmas[0]])

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
        width = len(players)
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
                        f"{mask:0{width}b}": values[mask]
                        for mask in range(1 << len(players))
                    },
                    "shapley": shapley_row,
                    "mobius": {
                        f"{mask:0{width}b}": mobius_row[mask]
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
                    "mask": f"{mask:0{width}b}",
                    "players": list(coalition_names(mask, players)),
                    "mean": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        totals = np.asarray(total_values, dtype=np.float64)
        total_low, total_high = bootstrap_ci(
            totals,
            seed=args.seed,
            label=f"{label}:total",
            samples=args.bootstrap_samples,
        )
        return {
            "coalition_means": {
                f"{mask:0{width}b}": float(
                    np.mean(
                        [
                            values_by_mask[mask][episode]
                            for episode in summary_episodes
                        ]
                    )
                )
                for mask in range(1 << len(players))
            },
            "total_delta_mean": float(totals.mean()),
            "total_delta_ci95_low": total_low,
            "total_delta_ci95_high": total_high,
            "shapley": shapley_rows,
            "mobius": mobius_rows,
            "max_abs_efficiency_residual": float(
                np.max(np.abs(np.asarray(residuals)))
            ),
            "contexts": context_rows,
        }

    integrated_values = {
        mask: {
            episode: float(
                np.mean([context_mse[mask][sigma][episode] for sigma in sigmas])
            )
            for episode in episodes
        }
        for mask in range(8)
    }
    full_by_sigma = {
        str(sigma): summarize_values(
            {mask: context_mse[mask][sigma] for mask in range(8)},
            PLAYERS,
            f"full:sigma:{sigma}",
        )
        for sigma in sigmas
    }
    full_integrated = summarize_values(
        integrated_values, PLAYERS, "full:integrated"
    )
    chunk_reports = {}
    chunk_ids = sorted(
        {
            chunk
            for episode_chunks in chunk_mse[0][sigmas[0]].values()
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
                    chunk in chunk_mse[mask][sigma].get(episode, {})
                    for mask in range(8)
                )
            ]
            sigma_reports[str(sigma)] = summarize_values(
                {
                    mask: {
                        episode: chunk_mse[mask][sigma][episode][chunk]
                        for episode in chunk_episodes
                    }
                    for mask in range(8)
                },
                PLAYERS,
                f"full:chunk:{chunk}:sigma:{sigma}",
                chunk_episodes,
            )
        chunk_reports[str(chunk)] = sigma_reports

    conditioned_reports = {}
    conditioned_efficiency = []
    for history_name, history_bit in (("base", 0), ("selected", 4)):
        by_sigma = {}
        for sigma in sigmas:
            summary = summarize_values(
                {
                    kv_mask: context_mse[kv_mask | history_bit][sigma]
                    for kv_mask in range(4)
                },
                KV_PLAYERS,
                f"kv:{history_name}:sigma:{sigma}",
            )
            conditioned_efficiency.append(summary["max_abs_efficiency_residual"])
            by_sigma[str(sigma)] = summary
        integrated = summarize_values(
            {
                kv_mask: integrated_values[kv_mask | history_bit]
                for kv_mask in range(4)
            },
            KV_PLAYERS,
            f"kv:{history_name}:integrated",
        )
        conditioned_efficiency.append(integrated["max_abs_efficiency_residual"])
        conditioned_reports[history_name] = {
            "history_bit": int(bool(history_bit)),
            "by_sigma": by_sigma,
            "integrated": integrated,
        }

    reference_validation = None
    if args.reference_json is not None:
        reference = json.loads(args.reference_json.read_text())
        base_differences = []
        selected_differences = []
        for sigma in sigmas:
            expected_base = reference["context_metrics"]["phase1_video"][str(sigma)]
            expected_selected = reference["context_metrics"]["selected_video"][
                str(sigma)
            ]
            for episode in episodes:
                base_differences.append(
                    abs(
                        context_mse[0][sigma][episode]
                        - float(expected_base[str(episode)])
                    )
                )
                if args.video_scope == "all":
                    selected_differences.append(
                        abs(
                            context_mse[7][sigma][episode]
                            - float(expected_selected[str(episode)])
                        )
                    )
        reference_validation = {
            "path": str(args.reference_json),
            "base_000_max_abs_context_mse_delta": max(base_differences),
            "selected_111_applicable": args.video_scope == "all",
            "selected_111_max_abs_context_mse_delta": (
                max(selected_differences) if selected_differences else None
            ),
        }

    expected_entries = reconstruction_stats["entry_count_per_condition"]
    reconstruction_exact = all(
        reconstruction_stats[condition][f"{field}_exact_count"]
        == expected_entries
        for condition in ("base", "selected")
        for field in ("S", "z")
    )
    noise_sha_after = tensor_sha256(noise_by_pair)
    efficiency_max = max(
        [full_integrated["max_abs_efficiency_residual"], *conditioned_efficiency]
    )
    endpoint_output_pass = maximum_base_endpoint_output_delta == 0.0 and (
        args.video_scope != "all"
        or maximum_selected_endpoint_output_delta == 0.0
    )
    reference_pass = reference_validation is None or (
        reference_validation["base_000_max_abs_context_mse_delta"] <= 1e-12
        and (
            args.video_scope != "all"
            or reference_validation["selected_111_max_abs_context_mse_delta"]
            <= 1e-12
        )
    )
    validation_pass = (
        all(finite_status.values())
        and reconstruction_exact
        and all_query_caches_unchanged
        and noise_sha_before == noise_sha_after
        and endpoint_output_pass
        and efficiency_max <= 1e-12
        and reference_pass
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
        "video_scope": args.video_scope,
        "action_noise_sha256": noise_sha_before,
        "parameter_groups": group_contract,
        "controlled_effect_contract": {
            "expert_video_action_endpoints_fixed": True,
            "cache_reencoded_per_weight_condition": True,
            "K_switches_rotated_numerator_K_and_phiK_normalizer_together": True,
            "S_z_built_with_production_clean_state_from_tokens": True,
            "hybrid_K_V_states_are_off_manifold_internal_controls": True,
            "out_of_scope_video_entries": "phase1_reconstruction",
            "mask_bits": {
                "bit_0": PLAYERS[0],
                "bit_1": PLAYERS[1],
                "bit_2": PLAYERS[2],
            },
            "endpoint_meaning": {
                "000": "all Phase1 cache paths",
                "111": (
                    "all selected cache paths"
                    if args.video_scope == "all"
                    else "selected scoped K/V and history; Phase1 video outside scope"
                ),
            },
        },
        "validation": {
            "result": "PASS" if validation_pass else "FAIL",
            "finite": finite_status,
            "reconstruction": reconstruction_stats,
            "reconstruction_all_exact": reconstruction_exact,
            "all_query_caches_unchanged": all_query_caches_unchanged,
            "noise_unchanged": noise_sha_before == noise_sha_after,
            "base_000_direct_max_abs_output_delta": (
                maximum_base_endpoint_output_delta
            ),
            "selected_111_direct_applicable": args.video_scope == "all",
            "selected_111_direct_max_abs_output_delta": (
                maximum_selected_endpoint_output_delta
            ),
            "max_shapley_efficiency_residual": efficiency_max,
            "reference": reference_validation,
        },
        "full_factorial": {
            "players": list(PLAYERS),
            "by_sigma": full_by_sigma,
            "integrated": full_integrated,
            "by_chunk_sigma": chunk_reports,
        },
        "kv_conditioned_on_action_history": conditioned_reports,
        "total_seconds": time.perf_counter() - started,
    }
    json_path = args.output_dir / "action_raw_kv_shapley.json"
    markdown_path = args.output_dir / "action_raw_kv_shapley.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))
    if not validation_pass:
        raise RuntimeError("raw K/V Shapley validation failed")


if __name__ == "__main__":
    main()
