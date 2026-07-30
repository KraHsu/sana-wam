#!/usr/bin/env python
"""Verify the production video helper against the Phase-5 FP32-master oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT if (ROOT / "src").is_dir() else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Sana"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from probe_sana_action_denoising import to_device  # noqa: E402
from probe_sana_ar_decomposition import build_expert_history  # noqa: E402
from probe_sana_video_euler_trace import tensor_sha256, trace_one  # noqa: E402
from sana_wam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair-index", type=int, required=True)
    parser.add_argument("--context-index", type=int, required=True)
    parser.add_argument("--expected-episode", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    _, architecture = load_from_checkpoint_dir(
        str(args.run_dir), device=args.device, ckpt_name=args.checkpoint
    )
    architecture.eval()
    driver = architecture._mot_driver or architecture.build_mot_driver()
    device = torch.device(args.device)

    prepared_path = args.prepared_dir / f"prepared_pair_{args.pair_index:02d}.pt"
    saved = torch.load(prepared_path, map_location="cpu", weights_only=False)
    metadata = saved["samples"][args.context_index]
    if int(metadata["episode_index"]) != args.expected_episode:
        raise RuntimeError("registered episode does not match the prepared pair")
    payload = to_device(saved["tensors"], device, architecture.dtype)

    manual = trace_one(
        architecture,
        driver,
        payload=payload,
        pair_index=args.pair_index,
        context_index=args.context_index,
        chunk=args.chunk,
        steps=args.steps,
        common_progress=10,
        mode="fp32_accum",
        repeat=0,
        seed=args.seed,
    )
    manual_model_endpoint = manual.endpoint_full.to(architecture.dtype).float()

    actions = payload["actions"]
    latents = payload["input_latents"]
    frame_chunk_size = int(architecture._ar_frame_chunk_size)
    chunks = latents.shape[2] // frame_chunk_size
    tokens_per_chunk = actions.shape[1] // chunks
    video_chunks = list(latents.split(frame_chunk_size, dim=2))
    action_chunks = list(actions.split(tokens_per_chunk, dim=1))
    proprio_chunks = [
        payload["proprio_seq"][:, index * tokens_per_chunk] for index in range(chunks)
    ]
    context, context_mask = architecture._rollout_step_context(
        payload["context"], payload["seq_lens"], payload["proprio_state"]
    )
    production_cache = build_expert_history(
        architecture,
        driver,
        video_chunks,
        action_chunks,
        proprio_chunks,
        stop_chunk=args.chunk,
        context=context,
        context_mask=context_mask,
    )
    cache_before = production_cache.info()
    video_proprio, _ = architecture._rollout_proprio_deltas(proprio_chunks[args.chunk])

    architecture.video_backbone.scheduler.set_timesteps(args.steps)
    sigmas = [
        float(value) for value in architecture.video_backbone.scheduler.sigmas.tolist()
    ] + [0.0]
    timesteps = architecture.video_backbone.scheduler.timesteps
    derived_seed = args.seed + args.pair_index * 1009 + args.chunk * 100_003

    expected_generator = torch.Generator(device=device).manual_seed(derived_seed)
    expected_noise = torch.randn(
        video_chunks[args.chunk].shape,
        generator=expected_generator,
        device=device,
        dtype=architecture.dtype,
    )
    expected_generator_state = expected_generator.get_state().clone()

    production_generator = torch.Generator(device=device).manual_seed(derived_seed)
    production_output = architecture._denoise_video_chunk(
        driver,
        production_cache,
        frame_id=2 * args.chunk,
        like=video_chunks[args.chunk],
        v_sigmas=sigmas,
        v_ts=timesteps,
        context=context,
        context_mask=context_mask,
        gen=production_generator,
        v_proprio=video_proprio,
    )
    production_cpu = production_output.float().cpu()
    cache_after = production_cache.info()

    delta = production_cpu - manual_model_endpoint
    endpoint_equal = torch.equal(production_cpu, manual_model_endpoint)
    rng_equal = torch.equal(production_generator.get_state(), expected_generator_state)
    noise_hash = tensor_sha256(expected_noise)
    cache_delta = cache_after["entries"] - cache_before["entries"]
    cache_ok = (
        cache_delta == architecture.video_backbone.num_layers
        and cache_after["predicted_entries"] - cache_before["predicted_entries"]
        == architecture.video_backbone.num_layers
    )

    result = {
        "protocol": "Phase-6 production FP32 video Euler master-state verification",
        "pass": bool(
            endpoint_equal
            and rng_equal
            and manual.cache_unchanged
            and manual.noise_sha256 == noise_hash
            and cache_ok
        ),
        "case": {
            "pair_index": args.pair_index,
            "context_index": args.context_index,
            "episode_index": args.expected_episode,
            "chunk": args.chunk,
            "steps": args.steps,
            "seed": args.seed,
            "derived_seed": derived_seed,
        },
        "contract": {
            "model_dtype": str(architecture.dtype),
            "production_return_dtype": str(production_output.dtype),
            "manual_mode": "bf16_forward_fp32_accumulation",
            "manual_endpoint_compared": "final_model_state",
            "endpoint_tensor_equal": endpoint_equal,
            "endpoint_delta_rms": float(delta.double().square().mean().sqrt().item()),
            "endpoint_delta_abs_max": float(delta.abs().max().item()),
            "production_endpoint_sha256": tensor_sha256(production_output),
            "manual_model_endpoint_sha256": tensor_sha256(
                manual_model_endpoint.to(production_output.dtype)
            ),
            "noise_sha256": noise_hash,
            "manual_noise_sha256": manual.noise_sha256,
            "rng_after_single_draw_equal": rng_equal,
            "manual_query_cache_unchanged": manual.cache_unchanged,
            "production_final_cache_write_ok": cache_ok,
            "production_cache_before": cache_before,
            "production_cache_after": cache_after,
        },
        "artifacts": {
            "prepared_pair": str(prepared_path),
            "prepared_pair_sha256": sha256_file(prepared_path),
            "checkpoint": str(args.run_dir / args.checkpoint),
            "checkpoint_sha256": sha256_file(args.run_dir / args.checkpoint),
            "architecture_source_sha256": sha256_file(
                REPO_ROOT / "src/sana_wam/model/architecture.py"
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "training_run": False,
        "closed_loop_run": False,
    }
    output = args.output_dir / "phase6_fp32_master_contract.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"], "output": str(output)}))
    if not result["pass"]:
        raise RuntimeError("production FP32 master-state contract verification failed")


if __name__ == "__main__":
    main()
