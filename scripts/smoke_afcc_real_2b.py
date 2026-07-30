#!/usr/bin/env python3
"""One-row, backward-only real-2B AFCC diagnostic on H200."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

import precompute_phase1_action_reference as reference_support  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if _sha256_file(resolved) != expected:
        raise ValueError(f"{name} SHA256 differs")
    return resolved


def _exclusive_json(path: Path, value) -> None:
    from sana_wam.train.action_facing_cache_reference import canonical_json_bytes

    payload = canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        os.fchmod(descriptor, 0o400)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-step", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite smoke output: {args.output}")
    if not args.output.parent.is_dir():
        raise FileNotFoundError("smoke output parent must exist")
    config_path = _verify(args.config, args.config_sha256, "student config")
    manifest_path = _verify(
        args.reference_manifest,
        args.reference_manifest_sha256,
        "AFCC smoke reference manifest",
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version")
        != "sana-phase6-afcc-reference-manifest-smoke-v1"
        or manifest.get("row_count") != 1
    ):
        raise ValueError("real-2B smoke requires the one-row AFCC smoke reference")
    row = manifest["rows"][0]
    if row["global_step"] != args.global_step:
        raise ValueError("AFCC smoke reference global step differs")

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from sana_wam.config import flatten_model_cfg
    from sana_wam.model import build_architecture
    from sana_wam.train.phase6_arm_config import VIDEO_PATTERNS

    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("real-2B AFCC smoke requires one CUDA process")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    cfg = OmegaConf.load(config_path)
    cfg.model.architecture.action_facing_cache_consistency_weight = 1.0
    if (
        float(cfg.model.architecture.video_local_expansion_weight) != 1.0
        or float(cfg.model.architecture.action_non_regression_weight) != 0.0
        or cfg.model.architecture.action_video_memory_adapter.enabled is not False
        or cfg.model.video_backbone.continuous_timestep_conditioning is not True
        or tuple(cfg.training.trainable_parameter_patterns) != VIDEO_PATTERNS
    ):
        raise ValueError("student config is not the exact T1/E1/A0 control condition")
    reference_support.seed_reference_process(int(cfg.training.seed))
    dataset, plan, _dataset_contract = reference_support.build_bound_dataset(
        cfg,
        expected_source_manifest_sha256=args.source_manifest_sha256,
    )
    architecture = build_architecture(flatten_model_cfg(cfg.model))
    architecture.set_dtype_device(torch.bfloat16, device)
    checkpoint = _verify(
        Path(cfg.training.init_checkpoint),
        str(cfg.training.init_checkpoint_sha256),
        "student checkpoint",
    )
    architecture.load_checkpoint(checkpoint, strict=True)
    architecture.init_training_schedulers(1000)
    architecture.set_training_runtime(use_gradient_checkpointing=False)

    match_counts = {pattern: 0 for pattern in VIDEO_PATTERNS}
    trainables = []
    for name, parameter in architecture.named_parameters():
        matched = any(fnmatch.fnmatchcase(name, pattern) for pattern in VIDEO_PATTERNS)
        parameter.requires_grad_(matched)
        if matched:
            trainables.append((name, parameter))
            for pattern in VIDEO_PATTERNS:
                if fnmatch.fnmatchcase(name, pattern):
                    match_counts[pattern] += 1
    if len(trainables) != 380 or any(count == 0 for count in match_counts.values()):
        raise RuntimeError("real-2B AFCC trainable allowlist differs")
    architecture.train()
    architecture.video_backbone.vae.eval()
    architecture.video_backbone.text_encoder.eval()

    stats = dataset.action_stats
    stats = stats() if callable(stats) else stats
    architecture.action_mean.copy_(
        torch.from_numpy(stats["mean"].astype(np.float32))
    )
    architecture.action_std.copy_(
        torch.from_numpy(
            np.maximum(stats["std"].astype(np.float32), 1.0e-3)
        )
    )
    sample = dataset[plan.rows[args.global_step - 1].identity.dataset_index]
    inputs = architecture.prepare_inputs([sample])
    inputs["phase6_common_input_trace_required"] = True
    shard_path = Path(row["path"])
    _verify(shard_path, row["file_sha256"], "AFCC smoke reference shard")
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        reference = handle.get_tensor("action_facing_video_numerator")
    inputs["phase1_action_facing_cache_reference"] = (
        reference.unsqueeze(0).to(device=device).detach()
    )
    result = architecture.compute_loss(
        **inputs,
        lambda_video=1.0,
        lambda_action=0.0,
    )
    if result["phase6_common_input_trace_sha256"] != row[
        "common_input_trace_sha256"
    ]:
        raise RuntimeError("student common input/noise trace differs from Phase1")
    afcc = result["phase6_action_facing_cache_consistency_surrogate"]
    total = result["loss"]
    if (
        afcc.ndim != 0
        or total.ndim != 0
        or not bool(torch.isfinite(afcc))
        or not bool(torch.isfinite(total))
        or float(afcc.detach()) <= 0.0
    ):
        raise RuntimeError("real-2B AFCC losses are invalid")

    selected = [
        (name, parameter)
        for name, parameter in trainables
        if name.endswith("blocks.0.attn.qkv.weight")
        or name.endswith("blocks.19.attn.qkv.weight")
    ]
    if len(selected) != 2:
        raise RuntimeError("AFCC gradient probe parameter selection differs")
    selected_gradients = torch.autograd.grad(
        afcc,
        [parameter for _name, parameter in selected],
        retain_graph=True,
        allow_unused=True,
        materialize_grads=False,
    )
    afcc_gradient_probe = {}
    for (name, _parameter), gradient in zip(
        selected, selected_gradients, strict=True
    ):
        connected = (
            gradient is not None
            and bool(torch.isfinite(gradient).all())
            and bool(torch.count_nonzero(gradient))
        )
        if not connected:
            raise RuntimeError(f"AFCC source loss is not connected to {name}")
        afcc_gradient_probe[name] = float(
            torch.linalg.vector_norm(gradient.float()).item()
        )

    total.backward()
    finite = True
    nonzero_count = 0
    missing_count = 0
    squared_norms = []
    for _name, parameter in trainables:
        gradient = parameter.grad
        if gradient is None:
            missing_count += 1
            continue
        finite = finite and bool(torch.isfinite(gradient).all())
        norm = float(torch.linalg.vector_norm(gradient.float()).item())
        if norm > 0.0:
            nonzero_count += 1
        squared_norms.append(norm * norm)
    if not finite or nonzero_count == 0:
        raise RuntimeError("real-2B total gradients are non-finite or disconnected")
    frozen_gradient_count = sum(
        parameter.grad is not None
        for _name, parameter in architecture.named_parameters()
        if not parameter.requires_grad
    )
    if frozen_gradient_count != 0:
        raise RuntimeError("AFCC backward wrote gradients to frozen parameters")
    report = {
        "afcc_gradient_probe_norms": afcc_gradient_probe,
        "afcc_loss": float(afcc.detach().float().item()),
        "closed_loop": False,
        "config_sha256": args.config_sha256,
        "formal_evidence": False,
        "frozen_gradient_count": frozen_gradient_count,
        "global_step": args.global_step,
        "gradient_checkpointing_effective": False,
        "manifest_sha256": args.reference_manifest_sha256,
        "optimizer_created": False,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "schema_version": "sana-phase6-afcc-real-2b-smoke-v1",
        "source_manifest_sha256": args.source_manifest_sha256,
        "total_gradient_global_norm": math.sqrt(math.fsum(squared_norms)),
        "total_gradient_missing_tensor_count": missing_count,
        "total_gradient_nonzero_tensor_count": nonzero_count,
        "total_loss": float(total.detach().float().item()),
        "training_started": False,
        "wall_seconds": time.monotonic() - started,
    }
    _exclusive_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
