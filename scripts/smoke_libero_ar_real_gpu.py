#!/usr/bin/env python
"""Run one update-free full-2B AR forward on a fixed real LIBERO sample.

This is a source/model-construction smoke, not checkpoint validation, training,
or benchmark evaluation.  It loads the published SANA-Video, Wan VAE, and
Gemma assets from local absolute paths, fresh-initializes the LIBERO 7D action
and 8D proprio modules, prepares Spatial episode 0 / frame 0, and executes
exactly one deterministic AR forward under ``torch.inference_mode()``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))

SPATIAL_NAME = "libero_spatial_no_noops_1.0.0_lerobot"
PARQUET_SHA256 = "3f875604fad478765549128759edfb33a64b69b7b82decebc9e4f38155f20a8c"
DEFAULT_CONFIG = ROOT / "configs/benchmarks/libero/train_libero_ar_baseline.yaml"
DEFAULT_SEED = 20260806
MAX_IDLE_MEMORY_USED_MIB = 256
EXPECTED_BASE_PARTIAL_LOAD_PREFIX = (
    "SANA ckpt load partial: 280 missing, 0 unexpected keys."
)
EXPECTED_BASE_PARTIAL_LOAD_SAMPLES = (
    "blocks.0.learnable_fa_scale",
    "blocks.0.attn.kernel_func.alpha",
    "blocks.0.attn.kernel_func.w1.weight",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _repo_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _query_gpu(physical_index: int) -> dict[str, Any]:
    fields = (
        "index,uuid,name,pci.bus_id,memory.used,memory.free,memory.total,"
        "utilization.gpu"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_index}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one nvidia-smi GPU row, got {rows!r}")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 8:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {rows[0]!r}")
    return {
        "physical_index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "pci_bus_id": values[3],
        "memory_used_mib": int(values[4]),
        "memory_free_mib": int(values[5]),
        "memory_total_mib": int(values[6]),
        "utilization_percent": int(values[7]),
    }


def _selected_compute_processes(gpu_uuid: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for raw in completed.stdout.splitlines():
        values = [value.strip() for value in raw.split(",")]
        if len(values) != 4 or values[0] != gpu_uuid:
            continue
        rows.append(
            {
                "gpu_uuid": values[0],
                "pid": int(values[1]),
                "process_name": values[2],
                "used_gpu_memory_mib": int(values[3]),
            }
        )
    return rows


def _assert_idle_gpu(physical_index: int, expected_uuid: str) -> dict[str, Any]:
    gpu = _query_gpu(physical_index)
    if gpu["uuid"] != expected_uuid:
        raise RuntimeError(
            f"GPU UUID mismatch for physical index {physical_index}: "
            f"expected {expected_uuid}, got {gpu['uuid']}"
        )
    processes = _selected_compute_processes(expected_uuid)
    if processes:
        raise RuntimeError(f"selected GPU has compute processes: {processes!r}")
    if gpu["memory_used_mib"] > MAX_IDLE_MEMORY_USED_MIB:
        raise RuntimeError(
            "selected GPU is not idle: "
            f"memory_used_mib={gpu['memory_used_mib']}"
        )
    return gpu


def _find_fixed_sample(dataset):
    for dataset_index, (episode_position, start) in enumerate(dataset._windows):
        episode = dataset._episodes[episode_position]
        if (
            episode.dataset == SPATIAL_NAME
            and episode.episode_index == 0
            and start == 0
        ):
            return dataset_index, episode
    raise RuntimeError("fixed LIBERO Spatial episode 0 / start 0 was not found")


def _tensor_summary(tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(values.isfinite().all().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std().item()),
    }


class _WarningCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _write_report(path: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    print(payload, end="", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # Prevent implicit hub access before importing Transformers through the
    # SANA pipeline.  The configured model paths are all absolute local paths.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {str(args.physical_gpu), args.expected_gpu_uuid}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the selected physical GPU "
            f"({args.physical_gpu} or {args.expected_gpu_uuid}), got {visible!r}"
        )

    stats_candidate = args.stats.expanduser()
    config_candidate = args.config.expanduser()
    report_candidate = args.report.expanduser()
    for label, path in (("stats", stats_candidate), ("config", config_candidate)):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if report_candidate.exists() or report_candidate.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite smoke report: {report_candidate}"
        )
    stats_path = stats_candidate.resolve()
    config_path = config_candidate.resolve()
    report_path = report_candidate.resolve()

    gpu_lock_path = Path(f"/tmp/sana-wam-{args.expected_gpu_uuid}.lock")
    gpu_lock = gpu_lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"selected GPU lock is already held: {gpu_lock_path}") from exc

    gpu_before = _assert_idle_gpu(args.physical_gpu, args.expected_gpu_uuid)

    # Keep torch/model imports after the physical-GPU and offline checks.
    import torch

    from sana_wam.config import flatten_model_cfg
    from sana_wam.dataloader.libero_dataset import LiberoLeRobotDataset
    from sana_wam.model import build_architecture

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "smoke requires CUDA with exactly one visible GPU; "
            f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = OmegaConf.load(config_path)
    if cfg.model.architecture.variant != "autoregressive":
        raise RuntimeError("LIBERO smoke config must select autoregressive architecture")
    if cfg.model.architecture.action_dim != 7 or cfg.model.architecture.state_dim != 8:
        raise RuntimeError(
            "LIBERO smoke config must retain the 7D action / 8D state contract"
        )
    if cfg.model.architecture.ar_chunkwise_temporal_ops is not True:
        raise RuntimeError("LIBERO padded-tail smoke requires ar_chunkwise_temporal_ops=true")
    cfg.dataloader.action_stats_path = str(stats_path)

    sample_started = time.perf_counter()
    dataset = LiberoLeRobotDataset.from_config(cfg.dataloader, split="train")
    dataset_index, episode = _find_fixed_sample(dataset)
    if _sha256_file(episode.data_path()) != PARQUET_SHA256:
        raise RuntimeError("fixed LIBERO Spatial episode 0 Parquet SHA differs")
    sample = dataset[dataset_index]
    sample_seconds = time.perf_counter() - sample_started
    if sample["action_alignment"] != "observation_t_to_action_t":
        raise RuntimeError("LIBERO sample action alignment differs")

    # Recheck immediately before model construction; never take over an active GPU.
    gpu_preconstruction = _assert_idle_gpu(
        args.physical_gpu, args.expected_gpu_uuid
    )
    torch.cuda.reset_peak_memory_stats(device)
    warning_capture = _WarningCapture()
    builder_logger = logging.getLogger(
        "sana_wam.model.video_backbone.sana.pipeline_builder"
    )
    builder_logger.addHandler(warning_capture)
    build_started = time.perf_counter()
    try:
        flat_cfg = flatten_model_cfg(cfg.model)
        OmegaConf.update(flat_cfg, "video_backbone._device", "cuda:0", merge=False)
        architecture = build_architecture(flat_cfg)
        architecture.set_dtype_device(torch.bfloat16, device)
    finally:
        builder_logger.removeHandler(warning_capture)
    build_seconds = time.perf_counter() - build_started
    torch.cuda.synchronize(device)
    build_memory = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    partial_load_warnings = [
        message
        for message in warning_capture.messages
        if "ckpt load partial" in message.lower()
    ]
    expected_partial_load = (
        len(partial_load_warnings) == 1
        and partial_load_warnings[0].startswith(EXPECTED_BASE_PARTIAL_LOAD_PREFIX)
        and all(
            sample in partial_load_warnings[0]
            for sample in EXPECTED_BASE_PARTIAL_LOAD_SAMPLES
        )
    )
    if not expected_partial_load:
        raise RuntimeError(
            "SANA base checkpoint partial-load identity differs from the known "
            "fresh AR additions: "
            f"{warning_capture.messages!r}"
        )

    video_backbone = architecture.video_backbone
    pipe = video_backbone._pipe
    if any(
        value is None
        for value in (pipe.dit, pipe.vae, pipe.text_encoder, pipe.tokenizer)
    ):
        raise RuntimeError("SANA DiT/VAE/Gemma/tokenizer construction was incomplete")
    meta_parameters = [
        name for name, parameter in architecture.named_parameters()
        if parameter.device.type == "meta"
    ]
    if meta_parameters:
        raise RuntimeError(f"architecture contains meta parameters: {meta_parameters[:8]}")

    architecture.requires_grad_(False)
    architecture.eval()
    if any(parameter.requires_grad for parameter in architecture.parameters()):
        raise RuntimeError("update-free smoke failed to freeze every parameter")
    parameter_versions = tuple(
        parameter._version for parameter in architecture.parameters()
    )
    parameter_count = sum(parameter.numel() for parameter in architecture.parameters())

    torch.cuda.reset_peak_memory_stats(device)
    prepare_started = time.perf_counter()
    with torch.inference_mode():
        inputs = architecture.prepare_inputs([sample])
    torch.cuda.synchronize(device)
    prepare_seconds = time.perf_counter() - prepare_started
    prepare_memory = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }

    clean_video = inputs["input_latents"].to(device=device, dtype=torch.bfloat16)
    clean_actions = inputs["actions"].to(device=device, dtype=torch.bfloat16)
    batch_size, _channels, latent_frames, _latent_h, _latent_w = clean_video.shape
    action_tokens = clean_actions.shape[1]
    frame_chunk_size = int(cfg.model.architecture.ar_frame_chunk_size)
    if latent_frames % frame_chunk_size:
        raise RuntimeError(
            f"latent T={latent_frames} is not divisible by chunk size {frame_chunk_size}"
        )
    num_chunks = latent_frames // frame_chunk_size
    if action_tokens % num_chunks:
        raise RuntimeError(
            f"action tokens={action_tokens} are not divisible by chunks={num_chunks}"
        )
    action_tokens_per_chunk = action_tokens // num_chunks

    generator = torch.Generator(device=device).manual_seed(args.seed)
    video_noise = torch.randn(
        clean_video.shape, generator=generator, device=device, dtype=torch.bfloat16
    )
    action_noise = torch.randn(
        clean_actions.shape, generator=generator, device=device, dtype=torch.bfloat16
    )
    sigma = 0.5
    noisy_video = clean_video * (1.0 - sigma) + video_noise * sigma
    noisy_actions = clean_actions * (1.0 - sigma) + action_noise * sigma
    video_timestep_dtype = architecture._video_timestep_dtype(torch.bfloat16)
    video_timesteps = torch.full(
        (batch_size, latent_frames),
        500.0,
        device=device,
        dtype=video_timestep_dtype,
    )
    action_timesteps = torch.full(
        (batch_size, action_tokens), 500.0, device=device, dtype=torch.bfloat16
    )
    if bool(cfg.model.architecture.ar_bootstrap_clean_prefix):
        noisy_video[:, :, 0] = clean_video[:, :, 0]
        video_timesteps[:, 0] = 0

    proprio_seq = inputs["proprio_seq"].to(device=device, dtype=torch.bfloat16)
    proprio_indices = [chunk * action_tokens_per_chunk for chunk in range(num_chunks)]
    proprio_per_chunk = proprio_seq[:, proprio_indices]

    torch.cuda.reset_peak_memory_stats(device)
    forward_started = time.perf_counter()
    with torch.inference_mode():
        video_prediction, action_prediction = architecture(
            noisy_actions,
            None,
            proprio_state=inputs["proprio_state"],
            proprio_per_chunk=proprio_per_chunk,
            latents=noisy_video,
            ar_clean_latents=clean_video,
            ar_clean_actions=clean_actions,
            ar_video_frame_timesteps=video_timesteps,
            ar_clean_video_frame_timesteps=torch.zeros_like(video_timesteps),
            ar_action_token_timesteps=action_timesteps,
            ar_clean_action_token_timesteps=torch.zeros_like(action_timesteps),
            ar_video_is_pad=inputs.get("video_is_pad"),
            ar_action_is_pad=inputs.get("action_is_pad"),
            ar_frame_chunk_size=frame_chunk_size,
            ar_attn_window=int(cfg.model.architecture.ar_attn_window),
            timestep=video_timesteps.mean(dim=1),
            context=inputs["context"],
            seq_lens=inputs["seq_lens"],
        )
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    forward_memory = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }

    if video_prediction.shape != clean_video.shape:
        raise RuntimeError(
            f"video prediction shape differs: {tuple(video_prediction.shape)}"
        )
    if action_prediction is None or action_prediction.shape != clean_actions.shape:
        shape = None if action_prediction is None else tuple(action_prediction.shape)
        raise RuntimeError(f"action prediction shape differs: {shape}")
    video_summary = _tensor_summary(video_prediction)
    action_summary = _tensor_summary(action_prediction)
    if not video_summary["finite"] or not action_summary["finite"]:
        raise RuntimeError(
            "single AR forward produced non-finite output: "
            f"video={video_summary['finite']} action={action_summary['finite']}"
        )
    if (
        tuple(parameter._version for parameter in architecture.parameters())
        != parameter_versions
    ):
        raise RuntimeError("a model parameter version changed during update-free smoke")

    gpu_after = _query_gpu(args.physical_gpu)
    report = {
        "architecture": {
            "action_backbone": type(architecture.action_backbone).__name__,
            "action_dim": architecture.action_dim,
            "action_head_initialization": "fresh_seeded_no_checkpoint",
            "class": type(architecture).__name__,
            "frame_chunk_size": frame_chunk_size,
            "latent_chunks": num_chunks,
            "parameter_count": parameter_count,
            "proprio_dim": architecture.proprio_dim,
            "requires_grad_parameter_count": 0,
            "video_backbone": type(video_backbone).__name__,
            "video_fresh_initialization": {
                "base_checkpoint_missing_key_count": 280,
                "scope": "ar_learnable_feature_map_and_window_attention_additions",
            },
            "video_pretrained_asset_loaded": True,
        },
        "assets": {
            "gemma_path": str(cfg.model.video_backbone.text_encoder_name),
            "sana_path": str(cfg.model.video_backbone.model_path),
            "stats_path": str(stats_path),
            "stats_sha256": _sha256_file(stats_path),
        },
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "after": gpu_after,
            "before": gpu_before,
            "cuda_visible_devices": visible,
            "lock_path": str(gpu_lock_path),
            "preconstruction": gpu_preconstruction,
            "torch_cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "visible_ordinal": 0,
        },
        "inputs": {
            "action_is_pad_true": int(inputs["action_is_pad"].sum().item()),
            "action_tokens_per_chunk": action_tokens_per_chunk,
            "actions": list(clean_actions.shape),
            "context": list(inputs["context"].shape),
            "input_latents": list(clean_video.shape),
            "proprio_per_chunk": list(proprio_per_chunk.shape),
            "proprio_seq": list(proprio_seq.shape),
            "proprio_state": list(inputs["proprio_state"].shape),
            "video_is_pad_true": int(inputs["video_is_pad"].sum().item()),
        },
        "memory": {
            "build": build_memory,
            "forward": forward_memory,
            "prepare": prepare_memory,
        },
        "outputs": {
            "action_prediction": action_summary,
            "video_prediction": video_summary,
        },
        "repo_commit": _repo_commit(),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "result": "PASS",
        "sample": {
            "action_alignment": sample["action_alignment"],
            "dataset": episode.dataset,
            "episode_index": episode.episode_index,
            "episode_length": episode.length,
            "parquet_sha256": PARQUET_SHA256,
            "start_frame": sample["start_frame"],
            "task": sample["task_name"],
        },
        "schema_version": "sana-wam-libero-ar-real-gpu-update-free-smoke-v1",
        "scope": {
            "backward_executed": False,
            "benchmark_evaluation_executed": False,
            "architecture_forward_calls": 1,
            "optimizer_created": False,
            "parameter_update_executed": False,
            "sana_base_pretrained_checkpoint_loaded": True,
            "sana_wam_training_checkpoint_loaded": False,
            "sana_wam_training_checkpoint_saved": False,
            "simulator_executed": False,
            "training_executed": False,
        },
        "seed": args.seed,
        "timings_seconds": {
            "model_build": build_seconds,
            "model_forward": forward_seconds,
            "prepare_real_sample": prepare_seconds,
            "sample_load": sample_seconds,
        },
        "warnings": warning_capture.messages,
    }
    _write_report(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
