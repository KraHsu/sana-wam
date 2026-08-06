#!/usr/bin/env python
"""Run one non-formal full-2B AR update on a fixed real LIBERO sample.

The runner is deliberately narrower than ``Trainer.train``: it constructs the
normal LIBERO ``Trainer``, prepares one fixed sample, computes one loss, runs
one backward and one AdamW step, writes one exclusive JSON report, and exits.
It never creates or saves a SANA-WAM checkpoint and never enters a simulator or
benchmark loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
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

from smoke_libero_ar_real_gpu import (  # noqa: E402
    DEFAULT_CONFIG,
    EXPECTED_BASE_PARTIAL_LOAD_PREFIX,
    EXPECTED_BASE_PARTIAL_LOAD_SAMPLES,
    PARQUET_SHA256,
    SPATIAL_NAME,
    _assert_idle_gpu,
    _find_fixed_sample,
    _query_gpu,
    _repo_commit,
    _sha256_file,
    _write_report,
)


NONFORMAL_METADATA_STATS_SHA256 = (
    "333b2cb1e150b451ee1cf6833b1e628ebb50914a466f2cb1fa1808cff8e9f2d0"
)
EXPECTED_SANA_COMMIT = "16b9cec673e3335724ba2d8db25de7f9ed229292"
EXPECTED_SANA_ASSET_SHA256 = {
    "checkpoints/SANA_Video_2B_480p.pth": (
        "052c4022488949153fe6400631725bcb8c739362a6d34edeab26236d8a23f1e7"
    ),
    "config.json": (
        "d34650e1e7b0ab502051e8fe2e829ed2c93bbcad0aa9fa3ab46574491b3e8288"
    ),
    "vae/Wan2.1_VAE.pth": (
        "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
    ),
}
EXPECTED_GEMMA_ASSET_SHA256 = {
    "config.json": (
        "eacec6c5ca317a87ed2c46789d9705b9274db5027e7ba59da739bfae23addb55"
    ),
    "model-00001-of-00002.safetensors": (
        "532d792c9178805064170a3ec485b7dedbfccc6fd297b92c31a6091b6c7e41bf"
    ),
    "model-00002-of-00002.safetensors": (
        "6d6d9ce84db398fb6e0191f91542e5da0a73da2cb695e172a24edc2146dc8d20"
    ),
    "model.safetensors.index.json": (
        "ada0043f3e3b2e5ab2f445cad9c0fbbf9d91ad444675e6a82b822591c63abf5a"
    ),
    "special_tokens_map.json": (
        "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351"
    ),
    "tokenizer.json": (
        "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922"
    ),
    "tokenizer.model": (
        "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"
    ),
    "tokenizer_config.json": (
        "cb32b7929c62608d46572e813112b3ad8a841fb98fdd6a4da8559e368a951c89"
    ),
}
EXPECTED_SPATIAL_SAMPLE_ASSET_SHA256 = {
    "data/chunk-000/episode_000000.parquet": PARQUET_SHA256,
    "meta/episodes.jsonl": (
        "690349688e96d984007bf2cca3ccf2683dfc809f03dab55e1159e4cc71f150b7"
    ),
    "meta/info.json": (
        "8995d3b6f7b723137fe7273eba41e9356be8b3a9176ec11678c13c7215676c85"
    ),
    "meta/tasks.jsonl": (
        "399841cf8e861052d3fb7214ed619b5a10b4eded2d6f107e7fed338a233dd0e1"
    ),
    "videos/chunk-000/observation.images.image/episode_000000.mp4": (
        "eba9a9b36611f7e1061f65233b3b0c53eaa3f9c347dfd073f7470a821695fdf7"
    ),
    "videos/chunk-000/observation.images.wrist_image/episode_000000.mp4": (
        "cf4f97b9405e6e06d42f3816c902a7cec22ba0497dc4ea3fa201729302ba4168"
    ),
}
EXPECTED_TRAINABLE_ROOTS = (
    "action_backbone",
    "proprio_encoder",
    "proprio_video_embed",
    "proprio_action_embed",
)
_ACTIVE_RUN_ROOT: Path | None = None


def _parameter_root(name: str) -> str:
    return name.split(".", 1)[0]


def _tensor_scalar(value: Any) -> float:
    return float(value.detach().float().item())


def _optimizer_state_is_finite(optimizer: Any) -> bool:
    import torch

    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not bool(
                torch.isfinite(value).all().item()
            ):
                return False
    return True


def _capture_update_probes(named_trainables):
    """Capture high-gradient scalars without cloning complete parameters."""

    import torch

    probes = []
    for name, parameter in named_trainables:
        gradient = parameter.grad
        if gradient is None or not bool(torch.count_nonzero(gradient).item()):
            continue
        flat_gradient = gradient.detach().reshape(-1)
        count = min(8, flat_gradient.numel())
        indices = flat_gradient.abs().topk(count, sorted=False).indices.tolist()
        for index in indices:
            probes.append(
                {
                    "before": _tensor_scalar(
                        parameter.detach().reshape(-1)[index]
                    ),
                    "gradient": _tensor_scalar(flat_gradient[index]),
                    "index": index,
                    "name": name,
                    "parameter": parameter,
                }
            )
    return probes


def _summarize_updates(probes) -> tuple[dict[str, Any], set[int]]:
    changed = []
    changed_parameter_ids = set()
    for probe in probes:
        after = _tensor_scalar(
            probe["parameter"].detach().reshape(-1)[probe["index"]]
        )
        if after != probe["before"]:
            changed_parameter_ids.add(id(probe["parameter"]))
            changed.append(
                {
                    "after": after,
                    "before": probe["before"],
                    "gradient": probe["gradient"],
                    "index": probe["index"],
                    "name": probe["name"],
                }
            )
    return (
        {
            "changed_probe_count": len(changed),
            "changed_probe_samples": changed[:8],
            "changed_trainable_roots": sorted(
                {_parameter_root(row["name"]) for row in changed}
            ),
            "probe_count": len(probes),
        },
        changed_parameter_ids,
    )


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_output(arguments: list[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_source_tree_clean(expected_commit: str) -> str:
    if _git_output(["rev-parse", "HEAD"]) != expected_commit:
        raise RuntimeError("repository HEAD changed during source preflight")
    tracked_status = _git_output(
        [
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--ignore-submodules=untracked",
        ]
    )
    if tracked_status:
        raise RuntimeError(
            "one-update smoke requires a clean tracked worktree: "
            f"{tracked_status!r}"
        )
    gitlink = _git_output(["rev-parse", "HEAD:third_party/Sana"])
    sana_root = ROOT / "third_party" / "Sana"
    sana_head = _git_output(["rev-parse", "HEAD"], cwd=sana_root)
    sana_status = _git_output(
        ["status", "--porcelain=v1", "--untracked-files=no"], cwd=sana_root
    )
    if gitlink != EXPECTED_SANA_COMMIT or sana_head != EXPECTED_SANA_COMMIT:
        raise RuntimeError(
            "Sana gitlink/worktree commit differs: "
            f"gitlink={gitlink} worktree={sana_head}"
        )
    if sana_status:
        raise RuntimeError(
            f"Sana tracked worktree is dirty: {sana_status!r}"
        )
    return sana_head


def _verify_pinned_asset(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"pinned asset must be a regular non-symlink file: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"pinned asset SHA differs for {path}: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "size_bytes": path.stat().st_size,
    }


def _verify_external_assets(cfg: Any) -> dict[str, dict[str, Any]]:
    roots = {
        "gemma": Path(str(cfg.model.video_backbone.text_encoder_name)),
        "sana": Path(str(cfg.model.video_backbone.model_path)),
        "spatial_sample": Path(str(cfg.dataloader.dataset_roots[0])),
    }
    expected_root_names = {
        "gemma": "gemma-2-2b-it",
        "sana": "SANA-Video_2B_480p",
        "spatial_sample": SPATIAL_NAME,
    }
    for label, root in roots.items():
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(
                f"{label} asset root must be a non-symlink directory: {root}"
            )
        if root.name != expected_root_names[label]:
            raise RuntimeError(f"unexpected {label} asset root: {root}")

    checkpoint_dir = roots["sana"] / "checkpoints"
    observed_checkpoint_names = sorted(
        child.name
        for child in checkpoint_dir.iterdir()
        if child.name.endswith(".pth")
    )
    if observed_checkpoint_names != ["SANA_Video_2B_480p.pth"]:
        raise RuntimeError(
            "SANA bundle checkpoint candidates differ: "
            f"{observed_checkpoint_names!r}"
        )

    manifest = {
        "gemma": EXPECTED_GEMMA_ASSET_SHA256,
        "sana": EXPECTED_SANA_ASSET_SHA256,
        "spatial_sample": EXPECTED_SPATIAL_SAMPLE_ASSET_SHA256,
    }
    verified = {}
    for label, expected_files in manifest.items():
        root = roots[label]
        for relative_path, expected_sha256 in expected_files.items():
            key = f"{label}/{relative_path}"
            verified[key] = _verify_pinned_asset(
                root / relative_path, expected_sha256
            )
    return dict(sorted(verified.items()))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_run_root(candidate: Path) -> Path:
    if not candidate.is_absolute():
        raise ValueError("run root must be an absolute path")
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"refusing to reuse run root: {candidate}")
    parent = candidate.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"run-root parent must be a non-symlink directory: {parent}")
    root = parent.resolve() / candidate.name
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    _fsync_directory(root.parent)
    return root


def _freeze_run_root(root: Path) -> None:
    for child in root.iterdir():
        if child.is_symlink() or not child.is_file():
            raise RuntimeError(f"unexpected run-root entry: {child}")
        child.chmod(0o400)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o500)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(root.parent)


def _terminalize_failure(root: Path, error: BaseException) -> None:
    marker = root / "FAILED.json"
    if not marker.exists() and not marker.is_symlink():
        _write_report(
            marker,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "error_type": type(error).__name__,
                "result": "FAIL",
                "schema_version": (
                    "sana-wam-libero-ar-real-gpu-one-update-failure-v1"
                ),
            },
        )
    _freeze_run_root(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--accept-nonformal-metadata-stats",
        action="store_true",
        help="Acknowledge that the fixed stats artifact is not training admission.",
    )
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

    if not args.accept_nonformal_metadata_stats:
        raise RuntimeError(
            "one-update smoke requires explicit non-formal metadata-stats "
            "acknowledgement"
        )
    if args.seed < 0:
        raise ValueError("seed must be non-negative")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {str(args.physical_gpu), args.expected_gpu_uuid}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the selected physical GPU "
            f"({args.physical_gpu} or {args.expected_gpu_uuid}), got {visible!r}"
        )

    stats_candidate = args.stats.expanduser()
    config_candidate = args.config.expanduser()
    runner_path = Path(__file__).resolve()
    run_root_candidate = args.run_root.expanduser()
    for label, path in (("stats", stats_candidate), ("config", config_candidate)):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    stats_path = stats_candidate.resolve()
    config_path = config_candidate.resolve()
    run_root = _create_run_root(run_root_candidate)
    global _ACTIVE_RUN_ROOT
    _ACTIVE_RUN_ROOT = run_root

    actual_commit = _repo_commit()
    actual_config_sha256 = _sha256_file(config_path)
    actual_runner_sha256 = _sha256_file(runner_path)
    actual_stats_sha256 = _sha256_file(stats_path)
    expected_identity = {
        "config_sha256": args.expected_config_sha256.lower(),
        "repo_commit": args.expected_repo_commit.lower(),
        "runner_sha256": args.expected_runner_sha256.lower(),
        "sana_commit": EXPECTED_SANA_COMMIT,
        "stats_sha256": NONFORMAL_METADATA_STATS_SHA256,
    }
    actual_identity = {
        "config_sha256": actual_config_sha256,
        "repo_commit": actual_commit,
        "runner_sha256": actual_runner_sha256,
        "sana_commit": EXPECTED_SANA_COMMIT,
        "stats_sha256": actual_stats_sha256,
    }
    if actual_identity != expected_identity:
        raise RuntimeError(
            "one-update smoke source identity differs: "
            f"expected={expected_identity!r} actual={actual_identity!r}"
        )
    actual_identity["sana_commit"] = _assert_source_tree_clean(actual_commit)

    cfg = OmegaConf.load(config_path)
    if cfg.model.architecture.variant != "autoregressive":
        raise RuntimeError("LIBERO update smoke requires the AR architecture")
    if cfg.model.video_backbone.continuous_timestep_conditioning is not True:
        raise RuntimeError("LIBERO update smoke requires explicit T1=true")
    if tuple(cfg.training.trainable_modules) != EXPECTED_TRAINABLE_ROOTS:
        raise RuntimeError("LIBERO trainable module allowlist differs")
    if cfg.training.use_gradient_checkpointing is not True:
        raise RuntimeError("LIBERO update smoke requires gradient checkpointing")
    if OmegaConf.select(cfg, "training.init_checkpoint", default=None) is not None:
        raise RuntimeError("LIBERO update smoke forbids a SANA-WAM init checkpoint")
    if (
        OmegaConf.select(
            cfg, "model.video_backbone.init_dit_from", default=None
        )
        is not None
    ):
        raise RuntimeError(
            "LIBERO update smoke forbids video_backbone.init_dit_from"
        )
    if int(cfg.training.seed) != args.seed:
        raise RuntimeError(
            "CLI seed must equal training.seed so fresh initialization and "
            "loss noise share one recorded seed"
        )
    if float(cfg.training.lambda_video) != 0.0:
        raise RuntimeError("LIBERO update smoke requires lambda_video=0")
    if float(cfg.training.lambda_action) != 1.0:
        raise RuntimeError("LIBERO update smoke requires lambda_action=1")
    external_assets = _verify_external_assets(cfg)
    actual_identity["external_assets_sha256"] = _sha256_json(external_assets)

    import fcntl

    lock_path = Path(f"/tmp/sana-wam-{args.expected_gpu_uuid}.lock")
    gpu_lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"selected GPU lock is already held: {lock_path}") from exc
    gpu_before = _assert_idle_gpu(args.physical_gpu, args.expected_gpu_uuid)

    import torch

    from sana_wam.train.libero_contract import validate_libero_training_config
    from sana_wam.train.trainer import Trainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "smoke requires CUDA with exactly one visible GPU; "
            f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # The source config deliberately points to an absent future exact-statistics
    # artifact.  This in-memory override is scoped to the non-formal one-update
    # process and is recorded in the result.
    if cfg.training.action_stats_sha256 is not None:
        raise RuntimeError("source config must leave production stats SHA null")
    cfg.dataloader.action_stats_path = str(stats_path)
    validate_libero_training_config(cfg, require_materialized_stats=False)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    warning_messages: list[str] = []

    class WarningCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warning_messages.append(record.getMessage())

    warning_capture = WarningCapture(level=logging.WARNING)
    builder_logger = logging.getLogger(
        "sana_wam.model.video_backbone.sana.pipeline_builder"
    )
    builder_logger.addHandler(warning_capture)
    gpu_preconstruction = _assert_idle_gpu(
        args.physical_gpu, args.expected_gpu_uuid
    )
    torch.cuda.reset_peak_memory_stats(device)
    build_started = time.perf_counter()
    try:
        trainer = Trainer(cfg)
    finally:
        builder_logger.removeHandler(warning_capture)
    torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - build_started
    build_memory = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
    }

    partial_load_warnings = [
        message for message in warning_messages if "ckpt load partial" in message.lower()
    ]
    if not (
        len(partial_load_warnings) == 1
        and partial_load_warnings[0].startswith(EXPECTED_BASE_PARTIAL_LOAD_PREFIX)
        and all(
            sample in partial_load_warnings[0]
            for sample in EXPECTED_BASE_PARTIAL_LOAD_SAMPLES
        )
    ):
        raise RuntimeError(
            "SANA base checkpoint partial-load identity differs from the known "
            f"fresh AR additions: {warning_messages!r}"
        )

    architecture = trainer.architecture
    video_backbone = architecture.video_backbone
    if video_backbone.continuous_timestep_conditioning is not True:
        raise RuntimeError("live video backbone did not retain T1")
    if trainer._trainable_module_allowlist != EXPECTED_TRAINABLE_ROOTS:
        raise RuntimeError("live Trainer trainable allowlist differs")

    named_trainables = [
        (name, parameter)
        for name, parameter in architecture.named_parameters()
        if parameter.requires_grad
    ]
    if not named_trainables:
        raise RuntimeError("Trainer selected no trainable parameters")
    observed_roots = tuple(sorted({_parameter_root(name) for name, _ in named_trainables}))
    if observed_roots != tuple(sorted(EXPECTED_TRAINABLE_ROOTS)):
        raise RuntimeError(
            f"trainable parameter roots differ: observed={observed_roots!r}"
        )
    named_frozen = [
        (name, parameter)
        for name, parameter in architecture.named_parameters()
        if not parameter.requires_grad
    ]
    frozen_versions = {
        name: parameter._version for name, parameter in named_frozen
    }
    trainable_versions = {
        name: parameter._version for name, parameter in named_trainables
    }

    dataset_index, episode = _find_fixed_sample(trainer.dataset)
    if episode.dataset != SPATIAL_NAME or _sha256_file(
        episode.data_path()
    ) != PARQUET_SHA256:
        raise RuntimeError("fixed real LIBERO sample identity differs")
    sample_started = time.perf_counter()
    sample = trainer.dataset[dataset_index]
    sample_seconds = time.perf_counter() - sample_started
    if sample["action_alignment"] != "observation_t_to_action_t":
        raise RuntimeError("LIBERO action alignment differs")

    trainer._set_training_mode()
    torch.cuda.reset_peak_memory_stats(device)
    prepare_started = time.perf_counter()
    with torch.no_grad():
        inputs = architecture.prepare_inputs([sample])
    torch.cuda.synchronize(device)
    prepare_seconds = time.perf_counter() - prepare_started
    prepare_memory = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
    }
    if inputs.get("use_gradient_checkpointing") is not True:
        raise RuntimeError("prepared inputs dropped gradient checkpointing")
    if inputs.get("use_gradient_checkpointing_offload") is not False:
        raise RuntimeError("update smoke requires checkpoint offload=false")

    video_timestep_observations: list[dict[str, Any]] = []
    action_timestep_dtypes: set[str] = set()
    architecture_forward_calls = 0
    video_embedder = video_backbone._dit.t_embedder
    original_video_embedder_forward = video_embedder.forward
    original_action_prepare_state = architecture.action_backbone.prepare_state
    original_architecture_forward = architecture.forward

    def capture_video_timestep(timestep):
        values = timestep.detach().float()
        fractional = (values - values.round()).abs() > 1.0e-6
        video_timestep_observations.append(
            {
                "dtype": str(timestep.dtype),
                "fractional_count": int(fractional.sum().item()),
                "max": float(values.max().item()),
                "min": float(values.min().item()),
                "shape": list(timestep.shape),
            }
        )
        return original_video_embedder_forward(timestep)

    def capture_action_timestep(*positional, **keyword):
        if len(positional) >= 2 and isinstance(positional[1], torch.Tensor):
            action_timestep_dtypes.add(str(positional[1].dtype))
        token_timestep = keyword.get("token_timesteps")
        if isinstance(token_timestep, torch.Tensor):
            action_timestep_dtypes.add(str(token_timestep.dtype))
        return original_action_prepare_state(*positional, **keyword)

    def capture_architecture_forward(*positional, **keyword):
        nonlocal architecture_forward_calls
        architecture_forward_calls += 1
        return original_architecture_forward(*positional, **keyword)

    video_embedder.forward = capture_video_timestep
    architecture.action_backbone.prepare_state = capture_action_timestep
    architecture.forward = capture_architecture_forward

    model_groups = trainer._param_groups()
    if len(model_groups) != 2:
        raise RuntimeError(
            f"expected action/proprio optimizer groups, got {len(model_groups)}"
        )
    if any(float(group["lr"]) <= 0 for group in model_groups):
        raise RuntimeError("one-update probe requires positive base learning rates")
    params = [parameter for group in model_groups for parameter in group["params"]]
    parameter_ids = [id(parameter) for parameter in params]
    trainable_ids = [id(parameter) for _name, parameter in named_trainables]
    if (
        len(parameter_ids) != len(set(parameter_ids))
        or len(parameter_ids) != len(trainable_ids)
        or set(parameter_ids) != set(trainable_ids)
    ):
        raise RuntimeError("optimizer parameter set differs from trainable parameters")
    optimizer_groups, master_pairs = trainer._optimizer_param_groups(model_groups)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(cfg.training.weight_decay),
        betas=(0.9, 0.95),
    )

    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    update_started = time.perf_counter()
    try:
        result = architecture.compute_loss(
            lambda_video=trainer.lambda_video,
            lambda_action=trainer.lambda_action,
            **inputs,
        )
        loss = result["loss"]
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("one-update smoke produced a non-finite scalar loss")
        loss.backward()
    finally:
        video_embedder.forward = original_video_embedder_forward
        architecture.action_backbone.prepare_state = original_action_prepare_state
        architecture.forward = original_architecture_forward

    if architecture_forward_calls != 1:
        raise RuntimeError(
            "one-update smoke expected exactly one architecture forward, "
            f"got {architecture_forward_calls}"
        )

    missing_gradients = []
    nonzero_gradient_tensors = 0
    max_gradient_abs = 0.0
    nonzero_gradient_roots: set[str] = set()
    for name, parameter in named_trainables:
        gradient = parameter.grad
        if gradient is None:
            missing_gradients.append(name)
            continue
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError(f"non-finite gradient: {name}")
        maximum = float(gradient.detach().abs().max().float().item())
        max_gradient_abs = max(max_gradient_abs, maximum)
        if maximum > 0.0:
            nonzero_gradient_tensors += 1
            nonzero_gradient_roots.add(_parameter_root(name))
    if nonzero_gradient_tensors == 0 or not math.isfinite(max_gradient_abs):
        raise RuntimeError("one-update smoke produced no finite nonzero gradients")
    if nonzero_gradient_roots != set(EXPECTED_TRAINABLE_ROOTS):
        raise RuntimeError(
            "nonzero-gradient roots differ: "
            f"{sorted(nonzero_gradient_roots)!r}"
        )
    frozen_gradients = [
        name for name, parameter in named_frozen if parameter.grad is not None
    ]
    if frozen_gradients:
        raise RuntimeError(
            f"frozen parameters received gradients: {frozen_gradients[:8]!r}"
        )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        params, float(cfg.training.grad_clip)
    )
    if not bool(torch.isfinite(grad_norm).item()):
        raise RuntimeError("gradient norm is non-finite")
    probes = _capture_update_probes(named_trainables)
    if not probes:
        raise RuntimeError("no nonzero-gradient update probes were captured")
    if master_pairs:
        trainer._sync_master_gradients(master_pairs)
    optimizer.step()
    if master_pairs:
        trainer._copy_master_parameters_to_model(master_pairs)
    nonfinite_trainables = [
        name
        for name, parameter in named_trainables
        if not bool(torch.isfinite(parameter).all().item())
    ]
    if nonfinite_trainables:
        raise RuntimeError(
            "AdamW produced non-finite trainable parameters: "
            f"{nonfinite_trainables[:8]!r}"
        )
    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - update_started
    update_memory = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
    }

    update_summary, changed_parameter_ids = _summarize_updates(probes)
    if update_summary["changed_probe_count"] == 0:
        raise RuntimeError(
            "AdamW completed but no max-gradient parameter probe changed"
        )
    unchanged_optimizer_groups = [
        index
        for index, group in enumerate(model_groups)
        if not any(id(parameter) in changed_parameter_ids for parameter in group["params"])
    ]
    if unchanged_optimizer_groups:
        raise RuntimeError(
            "optimizer groups had no observed numeric parameter change: "
            f"{unchanged_optimizer_groups!r}"
        )
    if not _optimizer_state_is_finite(optimizer):
        raise RuntimeError("AdamW state contains non-finite values")

    frozen_version_changes = [
        name
        for name, parameter in named_frozen
        if parameter._version != frozen_versions[name]
    ]
    if frozen_version_changes:
        raise RuntimeError(
            "frozen parameters changed version: "
            f"{frozen_version_changes[:8]!r}"
        )
    trainable_version_change_count = sum(
        parameter._version != trainable_versions[name]
        for name, parameter in named_trainables
    )
    if trainable_version_change_count == 0:
        raise RuntimeError("no trainable parameter version changed")

    if not video_timestep_observations:
        raise RuntimeError("T1 video timestep embedder was never called")
    if {row["dtype"] for row in video_timestep_observations} != {
        "torch.float32"
    }:
        raise RuntimeError("T1 video timestep embedder observed a non-FP32 value")
    if sum(
        row["fractional_count"] for row in video_timestep_observations
    ) == 0:
        raise RuntimeError("T1 embedder observed no fractional video timestep")
    if action_timestep_dtypes != {"torch.bfloat16"}:
        raise RuntimeError(
            f"action timestep dtypes differ: {sorted(action_timestep_dtypes)!r}"
        )

    gpu_after_update = _query_gpu(args.physical_gpu)
    result_summary = {
        key: _tensor_scalar(value)
        for key, value in result.items()
        if hasattr(value, "numel") and value.numel() == 1
    }
    report = {
        "architecture": {
            "action_dim": architecture.action_dim,
            "class": type(architecture).__name__,
            "continuous_timestep_conditioning": True,
            "parameter_count": sum(
                parameter.numel() for parameter in architecture.parameters()
            ),
            "proprio_dim": architecture.proprio_dim,
            "trainable_parameter_count": sum(
                parameter.numel() for _name, parameter in named_trainables
            ),
            "trainable_parameter_tensor_count": len(named_trainables),
            "trainable_roots": list(EXPECTED_TRAINABLE_ROOTS),
            "video_backbone_frozen": all(
                not parameter.requires_grad
                for parameter in architecture.video_backbone.parameters()
            ),
        },
        "assets": {
            "config_path": str(config_path),
            "config_sha256": actual_config_sha256,
            "external_files": external_assets,
            "external_files_manifest_sha256": actual_identity[
                "external_assets_sha256"
            ],
            "runtime_stats_path_override": True,
            "source_training_action_stats_sha256": None,
            "stats_admission": "NONFORMAL_METADATA_BOOTSTRAP_ONLY",
            "stats_path": str(stats_path),
            "stats_sha256": actual_stats_sha256,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gradient": {
            "pre_clip_global_norm": _tensor_scalar(grad_norm),
            "pre_clip_max_abs": max_gradient_abs,
            "missing_gradient_count": len(missing_gradients),
            "missing_gradient_samples": missing_gradients[:8],
            "nonzero_gradient_tensor_count": nonzero_gradient_tensors,
            "nonzero_gradient_roots": sorted(nonzero_gradient_roots),
        },
        "gpu": {
            "after_update": gpu_after_update,
            "before": gpu_before,
            "cuda_visible_devices": visible,
            "lock_path": str(lock_path),
            "preconstruction": gpu_preconstruction,
            "torch_cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "visible_ordinal": 0,
        },
        "identity": {
            **actual_identity,
            "identity_sha256": _sha256_json(actual_identity),
        },
        "inputs": {
            "action_is_pad_true": int(inputs["action_is_pad"].sum().item()),
            "actions": list(inputs["actions"].shape),
            "context": list(inputs["context"].shape),
            "input_latents": list(inputs["input_latents"].shape),
            "proprio_seq": list(inputs["proprio_seq"].shape),
            "proprio_state": list(inputs["proprio_state"].shape),
            "video_is_pad_true": int(inputs["video_is_pad"].sum().item()),
        },
        "losses": result_summary,
        "memory": {
            "build": build_memory,
            "prepare": prepare_memory,
            "update": update_memory,
        },
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
        "schema_version": "sana-wam-libero-ar-real-gpu-one-update-smoke-v1",
        "scope": {
            "architecture_loss_forward_calls": architecture_forward_calls,
            "backward_executed": True,
            "benchmark_evaluation_executed": False,
            "formal_training_executed": False,
            "optimizer_steps": 1,
            "parameter_update_executed": True,
            "sana_base_pretrained_checkpoint_loaded": True,
            "sana_wam_training_checkpoint_loaded": False,
            "sana_wam_training_checkpoint_saved": False,
            "simulator_executed": False,
        },
        "seed": {
            "initialization_and_loss": args.seed,
            "training_config": int(cfg.training.seed),
        },
        "timesteps": {
            "action_dtypes": sorted(action_timestep_dtypes),
            "video_embedder_call_count": len(video_timestep_observations),
            "video_observations": video_timestep_observations,
        },
        "timings_seconds": {
            "model_and_dataset_build": build_seconds,
            "prepare_real_sample": prepare_seconds,
            "sample_load": sample_seconds,
            "single_update": update_seconds,
        },
        "update": {
            **update_summary,
            "optimizer": "AdamW",
            "optimizer_master_parameter_count": len(master_pairs),
            "probe_learning_rates": [
                float(group["lr"]) for group in model_groups
            ],
            "production_accumulation_equivalent": False,
            "production_gradient_accumulation_steps": int(
                cfg.training.gradient_accumulation_steps
            ),
            "production_schedule_equivalent": False,
            "production_warmup_steps": int(cfg.training.warmup_steps),
            "trainable_parameters_finite": True,
            "trainable_version_change_count": trainable_version_change_count,
        },
        "warnings": warning_messages,
    }
    _write_report(run_root / "RESULT.json", report)
    _freeze_run_root(run_root)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as error:
        if _ACTIVE_RUN_ROOT is not None:
            try:
                _terminalize_failure(_ACTIVE_RUN_ROOT, error)
            except BaseException as freeze_error:
                print(
                    "failed to terminalize one-update smoke root: "
                    f"{type(freeze_error).__name__}: {freeze_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    raise SystemExit(exit_code)
