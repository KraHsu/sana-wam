#!/usr/bin/env python
"""Run the bounded LIBERO T3 held-out loss-recipe transfer screen."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
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
)
from smoke_libero_ar_real_update_gpu import (  # noqa: E402
    EXPECTED_SPATIAL_SAMPLE_ASSET_SHA256,
    EXPECTED_PRESERVE_FROZEN_INPUT_GRAD_MODULES,
    EXPECTED_SANA_COMMIT,
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    EXPECTED_TRAINABLE_ROOTS,
    EXPECTED_TRAINABLE_TENSOR_COUNT,
    _assert_source_tree_clean,
    _capture_update_probes,
    _create_run_root,
    _freeze_run_root,
    _optimizer_state_is_finite,
    _parameter_root,
    _sha256_json,
    _summarize_updates,
    _tensor_scalar,
    _verify_external_assets,
    _verify_pinned_asset,
)


T3_UPDATE_STEPS = 20
T3_LEARNABILITY_RATIO = 0.95
T3_INITIALIZATION_SEED = 20260806
T3_LOSS_RECIPE_SEED = 20260826
T3_HELDOUT_RECIPE_SEEDS = (897500337, 2142397805, 1238092489)
T3_HELDOUT_PAYLOAD_SHA256 = (
    "08c9698412591ea0f219f56d2f293482c06d2303fb70c2de85f41a314f64d856",
    "ef4652c542991a4ab1b7fc1d031e27ca0cda41d8cc509696458b83c8d1b8827c",
    "f24a4b3b80a29dcc8b8dd1bf36cfcc5d747a69dd3ed50889112c6c3a76f055cd",
)
T3_HELDOUT_MEDIAN_RATIO = 0.95
T3_HELDOUT_MIN_IMPROVED = 2
T3_EPISODE_LENGTH = 110
T3_TASK_INDEX = 0
T3_TASK_NAME = "pick up the black bowl next to the cookie box and place it on the plate"
SELECTED_STATS_SHA256 = (
    "e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7"
)
SELECTED_POPULATION_SHA256 = (
    "7ed9772facf261299022e55169bcdaaa49fe3a7a20e0057e08cf419b5e584146"
)
T1_PREDECESSOR_RESULT_SHA256 = (
    "9d9c139fd67baa8c131ad9e6537862536b8262b732c2fda668a7c8b8b6a8621a"
)
T1_PREDECESSOR_RESULT_PATH = Path(
    "/tmp/sana-wam-libero-t1-one-update-fp32master-0337e28-20260806-a3/RESULT.json"
)
T2_PREDECESSOR_RESULT_SHA256 = (
    "67d250cdb13470bea9e9fa53531d3c76c65145e5040dc7a45079d84ac4960a57"
)
T2_PREDECESSOR_RESULT_PATH = Path(
    "/DATA/share/sana_wam_libero_nonformal_screens/t2/2dc1ce730df3/"
    "libero-t2-fixed20-20260806-a1/RESULT.json"
)
T2_PREDECESSOR_SOURCE_COMMIT = "2dc1ce730df37bd9a2a71e4946f153380a8d4649"
T2_PREDECESSOR_RUNNER_SHA256 = (
    "f5e5e3e80d6b5afd586e54e83a7e63dcf43280192363c70580aa03145267e9a6"
)
T2_PREDECESSOR_RUNNER_PATH = ROOT / (
    "scripts/smoke_libero_ar_t2_fixed_sample_20step_gpu.py"
)
SPATIAL_STATS_GR00T_SHA256 = (
    "0a4b08f5afcdcbe186ec70ea6ff2569233bab706a4d99ed603b23a7688bc33bb"
)
_ACTIVE_RUN_ROOT: Path | None = None


def _heldout_seed_manifest() -> list[dict[str, Any]]:
    if (
        len(set(T3_HELDOUT_RECIPE_SEEDS)) != 3
        or T3_LOSS_RECIPE_SEED in T3_HELDOUT_RECIPE_SEEDS
    ):
        raise RuntimeError("T3 held-out seed set is not three distinct unseen seeds")
    rows = []
    for index, (expected_seed, expected_payload_sha256) in enumerate(
        zip(
            T3_HELDOUT_RECIPE_SEEDS,
            T3_HELDOUT_PAYLOAD_SHA256,
            strict=True,
        ),
        start=1,
    ):
        payload = (
            "SANA-WAM/LIBERO/T3_HELDOUT_RECIPE_V1\n"
            f"T2_RESULT_SHA256={T2_PREDECESSOR_RESULT_SHA256}\n"
            f"INDEX={index}\n"
        ).encode("ascii")
        digest = hashlib.sha256(payload).digest()
        payload_sha256 = digest.hex()
        seed = 1 + (int.from_bytes(digest[:8], "big") % 2147483646)
        if seed != expected_seed or payload_sha256 != expected_payload_sha256:
            raise RuntimeError("T3 held-out seed derivation differs")
        rows.append(
            {
                "derivation_index": index,
                "payload_sha256": payload_sha256,
                "seed": seed,
            }
        )
    return rows


@contextlib.contextmanager
def _fixed_rng_recipe(torch_module: Any, *, seed: int, cuda_device: int | None):
    """Reset one loss recipe while restoring the caller's RNG states."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [] if cuda_device is None else [cuda_device]
    try:
        with torch_module.random.fork_rng(devices=devices, enabled=True):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch_module.manual_seed(seed)
            if cuda_device is not None:
                torch_module.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _learnability_summary(
    initial_action_loss: float,
    pre_update_action_losses: list[float],
    post_update_action_loss: float,
) -> dict[str, Any]:
    if len(pre_update_action_losses) != T3_UPDATE_STEPS:
        raise ValueError(f"T3 requires exactly {T3_UPDATE_STEPS} pre-update losses")
    values = [initial_action_loss, *pre_update_action_losses, post_update_action_loss]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("T3 losses must be finite and non-negative")
    if initial_action_loss <= 0.0:
        raise ValueError("T3 initial action loss must be strictly positive")
    first_five_median = statistics.median(pre_update_action_losses[:5])
    last_five_median = statistics.median(pre_update_action_losses[-5:])
    ratio = post_update_action_loss / initial_action_loss
    return {
        "first_five_median": first_five_median,
        "initial_measurement": initial_action_loss,
        "initial_pre_update": pre_update_action_losses[0],
        "last_five_median": last_five_median,
        "post_update": post_update_action_loss,
        "post_to_initial_ratio": ratio,
        "relative_drop": 1.0 - ratio,
        "threshold_ratio": T3_LEARNABILITY_RATIO,
        "loss_gate": (
            post_update_action_loss <= T3_LEARNABILITY_RATIO * initial_action_loss
            and last_five_median <= T3_LEARNABILITY_RATIO * first_five_median
        ),
    }


def _heldout_transfer_summary(
    before_action_losses: dict[int, float],
    after_action_losses: dict[int, float],
) -> dict[str, Any]:
    expected_seeds = set(T3_HELDOUT_RECIPE_SEEDS)
    if set(before_action_losses) != expected_seeds or set(after_action_losses) != (
        expected_seeds
    ):
        raise ValueError("T3 held-out loss seeds differ from the frozen set")

    rows = []
    for seed in T3_HELDOUT_RECIPE_SEEDS:
        before = before_action_losses[seed]
        after = after_action_losses[seed]
        if (
            not math.isfinite(before)
            or not math.isfinite(after)
            or before <= 0.0
            or after < 0.0
        ):
            raise ValueError("T3 held-out losses must be finite with positive pre-loss")
        ratio = after / before
        rows.append(
            {
                "after_action_loss": after,
                "absolute_delta": after - before,
                "before_action_loss": before,
                "improved": after < before,
                "post_to_pre_ratio": ratio,
                "relative_drop": 1.0 - ratio,
                "seed": seed,
            }
        )

    median_ratio = statistics.median(row["post_to_pre_ratio"] for row in rows)
    improved_count = sum(row["improved"] for row in rows)
    return {
        "heldout_seed_count": len(rows),
        "improved_count": improved_count,
        "median_post_to_pre_ratio": median_ratio,
        "minimum_improved_count": T3_HELDOUT_MIN_IMPROVED,
        "per_seed": rows,
        "sorted_post_to_pre_ratios": sorted(row["post_to_pre_ratio"] for row in rows),
        "threshold_median_ratio": T3_HELDOUT_MEDIAN_RATIO,
        "transfer_gate": (
            median_ratio <= T3_HELDOUT_MEDIAN_RATIO
            and improved_count >= T3_HELDOUT_MIN_IMPROVED
        ),
    }


def _write_terminal_report(path: Path, report: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            report,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def _tensor_observation(tensor: Any) -> dict[str, Any]:
    values = tensor.detach().float().reshape(-1)
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "values": [float(value) for value in values.tolist()],
    }


def _terminalize_failure(root: Path, error: BaseException) -> None:
    marker = root / "FAILED.json"
    result_marker = root / "RESULT.json"
    if (
        not result_marker.exists()
        and not result_marker.is_symlink()
        and not marker.exists()
        and not marker.is_symlink()
    ):
        _write_terminal_report(
            marker,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "error_type": type(error).__name__,
                "result": "FAIL",
                "schema_version": "sana-wam-libero-t3-fixed-sample-failure-v1",
            },
        )
    _freeze_run_root(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--steps", type=int, default=T3_UPDATE_STEPS)
    parser.add_argument(
        "--initialization-seed", type=int, default=T3_INITIALIZATION_SEED
    )
    parser.add_argument("--loss-recipe-seed", type=int, default=T3_LOSS_RECIPE_SEED)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

    if args.steps != T3_UPDATE_STEPS:
        raise ValueError(f"T3 steps must be exactly {T3_UPDATE_STEPS}")
    if args.initialization_seed != T3_INITIALIZATION_SEED:
        raise ValueError("T3 initialization seed differs")
    if args.loss_recipe_seed != T3_LOSS_RECIPE_SEED:
        raise ValueError("T3 loss recipe seed differs")
    if len(args.nonce) != 32 or any(
        character not in "0123456789abcdef" for character in args.nonce
    ):
        raise ValueError("T3 nonce must be exactly 32 lowercase hex characters")
    expected_root_name = f"libero-t3-heldout3-fixed20-{args.nonce}"
    if args.run_root.name != expected_root_name:
        raise ValueError("T3 run-root basename does not bind the nonce")
    if args.run_root.parent.name != args.expected_repo_commit[:12]:
        raise ValueError("T3 run-root parent does not bind the source commit")
    heldout_seed_manifest = _heldout_seed_manifest()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {str(args.physical_gpu), args.expected_gpu_uuid}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the selected physical GPU "
            f"({args.physical_gpu} or {args.expected_gpu_uuid}), got {visible!r}"
        )

    config_candidate = args.config.expanduser()
    if config_candidate.is_symlink() or not config_candidate.is_file():
        raise ValueError(
            f"config must be a regular non-symlink file: {config_candidate}"
        )
    config_path = config_candidate.resolve()
    runner_path = Path(__file__).resolve()
    run_root = _create_run_root(args.run_root.expanduser())
    global _ACTIVE_RUN_ROOT
    _ACTIVE_RUN_ROOT = run_root

    actual_commit = _repo_commit()
    actual_config_sha256 = _sha256_file(config_path)
    actual_runner_sha256 = _sha256_file(runner_path)
    expected_identity = {
        "config_sha256": args.expected_config_sha256.lower(),
        "repo_commit": args.expected_repo_commit.lower(),
        "runner_sha256": args.expected_runner_sha256.lower(),
        "sana_commit": EXPECTED_SANA_COMMIT,
    }
    actual_identity = {
        "config_sha256": actual_config_sha256,
        "repo_commit": actual_commit,
        "runner_sha256": actual_runner_sha256,
        "sana_commit": EXPECTED_SANA_COMMIT,
    }
    if actual_identity != expected_identity:
        raise RuntimeError(
            "T3 source identity differs: "
            f"expected={expected_identity!r} actual={actual_identity!r}"
        )
    actual_identity["sana_commit"] = _assert_source_tree_clean(actual_commit)

    cfg = OmegaConf.load(config_path)
    if cfg.model.architecture.variant != "autoregressive":
        raise RuntimeError("T3 requires the AR architecture")
    if cfg.model.video_backbone.continuous_timestep_conditioning is not True:
        raise RuntimeError("T3 requires continuous FP32 video timesteps")
    if cfg.training.use_gradient_checkpointing is not True:
        raise RuntimeError("T3 requires gradient checkpointing")
    if int(cfg.training.seed) != args.initialization_seed:
        raise RuntimeError("T3 CLI initialization seed must equal training.seed")
    if tuple(cfg.training.trainable_modules) != EXPECTED_TRAINABLE_ROOTS:
        raise RuntimeError("T3 trainable module allowlist differs")
    if (
        tuple(cfg.training.preserve_frozen_input_grad_modules)
        != EXPECTED_PRESERVE_FROZEN_INPUT_GRAD_MODULES
    ):
        raise RuntimeError("T3 frozen input-gradient preservation differs")
    if cfg.training.optimizer_master_weights is not True:
        raise RuntimeError("T3 requires persistent FP32 optimizer masters")
    if (
        float(cfg.training.lambda_video) != 0.0
        or float(cfg.training.lambda_action) != 1.0
    ):
        raise RuntimeError("T3 requires action-only loss weighting")
    if float(cfg.training.weight_decay) != 0.0:
        raise RuntimeError("T3 requires weight_decay=0")
    if (
        float(cfg.training.action_lr) != 1.0e-4
        or float(cfg.training.video_lr) != 1.0e-4
    ):
        raise RuntimeError("T3 requires fixed base learning rates of 1e-4")
    if float(cfg.training.grad_clip) != 1.0:
        raise RuntimeError("T3 requires the fixed gradient clip bound 1.0")
    forbidden_checkpoint_fields = (
        "training.init_checkpoint",
        "training.resume_checkpoint",
        "training.resume_from_checkpoint",
        "training.resume_manifest",
        "model.video_backbone.init_dit_from",
    )
    for path in forbidden_checkpoint_fields:
        if OmegaConf.select(cfg, path, default=None) is not None:
            raise RuntimeError(f"T3 forbids checkpoint field {path}")
    if str(cfg.training.action_stats_sha256) != SELECTED_STATS_SHA256:
        raise RuntimeError("T3 selected-row stats SHA pin differs")
    if str(cfg.training.action_stats_population_sha256) != SELECTED_POPULATION_SHA256:
        raise RuntimeError("T3 selected population SHA pin differs")
    stats_path = Path(str(cfg.dataloader.action_stats_path)).expanduser()
    if stats_path.is_symlink() or not stats_path.is_file():
        raise RuntimeError("T3 stats artifact must be a regular non-symlink file")
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T3 selected-row stats artifact SHA differs")

    from sana_wam.train.libero_contract import validate_libero_training_config

    # This is the only full numeric live-source preflight. It runs before any
    # model construction or CUDA allocation.
    validate_libero_training_config(cfg, require_materialized_stats=True)
    external_assets = _verify_external_assets(cfg)
    spatial_root = Path(str(cfg.dataloader.dataset_roots[0]))
    external_assets["spatial_sample/meta/stats_gr00t.json"] = _verify_pinned_asset(
        spatial_root / "meta/stats_gr00t.json",
        SPATIAL_STATS_GR00T_SHA256,
    )
    external_assets["t1_predecessor/RESULT.json"] = _verify_pinned_asset(
        T1_PREDECESSOR_RESULT_PATH,
        T1_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t2_predecessor/RESULT.json"] = _verify_pinned_asset(
        T2_PREDECESSOR_RESULT_PATH,
        T2_PREDECESSOR_RESULT_SHA256,
    )
    external_assets["t2_predecessor/runner.py"] = _verify_pinned_asset(
        T2_PREDECESSOR_RUNNER_PATH,
        T2_PREDECESSOR_RUNNER_SHA256,
    )
    t2_predecessor = json.loads(T2_PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        t2_predecessor.get("valid_run") is not True
        or t2_predecessor.get("scientific_verdict") != "T2_FIXED_SAMPLE_LEARNABILITY_GO"
        or t2_predecessor.get("identity", {}).get("repo_commit")
        != T2_PREDECESSOR_SOURCE_COMMIT
        or t2_predecessor.get("identity", {}).get("runner_sha256")
        != T2_PREDECESSOR_RUNNER_SHA256
    ):
        raise RuntimeError("T3 predecessor T2 evidence semantics differ")
    external_assets = dict(sorted(external_assets.items()))
    actual_identity.update(
        {
            "external_assets_sha256": _sha256_json(external_assets),
            "stats_population_sha256": SELECTED_POPULATION_SHA256,
            "stats_sha256": SELECTED_STATS_SHA256,
            "t1_predecessor_result_sha256": T1_PREDECESSOR_RESULT_SHA256,
            "t2_predecessor_result_sha256": T2_PREDECESSOR_RESULT_SHA256,
            "t2_predecessor_runner_sha256": T2_PREDECESSOR_RUNNER_SHA256,
            "t2_predecessor_source_commit": T2_PREDECESSOR_SOURCE_COMMIT,
        }
    )

    import fcntl

    lock_path = Path(f"/tmp/sana-wam-{args.expected_gpu_uuid}.lock")
    gpu_lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"selected GPU lock is already held: {lock_path}") from exc
    gpu_before = _assert_idle_gpu(args.physical_gpu, args.expected_gpu_uuid)
    if "H200" not in gpu_before["name"].upper():
        raise RuntimeError(f"T3 selected GPU is not an H200: {gpu_before['name']}")

    import torch

    from sana_wam.train.trainer import Trainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "T3 requires exactly one visible CUDA GPU; "
            f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.initialization_seed)
    np.random.seed(args.initialization_seed % (2**32))
    torch.manual_seed(args.initialization_seed)
    torch.cuda.manual_seed_all(args.initialization_seed)

    warning_messages: list[str] = []

    class WarningCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warning_messages.append(record.getMessage())

    warning_capture = WarningCapture(level=logging.WARNING)
    builder_logger = logging.getLogger(
        "sana_wam.model.video_backbone.sana.pipeline_builder"
    )
    builder_logger.addHandler(warning_capture)
    gpu_preconstruction = _assert_idle_gpu(args.physical_gpu, args.expected_gpu_uuid)
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
        message
        for message in warning_messages
        if "ckpt load partial" in message.lower()
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
    if any(parameter.requires_grad for parameter in video_backbone.parameters()):
        raise RuntimeError("T3 video-backbone parameters are not frozen")
    named_trainables = [
        (name, parameter)
        for name, parameter in architecture.named_parameters()
        if parameter.requires_grad
    ]
    named_frozen = [
        (name, parameter)
        for name, parameter in architecture.named_parameters()
        if not parameter.requires_grad
    ]
    observed_roots = tuple(
        sorted({_parameter_root(name) for name, _parameter in named_trainables})
    )
    if observed_roots != tuple(sorted(EXPECTED_TRAINABLE_ROOTS)):
        raise RuntimeError(f"T3 trainable parameter roots differ: {observed_roots!r}")
    if len(named_trainables) != EXPECTED_TRAINABLE_TENSOR_COUNT:
        raise RuntimeError("T3 trainable tensor count differs")
    if sum(parameter.numel() for _name, parameter in named_trainables) != (
        EXPECTED_TRAINABLE_PARAMETER_COUNT
    ):
        raise RuntimeError("T3 trainable parameter count differs")
    frozen_versions = {name: parameter._version for name, parameter in named_frozen}

    dataset_index, episode = _find_fixed_sample(trainer.dataset)
    if (
        episode.dataset != SPATIAL_NAME
        or _sha256_file(episode.data_path()) != PARQUET_SHA256
    ):
        raise RuntimeError("T3 fixed real LIBERO sample identity differs")
    if (
        episode.episode_index != 0
        or episode.length != T3_EPISODE_LENGTH
        or episode.task_index != T3_TASK_INDEX
        or episode.task != T3_TASK_NAME
    ):
        raise RuntimeError("T3 fixed LIBERO episode metadata differs")
    sample_started = time.perf_counter()
    sample = trainer.dataset[dataset_index]
    sample_seconds = time.perf_counter() - sample_started
    if (
        sample["action_alignment"] != "observation_t_to_action_t"
        or sample["episode_index"] != 0
        or sample["episode_length"] != T3_EPISODE_LENGTH
        or sample["start_frame"] != 0
        or sample["task_index"] != T3_TASK_INDEX
        or sample["task_name"] != T3_TASK_NAME
    ):
        raise RuntimeError("T3 fixed LIBERO sample metadata differs")
    spatial_assets_after_sample = {
        f"spatial_sample/{relative_path}": _verify_pinned_asset(
            spatial_root / relative_path, expected_sha256
        )
        for relative_path, expected_sha256 in sorted(
            EXPECTED_SPATIAL_SAMPLE_ASSET_SHA256.items()
        )
    }
    for key, observed in spatial_assets_after_sample.items():
        if external_assets[key] != observed:
            raise RuntimeError(f"T3 Spatial asset changed while loading: {key}")
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T3 selected-row stats changed while loading")
    actual_identity["spatial_assets_post_sample_sha256"] = _sha256_json(
        spatial_assets_after_sample
    )

    trainer._set_training_mode()
    if any(module.training for module in video_backbone.modules()):
        raise RuntimeError("T3 preserved video backbone left eval mode")
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
        raise RuntimeError("T3 prepared inputs dropped gradient checkpointing")
    if inputs.get("use_gradient_checkpointing_offload") is not False:
        raise RuntimeError("T3 requires checkpoint offload=false")
    input_tensor_versions = {
        key: value._version
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    model_groups = trainer._param_groups()
    if len(model_groups) != 2 or any(
        float(group["lr"]) != 1.0e-4 for group in model_groups
    ):
        raise RuntimeError("T3 optimizer group/LR structure differs")
    params = [parameter for group in model_groups for parameter in group["params"]]
    parameter_ids = [id(parameter) for parameter in params]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != {
        id(parameter) for _name, parameter in named_trainables
    }:
        raise RuntimeError("T3 optimizer parameters differ from trainables")
    optimizer_groups, master_pairs = trainer._optimizer_param_groups(model_groups)
    if len(master_pairs) != len(params):
        raise RuntimeError("T3 FP32 master coverage differs")
    if [id(model) for model, _master in master_pairs] != parameter_ids:
        raise RuntimeError("T3 FP32 master/model ordering differs")
    masters = [master for _model, master in master_pairs]
    initial_master_ids = [id(master) for master in masters]
    if (
        len(initial_master_ids) != len(set(initial_master_ids))
        or any(model.dtype != torch.bfloat16 for model, _master in master_pairs)
        or any(master.dtype != torch.float32 for _model, master in master_pairs)
        or any(not master.requires_grad for _model, master in master_pairs)
        or any(model.shape != master.shape for model, master in master_pairs)
        or any(model.device != master.device for model, master in master_pairs)
    ):
        raise RuntimeError("T3 FP32 master identity/dtype differs")
    if [
        id(parameter) for group in optimizer_groups for parameter in group["params"]
    ] != initial_master_ids:
        raise RuntimeError("T3 optimizer groups do not contain the FP32 masters")
    if [len(group["params"]) for group in optimizer_groups] != [
        len(group["params"]) for group in model_groups
    ] or [float(group["lr"]) for group in optimizer_groups] != [
        float(group["lr"]) for group in model_groups
    ]:
        raise RuntimeError("T3 FP32 master optimizer group structure differs")
    trainable_name_by_id = {id(parameter): name for name, parameter in named_trainables}
    named_masters = [
        (trainable_name_by_id[id(model)], master) for model, master in master_pairs
    ]
    model_by_name = dict(named_trainables)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(cfg.training.weight_decay),
        betas=(0.9, 0.95),
    )

    def probe_mutation_snapshot() -> dict[str, Any]:
        if any(parameter.grad is not None for _name, parameter in named_trainables):
            raise RuntimeError("T3 update-free probe observed model gradients")
        if any(parameter.grad is not None for _name, parameter in named_frozen):
            raise RuntimeError("T3 update-free probe observed frozen gradients")
        if any(master.grad is not None for _name, master in named_masters):
            raise RuntimeError("T3 update-free probe observed master gradients")
        optimizer_state = []
        for name, master in named_masters:
            for key, value in sorted(optimizer.state.get(master, {}).items()):
                if isinstance(value, torch.Tensor):
                    observed = (
                        "tensor",
                        id(value),
                        value.data_ptr(),
                        value._version,
                    )
                else:
                    observed = ("value", repr(value))
                optimizer_state.append((name, key, observed))
        return {
            "architecture_buffers": [
                (name, id(buffer), buffer.data_ptr(), buffer._version)
                for name, buffer in architecture.named_buffers()
            ],
            "architecture_module_modes": [
                (name, module.training) for name, module in architecture.named_modules()
            ],
            "architecture_parameters": [
                (name, id(parameter), parameter.data_ptr(), parameter._version)
                for name, parameter in architecture.named_parameters()
            ],
            "master_parameters": [
                (name, id(master), master.data_ptr(), master._version)
                for name, master in named_masters
            ],
            "optimizer_state": optimizer_state,
        }

    recipe_capture: dict[str, list[dict[str, Any]]] = {
        "action": [],
        "video": [],
    }
    architecture_forward_calls = 0
    video_embedder = video_backbone._dit.t_embedder
    original_video_embedder_forward = video_embedder.forward
    original_action_prepare_state = architecture.action_backbone.prepare_state
    original_architecture_forward = architecture.forward

    def capture_video_timestep(timestep):
        recipe_capture["video"].append(_tensor_observation(timestep))
        return original_video_embedder_forward(timestep)

    def capture_action_timestep(*positional, **keyword):
        if len(positional) >= 2 and isinstance(positional[1], torch.Tensor):
            recipe_capture["action"].append(_tensor_observation(positional[1]))
        token_timestep = keyword.get("token_timesteps")
        if isinstance(token_timestep, torch.Tensor):
            recipe_capture["action"].append(_tensor_observation(token_timestep))
        return original_action_prepare_state(*positional, **keyword)

    def capture_architecture_forward(*positional, **keyword):
        nonlocal architecture_forward_calls
        architecture_forward_calls += 1
        return original_architecture_forward(*positional, **keyword)

    video_embedder.forward = capture_video_timestep
    architecture.action_backbone.prepare_state = capture_action_timestep
    architecture.forward = capture_architecture_forward

    def fixed_forward(
        *, backward: bool, recipe_seed: int
    ) -> tuple[dict[str, float], str]:
        recipe_capture["action"].clear()
        recipe_capture["video"].clear()
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()
        torch_cpu_rng_before = torch.random.get_rng_state().clone()
        torch_cuda_rng_before = torch.cuda.get_rng_state().clone()
        with _fixed_rng_recipe(
            torch,
            seed=recipe_seed,
            cuda_device=torch.cuda.current_device(),
        ):
            result = architecture.compute_loss(
                lambda_video=trainer.lambda_video,
                lambda_action=trainer.lambda_action,
                **inputs,
            )
            loss = result["loss"]
            loss_action = result["loss_action"]
            if (
                loss.ndim != 0
                or loss_action.ndim != 0
                or not bool(torch.isfinite(loss).item())
                or not bool(torch.isfinite(loss_action).item())
            ):
                raise RuntimeError("T3 produced a non-finite scalar loss")
            scalars = {
                "loss": _tensor_scalar(loss),
                "loss_action": _tensor_scalar(loss_action),
            }
            tolerance = 1.0e-6 + 1.0e-6 * abs(scalars["loss_action"])
            if abs(scalars["loss"] - scalars["loss_action"]) > tolerance:
                raise RuntimeError("T3 total/action loss identity differs")
            signature = _sha256_json(recipe_capture)
            if not recipe_capture["action"] or not recipe_capture["video"]:
                raise RuntimeError("T3 stochastic recipe capture is incomplete")
            if backward:
                loss.backward()
        numpy_rng_after = np.random.get_state()
        if (
            random.getstate() != python_rng_before
            or numpy_rng_after[0] != numpy_rng_before[0]
            or not np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
            or numpy_rng_after[2:] != numpy_rng_before[2:]
            or not torch.equal(torch.random.get_rng_state(), torch_cpu_rng_before)
            or not torch.equal(torch.cuda.get_rng_state(), torch_cuda_rng_before)
        ):
            raise RuntimeError("T3 fixed recipe did not restore caller RNG state")
        del loss, loss_action, result
        return scalars, signature

    per_step: list[dict[str, Any]] = []
    training_recipe_signatures: list[str] = []
    heldout_before: dict[int, dict[str, float]] = {}
    heldout_after: dict[int, dict[str, float]] = {}
    heldout_recipe_signatures: dict[int, list[str]] = {
        seed: [] for seed in T3_HELDOUT_RECIPE_SEEDS
    }
    initial_probe: dict[str, float]
    final_probe: dict[str, float]
    master_probes = None
    model_probes = None
    first_master_update = None
    first_projected_update = None
    update_sentinels: list[dict[str, Any]] = []
    update_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for parameter in params:
            parameter.grad = None
        optimizer.zero_grad(set_to_none=True)
        if optimizer.state:
            raise RuntimeError("T3 optimizer state exists before held-out pre probes")
        heldout_pre_state = probe_mutation_snapshot()
        for heldout_seed in T3_HELDOUT_RECIPE_SEEDS:
            scalars, signature = fixed_forward(
                backward=False,
                recipe_seed=heldout_seed,
            )
            heldout_before[heldout_seed] = scalars
            heldout_recipe_signatures[heldout_seed].append(signature)
        if probe_mutation_snapshot() != heldout_pre_state or optimizer.state:
            raise RuntimeError("T3 held-out pre probes mutated training state")

        initial_probe, initial_signature = fixed_forward(
            backward=False,
            recipe_seed=args.loss_recipe_seed,
        )
        training_recipe_signatures.append(initial_signature)

        for step_index in range(T3_UPDATE_STEPS):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for parameter in params:
                parameter.grad = None
            step_scalars, recipe_signature = fixed_forward(
                backward=True,
                recipe_seed=args.loss_recipe_seed,
            )
            training_recipe_signatures.append(recipe_signature)
            if step_index == 0:
                tolerance = 1.0e-6 + 1.0e-6 * abs(initial_probe["loss_action"])
                if (
                    abs(step_scalars["loss_action"] - initial_probe["loss_action"])
                    > tolerance
                ):
                    raise RuntimeError(
                        "T3 fixed initial loss recipe is not reproducible"
                    )

            missing_gradients = []
            nonzero_gradient_tensors = 0
            nonzero_gradient_roots: set[str] = set()
            max_gradient_abs = 0.0
            for name, parameter in named_trainables:
                gradient = parameter.grad
                if gradient is None:
                    missing_gradients.append(name)
                    continue
                if not bool(torch.isfinite(gradient).all().item()):
                    raise RuntimeError(
                        f"T3 non-finite gradient at step {step_index + 1}: {name}"
                    )
                maximum = float(gradient.detach().abs().max().float().item())
                max_gradient_abs = max(max_gradient_abs, maximum)
                if maximum > 0.0:
                    nonzero_gradient_tensors += 1
                    nonzero_gradient_roots.add(_parameter_root(name))
            if missing_gradients:
                raise RuntimeError(
                    f"T3 missing gradients at step {step_index + 1}: "
                    f"{missing_gradients[:8]!r}"
                )
            if nonzero_gradient_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    f"T3 nonzero-gradient roots differ at step {step_index + 1}: "
                    f"{sorted(nonzero_gradient_roots)!r}"
                )
            if any(parameter.grad is not None for _name, parameter in named_frozen):
                raise RuntimeError("T3 frozen parameters received gradients")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, float(cfg.training.grad_clip)
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError("T3 gradient norm is non-finite")
            trainer._sync_master_gradients(master_pairs)
            if step_index == 0:
                invalid_master_gradients = [
                    name
                    for name, master in named_masters
                    if master.grad is None
                    or master.grad.dtype != torch.float32
                    or not bool(torch.isfinite(master.grad).all().item())
                    or not bool(torch.count_nonzero(master.grad).item())
                ]
                if invalid_master_gradients:
                    raise RuntimeError(
                        "T3 first-step masters lack finite nonzero FP32 gradients: "
                        f"{invalid_master_gradients[:8]!r}"
                    )
                master_probes = _capture_update_probes(named_masters)
                model_probes = _capture_update_probes(named_trainables)
                if {row["name"] for row in master_probes} != {
                    name for name, _master in named_masters
                }:
                    raise RuntimeError("T3 first-step master probe coverage differs")

                for root in EXPECTED_TRAINABLE_ROOTS:
                    candidates = sorted(
                        (
                            probe
                            for probe in master_probes
                            if _parameter_root(probe["name"]) == root
                        ),
                        key=lambda probe: abs(probe["gradient"]),
                        reverse=True,
                    )[:4]
                    if len(candidates) != 4:
                        raise RuntimeError(f"T3 lacks four update sentinels for {root}")
                    update_sentinels.extend(candidates)

            sentinel_before = [
                {
                    "before_master": _tensor_scalar(
                        probe["parameter"].detach().reshape(-1)[probe["index"]]
                    ),
                    "before_model": _tensor_scalar(
                        model_by_name[probe["name"]]
                        .detach()
                        .reshape(-1)[probe["index"]]
                    ),
                    "gradient": _tensor_scalar(
                        probe["parameter"].grad.detach().reshape(-1)[probe["index"]]
                    ),
                    "index": probe["index"],
                    "name": probe["name"],
                }
                for probe in update_sentinels
            ]

            optimizer.step()
            if step_index == 0:
                first_master_update, changed_master_ids = _summarize_updates(
                    master_probes
                )
                if len(changed_master_ids) != len(named_masters):
                    raise RuntimeError("T3 first step did not update every FP32 master")
            trainer._copy_master_parameters_to_model(master_pairs)
            sentinel_updates = []
            for before, probe in zip(sentinel_before, update_sentinels, strict=True):
                after_master = _tensor_scalar(
                    probe["parameter"].detach().reshape(-1)[probe["index"]]
                )
                after_model = _tensor_scalar(
                    model_by_name[probe["name"]].detach().reshape(-1)[probe["index"]]
                )
                row = {
                    **before,
                    "after_master": after_master,
                    "after_model": after_model,
                    "master_delta": after_master - before["before_master"],
                    "model_delta": after_model - before["before_model"],
                    "root": _parameter_root(probe["name"]),
                }
                if not all(
                    math.isfinite(row[key])
                    for key in (
                        "after_master",
                        "after_model",
                        "master_delta",
                        "model_delta",
                    )
                ):
                    raise RuntimeError("T3 update sentinel became non-finite")
                sentinel_updates.append(row)
            changed_master_sentinels = [
                row for row in sentinel_updates if row["master_delta"] != 0.0
            ]
            changed_master_sentinel_roots = {
                row["root"] for row in changed_master_sentinels
            }
            if changed_master_sentinel_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    "T3 sampled FP32 master update roots differ at step "
                    f"{step_index + 1}: "
                    f"{sorted(changed_master_sentinel_roots)!r}"
                )
            sentinel_adam_steps = {
                _tensor_scalar(optimizer.state[probe["parameter"]]["step"])
                for probe in update_sentinels
            }
            if sentinel_adam_steps != {float(step_index + 1)}:
                raise RuntimeError(
                    "T3 sampled AdamW step counters differ at step "
                    f"{step_index + 1}: {sorted(sentinel_adam_steps)!r}"
                )
            if step_index in {0, T3_UPDATE_STEPS - 1}:
                projection_mismatches = [
                    trainable_name_by_id[id(model)]
                    for model, master in master_pairs
                    if not torch.equal(
                        model.detach(), master.detach().to(dtype=model.dtype)
                    )
                ]
                if projection_mismatches:
                    raise RuntimeError(
                        f"T3 BF16 projection differs: {projection_mismatches[:8]!r}"
                    )
                if not _optimizer_state_is_finite(optimizer):
                    raise RuntimeError("T3 AdamW state is non-finite")
            if step_index == 0:
                first_projected_update, _changed_model_ids = _summarize_updates(
                    model_probes
                )
            torch.cuda.synchronize(device)
            per_step.append(
                {
                    "gradient_nonzero_roots": sorted(nonzero_gradient_roots),
                    "gradient_nonzero_tensor_count": nonzero_gradient_tensors,
                    "loss": step_scalars["loss"],
                    "loss_action": step_scalars["loss_action"],
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "pre_clip_global_norm": _tensor_scalar(grad_norm),
                    "pre_clip_max_abs": max_gradient_abs,
                    "recipe_signature_sha256": recipe_signature,
                    "sampled_bf16_update_changed_count": sum(
                        row["model_delta"] != 0.0 for row in sentinel_updates
                    ),
                    "sampled_bf16_update_changed_roots": sorted(
                        {
                            row["root"]
                            for row in sentinel_updates
                            if row["model_delta"] != 0.0
                        }
                    ),
                    "sampled_bf16_update_l2": math.sqrt(
                        sum(row["model_delta"] ** 2 for row in sentinel_updates)
                    ),
                    "sampled_bf16_update_max_abs": max(
                        abs(row["model_delta"]) for row in sentinel_updates
                    ),
                    "sampled_master_update_changed_count": len(
                        changed_master_sentinels
                    ),
                    "sampled_master_update_changed_roots": sorted(
                        changed_master_sentinel_roots
                    ),
                    "sampled_master_update_l2": math.sqrt(
                        sum(row["master_delta"] ** 2 for row in sentinel_updates)
                    ),
                    "sampled_master_update_max_abs": max(
                        abs(row["master_delta"]) for row in sentinel_updates
                    ),
                    "sampled_optimizer_step_values": sorted(sentinel_adam_steps),
                    "seconds": time.perf_counter() - step_started,
                    "step": step_index + 1,
                }
            )

        optimizer.zero_grad(set_to_none=True)
        for parameter in params:
            parameter.grad = None
        heldout_post_state = probe_mutation_snapshot()
        final_probe, final_signature = fixed_forward(
            backward=False,
            recipe_seed=args.loss_recipe_seed,
        )
        training_recipe_signatures.append(final_signature)
        for heldout_seed in T3_HELDOUT_RECIPE_SEEDS:
            scalars, signature = fixed_forward(
                backward=False,
                recipe_seed=heldout_seed,
            )
            heldout_after[heldout_seed] = scalars
            heldout_recipe_signatures[heldout_seed].append(signature)
        if probe_mutation_snapshot() != heldout_post_state:
            raise RuntimeError("T3 held-out post probes mutated training state")
    finally:
        video_embedder.forward = original_video_embedder_forward
        architecture.action_backbone.prepare_state = original_action_prepare_state
        architecture.forward = original_architecture_forward

    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - update_started
    if architecture_forward_calls != T3_UPDATE_STEPS + 8:
        raise RuntimeError(
            f"T3 architecture forward count differs: {architecture_forward_calls}"
        )
    if (
        len(training_recipe_signatures) != T3_UPDATE_STEPS + 2
        or len(set(training_recipe_signatures)) != 1
    ):
        raise RuntimeError("T3 training RNG recipe signatures differ across forwards")
    invalid_heldout_signatures = {
        seed: signatures
        for seed, signatures in heldout_recipe_signatures.items()
        if len(signatures) != 2 or len(set(signatures)) != 1
    }
    if invalid_heldout_signatures:
        raise RuntimeError(
            f"T3 held-out RNG recipe signatures differ: {invalid_heldout_signatures!r}"
        )
    recipe_identity_signatures = [training_recipe_signatures[0]] + [
        heldout_recipe_signatures[seed][0] for seed in T3_HELDOUT_RECIPE_SEEDS
    ]
    if len(set(recipe_identity_signatures)) != 4:
        raise RuntimeError("T3 training/held-out recipe identities are not distinct")
    final_optimizer_master_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if final_optimizer_master_ids != initial_master_ids:
        raise RuntimeError("T3 FP32 master objects changed during the loop")
    input_version_changes = [
        key
        for key, version in input_tensor_versions.items()
        if inputs[key]._version != version
    ]
    if input_version_changes:
        raise RuntimeError(
            f"T3 prepared inputs were mutated: {input_version_changes!r}"
        )
    frozen_version_changes = [
        name
        for name, parameter in named_frozen
        if parameter._version != frozen_versions[name]
    ]
    if frozen_version_changes:
        raise RuntimeError(
            f"T3 frozen parameters changed: {frozen_version_changes[:8]!r}"
        )
    if any(not bool(torch.isfinite(master).all().item()) for master in masters):
        raise RuntimeError("T3 FP32 masters became non-finite")
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in params):
        raise RuntimeError("T3 BF16 trainables became non-finite")
    if not _optimizer_state_is_finite(optimizer):
        raise RuntimeError("T3 final AdamW state is non-finite")
    adam_steps: list[float] = []
    for master in masters:
        state = optimizer.state.get(master)
        if not state or "step" not in state:
            raise RuntimeError("T3 AdamW state is missing for an FP32 master")
        step_value = _tensor_scalar(state["step"])
        if not math.isfinite(step_value):
            raise RuntimeError("T3 AdamW step counter is non-finite")
        adam_steps.append(step_value)
    if set(adam_steps) != {float(T3_UPDATE_STEPS)}:
        raise RuntimeError(
            f"T3 AdamW step counters differ: {sorted(set(adam_steps))!r}"
        )

    cumulative_master_update, cumulative_master_ids = _summarize_updates(master_probes)
    cumulative_projected_update, _cumulative_model_ids = _summarize_updates(
        model_probes
    )
    if len(cumulative_master_ids) != len(named_masters) or set(
        cumulative_master_update["changed_trainable_roots"]
    ) != set(EXPECTED_TRAINABLE_ROOTS):
        raise RuntimeError("T3 cumulative FP32 master update coverage differs")

    learnability = _learnability_summary(
        initial_probe["loss_action"],
        [row["loss_action"] for row in per_step],
        final_probe["loss_action"],
    )
    heldout_transfer = _heldout_transfer_summary(
        {seed: heldout_before[seed]["loss_action"] for seed in T3_HELDOUT_RECIPE_SEEDS},
        {seed: heldout_after[seed]["loss_action"] for seed in T3_HELDOUT_RECIPE_SEEDS},
    )
    heldout_transfer_report = {
        **heldout_transfer,
        "per_seed": [
            {
                **row,
                "after_total_loss": heldout_after[row["seed"]]["loss"],
                "before_total_loss": heldout_before[row["seed"]]["loss"],
                "post_recipe_signature_sha256": heldout_recipe_signatures[row["seed"]][
                    1
                ],
                "pre_recipe_signature_sha256": heldout_recipe_signatures[row["seed"]][
                    0
                ],
            }
            for row in heldout_transfer["per_seed"]
        ],
        "probe_state_unchanged": True,
    }
    projected_roots = set(cumulative_projected_update["changed_trainable_roots"])
    secondary_diagnostic_warnings = []
    if not learnability["loss_gate"]:
        secondary_diagnostic_warnings.append(
            "training-recipe T2 learnability gate did not reproduce; "
            "this is not part of the frozen T3 primary transfer gate"
        )
    if projected_roots != set(EXPECTED_TRAINABLE_ROOTS):
        secondary_diagnostic_warnings.append(
            "20-step BF16 projection did not visibly change every trainable root; "
            "this is not part of the frozen T3 primary transfer gate"
        )
    verdict_reasons = []
    if not heldout_transfer["transfer_gate"]:
        verdict_reasons.append(
            "held-out recipe transfer did not clear the median/2-of-3 gate"
        )
    verdict = (
        "T3_HELDOUT_RECIPE_TRANSFER_GO"
        if not verdict_reasons
        else "T3_HELDOUT_RECIPE_TRANSFER_INCONCLUSIVE"
    )
    gpu_after = _query_gpu(args.physical_gpu)
    report = {
        "architecture": {
            "action_dim": architecture.action_dim,
            "class": type(architecture).__name__,
            "continuous_timestep_conditioning": True,
            "parameter_count": sum(
                parameter.numel() for parameter in architecture.parameters()
            ),
            "proprio_dim": architecture.proprio_dim,
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
            "trainable_parameter_tensor_count": EXPECTED_TRAINABLE_TENSOR_COUNT,
            "trainable_roots": list(EXPECTED_TRAINABLE_ROOTS),
            "video_backbone_frozen": True,
        },
        "assets": {
            "config_path": str(config_path),
            "config_sha256": actual_config_sha256,
            "external_files": external_assets,
            "external_files_manifest_sha256": actual_identity["external_assets_sha256"],
            "spatial_files_post_sample": spatial_assets_after_sample,
            "spatial_files_post_sample_manifest_sha256": actual_identity[
                "spatial_assets_post_sample_sha256"
            ],
            "stats_path": str(stats_path.resolve()),
            "stats_population_sha256": SELECTED_POPULATION_SHA256,
            "stats_sha256": SELECTED_STATS_SHA256,
            "stats_validation": {
                "formal_training_admission_granted": False,
                "live_numeric_pretrainer_validation_executed": True,
                "materialized_selected_row_v2_required": True,
            },
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "after": gpu_after,
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
        "learnability": {
            **learnability,
            "final_probe_total_loss": final_probe["loss"],
            "initial_probe_action_loss": initial_probe["loss_action"],
            "initial_probe_total_loss": initial_probe["loss"],
            "pre_update_action_loss_curve": [row["loss_action"] for row in per_step],
            "pre_update_total_loss_curve": [row["loss"] for row in per_step],
            "role": "secondary_diagnostic_not_t3_primary_gate",
        },
        "heldout_recipe_transfer": heldout_transfer_report,
        "memory": {
            "build": build_memory,
            "prepare": prepare_memory,
            "update_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "update_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "optimizer": {
            "adam_step_values": sorted(set(adam_steps)),
            "betas": [0.9, 0.95],
            "constant_learning_rates": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
            "cumulative_master_update": cumulative_master_update,
            "cumulative_projected_bf16_update": cumulative_projected_update,
            "first_master_update": first_master_update,
            "first_projected_bf16_update": first_projected_update,
            "master_object_identity_persistent": True,
            "optimizer_master_parameter_count": len(master_pairs),
            "optimizer_master_parameter_elements": sum(
                master.numel() for master in masters
            ),
            "optimizer_state_finite": True,
            "production_accumulation_equivalent": False,
            "production_schedule_equivalent": False,
            "projection_exact_after_steps": [1, T3_UPDATE_STEPS],
            "update_sentinel_count": len(update_sentinels),
            "update_sentinel_names": [
                {
                    "index": probe["index"],
                    "name": probe["name"],
                    "root": _parameter_root(probe["name"]),
                }
                for probe in update_sentinels
            ],
            "weight_decay": float(cfg.training.weight_decay),
        },
        "per_step": per_step,
        "randomness": {
            "heldout_seed_manifest": heldout_seed_manifest,
            "heldout_signatures_per_seed": {
                str(seed): heldout_recipe_signatures[seed]
                for seed in T3_HELDOUT_RECIPE_SEEDS
            },
            "initialization_seed": args.initialization_seed,
            "training_loss_recipe_seed": args.loss_recipe_seed,
            "recipe_reset": "python+numpy+torch-cpu+torch-cuda before every forward",
            "rng_state_restored_after_every_forward": True,
            "total_recipe_signatures": (
                len(training_recipe_signatures)
                + sum(len(value) for value in heldout_recipe_signatures.values())
            ),
            "unique_recipe_identity_count": len(set(recipe_identity_signatures)),
            "training_recipe_signature_sha256": training_recipe_signatures[0],
            "training_recipe_signatures": len(training_recipe_signatures),
            "unique_training_recipe_signatures": len(set(training_recipe_signatures)),
        },
        "execution_result": "PASS",
        "harness_result": "PASS",
        "result": verdict,
        "run": {
            "nonce": args.nonce,
            "root": str(run_root),
        },
        "sample": {
            "action_alignment": sample["action_alignment"],
            "dataset": episode.dataset,
            "episode_index": episode.episode_index,
            "episode_length": episode.length,
            "parquet_sha256": PARQUET_SHA256,
            "start_frame": sample["start_frame"],
            "task": sample["task_name"],
            "task_index": sample["task_index"],
        },
        "schema_version": "sana-wam-libero-t3-heldout-recipe-transfer-v1",
        "scope": {
            "architecture_forwards_total": T3_UPDATE_STEPS + 8,
            "architecture_measurement_forwards": 8,
            "architecture_training_forwards": T3_UPDATE_STEPS,
            "backward_calls": T3_UPDATE_STEPS,
            "benchmark_evaluation_executed": False,
            "formal_training_executed": False,
            "heldout_measurement_forwards": 6,
            "main_recipe_measurement_forwards": 2,
            "optimizer_steps": T3_UPDATE_STEPS,
            "probe_order": [
                "heldout_pre_h1",
                "heldout_pre_h2",
                "heldout_pre_h3",
                "training_recipe_pre",
                "training_recipe_update_1_through_20",
                "training_recipe_post",
                "heldout_post_h1",
                "heldout_post_h2",
                "heldout_post_h3",
            ],
            "sana_base_pretrained_checkpoint_loaded": True,
            "sana_wam_training_checkpoint_loaded": False,
            "sana_wam_training_checkpoint_saved": False,
            "simulator_executed": False,
        },
        "timings_seconds": {
            "model_and_dataset_build": build_seconds,
            "prepare_real_sample": prepare_seconds,
            "sample_load": sample_seconds,
            "twenty_updates_and_eight_measurements": update_seconds,
        },
        "valid_run": True,
        "scientific_verdict": verdict,
        "secondary_diagnostic_warnings": secondary_diagnostic_warnings,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "warnings": warning_messages,
    }
    if any(run_root.iterdir()):
        raise RuntimeError("T3 run root was not empty before terminal write")
    _write_terminal_report(run_root / "RESULT.json", report)
    _freeze_run_root(run_root)
    _ACTIVE_RUN_ROOT = None
    gpu_lock.close()
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
                    "failed to terminalize T3 root: "
                    f"{type(freeze_error).__name__}: {freeze_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    raise SystemExit(exit_code)
