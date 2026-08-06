#!/usr/bin/env python
"""Run the bounded LIBERO T2 fixed-sample 20-update learnability screen."""

from __future__ import annotations

import argparse
import contextlib
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


T2_UPDATE_STEPS = 20
T2_LEARNABILITY_RATIO = 0.95
T2_INITIALIZATION_SEED = 20260806
T2_LOSS_RECIPE_SEED = 20260826
T2_EPISODE_LENGTH = 110
T2_TASK_INDEX = 0
T2_TASK_NAME = "pick up the black bowl next to the cookie box and place it on the plate"
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
SPATIAL_STATS_GR00T_SHA256 = (
    "0a4b08f5afcdcbe186ec70ea6ff2569233bab706a4d99ed603b23a7688bc33bb"
)
_ACTIVE_RUN_ROOT: Path | None = None


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
    if len(pre_update_action_losses) != T2_UPDATE_STEPS:
        raise ValueError(f"T2 requires exactly {T2_UPDATE_STEPS} pre-update losses")
    values = [initial_action_loss, *pre_update_action_losses, post_update_action_loss]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("T2 losses must be finite and non-negative")
    if initial_action_loss <= 0.0:
        raise ValueError("T2 initial action loss must be strictly positive")
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
        "threshold_ratio": T2_LEARNABILITY_RATIO,
        "loss_gate": (
            post_update_action_loss <= T2_LEARNABILITY_RATIO * initial_action_loss
            and last_five_median <= T2_LEARNABILITY_RATIO * first_five_median
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
                "schema_version": "sana-wam-libero-t2-fixed-sample-failure-v1",
            },
        )
    _freeze_run_root(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--steps", type=int, default=T2_UPDATE_STEPS)
    parser.add_argument(
        "--initialization-seed", type=int, default=T2_INITIALIZATION_SEED
    )
    parser.add_argument("--loss-recipe-seed", type=int, default=T2_LOSS_RECIPE_SEED)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

    if args.steps != T2_UPDATE_STEPS:
        raise ValueError(f"T2 steps must be exactly {T2_UPDATE_STEPS}")
    if args.initialization_seed != T2_INITIALIZATION_SEED:
        raise ValueError("T2 initialization seed differs")
    if args.loss_recipe_seed != T2_LOSS_RECIPE_SEED:
        raise ValueError("T2 loss recipe seed differs")
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
            "T2 source identity differs: "
            f"expected={expected_identity!r} actual={actual_identity!r}"
        )
    actual_identity["sana_commit"] = _assert_source_tree_clean(actual_commit)

    cfg = OmegaConf.load(config_path)
    if cfg.model.architecture.variant != "autoregressive":
        raise RuntimeError("T2 requires the AR architecture")
    if cfg.model.video_backbone.continuous_timestep_conditioning is not True:
        raise RuntimeError("T2 requires continuous FP32 video timesteps")
    if cfg.training.use_gradient_checkpointing is not True:
        raise RuntimeError("T2 requires gradient checkpointing")
    if int(cfg.training.seed) != args.initialization_seed:
        raise RuntimeError("T2 CLI initialization seed must equal training.seed")
    if tuple(cfg.training.trainable_modules) != EXPECTED_TRAINABLE_ROOTS:
        raise RuntimeError("T2 trainable module allowlist differs")
    if (
        tuple(cfg.training.preserve_frozen_input_grad_modules)
        != EXPECTED_PRESERVE_FROZEN_INPUT_GRAD_MODULES
    ):
        raise RuntimeError("T2 frozen input-gradient preservation differs")
    if cfg.training.optimizer_master_weights is not True:
        raise RuntimeError("T2 requires persistent FP32 optimizer masters")
    if (
        float(cfg.training.lambda_video) != 0.0
        or float(cfg.training.lambda_action) != 1.0
    ):
        raise RuntimeError("T2 requires action-only loss weighting")
    if float(cfg.training.weight_decay) != 0.0:
        raise RuntimeError("T2 requires weight_decay=0")
    if (
        float(cfg.training.action_lr) != 1.0e-4
        or float(cfg.training.video_lr) != 1.0e-4
    ):
        raise RuntimeError("T2 requires fixed base learning rates of 1e-4")
    if float(cfg.training.grad_clip) != 1.0:
        raise RuntimeError("T2 requires the fixed gradient clip bound 1.0")
    forbidden_checkpoint_fields = (
        "training.init_checkpoint",
        "training.resume_checkpoint",
        "training.resume_from_checkpoint",
        "training.resume_manifest",
        "model.video_backbone.init_dit_from",
    )
    for path in forbidden_checkpoint_fields:
        if OmegaConf.select(cfg, path, default=None) is not None:
            raise RuntimeError(f"T2 forbids checkpoint field {path}")
    if str(cfg.training.action_stats_sha256) != SELECTED_STATS_SHA256:
        raise RuntimeError("T2 selected-row stats SHA pin differs")
    if str(cfg.training.action_stats_population_sha256) != SELECTED_POPULATION_SHA256:
        raise RuntimeError("T2 selected population SHA pin differs")
    stats_path = Path(str(cfg.dataloader.action_stats_path)).expanduser()
    if stats_path.is_symlink() or not stats_path.is_file():
        raise RuntimeError("T2 stats artifact must be a regular non-symlink file")
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T2 selected-row stats artifact SHA differs")

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
    external_assets = dict(sorted(external_assets.items()))
    actual_identity.update(
        {
            "external_assets_sha256": _sha256_json(external_assets),
            "stats_population_sha256": SELECTED_POPULATION_SHA256,
            "stats_sha256": SELECTED_STATS_SHA256,
            "t1_predecessor_result_sha256": T1_PREDECESSOR_RESULT_SHA256,
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
        raise RuntimeError(f"T2 selected GPU is not an H200: {gpu_before['name']}")

    import torch

    from sana_wam.train.trainer import Trainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "T2 requires exactly one visible CUDA GPU; "
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
        raise RuntimeError("T2 video-backbone parameters are not frozen")
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
        raise RuntimeError(f"T2 trainable parameter roots differ: {observed_roots!r}")
    if len(named_trainables) != EXPECTED_TRAINABLE_TENSOR_COUNT:
        raise RuntimeError("T2 trainable tensor count differs")
    if sum(parameter.numel() for _name, parameter in named_trainables) != (
        EXPECTED_TRAINABLE_PARAMETER_COUNT
    ):
        raise RuntimeError("T2 trainable parameter count differs")
    frozen_versions = {name: parameter._version for name, parameter in named_frozen}

    dataset_index, episode = _find_fixed_sample(trainer.dataset)
    if (
        episode.dataset != SPATIAL_NAME
        or _sha256_file(episode.data_path()) != PARQUET_SHA256
    ):
        raise RuntimeError("T2 fixed real LIBERO sample identity differs")
    if (
        episode.episode_index != 0
        or episode.length != T2_EPISODE_LENGTH
        or episode.task_index != T2_TASK_INDEX
        or episode.task != T2_TASK_NAME
    ):
        raise RuntimeError("T2 fixed LIBERO episode metadata differs")
    sample_started = time.perf_counter()
    sample = trainer.dataset[dataset_index]
    sample_seconds = time.perf_counter() - sample_started
    if (
        sample["action_alignment"] != "observation_t_to_action_t"
        or sample["episode_index"] != 0
        or sample["episode_length"] != T2_EPISODE_LENGTH
        or sample["start_frame"] != 0
        or sample["task_index"] != T2_TASK_INDEX
        or sample["task_name"] != T2_TASK_NAME
    ):
        raise RuntimeError("T2 fixed LIBERO sample metadata differs")
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
            raise RuntimeError(f"T2 Spatial asset changed while loading: {key}")
    if _sha256_file(stats_path) != SELECTED_STATS_SHA256:
        raise RuntimeError("T2 selected-row stats changed while loading")
    actual_identity["spatial_assets_post_sample_sha256"] = _sha256_json(
        spatial_assets_after_sample
    )

    trainer._set_training_mode()
    if any(module.training for module in video_backbone.modules()):
        raise RuntimeError("T2 preserved video backbone left eval mode")
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
        raise RuntimeError("T2 prepared inputs dropped gradient checkpointing")
    if inputs.get("use_gradient_checkpointing_offload") is not False:
        raise RuntimeError("T2 requires checkpoint offload=false")
    input_tensor_versions = {
        key: value._version
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    model_groups = trainer._param_groups()
    if len(model_groups) != 2 or any(
        float(group["lr"]) != 1.0e-4 for group in model_groups
    ):
        raise RuntimeError("T2 optimizer group/LR structure differs")
    params = [parameter for group in model_groups for parameter in group["params"]]
    parameter_ids = [id(parameter) for parameter in params]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != {
        id(parameter) for _name, parameter in named_trainables
    }:
        raise RuntimeError("T2 optimizer parameters differ from trainables")
    optimizer_groups, master_pairs = trainer._optimizer_param_groups(model_groups)
    if len(master_pairs) != len(params):
        raise RuntimeError("T2 FP32 master coverage differs")
    if [id(model) for model, _master in master_pairs] != parameter_ids:
        raise RuntimeError("T2 FP32 master/model ordering differs")
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
        raise RuntimeError("T2 FP32 master identity/dtype differs")
    if [
        id(parameter) for group in optimizer_groups for parameter in group["params"]
    ] != initial_master_ids:
        raise RuntimeError("T2 optimizer groups do not contain the FP32 masters")
    if [len(group["params"]) for group in optimizer_groups] != [
        len(group["params"]) for group in model_groups
    ] or [float(group["lr"]) for group in optimizer_groups] != [
        float(group["lr"]) for group in model_groups
    ]:
        raise RuntimeError("T2 FP32 master optimizer group structure differs")
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

    def fixed_forward(*, backward: bool) -> tuple[dict[str, float], str]:
        recipe_capture["action"].clear()
        recipe_capture["video"].clear()
        with _fixed_rng_recipe(
            torch,
            seed=args.loss_recipe_seed,
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
                raise RuntimeError("T2 produced a non-finite scalar loss")
            scalars = {
                "loss": _tensor_scalar(loss),
                "loss_action": _tensor_scalar(loss_action),
            }
            tolerance = 1.0e-6 + 1.0e-6 * abs(scalars["loss_action"])
            if abs(scalars["loss"] - scalars["loss_action"]) > tolerance:
                raise RuntimeError("T2 total/action loss identity differs")
            signature = _sha256_json(recipe_capture)
            if not recipe_capture["action"] or not recipe_capture["video"]:
                raise RuntimeError("T2 stochastic recipe capture is incomplete")
            if backward:
                loss.backward()
        del loss, loss_action, result
        return scalars, signature

    per_step: list[dict[str, Any]] = []
    recipe_signatures: list[str] = []
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
        initial_probe, initial_signature = fixed_forward(backward=False)
        recipe_signatures.append(initial_signature)

        for step_index in range(T2_UPDATE_STEPS):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for parameter in params:
                parameter.grad = None
            step_scalars, recipe_signature = fixed_forward(backward=True)
            recipe_signatures.append(recipe_signature)
            if step_index == 0:
                tolerance = 1.0e-6 + 1.0e-6 * abs(initial_probe["loss_action"])
                if (
                    abs(step_scalars["loss_action"] - initial_probe["loss_action"])
                    > tolerance
                ):
                    raise RuntimeError(
                        "T2 fixed initial loss recipe is not reproducible"
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
                        f"T2 non-finite gradient at step {step_index + 1}: {name}"
                    )
                maximum = float(gradient.detach().abs().max().float().item())
                max_gradient_abs = max(max_gradient_abs, maximum)
                if maximum > 0.0:
                    nonzero_gradient_tensors += 1
                    nonzero_gradient_roots.add(_parameter_root(name))
            if missing_gradients:
                raise RuntimeError(
                    f"T2 missing gradients at step {step_index + 1}: "
                    f"{missing_gradients[:8]!r}"
                )
            if nonzero_gradient_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    f"T2 nonzero-gradient roots differ at step {step_index + 1}: "
                    f"{sorted(nonzero_gradient_roots)!r}"
                )
            if any(parameter.grad is not None for _name, parameter in named_frozen):
                raise RuntimeError("T2 frozen parameters received gradients")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, float(cfg.training.grad_clip)
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError("T2 gradient norm is non-finite")
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
                        "T2 first-step masters lack finite nonzero FP32 gradients: "
                        f"{invalid_master_gradients[:8]!r}"
                    )
                master_probes = _capture_update_probes(named_masters)
                model_probes = _capture_update_probes(named_trainables)
                if {row["name"] for row in master_probes} != {
                    name for name, _master in named_masters
                }:
                    raise RuntimeError("T2 first-step master probe coverage differs")

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
                        raise RuntimeError(f"T2 lacks four update sentinels for {root}")
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
                    raise RuntimeError("T2 first step did not update every FP32 master")
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
                    raise RuntimeError("T2 update sentinel became non-finite")
                sentinel_updates.append(row)
            changed_master_sentinels = [
                row for row in sentinel_updates if row["master_delta"] != 0.0
            ]
            changed_master_sentinel_roots = {
                row["root"] for row in changed_master_sentinels
            }
            if changed_master_sentinel_roots != set(EXPECTED_TRAINABLE_ROOTS):
                raise RuntimeError(
                    "T2 sampled FP32 master update roots differ at step "
                    f"{step_index + 1}: "
                    f"{sorted(changed_master_sentinel_roots)!r}"
                )
            sentinel_adam_steps = {
                _tensor_scalar(optimizer.state[probe["parameter"]]["step"])
                for probe in update_sentinels
            }
            if sentinel_adam_steps != {float(step_index + 1)}:
                raise RuntimeError(
                    "T2 sampled AdamW step counters differ at step "
                    f"{step_index + 1}: {sorted(sentinel_adam_steps)!r}"
                )
            if step_index in {0, T2_UPDATE_STEPS - 1}:
                projection_mismatches = [
                    trainable_name_by_id[id(model)]
                    for model, master in master_pairs
                    if not torch.equal(
                        model.detach(), master.detach().to(dtype=model.dtype)
                    )
                ]
                if projection_mismatches:
                    raise RuntimeError(
                        f"T2 BF16 projection differs: {projection_mismatches[:8]!r}"
                    )
                if not _optimizer_state_is_finite(optimizer):
                    raise RuntimeError("T2 AdamW state is non-finite")
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
        final_probe, final_signature = fixed_forward(backward=False)
        recipe_signatures.append(final_signature)
    finally:
        video_embedder.forward = original_video_embedder_forward
        architecture.action_backbone.prepare_state = original_action_prepare_state
        architecture.forward = original_architecture_forward

    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - update_started
    if architecture_forward_calls != T2_UPDATE_STEPS + 2:
        raise RuntimeError(
            f"T2 architecture forward count differs: {architecture_forward_calls}"
        )
    if len(set(recipe_signatures)) != 1:
        raise RuntimeError("T2 fixed RNG recipe signatures differ across forwards")
    final_optimizer_master_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if final_optimizer_master_ids != initial_master_ids:
        raise RuntimeError("T2 FP32 master objects changed during the loop")
    input_version_changes = [
        key
        for key, version in input_tensor_versions.items()
        if inputs[key]._version != version
    ]
    if input_version_changes:
        raise RuntimeError(
            f"T2 prepared inputs were mutated: {input_version_changes!r}"
        )
    frozen_version_changes = [
        name
        for name, parameter in named_frozen
        if parameter._version != frozen_versions[name]
    ]
    if frozen_version_changes:
        raise RuntimeError(
            f"T2 frozen parameters changed: {frozen_version_changes[:8]!r}"
        )
    if any(not bool(torch.isfinite(master).all().item()) for master in masters):
        raise RuntimeError("T2 FP32 masters became non-finite")
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in params):
        raise RuntimeError("T2 BF16 trainables became non-finite")
    if not _optimizer_state_is_finite(optimizer):
        raise RuntimeError("T2 final AdamW state is non-finite")
    adam_steps: list[float] = []
    for master in masters:
        state = optimizer.state.get(master)
        if not state or "step" not in state:
            raise RuntimeError("T2 AdamW state is missing for an FP32 master")
        step_value = _tensor_scalar(state["step"])
        if not math.isfinite(step_value):
            raise RuntimeError("T2 AdamW step counter is non-finite")
        adam_steps.append(step_value)
    if set(adam_steps) != {float(T2_UPDATE_STEPS)}:
        raise RuntimeError(
            f"T2 AdamW step counters differ: {sorted(set(adam_steps))!r}"
        )

    cumulative_master_update, cumulative_master_ids = _summarize_updates(master_probes)
    cumulative_projected_update, _cumulative_model_ids = _summarize_updates(
        model_probes
    )
    if len(cumulative_master_ids) != len(named_masters) or set(
        cumulative_master_update["changed_trainable_roots"]
    ) != set(EXPECTED_TRAINABLE_ROOTS):
        raise RuntimeError("T2 cumulative FP32 master update coverage differs")

    learnability = _learnability_summary(
        initial_probe["loss_action"],
        [row["loss_action"] for row in per_step],
        final_probe["loss_action"],
    )
    projected_roots = set(cumulative_projected_update["changed_trainable_roots"])
    verdict_reasons = []
    if not learnability["loss_gate"]:
        verdict_reasons.append("fixed action loss did not clear the 5% gate")
    if projected_roots != set(EXPECTED_TRAINABLE_ROOTS):
        verdict_reasons.append(
            "20-step BF16 projection did not visibly change every trainable root"
        )
    verdict = (
        "T2_FIXED_SAMPLE_LEARNABILITY_GO"
        if not verdict_reasons
        else "T2_FIXED_SAMPLE_INCONCLUSIVE"
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
        },
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
            "projection_exact_after_steps": [1, T2_UPDATE_STEPS],
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
            "initialization_seed": args.initialization_seed,
            "loss_recipe_seed": args.loss_recipe_seed,
            "recipe_reset": "python+numpy+torch-cpu+torch-cuda before every forward",
            "recipe_signature_sha256": recipe_signatures[0],
            "total_recipe_signatures": len(recipe_signatures),
            "unique_recipe_signatures": len(set(recipe_signatures)),
        },
        "execution_result": "PASS",
        "harness_result": "PASS",
        "result": verdict,
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
        "schema_version": "sana-wam-libero-t2-fixed-sample-20step-v1",
        "scope": {
            "architecture_measurement_forwards": 2,
            "architecture_training_forwards": T2_UPDATE_STEPS,
            "backward_calls": T2_UPDATE_STEPS,
            "benchmark_evaluation_executed": False,
            "formal_training_executed": False,
            "optimizer_steps": T2_UPDATE_STEPS,
            "sana_base_pretrained_checkpoint_loaded": True,
            "sana_wam_training_checkpoint_loaded": False,
            "sana_wam_training_checkpoint_saved": False,
            "simulator_executed": False,
        },
        "timings_seconds": {
            "model_and_dataset_build": build_seconds,
            "prepare_real_sample": prepare_seconds,
            "sample_load": sample_seconds,
            "twenty_updates_and_two_measurements": update_seconds,
        },
        "valid_run": True,
        "scientific_verdict": verdict,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "warnings": warning_messages,
    }
    if any(run_root.iterdir()):
        raise RuntimeError("T2 run root was not empty before terminal write")
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
                    "failed to terminalize T2 root: "
                    f"{type(freeze_error).__name__}: {freeze_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    raise SystemExit(exit_code)
