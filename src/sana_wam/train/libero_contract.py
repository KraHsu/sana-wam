"""Fail-closed configuration contract for LIBERO-native SANA-WAM runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from sana_wam.dataloader.libero_dataset import validate_libero_benchmark_contract
from sana_wam.dataloader.libero_stats import (
    LIBERO_ACTION_DIM,
    LIBERO_ACTION_MODE,
    LIBERO_SELECTED_STATS_SCHEMA_VERSION,
    LIBERO_STATE_DIM,
    LIBERO_STATE_MODE,
    load_libero_stats,
    sha256_file,
)
from sana_wam.dataloader.libero_selected_stats import (
    validate_libero_selected_stats_for_config,
    validate_libero_window_coverage_for_config,
)


LIBERO_PRODUCTION_DATASET_ROOTS = tuple(
    sorted(
        (
            "/DATA/share/LIBERO/libero/libero_spatial_no_noops_1.0.0_lerobot",
            "/DATA/share/LIBERO/libero/libero_object_no_noops_1.0.0_lerobot",
            "/DATA/share/LIBERO/libero/libero_goal_no_noops_1.0.0_lerobot",
            "/DATA/share/LIBERO/libero/libero_10_no_noops_1.0.0_lerobot",
        )
    )
)
LIBERO_PRODUCTION_STATS_PATH = (
    "/DATA/share/LIBERO/sana_wam_libero_train_all4_excl_goal82_stats_v2.npy"
)
LIBERO_PRODUCTION_POPULATION_COUNTS = {
    "action_rows": 271_644,
    "excluded_episodes": 1,
    "selected_episodes": 1_692,
    "source_episodes": 1_693,
    "source_state_rows": 273_465,
    "state_rows": 273_336,
    "suites": 4,
}
LIBERO_R10_QKV_ADAPT_TRAINABLE_CONTRACT = "r10_qkv_adapt_v1"
LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS = (
    "action_backbone.*",
    "proprio_encoder.*",
    "proprio_video_embed.*",
    "proprio_action_embed.*",
    "video_backbone.dit.blocks.*.attn.qkv.weight",
)
LIBERO_R10_QKV_ADAPT_EVAL_MODULES = ("video_backbone",)


def _config_value(config: Any, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, path, default=default)
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return value


def _validate_libero_production_population_config(config: Any) -> None:
    exact = {
        "dataloader.excluded_episodes": ["libero_goal_no_noops_1.0.0_lerobot:82"],
        "dataloader.seed": 20260806,
        "dataloader.split": "train",
        "dataloader.val_ratio": 0.0,
        "dataloader.verify_known_repairs": True,
        "dataloader.verify_stats_source": True,
    }
    for path, expected in exact.items():
        observed = _config_value(config, path, default=None)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"formal LIBERO population requires {path}={expected!r}, "
                f"got {observed!r}"
            )

    for path, minimum in (
        ("dataloader.num_frames", 2),
        ("dataloader.repeat", 1),
        ("dataloader.window_stride", 1),
    ):
        observed = OmegaConf.select(config, path, default=None)
        if observed is not None and (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < minimum
        ):
            raise ValueError(
                f"LIBERO training contract requires {path} to be an integer "
                f">= {minimum}, got {observed!r}"
            )

    raw_roots = _config_value(config, "dataloader.dataset_roots", default=None)
    if (
        not isinstance(raw_roots, list)
        or len(raw_roots) != 4
        or any(not isinstance(root, str) for root in raw_roots)
    ):
        raise ValueError("formal LIBERO population requires exactly four dataset roots")
    candidates = [Path(str(root)).expanduser() for root in raw_roots]
    if any(candidate.is_symlink() for candidate in candidates):
        raise ValueError("formal LIBERO dataset roots must not be symlinks")
    observed_roots = tuple(sorted(str(candidate.resolve()) for candidate in candidates))
    if observed_roots != LIBERO_PRODUCTION_DATASET_ROOTS:
        raise ValueError("formal LIBERO dataset roots differ from production roots")

    raw_stats_path = _config_value(config, "dataloader.action_stats_path", default=None)
    if not isinstance(raw_stats_path, str) or not raw_stats_path:
        raise ValueError("formal LIBERO population requires action_stats_path")
    stats_candidate = Path(raw_stats_path).expanduser()
    if stats_candidate.is_symlink() or str(stats_candidate.resolve()) != (
        LIBERO_PRODUCTION_STATS_PATH
    ):
        raise ValueError(
            "formal LIBERO action_stats_path differs from reserved v2 path"
        )


def validate_libero_production_stats_source_config(cfg: Any) -> None:
    """Validate all population-defining pins before exclusive publication."""

    config = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)
    _validate_libero_production_population_config(config)


def _validate_libero_trainable_config(config: Any) -> None:
    """Keep parameter-pattern training closed except for the named R10 arm."""

    contract = _config_value(
        config,
        "training.libero_trainable_contract",
        default=None,
    )
    patterns = _config_value(
        config,
        "training.trainable_parameter_patterns",
        default=None,
    )
    preserve_input_grad = _config_value(
        config,
        "training.preserve_frozen_input_grad_modules",
        default=None,
    )

    if contract is None:
        if patterns is not None:
            raise ValueError(
                "LIBERO trainable_parameter_patterns require an explicit "
                "training.libero_trainable_contract"
            )
        if preserve_input_grad is None or tuple(preserve_input_grad) != (
            "video_backbone",
        ):
            raise ValueError(
                "LIBERO training contract requires "
                "training.preserve_frozen_input_grad_modules=['video_backbone']"
            )
        return

    if contract != LIBERO_R10_QKV_ADAPT_TRAINABLE_CONTRACT:
        raise ValueError(
            f"unknown LIBERO training.libero_trainable_contract: {contract!r}"
        )
    if type(patterns) is not list or tuple(patterns) != (
        LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS
    ):
        raise ValueError(
            "R10-QKV-ADAPT requires the exact frozen "
            "training.trainable_parameter_patterns"
        )
    eval_modules = _config_value(
        config,
        "training.eval_modules",
        default=None,
    )
    if type(eval_modules) is not list or tuple(eval_modules) != (
        LIBERO_R10_QKV_ADAPT_EVAL_MODULES
    ):
        raise ValueError(
            "R10-QKV-ADAPT requires training.eval_modules=['video_backbone']"
        )
    for path in (
        "training.trainable_modules",
        "training.preserve_frozen_input_grad_modules",
        "training.freeze",
    ):
        if _config_value(config, path, default=None) is not None:
            raise ValueError(f"R10-QKV-ADAPT requires {path} to be absent")


def validate_libero_training_config(
    cfg: Any,
    *,
    dataset: Any | None = None,
    require_materialized_stats: bool = False,
) -> None:
    """Reject semantic or dimensional drift before constructing ``Trainer``.

    The generic Trainer is part of frozen CACH predecessor evidence, so the
    LIBERO-specific preflight lives in its own launcher/module.  A caller may
    additionally pass the constructed dataset for a second dimension check.
    """

    config = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)
    if OmegaConf.select(config, "dataloader.type", default=None) != "libero":
        raise ValueError("LIBERO launcher requires dataloader.type=libero")

    exact = {
        "model.architecture.action_dim": LIBERO_ACTION_DIM,
        "model.architecture.state_dim": LIBERO_STATE_DIM,
        "model.architecture.use_proprioception": True,
        "model.architecture.delta_action": False,
        "model.video_backbone.continuous_timestep_conditioning": True,
        "dataloader.action_mode": LIBERO_ACTION_MODE,
        "dataloader.state_mode": LIBERO_STATE_MODE,
        "dataloader.normalize_mode": "min-max",
        "dataloader.state_normalize_mode": "min-max",
        "dataloader.delta_action": False,
        "dataloader.training_video_rotation_degrees": 0,
        "training.optimizer_master_weights": True,
    }
    for path, expected in exact.items():
        observed = OmegaConf.select(config, path, default=None)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"LIBERO training contract requires {path}={expected!r}, "
                f"got {observed!r}"
            )

    _validate_libero_trainable_config(config)

    contract = OmegaConf.select(config, "dataloader.benchmark_contract", default=None)
    if OmegaConf.is_config(contract):
        contract = OmegaConf.to_container(contract, resolve=True)
    validate_libero_benchmark_contract(contract)

    if dataset is not None:
        if getattr(dataset, "action_dim", None) != LIBERO_ACTION_DIM:
            raise ValueError("LIBERO dataset/model action_dim mismatch")
        if getattr(dataset, "state_dim", None) != LIBERO_STATE_DIM:
            raise ValueError("LIBERO dataset/model state_dim mismatch")

    if not require_materialized_stats:
        return
    validate_libero_production_stats_source_config(config)
    validate_libero_window_coverage_for_config(config)
    stats_path = OmegaConf.select(config, "dataloader.action_stats_path", default=None)
    expected_sha256 = OmegaConf.select(
        config, "training.action_stats_sha256", default=None
    )
    if not isinstance(stats_path, str) or not stats_path:
        raise ValueError("LIBERO execution requires dataloader.action_stats_path")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("LIBERO execution requires a pinned action_stats_sha256")
    observed_sha256 = sha256_file(Path(stats_path))
    if observed_sha256 != expected_sha256.lower():
        raise ValueError("LIBERO action_stats_sha256 differs")
    payload = load_libero_stats(stats_path)
    if payload.get("schema_version") != LIBERO_SELECTED_STATS_SCHEMA_VERSION:
        raise ValueError(
            "formal LIBERO execution requires exclusion-aware selected-row stats v2"
        )
    expected_population_sha256 = OmegaConf.select(
        config,
        "training.action_stats_population_sha256",
        default=None,
    )
    if (
        not isinstance(expected_population_sha256, str)
        or len(expected_population_sha256) != 64
    ):
        raise ValueError(
            "LIBERO execution requires a pinned action_stats_population_sha256"
        )
    observed_population_sha256 = payload["population_manifest"].get(
        "population_manifest_sha256"
    )
    if observed_population_sha256 != expected_population_sha256.lower():
        raise ValueError("LIBERO action_stats_population_sha256 differs")
    if dict(payload["population_manifest"].get("counts", {})) != (
        LIBERO_PRODUCTION_POPULATION_COUNTS
    ):
        raise ValueError("formal LIBERO selected population counts differ")
    validate_libero_selected_stats_for_config(
        payload,
        config,
        verify_live_sources=True,
    )


__all__ = [
    "LIBERO_PRODUCTION_DATASET_ROOTS",
    "LIBERO_PRODUCTION_POPULATION_COUNTS",
    "LIBERO_PRODUCTION_STATS_PATH",
    "LIBERO_R10_QKV_ADAPT_TRAINABLE_CONTRACT",
    "LIBERO_R10_QKV_ADAPT_EVAL_MODULES",
    "LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS",
    "validate_libero_production_stats_source_config",
    "validate_libero_training_config",
]
