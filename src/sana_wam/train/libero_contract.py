"""Fail-closed configuration contract for LIBERO-native SANA-WAM runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from sana_wam.dataloader.libero_dataset import validate_libero_benchmark_contract
from sana_wam.dataloader.libero_stats import (
    LIBERO_ACTION_DIM,
    LIBERO_ACTION_MODE,
    LIBERO_STATE_DIM,
    LIBERO_STATE_MODE,
    sha256_file,
)


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
        if observed != expected:
            raise ValueError(
                f"LIBERO training contract requires {path}={expected!r}, "
                f"got {observed!r}"
            )

    preserve_input_grad = OmegaConf.select(
        config,
        "training.preserve_frozen_input_grad_modules",
        default=None,
    )
    if preserve_input_grad is None or tuple(preserve_input_grad) != (
        "video_backbone",
    ):
        raise ValueError(
            "LIBERO training contract requires "
            "training.preserve_frozen_input_grad_modules=['video_backbone']"
        )

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


__all__ = ["validate_libero_training_config"]
