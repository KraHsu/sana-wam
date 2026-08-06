"""LIBERO-specific policy server layered around frozen generic deploy code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from sana_wam.deploy.libero_model_loader import LiberoActionStateNormalizer
from sana_wam.deploy.policy_server import PolicyServer
from sana_wam.train.libero_contract import validate_libero_training_config


_CRITICAL_CHECKPOINT_PATHS = (
    "dataloader.type",
    "dataloader.action_mode",
    "dataloader.state_mode",
    "dataloader.normalize_mode",
    "dataloader.state_normalize_mode",
    "dataloader.multiview",
    "dataloader.camera_layout",
    "dataloader.target_camera",
    "dataloader.benchmark_contract",
    "model.architecture.action_dim",
    "model.architecture.state_dim",
    "model.architecture.use_proprioception",
    "model.architecture.delta_action",
)


def _plain(value: Any) -> Any:
    return (
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    )


def reject_libero_checkpoint_contract_overrides(
    training_cfg: Any, deploy_cfg: Any
) -> None:
    """Prevent deploy-side YAML from impersonating a different checkpoint."""

    deployed_config = deploy_cfg if deploy_cfg is not None else OmegaConf.create({})
    absent = "__SANA_WAM_LIBERO_CHECKPOINT_FIELD_ABSENT__"
    for path in _CRITICAL_CHECKPOINT_PATHS:
        deployed = OmegaConf.select(deployed_config, path, default=absent)
        if deployed == absent:
            continue
        saved = OmegaConf.select(training_cfg, path, default=absent)
        if _plain(deployed) != _plain(saved):
            raise ValueError(
                f"deploy config cannot override LIBERO checkpoint identity field {path}"
            )


class LiberoPolicyServer(PolicyServer):
    """PolicyServer with LIBERO identity and per-step episode echoes."""

    def __init__(self, engine, cfg, *, identity_cfg=None, **kwargs) -> None:
        self._libero_identity_cfg = cfg if identity_cfg is None else identity_cfg
        super().__init__(engine=engine, cfg=cfg, **kwargs)

    def _build_deployment_identity(self, cfg) -> dict:
        identity_cfg = self._libero_identity_cfg
        validate_libero_training_config(identity_cfg)
        identity = super()._build_deployment_identity(cfg)
        architecture = getattr(self.engine, "architecture", None)
        normalizer = getattr(architecture, "action_normalizer", None)
        if not isinstance(normalizer, LiberoActionStateNormalizer):
            raise ValueError("LIBERO server requires the dual-stream normalizer")

        def select(path: str, default=None):
            return OmegaConf.select(identity_cfg, path, default=default)

        identity.update(
            {
                "dataloader_type": select("dataloader.type"),
                "action_dim": select("model.architecture.action_dim"),
                "state_dim": select("model.architecture.state_dim"),
                "action_mode": select("dataloader.action_mode"),
                "state_mode": select("dataloader.state_mode"),
                "normalize_mode": select("dataloader.normalize_mode"),
                "state_normalize_mode": select("dataloader.state_normalize_mode"),
                "multiview": bool(select("dataloader.multiview", False)),
                "camera_layout": _plain(select("dataloader.camera_layout")),
                "target_camera": select("dataloader.target_camera"),
                "benchmark_contract": _plain(select("dataloader.benchmark_contract")),
                "normalizers": {
                    "action": {
                        "active": True,
                        "configured_mode": select("dataloader.normalize_mode"),
                        "dim": int(
                            np.asarray(normalizer.action_normalizer.stats["mean"]).size
                        ),
                        "explicit": True,
                    },
                    "state": {
                        "active": True,
                        "configured_mode": select("dataloader.state_normalize_mode"),
                        "dim": int(
                            np.asarray(normalizer.state_normalizer.stats["mean"]).size
                        ),
                        "explicit": True,
                    },
                },
            }
        )
        return identity

    def predict(self, obs: dict) -> dict:
        result = super().predict(obs)
        result["episode_key"] = self._active_episode_key
        result["model_noise_seed"] = self._engine_runtime_info().get(
            "current_model_noise_seed"
        )
        return result


def build_libero_server_from_config(
    cfg: Any,
    ckpt_dir: str,
    device: str = "cuda",
    ckpt_name: str | None = None,
) -> LiberoPolicyServer:
    """Build one strict LIBERO server without mutating frozen generic sources."""

    from sana_wam.cach.authority import reject_cach_deploy_before_runtime
    from sana_wam.deploy import build_engine
    from sana_wam.deploy.libero_model_loader import (
        load_libero_from_checkpoint_dir,
        resolve_libero_checkpoint_path,
    )
    from sana_wam.deploy.policy_server import (
        _infer_video_num_frames,
        _normalize_compile_mode_in_cfg,
    )
    from sana_wam.dataloader.libero_stats import sha256_file
    from sana_wam.train.cach_stage0_guard import reject_cach_stage0_base_config

    deploy_cfg = (
        OmegaConf.create({})
        if cfg is None
        else OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=False)
            if OmegaConf.is_config(cfg)
            else cfg
        )
    )
    reject_cach_stage0_base_config(
        deploy_cfg,
        entrypoint="sana_wam.deploy.libero_policy_server",
    )
    reject_cach_deploy_before_runtime(deploy_cfg)

    checkpoint_candidate = Path(ckpt_dir).expanduser()
    if checkpoint_candidate.is_symlink():
        raise ValueError(
            f"LIBERO checkpoint directory must not be a symlink: {checkpoint_candidate}"
        )
    checkpoint_dir = checkpoint_candidate.resolve()
    config_path = checkpoint_dir / "config.yaml"
    if (
        not checkpoint_dir.is_dir()
        or not config_path.is_file()
        or config_path.is_symlink()
    ):
        raise ValueError(f"invalid LIBERO checkpoint directory: {checkpoint_dir}")
    training_cfg = OmegaConf.load(config_path)
    validate_libero_training_config(training_cfg)
    reject_libero_checkpoint_contract_overrides(training_cfg, deploy_cfg)
    _normalize_compile_mode_in_cfg(deploy_cfg)

    configured_name = OmegaConf.select(deploy_cfg, "checkpoint_name", default=None)
    if (
        ckpt_name is not None
        and configured_name is not None
        and ckpt_name != configured_name
    ):
        raise ValueError("CLI and deploy config checkpoint_name disagree")
    selected_name = ckpt_name if ckpt_name is not None else configured_name
    selected_path, selected_name = resolve_libero_checkpoint_path(
        checkpoint_dir, selected_name
    )
    checkpoint_sha256 = sha256_file(selected_path)

    training_cfg, architecture = load_libero_from_checkpoint_dir(
        str(checkpoint_dir), device=device, ckpt_name=selected_name
    )
    dataloader_cfg = OmegaConf.select(training_cfg, "dataloader")
    inference_cfg = OmegaConf.select(
        deploy_cfg, "inference", default=OmegaConf.create({})
    )
    fallbacks = {
        "num_frames": OmegaConf.select(dataloader_cfg, "num_frames", default=33),
        "video_num_frames": _infer_video_num_frames(dataloader_cfg),
        "height": OmegaConf.select(dataloader_cfg, "height", default=480),
        "width": OmegaConf.select(dataloader_cfg, "width", default=832),
    }
    for field, value in fallbacks.items():
        if OmegaConf.select(inference_cfg, field, default=None) is None:
            OmegaConf.update(inference_cfg, field, value, merge=False)
    OmegaConf.update(deploy_cfg, "inference", inference_cfg, merge=True)

    merged = OmegaConf.merge(training_cfg, deploy_cfg)
    engine = build_engine(
        cfg=merged,
        architecture=architecture,
        training_cfg=training_cfg,
    )
    server = LiberoPolicyServer(
        engine=engine,
        cfg=merged,
        identity_cfg=training_cfg,
    )
    server._deployment_identity.update(
        {
            "checkpoint_path": str(selected_path),
            "checkpoint_size": selected_path.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    return server


__all__ = [
    "LiberoPolicyServer",
    "build_libero_server_from_config",
    "reject_libero_checkpoint_contract_overrides",
]
