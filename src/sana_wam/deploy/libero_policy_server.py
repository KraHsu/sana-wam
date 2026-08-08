"""LIBERO-specific policy server layered around frozen generic deploy code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from sana_wam.deploy.libero_model_loader import LiberoActionStateNormalizer
from sana_wam.deploy.policy_server import PolicyServer
from sana_wam.train.libero_contract import (
    LIBERO_ACTION_LOSS_WEIGHTINGS,
    validate_libero_training_config,
)


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
    "dataloader.num_frames",
    "dataloader.action_horizon",
    "dataloader.require_full_action_horizon",
    "dataloader.include_terminal_full_horizon",
    "dataloader.window_stride",
    "dataloader.video_context_mode",
    "dataloader.video_stride",
    "dataloader.temporal_compression",
    "dataloader.causal_temporal",
    "dataloader.height",
    "dataloader.width",
    "model.architecture.framework",
    "model.architecture.variant",
    "model.architecture.action_dim",
    "model.architecture.state_dim",
    "model.architecture.use_proprioception",
    "model.architecture.delta_action",
    "model.architecture.attention_mask_mode",
    "model.architecture.video_attention_mask_mode",
    "model.architecture.ar_frame_chunk_size",
    "model.architecture.ar_chunkwise_temporal_ops",
    "model.architecture.ar_attn_window",
    "model.architecture.ar_action_horizon_rope",
    "model.architecture.ar_noisy_cond_prob",
    "model.architecture.ar_cond_max_ratio",
    "model.architecture.proprio_per_chunk",
    "model.architecture.proprio_action_dropout_prob",
    "model.architecture.ar_bootstrap_clean_prefix",
    "model.architecture.action_loss_weighting",
    "model.video_backbone.attn_kernel",
    "model.video_backbone.continuous_timestep_conditioning",
    "model.video_backbone.model_kwargs.y_norm",
    "model.video_backbone.model_kwargs.y_norm_scale_factor",
    "model.action_backbone.attn_kernel",
)

_LIBERO_TEMPORAL_ALIGNMENT = "causal_past_video_0_to_future_action_1"
_ACTION_SCHEDULER_SHIFT = 5.0
_ACTION_TRAIN_TIMESTEPS = 1000


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


def validate_libero_train_deploy_parity(training_cfg: Any, runtime_cfg: Any) -> None:
    """Pin the causal single-chunk training/deployment implementation graph.

    This establishes identical input indexing, block geometry, horizon identity,
    proprio sources and action scheduler.  Expert observations and closed-loop
    observations can still have different values; that scientific distribution
    gap is not disguised as implementation parity.
    """

    num_frames = OmegaConf.select(training_cfg, "dataloader.num_frames", default=None)
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 2:
        raise ValueError("LIBERO checkpoint must record dataloader.num_frames >= 2")

    action_horizon = OmegaConf.select(
        training_cfg, "dataloader.action_horizon", default=None
    )
    if (
        isinstance(action_horizon, bool)
        or not isinstance(action_horizon, int)
        or action_horizon < 1
    ):
        raise ValueError("LIBERO checkpoint must record a positive action_horizon")

    absent = "__SANA_WAM_LIBERO_CHECKPOINT_FIELD_ABSENT__"
    saved_action_loss_weighting = OmegaConf.select(
        training_cfg,
        "model.architecture.action_loss_weighting",
        default=absent,
    )
    action_loss_weighting = (
        "none"
        if saved_action_loss_weighting == absent
        else saved_action_loss_weighting
    )
    if action_loss_weighting not in LIBERO_ACTION_LOSS_WEIGHTINGS:
        raise ValueError(
            "LIBERO checkpoint records unsupported action_loss_weighting "
            f"{action_loss_weighting!r}"
        )

    exact = {
        "policy.history_len": num_frames,
        "policy.temporal_ensemble": False,
        "inference.ar_obs_chunk_mode": "rolling_buffer",
        "inference.ar_obs_latent_band": "auto",
        "inference.ar_reset_cache_each_generation": True,
        "inference.ar_proprio_mode": "per_step",
        "inference.cache_feedback_mode": "predicted",
        "inference.cache_feedback_fallback": "predicted",
        "inference.num_frames": num_frames,
        "inference.height": 384,
        "inference.width": 320,
        "inference.generation_zero_rerank.enabled": False,
        "optimization.async_inference.mode": "none",
        "optimization.decode_video": False,
        "model.architecture.framework": "dual_system",
        "model.architecture.variant": "autoregressive",
        "model.architecture.attention_mask_mode": "joint",
        "model.architecture.video_attention_mask_mode": "first_frame_causal",
        "model.architecture.ar_frame_chunk_size": 2,
        "model.architecture.ar_chunkwise_temporal_ops": True,
        "model.architecture.ar_attn_window": 1,
        "model.architecture.ar_action_horizon_rope": True,
        "model.architecture.ar_noisy_cond_prob": 0.0,
        "model.architecture.ar_cond_max_ratio": 0.0,
        "model.architecture.proprio_per_chunk": True,
        "model.architecture.proprio_action_dropout_prob": 0.0,
        "model.architecture.ar_bootstrap_clean_prefix": True,
        "model.architecture.action_loss_weighting": action_loss_weighting,
        "model.video_backbone.attn_kernel": "linear_relu",
        "model.video_backbone.continuous_timestep_conditioning": True,
        "model.action_backbone.attn_kernel": "linear_relu",
        "dataloader.causal_temporal": True,
        "dataloader.action_horizon": action_horizon,
        "dataloader.require_full_action_horizon": True,
        "dataloader.include_terminal_full_horizon": True,
        "dataloader.window_stride": action_horizon,
        "dataloader.video_context_mode": "causal_past",
        "dataloader.video_stride": 4,
        "dataloader.temporal_compression": 4,
        "dataloader.height": 384,
        "dataloader.width": 320,
    }
    for path, expected in exact.items():
        runtime_default = (
            "none"
            if path == "model.architecture.action_loss_weighting"
            and saved_action_loss_weighting == absent
            else "__ABSENT__"
        )
        observed = OmegaConf.select(runtime_cfg, path, default=runtime_default)
        if _plain(observed) != expected:
            raise ValueError(
                f"LIBERO train/deploy parity requires {path}={expected!r}, "
                f"got {_plain(observed)!r}"
            )

    execute_horizon = OmegaConf.select(
        runtime_cfg, "policy.execute_horizon", default=None
    )
    if execute_horizon is not None and (
        isinstance(execute_horizon, bool)
        or not isinstance(execute_horizon, int)
        or execute_horizon < 1
        or execute_horizon > action_horizon
    ):
        raise ValueError(
            "LIBERO policy.execute_horizon must be null or in [1, action_horizon]"
        )

    action_steps = OmegaConf.select(runtime_cfg, "inference.action_steps", default=None)
    if isinstance(action_steps, bool) or not isinstance(action_steps, int) or action_steps < 1:
        raise ValueError("LIBERO deployment requires positive integer action_steps")
    shift = OmegaConf.select(runtime_cfg, "inference.shift", default=None)
    if isinstance(shift, bool) or not isinstance(shift, (int, float)):
        raise ValueError("LIBERO deployment requires numeric inference.shift")
    if float(shift) != _ACTION_SCHEDULER_SHIFT:
        raise ValueError(
            "LIBERO action scheduler shift differs from the training scheduler"
        )
    if OmegaConf.select(
        runtime_cfg, "inference.ar_action_tokens_per_chunk", default=None
    ) is not None:
        raise ValueError(
            "LIBERO deployment must derive action_tokens_per_chunk from training geometry"
        )
    video_num_frames = OmegaConf.select(
        runtime_cfg, "inference.video_num_frames", default=None
    )
    expected_video_frames = (num_frames - 1) // 4 + 1
    if video_num_frames != expected_video_frames:
        raise ValueError(
            "LIBERO deploy video_num_frames differs from training pixel cadence"
        )
    latent_frames = 1 + (expected_video_frames - 1) // 4
    frame_chunk_size = OmegaConf.select(
        training_cfg, "model.architecture.ar_frame_chunk_size", default=None
    )
    if latent_frames % frame_chunk_size != 0:
        raise ValueError("LIBERO latent frames do not divide into AR chunks")
    chunks = latent_frames // frame_chunk_size
    if chunks != 1:
        raise ValueError("causal single-chunk LIBERO requires one video chunk")
    if num_frames != 17:
        raise ValueError("LIBERO causal history geometry differs")

    frozen = OmegaConf.select(
        training_cfg, "training.deployment_contract", default=None
    )
    if frozen is None:
        raise ValueError("causal LIBERO checkpoint lacks deployment_contract")
    expected_contract = {
        "schema_version": "sana-wam-libero-causal-single-chunk-parity-v1",
        "history_len": num_frames,
        "action_steps": action_steps,
        "action_scheduler_shift": _ACTION_SCHEDULER_SHIFT,
        "action_scheduler_train_timesteps": _ACTION_TRAIN_TIMESTEPS,
        "temporal_alignment": _LIBERO_TEMPORAL_ALIGNMENT,
        "ar_obs_chunk_mode": "rolling_buffer",
        "ar_obs_latent_band": "auto",
        "ar_reset_cache_each_generation": True,
        "ar_proprio_mode": "per_step",
        "cache_feedback_mode": "predicted",
        "context_proprio_source": "current",
        "chunk_proprio_source": "current",
        "video_context_mode": "causal_past",
        "video_num_frames": expected_video_frames,
        "video_chunks_per_window": 1,
        "action_horizon": action_horizon,
        "anchor_stride": action_horizon,
        "full_horizon_only": True,
        "action_tokens_per_chunk": action_horizon,
        "action_horizon_rope": f"unique_0_to_{action_horizon - 1}",
        "ar_attention_window": 1,
        "resolution": [384, 320],
    }
    observed_contract = _plain(frozen)
    if observed_contract != expected_contract:
        raise ValueError(
            "saved LIBERO deployment_contract differs from runtime semantics"
        )


def _validate_runtime_action_horizon(engine: Any, training_cfg: Any) -> None:
    """Bind the runtime chunk width to the frozen checkpoint horizon."""

    expected = OmegaConf.select(
        training_cfg, "dataloader.action_horizon", default=None
    )
    observed = getattr(engine, "_action_tokens_per_chunk", None)
    if observed != expected:
        raise ValueError(
            "causal LIBERO runtime action tokens must match checkpoint "
            f"action_horizon={expected}, got {observed}"
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
                "train_deploy_parity": {
                    "schema_version": (
                        "sana-wam-libero-causal-single-chunk-parity-v1"
                    ),
                    "history_len": identity.get("history_len"),
                    "action_steps": identity.get("action_steps"),
                    "action_scheduler_shift": _ACTION_SCHEDULER_SHIFT,
                    "action_scheduler_train_timesteps": _ACTION_TRAIN_TIMESTEPS,
                    "action_loss_weighting": select(
                        "model.architecture.action_loss_weighting", "none"
                    ),
                    "temporal_alignment": _LIBERO_TEMPORAL_ALIGNMENT,
                    "ar_obs_chunk_mode": identity.get("ar_obs_chunk_mode"),
                    "ar_obs_latent_band": identity.get("ar_obs_latent_band"),
                    "ar_proprio_mode": identity.get("ar_proprio_mode"),
                    "cache_feedback_mode": identity.get("cache_feedback_mode"),
                    "reset_cache_each_generation": identity.get(
                        "reset_cache_each_generation"
                    ),
                    "context_proprio_source": "current",
                    "chunk_proprio_source": "current",
                    "video_context_mode": select("dataloader.video_context_mode"),
                    "video_stride": select("dataloader.video_stride"),
                    "temporal_compression": select(
                        "dataloader.temporal_compression"
                    ),
                    "action_horizon_rope": (
                        f"unique_0_to_{int(select('dataloader.action_horizon', 28)) - 1}"
                    ),
                    "expert_vs_closed_loop_observation_values_may_differ": True,
                },
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

    # Resolve and reject configuration drift before constructing the 2B model.
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
    validate_libero_train_deploy_parity(
        training_cfg, OmegaConf.merge(training_cfg, deploy_cfg)
    )

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
    saved_config_sha256 = sha256_file(config_path)
    stats_path = Path(
        str(OmegaConf.select(training_cfg, "dataloader.action_stats_path"))
    ).expanduser()
    if stats_path.is_symlink() or not stats_path.is_file():
        raise ValueError(f"invalid LIBERO action-stats artifact: {stats_path}")
    action_stats_sha256 = sha256_file(stats_path)

    training_cfg, architecture = load_libero_from_checkpoint_dir(
        str(checkpoint_dir), device=device, ckpt_name=selected_name
    )
    merged = OmegaConf.merge(training_cfg, deploy_cfg)
    validate_libero_train_deploy_parity(training_cfg, merged)
    engine = build_engine(
        cfg=merged,
        architecture=architecture,
        training_cfg=training_cfg,
    )
    if (
        type(architecture).__name__ != "DualSystemARArchitecture"
        or type(engine).__name__ != "ARInferenceEngine"
    ):
        raise ValueError(
            "causal LIBERO requires DualSystemARArchitecture + ARInferenceEngine"
        )
    _validate_runtime_action_horizon(engine, training_cfg)

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
            "saved_config_sha256": saved_config_sha256,
            "action_stats_path": str(stats_path.resolve()),
            "action_stats_sha256": action_stats_sha256,
        }
    )
    return server


__all__ = [
    "LiberoPolicyServer",
    "build_libero_server_from_config",
    "reject_libero_checkpoint_contract_overrides",
    "validate_libero_train_deploy_parity",
]
