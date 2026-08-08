from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from sana_wam.dataloader.libero_stats import sha256_file
from sana_wam.dataloader.transforms.normalize import ActionNormalizer, Normalizer
from sana_wam.deploy.libero_model_loader import (
    LiberoActionStateNormalizer,
    build_libero_dual_normalizer,
    resolve_libero_checkpoint_path,
)
from sana_wam.deploy.libero_policy_server import (
    reject_libero_checkpoint_contract_overrides,
)
from sana_wam.train.libero_contract import (
    LIBERO_R10_QKV_ADAPT_EVAL_MODULES,
    LIBERO_R10_QKV_ADAPT_TRAINABLE_CONTRACT,
    LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS,
    LIBERO_R11_DIT_TRUNK_ADAPT_EVAL_MODULES,
    LIBERO_R11_DIT_TRUNK_ADAPT_TRAINABLE_CONTRACT,
    LIBERO_R11_DIT_TRUNK_ADAPT_TRAINABLE_PARAMETER_PATTERNS,
    validate_libero_training_config,
)


def _stats(dim: int, *, low: float = -1.0, high: float = 1.0) -> dict:
    return {
        "mean": np.zeros(dim, dtype=np.float32),
        "std": np.ones(dim, dtype=np.float32),
        "min": np.full(dim, low, dtype=np.float32),
        "max": np.full(dim, high, dtype=np.float32),
        "q01": np.full(dim, low, dtype=np.float32),
        "q99": np.full(dim, high, dtype=np.float32),
    }


def _expected_contract() -> dict:
    payload = yaml.safe_load(
        Path("benchmarks/libero/policy_config.yml").read_text(encoding="utf-8")
    )
    return payload["expected_server_contract"]


def _libero_config(*, stats_sha256: str | None = None):
    return OmegaConf.create(
        {
            "model": {
                "architecture": {
                    "action_dim": 7,
                    "state_dim": 8,
                    "use_proprioception": True,
                    "delta_action": False,
                },
                "video_backbone": {"continuous_timestep_conditioning": True},
            },
            "dataloader": {
                "type": "libero",
                "action_mode": "libero_relative_eef",
                "state_mode": "libero_eef_axis_angle_gripper",
                "normalize_mode": "min-max",
                "state_normalize_mode": "min-max",
                "delta_action": False,
                "training_video_rotation_degrees": 0,
                "benchmark_contract": _expected_contract(),
            },
            "training": {
                "action_stats_population_sha256": None,
                "action_stats_sha256": stats_sha256,
                "optimizer_master_weights": True,
                "preserve_frozen_input_grad_modules": ["video_backbone"],
            },
        }
    )


def _r10_qkv_adapt_config():
    cfg = _libero_config()
    del cfg.training.preserve_frozen_input_grad_modules
    cfg.training.libero_trainable_contract = LIBERO_R10_QKV_ADAPT_TRAINABLE_CONTRACT
    cfg.training.trainable_parameter_patterns = list(
        LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS
    )
    cfg.training.eval_modules = list(LIBERO_R10_QKV_ADAPT_EVAL_MODULES)
    return cfg


def _r11_dit_trunk_adapt_config():
    cfg = _libero_config()
    del cfg.training.preserve_frozen_input_grad_modules
    cfg.model.architecture.action_loss_weighting = "none"
    cfg.training.libero_trainable_contract = (
        LIBERO_R11_DIT_TRUNK_ADAPT_TRAINABLE_CONTRACT
    )
    cfg.training.trainable_parameter_patterns = list(
        LIBERO_R11_DIT_TRUNK_ADAPT_TRAINABLE_PARAMETER_PATTERNS
    )
    cfg.training.eval_modules = list(
        LIBERO_R11_DIT_TRUNK_ADAPT_EVAL_MODULES
    )
    cfg.training.lambda_video = 0.0
    cfg.training.lambda_action = 1.0
    cfg.training.video_lr = 1.5e-5
    cfg.training.action_lr = 1.5e-5
    return cfg


def _write_stats(path: Path, *, action_dim: int = 7, include_state: bool = True) -> str:
    payload = {
        "libero_relative_eef": _stats(action_dim),
        "num_timesteps": 10,
        "schema_version": "sana-wam-libero-stats-v1",
        "source_manifest": {"schema_version": "test"},
    }
    if include_state:
        payload["libero_eef_axis_angle_gripper"] = _stats(8, low=0.0, high=4.0)
    np.save(path, payload, allow_pickle=True)
    return sha256_file(path)


def test_libero_action_and_state_normalizers_are_independent(tmp_path: Path) -> None:
    stats_path = tmp_path / "action_stats.npy"
    digest = _write_stats(stats_path)
    action = ActionNormalizer(mode="min_max", stats=_stats(7))
    composite = build_libero_dual_normalizer(
        _libero_config(stats_sha256=digest),
        ckpt_dir=tmp_path,
        action_normalizer=action,
    )

    assert composite.action_normalizer.stats["mean"].shape == (7,)
    assert composite.state_normalizer.stats["mean"].shape == (8,)
    normalized_state = composite.normalize(np.full(8, 2.0, dtype=np.float32))
    np.testing.assert_allclose(normalized_state, 0.0)
    normalized_action = composite.normalize(np.zeros((3, 7), dtype=np.float32))
    np.testing.assert_allclose(normalized_action, 0.0)
    np.testing.assert_allclose(composite.unnormalize(normalized_action), 0.0)


def test_composite_normalizer_rejects_unknown_or_state_output_dimensions() -> None:
    composite = LiberoActionStateNormalizer(
        action_normalizer=ActionNormalizer(mode="min_max", stats=_stats(7)),
        state_normalizer=Normalizer(mode="min_max", stats=_stats(8)),
    )
    with pytest.raises(ValueError, match="only 7D actions or 8D state"):
        composite.normalize(np.zeros(9, dtype=np.float32))
    with pytest.raises(ValueError, match="requires trailing dimension 7"):
        composite.unnormalize(np.zeros(8, dtype=np.float32))


def test_libero_normalizer_missing_or_wrong_dimension_fails(tmp_path: Path) -> None:
    stats_path = tmp_path / "action_stats.npy"
    digest = _write_stats(stats_path, action_dim=20)
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        build_libero_dual_normalizer(
            _libero_config(stats_sha256=digest),
            ckpt_dir=tmp_path,
            action_normalizer=ActionNormalizer(mode="min_max", stats=_stats(20)),
        )

    digest = _write_stats(stats_path, include_state=False)
    with pytest.raises(ValueError, match="has no"):
        build_libero_dual_normalizer(
            _libero_config(stats_sha256=digest),
            ckpt_dir=tmp_path,
            action_normalizer=ActionNormalizer(mode="min_max", stats=_stats(7)),
        )


def test_deploy_config_cannot_override_saved_libero_identity() -> None:
    saved = _libero_config()
    reject_libero_checkpoint_contract_overrides(
        saved, OmegaConf.create({"inference": {"action_steps": 4}})
    )
    with pytest.raises(ValueError, match="dataloader.action_mode"):
        reject_libero_checkpoint_contract_overrides(
            saved,
            OmegaConf.create(
                {"dataloader": {"action_mode": "coincidentally_seven_dimensional"}}
            ),
        )
    same = OmegaConf.create({"dataloader": {"action_mode": "libero_relative_eef"}})
    reject_libero_checkpoint_contract_overrides(saved, same)

    OmegaConf.update(
        saved,
        "model.architecture.action_loss_weighting",
        "low_noise",
        merge=False,
    )
    with pytest.raises(ValueError, match="action_loss_weighting"):
        reject_libero_checkpoint_contract_overrides(
            saved,
            OmegaConf.create(
                {"model": {"architecture": {"action_loss_weighting": "none"}}}
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("y_norm", False), ("y_norm_scale_factor", 1.0)],
)
def test_deploy_config_cannot_override_saved_sana_text_norm(field, value) -> None:
    saved = _libero_config()
    OmegaConf.update(
        saved, "model.video_backbone.model_kwargs.y_norm", True, merge=False
    )
    OmegaConf.update(
        saved,
        "model.video_backbone.model_kwargs.y_norm_scale_factor",
        0.01,
        merge=False,
    )
    deployed = OmegaConf.create(
        {"model": {"video_backbone": {"model_kwargs": {field: value}}}}
    )

    with pytest.raises(ValueError, match=field):
        reject_libero_checkpoint_contract_overrides(saved, deployed)


def test_training_contract_rejects_model_or_temporal_semantic_drift() -> None:
    cfg = _libero_config()
    validate_libero_training_config(cfg)
    OmegaConf.update(cfg, "model.architecture.action_dim", 20, merge=False)
    with pytest.raises(ValueError, match="action_dim"):
        validate_libero_training_config(cfg)

    cfg = _libero_config()
    OmegaConf.update(
        cfg,
        "model.video_backbone.continuous_timestep_conditioning",
        False,
        merge=False,
    )
    with pytest.raises(ValueError, match="continuous_timestep_conditioning"):
        validate_libero_training_config(cfg)

    cfg = _libero_config()
    OmegaConf.update(
        cfg,
        "training.preserve_frozen_input_grad_modules",
        [],
        merge=False,
    )
    with pytest.raises(ValueError, match="preserve_frozen_input_grad_modules"):
        validate_libero_training_config(cfg)

    cfg = _libero_config()
    OmegaConf.update(
        cfg,
        "training.optimizer_master_weights",
        False,
        merge=False,
    )
    with pytest.raises(ValueError, match="optimizer_master_weights"):
        validate_libero_training_config(cfg)


@pytest.mark.parametrize("mode", [None, "none", "bsmntw", "low_noise"])
def test_training_contract_accepts_action_loss_weighting_modes(mode) -> None:
    cfg = _libero_config()
    if mode is not None:
        OmegaConf.update(
            cfg,
            "model.architecture.action_loss_weighting",
            mode,
            merge=False,
        )
    validate_libero_training_config(cfg)


def test_training_contract_rejects_unknown_action_loss_weighting() -> None:
    cfg = _libero_config()
    OmegaConf.update(
        cfg,
        "model.architecture.action_loss_weighting",
        "mystery",
        merge=False,
    )
    with pytest.raises(ValueError, match="action_loss_weighting to be one of"):
        validate_libero_training_config(cfg)


def test_r10_qkv_adapt_trainable_contract_accepts_only_frozen_patterns() -> None:
    cfg = _r10_qkv_adapt_config()
    validate_libero_training_config(cfg)

    without_revision = _libero_config()
    without_revision.training.trainable_parameter_patterns = list(
        LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS
    )
    with pytest.raises(ValueError, match="explicit.*libero_trainable_contract"):
        validate_libero_training_config(without_revision)

    without_eval_mode = _r10_qkv_adapt_config()
    del without_eval_mode.training.eval_modules
    with pytest.raises(ValueError, match="eval_modules.*video_backbone"):
        validate_libero_training_config(without_eval_mode)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            "training.libero_trainable_contract",
            "r10_qkv_adapt_v2",
            "unknown.*libero_trainable_contract",
        ),
        (
            "training.trainable_parameter_patterns",
            list(LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS[:-1]),
            "exact frozen.*trainable_parameter_patterns",
        ),
        (
            "training.trainable_parameter_patterns",
            [
                *LIBERO_R10_QKV_ADAPT_TRAINABLE_PARAMETER_PATTERNS[:-1],
                "video_backbone.dit.blocks.*.attn.*",
            ],
            "exact frozen.*trainable_parameter_patterns",
        ),
        (
            "training.trainable_modules",
            ["action_backbone"],
            "trainable_modules.*absent",
        ),
        (
            "training.eval_modules",
            ["video_backbone.vae"],
            "eval_modules.*video_backbone",
        ),
        (
            "training.preserve_frozen_input_grad_modules",
            ["video_backbone"],
            "preserve_frozen_input_grad_modules.*absent",
        ),
        ("training.freeze", [], "training.freeze.*absent"),
    ],
)
def test_r10_qkv_adapt_trainable_contract_fails_closed(
    path: str,
    value,
    message: str,
) -> None:
    cfg = _r10_qkv_adapt_config()
    OmegaConf.update(cfg, path, value, merge=False)
    with pytest.raises(ValueError, match=message):
        validate_libero_training_config(cfg)


def test_r11_dit_trunk_adapt_contract_is_exact_and_accepted() -> None:
    expected_patterns = (
        "action_backbone.*",
        "proprio_encoder.*",
        "proprio_video_embed.*",
        "proprio_action_embed.*",
        "video_backbone.dit.x_embedder.*",
        "video_backbone.dit.t_embedder.*",
        "video_backbone.dit.t_block.*",
        "video_backbone.dit.y_embedder.*",
        "video_backbone.dit.attention_y_norm.*",
        "video_backbone.dit.blocks.*",
    )
    assert (
        LIBERO_R11_DIT_TRUNK_ADAPT_TRAINABLE_PARAMETER_PATTERNS
        == expected_patterns
    )
    assert LIBERO_R11_DIT_TRUNK_ADAPT_EVAL_MODULES == (
        "video_backbone",
    )
    assert not any("final_layer" in pattern for pattern in expected_patterns)

    validate_libero_training_config(_r11_dit_trunk_adapt_config())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            "training.trainable_parameter_patterns",
            list(
                LIBERO_R11_DIT_TRUNK_ADAPT_TRAINABLE_PARAMETER_PATTERNS[
                    :-1
                ]
            ),
            "exact action-loss-reachable.*trainable_parameter_patterns",
        ),
        (
            "training.trainable_parameter_patterns",
            [
                *LIBERO_R11_DIT_TRUNK_ADAPT_TRAINABLE_PARAMETER_PATTERNS,
                "video_backbone.dit.final_layer.*",
            ],
            "exact action-loss-reachable.*trainable_parameter_patterns",
        ),
        (
            "training.trainable_parameter_patterns",
            [
                *LIBERO_R11_DIT_TRUNK_ADAPT_TRAINABLE_PARAMETER_PATTERNS[
                    :4
                ],
                "video_backbone.dit.*",
            ],
            "exact action-loss-reachable.*trainable_parameter_patterns",
        ),
        (
            "training.eval_modules",
            ["video_backbone.vae", "video_backbone.text_encoder"],
            "eval_modules.*video_backbone",
        ),
        (
            "training.trainable_modules",
            ["action_backbone"],
            "trainable_modules.*absent",
        ),
        ("training.freeze", [], "training.freeze.*absent"),
        (
            "model.architecture.action_loss_weighting",
            "low_noise",
            "action_loss_weighting",
        ),
        ("training.lambda_video", 1.0, "lambda_video"),
        ("training.lambda_video", 0, "lambda_video"),
        ("training.lambda_action", 0.0, "lambda_action"),
        ("training.video_lr", 1.0e-5, "video_lr"),
        ("training.action_lr", 1.0e-5, "action_lr"),
        (
            "training.trainable_parameter_dtype",
            "float32",
            "trainable_parameter_dtype.*absent",
        ),
        (
            "training.trainable_parameter_dtype",
            None,
            "trainable_parameter_dtype.*absent",
        ),
    ],
)
def test_r11_dit_trunk_adapt_contract_fails_closed(
    path: str,
    value,
    message: str,
) -> None:
    cfg = _r11_dit_trunk_adapt_config()
    OmegaConf.update(cfg, path, value, merge=False)
    with pytest.raises(ValueError, match=message):
        validate_libero_training_config(cfg)


def test_training_contract_rejects_partial_or_extended_benchmark_identity() -> None:
    cfg = _libero_config()
    cfg.dataloader.benchmark_contract.extra_unreviewed_field = True
    with pytest.raises(ValueError, match="canonical contract"):
        validate_libero_training_config(cfg)


def test_checkpoint_selection_is_exact_in_directory_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoint_step_2.safetensors").write_bytes(b"two")
    latest = tmp_path / "checkpoint_step_10.safetensors"
    latest.write_bytes(b"ten")
    path, name = resolve_libero_checkpoint_path(tmp_path)
    assert path == latest
    assert name == latest.name

    with pytest.raises(ValueError, match="must match"):
        resolve_libero_checkpoint_path(tmp_path, "../external.safetensors")
    link = tmp_path / "checkpoint_step_11.safetensors"
    link.symlink_to(latest)
    with pytest.raises(ValueError, match="must not be a symlink"):
        resolve_libero_checkpoint_path(tmp_path, link.name)
