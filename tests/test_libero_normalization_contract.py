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
from sana_wam.train.libero_contract import validate_libero_training_config


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
                "action_stats_sha256": stats_sha256,
                "preserve_frozen_input_grad_modules": ["video_backbone"],
            },
        }
    )


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
