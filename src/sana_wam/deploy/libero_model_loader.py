"""LIBERO checkpoint loader isolated from frozen CACH predecessor sources."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from sana_wam.dataloader.libero_stats import (
    LIBERO_ACTION_DIM,
    LIBERO_ACTION_MODE,
    LIBERO_STATE_DIM,
    LIBERO_STATE_MODE,
    load_libero_stats,
    sha256_file,
)
from sana_wam.dataloader.transforms.normalize import Normalizer, YAML_TO_NORM_MODE
from sana_wam.train.libero_contract import validate_libero_training_config


def _last_dim(value: Any, *, label: str) -> int:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"{label} must have a trailing feature dimension")
    return int(array.shape[-1])


def resolve_libero_checkpoint_path(
    ckpt_dir: str | Path, ckpt_name: str | None = None
) -> tuple[Path, str]:
    """Resolve one regular in-directory checkpoint and return ``(path, name)``."""

    from sana_wam.deploy.model_loader import _find_latest_checkpoint

    directory_candidate = Path(ckpt_dir).expanduser()
    if directory_candidate.is_symlink():
        raise ValueError(
            f"LIBERO checkpoint directory must not be a symlink: {directory_candidate}"
        )
    directory = directory_candidate.resolve()
    if not directory.is_dir():
        raise ValueError(f"invalid LIBERO checkpoint directory: {directory}")
    selected_name = ckpt_name
    if selected_name is None:
        selected_name = Path(_find_latest_checkpoint(str(directory))).name
    selected_name = str(selected_name)
    if (
        Path(selected_name).name != selected_name
        or re.fullmatch(r"checkpoint_step_[0-9]+\.safetensors", selected_name) is None
    ):
        raise ValueError(
            "LIBERO checkpoint_name must match checkpoint_step_<N>.safetensors basename"
        )
    selected_candidate = directory / selected_name
    if selected_candidate.is_symlink():
        raise ValueError(
            f"LIBERO checkpoint file must not be a symlink: {selected_candidate}"
        )
    selected_path = selected_candidate.resolve()
    if selected_path.parent != directory or not selected_path.is_file():
        raise ValueError(f"invalid LIBERO checkpoint file: {selected_path}")
    return selected_path, selected_name


class LiberoActionStateNormalizer:
    """Route 7D action and 8D state arrays to independent normalizers.

    Existing deployment engines use ``architecture.action_normalizer`` for
    action history, public action decoding, and (historically) proprioception.
    This composite preserves that interface without applying 7D action stats to
    LIBERO's independent 8D state vector.
    """

    def __init__(self, *, action_normalizer: Any, state_normalizer: Any) -> None:
        if action_normalizer is None or state_normalizer is None:
            raise ValueError("LIBERO requires active action and state normalizers")
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
        self.stats = action_normalizer.stats
        self.mode = action_normalizer.mode

    def normalize(self, value: Any) -> np.ndarray:
        dim = _last_dim(value, label="LIBERO normalization input")
        if dim == LIBERO_ACTION_DIM:
            return self.action_normalizer.normalize(np.asarray(value))
        if dim == LIBERO_STATE_DIM:
            return self.state_normalizer.normalize(np.asarray(value))
        raise ValueError(
            "LIBERO normalizer accepts only 7D actions or 8D state; "
            f"got trailing dimension {dim}"
        )

    def unnormalize(self, value: Any) -> np.ndarray:
        dim = _last_dim(value, label="LIBERO action output")
        if dim != LIBERO_ACTION_DIM:
            raise ValueError(
                f"LIBERO action unnormalization requires trailing dimension 7, got {dim}"
            )
        return self.action_normalizer.unnormalize(np.asarray(value))


def _validate_stats_vector(stats: dict[str, Any], *, dim: int, label: str) -> None:
    for field in ("mean", "std", "min", "max", "q01", "q99"):
        value = np.asarray(stats.get(field), dtype=np.float32)
        if value.shape != (dim,) or not np.isfinite(value).all():
            raise ValueError(
                f"{label}.{field} must be finite shape ({dim},), got {value.shape}"
            )


def build_libero_dual_normalizer(
    cfg: Any, *, ckpt_dir: str | Path, action_normalizer: Any
) -> LiberoActionStateNormalizer:
    """Bind the generic action normalizer to independently loaded state stats."""

    validate_libero_training_config(cfg)
    checkpoint_dir = Path(ckpt_dir).expanduser().resolve()
    stats_path = checkpoint_dir / "action_stats.npy"
    expected_sha256 = OmegaConf.select(
        cfg, "training.action_stats_sha256", default=None
    )
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("LIBERO checkpoint config has no pinned action_stats_sha256")
    if sha256_file(stats_path) != expected_sha256.lower():
        raise ValueError("LIBERO checkpoint action_stats.npy SHA256 differs")

    payload = load_libero_stats(stats_path)
    action_stats = payload[LIBERO_ACTION_MODE]
    state_stats = payload[LIBERO_STATE_MODE]
    _validate_stats_vector(action_stats, dim=LIBERO_ACTION_DIM, label="LIBERO action")
    _validate_stats_vector(state_stats, dim=LIBERO_STATE_DIM, label="LIBERO state")
    if action_normalizer is None:
        raise ValueError("LIBERO checkpoint did not construct an action normalizer")
    _validate_stats_vector(
        action_normalizer.stats,
        dim=LIBERO_ACTION_DIM,
        label="loaded LIBERO action",
    )
    for field in ("mean", "std", "min", "max", "q01", "q99"):
        if not np.array_equal(
            np.asarray(action_normalizer.stats[field]), np.asarray(action_stats[field])
        ):
            raise ValueError(f"loaded LIBERO action normalizer {field} differs")

    state_mode = OmegaConf.select(cfg, "dataloader.state_normalize_mode", default=None)
    try:
        internal_mode = YAML_TO_NORM_MODE[state_mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"unsupported LIBERO state normalization mode {state_mode!r}"
        ) from exc
    state_normalizer = Normalizer(mode=internal_mode, stats=state_stats)
    return LiberoActionStateNormalizer(
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
    )


def load_libero_from_checkpoint_dir(
    ckpt_dir: str,
    device: str = "cuda",
    ckpt_name: str | None = None,
):
    """Load a LIBERO checkpoint, then install the dual-stream normalizer."""

    from sana_wam.deploy.model_loader import load_from_checkpoint_dir

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
    saved_cfg = OmegaConf.load(config_path)
    validate_libero_training_config(saved_cfg)

    _selected_path, selected_name = resolve_libero_checkpoint_path(
        checkpoint_dir, ckpt_name
    )

    cfg, architecture = load_from_checkpoint_dir(
        str(checkpoint_dir), device=device, ckpt_name=selected_name
    )
    validate_libero_training_config(cfg)
    composite = build_libero_dual_normalizer(
        cfg,
        ckpt_dir=checkpoint_dir,
        action_normalizer=getattr(architecture, "action_normalizer", None),
    )
    architecture.attach_action_normalizer(composite)
    return cfg, architecture


__all__ = [
    "LiberoActionStateNormalizer",
    "build_libero_dual_normalizer",
    "load_libero_from_checkpoint_dir",
    "resolve_libero_checkpoint_path",
]
