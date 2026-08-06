"""LIBERO LeRobot v2.1 dataset for sana-wam training.

The loader is intentionally independent of the LIBERO simulator and the
``lerobot`` Python package.  Numeric episode data is read lazily from Parquet
with optional PyArrow, while the two AV1 videos are decoded with PyAV (already a
core sana-wam dependency).  Samples preserve the closed-loop temporal contract:
observation/state at time ``t`` supervises LIBERO's relative command at the
same time ``t``; RoboTwin's absolute-state ``t+1`` convention is not reused.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sana_wam.dataloader.base_dataset import BaseActionDataset
from sana_wam.dataloader.libero_stats import (
    LIBERO_ACTION_DIM,
    LIBERO_ACTION_MODE,
    LIBERO_SELECTED_STATS_SCHEMA_VERSION,
    LIBERO_STATE_DIM,
    LIBERO_STATE_MODE,
    build_source_manifest,
    load_libero_stats,
    sha256_file,
)
from sana_wam.dataloader.libero_selected_stats import (
    validate_libero_selected_stats_for_config,
)
from sana_wam.dataloader.libero_selection import (
    LiberoEpisode,
    read_libero_episode_arrays,
    select_libero_episodes,
)
from sana_wam.dataloader.transforms.multiview import (
    assemble_multiview_layout,
    format_prompt_for_inference,
)
from sana_wam.dataloader.transforms.normalize import (
    YAML_TO_NORM_MODE,
    ActionNormalizer,
    Normalizer,
)


LIBERO_CAMERA_LAYOUT = (
    "head_camera",
    "left_wrist_camera",
    "right_wrist_camera",
)
LIBERO_VIDEO_KEY_TO_CAMERA = {
    "observation.images.image": "head_camera",
    "observation.images.wrist_image": "left_wrist_camera",
}
LIBERO_DATASET_SCHEMA_VERSION = "sana-wam-libero-lerobot-v1"
LIBERO_TRAINING_DATASET_NAMES = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)

_KNOWN_GOAL_DATASET = "libero_goal_no_noops_1.0.0_lerobot"
_KNOWN_CORRUPTED_WRIST_VIDEO = (
    "videos/chunk-000/observation.images.wrist_image/episode_000082.mp4"
)
_KNOWN_CORRUPTED_WRIST_SHA256 = (
    "b3788859f4e52d882478ef3b2ffdc73d8419508e245940430efc047bd41ef732"
)
_KNOWN_INCOMPATIBLE_169_FRAME_PATCH_SHA256 = (
    "b260ad525ee22f47a517b30f18147b01a2a92791eb18235e37b307ab955d739b"
)
_KNOWN_CORRUPTED_EPISODE_KEY = f"{_KNOWN_GOAL_DATASET}:82"


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        value = config.get(key, default)
    elif hasattr(config, "get"):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    return default if value is None else value


def _normalization_mode(value: Any, *, field: str) -> str | None:
    if value is None or (
        isinstance(value, str) and value.lower() in {"", "none", "null"}
    ):
        return None
    if not isinstance(value, str) or value not in YAML_TO_NORM_MODE:
        raise ValueError(
            f"{field} must be one of {sorted(YAML_TO_NORM_MODE)} or null, got {value!r}"
        )
    return value


def _validate_stat_shape(stats: Mapping[str, Any], *, dim: int, label: str) -> None:
    for field in ("mean", "std", "min", "max", "q01", "q99"):
        array = np.asarray(stats.get(field), dtype=np.float32)
        if array.shape != (dim,) or not np.isfinite(array).all():
            raise ValueError(
                f"{label}.{field} must be finite shape ({dim},), got {array.shape}"
            )


def canonical_libero_benchmark_contract() -> dict[str, Any]:
    """Return the one complete checkpoint/client identity accepted here."""

    return {
        "schema_version": "sana-wam-libero-policy-v1",
        "benchmark": "libero",
        "dataloader_type": "libero",
        "action_mode": LIBERO_ACTION_MODE,
        "state_mode": LIBERO_STATE_MODE,
        "normalize_required": True,
        "state_dim": LIBERO_STATE_DIM,
        "state_layout": [
            "eef_x",
            "eef_y",
            "eef_z",
            "eef_axis_angle_x",
            "eef_axis_angle_y",
            "eef_axis_angle_z",
            "gripper_qpos_0",
            "gripper_qpos_1",
        ],
        "action_dim": LIBERO_ACTION_DIM,
        "action_layout": [
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_axis_angle_x",
            "delta_axis_angle_y",
            "delta_axis_angle_z",
            "open_gripper",
        ],
        "action_representation": "relative_delta_pose",
        "action_output_space": "denormalized_checkpoint_units",
        "normalization": {
            "action": {
                "active": True,
                "mode": "min-max",
                "stats_key": LIBERO_ACTION_MODE,
                "dim": LIBERO_ACTION_DIM,
            },
            "state": {
                "active": True,
                "mode": "min-max",
                "stats_key": LIBERO_STATE_MODE,
                "dim": LIBERO_STATE_DIM,
            },
        },
        "gripper": {
            "model_open": 1.0,
            "model_closed": 0.0,
            "threshold": 0.5,
            "env_open": -1.0,
            "env_closed": 1.0,
        },
        "multiview": True,
        "camera_layout": list(LIBERO_CAMERA_LAYOUT),
        "camera_mapping": {
            "agentview_image": "head_camera",
            "robot0_eye_in_hand_image": "left_wrist_camera",
            "missing": "right_wrist_camera",
        },
        "missing_right_camera_fill": "black",
        "training_video_rotation_degrees": 0,
        "simulator_video_rotation_degrees": 180,
        "training_dataset_names": list(LIBERO_TRAINING_DATASET_NAMES),
        "excluded_training_episodes": [_KNOWN_CORRUPTED_EPISODE_KEY],
    }


def validate_libero_benchmark_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the full saved contract consumed by the benchmark client."""

    from omegaconf import OmegaConf

    if OmegaConf.is_config(contract):
        contract = OmegaConf.to_container(contract, resolve=True)
    if not isinstance(contract, Mapping):
        raise ValueError("LIBERO dataloader requires benchmark_contract mapping")
    value = copy.deepcopy(dict(contract))
    if value != canonical_libero_benchmark_contract():
        raise ValueError(
            "LIBERO benchmark_contract differs from the canonical contract"
        )
    return value


def _validate_temporal_contract(
    *, num_frames: int, video_stride: int, temporal_compression: int, causal: bool
) -> tuple[list[int], int]:
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")
    if video_stride <= 0 or (num_frames - 1) % video_stride != 0:
        raise ValueError("(num_frames - 1) must be divisible by video_stride")
    indices = list(range(0, num_frames, video_stride))
    video_frames = len(indices)
    if temporal_compression <= 0:
        raise ValueError("temporal_compression must be positive")
    valid = (
        (video_frames - 1) % temporal_compression == 0
        if causal
        else video_frames % temporal_compression == 0
    )
    if not valid:
        rule = "(N-1)" if causal else "N"
        raise ValueError(
            f"LIBERO video frames violate {rule} % {temporal_compression} == 0"
        )
    return indices, video_frames


class LiberoLeRobotDataset(BaseActionDataset):
    """Windowed multi-suite LIBERO dataset with exact 8D/7D semantics."""

    @classmethod
    def from_config(cls, config: Any, split: str = "train") -> "LiberoLeRobotDataset":
        allowed = {
            "type",
            "dataset_roots",
            "dataset_root",
            "action_stats_path",
            "benchmark_contract",
            "num_frames",
            "video_stride",
            "window_stride",
            "height",
            "width",
            "split",
            "val_ratio",
            "repeat",
            "seed",
            "normalize_mode",
            "state_normalize_mode",
            "action_mode",
            "state_mode",
            "temporal_compression",
            "causal_temporal",
            "multiview",
            "camera_layout",
            "target_camera",
            "training_video_rotation_degrees",
            "delta_action",
            "filter_static_segments",
            "verify_stats_source",
            "verify_known_repairs",
            "excluded_episodes",
        }
        keys = set(config.keys()) if hasattr(config, "keys") else set()
        unknown = sorted(keys - allowed)
        if unknown:
            raise ValueError(f"unknown LIBERO dataloader fields: {unknown}")
        configured_split = _get(config, "split", split)
        if configured_split != split:
            raise ValueError(
                f"LIBERO config split={configured_split!r} disagrees with requested {split!r}"
            )
        if _get(config, "delta_action", False) is not False:
            raise ValueError("LIBERO relative commands require delta_action=false")
        if _get(config, "filter_static_segments", False) is not False:
            raise ValueError(
                "LIBERO loader does not implement static-segment filtering"
            )
        roots = _get(config, "dataset_roots", None)
        if roots is None:
            single = _get(config, "dataset_root", None)
            roots = [] if single is None else [single]
        if isinstance(roots, (str, Path)):
            roots = [roots]
        camera_layout = _get(config, "camera_layout", list(LIBERO_CAMERA_LAYOUT))
        return cls(
            dataset_roots=list(roots),
            action_stats_path=_get(config, "action_stats_path", None),
            benchmark_contract=_get(config, "benchmark_contract", None),
            num_frames=int(_get(config, "num_frames", 33)),
            video_stride=int(_get(config, "video_stride", 4)),
            window_stride=int(_get(config, "window_stride", 1)),
            height=int(_get(config, "height", 384)),
            width=int(_get(config, "width", 320)),
            split=split,
            val_ratio=float(_get(config, "val_ratio", 0.05)),
            repeat=int(_get(config, "repeat", 1)),
            seed=int(_get(config, "seed", 42)),
            normalize_mode=_get(config, "normalize_mode", "min-max"),
            state_normalize_mode=_get(config, "state_normalize_mode", "min-max"),
            action_mode=_get(config, "action_mode", LIBERO_ACTION_MODE),
            state_mode=_get(config, "state_mode", LIBERO_STATE_MODE),
            temporal_compression=int(_get(config, "temporal_compression", 4)),
            causal_temporal=bool(_get(config, "causal_temporal", True)),
            multiview=bool(_get(config, "multiview", True)),
            camera_layout=list(camera_layout),
            target_camera=_get(config, "target_camera", "head_camera"),
            training_video_rotation_degrees=int(
                _get(config, "training_video_rotation_degrees", 0)
            ),
            verify_stats_source=bool(_get(config, "verify_stats_source", True)),
            verify_known_repairs=bool(_get(config, "verify_known_repairs", True)),
            excluded_episodes=list(
                _get(
                    config,
                    "excluded_episodes",
                    [_KNOWN_CORRUPTED_EPISODE_KEY],
                )
            ),
        )

    def __init__(
        self,
        *,
        dataset_roots: Sequence[str | Path],
        action_stats_path: str | Path | None,
        benchmark_contract: Mapping[str, Any] | None,
        num_frames: int = 33,
        video_stride: int = 4,
        window_stride: int = 1,
        height: int = 384,
        width: int = 320,
        split: str = "train",
        val_ratio: float = 0.05,
        repeat: int = 1,
        seed: int = 42,
        normalize_mode: str | None = "min-max",
        state_normalize_mode: str | None = "min-max",
        action_mode: str = LIBERO_ACTION_MODE,
        state_mode: str = LIBERO_STATE_MODE,
        temporal_compression: int = 4,
        causal_temporal: bool = True,
        multiview: bool = True,
        camera_layout: Sequence[str] = LIBERO_CAMERA_LAYOUT,
        target_camera: str = "head_camera",
        training_video_rotation_degrees: int = 0,
        verify_stats_source: bool = True,
        verify_known_repairs: bool = True,
        excluded_episodes: Sequence[str] = (_KNOWN_CORRUPTED_EPISODE_KEY,),
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError("LIBERO split must be 'train' or 'val'")
        if not 0.0 <= val_ratio <= 1.0:
            raise ValueError("val_ratio must be in [0, 1]")
        if repeat <= 0 or window_stride <= 0:
            raise ValueError("repeat and window_stride must be positive")
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError(
                "LIBERO output resolution must be positive and divisible by 32"
            )
        if action_mode != LIBERO_ACTION_MODE or state_mode != LIBERO_STATE_MODE:
            raise ValueError(
                "LIBERO action_mode/state_mode differs from the frozen schema"
            )
        if not multiview or list(camera_layout) != list(LIBERO_CAMERA_LAYOUT):
            raise ValueError(
                "LIBERO training requires the frozen three-slot camera layout"
            )
        if target_camera != "head_camera":
            raise ValueError("LIBERO target_camera must be 'head_camera'")
        if training_video_rotation_degrees != 0:
            raise ValueError(
                "LIBERO LeRobot MP4 is already in training orientation; "
                "training_video_rotation_degrees must be 0"
            )

        self.dataset_type = "libero"
        self.schema_version = LIBERO_DATASET_SCHEMA_VERSION
        self.action_mode = action_mode
        self.state_mode = state_mode
        self._action_dim = LIBERO_ACTION_DIM
        self._state_dim = LIBERO_STATE_DIM
        self.num_frames = int(num_frames)
        self.num_action_steps = self.num_frames - 1
        self.video_stride = int(video_stride)
        self.window_stride = int(window_stride)
        self.temporal_compression = int(temporal_compression)
        self.causal_temporal = bool(causal_temporal)
        self._video_sample_indices, self.num_video_frames = _validate_temporal_contract(
            num_frames=self.num_frames,
            video_stride=self.video_stride,
            temporal_compression=self.temporal_compression,
            causal=self.causal_temporal,
        )
        self.height = int(height)
        self.width = int(width)
        self.split = split
        self.val_ratio = float(val_ratio)
        self.repeat = int(repeat)
        self.seed = int(seed)
        self.multiview = True
        self.camera_layout = list(camera_layout)
        self.target_camera = target_camera
        self.training_video_rotation_degrees = 0
        self.delta_action = False
        self.benchmark_contract = validate_libero_benchmark_contract(
            benchmark_contract or {}
        )
        self.excluded_episodes = frozenset(str(value) for value in excluded_episodes)
        if self.excluded_episodes != frozenset(
            self.benchmark_contract["excluded_training_episodes"]
        ):
            raise ValueError("LIBERO excluded_episodes differs from benchmark_contract")
        self._observed_exclusions: set[str] = set()

        self.normalize_mode = _normalization_mode(
            normalize_mode, field="normalize_mode"
        )
        self.state_normalize_mode = _normalization_mode(
            state_normalize_mode, field="state_normalize_mode"
        )
        if self.normalize_mode != "min-max" or self.state_normalize_mode != "min-max":
            raise ValueError(
                "the frozen LIBERO benchmark contract requires min-max action and state normalization"
            )

        candidates = tuple(Path(root).expanduser() for root in dataset_roots)
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(
                    f"LIBERO dataset root must not be a symlink: {candidate}"
                )
        self.dataset_roots = tuple(
            sorted(
                (candidate.resolve() for candidate in candidates),
                key=lambda root: root.name,
            )
        )
        if not self.dataset_roots:
            raise ValueError("LIBERO dataset_roots must not be empty")
        if len({root.name for root in self.dataset_roots}) != len(self.dataset_roots):
            raise ValueError("LIBERO dataset root basenames must be unique")
        for root in self.dataset_roots:
            if not root.is_dir():
                raise ValueError(f"invalid LIBERO dataset root: {root}")
        if verify_known_repairs:
            observed_names = tuple(root.name for root in self.dataset_roots)
            expected_names = tuple(sorted(LIBERO_TRAINING_DATASET_NAMES))
            if observed_names != expected_names:
                raise ValueError(
                    "LIBERO production dataset roots must use the frozen suite names; "
                    f"expected {expected_names!r}, got {observed_names!r}"
                )
            self._verify_known_data_repairs()

        if action_stats_path is None:
            raise ValueError("LIBERO requires an explicit action_stats_path")
        stats_candidate = Path(action_stats_path).expanduser()
        if stats_candidate.is_symlink():
            raise ValueError(
                f"LIBERO action_stats_path must not be a symlink: {stats_candidate}"
            )
        resolved_stats = stats_candidate.resolve()
        payload = load_libero_stats(resolved_stats)
        if verify_stats_source:
            if payload["schema_version"] == LIBERO_SELECTED_STATS_SCHEMA_VERSION:
                from omegaconf import OmegaConf

                stats_config = OmegaConf.create(
                    {
                        "dataloader": {
                            "dataset_roots": [str(root) for root in self.dataset_roots],
                            "excluded_episodes": sorted(self.excluded_episodes),
                            "num_frames": self.num_frames,
                            "repeat": self.repeat,
                            "seed": self.seed,
                            "split": self.split,
                            "val_ratio": self.val_ratio,
                            "window_stride": self.window_stride,
                        }
                    }
                )
                validate_libero_selected_stats_for_config(
                    payload,
                    stats_config,
                    verify_live_sources=True,
                    # The dedicated launcher already performs the one full
                    # numeric recomputation before constructing Trainer.  The
                    # dataset independently re-hashes the live manifest but
                    # does not read and reduce every selected row a second time.
                    verify_numeric_sources=False,
                )
            elif payload["source_manifest"] != build_source_manifest(
                self.dataset_roots
            ):
                raise ValueError(
                    "LIBERO action_stats source_manifest differs from dataset roots"
                )
        action_stats = payload[LIBERO_ACTION_MODE]
        state_stats = payload[LIBERO_STATE_MODE]
        _validate_stat_shape(
            action_stats, dim=LIBERO_ACTION_DIM, label=LIBERO_ACTION_MODE
        )
        _validate_stat_shape(state_stats, dim=LIBERO_STATE_DIM, label=LIBERO_STATE_MODE)
        self.action_stats_path = str(resolved_stats)
        self._action_stats = copy.deepcopy(action_stats)
        self._state_stats = copy.deepcopy(state_stats)
        self._action_normalizer = ActionNormalizer(
            mode=YAML_TO_NORM_MODE[self.normalize_mode], stats=action_stats
        )
        self._state_normalizer = Normalizer(
            mode=YAML_TO_NORM_MODE[self.state_normalize_mode], stats=state_stats
        )

        selection = select_libero_episodes(
            self.dataset_roots,
            split=self.split,
            val_ratio=self.val_ratio,
            seed=self.seed,
            excluded_episodes=tuple(self.excluded_episodes),
            require_all_exclusions=verify_known_repairs,
        )
        self._observed_exclusions = {episode.identity for episode in selection.excluded}
        for episode in selection.eligible:
            for video_key in LIBERO_VIDEO_KEY_TO_CAMERA:
                video_path = episode.video_path(video_key)
                if video_path.is_symlink() or not video_path.is_file():
                    raise ValueError(
                        "LIBERO episode video must be a regular non-symlink "
                        f"file: {video_path}"
                    )
        self._episodes = selection.selected
        self._windows = tuple(
            (episode_index, start)
            for episode_index, episode in enumerate(self._episodes)
            for start in range(0, episode.length - 1, self.window_stride)
        )
        if not self._windows:
            raise ValueError(
                "LIBERO dataset contains no valid obs(t)->action(t) windows"
            )

    def _verify_known_data_repairs(self) -> None:
        for root in self.dataset_roots:
            if root.name != _KNOWN_GOAL_DATASET:
                continue
            path = root / _KNOWN_CORRUPTED_WRIST_VIDEO
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"known LIBERO repair target is missing: {path}")
            actual = sha256_file(path)
            if actual == _KNOWN_CORRUPTED_WRIST_SHA256 and (
                _KNOWN_CORRUPTED_EPISODE_KEY in self.excluded_episodes
            ):
                continue
            if actual == _KNOWN_CORRUPTED_WRIST_SHA256:
                raise ValueError(
                    "LIBERO Goal wrist episode_000082 is the known corrupted asset; "
                    "exclude the complete episode in the saved data contract"
                )
            if actual == _KNOWN_INCOMPATIBLE_169_FRAME_PATCH_SHA256:
                raise ValueError(
                    "the available Isaac-GR00T episode_000082 patch has 169 frames "
                    "but this LeRobot snapshot has 129 rows; the assets must not be combined"
                )
            raise ValueError(
                "LIBERO Goal episode_000082 is not excluded and has no pinned "
                "snapshot-compatible repaired wrist video"
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_stats(self) -> dict[str, np.ndarray]:
        return copy.deepcopy(self._action_stats)

    @property
    def state_stats(self) -> dict[str, np.ndarray]:
        return copy.deepcopy(self._state_stats)

    def __len__(self) -> int:
        return len(self._windows) * self.repeat

    def _resolve_index(self, index: int) -> tuple[LiberoEpisode, int]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("LIBERO dataset index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index, start = self._windows[index % len(self._windows)]
        return self._episodes[episode_index], start

    def _read_episode_arrays(
        self, episode: LiberoEpisode
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return read_libero_episode_arrays(episode)

    def _read_video_indices(
        self, path: Path, requested_indices: Sequence[int]
    ) -> dict[int, Image.Image]:
        try:
            import av
        except ImportError as exc:  # pragma: no cover - av is a core dependency
            raise RuntimeError("LIBERO video decoding requires PyAV") from exc
        requested = set(int(index) for index in requested_indices)
        frames: dict[int, Image.Image] = {}
        with av.open(str(path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in requested:
                    frames[frame_index] = Image.fromarray(
                        frame.to_ndarray(format="rgb24"), mode="RGB"
                    )
                if len(frames) == len(requested):
                    break
        missing = sorted(requested - set(frames))
        if missing:
            raise ValueError(f"LIBERO video {path} is missing frames {missing}")
        return frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, start = self._resolve_index(index)
        states, actions, _task_indices = self._read_episode_arrays(episode)
        actual_len = min(self.num_frames, episode.length - start)
        if actual_len < 2:
            raise IndexError("LIBERO window has no obs(t)->action(t) label")

        state_window = states[start : start + actual_len]
        # LIBERO's relative command is aligned with the observation at the same
        # row. Keep only commands that have a corresponding next video/state.
        action_window = actions[start : start + actual_len - 1]
        state_padded = np.concatenate(
            [
                state_window,
                np.repeat(state_window[-1:], self.num_frames - actual_len, axis=0),
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        action_padded = np.zeros(
            (self.num_action_steps, LIBERO_ACTION_DIM), dtype=np.float32
        )
        action_padded[: actual_len - 1] = action_window
        state_normalized = self._state_normalizer.normalize(state_padded)
        action_normalized = self._action_normalizer.normalize(action_padded)

        raw_video_indices = [
            min(start + relative, start + actual_len - 1)
            for relative in self._video_sample_indices
        ]
        decoded = {
            video_key: self._read_video_indices(
                episode.video_path(video_key), sorted(set(raw_video_indices))
            )
            for video_key in LIBERO_VIDEO_KEY_TO_CAMERA
        }
        video = []
        for raw_index in raw_video_indices:
            frames_by_camera = {
                camera: decoded[video_key][raw_index]
                for video_key, camera in LIBERO_VIDEO_KEY_TO_CAMERA.items()
            }
            # The stored LeRobot MP4s are already in the canonical training
            # orientation. Only simulator observations are rotated by the eval
            # adapter; rotating here would create a train/eval mismatch.
            video.append(
                assemble_multiview_layout(
                    frames_by_camera,
                    self.camera_layout,
                    self.height,
                    self.width,
                )
            )

        action_mask = torch.tensor(
            [step < actual_len - 1 for step in range(self.num_action_steps)],
            dtype=torch.bool,
        )
        video_mask = torch.tensor(
            [relative < actual_len for relative in self._video_sample_indices],
            dtype=torch.bool,
        )
        return {
            "video": video,
            "first_frame_image": [video[0]],
            "action": torch.from_numpy(action_normalized),
            "action_mask": action_mask,
            "video_mask": video_mask,
            "proprio": torch.from_numpy(state_normalized[:1]),
            "proprio_mask": torch.tensor([True], dtype=torch.bool),
            "proprio_seq": torch.from_numpy(state_normalized),
            "num_clean_prefix_latent": 0,
            "num_clean_prefix_actions": 0,
            "prompt": format_prompt_for_inference(episode.task),
            "dataset_type": "libero",
            "dataset_name": episode.dataset,
            "episode_index": episode.episode_index,
            "episode_path": str(episode.data_path()),
            "episode_length": episode.length,
            "start_frame": start,
            "end_frame": start + actual_len,
            "task_name": episode.task,
            "task_index": episode.task_index,
            "action_alignment": "observation_t_to_action_t",
        }


__all__ = [
    "LIBERO_CAMERA_LAYOUT",
    "LIBERO_DATASET_SCHEMA_VERSION",
    "LIBERO_TRAINING_DATASET_NAMES",
    "LIBERO_VIDEO_KEY_TO_CAMERA",
    "LiberoEpisode",
    "LiberoLeRobotDataset",
    "canonical_libero_benchmark_contract",
    "validate_libero_benchmark_contract",
]
