"""Additive AV-3 paired reduced-scale screen for the frozen CACH-A4 graph.

The module deliberately reuses the frozen AV-2 real-data adapter and the
frozen A4-R2 vendor bridge.  Its only new semantics are the prospective
four-microbatch cohort, alternating correct-action training, the registered
1/2/4-horizon metric, and the AV-3 typed classifier.  Targets remain outside
every model task and enter only the positive loss and update-free metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import time
from types import MappingProxyType
from typing import Any, Final, Literal

import torch
from torch import Tensor, nn

from sana_wam.model import cach_av2_tiny_real_data_overfit_execution as av2


a4 = av2.a4

AV3_ARCHITECTURE_ID: Final = (
    "CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1"
)
AV3_CONFIG_SCHEMA: Final = "cach.cach_a4.av3_paired_reduced_scale.config.v1"
AV3_RESULT_SCHEMA: Final = "cach.cach_a4.av3_paired_reduced_scale.result.v1"
AV3_DATA_MANIFEST_SCHEMA: Final = (
    "cach.cach_a4.av3_paired_reduced_scale.data_selection.v1"
)
AV3_DATA_MANIFEST_PATH: Final = (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av3/"
    "CACH_A4_AV3_DATA_SELECTION.json"
)
AV3_DATA_MANIFEST_SHA256_PLACEHOLDER: Final = (
    "FUTURE_CACH_A4_AV3_DATA_SELECTION_FULL_SHA256"
)

AV3_TASK_NAME: Final = "adjust_bottle"
AV3_VARIANT: Final = "aloha-agilex_clean_50"
AV3_CAMERA: Final = "head_camera"
AV3_TRAIN_MICROBATCH_EPISODES: Final = (
    tuple(range(16, 24)),
    tuple(range(24, 32)),
)
AV3_HELDOUT_MICROBATCH_EPISODES: Final = (
    tuple(range(32, 40)),
    tuple(range(40, 48)),
)
AV3_EXCLUDED_AV2_EPISODES: Final = tuple(range(16))
AV3_SHUFFLE_PERMUTATION: Final = (1, 0, 3, 2, 5, 4, 7, 6)
AV3_HORIZONS: Final = (1, 2, 4)
AV3_HORIZON_WEIGHTS: Final = MappingProxyType({1: 0.20, 2: 0.30, 4: 0.50})
AV3_METRIC_EPSILON: Final = 1.0e-12
AV3_SCREEN_SEED: Final = 202608030
AV3_SHARED_INITIALIZATION_SEED: Final = 2026080331
AV3_CANDIDATE_ONLY_INITIALIZATION_SEED: Final = 2026080522
AV3_STEPS_PER_ARM: Final = 1000
AV3_LR_START: Final = 3.0e-3
AV3_LR_END: Final = 3.0e-5
AV3_GRAD_CLIP_NORM: Final = 1.0
AV3_MAX_WALL_MINUTES: Final = 60
AV3_REFERENCE_PARAMETER_COUNT: Final = 752_439
AV3_CANDIDATE_PARAMETER_COUNT: Final = 766_199
AV3_ACTION_VARIANCE_MIN: Final = 1.0e-8
AV3_SHUFFLE_MSE_MIN: Final = 1.0e-8
AV3_MOTION_ENERGY_MIN: Final = 1.0e-8
AV3_SELECTION_SHA256: Final = (
    "49a56dc087d693a4f25005095f390996dab8639eb4e387bf8c70b872dfeb50d9"
)

AV3ActionMode = Literal["correct", "shuffle", "no_action"]


class AV3ExecutionError(RuntimeError):
    """The frozen AV-3 data, implementation, or execution contract failed."""


@dataclass(frozen=True)
class AV3RealDataBundle:
    """Four ordered batch-8 microbatches; targets never become model inputs."""

    dataset_root: str
    train_microbatches: tuple[av2.AV2RealSplit, av2.AV2RealSplit]
    heldout_microbatches: tuple[av2.AV2RealSplit, av2.AV2RealSplit]
    selection_sha256: str
    manifest_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AV3ExecutionError(f"{label} must be a mapping")
    return value


def _selection_digest_material() -> dict[str, object]:
    return {
        "action_rows": [1, 33],
        "camera": AV3_CAMERA,
        "frame_indices": [0, 8, 16, 24, 32],
        "heldout_episode_ids": list(range(32, 48)),
        "normalize": None,
        "raw_interval": [0, 33],
        "target": "channel_mean_div_255_relative_to_frame0",
        "task": AV3_TASK_NAME,
        "train_episode_ids": list(range(16, 32)),
        "variant": AV3_VARIANT,
    }


def _canonical_compact_no_lf(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _verify_manifest_selection(
    value: Mapping[str, object], *, dataset_root: str
) -> str:
    expected_top_level = {
        "action_normalization": None,
        "action_rows": [1, 33],
        "camera": AV3_CAMERA,
        "dataset_root": dataset_root,
        "frame_indices": [0, 8, 16, 24, 32],
        "heldout_episode_ids": list(range(32, 48)),
        "heldout_microbatches": [
            list(item) for item in AV3_HELDOUT_MICROBATCH_EPISODES
        ],
        "raw_rows": [0, 33],
        "task": AV3_TASK_NAME,
        "train_episode_ids": list(range(16, 32)),
        "train_microbatches": [list(item) for item in AV3_TRAIN_MICROBATCH_EPISODES],
        "variant": AV3_VARIANT,
        "within_microbatch_shuffle_permutation": list(AV3_SHUFFLE_PERMUTATION),
    }
    for name, expected in expected_top_level.items():
        if value.get(name) != expected or type(value.get(name)) is not type(expected):
            raise AV3ExecutionError(f"AV-3 manifest {name} differs")
    material = _selection_digest_material()
    canonical_json = _canonical_compact_no_lf(material)
    observed_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    if observed_sha != AV3_SELECTION_SHA256:
        raise AV3ExecutionError("internal AV-3 selection SHA constant differs")
    digest = _require_mapping(
        value.get("selection_digest"), label="manifest.selection_digest"
    )
    expected_digest = {
        "canonical_json": canonical_json,
        "payload_has_trailing_lf": False,
        "serialization": "UTF8_JSON_SORTED_KEYS_COMPACT_SEPARATORS_NO_TRAILING_LF",
        "sha256": AV3_SELECTION_SHA256,
    }
    if dict(digest) != expected_digest:
        raise AV3ExecutionError("AV-3 manifest selection digest differs")
    return observed_sha


def _expected_file_rows(dataset_root: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for split, microbatches in (
        ("train", AV3_TRAIN_MICROBATCH_EPISODES),
        ("heldout", AV3_HELDOUT_MICROBATCH_EPISODES),
    ):
        for microbatch_index, episode_ids in enumerate(microbatches):
            for episode_id in episode_ids:
                rows.append(
                    {
                        "episode_id": episode_id,
                        "microbatch_index": microbatch_index,
                        "path": str(av2._episode_path(dataset_root, episode_id)),
                        "split": split,
                    }
                )
    return tuple(rows)


def _verify_selected_files(
    dataset_root: Path, selected_files: object
) -> Mapping[str, str]:
    if not isinstance(selected_files, list):
        raise AV3ExecutionError("manifest.selected_files must be a list")
    expected_rows = _expected_file_rows(dataset_root)
    if len(selected_files) != len(expected_rows):
        raise AV3ExecutionError("manifest selected-file count differs from AV-3")
    verified: dict[str, str] = {}
    for index, (raw, expected) in enumerate(zip(selected_files, expected_rows, strict=True)):
        row = _require_mapping(raw, label=f"selected_files[{index}]")
        if frozenset(row) != {
            "episode_id",
            "episode_length",
            "microbatch_index",
            "path",
            "sha256",
            "size_bytes",
            "split",
        }:
            raise AV3ExecutionError("selected-file keys differ from AV-3")
        for name in ("episode_id", "microbatch_index", "path", "split"):
            if row.get(name) != expected[name] or type(row.get(name)) is not type(
                expected[name]
            ):
                raise AV3ExecutionError(f"selected-file {name} differs at index {index}")
        size_bytes = row.get("size_bytes")
        episode_length = row.get("episode_length")
        digest = row.get("sha256")
        if (
            type(size_bytes) is not int
            or size_bytes <= 0
            or type(episode_length) is not int
            or episode_length < 33
            or not _is_sha256(digest)
        ):
            raise AV3ExecutionError("selected-file size/SHA is malformed")
        path = Path(str(row["path"]))
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise AV3ExecutionError(f"selected file is not a regular non-symlink: {path}")
        if status.st_size != size_bytes:
            raise AV3ExecutionError(f"selected file size drifted: {path}")
        observed = _file_sha256(path)
        if observed != digest:
            raise AV3ExecutionError(f"selected file SHA drifted: {path}")
        verified[str(path)] = observed
    return MappingProxyType(verified)


def _derived_manifest_table(value: object) -> Mapping[tuple[str, int], Mapping[str, object]]:
    if not isinstance(value, list) or len(value) != 4:
        raise AV3ExecutionError("manifest.derived_microbatches must contain four entries")
    result: dict[tuple[str, int], Mapping[str, object]] = {}
    for raw in value:
        row = _require_mapping(raw, label="derived_microbatch")
        if frozenset(row) != {
            "episode_ids",
            "microbatch_index",
            "split",
            "tensor_sha256",
        }:
            raise AV3ExecutionError("derived-microbatch keys differ from AV-3")
        split = row.get("split")
        microbatch_index = row.get("microbatch_index")
        if split not in ("train", "heldout") or type(microbatch_index) is not int:
            raise AV3ExecutionError("derived-microbatch identity is malformed")
        key = (str(split), microbatch_index)
        expected_ids = (
            AV3_TRAIN_MICROBATCH_EPISODES
            if split == "train"
            else AV3_HELDOUT_MICROBATCH_EPISODES
        )
        if microbatch_index not in (0, 1) or row.get("episode_ids") != list(
            expected_ids[microbatch_index]
        ):
            raise AV3ExecutionError("derived-microbatch episodes differ")
        hashes = _require_mapping(row.get("tensor_sha256"), label="tensor_sha256")
        if frozenset(hashes) != {
            "action_valid_mask",
            "actions",
            "frame_valid_mask",
            "noisy_video",
            "proprio",
            "target",
        }:
            raise AV3ExecutionError("derived tensor SHA keys differ")
        if not all(_is_sha256(item) for item in hashes.values()):
            raise AV3ExecutionError("derived tensor SHA is malformed")
        if key in result:
            raise AV3ExecutionError("duplicate derived-microbatch identity")
        result[key] = row
    if frozenset(result) != {
        ("train", 0),
        ("train", 1),
        ("heldout", 0),
        ("heldout", 1),
    }:
        raise AV3ExecutionError("derived-microbatch coverage differs")
    return MappingProxyType(result)


def load_av3_real_data(
    dataset_root: str | os.PathLike[str],
    manifest: Mapping[str, object],
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    expected_manifest_sha256: str | None = None,
) -> AV3RealDataBundle:
    """Load only the frozen four microbatches and verify every manifest pin."""

    root = Path(dataset_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise AV3ExecutionError("dataset_root must resolve to a directory")
    value = _require_mapping(manifest, label="manifest")
    if value.get("schema") != AV3_DATA_MANIFEST_SCHEMA:
        raise AV3ExecutionError("AV-3 data manifest schema differs")
    expected_selection_sha = _verify_manifest_selection(value, dataset_root=str(root))

    manifest_sha = _canonical_sha256(value)
    if expected_manifest_sha256 is not None:
        if not _is_sha256(expected_manifest_sha256) or manifest_path is None:
            raise AV3ExecutionError("manifest file pin is incomplete")
        resolved_manifest_path = Path(manifest_path).resolve(strict=True)
        if resolved_manifest_path.read_bytes() != _canonical_bytes(value):
            raise AV3ExecutionError("AV-3 manifest file is not canonical JSON+LF")
        observed_manifest_sha = _file_sha256(resolved_manifest_path)
        if observed_manifest_sha != expected_manifest_sha256:
            raise AV3ExecutionError("AV-3 manifest file SHA differs")
        manifest_sha = observed_manifest_sha

    verified_files = _verify_selected_files(root, value.get("selected_files"))
    derived = _derived_manifest_table(value.get("derived_microbatches"))
    train = tuple(
        av2._load_split(root, name="train", episode_ids=episode_ids)
        for episode_ids in AV3_TRAIN_MICROBATCH_EPISODES
    )
    heldout = tuple(
        av2._load_split(root, name="heldout", episode_ids=episode_ids)
        for episode_ids in AV3_HELDOUT_MICROBATCH_EPISODES
    )
    if len(train) != 2 or len(heldout) != 2:
        raise AV3ExecutionError("AV-3 microbatch count differs")
    for split_name, microbatches in (("train", train), ("heldout", heldout)):
        for microbatch_index, split in enumerate(microbatches):
            av2._validate_real_split(split)
            expected_row = derived[(split_name, microbatch_index)]
            observed_tensor_sha = {
                **dict(split.tensor_sha256),
                "action_valid_mask": av2._tensor_sha256(split.action_valid_mask),
                "frame_valid_mask": av2._tensor_sha256(split.frame_valid_mask),
            }
            if observed_tensor_sha != dict(expected_row["tensor_sha256"]):
                raise AV3ExecutionError("derived tensor SHA differs from manifest")
            if dict(split.episode_file_sha256) != {
                path: verified_files[path] for path in split.episode_file_sha256
            }:
                raise AV3ExecutionError("episode file SHA differs from manifest")

    train_windows = tuple(window for split in train for window in split.windows)
    heldout_windows = tuple(window for split in heldout for window in split.windows)
    av2.av2_scaffold.assert_train_holdout_disjoint(train_windows, heldout_windows)
    observed_ids = {int(window.episode_id) for window in (*train_windows, *heldout_windows)}
    expected_ids = set(range(16, 48))
    if observed_ids != expected_ids or observed_ids & set(AV3_EXCLUDED_AV2_EPISODES):
        raise AV3ExecutionError("AV-3/AV-2 episode identity contract differs")
    return AV3RealDataBundle(
        dataset_root=str(root),
        train_microbatches=(train[0], train[1]),
        heldout_microbatches=(heldout[0], heldout[1]),
        selection_sha256=expected_selection_sha,
        manifest_sha256=manifest_sha,
    )


def _split_action_statistics(
    microbatches: Sequence[av2.AV2RealSplit],
) -> Mapping[str, float | bool]:
    actions: list[Tensor] = []
    shuffled: list[Tensor] = []
    finite = True
    masks_nonempty = True
    permutation = torch.tensor(AV3_SHUFFLE_PERMUTATION, dtype=torch.long)
    for split in microbatches:
        av2._validate_real_split(split)
        active = split.actions.masked_select(
            split.action_valid_mask.unsqueeze(-1).expand_as(split.actions)
        )
        shifted = split.actions.index_select(0, permutation).masked_select(
            split.action_valid_mask.unsqueeze(-1).expand_as(split.actions)
        )
        actions.append(active.double())
        shuffled.append(shifted.double())
        finite = finite and bool(torch.isfinite(split.actions).all().item())
        masks_nonempty = masks_nonempty and bool(split.action_valid_mask.any().item())
    values = torch.cat(actions)
    shuffled_values = torch.cat(shuffled)
    variance = float(values.var(unbiased=False).item())
    shuffle_mse = float((values - shuffled_values).square().mean().item())
    return MappingProxyType(
        {
            "action_mask_nonempty": masks_nonempty,
            "action_variance": variance,
            "finite": finite and math.isfinite(variance) and math.isfinite(shuffle_mse),
            "shuffle_mismatch_mse": shuffle_mse,
        }
    )


def _motion_energy(microbatches: Sequence[av2.AV2RealSplit]) -> Mapping[str, float]:
    values: dict[str, float] = {}
    for horizon in AV3_HORIZONS:
        numerator = 0.0
        count = 0
        for split in microbatches:
            target = split.target.double()
            delta = target[:, horizon] - target[:, 0]
            numerator += float(delta.square().sum().item())
            count += delta.numel()
        if count == 0:
            raise AV3ExecutionError("motion-energy denominator is empty")
        values[str(horizon)] = numerator / float(count)
    return MappingProxyType(values)


def assess_av3_data_adequacy(bundle: AV3RealDataBundle) -> Mapping[str, object]:
    """Apply the strict prospective split and per-horizon adequacy gates."""

    reports: dict[str, object] = {}
    for name, microbatches in (
        ("train", bundle.train_microbatches),
        ("heldout", bundle.heldout_microbatches),
    ):
        action = dict(_split_action_statistics(microbatches))
        motion = dict(_motion_energy(microbatches))
        frame_masks_nonempty = all(
            bool(split.frame_valid_mask[:, 1:].any().item()) for split in microbatches
        )
        adequate = (
            bool(action["finite"])
            and bool(action["action_mask_nonempty"])
            and frame_masks_nonempty
            and float(action["action_variance"]) > AV3_ACTION_VARIANCE_MIN
            and float(action["shuffle_mismatch_mse"]) > AV3_SHUFFLE_MSE_MIN
            and all(value > AV3_MOTION_ENERGY_MIN for value in motion.values())
        )
        reports[name] = {
            **action,
            "frame_mask_nonempty": frame_masks_nonempty,
            "motion_energy_by_horizon": motion,
            "adequate": adequate,
        }
    heldout_microbatch_motion = [
        dict(_motion_energy((split,))) for split in bundle.heldout_microbatches
    ]
    heldout_microbatches_adequate = all(
        all(value > AV3_MOTION_ENERGY_MIN for value in report.values())
        for report in heldout_microbatch_motion
    )
    reports["heldout_microbatch_motion_energy"] = heldout_microbatch_motion
    reports["all_adequate"] = bool(
        reports["train"]["adequate"]  # type: ignore[index]
        and reports["heldout"]["adequate"]  # type: ignore[index]
        and heldout_microbatches_adequate
    )
    return MappingProxyType(reports)


def _prediction_tensor(value: Tensor, *, label: str) -> Tensor:
    if tuple(value.shape) != (8, 5, 3, 1, 1) or value.dtype is not torch.float32:
        raise AV3ExecutionError(f"{label} must be float32 [8,5,3,1,1]")
    if not bool(torch.isfinite(value).all().item()):
        raise AV3ExecutionError(f"{label} is non-finite")
    return value


def compute_av3_metrics(
    predictions: Mapping[str, Mapping[str, Sequence[Tensor]]],
    targets: Sequence[Tensor],
    frame_masks: Sequence[Tensor],
) -> Mapping[str, object]:
    """Compute the frozen V/D/Q/PRIMARY metric with float64 accumulation."""

    if len(targets) != 2 or len(frame_masks) != 2:
        raise AV3ExecutionError("AV-3 metrics require two heldout microbatches")
    for index, (target, mask) in enumerate(zip(targets, frame_masks, strict=True)):
        _prediction_tensor(target, label=f"target[{index}]")
        if (
            tuple(mask.shape) != (8, 5)
            or mask.dtype is not torch.bool
            or not bool(mask[:, list(AV3_HORIZONS)].all().item())
        ):
            raise AV3ExecutionError("heldout metric mask differs from AV-3")
    if frozenset(predictions) != {"reference", "candidate"}:
        raise AV3ExecutionError("metric arms differ from AV-3")
    for arm in ("reference", "candidate"):
        if frozenset(predictions[arm]) != {"correct", "shuffle", "no_action"}:
            raise AV3ExecutionError("metric modes differ from AV-3")
        if any(len(predictions[arm][mode]) != 2 for mode in predictions[arm]):
            raise AV3ExecutionError("metric microbatch count differs from AV-3")

    q: dict[str, float] = {}
    for horizon in AV3_HORIZONS:
        numerator = 0.0
        for target in targets:
            target64 = target.double()
            numerator += float(
                ((target64[:, horizon] - target64[:, 0]).square().sum()).item()
            )
        q[str(horizon)] = numerator / 48.0

    video: dict[str, dict[str, dict[str, float]]] = {}
    delta: dict[str, dict[str, dict[str, float]]] = {}
    primary: dict[str, dict[str, float]] = {}
    aggregates: dict[str, dict[str, Mapping[str, float]]] = {}
    for arm in ("reference", "candidate"):
        video[arm] = {}
        delta[arm] = {}
        primary[arm] = {}
        aggregates[arm] = {}
        for mode in ("correct", "shuffle", "no_action"):
            v_h: dict[str, float] = {}
            d_h: dict[str, float] = {}
            for horizon in AV3_HORIZONS:
                v_numerator = 0.0
                d_numerator = 0.0
                for index, target in enumerate(targets):
                    prediction = _prediction_tensor(
                        predictions[arm][mode][index],
                        label=f"prediction[{arm},{mode},{index}]",
                    ).double()
                    target64 = target.double()
                    v_numerator += float(
                        ((prediction[:, horizon] - target64[:, horizon]).square().sum()).item()
                    )
                    d_numerator += float(
                        (
                            (
                                (prediction[:, horizon] - prediction[:, 0])
                                - (target64[:, horizon] - target64[:, 0])
                            )
                            .square()
                            .sum()
                        ).item()
                    )
                v_h[str(horizon)] = v_numerator / 48.0
                d_h[str(horizon)] = d_numerator / 48.0
            v124 = sum(AV3_HORIZON_WEIGHTS[h] * v_h[str(h)] for h in AV3_HORIZONS)
            d124 = sum(AV3_HORIZON_WEIGHTS[h] * d_h[str(h)] for h in AV3_HORIZONS)
            q124 = sum(AV3_HORIZON_WEIGHTS[h] * q[str(h)] for h in AV3_HORIZONS)
            video_nmse = v124 / max(q124, AV3_METRIC_EPSILON)
            delta_nmse = d124 / max(q124, AV3_METRIC_EPSILON)
            value = 0.5 * video_nmse + 0.5 * delta_nmse
            video[arm][mode] = v_h
            delta[arm][mode] = d_h
            aggregates[arm][mode] = MappingProxyType(
                {
                    "D124": d124,
                    "Q124": q124,
                    "V124": v124,
                    "delta_nmse": delta_nmse,
                    "video_nmse": video_nmse,
                }
            )
            primary[arm][mode] = value

    reference_predictions = predictions["reference"]
    reference_modes_bitwise_equal = all(
        torch.equal(reference_predictions["correct"][index], reference_predictions[mode][index])
        for index in range(2)
        for mode in ("shuffle", "no_action")
    )
    candidate_correct = primary["candidate"]["correct"]
    reference_correct = primary["reference"]["correct"]
    decision = {
        "candidate_gain_vs_reference": 1.0
        - candidate_correct / max(reference_correct, AV3_METRIC_EPSILON),
        "candidate_to_reference_primary_ratio": candidate_correct
        / max(reference_correct, AV3_METRIC_EPSILON),
        "correct_vs_no_action_improvement": 1.0
        - candidate_correct
        / max(primary["candidate"]["no_action"], AV3_METRIC_EPSILON),
        "correct_vs_shuffle_improvement": 1.0
        - candidate_correct
        / max(primary["candidate"]["shuffle"], AV3_METRIC_EPSILON),
        "h1_candidate_vs_reference_ratio": video["candidate"]["correct"]["1"]
        / max(video["reference"]["correct"]["1"], AV3_METRIC_EPSILON),
    }
    if not all(math.isfinite(value) for value in (*q.values(), *decision.values())):
        raise AV3ExecutionError("AV-3 decision metric is non-finite")
    return MappingProxyType(
        {
            "D": delta,
            "Q": q,
            "V": video,
            "aggregates": aggregates,
            "decision_metrics": decision,
            "primary": primary,
            "reference_modes_bitwise_equal": reference_modes_bitwise_equal,
        }
    )


def classify_av3(
    *, all_validity: bool, metrics: Mapping[str, object]
) -> tuple[str, str | None, Mapping[str, bool]]:
    """Apply the frozen GO/strong-stop/exhausted-review AV-3 state machine."""

    required = (
        "candidate_gain_vs_reference",
        "candidate_to_reference_primary_ratio",
        "correct_vs_shuffle_improvement",
        "h1_candidate_vs_reference_ratio",
    )
    if any(name not in metrics for name in required):
        raise AV3ExecutionError("AV-3 classifier metrics are incomplete")
    values = {name: float(metrics[name]) for name in required}
    if not all(math.isfinite(value) for value in values.values()):
        return "NUMERICAL_INVALID", "NONFINITE_DECISION_METRIC", MappingProxyType({})
    checks = {
        "candidate_gain_vs_reference": values["candidate_gain_vs_reference"] >= 0.05,
        "correct_vs_shuffle_improvement": values["correct_vs_shuffle_improvement"]
        >= 0.05,
        "h1_candidate_vs_reference_ratio": values["h1_candidate_vs_reference_ratio"]
        <= 1.05,
    }
    if not all_validity:
        return "INVALID_RUN", "AV3_VALIDITY_FAILURE", MappingProxyType(checks)
    if all(checks.values()):
        return "GO", None, MappingProxyType(checks)
    strong_stop = (
        values["candidate_gain_vs_reference"] <= 0.0
        and values["correct_vs_shuffle_improvement"] <= 0.0
    ) or values["candidate_to_reference_primary_ratio"] >= 1.25
    if strong_stop:
        return "REDUCED_ARCH_STOP", "AV3_STRONG_STOP_LINE", MappingProxyType(checks)
    return (
        "REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED",
        "AV3_REVIEW_BAND_TOKEN_CONSUMED",
        MappingProxyType(checks),
    )


def _microbatch_schedule(step: int) -> int:
    if type(step) is not int or not 0 <= step < AV3_STEPS_PER_ARM:
        raise AV3ExecutionError("optimizer step is outside the AV-3 schedule")
    return step % 2


def _train_serial_arm(
    arm: nn.Module,
    *,
    names: Sequence[str],
    parameters: Sequence[nn.Parameter],
    tasks: Sequence[a4.A4Task],
    targets: Sequence[Tensor],
    masks: Sequence[Tensor],
    deadline: float,
    label: str,
    progress_callback: Callable[[Mapping[str, object]], None] | None,
    diagnostic_parameter_groups: Mapping[str, Sequence[nn.Parameter]] | None = None,
) -> Mapping[str, object]:
    if not (len(tasks) == len(targets) == len(masks) == 2):
        raise AV3ExecutionError("training requires exactly two microbatches")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=AV3_LR_START,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    loss_trace: list[float] = []
    lr_trace: list[float] = []
    microbatch_trace: list[int] = []
    step0_gradient_rms = 0.0
    step0_gradient_rms_by_scope: dict[str, float] = {}
    clip_norm_max = 0.0
    arm.train(True)
    for step in range(AV3_STEPS_PER_ARM):
        if time.monotonic() >= deadline:
            raise AV3ExecutionError("AV-3 exceeded the total wall-clock budget")
        microbatch_index = _microbatch_schedule(step)
        microbatch_trace.append(microbatch_index)
        lr_trace.append(
            av2._set_cosine_lr(
                optimizer,
                step=step,
                total_steps=AV3_STEPS_PER_ARM,
                start=AV3_LR_START,
                end=AV3_LR_END,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        task = tasks[microbatch_index]
        av2.assert_target_sentinels(task)
        output = arm(task, mode="correct")
        if not bool(torch.isfinite(output.video_prediction).all().item()):
            raise AV3ExecutionError("AV-3 training output is non-finite")
        loss = av2.masked_future_mse(
            output.video_prediction,
            targets[microbatch_index],
            masks[microbatch_index],
        )
        if not bool(torch.isfinite(loss).item()):
            raise AV3ExecutionError("AV-3 training loss is non-finite")
        loss_trace.append(float(loss.detach().item()))
        loss.backward()
        if step == 0:
            step0_gradient_rms = av2._gradient_rms(parameters)
            if diagnostic_parameter_groups is not None:
                step0_gradient_rms_by_scope = {
                    name: av2._gradient_rms(scope)
                    for name, scope in diagnostic_parameter_groups.items()
                }
        clip = float(
            torch.nn.utils.clip_grad_norm_(parameters, AV3_GRAD_CLIP_NORM)
            .detach()
            .item()
        )
        if not math.isfinite(clip):
            raise AV3ExecutionError("AV-3 gradient norm is non-finite")
        clip_norm_max = max(clip_norm_max, clip)
        optimizer.step()
        if progress_callback is not None and (
            step == 0 or (step + 1) % 100 == 0 or step + 1 == AV3_STEPS_PER_ARM
        ):
            progress_callback(
                {
                    "arm": label,
                    "loss": loss_trace[-1],
                    "lr": lr_trace[-1],
                    "microbatch_index": microbatch_index,
                    "mode": "correct",
                    "step": step + 1,
                    "steps": AV3_STEPS_PER_ARM,
                }
            )
    return MappingProxyType(
        {
            "clip_norm_max": clip_norm_max,
            "final_loss_pre_update": loss_trace[-1],
            "initial_loss_pre_update": loss_trace[0],
            "loss_trace": tuple(loss_trace),
            "lr_end_observed": lr_trace[-1],
            "lr_start_observed": lr_trace[0],
            "microbatch_trace": tuple(microbatch_trace),
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "parameter_names": tuple(names),
            "step0_gradient_rms": step0_gradient_rms,
            "step0_gradient_rms_by_scope": step0_gradient_rms_by_scope,
            "steps_completed": len(loss_trace),
            "train_action_mode": "correct",
        }
    )


def _evaluate_outputs(
    arm: nn.Module,
    tasks: Sequence[a4.A4Task],
    modes: Sequence[str],
) -> Mapping[str, tuple[a4.A4SequenceOutput, a4.A4SequenceOutput]]:
    if len(tasks) != 2:
        raise AV3ExecutionError("evaluation requires two microbatches")
    was_training = arm.training
    arm.train(False)
    outputs: dict[str, tuple[a4.A4SequenceOutput, a4.A4SequenceOutput]] = {}
    with torch.no_grad():
        for mode in modes:
            values = []
            for task in tasks:
                av2.assert_target_sentinels(task)
                output = arm(task, mode=mode)
                _prediction_tensor(output.video_prediction, label=f"{mode} output")
                values.append(output)
            outputs[mode] = (values[0], values[1])
    arm.train(was_training)
    return MappingProxyType(outputs)


def _predictions(
    outputs: Mapping[str, Sequence[a4.A4SequenceOutput]],
) -> Mapping[str, tuple[Tensor, Tensor]]:
    return MappingProxyType(
        {
            mode: (values[0].video_prediction, values[1].video_prediction)
            for mode, values in outputs.items()
            if mode in ("correct", "shuffle", "no_action")
        }
    )


def _mean_correct_loss(
    arm: nn.Module,
    tasks: Sequence[a4.A4Task],
    targets: Sequence[Tensor],
    masks: Sequence[Tensor],
) -> float:
    outputs = _evaluate_outputs(arm, tasks, ("correct",))["correct"]
    values = [
        float(
            av2.masked_future_mse(output.video_prediction, target, mask)
            .detach()
            .double()
            .item()
        )
        for output, target, mask in zip(outputs, targets, masks, strict=True)
    ]
    return sum(values) / 2.0


def _config_mapping(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    return _require_mapping(config.get(name), label=f"config.{name}")


def validate_av3_config(
    config: Mapping[str, object], *, allow_manifest_sha_placeholder: bool = False
) -> None:
    """Reject materialized config drift before any data/model execution."""

    if config.get("schema") != AV3_CONFIG_SCHEMA:
        raise AV3ExecutionError("AV-3 config schema differs")
    if config.get("architecture_id") != AV3_ARCHITECTURE_ID:
        raise AV3ExecutionError("AV-3 architecture id differs")
    data = _config_mapping(config, "data_contract")
    if data.get("manifest_path") != AV3_DATA_MANIFEST_PATH:
        raise AV3ExecutionError("AV-3 data manifest path differs")
    manifest_sha = data.get("manifest_sha256")
    if not _is_sha256(manifest_sha) and not (
        allow_manifest_sha_placeholder
        and manifest_sha == AV3_DATA_MANIFEST_SHA256_PLACEHOLDER
    ):
        raise AV3ExecutionError("AV-3 data manifest SHA is unresolved")
    expected_microbatches = {
        "train_microbatches": [list(value) for value in AV3_TRAIN_MICROBATCH_EPISODES],
        "heldout_microbatches": [
            list(value) for value in AV3_HELDOUT_MICROBATCH_EPISODES
        ],
    }
    for name, expected in expected_microbatches.items():
        if data.get(name) != expected:
            raise AV3ExecutionError(f"AV-3 {name} differs")
    training = _config_mapping(config, "training_contract")
    expected_training = {
        "candidate_only_initialization_seed": AV3_CANDIDATE_ONLY_INITIALIZATION_SEED,
        "checkpoint_load": False,
        "checkpoint_save": False,
        "final_model_state_save": False,
        "gradient_clip_global_l2": AV3_GRAD_CLIP_NORM,
        "lr_end": AV3_LR_END,
        "lr_schedule": "COSINE",
        "lr_start": AV3_LR_START,
        "optimizer": "AdamW",
        "optimizer_steps_per_arm": AV3_STEPS_PER_ARM,
        "screen_seed": AV3_SCREEN_SEED,
        "serial_arm_order": ["reference", "candidate"],
        "shared_initialization_seed": AV3_SHARED_INITIALIZATION_SEED,
        "train_action_mode": "CORRECT_ONLY",
        "train_batch_schedule": "STEP_INDEX_MODULO_2",
    }
    for name, expected in expected_training.items():
        if training.get(name) != expected or type(training.get(name)) is not type(expected):
            raise AV3ExecutionError(f"AV-3 training_contract.{name} differs")


def _require_before_deadline(deadline: float, *, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise AV3ExecutionError(f"AV-3 total deadline expired at {stage}")


def run_av3_paired_reduced_scale(
    config: Mapping[str, object],
    *,
    expected_gpu_uuid: str,
    total_deadline_monotonic: float,
    source_data_and_predecessor_pins_match: bool,
    manifest: Mapping[str, object],
    manifest_path: str | os.PathLike[str],
    theta0_callback: Callable[[Mapping[str, object]], None],
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    preloaded_bundle: AV3RealDataBundle | None = None,
) -> Mapping[str, object]:
    """Execute the frozen seed-0 pair; the external runner owns root lifecycle."""

    started = time.monotonic()
    validate_av3_config(config)
    resources = _config_mapping(config, "resource_contract")
    if (
        isinstance(total_deadline_monotonic, bool)
        or not isinstance(total_deadline_monotonic, (int, float))
        or not math.isfinite(float(total_deadline_monotonic))
    ):
        raise AV3ExecutionError("AV-3 total deadline is malformed")
    deadline = float(total_deadline_monotonic)
    if deadline - started > float(resources["max_total_wall_minutes"]) * 60.0:
        raise AV3ExecutionError("AV-3 deadline grants more than the total budget")
    _require_before_deadline(deadline, stage="orchestrator_entry")
    if source_data_and_predecessor_pins_match is not True:
        raise AV3ExecutionError("source/data/predecessor pin proof is absent")
    if not callable(theta0_callback):
        raise AV3ExecutionError("theta0 callback is required before first forward")
    data = _config_mapping(config, "data_contract")
    if preloaded_bundle is None:
        bundle = load_av3_real_data(
            str(data["dataset_root"]),
            manifest,
            manifest_path=manifest_path,
            expected_manifest_sha256=str(data["manifest_sha256"]),
        )
    else:
        if not isinstance(preloaded_bundle, AV3RealDataBundle):
            raise AV3ExecutionError("preloaded AV-3 bundle has the wrong type")
        configured_root = str(Path(str(data["dataset_root"])).expanduser().resolve(strict=True))
        if (
            preloaded_bundle.dataset_root != configured_root
            or preloaded_bundle.manifest_sha256 != data["manifest_sha256"]
            or preloaded_bundle.selection_sha256 != AV3_SELECTION_SHA256
        ):
            raise AV3ExecutionError("preloaded AV-3 bundle pins differ from config")
        for split in (
            *preloaded_bundle.train_microbatches,
            *preloaded_bundle.heldout_microbatches,
        ):
            av2._validate_real_split(split)
        observed_preloaded_ids = {
            int(window.episode_id)
            for split in (
                *preloaded_bundle.train_microbatches,
                *preloaded_bundle.heldout_microbatches,
            )
            for window in split.windows
        }
        if observed_preloaded_ids != set(range(16, 48)):
            raise AV3ExecutionError("preloaded AV-3 episode identities differ")
        for split, expected_ids in zip(
            preloaded_bundle.train_microbatches
            + preloaded_bundle.heldout_microbatches,
            AV3_TRAIN_MICROBATCH_EPISODES + AV3_HELDOUT_MICROBATCH_EPISODES,
            strict=True,
        ):
            if tuple(int(window.episode_id) for window in split.windows) != expected_ids:
                raise AV3ExecutionError("preloaded AV-3 microbatch order differs")
        bundle = preloaded_bundle
    adequacy = assess_av3_data_adequacy(bundle)
    if not bool(adequacy["all_adequate"]):
        raise AV3ExecutionError("fixed AV-3 data cohort is inadequate")
    _require_before_deadline(deadline, stage="post_data_adequacy")

    hardware = _config_mapping(config, "hardware_contract")
    if (
        hardware.get("gpu_uuid") != expected_gpu_uuid
        or hardware.get("logical_device") != "cuda:0"
        or hardware.get("gpu_count") != 1
    ):
        raise AV3ExecutionError("hardware binding differs from AV-3")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu_uuid:
        raise AV3ExecutionError("CUDA_VISIBLE_DEVICES differs from AV-3")
    if os.environ.get("FUSED_GDN_PRECISION") != "0":
        raise AV3ExecutionError("FUSED_GDN_PRECISION must equal 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AV3ExecutionError("AV-3 requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(AV3_SCREEN_SEED)
    torch.cuda.manual_seed_all(AV3_SCREEN_SEED)
    _require_before_deadline(deadline, stage="post_cuda_binding")

    train_tasks = tuple(
        av2.materialize_a4_task(split, device=device)
        for split in bundle.train_microbatches
    )
    heldout_tasks = tuple(
        av2.materialize_a4_task(split, device=device)
        for split in bundle.heldout_microbatches
    )
    train_targets = tuple(av2._device_copy(split.target, device) for split in bundle.train_microbatches)
    heldout_targets = tuple(
        av2._device_copy(split.target, device) for split in bundle.heldout_microbatches
    )
    train_masks = tuple(
        av2._device_copy(split.frame_valid_mask, device) for split in bundle.train_microbatches
    )
    heldout_masks = tuple(
        av2._device_copy(split.frame_valid_mask, device)
        for split in bundle.heldout_microbatches
    )
    if a4.A4_CANDIDATE_ONLY_SEED != AV3_CANDIDATE_ONLY_INITIALIZATION_SEED:
        raise AV3ExecutionError("A4 candidate named seed differs")
    if a4.a3.a2r1.AV1B_SHARED_NAMED_SEED != AV3_SHARED_INITIALIZATION_SEED:
        raise AV3ExecutionError("A4 common named seed differs")
    pair = a4.build_cach_a4_pair(device=device)
    reference_names, reference_parameters = av2._named_parameters(
        pair.reference, pair.reference_common_trainable_names
    )
    candidate_names, candidate_parameters = av2._named_parameters(
        pair.candidate,
        pair.candidate_common_trainable_names | pair.candidate_action_trainable_names,
    )
    candidate_action_names, candidate_action_parameters = av2._named_parameters(
        pair.candidate, pair.candidate_action_trainable_names
    )
    if sum(value.numel() for value in reference_parameters) != AV3_REFERENCE_PARAMETER_COUNT:
        raise AV3ExecutionError("reference trainable parameter count differs")
    if sum(value.numel() for value in candidate_parameters) != AV3_CANDIDATE_PARAMETER_COUNT:
        raise AV3ExecutionError("candidate trainable parameter count differs")
    theta0_payload = {
        "candidate_only_initialization_seed": AV3_CANDIDATE_ONLY_INITIALIZATION_SEED,
        "manifests": pair.theta0_manifests,
        "screen_seed": AV3_SCREEN_SEED,
        "shared_initialization_seed": AV3_SHARED_INITIALIZATION_SEED,
    }
    # This callback is the sole theta0-manifest publication boundary.  It is
    # deliberately completed before the first invocation of either model arm.
    theta0_callback(theta0_payload)
    _require_before_deadline(deadline, stage="post_theta0_manifest_publication")

    reference_initial = av2._snapshot(reference_names, reference_parameters)
    candidate_initial = av2._snapshot(candidate_names, candidate_parameters)
    candidate_action_initial = av2._snapshot(
        candidate_action_names, candidate_action_parameters
    )
    theta0_reference = _evaluate_outputs(
        pair.reference, heldout_tasks, ("correct", "shuffle", "no_action")
    )
    theta0_candidate = _evaluate_outputs(
        pair.candidate, heldout_tasks, ("correct", "shuffle", "no_action")
    )
    theta0_metrics = compute_av3_metrics(
        {
            "reference": _predictions(theta0_reference),
            "candidate": _predictions(theta0_candidate),
        },
        heldout_targets,
        heldout_masks,
    )
    theta0_candidate_identity = all(
        torch.equal(output.video_prediction, output.common_prediction)
        for outputs in theta0_candidate.values()
        for output in outputs
    )
    theta0_common_equal = all(
        torch.equal(
            theta0_reference["correct"][index].common_prediction,
            theta0_candidate["correct"][index].common_prediction,
        )
        for index in range(2)
    )
    theta0_train_reference_loss = _mean_correct_loss(
        pair.reference, train_tasks, train_targets, train_masks
    )
    theta0_train_candidate_loss = _mean_correct_loss(
        pair.candidate, train_tasks, train_targets, train_masks
    )

    reference_training = _train_serial_arm(
        pair.reference,
        names=reference_names,
        parameters=reference_parameters,
        tasks=train_tasks,
        targets=train_targets,
        masks=train_masks,
        deadline=deadline,
        label="reference",
        progress_callback=progress_callback,
    )
    candidate_training = _train_serial_arm(
        pair.candidate,
        names=candidate_names,
        parameters=candidate_parameters,
        tasks=train_tasks,
        targets=train_targets,
        masks=train_masks,
        deadline=deadline,
        label="candidate",
        progress_callback=progress_callback,
        diagnostic_parameter_groups={"candidate_action": candidate_action_parameters},
    )
    _require_before_deadline(deadline, stage="post_training")
    final_reference = _evaluate_outputs(
        pair.reference, heldout_tasks, ("correct", "shuffle", "no_action")
    )
    final_candidate = _evaluate_outputs(
        pair.candidate,
        heldout_tasks,
        ("correct", "shuffle", "no_action", "seam_disabled"),
    )
    final_metrics = compute_av3_metrics(
        {
            "reference": _predictions(final_reference),
            "candidate": _predictions(final_candidate),
        },
        heldout_targets,
        heldout_masks,
    )
    final_reference_train_loss = _mean_correct_loss(
        pair.reference, train_tasks, train_targets, train_masks
    )
    final_candidate_train_loss = _mean_correct_loss(
        pair.candidate, train_tasks, train_targets, train_masks
    )

    if pair.candidate.action_stream is None:
        raise AV3ExecutionError("candidate action stream is absent")
    stream_diagnostics = []
    for output in final_candidate["correct"]:
        if output.raw_action is None or output.action_present_mask is None:
            raise AV3ExecutionError("candidate diagnostic action tensors are absent")
        stream_diagnostics.append(
            dict(
                a4._state_stream_diagnostics(
                    pair.candidate.action_stream,
                    output.raw_action,
                    output.final_common_hidden.detach(),
                    output.action_present_mask,
                )
            )
        )
    reference_modes_equal = bool(final_metrics["reference_modes_bitwise_equal"])
    no_action_bypass = all(
        torch.equal(output.video_prediction, output.common_prediction)
        and not bool(output.action_delta.count_nonzero().item())
        and output.reducer_calls_this_forward == 0
        and output.action_stream_calls_this_forward == 0
        for output in final_candidate["no_action"]
    )
    seam_disabled_bypass = all(
        torch.equal(output.video_prediction, output.common_prediction)
        and not bool(output.action_delta.count_nonzero().item())
        and output.reducer_calls_this_forward == 0
        and output.action_stream_calls_this_forward == 0
        for output in final_candidate["seam_disabled"]
    )
    target_invariant = all(
        av2._target_sentinel_perturbation_bitwise_invariant(arm, task, output)
        for arm, outputs in (
            (pair.reference, final_reference["correct"]),
            (pair.candidate, final_candidate["correct"]),
        )
        for task, output in zip(heldout_tasks, outputs, strict=True)
    )
    target_digests = {
        split.tensor_sha256["target"]
        for split in (*bundle.train_microbatches, *bundle.heldout_microbatches)
    }
    forward_digests = {
        digest
        for task in (*train_tasks, *heldout_tasks)
        for digest in task.source_tensor_digests.values()
    }
    decision_metrics = final_metrics["decision_metrics"]
    candidate_action_gradient = float(
        candidate_training["step0_gradient_rms_by_scope"]["candidate_action"]  # type: ignore[index]
    )
    reference_update = av2._update_rms(
        reference_names, reference_parameters, reference_initial
    )
    candidate_update = av2._update_rms(
        candidate_names, candidate_parameters, candidate_initial
    )
    candidate_action_update = av2._update_rms(
        candidate_action_names, candidate_action_parameters, candidate_action_initial
    )
    validity = {
        "action_and_state_jvps_positive": all(
            float(report[name]) > 0.0
            for report in stream_diagnostics
            for name in (
                "action_jvp_rms",
                "common_state_to_output_jvp_rms",
                "delta_state_to_future_output_jvp_rms",
                "past_action_to_future_output_jvp_rms",
            )
        ),
        "action_seam_gradient_positive": candidate_action_gradient > 0.0,
        "action_seam_updated": candidate_action_update > 0.0,
        "both_arms_exact_steps": (
            reference_training["steps_completed"] == AV3_STEPS_PER_ARM
            and candidate_training["steps_completed"] == AV3_STEPS_PER_ARM
        ),
        "candidate_identity_at_theta0": theta0_candidate_identity,
        "candidate_optimizer_updated": candidate_update > 0.0,
        "candidate_train_loss_declined": (
            final_candidate_train_loss < theta0_train_candidate_loss
        ),
        "data_adequate": bool(adequacy["all_adequate"]),
        "deadline_respected": time.monotonic() < deadline,
        "finite_metrics": all(
            math.isfinite(float(value)) for value in decision_metrics.values()  # type: ignore[union-attr]
        ),
        "future_to_prefix_exact_zero": all(
            float(report["future_to_prefix_max_abs"]) == 0.0
            for report in stream_diagnostics
        ),
        "no_action_bypass_exact": no_action_bypass,
        "oddness_exact_zero": all(
            float(report["oddness_max_abs"]) == 0.0 for report in stream_diagnostics
        ),
        "predecessor_pins_match": source_data_and_predecessor_pins_match,
        "reference_modes_bitwise_equal": reference_modes_equal,
        "reference_optimizer_updated": reference_update > 0.0,
        "reference_train_loss_declined": (
            final_reference_train_loss < theta0_train_reference_loss
        ),
        "seam_disabled_bypass_exact": seam_disabled_bypass,
        "target_digest_absent_from_forward": target_digests.isdisjoint(forward_digests),
        "target_sentinel_invariant": target_invariant,
        "theta0_common_equal": theta0_common_equal,
        "zero_origin_exact_zero": all(
            float(report["zero_origin_max_abs"]) == 0.0 for report in stream_diagnostics
        ),
    }
    validity["run_valid"] = all(validity.values())
    verdict, reason_code, verdict_checks = classify_av3(
        all_validity=validity["run_valid"],
        metrics=decision_metrics,  # type: ignore[arg-type]
    )
    completed = time.monotonic()
    elapsed = completed - started
    rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    if (
        completed >= deadline
        or rss_bytes > int(resources["max_process_rss_bytes"])
        or peak_cuda_bytes > int(resources["max_gpu_allocated_memory_bytes"])
    ):
        raise AV3ExecutionError("AV-3 resource ceiling exceeded")
    return MappingProxyType(
        {
            "architecture_id": AV3_ARCHITECTURE_ID,
            "data": {
                "adequacy": dict(adequacy),
                "manifest_sha256": bundle.manifest_sha256,
                "selection_sha256": bundle.selection_sha256,
            },
            "diagnostics": {
                "candidate_action_step0_gradient_rms": candidate_action_gradient,
                "candidate_action_update_rms": candidate_action_update,
                "candidate_update_rms": candidate_update,
                "reference_update_rms": reference_update,
                "state_stream_by_heldout_microbatch": stream_diagnostics,
            },
            "evaluation": {
                "final": dict(final_metrics),
                "theta0_diagnostic_only": dict(theta0_metrics),
            },
            "reason_code": reason_code,
            "resources": {
                "elapsed_seconds": elapsed,
                "remaining_total_deadline_seconds": deadline - completed,
                "peak_cuda_allocated_bytes": peak_cuda_bytes,
                "process_rss_high_water_bytes": rss_bytes,
            },
            "schema": AV3_RESULT_SCHEMA,
            "theta0": theta0_payload,
            "training": {
                "candidate": dict(candidate_training),
                "reference": dict(reference_training),
            },
            "validity": validity,
            "verdict": verdict,
            "verdict_checks": dict(verdict_checks),
        }
    )


__all__ = [
    "AV3_ARCHITECTURE_ID",
    "AV3_CONFIG_SCHEMA",
    "AV3_DATA_MANIFEST_PATH",
    "AV3_DATA_MANIFEST_SCHEMA",
    "AV3_HELDOUT_MICROBATCH_EPISODES",
    "AV3RealDataBundle",
    "AV3ExecutionError",
    "AV3_RESULT_SCHEMA",
    "AV3_SCREEN_SEED",
    "AV3_STEPS_PER_ARM",
    "AV3_TRAIN_MICROBATCH_EPISODES",
    "assess_av3_data_adequacy",
    "classify_av3",
    "compute_av3_metrics",
    "load_av3_real_data",
    "run_av3_paired_reduced_scale",
    "validate_av3_config",
]
