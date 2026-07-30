from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from sana_wam.dataloader import robotwin_history_dagger_dataset as history_dagger


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jpeg(color: tuple[int, int, int]) -> np.ndarray:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="JPEG", quality=100)
    return np.frombuffer(stream.getvalue(), dtype=np.uint8)


def _write_stats(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    values = {
        "mean": np.zeros(history_dagger.ACTION_DIM, dtype=np.float32),
        "std": np.ones(history_dagger.ACTION_DIM, dtype=np.float32),
        "min": np.full(history_dagger.ACTION_DIM, -2.0, dtype=np.float32),
        "max": np.full(history_dagger.ACTION_DIM, 2.0, dtype=np.float32),
    }
    path = tmp_path / "action_stats.npy"
    np.save(path, {"eef": values, "num_timesteps": 1})
    digest = _sha256(path)
    monkeypatch.setattr(history_dagger, "BASELINE_ACTION_STATS_SHA256", digest)
    return path, digest


def _history_arrays(*, anchor_generation: int = 4, expert_steps: int = 3) -> dict:
    env_step = np.arange(113, dtype=np.int64)
    chunk_offset = np.concatenate(
        [np.tile(np.arange(28, dtype=np.int64), 4), np.asarray([0], dtype=np.int64)]
    )
    generation_index = np.concatenate(
        [np.repeat(anchor_generation - 4 + index, 28) for index in range(4)]
        + [np.asarray([anchor_generation])]
    ).astype(np.int64)
    generated = chunk_offset == 0
    applied = np.ones(113, dtype=bool)
    applied[-1] = False

    pre_state = np.zeros((113, 20), dtype=np.float32)
    policy_action = np.zeros((113, 20), dtype=np.float32)
    for step in range(113):
        pre_state[step] = 0.001 * step + np.arange(20, dtype=np.float32) * 0.002
        policy_action[step] = (
            -0.5 + 0.005 * step + np.arange(20, dtype=np.float32) * 0.001
        )

    expert_states = np.empty((expert_steps + 1, 20), dtype=np.float32)
    expert_states[0] = pre_state[-1]
    for step in range(expert_steps):
        expert_states[step + 1] = pre_state[-1] + 0.05 * (step + 1)

    expert_chunk = np.repeat(expert_states[-1:], 28, axis=0)
    expert_chunk[:expert_steps] = expert_states[1:]
    action = np.concatenate([policy_action[28:112], expert_chunk], axis=0)
    action_mask = np.zeros(112, dtype=bool)
    action_mask[84 : 84 + expert_steps] = True
    proprio = pre_state[-1:]
    boundary_states = pre_state[[28, 56, 84, 112]]
    proprio_seq = np.empty((113, 20), dtype=np.float32)
    for chunk, (start, end) in enumerate(zip((0, 28, 56, 84), (28, 56, 84, 113))):
        proprio_seq[start:end] = boundary_states[chunk]
    return {
        "env_step": env_step,
        "chunk_offset": chunk_offset,
        "generation_index": generation_index,
        "generated": generated,
        "applied": applied,
        "pre_state": pre_state,
        "policy_action": policy_action,
        "expert_states": expert_states,
        "action": action,
        "action_mask": action_mask,
        "proprio": proprio,
        "proprio_seq": proprio_seq,
    }


def _write_artifact(
    path: Path,
    *,
    expert_steps: int = 3,
    mutate=None,
) -> dict:
    arrays = _history_arrays(expert_steps=expert_steps)
    with h5py.File(path, "w") as artifact:
        history = artifact.create_group("history")
        for key in (
            "env_step",
            "generation_index",
            "chunk_offset",
            "generated",
            "applied",
            "pre_state",
            "policy_action",
        ):
            history.create_dataset(key, data=arrays[key])

        camera_group = history.create_group("camera_jpeg")
        vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))
        camera_colors = {
            "head": (250, 4, 3),
            "left": (3, 250, 4),
            "right": (4, 3, 250),
        }
        for camera, color in camera_colors.items():
            dataset = camera_group.create_dataset(camera, shape=(113,), dtype=vlen_u8)
            encoded = _jpeg(color)
            for frame in range(113):
                dataset[frame] = encoded

        expert = artifact.create_group("expert")
        expert.create_dataset("states", data=arrays["expert_states"])
        for key in ("action", "action_mask", "proprio", "proprio_seq"):
            artifact.create_dataset(key, data=arrays[key])
        if mutate is not None:
            mutate(artifact)
    return arrays


def _provenance(stats_sha: str, takeover_plan_sha: str) -> dict:
    return {
        "policy_history_len": 113,
        "num_frames": 113,
        "video_stride": 4,
        "temporal_compression": 4,
        "causal_temporal": True,
        "ar_frame_chunk_size": 2,
        "action_tokens_per_chunk": 28,
        "action_dim": 20,
        "action_mode": "eef",
        "normalize_mode": "min-max",
        "height": 384,
        "width": 320,
        "multiview": True,
        "camera_layout": ["head_camera", "left_camera", "right_camera"],
        "checkpoint_sha256": history_dagger.BASELINE_CHECKPOINT_SHA256,
        "action_stats_sha256": stats_sha,
        "deploy_config_sha256": "a" * 64,
        "deploy_overrides_sha256": "c" * 64,
        "prompt_manifest_sha256": "b" * 64,
        "takeover_plan_sha256": takeover_plan_sha,
        "cache_feedback_mode": "predicted",
        "episode_noise_mode": "paired",
        "episode_noise_base_seed": 20260715,
        "environment_seed_index": 1,
        "step_limit_source": "robotwin_upstream_default",
        "repo_step_limit_override_count": 0,
    }


def _sample_record(
    artifact: Path,
    *,
    anchor_id: str = "anchor-000",
    episode_key: str = "robotwin/adjust_bottle/demo_clean/scene-200001/episode-0",
    environment_seed: int = 200001,
) -> dict:
    prompt = "Pick up the brown bottle using the right arm"
    return {
        "anchor_id": anchor_id,
        "episode_key": episode_key,
        "environment_seed": environment_seed,
        "anchor_env_step": 112,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "label_valid": True,
        "active_arm": "right",
        "artifact": {
            "path": artifact.name,
            "size_bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        },
    }


def _write_manifest(
    path: Path,
    stats_sha: str,
    samples: list[dict],
    *,
    mutate=None,
) -> str:
    plan_episodes = []
    anchors = []
    for episode_index, sample in enumerate(samples):
        planned = {
            "anchor_id": sample["anchor_id"],
            "episode_index": episode_index,
            "environment_seed": sample["environment_seed"],
            "prompt": sample["prompt"],
            "prompt_sha256": sample["prompt_sha256"],
            "anchor_env_step": sample["anchor_env_step"],
            "generation_index": sample["anchor_env_step"] // 28,
        }
        plan_episodes.append(planned)
        anchors.append(
            {
                **planned,
                "schema_version": 1,
                "task_name": "adjust_bottle",
                "task_config": "demo_clean",
                "episode_key": sample["episode_key"],
                "status": "success",
                "label_valid": True,
                "planner_failed": False,
                "expert_failed": False,
                "exception": None,
                "expert_capture_calls": sample.get("expert_label_count", 3) + 1,
                "expert_state_count": sample.get("expert_label_count", 3) + 1,
                "expert_label_count": sample.get("expert_label_count", 3),
                "sample": sample,
            }
        )
    plan = {
        "schema_version": 1,
        "kind": "robotwin_live_takeover_plan",
        "source_kind": history_dagger.SOURCE_KIND,
        "output_dir": str(path.parent),
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "policy_history_len": 113,
        "action_tokens_per_chunk": 28,
        "video_stride": 4,
        "temporal_compression": 4,
        "ar_frame_chunk_size": 2,
        "expert_save_freq": 15,
        "cache_feedback_mode": "predicted",
        "checkpoint_sha256": history_dagger.BASELINE_CHECKPOINT_SHA256,
        "deploy_config_sha256": "a" * 64,
        "deploy_overrides_sha256": "c" * 64,
        "prompt_manifest_sha256": "b" * 64,
        "episode_noise_mode": "paired",
        "episode_noise_base_seed": 20260715,
        "environment_seed_index": 1,
        "action_stats_path": "action_stats.npy",
        "action_stats_sha256": stats_sha,
        "step_limit_source": "robotwin_upstream_default",
        "repo_step_limit_override_count": 0,
        "upstream_step_limit": 400,
        "episodes": plan_episodes,
    }
    plan_path = path.parent / "takeover_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    plan_artifact = {
        "path": str(plan_path.resolve()),
        "size_bytes": plan_path.stat().st_size,
        "sha256": _sha256(plan_path),
    }
    results_path = path.parent / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in anchors),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source_kind": history_dagger.SOURCE_KIND,
        "complete": True,
        "action_space": "physical_eef20",
        "plan_artifact": plan_artifact,
        "results_artifact": {
            "path": results_path.name,
            "size_bytes": results_path.stat().st_size,
            "sha256": _sha256(results_path),
        },
        "provenance": _provenance(stats_sha, plan_artifact["sha256"]),
        "anchor_count": len(anchors),
        "valid_sample_count": len(samples),
        "anchors": anchors,
        "samples": samples,
    }
    if mutate is not None:
        mutate(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return _sha256(path)


def _build(
    manifest: Path,
    manifest_sha: str,
    stats: Path,
    stats_sha: str,
    **kwargs,
):
    return history_dagger.RoboTwinHistoryDaggerDataset(
        source_manifest_path=manifest,
        source_manifest_sha256=manifest_sha,
        action_stats_path=stats,
        action_stats_sha256=stats_sha,
        **kwargs,
    )


def _valid_source(tmp_path: Path, monkeypatch, *, expert_steps: int = 3, mutate=None):
    stats, stats_sha = _write_stats(tmp_path, monkeypatch)
    artifact = tmp_path / "anchor-000.hdf5"
    arrays = _write_artifact(artifact, expert_steps=expert_steps, mutate=mutate)
    manifest = tmp_path / "dataset_manifest.json"
    manifest_sha = _write_manifest(manifest, stats_sha, [_sample_record(artifact)])
    return stats, stats_sha, artifact, arrays, manifest, manifest_sha


def test_valid_anchor_builds_exact_history_conditioned_sample(tmp_path, monkeypatch):
    stats, stats_sha, _, arrays, manifest, manifest_sha = _valid_source(
        tmp_path, monkeypatch, expert_steps=3
    )
    dataset = _build(manifest, manifest_sha, stats, stats_sha)

    assert len(dataset) == 1
    sample = dataset[0]
    assert len(sample["video"]) == 29
    assert all(frame.size == (320, 384) for frame in sample["video"])
    assert sample["action"].shape == (112, 20)
    assert sample["action_mask"].shape == (112,)
    assert sample["proprio"].shape == (1, 20)
    assert sample["proprio_seq"].shape == (113, 20)
    assert sample["video_mask"].shape == (29,)
    assert bool(sample["video_mask"].all())
    assert not bool(sample["action_mask"][:84].any())
    assert bool(sample["action_mask"][84:87].all())
    assert not bool(sample["action_mask"][87:].any())
    assert sample["num_clean_prefix_actions"] == 84

    expected_action = torch.from_numpy(arrays["action"] * 0.5)
    expected_proprio = torch.from_numpy(arrays["proprio"] * 0.5)
    expected_proprio_seq = torch.from_numpy(arrays["proprio_seq"] * 0.5)
    torch.testing.assert_close(sample["action"], expected_action)
    torch.testing.assert_close(sample["proprio"], expected_proprio)
    torch.testing.assert_close(sample["proprio_seq"], expected_proprio_seq)
    for output_index, raw_index in enumerate((0, 28, 56, 84)):
        torch.testing.assert_close(
            sample["proprio_seq"][raw_index],
            torch.from_numpy(
                arrays["pre_state"][(28, 56, 84, 112)[output_index]] * 0.5
            ),
        )

    # Client JPEGs are Pillow-encoded RGB. A cv2-style decode would swap this red
    # head-camera region to blue and fail the channel-order assertion.
    head_pixel = np.asarray(sample["video"][0])[20, 20]
    assert int(head_pixel[0]) > 220
    assert int(head_pixel[2]) < 30
    assert sample["prompt"] == (
        "A video recorded from a robot's point of view executing the following "
        "instruction: Pick up the brown bottle using the right arm"
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda manifest: manifest["provenance"].update({"policy_history_len": 10}),
            "policy_history_len",
        ),
        (
            lambda manifest: manifest["provenance"].update(
                {"repo_step_limit_override_count": 1}
            ),
            "repo_step_limit_override_count",
        ),
        (
            lambda manifest: manifest["samples"][0].update(
                {"environment_seed": 100001}
            ),
            "200xxx",
        ),
    ],
)
def test_manifest_contract_is_fail_closed(tmp_path, monkeypatch, mutate, match):
    stats, stats_sha = _write_stats(tmp_path, monkeypatch)
    artifact = tmp_path / "anchor-000.hdf5"
    _write_artifact(artifact)
    manifest = tmp_path / "dataset_manifest.json"
    manifest_sha = _write_manifest(
        manifest, stats_sha, [_sample_record(artifact)], mutate=mutate
    )

    with pytest.raises(ValueError, match=match):
        _build(manifest, manifest_sha, stats, stats_sha)


def test_manifest_sha_mismatch_is_rejected(tmp_path, monkeypatch):
    stats, stats_sha, _, _, manifest, _ = _valid_source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        _build(manifest, "0" * 64, stats, stats_sha)


def test_missing_manifest_path_is_rejected(tmp_path, monkeypatch):
    stats, stats_sha = _write_stats(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="source_manifest_path"):
        history_dagger.RoboTwinHistoryDaggerDataset(
            source_manifest_path=None,
            source_manifest_sha256="0" * 64,
            action_stats_path=stats,
            action_stats_sha256=stats_sha,
        )
    with pytest.raises(FileNotFoundError, match="source manifest"):
        _build(tmp_path / "missing.json", "0" * 64, stats, stats_sha)


def test_artifact_sha_mismatch_is_rejected(tmp_path, monkeypatch):
    stats, stats_sha = _write_stats(tmp_path, monkeypatch)
    artifact = tmp_path / "anchor-000.hdf5"
    _write_artifact(artifact)
    sample = _sample_record(artifact)
    sample["artifact"]["sha256"] = "0" * 64
    manifest = tmp_path / "dataset_manifest.json"
    manifest_sha = _write_manifest(manifest, stats_sha, [sample])
    with pytest.raises(ValueError, match="artifact SHA256 mismatch"):
        _build(manifest, manifest_sha, stats, stats_sha)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda artifact: artifact["history/generated"].__setitem__(-1, False),
            "generated flags",
        ),
        (
            lambda artifact: artifact["history/env_step"].__setitem__(-1, 1114),
            "env_step has a gap",
        ),
        (
            lambda artifact: artifact["history/chunk_offset"].__setitem__(-2, 26),
            "phase-locked",
        ),
        (
            lambda artifact: artifact["history/camera_jpeg"].__delitem__("right"),
            "camera_jpeg/right",
        ),
        (
            lambda artifact: artifact["expert/states"].__setitem__(
                0, np.ones(20, dtype=np.float32)
            ),
            "expert state 0",
        ),
        (
            lambda artifact: artifact["action"].__setitem__(
                (0, 0), artifact["action"][0, 0] + 0.25
            ),
            "root action does not match",
        ),
    ],
)
def test_artifact_contract_is_fail_closed(tmp_path, monkeypatch, mutate, match):
    stats, stats_sha, _, _, manifest, manifest_sha = _valid_source(
        tmp_path, monkeypatch, mutate=mutate
    )
    with pytest.raises(ValueError, match=match):
        _build(manifest, manifest_sha, stats, stats_sha)


def test_history_array_with_ten_rows_is_rejected(tmp_path, monkeypatch):
    def truncate_env_step(artifact):
        del artifact["history/env_step"]
        artifact["history"].create_dataset(
            "env_step", data=np.arange(10, dtype=np.int64)
        )

    stats, stats_sha, _, _, manifest, manifest_sha = _valid_source(
        tmp_path, monkeypatch, mutate=truncate_env_step
    )
    with pytest.raises(ValueError, match=r"history/env_step must have shape \(113,\)"):
        _build(manifest, manifest_sha, stats, stats_sha)


def test_duplicate_anchor_is_rejected(tmp_path, monkeypatch):
    stats, stats_sha = _write_stats(tmp_path, monkeypatch)
    first = tmp_path / "first.hdf5"
    second = tmp_path / "second.hdf5"
    _write_artifact(first)
    _write_artifact(second)
    samples = [
        _sample_record(first, anchor_id="same"),
        _sample_record(second, anchor_id="same", environment_seed=200002),
    ]
    manifest = tmp_path / "dataset_manifest.json"
    manifest_sha = _write_manifest(manifest, stats_sha, samples)

    with pytest.raises(ValueError, match="duplicate anchor_id"):
        _build(manifest, manifest_sha, stats, stats_sha)


def test_failed_anchor_cannot_disappear_from_complete_ledger(tmp_path, monkeypatch):
    stats, stats_sha = _write_stats(tmp_path, monkeypatch)
    first = tmp_path / "first.hdf5"
    second = tmp_path / "second.hdf5"
    _write_artifact(first)
    _write_artifact(second)
    samples = [
        _sample_record(first, anchor_id="success", environment_seed=200001),
        _sample_record(second, anchor_id="failure", environment_seed=200002),
    ]
    manifest_path = tmp_path / "dataset_manifest.json"
    _write_manifest(manifest_path, stats_sha, samples)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failed = manifest["anchors"][1]
    failed.update(
        {
            "status": "expert_failed",
            "label_valid": False,
            "expert_failed": True,
            "sample": None,
        }
    )
    manifest["samples"] = manifest["samples"][:1]
    manifest["valid_sample_count"] = 1
    results_path = tmp_path / manifest["results_artifact"]["path"]
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest["anchors"]),
        encoding="utf-8",
    )
    manifest["results_artifact"].update(
        {
            "size_bytes": results_path.stat().st_size,
            "sha256": _sha256(results_path),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    dataset = _build(
        manifest_path, _sha256(manifest_path), stats, stats_sha, val_ratio=0.0
    )
    assert len(dataset) == 1

    manifest["anchors"] = manifest["anchors"][:1]
    manifest["anchor_count"] = 1
    results_path.write_text(
        json.dumps(manifest["anchors"][0], sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["results_artifact"].update(
        {
            "size_bytes": results_path.stat().st_size,
            "sha256": _sha256(results_path),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="anchor_count|exactly one result"):
        _build(manifest_path, _sha256(manifest_path), stats, stats_sha)


def test_generation_metadata_must_be_phase_locked_to_environment_clock(
    tmp_path, monkeypatch
):
    stats, stats_sha = _write_stats(tmp_path, monkeypatch)
    artifact = tmp_path / "anchor-000.hdf5"
    _write_artifact(artifact)
    with h5py.File(artifact, "r+") as handle:
        handle["history/env_step"][:] += 1
    sample = _sample_record(artifact)
    sample["anchor_env_step"] = 113
    sample["artifact"].update(
        {"size_bytes": artifact.stat().st_size, "sha256": _sha256(artifact)}
    )
    manifest = tmp_path / "dataset_manifest.json"
    manifest_sha = _write_manifest(manifest, stats_sha, [sample])

    with pytest.raises(ValueError, match="phase-locked"):
        _build(manifest, manifest_sha, stats, stats_sha)


def test_artifact_change_after_dataset_initialization_is_rejected(
    tmp_path, monkeypatch
):
    stats, stats_sha, artifact, _, manifest, manifest_sha = _valid_source(
        tmp_path, monkeypatch
    )
    dataset = _build(manifest, manifest_sha, stats, stats_sha)

    with h5py.File(artifact, "r+") as handle:
        handle["action"][0, 0] += 0.125

    with pytest.raises(RuntimeError, match="immutable anchor artifact changed"):
        dataset[0]


def test_episode_group_split_never_leaks_anchors(tmp_path, monkeypatch):
    stats, stats_sha = _write_stats(tmp_path, monkeypatch)
    samples = []
    for episode in range(2):
        for anchor in range(2):
            artifact = tmp_path / f"episode{episode}-anchor{anchor}.hdf5"
            _write_artifact(artifact)
            samples.append(
                _sample_record(
                    artifact,
                    anchor_id=f"ep{episode}-a{anchor}",
                    episode_key=f"robotwin/adjust_bottle/scene-20000{episode + 1}/episode-{episode}",
                    environment_seed=200001 + episode * 2 + anchor,
                )
            )
    manifest = tmp_path / "dataset_manifest.json"
    manifest_sha = _write_manifest(manifest, stats_sha, samples)

    train = _build(
        manifest,
        manifest_sha,
        stats,
        stats_sha,
        split="train",
        val_ratio=0.5,
        seed=17,
    )
    val = _build(
        manifest,
        manifest_sha,
        stats,
        stats_sha,
        split="val",
        val_ratio=0.5,
        seed=17,
    )
    assert train.episode_keys.isdisjoint(val.episode_keys)
    assert len(train) == len(val) == 2
    assert {train[index]["episode_key"] for index in range(len(train))} == set(
        train.episode_keys
    )
    assert {val[index]["episode_key"] for index in range(len(val))} == set(
        val.episode_keys
    )


def test_config_rejects_static_filter_and_repeat(tmp_path, monkeypatch):
    stats, stats_sha, _, _, manifest, manifest_sha = _valid_source(
        tmp_path, monkeypatch
    )
    with pytest.raises(ValueError, match="filter_static_segments"):
        _build(
            manifest,
            manifest_sha,
            stats,
            stats_sha,
            filter_static_segments=True,
        )
    with pytest.raises(ValueError, match="repeat"):
        _build(manifest, manifest_sha, stats, stats_sha, repeat=2)
