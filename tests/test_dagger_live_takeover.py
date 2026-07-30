from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from benchmarks.robotwin import dagger_live_takeover as live_takeover
from benchmarks.robotwin.dagger_live_takeover import (
    ACTION_TOKENS_PER_CHUNK,
    HISTORY_LEN,
    LiveTakeoverPlanError,
    LiveTakeoverRecorder,
)
from sana_wam.dataloader import robotwin_history_dagger_dataset as history_dagger


CHECKPOINT_SHA = "a" * 64
DEPLOY_CONFIG_SHA = "b" * 64
DEPLOY_OVERRIDES_SHA = "c" * 64
PROMPT = "lift the bottle upright with the correct arm"


def _jpeg() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), (20, 80, 140)).save(stream, format="JPEG", quality=100)
    return stream.getvalue()


JPEG = _jpeg()


@pytest.fixture(autouse=True)
def _test_baseline_hashes(monkeypatch):
    monkeypatch.setattr(live_takeover, "BASELINE_CHECKPOINT_SHA256", CHECKPOINT_SHA)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _server_info(**identity_overrides):
    identity = {
        "history_len": 113,
        "num_frames": 113,
        "video_num_frames": 29,
        "video_stride": 4,
        "action_tokens_per_chunk": 28,
        "ar_frame_chunk_size": 2,
        "action_mode": "eef",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "deploy_config_sha256": DEPLOY_CONFIG_SHA,
        "deploy_overrides_sha256": DEPLOY_OVERRIDES_SHA,
        "cache_feedback_mode": "predicted",
        "episode_noise_mode": "paired",
        "episode_noise_base_seed": 20260715,
    }
    identity.update(identity_overrides)
    return {
        "policy_config": {"execute_horizon": None},
        "async_inference": {"enabled": False},
        "inference_runtime": {
            "cache_feedback_mode": "predicted",
            "episode_noise_mode": "paired",
            "episode_noise_base_seed": 20260715,
            "deployment_identity": identity,
        },
    }


def _write_plan(tmp_path: Path, monkeypatch, **changes):
    stats_path = tmp_path / "action_stats.npy"
    np.save(
        stats_path,
        {
            "eef": {
                "mean": np.zeros(20, dtype=np.float32),
                "std": np.ones(20, dtype=np.float32),
                "min": np.full(20, -200.0, dtype=np.float32),
                "max": np.full(20, 200.0, dtype=np.float32),
            },
            "num_timesteps": 1,
        },
    )
    stats_sha = _sha256(stats_path)
    plan = {
        "schema_version": 1,
        "kind": "robotwin_live_takeover_plan",
        "source_kind": "fresh_on_policy_history113_dagger",
        "output_dir": str(tmp_path / "dataset"),
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "policy_history_len": 113,
        "action_tokens_per_chunk": 28,
        "video_stride": 4,
        "temporal_compression": 4,
        "ar_frame_chunk_size": 2,
        "expert_save_freq": 15,
        "cache_feedback_mode": "predicted",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "deploy_config_sha256": DEPLOY_CONFIG_SHA,
        "deploy_overrides_sha256": DEPLOY_OVERRIDES_SHA,
        "prompt_manifest_sha256": "d" * 64,
        "episode_noise_mode": "paired",
        "episode_noise_base_seed": 20260715,
        "environment_seed_index": 1,
        "action_stats_path": str(stats_path),
        "action_stats_sha256": stats_sha,
        "step_limit_source": "robotwin_upstream_default",
        "repo_step_limit_override_count": 0,
        "upstream_step_limit": 400,
        "episodes": [
            {
                "anchor_id": "scene-200001-g004",
                "episode_index": 0,
                "environment_seed": 200001,
                "prompt": PROMPT,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "anchor_env_step": 112,
                "generation_index": 4,
                "control_success": False,
                "control_step_count": 400,
                "control_anchor_observed": True,
            }
        ],
    }
    plan.update(changes)
    monkeypatch.setattr(live_takeover, "BASELINE_ACTION_STATS_SHA256", stats_sha)
    path = tmp_path / "takeover_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, plan


def _context():
    return {
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "episode_index": 0,
        "environment_seed": 200001,
        "episode_key": "robotwin/adjust_bottle/scene-200001/episode-0",
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "prompt_manifest_sha256": "d" * 64,
        "noise_pair_key": "robotwin/adjust_bottle/demo_clean/scene-200001",
    }


def _policy_info(step: int):
    generation, offset = divmod(step, ACTION_TOKENS_PER_CHUNK)
    info = {
        "policy_step": step,
        "generation_index": generation,
        "chunk_offset": offset,
        "buffer_remaining": ACTION_TOKENS_PER_CHUNK - offset - 1,
        "generated": offset == 0,
    }
    if offset == 0:
        info["chunk_len"] = ACTION_TOKENS_PER_CHUNK
    return info


class _Pose:
    def __init__(self, position):
        self.p = np.asarray(position, dtype=np.float64)


class _Bottle:
    def __init__(self):
        self.position = np.asarray([0.0, 0.0, 0.75], dtype=np.float64)

    def get_name(self):
        return "bottle"

    def get_pose(self):
        return _Pose(self.position.copy())


class _ExpertEnv:
    task_name = "adjust_bottle"
    save_freq = 15
    step_lim = 400
    qpose_tag = 0
    left_target_pose = [-0.25, -0.12, 0.95, 0, 1, 0, 0]
    right_target_pose = [0.25, -0.12, 0.95, 0, 1, 0, 0]
    dagger_bottle_grasped = False

    def __init__(self, state, mode="success"):
        self.state = np.asarray(state, dtype=np.float32).copy()
        self.bottle = _Bottle()
        self.mode = mode
        self.plan_success = True
        self.eval_success = False
        self.take_action_cnt = 112
        self.task_success = False
        self.original_picture_calls = 0
        self.moves = []

    def _take_picture(self):
        self.original_picture_calls += 1

    def get_obs(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            }
        }

    def check_success(self):
        return self.task_success

    def grasp_actor(self, bottle, *, arm_tag, pre_grasp_dis):
        if self.mode == "no_grasp_pose":
            raise AssertionError("target_pose cannot be None for move action.")
        if self.mode == "exception":
            raise RuntimeError("expert exploded")
        return "grasp"

    def move_by_displacement(self, *, arm_tag, z, move_axis):
        return "lift"

    def place_actor(self, bottle, **kwargs):
        return "place"

    def move(self, action):
        self.moves.append(action)
        self._take_picture()
        for _ in range(12):
            self.state = self.state + np.float32(0.001)
            if action == "grasp" and self.mode != "no_closure":
                self.state[9] = np.float32(0.0)
            if action == "lift" and self.mode != "closure_no_lift":
                self.bottle.position[2] += 0.006
            self._take_picture()
        if self.mode == "planner_failed" and action == "grasp":
            self.plan_success = False
            return False
        if action == "place" and self.mode in {
            "success",
            "no_closure",
            "closure_no_lift",
        }:
            self.task_success = True
        return True


def _enabled_recorder(tmp_path, monkeypatch, **plan_changes):
    path, plan = _write_plan(tmp_path, monkeypatch, **plan_changes)
    monkeypatch.setenv("ROBOTWIN_DAGGER_PLAN", str(path))
    recorder = LiveTakeoverRecorder.from_env(_server_info())
    recorder.start_episode(_context())
    return recorder, Path(plan["output_dir"])


def _drive_to_anchor(
    recorder,
    env,
    *,
    anchor_policy_override=None,
    camera_override=None,
    end_step=HISTORY_LEN - 1,
    before_step=None,
):
    outcome = None
    for step in range(end_step + 1):
        state = np.full(20, step, dtype=np.float32)
        env.state = state.copy()
        if before_step is not None:
            before_step(step, env)
        policy = _policy_info(step)
        if step == HISTORY_LEN - 1 and anchor_policy_override:
            policy.update(anchor_policy_override)
        cameras = {"head": JPEG, "left": JPEG, "right": JPEG}
        if camera_override is not None and step == camera_override[0]:
            cameras[camera_override[1]] = camera_override[2]
        outcome = recorder.process_step(
            env,
            step,
            PROMPT,
            cameras,
            state,
            np.full(20, step + 0.5, dtype=np.float32),
            policy,
            lambda task_env: task_env.state,
        )
        if outcome is not None:
            break
    return outcome


def test_unset_plan_is_a_strict_noop(monkeypatch):
    monkeypatch.delenv("ROBOTWIN_DAGGER_PLAN", raising=False)
    recorder = LiveTakeoverRecorder.from_env(None)

    assert recorder.enabled is False
    assert (
        recorder.process_step(None, 0, "", {}, None, None, {}, lambda env: None) is None
    )
    recorder.start_episode({})
    recorder.close()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"policy_history_len": 10}, "policy_history_len must be 113"),
        ({"action_tokens_per_chunk": 27}, "action_tokens_per_chunk must be 28"),
        (
            {"cache_feedback_mode": "measured"},
            "requires cache_feedback_mode='predicted'",
        ),
        (
            {
                "episodes": [
                    {
                        "anchor_id": "bad-seed",
                        "episode_index": 0,
                        "environment_seed": 100001,
                        "prompt": PROMPT,
                        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                        "anchor_env_step": 112,
                        "generation_index": 4,
                    }
                ]
            },
            "outside disjoint training range",
        ),
    ],
)
def test_plan_contract_is_fail_closed(tmp_path, monkeypatch, changes, message):
    path, _ = _write_plan(tmp_path, monkeypatch, **changes)
    monkeypatch.setenv("ROBOTWIN_DAGGER_PLAN", str(path))

    with pytest.raises(LiveTakeoverPlanError, match=message):
        LiveTakeoverRecorder.from_env(_server_info())


@pytest.mark.parametrize(
    ("identity_change", "message"),
    [
        ({"history_len": 10}, "history_len must be 113"),
        ({"deploy_config_sha256": "d" * 64}, "config SHA-256"),
        ({"deploy_overrides_sha256": "e" * 64}, "overrides SHA-256"),
    ],
)
def test_deployment_identity_is_fail_closed(
    tmp_path, monkeypatch, identity_change, message
):
    path, _ = _write_plan(tmp_path, monkeypatch)
    monkeypatch.setenv("ROBOTWIN_DAGGER_PLAN", str(path))

    with pytest.raises(LiveTakeoverPlanError, match=message):
        LiveTakeoverRecorder.from_env(_server_info(**identity_change))


def test_environment_step_limit_override_is_rejected(tmp_path, monkeypatch):
    path, _ = _write_plan(tmp_path, monkeypatch)
    monkeypatch.setenv("ROBOTWIN_DAGGER_PLAN", str(path))
    monkeypatch.setenv("ROBOTWIN_STEP_LIMITS_PATH", str(tmp_path / "limits.yml"))

    with pytest.raises(LiveTakeoverPlanError, match="must be unset"):
        LiveTakeoverRecorder.from_env(_server_info())


def test_active_repository_step_limit_mapping_is_rejected(tmp_path, monkeypatch):
    path, _ = _write_plan(tmp_path, monkeypatch)
    monkeypatch.setenv("ROBOTWIN_DAGGER_PLAN", str(path))
    fake_module = tmp_path / "fake_benchmarks" / "dagger_live_takeover.py"
    fake_module.parent.mkdir()
    fake_module.write_text("", encoding="utf-8")
    fake_module.with_name("step_limits.yml").write_text(
        "adjust_bottle: 160\n", encoding="utf-8"
    )
    monkeypatch.setattr(live_takeover, "__file__", str(fake_module))

    with pytest.raises(LiveTakeoverPlanError, match="zero active mappings"):
        LiveTakeoverRecorder.from_env(_server_info())


def test_success_writes_exact_history_raw_sample_and_manifest(tmp_path, monkeypatch):
    recorder, output_dir = _enabled_recorder(tmp_path, monkeypatch)
    env = _ExpertEnv(np.zeros(20, dtype=np.float32))

    outcome = _drive_to_anchor(recorder, env)

    assert outcome is not None
    assert outcome.consumed is True
    assert outcome.success is True
    assert outcome.status == "success"
    assert env.eval_success is True
    assert env.original_picture_calls == 0
    assert env._take_picture.__func__ is _ExpertEnv._take_picture

    recorder.close()
    sample_path = output_dir / outcome.sample_path
    with h5py.File(sample_path, "r") as handle:
        assert handle.attrs["action_space"] == "physical_eef20"
        assert handle["action"].shape == (112, 20)
        assert handle["proprio"].shape == (1, 20)
        assert handle["proprio_seq"].shape == (113, 20)
        np.testing.assert_array_equal(
            handle["history/env_step"][:], np.arange(113, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            handle["history/generation_index"][:],
            np.repeat(np.arange(5, dtype=np.int64), [28, 28, 28, 28, 1]),
        )
        np.testing.assert_array_equal(
            handle["history/chunk_offset"][:],
            np.concatenate([np.arange(28)] * 4 + [np.array([0])]),
        )
        np.testing.assert_array_equal(
            handle["history/applied"][:],
            np.asarray([True] * 112 + [False]),
        )
        np.testing.assert_allclose(
            handle["action"][:84],
            np.stack(
                [np.full(20, step + 0.5, dtype=np.float32) for step in range(28, 112)]
            ),
        )
        np.testing.assert_array_equal(
            handle["action_mask"][:],
            np.asarray([False] * 84 + [True] * 28),
        )
        expected_proprio_seq = np.empty((113, 20), dtype=np.float32)
        for state_step, (start, end) in zip(
            (28, 56, 84, 112), ((0, 28), (28, 56), (56, 84), (84, 113))
        ):
            expected_proprio_seq[start:end] = state_step
        np.testing.assert_allclose(handle["proprio_seq"][:], expected_proprio_seq)
        np.testing.assert_allclose(handle["proprio"][:], np.full((1, 20), 112))
        assert handle["expert/states"].shape == (29, 20)
        assert handle["expert/bottle_positions"].shape == (29, 3)
        assert bool(handle.attrs["label_gripper_closed"]) is True
        assert handle.attrs["label_first_close_index"] == 0
        assert handle.attrs["label_bottle_lift_delta"] >= 0.05
        assert bool(handle.attrs["label_semantic_valid"]) is True
        assert handle["history/camera_jpeg/head"][0].tobytes() == JPEG

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["active_arm"] == "left"
    assert rows[0]["label_gripper_closed"] is True
    assert rows[0]["label_first_close_index"] == 0
    assert rows[0]["label_bottle_lift_delta"] >= 0.05
    assert rows[0]["label_semantic_valid"] is True
    assert rows[0]["sample"]["label_valid"] is True
    artifact = rows[0]["sample"]["artifact"]
    assert artifact["size_bytes"] == sample_path.stat().st_size
    assert artifact["sha256"] == _sha256(sample_path)

    manifest_bytes = (output_dir / "dataset_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["valid_sample_count"] == 1
    assert manifest["samples"][0]["artifact"] == artifact
    expected_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    assert (output_dir / "dataset_manifest.sha256").read_text().split()[
        0
    ] == expected_manifest_sha
    assert not list(output_dir.rglob("*.tmp"))

    stats_path = tmp_path / "action_stats.npy"
    stats_sha = _sha256(stats_path)
    monkeypatch.setattr(history_dagger, "BASELINE_CHECKPOINT_SHA256", CHECKPOINT_SHA)
    monkeypatch.setattr(history_dagger, "BASELINE_ACTION_STATS_SHA256", stats_sha)
    dataset = history_dagger.RoboTwinHistoryDaggerDataset(
        source_manifest_path=output_dir / "dataset_manifest.json",
        source_manifest_sha256=expected_manifest_sha,
        action_stats_path=stats_path,
        action_stats_sha256=stats_sha,
    )
    training_sample = dataset[0]
    assert len(dataset) == 1
    assert len(training_sample["video"]) == 29
    assert tuple(training_sample["action"].shape) == (112, 20)
    assert training_sample["action_mask"][:84].sum().item() == 0
    assert training_sample["action_mask"][84:].sum().item() == 28


@pytest.mark.parametrize(
    ("mode", "expected_closed", "expected_lift"),
    [
        ("no_closure", False, True),
        ("closure_no_lift", True, False),
    ],
)
def test_successful_expert_with_incomplete_labels_is_rejected(
    tmp_path, monkeypatch, mode, expected_closed, expected_lift
):
    recorder, output_dir = _enabled_recorder(tmp_path, monkeypatch)
    env = _ExpertEnv(np.zeros(20, dtype=np.float32), mode=mode)

    outcome = _drive_to_anchor(recorder, env)
    recorder.close()

    assert outcome.status == "label_incomplete"
    assert outcome.success is False
    assert env.task_success is True
    assert env.eval_success is False
    assert env.take_action_cnt == env.step_lim
    row = json.loads((output_dir / "results.jsonl").read_text().strip())
    assert row["status"] == "label_incomplete"
    assert row["label_valid"] is False
    assert row["active_arm"] == "left"
    assert row["label_gripper_closed"] is expected_closed
    assert (row["label_bottle_lift_delta"] >= 0.05) is expected_lift
    assert row["label_semantic_valid"] is False
    assert row["sample"] is None
    assert list((output_dir / "samples").iterdir()) == []

    manifest = json.loads(
        (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["valid_sample_count"] == 0
    assert manifest["samples"] == []
    assert manifest["anchors"][0]["status"] == "label_incomplete"


def test_full_expert_suffix_writes_standard_complete_episode(tmp_path, monkeypatch):
    recorder, output_dir = _enabled_recorder(
        tmp_path,
        monkeypatch,
        full_expert_suffix={"enabled": True},
    )
    env = _ExpertEnv(np.zeros(20, dtype=np.float32))

    outcome = _drive_to_anchor(recorder, env)
    recorder.close()

    assert outcome.status == "success"
    manifest = json.loads(
        (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["full_expert_suffix_count"] == 1
    suffix = manifest["anchors"][0]["full_expert_suffix"]
    assert suffix["learner_frame_count"] == 113
    assert suffix["expert_frame_count"] == 39
    assert suffix["frame_count"] == 39

    suffix_root = output_dir / "full_suffix_dataset"
    episode_path = suffix_root / suffix["artifact"]["path"]
    with h5py.File(episode_path, "r") as handle:
        assert handle.attrs["source_kind"] == "full_expert_suffix"
        assert handle["observation/head_camera/rgb"].shape == (39,)
        assert handle["observation/left_camera/rgb"].shape == (39,)
        assert handle["observation/right_camera/rgb"].shape == (39,)
        assert handle["endpose/left_endpose"].shape == (39, 7)
        assert handle["endpose/right_endpose"].shape == (39, 7)
        assert handle["endpose/left_gripper"].shape == (39,)
        assert handle["endpose/right_gripper"].shape == (39,)

    suffix_manifest = json.loads(
        (suffix_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert suffix_manifest["complete"] is True
    assert suffix_manifest["sample_count"] == 1
    assert suffix_manifest["samples"] == [suffix]


def test_proximity_trigger_consumes_near_boundary_and_records_actual_anchor(
    tmp_path, monkeypatch
):
    trigger = {
        "kind": "active_eef_bottle_proximity",
        "min_env_step": 112,
        "max_env_step": 112,
        "distance_m_at_most": 0.12,
    }
    recorder, output_dir = _enabled_recorder(
        tmp_path, monkeypatch, anchor_trigger=trigger
    )
    env = _ExpertEnv(np.zeros(20, dtype=np.float32))
    env.bottle.position = np.full(3, 112.0, dtype=np.float64)

    outcome = _drive_to_anchor(recorder, env)
    recorder.close()

    assert outcome.status == "success"
    row = json.loads((output_dir / "results.jsonl").read_text().strip())
    assert row["anchor_mode"] == "active_eef_bottle_proximity"
    assert row["planned_anchor_env_step"] == 112
    assert row["anchor_env_step"] == 112
    assert row["anchor_triggered"] is True
    assert row["anchor_trigger_candidate_count"] == 1
    assert row["anchor_trigger_distance_m"] == pytest.approx(0.0)
    assert row["anchor_trigger_min_distance_m"] == pytest.approx(0.0)
    assert row["active_arm"] == "left"
    assert row["label_semantic_valid"] is True

    sample_path = output_dir / outcome.sample_path
    with h5py.File(sample_path, "r") as handle:
        assert handle.attrs["anchor_mode"] == "active_eef_bottle_proximity"
        assert handle.attrs["planned_anchor_env_step"] == 112
        assert handle.attrs["anchor_trigger_distance_m"] == pytest.approx(0.0)


def test_proximity_trigger_waits_for_a_later_eligible_boundary(tmp_path, monkeypatch):
    trigger = {
        "kind": "active_eef_bottle_proximity",
        "min_env_step": 112,
        "max_env_step": 140,
        "distance_m_at_most": 0.12,
    }
    episode = {
        "anchor_id": "scene-200001-proximity-deadline-140",
        "episode_index": 0,
        "environment_seed": 200001,
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "anchor_env_step": 140,
        "generation_index": 5,
        "control_success": False,
        "control_step_count": 400,
        "control_anchor_observed": True,
    }
    recorder, output_dir = _enabled_recorder(
        tmp_path,
        monkeypatch,
        anchor_trigger=trigger,
        episodes=[episode],
    )
    env = _ExpertEnv(np.zeros(20, dtype=np.float32))

    def move_bottle_near_at_second_boundary(step, task_env):
        if step == 140:
            task_env.bottle.position = np.full(3, 140.0, dtype=np.float64)

    outcome = _drive_to_anchor(
        recorder,
        env,
        end_step=140,
        before_step=move_bottle_near_at_second_boundary,
    )
    recorder.close()

    assert outcome.status == "success"
    row = json.loads((output_dir / "results.jsonl").read_text().strip())
    assert row["planned_anchor_env_step"] == 140
    assert row["anchor_env_step"] == 140
    assert row["generation_index"] == 5
    assert row["anchor_trigger_candidate_count"] == 2
    assert row["anchor_trigger_distance_m"] == pytest.approx(0.0)


def test_proximity_trigger_deadline_without_contact_writes_no_sample(
    tmp_path, monkeypatch
):
    trigger = {
        "kind": "active_eef_bottle_proximity",
        "min_env_step": 112,
        "max_env_step": 112,
        "distance_m_at_most": 0.12,
    }
    recorder, output_dir = _enabled_recorder(
        tmp_path, monkeypatch, anchor_trigger=trigger
    )
    env = _ExpertEnv(np.zeros(20, dtype=np.float32))

    outcome = _drive_to_anchor(recorder, env)
    recorder.close()

    assert outcome.status == "anchor_not_triggered"
    assert outcome.success is False
    assert env.moves == []
    row = json.loads((output_dir / "results.jsonl").read_text().strip())
    assert row["anchor_env_step"] is None
    assert row["planned_anchor_env_step"] == 112
    assert row["anchor_triggered"] is False
    assert row["anchor_trigger_candidate_count"] == 1
    assert row["anchor_trigger_min_distance_m"] > 0.12
    assert row["label_valid"] is False
    assert row["sample"] is None

    manifest = json.loads(
        (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["valid_sample_count"] == 0
    assert manifest["samples"] == []
    assert manifest["provenance"]["anchor_trigger"] == trigger


def test_full_expert_suffix_plan_contract_is_fail_closed(tmp_path, monkeypatch):
    path, _ = _write_plan(
        tmp_path,
        monkeypatch,
        full_expert_suffix={"enabled": False},
    )
    monkeypatch.setenv("ROBOTWIN_DAGGER_PLAN", str(path))

    with pytest.raises(
        LiveTakeoverPlanError, match="full_expert_suffix.enabled must be true"
    ):
        LiveTakeoverRecorder.from_env(_server_info())


@pytest.mark.parametrize(
    ("trigger", "message"),
    [
        (
            {
                "kind": "wrong",
                "min_env_step": 112,
                "max_env_step": 112,
                "distance_m_at_most": 0.12,
            },
            "anchor_trigger.kind",
        ),
        (
            {
                "kind": "active_eef_bottle_proximity",
                "min_env_step": 113,
                "max_env_step": 140,
                "distance_m_at_most": 0.12,
            },
            "ordered generation boundaries",
        ),
        (
            {
                "kind": "active_eef_bottle_proximity",
                "min_env_step": 112,
                "max_env_step": 112,
                "distance_m_at_most": 0.0,
            },
            "finite in",
        ),
    ],
)
def test_proximity_trigger_plan_contract_is_fail_closed(
    tmp_path, monkeypatch, trigger, message
):
    path, _ = _write_plan(tmp_path, monkeypatch, anchor_trigger=trigger)
    monkeypatch.setenv("ROBOTWIN_DAGGER_PLAN", str(path))

    with pytest.raises(LiveTakeoverPlanError, match=message):
        LiveTakeoverRecorder.from_env(_server_info())


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    [
        ("planner_failed", "planner_failed"),
        ("no_grasp_pose", "planner_failed"),
        ("expert_failed", "expert_failed"),
        ("exception", "exception"),
    ],
)
def test_every_failed_anchor_gets_one_result(
    tmp_path, monkeypatch, mode, expected_status
):
    recorder, output_dir = _enabled_recorder(tmp_path, monkeypatch)
    env = _ExpertEnv(np.zeros(20, dtype=np.float32), mode=mode)

    outcome = _drive_to_anchor(recorder, env)
    recorder.close()

    assert outcome.status == expected_status
    assert outcome.success is False
    assert env.take_action_cnt == env.step_lim
    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["status"] == expected_status
    assert rows[0]["label_valid"] is False
    assert rows[0]["active_arm"] == "left"
    assert rows[0]["control_success"] is False
    assert rows[0]["control_step_count"] == 400
    assert rows[0]["control_anchor_observed"] is True
    assert rows[0]["sample"] is None
    assert list((output_dir / "samples").iterdir()) == []


def test_non_boundary_anchor_is_consumed_as_exception(tmp_path, monkeypatch):
    recorder, output_dir = _enabled_recorder(tmp_path, monkeypatch)
    env = _ExpertEnv(np.zeros(20, dtype=np.float32))

    outcome = _drive_to_anchor(
        recorder,
        env,
        anchor_policy_override={"generated": False},
    )
    recorder.close()

    assert outcome.status == "exception"
    assert outcome.consumed is True
    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert "inconsistent with chunk_offset=0" in rows[0]["exception"]


def test_missing_camera_fails_before_anchor_without_silent_drop(tmp_path, monkeypatch):
    recorder, output_dir = _enabled_recorder(tmp_path, monkeypatch)
    env = _ExpertEnv(np.zeros(20, dtype=np.float32))

    outcome = _drive_to_anchor(
        recorder,
        env,
        camera_override=(7, "right", None),
    )
    recorder.close()

    assert outcome.status == "exception"
    assert outcome.consumed is True
    row = json.loads((output_dir / "results.jsonl").read_text().strip())
    assert row["observed_env_step"] == 7
    assert "camera_jpegs.right must be JPEG bytes" in row["exception"]


def test_runtime_task_step_limit_must_match_upstream(tmp_path, monkeypatch):
    recorder, output_dir = _enabled_recorder(tmp_path, monkeypatch)
    env = _ExpertEnv(np.zeros(20, dtype=np.float32))
    env.step_lim = 160

    outcome = _drive_to_anchor(recorder, env)
    recorder.close()

    assert outcome.status == "exception"
    row = json.loads((output_dir / "results.jsonl").read_text().strip())
    assert "pinned RoboTwin upstream limit" in row["exception"]


def test_close_records_unobserved_anchor_exactly_once(tmp_path, monkeypatch):
    recorder, output_dir = _enabled_recorder(tmp_path, monkeypatch)

    recorder.close()
    recorder.close()

    rows = (output_dir / "results.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["exception"] == "anchor_not_observed_before_close"
