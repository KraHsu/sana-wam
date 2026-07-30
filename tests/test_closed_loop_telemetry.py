from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest
from omegaconf import OmegaConf

from benchmarks.robotwin.closed_loop_telemetry import ClientTelemetryRecorder
from benchmarks.robotwin.eval_policy_wrapper import _load_prompt_manifest
from scripts import build_robotwin_prompt_manifest as manifest_builder
from scripts import verify_paired_noise_telemetry as pair_verifier
from scripts.build_robotwin_prompt_manifest import build_manifest, write_manifest
from scripts.verify_paired_noise_telemetry import (
    _validate_chunk,
    load_client_run,
    load_run,
)
from sana_wam.deploy.telemetry import TelemetryRecorder


def _write_verifiable_server_run(tmp_path, *, action_mode="eef"):
    action_dim = {"eef": 20, "joint": 14}.get(action_mode, 20)
    identity = {
        "telemetry_schema_version": 1,
        "engine_class": "ARInferenceEngine",
        "architecture_class": "DualSystemARArchitecture",
        "checkpoint_path": "/checkpoint.safetensors",
        "checkpoint_size": 123,
        "checkpoint_sha256": "abc",
        "history_len": 113,
        "video_steps": 10,
        "action_steps": 10,
        "episode_noise_mode": "paired",
        "episode_noise_base_seed": 20260715,
        "cache_feedback_mode": "predicted",
        "action_tokens_per_chunk": 28,
        "action_mode": action_mode,
    }
    runtime = {
        "current_noise_pair_key": "adjust_bottle/demo_clean/100001",
        "current_model_noise_seed": 17,
        "cache_feedback_mode": "predicted",
        "cache_feedback_commits": 0,
        "cache_feedback_fallbacks": 0,
        "ar_step": 1,
        "deployment_identity": identity,
    }
    recorder = TelemetryRecorder(
        OmegaConf.create(
            {
                "telemetry": {
                    "enabled": True,
                    "output_dir": str(tmp_path),
                    "include_chunks": True,
                    "flush_every": 1,
                }
            }
        )
    )
    recorder.start_episode(
        {
            "episode_key": "attempt-1",
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "environment_seed": 100001,
        },
        runtime,
    )
    predicted = np.zeros((28, action_dim), dtype=np.float32)
    recorder.record_step(
        request_id=1,
        state=np.zeros(action_dim, dtype=np.float32),
        action=np.zeros(action_dim, dtype=np.float32),
        latency_ms=1,
        policy_info={
            "generated": True,
            "generation_index": 0,
            "action_frame_id": 1,
            "cache_feedback": {
                "status": "predicted",
                "reason": "configured",
            },
            "predicted_actions": predicted,
            "predicted_actions_normalized": predicted,
        },
        runtime_info=runtime,
    )
    recorder.end_episode(runtime, reason="shutdown")
    recorder.close()
    step = json.loads((tmp_path / "steps.jsonl").read_text().strip())
    return tmp_path / step["chunk_npz"]


def _write_client_run(tmp_path, episode_keys=("scene-1",), *, action_mode="eef"):
    if action_mode == "eef":
        server_dim, sent_dim = 20, 16
    else:
        server_dim = sent_dim = 14
    recorder = ClientTelemetryRecorder(
        enabled=True, output_dir=str(tmp_path), flush_every=1
    )
    for index, episode_key in enumerate(episode_keys):
        environment_seed = 100001 + index
        context = {
            "episode_key": episode_key,
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "environment_seed": environment_seed,
            "noise_pair_key": f"adjust_bottle/demo_clean/{environment_seed}",
        }
        recorder.start_episode(context, {"status": "ok"})
        server_action = np.zeros(server_dim, dtype=np.float32)
        recorder.record_step(
            env_step=0,
            prompt="adjust bottle",
            pre_state=np.zeros(server_dim, dtype=np.float32),
            server_action=server_action,
            policy_action=server_action,
            sent_action=np.zeros(sent_dim, dtype=np.float32),
            post_state=np.zeros(server_dim, dtype=np.float32),
            response={"step": 1, "latency_ms": 3.0},
            success=False,
        )
    recorder.close()


def _write_complete_paired_run(
    root,
    *,
    mode="predicted",
    environment_seeds=(100001,),
    prompts=None,
    noise_pair_keys=None,
    generation_count=1,
    replay_manifest=None,
):
    prompts = prompts or tuple(
        f"adjust bottle instruction {index}" for index in range(len(environment_seeds))
    )
    assert len(prompts) == len(environment_seeds)
    noise_pair_keys = noise_pair_keys or tuple(
        f"robotwin/adjust_bottle/demo_clean/scene-{environment_seed}"
        for environment_seed in environment_seeds
    )
    assert len(noise_pair_keys) == len(environment_seeds)
    server_root = root / "telemetry" / "server"
    identity = {
        "telemetry_schema_version": 1,
        "engine_class": "ARInferenceEngine",
        "architecture_class": "DualSystemARArchitecture",
        "checkpoint_path": "/checkpoint.safetensors",
        "checkpoint_size": 123,
        "checkpoint_sha256": "abc",
        "history_len": 113,
        "video_steps": 10,
        "action_steps": 10,
        "episode_noise_mode": "paired",
        "episode_noise_base_seed": 20260715,
        "cache_feedback_mode": mode,
        "action_tokens_per_chunk": 28,
        "action_mode": "eef",
    }
    generation_zero = []
    for ordinal in range(len(environment_seeds)):
        generation_zero.append(
            (
                np.full((28, 20), ordinal + 0.25, dtype=np.float32),
                np.full((28, 20), ordinal - 0.5, dtype=np.float32),
            )
        )

    if replay_manifest is None:
        replay_manifest = mode != "predicted"
    manifest_metadata = None
    if replay_manifest:
        manifest_path = (root / "prompt_manifest.json").resolve()
        manifest = {
            "schema_version": 1,
            "kind": "robotwin_prompt_replay",
            "instruction_type": "unseen",
            "source": {
                "run_path": "/control",
                "run_id": "paired_predicted_test",
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "episode_noise_base_seed": identity["episode_noise_base_seed"],
                "episode_count": len(environment_seeds),
            },
            "episodes": [],
        }
        for ordinal, (environment_seed, pair_key, prompt) in enumerate(
            zip(environment_seeds, noise_pair_keys, prompts)
        ):
            raw, normalized = generation_zero[ordinal]
            manifest["episodes"].append(
                {
                    "ordinal": ordinal,
                    "episode_index": ordinal,
                    "task_name": "adjust_bottle",
                    "task_config": "demo_clean",
                    "environment_seed": environment_seed,
                    "noise_pair_key": pair_key,
                    "model_noise_seed": 17 + ordinal,
                    "source_episode_key": f"episode-{ordinal}",
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "generation_zero": {
                        "shape": [28, 20],
                        "dtype": "float32",
                        "predicted_actions_sha256": hashlib.sha256(
                            raw.tobytes()
                        ).hexdigest(),
                        "predicted_actions_normalized_sha256": hashlib.sha256(
                            normalized.tobytes()
                        ).hexdigest(),
                    },
                }
            )
        encoded = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(encoded)
        manifest_metadata = {
            "prompt_source": "manifest",
            "prompt_manifest_path": str(manifest_path),
            "prompt_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    server = TelemetryRecorder(
        OmegaConf.create(
            {
                "telemetry": {
                    "enabled": True,
                    "output_dir": str(server_root),
                    "include_chunks": True,
                    "flush_every": 1,
                }
            }
        )
    )
    episode_steps = []
    for ordinal, (environment_seed, pair_key, prompt) in enumerate(
        zip(environment_seeds, noise_pair_keys, prompts)
    ):
        runtime = {
            "current_noise_pair_key": pair_key,
            "current_model_noise_seed": 17 + ordinal,
            "cache_feedback_mode": mode,
            "cache_feedback_commits": 0,
            "cache_feedback_fallbacks": 0,
            "ar_step": 0,
            "deployment_identity": identity,
        }
        episode_key = f"episode-{ordinal}"
        context = {
            "episode_key": episode_key,
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "environment_seed": environment_seed,
            "episode_index": ordinal,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        if manifest_metadata is not None:
            context.update(manifest_metadata)
            context["prompt_manifest_ordinal"] = ordinal
        server.start_episode(
            context,
            runtime,
        )
        recorded_steps = []
        post_states = []
        total_steps = 1 + 28 * (generation_count - 1)
        for offset in range(total_steps):
            generation = offset // 28
            generated = offset % 28 == 0
            pre_state = np.full(
                20, np.float32(ordinal + offset * 0.001), dtype=np.float32
            )
            post_state = np.full(
                20, np.float32(ordinal + (offset + 1) * 0.001), dtype=np.float32
            )
            predicted = np.full((28, 20), ordinal + generation + 0.25, dtype=np.float32)
            predicted_normalized = np.full(
                (28, 20), ordinal + generation - 0.5, dtype=np.float32
            )
            action = predicted[offset % 28]
            policy_info = {
                "generated": generated,
                "generation_index": generation,
            }
            if generated:
                action_frame_id = 1 if generation == 0 else 2 * generation + 3
                feedback = (
                    {"status": "predicted", "reason": "configured"}
                    if mode == "predicted"
                    else {"status": "skipped", "reason": "episode_start"}
                )
                if mode in {"measured", "reencode_predicted"} and generation > 0:
                    previous_frame_id = (
                        1 if generation == 1 else 2 * (generation - 1) + 3
                    )
                    if mode == "measured":
                        feedback_actions = np.stack(post_states[-28:], axis=0)
                        feedback_normalized = feedback_actions
                        feedback = {
                            "status": "measured",
                            "reason": "achieved_state_history",
                            "frame_id": previous_frame_id,
                            "actions": feedback_actions,
                            "actions_normalized": feedback_normalized,
                        }
                    else:
                        feedback_actions = np.full(
                            (28, 20), ordinal + generation - 1 + 0.25, dtype=np.float32
                        )
                        feedback_normalized = np.full(
                            (28, 20), ordinal + generation - 1 - 0.5, dtype=np.float32
                        )
                        feedback = {
                            "status": "reencoded_predicted",
                            "reason": "previous_predicted_chunk",
                            "frame_id": previous_frame_id,
                            "source_generation_index": generation - 1,
                            "actions": feedback_actions,
                            "actions_normalized": feedback_normalized,
                        }
                policy_info.update(
                    {
                        "action_frame_id": action_frame_id,
                        "cache_feedback": feedback,
                        "predicted_actions": predicted,
                        "predicted_actions_normalized": predicted_normalized,
                    }
                )
                runtime["ar_step"] = generation + 1
                runtime["cache_feedback_commits"] = (
                    generation if mode in {"measured", "reencode_predicted"} else 0
                )
            server.record_step(
                request_id=offset + 1,
                state=pre_state,
                action=action,
                latency_ms=1,
                policy_info=policy_info,
                runtime_info=runtime,
            )
            recorded_steps.append((pre_state, action.copy(), post_state))
            post_states.append(post_state)
        server.end_episode(runtime, reason="episode_boundary")
        episode_steps.append(recorded_steps)
    server.close()

    client_root = root / "telemetry" / "client"
    client = ClientTelemetryRecorder(
        enabled=True, output_dir=str(client_root), flush_every=1
    )
    for ordinal, (environment_seed, prompt, pair_key, recorded_steps) in enumerate(
        zip(environment_seeds, prompts, noise_pair_keys, episode_steps)
    ):
        context = {
            "episode_key": f"episode-{ordinal}",
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "environment_seed": environment_seed,
            "episode_index": ordinal,
            "noise_pair_key": pair_key,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        if manifest_metadata is not None:
            context.update(manifest_metadata)
            context["prompt_manifest_ordinal"] = ordinal
        client.start_episode(context, {"status": "ok"})
        for offset, (pre_state, action, post_state) in enumerate(recorded_steps):
            client.record_step(
                env_step=offset,
                prompt=prompt,
                pre_state=pre_state,
                server_action=action,
                policy_action=action,
                sent_action=np.zeros(16, dtype=np.float32),
                post_state=post_state,
                response={"step": offset + 1, "latency_ms": 1.0},
                success=False,
            )
    client.close()
    return sorted((server_root / "chunks").glob("*.npz"))


def _invoke_pair_verifier(
    monkeypatch,
    control,
    treatment,
    *,
    expected_episodes,
    include_clients=False,
    allow_control_prefix=False,
    control_mode="predicted",
    treatment_mode="measured",
    expected_manifest_sha256=None,
):
    argv = [
        "verify_paired_noise_telemetry.py",
        str(control),
        str(treatment),
        "--expected-episodes",
        str(expected_episodes),
        "--control-mode",
        control_mode,
        "--treatment-mode",
        treatment_mode,
    ]
    if include_clients:
        argv.extend(
            [
                "--control-client",
                str(control),
                "--treatment-client",
                str(treatment),
            ]
        )
    if allow_control_prefix:
        argv.append("--allow-control-prefix")
    if expected_manifest_sha256 is not None:
        argv.extend(["--expected-prompt-manifest-sha256", expected_manifest_sha256])
    monkeypatch.setattr(sys, "argv", argv)
    return pair_verifier.main()


def _rewrite_jsonl(path, transform):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        transform(record)
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records))


def _update_treatment_start_context(root, target_episode_index, **updates):
    for producer in ("server", "client"):
        path = root / "telemetry" / producer / "episodes.jsonl"

        def update(record):
            context = record.get("context", {})
            if (
                record.get("event") == "episode_start"
                and context.get("episode_index") == target_episode_index
            ):
                context.update(updates)

        _rewrite_jsonl(path, update)


def test_disabled_server_telemetry_creates_no_files(tmp_path):
    recorder = TelemetryRecorder(
        OmegaConf.create({"telemetry": {"enabled": False, "output_dir": str(tmp_path)}})
    )
    recorder.start_episode({"episode_key": "ep"}, {})
    recorder.record_step(
        request_id=1,
        state=np.zeros(20),
        action=np.zeros(20),
        latency_ms=1.0,
        policy_info={},
        runtime_info={},
    )
    recorder.close()
    assert list(tmp_path.iterdir()) == []


def test_server_telemetry_writes_step_and_chunk_npz(tmp_path):
    recorder = TelemetryRecorder(
        OmegaConf.create(
            {
                "telemetry": {
                    "enabled": True,
                    "output_dir": str(tmp_path),
                    "include_chunks": True,
                    "flush_every": 1,
                }
            }
        )
    )
    recorder.start_episode(
        {"episode_key": "adjust_bottle/100001"},
        {"current_model_noise_seed": 7},
    )
    predicted = np.arange(60, dtype=np.float32).reshape(3, 20)
    measured = predicted + 0.5
    recorder.record_step(
        request_id=1,
        state=np.zeros(20, dtype=np.float32),
        action=predicted[0],
        latency_ms=2.5,
        policy_info={
            "generated": True,
            "generation_index": 0,
            "predicted_actions": predicted,
            "predicted_actions_normalized": predicted / 10,
            "cache_feedback": {
                "status": "measured",
                "actions": measured,
                "actions_normalized": measured / 10,
            },
        },
        runtime_info={"ar_step": 1},
    )
    recorder.close()

    episode = json.loads((tmp_path / "episodes.jsonl").read_text().strip())
    step = json.loads((tmp_path / "steps.jsonl").read_text().strip())
    assert episode["episode_key"] == "adjust_bottle/100001"
    assert step["observed_gripper"] == [0.0, 0.0]
    assert step["policy"]["cache_feedback"]["status"] == "measured"
    chunk_path = tmp_path / step["chunk_npz"]
    with np.load(chunk_path) as chunk:
        np.testing.assert_array_equal(chunk["predicted_actions"], predicted)
        np.testing.assert_array_equal(chunk["measured_feedback_actions"], measured)


def test_client_telemetry_records_sent_and_post_state(tmp_path):
    recorder = ClientTelemetryRecorder(
        enabled=True, output_dir=str(tmp_path), flush_every=1
    )
    context = {"episode_key": "scene-1", "environment_seed": 100001}
    recorder.start_episode(context, {"status": "ok"})
    command = np.zeros(20, dtype=np.float32)
    command[[9, 19]] = 1
    sent = np.zeros(16, dtype=np.float32)
    sent[[7, 15]] = 1
    post = command.copy()
    post[0] = 0.1
    recorder.record_step(
        env_step=0,
        prompt="adjust bottle",
        pre_state=np.zeros(20),
        server_action=command,
        policy_action=command,
        sent_action=sent,
        post_state=post,
        response={"step": 1, "latency_ms": 3.0},
        success=False,
    )
    recorder.close()

    step = json.loads((tmp_path / "steps.jsonl").read_text().strip())
    assert step["environment_seed"] == 100001
    assert step["schema_version"] == 1
    assert step["server_action_space"] == "eef20"
    assert step["sent_action_space"] == "ee16"
    assert step["sent_action_gripper"] == [1.0, 1.0]
    assert step["post_gripper"] == [1.0, 1.0]
    assert step["returned_vs_post_error"]["left_xyz_l2"] == np.float32(0.1)
    episode_events = [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in episode_events] == [
        "episode_start",
        "episode_end",
    ]
    assert episode_events[-1]["step_count"] == 1
    assert episode_events[-1]["success"] is False


def test_pair_verifier_reads_scene_and_model_seed(tmp_path):
    _write_verifiable_server_run(tmp_path)

    run = load_run(tmp_path, expected_mode="predicted", expected_episodes=1)
    assert run["pairs"] == [
        (
            "adjust_bottle",
            "demo_clean",
            100001,
            "adjust_bottle/demo_clean/100001",
            17,
        )
    ]

    episode_path = tmp_path / "episodes.jsonl"
    episode_path.write_text(episode_path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="incomplete server episodes"):
        load_run(tmp_path, expected_mode="predicted", expected_episodes=1)


def test_pair_verifier_accepts_joint_chunk_contract(tmp_path):
    _write_verifiable_server_run(tmp_path, action_mode="joint")

    load_run(tmp_path, expected_mode="predicted", expected_episodes=1)


@pytest.mark.parametrize(
    ("field", "bad_array", "message"),
    [
        (
            "predicted_actions",
            np.zeros((27, 20), dtype=np.float32),
            "invalid predicted_actions shape",
        ),
        (
            "predicted_actions_normalized",
            np.zeros((28, 20), dtype=np.float64),
            "invalid predicted_actions_normalized dtype",
        ),
    ],
)
def test_pair_verifier_rejects_invalid_chunk_contract(
    tmp_path, field, bad_array, message
):
    chunk_path = _write_verifiable_server_run(tmp_path)
    with np.load(chunk_path, allow_pickle=False) as chunk:
        arrays = {name: chunk[name] for name in chunk.files}
    arrays[field] = bad_array
    np.savez_compressed(chunk_path, **arrays)

    with pytest.raises(ValueError, match=message):
        load_run(tmp_path, expected_mode="predicted", expected_episodes=1)


@pytest.mark.parametrize(
    ("field", "bad_array", "message"),
    [
        (
            "measured_feedback_actions",
            np.zeros((27, 20), dtype=np.float32),
            "invalid measured_feedback_actions shape",
        ),
        (
            "measured_feedback_actions_normalized",
            np.zeros((28, 20), dtype=np.float64),
            "invalid measured_feedback_actions_normalized dtype",
        ),
    ],
)
def test_pair_verifier_rejects_invalid_measured_chunk_contract(
    tmp_path, field, bad_array, message
):
    chunk_path = tmp_path / "measured.npz"
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int16),
        "predicted_actions": np.zeros((28, 20), dtype=np.float32),
        "predicted_actions_normalized": np.zeros((28, 20), dtype=np.float32),
        "measured_feedback_actions": np.zeros((28, 20), dtype=np.float32),
        "measured_feedback_actions_normalized": np.zeros((28, 20), dtype=np.float32),
    }
    arrays[field] = bad_array
    np.savez_compressed(chunk_path, **arrays)

    with pytest.raises(ValueError, match=message):
        _validate_chunk(chunk_path, measured_required=True, expected_shape=(28, 20))


def test_pair_verifier_rejects_unsupported_action_mode(tmp_path):
    _write_verifiable_server_run(tmp_path, action_mode="cartesian")

    with pytest.raises(ValueError, match="unsupported action_mode"):
        load_run(tmp_path, expected_mode="predicted", expected_episodes=1)


def test_client_verifier_accepts_joint_action_contract(tmp_path):
    _write_client_run(tmp_path, action_mode="joint")

    load_client_run(tmp_path, expected_episodes=1)


def test_client_verifier_rejects_cross_episode_step_attribution(tmp_path):
    _write_client_run(tmp_path, episode_keys=("scene-1", "scene-2"))
    steps_path = tmp_path / "steps.jsonl"
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    steps[1]["episode_key"] = steps[0]["episode_key"]
    steps[1]["environment_seed"] = steps[0]["environment_seed"]
    steps_path.write_text("".join(f"{json.dumps(step)}\n" for step in steps))

    with pytest.raises(ValueError, match="non-contiguous client env_step"):
        load_client_run(tmp_path, expected_episodes=2)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("pre_proprio", [0.0] * 19, "pre_proprio"),
        ("server_action", [0.0] * 19, "server_action"),
        ("policy_action", [0.0] * 19, "policy_action"),
        ("sent_action", [0.0] * 15, "sent_action"),
        ("post_proprio", [0.0] * 19, "post_proprio"),
        ("post_gripper", [0.0], "post_gripper"),
        ("sent_action", [float("nan")] + [0.0] * 15, "non-finite.*sent_action"),
    ],
)
def test_client_verifier_rejects_invalid_vectors(tmp_path, field, bad_value, message):
    _write_client_run(tmp_path)
    steps_path = tmp_path / "steps.jsonl"
    step = json.loads(steps_path.read_text().strip())
    step[field] = bad_value
    steps_path.write_text(json.dumps(step) + "\n")

    with pytest.raises(ValueError, match=message):
        load_client_run(tmp_path, expected_episodes=1)


def test_pair_verifier_rejects_prompt_mismatch(tmp_path, monkeypatch):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, prompts=("control prompt",))
    _write_complete_paired_run(
        treatment,
        mode="measured",
        prompts=("different treatment prompt",),
    )

    with pytest.raises(SystemExit, match="prompt mismatch at episode 0"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


def test_pair_verifier_accepts_complete_artifact_hard_gate(tmp_path, monkeypatch):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, generation_count=2)
    _write_complete_paired_run(
        treatment,
        mode="measured",
        generation_count=2,
    )

    assert (
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )
        == 0
    )


def test_pair_verifier_accepts_prompt_replayed_predicted_replicate(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    replicate = tmp_path / "replicate"
    _write_complete_paired_run(control, generation_count=2)
    _write_complete_paired_run(
        replicate,
        generation_count=2,
        replay_manifest=True,
    )

    assert (
        _invoke_pair_verifier(
            monkeypatch,
            control,
            replicate,
            expected_episodes=1,
            include_clients=True,
            treatment_mode="predicted",
        )
        == 0
    )


def test_pair_verifier_rejects_predicted_replicate_without_manifest(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    replicate = tmp_path / "replicate"
    _write_complete_paired_run(control)
    _write_complete_paired_run(replicate)

    with pytest.raises(SystemExit, match="prompt manifest is missing"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            replicate,
            expected_episodes=1,
            include_clients=True,
            treatment_mode="predicted",
        )


def test_pair_verifier_accepts_reencoded_predicted_feedback(tmp_path, monkeypatch):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, generation_count=2)
    _write_complete_paired_run(
        treatment,
        mode="reencode_predicted",
        generation_count=2,
    )

    assert (
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
            treatment_mode="reencode_predicted",
        )
        == 0
    )


@pytest.mark.parametrize(
    "field",
    [
        "reencoded_predicted_actions",
        "reencoded_predicted_actions_normalized",
    ],
)
def test_pair_verifier_rejects_reencoded_predicted_content_mismatch(
    tmp_path, monkeypatch, field
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, generation_count=2)
    chunks = _write_complete_paired_run(
        treatment,
        mode="reencode_predicted",
        generation_count=2,
    )
    chunk_path = chunks[1]
    with np.load(chunk_path, allow_pickle=False) as chunk:
        arrays = {name: chunk[name] for name in chunk.files}
    arrays[field] = arrays[field].copy()
    arrays[field][0, 0] += np.float32(1.0)
    np.savez_compressed(chunk_path, **arrays)

    with pytest.raises(SystemExit, match=f"{field}/previous .* bytes mismatch"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            treatment_mode="reencode_predicted",
        )


def test_pair_verifier_rejects_reencode_source_generation_mismatch(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, generation_count=2)
    _write_complete_paired_run(
        treatment,
        mode="reencode_predicted",
        generation_count=2,
    )
    server_steps = treatment / "telemetry" / "server" / "steps.jsonl"

    def change_source_generation(record):
        policy = record.get("policy", {})
        if policy.get("generated") and policy.get("generation_index") == 1:
            policy["cache_feedback"]["source_generation_index"] = 1

    _rewrite_jsonl(server_steps, change_source_generation)

    with pytest.raises(ValueError, match="source_generation_index"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            treatment_mode="reencode_predicted",
        )


def test_pair_verifier_rejects_terminal_steady_action_frame_mismatch(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, generation_count=2)
    _write_complete_paired_run(
        treatment,
        mode="reencode_predicted",
        generation_count=2,
    )
    server_steps = treatment / "telemetry" / "server" / "steps.jsonl"

    def change_terminal_action_frame(record):
        policy = record.get("policy", {})
        if policy.get("generated") and policy.get("generation_index") == 1:
            policy["action_frame_id"] = 7

    _rewrite_jsonl(server_steps, change_terminal_action_frame)

    with pytest.raises(ValueError, match="action_frame_id"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            treatment_mode="reencode_predicted",
        )


def test_pair_verifier_requires_clients_for_artifact_hard_gate(tmp_path, monkeypatch):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control)
    _write_complete_paired_run(treatment, mode="measured")

    with pytest.raises(SystemExit, match="client artifacts are required"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
        )


def test_pair_verifier_rejects_non_manifest_treatment_prompt_source(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control)
    _write_complete_paired_run(treatment, mode="measured")
    _update_treatment_start_context(treatment, 0, prompt_source="robotwin")

    with pytest.raises(SystemExit, match="prompt_source is not manifest"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


def test_pair_verifier_rejects_treatment_manifest_ordinal_mismatch(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control)
    _write_complete_paired_run(treatment, mode="measured")
    _update_treatment_start_context(
        treatment,
        0,
        episode_index=1,
        prompt_manifest_ordinal=1,
    )

    with pytest.raises(SystemExit, match="manifest ordinal/episode_index mismatch"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("prompt_manifest_path", "/different/prompt_manifest.json"),
        ("prompt_manifest_sha256", "0" * 64),
    ],
)
def test_pair_verifier_rejects_mixed_treatment_manifest_provenance(
    tmp_path, monkeypatch, field, bad_value
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    seeds = (100001, 100002)
    _write_complete_paired_run(control, environment_seeds=seeds)
    _write_complete_paired_run(
        treatment,
        mode="measured",
        environment_seeds=seeds,
    )
    _update_treatment_start_context(treatment, 1, **{field: bad_value})

    with pytest.raises(SystemExit, match="do not share one prompt manifest path/SHA"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=2,
            include_clients=True,
        )


def test_pair_verifier_rejects_treatment_manifest_artifact_sha_mismatch(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control)
    _write_complete_paired_run(treatment, mode="measured")
    manifest_path = treatment / "prompt_manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(SystemExit, match="SHA-256 does not match run artifact"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


def test_pair_verifier_enforces_expected_manifest_sha256(tmp_path, monkeypatch):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control)
    _write_complete_paired_run(treatment, mode="measured")

    with pytest.raises(SystemExit, match="does not match expected"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
            expected_manifest_sha256="0" * 64,
        )


def test_pair_verifier_requires_same_manifest_for_two_replay_runs(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, replay_manifest=True)
    _write_complete_paired_run(treatment, mode="reencode_predicted")
    manifest_path = treatment / "prompt_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["run_id"] = "different-but-otherwise-valid-source"
    encoded = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(encoded)
    _update_treatment_start_context(
        treatment,
        0,
        prompt_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
    )

    with pytest.raises(SystemExit, match="manifest SHA-256 values do not match"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
            treatment_mode="reencode_predicted",
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("checkpoint_sha256", "bad-checkpoint", "checkpoint does not match control"),
        ("episode_noise_base_seed", 20260716, "base seed does not match control"),
    ],
)
def test_pair_verifier_rejects_manifest_source_identity_mismatch(
    tmp_path, monkeypatch, field, bad_value, message
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control)
    _write_complete_paired_run(treatment, mode="measured")
    manifest_path = treatment / "prompt_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"][field] = bad_value
    encoded = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(encoded)
    _update_treatment_start_context(
        treatment,
        0,
        prompt_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
    )

    with pytest.raises(SystemExit, match=message):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


def test_pair_verifier_rejects_server_client_request_mismatch(tmp_path, monkeypatch):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control)
    _write_complete_paired_run(treatment, mode="measured")
    client_steps = treatment / "telemetry" / "client" / "steps.jsonl"

    def change_first_request(record):
        if record.get("env_step") == 0:
            record["server_step"] = 2

    _rewrite_jsonl(client_steps, change_first_request)

    with pytest.raises(SystemExit, match="request mapping mismatch"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


def test_pair_verifier_rejects_server_observed_client_pre_bytes_mismatch(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control)
    _write_complete_paired_run(treatment, mode="measured")
    client_steps = treatment / "telemetry" / "client" / "steps.jsonl"

    def change_signed_zero(record):
        if record.get("env_step") == 0:
            record["pre_proprio"][0] = -0.0

    _rewrite_jsonl(client_steps, change_signed_zero)

    with pytest.raises(SystemExit, match="server observed/client pre mismatch"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


def test_pair_verifier_rejects_client_post_next_pre_bytes_mismatch(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, generation_count=2)
    _write_complete_paired_run(treatment, mode="measured", generation_count=2)
    client_steps = treatment / "telemetry" / "client" / "steps.jsonl"
    server_steps = treatment / "telemetry" / "server" / "steps.jsonl"

    def change_second_pre(record):
        if record.get("env_step") == 1:
            record["pre_proprio"][0] += 0.5

    def change_second_observed(record):
        if record.get("request_id") == 2:
            record["observed_proprio"][0] += 0.5

    _rewrite_jsonl(client_steps, change_second_pre)
    _rewrite_jsonl(server_steps, change_second_observed)

    with pytest.raises(SystemExit, match="client post/next pre mismatch"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


def test_pair_verifier_rejects_measured_feedback_client_post_mismatch(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, generation_count=2)
    chunks = _write_complete_paired_run(
        treatment,
        mode="measured",
        generation_count=2,
    )
    chunk_path = chunks[1]
    with np.load(chunk_path, allow_pickle=False) as chunk:
        arrays = {name: chunk[name] for name in chunk.files}
    arrays["measured_feedback_actions"] = arrays["measured_feedback_actions"].copy()
    arrays["measured_feedback_actions"][0, 0] += np.float32(1.0)
    np.savez_compressed(chunk_path, **arrays)

    with pytest.raises(SystemExit, match="measured feedback/client post mismatch"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


def test_pair_verifier_rejects_measured_feedback_frame_mismatch(tmp_path, monkeypatch):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, generation_count=2)
    _write_complete_paired_run(treatment, mode="measured", generation_count=2)
    server_steps = treatment / "telemetry" / "server" / "steps.jsonl"

    def change_feedback_frame(record):
        policy = record.get("policy", {})
        if policy.get("generated") and policy.get("generation_index") == 1:
            policy["cache_feedback"]["frame_id"] = 999

    _rewrite_jsonl(server_steps, change_feedback_frame)

    with pytest.raises(SystemExit, match="measured feedback frame_id mismatch"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


def test_pair_verifier_rejects_measured_commit_total_mismatch(tmp_path, monkeypatch):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, generation_count=2)
    _write_complete_paired_run(treatment, mode="measured", generation_count=2)
    episodes_path = treatment / "telemetry" / "server" / "episodes.jsonl"

    def change_commit_total(record):
        if record.get("event") == "episode_end":
            record["runtime"]["cache_feedback_commits"] = 0

    _rewrite_jsonl(episodes_path, change_commit_total)

    with pytest.raises(ValueError, match="commits=0, expected 1"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
        )


@pytest.mark.parametrize("field", ["predicted_actions", "predicted_actions_normalized"])
def test_pair_verifier_rejects_generation_zero_mismatch(tmp_path, monkeypatch, field):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control)
    treatment_chunks = _write_complete_paired_run(treatment, mode="measured")
    chunk_path = treatment_chunks[0]
    with np.load(chunk_path, allow_pickle=False) as chunk:
        arrays = {name: chunk[name] for name in chunk.files}
    arrays[field] = arrays[field].copy()
    arrays[field][0, 0] += np.float32(1.0)
    np.savez_compressed(chunk_path, **arrays)

    with pytest.raises(SystemExit, match=f"generation-zero {field} mismatch"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
        )


@pytest.mark.parametrize("field", ["predicted_actions", "predicted_actions_normalized"])
def test_pair_verifier_generation_zero_gate_distinguishes_signed_zero(
    tmp_path, monkeypatch, field
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    control_chunks = _write_complete_paired_run(control)
    treatment_chunks = _write_complete_paired_run(treatment, mode="measured")
    for chunk_path, value in (
        (control_chunks[0], np.float32(0.0)),
        (treatment_chunks[0], np.float32(-0.0)),
    ):
        with np.load(chunk_path, allow_pickle=False) as chunk:
            arrays = {name: chunk[name] for name in chunk.files}
        arrays[field] = arrays[field].copy()
        arrays[field][0, 0] = value
        np.savez_compressed(chunk_path, **arrays)

    with pytest.raises(SystemExit, match=f"generation-zero {field} mismatch"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
        )


def test_pair_verifier_rejects_generation_zero_observed_proprio_mismatch(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control)
    _write_complete_paired_run(treatment, mode="measured")
    steps_path = treatment / "telemetry" / "server" / "steps.jsonl"
    step = json.loads(steps_path.read_text().strip())
    step["observed_proprio"][0] = 1.0
    steps_path.write_text(json.dumps(step) + "\n")

    with pytest.raises(SystemExit, match="generation-zero observed_proprio mismatch"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
        )


def test_pair_verifier_accepts_control_prefix(tmp_path, monkeypatch):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(
        control,
        environment_seeds=(100001, 100002),
        prompts=("first prompt", "second prompt"),
    )
    _write_complete_paired_run(
        treatment,
        mode="measured",
        environment_seeds=(100001,),
        prompts=("first prompt",),
    )

    assert (
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=1,
            include_clients=True,
            allow_control_prefix=True,
        )
        == 0
    )


def test_pair_verifier_rejects_short_control_prefix(tmp_path, monkeypatch):
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_complete_paired_run(control, environment_seeds=(100001,))
    _write_complete_paired_run(
        treatment,
        mode="measured",
        environment_seeds=(100001, 100002),
    )

    with pytest.raises(ValueError, match="control has 1 episodes, need 2"):
        _invoke_pair_verifier(
            monkeypatch,
            control,
            treatment,
            expected_episodes=2,
            allow_control_prefix=True,
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("prompt", "wrong context prompt", "prompt/context mismatch"),
        ("prompt_sha256", "0" * 64, "prompt SHA-256 mismatch"),
    ],
)
def test_client_verifier_rejects_context_prompt_metadata_mismatch(
    tmp_path, field, bad_value, message
):
    _write_client_run(tmp_path)
    episodes_path = tmp_path / "episodes.jsonl"
    events = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    events[0]["context"][field] = bad_value
    episodes_path.write_text("".join(f"{json.dumps(event)}\n" for event in events))

    with pytest.raises(ValueError, match=message):
        load_client_run(tmp_path, expected_episodes=1)


def test_prompt_manifest_builder_produces_runtime_loadable_manifest(tmp_path):
    source = tmp_path / "control"
    prompts = ("first exact prompt", "second exact prompt")
    _write_complete_paired_run(
        source,
        environment_seeds=(100001, 100002),
        prompts=prompts,
    )

    payload = build_manifest(
        source,
        minimum_episodes=2,
        instruction_type="unseen",
    )
    assert payload["kind"] == "robotwin_prompt_replay"
    assert payload["source"]["episode_count"] == 2
    assert [episode["prompt"] for episode in payload["episodes"]] == list(prompts)
    assert payload["episodes"][0]["generation_zero"]["shape"] == [28, 20]
    assert payload["episodes"][0]["generation_zero"]["dtype"] == "float32"

    manifest_path = tmp_path / "prompt_manifest.json"
    digest = write_manifest(payload, manifest_path)
    loaded = _load_prompt_manifest(str(manifest_path))
    assert loaded is not None
    assert loaded["sha256"] == digest
    assert len(loaded["entries"]) == 2


def test_prompt_manifest_builder_rejects_empty_prompt(tmp_path):
    source = tmp_path / "control"
    _write_complete_paired_run(source, prompts=("",))

    with pytest.raises(ValueError, match="has an empty prompt"):
        build_manifest(source)


def test_prompt_manifest_builder_rejects_duplicate_scene(tmp_path, monkeypatch):
    scene = (
        "adjust_bottle",
        "demo_clean",
        100001,
        "robotwin/adjust_bottle/demo_clean/scene-100001",
    )
    generation_zero = {
        "predicted_actions": np.zeros((28, 20), dtype=np.float32),
        "predicted_actions_normalized": np.zeros((28, 20), dtype=np.float32),
    }
    control = {
        "pairs": [(*scene, 17), (*scene, 18)],
        "starts": [
            {
                "episode_key": "episode-0",
                "context": {"episode_index": 0},
                "run_id": "test",
            },
            {
                "episode_key": "episode-1",
                "context": {"episode_index": 1},
                "run_id": "test",
            },
        ],
        "identity": {
            "checkpoint_sha256": "abc",
            "episode_noise_base_seed": 20260715,
        },
        "generation_zero": [generation_zero, generation_zero],
    }
    client = {
        "pairs": [scene, scene],
        "prompts": ["first prompt", "duplicate scene prompt"],
    }
    monkeypatch.setattr(manifest_builder, "load_run", lambda *args, **kwargs: control)
    monkeypatch.setattr(
        manifest_builder, "load_client_run", lambda *args, **kwargs: client
    )

    with pytest.raises(ValueError, match="duplicate.*scene|scene.*duplicate"):
        build_manifest(tmp_path)


def test_prompt_manifest_builder_rejects_noncanonical_noise_pair_key(tmp_path):
    source = tmp_path / "control"
    _write_complete_paired_run(
        source,
        noise_pair_keys=("adjust_bottle/demo_clean/100001",),
    )

    with pytest.raises(ValueError, match="noise_pair_key.*mismatch"):
        build_manifest(source)


def test_prompt_manifest_builder_enforces_minimum_episode_count(tmp_path):
    source = tmp_path / "control"
    _write_complete_paired_run(source)

    with pytest.raises(ValueError, match="only 1 episodes, need at least 2"):
        build_manifest(source, minimum_episodes=2)
