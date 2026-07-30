from __future__ import annotations

import hashlib
import json

import pytest

from scripts import build_robotwin_live_takeover_plan as plan_builder


def _episode(seed: int, ordinal: int) -> dict:
    prompt = f"prompt-{seed}"
    return {
        "ordinal": ordinal,
        "episode_index": ordinal,
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "environment_seed": seed,
        "noise_pair_key": f"robotwin/adjust_bottle/demo_clean/scene-{seed}",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


@pytest.mark.parametrize("anchor", [112, 140, 392])
def test_validate_anchor_accepts_reachable_generation_boundaries(anchor):
    plan_builder._validate_anchor(anchor, 400)


@pytest.mark.parametrize("anchor", [True, 111, 113, 400, 420])
def test_validate_anchor_rejects_invalid_or_unreachable_steps(anchor):
    with pytest.raises(ValueError, match="generation boundary"):
        plan_builder._validate_anchor(anchor, 400)


def test_build_anchor_trigger_accepts_ordered_boundaries():
    trigger = plan_builder._build_anchor_trigger(
        min_env_step=112,
        max_env_step=280,
        distance_m_at_most=0.12,
        upstream_step_limit=400,
    )

    assert trigger == {
        "kind": "active_eef_bottle_proximity",
        "min_env_step": 112,
        "max_env_step": 280,
        "distance_m_at_most": 0.12,
    }


@pytest.mark.parametrize(
    ("minimum", "maximum", "distance", "message"),
    [
        (140, 112, 0.12, "must not exceed"),
        (113, 280, 0.12, "generation boundary"),
        (112, 280, 0.0, "finite"),
        (112, 280, float("nan"), "finite"),
    ],
)
def test_build_anchor_trigger_rejects_invalid_contract(
    minimum, maximum, distance, message
):
    with pytest.raises(ValueError, match=message):
        plan_builder._build_anchor_trigger(
            min_env_step=minimum,
            max_env_step=maximum,
            distance_m_at_most=distance,
            upstream_step_limit=400,
        )


def test_validate_output_dir_rejects_nonempty_directory(tmp_path):
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    (output_dir / "partial.hdf5").write_bytes(b"partial")

    with pytest.raises(ValueError, match="must be empty"):
        plan_builder._validate_output_dir(output_dir)


def test_build_planned_episodes_requires_exact_ordered_scene_identity():
    episodes = [_episode(200001, 0), _episode(200002, 1)]

    planned = plan_builder._build_planned_episodes(episodes, 112)

    assert [entry["environment_seed"] for entry in planned] == [200001, 200002]
    assert [entry["generation_index"] for entry in planned] == [4, 4]
    episodes[1]["environment_seed"] = 200001
    with pytest.raises(ValueError, match="strictly increasing"):
        plan_builder._build_planned_episodes(episodes, 112)


def _control_fixture(tmp_path, monkeypatch):
    identity = {
        "episode_noise_base_seed": 20260715,
        "episode_noise_mode": "paired",
    }
    runtime = {
        "deployment_identity": identity,
        "cache_feedback_mode": "predicted",
        "episode_noise_mode": "paired",
        "episode_noise_base_seed": 20260715,
    }
    before = {"total_requests": 0, "inference_runtime": runtime}
    after = {
        "total_requests": 113,
        "inference_runtime": runtime,
        "episode": {"key": "episode-0"},
    }
    (tmp_path / "eval.exit").write_text("0\n", encoding="utf-8")
    (tmp_path / "info.before.json").write_text(json.dumps(before), encoding="utf-8")
    (tmp_path / "info.after.json").write_text(json.dumps(after), encoding="utf-8")
    manifest = {"instruction_type": "unseen", "episodes": [_episode(200001, 0)]}

    server_steps = [{"record": {}} for _ in range(113)]
    server_steps[112] = {
        "record": {
            "policy": {
                "policy_step": 112,
                "generation_index": 4,
                "chunk_offset": 0,
                "generated": True,
            }
        }
    }
    client_steps = [{"record": {"server_step": step + 1}} for step in range(113)]
    server = {
        "identity": identity,
        "starts": [{"episode_key": "episode-0"}],
        "steps": [server_steps],
    }
    client = {
        "starts": [{"context": {"episode_key": "episode-0"}}],
        "ends": [
            {
                "episode_key": "episode-0",
                "success": False,
                "step_count": 113,
            }
        ],
        "steps": [client_steps],
    }
    monkeypatch.setattr(
        plan_builder, "rebuild_prompt_manifest", lambda *args, **kwargs: manifest
    )
    monkeypatch.setattr(plan_builder, "load_run", lambda *args, **kwargs: server)
    monkeypatch.setattr(plan_builder, "load_client_run", lambda *args, **kwargs: client)
    return manifest, before, after, server, client


def test_validate_control_run_cross_checks_info_telemetry_and_anchor(
    tmp_path, monkeypatch
):
    manifest, before, _, _, _ = _control_fixture(tmp_path, monkeypatch)

    validated_info, noise_seed, outcomes = plan_builder._validate_control_run(
        tmp_path, prompt_manifest=manifest, anchor_env_step=112
    )

    assert validated_info == before
    assert noise_seed == 20260715
    assert outcomes == [
        {
            "control_success": False,
            "control_step_count": 113,
            "control_anchor_observed": True,
        }
    ]


def test_validate_control_run_supports_per_episode_request_counter_reset(
    tmp_path, monkeypatch
):
    manifest, _, after, server, client = _control_fixture(tmp_path, monkeypatch)
    manifest["episodes"].append(_episode(200002, 1))
    server["starts"].append({"episode_key": "episode-1"})
    server["steps"].append(list(server["steps"][0]))
    client["starts"].append({"context": {"episode_key": "episode-1"}})
    client["ends"].append(
        {"episode_key": "episode-1", "success": False, "step_count": 113}
    )
    client["steps"].append(list(client["steps"][0]))
    after["episode"]["key"] = "episode-1"
    (tmp_path / "info.after.json").write_text(json.dumps(after), encoding="utf-8")

    _, noise_seed, outcomes = plan_builder._validate_control_run(
        tmp_path, prompt_manifest=manifest, anchor_env_step=112
    )

    assert noise_seed == 20260715
    assert len(outcomes) == 2


def test_validate_control_run_allows_success_before_anchor(tmp_path, monkeypatch):
    manifest, _, after, server, client = _control_fixture(tmp_path, monkeypatch)
    server["steps"][0] = server["steps"][0][:80]
    client["steps"][0] = client["steps"][0][:80]
    client["ends"][0].update(success=True, step_count=80)
    after["total_requests"] = 80
    (tmp_path / "info.after.json").write_text(json.dumps(after), encoding="utf-8")

    _, _, outcomes = plan_builder._validate_control_run(
        tmp_path, prompt_manifest=manifest, anchor_env_step=112
    )

    assert outcomes == [
        {
            "control_success": True,
            "control_step_count": 80,
            "control_anchor_observed": False,
        }
    ]


def test_validate_control_run_rejects_failure_before_anchor(tmp_path, monkeypatch):
    manifest, _, after, server, client = _control_fixture(tmp_path, monkeypatch)
    server["steps"][0] = server["steps"][0][:80]
    client["steps"][0] = client["steps"][0][:80]
    client["ends"][0]["step_count"] = 80
    after["total_requests"] = 80
    (tmp_path / "info.after.json").write_text(json.dumps(after), encoding="utf-8")

    with pytest.raises(ValueError, match="failed control episode.*before anchor"):
        plan_builder._validate_control_run(
            tmp_path, prompt_manifest=manifest, anchor_env_step=112
        )


def test_validate_control_run_rejects_identity_drift(tmp_path, monkeypatch):
    manifest, _, after, _, _ = _control_fixture(tmp_path, monkeypatch)
    after["inference_runtime"] = {
        **after["inference_runtime"],
        "deployment_identity": {
            **after["inference_runtime"]["deployment_identity"],
            "episode_noise_base_seed": 7,
        },
    }
    (tmp_path / "info.after.json").write_text(json.dumps(after), encoding="utf-8")

    with pytest.raises(ValueError, match="identity changed"):
        plan_builder._validate_control_run(
            tmp_path, prompt_manifest=manifest, anchor_env_step=112
        )


def test_validate_control_run_rejects_manifest_not_derived_from_telemetry(
    tmp_path, monkeypatch
):
    manifest, _, _, _, _ = _control_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        plan_builder,
        "rebuild_prompt_manifest",
        lambda *args, **kwargs: {**manifest, "instruction_type": "seen"},
    )

    with pytest.raises(ValueError, match="does not exactly match"):
        plan_builder._validate_control_run(
            tmp_path, prompt_manifest=manifest, anchor_env_step=112
        )
