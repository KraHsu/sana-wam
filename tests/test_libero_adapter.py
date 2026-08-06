from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmarks.libero import sana_wam2libero_interface as adapter


def _observation() -> dict:
    head = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    wrist = (head + 50).astype(np.uint8)
    return {
        "agentview_image": head,
        "robot0_eye_in_hand_image": wrist,
        "robot0_eef_pos": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.01, -0.01], dtype=np.float32),
    }


def test_adapter_has_no_eager_simulator_imports() -> None:
    source_path = Path(adapter.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert not {"libero", "robosuite", "mujoco", "torch"} & imported_roots


def test_quaternion_to_axis_angle_identity_and_pi_rotation() -> None:
    np.testing.assert_array_equal(
        adapter.quaternion_xyzw_to_axis_angle([0.0, 0.0, 0.0, 1.0]),
        np.zeros(3, dtype=np.float32),
    )
    rotation = adapter.quaternion_xyzw_to_axis_angle([0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(rotation, [0.0, 0.0, np.pi], atol=1e-6)


def test_quaternion_formula_preserves_training_conversion_sign_semantics() -> None:
    angle = 2.0 * np.pi / 3.0
    positive_quaternion = [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
    negative_quaternion = [-value for value in positive_quaternion]
    positive = adapter.quaternion_xyzw_to_axis_angle(positive_quaternion)
    negative = adapter.quaternion_xyzw_to_axis_angle(negative_quaternion)
    np.testing.assert_allclose(positive, [0.0, 0.0, angle], atol=1e-6)
    np.testing.assert_allclose(negative, [0.0, 0.0, angle - 2 * np.pi], atol=1e-6)
    assert not np.array_equal(positive, negative)


def test_quaternion_invalid_values_fail() -> None:
    with pytest.raises(ValueError, match="shape"):
        adapter.quaternion_xyzw_to_axis_angle([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="zero norm"):
        adapter.quaternion_xyzw_to_axis_angle([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        adapter.quaternion_xyzw_to_axis_angle([0.0, np.nan, 0.0, 1.0])


def test_extract_state_uses_eef_axis_angle_and_two_gripper_qpos() -> None:
    observation = _observation()
    observation["robot0_eef_quat"] = np.array(
        [0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)], dtype=np.float32
    )
    state = adapter.extract_libero_state(observation)
    assert state.shape == (8,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(
        state,
        [0.1, -0.2, 0.3, 0.0, np.pi / 2, 0.0, 0.01, -0.01],
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("robot0_eef_pos", [0.0, 1.0], "shape"),
        ("robot0_eef_quat", [0.0, 0.0, np.inf, 1.0], "non-finite"),
        ("robot0_gripper_qpos", [0.0], "shape"),
    ],
)
def test_extract_state_rejects_contract_drift(key, value, message) -> None:
    observation = _observation()
    observation[key] = value
    with pytest.raises(ValueError, match=message):
        adapter.extract_libero_state(observation)


def test_camera_mapping_rotates_both_images_and_leaves_right_missing() -> None:
    observation = _observation()
    cameras = adapter.extract_libero_cameras(observation)
    np.testing.assert_array_equal(
        cameras["head"], observation["agentview_image"][::-1, ::-1]
    )
    np.testing.assert_array_equal(
        cameras["left"], observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    assert cameras["head"].flags.c_contiguous
    assert cameras["left"].flags.c_contiguous
    assert cameras["right"] is None


def test_camera_mapping_rejects_wrong_shape_or_dtype() -> None:
    observation = _observation()
    observation["agentview_image"] = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxWx3"):
        adapter.extract_libero_cameras(observation)
    observation = _observation()
    observation["agentview_image"] = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        adapter.extract_libero_cameras(observation)


def test_policy_action_passes_motion_and_maps_gripper() -> None:
    closed = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.5], dtype=np.float32)
    opened = closed.copy()
    opened[-1] = 0.5001
    closed_env = adapter.policy_action_to_libero(closed)
    open_env = adapter.policy_action_to_libero(opened)
    np.testing.assert_array_equal(closed_env[:6], closed[:6])
    np.testing.assert_array_equal(open_env[:6], opened[:6])
    assert closed_env[-1] == 1.0
    assert open_env[-1] == -1.0


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(14, dtype=np.float32),
        np.zeros(20, dtype=np.float32),
        np.zeros((1, 7), dtype=np.float32),
    ],
)
def test_policy_action_rejects_non_7d_checkpoint_outputs(action) -> None:
    with pytest.raises(ValueError, match="exact 7D"):
        adapter.policy_action_to_libero(action)


def _expected_server_contract() -> dict:
    config = yaml.safe_load(
        Path("benchmarks/libero/policy_config.yml").read_text(encoding="utf-8")
    )
    return config["expected_server_contract"]


def _server_info(contract: dict | None = None) -> dict:
    expected = _expected_server_contract() if contract is None else contract
    return {
        "model": "sana-wam",
        "inference_runtime": {
            "deployment_identity": {
                "benchmark_contract": expected,
                "dataloader_type": "libero",
                "action_mode": "libero_relative_eef",
                "state_mode": "libero_eef_axis_angle_gripper",
                "action_dim": 7,
                "state_dim": 8,
                "multiview": True,
                "camera_layout": [
                    "head_camera",
                    "left_wrist_camera",
                    "right_wrist_camera",
                ],
                "normalize_mode": "min-max",
                "normalizers": {
                    "action": {
                        "active": True,
                        "configured_mode": "min-max",
                        "dim": 7,
                        "explicit": True,
                    },
                    "state": {
                        "active": True,
                        "configured_mode": "min-max",
                        "dim": 8,
                        "explicit": True,
                    },
                },
                "episode_noise_mode": "paired",
            }
        },
    }


def test_server_contract_rejects_robotwin_and_semantically_wrong_7d() -> None:
    expected = _expected_server_contract()
    robotwin = _server_info(expected)
    robotwin["inference_runtime"]["deployment_identity"]["action_dim"] = 20
    with pytest.raises(RuntimeError, match="action_dim"):
        adapter.validate_server_contract(robotwin, expected)

    wrong_semantics = _server_info(expected)
    wrong_semantics["inference_runtime"]["deployment_identity"][
        "benchmark_contract"
    ] = {**expected, "action_representation": "absolute_pose"}
    with pytest.raises(RuntimeError, match="frozen LIBERO"):
        adapter.validate_server_contract(wrong_semantics, expected)


def test_model_noise_seed_is_stable_and_episode_specific() -> None:
    arguments = {
        "base_seed": 1234,
        "suite": "libero_goal",
        "task_id": 2,
        "trial_index": 3,
        "init_state_index": 3,
    }
    seed = adapter.derive_model_noise_seed(**arguments)
    assert seed == adapter.derive_model_noise_seed(**arguments)
    assert 0 <= seed < 2**63
    assert seed != adapter.derive_model_noise_seed(
        **{**arguments, "trial_index": 4, "init_state_index": 4}
    )


def test_policy_action_rejects_nonfinite_or_wrong_gripper_convention() -> None:
    action = np.zeros(7, dtype=np.float32)
    action[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        adapter.policy_action_to_libero(action)
    action = np.zeros(7, dtype=np.float32)
    action[-1] = -1.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adapter.policy_action_to_libero(action)


def test_http_client_uses_named_reset_and_one_current_observation(monkeypatch) -> None:
    calls: dict[str, list] = {"get": [], "reset": [], "post": []}
    active: dict = {}

    def fake_get(server, endpoint, timeout=10):
        calls["get"].append((server, endpoint, timeout))
        if endpoint == "/health":
            return {"status": "healthy"}
        if endpoint == "/info":
            return _server_info()
        raise AssertionError(endpoint)

    def fake_reset(server, timeout=10, **kwargs):
        calls["reset"].append((server, timeout, kwargs))
        active.update(kwargs)
        return {
            "status": "ok",
            "episode_key": kwargs["episode_key"],
            "model_noise_seed": kwargs["model_noise_seed"],
            "duplicate": False,
        }

    def fake_encode(image):
        return f"encoded-{int(np.asarray(image).sum())}"

    def fake_post(server, endpoint, payload, timeout=300):
        calls["post"].append((server, endpoint, payload, timeout))
        return {
            "action": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0],
            "step": len(calls["post"]),
            "episode_key": active["episode_key"],
            "model_noise_seed": active["model_noise_seed"],
        }

    monkeypatch.setattr(adapter.client, "get", fake_get)
    monkeypatch.setattr(adapter.client, "reset", fake_reset)
    monkeypatch.setattr(adapter.client, "encode_numpy_b64", fake_encode)
    monkeypatch.setattr(adapter.client, "post", fake_post)

    policy = adapter.LiberoPolicyClient(
        host="policy.local",
        http_port=9988,
        request_timeout=12,
        expected_server_contract=_expected_server_contract(),
        model_noise_base_seed=1234,
        run_nonce="test-run",
        health_timeout=1,
        health_poll_interval=0.001,
    )
    with pytest.raises(RuntimeError, match="reset_episode"):
        policy.predict(_observation(), "pick the bowl")

    episode_key = policy.reset_episode(
        suite="libero_spatial",
        task_id=2,
        trial_index=0,
        init_state_index=7,
        task_name="pick_bowl",
        prompt="pick the bowl",
    )
    action = policy.predict(_observation(), "pick the bowl")

    assert episode_key == "libero/run-test-run/libero_spatial/task-2/trial-0/init-7"
    assert calls["reset"][0][2]["metadata"]["noise_pair_key"] == (
        "libero/libero_spatial/task-2/trial-0/init-7"
    )
    expected_noise_seed = adapter.derive_model_noise_seed(
        base_seed=1234,
        suite="libero_spatial",
        task_id=2,
        trial_index=0,
        init_state_index=7,
    )
    assert calls["reset"][0][2]["model_noise_seed"] == expected_noise_seed
    assert len(calls["post"]) == 1
    payload = calls["post"][0][2]
    assert payload["images"]["head_camera"].startswith("encoded-")
    assert payload["images"]["left_wrist_camera"].startswith("encoded-")
    assert payload["images"]["right_wrist_camera"] is None
    assert len(payload["state"]) == 8
    assert payload["prompt"] == "pick the bowl"
    np.testing.assert_allclose(action[:6], [0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    assert action[-1] == -1.0
    assert policy.request_count == 1


@pytest.mark.parametrize("bad_step", [None, 0, 2, True])
def test_http_client_rejects_missing_or_out_of_sequence_server_step(
    monkeypatch, bad_step
) -> None:
    expected_contract = _expected_server_contract()
    active: dict = {}
    monkeypatch.setattr(
        adapter.client,
        "get",
        lambda _server, endpoint, timeout=10: (
            {"status": "healthy"} if endpoint == "/health" else _server_info()
        ),
    )

    def fake_reset(_server, timeout=10, **kwargs):
        active.update(kwargs)
        return {
            "status": "ok",
            "episode_key": kwargs["episode_key"],
            "model_noise_seed": kwargs["model_noise_seed"],
            "duplicate": False,
        }

    monkeypatch.setattr(adapter.client, "reset", fake_reset)
    monkeypatch.setattr(adapter.client, "encode_numpy_b64", lambda _image: "image")
    monkeypatch.setattr(
        adapter.client,
        "post",
        lambda *_args, **_kwargs: {
            "action": [0.0] * 7,
            "step": bad_step,
            "episode_key": active["episode_key"],
            "model_noise_seed": active["model_noise_seed"],
        },
    )
    policy = adapter.LiberoPolicyClient(
        expected_server_contract=expected_contract,
        model_noise_base_seed=1,
        run_nonce="step-test",
        health_timeout=1,
        health_poll_interval=0.001,
    )
    policy.reset_episode(
        suite="libero_object",
        task_id=0,
        trial_index=0,
        init_state_index=0,
        task_name="task",
        prompt="prompt",
    )
    with pytest.raises(RuntimeError, match="step mismatch"):
        policy.predict(_observation(), "prompt")


def test_http_client_rejects_duplicate_reset_and_seed_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        adapter.client,
        "get",
        lambda _server, endpoint, timeout=10: (
            {"status": "healthy"} if endpoint == "/health" else _server_info()
        ),
    )
    policy = adapter.LiberoPolicyClient(
        expected_server_contract=_expected_server_contract(),
        model_noise_base_seed=9,
        run_nonce="reset-test",
        health_timeout=1,
        health_poll_interval=0.001,
    )
    monkeypatch.setattr(
        adapter.client,
        "reset",
        lambda _server, timeout=10, **kwargs: {
            "status": "ok",
            "episode_key": kwargs["episode_key"],
            "model_noise_seed": kwargs["model_noise_seed"],
            "duplicate": True,
        },
    )
    reset_args = {
        "suite": "libero_10",
        "task_id": 1,
        "trial_index": 0,
        "init_state_index": 0,
        "task_name": "task",
        "prompt": "prompt",
    }
    with pytest.raises(RuntimeError, match="duplicate reset"):
        policy.reset_episode(**reset_args)

    monkeypatch.setattr(
        adapter.client,
        "reset",
        lambda _server, timeout=10, **kwargs: {
            "status": "ok",
            "episode_key": kwargs["episode_key"],
            "model_noise_seed": kwargs["model_noise_seed"] + 1,
            "duplicate": False,
        },
    )
    with pytest.raises(RuntimeError, match="noise seed mismatch"):
        policy.reset_episode(**reset_args)

    monkeypatch.setattr(
        adapter.client,
        "reset",
        lambda _server, timeout=10, **kwargs: {
            "status": "ok",
            "model_noise_seed": kwargs["model_noise_seed"],
            "duplicate": False,
        },
    )
    with pytest.raises(RuntimeError, match="reset key mismatch"):
        policy.reset_episode(**reset_args)
