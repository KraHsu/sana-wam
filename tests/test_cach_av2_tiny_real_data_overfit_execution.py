from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import pytest
import torch

from sana_wam.model import cach_av2_tiny_real_data_overfit_execution as av2e


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/cach_av2_tiny_real_data_overfit_execution.yaml"
)
RUNNER_PATH = (
    REPO_ROOT / "scripts/run_cach_av2_tiny_real_data_overfit_execution.py"
)
EXPECTED_CONFIG_SHA256 = (
    "c2aeb839b711318c7c10c06a93adff3d3de036d93f2ec061748e27e06b04b62e"
)
AUTHORITY_SHA256 = (
    "7a2c2957f794de03fb46715d2446fc8780d09e237d70e6ac4051f69bf284aef2"
)
GPU_UUID = "GPU-1ec28cfb-f501-23f3-f865-275a744ca053"
NONCE = "96c7c972d34733dd37475b183c136ee0"
RUN_ROOT = (
    "/DATA/share/sana_cach_wam_nonformal_screens/"
    "cach_a4_av2_tiny_real_data/06f5d09127f8/"
    f"cach-a4-av2-{NONCE}"
)


def _config_bytes() -> bytes:
    return CONFIG_PATH.read_bytes()


def _config() -> dict[str, object]:
    value = json.loads(_config_bytes())
    assert isinstance(value, dict)
    return value


def _load_runner():
    spec = importlib.util.spec_from_file_location("_av2_execution_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_mock_episode(path: Path, episode_id: int) -> None:
    """Write the minimum fixed RoboTwin schema with encoded color frames."""

    cv2 = pytest.importorskip("cv2")
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.arange(33, dtype=np.float64)
    left = np.zeros((33, 7), dtype=np.float64)
    right = np.zeros((33, 7), dtype=np.float64)
    left[:, 0] = 100.0 * episode_id + rows
    left[:, 1] = 10.0 + rows / 10.0
    left[:, 2] = -5.0
    right[:, 0] = -100.0 * episode_id - rows
    right[:, 1] = 20.0
    right[:, 2] = rows / 20.0
    # xyzw identity quaternion: the final component is w.
    left[:, 6] = 1.0
    right[:, 6] = 1.0
    left_grip = (rows / 64.0).astype(np.float64)
    right_grip = (1.0 - rows / 64.0).astype(np.float64)

    with h5py.File(path, "w") as handle:
        endpose = handle.create_group("endpose")
        endpose.create_dataset("left_endpose", data=left)
        endpose.create_dataset("right_endpose", data=right)
        endpose.create_dataset("left_gripper", data=left_grip)
        endpose.create_dataset("right_gripper", data=right_grip)
        camera = handle.create_group("observation").create_group("head_camera")
        encoded_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        encoded = camera.create_dataset("rgb", shape=(33,), dtype=encoded_dtype)
        for row in range(33):
            # Distinct native B/G/R values make an accidental cvtColor visible.
            image = np.empty((8, 8, 3), dtype=np.uint8)
            image[..., 0] = 16 + row
            image[..., 1] = 48 + episode_id
            image[..., 2] = 96 - row
            ok, payload = cv2.imencode(".jpg", image)
            assert ok
            encoded[row] = payload.reshape(-1)


def _write_mock_dataset(root: Path) -> None:
    data = root / "adjust_bottle/aloha-agilex_clean_50/data"
    for episode_id in range(16):
        _write_mock_episode(data / f"episode{episode_id}.hdf5", episode_id)


def test_execution_config_is_canonical_and_exact() -> None:
    payload = _config_bytes()
    config = _config()
    canonical = (
        json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert payload == canonical
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_CONFIG_SHA256
    assert config["schema"] == av2e.AV2_EXECUTION_SCHEMA
    assert config["architecture_id"] == av2e.AV2_ARCHITECTURE_ID
    assert config["operator_class"] == "VENDOR_KERNEL"
    assert config["integration_path"] == "EXPERIMENTAL_PATH"

    authority = config["authority"]
    assert isinstance(authority, dict)
    statement = authority["execution_statement"]
    assert isinstance(statement, str)
    assert statement == "授权 AV2 tiny real-data 单GPU运行"
    assert len(statement.encode("utf-8")) == 38
    assert hashlib.sha256(statement.encode("utf-8")).hexdigest() == AUTHORITY_SHA256
    assert authority["execution_statement_sha256"] == AUTHORITY_SHA256
    assert authority["token_or_claim_operation"] is False

    data = config["data_contract"]
    assert isinstance(data, dict)
    assert data == {
        "action_dim": 20,
        "action_mode": "ABSOLUTE_EEF20",
        "action_rows": [1, 33],
        "batch_size": 8,
        "camera": "head_camera",
        "dataset_root": "/DATA/share/RoboTwin2.0/dataset",
        "frame_indices": [0, 8, 16, 24, 32],
        "heldout_episode_ids": list(range(8, 16)),
        "noisy_video_recipe": "ANCHOR_RGB_MEAN_DIV_255_REPEAT_5",
        "normalize": None,
        "observation_decoder": "CV2_IMDECODE_COLOR_NO_EXTRA_CONVERSION",
        "observation_projection": "SPATIAL_RGB_MEAN_DIV_255",
        "proprio_recipe": (
            "ABSOLUTE_EEF20_RAW_ROW_0_REPEAT_2_NORMALIZATION_NONE"
        ),
        "raw_end_exclusive": 33,
        "raw_start": 0,
        "shuffle_permutation": [1, 0, 3, 2, 5, 4, 7, 6],
        "target_forward_access": False,
        "target_recipe": "RGB_MEAN_DIV_255_MINUS_ANCHOR_FRAME_0",
        "task": "adjust_bottle",
        "train_episode_ids": list(range(8)),
        "variant": "aloha-agilex_clean_50",
        "window_count_per_split": 8,
    }

    training = config["training_contract"]
    assert isinstance(training, dict)
    assert training["optimizer"] == "AdamW"
    assert training["lr_start"] == pytest.approx(3e-3)
    assert training["lr_end"] == pytest.approx(3e-5)
    assert training["max_steps_per_arm"] == 1000
    assert training["serial_arm_order"] == ["reference", "candidate"]
    assert training["fresh_initialization"] is True
    assert training["checkpoint_load"] is False
    assert training["checkpoint_save"] is False
    assert training["save_final_screen_state"] is False

    hardware = config["hardware_contract"]
    assert isinstance(hardware, dict)
    assert hardware["gpu_count"] == 1
    assert hardware["physical_gpu_index"] == 0
    assert hardware["gpu_uuid"] == GPU_UUID
    assert hardware["cuda_visible_devices"] == GPU_UUID

    run = config["run_contract"]
    assert isinstance(run, dict)
    assert run["nonce"] == NONCE
    assert run["root"] == RUN_ROOT
    assert run["automatic_rerun"] is False
    assert run["reuse"] is False

    assert config["thresholds"] == {
        "candidate_heldout_decrease_min": 0.05,
        "candidate_vs_reference_noninferiority_ratio_max": 1.0,
        "correct_vs_no_action_improvement_min": 0.05,
        "correct_vs_shuffle_improvement_min": 0.05,
        "min_action_variance": 1e-8,
        "min_motion_mse": 1e-8,
        "min_shuffle_mse": 1e-8,
    }
    assert config["verdict_mapping"] == {
        "all_go_checks_pass": "AV2_GO",
        "data_invalid": "DATA_INADEQUATE",
        "implementation_invalid": "IMPLEMENTATION_INVALID",
        "numerical_invalid": "NUMERICAL_INVALID",
        "review_band_with_consumed_token": (
            "REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED"
        ),
        "review_token_state": "CONSUMED_NO_NEW_TOKEN_ALLOWED",
        "stop_line_with_consumed_token": "REDUCED_ARCH_STOP",
    }


def test_real_data_loader_uses_fixed_disjoint_rows_without_normalization(
    tmp_path: Path,
) -> None:
    _write_mock_dataset(tmp_path)
    bundle = av2e.load_av2_real_data(tmp_path)

    assert [window.episode_id for window in bundle.train.windows] == [
        str(index) for index in range(8)
    ]
    assert [window.episode_id for window in bundle.heldout.windows] == [
        str(index) for index in range(8, 16)
    ]
    for split in (bundle.train, bundle.heldout):
        assert split.actions.shape == (8, 32, 20)
        assert split.noisy_video.shape == (8, 5, 3, 1, 1)
        assert split.target.shape == (8, 5, 3, 1, 1)
        assert split.proprio.shape == (8, 2, 20)
        assert torch.equal(
            split.noisy_video,
            split.noisy_video[:, :1].expand_as(split.noisy_video),
        )
        assert int(split.target[:, 0].count_nonzero()) == 0
        assert torch.equal(split.proprio[:, 0], split.proprio[:, 1])
        assert bool(split.action_valid_mask.all())
        assert bool(split.frame_valid_mask.all())

    # Row 1 is the first action, while row 0 is the repeated proprio state.
    # Values greater than one also prove there was no hidden normalization.
    assert bundle.train.actions[0, 0, 0].item() == pytest.approx(1.0)
    assert bundle.train.proprio[0, 0, 0].item() == pytest.approx(0.0)
    assert bundle.heldout.actions[0, 0, 0].item() == pytest.approx(801.0)
    assert bundle.heldout.proprio[0, 0, 0].item() == pytest.approx(800.0)

    # OpenCV-native BGR is deliberately retained: B rises and R falls.
    train_motion = bundle.train.target[0, 1, :, 0, 0]
    assert train_motion[0].item() > 0.0
    assert train_motion[2].item() < 0.0
    adequacy = av2e.assess_data_adequacy(bundle)
    assert adequacy["train"]["adequate"] is True
    assert adequacy["heldout"]["adequate"] is True
    assert adequacy["all_adequate"] is True


def test_masked_future_mse_ignores_anchor_and_scores_only_valid_future() -> None:
    prediction = torch.zeros((8, 5, 3, 1, 1), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    mask = torch.ones((8, 5), dtype=torch.bool)
    prediction[:, 0] = 1000.0
    assert av2e.masked_future_mse(prediction, target, mask).item() == 0.0

    prediction[:, 1] = 2.0
    assert av2e.masked_future_mse(prediction, target, mask).item() == pytest.approx(1.0)
    only_first_future = torch.zeros_like(mask)
    only_first_future[:, 1] = True
    assert av2e.masked_future_mse(
        prediction, target, only_first_future
    ).item() == pytest.approx(4.0)

    anchor_only = torch.zeros_like(mask)
    anchor_only[:, 0] = True
    with pytest.raises(av2e.AV2ExecutionError, match="future loss mask is empty"):
        av2e.masked_future_mse(prediction, target, anchor_only)


def test_materialized_forward_contract_cannot_read_real_target() -> None:
    source = inspect.getsource(av2e.materialize_a4_task)
    tree = ast.parse(source)
    split_target_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "split"
        and node.attr == "target"
    ]
    assert split_target_reads == []
    assert "base_target=sentinel.clone()" in source
    assert "video_target=sentinel.clone()" in source
    assert "assert_target_sentinels(task)" in source


def test_classify_av2_uses_frozen_go_stop_review_and_invalid_states() -> None:
    go_metrics = {
        "candidate_heldout_decrease": 0.05,
        "candidate_vs_reference_ratio": 1.0,
        "correct_vs_no_action_improvement": 0.05,
        "correct_vs_shuffle_improvement": 0.05,
    }
    verdict, reason, checks = av2e.classify_av2(
        all_validity=True, metrics=go_metrics
    )
    assert verdict == "AV2_GO"
    assert reason is None
    assert all(checks.values())

    stop_metrics = {
        "candidate_heldout_decrease": 0.01,
        "candidate_vs_reference_ratio": 1.0,
        "correct_vs_no_action_improvement": 0.0,
        "correct_vs_shuffle_improvement": 0.0,
    }
    verdict, reason, checks = av2e.classify_av2(
        all_validity=True, metrics=stop_metrics
    )
    assert verdict == "REDUCED_ARCH_STOP"
    assert reason == "AV2_ACTION_CONDITIONING_UNSUPPORTED"
    assert checks["candidate_heldout_decrease"] is False

    review_metrics = {
        "candidate_heldout_decrease": 0.01,
        "candidate_vs_reference_ratio": 0.99,
        "correct_vs_no_action_improvement": 0.01,
        "correct_vs_shuffle_improvement": 0.01,
    }
    verdict, reason, checks = av2e.classify_av2(
        all_validity=True, metrics=review_metrics
    )
    assert verdict == "REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED"
    assert reason == "AV2_REVIEW_BAND_NO_BUG_BUDGET"
    assert not all(checks.values())

    verdict, reason, _ = av2e.classify_av2(
        all_validity=False, metrics=go_metrics
    )
    assert verdict == "AV2_INVALID_FAIL_CLOSED"
    assert reason == "AV2_VALIDITY_FAILURE"

    nonfinite = {**go_metrics, "candidate_heldout_decrease": float("nan")}
    verdict, reason, _ = av2e.classify_av2(
        all_validity=True, metrics=nonfinite
    )
    assert verdict == "AV2_INVALID_FAIL_CLOSED"
    assert reason == "NONFINITE_DECISION_METRIC"


def test_execution_runner_static_preflight_has_no_capability_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_runner()
    calls: list[str] = []

    def forbidden(name: str):
        def call(*_args: object, **_kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"static preflight reached {name}")

        return call

    for name in (
        "_load_real_data_after_preflight",
        "_probe_gpu_after_preflight",
        "_create_root_after_preflight",
        "_execute_after_preflight",
    ):
        assert hasattr(runner, name)
        monkeypatch.setattr(runner, name, forbidden(name))

    assert runner.main(["--static-preflight"]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["state"] == "STATIC_PREFLIGHT_OK"
    assert emitted["schema"] == av2e.AV2_EXECUTION_SCHEMA
    assert calls == []
