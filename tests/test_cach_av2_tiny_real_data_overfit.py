from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from sana_wam.model import cach_av2_tiny_real_data_overfit as av2
from sana_wam.model import cach_av1b_a4_state_conditioned_causal_odd_stream_r2 as a4


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/cach_av2_tiny_real_data_overfit.yaml"
CARD_PATH = (
    REPO_ROOT
    / "docs/cach_sana_wam/architecture_validation/cach_a4_av2/"
    "CACH_A4_AV2_TINY_REAL_DATA_OVERFIT_RUN_CARD.json"
)
RUNNER_PATH = REPO_ROOT / "scripts/run_cach_av2_tiny_real_data_overfit.py"
R2_RESULT = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r2/"
    "06f5d09127f8/cach-a4-state-stream-r2-f294a23c285082ae61b1685accec416e/"
    "RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _synthetic_adequate_fixture() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026080601)
    actions = torch.randn((8, 32, 20), generator=generator, dtype=torch.float32)
    action_mask = torch.ones((8, 32), dtype=torch.bool)
    permutation = torch.tensor(av2.AV2_SHUFFLE_PERMUTATION, dtype=torch.long)
    shuffled = actions.index_select(0, permutation)
    target = torch.zeros((8, 5, 3, 1, 1), dtype=torch.float32)
    row_scale = torch.arange(1, 9, dtype=torch.float32).view(8, 1, 1, 1)
    for horizon in range(1, 5):
        target[:, horizon] = float(horizon) * 0.01 * row_scale
    frame_mask = torch.ones((8, 5), dtype=torch.bool)
    return actions, action_mask, target, frame_mask, shuffled


def test_av2_static_config_is_pending_and_exact() -> None:
    validated = av2.validate_av2_static_config(_config())
    assert validated.schema == av2.AV2_CONFIG_SCHEMA
    assert validated.state == av2.AV2_CONFIG_STATE
    assert validated.operator_class == "VENDOR_KERNEL"
    assert validated.integration_path == "EXPERIMENTAL_PATH"
    assert validated.training_contract["max_steps_per_arm"] == 1000
    assert validated.training_contract["checkpoint_load"] is False
    assert validated.training_contract["checkpoint_save"] is False
    assert validated.authority.complete is False
    with pytest.raises(av2.AV2AuthorityBlocked, match="successor execution card"):
        validated.authority.require_execution_authority()

    drifted = _config()
    drifted["unexpected"] = True
    with pytest.raises(av2.AV2ContractError, match="keys differ"):
        av2.validate_av2_static_config(drifted)


def test_av2_runner_static_preflight_and_auth_blocked_before_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    static = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--static-preflight"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    assert static.returncode == 0, static.stderr
    static_result = json.loads(static.stdout)
    assert static_result["state"] == "STATIC_PREFLIGHT_OK"
    assert static_result["schema"] == av2.AV2_CONFIG_SCHEMA

    blocked = subprocess.run(
        [sys.executable, str(RUNNER_PATH)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    assert blocked.returncode == 2, blocked.stderr
    assert json.loads(blocked.stdout)["state"] == "AUTH_BLOCKED"

    spec = importlib.util.spec_from_file_location("_av2_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls: list[str] = []
    runner._access_dataset_after_authority = lambda: calls.append("dataset")
    runner._probe_gpu_after_authority = lambda: calls.append("gpu")
    runner._import_vendor_after_authority = lambda: calls.append("vendor")
    runner._create_root_after_authority = lambda: calls.append("root")
    assert runner.main([]) == 2
    assert calls == []

    capsys.readouterr()
    original_read_bytes = Path.read_bytes
    source_loader_calls: list[tuple[object, ...]] = []

    def tampered_source_read(path: Path) -> bytes:
        if path == runner.DEFAULT_SOURCE:
            return b"tampered AV-2 source bytes"
        return original_read_bytes(path)

    def forbidden_source_loader(*args: object, **_kwargs: object):
        source_loader_calls.append(args)
        raise AssertionError("tampered AV-2 source reached loader construction")

    with monkeypatch.context() as source_patch:
        source_patch.setattr(Path, "read_bytes", tampered_source_read)
        source_patch.setattr(
            runner.importlib.util,
            "spec_from_file_location",
            forbidden_source_loader,
        )
        assert runner.main(["--static-preflight"]) == 2
        source_rejected = json.loads(capsys.readouterr().out)
        assert source_rejected["state"] == "SOURCE_SHA_MISMATCH"
    assert source_loader_calls == []

    config_parse_calls: list[bytes] = []

    def tampered_config_read(path: Path) -> bytes:
        if path == runner.DEFAULT_CONFIG:
            return b"tampered AV-2 config bytes"
        return original_read_bytes(path)

    def forbidden_config_parse(payload: bytes, *, name: str):
        del name
        config_parse_calls.append(payload)
        raise AssertionError("tampered AV-2 config reached JSON parsing")

    with monkeypatch.context() as config_patch:
        config_patch.setattr(Path, "read_bytes", tampered_config_read)
        config_patch.setattr(runner, "_strict_json_mapping", forbidden_config_parse)
        assert runner.main(["--static-preflight"]) == 2
        config_rejected = json.loads(capsys.readouterr().out)
        assert config_rejected["state"] == "CONFIG_SHA_MISMATCH"
    assert config_parse_calls == []

    path_reads: list[Path] = []
    source_imports: list[Path] = []

    def forbidden_read(path: Path) -> bytes:
        path_reads.append(path)
        raise AssertionError("noncanonical AV-2 path was read")

    def forbidden_import(path: Path):
        source_imports.append(path)
        raise AssertionError("noncanonical AV-2 source was imported")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(runner, "_load_contract_source", forbidden_import)
    for argv in (
        ["--config", "/tmp/noncanonical-av2-config.json"],
        ["--source", "/tmp/noncanonical-av2-source.py"],
    ):
        assert runner.main(argv) == 2
        rejected = json.loads(capsys.readouterr().out)
        assert rejected["state"] == "PATH_REJECTED"
    assert path_reads == []
    assert source_imports == []


def test_av2_window_split_is_exact_and_disjoint() -> None:
    train = tuple(
        av2.WindowIdentity("task", f"train-{index}", 0, 33)
        for index in range(8)
    )
    heldout = tuple(
        av2.WindowIdentity("task", f"heldout-{index}", 0, 33)
        for index in range(8)
    )
    av2.assert_train_holdout_disjoint(train, heldout)
    with pytest.raises(av2.AV2ContractError, match="overlap"):
        av2.assert_train_holdout_disjoint(train, heldout[:-1] + (train[0],))
    overlapping_interval = av2.WindowIdentity("task", "train-0", 16, 49)
    with pytest.raises(av2.AV2ContractError, match="raw intervals overlap"):
        av2.assert_train_holdout_disjoint(
            train, heldout[:-1] + (overlapping_interval,)
        )
    with pytest.raises(av2.AV2ContractError, match="33 raw rows"):
        av2.WindowIdentity("task", "episode", 0, 32)


def test_av2_data_adequacy_accepts_only_informative_typed_data() -> None:
    actions, action_mask, target, frame_mask, shuffled = _synthetic_adequate_fixture()
    report = av2.evaluate_data_adequacy(
        actions=actions,
        action_valid_mask=action_mask,
        target=target,
        frame_valid_mask=frame_mask,
        shuffled_actions=shuffled,
    )
    assert report.adequate is True
    assert report.action_variance > 0.0
    assert report.shuffle_mse > 0.0
    assert report.motion_mse > 0.0

    cases = (
        {"action_valid_mask": torch.zeros_like(action_mask)},
        {"actions": torch.zeros_like(actions), "shuffled_actions": torch.zeros_like(actions)},
        {"shuffled_actions": actions.clone()},
        {"target": torch.zeros_like(target)},
        {"frame_valid_mask": torch.zeros_like(frame_mask)},
    )
    base = {
        "actions": actions,
        "action_valid_mask": action_mask,
        "target": target,
        "frame_valid_mask": frame_mask,
        "shuffled_actions": shuffled,
    }
    for changed in cases:
        assert av2.evaluate_data_adequacy(**{**base, **changed}).adequate is False

    with pytest.raises(av2.AV2ContractError, match="must stay on CPU"):
        av2.evaluate_data_adequacy(
            **{
                **base,
                "actions": torch.empty((8, 32, 20), device="meta"),
                "shuffled_actions": torch.empty((8, 32, 20), device="meta"),
            }
        )


def test_av2_counterfactual_metrics_use_same_target_and_final_values() -> None:
    target = torch.zeros((8, 5, 3, 1, 1), dtype=torch.float32)
    frame_mask = torch.ones((8, 5), dtype=torch.bool)
    metrics = av2.counterfactual_mse_metrics(
        correct=torch.zeros_like(target),
        shuffled=torch.ones_like(target),
        no_action=torch.full_like(target, 2.0),
        reference=torch.full_like(target, 0.5),
        target=target,
        frame_valid_mask=frame_mask,
    )
    assert metrics["correct_mse"] == 0.0
    assert metrics["shuffled_mse"] == 1.0
    assert metrics["no_action_mse"] == 4.0
    assert metrics["reference_mse"] == 0.25
    assert metrics["correct_vs_shuffle_improvement"] == 1.0
    assert metrics["correct_vs_no_action_improvement"] == 1.0


def test_av2_fresh_shared_theta0_is_equal_disjoint_and_alias_free() -> None:
    reference = {"common.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    candidate = {name: value.clone() for name, value in reference.items()}
    action = {"action.output.weight": torch.zeros((3, 4), dtype=torch.float32)}
    manifest = av2.shared_theta0_manifest(
        reference_shared=reference,
        candidate_shared=candidate,
        candidate_only=action,
    )
    assert manifest["fresh_initialization"] is True
    assert manifest["storage_alias_free"] is True
    assert len(manifest["combined_sha256"]) == 64

    with pytest.raises(av2.AV2ContractError, match="bytes differ"):
        av2.shared_theta0_manifest(
            reference_shared=reference,
            candidate_shared={"common.weight": candidate["common.weight"] + 1.0},
            candidate_only=action,
        )
    with pytest.raises(av2.AV2ContractError, match="storage aliases"):
        av2.shared_theta0_manifest(
            reference_shared=reference,
            candidate_shared=reference,
            candidate_only=action,
        )

    meta_shared = {"common.weight": torch.empty((3, 4), device="meta")}
    meta_action = {"action.output.weight": torch.empty((3, 4), device="meta")}
    non_cpu_cases = (
        (meta_shared, {"common.weight": meta_shared["common.weight"].clone()}, action),
        (reference, meta_shared, action),
        (reference, candidate, meta_action),
    )
    for non_cpu_reference, non_cpu_candidate, non_cpu_action in non_cpu_cases:
        with pytest.raises(av2.AV2ContractError, match="must stay on CPU"):
            av2.shared_theta0_manifest(
                reference_shared=non_cpu_reference,
                candidate_shared=non_cpu_candidate,
                candidate_only=non_cpu_action,
            )


def test_av2_cpu_a4_seam_preserves_odd_zero_mask_and_backward() -> None:
    stream = a4.StateConditionedCausalOddActionStream(a4.A4BridgeSpec())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026080602)
    actions = torch.randn((8, 5, 20), generator=generator, requires_grad=True)
    common = torch.randn((8, 5, 64), generator=generator, requires_grad=True)
    mask = torch.tensor((False, True, True, True, True), dtype=torch.bool)
    mask = mask.view(1, 5, 1).expand(8, -1, -1)
    positive = stream(actions, common.detach(), mask, count_calls=False)
    negative = stream(-actions, common.detach(), mask, count_calls=False)
    zero = stream(torch.zeros_like(actions), common.detach(), mask, count_calls=False)
    assert torch.equal(positive.odd_input, -negative.odd_input)
    assert torch.equal(positive.write, -negative.write)
    assert torch.equal(positive.recurrent_state, -negative.recurrent_state)
    assert int(positive.delta[:, 0].count_nonzero()) == 0
    assert int(zero.delta.count_nonzero()) == 0
    loss = (positive.delta - torch.ones_like(positive.delta)).square().mean()
    loss.backward()
    assert stream.output_projection.weight.grad is not None
    assert torch.isfinite(stream.output_projection.weight.grad).all()


def test_av2_card_pins_operator_go_without_authorizing_execution() -> None:
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    assert card["state"] == "SCAFFOLD_ONLY_AV2_EXECUTION_NOT_AUTHORIZED"
    assert card["operator_eligibility_mapping"] == {
        "governing_plan_typed_class": "OPERATOR_GO",
        "predecessor_typed_verdict": "OPERATOR_GO_CAUSAL_STREAM_COMMON_STABLE",
        "scope": "ELIGIBILITY_ONLY_NO_AUTOMATIC_AV2_EXECUTION",
    }
    assert card["execution"]["av2_execution_authorized"] is False
    assert card["execution"]["future_execution_card_required"] is True
    assert card["predecessor_a4_r2"]["result"]["sha256"] == _sha256(R2_RESULT)
    assert card["review_token"]["state"] == "CONSUMED_NO_NEW_TOKEN_ALLOWED"
    assert "REAL_DATA" in card["forbidden"]
    assert "GPU_CUDA_TRITON_VENDOR" in card["forbidden"]
