from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT / "src"))

from sana_wam.train import phase6_run_integrity as integrity  # noqa: E402


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _authorization(run_directory: Path, arm: str = "T1_E1A1") -> dict:
    return {
        "arm": arm,
        "arm_config_projection_sha256": "a" * 64,
        "closed_loop_allowed": False,
        "factors": dict(integrity.ARM_FACTORS[arm]),
        "operation": "phase6_training",
        "optimizer_training_allowed": True,
        "real_2b_smoke_completed": True,
        "reference_gpu_forward_completed": True,
        "resume_allowed": False,
        "run_directory": str(run_directory),
        "run_id": run_directory.name,
    }


def test_validated_launch_context_is_exact_and_rehashes_all_inputs(tmp_path: Path):
    config = tmp_path / "arm.yaml"
    manifest = tmp_path / "launch.json"
    ticket = tmp_path / "ticket.json"
    config.write_bytes(b"arm config")
    manifest.write_bytes(b"launch manifest")
    ticket.write_bytes(b"ticket")
    run_directory = tmp_path / "phase6_primary"
    value = {
        "arm": "T1_E1A1",
        "authorization": _authorization(run_directory),
        "config_path": str(config),
        "config_sha256": _digest(config.read_bytes()),
        "launch_manifest_path": str(manifest),
        "launch_manifest_sha256": _digest(manifest.read_bytes()),
        "output_directory": str(run_directory),
        "run_id": run_directory.name,
        "ticket_path": str(ticket),
        "ticket_sha256": _digest(ticket.read_bytes()),
    }
    context = integrity.Phase6ValidatedLaunchContext.from_mapping(value)
    assert context.authorization == value["authorization"]
    context.revalidate_pinned_files()

    ticket.write_bytes(b"changed")
    with pytest.raises(integrity.RunIntegrityError, match="launch ticket changed"):
        context.revalidate_pinned_files()
    with pytest.raises(integrity.RunIntegrityError, match="keys differ"):
        integrity.Phase6ValidatedLaunchContext.from_mapping({**value, "extra": 1})


class _FakeNumpyBytes:
    def __init__(self, data: bytes):
        self._data = data

    def tobytes(self) -> bytes:
        return self._data


class _FakeByteTensor:
    def __init__(self, data: bytes):
        self._data = data

    def numpy(self) -> _FakeNumpyBytes:
        return _FakeNumpyBytes(self._data)


class _FakeTensor:
    dtype = "torch.float32"
    shape = (1,)

    def __init__(self, data: bytes):
        self._data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def reshape(self, _size: int):
        return self

    def view(self, _dtype):
        return _FakeByteTensor(self._data)


def _install_fake_safetensors(
    monkeypatch: pytest.MonkeyPatch, states: dict[str, dict[str, bytes]]
) -> None:
    torch_module = ModuleType("torch")
    torch_module.uint8 = object()
    safetensors_module = ModuleType("safetensors")

    class FakeSafeOpen:
        def __init__(self, path, **_kwargs):
            self._state = states[str(Path(path))]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def keys(self):
            return self._state.keys()

        def get_tensor(self, key):
            return _FakeTensor(self._state[key])

    safetensors_module.safe_open = FakeSafeOpen
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_module)


def test_frozen_builder_reads_both_checkpoints_and_enforces_exact_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    initial = tmp_path / "initial.safetensors"
    final = tmp_path / "checkpoint_step_504.safetensors"
    initial.write_bytes(b"initial container")
    final.write_bytes(b"final container")
    action = {
        f"action_backbone.tensor_{index:03d}": f"action-{index}".encode()
        for index in range(556)
    }
    proprio = {key: f"proprio-{key}".encode() for key in integrity._PROPRIO_STATE_KEYS}
    adapter = {key: f"adapter-{key}".encode() for key in integrity._ADAPTER_STATE_KEYS}
    states = {
        str(initial): {**action, **proprio},
        str(final): {**action, **proprio, **adapter},
    }
    _install_fake_safetensors(monkeypatch, states)
    manifest = integrity.build_frozen_tensor_verification_manifest(
        initial_checkpoint_path=initial,
        step504_checkpoint_path=final,
        initial_checkpoint_sha256=_digest(initial.read_bytes()),
        step504_checkpoint_sha256=_digest(final.read_bytes()),
        arm="T1_E1A1",
    )
    assert manifest["action_backbone"]["tensor_count"] == 556
    assert manifest["action_backbone"]["bitwise_equal"] is True
    assert manifest["proprio"]["tensor_count"] == 6
    assert manifest["action_video_memory_adapter"]["selected_keys"] == (
        integrity._ADAPTER_STATE_KEYS
    )

    states[str(final)]["action_backbone.tensor_000"] = b"changed"
    with pytest.raises(integrity.RunIntegrityError, match="not bitwise identical"):
        integrity.build_frozen_tensor_verification_manifest(
            initial_checkpoint_path=initial,
            step504_checkpoint_path=final,
            initial_checkpoint_sha256=_digest(initial.read_bytes()),
            step504_checkpoint_sha256=_digest(final.read_bytes()),
            arm="T1_E1A1",
        )


@pytest.mark.parametrize(
    ("checkpoint_role", "message"),
    (
        ("initial", "initial checkpoint proprio key set"),
        ("final", "step504 checkpoint proprio key set"),
    ),
)
def test_frozen_builder_rejects_extra_proprio_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_role: str,
    message: str,
):
    initial = tmp_path / "initial.safetensors"
    final = tmp_path / "checkpoint_step_504.safetensors"
    initial.write_bytes(b"initial container")
    final.write_bytes(b"final container")
    action = {
        f"action_backbone.tensor_{index:03d}": f"action-{index}".encode()
        for index in range(556)
    }
    proprio = {key: f"proprio-{key}".encode() for key in integrity._PROPRIO_STATE_KEYS}
    adapter = {key: f"adapter-{key}".encode() for key in integrity._ADAPTER_STATE_KEYS}
    states = {
        str(initial): {**action, **proprio},
        str(final): {**action, **proprio, **adapter},
    }
    target = initial if checkpoint_role == "initial" else final
    states[str(target)]["proprio_encoder.unexpected_state"] = b"unexpected"
    _install_fake_safetensors(monkeypatch, states)

    with pytest.raises(integrity.RunIntegrityError, match=message):
        integrity.build_frozen_tensor_verification_manifest(
            initial_checkpoint_path=initial,
            step504_checkpoint_path=final,
            initial_checkpoint_sha256=_digest(initial.read_bytes()),
            step504_checkpoint_sha256=_digest(final.read_bytes()),
            arm="T1_E1A1",
        )


def test_resolved_config_builder_checks_saved_and_deploy_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    arm = "T1_E0A0"
    projection = {"arm": arm, "factors": dict(integrity.ARM_FACTORS[arm])}
    fake_arm_config = ModuleType("sana_wam.train.phase6_arm_config")
    fake_arm_config.build_phase6_arm_projection = lambda selected: projection
    fake_arm_config.canonical_json_bytes = integrity.canonical_json_bytes
    fake_arm_config.load_arm_config_bytes = (
        lambda data, source: json.loads(data.decode("utf-8"))
    )
    fake_arm_config.validate_arm_config = (
        lambda config, verify_files=False: config["training"]["phase6_arm"]
    )
    monkeypatch.setitem(
        sys.modules, "sana_wam.train.phase6_arm_config", fake_arm_config
    )
    config_value = {"training": {"phase6_arm": arm}}
    payload = json.dumps(config_value, sort_keys=True).encode("utf-8")
    input_config = tmp_path / "input.yaml"
    resolved = tmp_path / "resolved_config.yaml"
    deploy = tmp_path / "config.yaml"
    for path in (input_config, resolved, deploy):
        path.write_bytes(payload)
    projection_sha = _digest(integrity.canonical_json_bytes(projection))
    manifest = integrity.build_resolved_config_verification_manifest(
        arm=arm,
        input_arm_config_path=input_config,
        input_arm_config_sha256=_digest(payload),
        resolved_config_path=resolved,
        deploy_config_path=deploy,
        expected_projection_sha256=projection_sha,
    )
    assert manifest["deploy_config_equivalent"] is True
    assert manifest["frozen_projection_sha256"] == projection_sha

    deploy.write_bytes(b"different")
    with pytest.raises(integrity.RunIntegrityError, match="differs"):
        integrity.build_resolved_config_verification_manifest(
            arm=arm,
            input_arm_config_path=input_config,
            input_arm_config_sha256=_digest(payload),
            resolved_config_path=resolved,
            deploy_config_path=deploy,
            expected_projection_sha256=projection_sha,
        )
