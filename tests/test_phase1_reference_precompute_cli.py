from __future__ import annotations

import builtins
from copy import deepcopy
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from sana_wam.train.action_reference_table import (
    EXPECTED_ACTION_STATS_SHA256,
    EXPECTED_REFERENCE_CHECKPOINT_SHA256,
    PRECOMPUTE_CONFIG_SCHEMA_VERSION,
)
from sana_wam.train.phase6_downstream_pins import (
    DATASET_CONTRACT_ARTIFACT_SHA256,
    FIXED_AMENDMENT_CONFIG_PINS,
    TASK_PLAN_ARTIFACT_SHA256,
)


TEST_DATASET_CONTRACT_SHA256 = DATASET_CONTRACT_ARTIFACT_SHA256


SCRIPT = Path(__file__).parents[1] / "scripts" / "precompute_phase1_action_reference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase1_reference_cli_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forbid_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("invalid precompute input reached a torch import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _config():
    return {
        "schema_version": PRECOMPUTE_CONFIG_SCHEMA_VERSION,
        "model": {
            "architecture": {
                "video_local_expansion_weight": 0.0,
                "action_non_regression_weight": 0.0,
                "action_video_memory_adapter": {"enabled": False},
            },
            "video_backbone": {"continuous_timestep_conditioning": False},
        },
        "dataloader": {
            "type": "robotwin",
            "filter_static_segments": False,
            "growing_history": False,
            "text_embedding_cache_dir": None,
            "text_embedding_dropout": 0.0,
            "vae_cache_dir": None,
        },
        "training": {
            "phase1_action_reference_precompute_mode": True,
            "init_checkpoint_sha256": EXPECTED_REFERENCE_CHECKPOINT_SHA256,
            "phase1_reference_checkpoint_sha256": EXPECTED_REFERENCE_CHECKPOINT_SHA256,
            "action_stats_sha256": EXPECTED_ACTION_STATS_SHA256,
            "phase6_plan_sha256": "8" * 64,
            "phase6_identity_sha256": "9" * 64,
            "phase6_plan_artifact_sha256": TASK_PLAN_ARTIFACT_SHA256,
            "phase6_dataset_contract_artifact_sha256": (
                TEST_DATASET_CONTRACT_SHA256
            ),
            **FIXED_AMENDMENT_CONFIG_PINS,
            "lambda_video": 1.0,
            "lambda_action": 0.0,
            "expected_world_size": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 504,
            "seed": 20260724,
            "num_workers": 0,
        },
        "execution": {
            "autograd_context": "torch.no_grad",
            "autograd_enabled": False,
            "backward_called": False,
            "closed_loop": False,
            "forward_only": True,
            "gradient_checkpointing": False,
            "model_mode": "eval",
            "model_result_key": "phase6_action_unweighted_mse",
            "optimizer_created": False,
            "phase6_plan_rows_required": True,
            "reference_input_key": None,
            "resume_allowed": False,
            "row_context_contract": (
                "plan-row prompt plus dataset-contract logical-length/episode/instruction"
            ),
            "row_noise_contract": "phase6_plan_row.domain_seeds",
            "spot_check_global_steps": [1, 253, 504],
            "training_started": False,
        },
    }


def _cfg(config):
    training = config["training"]
    return SimpleNamespace(
        training=SimpleNamespace(
            init_checkpoint_sha256=training["init_checkpoint_sha256"],
            action_stats_sha256=training["action_stats_sha256"],
            phase6_plan_sha256=training["phase6_plan_sha256"],
            phase6_identity_sha256=training["phase6_identity_sha256"],
            phase6_dataset_contract_artifact_sha256=training[
                "phase6_dataset_contract_artifact_sha256"
            ],
        )
    )


def _args(tmp_path: Path):
    return SimpleNamespace(
        command="build",
        config=tmp_path / "config.yaml",
        config_file_sha256="a" * 64,
        preflight_request=tmp_path / "request.json",
        preflight_request_sha256="c" * 64,
        preflight_report=tmp_path / "report.json",
        source_manifest=tmp_path / "source.json",
        source_manifest_sha256="b" * 64,
        output=tmp_path / "output" / "table.json",
        build_manifest=tmp_path / "output" / "build.json",
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "video_backbone", "continuous_timestep_conditioning"), True),
        (("model", "architecture", "video_local_expansion_weight"), 1.0),
        (("model", "architecture", "action_non_regression_weight"), 1.0),
        (("model", "architecture", "action_video_memory_adapter", "enabled"), True),
        (("training", "lambda_action"), 1.0),
        (("training", "num_workers"), 1),
    ],
)
def test_wrong_reference_config_fails_before_torch_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path, value
):
    module = _load_module()
    args = _args(tmp_path)
    (tmp_path / "output").mkdir()
    config = deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "verify_file", lambda path, expected, name: str(path))
    monkeypatch.setattr(
        module,
        "load_or_run_reference_preflight",
        lambda args: {
            "pins": {
                "observed_source_manifest_sha256": "b" * 64,
                "observed_input_config_sha256": "a" * 64,
            },
            "effective_output_root": {"absolute_path": str(tmp_path / "output")},
            "request_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        module,
        "load_config",
        lambda path, sha: (_cfg(config), config, str(path)),
    )
    _forbid_torch_import(monkeypatch)

    with pytest.raises(ValueError):
        module.main()


def test_wrong_world_size_fails_before_preflight_or_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    monkeypatch.setattr(module, "parse_args", lambda: _args(tmp_path))
    monkeypatch.setenv("WORLD_SIZE", "2")
    _forbid_torch_import(monkeypatch)
    with pytest.raises(RuntimeError, match="WORLD_SIZE=1"):
        module.main()


def test_spot_envelope_uses_content_hashes_not_unregistered_paths():
    module = _load_module()
    config = _config()
    envelope = {
        "precompute_config_file_sha256": "a" * 64,
        "precompute_preflight_request_sha256": "b" * 64,
        "precompute_preflight_report_sha256": "c" * 64,
        "precompute_source_manifest_sha256": "d" * 64,
        "precompute_config": config,
    }
    module.validate_spot_reference_envelope(
        envelope,
        config_sha256="a" * 64,
        preflight_request_sha256="b" * 64,
        preflight_report_sha256="c" * 64,
        source_sha256="d" * 64,
        artifact_config=config,
    )


def test_cuda_device_uuid_uses_libcuda_physical_uuid(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module()
    uuid_bytes = bytes.fromhex("0123456789abcdef0123456789abcdef")
    calls = []

    class FakeCudaDriver:
        def cuInit(self, flags):
            calls.append(("cuInit", flags))
            return 0

        def cuDeviceGet(self, output, device_index):
            calls.append(("cuDeviceGet", device_index))
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = 7
            return 0

        def cuDeviceGetUuid(self, output, device):
            calls.append(("cuDeviceGetUuid", device.value))
            ctypes.memmove(output, uuid_bytes, len(uuid_bytes))
            return 0

    monkeypatch.setattr(module.ctypes, "CDLL", lambda name: FakeCudaDriver())

    assert module._cuda_device_uuid(3) == (
        "GPU-01234567-89ab-cdef-0123-456789abcdef"
    )
    assert calls == [("cuInit", 0), ("cuDeviceGet", 3), ("cuDeviceGetUuid", 7)]


@pytest.mark.parametrize(
    "property_uuid",
    [
        "GPU-fedcba98-7654-3210-fedc-ba9876543210",
        b"GPU-fedcba98-7654-3210-fedc-ba9876543210",
    ],
)
def test_gpu_runtime_descriptor_keeps_valid_property_uuid(
    monkeypatch: pytest.MonkeyPatch, property_uuid
):
    module = _load_module()
    properties = SimpleNamespace(
        uuid=property_uuid,
        multi_processor_count=132,
        total_memory=141_000_000_000,
    )
    cuda = SimpleNamespace(
        get_device_capability=lambda device: (9, 0),
        get_device_properties=lambda device: properties,
        device_count=lambda: 8,
        get_device_name=lambda device: "NVIDIA H200",
    )
    torch = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda="12.8"),
        __version__="2.7.1",
    )
    monkeypatch.setattr(module, "cuda_driver_version", lambda: "12.8")
    monkeypatch.setattr(
        module,
        "_cuda_device_uuid",
        lambda device_index: pytest.fail("valid property UUID must not use fallback"),
    )

    descriptor = module.gpu_runtime_descriptor(torch, SimpleNamespace(index=2))

    assert descriptor["cuda_device_uuid"] == (
        "GPU-fedcba98-7654-3210-fedc-ba9876543210"
    )


@pytest.mark.parametrize("property_uuid", [None, "logical-index-0", b"not-a-uuid"])
def test_gpu_runtime_descriptor_falls_back_when_property_uuid_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, property_uuid
):
    module = _load_module()
    properties = SimpleNamespace(
        multi_processor_count=132,
        total_memory=141_000_000_000,
    )
    if property_uuid is not None:
        properties.uuid = property_uuid
    cuda = SimpleNamespace(
        get_device_capability=lambda device: (9, 0),
        get_device_properties=lambda device: properties,
        device_count=lambda: 8,
        get_device_name=lambda device: "NVIDIA H200",
    )
    torch = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda="12.8"),
        __version__="2.7.1",
    )
    observed_indices = []
    monkeypatch.setattr(module, "cuda_driver_version", lambda: "12.8")
    monkeypatch.setattr(
        module,
        "_cuda_device_uuid",
        lambda device_index: observed_indices.append(device_index)
        or "GPU-01234567-89ab-cdef-0123-456789abcdef",
    )

    descriptor = module.gpu_runtime_descriptor(torch, SimpleNamespace(index=4))

    assert observed_indices == [4]
    assert descriptor["cuda_device_uuid"] == (
        "GPU-01234567-89ab-cdef-0123-456789abcdef"
    )
