from __future__ import annotations

import ast
import copy
from hashlib import sha256
import importlib
import importlib.util
import json
import os
from pathlib import Path
import inspect
import subprocess
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SOURCE_ROOT = ROOT / "src"
sys.path.insert(0, str(CONFIG_SOURCE_ROOT))
try:
    importlib.import_module("yaml")
except ImportError:
    yaml_stub = types.ModuleType("yaml")

    class _SafeLoader:
        @classmethod
        def add_constructor(cls, tag, constructor):
            del cls, tag, constructor

    class _YamlError(Exception):
        pass

    yaml_stub.SafeLoader = _SafeLoader
    yaml_stub.YAMLError = _YamlError
    yaml_stub.load = lambda *args, **kwargs: None
    yaml_stub.resolver = types.SimpleNamespace(
        BaseResolver=types.SimpleNamespace(DEFAULT_MAPPING_TAG="mapping")
    )
    sys.modules["yaml"] = yaml_stub

arm_config = importlib.import_module("sana_wam.train.phase6_arm_config")
PROJECTION_MANIFEST = arm_config.build_phase6_arm_projection_manifest()
PROJECTION_BY_ARM = {
    arm: sha256(
        arm_config.canonical_json_bytes(
            arm_config.build_phase6_legacy_arm_projection(arm)
        )
    ).hexdigest()
    for arm in arm_config.ARM_NAMES
}
MODULE_PATH = ROOT / "src" / "sana_wam" / "train" / "phase6_smoke_gate.py"
RUNNER_PATH = ROOT / "scripts" / "run_phase6_real_2b_smoke.py"
SPEC = importlib.util.spec_from_file_location(
    "phase6_smoke_gate_candidate", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _digest(label: str | bytes) -> str:
    if isinstance(label, str):
        label = label.encode("utf-8")
    return sha256(label).hexdigest()


CASE_FACTORS = {
    "T0_E0A0": (False, False, False),
    "T1_E0A0": (True, False, False),
    "T1_E1A0": (True, True, False),
    "T1_E0A1": (True, False, True),
    "T1_E1A1": (True, True, True),
}
LEGACY_CASE_SHA256 = {
    "T0_E0A0": "4438af45e0534fc419288c7f4e1d4e33520bf4ddda1cdfff9e24633333dcde08",
    "T1_E0A0": "bcfcdee361d385d7376f0b064e7aa5b3ed87869c56bb727f8272322fd3e5c423",
    "T1_E1A0": "e4767fa329b11310d25ff646a280b459dad1f762cf40ff07283a52e2aa773fa3",
    "T1_E0A1": "f136510428b2a7affb58257466ef333358b94d81f15e86afa082df49d854a37a",
    "T1_E1A1": "7acccbbb6690ca3cf8cf538995e111bfc01df9a56b1f1e534e8d49d129a1a9c2",
}


def _bindings() -> dict[str, str]:
    return {
        "action_stats_sha256": _digest("action stats"),
        "arm_projection_manifest_sha256": _digest(
            gate.canonical_json_bytes(PROJECTION_MANIFEST)
        ),
        "dataset_contract_sha256": _digest("dataset contract"),
        "identity_sha256": _digest("identity"),
        "phase1_checkpoint_sha256": _digest("phase1 checkpoint"),
        "plan_sha256": _digest("plan"),
        "preflight_report_sha256": _digest("preflight report"),
        "preflight_request_sha256": _digest("preflight request"),
        "reference_artifact_sha256": _digest("action reference"),
        "runtime_support_manifest_sha256": _digest("runtime support"),
        "smoke_runtime_config_sha256": _digest("smoke runtime config"),
        "source_manifest_sha256": _digest("source manifest"),
        "spot_artifact_sha256": _digest("reference spot"),
        "student_checkpoint_sha256": _digest("student checkpoint"),
    }


def _paired_row() -> dict:
    return {
        "action_sigma": 1.0,
        "common_input_trace_sha256": _digest("common input trace"),
        "dataset_contract_row_sha256": _digest("dataset row"),
        "dataset_index": 83202,
        "global_step": 1,
        "plan_row_sha256": _digest("plan row"),
        "task_name": "move_can_pot",
    }


def _case(case_id: str, factors: tuple[bool, bool, bool]) -> dict:
    continuous, expansion, adapter = factors
    adapter_keys = [
        "action_video_memory_adapter.blocks.0.key_up",
        "action_video_memory_adapter.blocks.0.value_up",
    ]
    key_sha = _digest(gate.canonical_json_bytes(adapter_keys if adapter else []))
    frozen = _digest(f"{case_id} frozen")
    return {
        "action_adapter_gate": {
            "adapter_gradient_connected": adapter,
            "adapter_gradient_finite": adapter,
            "adapter_gradient_zero_allowed": adapter,
            "enabled": adapter,
            "nr_action_backbone_gradient_absent": adapter,
            "nr_autograd_probe_executed": adapter,
            "nr_loss_finite": adapter,
            "nr_video_gradient_absent": adapter,
        },
        "adapter_key_gate": {
            "complete": True,
            "enabled": adapter,
            "expected_key_count": len(adapter_keys) if adapter else 0,
            "expected_key_set_sha256": key_sha,
            "observed_key_count": len(adapter_keys) if adapter else 0,
            "observed_key_set_sha256": key_sha,
            "partial": False,
        },
        "backward_gate": {
            "all_required_gradients_finite": case_id == "T1_E1A1",
            "executed": case_id == "T1_E1A1",
            "full_loss_finite": True,
            "optimizer_step_after_backward": False,
        },
        "batch_size": 1,
        "case_id": case_id,
        "closed_loop_started": False,
        "config_projection_sha256": PROJECTION_BY_ARM[case_id],
        "expansion_gate": {
            "additional_video_forward_count": 1 if expansion else 0,
            "disabled_loss_exact_zero": not expansion,
            "enabled": expansion,
            "loss_finite": True,
            "loss_float32_bits": "3f800000" if expansion else "00000000",
            "on_path_loss_present": True,
        },
        "factors": {
            "action_adapter": adapter,
            "continuous_time": continuous,
            "expansion": expansion,
        },
        "formal_training_started": False,
        "frozen_action_proprio_gate": {
            "after_sha256": frozen,
            "before_sha256": frozen,
            "bitwise_unchanged": True,
            "state_tensor_count": 12,
        },
        "gpu_forward_executed": True,
        "memory_gate": {
            "below_limit": True,
            "limit_bytes": gate.PEAK_RESERVED_LIMIT_BYTES,
            "peak_reserved_bytes": gate.PEAK_RESERVED_LIMIT_BYTES - 1,
            "peak_stats_reset_before_case": True,
        },
        "model_parameter_count": 2_100_001_024 if adapter else 2_100_000_000,
        "optimizer_creation_count": 0,
        "optimizer_step_count": 0,
        "paired_fixed_row": _paired_row(),
        "primary_checkpoint_written": False,
        "timestep_gate": {
            "continuous_coordinate_consumed": continuous,
            "embedder_boundary_dtype": "torch.float32",
            "embedder_boundary_fractional_value_count": 1 if continuous else 0,
            "embedder_boundary_value_count": 10,
            "embedder_boundary_value_sha256": _digest(f"{case_id} timestep boundary"),
            "mode": "continuous_t1" if continuous else "integer_t0",
            "observed_at_t_embedder_boundary": True,
            "old_integer_contract_reproduced": not continuous,
            "t1_long_conversion_detected": False,
        },
        "world_size": 1,
    }


def _artifact() -> dict:
    mode = "registered_five_arms"
    ids = list(CASE_FACTORS)
    factors = {
        case_id: (
            case_id.startswith("T1"),
            "_E1" in case_id,
            case_id.endswith("A1"),
        )
        for case_id in ids
    }
    adapter_keys = [
        "action_video_memory_adapter.blocks.0.key_up",
        "action_video_memory_adapter.blocks.0.value_up",
    ]
    key_sha = _digest(gate.canonical_json_bytes(adapter_keys))
    bindings = _bindings()
    output_sha = _digest("identity output")
    return {
        "adapter_key_contract": {
            "a0_live_key_count": 0,
            "a0_student_checkpoint_key_count": 0,
            "a1_expected_key_count": len(adapter_keys),
            "a1_expected_key_set_sha256": key_sha,
            "a1_live_key_set_sha256": key_sha,
            "a1_temporary_checkpoint_key_set_sha256": key_sha,
            "adapter_prefix": "action_video_memory_adapter.",
            "all_a0_absent": True,
            "all_a1_complete": True,
        },
        "artifact_role": "real_2b_smoke_gate",
        "authorization": copy.deepcopy(gate.SMOKE_AUTHORIZATION),
        "bindings": bindings,
        "cases": [_case(case_id, factors[case_id]) for case_id in ids],
        "completed_at_utc": "2026-07-24T04:05:06Z",
        "coverage": {
            "every_case_gpu_forward": True,
            "every_case_same_paired_row": True,
            "mode": mode,
            "observed_cases": ids,
            "required_cases": ids,
        },
        "e1a1_checkpoint_reload": {
            "adapter_keys_complete": True,
            "backward_completed_before_save": True,
            "case_id": "T1_E1A1",
            "checkpoint_sha256": _digest("temporary checkpoint"),
            "checkpoint_size_bytes": gate.MINIMUM_FULL_CHECKPOINT_SIZE_BYTES,
            "full_checkpoint": True,
            "path_within_ephemeral_root": True,
            "pre_save_state_sha256": _digest("model state"),
            "reloaded_state_sha256": _digest("model state"),
            "state_bitwise_equal": True,
            "state_tensor_count": 2048,
            "strict_reload": True,
            "temporary_checkpoint_deleted": True,
        },
        **gate.recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "gates": {
            "action_proprio_frozen": True,
            "adapter_key_contract": True,
            "adapter_only_nr_connected_finite": True,
            "case_coverage": True,
            "closed_loop_absent": True,
            "e1a1_backward": True,
            "expansion_contract": True,
            "formal_training_absent": True,
            "identity_enabled_bitwise": True,
            "memory_strictly_below_limit": True,
            "optimizer_creation_absent": True,
            "optimizer_step_absent": True,
            "paired_fixed_row": True,
            "primary_checkpoint_absent": True,
            "real_2b_model": True,
            "temporary_checkpoint_deleted": True,
            "temporary_full_checkpoint_reload": True,
            "timestep_contract": True,
        },
        "gpu_smoke_executed": True,
        "host": {
            "cuda_capability": [9, 0],
            "cuda_device_name": "NVIDIA H200",
            "cuda_device_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
            "cuda_driver_version": "550.90",
            "cuda_runtime_version": "12.4",
            "cuda_total_memory_bytes": 141 * 2**30,
            "device_index": 0,
            "hostname": "h200-host",
            "multi_processor_count": 132,
            "torch_version": "2.5.1+cu124",
            "world_size": 1,
        },
        "identity_adapter_bitwise": {
            "adapter_restored_to_exact_identity": True,
            "baseline_case_id": "T1_E0A0",
            "baseline_output_sha256": output_sha,
            "bitwise_equal": True,
            "compared_tensor_count": 2,
            "enabled_case_id": "T1_E0A1",
            "enabled_output_sha256": output_sha,
            "same_prepared_input": True,
        },
        "optimizer_gate": {
            "blockade_installed_before_runtime_factory": True,
            "creation_allowed": False,
            "creation_count": 0,
            "step_allowed": False,
            "step_count": 0,
        },
        "paired_fixed_row": _paired_row(),
        "real_2b_model": {
            "full_student_checkpoint_loaded": True,
            "mini_model": False,
            "mock_model": False,
            "model_dtype": "torch.bfloat16",
            "model_factory": "SanaMSVideoCamCtrl_1600M_P1_D20",
            "model_label": "SANA-Video 2B 480p",
            "student_checkpoint_sha256": bindings["student_checkpoint_sha256"],
            "total_parameter_count": 2_100_000_000,
            "video_dit_parameter_count": 1_600_000_000,
        },
        "schema_version": gate.SMOKE_ARTIFACT_SCHEMA_VERSION,
        "status": "pass",
    }


def _validator_kwargs(value: dict, data: bytes) -> dict[str, str]:
    bindings = value["bindings"]
    return {
        "expected_action_stats_sha256": bindings["action_stats_sha256"],
        "expected_arm_projection_manifest_sha256": bindings[
            "arm_projection_manifest_sha256"
        ],
        "expected_artifact_sha256": _digest(data),
        "expected_dataset_contract_sha256": bindings["dataset_contract_sha256"],
        "expected_identity_sha256": bindings["identity_sha256"],
        "expected_phase1_checkpoint_sha256": bindings["phase1_checkpoint_sha256"],
        "expected_plan_sha256": bindings["plan_sha256"],
        "expected_preflight_report_sha256": bindings["preflight_report_sha256"],
        "expected_preflight_request_sha256": bindings["preflight_request_sha256"],
        "expected_reference_artifact_sha256": bindings["reference_artifact_sha256"],
        "expected_runtime_support_manifest_sha256": bindings[
            "runtime_support_manifest_sha256"
        ],
        "expected_source_manifest_sha256": bindings["source_manifest_sha256"],
        "expected_smoke_runtime_config_sha256": bindings["smoke_runtime_config_sha256"],
        "expected_spot_artifact_sha256": bindings["spot_artifact_sha256"],
        "expected_student_checkpoint_sha256": bindings["student_checkpoint_sha256"],
    }


def _validate(value: dict) -> dict:
    data = gate.canonical_json_bytes(value)
    return gate.validate_phase6_smoke_artifact_bytes(
        data, **_validator_kwargs(value, data)
    )


class SmokeArtifactTests(unittest.TestCase):
    def test_legacy_gpu_case_payloads_remain_byte_identical(self) -> None:
        for arm, factors in CASE_FACTORS.items():
            with self.subTest(arm=arm):
                case = _case(arm, factors)
                self.assertIn("formal_training_started", case)
                self.assertNotIn("historical_formal_training_started", case)
                self.assertNotIn("recovery_cohort_started", case)
                self.assertNotIn("recovery_closed_loop_started", case)
                self.assertEqual(
                    _digest(gate.canonical_json_bytes(case)),
                    LEGACY_CASE_SHA256[arm],
                )

    def test_rebound_top_level_rejects_legacy_global_training_flag(self) -> None:
        value = _artifact()
        value["formal_training_started"] = False
        value["closed_loop_started"] = False
        for key in gate.recovery_lifecycle():
            value.pop(key)
        data = gate.canonical_json_bytes(value)
        with self.assertRaises(gate.Phase6SmokeValidationError):
            gate.validate_phase6_smoke_artifact_bytes(
                data,
                **_validator_kwargs(value, data),
            )

    def test_valid_registered_five_arm_artifact(self) -> None:
        value = _artifact()
        self.assertEqual(_validate(value), value)

    def test_every_external_binding_is_independently_checked(self) -> None:
        value = _artifact()
        for key in sorted(value["bindings"]):
            with self.subTest(key=key):
                mutated = copy.deepcopy(value)
                mutated["bindings"][key] = _digest(f"mutated {key}")
                data = gate.canonical_json_bytes(mutated)
                kwargs = _validator_kwargs(value, data)
                with self.assertRaises(gate.Phase6SmokeValidationError):
                    gate.validate_phase6_smoke_artifact_bytes(data, **kwargs)

    def test_exact_top_level_keys_fail_closed(self) -> None:
        value = _artifact()
        value["unregistered"] = True
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "keys differ"):
            _validate(value)

    def test_authorization_projection_and_row_are_independently_frozen(self) -> None:
        value = _artifact()
        value["authorization"]["paired_global_step"] = 2
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "authorization"):
            _validate(value)
        value = _artifact()
        value["cases"][0]["config_projection_sha256"] = _digest("wrong projection")
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "projection SHA"):
            _validate(value)
        value = _artifact()
        value["paired_fixed_row"]["global_step"] = 2
        for case in value["cases"]:
            case["paired_fixed_row"] = copy.deepcopy(value["paired_fixed_row"])
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "frozen step 1"):
            _validate(value)

    def test_noncanonical_and_duplicate_json_are_rejected(self) -> None:
        value = _artifact()
        pretty = json.dumps(value, indent=2).encode()
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "canonical"):
            gate.validate_phase6_smoke_artifact_bytes(
                pretty, **_validator_kwargs(value, pretty)
            )
        canonical = gate.canonical_json_bytes(value)
        duplicate = canonical.replace(
            b'{"adapter_key_contract":',
            b'{"status":"pass","adapter_key_contract":',
            1,
        )
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "duplicate"):
            gate.validate_phase6_smoke_artifact_bytes(
                duplicate, **_validator_kwargs(value, duplicate)
            )

    def test_missing_case_or_wrong_factor_fails(self) -> None:
        value = _artifact()
        value["cases"].pop()
        with self.assertRaises(gate.Phase6SmokeValidationError):
            _validate(value)
        value = _artifact()
        value["cases"][1]["factors"]["continuous_time"] = False
        with self.assertRaises(gate.Phase6SmokeValidationError):
            _validate(value)

    def test_peak_must_be_strictly_below_130_gib(self) -> None:
        value = _artifact()
        value["cases"][0]["memory_gate"]["peak_reserved_bytes"] = 130 * 2**30
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "reached 130"):
            _validate(value)

    def test_timestep_and_expansion_contracts_fail_closed(self) -> None:
        value = _artifact()
        value["cases"][0]["timestep_gate"]["old_integer_contract_reproduced"] = False
        with self.assertRaises(gate.Phase6SmokeValidationError):
            _validate(value)
        value = _artifact()
        value["cases"][2]["expansion_gate"]["additional_video_forward_count"] = 0
        with self.assertRaises(gate.Phase6SmokeValidationError):
            _validate(value)
        value = _artifact()
        value["cases"][0]["expansion_gate"]["loss_float32_bits"] = "80000000"
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, r"not \+0.0"):
            _validate(value)
        value = _artifact()
        value["cases"][1]["timestep_gate"][
            "embedder_boundary_fractional_value_count"
        ] = 0
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "fractional"):
            _validate(value)

    def test_adapter_nr_and_key_contracts_fail_closed(self) -> None:
        value = _artifact()
        value["cases"][3]["action_adapter_gate"]["adapter_gradient_connected"] = False
        with self.assertRaises(gate.Phase6SmokeValidationError):
            _validate(value)
        value = _artifact()
        value["cases"][3]["adapter_key_gate"]["observed_key_count"] -= 1
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "incomplete"):
            _validate(value)
        value = _artifact()
        value["adapter_key_contract"]["a0_student_checkpoint_key_count"] = 1
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "A0"):
            _validate(value)

    def test_e1a1_backward_and_full_reload_are_mandatory(self) -> None:
        value = _artifact()
        value["cases"][-1]["backward_gate"]["executed"] = False
        with self.assertRaises(gate.Phase6SmokeValidationError):
            _validate(value)
        value = _artifact()
        value["e1a1_checkpoint_reload"]["checkpoint_size_bytes"] -= 1
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "not a full"):
            _validate(value)
        value = _artifact()
        value["e1a1_checkpoint_reload"]["temporary_checkpoint_deleted"] = False
        with self.assertRaises(gate.Phase6SmokeValidationError):
            _validate(value)

    def test_identity_and_frozen_state_are_bitwise_gates(self) -> None:
        value = _artifact()
        value["identity_adapter_bitwise"]["enabled_output_sha256"] = _digest(
            "different"
        )
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "not bitwise"):
            _validate(value)
        value = _artifact()
        value["cases"][0]["frozen_action_proprio_gate"]["after_sha256"] = _digest(
            "changed"
        )
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "changed"):
            _validate(value)

    def test_no_optimizer_primary_training_or_closed_loop_is_admissible(self) -> None:
        mutations = (
            (
                "cases.0.optimizer_step_count",
                lambda v: v["cases"][0].__setitem__("optimizer_step_count", 1),
            ),
            (
                "cases.0.primary",
                lambda v: v["cases"][0].__setitem__("primary_checkpoint_written", True),
            ),
            (
                "historical",
                lambda v: v.__setitem__("historical_formal_training_started", False),
            ),
            (
                "recovery",
                lambda v: v.__setitem__("recovery_cohort_started", True),
            ),
            (
                "registration",
                lambda v: v.__setitem__(
                    "registered_before_recovery_cohort_started", False
                ),
            ),
            (
                "closed",
                lambda v: v.__setitem__("recovery_closed_loop_started", True),
            ),
            (
                "checkpoint-type-alias",
                lambda v: v.__setitem__("historical_checkpoint_count", False),
            ),
            (
                "per-arm-step-type-alias",
                lambda v: v.__setitem__(
                    "historical_optimizer_step_calls_per_arm", True
                ),
            ),
            (
                "lr-scale-type-alias",
                lambda v: v.__setitem__("historical_scheduled_lr_scale_at_step0", 0),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                value = _artifact()
                mutate(value)
                with self.assertRaises(gate.Phase6SmokeValidationError):
                    _validate(value)
        value = _artifact()
        value["optimizer_gate"]["creation_count"] = 1
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "creation_count"):
            _validate(value)

    def test_h200_runtime_descriptor_requires_physical_uuid(self) -> None:
        value = _artifact()
        value["host"]["cuda_device_uuid"] = "logical-index-0"
        with self.assertRaisesRegex(gate.Phase6SmokeValidationError, "GPU UUID"):
            _validate(value)

    def test_public_validator_signature_includes_runtime_bundle_pin(self) -> None:
        signature = inspect.signature(gate.validate_phase6_smoke_artifact_bytes)
        self.assertIn("expected_smoke_runtime_config_sha256", signature.parameters)
        self.assertIs(
            signature.parameters["expected_smoke_runtime_config_sha256"].default,
            inspect.Parameter.empty,
        )
        self.assertEqual(gate.SMOKE_AUTHORIZATION, arm_config._SMOKE_AUTHORIZATION)


class StaticRunnerTests(unittest.TestCase):
    def test_validator_module_has_no_torch_import(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and (
                any(
                    alias.name == "torch" or alias.name.startswith("torch.")
                    for alias in node.names
                )
                if isinstance(node, ast.Import)
                else node.module is not None
                and (node.module == "torch" or node.module.startswith("torch."))
            )
        ]
        self.assertEqual(imports, [])

    def test_runner_delays_torch_import_until_live_execution(self) -> None:
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        torch_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            and any(alias.name == "torch" for alias in node.names)
        ]
        self.assertEqual(len(torch_imports), 1)
        current = parents[torch_imports[0]]
        while not isinstance(current, ast.FunctionDef):
            current = parents[current]
        self.assertEqual(current.name, "_run_smoke")

    def test_runner_never_calls_optimizer_step_or_training_loop(self) -> None:
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        forbidden_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "step",
                "train",
            }:
                forbidden_calls.append(node.func.attr)
        self.assertEqual(forbidden_calls, [])

    def test_runner_consumes_live_nr_surrogate_and_runtime_uses_dit_only(self) -> None:
        runner_source = RUNNER_PATH.read_text(encoding="utf-8")
        runtime_source = (
            ROOT / "src" / "sana_wam" / "train" / "phase6_smoke_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"phase6_action_non_regression_surrogate"', runner_source)
        self.assertIn("def video_dit_parameters", runtime_source)
        self.assertIn('getattr(self.model.video_backbone, "dit", None)', runtime_source)
        self.assertNotIn(
            "return self.model.video_backbone.parameters()", runtime_source
        )
        self.assertIn("register_forward_pre_hook", runtime_source)

    def test_request_shaped_fixture_hydrates_explicit_source_manifest(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_gate = types.ModuleType("sana_wam.train.phase6_smoke_gate")
        fake_gate.RUNTIME_SUPPORT_API_VERSION = gate.RUNTIME_SUPPORT_API_VERSION
        fake_gate.SMOKE_AUTHORIZATION = gate.SMOKE_AUTHORIZATION
        fake_gate.canonical_json_bytes = gate.canonical_json_bytes
        fake_omegaconf = types.ModuleType("omegaconf")

        class _OmegaConf:
            @staticmethod
            def create(value):
                return value

        fake_omegaconf.OmegaConf = _OmegaConf
        runtime_path = ROOT / "src" / "sana_wam" / "train" / "phase6_smoke_runtime.py"
        with mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "omegaconf": fake_omegaconf,
                "sana_wam.train.phase6_smoke_gate": fake_gate,
            },
        ):
            spec = importlib.util.spec_from_file_location(
                "runtime_hydration_probe", runtime_path
            )
            assert spec is not None and spec.loader is not None
            runtime = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runtime)
            projection = next(
                item["projection"]
                for item in PROJECTION_MANIFEST["arms"]
                if item["arm"] == "T1_E1A1"
            )
            runtime_config = {
                "arm": "T1_E1A1",
                "factors": {"A": True, "E": True, "T": True},
                "projection": projection,
            }
            artifact_roles = {
                role: f"C:/pins/{role}"
                for role in (
                    "action_reference",
                    "action_reference_spot_check",
                    "action_stats",
                    "dataset_contract",
                    "phase1_checkpoint",
                    "registry",
                    "source_manifest",
                    "student_checkpoint",
                    "task_plan",
                )
            }
            preflight = {
                "artifacts": {
                    role: {"path": path, "sha256": _digest(role)}
                    for role, path in artifact_roles.items()
                    if role
                    not in {
                        "action_stats",
                        "phase1_checkpoint",
                        "source_manifest",
                        "student_checkpoint",
                    }
                },
                "runtime_files": {
                    role: {"path": artifact_roles[role], "sha256": _digest(role)}
                    for role in (
                        "action_stats",
                        "phase1_checkpoint",
                        "student_checkpoint",
                    )
                },
                "source_manifest": {
                    "path": artifact_roles["source_manifest"],
                    "sha256": _digest("source_manifest"),
                },
            }
            hydrated = runtime._hydrate_config(
                runtime_config,
                artifact_paths=artifact_roles,
                preflight_request=preflight,
                ephemeral_directory="C:/phase6/smoke-tmp/run",
            )
        training = hydrated["training"]
        self.assertEqual(
            training["phase6_code_source_manifest"], artifact_roles["source_manifest"]
        )
        self.assertEqual(
            training["phase6_code_source_manifest_sha256"],
            preflight["source_manifest"]["sha256"],
        )
        self.assertEqual(
            hydrated["dataloader"]["action_stats_path"], artifact_roles["action_stats"]
        )

    def test_help_does_not_require_torch(self) -> None:
        environment = dict(os.environ)
        source_root = str(ROOT / "src")
        environment["PYTHONPATH"] = source_root
        result = subprocess.run(
            [sys.executable, str(RUNNER_PATH), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validate-only", result.stdout)


if __name__ == "__main__":
    unittest.main()
