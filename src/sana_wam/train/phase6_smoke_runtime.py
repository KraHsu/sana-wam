"""Production construction adapter for the Phase-6 real-2B smoke runner.

The runner imports this module only after the smoke preflight request/report are
revalidated.  No training config is inferred: the preflight ``input_config`` is
a canonical sidecar which pins five independent canonical arm-runtime configs.
Those configs embed the exact frozen arm projections and a smoke-only
construction contract; current/future preflight or gate hashes are not allowed
inside them, avoiding a provenance cycle.
"""

from __future__ import annotations

from copy import deepcopy
import gc
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import torch

from sana_wam.train.phase6_smoke_gate import (
    RUNTIME_SUPPORT_API_VERSION,
    SMOKE_AUTHORIZATION,
    canonical_json_bytes,
)
from sana_wam.train.phase6_recovery import recovery_lifecycle


SIDECAR_SCHEMA_VERSION = "sana-phase6-real-2b-smoke-runtime-config-v2"
ARM_RUNTIME_CONFIG_SCHEMA_VERSION = "sana-phase6-real-2b-smoke-arm-runtime-config-v2"
SOURCE_REPAIR_AMENDMENT_PATH = Path(
    "/DATA/share/sana_phase6_principled_constraints_20260724/"
    "integration_patches_v4/real_2b_smoke_source_repair_20260724_v2/"
    "source_repair_amendment.json"
)
SOURCE_REPAIR_AMENDMENT_SHA256 = (
    "7d5ac04b3e305f5fc89256cf51abca64b3b64b8577eba0dbbdff81a74592dcf2"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ADAPTER_PREFIX = "action_video_memory_adapter."
_FROZEN_STATE_PREFIXES = (
    "action_backbone.",
    "proprio_encoder.",
    "proprio_video_embed.",
    "proprio_action_embed.",
)
_FROZEN_STATE_EXACT_NAMES = ("action_mean", "action_std")
_ARM_FACTORS = {
    "T0_E0A0": {"A": False, "E": False, "T": False},
    "T1_E0A0": {"A": False, "E": False, "T": True},
    "T1_E1A0": {"A": False, "E": True, "T": True},
    "T1_E0A1": {"A": True, "E": False, "T": True},
    "T1_E1A1": {"A": True, "E": True, "T": True},
}
_RECOVERY_LIFECYCLE = recovery_lifecycle()
_CONSTRUCTION_KEYS = frozenset(
    {
        "batch_size",
        "closed_loop_allowed",
        "gradient_checkpointing",
        "identity_adapter_warm_start",
        "model_dtype",
        "num_workers",
        "optimizer_creation_allowed",
        "optimizer_step_allowed",
        "paired_global_step",
        "primary_checkpoint_allowed",
        "strict_student_checkpoint_load",
        "temporary_checkpoint_only",
        "world_size",
    }
)
_INPUT_ROLE_KEYS = frozenset(
    {
        "action_reference",
        "action_reference_spot_check",
        "action_stats",
        "dataset_contract",
        "phase1_checkpoint",
        "registry",
        "source_manifest",
        "student_checkpoint",
        "task_plan",
    }
)


class Phase6SmokeRuntimeError(RuntimeError):
    """Raised if construction differs from the frozen smoke sidecar."""


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Phase6SmokeRuntimeError(f"{label} must be an object")
    observed = frozenset(value)
    if observed != expected:
        raise Phase6SmokeRuntimeError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Phase6SmokeRuntimeError(f"{label} must be lowercase SHA256")
    return value


def _load_canonical(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise Phase6SmokeRuntimeError(f"{label} must be an absolute regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Phase6SmokeRuntimeError(f"{label} is not regular")
        chunks = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise Phase6SmokeRuntimeError(f"{label} changed while being read")
    data = b"".join(chunks)
    if not hmac.compare_digest(sha256(data).hexdigest(), expected_sha256):
        raise Phase6SmokeRuntimeError(f"{label} SHA256 differs")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase6SmokeRuntimeError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict) or data != canonical_json_bytes(value):
        raise Phase6SmokeRuntimeError(f"{label} must be canonical JSON")
    return value


def _validate_source_repair_amendment() -> None:
    value = _load_canonical(
        SOURCE_REPAIR_AMENDMENT_PATH,
        SOURCE_REPAIR_AMENDMENT_SHA256,
        "source repair amendment",
    )
    if value.get("schema_version") != "sana-phase6-source-repair-amendment-v2":
        raise Phase6SmokeRuntimeError("source repair amendment schema differs")
    backlink = value.get("active_chain_backlink")
    if (
        not isinstance(backlink, dict)
        or backlink.get("required_amendment_path_constant")
        != str(SOURCE_REPAIR_AMENDMENT_PATH)
        or backlink.get("required_runtime_factory_stable_hash_check") is not True
    ):
        raise Phase6SmokeRuntimeError("source repair amendment backlink differs")


def _expected_construction(arm: str) -> dict[str, Any]:
    return {
        "batch_size": 1,
        "closed_loop_allowed": False,
        "gradient_checkpointing": True,
        "identity_adapter_warm_start": _ARM_FACTORS[arm]["A"],
        "model_dtype": "torch.bfloat16",
        "num_workers": 0,
        "optimizer_creation_allowed": False,
        "optimizer_step_allowed": False,
        "paired_global_step": 1,
        "primary_checkpoint_allowed": False,
        "strict_student_checkpoint_load": True,
        "temporary_checkpoint_only": True,
        "world_size": 1,
    }


def _expected_input_roles() -> dict[str, str]:
    return {key: key for key in sorted(_INPUT_ROLE_KEYS)}


def _validate_arm_config(
    value: dict[str, Any],
    *,
    arm: str,
    projection_entry: dict[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        value,
        frozenset(
            {
                "arm",
                "construction_contract",
                "factors",
                "input_roles",
                "projection",
                "projection_sha256",
                *_RECOVERY_LIFECYCLE,
                "schema_version",
            }
        ),
        f"{arm} runtime config",
    )
    if value["schema_version"] != ARM_RUNTIME_CONFIG_SCHEMA_VERSION:
        raise Phase6SmokeRuntimeError(f"{arm} runtime config schema differs")
    if value["arm"] != arm or value["factors"] != _ARM_FACTORS[arm]:
        raise Phase6SmokeRuntimeError(f"{arm} runtime factor identity differs")
    for key, expected in _RECOVERY_LIFECYCLE.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise Phase6SmokeRuntimeError(f"{arm} runtime lifecycle {key} differs")
    projection_sha = _digest(value["projection_sha256"], f"{arm} projection SHA")
    if value["projection"] != projection_entry["projection"]:
        raise Phase6SmokeRuntimeError(f"{arm} embedded projection differs")
    if projection_sha != projection_entry["projection_sha256"]:
        raise Phase6SmokeRuntimeError(f"{arm} projection SHA differs")
    if sha256(canonical_json_bytes(value["projection"])).hexdigest() != projection_sha:
        raise Phase6SmokeRuntimeError(f"{arm} projection content/SHA differs")
    construction = _exact_keys(
        value["construction_contract"],
        _CONSTRUCTION_KEYS,
        f"{arm} construction contract",
    )
    if construction != _expected_construction(arm):
        raise Phase6SmokeRuntimeError(f"{arm} construction contract differs")
    roles = _exact_keys(value["input_roles"], _INPUT_ROLE_KEYS, f"{arm} input roles")
    if roles != _expected_input_roles():
        raise Phase6SmokeRuntimeError(f"{arm} input-role mapping differs")
    return value


def _load_runtime_configs(
    preflight_request: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    input_pin = _exact_keys(
        preflight_request.get("input_config"),
        frozenset({"path", "sha256"}),
        "smoke preflight input_config",
    )
    sidecar = _load_canonical(
        Path(input_pin["path"]),
        _digest(input_pin["sha256"], "sidecar SHA256"),
        "smoke runtime sidecar",
    )
    from sana_wam.train.phase6_arm_config import (
        build_phase6_arm_projection_manifest,
        validate_phase6_smoke_runtime_bundle_bytes,
    )

    validated_sidecar = validate_phase6_smoke_runtime_bundle_bytes(
        canonical_json_bytes(sidecar),
        expected_artifact_sha256=input_pin["sha256"],
        expected_projection_manifest_sha256=preflight_request["artifacts"][
            "arm_projection_manifest"
        ]["sha256"],
        verify_files=True,
    )
    if validated_sidecar != sidecar:
        raise Phase6SmokeRuntimeError("producer/runtime smoke bundle values differ")
    _exact_keys(
        sidecar,
        frozenset(
            {
                "arms",
                "authorization",
                *_RECOVERY_LIFECYCLE,
                "registered_before_recovery_cohort_started",
                "runtime_api_version",
                "schema_version",
            }
        ),
        "smoke runtime sidecar",
    )
    if sidecar["schema_version"] != SIDECAR_SCHEMA_VERSION:
        raise Phase6SmokeRuntimeError("smoke runtime sidecar schema differs")
    if sidecar["runtime_api_version"] != RUNTIME_SUPPORT_API_VERSION:
        raise Phase6SmokeRuntimeError("smoke runtime sidecar API differs")
    if sidecar["authorization"] != SMOKE_AUTHORIZATION:
        raise Phase6SmokeRuntimeError("smoke sidecar authorization differs")
    if sidecar["registered_before_recovery_cohort_started"] is not True:
        raise Phase6SmokeRuntimeError(
            "smoke sidecar was not registered before the recovery cohort"
        )
    for key, expected in _RECOVERY_LIFECYCLE.items():
        if type(sidecar[key]) is not type(expected) or sidecar[key] != expected:
            raise Phase6SmokeRuntimeError(f"smoke sidecar lifecycle {key} differs")

    projection_manifest = build_phase6_arm_projection_manifest()
    projections = {entry["arm"]: entry for entry in projection_manifest["arms"]}
    arm_pins = sidecar["arms"]
    if not isinstance(arm_pins, list) or len(arm_pins) != 5:
        raise Phase6SmokeRuntimeError("smoke sidecar must pin five arm configs")
    configs: dict[str, dict[str, Any]] = {}
    for arm, pin in zip(SMOKE_AUTHORIZATION["arms"], arm_pins):
        _exact_keys(
            pin,
            frozenset({"arm", "path", "projection_sha256", "sha256"}),
            f"{arm} sidecar pin",
        )
        if pin["arm"] != arm:
            raise Phase6SmokeRuntimeError("smoke sidecar arm order differs")
        if pin["projection_sha256"] != projections[arm]["projection_sha256"]:
            raise Phase6SmokeRuntimeError(f"{arm} sidecar projection pin differs")
        config = _load_canonical(
            Path(pin["path"]),
            _digest(pin["sha256"], f"{arm} runtime-config SHA256"),
            f"{arm} runtime config",
        )
        configs[arm] = _validate_arm_config(
            config, arm=arm, projection_entry=projections[arm]
        )
    return configs


def _hydrate_config(
    runtime_config: dict[str, Any],
    *,
    artifact_paths: dict[str, str],
    preflight_request: dict[str, Any],
    ephemeral_directory: str,
) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise Phase6SmokeRuntimeError(
            "OmegaConf is required for real-2B smoke"
        ) from exc

    config = deepcopy(runtime_config["projection"])
    training = config["training"]
    dataloader = config["dataloader"]
    runtime_pins = preflight_request["runtime_files"]
    artifact_pins = preflight_request["artifacts"]
    source_pin = preflight_request["source_manifest"]

    def bind_training(path_key: str, sha_key: str, role: str, *, runtime: bool = False):
        pins = runtime_pins if runtime else artifact_pins
        training[path_key] = artifact_paths[role]
        training[sha_key] = pins[role]["sha256"]

    bind_training("phase6_registry", "phase6_registry_sha256", "registry")
    bind_training("phase6_plan_artifact", "phase6_plan_artifact_sha256", "task_plan")
    bind_training(
        "phase6_dataset_contract_artifact",
        "phase6_dataset_contract_artifact_sha256",
        "dataset_contract",
    )
    training["phase6_code_source_manifest"] = artifact_paths["source_manifest"]
    training["phase6_code_source_manifest_sha256"] = source_pin["sha256"]
    bind_training(
        "init_checkpoint",
        "init_checkpoint_sha256",
        "student_checkpoint",
        runtime=True,
    )
    bind_training(
        "phase1_reference_checkpoint",
        "phase1_reference_checkpoint_sha256",
        "phase1_checkpoint",
        runtime=True,
    )
    training["action_stats_sha256"] = runtime_pins["action_stats"]["sha256"]
    dataloader["action_stats_path"] = artifact_paths["action_stats"]

    if runtime_config["factors"]["A"]:
        bind_training(
            "phase1_action_reference_artifact",
            "phase1_action_reference_artifact_sha256",
            "action_reference",
        )
        training["phase1_action_reference_checkpoint_sha256"] = runtime_pins[
            "phase1_checkpoint"
        ]["sha256"]
        training["phase1_action_reference_live_spot_check_required"] = True
        bind_training(
            "phase1_action_reference_live_spot_check_artifact",
            "phase1_action_reference_live_spot_check_artifact_sha256",
            "action_reference_spot_check",
        )
    else:
        for key in tuple(training):
            if key.startswith("phase1_action_reference_"):
                training.pop(key)

    # These are smoke-only operational sinks.  The 504-step scientific contract
    # remains intact, but no trainer loop or checkpoint schedule can be entered.
    training["output_dir"] = str(Path(ephemeral_directory) / runtime_config["arm"])
    training["num_workers"] = 0
    training["save_initial_checkpoint"] = False
    training["save_steps"] = 0
    training["save_at_steps"] = []
    training["keep_last_k"] = 0
    return OmegaConf.create(config)


class _SmokeTrainerMixin:
    """Accept only the already revalidated smoke preflight pair."""

    def __init__(self, cfg, *, smoke_preflight_report: dict[str, Any]):
        self._authorized_smoke_report = smoke_preflight_report
        super().__init__(cfg)

    def _run_phase6_preflight_early(self) -> None:
        report = self._authorized_smoke_report
        if report.get("purpose") != "smoke" or report.get("status") != "pass":
            raise Phase6SmokeRuntimeError(
                "Trainer construction lacks a passing smoke report"
            )
        if report.get("authorization") != SMOKE_AUTHORIZATION:
            raise Phase6SmokeRuntimeError("Trainer smoke authorization differs")
        payload = canonical_json_bytes(report)
        self._phase6_preflight_report = report
        self._phase6_preflight_report_bytes = payload
        self._phase6_preflight_report_sha256 = sha256(payload).hexdigest()

    def _validate_phase6_launch_context_early(self) -> None:
        """Keep smoke authorization distinct from formal-training authorization."""

        if self._phase6_launch_context is not None:
            raise ValueError("real-2B smoke forbids a formal-training launch_context")
        report = self._phase6_preflight_report
        payload = canonical_json_bytes(report)
        if (
            report is not self._authorized_smoke_report
            or self._phase6_preflight_report_bytes != payload
            or self._phase6_preflight_report_sha256 != sha256(payload).hexdigest()
        ):
            raise ValueError(
                "Trainer smoke preflight authority changed during construction"
            )
        expected_lifecycle = {
            **_RECOVERY_LIFECYCLE,
            "reference_precompute_completed": True,
            "registered_before_recovery_cohort_started": True,
            "smoke_gpu_started": False,
        }
        if any(
            type(report.get(key)) is not type(expected) or report.get(key) != expected
            for key, expected in expected_lifecycle.items()
        ):
            raise ValueError(
                "Trainer smoke preflight lifecycle differs from authorization"
            )


def _build_smoke_trainer(cfg: Any, report: dict[str, Any]) -> Any:
    from sana_wam.train.trainer import Trainer

    class _SmokeTrainer(_SmokeTrainerMixin, Trainer):
        pass

    return _SmokeTrainer(cfg, smoke_preflight_report=report)


class _CaseHandle:
    def __init__(
        self,
        *,
        runtime_config: dict[str, Any],
        artifact_paths: dict[str, str],
        preflight_request: dict[str, Any],
        preflight_report: dict[str, Any],
        paired_fixed_row: dict[str, Any],
        ephemeral_directory: str,
    ):
        self.runtime_config = runtime_config
        self.paired_fixed_row = paired_fixed_row
        cfg = _hydrate_config(
            runtime_config,
            artifact_paths=artifact_paths,
            preflight_request=preflight_request,
            ephemeral_directory=ephemeral_directory,
        )
        self.trainer = _build_smoke_trainer(cfg, preflight_report)
        self.model = self.trainer.architecture
        sampler_indices = list(self.trainer._phase6_plan_sampler)
        global_step = runtime_config["construction_contract"]["paired_global_step"]
        if len(sampler_indices) != 504:
            raise Phase6SmokeRuntimeError(
                "runtime plan sampler is not the exact 504 rows"
            )
        dataset_index = sampler_indices[global_step - 1]
        if dataset_index != paired_fixed_row["dataset_index"]:
            raise Phase6SmokeRuntimeError("runtime paired dataset index differs")
        self.batch = [self.trainer.dataset[dataset_index]]
        self.trainer._set_training_mode()
        self.model_metadata = {
            "full_student_checkpoint_loaded": True,
            "mini_model": False,
            "mock_model": False,
            "model_dtype": "torch.bfloat16",
            "model_factory": "SanaMSVideoCamCtrl_1600M_P1_D20",
            "model_label": "SANA-Video 2B 480p",
            "student_checkpoint_sha256": preflight_request["runtime_files"][
                "student_checkpoint"
            ]["sha256"],
        }

    def video_dit_parameters(self):
        dit = getattr(self.model.video_backbone, "dit", None)
        if not isinstance(dit, torch.nn.Module):
            raise Phase6SmokeRuntimeError(
                "real-2B video backbone lacks its SANA DiT core"
            )
        return dit.parameters()

    def video_trainable_named_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and not name.startswith(_ADAPTER_PREFIX)
        ]

    def adapter_named_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if name.startswith(_ADAPTER_PREFIX)
        ]

    def action_backbone_named_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if name in _FROZEN_STATE_EXACT_NAMES
            or name.startswith(_FROZEN_STATE_PREFIXES)
        ]

    def frozen_action_proprio_state_items(self):
        return {
            name: tensor
            for name, tensor in self.model.state_dict().items()
            if name in _FROZEN_STATE_EXACT_NAMES
            or name.startswith(_FROZEN_STATE_PREFIXES)
        }

    def expected_adapter_state_keys(self):
        adapter = getattr(self.model, "action_video_memory_adapter", None)
        if adapter is None:
            return []
        return [f"{_ADAPTER_PREFIX}{name}" for name in adapter.state_dict()]

    def run_forward(self) -> dict[str, Any]:
        captures: list[tuple[Any, dict[str, Any]]] = []
        boundary_values: list[torch.Tensor] = []
        original_forward = self.model.forward
        t_embedder = getattr(
            getattr(self.model.video_backbone, "dit", None), "t_embedder", None
        )
        if not isinstance(t_embedder, torch.nn.Module):
            raise Phase6SmokeRuntimeError(
                "SANA DiT lacks the timestep embedder boundary"
            )

        def capture_timestep_boundary(_module, inputs):
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                raise Phase6SmokeRuntimeError(
                    "timestep embedder boundary input is malformed"
                )
            boundary_values.append(inputs[0].detach())

        boundary_hook = t_embedder.register_forward_pre_hook(capture_timestep_boundary)

        def capturing_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            captures.append((output, dict(kwargs)))
            return output

        self.model.forward = capturing_forward
        try:
            result = self.trainer._compute_loss(
                self.batch,
                expected_phase6_step=SMOKE_AUTHORIZATION["paired_global_step"],
            )
        finally:
            self.model.forward = original_forward
            boundary_hook.remove()
        expected_forward_count = 2 if self.runtime_config["factors"]["E"] else 1
        if len(captures) != expected_forward_count:
            raise Phase6SmokeRuntimeError(
                f"{self.runtime_config['arm']} forward count differs: {len(captures)}"
            )
        video_prediction, action_prediction = captures[0][0]
        if not isinstance(video_prediction, torch.Tensor) or not isinstance(
            action_prediction, torch.Tensor
        ):
            raise Phase6SmokeRuntimeError(
                "on-path forward did not return both predictions"
            )
        timestep = captures[0][1].get("ar_video_frame_timesteps")
        continuous = self.runtime_config["factors"]["T"]
        if not isinstance(timestep, torch.Tensor):
            raise Phase6SmokeRuntimeError("on-path forward omitted video timesteps")
        if continuous and timestep.dtype is not torch.float32:
            raise Phase6SmokeRuntimeError(
                "T1 did not consume FP32 continuous coordinates"
            )
        if not continuous:
            expected_dtype = getattr(self.model, "_dtype", torch.bfloat16)
            if timestep.dtype is not expected_dtype:
                raise Phase6SmokeRuntimeError("T0 timestep input dtype differs")
        if not boundary_values or any(
            value.dtype is not torch.float32 for value in boundary_values
        ):
            raise Phase6SmokeRuntimeError(
                "real SANA timestep embedder boundary was not observed in FP32"
            )
        surrogate = result.get("phase6_action_non_regression_surrogate")
        if self.runtime_config["factors"]["A"]:
            if not isinstance(surrogate, torch.Tensor) or not surrogate.requires_grad:
                raise Phase6SmokeRuntimeError(
                    "A1 lacks the non-detached NR surrogate smoke seam"
                )
        else:
            surrogate = result["loss"].new_zeros((), dtype=torch.float32)
        return {
            "additional_video_forward_count": len(captures) - 1,
            "common_input_trace_sha256": result["phase6_common_input_trace_sha256"],
            "identity_outputs": {
                "action_prediction": action_prediction,
                "video_prediction": video_prediction,
            },
            "loss": result["loss"],
            "loss_action_non_regression": result["loss_action_non_regression"],
            "loss_video_local_expansion": result["loss_video_local_expansion"],
            "loss_video_on_path": result["loss_video_on_path"],
            "paired_fixed_row": dict(self.paired_fixed_row),
            "phase6_action_non_regression_surrogate": surrogate,
            "timestep_embedder_boundary_values": tuple(boundary_values),
            "video_continuous_conditioning_enabled": bool(
                self.model.video_backbone.continuous_timestep_conditioning
            ),
        }

    def save_full_checkpoint(self, path: str) -> None:
        destination = Path(path)
        if destination.exists():
            raise Phase6SmokeRuntimeError("temporary checkpoint path already exists")
        self.model.save_checkpoint(str(destination))

    def close(self) -> None:
        if getattr(self, "trainer", None) is not None:
            self.trainer.dataset = None
            self.trainer.architecture = None
            self.model = None
            self.batch = None
            self.trainer = None
            gc.collect()


class _Phase6SmokeRuntime:
    def __init__(
        self,
        *,
        preflight_request: dict[str, Any],
        preflight_report: dict[str, Any],
        artifact_paths: dict[str, str],
        device: torch.device,
        ephemeral_directory: str,
    ):
        del device  # Trainer binds LOCAL_RANK to the already selected CUDA device.
        if preflight_request.get("purpose") != "smoke":
            raise Phase6SmokeRuntimeError(
                "runtime received a non-smoke preflight request"
            )
        if preflight_request.get("authorization") != SMOKE_AUTHORIZATION:
            raise Phase6SmokeRuntimeError("runtime preflight authorization differs")
        if (
            preflight_report.get("purpose") != "smoke"
            or preflight_report.get("status") != "pass"
            or preflight_report.get("authorization") != SMOKE_AUTHORIZATION
        ):
            raise Phase6SmokeRuntimeError("runtime received a non-passing smoke report")
        if (
            preflight_report.get("request_sha256")
            != sha256(canonical_json_bytes(preflight_request)).hexdigest()
        ):
            raise Phase6SmokeRuntimeError(
                "runtime preflight request/report binding differs"
            )
        self.preflight_request = preflight_request
        self.preflight_report = preflight_report
        self.artifact_paths = dict(artifact_paths)
        self.ephemeral_directory = ephemeral_directory
        self.configs = _load_runtime_configs(preflight_request)

    @staticmethod
    def checkpoint_keys(path: str):
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            return list(checkpoint.keys())

    def _open(
        self,
        *,
        case_id: str,
        factors: dict[str, bool],
        config_projection: dict[str, Any],
        paired_fixed_row: dict[str, Any],
    ) -> _CaseHandle:
        config = self.configs.get(case_id)
        if config is None:
            raise Phase6SmokeRuntimeError(f"unregistered smoke arm: {case_id}")
        expected_factors = {
            "action_adapter": config["factors"]["A"],
            "continuous_time": config["factors"]["T"],
            "expansion": config["factors"]["E"],
        }
        if factors != expected_factors or config_projection != config["projection"]:
            raise Phase6SmokeRuntimeError(
                f"{case_id} runtime request differs from sidecar"
            )
        return _CaseHandle(
            runtime_config=config,
            artifact_paths=self.artifact_paths,
            preflight_request=self.preflight_request,
            preflight_report=self.preflight_report,
            paired_fixed_row=paired_fixed_row,
            ephemeral_directory=self.ephemeral_directory,
        )

    def open_case(self, **kwargs) -> _CaseHandle:
        return self._open(**kwargs)

    def reload_case_from_full_checkpoint(
        self, *, checkpoint_path: str, strict: bool, **kwargs
    ) -> _CaseHandle:
        if strict is not True:
            raise Phase6SmokeRuntimeError("temporary checkpoint reload must be strict")
        handle = self._open(**kwargs)
        handle.model.load_checkpoint(checkpoint_path, strict=True)
        return handle

    def close(self) -> None:
        self.configs.clear()
        gc.collect()


def create_phase6_smoke_runtime(**kwargs) -> _Phase6SmokeRuntime:
    """Construct the source-manifest-pinned real-2B smoke runtime."""

    _validate_source_repair_amendment()
    return _Phase6SmokeRuntime(**kwargs)


__all__ = [
    "ARM_RUNTIME_CONFIG_SCHEMA_VERSION",
    "Phase6SmokeRuntimeError",
    "RUNTIME_SUPPORT_API_VERSION",
    "SIDECAR_SCHEMA_VERSION",
    "SOURCE_REPAIR_AMENDMENT_PATH",
    "SOURCE_REPAIR_AMENDMENT_SHA256",
    "create_phase6_smoke_runtime",
]
