"""Materialize and statically validate the five frozen Phase-6 arm configs."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from .phase6_downstream_pins import (
    EMBEDDED_PLAN_SHA256,
    FIXED_AMENDMENT_CONFIG_PINS,
    IDENTITY_SHA256,
    TASK_PLAN_ARTIFACT_SHA256,
)
from .phase6_recovery import (
    ALLOWED_PROJECTION_LEAF_CHANGES,
    OPERATIONAL_ROOT,
    PLANROW_RECOVERY_AUTHORITY_PATH,
    PLANROW_RECOVERY_AUTHORITY_SHA256,
    RECOVERY_RUN_DIRECTORIES,
    RECOVERY_RUN_IDS,
    recovery_lifecycle,
    validate_phase6_planrow_recovery_authority,
)


class Phase6ArmConfigError(ValueError):
    """Raised when a template, pin, or materialized arm violates the registry."""


ARM_FACTORS = {
    "T0_E0A0": (False, False, False),
    "T1_E0A0": (True, False, False),
    "T1_E1A0": (True, True, False),
    "T1_E0A1": (True, False, True),
    "T1_E1A1": (True, True, True),
}
ARM_NAMES = tuple(ARM_FACTORS)
TEMPLATE_NAMES = tuple(f"train_phase6_{arm}.yaml.in" for arm in ARM_NAMES)
CONFIG_NAMES = tuple(f"train_phase6_{arm}.yaml" for arm in ARM_NAMES)

ARM_PROJECTION_SCHEMA_VERSION = "sana-phase6-arm-config-projection-v1"
ARM_PROJECTION_MANIFEST_SCHEMA_VERSION = "sana-phase6-arm-config-projection-manifest-v2"
ARM_PROJECTION_EQUIVALENCE_SCHEMA_VERSION = (
    "sana-phase6-arm-projection-scientific-equivalence-v1"
)
ARM_MATERIALIZATION_PINS_SCHEMA_VERSION = "sana-phase6-arm-materialization-pins-v2"
SMOKE_RUNTIME_BUNDLE_SCHEMA_VERSION = "sana-phase6-real-2b-smoke-runtime-config-v2"
SMOKE_RUNTIME_ARM_SCHEMA_VERSION = "sana-phase6-real-2b-smoke-arm-runtime-config-v2"
LEGACY_PRIMARY_RUN_IDS = {arm: f"phase6_20260724_{arm}_primary" for arm in ARM_NAMES}
LEGACY_PRIMARY_RUN_DIRECTORIES = {
    arm: f"{OPERATIONAL_ROOT}/arms/{arm}/{LEGACY_PRIMARY_RUN_IDS[arm]}"
    for arm in ARM_NAMES
}
PRIMARY_RUN_IDS = dict(RECOVERY_RUN_IDS)
PRIMARY_RUN_DIRECTORIES = dict(RECOVERY_RUN_DIRECTORIES)
LEGACY_ARM_PROJECTION_MANIFEST_PATH = (
    f"{OPERATIONAL_ROOT}/artifacts/phase6_arm_projection_manifest_v1.json"
)
LEGACY_ARM_PROJECTION_MANIFEST_SHA256 = (
    "2d8eb8e16cb3dd259aea77d0d54934c3f2fadd34e96d0c162bdd2771dfc4e0e2"
)

DATASET_ARTIFACT_TOKEN = "__PHASE6_DATASET_CONTRACT_ARTIFACT__"
DATASET_SHA_TOKEN = "__PHASE6_DATASET_CONTRACT_SHA256__"
CODE_SOURCE_ARTIFACT_TOKEN = "__PHASE6_CODE_SOURCE_MANIFEST__"
CODE_SOURCE_SHA_TOKEN = "__PHASE6_CODE_SOURCE_MANIFEST_SHA256__"
REFERENCE_ARTIFACT_TOKEN = "__PHASE1_ACTION_REFERENCE_ARTIFACT__"
REFERENCE_SHA_TOKEN = "__PHASE1_ACTION_REFERENCE_SHA256__"
REFERENCE_SPOT_ARTIFACT_TOKEN = "__PHASE1_ACTION_REFERENCE_SPOT_ARTIFACT__"
REFERENCE_SPOT_SHA_TOKEN = "__PHASE1_ACTION_REFERENCE_SPOT_SHA256__"
_COMMON_DYNAMIC_PIN_BASES = {
    "dataset_contract": "phase6_dataset_contract_artifact",
    "source_manifest": "phase6_code_source_manifest",
    "runtime_support_manifest": "phase6_runtime_support_manifest",
    "reference_preflight_request": "phase6_reference_preflight_request",
    "reference_preflight_report": "phase6_reference_preflight_report",
    "reference_rebind_receipt": "phase6_reference_rebind_receipt",
    "action_reference": "phase6_action_reference_provenance_artifact",
    "action_reference_spot_check": "phase6_action_reference_spot_provenance_artifact",
    "arm_projection_manifest": "phase6_arm_projection_manifest",
    "projection_equivalence_receipt": "phase6_scientific_equivalence_receipt",
    "recovery_authority": "phase6_planrow_recovery_authority",
    "smoke_runtime_config": "phase6_smoke_runtime_config",
    "smoke_preflight_request": "phase6_smoke_preflight_request",
    "smoke_preflight_report": "phase6_smoke_preflight_report",
    "smoke_rebind_receipt": "phase6_smoke_rebind_receipt",
    "evidence_rebinding_receipt": "phase6_evidence_rebinding_receipt",
    "real_2b_smoke_gate": "phase6_real_2b_smoke_gate_artifact",
}
_ARM_DYNAMIC_PIN_BASES = {
    "scientific_projection": "phase6_scientific_projection",
    "training_preflight_request": "phase6_preflight_request",
    "training_preflight_report": "phase6_preflight_report",
}
COMMON_DYNAMIC_PIN_FIELDS = tuple(_COMMON_DYNAMIC_PIN_BASES.items())
ARM_DYNAMIC_PIN_FIELDS = tuple(_ARM_DYNAMIC_PIN_BASES.items())
_DYNAMIC_TOKENS = {
    role: (
        f"__{base.upper()}__",
        f"__{base.upper()}_SHA256__",
    )
    for role, base in {**_COMMON_DYNAMIC_PIN_BASES, **_ARM_DYNAMIC_PIN_BASES}.items()
}
_DYNAMIC_TOKENS["dataset_contract"] = (
    DATASET_ARTIFACT_TOKEN,
    DATASET_SHA_TOKEN,
)
_PLACEHOLDER_MARKERS = ("placeholder", "replace_me", "todo", "__phase")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

TRAIN_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_bell",
    "grab_roller",
    "handover_mic",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)
HOLDOUT_TASKS = (
    "click_alarmclock",
    "dump_bin_bigbin",
    "handover_block",
    "hanging_mug",
    "lift_pot",
    "move_stapler_pad",
    "place_cans_plasticbox",
    "rotate_qrcode",
)

VIDEO_PATTERNS = (
    "video_backbone.dit.blocks.*.attn.*",
    "video_backbone.dit.blocks.*.flash_attn_additional.*",
    "video_backbone.dit.blocks.*.mlp.t_conv.*",
)
ADAPTER_PATTERN = "action_video_memory_adapter.*"

KNOWN_TRAINING_PINS = {
    "phase6_protocol_document": (
        f"{OPERATIONAL_ROOT}/PHASE6_PRINCIPLED_ARCHITECTURE_PROTOCOL_20260724.md"
    ),
    "phase6_protocol_document_sha256": (
        "765fcc884cf4cb168fac676de55801667b561fde2cc0cf98a82007ef5469d879"
    ),
    "phase6_registry": f"{OPERATIONAL_ROOT}/phase6_registry_seed20260724.json",
    "phase6_registry_sha256": (
        "9166e1ecde74f14d7d69ae2f5072b10732b01b9f3b0004c438d0e97ed94b00b1"
    ),
    "phase6_operational_storage_amendment": (
        f"{OPERATIONAL_ROOT}/phase6_operational_storage_amendment_20260724.json"
    ),
    "phase6_operational_storage_amendment_sha256": (
        "e08dd7ef33f381a34808cc166e5eae90d7ec20fd9956b481a7bf984b5a8e5c8f"
    ),
    "phase6_external_components_manifest": (
        f"{OPERATIONAL_ROOT}/phase6_external_components_manifest_20260724.json"
    ),
    "phase6_external_components_manifest_sha256": (
        "6f886e00843527ec1933e2f4ebd03dd1e7bf0069e5ce0892bd834b893a6a1b2a"
    ),
    **FIXED_AMENDMENT_CONFIG_PINS,
    "phase6_plan_artifact": f"{OPERATIONAL_ROOT}/artifacts/phase6_task_plan_504.json",
    "phase6_plan_artifact_sha256": TASK_PLAN_ARTIFACT_SHA256,
    "phase6_plan_sha256": EMBEDDED_PLAN_SHA256,
    "phase6_identity_sha256": IDENTITY_SHA256,
    "init_checkpoint": (
        "/home/zch/workspace/sana-wam/logs/sana_principles_audit_20260722/"
        "trajectory_mixed_step0_run/checkpoint_step_0.safetensors"
    ),
    "init_checkpoint_sha256": (
        "e9549aff484da56eada21972f00fb3aa2f95f8a65397a55cf0f38c79e6254137"
    ),
    "action_stats_sha256": (
        "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"
    ),
    "phase1_reference_checkpoint": (
        "/home/zch/workspace/sana-wam/logs/sana_principles_audit_20260722/"
        "phase1_weights_chunkwise_arch_run/checkpoint_step_0.safetensors"
    ),
    "phase1_reference_checkpoint_sha256": (
        "ba7bb59fe7e44f6efd6a4cef79f66aff7ccf5dbe82a5a166634e35055e94f089"
    ),
}

_PROJECTION_TRAINING_KEYS = (
    "phase6_arm",
    *KNOWN_TRAINING_PINS.keys(),
    "phase6_run_id",
    "output_dir",
    "trainable_parameter_patterns",
    "optimizer_master_weights",
    "eval_modules",
    "save_initial_checkpoint",
    "max_steps",
    "save_steps",
    "save_at_steps",
    "keep_last_k",
    "batch_size",
    "gradient_accumulation_steps",
    "num_workers",
    "video_lr",
    "action_lr",
    "weight_decay",
    "grad_clip",
    "warmup_steps",
    "lr_schedule_steps",
    "lambda_video",
    "lambda_action",
    "use_gradient_checkpointing",
    "deepspeed",
    "expected_world_size",
    "seed",
    "debug",
    "phase6_launch_required",
)
REFERENCE_CHECKPOINT_SHA256 = (
    "ba7bb59fe7e44f6efd6a4cef79f66aff7ccf5dbe82a5a166634e35055e94f089"
)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise Phase6ArmConfigError("all config mapping keys must be strings")
        if key in result:
            raise Phase6ArmConfigError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one deterministic, finite JSON artifact with a trailing newline."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise Phase6ArmConfigError(f"value is not canonical JSON: {exc}") from exc


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase6ArmConfigError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(data: bytes, name: str) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Phase6ArmConfigError(f"non-finite JSON value: {value}")
            ),
        )
    except Phase6ArmConfigError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError, ValueError) as exc:
        raise Phase6ArmConfigError(f"invalid {name} JSON: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml_text(text: str, *, source: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise Phase6ArmConfigError(f"{source} must decode as text")
    try:
        value = yaml.load(text, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, UnicodeError) as exc:
        raise Phase6ArmConfigError(f"invalid YAML in {source}: {exc}") from exc
    if type(value) is not dict:
        raise Phase6ArmConfigError(f"{source} must contain one top-level mapping")
    return value


def load_arm_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise Phase6ArmConfigError(f"cannot read config {source}: {exc}") from exc
    return _load_yaml_text(text, source=os.fspath(source))


def load_arm_config_bytes(data: bytes, *, source: str) -> dict[str, Any]:
    """Parse already stable-read YAML bytes without reopening the file."""

    if not isinstance(data, bytes):
        raise TypeError("arm config data must be bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise Phase6ArmConfigError(f"cannot decode config {source}: {exc}") from exc
    return _load_yaml_text(text, source=source)


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Phase6ArmConfigError(f"{name} must be one lowercase SHA256 digest")
    if value == "0" * 64:
        raise Phase6ArmConfigError(f"{name} cannot be an all-zero placeholder")
    return value


def _validate_recovery_lifecycle_fields(value: Mapping[str, Any], name: str) -> None:
    for key, expected in recovery_lifecycle().items():
        observed = value.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise Phase6ArmConfigError(f"{name} {key} differs")
    if "closed_loop_started" in value:
        raise Phase6ArmConfigError(f"{name} uses the legacy closed-loop field")


def _require_artifact_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise Phase6ArmConfigError(f"{name} must be a non-empty exact path")
    lowered = value.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        raise Phase6ArmConfigError(f"{name} still contains a placeholder")
    if not Path(value).is_absolute():
        raise Phase6ArmConfigError(f"{name} must be an absolute artifact path")
    return value


def _verify_file(path: str, expected_sha256: str, name: str) -> None:
    source = Path(path)
    if not source.is_file():
        raise Phase6ArmConfigError(f"{name} is not a file: {source}")
    actual = _sha256_file(source)
    if actual != expected_sha256:
        raise Phase6ArmConfigError(
            f"{name} SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )


def _expected_model(arm: str) -> dict[str, Any]:
    continuous, expansion, adapter = ARM_FACTORS[arm]
    return {
        "architecture": {
            "framework": "dual_system",
            "variant": "autoregressive",
            "action_dim": 20,
            "use_proprioception": True,
            "state_dim": 20,
            "bridge_layers": None,
            "bridge_interval": 1,
            "mot_checkpoint_mixed_attn": True,
            "attention_mask_mode": "joint",
            "video_attention_mask_mode": "first_frame_causal",
            "ar_frame_chunk_size": 2,
            "ar_chunkwise_temporal_ops": True,
            "video_on_path_loss_weight": 1.0,
            "video_trajectory_endpoint_weight": 0.0,
            "video_trajectory_velocity_weight": 0.0,
            "video_trajectory_consistency_weight": 0.0,
            "video_local_expansion_weight": 1.0 if expansion else 0.0,
            "action_non_regression_weight": 1.0 if adapter else 0.0,
            "ar_attn_window": 72,
            "ar_noisy_cond_prob": 0.5,
            "ar_cond_max_ratio": 0.3,
            "proprio_per_chunk": True,
            "ar_bootstrap_clean_prefix": True,
            "proprio_action_dropout_prob": 0.0,
            "action_loss_weighting": "none",
            "action_video_memory_adapter": {
                "enabled": adapter,
                "rank": 8,
                "z_scale_min": 0.5,
                "z_scale_max": 2.0,
                "init_seed": 0,
            },
            "aligned_positive_rope_kernel": {
                "mode": "absolute_rope",
                "beta": 16.0,
                "delta": 1.0e-4,
            },
            "joint_graft": False,
            "joint_graft_idx": None,
            "joint_graft_every_n": 0,
        },
        "video_backbone": {
            "name": "sana_video_2b",
            "model_path": "/DATA/share/SANA-Video_2B_480p",
            "freeze": False,
            "attn_kernel": "linear_relu",
            "text_encoder_name": "/DATA/share/gemma-2-2b-it",
            "flow_shift": 3.0,
            "fp32_attention": False,
            "continuous_timestep_conditioning": continuous,
            "model_kwargs": {
                "additional_flash_attn": "window_flash",
                "flash_attn_window_count": [1, 1, 4],
                "linear_feature_map": "learnable",
                "strictly_positive_feature_map": {
                    "beta": 16.0,
                    "delta": 1.0e-4,
                },
            },
        },
        "action_backbone": {
            "dim": 1152,
            "ffn_dim": 4608,
            "attn_kernel": "linear_relu",
        },
    }


def _expected_dataloader() -> dict[str, Any]:
    return {
        "type": "robotwin",
        "dataset_dir": "/DATA/share/RoboTwin2.0/dataset",
        "task_name": None,
        "robot": "aloha-agilex",
        "variant": "clean_50",
        "action_mode": "eef",
        "num_frames": 113,
        "video_stride": 4,
        "height": 384,
        "width": 320,
        "split": "train",
        "val_ratio": 0.0,
        "repeat": 1,
        "window_stride": 1,
        "multiview": True,
        "camera_layout": ["head_camera", "left_camera", "right_camera"],
        "target_camera": "head_camera",
        "normalize_mode": "min-max",
        "action_stats_path": (
            "/home/zch/workspace/sana-wam/logs/sana_principles_audit_20260722/"
            "trajectory_mixed_step0_run/action_stats.npy"
        ),
        "filter_static_segments": False,
        "static_segment_threshold": 1.0e-5,
        "max_static_retry": 3,
        "train_tasks": list(TRAIN_TASKS),
        "holdout_tasks": list(HOLDOUT_TASKS),
        "text_embedding_cache_dir": None,
        "text_embedding_dropout": 0.0,
        "vae_cache_dir": None,
        "temporal_compression": 4,
        "causal_temporal": True,
        "vae_type": "wan",
        "growing_history": False,
        "history_min_frames": 1,
        "history_stride": 1,
        "gdn_chunk_size": 1,
        "delta_action": False,
        "seed": 42,
        "num_val_samples": 4,
        "backbone": None,
    }


def _expected_training(
    arm: str,
    *,
    common_pins: Mapping[str, tuple[str, str]],
    arm_pins: Mapping[str, tuple[str, str]],
    run_ids: Mapping[str, str] = PRIMARY_RUN_IDS,
    run_directories: Mapping[str, str] = PRIMARY_RUN_DIRECTORIES,
) -> dict[str, Any]:
    adapter = ARM_FACTORS[arm][2]
    value = {
        "phase6_arm": arm,
        **KNOWN_TRAINING_PINS,
        "phase6_run_id": run_ids[arm],
        "output_dir": run_directories[arm],
        "trainable_parameter_patterns": list(VIDEO_PATTERNS),
        "optimizer_master_weights": True,
        "eval_modules": ["video_backbone.vae", "video_backbone.text_encoder"],
        "save_initial_checkpoint": False,
        "max_steps": 504,
        "save_steps": 0,
        "save_at_steps": [504],
        "keep_last_k": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_workers": 4,
        "video_lr": 5.0e-6,
        "action_lr": 1.0e-5,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "warmup_steps": 50,
        "lr_schedule_steps": 504,
        "lambda_video": 1.0,
        "lambda_action": 0.0,
        "use_gradient_checkpointing": True,
        "deepspeed": None,
        "expected_world_size": 1,
        "seed": 20260724,
        "debug": False,
        "phase6_launch_required": True,
    }
    for role, base in _COMMON_DYNAMIC_PIN_BASES.items():
        path, digest = common_pins[role]
        value[base] = path
        value[f"{base}_sha256"] = digest
    for role, base in _ARM_DYNAMIC_PIN_BASES.items():
        path, digest = arm_pins[role]
        value[base] = path
        value[f"{base}_sha256"] = digest
    if adapter:
        value["trainable_parameter_patterns"].append(ADAPTER_PATTERN)
        value.update(
            {
                "action_memory_lr": 1.0e-4,
                "phase1_action_reference_artifact": common_pins["action_reference"][0],
                "phase1_action_reference_artifact_sha256": common_pins[
                    "action_reference"
                ][1],
                "phase1_action_reference_checkpoint_sha256": (
                    REFERENCE_CHECKPOINT_SHA256
                ),
                "phase1_action_reference_live_spot_check_required": True,
                "phase1_action_reference_live_spot_check_artifact": common_pins[
                    "action_reference_spot_check"
                ][0],
                "phase1_action_reference_live_spot_check_artifact_sha256": (
                    common_pins["action_reference_spot_check"][1]
                ),
            }
        )
    return value


def _factor_mapping(arm: str) -> dict[str, bool]:
    try:
        continuous, expansion, adapter = ARM_FACTORS[arm]
    except KeyError as exc:
        raise Phase6ArmConfigError(f"unknown Phase-6 arm: {arm!r}") from exc
    return {"A": adapter, "E": expansion, "T": continuous}


def _build_phase6_arm_projection(
    arm: str,
    *,
    run_ids: Mapping[str, str],
    run_directories: Mapping[str, str],
) -> dict[str, Any]:
    """Build one projection with explicitly selected operational run leaves."""

    if arm not in ARM_FACTORS:
        raise Phase6ArmConfigError(f"unknown Phase-6 arm: {arm!r}")
    common_pins = {
        role: (f"/projection-only/{role}.json", f"{index:x}" * 64)
        for index, role in enumerate(_COMMON_DYNAMIC_PIN_BASES, start=1)
    }
    arm_pins = {
        role: (f"/projection-only/{arm}-{role}.json", f"{index:x}" * 64)
        for index, role in enumerate(_ARM_DYNAMIC_PIN_BASES, start=12)
    }
    training = _expected_training(
        arm,
        common_pins=common_pins,
        arm_pins=arm_pins,
        run_ids=run_ids,
        run_directories=run_directories,
    )
    projected_training = {key: training[key] for key in _PROJECTION_TRAINING_KEYS}
    if ARM_FACTORS[arm][2]:
        for key in (
            "action_memory_lr",
            "phase1_action_reference_checkpoint_sha256",
            "phase1_action_reference_live_spot_check_required",
        ):
            projected_training[key] = training[key]
    return {
        "arm": arm,
        "dataloader": _expected_dataloader(),
        "factors": _factor_mapping(arm),
        "model": _expected_model(arm),
        "schema_version": ARM_PROJECTION_SCHEMA_VERSION,
        "training": projected_training,
    }


def build_phase6_arm_projection(arm: str) -> dict[str, Any]:
    """Return the frozen projection for the fresh plan-row recovery cohort."""

    return _build_phase6_arm_projection(
        arm,
        run_ids=PRIMARY_RUN_IDS,
        run_directories=PRIMARY_RUN_DIRECTORIES,
    )


def build_phase6_legacy_arm_projection(arm: str) -> dict[str, Any]:
    """Reproduce the immutable projection used by the failed v3 cohort."""

    return _build_phase6_arm_projection(
        arm,
        run_ids=LEGACY_PRIMARY_RUN_IDS,
        run_directories=LEGACY_PRIMARY_RUN_DIRECTORIES,
    )


def build_phase6_arm_projection_manifest() -> dict[str, Any]:
    arms = []
    for arm in ARM_NAMES:
        projection = build_phase6_arm_projection(arm)
        projection_sha256 = sha256(canonical_json_bytes(projection)).hexdigest()
        arms.append(
            {
                "arm": arm,
                "factors": _factor_mapping(arm),
                "projection": projection,
                "projection_sha256": projection_sha256,
                "run_directory": PRIMARY_RUN_DIRECTORIES[arm],
                "run_id": PRIMARY_RUN_IDS[arm],
            }
        )
    return {
        "arms": arms,
        **recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "schema_version": ARM_PROJECTION_MANIFEST_SCHEMA_VERSION,
    }


def build_phase6_legacy_arm_projection_manifest() -> dict[str, Any]:
    """Reproduce the registered v1 manifest without treating it as current state."""

    arms = []
    for arm in ARM_NAMES:
        projection = build_phase6_legacy_arm_projection(arm)
        arms.append(
            {
                "arm": arm,
                "factors": _factor_mapping(arm),
                "projection": projection,
                "projection_sha256": sha256(
                    canonical_json_bytes(projection)
                ).hexdigest(),
                "run_directory": LEGACY_PRIMARY_RUN_DIRECTORIES[arm],
                "run_id": LEGACY_PRIMARY_RUN_IDS[arm],
            }
        )
    return {
        "arms": arms,
        "closed_loop_started": False,
        "formal_training_started": False,
        "registered_before_phase6_training_results": True,
        "schema_version": "sana-phase6-arm-config-projection-manifest-v1",
    }


def _projection_leaf_changes(
    before: Any, after: Any, *, prefix: str = ""
) -> list[dict[str, Any]]:
    if type(before) is dict and type(after) is dict:
        if set(before) != set(after):
            raise Phase6ArmConfigError(
                f"projection structure differs outside the operational allowlist: {prefix}"
            )
        changes: list[dict[str, Any]] = []
        for key in sorted(before):
            path = f"{prefix}.{key}" if prefix else key
            changes.extend(
                _projection_leaf_changes(before[key], after[key], prefix=path)
            )
        return changes
    if type(before) is list and type(after) is list:
        if len(before) != len(after):
            raise Phase6ArmConfigError(
                f"projection list length differs outside the allowlist: {prefix}"
            )
        changes = []
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            changes.extend(
                _projection_leaf_changes(old, new, prefix=f"{prefix}[{index}]")
            )
        return changes
    if before == after and type(before) is type(after):
        return []
    return [{"after": after, "before": before, "path": prefix}]


def _scientific_projection_payload(projection: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(canonical_json_bytes(projection))
    for path in ALLOWED_PROJECTION_LEAF_CHANGES:
        parent, leaf = path.split(".", 1)
        del value[parent][leaf]
    return value


def build_phase6_projection_equivalence_receipt() -> dict[str, Any]:
    """Prove that recovery projections change only two operational leaves."""

    legacy_manifest = build_phase6_legacy_arm_projection_manifest()
    legacy_manifest_sha = sha256(canonical_json_bytes(legacy_manifest)).hexdigest()
    if legacy_manifest_sha != LEGACY_ARM_PROJECTION_MANIFEST_SHA256:
        raise Phase6ArmConfigError("legacy projection manifest no longer reproduces")
    arms = []
    for arm in ARM_NAMES:
        before = build_phase6_legacy_arm_projection(arm)
        after = build_phase6_arm_projection(arm)
        changes = _projection_leaf_changes(before, after)
        if [item["path"] for item in changes] != list(ALLOWED_PROJECTION_LEAF_CHANGES):
            raise Phase6ArmConfigError(
                f"{arm} projection delta exceeds the two-leaf operational allowlist"
            )
        before_science = _scientific_projection_payload(before)
        after_science = _scientific_projection_payload(after)
        if before_science != after_science:
            raise Phase6ArmConfigError(f"{arm} scientific projection payload differs")
        arms.append(
            {
                "arm": arm,
                "changes": changes,
                "legacy_projection_sha256": sha256(
                    canonical_json_bytes(before)
                ).hexdigest(),
                "recovery_projection_sha256": sha256(
                    canonical_json_bytes(after)
                ).hexdigest(),
                "scientific_payload_sha256": sha256(
                    canonical_json_bytes(before_science)
                ).hexdigest(),
            }
        )
    return {
        "allowed_projection_leaf_changes": list(ALLOWED_PROJECTION_LEAF_CHANGES),
        "arms": arms,
        **recovery_lifecycle(),
        "legacy_projection_manifest": {
            "path": LEGACY_ARM_PROJECTION_MANIFEST_PATH,
            "sha256": legacy_manifest_sha,
        },
        "registered_before_recovery_cohort_started": True,
        "schema_version": ARM_PROJECTION_EQUIVALENCE_SCHEMA_VERSION,
        "scientific_equivalence": True,
    }


def validate_phase6_projection_equivalence_receipt_bytes(
    data: bytes,
    *,
    expected_artifact_sha256: str,
    verify_legacy_manifest: bool = True,
) -> dict[str, Any]:
    """Validate the exact canonical receipt and optionally its immutable v1 input."""

    expected_sha = _require_digest(
        expected_artifact_sha256, "projection-equivalence receipt SHA256"
    )
    if sha256(data).hexdigest() != expected_sha:
        raise Phase6ArmConfigError("projection-equivalence receipt SHA256 mismatch")
    value = _strict_json_loads(data, "projection-equivalence receipt")
    expected = build_phase6_projection_equivalence_receipt()
    _validate_recovery_lifecycle_fields(value, "projection-equivalence lifecycle")
    if data != canonical_json_bytes(value) or value != expected:
        raise Phase6ArmConfigError("projection-equivalence receipt differs")
    if verify_legacy_manifest:
        path = Path(LEGACY_ARM_PROJECTION_MANIFEST_PATH)
        if path.read_bytes() != canonical_json_bytes(
            build_phase6_legacy_arm_projection_manifest()
        ):
            raise Phase6ArmConfigError("registered legacy projection manifest differs")
    return json.loads(canonical_json_bytes(value))


def validate_phase6_arm_projection_manifest_bytes(
    data: bytes,
    *,
    expected_artifact_sha256: str,
    expected_arm: str | None = None,
    expected_projection_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the exact five-arm projection artifact without importing torch."""

    if not isinstance(data, bytes):
        raise TypeError("arm projection manifest must be bytes")
    expected_artifact = _require_digest(
        expected_artifact_sha256, "arm projection manifest SHA256"
    )
    if sha256(data).hexdigest() != expected_artifact:
        raise Phase6ArmConfigError("arm projection manifest file SHA256 mismatch")
    value = _strict_json_loads(data, "arm projection manifest")
    if not isinstance(value, dict) or data != canonical_json_bytes(value):
        raise Phase6ArmConfigError("arm projection manifest must be canonical JSON")
    expected_value = build_phase6_arm_projection_manifest()
    _validate_recovery_lifecycle_fields(value, "arm projection manifest lifecycle")
    if value != expected_value:
        raise Phase6ArmConfigError("arm projection manifest differs from frozen design")
    if expected_arm is not None:
        if expected_arm not in ARM_FACTORS:
            raise Phase6ArmConfigError(f"unknown expected arm: {expected_arm!r}")
        expected_projection = _require_digest(
            expected_projection_sha256,
            "expected arm config projection SHA256",
        )
        entry = next(item for item in value["arms"] if item["arm"] == expected_arm)
        if entry["projection_sha256"] != expected_projection:
            raise Phase6ArmConfigError(
                f"{expected_arm} config projection SHA256 mismatch"
            )
    elif expected_projection_sha256 is not None:
        raise Phase6ArmConfigError("expected_projection_sha256 requires expected_arm")
    return json.loads(canonical_json_bytes(value))


def write_phase6_arm_projection_manifest(
    path: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> str:
    """Publish a projection manifest once; registered artifacts are immutable."""

    expected = build_phase6_arm_projection_manifest()
    _validate_recovery_lifecycle_fields(
        dict(manifest), "arm projection manifest lifecycle"
    )
    if dict(manifest) != expected:
        raise Phase6ArmConfigError(
            "refusing to publish a non-frozen projection manifest"
        )
    payload = canonical_json_bytes(expected)
    destination = Path(path)
    if not destination.is_absolute() or not destination.parent.is_dir():
        raise Phase6ArmConfigError(
            "projection manifest needs an absolute path with an existing parent"
        )
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {destination}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return sha256(payload).hexdigest()


def write_phase6_arm_projection_bundle(
    manifest_path: str | os.PathLike[str],
    projection_paths: Mapping[str, str | os.PathLike[str]],
) -> dict[str, str]:
    """Exclusively publish the manifest plus five raw canonical projections."""

    if type(projection_paths) is not dict or tuple(projection_paths) != ARM_NAMES:
        raise Phase6ArmConfigError("projection bundle arm path order/set differs")
    destinations = {
        "manifest": Path(manifest_path),
        **{arm: Path(projection_paths[arm]) for arm in ARM_NAMES},
    }
    normalized = []
    for label, destination in destinations.items():
        if not destination.is_absolute() or not destination.parent.is_dir():
            raise Phase6ArmConfigError(
                f"projection bundle {label} needs an absolute path with existing parent"
            )
        normalized.append(os.path.normcase(os.path.abspath(destination)))
    if len(set(normalized)) != len(normalized):
        raise Phase6ArmConfigError("projection bundle destinations must be distinct")
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite projection bundle: {existing}")
    payloads = {
        "manifest": canonical_json_bytes(build_phase6_arm_projection_manifest()),
        **{
            arm: canonical_json_bytes(build_phase6_arm_projection(arm))
            for arm in ARM_NAMES
        },
    }
    created: list[Path] = []
    try:
        for label, destination in destinations.items():
            descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payloads[label])
                stream.flush()
                os.fsync(stream.fileno())
            created.append(destination)
    except BaseException:
        for destination in created:
            destination.unlink()
        raise
    return {label: sha256(payload).hexdigest() for label, payload in payloads.items()}


_SMOKE_RUNTIME_API_VERSION = "sana-phase6-real-2b-smoke-runtime-v1"
_SMOKE_AUTHORIZATION = {
    "operation": "phase6_real_2b_smoke",
    "paired_global_step": 1,
    "arms": list(ARM_NAMES),
    "backward_arm": "T1_E1A1",
    "checkpoint_roundtrip_arm": "T1_E1A1",
    "identity_pair": ["T1_E0A0", "T1_E0A1"],
    "memory_limit_bytes": 130 * 2**30,
    "reference_gpu_forward_completed": True,
    "smoke_gpu_forward_allowed": True,
    "backward_allowed": True,
    "temporary_checkpoint_roundtrip_allowed": True,
    "optimizer_creation_allowed": False,
    "optimizer_step_allowed": False,
    "formal_training_allowed": False,
    "closed_loop_allowed": False,
}
_SMOKE_INPUT_ROLES = (
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


def _smoke_construction_contract(arm: str) -> dict[str, Any]:
    return {
        "batch_size": 1,
        "closed_loop_allowed": False,
        "gradient_checkpointing": True,
        "identity_adapter_warm_start": ARM_FACTORS[arm][2],
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


def build_phase6_smoke_runtime_config(
    arm: str,
    *,
    template_path: str,
    template_sha256: str,
    projection_manifest_sha256: str,
) -> dict[str, Any]:
    """Build one acyclic runtime config containing the exact full projection."""

    # These pins are generation-time guards. Runtime does not depend on templates;
    # it consumes only the embedded projection and the preflight artifact roles.
    _require_artifact_path(template_path, "smoke runtime template path")
    _require_digest(template_sha256, "smoke runtime template SHA256")
    expected_manifest_sha = sha256(
        canonical_json_bytes(build_phase6_arm_projection_manifest())
    ).hexdigest()
    if (
        _require_digest(
            projection_manifest_sha256, "smoke runtime projection-manifest SHA256"
        )
        != expected_manifest_sha
    ):
        raise Phase6ArmConfigError("smoke runtime projection-manifest SHA differs")
    projection = build_phase6_arm_projection(arm)
    return {
        "arm": arm,
        "construction_contract": _smoke_construction_contract(arm),
        "factors": _factor_mapping(arm),
        **recovery_lifecycle(),
        "input_roles": {role: role for role in _SMOKE_INPUT_ROLES},
        "projection": projection,
        "projection_sha256": sha256(canonical_json_bytes(projection)).hexdigest(),
        "schema_version": SMOKE_RUNTIME_ARM_SCHEMA_VERSION,
    }


def _validate_smoke_template(
    arm: str, path: str, digest: str, *, verify_file: bool
) -> None:
    if not verify_file:
        return
    _verify_file(path, digest, f"{arm} smoke base template")
    template = load_arm_config(path)
    projection = build_phase6_arm_projection(arm)
    if template.get("model") != projection["model"]:
        raise Phase6ArmConfigError(f"{arm} smoke template model differs")
    if template.get("dataloader") != projection["dataloader"]:
        raise Phase6ArmConfigError(f"{arm} smoke template dataloader differs")
    training = template.get("training")
    if type(training) is not dict:
        raise Phase6ArmConfigError(f"{arm} smoke template training section is absent")
    for key, expected in projection["training"].items():
        if training.get(key) != expected:
            raise Phase6ArmConfigError(
                f"{arm} smoke template scientific training.{key} differs"
            )


def build_phase6_smoke_runtime_bundle(
    *,
    projection_manifest_path: str,
    projection_manifest_sha256: str,
    template_pins: Mapping[str, Mapping[str, str]],
    runtime_config_pins: Mapping[str, Mapping[str, str]],
    verify_files: bool = True,
) -> dict[str, Any]:
    """Bind five complete runtime configs without request/report/gate cycles."""

    manifest_path = _require_artifact_path(
        projection_manifest_path, "smoke projection-manifest path"
    )
    manifest_digest = _require_digest(
        projection_manifest_sha256, "smoke projection-manifest SHA256"
    )
    expected_manifest_digest = sha256(
        canonical_json_bytes(build_phase6_arm_projection_manifest())
    ).hexdigest()
    if manifest_digest != expected_manifest_digest:
        raise Phase6ArmConfigError("smoke projection-manifest SHA differs")
    if verify_files:
        validate_phase6_arm_projection_manifest_bytes(
            Path(manifest_path).read_bytes(),
            expected_artifact_sha256=manifest_digest,
        )
    if type(template_pins) is not dict or tuple(template_pins) != ARM_NAMES:
        raise Phase6ArmConfigError("smoke template pin arm order/set differs")
    if type(runtime_config_pins) is not dict or tuple(runtime_config_pins) != ARM_NAMES:
        raise Phase6ArmConfigError("smoke runtime-config pin arm order/set differs")
    arms = []
    for arm in ARM_NAMES:
        template_path, template_digest = _validated_pin(
            template_pins[arm], f"smoke {arm} template", verify_file=verify_files
        )
        config_path, config_digest = _validated_pin(
            runtime_config_pins[arm],
            f"smoke {arm} runtime config",
            verify_file=verify_files,
        )
        _validate_smoke_template(
            arm, template_path, template_digest, verify_file=verify_files
        )
        expected_config = build_phase6_smoke_runtime_config(
            arm,
            template_path=template_path,
            template_sha256=template_digest,
            projection_manifest_sha256=manifest_digest,
        )
        expected_bytes = canonical_json_bytes(expected_config)
        if config_digest != sha256(expected_bytes).hexdigest():
            raise Phase6ArmConfigError(f"smoke {arm} runtime-config SHA differs")
        if verify_files and Path(config_path).read_bytes() != expected_bytes:
            raise Phase6ArmConfigError(f"smoke {arm} runtime-config bytes differ")
        arms.append(
            {
                "arm": arm,
                "path": config_path,
                "projection_sha256": expected_config["projection_sha256"],
                "sha256": config_digest,
            }
        )
    return {
        "arms": arms,
        "authorization": json.loads(canonical_json_bytes(_SMOKE_AUTHORIZATION)),
        **recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "runtime_api_version": _SMOKE_RUNTIME_API_VERSION,
        "schema_version": SMOKE_RUNTIME_BUNDLE_SCHEMA_VERSION,
    }


def validate_phase6_smoke_runtime_bundle_bytes(
    data: bytes,
    *,
    expected_artifact_sha256: str,
    expected_projection_manifest_sha256: str,
    verify_files: bool = True,
) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError("smoke runtime bundle must be bytes")
    artifact_digest = _require_digest(
        expected_artifact_sha256, "smoke runtime bundle SHA256"
    )
    if sha256(data).hexdigest() != artifact_digest:
        raise Phase6ArmConfigError("smoke runtime bundle file SHA256 mismatch")
    value = _strict_json_loads(data, "smoke runtime bundle")
    if type(value) is not dict or data != canonical_json_bytes(value):
        raise Phase6ArmConfigError("smoke runtime bundle must be canonical JSON")
    expected_keys = {
        "arms",
        "authorization",
        *recovery_lifecycle(),
        "registered_before_recovery_cohort_started",
        "runtime_api_version",
        "schema_version",
    }
    if set(value) != expected_keys:
        raise Phase6ArmConfigError("smoke runtime bundle keys differ")
    if value["schema_version"] != SMOKE_RUNTIME_BUNDLE_SCHEMA_VERSION:
        raise Phase6ArmConfigError("smoke runtime bundle schema differs")
    if value["runtime_api_version"] != _SMOKE_RUNTIME_API_VERSION:
        raise Phase6ArmConfigError("smoke runtime API version differs")
    if value["authorization"] != _SMOKE_AUTHORIZATION:
        raise Phase6ArmConfigError("smoke runtime authorization differs")
    _validate_recovery_lifecycle_fields(value, "smoke runtime bundle")
    if value["registered_before_recovery_cohort_started"] is not True:
        raise Phase6ArmConfigError(
            "smoke runtime bundle registration lifecycle differs"
        )
    manifest_digest = _require_digest(
        expected_projection_manifest_sha256,
        "expected smoke projection-manifest SHA256",
    )
    if (
        manifest_digest
        != sha256(
            canonical_json_bytes(build_phase6_arm_projection_manifest())
        ).hexdigest()
    ):
        raise Phase6ArmConfigError("smoke runtime projection-manifest pin differs")
    arms = value["arms"]
    if not isinstance(arms, list) or [entry.get("arm") for entry in arms] != list(
        ARM_NAMES
    ):
        raise Phase6ArmConfigError("smoke runtime bundle arm order/set differs")
    for arm, entry in zip(ARM_NAMES, arms, strict=True):
        if type(entry) is not dict or set(entry) != {
            "arm",
            "path",
            "projection_sha256",
            "sha256",
        }:
            raise Phase6ArmConfigError(f"smoke runtime {arm} pin keys differ")
        config_path = _require_artifact_path(
            entry["path"], f"smoke runtime {arm} config path"
        )
        config_digest = _require_digest(
            entry["sha256"], f"smoke runtime {arm} config SHA256"
        )
        expected_projection = build_phase6_arm_projection(arm)
        expected_projection_sha = sha256(
            canonical_json_bytes(expected_projection)
        ).hexdigest()
        if entry["projection_sha256"] != expected_projection_sha:
            raise Phase6ArmConfigError(f"smoke runtime {arm} projection differs")
        if verify_files:
            _verify_file(config_path, config_digest, f"smoke runtime {arm} config")
            config_data = Path(config_path).read_bytes()
            config = _strict_json_loads(config_data, f"smoke runtime {arm} config")
            if config_data != canonical_json_bytes(config):
                raise Phase6ArmConfigError(
                    f"smoke runtime {arm} config must be canonical JSON"
                )
            expected_config = {
                "arm": arm,
                "construction_contract": _smoke_construction_contract(arm),
                "factors": _factor_mapping(arm),
                **recovery_lifecycle(),
                "input_roles": {role: role for role in _SMOKE_INPUT_ROLES},
                "projection": expected_projection,
                "projection_sha256": expected_projection_sha,
                "schema_version": SMOKE_RUNTIME_ARM_SCHEMA_VERSION,
            }
            _validate_recovery_lifecycle_fields(
                config, f"smoke runtime {arm} config lifecycle"
            )
            if config != expected_config:
                raise Phase6ArmConfigError(
                    f"smoke runtime {arm} config differs from frozen contract"
                )
    return json.loads(canonical_json_bytes(value))


def write_phase6_smoke_runtime_bundle(
    *,
    bundle_path: str | os.PathLike[str],
    projection_manifest_path: str,
    projection_manifest_sha256: str,
    template_dir: str | os.PathLike[str],
    runtime_config_paths: Mapping[str, str | os.PathLike[str]],
) -> dict[str, str]:
    """Exclusively publish five runtime configs and their acyclic sidecar."""

    if (
        type(runtime_config_paths) is not dict
        or tuple(runtime_config_paths) != ARM_NAMES
    ):
        raise Phase6ArmConfigError("smoke runtime output arm order/set differs")
    manifest_path = _require_artifact_path(
        projection_manifest_path, "smoke projection-manifest path"
    )
    manifest_digest = _require_digest(
        projection_manifest_sha256, "smoke projection-manifest SHA256"
    )
    validate_phase6_arm_projection_manifest_bytes(
        Path(manifest_path).read_bytes(),
        expected_artifact_sha256=manifest_digest,
    )
    templates = Path(template_dir)
    template_pins = {}
    config_paths = {}
    payloads = {}
    for arm, template_name in zip(ARM_NAMES, TEMPLATE_NAMES, strict=True):
        template = (templates / template_name).resolve()
        template_pins[arm] = {
            "path": os.fspath(template),
            "sha256": _sha256_file(template),
        }
        _validate_smoke_template(
            arm,
            template_pins[arm]["path"],
            template_pins[arm]["sha256"],
            verify_file=True,
        )
        destination = Path(runtime_config_paths[arm])
        if not destination.is_absolute() or not destination.parent.is_dir():
            raise Phase6ArmConfigError(
                f"smoke runtime {arm} output needs absolute path/existing parent"
            )
        config_paths[arm] = destination
        config = build_phase6_smoke_runtime_config(
            arm,
            template_path=template_pins[arm]["path"],
            template_sha256=template_pins[arm]["sha256"],
            projection_manifest_sha256=manifest_digest,
        )
        payloads[arm] = canonical_json_bytes(config)
    runtime_pins = {
        arm: {
            "path": os.fspath(config_paths[arm]),
            "sha256": sha256(payloads[arm]).hexdigest(),
        }
        for arm in ARM_NAMES
    }
    bundle = build_phase6_smoke_runtime_bundle(
        projection_manifest_path=manifest_path,
        projection_manifest_sha256=manifest_digest,
        template_pins=template_pins,
        runtime_config_pins=runtime_pins,
        verify_files=False,
    )
    bundle_destination = Path(bundle_path)
    if not bundle_destination.is_absolute() or not bundle_destination.parent.is_dir():
        raise Phase6ArmConfigError(
            "smoke runtime bundle output needs absolute path/existing parent"
        )
    destinations = {**config_paths, "bundle": bundle_destination}
    if len({os.path.abspath(path) for path in destinations.values()}) != len(
        destinations
    ):
        raise Phase6ArmConfigError("smoke runtime output paths must be distinct")
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite smoke runtime bundle: {existing}")
    all_payloads = {**payloads, "bundle": canonical_json_bytes(bundle)}
    created = []
    try:
        for label, destination in destinations.items():
            descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(all_payloads[label])
                stream.flush()
                os.fsync(stream.fileno())
            created.append(destination)
    except BaseException:
        for destination in created:
            destination.unlink()
        raise
    return {
        label: sha256(payload).hexdigest() for label, payload in all_payloads.items()
    }


def _deep_exact(observed: Any, expected: Any, path: str) -> None:
    if type(observed) is not type(expected):
        raise Phase6ArmConfigError(
            f"{path} type differs: expected {type(expected).__name__}, "
            f"got {type(observed).__name__}"
        )
    if isinstance(expected, dict):
        if set(observed) != set(expected):
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            raise Phase6ArmConfigError(
                f"{path} keys differ: missing={missing}, extra={extra}"
            )
        for key, value in expected.items():
            _deep_exact(observed[key], value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(observed) != len(expected):
            raise Phase6ArmConfigError(
                f"{path} length differs: expected {len(expected)}, got {len(observed)}"
            )
        for index, value in enumerate(expected):
            _deep_exact(observed[index], value, f"{path}[{index}]")
        return
    if observed != expected:
        raise Phase6ArmConfigError(
            f"{path} differs: expected {expected!r}, got {observed!r}"
        )


def validate_arm_config(
    config: Mapping[str, Any], *, verify_files: bool = False
) -> str:
    """Validate one fully materialized config and return its arm name."""
    if type(config) is not dict:
        raise Phase6ArmConfigError("arm config must be one plain dict")
    training = config.get("training")
    if type(training) is not dict:
        raise Phase6ArmConfigError("training must be one plain dict")
    arm = training.get("phase6_arm")
    if arm not in ARM_FACTORS:
        raise Phase6ArmConfigError(f"unknown Phase-6 arm: {arm!r}")

    common_pins: dict[str, tuple[str, str]] = {}
    for role, base in _COMMON_DYNAMIC_PIN_BASES.items():
        common_pins[role] = (
            _require_artifact_path(training.get(base), f"training.{base}"),
            _require_digest(training.get(f"{base}_sha256"), f"training.{base}_sha256"),
        )
    arm_pins: dict[str, tuple[str, str]] = {}
    for role, base in _ARM_DYNAMIC_PIN_BASES.items():
        arm_pins[role] = (
            _require_artifact_path(training.get(base), f"training.{base}"),
            _require_digest(training.get(f"{base}_sha256"), f"training.{base}_sha256"),
        )
    all_paths = [path for path, _digest in (*common_pins.values(), *arm_pins.values())]
    if len(set(all_paths)) != len(all_paths):
        raise Phase6ArmConfigError(
            "dynamic artifact roles must resolve to distinct paths"
        )
    adapter = ARM_FACTORS[arm][2]
    if adapter:
        reference_artifact = _require_artifact_path(
            training.get("phase1_action_reference_artifact"),
            "training.phase1_action_reference_artifact",
        )
        reference_sha256 = _require_digest(
            training.get("phase1_action_reference_artifact_sha256"),
            "training.phase1_action_reference_artifact_sha256",
        )
        spot_artifact = _require_artifact_path(
            training.get("phase1_action_reference_live_spot_check_artifact"),
            "training.phase1_action_reference_live_spot_check_artifact",
        )
        spot_sha256 = _require_digest(
            training.get("phase1_action_reference_live_spot_check_artifact_sha256"),
            "training.phase1_action_reference_live_spot_check_artifact_sha256",
        )
        if (reference_artifact, reference_sha256) != common_pins["action_reference"]:
            raise Phase6ArmConfigError(
                "A1 action-reference loader pin differs from common provenance"
            )
        if (spot_artifact, spot_sha256) != common_pins["action_reference_spot_check"]:
            raise Phase6ArmConfigError(
                "A1 live-spot loader pin differs from common provenance"
            )
    else:
        reference_artifact = reference_sha256 = None

    expected = {
        "model": _expected_model(arm),
        "dataloader": _expected_dataloader(),
        "training": _expected_training(
            arm,
            common_pins=common_pins,
            arm_pins=arm_pins,
        ),
    }
    _deep_exact(config, expected, "config")

    expected_projection_sha256 = sha256(
        canonical_json_bytes(build_phase6_arm_projection(arm))
    ).hexdigest()
    if arm_pins["scientific_projection"][1] != expected_projection_sha256:
        raise Phase6ArmConfigError(
            f"{arm} scientific-projection SHA256 differs from frozen projection"
        )
    expected_manifest_sha256 = sha256(
        canonical_json_bytes(build_phase6_arm_projection_manifest())
    ).hexdigest()
    if common_pins["arm_projection_manifest"][1] != expected_manifest_sha256:
        raise Phase6ArmConfigError(
            "arm-projection manifest SHA256 differs from frozen projection bundle"
        )

    if verify_files:
        file_pins = (
            ("phase6_protocol_document", "phase6_protocol_document_sha256"),
            ("phase6_registry", "phase6_registry_sha256"),
            (
                "phase6_operational_storage_amendment",
                "phase6_operational_storage_amendment_sha256",
            ),
            (
                "phase6_external_components_manifest",
                "phase6_external_components_manifest_sha256",
            ),
            (
                "phase6_expansion_eligibility_amendment",
                "phase6_expansion_eligibility_amendment_sha256",
            ),
            (
                "phase6_padding_semantics_amendment",
                "phase6_padding_semantics_amendment_sha256",
            ),
            ("phase6_plan_artifact", "phase6_plan_artifact_sha256"),
            ("init_checkpoint", "init_checkpoint_sha256"),
            (
                "phase1_reference_checkpoint",
                "phase1_reference_checkpoint_sha256",
            ),
        )
        for path_key, digest_key in file_pins:
            _verify_file(training[path_key], training[digest_key], path_key)
        for role, (path, digest) in {**common_pins, **arm_pins}.items():
            _verify_file(path, digest, role)
        projection_path = Path(arm_pins["scientific_projection"][0])
        if projection_path.read_bytes() != canonical_json_bytes(
            build_phase6_arm_projection(arm)
        ):
            raise Phase6ArmConfigError(
                f"{arm} scientific-projection sidecar bytes differ"
            )
        manifest_path = Path(common_pins["arm_projection_manifest"][0])
        validate_phase6_arm_projection_manifest_bytes(
            manifest_path.read_bytes(),
            expected_artifact_sha256=common_pins["arm_projection_manifest"][1],
            expected_arm=arm,
            expected_projection_sha256=arm_pins["scientific_projection"][1],
        )
        _verify_file(
            config["dataloader"]["action_stats_path"],
            training["action_stats_sha256"],
            "action_stats",
        )
        if adapter:
            _verify_file(
                reference_artifact,
                reference_sha256,
                "phase1_action_reference_artifact",
            )
    return arm


def validate_arm_set(
    configs: Sequence[Mapping[str, Any]], *, verify_files: bool = False
) -> None:
    if isinstance(configs, (str, bytes)) or not isinstance(configs, Sequence):
        raise Phase6ArmConfigError("configs must be a sequence of five mappings")
    by_arm = {}
    for config in configs:
        arm = validate_arm_config(config, verify_files=verify_files)
        if arm in by_arm:
            raise Phase6ArmConfigError(f"duplicate Phase-6 arm: {arm}")
        by_arm[arm] = config
    if set(by_arm) != set(ARM_NAMES):
        raise Phase6ArmConfigError(
            f"arm set differs: expected={list(ARM_NAMES)}, got={sorted(by_arm)}"
        )

    for role, base in _COMMON_DYNAMIC_PIN_BASES.items():
        observed = {
            (
                config["training"][base],
                config["training"][f"{base}_sha256"],
            )
            for config in by_arm.values()
        }
        if len(observed) != 1:
            raise Phase6ArmConfigError(
                f"all five arms must share one {role.replace('_', '-')} pin"
            )
    reference_pins = {
        (
            by_arm[arm]["training"]["phase1_action_reference_artifact"],
            by_arm[arm]["training"]["phase1_action_reference_artifact_sha256"],
        )
        for arm in ARM_NAMES
        if ARM_FACTORS[arm][2]
    }
    if len(reference_pins) != 1:
        raise Phase6ArmConfigError("both A1 arms must share one reference artifact")
    for role, base in _ARM_DYNAMIC_PIN_BASES.items():
        observed = [
            (
                by_arm[arm]["training"][base],
                by_arm[arm]["training"][f"{base}_sha256"],
            )
            for arm in ARM_NAMES
        ]
        if len(set(observed)) != len(observed):
            raise Phase6ArmConfigError(
                f"each arm needs a distinct {role.replace('_', '-')} pin"
            )


def validate_arm_directory(
    config_dir: str | os.PathLike[str], *, verify_files: bool = False
) -> tuple[Path, ...]:
    root = Path(config_dir)
    expected = tuple(root / name for name in CONFIG_NAMES)
    present = tuple(sorted(path.name for path in root.glob("train_phase6_*.yaml")))
    if present != tuple(sorted(CONFIG_NAMES)):
        raise Phase6ArmConfigError(
            f"materialized config filenames differ: expected={CONFIG_NAMES}, got={present}"
        )
    configs = [load_arm_config(path) for path in expected]
    validate_arm_set(configs, verify_files=verify_files)
    return expected


def _validated_pin(value: Any, name: str, *, verify_file: bool) -> tuple[str, str]:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise Phase6ArmConfigError(f"{name} must contain only path and sha256")
    path = _require_artifact_path(value["path"], f"{name}.path")
    digest = _require_digest(value["sha256"], f"{name}.sha256")
    if verify_file:
        _verify_file(path, digest, name)
    return path, digest


def validate_phase6_materialization_pins(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, tuple[str, str]]]]:
    """Validate the acyclic dynamic inputs used to render all final YAML files."""

    expected_keys = {
        "arms",
        "common",
        *recovery_lifecycle(),
        "registered_before_recovery_cohort_started",
        "schema_version",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise Phase6ArmConfigError("materialization-pins top-level keys differ")
    if value["schema_version"] != ARM_MATERIALIZATION_PINS_SCHEMA_VERSION:
        raise Phase6ArmConfigError("materialization-pins schema differs")
    _validate_recovery_lifecycle_fields(value, "materialization-pins")
    if value["registered_before_recovery_cohort_started"] is not True:
        raise Phase6ArmConfigError("materialization-pins registration differs")
    common_value = value["common"]
    if type(common_value) is not dict or set(common_value) != set(
        _COMMON_DYNAMIC_PIN_BASES
    ):
        raise Phase6ArmConfigError("materialization-pins common roles differ")
    common = {
        role: _validated_pin(
            common_value[role],
            f"materialization common.{role}",
            verify_file=verify_files,
        )
        for role in _COMMON_DYNAMIC_PIN_BASES
    }
    arms_value = value["arms"]
    if type(arms_value) is not dict or set(arms_value) != set(ARM_NAMES):
        raise Phase6ArmConfigError("materialization-pins arm order/set differs")
    arms: dict[str, dict[str, tuple[str, str]]] = {}
    for arm in ARM_NAMES:
        arm_value = arms_value[arm]
        if type(arm_value) is not dict or set(arm_value) != set(_ARM_DYNAMIC_PIN_BASES):
            raise Phase6ArmConfigError(f"materialization-pins {arm} roles differ")
        arms[arm] = {
            role: _validated_pin(
                arm_value[role],
                f"materialization arms.{arm}.{role}",
                verify_file=verify_files,
            )
            for role in _ARM_DYNAMIC_PIN_BASES
        }
    paths = [path for path, _digest in common.values()]
    paths.extend(path for arm in ARM_NAMES for path, _digest in arms[arm].values())
    if len(paths) != len(set(paths)):
        raise Phase6ArmConfigError("materialization artifact paths must be role-unique")

    if common["recovery_authority"] != (
        PLANROW_RECOVERY_AUTHORITY_PATH,
        _require_digest(PLANROW_RECOVERY_AUTHORITY_SHA256, "recovery authority SHA256"),
    ):
        raise Phase6ArmConfigError("materialization recovery-authority pin differs")
    expected_equivalence = canonical_json_bytes(
        build_phase6_projection_equivalence_receipt()
    )
    if (
        common["projection_equivalence_receipt"][1]
        != sha256(expected_equivalence).hexdigest()
    ):
        raise Phase6ArmConfigError("materialization equivalence-receipt SHA differs")
    if verify_files:
        try:
            validate_phase6_planrow_recovery_authority()
            validate_phase6_projection_equivalence_receipt_bytes(
                Path(common["projection_equivalence_receipt"][0]).read_bytes(),
                expected_artifact_sha256=common["projection_equivalence_receipt"][1],
            )
        except (OSError, TypeError, ValueError) as exc:
            raise Phase6ArmConfigError(
                f"materialization recovery authority differs: {exc}"
            ) from exc

    expected_manifest = canonical_json_bytes(build_phase6_arm_projection_manifest())
    if common["arm_projection_manifest"][1] != sha256(expected_manifest).hexdigest():
        raise Phase6ArmConfigError("materialization projection-manifest SHA differs")
    if (
        verify_files
        and Path(common["arm_projection_manifest"][0]).read_bytes() != expected_manifest
    ):
        raise Phase6ArmConfigError("materialization projection-manifest bytes differ")
    for arm in ARM_NAMES:
        expected_projection = canonical_json_bytes(build_phase6_arm_projection(arm))
        if (
            arms[arm]["scientific_projection"][1]
            != sha256(expected_projection).hexdigest()
        ):
            raise Phase6ArmConfigError(f"materialization {arm} projection SHA differs")
        if (
            verify_files
            and Path(arms[arm]["scientific_projection"][0]).read_bytes()
            != expected_projection
        ):
            raise Phase6ArmConfigError(f"materialization {arm} projection bytes differ")
    return common, arms


def load_phase6_materialization_pins(
    path: str | os.PathLike[str], *, expected_sha256: str
) -> dict[str, Any]:
    source = Path(path)
    data = source.read_bytes()
    expected = _require_digest(expected_sha256, "materialization-pins SHA256")
    if sha256(data).hexdigest() != expected:
        raise Phase6ArmConfigError("materialization-pins file SHA256 mismatch")
    value = _strict_json_loads(data, "materialization-pins")
    if type(value) is not dict or data != canonical_json_bytes(value):
        raise Phase6ArmConfigError("materialization-pins must be canonical JSON")
    validate_phase6_materialization_pins(value)
    return value


def _replacement_value(value: str) -> str:
    """Escape a value for replacement inside an existing double-quoted scalar."""
    return json.dumps(value, ensure_ascii=True)[1:-1]


def materialize_arm_configs(
    *,
    template_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    materialization_pins: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Verify unknown artifacts, render all five templates, and refuse overwrite."""
    common_pins, arm_pins = validate_phase6_materialization_pins(
        materialization_pins, verify_files=True
    )

    templates = Path(template_dir)
    present = tuple(sorted(path.name for path in templates.glob("*.yaml.in")))
    if present != tuple(sorted(TEMPLATE_NAMES)):
        raise Phase6ArmConfigError(
            f"template filenames differ: expected={TEMPLATE_NAMES}, got={present}"
        )

    rendered: list[tuple[str, str, dict[str, Any]]] = []
    for arm, template_name, config_name in zip(
        ARM_NAMES, TEMPLATE_NAMES, CONFIG_NAMES, strict=True
    ):
        template = (templates / template_name).read_text(encoding="utf-8")
        expected_reference_count = 1 if ARM_FACTORS[arm][2] else 0
        expected_counts = {
            REFERENCE_ARTIFACT_TOKEN: expected_reference_count,
            REFERENCE_SHA_TOKEN: expected_reference_count,
            REFERENCE_SPOT_ARTIFACT_TOKEN: expected_reference_count,
            REFERENCE_SPOT_SHA_TOKEN: expected_reference_count,
        }
        expected_counts.update(
            {token: 1 for pair in _DYNAMIC_TOKENS.values() for token in pair}
        )
        for token, expected_count in expected_counts.items():
            actual_count = template.count(token)
            if actual_count != expected_count:
                raise Phase6ArmConfigError(
                    f"{template_name} token {token} count differs: "
                    f"expected {expected_count}, got {actual_count}"
                )
        replacements = {}
        for role, (path, digest) in common_pins.items():
            path_token, digest_token = _DYNAMIC_TOKENS[role]
            replacements[path_token] = _replacement_value(path)
            replacements[digest_token] = digest
        for role, (path, digest) in arm_pins[arm].items():
            path_token, digest_token = _DYNAMIC_TOKENS[role]
            replacements[path_token] = _replacement_value(path)
            replacements[digest_token] = digest
        replacements[REFERENCE_ARTIFACT_TOKEN] = _replacement_value(
            common_pins["action_reference"][0]
        )
        replacements[REFERENCE_SHA_TOKEN] = common_pins["action_reference"][1]
        replacements[REFERENCE_SPOT_ARTIFACT_TOKEN] = _replacement_value(
            common_pins["action_reference_spot_check"][0]
        )
        replacements[REFERENCE_SPOT_SHA_TOKEN] = common_pins[
            "action_reference_spot_check"
        ][1]
        for token, replacement in replacements.items():
            template = template.replace(token, replacement)
        if "__PHASE" in template:
            raise Phase6ArmConfigError(
                f"{template_name} retained a forbidden placeholder"
            )
        config = _load_yaml_text(template, source=template_name)
        observed_arm = validate_arm_config(config)
        if observed_arm != arm:
            raise Phase6ArmConfigError(
                f"{template_name} rendered arm {observed_arm}, expected {arm}"
            )
        rendered.append((config_name, template, config))
    validate_arm_set([item[2] for item in rendered])

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    targets = tuple(destination / item[0] for item in rendered)
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite materialized configs: {existing}")
    created = []
    try:
        for target, (_, text, _) in zip(targets, rendered, strict=True):
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o444,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            created.append(target)
    except BaseException:
        for target in created:
            target.unlink()
        raise
    return targets


__all__ = [
    "ARM_DYNAMIC_PIN_FIELDS",
    "ARM_MATERIALIZATION_PINS_SCHEMA_VERSION",
    "ARM_NAMES",
    "ARM_PROJECTION_EQUIVALENCE_SCHEMA_VERSION",
    "ARM_PROJECTION_MANIFEST_SCHEMA_VERSION",
    "ARM_PROJECTION_SCHEMA_VERSION",
    "COMMON_DYNAMIC_PIN_FIELDS",
    "CONFIG_NAMES",
    "LEGACY_PRIMARY_RUN_DIRECTORIES",
    "LEGACY_PRIMARY_RUN_IDS",
    "PRIMARY_RUN_DIRECTORIES",
    "PRIMARY_RUN_IDS",
    "SMOKE_RUNTIME_ARM_SCHEMA_VERSION",
    "SMOKE_RUNTIME_BUNDLE_SCHEMA_VERSION",
    "Phase6ArmConfigError",
    "build_phase6_arm_projection",
    "build_phase6_arm_projection_manifest",
    "build_phase6_legacy_arm_projection",
    "build_phase6_legacy_arm_projection_manifest",
    "build_phase6_projection_equivalence_receipt",
    "build_phase6_smoke_runtime_bundle",
    "build_phase6_smoke_runtime_config",
    "canonical_json_bytes",
    "load_arm_config",
    "load_arm_config_bytes",
    "load_phase6_materialization_pins",
    "materialize_arm_configs",
    "validate_arm_config",
    "validate_arm_directory",
    "validate_arm_set",
    "validate_phase6_arm_projection_manifest_bytes",
    "validate_phase6_materialization_pins",
    "validate_phase6_projection_equivalence_receipt_bytes",
    "validate_phase6_smoke_runtime_bundle_bytes",
    "write_phase6_arm_projection_bundle",
    "write_phase6_arm_projection_manifest",
    "write_phase6_smoke_runtime_bundle",
]
