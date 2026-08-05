"""Strict, torch-free schema validation for the CACH Stage-1 config."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sana_wam.cach.authority import CACH_VARIANT

STAGE1_CONFIG_SCHEMA = "cach-sana-wam-config-stage1-v1"

ACTION_ORDER = (
    "left_xyz_x",
    "left_xyz_y",
    "left_xyz_z",
    "left_rot6d_0",
    "left_rot6d_1",
    "left_rot6d_2",
    "left_rot6d_3",
    "left_rot6d_4",
    "left_rot6d_5",
    "left_gripper",
    "right_xyz_x",
    "right_xyz_y",
    "right_xyz_z",
    "right_rot6d_0",
    "right_rot6d_1",
    "right_rot6d_2",
    "right_rot6d_3",
    "right_rot6d_4",
    "right_rot6d_5",
    "right_gripper",
)


class CACHConfigError(ValueError):
    """A config violates the immutable CACH Stage-1 contract."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        try:
            value = {key: value[key] for key in value.keys()}
        except (AttributeError, KeyError, TypeError) as exc:
            raise CACHConfigError(
                "CONFIG_TYPE_MISMATCH", f"{name} must be a mapping"
            ) from exc
    if not all(type(key) is str for key in value):
        raise CACHConfigError(
            "CONFIG_TYPE_MISMATCH", f"{name} keys must be strings"
        )
    return value


def _exact_keys(
    value: Any,
    expected: set[str],
    name: str,
) -> Mapping[str, Any]:
    mapping = _mapping(value, name)
    if set(mapping) != expected:
        raise CACHConfigError(
            "CONFIG_KEY_SET_MISMATCH",
            f"{name} keys differ: expected={sorted(expected)}, got={sorted(mapping)}",
        )
    return mapping


def _literal(observed: Any, expected: Any, name: str) -> None:
    if type(observed) is not type(expected):
        raise CACHConfigError(
            "CONFIG_LITERAL_MISMATCH",
            f"{name} must be exactly {expected!r}, got {observed!r}",
        )
    if type(expected) is dict:
        if set(observed) != set(expected):
            raise CACHConfigError(
                "CONFIG_LITERAL_MISMATCH",
                f"{name} must be exactly {expected!r}, got {observed!r}",
            )
        for key, literal in expected.items():
            _literal(observed[key], literal, f"{name}.{key}")
        return
    if type(expected) is list:
        if len(observed) != len(expected):
            raise CACHConfigError(
                "CONFIG_LITERAL_MISMATCH",
                f"{name} must be exactly {expected!r}, got {observed!r}",
            )
        for index, literal in enumerate(expected):
            _literal(observed[index], literal, f"{name}[{index}]")
        return
    if observed != expected:
        raise CACHConfigError(
            "CONFIG_LITERAL_MISMATCH",
            f"{name} must be exactly {expected!r}, got {observed!r}",
        )


def _reject_forbidden_tree(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if "afcc" in normalized or "phase6" in normalized:
                raise CACHConfigError(
                    "CACH_AFCC_ISOLATION_VIOLATION",
                    f"{path}.{key} is forbidden in CACH",
                )
            if (
                "action_reference" in normalized
                or "action_facing_cache_reference" in normalized
            ):
                raise CACHConfigError(
                    "CACH_AFCC_ISOLATION_VIOLATION",
                    f"{path}.{key} is forbidden in CACH",
                )
            if key == "F":
                if type(child) is not int or child != 0:
                    raise CACHConfigError(
                        "CACH_AFCC_ISOLATION_VIOLATION",
                        f"{path}.F must be the integer 0",
                    )
            _reject_forbidden_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_tree(child, f"{path}[{index}]")


def validate_stage1_config(value: Any) -> Mapping[str, Any]:
    """Validate the complete implementation-only Stage-1 YAML mapping.

    The config is intentionally non-executable.  It closes dangerous legacy
    defaults while retaining null runtime/data coordinates as explicit
    blockers.
    """

    root = _exact_keys(
        value,
        {
            "admission",
            "cach_stage0",
            "cach_stage1",
            "dataloader",
            "model",
            "training",
        },
        "config",
    )
    _reject_forbidden_tree(root)

    _literal(
        root["cach_stage0"],
        {"compatibility_denial_latch": "cach_stage1_nonexecutable_v1"},
        "cach_stage0",
    )

    marker = _exact_keys(
        root["cach_stage1"],
        {
            "allowed_actions",
            "execution_enabled",
            "schema_version",
            "scientific_eligible",
            "status",
        },
        "cach_stage1",
    )
    for field, expected in {
        "schema_version": STAGE1_CONFIG_SCHEMA,
        "status": "implementation_only",
        "execution_enabled": False,
        "scientific_eligible": False,
        "allowed_actions": ["static_review", "pure_contract_tests"],
    }.items():
        _literal(marker[field], expected, f"cach_stage1.{field}")

    model = _exact_keys(
        root["model"],
        {"action_backbone", "architecture", "video_backbone"},
        "model",
    )
    architecture = _exact_keys(
        model["architecture"],
        {
            "action_condition",
            "action_dim",
            "block_attn_res",
            "causal_softmax_anchors",
            "episode_bootstrap",
            "frame_chunk_size",
            "framework",
            "hybrid_cache",
            "initialization_mode",
            "observed_prefix_chunks",
            "proprio_boundary",
            "proprio_per_chunk",
            "proprio_value_binding",
            "self_forcing",
            "state_dim",
            "use_proprioception",
            "variant",
        },
        "model.architecture",
    )
    for field, expected in {
        "framework": "dual_system",
        "variant": CACH_VARIANT,
        "action_dim": 20,
        "state_dim": 20,
        "use_proprioception": True,
        "proprio_per_chunk": True,
        "proprio_boundary": "layout_committed_boundary_v1",
        "frame_chunk_size": 3,
        "proprio_value_binding": "semantic_float32_sha256_v1",
        "episode_bootstrap": "first_frame_pinned",
        "observed_prefix_chunks": 0,
        "initialization_mode": "complete_random_v1",
    }.items():
        _literal(architecture[field], expected, f"model.architecture.{field}")

    action_condition = _exact_keys(
        architecture["action_condition"],
        {
            "enabled",
            "action_input_slots",
            "condition_shape",
            "initialization",
            "input_dim",
            "internal_vendor_seam",
            "padded_condition_value",
            "partial_tail_mask",
            "public_name",
            "reducer",
        },
        "model.architecture.action_condition",
    )
    for field, expected in {
        "enabled": True,
        "action_input_slots": "fixed_chunk_capacity_with_prefix_mask_v1",
        "condition_shape": "fixed_b_k_a_v1",
        "initialization": "post_factory_exact_zero_v1",
        "input_dim": 20,
        "internal_vendor_seam": "use_delta_pose_additive",
        "padded_condition_value": "exact_zero",
        "partial_tail_mask": "explicit_boolean_b_k_v1",
        "public_name": "action_condition",
        "reducer": "end_of_latent_bin_command_v1",
    }.items():
        _literal(
            action_condition[field],
            expected,
            f"model.architecture.action_condition.{field}",
        )

    anchors = _exact_keys(
        architecture["causal_softmax_anchors"],
        {"enabled", "indices"},
        "model.architecture.causal_softmax_anchors",
    )
    _literal(anchors["enabled"], False, "causal_softmax_anchors.enabled")
    _literal(anchors["indices"], [], "causal_softmax_anchors.indices")
    _literal(
        architecture["block_attn_res"],
        {"enabled": False},
        "model.architecture.block_attn_res",
    )
    _literal(
        architecture["self_forcing"],
        {"enabled": False},
        "model.architecture.self_forcing",
    )
    cache = _exact_keys(
        architecture["hybrid_cache"],
        {
            "attnres_depth_state",
            "deploy_applied_ack_enabled",
            "expected_gdn_layer_count",
            "expected_softmax_layer_count",
            "gdn_temporal_state",
            "layout_binding",
            "offline_commit_source",
            "self_forcing_generated_pair_enabled",
            "softmax_temporal_state",
        },
        "model.architecture.hybrid_cache",
    )
    for field, expected in {
        "attnres_depth_state": "forward_local",
        "deploy_applied_ack_enabled": False,
        "expected_gdn_layer_count": 20,
        "expected_softmax_layer_count": 0,
        "gdn_temporal_state": "full_history",
        "layout_binding": "exact_chunk_action_layout_instance_v1",
        "offline_commit_source": "teacher_forcing_dataset_pair",
        "self_forcing_generated_pair_enabled": False,
        "softmax_temporal_state": "not_instantiated_in_cach_a",
    }.items():
        _literal(cache[field], expected, f"model.architecture.hybrid_cache.{field}")

    video = _exact_keys(
        model["video_backbone"],
        {
            "attn_kernel",
            "chunk_size",
            "delta_pose_additive_dim",
            "flow_shift",
            "fp32_attention",
            "freeze",
            "gdn_streaming",
            "init_dit_from",
            "model_path",
            "name",
            "text_encoder_name",
            "use_delta_pose_additive",
            "use_first_frame_cond",
            "vae_path",
            "vae_type",
        },
        "model.video_backbone",
    )
    for field, expected in {
        "name": "sana_video_2b",
        "model_path": None,
        "init_dit_from": None,
        "vae_type": "ltx2",
        "vae_path": "/DATA/share/SANA-WM_streaming/ltx2_causal_vae",
        "text_encoder_name": "/DATA/share/gemma-2-2b-it",
        "freeze": False,
        "attn_kernel": "gdn",
        "chunk_size": 3,
        "fp32_attention": True,
        "use_first_frame_cond": True,
        "gdn_streaming": True,
        "use_delta_pose_additive": True,
        "delta_pose_additive_dim": 20,
        "flow_shift": None,
    }.items():
        _literal(video[field], expected, f"model.video_backbone.{field}")

    action_backbone = _exact_keys(
        model["action_backbone"],
        {"dim", "ffn_dim"},
        "model.action_backbone",
    )
    _literal(action_backbone["dim"], 1152, "model.action_backbone.dim")
    _literal(action_backbone["ffn_dim"], 4608, "model.action_backbone.ffn_dim")

    dataloader = _exact_keys(
        root["dataloader"],
        {
            "action_dim",
            "action_mode",
            "action_order",
            "action_representation",
            "action_stats_path",
            "allow_random_row_substitution",
            "camera_layout",
            "causal_temporal",
            "dataset_dir",
            "dataset_manifest",
            "delta_action",
            "episode_origin_only",
            "filter_static_segments",
            "height",
            "growing_history",
            "normalize_mode",
            "num_frames",
            "real_data_admission",
            "reject_nonzero_row_start",
            "repeat",
            "robot",
            "row_rate_manifest",
            "seed",
            "source_type",
            "task_name",
            "temporal_compression",
            "text_embedding_cache_dir",
            "train_tasks",
            "type",
            "vae_cache_dir",
            "vae_type",
            "val_ratio",
            "variant",
            "video_stride",
            "window_stride",
            "width",
        },
        "dataloader",
    )
    for field, expected in {
        "type": "cach_episode_origin",
        "source_type": "robotwin",
        "dataset_dir": "/DATA/share/RoboTwin2.0/dataset",
        "dataset_manifest": None,
        "row_rate_manifest": None,
        "real_data_admission": False,
        "robot": "aloha-agilex",
        "variant": None,
        "task_name": None,
        "train_tasks": None,
        "action_mode": "eef",
        "action_representation": "absolute_eef_target_xyz_rot6d_gripper",
        "action_dim": 20,
        "action_order": list(ACTION_ORDER),
        "delta_action": False,
        "vae_type": "ltx2",
        "temporal_compression": 8,
        "causal_temporal": True,
        "video_stride": 1,
        "num_frames": 33,
        "height": 384,
        "width": 320,
        "camera_layout": ["head_camera", "left_camera", "right_camera"],
        "growing_history": False,
        "episode_origin_only": True,
        "reject_nonzero_row_start": True,
        "repeat": 1,
        "seed": None,
        "val_ratio": 0.0,
        "window_stride": 1,
        "normalize_mode": None,
        "action_stats_path": None,
        "filter_static_segments": False,
        "allow_random_row_substitution": False,
        "text_embedding_cache_dir": None,
        "vae_cache_dir": None,
    }.items():
        _literal(dataloader[field], expected, f"dataloader.{field}")

    training = _exact_keys(
        root["training"],
        {
            "action_lr",
            "batch_size",
            "freeze",
            "grad_clip",
            "gradient_accumulation_steps",
            "init_checkpoint",
            "init_checkpoint_sha256",
            "keep_last_k",
            "lambda_action",
            "lambda_video",
            "max_steps",
            "num_workers",
            "output_dir",
            "save_steps",
            "seed",
            "video_lr",
            "warmup_steps",
            "weight_decay",
        },
        "training",
    )
    nullable = set(training) - {"freeze", "lambda_action", "lambda_video"}
    for field in nullable:
        _literal(training[field], None, f"training.{field}")
    _literal(
        training["freeze"],
        ["video_backbone.vae", "video_backbone.text_encoder"],
        "training.freeze",
    )
    _literal(training["lambda_video"], 1.0, "training.lambda_video")
    _literal(training["lambda_action"], 1.0, "training.lambda_action")

    admission = _exact_keys(
        root["admission"],
        {
            "allow_capture",
            "allow_contract_tests",
            "allow_evaluation",
            "allow_gpu",
            "allow_model_execution",
            "allow_run_root_creation",
            "allow_training",
            "authority",
        },
        "admission",
    )
    _literal(
        admission["authority"],
        "docs/cach_sana_wam/stage1/CACH_STAGE1_AUTHORITY.draft.json",
        "admission.authority",
    )
    _literal(
        admission["allow_contract_tests"],
        True,
        "admission.allow_contract_tests",
    )
    for field in (
        "allow_capture",
        "allow_evaluation",
        "allow_gpu",
        "allow_model_execution",
        "allow_run_root_creation",
        "allow_training",
    ):
        _literal(admission[field], False, f"admission.{field}")
    return root


__all__ = [
    "ACTION_ORDER",
    "CACHConfigError",
    "STAGE1_CONFIG_SCHEMA",
    "validate_stage1_config",
]
