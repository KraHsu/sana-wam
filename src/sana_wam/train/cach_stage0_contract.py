"""Torch-free, fail-closed validation for CACH-SANA-WAM Stage 0 artifacts.

This module deliberately cannot launch a model.  It validates the draft source,
candidate, and authority graph before any future launcher is allowed to import
torch, reserve a GPU, or create a run root.
"""

from __future__ import annotations

import json
import os
import platform
import re
import stat
import subprocess
import sys
from hashlib import sha256
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_SCHEMA = "cach-stage0-source-manifest-draft-v1"
CANDIDATE_SCHEMA = "cach-candidate-spec-draft-v1"
AUTHORITY_SCHEMA = "cach-stage0-authority-draft-v1"
STAGE0_CONFIG_SHA256 = (
    "8ff4d6ca94305b6aecfbf229d494faa92309ff75a5eaec8f369bb75aab6d14c3"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DEFAULT_SMALL_FILE_LIMIT = 8 * 1024 * 1024

_REPOSITORY_FILE_PATHS = {
    "benchmarks/robotwin/sana_wam2robotwin_interface.py",
    "benchmarks/robotwin/eval_policy_wrapper.py",
    "benchmarks/robotwin/single_eval.sh",
    "benchmarks/utils/action_conversion.py",
    "benchmarks/utils/client.py",
    "configs/deploy_gdn_ar.yaml",
    "configs/train_gdn_ar.yaml",
    "configs/train_sana_wm_gdn_ar.yaml",
    "scripts/check_cach_stage0_reserved_config.py",
    "scripts/deploy.py",
    "scripts/eval_ar_lownoise_seedfixed.sh",
    "scripts/precompute_vae_latents.py",
    "scripts/run_ar_lownoise_paired_arm.sh",
    "scripts/smoke_ar_lownoise_gpu.py",
    "scripts/train.py",
    "src/sana_wam/__init__.py",
    "src/sana_wam/config.py",
    "src/sana_wam/dataloader/robotwin_dataset.py",
    "src/sana_wam/dataloader/robotwin_stats_computation.py",
    "src/sana_wam/deploy/__init__.py",
    "src/sana_wam/deploy/gdn_ar_engine.py",
    "src/sana_wam/deploy/model_loader.py",
    "src/sana_wam/deploy/policy.py",
    "src/sana_wam/deploy/policy_server.py",
    "src/sana_wam/model/__init__.py",
    "src/sana_wam/model/base.py",
    "src/sana_wam/model/gdn_ar.py",
    "src/sana_wam/model/video_backbone/sana/adapter.py",
    "src/sana_wam/model/video_backbone/sana/pipeline_builder.py",
    "src/sana_wam/train/checkpointing.py",
    "src/sana_wam/train/cach_stage0_guard.py",
    "src/sana_wam/train/__init__.py",
    "src/sana_wam/train/trainer.py",
    "third_party/Sana/diffusion/model/nets/basic_modules.py",
    "third_party/Sana/diffusion/model/nets/sana_gdn_blocks.py",
    "third_party/Sana/diffusion/model/nets/sana_gdn_camctrl_blocks.py",
    "third_party/Sana/diffusion/model/nets/sana_multi_scale_video_camctrl.py",
    "third_party/Sana/diffusion/model/ops/fused_gdn.py",
    "third_party/Sana/diffusion/model/ops/fused_streaming.py",
    "third_party/Sana/diffusion/scheduler/self_forcing_flow_euler_sampler.py",
}

_AUTHORITY_PIN_PATHS = {
    "candidate_spec": "docs/cach_sana_wam/stage0/CACH_A_CANDIDATE_SPEC.draft.json",
    "config": "configs/experiments/cach_sana_wam_v0.yaml",
    "plan": (
        "docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_"
        "DEVELOPMENT_PLAN_20260731.md"
    ),
    "source_manifest": "docs/cach_sana_wam/stage0/SOURCE_MANIFEST.draft.json",
}

_AUTHORITY_DESIGN_PATHS = {
    "auxiliary_guard_tests": "tests/test_cach_stage0_auxiliary_guards.py",
    "cache_commit": (
        "docs/cach_sana_wam/stage0/"
        "HYBRID_CACHE_COMMIT_AND_APPLIED_ACTION_DESIGN.md"
    ),
    "candidate_launcher": "scripts/launch_cach_sana_wam.py",
    "checkpoint_config_guard_tests": (
        "tests/test_cach_stage0_checkpoint_config_guard.py"
    ),
    "config_guard_cli": "scripts/check_cach_stage0_reserved_config.py",
    "contract_tests": "tests/test_cach_stage0_contract.py",
    "data_scale": "docs/cach_sana_wam/stage0/DATA_AND_SCALE_DESIGN.md",
    "deploy_entrypoint_guard": "scripts/deploy.py",
    "deploy_guard_tests": "tests/test_cach_stage0_deploy_guard.py",
    "execution_surface_guard_tests": (
        "tests/test_cach_stage0_execution_surface_guards.py"
    ),
    "governance_agents": "docs/cach_sana_wam/stage0/governance/AGENTS.md",
    "governance_start_here": (
        "docs/cach_sana_wam/stage0/governance/START_HERE_20260730.md"
    ),
    "layout_bootstrap": (
        "docs/cach_sana_wam/stage0/"
        "CHUNK_ACTION_LAYOUT_AND_BOOTSTRAP_DESIGN.md"
    ),
    "gpu_smoke_entrypoint_guard": "scripts/smoke_ar_lownoise_gpu.py",
    "source_initialization_launcher": (
        "docs/cach_sana_wam/stage0/"
        "SOURCE_INITIALIZATION_AND_LAUNCHER_DESIGN.md"
    ),
    "stage0_contract": "src/sana_wam/train/cach_stage0_contract.py",
    "stage0_config_guard": "src/sana_wam/train/cach_stage0_guard.py",
    "stage0_readme": "docs/cach_sana_wam/stage0/README.md",
    "static_verifier": "scripts/verify_cach_stage0.py",
    "test_plan": "docs/cach_sana_wam/stage0/TEST_PLAN.md",
    "train_entrypoint_guard": "scripts/train.py",
    "train_entrypoint_guard_tests": "tests/test_cach_stage0_train_guard.py",
    "vae_cache_entrypoint_guard": "scripts/precompute_vae_latents.py",
}


class CACHStage0Error(ValueError):
    """Raised when a Stage 0 artifact is incomplete, changed, or executable."""


def strict_json_bytes(data: bytes, name: str) -> dict[str, Any]:
    """Load one UTF-8 JSON object while rejecting duplicate keys and NaN/Inf."""

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise CACHStage0Error(f"duplicate JSON key in {name}: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CACHStage0Error(f"non-finite JSON in {name}: {token}")
            ),
        )
    except CACHStage0Error:
        raise
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
        raise CACHStage0Error(f"invalid {name}: {exc}") from exc
    if type(value) is not dict:
        raise CACHStage0Error(f"{name} must be one JSON object")
    return value


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise CACHStage0Error(
            f"{name} keys differ: expected={sorted(expected)}, got={observed}"
        )
    return value


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise CACHStage0Error(f"{name} must be one lowercase SHA256")
    if value == "0" * 64:
        raise CACHStage0Error(f"{name} cannot be all-zero")
    return value


def _safe_relative_path(value: Any, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise CACHStage0Error(f"{name} must be one canonical repository-relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise CACHStage0Error(f"{name} must be one canonical repository-relative path")
    return value


def _pin(value: Any, name: str, *, expected_path: str | None = None) -> dict[str, Any]:
    pin = _exact_keys(value, {"path", "sha256"}, name)
    path = _safe_relative_path(pin["path"], f"{name}.path")
    if expected_path is not None and path != expected_path:
        raise CACHStage0Error(f"{name}.path differs")
    _digest(pin["sha256"], f"{name}.sha256")
    return pin


def _exact_literal_tree(observed: Any, expected: Any, name: str) -> None:
    """Compare a literal tree without Python's bool/int/float equality coercions."""

    if type(observed) is not type(expected):
        raise CACHStage0Error(f"{name} type differs")
    if type(expected) is dict:
        if set(observed) != set(expected):
            raise CACHStage0Error(f"{name} keys differ")
        for key, literal in expected.items():
            _exact_literal_tree(observed[key], literal, f"{name}.{key}")
        return
    if type(expected) is list:
        if len(observed) != len(expected):
            raise CACHStage0Error(f"{name} length differs")
        for index, literal in enumerate(expected):
            _exact_literal_tree(observed[index], literal, f"{name}[{index}]")
        return
    if observed != expected:
        raise CACHStage0Error(f"{name} differs")


def _require_literals(value: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    for field, literal in expected.items():
        _exact_literal_tree(value.get(field), literal, f"{name}.{field}")


def _blocked_header(value: dict[str, Any], *, schema: str, role: str, name: str) -> None:
    if value.get("schema_version") != schema:
        raise CACHStage0Error(f"{name} schema differs")
    if value.get("artifact_role") != role:
        raise CACHStage0Error(f"{name} role differs")
    if value.get("status") != "draft_blocked":
        raise CACHStage0Error(f"{name} must remain draft_blocked")
    if value.get("scientific_eligible") is not False:
        raise CACHStage0Error(f"{name} cannot be scientifically eligible")
    blockers = value.get("blockers")
    if type(blockers) is not list or not blockers or not all(
        type(item) is str and item for item in blockers
    ):
        raise CACHStage0Error(f"{name} must retain non-empty blockers")


def validate_source_manifest(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "artifact_role",
        "blockers",
        "dataset",
        "documents",
        "external_models",
        "forbidden_initial_inputs",
        "observed_at_utc",
        "repository",
        "repository_files",
        "runtime",
        "schema_version",
        "scientific_eligible",
        "status",
    }
    _exact_keys(value, expected, "source manifest")
    _blocked_header(
        value,
        schema=SOURCE_SCHEMA,
        role="cach_stage0_source_manifest_draft",
        name="source manifest",
    )

    repository = _exact_keys(
        value["repository"],
        {
            "additional_research_tree",
            "branch",
            "head",
            "path",
            "protected_modified_file",
            "sana_head",
            "stage0_start_status_entry_count_uall",
            "stage0_start_status_sha256_porcelain_v1_z_uall",
        },
        "repository",
    )
    research_tree = _exact_keys(
        repository["additional_research_tree"],
        {"handoff_path", "handoff_sha256", "head", "path"},
        "repository.additional_research_tree",
    )
    _require_literals(
        research_tree,
        {
            "handoff_path": "AGENT_HANDOFF_20260730.md",
            "handoff_sha256": (
                "76f251fde90cde1a2b970f6360d95f4dc51074f41579baebe130a283a4ab7920"
            ),
            "head": "9586486f2a9f5172d57b325e32093a3e018d34c0",
            "path": "/home/zch/workspace/sana-afcc-handoff",
        },
        "repository.additional_research_tree",
    )
    _require_literals(
        repository,
        {
            "branch": "handoff/sana-wam-20260730",
            "head": "605f1c134b4c983ff80f8489c4bc8847036329e2",
            "path": "/home/zch/workspace/sana-wam",
            "sana_head": "16b9cec673e3335724ba2d8db25de7f9ed229292",
            "stage0_start_status_entry_count_uall": 387,
            "stage0_start_status_sha256_porcelain_v1_z_uall": (
                "07b79900262b11bc55a180475ef133a99cc2a08e9a66eff492c582ace673d758"
            ),
        },
        "repository",
    )
    protected = _exact_keys(
        repository["protected_modified_file"],
        {"head_sha256", "path", "worktree_diff_sha256", "worktree_sha256"},
        "repository.protected_modified_file",
    )
    if protected["path"] != "tests/test_phase6_candidate_eligibility.py":
        raise CACHStage0Error("protected modified file path differs")
    for field in ("head_sha256", "worktree_diff_sha256", "worktree_sha256"):
        _digest(protected[field], f"repository.protected_modified_file.{field}")

    repository_files = value["repository_files"]
    if type(repository_files) is not dict or set(repository_files) != _REPOSITORY_FILE_PATHS:
        raise CACHStage0Error("repository source path closure differs")
    for path, digest in repository_files.items():
        _safe_relative_path(path, "repository source path")
        _digest(digest, f"repository file {path}")

    documents = value["documents"]
    if type(documents) is not dict or set(documents) != {
        "deep_research_report",
        "development_plan",
        "handoff",
        "root_agents",
        "root_start_here",
    }:
        raise CACHStage0Error("source document closure differs")
    expected_documents = {
        "deep_research_report": "/home/zch/workspace/sana-wam/deep-research-report.md",
        "development_plan": (
            "/home/zch/workspace/sana-wam/docs/"
            "CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md"
        ),
        "handoff": (
            "/home/zch/workspace/sana-wam/docs/agent_handoff/"
            "AGENT_HANDOFF_20260730.md"
        ),
        "root_agents": "/home/zch/workspace/sana-wam/AGENTS.md",
        "root_start_here": "/home/zch/workspace/sana-wam/START_HERE_20260730.md",
    }
    for name, pin in documents.items():
        _exact_keys(pin, {"path", "sha256"}, f"document {name}")
        _digest(pin.get("sha256"), f"document {name}")
        if pin.get("path") != expected_documents[name]:
            raise CACHStage0Error(f"document {name} path differs")

    dataset = _exact_keys(
        value["dataset"],
        {
            "episode_counts",
            "file_count",
            "full_content_sha256",
            "hdf5_timestamp_field_observed",
            "rate_provenance_receipt",
            "realpath",
            "task_directory_count",
            "tree_size_human_snapshot",
        },
        "dataset",
    )
    _require_literals(
        dataset,
        {
            "file_count": 110852,
            "full_content_sha256": None,
            "hdf5_timestamp_field_observed": False,
            "rate_provenance_receipt": None,
            "realpath": "/DATA/share/RoboTwin2.0/dataset",
            "task_directory_count": 50,
            "tree_size_human_snapshot": "896G",
        },
        "dataset",
    )
    episode_counts = _exact_keys(
        dataset["episode_counts"],
        {"aloha-agilex_clean_50", "aloha-agilex_randomized_500"},
        "dataset.episode_counts",
    )
    _require_literals(
        episode_counts,
        {"aloha-agilex_clean_50": 2500, "aloha-agilex_randomized_500": 25000},
        "dataset.episode_counts",
    )

    runtime = _exact_keys(
        value["runtime"],
        {
            "cuda_build",
            "diffusers",
            "driver",
            "h5py",
            "numpy",
            "omegaconf",
            "package_inventory_sha256",
            "python",
            "python_executable_realpath",
            "python_executable_sha256",
            "torch",
            "transformers",
            "venv_pip_available",
        },
        "runtime",
    )
    _require_literals(
        runtime,
        {
            "cuda_build": "12.8",
            "diffusers": "0.38.0",
            "driver": "570.211.01",
            "h5py": "3.16.0",
            "numpy": "2.4.6",
            "omegaconf": "2.3.1",
            "package_inventory_sha256": None,
            "python": "3.12.13",
            "python_executable_realpath": (
                "/home/zch/.local/share/uv/python/"
                "cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
            ),
            "python_executable_sha256": (
                "339de109f072e52523066d97a2b3372e358a10cc00c6a9df62c28a6618061900"
            ),
            "torch": "2.7.1+cu128",
            "transformers": "5.12.1",
            "venv_pip_available": False,
        },
        "runtime",
    )

    models = value["external_models"]
    if type(models) is not dict or set(models) != {"ltx2_causal_vae", "text_encoder"}:
        raise CACHStage0Error("external model closure differs")
    for model_name, model in models.items():
        if model.get("frozen") is not True:
            raise CACHStage0Error(f"{model_name} must be frozen")
        if type(model.get("realpath")) is not str or not Path(model["realpath"]).is_absolute():
            raise CACHStage0Error(f"{model_name} realpath must be absolute")
        files = model.get("files")
        if type(files) is not list or not files:
            raise CACHStage0Error(f"{model_name} file inventory is empty")
        file_paths = [item.get("path") for item in files if type(item) is dict]
        if (
            len(file_paths) != len(files)
            or not all(type(path) is str for path in file_paths)
            or len(file_paths) != len(set(file_paths))
        ):
            raise CACHStage0Error(f"{model_name} file inventory paths are not unique")
        for item in files:
            _exact_keys(
                item,
                {"path", "sha256", "size_bytes"},
                f"{model_name} file inventory item",
            )
            _safe_relative_path(item.get("path"), f"{model_name} file path")
            _digest(item.get("sha256"), f"{model_name}/{item.get('path')}")
            if type(item.get("size_bytes")) is not int or item["size_bytes"] < 0:
                raise CACHStage0Error(f"{model_name} file size differs")

    vae = _exact_keys(
        models["ltx2_causal_vae"],
        {
            "causal_first_frame_contract",
            "decoder_causal",
            "encoder_causal",
            "files",
            "frozen",
            "latent_channels",
            "realpath",
            "spatial_compression_ratio",
            "temporal_compression_ratio",
        },
        "ltx2_causal_vae",
    )
    _require_literals(
        vae,
        {
            "causal_first_frame_contract": (
                "first_frame_latent_then_8_frame_groups"
            ),
            "decoder_causal": True,
            "encoder_causal": True,
            "frozen": True,
            "latent_channels": 128,
            "realpath": "/DATA/share/SANA-WM_streaming/ltx2_causal_vae",
            "spatial_compression_ratio": 32,
            "temporal_compression_ratio": 8,
        },
        "ltx2_causal_vae",
    )
    if {item["path"] for item in vae["files"]} != {
        "config.json",
        "diffusion_pytorch_model.safetensors",
    }:
        raise CACHStage0Error("LTX2 file closure differs")

    text = _exact_keys(
        models["text_encoder"],
        {"files", "frozen", "model_type", "realpath"},
        "text_encoder",
    )
    _require_literals(
        text,
        {
            "frozen": True,
            "model_type": "gemma2",
            "realpath": "/DATA/share/gemma-2-2b-it",
        },
        "text_encoder",
    )
    if {item["path"] for item in text["files"]} != {
        ".msc",
        ".mv",
        "README.md",
        "config.json",
        "configuration.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }:
        raise CACHStage0Error("text encoder file closure differs")

    forbidden = value["forbidden_initial_inputs"]
    if type(forbidden) is not list or len(forbidden) != 1:
        raise CACHStage0Error("forbidden initial input closure differs")
    item = _exact_keys(
        forbidden[0], {"path", "reason", "size_bytes"}, "forbidden initial input"
    )
    _require_literals(
        item,
        {
            "path": "/DATA/share/SANA-WM_streaming/sana_dit/model.pt",
            "reason": "CACH v0 initial training rejects pretrained video-DiT",
            "size_bytes": 31950734339,
        },
        "forbidden initial input",
    )
    if type(value["observed_at_utc"]) is not str or not value["observed_at_utc"].endswith("Z"):
        raise CACHStage0Error("source observation timestamp differs")
    return value


def validate_candidate_spec(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "afcc_isolation",
        "architecture",
        "artifact_role",
        "blockers",
        "campaign",
        "checkpoint",
        "data",
        "execution_authorized",
        "gates",
        "initialization",
        "plan",
        "prohibited",
        "root",
        "schema_version",
        "scientific_eligible",
        "source_manifest",
        "status",
        "training",
    }
    _exact_keys(value, expected, "candidate spec")
    _blocked_header(
        value,
        schema=CANDIDATE_SCHEMA,
        role="cach_a_candidate_spec_draft",
        name="candidate spec",
    )
    if value.get("execution_authorized") is not False:
        raise CACHStage0Error("candidate spec cannot authorize execution")

    afcc = _exact_keys(
        value["afcc_isolation"],
        {
            "allow_afcc_reference",
            "allow_afcc_weight",
            "candidate_F",
            "reference_F",
            "required_before_trainer_construction",
        },
        "candidate afcc_isolation",
    )
    _require_literals(
        afcc,
        {
            "allow_afcc_reference": False,
            "allow_afcc_weight": False,
            "candidate_F": 0,
            "reference_F": 0,
            "required_before_trainer_construction": True,
        },
        "candidate afcc_isolation",
    )

    architecture = _exact_keys(
        value["architecture"],
        {
            "action_conditioning",
            "anchors",
            "attnres",
            "bootstrap",
            "cache",
            "frame_chunk_size",
            "layout_spec",
            "proprio_boundary",
            "self_forcing",
            "variant",
        },
        "candidate architecture",
    )
    _require_literals(
        architecture,
        {"frame_chunk_size": 3, "variant": "cach_sana_wam_v0"},
        "candidate architecture",
    )
    _require_literals(
        _exact_keys(
            architecture["action_conditioning"],
            {"candidate", "reference", "seam"},
            "candidate action_conditioning",
        ),
        {
            "candidate": "zero_init_action_to_video_adapter",
            "reference": "disabled_identity_bypass",
            "seam": "post_self_attention_use_delta_pose_additive_internal_only",
        },
        "candidate action_conditioning",
    )
    for name in ("anchors", "attnres", "self_forcing"):
        _require_literals(
            _exact_keys(architecture[name], {"enabled"}, f"candidate {name}"),
            {"enabled": False},
            f"candidate {name}",
        )
    _require_literals(
        _exact_keys(
            architecture["bootstrap"],
            {
                "episode_bootstrap",
                "episode_origin_only",
                "observed_prefix_chunks",
                "reject_ar_observed_prefix_chunks",
                "reject_nonzero_clean_prefix",
                "reject_nonzero_row_start",
            },
            "candidate bootstrap",
        ),
        {
            "episode_bootstrap": "first_frame_pinned",
            "episode_origin_only": True,
            "observed_prefix_chunks": 0,
            "reject_ar_observed_prefix_chunks": True,
            "reject_nonzero_clean_prefix": True,
            "reject_nonzero_row_start": True,
        },
        "candidate bootstrap",
    )
    _require_literals(
        _exact_keys(
            architecture["cache"],
            {
                "attnres_depth_state",
                "commit_source_policy",
                "expected_gdn_layer_count",
                "expected_softmax_layer_count",
                "gdn_temporal_state",
                "softmax_policy_if_instantiated",
                "softmax_temporal_state",
            },
            "candidate cache",
        ),
        {
            "attnres_depth_state": "forward_local",
            "commit_source_policy": {
                "commanded_action_only_commit_allowed": False,
                "deploy_applied_ack_enabled": False,
                "offline_training": "teacher_forcing_dataset_pair",
                "self_forcing_generated_pair_enabled": False,
            },
            "expected_gdn_layer_count": 20,
            "expected_softmax_layer_count": 0,
            "gdn_temporal_state": "full_history",
            "softmax_policy_if_instantiated": (
                "exactly_one_previous_committed_chunk"
            ),
            "softmax_temporal_state": "not_instantiated_in_cach_a",
        },
        "candidate cache",
    )
    layout = _exact_keys(
        architecture["layout_spec"],
        {
            "action_rate_source",
            "causal_vae_temporal_compression",
            "path",
            "sha256",
            "video_stride",
        },
        "candidate layout_spec",
    )
    _require_literals(
        layout,
        {
            "action_rate_source": None,
            "causal_vae_temporal_compression": 8,
            "path": (
                "docs/cach_sana_wam/stage0/"
                "CHUNK_ACTION_LAYOUT_AND_BOOTSTRAP_DESIGN.md"
            ),
            "video_stride": 1,
        },
        "candidate layout_spec",
    )
    _digest(layout["sha256"], "candidate layout_spec.sha256")
    _require_literals(
        _exact_keys(
            architecture["proprio_boundary"],
            {
                "chunk0_raw_index",
                "continuation",
                "equal_rate_raw_index",
                "model_input",
                "reject_clip_level_state_for_multichunk",
            },
            "candidate proprio_boundary",
        ),
        {
            "chunk0_raw_index": 0,
            "continuation": (
                "latest_observed_state_after_previous_committed_action_span"
            ),
            "equal_rate_raw_index": "action_span_start",
            "model_input": "exact_selected_per_chunk_states_only",
            "reject_clip_level_state_for_multichunk": True,
        },
        "candidate proprio_boundary",
    )

    campaign = _exact_keys(
        value["campaign"],
        {
            "campaign_id",
            "candidate",
            "candidate_revision",
            "purpose",
            "reference",
            "scope",
            "unique_delta",
        },
        "candidate campaign",
    )
    _require_literals(
        campaign,
        {
            "campaign_id": None,
            "candidate": "CACH-A",
            "candidate_revision": None,
            "purpose": "action_conditioned_dynamics_increment",
            "reference": "REF-GDN-CORRECTED",
            "scope": "future_reduced_mechanism_campaign",
            "unique_delta": (
                "candidate video dynamics reads registered noisy/current and "
                "committed/past action conditioning; reference uses identity bypass"
            ),
        },
        "candidate campaign",
    )

    checkpoint = _exact_keys(
        value["checkpoint"],
        {
            "final_endpoint",
            "init_dit_from",
            "initial_checkpoint",
            "load_missing_keys_allowlist",
            "load_unexpected_keys_allowlist",
            "model_path",
            "pretrained_video_dit_allowed",
            "resume_allowed_only_same_revision_exact_schema",
        },
        "candidate checkpoint",
    )
    _require_literals(
        checkpoint,
        {
            "final_endpoint": None,
            "init_dit_from": None,
            "initial_checkpoint": None,
            "load_missing_keys_allowlist": [],
            "load_unexpected_keys_allowlist": [],
            "model_path": None,
            "pretrained_video_dit_allowed": False,
            "resume_allowed_only_same_revision_exact_schema": True,
        },
        "candidate checkpoint",
    )

    data = _exact_keys(
        value["data"],
        {
            "action_dim",
            "action_mode",
            "action_order",
            "action_representation",
            "data_order_manifest",
            "dataset_manifest",
            "delta_action",
            "normalization_stats",
            "row_rate_manifest",
            "task_weights",
        },
        "candidate data",
    )
    _require_literals(
        data,
        {
            "action_dim": 20,
            "action_mode": "eef",
            "action_order": [
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
            ],
            "action_representation": "absolute_eef_target_xyz_rot6d_gripper",
            "data_order_manifest": None,
            "dataset_manifest": None,
            "delta_action": False,
            "normalization_stats": None,
            "row_rate_manifest": None,
            "task_weights": None,
        },
        "candidate data",
    )
    gates = _exact_keys(
        value["gates"],
        {
            "G0_static",
            "G1_unit",
            "G2_numeric",
            "G3_update_free",
            "G4_registered_train",
            "threshold_manifest",
        },
        "candidate gates",
    )
    _require_literals(
        gates,
        {
            "G0_static": "blocked",
            "G1_unit": "not_run",
            "G2_numeric": "not_run",
            "G3_update_free": "not_run",
            "G4_registered_train": "not_authorized",
            "threshold_manifest": None,
        },
        "candidate gates",
    )
    initialization = _exact_keys(
        value["initialization"],
        {
            "candidate_operator_specific_seed",
            "complete_hybrid_from_step_0",
            "frozen_components",
            "initialization_mode",
            "reference_operator_specific_seed",
            "shared_tensor_init_digest",
            "shared_tensor_seed",
            "trainable_components",
        },
        "candidate initialization",
    )
    _require_literals(
        initialization,
        {
            "candidate_operator_specific_seed": None,
            "complete_hybrid_from_step_0": True,
            "frozen_components": [
                "video_backbone.vae",
                "video_backbone.text_encoder",
            ],
            "initialization_mode": "complete_random_v1",
            "reference_operator_specific_seed": None,
            "shared_tensor_init_digest": None,
            "shared_tensor_seed": None,
            "trainable_components": [
                "video_dit",
                "gdn",
                "action_backbone",
                "proprio_encoder",
                "action_conditioner",
            ],
        },
        "candidate initialization",
    )

    _pin(value["plan"], "candidate plan", expected_path=_AUTHORITY_PIN_PATHS["plan"])
    _pin(
        value["source_manifest"],
        "candidate source manifest",
        expected_path=_AUTHORITY_PIN_PATHS["source_manifest"],
    )

    prohibited = [
        "checkpoint_selection",
        "task_selection",
        "seed_selection",
        "anchor_or_ratio_sweep",
        "rerank",
        "best_of_n",
        "temporal_ensemble",
        "dagger_or_recovery",
        "approximate_replay",
        "loss_subtraction",
    ]
    if value["prohibited"] != prohibited:
        raise CACHStage0Error("candidate prohibited mechanism list differs")

    root = _exact_keys(
        value["root"],
        {"candidate_root", "reference_root", "root_nonce"},
        "candidate root",
    )
    _require_literals(
        root,
        {"candidate_root": None, "reference_root": None, "root_nonce": None},
        "candidate root",
    )
    training = _exact_keys(
        value["training"],
        {"data_order", "optimizer", "reduced_mechanism_only", "seed", "step_budget"},
        "candidate training",
    )
    _require_literals(
        training,
        {
            "data_order": None,
            "optimizer": None,
            "reduced_mechanism_only": True,
            "seed": None,
            "step_budget": None,
        },
        "candidate training",
    )
    return value


def validate_stage0_config(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact non-executable YAML mapping after safe resolution."""

    expected = {
        "cach_stage0": {
            "schema_version": "cach-sana-wam-config-draft-v1",
            "status": "draft_blocked",
            "execution_enabled": False,
            "scientific_eligible": False,
            "plan": _AUTHORITY_PIN_PATHS["plan"],
            "source_manifest": _AUTHORITY_PIN_PATHS["source_manifest"],
            "candidate_spec": _AUTHORITY_PIN_PATHS["candidate_spec"],
        },
        "model": {
            "architecture": {
                "framework": "dual_system",
                "variant": "cach_sana_wam_v0",
                "action_dim": 20,
                "state_dim": 20,
                "use_proprioception": True,
                "proprio_per_chunk": True,
                "proprio_boundary": "layout_committed_boundary_v1",
                "frame_chunk_size": 3,
                "episode_bootstrap": "first_frame_pinned",
                "observed_prefix_chunks": 0,
                "action_condition": {
                    "enabled": True,
                    "public_name": "action_condition",
                    "internal_vendor_seam": "use_delta_pose_additive",
                    "initialization": "zero",
                },
                "causal_softmax_anchors": {"enabled": False, "indices": []},
                "block_attn_res": {"enabled": False},
                "self_forcing": {"enabled": False},
                "initialization_mode": "complete_random_v1",
            },
            "video_backbone": {
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
                "flow_shift": None,
            },
            "action_backbone": {"dim": 1152, "ffn_dim": 4608},
        },
        "dataloader": {
            "type": "robotwin",
            "dataset_dir": "/DATA/share/RoboTwin2.0/dataset",
            "dataset_manifest": None,
            "row_rate_manifest": None,
            "robot": "aloha-agilex",
            "variant": None,
            "action_mode": "eef",
            "delta_action": False,
            "vae_type": "ltx2",
            "temporal_compression": 8,
            "causal_temporal": True,
            "video_stride": 1,
            "height": 384,
            "width": 320,
            "multiview": True,
            "camera_layout": [
                "head_camera",
                "left_camera",
                "right_camera",
            ],
            "growing_history": False,
            "episode_origin_only": True,
            "reject_nonzero_row_start": True,
        },
        "training": {
            "output_dir": None,
            "init_checkpoint": None,
            "init_checkpoint_sha256": None,
            "max_steps": None,
            "save_steps": None,
            "keep_last_k": None,
            "batch_size": None,
            "gradient_accumulation_steps": None,
            "num_workers": None,
            "video_lr": None,
            "action_lr": None,
            "weight_decay": None,
            "grad_clip": None,
            "warmup_steps": None,
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "freeze": [
                "video_backbone.vae",
                "video_backbone.text_encoder",
            ],
            "seed": None,
        },
        "admission": {
            "authority": (
                "docs/cach_sana_wam/stage0/CACH_A_AUTHORITY.draft.json"
            ),
            "allow_launch": False,
            "allow_test": False,
            "allow_training": False,
            "allow_evaluation": False,
            "allow_capture": False,
        },
    }
    _exact_literal_tree(value, expected, "resolved CACH Stage-0 config")
    return value


def validate_authority(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "allowed_actions",
        "artifact_role",
        "blockers",
        "candidate_spec",
        "config",
        "decision",
        "designs",
        "execution_authorized",
        "plan",
        "prohibited_actions",
        "registered",
        "schema_version",
        "scientific_eligible",
        "source_manifest",
        "status",
    }
    _exact_keys(value, expected, "authority")
    _blocked_header(
        value,
        schema=AUTHORITY_SCHEMA,
        role="cach_stage0_authority_draft",
        name="authority",
    )
    if value.get("decision") != "deny_execution":
        raise CACHStage0Error("draft authority must deny execution")
    if value.get("registered") is not False:
        raise CACHStage0Error("draft authority cannot be registered")
    if value.get("execution_authorized") is not False:
        raise CACHStage0Error("draft authority cannot authorize execution")
    if value.get("allowed_actions") != ["read_only_review", "static_diff_review"]:
        raise CACHStage0Error("draft authority allowed actions differ")
    for role, expected_path in _AUTHORITY_PIN_PATHS.items():
        _pin(value[role], f"authority {role}", expected_path=expected_path)
    if value["config"]["sha256"] != STAGE0_CONFIG_SHA256:
        raise CACHStage0Error("draft authority config bytes differ")
    designs = value.get("designs")
    if type(designs) is not dict or set(designs) != set(_AUTHORITY_DESIGN_PATHS):
        raise CACHStage0Error("authority design closure differs")
    for role, expected_path in _AUTHORITY_DESIGN_PATHS.items():
        _pin(
            designs[role],
            f"authority design {role}",
            expected_path=expected_path,
        )
    prohibited = [
        "tests",
        "model_execution",
        "training",
        "evaluation",
        "capture",
        "run_root_creation",
        "gpu_reservation",
        "stage1_implementation",
    ]
    if value.get("prohibited_actions") != prohibited:
        raise CACHStage0Error("draft authority prohibited actions differ")
    return value


def read_regular_bytes(
    path: Path, name: str, *, max_bytes: int = _DEFAULT_SMALL_FILE_LIMIT
) -> bytes:
    """Read one regular non-symlink file through an O_NOFOLLOW descriptor."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise CACHStage0Error(f"{name} byte limit must be positive")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CACHStage0Error(f"cannot open {name}: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CACHStage0Error(f"{name} is not a regular file: {path}")
        if before.st_size > max_bytes:
            raise CACHStage0Error(
                f"{name} exceeds the {max_bytes}-byte Stage-0 read limit: {path}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise CACHStage0Error(
                    f"{name} exceeds the {max_bytes}-byte Stage-0 read limit: {path}"
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise CACHStage0Error(f"{name} changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def file_sha256(path: Path, name: str) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CACHStage0Error(f"cannot open {name}: {path}: {exc}") from exc
    digest = sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CACHStage0Error(f"{name} is not a regular file: {path}")
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise CACHStage0Error(f"{name} changed while being hashed: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _git(repository_root: Path, *args: str) -> str:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "diff.external=",
                "-C",
                os.fspath(repository_root),
                *args,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise CACHStage0Error(f"git identity check failed: {args!r}: {exc}") from exc
    return result.stdout.strip()


def _git_bytes(repository_root: Path, *args: str) -> bytes:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "diff.external=",
                "-C",
                os.fspath(repository_root),
                *args,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise CACHStage0Error(f"git byte check failed: {args!r}: {exc}") from exc
    return result.stdout


def resolve_repository_file(
    repository_root: Path, relative_path: str, name: str
) -> Path:
    """Resolve a canonical pin beneath repository_root without allowing escape."""

    _safe_relative_path(relative_path, name)
    root = repository_root.resolve()
    path = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise CACHStage0Error(f"{name} escapes or is missing from repository") from exc
    return path


def _repository_path(
    manifest_repository_root: Path,
    actual_repository_root: Path,
    absolute_path: str,
    name: str,
) -> Path:
    source = Path(absolute_path)
    if not source.is_absolute():
        raise CACHStage0Error(f"{name} manifest path must be absolute")
    try:
        relative = source.relative_to(manifest_repository_root)
    except ValueError as exc:
        raise CACHStage0Error(f"{name} is outside the manifest repository") from exc
    return resolve_repository_file(
        actual_repository_root, PurePosixPath(relative.as_posix()).as_posix(), name
    )


def verify_runtime_identity(source_manifest: dict[str, Any]) -> None:
    """Verify the Stage-0 interpreter/distribution subset without importing torch."""

    runtime = source_manifest["runtime"]
    executable = Path(runtime["python_executable_realpath"])
    if Path(sys.executable).resolve(strict=True) != executable:
        raise CACHStage0Error("runtime Python executable realpath differs")
    if file_sha256(executable, "runtime Python executable") != runtime[
        "python_executable_sha256"
    ]:
        raise CACHStage0Error("runtime Python executable bytes differ")
    if platform.python_version() != runtime["python"]:
        raise CACHStage0Error("runtime Python version differs")
    distributions = {
        "diffusers": "diffusers",
        "h5py": "h5py",
        "numpy": "numpy",
        "omegaconf": "omegaconf",
        "torch": "torch",
        "transformers": "transformers",
    }
    for field, distribution in distributions.items():
        try:
            observed = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise CACHStage0Error(
                f"runtime distribution is missing: {distribution}"
            ) from exc
        if observed != runtime[field]:
            raise CACHStage0Error(
                f"runtime distribution version differs: {distribution}"
            )


def _directory_regular_file_inventory(root: Path, name: str) -> set[str]:
    inventory: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for child in list(directory_names):
            path = directory_path / child
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise CACHStage0Error(f"{name} contains a non-directory node: {path}")
        for child in file_names:
            path = directory_path / child
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise CACHStage0Error(f"{name} contains a non-regular file: {path}")
            relative = path.relative_to(root).as_posix()
            _safe_relative_path(relative, f"{name} inventory path")
            if relative in inventory:
                raise CACHStage0Error(f"{name} contains a duplicate path: {relative}")
            inventory.add(relative)
    return inventory


def verify_dataset_snapshot(source_manifest: dict[str, Any]) -> None:
    """Recount the explicitly limited dataset snapshot (not a content manifest)."""

    dataset = source_manifest["dataset"]
    declared_root = Path(dataset["realpath"])
    root = declared_root.resolve(strict=True)
    if root != declared_root or not root.is_dir():
        raise CACHStage0Error("dataset path is not its declared directory realpath")
    task_count = sum(
        1
        for child in root.iterdir()
        if child.is_dir() and not child.is_symlink()
    )
    if task_count != dataset["task_directory_count"]:
        raise CACHStage0Error("dataset task-directory count changed")

    file_count = 0
    episode_counts = {name: 0 for name in dataset["episode_counts"]}
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for child in directory_names:
            path = directory_path / child
            if path.is_symlink():
                raise CACHStage0Error(f"dataset contains a symlink directory: {path}")
        for child in file_names:
            path = directory_path / child
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise CACHStage0Error(f"dataset contains a non-regular file: {path}")
            file_count += 1
            if path.suffix == ".hdf5":
                parts = set(path.relative_to(root).parts)
                for cohort in episode_counts:
                    if cohort in parts:
                        episode_counts[cohort] += 1
    if file_count != dataset["file_count"]:
        raise CACHStage0Error("dataset regular-file count changed")
    if episode_counts != dataset["episode_counts"]:
        raise CACHStage0Error("dataset episode-count snapshot changed")


def verify_repository_inputs(
    repository_root: Path,
    source_manifest: dict[str, Any],
    *,
    verify_external: bool = False,
) -> None:
    repository_root = repository_root.resolve()
    repository = source_manifest["repository"]
    if _git(repository_root, "rev-parse", "HEAD") != repository["head"]:
        raise CACHStage0Error("main repository HEAD differs")
    if _git(repository_root, "symbolic-ref", "--short", "HEAD") != repository["branch"]:
        raise CACHStage0Error("main repository branch differs")
    if _git(repository_root / "third_party/Sana", "rev-parse", "HEAD") != repository["sana_head"]:
        raise CACHStage0Error("Sana repository HEAD differs")
    research = repository["additional_research_tree"]
    research_root = Path(research["path"]).resolve(strict=True)
    if research_root != Path(research["path"]) or not research_root.is_dir():
        raise CACHStage0Error("additional research tree path differs")
    if _git(research_root, "rev-parse", "HEAD") != research["head"]:
        raise CACHStage0Error("additional research tree HEAD differs")
    research_handoff = research_root / research["handoff_path"]
    if (
        file_sha256(research_handoff, "additional research-tree handoff")
        != research["handoff_sha256"]
    ):
        raise CACHStage0Error("additional research-tree handoff changed")

    for relative, expected in source_manifest["repository_files"].items():
        path = resolve_repository_file(
            repository_root, relative, f"repository file {relative}"
        )
        observed = file_sha256(path, f"repository file {relative}")
        if observed != expected:
            raise CACHStage0Error(f"repository file changed: {relative}")

    manifest_root = Path(repository["path"])
    for name, pin in source_manifest["documents"].items():
        path = _repository_path(
            manifest_root, repository_root, pin["path"], f"document {name}"
        )
        if file_sha256(path, f"document {name}") != pin["sha256"]:
            raise CACHStage0Error(f"document changed: {name}")

    dirty = repository["protected_modified_file"]
    dirty_path = resolve_repository_file(
        repository_root, dirty["path"], "protected modified file"
    )
    if file_sha256(dirty_path, "protected modified file") != dirty["worktree_sha256"]:
        raise CACHStage0Error("protected modified file changed")
    head_bytes = _git_bytes(repository_root, "show", f"HEAD:{dirty['path']}")
    if sha256(head_bytes).hexdigest() != dirty["head_sha256"]:
        raise CACHStage0Error("protected modified file HEAD bytes differ")
    diff = _git_bytes(
        repository_root,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--",
        dirty["path"],
    )
    if sha256(diff).hexdigest() != dirty["worktree_diff_sha256"]:
        raise CACHStage0Error("protected modified file diff changed")

    verify_runtime_identity(source_manifest)
    verify_dataset_snapshot(source_manifest)

    if verify_external:
        for model_name, model in source_manifest["external_models"].items():
            declared_root = Path(model["realpath"])
            root = declared_root.resolve(strict=True)
            if root != declared_root:
                raise CACHStage0Error(f"{model_name} path is not its realpath")
            expected_paths = {item["path"] for item in model["files"]}
            inventory_before = _directory_regular_file_inventory(root, model_name)
            if inventory_before != expected_paths:
                raise CACHStage0Error(f"{model_name} directory inventory changed")
            for item in model["files"]:
                path = root.joinpath(*PurePosixPath(item["path"]).parts)
                try:
                    path.resolve(strict=True).relative_to(root)
                except (OSError, ValueError) as exc:
                    raise CACHStage0Error(
                        f"external model file escapes source root: {path}"
                    ) from exc
                observed_size = path.stat(follow_symlinks=False).st_size
                if observed_size != item["size_bytes"]:
                    raise CACHStage0Error(f"external model file size changed: {path}")
                if file_sha256(path, f"{model_name}/{item['path']}") != item["sha256"]:
                    raise CACHStage0Error(f"external model file changed: {path}")
            if model_name == "ltx2_causal_vae":
                config = strict_json_bytes(
                    read_regular_bytes(root / "config.json", "LTX2 config"),
                    "LTX2 config",
                )
                expected_config = {
                    "decoder_causal": model["decoder_causal"],
                    "encoder_causal": model["encoder_causal"],
                    "latent_channels": model["latent_channels"],
                    "spatial_compression_ratio": model[
                        "spatial_compression_ratio"
                    ],
                    "temporal_compression_ratio": model[
                        "temporal_compression_ratio"
                    ],
                }
                for field, expected in expected_config.items():
                    if config.get(field) != expected:
                        raise CACHStage0Error(
                            f"LTX2 config contract differs: {field}"
                        )
            inventory_after = _directory_regular_file_inventory(root, model_name)
            if inventory_after != inventory_before:
                raise CACHStage0Error(
                    f"{model_name} directory inventory changed while being verified"
                )


def verify_governance_mirrors(repository_root: Path) -> None:
    tracked = repository_root / "docs/cach_sana_wam/stage0/governance"
    for name in ("AGENTS.md", "START_HERE_20260730.md"):
        source = read_regular_bytes(tracked / name, f"tracked governance {name}")
        deployed = read_regular_bytes(repository_root / name, f"root governance {name}")
        if deployed != source:
            raise CACHStage0Error(f"root governance mirror differs: {name}")


__all__ = [
    "AUTHORITY_SCHEMA",
    "CACHStage0Error",
    "CANDIDATE_SCHEMA",
    "SOURCE_SCHEMA",
    "STAGE0_CONFIG_SHA256",
    "file_sha256",
    "read_regular_bytes",
    "resolve_repository_file",
    "strict_json_bytes",
    "validate_authority",
    "validate_candidate_spec",
    "validate_source_manifest",
    "validate_stage0_config",
    "verify_governance_mirrors",
    "verify_dataset_snapshot",
    "verify_repository_inputs",
    "verify_runtime_identity",
]
