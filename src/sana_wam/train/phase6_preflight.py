"""Torch-free, fail-closed Phase-6 source and artifact preflight validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import hmac
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
from types import ModuleType
from typing import Any

from sana_wam.dataloader.phase6_expansion_support import (
    EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
    EXPANSION_SUPPORT_CONTRACT_SHA256,
    EXPANSION_SUPPORT_KEYS,
    ExpansionSupportError,
    expansion_support_contract,
    validate_expansion_support,
)
from sana_wam.dataloader.phase6_padding_semantics import (
    PADDING_SEMANTICS_AMENDMENT_SHA256,
    PADDING_SEMANTICS_CONTRACT_SHA256,
    PaddingSemanticsError,
    validate_padding_eligibility_dependency,
    validate_padding_semantics_contract,
)

from .phase6_downstream_pins import (
    DATASET_CONTRACT_ARTIFACT_SHA256,
    EMBEDDED_PLAN_SHA256,
    IDENTITY_SHA256,
    TASK_PLAN_ARTIFACT_SHA256,
)
from .phase6_recovery import (
    PLANROW_RECOVERY_AUTHORITY_PATH,
    PLANROW_RECOVERY_AUTHORITY_SHA256,
    recovery_lifecycle,
    validate_phase6_planrow_recovery_authority,
)


REQUEST_SCHEMA_VERSION = "sana-phase6-preflight-request-v4"
SOURCE_MANIFEST_SCHEMA_VERSION = "sana-phase6-source-runtime-manifest-v3"
REPORT_SCHEMA_VERSION = "sana-phase6-preflight-report-v4"
RUNTIME_SUPPORT_SCHEMA_VERSION = "sana-phase6-runtime-support-manifest-v1"

_METADATA_SIZE_LIMIT = 128 * 1024 * 1024
_HASH_BLOCK_SIZE = 8 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FLOAT32_BITS_RE = re.compile(r"[0-9a-f]{8}\Z")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_EXPECTED_REPOSITORY_HEAD = "b2a31ad377bd9ce6baae1a73d7876a6d3cda53db"
_EXPECTED_SANA_HEAD = "c1c48d8d44a09cf7459c51830a9cc7a988431d4d"

_SOURCE_INVENTORY_GLOBS = (
    ".python-version",
    "configs/phase6/templates/*.yaml.in",
    "pyproject.toml",
    "scripts/**/*.py",
    "src/**/*.py",
    "tests/**/*.py",
    "third_party/Sana/**/*.py",
    "uv.lock",
)

_RUNTIME_DISTRIBUTIONS = (
    "Pillow",
    "PyYAML",
    "av",
    "diffusers",
    "einops",
    "flash-linear-attention",
    "ftfy",
    "h5py",
    "imageio",
    "mmcv-lite",
    "mmengine",
    "numpy",
    "opencv-python",
    "omegaconf",
    "protobuf",
    "qwen-vl-utils",
    "regex",
    "safetensors",
    "scipy",
    "sentencepiece",
    "timm",
    "torch",
    "torchvision",
    "tqdm",
    "transformers",
    "triton",
)

_REFERENCE_PRECOMPUTE_ARTIFACT_ROLES = (
    "registry",
    "protocol",
    "storage_amendment",
    "external_components_manifest",
    "expansion_eligibility_amendment",
    "padding_semantics_amendment",
    "task_plan",
    "dataset_contract",
    "runtime_support_manifest",
    "projection_equivalence_receipt",
    "recovery_authority",
)
_SMOKE_ARTIFACT_ROLES = _REFERENCE_PRECOMPUTE_ARTIFACT_ROLES + (
    "reference_preflight_request",
    "reference_preflight_report",
    "reference_rebind_receipt",
    "action_reference",
    "action_reference_spot_check",
    "arm_projection_manifest",
)
_TRAINING_ARTIFACT_ROLES = _SMOKE_ARTIFACT_ROLES + (
    "smoke_preflight_request",
    "smoke_preflight_report",
    "smoke_rebind_receipt",
    "evidence_rebinding_receipt",
    "real_2b_smoke_gate",
)
_RUNTIME_ROLES = (
    "student_checkpoint",
    "phase1_checkpoint",
    "action_stats",
)

_REGISTRY_KEYS = frozenset(
    {
        "protocol",
        "registered_before_phase6_training_results",
        "seed",
        "initial_checkpoint",
        "risk_reference_checkpoint",
        "action_stats_sha256",
        "data",
        "deployment_contract",
        "time_conditioning",
        "expansion_constraint",
        "action_interface",
        "action_non_regression",
        "arms",
        "common_training",
        "primary_gates",
        "output_root",
        "constraints",
    }
)
_STORAGE_KEYS = frozenset(
    {
        "amendment",
        "effective_output_root",
        "original_output_root",
        "reason",
        "registered_before_phase6_training_results",
        "registered_protocol_sha256",
        "scientific_contract_changed",
        "storage_checks",
    }
)
_EXPANSION_AMENDMENT_KEYS = frozenset(
    {
        "candidate_pool_policy",
        "constraints",
        "execution_state",
        "expansion_support_contract",
        "expansion_support_contract_sha256",
        "incident",
        "old_plan_structural_audit",
        "reason",
        "registered_before_phase6_training_results",
        "registered_contract_pins",
        "schema_version",
        "scientific_factor_changed",
        "superseded_plan_pins",
    }
)
_EXPANSION_CANDIDATE_POOL_KEYS = frozenset(
    {
        "action_sigma_schedule_changed",
        "eligible_candidate_count",
        "excluded_candidate_count",
        "excluded_per_task",
        "ordinary_dataset_global_indices_changed",
        "protocol_seed_changed",
        "raw_candidate_count",
        "runtime_skip_allowed",
        "slot_specific_backfill_allowed",
        "strategy",
        "task_count",
        "task_order_changed",
    }
)
_PADDING_AMENDMENT_KEYS = frozenset(
    {
        "constraints",
        "discovery",
        "eligibility_dependency",
        "execution_state",
        "implementation_requirements",
        "padding_semantics_contract",
        "padding_semantics_contract_hash_serialization",
        "padding_semantics_contract_sha256",
        "reason",
        "registered_before_phase6_training_results",
        "registered_contract_pins",
        "schema_version",
        "scientific_factor_changed",
        "source_manifest_at_discovery_sha256",
        "uniformity",
    }
)
_PADDING_CONSTRAINT_KEYS = frozenset(
    {
        "best_of_n",
        "closed_loop",
        "critics",
        "dagger",
        "recovery_annotations",
        "reranking",
        "task_specific_sampler",
    }
)
_TASK_ARTIFACT_KEYS = frozenset(
    {
        "artifact_schema_version",
        "expansion_eligibility_amendment_sha256",
        "expansion_support_contract",
        "expansion_support_contract_sha256",
        "identity_sha256",
        "plan",
        "plan_sha256",
    }
)
_PLAN_KEYS = frozenset(
    {
        "action_sigmas",
        "holdout_task_sha256",
        "holdout_tasks",
        "optimizer_steps",
        "protocol_seed",
        "rows",
        "schema_version",
        "seed_domains",
        "sigma_exposures_per_task",
        "source_contract",
        "task_cycles",
        "train_task_sha256",
        "train_tasks",
    }
)
_PLAN_ROW_KEYS = frozenset(
    {
        "action_sigma",
        "cycle",
        "domain_seeds",
        "global_step",
        "identity",
        "position_in_cycle",
    }
)
_PLAN_IDENTITY_KEYS = frozenset(
    {
        "dataset_index",
        "episode_index",
        "episode_path",
        "prompt",
        "source_dataset",
        "source_kind",
        "source_variant",
        "start_frame",
        "task_name",
    }
)
_DATASET_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "plan_sha256",
        "identity_sha256",
        "action_stats_sha256",
        "expansion_eligibility_amendment_sha256",
        "expansion_support_contract",
        "expansion_support_contract_sha256",
        "padding_semantics_amendment_sha256",
        "padding_semantics_contract",
        "padding_semantics_contract_sha256",
        "preprocessing_contract",
        "rows",
        "episodes",
    }
)
_DATASET_ROW_KEYS = frozenset(
    {
        "global_step",
        "dataset_index",
        "task_name",
        "episode_index",
        "episode_path",
        "start_frame",
        "window_logical_length",
        "episode_sha256",
        "instruction_source_sha256",
        *EXPANSION_SUPPORT_KEYS,
    }
)
_REFERENCE_KEYS = frozenset(
    {
        "action_stats_sha256",
        "dataset_contract_artifact_sha256",
        "identity_sha256",
        "plan_sha256",
        "precompute_config",
        "precompute_config_file_sha256",
        "precompute_config_sha256",
        "precompute_gpu_runtime",
        "precompute_gpu_runtime_sha256",
        "precompute_preflight_report_sha256",
        "precompute_preflight_request_sha256",
        "precompute_source_manifest_sha256",
        "reference_checkpoint_sha256",
        "rows",
        "schema_version",
    }
)
_REFERENCE_ROW_KEYS = frozenset(
    {
        "action_sigma",
        "common_input_trace_sha256",
        "dataset_contract_row_sha256",
        "dataset_index",
        "error_float32_bits",
        "global_step",
        "plan_row_sha256",
        "reference_forward_context_sha256",
        "t0_reference_forward_trace_sha256",
        "task_name",
    }
)
_SPOT_KEYS = frozenset(
    {
        "schema_version",
        "verification_passed",
        "gpu_forward_executed",
        "training_started",
        "closed_loop",
        "reference_artifact_sha256",
        "plan_sha256",
        "identity_sha256",
        "dataset_contract_artifact_sha256",
        "reference_checkpoint_sha256",
        "action_stats_sha256",
        "precompute_config_file_sha256",
        "precompute_config_sha256",
        "precompute_gpu_runtime_sha256",
        "precompute_preflight_report_sha256",
        "precompute_preflight_request_sha256",
        "precompute_source_manifest_sha256",
        "spot_gpu_runtime",
        "spot_gpu_runtime_sha256",
        "spot_preflight_report_sha256",
        "spot_preflight_request_sha256",
        "global_steps",
        "rows",
    }
)
_SPOT_ROW_KEYS = frozenset(
    {
        "global_step",
        "expected_error_float32_bits",
        "observed_error_float32_bits",
        "exact_match",
        "plan_row_sha256",
        "dataset_contract_row_sha256",
        "common_input_trace_sha256",
        "reference_forward_context_sha256",
        "t0_reference_forward_trace_sha256",
    }
)

_REFERENCE_PRECOMPUTE_AUTHORIZATION = {
    "operation": "phase1_action_reference_v2_precompute",
    "arm": "T0_E0A0",
    "continuous_time": False,
    "expansion": False,
    "action_adapter_nr": False,
    "lambda_action": 0.0,
}
_SMOKE_AUTHORIZATION = {
    "operation": "phase6_real_2b_smoke",
    "paired_global_step": 1,
    "arms": [
        "T0_E0A0",
        "T1_E0A0",
        "T1_E1A0",
        "T1_E0A1",
        "T1_E1A1",
    ],
    "backward_arm": "T1_E1A1",
    "checkpoint_roundtrip_arm": "T1_E1A1",
    "identity_pair": ["T1_E0A0", "T1_E0A1"],
    "memory_limit_bytes": 139_586_437_120,
    "reference_gpu_forward_completed": True,
    "smoke_gpu_forward_allowed": True,
    "backward_allowed": True,
    "temporary_checkpoint_roundtrip_allowed": True,
    "optimizer_creation_allowed": False,
    "optimizer_step_allowed": False,
    "formal_training_allowed": False,
    "closed_loop_allowed": False,
}
_TRAINING_AUTHORIZATION_KEYS = frozenset(
    {
        "operation",
        "arm",
        "factors",
        "arm_config_projection_sha256",
        "run_id",
        "run_directory",
        "reference_gpu_forward_completed",
        "real_2b_smoke_completed",
        "optimizer_training_allowed",
        "closed_loop_allowed",
        "resume_allowed",
    }
)


class PreflightValidationError(ValueError):
    """Raised before expensive work when any Phase-6 binding is invalid."""


@dataclass(frozen=True)
class _FrozenPolicy:
    fixed_artifact_sha256: tuple[tuple[str, str], ...]
    runtime_sha256: tuple[tuple[str, str], ...]
    external_components: tuple[tuple[str, int, str], ...]
    effective_output_root: str
    original_output_root: str
    plan_sha256: str
    identity_sha256: str
    optimizer_steps: int
    seed: int
    runtime_support_files: tuple[tuple[str, int, str], ...] = ()


_FROZEN_POLICY = _FrozenPolicy(
    fixed_artifact_sha256=(
        (
            "registry",
            "9166e1ecde74f14d7d69ae2f5072b10732b01b9f3b0004c438d0e97ed94b00b1",
        ),
        (
            "protocol",
            "765fcc884cf4cb168fac676de55801667b561fde2cc0cf98a82007ef5469d879",
        ),
        (
            "storage_amendment",
            "e08dd7ef33f381a34808cc166e5eae90d7ec20fd9956b481a7bf984b5a8e5c8f",
        ),
        (
            "external_components_manifest",
            "6f886e00843527ec1933e2f4ebd03dd1e7bf0069e5ce0892bd834b893a6a1b2a",
        ),
        (
            "expansion_eligibility_amendment",
            EXPANSION_ELIGIBILITY_AMENDMENT_SHA256,
        ),
        (
            "padding_semantics_amendment",
            PADDING_SEMANTICS_AMENDMENT_SHA256,
        ),
        (
            "task_plan",
            TASK_PLAN_ARTIFACT_SHA256,
        ),
        (
            "dataset_contract",
            DATASET_CONTRACT_ARTIFACT_SHA256,
        ),
    ),
    runtime_sha256=(
        (
            "student_checkpoint",
            "e9549aff484da56eada21972f00fb3aa2f95f8a65397a55cf0f38c79e6254137",
        ),
        (
            "phase1_checkpoint",
            "ba7bb59fe7e44f6efd6a4cef79f66aff7ccf5dbe82a5a166634e35055e94f089",
        ),
        (
            "action_stats",
            "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc",
        ),
    ),
    external_components=(
        (
            "/DATA/share/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth",
            8_245_160_990,
            "052c4022488949153fe6400631725bcb8c739362a6d34edeab26236d8a23f1e7",
        ),
        (
            "/DATA/share/SANA-Video_2B_480p/vae/Wan2.1_VAE.pth",
            507_609_880,
            "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
        ),
        (
            "/DATA/share/gemma-2-2b-it/model-00001-of-00002.safetensors",
            4_988_025_760,
            "532d792c9178805064170a3ec485b7dedbfccc6fd297b92c31a6091b6c7e41bf",
        ),
        (
            "/DATA/share/gemma-2-2b-it/model-00002-of-00002.safetensors",
            240_691_728,
            "6d6d9ce84db398fb6e0191f91542e5da0a73da2cb695e172a24edc2146dc8d20",
        ),
        (
            "/DATA/share/gemma-2-2b-it/model.safetensors.index.json",
            24_223,
            "ada0043f3e3b2e5ab2f445cad9c0fbbf9d91ad444675e6a82b822591c63abf5a",
        ),
        (
            "/DATA/share/gemma-2-2b-it/tokenizer.json",
            17_525_357,
            "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922",
        ),
        (
            "/DATA/share/gemma-2-2b-it/tokenizer.model",
            4_241_003,
            "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
        ),
    ),
    effective_output_root="/DATA/share/sana_phase6_principled_constraints_20260724",
    original_output_root="/DATA/zch/sana_phase6_principled_constraints_20260724",
    plan_sha256=EMBEDDED_PLAN_SHA256,
    identity_sha256=IDENTITY_SHA256,
    optimizer_steps=504,
    seed=20260724,
    runtime_support_files=(
        (
            "/DATA/share/SANA-Video_2B_480p/config.json",
            51,
            "d34650e1e7b0ab502051e8fe2e829ed2c93bbcad0aa9fa3ab46574491b3e8288",
        ),
        (
            "/DATA/share/gemma-2-2b-it/config.json",
            838,
            "eacec6c5ca317a87ed2c46789d9705b9274db5027e7ba59da739bfae23addb55",
        ),
        (
            "/DATA/share/gemma-2-2b-it/generation_config.json",
            187,
            "a543a5d299bc2b20c52bd87ed174f561266510b57a392e12b5b5d758d798ce05",
        ),
        (
            "/DATA/share/gemma-2-2b-it/special_tokens_map.json",
            636,
            "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
        ),
        (
            "/DATA/share/gemma-2-2b-it/tokenizer_config.json",
            46_996,
            "cb32b7929c62608d46572e813112b3ad8a841fb98fdd6a4da8559e368a951c89",
        ),
    ),
)


def canonical_json_bytes(value: Any, *, ensure_ascii: bool = False) -> bytes:
    """Encode compact, sorted canonical JSON with exactly one trailing newline."""

    try:
        encoding = "ascii" if ensure_ascii else "utf-8"
        return (
            json.dumps(
                value,
                ensure_ascii=ensure_ascii,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(encoding)
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PreflightValidationError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PreflightValidationError(
            f"value is not pretty-canonical JSON: {exc}"
        ) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PreflightValidationError(f"non-finite JSON number: {value}")
    return parsed


def _strict_json_loads(data: bytes, name: str) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_parse_finite_float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PreflightValidationError(f"non-finite JSON value: {value}")
            ),
        )
    except PreflightValidationError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError, ValueError) as exc:
        raise PreflightValidationError(f"invalid {name} JSON: {exc}") from exc


def _require_exact_keys(value: Any, expected: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise PreflightValidationError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != set(expected):
        raise PreflightValidationError(
            f"{name} keys differ; missing={sorted(set(expected) - actual)!r}, "
            f"extra={sorted(actual - set(expected))!r}"
        )


def _validate_recovery_lifecycle_fields(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise PreflightValidationError(f"{name} must be a mapping")
    for key, expected in recovery_lifecycle().items():
        observed = value.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise PreflightValidationError(f"{name}.{key} differs")
    if "closed_loop_started" in value:
        raise PreflightValidationError(f"{name} uses the legacy closed-loop field")


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PreflightValidationError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise PreflightValidationError(f"{name} must be a JSON boolean")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PreflightValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, name: str, *, reject_placeholder: bool) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PreflightValidationError(
            f"{name} must be a lowercase 64-hex SHA256 digest"
        )
    placeholders = {character * 64 for character in "0123456789abcdef"} | {
        "0123456789abcdef" * 4,
        "deadbeef" * 8,
    }
    if reject_placeholder and value in placeholders:
        raise PreflightValidationError(f"{name} is a placeholder digest")
    return value


def _require_absolute_path(value: Any, name: str) -> str:
    path = _require_string(value, name)
    if not os.path.isabs(path):
        raise PreflightValidationError(f"{name} must be an absolute path")
    return path


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _observe_file(
    declared_path: str,
    name: str,
    *,
    capture: bool = False,
    expected_size: int | None = None,
    file_hasher: Callable[[str], str] | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    """Hash a regular file while binding its spelling, target, and stat identity."""

    path = _require_absolute_path(declared_path, f"{name}.path")
    absolute = os.path.abspath(path)
    try:
        resolved_before = os.path.realpath(absolute)
        lstat_before = os.lstat(absolute)
        stat_before = os.stat(absolute)
    except OSError as exc:
        raise PreflightValidationError(
            f"cannot stat {name}: {absolute}: {exc}"
        ) from exc
    if not stat.S_ISREG(stat_before.st_mode):
        raise PreflightValidationError(f"{name} is not a regular file: {absolute}")
    if expected_size is not None and stat_before.st_size != expected_size:
        raise PreflightValidationError(
            f"{name} size mismatch: expected {expected_size}, got {stat_before.st_size}"
        )
    if capture and stat_before.st_size > _METADATA_SIZE_LIMIT:
        raise PreflightValidationError(
            f"{name} exceeds metadata size limit {_METADATA_SIZE_LIMIT}"
        )
    if capture and file_hasher is not None:
        raise PreflightValidationError("injected file_hasher cannot capture metadata")

    data: bytes | None = None
    try:
        with open(absolute, "rb") as stream:
            descriptor_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(descriptor_before.st_mode):
                raise PreflightValidationError(
                    f"{name} descriptor is not a regular file: {absolute}"
                )
            if (
                descriptor_before.st_dev,
                descriptor_before.st_ino,
            ) != (stat_before.st_dev, stat_before.st_ino):
                raise PreflightValidationError(
                    f"{name} target changed before hashing: {absolute}"
                )
            if file_hasher is None:
                digest = sha256()
                chunks: list[bytes] | None = [] if capture else None
                for block in iter(lambda: stream.read(_HASH_BLOCK_SIZE), b""):
                    digest.update(block)
                    if chunks is not None:
                        chunks.append(block)
                observed_sha256 = digest.hexdigest()
                if chunks is not None:
                    data = b"".join(chunks)
            else:
                observed_sha256 = file_hasher(absolute)
            descriptor_after = os.fstat(stream.fileno())
    except PreflightValidationError:
        raise
    except OSError as exc:
        raise PreflightValidationError(
            f"cannot hash {name}: {absolute}: {exc}"
        ) from exc

    _require_sha256(
        observed_sha256, f"observed {name} SHA256", reject_placeholder=False
    )
    try:
        resolved_after = os.path.realpath(absolute)
        lstat_after = os.lstat(absolute)
        stat_after = os.stat(absolute)
    except OSError as exc:
        raise PreflightValidationError(
            f"cannot restat {name} after hashing: {absolute}: {exc}"
        ) from exc
    stable_before = (
        resolved_before,
        _stat_signature(lstat_before),
        _stat_signature(stat_before),
        _stat_signature(descriptor_before),
    )
    stable_after = (
        resolved_after,
        _stat_signature(lstat_after),
        _stat_signature(stat_after),
        _stat_signature(descriptor_after),
    )
    if stable_before != stable_after:
        raise PreflightValidationError(f"{name} changed while hashing: {absolute}")
    return (
        {
            "path": path,
            "absolute_path": absolute,
            "realpath": resolved_before,
            "size_bytes": stat_before.st_size,
            "sha256": observed_sha256,
        },
        data,
    )


class _StableHashMemo:
    """Reuse large-file hashes only while path identity and metadata stay exact."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[Any, ...]] = {}

    def __call__(self, path: str) -> str:
        absolute = os.path.abspath(path)
        try:
            current = (
                os.path.realpath(absolute),
                _stat_signature(os.lstat(absolute)),
                _stat_signature(os.stat(absolute)),
            )
        except OSError as exc:
            raise PreflightValidationError(
                f"cannot stat memoized hash target {absolute}: {exc}"
            ) from exc
        cached = self._cache.get(absolute)
        if cached is not None:
            if current != cached[:3]:
                raise PreflightValidationError(
                    f"memoized hash target changed during preflight: {absolute}"
                )
            return cached[3]

        observed, _ = _observe_file(absolute, "memoized hash target")
        try:
            stable = (
                os.path.realpath(absolute),
                _stat_signature(os.lstat(absolute)),
                _stat_signature(os.stat(absolute)),
            )
        except OSError as exc:
            raise PreflightValidationError(
                f"cannot restat memoized hash target {absolute}: {exc}"
            ) from exc
        if current != stable:
            raise PreflightValidationError(
                f"memoized hash target changed while hashing: {absolute}"
            )
        self._cache[absolute] = (*stable, observed["sha256"])
        return observed["sha256"]


class _TargetRegistry:
    def __init__(self) -> None:
        self._labels: dict[str, str] = {}

    def add(self, observed: Mapping[str, Any], label: str) -> None:
        key = os.path.normcase(os.path.normpath(observed["realpath"]))
        previous = self._labels.get(key)
        if previous is not None:
            raise PreflightValidationError(
                f"duplicate real file target for {previous} and {label}: "
                f"{observed['realpath']}"
            )
        self._labels[key] = label


def _check_registration_flags(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        if "registered_before_phase6_training_results" in value:
            if value["registered_before_phase6_training_results"] is not True:
                raise PreflightValidationError(
                    f"{name}.registered_before_phase6_training_results must be true"
                )
        if "training_or_gpu_started" in value:
            if value["training_or_gpu_started"] is not False:
                raise PreflightValidationError(
                    f"{name}.training_or_gpu_started must be false"
                )
        for key, child in value.items():
            _check_registration_flags(child, f"{name}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_registration_flags(child, f"{name}[{index}]")


def _validate_registry(value: Any, policy: _FrozenPolicy) -> None:
    _require_exact_keys(value, _REGISTRY_KEYS, "registry")
    if value["registered_before_phase6_training_results"] is not True:
        raise PreflightValidationError("registry registration flag must be true")
    if value["seed"] != policy.seed:
        raise PreflightValidationError("registry seed differs from frozen seed")
    if value["output_root"] != policy.original_output_root:
        raise PreflightValidationError("registry original output root differs")
    runtime = dict(policy.runtime_sha256)
    for key, role, expected_role in (
        ("initial_checkpoint", "student_checkpoint", "student initialization"),
        (
            "risk_reference_checkpoint",
            "phase1_checkpoint",
            "one-sided action-function reference",
        ),
    ):
        checkpoint = value[key]
        _require_exact_keys(
            checkpoint, frozenset({"role", "path", "sha256"}), f"registry.{key}"
        )
        if checkpoint["role"] != expected_role:
            raise PreflightValidationError(f"registry.{key}.role differs")
        if checkpoint["sha256"] != runtime[role]:
            raise PreflightValidationError(f"registry.{key}.sha256 differs")
        _require_string(checkpoint["path"], f"registry.{key}.path")
    if value["action_stats_sha256"] != runtime["action_stats"]:
        raise PreflightValidationError("registry action-stats SHA differs")
    if not isinstance(value["data"], Mapping):
        raise PreflightValidationError("registry.data must be an object")
    if value["data"].get("optimizer_steps") != policy.optimizer_steps:
        raise PreflightValidationError("registry optimizer_steps differs")


def _validate_storage(value: Any, policy: _FrozenPolicy) -> None:
    _require_exact_keys(value, _STORAGE_KEYS, "storage amendment")
    fixed = dict(policy.fixed_artifact_sha256)
    if value["effective_output_root"] != policy.effective_output_root:
        raise PreflightValidationError("storage effective output root differs")
    if value["original_output_root"] != policy.original_output_root:
        raise PreflightValidationError("storage original output root differs")
    if value["registered_before_phase6_training_results"] is not True:
        raise PreflightValidationError("storage registration flag must be true")
    if value["scientific_contract_changed"] is not False:
        raise PreflightValidationError("storage scientific-contract flag must be false")
    # This historical field is misnamed but byte-frozen to the registry digest.
    if value["registered_protocol_sha256"] != fixed["registry"]:
        raise PreflightValidationError(
            "storage legacy registered_protocol_sha256 must equal registry SHA256"
        )
    checks = value["storage_checks"]
    _require_exact_keys(
        checks,
        frozenset(
            {
                "effective_root_parent_writable",
                "filesystem_available_bytes_class",
                "original_root_parent_writable",
            }
        ),
        "storage amendment.storage_checks",
    )
    if checks["effective_root_parent_writable"] is not True:
        raise PreflightValidationError("registered effective parent must be writable")
    if checks["original_root_parent_writable"] is not False:
        raise PreflightValidationError("registered original parent must be unwritable")


def _require_exact_typed_value(value: Any, expected: Any, name: str) -> None:
    if type(value) is not type(expected):
        raise PreflightValidationError(f"{name} has an inexact JSON type")
    if isinstance(expected, dict):
        _require_exact_keys(value, frozenset(expected), name)
        for key, child in expected.items():
            _require_exact_typed_value(value[key], child, f"{name}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise PreflightValidationError(f"{name} has an unexpected length")
        for index, (child, expected_child) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_exact_typed_value(child, expected_child, f"{name}[{index}]")
    elif value != expected:
        raise PreflightValidationError(f"{name} differs from its frozen value")


def _validate_amendment_execution(value: Any, name: str) -> None:
    _require_exact_typed_value(
        value,
        {
            "closed_loop_started": False,
            "formal_training_started": False,
            "optimizer_steps": 0,
            "training_started": False,
        },
        f"{name}.execution_state",
    )


def _validate_registered_contract_pins(
    value: Any, policy: _FrozenPolicy, name: str
) -> None:
    fixed = dict(policy.fixed_artifact_sha256)
    _require_exact_typed_value(
        value,
        {
            "external_components_sha256": fixed["external_components_manifest"],
            "protocol_sha256": fixed["protocol"],
            "registry_sha256": fixed["registry"],
            "storage_amendment_sha256": fixed["storage_amendment"],
        },
        f"{name}.registered_contract_pins",
    )


def _validate_expansion_amendment(value: Any, policy: _FrozenPolicy) -> dict[str, Any]:
    name = "expansion eligibility amendment"
    _require_exact_keys(value, _EXPANSION_AMENDMENT_KEYS, name)
    if value["schema_version"] != "sana-phase6-expansion-eligibility-amendment-v1":
        raise PreflightValidationError(f"{name} schema differs")
    if (
        value["registered_before_phase6_training_results"] is not True
        or value["scientific_factor_changed"] is not False
    ):
        raise PreflightValidationError(f"{name} lifecycle flags differ")
    _validate_amendment_execution(value["execution_state"], name)
    _validate_registered_contract_pins(value["registered_contract_pins"], policy, name)
    constraints = value["constraints"]
    _require_exact_keys(
        constraints,
        frozenset(
            {
                "best_of_n",
                "closed_loop",
                "critics",
                "dagger",
                "outcome_or_model_based_sampler_selection",
                "recovery_annotations",
                "reranking",
                "task_specific_filter",
            }
        ),
        f"{name}.constraints",
    )
    if any(
        type(constraints[key]) is not bool or constraints[key] for key in constraints
    ):
        raise PreflightValidationError(f"{name} forbidden constraints differ")
    contract = value["expansion_support_contract"]
    expected_contract = expansion_support_contract()
    _require_exact_typed_value(contract, expected_contract, f"{name}.contract")
    if (
        sha256(canonical_json_bytes(contract)).hexdigest()
        != EXPANSION_SUPPORT_CONTRACT_SHA256
        or value["expansion_support_contract_sha256"]
        != EXPANSION_SUPPORT_CONTRACT_SHA256
    ):
        raise PreflightValidationError(f"{name} nested contract SHA differs")
    pool = value["candidate_pool_policy"]
    _require_exact_keys(
        pool,
        _EXPANSION_CANDIDATE_POOL_KEYS,
        f"{name}.candidate_pool_policy",
    )
    for key, expected in (
        ("raw_candidate_count", 467_038),
        ("eligible_candidate_count", 460_738),
        ("excluded_candidate_count", 6_300),
        ("task_count", 42),
    ):
        if (
            _require_int(pool.get(key), f"{name}.candidate_pool_policy.{key}")
            != expected
        ):
            raise PreflightValidationError(f"{name} candidate-pool count differs")
    if (
        _require_int(
            pool.get("excluded_per_task"),
            f"{name}.candidate_pool_policy.excluded_per_task",
        )
        != 150
    ):
        raise PreflightValidationError(f"{name} per-task exclusions differ")
    for key in (
        "action_sigma_schedule_changed",
        "ordinary_dataset_global_indices_changed",
        "protocol_seed_changed",
        "runtime_skip_allowed",
        "slot_specific_backfill_allowed",
        "task_order_changed",
    ):
        if pool.get(key) is not False:
            raise PreflightValidationError(f"{name} {key} must be false")
    _require_string(pool["strategy"], f"{name}.candidate_pool_policy.strategy")
    return dict(expected_contract)


def _validate_padding_amendment(
    value: Any,
    policy: _FrozenPolicy,
    *,
    expansion_contract: Mapping[str, Any],
) -> dict[str, Any]:
    name = "padding semantics amendment"
    _require_exact_keys(value, _PADDING_AMENDMENT_KEYS, name)
    if value["schema_version"] != "sana-phase6-padding-semantics-amendment-v1":
        raise PreflightValidationError(f"{name} schema differs")
    if (
        value["registered_before_phase6_training_results"] is not True
        or value["scientific_factor_changed"] is not False
    ):
        raise PreflightValidationError(f"{name} lifecycle flags differ")
    _validate_amendment_execution(value["execution_state"], name)
    _validate_registered_contract_pins(value["registered_contract_pins"], policy, name)
    constraints = value["constraints"]
    _require_exact_keys(
        constraints,
        _PADDING_CONSTRAINT_KEYS,
        f"{name}.constraints",
    )
    if any(
        type(constraints[key]) is not bool or constraints[key] for key in constraints
    ):
        raise PreflightValidationError(f"{name} forbidden constraints differ")
    try:
        contract = validate_padding_semantics_contract(
            value["padding_semantics_contract"]
        )
        dependency = value["eligibility_dependency"]
        _require_exact_keys(
            dependency,
            frozenset({"amendment_sha256", "support_contract_sha256"}),
            f"{name}.eligibility_dependency",
        )
        validate_padding_eligibility_dependency(
            expansion_eligibility_amendment_sha256=dependency["amendment_sha256"],
            expansion_support_contract_sha256=dependency["support_contract_sha256"],
        )
    except (PaddingSemanticsError, TypeError) as exc:
        raise PreflightValidationError(
            f"{name} contract/dependency differs: {exc}"
        ) from exc
    if (
        value["padding_semantics_contract_sha256"] != PADDING_SEMANTICS_CONTRACT_SHA256
        or sha256(canonical_json_bytes(contract, ensure_ascii=True)).hexdigest()
        != PADDING_SEMANTICS_CONTRACT_SHA256
    ):
        raise PreflightValidationError(f"{name} nested contract SHA differs")
    if expansion_contract != expansion_support_contract():
        raise PreflightValidationError(f"{name} expansion dependency value differs")
    requirements = value["implementation_requirements"]
    if (
        type(requirements) is not dict
        or not requirements
        or any(type(flag) is not bool or not flag for flag in requirements.values())
    ):
        raise PreflightValidationError(f"{name} implementation requirements differ")
    return contract


def _validate_external_manifest(
    value: Any,
    policy: _FrozenPolicy,
) -> None:
    _require_exact_keys(
        value,
        frozenset(
            {
                "components",
                "registered_before_phase6_training_results",
                "schema_version",
                "training_or_gpu_started",
            }
        ),
        "external-components manifest",
    )
    if value["schema_version"] != "sana-phase6-external-components-v1":
        raise PreflightValidationError("external-components schema differs")
    if value["registered_before_phase6_training_results"] is not True:
        raise PreflightValidationError(
            "external-components registration flag must be true"
        )
    if value["training_or_gpu_started"] is not False:
        raise PreflightValidationError("external-components GPU flag must be false")
    expected = [
        {"path": path, "sha256": digest, "size_bytes": size}
        for path, size, digest in policy.external_components
    ]
    if value["components"] != expected:
        raise PreflightValidationError(
            "external-components list/order/path/size/SHA differs from frozen policy"
        )


def build_phase6_runtime_support_manifest() -> dict[str, Any]:
    """Build the frozen manifest for small runtime configuration files."""

    return {
        "files": [
            {"path": path, "sha256": digest, "size_bytes": size}
            for path, size, digest in _FROZEN_POLICY.runtime_support_files
        ],
        "registered_before_phase6_training_results": True,
        "schema_version": RUNTIME_SUPPORT_SCHEMA_VERSION,
        "training_or_gpu_started": False,
    }


def _validate_runtime_support_manifest(
    value: Any,
    policy: _FrozenPolicy,
    *,
    targets: _TargetRegistry,
    file_hasher: Callable[[str], str] | None,
) -> list[dict[str, Any]]:
    _require_exact_keys(
        value,
        frozenset(
            {
                "files",
                "registered_before_phase6_training_results",
                "schema_version",
                "training_or_gpu_started",
            }
        ),
        "runtime-support manifest",
    )
    if value["schema_version"] != RUNTIME_SUPPORT_SCHEMA_VERSION:
        raise PreflightValidationError("runtime-support schema differs")
    if value["registered_before_phase6_training_results"] is not True:
        raise PreflightValidationError("runtime-support registration flag must be true")
    if value["training_or_gpu_started"] is not False:
        raise PreflightValidationError("runtime-support GPU flag must be false")
    expected = [
        {"path": path, "sha256": digest, "size_bytes": size}
        for path, size, digest in policy.runtime_support_files
    ]
    if not expected or value["files"] != expected:
        raise PreflightValidationError(
            "runtime-support file list/order/path/size/SHA differs"
        )
    observed_files = []
    for index, (path, size, digest) in enumerate(policy.runtime_support_files):
        observed, _ = _observe_file(
            path,
            f"runtime-support file {index}",
            expected_size=size,
            file_hasher=file_hasher,
        )
        if not hmac.compare_digest(observed["sha256"], digest):
            raise PreflightValidationError(f"runtime-support file {index} SHA mismatch")
        targets.add(observed, f"runtime-support:{index}")
        observed_files.append({"manifest_index": index, **observed})
    return observed_files


def _task_identity_sha256(plan: Mapping[str, Any]) -> str:
    payload = {
        "holdout_task_sha256": plan["holdout_task_sha256"],
        "ordered_identities": [
            {
                "episode_path": row["identity"]["episode_path"],
                "prompt": row["identity"]["prompt"],
                "start_frame": row["identity"]["start_frame"],
                "task_name": row["identity"]["task_name"],
            }
            for row in plan["rows"]
        ],
        "schema_version": "sana-phase6-identity-sequence-v1",
        "train_task_sha256": plan["train_task_sha256"],
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_task_plan(
    value: Any,
    policy: _FrozenPolicy,
    *,
    expansion_contract: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    _require_exact_keys(value, _TASK_ARTIFACT_KEYS, "task-plan artifact")
    if value["artifact_schema_version"] != "sana-phase6-task-plan-artifact-v2":
        raise PreflightValidationError("task-plan artifact schema differs")
    if (
        value["expansion_eligibility_amendment_sha256"]
        != EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        or value["expansion_support_contract_sha256"]
        != EXPANSION_SUPPORT_CONTRACT_SHA256
    ):
        raise PreflightValidationError("task-plan expansion provenance differs")
    _require_exact_typed_value(
        value["expansion_support_contract"],
        dict(expansion_contract),
        "task-plan expansion_support_contract",
    )
    if (
        sha256(canonical_json_bytes(value["expansion_support_contract"])).hexdigest()
        != EXPANSION_SUPPORT_CONTRACT_SHA256
    ):
        raise PreflightValidationError("task-plan support-contract SHA differs")
    if value["plan_sha256"] != policy.plan_sha256:
        raise PreflightValidationError("task-plan embedded plan SHA differs")
    if value["identity_sha256"] != policy.identity_sha256:
        raise PreflightValidationError("task-plan identity SHA differs")
    plan = value["plan"]
    if not isinstance(plan, Mapping):
        raise PreflightValidationError("task-plan payload must be an object")
    _require_exact_keys(plan, _PLAN_KEYS, "task-plan payload")
    if plan.get("schema_version") != "sana-phase6-task-plan-v1":
        raise PreflightValidationError("task-plan payload schema differs")
    if (
        _require_int(plan.get("optimizer_steps"), "task-plan optimizer_steps")
        != policy.optimizer_steps
    ):
        raise PreflightValidationError("task-plan optimizer_steps differs")
    for key in ("task_cycles", "sigma_exposures_per_task"):
        _require_int(plan.get(key), f"task-plan {key}", minimum=1)
    _require_int(plan.get("protocol_seed"), "task-plan protocol_seed", minimum=1)
    if (
        type(plan.get("action_sigmas")) is not list
        or any(type(sigma) is not float for sigma in plan["action_sigmas"])
        or plan["action_sigmas"] != [1.0, 0.9, 0.5]
    ):
        raise PreflightValidationError("task-plan action_sigmas differ")
    observed_plan_sha = sha256(canonical_json_bytes(plan)).hexdigest()
    if observed_plan_sha != policy.plan_sha256:
        raise PreflightValidationError("task-plan payload SHA differs")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != policy.optimizer_steps:
        raise PreflightValidationError("task-plan row count differs")
    for index, row in enumerate(rows, start=1):
        _require_exact_keys(row, _PLAN_ROW_KEYS, f"task-plan row {index}")
        if (
            _require_int(
                row.get("global_step"), f"task-plan row {index}.global_step", minimum=1
            )
            != index
        ):
            raise PreflightValidationError(f"task-plan row {index} is invalid")
        expected_cycle = (index - 1) // 42
        expected_position = (index - 1) % 42
        if (
            _require_int(row.get("cycle"), f"task-plan row {index}.cycle")
            != expected_cycle
            or _require_int(
                row.get("position_in_cycle"),
                f"task-plan row {index}.position_in_cycle",
            )
            != expected_position
        ):
            raise PreflightValidationError(f"task-plan row {index} position differs")
        identity = row.get("identity")
        _require_exact_keys(
            identity, _PLAN_IDENTITY_KEYS, f"task-plan row {index}.identity"
        )
        _require_string(identity.get("task_name"), f"task-plan row {index}.task_name")
        for key in ("dataset_index", "episode_index", "start_frame"):
            _require_int(identity.get(key), f"task-plan row {index}.identity.{key}")
        for key in (
            "episode_path",
            "prompt",
            "source_dataset",
            "source_kind",
            "source_variant",
        ):
            _require_string(identity.get(key), f"task-plan row {index}.identity.{key}")
        if (
            identity["source_dataset"] != "RoboTwin"
            or identity["source_kind"] != "ordinary_expert"
            or identity["source_variant"] != "clean_50"
        ):
            raise PreflightValidationError(f"task-plan row {index} source differs")
        if type(row.get("action_sigma")) is not float or row["action_sigma"] not in (
            1.0,
            0.9,
            0.5,
        ):
            raise PreflightValidationError(
                f"task-plan row {index}.action_sigma is invalid"
            )
        seeds = row.get("domain_seeds")
        if type(seeds) is not dict or any(
            type(seed) is not int or seed < 0 or seed >= 2**64
            for seed in seeds.values()
        ):
            raise PreflightValidationError(f"task-plan row {index} seeds differ")
    if _task_identity_sha256(plan) != policy.identity_sha256:
        raise PreflightValidationError("task-plan recomputed identity SHA differs")
    return rows


def _validate_dataset_contract(
    value: Any,
    policy: _FrozenPolicy,
    plan_rows: Sequence[Mapping[str, Any]],
    *,
    expansion_contract: Mapping[str, Any],
    padding_contract: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    _require_exact_keys(value, _DATASET_CONTRACT_KEYS, "dataset contract")
    if value["schema_version"] != "sana-phase6-dataset-contract-v2":
        raise PreflightValidationError("dataset-contract schema differs")
    runtime = dict(policy.runtime_sha256)
    expected_pins = {
        "plan_sha256": policy.plan_sha256,
        "identity_sha256": policy.identity_sha256,
        "action_stats_sha256": runtime["action_stats"],
    }
    for key, expected in expected_pins.items():
        if value[key] != expected:
            raise PreflightValidationError(f"dataset-contract embedded {key} differs")
    if (
        value["expansion_eligibility_amendment_sha256"]
        != EXPANSION_ELIGIBILITY_AMENDMENT_SHA256
        or value["expansion_support_contract_sha256"]
        != EXPANSION_SUPPORT_CONTRACT_SHA256
        or value["padding_semantics_amendment_sha256"]
        != PADDING_SEMANTICS_AMENDMENT_SHA256
        or value["padding_semantics_contract_sha256"]
        != PADDING_SEMANTICS_CONTRACT_SHA256
    ):
        raise PreflightValidationError("dataset-contract amendment provenance differs")
    _require_exact_typed_value(
        value["expansion_support_contract"],
        dict(expansion_contract),
        "dataset-contract expansion_support_contract",
    )
    _require_exact_typed_value(
        value["padding_semantics_contract"],
        dict(padding_contract),
        "dataset-contract padding_semantics_contract",
    )
    if (
        sha256(canonical_json_bytes(value["expansion_support_contract"])).hexdigest()
        != EXPANSION_SUPPORT_CONTRACT_SHA256
        or sha256(
            canonical_json_bytes(value["padding_semantics_contract"], ensure_ascii=True)
        ).hexdigest()
        != PADDING_SEMANTICS_CONTRACT_SHA256
    ):
        raise PreflightValidationError("dataset-contract nested contract SHA differs")
    try:
        validate_padding_eligibility_dependency(
            expansion_eligibility_amendment_sha256=value[
                "expansion_eligibility_amendment_sha256"
            ],
            expansion_support_contract_sha256=value[
                "expansion_support_contract_sha256"
            ],
        )
    except PaddingSemanticsError as exc:
        raise PreflightValidationError(
            f"dataset-contract padding dependency differs: {exc}"
        ) from exc
    preprocessing = value["preprocessing_contract"]
    if not isinstance(preprocessing, Mapping):
        raise PreflightValidationError(
            "dataset-contract preprocessing_contract must be an object"
        )
    if preprocessing.get("action_stats_sha256") != runtime["action_stats"]:
        raise PreflightValidationError(
            "dataset-contract preprocessing action-stats SHA differs"
        )
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != policy.optimizer_steps:
        raise PreflightValidationError("dataset-contract row count differs")
    for index, (row, plan_row) in enumerate(zip(rows, plan_rows, strict=True), start=1):
        _require_exact_keys(row, _DATASET_ROW_KEYS, f"dataset-contract row {index}")
        identity = plan_row["identity"]
        for key in ("global_step", "dataset_index", "episode_index", "start_frame"):
            _require_int(row[key], f"dataset-contract row {index}.{key}")
        for key in ("task_name", "episode_path"):
            _require_string(row[key], f"dataset-contract row {index}.{key}")
        for key, expected in (
            ("global_step", index),
            ("dataset_index", identity.get("dataset_index")),
            ("task_name", identity.get("task_name")),
            ("episode_index", identity.get("episode_index")),
            ("episode_path", identity.get("episode_path")),
            ("start_frame", identity.get("start_frame")),
        ):
            if row[key] != expected:
                raise PreflightValidationError(
                    f"dataset-contract row {index}.{key} differs from plan"
                )
        _require_int(
            row["window_logical_length"],
            f"dataset-contract row {index}.window_logical_length",
            minimum=2,
        )
        _require_sha256(
            row["episode_sha256"],
            f"dataset-contract row {index}.episode_sha256",
            reject_placeholder=False,
        )
        _require_sha256(
            row["instruction_source_sha256"],
            f"dataset-contract row {index}.instruction_source_sha256",
            reject_placeholder=False,
        )
        try:
            support = validate_expansion_support(
                {key: row[key] for key in EXPANSION_SUPPORT_KEYS},
                require_eligible=True,
            )
        except ExpansionSupportError as exc:
            raise PreflightValidationError(
                f"dataset-contract row {index} expansion support differs: {exc}"
            ) from exc
        if support["actual_valid_raw_frames"] > row["window_logical_length"]:
            raise PreflightValidationError(
                f"dataset-contract row {index} support exceeds logical window"
            )
    if not isinstance(value["episodes"], list) or not value["episodes"]:
        raise PreflightValidationError("dataset-contract episodes must be non-empty")
    return rows


def _validate_action_reference(
    value: Any,
    policy: _FrozenPolicy,
    plan_rows: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_contract_artifact_sha256: str,
    source_manifest_sha256: str,
    precompute_config_file_sha256: str,
    precompute_request_sha256: str,
    precompute_report_sha256: str,
) -> None:
    _require_exact_keys(value, _REFERENCE_KEYS, "action reference")
    if value["schema_version"] != "sana-phase6-action-reference-v2":
        raise PreflightValidationError("action-reference schema differs")
    runtime = dict(policy.runtime_sha256)
    expected_pins = {
        "plan_sha256": policy.plan_sha256,
        "identity_sha256": policy.identity_sha256,
        "dataset_contract_artifact_sha256": dataset_contract_artifact_sha256,
        "reference_checkpoint_sha256": runtime["phase1_checkpoint"],
        "action_stats_sha256": runtime["action_stats"],
        "precompute_source_manifest_sha256": source_manifest_sha256,
        "precompute_config_file_sha256": precompute_config_file_sha256,
        "precompute_preflight_request_sha256": precompute_request_sha256,
        "precompute_preflight_report_sha256": precompute_report_sha256,
    }
    for key, expected in expected_pins.items():
        if value[key] != expected:
            raise PreflightValidationError(f"action-reference embedded {key} differs")
    for key in (
        "precompute_config_sha256",
        "precompute_gpu_runtime_sha256",
    ):
        _require_sha256(value[key], f"action-reference {key}", reject_placeholder=True)
    precompute_config = value["precompute_config"]
    if not isinstance(precompute_config, Mapping):
        raise PreflightValidationError(
            "action-reference precompute_config must be an object"
        )
    if (
        sha256(canonical_json_bytes(precompute_config)).hexdigest()
        != value["precompute_config_sha256"]
    ):
        raise PreflightValidationError("action-reference precompute-config SHA differs")
    from sana_wam.train.action_reference_table import (
        canonical_json_bytes as reference_canonical_json_bytes,
        validate_gpu_runtime,
        validate_precompute_config,
    )

    try:
        gpu_runtime = validate_gpu_runtime(
            value["precompute_gpu_runtime"], "precompute_gpu_runtime"
        )
    except (TypeError, ValueError) as exc:
        raise PreflightValidationError(
            f"action-reference GPU-runtime descriptor differs: {exc}"
        ) from exc
    if (
        sha256(reference_canonical_json_bytes(gpu_runtime)).hexdigest()
        != value["precompute_gpu_runtime_sha256"]
    ):
        raise PreflightValidationError("action-reference GPU-runtime SHA differs")
    try:
        validate_precompute_config(
            precompute_config,
            reference_checkpoint_sha256=runtime["phase1_checkpoint"],
            action_stats_sha256=runtime["action_stats"],
            plan_sha256=policy.plan_sha256,
            identity_sha256=policy.identity_sha256,
            dataset_contract_artifact_sha256=(dataset_contract_artifact_sha256),
        )
    except (TypeError, ValueError) as exc:
        raise PreflightValidationError(
            f"action-reference precompute config differs: {exc}"
        ) from exc
    if precompute_config.get("schema_version") != (
        "sana-phase6-action-reference-precompute-config-v1"
    ):
        raise PreflightValidationError(
            "action-reference precompute-config schema differs"
        )
    model = precompute_config.get("model")
    dataloader = precompute_config.get("dataloader")
    training = precompute_config.get("training")
    execution = precompute_config.get("execution")
    if not all(
        isinstance(item, Mapping) for item in (model, dataloader, training, execution)
    ):
        raise PreflightValidationError(
            "action-reference precompute config sections differ"
        )
    video_backbone = model.get("video_backbone")
    architecture = model.get("architecture")
    if not isinstance(video_backbone, Mapping) or not isinstance(architecture, Mapping):
        raise PreflightValidationError("action-reference model config sections differ")
    adapter = architecture.get("action_video_memory_adapter")
    if (
        video_backbone.get("continuous_timestep_conditioning") is not False
        or type(architecture.get("video_local_expansion_weight")) is not float
        or architecture.get("video_local_expansion_weight") != 0.0
        or type(architecture.get("action_non_regression_weight")) is not float
        or architecture.get("action_non_regression_weight") != 0.0
        or not isinstance(adapter, Mapping)
        or adapter.get("enabled") is not False
    ):
        raise PreflightValidationError("action-reference must use exact T0/E0/A0 model")
    if (
        type(training.get("lambda_action")) is not float
        or training.get("lambda_action") != 0.0
        or type(training.get("lambda_video")) is not float
        or training.get("lambda_video") != 1.0
        or training.get("init_checkpoint_sha256") != runtime["phase1_checkpoint"]
        or training.get("phase1_reference_checkpoint_sha256")
        != runtime["phase1_checkpoint"]
        or training.get("phase6_dataset_contract_artifact_sha256")
        != dataset_contract_artifact_sha256
        or training.get("phase1_action_reference_precompute_mode") is not True
    ):
        raise PreflightValidationError("action-reference training config differs")
    expected_execution = {
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
    }
    if dict(execution) != expected_execution:
        raise PreflightValidationError("action-reference execution config differs")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != policy.optimizer_steps:
        raise PreflightValidationError("action-reference row count differs")
    provenance = {
        "action_stats_sha256": runtime["action_stats"],
        "dataset_contract_artifact_sha256": dataset_contract_artifact_sha256,
        "precompute_config_sha256": value["precompute_config_sha256"],
        "precompute_gpu_runtime_sha256": value["precompute_gpu_runtime_sha256"],
        "precompute_preflight_report_sha256": value[
            "precompute_preflight_report_sha256"
        ],
        "precompute_preflight_request_sha256": value[
            "precompute_preflight_request_sha256"
        ],
        "precompute_source_manifest_sha256": source_manifest_sha256,
        "reference_checkpoint_sha256": runtime["phase1_checkpoint"],
    }
    for index, (row, plan_row, dataset_row) in enumerate(
        zip(rows, plan_rows, dataset_rows, strict=True), start=1
    ):
        _require_exact_keys(row, _REFERENCE_ROW_KEYS, f"action-reference row {index}")
        identity = plan_row["identity"]
        expected = {
            "global_step": index,
            "task_name": identity["task_name"],
            "dataset_index": identity["dataset_index"],
            "action_sigma": plan_row["action_sigma"],
        }
        if any(row[key] != item for key, item in expected.items()):
            raise PreflightValidationError(
                f"action-reference row {index} differs from task plan"
            )
        bits = row["error_float32_bits"]
        if not isinstance(bits, str) or _FLOAT32_BITS_RE.fullmatch(bits) is None:
            raise PreflightValidationError(
                f"action-reference row {index} has invalid float32 bits"
            )
        error = struct.unpack(">f", bytes.fromhex(bits))[0]
        if not math.isfinite(error) or error < 0.0:
            raise PreflightValidationError(
                f"action-reference row {index} error must be finite/non-negative"
            )
        if bits == "80000000":
            raise PreflightValidationError(
                f"action-reference row {index} negative zero is forbidden"
            )
        plan_row_sha256 = sha256(canonical_json_bytes(plan_row)).hexdigest()
        dataset_row_sha256 = sha256(canonical_json_bytes(dataset_row)).hexdigest()
        common_trace_sha256 = _require_sha256(
            row["common_input_trace_sha256"],
            f"action-reference row {index}.common_input_trace_sha256",
            reject_placeholder=False,
        )
        t0_trace_sha256 = _require_sha256(
            row["t0_reference_forward_trace_sha256"],
            f"action-reference row {index}.t0_reference_forward_trace_sha256",
            reject_placeholder=False,
        )
        context = {
            **provenance,
            "plan_row_sha256": plan_row_sha256,
            "dataset_contract_row_sha256": dataset_row_sha256,
            "common_input_trace_sha256": common_trace_sha256,
            "t0_reference_forward_trace_sha256": t0_trace_sha256,
        }
        expected_hashes = {
            "plan_row_sha256": plan_row_sha256,
            "dataset_contract_row_sha256": dataset_row_sha256,
            "reference_forward_context_sha256": sha256(
                canonical_json_bytes(context)
            ).hexdigest(),
        }
        if any(row[key] != item for key, item in expected_hashes.items()):
            raise PreflightValidationError(
                f"action-reference row {index} provenance differs"
            )


def _validate_action_reference_spot_check(
    value: Any,
    reference: Mapping[str, Any],
    *,
    reference_artifact_sha256: str,
) -> None:
    _require_exact_keys(value, _SPOT_KEYS, "action-reference spot check")
    if value["schema_version"] != "sana-phase6-action-reference-live-spot-v2":
        raise PreflightValidationError("action-reference spot schema differs")
    if (
        value["verification_passed"] is not True
        or value["gpu_forward_executed"] is not True
        or value["training_started"] is not False
        or value["closed_loop"] is not False
    ):
        raise PreflightValidationError("action-reference spot lifecycle differs")
    expected_pins = {
        "reference_artifact_sha256": reference_artifact_sha256,
        "plan_sha256": reference["plan_sha256"],
        "identity_sha256": reference["identity_sha256"],
        "dataset_contract_artifact_sha256": reference[
            "dataset_contract_artifact_sha256"
        ],
        "reference_checkpoint_sha256": reference["reference_checkpoint_sha256"],
        "action_stats_sha256": reference["action_stats_sha256"],
        "precompute_config_file_sha256": reference["precompute_config_file_sha256"],
        "precompute_config_sha256": reference["precompute_config_sha256"],
        "precompute_gpu_runtime_sha256": reference["precompute_gpu_runtime_sha256"],
        "precompute_preflight_report_sha256": reference[
            "precompute_preflight_report_sha256"
        ],
        "precompute_preflight_request_sha256": reference[
            "precompute_preflight_request_sha256"
        ],
        "precompute_source_manifest_sha256": reference[
            "precompute_source_manifest_sha256"
        ],
        "spot_gpu_runtime_sha256": reference["precompute_gpu_runtime_sha256"],
        "spot_preflight_report_sha256": reference["precompute_preflight_report_sha256"],
        "spot_preflight_request_sha256": reference[
            "precompute_preflight_request_sha256"
        ],
    }
    if any(value[key] != expected for key, expected in expected_pins.items()):
        raise PreflightValidationError("action-reference spot provenance differs")
    from sana_wam.train.action_reference_table import (
        canonical_json_bytes as reference_canonical_json_bytes,
        validate_gpu_runtime,
    )

    try:
        gpu_runtime = validate_gpu_runtime(
            value["spot_gpu_runtime"], "spot_gpu_runtime"
        )
    except (TypeError, ValueError) as exc:
        raise PreflightValidationError(
            f"action-reference spot GPU-runtime differs: {exc}"
        ) from exc
    if (
        sha256(reference_canonical_json_bytes(gpu_runtime)).hexdigest()
        != value["spot_gpu_runtime_sha256"]
    ):
        raise PreflightValidationError("action-reference spot GPU-runtime SHA differs")
    steps = [1, 253, 504]
    if value["global_steps"] != steps:
        raise PreflightValidationError("action-reference spot steps differ")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != len(steps):
        raise PreflightValidationError("action-reference spot rows differ")
    reference_rows = reference["rows"]
    for row, step in zip(rows, steps, strict=True):
        _require_exact_keys(row, _SPOT_ROW_KEYS, "action-reference spot row")
        reference_row = reference_rows[step - 1]
        expected = {
            "global_step": step,
            "expected_error_float32_bits": reference_row["error_float32_bits"],
            "observed_error_float32_bits": reference_row["error_float32_bits"],
            "exact_match": True,
            "common_input_trace_sha256": reference_row["common_input_trace_sha256"],
            "plan_row_sha256": reference_row["plan_row_sha256"],
            "dataset_contract_row_sha256": reference_row["dataset_contract_row_sha256"],
            "reference_forward_context_sha256": reference_row[
                "reference_forward_context_sha256"
            ],
            "t0_reference_forward_trace_sha256": reference_row[
                "t0_reference_forward_trace_sha256"
            ],
        }
        if dict(row) != expected:
            raise PreflightValidationError(f"action-reference spot row {step} differs")


def _validate_output_root(path_value: Any, policy: _FrozenPolicy) -> dict[str, Any]:
    path = _require_absolute_path(path_value, "effective_output_root")
    if path != policy.effective_output_root:
        raise PreflightValidationError(
            "effective_output_root differs from the frozen storage amendment"
        )
    if path == policy.original_output_root or path.startswith(
        policy.original_output_root + os.sep
    ):
        raise PreflightValidationError(
            "fallback to the original /DATA/zch root is forbidden"
        )
    absolute = os.path.abspath(path)
    realpath = os.path.realpath(absolute)
    expected_realpath = os.path.realpath(policy.effective_output_root)
    if os.path.normcase(realpath) != os.path.normcase(expected_realpath):
        raise PreflightValidationError("effective output root resolves elsewhere")
    if os.path.normcase(realpath) != os.path.normcase(absolute):
        raise PreflightValidationError("effective output root may not be a symlink")
    if not os.path.isdir(absolute):
        raise PreflightValidationError(
            "effective output root must already be a directory"
        )
    parent = os.path.dirname(absolute)
    if not os.path.isdir(parent):
        raise PreflightValidationError("effective output parent is not a directory")
    parent_writable = os.access(parent, os.W_OK | os.X_OK)
    root_writable = os.access(absolute, os.W_OK | os.X_OK)
    if not parent_writable or not root_writable:
        raise PreflightValidationError(
            "effective output root and its parent must be writable/searchable"
        )
    return {
        "path": path,
        "absolute_path": absolute,
        "realpath": realpath,
        "parent_path": parent,
        "parent_realpath": os.path.realpath(parent),
        "parent_writable": parent_writable,
        "root_writable": root_writable,
    }


def _within_repository(path: str, repository_realpath: str) -> bool:
    try:
        common = os.path.commonpath([path, repository_realpath])
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(repository_realpath)


def _discover_source_paths(repository_root: str) -> list[str]:
    root = Path(repository_root)
    paths: set[str] = set()
    for pattern in _SOURCE_INVENTORY_GLOBS:
        for candidate in root.glob(pattern):
            if not candidate.is_file():
                raise PreflightValidationError(
                    f"source inventory match is not a regular file: {candidate}"
                )
            paths.add(candidate.relative_to(root).as_posix())
    if "scripts/train.py" not in paths:
        raise PreflightValidationError("source inventory must contain scripts/train.py")
    if not any(path.startswith("src/") for path in paths):
        raise PreflightValidationError("source inventory contains no src/**/*.py files")
    expected_templates = {
        f"configs/phase6/templates/train_phase6_{arm}.yaml.in"
        for arm in ("T0_E0A0", "T1_E0A0", "T1_E1A0", "T1_E0A1", "T1_E1A1")
    }
    observed_templates = {
        path for path in paths if path.startswith("configs/phase6/templates/")
    }
    if observed_templates != expected_templates:
        raise PreflightValidationError(
            "source inventory must contain exactly the five recovery templates"
        )
    if not any(path.startswith("tests/") for path in paths):
        raise PreflightValidationError(
            "source inventory contains no tests/**/*.py files"
        )
    missing_runtime_files = sorted(
        {".python-version", "pyproject.toml", "uv.lock"} - paths
    )
    if missing_runtime_files:
        raise PreflightValidationError(
            "source inventory is missing runtime lock/config files: "
            f"{missing_runtime_files}"
        )
    return sorted(paths)


def _runtime_environment_fingerprint() -> dict[str, Any]:
    distributions = []
    missing = []
    for name in _RUNTIME_DISTRIBUTIONS:
        try:
            version = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            missing.append(name)
            continue
        distributions.append({"name": name, "version": version})
    if missing:
        raise PreflightValidationError(
            f"required runtime distributions are missing: {missing}"
        )
    libc_name, libc_version = platform.libc_ver()
    return {
        "abi_cache_tag": sys.implementation.cache_tag,
        "distributions": distributions,
        "implementation": sys.implementation.name,
        "libc": {"name": libc_name, "version": libc_version},
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "python_executable_realpath": os.path.realpath(sys.executable),
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        "soabi": sysconfig.get_config_var("SOABI"),
    }


def _git_stdout(repository_root: str, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", repository_root, "-c", "core.quotePath=false", *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise PreflightValidationError(
            f"cannot inspect repository Git state: {exc}"
        ) from exc
    return completed.stdout


def _repository_vcs_fingerprint(repository_root: str) -> dict[str, Any]:
    root_head = _git_stdout(repository_root, "rev-parse", "HEAD").strip()
    sana_gitlink = _git_stdout(
        repository_root, "rev-parse", "HEAD:third_party/Sana"
    ).strip()
    sana_root = os.path.join(repository_root, "third_party", "Sana")
    sana_head = _git_stdout(sana_root, "rev-parse", "HEAD").strip()
    for value, name in (
        (root_head, "root HEAD"),
        (sana_gitlink, "Sana gitlink"),
        (sana_head, "Sana HEAD"),
    ):
        if _GIT_COMMIT_RE.fullmatch(value) is None:
            raise PreflightValidationError(f"{name} is not a full Git commit")
    if root_head != _EXPECTED_REPOSITORY_HEAD:
        raise PreflightValidationError(
            "repository HEAD differs from the frozen baseline"
        )
    if sana_gitlink != _EXPECTED_SANA_HEAD or sana_head != sana_gitlink:
        raise PreflightValidationError("Sana submodule HEAD/gitlink differs")

    root_status = _git_stdout(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ":(glob)configs/phase6/templates/*.yaml.in",
        ":(glob)scripts/**/*.py",
        ":(glob)src/**/*.py",
        ":(glob)tests/**/*.py",
        ".python-version",
        "pyproject.toml",
        "uv.lock",
    ).splitlines()
    sana_status = _git_stdout(
        sana_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ":(glob)**/*.py",
    ).splitlines()
    return {
        "repository_head": root_head,
        "root_source_status_lines": root_status,
        "root_source_status_sha256": sha256(
            canonical_json_bytes(root_status)
        ).hexdigest(),
        "sana_gitlink_head": sana_gitlink,
        "sana_source_status_lines": sana_status,
        "sana_source_status_sha256": sha256(
            canonical_json_bytes(sana_status)
        ).hexdigest(),
        "sana_worktree_head": sana_head,
    }


def build_phase6_source_manifest(
    repository_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build the complete sorted Python source inventory after code is frozen."""

    declared_root = _require_absolute_path(
        os.fspath(repository_root), "repository_root"
    )
    absolute_root = os.path.abspath(declared_root)
    if not os.path.isdir(absolute_root):
        raise PreflightValidationError("repository_root must be an existing directory")
    repository_realpath = os.path.realpath(absolute_root)
    try:
        validate_phase6_planrow_recovery_authority(repository_root=absolute_root)
    except (OSError, TypeError, ValueError) as exc:
        raise PreflightValidationError(
            f"plan-row recovery authority validation failed: {exc}"
        ) from exc
    targets = _TargetRegistry()
    files: list[dict[str, Any]] = []
    for relative_path in _discover_source_paths(absolute_root):
        absolute_path = os.path.join(absolute_root, *relative_path.split("/"))
        observed, _ = _observe_file(absolute_path, f"source inventory {relative_path}")
        if not _within_repository(observed["realpath"], repository_realpath):
            raise PreflightValidationError(
                f"source inventory target escapes repository: {relative_path}"
            )
        targets.add(observed, f"source:{relative_path}")
        files.append(
            {
                "path": relative_path,
                "realpath": observed["realpath"],
                "sha256": observed["sha256"],
                "size_bytes": observed["size_bytes"],
            }
        )
    return {
        "files": files,
        **recovery_lifecycle(),
        "inventory_globs": list(_SOURCE_INVENTORY_GLOBS),
        "recovery_authority": {
            "path": PLANROW_RECOVERY_AUTHORITY_PATH,
            "sha256": _require_sha256(
                PLANROW_RECOVERY_AUTHORITY_SHA256,
                "plan-row recovery authority SHA256",
                reject_placeholder=True,
            ),
        },
        "registered_before_recovery_cohort_started": True,
        "repository_root": declared_root,
        "repository_vcs": _repository_vcs_fingerprint(absolute_root),
        "runtime_environment": _runtime_environment_fingerprint(),
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
    }


def _validate_source_manifest(
    value: Any,
    *,
    request_repository_root: str,
    repository_realpath: str,
    targets: _TargetRegistry,
    file_hasher: Callable[[str], str] | None,
) -> list[dict[str, Any]]:
    _require_exact_keys(
        value,
        frozenset(
            {
                "files",
                *recovery_lifecycle(),
                "inventory_globs",
                "recovery_authority",
                "registered_before_recovery_cohort_started",
                "repository_root",
                "repository_vcs",
                "runtime_environment",
                "schema_version",
            }
        ),
        "source_manifest",
    )
    if value["schema_version"] != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise PreflightValidationError("source-manifest schema differs")
    if value["repository_root"] != request_repository_root:
        raise PreflightValidationError("source-manifest repository_root differs")
    if value["inventory_globs"] != list(_SOURCE_INVENTORY_GLOBS):
        raise PreflightValidationError("source-manifest inventory globs differ")
    if value["runtime_environment"] != _runtime_environment_fingerprint():
        raise PreflightValidationError(
            "source-manifest runtime environment fingerprint differs"
        )
    if value["repository_vcs"] != _repository_vcs_fingerprint(request_repository_root):
        raise PreflightValidationError(
            "source-manifest repository VCS fingerprint differs"
        )
    if value["registered_before_recovery_cohort_started"] is not True:
        raise PreflightValidationError("source-manifest registration flag must be true")
    _validate_recovery_lifecycle_fields(value, "source-manifest lifecycle")
    expected_authority = {
        "path": PLANROW_RECOVERY_AUTHORITY_PATH,
        "sha256": _require_sha256(
            PLANROW_RECOVERY_AUTHORITY_SHA256,
            "plan-row recovery authority SHA256",
            reject_placeholder=True,
        ),
    }
    if value["recovery_authority"] != expected_authority:
        raise PreflightValidationError("source-manifest recovery authority differs")
    try:
        validate_phase6_planrow_recovery_authority(
            repository_root=request_repository_root
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PreflightValidationError(
            f"source-manifest recovery authority validation failed: {exc}"
        ) from exc
    files = value["files"]
    if not isinstance(files, list) or not files:
        raise PreflightValidationError("source_manifest.files must be a non-empty list")
    expected_paths = _discover_source_paths(request_repository_root)
    manifest_paths = [
        entry.get("path") if isinstance(entry, Mapping) else None for entry in files
    ]
    if manifest_paths != expected_paths:
        raise PreflightValidationError(
            "source-manifest paths differ from the sorted complete repository inventory"
        )
    observed_files: list[dict[str, Any]] = []
    for index, entry in enumerate(files):
        name = f"source_manifest.files[{index}]"
        _require_exact_keys(
            entry,
            frozenset({"path", "realpath", "sha256", "size_bytes"}),
            name,
        )
        relative_path = _require_string(entry["path"], f"{name}.path")
        absolute = os.path.abspath(
            os.path.join(request_repository_root, *relative_path.split("/"))
        )
        expected_sha = _require_sha256(
            entry["sha256"], f"{name}.sha256", reject_placeholder=True
        )
        expected_realpath = _require_absolute_path(
            entry["realpath"], f"{name}.realpath"
        )
        expected_size = _require_int(entry["size_bytes"], f"{name}.size_bytes")
        observed, _ = _observe_file(
            absolute,
            f"source {relative_path}",
            expected_size=expected_size,
            file_hasher=file_hasher,
        )
        if not _within_repository(observed["realpath"], repository_realpath):
            raise PreflightValidationError(
                f"source resolves outside repository: {relative_path}"
            )
        if os.path.normcase(observed["realpath"]) != os.path.normcase(
            expected_realpath
        ):
            raise PreflightValidationError(f"source {relative_path} realpath mismatch")
        if not hmac.compare_digest(observed["sha256"], expected_sha):
            raise PreflightValidationError(
                f"source {relative_path} SHA mismatch: expected {expected_sha}, "
                f"got {observed['sha256']}"
            )
        targets.add(observed, f"source:{relative_path}")
        observed_files.append(
            {
                "inventory_path": relative_path,
                **observed,
            }
        )
    return observed_files


def _pin_for_role(
    pins: Mapping[str, Any],
    role: str,
    *,
    fixed_digest: str | None,
) -> tuple[str, str]:
    pin = pins[role]
    _require_exact_keys(pin, frozenset({"path", "sha256"}), f"{role} pin")
    path = _require_absolute_path(pin["path"], f"{role} pin.path")
    expected = _require_sha256(
        pin["sha256"],
        f"{role} pin.sha256",
        reject_placeholder=fixed_digest is None,
    )
    if fixed_digest is not None and not hmac.compare_digest(expected, fixed_digest):
        raise PreflightValidationError(
            f"{role} requested SHA differs from frozen SHA256"
        )
    return path, expected


def _require_receipt_pin_backlink(
    value: Any, expected: Mapping[str, Any], name: str
) -> None:
    if not isinstance(value, Mapping):
        raise PreflightValidationError(f"{name} must be a pin mapping")
    if value.get("path") != expected.get("path") or value.get("sha256") != expected.get(
        "sha256"
    ):
        raise PreflightValidationError(f"{name} differs")


def _validate_rebind_receipt_chain(
    *,
    purpose: str,
    artifacts: Mapping[str, Any],
    values: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> None:
    if purpose not in ("smoke", "training"):
        return

    reference = values["reference_rebind_receipt"]
    _validate_recovery_lifecycle_fields(reference, "reference rebind receipt")
    if (
        reference.get("schema_version")
        != "sana-phase6-reference-v9-cpu-rebind-receipt-v1"
        or reference.get("artifact_role")
        != "phase6_reference_v9_cpu_byte_identical_rebind"
        or reference.get("status") != "pass"
    ):
        raise PreflightValidationError("reference rebind receipt identity differs")
    reference_authorization = reference.get("authorization")
    reference_outputs = reference.get("new")
    if not isinstance(reference_authorization, Mapping) or not isinstance(
        reference_outputs, Mapping
    ):
        raise PreflightValidationError("reference rebind receipt bindings differ")
    _require_receipt_pin_backlink(
        reference_authorization.get("source_manifest_v9"),
        source_manifest,
        "reference receipt source-manifest backlink",
    )
    _require_receipt_pin_backlink(
        reference_authorization.get("recovery_authority"),
        artifacts["recovery_authority"],
        "reference receipt recovery-authority backlink",
    )
    _require_receipt_pin_backlink(
        reference_outputs.get("request_v10"),
        artifacts["reference_preflight_request"],
        "reference receipt request backlink",
    )
    _require_receipt_pin_backlink(
        reference_outputs.get("report_v10"),
        artifacts["reference_preflight_report"],
        "reference receipt report backlink",
    )

    if purpose != "training":
        return

    smoke = values["smoke_rebind_receipt"]
    _validate_recovery_lifecycle_fields(smoke, "smoke rebind receipt")
    if (
        smoke.get("schema_version") != "sana-phase6-smoke-v8-cpu-rebind-receipt-v1"
        or smoke.get("artifact_role") != "phase6_smoke_v8_cpu_byte_identical_rebind"
        or smoke.get("status") != "pass"
    ):
        raise PreflightValidationError("smoke rebind receipt identity differs")
    smoke_authorization = smoke.get("authorization")
    smoke_outputs = smoke.get("new")
    if not isinstance(smoke_authorization, Mapping) or not isinstance(
        smoke_outputs, Mapping
    ):
        raise PreflightValidationError("smoke rebind receipt bindings differ")
    _require_receipt_pin_backlink(
        smoke_authorization.get("source_manifest_v9"),
        source_manifest,
        "smoke receipt source-manifest backlink",
    )
    _require_receipt_pin_backlink(
        smoke_authorization.get("recovery_authority"),
        artifacts["recovery_authority"],
        "smoke receipt recovery-authority backlink",
    )
    _require_receipt_pin_backlink(
        smoke.get("reference_rebind_receipt"),
        artifacts["reference_rebind_receipt"],
        "smoke receipt reference-receipt backlink",
    )
    for output_role, artifact_role in (
        ("request_v10", "smoke_preflight_request"),
        ("report_v10", "smoke_preflight_report"),
        ("gate_v8", "real_2b_smoke_gate"),
    ):
        _require_receipt_pin_backlink(
            smoke_outputs.get(output_role),
            artifacts[artifact_role],
            f"smoke receipt {artifact_role} backlink",
        )

    evidence = values["evidence_rebinding_receipt"]
    _validate_recovery_lifecycle_fields(evidence, "evidence-rebinding receipt")
    if (
        evidence.get("schema_version") != "sana-phase6-evidence-rebinding-receipt-v6"
        or evidence.get("artifact_role")
        != "phase6_planrow_recovery_historical_gpu_evidence_rebinding"
        or evidence.get("status") != "pass"
    ):
        raise PreflightValidationError("evidence-rebinding receipt identity differs")
    for field, expected_pin in (
        ("reference_rebind_receipt", artifacts["reference_rebind_receipt"]),
        ("smoke_rebind_receipt", artifacts["smoke_rebind_receipt"]),
        ("source_manifest_v9", source_manifest),
        ("recovery_authority", artifacts["recovery_authority"]),
    ):
        _require_receipt_pin_backlink(
            evidence.get(field), expected_pin, f"evidence receipt {field} backlink"
        )


def _capture_rebind_builder_bytes(
    source_manifest_value: Mapping[str, Any],
    *,
    repository_root: str,
    expected_builder_path: str,
) -> bytes:
    builder_entries = [
        entry
        for entry in source_manifest_value.get("files", [])
        if isinstance(entry, Mapping)
        and entry.get("path") == "scripts/build_phase6_recovery_rebind.py"
    ]
    if len(builder_entries) != 1:
        raise PreflightValidationError(
            "source manifest recovery-rebind builder entry differs"
        )
    builder_entry = builder_entries[0]
    _require_exact_keys(
        builder_entry,
        frozenset({"path", "realpath", "sha256", "size_bytes"}),
        "source manifest recovery-rebind builder entry",
    )
    builder_path = os.path.abspath(
        os.path.join(repository_root, "scripts", "build_phase6_recovery_rebind.py")
    )
    if builder_path != expected_builder_path:
        raise PreflightValidationError(
            "recovery-rebind builder path differs from the formal DAG"
        )
    observed_builder, builder_bytes = _observe_file(
        builder_path,
        "recovery-rebind builder",
        capture=True,
        expected_size=_require_int(
            builder_entry["size_bytes"],
            "source manifest recovery-rebind builder size",
        ),
    )
    if (
        observed_builder["realpath"] != builder_entry["realpath"]
        or observed_builder["sha256"] != builder_entry["sha256"]
        or builder_bytes is None
    ):
        raise PreflightValidationError(
            "recovery-rebind builder differs from the source manifest"
        )
    return builder_bytes


def _validate_full_rebind_receipts(
    *,
    purpose: str,
    artifacts: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_manifest_value: Mapping[str, Any],
    repository_root: str,
) -> None:
    if purpose not in ("smoke", "training"):
        return
    try:
        from sana_wam.train.phase6_artifact_dag import (
            FROZEN_PATHS,
            RECOVERY_REBIND_BUILDER,
            validate_evidence_rebinding_receipt,
        )

        expected_paths = {
            "recovery_authority": FROZEN_PATHS.artifacts["recovery_authority"],
            "reference_preflight_request": FROZEN_PATHS.artifacts[
                "reference_preflight_request"
            ],
            "reference_preflight_report": FROZEN_PATHS.artifacts[
                "reference_preflight_report"
            ],
            "reference_rebind_receipt": FROZEN_PATHS.artifacts[
                "reference_rebind_receipt"
            ],
        }
        if source_manifest.get("path") != FROZEN_PATHS.artifacts["source_manifest"]:
            raise PreflightValidationError(
                "rebind source-manifest path differs from the formal DAG"
            )
        for role, expected_path in expected_paths.items():
            if artifacts[role].get("path") != expected_path:
                raise PreflightValidationError(
                    f"{role.replace('_', '-')} path differs from the formal DAG"
                )

        if purpose == "training":
            for role in (
                "smoke_preflight_request",
                "smoke_preflight_report",
                "smoke_rebind_receipt",
                "evidence_rebinding_receipt",
                "real_2b_smoke_gate",
            ):
                if artifacts[role].get("path") != FROZEN_PATHS.artifacts[role]:
                    raise PreflightValidationError(
                        f"{role.replace('_', '-')} path differs from the formal DAG"
                    )
            validate_evidence_rebinding_receipt(
                artifacts["evidence_rebinding_receipt"]["sha256"],
                paths=FROZEN_PATHS,
            )
            return

        builder_bytes = _capture_rebind_builder_bytes(
            source_manifest_value,
            repository_root=repository_root,
            expected_builder_path=RECOVERY_REBIND_BUILDER,
        )
        builder_path = RECOVERY_REBIND_BUILDER

        module_name = "sana_phase6_preflight_reference_rebind_validator"
        module = ModuleType(module_name)
        module.__file__ = builder_path
        sys.modules[module_name] = module
        try:
            exec(
                compile(builder_bytes, builder_path, "exec", dont_inherit=True),
                module.__dict__,
            )

            def pin(value: Mapping[str, Any], _label: str):
                info = os.lstat(value["path"])
                return module.Pin(value["path"], value["sha256"], info.st_size, 0o400)

            module.validate_reference_rebind_receipt(
                pin(artifacts["reference_rebind_receipt"], "reference receipt"),
                source_pin=pin(source_manifest, "source manifest"),
                authority_pin=pin(
                    artifacts["recovery_authority"], "recovery authority"
                ),
            )
        finally:
            sys.modules.pop(module_name, None)
    except PreflightValidationError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PreflightValidationError(
            f"full recovery rebind receipt replay failed: {exc}"
        ) from exc


def _validate_prior_authorization_pair(
    request: Mapping[str, Any],
    report: Mapping[str, Any],
    request_bytes: bytes,
    report_bytes: bytes,
    *,
    expected_purpose: str,
    policy: _FrozenPolicy,
    file_hasher: Callable[[str], str] | None,
) -> None:
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise PreflightValidationError("prior authorization request schema differs")
    if request.get("purpose") != expected_purpose:
        raise PreflightValidationError("prior authorization request purpose differs")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise PreflightValidationError("prior authorization report schema differs")
    if report.get("purpose") != expected_purpose or report.get("status") != "pass":
        raise PreflightValidationError("prior authorization report did not pass")
    _require_exact_typed_value(
        report.get("authorization"),
        request.get("authorization"),
        "prior authorization request/report",
    )
    if report.get("request_sha256") != sha256(request_bytes).hexdigest():
        raise PreflightValidationError("prior authorization request SHA differs")
    if (
        request.get("registered_before_recovery_cohort_started") is not True
        or report.get("registered_before_recovery_cohort_started") is not True
    ):
        raise PreflightValidationError("prior authorization registration differs")
    _validate_recovery_lifecycle_fields(
        request, "prior authorization request lifecycle"
    )
    _validate_recovery_lifecycle_fields(report, "prior authorization report lifecycle")
    if expected_purpose == "reference_precompute":
        lifecycle = (
            *recovery_lifecycle(),
            "reference_gpu_started",
        )
    elif expected_purpose == "smoke":
        lifecycle = (
            "reference_precompute_completed",
            "smoke_gpu_started",
            *recovery_lifecycle(),
        )
    else:
        raise PreflightValidationError("invalid prior authorization purpose")
    for key in lifecycle:
        if type(report.get(key)) is not type(request.get(key)) or report.get(
            key
        ) != request.get(key):
            raise PreflightValidationError(
                f"prior authorization lifecycle {key} differs"
            )

    # A prior report is authorization only if the complete request still
    # reproduces it. Partial pin checks cannot establish the original gate.
    reproduced = _validate_with_policy(
        request,
        policy=policy,
        file_hasher=file_hasher,
    )
    if report_bytes != canonical_json_bytes(reproduced):
        raise PreflightValidationError(
            "prior authorization report does not reproduce from current artifacts"
        )


def _validate_with_policy(
    request: Mapping[str, Any],
    *,
    policy: _FrozenPolicy,
    file_hasher: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    purpose = request.get("purpose")
    common_lifecycle_keys = set(recovery_lifecycle())
    if purpose == "reference_precompute":
        lifecycle_keys = common_lifecycle_keys | {"reference_gpu_started"}
        artifact_roles = _REFERENCE_PRECOMPUTE_ARTIFACT_ROLES
        expected_authorization = _REFERENCE_PRECOMPUTE_AUTHORIZATION
    elif purpose == "smoke":
        lifecycle_keys = common_lifecycle_keys | {
            "reference_precompute_completed",
            "smoke_gpu_started",
        }
        artifact_roles = _SMOKE_ARTIFACT_ROLES
        expected_authorization = _SMOKE_AUTHORIZATION
    elif purpose == "training":
        lifecycle_keys = common_lifecycle_keys | {
            "reference_precompute_completed",
            "smoke_completed",
        }
        artifact_roles = _TRAINING_ARTIFACT_ROLES
        expected_authorization = None
    else:
        raise PreflightValidationError(
            "preflight purpose must be reference_precompute, smoke, or training"
        )
    _require_exact_keys(
        request,
        frozenset(
            {
                "authorization",
                "artifacts",
                "effective_output_root",
                "input_config",
                *lifecycle_keys,
                "purpose",
                "registered_before_recovery_cohort_started",
                "repository_root",
                "runtime_files",
                "schema_version",
                "source_manifest",
            }
        ),
        "preflight request",
    )
    if request["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise PreflightValidationError("preflight request schema differs")
    if request["registered_before_recovery_cohort_started"] is not True:
        raise PreflightValidationError(
            "preflight request registration flag must be true"
        )
    _validate_recovery_lifecycle_fields(request, "preflight request lifecycle")
    if purpose == "training":
        _require_exact_keys(
            request["authorization"],
            _TRAINING_AUTHORIZATION_KEYS,
            "preflight authorization",
        )
    else:
        _require_exact_typed_value(
            request["authorization"],
            expected_authorization,
            "preflight authorization",
        )
    if purpose == "reference_precompute":
        if request["reference_gpu_started"] is not False:
            raise PreflightValidationError("reference-precompute lifecycle differs")
    elif purpose == "smoke":
        if (
            request["reference_precompute_completed"] is not True
            or request["smoke_gpu_started"] is not False
        ):
            raise PreflightValidationError("smoke preflight lifecycle differs")
    elif (
        request["reference_precompute_completed"] is not True
        or request["smoke_completed"] is not True
    ):
        raise PreflightValidationError("training preflight lifecycle differs")
    _check_registration_flags(request, "request")

    if purpose == "training":
        authorization = request["authorization"]
        if authorization["operation"] != "phase6_training":
            raise PreflightValidationError("training authorization operation differs")
        for key, expected in (
            ("reference_gpu_forward_completed", True),
            ("real_2b_smoke_completed", True),
            ("optimizer_training_allowed", True),
            ("closed_loop_allowed", False),
            ("resume_allowed", False),
        ):
            if authorization[key] is not expected:
                raise PreflightValidationError(f"training authorization {key} differs")
        for key in ("arm", "run_id"):
            if not isinstance(authorization[key], str) or not authorization[key]:
                raise PreflightValidationError(
                    f"training authorization {key} must be non-empty"
                )
        _require_sha256(
            authorization["arm_config_projection_sha256"],
            "training authorization arm_config_projection_sha256",
            reject_placeholder=True,
        )
        _require_absolute_path(
            authorization["run_directory"],
            "training authorization run_directory",
        )
        factors = authorization["factors"]
        _require_exact_keys(factors, frozenset({"A", "E", "T"}), "arm factors")
        if any(type(factors[key]) is not bool for key in ("A", "E", "T")):
            raise PreflightValidationError("training arm factors must be booleans")

    repository_root = _require_absolute_path(
        request["repository_root"], "repository_root"
    )
    repository_absolute = os.path.abspath(repository_root)
    if not os.path.isdir(repository_absolute):
        raise PreflightValidationError("repository_root must be an existing directory")
    repository_realpath = os.path.realpath(repository_absolute)
    output_root = _validate_output_root(request["effective_output_root"], policy)

    artifacts = request["artifacts"]
    _require_exact_keys(artifacts, frozenset(artifact_roles), "artifacts")
    runtime_pins = request["runtime_files"]
    _require_exact_keys(runtime_pins, frozenset(_RUNTIME_ROLES), "runtime_files")
    fixed_artifacts = dict(policy.fixed_artifact_sha256)
    fixed_runtime = dict(policy.runtime_sha256)
    targets = _TargetRegistry()

    input_config_path, input_config_expected = _pin_for_role(
        {"input_config": request["input_config"]},
        "input_config",
        fixed_digest=None,
    )
    observed_input_config, input_config_bytes = _observe_file(
        input_config_path, "input config", capture=True
    )
    if not hmac.compare_digest(observed_input_config["sha256"], input_config_expected):
        raise PreflightValidationError("input-config SHA mismatch")
    if not _within_repository(observed_input_config["realpath"], repository_realpath):
        raise PreflightValidationError("input config resolves outside repository")
    targets.add(observed_input_config, "input_config")
    if input_config_bytes is None:
        raise AssertionError("captured input-config bytes are absent")

    observed_artifacts: list[dict[str, Any]] = []
    artifact_values: dict[str, Any] = {}
    artifact_bytes: dict[str, bytes] = {}
    for role in artifact_roles:
        path, expected = _pin_for_role(
            artifacts,
            role,
            fixed_digest=fixed_artifacts.get(role),
        )
        observed, data = _observe_file(path, f"artifact {role}", capture=True)
        if not hmac.compare_digest(observed["sha256"], expected):
            raise PreflightValidationError(
                f"artifact {role} SHA mismatch: expected {expected}, "
                f"got {observed['sha256']}"
            )
        if role in {
            "reference_preflight_request",
            "reference_preflight_report",
            "smoke_preflight_request",
            "smoke_preflight_report",
        } and not _within_repository(observed["realpath"], output_root["realpath"]):
            raise PreflightValidationError(
                f"authorization record {role} is outside the operational root"
            )
        targets.add(observed, f"artifact:{role}")
        observed_artifacts.append({"role": role, **observed})
        if data is None:
            raise AssertionError("captured artifact bytes are absent")
        artifact_bytes[role] = data
        if role != "protocol":
            value = _strict_json_loads(data, role)
            if not isinstance(value, Mapping):
                raise PreflightValidationError(f"artifact {role} must be a JSON object")
            _check_registration_flags(value, f"artifact.{role}")
            artifact_values[role] = value

    if artifact_bytes["storage_amendment"] != _pretty_json_bytes(
        artifact_values["storage_amendment"]
    ):
        raise PreflightValidationError("storage amendment is not pretty-canonical JSON")
    if artifact_bytes["external_components_manifest"] != _pretty_json_bytes(
        artifact_values["external_components_manifest"]
    ):
        raise PreflightValidationError(
            "external-components manifest is not pretty-canonical JSON"
        )
    for role in (
        "expansion_eligibility_amendment",
        "padding_semantics_amendment",
    ):
        if artifact_bytes[role] != canonical_json_bytes(
            artifact_values[role], ensure_ascii=True
        ):
            raise PreflightValidationError(
                f"{role.replace('_', '-')} is not canonical JSON"
            )
    if artifact_bytes["task_plan"] != canonical_json_bytes(
        artifact_values["task_plan"]
    ):
        raise PreflightValidationError("task-plan artifact is not canonical JSON")
    if artifact_bytes["dataset_contract"] != canonical_json_bytes(
        artifact_values["dataset_contract"], ensure_ascii=True
    ):
        raise PreflightValidationError(
            "dataset-contract artifact is not canonical JSON"
        )
    if artifact_bytes["runtime_support_manifest"] != canonical_json_bytes(
        artifact_values["runtime_support_manifest"]
    ):
        raise PreflightValidationError("runtime-support manifest is not canonical JSON")
    for role in ("projection_equivalence_receipt", "recovery_authority"):
        if artifact_bytes[role] != canonical_json_bytes(artifact_values[role]):
            raise PreflightValidationError(
                f"{role.replace('_', '-')} is not canonical JSON"
            )
    if purpose in ("smoke", "training"):
        canonical_roles = [
            "reference_preflight_request",
            "reference_preflight_report",
            "reference_rebind_receipt",
            "action_reference",
            "action_reference_spot_check",
            "arm_projection_manifest",
        ]
        if purpose == "training":
            canonical_roles.extend(
                [
                    "smoke_preflight_request",
                    "smoke_preflight_report",
                    "smoke_rebind_receipt",
                    "evidence_rebinding_receipt",
                    "real_2b_smoke_gate",
                ]
            )
        for role in canonical_roles:
            if artifact_bytes[role] != canonical_json_bytes(artifact_values[role]):
                raise PreflightValidationError(
                    f"{role.replace('_', '-')} artifact is not canonical JSON"
                )

    _validate_rebind_receipt_chain(
        purpose=purpose,
        artifacts=artifacts,
        values=artifact_values,
        source_manifest=request["source_manifest"],
    )

    _validate_registry(artifact_values["registry"], policy)
    _validate_storage(artifact_values["storage_amendment"], policy)
    _validate_external_manifest(artifact_values["external_components_manifest"], policy)
    expansion_contract = _validate_expansion_amendment(
        artifact_values["expansion_eligibility_amendment"], policy
    )
    padding_contract = _validate_padding_amendment(
        artifact_values["padding_semantics_amendment"],
        policy,
        expansion_contract=expansion_contract,
    )
    observed_runtime_support = _validate_runtime_support_manifest(
        artifact_values["runtime_support_manifest"],
        policy,
        targets=targets,
        file_hasher=file_hasher,
    )
    if artifacts["recovery_authority"] != {
        "path": PLANROW_RECOVERY_AUTHORITY_PATH,
        "sha256": _require_sha256(
            PLANROW_RECOVERY_AUTHORITY_SHA256,
            "plan-row recovery authority SHA256",
            reject_placeholder=True,
        ),
    }:
        raise PreflightValidationError("recovery-authority artifact pin differs")
    try:
        validate_phase6_planrow_recovery_authority(repository_root=repository_absolute)
        from sana_wam.train.phase6_arm_config import (
            validate_phase6_projection_equivalence_receipt_bytes,
        )

        validate_phase6_projection_equivalence_receipt_bytes(
            artifact_bytes["projection_equivalence_receipt"],
            expected_artifact_sha256=artifacts["projection_equivalence_receipt"][
                "sha256"
            ],
        )
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise PreflightValidationError(
            f"recovery authority/equivalence validation failed: {exc}"
        ) from exc
    plan_rows = _validate_task_plan(
        artifact_values["task_plan"],
        policy,
        expansion_contract=expansion_contract,
    )
    dataset_rows = _validate_dataset_contract(
        artifact_values["dataset_contract"],
        policy,
        plan_rows,
        expansion_contract=expansion_contract,
        padding_contract=padding_contract,
    )
    observed_by_role = {item["role"]: item["sha256"] for item in observed_artifacts}
    requested_source_pin = request["source_manifest"]
    _require_exact_keys(
        requested_source_pin,
        frozenset({"path", "sha256"}),
        "source_manifest pin",
    )
    _require_sha256(
        requested_source_pin["sha256"],
        "source_manifest pin.sha256",
        reject_placeholder=True,
    )
    source_manifest_path, source_manifest_expected = _pin_for_role(
        {"source_manifest": request["source_manifest"]},
        "source_manifest",
        fixed_digest=None,
    )
    observed_source_manifest, source_manifest_bytes = _observe_file(
        source_manifest_path, "source manifest", capture=True
    )
    if not hmac.compare_digest(
        observed_source_manifest["sha256"], source_manifest_expected
    ):
        raise PreflightValidationError(
            "source-manifest SHA mismatch: expected "
            f"{source_manifest_expected}, got {observed_source_manifest['sha256']}"
        )
    targets.add(observed_source_manifest, "source_manifest")
    if source_manifest_bytes is None:
        raise AssertionError("captured source-manifest bytes are absent")
    source_manifest_value = _strict_json_loads(source_manifest_bytes, "source manifest")
    if not isinstance(source_manifest_value, Mapping):
        raise PreflightValidationError("source manifest must be a JSON object")
    if source_manifest_bytes != canonical_json_bytes(source_manifest_value):
        raise PreflightValidationError("source manifest is not canonical JSON")
    _check_registration_flags(source_manifest_value, "source_manifest")
    observed_sources = _validate_source_manifest(
        source_manifest_value,
        request_repository_root=repository_root,
        repository_realpath=repository_realpath,
        targets=targets,
        file_hasher=file_hasher,
    )
    _validate_full_rebind_receipts(
        purpose=purpose,
        artifacts=artifacts,
        source_manifest=request["source_manifest"],
        source_manifest_value=source_manifest_value,
        repository_root=repository_root,
    )

    observed_runtime: list[dict[str, Any]] = []
    for role in _RUNTIME_ROLES:
        path, expected = _pin_for_role(
            runtime_pins,
            role,
            fixed_digest=fixed_runtime[role],
        )
        observed, _ = _observe_file(
            path, f"runtime file {role}", file_hasher=file_hasher
        )
        if not hmac.compare_digest(observed["sha256"], expected):
            raise PreflightValidationError(
                f"runtime file {role} SHA mismatch: expected {expected}, "
                f"got {observed['sha256']}"
            )
        targets.add(observed, f"runtime:{role}")
        observed_runtime.append({"role": role, **observed})

    observed_external: list[dict[str, Any]] = []
    for index, (path, size_bytes, expected) in enumerate(policy.external_components):
        observed, _ = _observe_file(
            path,
            f"external component {index}",
            expected_size=size_bytes,
            file_hasher=file_hasher,
        )
        if not hmac.compare_digest(observed["sha256"], expected):
            raise PreflightValidationError(
                f"external component {index} SHA mismatch: expected {expected}, "
                f"got {observed['sha256']}"
            )
        targets.add(observed, f"external:{index}")
        observed_external.append({"manifest_index": index, **observed})

    arm_manifest = None
    observed_prior_input_configs: list[dict[str, Any]] = []
    if purpose in ("smoke", "training"):
        reference_request = artifact_values["reference_preflight_request"]
        reference_report = artifact_values["reference_preflight_report"]
        _validate_prior_authorization_pair(
            reference_request,
            reference_report,
            artifact_bytes["reference_preflight_request"],
            artifact_bytes["reference_preflight_report"],
            expected_purpose="reference_precompute",
            policy=policy,
            file_hasher=file_hasher,
        )
        reference_config_path, reference_config_expected = _pin_for_role(
            {"reference_input_config": reference_request["input_config"]},
            "reference_input_config",
            fixed_digest=None,
        )
        observed_reference_config, _ = _observe_file(
            reference_config_path, "reference input config"
        )
        if not hmac.compare_digest(
            observed_reference_config["sha256"], reference_config_expected
        ):
            raise PreflightValidationError("reference input-config SHA mismatch")
        if not _within_repository(
            observed_reference_config["realpath"], repository_realpath
        ):
            raise PreflightValidationError(
                "reference input config resolves outside repository"
            )
        targets.add(observed_reference_config, "prior:reference_input_config")
        observed_prior_input_configs.append(
            {"purpose": "reference_precompute", **observed_reference_config}
        )
        reference_report_pins = reference_report.get("pins")
        if not isinstance(reference_report_pins, Mapping):
            raise PreflightValidationError(
                "reference-precompute report pins are absent"
            )
        expected_reference_report_pins = {
            "observed_dataset_contract_sha256": observed_by_role["dataset_contract"],
            "observed_expansion_eligibility_amendment_sha256": observed_by_role[
                "expansion_eligibility_amendment"
            ],
            "observed_padding_semantics_amendment_sha256": observed_by_role[
                "padding_semantics_amendment"
            ],
            "observed_input_config_sha256": reference_request["input_config"]["sha256"],
            "observed_runtime_support_manifest_sha256": observed_by_role[
                "runtime_support_manifest"
            ],
            "observed_source_manifest_sha256": observed_source_manifest["sha256"],
        }
        if any(
            reference_report_pins.get(key) != expected
            for key, expected in expected_reference_report_pins.items()
        ):
            raise PreflightValidationError(
                "reference-precompute report/current provenance differs"
            )

        _validate_action_reference(
            artifact_values["action_reference"],
            policy,
            plan_rows,
            dataset_rows,
            dataset_contract_artifact_sha256=observed_by_role["dataset_contract"],
            source_manifest_sha256=observed_source_manifest["sha256"],
            precompute_config_file_sha256=reference_report_pins[
                "observed_input_config_sha256"
            ],
            precompute_request_sha256=observed_by_role["reference_preflight_request"],
            precompute_report_sha256=observed_by_role["reference_preflight_report"],
        )
        _validate_action_reference_spot_check(
            artifact_values["action_reference_spot_check"],
            artifact_values["action_reference"],
            reference_artifact_sha256=observed_by_role["action_reference"],
        )

        try:
            from sana_wam.train.phase6_arm_config import (
                validate_phase6_arm_projection_manifest_bytes,
                validate_phase6_smoke_runtime_bundle_bytes,
            )

            expected_arm = (
                request["authorization"]["arm"] if purpose == "training" else None
            )
            expected_projection = (
                request["authorization"]["arm_config_projection_sha256"]
                if purpose == "training"
                else None
            )
            arm_manifest = validate_phase6_arm_projection_manifest_bytes(
                artifact_bytes["arm_projection_manifest"],
                expected_artifact_sha256=observed_by_role["arm_projection_manifest"],
                expected_arm=expected_arm,
                expected_projection_sha256=expected_projection,
            )
            if purpose == "smoke":
                validate_phase6_smoke_runtime_bundle_bytes(
                    input_config_bytes,
                    expected_artifact_sha256=observed_input_config["sha256"],
                    expected_projection_manifest_sha256=observed_by_role[
                        "arm_projection_manifest"
                    ],
                )
        except (ImportError, TypeError, ValueError) as exc:
            raise PreflightValidationError(
                f"arm projection manifest validation failed: {exc}"
            ) from exc

    if purpose == "training":
        authorization = request["authorization"]
        arm_entry = next(
            item for item in arm_manifest["arms"] if item["arm"] == authorization["arm"]
        )
        if observed_input_config["sha256"] != authorization[
            "arm_config_projection_sha256"
        ] or input_config_bytes != canonical_json_bytes(arm_entry["projection"]):
            raise PreflightValidationError(
                "training input config is not the canonical arm projection sidecar"
            )
        for key, expected in (
            ("factors", arm_entry["factors"]),
            ("run_id", arm_entry["run_id"]),
            ("run_directory", arm_entry["run_directory"]),
        ):
            _require_exact_typed_value(
                authorization[key],
                expected,
                f"training authorization {key}",
            )

        smoke_request = artifact_values["smoke_preflight_request"]
        smoke_report = artifact_values["smoke_preflight_report"]
        _validate_prior_authorization_pair(
            smoke_request,
            smoke_report,
            artifact_bytes["smoke_preflight_request"],
            artifact_bytes["smoke_preflight_report"],
            expected_purpose="smoke",
            policy=policy,
            file_hasher=file_hasher,
        )
        smoke_config_path, smoke_config_expected = _pin_for_role(
            {"smoke_input_config": smoke_request["input_config"]},
            "smoke_input_config",
            fixed_digest=None,
        )
        observed_smoke_config, smoke_config_bytes = _observe_file(
            smoke_config_path, "smoke input config", capture=True
        )
        if not hmac.compare_digest(
            observed_smoke_config["sha256"], smoke_config_expected
        ):
            raise PreflightValidationError("smoke input-config SHA mismatch")
        if not _within_repository(
            observed_smoke_config["realpath"], repository_realpath
        ):
            raise PreflightValidationError(
                "smoke input config resolves outside repository"
            )
        targets.add(observed_smoke_config, "prior:smoke_input_config")
        observed_prior_input_configs.append(
            {"purpose": "smoke", **observed_smoke_config}
        )
        if smoke_config_bytes is None:
            raise AssertionError("captured smoke input-config bytes are absent")
        try:
            from sana_wam.train.phase6_arm_config import (
                validate_phase6_smoke_runtime_bundle_bytes,
            )

            validate_phase6_smoke_runtime_bundle_bytes(
                smoke_config_bytes,
                expected_artifact_sha256=observed_smoke_config["sha256"],
                expected_projection_manifest_sha256=observed_by_role[
                    "arm_projection_manifest"
                ],
            )
        except (ImportError, TypeError, ValueError) as exc:
            raise PreflightValidationError(
                f"prior smoke runtime-bundle validation failed: {exc}"
            ) from exc
        smoke_report_pins = smoke_report.get("pins")
        if not isinstance(smoke_report_pins, Mapping):
            raise PreflightValidationError("smoke report pins are absent")
        expected_smoke_report_pins = {
            "observed_action_reference_sha256": observed_by_role["action_reference"],
            "observed_action_reference_spot_check_sha256": observed_by_role[
                "action_reference_spot_check"
            ],
            "observed_arm_projection_manifest_sha256": observed_by_role[
                "arm_projection_manifest"
            ],
            "observed_reference_rebind_receipt_sha256": observed_by_role[
                "reference_rebind_receipt"
            ],
            "observed_input_config_sha256": observed_smoke_config["sha256"],
            "observed_expansion_eligibility_amendment_sha256": observed_by_role[
                "expansion_eligibility_amendment"
            ],
            "observed_padding_semantics_amendment_sha256": observed_by_role[
                "padding_semantics_amendment"
            ],
            "observed_runtime_support_manifest_sha256": observed_by_role[
                "runtime_support_manifest"
            ],
            "observed_source_manifest_sha256": observed_source_manifest["sha256"],
        }
        if any(
            smoke_report_pins.get(key) != expected
            for key, expected in expected_smoke_report_pins.items()
        ):
            raise PreflightValidationError("smoke report/current provenance differs")

        runtime_digests = dict(policy.runtime_sha256)
        try:
            from sana_wam.train.phase6_smoke_gate import (
                validate_phase6_smoke_artifact_bytes,
            )

            smoke_artifact = validate_phase6_smoke_artifact_bytes(
                artifact_bytes["real_2b_smoke_gate"],
                expected_artifact_sha256=observed_by_role["real_2b_smoke_gate"],
                expected_source_manifest_sha256=observed_source_manifest["sha256"],
                expected_runtime_support_manifest_sha256=observed_by_role[
                    "runtime_support_manifest"
                ],
                expected_smoke_runtime_config_sha256=observed_smoke_config["sha256"],
                expected_student_checkpoint_sha256=runtime_digests[
                    "student_checkpoint"
                ],
                expected_phase1_checkpoint_sha256=runtime_digests["phase1_checkpoint"],
                expected_action_stats_sha256=runtime_digests["action_stats"],
                expected_plan_sha256=policy.plan_sha256,
                expected_identity_sha256=policy.identity_sha256,
                expected_dataset_contract_sha256=observed_by_role["dataset_contract"],
                expected_reference_artifact_sha256=observed_by_role["action_reference"],
                expected_spot_artifact_sha256=observed_by_role[
                    "action_reference_spot_check"
                ],
                expected_arm_projection_manifest_sha256=observed_by_role[
                    "arm_projection_manifest"
                ],
                expected_preflight_request_sha256=observed_by_role[
                    "smoke_preflight_request"
                ],
                expected_preflight_report_sha256=observed_by_role[
                    "smoke_preflight_report"
                ],
            )
        except (ImportError, TypeError, ValueError) as exc:
            raise PreflightValidationError(
                f"real-2B smoke-gate validation failed: {exc}"
            ) from exc
        _validate_recovery_lifecycle_fields(
            smoke_artifact, "real-2B smoke artifact lifecycle"
        )
        if (
            smoke_artifact.get("status") != "pass"
            or smoke_artifact.get("registered_before_recovery_cohort_started")
            is not True
        ):
            raise PreflightValidationError(
                "real-2B smoke artifact is not a pre-training pass"
            )
        reference_row = artifact_values["action_reference"]["rows"][0]
        expected_paired_row = {
            "action_sigma": reference_row["action_sigma"],
            "common_input_trace_sha256": reference_row["common_input_trace_sha256"],
            "dataset_contract_row_sha256": reference_row["dataset_contract_row_sha256"],
            "dataset_index": reference_row["dataset_index"],
            "global_step": reference_row["global_step"],
            "plan_row_sha256": reference_row["plan_row_sha256"],
            "task_name": reference_row["task_name"],
        }
        if smoke_artifact.get("paired_fixed_row") != expected_paired_row:
            raise PreflightValidationError(
                "real-2B smoke paired row differs from action reference row 1"
            )
        equivalence_arms = artifact_values["projection_equivalence_receipt"].get("arms")
        if not isinstance(equivalence_arms, list) or any(
            not isinstance(item, Mapping) for item in equivalence_arms
        ):
            raise PreflightValidationError(
                "projection equivalence receipt arm map differs"
            )
        recovery_projection_by_arm = {
            item["arm"]: item["recovery_projection_sha256"] for item in equivalence_arms
        }
        legacy_projection_by_arm = {
            item["arm"]: item["legacy_projection_sha256"] for item in equivalence_arms
        }
        manifest_projection_by_arm = {
            item["arm"]: item["projection_sha256"] for item in arm_manifest["arms"]
        }
        if (
            len(recovery_projection_by_arm) != len(equivalence_arms)
            or len(legacy_projection_by_arm) != len(equivalence_arms)
            or manifest_projection_by_arm != recovery_projection_by_arm
        ):
            raise PreflightValidationError(
                "recovery arm manifest differs from equivalence receipt"
            )
        observed_cases = smoke_artifact.get("cases")
        if not isinstance(observed_cases, list) or any(
            legacy_projection_by_arm.get(case.get("case_id"))
            != case.get("config_projection_sha256")
            for case in observed_cases
            if isinstance(case, Mapping)
        ):
            raise PreflightValidationError(
                "real-2B smoke legacy projection differs from equivalence receipt"
            )

    report_pins = {
        "observed_dataset_contract_sha256": observed_by_role["dataset_contract"],
        "observed_embedded_identity_sha256": policy.identity_sha256,
        "observed_embedded_plan_sha256": policy.plan_sha256,
        "observed_external_manifest_sha256": observed_by_role[
            "external_components_manifest"
        ],
        "observed_expansion_eligibility_amendment_sha256": observed_by_role[
            "expansion_eligibility_amendment"
        ],
        "observed_input_config_sha256": observed_input_config["sha256"],
        "observed_protocol_sha256": observed_by_role["protocol"],
        "observed_padding_semantics_amendment_sha256": observed_by_role[
            "padding_semantics_amendment"
        ],
        "observed_registry_sha256": observed_by_role["registry"],
        "observed_runtime_support_manifest_sha256": observed_by_role[
            "runtime_support_manifest"
        ],
        "observed_source_manifest_sha256": observed_source_manifest["sha256"],
        "observed_storage_amendment_sha256": observed_by_role["storage_amendment"],
        "observed_task_plan_artifact_sha256": observed_by_role["task_plan"],
    }
    if purpose in ("smoke", "training"):
        report_pins.update(
            {
                "observed_action_reference_sha256": observed_by_role[
                    "action_reference"
                ],
                "observed_action_reference_spot_check_sha256": observed_by_role[
                    "action_reference_spot_check"
                ],
                "observed_arm_projection_manifest_sha256": observed_by_role[
                    "arm_projection_manifest"
                ],
                "observed_reference_preflight_report_sha256": observed_by_role[
                    "reference_preflight_report"
                ],
                "observed_reference_preflight_request_sha256": observed_by_role[
                    "reference_preflight_request"
                ],
                "observed_reference_rebind_receipt_sha256": observed_by_role[
                    "reference_rebind_receipt"
                ],
            }
        )
    if purpose == "training":
        report_pins.update(
            {
                "observed_real_2b_smoke_gate_sha256": observed_by_role[
                    "real_2b_smoke_gate"
                ],
                "observed_smoke_preflight_report_sha256": observed_by_role[
                    "smoke_preflight_report"
                ],
                "observed_smoke_preflight_request_sha256": observed_by_role[
                    "smoke_preflight_request"
                ],
                "observed_smoke_rebind_receipt_sha256": observed_by_role[
                    "smoke_rebind_receipt"
                ],
                "observed_evidence_rebinding_receipt_sha256": observed_by_role[
                    "evidence_rebinding_receipt"
                ],
            }
        )
    report = {
        "authorization": dict(request["authorization"]),
        "artifacts": observed_artifacts,
        "effective_output_root": output_root,
        "external_components": observed_external,
        "input_config": observed_input_config,
        "input_config_role": {
            "reference_precompute": "precompute_yaml",
            "smoke": "five_arm_smoke_runtime_bundle",
            "training": "arm_scientific_projection_sidecar",
        }[purpose],
        "pins": report_pins,
        "prior_input_configs": observed_prior_input_configs,
        "purpose": purpose,
        "registered_before_recovery_cohort_started": True,
        "repository": {
            "path": repository_root,
            "absolute_path": repository_absolute,
            "realpath": repository_realpath,
        },
        "request_sha256": sha256(canonical_json_bytes(request)).hexdigest(),
        "runtime_files": observed_runtime,
        "runtime_support_files": observed_runtime_support,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_files": observed_sources,
        "source_manifest": observed_source_manifest,
        "status": "pass",
        "validated_regular_file_count": (
            len(observed_artifacts)
            + 1
            + 1
            + len(observed_runtime)
            + len(observed_runtime_support)
            + len(observed_prior_input_configs)
            + len(observed_external)
            + len(observed_sources)
        ),
    }
    for key in lifecycle_keys:
        report[key] = request[key]
    canonical_json_bytes(report)
    return report


def validate_phase6_preflight(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen Phase-6 request without importing torch or CUDA code."""

    if not isinstance(request, Mapping):
        raise PreflightValidationError("preflight request must be a JSON object")
    return _validate_with_policy(
        request,
        policy=_FROZEN_POLICY,
        file_hasher=_StableHashMemo(),
    )


def load_canonical_preflight_request(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read a canonical request from a stable regular file."""

    absolute = os.path.abspath(os.fspath(path))
    observed, data = _observe_file(absolute, "preflight request", capture=True)
    if expected_sha256 is not None:
        expected = _require_sha256(
            expected_sha256,
            "expected preflight-request SHA256",
            reject_placeholder=True,
        )
        if not hmac.compare_digest(observed["sha256"], expected):
            raise PreflightValidationError(
                "preflight-request SHA mismatch: expected "
                f"{expected}, got {observed['sha256']}"
            )
    if data is None:
        raise AssertionError("captured request bytes are absent")
    value = _strict_json_loads(data, "preflight request")
    if not isinstance(value, dict):
        raise PreflightValidationError("preflight request must be a JSON object")
    if data != canonical_json_bytes(value):
        raise PreflightValidationError("preflight request is not canonical JSON")
    return value


def load_canonical_source_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a source manifest with strict duplicate/non-finite rejection."""

    absolute = os.path.abspath(os.fspath(path))
    _observed, data = _observe_file(absolute, "source manifest", capture=True)
    if data is None:
        raise AssertionError("captured source-manifest bytes are absent")
    value = _strict_json_loads(data, "source manifest")
    if not isinstance(value, dict):
        raise PreflightValidationError("source manifest must be a JSON object")
    if data != canonical_json_bytes(value):
        raise PreflightValidationError("source manifest is not canonical JSON")
    return value


def load_canonical_preflight_report(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Read one immutable canonical preflight report by exact SHA256."""

    absolute = os.path.abspath(os.fspath(path))
    observed, data = _observe_file(absolute, "preflight report", capture=True)
    expected = _require_sha256(
        expected_sha256,
        "expected preflight-report SHA256",
        reject_placeholder=True,
    )
    if not hmac.compare_digest(observed["sha256"], expected):
        raise PreflightValidationError("preflight-report SHA mismatch")
    if data is None:
        raise AssertionError("captured preflight-report bytes are absent")
    value = _strict_json_loads(data, "preflight report")
    if not isinstance(value, dict) or data != canonical_json_bytes(value):
        raise PreflightValidationError("preflight report is not canonical JSON")
    if value.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise PreflightValidationError("preflight report schema differs")
    return value


def _write_canonical_mapping(
    path: str | os.PathLike[str], value: Mapping[str, Any]
) -> str:
    destination = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(destination)
    if not os.path.isdir(parent):
        raise PreflightValidationError(f"report parent does not exist: {parent}")
    payload = canonical_json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise PreflightValidationError(
                f"refusing to overwrite registered artifact: {destination}"
            ) from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return sha256(payload).hexdigest()


def write_canonical_report(
    path: str | os.PathLike[str], report: Mapping[str, Any]
) -> str:
    """Publish a canonical report exactly once without overwriting."""

    return _write_canonical_mapping(path, report)


def write_phase6_source_manifest(
    path: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> str:
    """Atomically write a source inventory built after all code is frozen."""

    return _write_canonical_mapping(path, manifest)


def write_phase6_runtime_support_manifest(
    path: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> str:
    """Publish the exact runtime-support manifest once."""

    expected = build_phase6_runtime_support_manifest()
    if dict(manifest) != expected:
        raise PreflightValidationError(
            "refusing to publish a non-frozen runtime-support manifest"
        )
    return _write_canonical_mapping(path, expected)


def run_phase6_preflight(
    request_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str],
    *,
    expected_request_sha256: str,
    expected_purpose: str,
) -> dict[str, Any]:
    """Mandatory launcher/Trainer entry point: validate, then atomically report."""

    request = load_canonical_preflight_request(
        request_path, expected_sha256=expected_request_sha256
    )
    if expected_purpose not in ("training", "smoke", "reference_precompute"):
        raise PreflightValidationError("expected preflight purpose is invalid")
    if request.get("purpose") != expected_purpose:
        raise PreflightValidationError(
            f"preflight purpose mismatch: expected {expected_purpose!r}"
        )
    output_root = Path(request["effective_output_root"]).resolve(strict=True)
    report_destination = Path(report_path).expanduser().absolute()
    report_parent = report_destination.parent.resolve(strict=True)
    resolved_destination = report_parent / report_destination.name
    if os.path.commonpath((str(resolved_destination), str(output_root))) != str(
        output_root
    ):
        raise PreflightValidationError(
            "preflight report must be inside the authorized output root"
        )
    report = validate_phase6_preflight(request)
    if report.get("purpose") != expected_purpose:
        raise PreflightValidationError("validated preflight report purpose changed")
    write_canonical_report(report_path, report)
    return report


def load_and_revalidate_phase6_preflight_report(
    request_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str],
    *,
    expected_request_sha256: str,
    expected_report_sha256: str,
    expected_purpose: str,
) -> dict[str, Any]:
    """Stable-read an issued report and reproduce it without rewriting bytes."""

    if expected_purpose not in ("training", "smoke", "reference_precompute"):
        raise PreflightValidationError("expected preflight purpose is invalid")
    request = load_canonical_preflight_request(
        request_path, expected_sha256=expected_request_sha256
    )
    if request.get("purpose") != expected_purpose:
        raise PreflightValidationError(
            f"preflight purpose mismatch: expected {expected_purpose!r}"
        )
    output_root = Path(request["effective_output_root"]).resolve(strict=True)
    resolved_report = Path(report_path).resolve(strict=True)
    if os.path.commonpath((str(resolved_report), str(output_root))) != str(output_root):
        raise PreflightValidationError(
            "preflight report must be inside the authorized output root"
        )
    observed_report = load_canonical_preflight_report(
        report_path, expected_sha256=expected_report_sha256
    )
    reproduced_report = validate_phase6_preflight(request)
    if canonical_json_bytes(observed_report) != canonical_json_bytes(reproduced_report):
        raise PreflightValidationError(
            "persisted preflight report differs from fresh validation"
        )
    return observed_report


__all__ = [
    "PreflightValidationError",
    "REPORT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "RUNTIME_SUPPORT_SCHEMA_VERSION",
    "build_phase6_runtime_support_manifest",
    "build_phase6_source_manifest",
    "canonical_json_bytes",
    "load_canonical_preflight_request",
    "load_canonical_preflight_report",
    "load_and_revalidate_phase6_preflight_report",
    "load_canonical_source_manifest",
    "run_phase6_preflight",
    "validate_phase6_preflight",
    "write_canonical_report",
    "write_phase6_runtime_support_manifest",
    "write_phase6_source_manifest",
]
