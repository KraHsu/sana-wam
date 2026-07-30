"""Fail-closed governance for the fresh Phase-6 plan-row recovery cohort.

The graph is acyclic: a baseline snapshot precedes a tokenized-source proposal;
a review subject binds that proposal; two independent reviews bind only the
subject/proposal; the final authority binds all four; and only then is its SHA
rendered into this module. Runtime validation reconstructs the reviewed token.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

REPOSITORY_ROOT = "/home/zch/workspace/sana-wam"
OPERATIONAL_ROOT = "/DATA/share/sana_phase6_principled_constraints_20260724"
RECOVERY_EVIDENCE_ROOT = f"{OPERATIONAL_ROOT}/integration_patches_v5/phase6_plan_row_envelope_recovery_20260725_v1"
PRIOR_RECOVERY_ROOT = f"{OPERATIONAL_ROOT}/integration_patches_v5/phase6_plan_row_envelope_recovery_20260725_v4"
RECOVERY_ROOT = f"{OPERATIONAL_ROOT}/integration_patches_v5/phase6_plan_row_envelope_recovery_20260725_v5"
RECOVERY_BASELINE_SNAPSHOT_PATH = (
    f"{RECOVERY_ROOT}/recovery_source_before_snapshot.json"
)
RECOVERY_PROPOSAL_PATH = f"{RECOVERY_ROOT}/planrow_recovery_proposal.json"
RECOVERY_REVIEW_SUBJECT_PATH = f"{RECOVERY_ROOT}/planrow_recovery_review_subject.json"
RECOVERY_RENDER_PLAN_PATH = f"{RECOVERY_ROOT}/phase6_recovery_source_render_plan.json"
RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH = (
    f"{RECOVERY_ROOT}/source_recovery_execution_receipt.json"
)
RECOVERY_REVIEW_RECEIPT_PATHS = {
    "A": f"{RECOVERY_ROOT}/planrow_recovery_reviewer_a_receipt.json",
    "B": f"{RECOVERY_ROOT}/planrow_recovery_reviewer_b_receipt.json",
}
PLANROW_RECOVERY_AUTHORITY_PATH = f"{RECOVERY_ROOT}/planrow_recovery_authority.json"
PLANROW_RECOVERY_AUTHORITY_SHA256 = "9eaac4f81ca9220e03674c47c16a76d326e81ad85366b3e9261386e1d65eeb1c"
_PLANROW_RECOVERY_AUTHORITY_SHA256_TOKEN = (
    "@PHASE6_PLANROW_" + "RECOVERY_AUTHORITY_SHA256@"
)
RECOVERY_BASELINE_SCHEMA_VERSION = "sana-phase6-recovery-source-baseline-v1"
RECOVERY_PROPOSAL_SCHEMA_VERSION = "sana-phase6-planrow-recovery-proposal-v1"
RECOVERY_REVIEW_SUBJECT_SCHEMA_VERSION = (
    "sana-phase6-planrow-recovery-review-subject-v1"
)
RECOVERY_REVIEW_SCHEMA_VERSION = "sana-phase6-planrow-recovery-review-v1"
RECOVERY_AUTHORITY_SCHEMA_VERSION = "sana-phase6-planrow-recovery-authority-v1"
RECOVERY_RENDER_PLAN_SCHEMA_VERSION = "sana-phase6-recovery-source-render-plan-v1"
RECOVERY_SOURCE_INVENTORY_SCHEMA_VERSION = "sana-phase6-tokenized-source-inventory-v1"
RECOVERY_SOURCE_PATH = "src/sana_wam/train/phase6_recovery.py"
SOURCE_INVENTORY_GLOBS = (
    ".python-version",
    "configs/phase6/templates/*.yaml.in",
    "pyproject.toml",
    "scripts/**/*.py",
    "src/**/*.py",
    "tests/**/*.py",
    "third_party/Sana/**/*.py",
    "uv.lock",
)

PRIOR_SOURCE_REPAIR_AMENDMENT_PATH = f"{OPERATIONAL_ROOT}/integration_patches_v4/phase6_launch_runtime_symlink_source_repair_20260724_v2/source_repair_amendment.json"
PRIOR_SOURCE_REPAIR_AMENDMENT_SHA256 = (
    "623390e935dcca25d62cdc532109246c589170186ff39d111803ebfc9b674d41"
)
PRIOR_SOURCE_REPAIR_AMENDMENT_SIZE = 8096
PRIOR_SOURCE_MANIFEST_PATH = (
    f"{OPERATIONAL_ROOT}/artifacts/phase6_source_runtime_manifest_v8.json"
)
PRIOR_SOURCE_MANIFEST_SHA256 = (
    "be449247fdda960bace72a438f02187354fe814608cce575f5b869ebaf9dac3b"
)
PRIOR_SOURCE_MANIFEST_SIZE = 151355
PRIOR_RECOVERY_AUTHORITY = {
    "path": f"{PRIOR_RECOVERY_ROOT}/planrow_recovery_authority.json",
    "sha256": "d8985ade8171830ffb8313a24ba0f7c1cfdca303b19bcbfe87e4e1e13246fe5f",
}
PRIOR_RECOVERY_CANDIDATE_ROOT = "/tmp/phase6-recovery-v4-final-candidate-20260725-2027"
PRIOR_RECOVERY_AUTHORITY_PIN = {
    "mode": "0400",
    **PRIOR_RECOVERY_AUTHORITY,
    "size_bytes": 14246,
}
PRIOR_RECOVERY_PROPOSAL_PIN = {
    "mode": "0400",
    "path": f"{PRIOR_RECOVERY_ROOT}/planrow_recovery_proposal.json",
    "sha256": "d7bdf4765eecde221c88391f66200b5192eb4b1564b68e27937196c531983083",
    "size_bytes": 100774,
}
PRIOR_RECOVERY_RENDER_PLAN_PIN = {
    "mode": "0400",
    "path": f"{PRIOR_RECOVERY_ROOT}/phase6_recovery_source_render_plan.json",
    "sha256": "10b18b1426cf78f6586a1e0b08ea8486728c4a5cc2b9c6b5456d01dc5df2d583",
    "size_bytes": 1141,
}
PRIOR_SOURCE_EXECUTION_RECEIPT_PIN = {
    "mode": "0400",
    "path": f"{PRIOR_RECOVERY_ROOT}/source_recovery_execution_receipt.json",
    "sha256": "fa641d998e642364c56e41d190a49eda69047e7be243505ee210c0d22f8430c1",
    "size_bytes": 96612,
}
PRIOR_LAUNCH_FAILURE_RECEIPT_PIN = {
    "mode": "0400",
    "path": f"{PRIOR_RECOVERY_ROOT}/launch_preflight_failure_receipt_v1.json",
    "sha256": "a4f4c2ec63d7f2c4f99c76c182eaff2d1ebc2182f64c29ebda4dc40d295e0df4",
    "size_bytes": 6050,
}
PRIOR_TEMPLATE_MODE_RECEIPT_PIN = {
    "mode": "0400",
    "path": f"{PRIOR_RECOVERY_ROOT}/launch_retry_template_mode_receipt_v1.json",
    "sha256": "d61eaf2dd6f315d5aa88857ae6f799b68c1174642e89d6bf5b4b3625ac84b6f8",
    "size_bytes": 2412,
}
FAILED_RUN_INVENTORY_PATH = (
    f"{RECOVERY_EVIDENCE_ROOT}/failed_run_readonly_inventory_v1.json"
)
FAILED_RUN_INVENTORY_SHA256 = (
    "846cf943cb6d6f142a41795022e4f700f0e8b325733a508b31c01e451ed956ef"
)
FAILED_RUN_INVENTORY_SIZE = 64333
FAILED_INVENTORY_SUPPORT_BUNDLE_PATH = (
    f"{RECOVERY_EVIDENCE_ROOT}/failed_inventory_bootstrap_receipt.json"
)
FAILED_INVENTORY_SUPPORT_BUNDLE_SHA256 = (
    "1bab2644aae743c14970fea6bc4f3e0c5f05160e785d8e9673d9884ef37e5a0a"
)
FAILED_INVENTORY_SUPPORT_BUNDLE_SIZE = 4085
FAILED_INVENTORY_REVIEW_SUBJECT_PATH = (
    f"{RECOVERY_EVIDENCE_ROOT}/failed_inventory_review_subject.json"
)
FAILED_INVENTORY_REVIEW_SUBJECT_SHA256 = (
    "cbea6a4144d3e5093be180f8ac6cad994e27f3bdbd15067aaf75531984d9f6cb"
)
FAILED_INVENTORY_REVIEW_SUBJECT_SIZE = 2609
FAILED_INVENTORY_REVIEW_RECEIPT_PINS = {
    "A": {
        "path": f"{RECOVERY_EVIDENCE_ROOT}/failed_inventory_reviewer_a_receipt.json",
        "sha256": "2925fdd84af59a3867a545646be6f3a4a544c0959e493b5bf1cb1b30d6211d34",
        "size_bytes": 1741,
    },
    "B": {
        "path": f"{RECOVERY_EVIDENCE_ROOT}/failed_inventory_reviewer_b_receipt.json",
        "sha256": "5c3eda5c5238727fc944d2b8bae40d7b3773b83c0933aa2ac4066502461cf51e",
        "size_bytes": 1747,
    },
}
ROWFIX_SUPPORT_BUNDLE_PATH = (
    f"{RECOVERY_EVIDENCE_ROOT}/planrow_rowfix_review_bootstrap_receipt.json"
)
ROWFIX_SUPPORT_BUNDLE_SHA256 = (
    "7dcb6cd784c2b99de233c9ab2207e6b34d89b9ff3d65c4021844b80f0f77a296"
)
ROWFIX_SUPPORT_BUNDLE_SIZE = 3185
ROWFIX_REVIEW_SUBJECT_PATH = (
    f"{RECOVERY_EVIDENCE_ROOT}/planrow_rowfix_review_subject.json"
)
ROWFIX_REVIEW_SUBJECT_SHA256 = (
    "e1cc7e30dc862a54ae3c9200b9f52764c96a0870329db9f9f66876ee11961caa"
)
ROWFIX_REVIEW_SUBJECT_SIZE = 3223
ROWFIX_REVIEW_RECEIPT_PINS = {
    "code": {
        "path": f"{RECOVERY_EVIDENCE_ROOT}/planrow_rowfix_code_review_receipt.json",
        "sha256": "eb1b26fa7c7482bbb16450b1b03516e4497aa5aaf415bb2b9a64ef7d50ba7ee9",
        "size_bytes": 1852,
    },
    "test": {
        "path": f"{RECOVERY_EVIDENCE_ROOT}/planrow_rowfix_test_review_receipt.json",
        "sha256": "49e053e9744f2482a3ed76c65d2e9b223d2e485d415e6dc7f2d3601ec4b86474",
        "size_bytes": 1852,
    },
}
PUBLISHED_ROWFIX_SOURCE_SHA256 = (
    "92847b3187abd5e790a9f19e46198ddc7e5aba45ce986c8f94bc60ffaa73a86d"
)
PUBLISHED_ROWFIX_TEST_SHA256 = (
    "59778ccb485702cf7e210304fb835e2e9c6ea47b71461f65c82d74a4d7d94b44"
)

ARM_NAMES = ("T0_E0A0", "T1_E0A0", "T1_E1A0", "T1_E0A1", "T1_E1A1")
RECOVERY_RUN_IDS = {arm: f"phase6_20260725_{arm}_planrowfix_r2" for arm in ARM_NAMES}
RECOVERY_RUN_DIRECTORIES = {
    arm: f"{OPERATIONAL_ROOT}/arms/{arm}/{RECOVERY_RUN_IDS[arm]}" for arm in ARM_NAMES
}
ALLOWED_PROJECTION_LEAF_CHANGES = ("training.output_dir", "training.phase6_run_id")
RECOVERY_VERSIONED_OUTPUTS = {
    "arm_projection_manifest": f"{OPERATIONAL_ROOT}/artifacts/phase6_arm_projection_manifest_v6.json",
    "evidence_rebinding_receipt": f"{RECOVERY_ROOT}/evidence_rebinding_receipt_v6.json",
    "generated_config_root": f"{REPOSITORY_ROOT}/configs/phase6/generated_v8",
    "launch_manifest": f"{OPERATIONAL_ROOT}/launch/phase6_launch_manifest_v8.json",
    "materialization_pins": f"{OPERATIONAL_ROOT}/artifacts/phase6_arm_materialization_pins_v8.json",
    "materialized_config_root": f"{REPOSITORY_ROOT}/configs/phase6/materialized_v8",
    "projection_equivalence_receipt": f"{OPERATIONAL_ROOT}/artifacts/phase6_arm_projection_scientific_equivalence_v5.json",
    "reference_artifact": f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_v9.json",
    "reference_build_manifest": f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_v9_build_manifest.json",
    "reference_preflight_report": f"{OPERATIONAL_ROOT}/preflight/reference_precompute_report_v10.json",
    "reference_preflight_request": f"{OPERATIONAL_ROOT}/preflight/reference_precompute_request_v10.json",
    "reference_rebind_receipt": f"{RECOVERY_ROOT}/reference_v9_cpu_rebind_receipt.json",
    "reference_spot_artifact": f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_spot_v9.json",
    "smoke_gate": f"{OPERATIONAL_ROOT}/smoke/phase6_real_2b_smoke_gate_v8.json",
    "smoke_preflight_report": f"{OPERATIONAL_ROOT}/preflight/smoke_report_v10.json",
    "smoke_preflight_request": f"{OPERATIONAL_ROOT}/preflight/smoke_request_v10.json",
    "smoke_rebind_receipt": f"{RECOVERY_ROOT}/smoke_v8_cpu_rebind_receipt.json",
    "source_execution_receipt": RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH,
    "source_render_plan": RECOVERY_RENDER_PLAN_PATH,
    "source_manifest": f"{OPERATIONAL_ROOT}/artifacts/phase6_source_runtime_manifest_v9.json",
    "ticket_root": f"{OPERATIONAL_ROOT}/launch/tickets_v8",
    "training_preflight_reports": {
        arm: f"{OPERATIONAL_ROOT}/preflight/training_{arm}_report_v10.json"
        for arm in ARM_NAMES
    },
    "training_preflight_requests": {
        arm: f"{OPERATIONAL_ROOT}/preflight/training_{arm}_request_v10.json"
        for arm in ARM_NAMES
    },
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_REGISTERED_BYTES = 128 * 1024 * 1024
_MAX_SOURCE_BYTES = 128 * 1024 * 1024
_PIN_KEYS = {"mode", "path", "sha256", "size_bytes"}
_FULL_PIN_KEYS = {
    "device",
    "gid",
    "inode",
    "mode",
    "nlink",
    "path",
    "sha256",
    "size_bytes",
    "uid",
}


class Phase6RecoveryAuthorityError(ValueError):
    """Raised when recovery governance or one of its immutable pins differs."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise Phase6RecoveryAuthorityError(
            f"cannot encode canonical JSON: {exc}"
        ) from exc


def _strict_json(data: bytes, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise Phase6RecoveryAuthorityError(
                    f"duplicate JSON key in {name}: {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON: {token}")
            ),
        )
    except (UnicodeError, ValueError, TypeError) as exc:
        raise Phase6RecoveryAuthorityError(f"invalid {name}: {exc}") from exc
    if type(value) is not dict or data != canonical_json_bytes(value):
        raise Phase6RecoveryAuthorityError(f"{name} must be one canonical JSON object")
    return value


def _exact(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise Phase6RecoveryAuthorityError(f"{name} keys differ")
    return value


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise Phase6RecoveryAuthorityError(f"{name} must be a lowercase SHA256")
    return value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise Phase6RecoveryAuthorityError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(ord(c) < 0x20 for c in value)
    ):
        raise Phase6RecoveryAuthorityError(
            f"{name} must be a normalized non-empty string"
        )
    return value


def _utc(value: Any, name: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise Phase6RecoveryAuthorityError(f"{name} must be a UTC timestamp")
    try:
        observed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Phase6RecoveryAuthorityError(f"{name} differs") from exc
    if observed > datetime.now(timezone.utc):
        raise Phase6RecoveryAuthorityError(f"{name} is future-dated")
    return observed


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    base = (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )
    # Windows may refresh creation/change metadata while a read-only handle is
    # opened; production POSIX validation retains the stronger ctime binding.
    return base + ((value.st_ctime_ns,) if os.name == "posix" else ())


def _mode_matches(observed: int, expected: int) -> bool:
    mode = stat.S_IMODE(observed)
    return mode == (0o444 if os.name == "nt" and expected == 0o400 else expected)


def _typed_equal(observed: Any, expected: Any, name: str) -> None:
    if type(observed) is not type(expected):
        raise Phase6RecoveryAuthorityError(f"{name} type differs")
    if type(expected) is dict:
        if set(observed) != set(expected):
            raise Phase6RecoveryAuthorityError(f"{name} keys differ")
        for key in sorted(expected):
            _typed_equal(observed[key], expected[key], f"{name}.{key}")
    elif type(expected) is list:
        if len(observed) != len(expected):
            raise Phase6RecoveryAuthorityError(f"{name} length differs")
        for index, (actual, wanted) in enumerate(zip(observed, expected, strict=True)):
            _typed_equal(actual, wanted, f"{name}[{index}]")
    elif observed != expected:
        raise Phase6RecoveryAuthorityError(f"{name} differs")


def validate_recovery_lifecycle(
    value: Any, name: str = "recovery lifecycle", *, exact: bool = True
) -> dict[str, Any]:
    expected = recovery_lifecycle()
    if type(value) is not dict or not set(expected).issubset(value):
        raise Phase6RecoveryAuthorityError(f"{name} keys differ")
    if exact and set(value) != set(expected):
        raise Phase6RecoveryAuthorityError(f"{name} keys differ")
    for key, wanted in expected.items():
        _typed_equal(value[key], wanted, f"{name}.{key}")
    return deepcopy(value)


def _stable_read_fd(
    path_value: str,
    *,
    name: str,
    expected_size: int | None,
    expected_sha256: str | None,
    expected_mode: int | None,
    require_single_link: bool,
    maximum_bytes: int,
    allow_empty: bool = False,
) -> tuple[dict[str, Any], bytes]:
    if type(path_value) is not str or not os.path.isabs(path_value):
        raise Phase6RecoveryAuthorityError(f"{name} path must be absolute")
    path = Path(path_value)
    descriptor = -1
    try:
        resolved_before = os.path.realpath(path)
        path_before = os.lstat(path)
        if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
            raise Phase6RecoveryAuthorityError(
                f"{name} must be a non-symlink regular file"
            )
        if expected_mode is not None and not _mode_matches(
            path_before.st_mode, expected_mode
        ):
            raise Phase6RecoveryAuthorityError(f"{name} mode differs")
        if require_single_link and path_before.st_nlink != 1:
            raise Phase6RecoveryAuthorityError(f"{name} link count differs")
        if expected_size is not None and path_before.st_size != expected_size:
            raise Phase6RecoveryAuthorityError(f"{name} size differs")
        if (
            path_before.st_size < 1 and not allow_empty
        ) or path_before.st_size > maximum_bytes:
            raise Phase6RecoveryAuthorityError(f"{name} size is outside the limit")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if os.name == "posix" and nofollow == 0:
            raise Phase6RecoveryAuthorityError(
                "O_NOFOLLOW is required for governance reads"
            )
        descriptor = os.open(path, flags | nofollow)
        fd_before = os.fstat(descriptor)
        if (
            (path_before.st_dev, path_before.st_ino)
            != (fd_before.st_dev, fd_before.st_ino)
            or not stat.S_ISREG(fd_before.st_mode)
            or (
                expected_mode is not None
                and not _mode_matches(fd_before.st_mode, expected_mode)
            )
            or (require_single_link and fd_before.st_nlink != 1)
            or (expected_size is not None and fd_before.st_size != expected_size)
        ):
            raise Phase6RecoveryAuthorityError(f"{name} descriptor binding differs")
        chunks: list[bytes] = []
        remaining = fd_before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise Phase6RecoveryAuthorityError(f"{name} ended before its stat size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise Phase6RecoveryAuthorityError(f"{name} grew while reading")
        fd_after = os.fstat(descriptor)
        path_after = os.lstat(path)
        resolved_after = os.path.realpath(path)
    except Phase6RecoveryAuthorityError:
        raise
    except OSError as exc:
        raise Phase6RecoveryAuthorityError(f"cannot read {name}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        resolved_before != resolved_after
        or _stat_signature(path_before) != _stat_signature(fd_before)
        or _stat_signature(fd_before) != _stat_signature(fd_after)
        or _stat_signature(fd_after) != _stat_signature(path_after)
    ):
        raise Phase6RecoveryAuthorityError(f"{name} changed while reading")
    data = b"".join(chunks)
    observed_sha = sha256(data).hexdigest()
    if expected_sha256 is not None and not hmac.compare_digest(
        observed_sha, expected_sha256
    ):
        raise Phase6RecoveryAuthorityError(f"{name} SHA256 differs")
    return (
        {
            "mode": f"{stat.S_IMODE(path_before.st_mode):04o}",
            "path": path_value,
            "sha256": observed_sha,
            "size_bytes": len(data),
        },
        data,
    )


def _pin(value: Any, name: str, *, expected_mode: str = "0400") -> dict[str, Any]:
    pin = _exact(value, set(_PIN_KEYS), f"{name} pin")
    if pin["mode"] != expected_mode:
        raise Phase6RecoveryAuthorityError(f"{name} registered mode differs")
    if type(pin["path"]) is not str or not os.path.isabs(pin["path"]):
        raise Phase6RecoveryAuthorityError(f"{name} path must be absolute")
    _digest(pin["sha256"], f"{name} SHA256")
    _positive_int(pin["size_bytes"], f"{name} size")
    return pin


def _stable_registered_pin(value: Any, name: str) -> tuple[dict[str, Any], bytes]:
    pin = _pin(value, name)
    _observed, data = _stable_read_fd(
        pin["path"],
        name=name,
        expected_size=pin["size_bytes"],
        expected_sha256=pin["sha256"],
        expected_mode=0o400,
        require_single_link=True,
        maximum_bytes=_MAX_REGISTERED_BYTES,
        allow_empty=False,
    )
    return pin, data


def _stable_repository_file(path: Path, relative: str) -> tuple[dict[str, Any], bytes]:
    observed, data = _stable_read_fd(
        os.path.abspath(path),
        name=f"source inventory {relative}",
        expected_size=None,
        expected_sha256=None,
        expected_mode=None,
        require_single_link=False,
        maximum_bytes=_MAX_SOURCE_BYTES,
        allow_empty=True,
    )
    return {
        "path": relative,
        "sha256": observed["sha256"],
        "size_bytes": observed["size_bytes"],
    }, data


def _expected_supporting_bundles() -> dict[str, dict[str, Any]]:
    return {
        "failed_inventory_bootstrap": {
            "mode": "0400",
            "path": FAILED_INVENTORY_SUPPORT_BUNDLE_PATH,
            "sha256": FAILED_INVENTORY_SUPPORT_BUNDLE_SHA256,
            "size_bytes": FAILED_INVENTORY_SUPPORT_BUNDLE_SIZE,
        },
        "failed_inventory_review_A": {
            "mode": "0400",
            **FAILED_INVENTORY_REVIEW_RECEIPT_PINS["A"],
        },
        "failed_inventory_review_B": {
            "mode": "0400",
            **FAILED_INVENTORY_REVIEW_RECEIPT_PINS["B"],
        },
        "failed_inventory_review_subject": {
            "mode": "0400",
            "path": FAILED_INVENTORY_REVIEW_SUBJECT_PATH,
            "sha256": FAILED_INVENTORY_REVIEW_SUBJECT_SHA256,
            "size_bytes": FAILED_INVENTORY_REVIEW_SUBJECT_SIZE,
        },
        "failed_run_inventory": {
            "mode": "0400",
            "path": FAILED_RUN_INVENTORY_PATH,
            "sha256": FAILED_RUN_INVENTORY_SHA256,
            "size_bytes": FAILED_RUN_INVENTORY_SIZE,
        },
        "rowfix_review_code": {"mode": "0400", **ROWFIX_REVIEW_RECEIPT_PINS["code"]},
        "rowfix_review_bootstrap": {
            "mode": "0400",
            "path": ROWFIX_SUPPORT_BUNDLE_PATH,
            "sha256": ROWFIX_SUPPORT_BUNDLE_SHA256,
            "size_bytes": ROWFIX_SUPPORT_BUNDLE_SIZE,
        },
        "rowfix_review_subject": {
            "mode": "0400",
            "path": ROWFIX_REVIEW_SUBJECT_PATH,
            "sha256": ROWFIX_REVIEW_SUBJECT_SHA256,
            "size_bytes": ROWFIX_REVIEW_SUBJECT_SIZE,
        },
        "rowfix_review_test": {"mode": "0400", **ROWFIX_REVIEW_RECEIPT_PINS["test"]},
    }


def _expected_prior_provenance() -> dict[str, dict[str, Any]]:
    return {
        "launch_preflight_failure_v4": deepcopy(PRIOR_LAUNCH_FAILURE_RECEIPT_PIN),
        "launch_retry_template_modes_v4": deepcopy(PRIOR_TEMPLATE_MODE_RECEIPT_PIN),
        "source_execution_receipt_v4": deepcopy(PRIOR_SOURCE_EXECUTION_RECEIPT_PIN),
        "source_manifest_v8": {
            "mode": "0400",
            "path": PRIOR_SOURCE_MANIFEST_PATH,
            "sha256": PRIOR_SOURCE_MANIFEST_SHA256,
            "size_bytes": PRIOR_SOURCE_MANIFEST_SIZE,
        },
        "source_repair_amendment": {
            "mode": "0400",
            "path": PRIOR_SOURCE_REPAIR_AMENDMENT_PATH,
            "sha256": PRIOR_SOURCE_REPAIR_AMENDMENT_SHA256,
            "size_bytes": PRIOR_SOURCE_REPAIR_AMENDMENT_SIZE,
        },
    }


def recovery_lifecycle() -> dict[str, Any]:
    """Return the complete historical and current recovery lifecycle."""
    return {
        "historical_formal_training_started": True,
        "historical_optimizer_step_calls_per_arm": 1,
        "historical_optimizer_step_calls_total": 5,
        "historical_scheduled_lr_scale_at_step0": 0.0,
        "historical_nonzero_weight_update_proven": False,
        "historical_durable_optimizer_state": False,
        "historical_durable_metrics_rows_total": 0,
        "historical_checkpoint_count": 0,
        "historical_summary_count": 0,
        "historical_completion_count": 0,
        "historical_old_cohort_completed": False,
        "historical_old_cohort_resumable": False,
        "historical_old_cohort_retryable_in_place": False,
        "historical_scientific_results_available": False,
        "recovery_cohort_started": False,
        "recovery_closed_loop_started": False,
    }


def recovery_authority_lifecycle() -> dict[str, Any]:
    """Compatibility alias for callers that need the full lifecycle."""
    return recovery_lifecycle()


def _scientific_equivalence_contract() -> dict[str, Any]:
    return {
        "allowed_projection_leaf_changes": list(ALLOWED_PROJECTION_LEAF_CHANGES),
        "dataset_plan_identity_frozen": True,
        "model_optimizer_seed_checkpoint_frozen": True,
        "resume_allowed": False,
        "run_directories": dict(RECOVERY_RUN_DIRECTORIES),
        "run_ids": dict(RECOVERY_RUN_IDS),
    }


def _source_render_contract() -> dict[str, Any]:
    return {
        "algorithm": "single-ascii-token-replacement-v1",
        "authority_sha256_token": _PLANROW_RECOVERY_AUTHORITY_SHA256_TOKEN,
        "replacement_count": 1,
        "repository_relative_path": RECOVERY_SOURCE_PATH,
    }


def render_phase6_recovery_source(
    tokenized_source: bytes, authority_sha256: str
) -> bytes:
    """Render exactly one final-authority SHA into reviewed tokenized source."""
    if not isinstance(tokenized_source, bytes):
        raise TypeError("tokenized recovery source must be bytes")
    digest = _digest(authority_sha256, "recovery authority render SHA256").encode(
        "ascii"
    )
    token = _PLANROW_RECOVERY_AUTHORITY_SHA256_TOKEN.encode("ascii")
    if tokenized_source.count(token) != 1 or tokenized_source.count(digest) != 0:
        raise Phase6RecoveryAuthorityError("recovery source token count differs")
    result = tokenized_source.replace(token, digest)
    if result.count(digest) != 1 or result.count(token) != 0:
        raise Phase6RecoveryAuthorityError("recovery source render count differs")
    return result


def _normalize_recovery_source(data: bytes, authority_sha256: str | None) -> bytes:
    token = _PLANROW_RECOVERY_AUTHORITY_SHA256_TOKEN.encode("ascii")
    if authority_sha256 is None:
        if data.count(token) != 1:
            raise Phase6RecoveryAuthorityError(
                "tokenized recovery source must contain exactly one authority token"
            )
        return data
    digest = _digest(authority_sha256, "rendered recovery authority SHA256").encode(
        "ascii"
    )
    if data.count(token) != 0 or data.count(digest) != 1:
        raise Phase6RecoveryAuthorityError("rendered recovery source binding differs")
    return data.replace(digest, token)


def _discover_source_inventory(
    repository_root: str, authority_sha256: str | None
) -> list[dict[str, Any]]:
    if type(repository_root) is not str or not os.path.isabs(repository_root):
        raise Phase6RecoveryAuthorityError("repository_root must be absolute")
    root = Path(repository_root)
    try:
        root_real = root.resolve(strict=True)
    except OSError as exc:
        raise Phase6RecoveryAuthorityError(
            f"cannot resolve repository_root: {exc}"
        ) from exc
    if not root_real.is_dir():
        raise Phase6RecoveryAuthorityError("repository_root must be a directory")
    paths: set[str] = set()
    for pattern in SOURCE_INVENTORY_GLOBS:
        for candidate in root.glob(pattern):
            paths.add(candidate.relative_to(root).as_posix())
    if RECOVERY_SOURCE_PATH not in paths:
        raise Phase6RecoveryAuthorityError("source inventory omits phase6_recovery.py")
    files: list[dict[str, Any]] = []
    for relative in sorted(paths):
        path = root.joinpath(*relative.split("/"))
        pin, data = _stable_repository_file(path, relative)
        try:
            resolved = path.resolve(strict=True)
            if os.path.commonpath(
                (os.fspath(root_real), os.fspath(resolved))
            ) != os.fspath(root_real):
                raise Phase6RecoveryAuthorityError(
                    f"source inventory escapes repository: {relative}"
                )
        except (OSError, ValueError) as exc:
            if isinstance(exc, Phase6RecoveryAuthorityError):
                raise
            raise Phase6RecoveryAuthorityError(
                f"cannot resolve source {relative}: {exc}"
            ) from exc
        if relative == RECOVERY_SOURCE_PATH:
            normalized = _normalize_recovery_source(data, authority_sha256)
            pin = {
                "path": relative,
                "sha256": sha256(normalized).hexdigest(),
                "size_bytes": len(normalized),
            }
        files.append(pin)
    return files


def _discover_baseline_inventory(repository_root: str) -> list[dict[str, Any]]:
    root = Path(repository_root)
    if type(repository_root) is not str or not os.path.isabs(repository_root):
        raise Phase6RecoveryAuthorityError("baseline repository_root must be absolute")
    try:
        root_real = root.resolve(strict=True)
    except OSError as exc:
        raise Phase6RecoveryAuthorityError(
            f"cannot resolve baseline repository: {exc}"
        ) from exc
    paths = {
        candidate.relative_to(root).as_posix()
        for pattern in SOURCE_INVENTORY_GLOBS
        for candidate in root.glob(pattern)
    }
    files: list[dict[str, Any]] = []
    for relative in sorted(paths):
        path = root.joinpath(*relative.split("/"))
        pin, _data = _stable_repository_file(path, relative)
        resolved = path.resolve(strict=True)
        if os.path.commonpath((os.fspath(root_real), os.fspath(resolved))) != os.fspath(
            root_real
        ):
            raise Phase6RecoveryAuthorityError(
                f"baseline source escapes repository: {relative}"
            )
        files.append(pin)
    if not files:
        raise Phase6RecoveryAuthorityError("baseline source inventory is empty")
    return files


def _file_pin_map(files: Any, name: str) -> dict[str, dict[str, Any]]:
    if type(files) is not list or not files:
        raise Phase6RecoveryAuthorityError(f"{name} files differ")
    result: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, entry in enumerate(files):
        label = f"{name} files[{index}]"
        _exact(entry, {"path", "sha256", "size_bytes"}, label)
        relative = _text(entry["path"], f"{label}.path")
        if relative.startswith("/") or ".." in relative.split("/"):
            raise Phase6RecoveryAuthorityError(
                f"{label}.path is not repository-relative"
            )
        pin = {
            "sha256": _digest(entry["sha256"], f"{label}.sha256"),
            "size_bytes": _positive_int(
                entry["size_bytes"], f"{label}.size_bytes", allow_zero=True
            ),
        }
        if relative in result:
            raise Phase6RecoveryAuthorityError(f"{name} path duplicates")
        result[relative] = pin
        order.append(relative)
    if order != sorted(order):
        raise Phase6RecoveryAuthorityError(f"{name} paths are not sorted")
    return result


def _require_baseline_matches_prior(
    baseline_files: Any, prior_files: Mapping[str, Mapping[str, Any]]
) -> None:
    baseline = _file_pin_map(baseline_files, "recovery source baseline")
    missing_or_changed = [
        path for path, pin in prior_files.items() if baseline.get(path) != pin
    ]
    if missing_or_changed:
        raise Phase6RecoveryAuthorityError(
            "baseline source differs from source-manifest-v8: "
            f"{missing_or_changed[:10]}"
        )


def build_phase6_recovery_baseline_snapshot(
    *, repository_root: str = REPOSITORY_ROOT, registered_at_utc: str
) -> dict[str, Any]:
    """Build the source-before snapshot against the still-formal repository."""
    if repository_root != REPOSITORY_ROOT:
        raise Phase6RecoveryAuthorityError("baseline target repository root differs")
    _utc(registered_at_utc, "baseline snapshot registration")
    prior_files = _validate_prior_provenance(_expected_prior_provenance())
    files = _discover_baseline_inventory(repository_root)
    _require_baseline_matches_prior(files, prior_files)
    return {
        "artifact_path": RECOVERY_BASELINE_SNAPSHOT_PATH,
        "artifact_role": "phase6_recovery_source_before_snapshot",
        "files": files,
        "inventory_globs": list(SOURCE_INVENTORY_GLOBS),
        "prior_source_manifest": deepcopy(
            _expected_prior_provenance()["source_manifest_v8"]
        ),
        "registered_at_utc": registered_at_utc,
        "registered_before_recovery_source_change": True,
        "repository_root": repository_root,
        "schema_version": RECOVERY_BASELINE_SCHEMA_VERSION,
        "status": "registered",
    }


def validate_phase6_recovery_baseline_snapshot_bytes(data: bytes) -> dict[str, Any]:
    baseline = _strict_json(data, "recovery source baseline")
    _exact(
        baseline,
        {
            "artifact_path",
            "artifact_role",
            "files",
            "inventory_globs",
            "prior_source_manifest",
            "registered_at_utc",
            "registered_before_recovery_source_change",
            "repository_root",
            "schema_version",
            "status",
        },
        "recovery source baseline",
    )
    if (
        baseline["schema_version"] != RECOVERY_BASELINE_SCHEMA_VERSION
        or baseline["artifact_role"] != "phase6_recovery_source_before_snapshot"
        or baseline["artifact_path"] != RECOVERY_BASELINE_SNAPSHOT_PATH
        or baseline["repository_root"] != REPOSITORY_ROOT
        or baseline["inventory_globs"] != list(SOURCE_INVENTORY_GLOBS)
        or baseline["prior_source_manifest"]
        != _expected_prior_provenance()["source_manifest_v8"]
        or baseline["registered_before_recovery_source_change"] is not True
        or baseline["status"] != "registered"
    ):
        raise Phase6RecoveryAuthorityError("recovery source baseline identity differs")
    _utc(baseline["registered_at_utc"], "baseline snapshot registration")
    prior_files = _validate_prior_provenance(_expected_prior_provenance())
    _require_baseline_matches_prior(baseline["files"], prior_files)
    return deepcopy(baseline)


def _validate_prior_source_manifest(data: bytes) -> dict[str, dict[str, Any]]:
    value = _strict_json(data, "prior source manifest")
    schema = value.get("schema_version")
    if schema == "sana-phase6-source-runtime-manifest-v2":
        _exact(
            value,
            {
                "files",
                "inventory_globs",
                "registered_before_phase6_training_results",
                "repository_root",
                "repository_vcs",
                "runtime_environment",
                "schema_version",
                "training_or_gpu_started",
            },
            "prior source manifest v4",
        )
        identity_matches = (
            value["registered_before_phase6_training_results"] is True
            and value["training_or_gpu_started"] is False
        )
    elif schema == "sana-phase6-source-runtime-manifest-v3":
        _exact(
            value,
            {
                "files",
                "inventory_globs",
                "recovery_authority",
                "registered_before_recovery_cohort_started",
                "repository_root",
                "repository_vcs",
                "runtime_environment",
                "schema_version",
                *recovery_lifecycle(),
            },
            "prior recovery source manifest v8",
        )
        identity_matches = (
            value["registered_before_recovery_cohort_started"] is True
            and value["recovery_authority"] == PRIOR_RECOVERY_AUTHORITY
            and all(
                type(value[key]) is type(expected) and value[key] == expected
                for key, expected in recovery_lifecycle().items()
            )
        )
    else:
        identity_matches = False
    if (
        not identity_matches
        or value.get("repository_root") != REPOSITORY_ROOT
        or type(value.get("files")) is not list
        or not value["files"]
    ):
        raise Phase6RecoveryAuthorityError("prior source manifest identity differs")
    paths: list[str] = []
    pins: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(value["files"]):
        label = f"prior source manifest files[{index}]"
        _exact(entry, {"path", "realpath", "sha256", "size_bytes"}, label)
        paths.append(_text(entry["path"], f"{label}.path"))
        _text(entry["realpath"], f"{label}.realpath")
        digest = _digest(entry["sha256"], f"{label}.sha256")
        size = _positive_int(
            entry["size_bytes"], f"{label}.size_bytes", allow_zero=True
        )
        pins[paths[-1]] = {"sha256": digest, "size_bytes": size}
    if paths != sorted(set(paths)):
        raise Phase6RecoveryAuthorityError("prior source manifest paths differ")
    return pins


def _validate_prior_source_execution_receipt(
    data: bytes, prior_source_files: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    value = _strict_json(data, "prior recovery source execution receipt")
    _exact(
        value,
        {
            "artifact_path",
            "artifact_role",
            "authority",
            "authorized_source_delta",
            "candidate_source_root",
            "files",
            "gpu_executed",
            "lifecycle",
            "proposal",
            "receipt_is_commit_marker",
            "registered_at_utc",
            "rendered_source_inventory",
            "schema_version",
            "source_render_plan",
            "source_write_completed",
            "status",
            "target_repository_root",
        },
        "prior recovery source execution receipt",
    )
    if (
        value["schema_version"] != "sana-phase6-recovery-source-execution-receipt-v1"
        or value["artifact_role"] != "phase6_recovery_source_execution_receipt"
        or value["artifact_path"] != PRIOR_SOURCE_EXECUTION_RECEIPT_PIN["path"]
        or value["authority"] != PRIOR_RECOVERY_AUTHORITY_PIN
        or value["proposal"] != PRIOR_RECOVERY_PROPOSAL_PIN
        or value["source_render_plan"] != PRIOR_RECOVERY_RENDER_PLAN_PIN
        or value["candidate_source_root"] != PRIOR_RECOVERY_CANDIDATE_ROOT
        or value["target_repository_root"] != REPOSITORY_ROOT
        or value["gpu_executed"] is not False
        or value["source_write_completed"] is not True
        or value["receipt_is_commit_marker"] is not True
        or value["status"] != "registered"
        or value["lifecycle"] != recovery_lifecycle()
    ):
        raise Phase6RecoveryAuthorityError(
            "prior recovery source execution receipt identity differs"
        )
    _utc(value["registered_at_utc"], "prior source execution receipt registration")

    inventory = _exact(
        value["rendered_source_inventory"],
        {"files", "inventory_globs", "representation", "schema_version"},
        "prior rendered source inventory",
    )
    if (
        inventory["schema_version"] != "sana-phase6-raw-rendered-source-inventory-v1"
        or inventory["representation"] != "raw-rendered"
        or inventory["inventory_globs"] != list(SOURCE_INVENTORY_GLOBS)
        or type(inventory["files"]) is not list
    ):
        raise Phase6RecoveryAuthorityError("prior rendered source inventory differs")
    expected_inventory = [
        {"path": path, **deepcopy(prior_source_files[path])}
        for path in sorted(prior_source_files)
    ]
    if inventory["files"] != expected_inventory:
        raise Phase6RecoveryAuthorityError(
            "prior rendered source inventory file pins differ"
        )

    if (
        type(value["files"]) is not list
        or type(value["authorized_source_delta"]) is not list
    ):
        raise Phase6RecoveryAuthorityError("prior source execution delta differs")
    modes: dict[str, int] = {}
    file_entries: dict[str, dict[str, Any]] = {}
    for index, entry_value in enumerate(value["files"]):
        label = f"prior source execution files[{index}]"
        entry = _exact(
            entry_value,
            {
                "before",
                "before_mode",
                "change",
                "path",
                "rendered",
                "rendered_mode",
                "state",
                "tokenized",
                "tokenized_mode",
            },
            label,
        )
        path = _text(entry["path"], f"{label}.path")
        if path not in prior_source_files or path in file_entries:
            raise Phase6RecoveryAuthorityError(f"{label}.path differs")
        if (
            entry["state"] != "rendered_final"
            or entry["rendered"] != prior_source_files[path]
            or entry["rendered_mode"] not in {"0644", "0664"}
            or entry["tokenized_mode"] != entry["rendered_mode"]
        ):
            raise Phase6RecoveryAuthorityError(f"{label} rendered identity differs")
        if entry["change"] == "added":
            if (
                entry["before"] is not None
                or entry["before_mode"] is not None
                or entry["rendered_mode"] != "0644"
            ):
                raise Phase6RecoveryAuthorityError(f"{label} added mode differs")
        elif entry["change"] == "modified":
            if (
                type(entry["before"]) is not dict
                or entry["before_mode"] not in {"0644", "0664"}
                or entry["rendered_mode"] != entry["before_mode"]
            ):
                raise Phase6RecoveryAuthorityError(f"{label} modified mode differs")
        else:
            raise Phase6RecoveryAuthorityError(f"{label}.change differs")
        modes[path] = int(entry["rendered_mode"], 8)
        file_entries[path] = entry
    if list(file_entries) != sorted(file_entries):
        raise Phase6RecoveryAuthorityError("prior source execution file order differs")

    delta_paths: list[str] = []
    for index, delta_value in enumerate(value["authorized_source_delta"]):
        label = f"prior authorized source delta[{index}]"
        delta = _exact(
            delta_value,
            {"before", "candidate", "change", "path"},
            label,
        )
        path = _text(delta["path"], f"{label}.path")
        entry = file_entries.get(path)
        if entry is None or (
            delta["before"] != entry["before"]
            or delta["candidate"] != entry["tokenized"]
            or delta["change"] != entry["change"]
        ):
            raise Phase6RecoveryAuthorityError(f"{label} receipt binding differs")
        delta_paths.append(path)
    if delta_paths != sorted(file_entries):
        raise Phase6RecoveryAuthorityError(
            "prior authorized source delta order differs"
        )
    return modes


def _validate_prior_template_mode_receipt(
    data: bytes, prior_source_files: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    value = _strict_json(data, "prior launch-retry template mode receipt")
    _exact(
        value,
        {
            "artifact_path",
            "artifact_role",
            "files",
            "launch_preflight_failure",
            "registered_at_utc",
            "registered_before_planrowfix_r2_source_change",
            "schema_version",
            "source_manifest_v8",
            "status",
        },
        "prior launch-retry template mode receipt",
    )
    expected = _expected_prior_provenance()
    if (
        value["schema_version"] != "sana-phase6-launch-retry-template-mode-receipt-v1"
        or value["artifact_role"] != "phase6_launch_retry_template_mode_provenance"
        or value["artifact_path"] != PRIOR_TEMPLATE_MODE_RECEIPT_PIN["path"]
        or value["launch_preflight_failure"] != PRIOR_LAUNCH_FAILURE_RECEIPT_PIN
        or value["source_manifest_v8"] != expected["source_manifest_v8"]
        or value["registered_before_planrowfix_r2_source_change"] is not True
        or value["status"] != "registered"
        or type(value["files"]) is not list
    ):
        raise Phase6RecoveryAuthorityError(
            "prior launch-retry template mode receipt identity differs"
        )
    _utc(value["registered_at_utc"], "prior template mode receipt registration")
    expected_paths = sorted(
        f"configs/phase6/templates/train_phase6_{arm}.yaml.in" for arm in ARM_NAMES
    )
    modes: dict[str, int] = {}
    for index, item in enumerate(value["files"]):
        label = f"prior template mode files[{index}]"
        entry = _exact(item, {"before", "mode", "path", "realpath"}, label)
        path = _text(entry["path"], f"{label}.path")
        if (
            index >= len(expected_paths)
            or path != expected_paths[index]
            or entry["before"] != prior_source_files.get(path)
            or entry["mode"] != "0664"
            or entry["realpath"]
            != os.fspath(Path(REPOSITORY_ROOT).joinpath(*path.split("/")).resolve())
        ):
            raise Phase6RecoveryAuthorityError(f"{label} identity differs")
        modes[path] = 0o664
    if list(modes) != expected_paths:
        raise Phase6RecoveryAuthorityError("prior template mode path set differs")
    return modes


def prior_recovery_source_modes() -> dict[str, int]:
    """Return modes authenticated by the immutable prior source commit receipt."""

    expected = _expected_prior_provenance()
    _manifest_pin, manifest_data = _stable_registered_pin(
        expected["source_manifest_v8"], "prior source_manifest_v8"
    )
    prior_source_files = _validate_prior_source_manifest(manifest_data)
    _receipt_pin, receipt_data = _stable_registered_pin(
        expected["source_execution_receipt_v4"],
        "prior source_execution_receipt_v4",
    )
    modes = _validate_prior_source_execution_receipt(receipt_data, prior_source_files)
    _template_pin, template_data = _stable_registered_pin(
        expected["launch_retry_template_modes_v4"],
        "prior launch_retry_template_modes_v4",
    )
    template_modes = _validate_prior_template_mode_receipt(
        template_data, prior_source_files
    )
    if set(modes) & set(template_modes):
        raise Phase6RecoveryAuthorityError("prior source mode authorities overlap")
    return {**modes, **template_modes}


def _full_receipt_pin(value: Any, name: str) -> dict[str, Any]:
    pin = _exact(value, set(_FULL_PIN_KEYS), name)
    _text(pin["path"], f"{name}.path")
    _digest(pin["sha256"], f"{name}.sha256")
    _positive_int(pin["size_bytes"], f"{name}.size_bytes", allow_zero=True)
    for key in ("device", "gid", "inode", "nlink", "uid"):
        _positive_int(pin[key], f"{name}.{key}", allow_zero=True)
    if type(pin["mode"]) is not str or re.fullmatch(r"[0-7]{4}", pin["mode"]) is None:
        raise Phase6RecoveryAuthorityError(f"{name}.mode differs")
    return pin


def _registered_projection(pin: Mapping[str, Any]) -> dict[str, Any]:
    return {key: pin[key] for key in ("mode", "path", "sha256", "size_bytes")}


def _validate_failed_inventory(data: bytes) -> None:
    inventory = _strict_json(data, "failed-run inventory")
    _exact(
        inventory,
        {
            "absence_contract",
            "aggregate_facts",
            "arms",
            "evidence_preservation",
            "failed_launch",
            "failure_signature",
            "host",
            "observed_at_utc",
            "old_source_pins",
            "operational_root",
            "quiescence_observation",
            "recovery_candidate_pins",
            "schema_version",
            "task_plan",
        },
        "failed-run inventory",
    )
    facts = _exact(
        inventory["aggregate_facts"],
        {
            "affected_arm_count",
            "checkpoint_count_total",
            "completion_manifest_count_total",
            "durable_metrics_rows_total",
            "durable_optimizer_state_available",
            "formal_training_started",
            "old_cohort_completed",
            "old_cohort_resumable",
            "old_cohort_retryable_in_place",
            "optimizer_step_calls_per_arm",
            "optimizer_step_calls_total",
            "scientific_results_available",
            "summary_count_total",
        },
        "failed-run aggregate facts",
    )
    expected = {
        "affected_arm_count": 5,
        "checkpoint_count_total": 0,
        "completion_manifest_count_total": 0,
        "durable_metrics_rows_total": 0,
        "durable_optimizer_state_available": False,
        "formal_training_started": True,
        "old_cohort_completed": False,
        "old_cohort_resumable": False,
        "old_cohort_retryable_in_place": False,
        "optimizer_step_calls_per_arm": 1,
        "optimizer_step_calls_total": 5,
        "scientific_results_available": False,
        "summary_count_total": 0,
    }
    if (
        inventory["schema_version"] != "phase6-failed-run-readonly-inventory-v1"
        or inventory["operational_root"] != OPERATIONAL_ROOT
        or facts != expected
    ):
        raise Phase6RecoveryAuthorityError("failed-run inventory lifecycle differs")
    arms = inventory["arms"]
    if type(arms) is not list or [
        entry.get("arm") for entry in arms if type(entry) is dict
    ] != list(ARM_NAMES):
        raise Phase6RecoveryAuthorityError("failed-run inventory arms differ")
    for arm, entry in zip(ARM_NAMES, arms, strict=True):
        if (
            entry.get("optimizer_step_calls") != 1
            or entry.get("optimizer_state_not_durable") is not True
            or entry.get("nonzero_weight_update_proven") is not False
            or entry.get("resume_allowed") is not False
            or entry.get("scientific_result_available") is not False
            or entry.get("run_id") != f"phase6_20260724_{arm}_primary"
        ):
            raise Phase6RecoveryAuthorityError(
                f"failed-run inventory arm {arm} differs"
            )


def _validate_supporting_bundles(value: Any) -> None:
    expected = _expected_supporting_bundles()
    if value != expected:
        raise Phase6RecoveryAuthorityError("recovery supporting-bundle pins differ")
    payloads: dict[str, bytes] = {}
    for role in sorted(expected):
        _pin_value, payloads[role] = _stable_registered_pin(expected[role], role)
    _validate_failed_inventory(payloads["failed_run_inventory"])

    failed = _strict_json(
        payloads["failed_inventory_bootstrap"], "failed-inventory bootstrap"
    )
    _exact(
        failed,
        {
            "artifact_role",
            "builder_input",
            "candidate_inputs",
            "lifecycle",
            "outputs",
            "registered_at_utc",
            "schema_version",
            "status",
        },
        "failed-inventory bootstrap",
    )
    outputs = _exact(
        failed["outputs"],
        {
            "bootstrap_builder",
            "inventory",
            "inventory_builder",
            "review_subject",
            "reviewer_A",
            "reviewer_B",
        },
        "failed-inventory bootstrap outputs",
    )
    inventory_output = _full_receipt_pin(outputs["inventory"], "bootstrap inventory")
    subject_output = _full_receipt_pin(
        outputs["review_subject"], "failed bootstrap subject"
    )
    review_a_output = _full_receipt_pin(
        outputs["reviewer_A"], "failed bootstrap review A"
    )
    review_b_output = _full_receipt_pin(
        outputs["reviewer_B"], "failed bootstrap review B"
    )
    if (
        failed["schema_version"] != "sana-phase6-failed-inventory-bootstrap-receipt-v1"
        or failed["artifact_role"]
        != "phase6_failed_inventory_bootstrap_publication_receipt"
        or failed["status"] != "registered"
        or failed["lifecycle"]
        != {
            "closed_loop_started": False,
            "durable_optimizer_state_available": False,
            "historical_formal_training_started": True,
            "recovery_cohort_started": False,
        }
        or _registered_projection(inventory_output) != expected["failed_run_inventory"]
        or _registered_projection(subject_output)
        != expected["failed_inventory_review_subject"]
        or _registered_projection(review_a_output)
        != expected["failed_inventory_review_A"]
        or _registered_projection(review_b_output)
        != expected["failed_inventory_review_B"]
    ):
        raise Phase6RecoveryAuthorityError("failed-inventory bootstrap differs")
    _utc(failed["registered_at_utc"], "failed-inventory bootstrap registration")
    failed_subject = _strict_json(
        payloads["failed_inventory_review_subject"], "failed inventory subject"
    )
    _exact(
        failed_subject,
        {
            "artifact_role",
            "builder",
            "deprecated_candidates",
            "evidence_root",
            "historical_claims",
            "inventory",
            "review_requirements",
            "schema_version",
        },
        "failed inventory subject",
    )
    if (
        failed_subject["schema_version"]
        != "sana-phase6-failed-run-inventory-review-subject-v1"
        or failed_subject["artifact_role"]
        != "phase6_failed_run_inventory_review_subject"
    ):
        raise Phase6RecoveryAuthorityError("failed inventory subject differs")
    for slot in ("A", "B"):
        receipt = _strict_json(
            payloads[f"failed_inventory_review_{slot}"],
            f"failed inventory review {slot}",
        )
        _exact(
            receipt,
            {
                "artifact_role",
                "attestations",
                "decision",
                "dynamic_replay",
                "registered_at_utc",
                "review_slot",
                "review_subject",
                "reviewed_candidate_subject",
                "reviewer",
                "schema_version",
                "status",
            },
            f"failed inventory review {slot}",
        )
        receipt_subject = _exact(
            receipt["review_subject"],
            {"path", "sha256", "size_bytes"},
            f"failed inventory review {slot} subject pin",
        )
        if (
            receipt["schema_version"]
            != "sana-phase6-failed-run-inventory-review-receipt-v1"
            or receipt["artifact_role"]
            != "phase6_failed_run_inventory_independent_review_receipt"
            or receipt["review_slot"] != slot
            or receipt["decision"] != "approve"
            or receipt["status"] != "registered"
            or receipt_subject["sha256"]
            != expected["failed_inventory_review_subject"]["sha256"]
        ):
            raise Phase6RecoveryAuthorityError(
                f"failed inventory review {slot} differs"
            )
        _utc(
            receipt["registered_at_utc"], f"failed inventory review {slot} registration"
        )

    rowfix = _strict_json(
        payloads["rowfix_review_bootstrap"], "rowfix review bootstrap"
    )
    _exact(
        rowfix,
        {
            "artifact_role",
            "builder_input",
            "candidate_inputs",
            "failed_run_inventory",
            "outputs",
            "registered_at_utc",
            "rowfix",
            "schema_version",
            "status",
        },
        "rowfix review bootstrap",
    )
    rowfix_outputs = _exact(
        rowfix["outputs"], {"builder", "code", "subject", "test"}, "rowfix outputs"
    )
    inventory_pin = _full_receipt_pin(
        rowfix["failed_run_inventory"], "rowfix inventory"
    )
    rowfix_subject_output = _full_receipt_pin(
        rowfix_outputs["subject"], "rowfix subject output"
    )
    rowfix_code_output = _full_receipt_pin(rowfix_outputs["code"], "rowfix code output")
    rowfix_test_output = _full_receipt_pin(rowfix_outputs["test"], "rowfix test output")
    if (
        rowfix["schema_version"]
        != "sana-phase6-planrow-rowfix-review-bootstrap-receipt-v1"
        or rowfix["artifact_role"] != "phase6_planrow_rowfix_review_bootstrap_receipt"
        or rowfix["status"] != "registered"
        or rowfix["rowfix"]
        != {
            "source_sha256": PUBLISHED_ROWFIX_SOURCE_SHA256,
            "test_sha256": PUBLISHED_ROWFIX_TEST_SHA256,
        }
        or _registered_projection(inventory_pin) != expected["failed_run_inventory"]
        or _registered_projection(rowfix_subject_output)
        != expected["rowfix_review_subject"]
        or _registered_projection(rowfix_code_output) != expected["rowfix_review_code"]
        or _registered_projection(rowfix_test_output) != expected["rowfix_review_test"]
    ):
        raise Phase6RecoveryAuthorityError("rowfix review bootstrap differs")
    _utc(rowfix["registered_at_utc"], "rowfix bootstrap registration")
    rowfix_subject = _strict_json(
        payloads["rowfix_review_subject"], "rowfix review subject"
    )
    _exact(
        rowfix_subject,
        {
            "artifact_role",
            "before",
            "candidate",
            "deprecated_test_candidates",
            "production_dependencies",
            "required_contract",
            "required_tests",
            "schema_version",
        },
        "rowfix review subject",
    )
    if (
        rowfix_subject["schema_version"]
        != "sana-phase6-plan-row-envelope-rowfix-review-subject-v1"
        or rowfix_subject["artifact_role"]
        != "phase6_plan_row_envelope_rowfix_review_subject"
    ):
        raise Phase6RecoveryAuthorityError("rowfix review subject differs")
    for role in ("code", "test"):
        receipt = _strict_json(
            payloads[f"rowfix_review_{role}"], f"rowfix review {role}"
        )
        _exact(
            receipt,
            {
                "artifact_role",
                "attestations",
                "candidate",
                "decision",
                "dynamic_evidence",
                "registered_at_utc",
                "review_role",
                "review_subject",
                "reviewed_candidate_subject",
                "reviewer",
                "schema_version",
                "status",
            },
            f"rowfix review {role}",
        )
        receipt_subject = _exact(
            receipt["review_subject"],
            {"path", "sha256", "size_bytes"},
            f"rowfix review {role} subject pin",
        )
        if (
            receipt["schema_version"]
            != "sana-phase6-plan-row-envelope-rowfix-review-receipt-v1"
            or receipt["artifact_role"]
            != "phase6_plan_row_envelope_rowfix_independent_review_receipt"
            or receipt["review_role"] != role
            or receipt["decision"] != "approve"
            or receipt["status"] != "registered"
            or receipt_subject["sha256"] != expected["rowfix_review_subject"]["sha256"]
        ):
            raise Phase6RecoveryAuthorityError(f"rowfix review {role} differs")
        _utc(receipt["registered_at_utc"], f"rowfix review {role} registration")


def _validate_prior_provenance(value: Any) -> dict[str, dict[str, Any]]:
    expected = _expected_prior_provenance()
    if value != expected:
        raise Phase6RecoveryAuthorityError("prior source-repair provenance differs")
    payloads: dict[str, bytes] = {}
    for role in sorted(expected):
        _pin_value, data = _stable_registered_pin(expected[role], f"prior {role}")
        payloads[role] = data
    prior_source_files = _validate_prior_source_manifest(payloads["source_manifest_v8"])
    _validate_prior_source_execution_receipt(
        payloads["source_execution_receipt_v4"], prior_source_files
    )
    _validate_prior_template_mode_receipt(
        payloads["launch_retry_template_modes_v4"], prior_source_files
    )
    return prior_source_files


def _source_delta(
    before: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    delta: list[dict[str, Any]] = []
    for path in sorted(set(before) | set(candidate)):
        old = deepcopy(before.get(path))
        new = deepcopy(candidate.get(path))
        if old == new:
            continue
        change = "added" if old is None else "deleted" if new is None else "modified"
        delta.append({"before": old, "candidate": new, "change": change, "path": path})
    return delta


def _validate_source_delta(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise Phase6RecoveryAuthorityError("authorized_source_delta must be a list")
    paths: list[str] = []
    for index, entry in enumerate(value):
        label = f"authorized_source_delta[{index}]"
        _exact(entry, {"before", "candidate", "change", "path"}, label)
        path = _text(entry["path"], f"{label}.path")
        paths.append(path)
        before, candidate = entry["before"], entry["candidate"]
        for role, pin in (("before", before), ("candidate", candidate)):
            if pin is not None:
                _exact(pin, {"sha256", "size_bytes"}, f"{label}.{role}")
                _digest(pin["sha256"], f"{label}.{role}.sha256")
                _positive_int(
                    pin["size_bytes"], f"{label}.{role}.size_bytes", allow_zero=True
                )
        expected_change = (
            "added"
            if before is None
            else "deleted"
            if candidate is None
            else "modified"
        )
        if before is None and candidate is None:
            raise Phase6RecoveryAuthorityError(f"{label} has no transition")
        if entry["change"] != expected_change:
            raise Phase6RecoveryAuthorityError(f"{label}.change differs")
    if paths != sorted(set(paths)):
        raise Phase6RecoveryAuthorityError("authorized_source_delta paths differ")
    return value


def _proposal_contract_sha256(proposal: Mapping[str, Any]) -> str:
    keys = (
        "authorized_source_delta",
        "baseline_source_snapshot",
        "lifecycle",
        "prior_provenance",
        "scientific_equivalence_contract",
        "source_inventory",
        "source_render_contract",
        "supporting_bundles",
        "versioned_outputs",
    )
    return sha256(
        canonical_json_bytes({key: proposal[key] for key in keys})
    ).hexdigest()


def build_phase6_recovery_proposal(
    *,
    baseline_source_snapshot_pin: Mapping[str, Any],
    candidate_source_root: str,
    target_repository_root: str = REPOSITORY_ROOT,
    registered_at_utc: str,
) -> dict[str, Any]:
    """Construct the proposal from a registered baseline and tokenized candidate."""
    if target_repository_root != REPOSITORY_ROOT:
        raise Phase6RecoveryAuthorityError("target repository root differs")
    if type(candidate_source_root) is not str or not os.path.isabs(
        candidate_source_root
    ):
        raise Phase6RecoveryAuthorityError("candidate source root must be absolute")
    _utc(registered_at_utc, "recovery proposal registration")
    baseline_pin = _pin(dict(baseline_source_snapshot_pin), "baseline source snapshot")
    if baseline_pin["path"] != RECOVERY_BASELINE_SNAPSHOT_PATH:
        raise Phase6RecoveryAuthorityError("baseline source snapshot path differs")
    _observed, baseline_data = _stable_registered_pin(
        baseline_pin, "baseline source snapshot"
    )
    baseline = validate_phase6_recovery_baseline_snapshot_bytes(baseline_data)
    if _utc(baseline["registered_at_utc"], "baseline registration") > _utc(
        registered_at_utc, "proposal registration"
    ):
        raise Phase6RecoveryAuthorityError("proposal predates source baseline")
    supporting = _expected_supporting_bundles()
    prior = _expected_prior_provenance()
    _validate_supporting_bundles(supporting)
    _validate_prior_provenance(prior)
    files = _discover_source_inventory(candidate_source_root, None)
    before = _file_pin_map(baseline["files"], "recovery source baseline")
    candidate = _file_pin_map(files, "tokenized source inventory")
    proposal = {
        "artifact_path": RECOVERY_PROPOSAL_PATH,
        "artifact_role": "phase6_planrow_recovery_proposal",
        "authorized_source_delta": _source_delta(before, candidate),
        "baseline_source_snapshot": baseline_pin,
        "decision_requested": "authorize_fresh_planrowfix_recovery_cohort",
        "lifecycle": recovery_lifecycle(),
        "prior_provenance": prior,
        "registered_at_utc": registered_at_utc,
        "target_repository_root": target_repository_root,
        "schema_version": RECOVERY_PROPOSAL_SCHEMA_VERSION,
        "scientific_equivalence_contract": _scientific_equivalence_contract(),
        "source_inventory": {
            "candidate_source_root_at_registration": candidate_source_root,
            "files": files,
            "inventory_globs": list(SOURCE_INVENTORY_GLOBS),
            "normalized_authority_source_path": RECOVERY_SOURCE_PATH,
            "schema_version": RECOVERY_SOURCE_INVENTORY_SCHEMA_VERSION,
        },
        "source_render_contract": _source_render_contract(),
        "status": "proposed",
        "supporting_bundles": supporting,
        "versioned_outputs": deepcopy(RECOVERY_VERSIONED_OUTPUTS),
    }
    canonical_json_bytes(proposal)
    return proposal


def _validate_phase6_recovery_proposal_bytes(
    data: bytes,
    *,
    source_root: str,
    authority_sha256: str | None = None,
    require_live_formal_baseline: bool,
) -> dict[str, Any]:
    """Validate the proposal and prove its delta covers every observed transition."""
    if not isinstance(data, bytes):
        raise TypeError("recovery proposal must be bytes")
    if type(require_live_formal_baseline) is not bool:
        raise TypeError("live formal baseline requirement must be bool")
    proposal = _strict_json(data, "recovery proposal")
    _exact(
        proposal,
        {
            "artifact_path",
            "artifact_role",
            "authorized_source_delta",
            "baseline_source_snapshot",
            "decision_requested",
            "lifecycle",
            "prior_provenance",
            "registered_at_utc",
            "target_repository_root",
            "schema_version",
            "scientific_equivalence_contract",
            "source_inventory",
            "source_render_contract",
            "status",
            "supporting_bundles",
            "versioned_outputs",
        },
        "recovery proposal",
    )
    if (
        proposal["schema_version"] != RECOVERY_PROPOSAL_SCHEMA_VERSION
        or proposal["artifact_role"] != "phase6_planrow_recovery_proposal"
        or proposal["artifact_path"] != RECOVERY_PROPOSAL_PATH
        or proposal["decision_requested"]
        != "authorize_fresh_planrowfix_recovery_cohort"
        or proposal["status"] != "proposed"
        or proposal["target_repository_root"] != REPOSITORY_ROOT
    ):
        raise Phase6RecoveryAuthorityError("recovery proposal identity differs")
    proposal_time = _utc(
        proposal["registered_at_utc"], "recovery proposal registration"
    )
    validate_recovery_lifecycle(proposal["lifecycle"], "recovery proposal lifecycle")
    _typed_equal(
        proposal["scientific_equivalence_contract"],
        _scientific_equivalence_contract(),
        "recovery proposal scientific contract",
    )
    _typed_equal(
        proposal["source_render_contract"],
        _source_render_contract(),
        "recovery proposal source-render contract",
    )
    _typed_equal(
        proposal["versioned_outputs"],
        RECOVERY_VERSIONED_OUTPUTS,
        "recovery proposal versioned outputs",
    )
    _validate_supporting_bundles(proposal["supporting_bundles"])
    _validate_prior_provenance(proposal["prior_provenance"])
    baseline_pin = _pin(
        proposal["baseline_source_snapshot"], "baseline source snapshot"
    )
    if baseline_pin["path"] != RECOVERY_BASELINE_SNAPSHOT_PATH:
        raise Phase6RecoveryAuthorityError("baseline source snapshot path differs")
    _baseline_observed, baseline_data = _stable_registered_pin(
        baseline_pin, "baseline source snapshot"
    )
    baseline = validate_phase6_recovery_baseline_snapshot_bytes(baseline_data)
    if _utc(baseline["registered_at_utc"], "baseline registration") > proposal_time:
        raise Phase6RecoveryAuthorityError("baseline snapshot postdates proposal")
    if authority_sha256 is None and require_live_formal_baseline:
        live_baseline = _discover_baseline_inventory(proposal["target_repository_root"])
        if baseline["files"] != live_baseline:
            raise Phase6RecoveryAuthorityError(
                "registered baseline differs from the live pre-render formal repository"
            )
    inventory = _exact(
        proposal["source_inventory"],
        {
            "candidate_source_root_at_registration",
            "files",
            "inventory_globs",
            "normalized_authority_source_path",
            "schema_version",
        },
        "tokenized source inventory",
    )
    if (
        inventory["schema_version"] != RECOVERY_SOURCE_INVENTORY_SCHEMA_VERSION
        or inventory["inventory_globs"] != list(SOURCE_INVENTORY_GLOBS)
        or inventory["normalized_authority_source_path"] != RECOVERY_SOURCE_PATH
    ):
        raise Phase6RecoveryAuthorityError(
            "tokenized source inventory contract differs"
        )
    candidate_registration_root = inventory["candidate_source_root_at_registration"]
    if type(candidate_registration_root) is not str or not os.path.isabs(
        candidate_registration_root
    ):
        raise Phase6RecoveryAuthorityError(
            "candidate source registration root must be absolute"
        )
    expected_source_root = (
        candidate_registration_root
        if authority_sha256 is None
        else proposal["target_repository_root"]
    )
    if source_root != expected_source_root:
        raise Phase6RecoveryAuthorityError("proposal source-root phase binding differs")
    observed_files = _discover_source_inventory(source_root, authority_sha256)
    if inventory["files"] != observed_files:
        raise Phase6RecoveryAuthorityError(
            "reviewed source inventory differs from repository"
        )
    before = _file_pin_map(baseline["files"], "recovery source baseline")
    candidate = _file_pin_map(inventory["files"], "tokenized source inventory")
    _validate_source_delta(proposal["authorized_source_delta"])
    if proposal["authorized_source_delta"] != _source_delta(before, candidate):
        raise Phase6RecoveryAuthorityError("authorized_source_delta is incomplete")
    return deepcopy(proposal)


def validate_phase6_recovery_proposal_bytes(
    data: bytes,
    *,
    source_root: str,
    authority_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a proposal against the live pre-render or rendered formal phase."""
    return _validate_phase6_recovery_proposal_bytes(
        data,
        source_root=source_root,
        authority_sha256=authority_sha256,
        require_live_formal_baseline=True,
    )


def validate_phase6_recovery_proposal_apply_bytes(
    data: bytes, *, candidate_source_root: str
) -> dict[str, Any]:
    """Validate a reviewed candidate while an idempotent source apply is in flight."""
    return _validate_phase6_recovery_proposal_bytes(
        data,
        source_root=candidate_source_root,
        require_live_formal_baseline=False,
    )


def build_phase6_recovery_review_subject(
    *,
    proposal_pin: Mapping[str, Any],
    candidate_source_root: str,
    registered_at_utc: str,
) -> dict[str, Any]:
    """Construct the immutable subject that both reviewers must bind."""
    normalized = _pin(dict(proposal_pin), "recovery proposal")
    if normalized["path"] != RECOVERY_PROPOSAL_PATH:
        raise Phase6RecoveryAuthorityError("recovery proposal path differs")
    _observed, proposal_data = _stable_registered_pin(normalized, "recovery proposal")
    proposal = validate_phase6_recovery_proposal_bytes(
        proposal_data, source_root=candidate_source_root
    )
    subject_time = _utc(registered_at_utc, "review subject registration")
    if subject_time < _utc(proposal["registered_at_utc"], "proposal registration"):
        raise Phase6RecoveryAuthorityError("review subject predates proposal")
    subject = {
        "artifact_path": RECOVERY_REVIEW_SUBJECT_PATH,
        "artifact_role": "phase6_planrow_recovery_review_subject",
        "proposal": normalized,
        "proposal_contract_sha256": _proposal_contract_sha256(proposal),
        "registered_at_utc": registered_at_utc,
        "schema_version": RECOVERY_REVIEW_SUBJECT_SCHEMA_VERSION,
        "status": "registered",
    }
    canonical_json_bytes(subject)
    return subject


def _validate_phase6_recovery_review_subject_bytes(
    data: bytes,
    *,
    source_root: str,
    authority_sha256: str | None = None,
    require_live_formal_baseline: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    subject = _strict_json(data, "recovery review subject")
    _exact(
        subject,
        {
            "artifact_path",
            "artifact_role",
            "proposal",
            "proposal_contract_sha256",
            "registered_at_utc",
            "schema_version",
            "status",
        },
        "recovery review subject",
    )
    if (
        subject["schema_version"] != RECOVERY_REVIEW_SUBJECT_SCHEMA_VERSION
        or subject["artifact_role"] != "phase6_planrow_recovery_review_subject"
        or subject["artifact_path"] != RECOVERY_REVIEW_SUBJECT_PATH
        or subject["status"] != "registered"
    ):
        raise Phase6RecoveryAuthorityError("recovery review subject identity differs")
    subject_time = _utc(subject["registered_at_utc"], "review subject registration")
    proposal_pin = _pin(subject["proposal"], "recovery proposal")
    if proposal_pin["path"] != RECOVERY_PROPOSAL_PATH:
        raise Phase6RecoveryAuthorityError("review subject proposal path differs")
    _observed, proposal_data = _stable_registered_pin(proposal_pin, "recovery proposal")
    proposal = _validate_phase6_recovery_proposal_bytes(
        proposal_data,
        source_root=source_root,
        authority_sha256=authority_sha256,
        require_live_formal_baseline=require_live_formal_baseline,
    )
    if subject["proposal_contract_sha256"] != _proposal_contract_sha256(
        proposal
    ) or subject_time < _utc(proposal["registered_at_utc"], "proposal registration"):
        raise Phase6RecoveryAuthorityError("review subject proposal binding differs")
    return deepcopy(subject), proposal


def validate_phase6_recovery_review_subject_bytes(
    data: bytes,
    *,
    source_root: str,
    authority_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a review subject against the live pre-render or rendered phase."""
    return _validate_phase6_recovery_review_subject_bytes(
        data,
        source_root=source_root,
        authority_sha256=authority_sha256,
        require_live_formal_baseline=True,
    )


def _review_attestations() -> dict[str, bool]:
    return {
        "authorized_source_delta_complete": True,
        "baseline_before_pins_verified": True,
        "fail_closed_schema_verified": True,
        "historical_lifecycle_verified": True,
        "independent_review": True,
        "old_cohort_immutable": True,
        "proposal_and_subject_sha256_verified": True,
        "scientific_equivalence_contract_verified": True,
        "supporting_bundles_verified": True,
        "tokenized_source_render_verified": True,
    }


def build_phase6_recovery_review_receipt(
    *,
    slot: str,
    review_subject_pin: Mapping[str, Any],
    candidate_source_root: str,
    reviewer_identity_claim: Mapping[str, str],
    session_or_task_id: str,
    registered_at_utc: str,
) -> dict[str, Any]:
    """Construct one independent review receipt without referencing the final authority."""
    if slot not in RECOVERY_REVIEW_RECEIPT_PATHS:
        raise Phase6RecoveryAuthorityError("recovery review slot differs")
    subject_pin = _pin(dict(review_subject_pin), "recovery review subject")
    if subject_pin["path"] != RECOVERY_REVIEW_SUBJECT_PATH:
        raise Phase6RecoveryAuthorityError("recovery review subject path differs")
    _observed, subject_data = _stable_registered_pin(
        subject_pin, "recovery review subject"
    )
    subject, proposal = validate_phase6_recovery_review_subject_bytes(
        subject_data, source_root=candidate_source_root
    )
    claim = _exact(
        dict(reviewer_identity_claim),
        {"identity", "identity_namespace"},
        "reviewer identity claim",
    )
    _text(claim["identity"], "reviewer identity")
    _text(claim["identity_namespace"], "reviewer namespace")
    _text(session_or_task_id, "reviewer session/task")
    review_time = _utc(registered_at_utc, "review registration")
    if review_time < _utc(subject["registered_at_utc"], "review subject registration"):
        raise Phase6RecoveryAuthorityError("review predates review subject")
    review = {
        "artifact_path": RECOVERY_REVIEW_RECEIPT_PATHS[slot],
        "artifact_role": "phase6_planrow_recovery_independent_review",
        "attestations": _review_attestations(),
        "decision": "approve",
        "lifecycle": deepcopy(proposal["lifecycle"]),
        "proposal": deepcopy(subject["proposal"]),
        "proposal_contract_sha256": _proposal_contract_sha256(proposal),
        "registered_at_utc": registered_at_utc,
        "review_slot": slot,
        "review_subject": subject_pin,
        "reviewer": {
            "identity_claim": claim,
            "identity_sha256": sha256(canonical_json_bytes(claim)).hexdigest(),
            "session_or_task_id": session_or_task_id,
        },
        "schema_version": RECOVERY_REVIEW_SCHEMA_VERSION,
        "status": "registered",
        "supporting_bundles": deepcopy(proposal["supporting_bundles"]),
    }
    canonical_json_bytes(review)
    return review


def validate_phase6_recovery_review_receipt_bytes(
    data: bytes,
    *,
    slot: str,
    expected_review_subject_pin: Mapping[str, Any],
    review_subject: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str, datetime]:
    review = _strict_json(data, f"recovery review {slot}")
    _exact(
        review,
        {
            "artifact_path",
            "artifact_role",
            "attestations",
            "decision",
            "lifecycle",
            "proposal",
            "proposal_contract_sha256",
            "registered_at_utc",
            "review_slot",
            "review_subject",
            "reviewer",
            "schema_version",
            "status",
            "supporting_bundles",
        },
        f"recovery review {slot}",
    )
    if slot not in RECOVERY_REVIEW_RECEIPT_PATHS:
        raise Phase6RecoveryAuthorityError("recovery review slot differs")
    review_subject_pin = _pin(
        review["review_subject"], f"recovery review {slot} subject"
    )
    review_proposal_pin = _pin(review["proposal"], f"recovery review {slot} proposal")
    if (
        review["schema_version"] != RECOVERY_REVIEW_SCHEMA_VERSION
        or review["artifact_role"] != "phase6_planrow_recovery_independent_review"
        or review["artifact_path"] != RECOVERY_REVIEW_RECEIPT_PATHS[slot]
        or review["review_slot"] != slot
        or review["decision"] != "approve"
        or review["status"] != "registered"
        or review_subject_pin != expected_review_subject_pin
        or review_proposal_pin != review_subject["proposal"]
        or review["proposal_contract_sha256"] != _proposal_contract_sha256(proposal)
    ):
        raise Phase6RecoveryAuthorityError(f"recovery review {slot} authority differs")
    validate_recovery_lifecycle(
        review["lifecycle"], f"recovery review {slot} lifecycle"
    )
    _typed_equal(
        review["supporting_bundles"],
        proposal["supporting_bundles"],
        f"recovery review {slot} supporting bundles",
    )
    _typed_equal(
        review["attestations"],
        _review_attestations(),
        f"recovery review {slot} attestations",
    )
    review_time = _utc(
        review["registered_at_utc"], f"recovery review {slot} registration"
    )
    if review_time < _utc(
        review_subject["registered_at_utc"], "review subject registration"
    ):
        raise Phase6RecoveryAuthorityError(f"recovery review {slot} predates subject")
    reviewer = _exact(
        review["reviewer"],
        {"identity_claim", "identity_sha256", "session_or_task_id"},
        f"reviewer {slot}",
    )
    claim = _exact(
        reviewer["identity_claim"],
        {"identity", "identity_namespace"},
        f"reviewer {slot} claim",
    )
    _text(claim["identity"], f"reviewer {slot} identity")
    _text(claim["identity_namespace"], f"reviewer {slot} namespace")
    identity_sha = _digest(
        reviewer["identity_sha256"], f"reviewer {slot} identity SHA256"
    )
    if not hmac.compare_digest(
        identity_sha, sha256(canonical_json_bytes(claim)).hexdigest()
    ):
        raise Phase6RecoveryAuthorityError(
            f"reviewer {slot} identity claim hash differs"
        )
    session = _text(reviewer["session_or_task_id"], f"reviewer {slot} session/task")
    return deepcopy(review), identity_sha, session, review_time


def _authority_contract_from_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "authorized_source_delta": deepcopy(proposal["authorized_source_delta"]),
        "baseline_source_snapshot": deepcopy(proposal["baseline_source_snapshot"]),
        "lifecycle": deepcopy(proposal["lifecycle"]),
        "prior_provenance": deepcopy(proposal["prior_provenance"]),
        "scientific_equivalence_contract": deepcopy(
            proposal["scientific_equivalence_contract"]
        ),
        "source_render_contract": deepcopy(proposal["source_render_contract"]),
        "supporting_bundles": deepcopy(proposal["supporting_bundles"]),
        "versioned_outputs": deepcopy(proposal["versioned_outputs"]),
    }


def build_phase6_planrow_recovery_authority(
    *,
    review_subject_pin: Mapping[str, Any],
    review_pins: Mapping[str, Mapping[str, Any]],
    candidate_source_root: str,
    registered_at_utc: str,
) -> dict[str, Any]:
    """Construct the final authority from the registered subject and two reviews."""
    subject_pin = _pin(dict(review_subject_pin), "recovery review subject")
    if subject_pin["path"] != RECOVERY_REVIEW_SUBJECT_PATH:
        raise Phase6RecoveryAuthorityError("recovery review subject path differs")
    _subject_observed, subject_data = _stable_registered_pin(
        subject_pin, "recovery review subject"
    )
    subject, proposal = validate_phase6_recovery_review_subject_bytes(
        subject_data, source_root=candidate_source_root
    )
    reviews = _exact(dict(review_pins), {"A", "B"}, "recovery review pins")
    normalized_reviews: dict[str, dict[str, Any]] = {}
    identities: set[str] = set()
    sessions: set[str] = set()
    review_times: list[datetime] = []
    for slot in ("A", "B"):
        pin = _pin(dict(reviews[slot]), f"recovery review {slot}")
        if pin["path"] != RECOVERY_REVIEW_RECEIPT_PATHS[slot]:
            raise Phase6RecoveryAuthorityError(f"recovery review {slot} path differs")
        _observed, review_data = _stable_registered_pin(pin, f"recovery review {slot}")
        _review, identity, session, review_time = (
            validate_phase6_recovery_review_receipt_bytes(
                review_data,
                slot=slot,
                expected_review_subject_pin=subject_pin,
                review_subject=subject,
                proposal=proposal,
            )
        )
        normalized_reviews[slot] = pin
        identities.add(identity)
        sessions.add(session)
        review_times.append(review_time)
    if len(identities) != 2 or len(sessions) != 2:
        raise Phase6RecoveryAuthorityError("recovery reviewers are not independent")
    authority_time = _utc(registered_at_utc, "recovery authority registration")
    if any(review_time > authority_time for review_time in review_times):
        raise Phase6RecoveryAuthorityError("recovery authority predates a review")
    authority = {
        "artifact_path": PLANROW_RECOVERY_AUTHORITY_PATH,
        "artifact_role": "phase6_planrow_recovery_authority",
        **_authority_contract_from_proposal(proposal),
        "decision": "authorize_fresh_planrowfix_recovery_cohort",
        "independent_review_registrations": normalized_reviews,
        "proposal": deepcopy(subject["proposal"]),
        "registered_at_utc": registered_at_utc,
        "review_subject": subject_pin,
        "schema_version": RECOVERY_AUTHORITY_SCHEMA_VERSION,
        "status": "registered",
    }
    canonical_json_bytes(authority)
    return authority


def _validate_phase6_recovery_authority_graph(
    data: bytes,
    *,
    expected_authority_sha256: str,
    source_root: str,
    rendered_authority_sha256: str | None,
    require_live_formal_baseline: bool,
) -> dict[str, Any]:
    expected_sha = _digest(
        expected_authority_sha256, "plan-row recovery authority SHA256"
    )
    if not isinstance(data, bytes) or not hmac.compare_digest(
        sha256(data).hexdigest(), expected_sha
    ):
        raise Phase6RecoveryAuthorityError("plan-row recovery authority SHA256 differs")
    authority = _strict_json(data, "plan-row recovery authority")
    _exact(
        authority,
        {
            "artifact_path",
            "artifact_role",
            "authorized_source_delta",
            "baseline_source_snapshot",
            "decision",
            "independent_review_registrations",
            "lifecycle",
            "prior_provenance",
            "proposal",
            "registered_at_utc",
            "review_subject",
            "schema_version",
            "scientific_equivalence_contract",
            "source_render_contract",
            "status",
            "supporting_bundles",
            "versioned_outputs",
        },
        "plan-row recovery authority",
    )
    if (
        authority["schema_version"] != RECOVERY_AUTHORITY_SCHEMA_VERSION
        or authority["artifact_role"] != "phase6_planrow_recovery_authority"
        or authority["artifact_path"] != PLANROW_RECOVERY_AUTHORITY_PATH
        or authority["decision"] != "authorize_fresh_planrowfix_recovery_cohort"
        or authority["status"] != "registered"
    ):
        raise Phase6RecoveryAuthorityError(
            "plan-row recovery authority identity differs"
        )
    authority_time = _utc(
        authority["registered_at_utc"], "recovery authority registration"
    )
    subject_pin = _pin(authority["review_subject"], "recovery review subject")
    if subject_pin["path"] != RECOVERY_REVIEW_SUBJECT_PATH:
        raise Phase6RecoveryAuthorityError("final review subject path differs")
    _subject_observed, subject_data = _stable_registered_pin(
        subject_pin, "recovery review subject"
    )
    subject, proposal = _validate_phase6_recovery_review_subject_bytes(
        subject_data,
        source_root=source_root,
        authority_sha256=rendered_authority_sha256,
        require_live_formal_baseline=require_live_formal_baseline,
    )
    authority_proposal_pin = _pin(authority["proposal"], "final recovery proposal")
    if authority_proposal_pin != subject["proposal"]:
        raise Phase6RecoveryAuthorityError(
            "final proposal pin differs from review subject"
        )
    for key, expected in _authority_contract_from_proposal(proposal).items():
        _typed_equal(authority[key], expected, f"final authority {key}")
    validate_recovery_lifecycle(authority["lifecycle"], "final authority lifecycle")
    reviews = _exact(
        authority["independent_review_registrations"], {"A", "B"}, "independent reviews"
    )
    identities: set[str] = set()
    sessions: set[str] = set()
    for slot in ("A", "B"):
        review_pin = _pin(reviews[slot], f"independent review {slot}")
        if review_pin["path"] != RECOVERY_REVIEW_RECEIPT_PATHS[slot]:
            raise Phase6RecoveryAuthorityError(
                f"independent review {slot} path differs"
            )
        _review_observed, review_data = _stable_registered_pin(
            review_pin, f"independent review {slot}"
        )
        _review, identity, session, review_time = (
            validate_phase6_recovery_review_receipt_bytes(
                review_data,
                slot=slot,
                expected_review_subject_pin=subject_pin,
                review_subject=subject,
                proposal=proposal,
            )
        )
        if review_time > authority_time:
            raise Phase6RecoveryAuthorityError(
                f"independent review {slot} postdates authority"
            )
        identities.add(identity)
        sessions.add(session)
    if len(identities) != 2 or len(sessions) != 2:
        raise Phase6RecoveryAuthorityError(
            "reviewer identities and sessions/tasks must both be distinct"
        )
    return deepcopy(authority)


def validate_phase6_planrow_recovery_authority_prerender_bytes(
    data: bytes,
    *,
    expected_authority_sha256: str,
    candidate_source_root: str,
) -> dict[str, Any]:
    """Validate the final graph while the reviewed candidate still has its token."""
    return _validate_phase6_recovery_authority_graph(
        data,
        expected_authority_sha256=expected_authority_sha256,
        source_root=candidate_source_root,
        rendered_authority_sha256=None,
        require_live_formal_baseline=True,
    )


def validate_phase6_planrow_recovery_authority_apply_bytes(
    data: bytes,
    *,
    expected_authority_sha256: str,
    candidate_source_root: str,
) -> dict[str, Any]:
    """Validate the immutable graph during a before/final roll-forward apply."""
    return _validate_phase6_recovery_authority_graph(
        data,
        expected_authority_sha256=expected_authority_sha256,
        source_root=candidate_source_root,
        rendered_authority_sha256=None,
        require_live_formal_baseline=False,
    )


def _proposal_recovery_source_pin(
    authority: Mapping[str, Any],
    *,
    candidate_source_root: str,
    apply_phase: bool,
) -> dict[str, Any]:
    proposal_pin = _pin(authority["proposal"], "render-plan proposal")
    _observed, proposal_data = _stable_registered_pin(
        proposal_pin, "render-plan proposal"
    )
    proposal = (
        validate_phase6_recovery_proposal_apply_bytes(
            proposal_data, candidate_source_root=candidate_source_root
        )
        if apply_phase
        else validate_phase6_recovery_proposal_bytes(
            proposal_data, source_root=candidate_source_root
        )
    )
    matches = [
        entry
        for entry in proposal["source_inventory"]["files"]
        if entry["path"] == RECOVERY_SOURCE_PATH
    ]
    if len(matches) != 1:
        raise Phase6RecoveryAuthorityError(
            "proposal recovery source inventory binding differs"
        )
    return deepcopy(matches[0])


def build_phase6_recovery_source_render_plan(
    *,
    authority_pin: Mapping[str, Any],
    candidate_source_root: str,
    target_repository_root: str = REPOSITORY_ROOT,
    registered_at_utc: str,
) -> dict[str, Any]:
    """Build the immutable single-token render plan before source mutation."""
    normalized = _pin(dict(authority_pin), "plan-row recovery authority")
    if normalized["path"] != PLANROW_RECOVERY_AUTHORITY_PATH:
        raise Phase6RecoveryAuthorityError("render-plan authority path differs")
    _observed, authority_data = _stable_registered_pin(
        normalized, "plan-row recovery authority"
    )
    authority = validate_phase6_planrow_recovery_authority_prerender_bytes(
        authority_data,
        expected_authority_sha256=normalized["sha256"],
        candidate_source_root=candidate_source_root,
    )
    registered = _utc(registered_at_utc, "render-plan registration")
    if registered < _utc(
        authority["registered_at_utc"], "recovery authority registration"
    ):
        raise Phase6RecoveryAuthorityError("render plan predates authority")
    if target_repository_root != REPOSITORY_ROOT:
        raise Phase6RecoveryAuthorityError("render target repository root differs")
    source_path = Path(candidate_source_root).joinpath(*RECOVERY_SOURCE_PATH.split("/"))
    tokenized_pin, tokenized = _stable_repository_file(
        source_path, RECOVERY_SOURCE_PATH
    )
    if tokenized_pin != _proposal_recovery_source_pin(
        authority,
        candidate_source_root=candidate_source_root,
        apply_phase=False,
    ):
        raise Phase6RecoveryAuthorityError("render-plan tokenized proposal pin differs")
    rendered = render_phase6_recovery_source(tokenized, normalized["sha256"])
    return {
        "artifact_path": RECOVERY_RENDER_PLAN_PATH,
        "artifact_role": "phase6_recovery_source_render_plan",
        "authority": normalized,
        "candidate_source_root": candidate_source_root,
        "registered_at_utc": registered_at_utc,
        "rendered_source": {
            "path": RECOVERY_SOURCE_PATH,
            "sha256": sha256(rendered).hexdigest(),
            "size_bytes": len(rendered),
        },
        "schema_version": RECOVERY_RENDER_PLAN_SCHEMA_VERSION,
        "source_write_performed": False,
        "status": "ready",
        "target_repository_root": target_repository_root,
        "tokenized_source": tokenized_pin,
    }


def validate_phase6_recovery_source_render_plan_bytes(
    data: bytes,
    *,
    candidate_source_root: str,
    expected_authority_sha256: str,
) -> dict[str, Any]:
    """Validate a render plan without requiring the formal tree to stay baseline."""
    plan = _strict_json(data, "recovery source render plan")
    _exact(
        plan,
        {
            "artifact_path",
            "artifact_role",
            "authority",
            "candidate_source_root",
            "registered_at_utc",
            "rendered_source",
            "schema_version",
            "source_write_performed",
            "status",
            "target_repository_root",
            "tokenized_source",
        },
        "recovery source render plan",
    )
    if (
        plan["schema_version"] != RECOVERY_RENDER_PLAN_SCHEMA_VERSION
        or plan["artifact_role"] != "phase6_recovery_source_render_plan"
        or plan["artifact_path"] != RECOVERY_RENDER_PLAN_PATH
        or plan["candidate_source_root"] != candidate_source_root
        or plan["target_repository_root"] != REPOSITORY_ROOT
        or plan["source_write_performed"] is not False
        or plan["status"] != "ready"
    ):
        raise Phase6RecoveryAuthorityError(
            "recovery source render plan identity differs"
        )
    authority_pin = _pin(plan["authority"], "render-plan authority")
    expected_sha = _digest(
        expected_authority_sha256, "render-plan expected authority SHA256"
    )
    if authority_pin[
        "path"
    ] != PLANROW_RECOVERY_AUTHORITY_PATH or not hmac.compare_digest(
        authority_pin["sha256"], expected_sha
    ):
        raise Phase6RecoveryAuthorityError("render-plan authority binding differs")
    _authority_observed, authority_data = _stable_registered_pin(
        authority_pin, "render-plan authority"
    )
    authority = validate_phase6_planrow_recovery_authority_apply_bytes(
        authority_data,
        expected_authority_sha256=expected_sha,
        candidate_source_root=candidate_source_root,
    )
    registered = _utc(plan["registered_at_utc"], "render-plan registration")
    if registered < _utc(
        authority["registered_at_utc"], "recovery authority registration"
    ):
        raise Phase6RecoveryAuthorityError("render plan predates authority")
    tokenized_pin = _exact(
        plan["tokenized_source"], {"path", "sha256", "size_bytes"}, "tokenized source"
    )
    rendered_pin = _exact(
        plan["rendered_source"], {"path", "sha256", "size_bytes"}, "rendered source"
    )
    if (
        tokenized_pin["path"] != RECOVERY_SOURCE_PATH
        or rendered_pin["path"] != RECOVERY_SOURCE_PATH
    ):
        raise Phase6RecoveryAuthorityError("render-plan source path differs")
    _digest(tokenized_pin["sha256"], "render-plan tokenized SHA256")
    _positive_int(
        tokenized_pin["size_bytes"], "render-plan tokenized size", allow_zero=True
    )
    _digest(rendered_pin["sha256"], "render-plan rendered SHA256")
    _positive_int(
        rendered_pin["size_bytes"], "render-plan rendered size", allow_zero=True
    )
    source_path = Path(candidate_source_root).joinpath(*RECOVERY_SOURCE_PATH.split("/"))
    observed_tokenized, tokenized = _stable_repository_file(
        source_path, RECOVERY_SOURCE_PATH
    )
    if observed_tokenized != _proposal_recovery_source_pin(
        authority,
        candidate_source_root=candidate_source_root,
        apply_phase=True,
    ):
        raise Phase6RecoveryAuthorityError("render-plan tokenized proposal pin differs")
    rendered = render_phase6_recovery_source(tokenized, expected_sha)
    if tokenized_pin != observed_tokenized or rendered_pin != {
        "path": RECOVERY_SOURCE_PATH,
        "sha256": sha256(rendered).hexdigest(),
        "size_bytes": len(rendered),
    }:
        raise Phase6RecoveryAuthorityError("render-plan source pins differ")
    return deepcopy(plan)


def validate_phase6_planrow_recovery_authority_bytes(
    data: bytes,
    *,
    expected_authority_sha256: str,
    target_repository_root: str = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Validate the final graph against its rendered formal source tree."""
    return _validate_phase6_recovery_authority_graph(
        data,
        expected_authority_sha256=expected_authority_sha256,
        source_root=target_repository_root,
        rendered_authority_sha256=expected_authority_sha256,
        require_live_formal_baseline=True,
    )


def validate_phase6_planrow_recovery_authority(
    *, repository_root: str = REPOSITORY_ROOT
) -> dict[str, Any]:
    """Read and validate the registered final authority before downstream work."""
    authority_sha = _digest(
        PLANROW_RECOVERY_AUTHORITY_SHA256, "plan-row recovery authority SHA256"
    )
    _observed, data = _stable_read_fd(
        PLANROW_RECOVERY_AUTHORITY_PATH,
        name="plan-row recovery authority",
        expected_size=None,
        expected_sha256=authority_sha,
        expected_mode=0o400,
        require_single_link=True,
        maximum_bytes=_MAX_REGISTERED_BYTES,
        allow_empty=False,
    )
    return validate_phase6_planrow_recovery_authority_bytes(
        data,
        expected_authority_sha256=authority_sha,
        target_repository_root=repository_root,
    )


__all__ = [
    "ALLOWED_PROJECTION_LEAF_CHANGES",
    "ARM_NAMES",
    "OPERATIONAL_ROOT",
    "PLANROW_RECOVERY_AUTHORITY_PATH",
    "PLANROW_RECOVERY_AUTHORITY_SHA256",
    "RECOVERY_AUTHORITY_SCHEMA_VERSION",
    "RECOVERY_BASELINE_SCHEMA_VERSION",
    "RECOVERY_BASELINE_SNAPSHOT_PATH",
    "RECOVERY_PROPOSAL_PATH",
    "RECOVERY_PROPOSAL_SCHEMA_VERSION",
    "RECOVERY_REVIEW_RECEIPT_PATHS",
    "RECOVERY_REVIEW_SCHEMA_VERSION",
    "RECOVERY_REVIEW_SUBJECT_PATH",
    "RECOVERY_REVIEW_SUBJECT_SCHEMA_VERSION",
    "RECOVERY_RENDER_PLAN_PATH",
    "RECOVERY_RENDER_PLAN_SCHEMA_VERSION",
    "RECOVERY_ROOT",
    "RECOVERY_RUN_DIRECTORIES",
    "RECOVERY_RUN_IDS",
    "RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH",
    "RECOVERY_SOURCE_PATH",
    "RECOVERY_VERSIONED_OUTPUTS",
    "SOURCE_INVENTORY_GLOBS",
    "Phase6RecoveryAuthorityError",
    "build_phase6_planrow_recovery_authority",
    "build_phase6_recovery_baseline_snapshot",
    "build_phase6_recovery_proposal",
    "build_phase6_recovery_review_receipt",
    "build_phase6_recovery_review_subject",
    "build_phase6_recovery_source_render_plan",
    "canonical_json_bytes",
    "recovery_authority_lifecycle",
    "recovery_lifecycle",
    "render_phase6_recovery_source",
    "validate_recovery_lifecycle",
    "validate_phase6_planrow_recovery_authority",
    "validate_phase6_planrow_recovery_authority_apply_bytes",
    "validate_phase6_planrow_recovery_authority_bytes",
    "validate_phase6_planrow_recovery_authority_prerender_bytes",
    "validate_phase6_recovery_baseline_snapshot_bytes",
    "validate_phase6_recovery_proposal_bytes",
    "validate_phase6_recovery_proposal_apply_bytes",
    "validate_phase6_recovery_review_receipt_bytes",
    "validate_phase6_recovery_review_subject_bytes",
    "validate_phase6_recovery_source_render_plan_bytes",
]
