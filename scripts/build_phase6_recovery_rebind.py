#!/usr/bin/env python3
"""CPU-only provenance rebind for the fresh Phase-6 plan-row recovery cohort.

This builder reuses the immutable numerical outputs of the historical reference
and real-2B smoke runs.  It does not import torch, execute a model, create an
optimizer, issue a launch ticket, or create a training run directory.

The source-v6 and recovery-authority digests are command-line inputs on
purpose: embedding either digest here would create a source-manifest hash
cycle. Formal publication is opt-in via ``--publish``: deterministic same-dir
temps use O_CREAT|O_EXCL, and complete files are installed with an atomic
no-replace hard link. Exact supports left by an interrupted receipt-last
publication may be reused, but no existing byte is ever overwritten.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import os
import re
import stat
import sys
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = "/home/zch/workspace/sana-wam"
OPERATIONAL_ROOT = "/DATA/share/sana_phase6_principled_constraints_20260724"
RECOVERY_ROOT = (
    f"{OPERATIONAL_ROOT}/integration_patches_v5/"
    "phase6_plan_row_envelope_recovery_20260725_v5"
)

OLD_REFERENCE_REQUEST = (
    f"{OPERATIONAL_ROOT}/preflight/reference_precompute_request_v5.json"
)
OLD_REFERENCE_REPORT = (
    f"{OPERATIONAL_ROOT}/preflight/reference_precompute_report_v5.json"
)
OLD_REFERENCE = f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_v4.json"
OLD_REFERENCE_BUILD = (
    f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_v4_build_manifest.json"
)
OLD_REFERENCE_SPOT = (
    f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_spot_v4.json"
)
OLD_SMOKE_REQUEST = f"{OPERATIONAL_ROOT}/preflight/smoke_request_v5.json"
OLD_SMOKE_REPORT = f"{OPERATIONAL_ROOT}/preflight/smoke_report_v5.json"
OLD_SMOKE_GATE = f"{OPERATIONAL_ROOT}/smoke/phase6_real_2b_smoke_gate_v3.json"
OLD_SUPPORT_ROOT = (
    f"{OPERATIONAL_ROOT}/integration_patches_v4/"
    "phase6_launch_runtime_symlink_source_repair_20260724_v2"
)
REFERENCE_NUMERIC_PAYLOAD = f"{OLD_SUPPORT_ROOT}/reference_numeric_payload.json"
REFERENCE_SPOT_NUMERIC_PAYLOAD = (
    f"{OLD_SUPPORT_ROOT}/reference_spot_numeric_payload.json"
)
SMOKE_NONBINDING_PAYLOAD = f"{OLD_SUPPORT_ROOT}/smoke_nonbinding_payload.json"

NEW_REFERENCE_REQUEST = (
    f"{OPERATIONAL_ROOT}/preflight/reference_precompute_request_v10.json"
)
NEW_REFERENCE_REPORT = (
    f"{OPERATIONAL_ROOT}/preflight/reference_precompute_report_v10.json"
)
NEW_REFERENCE = f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_v9.json"
NEW_REFERENCE_BUILD = (
    f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_v9_build_manifest.json"
)
NEW_REFERENCE_SPOT = (
    f"{OPERATIONAL_ROOT}/reference/phase1_action_reference_spot_v9.json"
)
NEW_SMOKE_REQUEST = f"{OPERATIONAL_ROOT}/preflight/smoke_request_v10.json"
NEW_SMOKE_REPORT = f"{OPERATIONAL_ROOT}/preflight/smoke_report_v10.json"
NEW_SMOKE_GATE = f"{OPERATIONAL_ROOT}/smoke/phase6_real_2b_smoke_gate_v8.json"
REFERENCE_RECEIPT = f"{RECOVERY_ROOT}/reference_v9_cpu_rebind_receipt.json"
SMOKE_RECEIPT = f"{RECOVERY_ROOT}/smoke_v8_cpu_rebind_receipt.json"
EVIDENCE_REBINDING_RECEIPT = f"{RECOVERY_ROOT}/evidence_rebinding_receipt_v6.json"

OLD_REQUEST_SCHEMA = "sana-phase6-preflight-request-v3"
OLD_REPORT_SCHEMA = "sana-phase6-preflight-report-v3"
NEW_REQUEST_SCHEMA = "sana-phase6-preflight-request-v4"
NEW_REPORT_SCHEMA = "sana-phase6-preflight-report-v4"
REFERENCE_SCHEMA = "sana-phase6-action-reference-v2"
REFERENCE_BUILD_SCHEMA = "sana-phase6-action-reference-build-v2"
REFERENCE_SPOT_SCHEMA = "sana-phase6-action-reference-live-spot-v2"
OLD_SMOKE_SCHEMA = "sana-phase6-real-2b-smoke-artifact-v1"
NEW_SMOKE_SCHEMA = "sana-phase6-real-2b-smoke-artifact-v2"
ARM_NAMES = ("T0_E0A0", "T1_E0A0", "T1_E1A0", "T1_E0A1", "T1_E1A1")
SPOT_STEPS = (1, 253, 504)
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class RebindError(RuntimeError):
    """Raised before publication whenever a provenance invariant differs."""


@dataclass(frozen=True)
class Pin:
    path: str
    sha256: str
    size_bytes: int
    mode: int

    def __post_init__(self) -> None:
        if not (os.path.isabs(self.path) or self.path.startswith("/")):
            raise RebindError(f"pin path must be absolute: {self.path}")
        if type(self.sha256) is not str or _SHA_RE.fullmatch(self.sha256) is None:
            raise RebindError(f"pin SHA256 differs: {self.path}")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise RebindError(f"pin size differs: {self.path}")
        if self.mode not in {0o400, 0o440, 0o444, 0o600, 0o640}:
            raise RebindError(f"pin mode differs: {self.path}")

    def value(self) -> dict[str, Any]:
        return {
            "mode": f"{self.mode:04o}",
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


FORMAL_OLD_PINS = {
    "reference_request": Pin(
        OLD_REFERENCE_REQUEST,
        "7778bd7c6fe5f2de4b89932abac0a22b3bed91c4d412a50683792b86cb039f9c",
        3521,
        0o444,
    ),
    "reference_report": Pin(
        OLD_REFERENCE_REPORT,
        "12373b36f9975f2116fc3c5f11b3fc0355fe58e792b4e2c8b19d5cdae63f5f5f",
        216244,
        0o600,
    ),
    "reference": Pin(
        OLD_REFERENCE,
        "85bf1fae230bb5f44a980b1c7a605bb779b069509b6b6afba4cd0e67135d0145",
        313548,
        0o400,
    ),
    "reference_build": Pin(
        OLD_REFERENCE_BUILD,
        "f7295443a725b03cfe9f01f8f549fa45ea19b3c1fde6ece8b1234ffb72a9a057",
        1770,
        0o400,
    ),
    "reference_spot": Pin(
        OLD_REFERENCE_SPOT,
        "5e840a6a3f1e1f28d71e9344f2f049984f10670235dd4bceda819a8f3069b535",
        3821,
        0o400,
    ),
    "smoke_request": Pin(
        OLD_SMOKE_REQUEST,
        "7ea2530529f0705ba4ab09d4d1246ee1376e3ac371ff2a1659b5c56897dff8c0",
        5009,
        0o444,
    ),
    "smoke_report": Pin(
        OLD_SMOKE_REPORT,
        "73ed05b5caa7168b6cc39271ce249755994b321c644cd5734d225dc74d938c8b",
        220064,
        0o600,
    ),
    "smoke_gate": Pin(
        OLD_SMOKE_GATE,
        "f707e6f84d8273f98391a88de5fc94fae8817a3266f4e0562bbc9a61067e6346",
        18101,
        0o400,
    ),
    "reference_numeric_payload": Pin(
        REFERENCE_NUMERIC_PAYLOAD,
        "50285a7ef6cf283ae474d5a71a3e9d7065f11461ca7a0072ee571665b0435d24",
        117317,
        0o400,
    ),
    "reference_spot_numeric_payload": Pin(
        REFERENCE_SPOT_NUMERIC_PAYLOAD,
        "a282cb418958ce650c443629742e9a4dba9cee3ae1199d379cb2dd0554d78930",
        1281,
        0o400,
    ),
    "smoke_nonbinding_payload": Pin(
        SMOKE_NONBINDING_PAYLOAD,
        "47b84b0481fd944fd043ca299ea92c00216ee13e990c3f52b90a34c4a4a48a25",
        16789,
        0o400,
    ),
}


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
        raise RebindError(f"cannot encode canonical JSON: {exc}") from exc


def strict_json(data: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RebindError(f"duplicate key in {label}: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeError, ValueError, TypeError) as exc:
        raise RebindError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict or canonical_json_bytes(value) != data:
        raise RebindError(f"{label} is not one canonical JSON object")
    return value


def _require_sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise RebindError(f"{label} must be a lowercase SHA256")
    return value


def _require_exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise RebindError(f"{label} keys differ")
    return value


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    signature = (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_size),
        int(value.st_mtime_ns),
    )
    # Opening a file updates the emulated ctime on Windows. Linux formal runs
    # retain ctime in the race signature.
    return signature if os.name == "nt" else (*signature, int(value.st_ctime_ns))


def _mode_matches(observed: int, expected: int) -> bool:
    if os.name != "nt":
        return observed == expected
    # Windows exposes only its read-only attribute through POSIX mode bits.
    return observed == (0o444 if expected & 0o222 == 0 else 0o666)


def _stable_read_with_links(pin: Pin, allowed_nlinks: frozenset[int]) -> bytes:
    """Read a SHA-pinned regular file with an explicit hard-link contract."""

    path_before = os.lstat(pin.path)
    if (
        stat.S_ISLNK(path_before.st_mode)
        or not stat.S_ISREG(path_before.st_mode)
        or not _mode_matches(stat.S_IMODE(path_before.st_mode), pin.mode)
        or path_before.st_nlink not in allowed_nlinks
        or path_before.st_size != pin.size_bytes
    ):
        raise RebindError(f"stable input registration differs: {pin.path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if os.name == "posix" and nofollow == 0:
        raise RebindError("O_NOFOLLOW is required for formal stable reads")
    descriptor = os.open(pin.path, flags | nofollow)
    try:
        fd_before = os.fstat(descriptor)
        if not stat.S_ISREG(fd_before.st_mode) or (
            fd_before.st_dev,
            fd_before.st_ino,
        ) != (path_before.st_dev, path_before.st_ino):
            raise RebindError(f"stable descriptor binding differs: {pin.path}")
        digest = sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        fd_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = os.lstat(pin.path)
    if (
        _stat_signature(path_before) != _stat_signature(fd_before)
        or _stat_signature(fd_before) != _stat_signature(fd_after)
        or _stat_signature(fd_after) != _stat_signature(path_after)
        or not hmac.compare_digest(digest.hexdigest(), pin.sha256)
    ):
        raise RebindError(f"stable input changed or SHA256 differs: {pin.path}")
    return b"".join(chunks)


def stable_read(pin: Pin) -> bytes:
    """Read a SHA-pinned single-link regular file through one stable descriptor."""

    return _stable_read_with_links(pin, frozenset({1}))


def pin_payload(path: str, data: bytes, mode: int) -> Pin:
    return Pin(path, sha256(data).hexdigest(), len(data), mode)


def _temporary_output_path(path: str, data: bytes) -> str:
    absolute = os.path.abspath(path)
    digest = sha256(data).hexdigest()
    return os.path.join(
        os.path.dirname(absolute), f".{os.path.basename(absolute)}.rebind-{digest}.tmp"
    )


def _unlink_exact_path(
    path: str,
    *,
    expected_inode: tuple[int, int],
    final_path: str | None = None,
    final_mode: int | None = None,
) -> None:
    observed = os.lstat(path)
    if (
        not stat.S_ISREG(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != expected_inode
    ):
        raise RebindError(f"refusing to unlink replaced output inode: {path}")
    if os.name == "nt":
        os.chmod(path, 0o600)
    os.unlink(path)
    if (
        final_path is not None
        and final_mode is not None
        and os.path.lexists(final_path)
    ):
        os.chmod(final_path, final_mode)


def _recover_deterministic_temp(path: str, data: bytes, mode: int) -> None:
    """Finish cleanup after a crash between hard-link and temp unlink."""

    absolute = os.path.abspath(path)
    temporary = _temporary_output_path(absolute, data)
    if not os.path.lexists(temporary):
        return
    pin = pin_payload(temporary, data, mode)
    info = os.lstat(temporary)
    if info.st_nlink == 1:
        _stable_read_with_links(pin, frozenset({1}))
        if os.path.lexists(absolute):
            raise RebindError(
                f"deterministic temp and final use different inodes: {absolute}"
            )
        return
    if info.st_nlink != 2 or not os.path.lexists(absolute):
        raise RebindError(f"deterministic output temp link count differs: {temporary}")
    _stable_read_with_links(pin, frozenset({2}))
    final_info = os.lstat(absolute)
    if (final_info.st_dev, final_info.st_ino) != (info.st_dev, info.st_ino):
        raise RebindError(f"deterministic temp/final inode differs: {absolute}")
    _unlink_exact_path(
        temporary,
        expected_inode=(info.st_dev, info.st_ino),
        final_path=absolute,
        final_mode=mode,
    )
    parent = os.path.dirname(absolute)
    if os.name == "posix":
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    stable_read(pin_payload(absolute, data, mode))


def _write_exclusive(path: str, data: bytes, mode: int) -> None:
    """Atomically link one complete support without replacing an existing inode."""

    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute)
    parent_info = os.lstat(parent)
    if (
        stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or os.path.realpath(parent) != parent
    ):
        raise RebindError(f"output parent is not a canonical directory: {parent}")
    temporary = _temporary_output_path(absolute, data)
    _recover_deterministic_temp(absolute, data, mode)
    if os.path.lexists(absolute):
        raise FileExistsError(absolute)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    created_here = False
    linked_here = False
    temp_inode: tuple[int, int] | None = None
    try:
        try:
            descriptor = os.open(temporary, flags, 0o600)
        except FileExistsError:
            temp_info = os.lstat(temporary)
            temp_inode = (temp_info.st_dev, temp_info.st_ino)
            _stable_read_with_links(pin_payload(temporary, data, mode), frozenset({1}))
        else:
            created_here = True
            try:
                descriptor_info = os.fstat(descriptor)
                temp_inode = (descriptor_info.st_dev, descriptor_info.st_ino)
                view = memoryview(data)
                offset = 0
                while offset < len(view):
                    count = os.write(descriptor, view[offset:])
                    if count <= 0:
                        raise RebindError(f"short exclusive write: {absolute}")
                    offset += count
                os.fsync(descriptor)
                os.fchmod(descriptor, mode)
                os.fsync(descriptor)
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or not _mode_matches(stat.S_IMODE(info.st_mode), mode)
                    or info.st_nlink != 1
                    or info.st_size != len(data)
                ):
                    raise RebindError(
                        f"exclusive output descriptor differs: {absolute}"
                    )
            finally:
                os.close(descriptor)
            _stable_read_with_links(pin_payload(temporary, data, mode), frozenset({1}))
        try:
            os.link(temporary, absolute, follow_symlinks=False)
            linked_here = True
        except FileExistsError:
            temp_info = os.lstat(temporary)
            final_info = os.lstat(absolute)
            if (temp_info.st_dev, temp_info.st_ino) == (
                final_info.st_dev,
                final_info.st_ino,
            ):
                _recover_deterministic_temp(absolute, data, mode)
                raise
            if not created_here:
                raise RebindError(
                    f"pre-existing deterministic temp raced with final: {absolute}"
                ) from None
            stable_read(pin_payload(absolute, data, mode))
            raise
        if os.name == "posix":
            directory = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        if temp_inode is None:
            raise RebindError("exclusive temp inode registration is absent")
        _unlink_exact_path(
            temporary,
            expected_inode=temp_inode,
            final_path=absolute,
            final_mode=mode,
        )
        if os.name == "posix":
            directory = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        stable_read(pin_payload(absolute, data, mode))
    except BaseException:
        if linked_here and temp_inode is not None and os.path.lexists(absolute):
            final_info = os.lstat(absolute)
            if (final_info.st_dev, final_info.st_ino) != temp_inode:
                raise RebindError(
                    f"refusing to roll back replaced final inode: {absolute}"
                )
            _unlink_exact_path(absolute, expected_inode=temp_inode)
        if created_here and temp_inode is not None and os.path.lexists(temporary):
            _unlink_exact_path(temporary, expected_inode=temp_inode)
        if os.name == "posix":
            directory = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        raise


def publish_bundle(
    payloads: Mapping[str, bytes],
    *,
    modes: Mapping[str, int],
    commit_path: str,
) -> dict[str, str]:
    """Publish complete supports first and atomically link the receipt last."""

    if not payloads or set(payloads) != set(modes) or commit_path not in payloads:
        raise RebindError("publication payload, mode, or commit set differs")
    order = [path for path in payloads if path != commit_path] + [commit_path]
    if len({os.path.abspath(path) for path in order}) != len(order):
        raise RebindError("publication paths are not distinct")
    expected = {
        path: pin_payload(os.path.abspath(path), payloads[path], modes[path])
        for path in order
    }
    for path in order:
        _recover_deterministic_temp(path, payloads[path], modes[path])
    initially_present = {path: os.path.lexists(path) for path in order}
    if initially_present[commit_path] and not all(initially_present.values()):
        raise RebindError("receipt is visible while a publication support is absent")
    states: dict[str, str] = {}
    for path in order:
        pin = expected[path]
        if os.path.lexists(path):
            observed = stable_read(pin)
            if observed != payloads[path]:
                raise RebindError(f"existing publication support differs: {path}")
            states[path] = "recovered_exact_existing"
        else:
            if path == commit_path:
                for support in order[:-1]:
                    if stable_read(expected[support]) != payloads[support]:
                        raise RebindError(
                            f"support changed before receipt commit: {support}"
                        )
            try:
                _write_exclusive(path, payloads[path], modes[path])
            except FileExistsError:
                if stable_read(pin) != payloads[path]:
                    raise RebindError(f"racing output differs: {path}") from None
                states[path] = "recovered_exact_existing"
            else:
                states[path] = "created_this_invocation"
        if stable_read(pin) != payloads[path]:
            raise RebindError(f"publication readback differs: {path}")
    return states


def require_cpu_environment() -> None:
    expected = {
        "CUDA_VISIBLE_DEVICES": "",
        "GIT_OPTIONAL_LOCKS": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise RebindError(f"{key} must be explicitly {value!r}")
    if "torch" in sys.modules:
        raise RebindError("torch was imported into the CPU-only builder")


def recovery_lifecycle() -> dict[str, Any]:
    return {
        "historical_checkpoint_count": 0,
        "historical_completion_count": 0,
        "historical_durable_metrics_rows_total": 0,
        "historical_durable_optimizer_state": False,
        "historical_formal_training_started": True,
        "historical_nonzero_weight_update_proven": False,
        "historical_old_cohort_completed": False,
        "historical_old_cohort_resumable": False,
        "historical_old_cohort_retryable_in_place": False,
        "historical_optimizer_step_calls_per_arm": 1,
        "historical_optimizer_step_calls_total": 5,
        "historical_scheduled_lr_scale_at_step0": 0.0,
        "historical_scientific_results_available": False,
        "historical_summary_count": 0,
        "recovery_cohort_started": False,
        "recovery_closed_loop_started": False,
    }


def _validate_created_at(value: str) -> str:
    try:
        observed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (TypeError, ValueError) as exc:
        raise RebindError("created-at UTC differs") from exc
    if observed > datetime.now(timezone.utc):
        raise RebindError("created-at UTC is future-dated")
    return value


def validate_rebind_creation_chronology(
    reference_created_at_utc: str, smoke_created_at_utc: str
) -> None:
    """Reject a smoke/evidence commit timestamp earlier than its reference receipt."""

    reference_value = _validate_created_at(reference_created_at_utc)
    smoke_value = _validate_created_at(smoke_created_at_utc)
    reference_created = datetime.strptime(
        reference_value, "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    smoke_created = datetime.strptime(smoke_value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    if reference_created > smoke_created:
        raise RebindError("reference/smoke rebind creation chronology differs")


def _pair_lifecycle(purpose: str, *, historical: bool) -> dict[str, Any]:
    if historical:
        common = recovery_lifecycle()
        if purpose == "reference_precompute":
            common["reference_gpu_started"] = False
        elif purpose == "smoke":
            common.update(
                {
                    "reference_precompute_completed": True,
                    "smoke_gpu_started": False,
                }
            )
    else:
        common = {}
        if purpose == "reference_precompute":
            common.update({"reference_gpu_started": False, "training_started": False})
        elif purpose == "smoke":
            common.update(
                {
                    "closed_loop_started": False,
                    "formal_training_started": False,
                    "reference_precompute_completed": True,
                    "smoke_gpu_started": False,
                }
            )
    if purpose not in {"reference_precompute", "smoke"}:
        raise RebindError(f"unsupported preflight purpose: {purpose}")
    return common


def validate_preflight_pair(
    request: Mapping[str, Any],
    report: Mapping[str, Any],
    request_bytes: bytes,
    report_bytes: bytes,
    *,
    purpose: str,
    historical: bool,
) -> None:
    request_schema = NEW_REQUEST_SCHEMA if historical else OLD_REQUEST_SCHEMA
    report_schema = NEW_REPORT_SCHEMA if historical else OLD_REPORT_SCHEMA
    if (
        request.get("schema_version") != request_schema
        or report.get("schema_version") != report_schema
        or request.get("purpose") != purpose
        or report.get("purpose") != purpose
        or report.get("status") != "pass"
        or report.get("authorization") != request.get("authorization")
        or report.get("request_sha256") != sha256(request_bytes).hexdigest()
        or canonical_json_bytes(dict(request)) != request_bytes
        or canonical_json_bytes(dict(report)) != report_bytes
    ):
        raise RebindError(f"{purpose} preflight request/report pair differs")
    registration = (
        "registered_before_recovery_cohort_started"
        if historical
        else "registered_before_phase6_training_results"
    )
    if request.get(registration) is not True or report.get(registration) is not True:
        raise RebindError(f"{purpose} preflight registration differs")
    for key, expected in _pair_lifecycle(purpose, historical=historical).items():
        if (
            type(request.get(key)) is not type(expected)
            or request.get(key) != expected
            or type(report.get(key)) is not type(expected)
            or report.get(key) != expected
        ):
            raise RebindError(f"{purpose} preflight lifecycle {key} differs")


def validate_recovery_bindings(
    request: Mapping[str, Any], *, source_pin: Pin, authority_pin: Pin
) -> None:
    if request.get("source_manifest") != {
        "path": source_pin.path,
        "sha256": source_pin.sha256,
    }:
        raise RebindError("new preflight source-v6 CLI binding differs")
    artifacts = request.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("recovery_authority") != {
        "path": authority_pin.path,
        "sha256": authority_pin.sha256,
    }:
        raise RebindError("new preflight recovery-authority CLI binding differs")
    for key, expected in recovery_lifecycle().items():
        if type(request.get(key)) is not type(expected) or request.get(key) != expected:
            raise RebindError(f"new preflight recovery lifecycle {key} differs")
    if request.get("registered_before_recovery_cohort_started") is not True:
        raise RebindError("new preflight recovery registration differs")


def leaf_differences(
    before: Any, after: Any, prefix: tuple[str | int, ...] = ()
) -> set[tuple[str | int, ...]]:
    if type(before) is not type(after):
        return {prefix}
    if isinstance(before, Mapping):
        if set(before) != set(after):
            return {prefix}
        result: set[tuple[str | int, ...]] = set()
        for key in before:
            result.update(leaf_differences(before[key], after[key], (*prefix, key)))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return {prefix}
        result = set()
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            result.update(leaf_differences(left, right, (*prefix, index)))
        return result
    return set() if before == after else {prefix}


def project_reference_numeric(value: Mapping[str, Any]) -> bytes:
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 504:
        raise RebindError("reference numeric projection row count differs")
    retained = (
        "common_input_trace_sha256",
        "error_float32_bits",
        "t0_reference_forward_trace_sha256",
    )
    try:
        projected = {
            "precompute_gpu_runtime": deepcopy(value["precompute_gpu_runtime"]),
            "rows": [{key: row[key] for key in retained} for row in rows],
        }
    except (KeyError, TypeError) as exc:
        raise RebindError(f"reference numeric projection differs: {exc}") from exc
    return canonical_json_bytes(projected)


def project_reference_spot_numeric(value: Mapping[str, Any]) -> bytes:
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise RebindError("reference spot projection row count differs")
    retained = (
        "common_input_trace_sha256",
        "expected_error_float32_bits",
        "global_step",
        "observed_error_float32_bits",
        "t0_reference_forward_trace_sha256",
    )
    try:
        projected = {
            "rows": [{key: row[key] for key in retained} for row in rows],
            "spot_gpu_runtime": deepcopy(value["spot_gpu_runtime"]),
        }
    except (KeyError, TypeError) as exc:
        raise RebindError(f"reference spot numeric projection differs: {exc}") from exc
    return canonical_json_bytes(projected)


def project_smoke_legacy_nonbinding(value: Mapping[str, Any]) -> bytes:
    if "bindings" not in value:
        raise RebindError("smoke gate has no bindings")
    return canonical_json_bytes(
        {key: deepcopy(item) for key, item in value.items() if key != "bindings"}
    )


def project_smoke_scientific(value: Mapping[str, Any]) -> bytes:
    """Drop only the versioned wrapper; preserve every historical case byte."""

    projected = deepcopy(dict(value))
    projected.pop("bindings", None)
    for key in (
        "schema_version",
        "formal_training_started",
        "closed_loop_started",
        "registered_before_recovery_cohort_started",
        *recovery_lifecycle(),
    ):
        projected.pop(key, None)
    cases = projected.get("cases")
    if not isinstance(cases, list) or len(cases) != len(ARM_NAMES):
        raise RebindError("smoke scientific projection case count differs")
    return canonical_json_bytes(projected)


def _pin_map(
    roles: Mapping[str, tuple[str, bytes, int]],
) -> dict[str, dict[str, Any]]:
    return {
        role: pin_payload(path, data, mode).value()
        for role, (path, data, mode) in roles.items()
    }


def _new_context_sha(row: Mapping[str, Any], reference: Mapping[str, Any]) -> str:
    context = {
        "action_stats_sha256": reference["action_stats_sha256"],
        "common_input_trace_sha256": row["common_input_trace_sha256"],
        "dataset_contract_artifact_sha256": reference[
            "dataset_contract_artifact_sha256"
        ],
        "dataset_contract_row_sha256": row["dataset_contract_row_sha256"],
        "plan_row_sha256": row["plan_row_sha256"],
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
        "reference_checkpoint_sha256": reference["reference_checkpoint_sha256"],
        "t0_reference_forward_trace_sha256": row["t0_reference_forward_trace_sha256"],
    }
    for key, item in context.items():
        _require_sha(item, f"reference context {key}")
    return sha256(canonical_json_bytes(context)).hexdigest()


def _build_reference_payloads(
    *,
    old_request_bytes: bytes,
    old_report_bytes: bytes,
    old_reference_bytes: bytes,
    old_build_bytes: bytes,
    old_spot_bytes: bytes,
    numeric_payload_bytes: bytes,
    spot_numeric_payload_bytes: bytes,
    new_request: Mapping[str, Any],
    new_report: Mapping[str, Any],
    reference_config: Mapping[str, Any],
    source_pin: Pin,
    authority_pin: Pin,
    created_at_utc: str,
    output_paths: Mapping[str, str],
    old_pins: Mapping[str, Pin],
) -> dict[str, bytes]:
    """Purely reconstruct reference bytes without inspecting the ambient runtime."""

    created = _validate_created_at(created_at_utc)
    old_request = strict_json(old_request_bytes, "old reference request-v5")
    old_report = strict_json(old_report_bytes, "old reference report-v5")
    validate_preflight_pair(
        old_request,
        old_report,
        old_request_bytes,
        old_report_bytes,
        purpose="reference_precompute",
        historical=False,
    )
    new_request_bytes = canonical_json_bytes(dict(new_request))
    new_report_bytes = canonical_json_bytes(dict(new_report))
    validate_preflight_pair(
        new_request,
        new_report,
        new_request_bytes,
        new_report_bytes,
        purpose="reference_precompute",
        historical=True,
    )
    validate_recovery_bindings(
        new_request, source_pin=source_pin, authority_pin=authority_pin
    )

    old_reference = strict_json(old_reference_bytes, "old action reference-v4")
    old_build = strict_json(old_build_bytes, "old reference build-v4")
    old_spot = strict_json(old_spot_bytes, "old reference spot-v4")
    if (
        old_reference.get("schema_version") != REFERENCE_SCHEMA
        or old_build.get("schema_version") != REFERENCE_BUILD_SCHEMA
        or old_spot.get("schema_version") != REFERENCE_SPOT_SCHEMA
    ):
        raise RebindError("old reference artifact schema differs")
    numeric_payload = strict_json(numeric_payload_bytes, "reference numeric payload")
    spot_numeric_payload = strict_json(
        spot_numeric_payload_bytes, "reference spot numeric payload"
    )
    if (
        project_reference_numeric(old_reference) != numeric_payload_bytes
        or project_reference_spot_numeric(old_spot) != spot_numeric_payload_bytes
        or canonical_json_bytes(numeric_payload) != numeric_payload_bytes
        or canonical_json_bytes(spot_numeric_payload) != spot_numeric_payload_bytes
    ):
        raise RebindError("historical reference projection payload differs")

    config_keys = {"dataloader", "model", "training"}
    if type(reference_config) is not dict or set(reference_config) != config_keys:
        raise RebindError("reference precompute config keys differ")
    old_config = old_reference.get("precompute_config")
    if not isinstance(old_config, Mapping) or set(old_config) != {
        "dataloader",
        "execution",
        "model",
        "schema_version",
        "training",
    }:
        raise RebindError("old embedded reference config differs")
    artifact_config = {
        "dataloader": deepcopy(reference_config["dataloader"]),
        "execution": deepcopy(old_config["execution"]),
        "model": deepcopy(reference_config["model"]),
        "schema_version": old_config["schema_version"],
        "training": deepcopy(reference_config["training"]),
    }
    expected_config_changes = {
        ("training", "phase6_code_source_manifest"),
        ("training", "phase6_code_source_manifest_sha256"),
    }
    if leaf_differences(old_config, artifact_config) != expected_config_changes:
        raise RebindError("reference embedded config delta exceeds source-v6 binding")
    config_sha = sha256(canonical_json_bytes(artifact_config)).hexdigest()
    request_sha = sha256(new_request_bytes).hexdigest()
    report_sha = sha256(new_report_bytes).hexdigest()
    config_file_sha = _require_sha(
        new_request["input_config"]["sha256"], "reference config file SHA256"
    )

    reference = deepcopy(old_reference)
    reference.update(
        {
            "precompute_config": artifact_config,
            "precompute_config_file_sha256": config_file_sha,
            "precompute_config_sha256": config_sha,
            "precompute_preflight_report_sha256": report_sha,
            "precompute_preflight_request_sha256": request_sha,
            "precompute_source_manifest_sha256": source_pin.sha256,
        }
    )
    gpu_runtime_sha = sha256(
        canonical_json_bytes(reference["precompute_gpu_runtime"])
    ).hexdigest()
    if reference.get("precompute_gpu_runtime_sha256") != gpu_runtime_sha:
        raise RebindError("historical reference GPU-runtime digest differs")
    rows = reference.get("rows")
    if not isinstance(rows, list) or len(rows) != 504:
        raise RebindError("reference row count differs")
    for row in rows:
        row["reference_forward_context_sha256"] = _new_context_sha(row, reference)
    reference_bytes = canonical_json_bytes(reference)
    reference_sha = sha256(reference_bytes).hexdigest()

    build = deepcopy(old_build)
    build.update(
        {
            "precompute_config_file_sha256": config_file_sha,
            "precompute_config_path": new_request["input_config"]["path"],
            "precompute_config_sha256": config_sha,
            "precompute_preflight_report_sha256": report_sha,
            "precompute_preflight_request_sha256": request_sha,
            "precompute_source_manifest_sha256": source_pin.sha256,
            "reference_artifact_path": output_paths["reference"],
            "reference_artifact_sha256": reference_sha,
        }
    )
    build_bytes = canonical_json_bytes(build)

    spot = deepcopy(old_spot)
    spot.update(
        {
            "precompute_config_file_sha256": config_file_sha,
            "precompute_config_sha256": config_sha,
            "precompute_preflight_report_sha256": report_sha,
            "precompute_preflight_request_sha256": request_sha,
            "precompute_source_manifest_sha256": source_pin.sha256,
            "reference_artifact_sha256": reference_sha,
            "spot_preflight_report_sha256": report_sha,
            "spot_preflight_request_sha256": request_sha,
        }
    )
    spot_rows = spot.get("rows")
    if (
        spot.get("global_steps") != list(SPOT_STEPS)
        or not isinstance(spot_rows, list)
        or len(spot_rows) != len(SPOT_STEPS)
    ):
        raise RebindError("reference spot step contract differs")
    for spot_row, step in zip(spot_rows, SPOT_STEPS, strict=True):
        if spot_row.get("global_step") != step:
            raise RebindError("reference spot row order differs")
        spot_row["reference_forward_context_sha256"] = rows[step - 1][
            "reference_forward_context_sha256"
        ]
    spot_bytes = canonical_json_bytes(spot)

    allowed_reference_changes = {
        ("precompute_config_file_sha256",),
        ("precompute_config_sha256",),
        ("precompute_preflight_report_sha256",),
        ("precompute_preflight_request_sha256",),
        ("precompute_source_manifest_sha256",),
        ("precompute_config", "training", "phase6_code_source_manifest"),
        ("precompute_config", "training", "phase6_code_source_manifest_sha256"),
        *{("rows", index, "reference_forward_context_sha256") for index in range(504)},
    }
    if leaf_differences(old_reference, reference) != allowed_reference_changes:
        raise RebindError("reference-v8 exact provenance change mask differs")
    allowed_build_changes = {
        (key,)
        for key in (
            "precompute_config_file_sha256",
            "precompute_config_path",
            "precompute_config_sha256",
            "precompute_preflight_report_sha256",
            "precompute_preflight_request_sha256",
            "precompute_source_manifest_sha256",
            "reference_artifact_path",
            "reference_artifact_sha256",
        )
    }
    if leaf_differences(old_build, build) != allowed_build_changes:
        raise RebindError("reference-v8 build-manifest change mask differs")
    allowed_spot_changes = {
        *(
            (key,)
            for key in (
                "precompute_config_file_sha256",
                "precompute_config_sha256",
                "precompute_preflight_report_sha256",
                "precompute_preflight_request_sha256",
                "precompute_source_manifest_sha256",
                "reference_artifact_sha256",
                "spot_preflight_report_sha256",
                "spot_preflight_request_sha256",
            )
        ),
        *{("rows", index, "reference_forward_context_sha256") for index in range(3)},
    }
    if leaf_differences(old_spot, spot) != allowed_spot_changes:
        raise RebindError("reference-v8 spot change mask differs")
    new_numeric = project_reference_numeric(reference)
    new_spot_numeric = project_reference_spot_numeric(spot)
    if (
        new_numeric != numeric_payload_bytes
        or new_spot_numeric != spot_numeric_payload_bytes
    ):
        raise RebindError("old/new reference numeric projection is not byte-identical")

    new_roles = {
        "request_v10": (output_paths["request"], new_request_bytes, 0o444),
        "report_v10": (output_paths["report"], new_report_bytes, 0o600),
        "reference_v9": (output_paths["reference"], reference_bytes, 0o400),
        "build_v9": (output_paths["build"], build_bytes, 0o400),
        "spot_v9": (output_paths["spot"], spot_bytes, 0o400),
    }
    receipt = {
        "artifact_role": "phase6_reference_v9_cpu_byte_identical_rebind",
        "authorization": {
            "recovery_authority": authority_pin.value(),
            "source_manifest_v9": source_pin.value(),
        },
        "created_at_utc": created,
        "gpu_execution": {
            "gpu_executed_in_rebind": False,
            "historical_reference_gpu_evidence_reused": True,
            "torch_imported": False,
        },
        **recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "new": _pin_map(new_roles),
        "old": {role: old_pins[role].value() for role in sorted(old_pins)},
        "projections": {
            "reference_504_rows": {
                "byte_identical": True,
                "new_sha256": sha256(new_numeric).hexdigest(),
                "old_sha256": sha256(
                    project_reference_numeric(old_reference)
                ).hexdigest(),
                "payload": old_pins["reference_numeric_payload"].value(),
            },
            "spot_3_rows": {
                "byte_identical": True,
                "new_sha256": sha256(new_spot_numeric).hexdigest(),
                "old_sha256": sha256(
                    project_reference_spot_numeric(old_spot)
                ).hexdigest(),
                "payload": old_pins["reference_spot_numeric_payload"].value(),
            },
        },
        "schema_version": "sana-phase6-reference-v9-cpu-rebind-receipt-v1",
        "status": "pass",
    }
    receipt_bytes = canonical_json_bytes(receipt)
    return {
        output_paths["request"]: new_request_bytes,
        output_paths["report"]: new_report_bytes,
        output_paths["reference"]: reference_bytes,
        output_paths["build"]: build_bytes,
        output_paths["spot"]: spot_bytes,
        output_paths["receipt"]: receipt_bytes,
    }


def build_reference_payloads(**kwargs: Any) -> dict[str, bytes]:
    """Build publishable reference bytes only in the registered CPU environment."""

    require_cpu_environment()
    payloads = _build_reference_payloads(**kwargs)
    require_cpu_environment()
    return payloads


def _new_smoke_bindings(
    old_bindings: Mapping[str, Any],
    new_request: Mapping[str, Any],
    *,
    request_sha: str,
    report_sha: str,
) -> dict[str, str]:
    bindings = deepcopy(dict(old_bindings))
    artifacts = new_request["artifacts"]
    runtime = new_request["runtime_files"]
    changed = {
        "action_stats_sha256": runtime["action_stats"]["sha256"],
        "arm_projection_manifest_sha256": artifacts["arm_projection_manifest"][
            "sha256"
        ],
        "dataset_contract_sha256": artifacts["dataset_contract"]["sha256"],
        "phase1_checkpoint_sha256": runtime["phase1_checkpoint"]["sha256"],
        "preflight_report_sha256": report_sha,
        "preflight_request_sha256": request_sha,
        "reference_artifact_sha256": artifacts["action_reference"]["sha256"],
        "runtime_support_manifest_sha256": artifacts["runtime_support_manifest"][
            "sha256"
        ],
        "smoke_runtime_config_sha256": new_request["input_config"]["sha256"],
        "source_manifest_sha256": new_request["source_manifest"]["sha256"],
        "spot_artifact_sha256": artifacts["action_reference_spot_check"]["sha256"],
        "student_checkpoint_sha256": runtime["student_checkpoint"]["sha256"],
    }
    if not set(changed).issubset(bindings):
        raise RebindError("smoke binding schema differs")
    bindings.update(changed)
    for key, value in bindings.items():
        _require_sha(value, f"new smoke binding {key}")
    return bindings


def _projection_sha_by_arm(manifest: Mapping[str, Any]) -> dict[str, str]:
    arms = manifest.get("arms")
    if not isinstance(arms, list) or [item.get("arm") for item in arms] != list(
        ARM_NAMES
    ):
        raise RebindError("recovery arm-projection manifest order differs")
    result = {}
    for item in arms:
        result[item["arm"]] = _require_sha(
            item.get("projection_sha256"),
            f"recovery projection {item.get('arm')} SHA256",
        )
    return result


def _build_smoke_payloads(
    *,
    old_request_bytes: bytes,
    old_report_bytes: bytes,
    old_gate_bytes: bytes,
    old_nonbinding_payload_bytes: bytes,
    new_request: Mapping[str, Any],
    new_report: Mapping[str, Any],
    projection_manifest: Mapping[str, Any],
    projection_manifest_bytes: bytes,
    source_pin: Pin,
    authority_pin: Pin,
    reference_receipt_pin: Pin,
    created_at_utc: str,
    output_paths: Mapping[str, str],
    old_pins: Mapping[str, Pin],
    production_validator: Callable[[bytes, Mapping[str, str]], None] | None = None,
) -> dict[str, bytes]:
    """Purely reconstruct smoke bytes without inspecting the ambient runtime."""

    created = _validate_created_at(created_at_utc)
    old_request = strict_json(old_request_bytes, "old smoke request-v5")
    old_report = strict_json(old_report_bytes, "old smoke report-v5")
    validate_preflight_pair(
        old_request,
        old_report,
        old_request_bytes,
        old_report_bytes,
        purpose="smoke",
        historical=False,
    )
    new_request_bytes = canonical_json_bytes(dict(new_request))
    new_report_bytes = canonical_json_bytes(dict(new_report))
    validate_preflight_pair(
        new_request,
        new_report,
        new_request_bytes,
        new_report_bytes,
        purpose="smoke",
        historical=True,
    )
    validate_recovery_bindings(
        new_request, source_pin=source_pin, authority_pin=authority_pin
    )
    manifest_sha = sha256(projection_manifest_bytes).hexdigest()
    if (
        canonical_json_bytes(dict(projection_manifest)) != projection_manifest_bytes
        or new_request["artifacts"]["arm_projection_manifest"]["sha256"] != manifest_sha
    ):
        raise RebindError("smoke arm-projection manifest binding differs")
    # Validate the recovery manifest, but never rewrite the historical GPU case pins.
    _projection_sha_by_arm(projection_manifest)

    old_gate = strict_json(old_gate_bytes, "old smoke gate-v3")
    if old_gate.get("schema_version") != OLD_SMOKE_SCHEMA:
        raise RebindError("old smoke gate schema differs")
    legacy_nonbinding = strict_json(
        old_nonbinding_payload_bytes, "old smoke nonbinding payload"
    )
    if (
        canonical_json_bytes(legacy_nonbinding) != old_nonbinding_payload_bytes
        or project_smoke_legacy_nonbinding(old_gate) != old_nonbinding_payload_bytes
    ):
        raise RebindError("historical smoke nonbinding payload differs")

    request_sha = sha256(new_request_bytes).hexdigest()
    report_sha = sha256(new_report_bytes).hexdigest()
    gate = deepcopy(old_gate)
    gate["schema_version"] = NEW_SMOKE_SCHEMA
    gate["bindings"] = _new_smoke_bindings(
        old_gate["bindings"],
        new_request,
        request_sha=request_sha,
        report_sha=report_sha,
    )
    cases = gate.get("cases")
    if not isinstance(cases, list) or [item.get("case_id") for item in cases] != list(
        ARM_NAMES
    ):
        raise RebindError("historical smoke case order differs")
    if canonical_json_bytes(cases) != canonical_json_bytes(old_gate["cases"]):
        raise RebindError("historical smoke case payload changed during rebind")
    if (
        gate.pop("formal_training_started", None) is not False
        or gate.pop("closed_loop_started", None) is not False
    ):
        raise RebindError("historical smoke top-level lifecycle differs")
    gate.update(recovery_lifecycle())
    gate["registered_before_recovery_cohort_started"] = True
    gate_bytes = canonical_json_bytes(gate)
    if project_smoke_scientific(old_gate) != project_smoke_scientific(gate):
        raise RebindError("old/new smoke scientific projection differs")
    if production_validator is not None:
        production_validator(gate_bytes, gate["bindings"])

    new_roles = {
        "request_v10": (output_paths["request"], new_request_bytes, 0o444),
        "report_v10": (output_paths["report"], new_report_bytes, 0o600),
        "gate_v8": (output_paths["gate"], gate_bytes, 0o400),
    }
    scientific_sha = sha256(project_smoke_scientific(gate)).hexdigest()
    smoke_receipt = {
        "artifact_role": "phase6_smoke_v8_cpu_byte_identical_rebind",
        "authorization": {
            "recovery_authority": authority_pin.value(),
            "source_manifest_v9": source_pin.value(),
        },
        "created_at_utc": created,
        "gpu_execution": {
            "gpu_executed_in_rebind": False,
            "historical_real_2b_gpu_evidence_reused": True,
            "torch_imported": False,
        },
        **recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "new": _pin_map(new_roles),
        "old": {role: old_pins[role].value() for role in sorted(old_pins)},
        "projection": {
            "legacy_nonbinding_payload": old_pins["smoke_nonbinding_payload"].value(),
            "operational_leaves_excluded": [
                "bindings",
                "schema_version",
                "historical/recovery lifecycle envelope",
                "registered_before_recovery_cohort_started",
            ],
            "scientific_payload_byte_identical": True,
            "scientific_payload_sha256": scientific_sha,
        },
        "projection_equivalence_receipt": new_request["artifacts"][
            "projection_equivalence_receipt"
        ],
        "reference_rebind_receipt": reference_receipt_pin.value(),
        "schema_version": "sana-phase6-smoke-v8-cpu-rebind-receipt-v1",
        "status": "pass",
    }
    smoke_receipt_bytes = canonical_json_bytes(smoke_receipt)
    smoke_receipt_pin = pin_payload(
        output_paths["smoke_receipt"], smoke_receipt_bytes, 0o400
    )
    evidence = {
        "artifact_role": "phase6_planrow_recovery_historical_gpu_evidence_rebinding",
        "created_at_utc": created,
        "gpu_reexecution_performed": False,
        "historical_gpu_evidence_reused": True,
        **recovery_lifecycle(),
        "registered_before_recovery_cohort_started": True,
        "reference_rebind_receipt": reference_receipt_pin.value(),
        "recovery_authority": authority_pin.value(),
        "schema_version": "sana-phase6-evidence-rebinding-receipt-v6",
        "smoke_rebind_receipt": smoke_receipt_pin.value(),
        "source_manifest_v9": source_pin.value(),
        "status": "pass",
        "torch_imported": False,
        "validation": {
            "reference_504_numeric_projection_byte_identical": True,
            "reference_spot_3_numeric_projection_byte_identical": True,
            "smoke_scientific_projection_byte_identical": True,
        },
        "versioned_outputs": {
            **_pin_map(new_roles),
            "reference_receipt": reference_receipt_pin.value(),
            "smoke_receipt": smoke_receipt_pin.value(),
        },
    }
    if smoke_receipt["created_at_utc"] != evidence["created_at_utc"]:
        raise RebindError("smoke/evidence creation timestamps differ")
    evidence_bytes = canonical_json_bytes(evidence)
    return {
        output_paths["request"]: new_request_bytes,
        output_paths["report"]: new_report_bytes,
        output_paths["gate"]: gate_bytes,
        output_paths["smoke_receipt"]: smoke_receipt_bytes,
        output_paths["evidence_receipt"]: evidence_bytes,
    }


def build_smoke_payloads(**kwargs: Any) -> dict[str, bytes]:
    """Build publishable smoke bytes only in the registered CPU environment."""

    require_cpu_environment()
    payloads = _build_smoke_payloads(**kwargs)
    require_cpu_environment()
    return payloads


def pin_from_value(value: Any, label: str) -> Pin:
    pin = _require_exact(
        value, {"mode", "path", "sha256", "size_bytes"}, f"{label} pin"
    )
    try:
        mode = int(pin["mode"], 8)
    except (TypeError, ValueError) as exc:
        raise RebindError(f"{label} pin mode differs") from exc
    return Pin(pin["path"], pin["sha256"], pin["size_bytes"], mode)


def _validate_receipt_lifecycle(value: Mapping[str, Any], label: str) -> None:
    for key, expected in recovery_lifecycle().items():
        if type(value.get(key)) is not type(expected) or value.get(key) != expected:
            raise RebindError(f"{label} lifecycle {key} differs")
    if value.get("registered_before_recovery_cohort_started") is not True:
        raise RebindError(f"{label} registration differs")


def validate_reference_rebind_receipt(
    receipt_pin: Pin, *, source_pin: Pin, authority_pin: Pin
) -> dict[str, Any]:
    """Replay every formal reference receipt input and numeric projection."""

    receipt_bytes = stable_read(receipt_pin)
    receipt = strict_json(receipt_bytes, "reference rebind receipt")
    stable_read(source_pin)
    stable_read(authority_pin)
    _require_exact(
        receipt,
        {
            "artifact_role",
            "authorization",
            "created_at_utc",
            "gpu_execution",
            *recovery_lifecycle(),
            "registered_before_recovery_cohort_started",
            "new",
            "old",
            "projections",
            "schema_version",
            "status",
        },
        "reference rebind receipt",
    )
    _validate_receipt_lifecycle(receipt, "reference rebind receipt")
    if (
        receipt["schema_version"] != "sana-phase6-reference-v9-cpu-rebind-receipt-v1"
        or receipt["artifact_role"] != "phase6_reference_v9_cpu_byte_identical_rebind"
        or receipt["status"] != "pass"
        or receipt["authorization"]
        != {
            "recovery_authority": authority_pin.value(),
            "source_manifest_v9": source_pin.value(),
        }
        or receipt["gpu_execution"]
        != {
            "gpu_executed_in_rebind": False,
            "historical_reference_gpu_evidence_reused": True,
            "torch_imported": False,
        }
    ):
        raise RebindError("reference rebind receipt authority differs")
    old_roles = (
        "reference",
        "reference_build",
        "reference_numeric_payload",
        "reference_report",
        "reference_request",
        "reference_spot",
        "reference_spot_numeric_payload",
    )
    expected_old = {role: FORMAL_OLD_PINS[role].value() for role in old_roles}
    if receipt["old"] != expected_old:
        raise RebindError("reference rebind historical formal pins differ")
    old_bytes = {role: stable_read(FORMAL_OLD_PINS[role]) for role in old_roles}
    old_request = strict_json(
        old_bytes["reference_request"], "old reference request-v5"
    )
    old_report = strict_json(old_bytes["reference_report"], "old reference report-v5")
    validate_preflight_pair(
        old_request,
        old_report,
        old_bytes["reference_request"],
        old_bytes["reference_report"],
        purpose="reference_precompute",
        historical=False,
    )
    new_paths = {
        "build_v9": NEW_REFERENCE_BUILD,
        "reference_v9": NEW_REFERENCE,
        "report_v10": NEW_REFERENCE_REPORT,
        "request_v10": NEW_REFERENCE_REQUEST,
        "spot_v9": NEW_REFERENCE_SPOT,
    }
    if set(receipt["new"]) != set(new_paths):
        raise RebindError("reference rebind new role set differs")
    new_pins = {
        role: pin_from_value(receipt["new"][role], f"reference new {role}")
        for role in new_paths
    }
    if any(new_pins[role].path != path for role, path in new_paths.items()):
        raise RebindError("reference rebind new paths differ")
    new_bytes = {role: stable_read(pin) for role, pin in new_pins.items()}
    new_request = strict_json(new_bytes["request_v10"], "reference request-v7")
    new_report = strict_json(new_bytes["report_v10"], "reference report-v7")
    validate_preflight_pair(
        new_request,
        new_report,
        new_bytes["request_v10"],
        new_bytes["report_v10"],
        purpose="reference_precompute",
        historical=True,
    )
    validate_recovery_bindings(
        new_request, source_pin=source_pin, authority_pin=authority_pin
    )
    old_reference = strict_json(old_bytes["reference"], "old reference-v4")
    old_spot = strict_json(old_bytes["reference_spot"], "old reference spot-v4")
    new_reference = strict_json(new_bytes["reference_v9"], "new reference-v8")
    new_spot = strict_json(new_bytes["spot_v9"], "new reference spot-v8")
    old_numeric = project_reference_numeric(old_reference)
    old_spot_numeric = project_reference_spot_numeric(old_spot)
    new_numeric = project_reference_numeric(new_reference)
    new_spot_numeric = project_reference_spot_numeric(new_spot)
    if (
        old_numeric != old_bytes["reference_numeric_payload"]
        or old_spot_numeric != old_bytes["reference_spot_numeric_payload"]
        or new_numeric != old_numeric
        or new_spot_numeric != old_spot_numeric
    ):
        raise RebindError("reference receipt numeric projection replay differs")
    expected_projections = {
        "reference_504_rows": {
            "byte_identical": True,
            "new_sha256": sha256(new_numeric).hexdigest(),
            "old_sha256": sha256(old_numeric).hexdigest(),
            "payload": FORMAL_OLD_PINS["reference_numeric_payload"].value(),
        },
        "spot_3_rows": {
            "byte_identical": True,
            "new_sha256": sha256(new_spot_numeric).hexdigest(),
            "old_sha256": sha256(old_spot_numeric).hexdigest(),
            "payload": FORMAL_OLD_PINS["reference_spot_numeric_payload"].value(),
        },
    }
    if receipt["projections"] != expected_projections:
        raise RebindError("reference receipt projection declaration differs")

    # Recreate the complete versioned bundle from the registered historical
    # inputs. Numeric projection equality alone cannot authenticate the
    # provenance-only fields or the build-manifest backlinks.
    old_config = old_reference.get("precompute_config")
    if not isinstance(old_config, Mapping):
        raise RebindError("historical reference embedded config differs")
    replay_training = deepcopy(old_config.get("training"))
    if not isinstance(replay_training, dict):
        raise RebindError("historical reference training config differs")
    replay_training["phase6_code_source_manifest"] = source_pin.path
    replay_training["phase6_code_source_manifest_sha256"] = source_pin.sha256
    replay_config = {
        "dataloader": deepcopy(old_config.get("dataloader")),
        "model": deepcopy(old_config.get("model")),
        "training": replay_training,
    }
    output_paths = {
        "build": NEW_REFERENCE_BUILD,
        "receipt": receipt_pin.path,
        "reference": NEW_REFERENCE,
        "report": NEW_REFERENCE_REPORT,
        "request": NEW_REFERENCE_REQUEST,
        "spot": NEW_REFERENCE_SPOT,
    }
    expected_payloads = _build_reference_payloads(
        old_request_bytes=old_bytes["reference_request"],
        old_report_bytes=old_bytes["reference_report"],
        old_reference_bytes=old_bytes["reference"],
        old_build_bytes=old_bytes["reference_build"],
        old_spot_bytes=old_bytes["reference_spot"],
        numeric_payload_bytes=old_bytes["reference_numeric_payload"],
        spot_numeric_payload_bytes=old_bytes["reference_spot_numeric_payload"],
        new_request=new_request,
        new_report=new_report,
        reference_config=replay_config,
        source_pin=source_pin,
        authority_pin=authority_pin,
        created_at_utc=receipt["created_at_utc"],
        output_paths=output_paths,
        old_pins={role: FORMAL_OLD_PINS[role] for role in old_roles},
    )
    observed_payloads = {
        NEW_REFERENCE_BUILD: new_bytes["build_v9"],
        NEW_REFERENCE: new_bytes["reference_v9"],
        NEW_REFERENCE_REPORT: new_bytes["report_v10"],
        NEW_REFERENCE_REQUEST: new_bytes["request_v10"],
        NEW_REFERENCE_SPOT: new_bytes["spot_v9"],
        receipt_pin.path: receipt_bytes,
    }
    for path, expected_bytes in expected_payloads.items():
        if observed_payloads.get(path) != expected_bytes:
            raise RebindError(
                f"reference receipt deterministic output replay differs: {path}"
            )
    return receipt


def validate_smoke_rebind_receipt(
    receipt_pin: Pin,
    *,
    source_pin: Pin,
    authority_pin: Pin,
    reference_receipt_pin: Pin,
) -> dict[str, Any]:
    """Replay the formal smoke anchor and normalized scientific projection."""

    receipt_bytes = stable_read(receipt_pin)
    receipt = strict_json(receipt_bytes, "smoke rebind receipt")
    stable_read(source_pin)
    stable_read(authority_pin)
    stable_read(reference_receipt_pin)
    _require_exact(
        receipt,
        {
            "artifact_role",
            "authorization",
            "created_at_utc",
            "gpu_execution",
            *recovery_lifecycle(),
            "registered_before_recovery_cohort_started",
            "new",
            "old",
            "projection",
            "projection_equivalence_receipt",
            "reference_rebind_receipt",
            "schema_version",
            "status",
        },
        "smoke rebind receipt",
    )
    _validate_receipt_lifecycle(receipt, "smoke rebind receipt")
    if (
        receipt["schema_version"] != "sana-phase6-smoke-v8-cpu-rebind-receipt-v1"
        or receipt["artifact_role"] != "phase6_smoke_v8_cpu_byte_identical_rebind"
        or receipt["status"] != "pass"
        or receipt["authorization"]
        != {
            "recovery_authority": authority_pin.value(),
            "source_manifest_v9": source_pin.value(),
        }
        or receipt["reference_rebind_receipt"] != reference_receipt_pin.value()
        or receipt["gpu_execution"]
        != {
            "gpu_executed_in_rebind": False,
            "historical_real_2b_gpu_evidence_reused": True,
            "torch_imported": False,
        }
    ):
        raise RebindError("smoke rebind receipt authority differs")
    old_roles = (
        "smoke_gate",
        "smoke_nonbinding_payload",
        "smoke_report",
        "smoke_request",
    )
    if receipt["old"] != {role: FORMAL_OLD_PINS[role].value() for role in old_roles}:
        raise RebindError("smoke rebind historical formal pins differ")
    old_bytes = {role: stable_read(FORMAL_OLD_PINS[role]) for role in old_roles}
    old_request = strict_json(old_bytes["smoke_request"], "old smoke request-v5")
    old_report = strict_json(old_bytes["smoke_report"], "old smoke report-v5")
    validate_preflight_pair(
        old_request,
        old_report,
        old_bytes["smoke_request"],
        old_bytes["smoke_report"],
        purpose="smoke",
        historical=False,
    )
    new_paths = {
        "gate_v8": NEW_SMOKE_GATE,
        "report_v10": NEW_SMOKE_REPORT,
        "request_v10": NEW_SMOKE_REQUEST,
    }
    if set(receipt["new"]) != set(new_paths):
        raise RebindError("smoke rebind new role set differs")
    new_pins = {
        role: pin_from_value(receipt["new"][role], f"smoke new {role}")
        for role in new_paths
    }
    if any(new_pins[role].path != path for role, path in new_paths.items()):
        raise RebindError("smoke rebind new paths differ")
    new_bytes = {role: stable_read(pin) for role, pin in new_pins.items()}
    request = strict_json(new_bytes["request_v10"], "smoke request-v7")
    report = strict_json(new_bytes["report_v10"], "smoke report-v7")
    validate_preflight_pair(
        request,
        report,
        new_bytes["request_v10"],
        new_bytes["report_v10"],
        purpose="smoke",
        historical=True,
    )
    validate_recovery_bindings(
        request, source_pin=source_pin, authority_pin=authority_pin
    )
    old_gate = strict_json(old_bytes["smoke_gate"], "old smoke gate-v3")
    new_gate = strict_json(new_bytes["gate_v8"], "new smoke gate-v7")
    if project_smoke_legacy_nonbinding(old_gate) != old_bytes[
        "smoke_nonbinding_payload"
    ] or project_smoke_scientific(old_gate) != project_smoke_scientific(new_gate):
        raise RebindError("smoke receipt projection replay differs")
    expected_projection = {
        "legacy_nonbinding_payload": FORMAL_OLD_PINS[
            "smoke_nonbinding_payload"
        ].value(),
        "operational_leaves_excluded": [
            "bindings",
            "schema_version",
            "historical/recovery lifecycle envelope",
            "registered_before_recovery_cohort_started",
        ],
        "scientific_payload_byte_identical": True,
        "scientific_payload_sha256": sha256(
            project_smoke_scientific(new_gate)
        ).hexdigest(),
    }
    if (
        receipt["projection"] != expected_projection
        or receipt["projection_equivalence_receipt"]
        != request["artifacts"]["projection_equivalence_receipt"]
    ):
        raise RebindError("smoke receipt projection declaration differs")

    projection_value = request.get("artifacts", {}).get("arm_projection_manifest")
    projection_pin_value = _require_exact(
        projection_value,
        {"path", "sha256"},
        "smoke request arm-projection pin",
    )
    projection_path = projection_pin_value["path"]
    projection_info = os.lstat(projection_path)
    projection_pin = Pin(
        projection_path,
        _require_sha(
            projection_pin_value["sha256"], "smoke request arm-projection SHA256"
        ),
        projection_info.st_size,
        stat.S_IMODE(projection_info.st_mode),
    )
    projection_bytes = stable_read(projection_pin)
    projection_manifest = strict_json(
        projection_bytes, "smoke request arm-projection manifest"
    )
    output_paths = {
        "evidence_receipt": EVIDENCE_REBINDING_RECEIPT,
        "gate": NEW_SMOKE_GATE,
        "report": NEW_SMOKE_REPORT,
        "request": NEW_SMOKE_REQUEST,
        "smoke_receipt": receipt_pin.path,
    }
    expected_payloads = _build_smoke_payloads(
        old_request_bytes=old_bytes["smoke_request"],
        old_report_bytes=old_bytes["smoke_report"],
        old_gate_bytes=old_bytes["smoke_gate"],
        old_nonbinding_payload_bytes=old_bytes["smoke_nonbinding_payload"],
        new_request=request,
        new_report=report,
        projection_manifest=projection_manifest,
        projection_manifest_bytes=projection_bytes,
        source_pin=source_pin,
        authority_pin=authority_pin,
        reference_receipt_pin=reference_receipt_pin,
        created_at_utc=receipt["created_at_utc"],
        output_paths=output_paths,
        old_pins={role: FORMAL_OLD_PINS[role] for role in old_roles},
        production_validator=None,
    )
    observed_payloads = {
        NEW_SMOKE_GATE: new_bytes["gate_v8"],
        NEW_SMOKE_REPORT: new_bytes["report_v10"],
        NEW_SMOKE_REQUEST: new_bytes["request_v10"],
        receipt_pin.path: receipt_bytes,
    }
    for path in (
        NEW_SMOKE_REQUEST,
        NEW_SMOKE_REPORT,
        NEW_SMOKE_GATE,
        receipt_pin.path,
    ):
        if observed_payloads[path] != expected_payloads[path]:
            raise RebindError(
                f"smoke receipt deterministic output replay differs: {path}"
            )
    return receipt


def _read_formal_roles(roles: Sequence[str]) -> dict[str, bytes]:
    return {role: stable_read(FORMAL_OLD_PINS[role]) for role in roles}


def _load_production_modules() -> tuple[Any, Any]:
    try:
        from sana_wam.train import phase6_artifact_dag as dag
        from sana_wam.train import phase6_preflight as preflight
    except ImportError as exc:
        raise RebindError(
            f"cannot import source-v6 production metadata modules: {exc}"
        ) from exc
    if (
        dag.REQUEST_SCHEMA_VERSION != NEW_REQUEST_SCHEMA
        or preflight.REQUEST_SCHEMA_VERSION != NEW_REQUEST_SCHEMA
        or preflight.REPORT_SCHEMA_VERSION != NEW_REPORT_SCHEMA
    ):
        raise RebindError("source-v6 production preflight schema differs")
    if "torch" in sys.modules:
        raise RebindError("source-v6 metadata import pulled in torch")
    return dag, preflight


def _production_reference_inputs(
    source_pin: Pin, authority_pin: Pin
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dag, preflight = _load_production_modules()
    request = dag.build_reference_preflight_request()
    validate_recovery_bindings(
        request, source_pin=source_pin, authority_pin=authority_pin
    )
    report = preflight.validate_phase6_preflight(request)
    config = dag.build_reference_precompute_config()
    return request, report, config


def _production_smoke_inputs(
    source_pin: Pin, authority_pin: Pin
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes, Callable[..., None]]:
    dag, preflight = _load_production_modules()
    request = dag.build_smoke_preflight_request()
    validate_recovery_bindings(
        request, source_pin=source_pin, authority_pin=authority_pin
    )
    report = preflight.validate_phase6_preflight(request)
    projection_path = request["artifacts"]["arm_projection_manifest"]["path"]
    projection_pin = Pin(
        projection_path,
        request["artifacts"]["arm_projection_manifest"]["sha256"],
        os.lstat(projection_path).st_size,
        stat.S_IMODE(os.lstat(projection_path).st_mode),
    )
    projection_bytes = stable_read(projection_pin)
    projection = strict_json(projection_bytes, "recovery projection manifest")

    def validator(data: bytes, bindings: Mapping[str, str]) -> None:
        try:
            from sana_wam.train.phase6_smoke_gate import (
                validate_phase6_smoke_artifact_bytes,
            )
        except ImportError as exc:
            raise RebindError(
                f"cannot import production smoke validator: {exc}"
            ) from exc
        kwargs = {
            "expected_action_stats_sha256": bindings["action_stats_sha256"],
            "expected_arm_projection_manifest_sha256": bindings[
                "arm_projection_manifest_sha256"
            ],
            "expected_artifact_sha256": sha256(data).hexdigest(),
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
            "expected_smoke_runtime_config_sha256": bindings[
                "smoke_runtime_config_sha256"
            ],
            "expected_source_manifest_sha256": bindings["source_manifest_sha256"],
            "expected_spot_artifact_sha256": bindings["spot_artifact_sha256"],
            "expected_student_checkpoint_sha256": bindings["student_checkpoint_sha256"],
        }
        validate_phase6_smoke_artifact_bytes(data, **kwargs)

    return request, report, projection, projection_bytes, validator


def _cli_pin(path: str, digest: str, size: int, mode: int, label: str) -> Pin:
    pin = Pin(os.path.abspath(path), digest, size, mode)
    value = strict_json(stable_read(pin), label)
    if label == "source manifest-v9":
        for key, expected in recovery_lifecycle().items():
            if type(value.get(key)) is not type(expected) or value.get(key) != expected:
                raise RebindError(f"source-v6 recovery lifecycle {key} differs")
        if value.get("registered_before_recovery_cohort_started") is not True:
            raise RebindError("source-v6 recovery registration differs")
    return pin


def _reference_cli(args: argparse.Namespace) -> dict[str, Any]:
    source_pin = _cli_pin(
        args.source_manifest_v9,
        args.source_manifest_v9_sha256,
        args.source_manifest_v9_size,
        0o400,
        "source manifest-v9",
    )
    authority_pin = _cli_pin(
        args.recovery_authority,
        args.recovery_authority_sha256,
        args.recovery_authority_size,
        0o400,
        "recovery authority",
    )
    old = _read_formal_roles(
        (
            "reference_request",
            "reference_report",
            "reference",
            "reference_build",
            "reference_spot",
            "reference_numeric_payload",
            "reference_spot_numeric_payload",
        )
    )
    request, report, config = _production_reference_inputs(source_pin, authority_pin)
    output_paths = {
        "request": NEW_REFERENCE_REQUEST,
        "report": NEW_REFERENCE_REPORT,
        "reference": NEW_REFERENCE,
        "build": NEW_REFERENCE_BUILD,
        "spot": NEW_REFERENCE_SPOT,
        "receipt": REFERENCE_RECEIPT,
    }
    old_pins = {
        key: FORMAL_OLD_PINS[key]
        for key in (
            "reference_request",
            "reference_report",
            "reference",
            "reference_build",
            "reference_spot",
            "reference_numeric_payload",
            "reference_spot_numeric_payload",
        )
    }
    payloads = build_reference_payloads(
        old_request_bytes=old["reference_request"],
        old_report_bytes=old["reference_report"],
        old_reference_bytes=old["reference"],
        old_build_bytes=old["reference_build"],
        old_spot_bytes=old["reference_spot"],
        numeric_payload_bytes=old["reference_numeric_payload"],
        spot_numeric_payload_bytes=old["reference_spot_numeric_payload"],
        new_request=request,
        new_report=report,
        reference_config=config,
        source_pin=source_pin,
        authority_pin=authority_pin,
        created_at_utc=args.created_at_utc,
        output_paths=output_paths,
        old_pins=old_pins,
    )
    digests = {path: sha256(data).hexdigest() for path, data in payloads.items()}
    states = None
    if args.publish:
        modes = {
            output_paths["request"]: 0o444,
            output_paths["report"]: 0o600,
            output_paths["reference"]: 0o400,
            output_paths["build"]: 0o400,
            output_paths["spot"]: 0o400,
            output_paths["receipt"]: 0o400,
        }
        states = publish_bundle(
            payloads, modes=modes, commit_path=output_paths["receipt"]
        )
    return {
        "command": "reference",
        "digests": digests,
        "gpu_executed": False,
        **recovery_lifecycle(),
        "publication": states,
        "status": "pass",
        "torch_imported": False,
    }


def _smoke_cli(args: argparse.Namespace) -> dict[str, Any]:
    source_pin = _cli_pin(
        args.source_manifest_v9,
        args.source_manifest_v9_sha256,
        args.source_manifest_v9_size,
        0o400,
        "source manifest-v9",
    )
    authority_pin = _cli_pin(
        args.recovery_authority,
        args.recovery_authority_sha256,
        args.recovery_authority_size,
        0o400,
        "recovery authority",
    )
    reference_receipt_info = os.lstat(REFERENCE_RECEIPT)
    reference_receipt_pin = Pin(
        REFERENCE_RECEIPT,
        args.reference_receipt_sha256,
        reference_receipt_info.st_size,
        0o400,
    )
    reference_receipt = validate_reference_rebind_receipt(
        reference_receipt_pin,
        source_pin=source_pin,
        authority_pin=authority_pin,
    )
    validate_rebind_creation_chronology(
        reference_receipt["created_at_utc"], args.created_at_utc
    )
    old = _read_formal_roles(
        ("smoke_request", "smoke_report", "smoke_gate", "smoke_nonbinding_payload")
    )
    request, report, projection, projection_bytes, validator = _production_smoke_inputs(
        source_pin, authority_pin
    )
    output_paths = {
        "request": NEW_SMOKE_REQUEST,
        "report": NEW_SMOKE_REPORT,
        "gate": NEW_SMOKE_GATE,
        "smoke_receipt": SMOKE_RECEIPT,
        "evidence_receipt": EVIDENCE_REBINDING_RECEIPT,
    }
    old_pins = {
        key: FORMAL_OLD_PINS[key]
        for key in (
            "smoke_request",
            "smoke_report",
            "smoke_gate",
            "smoke_nonbinding_payload",
        )
    }
    payloads = build_smoke_payloads(
        old_request_bytes=old["smoke_request"],
        old_report_bytes=old["smoke_report"],
        old_gate_bytes=old["smoke_gate"],
        old_nonbinding_payload_bytes=old["smoke_nonbinding_payload"],
        new_request=request,
        new_report=report,
        projection_manifest=projection,
        projection_manifest_bytes=projection_bytes,
        source_pin=source_pin,
        authority_pin=authority_pin,
        reference_receipt_pin=reference_receipt_pin,
        created_at_utc=args.created_at_utc,
        output_paths=output_paths,
        old_pins=old_pins,
        production_validator=validator,
    )
    digests = {path: sha256(data).hexdigest() for path, data in payloads.items()}
    states = None
    if args.publish:
        modes = {
            output_paths["request"]: 0o444,
            output_paths["report"]: 0o600,
            output_paths["gate"]: 0o400,
            output_paths["smoke_receipt"]: 0o400,
            output_paths["evidence_receipt"]: 0o400,
        }
        states = publish_bundle(
            payloads, modes=modes, commit_path=output_paths["evidence_receipt"]
        )
    return {
        "command": "smoke",
        "digests": digests,
        "gpu_executed": False,
        **recovery_lifecycle(),
        "publication": states,
        "status": "pass",
        "torch_imported": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-manifest-v9", required=True)
    common.add_argument("--source-manifest-v9-sha256", required=True)
    common.add_argument("--source-manifest-v9-size", required=True, type=int)
    common.add_argument("--recovery-authority", required=True)
    common.add_argument("--recovery-authority-sha256", required=True)
    common.add_argument("--recovery-authority-size", required=True, type=int)
    common.add_argument("--created-at-utc", required=True)
    common.add_argument(
        "--publish",
        action="store_true",
        help="publish with O_EXCL; absence computes and validates only",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("reference", parents=[common])
    smoke = commands.add_parser("smoke", parents=[common])
    smoke.add_argument("--reference-receipt-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        require_cpu_environment()
        result = (
            _reference_cli(args) if args.command == "reference" else _smoke_cli(args)
        )
    except (OSError, RebindError, TypeError, ValueError) as exc:
        print(
            canonical_json_bytes({"status": "fail", "error": str(exc)}).decode(), end=""
        )
        return 2
    print(canonical_json_bytes(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
