#!/usr/bin/env python3
"""Idempotently apply an authorized Phase-6 recovery source overlay."""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, ExitStack
import ctypes
from copy import deepcopy
from hashlib import sha256
import importlib.util
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wam.train import phase6_recovery as recovery  # noqa: E402


class SourceApplyError(RuntimeError):
    """Raised before an unauthorized or ambiguous source transition."""


SOURCE_EXECUTION_SCHEMA_VERSION = "sana-phase6-recovery-source-execution-receipt-v1"
_REQUIRED_CPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "GIT_OPTIONAL_LOCKS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
}
_ADDED_SOURCE_MODE = 0o644


def _load_publication_module():
    path = Path(__file__).with_name("build_phase6_recovery_governance.py")
    spec = importlib.util.spec_from_file_location(
        "phase6_recovery_governance_publication", path
    )
    if spec is None or spec.loader is None:
        raise SourceApplyError("cannot load governance publication helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publication = _load_publication_module()


def _assert_cpu_only_runtime() -> None:
    if not sys.dont_write_bytecode:
        raise SourceApplyError("source apply requires python -B/dont-write-bytecode")
    for key, expected in _REQUIRED_CPU_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise SourceApplyError(f"source apply environment {key} differs")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise SourceApplyError("source apply forbids torch imports")


def _source_apply_lock_path() -> Path:
    return Path(recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH).with_name(
        ".phase6-source-recovery-apply.lock"
    )


@contextmanager
def _source_apply_lock():
    path = _source_apply_lock_path()
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise SourceApplyError("source apply lock parent differs")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if os.name == "posix" and nofollow == 0:
        raise SourceApplyError("O_NOFOLLOW is required for source apply locking")
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow,
        0o600,
    )
    locked = False
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or (os.name == "posix" and observed.st_uid != os.geteuid())
            or observed.st_size not in {0, 1}
        ):
            raise SourceApplyError("source apply lock identity differs")
        if os.name == "posix":
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
        if observed.st_size == 0:
            os.fchmod(descriptor, 0o600)
            if os.write(descriptor, b"\0") != 1:
                raise OSError("source apply lock initialization made no progress")
            os.fsync(descriptor)
            publication._fsync_parent(path)
        if os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            locked = True
        rebound = os.fstat(descriptor)
        path_stat = os.lstat(path)
        if (
            _identity(rebound) != _identity(observed)
            or _identity(path_stat) != _identity(observed)
            or rebound.st_nlink != 1
            or rebound.st_size != 1
            or not _mode_matches(rebound.st_mode, 0o600)
        ):
            raise SourceApplyError("source apply lock binding differs")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, 1) != b"\0":
            raise SourceApplyError("source apply lock content differs")
        yield
    finally:
        if locked:
            if os.name == "posix":
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            else:
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.close(descriptor)


def _pin(value: Any, name: str, *, allow_none: bool = False) -> dict[str, Any] | None:
    if value is None and allow_none:
        return None
    if type(value) is not dict or set(value) != {"sha256", "size_bytes"}:
        raise SourceApplyError(f"{name} pin keys differ")
    digest = value["sha256"]
    size = value["size_bytes"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(size) is not int
        or size < 0
    ):
        raise SourceApplyError(f"{name} pin differs")
    return {"sha256": digest, "size_bytes": size}


def _file_map(value: Any, name: str) -> dict[str, dict[str, Any]]:
    if type(value) is not list or not value:
        raise SourceApplyError(f"{name} inventory differs")
    result: dict[str, dict[str, Any]] = {}
    paths: list[str] = []
    for index, entry in enumerate(value):
        if type(entry) is not dict or set(entry) != {"path", "sha256", "size_bytes"}:
            raise SourceApplyError(f"{name}[{index}] keys differ")
        relative = _relative(entry["path"], f"{name}[{index}].path")
        if relative in result:
            raise SourceApplyError(f"{name} path duplicates")
        result[relative] = _pin(
            {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]},
            f"{name}[{index}]",
        )
        paths.append(relative)
    if paths != sorted(paths):
        raise SourceApplyError(f"{name} paths are not sorted")
    return result


def _relative(value: Any, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise SourceApplyError(f"{name} is not repository-relative")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise SourceApplyError(f"{name} is not repository-relative")
    return value


def _repository_root(value: str, name: str) -> tuple[Path, Path]:
    if type(value) is not str or not os.path.isabs(value):
        raise SourceApplyError(f"{name} must be absolute")
    root = Path(value)
    observed = os.lstat(root)
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise SourceApplyError(f"{name} must be a non-symlink directory")
    return root, root.resolve(strict=True)


@contextmanager
def _repository_root_fd(root: Path):
    if os.name != "posix":
        yield None
        return
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(root.anchor, flags)
    try:
        for component in root.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        path_stat = os.lstat(root)
        fd_stat = os.fstat(descriptor)
        if _identity(path_stat) != _identity(fd_stat) or not stat.S_ISDIR(
            fd_stat.st_mode
        ):
            raise SourceApplyError("repository root descriptor binding differs")
        yield descriptor
    finally:
        os.close(descriptor)


def _assert_repository_root_binding(root: Path, root_fd: int | None) -> None:
    if root_fd is None:
        return
    try:
        path_stat = os.lstat(root)
        fd_stat = os.fstat(root_fd)
    except OSError as exc:
        raise SourceApplyError("repository root binding is unavailable") from exc
    if (
        stat.S_ISLNK(path_stat.st_mode)
        or not stat.S_ISDIR(path_stat.st_mode)
        or not stat.S_ISDIR(fd_stat.st_mode)
        or _identity(path_stat) != _identity(fd_stat)
    ):
        raise SourceApplyError("repository root path was rebound")


@contextmanager
def _repository_parent_fd(root_fd: int | None, relative: str):
    if root_fd is None:
        yield None
        return
    parts = PurePosixPath(_relative(relative, "source parent path")).parts[:-1]
    descriptor = os.dup(root_fd)
    try:
        for component in parts:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise SourceApplyError("source parent descriptor is not a directory")
        yield descriptor
    finally:
        os.close(descriptor)


def _assert_parent_binding(item: Mapping[str, Any]) -> None:
    root_fd = item["target_root_fd"]
    if root_fd is None:
        return
    _assert_repository_root_binding(item["target_root"], root_fd)
    try:
        with _repository_parent_fd(root_fd, item["path"]) as current_fd:
            current = os.fstat(current_fd)
            bound = os.fstat(item["parent_fd"])
    except OSError as exc:
        raise SourceApplyError(
            f"source parent binding is unavailable: {item['path']}"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or not stat.S_ISDIR(bound.st_mode)
        or _identity(current) != _identity(bound)
    ):
        raise SourceApplyError(f"source parent path was rebound: {item['path']}")


def _assert_plan_bindings(plan: list[dict[str, Any]]) -> None:
    for item in plan:
        _assert_parent_binding(item)


def _stat_at(path: Path, dir_fd: int | None) -> os.stat_result:
    if dir_fd is None:
        return os.lstat(path)
    return os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False)


def _open_at(path: Path, flags: int, mode: int, dir_fd: int | None) -> int:
    if dir_fd is None:
        return os.open(path, flags, mode)
    return os.open(path.name, flags, mode, dir_fd=dir_fd)


def _unlink_at(path: Path, dir_fd: int | None) -> None:
    if dir_fd is None:
        os.unlink(path)
    else:
        os.unlink(path.name, dir_fd=dir_fd)


def _link_at(source: Path, destination: Path, dir_fd: int | None) -> None:
    if dir_fd is None:
        os.link(source, destination, follow_symlinks=False)
    else:
        os.link(
            source.name,
            destination.name,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
            follow_symlinks=False,
        )


def _fsync_parent_at(path: Path, dir_fd: int | None) -> None:
    if dir_fd is None:
        publication._fsync_parent(path)
    else:
        os.fsync(dir_fd)


def _inside(root_real: Path, value: Path, name: str) -> None:
    try:
        common = os.path.commonpath((os.fspath(root_real), os.fspath(value)))
    except ValueError as exc:
        raise SourceApplyError(f"{name} escapes repository") from exc
    if common != os.fspath(root_real):
        raise SourceApplyError(f"{name} escapes repository")


def _repository_path(
    root: Path,
    root_real: Path,
    relative: str,
    *,
    require_leaf: bool,
) -> Path:
    relative = _relative(relative, "source path")
    path = root.joinpath(*relative.split("/"))
    try:
        parent_real = path.parent.resolve(strict=True)
    except OSError as exc:
        raise SourceApplyError(f"source parent is unavailable: {relative}") from exc
    _inside(root_real, parent_real, f"source parent {relative}")
    parent_stat = os.lstat(parent_real)
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise SourceApplyError(f"source parent is not a directory: {relative}")
    bound_path = parent_real / path.name
    try:
        leaf_stat = os.lstat(bound_path)
    except FileNotFoundError:
        if require_leaf:
            raise SourceApplyError(f"required source is absent: {relative}")
        return bound_path
    if stat.S_ISLNK(leaf_stat.st_mode) or not stat.S_ISREG(leaf_stat.st_mode):
        raise SourceApplyError(f"source is not a regular file: {relative}")
    resolved = bound_path.resolve(strict=True)
    _inside(root_real, resolved, f"source {relative}")
    return bound_path


def _mode_matches(observed: int, expected: int) -> bool:
    actual = stat.S_IMODE(observed)
    if os.name == "nt":
        wanted = 0o666 if expected & 0o222 else 0o444
        return actual == wanted
    return actual == expected


def _stable_source(
    path: Path,
    relative: str,
    expected: Mapping[str, Any],
    *,
    require_single_link: bool,
    expected_mode: int | None = None,
    dir_fd: int | None = None,
) -> tuple[bytes, os.stat_result]:
    pin = _pin(dict(expected), f"source {relative}")
    if dir_fd is not None:
        before = _stat_at(path, dir_fd)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (require_single_link and before.st_nlink != 1)
            or before.st_size != pin["size_bytes"]
            or (
                expected_mode is not None
                and not _mode_matches(before.st_mode, expected_mode)
            )
        ):
            raise SourceApplyError(
                f"source descriptor precondition differs: {relative}"
            )
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=dir_fd,
        )
        try:
            fd_before = os.fstat(descriptor)
            if (
                _identity(fd_before) != _identity(before)
                or (require_single_link and fd_before.st_nlink != 1)
                or fd_before.st_size != pin["size_bytes"]
            ):
                raise SourceApplyError(f"source descriptor binding differs: {relative}")
            chunks: list[bytes] = []
            remaining = fd_before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise SourceApplyError(f"source ended early: {relative}")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise SourceApplyError(f"source grew while reading: {relative}")
            fd_after = os.fstat(descriptor)
            after = _stat_at(path, dir_fd)
        finally:
            os.close(descriptor)
        if not (
            recovery._stat_signature(before)
            == recovery._stat_signature(fd_before)
            == recovery._stat_signature(fd_after)
            == recovery._stat_signature(after)
        ):
            raise SourceApplyError(f"source changed while reading: {relative}")
        data = b"".join(chunks)
        if sha256(data).hexdigest() != pin["sha256"]:
            raise recovery.Phase6RecoveryAuthorityError(
                f"source apply {relative} SHA256 differs"
            )
        return data, after
    _observed, data = recovery._stable_read_fd(
        os.fspath(path),
        name=f"source apply {relative}",
        expected_size=pin["size_bytes"],
        expected_sha256=pin["sha256"],
        expected_mode=None,
        require_single_link=require_single_link,
        maximum_bytes=recovery._MAX_SOURCE_BYTES,
        allow_empty=True,
    )
    observed = os.lstat(path)
    if expected_mode is not None and not _mode_matches(observed.st_mode, expected_mode):
        raise SourceApplyError(f"source mode differs: {relative}")
    return data, observed


def _source_pin(data: bytes) -> dict[str, Any]:
    return {"sha256": sha256(data).hexdigest(), "size_bytes": len(data)}


def _stage_path(
    destination: Path,
    relative: str,
    rendered: Mapping[str, Any],
    authority_sha256: str,
) -> Path:
    tag = sha256(relative.encode("utf-8")).hexdigest()[:16]
    return destination.parent / (
        f".phase6-source-{authority_sha256[:16]}-{rendered['sha256']}-{tag}.tmp"
    )


def _witness_path(stage: Path) -> Path:
    return stage.with_name(f"{stage.name}.exchange-witness")


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _require_private_regular(
    observed: os.stat_result,
    name: str,
    *,
    link_counts: set[int],
) -> None:
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink not in link_counts
        or (os.name == "posix" and observed.st_uid != os.geteuid())
    ):
        raise SourceApplyError(f"{name} is not private and regularly linked")


def _unlink_bound(
    path: Path,
    identity: tuple[int, int],
    *,
    remaining_path: Path | None = None,
    remaining_mode: int | None = None,
    dir_fd: int | None = None,
) -> bool:
    try:
        observed = _stat_at(path, dir_fd)
    except FileNotFoundError:
        return False
    if (
        _identity(observed) != identity
        or stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
    ):
        raise SourceApplyError("staged source was replaced before unlink")
    original_mode = stat.S_IMODE(observed.st_mode)
    if os.name == "nt":
        os.chmod(path, 0o600)
    try:
        _unlink_at(path, dir_fd)
    except BaseException:
        if os.name == "nt":
            try:
                rebound = _stat_at(path, dir_fd)
                if _identity(rebound) == identity:
                    os.chmod(path, original_mode)
            except FileNotFoundError:
                pass
        raise
    if remaining_path is not None:
        remaining = _stat_at(remaining_path, dir_fd)
        if _identity(remaining) != identity:
            raise SourceApplyError("remaining source hardlink was replaced")
        if os.name == "nt" and remaining_mode is not None:
            os.chmod(remaining_path, remaining_mode)
    return True


def _fsync_staged(
    path: Path, identity: tuple[int, int], mode: int, dir_fd: int | None
) -> None:
    if os.name == "nt":
        os.chmod(path, 0o600)
    flags = (
        (os.O_RDWR if os.name == "nt" else os.O_RDONLY)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = _open_at(path, flags, 0o600, dir_fd)
        observed = os.fstat(descriptor)
        if _identity(observed) != identity or observed.st_nlink != 1:
            raise SourceApplyError("staged source descriptor binding differs")
        os.fsync(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.name == "nt":
            rebound = _stat_at(path, dir_fd)
            if _identity(rebound) != identity:
                raise SourceApplyError("staged source path was replaced")
            os.chmod(path, mode)


def _prepare_stage(item: dict[str, Any]) -> None:
    _assert_parent_binding(item)
    stage = item["stage_path"]
    data = item["rendered_data"]
    rendered = item["rendered"]
    mode = item["mode"]
    parent_fd = item["parent_fd"]
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _attempt in range(2):
        try:
            descriptor = _open_at(stage, flags, mode, parent_fd)
        except FileExistsError:
            before = _stat_at(stage, parent_fd)
            _require_private_regular(before, "staged source", link_counts={1})
            try:
                _stable_source(
                    stage,
                    item["path"],
                    rendered,
                    require_single_link=True,
                    expected_mode=mode,
                    dir_fd=parent_fd,
                )
            except (SourceApplyError, recovery.Phase6RecoveryAuthorityError):
                after = _stat_at(stage, parent_fd)
                if _signature(before) != _signature(after):
                    raise SourceApplyError("staged source changed during recovery")
                raise SourceApplyError(
                    f"unwitnessed staged source differs: {item['path']}"
                )
            _fsync_staged(stage, _identity(before), mode, parent_fd)
            _fsync_parent_at(stage, parent_fd)
            _assert_parent_binding(item)
            return
        created_identity: tuple[int, int] | None = None
        try:
            created = os.fstat(descriptor)
            created_identity = _identity(created)
            if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
                raise SourceApplyError("new staged source identity differs")
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("staged source write made no progress")
                remaining = remaining[written:]
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            if created_identity is not None and _unlink_bound(
                stage, created_identity, dir_fd=parent_fd
            ):
                _fsync_parent_at(stage, parent_fd)
            raise
        os.close(descriptor)
        try:
            observed = _stat_at(stage, parent_fd)
            if _identity(observed) != created_identity:
                raise SourceApplyError("staged source was replaced after writing")
            _stable_source(
                stage,
                item["path"],
                rendered,
                require_single_link=True,
                expected_mode=mode,
                dir_fd=parent_fd,
            )
            _fsync_parent_at(stage, parent_fd)
            _assert_parent_binding(item)
        except BaseException:
            if created_identity is not None and _unlink_bound(
                stage, created_identity, dir_fd=parent_fd
            ):
                _fsync_parent_at(stage, parent_fd)
            raise
        return
    raise SourceApplyError("cannot recover deterministic source stage")


def _matches_source(
    item: Mapping[str, Any],
    path: Path,
    pin: Mapping[str, Any],
    mode: int,
    *,
    require_single_link: bool,
) -> bool:
    try:
        _stable_source(
            path,
            item["path"],
            pin,
            require_single_link=require_single_link,
            expected_mode=mode,
            dir_fd=item["parent_fd"],
        )
    except (SourceApplyError, recovery.Phase6RecoveryAuthorityError):
        return False
    return True


def _modified_witness_state(
    item: Mapping[str, Any], observed: os.stat_result
) -> str | None:
    if item["change"] != "modified":
        return None
    parent_fd = item["parent_fd"]
    try:
        witness = _stat_at(item["witness_path"], parent_fd)
    except FileNotFoundError:
        return None
    _require_private_regular(witness, "modified exchange witness", link_counts={2})
    try:
        stage = _stat_at(item["stage_path"], parent_fd)
    except FileNotFoundError:
        stage = None
    if _identity(witness) == _identity(observed):
        if observed.st_nlink != 2:
            raise SourceApplyError("modified exchange target witness differs")
        _stable_source(
            item["destination"],
            item["path"],
            item["rendered"],
            require_single_link=False,
            expected_mode=item["mode"],
            dir_fd=parent_fd,
        )
        return "exchange_window"
    if stage is not None and _identity(witness) == _identity(stage):
        _require_private_regular(stage, "modified witnessed stage", link_counts={2})
        _stable_source(
            item["stage_path"],
            item["path"],
            item["rendered"],
            require_single_link=False,
            expected_mode=item["mode"],
            dir_fd=parent_fd,
        )
        return "witness_window"
    raise SourceApplyError("modified exchange witness binding differs")


def _unlink_witness(item: Mapping[str, Any], remaining: Path) -> None:
    parent_fd = item["parent_fd"]
    witness = _stat_at(item["witness_path"], parent_fd)
    _unlink_bound(
        item["witness_path"],
        _identity(witness),
        remaining_path=remaining,
        remaining_mode=item["mode"],
        dir_fd=parent_fd,
    )
    _fsync_parent_at(remaining, parent_fd)


def _recover_modified_witness_window(item: dict[str, Any], state: str) -> str:
    parent_fd = item["parent_fd"]
    _assert_parent_binding(item)
    if state == "witness_window":
        if _matches_source(
            item,
            item["destination"],
            item["before"],
            item["before_mode"],
            require_single_link=True,
        ):
            _unlink_witness(item, item["stage_path"])
            _assert_parent_binding(item)
            return "before"
        if _matches_source(
            item,
            item["destination"],
            item["rendered"],
            item["mode"],
            require_single_link=True,
        ):
            raise SourceApplyError("modified witness state is ambiguous")
        stage = _stat_at(item["stage_path"], parent_fd)
        _unlink_witness(item, item["stage_path"])
        _unlink_bound(item["stage_path"], _identity(stage), dir_fd=parent_fd)
        _fsync_parent_at(item["stage_path"], parent_fd)
        _assert_parent_binding(item)
        raise SourceApplyError(f"displaced target drift was restored: {item['path']}")

    destination = _stat_at(item["destination"], parent_fd)
    witness = _stat_at(item["witness_path"], parent_fd)
    rendered_identity = _identity(destination)
    if _identity(witness) != rendered_identity:
        raise SourceApplyError("modified exchange witness changed during recovery")
    try:
        stage = _stat_at(item["stage_path"], parent_fd)
    except FileNotFoundError:
        _unlink_witness(item, item["destination"])
        _assert_parent_binding(item)
        return "rendered_final"
    _require_private_regular(stage, "modified displaced stage", link_counts={1})
    if _matches_source(
        item,
        item["stage_path"],
        item["before"],
        item["before_mode"],
        require_single_link=True,
    ):
        _unlink_bound(item["stage_path"], _identity(stage), dir_fd=parent_fd)
        _fsync_parent_at(item["stage_path"], parent_fd)
        _unlink_witness(item, item["destination"])
        _assert_parent_binding(item)
        return "rendered_final"
    if _matches_source(
        item,
        item["stage_path"],
        item["rendered"],
        item["mode"],
        require_single_link=True,
    ):
        raise SourceApplyError("modified displaced stage is ambiguous")

    displaced_identity = _identity(stage)
    _rename_exchange(item["stage_path"], item["destination"], parent_fd)
    _fsync_parent_at(item["destination"], parent_fd)
    restored = _stat_at(item["destination"], parent_fd)
    rendered_stage = _stat_at(item["stage_path"], parent_fd)
    rebound_witness = _stat_at(item["witness_path"], parent_fd)
    if (
        _identity(restored) != displaced_identity
        or _identity(rendered_stage) != rendered_identity
        or _identity(rebound_witness) != rendered_identity
        or rendered_stage.st_nlink != 2
        or rebound_witness.st_nlink != 2
    ):
        raise SourceApplyError("modified drift restoration binding differs")
    _unlink_witness(item, item["stage_path"])
    rendered_stage = _stat_at(item["stage_path"], parent_fd)
    _unlink_bound(item["stage_path"], _identity(rendered_stage), dir_fd=parent_fd)
    _fsync_parent_at(item["stage_path"], parent_fd)
    _assert_parent_binding(item)
    raise SourceApplyError(f"displaced target drift was restored: {item['path']}")


def _classify_target(item: dict[str, Any], *, recover_link_window: bool) -> str:
    relative = item["path"]
    destination = item["destination"]
    parent_fd = item["parent_fd"]
    try:
        observed = _stat_at(destination, parent_fd)
    except FileNotFoundError:
        if item["change"] == "added":
            return "absent"
        raise SourceApplyError(f"modified target is absent: {relative}")
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise SourceApplyError(f"target is not a regular file: {relative}")
    witness_state = _modified_witness_state(item, observed)
    if witness_state is not None:
        if not recover_link_window:
            return witness_state
        return _recover_modified_witness_window(item, witness_state)
    stage = item["stage_path"]
    if item["change"] == "added" and observed.st_nlink == 2:
        try:
            stage_stat = _stat_at(stage, parent_fd)
        except FileNotFoundError:
            stage_stat = None
        if stage_stat is None or _identity(stage_stat) != _identity(observed):
            raise SourceApplyError(f"added target has an unknown hardlink: {relative}")
        _stable_source(
            destination,
            relative,
            item["rendered"],
            require_single_link=False,
            expected_mode=item["mode"],
            dir_fd=parent_fd,
        )
        if not recover_link_window:
            return "link_window"
        _unlink_bound(
            stage,
            _identity(stage_stat),
            remaining_path=destination,
            remaining_mode=item["mode"],
            dir_fd=parent_fd,
        )
        _fsync_parent_at(destination, parent_fd)
        observed = _stat_at(destination, parent_fd)
    if observed.st_nlink != 1:
        raise SourceApplyError(f"target link count differs: {relative}")
    try:
        _stable_source(
            destination,
            relative,
            item["rendered"],
            require_single_link=True,
            expected_mode=item["mode"],
            dir_fd=parent_fd,
        )
        return "rendered_final"
    except (SourceApplyError, recovery.Phase6RecoveryAuthorityError) as rendered_exc:
        if item["change"] != "modified":
            raise SourceApplyError(
                f"target drifted outside before/final states: {relative}"
            ) from rendered_exc
    try:
        _stable_source(
            destination,
            relative,
            item["before"],
            require_single_link=True,
            expected_mode=item["before_mode"],
            dir_fd=parent_fd,
        )
        return "before"
    except (SourceApplyError, recovery.Phase6RecoveryAuthorityError) as before_exc:
        raise SourceApplyError(
            f"target drifted outside before/final states: {relative}"
        ) from before_exc


def _actual_inventory_paths(root: Path) -> set[str]:
    return {
        candidate.relative_to(root).as_posix()
        for pattern in recovery.SOURCE_INVENTORY_GLOBS
        for candidate in root.glob(pattern)
    }


def _load_contract(args: argparse.Namespace) -> dict[str, Any]:
    authority_pin = {
        "mode": "0400",
        "path": recovery.PLANROW_RECOVERY_AUTHORITY_PATH,
        "sha256": args.authority_sha256,
        "size_bytes": args.authority_size_bytes,
    }
    _normalized, authority_data = recovery._stable_registered_pin(
        authority_pin, "plan-row recovery authority"
    )
    authority = recovery.validate_phase6_planrow_recovery_authority_apply_bytes(
        authority_data,
        expected_authority_sha256=args.authority_sha256,
        candidate_source_root=args.candidate_source_root,
    )
    proposal_pin = recovery._pin(authority["proposal"], "recovery proposal")
    _proposal_observed, proposal_data = recovery._stable_registered_pin(
        proposal_pin, "recovery proposal"
    )
    proposal = recovery.validate_phase6_recovery_proposal_apply_bytes(
        proposal_data, candidate_source_root=args.candidate_source_root
    )
    baseline_pin = recovery._pin(
        authority["baseline_source_snapshot"], "recovery source baseline"
    )
    _baseline_observed, baseline_data = recovery._stable_registered_pin(
        baseline_pin, "recovery source baseline"
    )
    baseline = recovery.validate_phase6_recovery_baseline_snapshot_bytes(baseline_data)
    render_plan_pin = {
        "mode": "0400",
        "path": recovery.RECOVERY_RENDER_PLAN_PATH,
        "sha256": args.render_plan_sha256,
        "size_bytes": args.render_plan_size_bytes,
    }
    _render_observed, render_plan_data = recovery._stable_registered_pin(
        render_plan_pin, "recovery source render plan"
    )
    render_plan = recovery.validate_phase6_recovery_source_render_plan_bytes(
        render_plan_data,
        candidate_source_root=args.candidate_source_root,
        expected_authority_sha256=args.authority_sha256,
    )
    if render_plan["authority"] != authority_pin:
        raise SourceApplyError("render-plan authority pin differs")
    prior_source_modes = recovery.prior_recovery_source_modes()
    return {
        "authority": authority,
        "authority_data": authority_data,
        "authority_pin": authority_pin,
        "baseline": baseline,
        "proposal": proposal,
        "prior_source_modes": prior_source_modes,
        "render_plan": render_plan,
        "render_plan_pin": render_plan_pin,
    }


def _build_plan(
    contract: Mapping[str, Any],
    candidate_root: Path,
    candidate_real: Path,
    target_root: Path,
    target_real: Path,
    authority_sha256: str,
    candidate_root_fd: int | None,
    target_root_fd: int | None,
    stack: ExitStack,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    baseline_files = _file_map(contract["baseline"]["files"], "baseline files")
    inventory = contract["proposal"]["source_inventory"]
    if type(inventory) is not dict or type(inventory.get("files")) is not list:
        raise SourceApplyError("proposal source inventory differs")
    candidate_files = _file_map(inventory["files"], "candidate files")
    delta = contract["authority"]["authorized_source_delta"]
    if type(delta) is not list or not delta:
        raise SourceApplyError("authorized source delta is empty")
    plan: list[dict[str, Any]] = []
    delta_paths: set[str] = set()
    for index, entry in enumerate(delta):
        if type(entry) is not dict or set(entry) != {
            "before",
            "candidate",
            "change",
            "path",
        }:
            raise SourceApplyError(f"authorized delta[{index}] keys differ")
        relative = _relative(entry["path"], f"authorized delta[{index}].path")
        change = entry["change"]
        before = _pin(
            entry["before"], f"authorized delta before {relative}", allow_none=True
        )
        tokenized = _pin(
            entry["candidate"],
            f"authorized delta candidate {relative}",
            allow_none=True,
        )
        if change == "deleted" or tokenized is None:
            raise SourceApplyError(f"source deletion is forbidden: {relative}")
        if change not in {"added", "modified"}:
            raise SourceApplyError(f"source change differs: {relative}")
        if (change == "added") != (before is None):
            raise SourceApplyError(f"source before/change binding differs: {relative}")
        if candidate_files.get(relative) != tokenized:
            raise SourceApplyError(f"candidate inventory pin differs: {relative}")
        if baseline_files.get(relative) != before:
            raise SourceApplyError(f"baseline inventory pin differs: {relative}")
        candidate_path = _repository_path(
            candidate_root, candidate_real, relative, require_leaf=True
        )
        with _repository_parent_fd(candidate_root_fd, relative) as candidate_parent_fd:
            tokenized_data, candidate_stat = _stable_source(
                candidate_path,
                relative,
                tokenized,
                require_single_link=True,
                dir_fd=candidate_parent_fd,
            )
        rendered_data = (
            recovery.render_phase6_recovery_source(tokenized_data, authority_sha256)
            if relative == recovery.RECOVERY_SOURCE_PATH
            else tokenized_data
        )
        rendered = _source_pin(rendered_data)
        if relative == recovery.RECOVERY_SOURCE_PATH:
            if contract["render_plan"]["tokenized_source"] != {
                "path": relative,
                **tokenized,
            } or contract["render_plan"]["rendered_source"] != {
                "path": relative,
                **rendered,
            }:
                raise SourceApplyError("render-plan recovery source pins differ")
        if before == rendered:
            raise SourceApplyError(f"source transition is ambiguous: {relative}")
        destination = _repository_path(
            target_root, target_real, relative, require_leaf=change == "modified"
        )
        parent_fd = stack.enter_context(_repository_parent_fd(target_root_fd, relative))
        candidate_mode = stat.S_IMODE(candidate_stat.st_mode)
        if (
            candidate_mode & 0o400 == 0
            or candidate_mode & 0o7000
            or (os.name == "posix" and candidate_mode & 0o002)
        ):
            raise SourceApplyError(f"candidate source mode is unsafe: {relative}")
        before_mode = (
            contract["prior_source_modes"].get(relative)
            if change == "modified"
            else None
        )
        if change == "modified" and before_mode not in {0o644, 0o664}:
            raise SourceApplyError(
                f"modified source mode lacks prior receipt authority: {relative}"
            )
        if change == "modified" and candidate_mode != before_mode:
            raise SourceApplyError(
                f"candidate source mode differs from prior: {relative}"
            )
        mode = before_mode if change == "modified" else _ADDED_SOURCE_MODE
        if mode & 0o400 == 0 or mode & 0o7002:
            raise SourceApplyError(f"source mode is unsafe: {relative}")
        stage_path = _stage_path(destination, relative, rendered, authority_sha256)
        item = {
            "before": before,
            "before_mode": before_mode,
            "candidate_path": candidate_path,
            "candidate_mode": f"{candidate_mode:04o}",
            "change": change,
            "destination": destination,
            "mode": mode,
            "parent_fd": parent_fd,
            "path": relative,
            "rendered": rendered,
            "rendered_data": rendered_data,
            "stage_path": stage_path,
            "target_real": target_real,
            "target_root_fd": target_root_fd,
            "target_root": target_root,
            "tokenized": tokenized,
            "witness_path": _witness_path(stage_path),
        }
        plan.append(item)
        delta_paths.add(relative)
    if [item["path"] for item in plan] != sorted(delta_paths):
        raise SourceApplyError("authorized delta paths are not sorted and unique")
    if set(candidate_files) != set(baseline_files) | {
        item["path"] for item in plan if item["change"] == "added"
    }:
        raise SourceApplyError("baseline/candidate path transition differs")
    return plan, baseline_files, candidate_files


def _validate_rendered_python(plan: list[dict[str, Any]]) -> None:
    for item in plan:
        if not item["path"].endswith(".py"):
            continue
        try:
            compile(
                item["rendered_data"],
                item["path"],
                "exec",
                ast.PyCF_ONLY_AST,
                dont_inherit=True,
            )
        except (SyntaxError, UnicodeError, ValueError) as exc:
            raise SourceApplyError(
                f"rendered Python syntax differs: {item['path']}: {exc}"
            ) from exc


def _validate_target_inventory(
    target_root: Path,
    target_real: Path,
    plan: list[dict[str, Any]],
    baseline_files: Mapping[str, Mapping[str, Any]],
    candidate_files: Mapping[str, Mapping[str, Any]],
    target_root_fd: int | None,
    *,
    recover_link_windows: bool = True,
) -> dict[str, str]:
    _assert_plan_bindings(plan)
    actual_paths = _actual_inventory_paths(target_root)
    extra = actual_paths - set(candidate_files)
    if extra:
        raise SourceApplyError(
            f"formal repository has extra source paths: {sorted(extra)[:10]}"
        )
    missing_baseline = set(baseline_files) - actual_paths
    if missing_baseline:
        raise SourceApplyError(
            f"formal repository omits baseline paths: {sorted(missing_baseline)[:10]}"
        )
    delta_paths = {item["path"] for item in plan}
    for relative in sorted(set(candidate_files) - delta_paths):
        path = _repository_path(target_root, target_real, relative, require_leaf=True)
        with _repository_parent_fd(target_root_fd, relative) as parent_fd:
            _data, observed = _stable_source(
                path,
                relative,
                candidate_files[relative],
                require_single_link=True,
                dir_fd=parent_fd,
            )
        mode = stat.S_IMODE(observed.st_mode)
        if mode & 0o400 == 0 or mode & 0o7000 or (os.name == "posix" and mode & 0o002):
            raise SourceApplyError(f"unchanged source mode is unsafe: {relative}")
    states = {
        item["path"]: _classify_target(item, recover_link_window=False) for item in plan
    }
    absent_allowed = {
        item["path"]
        for item in plan
        if item["change"] == "added" and states[item["path"]] == "absent"
    }
    if set(candidate_files) - actual_paths != absent_allowed:
        raise SourceApplyError("formal repository partial inventory differs")
    for item in plan:
        if states[item["path"]] not in {
            "exchange_window",
            "link_window",
            "witness_window",
        }:
            continue
        if not recover_link_windows:
            continue
        recovered = _classify_target(item, recover_link_window=True)
        expected = (
            "before" if states[item["path"]] == "witness_window" else "rendered_final"
        )
        if recovered != expected:
            raise SourceApplyError(
                f"source link-window recovery failed: {item['path']}"
            )
        states[item["path"]] = recovered
    _assert_plan_bindings(plan)
    return states


def _apply_plan(plan: list[dict[str, Any]], states: Mapping[str, str]) -> None:
    _assert_plan_bindings(plan)
    for item in plan:
        if states[item["path"]] != "rendered_final":
            _prepare_stage(item)
    for item in plan:
        parent_fd = item["parent_fd"]
        current = _classify_target(item, recover_link_window=True)
        if current == "rendered_final":
            try:
                stage_stat = _stat_at(item["stage_path"], parent_fd)
            except FileNotFoundError:
                continue
            _require_private_regular(
                stage_stat, "unexpected staged source", link_counts={1}
            )
            matches_rendered = _matches_source(
                item,
                item["stage_path"],
                item["rendered"],
                item["mode"],
                require_single_link=True,
            )
            matches_before = item["change"] == "modified" and _matches_source(
                item,
                item["stage_path"],
                item["before"],
                item["before_mode"],
                require_single_link=True,
            )
            if not (matches_rendered or matches_before):
                raise SourceApplyError(
                    f"unwitnessed staged source differs: {item['path']}"
                )
            _unlink_bound(item["stage_path"], _identity(stage_stat), dir_fd=parent_fd)
            _fsync_parent_at(item["stage_path"], parent_fd)
            _assert_parent_binding(item)
            continue
        if item["change"] == "modified":
            if current != "before":
                raise SourceApplyError(f"modified source state differs: {item['path']}")
            _stable_source(
                item["stage_path"],
                item["path"],
                item["rendered"],
                require_single_link=True,
                expected_mode=item["mode"],
                dir_fd=parent_fd,
            )
            _replace_modified(item)
        else:
            if current != "absent":
                raise SourceApplyError(f"added source state differs: {item['path']}")
            try:
                _assert_parent_binding(item)
                _link_at(item["stage_path"], item["destination"], parent_fd)
            except FileExistsError:
                if _classify_target(item, recover_link_window=True) != "rendered_final":
                    raise
                continue
            stage_stat = _stat_at(item["stage_path"], parent_fd)
            final_stat = _stat_at(item["destination"], parent_fd)
            if (
                _identity(stage_stat) != _identity(final_stat)
                or stage_stat.st_nlink != 2
                or final_stat.st_nlink != 2
            ):
                raise SourceApplyError("added source hardlink binding differs")
            _stable_source(
                item["destination"],
                item["path"],
                item["rendered"],
                require_single_link=False,
                expected_mode=item["mode"],
                dir_fd=parent_fd,
            )
            _unlink_bound(
                item["stage_path"],
                _identity(stage_stat),
                remaining_path=item["destination"],
                remaining_mode=item["mode"],
                dir_fd=parent_fd,
            )
            _fsync_parent_at(item["destination"], parent_fd)
            _assert_parent_binding(item)
        if _classify_target(item, recover_link_window=True) != "rendered_final":
            raise SourceApplyError(
                f"source did not reach rendered final: {item['path']}"
            )
    _assert_plan_bindings(plan)


def _rename_exchange(
    source: Path, destination: Path, parent_fd: int | None = None
) -> bool:
    if not sys.platform.startswith("linux"):
        raise SourceApplyError("Linux renameat2 is required for modified source apply")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise SourceApplyError("renameat2 is required for modified source CAS") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd if parent_fd is not None else -100,
        os.fsencode(source.name if parent_fd is not None else source),
        parent_fd if parent_fd is not None else -100,
        os.fsencode(destination.name if parent_fd is not None else destination),
        2,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), os.fspath(destination))
    return True


def _rollback_modified_exchange(
    item: Mapping[str, Any],
    rendered_identity: tuple[int, int],
    displaced_identity: tuple[int, int],
    rendered_signature: tuple[int, ...],
    displaced_signature: tuple[int, ...],
) -> None:
    parent_fd = item["parent_fd"]
    destination = _stat_at(item["destination"], parent_fd)
    stage = _stat_at(item["stage_path"], parent_fd)
    witness = _stat_at(item["witness_path"], parent_fd)
    if (
        _identity(destination) != rendered_identity
        or _identity(stage) != displaced_identity
        or _identity(witness) != rendered_identity
        or _signature(destination) != rendered_signature
        or _signature(stage) != displaced_signature
        or _signature(witness) != rendered_signature
        or destination.st_nlink != 2
        or witness.st_nlink != 2
    ):
        raise SourceApplyError("modified source CAS rollback precondition differs")
    _stable_source(
        item["destination"],
        item["path"],
        item["rendered"],
        require_single_link=False,
        expected_mode=item["mode"],
        dir_fd=parent_fd,
    )
    _stable_source(
        item["witness_path"],
        item["path"],
        item["rendered"],
        require_single_link=False,
        expected_mode=item["mode"],
        dir_fd=parent_fd,
    )
    _rename_exchange(item["stage_path"], item["destination"], parent_fd)
    _fsync_parent_at(item["destination"], parent_fd)
    restored = _stat_at(item["destination"], parent_fd)
    rendered_stage = _stat_at(item["stage_path"], parent_fd)
    rebound_witness = _stat_at(item["witness_path"], parent_fd)
    if (
        _identity(restored) != displaced_identity
        or _identity(rendered_stage) != rendered_identity
        or _identity(rebound_witness) != rendered_identity
        or rendered_stage.st_nlink != 2
        or rebound_witness.st_nlink != 2
    ):
        raise SourceApplyError("modified source CAS rollback binding differs")
    _unlink_witness(item, item["stage_path"])


def _replace_modified(item: dict[str, Any]) -> None:
    parent_fd = item["parent_fd"]
    _assert_parent_binding(item)
    rendered_stage = _stat_at(item["stage_path"], parent_fd)
    _require_private_regular(rendered_stage, "modified rendered stage", link_counts={1})
    _stable_source(
        item["stage_path"],
        item["path"],
        item["rendered"],
        require_single_link=True,
        expected_mode=item["mode"],
        dir_fd=parent_fd,
    )
    rendered_identity = _identity(rendered_stage)
    _link_at(item["stage_path"], item["witness_path"], parent_fd)
    _fsync_parent_at(item["witness_path"], parent_fd)
    witnessed_stage = _stat_at(item["stage_path"], parent_fd)
    witness = _stat_at(item["witness_path"], parent_fd)
    if (
        _identity(witnessed_stage) != rendered_identity
        or _identity(witness) != rendered_identity
        or witnessed_stage.st_nlink != 2
        or witness.st_nlink != 2
    ):
        raise SourceApplyError("modified exchange witness creation differs")
    try:
        _assert_parent_binding(item)
        _rename_exchange(item["stage_path"], item["destination"], parent_fd)
    except BaseException:
        try:
            destination = _stat_at(item["destination"], parent_fd)
            stage = _stat_at(item["stage_path"], parent_fd)
            witness = _stat_at(item["witness_path"], parent_fd)
            if (
                _identity(destination) == rendered_identity
                and _identity(witness) == rendered_identity
            ):
                _fsync_parent_at(item["destination"], parent_fd)
            elif (
                _identity(stage) == rendered_identity
                and _identity(witness) == rendered_identity
            ):
                _unlink_witness(item, item["stage_path"])
            else:
                raise SourceApplyError("modified exchange failure state is ambiguous")
        except BaseException as recovery_exc:
            raise SourceApplyError(
                f"modified exchange failure recovery failed: {item['path']}: "
                f"{recovery_exc}"
            ) from recovery_exc
        raise
    _fsync_parent_at(item["destination"], parent_fd)
    displaced = _stat_at(item["stage_path"], parent_fd)
    installed = _stat_at(item["destination"], parent_fd)
    witness = _stat_at(item["witness_path"], parent_fd)
    displaced_identity = _identity(displaced)
    displaced_signature = _signature(displaced)
    rendered_signature = _signature(installed)
    if (
        _identity(installed) != rendered_identity
        or _identity(witness) != rendered_identity
        or installed.st_nlink != 2
        or witness.st_nlink != 2
    ):
        raise SourceApplyError("modified source CAS exchange binding differs")
    try:
        _assert_parent_binding(item)
        _stable_source(
            item["destination"],
            item["path"],
            item["rendered"],
            require_single_link=False,
            expected_mode=item["mode"],
            dir_fd=parent_fd,
        )
        _stable_source(
            item["stage_path"],
            item["path"],
            item["before"],
            require_single_link=True,
            expected_mode=item["before_mode"],
            dir_fd=parent_fd,
        )
    except BaseException:
        try:
            _rollback_modified_exchange(
                item,
                rendered_identity,
                displaced_identity,
                rendered_signature,
                displaced_signature,
            )
        except BaseException as rollback_exc:
            raise SourceApplyError(
                f"modified source CAS rollback failed: {item['path']}: {rollback_exc}"
            ) from rollback_exc
        raise
    _unlink_bound(item["stage_path"], displaced_identity, dir_fd=parent_fd)
    _fsync_parent_at(item["destination"], parent_fd)
    _unlink_witness(item, item["destination"])
    _assert_parent_binding(item)


def _receipt_files(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "before": deepcopy(item["before"]),
            "before_mode": (
                f"{item['before_mode']:04o}"
                if item["before_mode"] is not None
                else None
            ),
            "change": item["change"],
            "path": item["path"],
            "rendered": deepcopy(item["rendered"]),
            "rendered_mode": f"{item['mode']:04o}",
            "state": "rendered_final",
            "tokenized": deepcopy(item["tokenized"]),
            "tokenized_mode": item["candidate_mode"],
        }
        for item in plan
    ]


def _rendered_inventory(
    plan: list[dict[str, Any]],
    candidate_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rendered = {path: deepcopy(pin) for path, pin in candidate_files.items()}
    for item in plan:
        rendered[item["path"]] = deepcopy(item["rendered"])
    return {
        "files": [{"path": path, **rendered[path]} for path in sorted(rendered)],
        "inventory_globs": list(recovery.SOURCE_INVENTORY_GLOBS),
        "representation": "raw-rendered",
        "schema_version": "sana-phase6-raw-rendered-source-inventory-v1",
    }


def _build_receipt(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    plan: list[dict[str, Any]],
    candidate_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    registered = recovery._utc(args.registered_at_utc, "source execution registration")
    authority_time = recovery._utc(
        contract["authority"]["registered_at_utc"], "recovery authority registration"
    )
    render_plan_time = recovery._utc(
        contract["render_plan"]["registered_at_utc"], "render-plan registration"
    )
    if registered < max(authority_time, render_plan_time):
        raise SourceApplyError("source execution receipt predates an input")
    return {
        "artifact_path": recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH,
        "artifact_role": "phase6_recovery_source_execution_receipt",
        "authorized_source_delta": deepcopy(
            contract["authority"]["authorized_source_delta"]
        ),
        "authority": deepcopy(contract["authority_pin"]),
        "candidate_source_root": args.candidate_source_root,
        "files": _receipt_files(plan),
        "gpu_executed": False,
        "lifecycle": recovery.recovery_lifecycle(),
        "proposal": deepcopy(contract["authority"]["proposal"]),
        "receipt_is_commit_marker": True,
        "registered_at_utc": args.registered_at_utc,
        "rendered_source_inventory": _rendered_inventory(plan, candidate_files),
        "schema_version": SOURCE_EXECUTION_SCHEMA_VERSION,
        "source_write_completed": True,
        "source_render_plan": deepcopy(contract["render_plan_pin"]),
        "status": "registered",
        "target_repository_root": args.target_repository_root,
    }


def _validate_receipt_bytes(
    data: bytes,
    *,
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    plan: list[dict[str, Any]],
    candidate_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = recovery._strict_json(data, "source execution receipt")
    expected_keys = {
        "artifact_path",
        "artifact_role",
        "authorized_source_delta",
        "authority",
        "candidate_source_root",
        "files",
        "gpu_executed",
        "lifecycle",
        "proposal",
        "receipt_is_commit_marker",
        "registered_at_utc",
        "rendered_source_inventory",
        "schema_version",
        "source_write_completed",
        "source_render_plan",
        "status",
        "target_repository_root",
    }
    if type(receipt) is not dict or set(receipt) != expected_keys:
        raise SourceApplyError("source execution receipt keys differ")
    registered = recovery._utc(
        receipt["registered_at_utc"], "source execution registration"
    )
    authority_time = recovery._utc(
        contract["authority"]["registered_at_utc"],
        "recovery authority registration",
    )
    render_plan_time = recovery._utc(
        contract["render_plan"]["registered_at_utc"], "render-plan registration"
    )
    if registered < max(authority_time, render_plan_time):
        raise SourceApplyError("source execution receipt predates an input")
    recovery.validate_recovery_lifecycle(
        receipt["lifecycle"], "source execution lifecycle"
    )
    expected = {
        "artifact_path": recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH,
        "artifact_role": "phase6_recovery_source_execution_receipt",
        "authorized_source_delta": contract["authority"]["authorized_source_delta"],
        "authority": contract["authority_pin"],
        "candidate_source_root": args.candidate_source_root,
        "files": _receipt_files(plan),
        "gpu_executed": False,
        "lifecycle": recovery.recovery_lifecycle(),
        "proposal": contract["authority"]["proposal"],
        "receipt_is_commit_marker": True,
        "rendered_source_inventory": _rendered_inventory(plan, candidate_files),
        "schema_version": SOURCE_EXECUTION_SCHEMA_VERSION,
        "source_write_completed": True,
        "source_render_plan": contract["render_plan_pin"],
        "status": "registered",
        "target_repository_root": args.target_repository_root,
    }
    for key, value in expected.items():
        try:
            recovery._typed_equal(
                receipt[key], value, f"source execution receipt {key}"
            )
        except recovery.Phase6RecoveryAuthorityError as exc:
            raise SourceApplyError(str(exc)) from exc
    return deepcopy(receipt)


def _existing_receipt(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    plan: list[dict[str, Any]],
    candidate_files: Mapping[str, Mapping[str, Any]],
) -> tuple[bytes, str] | None:
    path = Path(recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH)
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None
    _observed, data = recovery._stable_read_fd(
        os.fspath(path),
        name="source execution receipt",
        expected_size=None,
        expected_sha256=None,
        expected_mode=0o400,
        require_single_link=False,
        maximum_bytes=recovery._MAX_REGISTERED_BYTES,
        allow_empty=False,
    )
    _validate_receipt_bytes(
        data,
        args=args,
        contract=contract,
        plan=plan,
        candidate_files=candidate_files,
    )
    digest = publication._publish_exact(os.fspath(path), os.fspath(path), data)
    return data, digest


def _remove_exact_receipt(data: bytes) -> None:
    path = Path(recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH)
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise SourceApplyError("published source receipt rollback binding differs")
    _pin, existing = recovery._stable_read_fd(
        os.fspath(path),
        name="published source execution receipt rollback",
        expected_size=len(data),
        expected_sha256=sha256(data).hexdigest(),
        expected_mode=0o400,
        require_single_link=True,
        maximum_bytes=recovery._MAX_REGISTERED_BYTES,
        allow_empty=False,
    )
    if existing != data:
        raise SourceApplyError("published source receipt rollback bytes differ")
    _unlink_bound(path, _identity(observed))
    publication._fsync_parent(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source-root", required=True)
    parser.add_argument("--target-repository-root", required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--authority-size-bytes", required=True, type=int)
    parser.add_argument("--render-plan-sha256", required=True)
    parser.add_argument("--render-plan-size-bytes", required=True, type=int)
    parser.add_argument("--registered-at-utc", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _execute_plan(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    candidate_root: Path,
    candidate_real: Path,
    target_root: Path,
    target_real: Path,
    candidate_root_fd: int | None,
    target_root_fd: int | None,
    stack: ExitStack,
) -> tuple[str, int]:
    plan, baseline_files, candidate_files = _build_plan(
        contract,
        candidate_root,
        candidate_real,
        target_root,
        target_real,
        args.authority_sha256,
        candidate_root_fd,
        target_root_fd,
        stack,
    )
    _validate_rendered_python(plan)
    try:
        os.lstat(recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH)
        receipt_present = True
    except FileNotFoundError:
        receipt_present = False
    states = _validate_target_inventory(
        target_root,
        target_real,
        plan,
        baseline_files,
        candidate_files,
        target_root_fd,
        recover_link_windows=not receipt_present,
    )
    exact_baseline = all(
        states[item["path"]] == ("absent" if item["change"] == "added" else "before")
        for item in plan
    )
    if exact_baseline:
        recovery.validate_phase6_planrow_recovery_authority_prerender_bytes(
            contract["authority_data"],
            expected_authority_sha256=args.authority_sha256,
            candidate_source_root=args.candidate_source_root,
        )
    all_final = all(state == "rendered_final" for state in states.values())
    if receipt_present:
        if not all_final:
            raise SourceApplyError(
                "registered source execution receipt requires an exact final tree"
            )
        recovery.validate_phase6_planrow_recovery_authority_bytes(
            contract["authority_data"],
            expected_authority_sha256=args.authority_sha256,
            target_repository_root=args.target_repository_root,
        )
        existing = _existing_receipt(args, contract, plan, candidate_files)
        if existing is None:
            raise SourceApplyError("registered source execution receipt disappeared")
        data, digest = existing
        _assert_cpu_only_runtime()
        return digest, len(data)
    _apply_plan(plan, states)
    _validate_target_inventory(
        target_root,
        target_real,
        plan,
        baseline_files,
        candidate_files,
        target_root_fd,
    )
    recovery.validate_phase6_planrow_recovery_authority_bytes(
        contract["authority_data"],
        expected_authority_sha256=args.authority_sha256,
        target_repository_root=args.target_repository_root,
    )
    existing = _existing_receipt(args, contract, plan, candidate_files)
    if existing is not None:
        data, digest = existing
        _assert_cpu_only_runtime()
        return digest, len(data)
    receipt = _build_receipt(args, contract, plan, candidate_files)
    data = recovery.canonical_json_bytes(receipt)
    _validate_receipt_bytes(
        data,
        args=args,
        contract=contract,
        plan=plan,
        candidate_files=candidate_files,
    )
    final_states = _validate_target_inventory(
        target_root,
        target_real,
        plan,
        baseline_files,
        candidate_files,
        target_root_fd,
    )
    if any(state != "rendered_final" for state in final_states.values()):
        raise SourceApplyError("formal repository left the final state before receipt")
    recovery.validate_phase6_planrow_recovery_authority_bytes(
        contract["authority_data"],
        expected_authority_sha256=args.authority_sha256,
        target_repository_root=args.target_repository_root,
    )
    _assert_cpu_only_runtime()
    digest = publication._publish_exact(
        args.output, recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH, data
    )
    try:
        committed_states = _validate_target_inventory(
            target_root,
            target_real,
            plan,
            baseline_files,
            candidate_files,
            target_root_fd,
            recover_link_windows=False,
        )
        if any(state != "rendered_final" for state in committed_states.values()):
            raise SourceApplyError(
                "formal repository left the final state during receipt publication"
            )
        recovery.validate_phase6_planrow_recovery_authority_bytes(
            contract["authority_data"],
            expected_authority_sha256=args.authority_sha256,
            target_repository_root=args.target_repository_root,
        )
    except BaseException:
        try:
            _remove_exact_receipt(data)
        except BaseException as rollback_exc:
            raise SourceApplyError(
                f"source receipt rollback failed after final-tree drift: {rollback_exc}"
            ) from rollback_exc
        raise
    _assert_cpu_only_runtime()
    return digest, len(data)


def _execute_under_lock(args: argparse.Namespace) -> tuple[str, int]:
    _assert_cpu_only_runtime()
    if not sys.platform.startswith("linux"):
        raise SourceApplyError("formal source apply requires Linux renameat2")
    if args.target_repository_root != recovery.REPOSITORY_ROOT:
        raise SourceApplyError("target repository root differs")
    if args.output != recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH:
        raise SourceApplyError("source execution output path differs")
    candidate_root, candidate_real = _repository_root(
        args.candidate_source_root, "candidate source root"
    )
    target_root, target_real = _repository_root(
        args.target_repository_root, "target repository root"
    )
    if candidate_real == target_real:
        raise SourceApplyError("candidate and target repositories must be distinct")
    contract = _load_contract(args)
    with ExitStack() as stack:
        candidate_root_fd = stack.enter_context(_repository_root_fd(candidate_root))
        target_root_fd = stack.enter_context(_repository_root_fd(target_root))
        return _execute_plan(
            args,
            contract,
            candidate_root,
            candidate_real,
            target_root,
            target_real,
            candidate_root_fd,
            target_root_fd,
            stack,
        )


def _execute(args: argparse.Namespace) -> tuple[str, int]:
    _assert_cpu_only_runtime()
    if not sys.platform.startswith("linux"):
        raise SourceApplyError("formal source apply requires Linux renameat2")
    if args.target_repository_root != recovery.REPOSITORY_ROOT:
        raise SourceApplyError("target repository root differs")
    if args.output != recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH:
        raise SourceApplyError("source execution output path differs")
    with _source_apply_lock():
        return _execute_under_lock(args)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        digest, size = _execute(args)
    except (
        FileExistsError,
        OSError,
        SourceApplyError,
        TypeError,
        ValueError,
        recovery.Phase6RecoveryAuthorityError,
    ) as exc:
        print(f"Phase-6 recovery source apply failed: {exc}", file=sys.stderr)
        return 2
    print(
        "PASS "
        f"output={recovery.RECOVERY_SOURCE_EXECUTION_RECEIPT_PATH} "
        f"sha256={digest} size={size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
