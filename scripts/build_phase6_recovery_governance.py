#!/usr/bin/env python3
"""Build the acyclic Phase-6 recovery governance chain without torch or CUDA."""

from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path
import stat
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wam.train import phase6_recovery as recovery  # noqa: E402


class GovernanceCliError(RuntimeError):
    """Raised before an immutable governance output can be published."""


_REQUIRED_CPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "GIT_OPTIONAL_LOCKS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
}


def _assert_cpu_only_runtime() -> None:
    if not sys.dont_write_bytecode:
        raise GovernanceCliError(
            "governance CLI requires python -B/dont-write-bytecode"
        )
    for key, expected in _REQUIRED_CPU_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise GovernanceCliError(f"governance CLI environment {key} differs")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise GovernanceCliError("governance CLI forbids torch imports")


def _pin(path: str, digest: str, size: int) -> dict[str, Any]:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise GovernanceCliError("pin SHA256 must be lowercase hexadecimal")
    if type(size) is not int or size < 1:
        raise GovernanceCliError("pin size must be a positive integer")
    return {"mode": "0400", "path": path, "sha256": digest, "size_bytes": size}


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publication_temp_path(path: Path, digest: str) -> Path:
    target_tag = sha256(os.fsencode(path.name)).hexdigest()[:16]
    return path.with_name(f".phase6-publish-{digest}-{target_tag}.tmp")


def _file_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _file_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _unlink_bound(
    path: Path,
    identity: tuple[int, int],
    *,
    remaining_path: Path | None = None,
) -> bool:
    try:
        observed = os.lstat(path)
    except FileNotFoundError:
        return False
    if (
        _file_identity(observed) != identity
        or stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
    ):
        raise GovernanceCliError(
            f"refusing to unlink replaced publication temp: {path}"
        )
    if os.name == "nt":
        os.chmod(path, 0o600)
    try:
        os.unlink(path)
    except BaseException:
        if os.name == "nt":
            try:
                rebound = os.lstat(path)
                if _file_identity(rebound) == identity:
                    os.chmod(path, 0o400)
            except FileNotFoundError:
                pass
        raise
    if remaining_path is not None:
        remaining = os.lstat(remaining_path)
        if _file_identity(remaining) != identity:
            raise GovernanceCliError("remaining publication hardlink was replaced")
        if os.name == "nt":
            os.chmod(remaining_path, 0o400)
    return True


def _stable_exact_publication_file(
    path: Path,
    data: bytes,
    digest: str,
    *,
    name: str,
    require_single_link: bool,
) -> None:
    _observed, readback = recovery._stable_read_fd(
        os.fspath(path),
        name=name,
        expected_size=len(data),
        expected_sha256=digest,
        expected_mode=0o400,
        require_single_link=require_single_link,
        maximum_bytes=recovery._MAX_REGISTERED_BYTES,
        allow_empty=False,
    )
    if readback != data:
        raise GovernanceCliError(f"{name} bytes differ despite their digest")


def _fsync_bound_file(path: Path, identity: tuple[int, int]) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if os.name == "posix" and nofollow == 0:
        raise GovernanceCliError("O_NOFOLLOW is required for governance publication")
    if os.name == "nt":
        before = os.lstat(path)
        if _file_identity(before) != identity:
            raise GovernanceCliError("staged governance path binding differs")
        os.chmod(path, 0o600)
    flags = (
        (os.O_RDWR if os.name == "nt" else os.O_RDONLY)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
    )
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        observed = os.fstat(descriptor)
        if (
            _file_identity(observed) != identity
            or not stat.S_ISREG(observed.st_mode)
            or (os.name != "nt" and not recovery._mode_matches(observed.st_mode, 0o400))
            or observed.st_nlink != 1
        ):
            raise GovernanceCliError("staged governance descriptor binding differs")
        os.fsync(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.name == "nt":
            rebound = os.lstat(path)
            if _file_identity(rebound) != identity:
                raise GovernanceCliError("staged governance path was replaced")
            os.chmod(path, 0o400)


def _cleanup_publication_temp(
    temp: Path, identity: tuple[int, int], final: Path
) -> None:
    remaining_path = None
    try:
        final_stat = os.lstat(final)
        if _file_identity(final_stat) == identity:
            remaining_path = final
    except FileNotFoundError:
        pass
    if _unlink_bound(temp, identity, remaining_path=remaining_path):
        _fsync_parent(temp)


def _recover_existing_final(path: Path, temp: Path, data: bytes, digest: str) -> bool:
    try:
        final_stat = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(final_stat.st_mode) or not stat.S_ISREG(final_stat.st_mode):
        raise FileExistsError(f"refusing to overwrite governance artifact: {path}")
    try:
        temp_stat = os.lstat(temp)
    except FileNotFoundError:
        temp_stat = None
    same_link_window = temp_stat is not None and _file_identity(
        temp_stat
    ) == _file_identity(final_stat)
    try:
        if same_link_window:
            if (
                stat.S_ISLNK(temp_stat.st_mode)
                or not stat.S_ISREG(temp_stat.st_mode)
                or final_stat.st_nlink != 2
                or temp_stat.st_nlink != 2
            ):
                raise GovernanceCliError(
                    "publication link window has an unexpected link count"
                )
            _stable_exact_publication_file(
                path,
                data,
                digest,
                name="linked governance artifact",
                require_single_link=False,
            )
            _unlink_bound(temp, _file_identity(temp_stat), remaining_path=path)
            _fsync_parent(path)
        _stable_exact_publication_file(
            path,
            data,
            digest,
            name="published governance artifact",
            require_single_link=True,
        )
    except (GovernanceCliError, recovery.Phase6RecoveryAuthorityError) as exc:
        raise FileExistsError(
            f"refusing to overwrite governance artifact: {path}"
        ) from exc
    return True


def _prepare_publication_temp(temp: Path, data: bytes, digest: str) -> tuple[int, int]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _attempt in range(2):
        try:
            descriptor = os.open(temp, flags, 0o600)
        except FileExistsError:
            before = os.lstat(temp)
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (os.name == "posix" and before.st_uid != os.geteuid())
            ):
                raise GovernanceCliError(
                    f"publication temp is not a private regular file: {temp}"
                )
            try:
                _stable_exact_publication_file(
                    temp,
                    data,
                    digest,
                    name="staged governance artifact",
                    require_single_link=True,
                )
            except (GovernanceCliError, recovery.Phase6RecoveryAuthorityError):
                after = os.lstat(temp)
                if _file_signature(before) != _file_signature(after):
                    raise GovernanceCliError(
                        "publication temp changed while checking crash recovery"
                    )
                _unlink_bound(temp, _file_identity(before))
                _fsync_parent(temp)
                continue
            _fsync_bound_file(temp, _file_identity(before))
            _fsync_parent(temp)
            return _file_identity(before)
        created_identity: tuple[int, int] | None = None
        try:
            created = os.fstat(descriptor)
            if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
                raise GovernanceCliError("new publication temp identity differs")
            created_identity = _file_identity(created)
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("publication temp write made no progress")
                remaining = remaining[written:]
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            if created_identity is not None:
                if _unlink_bound(temp, created_identity):
                    _fsync_parent(temp)
            raise
        os.close(descriptor)
        try:
            observed = os.lstat(temp)
            if _file_identity(observed) != created_identity:
                raise GovernanceCliError("publication temp was replaced after writing")
            _stable_exact_publication_file(
                temp,
                data,
                digest,
                name="staged governance artifact",
                require_single_link=True,
            )
            _fsync_parent(temp)
        except BaseException:
            if created_identity is not None and _unlink_bound(temp, created_identity):
                _fsync_parent(temp)
            raise
        return created_identity
    raise GovernanceCliError("could not recover the deterministic publication temp")


def _publish_exact(path_value: str, expected_path: str, data: bytes) -> str:
    if path_value != expected_path:
        raise GovernanceCliError("output path differs from the fixed governance path")
    if (
        not isinstance(data, bytes)
        or not data
        or len(data) > recovery._MAX_REGISTERED_BYTES
    ):
        raise GovernanceCliError("governance publication bytes must be non-empty")
    path = Path(path_value)
    if not path.is_absolute() or not path.parent.is_dir() or path.parent.is_symlink():
        raise GovernanceCliError(
            "output parent must be an existing non-symlink directory"
        )
    digest = sha256(data).hexdigest()
    temp = _publication_temp_path(path, digest)
    if _recover_existing_final(path, temp, data, digest):
        return digest
    temp_identity: tuple[int, int] | None = None
    link_binding_validated = False
    linked_identity: tuple[int, int] | None = None
    try:
        temp_identity = _prepare_publication_temp(temp, data, digest)
        os.link(temp, path, follow_symlinks=False)
        final_stat = os.lstat(path)
        temp_stat = os.lstat(temp)
        linked_identity = _file_identity(final_stat)
        if (
            linked_identity != temp_identity
            or _file_identity(temp_stat) != temp_identity
            or final_stat.st_nlink != 2
            or temp_stat.st_nlink != 2
        ):
            raise GovernanceCliError("publication hardlink binding differs")
        link_binding_validated = True
        _stable_exact_publication_file(
            path,
            data,
            digest,
            name="linked governance artifact",
            require_single_link=False,
        )
        _unlink_bound(temp, temp_identity, remaining_path=path)
        _fsync_parent(path)
        _stable_exact_publication_file(
            path,
            data,
            digest,
            name="published governance artifact",
            require_single_link=True,
        )
    except FileExistsError as exc:
        try:
            recovered = _recover_existing_final(path, temp, data, digest)
        except BaseException:
            if temp_identity is not None:
                _cleanup_publication_temp(temp, temp_identity, path)
            raise
        if temp_identity is not None:
            _cleanup_publication_temp(temp, temp_identity, path)
        if recovered:
            return digest
        raise FileExistsError(
            f"refusing to overwrite governance artifact: {path}"
        ) from exc
    except BaseException:
        if not link_binding_validated and linked_identity is not None:
            try:
                _unlink_bound(path, linked_identity, remaining_path=temp)
            except (FileNotFoundError, GovernanceCliError, OSError):
                pass
        if temp_identity is not None:
            try:
                _cleanup_publication_temp(temp, temp_identity, path)
            except (FileNotFoundError, GovernanceCliError, OSError):
                pass
        raise
    return digest


def _common_time(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registered-at-utc", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser("baseline")
    baseline.add_argument("--target-repository-root", required=True)
    baseline.add_argument("--output", required=True)
    _common_time(baseline)

    proposal = commands.add_parser("proposal")
    proposal.add_argument("--candidate-source-root", required=True)
    proposal.add_argument("--target-repository-root", required=True)
    proposal.add_argument("--baseline-sha256", required=True)
    proposal.add_argument("--baseline-size-bytes", required=True, type=int)
    proposal.add_argument("--output", required=True)
    _common_time(proposal)

    subject = commands.add_parser("subject")
    subject.add_argument("--candidate-source-root", required=True)
    subject.add_argument("--proposal-sha256", required=True)
    subject.add_argument("--proposal-size-bytes", required=True, type=int)
    subject.add_argument("--output", required=True)
    _common_time(subject)

    review = commands.add_parser("review")
    review.add_argument("--candidate-source-root", required=True)
    review.add_argument("--review-subject-sha256", required=True)
    review.add_argument("--review-subject-size-bytes", required=True, type=int)
    review.add_argument("--slot", required=True, choices=("A", "B"))
    review.add_argument("--reviewer-identity", required=True)
    review.add_argument("--reviewer-identity-namespace", required=True)
    review.add_argument("--session-or-task-id", required=True)
    review.add_argument("--output", required=True)
    _common_time(review)

    final = commands.add_parser("final")
    final.add_argument("--candidate-source-root", required=True)
    final.add_argument("--review-subject-sha256", required=True)
    final.add_argument("--review-subject-size-bytes", required=True, type=int)
    for slot in ("a", "b"):
        final.add_argument(f"--review-{slot}-sha256", required=True)
        final.add_argument(f"--review-{slot}-size-bytes", required=True, type=int)
    final.add_argument("--output", required=True)
    _common_time(final)

    render = commands.add_parser("render-plan")
    render.add_argument("--candidate-source-root", required=True)
    render.add_argument("--authority-sha256", required=True)
    render.add_argument("--authority-size-bytes", required=True, type=int)
    render.add_argument("--target-repository-root", required=True)
    render.add_argument("--output", required=True)
    _common_time(render)
    return parser


def _build(args: argparse.Namespace) -> tuple[str, bytes]:
    if args.command == "baseline":
        value = recovery.build_phase6_recovery_baseline_snapshot(
            repository_root=args.target_repository_root,
            registered_at_utc=args.registered_at_utc,
        )
        data = recovery.canonical_json_bytes(value)
        recovery.validate_phase6_recovery_baseline_snapshot_bytes(data)
        return recovery.RECOVERY_BASELINE_SNAPSHOT_PATH, data

    if args.command == "proposal":
        baseline_pin = _pin(
            recovery.RECOVERY_BASELINE_SNAPSHOT_PATH,
            args.baseline_sha256,
            args.baseline_size_bytes,
        )
        value = recovery.build_phase6_recovery_proposal(
            baseline_source_snapshot_pin=baseline_pin,
            candidate_source_root=args.candidate_source_root,
            target_repository_root=args.target_repository_root,
            registered_at_utc=args.registered_at_utc,
        )
        data = recovery.canonical_json_bytes(value)
        recovery.validate_phase6_recovery_proposal_bytes(
            data, source_root=args.candidate_source_root
        )
        return recovery.RECOVERY_PROPOSAL_PATH, data

    if args.command == "subject":
        proposal_pin = _pin(
            recovery.RECOVERY_PROPOSAL_PATH,
            args.proposal_sha256,
            args.proposal_size_bytes,
        )
        value = recovery.build_phase6_recovery_review_subject(
            proposal_pin=proposal_pin,
            candidate_source_root=args.candidate_source_root,
            registered_at_utc=args.registered_at_utc,
        )
        data = recovery.canonical_json_bytes(value)
        recovery.validate_phase6_recovery_review_subject_bytes(
            data, source_root=args.candidate_source_root
        )
        return recovery.RECOVERY_REVIEW_SUBJECT_PATH, data

    if args.command == "review":
        subject_pin = _pin(
            recovery.RECOVERY_REVIEW_SUBJECT_PATH,
            args.review_subject_sha256,
            args.review_subject_size_bytes,
        )
        value = recovery.build_phase6_recovery_review_receipt(
            slot=args.slot,
            review_subject_pin=subject_pin,
            candidate_source_root=args.candidate_source_root,
            reviewer_identity_claim={
                "identity": args.reviewer_identity,
                "identity_namespace": args.reviewer_identity_namespace,
            },
            session_or_task_id=args.session_or_task_id,
            registered_at_utc=args.registered_at_utc,
        )
        data = recovery.canonical_json_bytes(value)
        _subject_observed, subject_data = recovery._stable_registered_pin(
            subject_pin, "recovery review subject"
        )
        subject, proposal = recovery.validate_phase6_recovery_review_subject_bytes(
            subject_data, source_root=args.candidate_source_root
        )
        recovery.validate_phase6_recovery_review_receipt_bytes(
            data,
            slot=args.slot,
            expected_review_subject_pin=subject_pin,
            review_subject=subject,
            proposal=proposal,
        )
        return recovery.RECOVERY_REVIEW_RECEIPT_PATHS[args.slot], data

    if args.command == "final":
        subject_pin = _pin(
            recovery.RECOVERY_REVIEW_SUBJECT_PATH,
            args.review_subject_sha256,
            args.review_subject_size_bytes,
        )
        review_pins = {
            "A": _pin(
                recovery.RECOVERY_REVIEW_RECEIPT_PATHS["A"],
                args.review_a_sha256,
                args.review_a_size_bytes,
            ),
            "B": _pin(
                recovery.RECOVERY_REVIEW_RECEIPT_PATHS["B"],
                args.review_b_sha256,
                args.review_b_size_bytes,
            ),
        }
        value = recovery.build_phase6_planrow_recovery_authority(
            review_subject_pin=subject_pin,
            review_pins=review_pins,
            candidate_source_root=args.candidate_source_root,
            registered_at_utc=args.registered_at_utc,
        )
        data = recovery.canonical_json_bytes(value)
        recovery.validate_phase6_planrow_recovery_authority_prerender_bytes(
            data,
            expected_authority_sha256=sha256(data).hexdigest(),
            candidate_source_root=args.candidate_source_root,
        )
        return recovery.PLANROW_RECOVERY_AUTHORITY_PATH, data

    if args.command == "render-plan":
        authority_pin = _pin(
            recovery.PLANROW_RECOVERY_AUTHORITY_PATH,
            args.authority_sha256,
            args.authority_size_bytes,
        )
        value = recovery.build_phase6_recovery_source_render_plan(
            authority_pin=authority_pin,
            candidate_source_root=args.candidate_source_root,
            target_repository_root=args.target_repository_root,
            registered_at_utc=args.registered_at_utc,
        )
        data = recovery.canonical_json_bytes(value)
        recovery.validate_phase6_recovery_source_render_plan_bytes(
            data,
            candidate_source_root=args.candidate_source_root,
            expected_authority_sha256=args.authority_sha256,
        )
        return recovery.RECOVERY_RENDER_PLAN_PATH, data
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        _assert_cpu_only_runtime()
        args = _parser().parse_args(argv)
        expected_path, data = _build(args)
        _assert_cpu_only_runtime()
        digest = _publish_exact(args.output, expected_path, data)
    except (
        FileExistsError,
        GovernanceCliError,
        OSError,
        TypeError,
        ValueError,
        recovery.Phase6RecoveryAuthorityError,
    ) as exc:
        print(f"Phase-6 recovery governance failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"PASS command={args.command} output={args.output} sha256={digest} size={len(data)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
