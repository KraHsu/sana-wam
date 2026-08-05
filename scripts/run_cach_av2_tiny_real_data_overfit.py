#!/usr/bin/env python3
"""Fail-closed launcher scaffold for the CACH-A4 AV-2 tiny-data screen.

This revision deliberately implements no dataset, CUDA, vendor-operator,
optimizer, or result-root capability.  ``--static-preflight`` is limited to
the additive AV-2 contract source and JSON config.  The ordinary path must
clear the source-owned authority gate before any future execution capability
can be reached.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Final, Mapping, Sequence


REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE: Final = (
    REPO_ROOT / "src/sana_wam/model/cach_av2_tiny_real_data_overfit.py"
)
DEFAULT_CONFIG: Final = (
    REPO_ROOT / "configs/experiments/cach_av2_tiny_real_data_overfit.yaml"
)
EXPECTED_SOURCE_SHA256: Final = (
    "510e85e9c060cefae9db04bd4b64916994b9ae0dd733af21473d3f97d6a5c1cf"
)
EXPECTED_CONFIG_SHA256: Final = (
    "de97f6a2b214c95be0de5999f5b5451d7d7ce5308eaa5f7d0e8b87a378ab7642"
)
_CONTRACT_MODULE_NAME: Final = "_cach_av2_tiny_real_data_overfit_contract"


class LauncherPreflightError(RuntimeError):
    """Typed launcher-scaffold failure with no execution side effects."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _strict_json_mapping(payload: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise LauncherPreflightError(
            "CONFIG_INVALID", f"strict JSON parse failed for {name}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise LauncherPreflightError(
            "CONFIG_INVALID", f"{name} must contain one JSON object"
        )
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _require_canonical_paths(*, config_path: Path, source_path: Path) -> None:
    """Reject path drift without resolving, reading, or importing either path."""

    if config_path != DEFAULT_CONFIG or source_path != DEFAULT_SOURCE:
        raise LauncherPreflightError(
            "PATH_REJECTED",
            "AV-2 scaffold accepts only the fixed default config/source paths",
        )


def _load_contract_source(path: Path) -> tuple[ModuleType, bytes]:
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise LauncherPreflightError(
            "SOURCE_INVALID", f"cannot read AV-2 contract source: {exc}"
        ) from exc
    if _sha256(source_bytes) != EXPECTED_SOURCE_SHA256:
        raise LauncherPreflightError(
            "SOURCE_SHA_MISMATCH",
            "AV-2 contract source bytes differ from the fixed SHA256",
        )
    spec = importlib.util.spec_from_file_location(_CONTRACT_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise LauncherPreflightError(
            "SOURCE_INVALID", "cannot construct AV-2 contract source loader"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CONTRACT_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(_CONTRACT_MODULE_NAME, None)
        raise LauncherPreflightError(
            "SOURCE_INVALID", f"cannot import AV-2 contract source: {exc}"
        ) from exc
    return module, source_bytes


def _load_and_validate_static_contract(
    *, config_path: Path, source_path: Path
) -> tuple[ModuleType, Mapping[str, Any], Mapping[str, Any]]:
    _require_canonical_paths(config_path=config_path, source_path=source_path)
    contract, source_bytes = _load_contract_source(source_path)
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise LauncherPreflightError(
            "CONFIG_INVALID", f"cannot read AV-2 config: {exc}"
        ) from exc
    if _sha256(config_bytes) != EXPECTED_CONFIG_SHA256:
        raise LauncherPreflightError(
            "CONFIG_SHA_MISMATCH",
            "AV-2 config bytes differ from the fixed SHA256",
        )
    config = _strict_json_mapping(config_bytes, name=str(config_path))
    validator = getattr(contract, "validate_av2_static_config", None)
    if not callable(validator):
        raise LauncherPreflightError(
            "SOURCE_INVALID", "AV-2 source lacks validate_av2_static_config"
        )
    try:
        validated = validator(config)
    except Exception as exc:
        raise LauncherPreflightError(
            "CONFIG_INVALID", f"AV-2 static schema validation failed: {exc}"
        ) from exc
    schema = getattr(contract, "AV2_CONFIG_SCHEMA", None)
    if not isinstance(schema, str) or not schema:
        raise LauncherPreflightError(
            "SOURCE_INVALID", "AV-2 source lacks a non-empty AV2_CONFIG_SCHEMA"
        )
    summary: dict[str, Any] = {
        "config_path": _display_path(config_path),
        "config_sha256": _sha256(config_bytes),
        "schema": schema,
        "source_path": _display_path(source_path),
        "source_sha256": _sha256(source_bytes),
        "validated_config_type": type(validated).__name__,
    }
    return contract, config, summary


def static_preflight(
    *, config_path: Path = DEFAULT_CONFIG, source_path: Path = DEFAULT_SOURCE
) -> Mapping[str, Any]:
    """Validate only the AV-2 config/source/schema bytes."""

    _, _, summary = _load_and_validate_static_contract(
        config_path=config_path,
        source_path=source_path,
    )
    return {"state": "STATIC_PREFLIGHT_OK", **summary}


def _require_execution_authority(
    contract: ModuleType, config: Mapping[str, Any]
) -> None:
    gate_type = getattr(contract, "AV2AuthorityGate", None)
    if gate_type is None or not callable(getattr(gate_type, "from_mapping", None)):
        raise LauncherPreflightError(
            "SOURCE_INVALID", "AV-2 source lacks AV2AuthorityGate.from_mapping"
        )
    authority = config.get("authority")
    if not isinstance(authority, Mapping):
        raise LauncherPreflightError(
            "CONFIG_INVALID", "AV-2 config authority must be a mapping"
        )
    try:
        gate = gate_type.from_mapping(authority)
        gate.require_execution_authority()
    except Exception as exc:
        blocked_type = getattr(contract, "AV2AuthorityBlocked", ())
        is_typed_block = isinstance(blocked_type, type) and isinstance(
            exc, blocked_type
        )
        is_coded_block = getattr(exc, "code", None) == "AUTH_BLOCKED"
        if is_typed_block or is_coded_block or "AUTH_BLOCKED" in str(exc):
            raise LauncherPreflightError("AUTH_BLOCKED", str(exc)) from exc
        raise LauncherPreflightError(
            "CONFIG_INVALID", f"AV-2 authority gate is invalid: {exc}"
        ) from exc


# These capability seams are intentionally unreachable while the frozen config
# remains pending.  Keeping them as distinct callables lets the CPU test prove
# that AUTH_BLOCKED happens before dataset/GPU/vendor/root access.
def _access_dataset_after_authority() -> None:
    raise LauncherPreflightError(
        "EXECUTION_NOT_IMPLEMENTED", "AV-2 dataset access is not implemented"
    )


def _probe_gpu_after_authority() -> None:
    raise LauncherPreflightError(
        "EXECUTION_NOT_IMPLEMENTED", "AV-2 GPU access is not implemented"
    )


def _import_vendor_after_authority() -> None:
    raise LauncherPreflightError(
        "EXECUTION_NOT_IMPLEMENTED", "AV-2 vendor import is not implemented"
    )


def _create_root_after_authority() -> None:
    raise LauncherPreflightError(
        "EXECUTION_NOT_IMPLEMENTED", "AV-2 root creation is not implemented"
    )


def _authorized_execution_not_implemented() -> None:
    """Future execution boundary; no capability is implemented this phase."""

    _access_dataset_after_authority()
    _probe_gpu_after_authority()
    _import_vendor_after_authority()
    _create_root_after_authority()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--static-preflight",
        action="store_true",
        help="validate only config/source/schema; execute no capability",
    )
    return parser


def _emit(value: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        contract, config, summary = _load_and_validate_static_contract(
            config_path=args.config,
            source_path=args.source,
        )
        if args.static_preflight:
            _emit({"state": "STATIC_PREFLIGHT_OK", **summary})
            return 0
        _require_execution_authority(contract, config)
        _authorized_execution_not_implemented()
    except LauncherPreflightError as exc:
        _emit({"error": f"{type(exc).__name__}: {exc}", "state": exc.code})
        return 2
    raise AssertionError("unreachable AV-2 launcher scaffold state")


if __name__ == "__main__":
    raise SystemExit(main())
