#!/usr/bin/env python3
"""Fresh-root AV-2 R2 runner with a mechanical TorchVersion JSON repair.

R2 preserves the frozen data/model/config/test and the R1 idle-probe repair.
It uses a fresh namespace/root/nonce and fresh model initialization.  The only
result-path change is converting PyTorch 2.7's ``TorchVersion`` to an exact
built-in string before canonical JSON encoding.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib.util
from pathlib import Path
import signal
import sys
from types import ModuleType
from typing import Any


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
R1_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_av2_tiny_real_data_overfit_execution_r1.py"
R2_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_av2_tiny_real_data_overfit_execution_r2.py"
R2_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av2_execution/"
    "CACH_A4_AV2_EXECUTION_R2_RUN_CARD.json"
)
R2_SCHEMA = "cach.architecture_validation.cach_a4_av2_execution_r2_run_card.v1"
R2_STATE = "AUTHORIZED_FRESH_ROOT_TORCHVERSION_SERIALIZATION_REPAIR"
R2_AUTHORITY_STATEMENT = "授权 fresh-root 修复重跑"
R2_AUTHORITY_SHA256 = (
    "4be4e13b959d7079dd3e4ea2b7c84f5b59c11961523c5c6e9be745edcd69e1d3"
)
R2_NONCE = "b795b1b6edba6aeb584c06db1d166bf7"
R2_NAMESPACE = Path(
    "/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_av2_tiny_real_data_r2"
)
R2_NAMESPACE_LEAF = R2_NAMESPACE / "06f5d09127f8"
R2_ROOT = R2_NAMESPACE_LEAF / f"cach-a4-av2-r2-{R2_NONCE}"


def _load_r1() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_av2_execution_r1_runner", R1_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen AV-2 R1 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R1 = _load_r1()
BASE = R1.BASE


def _verify_r2_card() -> Mapping[str, object]:
    card = BASE._strict_json(R2_CARD_PATH)
    if card.get("schema") != R2_SCHEMA or card.get("state") != R2_STATE:
        raise BASE.PreflightError("AV-2 R2 card identity differs")
    authority = card.get("authority")
    execution = card.get("execution")
    if not isinstance(authority, Mapping) or not isinstance(execution, Mapping):
        raise BASE.PreflightError("AV-2 R2 authority/execution mapping is absent")
    if (
        authority.get("statement") != R2_AUTHORITY_STATEMENT
        or authority.get("statement_utf8_bytes") != 30
        or authority.get("statement_sha256") != R2_AUTHORITY_SHA256
        or authority.get("interpretation")
        != "FRESH_ROOT_FRESH_INITIALIZATION_SERIALIZATION_REPAIR_RERUN"
        or execution.get("root") != str(R2_ROOT)
        or execution.get("namespace") != str(R2_NAMESPACE)
        or execution.get("nonce") != R2_NONCE
        or execution.get("gpu_uuid") != BASE.EXPECTED_GPU_UUID
        or execution.get("max_steps_per_arm") != 1000
        or execution.get("checkpoint_load") is not False
        or execution.get("checkpoint_save") is not False
        or execution.get("root_reuse") is not False
        or execution.get("automatic_additional_rerun") is not False
    ):
        raise BASE.PreflightError("AV-2 R2 authority or execution identity differs")
    pins = card.get("materialized_pins")
    expected = {
        "r1_card": R1.R1_CARD_PATH,
        "r1_runner": R1_RUNNER_PATH,
        "r2_runner": R2_RUNNER_PATH,
        "failed_result": Path(card["failed_root"]["result_path"]),
        "failed_freeze_receipt": Path(card["failed_root"]["freeze_receipt_path"]),
        "failed_progress": Path(card["failed_root"]["progress_path"]),
        "failed_run_context": Path(card["failed_root"]["run_context_path"]),
    }
    if not isinstance(pins, Mapping) or set(pins) != set(expected):
        raise BASE.PreflightError("AV-2 R2 materialized pin roles differ")
    for role, path in expected.items():
        BASE._verify_pin(pins[role], expected_path=path, label=f"r2.materialized_pins.{role}")
    direct = card.get("direct_frozen_pins")
    direct_expected = {
        "base_card": BASE.EXPECTED_CARD,
        "base_runner": BASE.EXPECTED_RUNNER,
        "data_manifest": BASE.EXPECTED_DATA_MANIFEST,
        "model_source": BASE.EXPECTED_SOURCE,
        "config": BASE.EXPECTED_CONFIG,
        "test": BASE.EXPECTED_TEST,
    }
    if not isinstance(direct, Mapping) or set(direct) != set(direct_expected):
        raise BASE.PreflightError("AV-2 R2 direct frozen pin roles differ")
    for role, path in direct_expected.items():
        BASE._verify_pin(
            direct[role], expected_path=path, label=f"r2.direct_frozen_pins.{role}"
        )
    repair = card.get("repair_scope")
    if (
        not isinstance(repair, Mapping)
        or repair.get("result_path_code_change")
        != "CANONICAL_JSON_CONVERT_TORCHVERSION_TO_EXACT_STR"
        or repair.get("root_lifecycle_code_change")
        != "HOLD_LIFECYCLE_AND_INSTALL_SIGNAL_HANDLERS_BEFORE_ROOT_CREATE"
    ):
        raise BASE.PreflightError("AV-2 R2 repair scope differs")
    return card


def _r2_static_preflight() -> tuple[Mapping[str, object], Mapping[str, object]]:
    card = _verify_r2_card()
    _r1_card, predecessor = R1._r1_static_preflight()
    return card, predecessor


def _install_r2_runtime_bindings() -> None:
    """Patch only fresh-root constants and canonical TorchVersion handling."""

    predecessor_jsonable = BASE._jsonable

    def r2_jsonable(value: object) -> object:
        value_type = type(value)
        if (
            isinstance(value, str)
            and value_type.__module__ == "torch.torch_version"
            and value_type.__name__ == "TorchVersion"
        ):
            return str(value)
        return predecessor_jsonable(value)

    BASE._jsonable = r2_jsonable
    BASE.EXPECTED_NAMESPACE = R2_NAMESPACE
    BASE.EXPECTED_NAMESPACE_LEAF = R2_NAMESPACE_LEAF
    BASE.EXPECTED_NONCE = R2_NONCE
    BASE.EXPECTED_ROOT = R2_ROOT
    BASE.EXPECTED_ENV = {
        "CUDA_VISIBLE_DEVICES": BASE.EXPECTED_GPU_UUID,
        "FUSED_GDN_PRECISION": "0",
        "GDN_DISABLE_COMPILE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TORCHDYNAMO_DISABLE": "1",
        "TRITON_CACHE_DIR": str(R2_ROOT / "triton_cache"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lifecycle: Any = None
    previous_handlers: dict[int, object] = {}

    def interrupted(signum: int, _frame: object) -> None:
        raise BASE.RunnerInterrupted(f"received signal {signum}")

    try:
        r2_card, preflight = _r2_static_preflight()
        _install_r2_runtime_bindings()
        if args.static_preflight:
            print(
                BASE._canonical(
                    {
                        "schema": R2_SCHEMA,
                        "state": "STATIC_PREFLIGHT_OK",
                        "r2_card_sha256": BASE._sha(R2_CARD_PATH),
                        "r1_card_sha256": BASE._sha(R1.R1_CARD_PATH),
                        "r2_runner_sha256": BASE._sha(R2_RUNNER_PATH),
                        "root": str(R2_ROOT),
                    }
                ).decode("utf-8"),
                end="",
            )
            return 0
        BASE._check_exact_env()
        source, bundle, adequacy = BASE._load_real_data_after_preflight(preflight)
        pre_root_gpu = R1._relaxed_idle_probe()
        lifecycle = BASE.RootLifecycle()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        lifecycle.create()
        post_root_gpu = R1._relaxed_idle_probe()
        lifecycle.publish(
            "RUN_CONTEXT.json",
            {
                "schema": "cach.cach_a4.av2_tiny_real_data.run_context.v1",
                "launcher_revision": "R2_TORCHVERSION_SERIALIZATION_REPAIR",
                "root": str(R2_ROOT),
                "nonce": R2_NONCE,
                "gpu": post_root_gpu,
                "pre_root_gpu": pre_root_gpu,
                "fresh_authority_sha256": R2_AUTHORITY_SHA256,
                "r2_card_sha256": BASE._sha(R2_CARD_PATH),
                "r1_card_sha256": BASE._sha(R1.R1_CARD_PATH),
                "data_manifest_sha256": BASE._sha(BASE.EXPECTED_DATA_MANIFEST),
                "materialized_source_sha256": BASE._sha(BASE.EXPECTED_SOURCE),
                "config_sha256": BASE._sha(BASE.EXPECTED_CONFIG),
                "r1_runner_sha256": BASE._sha(R1_RUNNER_PATH),
                "r2_runner_sha256": BASE._sha(R2_RUNNER_PATH),
                "test_sha256": BASE._sha(BASE.EXPECTED_TEST),
                "exact_env": dict(BASE.EXPECTED_ENV),
                "repair_scope": r2_card["repair_scope"],
                "fresh_initialization": True,
                "checkpoint_load": False,
                "checkpoint_save": False,
                "formal_admission": False,
                "automatic_additional_rerun": False,
            },
        )
        progress: list[object] = [
            {"event": "post_root_pre_cuda_gpu_idle_snapshot", "payload": post_root_gpu}
        ]
        screen = BASE._execute_after_preflight(
            source,
            preflight["config"],
            bundle,
            progress,
        )
        # Prove the complete result is canonical-JSON encodable before the
        # first attempt to publish it; this specifically exercises TorchVersion.
        screen_canonical_sha256 = BASE._sha_bytes(BASE._canonical(screen))
        lifecycle.publish("DATA_EVIDENCE.json", {"adequacy": adequacy})
        lifecycle.publish("PROGRESS_EVENTS.json", progress)
        lifecycle.publish("SCREEN_RESULT.json", screen)
        jit = BASE._jit_inventory(R2_ROOT)
        lifecycle.publish("JIT_INVENTORY.json", jit)
        validity = screen.get("validity") if isinstance(screen, Mapping) else None
        verdict = screen.get("verdict") if isinstance(screen, Mapping) else None
        reason = screen.get("reason_code") if isinstance(screen, Mapping) else None
        if not isinstance(validity, Mapping) or not isinstance(verdict, str):
            raise RuntimeError("execution core returned malformed result")
        terminal = {
            "schema": BASE.RESULT_SCHEMA,
            "terminal_state": verdict,
            "typed_verdict": verdict,
            "reason_code": reason,
            "valid_result": bool(validity.get("run_valid")),
            "av3_unlocked": verdict == "AV2_GO" and bool(validity.get("run_valid")),
            "launcher_revision": "R2_TORCHVERSION_SERIALIZATION_REPAIR",
            "root": str(R2_ROOT),
            "nonce": R2_NONCE,
            "gpu_uuid": BASE.EXPECTED_GPU_UUID,
            "screen_result_sha256": lifecycle.published["SCREEN_RESULT.json"],
            "screen_canonical_sha256_prepublication": screen_canonical_sha256,
            "jit_inventory_sha256": lifecycle.published["JIT_INVENTORY.json"],
            "automatic_rerun_allowed": False,
            "checkpoint_load_or_save": False,
            "formal_claim_allowed": False,
        }
        result_sha = lifecycle.publish("RESULT.json", terminal)
        lifecycle.freeze(terminal_state=verdict, result_sha256=result_sha)
        print(BASE._canonical(terminal).decode("utf-8"), end="")
        return 0 if bool(terminal["valid_result"]) else 2
    except BaseException as exc:
        if lifecycle is not None:
            lifecycle.freeze_failure(exc)
        print(
            BASE._canonical(
                {
                    "schema": BASE.RESULT_SCHEMA,
                    "state": "AV2_INVALID_FAIL_CLOSED",
                    "launcher_revision": "R2_TORCHVERSION_SERIALIZATION_REPAIR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "root_created": bool(lifecycle and lifecycle.created),
                    "automatic_rerun_allowed": False,
                }
            ).decode("utf-8"),
            end="",
        )
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
