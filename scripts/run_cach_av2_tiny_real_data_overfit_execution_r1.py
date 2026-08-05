#!/usr/bin/env python3
"""Additive pre-root idle-probe repair for the frozen AV-2 runner.

The predecessor runner rejected PyTorch's own 4 MiB driver bookkeeping after
CPU data loading despite zero utilization and no compute application.  This
wrapper preserves every predecessor pin, data byte, model path, budget, root,
and execution function.  Its only behavioral change is an idle threshold of
at most 8 MiB while still requiring zero utilization and no compute app.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib.util
import os
from pathlib import Path
import signal
import sys
from types import ModuleType
from typing import Any


REPO_ROOT = Path("/home/zch/workspace/sana-wam")
BASE_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_av2_tiny_real_data_overfit_execution.py"
R1_RUNNER_PATH = REPO_ROOT / "scripts/run_cach_av2_tiny_real_data_overfit_execution_r1.py"
R1_CARD_PATH = REPO_ROOT / (
    "docs/cach_sana_wam/architecture_validation/cach_a4_av2_execution/"
    "CACH_A4_AV2_EXECUTION_R1_RUN_CARD.json"
)
R1_SCHEMA = "cach.architecture_validation.cach_a4_av2_execution_r1_run_card.v1"
R1_STATE = "AUTHORIZED_PRE_ROOT_IDLE_PROBE_REPAIR"
MAX_SELF_DRIVER_MEMORY_MIB = 8


def _load_base() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_av2_execution_base_runner", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen AV-2 predecessor runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def _verify_r1_card() -> Mapping[str, object]:
    card = BASE._strict_json(R1_CARD_PATH)
    if card.get("schema") != R1_SCHEMA or card.get("state") != R1_STATE:
        raise BASE.PreflightError("AV-2 R1 card identity differs")
    authority = card.get("authority")
    execution = card.get("execution")
    if not isinstance(authority, Mapping) or not isinstance(execution, Mapping):
        raise BASE.PreflightError("AV-2 R1 authority/execution mapping is absent")
    if (
        authority.get("statement_sha256") != BASE.AUTHORITY_SHA256
        or authority.get("interpretation")
        != "CONTINUE_AUTHORIZED_AV2_AFTER_PRE_ROOT_NO_MODEL_LAUNCHER_FAILURE"
        or execution.get("root") != str(BASE.EXPECTED_ROOT)
        or execution.get("nonce") != BASE.EXPECTED_NONCE
        or execution.get("gpu_uuid") != BASE.EXPECTED_GPU_UUID
        or execution.get("max_idle_driver_memory_mib") != MAX_SELF_DRIVER_MEMORY_MIB
        or execution.get("automatic_model_rerun") is not False
    ):
        raise BASE.PreflightError("AV-2 R1 authority or repair scope differs")
    pins = card.get("materialized_pins")
    expected = {
        "predecessor_card": BASE.EXPECTED_CARD,
        "predecessor_runner": BASE_RUNNER_PATH,
        "r1_runner": R1_RUNNER_PATH,
    }
    if not isinstance(pins, Mapping) or set(pins) != set(expected):
        raise BASE.PreflightError("AV-2 R1 materialized pin roles differ")
    for role, path in expected.items():
        BASE._verify_pin(pins[role], expected_path=path, label=f"r1.materialized_pins.{role}")
    return card


def _r1_static_preflight() -> tuple[Mapping[str, object], Mapping[str, object]]:
    card = _verify_r1_card()
    predecessor = BASE._static_preflight()
    return card, predecessor


def _relaxed_idle_probe() -> Mapping[str, object]:
    matches = [
        row for row in BASE._nvidia_rows()
        if row["index"] == BASE.EXPECTED_GPU_INDEX
    ]
    if len(matches) != 1 or matches[0]["uuid"] != BASE.EXPECTED_GPU_UUID:
        raise RuntimeError("physical GPU identity differs from AV-2 binding")
    apps = [
        row for row in BASE._compute_apps()
        if row["gpu_uuid"] == BASE.EXPECTED_GPU_UUID
    ]
    row = matches[0]
    if (
        row["memory_used_mib"] > MAX_SELF_DRIVER_MEMORY_MIB
        or row["utilization_gpu_percent"] != 0
        or apps
    ):
        raise RuntimeError(f"reserved GPU is not idle under R1: gpu={row}, apps={apps}")
    return {
        **row,
        "compute_apps": apps,
        "captured_unix_ns": __import__("time").time_ns(),
        "idle_probe_revision": "ALLOW_SELF_DRIVER_MEMORY_LE_8_MIB_NO_APPS_ZERO_UTIL",
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
        r1_card, preflight = _r1_static_preflight()
        if args.static_preflight:
            print(
                BASE._canonical(
                    {
                        "schema": R1_SCHEMA,
                        "state": "STATIC_PREFLIGHT_OK",
                        "r1_card_sha256": BASE._sha(R1_CARD_PATH),
                        "predecessor_card_sha256": BASE._sha(BASE.EXPECTED_CARD),
                        "r1_runner_sha256": BASE._sha(R1_RUNNER_PATH),
                    }
                ).decode("utf-8"),
                end="",
            )
            return 0
        BASE._check_exact_env()
        source, bundle, adequacy = BASE._load_real_data_after_preflight(preflight)
        pre_root_gpu = _relaxed_idle_probe()
        lifecycle = BASE._create_root_after_preflight()
        post_root_gpu = _relaxed_idle_probe()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        lifecycle.publish(
            "RUN_CONTEXT.json",
            {
                "schema": "cach.cach_a4.av2_tiny_real_data.run_context.v1",
                "launcher_revision": "R1_PRE_ROOT_IDLE_PROBE_REPAIR",
                "root": str(BASE.EXPECTED_ROOT),
                "nonce": BASE.EXPECTED_NONCE,
                "gpu": post_root_gpu,
                "pre_root_gpu": pre_root_gpu,
                "authority_sha256": BASE.AUTHORITY_SHA256,
                "r1_card_sha256": BASE._sha(R1_CARD_PATH),
                "predecessor_card_sha256": BASE._sha(BASE.EXPECTED_CARD),
                "data_manifest_sha256": BASE._sha(BASE.EXPECTED_DATA_MANIFEST),
                "materialized_source_sha256": BASE._sha(BASE.EXPECTED_SOURCE),
                "config_sha256": BASE._sha(BASE.EXPECTED_CONFIG),
                "predecessor_runner_sha256": BASE._sha(BASE_RUNNER_PATH),
                "r1_runner_sha256": BASE._sha(R1_RUNNER_PATH),
                "test_sha256": BASE._sha(BASE.EXPECTED_TEST),
                "exact_env": dict(BASE.EXPECTED_ENV),
                "repair_scope": r1_card["repair_scope"],
                "checkpoint_load": False,
                "checkpoint_save": False,
                "formal_admission": False,
                "automatic_model_rerun": False,
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
        lifecycle.publish("DATA_EVIDENCE.json", {"adequacy": adequacy})
        lifecycle.publish("PROGRESS_EVENTS.json", progress)
        lifecycle.publish("SCREEN_RESULT.json", screen)
        jit = BASE._jit_inventory(BASE.EXPECTED_ROOT)
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
            "launcher_revision": "R1_PRE_ROOT_IDLE_PROBE_REPAIR",
            "root": str(BASE.EXPECTED_ROOT),
            "nonce": BASE.EXPECTED_NONCE,
            "gpu_uuid": BASE.EXPECTED_GPU_UUID,
            "screen_result_sha256": lifecycle.published["SCREEN_RESULT.json"],
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
                    "launcher_revision": "R1_PRE_ROOT_IDLE_PROBE_REPAIR",
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
