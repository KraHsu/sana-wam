#!/usr/bin/env python3
"""Materialize Phase-6 metadata stages without importing torch or using a GPU."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wam.train.phase6_artifact_dag import (  # noqa: E402
    Phase6ArtifactDagError,
    build_artifact_dag_plan,
    canonical_json_bytes,
    materialize_after_source_freeze,
    materialize_final_sink,
    materialize_reference_inputs,
    materialize_smoke_request,
    materialize_training_pins,
    materialize_training_requests,
)
from sana_wam.train.phase6_recovery import recovery_lifecycle  # noqa: E402


_NODE_BY_COMMAND = {
    "after-source-freeze": "after_source_freeze",
    "reference-inputs": "reference_inputs",
    "reference-rebind": "reference_numeric_cpu_rebind",
    "smoke-request": "smoke_request",
    "smoke-rebind": "smoke_numeric_cpu_rebind",
    "training-requests": "training_requests",
    "training-pins": "materialization_pins",
    "final-sink": "final_sink",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the immutable, acyclic Phase-6 artifact DAG. This CLI never "
            "runs CUDA, a model, training, or closed-loop evaluation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="print the complete deterministic DAG")
    for command in _NODE_BY_COMMAND:
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help="print this stage and do not inspect or write artifacts",
        )
        if command == "final-sink":
            subparser.add_argument(
                "--materialization-pins-sha256",
                help="externally recorded SHA256 from the training-pins stage",
            )
        if command in {"reference-rebind", "smoke-rebind"}:
            subparser.add_argument("--source-manifest-v9")
            subparser.add_argument("--source-manifest-v9-sha256")
            subparser.add_argument("--source-manifest-v9-size", type=int)
            subparser.add_argument("--recovery-authority")
            subparser.add_argument("--recovery-authority-sha256")
            subparser.add_argument("--recovery-authority-size", type=int)
            subparser.add_argument("--created-at-utc")
            if command == "smoke-rebind":
                subparser.add_argument("--reference-receipt-sha256")
        if command == "training-requests":
            subparser.add_argument("--evidence-rebinding-receipt-sha256")
    return parser


def _print(value) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def _dry_run(command: str) -> dict:
    plan = build_artifact_dag_plan()
    node_id = _NODE_BY_COMMAND[command]
    node = next(item for item in plan["nodes"] if item["id"] == node_id)
    return {
        "command": command,
        "dry_run": True,
        "node": node,
        "plan_sha256": sha256(canonical_json_bytes(plan)).hexdigest(),
        "writes_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            _print(build_artifact_dag_plan())
            return 0
        if args.dry_run:
            _print(_dry_run(args.command))
            return 0
        if args.command == "after-source-freeze":
            artifacts = materialize_after_source_freeze()
        elif args.command == "reference-inputs":
            artifacts = materialize_reference_inputs()
        elif args.command in {"reference-rebind", "smoke-rebind"}:
            required = (
                "source_manifest_v9",
                "source_manifest_v9_sha256",
                "source_manifest_v9_size",
                "recovery_authority",
                "recovery_authority_sha256",
                "recovery_authority_size",
                "created_at_utc",
            )
            missing = [name for name in required if getattr(args, name) is None]
            if args.command == "smoke-rebind" and not args.reference_receipt_sha256:
                missing.append("reference_receipt_sha256")
            if missing:
                raise Phase6ArtifactDagError(
                    f"{args.command} missing required CLI bindings: {sorted(missing)}"
                )
            from build_phase6_recovery_rebind import (  # noqa: PLC0415
                _reference_cli,
                _smoke_cli,
                require_cpu_environment,
            )

            require_cpu_environment()
            args.publish = True
            artifacts = (
                _reference_cli(args)
                if args.command == "reference-rebind"
                else _smoke_cli(args)
            )
        elif args.command == "smoke-request":
            artifacts = materialize_smoke_request()
        elif args.command == "training-requests":
            if not args.evidence_rebinding_receipt_sha256:
                raise Phase6ArtifactDagError(
                    "training-requests requires --evidence-rebinding-receipt-sha256"
                )
            artifacts = materialize_training_requests(
                evidence_rebinding_receipt_sha256=(
                    args.evidence_rebinding_receipt_sha256
                )
            )
        elif args.command == "training-pins":
            artifacts = materialize_training_pins()
        elif args.command == "final-sink":
            if not args.materialization_pins_sha256:
                raise Phase6ArtifactDagError(
                    "final-sink requires --materialization-pins-sha256"
                )
            artifacts = materialize_final_sink(
                materialization_pins_sha256=args.materialization_pins_sha256
            )
        else:
            raise AssertionError(f"unhandled command: {args.command}")
        _print(
            {
                "artifacts": artifacts,
                "command": args.command,
                **recovery_lifecycle(),
                "status": "materialized",
            }
        )
    except (
        FileExistsError,
        OSError,
        Phase6ArtifactDagError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Phase-6 artifact DAG failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
