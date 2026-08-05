"""Non-executable Stage-1 launcher contract for CACH-SANA-WAM.

The only constructive operation in this module is producing a deterministic
review plan.  Model construction, training, and deployment remain hard-denied
before any runtime/model seam is reachable.
"""

from __future__ import annotations

from typing import Any

from sana_wam.cach.authority import (
    CACHAuthorityError,
    CACH_VARIANT,
    reject_cach_deploy_before_runtime,
    reject_cach_model_build,
    reject_cach_training_before_runtime,
)
from sana_wam.cach.config import validate_stage1_config
from sana_wam.cach.verifier import build_stage1_static_report

_REVIEW_STEPS = (
    (
        "config_schema",
        "validate the exact Stage-1 config and disabled execution flags",
    ),
    (
        "fresh_initialization",
        "reject pretrained DiT and initial checkpoint coordinates",
    ),
    (
        "afcc_isolation",
        "reject AFCC, action-reference, Phase-6, and loss-subtraction state",
    ),
    (
        "synthetic_contract_tests",
        "review pure fake-tensor inventory and checkpoint-schema tests",
    ),
    (
        "real_inventory_blocker",
        "retain the real-model inventory and checkpoint schema as unverified",
    ),
)


def build_contract_review_plan(config: Any) -> dict[str, Any]:
    """Return the sole Stage-1 launcher product: a review-only plan."""

    root = validate_stage1_config(config)
    report = build_stage1_static_report(root)
    return {
        "schema": "cach.stage1_launcher_review_plan.v1",
        "variant": CACH_VARIANT,
        "stage": "stage1",
        "mode": "review_only",
        "allowed_actions": ["static_review", "pure_contract_tests"],
        "ordered_checks": [
            {
                "sequence": index,
                "check": check,
                "contract": contract,
            }
            for index, (check, contract) in enumerate(_REVIEW_STEPS, start=1)
        ],
        "model_build_authorized": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "evaluation_authorized": False,
        "run_root_creation_authorized": False,
        "worker_spawn_authorized": False,
        "execution_admission_valid": False,
        "scientific_eligible": False,
        "static_report": report,
    }


def refuse_stage1_operation(config: Any, operation: str) -> None:
    """Hard-deny model build, training, deploy, and every other runtime action."""

    root = validate_stage1_config(config)
    if operation == "model_build":
        reject_cach_model_build(root)
    if operation == "training":
        reject_cach_training_before_runtime(root)
    if operation == "deploy":
        reject_cach_deploy_before_runtime(root)
    raise CACHAuthorityError(
        "CACH Stage 1 launcher only produces a reviewable contract plan; "
        f"runtime operation is denied: {operation!r}"
    )


__all__ = [
    "build_contract_review_plan",
    "refuse_stage1_operation",
]
