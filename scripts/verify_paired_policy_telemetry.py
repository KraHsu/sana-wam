#!/usr/bin/env python
"""Validate a same-checkpoint execution-horizon paired RoboTwin evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

try:
    from scripts.verify_paired_model_telemetry import (
        _require_checkpoint_binding,
        _require_server_client_prompts,
        _require_treatment_manifest_bound_to_control,
    )
    from scripts.verify_paired_noise_telemetry import (
        _prefix_run,
        _require_generation_zero_equal,
        _require_server_client_mapping,
        load_client_run,
        load_run,
    )
except ModuleNotFoundError:  # Direct script execution.
    from verify_paired_model_telemetry import (
        _require_checkpoint_binding,
        _require_server_client_prompts,
        _require_treatment_manifest_bound_to_control,
    )
    from verify_paired_noise_telemetry import (
        _prefix_run,
        _require_generation_zero_equal,
        _require_server_client_mapping,
        load_client_run,
        load_run,
    )


_DEPLOY_CONFIG_IDENTITY_FIELDS = {
    "deploy_config_path",
    "deploy_config_size",
    "deploy_config_sha256",
}


def _identity_without_deploy_config(identity: dict) -> dict:
    return {
        key: value
        for key, value in identity.items()
        if key not in _DEPLOY_CONFIG_IDENTITY_FIELDS
    }


def _load_bound_config(identity: dict, *, label: str) -> dict:
    raw_path = identity.get("deploy_config_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise SystemExit(f"{label} deploy config path is invalid")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise SystemExit(f"{label} deploy config is missing: {path}")
    encoded = path.read_bytes()
    if identity.get("deploy_config_size") != len(encoded):
        raise SystemExit(f"{label} deploy config size does not match telemetry")
    digest = hashlib.sha256(encoded).hexdigest()
    if identity.get("deploy_config_sha256") != digest:
        raise SystemExit(f"{label} deploy config SHA-256 does not match telemetry")
    try:
        config = yaml.safe_load(encoded)
    except yaml.YAMLError as exc:
        raise SystemExit(f"{label} deploy config is invalid: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"{label} deploy config is not a mapping")
    return config


def _require_info_binding(
    run_path: Path,
    identity: dict,
    *,
    label: str,
    expected_horizon: int | None,
) -> None:
    for filename in ("info.before.json", "info.after.json"):
        path = run_path / filename
        if not path.is_file():
            raise SystemExit(f"{label} is missing {filename}")
        info = json.loads(path.read_text())
        runtime_identity = info.get("inference_runtime", {}).get(
            "deployment_identity"
        )
        if runtime_identity != identity:
            raise SystemExit(f"{label} {filename} identity does not match telemetry")
        policy = info.get("policy_config", {})
        if policy.get("execute_horizon") != expected_horizon:
            raise SystemExit(
                f"{label} {filename} execute_horizon mismatch: "
                f"{policy.get('execute_horizon')!r} != {expected_horizon!r}"
            )
        if policy.get("temporal_ensemble") is not True:
            raise SystemExit(f"{label} {filename} temporal_ensemble is not enabled")
        expected_ensemble = expected_horizon is not None
        effective = info.get("async_inference", {}).get(
            "effective_temporal_ensemble"
        )
        if effective is not expected_ensemble:
            raise SystemExit(
                f"{label} {filename} effective_temporal_ensemble mismatch"
            )


def _require_generation_cadence(run: dict, *, label: str, horizon: int) -> None:
    for episode_index, (steps, generations) in enumerate(
        zip(run["steps"], run["generations"])
    ):
        expected_requests = list(range(1, len(steps) + 1, horizon))
        actual_requests = [generation["request_id"] for generation in generations]
        if actual_requests != expected_requests:
            raise SystemExit(
                f"{label} generation cadence mismatch at episode {episode_index}: "
                f"{actual_requests} != {expected_requests}"
            )
        for offset, step in enumerate(steps):
            policy = step["record"].get("policy", {})
            expected_generated = offset % horizon == 0
            if policy.get("generated") is not expected_generated:
                raise SystemExit(
                    f"{label} generated flag mismatch at episode {episode_index} "
                    f"request={offset + 1}"
                )
            expected_chunk_offset = offset % horizon
            if policy.get("chunk_offset") != expected_chunk_offset:
                raise SystemExit(
                    f"{label} chunk offset mismatch at episode {episode_index} "
                    f"request={offset + 1}"
                )


def verify_policy_pair(
    control_path: Path,
    treatment_path: Path,
    *,
    expected_episodes: int,
    expected_treatment_horizon: int,
    allow_control_prefix: bool = False,
    expected_prompt_manifest_sha256: str | None = None,
) -> dict:
    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive")
    if expected_treatment_horizon < 1:
        raise ValueError("expected treatment horizon must be positive")

    control = load_run(
        control_path,
        expected_mode="predicted",
        expected_episodes=None if allow_control_prefix else expected_episodes,
    )
    control_client = load_client_run(
        control_path,
        expected_episodes=None if allow_control_prefix else expected_episodes,
    )
    if allow_control_prefix:
        control = _prefix_run(control, expected_episodes)
        control_client = _prefix_run(control_client, expected_episodes)
    treatment = load_run(
        treatment_path,
        expected_mode="predicted",
        expected_episodes=expected_episodes,
    )
    treatment_client = load_client_run(
        treatment_path, expected_episodes=expected_episodes
    )

    if control["pairs"] != treatment["pairs"]:
        raise SystemExit("ordered scene/model-noise pairs do not match")
    control_identity = control["identity"]
    treatment_identity = treatment["identity"]
    if control_identity.get("checkpoint_sha256") != treatment_identity.get(
        "checkpoint_sha256"
    ):
        raise SystemExit("control/treatment checkpoints differ")
    if _identity_without_deploy_config(
        control_identity
    ) != _identity_without_deploy_config(treatment_identity):
        raise SystemExit(
            "control/treatment identities differ outside deploy config provenance"
        )
    _require_checkpoint_binding(control, control_path, label="control")
    _require_checkpoint_binding(treatment, treatment_path, label="treatment")

    control_config = _load_bound_config(control_identity, label="control")
    treatment_config = _load_bound_config(treatment_identity, label="treatment")
    control_horizon = control_config.get("policy", {}).get("execute_horizon")
    treatment_horizon = treatment_config.get("policy", {}).get("execute_horizon")
    if control_horizon is not None:
        raise SystemExit(f"control execute_horizon is not null: {control_horizon!r}")
    if treatment_horizon != expected_treatment_horizon:
        raise SystemExit(
            f"treatment execute_horizon mismatch: {treatment_horizon!r} != "
            f"{expected_treatment_horizon}"
        )
    normalized_treatment = deepcopy(treatment_config)
    normalized_treatment["policy"]["execute_horizon"] = None
    if normalized_treatment != control_config:
        raise SystemExit("deploy configs differ by more than policy.execute_horizon")

    action_tokens = int(control_identity["action_tokens_per_chunk"])
    if expected_treatment_horizon >= action_tokens:
        raise SystemExit(
            "treatment horizon must be shorter than the generated action chunk"
        )
    _require_info_binding(
        control_path,
        control_identity,
        label="control",
        expected_horizon=None,
    )
    _require_info_binding(
        treatment_path,
        treatment_identity,
        label="treatment",
        expected_horizon=expected_treatment_horizon,
    )
    _require_generation_cadence(control, label="control", horizon=action_tokens)
    _require_generation_cadence(
        treatment, label="treatment", horizon=expected_treatment_horizon
    )
    _require_generation_zero_equal(control, treatment)

    expected_pairs = [pair[:4] for pair in control["pairs"]]
    if control_client["pairs"] != expected_pairs:
        raise SystemExit("control client/server episode identities do not match")
    if treatment_client["pairs"] != expected_pairs:
        raise SystemExit("treatment client/server episode identities do not match")
    _require_server_client_mapping(control, control_client, label="control")
    _require_server_client_mapping(treatment, treatment_client, label="treatment")
    _require_server_client_prompts(control, control_client, label="control")
    _require_server_client_prompts(treatment, treatment_client, label="treatment")
    for episode_index, (control_prompt, treatment_prompt) in enumerate(
        zip(control_client["prompts"], treatment_client["prompts"])
    ):
        if control_prompt != treatment_prompt:
            raise SystemExit(
                f"control/treatment prompt mismatch at episode {episode_index}"
            )

    manifest_sha256 = _require_treatment_manifest_bound_to_control(
        control,
        treatment,
        treatment_client,
        control_path=control_path,
        treatment_path=treatment_path,
        expected_sha256=expected_prompt_manifest_sha256,
    )
    return {
        "episodes": expected_episodes,
        "checkpoint_sha256": control_identity["checkpoint_sha256"],
        "control_execute_horizon": None,
        "treatment_execute_horizon": expected_treatment_horizon,
        "action_tokens_per_chunk": action_tokens,
        "prompt_manifest_sha256": manifest_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-treatment-horizon", type=int, required=True)
    parser.add_argument("--allow-control-prefix", action="store_true")
    parser.add_argument("--expected-prompt-manifest-sha256")
    args = parser.parse_args()
    try:
        report = verify_policy_pair(
            args.control,
            args.treatment,
            expected_episodes=args.expected_episodes,
            expected_treatment_horizon=args.expected_treatment_horizon,
            allow_control_prefix=args.allow_control_prefix,
            expected_prompt_manifest_sha256=args.expected_prompt_manifest_sha256,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print("paired_policy_telemetry: PASS " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())