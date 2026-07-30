#!/usr/bin/env python
"""Validate a prompt/scene/noise-paired evaluation of two different models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.verify_paired_noise_telemetry import (
        SCHEMA_VERSION,
        _array_sha256,
        _artifact_root,
        _prefix_run,
        _require_server_client_mapping,
        _same_contiguous_bytes,
        load_client_run,
        load_run,
    )
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from verify_paired_noise_telemetry import (
        SCHEMA_VERSION,
        _array_sha256,
        _artifact_root,
        _prefix_run,
        _require_server_client_mapping,
        _same_contiguous_bytes,
        load_client_run,
        load_run,
    )


CHECKPOINT_IDENTITY_FIELDS = {
    "checkpoint_path",
    "checkpoint_size",
    "checkpoint_sha256",
}


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity_without_checkpoint(identity: dict) -> dict:
    return {
        key: value
        for key, value in identity.items()
        if key not in CHECKPOINT_IDENTITY_FIELDS
    }


def _require_identity_everywhere(run: dict, *, label: str) -> None:
    identity = run["identity"]
    for episode_index, end in enumerate(run["ends"]):
        if end.get("runtime", {}).get("deployment_identity") != identity:
            raise SystemExit(
                f"{label} deployment identity changed at episode {episode_index} end"
            )
    for episode_index, steps in enumerate(run["steps"]):
        for request_id, step in enumerate(steps, 1):
            if step["record"].get("runtime", {}).get("deployment_identity") != identity:
                raise SystemExit(
                    f"{label} deployment identity changed at episode "
                    f"{episode_index} request={request_id}"
                )


def _require_checkpoint_binding(run: dict, run_path: Path, *, label: str) -> None:
    identity = run["identity"]
    checkpoint_path = identity.get("checkpoint_path")
    checkpoint_size = identity.get("checkpoint_size")
    checkpoint_sha256 = identity.get("checkpoint_sha256")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise SystemExit(f"{label} checkpoint path is invalid")
    if (
        isinstance(checkpoint_size, bool)
        or not isinstance(checkpoint_size, int)
        or checkpoint_size < 1
    ):
        raise SystemExit(f"{label} checkpoint size is invalid")
    if not _is_sha256(checkpoint_sha256):
        raise SystemExit(f"{label} checkpoint SHA-256 is invalid")

    _require_identity_everywhere(run, label=label)

    sidecar = _artifact_root(run_path) / "checkpoint.sha256"
    if not sidecar.is_file():
        raise SystemExit(f"{label} checkpoint sidecar is missing: {sidecar}")
    lines = [line for line in sidecar.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise SystemExit(f"{label} checkpoint sidecar must contain exactly one record")
    fields = lines[0].split(maxsplit=1)
    if len(fields) != 2:
        raise SystemExit(f"{label} checkpoint sidecar is malformed")
    sidecar_sha256, sidecar_path = fields
    if sidecar_sha256 != checkpoint_sha256 or sidecar_path != checkpoint_path:
        raise SystemExit(
            f"{label} checkpoint sidecar does not match telemetry identity"
        )

    artifact = Path(checkpoint_path).expanduser()
    if artifact.is_file() and artifact.stat().st_size != checkpoint_size:
        raise SystemExit(f"{label} checkpoint size does not match the local artifact")


def _require_server_client_prompts(server: dict, client: dict, *, label: str) -> None:
    for episode_index, (server_start, client_start, prompt) in enumerate(
        zip(server["starts"], client["starts"], client["prompts"])
    ):
        server_context = server_start.get("context", {})
        client_context = client_start.get("context", {})
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for producer, context in (
            ("server", server_context),
            ("client", client_context),
        ):
            if context.get("prompt") != prompt:
                raise SystemExit(
                    f"{label} {producer} prompt mismatch at episode {episode_index}"
                )
            if context.get("prompt_sha256") != digest:
                raise SystemExit(
                    f"{label} {producer} prompt SHA-256 mismatch at episode "
                    f"{episode_index}"
                )


def _require_initial_observation_equal(control: dict, treatment: dict) -> None:
    for episode_index, (control_zero, treatment_zero) in enumerate(
        zip(control["generation_zero"], treatment["generation_zero"])
    ):
        if not _same_contiguous_bytes(
            control_zero["observed_proprio"], treatment_zero["observed_proprio"]
        ):
            raise SystemExit(
                f"generation-zero observed_proprio mismatch at episode "
                f"{episode_index} scene={control['pairs'][episode_index][:3]!r}"
            )


def _require_treatment_manifest_bound_to_control(
    control: dict,
    treatment: dict,
    treatment_client: dict,
    *,
    control_path: Path,
    treatment_path: Path,
    expected_sha256: str | None,
) -> str:
    manifest_path = _artifact_root(treatment_path) / "prompt_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"treatment prompt manifest is missing: {manifest_path}")
    encoded = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and manifest_sha256 != expected_sha256:
        raise SystemExit(
            f"treatment prompt manifest SHA-256 {manifest_sha256} does not match "
            f"expected {expected_sha256}"
        )
    try:
        manifest = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid treatment prompt manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit("treatment prompt manifest must be a JSON object")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != "robotwin_prompt_replay"
    ):
        raise SystemExit("invalid treatment prompt manifest schema/kind")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise SystemExit("treatment prompt manifest has invalid source provenance")
    expected_control_root = _artifact_root(control_path).resolve()
    source_run_path = source.get("run_path")
    if (
        not isinstance(source_run_path, str)
        or Path(source_run_path).expanduser().resolve() != expected_control_root
    ):
        raise SystemExit("treatment prompt manifest source run is not the control")
    if source.get("run_id") != control["starts"][0].get("run_id"):
        raise SystemExit("treatment prompt manifest source run_id is not the control")
    if source.get("checkpoint_sha256") != control["identity"]["checkpoint_sha256"]:
        raise SystemExit("treatment prompt manifest checkpoint is not the control")
    if (
        source.get("episode_noise_base_seed")
        != control["identity"]["episode_noise_base_seed"]
    ):
        raise SystemExit("treatment prompt manifest base seed is not the control")

    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < len(treatment["pairs"]):
        raise SystemExit("treatment prompt manifest has insufficient episodes")
    if source.get("episode_count") != len(episodes):
        raise SystemExit("treatment prompt manifest source episode count is invalid")

    for episode_index, (server_start, client_start) in enumerate(
        zip(treatment["starts"], treatment_client["starts"])
    ):
        pair = control["pairs"][episode_index]
        prompt = treatment_client["prompts"][episode_index]
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        entry = episodes[episode_index]
        if not isinstance(entry, dict):
            raise SystemExit(
                f"treatment prompt manifest episode {episode_index} is invalid"
            )
        expected_entry = {
            "ordinal": episode_index,
            "episode_index": episode_index,
            "task_name": pair[0],
            "task_config": pair[1],
            "environment_seed": pair[2],
            "noise_pair_key": pair[3],
            "model_noise_seed": pair[4],
            "source_episode_key": control["starts"][episode_index]["episode_key"],
            "prompt": prompt,
            "prompt_sha256": prompt_sha256,
        }
        if any(entry.get(key) != value for key, value in expected_entry.items()):
            raise SystemExit(
                f"treatment prompt manifest entry mismatch at episode {episode_index}"
            )

        # The replay manifest may record control generation-zero output hashes.
        # Bind those hashes to the control only; treatment outputs are model-dependent.
        generation_zero = entry.get("generation_zero", {})
        control_zero = control["generation_zero"][episode_index]
        if generation_zero.get("predicted_actions_sha256") != _array_sha256(
            control_zero["predicted_actions"]
        ) or generation_zero.get(
            "predicted_actions_normalized_sha256"
        ) != _array_sha256(control_zero["predicted_actions_normalized"]):
            raise SystemExit(
                f"treatment prompt manifest control generation-zero hash mismatch "
                f"at episode {episode_index}"
            )

        for producer, context in (
            ("server", server_start.get("context", {})),
            ("client", client_start.get("context", {})),
        ):
            if context.get("prompt_source") != "manifest":
                raise SystemExit(
                    f"treatment {producer} prompt_source is not manifest at "
                    f"episode {episode_index}"
                )
            if (
                context.get("episode_index") != episode_index
                or context.get("prompt_manifest_ordinal") != episode_index
            ):
                raise SystemExit(
                    f"treatment {producer} manifest ordinal mismatch at episode "
                    f"{episode_index}"
                )
            context_path = context.get("prompt_manifest_path")
            if (
                not isinstance(context_path, str)
                or Path(context_path).expanduser().resolve() != manifest_path.resolve()
            ):
                raise SystemExit(
                    f"treatment {producer} manifest path mismatch at episode "
                    f"{episode_index}"
                )
            if context.get("prompt_manifest_sha256") != manifest_sha256:
                raise SystemExit(
                    f"treatment {producer} manifest SHA-256 mismatch at episode "
                    f"{episode_index}"
                )
    return manifest_sha256


def verify_model_pair(
    control_path: Path,
    treatment_path: Path,
    *,
    expected_episodes: int,
    control_client_path: Path,
    treatment_client_path: Path,
    allow_control_prefix: bool = False,
    expected_prompt_manifest_sha256: str | None = None,
) -> dict:
    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive")
    if expected_prompt_manifest_sha256 is not None and not _is_sha256(
        expected_prompt_manifest_sha256
    ):
        raise ValueError("expected prompt-manifest SHA-256 must be lowercase hex")

    control = load_run(
        control_path,
        expected_mode="predicted",
        expected_episodes=None if allow_control_prefix else expected_episodes,
    )
    if allow_control_prefix:
        control = _prefix_run(control, expected_episodes)
    treatment = load_run(
        treatment_path,
        expected_mode="predicted",
        expected_episodes=expected_episodes,
    )

    if control["pairs"] != treatment["pairs"]:
        raise SystemExit("ordered scene/model-noise pairs do not match")
    if _identity_without_checkpoint(
        control["identity"]
    ) != _identity_without_checkpoint(treatment["identity"]):
        raise SystemExit(
            "control/treatment non-checkpoint deployment identities do not match"
        )
    if (
        control["identity"]["checkpoint_sha256"]
        == treatment["identity"]["checkpoint_sha256"]
    ):
        raise SystemExit("control/treatment checkpoint SHA-256 values must differ")
    _require_checkpoint_binding(control, control_path, label="control")
    _require_checkpoint_binding(treatment, treatment_path, label="treatment")
    _require_initial_observation_equal(control, treatment)

    control_client = load_client_run(
        control_client_path,
        expected_episodes=None if allow_control_prefix else expected_episodes,
    )
    if allow_control_prefix:
        control_client = _prefix_run(control_client, expected_episodes)
    treatment_client = load_client_run(
        treatment_client_path, expected_episodes=expected_episodes
    )
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
                f"control/treatment prompt mismatch at episode {episode_index} "
                f"scene={expected_pairs[episode_index][:3]!r}"
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
        "episodes": len(control["pairs"]),
        "control_checkpoint_sha256": control["identity"]["checkpoint_sha256"],
        "treatment_checkpoint_sha256": treatment["identity"]["checkpoint_sha256"],
        "prompt_manifest_sha256": manifest_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--control-client", type=Path, required=True)
    parser.add_argument("--treatment-client", type=Path, required=True)
    parser.add_argument("--allow-control-prefix", action="store_true")
    parser.add_argument("--expected-prompt-manifest-sha256")
    args = parser.parse_args()

    try:
        report = verify_model_pair(
            args.control,
            args.treatment,
            expected_episodes=args.expected_episodes,
            control_client_path=args.control_client,
            treatment_client_path=args.treatment_client,
            allow_control_prefix=args.allow_control_prefix,
            expected_prompt_manifest_sha256=args.expected_prompt_manifest_sha256,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        "paired_model_telemetry: PASS "
        f"episodes={report['episodes']} "
        f"control_checkpoint={report['control_checkpoint_sha256']} "
        f"treatment_checkpoint={report['treatment_checkpoint_sha256']} "
        f"manifest={report['prompt_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
