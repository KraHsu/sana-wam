#!/usr/bin/env python
"""Fail-closed validation for multi-noise stochastic-support rollouts."""

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
    )
    from scripts.verify_paired_noise_telemetry import (
        _require_server_client_mapping,
        load_client_run,
        load_run,
    )
    from scripts.verify_paired_policy_telemetry import (
        _identity_without_deploy_config,
        _load_bound_config,
        _require_generation_cadence,
    )
except ModuleNotFoundError:  # Direct script execution.
    from verify_paired_model_telemetry import (
        _require_checkpoint_binding,
        _require_server_client_prompts,
    )
    from verify_paired_noise_telemetry import (
        _require_server_client_mapping,
        load_client_run,
        load_run,
    )
    from verify_paired_policy_telemetry import (
        _identity_without_deploy_config,
        _load_bound_config,
        _require_generation_cadence,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _derive_model_seed(base_seed: int, noise_pair_key: str) -> int:
    material = f"sana-wam/model-noise/v1\0{base_seed}\0{noise_pair_key}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _normalize_identity(identity: dict, *, canonical_base_seed: int) -> dict:
    normalized = _identity_without_deploy_config(identity)
    normalized["episode_noise_base_seed"] = canonical_base_seed
    return normalized


def _require_info(
    run_dir: Path,
    identity: dict,
    *,
    label: str,
    expected_base_seed: int,
) -> None:
    for filename in ("info.before.json", "info.after.json"):
        path = run_dir / filename
        if not path.is_file():
            raise SystemExit(f"{label} is missing {filename}")
        info = json.loads(path.read_text())
        runtime = info.get("inference_runtime", {})
        if runtime.get("deployment_identity") != identity:
            raise SystemExit(f"{label} {filename} identity mismatch")
        if runtime.get("episode_noise_base_seed") != expected_base_seed:
            raise SystemExit(f"{label} {filename} noise base mismatch")
        if runtime.get("cache_feedback_mode") != "predicted":
            raise SystemExit(f"{label} {filename} feedback mode mismatch")
        policy = info.get("policy_config", {})
        if policy.get("execute_horizon") is not None:
            raise SystemExit(f"{label} {filename} is not greedy execution")
        if info.get("async_inference", {}).get("effective_temporal_ensemble") is not False:
            raise SystemExit(f"{label} {filename} unexpectedly ensembles actions")


def _require_manifest_context(
    server: dict,
    client: dict,
    *,
    run_manifest_path: Path,
    manifest_sha256: str,
    label: str,
) -> None:
    for episode_index, (server_start, client_start) in enumerate(
        zip(server["starts"], client["starts"])
    ):
        for producer, context in (
            ("server", server_start.get("context", {})),
            ("client", client_start.get("context", {})),
        ):
            if context.get("prompt_source") != "manifest":
                raise SystemExit(
                    f"{label} {producer} prompt source mismatch at {episode_index}"
                )
            if context.get("prompt_manifest_ordinal") != episode_index:
                raise SystemExit(
                    f"{label} {producer} manifest ordinal mismatch at {episode_index}"
                )
            context_path = context.get("prompt_manifest_path")
            if not isinstance(context_path, str) or Path(context_path).resolve() != run_manifest_path.resolve():
                raise SystemExit(
                    f"{label} {producer} manifest path mismatch at {episode_index}"
                )
            if context.get("prompt_manifest_sha256") != manifest_sha256:
                raise SystemExit(
                    f"{label} {producer} manifest SHA mismatch at {episode_index}"
                )


def verify_support_plan(plan_path: Path, *, canonical_config_path: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != 1 or plan.get("kind") != "stochastic_support_best_of_4_plan":
        raise SystemExit("invalid support-plan schema/kind")
    replicas = plan.get("replicas")
    failed_seeds = plan.get("failed_environment_seeds")
    if not isinstance(replicas, list) or len(replicas) != 4:
        raise SystemExit("support plan must contain four replicas")
    if not isinstance(failed_seeds, list) or len(failed_seeds) != 15:
        raise SystemExit("support plan must contain fifteen failed scenes")
    if failed_seeds != sorted(failed_seeds) or len(set(failed_seeds)) != 15:
        raise SystemExit("failed-scene schedule is not unique and sorted")

    control_path = Path(plan["control_run"])
    control = load_run(control_path, expected_mode="predicted", expected_episodes=30)
    control_client = load_client_run(control_path, expected_episodes=30)
    control_identity = control["identity"]
    canonical_base_seed = int(control_identity["episode_noise_base_seed"])
    canonical_config = yaml.safe_load(canonical_config_path.read_bytes())
    expected_checkpoint = plan["checkpoint_sha256"]
    if control_identity["checkpoint_sha256"] != expected_checkpoint:
        raise SystemExit("support plan checkpoint does not match control")
    control_by_seed = {
        pair[2]: (pair, start, prompt)
        for pair, start, prompt in zip(
            control["pairs"], control["starts"], control_client["prompts"]
        )
    }

    all_model_seeds = {seed: set() for seed in failed_seeds}
    success_by_replica = []
    success_by_scene = {seed: [] for seed in failed_seeds}
    reference_schedule = None
    base_seeds = set()
    for expected_index, replica in enumerate(replicas):
        label = f"replica_{expected_index}"
        if replica.get("replica_index") != expected_index:
            raise SystemExit(f"{label} index mismatch")
        base_seed = int(replica["episode_noise_base_seed"])
        if base_seed in base_seeds or base_seed == canonical_base_seed:
            raise SystemExit(f"{label} noise base is not independent")
        base_seeds.add(base_seed)
        config_path = Path(replica["config_path"])
        source_manifest_path = Path(replica["prompt_manifest_path"])
        run_dir = Path(replica["run_dir"])
        if _sha256(config_path) != replica["config_sha256"]:
            raise SystemExit(f"{label} source config SHA mismatch")
        if _sha256(source_manifest_path) != replica["prompt_manifest_sha256"]:
            raise SystemExit(f"{label} source manifest SHA mismatch")
        run_manifest_path = run_dir / "prompt_manifest.json"
        if run_manifest_path.read_bytes() != source_manifest_path.read_bytes():
            raise SystemExit(f"{label} run manifest differs from immutable source")
        if (run_dir / "eval.exit").read_text().strip() != "0":
            raise SystemExit(f"{label} evaluation exit is not zero")

        manifest = json.loads(source_manifest_path.read_text())
        episodes = manifest.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != 15:
            raise SystemExit(f"{label} manifest episode count mismatch")
        schedule = [episode.get("environment_seed") for episode in episodes]
        if schedule != failed_seeds:
            raise SystemExit(f"{label} failed-scene schedule mismatch")
        prompt_schedule = [episode.get("prompt") for episode in episodes]
        if reference_schedule is None:
            reference_schedule = prompt_schedule
        elif prompt_schedule != reference_schedule:
            raise SystemExit(f"{label} prompt schedule differs across replicas")

        run = load_run(run_dir, expected_mode="predicted", expected_episodes=15)
        client = load_client_run(run_dir, expected_episodes=15)
        identity = run["identity"]
        if identity["checkpoint_sha256"] != expected_checkpoint:
            raise SystemExit(f"{label} checkpoint mismatch")
        if _normalize_identity(
            identity, canonical_base_seed=canonical_base_seed
        ) != _normalize_identity(
            control_identity, canonical_base_seed=canonical_base_seed
        ):
            raise SystemExit(f"{label} identity differs outside config/noise base")
        _require_checkpoint_binding(run, run_dir, label=label)
        bound_config = _load_bound_config(identity, label=label)
        if bound_config.get("inference", {}).get("episode_noise_base_seed") != base_seed:
            raise SystemExit(f"{label} bound config noise base mismatch")
        normalized_config = deepcopy(bound_config)
        normalized_config["inference"]["episode_noise_base_seed"] = canonical_base_seed
        if normalized_config != canonical_config:
            raise SystemExit(f"{label} config differs by more than noise base")
        _require_info(run_dir, identity, label=label, expected_base_seed=base_seed)
        _require_generation_cadence(
            run,
            label=label,
            horizon=int(identity["action_tokens_per_chunk"]),
        )

        expected_pairs = []
        for ordinal, episode in enumerate(episodes):
            environment_seed = episode["environment_seed"]
            control_pair, control_start, control_prompt = control_by_seed[
                environment_seed
            ]
            pair_key = episode["noise_pair_key"]
            derived_seed = _derive_model_seed(base_seed, pair_key)
            if episode.get("ordinal") != ordinal or episode.get("episode_index") != ordinal:
                raise SystemExit(f"{label} manifest ordinal mismatch at {ordinal}")
            if episode.get("model_noise_seed") != derived_seed:
                raise SystemExit(f"{label} manifest model seed mismatch at {ordinal}")
            if episode.get("prompt") != control_prompt:
                raise SystemExit(f"{label} prompt differs from control at {ordinal}")
            if episode.get("source_episode_key") != control_start["episode_key"]:
                raise SystemExit(f"{label} source episode mismatch at {ordinal}")
            if pair_key != control_pair[3]:
                raise SystemExit(f"{label} pair key differs from control at {ordinal}")
            all_model_seeds[environment_seed].add(derived_seed)
            expected_pairs.append(
                (
                    episode["task_name"],
                    episode["task_config"],
                    environment_seed,
                    pair_key,
                    derived_seed,
                )
            )
        if run["pairs"] != expected_pairs:
            raise SystemExit(f"{label} server scene/noise pairs mismatch")
        if client["pairs"] != [pair[:4] for pair in expected_pairs]:
            raise SystemExit(f"{label} client scene pairs mismatch")
        _require_server_client_mapping(run, client, label=label)
        _require_server_client_prompts(run, client, label=label)
        _require_manifest_context(
            run,
            client,
            run_manifest_path=run_manifest_path,
            manifest_sha256=replica["prompt_manifest_sha256"],
            label=label,
        )

        successes = []
        for pair, end in zip(run["pairs"], client["ends"]):
            if end["success"]:
                successes.append(pair[2])
                success_by_scene[pair[2]].append(expected_index)
        success_by_replica.append(successes)

    if any(len(values) != 4 for values in all_model_seeds.values()):
        raise SystemExit("model noise seeds are not unique across all replicas")
    recovered = [seed for seed, indices in success_by_scene.items() if indices]
    return {
        "replicas": 4,
        "scenes": 15,
        "rollouts": 60,
        "checkpoint_sha256": expected_checkpoint,
        "success_rollouts": sum(len(values) for values in success_by_replica),
        "success_rate": sum(len(values) for values in success_by_replica) / 60,
        "successes_by_replica": success_by_replica,
        "recovered_scenes": recovered,
        "recovered_scene_count": len(recovered),
        "best_of_4_rate": len(recovered) / 15,
        "successful_replica_indices_by_scene": {
            str(seed): indices for seed, indices in success_by_scene.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--canonical-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_support_plan(
        args.plan, canonical_config_path=args.canonical_config
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite output: {args.output}")
        args.output.write_text(encoded)
    print("stochastic_support_verification: PASS")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())