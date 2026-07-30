#!/usr/bin/env python
"""Fail-closed validation for balanced same-scene outcome groups."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
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
        _load_bound_config,
        _require_generation_cadence,
    )
    from scripts.verify_stochastic_support_runs import (
        _derive_model_seed,
        _normalize_identity,
        _require_info,
        _require_manifest_context,
        _sha256,
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
        _load_bound_config,
        _require_generation_cadence,
    )
    from verify_stochastic_support_runs import (
        _derive_model_seed,
        _normalize_identity,
        _require_info,
        _require_manifest_context,
        _sha256,
    )


def _load_source_run(path: Path, cache: dict) -> tuple[dict, dict, dict]:
    resolved = path.resolve()
    if resolved not in cache:
        manifest = json.loads((resolved / "prompt_manifest.json").read_text())
        episode_count = len(manifest.get("episodes", []))
        cache[resolved] = (
            load_run(
                resolved,
                expected_mode="predicted",
                expected_episodes=episode_count,
            ),
            load_client_run(resolved, expected_episodes=episode_count),
            manifest,
        )
    return cache[resolved]


def verify_outcome_groups(plan_path: Path, *, canonical_config_path: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != 1 or plan.get("kind") != "outcome_groups_g5_plan":
        raise SystemExit("invalid outcome-group plan schema/kind")
    selected = plan.get("selected_scenes")
    replicas = plan.get("replicas")
    if not isinstance(selected, list) or len(selected) != 30:
        raise SystemExit("outcome-group plan must select 30 scenes")
    if not isinstance(replicas, list) or len(replicas) != 4:
        raise SystemExit("outcome-group plan must contain four new replicas")
    environment_seeds = [row.get("environment_seed") for row in selected]
    if environment_seeds != sorted(environment_seeds) or len(set(environment_seeds)) != 30:
        raise SystemExit("selected scene schedule is not unique and sorted")
    if any(seed < 200000 for seed in environment_seeds):
        raise SystemExit("strict evaluation seed leaked into outcome training groups")

    strata = Counter(
        (row.get("prompt_group"), bool(row.get("baseline_success")))
        for row in selected
    )
    expected_strata = {
        (prompt_group, outcome): 5
        for prompt_group in ("left", "right", "unspecified")
        for outcome in (False, True)
    }
    if strata != expected_strata:
        raise SystemExit(f"selection strata mismatch: {strata!r}")

    expected_checkpoint = plan["checkpoint_sha256"]
    canonical_base_seed = int(plan["canonical_episode_noise_base_seed"])
    canonical_config = yaml.safe_load(canonical_config_path.read_bytes())
    source_cache = {}
    reference_identity = None
    source_samples = {}
    for row in selected:
        seed = row["environment_seed"]
        source_path = Path(row["source_run"])
        server, client, source_manifest = _load_source_run(
            source_path, source_cache
        )
        identity = server["identity"]
        if identity["checkpoint_sha256"] != expected_checkpoint:
            raise SystemExit(f"source checkpoint mismatch for seed {seed}")
        if identity["episode_noise_base_seed"] != canonical_base_seed:
            raise SystemExit(f"source noise base mismatch for seed {seed}")
        if reference_identity is None:
            reference_identity = identity
        elif _normalize_identity(
            identity, canonical_base_seed=canonical_base_seed
        ) != _normalize_identity(
            reference_identity, canonical_base_seed=canonical_base_seed
        ):
            raise SystemExit(f"source identity mismatch for seed {seed}")
        if _sha256(source_path / "prompt_manifest.json") != row[
            "source_prompt_manifest_sha256"
        ]:
            raise SystemExit(f"source prompt manifest SHA mismatch for seed {seed}")
        matches = [
            index for index, pair in enumerate(server["pairs"]) if pair[2] == seed
        ]
        if len(matches) != 1:
            raise SystemExit(f"source scene lookup mismatch for seed {seed}")
        index = matches[0]
        pair = server["pairs"][index]
        start = server["starts"][index]
        end = client["ends"][index]
        prompt = client["prompts"][index]
        source_entry = source_manifest["episodes"][index]
        expected_fields = {
            "task_name": pair[0],
            "task_config": pair[1],
            "noise_pair_key": pair[3],
            "baseline_model_noise_seed": pair[4],
            "baseline_success": bool(end["success"]),
            "prompt": prompt,
            "prompt_sha256": source_entry["prompt_sha256"],
            "source_episode_key": start["episode_key"],
        }
        if any(row.get(key) != value for key, value in expected_fields.items()):
            raise SystemExit(f"source sample provenance mismatch for seed {seed}")
        source_samples[seed] = {
            "model_noise_seed": pair[4],
            "success": bool(end["success"]),
            "prompt": prompt,
            "pair": pair,
        }

    assert reference_identity is not None
    model_seeds_by_scene = {
        seed: {source_samples[seed]["model_noise_seed"]}
        for seed in environment_seeds
    }
    outcomes_by_scene = {
        seed: [source_samples[seed]["success"]] for seed in environment_seeds
    }
    successes_by_replica = []
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
        if not isinstance(episodes, list) or len(episodes) != 30:
            raise SystemExit(f"{label} manifest episode count mismatch")
        if [entry.get("environment_seed") for entry in episodes] != environment_seeds:
            raise SystemExit(f"{label} scene schedule mismatch")
        run = load_run(run_dir, expected_mode="predicted", expected_episodes=30)
        client = load_client_run(run_dir, expected_episodes=30)
        identity = run["identity"]
        if identity["checkpoint_sha256"] != expected_checkpoint:
            raise SystemExit(f"{label} checkpoint mismatch")
        if _normalize_identity(
            identity, canonical_base_seed=canonical_base_seed
        ) != _normalize_identity(
            reference_identity, canonical_base_seed=canonical_base_seed
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
        for ordinal, (entry, selected_row) in enumerate(zip(episodes, selected)):
            seed = selected_row["environment_seed"]
            pair_key = selected_row["noise_pair_key"]
            derived_seed = _derive_model_seed(base_seed, pair_key)
            expected_entry_fields = {
                "ordinal": ordinal,
                "episode_index": ordinal,
                "task_name": selected_row["task_name"],
                "task_config": selected_row["task_config"],
                "environment_seed": seed,
                "noise_pair_key": pair_key,
                "model_noise_seed": derived_seed,
                "source_episode_key": selected_row["source_episode_key"],
                "prompt": selected_row["prompt"],
                "prompt_sha256": selected_row["prompt_sha256"],
                "prompt_group": selected_row["prompt_group"],
                "source_run": selected_row["source_run"],
                "baseline_model_noise_seed": selected_row[
                    "baseline_model_noise_seed"
                ],
                "baseline_success": selected_row["baseline_success"],
            }
            if any(entry.get(key) != value for key, value in expected_entry_fields.items()):
                raise SystemExit(f"{label} manifest entry mismatch at {ordinal}")
            if derived_seed in model_seeds_by_scene[seed]:
                raise SystemExit(f"{label} duplicate model seed for scene {seed}")
            model_seeds_by_scene[seed].add(derived_seed)
            expected_pairs.append(
                (
                    entry["task_name"],
                    entry["task_config"],
                    seed,
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
        replica_successes = []
        for pair, end in zip(run["pairs"], client["ends"]):
            success = bool(end["success"])
            outcomes_by_scene[pair[2]].append(success)
            if success:
                replica_successes.append(pair[2])
        successes_by_replica.append(replica_successes)

    if any(len(values) != 5 for values in model_seeds_by_scene.values()):
        raise SystemExit("each scene does not have five unique model-noise seeds")
    if any(len(values) != 5 for values in outcomes_by_scene.values()):
        raise SystemExit("each scene does not have five terminal outcomes")

    positive_counts = {
        seed: sum(outcomes) for seed, outcomes in outcomes_by_scene.items()
    }
    mixed_scenes = [seed for seed, count in positive_counts.items() if 0 < count < 5]
    all_fail = [seed for seed, count in positive_counts.items() if count == 0]
    all_success = [seed for seed, count in positive_counts.items() if count == 5]
    subgroup_reports = {}
    for prompt_group in ("left", "right", "unspecified"):
        seeds = [
            row["environment_seed"]
            for row in selected
            if row["prompt_group"] == prompt_group
        ]
        subgroup_reports[prompt_group] = {
            "scenes": len(seeds),
            "success_rollouts": sum(positive_counts[seed] for seed in seeds),
            "total_rollouts": len(seeds) * 5,
            "mixed_scenes": sum(seed in mixed_scenes for seed in seeds),
            "all_fail_scenes": sum(seed in all_fail for seed in seeds),
            "all_success_scenes": sum(seed in all_success for seed in seeds),
        }
    baseline_strata_reports = {}
    for baseline_success in (False, True):
        label = "baseline_success" if baseline_success else "baseline_failure"
        seeds = [
            row["environment_seed"]
            for row in selected
            if row["baseline_success"] is baseline_success
        ]
        new_successes = sum(
            sum(outcomes_by_scene[seed][1:]) for seed in seeds
        )
        baseline_strata_reports[label] = {
            "scenes": len(seeds),
            "new_success_rollouts": new_successes,
            "new_rollouts": len(seeds) * 4,
            "mixed_scenes": sum(seed in mixed_scenes for seed in seeds),
        }
    return {
        "checkpoint_sha256": expected_checkpoint,
        "scenes": 30,
        "group_size": 5,
        "existing_rollouts": 30,
        "new_rollouts": 120,
        "total_rollouts": 150,
        "new_success_rollouts": sum(
            len(successes) for successes in successes_by_replica
        ),
        "successes_by_replica": successes_by_replica,
        "total_success_rollouts": sum(positive_counts.values()),
        "mixed_scene_count": len(mixed_scenes),
        "mixed_scenes": mixed_scenes,
        "all_fail_scene_count": len(all_fail),
        "all_fail_scenes": all_fail,
        "all_success_scene_count": len(all_success),
        "all_success_scenes": all_success,
        "positive_count_histogram": dict(
            sorted(Counter(positive_counts.values()).items())
        ),
        "positive_counts_by_scene": {
            str(seed): count for seed, count in positive_counts.items()
        },
        "outcomes_by_scene": {
            str(seed): outcomes for seed, outcomes in outcomes_by_scene.items()
        },
        "prompt_subgroups": subgroup_reports,
        "baseline_outcome_strata": baseline_strata_reports,
        "source_run_count": len(source_cache),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--canonical-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_outcome_groups(
        args.plan, canonical_config_path=args.canonical_config
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite output: {args.output}")
        args.output.write_text(encoded)
    print("outcome_group_verification: PASS")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())