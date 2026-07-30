#!/usr/bin/env python
"""Fail-closed verification for common-future-noise outcome rollouts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

try:
    from scripts.verify_paired_model_telemetry import (
        _require_checkpoint_binding,
        _require_server_client_prompts,
    )
    from scripts.verify_paired_noise_telemetry import (
        _array_sha256,
        _require_server_client_mapping,
        _same_contiguous_bytes,
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
        _array_sha256,
        _require_server_client_mapping,
        _same_contiguous_bytes,
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


def _model_seed(base_seed: int, noise_pair_key: str) -> int:
    material = f"sana-wam/model-noise/v1\0{base_seed}\0{noise_pair_key}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _future_seed(base_seed: int, noise_pair_key: str, generation_index: int) -> int:
    material = (
        "sana-wam/common-future-noise/v1\0"
        f"{base_seed}\0{noise_pair_key}\0{generation_index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _validate_plan_artifacts(plan: dict) -> list[dict]:
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "common_future_outcome_g5_plan"
        or plan.get("candidate_count") != 5
        or plan.get("scene_count") != 30
        or plan.get("train_scene_count") != 24
        or plan.get("heldout_scene_count") != 6
        or plan.get("total_rollouts") != 150
    ):
        raise SystemExit("invalid common-future plan schema/counts")
    groups = plan.get("groups")
    replicas = plan.get("replicas")
    if not isinstance(groups, list) or len(groups) != 30:
        raise SystemExit("plan does not contain thirty groups")
    if not isinstance(replicas, list) or len(replicas) != 5:
        raise SystemExit("plan does not contain five candidates")
    if len({group["environment_seed"] for group in groups}) != 30:
        raise SystemExit("plan scene schedule is not unique")
    strata = {}
    for group in groups:
        key = (group["prompt_group"], bool(group["baseline_success"]))
        split = group.get("split")
        strata.setdefault(key, {"train": 0, "heldout": 0})[split] += 1
        source_array = Path(group["source_group_array"])
        if _sha256(source_array) != group["source_group_array_sha256"]:
            raise SystemExit(
                f"source group array changed for scene {group['environment_seed']}"
            )
        if len(group["candidate_generation_zero_normalized_sha256"]) != 5:
            raise SystemExit("candidate action-hash count mismatch")
    if any(counts != {"train": 4, "heldout": 1} for counts in strata.values()):
        raise SystemExit("train/heldout split is not 4/1 in every stratum")
    if len(strata) != 6:
        raise SystemExit("plan does not contain six prompt/outcome strata")

    base_seeds = plan["candidate_episode_noise_base_seeds"]
    common_base = int(plan["common_future_noise_base_seed"])
    canonical_config = yaml.safe_load(Path(plan["canonical_config"]).read_bytes())
    for index, replica in enumerate(replicas):
        if replica.get("candidate_index") != index:
            raise SystemExit(f"candidate {index} index mismatch")
        if replica.get("episode_noise_base_seed") != base_seeds[index]:
            raise SystemExit(f"candidate {index} episode base mismatch")
        if replica.get("common_future_noise_base_seed") != common_base:
            raise SystemExit(f"candidate {index} common-future base mismatch")
        config_path = Path(replica["config_path"])
        manifest_path = Path(replica["prompt_manifest_path"])
        if _sha256(config_path) != replica["config_sha256"]:
            raise SystemExit(f"candidate {index} config SHA mismatch")
        if _sha256(manifest_path) != replica["prompt_manifest_sha256"]:
            raise SystemExit(f"candidate {index} manifest SHA mismatch")
        config = yaml.safe_load(config_path.read_bytes())
        inference = config.get("inference", {})
        if (
            inference.get("episode_noise_base_seed") != base_seeds[index]
            or inference.get("generation_noise_schedule")
            != {"mode": "common_future", "base_seed": common_base}
        ):
            raise SystemExit(f"candidate {index} noise config mismatch")
        normalized = deepcopy(config)
        normalized["inference"]["episode_noise_base_seed"] = base_seeds[0]
        if normalized != canonical_config:
            raise SystemExit(f"candidate {index} config differs outside episode seed")
        manifest = json.loads(manifest_path.read_text())
        episodes = manifest.get("episodes")
        if (
            manifest.get("kind") != "robotwin_prompt_replay"
            or not isinstance(episodes, list)
            or len(episodes) != 30
        ):
            raise SystemExit(f"candidate {index} manifest is invalid")
        for ordinal, (group, episode) in enumerate(zip(groups, episodes)):
            expected_seed = _model_seed(base_seeds[index], group["noise_pair_key"])
            if (
                episode.get("ordinal") != ordinal
                or episode.get("episode_index") != ordinal
                or episode.get("environment_seed") != group["environment_seed"]
                or episode.get("noise_pair_key") != group["noise_pair_key"]
                or episode.get("model_noise_seed") != expected_seed
                or episode.get("prompt") != group["prompt"]
                or episode.get("split") != group["split"]
                or episode.get("candidate_index") != index
                or episode.get("expected_generation_zero_normalized_sha256")
                != group["candidate_generation_zero_normalized_sha256"][index]
            ):
                raise SystemExit(
                    f"candidate {index} manifest mismatch at ordinal {ordinal}"
                )
    return replicas


def verify_runs(plan_path: Path, *, plan_only: bool = False) -> dict:
    plan = json.loads(plan_path.read_text())
    replicas = _validate_plan_artifacts(plan)
    if plan_only:
        return {"status": "PLAN_VALID", "scenes": 30, "rollouts": 150}

    groups = plan["groups"]
    common_base = int(plan["common_future_noise_base_seed"])
    base_seeds = plan["candidate_episode_noise_base_seeds"]
    runs = []
    clients = []
    reference_identity = None
    candidate_success = []
    for index, replica in enumerate(replicas):
        run_dir = Path(replica["run_dir"])
        if (run_dir / "eval.exit").read_text().strip() != "0":
            raise SystemExit(f"candidate {index} evaluation exit is not zero")
        run_manifest = run_dir / "prompt_manifest.json"
        source_manifest = Path(replica["prompt_manifest_path"])
        if run_manifest.read_bytes() != source_manifest.read_bytes():
            raise SystemExit(f"candidate {index} run manifest changed")
        run = load_run(run_dir, expected_mode="predicted", expected_episodes=30)
        client = load_client_run(run_dir, expected_episodes=30)
        identity = run["identity"]
        if (
            identity.get("checkpoint_sha256") != plan["checkpoint_sha256"]
            or identity.get("generation_noise_schedule") != "common_future"
            or identity.get("common_future_noise_base_seed") != common_base
            or identity.get("episode_noise_base_seed") != base_seeds[index]
        ):
            raise SystemExit(f"candidate {index} deployment identity mismatch")
        normalized_identity = _identity_without_deploy_config(identity)
        normalized_identity["episode_noise_base_seed"] = base_seeds[0]
        if reference_identity is None:
            reference_identity = normalized_identity
        elif normalized_identity != reference_identity:
            raise SystemExit(f"candidate {index} identity differs outside episode seed")
        _require_checkpoint_binding(run, run_dir, label=f"candidate_{index}")
        _load_bound_config(identity, label=f"candidate_{index}")
        _require_generation_cadence(
            run,
            label=f"candidate_{index}",
            horizon=int(identity["action_tokens_per_chunk"]),
        )
        expected_pairs = [
            (
                group["task_name"],
                group["task_config"],
                group["environment_seed"],
                group["noise_pair_key"],
                group["candidate_model_noise_seeds"][index],
            )
            for group in groups
        ]
        if run["pairs"] != expected_pairs:
            raise SystemExit(f"candidate {index} server pair schedule mismatch")
        if client["pairs"] != [pair[:4] for pair in expected_pairs]:
            raise SystemExit(f"candidate {index} client pair schedule mismatch")
        _require_server_client_mapping(run, client, label=f"candidate_{index}")
        _require_server_client_prompts(run, client, label=f"candidate_{index}")

        for episode_index, (group, zero, steps) in enumerate(
            zip(groups, run["generation_zero"], run["steps"])
        ):
            expected_hash = group["candidate_generation_zero_normalized_sha256"][
                index
            ]
            if _array_sha256(zero["predicted_actions_normalized"]) != expected_hash:
                raise SystemExit(
                    f"candidate {index} generation zero changed at scene "
                    f"{group['environment_seed']}"
                )
            for step in steps:
                policy = step["record"].get("policy", {})
                if not policy.get("generated"):
                    continue
                generation_index = int(policy["generation_index"])
                expected_seed = (
                    group["candidate_model_noise_seeds"][index]
                    if generation_index == 0
                    else _future_seed(
                        common_base, group["noise_pair_key"], generation_index
                    )
                )
                if (
                    policy.get("generation_noise_schedule") != "common_future"
                    or policy.get("generation_noise_seed") != expected_seed
                ):
                    raise SystemExit(
                        f"candidate {index} future seed mismatch at scene "
                        f"{group['environment_seed']} generation {generation_index}"
                    )
        successes = [bool(end["success"]) for end in client["ends"]]
        candidate_success.append(successes)
        runs.append(run)
        clients.append(client)

    for episode_index, group in enumerate(groups):
        reference = runs[0]["generation_zero"][episode_index]["observed_proprio"]
        for index in range(1, 5):
            observed = runs[index]["generation_zero"][episode_index][
                "observed_proprio"
            ]
            if not _same_contiguous_bytes(reference, observed):
                raise SystemExit(
                    f"initial proprio differs at scene {group['environment_seed']}"
                )

    labels = np.asarray(candidate_success, dtype=np.bool_).T
    positive_counts = labels.sum(axis=1)
    mixed = np.logical_and(positive_counts > 0, positive_counts < 5)
    split_metrics = {}
    for split in ("train", "heldout"):
        indices = [i for i, group in enumerate(groups) if group["split"] == split]
        split_metrics[split] = {
            "scenes": len(indices),
            "positive_rollouts": int(labels[indices].sum()),
            "mixed_scenes": int(mixed[indices].sum()),
            "all_fail_scenes": int((positive_counts[indices] == 0).sum()),
            "all_success_scenes": int((positive_counts[indices] == 5).sum()),
        }
    return {
        "status": "PASS",
        "scenes": 30,
        "rollouts": 150,
        "positive_rollouts": int(labels.sum()),
        "negative_rollouts": int(labels.size - labels.sum()),
        "mixed_scenes": int(mixed.sum()),
        "all_fail_scenes": int((positive_counts == 0).sum()),
        "all_success_scenes": int((positive_counts == 5).sum()),
        "successes_by_candidate": labels.sum(axis=0).astype(int).tolist(),
        "split_metrics": split_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_runs(args.plan, plan_only=args.plan_only)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite output: {args.output}")
        args.output.write_text(encoded)
    print("common_future_outcome_verification: PASS")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
