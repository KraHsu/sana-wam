#!/usr/bin/env python
"""Validate a same-checkpoint generation-zero outcome-reranker evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

try:
    from scripts.verify_paired_model_telemetry import (
        _is_sha256,
        _require_checkpoint_binding,
        _require_initial_observation_equal,
        _require_server_client_prompts,
        _require_treatment_manifest_bound_to_control,
    )
    from scripts.verify_paired_noise_telemetry import (
        _array_sha256,
        _prefix_run,
        _require_server_client_mapping,
        load_client_run,
        load_run,
    )
    from scripts.verify_paired_policy_telemetry import (
        _identity_without_deploy_config,
        _load_bound_config,
        _require_generation_cadence,
        _require_info_binding,
    )
except ModuleNotFoundError:  # Direct script execution.
    from verify_paired_model_telemetry import (
        _is_sha256,
        _require_checkpoint_binding,
        _require_initial_observation_equal,
        _require_server_client_prompts,
        _require_treatment_manifest_bound_to_control,
    )
    from verify_paired_noise_telemetry import (
        _array_sha256,
        _prefix_run,
        _require_server_client_mapping,
        load_client_run,
        load_run,
    )
    from verify_paired_policy_telemetry import (
        _identity_without_deploy_config,
        _load_bound_config,
        _require_generation_cadence,
        _require_info_binding,
    )


def _identity_without_rerank(identity: dict) -> dict:
    normalized = _identity_without_deploy_config(identity)
    normalized.pop("generation_zero_rerank", None)
    return normalized


def _candidate_seed(base_seed: int, candidate_index: int) -> int:
    material = (
        "sana-wam/generation-zero-rerank/v1\0"
        f"{int(base_seed)}\0{int(candidate_index)}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _require_rerank_artifact(
    identity: dict, *, expected_sha256: str, candidate_count: int
) -> dict:
    rerank = identity.get("generation_zero_rerank")
    if not isinstance(rerank, dict):
        raise SystemExit("treatment deployment identity has no reranker")
    if (
        rerank.get("enabled") is not True
        or rerank.get("candidate_count") != candidate_count
        or rerank.get("invocations") != 0
        or rerank.get("last_selection") is not None
    ):
        raise SystemExit("invalid treatment reranker deployment identity")
    ranker = rerank.get("ranker")
    if not isinstance(ranker, dict):
        raise SystemExit("treatment ranker identity is missing")
    expected = {
        "schema_version": 1,
        "sha256": expected_sha256,
        "action_tokens": 28,
        "action_dim": 20,
        "feature_dim": 180,
    }
    if any(ranker.get(key) != value for key, value in expected.items()):
        raise SystemExit("treatment ranker identity does not match expected artifact")
    path_value = ranker.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise SystemExit("treatment ranker path is invalid")
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise SystemExit(f"treatment ranker artifact is missing: {path}")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit("treatment ranker artifact SHA-256 does not match telemetry")
    return rerank


def _require_rerank_episode(
    control: dict,
    treatment: dict,
    episode_index: int,
    *,
    expected_sha256: str,
    candidate_count: int,
) -> None:
    steps = treatment["steps"][episode_index]
    rerank_steps = [
        step
        for step in steps
        if "generation_zero_rerank" in step["record"].get("policy", {})
    ]
    if len(rerank_steps) != 1:
        raise SystemExit(
            f"episode {episode_index} has {len(rerank_steps)} rerank records, expected 1"
        )
    step = rerank_steps[0]
    policy = step["record"]["policy"]
    if (
        step["request_id"] != 1
        or policy.get("generated") is not True
        or policy.get("generation_index") != 0
        or policy.get("chunk_offset") != 0
    ):
        raise SystemExit(f"episode {episode_index} reranking did not occur at generation zero")
    rerank = policy["generation_zero_rerank"]
    seeds = rerank.get("candidate_seeds")
    scores = rerank.get("candidate_scores")
    hashes = rerank.get("candidate_actions_normalized_sha256")
    if (
        rerank.get("ranker_sha256") != expected_sha256
        or rerank.get("candidate_count") != candidate_count
        or rerank.get("generation_index") != 0
        or rerank.get("ar_step_after") != 1
        or not isinstance(seeds, list)
        or not isinstance(scores, list)
        or not isinstance(hashes, list)
        or len(seeds) != candidate_count
        or len(scores) != candidate_count
        or len(hashes) != candidate_count
    ):
        raise SystemExit(f"episode {episode_index} has invalid rerank telemetry")

    base_seed = treatment["pairs"][episode_index][4]
    expected_seeds = [base_seed] + [
        _candidate_seed(base_seed, index) for index in range(1, candidate_count)
    ]
    if seeds != expected_seeds or len(set(seeds)) != candidate_count:
        raise SystemExit(f"episode {episode_index} candidate seeds are invalid")
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        for score in scores
    ):
        raise SystemExit(f"episode {episode_index} candidate scores are invalid")
    if any(not _is_sha256(value) for value in hashes):
        raise SystemExit(f"episode {episode_index} candidate action hashes are invalid")

    selected = rerank.get("selected_candidate_index")
    expected_selected = max(range(candidate_count), key=lambda index: scores[index])
    if (
        selected != expected_selected
        or rerank.get("selected_candidate_seed") != seeds[expected_selected]
    ):
        raise SystemExit(f"episode {episode_index} selected candidate is not argmax")

    control_zero = control["generation_zero"][episode_index]
    treatment_zero = treatment["generation_zero"][episode_index]
    if hashes[0] != _array_sha256(control_zero["predicted_actions_normalized"]):
        raise SystemExit(
            f"episode {episode_index} candidate zero does not reproduce control"
        )
    if hashes[selected] != _array_sha256(
        treatment_zero["predicted_actions_normalized"]
    ):
        raise SystemExit(
            f"episode {episode_index} selected candidate does not match saved chunk"
        )

    end_runtime = treatment["ends"][episode_index].get("runtime", {})
    end_rerank = end_runtime.get("generation_zero_rerank", {})
    if (
        end_rerank.get("enabled") is not True
        or end_rerank.get("candidate_count") != candidate_count
        or end_rerank.get("invocations") != 1
        or end_rerank.get("last_selection") != rerank
    ):
        raise SystemExit(f"episode {episode_index} end runtime lost rerank state")

    for generation_step in (
        step for step in steps if step["record"].get("policy", {}).get("generated")
    ):
        record = generation_step["record"]
        generation_index = record["policy"]["generation_index"]
        if generation_index > 0 and "generation_zero_rerank" in record["policy"]:
            raise SystemExit(
                f"episode {episode_index} reranker reappeared at generation "
                f"{generation_index}"
            )
        chunk_path = treatment["root"] / record["chunk_npz"]
        with np.load(chunk_path, allow_pickle=False) as arrays:
            expected_arrays = {
                "schema_version",
                "predicted_actions",
                "predicted_actions_normalized",
            }
            if set(arrays.files) != expected_arrays:
                raise SystemExit(
                    f"episode {episode_index} chunk contains non-selected candidate arrays"
                )


def verify_rerank_pair(
    control_path: Path,
    treatment_path: Path,
    *,
    expected_episodes: int,
    expected_ranker_sha256: str,
    candidate_count: int,
    allow_control_prefix: bool = False,
    expected_prompt_manifest_sha256: str | None = None,
) -> dict:
    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive")
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    if not _is_sha256(expected_ranker_sha256):
        raise ValueError("expected ranker SHA-256 must be lowercase hex")

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
        treatment_path, expected_mode="predicted", expected_episodes=expected_episodes
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
    if _identity_without_rerank(control_identity) != _identity_without_rerank(
        treatment_identity
    ):
        raise SystemExit(
            "control/treatment identities differ outside reranker/config provenance"
        )
    control_rerank = control_identity.get("generation_zero_rerank")
    if isinstance(control_rerank, dict) and control_rerank.get("enabled") is not False:
        raise SystemExit("control unexpectedly enables generation-zero reranking")
    _require_rerank_artifact(
        treatment_identity,
        expected_sha256=expected_ranker_sha256,
        candidate_count=candidate_count,
    )
    _require_checkpoint_binding(control, control_path, label="control")
    _require_checkpoint_binding(treatment, treatment_path, label="treatment")
    _require_initial_observation_equal(control, treatment)

    control_config = _load_bound_config(control_identity, label="control")
    treatment_config = _load_bound_config(treatment_identity, label="treatment")
    treatment_rerank = treatment_config.get("inference", {}).get(
        "generation_zero_rerank"
    )
    if not isinstance(treatment_rerank, dict):
        raise SystemExit("treatment deploy config has no generation_zero_rerank")
    if (
        treatment_rerank.get("enabled") is not True
        or treatment_rerank.get("candidate_count") != candidate_count
        or treatment_rerank.get("expected_ranker_sha256")
        != expected_ranker_sha256
    ):
        raise SystemExit("treatment deploy config reranker does not match expected")
    normalized_treatment = deepcopy(treatment_config)
    normalized_treatment["inference"].pop("generation_zero_rerank")
    if normalized_treatment != control_config:
        raise SystemExit(
            "deploy configs differ by more than inference.generation_zero_rerank"
        )

    _require_info_binding(
        control_path, control_identity, label="control", expected_horizon=None
    )
    _require_info_binding(
        treatment_path, treatment_identity, label="treatment", expected_horizon=None
    )
    horizon = int(control_identity["action_tokens_per_chunk"])
    _require_generation_cadence(control, label="control", horizon=horizon)
    _require_generation_cadence(treatment, label="treatment", horizon=horizon)
    for episode_index in range(expected_episodes):
        _require_rerank_episode(
            control,
            treatment,
            episode_index,
            expected_sha256=expected_ranker_sha256,
            candidate_count=candidate_count,
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
    if control_client["prompts"] != treatment_client["prompts"]:
        raise SystemExit("control/treatment prompts do not match")
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
        "ranker_sha256": expected_ranker_sha256,
        "candidate_count": candidate_count,
        "prompt_manifest_sha256": manifest_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-ranker-sha256", required=True)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--allow-control-prefix", action="store_true")
    parser.add_argument("--expected-prompt-manifest-sha256")
    args = parser.parse_args()
    try:
        report = verify_rerank_pair(
            args.control,
            args.treatment,
            expected_episodes=args.expected_episodes,
            expected_ranker_sha256=args.expected_ranker_sha256,
            candidate_count=args.candidate_count,
            allow_control_prefix=args.allow_control_prefix,
            expected_prompt_manifest_sha256=args.expected_prompt_manifest_sha256,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print("paired_rerank_telemetry: PASS " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
