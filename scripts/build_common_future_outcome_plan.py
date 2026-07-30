#!/usr/bin/env python
"""Build a balanced common-future-noise counterfactual rollout campaign."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


CHECKPOINT_SHA256 = "aea6b58c9107aeeeffba561f39ffedfc3f8fd8420a4abfcab9038c8cdd4a129d"
CANDIDATE_BASE_SEEDS = (20260715, 2026072401, 2026072402, 2026072403, 2026072404)
COMMON_FUTURE_BASE_SEED = 20260722001
SELECTION_NAMESPACE = "common-future-outcome-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _model_seed(base_seed: int, noise_pair_key: str) -> int:
    material = f"sana-wam/model-noise/v1\0{base_seed}\0{noise_pair_key}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return _sha256(path)


def _heldout_scenes(groups: list[dict]) -> set[int]:
    strata: dict[tuple[str, bool], list[dict]] = {}
    for group in groups:
        key = (group["prompt_group"], bool(group["baseline_success"]))
        strata.setdefault(key, []).append(group)
    expected = {
        (prompt_group, success)
        for prompt_group in ("left", "right", "unspecified")
        for success in (False, True)
    }
    if set(strata) != expected or any(len(values) != 5 for values in strata.values()):
        raise ValueError("source plan is not balanced at five scenes per stratum")
    heldout = set()
    for key, values in strata.items():
        ordered = sorted(
            values,
            key=lambda group: hashlib.sha256(
                f"{SELECTION_NAMESPACE}|heldout|{group['environment_seed']}".encode()
            ).hexdigest(),
        )
        heldout.add(int(ordered[0]["environment_seed"]))
    return heldout


def build_plan(
    source_plan_path: Path,
    source_dataset_manifest_path: Path,
    canonical_config_path: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    source_plan = json.loads(source_plan_path.read_text())
    source_dataset = json.loads(source_dataset_manifest_path.read_text())
    if source_plan.get("kind") != "outcome_groups_g5_plan":
        raise ValueError("invalid source outcome plan")
    if (
        source_dataset.get("kind")
        != "robotwin_generation_zero_outcome_preference_dataset"
    ):
        raise ValueError("invalid source preference dataset")
    if source_plan.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("source checkpoint mismatch")
    source_groups = source_plan["selected_scenes"]
    if len(source_groups) != 30 or len(source_dataset.get("groups", [])) != 30:
        raise ValueError("expected thirty source groups")

    dataset_root = source_dataset_manifest_path.parent
    dataset_by_seed = {
        int(group["environment_seed"]): group for group in source_dataset["groups"]
    }
    groups = []
    for scene in source_groups:
        environment_seed = int(scene["environment_seed"])
        dataset_group = dataset_by_seed[environment_seed]
        array_path = dataset_root / dataset_group["group_array"]["path"]
        with np.load(array_path, allow_pickle=False) as arrays:
            actions_normalized = arrays["predicted_actions_normalized"].copy()
            model_noise_seeds = arrays["model_noise_seed"].astype(np.int64).copy()
        if actions_normalized.shape != (5, 28, 20):
            raise ValueError(f"bad candidate actions for scene {environment_seed}")
        expected_seeds = np.asarray(
            [
                _model_seed(base_seed, scene["noise_pair_key"])
                for base_seed in CANDIDATE_BASE_SEEDS
            ],
            dtype=np.int64,
        )
        if not np.array_equal(model_noise_seeds, expected_seeds):
            raise ValueError(
                f"source candidate seed schedule mismatch for scene {environment_seed}"
            )
        groups.append(
            {
                **deepcopy(scene),
                "candidate_model_noise_seeds": expected_seeds.tolist(),
                "candidate_generation_zero_normalized_sha256": [
                    _array_sha256(actions_normalized[index]) for index in range(5)
                ],
                "source_group_array": str(array_path.resolve()),
                "source_group_array_sha256": _sha256(array_path),
            }
        )
    heldout = _heldout_scenes(groups)
    for group in groups:
        group["split"] = (
            "heldout" if group["environment_seed"] in heldout else "train"
        )

    output_dir.mkdir(parents=True)
    canonical_config = yaml.safe_load(canonical_config_path.read_bytes())
    replicas = []
    for candidate_index, episode_base_seed in enumerate(CANDIDATE_BASE_SEEDS):
        replica_dir = output_dir / f"candidate_{candidate_index}"
        replica_dir.mkdir()
        config = deepcopy(canonical_config)
        inference = config.setdefault("inference", {})
        inference["episode_noise_mode"] = "paired"
        inference["episode_noise_base_seed"] = episode_base_seed
        inference["generation_noise_schedule"] = {
            "mode": "common_future",
            "base_seed": COMMON_FUTURE_BASE_SEED,
        }
        config_path = replica_dir / "deploy_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

        episodes = []
        for ordinal, group in enumerate(groups):
            episodes.append(
                {
                    "ordinal": ordinal,
                    "episode_index": ordinal,
                    "task_name": group["task_name"],
                    "task_config": group["task_config"],
                    "environment_seed": group["environment_seed"],
                    "noise_pair_key": group["noise_pair_key"],
                    "model_noise_seed": group["candidate_model_noise_seeds"][
                        candidate_index
                    ],
                    "prompt": group["prompt"],
                    "prompt_sha256": group["prompt_sha256"],
                    "prompt_group": group["prompt_group"],
                    "baseline_success": group["baseline_success"],
                    "source_episode_key": group["source_episode_key"],
                    "source_run": group["source_run"],
                    "split": group["split"],
                    "candidate_index": candidate_index,
                    "expected_generation_zero_normalized_sha256": group[
                        "candidate_generation_zero_normalized_sha256"
                    ][candidate_index],
                }
            )
        manifest = {
            "schema_version": 1,
            "kind": "robotwin_prompt_replay",
            "instruction_type": "unseen",
            "source": {
                "kind": "common_future_outcome_counterfactuals",
                "source_plan": str(source_plan_path.resolve()),
                "source_plan_sha256": _sha256(source_plan_path),
                "episode_count": len(episodes),
                "candidate_index": candidate_index,
                "episode_noise_base_seed": episode_base_seed,
                "common_future_noise_base_seed": COMMON_FUTURE_BASE_SEED,
            },
            "episodes": episodes,
        }
        manifest_path = replica_dir / "prompt_manifest.json"
        manifest_sha256 = _write_json(manifest_path, manifest)

        http_port = 19200 + 10 * candidate_index
        run_dir = replica_dir / "run"
        runner_path = replica_dir / "runner.sh"
        runner_log = replica_dir / "runner.log"
        runner_path.write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zch/workspace/sana-wam
RG_DIR=/home/zch/.vscode-server/cli/servers/Stable-4fe60c8b1cdac1c4c174f2fb180d0d758272d713/server/node_modules/@vscode/ripgrep-universal/bin/linux-x64
export PATH=\"$RG_DIR:$PATH\"
cd \"$ROOT\"
[[ -f \"{manifest_path.resolve()}\" && -f \"{config_path.resolve()}\" ]]
[[ ! -e \"{run_dir.resolve()}\" && ! -e \"{run_dir.resolve()}.run-lock\" ]]
env \\
  -u PAIR_WITH \\
  -u PAIR_ALLOW_CONTROL_PREFIX \\
  -u ROBOTWIN_ENV_SEED_OFFSET \\
  -u ROBOTWIN_ENV_SEED_INDEX \\
  -u ROBOTWIN_STEP_LIMITS_PATH \\
  -u ROBOTWIN_DAGGER_PLAN \\
  -u ROBOTWIN_POLICY_CONFIG_PATH \\
  ROBOTWIN_TEST_NUM=30 \\
  RUN_DIR=\"{run_dir.resolve()}\" \\
  ROBOTWIN_PROMPT_MANIFEST=\"{manifest_path.resolve()}\" \\
  EXPECTED_PROMPT_MANIFEST_SHA256=\"{manifest_sha256}\" \\
  CHECKPOINT_DIR=/home/zch/wuji-openwam-dev/sandbox/sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25 \\
  CHECKPOINT_NAME=checkpoint_step_12000.safetensors \\
  EXPECTED_CHECKPOINT_SHA={CHECKPOINT_SHA256} \\
  DEPLOY_CONFIG=\"{config_path.resolve()}\" \\
  LABEL=common_future_g5_c{candidate_index}_n30 \\
  HTTP_PORT={http_port} \\
  SERVER_GPU={candidate_index} \\
  SIM_GPU={candidate_index} \\
  bash scripts/run_ar_lownoise_paired_arm.sh predicted
echo COMMON_FUTURE_CANDIDATE_{candidate_index}_COMPLETE
"""
        )
        runner_path.chmod(0o755)
        replicas.append(
            {
                "candidate_index": candidate_index,
                "episode_noise_base_seed": episode_base_seed,
                "common_future_noise_base_seed": COMMON_FUTURE_BASE_SEED,
                "config_path": str(config_path.resolve()),
                "config_sha256": _sha256(config_path),
                "prompt_manifest_path": str(manifest_path.resolve()),
                "prompt_manifest_sha256": manifest_sha256,
                "runner_path": str(runner_path.resolve()),
                "runner_log": str(runner_log.resolve()),
                "run_dir": str(run_dir.resolve()),
                "http_port": http_port,
                "server_gpu": candidate_index,
                "sim_gpu": candidate_index,
            }
        )

    plan = {
        "schema_version": 1,
        "kind": "common_future_outcome_g5_plan",
        "status": "READY",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "candidate_count": 5,
        "scene_count": 30,
        "train_scene_count": 24,
        "heldout_scene_count": 6,
        "total_rollouts": 150,
        "candidate_episode_noise_base_seeds": list(CANDIDATE_BASE_SEEDS),
        "common_future_noise_base_seed": COMMON_FUTURE_BASE_SEED,
        "selection_namespace": SELECTION_NAMESPACE,
        "manual_recovery_labels": 0,
        "task_specific_trigger_rules": 0,
        "reward": "RoboTwin terminal success boolean",
        "source_plan": str(source_plan_path.resolve()),
        "source_plan_sha256": _sha256(source_plan_path),
        "source_dataset_manifest": str(source_dataset_manifest_path.resolve()),
        "source_dataset_manifest_sha256": _sha256(source_dataset_manifest_path),
        "canonical_config": str(canonical_config_path.resolve()),
        "canonical_config_sha256": _sha256(canonical_config_path),
        "groups": groups,
        "replicas": replicas,
    }
    plan_path = output_dir / "common_future_plan.json"
    _write_json(plan_path, plan)

    launcher = output_dir / "launch_all.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd /home/zch/workspace/sana-wam"]
    for replica in replicas:
        index = replica["candidate_index"]
        lines.extend(
            [
                f"session=common_future_g5_c{index}_20260722",
                'tmux has-session -t "$session" 2>/dev/null && { echo "session exists: $session" >&2; exit 1; } || true',
                f"[[ ! -e \"{replica['run_dir']}\" && ! -e \"{replica['run_dir']}.run-lock\" ]]",
            ]
        )
    for replica in replicas:
        index = replica["candidate_index"]
        lines.extend(
            [
                f"tmux new-session -d -s common_future_g5_c{index}_20260722 -c /home/zch/workspace/sana-wam \"bash {replica['runner_path']} > {replica['runner_log']} 2>&1\"",
                f"echo launched_candidate_{index}",
            ]
        )
    launcher.write_text("\n".join(lines) + "\n")
    launcher.chmod(0o755)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_plan", type=Path)
    parser.add_argument("source_dataset_manifest", type=Path)
    parser.add_argument("canonical_config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    plan = build_plan(
        args.source_plan,
        args.source_dataset_manifest,
        args.canonical_config,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                key: plan[key]
                for key in (
                    "status",
                    "scene_count",
                    "train_scene_count",
                    "heldout_scene_count",
                    "candidate_count",
                    "total_rollouts",
                    "common_future_noise_base_seed",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
