#!/usr/bin/env python
"""Build a strict live-takeover plan from a validated 200xxx control run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

try:
    from scripts.build_robotwin_prompt_manifest import (
        build_manifest as rebuild_prompt_manifest,
    )
    from scripts.verify_paired_noise_telemetry import load_client_run, load_run
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from build_robotwin_prompt_manifest import (
        build_manifest as rebuild_prompt_manifest,
    )
    from verify_paired_noise_telemetry import load_client_run, load_run


CHECKPOINT_SHA256 = "aea6b58c9107aeeeffba561f39ffedfc3f8fd8420a4abfcab9038c8cdd4a129d"
ACTION_STATS_SHA256 = "2404b012a04f4e7d09e72fdfdddb4288099866a32f5e42e127cf4b5613df5cfc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} is not a lowercase SHA-256 digest")
    return value


def _load_json_object(path: Path, *, field: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"control run is missing {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to parse control {field}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"control {field} must be a JSON object")
    return value


def _validate_anchor(anchor_env_step: int, upstream_step_limit: int) -> None:
    if (
        isinstance(anchor_env_step, bool)
        or not isinstance(anchor_env_step, int)
        or anchor_env_step < 112
        or anchor_env_step % 28 != 0
        or anchor_env_step >= upstream_step_limit
    ):
        raise ValueError(
            "anchor_env_step must be a generation boundary >= 112 and below "
            f"the RoboTwin default limit {upstream_step_limit}"
        )


def _build_anchor_trigger(
    *,
    min_env_step: int,
    max_env_step: int,
    distance_m_at_most: float,
    upstream_step_limit: int,
) -> dict:
    _validate_anchor(min_env_step, upstream_step_limit)
    _validate_anchor(max_env_step, upstream_step_limit)
    if min_env_step > max_env_step:
        raise ValueError("anchor trigger min_env_step must not exceed max_env_step")
    if (
        isinstance(distance_m_at_most, bool)
        or not isinstance(distance_m_at_most, (int, float))
        or not math.isfinite(distance_m_at_most)
        or not 0.0 < float(distance_m_at_most) <= 0.5
    ):
        raise ValueError("anchor trigger distance must be finite in (0, 0.5]")
    return {
        "kind": "active_eef_bottle_proximity",
        "min_env_step": min_env_step,
        "max_env_step": max_env_step,
        "distance_m_at_most": float(distance_m_at_most),
    }


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"takeover output_dir is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise ValueError(f"takeover output_dir must be empty: {resolved}")
    return resolved


def _build_planned_episodes(episodes: list[dict], anchor_env_step: int) -> list[dict]:
    planned = []
    previous_seed = None
    for ordinal, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"prompt manifest episode {ordinal} is not an object")
        seed = episode.get("environment_seed")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 200000 <= seed < 300000
        ):
            raise ValueError(f"episode {ordinal} is not in the disjoint 200xxx range")
        if previous_seed is not None and seed <= previous_seed:
            raise ValueError("prompt manifest seeds must be strictly increasing")
        previous_seed = seed
        if (
            episode.get("task_name") != "adjust_bottle"
            or episode.get("task_config") != "demo_clean"
            or episode.get("noise_pair_key")
            != f"robotwin/adjust_bottle/demo_clean/scene-{seed}"
        ):
            raise ValueError(f"episode {ordinal} task/noise identity is invalid")
        prompt = episode.get("prompt")
        prompt_sha = episode.get("prompt_sha256")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or hashlib.sha256(prompt.encode()).hexdigest() != prompt_sha
        ):
            raise ValueError(f"episode {ordinal} prompt SHA-256 mismatch")
        if episode.get("ordinal") != ordinal or episode.get("episode_index") != ordinal:
            raise ValueError("prompt manifest ordinals must be contiguous")
        planned.append(
            {
                "anchor_id": f"adjust_bottle-seed{seed}-step{anchor_env_step:06d}",
                "episode_index": ordinal,
                "environment_seed": seed,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
                "anchor_env_step": anchor_env_step,
                "generation_index": anchor_env_step // 28,
            }
        )
    return planned


def _validate_control_run(
    control_run: Path,
    *,
    prompt_manifest: dict,
    anchor_env_step: int,
) -> tuple[dict, int, list[dict]]:
    eval_exit_path = control_run / "eval.exit"
    if not eval_exit_path.is_file() or eval_exit_path.read_text().strip() != "0":
        raise ValueError("control run must have an exact eval.exit status of 0")

    before = _load_json_object(
        control_run / "info.before.json", field="info.before.json"
    )
    after = _load_json_object(control_run / "info.after.json", field="info.after.json")
    before_runtime = before.get("inference_runtime")
    after_runtime = after.get("inference_runtime")
    if not isinstance(before_runtime, dict) or not isinstance(after_runtime, dict):
        raise ValueError("control info files are missing inference_runtime")
    identity = before_runtime.get("deployment_identity")
    if (
        not isinstance(identity, dict)
        or after_runtime.get("deployment_identity") != identity
    ):
        raise ValueError("control deployment identity changed during evaluation")
    if before.get("total_requests") != 0:
        raise ValueError("control info.before.json must precede every policy request")

    episodes = prompt_manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("control prompt manifest is invalid or empty")
    instruction_type = prompt_manifest.get("instruction_type")
    if not isinstance(instruction_type, str) or not instruction_type.strip():
        raise ValueError("control prompt manifest has no instruction_type")
    rebuilt_manifest = rebuild_prompt_manifest(
        control_run,
        expected_episodes=len(episodes),
        instruction_type=instruction_type,
    )
    if rebuilt_manifest != prompt_manifest:
        raise ValueError(
            "control prompt manifest does not exactly match validated server/client "
            "telemetry"
        )

    server = load_run(
        control_run, expected_mode="predicted", expected_episodes=len(episodes)
    )
    client = load_client_run(control_run, expected_episodes=len(episodes))
    if server["identity"] != identity:
        raise ValueError("control /info identity does not match server telemetry")
    last_episode_requests = len(server["steps"][-1])
    if after.get("total_requests") != last_episode_requests:
        raise ValueError(
            "control info.after.json request count does not match the final server "
            "episode"
        )
    final_episode_key = server["starts"][-1].get("episode_key")
    if after.get("episode", {}).get("key") != final_episode_key:
        raise ValueError(
            "control info.after.json episode does not match final server telemetry"
        )

    noise_base_seed = identity.get("episode_noise_base_seed")
    if isinstance(noise_base_seed, bool) or not isinstance(noise_base_seed, int):
        raise ValueError("control identity episode_noise_base_seed must be an integer")
    for label, runtime in (("before", before_runtime), ("after", after_runtime)):
        if runtime.get("cache_feedback_mode") != "predicted":
            raise ValueError(
                f"control {label} runtime cache feedback must be predicted"
            )
        if runtime.get("episode_noise_mode") != "paired":
            raise ValueError(f"control {label} runtime episode noise must be paired")
        if runtime.get("episode_noise_base_seed") != noise_base_seed:
            raise ValueError(
                f"control {label} runtime noise base seed does not match identity"
            )

    client_ends = {end.get("episode_key"): end for end in client["ends"]}
    control_outcomes = []
    for episode_index, (server_steps, client_steps) in enumerate(
        zip(server["steps"], client["steps"], strict=True)
    ):
        server_key = server["starts"][episode_index].get("episode_key")
        client_key = (
            client["starts"][episode_index].get("context", {}).get("episode_key")
        )
        if server_key != client_key or len(server_steps) != len(client_steps):
            raise ValueError(
                f"control episode {episode_index} server/client trajectory mismatch"
            )
        end = client_ends.get(client_key)
        if not isinstance(end, dict):
            raise ValueError(f"control episode {episode_index} has no client outcome")
        control_success = end.get("success")
        if not isinstance(control_success, bool):
            raise ValueError(f"control episode {episode_index} has invalid success")
        anchor_observed = len(server_steps) > anchor_env_step
        if not anchor_observed:
            if not control_success:
                raise ValueError(
                    f"failed control episode {episode_index} ended before anchor "
                    f"{anchor_env_step}"
                )
            control_outcomes.append(
                {
                    "control_success": True,
                    "control_step_count": len(server_steps),
                    "control_anchor_observed": False,
                }
            )
            continue
        server_anchor = server_steps[anchor_env_step]["record"]
        client_anchor = client_steps[anchor_env_step]["record"]
        expected_policy = {
            "policy_step": anchor_env_step,
            "generation_index": anchor_env_step // 28,
            "chunk_offset": 0,
            "generated": True,
        }
        policy = server_anchor.get("policy")
        if not isinstance(policy, dict) or any(
            policy.get(field) != expected for field, expected in expected_policy.items()
        ):
            raise ValueError(
                f"control episode {episode_index} has no valid generation-boundary "
                f"policy record at env_step {anchor_env_step}"
            )
        if client_anchor.get("server_step") != anchor_env_step + 1:
            raise ValueError(
                f"control episode {episode_index} client/server anchor is misaligned"
            )
        control_outcomes.append(
            {
                "control_success": control_success,
                "control_step_count": len(server_steps),
                "control_anchor_observed": True,
            }
        )
    return before, noise_base_seed, control_outcomes


def build_plan(
    control_run: Path,
    *,
    output_dir: Path,
    action_stats_path: Path,
    robotwin_path: Path,
    repo_step_limits_path: Path,
    anchor_env_step: int,
    anchor_trigger_distance_m: float | None = None,
    anchor_min_env_step: int = 112,
    capture_full_expert_suffix: bool = False,
) -> dict:
    control_run = control_run.expanduser().resolve()
    prompt_path = control_run / "prompt_manifest.json"
    prompt_manifest = _load_json_object(prompt_path, field="prompt_manifest.json")
    episodes = prompt_manifest.get("episodes")
    if (
        prompt_manifest.get("schema_version") != 1
        or prompt_manifest.get("kind") != "robotwin_prompt_replay"
        or not isinstance(episodes, list)
        or not episodes
    ):
        raise ValueError("control prompt manifest is invalid or empty")
    if prompt_manifest.get("source", {}).get("episode_count") != len(episodes):
        raise ValueError("control prompt manifest episode count is inconsistent")

    _validate_anchor(anchor_env_step, 400)
    anchor_trigger = None
    if anchor_trigger_distance_m is not None:
        anchor_trigger = _build_anchor_trigger(
            min_env_step=anchor_min_env_step,
            max_env_step=anchor_env_step,
            distance_m_at_most=anchor_trigger_distance_m,
            upstream_step_limit=400,
        )
    info, noise_base_seed, control_outcomes = _validate_control_run(
        control_run,
        prompt_manifest=prompt_manifest,
        anchor_env_step=anchor_env_step,
    )

    runtime = info.get("inference_runtime", {})
    identity = runtime.get("deployment_identity", {})
    exact_identity = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "history_len": 113,
        "num_frames": 113,
        "video_num_frames": 29,
        "video_stride": 4,
        "action_mode": "eef",
        "normalize_mode": "min-max",
        "action_tokens_per_chunk": 28,
        "ar_frame_chunk_size": 2,
        "cache_feedback_mode": "predicted",
        "episode_noise_mode": "paired",
    }
    for field, expected in exact_identity.items():
        if identity.get(field) != expected:
            raise ValueError(
                f"control deployment_identity.{field} must be {expected!r}, "
                f"got {identity.get(field)!r}"
            )
    deploy_config_sha = _require_sha(
        identity.get("deploy_config_sha256"), "deploy_config_sha256"
    )
    deploy_overrides_sha = _require_sha(
        identity.get("deploy_overrides_sha256"), "deploy_overrides_sha256"
    )

    action_stats_path = action_stats_path.expanduser().resolve()
    if not action_stats_path.is_file():
        raise FileNotFoundError(f"action stats not found: {action_stats_path}")
    if _sha256(action_stats_path) != ACTION_STATS_SHA256:
        raise ValueError("action stats do not match the fixed baseline SHA-256")

    repo_step_limits_path = repo_step_limits_path.expanduser().resolve()
    if yaml.safe_load(repo_step_limits_path.read_text(encoding="utf-8")) is not None:
        raise ValueError("repository step_limits.yml contains an active override")
    upstream_limits_path = (
        robotwin_path.expanduser().resolve() / "task_config" / "_eval_step_limit.yml"
    )
    upstream_limits = yaml.safe_load(upstream_limits_path.read_text(encoding="utf-8"))
    if upstream_limits.get("adjust_bottle") != 400:
        raise ValueError("RoboTwin upstream adjust_bottle step limit must be 400")

    _validate_anchor(anchor_env_step, upstream_limits["adjust_bottle"])
    output_dir = _validate_output_dir(output_dir)
    planned_episodes = _build_planned_episodes(episodes, anchor_env_step)
    for entry, outcome in zip(planned_episodes, control_outcomes, strict=True):
        entry.update(outcome)

    plan = {
        "schema_version": 1,
        "kind": "robotwin_live_takeover_plan",
        "source_kind": "fresh_on_policy_history113_dagger",
        "output_dir": str(output_dir),
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "policy_history_len": 113,
        "action_tokens_per_chunk": 28,
        "video_stride": 4,
        "temporal_compression": 4,
        "ar_frame_chunk_size": 2,
        "expert_save_freq": 15,
        "cache_feedback_mode": "predicted",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "deploy_config_sha256": deploy_config_sha,
        "deploy_overrides_sha256": deploy_overrides_sha,
        "prompt_manifest_sha256": _sha256(prompt_path),
        "episode_noise_mode": "paired",
        "episode_noise_base_seed": noise_base_seed,
        "environment_seed_index": 1,
        "action_stats_path": str(action_stats_path),
        "action_stats_sha256": ACTION_STATS_SHA256,
        "step_limit_source": "robotwin_upstream_default",
        "repo_step_limit_override_count": 0,
        "upstream_step_limit": 400,
        "episodes": planned_episodes,
        "metadata": {
            "control_run": str(control_run),
            "control_prompt_manifest": str(prompt_path),
            "control_prompt_manifest_sha256": _sha256(prompt_path),
        },
    }
    if anchor_trigger is not None:
        plan["anchor_trigger"] = anchor_trigger
    if capture_full_expert_suffix:
        plan["full_expert_suffix"] = {"enabled": True}
    return plan


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control_run", type=Path)
    parser.add_argument("output_plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--anchor-env-step",
        type=int,
        default=112,
        help="fixed anchor step, or the trigger deadline when proximity mode is enabled",
    )
    parser.add_argument(
        "--anchor-trigger-distance-m",
        type=float,
        help="enable active-EEF/bottle proximity triggering at generation boundaries",
    )
    parser.add_argument(
        "--anchor-min-env-step",
        type=int,
        default=112,
        help="first eligible boundary for proximity triggering",
    )
    parser.add_argument(
        "--capture-full-expert-suffix",
        action="store_true",
        help="capture the complete successful expert suffix as standard RoboTwin episodes",
    )
    parser.add_argument(
        "--action-stats-path",
        type=Path,
        default=Path(
            "/home/zch/wuji-openwam-dev/sandbox/sana_ar_graft_AC_lownoise_only/"
            "2026-07-04_13-03-25/action_stats.npy"
        ),
    )
    parser.add_argument(
        "--robotwin-path",
        type=Path,
        default=Path(os.environ.get("ROBOTWIN_PATH", "/home/zch/RoboTwin")),
    )
    parser.add_argument(
        "--repo-step-limits-path",
        type=Path,
        default=root / "benchmarks" / "robotwin" / "step_limits.yml",
    )
    args = parser.parse_args()
    output_plan = args.output_plan.expanduser().resolve()
    if output_plan.exists():
        raise FileExistsError(f"refusing to overwrite takeover plan: {output_plan}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_plan.is_relative_to(output_dir):
        raise ValueError("output_plan must not be written inside takeover output_dir")
    plan = build_plan(
        args.control_run,
        output_dir=output_dir,
        action_stats_path=args.action_stats_path,
        robotwin_path=args.robotwin_path,
        repo_step_limits_path=args.repo_step_limits_path,
        anchor_env_step=args.anchor_env_step,
        anchor_trigger_distance_m=args.anchor_trigger_distance_m,
        anchor_min_env_step=args.anchor_min_env_step,
        capture_full_expert_suffix=args.capture_full_expert_suffix,
    )
    encoded = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_plan.with_name(f".{output_plan.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_plan)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"takeover_plan: PASS episodes={len(plan['episodes'])} "
        f"sha256={hashlib.sha256(encoded).hexdigest()} output={output_plan}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
