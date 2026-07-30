#!/usr/bin/env python
"""Build an exact per-scene prompt manifest from a validated control run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    from scripts.verify_paired_noise_telemetry import load_client_run, load_run
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from verify_paired_noise_telemetry import load_client_run, load_run


SCHEMA_VERSION = 1


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_manifest(
    source: Path,
    *,
    minimum_episodes: int | None = None,
    expected_episodes: int | None = None,
    instruction_type: str = "unseen",
) -> dict:
    control = load_run(
        source, expected_mode="predicted", expected_episodes=expected_episodes
    )
    episode_count = len(control["pairs"])
    if minimum_episodes is not None and episode_count < minimum_episodes:
        raise ValueError(
            f"control has only {episode_count} episodes, need at least {minimum_episodes}"
        )
    client = load_client_run(source, expected_episodes=episode_count)
    expected_pairs = [pair[:4] for pair in control["pairs"]]
    if client["pairs"] != expected_pairs:
        raise ValueError("control client/server episode identities do not match")

    scene_ids = []
    for task_name, task_config, environment_seed, noise_pair_key in client["pairs"]:
        scene = (task_name, task_config, environment_seed)
        if scene in scene_ids:
            raise ValueError(f"duplicate control scene identity: {scene!r}")
        scene_ids.append(scene)
        canonical_pair_key = (
            f"robotwin/{task_name}/{task_config}/scene-{environment_seed}"
        )
        if noise_pair_key != canonical_pair_key:
            raise ValueError(
                f"noise_pair_key canonical mismatch for scene {scene!r}: "
                f"{noise_pair_key!r} != {canonical_pair_key!r}"
            )

    episodes = []
    for ordinal, (pair, prompt, generation_zero) in enumerate(
        zip(client["pairs"], client["prompts"], control["generation_zero"])
    ):
        task_name, task_config, environment_seed, noise_pair_key = pair
        raw = generation_zero["predicted_actions"]
        normalized = generation_zero["predicted_actions_normalized"]
        episodes.append(
            {
                "ordinal": ordinal,
                "task_name": task_name,
                "task_config": task_config,
                "environment_seed": environment_seed,
                "episode_index": control["starts"][ordinal]
                .get("context", {})
                .get("episode_index"),
                "noise_pair_key": noise_pair_key,
                "model_noise_seed": control["pairs"][ordinal][4],
                "source_episode_key": control["starts"][ordinal]["episode_key"],
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "generation_zero": {
                    "shape": list(raw.shape),
                    "dtype": str(raw.dtype),
                    "predicted_actions_sha256": _array_sha256(raw),
                    "predicted_actions_normalized_sha256": _array_sha256(normalized),
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "robotwin_prompt_replay",
        "instruction_type": instruction_type,
        "source": {
            "run_path": str(source.expanduser().resolve()),
            "run_id": control["starts"][0].get("run_id"),
            "checkpoint_sha256": control["identity"]["checkpoint_sha256"],
            "episode_noise_base_seed": control["identity"]["episode_noise_base_seed"],
            "episode_count": episode_count,
        },
        "episodes": episodes,
    }


def write_manifest(payload: dict, output: Path) -> str:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prompt manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="validated predicted control run")
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-episodes", type=int)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--instruction-type", default="unseen")
    args = parser.parse_args()
    if args.minimum_episodes is not None and args.minimum_episodes < 1:
        parser.error("--minimum-episodes must be positive")
    if args.expected_episodes is not None and args.expected_episodes < 1:
        parser.error("--expected-episodes must be positive")

    payload = build_manifest(
        args.source,
        minimum_episodes=args.minimum_episodes,
        expected_episodes=args.expected_episodes,
        instruction_type=args.instruction_type,
    )
    digest = write_manifest(payload, args.output)
    print(
        f"prompt_manifest: PASS episodes={len(payload['episodes'])} sha256={digest} "
        f"output={args.output.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
