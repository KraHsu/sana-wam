#!/usr/bin/env python
"""Validate two complete telemetry runs and their ordered scene/noise pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
VALID_MODES = ("predicted", "reencode_predicted", "measured")
FEEDBACK_CONTRACTS = {
    "reencode_predicted": {
        "status": "reencoded_predicted",
        "reason": "previous_predicted_chunk",
        "raw_key": "reencoded_predicted_actions",
        "normalized_key": "reencoded_predicted_actions_normalized",
    },
    "measured": {
        "status": "measured",
        "reason": "achieved_state_history",
        "raw_key": "measured_feedback_actions",
        "normalized_key": "measured_feedback_actions_normalized",
    },
}


def _same_contiguous_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare the exact C-order representation, including signed zero bits."""
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def _telemetry_dir(path: Path) -> Path:
    if (path / "telemetry" / "server" / "episodes.jsonl").is_file():
        return path / "telemetry" / "server"
    return path


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: unsupported schema {record.get('schema_version')!r}"
                )
            records.append(record)
    return records


def _pair_from_start(event: dict) -> tuple:
    context = event.get("context", {})
    runtime = event.get("runtime", {})
    return (
        context.get("task_name"),
        context.get("task_config"),
        context.get("environment_seed"),
        runtime.get("current_noise_pair_key"),
        runtime.get("current_model_noise_seed"),
    )


def _chunk_shape(identity: dict, root: Path) -> tuple[int, int]:
    tokens = identity.get("action_tokens_per_chunk")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError(f"invalid action_tokens_per_chunk in {root}: {tokens!r}")
    action_mode = identity.get("action_mode")
    try:
        action_dim = {"eef": 20, "joint": 14}[action_mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported action_mode in {root}: {action_mode!r}") from exc
    return tokens, action_dim


def _validate_chunk(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    measured_required: bool | None = None,
    feedback_mode: str | None = None,
) -> dict[str, np.ndarray]:
    # Preserve the helper's original API for callers that only know measured mode.
    if measured_required is not None:
        legacy_mode = "measured" if measured_required else None
        if feedback_mode is not None and feedback_mode != legacy_mode:
            raise ValueError("conflicting feedback-mode requirements")
        feedback_mode = legacy_mode
    if feedback_mode is not None and feedback_mode not in FEEDBACK_CONTRACTS:
        raise ValueError(f"unsupported chunk feedback mode: {feedback_mode!r}")

    if not path.is_file():
        raise ValueError(f"referenced chunk does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as chunk:
            schema = chunk["schema_version"]
            if (
                schema.shape != ()
                or schema.dtype != np.dtype(np.int16)
                or int(schema) != SCHEMA_VERSION
            ):
                raise ValueError(f"{path}: bad NPZ schema")
            predicted = chunk["predicted_actions"]
            predicted_norm = chunk["predicted_actions_normalized"]
            for name, array in (
                ("predicted_actions", predicted),
                ("predicted_actions_normalized", predicted_norm),
            ):
                if array.shape != expected_shape:
                    raise ValueError(
                        f"{path}: invalid {name} shape {array.shape}, "
                        f"expected {expected_shape}"
                    )
                if array.dtype != np.dtype(np.float32):
                    raise ValueError(
                        f"{path}: invalid {name} dtype {array.dtype}, expected float32"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"{path}: non-finite {name}")
            validated = {
                "predicted_actions": predicted.copy(),
                "predicted_actions_normalized": predicted_norm.copy(),
            }
            all_feedback_keys = {
                contract[key]
                for contract in FEEDBACK_CONTRACTS.values()
                for key in ("raw_key", "normalized_key")
            }
            present = all_feedback_keys.intersection(chunk.files)
            expected_keys = set()
            if feedback_mode is not None:
                contract = FEEDBACK_CONTRACTS[feedback_mode]
                expected_keys = {contract["raw_key"], contract["normalized_key"]}
            if feedback_mode is not None and present != expected_keys:
                label = "measured" if feedback_mode == "measured" else feedback_mode
                raise ValueError(f"{path}: complete {label} feedback is missing")
            if feedback_mode is None and present:
                label = (
                    "measured"
                    if present
                    <= {
                        "measured_feedback_actions",
                        "measured_feedback_actions_normalized",
                    }
                    else "cache"
                )
                raise ValueError(f"{path}: unexpected {label} feedback")
            if feedback_mode is not None:
                for name in sorted(expected_keys):
                    array = chunk[name]
                    if array.shape != expected_shape:
                        raise ValueError(
                            f"{path}: invalid {name} shape {array.shape}, "
                            f"expected {expected_shape}"
                        )
                    if array.dtype != np.dtype(np.float32):
                        raise ValueError(
                            f"{path}: invalid {name} dtype {array.dtype}, "
                            "expected float32"
                        )
                    if not np.isfinite(array).all():
                        raise ValueError(f"{path}: non-finite {name}")
                    validated[name] = array.copy()
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(str(path)):
            raise
        raise ValueError(f"cannot validate chunk {path}: {exc}") from exc
    return validated


def load_run(
    path: Path, *, expected_mode: str, expected_episodes: int | None = None
) -> dict:
    if expected_mode not in VALID_MODES:
        raise ValueError(f"unsupported cache-feedback mode: {expected_mode!r}")
    root = _telemetry_dir(path)
    episode_records = _read_jsonl(root / "episodes.jsonl")
    step_records = _read_jsonl(root / "steps.jsonl")
    if any(
        record.get("producer") != "server" for record in episode_records + step_records
    ):
        raise ValueError(f"non-server record in {root}")

    starts = [
        record for record in episode_records if record.get("event") == "episode_start"
    ]
    ends = [
        record for record in episode_records if record.get("event") == "episode_end"
    ]
    if len(starts) != len(ends) or not starts:
        raise ValueError(
            f"incomplete server episodes in {root}: starts={len(starts)} ends={len(ends)}"
        )
    if expected_episodes is not None and len(starts) != expected_episodes:
        raise ValueError(
            f"episode count mismatch in {root}: {len(starts)} != {expected_episodes}"
        )
    starts.sort(key=lambda record: int(record["episode_number"]))
    ends.sort(key=lambda record: int(record["episode_number"]))
    if [record["episode_number"] for record in starts] != list(range(len(starts))):
        raise ValueError(f"non-contiguous episode numbers in {root}")
    for start, end in zip(starts, ends):
        if (
            start["episode_number"] != end["episode_number"]
            or start["episode_key"] != end["episode_key"]
        ):
            raise ValueError(f"episode start/end identity mismatch in {root}")
        start_runtime = start.get("runtime", {})
        end_runtime = end.get("runtime", {})
        if start_runtime.get("cache_feedback_mode") != expected_mode:
            raise ValueError(f"{root}: expected mode {expected_mode!r}")
        fallbacks = int(end_runtime.get("cache_feedback_fallbacks", -1))
        commits = int(end_runtime.get("cache_feedback_commits", -1))
        ar_steps = int(end_runtime.get("ar_step", -1))
        if fallbacks != 0 or ar_steps < 1:
            raise ValueError(
                f"{root}: invalid end runtime fallbacks={fallbacks} ar_step={ar_steps}"
            )
        expected_commits = (
            max(0, ar_steps - 1) if expected_mode in FEEDBACK_CONTRACTS else 0
        )
        if commits != expected_commits:
            raise ValueError(
                f"{root}: commits={commits}, expected {expected_commits} for ar_step={ar_steps}"
            )

    pairs = [_pair_from_start(start) for start in starts]
    if any(pair[2] is None or pair[3] is None or pair[4] is None for pair in pairs):
        raise ValueError(f"missing environment/pair/model seed in {root}")
    if len({pair[3] for pair in pairs}) != len(pairs):
        raise ValueError(f"noise_pair_key is not unique per episode in {root}")
    if len({pair[4] for pair in pairs}) != len(pairs):
        raise ValueError(f"model_noise_seed is not unique per episode in {root}")

    identity = starts[0].get("runtime", {}).get("deployment_identity", {})
    required_identity = {
        "telemetry_schema_version",
        "engine_class",
        "architecture_class",
        "checkpoint_path",
        "checkpoint_size",
        "checkpoint_sha256",
        "history_len",
        "video_steps",
        "action_steps",
        "episode_noise_mode",
        "episode_noise_base_seed",
        "cache_feedback_mode",
        "action_tokens_per_chunk",
        "action_mode",
    }
    if required_identity.difference(identity) or any(
        identity.get(key) is None for key in required_identity
    ):
        raise ValueError(f"incomplete deployment identity in {root}")
    if any(
        start.get("runtime", {}).get("deployment_identity") != identity
        for start in starts
    ):
        raise ValueError(f"deployment identity changed within {root}")
    if identity["cache_feedback_mode"] != expected_mode:
        raise ValueError(f"identity mode mismatch in {root}")
    expected_chunk_shape = _chunk_shape(identity, root)

    start_by_key = {start["episode_key"]: start for start in starts}
    generated_by_episode = {episode_key: 0 for episode_key in start_by_key}
    generation_zero_by_episode = {}
    steps_by_episode = {episode_key: [] for episode_key in start_by_key}
    generations_by_episode = {episode_key: [] for episode_key in start_by_key}
    for step in step_records:
        episode_key = step.get("episode_key")
        if episode_key not in generated_by_episode:
            raise ValueError(
                f"step references unknown episode {episode_key!r} in {root}"
            )
        start = start_by_key[episode_key]
        if step.get("episode_number") != start["episode_number"]:
            raise ValueError(f"server step episode number mismatch in {root}")
        request_id = step.get("request_id")
        expected_request_id = len(steps_by_episode[episode_key]) + 1
        if (
            isinstance(request_id, bool)
            or not isinstance(request_id, int)
            or request_id != expected_request_id
        ):
            raise ValueError(
                f"non-contiguous server request_id in {root}: "
                f"{request_id!r} != {expected_request_id}"
            )
        observed = _numeric_vector(
            step, "observed_proprio", expected_chunk_shape[1], root
        )
        returned = _numeric_vector(
            step, "returned_action", expected_chunk_shape[1], root
        )
        server_step = {
            "record": step,
            "request_id": request_id,
            "observed_proprio": observed,
            "returned_action": returned,
        }
        steps_by_episode[episode_key].append(server_step)

        policy = step.get("policy", {})
        generated = policy.get("generated")
        if not isinstance(generated, bool):
            raise ValueError(f"invalid generated flag in {root}: {generated!r}")
        chunk_path = step.get("chunk_npz")
        if not generated:
            if chunk_path is not None:
                raise ValueError(f"non-generation step references chunk in {root}")
            continue
        generation = policy.get("generation_index")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError(f"invalid generation index in {root}: {generation!r}")
        if generation != generated_by_episode[episode_key]:
            raise ValueError(f"non-contiguous generation index in {root}")
        if chunk_path is None:
            raise ValueError(f"generation has no NPZ in {root}")
        feedback_mode = (
            expected_mode
            if expected_mode in FEEDBACK_CONTRACTS and generation > 0
            else None
        )
        validated = _validate_chunk(
            root / chunk_path,
            feedback_mode=feedback_mode,
            expected_shape=expected_chunk_shape,
        )
        action_frame_id = policy.get("action_frame_id")
        if isinstance(action_frame_id, bool) or not isinstance(action_frame_id, int):
            raise ValueError(f"invalid generation action_frame_id in {root}")
        expected_action_frame_id = 1 if generation == 0 else 2 * generation + 3
        if action_frame_id != expected_action_frame_id:
            raise ValueError(
                f"invalid generation action_frame_id in {root}: "
                f"{action_frame_id} != {expected_action_frame_id}"
            )
        feedback = policy.get("cache_feedback", {})
        if not isinstance(feedback, dict):
            raise ValueError(f"invalid cache-feedback metadata in {root}")
        generation_record = {
            **validated,
            "generation_index": generation,
            "request_id": request_id,
            "action_frame_id": action_frame_id,
            "feedback_frame_id": feedback.get("frame_id"),
            "feedback_source_generation_index": feedback.get("source_generation_index"),
        }
        generations_by_episode[episode_key].append(generation_record)
        if generation == 0:
            if expected_mode in FEEDBACK_CONTRACTS:
                if (
                    feedback.get("status") != "skipped"
                    or feedback.get("reason") != "episode_start"
                ):
                    raise ValueError(
                        f"invalid generation-zero {expected_mode} feedback in {root}"
                    )
            elif (
                feedback.get("status") != "predicted"
                or feedback.get("reason") != "configured"
            ):
                raise ValueError(f"invalid predicted feedback status in {root}")
            generation_record["observed_proprio"] = observed
            generation_zero_by_episode[episode_key] = generation_record
        elif expected_mode in FEEDBACK_CONTRACTS:
            contract = FEEDBACK_CONTRACTS[expected_mode]
            if (
                feedback.get("status") != contract["status"]
                or feedback.get("reason") != contract["reason"]
            ):
                raise ValueError(
                    f"missing {expected_mode} cache-feedback status in {root}"
                )
            if expected_mode == "reencode_predicted":
                source_generation = feedback.get("source_generation_index")
                if (
                    isinstance(source_generation, bool)
                    or not isinstance(source_generation, int)
                    or source_generation != generation - 1
                ):
                    raise ValueError(
                        f"invalid reencode source_generation_index in {root}: "
                        f"{source_generation!r} != {generation - 1}"
                    )
        elif (
            feedback.get("status") != "predicted"
            or feedback.get("reason") != "configured"
        ):
            raise ValueError(f"invalid predicted feedback status in {root}")
        generated_by_episode[episode_key] += 1
    for end in ends:
        expected = int(end["runtime"]["ar_step"])
        if generated_by_episode[end["episode_key"]] != expected:
            raise ValueError(f"generation/ar_step mismatch in {root}")
    if generation_zero_by_episode.keys() != generated_by_episode.keys():
        raise ValueError(f"missing generation-zero chunk in {root}")

    reported_commits = sum(
        int(end.get("runtime", {}).get("cache_feedback_commits", -1)) for end in ends
    )
    feedback_generations = sum(
        max(0, len(generations_by_episode[start["episode_key"]]) - 1)
        for start in starts
    )
    expected_total_commits = (
        feedback_generations if expected_mode in FEEDBACK_CONTRACTS else 0
    )
    if reported_commits != expected_total_commits:
        raise ValueError(
            f"{root}: total commits={reported_commits}, expected "
            f"{expected_total_commits} feedback generations"
        )

    return {
        "root": root,
        "pairs": pairs,
        "identity": identity,
        "starts": starts,
        "ends": ends,
        "generation_zero": [
            generation_zero_by_episode[start["episode_key"]] for start in starts
        ],
        "steps": [steps_by_episode[start["episode_key"]] for start in starts],
        "generations": [
            generations_by_episode[start["episode_key"]] for start in starts
        ],
        "reported_commits": reported_commits,
    }


def _numeric_vector(record: dict, field: str, size: int, root: Path) -> np.ndarray:
    value = record.get(field)
    array = np.asarray(value)
    if array.shape != (size,) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"invalid {field} in {root}: expected numeric vector length {size}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite {field} in {root}")
    return array.astype(np.float32, copy=True)


def _validate_client_step(step: dict, root: Path) -> dict[str, np.ndarray | dict]:
    spaces = (
        step.get("pre_action_space"),
        step.get("server_action_space"),
        step.get("sent_action_space"),
        step.get("post_action_space"),
    )
    if spaces == ("eef20", "eef20", "ee16", "eef20"):
        vector_sizes = {
            "pre_proprio": 20,
            "server_action": 20,
            "policy_action": 20,
            "sent_action": 16,
            "post_proprio": 20,
        }
    elif spaces == ("qpos14", "qpos14", "qpos14", "qpos14"):
        vector_sizes = {
            "pre_proprio": 14,
            "server_action": 14,
            "policy_action": 14,
            "sent_action": 14,
            "post_proprio": 14,
        }
    else:
        raise ValueError(f"invalid client action-space metadata in {root}: {spaces}")

    validated: dict[str, np.ndarray | dict] = {"record": step}
    for field, size in vector_sizes.items():
        validated[field] = _numeric_vector(step, field, size, root)
    for field in (
        "pre_gripper",
        "server_action_gripper",
        "sent_action_gripper",
        "post_gripper",
    ):
        _numeric_vector(step, field, 2, root)
    if step.get("post_state_error") is not None:
        raise ValueError(f"client post-state capture failed in {root}")
    return validated


def load_client_run(path: Path, *, expected_episodes: int | None) -> dict:
    root = path
    if (path / "telemetry" / "client" / "episodes.jsonl").is_file():
        root = path / "telemetry" / "client"
    episodes = _read_jsonl(root / "episodes.jsonl")
    steps = _read_jsonl(root / "steps.jsonl")
    if any(record.get("producer") != "robotwin_client" for record in episodes + steps):
        raise ValueError(f"non-client record in {root}")
    starts = [record for record in episodes if record.get("event") == "episode_start"]
    ends = [record for record in episodes if record.get("event") == "episode_end"]
    if len(starts) != len(ends) or not starts:
        raise ValueError(
            f"incomplete client episodes in {root}: starts={len(starts)} ends={len(ends)}"
        )
    if expected_episodes is not None and len(starts) != expected_episodes:
        raise ValueError(
            f"client episode count mismatch in {root}: "
            f"{len(starts)} != {expected_episodes}"
        )
    start_by_key = {}
    for start in starts:
        episode_key = start.get("context", {}).get("episode_key")
        if (
            not isinstance(episode_key, str)
            or not episode_key
            or episode_key in start_by_key
        ):
            raise ValueError(f"invalid or duplicate client episode key in {root}")
        start_by_key[episode_key] = start
    end_by_key = {}
    for end in ends:
        episode_key = end.get("episode_key")
        if (
            not isinstance(episode_key, str)
            or not episode_key
            or episode_key in end_by_key
        ):
            raise ValueError(f"invalid or duplicate client episode end key in {root}")
        end_by_key[episode_key] = end
    if start_by_key.keys() != end_by_key.keys():
        raise ValueError(f"client episode start/end identity mismatch in {root}")

    steps_by_key = {episode_key: [] for episode_key in start_by_key}
    for step in steps:
        episode_key = step.get("episode_key")
        if episode_key not in steps_by_key:
            raise ValueError(
                f"client step references unknown episode {episode_key!r} in {root}"
            )
        expected_seed = (
            start_by_key[episode_key].get("context", {}).get("environment_seed")
        )
        if step.get("environment_seed") != expected_seed:
            raise ValueError(f"client step environment seed mismatch in {root}")
        env_step = step.get("env_step")
        expected_env_step = len(steps_by_key[episode_key])
        if (
            isinstance(env_step, bool)
            or not isinstance(env_step, int)
            or env_step != expected_env_step
        ):
            raise ValueError(
                f"non-contiguous client env_step in {root}: "
                f"{env_step!r} != {expected_env_step}"
            )
        server_step = step.get("server_step")
        if isinstance(server_step, bool) or not isinstance(server_step, int):
            raise ValueError(f"invalid client server_step in {root}: {server_step!r}")
        steps_by_key[episode_key].append(_validate_client_step(step, root))

    prompts_by_key = {}
    for episode_key, start in start_by_key.items():
        end = end_by_key[episode_key]
        if start.get("context", {}).get("environment_seed") != end.get(
            "environment_seed"
        ):
            raise ValueError(f"client episode environment seed mismatch in {root}")
        if (
            not isinstance(end.get("success"), bool)
            or int(end.get("step_count", -1)) < 1
        ):
            raise ValueError(f"invalid client episode outcome in {root}")
        if len(steps_by_key[episode_key]) != int(end["step_count"]):
            raise ValueError(
                f"client step count mismatch for episode {episode_key!r} in {root}: "
                f"{len(steps_by_key[episode_key])} != {end['step_count']}"
            )
        prompt_values = {
            step["record"].get("prompt") for step in steps_by_key[episode_key]
        }
        if len(prompt_values) != 1:
            raise ValueError(
                f"client episode {episode_key!r} has multiple prompts in {root}"
            )
        prompt = next(iter(prompt_values))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"client episode {episode_key!r} has an empty prompt in {root}"
            )
        context = start.get("context", {})
        if context.get("prompt") is not None and context["prompt"] != prompt:
            raise ValueError(f"client episode prompt/context mismatch in {root}")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if (
            context.get("prompt_sha256") is not None
            and context["prompt_sha256"] != digest
        ):
            raise ValueError(f"client episode prompt SHA-256 mismatch in {root}")
        prompts_by_key[episode_key] = prompt
    pairs = [
        (
            start.get("context", {}).get("task_name"),
            start.get("context", {}).get("task_config"),
            start.get("context", {}).get("environment_seed"),
            start.get("context", {}).get("noise_pair_key"),
        )
        for start in starts
    ]
    return {
        "root": root,
        "pairs": pairs,
        "starts": starts,
        "ends": ends,
        "prompts": [
            prompts_by_key[start.get("context", {})["episode_key"]] for start in starts
        ],
        "steps": [
            steps_by_key[start.get("context", {})["episode_key"]] for start in starts
        ],
    }


def _prefix_run(run: dict, count: int) -> dict:
    if len(run["pairs"]) < count:
        raise ValueError(f"control has {len(run['pairs'])} episodes, need {count}")
    prefixed = dict(run)
    for key in (
        "pairs",
        "starts",
        "ends",
        "generation_zero",
        "prompts",
        "steps",
        "generations",
    ):
        if key in prefixed:
            prefixed[key] = prefixed[key][:count]
    if "reported_commits" in prefixed:
        prefixed["reported_commits"] = sum(
            int(end.get("runtime", {}).get("cache_feedback_commits", -1))
            for end in prefixed["ends"]
        )
    return prefixed


def _require_generation_zero_equal(control: dict, treatment: dict) -> None:
    for index, (control_chunk, treatment_chunk) in enumerate(
        zip(control["generation_zero"], treatment["generation_zero"])
    ):
        for field in (
            "observed_proprio",
            "predicted_actions",
            "predicted_actions_normalized",
        ):
            if not _same_contiguous_bytes(control_chunk[field], treatment_chunk[field]):
                raise SystemExit(
                    f"generation-zero {field} mismatch at episode {index} "
                    f"scene={control['pairs'][index][:3]!r}"
                )


def _require_server_client_mapping(server: dict, client: dict, *, label: str) -> None:
    if len(server["steps"]) != len(client["steps"]):
        raise SystemExit(f"{label} server/client episode counts do not match")
    for episode_index, (server_steps, client_steps) in enumerate(
        zip(server["steps"], client["steps"])
    ):
        server_key = server["starts"][episode_index].get("episode_key")
        client_key = (
            client["starts"][episode_index].get("context", {}).get("episode_key")
        )
        if server_key != client_key:
            raise SystemExit(
                f"{label} server/client episode_key mismatch at episode {episode_index}"
            )
        if len(server_steps) != len(client_steps):
            raise SystemExit(
                f"{label} server/client request count mismatch at episode "
                f"{episode_index}: {len(server_steps)} != {len(client_steps)}"
            )
        for offset, (server_step, client_step) in enumerate(
            zip(server_steps, client_steps)
        ):
            request_id = offset + 1
            client_record = client_step["record"]
            if (
                server_step["request_id"] != request_id
                or client_record.get("server_step") != request_id
            ):
                raise SystemExit(
                    f"{label} server/client request mapping mismatch at episode "
                    f"{episode_index} request={request_id}"
                )
            if not _same_contiguous_bytes(
                server_step["observed_proprio"], client_step["pre_proprio"]
            ):
                raise SystemExit(
                    f"{label} server observed/client pre mismatch at episode "
                    f"{episode_index} request={request_id}"
                )
            if not _same_contiguous_bytes(
                server_step["returned_action"], client_step["server_action"]
            ):
                raise SystemExit(
                    f"{label} server returned/client action mismatch at episode "
                    f"{episode_index} request={request_id}"
                )
            if offset + 1 < len(client_steps) and not _same_contiguous_bytes(
                client_step["post_proprio"],
                client_steps[offset + 1]["pre_proprio"],
            ):
                raise SystemExit(
                    f"{label} client post/next pre mismatch at episode "
                    f"{episode_index} request={request_id}"
                )


def _require_measured_feedback_mapping(treatment: dict, client: dict) -> None:
    action_tokens = treatment["identity"]["action_tokens_per_chunk"]
    matched_commits = 0
    for episode_index, (generations, client_steps) in enumerate(
        zip(treatment["generations"], client["steps"])
    ):
        for previous, current in zip(generations, generations[1:]):
            previous_request = previous["request_id"]
            current_request = current["request_id"]
            if current_request - previous_request != action_tokens:
                raise SystemExit(
                    f"measured generation request gap mismatch at episode "
                    f"{episode_index} generation={current['generation_index']}: "
                    f"{current_request - previous_request} != {action_tokens}"
                )
            covered_steps = client_steps[previous_request - 1 : current_request - 1]
            if len(covered_steps) != action_tokens:
                raise SystemExit(
                    f"measured generation client coverage mismatch at episode "
                    f"{episode_index} generation={current['generation_index']}"
                )
            expected_feedback = np.stack(
                [step["post_proprio"] for step in covered_steps], axis=0
            ).astype(np.float32, copy=False)
            measured_feedback = current.get("measured_feedback_actions")
            if measured_feedback is None or not _same_contiguous_bytes(
                measured_feedback, expected_feedback
            ):
                raise SystemExit(
                    f"measured feedback/client post mismatch at episode "
                    f"{episode_index} generation={current['generation_index']}"
                )
            feedback_frame_id = current.get("feedback_frame_id")
            if (
                isinstance(feedback_frame_id, bool)
                or not isinstance(feedback_frame_id, int)
                or feedback_frame_id != previous["action_frame_id"]
            ):
                raise SystemExit(
                    f"measured feedback frame_id mismatch at episode "
                    f"{episode_index} generation={current['generation_index']}"
                )
            matched_commits += 1
    if matched_commits != treatment["reported_commits"]:
        raise SystemExit(
            f"measured feedback commit total mismatch: matched={matched_commits} "
            f"reported={treatment['reported_commits']}"
        )


def _require_reencoded_predicted_mapping(run: dict, *, label: str) -> None:
    """Prove every refreshed cache frame reuses the previous predicted bytes."""
    action_tokens = run["identity"]["action_tokens_per_chunk"]
    contract = FEEDBACK_CONTRACTS["reencode_predicted"]
    matched_commits = 0
    for episode_index, generations in enumerate(run["generations"]):
        for previous, current in zip(generations, generations[1:]):
            generation = current["generation_index"]
            request_gap = current["request_id"] - previous["request_id"]
            if request_gap != action_tokens:
                raise SystemExit(
                    f"{label} reencode generation request gap mismatch at episode "
                    f"{episode_index} generation={generation}: "
                    f"{request_gap} != {action_tokens}"
                )
            if current.get("feedback_frame_id") != previous["action_frame_id"]:
                raise SystemExit(
                    f"{label} reencode feedback frame_id mismatch at episode "
                    f"{episode_index} generation={generation}"
                )
            if (
                current.get("feedback_source_generation_index")
                != previous["generation_index"]
            ):
                raise SystemExit(
                    f"{label} reencode source generation mismatch at episode "
                    f"{episode_index} generation={generation}"
                )
            for feedback_key, predicted_key in (
                (contract["raw_key"], "predicted_actions"),
                (contract["normalized_key"], "predicted_actions_normalized"),
            ):
                feedback_actions = current.get(feedback_key)
                if feedback_actions is None or not _same_contiguous_bytes(
                    feedback_actions, previous[predicted_key]
                ):
                    raise SystemExit(
                        f"{label} {feedback_key}/previous {predicted_key} bytes "
                        f"mismatch at episode {episode_index} generation={generation}"
                    )
            matched_commits += 1
    if matched_commits != run["reported_commits"]:
        raise SystemExit(
            f"{label} reencode feedback commit total mismatch: "
            f"matched={matched_commits} reported={run['reported_commits']}"
        )


def _artifact_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name in {"server", "client"} and resolved.parent.name == "telemetry":
        return resolved.parent.parent
    if resolved.name == "telemetry":
        return resolved.parent
    return resolved


def _is_direct_server_smoke(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return (
        resolved.name == "telemetry"
        and (resolved / "episodes.jsonl").is_file()
        and (resolved / "steps.jsonl").is_file()
    )


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _require_run_manifest_provenance(
    reference: dict,
    run: dict,
    client: dict,
    *,
    run_path: Path,
    label: str,
    expected_sha256: str | None = None,
) -> str:
    manifest_path = _artifact_root(run_path) / "prompt_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{label} prompt manifest is missing: {manifest_path}")
    encoded = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and manifest_sha256 != expected_sha256:
        raise SystemExit(
            f"{label} prompt manifest SHA-256 {manifest_sha256} does not match "
            f"expected {expected_sha256}"
        )
    try:
        manifest = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label} prompt manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit(f"{label} prompt manifest must be a JSON object")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != "robotwin_prompt_replay"
    ):
        raise SystemExit(f"invalid {label} prompt manifest schema/kind")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise SystemExit(f"{label} prompt manifest has invalid source provenance")
    if source.get("checkpoint_sha256") != reference["identity"]["checkpoint_sha256"]:
        raise SystemExit(f"{label} prompt manifest checkpoint does not match control")
    if (
        source.get("episode_noise_base_seed")
        != reference["identity"]["episode_noise_base_seed"]
    ):
        raise SystemExit(f"{label} prompt manifest base seed does not match control")

    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < len(run["pairs"]):
        raise SystemExit(f"{label} prompt manifest has insufficient episodes")
    if source.get("episode_count") != len(episodes):
        raise SystemExit(f"{label} prompt manifest source episode count is invalid")
    for ordinal, entry in enumerate(episodes):
        if not isinstance(entry, dict):
            raise SystemExit(f"{label} prompt manifest episode {ordinal} is invalid")
        if (
            isinstance(entry.get("ordinal"), bool)
            or entry.get("ordinal") != ordinal
            or isinstance(entry.get("episode_index"), bool)
            or entry.get("episode_index") != ordinal
        ):
            raise SystemExit(
                f"{label} prompt manifest ordinal/index mismatch at episode {ordinal}"
            )

    common_paths = set()
    common_hashes = set()
    for episode_index, (server_start, client_start) in enumerate(
        zip(run["starts"], client["starts"])
    ):
        server_context = server_start.get("context", {})
        client_context = client_start.get("context", {})
        for producer, context in (
            ("server", server_context),
            ("client", client_context),
        ):
            if context.get("prompt_source") != "manifest":
                raise SystemExit(
                    f"{label} {producer} prompt_source is not manifest at episode "
                    f"{episode_index}"
                )
            ordinal = context.get("prompt_manifest_ordinal")
            context_index = context.get("episode_index")
            if (
                isinstance(ordinal, bool)
                or ordinal != episode_index
                or isinstance(context_index, bool)
                or context_index != episode_index
            ):
                raise SystemExit(
                    f"{label} {producer} manifest ordinal/episode_index mismatch "
                    f"at episode {episode_index}"
                )
            context_path = context.get("prompt_manifest_path")
            context_sha256 = context.get("prompt_manifest_sha256")
            if not isinstance(context_path, str) or not context_path:
                raise SystemExit(
                    f"{label} {producer} prompt manifest path is invalid at "
                    f"episode {episode_index}"
                )
            if not isinstance(context_sha256, str) or not context_sha256:
                raise SystemExit(
                    f"{label} {producer} prompt manifest SHA-256 is invalid at "
                    f"episode {episode_index}"
                )
            common_paths.add(context_path)
            common_hashes.add(context_sha256)
        if server_context.get("prompt_manifest_path") != client_context.get(
            "prompt_manifest_path"
        ) or server_context.get("prompt_manifest_sha256") != client_context.get(
            "prompt_manifest_sha256"
        ):
            raise SystemExit(
                f"{label} server/client manifest provenance mismatch at episode "
                f"{episode_index}"
            )

        entry = episodes[episode_index]
        pair = reference["pairs"][episode_index]
        expected_fields = {
            "task_name": pair[0],
            "task_config": pair[1],
            "environment_seed": pair[2],
            "noise_pair_key": pair[3],
            "model_noise_seed": pair[4],
            "prompt": client["prompts"][episode_index],
            "prompt_sha256": hashlib.sha256(
                client["prompts"][episode_index].encode("utf-8")
            ).hexdigest(),
        }
        if any(entry.get(field) != value for field, value in expected_fields.items()):
            raise SystemExit(
                f"{label} prompt manifest entry mismatch at episode {episode_index}"
            )
        generation_zero = entry.get("generation_zero", {})
        for generation_zero_source in (
            reference["generation_zero"][episode_index],
            run["generation_zero"][episode_index],
        ):
            if generation_zero.get("predicted_actions_sha256") != _array_sha256(
                generation_zero_source["predicted_actions"]
            ) or generation_zero.get(
                "predicted_actions_normalized_sha256"
            ) != _array_sha256(generation_zero_source["predicted_actions_normalized"]):
                raise SystemExit(
                    f"{label} prompt manifest generation-zero hash mismatch at "
                    f"episode {episode_index}"
                )
        for context in (server_context, client_context):
            if (
                context.get("prompt") != expected_fields["prompt"]
                or context.get("prompt_sha256") != expected_fields["prompt_sha256"]
            ):
                raise SystemExit(
                    f"{label} prompt context mismatch at episode {episode_index}"
                )

    if len(common_paths) != 1 or len(common_hashes) != 1:
        raise SystemExit(f"{label} episodes do not share one prompt manifest path/SHA")
    recorded_sha256 = next(iter(common_hashes))
    if recorded_sha256 != manifest_sha256:
        raise SystemExit(f"{label} prompt manifest SHA-256 does not match run artifact")
    return manifest_sha256


def _normalized_identity(identity: dict) -> dict:
    normalized = dict(identity)
    normalized.pop("cache_feedback_mode", None)
    return normalized


def _claims_manifest(server: dict, client: dict) -> bool:
    contexts = [start.get("context", {}) for start in server["starts"]]
    contexts.extend(start.get("context", {}) for start in client["starts"])
    return any(context.get("prompt_source") == "manifest" for context in contexts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--control-client", "--reference-client", type=Path)
    parser.add_argument("--treatment-client", "--candidate-client", type=Path)
    parser.add_argument(
        "--control-mode",
        "--reference-mode",
        choices=VALID_MODES,
        default="predicted",
    )
    parser.add_argument(
        "--treatment-mode",
        "--candidate-mode",
        choices=VALID_MODES,
        default="measured",
    )
    parser.add_argument(
        "--allow-control-prefix",
        "--allow-reference-prefix",
        action="store_true",
        help="compare against a validated prefix of a longer control run",
    )
    parser.add_argument(
        "--expected-prompt-manifest-sha256",
        help="require this exact replay-manifest digest for final artifacts",
    )
    args = parser.parse_args()
    if args.expected_prompt_manifest_sha256 is not None:
        digest = args.expected_prompt_manifest_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            parser.error("--expected-prompt-manifest-sha256 must be lowercase hex")

    control = load_run(
        args.control,
        expected_mode=args.control_mode,
        expected_episodes=None if args.allow_control_prefix else args.expected_episodes,
    )
    if args.allow_control_prefix:
        control = _prefix_run(control, args.expected_episodes)
    treatment = load_run(
        args.treatment,
        expected_mode=args.treatment_mode,
        expected_episodes=args.expected_episodes,
    )
    if control["pairs"] != treatment["pairs"]:
        raise SystemExit("ordered scene/model-noise pairs do not match")
    if _normalized_identity(control["identity"]) != _normalized_identity(
        treatment["identity"]
    ):
        raise SystemExit("control/treatment deployment identities do not match")
    _require_generation_zero_equal(control, treatment)
    if args.control_mode == "reencode_predicted":
        _require_reencoded_predicted_mapping(control, label="control")
    if args.treatment_mode == "reencode_predicted":
        _require_reencoded_predicted_mapping(treatment, label="treatment")

    if bool(args.control_client) != bool(args.treatment_client):
        raise SystemExit("provide both --control-client and --treatment-client")
    server_only_smoke = args.control_client is None
    if server_only_smoke and not (
        _is_direct_server_smoke(args.control)
        and _is_direct_server_smoke(args.treatment)
    ):
        raise SystemExit(
            "client artifacts are required for final paired-artifact validation"
        )
    if args.control_client is not None:
        control_client = load_client_run(
            args.control_client,
            expected_episodes=None
            if args.allow_control_prefix
            else args.expected_episodes,
        )
        if args.allow_control_prefix:
            control_client = _prefix_run(control_client, args.expected_episodes)
        treatment_client = load_client_run(
            args.treatment_client, expected_episodes=args.expected_episodes
        )
        expected_pairs = [pair[:4] for pair in control["pairs"]]
        if control_client["pairs"] != expected_pairs:
            raise SystemExit("control client/server episode identities do not match")
        if treatment_client["pairs"] != expected_pairs:
            raise SystemExit("treatment client/server episode identities do not match")
        _require_server_client_mapping(control, control_client, label="control")
        _require_server_client_mapping(treatment, treatment_client, label="treatment")
        for index, (control_prompt, treatment_prompt) in enumerate(
            zip(control_client["prompts"], treatment_client["prompts"])
        ):
            if control_prompt != treatment_prompt:
                raise SystemExit(
                    f"control/treatment prompt mismatch at episode {index} "
                    f"scene={expected_pairs[index][:3]!r}"
                )
        treatment_manifest_sha = _require_run_manifest_provenance(
            control,
            treatment,
            treatment_client,
            run_path=args.treatment,
            label="treatment",
            expected_sha256=args.expected_prompt_manifest_sha256,
        )
        if _claims_manifest(control, control_client):
            control_manifest_sha = _require_run_manifest_provenance(
                control,
                control,
                control_client,
                run_path=args.control,
                label="control",
                expected_sha256=args.expected_prompt_manifest_sha256,
            )
            if control_manifest_sha != treatment_manifest_sha:
                raise SystemExit(
                    "control/treatment prompt manifest SHA-256 values do not match"
                )
        if args.control_mode == "measured":
            _require_measured_feedback_mapping(control, control_client)
        if args.treatment_mode == "measured":
            _require_measured_feedback_mapping(treatment, treatment_client)

    status = "SERVER_ONLY_SMOKE_PASS" if server_only_smoke else "PASS"
    print(f"paired_noise_telemetry: {status} episodes={len(control['pairs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
