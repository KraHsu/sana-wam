#!/usr/bin/env python
"""Run both AR deploy branches with the canonical low-noise checkpoint on GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Sana"))
os.environ.setdefault("GDN_DISABLE_COMPILE", "1")

from benchmarks.utils.action_conversion import robotwin_endpose_to_eef20d  # noqa: E402
from sana_wam.deploy.policy_server import build_server_from_config  # noqa: E402


DEFAULT_RUN = Path(
    "/home/zch/wuji-openwam-dev/sandbox/"
    "sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25"
)
DEFAULT_EPISODE = Path(
    "/DATA/share/RoboTwin2.0/dataset/adjust_bottle/"
    "aloha-agilex_clean_50/data/episode0.hdf5"
)
DEFAULT_INSTRUCTIONS = Path(
    "/DATA/share/RoboTwin2.0/dataset/adjust_bottle/"
    "aloha-agilex_clean_50/instructions/episode0.json"
)
EXPECTED_CHECKPOINT_BYTES = 11_689_906_174
SMOKE_STEPS = 57


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--ckpt-name", default="checkpoint_step_12000.safetensors")
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=ROOT / "configs" / "baselines" / "deploy_ar_lownoise_seedfixed.yaml",
    )
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_observations(episode: Path, instructions: Path):
    with instructions.open(encoding="utf-8") as stream:
        prompt = json.load(stream)["unseen"][0]

    observations = []
    with h5py.File(episode, "r") as data:
        for index in range(SMOKE_STEPS):
            cameras = {}
            for camera in ("head_camera", "left_camera", "right_camera"):
                raw = data[f"observation/{camera}/rgb"][index]
                frame = cv2.imdecode(
                    np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None or frame.shape != (240, 320, 3):
                    shape = None if frame is None else frame.shape
                    raise AssertionError(f"Unexpected {camera} frame {index}: {shape}")
                # RoboTwin encoded these arrays through cv2; this mirrors the train loader.
                cameras[camera] = Image.fromarray(frame)

            state = robotwin_endpose_to_eef20d(
                data["endpose/left_endpose"][index],
                data["endpose/right_endpose"][index],
                data["endpose/left_gripper"][index],
                data["endpose/right_gripper"][index],
            )
            if state.shape != (20,) or not np.isfinite(state).all():
                raise AssertionError(f"Invalid proprio at frame {index}: {state.shape}")
            observations.append((cameras, state))
    return prompt, observations


def gibibytes(value: int) -> float:
    return value / 2**30


class _RecordingRanker:
    """Capture candidate chunks without exposing them through deploy telemetry."""

    def __init__(self, ranker):
        self._ranker = ranker
        self.actions = []

    def score(self, actions):
        self.actions.append(np.asarray(actions, dtype=np.float32).copy())
        return self._ranker.score(actions)

    def __getattr__(self, name):
        return getattr(self._ranker, name)


def main() -> None:
    args = parse_args()
    checkpoint = args.ckpt_dir / args.ckpt_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise AssertionError(
            f"Checkpoint size mismatch: {checkpoint.stat().st_size} != {EXPECTED_CHECKPOINT_BYTES}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    prompt, observations = load_observations(args.episode, args.instructions)

    free, total = torch.cuda.mem_get_info(device)
    print(
        f"gpu={torch.cuda.get_device_name(device)} visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"free={gibibytes(free):.3f} GiB total={gibibytes(total):.3f} GiB",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    server = build_server_from_config(
        OmegaConf.load(args.deploy_config),
        str(args.ckpt_dir),
        device=str(device),
        ckpt_name=args.ckpt_name,
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    load_allocated = torch.cuda.memory_allocated(device)
    print(
        f"load_ok seconds={load_seconds:.3f} alloc={gibibytes(load_allocated):.3f} GiB "
        f"reserved={gibibytes(torch.cuda.memory_reserved(device)):.3f} GiB "
        f"peak_alloc={gibibytes(torch.cuda.max_memory_allocated(device)):.3f} GiB "
        f"peak_reserved={gibibytes(torch.cuda.max_memory_reserved(device)):.3f} GiB",
        flush=True,
    )

    engine = server.engine
    architecture = engine.architecture
    first_block = architecture.video_backbone._dit.blocks[0]
    assert type(engine).__name__ == "ARInferenceEngine"
    assert len(architecture.state_dict()) == 1741
    assert type(first_block.attn).__name__ == "LiteLAReLURope"
    assert type(first_block.attn.kernel_func).__name__ == "LearnableFeatureMap"
    assert type(first_block.flash_attn_additional).__name__ == "WindowAttention"
    assert architecture._ar_bootstrap_clean_prefix is True
    assert engine._video_steps == 10 and engine._action_steps == 10
    assert engine._action_tokens_per_chunk == 28 and engine._step_c == 0
    episode_context = {
        "episode_key": "gpu-smoke/adjust_bottle/episode0",
        "noise_pair_key": "gpu-smoke/adjust_bottle/scene0",
        "environment_seed": 0,
        "episode_index": 0,
    }
    reset_result = server.reset(episode_context)
    assert reset_result["duplicate"] is False
    if engine.runtime_info["episode_noise_mode"] == "paired":
        assert isinstance(reset_result["model_noise_seed"], int)

    rerank_enabled = engine.runtime_info["generation_zero_rerank"]["enabled"]
    baseline_chunk0 = None
    recording_ranker = None
    if rerank_enabled:
        # Isolate the ordinary candidate-0 path from the exact same episode state.
        # The subsequent policy reset restores the seed/cache before best-of-N runs.
        cameras, state = observations[0]
        probe_obs = server._decode_obs(
            {
                "images": {
                    "head_camera": cameras["head_camera"],
                    "left_wrist_camera": cameras["left_camera"],
                    "right_wrist_camera": cameras["right_camera"],
                },
                "prompt": prompt,
                "state": state.tolist(),
            }
        )
        server._policy.obs_history.append(probe_obs)
        probe_conditions = server._policy._build_conditions(probe_obs)
        feedback_plan, _ = engine._prepare_action_cache_feedback(probe_conditions)
        obs_latent = engine._build_obs_chunk(probe_conditions)
        context, seq_lens = engine._encode_prompt(probe_conditions["prompt"])
        proprio = engine._prep_proprio(probe_conditions.get("proprio_state"))
        engine._step_with_obs_latent(
            obs_latent,
            context,
            seq_lens,
            proprio,
            feedback_plan=feedback_plan,
        )
        baseline_chunk0 = engine._pending_action_feedback[
            "predicted_actions_normalized"
        ].copy()
        assert engine._step_c == 1

        server._policy.reset(episode_context=episode_context)
        assert engine._step_c == 0
        recording_ranker = _RecordingRanker(engine._outcome_ranker)
        engine._outcome_ranker = recording_ranker

    torch.cuda.reset_peak_memory_stats(device)
    actions = []
    latencies_ms = []
    try:
        for index, (cameras, state) in enumerate(observations):
            result = server.predict(
                {
                    "images": {
                        "head_camera": cameras["head_camera"],
                        "left_wrist_camera": cameras["left_camera"],
                        "right_wrist_camera": cameras["right_camera"],
                    },
                    "prompt": prompt,
                    "state": state.tolist(),
                }
            )
            action = np.asarray(result["action"], dtype=np.float32)
            assert result["step"] == index + 1
            assert action.shape == (20,) and np.isfinite(action).all()
            # Steps 0/28/56 generate bootstrap and two steady-state chunks.
            expected_ar_step = 1 if index < 28 else (2 if index < 56 else 3)
            assert engine._step_c == expected_ar_step
            if index in {0, 28, 56}:
                generation_index = index // 28
                assert result["policy"]["generation_noise_schedule"] == (
                    engine._generation_noise_schedule
                )
                if generation_index == 0:
                    expected_generation_seed = reset_result["model_noise_seed"]
                elif engine._generation_noise_schedule == "common_future":
                    expected_generation_seed = engine.derive_common_future_noise_seed(
                        engine._common_future_noise_base_seed,
                        episode_context["noise_pair_key"],
                        generation_index,
                    )
                else:
                    expected_generation_seed = None
                assert result["policy"].get("generation_noise_seed") == (
                    expected_generation_seed
                )
            if rerank_enabled and index == 0:
                rerank = result["policy"]["generation_zero_rerank"]
                assert rerank["generation_index"] == 0
                assert rerank["ar_step_after"] == 1
                assert rerank["candidate_count"] == 5
                assert len(rerank["candidate_seeds"]) == 5
                assert len(rerank["candidate_scores"]) == 5
                assert len(recording_ranker.actions) == 5
                np.testing.assert_array_equal(recording_ranker.actions[0], baseline_chunk0)
                selected = rerank["selected_candidate_index"]
                np.testing.assert_array_equal(
                    engine._pending_action_feedback[
                        "predicted_actions_normalized"
                    ],
                    recording_ranker.actions[selected],
                )
            elif rerank_enabled and index in {28, 56}:
                assert "generation_zero_rerank" not in result["policy"]
                assert len(recording_ranker.actions) == 5
            actions.append(action)
            latencies_ms.append(float(result["latency_ms"]))

        torch.cuda.synchronize(device)
        stacked_actions = np.stack(actions)
        assert stacked_actions.shape == (SMOKE_STEPS, 20)
        assert engine._step_c == 3
        if rerank_enabled:
            rerank_runtime = engine.runtime_info["generation_zero_rerank"]
            assert rerank_runtime["invocations"] == 1
            assert rerank_runtime["last_selection"]["generation_index"] == 0
        expected_history = int(OmegaConf.select(server.cfg, "policy.history_len"))
        assert server._policy.obs_history.maxlen == expected_history
        assert len(server._policy.obs_history) == min(SMOKE_STEPS, expected_history)
        feedback_mode = engine.runtime_info["cache_feedback_mode"]
        if feedback_mode in {"measured", "reencode_predicted"}:
            assert engine.runtime_info["cache_feedback_commits"] == 2
            assert engine.runtime_info["cache_feedback_fallbacks"] == 0
            expected_status = (
                "measured" if feedback_mode == "measured" else "reencoded_predicted"
            )
            assert (
                engine.runtime_info["last_cache_feedback"]["status"] == expected_status
            )

        peak_allocated = torch.cuda.max_memory_allocated(device)
        print("native_gpu_smoke: PASS", flush=True)
        print(
            f"actions={stacked_actions.shape} finite={np.isfinite(stacked_actions).all()} "
            f"step_c={engine._step_c}",
            flush=True,
        )
        print(
            f"chunk0_latency_ms={latencies_ms[0]:.2f} "
            f"steady1_chunk_latency_ms={latencies_ms[28]:.2f} "
            f"steady2_chunk_latency_ms={latencies_ms[56]:.2f}",
            flush=True,
        )
        print(
            f"infer_peak_alloc={gibibytes(peak_allocated):.3f} GiB "
            f"infer_peak_reserved={gibibytes(torch.cuda.max_memory_reserved(device)):.3f} GiB "
            f"infer_delta_over_postload={gibibytes(max(0, peak_allocated - load_allocated)):.3f} GiB",
            flush=True,
        )
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
