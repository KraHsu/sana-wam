#!/usr/bin/env python
"""Deterministic GPU differential smoke for external OpenWAM and sana-wam.

Run ``external`` and ``native`` in separate, sequential processes on the same
physical GPU, then run ``compare`` on the resulting artifacts. Both model runs
should use the external OpenWAM virtualenv so preprocessing and CUDA package
versions cannot become an uncontrolled variable.

This is an offline, teacher-forced differential: it feeds 29 recorded HDF5
observations through the real PolicyServer/WAMPolicy path. The first 28 returned
actions are the bootstrap chunk; observation 29 triggers the steady-state video
then action branch and returns that chunk's first action.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True


NATIVE_REPO = Path("/home/zch/workspace/sana-wam")
EXTERNAL_REPO = Path("/home/zch/wuji-openwam-dev")
DEFAULT_CKPT_DIR = (
    EXTERNAL_REPO / "sandbox/sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25"
)
DEFAULT_CKPT_NAME = "checkpoint_step_12000.safetensors"
DEFAULT_EPISODE = Path(
    "/DATA/share/RoboTwin2.0/dataset/adjust_bottle/aloha-agilex_clean_50/data/episode0.hdf5"
)
DEFAULT_INSTRUCTIONS = Path(
    "/DATA/share/RoboTwin2.0/dataset/adjust_bottle/"
    "aloha-agilex_clean_50/instructions/episode0.json"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/sana_ar_openwam_diff")
EXPECTED_CHECKPOINT_BYTES = 11_689_906_174
EXPECTED_CHECKPOINT_SHA256 = (
    "aea6b58c9107aeeeffba561f39ffedfc3f8fd8420a4abfcab9038c8cdd4a129d"
)
SMOKE_OBSERVATIONS = 29
ACTION_DIM = 20
ACTION_TOKENS_PER_CHUNK = 28
DEFAULT_SEED = 20_260_714

# These must be set before importing either stack or initializing CUDA.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["DISABLE_XFORMERS"] = "1"
os.environ["GDN_DISABLE_COMPILE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True, choices=("external", "native", "compare")
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Artifact prefix for a model run."
    )
    parser.add_argument(
        "--external-output",
        type=Path,
        default=None,
        help="External artifact prefix for compare.",
    )
    parser.add_argument(
        "--native-output",
        type=Path,
        default=None,
        help="Native artifact prefix for compare.",
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="Comparison JSON output path."
    )
    parser.add_argument("--external-repo", type=Path, default=EXTERNAL_REPO)
    parser.add_argument("--native-repo", type=Path, default=NATIVE_REPO)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--ckpt-name", default=DEFAULT_CKPT_NAME)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--verify-checkpoint-sha256",
        action="store_true",
        help="Hash the 10.9 GiB checkpoint before loading it.",
    )
    parser.add_argument(
        "--parity-max-abs",
        type=float,
        default=1e-3,
        help="Maximum normalized-action error accepted as numerical parity.",
    )
    parser.add_argument(
        "--parity-mean-abs",
        type=float,
        default=1e-4,
        help="Mean normalized-action error accepted as numerical parity.",
    )
    parser.add_argument(
        "--require-bitwise",
        action="store_true",
        help="Make compare fail unless both action artifacts are bitwise equal.",
    )
    return parser


def _artifact_paths(prefix: Path) -> tuple[Path, Path]:
    prefix = prefix.expanduser().resolve()
    if prefix.suffix in (".npz", ".json"):
        prefix = prefix.with_suffix("")
    return prefix.with_suffix(".npz"), prefix.with_suffix(".json")


def _default_prefix(mode: str) -> Path:
    name = "openwam_external" if mode == "external" else "sanawam_native"
    return DEFAULT_OUTPUT_DIR / name


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")


def _configure_stack_paths(stack: str, external_repo: Path, native_repo: Path) -> Path:
    external_repo = external_repo.expanduser().resolve()
    native_repo = native_repo.expanduser().resolve()
    removable = {
        str(external_repo),
        str(external_repo / "third_party/Sana"),
        str(native_repo),
        str(native_repo / "src"),
        str(native_repo / "third_party/Sana"),
    }
    sys.path[:] = [
        entry for entry in sys.path if str(Path(entry).resolve()) not in removable
    ]

    if stack == "external":
        repo = external_repo
        paths = (repo, repo / "third_party/Sana")
    else:
        repo = native_repo
        paths = (repo / "src", repo / "third_party/Sana", repo)
    sys.path[0:0] = [str(path) for path in paths]
    os.chdir(repo)
    return repo


def _load_action_conversion(native_repo: Path):
    source = native_repo / "benchmarks/utils/action_conversion.py"
    spec = importlib.util.spec_from_file_location(
        "_sana_diff_action_conversion", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load action conversion module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return source, module.robotwin_endpose_to_eef20d


def _load_observations(episode: Path, instructions: Path, conversion):
    import cv2
    import h5py
    import numpy as np
    from PIL import Image

    with instructions.open(encoding="utf-8") as stream:
        instruction_data = json.load(stream)
    prompt = str(instruction_data["unseen"][0])

    observations = []
    with h5py.File(episode, "r") as data:
        for index in range(SMOKE_OBSERVATIONS):
            cameras = {}
            for camera in ("head_camera", "left_camera", "right_camera"):
                raw = data[f"observation/{camera}/rgb"][index]
                frame = cv2.imdecode(
                    np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None or frame.shape != (240, 320, 3):
                    shape = None if frame is None else frame.shape
                    raise AssertionError(f"Unexpected {camera} frame {index}: {shape}")
                cameras[camera] = Image.fromarray(frame)

            state = np.asarray(
                conversion(
                    data["endpose/left_endpose"][index],
                    data["endpose/right_endpose"][index],
                    data["endpose/left_gripper"][index],
                    data["endpose/right_gripper"][index],
                ),
                dtype=np.float32,
            )
            if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
                raise AssertionError(f"Invalid proprio at frame {index}: {state.shape}")
            observations.append((cameras, state))
    return prompt, observations


def _set_determinism(seed: int, torch, np) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def _make_deploy_overlay(seed: int, device: str, OmegaConf):
    return OmegaConf.create(
        {
            "device": device,
            "inference": {
                "denoise_steps": 10,
                "video_steps": 10,
                "action_steps": 10,
                "seed": seed,
                "shift": 5.0,
                "ar_obs_chunk_mode": "rolling_buffer",
                "ar_obs_latent_band": "auto",
                "ar_proprio_mode": "per_step",
                "num_frames": 113,
                "video_num_frames": 29,
                "height": 384,
                "width": 320,
            },
            "policy": {
                "execute_horizon": None,
                "temporal_ensemble": True,
                "ensemble_decay": 0.5,
                "history_len": 10,
            },
            "optimization": {
                "decode_video": False,
                "compile": {"mode": "none"},
                "async_inference": {"mode": "none"},
                "prompt_embed_cache": {"maxsize": 32},
            },
        }
    )


def _import_stack(stack: str):
    if stack == "external":
        from openwam.deploy import build_engine
        from openwam.deploy.model_loader import load_from_checkpoint_dir
        from openwam.deploy.policy_server import PolicyServer

        return build_engine, load_from_checkpoint_dir, PolicyServer

    from sana_wam.deploy import build_engine
    from sana_wam.deploy.model_loader import load_from_checkpoint_dir
    from sana_wam.deploy.policy_server import PolicyServer

    return build_engine, load_from_checkpoint_dir, PolicyServer


def _install_unused_fla_stub(stack: str, torch) -> bool:
    """Let the native SANA package import when the shared runtime lacks FLA.

    The canonical model uses GLUMBConvTemp, not a GDN block. SANA imports its
    optional GDN module eagerly, though, so provide a fail-fast placeholder for
    ShortConvolution instead of mixing another virtualenv's Triton/FLA stack.
    """
    if stack != "native":
        return False
    try:
        from fla.modules import ShortConvolution as _  # noqa: F401

        return False
    except ModuleNotFoundError:
        fla_module = types.ModuleType("fla")
        modules_module = types.ModuleType("fla.modules")

        class ShortConvolution(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError(
                    "FLA stub must not be instantiated by the AR baseline"
                )

        modules_module.ShortConvolution = ShortConvolution
        fla_module.modules = modules_module
        sys.modules["fla"] = fla_module
        sys.modules["fla.modules"] = modules_module
        return True


def _source_metadata(stack: str, repo: Path) -> dict[str, str]:
    if stack == "external":
        package_root = repo / "openwam"
        ar_engine = package_root / "deploy/ar_engine.py"
        architecture = (
            package_root / "model/architectures/dual_system/autoregressive.py"
        )
        blocks_split = package_root / "model/video_backbone/sana/blocks_split.py"
    else:
        package_root = repo / "src/sana_wam"
        ar_engine = package_root / "deploy/ar_engine.py"
        architecture = package_root / "model/architecture.py"
        blocks_split = package_root / "model/video_backbone/sana/blocks_split.py"
    vendor = repo / "third_party/Sana/diffusion/model/nets/sana_blocks.py"
    return {
        "repo": str(repo),
        "ar_engine": str(ar_engine),
        "ar_engine_sha256": _sha256_file(ar_engine),
        "architecture": str(architecture),
        "architecture_sha256": _sha256_file(architecture),
        "blocks_split": str(blocks_split),
        "blocks_split_sha256": _sha256_file(blocks_split),
        "vendor_sana_blocks": str(vendor),
        "vendor_sana_blocks_sha256": _sha256_file(vendor),
    }


def _run_stack(args: argparse.Namespace) -> int:
    stack = args.mode
    if stack not in ("external", "native"):
        raise ValueError(stack)

    repo = _configure_stack_paths(stack, args.external_repo, args.native_repo)

    import cv2
    import h5py
    import numpy as np
    import PIL
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    _set_determinism(args.seed, torch, np)

    checkpoint = (args.ckpt_dir / args.ckpt_name).expanduser().resolve()
    checkpoint_config = args.ckpt_dir.expanduser().resolve() / "config.yaml"
    action_stats = args.ckpt_dir.expanduser().resolve() / "action_stats.npy"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not checkpoint_config.is_file():
        raise FileNotFoundError(checkpoint_config)
    if not action_stats.is_file():
        raise FileNotFoundError(action_stats)
    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise AssertionError(
            f"Checkpoint size mismatch: {checkpoint.stat().st_size} != {EXPECTED_CHECKPOINT_BYTES}"
        )
    checkpoint_sha256 = None
    if args.verify_checkpoint_sha256:
        checkpoint_sha256 = _sha256_file(checkpoint)
        if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise AssertionError(
                f"Checkpoint SHA-256 mismatch: {checkpoint_sha256} != {EXPECTED_CHECKPOINT_SHA256}"
            )

    conversion_source, conversion = _load_action_conversion(
        args.native_repo.expanduser().resolve()
    )
    prompt, observations = _load_observations(
        args.episode.expanduser().resolve(),
        args.instructions.expanduser().resolve(),
        conversion,
    )

    fla_stubbed = _install_unused_fla_stub(stack, torch)
    build_engine, load_from_checkpoint_dir, PolicyServer = _import_stack(stack)
    started = time.perf_counter()
    training_cfg, architecture = load_from_checkpoint_dir(
        str(args.ckpt_dir.expanduser().resolve()),
        device=str(device),
        ckpt_name=args.ckpt_name,
    )
    overlay = _make_deploy_overlay(args.seed, str(device), OmegaConf)
    cfg = OmegaConf.merge(training_cfg, overlay)
    engine = build_engine(cfg=cfg, architecture=architecture, training_cfg=training_cfg)
    server = PolicyServer(engine=engine, cfg=cfg)
    server._init_policy()
    server.reset()
    load_seconds = time.perf_counter() - started

    first_block = architecture.video_backbone._dit.blocks[0]
    model_types = {
        "engine": type(engine).__name__,
        "video_attention": type(first_block.attn).__name__,
        "feature_map": type(first_block.attn.kernel_func).__name__,
        "graft": type(first_block.flash_attn_additional).__name__,
        "action_normalizer": type(architecture.action_normalizer).__name__,
    }
    expected_types = {
        "engine": "ARInferenceEngine",
        "video_attention": "LiteLAReLURope",
        "feature_map": "LearnableFeatureMap",
        "graft": "WindowAttention",
        "action_normalizer": "ActionNormalizer",
    }
    if model_types != expected_types:
        raise AssertionError(f"Unexpected model topology: {model_types}")
    if len(architecture.state_dict()) != 1741:
        raise AssertionError(
            f"Unexpected state_dict size: {len(architecture.state_dict())}"
        )
    if engine._video_steps != 10 or engine._action_steps != 10:
        raise AssertionError((engine._video_steps, engine._action_steps))
    if engine._action_tokens_per_chunk != ACTION_TOKENS_PER_CHUNK:
        raise AssertionError(engine._action_tokens_per_chunk)
    if engine._step_c != 0 or engine._seed != args.seed:
        raise AssertionError((engine._step_c, engine._seed))
    if int(getattr(engine, "_graft_window", 0)) != 0:
        raise AssertionError(
            "External graft cache must remain disabled for the canonical checkpoint"
        )
    if server._policy.obs_history.maxlen != 10:
        raise AssertionError(server._policy.obs_history.maxlen)

    generated_chunks: list[Any] = []
    original_generate = engine.generate

    def traced_generate(conditions):
        result = original_generate(conditions)
        generated_chunks.append(np.asarray(result["actions"], dtype=np.float32).copy())
        return result

    engine.generate = traced_generate

    actions = []
    processed_image_hashes = []
    processed_image_shapes = []
    wrapped_prompt_hashes = []
    step_c_trace = []
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
            if result["step"] != index + 1:
                raise AssertionError((result["step"], index + 1))
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise AssertionError(f"Invalid action {index}: {action.shape}")

            processed = np.asarray(
                server._policy.obs_history[-1]["image"], dtype=np.uint8
            )
            wrapped_prompt = str(server._policy.obs_history[-1]["prompt"])
            actions.append(action)
            processed_image_hashes.append(_sha256_bytes(processed.tobytes()))
            processed_image_shapes.append(processed.shape)
            wrapped_prompt_hashes.append(_sha256_bytes(wrapped_prompt.encode("utf-8")))
            step_c_trace.append(int(engine._step_c))
            latencies_ms.append(float(result["latency_ms"]))

        torch.cuda.synchronize(device)
    finally:
        server.shutdown()

    actions_array = np.stack(actions).astype(np.float32, copy=False)
    chunks_array = np.stack(generated_chunks).astype(np.float32, copy=False)
    states_array = np.stack([state for _, state in observations]).astype(
        np.float32, copy=False
    )
    if actions_array.shape != (SMOKE_OBSERVATIONS, ACTION_DIM):
        raise AssertionError(actions_array.shape)
    if chunks_array.shape != (2, ACTION_TOKENS_PER_CHUNK, ACTION_DIM):
        raise AssertionError(chunks_array.shape)
    expected_actions = np.concatenate([chunks_array[0], chunks_array[1, :1]], axis=0)
    if not np.array_equal(actions_array, expected_actions):
        raise AssertionError(
            "Policy actions do not equal bootstrap chunk + first steady action"
        )
    expected_step_c = np.asarray([1] * ACTION_TOKENS_PER_CHUNK + [2], dtype=np.int64)
    if not np.array_equal(np.asarray(step_c_trace), expected_step_c):
        raise AssertionError(step_c_trace)
    if engine._step_c != 2:
        raise AssertionError(engine._step_c)
    if len(server._policy.obs_history) != 10:
        raise AssertionError(len(server._policy.obs_history))

    normalizer = architecture.action_normalizer
    normalized_actions = normalizer.normalize(actions_array)
    normalized_chunks = normalizer.normalize(chunks_array)
    v_sigmas = np.asarray(engine._v_sigmas, dtype=np.float64)
    a_sigmas = np.asarray(engine._a_sigmas, dtype=np.float64)
    v_timesteps = engine._v_ts.detach().float().cpu().numpy()
    a_timesteps = engine._a_ts.detach().float().cpu().numpy()

    output_prefix = args.output or _default_prefix(stack)
    npz_path, json_path = _artifact_paths(output_prefix)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        actions=actions_array,
        chunks=chunks_array,
        normalized_actions=normalized_actions,
        normalized_chunks=normalized_chunks,
        states=states_array,
        processed_image_hashes=np.asarray(processed_image_hashes, dtype="U64"),
        processed_image_shapes=np.asarray(processed_image_shapes, dtype=np.int64),
        wrapped_prompt_hashes=np.asarray(wrapped_prompt_hashes, dtype="U64"),
        video_sigmas=v_sigmas,
        action_sigmas=a_sigmas,
        video_timesteps=v_timesteps,
        action_timesteps=a_timesteps,
        engine_step_c_trace=np.asarray(step_c_trace, dtype=np.int64),
        engine_step_c=np.asarray(engine._step_c, dtype=np.int64),
        latencies_ms=np.asarray(latencies_ms, dtype=np.float64),
    )

    import diffusion.model.nets.sana_blocks as vendor_sana_blocks

    metadata = {
        "artifact_schema": 1,
        "stack": stack,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "npz": str(npz_path),
        "source": _source_metadata(stack, repo),
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "h5py": h5py.__version__,
            "pillow": PIL.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(device),
            "device": str(device),
            "vendor_module": str(Path(vendor_sana_blocks.__file__).resolve()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "disable_xformers": os.environ.get("DISABLE_XFORMERS"),
            "gdn_disable_compile": os.environ.get("GDN_DISABLE_COMPILE"),
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "unused_fla_stubbed": fla_stubbed,
        },
        "checkpoint": {
            "directory": str(args.ckpt_dir.expanduser().resolve()),
            "name": args.ckpt_name,
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
            "verified_sha256": checkpoint_sha256,
            "config_path": str(checkpoint_config),
            "config_sha256": _sha256_file(checkpoint_config),
            "action_stats_path": str(action_stats),
            "action_stats_sha256": _sha256_file(action_stats),
        },
        "input": {
            "episode": str(args.episode.expanduser().resolve()),
            "instructions": str(args.instructions.expanduser().resolve()),
            "prompt": prompt,
            "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "observations": SMOKE_OBSERVATIONS,
            "action_conversion": str(conversion_source),
            "action_conversion_sha256": _sha256_file(conversion_source),
        },
        "effective": {
            "seed": args.seed,
            "video_steps": engine._video_steps,
            "action_steps": engine._action_steps,
            "action_tokens_per_chunk": engine._action_tokens_per_chunk,
            "history_len": server._policy.obs_history.maxlen,
            "execute_horizon": server._policy.execute_horizon,
            "temporal_ensemble": server._policy.temporal_ensemble,
            "obs_chunk_mode": engine._obs_chunk_mode,
            "obs_latent_band": engine._obs_latent_band,
            "proprio_mode": engine._proprio_mode,
            "frame_chunk_size": engine._fcs,
            "attention_window": engine._window,
            "graft_cache_window": int(getattr(engine, "_graft_window", 0)),
            "final_step_c": engine._step_c,
        },
        "model": {
            **model_types,
            "state_keys": len(architecture.state_dict()),
            "dtype": str(architecture.dtype),
            "device": str(architecture.device),
        },
        "timing": {
            "load_seconds": load_seconds,
            "bootstrap_generate_ms": latencies_ms[0],
            "steady_generate_ms": latencies_ms[-1],
        },
        "arrays": {
            "actions": list(actions_array.shape),
            "chunks": list(chunks_array.shape),
            "processed_image_hashes": len(processed_image_hashes),
            "video_sigmas": v_sigmas.tolist(),
            "action_sigmas": a_sigmas.tolist(),
        },
    }
    _json_dump(json_path, metadata)
    print(f"{stack}_differential_smoke: PASS")
    print(f"npz={npz_path}")
    print(f"json={json_path}")
    print(
        f"actions={actions_array.shape} chunks={chunks_array.shape} step_c={engine._step_c}"
    )
    return 0


def _load_artifact(prefix: Path):
    import numpy as np

    npz_path, json_path = _artifact_paths(prefix)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    with json_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    arrays = np.load(npz_path, allow_pickle=False)
    return npz_path, json_path, metadata, arrays


def _delta_metrics(left, right) -> dict[str, Any]:
    import numpy as np

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    delta = np.abs(left - right)
    flat_index = int(delta.argmax()) if delta.size else 0
    worst_index = (
        [int(index) for index in np.unravel_index(flat_index, delta.shape)]
        if delta.size
        else []
    )
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    cosine = float(np.dot(left_flat, right_flat) / denominator) if denominator else 1.0
    return {
        "shape": list(left.shape),
        "bitwise_equal": bool(np.array_equal(left, right)),
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "p99_abs": float(np.percentile(delta, 99)) if delta.size else 0.0,
        "cosine": cosine,
        "worst_index": worst_index,
        "left_at_worst": float(left[tuple(worst_index)]) if worst_index else None,
        "right_at_worst": float(right[tuple(worst_index)]) if worst_index else None,
    }


def _compare(args: argparse.Namespace) -> int:
    import numpy as np

    external_prefix = args.external_output or _default_prefix("external")
    native_prefix = args.native_output or _default_prefix("native")
    ext_npz_path, ext_json_path, ext_meta, ext = _load_artifact(external_prefix)
    nat_npz_path, nat_json_path, nat_meta, nat = _load_artifact(native_prefix)

    required = {
        "actions": (SMOKE_OBSERVATIONS, ACTION_DIM),
        "chunks": (2, ACTION_TOKENS_PER_CHUNK, ACTION_DIM),
        "normalized_actions": (SMOKE_OBSERVATIONS, ACTION_DIM),
        "normalized_chunks": (2, ACTION_TOKENS_PER_CHUNK, ACTION_DIM),
        "states": (SMOKE_OBSERVATIONS, ACTION_DIM),
        "processed_image_hashes": (SMOKE_OBSERVATIONS,),
        "engine_step_c_trace": (SMOKE_OBSERVATIONS,),
    }
    shape_checks = {}
    for key, shape in required.items():
        ext_shape = tuple(ext[key].shape)
        nat_shape = tuple(nat[key].shape)
        shape_checks[key] = {
            "expected": list(shape),
            "external": list(ext_shape),
            "native": list(nat_shape),
            "pass": ext_shape == shape and nat_shape == shape,
        }

    expected_step_c = np.asarray([1] * ACTION_TOKENS_PER_CHUNK + [2], dtype=np.int64)
    preconditions = {
        "artifact_schema": ext_meta.get("artifact_schema")
        == nat_meta.get("artifact_schema")
        == 1,
        "stack_labels": ext_meta.get("stack") == "external"
        and nat_meta.get("stack") == "native",
        "same_checkpoint_path": ext_meta.get("checkpoint", {}).get("path")
        == nat_meta.get("checkpoint", {}).get("path"),
        "same_checkpoint_bytes": ext_meta.get("checkpoint", {}).get("bytes")
        == nat_meta.get("checkpoint", {}).get("bytes")
        == EXPECTED_CHECKPOINT_BYTES,
        "same_checkpoint_config": ext_meta.get("checkpoint", {}).get("config_sha256")
        == nat_meta.get("checkpoint", {}).get("config_sha256"),
        "same_action_stats": ext_meta.get("checkpoint", {}).get("action_stats_sha256")
        == nat_meta.get("checkpoint", {}).get("action_stats_sha256"),
        "same_seed": ext_meta.get("effective", {}).get("seed")
        == nat_meta.get("effective", {}).get("seed"),
        "same_effective_contract": all(
            ext_meta.get("effective", {}).get(key)
            == nat_meta.get("effective", {}).get(key)
            for key in (
                "video_steps",
                "action_steps",
                "action_tokens_per_chunk",
                "history_len",
                "execute_horizon",
                "temporal_ensemble",
                "obs_chunk_mode",
                "obs_latent_band",
                "proprio_mode",
                "frame_chunk_size",
                "attention_window",
                "graft_cache_window",
            )
        ),
        "same_python_executable": ext_meta.get("environment", {}).get(
            "python_executable"
        )
        == nat_meta.get("environment", {}).get("python_executable"),
        "same_runtime_versions": all(
            ext_meta.get("environment", {}).get(key)
            == nat_meta.get("environment", {}).get(key)
            for key in (
                "python_version",
                "torch",
                "torch_cuda",
                "numpy",
                "opencv",
                "h5py",
                "pillow",
            )
        ),
        "same_gpu": ext_meta.get("environment", {}).get("gpu")
        == nat_meta.get("environment", {}).get("gpu"),
        "same_cuda_visibility": ext_meta.get("environment", {}).get(
            "cuda_visible_devices"
        )
        == nat_meta.get("environment", {}).get("cuda_visible_devices"),
        "same_input_paths": all(
            ext_meta.get("input", {}).get(key) == nat_meta.get("input", {}).get(key)
            for key in ("episode", "instructions", "action_conversion")
        ),
        "same_base_prompt": ext_meta.get("input", {}).get("prompt_sha256")
        == nat_meta.get("input", {}).get("prompt_sha256"),
        "same_states": bool(np.array_equal(ext["states"], nat["states"])),
        "same_processed_images": bool(
            np.array_equal(ext["processed_image_hashes"], nat["processed_image_hashes"])
        ),
        "same_processed_image_shapes": bool(
            np.array_equal(ext["processed_image_shapes"], nat["processed_image_shapes"])
        ),
        "same_wrapped_prompt": bool(
            np.array_equal(ext["wrapped_prompt_hashes"], nat["wrapped_prompt_hashes"])
        ),
        "same_video_sigmas": bool(
            np.array_equal(ext["video_sigmas"], nat["video_sigmas"])
        ),
        "same_action_sigmas": bool(
            np.array_equal(ext["action_sigmas"], nat["action_sigmas"])
        ),
        "external_step_c_contract": bool(
            np.array_equal(ext["engine_step_c_trace"], expected_step_c)
        ),
        "native_step_c_contract": bool(
            np.array_equal(nat["engine_step_c_trace"], expected_step_c)
        ),
        "final_step_c": int(ext["engine_step_c"].item())
        == int(nat["engine_step_c"].item())
        == 2,
        "all_shapes": all(check["pass"] for check in shape_checks.values()),
    }

    metrics = {
        "physical_actions_all": _delta_metrics(ext["actions"], nat["actions"]),
        "physical_actions_bootstrap": _delta_metrics(
            ext["actions"][:28], nat["actions"][:28]
        ),
        "physical_action_steady_first": _delta_metrics(
            ext["actions"][28:], nat["actions"][28:]
        ),
        "physical_chunks_all": _delta_metrics(ext["chunks"], nat["chunks"]),
        "physical_chunk_bootstrap": _delta_metrics(ext["chunks"][0], nat["chunks"][0]),
        "physical_chunk_steady": _delta_metrics(ext["chunks"][1], nat["chunks"][1]),
        "normalized_actions_all": _delta_metrics(
            ext["normalized_actions"], nat["normalized_actions"]
        ),
        "normalized_chunks_all": _delta_metrics(
            ext["normalized_chunks"], nat["normalized_chunks"]
        ),
        "normalized_chunk_bootstrap": _delta_metrics(
            ext["normalized_chunks"][0], nat["normalized_chunks"][0]
        ),
        "normalized_chunk_steady": _delta_metrics(
            ext["normalized_chunks"][1], nat["normalized_chunks"][1]
        ),
        "physical_xyz": _delta_metrics(
            ext["chunks"][:, :, [0, 1, 2, 10, 11, 12]],
            nat["chunks"][:, :, [0, 1, 2, 10, 11, 12]],
        ),
        "physical_grippers": _delta_metrics(
            ext["chunks"][:, :, [9, 19]],
            nat["chunks"][:, :, [9, 19]],
        ),
    }

    preconditions_pass = all(preconditions.values())
    normalized = metrics["normalized_chunks_all"]
    bitwise = bool(
        metrics["physical_actions_all"]["bitwise_equal"]
        and metrics["physical_chunks_all"]["bitwise_equal"]
        and normalized["bitwise_equal"]
    )
    numeric = bool(
        normalized["max_abs"] <= args.parity_max_abs
        and normalized["mean_abs"] <= args.parity_mean_abs
    )
    if not preconditions_pass:
        status = "INVALID_PRECONDITION"
    elif bitwise:
        status = "PASS_BITWISE"
    elif numeric:
        status = "PASS_NUMERIC"
    else:
        status = "FAIL_DIVERGED"

    report_path = args.report or (DEFAULT_OUTPUT_DIR / "comparison.json")
    report = {
        "artifact_schema": 1,
        "status": status,
        "thresholds": {
            "normalized_max_abs": args.parity_max_abs,
            "normalized_mean_abs": args.parity_mean_abs,
            "require_bitwise": args.require_bitwise,
        },
        "artifacts": {
            "external_npz": str(ext_npz_path),
            "external_json": str(ext_json_path),
            "native_npz": str(nat_npz_path),
            "native_json": str(nat_json_path),
        },
        "shape_checks": shape_checks,
        "preconditions": preconditions,
        "metrics": metrics,
        "interpretation": {
            "bootstrap_rows": "actions[0:28] and chunks[0]",
            "steady_rows": "actions[28] and chunks[1]",
            "xyz_warning_meters": 0.002,
            "gripper_warning": 0.02,
            "xyz_warning_triggered": metrics["physical_xyz"]["max_abs"] > 0.002,
            "gripper_warning_triggered": metrics["physical_grippers"]["max_abs"] > 0.02,
        },
    }
    _json_dump(report_path.expanduser().resolve(), report)

    print(f"comparison_status={status}")
    print(f"report={report_path.expanduser().resolve()}")
    print(
        "normalized_chunks "
        f"max_abs={normalized['max_abs']:.9g} "
        f"mean_abs={normalized['mean_abs']:.9g} "
        f"bitwise={normalized['bitwise_equal']}"
    )
    print(
        "processed_images_equal="
        f"{preconditions['same_processed_images']} sigmas_equal="
        f"{preconditions['same_video_sigmas'] and preconditions['same_action_sigmas']}"
    )

    if status in ("INVALID_PRECONDITION", "FAIL_DIVERGED"):
        return 1
    if args.require_bitwise and status != "PASS_BITWISE":
        return 1
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    if args.mode == "compare":
        return _compare(args)
    return _run_stack(args)


if __name__ == "__main__":
    raise SystemExit(main())
