"""RoboTwin eval adapter for the sana-wam Policy Server.

Loaded by RoboTwin's ``eval_policy.py`` via ``--policy_name``. It
communicates with a running sana-wam HTTP server instead of loading model
weights directly, so the RoboTwin client environment only needs::

    numpy, opencv-python, Pillow   (see requirements.txt)

Protocol overview:

    POST /predict  — send 3 camera JPEGs + task prompt, get action vector
    POST /reset    — clear server episode state before a new rollout
    GET  /health   — liveness probe

Server default ports: WS=8850, HTTP=8848.

Camera mapping from RoboTwin to the sana-wam server's fixed client API names:

    RoboTwin              →  sana-wam client field
    head_camera           →  head_camera        (required)
    left_camera           →  left_wrist_camera  (optional)
    right_camera          →  right_wrist_camera (optional)
    front_camera          →  dropped (not part of the sana-wam contract)

All image preprocessing (resize, multi-view composition) and prompt
wrapping happen server-side, driven by the checkpoint's saved
``config.yaml``. This client just streams raw camera frames.
"""

# benchmarks.utils lives one level up. single_eval.sh only puts benchmarks/robotwin/
# on PYTHONPATH, so add the project root here to make the shared helpers
# (payload assembly, HTTP wrappers, action conversions) importable from the
# RoboTwin eval process too.
import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import atexit  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from typing import Dict, Optional  # noqa: E402

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from benchmarks.utils import action_conversion, client  # noqa: E402

try:  # package import in tests; top-level import in RoboTwin's policy loader
    from .closed_loop_telemetry import ClientTelemetryRecorder  # noqa: E402
    from .dagger_live_takeover import LiveTakeoverRecorder  # noqa: E402
except ImportError:  # pragma: no cover - exercised in the external RoboTwin process
    from closed_loop_telemetry import ClientTelemetryRecorder  # noqa: E402
    from dagger_live_takeover import LiveTakeoverRecorder  # noqa: E402

# Fields that earlier versions of policy_config.yml used. They are ignored by
# the current client contract (server decides multiview/single-view and camera
# layout from the checkpoint's config.yaml), but we log them once so users can
# clean up their YAML.
_DEPRECATED_YAML_FIELDS = (
    "image_key",
    "image_size",
    "multiview",
    "multiview_image_size",
)

# --- Per-task step_lim overrides ---
# A single YAML file of {task_name: int} lets users override RoboTwin's
# upstream task_config/_eval_step_limit.yml without touching the RoboTwin
# source tree. Missing tasks keep their upstream value (RoboTwin falls back
# to 1000 when neither side defines one).
_STEP_LIMITS_PATH = os.environ.get(
    "ROBOTWIN_STEP_LIMITS_PATH",
    os.path.join(os.path.dirname(__file__), "step_limits.yml"),
)


def _load_step_lim_overrides(path: str) -> Dict[str, int]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as exc:
        print(f"[SanaWAMClient] Failed to load step_lim overrides from {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        print(f"[SanaWAMClient] {path} must be a task_name->int mapping; ignoring.")
        return {}
    out: Dict[str, int] = {}
    for k, v in data.items():
        # Reject bool (which is an int subclass in Python) and non-int numerics
        # explicitly — silent ``int(True) == 1`` or ``int(160.9) == 160`` caps
        # would be nearly impossible to debug from a one-line override log.
        if isinstance(v, bool) or not isinstance(v, int):
            print(
                f"[SanaWAMClient] Skipping step_lim override {k!r}={v!r}: "
                f"value must be a plain int (got {type(v).__name__})"
            )
            continue
        out[str(k)] = v
    return out


_STEP_LIM_OVERRIDES: Dict[str, int] = _load_step_lim_overrides(_STEP_LIMITS_PATH)
_MISSING_TASK_NAME_WARNED: bool = False
# RoboTwin resets ``TASK_ENV.step_lim`` from its own upstream YAML at the start
# of every episode, so a naive ``prev != override`` check would re-log on every
# episode. Track the ``(task_name, override)`` pairs we have already announced
# and stay silent for the rest of the process.
_LOGGED_OVERRIDES: set = set()


def _parse_bool(value, default: bool = False) -> bool:
    """Parse YAML/CLI boolean values without treating "false" as truthy."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off", "none", "null", ""):
            return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def _parse_optional_int(value, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(
            f"{field_name} must be a positive integer or null, got {value!r}"
        )
    return parsed


def _apply_step_lim_override(task_env) -> None:
    # First step of each episode: apply per-task step_lim override (if any).
    # The RoboTwin eval loop re-checks ``task_env.step_lim`` on every iteration,
    # so mutating it here takes effect from the next iteration onward.
    if not _STEP_LIM_OVERRIDES or getattr(task_env, "take_action_cnt", -1) != 0:
        return
    task_name = getattr(task_env, "task_name", None)
    if not task_name:
        global _MISSING_TASK_NAME_WARNED
        if not _MISSING_TASK_NAME_WARNED:
            _MISSING_TASK_NAME_WARNED = True
            print(
                "[SanaWAMClient] step_lim overrides loaded but TASK_ENV.task_name "
                f"is missing/empty ({task_name!r}); overrides will not be applied."
            )
        return
    override = _STEP_LIM_OVERRIDES.get(task_name)
    if override is None or getattr(task_env, "step_lim", None) == override:
        return
    prev = getattr(task_env, "step_lim", None)
    task_env.step_lim = override
    key = (task_name, override)
    if key not in _LOGGED_OVERRIDES:
        _LOGGED_OVERRIDES.add(key)
        print(f"[SanaWAMClient] step_lim override: {task_name} {prev} -> {override}")


class ModelClient:
    """RoboTwin ``ModelClient`` backed by the sana-wam Policy Server.

    The server manages action chunking internally, so this client calls
    ``POST /predict`` every step and lets the server decide whether to run
    full diffusion inference or pop a cached action from its buffer.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        http_port: int = 8848,
        send_state: bool = True,
        state_dim: Optional[int] = None,
        request_timeout: int = 300,
        action_indices: Optional[list] = None,
        action_type: str = "qpos",
        debug: bool = False,
        debug_dir: str = "./debug_images",
        telemetry_enabled: bool = False,
        telemetry_dir: str = "./telemetry/client",
        telemetry_flush_every: int = 32,
        **kwargs,  # absorb unused YAML fields for backwards compatibility
    ) -> None:
        """
        Args:
            host:            sana-wam server hostname / IP.
            http_port:       sana-wam HTTP port (default 8848).
            send_state:      Whether to include the robot proprioceptive state
                             vector in the ``/predict`` request.
            state_dim:       Optional expected proprio dimension. When set, the
                             client fails fast if the extracted state does not
                             match the checkpoint's architecture.state_dim.
            request_timeout: HTTP timeout in seconds.
            action_indices:  Optional index list to reorder the returned action
                             vector before passing it to the environment.
                             None = no reordering.
            action_type:     How the returned action is passed to ``take_action``.
                             ``'qpos'`` means 14D joint angles straight through;
                             ``'ee'`` converts the server's 20D EEF output to
                             the 16D end-effector action expected by RoboTwin.
            debug:           Save per-step images + JSON metadata under debug_dir.
            debug_dir:       Root directory for debug artifacts.
        """
        _VALID_ACTION_TYPES = ("qpos", "ee")
        if action_type not in _VALID_ACTION_TYPES:
            raise ValueError(
                f"[SanaWAMClient] Unsupported action_type={action_type!r}; "
                f"expected one of {_VALID_ACTION_TYPES}. "
                f"Note: EEF-mode training uses 'ee' (NOT 'eef')."
            )

        self._send_state = send_state
        self._state_dim = state_dim
        self._request_timeout = request_timeout
        self._action_indices = action_indices
        self._action_type = action_type
        self._task_description = ""
        self._debug = debug
        self._debug_dir = debug_dir
        self._episode = -1  # incremented to 0 on the first reset_model() call
        self._step = 0
        self._episode_context: dict = {}
        self._episode_key: Optional[str] = None
        self._noise_pair_key: Optional[str] = None
        self._awaiting_first_prompt = False
        self._prompt_revision = 0
        self._reset_session_id = os.environ.get(
            "ROBOTWIN_RESET_SESSION_ID", uuid.uuid4().hex[:12]
        )
        self._pending_reset: Optional[dict] = None
        self._last_interaction: dict = {}
        self._telemetry = ClientTelemetryRecorder(
            enabled=telemetry_enabled,
            output_dir=telemetry_dir,
            flush_every=telemetry_flush_every,
        )

        self._server = f"http://{host}:{http_port}"

        # Warn about legacy YAML fields once so users know they're no-ops now.
        for field in _DEPRECATED_YAML_FIELDS:
            if field in kwargs:
                print(
                    f"[SanaWAMClient] Ignored legacy config field '{field}'={kwargs[field]!r}. "
                    f"Server decides multi-view and camera layout from the checkpoint's "
                    f"config.yaml — drop this field from policy_config.yml."
                )

        print(
            f"[SanaWAMClient] server={self._server} send_state={send_state} "
            f"state_dim={state_dim} "
            f"request_timeout={request_timeout}s action_type={action_type} "
            f"action_indices={action_indices} debug={debug} debug_dir={debug_dir}"
        )

        self._wait_until_healthy()
        info = client.get(self._server, "/info")
        self._server_info = dict(info)
        self._episode_noise_mode = info.get("inference_runtime", {}).get(
            "episode_noise_mode", "ambient"
        )
        if os.environ.get("ROBOTWIN_DAGGER_PLAN") and _STEP_LIM_OVERRIDES:
            raise RuntimeError(
                "live DAgger collection requires zero repository step-limit overrides"
            )
        self._dagger = LiveTakeoverRecorder.from_env(info)
        if self._dagger.enabled:
            if self._action_type != "ee" or self._action_indices is not None:
                raise RuntimeError(
                    "live DAgger collection requires action_type='ee' and no action_indices"
                )
            atexit.register(self._dagger.close)

    def set_episode_context(
        self,
        *,
        task_name=None,
        task_config=None,
        environment_seed=None,
        episode_index=None,
        prompt=None,
        prompt_sha256=None,
        prompt_source=None,
        prompt_manifest_ordinal=None,
        prompt_manifest_path=None,
        prompt_manifest_sha256=None,
        **_unused,
    ) -> None:
        """Receive the exact accepted RoboTwin scene identity from the wrapper."""
        new_context = {
            "task_name": task_name,
            "task_config": task_config,
            "environment_seed": environment_seed,
            "episode_index": episode_index,
        }
        prompt_context = {
            "prompt": prompt,
            "prompt_sha256": prompt_sha256,
            "prompt_source": prompt_source,
            "prompt_manifest_ordinal": prompt_manifest_ordinal,
            "prompt_manifest_path": prompt_manifest_path,
            "prompt_manifest_sha256": prompt_manifest_sha256,
        }
        new_context.update(
            {key: value for key, value in prompt_context.items() if value is not None}
        )
        if self._pending_reset is not None:
            pending_context = self._pending_reset.get("source_context")
            if pending_context != new_context:
                self._pending_reset = None
        self._episode_context = new_context

    def _wait_until_healthy(
        self, timeout_s: int = 300, poll_interval: float = 2.0
    ) -> None:
        deadline = time.monotonic() + timeout_s
        last_exc: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                if client.get(self._server, "/health").get("status") == "healthy":
                    print(f"[SanaWAMClient] Server healthy at {self._server}")
                    return
            except Exception as exc:
                last_exc = exc
            time.sleep(poll_interval)
        raise RuntimeError(
            f"sana-wam server did not become healthy within {timeout_s}s at {self._server}. Last error: {last_exc}"
        )

    def reset(
        self, task_description: str = "", reset_reason: str = "prompt_change"
    ) -> None:
        """Clear server episode state and (optionally) bump the debug episode counter.

        RoboTwin's ``reset_model()`` passes ``task_description=""`` at episode
        boundaries. We also reset internally when the task instruction changes
        mid-rollout, in which case ``task_description`` is non-empty.
        """
        is_boundary = task_description == ""
        pending = self._pending_reset
        if pending is None:
            context = dict(self._episode_context)
            environment_seed = context.get("environment_seed")
            if (
                is_boundary
                and self._episode_noise_mode == "paired"
                and environment_seed is None
            ):
                raise RuntimeError(
                    "paired episode noise requires RoboTwin wrapper scene metadata; "
                    "environment_seed is missing"
                )
            if is_boundary:
                next_episode = self._episode + 1
                next_prompt_revision = 0
                awaiting_first_prompt = True
                reset_reason = "episode_boundary"
                if environment_seed is None:
                    noise_pair_key = f"robotwin/fallback/episode-{next_episode:06d}"
                    episode_key = (
                        f"robotwin/session-{self._reset_session_id}/"
                        f"episode-{next_episode:06d}"
                    )
                else:
                    task_name = context.get("task_name") or "unknown_task"
                    task_config = context.get("task_config") or "unknown_config"
                    episode_index = context.get("episode_index")
                    episode_number = (
                        int(episode_index)
                        if episode_index is not None
                        else next_episode
                    )
                    noise_pair_key = f"robotwin/{task_name}/{task_config}/scene-{int(environment_seed)}"
                    episode_key = (
                        f"robotwin/session-{self._reset_session_id}/{task_name}/{task_config}/"
                        f"scene-{int(environment_seed)}/episode-{episode_number}"
                    )
            else:
                next_episode = self._episode
                next_prompt_revision = self._prompt_revision + 1
                awaiting_first_prompt = False
                noise_pair_key = self._noise_pair_key
                episode_key = f"{self._episode_key}/prompt-{next_prompt_revision}"
            metadata = {
                key: value for key, value in context.items() if value is not None
            }
            metadata.update(
                {
                    "reset_reason": reset_reason,
                    "noise_pair_key": noise_pair_key,
                    "reset_session_id": self._reset_session_id,
                }
            )
            pending = {
                "source_context": context,
                "is_boundary": is_boundary,
                "episode": next_episode,
                "episode_key": episode_key,
                "noise_pair_key": noise_pair_key,
                "prompt_revision": next_prompt_revision,
                "awaiting_first_prompt": awaiting_first_prompt,
                "task_description": task_description,
                "metadata": metadata,
            }
            self._pending_reset = pending

        result = client.reset(
            self._server,
            timeout=30,
            episode_key=pending["episode_key"],
            metadata=pending["metadata"],
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"[SanaWAMClient] Server reset failed: {result}")

        self._episode = pending["episode"]
        self._episode_key = pending["episode_key"]
        self._noise_pair_key = pending["noise_pair_key"]
        self._prompt_revision = pending["prompt_revision"]
        self._awaiting_first_prompt = pending["awaiting_first_prompt"]
        self._task_description = pending["task_description"]
        if pending["is_boundary"]:
            self._step = 0
            if self._debug:
                ep_dir = os.path.join(self._debug_dir, f"ep{self._episode:04d}")
                os.makedirs(ep_dir, exist_ok=True)
                print(f"[SanaWAMClient] debug images → {ep_dir}")
            telemetry_context = dict(pending["metadata"])
            telemetry_context["episode_key"] = self._episode_key
            self._telemetry.start_episode(telemetry_context, result)
            dagger = getattr(self, "_dagger", None)
            if dagger is not None and dagger.enabled:
                dagger.start_episode(telemetry_context)
        self._pending_reset = None

    def _save_debug_step(
        self,
        cams: dict,
        prompt: str,
        state_list: Optional[list],
        response: dict,
    ) -> None:
        """Save all per-step debug data under debug_dir/ep{N:04d}/step_{N:04d}/.

        Files written:
          head.jpg / left.jpg / right.jpg  — raw per-camera frames as client sent
                                             (missing wrist → *_missing.txt stub)
          meta.json                        — prompt, state, action, latency, step, episode
        """
        ep_dir = os.path.join(self._debug_dir, f"ep{self._episode:04d}")
        step_dir = os.path.join(ep_dir, f"step_{self._step:04d}")
        os.makedirs(step_dir, exist_ok=True)

        for name in ("head", "left", "right"):
            img = cams.get(name)
            if img is None:
                with open(os.path.join(step_dir, f"{name}_missing.txt"), "w") as f:
                    f.write("client sent None for this camera\n")
            else:
                cv.imwrite(
                    os.path.join(step_dir, f"{name}.jpg"),
                    cv.cvtColor(img, cv.COLOR_RGB2BGR),
                )

        meta = {
            "episode": self._episode,
            "step": self._step,
            "prompt": prompt,
            "state": state_list,
            "action": response.get("action"),
            "server_step": response.get("step"),
            "latency_ms": response.get("latency_ms"),
        }
        with open(os.path.join(step_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def save_debug_terminal(self, cams: dict, state) -> None:
        if not self._debug:
            return
        terminal_dir = os.path.join(
            self._debug_dir, f"ep{self._episode:04d}", "terminal"
        )
        os.makedirs(terminal_dir, exist_ok=True)
        for name in ("head", "left", "right"):
            image = cams.get(name)
            if image is None:
                raise ValueError(f"terminal debug camera {name!r} is missing")
            cv.imwrite(
                os.path.join(terminal_dir, f"{name}.jpg"),
                cv.cvtColor(image, cv.COLOR_RGB2BGR),
            )
        with open(os.path.join(terminal_dir, "meta.json"), "w") as stream:
            json.dump(
                {
                    "episode": self._episode,
                    "step": self._step,
                    "state": np.asarray(state, dtype=np.float32).reshape(-1).tolist(),
                    "success": True,
                },
                stream,
                indent=2,
            )

    def step(self, example: dict, step: int = 0) -> np.ndarray:
        """
        Submit one observation to the server and return the next action.

        Args:
            example: {
                "cams": {
                    "head":  np.ndarray,      # HxWx3 RGB uint8, required
                    "left":  np.ndarray|None, # HxWx3 RGB uint8, optional
                    "right": np.ndarray|None, # HxWx3 RGB uint8, optional
                },
                "lang":  str,                 # task instruction
                "state": np.ndarray,          # proprioceptive state (optional)
            }
            step: current episode timestep (unused; server manages chunking).

        Returns:
            action: np.ndarray, shape (action_dim,)
        """
        cams = example["cams"]
        prompt = str(example.get("lang", self._task_description))
        if self._episode_context.get(
            "prompt_source"
        ) == "manifest" and prompt != self._episode_context.get("prompt"):
            raise RuntimeError(
                "RoboTwin prompt changed during strict manifest replay: "
                f"{prompt!r} != {self._episode_context.get('prompt')!r}"
            )

        # Mirror the upstream pattern: reset if the task instruction changes.
        if self._awaiting_first_prompt:
            self._awaiting_first_prompt = False
            if prompt != self._task_description:
                self._task_description = prompt
        elif prompt and prompt != self._task_description:
            self.reset(prompt)

        state_arr = example.get("state", None)
        state_list: Optional[list] = None
        if self._send_state:
            if state_arr is None:
                raise ValueError(
                    "[SanaWAMClient] send_state=True requires example['state']. "
                    "New proprio-conditioned checkpoints need this field; set "
                    "send_state=false only for checkpoints trained without state."
                )
            state_np = np.asarray(state_arr, dtype=np.float32).reshape(-1)
            if self._state_dim is not None and state_np.size != self._state_dim:
                raise ValueError(
                    f"[SanaWAMClient] Extracted state_dim={state_np.size}, expected {self._state_dim}. "
                    "Check policy_config.yml: action_type/state_dim must match the checkpoint's "
                    "dataloader.action_mode and architecture.state_dim."
                )
            state_list = [float(v) for v in state_np]

        encoded_cams = {
            "head": client.encode_numpy_b64(cams["head"]),
            "left": client.encode_numpy_b64(cams["left"])
            if cams.get("left") is not None
            else None,
            "right": client.encode_numpy_b64(cams["right"])
            if cams.get("right") is not None
            else None,
        }
        payload = client.build_payload(
            head=encoded_cams["head"],
            left_wrist=encoded_cams["left"],
            right_wrist=encoded_cams["right"],
            prompt=prompt,
            state=state_list,
        )
        response = client.post(
            self._server, "/predict", payload, timeout=self._request_timeout
        )
        action = np.array(response["action"], dtype=np.float32)
        server_action = action.copy()

        self._step += 1
        if self._debug:
            self._save_debug_step(cams, prompt, state_list, response)

        if self._action_indices is not None:
            action = action[self._action_indices]

        self._last_interaction = {
            "env_step": int(step),
            "prompt": prompt,
            "pre_state": state_list,
            "server_action": server_action,
            "policy_action": action.copy(),
            "response": dict(response),
            "camera_jpegs": {
                key: None if value is None else base64.b64decode(value, validate=True)
                for key, value in encoded_cams.items()
            },
        }

        return action

    def record_execution(
        self,
        *,
        sent_action,
        post_state,
        success: bool,
        post_state_error: str | None = None,
    ) -> None:
        if not self._telemetry.enabled:
            return
        interaction = self._last_interaction
        self._telemetry.record_step(
            env_step=interaction.get("env_step", self._step - 1),
            prompt=interaction.get("prompt", self._task_description),
            pre_state=interaction.get("pre_state"),
            server_action=interaction.get("server_action"),
            policy_action=interaction.get("policy_action"),
            sent_action=sent_action,
            post_state=post_state,
            response=interaction.get("response", {}),
            success=success,
            post_state_error=post_state_error,
        )


# ---------------------------------------------------------------------------
# Module-level entry points called by RoboTwin's eval_policy.py
# ---------------------------------------------------------------------------


def get_model(usr_args: dict) -> ModelClient:
    return ModelClient(
        host=usr_args.get("host", "127.0.0.1"),
        http_port=int(usr_args.get("http_port", usr_args.get("port", 8848))),
        send_state=_parse_bool(usr_args.get("send_state", True), default=True),
        state_dim=_parse_optional_int(usr_args.get("state_dim", None), "state_dim"),
        request_timeout=int(usr_args.get("request_timeout", 300)),
        action_indices=usr_args.get("action_indices", None),
        action_type=usr_args.get("action_type", "qpos"),
        debug=_parse_bool(usr_args.get("debug", False), default=False),
        debug_dir=usr_args.get("debug_dir", "./debug_images"),
        telemetry_enabled=_parse_bool(
            usr_args.get("telemetry_enabled", False), default=False
        ),
        telemetry_dir=usr_args.get("telemetry_dir", "./telemetry/client"),
        telemetry_flush_every=int(usr_args.get("telemetry_flush_every", 32)),
        # Pass through everything else so legacy-field warnings can fire.
        **{k: v for k, v in usr_args.items() if k in _DEPRECATED_YAML_FIELDS},
    )


def reset_model(model: ModelClient) -> None:
    model.reset(task_description="")


def _extract_eef_proprio(observation: dict) -> np.ndarray:
    endpose = observation.get("endpose")
    if not isinstance(endpose, dict):
        available = ", ".join(sorted(observation.keys()))
        raise KeyError(
            "action_type='ee' requires RoboTwin endpose proprio matching training action_mode='eef'. "
            "Expected observation['endpose'] with left_endpose, right_endpose, left_gripper, right_gripper. "
            f"Available top-level observation keys: {available}"
        )

    return action_conversion.robotwin_endpose_to_eef20d(
        endpose["left_endpose"],
        endpose["right_endpose"],
        endpose["left_gripper"],
        endpose["right_gripper"],
    )


def _extract_proprio(model: ModelClient, observation: dict) -> np.ndarray:
    if model._action_type == "ee":
        return _extract_eef_proprio(observation)
    try:
        return np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    except KeyError as exc:
        available = ", ".join(sorted(observation.keys()))
        raise KeyError(
            "action_type='qpos' requires RoboTwin joint proprio matching training action_mode='joint'. "
            "Expected observation['joint_action']['vector']. "
            f"Available top-level observation keys: {available}"
        ) from exc


def _extract_direct_proprio(model: ModelClient, task_env) -> np.ndarray:
    """Read post-command robot state without rendering another camera observation."""
    if model._action_type == "ee":
        return action_conversion.robotwin_endpose_to_eef20d(
            task_env.get_arm_pose("left"),
            task_env.get_arm_pose("right"),
            task_env.robot.get_left_gripper_val(),
            task_env.robot.get_right_gripper_val(),
        )
    return np.asarray(
        task_env.robot.get_left_arm_jointState()
        + task_env.robot.get_right_arm_jointState(),
        dtype=np.float32,
    )


def eval(TASK_ENV, model: ModelClient, observation: dict) -> None:
    """Per-step callback invoked by RoboTwin's eval_policy.py.

    RoboTwin exposes three per-camera entries under ``observation["observation"]``
    (``head_camera`` / ``left_camera`` / ``right_camera``). They map positionally
    to the sana-wam client API's fixed fields (head / left_wrist / right_wrist).
    ``front_camera`` is ignored — it's not part of the server contract.
    """
    _apply_step_lim_override(TASK_ENV)

    instruction = TASK_ENV.get_instruction()
    obs = observation["observation"]

    example = {
        "cams": {
            "head": obs["head_camera"]["rgb"],
            "left": obs.get("left_camera", {}).get("rgb"),
            "right": obs.get("right_camera", {}).get("rgb"),
        },
        "lang": str(instruction),
        "state": _extract_proprio(model, observation) if model._send_state else None,
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    dagger = getattr(model, "_dagger", None)
    if dagger is not None and dagger.enabled:
        interaction = model._last_interaction
        outcome = dagger.process_step(
            task_env=TASK_ENV,
            env_step=TASK_ENV.take_action_cnt,
            prompt=str(instruction),
            camera_jpegs=interaction.get("camera_jpegs", {}),
            pre_state=example["state"],
            policy_action=action,
            policy_info=interaction.get("response", {}).get("policy", {}),
            extract_state=lambda env: _extract_direct_proprio(model, env),
        )
        if outcome is not None and outcome.consumed:
            if model._telemetry.enabled:
                post_state = None
                post_state_error = None
                try:
                    post_state = _extract_direct_proprio(model, TASK_ENV)
                except Exception as exc:
                    post_state_error = f"{type(exc).__name__}: {exc}"
                model.record_execution(
                    sent_action=None,
                    post_state=post_state,
                    success=outcome.success,
                    post_state_error=post_state_error,
                )
            return

    # EEF mode: convert 20D (xyz+rot6d+grip)×2 → 16D (xyz+quat+grip)×2.
    if model._action_type == "ee" and len(action) == 20:
        action = action_conversion.eef20d_to_ee16d(action)

    TASK_ENV.take_action(action, action_type=model._action_type)

    success = bool(getattr(TASK_ENV, "eval_success", False))
    post_state = None
    post_state_error = None
    debug_enabled = bool(getattr(model, "_debug", False))
    if model._telemetry.enabled or (debug_enabled and success):
        try:
            post_state = _extract_direct_proprio(model, TASK_ENV)
        except Exception as exc:  # diagnostics must never invalidate evaluation
            post_state_error = f"{type(exc).__name__}: {exc}"
    if debug_enabled and success:
        terminal = TASK_ENV.get_obs()["observation"]
        model.save_debug_terminal(
            {
                "head": terminal["head_camera"]["rgb"],
                "left": terminal.get("left_camera", {}).get("rgb"),
                "right": terminal.get("right_camera", {}).get("rgb"),
            },
            post_state,
        )
    if model._telemetry.enabled:
        model.record_execution(
            sent_action=action,
            post_state=post_state,
            success=success,
            post_state_error=post_state_error,
        )
