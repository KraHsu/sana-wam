"""Closed-loop inference engine for the block-autoregressive SANA WAM.

The block-AR architecture (:class:`DualSystemARArchitecture`) is *stateful*:
``ar_rollout`` walks a sequence of realized observations, ingesting each one as
confirmed clean history into a persistent linear-state KV cache and emitting one
action chunk per step. That does not fit the one-shot ``engine.generate`` seam
the deploy server / :class:`WAMPolicy` are built around — which is stateless and
re-runs from scratch each call.

:class:`ARInferenceEngine` bridges the two: it owns the persistent cache, driver,
flow-matching schedules, monotonic RNG, step counter, prompt-context cache and a
rolling pixel-frame buffer, and executes **exactly one AR step per ``generate``
call** by reusing the architecture's own per-step helpers verbatim
(``_rollout_step_context`` / ``_ingest_clean_video`` /
``_denoise_action_chunk``). Driving ``generate`` N times over the realized obs
stream is therefore numerically identical to a single ``ar_rollout`` over those
same observations (see ``tests/test_ar_rollout_engine.py`` equivalence gate).

Required usage (see ``configs/deploy_ar_sana.yaml``):

- :class:`WAMPolicy` MUST run in **greedy** mode (``execute_horizon=null``,
  ``temporal_ensemble=false``, ``async_inference.mode=none``). Each ``generate``
  ingests one realized obs and advances the cache by one chunk; receding-horizon,
  temporal ensembling and async all double-advance / reorder the cache and break
  the AR frame alignment.
- The policy/server must call :meth:`reset` at each episode boundary (wired via
  ``WAMPolicy.reset`` → ``engine.reset``) so the cache + step counter + pixel
  buffer are cleared between episodes.

Deploy-semantics knobs (eval-verification targets, handoff §2c F3/F4):

- ``inference.ar_obs_chunk_mode``: ``rolling_buffer`` (default) rebuilds the obs
  clip from the per-sim-step ``obs_history`` (WAMPolicy appends one frame every
  step) by taking the last ``num_frames`` frames and subsampling by
  ``video_stride`` — exactly the dataset's ``range(0, num_frames, video_stride)``
  — so the encoded latents match training's pixel cadence/motion. (A naive
  one-frame-per-``generate`` buffer would be ~``action_tokens_per_chunk``× too
  sparse and motion-off-distribution.) ``repeat_frame`` tiles the current frame
  (zero motion; debug only).
- ``inference.ar_obs_latent_band``: ``auto`` (default) takes the leading latents
  (training chunk 0, incl. the causal first-frame token) at episode start (step 0,
  whose obs is assigned rope[0,1]) and the trailing latents (latest chunk) after —
  matching how training assigns the first-chunk vs continuation-chunk band.
  ``leading`` / ``trailing`` force one band (F3 eval A/B).
- ``inference.ar_proprio_mode``: ``per_step`` (default) uses the rolling
  113-frame window's oldest state for the clip-level context token and the
  current state for per-chunk AdaLN, matching training's two proprio paths.
  ``single`` latches the episode's first state as a debug ablation.
- Every realized observation chunk ``c`` emits the aligned action chunk ``c``.
  In interleaved frame ids this is video ``2c`` followed by action ``2c+1``;
  closed-loop deployment must not skip action frame 3 or replace the observed
  video condition with an untrained predicted-video look-ahead.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from numbers import Integral
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from sana_wam.deploy.base import BaseInferenceEngine
from sana_wam.model.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)

_PROMPT_CTX_CACHE_MAXSIZE = 32


class ARInferenceEngine(BaseInferenceEngine):
    """One-AR-step-per-``generate`` engine for ``DualSystemARArchitecture``."""

    require_architecture = True

    def __init__(
        self,
        cfg,
        architecture: Optional[BaseWAMArchitecture] = None,
        action_backbone=None,
    ):
        super().__init__(
            cfg, architecture=architecture, action_backbone=action_backbone
        )
        arch = self.architecture
        if arch.video_backbone is None or arch.action_backbone is None:
            raise ValueError(
                "ARInferenceEngine requires an architecture with both backbones."
            )
        self._device = arch.device
        self._dtype = arch.dtype
        # Video denoising always retains an FP32 Euler master state; this is a
        # correctness contract rather than a sampler-selection knob.
        self._video_euler_master_dtype = "float32"

        # --- AR knobs off the architecture (authoritative) ---
        self._fcs = int(getattr(arch, "_ar_frame_chunk_size", 1))
        self._window = int(getattr(arch, "_ar_attn_window", 10_000))

        # --- deploy knobs off cfg.inference (with sane fallbacks) ---
        def _inf(key, default=None):
            return OmegaConf.select(cfg, f"inference.{key}", default=default)

        denoise_steps = int(_inf("denoise_steps", 4) or 4)
        self._video_steps = int(_inf("video_steps", denoise_steps) or denoise_steps)
        self._action_steps = int(_inf("action_steps", denoise_steps) or denoise_steps)
        self._action_scheduler_shift = float(_inf("shift", 5.0))
        if (
            not math.isfinite(self._action_scheduler_shift)
            or self._action_scheduler_shift <= 0.0
        ):
            raise ValueError("inference.shift must be a positive finite number")
        seed_cfg = _inf("seed", None)
        self._seed: Optional[int] = None if seed_cfg is None else int(seed_cfg)
        self._episode_noise_mode = (
            str(_inf("episode_noise_mode", "ambient")).strip().lower()
        )
        if self._episode_noise_mode not in {"ambient", "paired"}:
            raise ValueError(
                "inference.episode_noise_mode must be 'ambient' or 'paired', got "
                f"{self._episode_noise_mode!r}"
            )
        self._episode_noise_base_seed = int(_inf("episode_noise_base_seed", 0) or 0)
        self._generation_noise_schedule = str(
            _inf("generation_noise_schedule.mode", "monotonic")
        ).strip().lower()
        if self._generation_noise_schedule not in {"monotonic", "common_future"}:
            raise ValueError(
                "inference.generation_noise_schedule.mode must be 'monotonic' or "
                f"'common_future', got {self._generation_noise_schedule!r}"
            )
        self._common_future_noise_base_seed = int(
            _inf("generation_noise_schedule.base_seed", 0) or 0
        )
        self._last_generation_noise_seed: Optional[int] = None
        self._cache_feedback_mode = (
            str(_inf("cache_feedback_mode", "predicted")).strip().lower()
        )
        if self._cache_feedback_mode not in {
            "predicted",
            "measured",
            "reencode_predicted",
        }:
            raise ValueError(
                "inference.cache_feedback_mode must be 'predicted', 'measured', or "
                "'reencode_predicted', got "
                f"{self._cache_feedback_mode!r}"
            )
        self._cache_feedback_fallback = (
            str(_inf("cache_feedback_fallback", "predicted")).strip().lower()
        )
        if self._cache_feedback_fallback not in {"predicted", "error"}:
            raise ValueError(
                "inference.cache_feedback_fallback must be 'predicted' or 'error', got "
                f"{self._cache_feedback_fallback!r}"
            )
        self._episode_context: dict = {}
        self._episode_key: Optional[str] = None
        self._noise_pair_key: Optional[str] = None
        self._current_noise_seed: Optional[int] = (
            self._seed if self._episode_noise_mode == "ambient" else None
        )
        self._pending_action_feedback: Optional[dict] = None
        self._last_feedback: dict = {"status": "none", "reason": "episode_start"}
        self._feedback_commits = 0
        self._feedback_fallbacks = 0
        # Delta actions are only wired through the cross-attn deploy engine. Fail
        # loudly rather than silently emitting raw deltas as absolute commands.
        if bool(OmegaConf.select(cfg, "dataloader.delta_action", default=False)):
            raise NotImplementedError(
                "dataloader.delta_action=true is not supported by the AR engine; "
                "use the cross-attn engine, which reconstructs absolute = "
                "unnormalize(delta + normalize(current_state))."
            )
        self._obs_chunk_mode = str(_inf("ar_obs_chunk_mode", "rolling_buffer"))
        self._proprio_mode = str(_inf("ar_proprio_mode", "per_step"))
        self._reset_cache_each_generation = bool(
            _inf("ar_reset_cache_each_generation", False)
        )
        # Which latent band of the freshly-encoded clip is the current obs chunk.
        #   auto (default) = leading at step 0, trailing after — matches training,
        #     whose chunk 0 is lat[0:fcs] (incl. the causal first-frame token) @ rope[0,1]
        #     and whose later/current chunk is the trailing band.
        #   leading  = always first latents (training chunk-0 band).
        #   trailing = always most-recent latents (a training clip's last chunk).
        self._obs_latent_band = str(_inf("ar_obs_latent_band", "auto"))

        self._video_num_frames = int(self._resolve_video_num_frames(cfg))
        self._height = int(_inf("height", 480) or 480)
        self._width = int(_inf("width", 832) or 832)
        # Raw (pre-stride) frame window + stride — the engine must rebuild the obs
        # clip at the SAME pixel cadence training used (dataset takes
        # range(0, num_frames, video_stride)); greedy WAMPolicy otherwise only
        # hands the engine one frame per chunk (~24 sim steps apart) vs training's
        # video_stride (~4), making every obs latent motion-off-distribution.
        self._raw_num_frames = int(
            OmegaConf.select(cfg, "inference.num_frames", default=None)
            or OmegaConf.select(cfg, "dataloader.num_frames", default=33)
        )
        self._video_stride = max(
            1, int(OmegaConf.select(cfg, "dataloader.video_stride", default=1) or 1)
        )
        # VAE temporal compression (Wan=4 default, LTX2/SANA-WM=8) — see GDNAR engine.
        self._temporal_compression = max(
            1,
            int(
                OmegaConf.select(cfg, "dataloader.temporal_compression", default=4) or 4
            ),
        )

        # --- action geometry (from the trained backbone, never the YAML literal) ---
        self._action_dim = int(arch.action_dim)
        self._action_tokens_per_chunk = self._resolve_action_tokens_per_chunk(cfg)
        latent_frames = 1 + (
            self._video_num_frames - 1
        ) // self._temporal_compression
        self._video_chunks_per_window = latent_frames // self._fcs
        if self._reset_cache_each_generation and (
            latent_frames != self._fcs or self._video_chunks_per_window != 1
        ):
            raise ValueError(
                "ar_reset_cache_each_generation requires exactly one video chunk "
                f"per causal window, got latent_frames={latent_frames}, fcs={self._fcs}"
            )

        # Optional outcome-trained best-of-N selection is deliberately restricted
        # to generation zero, which is the only distribution represented by its
        # training data.
        self._generation_zero_rerank_enabled = bool(
            _inf("generation_zero_rerank.enabled", False)
        )
        self._generation_zero_rerank_candidate_count = int(
            _inf("generation_zero_rerank.candidate_count", 5) or 5
        )
        self._generation_zero_rerank_count = 0
        self._last_generation_zero_rerank: Optional[dict] = None
        self._outcome_ranker = None
        if self._generation_zero_rerank_enabled:
            if self._generation_zero_rerank_candidate_count < 2:
                raise ValueError(
                    "inference.generation_zero_rerank.candidate_count must be >= 2"
                )
            ranker_path = _inf("generation_zero_rerank.ranker_path", None)
            expected_sha256 = _inf(
                "generation_zero_rerank.expected_ranker_sha256", None
            )
            if not ranker_path or not expected_sha256:
                raise ValueError(
                    "enabled generation_zero_rerank requires ranker_path and "
                    "expected_ranker_sha256"
                )
            from sana_wam.deploy.outcome_ranker import OutcomeActionRanker

            self._outcome_ranker = OutcomeActionRanker.load(
                ranker_path, str(expected_sha256)
            )
            ranker_identity = self._outcome_ranker.identity
            expected_geometry = (
                ranker_identity["action_tokens"],
                ranker_identity["action_dim"],
            )
            engine_geometry = (self._action_tokens_per_chunk, self._action_dim)
            if engine_geometry != expected_geometry:
                raise ValueError(
                    f"engine action geometry {engine_geometry} does not match "
                    f"outcome ranker {expected_geometry}"
                )

        # --- persistent rollout state (per-step frames come from WAMPolicy.obs_history) ---
        self._prompt_ctx_cache: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._latched_proprio: Optional[np.ndarray] = None
        self._init_schedules_and_cache()
        self._episode_generation_index = 0

        logger.info(
            "ARInferenceEngine ready: fcs=%d window=%d video_steps=%d action_steps=%d "
            "action_dim=%d tokens/chunk=%d obs_chunk_mode=%s proprio_mode=%s "
            "noise_mode=%s cache_feedback=%s vnf=%d res=%dx%d g0_rerank=%s candidates=%d",
            self._fcs,
            self._window,
            self._video_steps,
            self._action_steps,
            self._action_dim,
            self._action_tokens_per_chunk,
            self._obs_chunk_mode,
            self._proprio_mode,
            self._episode_noise_mode,
            self._cache_feedback_mode,
            self._video_num_frames,
            self._height,
            self._width,
            self._generation_zero_rerank_enabled,
            self._generation_zero_rerank_candidate_count,
        )

    # ------------------------------------------------------------------ setup
    def _resolve_video_num_frames(self, cfg) -> int:
        vnf = OmegaConf.select(cfg, "inference.video_num_frames", default=None)
        if vnf is not None:
            return int(vnf)
        raw = int(
            OmegaConf.select(cfg, "inference.num_frames", default=None)
            or OmegaConf.select(cfg, "dataloader.num_frames", default=33)
        )
        stride = int(OmegaConf.select(cfg, "dataloader.video_stride", default=1) or 1)
        stride = stride if stride > 0 else 1
        return (raw - 1) // stride + 1

    def _resolve_action_tokens_per_chunk(self, cfg) -> int:
        override = OmegaConf.select(
            cfg, "inference.ar_action_tokens_per_chunk", default=None
        )
        # Latent frame count after the causal Wan VAE, then chunked.
        t_lat = 1 + (self._video_num_frames - 1) // self._temporal_compression
        num_chunks = max(1, t_lat // self._fcs)
        if override is not None:
            tokens = int(override)
        else:
            raw = int(
                OmegaConf.select(cfg, "inference.num_frames", default=None)
                or OmegaConf.select(cfg, "dataloader.num_frames", default=33)
            )
            total_actions = raw - 1  # dataset keeps frame 0 as proprio, 1.. as actions
            if total_actions % num_chunks != 0:
                raise ValueError(
                    f"action count {total_actions} not divisible by num_chunks={num_chunks} "
                    f"(t_lat={t_lat}, fcs={self._fcs}); set inference.ar_action_tokens_per_chunk."
                )
            tokens = total_actions // num_chunks
        if tokens <= 0:
            raise ValueError(f"action_tokens_per_chunk must be positive, got {tokens}")
        return tokens

    def _init_schedules_and_cache(self) -> None:
        """Build driver, flow-matching schedules, cache and RNG (mirrors ar_rollout)."""
        from sana_wam.model.ar.sana_ar_inference import ARLinearStateCache

        arch = self.architecture
        vb, ab = arch.video_backbone, arch.action_backbone
        self._driver = arch._mot_driver or arch.build_mot_driver()

        vb.scheduler.set_timesteps(self._video_steps)
        self._v_sigmas = [float(s) for s in vb.scheduler.sigmas.tolist()] + [0.0]
        self._v_ts = vb.scheduler.timesteps
        ab.scheduler.set_timesteps(
            self._action_steps, shift=self._action_scheduler_shift
        )
        self._a_sigmas = [float(s) for s in ab.scheduler.sigmas.tolist()] + [0.0]
        self._a_ts = ab.scheduler.timesteps
        self._action_scheduler_sigmas_sha256 = hashlib.sha256(
            np.asarray(self._a_sigmas, dtype="<f8").tobytes(order="C")
        ).hexdigest()

        self._cache = ARLinearStateCache(num_layers=vb.num_layers, window=self._window)
        self._gen = self._make_generator()
        self._step_c = 0

    def _make_generator(self) -> Optional[torch.Generator]:
        """Use ambient entropy by default; explicit seeds retain debug determinism."""
        seed = getattr(self, "_current_noise_seed", getattr(self, "_seed", None))
        if seed is None:
            return None
        return torch.Generator(device=self._device).manual_seed(seed)

    @staticmethod
    def derive_episode_noise_seed(base_seed: int, episode_key: str) -> int:
        """Derive a stable, domain-separated 63-bit model-noise seed."""
        material = f"sana-wam/model-noise/v1\0{int(base_seed)}\0{episode_key}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
            (1 << 63) - 1
        )

    @staticmethod
    def derive_rerank_candidate_seed(base_seed: int, candidate_index: int) -> int:
        """Derive an independent deterministic stream for a nonzero candidate."""
        if candidate_index <= 0:
            raise ValueError("rerank candidate_index must be positive")
        material = (
            "sana-wam/generation-zero-rerank/v1\0"
            f"{int(base_seed)}\0{int(candidate_index)}"
        ).encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
            (1 << 63) - 1
        )

    @staticmethod
    def derive_common_future_noise_seed(
        base_seed: int, noise_pair_key: str, generation_index: int
    ) -> int:
        """Derive candidate-independent noise for one post-bootstrap generation."""
        if generation_index < 1:
            raise ValueError("common-future generation_index must be positive")
        material = (
            "sana-wam/common-future-noise/v1\0"
            f"{int(base_seed)}\0{noise_pair_key}\0{int(generation_index)}"
        ).encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
            (1 << 63) - 1
        )

    def _prepare_generation_rng(self, generation_index: int) -> Optional[int]:
        """Select the RNG stream used by this generation and return its seed."""
        if generation_index != self._episode_generation_index:
            raise RuntimeError(
                "generation RNG index "
                f"{generation_index} != episode generation {self._episode_generation_index}"
            )
        if generation_index == 0:
            seed = self._current_noise_seed
        elif self._generation_noise_schedule == "common_future":
            if self._noise_pair_key is None:
                raise RuntimeError(
                    "common-future noise requires reset context field noise_pair_key "
                    "or episode_key"
                )
            seed = self.derive_common_future_noise_seed(
                self._common_future_noise_base_seed,
                self._noise_pair_key,
                generation_index,
            )
            self._gen = torch.Generator(device=self._device).manual_seed(seed)
        else:
            # The ordinary schedule consumes one monotonic generator stream, so
            # post-bootstrap generations do not have an independent seed.
            seed = None
        self._last_generation_noise_seed = seed
        return seed

    @staticmethod
    def _copy_branch_value(value):
        """Copy mutable containers while retaining immutable tensor references."""
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {
                key: ARInferenceEngine._copy_branch_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [ARInferenceEngine._copy_branch_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(ARInferenceEngine._copy_branch_value(item) for item in value)
        return value

    def _snapshot_rollout_state(self) -> dict:
        generator_state = None
        if self._gen is not None:
            generator_state = self._gen.get_state().clone()
        return {
            "cache": self._cache.snapshot(),
            "step_c": self._step_c,
            "episode_generation_index": self._episode_generation_index,
            "pending_action_feedback": self._copy_branch_value(
                self._pending_action_feedback
            ),
            "last_feedback": self._copy_branch_value(self._last_feedback),
            "feedback_commits": self._feedback_commits,
            "feedback_fallbacks": self._feedback_fallbacks,
            "latched_proprio": self._copy_branch_value(self._latched_proprio),
            "generator_state": generator_state,
        }

    def _restore_rollout_state(self, state: dict) -> None:
        self._cache.restore_snapshot(state["cache"])
        self._step_c = int(state["step_c"])
        self._episode_generation_index = int(state["episode_generation_index"])
        self._pending_action_feedback = self._copy_branch_value(
            state["pending_action_feedback"]
        )
        self._last_feedback = self._copy_branch_value(state["last_feedback"])
        self._feedback_commits = int(state["feedback_commits"])
        self._feedback_fallbacks = int(state["feedback_fallbacks"])
        self._latched_proprio = self._copy_branch_value(state["latched_proprio"])
        generator_state = state["generator_state"]
        if generator_state is None:
            self._gen = None
        else:
            self._gen = torch.Generator(device=self._device)
            self._gen.set_state(generator_state.clone())

    @torch.no_grad()
    def _rerank_generation_zero(
        self,
        obs_latent,
        context,
        seq_lens,
        proprio,
        *,
        context_proprio=None,
        feedback_plan: Optional[dict],
    ) -> dict:
        """Branch one pristine generation-zero state and retain only the winner."""
        if not self._generation_zero_rerank_enabled or self._outcome_ranker is None:
            raise RuntimeError("generation-zero reranker is not enabled")
        if (
            self._step_c != 0
            or self._episode_generation_index != 0
            or self._generation_zero_rerank_count != 0
        ):
            raise RuntimeError("outcome reranking is permitted exactly once at AR step zero")
        if self._current_noise_seed is None or self._gen is None:
            raise RuntimeError(
                "generation-zero reranking requires a deterministic episode model-noise seed"
            )

        base_seed = int(self._current_noise_seed)
        candidate_seeds = [base_seed] + [
            self.derive_rerank_candidate_seed(base_seed, index)
            for index in range(1, self._generation_zero_rerank_candidate_count)
        ]
        pristine = self._snapshot_rollout_state()
        candidates = []
        try:
            for index, seed in enumerate(candidate_seeds):
                self._restore_rollout_state(pristine)
                if index > 0:
                    self._gen = torch.Generator(device=self._device).manual_seed(seed)
                self._step_with_obs_latent(
                    obs_latent,
                    context,
                    seq_lens,
                    proprio,
                    context_proprio=context_proprio,
                    feedback_plan=feedback_plan,
                )
                actions_normalized = self._pending_action_feedback[
                    "predicted_actions_normalized"
                ].copy()
                score = float(self._outcome_ranker.score(actions_normalized))
                action_sha256 = hashlib.sha256(
                    np.ascontiguousarray(actions_normalized).tobytes(order="C")
                ).hexdigest()
                candidates.append(
                    {
                        "index": index,
                        "seed": seed,
                        "score": score,
                        "actions_normalized_sha256": action_sha256,
                        "state": self._snapshot_rollout_state(),
                    }
                )
            selected = max(candidates, key=lambda candidate: candidate["score"])
            self._restore_rollout_state(selected["state"])
        except Exception:
            self._restore_rollout_state(pristine)
            raise

        self._generation_zero_rerank_count += 1
        telemetry = {
            "ranker_sha256": self._outcome_ranker.sha256,
            "candidate_count": len(candidates),
            "candidate_seeds": candidate_seeds,
            "candidate_scores": [candidate["score"] for candidate in candidates],
            "candidate_actions_normalized_sha256": [
                candidate["actions_normalized_sha256"] for candidate in candidates
            ],
            "selected_candidate_index": selected["index"],
            "selected_candidate_seed": selected["seed"],
            "generation_index": 0,
            "ar_step_after": self._step_c,
        }
        self._last_generation_zero_rerank = telemetry.copy()
        return telemetry

    @staticmethod
    def _validate_noise_seed(value) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"model_noise_seed must be an integer, got {type(value).__name__}"
            )
        seed = int(value)
        if seed < 0 or seed >= 1 << 63:
            raise ValueError(f"model_noise_seed must be in [0, 2^63), got {seed}")
        return seed

    def _resolve_episode_noise_seed(self, episode_context: dict) -> Optional[int]:
        explicit = episode_context.get("model_noise_seed")
        if explicit is not None:
            return self._validate_noise_seed(explicit)
        if self._episode_noise_mode == "ambient":
            return self._seed
        noise_pair_key = episode_context.get("noise_pair_key") or episode_context.get(
            "episode_key"
        )
        if noise_pair_key is None or str(noise_pair_key).strip() == "":
            raise ValueError(
                "paired episode noise requires reset payload field 'noise_pair_key' or 'episode_key' "
                "or an explicit 'model_noise_seed'"
            )
        return self.derive_episode_noise_seed(
            self._episode_noise_base_seed, str(noise_pair_key)
        )

    # ------------------------------------------------------------- per-step core
    @torch.no_grad()
    def _step_with_obs_latent(
        self,
        obs_latent,
        context,
        seq_lens,
        proprio,
        feedback_plan: Optional[dict] = None,
        context_proprio=None,
    ) -> torch.Tensor:
        """One AR step from a ready obs *latent* — the loop body of ``ar_rollout``.

        Bypasses VAE / text encoding so the equivalence gate can drive it with the
        same mini backbones used to test ``ar_rollout``. Returns the predicted
        action chunk ``(B, action_tokens_per_chunk, action_dim)``.
        """
        arch = self.architecture
        obs = obs_latent.to(device=self._device, dtype=self._dtype)
        c = self._step_c
        obs_frame_id = 2 * c
        if (
            feedback_plan is not None
            and feedback_plan["frame_id"] != obs_frame_id - 1
        ):
            raise RuntimeError(
                "action cache-feedback frame is not adjacent to the current observation: "
                f"action={feedback_plan['frame_id']} obs={obs_frame_id}"
            )

        needs_feedback_transaction = (
            feedback_plan is not None or self._cache_feedback_mode != "predicted"
        )
        cache_snapshot = self._cache.snapshot() if needs_feedback_transaction else None
        feedback_state = (
            self._feedback_commits,
            self._feedback_fallbacks,
            self._last_feedback,
        )
        generator_state = (
            self._gen.get_state().clone()
            if cache_snapshot is not None and self._gen is not None
            else None
        )
        default_rng_state = None
        if cache_snapshot is not None and self._gen is None:
            if self._device.type == "cuda":
                default_rng_state = torch.cuda.get_rng_state(self._device).clone()
            elif self._device.type == "cpu":
                default_rng_state = torch.get_rng_state().clone()
        pending_feedback = self._pending_action_feedback

        def restore_rng_state() -> None:
            if generator_state is not None:
                self._gen.set_state(generator_state)
            elif default_rng_state is not None:
                if self._device.type == "cuda":
                    torch.cuda.set_rng_state(default_rng_state, self._device)
                else:
                    torch.set_rng_state(default_rng_state)

        try:
            self._cache.clear_pred()
            if context_proprio is None:
                context_proprio = proprio
            step_ctx, step_mask = arch._rollout_step_context(
                context, seq_lens, context_proprio
            )
            # F4: per-step proprio AdaLN deltas (in-distribution with per-chunk training).
            v_pe, a_pe = arch._rollout_proprio_deltas(proprio)

            if feedback_plan is not None and feedback_plan["frame_id"] < obs_frame_id:
                self._last_feedback = self._commit_action_cache_feedback(feedback_plan)

            # 1. ingest realized obs as confirmed clean video chunk at frame 2c.
            arch._ingest_clean_video(
                self._driver,
                self._cache,
                obs,
                frame_id=obs_frame_id,
                context=step_ctx,
                context_mask=step_mask,
                v_proprio=v_pe,
            )
        except Exception:
            if cache_snapshot is not None:
                self._cache.restore_snapshot(cache_snapshot)
                restore_rng_state()
                (
                    self._feedback_commits,
                    self._feedback_fallbacks,
                    self._last_feedback,
                ) = feedback_state
            raise
        try:
            # Training pairs clean video chunk c (frame 2c) with noisy action
            # chunk c (frame 2c+1).  Emit that exact successor here.  The former
            # look-ahead path generated video 2c+2 and action 2c+3, which skipped
            # action frame 3 after bootstrap and conditioned actions on a video
            # branch that R4 did not train.
            action_frame_id = 2 * c + 1
            pred_action = arch._denoise_action_chunk(
                self._driver,
                self._cache,
                frame_id=action_frame_id,
                batch=obs.shape[0],
                action_tokens=self._action_tokens_per_chunk,
                a_sigmas=self._a_sigmas,
                a_ts=self._a_ts,
                context=step_ctx,
                context_mask=step_mask,
                gen=self._gen,
                a_proprio=a_pe,
            )
            actions_normalized = pred_action.squeeze(0).float().cpu().numpy()
            actions = actions_normalized.copy()
            normalizer = getattr(self.architecture, "action_normalizer", None)
            if normalizer is not None:
                actions = normalizer.unnormalize(actions)
            actions = np.asarray(actions, dtype=np.float32)
            self._pending_action_feedback = {
                "frame_id": action_frame_id,
                "context": step_ctx,
                "context_mask": step_mask,
                "a_proprio": a_pe,
                "action_tokens": self._action_tokens_per_chunk,
                "source_generation_index": c,
                # Save the exact model outputs used by this generation. In
                # particular, never reconstruct normalized actions by applying
                # normalize() to the public/raw command on the next generation.
                "predicted_actions": actions.copy(),
                "predicted_actions_normalized": actions_normalized.copy(),
            }
            self._step_c += 1
            return pred_action
        except Exception:
            if cache_snapshot is not None:
                self._cache.restore_snapshot(cache_snapshot)
                restore_rng_state()
                self._pending_action_feedback = pending_feedback
                self._step_c = c
                (
                    self._feedback_commits,
                    self._feedback_fallbacks,
                    self._last_feedback,
                ) = feedback_state
            raise

    # ----------------------------------------------------------------- generate
    def _begin_fresh_single_chunk_generation(self, conditions: dict) -> None:
        """Discard the prior one-chunk cache at an exact greedy boundary.

        The episode-global RNG / telemetry index remains monotonic.  Only the
        model-local state is rebased so every request has the same video-0 ->
        action-1 graph used by causal single-chunk training.
        """

        if not getattr(self, "_reset_cache_each_generation", False):
            return
        if self._episode_generation_index == 0:
            if self._step_c != 0:
                raise RuntimeError("fresh single-chunk episode starts at local step zero")
            return
        if self._step_c != 1:
            raise RuntimeError(
                "fresh single-chunk deployment expected one completed local chunk"
            )
        cadence = self._feedback_cadence(conditions)
        if cadence != "ok":
            raise RuntimeError(
                f"invalid fresh single-chunk generation cadence: {cadence}"
            )
        self._cache.reset()
        self._step_c = 0
        self._pending_action_feedback = None
        self._last_feedback = {
            "status": "fresh_cache",
            "reason": "causal_single_chunk_boundary",
        }

    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        if self._episode_noise_mode == "paired" and self._current_noise_seed is None:
            raise RuntimeError(
                "paired episode noise is not initialized; call /reset with an episode_key first"
            )
        fallbacks_before = self._feedback_fallbacks
        try:
            self._begin_fresh_single_chunk_generation(conditions)
            feedback_plan, feedback = self._prepare_action_cache_feedback(conditions)
            prompt = conditions.get("prompt", "") or ""
            obs_latent = self._build_obs_chunk(conditions)
            context, seq_lens = self._encode_prompt(prompt)
            proprio = self._prep_proprio(conditions.get("proprio_state"))
            context_proprio = self._prep_window_context_proprio(
                conditions, fallback=proprio
            )

            local_chunk_index = self._step_c
            generation_index = self._episode_generation_index
            generation_noise_seed = self._prepare_generation_rng(generation_index)
            rerank_telemetry = None
            if self._generation_zero_rerank_enabled and generation_index == 0:
                rerank_telemetry = self._rerank_generation_zero(
                    obs_latent,
                    context,
                    seq_lens,
                    proprio,
                    context_proprio=context_proprio,
                    feedback_plan=feedback_plan,
                )
            else:
                self._step_with_obs_latent(
                    obs_latent,
                    context,
                    seq_lens,
                    proprio,
                    context_proprio=context_proprio,
                    feedback_plan=feedback_plan,
                )
            if feedback_plan is None:
                self._last_feedback = feedback
            self._pending_action_feedback["generated_at_policy_step"] = int(
                conditions.get("policy_step", 0) or 0
            )
            feedback = self._last_feedback.copy()

            actions_normalized = self._pending_action_feedback[
                "predicted_actions_normalized"
            ].copy()
            actions = self._pending_action_feedback["predicted_actions"].copy()
            self._episode_generation_index += 1
            telemetry = {
                "chunk_index": generation_index,
                "episode_generation_index": generation_index,
                "local_chunk_index": local_chunk_index,
                "action_frame_id": self._pending_action_feedback["frame_id"],
                "predicted_actions_normalized": actions_normalized,
                "predicted_actions": actions,
                "cache_feedback": feedback,
                "cache": self._cache.info(),
                "ar_step_after": self._step_c,
                "episode_generation_after": self._episode_generation_index,
                "generation_noise_schedule": self._generation_noise_schedule,
                "generation_noise_seed": generation_noise_seed,
            }
            if rerank_telemetry is not None:
                telemetry["generation_zero_rerank"] = rerank_telemetry
            return {
                "actions": actions,
                "video": None,
                "telemetry": telemetry,
            }
        except Exception:
            self._feedback_fallbacks = fallbacks_before
            raise

    def _feedback_cadence(self, conditions: dict) -> str:
        pending = self._pending_action_feedback
        if pending is None:
            return "episode_start"
        required = int(pending["action_tokens"])
        executed = int(conditions.get("executed_steps_since_generate", 0) or 0)
        if executed != required:
            return f"cadence_mismatch:{executed}!={required}"
        generated_at = pending.get("generated_at_policy_step")
        current_step = conditions.get("policy_step")
        if generated_at is not None and current_step is not None:
            elapsed = int(current_step) - int(generated_at)
            if elapsed != required:
                return f"policy_step_mismatch:{elapsed}!={required}"
        return "ok"

    def _measured_feedback_states(
        self, conditions: dict
    ) -> tuple[Optional[np.ndarray], str]:
        reason = self._feedback_cadence(conditions)
        if reason != "ok":
            return None, reason
        pending = self._pending_action_feedback
        assert pending is not None
        required = int(pending["action_tokens"])

        history = conditions.get("achieved_state_history")
        if history is None:
            history = [
                obs.get("state")
                for obs in (conditions.get("obs_history") or [])
                if isinstance(obs, dict) and obs.get("state") is not None
            ]
        states = [state for state in (history or []) if state is not None]
        if len(states) < required:
            return None, f"insufficient_history:{len(states)}<{required}"
        measured = np.asarray(states[-required:], dtype=np.float32)
        if measured.ndim != 2 or measured.shape != (required, self._action_dim):
            return (
                None,
                f"shape_mismatch:{measured.shape}!={(required, self._action_dim)}",
            )
        if not np.isfinite(measured).all():
            return None, "non_finite_history"
        return measured, "ok"

    def _feedback_failure(self, reason: str, **metadata) -> dict:
        if self._cache_feedback_fallback == "error":
            raise RuntimeError(f"action cache feedback unavailable: {reason}")
        self._feedback_fallbacks += 1
        return {"status": "fallback_predicted", "reason": reason, **metadata}

    @torch.no_grad()
    def _prepare_action_cache_feedback(
        self, conditions: dict
    ) -> tuple[Optional[dict], dict]:
        """Prepare a cache replacement without mutating cache or RNG."""
        if self._cache_feedback_mode == "predicted":
            return None, {"status": "predicted", "reason": "configured"}
        if self._pending_action_feedback is None:
            return None, {"status": "skipped", "reason": "episode_start"}

        reason = self._feedback_cadence(conditions)
        if reason != "ok":
            if reason.startswith(("cadence_mismatch:", "policy_step_mismatch:")):
                raise RuntimeError(f"invalid AR feedback cadence: {reason}")
            return None, self._feedback_failure(reason)

        if self._cache_feedback_mode == "reencode_predicted":
            pending = self._pending_action_feedback
            try:
                actions = np.asarray(pending["predicted_actions"], dtype=np.float32)
                actions_normalized = np.asarray(
                    pending["predicted_actions_normalized"], dtype=np.float32
                )
            except KeyError as exc:
                return None, self._feedback_failure(
                    f"predicted_chunk_missing:{exc.args[0]}"
                )
            expected = (int(pending["action_tokens"]), self._action_dim)
            if actions.shape != expected or actions_normalized.shape != expected:
                return None, self._feedback_failure(
                    "predicted_chunk_shape_mismatch",
                    actions_shape=list(actions.shape),
                    normalized_shape=list(actions_normalized.shape),
                )
            if (
                not np.isfinite(actions).all()
                or not np.isfinite(actions_normalized).all()
            ):
                return None, self._feedback_failure("non_finite_predicted_chunk")
            plan = dict(pending)
            plan.update(
                {
                    "actions": actions.copy(),
                    "actions_normalized": actions_normalized.copy(),
                    "feedback_source": "reencode_predicted",
                }
            )
            return plan, {
                "status": "prepared",
                "reason": "previous_predicted_chunk",
                "source_generation_index": int(plan["source_generation_index"]),
            }

        measured, reason = self._measured_feedback_states(conditions)
        if measured is None:
            return None, self._feedback_failure(reason)
        normalizer = getattr(self.architecture, "action_normalizer", None)
        measured_normalized = (
            np.asarray(normalizer.normalize(measured), dtype=np.float32)
            if normalizer is not None
            else measured.copy()
        )
        if not np.isfinite(measured_normalized).all():
            return None, self._feedback_failure("non_finite_normalized_history")
        plan = dict(self._pending_action_feedback)
        plan.update(
            {
                "actions": measured,
                "actions_normalized": measured_normalized,
                "feedback_source": "measured",
            }
        )
        return plan, {"status": "prepared", "reason": "achieved_state_history"}

    @torch.no_grad()
    def _commit_action_cache_feedback(self, plan: dict) -> dict:
        """Re-run a t=0 action chunk and replace its confirmed cache frame."""
        frame_id = int(plan["frame_id"])
        snapshot = self._cache.pop_frame(frame_id)
        if any(len(entries) != 1 or entries[0]["is_pred"] for entries in snapshot):
            self._cache.restore_frame(snapshot)
            return self._feedback_failure(
                "confirmed_action_frame_missing", frame_id=frame_id
            )

        ab = self.architecture.action_backbone
        actions = (
            torch.from_numpy(np.ascontiguousarray(plan["actions_normalized"]))
            .to(device=self._device, dtype=self._dtype)
            .unsqueeze(0)
        )
        frame_ids = torch.full(
            (actions.shape[1],), frame_id, dtype=torch.long, device=self._device
        )
        rope_positions = self.architecture._action_rope_positions(
            frame_ids, action_tokens_per_chunk=actions.shape[1]
        )
        a_proprio = plan["a_proprio"]
        extra = (
            {} if a_proprio is None else {"token_proprio_emb": a_proprio[:, None, :]}
        )
        try:
            state = ab.prepare_state(
                actions,
                torch.zeros(1, device=self._device, dtype=self._dtype),
                context=plan["context"],
                context_mask=plan["context_mask"],
                token_timesteps=torch.zeros(
                    1, actions.shape[1], device=self._device, dtype=self._dtype
                ),
                frame_ids=frame_ids,
                rope_positions=rope_positions,
                **extra,
            )
            self._driver.run_ar_chunk_through_backbone(
                ab, state, self._cache, frame_id, store_clean=True, is_pred=False
            )
        except Exception:
            self._cache.pop_frame(frame_id)
            self._cache.restore_frame(snapshot)
            raise

        self._feedback_commits += 1
        source = plan["feedback_source"]
        result = {
            "status": "measured" if source == "measured" else "reencoded_predicted",
            "reason": (
                "achieved_state_history"
                if source == "measured"
                else "previous_predicted_chunk"
            ),
            "frame_id": frame_id,
            "actions": plan["actions"],
            "actions_normalized": plan["actions_normalized"],
        }
        if source == "reencode_predicted":
            result["source_generation_index"] = int(plan["source_generation_index"])
        return result

    # Compatibility wrappers for callers/tests that used the original private
    # measured-only seam before cache feedback gained a second source.
    def _prepare_measured_action_feedback(
        self, conditions: dict
    ) -> tuple[Optional[dict], dict]:
        return self._prepare_action_cache_feedback(conditions)

    def _commit_measured_action_feedback(self, plan: dict) -> dict:
        return self._commit_action_cache_feedback(plan)

    # --------------------------------------------------------------- conditions
    def _encode_prompt(self, prompt: str):
        if prompt in self._prompt_ctx_cache:
            self._prompt_ctx_cache.move_to_end(prompt)
            return self._prompt_ctx_cache[prompt]
        vb = self.architecture.video_backbone
        context = vb._encode_text(prompt).to(device=self._device, dtype=self._dtype)
        seq_lens = torch.full(
            (context.shape[0],), context.shape[1], dtype=torch.long, device=self._device
        )
        self._prompt_ctx_cache[prompt] = (context, seq_lens)
        while len(self._prompt_ctx_cache) > _PROMPT_CTX_CACHE_MAXSIZE:
            self._prompt_ctx_cache.popitem(last=False)
        return context, seq_lens

    def _prep_proprio(self, proprio_state):
        """Normalize raw deploy proprio and return a ``(B, D)`` tensor (or None)."""
        if not self.architecture.uses_proprioception:
            return None
        if proprio_state is None:
            return None
        if self._proprio_mode == "single":
            if self._latched_proprio is None:
                self._latched_proprio = np.asarray(
                    proprio_state, dtype=np.float32
                ).reshape(-1)
            proprio_state = self._latched_proprio
        norm = self.architecture.normalize_deploy_proprio(
            np.asarray(proprio_state, dtype=np.float32).reshape(-1)
        )
        return (
            torch.from_numpy(np.asarray(norm, dtype=np.float32))
            .to(device=self._device, dtype=self._dtype)
            .unsqueeze(0)
        )

    def _prep_window_context_proprio(self, conditions: dict, *, fallback):
        """Prepare the clip-level proprio token for the configured causal graph.

        Causal single-chunk training uses the action anchor's current state for
        both the global token and its sole per-chunk AdaLN branch.  The legacy
        multi-chunk path retains its rolling-window source.
        """

        if not self.architecture.uses_proprioception or self._proprio_mode == "single":
            return fallback
        if getattr(self, "_reset_cache_each_generation", False):
            return fallback
        observations = self._recent_observations(conditions)
        if not observations:
            return fallback
        if observations[0].get("state") is None:
            raise ValueError("oldest rolling-window observation is missing proprio state")
        current_raw = conditions.get("proprio_state")
        latest_raw = observations[-1].get("state")
        if current_raw is None or latest_raw is None:
            raise ValueError("current rolling-window observation is missing proprio state")
        if not np.array_equal(
            np.asarray(latest_raw, dtype=np.float32).reshape(-1),
            np.asarray(current_raw, dtype=np.float32).reshape(-1),
        ):
            raise ValueError("current proprio_state differs from obs_history tail")
        window_start_state = observations[0]["state"]
        norm = self.architecture.normalize_deploy_proprio(
            np.asarray(window_start_state, dtype=np.float32).reshape(-1)
        )
        return (
            torch.from_numpy(np.asarray(norm, dtype=np.float32))
            .to(device=self._device, dtype=self._dtype)
            .unsqueeze(0)
        )

    def _recent_observations(self, conditions: dict) -> list[dict]:
        """Return the exact oldest-to-newest raw window used for video encoding."""

        observations = [
            obs
            for obs in (conditions.get("obs_history") or [])
            if isinstance(obs, dict) and obs.get("image") is not None
        ]
        return observations[-self._raw_num_frames :]

    def _recent_frames(self, conditions: dict) -> list:
        """Per-sim-step head frames, oldest→newest. Prefers ``obs_history`` (WAMPolicy
        appends one every step) over the single ``first_frame_image`` so the obs clip
        can be rebuilt at the training pixel cadence."""
        frames = [obs["image"] for obs in self._recent_observations(conditions)]
        if not frames:
            ff = conditions.get("first_frame_image")
            if ff:
                frames = [ff[0]]
        if not frames:
            raise ValueError(
                "ARInferenceEngine.generate requires conditions['obs_history'] or ['first_frame_image']."
            )
        return frames

    def _build_obs_clip(self, conditions: dict) -> list:
        """Reconstruct the training-cadence video clip: take the last ``num_frames``
        per-step frames and subsample by ``video_stride`` — mirroring the dataset's
        ``range(0, num_frames, video_stride)`` so the encoded latents match training's
        motion distribution. Left-pads at episode start (bootstrap; F3/R4 caveat)."""
        frames = self._recent_frames(conditions)
        if self._obs_chunk_mode == "repeat_frame":  # debug only: zero-motion clip
            return [frames[-1]] * self._video_num_frames
        recent = frames[-self._raw_num_frames :]
        if len(recent) < self._raw_num_frames:  # episode-start pad to a full window
            recent = [recent[0]] * (self._raw_num_frames - len(recent)) + recent
        clip = recent[
            :: self._video_stride
        ]  # newest frame included (num_frames-1 % stride == 0)
        if len(clip) > self._video_num_frames:
            clip = clip[-self._video_num_frames :]
        elif len(clip) < self._video_num_frames:
            clip = [clip[0]] * (self._video_num_frames - len(clip)) + clip
        return clip

    def _build_obs_chunk(self, conditions: dict) -> torch.Tensor:
        """Encode the recent observation into a ``(B, C, fcs, H_lat, W_lat)`` latent chunk."""
        vb = self.architecture.video_backbone
        if vb._pipe.vae is None:
            raise RuntimeError(
                "ARInferenceEngine obs encoding requires a loaded Wan VAE."
            )

        from sana_wam.model.video_backbone.sana.adapter import _pil_video_to_tensor

        clip = self._build_obs_clip(conditions)
        video = _pil_video_to_tensor([clip]).to(
            device=self._device, dtype=self._dtype
        )  # (1,3,T,H,W)
        latents = vb._pipe.vae.encode([video[0]], device=self._device, tiled=True)
        latents = latents.to(
            device=self._device, dtype=self._dtype
        )  # (1,C,T_lat,H_lat,W_lat)
        if latents.shape[2] < self._fcs:
            raise RuntimeError(
                f"VAE produced {latents.shape[2]} latent frames < ar_frame_chunk_size={self._fcs}; "
                f"increase inference.video_num_frames (got {self._video_num_frames})."
            )
        return self._select_obs_band(latents)

    def _select_obs_band(self, latents: torch.Tensor) -> torch.Tensor:
        """Pick the obs latent band. ``auto`` uses the leading band (training chunk 0,
        incl. the causal first-frame latent) at episode start (step 0) and the trailing
        band (latest chunk) thereafter — matching how training assigns rope[0,1] to the
        first chunk vs the continuation chunk. F3 bootstrap eval knob."""
        band = self._obs_latent_band
        if band == "auto":
            band = "leading" if self._step_c == 0 else "trailing"
        if band == "leading":  # training chunk 0 band (incl. causal first-frame)
            return latents[:, :, : self._fcs].contiguous()
        return latents[
            :, :, -self._fcs :
        ].contiguous()  # trailing = latest chunk of a training clip

    # -------------------------------------------------------------------- reset
    def reset(self, episode_context: Optional[dict] = None) -> None:
        """Clear rollout state and select this episode's model-noise stream."""
        context = dict(episode_context or {})
        self._current_noise_seed = self._resolve_episode_noise_seed(context)
        key = context.get("episode_key")
        self._episode_key = None if key is None else str(key)
        pair_key = context.get("noise_pair_key") or key
        self._noise_pair_key = None if pair_key is None else str(pair_key)
        self._episode_context = context
        self._cache.reset()
        self._step_c = 0
        self._episode_generation_index = 0
        self._gen = self._make_generator()
        self._latched_proprio = None
        self._pending_action_feedback = None
        self._last_feedback = {"status": "none", "reason": "episode_start"}
        self._feedback_commits = 0
        self._feedback_fallbacks = 0
        self._generation_zero_rerank_count = 0
        self._last_generation_zero_rerank = None
        self._last_generation_noise_seed = None
        logger.info(
            "AR episode reset: key=%r pair_key=%r noise_mode=%s model_noise_seed=%r cache_feedback=%s",
            self._episode_key,
            self._noise_pair_key,
            self._episode_noise_mode,
            self._current_noise_seed,
            self._cache_feedback_mode,
        )

    @property
    def runtime_info(self) -> dict:
        """Tensor-free runtime state exposed by the policy server."""
        feedback = {
            key: value
            for key, value in self._last_feedback.items()
            if key not in {"actions", "actions_normalized"}
        }
        ranker_identity = (
            None if self._outcome_ranker is None else self._outcome_ranker.identity
        )
        return {
            "video_euler_master_dtype": self._video_euler_master_dtype,
            "video_model_input_dtype": str(self._dtype),
            "episode_noise_mode": self._episode_noise_mode,
            "episode_noise_base_seed": self._episode_noise_base_seed,
            "configured_seed": self._seed,
            "current_episode_key": self._episode_key,
            "current_noise_pair_key": self._noise_pair_key,
            "current_model_noise_seed": self._current_noise_seed,
            "generation_noise_schedule": self._generation_noise_schedule,
            "common_future_noise_base_seed": self._common_future_noise_base_seed,
            "last_generation_noise_seed": self._last_generation_noise_seed,
            "cache_feedback_mode": self._cache_feedback_mode,
            "cache_feedback_fallback": self._cache_feedback_fallback,
            "cache_feedback_commits": self._feedback_commits,
            "cache_feedback_fallbacks": self._feedback_fallbacks,
            "action_steps": self._action_steps,
            "action_scheduler_shift": self._action_scheduler_shift,
            "action_scheduler_sigmas": list(self._a_sigmas),
            "action_scheduler_sigmas_sha256": self._action_scheduler_sigmas_sha256,
            "temporal_alignment": "observed_video_2c_to_action_2c_plus_1",
            "proprio_context_mode": (
                "current"
                if self._reset_cache_each_generation
                else "rolling_window_oldest"
            ),
            "reset_cache_each_generation": self._reset_cache_each_generation,
            "video_chunks_per_window": self._video_chunks_per_window,
            "ar_step": self._step_c,
            "episode_generation_index": self._episode_generation_index,
            "generation_zero_rerank": {
                "enabled": self._generation_zero_rerank_enabled,
                "candidate_count": self._generation_zero_rerank_candidate_count,
                "invocations": self._generation_zero_rerank_count,
                "ranker": ranker_identity,
                "last_selection": self._last_generation_zero_rerank,
            },
            "last_cache_feedback": feedback,
            "cache": self._cache.info(),
        }
