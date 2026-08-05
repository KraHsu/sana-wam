"""``SanaVideoBackbone`` — sana-wam VideoBackbone wrapper around SANA-Video.

Owns a :class:`SanaPipe` (DiT + VAE + text encoder) and exposes the standard
``prepare → run_block × N → finalize`` lifecycle plus the
``pre_attn_at_layer`` / ``post_attn_at_layer`` split that ``SanaMoTJointDriver``
consumes.

- ``prepare``, ``run_block``, ``finalize``: bit-equivalent to upstream
  ``SanaMSVideo.forward`` for per-sample timesteps.
- ``pre_attn_at_layer``: returns the rotated and unrotated kernel-applied
  Q/K so the driver can implement cumsum-linear-attn over a heterogeneous
  sequence. The ``attn_kernel`` property advertises ``linear_relu`` for
  driver dispatch.
- VideoBackbone methods not used by this model raise ``NotImplementedError``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sana_wam.cach.prefix_compaction import FixedKPrefixPlan
from sana_wam.model.video_backbone.adapter import BlockLoopState, VideoBackbone
from sana_wam.model.video_backbone.sana.blocks_split import SanaMSVideoSplit
from sana_wam.model.video_backbone.sana.pipeline_builder import (
    SanaPipe,
    build_mini_sana_pipeline,
    build_sana_pipeline,
)


def _pil_video_to_tensor(frames: Any) -> Tensor:
    """``list[list[PIL.Image]]`` → ``(B, 3, T, H, W)`` float in ``[-1, 1]``.

    Standard convention (uint8 RGB → ``/127.5 - 1``).
    """
    import numpy as np

    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError(
            f"Expected `frames` as a non-empty list of clips (each a list of PIL frames); "
            f"got {type(frames).__name__}."
        )
    arrs = []
    for clip in frames:
        if not isinstance(clip, (list, tuple)) or not clip:
            raise ValueError("Each per-sample entry in `frames` must be a non-empty list of PIL.Image frames.")
        clip_arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in clip], axis=0)
        arrs.append(clip_arr)
    stack = np.stack(arrs, axis=0)  # (B, T, H, W, 3) uint8
    t = torch.from_numpy(stack).to(dtype=torch.float32)
    t = t / 127.5 - 1.0
    return t.permute(0, 4, 1, 2, 3).contiguous()  # (B, 3, T, H, W)

logger = logging.getLogger(__name__)


class SanaVideoBackbone(VideoBackbone):
    """SANA-Video backbone for sana-wam.

    Owns a :class:`SanaPipe`; never exposes it. External code accesses
    capabilities through the VideoBackbone ABC.
    """

    supports_generic_mot_compile = False
    generic_mot_compile_skip_reason = "SANA uses linear-ReLU MoT attention and needs a dedicated compile helper"

    @classmethod
    def get_native_dit_patch_size(cls, pipe) -> Tuple[int, int, int]:
        """SANA-Video native DiT first-layer patch size.

        ``SanaMSVideo`` constructs ``x_embedder`` with ``patch_size=(1, 2, 2)``
        (see ``pipeline_builder._SANA_VIDEO_2B_480P_PRESET`` and
        ``build_mini_sana_pipeline``). Hard-coded — same rationale as Wan:
        invariant across the SANA-Video family and probing ``pipe.dit`` would
        couple this classmethod to a non-trivial pipeline state.
        """
        return (1, 2, 2)

    @classmethod
    def get_native_temporal_contract(cls, pipe) -> Tuple[int, bool]:
        """Native VAE temporal contract ``(temporal_compression, causal)``.

        VAE-aware: the LTX2 shim (``vae_type="ltx2"``, the SANA-WM world-model
        VAE) carries ``temporal_compression`` / ``causal`` attributes → ``(8,
        True)``. Everything else is the Wan2.1 family that SANA-Video 2B 480p
        reuses: ``(4, True)`` — causal first-frame token plus 4-frame tail
        grouping.
        """
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "temporal_compression"):
            return (int(vae.temporal_compression), bool(getattr(vae, "causal", True)))
        return (4, True)

    @classmethod
    def get_native_spatial_compression(cls, pipe) -> int:
        """Native VAE spatial downsample factor (pixels per latent cell, one side).

        Wan2.1 = 8; LTX2 = 32. Read from the VAE so latent↔pixel geometry stays
        consistent across VAE families. The shim exposes ``spatial_compression``;
        the Wan VAE exposes ``upsampling_factor`` (=8); default to 8.
        """
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            return int(getattr(vae, "spatial_compression", getattr(vae, "upsampling_factor", 8)))
        return 8

    def __init__(self, pipe: SanaPipe):
        super().__init__()
        self._pipe = pipe
        self._split = SanaMSVideoSplit(pipe.dit)

        # Opt-in clean-prefix (current-frame) conditioning. Default False keeps
        # SANA pure text-to-video (byte-identical to existing checkpoints). When
        # True, ``preprocess_input`` emits first_frame_latents +
        # num_clean_prefix_frames=1, and ``prepare`` switches the split-forward
        # to per-frame timestep modulation (frame-0 at t=0).
        self._use_first_frame_cond = bool(pipe.config.get("use_first_frame_cond", False))
        self._continuous_timestep_conditioning = bool(
            pipe.config.get("continuous_timestep_conditioning", False)
        )

        # Self-attention family: "linear_relu" (SANA LiteLAReLURope, MoT joint
        # path) or "gdn" (Sana-wm ChunkCausalGDNTriton, native run_block +
        # cross-attn bridge). Drives driver/architecture dispatch via the
        # ``attn_kernel`` property below.
        self._attn_kernel = str(pipe.config.get("attn_kernel", "linear_relu"))
        self._chunk_size = int(pipe.config.get("chunk_size", 3))
        self._aligned_positive_rope_kernel = False
        self._aligned_positive_rope_mode: str | None = None
        self._ar_temporal_chunk_frames: int | None = None

        # SanaPipe is @dataclass (not nn.Module), so `self._pipe = pipe`
        # doesn't auto-register its DiT/VAE/text_encoder as nn.Module
        # children — they stay invisible to self.parameters() /
        # named_children(), and the optimizer-groups builder finds 0
        # trainable params on the video side. Register them explicitly so
        # the standard parameter-discovery path works (the DiT becomes
        # trainable; the VAE / text encoder are already
        # requires_grad_(False) at construction time and stay frozen).
        # `_pipe.<name>` remains the canonical access — add_module just
        # files the same object reference under self._modules[name].
        self.add_module("dit", pipe.dit)
        if pipe.vae is not None:
            self.add_module("vae", pipe.vae)
        if pipe.text_encoder is not None:
            self.add_module("text_encoder", pipe.text_encoder)

        # Resolve DiT patch geometry + VAE temporal contract into backbone-owned
        # instance attributes so the ABC properties (dit_patch_size /
        # temporal_compression / causal_temporal) return without any branching.
        # SanaVideoBackbone doesn't accept an external encoder, so the native
        # values are the only values this backbone ever exposes.
        self._dit_patch_size = self.get_native_dit_patch_size(pipe)
        self._temporal_compression, self._causal_temporal = self.get_native_temporal_contract(pipe)
        self._spatial_compression = self.get_native_spatial_compression(pipe)

        first_param = next(pipe.dit.parameters(), None)
        self._dtype = first_param.dtype if first_param is not None else torch.bfloat16
        self._device = first_param.device if first_param is not None else torch.device("cuda")

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, source: Any, **kw: Any) -> "SanaVideoBackbone":
        """Build a SanaVideoBackbone from a config / dict / model directory.

        Delegates to :func:`build_sana_pipeline`; see that function's docstring
        for accepted shapes. For unit tests, use :meth:`from_mini_config`.
        """
        pipe = build_sana_pipeline(source, **kw)
        return cls(pipe)

    @classmethod
    def from_mini_config(cls, **mini_kwargs: Any) -> "SanaVideoBackbone":
        """Random-weight tiny backbone for unit tests. See
        :func:`build_mini_sana_pipeline` for available kwargs."""
        pipe = build_mini_sana_pipeline(**mini_kwargs)
        return cls(pipe)

    # ----------------------------------------------------------------
    # ABC: properties
    # ----------------------------------------------------------------

    @property
    def _dit(self) -> nn.Module:
        return self._pipe.dit

    @property
    def dim(self) -> int:
        return int(self._dit.hidden_size)

    @property
    def num_layers(self) -> int:
        return len(self._dit.blocks)

    @property
    def num_heads(self) -> int:
        """Number of SELF-attention heads.

        SANA's ``LiteLAReLURope`` exposes ``heads`` + ``dim`` (sana_blocks.py:326),
        while ``FlashAttention`` uses ``num_heads`` + ``head_dim``. Both are
        possible attn modules depending on the factory preset, so probe in order.
        Falls back to ``hidden_size // head_dim`` if neither attr is present.
        """
        attn = self._dit.blocks[0].attn
        for name in ("heads", "num_heads"):
            v = getattr(attn, name, None)
            if v is not None:
                return int(v)
        return int(self._dit.hidden_size // self.head_dim)

    @property
    def context_dim(self) -> int:
        """SANA-Video text context dim (Gemma-2-2B last_hidden_state = 2304).

        Read by :meth:`BaseWAMArchitecture._resolve_text_dim` to size the
        proprio encoder and any action-side context projections. SANA's DiT
        doesn't store ``caption_channels`` directly; we read it off the
        ``y_embedder.y_embedding`` buffer's last dim (the 480p preset pins
        this to 2304 via ``_SANA_VIDEO_2B_480P_PRESET``; the mini-config
        factory uses a tiny value for unit tests).
        """
        return int(self._dit.y_embedder.y_embedding.shape[-1])

    @property
    def ar_temporal_chunk_frames(self) -> int | None:
        """Physical AR chunk size used by non-cached temporal operators."""
        return self._ar_temporal_chunk_frames

    def configure_ar_chunkwise_temporal_ops(self, chunk_frames: int) -> None:
        """Pin window-flash and temporal-MLP boundaries to one AR chunk.

        The global linear branch owns cross-chunk state. SANA's parallel
        window attention and symmetric temporal convolution have no cache, so
        applying them to a full training clip would expose context unavailable
        to chunk-at-a-time inference.
        """
        frames = int(chunk_frames)
        if frames <= 0:
            raise ValueError("AR temporal chunk size must be positive")
        self._ar_temporal_chunk_frames = frames

    @property
    def head_dim(self) -> int:
        attn = self._dit.blocks[0].attn
        for name in ("dim", "head_dim"):
            v = getattr(attn, name, None)
            if v is not None:
                return int(v)
        raise AttributeError(
            f"SanaVideoBackbone.head_dim: attn module {type(attn).__name__!r} "
            "exposes neither 'dim' nor 'head_dim'."
        )

    @property
    def scheduler(self):
        return self._pipe.scheduler

    @property
    def submodule_names(self) -> list[str]:
        names = ["dit"]
        for n in ("vae", "text_encoder"):
            if getattr(self._pipe, n, None) is not None:
                names.append(n)
        return names

    @property
    def video_attention_mask_mode(self) -> str:
        """SANA-Video uses ``first_frame_causal`` semantics in sana-wam joint
        MoT (matching the FastWAM default); ``SanaMoTJointDriver`` relies on
        this."""
        return "first_frame_causal"

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        """Build the video↔video block of the joint MoT attention mask.

        Direct port of :meth:`WanVideoBackbone.build_video_to_video_mask`
        (``wan_adapter.py:203-248``); the math only depends on
        ``video_tokens_per_frame``, so the Wan↔SANA layout difference is
        transparent here.
        """
        if video_seq_len <= 0:
            raise ValueError(f"video_seq_len must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"video_tokens_per_frame must be positive, got {video_tokens_per_frame}")

        mode = self.video_attention_mask_mode
        if mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "video_seq_len must be divisible by video_tokens_per_frame in 'per_frame_causal' mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device))
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(
            f"Unsupported video_attention_mask_mode '{mode}'. "
            "Choose from: bidirectional, per_frame_causal, first_frame_causal."
        )

    @property
    def attn_kernel(self) -> str:
        """Driver/architecture dispatch hook.

        - ``"linear_relu"`` (SANA LiteLAReLURope): the architecture builds
          ``SanaMoTJointDriver`` and the action stream joins via concatenated
          MoT self-attention (factorized linear attn).
        - ``"gdn"`` (Sana-wm ChunkCausalGDNTriton): GDN is a frame-wise recurrence
          that cannot do concatenated cross-modal attention, so the architecture
          runs the video backbone natively (``run_block``) and the action stream
          attaches via the cross-attention bridge (``joint_cross_attn``).

        Not abstract on the base ABC — a softmax backbone need not define it;
        ``getattr(..., "softmax")`` at the dispatch site keeps the SDPA path.
        """
        return self._attn_kernel

    @property
    def continuous_timestep_conditioning(self) -> bool:
        """Whether video timesteps use the T1 FP32 continuous contract."""
        return self._continuous_timestep_conditioning

    @property
    def aligned_positive_rope_kernel(self) -> bool:
        """Whether split attention uses one positive kernel after RoPE."""
        return self._aligned_positive_rope_kernel

    @property
    def aligned_positive_rope_mode(self) -> str | None:
        return self._aligned_positive_rope_mode

    def configure_aligned_positive_rope_kernel(
        self, *, beta: float, delta: float, mode: str = "post_rope"
    ) -> int:
        """Enable the aligned kernel on every SANA linear-attention block.

        The aligned path deliberately requires the checkpoint-compatible
        strictly-positive learnable map. This fails closed rather than silently
        mixing the new positional order with an unrelated feature map.
        """
        if self._attn_kernel != "linear_relu":
            raise ValueError(
                "aligned_positive_rope_kernel requires attn_kernel='linear_relu'"
            )
        if mode not in {"post_rope", "unrotated", "absolute_rope"}:
            raise ValueError(
                "aligned positive kernel mode must be 'post_rope', "
                f"'unrotated', or 'absolute_rope', got {mode!r}"
            )
        from sana_wam.model.video_backbone.sana.strictly_positive_feature_map import (
            StrictlyPositiveLearnableFeatureMap,
        )

        configured = 0
        for block in self._dit.blocks:
            attention = getattr(block, "attn", None)
            feature_map = getattr(attention, "kernel_func", None)
            if not isinstance(feature_map, StrictlyPositiveLearnableFeatureMap):
                raise TypeError(
                    "aligned_positive_rope_kernel requires every video block to "
                    "use StrictlyPositiveLearnableFeatureMap"
                )
            if feature_map.beta != float(beta) or feature_map.delta != float(delta):
                raise ValueError(
                    "aligned_positive_rope_kernel beta/delta must match the "
                    "video strictly_positive_feature_map configuration"
                )
            attention.aligned_positive_rope_kernel = mode
            configured += 1
        if configured != self.num_layers:
            raise RuntimeError(
                "aligned_positive_rope_kernel did not configure every video layer"
            )
        self._aligned_positive_rope_kernel = True
        self._aligned_positive_rope_mode = mode
        return configured

    # ----------------------------------------------------------------
    # ABC: three-step lifecycle
    # ----------------------------------------------------------------

    def prepare(self, **pipeline_inputs: Any) -> BlockLoopState:
        """Patchify + RoPE + t_mod + y_embed. Mirrors upstream forward prelude.

        Accepts two equivalent input shapes:

        - SANA-native:
            ``x: (B, C, T, H, W)``, ``timestep: (B,)``,
            ``y: (B, 1, L, D)``, ``mask: (B, 1, 1, L)`` int/bool

        - sana-wam architecture flow (``BaseWAMArchitecture.compute_loss``):
            ``latents: (B, C, T, H, W)``, ``timestep: (B,)``,
            ``context: (B, L, D)``, optionally ``context_mask: (B, L) bool``
            or ``seq_lens: (B,) long``.

        When called from the architecture, this method translates the standard
        keys → SANA shapes inline so the rest of ``SanaMSVideoSplit.prepare`` is
        unchanged. With ``use_first_frame_cond`` OFF (the default) SANA's time
        embedding is per-sample and ``force_per_token_t_mod`` /
        ``zero_clean_prefix_t_mod`` are dropped. With it ON and a clean prefix
        flagged (``num_clean_prefix_frames>0``), the split-forward switches to
        per-frame timestep modulation with the prefix frame(s) at t=0.
        """
        # --- 1. latent ---
        if "x" in pipeline_inputs:
            x = pipeline_inputs.pop("x")
        elif "latents" in pipeline_inputs:
            x = pipeline_inputs.pop("latents")
        else:
            raise KeyError(
                "SanaVideoBackbone.prepare requires either 'x' or 'latents' in pipeline_inputs."
            )

        timestep = pipeline_inputs.pop("timestep")
        frame_is_pad = pipeline_inputs.pop("frame_is_pad", None)
        if frame_is_pad is not None:
            if not isinstance(frame_is_pad, Tensor):
                raise TypeError("frame_is_pad must be a tensor or None")
            if frame_is_pad.dtype != torch.bool or frame_is_pad.layout != torch.strided:
                raise TypeError("frame_is_pad must be a strided boolean tensor")
            expected_frame_mask_shape = (x.shape[0], x.shape[2])
            if tuple(frame_is_pad.shape) != expected_frame_mask_shape:
                raise ValueError(
                    "frame_is_pad must exactly match the video batch/frame shape "
                    f"{expected_frame_mask_shape}, got {tuple(frame_is_pad.shape)}"
                )
            if frame_is_pad.device != x.device:
                raise ValueError(
                    f"frame_is_pad must be on {x.device}, got {frame_is_pad.device}"
                )
            if self._ar_temporal_chunk_frames is None:
                raise ValueError(
                    "frame_is_pad requires configured AR chunkwise temporal operators"
                )
            frame_is_pad = frame_is_pad.detach()

        # --- 2. caption embeddings → (B, 1, L, D) ---
        if "y" in pipeline_inputs:
            y = pipeline_inputs.pop("y")
        elif "context" in pipeline_inputs:
            context = pipeline_inputs.pop("context")
            if context.dim() == 3:
                # (B, L, D) → (B, 1, L, D) — SANA's MultiHeadCrossAttention expects
                # a leading singleton dimension for the per-sample caption track.
                y = context.unsqueeze(1)
            else:
                y = context
        else:
            raise KeyError(
                "SanaVideoBackbone.prepare requires 'y' or 'context' in pipeline_inputs."
            )

        # --- 3. caption mask → (B, 1, 1, L) int16 ---
        # SANA's DISABLE_XFORMERS=1 path requires a non-None caption mask. The
        # architecture passes ``seq_lens`` (B,) or ``context_mask`` (B, L) bool;
        # both are converted to the (B, 1, 1, L) int16 layout the split expects.
        if "mask" in pipeline_inputs:
            mask = pipeline_inputs.pop("mask")
        else:
            context_mask = pipeline_inputs.pop("context_mask", None)
            seq_lens = pipeline_inputs.pop("seq_lens", None)
            B, L = y.shape[0], y.shape[2]
            if context_mask is not None:
                # (B, L) bool → (B, 1, 1, L)
                mask = context_mask.to(dtype=torch.int16, device=y.device).view(B, 1, 1, L)
            elif seq_lens is not None:
                positions = torch.arange(L, device=y.device).view(1, 1, 1, L)
                lens = seq_lens.to(device=y.device).view(B, 1, 1, 1)
                mask = (positions < lens).to(torch.int16)
            else:
                # All-attend mask: every text token is valid.
                mask = torch.ones(B, 1, 1, L, dtype=torch.int16, device=y.device)

        # Clean-prefix conditioning: when enabled and a prefix is flagged, the
        # split-forward modulates the first ``num_clean`` latent frames at t=0
        # (per-frame t-mod) instead of the per-sample broadcast. Read it before
        # the drop-loop below pops the key.
        num_clean = int(pipeline_inputs.get("num_clean_prefix_frames", 0) or 0)
        per_frame_t_mod = self._use_first_frame_cond and num_clean > 0

        # SANA's split.prepare doesn't consume the rest of the sana-wam keys
        # (``input_latents``, ``num_clean_prefix_frames``, ``height``, ``width``,
        # ``num_frames``, ``first_frame_latents``, ``actions``, ``proprio``,
        # padding masks, etc.). Drop them so we don't collide with the upstream
        # ``**kwargs`` slot.
        for k in (
            "input_latents",
            "num_clean_prefix_frames",
            "height",
            "width",
            "num_frames",
            "first_frame_latents",
            "fuse_vae_embedding_in_latents",
            "clip_feature",
            "force_per_token_t_mod",
            "zero_clean_prefix_t_mod",
            "actions",
            "proprio",
            "action_mask",
            "video_mask",
            "use_gradient_checkpointing",
            "use_gradient_checkpointing_offload",
        ):
            pipeline_inputs.pop(k, None)

        prep = self._split.prepare(
            x, timestep, y, mask,
            per_frame_t_mod=per_frame_t_mod,
            num_clean_prefix_frames=num_clean,
            continuous_timestep_conditioning=self._continuous_timestep_conditioning,
            **pipeline_inputs,
        )

        # Stash everything the per-block loop needs in ``extras`` to keep the
        # standard fields semantically pure (e.g. ``t_mod`` is the AdaLN
        # 6-scale modulation, not the raw time embedding).
        return BlockLoopState(
            x=prep["x"],
            t_mod=prep["t0"],
            freqs=prep["image_pos_embed"],
            context=prep["y"],
            context_mask=prep["y_lens"],
            f=prep["f"],
            h=prep["h"],
            w=prep["w"],
            t=prep["t"],
            extras={
                "split": self._split,
                "bs": prep["bs"],
                "block_kwargs": prep["kwargs"],
                "ar_temporal_chunk_frames": self._ar_temporal_chunk_frames,
                "frame_is_pad": frame_is_pad,
            },
        )

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        """One upstream-block step — used by the video-only path.

        Joint-MoT drivers call :meth:`pre_attn_at_layer` and
        :meth:`post_attn_at_layer` separately and skip this method.
        """
        split: SanaMSVideoSplit = state.extras["split"]
        temporal_chunk_frames = state.extras.get("ar_temporal_chunk_frames")
        if temporal_chunk_frames is not None:
            *_, post_state = split.block_pre_attn(
                block_id, state.x, state.t_mod, state.freqs
            )
            attn_out = split.native_attn(post_state)
            state.x = split.block_post_attn(
                block_id,
                attn_out,
                post_state,
                y=state.context,
                y_lens=state.context_mask,
                f=state.f,
                h=state.h,
                w=state.w,
                temporal_chunk_frames=temporal_chunk_frames,
                frame_is_pad=state.extras.get("frame_is_pad"),
            )
            return state
        out = split.run_block(
            block_id,
            {
                "x": state.x,
                "y": state.context,
                "t0": state.t_mod,
                "y_lens": state.context_mask,
                "f": state.f,
                "h": state.h,
                "w": state.w,
                "image_pos_embed": state.freqs,
                "kwargs": state.extras.get("block_kwargs", {}),
            },
        )
        state.x = out["x"]
        return state

    def finalize(self, state: BlockLoopState) -> Tensor:
        split: SanaMSVideoSplit = state.extras["split"]
        return split.finalize(
            {
                "x": state.x,
                "t": state.t,
            }
        )

    # ----------------------------------------------------------------
    # Cached autoregressive streaming (true-AR, arbitrary length)
    # ----------------------------------------------------------------
    #
    # The whole-clip ``prepare/run_block/finalize`` path above runs a GDN
    # backbone in one bounded shot. For arbitrary-length autoregression we
    # instead drive the model chunk-by-chunk with a rolling GDN **state cache**,
    # matching upstream Sana-wm's ``forward_long`` inference paradigm. The swap
    # below was proven viable in this repo by the Phase-0 spike: three in-place
    # ``__class__`` rebinds (weight-compatible) + a windowed RoPE + binding
    # upstream's streaming ``forward_long`` — no large upstream fork, no weight
    # surgery (a GDN checkpoint trained on the non-cached path serves unchanged).

    def enable_cached_streaming(self, *, max_seq_len: int = 1024) -> None:
        """Switch the GDN DiT in place to the cached chunk-streaming variant.

        Swaps each block's attention ``ChunkCausalGDNTriton →
        CachedChunkCausalGDN`` and FFN ``GLUMBConvTemp → CachedGLUMBConvTemp``
        (weight-compatible ``__class__`` rebind), swaps the positional embedding
        to ``CausalWanRotaryPosEmbed`` (absolute/windowed frame indices), sets
        ``pos_embed_type="casual_wan_rope"``, and binds
        ``SanaMSVideoCamCtrlStreaming.forward_long`` onto the instance.

        Idempotent. Requires ``attn_kernel == "gdn"`` — ``linear_relu`` has its
        own ``ARLinearStateCache`` streaming path and no cached GDN sibling.
        After this swap the whole-clip ``prepare/run_block/finalize`` path is no
        longer valid; drive the model via :meth:`run_chunk` instead.
        """
        if getattr(self, "_cached_streaming_enabled", False):
            return
        if self._attn_kernel != "gdn":
            raise RuntimeError(
                f"enable_cached_streaming requires attn_kernel='gdn', got "
                f"{self._attn_kernel!r}. linear_relu uses ARLinearStateCache instead."
            )
        import types as _types

        from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
        from diffusion.model.nets.sana_blocks import CausalWanRotaryPosEmbed
        from diffusion.model.nets.sana_gdn_blocks import CachedChunkCausalGDN
        from diffusion.model.nets.sana_multi_scale_video_camctrl import (
            SanaMSVideoCamCtrlStreaming,
        )

        dit = self._dit
        # Two build flavors reach this point:
        #  - from-scratch GDN-AR: the NON-streaming factory builds plain
        #    ``ChunkCausalGDNTriton`` attention on every (all-GDN, no-camctrl)
        #    block → rebind each to the cached sibling here.
        #  - pretrained SANA-WM (init_dit_from): the STREAMING factory already
        #    builds the correct cached, HYBRID per-block attention
        #    (``Cached*GDN*`` on GDN blocks, ``CachedSoftmax*`` on softmax blocks).
        #    Blindly rebinding those to ``CachedChunkCausalGDN`` would CLOBBER the
        #    softmax blocks into GDN and corrupt the rollout — so skip the attn
        #    rebind when the blocks are already cached, and only fix the FFN.
        attn_already_cached = type(dit.blocks[0].attn).__name__.startswith("Cached")
        for blk in dit.blocks:
            if not attn_already_cached:
                blk.attn.__class__ = CachedChunkCausalGDN  # same params, cached forward
            # The streaming factory leaves the FFN as plain GLUMBConvTemp (single-
            # tensor return); forward_long needs the cached variant (returns the
            # temporal-conv left-context cache). Always rebind.
            blk.mlp.__class__ = CachedGLUMBConvTemp  # temporal-conv left-context cache
        # The base ``self.rope`` is a ``WanRotaryPosEmbed`` (integer ppf); the
        # windowed ``forward_long`` path needs ``CausalWanRotaryPosEmbed`` which
        # accepts an absolute ``(start_f, end_f)`` window or explicit frame_index.
        head_dim = (
            dit.rope.attention_head_dim
            if hasattr(dit.rope, "attention_head_dim")
            else dit.blocks[0].attn.head_dim
        )
        dit.rope = CausalWanRotaryPosEmbed(
            attention_head_dim=head_dim,
            patch_size=dit.patch_size,
            max_seq_len=max_seq_len,
        ).to(device=self._device)
        dit.pos_embed_type = "casual_wan_rope"
        # ``forward_long`` lives on the Streaming subclass; bind it onto our base
        # instance (weights/buffers are shared via ``self``).
        dit.forward_long = _types.MethodType(SanaMSVideoCamCtrlStreaming.forward_long, dit)
        if hasattr(SanaMSVideoCamCtrlStreaming, "_is_softmax_option_y_block"):
            dit._is_softmax_option_y_block = SanaMSVideoCamCtrlStreaming._is_softmax_option_y_block
        self._cached_streaming_enabled = True

    @property
    def cached_streaming_enabled(self) -> bool:
        return bool(getattr(self, "_cached_streaming_enabled", False))

    @staticmethod
    def empty_kv_cache(num_blocks: int) -> list:
        """Fresh all-None GDN kv-cache: one 10-slot list per block.

        Slots 0/1 hold the GDN recurrent state ``S_kv (B,H,D,D)`` / ``S_z
        (B,H,D,1)``; the temporal-conv FFN uses the tail slot for left context.
        ``forward_long`` populates them on the first chunk and updates in place.
        """
        return [[None] * 10 for _ in range(num_blocks)]

    def run_chunk(
        self,
        chunk_latents: Tensor,
        timestep: Tensor,
        *,
        context: Tensor,
        kv_cache: list,
        start_f: int,
        end_f: int,
        save_kv_cache: bool = True,
        context_mask: Optional[Tensor] = None,
        seq_lens: Optional[Tensor] = None,
        frame_index: Optional[Tensor] = None,
        bridge_layers: Tuple[int, ...] = (),
        action_condition: Optional[Tensor] = None,
        frame_valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict, list]:
        """One cached-GDN chunk forward over an absolute latent-frame window.

        Args:
            chunk_latents: ``(B, C, T_chunk, H, W)`` noisy video for this chunk.
            timestep: ``(B,)`` video diffusion timestep.
            context: ``(B, L, D)`` raw caption context (forward_long embeds it).
            kv_cache: rolling GDN state cache (see :meth:`empty_kv_cache`),
                advanced in place and returned for the next chunk.
            start_f / end_f: absolute latent-frame window for this chunk's RoPE.
            bridge_layers: video block ids whose output hidden states are
                captured for the cross-attn action stream.
            action_condition: explicit ``(B, T_chunk, A)`` robot-action
                condition.  It is routed only through the vendor
                ``use_delta_pose_additive`` seam; no broadcast, implicit
                dtype/device conversion, or camera-pose alias is accepted.
            frame_valid_mask: explicit boolean ``(B, T_chunk)`` mask.  CACH
                requires it even for a full chunk so partial-tail padding can
                never be inferred from numeric values.  The main-tree adapter
                validates exact-zero fixed-K padding, compacts the common valid
                prefix before the vendor cached/Triton call, and restores
                fixed-K outputs with exact-zero padding.  The vendor cache thus
                advances through valid frames only.

        Returns:
            ``(video_pred_chunk, bridges, kv_cache)`` where ``bridges`` maps each
            requested layer id → ``(B, T_chunk_tokens, C)``. Bridge features are
            captured inside ``forward_long`` (the checkpoint-boundary output for
            each bridge layer), so they are autograd-connected — gradients from
            the action stream flow back into the video backbone.
        """
        if not self.cached_streaming_enabled:
            self.enable_cached_streaming()
        dit = self._dit
        action_seam_enabled = bool(
            self._pipe.config.get("use_delta_pose_additive", False)
        )
        if action_seam_enabled and action_condition is None:
            raise RuntimeError(
                "the enabled CACH action seam requires an explicit "
                "action_condition; identity-by-None is forbidden"
            )
        if action_seam_enabled and frame_valid_mask is None:
            raise RuntimeError(
                "the enabled CACH action seam requires an explicit "
                "frame_valid_mask"
            )
        prefix_plan: FixedKPrefixPlan | None = None
        if frame_valid_mask is not None:
            if not isinstance(frame_valid_mask, Tensor):
                raise TypeError("frame_valid_mask must be a torch Tensor")
            expected_mask_shape = (
                chunk_latents.shape[0],
                chunk_latents.shape[2],
            )
            if (
                frame_valid_mask.dtype is not torch.bool
                or tuple(frame_valid_mask.shape) != expected_mask_shape
            ):
                raise ValueError(
                    "frame_valid_mask must be exact boolean "
                    f"{expected_mask_shape}"
                )
            if frame_valid_mask.device != self._device:
                raise ValueError(
                    f"frame_valid_mask device must be {self._device}, "
                    f"got {frame_valid_mask.device}"
                )
            if frame_valid_mask.shape[1] > 1 and bool(
                (
                    (~frame_valid_mask[:, :-1])
                    & frame_valid_mask[:, 1:]
                ).any()
            ):
                raise ValueError("frame_valid_mask must be a contiguous prefix")
            prefix_plan = FixedKPrefixPlan.from_mask(frame_valid_mask)
        if action_condition is not None:
            if not isinstance(action_condition, Tensor):
                raise TypeError("action_condition must be a torch Tensor")
            if action_condition.ndim != 3:
                raise ValueError(
                    "action_condition must have exact shape (B, T_chunk, A)"
                )
            expected_shape = (
                chunk_latents.shape[0],
                chunk_latents.shape[2],
            )
            if tuple(action_condition.shape[:2]) != expected_shape:
                raise ValueError(
                    "action_condition batch/time axes must exactly match "
                    f"chunk_latents: expected={expected_shape}, "
                    f"got={tuple(action_condition.shape[:2])}"
                )
            configured_dim = int(
                self._pipe.config.get("delta_pose_additive_dim", 0) or 0
            )
            if configured_dim <= 0 or action_condition.shape[2] != configured_dim:
                raise ValueError(
                    "action_condition width differs from "
                    f"delta_pose_additive_dim={configured_dim}"
                )
            if not action_seam_enabled:
                raise RuntimeError(
                    "action_condition supplied while the registered additive "
                    "action seam is disabled"
                )
            if not bool(
                getattr(dit, "_cach_delta_pose_output_zero_verified", False)
            ):
                raise RuntimeError(
                    "action-conditioned DiT lacks the post-factory exact-zero "
                    "projection attestation"
                )
            if action_condition.dtype != self._dtype:
                raise TypeError(
                    f"action_condition dtype must be {self._dtype}, "
                    f"got {action_condition.dtype}"
                )
            if action_condition.device != self._device:
                raise ValueError(
                    f"action_condition device must be {self._device}, "
                    f"got {action_condition.device}"
                )
            assert frame_valid_mask is not None
            if bool(
                action_condition.masked_select(
                    ~frame_valid_mask.unsqueeze(-1)
                ).count_nonzero()
            ):
                raise ValueError(
                    "padded action_condition slots must be exact zero"
                )

        vendor_latents = chunk_latents
        vendor_action_condition = action_condition
        vendor_frame_index = frame_index
        if prefix_plan is not None:
            if end_f - start_f != prefix_plan.valid_slots:
                raise ValueError(
                    "absolute frame window must equal the valid prefix length"
                )
            vendor_latents = prefix_plan.compact_video(vendor_latents)
            vendor_frame_index = prefix_plan.compact_frame_index(frame_index)
            if action_condition is not None:
                vendor_action_condition = prefix_plan.compact_condition(
                    action_condition
                )
        vendor_latents = vendor_latents.to(
            device=self._device,
            dtype=self._dtype,
        )

        # context → y (B, 1, L, D); forward_long runs y_embedder internally.
        y = context.unsqueeze(1) if context.dim() == 3 else context
        B, L = y.shape[0], y.shape[2]
        # caption mask → (B, L) int16 (the layout forward_long expects).
        if context_mask is not None:
            mask = context_mask.to(dtype=torch.int16, device=y.device).view(B, L)
        elif seq_lens is not None:
            positions = torch.arange(L, device=y.device).view(1, L)
            mask = (positions < seq_lens.to(device=y.device).view(B, 1)).to(torch.int16)
        else:
            mask = torch.ones(B, L, dtype=torch.int16, device=y.device)

        # forward_long writes the bridge layers' checkpoint-boundary outputs into
        # this dict (grad-connected — a forward hook would escape an inner
        # activation and break non-reentrant checkpointing's recompute check).
        bridges: dict[int, Tensor] = {}
        video_pred, kv_cache = dit.forward_long(
            vendor_latents,
            timestep.to(device=self._device),
            y.to(device=self._device, dtype=self._dtype),
            mask=mask,
            start_f=start_f,
            end_f=end_f,
            frame_index=vendor_frame_index,
            kv_cache=kv_cache,
            save_kv_cache=save_kv_cache,
            bridge_layers=tuple(bridge_layers),
            bridge_out=bridges,
            **(
                {"delta_actions": vendor_action_condition}
                if vendor_action_condition is not None
                else {}
            ),
            # Cached/Triton GDN has no frame_valid_mask implementation.  The
            # exact common prefix was compacted above, so passing the mask on
            # would either fail or silently advance state through padding.
            # Eager block calls: the cached GDN state mutates in place, which is
            # incompatible with forward_long's internal grad checkpointing under
            # backprop. Per-chunk activations are small, so this is cheap.
            use_gradient_checkpointing=False,
        )
        if prefix_plan is not None:
            video_pred = prefix_plan.restore_video(video_pred)
            bridges = {
                layer_id: prefix_plan.restore_token_sequence(value)
                for layer_id, value in bridges.items()
            }
        return video_pred, bridges, kv_cache

    # ----------------------------------------------------------------
    # ABC: joint self-attention split
    # ----------------------------------------------------------------

    def pre_attn_at_layer(
        self, layer_id: int, state: BlockLoopState
    ) -> Tuple[Tensor, Tensor, Tensor, dict]:
        return self._split.block_pre_attn(layer_id, state.x, state.t_mod, state.freqs)

    def post_attn_at_layer(
        self,
        layer_id: int,
        state: BlockLoopState,
        attn_out: Tensor,
        post_state: dict,
    ) -> BlockLoopState:
        state.x = self._split.block_post_attn(
            layer_id,
            attn_out,
            post_state,
            y=state.context,
            y_lens=state.context_mask,
            f=state.f,
            h=state.h,
            w=state.w,
            temporal_chunk_frames=state.extras.get("ar_temporal_chunk_frames"),
            frame_is_pad=state.extras.get("frame_is_pad"),
        )
        return state

    # ----------------------------------------------------------------
    # ABC: preprocessing / decoding / device (not used by this model)
    # ----------------------------------------------------------------

    def preprocess_input(
        self,
        *,
        frames: Any = None,
        text: Any = None,
        pre_encoded_text: Optional[Tensor] = None,
        input_latents: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        proprio: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
        video_mask: Optional[Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        **kw: Any,
    ) -> dict:
        """Convert a raw sample → dict consumable by ``BaseWAMArchitecture.compute_loss``.

        SANA-Video is text-to-video; first-frame conditioning is out of scope
        here. ``FirstFrameConditioningTransform`` may inject
        ``first_frame_image`` / ``ref_images`` into the sample — those are
        silently dropped via ``**kw`` so the T2V semantics stay clean.

        Args:
            frames: ``list[list[PIL.Image]]`` per-sample video clips, or ``None``
              when ``input_latents`` is supplied (cache path).
            text: ``str`` or ``list[str]`` raw captions, or ``None`` when
              ``pre_encoded_text`` is supplied. Live encoding requires both
              ``self._pipe.text_encoder`` and ``self._pipe.tokenizer`` to be set.
            pre_encoded_text: ``(B, L, 2304)`` Gemma-2-2B last_hidden_state, or
              ``(L, 2304)`` (single sample, will be unsqueezed). Takes precedence
              over ``text`` when both are present (cache > live).
            input_latents: ``(B, 16, T_lat, H_lat, W_lat)`` pre-encoded latents;
              when set, the VAE encode step is skipped.

        Returns:
            Dict with ``input_latents``, ``context``, ``seq_lens``,
            ``num_clean_prefix_frames=0``, ``height``, ``width``, ``num_frames``,
            and any of ``actions`` / ``proprio`` / ``action_mask`` / ``video_mask``
            that were supplied.
        """
        device = self._device
        dtype = self._dtype

        # --- video → latents (cache > VAE encode) ---
        if input_latents is None:
            if frames is None:
                raise ValueError(
                    "SanaVideoBackbone.preprocess_input requires either 'frames' or 'input_latents'."
                )
            if self._pipe.vae is None:
                raise RuntimeError(
                    "SanaVideoBackbone.preprocess_input(frames=...) requires a loaded "
                    "VAE. Build the pipeline with vae_path pointing to Wan2.1_VAE.pth."
                )
            video = _pil_video_to_tensor(frames).to(device=device, dtype=dtype)
            B = video.shape[0]
            videos_list = [video[i] for i in range(B)]  # list of (3, T, H, W)
            with torch.no_grad():
                # ``tiled=True``: encode in (T,H,W) tiles. At 81 frames × 480p,
                # ``tiled=False`` would peak at ~30 GB of intermediate activations
                # through the WanVAE downsampling stack and OOM 80 GB cards once
                # the DiT/ActionDiT are also resident.
                input_latents = self._pipe.vae.encode(videos_list, device=device, tiled=True)
            input_latents = input_latents.to(device=device, dtype=dtype)
            vid_T, vid_H, vid_W = video.shape[2], video.shape[3], video.shape[4]
        else:
            input_latents = input_latents.to(device=device, dtype=dtype)
            B = input_latents.shape[0]
            vid_T = num_frames if num_frames is not None else input_latents.shape[2]
            sc = self._spatial_compression  # Wan=8, LTX2=32
            vid_H = height if height is not None else input_latents.shape[3] * sc
            vid_W = width if width is not None else input_latents.shape[4] * sc

        # --- text → context (pre-encoded > live Gemma) ---
        if pre_encoded_text is not None:
            context = pre_encoded_text
            if context.dim() == 2:
                context = context.unsqueeze(0)
            context = context.to(device=device, dtype=dtype)
        elif text is not None:
            if self._pipe.text_encoder is None or self._pipe.tokenizer is None:
                raise RuntimeError(
                    "Live text encoding requires a loaded Gemma-2-2B text encoder "
                    "and tokenizer. Pass `pre_encoded_text` or wire `text_encoder_name` "
                    "in the pipeline config."
                )
            context = self._encode_text(text).to(device=device, dtype=dtype)
        else:
            raise ValueError(
                "SanaVideoBackbone.preprocess_input requires either 'text' or 'pre_encoded_text'."
            )

        seq_lens = torch.full(
            (context.shape[0],), context.shape[1], dtype=torch.long, device=device
        )

        out: dict = {
            "input_latents": input_latents,
            "context": context,
            "seq_lens": seq_lens,
            "num_clean_prefix_frames": 0,  # SANA is pure T2V; no clean-prefix injection.
            "height": vid_H,
            "width": vid_W,
            "num_frames": vid_T,
        }
        if self._use_first_frame_cond:
            # Clean-prefix conditioning: latent frame 0 (= the first frame of the
            # training window = the current observation) is supplied clean.
            # The training loss clean-replaces ``latents[:, :, 0:1]`` with this
            # and trims it from the loss; ``prepare`` modulates it at t=0
            # (per-frame t-mod).
            out["first_frame_latents"] = input_latents[:, :, :1].clone()
            out["num_clean_prefix_frames"] = 1
        for name, value in (
            ("actions", actions),
            ("proprio", proprio),
            ("action_mask", action_mask),
            ("video_mask", video_mask),
        ):
            if value is not None:
                out[name] = value
        return out

    def _encode_text(self, text: Any) -> Tensor:
        """Tokenize + run Gemma-2-2B → ``last_hidden_state`` of shape (B, L, 2304).

        Live encoding path for deploy. Smoke tests bypass this by passing
        ``pre_encoded_text``.
        """
        if isinstance(text, str):
            text = [text]
        tokenizer = self._pipe.tokenizer
        encoder = self._pipe.text_encoder
        enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
            next(encoder.parameters()).device
        )
        with torch.no_grad():
            out = encoder(**enc, output_hidden_states=False)
        return out.last_hidden_state

    def get_submodule(self, name: str) -> nn.Module | None:
        return getattr(self._pipe, name, None)

    def set_submodule(self, name: str, module: nn.Module) -> None:
        if not hasattr(self._pipe, name):
            raise KeyError(f"SanaPipe has no submodule {name!r}")
        setattr(self._pipe, name, module)
        if name == "dit":
            # Re-target the split view at the new DiT instance.
            self._split = SanaMSVideoSplit(module)
        # Keep the nn.Module child registration in sync so parameter
        # discovery (self.parameters() / named_children()) sees the new
        # module. `add_module` overwrites self._modules[name] in place.
        if name in self._modules:
            self.add_module(name, module)

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        """Decode ``(B, 16, T_lat, H_lat, W_lat)`` latents → per-sample
        ``(3, T, H, W)`` videos in ``[-1, 1]``.

        Calls into the Wan2.1 VAE (``vae/Wan2.1_VAE.pth``) bundled with the
        SANA-Video 480p HF asset. ``tiled=True`` enables tile-by-tile decoding
        for inference paths where full-frame decode would OOM; training /
        smoke can pass ``tiled=False`` for speed.
        """
        if self._pipe.vae is None:
            raise RuntimeError(
                "SanaVideoBackbone.decode_video requires a loaded VAE. Build the "
                "pipeline with vae_path pointing to Wan2.1_VAE.pth."
            )
        B = latents.shape[0]
        latents_list = [latents[i] for i in range(B)]
        with torch.no_grad():
            videos = self._pipe.vae.decode(latents_list, device=self._device, tiled=tiled)
        return [videos[i] for i in range(videos.shape[0])]

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        self._dtype = dtype
        self._device = device
        self._pipe.dit.to(device=device, dtype=dtype)
        for name in ("vae", "text_encoder"):
            sub = getattr(self._pipe, name, None)
            if sub is not None:
                sub.to(device=device, dtype=dtype)
