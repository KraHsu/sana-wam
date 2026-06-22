"""Load a SANA-Video pipeline (DiT + VAE + text encoder) for sana-wam.

This wraps ``third_party/Sana`` upstream into a tiny ``SanaPipe`` object that
``SanaVideoBackbone`` then drives through the block-loop contract.
The goal here is **not** to reimplement SANA — we hold references to upstream
modules and feed them sana-wam-shaped inputs. Pin-bumps of the submodule are
transparent as long as the upstream class signatures wrapped here don't change.

Two pre-config'd variants are supported: ``sana_video_2b_480p`` (the only
HF-published video model at time of writing) and ``mini`` (random-weight tiny
model for unit tests). Real inference / training over the published 2B weights
requires the upstream ``diffusion.model.builder`` chain — we lazy-import that
chain on first use so that ``import sana_wam.model.video_backbone.sana`` stays
cheap and survives a broken timm install at import time.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# SANA's own inference scripts set this to disable xformers' SDPA hijack.
# Doing it on module import keeps the upstream's `_xformers_available` guard
# at False, which forces the deterministic mask path.
os.environ.setdefault("DISABLE_XFORMERS", "1")


@dataclass
class SanaPipe:
    """Thin container of the components ``SanaVideoBackbone`` needs.

    Mirrors the role of Wan's pipe object — backbone code reaches via
    attribute access, never deep-inspects this struct's internals.
    """

    dit: nn.Module
    """The ``SanaMSVideo`` DiT (or a mini variant for tests)."""

    vae: Optional[nn.Module] = None
    """``WanVAE`` — the SANA-Video 2B 480p HF ckpt re-uses the Wan VAE."""

    text_encoder: Optional[nn.Module] = None
    """Gemma-2-2B text encoder. Not bundled in the SANA HF ckpt — load
    separately via sana-wam's existing text-encoder utilities."""

    tokenizer: Optional[Any] = None

    scheduler: Optional[Any] = None
    """Flow-Euler scheduler with the upstream's video shift."""

    # Auxiliary metadata for the adapter
    config: dict = field(default_factory=dict)
    """Snapshot of upstream config (``model.*`` / ``vae.*`` / ``text_encoder.*``)."""


# ---------------------------------------------------------------------------
# Upstream model factory dispatch
# ---------------------------------------------------------------------------


def _resolve_upstream_factory(model_name: str) -> Any:
    """Return the upstream factory function registered under ``model_name``.

    Examples: ``SanaMSVideo_2000M_P2_D20`` is the 2B 480p model factory at
    ``third_party/Sana/diffusion/model/nets/sana_multi_scale_video.py:1054``.
    Importing the factory triggers `MODELS.register_module()` side effects
    in upstream, so the lookup must happen after the import.
    """
    # Lazy import — keeps adapter cheap to import and tolerates partial venvs.
    from diffusion.model.builder import MODELS  # type: ignore[import-not-found]
    from diffusion.model.nets import sana_multi_scale_video  # noqa: F401  # registers factories

    if model_name not in MODELS._module_dict:
        raise KeyError(
            f"SANA model factory {model_name!r} not registered. Available: "
            f"{sorted(MODELS._module_dict)[:8]}..."
        )
    return MODELS.get(model_name)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_sana_pipeline(
    cfg_or_path: Any,
    *,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    ckpt_dir: Optional[str] = None,
) -> SanaPipe:
    """Build a ``SanaPipe`` from a config or model directory.

    Accepts the same three input shapes as ``WanVideoBackbone.from_pretrained``:

    - ``omegaconf.DictConfig``: full Hydra config → reads ``video_backbone.*``
    - ``str`` directory: looks for ``checkpoints/*.pth`` + ``config.json``
    - ``dict``: full architecture params (or ``{"video_backbone": {...}}``)
      → reads ``video_backbone.*``; ``resolve_architecture_config`` returns
      plain dicts so this is the actual training-time entry point.
    """
    from omegaconf import DictConfig

    if isinstance(cfg_or_path, DictConfig):
        vb_cfg = cfg_or_path.get("video_backbone", cfg_or_path)
        spec = _spec_from_dictconfig(vb_cfg, ckpt_dir=ckpt_dir)
    elif isinstance(cfg_or_path, str):
        if not os.path.isdir(cfg_or_path):
            raise ValueError(
                f"build_sana_pipeline(str) expects a directory, got: {cfg_or_path!r}"
            )
        # str-dir entry is a smoke/test shortcut for "give me a pipe straight
        # from a HF bundle dir" — there's no sana-wam ckpt in this shape, so
        # the deploy self-contained fallback doesn't apply. Stay strict.
        spec = _spec_from_model_dir(cfg_or_path)
    elif isinstance(cfg_or_path, dict):
        # Unwrap ``{video_backbone: {...}}`` if the caller passed full
        # architecture params (which is what ``resolve_architecture_config``
        # produces).
        vb_cfg = cfg_or_path.get("video_backbone", cfg_or_path)
        spec = _spec_from_dict(vb_cfg, ckpt_dir=ckpt_dir)
    else:
        raise TypeError(
            f"build_sana_pipeline: unsupported source type {type(cfg_or_path).__name__}"
        )

    return _build_pipe_from_spec(spec, device=device, dtype=dtype, ckpt_dir=ckpt_dir)


def build_mini_sana_pipeline(
    *,
    depth: int = 2,
    hidden_size: int = 128,
    num_heads: int = 4,
    linear_head_dim: int = 32,
    f: int = 4,
    h: int = 8,
    w: int = 8,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    attn_kernel: str = "linear_relu",
    chunk_size: int = 3,
) -> SanaPipe:
    """Mini-config factory for unit tests: random-weight ``SanaMSVideo``.

    Avoids downloading the 2B checkpoint or initializing Gemma-2-2B. The
    resulting pipe has only ``dit`` populated; VAE / text encoder are ``None``
    and tests must not call ``preprocess_input`` / ``decode_video``.

    ``attn_kernel="gdn"`` builds a ChunkCausalGDNTriton mini DiT (GPU/Triton-only;
    CPU construction works but forward needs CUDA). Default ``linear_relu`` keeps
    the CPU-testable LiteLAReLURope path.
    """
    # GDN attention is only wired in the CamCtrl model class (its block builds
    # self.attn via ATTENTION_BLOCKS.get(attn_type); the plain SanaMSVideo block
    # has no GDN branch → self.attn=None). With camctrl_type unset the CamCtrl
    # block is a plain GDN self-attn block (no camera path) — exactly the WAM
    # need. linear_relu keeps the plain SanaMSVideo class (CPU-testable).
    _use_gdn = attn_kernel == "gdn"
    _attn_type = "ChunkCausalGDNTriton" if _use_gdn else "LiteLAReLURope"
    # Touch the upstream factory to trigger ``MODELS.register_module`` side effects.
    _resolve_upstream_factory("SanaMSVideo_2000M_P2_D20")
    # Direct class construction — bypass the factory wrapper so we can override
    # hidden_size + depth from defaults of the 2B model. GDN lives only in the
    # CamCtrl class; linear_relu uses the plain class.
    common = dict(
        input_size=h,
        # GDN production factory (SanaMSVideoCamCtrl_1600M_P1_D20) uses patch (1,1,1);
        # match it for the GDN mini so test token geometry mirrors production. The
        # linear_relu mini keeps SANA-Video's (1,2,2).
        patch_size=(1, 1, 1) if _use_gdn else (1, 2, 2),
        in_channels=16,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=2.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        pred_sigma=False,
        attn_type=_attn_type,
        ffn_type="GLUMBConvTemp",
        use_pe=True,
        pos_embed_type="wan_rope",
        qk_norm=True,
        cross_norm=True,
        y_norm=True,
        linear_head_dim=linear_head_dim,
        t_kernel_size=3,
        mlp_acts=("silu", "silu", None),
        model_max_length=8,
        caption_channels=64,
    )
    if _use_gdn:
        from diffusion.model.nets.sana_multi_scale_video_camctrl import (  # noqa: E402
            SanaMSVideoCamCtrl,
        )

        # camctrl_layers_num=0 → every block routes to the plain GDN self-attn
        # (ATTENTION_BLOCKS.get(attn_type)), no UCPE camera branch — the WAM
        # needs GDN dynamics, not camera-pose control.
        dit = SanaMSVideoCamCtrl(
            **common,
            camctrl_layers_num=0,
            chunk_size=chunk_size,
            chunk_split_strategy="uniform",
            conv_kernel_size=4,
            k_conv_only=True,
        )
    else:
        from diffusion.model.nets.sana_multi_scale_video import (  # noqa: E402
            SanaMSVideo as SanaMSVideoCls,
        )

        dit = SanaMSVideoCls(**common)
    dit = dit.to(device=device, dtype=dtype).eval()

    from sana_wam.model.video_backbone.sana.scheduler import SanaFlowSchedulerAdapter

    return SanaPipe(
        dit=dit,
        scheduler=SanaFlowSchedulerAdapter(),
        config={
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "fhw": (f, h, w),
            "attn_kernel": attn_kernel,
            "chunk_size": chunk_size,
        },
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass
class _PipeSpec:
    """Parsed inputs to ``_build_pipe_from_spec``."""

    model_factory: str = "SanaMSVideo_2000M_P2_D20"
    model_path: Optional[str] = None
    """Path to ``SANA_Video_2B_480p.pth`` (or HF URL); ``None`` ⇒ random init."""
    vae_path: Optional[str] = None
    vae_type: str = "wan"
    """VAE family. ``"wan"`` (default) = Wan2.1 causal VAE (16ch, 4x temporal,
    8x spatial) loaded from a ``.pth`` — the SANA-Video 2B 480p contract.
    ``"ltx2"`` = LTX2 causal VAE (128ch, 8x temporal, 32x spatial) loaded from a
    diffusers directory via ``AutoencoderKLLTX2Video.from_pretrained``; this is
    the VAE the SANA-WM streaming world model (``init_dit_from``) was trained
    with, so its 128-channel latent matches that DiT's ``in_channels``."""
    text_encoder_name: Optional[str] = None
    model_kwargs: dict = field(default_factory=dict)
    flow_shift: float = 3.0
    """Rectified-flow shift for the scheduler. Default ``3.0`` matches
    SANA-Video upstream (``model_wrapper.py:17``)."""
    fp32_attention: bool = False
    """Run the linear-attention numerator/denominator matmuls in fp32.

    Mirrors upstream Sana-wm's ``fp32_attention: true`` (e.g.
    ``../Sana/configs/sana_wm/sana_wm_chunk_causal_1600m_720p.yaml``). The
    blocks-split forward already honors ``getattr(attn, "fp32_attention", False)``
    (``blocks_split.py``); this flag is what finally *sets* that attribute on
    every attention module, closing a previously dead config path. Default
    ``False`` keeps bf16 attention (byte-identical to existing fork behavior)."""
    attn_kernel: str = "linear_relu"
    """Video self-attention family. ``"linear_relu"`` = SANA ``LiteLAReLURope``
    (factorized linear attn, the published 480p checkpoint). ``"gdn"`` = upstream
    Sana-wm ``ChunkCausalGDNTriton`` (gated delta net, frame-wise recurrence +
    chunk-causal). GDN swaps the preset's ``attn_type`` + GDN knobs and is
    GPU/Triton-only; the linear_relu checkpoint cannot be reused under GDN."""
    chunk_size: int = 3
    """GDN chunk size (frames per chunk) for chunk-causal attention. Ignored for
    linear_relu. Mirrors ``configs/sana_wm/*`` ``chunk_size``."""
    use_first_frame_cond: bool = False
    """Opt-in: condition the generated video on a clean frame-0 (the current
    observation). Default ``False`` keeps SANA pure text-to-video
    (byte-identical to existing checkpoints). When ``True`` the adapter sets
    ``first_frame_latents`` + ``num_clean_prefix_frames=1`` and the split-forward
    switches to per-frame timestep modulation (frame-0 at t=0)."""
    init_dit_from: Optional[str] = None
    """Opt-in (GDN only): path to a pretrained GDN DiT checkpoint to load AFTER
    construction. This is the seam for the SANA-WM streaming world model
    (``/DATA/share/SANA-WM_streaming/sana_dit/model.pt``) — a ``.pt`` GAN training
    state whose DiT weights live under ``['generator']`` with a ``model.`` prefix.
    Unlike ``model_path`` (which loads the linear_relu published ckpt and is
    rejected for GDN), ``init_dit_from`` builds the FULL CamCtrl streaming arch
    (so state_dict keys match) and loads it. Default ``None`` keeps GDN
    from-scratch (byte-identical to the existing GDN-AR track)."""


def _resolve_model_path_and_kwargs(
    model_path: Optional[str],
    model_kwargs: dict,
    vae_path: Optional[str],
    *,
    ckpt_dir: Optional[str] = None,
) -> tuple:
    """Shared auto-discovery + foot-gun guard for ``_spec_from_*``.

    Resolves ``model_path`` against two acceptable shapes:

    1. Local bundle directory (HF snapshot layout: ``config.json`` +
       ``checkpoints/*.pth`` + ``vae/Wan2.1_VAE.pth``) → returns the
       expanded ckpt path and auto-applies the published ``model_kwargs``
       preset (unless yaml already supplied explicit kwargs).
    2. Local ``.pth`` / ``.safetensors`` file path WITH explicit
       ``model_kwargs`` — caller is on the hook for the architecture spec.

    Raises ``ValueError`` on any other shape (HF URL, repo id, missing
    path, etc.) when ``model_kwargs`` is empty — i.e. anything where we
    couldn't determine the architecture. Without this guard, the factory
    would fall back to its bare defaults (``attn_type=flash``,
    ``ffn_type=mlp``, ``in_channels=4``, ``pred_sigma=True``) and
    ``load_state_dict(strict=False)`` would drop most of the published
    ckpt weights silently, leaving the DiT mostly random.

    Deploy self-contained fallback: when ``ckpt_dir`` is set (we are
    inside :func:`sana_wam.deploy.model_loader.load_from_checkpoint_dir`)
    and the bundle dir has ``config.json`` but no ``checkpoints/*.pth``,
    derive ``model_kwargs`` from ``config.json`` alone and return
    ``model_path=None`` so :func:`_build_pipe_from_spec` skips the
    pretrained ``.pth`` load. ``SanaVideoBackbone`` ``add_module``-
    registers the DiT (and VAE / text encoder) as nn.Module children
    (``adapter.py``), so the trained safetensors carries the full SANA
    backbone — the next ``architecture.load_checkpoint(...)`` call will
    overwrite the freshly-instantiated DiT regardless. Training callers
    (``ckpt_dir is None``) keep the strict fail-fast behaviour: training
    from random init would silently produce garbage.

    Returns ``(model_path, model_kwargs, vae_path)``.
    """
    if isinstance(model_path, str) and os.path.isdir(model_path):
        try:
            discovered = _spec_from_model_dir(model_path)
        except FileNotFoundError:
            if ckpt_dir is None:
                raise
            preset_kwargs = _preset_kwargs_from_config_json(model_path)
            if not preset_kwargs and not model_kwargs:
                raise
            vae_candidate = os.path.join(model_path, "vae", "Wan2.1_VAE.pth")
            if vae_path is None and os.path.isfile(vae_candidate):
                vae_path = vae_candidate
            if not model_kwargs:
                model_kwargs = preset_kwargs
            logger.info(
                "SANA deploy self-contained: %s/checkpoints/*.pth not found; "
                "using config.json preset for architecture spec and skipping "
                "pretrained DiT load (trained backbone weights load from %s).",
                model_path,
                ckpt_dir,
            )
            return None, model_kwargs, vae_path
        # Auto-discovered ckpt overrides ``model_path: <dir>`` (the dir is just
        # the bundle marker — the actual .pth lives in ``<dir>/checkpoints/``).
        model_path = discovered.model_path
        if vae_path is None:
            vae_path = discovered.vae_path
        if not model_kwargs:
            model_kwargs = discovered.model_kwargs
    elif isinstance(model_path, str) and model_path and not model_kwargs:
        raise ValueError(
            "SANA video_backbone.model_path is set but not a local bundle "
            f"directory ({model_path!r}) and no model_kwargs were provided. "
            "Either:\n"
            "  (a) Download the SANA-Video bundle locally and point "
            "model_path at the unpacked directory — _spec_from_model_dir "
            "will auto-apply the published preset; see docs/sana_vendor.md.\n"
            "  (b) Set model_kwargs explicitly in the yaml to match the "
            "checkpoint architecture (in_channels / attn_type / ffn_type / "
            "pred_sigma / etc.).\n"
            "Falling back to factory defaults here would silently load a "
            "mismatched DiT (strict=False drops most weights) and train "
            "against mostly-random init."
        )

    return model_path, model_kwargs, vae_path


def _spec_from_dictconfig(vb_cfg, *, ckpt_dir: Optional[str] = None) -> _PipeSpec:
    """Build spec from a Hydra ``video_backbone`` config.

    Yaml-level entry. Auto-discovery + foot-gun guard are shared with
    :func:`_spec_from_dict` (the plain-dict path that
    ``resolve_architecture_config`` produces) via
    :func:`_resolve_model_path_and_kwargs` — both training-time entry
    shapes go through the same gate. ``ckpt_dir`` is forwarded to
    distinguish deploy from training (deploy can self-contain).
    """
    model_path, model_kwargs, vae_path = _resolve_model_path_and_kwargs(
        model_path=vb_cfg.get("model_path"),
        model_kwargs=dict(vb_cfg.get("model_kwargs", {})),
        vae_path=vb_cfg.get("vae_path"),
        ckpt_dir=ckpt_dir,
    )

    return _PipeSpec(
        model_factory=str(vb_cfg.get("model_factory", "SanaMSVideo_2000M_P2_D20")),
        model_path=model_path,
        vae_path=vae_path,
        vae_type=str(vb_cfg.get("vae_type", "wan")),
        text_encoder_name=vb_cfg.get("text_encoder_name", "gemma-2-2b-it"),
        model_kwargs=model_kwargs,
        flow_shift=float(vb_cfg.get("flow_shift", 3.0)),
        fp32_attention=bool(vb_cfg.get("fp32_attention", False)),
        attn_kernel=str(vb_cfg.get("attn_kernel", "linear_relu")),
        chunk_size=int(vb_cfg.get("chunk_size", 3)),
        use_first_frame_cond=bool(vb_cfg.get("use_first_frame_cond", False)),
        init_dit_from=vb_cfg.get("init_dit_from", None),
    )


# Factory model_kwargs that must be passed to ``SanaMSVideo_2000M_P2_D20`` to
# reconstruct the architecture matching the published SANA-Video 2B 480p
# checkpoint. The factory's bare defaults are ``attn_type="flash",
# ffn_type="mlp", in_channels=4, pred_sigma=True`` — those silently load only
# a fraction of the ckpt under ``strict=False`` (random FFN, dropped temporal
# convs, mis-sized pos_embed). Source: SANA upstream training config
# ``configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml``.
_SANA_VIDEO_2B_480P_PRESET: dict = {
    "in_channels": 16,
    "input_size": 60,
    "attn_type": "LiteLAReLURope",
    "linear_head_dim": 112,
    "ffn_type": "GLUMBConvTemp",
    "qk_norm": True,
    "pred_sigma": False,
    "learn_sigma": False,
    "caption_channels": 2304,
    "model_max_length": 300,
    "mlp_ratio": 3,
    "cross_norm": True,
    "use_pe": True,
    "pos_embed_type": "wan_rope",
}

# GDN (gated delta net) overrides applied on top of the base preset when
# ``attn_kernel="gdn"``. Mirrors ``configs/sana_wm/sana_wm_chunk_causal_*.yaml``:
# ChunkCausalGDNTriton self-attn + the temporal short-conv knobs the GDN block
# expects. ``ffn_type`` stays GLUMBConvTemp (same as the 480p preset). There is
# no published GDN checkpoint for this arch, so GDN always random-inits and
# trains from scratch (the linear_relu weights are incompatible).
_GDN_ATTN_OVERRIDES: dict = {
    "attn_type": "ChunkCausalGDNTriton",
    "conv_kernel_size": 4,
    "k_conv_only": True,
    # Plain GDN self-attn on every layer (no UCPE camera branch). The WAM needs
    # GDN dynamics, not camera-pose control.
    "camctrl_layers_num": 0,
}

# Full architecture spec for GDN-from-scratch. The CamCtrl factory
# (SanaMSVideoCamCtrl_1600M_P1_D20) sets depth/hidden/patch/num_heads, but NOT
# in_channels (default 4) etc. The Wan VAE produces 16-channel video latents, so
# without in_channels=16 the patch-embed conv3d rejects the input. These mirror
# the linear_relu 480p preset (minus what the factory fixes) so a random-init GDN
# DiT matches the data/VAE/text contract. There is no published GDN checkpoint.
_GDN_ARCH_PRESET: dict = {
    "in_channels": 16,
    "ffn_type": "GLUMBConvTemp",
    "qk_norm": True,
    "pred_sigma": False,
    "learn_sigma": False,
    "caption_channels": 2304,
    "model_max_length": 300,
    "mlp_ratio": 3,
    "cross_norm": True,
    "use_pe": True,
    "pos_embed_type": "wan_rope",
}

# The GDN attn class is only wired in SanaMSVideoCamCtrl; the plain SanaMSVideo
# model has no GDN branch. So attn_kernel="gdn" also switches the model factory.
_GDN_MODEL_FACTORY = "SanaMSVideoCamCtrl_1600M_P1_D20"

# Streaming (cached chunk-causal AR) factory — same params as the non-streaming
# CamCtrl, but the cached-KV attention classes needed for forward_long rollout.
# Used when loading the SANA-WM streaming world model (init_dit_from).
_GDN_MODEL_FACTORY_STREAMING = "SanaMSVideoCamCtrlStreaming_1600M_P1_D20"

# Architecture kwargs that make a freshly-built GDN DiT's state_dict match the
# pretrained SANA-WM streaming checkpoint EXACTLY (872/872 keys, 0 missing/0
# unexpected; verified /tmp/m0_load_smoke.py 2026-06-22). Applied ONLY when
# init_dit_from is set, so the from-scratch GDN-AR track is unchanged. Every value
# here differs from a factory default and is required:
#   - in_channels=128: LTX2 latent (x_embedder (2240,128,1,1,1))
#   - linear_head_dim=112: 2240/112=20 GDN heads (default 32 → 70, WRONG)
#   - camctrl_layers_num=20 + softmax_every_n=4: cam branch on all 20 blocks,
#     softmax at {3,7,11,15,19}, GDN on the other 15
#   - y_norm=True: ckpt has attention_y_norm.weight
#   - chunk_plucker_* / cam_attn_compress / init_cam_from_base / chunk_split_strategy:
#     the trained camera-control sub-network the ckpt carries
_GDN_PRETRAINED_PRESET: dict = {
    "in_channels": 128,
    "pred_sigma": False,
    "learn_sigma": False,
    "caption_channels": 2304,
    "model_max_length": 300,
    "attn_type": "ChunkCausalGDNTriton",
    "ffn_type": "GLUMBConvTemp",
    "qk_norm": True,
    "cross_norm": True,
    "use_pe": True,
    "pos_embed_type": "wan_rope",
    "y_norm": True,
    "linear_head_dim": 112,
    "mlp_ratio": 3,
    "conv_kernel_size": 4,
    "k_conv_only": True,
    "t_kernel_size": 3,
    "camctrl_layers_num": 20,
    "softmax_every_n": 4,
    "cam_attn_compress": 1,
    "chunk_split_strategy": "first_chunk_plus_one",
    "init_cam_from_base": True,
    "use_chunk_plucker_post_attn": True,
    "chunk_plucker_channels": 48,
    "chunk_plucker_post_attn_blocks": 20,
}


def _apply_attn_kernel(model_kwargs: dict, attn_kernel: str, *, pretrained_gdn: bool = False) -> dict:
    """Return ``model_kwargs`` adjusted for ``attn_kernel``.

    ``"linear_relu"`` leaves the preset's ``attn_type`` (LiteLAReLURope) intact.
    ``"gdn"`` swaps to ChunkCausalGDNTriton and adds the GDN short-conv knobs.
    For the arch spec it fills EITHER the from-scratch preset (``in_channels=16``,
    no camera branch — the default GDN-AR track) OR, when ``pretrained_gdn`` is
    set, the SANA-WM preset (``in_channels=128``, full camctrl, 20 GDN heads,
    softmax-every-4) so a freshly-built DiT matches the pretrained checkpoint
    exactly. Caller-pinned yaml values always win (``setdefault``).
    """
    if attn_kernel == "gdn":
        merged = dict(model_kwargs)
        if pretrained_gdn:
            # SANA-WM pretrained: the preset is self-complete (carries attn_type +
            # the full camctrl spec). Do NOT layer _GDN_ATTN_OVERRIDES on top — its
            # camctrl_layers_num=0 would (via setdefault) suppress the camera branch
            # the checkpoint needs.
            presets = (_GDN_PRETRAINED_PRESET,)
        else:
            presets = (_GDN_ATTN_OVERRIDES, _GDN_ARCH_PRESET)
        for preset in presets:
            for k, v in preset.items():
                merged.setdefault(k, v)
        # A linear_relu preset's attn_type must be overridden for GDN.
        if str(merged.get("attn_type", "")).endswith("LiteLAReLURope"):
            merged["attn_type"] = _GDN_ATTN_OVERRIDES["attn_type"]
        return merged
    if attn_kernel not in ("linear_relu", "softmax"):
        raise ValueError(
            f"Unknown video_backbone.attn_kernel={attn_kernel!r}. "
            "Choose from: linear_relu, gdn, softmax."
        )
    return dict(model_kwargs)


def _preset_kwargs_from_config_json(path: str) -> dict:
    """Read the HF bundle marker ``config.json`` and return the matching
    architecture preset (empty dict if no recognizable ``model_name``).

    Shared between :func:`_spec_from_model_dir` (when a ``.pth`` is present)
    and :func:`_resolve_model_path_and_kwargs` (deploy self-contained
    fallback when ``.pth`` is absent).
    """
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return {}
    import json

    with open(config_path) as fh:
        bundle_cfg = json.load(fh)
    model_name = str(bundle_cfg.get("model_name", ""))
    if model_name.startswith("SANA-Video-2B-480"):
        return dict(_SANA_VIDEO_2B_480P_PRESET)
    return {}


def _spec_from_model_dir(path: str) -> _PipeSpec:
    """Auto-discover from a directory containing ``checkpoints/*.pth``."""
    # The HF SANA-Video repo layout is documented in docs/sana_vendor.md.
    ckpt_candidates = []
    ckpt_dir = os.path.join(path, "checkpoints")
    if os.path.isdir(ckpt_dir):
        ckpt_candidates = [
            os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pth")
        ]
    if not ckpt_candidates:
        raise FileNotFoundError(f"No checkpoints/*.pth under {path}")
    vae_path = os.path.join(path, "vae", "Wan2.1_VAE.pth")

    return _PipeSpec(
        model_path=ckpt_candidates[0],
        vae_path=vae_path if os.path.isfile(vae_path) else None,
        text_encoder_name="gemma-2-2b-it",
        model_kwargs=_preset_kwargs_from_config_json(path),
    )


def _spec_from_dict(d: dict, *, ckpt_dir: Optional[str] = None) -> _PipeSpec:
    """Build spec from a plain ``video_backbone`` dict.

    Plain-dict entry; this is the **training-time** path —
    ``resolve_architecture_config`` returns plain dicts (not DictConfig),
    so the trainer / deploy loader land here. Auto-discovery +
    foot-gun guard are shared with :func:`_spec_from_dictconfig` via
    :func:`_resolve_model_path_and_kwargs`. ``ckpt_dir`` is forwarded to
    distinguish deploy from training (deploy can self-contain).
    """
    model_path, model_kwargs, vae_path = _resolve_model_path_and_kwargs(
        model_path=d.get("model_path"),
        model_kwargs=dict(d.get("model_kwargs", {})),
        vae_path=d.get("vae_path"),
        ckpt_dir=ckpt_dir,
    )

    return _PipeSpec(
        model_factory=d.get("model_factory", "SanaMSVideo_2000M_P2_D20"),
        model_path=model_path,
        vae_path=vae_path,
        vae_type=str(d.get("vae_type", "wan")),
        text_encoder_name=d.get("text_encoder_name", "gemma-2-2b-it"),
        model_kwargs=model_kwargs,
        flow_shift=float(d.get("flow_shift", 3.0)),
        fp32_attention=bool(d.get("fp32_attention", False)),
        attn_kernel=str(d.get("attn_kernel", "linear_relu")),
        chunk_size=int(d.get("chunk_size", 3)),
        use_first_frame_cond=bool(d.get("use_first_frame_cond", False)),
        init_dit_from=d.get("init_dit_from", None),
    )


def _build_pipe_from_spec(
    spec: _PipeSpec,
    *,
    device: Optional[str],
    dtype: torch.dtype,
    ckpt_dir: Optional[str],
) -> SanaPipe:
    # GDN lives only in the CamCtrl model class — switch the factory too (a
    # linear_relu preset would name the plain SanaMSVideo factory, whose block
    # has no GDN branch). The user's explicit model_factory wins only if it's
    # already a CamCtrl factory.
    factory_name = spec.model_factory
    if spec.attn_kernel == "gdn" and "CamCtrl" not in factory_name:
        factory_name = _GDN_MODEL_FACTORY
        # Loading the pretrained SANA-WM world model needs the cached chunk-causal
        # streaming arch (its keys + the forward_long rollout path).
        if spec.init_dit_from is not None:
            factory_name = _GDN_MODEL_FACTORY_STREAMING
    factory = _resolve_upstream_factory(factory_name)
    model_kwargs = _apply_attn_kernel(
        spec.model_kwargs, spec.attn_kernel, pretrained_gdn=(spec.init_dit_from is not None)
    )
    if spec.attn_kernel == "gdn":
        model_kwargs.setdefault("chunk_size", spec.chunk_size)
    dit = factory(**model_kwargs)

    # Close the dead config path: blocks_split / GDN honor getattr(attn,
    # "fp32_attention", ...) but nothing set it. Stamp it onto every attention
    # submodule. linear_relu (LiteLA*) exposes ``kernel_func``; GDN exposes
    # ``A_log`` (the state-decay param) instead — both carry ``eps``.
    if spec.fp32_attention:
        n_set = 0
        for module in dit.modules():
            is_attn = hasattr(module, "eps") and (
                hasattr(module, "kernel_func") or hasattr(module, "A_log")
            )
            if is_attn:
                module.fp32_attention = True
                n_set += 1
        logger.info("fp32_attention enabled on %d SANA attention modules", n_set)

    # GDN is a different operator family — the published linear_relu checkpoint
    # cannot initialize it. Refuse a model_path (linear_relu) weight load so we
    # never silently train a half-random DiT against the wrong checkpoint. The
    # supported way to start GDN from real weights is ``init_dit_from`` (a GDN
    # checkpoint), handled below.
    if spec.attn_kernel == "gdn" and spec.model_path is not None:
        raise ValueError(
            "video_backbone.attn_kernel='gdn' is incompatible with loading the "
            f"linear_relu checkpoint at model_path={spec.model_path!r}. GDN must "
            "start from scratch (unset model_path) or load a GDN checkpoint via "
            "video_backbone.init_dit_from (e.g. the SANA-WM streaming model.pt)."
        )

    # GDN pretrained-weight seam: load the SANA-WM streaming world model. Its .pt
    # is a GAN training state — DiT weights live under ['generator'] with a
    # 'model.' prefix. The built streaming CamCtrl arch matches 872/872 keys
    # (verified M0); only the resolution-dependent ``pos_embed`` buffer differs
    # (vestigial under wan_rope — RoPE supplies position at runtime), so we drop
    # it and let the model keep its own.
    if spec.attn_kernel == "gdn" and spec.init_dit_from is not None:
        ckpt = torch.load(spec.init_dit_from, map_location="cpu", weights_only=False, mmap=True)
        gen = ckpt.get("generator", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(gen, dict) and "state_dict" in gen:
            gen = gen["state_dict"]
        state = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in gen.items()
        }
        model_sd = dit.state_dict()
        dropped = []
        for k in list(state.keys()):
            if k in model_sd and tuple(state[k].shape) != tuple(model_sd[k].shape):
                # Resolution-dependent buffers (pos_embed) — keep the model's own.
                dropped.append((k, tuple(state[k].shape), tuple(model_sd[k].shape)))
                state.pop(k)
        result = dit.load_state_dict(state, strict=False)
        if dropped:
            logger.warning(
                "init_dit_from: kept model init for %d shape-mismatched buffer(s) "
                "(resolution-dependent, e.g. pos_embed): %s",
                len(dropped),
                dropped[:4],
            )
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                "init_dit_from partial load: %d missing, %d unexpected. "
                "missing=%s unexpected=%s",
                len(result.missing_keys),
                len(result.unexpected_keys),
                result.missing_keys[:6],
                result.unexpected_keys[:6],
            )
        logger.info("Loaded SANA-WM GDN DiT from init_dit_from=%s", spec.init_dit_from)

    if spec.model_path is not None:
        # ``find_model`` is the upstream loader for ``.pth`` files; lazy import.
        from tools.download import find_model  # type: ignore[import-not-found]

        state = find_model(spec.model_path)
        if isinstance(state, dict) and "state_dict" in state and not any(
            k.startswith("blocks.") for k in state.keys()
        ):
            state = state["state_dict"]
        # SANA's load_state_dict tolerates shape mismatches by padding (see
        # sana_multi_scale_video.py:844-1016) — keep ``strict=False`` so we
        # don't trip on optional null-embed buffers.
        result = dit.load_state_dict(state, strict=False)
        # Surface partial loads: a clean published-ckpt load against the right
        # preset has 0 missing / 0 unexpected. Default factory kwargs vs the
        # 480p ckpt previously dropped GLUMBConvTemp + temporal-conv weights
        # silently and the smoke would just produce garbage.
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                "SANA ckpt load partial: %d missing, %d unexpected keys. "
                "Sample missing=%s sample unexpected=%s",
                len(result.missing_keys),
                len(result.unexpected_keys),
                result.missing_keys[:3],
                result.unexpected_keys[:3],
            )
        logger.info("Loaded SANA DiT weights from %s", spec.model_path)

    dit = dit.to(device=device or "cuda", dtype=dtype).eval()

    vae = (
        _load_vae(spec.vae_path, device=device, dtype=dtype, vae_type=spec.vae_type)
        if spec.vae_path
        else None
    )
    text_encoder, tokenizer = (
        _load_text_encoder(spec.text_encoder_name, device=device, dtype=dtype, ckpt_dir=ckpt_dir)
        if spec.text_encoder_name
        else (None, None)
    )

    from sana_wam.model.video_backbone.sana.scheduler import SanaFlowSchedulerAdapter

    scheduler = SanaFlowSchedulerAdapter(flow_shift=spec.flow_shift)

    return SanaPipe(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        config={
            "factory": spec.model_factory,
            "model_path": spec.model_path,
            "vae_path": spec.vae_path,
            "vae_type": spec.vae_type,
            "flow_shift": spec.flow_shift,
            "attn_kernel": spec.attn_kernel,
            "chunk_size": spec.chunk_size,
            "use_first_frame_cond": spec.use_first_frame_cond,
        },
    )


class _LTX2VAEShim(nn.Module):
    """Adapt the diffusers ``AutoencoderKLLTX2Video`` to the Wan VAE contract.

    The adapter calls ``vae.encode(list[(3,T,H,W)], device, tiled) -> (B,z,Tl,Hl,Wl)``
    and ``vae.decode(iterable[(z,Tl,Hl,Wl)], device, tiled) -> (B,3,T,H,W)``
    (see ``WanVideoVAE.encode``/``decode``). This wrapper exposes the same two
    methods on top of the diffusers per-batch tensor API.

    Geometry (from the LTX2 causal VAE config + SANA-WM bidirectional config):
    128-channel latent, 8x causal temporal (``Tl=(T-1)/8+1``), 32x spatial.
    ``scaling_factor=1.0`` and there are NO ``latents_mean/std`` → latents are
    used raw, exactly as the SANA-WM DiT was trained (and matching the Wan path,
    which also applies no extra scaling at the adapter). We take the latent
    distribution's ``mode()`` (deterministic) so cached latents are stable.

    Held as an ``nn.Module`` submodule so ``.to()`` / ``.requires_grad_()`` and
    the adapter's ``add_module("vae", ...)`` propagate to the diffusers VAE.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        cfg = vae.config
        self.z_dim = int(getattr(cfg, "latent_channels", 128))
        self.temporal_compression = int(getattr(cfg, "temporal_compression_ratio", 8))
        self.spatial_compression = int(getattr(cfg, "spatial_compression_ratio", 32))
        # Expose under the Wan attribute name too, so any spatial-factor probe
        # that falls back to ``upsampling_factor`` reads 32 (not Wan's 8).
        self.upsampling_factor = self.spatial_compression
        self.causal = True

    @property
    def _param(self):
        return next(self.vae.parameters())

    def encode(self, videos, device, tiled=False, **_ignored):
        if tiled:
            self.vae.enable_tiling()
        else:
            self.vae.disable_tiling()
        p = self._param
        out = []
        for video in videos:  # (3, T, H, W)
            x = video.unsqueeze(0).to(device=p.device, dtype=p.dtype)
            z = self.vae.encode(x).latent_dist.mode()  # deterministic for caching
            out.append(z.squeeze(0))
        return torch.stack(out)

    def decode(self, hidden_states, device, tiled=False, **_ignored):
        if tiled:
            self.vae.enable_tiling()
        else:
            self.vae.disable_tiling()
        p = self._param
        out = []
        for z in hidden_states:  # (z_dim, Tl, Hl, Wl)
            zz = z.unsqueeze(0).to(device=p.device, dtype=p.dtype)
            video = self.vae.decode(zz).sample.clamp_(-1, 1)
            out.append(video.squeeze(0))
        return torch.stack(out)


def _load_vae(path: Optional[str], *, device, dtype, vae_type: str = "wan"):
    """Load the video VAE.

    ``vae_type="wan"`` (default): Wan2.1 causal VAE from a local ``.pth`` file.
    SANA-Video 2B 480p reuses it; the ckpt is bundled in the HF asset
    (``<bundle>/vae/Wan2.1_VAE.pth``). We deliberately avoid ``model_pool`` /
    HuggingFace Hub here — CI / offline machines can't reach the Hub and the
    asset already has the file on disk.

    ``vae_type="ltx2"``: LTX2 causal VAE (128ch, 8x temporal, 32x spatial) from a
    diffusers *directory* via ``AutoencoderKLLTX2Video.from_pretrained``, wrapped
    in ``_LTX2VAEShim`` to expose the Wan encode/decode contract. This is the VAE
    the SANA-WM streaming world model (``init_dit_from``) was trained with.

    Returns ``None`` if path is missing/unset, so smoke tests that don't
    need pixel-space encode/decode can still construct the pipeline.
    """
    if vae_type == "ltx2":
        if not path or not os.path.isdir(path):
            logger.warning(
                "build_sana_pipeline: LTX2 VAE not loaded (path=%s is not a "
                "directory). decode_video and preprocess_input(frames=...) will "
                "be unavailable until a diffusers VAE directory is provided.",
                path,
            )
            return None
        from diffusers import AutoencoderKLLTX2Video

        # fp32 for numerical stability: the VAE is frozen and used to (re)compute
        # cached latents the frozen DiT consumes, so encode precision matters and
        # it is not in the training compute graph. The adapter casts the returned
        # latents to the pipe dtype.
        vae = AutoencoderKLLTX2Video.from_pretrained(path, torch_dtype=torch.float32)
        vae = vae.to(device=device or "cuda").eval()
        vae.requires_grad_(False)
        shim = _LTX2VAEShim(vae)
        logger.info(
            "Loaded LTX2 causal VAE from %s (z=%d, temporal=%dx, spatial=%dx)",
            path,
            shim.z_dim,
            shim.temporal_compression,
            shim.spatial_compression,
        )
        return shim

    if not path or not os.path.isfile(path):
        logger.warning(
            "build_sana_pipeline: VAE not loaded (path=%s missing). decode_video "
            "and preprocess_input(frames=...) will be unavailable until a path "
            "to Wan2.1_VAE.pth is provided.",
            path,
        )
        return None

    from sana_wam.model.video_backbone.wan.vae import WanVideoVAE

    # weights_only=True: torch>=2.6 defaults to False; pin the safe path
    # explicitly so a malformed ckpt can't smuggle arbitrary objects.
    state = torch.load(path, map_location="cpu", weights_only=True)
    vae = WanVideoVAE(z_dim=16)
    converter = getattr(WanVideoVAE, "state_dict_converter", None)
    if converter is not None:
        state = converter().from_civitai(state)
    result = vae.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        logger.warning(
            "Wan2.1 VAE load partial: %d missing, %d unexpected. "
            "Sample missing=%s sample unexpected=%s",
            len(result.missing_keys),
            len(result.unexpected_keys),
            result.missing_keys[:3],
            result.unexpected_keys[:3],
        )
    vae = vae.to(device=device or "cuda", dtype=dtype).eval()
    vae.requires_grad_(False)
    logger.info("Loaded Wan2.1 VAE from %s", path)
    return vae


def _load_text_encoder(name: Optional[str], *, device, dtype, ckpt_dir: Optional[str] = None):
    """Optional Gemma-2-2B-it loader for deploy. Smoke uses ``pre_encoded_text``.

    Gemma is gated on Hugging Face; CI usually doesn't have the token. We
    fail-soft (return ``(None, None)`` + WARNING) when the model isn't
    available locally so the pipeline still constructs. Callers that hand
    in ``pre_encoded_text`` via ``preprocess_input`` never touch this path
    — see ``SanaVideoBackbone.preprocess_input``.

    Deploy self-contained fallback: when ``ckpt_dir`` is set (we are inside
    ``load_from_checkpoint_dir``) and ``from_pretrained`` fails because the
    weights file is missing but ``config.json`` is present, build a meta-
    initialised shell from the config. The trained text encoder weights
    live in the sana-wam safetensors (text_encoder is ``add_module``-
    registered in ``SanaVideoBackbone.__init__``), so the subsequent
    ``architecture.load_checkpoint(...)`` fills the shell via
    ``load_state_dict(..., assign=True)`` (see ``base.py`` meta-tensor
    path). ``set_dtype_device`` then moves it onto the target device.
    """
    if not name:
        return None, None
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError:
        logger.warning(
            "build_sana_pipeline: transformers not installed; text encoder "
            "loading skipped (name=%s).",
            name,
        )
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
    except Exception as e:
        logger.warning(
            "build_sana_pipeline: failed to load text encoder tokenizer %r "
            "(%s: %s).",
            name,
            type(e).__name__,
            e,
        )
        return None, None

    used_meta_shell = False
    try:
        model = AutoModel.from_pretrained(name, torch_dtype=dtype)
    except Exception as load_err:
        if ckpt_dir is None:
            logger.warning(
                "build_sana_pipeline: failed to load text encoder %r (%s: %s). "
                "Pass pre_encoded_text via preprocess_input or wire a local "
                "Gemma path before deploy.",
                name,
                type(load_err).__name__,
                load_err,
            )
            return None, None
        try:
            from accelerate import init_empty_weights

            config = AutoConfig.from_pretrained(name)
            with init_empty_weights():
                model = AutoModel.from_config(config, torch_dtype=dtype)
        except Exception as shell_err:
            logger.warning(
                "build_sana_pipeline: failed to build deploy self-contained "
                "text encoder shell for %r (load_err=%s: %s; shell_err=%s: %s).",
                name,
                type(load_err).__name__,
                load_err,
                type(shell_err).__name__,
                shell_err,
            )
            return None, None
        used_meta_shell = True
        logger.info(
            "SANA deploy self-contained: text encoder %s weights missing; "
            "built empty (meta) shell from config.json (weights load from %s).",
            name,
            ckpt_dir,
        )

    model = model.eval()
    model.requires_grad_(False)
    if not used_meta_shell:
        model = model.to(device=device or "cuda")
        logger.info("Loaded text encoder %s (dtype=%s)", name, dtype)
    return model, tokenizer
