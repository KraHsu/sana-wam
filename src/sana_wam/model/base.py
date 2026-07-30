"""Abstract base class for the WAM (World-Action Model) architecture.

The shipped architecture is **Dual-System** (``framework=dual_system``): a SANA
video DiT plus a separate ActionDiT, coupled through MMDiT-style mixed attention
at every layer (the ``joint_self_attn`` variant, driven by
:class:`MoTJointDriver`). The block-autoregressive variant
(:class:`DualSystemARArchitecture`) extends that with diffusion-forcing.

The base composes a ``video_backbone`` and an ``action_backbone``; each concrete
architecture implements its own ``forward()``.
"""

import copy
import functools
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from sana_wam.model.compile_options import compile_mode


def _wrap_single_forward(module: nn.Module) -> None:
    """Wrap a single module's ``forward`` in ``torch.no_grad``. Idempotent."""
    if getattr(module, "_sana_wam_no_grad_wrapped", False):
        return
    original_forward = module.forward

    @functools.wraps(original_forward)
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            return original_forward(*args, **kwargs)

    module.forward = wrapped
    module._sana_wam_no_grad_wrapped = True


def _wrap_forward_in_no_grad(module: nn.Module) -> None:
    """Wrap ``forward`` of ``module`` AND every submodule in its subtree in ``torch.no_grad``.

    Recursion matters because callers commonly bypass the root forward and call a
    nested submodule directly (e.g. an ``extract_features`` that calls an inner
    model and skips an outer head) — wrapping only the root ``forward`` would
    leave that path grad-tracking. Recursively wrapping every descendant makes
    the semantic complete: any entry point into the frozen subtree is in
    ``no_grad``.

    Idempotent — a marker attribute on each module prevents double-wrapping if
    ``freeze_modules`` runs more than once. ``nn.Module.modules()`` deduplicates
    via its internal memo, so cyclic registrations (e.g. test fakes with
    ``self.model = self``) are visited once.

    **Subtree-level semantic, not per-parameter**: a trainable child under a frozen
    parent will NOT receive gradients, because every descendant ``forward`` is
    wrapped in ``no_grad``. For partial-freeze setups (e.g. LoRA on a frozen
    base), do NOT pass the parent's dotted path to ``freeze_modules``; pass the
    specific leaves you want frozen instead.
    """
    for sub in module.modules():
        _wrap_single_forward(sub)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from sana_wam.model.action_backbone.backbone import ActionBackbone
    from sana_wam.model.video_backbone.adapter import VideoBackbone


@dataclass
class ActionState:
    """Mutable state container used by the joint self-attention path.

    Only ``DualSystemSelfAttnArchitecture`` needs this — its action stream is
    threaded through ``MoTJointDriver``, which mutates the payload across
    layers.

    Fields:
        action_latents: (B, T_action, action_dim) noisy actions (input to forward).
        timestep: action diffusion timestep (raw shape preserved for the action
            backbone's internal use).
        payload: backbone-specific per-forward state (typically
            ``ActionDiTState``).
    """

    action_latents: Optional[Tensor] = None
    timestep: Optional[Tensor] = None
    payload: Optional[Any] = None


class BaseWAMArchitecture(ABC, nn.Module):
    """Base class for WAM architecture variants.

    Composes a ``video_backbone`` and an ``action_backbone`` plus optional
    extra backbones. Subclasses instantiate the appropriate ActionBackbone
    subclass in ``__init__`` and own the complete ``forward()`` control flow.

    Args:
        cfg: Architecture-specific configuration (OmegaConf DictConfig or dict).
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.video_backbone: Optional["VideoBackbone"] = None
        self.action_backbone: Optional["ActionBackbone"] = None
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        # Forward-time training runtime flags. Trainer calls
        # ``set_training_runtime`` once during construction so ``prepare_inputs``
        # can read these without the trainer having to thread them through.
        self._use_gradient_checkpointing = False
        self._use_gradient_checkpointing_offload = False
        self._max_timestep_boundary = 1.0
        self._min_timestep_boundary = 0.0

        # Optional action normalizer for deployment. The deploy engine uses it
        # to return real-scale actions; deploy-side proprio preprocessing uses
        # it to normalize raw robot state into the model's training space.
        self.action_normalizer = None

        if cfg is not None:
            self._init_video_backbone(cfg)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _init_video_backbone(self, cfg):
        """Build the SANA video backbone (single-backbone project — no registry).

        Two source shapes in ``cfg.video_backbone``:
        - ``_source`` present → deploy path (load from saved components dir/dict).
        - otherwise → training path (read ``video_backbone.*`` from the full cfg).
        """
        from sana_wam.model.video_backbone.sana import SanaVideoBackbone

        vb_cfg = self._cfg_get(cfg, "video_backbone", None)
        if vb_cfg is None:
            return
        source = self._cfg_get(vb_cfg, "_source", None)
        build_kwargs = {}
        build_device = self._cfg_get(vb_cfg, "_device", None)
        checkpoint_dir = self._cfg_get(vb_cfg, "_ckpt_dir", None)
        if build_device is not None:
            build_kwargs["device"] = str(build_device)
        if checkpoint_dir is not None:
            build_kwargs["ckpt_dir"] = str(checkpoint_dir)
        self.video_backbone = SanaVideoBackbone.from_pretrained(
            source if source is not None else cfg,
            **build_kwargs,
        )

    def _resolve_video_dim(self, cfg) -> int:
        """Resolve video_dim from config or video_backbone; raise if neither provides it."""
        dim = (
            int(cfg.get("video_dim", 0))
            if isinstance(cfg, dict)
            else int(getattr(cfg, "video_dim", 0))
        )
        if dim == 0 and self.video_backbone is not None:
            dim = self.video_backbone.dim
        if not dim:
            raise ValueError(
                "video_dim must be specified in config or inferred from video_backbone"
            )
        return dim

    def _resolve_text_dim(self, cfg, *, default: int = 4096) -> int:
        """Resolve the action-side context (text) dim.

        Priority: explicit ``cfg.text_dim`` → ``video_backbone.context_dim`` →
        ``default``. The backbone reports its text-encoder context dim via
        :attr:`VideoBackbone.context_dim`; configs that omit ``text_dim`` fall
        back to the default.
        """
        raw = self._cfg_get(cfg, "text_dim", None)
        if raw not in (None, 0):
            return int(raw)
        if self.video_backbone is not None:
            ctx_dim = getattr(self.video_backbone, "context_dim", None)
            if ctx_dim:
                return int(ctx_dim)
        return int(default)

    @property
    def backbones(self) -> dict[str, nn.Module]:
        """All backbone modules owned by this architecture.

        Subclasses with additional backbones should override this to
        include them. The returned dict
        is used by ``init_training_schedulers``, ``set_dtype_device``,
        ``move_frozen_to_device``, and ``get_component_specs`` to iterate
        over all backbones generically.
        """
        result = {}
        if self.video_backbone is not None:
            result["video_backbone"] = self.video_backbone
        if self.action_backbone is not None:
            result["action_backbone"] = self.action_backbone
        return result

    # --- Action-side properties (delegate to action_backbone) ---

    @property
    def action_scheduler(self):
        """Flow-matching scheduler for the action stream (owned by action_backbone)."""
        if self.action_backbone is None:
            raise RuntimeError("action_backbone is not initialized")
        return self.action_backbone.scheduler

    @property
    def video_scheduler(self):
        """Flow-matching scheduler for the video stream (owned by video_backbone)."""
        if self.video_backbone is None:
            raise RuntimeError("video_backbone is not initialized")
        return self.video_backbone.scheduler

    @property
    def action_dim(self) -> int:
        return (
            self.action_backbone.action_dim if self.action_backbone is not None else 0
        )

    @property
    def bridge_layers(self) -> tuple:
        return (
            getattr(self.action_backbone, "bridge_layers", ())
            if self.action_backbone is not None
            else ()
        )

    @property
    def expert_layers(self) -> tuple:
        return (
            getattr(self.action_backbone, "expert_layers", ())
            if self.action_backbone is not None
            else ()
        )

    @property
    def trainable_action_module(self) -> Optional[nn.Module]:
        """The nn.Module whose parameters are trained as the action model."""
        return self.action_backbone

    @property
    def uses_proprioception(self) -> bool:
        return bool(getattr(self, "_use_proprioception_context", False)) or (
            self.action_backbone is not None
            and self.action_backbone.uses_proprioception
        )

    def _init_proprio_context(self, cfg, *, text_dim: int = 4096) -> None:
        """Initialize FastWAM-style proprio-as-context conditioning."""
        enabled = bool(self._cfg_get(cfg, "use_proprioception", False))
        self._use_proprioception_context = enabled
        self.proprio_encoder: Optional[nn.Module] = None
        self.proprio_dim = 0
        self.context_dim = int(text_dim)
        if not enabled:
            return
        state_dim = int(self._cfg_get(cfg, "state_dim", 0) or 0)
        if state_dim <= 0:
            raise ValueError(
                "use_proprioception=True requires explicit state_dim for context-token proprio."
            )
        self.proprio_dim = state_dim
        self.proprio_encoder = nn.Linear(state_dim, self.context_dim)

    def _append_proprio_context_token(
        self, pipeline_inputs: dict, proprio_state: Optional[Tensor]
    ) -> dict:
        """Append one proprio token to raw text context and extend context_mask."""
        if not bool(getattr(self, "_use_proprioception_context", False)):
            return pipeline_inputs
        if self.proprio_encoder is None:
            raise RuntimeError(
                "proprio context is enabled but proprio_encoder is not initialized."
            )
        if proprio_state is None:
            raise ValueError(
                "use_proprioception=True requires `proprio_state` from sample['proprio'] or obs['state']."
            )
        if proprio_state.ndim == 1:
            proprio_state = proprio_state.unsqueeze(0)
        elif proprio_state.ndim == 3 and proprio_state.shape[1] == 1:
            proprio_state = proprio_state[:, 0, :]
        if proprio_state.ndim != 2:
            raise ValueError(
                f"proprio_state must be [B, D] or [B, 1, D], got shape {tuple(proprio_state.shape)}"
            )
        if proprio_state.shape[1] != self.proprio_dim:
            raise ValueError(
                f"proprio_state last dim must be {self.proprio_dim}, got {proprio_state.shape[1]}"
            )

        context = pipeline_inputs["context"]
        if context.shape[0] != proprio_state.shape[0]:
            if proprio_state.shape[0] == 1 and context.shape[0] > 1:
                proprio_state = proprio_state.expand(context.shape[0], -1)
            else:
                raise ValueError(
                    f"Batch mismatch between context and proprio_state: {context.shape[0]} vs {proprio_state.shape[0]}"
                )
        # Proprio modality-augmentation (training only) — forces the action to learn
        # a vision-grounded pathway instead of leaning on the near-sufficient proprio
        # state (decodability gate: proprio ~sufficient in-distribution; OOD-correction
        # probe: vision is the only grounded signal once proprio drifts). noise perturbs
        # the raw state (matches the OOD drift); dropout zeros+masks the whole token.
        noise_std = float(getattr(self, "_proprio_noise_std", 0.0) or 0.0)
        if self.training and noise_std > 0:
            proprio_state = proprio_state + torch.randn_like(proprio_state) * noise_std
        proprio_token = (
            self.proprio_encoder(
                proprio_state.to(
                    device=context.device, dtype=self.proprio_encoder.weight.dtype
                )
            )
            .to(dtype=context.dtype)
            .unsqueeze(1)
        )

        context_mask = pipeline_inputs.get("context_mask")
        if context_mask is None:
            seq_lens = pipeline_inputs.get("seq_lens")
            if seq_lens is not None:
                seq_lens = seq_lens.to(device=context.device)
                positions = torch.arange(
                    context.shape[1], device=context.device
                ).unsqueeze(0)
                context_mask = positions < seq_lens.unsqueeze(1)
            else:
                context_mask = torch.ones(
                    (context.shape[0], context.shape[1]),
                    dtype=torch.bool,
                    device=context.device,
                )
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)

        proprio_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        drop_p = float(getattr(self, "_proprio_dropout", 0.0) or 0.0)
        if self.training and drop_p > 0:
            # Per-row Bernoulli drop: zero the encoded token AND mask it out, so dropped
            # rows see NO proprio (a clean "absent" signal, not a degenerate min-pose).
            keep = (
                torch.rand(proprio_token.shape[0], device=proprio_token.device)
                >= drop_p
            )
            proprio_token = proprio_token * keep.view(-1, 1, 1).to(proprio_token.dtype)
            proprio_mask = proprio_mask & keep.view(-1, 1)
        updated = dict(pipeline_inputs)
        updated["context"] = torch.cat([context, proprio_token], dim=1)
        updated["context_mask"] = torch.cat([context_mask, proprio_mask], dim=1)
        # The appended proprio token can sit after padded text tokens, so the
        # resulting valid tokens are not necessarily a contiguous prefix.
        # Keep the original text seq_lens and make context_mask authoritative.
        return updated

    @property
    def action_mean(self) -> Tensor:
        if self.action_backbone is not None:
            return self.action_backbone.action_mean
        return torch.zeros(self.action_dim)

    @property
    def action_std(self) -> Tensor:
        if self.action_backbone is not None:
            return self.action_backbone.action_std
        return torch.ones(self.action_dim)

    # --- Device / dtype (top-level authority) ---

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Dispatch to each backbone — they own their own dtype/device handling."""
        self._dtype = dtype
        self._device = device
        proprio_encoder = getattr(self, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.to(dtype=dtype, device=device)
        for bb in self.backbones.values():
            bb.set_dtype_device(dtype, device)

    def attach_action_normalizer(self, normalizer) -> None:
        """Attach (or clear) an action normalizer used by ``generate``.

        Deployment paths build the same normalizer used by training from
        ``action_stats.npy``. ``generate`` uses it to return real-scale actions,
        while server-side proprio preprocessing uses it to normalize raw robot
        state into the model's training space. Pass ``None`` to clear.
        """
        self.action_normalizer = normalizer

    def normalize_deploy_proprio(self, proprio_state):
        """Normalize raw deploy proprio with the training action normalizer."""
        normalizer = getattr(self, "action_normalizer", None)
        if normalizer is None or proprio_state is None:
            return proprio_state

        import numpy as np
        import torch

        was_tensor = isinstance(proprio_state, torch.Tensor)
        device = proprio_state.device if was_tensor else None
        dtype = (
            proprio_state.dtype
            if was_tensor and proprio_state.is_floating_point()
            else None
        )
        arr = (
            proprio_state.detach().cpu().numpy()
            if was_tensor
            else np.asarray(proprio_state, dtype=np.float32)
        )
        norm = normalizer.normalize(arr.astype(np.float32, copy=False))
        if was_tensor:
            return torch.from_numpy(norm).to(
                device=device, dtype=dtype or torch.float32
            )
        return norm

    # --- Checkpoint save / load ---

    def save_checkpoint(self, path: str) -> None:
        """Save architecture state to safetensors."""
        from safetensors.torch import save_file

        state_dict = self.state_dict()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(state_dict, path)

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        allow_missing_patterns: tuple[str, ...] = (),
    ) -> None:
        """Load architecture state from a safetensors checkpoint.

        Meta-device sub-modules (self-contained deploy: empty shells built via
        ``from_empty`` / ``init_empty_weights``) need ``load_state_dict(...,
        assign=True)`` — the default in-place copy is a silent no-op against
        meta tensors and leaves the shells unpopulated. ``assign=True`` rebinds
        the parameter slot to the safetensors tensor instead. We only flip the
        flag when meta params actually exist so the training-resume path
        (real-device params, in-place copy preserves identity) is unchanged.

        ``allow_missing_patterns`` lists substrings of parameter names that are
        permitted to be absent from the checkpoint and kept at their freshly
        built (default-init) values instead of failing a strict load. This is
        the forward-compat hatch for params added to a *shared* module after a
        checkpoint was written — e.g. ``cross_attn.norm_kv.*`` was added to
        ``BridgeCrossAttention`` for the frozen linear-attn SANA backbone, but
        the same block is reused by older GDN cross-attn / GDN-AR checkpoints
        that predate it. Those keys default to a LayerNorm(weight=1, bias=0); on
        the O(1) GDN features this is a behavior change vs the original raw-KV
        training, so it is logged loudly. Unexpected keys are never tolerated,
        and any *other* missing key still fails — strictness is preserved
        everywhere except the named, safe-to-default params.
        """
        from safetensors.torch import load_file

        state_dict = load_file(path)
        has_meta = any(p.device.type == "meta" for p in self.parameters())
        missing, unexpected = self.load_state_dict(
            state_dict, strict=False, assign=has_meta
        )
        if allow_missing_patterns:
            tolerated = [
                k for k in missing if any(pat in k for pat in allow_missing_patterns)
            ]
            if tolerated:
                logger.warning(
                    "load_checkpoint: %d param(s) absent from checkpoint kept at default init "
                    "(matched allow_missing_patterns=%s): %s%s. For an older checkpoint this "
                    "alters that module's behavior vs its original training.",
                    len(tolerated),
                    allow_missing_patterns,
                    tolerated[:8],
                    " ..." if len(tolerated) > 8 else "",
                )
            missing = [k for k in missing if k not in tolerated]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Strict load failed: missing={missing}, unexpected={unexpected}"
            )

    # --- Training: module management ---

    def init_training_schedulers(self, num_timesteps: int = 1000) -> None:
        """Initialize all backbone schedulers for training.

        Single source of truth for the video α-shift:
        ``self.video_backbone.shift_video``. The same property is read by
        the deploy inference path at inference time, so
        the discrete training sigma buffer and the inference denoising
        trajectory are guaranteed to be sampled from the same shifted
        schedule — train/inference cannot drift regardless of which yaml
        file is loaded.

        Action backbone is intentionally NOT split: its scheduler always
        falls back to the template default, matching the
        Reconstruction-or-Semantics paper recipe (arXiv:2605.06388) which
        applies dim-dependent shift only on non-VAE video encoders.

        ``shift_video=None`` (the default for backbones without an explicit
        cfg override) yields bit-identical pre-PR behavior: each scheduler
        falls back to its template default (Wan = 5.0).
        """
        # ``getattr`` (rather than direct attribute access) so test doubles
        # / mocks that extend bare ``nn.Module`` instead of the
        # :class:`VideoBackbone` ABC still work — they simply don't carry a
        # ``shift_video`` attribute and we fall back to the scheduler's
        # template default, matching the production no-override path.
        video_shift = (
            getattr(self.video_backbone, "shift_video", None)
            if self.video_backbone is not None
            else None
        )
        action_loss_weighting = (
            getattr(self.action_backbone, "loss_weighting", None)
            if self.action_backbone is not None
            else None
        )
        for name, bb in self.backbones.items():
            if not hasattr(bb, "scheduler"):
                continue
            kwargs = {"training": True}
            if name == "video_backbone" and video_shift is not None:
                kwargs["shift"] = float(video_shift)
            if name == "action_backbone" and action_loss_weighting is not None:
                kwargs["loss_weighting"] = str(action_loss_weighting)
            bb.scheduler.set_timesteps(num_timesteps, **kwargs)

    def freeze_modules(self, names: list[str]) -> list[str]:
        """Freeze named sub-modules by dotted path. Returns actually frozen names.

        Single-point freeze API. Two effects per frozen submodule:

        1. ``module.requires_grad_(False)`` — optimizer cannot update its params.
        2. ``module.forward`` is wrapped in ``torch.no_grad`` so the frozen
           subtree never saves activations for backward. This is the full
           semantic of "freeze" — neither the trainer nor any backbone needs to
           inspect freeze status separately.

        For text_encoder / vae, which are already called under the
        ``@torch.no_grad()`` ``prepare_inputs`` decorator, the wrapper is a
        no-op (nested ``no_grad``). For modules called inside the training
        forward graph, the wrapper is what actually saves activation memory.

        Uses ``nn.Module.get_submodule()`` so dotted paths like
        ``video_backbone._pipe.text_encoder`` work naturally; unknown names
        are silently skipped, so a freeze list mentioning modules absent on a
        given architecture (e.g. ``vlm_backbone.vlm_model`` on dual_system)
        is harmless.
        """
        frozen = []
        for name in names:
            try:
                module = self.get_submodule(name)
            except (AttributeError, KeyError):
                module = None
            if module is not None:
                module.requires_grad_(False)
                # Set eval mode on the frozen subtree. Use modules() instead
                # of .eval() to avoid infinite recursion when a submodule has
                # self-referential aliases (e.g. HF model.model = self).
                for sub in module.modules():
                    sub.training = False
                _wrap_forward_in_no_grad(module)
                frozen.append(name)
        return frozen

    def get_trainable_modules(
        self, freeze_list: list[str] = ()
    ) -> dict[str, nn.Module]:
        """Return top-level trainable sub-modules.

        Walks ``self.named_children()`` and returns modules that have at
        least one parameter with ``requires_grad=True``, excluding those
        in *freeze_list*. Used by ``optimizer_groups.build_trainable_parameters``
        to source the param groups for the optimizer.
        """
        result = {}
        freeze_set = set(freeze_list)
        for name, mod in self.named_children():
            if name in freeze_set:
                continue
            if any(p.requires_grad for p in mod.parameters()):
                result[name] = mod
        return result

    def move_frozen_to_device(
        self, device: torch.device, names: tuple[str, ...] = ("text_encoder", "vae")
    ) -> None:
        """Move named frozen modules to device.

        Searches via ``get_submodule`` on self first, then on each backbone.
        """
        for name in names:
            mod = None
            try:
                mod = self.get_submodule(name)
            except (AttributeError, KeyError):
                pass
            if mod is None:
                for bb in self.backbones.values():
                    found = None
                    try:
                        found = bb.get_submodule(name)
                    except (AttributeError, KeyError):
                        found = None
                    if found is not None:
                        mod = found
                        break
            if mod is not None:
                mod.to(device=device)

    def get_component_specs(self, model_path: str) -> Optional[dict]:
        """Get component specs from all backbones for self-contained checkpoint config."""
        for bb in self.backbones.values():
            if hasattr(bb, "get_component_specs"):
                specs = bb.get_component_specs(model_path)
                if specs is not None:
                    return specs
        return None

    def copy_deploy_artifacts(self, output_dir: str, cfg) -> None:
        """Delegate to each backbone so deploy-time artifacts land in ``output_dir``.

        Trainer calls this once per checkpoint save (after ``save_config``).
        Backbones with no external artifacts can leave the default no-op.
        """
        for bb in self.backbones.values():
            if hasattr(bb, "copy_deploy_artifacts"):
                bb.copy_deploy_artifacts(output_dir, cfg)

    # --- Training: preprocessing ---

    @torch.no_grad()
    def preprocess(self, **kwargs) -> dict:
        """Encode raw frames/text into latents + context for training.

        Delegates to ``video_backbone.preprocess_input()``. External code
        (trainer) should call this instead of touching video_backbone directly.
        """
        return self.video_backbone.preprocess_input(**kwargs)

    def set_training_runtime(
        self,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,
    ) -> None:
        """Set forward-time training flags consumed by ``prepare_inputs``.

        Trainers call this once during construction. Keeping these on the
        architecture keeps ``prepare_inputs(batch)`` self-contained — the
        trainer no longer needs to thread these flags through every loss call.
        """
        self._use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._use_gradient_checkpointing_offload = bool(
            use_gradient_checkpointing_offload
        )
        self._max_timestep_boundary = float(max_timestep_boundary)
        self._min_timestep_boundary = float(min_timestep_boundary)

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        """Aggregate a list of dataset samples into a batched inputs dict.

        Absorbs the per-sample field collection used by the training forward
        step. The returned dict is designed to be
        unpacked directly into ``compute_loss`` via ``**inputs``.

        Args:
            batch: List of dataset samples (each a dict). A single dict is
                accepted as well and treated as a one-sample batch.

        Returns:
            Dict with all preprocessed video latents, text embeddings, action
            tensors, masks, and forward-time flags ready for ``compute_loss``.
        """
        from sana_wam.dataloader.transforms.pipeline import (
            FirstFrameConditioningTransform,
        )
        from sana_wam.utils import downsample_video_mask_to_latent

        if isinstance(batch, dict):
            batch = [batch]

        if not hasattr(self, "_pipeline_transform_instance"):
            self._pipeline_transform_instance = FirstFrameConditioningTransform()
        samples = [self._pipeline_transform_instance.apply(s) for s in batch]

        _dtype = self.dtype
        _device = self.device

        all_frames: list = []
        all_prompts: list = []
        all_ref_images: list = []
        all_pre_encoded_text: list = []
        all_pre_encoded_latents: list = []
        all_actions: list = []
        all_proprios: list = []
        all_proprio_seqs: list = []
        all_action_masks: list = []
        all_video_masks: list = []
        all_clean_prefix: list = []
        all_clean_prefix_actions: list = []
        all_phase6_plan_rows: list = []

        for sample in samples:
            all_frames.append(sample["video"])
            all_prompts.append(sample["prompt"])
            all_ref_images.append(sample.get("first_frame_image"))
            all_pre_encoded_text.append(sample.get("pre_encoded_text"))
            all_pre_encoded_latents.append(sample.get("pre_encoded_latents"))

            action = sample.get("action")
            if action is not None:
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action)
                action = action.to(dtype=_dtype, device=_device).unsqueeze(0)
            all_actions.append(action)

            proprio = sample.get("proprio")
            if self.uses_proprioception:
                if proprio is None:
                    raise ValueError(
                        "use_proprioception=True requires sample['proprio']; action[0] fallback is disabled."
                    )
                if isinstance(proprio, np.ndarray):
                    proprio = torch.from_numpy(proprio)
                proprio = proprio.to(dtype=_dtype, device=_device)
                if proprio.ndim == 1:
                    pass
                elif proprio.ndim == 2 and proprio.shape[0] == 1:
                    proprio = proprio[0]
                else:
                    raise ValueError(
                        f"sample['proprio'] must be [D] or [1, D], got shape {tuple(proprio.shape)}"
                    )
            all_proprios.append(proprio)

            pseq = sample.get(
                "proprio_seq"
            )  # F4: full state seq (T_state, D), optional
            if isinstance(pseq, np.ndarray):
                pseq = torch.from_numpy(pseq)
            if pseq is not None:
                pseq = pseq.to(dtype=_dtype, device=_device)
            all_proprio_seqs.append(pseq)

            amask = sample.get("action_mask", None)
            vmask = sample.get("video_mask", None)
            if isinstance(amask, np.ndarray):
                amask = torch.from_numpy(amask)
            if isinstance(vmask, np.ndarray):
                vmask = torch.from_numpy(vmask)
            all_action_masks.append(amask)
            all_video_masks.append(vmask)
            all_clean_prefix.append(int(sample.get("num_clean_prefix_latent", 0) or 0))
            all_clean_prefix_actions.append(
                int(sample.get("num_clean_prefix_actions", 0) or 0)
            )
            all_phase6_plan_rows.append(sample.get("phase6_plan_row"))

        ref_flags = [r is not None for r in all_ref_images]
        if any(ref_flags) and not all(ref_flags):
            raise ValueError(
                "Mixed reference images in batch: all samples must be consistent."
            )

        # Optional per-sample pre-encoded text embedding (cached offline).
        # Backbones that don't consume it silently drop the kwarg via ``**kw``.
        # All-or-nothing per batch; uniform L required for fixed-shape stacking
        # — padded variant deferred.
        pre_text_flags = [t is not None for t in all_pre_encoded_text]
        preprocess_extra: dict = {}
        if any(pre_text_flags):
            if not all(pre_text_flags):
                raise ValueError(
                    "Mixed pre_encoded_text in batch: every sample must carry the "
                    "field, or none. Check the dataloader cache wiring."
                )
            tensors: list = []
            for t in all_pre_encoded_text:
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t)
                if t.ndim == 3 and t.shape[0] == 1:
                    t = t[0]
                if t.ndim != 2:
                    raise ValueError(
                        f"pre_encoded_text must be (L, D) or (1, L, D); got {tuple(t.shape)}"
                    )
                tensors.append(t)
            lens = {t.shape[0] for t in tensors}
            if len(lens) > 1:
                raise ValueError(
                    f"Inconsistent sequence length across pre_encoded_text batch: "
                    f"{sorted(lens)}. The AR path requires uniform L within a batch; "
                    f"padded variant is deferred."
                )
            preprocess_extra["pre_encoded_text"] = torch.stack(
                [t.to(dtype=_dtype, device=_device) for t in tensors], dim=0
            )

        # Optional per-sample pre-encoded VAE latents (cached offline), mirroring
        # upstream Sana-wm's ``load_vae_feat`` / ``SanaWMZipLatentDataset``. When
        # present they skip the live VAE encode in ``preprocess_input`` (which is
        # the per-step bottleneck). All-or-nothing per batch; uniform latent shape
        # required for fixed-shape stacking.
        pre_lat_flags = [t is not None for t in all_pre_encoded_latents]
        if any(pre_lat_flags):
            if not all(pre_lat_flags):
                raise ValueError(
                    "Mixed pre_encoded_latents in batch: every sample must carry the "
                    "field, or none. Check the dataloader cache wiring."
                )
            lat_tensors: list = []
            for t in all_pre_encoded_latents:
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t)
                if t.dim() == 5 and t.shape[0] == 1:
                    t = t[0]
                if t.dim() != 4:
                    raise ValueError(
                        f"pre_encoded_latents must be (C, T, H, W) or (1, C, T, H, W); got {tuple(t.shape)}"
                    )
                lat_tensors.append(t)
            shapes = {tuple(t.shape) for t in lat_tensors}
            if len(shapes) > 1:
                raise ValueError(
                    f"Inconsistent pre_encoded_latents shapes across batch: {sorted(shapes)}. "
                    f"All cached latents in a batch must share (C, T, H, W)."
                )
            preprocess_extra["input_latents"] = torch.stack(
                [t.to(dtype=_dtype, device=_device) for t in lat_tensors], dim=0
            )

        preprocessed = self.preprocess(
            frames=all_frames,
            text=all_prompts,
            ref_images=all_ref_images if ref_flags[0] else None,
            **preprocess_extra,
        )

        action_data = (
            torch.cat(all_actions, dim=0) if all_actions[0] is not None else None
        )

        inputs = {
            **preprocessed,
            "latents": None,
            "cfg_scale": 1,
            "cfg_merge": False,
            "tiled": False,
            "use_gradient_checkpointing": self._use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self._use_gradient_checkpointing_offload,
            "max_timestep_boundary": self._max_timestep_boundary,
            "min_timestep_boundary": self._min_timestep_boundary,
            "actions": action_data,
        }

        phase6_row_flags = [row is not None for row in all_phase6_plan_rows]
        if any(phase6_row_flags) and not all(phase6_row_flags):
            raise ValueError(
                "Mixed Phase-6 plan metadata in batch: every sample must carry "
                "phase6_plan_row, or none may carry it."
            )
        if all(phase6_row_flags):
            if any(not isinstance(row, dict) for row in all_phase6_plan_rows):
                raise TypeError("phase6_plan_row must be a plain dict")
            # Keep the exact strings, hashes, and domain seeds available to both
            # the trainer's step check and Phase-6 loss terms.
            inputs["phase6_plan_rows"] = tuple(
                copy.deepcopy(row) for row in all_phase6_plan_rows
            )

        if self.uses_proprioception:
            inputs["proprio_state"] = torch.stack(all_proprios, dim=0).contiguous()
            if all_proprio_seqs and all(p is not None for p in all_proprio_seqs):
                inputs["proprio_seq"] = torch.stack(
                    all_proprio_seqs, dim=0
                ).contiguous()

        action_mask_flags = [mask is not None for mask in all_action_masks]
        if any(action_mask_flags) and not all(action_mask_flags):
            raise ValueError(
                "Mixed action masks in batch: every sample must carry action_mask, "
                "or none may carry it."
            )
        if all(action_mask_flags):
            inputs["action_is_pad"] = torch.stack(
                [~m for m in all_action_masks], dim=0
            ).to(device=_device)
        if all_video_masks[0] is not None:
            # ``latent[0]`` is a clean conditioning frame (and must be excluded
            # from the loss mask) when either the batch carries
            # ``first_frame_latents`` (per-batch first-frame conditioning) or the
            # backbone's configuration always reserves ``latent[0]`` for
            # conditioning (``needs_first_frame_skip``). SANA is text-to-video:
            # neither holds, so ``latent[0]`` is a fully-noised, supervised frame.
            skip_first = (
                inputs.get("first_frame_latents") is not None
                or self.video_backbone.needs_first_frame_skip
            )
            # Pass the backbone's temporal_compression so the tail-grouping
            # divisor matches the actual latent-T produced by the VAE. See
            # VideoBackbone.temporal_compression for the source-of-truth contract.
            temporal_factor = int(self.video_backbone.temporal_compression)
            latent_masks = [
                downsample_video_mask_to_latent(
                    ~m, temporal_factor=temporal_factor, skip_first=skip_first
                )
                for m in all_video_masks
            ]
            inputs["video_is_pad"] = torch.stack(latent_masks, dim=0).to(device=_device)

        # Per-sample clean-prefix length (in latent frames) for growing-history
        # training. Consumed ONLY by compute_loss (sigma=0 + unsupervised mask on
        # the prefix); deliberately NOT routed to the backbone forward, which
        # would trip SANA's per-frame t-mod reshape path. Always present (0 in
        # the legacy fixed-window path → frame-0-only clean prefix).
        if any(p > 0 for p in all_clean_prefix):
            inputs["num_clean_prefix_frames"] = torch.tensor(
                all_clean_prefix, dtype=torch.long, device=_device
            )

        # Per-sample clean-prefix length in ACTION tokens (growing-history, Design B):
        # tokens [0, k) are the observed/executed past actions; compute_loss pins them
        # clean (conditioning) and excludes them from the action loss. Absent ⇒ 0.
        if any(p > 0 for p in all_clean_prefix_actions):
            inputs["num_clean_prefix_actions"] = torch.tensor(
                all_clean_prefix_actions, dtype=torch.long, device=_device
            )

        return inputs

    # --- ZeRO-3 external-parameter protocol ---
    #
    # The MoT driver reads several leaf ``nn.Parameter`` (e.g. ``block.modulation``,
    # ``block.wan_und_qkv``) directly inside the architecture forward, bypassing the
    # owning submodule's ``__call__``. Under DeepSpeed ZeRO-3 those leaves are
    # partitioned and the forward-pre-hook that would gather them never fires for
    # the owner. The fix is the standard external-parameter protocol: register
    # the leaves against ``self`` (the architecture) and call ``self(...)`` so the
    # architecture-level forward-pre-hook gathers them before the raw read.

    def _iter_zero3_external_params(self):
        """Yield each raw-access leaf ``nn.Parameter`` that the MoT path reads.

        Default: empty. Overridden by every architecture whose training
        forward pulls partitioned leaves outside the owner submodule's
        ``__call__`` — concretely, the MoT-driven variants:

        - ``DualSystemSelfAttnArchitecture`` — video + action ``block.modulation``
        - ``DualSystemIDMArchitecture`` — same as joint_self_attn; IDM's
          ``compute_loss`` override routes its 3-branch forward through
          ``self.__call__`` so the same protocol applies.
        - ``TriSystemJointSelfAttnArchitecture`` — also understanding
          ``block.wan_und_qkv``.

        Cross-attn and shared variants go through standard ``block.__call__``
        and don't need to override.
        """
        return ()

    def _register_zero3_externals(self) -> None:
        """Register raw-access leaves as DeepSpeed ZeRO-3 external params of ``self``.

        Required because the MoT driver reads these leaves directly, bypassing
        the owning submodule's ``__call__``. Registering them makes DeepSpeed
        gather them on ``self.__call__``'s forward-pre-hook and hold through
        backward — without this AccumulateGrad sees a size-0 leaf.

        Idempotent + no-op when deepspeed isn't importable or params lack
        ``ds_id`` (non-ZeRO-3 paths, CPU mock tests). The gate only seals after
        at least one successful register so a pre-``accelerator.prepare`` call
        (params still un-partitioned) can be retried post-prepare. Architectures
        whose iterator is empty by design (cross-attn, shared variants) seal
        immediately — they will never need to register anything.
        """
        if getattr(self, "_zero3_externals_registered", False):
            return
        leaves = list(self._iter_zero3_external_params())
        if not leaves:
            self._zero3_externals_registered = True
            return
        try:
            from deepspeed.runtime.zero import register_external_parameter
        except ImportError:
            self._zero3_externals_registered = True
            return
        registered_any = False
        for p in leaves:
            if getattr(p, "ds_id", None) is not None:
                register_external_parameter(self, p)
                registered_any = True
        if registered_any:
            self._zero3_externals_registered = True

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Apply architecture-specific deploy-time compile optimizations."""
        mode = compile_mode(compile_cfg, default="none", strict=True)
        if mode in (None, "auto", "none"):
            return
        logger.warning(
            "torch.compile mode '%s' is not implemented for %s; running eager.",
            mode,
            type(self).__name__,
        )

    @abstractmethod
    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio_state: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the joint video + action forward.

        Each concrete architecture implements its own forward end-to-end —
        runs the video DiT block loop, captures or interleaves with the
        action stream as appropriate, and returns
        ``(video_noise_pred, action_noise_pred)``. When ``noisy_actions`` is
        None (CFG nega pass / video-only generation) the action term is
        None.
        """
        ...
