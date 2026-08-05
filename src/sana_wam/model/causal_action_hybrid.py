"""CACH-A architecture seam and layout-bound action conditioner.

Stage 1 intentionally leaves model execution disabled.  The class below makes
the future architecture dispatch explicit and implements the deterministic
layout-to-condition reduction, but its constructor consumes the Stage-1
authority guard before the legacy model builder can import weights, allocate a
runtime model, or reach CUDA.

The numerical activation of this class belongs to Stage 2.  In particular,
the typed hybrid-cache codec, partial-tail kernel behavior, and the explicit
committed-action summary operator are still admission blockers; none is
silently approximated here.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from sana_wam.cach.action_conditioning import (
    reduce_end_of_bin_action_condition,
)
from sana_wam.cach.authority import CACH_VARIANT, reject_cach_model_build
from sana_wam.model.video_backbone.sana.hybrid_cache import (
    CacheScratch,
    DenoiseReadView,
)
from sana_wam.model.action_chunk_layout import (
    ChunkActionLayout,
    LayoutContractError,
    LayoutReasonCode,
    SelectedProprioBinding,
    canonical_proprio_row_sha256,
    validate_cach_bootstrap_config,
)
from sana_wam.model.gdn_ar import DualSystemGDNARArchitecture


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_plain_mapping(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        return {key: value[key] for key in value.keys()}
    except (AttributeError, KeyError, TypeError) as exc:
        raise TypeError(f"{name} must be a mapping") from exc


def validate_cach_flat_model_config(cfg: Any) -> None:
    """Validate the model-local CACH-A invariants at the construction seam."""

    root = _as_plain_mapping(cfg, "flat CACH model config")
    if root.get("variant") != CACH_VARIANT:
        raise ValueError(f"CACH model variant must be {CACH_VARIANT!r}")
    validate_cach_bootstrap_config(root)
    if root.get("proprio_value_binding") != "semantic_float32_sha256_v1":
        raise ValueError("CACH proprio value binding contract differs")
    if root.get("initialization_mode") != "complete_random_v1":
        raise ValueError("CACH requires initialization_mode='complete_random_v1'")

    action_condition = _as_plain_mapping(
        root.get("action_condition"),
        "action_condition",
    )
    required_action_condition = {
        "action_input_slots": "fixed_chunk_capacity_with_prefix_mask_v1",
        "condition_shape": "fixed_b_k_a_v1",
        "enabled": True,
        "initialization": "post_factory_exact_zero_v1",
        "input_dim": 20,
        "internal_vendor_seam": "use_delta_pose_additive",
        "padded_condition_value": "exact_zero",
        "partial_tail_mask": "explicit_boolean_b_k_v1",
        "public_name": "action_condition",
        "reducer": "end_of_latent_bin_command_v1",
    }
    if action_condition != required_action_condition:
        raise ValueError("CACH action_condition contract differs")

    components = {}
    for name in ("causal_softmax_anchors", "block_attn_res", "self_forcing"):
        component = _as_plain_mapping(root.get(name), name)
        components[name] = component
        if component.get("enabled") is not False:
            raise ValueError(f"CACH-A Stage 1 requires {name}.enabled=false")
    if components["causal_softmax_anchors"].get("indices") != []:
        raise ValueError("CACH-A Stage 1 requires an empty anchor index set")

    video = _as_plain_mapping(root.get("video_backbone"), "video_backbone")
    if video.get("model_path") is not None or video.get("init_dit_from") is not None:
        raise ValueError("fresh CACH construction forbids pretrained video DiT state")
    if video.get("attn_kernel") != "gdn":
        raise ValueError("CACH-A requires the all-GDN reference backbone")
    if video.get("use_delta_pose_additive") is not True:
        raise ValueError("CACH-A requires the registered additive action seam")
    if video.get("delta_pose_additive_dim") != 20:
        raise ValueError("CACH-A action seam width must be exactly 20")


class CausalActionHybridArchitecture(DualSystemGDNARArchitecture):
    """Future CACH-A numerical architecture; Stage 1 construction is denied."""

    def __init__(self, cfg: Any = None, *, cach_build_capability: Any = None):
        # This is deliberately first.  Stage 1 cannot construct a video model,
        # allocate CUDA state, load a checkpoint, or reach the legacy GDN-AR
        # defaults.  Stage 2 must replace this authority with a reviewed token.
        reject_cach_model_build(cfg, cach_build_capability)

        # Unreachable under the Stage-1 authority, retained as the reviewed
        # implementation seam for Stage-2 activation.
        validate_cach_flat_model_config(cfg)
        super().__init__(cfg)
        self._observed_prefix_chunks = 0
        self._cach_action_dim = int(
            _cfg_get(_cfg_get(cfg, "action_condition", {}), "input_dim", 0)
        )
        if self.video_backbone is None:
            raise RuntimeError("CACH construction requires a video backbone")
        dit = self.video_backbone._dit
        if hasattr(dit, "cach_no_action_slot"):
            raise RuntimeError("CACH no-action slot was already registered")
        first_parameter = next(dit.parameters())
        dit.register_parameter(
            "cach_no_action_slot",
            nn.Parameter(
                torch.zeros(
                    self._cach_action_dim,
                    dtype=first_parameter.dtype,
                    device=first_parameter.device,
                )
            ),
        )

    @property
    def cach_no_action_slot(self) -> Tensor:
        if self.video_backbone is None:
            raise RuntimeError("CACH no-action slot requires a video backbone")
        slot = getattr(self.video_backbone._dit, "cach_no_action_slot", None)
        if not isinstance(slot, Tensor):
            raise RuntimeError("CACH no-action slot is absent")
        return slot

    def forward_chunk(
        self,
        chunk_latents: Tensor,
        *,
        action_layout: ChunkActionLayout,
        expected_layout_instance_digest: str,
        chunk_id: int,
        noisy_actions: Tensor,
        committed_actions: Optional[Tensor],
        cache_read_view: DenoiseReadView,
        cache_scratch: CacheScratch,
        video_timestep: Tensor,
        action_timestep: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        seq_lens: Optional[Tensor] = None,
        proprio_state: Optional[Tensor] = None,
        proprio_bindings: Sequence[SelectedProprioBinding],
        save_kv_cache: bool = False,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], object]:
        """Run one layout-bound read-only denoise pass.

        The transactional paired ``t=0`` commit is intentionally a separate
        typed-cache operation.  Calling this method with ``save_kv_cache=True``
        would reintroduce the legacy denoise/commit ambiguity and is rejected.
        """

        if not isinstance(action_layout, ChunkActionLayout):
            raise TypeError("action_layout must be a ChunkActionLayout")
        action_layout.verify_instance_digest(
            expected_layout_instance_digest
        )
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
            raise TypeError("chunk_id must be a plain integer")
        if chunk_id < 0 or chunk_id >= len(action_layout.chunks):
            raise IndexError("chunk_id is outside action_layout")
        chunk = action_layout.chunks[chunk_id]
        if not isinstance(cache_read_view, DenoiseReadView):
            raise TypeError("cache_read_view must be a typed DenoiseReadView")
        if not isinstance(cache_scratch, CacheScratch):
            raise TypeError("cache_scratch must be a typed CacheScratch")
        if (
            cache_read_view.layout_instance_digest
            != expected_layout_instance_digest
            or cache_read_view.next_chunk_id != chunk_id
            or cache_scratch.source_state_manifest_digest
            != cache_read_view.state_manifest_digest
        ):
            raise LayoutContractError(
                LayoutReasonCode.LAYOUT_INSTANCE_DIGEST_MISMATCH,
                "typed cache read view differs from the requested layout chunk",
            )
        if save_kv_cache:
            raise RuntimeError(
                "CACH denoise is read-only; paired t=0 commit uses the typed cache transaction"
            )
        if chunk_latents.ndim != 5:
            raise ValueError("chunk_latents must have exact rank 5")
        if chunk_latents.shape[2] != len(chunk.latent_valid_mask):
            raise ValueError(
                "chunk_latents must contain the fixed K slots; partial tails "
                "are controlled by the explicit layout mask"
            )
        if proprio_state is None or not isinstance(proprio_state, Tensor):
            raise TypeError("CACH requires the layout-selected proprio_state tensor")
        expected_proprio_shape = (chunk_latents.shape[0], 20)
        if tuple(proprio_state.shape) != expected_proprio_shape:
            raise ValueError(
                f"proprio_state must have exact shape {expected_proprio_shape}"
            )
        if proprio_state.dtype is not torch.float32:
            raise TypeError(
                "proprio_state must remain semantic float32 until its "
                "layout-bound value digest is verified"
            )
        if proprio_state.device != chunk_latents.device:
            raise ValueError("proprio_state and chunk_latents device differ")
        if not bool(torch.isfinite(proprio_state.detach()).all()):
            raise ValueError("proprio_state contains NaN or Inf")
        try:
            bound_rows = tuple(proprio_bindings)
        except TypeError as exc:
            raise TypeError(
                "proprio_bindings must be a finite per-batch sequence"
            ) from exc
        if len(bound_rows) != chunk_latents.shape[0] or any(
            not isinstance(binding, SelectedProprioBinding)
            for binding in bound_rows
        ):
            raise TypeError(
                "proprio_bindings must contain one SelectedProprioBinding "
                "per batch row"
            )
        for batch_index, binding in enumerate(bound_rows):
            value_digest = canonical_proprio_row_sha256(
                proprio_state[batch_index]
            )
            binding.verify(
                layout_instance_digest=expected_layout_instance_digest,
                chunk=chunk,
                proprio_value_sha256=value_digest,
            )
        if noisy_actions.dtype != chunk_latents.dtype:
            raise TypeError("noisy_actions and chunk_latents dtype differ")
        if noisy_actions.device != chunk_latents.device:
            raise ValueError("noisy_actions and chunk_latents device differ")
        if not bool(torch.isfinite(noisy_actions.detach()).all()):
            raise ValueError("noisy_actions contains NaN or Inf")

        action_condition_batch = reduce_end_of_bin_action_condition(
            noisy_actions,
            committed_actions=committed_actions,
            chunk=chunk,
            no_action_slot=self.cach_no_action_slot,
        )
        if tuple(action_condition_batch.condition.shape[:2]) != (
            chunk_latents.shape[0],
            chunk_latents.shape[2],
        ):
            raise RuntimeError(
                "action condition and fixed-K video chunk axes differ"
            )
        del (
            video_timestep,
            action_timestep,
            context,
            context_mask,
            seq_lens,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
        )
        raise RuntimeError(
            "CACH Stage 1 has no reviewed typed-cache/list[10] codec; "
            "model forward remains disabled until Stage 2 numerical admission"
        )

    def compute_loss(self, *args: Any, **kwargs: Any) -> dict:
        raise RuntimeError(
            "CACH Stage 1 has no executable training loop; the legacy fixed-atc "
            "GDN-AR compute_loss path is forbidden"
        )

    def forward(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "CACH is layout-driven and has no whole-clip forward entry point"
        )


__all__ = [
    "CausalActionHybridArchitecture",
    "reduce_end_of_bin_action_condition",
    "validate_cach_flat_model_config",
]
