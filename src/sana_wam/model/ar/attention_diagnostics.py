"""Opt-in, output-preserving diagnostics for SANA's structured AR attention."""

from __future__ import annotations

import types
from typing import Any

import torch
from einops import rearrange
from torch import Tensor


def _rms(value: Tensor) -> float:
    return float(value.double().square().mean().sqrt().item()) if value.numel() else 0.0


def _cosine(left: Tensor, right: Tensor) -> float:
    left_flat = left.double().flatten()
    right_flat = right.double().flatten()
    denom = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(left_flat, right_flat).div(denom).item())


class ARAttentionCapture:
    """Temporarily wrap one AR driver and record component-level attention flow.

    The wrapped mixed-attention method is called first and its tensor is returned
    unchanged. All diagnostics run afterward under ``no_grad`` on detached
    tensors. Exiting the context restores the original bound methods and removes
    every forward hook.
    """

    COMPONENTS = (
        "current_clean_video",
        "prior_clean_video",
        "clean_action_history",
        "within_frame_noisy_action",
    )

    def __init__(self, driver, *, capture_video_branches: bool = True) -> None:
        self.driver = driver
        self.capture_video_branches = bool(capture_video_branches)
        self.records: list[dict[str, Any]] = []
        self.query_group_records: list[dict[str, Any]] = []
        self.video_branch_records: list[dict[str, Any]] = []
        self.current_layer: int | None = None
        self._original_mixed = None
        self._original_step_impl = None
        self._original_video_post = None
        self._hook_handles = []
        self._pending_linear: dict[int, Tensor] = {}
        self._pending_flash: dict[int, list[Tensor]] = {}

    def __enter__(self) -> "ARAttentionCapture":
        if getattr(self.driver, "_ar_attention_capture_active", False):
            raise RuntimeError("AR attention diagnostics are already active on this driver")
        self.driver._ar_attention_capture_active = True
        self._original_mixed = self.driver._mixed_attention
        self._original_step_impl = self.driver._step_impl

        def wrapped_step(_driver, layer_id, *args, **kwargs):
            self.current_layer = int(layer_id)
            return self._original_step_impl(layer_id, *args, **kwargs)

        def wrapped_mixed(_driver, q_cat, k_cat, v_cat, attn_mask, **kwargs):
            output = self._original_mixed(
                q_cat, k_cat, v_cat, attn_mask, **kwargs
            )
            with torch.no_grad():
                self._record_attention(
                    q_cat.detach(),
                    k_cat.detach(),
                    v_cat.detach(),
                    kwargs["phi_q"].detach(),
                    kwargs["phi_k"].detach(),
                    output.detach(),
                )
            return output

        self.driver._step_impl = types.MethodType(wrapped_step, self.driver)
        self.driver._mixed_attention = types.MethodType(wrapped_mixed, self.driver)
        if self.capture_video_branches:
            self._register_video_hooks()
            self._wrap_video_post_attention()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._pending_linear.clear()
        self._pending_flash.clear()
        if self._original_mixed is not None:
            self.driver._mixed_attention = self._original_mixed
        if self._original_step_impl is not None:
            self.driver._step_impl = self._original_step_impl
        if self._original_video_post is not None:
            self.driver.vb.post_attn_at_layer = self._original_video_post
        self.driver._ar_attention_capture_active = False
        self.current_layer = None

    def _register_video_hooks(self) -> None:
        backbone = getattr(self.driver, "vb", None)
        dit = getattr(backbone, "_dit", None)
        blocks = getattr(dit, "blocks", ())
        for layer, block in enumerate(blocks):
            flash = getattr(block, "flash_attn_additional", None)
            if flash is None:
                continue

            def linear_hook(_module, _inputs, output, *, layer=layer):
                self._pending_linear[layer] = output.detach()

            def flash_hook(_module, _inputs, output, *, layer=layer, block=block):
                del block
                if layer not in self._pending_linear:
                    raise RuntimeError(
                        f"video flash hook at layer {layer} ran before the linear projection hook"
                    )
                self._pending_flash.setdefault(layer, []).append(output.detach())

            self._hook_handles.append(block.attn.proj.register_forward_hook(linear_hook))
            self._hook_handles.append(flash.register_forward_hook(flash_hook))

    def _wrap_video_post_attention(self) -> None:
        backbone = self.driver.vb
        self._original_video_post = backbone.post_attn_at_layer

        def wrapped_video_post(_backbone, layer_id, state, attn_out, post_state):
            residual = state.x.detach()
            result = self._original_video_post(
                layer_id, state, attn_out, post_state
            )
            layer = int(layer_id)
            linear = self._pending_linear.pop(layer, None)
            raw_chunks = self._pending_flash.pop(layer, None)
            if linear is None or not raw_chunks:
                raise RuntimeError(f"missing video branch tensors for layer {layer_id}")
            raw = torch.cat(raw_chunks, dim=1)
            if raw.shape != linear.shape:
                raise RuntimeError(
                    f"video flash chunks at layer {layer_id} reconstructed shape "
                    f"{tuple(raw.shape)}, expected {tuple(linear.shape)}"
                )
            scale = float(post_state["block"].learnable_fa_scale.detach().float().item())
            scaled = raw.float() * scale
            gate = post_state["gate_msa"].detach().float()
            if gate.dim() == 4:
                batch, tokens, dims = linear.shape
                frames = gate.shape[1]
                tokens_per_frame = tokens // frames
                gate = gate.expand(batch, frames, tokens_per_frame, dims).reshape(
                    batch, tokens, dims
                )
            gated_linear = gate * linear.float()
            gated_flash = gate * scaled
            full_output = result.x.detach().float()
            residual_float = residual.float()
            token_halves = linear.shape[1] // 2
            scopes = {"all": slice(None)}
            if token_halves * 2 == linear.shape[1]:
                scopes.update(
                    {
                        "noisy_copy": slice(0, token_halves),
                        "clean_copy": slice(token_halves, None),
                    }
                )
            for scope, token_slice in scopes.items():
                linear_scope = linear.float()[:, token_slice]
                scaled_scope = scaled[:, token_slice]
                gated_linear_scope = gated_linear[:, token_slice]
                gated_flash_scope = gated_flash[:, token_slice]
                residual_scope = residual_float[:, token_slice]
                output_scope = full_output[:, token_slice]
                linear_rms = _rms(linear_scope)
                scaled_rms = _rms(scaled_scope)
                gated_linear_rms = _rms(gated_linear_scope)
                gated_flash_rms = _rms(gated_flash_scope)
                residual_rms = _rms(residual_scope)
                block_delta = output_scope - residual_scope
                block_delta_rms = _rms(block_delta)
                self.video_branch_records.append(
                    {
                        "layer": int(layer_id),
                        "scope": scope,
                        "learnable_fa_scale": scale,
                        "linear_projected_rms": linear_rms,
                        "flash_raw_rms": _rms(raw.float()[:, token_slice]),
                        "flash_scaled_rms": scaled_rms,
                        "flash_scaled_over_linear_rms": scaled_rms
                        / max(linear_rms, 1e-30),
                        "linear_flash_cosine": _cosine(linear_scope, scaled_scope),
                        "combined_rms": _rms(linear_scope + scaled_scope),
                        "gated_linear_rms": gated_linear_rms,
                        "gated_flash_rms": gated_flash_rms,
                        "gated_flash_over_linear_rms": gated_flash_rms
                        / max(gated_linear_rms, 1e-30),
                        "gated_linear_flash_cosine": _cosine(
                            gated_linear_scope, gated_flash_scope
                        ),
                        "block_input_residual_rms": residual_rms,
                        "full_block_delta_rms": block_delta_rms,
                        "full_block_delta_over_input_rms": block_delta_rms
                        / max(residual_rms, 1e-30),
                        "full_block_output_rms": _rms(output_scope),
                        "full_block_unchanged_fraction": float(
                            (output_scope == residual_scope).float().mean().item()
                        ),
                        "block_input_abs_median": float(
                            residual_scope.abs().median().item()
                        ),
                        "full_block_delta_abs_median": float(
                            block_delta.abs().median().item()
                        ),
                    }
                )
            return result

        backbone.post_attn_at_layer = types.MethodType(wrapped_video_post, backbone)

    @staticmethod
    def _cat_indices(indices: list[Tensor], device: torch.device) -> Tensor:
        if not indices:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat([index.to(device) for index in indices])

    def _record_attention(
        self,
        q_cat: Tensor,
        k_cat: Tensor,
        v_cat: Tensor,
        phi_q: Tensor,
        phi_k: Tensor,
        output: Tensor,
    ) -> None:
        if self.current_layer is None:
            raise RuntimeError("mixed attention ran without a current layer id")
        meta = getattr(self.driver, "_ar_meta", None)
        if meta is None:
            raise RuntimeError("AR attention diagnostics require driver._ar_meta")
        heads = int(self.driver.num_heads)
        tq_all = rearrange(q_cat, "b s (h d) -> b h s d", h=heads).float()
        tk_all = rearrange(k_cat, "b s (h d) -> b h s d", h=heads).float()
        value_all = rearrange(v_cat, "b s (h d) -> b h s d", h=heads).float()
        pq_all = rearrange(phi_q, "b s (h d) -> b h s d", h=heads).float()
        pk_all = rearrange(phi_k, "b s (h d) -> b h s d", h=heads).float()
        output_all = rearrange(output, "b s (h d) -> b h s d", h=heads).float()
        device = q_cat.device
        frame_count = int(meta.num_frames)
        window = int(meta.window)

        self._record_query_groups(
            meta=meta,
            pq_all=pq_all,
            pk_all=pk_all,
            output_all=output_all,
            frame_count=frame_count,
            window=window,
        )

        for frame_id in range(1, frame_count, 2):
            qidx = meta.noisy_idx_per_frame[frame_id]
            if qidx is None:
                continue
            qidx = qidx.to(device)
            chunk = frame_id // 2
            low = max(0, frame_id - window)
            component_indices = {
                "current_clean_video": self._cat_indices(
                    [meta.clean_idx_per_frame[frame_id - 1]], device
                ),
                "prior_clean_video": self._cat_indices(
                    [
                        meta.clean_idx_per_frame[prior]
                        for prior in range(low, frame_id - 1)
                        if prior % 2 == 0 and meta.clean_idx_per_frame[prior] is not None
                    ],
                    device,
                ),
                "clean_action_history": self._cat_indices(
                    [
                        meta.clean_idx_per_frame[prior]
                        for prior in range(low, frame_id)
                        if prior % 2 == 1 and meta.clean_idx_per_frame[prior] is not None
                    ],
                    device,
                ),
                "within_frame_noisy_action": qidx,
            }
            query = tq_all[:, :, qidx]
            query_phi = pq_all[:, :, qidx]
            numerators: dict[str, Tensor] = {}
            denominators: dict[str, Tensor] = {}
            for component, indices in component_indices.items():
                if indices.numel() == 0:
                    numerators[component] = torch.zeros_like(query)
                    denominators[component] = torch.zeros(
                        *query.shape[:-1], 1, device=device, dtype=query.dtype
                    )
                    continue
                keys = tk_all[:, :, indices]
                values = value_all[:, :, indices]
                keys_phi = pk_all[:, :, indices]
                state = values.transpose(-1, -2) @ keys
                normalizer = keys_phi.sum(dim=-2, keepdim=True)
                numerators[component] = query @ state.transpose(-1, -2)
                denominators[component] = query_phi @ normalizer.transpose(-1, -2)

            total_denom = sum(denominators.values())
            contributions = {
                component: numerator / (total_denom + float(self.driver.eps))
                for component, numerator in numerators.items()
            }
            reconstructed = sum(contributions.values())
            actual = output_all[:, :, qidx]
            output_rms = _rms(actual)
            reconstruction_rmse = _rms(reconstructed - actual)
            for component in self.COMPONENTS:
                numerator = numerators[component]
                denominator = denominators[component]
                contribution = contributions[component]
                denominator_fraction = denominator / total_denom.clamp_min(1e-30)
                contribution_rms = _rms(contribution)
                self.records.append(
                    {
                        "layer": self.current_layer,
                        "chunk": chunk,
                        "action_frame_id": frame_id,
                        "component": component,
                        "key_tokens": int(component_indices[component].numel()),
                        "numerator_rms": _rms(numerator),
                        "denominator_mean": float(denominator.double().mean().item()),
                        "denominator_fraction_mean": float(
                            denominator_fraction.double().mean().item()
                        ),
                        "normalized_contribution_rms": contribution_rms,
                        "contribution_over_output_rms": contribution_rms
                        / max(output_rms, 1e-30),
                        "contribution_output_cosine": _cosine(contribution, actual),
                        "output_rms": output_rms,
                        "reconstruction_rmse": reconstruction_rmse,
                    }
                )

    def _record_query_groups(
        self,
        *,
        meta,
        pq_all: Tensor,
        pk_all: Tensor,
        output_all: Tensor,
        frame_count: int,
        window: int,
    ) -> None:
        device = pq_all.device
        for frame_id in range(frame_count):
            low = max(0, frame_id - window)
            for copy, noise_flag in (("noisy", 0), ("clean", 1)):
                plan = (
                    meta.noisy_idx_per_frame
                    if noise_flag == 0
                    else meta.clean_idx_per_frame
                )
                qidx = plan[frame_id]
                if qidx is None:
                    continue
                qidx = qidx.to(device)
                high = frame_id - 1 if noise_flag == 0 else frame_id
                clean_indices = self._cat_indices(
                    [
                        meta.clean_idx_per_frame[prior]
                        for prior in range(low, high + 1)
                        if meta.clean_idx_per_frame[prior] is not None
                    ],
                    device,
                )
                query_phi = pq_all[:, :, qidx]
                denominator = torch.zeros(
                    *query_phi.shape[:-1], 1, device=device, dtype=query_phi.dtype
                )
                if clean_indices.numel():
                    clean_normalizer = pk_all[:, :, clean_indices].sum(
                        dim=-2, keepdim=True
                    )
                    denominator += query_phi @ clean_normalizer.transpose(-1, -2)
                if noise_flag == 0:
                    self_normalizer = pk_all[:, :, qidx].sum(dim=-2, keepdim=True)
                    denominator += query_phi @ self_normalizer.transpose(-1, -2)

                flat_denom = denominator.double().flatten()
                group_output = output_all[:, :, qidx]
                flat_output = group_output.double().abs().flatten()
                quantiles = torch.tensor(
                    [0.001, 0.01, 0.5], device=device, dtype=torch.float64
                )
                denom_quantiles = torch.quantile(flat_denom, quantiles)
                output_quantiles = torch.quantile(flat_output, quantiles[1:])
                self.query_group_records.append(
                    {
                        "layer": self.current_layer,
                        "frame_id": frame_id,
                        "chunk": frame_id // 2,
                        "modality": "video" if frame_id % 2 == 0 else "action",
                        "copy": copy,
                        "query_tokens": int(qidx.numel()),
                        "query_phi_zero_fraction": float(
                            (query_phi.sum(dim=-1) == 0).float().mean().item()
                        ),
                        "denominator_zero_fraction": float(
                            (flat_denom == 0).float().mean().item()
                        ),
                        "denominator_below_1e_6_fraction": float(
                            (flat_denom < 1e-6).float().mean().item()
                        ),
                        "denominator_min": float(flat_denom.min().item()),
                        "denominator_p001": float(denom_quantiles[0].item()),
                        "denominator_p01": float(denom_quantiles[1].item()),
                        "denominator_median": float(denom_quantiles[2].item()),
                        "denominator_mean": float(flat_denom.mean().item()),
                        "output_rms": _rms(group_output),
                        "output_abs_p01": float(output_quantiles[0].item()),
                        "output_abs_median": float(output_quantiles[1].item()),
                        "output_abs_max": float(flat_output.max().item()),
                    }
                )


__all__ = ["ARAttentionCapture"]
