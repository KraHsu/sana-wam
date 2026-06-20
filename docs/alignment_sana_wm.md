# Alignment audit: sana-wam ↔ Sana-wm

Comparison of the **sana-wam** fork (robotics action-world model, `src/sana_wam/…`) against
upstream **Sana-wm** (`../Sana/diffusion/…`, branch `main`, commit `59629fd`). Goal: align
dataloader and attention per-component (numerical parity where meaningful, pattern adoption
elsewhere). Pinned submodule is `third_party/Sana` @ `v1.5.0-74-g6554c8d` (older than `../Sana`).

## 0. Decisions resolved before any code

| Decision | Fork today | Upstream Sana-wm (main) | Resolution |
|---|---|---|---|
| Self-attn family | `linear_relu` (`LiteLAReLURope` dual-track) | **GDN** (`ChunkCausalGDN` / `CachedChunkCausalGDN` / Triton) | **Keep `linear_relu`.** GDN is a different operator; swapping it would invalidate the frozen `SANA-Video_2B_480p` weights and the dual-stream ActionDiT trained against them. GDN belongs to a separate retraining track — out of scope. |
| VAE | Wan 2.1 causal VAE (tied to checkpoint) | `LTX2VAE` (`vae_stride [8,32,32]`, `vae_latent_dim 128`) | **Keep Wan VAE.** By-necessity divergence; documented, not changed. |
| RoPE | video: submodule `apply_rotary_emb`; action: `rope_apply_1d` (Wan/FastWAM numerics, θ=10000) | `wan_rope` | Audit numerics, pin with a test. Expected already-consistent (both Wan-derived). |

## 1. Attention

| Aspect | Fork (`src/sana_wam/model/ar/…`) | Upstream (`sana_blocks.py`, pinned submodule) | Status |
|---|---|---|---|
| Dual-track linear attn | `sana_linear_attn._expanded_linear_attn`: rotated `tilde` numerator, un-rotated `phi` denominator, `eps=1e-15` | `LiteLAReLURope.forward`: `z = 1/(k.sum().T @ q + eps)`, rotated `q/k_rotated` numerator, same eps | **Math matches.** Fork cites upstream line numbers. |
| Chunk-causal cumsum | `_chunked_linear_attn`: per-chunk `S = Σ v⊗tilde_k`, `z = Σ phi_k`, prefix-summed | `ChunkCausalAttention.forward`: `cumsum_vk`, `cumsum_k_sum` loop over frame chunks | **Same recurrence.** No parity test yet → Phase 3. |
| Streaming KV cache | `ARLinearStateCache` + `ar_inference_attn`: per-frame `(S,z)`, predicted/confirmed flag, window eviction | `CachedCausalAttention.forward`: `kv_cache=[cumsum_vk, cumsum_k_sum]`, `save_kv_cache` | **Same design.** Fork adds predicted-entry replacement (LingBot). Parity test → Phase 3. |
| AR duplicated-seq mask | `sana_ar_linear_attn`: `[v_noisy|v_clean|a_noisy|a_clean]`, frame-parity + windowed structured kernel | (none — fork-specific LingBot-VA topology) | Fork-only; correct by design (oracle-tested). |
| Joint MoT mixed attn | `mot_driver`/`sana_mot_driver`: concat video+action Q/K/V, masked SDPA or linear | (none — single-stream upstream) | Fork-only. |

**Takeaway:** attention is already aligned at the math level. Phase 3 only *guards* it with a
ported-upstream parity test; no behavior change.

## 2. Dataloader

| Aspect | Fork (`src/sana_wam/dataloader/…`) | Upstream (`diffusion/data/…`) | Action |
|---|---|---|---|
| Source format | RoboTwin HDF5 episodes (`robotwin_dataset.py`) | WebDataset TAR / zip-latent (`sana_wm_zip_latent_data.py`) | Keep HDF5 (domain). |
| VAE latents | encoded **live** every step in `base.preprocess` (`frames`→latent) | **precomputed** latent-cache zips (`vae_cache_dir`, `load_vae_feat=True`) | **Adopt opt-in latent cache** → Phase 2. Parallels existing `pre_encoded_text` seam in `base.prepare_inputs`. |
| Text features | `TextEmbeddingCacheTransform`: bucketed `<sha[:2]>/<sha>.safetensors`, CFG-empty entry | `load_text_feat` flag, flat npz | Fork **more advanced**; align naming/docs only. |
| Aspect-ratio bucketing | N/A — fixed multiview resolution (`height×width`, L-shape 3-cam) | `AspectRatioBatchSampler`, multi-scale | **N/A** for fixed-res robot cams; documented, skipped unless multi-res wanted. |
| Collation | `collate_fn=list` → `prepare_inputs` pools (variable-length PIL) | `custom_collate_fn` (default_collate + error log) | Keep; different by data shape. |
| Normalization | `ActionNormalizer` 5 modes + gripper override | (text-image; N/A) | Fork-only. |

## 3. RoPE / VAE / config conventions

| Knob | Fork | Upstream Sana-wm yaml | Action |
|---|---|---|---|
| `fp32_attention` | implicit (`fp32_attention` attr read by linear attn) | `fp32_attention: true` | Surface in `train_ar_sana.yaml`. |
| `qk_norm` | RMSNorm in backbones | `qk_norm: true` | Confirm + document (already on). |
| RoPE θ | action θ=10000 (`precompute_freqs_cis_1d`); video via submodule | `pos_embed_type: wan_rope` | Phase 1 numeric pin test. |
| scheduler | flow-matching (Euler), action/video schedulers | `flow_shift: 9.95`, `inference_flow_shift: 9.8`, `vis_sampler: chunk_flow_euler` | Surface `flow_shift` knobs where they bind; document fixed ones. |
| VAE | Wan 2.1 causal | `LTX2VAE_diffusers` | Keep (decision §0). |

## Scope summary

- **Do:** opt-in latent cache (Phase 2), RoPE/config surfacing + pin test (Phase 1), chunk-causal
  + KV-cache parity test (Phase 3).
- **Don't:** GDN family, LTX2VAE, Triton GDN, wan_rope swap — tied to retrained upstream Sana-wm,
  would break the fork's frozen `linear_relu` checkpoint.

## What was implemented

| Phase | Change | Files |
|---|---|---|
| 1 | RoPE pinned to Wan convention (θ=10000, polar, complex view) | `tests/test_action_rope_wan_parity.py` |
| 1 | `fp32_attention` config knob wired onto SANA attn modules (was a dead `getattr` path); `flow_shift`/`qk_norm` surfaced+documented | `pipeline_builder.py`, `configs/train_ar_sana.yaml` |
| 2 | Opt-in VAE latent cache (`load_vae_feat` analogue): `pre_encoded_latents`→`input_latents` through `prepare_inputs`; bucketed safetensors cache keyed on (episode, window, geometry) | `model/base.py`, `dataloader/transforms/vae_latent_cache.py`, `dataloader/robotwin_dataset.py`, `scripts/precompute_vae_latents.py`, `configs/train_ar_sana.yaml` |
| 2 | Cache round-trip + collation contract tests | `tests/test_vae_latent_cache.py` |
| 3 | Parity vs upstream's actual `ChunkCausalAttention`/`CachedCausalAttention` loops (not just the fork's internal oracle) | `tests/test_upstream_chunkcausal_parity.py` |

All 116 tests pass (was 98). No checkpoint/weight changes; latent cache and `fp32_attention`
are opt-in (defaults preserve byte-identical prior behavior). `flow_shift` stays `3.0` (480p
checkpoint value — **not** upstream's 720p `9.95`).

