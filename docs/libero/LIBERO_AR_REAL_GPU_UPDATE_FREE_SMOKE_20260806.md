# LIBERO AR real-sample single-GPU update-free smoke — 2026-08-06

## Outcome

PASS. A complete `DualSystemARArchitecture` was constructed on one H200 from
the LIBERO AR source config, a fixed real LIBERO Spatial sample was decoded and
encoded by the local Wan VAE and Gemma assets, and exactly one full-window AR
architecture forward completed with finite video and action outputs.

This is a **source/model-construction smoke only**. It is not training,
checkpoint validation, a simulator rollout, or a LIBERO benchmark result.
There is no trained SANA-WAM LIBERO checkpoint yet.

## Mode-frozen non-formal evidence

This evidence is immutable by file mode at the time of recording, but its
`/tmp` location is ephemeral and is not a durable formal evidence store.

- Result root:
  `/tmp/sana-wam-libero-ar-real-gpu.BWntjDjy`
- Result:
  `/tmp/sana-wam-libero-ar-real-gpu.BWntjDjy/RESULT.json`
- Result SHA256:
  `e63c728f57bfa2e6106ec6a60cfb8ad9bce1ac01f8489c0b5624e66b4854c079`
- Root mode: `0555`
- Result mode: `0444`
- Runner SHA256:
  `cc4556cb3e454b3c9b30ae9546bb5e2544af059d3d471f95fb8c1266e1cb5c0c`
- Config SHA256:
  `53e75eb1d3df1a8aafae786fa384e46b1aa6a2901d254020a8434c8c14420dca`
- Temporary metadata-bootstrap stats SHA256:
  `333b2cb1e150b451ee1cf6833b1e628ebb50914a466f2cb1fa1808cff8e9f2d0`
- Pre-run repository HEAD:
  `29c74452f5b5e03b6f33c6c129b1bedaadce80cf`

The stats artifact above came from the earlier CPU alignment smoke. It is
appropriate for this non-formal construction check, but it is not the future
training-split-only admission artifact.

The first attempt froze before model forward because the runner initially
rejected every base-checkpoint partial load:

- Frozen failed root: `/tmp/sana-wam-libero-ar-real-gpu.4x6TYA2n`
- Marker: `FAILED_PRE_FORWARD` (`0444`)
- Cause: the published SANA base checkpoint intentionally lacks 280 parameters
  introduced by the configured learnable feature map and window-attention
  additions.

The successful runner accepts the observed `280 missing / 0 unexpected` count
and the fixed leading missing-key samples. A changed count, any unexpected key,
or changed pinned leading samples remains a hard failure. It does not yet hash
the complete 280-key set.

## Exact scope

- Physical GPU: index `0`
- UUID: `GPU-1ec28cfb-f501-23f3-f865-275a744ca053`
- Visible device inside the process: `cuda:0`
- GPU was idle before construction and returned to zero used memory after exit.
- Torch: `2.7.1+cu128`
- Model dtype: BF16
- Seed: `20260806`
- Hugging Face/Transformers/Datasets offline modes were forced before model
  imports.
- All architecture parameters were set to `requires_grad=False`.
- Parameter tensor version counters were unchanged across preprocessing and
  forward.
- Optimizer creation, backward, parameter updates, SANA-WAM training-checkpoint
  load/save, training, simulator execution, and benchmark evaluation were all
  absent. The published SANA base asset checkpoint was loaded for construction.

The local published base assets were loaded for construction:

- `/DATA/share/SANA-Video_2B_480p`
- `/DATA/share/gemma-2-2b-it`

No SANA-WAM training checkpoint was loaded. Consequently, the 7D ActionDiT,
8D proprio modules, and the 280 AR video additions were fresh seeded
initializations. Output values establish numerical and interface closure only;
they say nothing about policy quality.

## Real sample and tensor closure

The fixed source sample was:

- Dataset: `libero_spatial_no_noops_1.0.0_lerobot`
- Episode: `0`, start frame `0`, length `110`
- Task: `pick up the black bowl next to the cookie box and place it on the plate`
- Alignment: `observation_t_to_action_t`
- Parquet SHA256:
  `3f875604fad478765549128759edfb33a64b69b7b82decebc9e4f38155f20a8c`

Prepared model inputs:

| Tensor | Shape | Notes |
|---|---:|---|
| video latents | `[1, 16, 8, 48, 40]` | real 29-frame RGB composite through Wan VAE |
| text context | `[1, 33, 2304]` | real task prompt through Gemma |
| actions | `[1, 112, 7]` | final 3 tokens padded |
| proprio state | `[1, 8]` | independently normalized state |
| proprio sequence | `[1, 113, 8]` | fixed training window |
| per-chunk proprio | `[1, 4, 8]` | indices `[0, 28, 56, 84]` |

The architecture used four latent chunks (`F=2`) and 28 action tokens per
chunk. The output shapes matched their noisy input shapes exactly:

| Output | Shape | Finite | Range |
|---|---:|---:|---:|
| video prediction | `[1, 16, 8, 48, 40]` | yes | `[-2.46875, 2.140625]` |
| action prediction | `[1, 112, 7]` | yes | `[-4.96875, 6.96875]` |

## Timing and memory

| Stage | Seconds | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|---:|
| model construction | 39.042 | 11,775,243,776 | 12,167,675,904 |
| real sample VAE/text preparation | 0.821 | 12,451,129,344 | 13,103,005,696 |
| one full AR forward | 0.414 | 12,800,135,168 | 13,384,024,064 |

## Source correction required for closure

The variable-length LIBERO window always supplies explicit action/video masks.
SANA rejects any `frame_is_pad` input unless its non-cached temporal operators
are constrained to physical AR chunks. The source template therefore now
sets:

```yaml
model:
  architecture:
    ar_chunkwise_temporal_ops: true
```

Without this field, the first real model forward deterministically fails with
`frame_is_pad requires configured AR chunkwise temporal operators`.

## Remaining blockers before training

1. Materialize immutable training-split-only action/state stats at the
   production path and pin its SHA in the training config.
2. `training.use_gradient_checkpointing: true` is currently dropped by AR
   `compute_loss` before its forward calls. Repair and test that propagation
   before any update smoke or training; this forward-only smoke intentionally
   used inference mode and is unaffected.
3. Decide and freeze the intended LIBERO video timestep contract. The current
   source config uses integerized BF16 T0 conditioning
   (`continuous_timestep_conditioning=false`), rather than the newer FP32
   continuous T1 path.
4. Run a separately authorized single-GPU update smoke only after the first
   three contracts close. Formal training and benchmark evaluation remain
   unauthorized.

## Verification commands

```bash
.venv/bin/ruff check \
  scripts/smoke_libero_ar_real_gpu.py \
  tests/test_libero_dataset.py

CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m pytest -q \
  tests/test_libero_dataset.py
```

Observed: ruff passed; the focused dataset command reported `9 passed`, and the
broader LIBERO/AR padding regression set reported `91 passed`.
