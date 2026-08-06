# LIBERO T3 Held-Out Recipe Transfer Screen

Status: completed on 2026-08-06 with frozen verdict
`T3_HELDOUT_RECIPE_TRANSFER_GO`.

## Question

T2 established that the production-shaped AR model can reduce one fixed real
LIBERO sample's loss under one fixed stochastic loss recipe. T3 asks whether the
same 20 updates also reduce loss under three stochastic recipes that never
participate in backward or parameter updates.

This remains a single-sample, single-initialization, non-formal architecture
screen. It does not test new episodes, task or suite generalization, simulator
success, benchmark performance, or stable long-run training.

## Frozen predecessor and unchanged variables

- T2 frozen result:
  `/DATA/share/sana_wam_libero_nonformal_screens/t2/2dc1ce730df3/libero-t2-fixed20-20260806-a1/RESULT.json`.
- T2 result SHA256:
  `67d250cdb13470bea9e9fa53531d3c76c65145e5040dc7a45079d84ac4960a57`.
- T2 execution source commit:
  `2dc1ce730df37bd9a2a71e4946f153380a8d4649`.
- T2 runner SHA256:
  `f5e5e3e80d6b5afd586e54e83a7e63dcf43280192363c70580aa03145267e9a6`.
- Config SHA256 remains
  `5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`;
  Sana remains `16b9cec673e3335724ba2d8db25de7f9ed229292`.

T3 keeps the same LIBERO Spatial episode 0/start 0 sample, selected-row v2
statistics and raw asset pins, fresh initialization seed `20260806`, training
loss-recipe seed `20260826`, four trainable roots, frozen video backbone,
action-only loss, AdamW FP32 masters, LR `1e-4`, betas `(0.9, 0.95)`, zero
weight decay, clip bound `1.0`, BF16 projection, and exactly 20 updates. It adds
no scheduler, accumulation, warmup, checkpoint, data, sample, LR arm, or early
stopping.

## Mechanically derived held-out seeds

The seeds are derived from the T2 result SHA, not chosen after observing a
loss. For index `i = 1, 2, 3`, the ASCII payload is:

```text
SANA-WAM/LIBERO/T3_HELDOUT_RECIPE_V1
T2_RESULT_SHA256=67d250cdb13470bea9e9fa53531d3c76c65145e5040dc7a45079d84ac4960a57
INDEX=i
```

The rule is
`1 + (big_endian_uint64(SHA256(payload)[0:8]) mod 2147483646)`:

- H1: seed `897500337`, payload SHA256
  `08c9698412591ea0f219f56d2f293482c06d2303fb70c2de85f41a314f64d856`.
- H2: seed `2142397805`, payload SHA256
  `ef4652c542991a4ab1b7fc1d031e27ca0cda41d8cc509696458b83c8d1b8827c`.
- H3: seed `1238092489`, payload SHA256
  `f24a4b3b80a29dcc8b8dd1bf36cfcc5d747a69dd3ed50889112c6c3a76f055cd`.

Their order is H1, H2, H3. They may not be replaced because of observed loss,
runtime failure, or an unfavorable result.

## Exact execution order and budget

1. H1, H2 and H3 pre-update probes, with no backward or update.
2. Training-recipe initial probe.
3. Twenty training-recipe forward/backward/clip/master-sync/AdamW/BF16-project
   updates, preserving the T2 order exactly.
4. Training-recipe final probe.
5. H1, H2 and H3 post-update probes, with no backward or update.

The fixed total is 28 architecture forwards: 20 training forwards and eight
measurements, of which six are held-out. There are exactly 20 backward calls,
20 optimizer steps, one prepared sample, one optimizer, and one persistent set
of FP32 masters.

The execution root must be fresh and follow
`.../t3/<execution-commit-12>/libero-t3-heldout3-fixed20-<32-hex-nonce>`.
The runner binds the parent to the expected source commit, binds the basename
to the nonce, and writes both absolute root and nonce into the terminal result.

Every probe uses the same training/module mode as T2 but returns only Python
scalars and a captured-timestep signature. Before/after state snapshots require
parameter, master, buffer, module-mode and optimizer-state identity/version to
remain unchanged; gradients must remain absent and caller Python/NumPy/Torch
CPU/CUDA RNG state must be restored. Each held-out seed's pre/post signature
must match. The training recipe retains 22 matching signatures and the initial
probe must still match step 1's pre-update loss within the T2 tolerance.
The captured training identity and three held-out identities must also be four
distinct signatures; otherwise the run fails rather than treating duplicate
recipes as three held-out probes.

## Pre-registered verdict

For each held-out seed, define `r_i = post_action_loss / pre_action_loss` and
`improved_i = post_action_loss < pre_action_loss`. The primary gate is exactly:

```text
median(r_1, r_2, r_3) <= 0.95 AND sum(improved_i) >= 2
```

- A valid run satisfying the gate is `T3_HELDOUT_RECIPE_TRANSFER_GO`.
- A valid run missing the gate is
  `T3_HELDOUT_RECIPE_TRANSFER_INCONCLUSIVE`; it is not a harness failure and
  does not authorize an automatic rerun.
- Identity, state-mutation, count, finite-value or runtime failures freeze
  `FAILED.json` and produce no scientific verdict.

The inherited T2 training-recipe endpoint/median gate, overshoot, clip-active
fraction and cumulative BF16-root coverage are recorded as secondary
diagnostics. The T3 primary verdict is intentionally the paired held-out gate;
if the T2-style diagnostic does not reproduce, it must be reported explicitly
but may not silently redefine the pre-registered T3 question.

The runner may load the pinned published SANA/Gemma/VAE construction assets but
must not load, resume or save a SANA-WAM checkpoint. It may not call
`Trainer.train`, run a simulator or benchmark evaluator, create a formal
training root, or claim sample/benchmark generalization. T3 does not authorize
formal training or the next experiment.

## Executed identity and evidence

- Execution source commit:
  `ac431f8727ac345c3bd0ad4a442e53801310fd81`.
- Runner SHA256:
  `0db0bfe086bc4b5ea92462e4832f0c448408656cca819796ec253f6226b4c017`.
- Config SHA256:
  `5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`.
- GPU: physical GPU 0,
  `GPU-1ec28cfb-f501-23f3-f865-275a744ca053` (`NVIDIA H200`).
- Nonce: `220afb60725d0cfd591bc4fe225cdd21`.
- Frozen root:
  `/DATA/share/sana_wam_libero_nonformal_screens/t3/ac431f8727ac/libero-t3-heldout3-fixed20-220afb60725d0cfd591bc4fe225cdd21`.
- `RESULT.json` SHA256:
  `c883608f2a47b6258f824d4d97a94f8a390d03bab671a592fb758eea61b3a01e`.
- Terminal permissions: root `0500`, result `0400`; no `FAILED.json` exists.
- Pre-execution validation: Ruff passed and all 72 `test_libero_*.py` CPU
  tests passed. The source commit was pushed before GPU execution.
- Post-process GPU check: 0 MiB used and 0% utilization.

## Result

All three held-out recipes improved, and the primary gate passed without a
borderline result:

| Recipe | Seed | Pre loss | Post loss | Post/pre | Drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 | 897500337 | 19.774147 | 2.461924 | 0.124502 | 87.55% |
| H2 | 2142397805 | 9.396403 | 3.915653 | 0.416718 | 58.33% |
| H3 | 1238092489 | 8.799880 | 3.667903 | 0.416813 | 58.32% |

- Improved count: `3/3`, versus the required `2/3`.
- Median paired ratio: `0.41671822538936226`, versus the maximum `0.95`.
- Pre/post captured signatures matched within each pair, and the training
  recipe plus H1/H2/H3 produced four distinct recipe identities.
- All 28 RNG contexts restored caller Python/NumPy/Torch CPU/CUDA state.
- Pre/post probe snapshots confirmed no change to parameter, master, buffer,
  module-mode or optimizer-state identity/version; no probe created gradients.
- Exactly 28 architecture forwards, 20 backward calls and 20 optimizer steps
  completed. Final Adam counters were all 20 and cumulative BF16-visible
  changes covered all four trainable roots.

The secondary training-recipe diagnostics reproduced T2 exactly: initial loss
`13.679718971252441`, the complete 20-point pre-update loss curve, all recorded
pre-clip gradient norms, final loss `1.7331912517547607`, and sampled master-root
coverage were value-identical to the frozen T2 result. There were no secondary
diagnostic warnings. This is strong evidence that the six added measurements
did not perturb the update path.

Model/dataset construction took `39.0128 s`; the 20 updates plus eight
measurements took `34.8056 s`. Peak update allocation/reservation was
29.19/31.49 GB (27.19/29.33 GiB), within the selected H200.

## Interpretation

T3 closes the narrow stochastic-recipe transfer question: on the same real
LIBERO window, parameters updated using recipe `20260826` improved all three
mechanically selected, gradient-held-out recipes. The T2 result was therefore
not merely memorization of one sampled diffusion/noise realization.

This still says nothing about a different observation/action window. It also
does not remove T2's optimization warning: the unchanged training curve still
spiked by 221.8x after the first update and all 20 gradients hit clipping.

The smallest next architecture experiment is T4 held-out-sample transfer:
retain the entire T2 training core and training recipe, but measure three
pre-frozen real windows from the same Spatial suite and task, different episodes
and start 0, before and after the 20 updates. Those samples must remain
update-free and use the training recipe so only the sample axis changes. A
learning-rate stability experiment remains separate and should not be combined
with T4.
