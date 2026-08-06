# LIBERO T3 Held-Out Recipe Transfer Screen

Status: pre-registered for execution on 2026-08-06. This document claims no T3
result until a unique terminal root has been frozen.

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
