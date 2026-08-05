# CACH-A3 normalized-common exact-odd output residual

Status: `ARCHITECTURE_FIRST_NONFORMAL_SUCCESSOR`

Canonical host/worktree: `H200` / `/home/zch/workspace/sana-wam`

Architecture ID: `CACH-A3-NORMALIZED-COMMON-EXACT-ODD-OUTPUT-RESIDUAL-v1`

## Authority and scope

The user continued the architecture-development task with the exact UTF-8 text
`继续` (SHA256
`7c9691192f1b73408bbe4c0cb6d00db94375ca9d8fce0a0d5985e7a5178f083f`).
This decision interprets that instruction only as authority for one additive,
single-GPU, synthetic, non-formal A3 screen.  It is not an A2 retry, AV2,
formal admission, real-data training, checkpoint use, deployment, or Global
Stage 3.  No review token may be created, reset, derived, or consumed.

## Frozen evidence and diagnosis

- A2-R1 card SHA256:
  `c7b710177defda6d44066c7b7b25c4b32d11ec106a92662b0701ca5f87f3401e`.
- A2-R1 RESULT SHA256:
  `a7767e47d718deda41762db4e250dbca0602f78fcff1651ef06a0d6da508c2ef`.
- A2-R1 RAW_METRICS SHA256:
  `557cb38344b81c0d6d3ab9a366d78f9d21807cff1d7214f6d36aa981cb50146c`.
- A2-R1 terminal state:
  `OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED / A2_DELTA_LIVE_COMMON_MODE_BLOCKED`.

A2 learned the counterfactual action delta accurately: delta NMSE was
`0.001281`, energy ratio `1.00634`, and alignment cosine `0.999366`.  Its final
error decomposed almost entirely into common error:
`44.894969 ~= 44.894647 common + 0.000320 half-delta`.

The old relative total-MSE separation threshold was scale-confounded.  With
target half-delta energy near `0.25` and common error near `44.9`, even a
perfect action residual changes total MSE by only about `0.56%`; requiring a
`50%` relative gap cannot measure action learnability in that state.

## Architecture decision

Both arms use the same action-blind vendor-GDN/FFN common trunk and the same
fresh common initialization.  A parameter-free RMS normalization is applied
to the final common hidden before the shared video output projection.  This is
a common stabilizer present in both arms and therefore is not the treatment
delta.

The candidate's only treatment delta is a parallel output-space action path:

```text
raw_action, typed_mask = frozen end-of-bin reducer(actions)
g(a) = bias-free Linear(20,64) -> SiLU -> bias-free Linear(64,20)
odd(a) = 0.5 * (g(a) - g(-a))
delta_video = zero-init bias-free Linear(20,3)(odd(a))
prediction = common_prediction + typed_active(delta_video)
```

The residual is added after the common output.  No norm, FFN, vendor block, or
common parameter consumes it.  Reference, full no-action, bootstrap and
`seam_disabled` are structural bypasses.  Inactive modes do not call the
reducer or action modules.

## Orthogonal mechanism training

This is a diagnostic mechanism screen, not the final production objective.
It uses only two positive losses; there is no loss subtraction:

- common parameters minimize MSE to the paired common target;
- candidate-only action parameters minimize MSE to the signed half-delta
  target.

Reference-common and candidate-common have independent but identically
configured AdamW optimizers.  Candidate action parameters have a disjoint
AdamW optimizer and gradient clip.  All use 200 final-step macrosteps and a
fixed cosine learning-rate schedule from `0.003` to `0.00003`.  No best-step,
checkpoint, restart, seed search, or budget extension is allowed.

## Decision metrics

Action quality is normalized by the registered target half-delta energy:

- delta NMSE, energy ratio and alignment cosine;
- explained action fraction;
- no-action recovery;
- shuffled-action penalty.

The old total-MSE relative gaps remain diagnostic-only.  Common prediction
parity, exact oddness, zero origin, typed bypass, finite gradients/updates,
200-step completion, final-step publication and common stability are recorded
separately.

A valid action GO only supports designing the next reduced architecture
screen.  It does not unlock AV2 or any formal/real-data work.  A common-path
block is reported independently and must be addressed without changing the
validated action residual.
