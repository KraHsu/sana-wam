# LIBERO T10 Exact-Balanced Joint-Objective Screen

Status: **pre-registered implementation / not executed**

Date: 2026-08-06

## 1. Core question

T9 froze `T9_COMMON_POSITION_EFFECT_SUPPORTED`: every suite had worse terminal
pair ratio at the oldest than at the newest Latin-square position, and three
of four suites had position Spearman rho at least 0.8. T10 asks the smallest
next question:

> Does replacing sequential single-suite optimizer updates with an exact,
> equal-weight four-suite objective remove the common retention penalty when a
> matched SEQ bridge holds optimizer steps, raw micro-calls, and cumulative
> scalar loss coefficient fixed?

This is an optimization-topology discriminator for the existing
production-shaped `DualSystemARArchitecture`. It does not add an adapter,
router, replay buffer, sample, checkpoint, seed, or architecture parameter.

## 2. Frozen predecessor

The direct predecessor is the immutable T9 aggregate:

- aggregate source commit
  `b7cded5fd9cf83ffabeba18a7e352b1cb4438b66`;
- aggregate runner SHA256
  `b884ef028a48ee37afda70e730c431db48ed30f62be7f59cb2c665b6072ffc0d`;
- aggregate RESULT SHA256
  `0cfc53b820939123de4bc2a626380495878d4d5c9a37a0be9be667173254149d`;
- aggregate root
  `/DATA/share/sana_wam_libero_nonformal_screens/t9_aggregate/b7cded5fd9cf/libero-t9-latin-square-combined-ab99dd8758ef03bb191d5fb48f3b9fbc`;
- typed verdict `T9_COMMON_POSITION_EFFECT_SUPPORTED`.

T10 reuses the exact T7-T9 A0-A3 update samples and H0-H3 update-free
same-task probes, their manifests and assets, configuration SHA256
`5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`,
initialization seed `20260806`, and loss-recipe seed `20260826`. Each fresh
run must reproduce the eight frozen pre-action losses before any update.

The matched SEQ bridge must also reproduce these frozen T7 terminal action
losses:

```json
{
  "A0": 14.97652816772461,
  "A1": 11.250626564025879,
  "A2": 5.85097074508667,
  "A3": 2.8098158836364746,
  "H0": 14.401052474975586,
  "H1": 11.137908935546875,
  "H2": 6.1973876953125,
  "H3": 2.8864619731903076
}
```

## 3. Matched SEQ and JOINT arms

T10 has two separate fresh-initialization arms. Each has 20 macro optimizer
steps. At every macro step, with no parameter or optimizer mutation between
micro-batches, both arms process update samples in the fixed order:

```text
A0 Spatial, A1 Object, A2 Goal, A3 LIBERO-10
```

The arm-specific loss-weight vectors are:

```text
SEQ:   cyclic one-hot [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] x 5
JOINT: [0.25,0.25,0.25,0.25] at every macro step
```

All four weighted losses call backward. Their gradients accumulate at the
same parameter point; then and only then the runner performs one global
gradient clip, one FP32-master AdamW step, and one exact FP32-master-to-BF16
projection. Model/master gradients are cleared once per macro step, not
between its four micro-batches. The zero-weight SEQ micro-batches match the
JOINT computation schedule without contributing an update gradient.

Therefore:

- optimizer steps remain 20, matching every T7-T9 arm;
- each A sample is evaluated and backpropagated 20 times per arm;
- each A sample has cumulative scalar loss coefficient 5 in both arms:
  five one-hot unit weights in SEQ and `20 * 0.25` in JOINT;
- H0-H3 have zero backward/update exposure;
- learning rate `1e-4`, AdamW betas `(0.9,0.95)`, weight decay 0, gradient
  clip 1.0, trainable roots, FP32 masters, and BF16 projection are unchanged.

The exact execution budget **per arm** is:

- 8 preparation calls;
- 96 architecture forwards: 8 pre, 80 training, 8 terminal;
- 80 backward calls, four per macro step;
- 20 optimizer steps;
- 20 raw forward/backward exposures and effective coefficient 5 per A label;
- zero training exposures for every H label.

The fixed micro order is a deterministic floating-point reduction order. It is
identical across arms, and no parameter, optimizer, or projection change may
occur inside a macro step. SEQ must reproduce the frozen T7 terminal losses;
otherwise the matched bridge is invalid and JOINT has no scientific status.
For every macro step, the runner must capture model-parameter, persistent
FP32-master, and optimizer-state tensor identity/version snapshots immediately
before the first micro-backward and immediately after the fourth. It may mark
the no-intra-macro-mutation contract true only when those snapshots are equal;
gradient accumulation is intentionally excluded from that equality check.

## 4. Dose-matching boundary

SEQ and JOINT match optimizer-step count, raw forward/backward count, fixed
micro order, and cumulative scalar loss coefficient per sample. The frozen T7
run remains the historical sequential reference; the zero-weight SEQ bridge
must reproduce it before the paired comparison is interpretable.

JOINT changes when each sample contributes nonzero gradient: every sample
contributes one quarter at every optimizer state instead of contributing one
unit at five cyclic states. AdamW, clipping, and nonlinear parameter paths
mean this is an operational matched intervention, not a proof that gradient
averaging alone is the unique causal mechanism.

This is not approximate replay: the exact four frozen update samples are
present in every registered macro objective with fixed registered
coefficients; no cached latent, gradient, or surrogate sample is used.

## 5. Terminal metrics

After macro step 20 in each arm, with gradients cleared and no further update,
measure `A0,A1,A2,A3,H0,H1,H2,H3` once in that order. For arm `x` and suite
`i` compute from raw pre/terminal action losses:

```text
rA_xi = A_terminal_xi / A_pre_xi
rH_xi = H_terminal_xi / H_pre_xi
q_xi  = (rA_xi + rH_xi) / 2
both_improved_xi = (rA_xi < 1 and rH_xi < 1)
```

All losses and ratios must be finite and strictly positive. The T9
four-position q medians, frozen before T10 execution, are:

| Suite | T9 median q |
|---|---:|
| Spatial | 0.7488868555671038 |
| Object | 0.6095596365043602 |
| Goal | 0.5490714609077374 |
| LIBERO-10 | 0.3536560999394767 |

The medians and JOINT-vs-SEQ componentwise/Pareto comparisons are diagnostics
only. They cannot change the primary verdict.

## 6. Frozen typed verdict

Each GPU arm may report only `T10_SEQ_BRIDGE_VALID` or
`T10_JOINT_ARM_VALID`; these establish execution validity, not the scientific
outcome. A standard-library CPU aggregator first requires SEQ to reproduce all
eight frozen T7 pre and terminal action losses within
`1e-6 + 1e-6 * abs(expected)`. If it does not, the paired result is invalid.

For valid SEQ and JOINT roots, the ordered primary classification is complete:

1. If all eight unrounded JOINT ratios `rA_0..rA_3,rH_0..rH_3` are strictly
   below 1, emit
   `T10_BALANCED_JOINT_SIMULTANEOUS_RETENTION_SUPPORTS_OVERWRITE`.
2. Else if any JOINT training ratio `rA_i` is greater than or equal to 1, emit
   `T10_MATCHED_DOSE_JOINT_TRAINING_FIT_FAILURE`.
3. Otherwise emit `T10_JOINT_FIT_WITH_HELDOUT_GAP`.

There is no result-dependent extension, best-suite subgroup, threshold change,
or automatic rerun. Equality with 1 follows the non-success branch. A
simultaneous-retention verdict provides a capacity witness and supports the
operational sequential-overwrite explanation under this fixed screen. Either
failure verdict does not by itself distinguish gradient conflict from model
capacity.

## 7. Source and root contract

The new files are:

```text
docs/libero/LIBERO_T10_EXACT_BALANCED_JOINT_20260806.md
scripts/smoke_libero_ar_t10_exact_balanced_joint_gpu.py
tests/test_smoke_libero_ar_t10_exact_balanced_joint_gpu.py
scripts/summarize_libero_ar_t10_exact_balanced_joint.py
tests/test_summarize_libero_ar_t10_exact_balanced_joint.py
```

Source freezing is intentionally two-stage: the first commit freezes the
paired-arm runner, tests, and this pre-registration; the second freezes the CPU
aggregator and its tests after mechanically pinning the first commit and arm
runner SHA256. Neither GPU arm may start until both source commits have been
validated and pushed.

The runner has the closed arm choice `SEQ|JOINT`. Its only valid root shapes
are:

```text
/DATA/share/sana_wam_libero_nonformal_screens/t10/<source-commit-12>/libero-t10-seq-matched-core-fixed20-<nonce>
/DATA/share/sana_wam_libero_nonformal_screens/t10/<source-commit-12>/libero-t10-joint-matched-core-fixed20-<nonce>
```

The source commit, runner/config SHA256, nonce, physical GPU/UUID, predecessor
pins, counts, micro order, per-micro losses/signatures, aggregate gradients,
optimizer telemetry, raw measurements, metrics, and typed verdict must be
recorded in one canonical immutable RESULT per arm. The CPU aggregate uses a
fresh root below `t10_aggregate/<aggregate-source-commit-12>/`, strongly binds
both arm results and emits the only scientific typed verdict. A failed root is
terminal and is not reused.

## 8. Scope

T10 is a single-seed, fixed-sample, non-formal real-data loss-space screen. It
does not run a simulator, rollout, success-rate benchmark, formal training,
formal evaluation, checkpoint load/save, full-data training, admission, or
deployment. The SANA base construction weight remains the pinned base-model
input and is not a SANA-WAM training checkpoint.

At pre-registration authoring time no T10 run root, nonce, GPU assignment, or
result exists. Source commit and runner identities are filled only by the
two-stage freeze above; execution evidence must never be backfilled into those
source identities.
