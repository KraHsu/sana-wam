# LIBERO T2 Fixed-Sample Learnability Screen

Status: pre-registered for execution on 2026-08-06; no T2 result is claimed by
this document before a frozen terminal root exists.

## Question

T2 asks one narrow architecture question: after the full SANA-WAM AR model has
passed the LIBERO real-data construction/forward and one-update checks, can its
four admitted trainable roots reduce one fixed-seed, fixed-recipe real-sample action loss
across 20 consecutive optimizer updates?

This is a non-formal fixed-sample learnability screen. It is not benchmark
evaluation, generalization evidence, simulator execution, or formal training.

## Frozen execution scope

- Dataset/sample: LIBERO Spatial, episode 0, start frame 0, using the already
  pinned real Parquet/video assets and selected-row all-four-suite statistics.
- Model: the production-shaped autoregressive SANA-WAM architecture, with the
  published SANA base checkpoint as a construction asset.
- Fresh initialization seed: `20260806`.
- Fixed loss-recipe seed, reset before every forward: `20260826`.
- Optimizer: one persistent AdamW instance over one persistent set of FP32
  optimizer masters, constant group LR `1e-4`, betas `(0.9, 0.95)`, zero weight
  decay, and gradient clipping at the configured `1.0` bound.
- Budget: exactly 20 backward calls and 20 optimizer steps, plus one initial and
  one final update-free loss measurement, for exactly 22 architecture forwards.
- Device: one exclusively locked, idle H200 selected immediately before launch.
- Output: one new immutable terminal root containing `RESULT.json` on a valid
  run or `FAILED.json` on a harness/runtime failure. A failed root is never
  reused or overwritten.

Argument, source-identity, or device admission failures that occur before the
exclusive root is created intentionally leave no root. Once creation succeeds,
any terminal execution failure freezes `FAILED.json`; a valid run freezes
`RESULT.json`.

The runner may load only the SHA-pinned published SANA/Gemma/VAE construction
assets. It must not load, resume, or save a SANA-WAM training checkpoint. It
must not call `Trainer.train`, start a simulator, execute a benchmark evaluator,
or write model weights.

## Validity checks

A valid T2 run must establish all of the following:

1. The repository, config, runner, Sana gitlink, selected-row statistics,
   predecessor T1 result, and fixed Spatial sample assets match their pinned
   identities before CUDA/model construction.
2. The model has exactly the four admitted trainable roots, 560 trainable
   tensors, and 639,653,063 trainable elements; the video backbone remains
   frozen.
3. The prepared real sample is built once and is not mutated during the loop.
4. Every step has finite gradients, nonzero gradient coverage in all four
   trainable roots, and no frozen-parameter gradients. Full optimizer-state
   finiteness is checked after steps 1 and 20 and again at completion; sampled
   Adam counters are checked on every step.
5. All 560 FP32 masters receive finite nonzero gradients and update at step 1;
   their object identities persist, their Adam counters end at exactly 20, and
   projection back to BF16 is exact after steps 1 and 20.
6. Every forward observes the same captured stochastic recipe, and the initial
   measurement equals the first pre-update loss within the fixed numerical
   tolerance.

Each step records loss, gradient summaries, peak memory, wall time, sampled
FP32-master/BF16 update magnitudes, changed sentinel counts/roots, and sampled
Adam step values. The update magnitudes are explicitly sentinel measurements,
not claims of a global parameter-delta norm; the full step-1 and cumulative
coverage checks remain separate.

## Pre-registered interpretation

The valid-run verdict is `T2_FIXED_SAMPLE_LEARNABILITY_GO` only when both gates
hold:

- final fixed-probe action loss is at most 95% of the initial action loss, and
  the median of the last five pre-update losses is at most 95% of the median of
  the first five; and
- accumulated BF16-visible parameter changes cover every admitted trainable
  root, including `proprio_encoder`.

If the run is valid but either gate is absent, the verdict is
`T2_FIXED_SAMPLE_INCONCLUSIVE`. This is a scientific result, not a harness
failure, and it does not authorize an automatic rerun. Contract, identity,
finite-state, step-count, or runtime violations produce a frozen failed root.

Regardless of verdict, T2 alone does not authorize T3, formal training,
checkpoint production, simulator evaluation, or a benchmark claim.
