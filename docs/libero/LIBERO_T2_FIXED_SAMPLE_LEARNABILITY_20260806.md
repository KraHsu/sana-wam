# LIBERO T2 Fixed-Sample Learnability Screen

Status: completed on 2026-08-06 with a frozen valid-run verdict of
`T2_FIXED_SAMPLE_LEARNABILITY_GO`.

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

## Executed identity and evidence

- Source commit: `2dc1ce730df37bd9a2a71e4946f153380a8d4649`.
- Runner SHA256:
  `f5e5e3e80d6b5afd586e54e83a7e63dcf43280192363c70580aa03145267e9a6`.
- Config SHA256:
  `5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`.
- Sana gitlink/worktree:
  `16b9cec673e3335724ba2d8db25de7f9ed229292`.
- GPU: physical GPU 0,
  `GPU-1ec28cfb-f501-23f3-f865-275a744ca053` (`NVIDIA H200`).
- Frozen root:
  `/DATA/share/sana_wam_libero_nonformal_screens/t2/2dc1ce730df3/libero-t2-fixed20-20260806-a1`.
- `RESULT.json` SHA256:
  `67d250cdb13470bea9e9fa53531d3c76c65145e5040dc7a45079d84ac4960a57`.
- Terminal permissions: root `0500`, result file `0400`; there is no
  `FAILED.json`.
- Pre-execution validation: Ruff passed; all 67 `test_libero_*.py` CPU tests
  passed. The source commit was pushed before GPU execution.
- Post-process device check: GPU 0 returned to 0 MiB used and 0% utilization.

## Result

The run is valid and cleared both pre-registered gates:

- Initial fixed-probe action loss: `13.679718971252441`.
- Final fixed-probe action loss after exactly 20 updates:
  `1.7331912517547607`.
- Final/initial ratio: `0.12669786970017552`, an `87.3302%` endpoint decrease.
- First-five pre-update median: `355.40460205078125`.
- Last-five pre-update median: `4.356044292449951`.
- All 22 captured stochastic-recipe signatures were identical.
- All 560 FP32 masters had finite, nonzero gradients and changed on step 1.
  The 16 fixed master sentinels changed across all four roots on every one of
  the 20 steps, and every final Adam counter was exactly 20.
- Cumulative BF16-visible changes covered all four roots, including
  `proprio_encoder`: 3,269 of 4,479 recorded high-gradient probes changed.
- Exact master-to-BF16 projection checks passed after steps 1 and 20; all final
  parameters, optimizer masters, and Adam state were finite.

The complete pre-update action-loss curve was:

```text
13.679719, 3034.322754, 1479.573730, 355.404602, 6.262050,
123.381508, 70.202202, 2.925097, 54.153545, 56.771427,
17.039402, 3.471757, 9.951302, 5.143774, 1.506911,
4.825734, 1.688791, 4.484559, 4.356044, 1.641986
```

The 20 updates plus two measurements took `32.0458 s`; model/dataset build took
`38.4700 s`. Peak update allocation/reservation was 26.94/28.42 GB
(25.09/26.47 GiB), well within the selected H200.

## Interpretation and remaining risk

T2 proves the full production-shaped AR architecture, its four trainable roots,
the real LIBERO sample path, FP32 optimizer masters, and BF16 projection can
perform sustained numerical learning on one fixed sample. It closes the narrow
single-sample learnability question.

It does not establish stable optimization. The first update raised the next
fixed loss from `13.68` to `3034.32`, the curve remained highly oscillatory,
and every recorded pre-clip gradient norm exceeded the `1.0` clip bound
(`66.5` minimum, `24704` maximum). The first-five median is therefore inflated
by early spikes; the independent endpoint decrease is strong, but the median
gate should not be read as smooth convergence. Eight of the 20 update outcomes
increased the fixed loss, and the twentieth update raised it from `1.641986` to
`1.733191` (`5.55%`). The more conservative last-five/initial median ratio was
still `0.3184`, so the endpoint conclusion does not depend only on the inflated
first-five median.

The smallest useful successor should keep the same sample, fresh
initialization, LR, 20 updates, and training loss recipe unchanged, while adding
update-free before/after probes for three pre-frozen held-out loss-recipe seeds.
That isolates whether T2 learned the real action target or merely one noise
realization. Only after that transfer check should a separate experiment vary
LR/gradient scale or expand to multiple samples; those variables should not be
changed together.

Regardless of verdict, T2 alone does not authorize T3, formal training,
checkpoint production, simulator evaluation, or a benchmark claim.
