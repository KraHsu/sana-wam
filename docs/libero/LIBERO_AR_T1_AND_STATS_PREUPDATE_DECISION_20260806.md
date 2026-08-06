# LIBERO AR T1 and statistics pre-update decision — 2026-08-06

## Decision

The first full-2B LIBERO `forward -> backward -> AdamW step` architecture smoke
will use continuous FP32 video timestep conditioning (T1). The source template
now sets `model.video_backbone.continuous_timestep_conditioning: true`
explicitly. T1 is a checkpoint-lineage contract: a run that starts with T1 must
retain it through deployment; it is not an inference-only switch.

This choice authorizes no GPU execution or training by itself. It selects the
path for a separately authorized, one-update, non-formal architecture smoke.

## Evidence and its limit

The frozen RoboTwin Phase-6 offline aggregate is:

- Path: `/DATA/share/sana_phase6_principled_constraints_20260724/offline_qualification_v1/formal_execution_v1/aggregate/offline_qualification_aggregate_v1.json`
- SHA256: `663b536d235f35cd728a70369d11409d36a3b9a4d9776210422286a3e88db099`
- Mode: `0400`
- Formal observations: `224`
- Evidence scope: `offline held-out and teacher-forced only`

For `T1_E0A0 - T0_E0A0`, where lower is better, the aggregate reports:

| Metric | Mean | 95% CI |
|---|---:|---:|
| action MSE | `-3.03058e-5` | `[-7.28966e-5, -4.33717e-6]` |
| step-10 latent energy | `-1.88029e-4` | `[-4.10170e-4, -3.50828e-5]` |
| step-10 coarse-motion energy | `-1.48680e-4` | `[-2.96463e-4, -5.32531e-5]` |

The continuous-time gate passed. The overall aggregate nevertheless failed its
action gate and explicitly granted no closed-loop permission. These data are
small RoboTwin offline effects, not LIBERO performance or success-rate
evidence.

T1 preserves fractional video diffusion coordinates in FP32 and adds no
parameters or state-dict keys. Action timesteps remain BF16. The important
compatibility risk is that the published SANA base is legacy T0, whereas the
Phase-6 evidence used video parameters adapted under T1. The LIBERO template
currently freezes the video backbone and trains only action/proprioception
modules. The one-update smoke can establish numerical and gradient closure for
that exact initialization, but only later benchmark evidence can establish
whether it is effective.

## Statistics boundary

The read-only H200 artifact
`/DATA/share/LIBERO/sana_wam_libero_all_minmax_stats.npy`, SHA256
`333b2cb1e150b451ee1cf6833b1e628ebb50914a466f2cb1fa1808cff8e9f2d0`, exactly
reproduces the earlier temporary metadata-bootstrap artifact. It contains four
suite aggregates for 1,693 episodes and 273,465 rows. It is adequate for a
non-formal architecture smoke, but it is not the exact selected training
population because Goal episode 82 is excluded by the dataset contract.

The actual selected population contains 1,692 episodes, 273,336 unique state
rows, and 271,644 unique supervised action rows (`[0, L-1)` per episode). The
source training template therefore reserves a distinct, currently absent path:

```text
/DATA/share/LIBERO/sana_wam_libero_train_all4_excl_goal82_stats_v2.npy
```

`training.action_stats_sha256` remains null. Before formal training, a
config-driven producer must recompute min/max, mean/std, and quantiles from the
unique selected Parquet action/state rows, bind the split/seed/exclusion and
episode identities, atomically publish the fresh artifact, and pin its SHA.
The existing `333b...` artifact is retained unchanged and must not be
overwritten, renamed as production evidence, or pinned for training.

## One-update smoke assertions

The next runner must fail closed unless all of the following hold:

- Config and live video backbone both report T1.
- Video timesteps reach the SANA timestep embedder as FP32 and include a
  fractional value; action timesteps remain BF16.
- Exactly the configured action/proprioception module allowlist is trainable;
  the video backbone and all other parameters remain frozen.
- One real LIBERO sample produces a finite loss, finite nonzero gradients, and
  one observable AdamW parameter update.
- No checkpoint is loaded or saved beyond the published SANA base asset; no
  simulator, benchmark evaluation, multi-step training loop, or formal run is
  started.

The runner directly hashes the SANA DiT and Wan VAE files, the Gemma model and
tokenizer files used by construction, and the fixed Spatial episode-0
Parquet/metadata/two-camera MP4 inputs before reserving a GPU. It also rejects
both `training.init_checkpoint` and `model.video_backbone.init_dit_from`.
Successful or failed future roots freeze as owner-read-only files (`0400`) in
an owner-read-and-enter-only directory (`0500`). These checks establish input
identity for this narrow smoke; they do not turn it into benchmark evidence.

The runner deliberately uses one microbatch and the configured positive base
learning rates (`1e-4`) without the production warmup scheduler. It therefore
tests parameter-update wiring, not equivalence to the template's eight-way
gradient accumulation or its zero-LR first warmup step. The result must report
both equivalence flags as false.

## CPU/static runner readiness

The T1 decision, strict contract check, one-update runner, and related LIBERO
regressions passed `138` CPU tests on H200; Ruff also passed. No GPU/model
construction or parameter update occurred during this verification.

A deliberate pre-GPU source-identity mismatch exercised terminal failure at:

```text
/tmp/sana-wam-libero-one-update-runner-failclosed-test-20260806-a1
```

The root contains only `FAILED.json`; root/file modes are `0555/0444`, and the
marker SHA256 is
`4851d5137991c39eed77580850c12dbff0506d6c687d7291d10cf08f630e231a`.
Because the identity check failed before the GPU lock/query and Torch import,
this test did not use CUDA. The `/tmp` root is non-formal and ephemeral.
It predates the runner's later `0500/0400` freeze tightening and remains
unchanged as historical failure evidence.
