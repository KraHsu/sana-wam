# LIBERO T4 Held-Out Sample Transfer Screen

Status: pre-registered for execution on 2026-08-06. This document claims no
T4 result before a fresh terminal root has been frozen.

## Question

T3 showed that the fixed 20-update LIBERO path transfers from its training
loss recipe to three gradient-held-out recipes on the same real window. T4
asks the next single-axis question: do those unchanged updates also reduce
action loss on three update-held-out episodes of the same Spatial task? These
episodes remain members of the dataset and normalization population; T4 is not
a strict dataset-holdout test.

T4 remains a non-formal, single-initialization architecture screen. It is not
simulator or benchmark evaluation, and it does not test a new task, suite,
training schedule, or checkpoint.

## Frozen predecessor and unchanged training core

- T3 frozen result:
  `/DATA/share/sana_wam_libero_nonformal_screens/t3/ac431f8727ac/libero-t3-heldout3-fixed20-220afb60725d0cfd591bc4fe225cdd21/RESULT.json`.
- T3 result SHA256:
  `c883608f2a47b6258f824d4d97a94f8a390d03bab671a592fb758eea61b3a01e`.
- T3 source commit:
  `ac431f8727ac345c3bd0ad4a442e53801310fd81`.
- T3 runner SHA256:
  `0db0bfe086bc4b5ea92462e4832f0c448408656cca819796ec253f6226b4c017`.
- Config SHA256:
  `5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`.
- Sana gitlink/worktree:
  `16b9cec673e3335724ba2d8db25de7f9ed229292`.

The update path remains T2/T3 exactly: LIBERO Spatial episode 0/start 0,
fresh initialization seed `20260806`, training loss-recipe seed `20260826`,
four admitted trainable roots, frozen video backbone, action-only loss, one
persistent AdamW over persistent FP32 masters, LR `1e-4`, betas `(0.9, 0.95)`,
zero weight decay, clip bound `1.0`, BF16 projection, and exactly 20 updates.
T4 adds no training sample, held-out recipe, LR arm, scheduler, warmup,
accumulation, early stopping, checkpoint, simulator, or evaluator.

## Mechanical held-out sample selection

The source metadata are frozen as follows:

- Dataset: `libero_spatial_no_noops_1.0.0_lerobot`.
- Training task index: `0`.
- Task: `pick up the black bowl next to the cookie box and place it on the plate`.
- `meta/episodes.jsonl` SHA256:
  `690349688e96d984007bf2cca3ccf2683dfc809f03dab55e1159e4cc71f150b7`.
- `meta/tasks.jsonl` SHA256:
  `399841cf8e861052d3fb7214ed619b5a10b4eded2d6f107e7fed338a233dd0e1`.

An eligible row must have that exact dataset and single task, resolve to task
index 0, differ from training episode 0, use start frame 0, and occur exactly
once as `(episode, start=0)` in the dataset window registry. The 45 eligible
rows, ordered by episode index inside the frozen manifest and serialized as
UTF-8 compact sorted-key JSON without a trailing LF, have SHA256
`e77462aa3380b1b14bee646b0965eb465baca76bff3d0805f9fc7d54d61419ef`.

For every eligible episode, the ASCII ranking payload, including its final LF,
is:

```text
SANA-WAM/LIBERO/T4_HELDOUT_SAMPLE_V1
T3_RESULT_SHA256=c883608f2a47b6258f824d4d97a94f8a390d03bab671a592fb758eea61b3a01e
DATASET=libero_spatial_no_noops_1.0.0_lerobot
TASK_INDEX=0
EPISODE_INDEX=<decimal episode index>
START_FRAME=0
```

The three lexicographically smallest payload SHA256 values define H1, H2 and
H3 in that order. They may not be reordered or replaced after any loss or
runtime outcome is observed.

| ID | Episode | Length | Ranking payload SHA256 |
| --- | ---: | ---: | --- |
| H1 | 16 | 111 | `02938329ed205ce9032b9bc7c54d0e3f1f1716d86b8526bdc3e220bd784b6c6e` |
| H2 | 405 | 130 | `09d59c3d6bec297af4f0d04d034d0f8ba65b2db2552f93cf8325946b1e6e3c72` |
| H3 | 40 | 141 | `0d8d7c4a8f60985b3b0d2259bffae34b99c911bce2757b7559294e02e229fef8` |

The selected raw assets are frozen independently:

| ID | Parquet SHA256 | Head-camera video SHA256 | Wrist-camera video SHA256 |
| --- | --- | --- | --- |
| H1/episode 16 | `6e440ab8884337435fae9b0868a0b3911fe8dca6be00ba1061f343e666d130db` | `cef539255b3e7322e6b4e37ef4a3fc5c9d9946685460c3e5f7bde772f1e1e544` | `9f6cf8979f6ab33a87084443218e86d18f27d29118d8880a338d4f4b8d60d6a1` |
| H2/episode 405 | `0887ba7d42fdfbe26bc2402055ab75073a3bc860408f3306dabc896f9d2a1611` | `46be6fe35ecf534cb64a259ff33b1bc19756180628c6859cd1e9272d1e92daf6` | `e6be28d3d32d836530643d92919340afcffd02747bcfb743511898cee7321f3e` |
| H3/episode 40 | `4b2f24a8818e18c489b041e4292553c0f76ca6f121de18bd695b0dd440806617` | `9d4d5a9f2082020193d7576f2174314fd831169b8086de59d125b11f2ff70650` | `b02619c14008af3dae2c39a8762d197cd19d3599d7a80d51553c5550713c7b9d` |

All metadata and selected assets must be regular non-symlink files and match
these identities before CUDA/model construction.

## Preparation, exact order and budget

The training episode is prepared first, preserving the T3 path. H1, H2 and H3
are then prepared once each. Forward recipes fork and restore caller RNG state,
so the extra preparations cannot alter the fixed training recipe. The frozen
T3 training-core equality check independently verifies that the training input
and all non-timing update telemetry remain identical. The four prepared input
mappings must be independent and immutable.

All probes use loss-recipe seed `20260826`, the T3 training/module modes, and
the same action-only loss. The execution order is exactly:

1. H1, H2 and H3 pre-update probes, with no backward or update.
2. Training-sample initial probe.
3. Twenty training-sample forward/backward/clip/master-sync/AdamW/BF16-project
   updates, preserving the T3 order exactly.
4. Training-sample final probe immediately after update 20.
5. H1, H2 and H3 post-update probes, with no backward or update.

The budget is exactly 28 architecture forwards: 20 training forwards and
eight measurements, of which six are held-out-sample probes. There are exactly
20 backward calls, 20 optimizer steps, four `prepare_inputs` calls, one
optimizer, and one persistent FP32-master set. No held-out tensor may enter a
backward call or parameter update.

## Validity and T3 core-reproduction gate

A valid run must retain all T3 source, asset, model, device, finite-state,
gradient-root, optimizer-master, Adam-counter, BF16-projection and exact-count
checks. In addition:

1. Every sample must match its frozen dataset/task/episode/start/length and raw
   asset identities after loading.
2. Every prepared tensor must retain its version; held-out inputs must not
   alias the training inputs or one another.
3. Each pre-probe and post-probe block must leave model/master values and
   identities, buffers, module modes, gradients and optimizer state
   unchanged. Before training, optimizer state remains empty; after training,
   every Adam counter remains exactly 20.
4. Action/video recipe capture must be non-empty. For each sample, pre/post
   captured signatures must match, and every forward must restore caller
   Python, NumPy, Torch CPU and Torch CUDA RNG states. Since every sample uses
   the same seed and fixed shapes, signatures need not be distinct by sample.
5. All losses must be finite and non-negative, every held-out pre loss must be
   strictly positive, and total/action loss equality must meet the frozen T3
   tolerance.

The attribution gate additionally requires the unchanged training core to
reproduce frozen T3. Initial/final loss, both complete 20-point loss curves,
and all non-time per-step loss, gradient, recipe, sampled FP32/BF16 update and
Adam-counter fields must equal the SHA-pinned T3 result. Timestamps, wall time,
memory telemetry and GPU utilization are excluded. A core mismatch may not be
interpreted as held-out-sample transfer and must prevent a GO verdict.

## Pre-registered scientific verdict

For each held-out sample define
`r_i = post_action_loss / pre_action_loss` and
`improved_i = post_action_loss < pre_action_loss`. Ratios, not raw-loss means,
are used because initial loss scales may differ by episode. The primary gate is
exactly:

```text
median(r_1, r_2, r_3) <= 0.95 AND sum(improved_i) >= 2
```

- A valid run satisfying both the T3 core-reproduction gate and the primary
  gate is `T4_HELDOUT_SAMPLE_TRANSFER_GO`.
- A valid run whose T3 core reproduces but whose primary transfer gate misses
  is `T4_HELDOUT_SAMPLE_TRANSFER_INCONCLUSIVE`; it does not authorize sample
  replacement or an automatic rerun.
- Training-core drift, identity, sample-selection, state-mutation, count,
  finite-value or runtime violations produce no scientific verdict and freeze
  `FAILED.json`.

The result must report every paired pre/post loss, absolute delta, ratio and
improvement flag; sorted ratios, median and improved count; the core comparison;
and the inherited T2/T3 overshoot and clip-active diagnostics. Best/minimum
loss, post-hoc sample selection, reranking and success-rate claims are forbidden.

## Fresh-root and execution boundary

Every execution requires a never-created absolute root under
`/DATA/share/sana_wam_libero_nonformal_screens/t4/<execution-commit-12>/`.
Its basename is
`libero-t4-heldout3-fixed20-<32-lowercase-hex-nonce>` and must bind the same
fresh nonce supplied to the runner. The execution card must freeze the exact
root before launch. No existing, failed or valid root may be reused, removed,
renamed or overwritten.

Argument, source/device admission or path failures before exclusive root
creation leave no root. After creation, a terminal execution failure writes
and freezes only `FAILED.json`; a valid scientific result writes and freezes
only `RESULT.json`. Terminal files and directories use the established
read-only `0400`/`0500` convention. There is no checkpoint or token namespace.

The runner may load only the pinned SANA/Gemma/VAE construction assets. It may
not load, resume or save a SANA-WAM checkpoint, call `Trainer.train`, train on
H1/H2/H3, run a simulator or benchmark evaluator, create a formal-training
root, or vary any non-sample variable. T4 does not itself authorize execution
or a successor experiment.

## Interpretation limits and successor logic

A GO means only that, under one initialization and one fixed stochastic recipe,
20 updates on Spatial task-0 episode 0 improved at least two of three frozen
episodes of the same task. It is evidence of within-task, cross-episode loss
transfer. The probes are update-held-out only: they are drawn from the same
training-distribution metadata and are included in the frozen normalization
population. It is not strict dataset holdout, cross-task or cross-suite
evidence, a joint new-sample and new-recipe factorial test, rollout success,
benchmark performance, stable optimization, or formal training. The three
samples become consumed evidence after their pre losses are revealed.

If T4 is GO, the next smallest independent axis is a same-suite cross-task
probe. If the core reproduces but T4 is inconclusive, the next architecture
question is fixed four-sample cyclic micro-learnability, using these consumed
episodes as training data and new mechanically selected episodes for
measurement. If the core does not reproduce, no sample-transfer conclusion is
allowed; only the root cause may be diagnosed before a separately authorized
fresh-root run.
