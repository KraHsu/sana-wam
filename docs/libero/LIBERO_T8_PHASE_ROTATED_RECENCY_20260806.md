# LIBERO T8 Phase-Rotated Recency Screen

Status: **executed once / valid non-formal screen / mixed inconclusive**

Date: 2026-08-06

Sections 1-13 preserve the complete T8 pre-registration. Section 14 records the
single authorized execution without changing any frozen predicate or threshold.
This document authorizes no successor run by itself.

## 1. Scientific question

T7 established a qualified four-suite cyclic loss-space GO, but its terminal
phase left Spatial as the least-recently updated sample. Spatial then regressed
on both its update sample A0 and its same-task, update-free probe H0, while the
other three suite pairs improved. T8 asks one deliberately narrow question:

> Does rotating only the phase of the same four-sample cycle, so that Spatial
> receives the final update, rescue both Spatial endpoints; and do probes taken
> immediately after each suite's fifth update show a terminal overwrite pattern
> consistent with update recency rather than a fixed suite ordering?

The T7 schedule was exactly:

```text
[A0 Spatial, A1 Object, A2 Goal, A3 LIBERO-10] x 5
```

The sole scientific intervention in T8 is:

```text
[A1 Object, A2 Goal, A3 LIBERO-10, A0 Spatial] x 5
```

This is a cyclic phase rotation, not a permutation of cyclic adjacency. The
directed cyclic neighbor relations remain A0 -> A1 -> A2 -> A3 -> A0. The
rotation moves Spatial's last direct update from step 17 to step 20 and moves
Object's last direct update from step 18 to step 17. Total optimizer dose,
per-sample dose, data, initialization, stochastic recipe, model, optimizer,
and terminal measurement order remain fixed. T8 additionally inserts one
update-free A/H phase-probe pair immediately after each suite receives its
fifth update. Those probes are measurement instrumentation, not extra training;
their state-preservation gates make them unable to change the update path.

T8 is a single-initialization, real-sample, non-formal loss-space architecture
screen. It is not formal training, admission, checkpoint production, simulator
execution, rollout, or LIBERO benchmark evaluation.

## 2. Frozen T7 predecessor and evidence boundary

Before CUDA or model construction, the T8 runner must verify the following
immutable direct predecessor:

- T7 RESULT root:
  `/DATA/share/sana_wam_libero_nonformal_screens/t7/d19109a2314f/libero-t7-foursuite-cyclic-fixed20-61e0a0ff817970e994b0875be4840ed6`;
- T7 RESULT file: `<root>/RESULT.json`;
- T7 RESULT SHA256:
  `9f5181a30cd0d6b676f09315244f7560cd3902004f5f17ca833330e33cb44d3a`;
- T7 source commit:
  `d19109a2314f8e7186571afed6b4acd816d7cfab`;
- T7 runner SHA256:
  `b6e764bf3d3566bf3d1fe5e2e3802dafef691f6a0162568337b49ca1a183310b`;
- T7 typed verdict: `T7_FOUR_SUITE_CYCLIC_GO`;
- T7 documentation state: post-run audit qualified, not formal admission.

T8 must preserve and independently verify the T7 RESULT's complete frozen
predecessor/source/asset/config/statistics/normalization/recipe lineage. The
most important direct pins are:

- reused config:
  `configs/benchmarks/libero/train_libero_ar_baseline.yaml`;
- config SHA256:
  `5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`;
- Sana gitlink/worktree:
  `16b9cec673e3335724ba2d8db25de7f9ed229292`;
- selected-row all-four-suite stats SHA256:
  `e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7`;
- selected normalization population SHA256:
  `7ed9772facf261299022e55169bcdaaa49fe3a7a20e0057e08cf419b5e584146`;
- frozen loss-recipe signature SHA256:
  `ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad`;
- T7 update-sample manifest SHA256:
  `5199c87af876c437ecec35b57da3e118a2924c44a9bb454c9846e7c0ca968aa6`;
- T7 same-task eligible manifest SHA256:
  `0b6bfe4c51e8b9e2ac3f538bac7853b5e1e871ea96dda5df6976ee27b5b25370`;
- T7 same-task selected manifest SHA256:
  `e969e76e4bcb8a3c2b0b3fffff35e0ec10c478e45cb6535b7bd91061d3a6a7ee`.

The configuration hash is a source constant in the T8 harness. It may not be
supplied as a caller-selected expected hash. A mismatch fails before CUDA.

T7's immutable RESULT remains evidence and must not be edited. Its qualified
GO establishes a numerical predecessor, not formal-training admission. T8 may
load only the same pinned offline SANA/Gemma/VAE base construction assets that
T7 admitted. It must not load the T7 post-update model, any SANA-WAM training
checkpoint, or any unlisted parameter state.

## 3. Fixed model and optimization system

The model remains the production-shaped `DualSystemARArchitecture`. The SANA
video backbone remains frozen. The complete trainable-root allowlist remains:

- `action_backbone`;
- `proprio_encoder`;
- `proprio_video_embed`;
- `proprio_action_embed`.

No architecture module, tensor shape, loss term, precision policy, optimizer
hyperparameter, initialization rule, preparation logic, prompt format, or
dataset configuration may differ from frozen T7. The fixed recipe is:

- fresh initialization seed `20260806`;
- recipe seed `20260826` before every forward;
- action loss weight 1.0 and video loss weight 0.0;
- one persistent FP32-master AdamW optimizer over the four trainable roots;
- learning rate `1e-4`, betas `(0.9, 0.95)`, weight decay 0;
- global gradient clip bound 1.0;
- BF16 trainable-model projection after every optimizer step;
- no scheduler, warmup, accumulation, batching, resampling, shuffle, early
  stop, best-step selection, retry, or SANA-WAM training-checkpoint load/save.

All 560 FP32 masters must be initialized by exact full-tensor equality to the
corresponding 560 BF16 trainable tensors cast to FP32. The optimizer state must
be empty before step 1, and the same optimizer and master objects must persist
through step 20.

## 4. Immutable update samples

The complete update set is the same four real samples as T7. No other sample
may contribute to backward or optimization.

| Label | Dataset | Task / episode / start | Task | Length |
|---|---|---|---|---:|
| A0 | `libero_spatial_no_noops_1.0.0_lerobot` | 0 / 0 / 0 | `pick up the black bowl next to the cookie box and place it on the plate` | 110 |
| A1 | `libero_object_no_noops_1.0.0_lerobot` | 3 / 82 / 0 | `pick up the bbq sauce and place it in the basket` | 135 |
| A2 | `libero_goal_no_noops_1.0.0_lerobot` | 2 / 70 / 0 | `open the top drawer and put the bowl inside` | 249 |
| A3 | `libero_10_no_noops_1.0.0_lerobot` | 3 / 259 / 0 | `turn on the stove and put the moka pot on it` | 228 |

The update assets are frozen exactly:

| Label | Relative asset | SHA256 |
|---|---|---|
| A0 | `data/chunk-000/episode_000000.parquet` | `3f875604fad478765549128759edfb33a64b69b7b82decebc9e4f38155f20a8c` |
| A0 | `videos/chunk-000/observation.images.image/episode_000000.mp4` | `eba9a9b36611f7e1061f65233b3b0c53eaa3f9c347dfd073f7470a821695fdf7` |
| A0 | `videos/chunk-000/observation.images.wrist_image/episode_000000.mp4` | `cf4f97b9405e6e06d42f3816c902a7cec22ba0497dc4ea3fa201729302ba4168` |
| A1 | `data/chunk-000/episode_000082.parquet` | `f25ac74111c22c9db0ff96d6021484f8a0e31d2e4413d47a160b83a0489cd1e9` |
| A1 | `videos/chunk-000/observation.images.image/episode_000082.mp4` | `38294e54072c35ff4fc7edf963b688bd90d463e4a1ba623aca680ce2106b6ee7` |
| A1 | `videos/chunk-000/observation.images.wrist_image/episode_000082.mp4` | `f935b70c9925f1a57bc4df7aa8c6519c614df2ece97661fa18d12d0a3788a923` |
| A2 | `data/chunk-000/episode_000070.parquet` | `6ce3cf58f533871b6413ea0e795f302060aca3a87b041f0110c77494a6d4b66b` |
| A2 | `videos/chunk-000/observation.images.image/episode_000070.mp4` | `af716b54f26f8cd284e715ab2335ecf86c59b5cc89706a7c51165094a5899b2b` |
| A2 | `videos/chunk-000/observation.images.wrist_image/episode_000070.mp4` | `ded7a8d03de99df3c720c3e74f87403378818a92f1cbea6e37a4d37d11fe1027` |
| A3 | `data/chunk-000/episode_000259.parquet` | `de9e09a471d8050a5624f016f765c61fba7514829990b59de37c4e3bae0eec37` |
| A3 | `videos/chunk-000/observation.images.image/episode_000259.mp4` | `c11a5751444371331df5b24ab76b85212e2979636d803afd2e0f59a002bd0804` |
| A3 | `videos/chunk-000/observation.images.wrist_image/episode_000259.mp4` | `8800f488e8cc6f71cdfdf66d6fc531f92c54f54edf55da1403c60cb064f91786` |

## 5. Immutable update-free probes

H0-H3 remain the same mechanically selected same-task episodes as T7. They
are update-held-out, not strict dataset or normalization holdouts, and are
already consumed evidence. They may be measured before and after the update
loop but must never enter backward, clipping, optimizer state, master updates,
or BF16 projection.

| Label | Dataset | Task / episode / start | Task | Length | Selection payload SHA256 |
|---|---|---|---|---:|---|
| H0 | `libero_spatial_no_noops_1.0.0_lerobot` | 0 / 30 / 0 | same as A0 | 125 | `0a33e0f8df2afd9f256cba15af973cc4b5f7dde3ddbc0b5866f454a535e4e83d` |
| H1 | `libero_object_no_noops_1.0.0_lerobot` | 3 / 166 / 0 | same as A1 | 129 | `0378af218d388de30af08e0be5c417dfb5a7275939dca80da5b1a15d9c841df1` |
| H2 | `libero_goal_no_noops_1.0.0_lerobot` | 2 / 248 / 0 | same as A2 | 184 | `02c53f50cf758bc8185dbedbdfc5b7c05d297389b5fd55e04d738e0f65856a94` |
| H3 | `libero_10_no_noops_1.0.0_lerobot` | 3 / 278 / 0 | same as A3 | 261 | `125e02f94d4e09f2216fbf649869d6bd3e1f0c43044992d031abc4dd8454b974` |

Their frozen assets are:

| Label | Relative asset | SHA256 |
|---|---|---|
| H0 | `data/chunk-000/episode_000030.parquet` | `fa8fd7f972b1567af1e8cd290af80649dbf64dc4f14dacb2b12e80ebba62489d` |
| H0 | `videos/chunk-000/observation.images.image/episode_000030.mp4` | `c93bb2ac9d565fff5e6939eebebf0ee34a73f6fa037510f8c27877943b9a09cc` |
| H0 | `videos/chunk-000/observation.images.wrist_image/episode_000030.mp4` | `2ae055381b99944dee3ef4d4e1b07d396e8723fa5616e5544d3a3dad8e64658c` |
| H1 | `data/chunk-000/episode_000166.parquet` | `ce386f7027088fb4d9648d7b8acaecc7fac8ed3150dceb5882659985a2d47b3a` |
| H1 | `videos/chunk-000/observation.images.image/episode_000166.mp4` | `645069ae5c04c1cf2ce8ab3891fecafbb9704db0e5bbee5f507117e595f23a4d` |
| H1 | `videos/chunk-000/observation.images.wrist_image/episode_000166.mp4` | `c48ca15a905b46e76ab7b7afa15e26611bcad182fd94e74b5382a532a1e1f6ed` |
| H2 | `data/chunk-000/episode_000248.parquet` | `2511639d496676d6c670c7143bfa331e185a4e41b407c8ccd4b8e4d32713a880` |
| H2 | `videos/chunk-000/observation.images.image/episode_000248.mp4` | `5f1803eb0dfe3a0e7f2b92cd8cfb77c6febfdf798812eb87fe9ca31405a2edad` |
| H2 | `videos/chunk-000/observation.images.wrist_image/episode_000248.mp4` | `855bfd92540b233e1352a540f6838295a1bb441aa81ebaa99e66313fb94775ba` |
| H3 | `data/chunk-000/episode_000278.parquet` | `369834fda56e67653a5273b6c07f46c70b87fdbc7f4361bf2a9e94e68bf05440` |
| H3 | `videos/chunk-000/observation.images.image/episode_000278.mp4` | `3ecd041eccfe14e5ef87abb4533b671ac1f3577d52a9474dfbda65dba058926b` |
| H3 | `videos/chunk-000/observation.images.wrist_image/episode_000278.mp4` | `b55954d2eeb821a4d737fb4b57a809a4cf55f31bb9d757845af85e07b9052fe5` |

The metadata pins used to rebuild the unchanged T7 manifests are:

| Suite | `tasks.jsonl` SHA256 | `episodes.jsonl` SHA256 | Fixed task / eligible count |
|---|---|---|---|
| Spatial | `399841cf8e861052d3fb7214ed619b5a10b4eded2d6f107e7fed338a233dd0e1` | `690349688e96d984007bf2cca3ccf2683dfc809f03dab55e1159e4cc71f150b7` | 0 / 42 |
| Object | `68ef5f9bc5a0bd74f46140f6721fa0ea74e997d74e37b8714a539f61337e7862` | `63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea` | 3 / 45 |
| Goal | `39f08f81b289ad3041f1c8ada88f679fe60774e9fde4083415881486edc23d55` | `548d91fe48b7d439248523dd3f7a5e4b15fc77d5eb1b7cfdd6da0033d422cb43` | 2 / 35 |
| LIBERO-10 | `45f9eb4d4b6b04999f64640c0aae380555372b7b273a904f5f459ad05d4a0a6a` | `5589f8f87cfddb34812782462160bf55b0d3082e404240682d1d0a89faba8265` | 3 / 40 |

All metadata and raw assets must be regular non-symlink files. They must match
their pins before CUDA and be re-hashed after all eight samples are
materialized. The runner must rebuild both unchanged T7 selection manifests
and require exact canonical hashes and counts before CUDA, then independently
confirm the 162 eligible dataset-qualified rows after dataset construction.

## 6. Exact starting-state projection

Fresh initialization must reproduce this exact frozen eight-dimensional T7
pre-action-loss target, in label order A0, A1, A2, A3, H0, H1, H2, H3:

```json
{
  "A0": 13.679718971252441,
  "A1": 14.696584701538086,
  "A2": 12.06648063659668,
  "A3": 20.43115234375,
  "H0": 13.379679679870605,
  "H1": 14.698094367980957,
  "H2": 12.026171684265137,
  "H3": 20.885684967041016
}
```

These eight numbers are fixed pins, not values learned from a T8 run. The
runtime equality gate uses the already frozen numerical tolerance
`abs(observed - expected) <= 1e-6 + 1e-6 * abs(expected)` for each component.
The report must serialize expected, observed, absolute error, allowed error,
and pass/fail for every label. No component may be dropped or substituted.

This vector, the source/config/Sana/seed/recipe lineage, the exact trainable
root names and sizes, empty optimizer state, and exact within-run FP32-master
initialization equality jointly define the T8 starting-state gate. T7 did not
record a bytewise fresh-model fingerprint, so T8 does **not** require or claim
a bytewise cross-run model hash. It also does not initialize from or claim
equality to the T7 post-update state.

## 7. Phase-rotated schedule and exact budget

Preparation consists of exactly eight independent single-sample calls in
unchanged order: A0, A1, A2, A3, H0, H1, H2, H3. Each prompt must equal
`format_prompt_for_inference(task_name)`. Prepared mappings and all contained
tensors must remain independent, non-aliasing, immutable, and unreprepared.

The complete execution order is:

1. update-free A0, A1, A2, A3 pre probes;
2. update-free H0, H1, H2, H3 pre probes;
3. begin the 20 forward/backward/clip/master-sync/AdamW/BF16-project updates
   under the rotated cycle `[A1,A2,A3,A0] x 5`, interleaving only the phase
   probes specified next;
4. immediately after step 17, before step 18, update-free A1 then H1 phase
   probes;
5. immediately after step 18, before step 19, update-free A2 then H2 phase
   probes;
6. immediately after step 19, before step 20, update-free A3 then H3 phase
   probes;
7. immediately after step 20, update-free A0 then H0 phase probes;
8. without any intervening update, re-preparation, or state mutation, the
   unified terminal block probes A0, A1, A2, A3, then H0, H1, H2, H3.

| Sample | T8 optimizer steps | T7 optimizer steps | Terminal-recency change |
|---|---|---|---|
| A0 Spatial | 4, 8, 12, 16, 20 | 1, 5, 9, 13, 17 | newest instead of oldest |
| A1 Object | 1, 5, 9, 13, 17 | 2, 6, 10, 14, 18 | oldest instead of second-oldest |
| A2 Goal | 2, 6, 10, 14, 18 | 3, 7, 11, 15, 19 | second-oldest instead of second-newest |
| A3 LIBERO-10 | 3, 7, 11, 15, 19 | 4, 8, 12, 16, 20 | second-newest instead of newest |

The exact budget is:

- 8 `prepare_inputs` calls;
- 44 architecture forwards: 8 pre probes, 20 update forwards, 8 phase probes,
  and 8 unified terminal probes;
- 20 backward calls;
- 20 optimizer steps;
- one persistent optimizer and one persistent FP32-master set;
- exactly five update forwards for each A sample;
- exactly one phase probe and one terminal probe for each A and H sample;
- zero H tensors in backward or update;
- zero re-preparation and zero SANA-WAM training-checkpoint load/save
  operations.

The step-20 A0/H0 phase pair and the subsequent unified-terminal A0/H0
measurements observe the same parameter, master, optimizer, buffer, prepared
input, module-mode, gradient, and RNG state. Their corresponding total/action
losses must therefore reproduce exactly. This deliberate duplicate is a
measurement-path and state-preservation check, not independent evidence.

Every forward resets recipe seed `20260826` while forking and restoring caller
Python, NumPy, Torch CPU, and Torch CUDA RNG states. All 44 recipe signatures
must equal the T7-pinned signature. There is no step 21 and no automatic
extension or fresh-root retry based on outcome.

## 8. Strengthened validity and attribution gates

A scientific classification is allowed only if every condition below passes:

1. The frozen T7 RESULT, source, runner, direct configuration, Sana gitlink,
   base construction assets, statistics, normalization population, metadata,
   manifests, update assets, and held-out assets match their pins.
2. The T8 source hard-codes the config path and SHA256 shown in Section 2;
   command-line substitution of the expected config identity is forbidden.
3. All eight sample identities, task texts, prompts, episode lengths, and
   start frames match exactly, occur once, and remain non-aliasing.
4. All eight pre losses reproduce the frozen vector in Section 6 before any
   optimizer update. All probe and update-forward total/action losses are
   finite and non-negative; every pre action loss is strictly positive; total
   loss equals action loss within the T7 tolerance.
5. All 560 FP32 masters equal their BF16 model tensors cast to FP32 at
   initialization, with full-tensor `torch.equal` checks, before optimizer
   state exists.
6. One persistent optimizer/master set is used throughout. No optimizer or
   master object is replaced. Every Adam counter is exactly 20 at termination.
7. Before the first model forward, the harness records the `_version` of every
   named model buffer. The complete global buffer-version map must remain
   unchanged after every pre probe, every optimizer-step forward/projection,
   every phase probe, and every terminal probe. A missing, added, aliased, or
   version-changed buffer invalidates the run.
8. Every step uses exactly the scheduled A label. No H tensor enters any loss
   used for backward, gradients, clipping, master updates, optimizer state, or
   BF16 projection.
9. Only the four admitted roots receive gradients or updates. Frozen parameter
   versions remain unchanged. Gradient, clipping, master-sync, optimizer, and
   projection telemetry is finite at every step.
10. After **every** optimizer step 1 through 20, every trainable BF16 tensor
    must equal the corresponding clipped/updated FP32 master projected to BF16
    by full-tensor `torch.equal`; sampled-only projection checks are forbidden.
11. Each update-free probe preserves parameters, masters, buffers, module
    modes, gradients, optimizer state, prepared tensors, and caller RNG state.
12. The step-20 A0/H0 phase and terminal observations are exact duplicates for
    total loss and action loss, with no intervening state or RNG mutation.
13. The exact 8/44/20/20 call budget, phase-rotated schedule, five updates per
    A label, zero updates per H label, and 44 identical recipe signatures hold.

Any source, identity, starting-state, schedule, state-mutation, aliasing,
finite-value, nonnegative-loss, projection, count, or attribution failure
prevents a scientific classification. A created root then freezes only
`FAILED.json`; the model must not be rerun automatically.

## 9. Frozen T7 comparator

T8 compares only its terminal pre/post action-loss ratios with these immutable
T7 numerical endpoints, in suite order Spatial, Object, Goal, LIBERO-10:

```json
{
  "T7_A_ratios": [
    1.0947979413317903,
    0.7655266031194603,
    0.4848945538719168,
    0.137526060026468
  ],
  "T7_H_ratios": [
    1.0763376119266599,
    0.7577791145367958,
    0.5153250642031886,
    0.13820288765943442
  ]
}
```

The corresponding frozen T7 pair-mean vector, computed directly from the
unrounded ratios above, is:

```text
qT7 = [
  1.085567776629225,
  0.761652858828128,
  0.5001098090375526,
  0.1378644738429512
]
```

The T7 A-ratio median was `0.6252105784956885` and the T7 H-ratio median
recomputed from the exact pins above is `0.6365520893699922`. These medians
are contextual diagnostics. No T7
ratio or pair mean may be recomputed from rounded presentation-table text.

## 10. Pre-registered T8 classification

For suite `i` in Spatial, Object, Goal, LIBERO-10 order, define:

```text
rA_phase_i = T8_A_phase_action_loss_i / T8_A_pre_action_loss_i
rH_phase_i = T8_H_phase_action_loss_i / T8_H_pre_action_loss_i
rA_terminal_i = T8_A_terminal_action_loss_i / T8_A_pre_action_loss_i
rH_terminal_i = T8_H_terminal_action_loss_i / T8_H_pre_action_loss_i

q_phase_i = mean(rA_phase_i, rH_phase_i)
q_terminal_i = mean(rA_terminal_i, rH_terminal_i)
overwrite_penalty_i = q_terminal_i / q_phase_i

both_improved_i =
    (rA_terminal_i < 1.0) AND (rH_terminal_i < 1.0)
```

Every `q_phase_i` must be finite and strictly positive; otherwise the run has
no scientific classification. The fixed number of updates after the phase
probe and before the unified terminal block is, in the same suite order:

```text
trailing_updates = [0, 3, 2, 1]
```

Thus A0 has no intervening update after its phase probe, A1 has three, A2 has
two, and A3 has one. The A0 phase/terminal equality gate implies
`overwrite_penalty_0 == 1.0`; A0 is not counted in the overwrite threshold.

Spearman correlation is computed as the ordinary Pearson correlation of
ascending average ranks, with tied values assigned their average rank. Inputs
are the four unrounded float64 ratio-derived values in fixed suite order. If a
rank vector has zero variance, the correlation is reported as JSON `null` and
its associated threshold predicate is false; NaN is forbidden. Inclusive
floating-point threshold comparisons admit only an absolute `1e-12` rounding
tolerance around the frozen threshold; they do not change that threshold.

Define the primary terminal-recency predicate:

```text
spatial_terminal_rescued =
    (rA_terminal_0 < 1.0) AND (rH_terminal_0 < 1.0)

overwrite_supported =
    sum(overwrite_penalty_i >= 1.02 for i in [1, 2, 3]) >= 2

rho_recency = Spearman(q_terminal, trailing_updates)

terminal_recency_supported =
    spatial_terminal_rescued
    AND overwrite_supported
    AND rho_recency >= 0.8
```

The overwrite threshold is inclusive. An `overwrite_penalty` of 1.02 means
that the suite-pair mean terminal ratio is at least 2% larger than its
immediate fifth-update phase ratio. This is a loss-space recency diagnostic,
not a claim about optimizer causality beyond this fixed schedule.

Define the alternative stable-suite-ranking predicate:

```text
rho_suite = Spearman(qT7, q_terminal)

spatial_terminal_double_regression =
    (rA_terminal_0 >= 1.0) AND (rH_terminal_0 >= 1.0)

suite_effect_supported =
    rho_suite >= 0.8 AND spatial_terminal_double_regression
```

For a four-value terminal ratio vector, `median` is the arithmetic mean of the
second and third values after ascending numeric sort. The prior T7 joint gate
is retained only as a required diagnostic:

```text
joint_viable_diagnostic =
    median(rA_terminal_0, rA_terminal_1,
           rA_terminal_2, rA_terminal_3) <= 0.95
    AND
    median(rH_terminal_0, rH_terminal_1,
           rH_terminal_2, rH_terminal_3) <= 0.95
    AND
    sum(both_improved_i) >= 3
```

`joint_viable_diagnostic` is not part of the T8 primary classification and
must not be used to promote or demote a terminal verdict.

Classification is deterministic and ordered:

1. If `terminal_recency_supported`, the terminal typed verdict is
   `T8_TERMINAL_RECENCY_SUPPORTED`.
2. Otherwise, if `suite_effect_supported`, the terminal typed verdict is
   `T8_SUITE_EFFECT_SUPPORTED`.
3. Every other valid outcome is `T8_PHASE_ROTATED_MIXED_INCONCLUSIVE`.

A validity failure has no scientific verdict and freezes `FAILED.json`.

Object becomes the least-recently updated suite under T8. Its phase-to-terminal
overwrite penalty participates only through the frozen at-least-two-of-three
predicate shared with Goal and LIBERO-10. Whether Object alone regresses,
improves less, or remains stable is separately reported as a diagnostic and is
not its own classification clause. Post-hoc best-sample, minimum-loss,
alternative correlations or medians, threshold changes, or sample reranking
are forbidden.

## 11. Required terminal report

A valid `RESULT.json` must include, without rounding away the canonical values:

- all eight expected/observed pre losses and per-component tolerance checks;
- all eight phase and terminal total/action losses, phase/pre and
  terminal/pre ratios, deltas, relative drops, and improvement flags;
- each suite's `q_phase`, `q_terminal`, trailing-update count, and
  `overwrite_penalty`, including all three `>= 1.02` comparisons;
- the exact step-20 A0/H0 phase-versus-terminal reproduction checks;
- per-suite `both_improved` flags, sorted terminal A/H ratio vectors, terminal
  A/H medians, joint-improved count, and `joint_viable_diagnostic`;
- `spatial_terminal_rescued`, `overwrite_supported`, `rho_recency`,
  `terminal_recency_supported`, `rho_suite`,
  `spatial_terminal_double_regression`, and `suite_effect_supported`;
- the frozen T7 A/H ratio and pair-mean vectors, per-label T8-minus-T7 terminal
  ratio deltas, and all rank vectors used in both Spearman calculations;
- the Object-oldest A1/H1 result as diagnostic-only telemetry;
- the five update losses for each A sample plus the global 20-step curve;
- exact schedule labels and identities for all 20 update steps;
- exact prepare/forward/backward/optimizer/per-sample counts;
- all 44 recipe signatures and caller RNG restoration checks;
- config, source, runner, T7 RESULT, Sana, data, manifest, stats,
  normalization, and base-asset identities;
- trainable/frozen root names and element counts, exact initial FP32-master
  equality, optimizer object identity, and all terminal Adam counters;
- the complete initial/final global buffer-version map and a per-event
  no-change assertion;
- full-tensor FP32-master-to-BF16 projection equality for each of steps 1-20;
- SANA-WAM training-checkpoint load/save, simulator, rollout, evaluation, and
  formal-training counters fixed at zero. The pinned SANA base construction
  assets remain the explicitly declared exception.

Minimum/best loss, checkpoint selection, retry, root reuse, hidden updates,
sample replacement, threshold changes, or outcome-driven source changes are
forbidden.

## 12. Root, immutability, and execution boundary

A later execution authority must freeze a source commit and runner SHA, then
bind exactly one never-created root of this form:

```text
/DATA/share/sana_wam_libero_nonformal_screens/t8/<execution-commit-12>/libero-t8-phase-rotated-fixed20-<32-lowercase-hex nonce>
```

All namespace ancestors must be real non-symlink directories. Creation must
be exclusive. The root may never be reused, renamed, removed, or overwritten.
A valid terminal root contains only canonical `RESULT.json`; an execution
failure root contains only canonical `FAILED.json`. The terminal directory is
frozen `0500` and its terminal JSON is frozen `0400`.

This pre-registration does not choose a nonce, create a namespace or root,
reserve a GPU, execute CUDA/JIT, construct a model, read sample tensors, or run
tests. Those actions require the separate execution authority.

Any authorized execution remains limited to one idle H200 GPU, fresh model
construction, the eight pinned real samples, the 24 update-free measurements
(8 pre, 8 phase, 8 terminal), and exactly 20 non-formal micro-updates. It may
not stop, alter, or contend with another user's process.

SANA-WAM training-checkpoint load/save, formal training, full-corpus training,
simulator execution, rollout, benchmark evaluation, admission, deployment, token
namespaces, real samples beyond A0-A3/H0-H3, and every unlisted capability are
forbidden.

## 13. Interpretation and successor boundary

`T8_TERMINAL_RECENCY_SUPPORTED` would show only that, for this one
initialization and fixed recipe, Spatial improved on both terminal endpoints,
at least two of the three earlier-finalized suite pairs incurred the frozen
phase-to-terminal overwrite penalty, and terminal pair-mean loss ratios were
strongly rank-associated with the number of trailing updates. It would support
a terminal-recency explanation for this particular T7/T8 contrast. It would
not prove that recency is the sole cause, that every suite improves, or that
rollout performance improves.

`T8_SUITE_EFFECT_SUPPORTED` would show that Spatial still regressed on both
terminal endpoints and that the T7 and T8 suite pair-mean rankings remained
strongly aligned despite the phase rotation. That supports a stable suite-side
effect over the simple terminal-recency hypothesis under this fixed dose. It
does not identify whether the cause is task difficulty, data statistics,
conditioning, interference, architecture, or optimization, and it does not
prove an architectural impossibility.

`T8_PHASE_ROTATED_MIXED_INCONCLUSIVE` covers every other valid combination,
including partial Spatial rescue, rescue without the overwrite/rank pattern,
or double Spatial regression without stable T7/T8 suite ranks. The joint
four-suite gate and Object-oldest behavior are always reported but remain
diagnostic-only under this pre-registration.

All A/H samples are consumed evidence. T8 uses one seed, one stochastic
recipe, one update sample and one same-task probe per suite, five updates per
suite, and loss-space endpoints only. It does not establish distributional
generalization, robustness across seeds or recipes, long-horizon stability,
instruction completion, LIBERO success rate, checkpoint quality, formal
training readiness, admission, or deployment.

No outcome authorizes a same-root rerun, automatic dose extension, alternate
order, sample replacement, formal training, SANA-WAM training checkpointing,
or evaluation. Any successor must be separately pre-registered with a fresh
source identity, fresh nonce, and never-created root.

## 14. Current state

T8 executed exactly once and produced a valid frozen non-formal result. There
was no retry, automatic extension, alternate schedule, or same-root rerun.

- source commit: `9cd1c490d14b3c2437e82225ee8cfdf58646837e`;
- runner SHA256:
  `9ec74d83c8443d78d2ba765dd766f82a96cccef894ae63d08cf9e2a8dbf97669`;
- root:
  `/DATA/share/sana_wam_libero_nonformal_screens/t8/9cd1c490d14b/libero-t8-phase-rotated-fixed20-2e1efcc69bc552affb5c85b7feeb5175`;
- RESULT SHA256:
  `bfdb852e14a5bd9b1c8e776be9f4ff108899eae65d557cebe42b06f1991b0a18`;
- physical GPU 0 / UUID
  `GPU-1ec28cfb-f501-23f3-f865-275a744ca053`;
- validity: `valid_run=true`, `execution_result=PASS`, `harness_result=PASS`;
- typed verdict: `T8_PHASE_ROTATED_MIXED_INCONCLUSIVE`.

The run matched the exact 8 prepare / 44 forward / 24 update-free measurement /
20 backward / 20 AdamW-step budget. All eight starting losses reproduced
exactly, all 20 full-tensor FP32-master-to-BF16 projections passed, global
buffers remained unchanged, and H0-H3 received zero updates. The terminal A/H
ratios were:

| Suite | phase A / H | terminal A / H | overwrite penalty | terminal pair |
|---|---|---|---:|---|
| Spatial | `0.661098 / 0.630339` | `0.661098 / 0.630339` | `1.000000` | improved |
| Object | `0.627326 / 0.618242` | `1.580494 / 1.597984` | `2.551829` | regressed |
| Goal | `0.821394 / 0.827440` | `0.596465 / 0.599601` | `0.725401` | improved |
| LIBERO-10 | `0.228250 / 0.219789` | `0.256395 / 0.247745` | `1.125215` | improved |

Spatial was rescued on both endpoints and two nonterminal suites met the
`>=1.02` overwrite threshold, but `rho_recency=0.4` was below `0.8`; therefore
`terminal_recency_supported=false`. The T7/T8 suite-rank correlation was
`0.7999999999999998`, but Spatial did not double-regress, so
`suite_effect_supported=false`. The diagnostic-only joint gate passed with A
median `0.6287817650`, H median `0.6149701490`, and 3/4 corresponding suite
pairs improved.

The result is evidence of strong phase-sensitive interference: making Spatial
most recent rescued it, while making Object oldest produced a large overwrite.
It does not satisfy the pre-registered monotone recency classification because
Goal improved further despite two trailing updates. It remains a one-seed,
eight-sample loss-space architecture screen—not formal training, rollout,
benchmark evaluation, SANA-WAM training-checkpoint evidence, admission, or
deployment authority.

An independent post-run audit recomputed the canonical JSON, all frozen pins,
budgets, ratios, both Spearman statistics, and the typed verdict without a
numerical discrepancy. It also found one reporting-strength gap: Section 8
asked for a serialized buffer-version assertion after every individual event,
while the harness records initial/final global maps plus update-free probe-group
snapshots, not a per-event map sequence. Initial and final buffer identities,
data pointers, and monotonic Torch `_version` values were identical, so no
buffer mutation was observed; nevertheless the stronger per-event reporting
claim was not fully evidenced. The immutable result is therefore retained as
audit-qualified non-formal architecture evidence and is not rerun or promoted
to admission. T8 also inherits T7's temporary T1 predecessor pin under `/tmp`;
that file existed with the exact required SHA at execution time but is not a
durable evidence location.
