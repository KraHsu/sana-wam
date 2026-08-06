# LIBERO T7 Four-Suite Cyclic Micro-Learnability Screen

Status: **executed / `T7_FOUR_SUITE_CYCLIC_GO`**

Date: 2026-08-06

Evidence source is the frozen execution commit
`d19109a2314f8e7186571afed6b4acd816d7cfab` and runner SHA stated in Section
10. This post-run documentation revision records the immutable result; it is
not the executed source and does not mutate or reattribute that root.

## 1. Scientific question

T7 asks one narrow architecture question: starting from the same fresh model
initialization and loss recipe as the frozen T6 predecessor, can one persistent
optimizer make exactly 20 action-only updates while cycling over one fixed real
sample from each configured LIBERO suite, and reduce action loss both on those
four update samples and on a mechanically selected, same-task, update-free
episode from each suite?

The fixed suite order is:

1. LIBERO Spatial;
2. LIBERO Object;
3. LIBERO Goal;
4. LIBERO-10.

The update order is exactly `[Spatial, Object, Goal, LIBERO-10] × 5`. Thus T7
keeps the T6 total update budget at 20 and changes the update-sample schedule,
not the total optimizer dose. Each suite receives exactly five direct updates.
Forty updates are not part of this experiment and may not be reached by
continuing the run after inspecting step 20.

T7 is a non-formal loss-space architecture screen. It is not training
admission, full training, checkpoint production, simulator execution, rollout,
or LIBERO benchmark evaluation.

## 2. Architecture and immutable T6 predecessor

The model remains the full production-shaped `DualSystemARArchitecture`, not
the reduced CACH-A4 scaffold. The SANA video backbone remains frozen, while the
only trainable roots remain:

- `action_backbone`;
- `proprio_encoder`;
- `proprio_video_embed`;
- `proprio_action_embed`.

Before CUDA or model construction, the runner must verify the direct
predecessor:

- T6 RESULT:
  `/DATA/share/sana_wam_libero_nonformal_screens/t6/708b1d856986/libero-t6-crosssuite3-fixed20-67a02fcc85508e03f136e09221a6a9d4/RESULT.json`;
- T6 RESULT SHA256:
  `4855f3771b80349547c985d137426cce79e25597f910eff88e136328424b8b89`;
- T6 source commit:
  `708b1d8569866608898031f9116566d52fdeb742`;
- T6 runner SHA256:
  `004f444168f26162f012c408193e119a8ff428a64f90e7dbc9ab595df9082903`;
- T6 typed verdict: `T6_CROSS_SUITE_TRANSFER_GO`;
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
  `ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad`.

T7 uses fresh initialization and may load only the pinned offline
SANA/Gemma/VAE base construction assets admitted by T6. It must not load,
resume, or save a SANA-WAM training checkpoint. It must not initialize from the
T6 post-update model state.

Because the T7 update schedule intentionally differs from T6, a T6 post-update
training-core equality requirement would be scientifically false. Instead, T7
must reproduce the pinned T6 starting-state projection in Section 7 before the
first optimizer update.

## 3. Fixed four-sample update set

The following four samples are the complete and immutable update set. A1-A3
are the consumed T6 cross-suite probes, deliberately promoted to update
samples by this separately pre-registered successor. No other real sample may
enter backward or optimizer update.

| Label | Dataset | Task / episode / start | Task | Length | T6-pinned pre action loss |
|---|---|---|---|---:|---:|
| A0 | `libero_spatial_no_noops_1.0.0_lerobot` | 0 / 0 / 0 | `pick up the black bowl next to the cookie box and place it on the plate` | 110 | `13.679718971252441` |
| A1 | `libero_object_no_noops_1.0.0_lerobot` | 3 / 82 / 0 | `pick up the bbq sauce and place it in the basket` | 135 | `14.696584701538086` |
| A2 | `libero_goal_no_noops_1.0.0_lerobot` | 2 / 70 / 0 | `open the top drawer and put the bowl inside` | 249 | `12.06648063659668` |
| A3 | `libero_10_no_noops_1.0.0_lerobot` | 3 / 259 / 0 | `turn on the stove and put the moka pot on it` | 228 | `20.43115234375` |

Their raw assets remain pinned exactly as follows:

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

All paths in this document are relative to the dataset named by their row.
Every metadata and raw-asset path must be a regular non-symlink file, must
match its pin before CUDA or model construction, and must be hashed again after
sample materialization.

## 4. Frozen same-task fresh-probe population

Each H sample must come from the same dataset and task as its paired A sample,
must use start frame 0, and must never enter backward or update. Restricting the
probe to the same task avoids adding a cross-task variable to the first cyclic
screen.

The frozen metadata and eligible counts are:

| Order | Dataset / update task | Tasks SHA256 | Episodes SHA256 | Eligible episodes |
|---:|---|---|---|---:|
| 0 | Spatial / task 0 | `399841cf8e861052d3fb7214ed619b5a10b4eded2d6f107e7fed338a233dd0e1` | `690349688e96d984007bf2cca3ccf2683dfc809f03dab55e1159e4cc71f150b7` | 42 |
| 1 | Object / task 3 | `68ef5f9bc5a0bd74f46140f6721fa0ea74e997d74e37b8714a539f61337e7862` | `63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea` | 45 |
| 2 | Goal / task 2 | `39f08f81b289ad3041f1c8ada88f679fe60774e9fde4083415881486edc23d55` | `548d91fe48b7d439248523dd3f7a5e4b15fc77d5eb1b7cfdd6da0033d422cb43` | 35 |
| 3 | LIBERO-10 / task 3 | `45f9eb4d4b6b04999f64640c0aae380555372b7b273a904f5f459ad05d4a0a6a` | `5589f8f87cfddb34812782462160bf55b0d3082e404240682d1d0a89faba8265` | 40 |

The canonical eligible manifest has schema
`sana-wam-libero-t7-four-suite-cyclic-same-task-probe-eligible-v1`. Its exact
top-level keys are `candidate_episode_count`, `candidate_suite_count`,
`candidate_task_count`, `excluded_samples`, `schema_version`, `start_frame`,
`suites`, and `t6_predecessor_result_sha256`. The exact counts are 162
episodes, four suites, and four tasks; `start_frame` is 0.

The `excluded_samples` array is ordered historical evidence, not a set sorted
after the fact. Each row has exactly `dataset`, `task_index`, and
`episode_index`, and the ten rows are exactly:

```json
[
  {"dataset":"libero_spatial_no_noops_1.0.0_lerobot","task_index":0,"episode_index":0},
  {"dataset":"libero_spatial_no_noops_1.0.0_lerobot","task_index":0,"episode_index":16},
  {"dataset":"libero_spatial_no_noops_1.0.0_lerobot","task_index":0,"episode_index":405},
  {"dataset":"libero_spatial_no_noops_1.0.0_lerobot","task_index":0,"episode_index":40},
  {"dataset":"libero_spatial_no_noops_1.0.0_lerobot","task_index":7,"episode_index":36},
  {"dataset":"libero_spatial_no_noops_1.0.0_lerobot","task_index":1,"episode_index":325},
  {"dataset":"libero_spatial_no_noops_1.0.0_lerobot","task_index":4,"episode_index":11},
  {"dataset":"libero_object_no_noops_1.0.0_lerobot","task_index":3,"episode_index":82},
  {"dataset":"libero_goal_no_noops_1.0.0_lerobot","task_index":2,"episode_index":70},
  {"dataset":"libero_10_no_noops_1.0.0_lerobot","task_index":3,"episode_index":259}
]
```

The configured exclusion of Goal episode 82 remains enforced by the live
dataset and normalization contract. It is outside the fixed Goal update task,
so it is not an eligible T7 row and is not an additional consumed-evidence row
in the array above.

Each suite row has exactly `dataset`, `episode_count`,
`episodes_jsonl_sha256`, `task_count`, `tasks`, `tasks_jsonl_sha256`, and
`update_task_index`. Each suite contains one task row with exactly
`episode_count`, `episodes`, `task`, and `task_index`; each episode row has
exactly `episode_index` and `length`. Suites retain Spatial, Object, Goal,
LIBERO-10 order, while episodes inside each task are ordered by integer episode
index.

Serialization is UTF-8 sorted-key compact JSON with `ensure_ascii=True`,
`allow_nan=False`, separators `(',', ':')`, and no final LF. The canonical
length and identity are:

- eligible manifest byte length: `8406`;
- eligible manifest SHA256:
  `0b6bfe4c51e8b9e2ac3f538bac7853b5e1e871ea96dda5df6976ee27b5b25370`.

The runner must rebuild this object from the pinned metadata before CUDA and
must independently require the live start-frame-0 registry to reproduce the
same 162 dataset-qualified rows after dataset construction.

## 5. Mechanical fresh-probe selection

Within each fixed update task, eligible episodes are ranked by the SHA256 of
this ASCII payload, including its final LF:

```text
SANA-WAM/LIBERO/T7_EPISODE_SELECTION_V1
T6_RESULT_SHA256=4855f3771b80349547c985d137426cce79e25597f910eff88e136328424b8b89
ELIGIBLE_MANIFEST_SHA256=0b6bfe4c51e8b9e2ac3f538bac7853b5e1e871ea96dda5df6976ee27b5b25370
DATASET=<exact dataset name>
TASK_INDEX=<base-10 task index>
EPISODE_INDEX=<base-10 episode index>
START_FRAME=0
```

The smallest hash in each suite selects H0-H3. The immutable result is:

| Label | Dataset | Task / episode / start | Length | Episode payload SHA256 |
|---|---|---|---:|---|
| H0 | Spatial | 0 / 30 / 0 | 125 | `0a33e0f8df2afd9f256cba15af973cc4b5f7dde3ddbc0b5866f454a535e4e83d` |
| H1 | Object | 3 / 166 / 0 | 129 | `0378af218d388de30af08e0be5c417dfb5a7275939dca80da5b1a15d9c841df1` |
| H2 | Goal | 2 / 248 / 0 | 184 | `02c53f50cf758bc8185dbedbdfc5b7c05d297389b5fd55e04d738e0f65856a94` |
| H3 | LIBERO-10 | 3 / 278 / 0 | 261 | `125e02f94d4e09f2216fbf649869d6bd3e1f0c43044992d031abc4dd8454b974` |

The task text of H0-H3 is exactly the task text of A0-A3 respectively.

The canonical selected manifest has schema
`sana-wam-libero-t7-four-suite-cyclic-same-task-probe-selection-v1`. Its exact
top-level keys are `eligible_manifest_sha256`, `samples`, `schema_version`,
`selection_rule`, `suite_order`, and `t6_predecessor_result_sha256`.
`suite_order` is the four full dataset names in Spatial, Object, Goal,
LIBERO-10 order. Each sample has exactly `assets`, `dataset`, `episode_index`,
`episode_length`, `episode_selection_payload_sha256`, `label`, `start_frame`,
`task`, and `task_index`. The exact selection-rule string is:

```text
in frozen config suite order spatial,object,goal,10; within each fixed update task rank eligible episodes by episode payload SHA256 and take first
```

It uses the same canonical JSON serialization rules as the eligible manifest.
Its frozen length and identity are:

- selected manifest byte length: `3351`;
- selected manifest SHA256:
  `e969e76e4bcb8a3c2b0b3fffff35e0ec10c478e45cb6535b7bd91061d3a6a7ee`.

No label, sample, suite order, or payload may be replaced because of length,
padding, prompt size, preparation cost, starting loss, runtime, or outcome.

## 6. Fresh-probe raw assets

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

## 7. Fresh initialization, fixed cycle, and exact budget

The frozen optimization recipe is:

- fresh initialization seed `20260806`;
- fixed loss-recipe seed `20260826` before every forward;
- action loss weight 1.0 and video loss weight 0.0;
- one persistent FP32-master AdamW optimizer over the four admitted roots;
- learning rate `1e-4`, betas `(0.9, 0.95)`, weight decay 0;
- global gradient clip bound 1.0;
- BF16 model projection is called after each optimizer step; full-tensor
  projection equality is explicitly checked after steps 1 and 20;
- no scheduler, warmup, gradient accumulation, sample shuffle, batching,
  resampling, early stopping, or SANA-WAM checkpoint.

Preparation consists of exactly eight independent single-sample calls in this
order: A0, A1, A2, A3, H0, H1, H2, H3. Prompt wrapping must equal
`format_prompt_for_inference(task_name)`. Every prepared mapping and tensor
must remain independent, non-aliasing, immutable, and unreprepared.

The exact execution order is:

1. update-free A0, A1, A2, A3 pre probes;
2. update-free H0, H1, H2, H3 pre probes;
3. 20 forward/backward/clip/master-sync/AdamW/BF16-project updates following
   the fixed cycle below;
4. update-free A0, A1, A2, A3 post probes immediately after step 20;
5. update-free H0, H1, H2, H3 post probes.

| Sample | Optimizer steps |
|---|---|
| A0 Spatial | 1, 5, 9, 13, 17 |
| A1 Object | 2, 6, 10, 14, 18 |
| A2 Goal | 3, 7, 11, 15, 19 |
| A3 LIBERO-10 | 4, 8, 12, 16, 20 |

The exact budget is:

- 8 `prepare_inputs` calls;
- 36 architecture forwards: 20 update forwards and 16 measurements;
- 20 backward calls;
- 20 optimizer steps;
- one optimizer and one persistent FP32-master set;
- five update forwards per A sample;
- zero H tensors in backward or optimizer update;
- zero post-probe re-preparations.

Every forward, including probes and update forwards, resets to recipe seed
`20260826` while forking and restoring caller Python, NumPy, Torch CPU, and
Torch CUDA RNG states. All 36 captured recipe signatures must equal the frozen
T6 signature.

Before update 1, the deterministic starting state must reproduce the frozen T6
source/config/Sana/seed/recipe and trainable-root structure, empty optimizer
state, and the four exact A pre action losses in Section 3. Runtime checks also
bind the 560 persistent FP32 masters to the 560 BF16 trainable tensors. The T6
RESULT does not contain a bytewise fresh-model parameter fingerprint, so this
contract does not claim a cross-run bytewise model or master fingerprint
comparison. T7 must not claim equality to the T6 post-update training core.

## 8. Validity and attribution gates

A scientific verdict is allowed only if all harness gates pass:

1. Source, runner, config, Sana, T6 RESULT, base assets, selected-row stats,
   normalization population, dataset metadata, update assets, and probe assets
   equal their frozen pins.
2. The canonical eligible and selected manifests reproduce exactly before
   CUDA. The live dataset registry independently reproduces every eligible and
   selected dataset-qualified row after construction.
3. Every A/H identity occurs exactly once at start frame 0 and matches its
   pinned dataset, task text, task index, episode index, length, and prompt.
4. All eight prepared mappings remain mutually non-aliasing and immutable;
   their object identities and tensor versions do not change.
5. Fresh initialization reproduces the frozen T6 source/config/Sana/seed/recipe,
   trainable-root structure, empty optimizer state, and four pinned A pre
   action losses. No bytewise T6 fresh-model fingerprint is available or
   claimed.
6. One persistent optimizer and FP32-master set are used for all 20 steps.
   Optimizer state is empty before step 1, no optimizer object is replaced,
   and all Adam counters are exactly 20 after step 20.
7. Each step uses exactly its scheduled A sample. No H tensor enters a loss
   used by backward, gradient calculation, clipping, master update, optimizer
   state, or BF16 projection.
8. Every update-free probe block preserves parameters, FP32 masters, buffers,
   module modes, gradients, optimizer state, prepared tensors, and caller RNG
   state.
9. Only the four admitted trainable roots receive gradients or optimizer
   updates; frozen parameter versions remain unchanged. Update-free probe
   blocks additionally preserve buffer versions. Gradient, update, clipping,
   master-sync, and sampled BF16-projection telemetry is finite for every
   step, with full projection equality checked at steps 1 and 20.
10. All losses are finite and non-negative, every pre action loss is strictly
    positive, and total loss equals action loss within the frozen T6 tolerance.
11. The exact 8/36/20/20 prepare/forward/backward/optimizer counts and exact
    per-sample update counts hold. There is no step 21, hidden retry, best-step
    restore, or checkpoint operation.

Any identity, source, selection, starting-state, schedule, state-mutation,
aliasing, finite-value, count, or runtime failure prevents a scientific
verdict. A created root then freezes only `FAILED.json`.

## 9. Pre-registered scientific gate

For suite `i` in Spatial, Object, Goal, LIBERO-10 order, define:

```text
rA_i = A_post_action_loss_i / A_pre_action_loss_i
rH_i = H_post_action_loss_i / H_pre_action_loss_i
A_improved_i = A_post_action_loss_i < A_pre_action_loss_i
H_improved_i = H_post_action_loss_i < H_pre_action_loss_i
both_improved_i = A_improved_i AND H_improved_i
```

For four ratios, `median` is explicitly the arithmetic mean of the second and
third values after ascending numeric sort. The sole primary gate is:

```text
median(rA_0, rA_1, rA_2, rA_3) <= 0.95
AND
median(rH_0, rH_1, rH_2, rH_3) <= 0.95
AND
sum(both_improved_i) >= 3
```

- A valid run passing all three clauses is
  `T7_FOUR_SUITE_CYCLIC_GO`.
- A valid run with exact attribution whose primary gate misses is
  `T7_FOUR_SUITE_CYCLIC_INCONCLUSIVE`.
- A run failing any validity or attribution gate has no scientific verdict and
  freezes `FAILED.json`.

The terminal report must include all eight paired pre/post total and action
losses, absolute deltas, ratios, relative drops, improvement flags, per-suite
joint flags, both sorted ratio vectors, both medians, joint-improved count,
the global and per-sample five-point update curves, exact sample/prompt/context
diagnostics, all recipe signatures, all identities, exact counts, and the
starting-state comparison. Minimum/best loss, post-hoc thresholds, sample
replacement, reranking, checkpoint selection, and automatic reruns are
forbidden.

## 10. Root and execution boundary

The executed authority fixed one never-created root of this form:

```text
/DATA/share/sana_wam_libero_nonformal_screens/t7/<execution-commit-12>/libero-t7-foursuite-cyclic-fixed20-<32-lowercase-hex nonce>
```

Namespace ancestors must be real non-symlink directories. The root must be
created exclusively and may never be reused, renamed, removed, or overwritten.
A valid terminal root contains only `RESULT.json`; an execution failure root
contains only `FAILED.json`. Terminal root mode is `0500` and terminal JSON
mode is `0400`.

The realized execution identity was:

- source commit: `d19109a2314f8e7186571afed6b4acd816d7cfab`;
- runner SHA256:
  `b6e764bf3d3566bf3d1fe5e2e3802dafef691f6a0162568337b49ca1a183310b`;
- config SHA256:
  `5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`;
- physical GPU 0 / UUID
  `GPU-1ec28cfb-f501-23f3-f865-275a744ca053`;
- nonce: `61e0a0ff817970e994b0875be4840ed6`;
- root:
  `/DATA/share/sana_wam_libero_nonformal_screens/t7/d19109a2314f/libero-t7-foursuite-cyclic-fixed20-61e0a0ff817970e994b0875be4840ed6`.

The execution used one H200 GPU, offline base construction assets, fresh model
construction, eight real-sample preparations, 16 update-free measurements,
and the fixed 20-step micro-update only.

Simulator execution, rollout, benchmark evaluation, checkpoint load/save,
formal training, admission, deployment, token namespaces, real-data samples
beyond A0-A3/H0-H3, and all unlisted capabilities remain forbidden.

## 11. Interpretation boundary

A GO would establish only that, under one initialization and one fixed
stochastic recipe, the fixed 20-step cyclic schedule reduced loss on both the
update sample and a same-task update-free episode in at least three of four
suites, while both four-sample median ratios met the pre-registered threshold.
It would not establish that every suite improved.

H0-H3 are optimizer-update-held-out only. They remain members of the
configured all-four-suite training corpus and of the selected-row
normalization population. They are not strict dataset or normalization
holdouts. The A and H episodes within a suite share a task, while visual
observations, episode length, and temporal coverage may differ. This is not a
cross-task probe or a factor-isolation study.

The screen uses one initialization, one recipe, five direct updates per suite,
one A sample and one H sample per suite. It does not establish distributional
generalization, cross-recipe or cross-seed robustness, long-horizon optimizer
stability, rollout success, instruction completion, LIBERO success rate,
checkpoint quality, formal-training readiness, admission, or deployment.

A0-A3 are already consumed evidence; H0-H3 become consumed evidence when
their first losses are revealed. No outcome permits replacing an H sample,
changing suite order, extending this root to 40 steps, or rerunning under the
same root.

## 12. Frozen successor logic

- If T7 is GO, the planned loss-space architecture ladder stops. The only next
  forward path is a separately pre-registered bounded multi-suite
  training/admission step. GO does not itself authorize formal training,
  checkpoint production, simulator rollout, or benchmark evaluation.
- If T7 is INCONCLUSIVE with all validity and starting-state gates passing, a
  separately pre-registered dose diagnostic may use a fresh initialization,
  corrected/frozen source, fresh nonce, and never-created root for exactly 40
  total updates with `[A0,A1,A2,A3] × 10`. It must keep the same optimizer,
  recipe, sample identities, order, measurements, and scientific threshold;
  it may not replace H0-H3. Those H samples must then be described as fixed,
  consumed, update-held-out endpoints, not new independent evidence. The
  40-step diagnostic is not automatic and is not authorized by this document.
- If T7 is FAILED, no micro-learnability interpretation is allowed. Only the
  root cause may be diagnosed. Any corrected execution requires a new frozen
  source revision, separate authority, fresh nonce, and never-created root.

## 13. Result state

The root terminalized with one read-only `RESULT.json` and no `FAILED.json`.
The canonical result SHA256 is
`9f5181a30cd0d6b676f09315244f7560cd3902004f5f17ca833330e33cb44d3a`.
The valid typed verdict is `T7_FOUR_SUITE_CYCLIC_GO`.

| Pair | A pre → post | A ratio | H pre → post | H ratio | Both improved |
|---|---:|---:|---:|---:|---|
| Spatial | 13.679719 → 14.976528 | 1.094798 | 13.379680 → 14.401052 | 1.076338 | no |
| Object | 14.696585 → 11.250627 | 0.765527 | 14.698094 → 11.137909 | 0.757779 | yes |
| Goal | 12.066481 → 5.850971 | 0.484895 | 12.026172 → 6.197388 | 0.515325 | yes |
| LIBERO-10 | 20.431152 → 2.809816 | 0.137526 | 20.885685 → 2.886462 | 0.138203 | yes |

The four A ratios have median `0.6252105785`; the four H ratios have median
`0.6365520894`. Object, Goal, and LIBERO-10 improved on both members of their
pair, so the joint count is 3/4 and all three pre-registered gate clauses pass.
Spatial regressed on both A0 and H0; GO therefore must not be described as
uniform four-suite improvement.

The exact execution counts were 8 preparations, 36 forwards, 20 backward
calls, and 20 optimizer steps. Each A sample entered five updates; every H
sample entered zero. All 36 stochastic recipe signatures were identical, all
560 Adam counters reached 20, and the persistent FP32-master state remained
finite. No SANA-WAM checkpoint was loaded or saved, and no simulator, rollout,
benchmark evaluation, formal training, admission, or deployment ran.

## 14. Post-run audit notes

The terminal result binds the exact config path content through the reported
SHA and the execution CLI identity comparison. The runner source itself accepts
the authority-provided expected config SHA rather than hard-coding the baseline
SHA, so reuse under a different source authority must not be treated as this
T7 result.

The runner also re-verifies the historical T1-T6 evidence chain. T1 currently
resides under its historical `/tmp` path and was present with the pinned SHA at
preflight; this is a reproducibility weakness for a future rerun, not a mutation
of this frozen terminal result. The scientific evidence here remains a single
initialization, fixed-recipe loss-space screen, with the interpretation limits
in Section 11.

The harness accepts each frozen starting loss within
`1e-6 + 1e-6 * abs(expected)` rather than requiring bitwise float equality.
In this realized result all four serialized observed values equal their pins
exactly. The harness does not compare frozen-buffer values across the update
loop, and it performs full-tensor BF16/master projection equality checks only
at steps 1 and 20; these are attribution-checking limits, not evidence of an
observed runtime failure.
