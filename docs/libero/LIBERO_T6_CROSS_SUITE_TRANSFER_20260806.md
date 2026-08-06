# LIBERO T6 Cross-Suite Transfer Screen

Status: **proposed / not executed**

Date: 2026-08-06

## 1. Scientific question

T6 asks one narrow question: after exactly 20 action-only updates on the fixed
LIBERO Spatial training sample (task 0 / episode 0 / start frame 0), does the
same freshly initialized production-shaped AR architecture reduce paired action
loss on one mechanically selected, optimizer-update-held-out sample from each
of the other three configured LIBERO suites?

The three probe suites are fixed in the existing config order:

1. LIBERO Object;
2. LIBERO Goal;
3. LIBERO-10.

T6 changes only the probe-suite axis. It does not repeat the T5 Spatial probes
at runtime, add training samples, change the optimizer or schedule, load a
SANA-WAM checkpoint, or run a simulator. The model, initialization, training
sample, stochastic recipe, optimizer state machine, and training-core
projection remain frozen from T5.

## 2. Architecture and immutable T5 predecessor

The model under test is the full production-shaped
`DualSystemARArchitecture`, not the reduced CACH-A4 scaffold. Its SANA video
backbone remains frozen. The only trainable roots remain:

- `action_backbone`;
- `proprio_encoder`;
- `proprio_video_embed`;
- `proprio_action_embed`.

T6 must verify the direct predecessor before model construction:

- T5 RESULT:
  `/DATA/share/sana_wam_libero_nonformal_screens/t5/5150693a0751/libero-t5-crosstask3-fixed20-cf7dd8a1b1cef03511d2026a48e4a271/RESULT.json`;
- T5 RESULT SHA256:
  `a656aaef1528537527fe830ad7d4107138b29e8e254b5606b43c46a47e323e83`;
- T5 source commit:
  `5150693a0751200ef431968a863c69f8ca08a7df`;
- T5 runner SHA256:
  `a4bbdcf752aa0a34a43ea4f51e7875f7fd160985280a7274b29e36350d9605c1`;
- reused config:
  `configs/benchmarks/libero/train_libero_ar_baseline.yaml`;
- config SHA256:
  `5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`;
- Sana gitlink/worktree:
  `16b9cec673e3335724ba2d8db25de7f9ed229292`;
- frozen T3/T4/T5 training-core projection SHA256:
  `e34a2dd0ae2dfe28303dcd9baf64ffc5800d002b0b7646b89dde2aafa132c352`;
- frozen loss-recipe signature SHA256:
  `ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad`.

The selected-row all-four-suite normalization artifact remains fixed:

- stats SHA256:
  `e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7`;
- selected population SHA256:
  `7ed9772facf261299022e55169bcdaaa49fe3a7a20e0057e08cf419b5e584146`.

The runner must reconstruct the T5 training-core projection from the frozen
RESULT, require its canonical SHA to equal the value above, and then require
the current T6 training-core projection to equal that exact object. The frozen
projection includes the training input, initial and final losses, both full
20-point loss curves, every non-timing per-step gradient/update/Adam field,
optimizer summaries, sampled FP32-master and BF16 projections, and stochastic
recipe signatures. Timestamp, wall time, memory, and GPU-utilization telemetry
remain outside the projection.

## 3. Frozen cross-suite candidate population

The suite order is immutable and follows the existing config after the Spatial
training suite:

| Order | Dataset | Tasks SHA256 | Episodes SHA256 | Eligible tasks | Eligible episodes |
|---:|---|---|---|---:|---:|
| 1 | `libero_object_no_noops_1.0.0_lerobot` | `68ef5f9bc5a0bd74f46140f6721fa0ea74e997d74e37b8714a539f61337e7862` | `63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea` | 10 | 454 |
| 2 | `libero_goal_no_noops_1.0.0_lerobot` | `39f08f81b289ad3041f1c8ada88f679fe60774e9fde4083415881486edc23d55` | `548d91fe48b7d439248523dd3f7a5e4b15fc77d5eb1b7cfdd6da0033d422cb43` | 10 | 427 |
| 3 | `libero_10_no_noops_1.0.0_lerobot` | `45f9eb4d4b6b04999f64640c0aae380555372b7b273a904f5f459ad05d4a0a6a` | `5589f8f87cfddb34812782462160bf55b0d3082e404240682d1d0a89faba8265` | 10 | 379 |

Every task index 0 through 9 in each listed suite is eligible. Every episode of
those tasks is eligible at start frame 0 except the already excluded and
repaired-population-ineligible sample
`libero_goal_no_noops_1.0.0_lerobot:82`. There are therefore exactly three
suites, 30 tasks, and 1,260 eligible episode rows. Object episode 82 is a
different dataset-qualified identity and remains eligible; it must not be
confused with excluded Goal episode 82.

The canonical candidate manifest uses schema
`sana-wam-libero-t6-cross-suite-eligible-v1`. Its exact top-level keys are
`candidate_episode_count`, `candidate_suite_count`, `candidate_task_count`,
`excluded_samples`, `schema_version`, `start_frame`, `suites`, and
`t5_predecessor_result_sha256`. The exact top-level counts are 1,260, 3, and 30;
`start_frame` is 0; and `excluded_samples` is exactly:

```json
[{"dataset":"libero_goal_no_noops_1.0.0_lerobot","episode_index":82}]
```

Each suite row has exactly `dataset`, `episode_count`,
`episodes_jsonl_sha256`, `task_count`, `tasks`, and `tasks_jsonl_sha256`.
Each task row has exactly `episode_count`, `episodes`, `task`, and
`task_index`; each episode row has exactly `episode_index` and `length`.
Suites retain the order Object, Goal, LIBERO-10; tasks and episodes are ordered
by their integer indices. Serialization is UTF-8 sorted-key compact JSON with
`ensure_ascii=True`, `allow_nan=False`, separators `(',', ':')`, and no final
LF. The canonical byte length is exactly 48,001.

- Eligible manifest SHA256:
  `c4ec46f7f1ede73556646da52501b3192c8d951b82d51ae8f9b18b70d15545f2`.

The runner must rebuild this manifest from the pinned metadata before CUDA or
model construction. After dataset construction it must independently require
the live start-frame-0 window registry, including the Goal exclusion, to match
the same population exactly.

## 4. Mechanical two-stage selection within each suite

The suites themselves are not ranked. In the fixed Object, Goal, LIBERO-10
order, tasks within each suite are ranked by this ASCII payload, including its
final LF:

```text
SANA-WAM/LIBERO/T6_TASK_SELECTION_V1
T5_RESULT_SHA256=a656aaef1528537527fe830ad7d4107138b29e8e254b5606b43c46a47e323e83
ELIGIBLE_MANIFEST_SHA256=c4ec46f7f1ede73556646da52501b3192c8d951b82d51ae8f9b18b70d15545f2
DATASET=<exact dataset name>
TASK_INDEX=<base-10 task index>
START_FRAME=0
```

The smallest task-payload SHA256 in each suite selects that suite's task. The
episodes belonging to that selected task are then ranked by this ASCII
payload, also including its final LF:

```text
SANA-WAM/LIBERO/T6_EPISODE_SELECTION_V1
T5_RESULT_SHA256=a656aaef1528537527fe830ad7d4107138b29e8e254b5606b43c46a47e323e83
ELIGIBLE_MANIFEST_SHA256=c4ec46f7f1ede73556646da52501b3192c8d951b82d51ae8f9b18b70d15545f2
DATASET=<exact dataset name>
TASK_INDEX=<base-10 task index>
EPISODE_INDEX=<base-10 episode index>
START_FRAME=0
```

The immutable result is:

| Label | Suite / task / episode | Task | Length | Task payload SHA256 | Episode payload SHA256 |
|---|---|---|---:|---|---|
| S1 | Object / task 3 / episode 82 | `pick up the bbq sauce and place it in the basket` | 135 | `61071697d2905f3282f0be449981512ea13639547e41402b56df8100942f8856` | `08a3351e6a26cb3e1690663c2d875185468bbf44b7c60bbafa64573377b4a6b7` |
| S2 | Goal / task 2 / episode 70 | `open the top drawer and put the bowl inside` | 249 | `0d5d43dedb9b602ee435ab3654a66eac53429afb9ead5a95ddc87a3941040b00` | `02d397d9f8a9168f3032f66e83710595c9424a7323929ea264cb8d5ff502bad2` |
| S3 | LIBERO-10 / task 3 / episode 259 | `turn on the stove and put the moka pot on it` | 228 | `2cd3fff8b786308553138be1a90b8219122b1e308b784e2652ddb811e1ef6f20` | `058ae9b8c4575e89dbc41c059d994ca69b3527b2c53477336da94d96ace76a3b` |

The canonical selected manifest uses schema
`sana-wam-libero-t6-cross-suite-selection-v1`. Its exact top-level keys are
`eligible_manifest_sha256`, `samples`, `schema_version`, `selection_rule`,
`suite_order`, and `t5_predecessor_result_sha256`. `suite_order` is exactly the
three full dataset names in Object, Goal, LIBERO-10 order. Each sample row has
exactly `assets`, `dataset`, `episode_index`, `episode_length`,
`episode_selection_payload_sha256`, `label`, `start_frame`, `task`,
`task_index`, and `task_selection_payload_sha256`. Its selection-rule value is
exactly:

```text
in frozen config suite order object,goal,10; within each suite rank tasks by task payload SHA256 and take first; within selected task rank episodes by episode payload SHA256 and take first
```

It binds the eligible-manifest SHA, T5 predecessor, suite order, selection
rule, all selected identities, payload hashes, lengths, and raw-asset pins.
Serialization uses the same canonical JSON rules as the eligible manifest and
has exact byte length 2,923.

- Selected manifest SHA256:
  `e96c234997bbe05edae7509fa6f6007d20a4f5b098e0d6e07dedeafc38e6a9c5`.

No suite, task, episode, label, or order may be replaced because of episode
length, temporal padding, prompt length, preparation cost, pre-loss, runtime,
or outcome.

## 5. Selected raw assets

All paths below are relative to the dataset root named in the first column.
Every path must be a regular non-symlink file and match its pin before CUDA or
model construction, and must be re-hashed after sample materialization.

| Label | Dataset | Relative path | SHA256 |
|---|---|---|---|
| S1 | Object | `data/chunk-000/episode_000082.parquet` | `f25ac74111c22c9db0ff96d6021484f8a0e31d2e4413d47a160b83a0489cd1e9` |
| S1 | Object | `videos/chunk-000/observation.images.image/episode_000082.mp4` | `38294e54072c35ff4fc7edf963b688bd90d463e4a1ba623aca680ce2106b6ee7` |
| S1 | Object | `videos/chunk-000/observation.images.wrist_image/episode_000082.mp4` | `f935b70c9925f1a57bc4df7aa8c6519c614df2ece97661fa18d12d0a3788a923` |
| S2 | Goal | `data/chunk-000/episode_000070.parquet` | `6ce3cf58f533871b6413ea0e795f302060aca3a87b041f0110c77494a6d4b66b` |
| S2 | Goal | `videos/chunk-000/observation.images.image/episode_000070.mp4` | `af716b54f26f8cd284e715ab2335ecf86c59b5cc89706a7c51165094a5899b2b` |
| S2 | Goal | `videos/chunk-000/observation.images.wrist_image/episode_000070.mp4` | `ded7a8d03de99df3c720c3e74f87403378818a92f1cbea6e37a4d37d11fe1027` |
| S3 | LIBERO-10 | `data/chunk-000/episode_000259.parquet` | `de9e09a471d8050a5624f016f765c61fba7514829990b59de37c4e3bae0eec37` |
| S3 | LIBERO-10 | `videos/chunk-000/observation.images.image/episode_000259.mp4` | `c11a5751444371331df5b24ab76b85212e2979636d803afd2e0f59a002bd0804` |
| S3 | LIBERO-10 | `videos/chunk-000/observation.images.wrist_image/episode_000259.mp4` | `8800f488e8cc6f71cdfdf66d6fc531f92c54f54edf55da1403c60cb064f91786` |

## 6. Frozen training core, preparation, and exact budget

The training path remains T5 exactly:

- fresh initialization seed `20260806`;
- fixed loss-recipe seed `20260826` before every forward;
- Spatial task 0 / episode 0 / start frame 0 as the only optimization sample;
- action loss weight 1.0 and video loss weight 0.0;
- one persistent FP32-master AdamW optimizer;
- learning rate `1e-4`, betas `(0.9, 0.95)`, weight decay 0;
- global gradient clip bound 1.0;
- BF16 model projection after every optimizer step;
- exactly 20 updates;
- no scheduler, warmup, accumulation, early stopping, or SANA-WAM checkpoint.

Preparation consists of exactly four independent single-sample calls, in this
order: training, S1 Object, S2 Goal, S3 LIBERO-10. Prompt wrapping must equal
`format_prompt_for_inference(task_name)` for each sample. The four prepared
mappings must remain independent, pairwise non-aliasing, and immutable. They
may not be batched, concatenated, padded to the training prompt, or reused.
Their context shapes, sequence lengths, and digests are reported outside the
frozen top-level training-input projection.

Forward and update order is exactly:

1. S1, S2, and S3 update-free pre probes;
2. training-sample update-free pre probe;
3. 20 training-sample forward/backward/clip/master-sync/AdamW/BF16-project
   updates, preserving the T5 order exactly;
4. training-sample update-free post probe immediately after update 20;
5. S1, S2, and S3 update-free post probes.

The exact budget is:

- 4 `prepare_inputs` calls;
- 28 architecture forwards: 20 training forwards and 8 measurements;
- 6 cross-suite measurement forwards;
- 20 backward calls;
- 20 optimizer steps;
- 1 optimizer and 1 persistent FP32-master set;
- 0 post-probe re-preparations;
- 0 S1/S2/S3 tensors in backward or update.

Each forward resets to recipe seed `20260826` while forking and restoring the
caller's Python, NumPy, Torch CPU, and Torch CUDA RNG states. All 28 captured
recipe signatures must equal the frozen signature in Section 2.

## 7. Validity and attribution gates

A scientific verdict is allowed only if all harness gates pass:

1. Source, runner, config, Sana, T5 RESULT, selected-row stats, dataset
   metadata, and raw-asset identities equal their pins.
2. The canonical eligible and selected manifests reproduce exactly, including
   fixed suite order and the dataset-qualified Goal episode-82 exclusion.
3. Each selected `(dataset, task, episode, start=0)` occurs exactly once in the
   live dataset registry and matches its frozen task text and episode length.
4. Training is prepared first; all four prompt contexts bind their exact task
   strings, are mutually distinct, and are not aliased or reused.
5. Every prepared tensor retains its object identity and version. No prepared
   mapping aliases another mapping.
6. Every update-free probe block preserves model parameters, FP32 masters,
   buffers, module modes, gradients, optimizer state, prepared tensors, and
   caller RNG state. Optimizer state is empty before training; all Adam counters
   are exactly 20 afterward.
7. All losses are finite and non-negative, each cross-suite pre loss is
   strictly positive, and total loss equals action loss within the frozen T5
   tolerance.
8. The exact 4/28/20/20 preparation/forward/backward/optimizer counts hold,
   with no cross-suite tensor entering backward or update.
9. The complete current training-core projection equals the frozen T5 object
   and SHA256 exactly.

Any core drift or identity, selection, state-mutation, aliasing, prompt,
recipe, finite-value, count, or runtime violation prevents a scientific
verdict. Such a run writes and freezes only `FAILED.json`.

## 8. Pre-registered scientific gate

For each suite sample define:

```text
ratio_i = post_action_loss_i / pre_action_loss_i
improved_i = post_action_loss_i < pre_action_loss_i
```

The sole primary gate is:

```text
median(ratio_1, ratio_2, ratio_3) <= 0.95
AND
sum(improved_i) >= 2
```

- A valid run with exact T5 core reproduction and a passing primary gate is
  `T6_CROSS_SUITE_TRANSFER_GO`.
- A valid run with exact T5 core reproduction whose primary gate misses is
  `T6_CROSS_SUITE_TRANSFER_INCONCLUSIVE`.
- A run failing any validity or attribution gate has no scientific verdict and
  freezes `FAILED.json`.

The terminal report must include all three paired pre/post total and action
losses, absolute deltas, ratios, relative drops, improvement flags, sorted
ratios, median, improved count, exact suite/sample identities, prompt/context
diagnostics, recipe signatures, source and asset identities, exact execution
counts, and training-core comparison. Best/minimum loss, post-hoc sample
selection, reranking, success-rate claims, and automatic reruns are forbidden.

## 9. Root and execution boundary

The only allowed root form is:

```text
/DATA/share/sana_wam_libero_nonformal_screens/t6/<execution-commit-12>/libero-t6-crosssuite3-fixed20-<32-lowercase-hex nonce>
```

The namespace ancestors must be real non-symlink directories. The execution
root must be created exclusively, may never be reused, renamed, removed, or
overwritten, and must terminalize fail-closed. A valid result contains only
`RESULT.json`; an execution failure contains only `FAILED.json`. Terminal root
mode is `0500` and terminal JSON mode is `0400`.

This screen permits one H200 GPU, offline local base assets, fresh model
construction, four real-sample preparations, six update-free cross-suite
probes, and the frozen 20-step loss-space micro-update. It forbids simulator
execution, rollout, benchmark evaluation, checkpoint load/save, formal
training, admission, deployment, and token namespaces. Execution requires a
separate authority fixing source commit, runner SHA, GPU identity, nonce, and
fresh root.

## 10. Interpretation boundary

A GO would establish only that, under one initialization and one fixed
stochastic recipe, the 20 Spatial task-0/episode-0 updates reduced action loss
on at least two of the three frozen Object/Goal/LIBERO-10 samples. Each suite
has only one probe. Suite, task language, objects, visual observations,
episode length, and temporal coverage change together, so T6 is not a pure
suite, language, or visual-factor isolation.

All probes are still members of the configured all-four-suite data and
selected-row normalization population. They are held out from optimizer
updates only; they are not a strict dataset or normalization holdout. A lower
LIBERO-10 loss does not show completion of its multi-stage instruction. T6
does not establish cross-suite distribution-level generalization,
cross-recipe cross-suite robustness, rollout success, LIBERO success rate,
stable optimization, formal training readiness, admission, or deployment.

The three selected samples become consumed evidence when their first losses
are revealed. No outcome permits replacing them or automatically creating a
fresh root.

## 11. Frozen successor logic

- If T6 is GO, the loss-transfer breadth ladder is complete. The next smallest
  separately pre-registered architecture question is four-suite cyclic
  micro-learnability: one fixed Spatial/Object/Goal/LIBERO-10 training sample
  enters a fixed update cycle, while a newly mechanically selected sample from
  each suite remains update-free. This does not itself authorize that
  successor, formal training, or rollout.
- If T6 is INCONCLUSIVE with exact core reproduction, cross-suite breadth does
  not advance. The selected samples may not be replaced or rerun. A subsequent
  diagnostic, if separately authorized, is suite-local isolated
  self-learnability on the consumed S1/S2/S3 identities to distinguish absent
  Spatial-to-suite transfer from an unlearnable suite data path.
- If T6 is FAILED, no transfer interpretation is allowed. Only the root cause
  may be diagnosed; any rerun requires a corrected frozen source revision,
  separate authorization, a fresh nonce, and a never-created root.

## 12. Result placeholder

No T6 execution root, GPU identity, source commit, runner SHA, loss, verdict,
or RESULT artifact exists at pre-registration time. Populate this section only
from the first valid terminal root. If the first authorized root fails, record
the immutable failure root and stop.
