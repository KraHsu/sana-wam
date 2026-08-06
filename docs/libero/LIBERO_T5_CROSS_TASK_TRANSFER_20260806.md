# LIBERO T5 Same-Suite Cross-Task Transfer Screen

Status: **executed / frozen GO**

Date: 2026-08-06

## 1. Scientific question

T5 asks one narrow question: after exactly 20 action-only updates on the fixed
LIBERO Spatial training sample (task 0 / episode 0 / start frame 0), does the
same freshly initialized production-shaped AR architecture reduce paired action
loss on three mechanically selected samples from three distinct non-training
Spatial tasks?

This is the next bounded architecture screen after T4. It changes the probe
axis from distinct episodes of task 0 to distinct tasks and episodes within the
same LIBERO Spatial suite. The model, initialization, optimizer, loss recipe,
training sample, update loop, and training-core projection remain frozen.

## 2. Architecture and immutable predecessor

The model under test is the full production-shaped `DualSystemARArchitecture`,
not the reduced CACH-A4 scaffold. Its SANA video backbone remains frozen. The
only trainable roots are the same four roots used by T2-T4:

- `action_backbone`
- `proprio_encoder`
- `proprio_video_embed`
- `proprio_action_embed`

T5 must verify these predecessor pins before model construction:

- T4 RESULT:
  `/DATA/share/sana_wam_libero_nonformal_screens/t4/7db8182ef46f/libero-t4-heldout3-fixed20-36120c596971575d286d378df42ac564/RESULT.json`
- T4 RESULT SHA256:
  `4c33aaff6d202068d77efc0ac406c74198c56e72527cfabdde046fc9a3a4b6f4`
- T4 source commit:
  `7db8182ef46fce367f04f66bfbc34690fa86592b`
- T4 runner SHA256:
  `3d449761861b7fe75816d4d186b2e82402c160422eeb5f490fbb61ed274f088c`
- Reused config:
  `configs/benchmarks/libero/train_libero_ar_baseline.yaml`
- Config SHA256:
  `5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`
- Sana gitlink:
  `16b9cec673e3335724ba2d8db25de7f9ed229292`
- Frozen T3/T4 training-core projection SHA256:
  `e34a2dd0ae2dfe28303dcd9baf64ffc5800d002b0b7646b89dde2aafa132c352`

The runner must reconstruct the T3 and T4 projections from their frozen
RESULTs, require exact equality between them, then require the current T5
training-core projection to equal the frozen T4 object and SHA exactly.

## 3. Frozen candidate population

Dataset: `libero_spatial_no_noops_1.0.0_lerobot`.

Metadata pins:

- `meta/tasks.jsonl` SHA256:
  `399841cf8e861052d3fb7214ed619b5a10b4eded2d6f107e7fed338a233dd0e1`
- `meta/episodes.jsonl` SHA256:
  `690349688e96d984007bf2cca3ccf2683dfc809f03dab55e1159e4cc71f150b7`

The eligible population consists of all start-frame-0 episodes belonging to
task indices 1 through 9. Task 0 is excluded because it is the optimization
task. The runner must rebuild an exact canonical manifest containing 9 tasks
and 386 episodes. Serialization is UTF-8, sorted compact JSON,
`ensure_ascii=True`, `allow_nan=False`, with no final LF.

- Eligible schema: `sana-wam-libero-t5-cross-task-eligible-v1`
- Eligible manifest SHA256:
  `7b15d57b58ef398e7f0c3cd9371db03bd3312e4a4f3c3168f356df0f612bba45`

## 4. Mechanical two-stage selection

Task selection payload, ASCII with a final LF:

```text
SANA-WAM/LIBERO/T5_TASK_SELECTION_V1
T4_RESULT_SHA256=4c33aaff6d202068d77efc0ac406c74198c56e72527cfabdde046fc9a3a4b6f4
ELIGIBLE_MANIFEST_SHA256=7b15d57b58ef398e7f0c3cd9371db03bd3312e4a4f3c3168f356df0f612bba45
DATASET=libero_spatial_no_noops_1.0.0_lerobot
TASK_INDEX=<base-10 task index>
START_FRAME=0
```

Rank all nine task payload SHA256 values ascending and take the first three.
Within each selected task, rank this episode payload ascending and take the
first episode:

```text
SANA-WAM/LIBERO/T5_EPISODE_SELECTION_V1
T4_RESULT_SHA256=4c33aaff6d202068d77efc0ac406c74198c56e72527cfabdde046fc9a3a4b6f4
ELIGIBLE_MANIFEST_SHA256=7b15d57b58ef398e7f0c3cd9371db03bd3312e4a4f3c3168f356df0f612bba45
DATASET=libero_spatial_no_noops_1.0.0_lerobot
TASK_INDEX=<base-10 task index>
EPISODE_INDEX=<base-10 episode index>
START_FRAME=0
```

The immutable result, in task-hash rank order, is:

| Label | Identity | Length | Task payload SHA256 | Episode payload SHA256 |
|---|---|---:|---|---|
| H1 | task 7 / episode 36 | 116 | `0a028e007220594e3d743dbd11db14944ca281ae37db8248ac7cf257cb742415` | `089747625a57d2d60e853d5d5c1c0c43ad6a01b477dbede629dda40933defcc9` |
| H2 | task 1 / episode 325 | 131 | `0ed2d254a2b9424e0c4a5dadc44532e66081c215edb6d651949c50c6e70d4432` | `0a92dadc9f71161d6bc9e40476fbceed1d9ebc8bb297f111b402d17c8b9a50ca` |
| H3 | task 4 / episode 11 | 84 | `34e97146b609f1ff31d2b9cf44a2e86c3fc84205383535f7bdb0d2280b8b7ff2` | `02dbc353aaa077f46f0ee7cb4513417a6343c20146d2abfbf197279f35bbd13e` |

The canonical selected manifest has schema
`sana-wam-libero-t5-cross-task-selection-v1`, length 2697 bytes, and SHA256
`e60eee38c8b4ba9f9e0ffe49eac3d102ced497981404ad3613ac30fa0f324886`.
No sample may be replaced because of length, padding, pre-loss, or outcome.
In particular, episode 11's shorter length and additional padding are part of
the frozen mechanical selection.

## 5. Selected raw assets

All paths are under
`/DATA/share/LIBERO/libero/libero_spatial_no_noops_1.0.0_lerobot/`.

| Sample | Relative path | SHA256 |
|---|---|---|
| H1 | `data/chunk-000/episode_000036.parquet` | `09d6f24419a6d70c766c42035c2cf993a59663070f1989a9ae54fa4cc83bc8f4` |
| H1 | `videos/chunk-000/observation.images.image/episode_000036.mp4` | `a84cc4614ed6d7d08c9945567e7d3675e9ff7afa19113b9eaa5f68e3fe8af419` |
| H1 | `videos/chunk-000/observation.images.wrist_image/episode_000036.mp4` | `5aca422da74b6d059a8faac3c7e7ce1fa5e853c1c98486b106ba94c092a22114` |
| H2 | `data/chunk-000/episode_000325.parquet` | `a7f51cc301d4e32a15717d6fea1384f45416d5fc5ab8e4183f7dee7c57b291d3` |
| H2 | `videos/chunk-000/observation.images.image/episode_000325.mp4` | `edd51702dfcf53b60db9e12be38d18acad88678657e4cf063ec01da7517ac8d6` |
| H2 | `videos/chunk-000/observation.images.wrist_image/episode_000325.mp4` | `8511a776845f75aefeb1a2b904d00d668a6c5ddc25d03f7480474c6436ba2e58` |
| H3 | `data/chunk-000/episode_000011.parquet` | `5659083b099c426d2e9f4dde654fe3e556bde369c5c99f40eb0e521373b99c7f` |
| H3 | `videos/chunk-000/observation.images.image/episode_000011.mp4` | `be57d261c1201b7258599f2b7f3b597e0464616daa8c92b8ffc4a5ef64c39a9a` |
| H3 | `videos/chunk-000/observation.images.wrist_image/episode_000011.mp4` | `a8ad057219920267213f88a801dc7aecf7f864c05b2beb327ea7820031167102` |

Each metadata and selected asset path must be a regular non-symlink file and
must match its pin before CUDA/model construction. Metadata and assets are
re-hashed after materialization.

## 6. Frozen training core and schedule

- Fresh initialization seed: `20260806`.
- Fixed loss-recipe seed before every forward: `20260826`.
- Training sample: Spatial task 0 / episode 0 / start frame 0.
- Loss weights: action 1.0, video 0.0.
- Optimizer: persistent FP32-master AdamW, learning rate `1e-4`, betas
  `(0.9, 0.95)`, weight decay 0, global gradient clip 1.0, BF16 projection
  after every optimizer step.
- Updates: exactly 20; only the training sample enters backward/update.
- No SANA-WAM checkpoint load or save.

Preparation is exactly four independent single-sample calls: training first,
then H1, H2, H3. Prompt wrapping must equal
`format_prompt_for_inference(task_name)` for each sample. Context lengths may
differ, so the four prepared mappings may not be batched, padded to the
training prompt, aliased, or reused. Context shape, `seq_lens`, and digest are
reported outside the frozen top-level training `inputs` projection.

Forward order is fixed:

1. H1, H2, H3 update-free pre probes.
2. Training-sample pre probe.
3. Twenty training-sample forward/backward/AdamW updates.
4. Training-sample update-free post probe.
5. H1, H2, H3 update-free post probes.

Exact budget:

- 4 `prepare_inputs` calls.
- 28 architecture forwards: 20 training and 8 measurements.
- 20 backwards.
- 20 optimizer steps.
- 0 post-probe re-preparations.
- 0 cross-task samples in backward or update.

## 7. Validity gates

A scientific verdict is allowed only if all harness gates pass:

- source/config/Sana/T1-T4/data/stats identities match their pins;
- live metadata reproduces the eligible and selected manifests exactly;
- each selected `(task, episode, start=0)` occurs exactly once;
- all four prompts match their task text and prepared contexts are pairwise
  distinct;
- prepared tensors do not alias across samples and retain identity/version;
- each update-free probe preserves parameters, masters, buffers, module modes,
  gradients, optimizer state, prepared tensors, and caller RNG state;
- all 28 forwards reproduce loss-recipe signature
  `ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad`;
- losses and optimizer state are finite and total loss equals action loss;
- exact 28/20/20/4 counts hold;
- current training-core projection equals frozen T4 exactly.

Any identity, state, count, runtime, or core mismatch freezes `FAILED.json`
with schema `sana-wam-libero-t5-cross-task-failure-v1`; it is not a scientific
INCONCLUSIVE result.

## 8. Frozen scientific gate

For each H sample, compute `ratio_i = post_action_loss_i / pre_action_loss_i`.
The primary gate is:

```text
median(ratio_1, ratio_2, ratio_3) <= 0.95
AND
strictly improved sample count >= 2
```

- Pass: `T5_CROSS_TASK_TRANSFER_GO`.
- Miss with every validity gate intact:
  `T5_CROSS_TASK_TRANSFER_INCONCLUSIVE`.

No sample replacement, reranking, automatic rerun, root reuse, or best-of-N is
allowed after observing losses. A GO advances to a separately preregistered
cross-suite update-free loss-transfer screen. An INCONCLUSIVE result stops this
branch for review; any cyclic multi-task follow-up requires a new contract.

## 9. Root and execution boundary

The only allowed root form is:

```text
/DATA/share/sana_wam_libero_nonformal_screens/t5/<execution-commit-12>/libero-t5-crosstask3-fixed20-<32-lowercase-hex nonce>
```

It must be created exclusively, written once, and terminalized fail-closed.
Terminal directory mode is `0500`; terminal JSON mode is `0400`. A terminal
root is immutable and cannot be reused.

This screen permits one H200 GPU, offline local assets, fresh model
construction, four real-sample preparations, update-free probes, and the fixed
20-step synthetic-loss-space micro-update. It forbids simulator execution,
rollout, checkpoint load/save, formal training, benchmark evaluation,
admission, deploy, and token namespaces.

## 10. Interpretation boundary

A GO would establish only same-suite, distinct-task-plus-distinct-episode,
update-heldout action-loss transfer for these three frozen samples. It is not a
pure language-task isolation because visual episodes also differ. It is not a
strict dataset or normalization holdout, cross-suite transfer, rollout success,
benchmark success rate, stable training, formal admission, or deployment
evidence.

## 11. Frozen result

The first and only T5 execution completed as a valid run.

- Scientific verdict: `T5_CROSS_TASK_TRANSFER_GO`.
- Source commit: `5150693a0751200ef431968a863c69f8ca08a7df`.
- Runner SHA256:
  `a4bbdcf752aa0a34a43ea4f51e7875f7fd160985280a7274b29e36350d9605c1`.
- GPU: physical GPU 0,
  `GPU-1ec28cfb-f501-23f3-f865-275a744ca053` (NVIDIA H200).
- Nonce: `cf7dd8a1b1cef03511d2026a48e4a271`.
- Immutable root:
  `/DATA/share/sana_wam_libero_nonformal_screens/t5/5150693a0751/libero-t5-crosstask3-fixed20-cf7dd8a1b1cef03511d2026a48e4a271`.
- RESULT SHA256:
  `a656aaef1528537527fe830ad7d4107138b29e8e254b5606b43c46a47e323e83`.
- Terminal modes: root `0500`, `RESULT.json` `0400`.

Paired action-loss results:

| Label | Identity | Pre | Post | Post/pre | Improved |
|---|---|---:|---:|---:|---|
| H1 | task 7 / episode 36 | `15.97777271` | `2.37489486` | `0.14863742` | yes |
| H2 | task 1 / episode 325 | `16.58133507` | `1.88198090` | `0.11349996` | yes |
| H3 | task 4 / episode 11 | `19.20838547` | `2.88868594` | `0.15038671` | yes |

The median post/pre ratio was `0.1486374165`; all three samples improved. This
cleared the frozen `median <= 0.95` and `improved_count >= 2` primary gate.

All validity gates passed:

- eligible and selected manifest SHA256 values reproduced exactly;
- the current training-core projection equaled frozen T4 exactly, with expected
  and observed SHA256 both
  `e34a2dd0ae2dfe28303dcd9baf64ffc5800d002b0b7646b89dde2aafa132c352`;
- the training-sample result reproduced action loss `13.67971897 -> 1.73319125`
  and ratio `0.1266978697`;
- exact counts were 4 preparations, 28 forwards, 20 backwards, and 20 AdamW
  steps, with 0 cross-task samples entering backward/update and 0
  post-probe re-preparations;
- all 28 loss-recipe signatures were identical to the frozen signature;
- 560 persistent FP32 masters remained finite, reached Adam step 20, and all
  four trainable roots changed in both FP32 master and BF16 projected probes;
- training/H1/H2/H3 prompt contexts were distinct. Their token lengths were
  33/33/36/35 respectively;
- no simulator, benchmark evaluator, formal training, or SANA-WAM checkpoint
  load/save ran; GPU 0 was released after terminalization.

This GO advances the architecture-validation line to a separately frozen
cross-suite loss-transfer question. It does not alter the interpretation limits
in Section 10 and does not authorize formal training or benchmark claims.
