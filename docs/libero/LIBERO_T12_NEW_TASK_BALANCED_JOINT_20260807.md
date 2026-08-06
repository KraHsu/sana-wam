# LIBERO T12：新任务 balanced-JOINT 复现

状态：**source staging；尚未执行，未创建运行 root。**

T12 只回答一个核心问题：T10/T11 已验证的 balanced simultaneous objective，
在四个此前没有进入 T1–T11 model-facing measurement 或 backward 的任务上，是否仍能
同时降低四个 update episode 与四个同任务、update-free episode 的 action loss。

本 screen 不是正式训练、benchmark 评测、rollout、部署或 admission；允许未来执行时
加载冻结 SANA base construction checkpoint，但禁止加载或保存 SANA-WAM training
checkpoint。

## 1. 直接前序与冻结继承

直接前序是 T11：

- source commit：`7d53d618234e3e37367c5b1e39e169742c2cf514`
- runner SHA256：
  `ab66fcf6052582631424c97f149c9187d942b1da253e201fc698142e5ad41b54`
- RESULT SHA256：
  `6da12f2e3622d4e6427070bfd10b3dab9550c338a834b7662309cefea98d7417`
- immutable root：
  `/DATA/share/sana_wam_libero_nonformal_screens/t11/7d53d618234e/libero-t11-same-task-new-episode-balanced-joint-fixed20-05e0d719fbc5c25e66ddf435cb47eba2`
- execution verdict：`T11_NEW_EPISODE_JOINT_ARM_VALID`
- scientific verdict：
  `T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT_REPLICATED`

T12 还保留 T10 underlying topology pins：

- aggregate source：`128e1be8cd48f8cef1b7c5f24d1bdecfbe45054a`
- aggregate runner SHA256：
  `b45e9536b35790c06e599ba919ed1370c64e4e1dff477019fab9f0ccc40b61c5`
- aggregate RESULT SHA256：
  `8fcd26ea3a59fe6e01cf8f279301c899ed5dab541ada9c57e2ce85888abd147b`

其余冻结继承为：

- config SHA256：
  `5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`
- Sana gitlink：`16b9cec673e3335724ba2d8db25de7f9ed229292`
- selected-row stats SHA256：
  `e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7`
- stats population SHA256：
  `7ed9772facf261299022e55169bcdaaa49fe3a7a20e0057e08cf419b5e584146`
- frozen recipe signature：
  `ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad`
- fresh initialization seed：`20260806`
- loss recipe seed：`20260826`

## 2. “新任务”的精确定义

`model-facing task` 定义为：T1–T11 中至少有一个属于该 dataset-qualified task 的
样本进入了 pre/post model measurement 或 backward。只在 metadata eligible manifest
中被枚举，不算 model-facing exposure；否则 T6 已枚举四个 suite 的全部 40 个任务，
T12 将没有可选任务。

冻结排除任务是：

| suite | 排除 task index |
|---|---|
| Spatial | `0, 1, 4, 7` |
| Object | `3` |
| Goal | `2` |
| LIBERO-10 | `3` |

Goal episode 82 因损坏数据契约继续排除，但它从未 model-facing，因此不把整个 Goal
task 5 判作已消费。Runner 还必须把 A8–H11 与完整 T1–T11 model-facing
`(dataset, task, episode, start)` 集合做显式不相交检查。

## 3. 双层机械选择

Eligible canonical JSON 使用 UTF-8、`ensure_ascii=True`、`allow_nan=False`、
sorted keys、compact separators，且 bytes 不附加 LF。

- schema：`sana-wam-libero-t12-new-task-balanced-joint-eligible-v1`
- suites / tasks / episodes：`4 / 33 / 1391`
- 每 suite task 数：`6 / 9 / 9 / 9`
- 每 suite episode 数：`254 / 408 / 391 / 338`
- canonical bytes：`53728`
- SHA256：
  `50d97577a4e0120ef9e6b42e1a227321b05ed19450626dc3dfa0c0810c997c47`

Task ranking payload 是 ASCII，每行及末行都以 LF 结束：

```text
SANA-WAM/LIBERO/T12_NEW_TASK_SELECTION_V1
T11_RESULT_SHA256=6da12f2e3622d4e6427070bfd10b3dab9550c338a834b7662309cefea98d7417
ELIGIBLE_MANIFEST_SHA256=50d97577a4e0120ef9e6b42e1a227321b05ed19450626dc3dfa0c0810c997c47
DATASET={dataset}
TASK_INDEX={task_index}
START_FRAME=0
```

Episode ranking payload 同样以 final LF 结束：

```text
SANA-WAM/LIBERO/T12_NEW_TASK_EPISODE_SELECTION_V1
T11_RESULT_SHA256=6da12f2e3622d4e6427070bfd10b3dab9550c338a834b7662309cefea98d7417
ELIGIBLE_MANIFEST_SHA256=50d97577a4e0120ef9e6b42e1a227321b05ed19450626dc3dfa0c0810c997c47
DATASET={dataset}
TASK_INDEX={task_index}
EPISODE_INDEX={episode_index}
START_FRAME=0
```

冻结规则原文：

```text
in frozen suite order spatial,object,goal,10; exclude every T1-T11 model-facing measured-or-backpropagated task identity; within each suite rank remaining tasks by exact task payload SHA256 and take first; within each selected task rank eligible episodes by exact episode payload SHA256 and assign first to update A8-A11 and second to heldout H8-H11
```

Selection manifest schema 为
`sana-wam-libero-t12-new-task-balanced-joint-selection-v1`；canonical bytes 为
`7044`，SHA256 为
`7f9ba75f17c7bdeaf3e68a70eaa9d752b26cf057250ff7f9c3fa37e2a895fe89`。

| label | suite / task | episode / length | task payload SHA256 | episode payload SHA256 |
|---|---|---:|---|---|
| A8 | Spatial / 5 | 202 / 97 | `390e72912baa953cddba592bf52ed7d95917295ec7fad58036fbc7cc7fdc9171` | `008d8d7c85b986bdfe1ed710e21790976c55bd394c17599c3c0f7eec6afe28cb` |
| H8 | Spatial / 5 | 226 / 103 | `390e72912baa953cddba592bf52ed7d95917295ec7fad58036fbc7cc7fdc9171` | `08f5db23ed09477086da0df378d7295b9df8f68e6efb0ca85f596bcc2055996e` |
| A9 | Object / 9 | 407 / 148 | `2c4d697fbfebb3a91288e1b499eba53ba755692cd06e0b7a6308866e90ab36dd` | `008b0edcf2727bc78f046a1a1cdf352bc0fa6dddca63e80cd5a37d166c0aeee5` |
| H9 | Object / 9 | 27 / 159 | `2c4d697fbfebb3a91288e1b499eba53ba755692cd06e0b7a6308866e90ab36dd` | `0134300bd200968998f9111d1ed95f422bb0198bcb7dabfe0871156e76a2295b` |
| A10 | Goal / 4 | 89 / 92 | `1b37e4af7c810362572f129a2295d6475c635d6b35ce0baf7a117106860bfe9f` | `006e635f2945db2c55133b3dacb0f26404be5dbaefb785c403e8f7ac452da960` |
| H10 | Goal / 4 | 387 / 121 | `1b37e4af7c810362572f129a2295d6475c635d6b35ce0baf7a117106860bfe9f` | `02c20c39e8e6696d1ae6f414efb995ee6d46b5e4ad0237f7625d6427784c2af3` |
| A11 | LIBERO-10 / 6 | 316 / 396 | `37f56c5d986865060b7e1b2fe1cedf1b18bb7bdcbb5ee4f929b8f09e4ec6d531` | `198b5cb255439ddf27d62451f1e108b62ed8c1d72268bedad35aba60debe2435` |
| H11 | LIBERO-10 / 6 | 314 / 383 | `37f56c5d986865060b7e1b2fe1cedf1b18bb7bdcbb5ee4f929b8f09e4ec6d531` | `1a2abad357d175d5afc557a0b77412acf69f89c9e05a31f20533c15ecc80de14` |

每个样本的 parquet、head-camera MP4 和 wrist-camera MP4 SHA256 都固化在 runner
和 selection manifest；24 个 asset SHA 互不重复。

## 4. 冻结执行拓扑与预算

- 单 arm，只允许 `JOINT`；无 SEQ、无 aggregate runner。
- 20 个 macro steps。
- 每个 macro 按 `A8,A9,A10,A11` 顺序做四次 forward/backward，每项 loss
  权重精确为 `0.25`。
- 四次 microback 之间禁止修改 model parameter、FP32 master 或 optimizer state；
  之后只允许一次 gradient clip、一次 FP32-master AdamW step、一次 BF16 projection。
- 每个 A raw exposure 为 20，累计有效 loss coefficient 为 5；H exposure 与
  coefficient 均为 0。
- 总预算：8 `prepare_inputs`、96 architecture forwards（16 measurement + 80
  training）、80 backward、20 optimizer steps。
- 20/20 macro 都必须保留 no-intra-macro mutation evidence。
- pre/post probes 必须保持 model/master/optimizer/buffer/mode/gradient、prepared
  tensors 与调用方 RNG 不变；所有 forward 使用同一冻结 recipe signature。

## 5. Validity 与严格三分支

执行有效性 verdict 固定为：

```text
T12_NEW_TASK_JOINT_ARM_VALID
```

任何 source/predecessor/config/data/manifest/asset/task/sample/root/GPU/state/count/
finite gate 失败，只能冻结 `FAILED.json`，不得给科学 verdict。

有效运行按未舍入 `rA=post_A/pre_A`、`rH=post_H/pre_H` 严格顺序分类：

1. 8/8 `rA,rH < 1`：`T12_NEW_TASK_BALANCED_JOINT_REPLICATED`
2. 否则，任一 `rA >= 1`：`T12_NEW_TASK_JOINT_TRAINING_FIT_FAILURE`
3. 否则：`T12_NEW_TASK_JOINT_HELDOUT_GAP`

边界 `1.0` 不算改善；NaN、inf、零或负 loss/ratio fail-closed。每个 suite 的
`q=(rA+rH)/2` 与冻结 T11 `q` 的比较只作难度/优化幅度诊断，不进入分类器。

## 6. Root、最小 source 与禁止项

只新增：

- `docs/libero/LIBERO_T12_NEW_TASK_BALANCED_JOINT_20260807.md`
- `scripts/smoke_libero_ar_t12_new_task_balanced_joint_gpu.py`
- `tests/test_smoke_libero_ar_t12_new_task_balanced_joint_gpu.py`

未来执行 namespace 为：

```text
/DATA/share/sana_wam_libero_nonformal_screens/t12/{source_commit[:12]}/libero-t12-new-task-balanced-joint-fixed20-{nonce}
```

必须 fresh one-shot、唯一 32 位小写十六进制 nonce，terminal root 冻结为 `0500`、
唯一 `RESULT.json` 或 `FAILED.json` 为 `0400`；禁止 root 复用、覆盖、自动重跑和
post-freeze mutation。

禁止真实 simulator、rollout、benchmark evaluation、正式训练、admission、deploy、
完整 2B training，以及 SANA-WAM checkpoint load/save。

## 7. 结果边界

若进入 `REPLICATED`，只说明 balanced-JOINT 的短程 action-loss 联合优化和同任务
新 episode 保持能力，在每个 suite 一个此前未 model-facing 的任务上复现。任务变化
同时带来 task text 与视觉 episode 变化，因此不能声称隔离了语言语义，也不能声称
zero-shot cross-task transfer。

本 screen 仍是 single-seed、每 suite 一对样本的 non-formal loss-space screen；任何
结果都不支持 rollout success、LIBERO benchmark score、正式训练 readiness 或部署能力。
