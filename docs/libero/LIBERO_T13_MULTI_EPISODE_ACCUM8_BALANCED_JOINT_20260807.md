# LIBERO T13：Multi-Episode Accumulation-8 Balanced-JOINT

状态：**source implementation；尚未执行，未创建运行 root。**

T13 只验证一个核心问题：在 T12 的四个固定任务上，把单个 update/heldout episode
扩展为每任务两个 update episode 与两个 heldout episode，并把每个 macro 的八次
singleton micro-backward 对齐 production 的 `batch_size=1`、
`gradient_accumulation_steps=8` 边界后，balanced-JOINT 是否仍能同时降低全部 16 个
样本的 action loss。

本 screen 不是正式训练、benchmark 评测、simulator rollout、admission 或部署。
它允许未来单 GPU 执行时加载冻结 SANA base construction checkpoint，但禁止加载或
保存 SANA-WAM training checkpoint。

## 1. 直接前序

直接前序固定为 T12：

- source commit：`bf4e6f43395e0ba2c177d81856ad37f87c6e91fe`
- runner SHA256：
  `baf19d25d39412a6be02ce35d5c06cae1e8a3089d1c005c35768ca0e0f3581d0`
- RESULT SHA256：
  `c1f045e62c854ab897305fbed504e765bd229d02479d2d11861f210d26abfe08`
- immutable root：
  `/DATA/share/sana_wam_libero_nonformal_screens/t12/bf4e6f43395e/libero-t12-new-task-balanced-joint-fixed20-09032d945cdf4dac558ab13b9dabafdd`
- T12 selection manifest SHA256：
  `7f9ba75f17c7bdeaf3e68a70eaa9d752b26cf057250ff7f9c3fa37e2a895fe89`
- inherited T12 eligible manifest：`53728` bytes，SHA256
  `50d97577a4e0120ef9e6b42e1a227321b05ed19450626dc3dfa0c0810c997c47`

T13 必须只读验证 T12 root 权限、唯一 RESULT、source/result/config pins、有效 verdict、
预算和未加载/保存 SANA-WAM checkpoint 的语义。T10/T11 仅作为更早 lineage，不替代
T12 的直接前序地位。

## 2. 机械选择

四个 task identity 原样继承 T12：Spatial task 5、Object task 9、Goal task 4、
LIBERO-10 task 6。候选集排除完整 T1–T12 model-facing sample identity；不使用 episode
length filter，`start_frame=0`。

候选 episode 数为：Spatial `41`、Object `48`、Goal `45`、LIBERO-10 `27`，合计
`161`。每个任务按下列 ASCII payload 的 SHA256 排序；payload 包含 final LF：

```text
SANA-WAM/LIBERO/T13_MULTI_EPISODE_ACCUM8_SELECTION_V1
T12_RESULT_SHA256=c1f045e62c854ab897305fbed504e765bd229d02479d2d11861f210d26abfe08
T12_SELECTION_MANIFEST_SHA256=7f9ba75f17c7bdeaf3e68a70eaa9d752b26cf057250ff7f9c3fa37e2a895fe89
DATASET={dataset}
TASK_INDEX={task_index}
EPISODE_INDEX={episode_index}
START_FRAME=0
```

rank 1/2 是 update，rank 3/4 是 heldout。Canonical selection manifest 使用 UTF-8、
`ensure_ascii=True`、`allow_nan=False`、sorted keys、compact separators、无末尾 LF：

- schema：`sana-wam-libero-t13-multi-episode-accum8-selection-v1`
- bytes：`16763`
- SHA256：`fa00b477558eb26ec5657b87e11e4d6023181f88ea023889e2e86d94b3fd5305`

| suite | update labels / episodes / lengths | heldout labels / episodes / lengths |
|---|---|---|
| Spatial | A12 `191/96`, A13 `144/92` | H12 `130/96`, H13 `397/101` |
| Object | A14 `19/161`, A15 `170/160` | H14 `394/148`, H15 `34/224` |
| Goal | A16 `75/116`, A17 `38/93` | H16 `154/113`, H17 `195/104` |
| LIBERO-10 | A18 `70/416`, A19 `10/383` | H18 `23/455`, H19 `306/505` |

每个样本的 parquet、head-camera MP4 与 wrist-camera MP4 SHA256 固化在 runner；
48 个 asset SHA 必须互不重复。任何 metadata、payload、selection 或 asset pin 不一致
都 fail-closed。

## 3. Production-matched accumulation-8 拓扑

- 只允许一个 `JOINT` arm。
- config 必须显式满足 `training.batch_size=1`、
  `training.gradient_accumulation_steps=8`。
- 20 个 macro steps。
- 每个 macro 严格按 `A12,A13,A14,A15,A16,A17,A18,A19` 做八次独立 singleton
  forward/backward；禁止拼接 prepared tensor 或 context。
- 每个 singleton loss 权重精确为 `0.125`，八项和精确为 1。
- 八次 micro-backward 之间禁止修改 model parameter、FP32 master 或 optimizer state。
- 第八次 backward 后只允许一次 gradient clip、一次 FP32-master AdamW step、一次
  BF16 projection。
- 每个 update sample raw exposure 为 `20`，累计 loss coefficient 为 `2.5`；每个
  task/suite 的两个 update sample 合计 coefficient 为 `5.0`。
- heldout sample exposure 与 coefficient 都为 `0`。

每个 label 单独执行 `architecture.prepare_inputs([raw_sample])`，保留生产路径原生
typed mask；不做 batch 拼接。四个 context group 必须分别满足 4-way 一致：

```text
(A12,A13,H12,H13)
(A14,A15,H14,H15)
(A16,A17,H16,H17)
(A18,A19,H18,H19)
```

Summary 必须按 dataset 分组为 list，禁止使用 `{dataset: row}` 覆盖同 suite 的第二个
样本。

## 4. 精确预算

- `prepare_inputs`：`16`
- architecture measurement forwards：`32`
- architecture training forwards：`160`
- architecture forwards 总计：`192`
- backward：`160`
- optimizer steps：`20`
- no-intra-macro mutation evidence：`20/20`

所有 pre/post probes 必须保持 model/master/optimizer/buffer/mode/gradient、prepared
tensors 与调用方 RNG 不变；所有 forward 使用同一冻结 recipe signature。

## 5. 严格科学分类

执行有效性 verdict 固定为：

```text
T13_MULTI_EPISODE_ACCUM8_JOINT_ARM_VALID
```

有效运行按全部未舍入 `post/pre` ratio 严格顺序分类：

1. 8 个 update 与 8 个 heldout ratio 全部 `<1`：
   `T13_MULTI_EPISODE_ACCUM8_BALANCED_JOINT_REPLICATED`
2. 否则，只要任一 update ratio `>=1`：
   `T13_MULTI_EPISODE_ACCUM8_JOINT_TRAINING_FIT_FAILURE`
3. 否则：`T13_MULTI_EPISODE_ACCUM8_JOINT_HELDOUT_GAP`

边界 `1.0` 不算改善；NaN、inf、零或负 loss/ratio fail-closed。每 suite 四个 ratio 的
均值 `q` 与 T12 `q` 的比较只作诊断，排除在分类器之外。

## 6. Source、root 与解释边界

本阶段只准备三个新增文件：

- `docs/libero/LIBERO_T13_MULTI_EPISODE_ACCUM8_BALANCED_JOINT_20260807.md`
- `scripts/smoke_libero_ar_t13_multi_episode_accum8_balanced_joint_gpu.py`
- `tests/test_smoke_libero_ar_t13_multi_episode_accum8_balanced_joint_gpu.py`

未来执行 namespace：

```text
/DATA/share/sana_wam_libero_nonformal_screens/t13/{source_commit[:12]}/libero-t13-multi-episode-accum8-balanced-joint-fixed20-{nonce}
```

root 必须 fresh one-shot；terminal root `0500`，唯一 `RESULT.json` 或 `FAILED.json`
为 `0400`。禁止 root 复用、覆盖、自动重跑与 post-freeze mutation。

即使得到 `REPLICATED`，也只支持单 seed、四任务、短程 loss-space 下的多 episode
accumulation-8 保持结论；不支持 rollout success、LIBERO benchmark score、正式训练
readiness 或部署能力。当前文档不包含运行结果，因为 T13 尚未执行。
