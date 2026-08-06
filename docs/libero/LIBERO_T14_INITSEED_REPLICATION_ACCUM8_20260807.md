# LIBERO T14：Fresh-Initialization-Seed Replication

状态：**已执行；valid frozen non-formal result；科学 verdict 为
`T14_INITSEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED`。**

T14 是 T13 的单轴复现。它只改变 SANA-WAM `Trainer`、进程和模型构造所用的
initialization seed，从 `20260806` 改为 `20260807`；数据选择、dataloader seed、
loss recipe、拓扑、优化器、预算、阈值和所有前序证据均保持不变。目标是判断 T13
的 16/16 短程 action-loss 改善是否依赖单一初始化。

本实验仍是单 GPU、real-data、non-formal architecture screen，不是正式训练、
LIBERO benchmark、simulator rollout、admission、部署或完整 2B 运行。允许使用冻结
SANA base construction checkpoint 构造 backbone；禁止加载或保存 SANA-WAM training
checkpoint。

## 1. 直接前序

直接前序固定为已冻结的 T13：

- source commit：`68fa88157973f59383a87be1cb3107f5824e64cf`
- runner SHA256：
  `ddbd141f3c32f9f89c4e936b1432d379cf56f0b886f84f38406fa4481654efb8`
- RESULT SHA256：
  `a75739991561965d212ad98cc2504cabf8696b2dcaf2e0ad5b46aa774ca00763`
- immutable root：
  `/DATA/share/sana_wam_libero_nonformal_screens/t13/68fa88157973/libero-t13-multi-episode-accum8-balanced-joint-fixed20-ce58b18994fa066b49b5e52bbd98b81e`
- execution verdict：`T13_MULTI_EPISODE_ACCUM8_JOINT_ARM_VALID`
- scientific verdict：`T13_MULTI_EPISODE_ACCUM8_BALANCED_JOINT_REPLICATED`

T14 必须只读重哈希 T13 runner、RESULT、root 和全部外部 asset，并验证 T13 的精确
预算、seed、selection manifest、16 个 ratio、四个 `q`、terminal mode 和禁止项。
T12 继续作为 underlying predecessor，但不能取代 T13 的直接前序地位。

## 2. 唯一实验轴

T14 config 必须是 production baseline config 的精确字节副本，除以下唯一一行：

```diff
-  seed: 20260806
+  seed: 20260807
```

固定 config：

- path：`configs/experiments/libero_t14_initseed_replication_accum8.yaml`
- SHA256：`36daaada09409c22ef2e57e3382ca2ef28c6847f9d2e043cad354464c3dde929`
- initialization seed：`20260807`
- dataloader seed：`20260806`（不变）
- training-loss recipe seed：`20260826`（不变）

runner 必须在任何模型或数据构造前验证上述三种 seed；不得把 dataloader 或 loss
recipe seed 随 initialization seed 一起移动。

## 3. 原样继承 T13 的数据与选择

T14 原样继承 T13 canonical selection manifest：

- schema：`sana-wam-libero-t13-multi-episode-accum8-selection-v1`
- bytes：`16763`
- SHA256：`fa00b477558eb26ec5657b87e11e4d6023181f88ea023889e2e86d94b3fd5305`
- eligible manifest：`53728` bytes，SHA256
  `50d97577a4e0120ef9e6b42e1a227321b05ed19450626dc3dfa0c0810c997c47`

不重新抽样，不新增或删除 episode，不改变 start frame 或 role：

| suite | update labels / episodes / lengths | heldout labels / episodes / lengths |
|---|---|---|
| Spatial | A12 `191/96`, A13 `144/92` | H12 `130/96`, H13 `397/101` |
| Object | A14 `19/161`, A15 `170/160` | H14 `394/148`, H15 `34/224` |
| Goal | A16 `75/116`, A17 `38/93` | H16 `154/113`, H17 `195/104` |
| LIBERO-10 | A18 `70/416`, A19 `10/383` | H18 `23/455`, H19 `306/505` |

runner 内冻结每个样本的 parquet、head-camera MP4 和 wrist-camera MP4 SHA256，共
48 个 role asset；任一内容、metadata、payload、顺序或 digest 漂移都 fail-closed。

## 4. 不变的 production-matched 拓扑与预算

- 只允许 `JOINT` arm。
- `training.batch_size=1`，`training.gradient_accumulation_steps=8`。
- 20 个 macro steps；每个 macro 按
  `A12,A13,A14,A15,A16,A17,A18,A19` 执行八次 singleton
  forward/backward。
- 每个 singleton loss coefficient 固定为 `0.125`；八项和为 1。
- 八次 micro-backward 之间 model parameter、FP32 master 与 optimizer state 必须满足
  预注册 no-mutation 契约；gradient 按 accumulation 语义正常累积。
- 第八次 backward 后只允许一次 gradient clip、一次 FP32-master AdamW step、一次
  BF16 projection。
- 每个 update sample 20 次 raw exposure、累计 coefficient `2.5`；heldout exposure
  和 coefficient 均为 0。
- 每个 label 单独调用一次 `architecture.prepare_inputs([raw_sample])`，不拼接
  prepared tensor 或 context。

精确预算：

- `prepare_inputs=16`
- measurement forwards `=32`
- training forwards `=160`
- architecture forwards `=192`
- backward calls `=160`
- optimizer steps `=20`
- no-intra-macro mutation evidence `=20/20`

## 5. 严格分类与诊断边界

有效执行 verdict 固定为：

```text
T14_INITSEED_REPLICATION_ACCUM8_JOINT_ARM_VALID
```

有效运行按全部未舍入 ratio 顺序分类：

1. 8 个 update 与 8 个 heldout ratio 全部 `<1`：
   `T14_INITSEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED`
2. 否则，只要任一 update ratio `>=1`：
   `T14_INITSEED_REPLICATION_ACCUM8_JOINT_TRAINING_FIT_FAILURE`
3. 否则：`T14_INITSEED_REPLICATION_ACCUM8_JOINT_HELDOUT_GAP`

边界 `1.0` 不算改善；NaN、inf、零或负 loss/ratio fail-closed。T14 每 suite 四个
ratio 的均值 `q` 与 T13 `q` 的比较只作初始化敏感性诊断，明确排除在 scientific
classifier 之外。

## 6. Source surface、root 与禁止项

本阶段只新增四个文件：

- `configs/experiments/libero_t14_initseed_replication_accum8.yaml`
- `docs/libero/LIBERO_T14_INITSEED_REPLICATION_ACCUM8_20260807.md`
- `scripts/smoke_libero_ar_t14_initseed_replication_accum8_gpu.py`
- `tests/test_smoke_libero_ar_t14_initseed_replication_accum8_gpu.py`

唯一执行 namespace：

```text
/DATA/share/sana_wam_libero_nonformal_screens/t14/{source_commit[:12]}/libero-t14-initseed-replication-accum8-balanced-joint-fixed20-{nonce}
```

root 必须 fresh、排他、one-shot。terminal root 设为 `0500`，且只允许一个
`RESULT.json` 或 `FAILED.json`（`0400`）。禁止复用、覆盖、自动重跑和 post-freeze
mutation；失败只能冻结当前 root，若需要修复必须另行审计并使用新的 source revision
与 fresh root。

禁止正式训练、benchmark evaluation、simulator/rollout、真实机器人、SANA-WAM
training checkpoint load/save、admission、deploy、AV2 或 Global Stage 3。即使得到
`REPLICATED`，结论也只覆盖第二个 initialization seed、同四任务、同 16 episodes 和
短程 loss-space；不等同于 LIBERO success rate，也不授予正式训练或评测资格。

## 7. 冻结运行结果

执行身份：

- source commit：`d230798ec66bd05fb5320bf8862b367fb0dfecbb`
- runner SHA256：
  `2e44ed88267bdc8b96deb79533a35f3327ea2e731ac1556c5716caebb4fca73c`
- immutable root：
  `/DATA/share/sana_wam_libero_nonformal_screens/t14/d230798ec66b/libero-t14-initseed-replication-accum8-balanced-joint-fixed20-a9d1dd749b4dd62b454324872401cdcb`
- RESULT SHA256：
  `9c0e6b5f83e989c76570363ea182a0d547c61ebabc997f760139906bee7a930d`
- execution verdict：`T14_INITSEED_REPLICATION_ACCUM8_JOINT_ARM_VALID`
- scientific verdict：
  `T14_INITSEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED`
- physical GPU：0 / `GPU-1ec28cfb-f501-23f3-f865-275a744ca053`

RESULT 是 `310934` bytes 的 canonical JSON+LF；terminal root 为 `0500`，唯一
`RESULT.json` 为 `0400`。全部 8 个 update 与 8 个 heldout sample 的未舍入 ratio
均严格小于 1：

| suite | update ratios | heldout ratios | T14 q | T13 q（诊断） |
|---|---:|---:|---:|---:|
| Spatial | `0.060421 / 0.064404` | `0.058037 / 0.058695` | `0.060389` | `0.103599` |
| Object | `0.045245 / 0.047849` | `0.056504 / 0.045259` | `0.048714` | `0.124816` |
| Goal | `0.055089 / 0.062708` | `0.063551 / 0.054517` | `0.058966` | `0.131623` |
| LIBERO-10 | `0.055452 / 0.055380` | `0.055519 / 0.054554` | `0.055226` | `0.079014` |

update ratio median 为 `0.055415958169840594`，heldout ratio median 为
`0.056011665064723035`。四个 T14 `q` 都严格低于对应 T13 `q`；该结果是有利的
初始化敏感性诊断，但按冻结契约排除在 scientific classifier 外，不能用来改变或扩大
16/16 严格改善 verdict 的含义。

运行精确执行 `16 prepare / 192 forward / 160 backward / 20 optimizer step`，其中
measurement/training forwards 分别为 `32 / 160`；20/20 macro 均通过八次
micro-backward 间 model parameter、FP32 master 与 optimizer state 的 no-mutation
证据。20 个 accumulation-8 macro update 加 32 次 measurement 耗时
`251.69797796569765 s`。update 阶段 CUDA peak allocated/reserved 分别为
`44551804416 / 49673142272` bytes。

本次只加载冻结 SANA base construction checkpoint；没有加载或保存 SANA-WAM training
checkpoint，没有执行 simulator、rollout、benchmark evaluation 或 formal training。
唯一运行 warning 是预期的 SANA partial load（`280 missing / 0 unexpected`），没有
secondary diagnostic warning。T14 因而在第二个 initialization seed 上复现了 T13
的多 episode accumulation-8 短程保持结论，为该定性结论不局限于 seed `20260806`
提供了直接证据；覆盖面仍只有两个 seed，且不构成 rollout success、正式训练或
benchmark admission。
