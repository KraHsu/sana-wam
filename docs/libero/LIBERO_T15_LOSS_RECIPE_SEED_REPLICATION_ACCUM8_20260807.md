# LIBERO T15：Global Fixed-Recipe-Seed Replication

状态：**source implementation；尚未创建运行 root，尚未执行 GPU screen。**

T15 是 T14 的单轴复现：只把所有 192 次 forward 共用的 fixed loss-recipe seed 从
`20260826` 改为 `20260827`。initialization seed、dataloader seed、config、16 个样本、
accumulation-8 拓扑、优化器、预算和严格分类器全部不变。若 T15 仍满足 16/16 ratio
严格 `<1`，本路线将停止继续堆叠同类 loss-only seed screen，下一核心问题转为最小
闭环行为 smoke。

本实验是单 GPU、real-data、non-formal architecture screen，不是正式训练、LIBERO
benchmark、simulator rollout、admission、部署或完整 2B 运行。允许加载冻结 SANA
base construction checkpoint；禁止加载或保存 SANA-WAM training checkpoint。

## 1. 直接前序

直接前序固定为 T14：

- source commit：`d230798ec66bd05fb5320bf8862b367fb0dfecbb`
- runner SHA256：
  `2e44ed88267bdc8b96deb79533a35f3327ea2e731ac1556c5716caebb4fca73c`
- RESULT SHA256：
  `9c0e6b5f83e989c76570363ea182a0d547c61ebabc997f760139906bee7a930d`
- immutable root：
  `/DATA/share/sana_wam_libero_nonformal_screens/t14/d230798ec66b/libero-t14-initseed-replication-accum8-balanced-joint-fixed20-a9d1dd749b4dd62b454324872401cdcb`
- execution verdict：`T14_INITSEED_REPLICATION_ACCUM8_JOINT_ARM_VALID`
- scientific verdict：
  `T14_INITSEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED`

T15 必须在任何 CUDA/model 构造前只读重哈希 T14 runner、RESULT 与外部资产，并验证
root/RESULT 权限、唯一 terminal 文件、source/config/runner identity、seed、recipe
signature、selection、16 ratios、四个 `q`、预算和禁止项。T13 只保留为 underlying
lineage，不能取代 T14 的直接前序地位。

## 2. 唯一实验轴与耦合边界

固定值：

- config：`configs/experiments/libero_t14_initseed_replication_accum8.yaml`
- config SHA256：
  `36daaada09409c22ef2e57e3382ca2ef28c6847f9d2e043cad354464c3dde929`
- initialization seed：`20260807`
- dataloader seed：`20260806`
- T14 global fixed-recipe seed：`20260826`
- T15 global fixed-recipe seed：`20260827`

现有 harness 在每次 forward 前重置 Python、NumPy、Torch CPU 与 Torch CUDA RNG，并
在 forward 后恢复调用方 RNG。一个 recipe seed 同时控制 32 次 measurement forward
与 160 次 training forward。因此 T15 回答的是“更换全局 fixed recipe 后，定性
16/16 结论是否复现”，不是隔离的纯 training-recipe 因果实验。不得在运行时静默拆成
measurement/training 两个 seed；T15 与 T14 的 `q` 差异只能作诊断，不能归因于训练
recipe 本身。

T15 必须证明全部 192 个 captured recipe signature 唯一且相同，并且该 signature 与
T14 冻结 signature
`ef6c3262a3de4f2a21ac908d5140e8d1bc629c2398a7be97630b264d81dab3ad`
不同；否则 fail-closed。

## 3. 原样继承的数据与选择

不重新抽样，原样继承 T13/T14 canonical selection：

- manifest bytes：`16763`
- manifest SHA256：
  `fa00b477558eb26ec5657b87e11e4d6023181f88ea023889e2e86d94b3fd5305`
- eligible manifest：`53728` bytes，SHA256
  `50d97577a4e0120ef9e6b42e1a227321b05ed19450626dc3dfa0c0810c997c47`

| suite | update labels / episodes / lengths | heldout labels / episodes / lengths |
|---|---|---|
| Spatial | A12 `191/96`, A13 `144/92` | H12 `130/96`, H13 `397/101` |
| Object | A14 `19/161`, A15 `170/160` | H14 `394/148`, H15 `34/224` |
| Goal | A16 `75/116`, A17 `38/93` | H16 `154/113`, H17 `195/104` |
| LIBERO-10 | A18 `70/416`, A19 `10/383` | H18 `23/455`, H19 `306/505` |

48 个 role asset、四个 context group、payload、顺序、task/start-frame identity 均须
与 T14 完全一致；任一漂移 fail-closed。

## 4. 不变的拓扑与预算

- `JOINT` only，`batch_size=1`，`gradient_accumulation_steps=8`。
- 20 个 macro；每个 macro 按 A12–A19 做八次 singleton forward/backward。
- 每个 singleton coefficient `0.125`；每 macro 八项和为 1。
- 八次 micro-backward 间 model parameter、FP32 master 与 optimizer state 不变；
  gradient 按 accumulation 语义累积。
- 每 macro 只允许一次 clip、FP32-master AdamW step 与 BF16 projection。
- 每个 update sample exposure `20`、累计 coefficient `2.5`；heldout exposure 和
  coefficient 都为 0。
- 每 label 单独 `architecture.prepare_inputs([raw_sample])`，不拼接 prepared tensor。

精确预算为 `16 prepare / 32 measurement forward / 160 training forward / 160
backward / 20 optimizer step`，architecture forward 总计 `192`，no-mutation evidence
必须为 `20/20`。

## 5. 严格分类与诊断

有效执行 verdict：

```text
T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_ARM_VALID
```

有效运行按未舍入 ratio 顺序分类：

1. 8 update + 8 heldout 全部 `<1`：
   `T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED`
2. 否则任一 update `>=1`：
   `T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_TRAINING_FIT_FAILURE`
3. 否则：`T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_HELDOUT_GAP`

边界 `1.0` 不算改善；NaN、inf、零或负 loss/ratio fail-closed。每 suite 的 T15 `q`
仍是两 update 与两 heldout ratio 的算术均值。冻结 T14 `q` 为 Spatial
`0.06038898107589587`、Object `0.04871424574422691`、Goal
`0.058966199161112065`、LIBERO-10 `0.05522630218227865`；`q-vs-T14` 明确排除在
classifier 外。

## 6. Source、root 与停止规则

只新增三个文件，不新增 config 或 architecture source：

- `docs/libero/LIBERO_T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_20260807.md`
- `scripts/smoke_libero_ar_t15_lossrecipe_replication_accum8_gpu.py`
- `tests/test_smoke_libero_ar_t15_lossrecipe_replication_accum8_gpu.py`

唯一 namespace：

```text
/DATA/share/sana_wam_libero_nonformal_screens/t15/{source_commit[:12]}/libero-t15-loss-recipe-seed-replication-accum8-balanced-joint-fixed20-{nonce}
```

root 必须 fresh、排他、one-shot；terminal root `0500`，唯一 RESULT/FAILED `0400`。
禁止复用、覆盖、自动重跑与 post-freeze mutation。禁止正式训练、benchmark、
simulator/rollout、真实机器人、SANA-WAM training checkpoint load/save、admission、
deploy、AV2 或 Global Stage 3。

若 T15 按首分支 replicated，则 initialization 与 global recipe 两个主要随机轴都已各有
第二点复现；停止追加同 cohort 的 loss-only seed screen，下一步只设计最小闭环行为
smoke，并在取得新增 simulator/checkpoint 权限前不执行。若 T15 不 replicated，则冻结
失败科学 verdict，先分析 recipe 敏感性，不自动重跑。

## 7. 待填写冻结结果

- source commit：待 source freeze
- runner SHA256：待格式化与 source freeze
- immutable root：尚未创建
- RESULT/FAILED SHA256：尚无
- execution/scientific verdict：尚无
