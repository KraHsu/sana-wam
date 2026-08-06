# LIBERO T15：Global Fixed-Recipe-Seed Replication

状态：**GPU screen 已完成并冻结；有效执行，16/16 ratio 严格 `<1`，
global fixed-recipe-seed 复现通过。**

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

## 7. 冻结身份与结果

- source commit：`d669a2e0bc201941fd911f484df36d7afec44cd6`
- runner SHA256：
  `25ebbe148287a7ee9795f56e9253b1db238d0cf5dbea52778c7b61932ee9023a`
- nonce：`9732531de18094840229d7ca0598a648`
- GPU：physical GPU 0，
  `GPU-1ec28cfb-f501-23f3-f865-275a744ca053`
- immutable root：
  `/DATA/share/sana_wam_libero_nonformal_screens/t15/d669a2e0bc20/libero-t15-loss-recipe-seed-replication-accum8-balanced-joint-fixed20-9732531de18094840229d7ca0598a648`
- RESULT：`313690` bytes，canonical UTF-8 sorted compact JSON + final LF
- RESULT SHA256：
  `9e4ee9cb515e5cfafc5598eff33041232ad702b32e641b26435bcb1c643820b7`
- execution verdict：
  `T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_ARM_VALID`
- scientific verdict：
  `T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED`

root 已冻结为 `0500`，唯一 terminal 文件 `RESULT.json` 已冻结为 `0400`；
进程退出码为 0，退出后指定 GPU 上无 compute process。

## 8. 未舍入数值结果

| suite | update ratios | heldout ratios | T15 `q` | `q - T14 q` |
|---|---|---|---:|---:|
| Spatial | A12 `0.05855667346799558`, A13 `0.06311710241127066` | H12 `0.07591331024723302`, H13 `0.08491452087955839` | `0.07062540175151441` | `+0.010236420675618543` |
| Object | A14 `0.04384963566930872`, A15 `0.045794253548511986` | H14 `0.03905547885554516`, H15 `0.12062984005446141` | `0.06233230203195682` | `+0.013618056287729906` |
| Goal | A16 `0.057714160550297905`, A17 `0.06551929631611449` | H16 `0.0783754119283018`, H17 `0.07015831533762534` | `0.06794179603308488` | `+0.008975596871972816` |
| LIBERO-10 | A18 `0.03814969204189277`, A19 `0.03538701931828637` | H18 `0.051707981996862315`, H19 `0.04530126764471062` | `0.04263649025043802` | `-0.012589811931840632` |

- update median ratio：`0.05175420704940495`
- heldout median ratio：`0.07303581279242918`
- 最差 ratio：H15 `0.12062984005446141`，仍距失败边界 `1.0` 很远。
- T15 全局 recipe signature：
  `1ed9a3c26a9035f9b58a38df471dc09fb62a5b9110d08c99488a1eb7e96b69c5`；
  192/192 forward 唯一且一致，并与 T14 signature 不同。
- T14 对比只是次要诊断：四个 suite 中仅 LIBERO-10 的 `q` 低于 T14；
  这不影响预先冻结的 16/16 classifier。

## 9. 执行与不变性证据

- 精确计数：`16 prepare / 32 measurement forward / 160 training forward /
  160 backward / 20 optimizer step`。
- A12–A19 各进入 20 次 update forward，H12–H19 的 backward/update exposure
  均为 0。
- 20/20 macro 均验证八次 micro-backward 之间 model parameters、FP32 master
  与 optimizer state 不变；20 次 AdamW state 均 finite，每次 BF16 projection
  与 FP32 master 严格同步。
- measurement/probe state 不变，prepared inputs 不变且无 alias；heldout 样本
  未进入 backward 或 update。
- 终态独立审计 `blocker/major/minor = 0/0/0`；102/102 个 external
  manifest 文件共 `14029037784` bytes 已逐字节 live rehash，size/SHA 差异为 0。
  所有 root 祖先 lstat 均无 symlink，source/config/Sana/T14 direct-predecessor pins
  与 T13/T14 selection bytes 均一致。
- update 核心加 32 次 measurement 用时 `252.71128380298615 s`；更新峰值
  allocated/reserved 分别为 `44551804416` / `49673142272` bytes。
- 唯一捕获 warning 为预期的 SANA base partial load：`280 missing / 0 unexpected`。
- 未执行 benchmark、simulator、formal training；未加载或保存 SANA-WAM
  training checkpoint。

## 10. 科学解释与下一门槛

T15 在保持 T14 initialization seed、dataloader seed、数据、任务、样本、优化器和
20×accum8 拓扑不变时，更换 global fixed-recipe seed 后再次得到 16/16
严格改善。结合 T14 的 initialization-seed 复现，当前证据支持：这个
same-cohort loss-fit 现象对已测的两个主要随机轴定性稳健。

该结论仍不是纯 training-recipe 因果效应：同一 recipe seed 同时控制
measurement 和 training forward，而且 `q` 的定量排序会随 recipe 变化。
它也不证明闭环控制、rollout success 或 benchmark 能力。

因为预先冻结的停止条件已满足，不再追加同 cohort loss-only seed screen。
下一核心实验应跨越行为证据门槛：先设计最小闭环 LIBERO behavior smoke，
在获得 simulator 和必要的 in-memory trained-state/checkpoint 能力后再执行。
