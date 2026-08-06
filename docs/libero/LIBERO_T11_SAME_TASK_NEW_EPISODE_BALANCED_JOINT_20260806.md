# LIBERO T11：同任务、新 episode 的 balanced-JOINT 复现

状态：**已执行并冻结；执行 verdict 为
`T11_NEW_EPISODE_JOINT_ARM_VALID`，科学 verdict 为
`T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT_REPLICATED`。**

T11 只回答一个核心问题：T10 的 balanced-JOINT 在同一四个任务、但完全不同的
update/probe episodes 上，是否仍能令四个 update loss 与四个 held-out loss 同时下降。
它不是正式训练、benchmark 评测、rollout 或部署 admission，也不产生或加载
SANA-WAM training checkpoint。

## 1. 直接前序证据

- T10 arm source commit：`6861e5a13fa8110986f1b46f3062de9f0b0e3954`
- T10 arm runner SHA256：
  `d7e3e7da3cd0a13e5217f0a044b2f36a4a703be7227149a897268f82b74857dd`
- T10 SEQ RESULT SHA256：
  `d92fe05fe523c346e90ab6a392ddad9c3ec41764d5581211e6223895a42e8937`
- T10 JOINT RESULT SHA256：
  `423ce3e01bea7368786b1c470a790af666baf7504a2235894a59b82efef3ea9b`
- T10 aggregate source commit：
  `128e1be8cd48f8cef1b7c5f24d1bdecfbe45054a`
- T10 aggregate runner SHA256：
  `b45e9536b35790c06e599ba919ed1370c64e4e1dff477019fab9f0ccc40b61c5`
- T10 aggregate RESULT SHA256：
  `8fcd26ea3a59fe6e01cf8f279301c899ed5dab541ada9c57e2ce85888abd147b`
- T10 科学结论：
  `T10_BALANCED_JOINT_SIMULTANEOUS_RETENTION_SUPPORTS_OVERWRITE`

Runner 必须逐文件核验上述两个 arm result、arm runner、aggregate result 与 aggregate
runner，并核验它们的 commit、runner、config、root、scope、verdict 和互相引用。

## 2. 冻结样本与机械选择

固定 suite 顺序为 Spatial、Object、Goal、LIBERO-10；固定任务索引分别为
`0, 3, 2, 3`。每个 suite 的第一个机械候选作为 update 样本 `A4..A7`，第二个作为
只测量、不反传的同任务 probe `H4..H7`。

精确 payload 是 ASCII，且最后一行也以 LF 结束：

```text
SANA-WAM/LIBERO/T11_NEW_SAMPLE_SELECTION_V1
T10_AGGREGATE_RESULT_SHA256=8fcd26ea3a59fe6e01cf8f279301c899ed5dab541ada9c57e2ce85888abd147b
DATASET={dataset}
TASK_INDEX={task_index}
EPISODE_INDEX={episode_index}
START_FRAME=0
```

选择规则原文：

```text
in frozen suite order spatial,object,goal,10; within each frozen T10 task exclude every T1-T10 measured-or-backpropagated episode and excluded Goal episode 82; rank remaining episodes by exact T11_NEW_SAMPLE_SELECTION_V1 payload SHA256; assign first to A4-A7 and second to H4-H7
```

Eligible population 共 `158` 个 episode，分 suite 为 `41/44/34/39`。排除集包含
T10 的 10 个原始 exclusions、T10 的 H0-H3，以及 Goal task 5 episode 82，共 15 行。

- eligible canonical JSON：`8676` bytes，SHA256
  `25486093e047d38d95151b36277874b89851f934720e3418e1f817d0b736745b`
- selection canonical JSON：`6198` bytes，SHA256
  `29e9ce96b8d4d6b0d5e41c6cc2d0fb887c4061990d4c2669a498d806dbd82dcd`
- canonical JSON 定义：UTF-8、`ensure_ascii=True`、`allow_nan=False`、键排序、
  compact separators，且 manifest bytes 本身不附加 LF。

| label | suite | episode / length | payload SHA256 |
|---|---|---:|---|
| A4 | Spatial | 76 / 133 | `08a163e72aa9a987e873698475f81d2b456379e509d8efc4f445ad3067aeda5d` |
| H4 | Spatial | 265 / 130 | `08d54f12cf43862296897de4000ce1d75b00474417e6fecfd04c5ec805e2516e` |
| A5 | Object | 64 / 129 | `0731ad908d69f809d0a20f8bfbe4d6c6a4f26042764dd41339990dd3cbf732f4` |
| H5 | Object | 337 / 134 | `0e837eb33e758153459105ce9adf417ef1505c808e445aa335d926592719a081` |
| A6 | Goal | 30 / 185 | `0ef8699b3f006e9460a703e92d55ad6dd611e5f4b4a30b52ee4af18315b27765` |
| H6 | Goal | 180 / 182 | `11b1eca88d1a1a10c1903c061f5a6035bbf2bd45aa335975c83ce837641f508b` |
| A7 | LIBERO-10 | 303 / 244 | `041f7d725846e2191e0c779bbda82323a319bc0a61530f3f0e268bca1390776d` |
| H7 | LIBERO-10 | 272 / 272 | `070192b8c136cf7d5c12d57f02ae10625839b2a9b88b871d79281025d70688ad` |

每个样本固定三个 asset SHA：

| label | parquet | image | wrist image |
|---|---|---|---|
| A4 | `cfd720e6e95c6a8a0bde52fcf69db35bde57aab401639fe4429ca2252769013b` | `ea8625462f38127e25e3a1ae055d2431ef12cde8bb158aa13aad0b1838dc4938` | `9428026582bf95cd48845c477be6280a55ef6b98c46bfea5de36fbcfbe88053d` |
| H4 | `d8f43909a98ccd8e13bbc50215ebcd607e0ee7f39895390edb4923086327a9e3` | `c1e35e4387dc0b16ad77009bc34f8e232db896e9f09eb1c0237fdf0d9060d24f` | `55108bb251273886af1f1770e3b6ac5bf2601ad56eabfde36bafdbdbd42762f4` |
| A5 | `d3d04574cf1906bd0f7902ce787821abb43d96570076511b6c112911ab2cf4de` | `21034dfa8b78a939e1f4e8e9f34ad5f3751b4d59c1520b74d170aca2159d7807` | `9e46aef2a66642b197dc6e359eb5fdc12ea8b2c1318a9bc7de3d750c8b5eb781` |
| H5 | `2de8a3be69381e3ffd74ebcc377a991298441bfa9d0ad67d45dc5199addb00af` | `2511a0e8c59d63db44217e0c1a79dc07c635a936a6ab03d545bef9c470ddd7af` | `3b2d3b3064ce400d94a01ae35b5004ec2c64b1c349e67347809fa9959849b3b0` |
| A6 | `7f01adce8356e1838a64053849f08bb53acc504835f73266643585bfbae36de4` | `326f095656208f578772858753ef9272408b1d346e0669bb5adb14c3d706d04a` | `e3b300a329ad6257d02f800f1911ea68c11174ff5f6e36d663ed11ad08fbcc87` |
| H6 | `46e510231b659177dcea98031e21bd5dda7ef8a71279bdb54a5b8f0e179ff9ea` | `d6bb499f59b5a3b185adffec4f0998656f4ff2f52d147930ab7fd5c74ad66095` | `3420e2b3b7bede74ba3f45773d973974ba7ac3d1ded4d6f829361db1360e2803` |
| A7 | `ecec19496a6e481ae861022d30cba46c834d1cf52925c1e152af78acc3a736af` | `b8efac07dd2068e14491d372adbd5f9131e4860bf3f5f0c98bfb1c15effec387` | `002e37672e44b889bdbdaace057b98330d4748462bd801df1b497455b6e0295e` |
| H7 | `ee3ce6f2a6a7388dd6266c4aec888dac5230645ce102d3d8609b8e6a85d45985` | `41f685382ce65d8f145da5700fee2141ecf0783aacfc66a716a166d094ffa775` | `79e8972ff43d00618780369a893e34b7160dc2c396cd43701ce7bdc987034818` |

## 3. 冻结执行拓扑

- 单 arm，固定为 `JOINT`；没有 SEQ arm 或 aggregate runner。
- fresh initialization seed：`20260806`；loss recipe seed：`20260826`。
- 20 个 macro steps；每个 macro 严格按 `A4,A5,A6,A7` 顺序做四次
  forward/backward，各 loss 权重精确为 `0.25`。
- 四次 microback 之间不得修改 model parameter、FP32 master parameter 或 optimizer
  state；每个 macro 只允许一次 gradient clip、一次 FP32-master AdamW step、一次 BF16
  projection。
- 每个 A 样本 raw exposure 为 20、累计有效 loss coefficient 为 5；所有 H 样本
  exposure/coefficient 均为 0。
- 总计：8 prepare、96 architecture forwards（16 measurement + 80 training）、80
  backward calls、20 optimizer steps。
- 使用 persistent FP32 optimizer masters；固定 action-only loss、learning rate、betas、
  weight decay、gradient clip、trainable allowlist 与冻结 video backbone。
- pre/post measurement 必须保持 prepared tensors、model/optimizer state 与调用方 RNG
  不变；所有 96 次 forward 使用同一冻结 recipe signature。

## 4. 两类 verdict 同时写入同一 RESULT

执行有效性 verdict 固定为：

```text
T11_NEW_EPISODE_JOINT_ARM_VALID
```

在有效运行内，用**未舍入**的四个 `rA=post_A/pre_A` 与四个
`rH=post_H/pre_H` 严格按下列顺序分类：

1. 若 8/8 `rA,rH < 1`：
   `T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT_REPLICATED`
2. 否则，若任一 `rA >= 1`：
   `T11_SAME_TASK_NEW_EPISODE_JOINT_TRAINING_FIT_FAILURE`
3. 否则：
   `T11_SAME_TASK_NEW_EPISODE_JOINT_HELDOUT_GAP`

边界值 `1.0` 不算改善。NaN、inf、零或负 loss/ratio 必须 fail-closed。无需也禁止另建
aggregate；execution verdict 与 scientific verdict 必须由同一个冻结 RESULT 同时记录。

RESULT 另记录每个 suite 的 `q=(rA+rH)/2`、冻结的 T10 JOINT `q`、两者差值及
`q_T11 < q_T10` 布尔量；该比较明确排除在上述科学分类器之外，只作为跨 episode
难度/优化幅度诊断，不改变任何 verdict。

## 5. Root 与边界

- namespace：`/DATA/share/sana_wam_libero_nonformal_screens/t11`
- slug：`libero-t11-same-task-new-episode-balanced-joint-fixed20`
- root 必须绑定 source commit 前 12 位与 32 位小写十六进制 nonce；只创建一次，终态
  `RESULT.json` 或 `FAILED.json` 后冻结，禁止复用、覆盖和自动重跑。
- 允许运行时加载冻结 SANA base construction checkpoint；禁止加载或保存任何
  SANA-WAM training checkpoint。
- 禁止 simulator、rollout、benchmark evaluation、正式训练、admission、完整 2B、部署。
- 本 screen 仅有一个 seed、每 suite 一对新 episodes，不能据此声称 benchmark success
  或正式训练 readiness。

## 6. 冻结执行结果

本轮仅执行一次，没有重跑或扩展预算。冻结身份与证据为：

- source commit：`7d53d618234e3e37367c5b1e39e169742c2cf514`
- runner SHA256：
  `ab66fcf6052582631424c97f149c9187d942b1da253e201fc698142e5ad41b54`
- immutable root：
  `/DATA/share/sana_wam_libero_nonformal_screens/t11/7d53d618234e/libero-t11-same-task-new-episode-balanced-joint-fixed20-05e0d719fbc5c25e66ddf435cb47eba2`
- nonce：`05e0d719fbc5c25e66ddf435cb47eba2`
- RESULT SHA256：
  `6da12f2e3622d4e6427070bfd10b3dab9550c338a834b7662309cefea98d7417`
- 冻结权限：root `0500`，`RESULT.json` `0400`；文件是唯一 canonical
  terminal result。
- 物理 GPU 0：`GPU-1ec28cfb-f501-23f3-f865-275a744ca053`。

四个 suite 的未舍入 ratio 与诊断性 `q=(rA+rH)/2` 如下。八个 ratio
全部严格小于 1，因此按冻结分类器唯一进入 `REPLICATED` 分支。

| suite | `rA` | `rH` | `q` | T10 JOINT `q` | `q - q_T10` |
|---|---:|---:|---:|---:|---:|
| Spatial | `0.20113512209310244` | `0.20696124596611650` | `0.20404818402960948` | `0.27320731955314786` | `-0.06915913552353839` |
| Object | `0.19894518695843905` | `0.20661460291178890` | `0.20277989493511397` | `0.26340073735799435` | `-0.06062084242288038` |
| Goal | `0.25910625185730085` | `0.22687573827231447` | `0.24299099506480765` | `0.32323180400807430` | `-0.08024080894326663` |
| LIBERO-10 | `0.13246952172421890` | `0.13349105932495006` | `0.13298029052458450` | `0.18725771119867107` | `-0.05427742067408659` |

update 与 held-out ratio 的中位数分别为 `0.20004015452577073` 和
`0.20678792443895272`。四个 T11 `q` 也都低于对应的冻结 T10 JOINT
`q`；这只是跨 episode 难度/优化幅度诊断，不参与科学 verdict。

执行账本与资源观测与冻结预算一致：8 次 `prepare_inputs`、96 次
architecture forward（其中 80 次 training、16 次 measurement）、80 次
backward、20 次 macro optimizer step；每个 A 的 raw exposure 是 20、累计
loss coefficient 是 5，H 从未参与 backward/update。20/20 macro 的
no-intra-macro mutation 证据全部通过。模型与数据集构造耗时
`40.55068732984364 s`，八个样本 prepare 合计约 `3.27397 s`，二十个 macro
update 加十六次 measurement 耗时 `125.35481818392873 s`。update 阶段
peak allocated/reserved 分别为 `32,521,826,816` / `37,557,895,168` bytes，
约 `30.288311` / `34.978516 GiB`。

结论是：T10 的 balanced simultaneous multi-suite objective 并非只对原固定八个
episode 有效；在相同四个 task、全新 update/probe episodes 上，它仍然令
4/4 update 与 4/4 same-task held-out loss 同时改善。这强化了 balanced-JOINT
作为 successor training topology 的 loss-space 证据，并降低了 T10 结果只是
episode-specific 巧合的解释力。但它仍是 single-seed、每 suite 只有一对新
episodes 的 short-horizon non-formal screen；样本属于训练 metadata/
normalization population，且仅测 action loss。本轮没有运行 simulator、rollout、
benchmark evaluation 或 formal training，也没有加载/保存 SANA-WAM training
checkpoint，因而不支持 benchmark success、closed-loop capability 或正式训练
readiness 声称。
