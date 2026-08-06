# CACH-SANA-WAM 架构设计、实验结果与阶段性分析

状态：`DEVELOPMENT_HISTORY_SNAPSHOT / NON_FORMAL / NOT_AN_EXECUTION_AUTHORITY`

整理日期：2026-08-07

规范主机：`H200`  
规范工作树：`/home/zch/workspace/sana-wam`  
主仓 T13 source commit：`68fa88157973f59383a87be1cb3107f5824e64cf`
Sana gitlink：`16b9cec673e3335724ba2d8db25de7f9ed229292`

本文是一份可独立阅读的历史快照，汇总截至当前已经设计的架构、基础实现、所有关键
architecture-validation 实验、无效运行及其解释。事实来自 H200 上的 frozen decision、
run card、source pin、`RESULT.json`、`SCREEN_RESULT.json`、token claim 和 freeze receipt，
不以对话记忆作为证据。

本文不是 run card、训练授权、formal admission、科学结论或部署许可；它不改变任何
predecessor、冻结 root、review token 或 gate 状态。

## 1. 当前结论

截至 2026-08-07，已实际执行的 action-conditioned 架构主线为：

```text
REF-GDN-CORRECTED
  └─ CACH-A
       └─ CACH-A2 zero-anchor seam
            └─ CACH-A3 normalized-common exact-odd output residual
                 └─ CACH-A4 state-conditioned causal exact-odd delta stream
                      ├─ AV-2 tiny real-data overfit: AV2_GO
                      └─ AV-3 paired reduced-scale: REDUCED_ARCH_STOP
```

核心判断如下：

1. action seam 的可训练性已经反复成立。AV-1、AV-1B、A2、A3、A4、AV-2 和 AV-3
   都观察到有限且非零的 action gradient、parameter update 或 JVP；AV-3-R3 也完成了
   每臂精确 1,000 个 AdamW update。因此最终负结果不是“action 分支未接上”或“没有
   优化”。
2. A2 证明 zero-anchor、bias-free、typed direct-bypass 可以消除旧 CACH-A 的零动作
   污染，但 action delta 很准时，common-mode error 仍主导总误差。
3. A3 用 normalized common path、exact-odd output residual 和正交 common/delta loss
   去掉了该混淆，在 synthetic screen 得到 `OPERATOR_GO_COMMON_STABLE`。
4. A4 再加入 state-conditioned causal delta recurrence，在 synthetic screen 得到
   `OPERATOR_GO_CAUSAL_STREAM_COMMON_STABLE`，并在 8 train / 8 heldout tiny real-data
   overfit 上得到 `AV2_GO`。
5. 更严格的 AV-3 使用新的 16 train / 16 heldout cohort、horizon 1/2/4 和显式
   motion/delta-sensitive primary。唯一有效运行 AV3-R3 得到
   `REDUCED_ARCH_STOP / AV3_STRONG_STOP_LINE`：candidate 的 absolute video fit 很好，
   但 anchor-relative dynamics 更差，correct action 相对 shuffle 的改善不足 5%，
   且 correct action 甚至劣于 no-action。
6. 因此，当前 **CACH-A4 reduced architecture path 已停止，AV-4 未解锁**。这是一条
   有效的 reduced architecture 负证据，不是 harness 失败。
7. 原始完整 CACH-SANA-WAM v0 package 尚未被科学验证或证伪：periodic anchors、
   AttnRes、generated-prefix/self-forcing、完整 2B、public production path、closed-loop
   和 multi-seed confirmation 都没有执行。
8. 项目没有进入正式训练、正式评测、formal admission、Global Stage 3、checkpoint
   训练或部署；不得宣称已有 504-step formal 结果或正式模型完成。
9. 在独立的 LIBERO production-shaped `DualSystemARArchitecture` 验证线上，T1 至 T13
   已依次闭合 one-update、fixed-sample 20-update learnability、held-out recipe、同任务
   episode、跨 Spatial task 以及跨 Object/Goal/LIBERO-10 suite 的 update-free action-loss
   transfer、四套件循环微学习与相位旋转诊断。T7 的 typed numerical verdict 为
   `T7_FOUR_SUITE_CYCLIC_GO`（受 post-run audit 限定）；T8 的有效终态为
   `T8_PHASE_ROTATED_MIXED_INCONCLUSIVE`。T9 补齐另外两个循环相位后得到冻结组合结论
   `T9_COMMON_POSITION_EFFECT_SUPPORTED`：4/4 suite 的最旧端点差于最新端点，其中
   Object、Goal、LIBERO-10 的位置 Spearman `rho>=0.8`。
10. LIBERO T7/T8/T9 不改变 CACH-A4 reduced-path stop，也不是 benchmark success：其证据仅为
    一个初始化、一个 recipe、每套件一个更新样本与一个同任务 fresh episode 的 20-step
    loss-space screen。T7 中 Spatial A/H 同时回退；T8 只旋转循环相位后 Spatial 被救回，
    但最旧的 Object A/H 转为回退。T9 表明跨四个冻结循环相位后存在共同 retention/
    position effect，但 Spatial 与 LIBERO-10 的中间位置仍非单调。probes 仍属于四套件
    数据和 normalization population；rollout、suite distribution generalization、正式
    训练和正式评测仍未执行。
11. T10 用 exact-balanced simultaneous JOINT objective 消除了 matched sequential overwrite；
    T11 在同任务全新 episodes 上复现，T12 又在四个此前从未 model-facing 的新任务上得到
    8/8 update/heldout ratio 严格小于 1。T13 保持 T12 任务，将每任务扩为两个 update
    与两个 heldout episodes，并将八次 singleton micro-backward 对齐 production
    accumulation-8 boundary 后，16/16 ratio 仍严格小于 1；update/heldout median 分别为
    `0.116163 / 0.109951`。这把证据扩展到多 episode accumulation，但仍不等于 rollout、
    benchmark 或正式训练结论。

## 2. 证据等级与命名

本文严格区分模型架构、实验阶段和 launcher 修订：

- `CACH-A/A2/A3/A4` 是架构假设。
- `AV-0/1/1B/2/3/4` 是验证阶段，不是新的模型架构。
- `R1/R2/R3` 如果只修 host、environment、mask、JSON serialization、cohort loader、
  CUDA allocator initialization 等问题，就是 harness/launcher revision，不是架构变化。
- invalid/blocked run 不是架构负证据；只有 validity 全通过的 actual-operator result
  才能形成 operator 或 reduced-architecture verdict。

算子证据也分层：

| 等级 | 含义 | 能支持的结论 |
|---|---|---|
| `TRANSITION_PROXY` | pure-Torch 近似递推 | 只支持 proxy learnability；不能接受或否决实际架构 |
| `REFERENCE_OPERATOR` | 与 production 数学递推/state 语义一致的小形状算子 | qualification 后可支持 operator/reduced screen |
| `VENDOR_KERNEL` | 实际 CUDA/Triton vendor kernel | 可支持 actual-operator screen；仍不等于 public production parity |

当前 actual-operator runs 均为 `VENDOR_KERNEL / EXPERIMENTAL_PATH`。这不证明 public
wrapper、dispatcher、durable cache owner、完整 2B 或 production runtime closure。

## 3. 架构全景

### 3.1 原始 CACH-SANA-WAM v0 总体蓝图

权威设计：
`docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md`，
SHA256 `969d9668ffed07176844fcbd84062825f9d5f0af0325959b0068c38fe481f0dd`。

`CACH` 表示 `Causal Action-Conditioned Hybrid`。完整 v0 的目标是让 video dynamics
显式读取 action，同时保持训练/部署一致的 chunk-causal visibility、时间对齐和 cache
事务语义。主要设计包括：

- source-pinned、冻结的 LTX2 causal VAE 与 text encoder；video DiT、GDN/softmax
  blocks、action backbone、proprio encoder 和 conditioner 从注册 seed 随机初始化；
- 20-layer hybrid video backbone；`REF-GDN-CORRECTED/CACH-A` 起点为 all-GDN，
  原始 CACH-H/full-v0 roadmap 才在 `[7,15]` 引入 causal softmax anchor；
- GDN state 是 recurrent full-history summary；
- periodic causal softmax anchors 固定在 zero-based layers `[7,15]`，只读取当前 noisy
  chunk 和一个 previous committed chunk，不是 full-history attention；
- Block AttnRes span `S=8`、depth-shared projection，并且只保留一次 forward 内的
  depth-local state；
- 双速率 action conditioning：chunk-rate proprio/history 与 action-token-rate noisy
  action candidate；
- `ChunkActionLayout` 作为 latent/action/Action-RoPE 对齐的唯一 source of truth，
  取代 fixed-ATC；
- bootstrap 固定为 `first_frame_pinned`、`observed_prefix_chunks=0`；
- paired cache commit 必须是单次 `t=0` 的 `(video_x0, action_x0)` 事务；
- action-to-video 公共接口为 `action_condition`，内部只允许 vendor
  `use_delta_pose_additive` seam；
- AFCC lineage 与 capability lineage 隔离；禁止 DAgger、人工 recovery、rerank、
  best-of-N、approximate replay 和 loss subtraction。

完整 v0 是 package 设计，不是当前已经训练完成的模型。实际 architecture-validation
为了快速获得效果信号，先只验证最小 load-bearing delta：action-to-video conditioning。

### 3.2 原始 candidate ladder

| 架构 | 相对 reference 的唯一增量 | 目的 | 当前状态 |
|---|---|---|---|
| `REF-GDN-CORRECTED` | 20-layer all-GDN、新 bootstrap/layout/RoPE/cache/commit；video action-blind | corrected causal reference | reduced/proxy/vendor common trunk 已实现；full-model commissioning 未完成 |
| `CACH-A` | action-to-video conditioning | 验证 action-conditioned dynamics | 已执行 AV-1/AV-1B/review300 |
| `CACH-H` | anchors `[7,15]` | 低剂量 global refresh | 仅设计，未实现效果实验 |
| `CACH-R` | 用 AttnRes router 替换 `CACH-H-SHAM` inert sham，`S=8` | 验证 routing，而非参数量 | 仅设计，未实现效果实验 |
| `CACH-SF` | registered generated-prefix objective/schedule | model-prefix robustness | objective/applied-action contract 未闭合，未实验 |

`CACH-H-SHAM` 只是 `CACH-R` campaign 内的 parameter-matched inert reference，不是
第三个实验成员。`[7,15]` 也不是已证明最优的 placement。

后续 A2/A3/A4 是用户明确授权的 successor architecture 分支，是对原单一 CACH-A
计划的显式偏离；它们不表示 CACH-H/R/SF 已经执行。

### 3.3 已执行 successor 架构对比

| 架构 | action delta 注入位置 | 结构性约束 | 主要要解决的问题 | 有效最高阶段 |
|---|---|---|---|---|
| `CACH-A` | 每个 GDN recurrence 后、FFN 前 | output projection theta0 zero；旧版 conditioner/projection 有 bias | 最小 action-to-video learnability | AV-1B review300：inconclusive |
| `CACH-A2` | 与 CACH-A 相同 | bias-free、typed mask、inactive/full-no-action direct bypass、zero-origin | 修复 zero action 非 identity 和无条件 bias 容量 | A2-R1：delta live，common blocked |
| `CACH-A3` | common video output 后的并行 residual | parameter-free RMS common、exact odd、zero-origin、orthogonal positive losses | 把 action delta 与 common-mode error 解耦 | synthetic operator GO |
| `CACH-A4` | common video output 后的 causal delta stream | state-conditioned、exact odd、action-blind decay、future→prefix zero | 让历史 action/state 影响未来输出 | AV-2 GO；AV-3 strong stop |

## 4. 基础实现阶段：它们不是架构胜负实验

### 4.1 Stage 0/1

Stage 0 冻结 layout、bootstrap、typed cache、paired commit、initialization/checkpoint、
data/scale 和 deny-execution scaffold，没有运行模型或测试。

Stage 1 实现独立 `sana_wam.cach` namespace、strict config/verifier、review-only
launcher、`ChunkActionLayout`、episode-origin wrapper、partial-tail mask、action seam、
typed hybrid cache、paired staging、trainable/checkpoint contract 和 AFCC isolation。

最终 Stage-1 lightweight suite 为 `209 passed, 0 failed`，AST `15/15`；但没有构造
模型、forward/backward/JVP、GPU、真实数据、checkpoint、训练或评测。因此
`gate_s1=not_claimed`。报告 SHA256：
`71c03f132895ee46705cadcf8629dfc2d9de94003a2fda8b35a8fa4bf030ef05`。

### 4.2 Stage 2 mini numerical subset

Stage 2 在 synthetic mini video-GDN/ActionDiT 上检查 prefix compaction、vendor cache
codec、paired staging、zero-init seam gradient、future separation、failure receipt 和
standalone applied-action ACK。

- scoped mini suite：`75 passed, 0 failed`；
- CPU regression：`264 passed, 10 skipped, 1 deselected, 0 failed`；
- 无 optimizer、训练、真实数据、checkpoint 或 formal root；
- `full_gate_s2=blocked_not_claimed`。

Manifest SHA256：
`8333c74d9bc8115937f98d9fc12a60da62cab49dd8895cd403047027315052fb`。

### 4.3 Stage 2B cache/ledger/recovery

Stage 2B 构建的是运行可靠性平台，不是新的 neural architecture：

| 层级 | 实现 | 记录结果 |
|---|---|---|
| L0/L1 | strict receipt store + canonical tensor snapshot codec | focused `63 passed`；full CACH `305 passed, 10 skipped, 1 deselected` |
| L2A | append-only `PREPARED`/terminal ledger + content-addressed state | focused `82 passed`；full CACH `397 passed, 1 deselected` |
| L2B | fresh-only single-process ledgered manager facade | focused `24 passed`；full CACH `421 passed, 1 deselected` |
| L3 | deterministic model-free recovery and state install | focused `31 passed`；L2A/L2B `106 passed`；full CACH `452 passed, 1 deselected` |

这些结果限于 CPU/synthetic/single-process；没有正式 filesystem admission、multi-process
fencing、真实 power-loss/remount、CUDA mapping 或 production recovery authority。

### 4.4 Phase-C production-shaped mini

Phase C 把 typed layout、public SANA `run_chunk` adapter、tiny ActionDiT、
`CACHNumericalCore`、legacy `list[10]` codec、Stage-2B types、committed-action history
和 single-pointer state transition 接成 production-shaped CPU mini path。

- hard suite `11/11`，regression `48/48`；
- C0–C8 均有 interface/proxy evidence；
- backend 仍是 pure-Torch transition proxy，不是 vendor GDN parity；
- 每项 `global_contract_closed=false`；
- 没有 GPU、optimizer、训练、真实数据、checkpoint、formal root 或 retained raw run。

Phase-C manifest SHA256：
`0b88a76c5d2bf73a72a830fb58d9faac091af77e738f341f843664eaced3ffff`；
report SHA256：
`cb4adbe49e03f3889e209de3ea9b4d043af96d726ae4f78ce8e35554252d0fd6`。

全局状态始终保持：

```text
GATE-S0 = NOT_CLAIMED
GATE-S1 = NOT_CLAIMED
GATE-S2 = BLOCKED_NOT_CLAIMED
Global Stage 3 = NOT_AUTHORIZED
```

## 5. CACH-A：最小 action-to-video seam

### 5.1 结构

`CACH-A` 与 `REF-GDN-CORRECTED` 共用 action-blind GDN/FFN common trunk。candidate
通过 end-of-bin reducer 得到 `[B,K,20]` action condition，经
`Linear(20,64) -> SiLU -> Linear(64,20)` conditioner，在每个 20 个 GDN block 的
vendor recurrence 后、FFN 前注入 `Linear(20,64)` residual。block output projection
在 theta0 为 exact zero，reference 结构性不读取 action 或 no-action tensor。

旧 CACH-A 的 conditioner 和 per-block action projection 均有 bias。即使输入 action
为零，训练后也可能产生非零 residual，所以“no-action slot 为零”不等于 same-arm
identity bypass。这成为 A2 的直接设计动机。

### 5.2 AV-0：只冻结实验契约

AV-0 card SHA256：
`c73a943047295338cd9de14e6e53199b463ea66ae92b99d4b0b3393f5ec34030`。

它预注册 CPU/FP32、batch 8、20-layer proxy、5 latent frames、chunks `[0,3,5]`、
correct/shuffle/no-action、每臂 AdamW 200 steps 和 final-step-only 判定。AV-0 本身
没有运行模型、没有 root、没有 RESULT，也没有 metric。

### 5.3 AV-1：transition proxy

条件：CPU FP32、pure-Torch 20-layer transition proxy、同一 deterministic synthetic
counterfactual batch、每臂 fresh-init AdamW 200 steps。

| 指标 | 结果 |
|---|---:|
| reference primary | `0.2567410721` |
| candidate correct primary | `0.0203621183` |
| candidate no-action primary | `0.2553796429` |
| candidate shuffle primary | `0.9350481107` |
| candidate gain vs reference | `0.9206900627` |
| counterfactual delta NMSE | `0.0461293577` |
| shuffle gap / no-action gap | `0.9782234539 / 0.9202672613` |
| loss, reference | `4.6158886 -> 0.2571111` |
| loss, candidate | `4.6158886 -> 0.0211008` |

typed state：`PROXY_GO`。RESULT SHA256：
`b7b23642d5772efb09e3c0ff09cb1aead3d5834b70b325ddc8b1c10d7438303e`。

解释：proxy 上 action mechanism 明显可学，但证据范围明确为
`PROXY_ONLY_ACTUAL_ARCHITECTURE_NOT_ASSESSED`；它只支持进入 AV-1B。

### 5.4 AV-1B：actual vendor single-call，200 steps

AV-1B 每个 video block 直接调用 vendor
`ChunkCausalGDNTriton(use_autograd_kernel=True)`。hidden 64、2 heads、20 layers；每个
block 一次完整 5-frame vendor call，chunk boundaries `[0,3,5]`。forward-direction
state 只在这次调用内部跨 chunk 保持，没有 cross-call persistent state I/O。

条件：物理 GPU 7，UUID
`GPU-41c95a43-ce96-fff3-33e0-739a3931d603`；每臂 200 AdamW steps；synthetic
single-batch；CUDA/Triton JIT；无真实数据和 checkpoint。

| 指标 | 结果 |
|---|---:|
| reference primary | `16.7294778355` |
| candidate correct / shuffle / no-action | `19.2627932344 / 20.1754714598 / 19.5698560658` |
| candidate gain vs reference | `-0.1514282408` |
| delta NMSE | `0.0445624579` |
| shuffle gap / no-action gap | `0.0452370210 / 0.0156906024` |
| loss drop | `0.9994151648` |

全部 validity 通过，action gradient/update/JVP 都为正；但 GO 所需的 gain 和 action
discrimination 未通过，strong-stop 又未成立。typed state：`REVIEW_ONCE`。
RESULT SHA256：
`d33e2e6bdf59cc1d2ffe51b32d48ce22daa4da1a902e3e9bded28cce29adb5de`。

解释：actual operator 能拟合 action delta，却没有把 correct action 与 shuffle/no-action
充分分开，而且 final candidate 比 reference 差。这不是 invalid run，因此触发唯一一次
有界复核。

### 5.5 AV-1B review300：同架构 fresh-init 300-step 复核

review300 没有改变架构、vendor operator、数据 recipe 或阈值；只做 fresh initialization
并把每臂预算从 200 机械延长到 300 steps。card SHA256：
`a552522a951384996aa8f18ce9cb8a5e6e130ac4d2db68515b5ea6303c6722bc`。

| 指标 | 结果 |
|---|---:|
| reference primary | `41.9110371580` |
| candidate correct / shuffle / no-action | `7.4107018424 / 8.3488921697 / 7.5887873885` |
| candidate gain vs reference | `0.8231801849` |
| delta NMSE | `0.0403946245` |
| shuffle gap / no-action gap | `0.1123730320 / 0.0234669305` |
| seam grad / update / trained action JVP | `0.00445378 / 0.00788394 / 0.00335644` |

root 内 RESULT 是按协议先冻结的 `VALID_REVIEW_PROVISIONAL_NO_VERDICT`，SHA256：
`3ca4ac673893923d29f60b999879132a93d515382f94bdaa462d91d3cdd1786d`。
最终 verdict 由一次性 claim 发布：`OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED`。
claim SHA256：
`e7c355d3c4e8bfb942f8baa8a5511fb5964c2c7f2d6c975ef5aec0daa458a91f`。

解释：增加 100 steps 后 candidate-vs-reference gain 转为明显正值，说明旧 run 不是简单
的“完全学不会”；但 correct-vs-shuffle/no-action separation 仍远低于冻结 GO 线。
唯一 review token 已消费，CACH-A 没有解锁 AV-2。后续 A2 是新的 successor 架构，
不是继续复用 review token。

## 6. CACH-A2：zero-anchor-preserving bias-free seam

架构 ID：`CACH-A2-ZERO-ANCHORED-BIAS-FREE-v1`。

### 6.1 结构

```text
phi(a)      = W2 * SiLU(W1 * a)                  # all bias-free
delta_l(a)  = U_l * (phi(a) - phi(0))            # U_l theta0 exact zero
output_t    = where(typed_mask_t,
                    post_vendor_t + delta_l(a_t),
                    post_vendor_t)
```

主要不变量：

- `W1/W2/U_l` 全部 `bias=False`；
- mask 只能来自 typed layout provenance，不能由 action 数值是否为零推断；
- inactive token gather/scatter direct bypass；
- full `no_action` 不读取或执行 conditioner/projection；
- active numeric-zero action 保持 zero-origin，但 presence 语义仍为 active；
- reference 完全不读取 action、anchor 或 mask。

A2 base execution 在 root 前因 SSH alias/actual hostname 混用和 Torch local version
`2.7.1+cu128` 比较错误而停止；没有创建 root，不是架构结果。R1 只修这两个 preflight
问题，架构、数据、预算和阈值不变。

### 6.2 A2-R1 结果

条件：vendor kernel、synthetic single-batch、每臂 300 AdamW steps。

| 指标 | 结果 |
|---|---:|
| reference primary | `41.9110371580` |
| candidate correct primary | `44.8571122926` |
| candidate gain vs reference | `-0.0702935392` |
| shuffle gap / no-action gap | `0.0217464211 / 0.0046553487` |
| half-delta NMSE | `0.0012813518` |
| half-delta alignment cosine | `0.9993663405` |
| half-delta energy ratio | `1.0063412693` |
| paired common prediction MSE | `44.8946470370` |

typed verdict：
`OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED / A2_DELTA_LIVE_COMMON_MODE_BLOCKED`。
RESULT SHA256：
`a7767e47d718deda41762db4e250dbca0602f78fcff1651ef06a0d6da508c2ef`。

解释：A2 把 action half-delta 学得非常准确，也证明 exact zero/direct-bypass 结构成立；
但总误差几乎完全来自 common mode。约 `44.895` 的 common error 淹没约 `0.000320`
的 half-delta error，因此用 total-MSE relative gap 衡量 action learnability 发生严重尺度
混淆。继续单纯增加 steps 或放宽 no-action/shuffle 阈值不能修复该问题。

## 7. CACH-A3：normalized-common exact-odd output residual

架构 ID：`CACH-A3-NORMALIZED-COMMON-EXACT-ODD-OUTPUT-RESIDUAL-v1`。

### 7.1 结构

两臂共用 action-blind vendor-GDN/FFN trunk；最终 common hidden 先经过 parameter-free
RMS normalization，再走 shared video output projection。candidate 唯一增量改为 common
output 之后的并行 action residual：

```text
g(a)        = bias-free Linear(20,64) -> SiLU -> bias-free Linear(64,20)
odd(a)      = 0.5 * (g(a) - g(-a))
delta_video = zero-init bias-free Linear(20,3)(odd(a))
prediction  = common_prediction + typed_active(delta_video)
```

common 参数只优化 paired common-target MSE；candidate action 参数只优化 signed
half-delta MSE。两项都是正 loss，没有 loss subtraction；optimizer scope 互斥。该设计
直接针对 A2 的 common/delta 混淆。

### 7.2 结果

条件：vendor kernel、synthetic、每臂 200 macrosteps、final-step only。

| 指标 | 结果 |
|---|---:|
| candidate/reference gain | `0.9476196610` |
| action explained fraction | `0.9578645920` |
| delta NMSE | `0.0421354080` |
| alignment cosine | `0.9787154226` |
| energy ratio | `0.9493068285` |
| no-action recovery | `0.9578645913` |
| shuffle penalty | `3.8143428410` |
| final common MSE | `0.0033002337` |
| oddness max/rms | `0 / 0` |

typed verdict：`OPERATOR_GO_COMMON_STABLE`。RESULT SHA256：
`81345a1c9b6e03d60dd587cb439ebdd9c2d60fc92a7120d80042c6a9e4dc78c3`。

解释：A3 在冻结 synthetic mechanism task 上同时解决了 common stability 和 action
delta measurement，说明 output-space exact-odd residual 是可行机制。但这仍是与其
训练分解高度一致的 synthetic screen，不自动支持真实数据或时序 generalization。

## 8. CACH-A4：state-conditioned causal exact-odd delta stream

架构 ID：`CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1`。

### 8.1 结构

令 `q_t` 为 detached、parameter-free RMS-normalized common hidden，`d_t` 为
candidate-only recurrent delta state：

```text
u_t      = 0.5 * (g(q_t, a_t) - g(q_t, -a_t))
lambda_t = sigmoid(W_decay(q_t))                 # action-blind decay
d_t      = lambda_t * d_(t-1) + m_t * W_write * u_t
r_t      = W_out * d_t                           # W_out theta0 exact zero
y_t      = common_t + typed_where(m_t, r_t, exact_zero)
```

它保持 exact odd、zero origin、inactive/no-action/seam-disabled direct bypass；future
action 不得影响 prefix，past action、common state 和 carried delta state 都必须对未来
output 有正 JVP。

### 8.2 A4 synthetic operator screen

初始 base run 因 PyTorch/Inductor 与 Triton 的 `triton_key` import 不兼容而
`HARNESS_REJECTED`；R1 修环境后，又因 diagnostic mask `[8,5,1]` 直接扩展到
`[8,5,3,1,1]` 的 shape bug 而 invalid。两者都不是架构负证据。R2 只修 diagnostic
mask shape，架构和预算不变。

R2 条件：vendor kernel、synthetic、每臂 200 macrosteps。

| 指标 | 结果 |
|---|---:|
| candidate/reference gain | `0.9829696974` |
| action explained fraction | `0.9967108493` |
| delta NMSE | `0.0032891507` |
| alignment cosine | `0.9983542596` |
| energy ratio | `0.9954834287` |
| no-action recovery | `0.9967108494` |
| shuffle penalty | `3.9843885560` |
| final common MSE | `0.0033002337` |
| action/common-state/past-action/delta-state JVP | 全部为正 |
| future→prefix、oddness、zero-origin | exact zero |

typed verdict：`OPERATOR_GO_CAUSAL_STREAM_COMMON_STABLE`。RESULT SHA256：
`fabce5d1ac38294f79a6313c35041a723f4cb1055045ce9825a6f24d4202babd`。

解释：A4 在 synthetic target 与 causal recurrence 相匹配时表现很好，且状态、动作和
因果 wiring 全部有效；这足以进入 tiny real-data screen，但不是生产或科学通过。

### 8.3 AV-2 tiny real-data overfit

AV-2 保持 A4-R2 架构，使用 RoboTwin `adjust_bottle / aloha-agilex_clean_50`：

- train episodes `0..7`，heldout `8..15`；
- 每个 episode 取 rows `[0,33)`；
- frames `[0,8,16,24,32]`；
- image 保持 OpenCV native BGR，经每帧 spatial channel mean / 255 投影为
  `[5,3,1,1]`；
- absolute EEF20 actions，correct/shuffle/no-action；
- 两臂 fresh initialization，每臂 1,000 AdamW steps；
- 不加载或保存 checkpoint。

base execution 的 exact-zero GPU idle probe 在 root 前被 1 MiB driver allocation 阻塞，
没有创建 root；R1 只把 idle 阈值修为 `<=8 MiB`。R1 随后进入 root，但终结 RESULT
时因 `TorchVersion` 非 JSON 类型而 `AV2_INVALID_FAIL_CLOSED`。R2 只机械修复
serialization，并使用 fresh root。

AV2-R2 有效结果：

| 指标 | 结果 |
|---|---:|
| candidate heldout correct MSE | `0.0001200027` |
| reference heldout MSE | `0.0032369387` |
| candidate/reference ratio | `0.0370729015` |
| correct vs shuffle improvement | `0.7791090291` |
| correct vs no-action improvement | `0.9846577303` |
| candidate train MSE | `0.0000172410` |
| reference train MSE | `0.0031023202` |
| train loss, candidate | `0.3864518 -> 0.0000171460` |
| train loss, reference | `0.3864518 -> 0.0030996869` |

所有 data adequacy、vendor call、JVP、oddness、zero-origin、future isolation、target
sentinel 和 bypass checks 通过。typed verdict：`AV2_GO`，`av3_unlocked=true`。
RESULT SHA256：
`3d1ae1b81347efd258dba1d05f93f4218bb7fd7706f8b2c5a3cc3419ae2230a6`；
SCREEN SHA256：
`3073ffa1cf2eb2f1a2d82a86d6b595f12d761f26b274a1e4b5e010a412af5a4c`。

解释：A4 能在一个很小且未按结果挑选的 real-data slice 上拟合真实 observation/action
对齐，并显著区分 shuffle/no-action。但该 screen 是 8+8 tiny overfit、spatial-mean
`1x1` projection 和 aggregate future MSE；它没有验证新 cohort 上的 horizon-specific
anchor-relative dynamics，也不能排除 tiny-set memorization。

### 8.4 AV-3 paired reduced-scale

AV-3 仍保持同一 A4 架构，但改用未参与 AV-2 判定的新 cohort：

- train episodes `16..31`，heldout `32..47`；
- 16 train / 16 heldout，microbatch 8；
- horizon `1/2/4`，权重 `0.20/0.30/0.50`；
- action-clamped single-call causal sequence；
- primary = `0.5 * video_nmse + 0.5 * delta_nmse`；
- 每臂 fresh-init 1,000 AdamW steps；candidate 只用 correct action 训练；
- shuffle/no-action 只用于 theta0 diagnostic 与 final heldout counterfactual evaluation。

三个前置 revision 都是 harness 问题：

1. base execution 在 root 前因 `zip(offsets, offsets[1:], strict=True)` 长度不等失败；
2. R1 在 root 前误用 AV-2 的 8+8 disjoint helper 检查 AV-3 的 16+16 cohort；
3. R2 创建并冻结 root，但在 CUDA allocator 初始化前调用
   `reset_peak_memory_stats(cuda:0)`，得到 `Invalid device argument`，状态
   `IMPLEMENTATION_INVALID`。

R3 只在该 module 内确保 allocator initialization 发生在冻结 reset call 前，不改变
architecture、data、optimizer、metric 或 threshold。AV3-R3 全部 22 项 validity
checks 为真，并完成每臂精确 1,000 steps。

最终 heldout：

| 指标 | reference correct | candidate correct | candidate shuffle | candidate no-action |
|---|---:|---:|---:|---:|
| primary | `0.6884537659` | `1.1457939966` | `1.1784093909` | `0.9499110402` |
| video NMSE | `0.3782301207` | `0.0058334732` | `0.0384557681` | `0.8985133573` |
| delta NMSE | `0.9986774111` | `2.2857545200` | `2.3183630137` | `1.0013087231` |

决策指标：

| 指标 | 结果 | GO/STOP 含义 |
|---|---:|---|
| candidate gain vs reference | `-0.6643005723` | candidate 明显更差 |
| candidate/reference primary ratio | `1.6643005723` | 超过 strong-stop `1.25` |
| correct vs shuffle improvement | `0.0276774732` | 低于 GO `0.05` |
| correct vs no-action improvement | `-0.2062118957` | correct action 劣于 no-action |
| h1 candidate/reference video-error ratio | `0.0013208987` | 单步 video error 通过 |

训练 loss：candidate `0.3895173 -> 0.0000738062`，reference
`0.3895173 -> 0.0031569076`。action seam gradient/update/JVP 为正；reference 三种
action mode bitwise equal；future→prefix、oddness、zero-origin、target sentinel 和
resource/deadline checks 全通过。

typed verdict：`REDUCED_ARCH_STOP / AV3_STRONG_STOP_LINE`，`av4_unlocked=false`。
RESULT SHA256：
`588c64dd7e6971e96a226dded4ec36c16cd301a51bf8b26f71b13e0885c5e4c5`；
SCREEN SHA256：
`f00d1bdc4593c35c032f499ffc6fa81ebd4259ad7e7b8d169534756a8984f4e0`。

解释：candidate 已把 absolute video values 拟合得非常好，但预测的 anchor-relative
motion/delta 错误约为目标 motion energy 的 `2.286` 倍；reference 的 delta NMSE 约
为 `0.999`。candidate 的 correct action 只比 shuffle 好 `2.77%`，且比 no-action
差 `20.62%`。结合低训练 loss、有效 optimizer updates 和全部 causal/wiring checks，
这表明当前 A4 treatment 在该 prospective cohort 上学到了不利于 heldout dynamics 的
action-conditioned 映射，而不是 launcher、数值或“训练没跑起来”的假阴性。

## 9. 关键实验总表

下表中的不同阶段 metric 定义不同，数值不可横向当作同一个 benchmark；可比较的是
各自冻结 card 下的 typed verdict 和 evidence level。

| 阶段 | 架构/数据/预算 | 终态 | 架构证据解释 |
|---|---|---|---|
| AV-0 | card-only | no run | 只冻结 CACH-A 问题与阈值 |
| AV-1 | CACH-A，CPU proxy，200/arm synthetic | `PROXY_GO` | proxy learnability positive；actual architecture 未评估 |
| AV-1B | CACH-A，vendor，200/arm synthetic | `REVIEW_ONCE` | valid；delta 可学但 action discrimination 弱 |
| review300 | 同 CACH-A，fresh 300/arm | `OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED` | valid；gain 正，但 shuffle/no-action 仍未达线；token consumed |
| A2 base | preflight | blocked before root | hostname/version launcher bug；非架构证据 |
| A2-R1 | A2，vendor，300/arm synthetic | `...A2_DELTA_LIVE_COMMON_MODE_BLOCKED` | valid；delta 极准，common error 主导 |
| A3 | A3，vendor，200/arm synthetic | `OPERATOR_GO_COMMON_STABLE` | valid synthetic mechanism GO |
| A4 base | vendor synthetic | `HARNESS_REJECTED` | Triton/Inductor env incompatibility；非架构证据 |
| A4-R1 | vendor synthetic | `HARNESS_REJECTED` | diagnostic mask shape bug；非架构证据 |
| A4-R2 | A4，vendor，200/arm synthetic | `OPERATOR_GO_CAUSAL_STREAM_COMMON_STABLE` | valid synthetic causal-stream GO |
| AV-2 base | pre-root | blocked before root | exact-zero idle probe 过严；非架构证据 |
| AV-2 R1 | A4，tiny real data | `AV2_INVALID_FAIL_CLOSED` | TorchVersion JSON serialization；非架构证据 |
| AV-2 R2 | A4，8+8 real data，1000/arm | `AV2_GO` | valid tiny overfit positive；只解锁 AV-3 |
| AV-3 base | pre-root | `ENV_BLOCKED` | strict zip observation bug；非架构证据 |
| AV-3 R1 | pre-root | blocked before root | 8+8 helper 错用于 16+16；非架构证据 |
| AV-3 R2 | root created | `IMPLEMENTATION_INVALID` | allocator init 前 reset；非架构证据 |
| AV-3 R3 | A4，16+16 real data，1000/arm | `REDUCED_ARCH_STOP` | 唯一有效 AV-3；A4 reduced path strong stop |

## 10. 跨阶段分析

### 10.1 已经建立的事实

- causal layout、typed action mask、zero-origin、oddness、future isolation、reference
  action-blind bypass 和 target isolation 在 reduced harness 中可实现并通过检查。
- actual vendor GDN forward/backward/JIT 可与 candidate action seam 联合优化。
- action conditioner 不是死分支；A2/A3/A4 都能学到非平凡 action delta。
- common/delta 正交化解决了 A2 synthetic common-mode 混淆。
- A4 recurrence 确实读取 common state、past action 和 carried delta state，并保持
  future-to-prefix exact zero。
- A4 可在 tiny 8+8 real-data slice 上 overfit 并通过 heldout counterfactual 对照。

### 10.2 当前失败的 load-bearing hypothesis

AV-3 直接回答“该 A4 reduced architecture 在新数据上是否改善 action-conditioned
short dynamics”。答案是否定的：

- candidate 的 video reconstruction 很好，但 motion/delta prediction 比 reference 差；
- correct-vs-shuffle 分离低于冻结阈值；
- correct action 比 no-action 更差；
- candidate/reference primary ratio 触发强停止；
- 低训练 loss、live gradients/updates/JVP 和全部 validity 排除了主要 harness/优化假阴性。

因此不能用 AV-2 的 `AV2_GO` 声称架构已验证成功。AV-2 与 AV-3 并不矛盾：前者是
8+8 tiny overfit 和 aggregate MSE，后者是 prospective 16+16 cohort 与显式
motion-sensitive primary；后者正是用于检查前者是否值得扩大投入。

### 10.3 尚未建立的事实

以下全部仍为 `NOT_ASSESSED` 或 blocked：

- 原始完整 CACH v0 package；
- CACH-H anchors、CACH-R AttnRes、CACH-SF generated-prefix；
- full 2B、真实 latent resolution、VAE/text encoder/checkpoint closure；
- public production wrapper/dispatcher/cache-owner parity；
- transitive Python/native/JIT runtime closure；
- complete-model C0–C8、Global Stage 3；
- multi-seed AV-4、正式 benchmark、closed-loop capability；
- registered/formal training、checkpoint save/resume、deployment。

`REDUCED_ARCH_STOP` 只停止当前 A4 reduced investment path，不是对完整 2B CACH package
的科学证伪；但它也不允许在没有新 architecture decision 的情况下继续追加 AV-3
预算、复用旧 root 或重跑旧 R1/R2。

## 11. 冻结证据索引

### 11.1 Governing plans

| Artifact | SHA256 |
|---|---|
| `docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md` | `969d9668ffed07176844fcbd84062825f9d5f0af0325959b0068c38fe481f0dd` |
| `docs/cach_sana_wam/global_stage3/post_phase_c_planning/ARCHITECTURE_VALIDATION_FIRST_PLAN_20260803_v1.md` | `06f5d09127f8bc2a945cdbdc095054f40b1fe8d9757a90eccbdf0604ea89e079` |

### 11.2 Architecture decisions/cards

| Artifact | SHA256 |
|---|---|
| AV-0 card | `c73a943047295338cd9de14e6e53199b463ea66ae92b99d4b0b3393f5ec34030` |
| AV-1B card | `c0e15bcb8c20441d7d989ec6d375284e45f24962c98e15b984dbbbbd198248fb` |
| AV-1B review300 card | `a552522a951384996aa8f18ce9cb8a5e6e130ac4d2db68515b5ea6303c6722bc` |
| A2 decision / card | `280e17b6e9a7433e64b20ceed587e998ab0ef1322fe7cc7467ee6a56f9589363` / `9d3c3973234df8a9593b254161ca07bf72f343c4475d9985e2901153b82a9713` |
| A2-R1 launcher decision / card | `db51c2d2fca6b0327a1466595d7297630a39a6091dfc45bc2f9633b10d283c23` / `c7b710177defda6d44066c7b7b25c4b32d11ec106a92662b0701ca5f87f3401e` |
| A3 decision / card | `ec99746eb5abbe6de29974d4fcf4e56a043fcbbf6df490264fe9fe4581b30979` / `674073b39c1964bbc9c955022437390445dbb2bddfe42403b2c495f5a900be19` |
| A4 decision / base card | `2a2efb3bda76dd36d2d1b98c0d26703de40704bc0744098a93b9820eac0b3a92` / `b069c5fcbfde49e1f2e7a7ae0feb47067910ed680d09b28eb8459deb22ef618d` |
| A4-R1 / R2 cards | `faddcddcb7a1a6a69f013f615f39e99e78b128b50fe7d4a6f67fe3b8a335c0a7` / `a45b55ffea796aa1edbcb46fae77654aa850965c312ed72b26797dffa4b1e6c6` |
| AV-2 decision / scaffold card | `b28ccdc0648c8f6105d507bdb714d0f15293285a5eefc37900052d3b11dc0d79` / `22939aa707a73ec40e37838d08dcb0b57eaa2169f3f5e64615096db61ee99f58` |
| AV-2 execution base / R1 / R2 cards | `5fec6ea1497548b19272deedc142caa05b715e3b130dad6bd3dc32e0ba19a376` / `975747b212bd41f29f18ebb35f8cc05be076939ac13e393fd907b853c521e7ac` / `5941abf3886e6b385d1b95d0b06dc2c72f718b8610e71d5502f42d82be12d882` |
| AV-3 decision / planning card | `a0a211915704b3230747021a566e06ff47478b20f3acb2681973e6ac9e3ebd30` / `33566c3291e49e4a87b4f83807d80255be67a5083615358c92bde924046bc82f` |
| AV-3 execution base / R1 / R2 / R3 cards | `72124b63e278490ff39262aaa5bcf5a71121a533df6f55aff7147d98b11e06f9` / `9e84e816fc9f6349a188a1adaebd8ede0ce92430a9db48d29ee52cd678a04c3e` / `9a3c7f4763b2cbef22618301df1321e3cbc93f0b5fccf11c0620fc461eed2c02` / `7e0cb156f8c22ab361be8147ff04f5424f8c13fc438ff39378c2c7e533aa69d0` |

### 11.3 Main result roots

| Run | Immutable root | RESULT SHA256 | Freeze receipt SHA256 |
|---|---|---|---|
| AV-1 | `/DATA/share/sana_cach_wam_nonformal_screens/av1/06f5d09127f8/c73a94304729-d755b67c667d9a22686c8d90d187832d` | `b7b23642d5772efb09e3c0ff09cb1aead3d5834b70b325ddc8b1c10d7438303e` | `c093b2b067c6390f2439c6132b08a31ab6e04869ec314b7ebc4c164673c9c774` |
| AV-1B | `/DATA/share/sana_cach_wam_nonformal_screens/av1b/06f5d09127f8/av1b-1a9a3a6ca4b18fac904fdb806c2a8fb3` | `d33e2e6bdf59cc1d2ffe51b32d48ce22daa4da1a902e3e9bded28cce29adb5de` | `2949acd034ffc6f59038d2ba3b2501854fcb0b94c11a8675d96d0ab40907f677` |
| review300 | `/DATA/share/sana_cach_wam_nonformal_screens/av1b_review300/06f5d09127f8/av1b-review300-bd32b08576c9d598c629b90286331b87` | `3ca4ac673893923d29f60b999879132a93d515382f94bdaa462d91d3cdd1786d` | `cbcc626e587d6b9b3efab5aa50171ef9de95110b6eb2fca0a9a1959036aa2a24` |
| A2-R1 | `/DATA/share/sana_cach_wam_nonformal_screens/cach_a2_r1/06f5d09127f8/cach-a2-r1-launcher-fix-dc17cb7bdbd70955ea4b91212f0d4ac3` | `a7767e47d718deda41762db4e250dbca0602f78fcff1651ef06a0d6da508c2ef` | `5e96575d2057d395e283924c7519e7cfcc51c560bf846d070547c1a26484cb97` |
| A3 | `/DATA/share/sana_cach_wam_nonformal_screens/cach_a3_orthogonal/06f5d09127f8/cach-a3-orthogonal-b2e55836da6fd5a51b568af613f74eef` | `81345a1c9b6e03d60dd587cb439ebdd9c2d60fc92a7120d80042c6a9e4dc78c3` | `69b72fd730b316509548e4dab198e4dd7d84565ee9f1d05a6cccd1e8a673c63c` |
| A4-R2 | `/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r2/06f5d09127f8/cach-a4-state-stream-r2-f294a23c285082ae61b1685accec416e` | `fabce5d1ac38294f79a6313c35041a723f4cb1055045ce9825a6f24d4202babd` | `ec70c0c409625b0d6d0d3c3e33d841e75313eb10ccea2b30f725be1fd0094a03` |
| AV-2 R2 | `/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_av2_tiny_real_data_r2/06f5d09127f8/cach-a4-av2-r2-b795b1b6edba6aeb584c06db1d166bf7` | `3d1ae1b81347efd258dba1d05f93f4218bb7fd7706f8b2c5a3cc3419ae2230a6` | `58d4ba15d800efc051ad4e92d3a948f2cb6f5d109c60ce683d0ac13aaf809ddc` |
| AV-3 R3 | `/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_av3_paired_reduced_scale_r3/06f5d09127f8/cach-a4-av3-r3-3bf88ade4d71e73035e9196fe00043fb` | `588c64dd7e6971e96a226dded4ec36c16cd301a51bf8b26f71b13e0885c5e4c5` | `3d4ca6c4822c1ca2d4b7efc272e263ff4cee2e7344c54a1e3eb30be539dd32bd` |

### 11.4 Frozen invalid roots

| Run | Immutable failure root | Failure | RESULT SHA256 | 解释 |
|---|---|---|---|---|
| A4 base | `/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream/06f5d09127f8/cach-a4-state-stream-ac7fb52eb22be3cce184a60a41f7e98e` | `BackendCompilerFailed: triton_key import` | `07f5246d1ad0d8b076f60fdb9f41c5ab53a3db8f9c4c9d31524f705b98a26323` | environment/harness invalid |
| A4-R1 | `/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r1/06f5d09127f8/cach-a4-state-stream-r1-9e604d41d671ac4e705816857457e5a4` | diagnostic mask shape `RuntimeError` | `ba24e73a316dcc92e2d759f90d9e309db9747db88f60842440bcd9e52b3875e0` | implementation invalid |
| AV-2 R1 root | `/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_av2_tiny_real_data/06f5d09127f8/cach-a4-av2-96c7c972d34733dd37475b183c136ee0` | `non-JSON result value: TorchVersion` | `0e329e52a2adf74d605bcbbfcc8dc7da766caa41be883fc8ee226ce2ba9a2344` | publication harness invalid |
| AV-3 R2 | `/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_av3_paired_reduced_scale_r2/06f5d09127f8/cach-a4-av3-r2-58cf90abd40b552226636f3c9455c1fb` | allocator reset `Invalid device argument` | `b2573479850abafa05724e15d0d5d937ff5af3fe8f3672cbee179c70e2f9f46a` | implementation invalid |

这些 root 同样必须保持 immutable；它们用于证明失败发生在哪里，不能被删除、覆盖、
复用或重解释为架构失败。

## 12. 截至当前的开发状态

```text
已完成：
  contract/layout/cache/ledger/recovery platform
  proxy and vendor learnability screens
  CACH-A -> A2 -> A3 -> A4 reduced architecture iterations
  A4 AV-2 tiny real-data overfit
  A4 AV-3 paired reduced-scale seed-0 screen

当前终态：
  CACH-A4 reduced path = REDUCED_ARCH_STOP
  AV-4 = NOT_UNLOCKED
  review token = CONSUMED
  independent LIBERO production-shaped AR path = T13 multi-episode accumulation-8 balanced joint replicated

未开始或未通过：
  full CACH v0 package validation
  CACH-H / CACH-R / CACH-SF experiments
  complete 2B / public production parity / Global Stage 3
  formal training / formal evaluation / closed-loop capability
```

如果继续研发，下一项必须是一个新的、明确注册的 architecture hypothesis 和 fresh
card/root，而不是再次执行 AV3-R1、延长 A4 预算、选择 checkpoint/window/seed，或复用
任何旧 root。该下一步尚未由本文选择或授权。

## 13. 后续补记：LIBERO production-shaped AR 验证线

在上述 RoboTwin/CACH-A4 reduced-path 停止后，项目新增了一条独立的 LIBERO
architecture-validation 线。它运行的是完整 production-shaped
`DualSystemARArchitecture`：约 5.84B 总参数，冻结 SANA video backbone，只训练
`action_backbone`、`proprio_encoder`、`proprio_video_embed` 和
`proprio_action_embed` 共 639,653,063 个参数；video timestep 使用 continuous FP32
T1，optimizer 使用 persistent FP32 masters，再精确投影到 BF16 model tensors。

这条线不是 AV3-R3 的重跑，也不是把 `REDUCED_ARCH_STOP` 改写成 GO。前者检验的是
CACH-A4 reduced action-delta operator 的 prospective motion/delta generalization；后者
检验完整 AR training path 在真实 LIBERO observation/action loss 上是否数值闭合并发生
有限迁移。两类证据对象、loss 和判定门都不同。

前置 closure 依次完成：CPU real-data smoke、单 GPU update-free forward、gradient-
checkpointing 传播修复、frozen-video input-gradient 修复、FP32 optimizer-master 修复，
以及排除 Goal episode 82 后的 selected-row statistics v2。没有加载或保存 SANA-WAM
training checkpoint，也没有运行 simulator 或 benchmark evaluator。

| 阶段 | 固定问题与预算 | 终态 | 关键结果 |
|---|---|---|---|
| T1 | Spatial ep0/start0，fresh init，1 个 AdamW update | one-update PASS | 560/560 FP32 masters 获得 finite nonzero grad 并更新；四个 trainable roots 全覆盖 |
| T2 | 同一 ep0、同一 recipe，20 updates | `T2_FIXED_SAMPLE_LEARNABILITY_GO` | action loss `13.679719 -> 1.733191`，ratio `0.126698`；训练曲线振荡且 20/20 gradients 被 clip |
| T3 | 保持 T2 core，增加 3 个 update-free held-out recipes | `T3_HELDOUT_RECIPE_TRANSFER_GO` | 3/3 改善；ratio `0.124502 / 0.416718 / 0.416813`，median `0.416718`；T2 core 逐值复现 |
| T4 | 保持 T2/T3 core，同 recipe，3 个 same-task update-held-out episodes | `T4_HELDOUT_SAMPLE_TRANSFER_GO` | ep16/405/40 全部改善；ratio `0.123714 / 0.139309 / 0.123339`，median `0.123714`；T3 core 逐值复现 |
| T5 | 保持 T3/T4 core，同 recipe，3 个 mechanically selected distinct-task episodes | `T5_CROSS_TASK_TRANSFER_GO` | task7/1/4 的 ep36/325/11 全部改善；ratio `0.148637 / 0.113500 / 0.150387`，median `0.148637`；T4 core 逐值复现 |
| T6 | 保持 T5 core，同 recipe，Object/Goal/LIBERO-10 各一个 mechanically selected sample | `T6_CROSS_SUITE_TRANSFER_GO` | S1/S2/S3 全部改善；ratio `0.157667 / 0.176770 / 0.152686`，median `0.157667`；T5 core 逐值复现 |
| T7 | A0-A3 四套件循环 5 轮，共 20 updates；每套件一个同任务 fresh heldout | typed numerical `T7_FOUR_SUITE_CYCLIC_GO` / audit-qualified | A median ratio `0.625211`，H median `0.636552`，3/4 对应套件双改善；Spatial A/H 分别回退 `9.48% / 7.63%` |
| T8 | 同一 A/H、同一剂量，仅把循环相位旋转为 `[Object,Goal,LIBERO-10,Spatial] × 5`，并在各套件第五次更新后加 phase probes | `T8_PHASE_ROTATED_MIXED_INCONCLUSIVE` | Spatial A/H 被救回，Object A/H 转为回退；2/3 nonterminal overwrite 过阈值，但 `rho_recency=0.4` 未过 `0.8` |
| T9 | 补齐循环相位 C=`[Goal,LIBERO-10,Spatial,Object]×5` 与 D=`[LIBERO-10,Spatial,Object,Goal]×5`，再只读组合 T7/T8/C/D | `T9_COMMON_POSITION_EFFECT_SUPPORTED` | 4/4 suite 均 `q(3)>q(0)`，Object/Goal/LIBERO-10 的 `rho>=0.8`，故 `endpoint_count=4, strong_count=3`；Spatial `rho=0.4` |
| T10 | matched SEQ 与四套件等权 JOINT 各 20 macro steps；每个 macro 固定 A0-A3 四次 micro-backward 后仅一次 update | `T10_BALANCED_JOINT_SIMULTANEOUS_RETENTION_SUPPORTS_OVERWRITE` | SEQ 对 T7 八个 terminal loss 零误差复现；JOINT 的 8/8 `rA/rH<1`，4/4 q 低于 T9 median，3/4 suite componentwise Pareto 优于 SEQ |
| T11 | 保持 T10 balanced-JOINT core/任务/seed/预算，为四个 suite 换用机械选定的全新 update/probe episodes | `T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT_REPLICATED` | 新 episodes 上 8/8 `rA/rH<1`，update/held-out median ratio 为 `0.200040 / 0.206788`；4/4 q 均低于 T10 JOINT q |
| T12 | 保持 T11 balanced-JOINT topology/seed/预算，换用四个此前从未 model-facing 的机械选定新任务，每任务一个 update 与同任务 heldout episode | `T12_NEW_TASK_BALANCED_JOINT_REPLICATED` | 新任务上 8/8 `rA/rH<1`，update/heldout median ratio 为 `0.099378 / 0.096183`；4/4 q 均低于 T11 q（只作诊断） |
| T13 | 保持 T12 四个任务/seed/20-step objective，每任务扩为两个 update 与两个 heldout episodes；每 macro 八次 singleton micro-backward 对齐 production accumulation-8 boundary | `T13_MULTI_EPISODE_ACCUM8_BALANCED_JOINT_REPLICATED` | 16/16 ratio `<1`，update/heldout median 为 `0.116163 / 0.109951`；4/4 q 略高于 T12（只作跨 cohort 诊断） |

T4 的 frozen training-core expected/observed projection SHA256 均为
`e34a2dd0ae2dfe28303dcd9baf64ffc5800d002b0b7646b89dde2aafa132c352`，排除了
successor measurements 改变 ep0 update path 的解释。精确执行计数为四次
`prepare_inputs`、28 次 architecture forward、20 次 backward、20 次 AdamW step；
三个 held-out episode 从未进入 backward/update，post probe 没有重新 prepare。

T5 把 probe 轴扩展到同一 Spatial suite 内三个不同的非训练 task。eligible population
由冻结 metadata 重建为 9 tasks / 386 episodes，再经过 task-level 和 per-task episode-
level 两阶段 SHA256 排序，固定得到 task7/ep36、task1/ep325、task4/ep11。三者 action
loss 分别从 `15.977773 / 16.581335 / 19.208385` 降到
`2.374895 / 1.881981 / 2.888686`，全部改善；median post/pre 为 `0.1486374165`。
T5 的 expected/observed training-core projection 仍精确等于
`e34a2dd0ae2dfe28303dcd9baf64ffc5800d002b0b7646b89dde2aafa132c352`，执行计数仍为
4 prepare / 28 forward / 20 backward / 20 AdamW step，且三个 cross-task samples
从未进入 backward/update。

T6 再把同一个 Spatial task0/ep0 20-step core 的 probe 轴扩展到三个非 Spatial suite。
候选域固定为 Object 454、Goal 427（只排除 Goal ep82）和 LIBERO-10 379 个 episode，
每套先对 10 个 task 做 SHA256 排序，再在选中 task 内对 episode 排序，固定得到 Object
task3/ep82、Goal task2/ep70、LIBERO-10 task3/ep259。三者 action loss 分别从
`14.696585 / 12.066481 / 20.431152` 降到 `2.317167 / 2.132989 / 3.119553`，全部改善；
median post/pre 为 `0.1576670424`。T6 expected/observed training-core 继续逐值等于上述
`e34a2dd...` projection，且精确执行 4 prepare / 28 forward / 20 backward / 20 AdamW
step；三个 suite probes 从未进入 backward/update。

T7 把 T6 的四个跨套件端点提升为 A0-A3 更新集，固定按
`[Spatial,Object,Goal,LIBERO-10] × 5` 使用一个 persistent FP32-master AdamW optimizer，
并为每套件机械选择一个同任务、零更新的 fresh episode H0-H3。A 样本的 post/pre ratio
为 `1.094798 / 0.765527 / 0.484895 / 0.137526`，median `0.6252105785`；H 样本为
`1.076338 / 0.757779 / 0.515325 / 0.138203`，median `0.6365520894`。Object、Goal、
LIBERO-10 在 A/H 两侧均改善，满足冻结的 3/4 联合门槛；Spatial 两侧同时回退，因此该
GO 不是 uniform four-suite improvement。运行精确执行 8 prepare / 36 forward / 20
backward / 20 AdamW step，每个 A 恰好 5 次更新、每个 H 为 0；没有 SANA-WAM
checkpoint、simulator、rollout 或 benchmark evaluation。

独立 post-run audit 重算了 canonical RESULT、source/config/runner pins、manifest、顺序、
计数和三重数值门，均一致；同时指出更强的预注册 validity 文字没有被全部采集证明：没有
跨 run 的 bytewise fresh-model/master fingerprint，update loop 没有全程 buffer baseline，
且全量 BF16/master projection equality 只在 step 1/20 检查。因此 T7 保留 immutable typed
numerical GO，但严格限定为 architecture signal，不能作为 formal training admission。

T8 保持 T7 的模型、初始化、recipe、八个样本和每样本五次更新，只旋转循环终端相位。
执行 source commit 为 `9cd1c490d14b3c2437e82225ee8cfdf58646837e`，runner SHA256 为
`9ec74d83c8443d78d2ba765dd766f82a96cccef894ae63d08cf9e2a8dbf97669`。
所有套件在自身第五次更新后的 A/H phase probe 都改善；终端 Spatial ratio 为
`0.661098 / 0.630339`，而 Object 为 `1.580494 / 1.597984`。Object 与 LIBERO-10 的
overwrite penalty 分别为 `2.551829 / 1.125215`，但 Goal 在两个 trailing updates 后继续
改善，使 `rho_recency=0.4`，故 primary recency predicate 为 false；Spatial 没有双回退，
故 suite-effect predicate 也为 false。诊断性 joint gate 仍通过：A/H median 分别为
`0.6287817650 / 0.6149701490`，3/4 套件双改善。运行严格为 8 prepare / 44 forward /
24 update-free measurements / 20 backward / 20 optimizer steps，未加载或保存
SANA-WAM checkpoint，也未运行 rollout 或 benchmark evaluation；仅执行一次，没有重跑。

独立 T8 post-run audit 对 canonical JSON、冻结 pins、预算、全部 ratio、两个 Spearman 和
typed verdict 的复算均无数值差异。审计同时限定：预注册要求逐 event 序列化 buffer-version
assertion，实际 RESULT 只保存 initial/final global map 和 probe-group snapshot；未观察到任何
buffer identity/data pointer/单调 `_version` 变化，但更强的逐 event 报告契约没有被完整采集。
因此 T8 与 T7 一样保留 immutable、audit-qualified non-formal evidence，不据此重跑或解锁
formal admission。T8 还继承了 T7 位于 `/tmp` 的 T1 predecessor pin；执行时文件及 SHA
正确，但该位置不是持久证据存储。

T9 保持同一个模型、初始化、recipe、八个样本和每样本五次更新，只执行剩余两个循环
相位。C 与 D 分别从 fresh model/empty optimizer 启动，source commit 均为
`f73a7950eded2787ad57b26523da686dc5722512`，arm runner SHA256 为
`d0d4cacf35d4116db3a8c166ffcdc92578a28f5037614a960e22b4ccbcc4a84c`；两臂各精确执行
8 prepare / 44 forward / 24 measurement / 20 backward / 20 optimizer step，并得到
`T9_LATIN_ARM_C_VALID` 与 `T9_LATIN_ARM_D_VALID`。C 的 terminal q 为 Spatial(k1)
`0.852055`、Object(k0) `0.457466`、Goal(k3) `1.130739`、LIBERO-10(k2)
`0.986213`；D 为 Spatial(k2) `0.358488`、Object(k1) `0.230310`、Goal(k0)
`0.317411`、LIBERO-10(k3) `0.455242`。

随后标准库 CPU harness 从 raw pre/terminal loss 强校验并组合 T7/T8/C/D 的 16 个 cell，
没有使用 GPU、构造模型、更新参数或加载 checkpoint。四个 suite 的 `[q(0),q(1),q(2),q(3)]`
分别为 Spatial `[0.645719,0.852055,0.358488,1.085568]`、Object
`[0.457466,0.230310,0.761653,1.589239]`、Goal
`[0.317411,0.500110,0.598033,1.130739]`、LIBERO-10
`[0.137864,0.252070,0.986213,0.455242]`；对应 Spearman rho 为
`0.4 / 0.8 / 1.0 / 0.8`，且四者都满足严格 `q(3)>q(0)`。因此冻结的
`endpoint_count=4, strong_count=3` 按预注册规则唯一产生
`T9_COMMON_POSITION_EFFECT_SUPPORTED`。这支持共同的终端位置/保留干扰，但不表示每个
suite 都单调；Spatial 明显非单调，LIBERO-10 在 k2 达到最差值。

T10 用 matched-core 两臂直接检验该位置效应是否来自 sequential optimizer overwrite。
SEQ 与 JOINT 都从 seed `20260806` 的 fresh model/empty optimizer 启动，都在每个 macro
按 A0-A3 固定顺序执行四次 forward/backward，再进行一次 clip、一次 persistent FP32-master
AdamW step 和一次 BF16 projection。SEQ 使用循环 one-hot 权重，JOINT 每步使用
`[0.25,0.25,0.25,0.25]`；两臂都是 20 optimizer steps、80 backward，每个 A raw exposure
20 且累计 scalar coefficient 5。因此区别只在四套件梯度是在不同 optimizer state 顺序进入，
还是在同一个 state 等权聚合。每个 macro 的运行时 identity/version 快照还证明四次
micro-backward 之间 model parameters、FP32 masters 与 optimizer state 均未更新。

SEQ 逐值复现 T7 的八个 terminal action loss，八项 absolute error 全部为 0，建立了精确
bridge。JOINT 的 Spatial/Object/Goal/LIBERO-10 `(rA,rH,q)` 分别为
`(0.276152,0.270262,0.273207)`、`(0.272380,0.254421,0.263401)`、
`(0.294436,0.352028,0.323232)`、`(0.186363,0.188152,0.187258)`。8/8 `rA/rH`
严格小于 1，故按预注册第一分支产生
`T10_BALANCED_JOINT_SIMULTANEOUS_RETENTION_SUPPORTS_OVERWRITE`。诊断上 JOINT 的 4/4 q
均低于 T9 四位置 median，且 Spatial/Object/Goal componentwise Pareto 优于 SEQ；
LIBERO-10 的 SEQ q 更低，但 JOINT 仍同时保留强 fit。该结果是在完全 matched update/dose
下的 capacity witness，强支持顺序覆盖的操作性解释，但不证明 gradient averaging 是唯一
机制，也不等价于 rollout 或正式 benchmark 成功。

T11 在不改变 T10 balanced-JOINT 训练拓扑的情况下，仅将四个原 update
样本与四个 same-task probe 全部替换为未被 T1-T10 测量或反传过的新
episodes。source commit 为 `7d53d618234e3e37367c5b1e39e169742c2cf514`，
runner SHA256 为
`ab66fcf6052582631424c97f149c9187d942b1da253e201fc698142e5ad41b54`。Spatial、
Object、Goal、LIBERO-10 的 `(rA,rH,q)` 分别为
`(0.201135,0.206961,0.204048)`、`(0.198945,0.206615,0.202780)`、
`(0.259106,0.226876,0.242991)`、`(0.132470,0.133491,0.132980)`。八个
ratio 全部严格小于 1，因此同一个冻结 RESULT 同时记录执行 verdict
`T11_NEW_EPISODE_JOINT_ARM_VALID` 和科学 verdict
`T11_SAME_TASK_NEW_EPISODE_BALANCED_JOINT_REPLICATED`。运行精确为 8 prepare /
96 forward / 80 backward / 20 macro optimizer step，20/20 macro 的 micro-backward
间无参数/master/optimizer-state 变异证据全部通过；update 阶段 peak
allocated/reserved 约为 `30.288311 / 34.978516 GiB`，20 个 update 加 16 次
measurement 耗时 `125.35481818392873 s`。没有 SANA-WAM checkpoint、
simulator、rollout、benchmark evaluation 或 formal training。

T12 继续保持 T11 的 fresh initialization seed、loss recipe、20-step balanced-JOINT
拓扑和单 suite 一对样本预算，只把任务身份替换为 T1-T11 从未 model-facing 的四个
机械选定任务。source commit 为 `bf4e6f43395e0ba2c177d81856ad37f87c6e91fe`，runner
SHA256 为 `baf19d25d39412a6be02ce35d5c06cae1e8a3089d1c005c35768ca0e0f3581d0`。
Spatial、Object、Goal、LIBERO-10 的 `(rA,rH,q)` 分别为
`(0.084443,0.078483,0.081463)`、`(0.116283,0.121064,0.118673)`、
`(0.114314,0.113883,0.114098)`、`(0.071269,0.071598,0.071433)`。8/8 ratio
严格小于 1，按预注册第一分支产生 `T12_NEW_TASK_BALANCED_JOINT_REPLICATED`；4/4 q
低于 T11 q 仅作诊断。运行精确为 8 prepare / 96 forward / 80 backward / 20 optimizer
step，20/20 no-intra-macro mutation 证据通过。独立审计重算 83 项检查、24/24 selected
assets 和全部前序 pins，得到 0 failure、0 Blocker/Major/Minor。没有 SANA-WAM checkpoint、
simulator、rollout、benchmark evaluation 或 formal training。

T13 保持 T12 的四个任务、initialization seed、loss recipe 和 20-step balanced-JOINT
objective，但在每任务上机械选择两个 update 与两个 heldout episodes，并把每 macro 的
八次 singleton micro-backward 对齐 production `batch_size=1`、
`gradient_accumulation_steps=8` boundary。source commit 为
`68fa88157973f59383a87be1cb3107f5824e64cf`，runner SHA256 为
`ddbd141f3c32f9f89c4e936b1432d379cf56f0b886f84f38406fa4481654efb8`。16 个样本的
ratio 全部严格小于 1；update/heldout median 分别为
`0.11616346529286567 / 0.10995106948665566`。Spatial、Object、Goal、LIBERO-10 的
四样本 q 分别为 `0.103599 / 0.124816 / 0.131623 / 0.079014`。4/4 q 均略高于 T12，
但该跨 cohort 比较预注册为 classifier 外诊断，不改变
`T13_MULTI_EPISODE_ACCUM8_BALANCED_JOINT_REPLICATED`。运行精确为 16 prepare /
192 forward / 160 backward / 20 optimizer step，20/20 no-intra-macro mutation 证据
通过；核心 update/measurement 耗时 `251.37178307957947 s`，update peak
allocated/reserved 为 `44551804416 / 49673142272` bytes。没有加载或保存 SANA-WAM
checkpoint，也没有 simulator、rollout、benchmark evaluation 或 formal training。

主要冻结证据：

| Run | Immutable root | RESULT SHA256 |
|---|---|---|
| T1 FP32-master | `/tmp/sana-wam-libero-t1-one-update-fp32master-0337e28-20260806-a3` | `9d9c139fd67baa8c131ad9e6537862536b8262b732c2fda668a7c8b8b6a8621a` |
| T2 | `/DATA/share/sana_wam_libero_nonformal_screens/t2/2dc1ce730df3/libero-t2-fixed20-20260806-a1` | `67d250cdb13470bea9e9fa53531d3c76c65145e5040dc7a45079d84ac4960a57` |
| T3 | `/DATA/share/sana_wam_libero_nonformal_screens/t3/ac431f8727ac/libero-t3-heldout3-fixed20-220afb60725d0cfd591bc4fe225cdd21` | `c883608f2a47b6258f824d4d97a94f8a390d03bab671a592fb758eea61b3a01e` |
| T4 | `/DATA/share/sana_wam_libero_nonformal_screens/t4/7db8182ef46f/libero-t4-heldout3-fixed20-36120c596971575d286d378df42ac564` | `4c33aaff6d202068d77efc0ac406c74198c56e72527cfabdde046fc9a3a4b6f4` |
| T5 | `/DATA/share/sana_wam_libero_nonformal_screens/t5/5150693a0751/libero-t5-crosstask3-fixed20-cf7dd8a1b1cef03511d2026a48e4a271` | `a656aaef1528537527fe830ad7d4107138b29e8e254b5606b43c46a47e323e83` |
| T6 | `/DATA/share/sana_wam_libero_nonformal_screens/t6/708b1d856986/libero-t6-crosssuite3-fixed20-67a02fcc85508e03f136e09221a6a9d4` | `4855f3771b80349547c985d137426cce79e25597f910eff88e136328424b8b89` |
| T7 | `/DATA/share/sana_wam_libero_nonformal_screens/t7/d19109a2314f/libero-t7-foursuite-cyclic-fixed20-61e0a0ff817970e994b0875be4840ed6` | `9f5181a30cd0d6b676f09315244f7560cd3902004f5f17ca833330e33cb44d3a` |
| T8 | `/DATA/share/sana_wam_libero_nonformal_screens/t8/9cd1c490d14b/libero-t8-phase-rotated-fixed20-2e1efcc69bc552affb5c85b7feeb5175` | `bfdb852e14a5bd9b1c8e776be9f4ff108899eae65d557cebe42b06f1991b0a18` |
| T9 arm C | `/DATA/share/sana_wam_libero_nonformal_screens/t9/f73a7950eded/libero-t9-arm-c-latin-fixed20-5b1425bc07c1162fe6eb0f04164f9b9e` | `213d61f4a63f42983e4a42db6db9410279cc6a898bffcf11f157c560adf39771` |
| T9 arm D | `/DATA/share/sana_wam_libero_nonformal_screens/t9/f73a7950eded/libero-t9-arm-d-latin-fixed20-741c81a12b6dda6592d9cc89b1b78765` | `7326bff58efe3b5d07e83db96aefe539f5da36e30ef453d3404166dade38aca0` |
| T9 aggregate | `/DATA/share/sana_wam_libero_nonformal_screens/t9_aggregate/b7cded5fd9cf/libero-t9-latin-square-combined-ab99dd8758ef03bb191d5fb48f3b9fbc` | `0cfc53b820939123de4bc2a626380495878d4d5c9a37a0be9be667173254149d` |
| T10 SEQ | `/DATA/share/sana_wam_libero_nonformal_screens/t10/6861e5a13fa8/libero-t10-seq-matched-core-fixed20-0d211cfc62324c0f4ab506cad2fd76f6` | `d92fe05fe523c346e90ab6a392ddad9c3ec41764d5581211e6223895a42e8937` |
| T10 JOINT | `/DATA/share/sana_wam_libero_nonformal_screens/t10/6861e5a13fa8/libero-t10-joint-matched-core-fixed20-561f95c143f58bc635259bff41ef4366` | `423ce3e01bea7368786b1c470a790af666baf7504a2235894a59b82efef3ea9b` |
| T10 aggregate | `/DATA/share/sana_wam_libero_nonformal_screens/t10_aggregate/128e1be8cd48/libero-t10-exact-balanced-joint-combined-f77c1cd125343922634395db86812db9` | `8fcd26ea3a59fe6e01cf8f279301c899ed5dab541ada9c57e2ce85888abd147b` |
| T11 JOINT | `/DATA/share/sana_wam_libero_nonformal_screens/t11/7d53d618234e/libero-t11-same-task-new-episode-balanced-joint-fixed20-05e0d719fbc5c25e66ddf435cb47eba2` | `6da12f2e3622d4e6427070bfd10b3dab9550c338a834b7662309cefea98d7417` |
| T12 JOINT | `/DATA/share/sana_wam_libero_nonformal_screens/t12/bf4e6f43395e/libero-t12-new-task-balanced-joint-fixed20-09032d945cdf4dac558ab13b9dabafdd` | `c1f045e62c854ab897305fbed504e765bd229d02479d2d11861f210d26abfe08` |
| T13 JOINT | `/DATA/share/sana_wam_libero_nonformal_screens/t13/68fa88157973/libero-t13-multi-episode-accum8-balanced-joint-fixed20-ce58b18994fa066b49b5e52bbd98b81e` | `a75739991561965d212ad98cc2504cabf8696b2dcaf2e0ad5b46aa774ca00763` |

当前可以支持的最强结论是：在相同 seed 的四次 fresh initialization、三组各八个及
一组十六个真实样本和固定 recipe 下，完整
production-shaped AR path 不仅具备单样本 learnability 与多轴 update-free loss transfer，
而且其四套件顺序训练中的共同 position/retention penalty 可被 exact-balanced joint
objective 消除。T10 的 matched SEQ 精确复现 T7，排除了新 harness 改变历史 sequential
path 的解释；matched JOINT 在相同 optimizer-step、raw-backward 和累计 loss coefficient
预算下让 8/8 train/fresh ratios 同时低于 1。T11 又在相同任务但全新
update/probe episodes 上复现了 8/8 同时改善；T12 再在四个从未 model-facing 的新任务
及其 fresh heldout episodes 上得到 8/8 同时改善，说明该 capacity/retention signal
既不依赖 T10 的特定 episode 组合，也不局限于 T1-T11 已消费任务。T13 进一步把每任务
扩成两个 update 与两个 heldout episodes，并在 production-matched accumulation-8 boundary
上得到 16/16 ratio 严格改善。因此 balanced simultaneous multi-suite objective 是当前
最有证据支持的 successor training topology，继续旋转 sequential phase、只重复 T10
固定样本或再次做单 episode 扩展已不再是高价值核心实验。

该结论仍是 single-seed、short-horizon、non-formal loss-space evidence。T7-T13 probes
属于训练 metadata 与 normalization population；T12/T13 只覆盖每 suite 一个新任务及
有限 episodes，不是严格数据集 holdout。结果不证明 closed-loop success、LIBERO
benchmark performance、suite-level distribution generalization、长程 optimizer stability
或正式训练。单层 episode/mini-batch 扩展已由 T13 完成；下一步高价值核心证据应检查
fresh initialization seed，并保持 fresh source/root。在那之前不能把 T13 verdict 直接升级成
正式训练、checkpoint 或 benchmark admission。
