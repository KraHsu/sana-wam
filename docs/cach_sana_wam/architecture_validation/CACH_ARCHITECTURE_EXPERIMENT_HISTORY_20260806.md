# CACH-SANA-WAM 架构设计、实验结果与阶段性分析

状态：`DEVELOPMENT_HISTORY_SNAPSHOT / NON_FORMAL / NOT_AN_EXECUTION_AUTHORITY`

整理日期：2026-08-06  
规范主机：`H200`  
规范工作树：`/home/zch/workspace/sana-wam`  
主仓 HEAD：`605f1c134b4c983ff80f8489c4bc8847036329e2`  
Sana gitlink：`16b9cec673e3335724ba2d8db25de7f9ed229292`

本文是一份可独立阅读的历史快照，汇总截至当前已经设计的架构、基础实现、所有关键
architecture-validation 实验、无效运行及其解释。事实来自 H200 上的 frozen decision、
run card、source pin、`RESULT.json`、`SCREEN_RESULT.json`、token claim 和 freeze receipt，
不以对话记忆作为证据。

本文不是 run card、训练授权、formal admission、科学结论或部署许可；它不改变任何
predecessor、冻结 root、review token 或 gate 状态。

## 1. 当前结论

截至 2026-08-06，已实际执行的 action-conditioned 架构主线为：

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

未开始或未通过：
  full CACH v0 package validation
  CACH-H / CACH-R / CACH-SF experiments
  complete 2B / public production parity / Global Stage 3
  formal training / formal evaluation / closed-loop capability
```

如果继续研发，下一项必须是一个新的、明确注册的 architecture hypothesis 和 fresh
card/root，而不是再次执行 AV3-R1、延长 A4 预算、选择 checkpoint/window/seed，或复用
任何旧 root。该下一步尚未由本文选择或授权。
