# CACH-A2 ZERO-ANCHOR-PRESERVING ACTION SEAM

状态：`FROZEN_ARCHITECTURE_DECISION_ONLY`

日期：2026-08-04  
规范主机：`H200`  
规范工作树：`/home/zch/workspace/sana-wam`

本文件冻结一个新的 reduced architecture hypothesis：
`CACH-A2-ZERO-ANCHORED-BIAS-FREE-v1`。它不是 AV-1B CACH-A 的预算延长、
bug-fix rerun、结果重解释或 AV-2 解锁。旧 review token 已消费；旧 root、card、
source quartet 和结果必须保持字节不变。

## 1. 当前授权和边界

本次用户授权原文：

> 授权在 H200 创建并冻结 CACH‑A2 ZERO‑ANCHOR‑PRESERVING ACTION SEAM 的 architecture decision 和 run card；允许只读引用 AV‑1B review300 冻结证据及静态 JSON/AST/SHA 校验；禁止创建运行 root、GPU/CUDA/JIT、模型或测试执行、参数更新、训练、真实数据、checkpoint、AV‑2、正式评测及 Global Stage 3。

本轮只允许新增并冻结本 architecture decision 和对应 run card。未来 source
implementation 与 execution 必须分别获得新的、绑定完整 SHA256 的明确授权。

## 2. 冻结 predecessor

- Governing plan：
  `docs/cach_sana_wam/global_stage3/post_phase_c_planning/ARCHITECTURE_VALIDATION_FIRST_PLAN_20260803_v1.md`
  SHA256 `06f5d09127f8bc2a945cdbdc095054f40b1fe8d9757a90eccbdf0604ea89e079`。
- AV-1B review300 card：
  `docs/cach_sana_wam/architecture_validation/av1b_review300/AV1B_REVIEW300_RUN_CARD.json`
  SHA256 `a552522a951384996aa8f18ce9cb8a5e6e130ac4d2db68515b5ea6303c6722bc`。
- Frozen result root：
  `/DATA/share/sana_cach_wam_nonformal_screens/av1b_review300/06f5d09127f8/av1b-review300-bd32b08576c9d598c629b90286331b87`。
- RESULT SHA256：
  `3ca4ac673893923d29f60b999879132a93d515382f94bdaa462d91d3cdd1786d`。
- RAW_EVIDENCE SHA256：
  `e527b8084d1f13d06557de947646ab92d5e947a598602b9bb4550c6d3eed2823`。
- FREEZE_RECEIPT SHA256：
  `cbcc626e587d6b9b3efab5aa50171ef9de95110b6eb2fca0a9a1959036aa2a24`。
- RUN_CONTEXT SHA256：
  `01846ef526eac2e8c8e8b524ae41f06e510cae9d53bc560165b5afb10a0a74e1`。
- Consumed claim SHA256：
  `e7c355d3c4e8bfb942f8baa8a5511fb5964c2c7f2d6c975ef5aec0daa458a91f`。
- Final typed verdict：`OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED`。

这些 evidence 只提供后继设计依据；不得改变或覆盖其结论。

## 3. 只读归因

### 3.1 已确认的 action 信号

review300 的 CACH-A 在 step 300 得到：

- candidate gain versus reference：`0.8231801848642116`；
- counterfactual delta NMSE：`0.04039462445612567`；
- learned half-delta RMS 约 `0.483888`，target half-delta RMS 为 `0.5`；
- horizon action-delta alignment cosine 约为 `0.970–0.989`；
- trained local action JVP RMS：`0.0033564402256160975`。

因此 action-delta branch 不是“没有学到 action”。

### 3.2 已确认的 zero-input 非 identity 容量

旧 CACH-A 的 conditioner 为带 bias 的
`Linear(20,64) -> SiLU -> Linear(64,20)`，每个 block 的输出投影也是带 bias 的
`Linear(20,64)`。`no_action_slot` 虽然固定为零，但零输入仍经过这些模块：

```text
delta_l(0) = W_l [ W_2 SiLU(b_1) + b_2 ] + b_l
```

训练后该残差通常非零。冻结参数摘要也证明 conditioner bias 和 block projection
bias 已从全零初始化发生更新。因此“zero action 不保证 identity bypass”是确定的结构
事实。旧 evidence 没有执行 seam-disabled 对照，不能声称这部分容量已经被定量证明为
收益来源。旧检查
`no_action_candidate_pair_predictions_bitwise_equal` 只证明两个 paired row 在
no-action 下相等，并不证明 same-arm no-action 与 seam-disabled path 相等。

最终 shuffle gap 仅 `0.11237303203902348`，no-action gap 仅
`0.02346693047938381`；这与 action seam 可同时提供无条件容量相容，但不能单独定量归因。

### 3.3 common-mode 是独立阻塞

旧 synthetic recipe 的 target action-effect energy 精确为 `0.25`，target 总尺度约
`0.255`；不存在“target 公共分量过大”这一解释。由冻结 paired evidence 正交分解：

- action-delta error energy 约 `0.01010`；
- candidate pair-common prediction MSE 约 `7.41859`；
- no-action common error 约 `7.35237`。

因此 action delta 的方向和幅度已经较好，主要绝对误差来自预测 common mode/fusion。
两臂 loss 又高度振荡；candidate final loss 约为其固定轨迹 minimum 的 15.9 倍。
继续单纯增加 step 不能被视为根因修复。

## 4. Architecture decision

### 4.1 Frozen identity

```text
architecture_id = CACH-A2-ZERO-ANCHORED-BIAS-FREE-v1
reference_arm   = REF-GDN-CORRECTED
candidate_arm   = CACH-A2
operator_level  = VENDOR_KERNEL
integration     = EXPERIMENTAL_PATH
```

CACH-A2 相对 review300 CACH-A 注册一个 composite seam delta，四个不可拆分的冻结
组成是：移除 candidate-only bias、typed presence mask、inactive/full-no-action direct
bypass、zero-origin residual。它们共同把 absolute biased action injection 改为
typed-mask-controlled、zero-origin、bias-free action residual。vendor GDN、注入位置、
reducer 的 end-of-bin action 映射、synthetic task、common model topology、ActionDiT
frozen diagnostic 和 reference structural bypass 均保持不变。

### 4.2 Canonical formula

令 `a0` 为固定全零 anchor。对每个 chunk/local latent，mask 必须机械生成：

```text
m_local[t] = chunk.latent_valid_mask[t]
             AND NOT chunk.latent_action_spans[t].anchor_no_action_slot
```

各 chunk 截到 `valid_latent_count` 后按 vendor frame 顺序拼接。对本 card 的 5 个
latent frame slot，correct/shuffle 的固定 mask 是
`[false, true, true, true, true]`，full no-action override 为全 false：

```text
phi(a)       = W2 * SiLU(W1 * a)
delta_l(a)   = U_l * (phi(a) - phi(a0))
h_l_plus(t)  = where(m_t,
                     h_l_post_vendor(t) + delta_l(a_t),
                     h_l_post_vendor(t))
```

硬约束：

- `W1`、`W2`、所有 `U_l` 均为 `bias=False`；
- `U_l` 在 theta0 精确全零；
- `a0` 为 immutable exact-zero buffer，不可训练；
- 因为所有映射 bias-free，`phi(a0)=0` 且 `delta_l(a0)=0` 为结构恒等式；
- mixed active/inactive call 必须先 gather active indices，只对 gathered action 执行
  conditioner/projection，再 scatter residual；inactive token 由 `where` 直接选择原始
  `h_l_post_vendor`，不能以“对全 tensor 执行后再加 masked zero”冒充 identity；
- full-call `mode=no_action` 不读取、不执行 action conditioner 或 output projection；
- 注入位置保持为每个 vendor GDN recurrence 之后、FFN 之前；
- 对 active token，规范实现可利用已证明的 `phi(a0)=0` 直接计算 `U_l*phi(a)`；
  `phi(a)-phi(a0)` 是规格恒等式，不要求额外执行 anchor forward，也不存在 detached
  anchor branch；
- active zero-valued action 的值残差仍为零，但 presence 语义不能由 action 数值推断。

### 4.3 Typed mask truth table

| token/source | `m_t` | action tensor semantics | seam behavior |
|---|---:|---|---|
| correct valid end-of-bin action | true | reducer-selected action | zero-origin residual |
| shuffled valid end-of-bin action | true | fixed permutation | zero-origin residual |
| valid action whose numeric value is zero | true | legitimate zero action | mathematical zero residual |
| bootstrap/no committed action | false | anchor only | direct identity bypass |
| padding/invalid token | false | zero padding | direct identity bypass |
| full `mode=no_action` | false for all tokens | action path unread | direct identity bypass |

`m_t` 只能来自 typed layout provenance；禁止以 `a_t != 0`、norm threshold 或其他
数值启发式生成。

mask 的规范 shape 为 `[batch=8, latent_frames=5, 1]`，随后只允许按 hidden channel
broadcast。raw padding action 不得生成额外 latent seam slot。actual vendor 输入在
拼接前已移除 latent padding，因此不得声称 vendor padding-token identity。padding 只在
一个独立 seam-helper fixture 上检查：hidden `[1,3,64]`、action `[1,3,20]`、mask
`[true,false,false]`，两个 inactive output 必须与输入 bitwise equal；该 fixture 不进入
vendor，也不构成 vendor padding 证据。`seam_disabled` 仅是同一 candidate/common-
weight state 的诊断 direct-bypass mode，不是第三个训练 arm。

### 4.4 Trainable/state contract

Candidate-only trainable parameters只能是：

```text
action_conditioner.0.weight
action_conditioner.2.weight
blocks.{0..19}.action_output_projection.weight
```

所有 candidate-only `Linear.bias` 属性必须为 `None`；任何 non-None bias Parameter、
buffer 或 state_dict key 都是 `IMPLEMENTATION_INVALID`。anchor 只能是 exact-zero
update-free buffer。reference 不得读取 action、anchor 或 mask。common parameters/
buffers 在 fresh theta0 必须逐字节相等。

## 5. A2 screen 的问题与测量

该 screen 只回答：

1. actual vendor operator 下，零锚点 action residual 是否可训练且对 action 有响应；
2. no-action/inactive token 是否在 theta0 和 final state 都保持 same-arm bitwise bypass；
3. 去掉无条件 bias shortcut 后，correct/shuffle/no-action 的分离是否达到旧硬 GO 线；
4. 若未达到，是 action-delta 失败，还是 common-mode/fusion 仍阻塞。

除旧 AV-1B 指标外必须报告：

- same-arm no-action versus seam-disabled bitwise equality 和 max-abs；
- actual-call bootstrap 在每个 block 的 post-seam 与同次 invocation 的 post-vendor
  hidden bitwise equality；独立 seam-helper padding fixture 的 inactive output identity；
- no-action candidate-only parameter gradient 必须全部为 `None`（报告 RMS 时只允许按
  预注册规则映射为 `0.0`），action-input JVP 必须 exact zero；
- per-block correct/no-action residual RMS；
- paired prediction common MSE；
- paired half-delta MSE、NMSE、energy ratio 和 alignment cosine；
- correct MSE normalized by target total energy；
- 两臂最后固定 25/50 个 train-loss 点的 mean、median、P90、min、max。

所有 correct/shuffle/no-action 指标必须在同一 trained theta、同一 frozen synthetic
batch 上成对计算。final-step primary 保留用于与 predecessor 比较；固定 window
statistics 只用于稳定性诊断。禁止 best-step、checkpoint selection、阈值回填或
post-result metric 选择。

## 6. 结果解释

- `OPERATOR_GO`：所有 validity、旧 hard GO separation 和 A2 exact-bypass 条件同时通过；
  只允许申请 AV-2 authority，不自动解锁或执行 AV-2。
- `OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED`，诊断子类
  `A2_DELTA_LIVE_COMMON_MODE_BLOCKED`：exact bypass 和 action-delta gates 通过，但旧
  shuffle/no-action hard separation 未全过且 common-mode calibration 不合格；不进入
  AV-2，不增加 step，下一候选必须显式处理 common/delta fusion。
- `OPERATOR_STOP`：valid run 同时满足预注册 strong-stop 条件；停止该 reduced A2
  hypothesis，完整架构仍为 `NOT_ASSESSED`。
- `OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED`，诊断子类 `A2_VALID_NEITHER`：valid run
  不属于以上状态；不进入 AV-2，不自动复跑。
- blocked/invalid 状态不是架构负证据，但任何再次执行都需要 fresh root、fresh nonce
  和新的明确授权。

A2 没有 review/budget-extension token。旧 review300 token 不复用、不派生、不重置。
若 A2 valid run 落入 inconclusive，最终 typed verdict 沿用 governing plan 的
`OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED`，其中 `REVIEW_EXHAUSTED` 明确继承自
predecessor 的全流程 token 状态，A2 自身不创建或消费 token。

## 6.1 与 governing plan 的关系

原计划注册的是单一 CACH-A 候选及唯一 review token。CACH-A2 是用户明确点名的新的
successor architecture registration，不是自动 candidate search、旧 CACH-A review 或
预算重跑。本轮只登记文档。未来 source authority 和 execution authority 都必须逐字
确认这一 plan deviation、旧 token 已消费且不产生新 token。即使 A2 得到
`OPERATOR_GO`，也只允许请求显式的 plan-deviation acceptance 与 AV-2 authority；
不得凭本 card 自动接入、解锁或执行 AV-2。

## 7. 明确不评估

本决策不评估真实数据、checkpoint、完整 2B、跨调用 persistent state、production
dispatcher/cache owner、registered/formal training、正式评测、deployment、AV-2 或
Global Stage 3。即使得到 `OPERATOR_GO`，也只能说明 synthetic single-batch
vendor-operator mechanism screen 通过。

## 8. 分阶段执行

1. 本轮：只冻结本 decision 与 run card。
2. Future source phase：只能排他新增 run card 固定的 bridge/config/runner/test quartet，
   静态校验后冻结为 `0444`；不得运行模型或测试。
3. Future execution phase：必须绑定 card 和 materialized quartet 的完整 SHA、精确单
   GPU index/UUID、唯一 root/nonce、预算与命令；只有届时用户明确授权后才可执行。

所有未列能力默认禁止。
