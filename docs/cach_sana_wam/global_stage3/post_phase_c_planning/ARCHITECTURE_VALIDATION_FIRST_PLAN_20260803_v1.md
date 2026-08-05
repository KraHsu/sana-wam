# CACH-SANA-WAM 架构验证优先计划（2026-08-03）

状态：`ACTIVE_PRIORITY_OVERLAY / PLAN_ONLY / EXECUTION_NOT_AUTHORIZED`

Canonical host：`H200`  
Canonical worktree：`/home/zch/workspace/sana-wam`

## 0. 文档身份与继承关系

本文是 Phase-C 之后的 additive successor。它不原位修改以下已固定字节：

| 基线 | SHA256 |
|---|---|
| `docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md` | `969d9668ffed07176844fcbd84062825f9d5f0af0325959b0068c38fe481f0dd` |
| `docs/cach_sana_wam/global_stage3/GLOBAL_STAGE2_CLOSURE_AND_STAGE3_UPDATE_FREE_ADMISSION_PLAN_20260802.md` | `93c015982b421161c137d40dee825de1f3ac5996b051a46591d9487b0eacecd8` |
| `docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/SOURCE_MANIFEST.json` | `0b88a76c5d2bf73a72a830fb58d9faac091af77e738f341f843664eaced3ffff` |
| `scripts/verify_cach_phase_c_production_path_mini.py` | `538c96496e7b72afb45995bd9ff3baa6902883ec0ee8d3d788b546e14d0a3191` |
| `docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/PHASE_C_LIGHTWEIGHT_TEST_REPORT.json` | `cb4adbe49e03f3889e209de3ea9b4d043af96d726ae4f78ce8e35554252d0fd6` |

本文只 supersede **2026-08-03 之后的研发优先级和执行顺序**，不改写历史事实、
负面结果、authority 边界或已有 artifact 的含义。未在 §10 明确列出的旧条款继续
有效。若本文以后发生实质修改，必须新建 `v2` 路径并固定新 SHA，不原位覆盖 v1。

Phase-C 历史 checkpoint 只在 caller 以 literal
`0b88a76c5d2bf73a72a830fb58d9faac091af77e738f341f843664eaced3ffff`
固定 manifest 并运行上表 verifier 时成立。本文不加入其 18-member source list。

本文本身只授权计划整理，不授权代码修改、GPU、真实数据、checkpoint、optimizer、
训练、评测、部署、capture 或正式 root。

## 1. 最高优先级约束：先验证架构效果

> **AVF-0（Architecture Validation First）**  
> 从现在起，研发资源优先用于尽快回答“这个架构是否值得继续”。凡是不直接影响
> 架构效果判断、不会造成不可逆损失、不会干扰他人作业、也不属于权限底线的
> 安全加固、检查闭环、manifest、launcher、recovery 和 admission 工作，统一后移。
> 只有在初步效果为正，或者效果不佳、需要排除实现/数据/度量错误时，才补做对应
> 的检查。不得在获得架构信号之前反复冻结清单、完善形式化证据或建设尚未使用的
> production 基础设施。

该约束的实际含义是：

1. 先做最小、可训练、能形成 action-conditioned dynamics 信号的实验；
2. 用快速否决而不是完整 admission 作为早期 gate；
3. 第一轮结果只使用 §6 的 typed screening state，不用于科学或生产结论；
4. 正信号出现后再投入 production parity、完整 C0--C8、runtime/source closure、
   正式 launcher、完整 admission-root machinery 和全规模实验；
5. 负信号只触发一次有界复核，不触发无限审计、调参或候选搜索。

## 2. 当前起点

Phase-C 已提供以下固定历史证据：

- production-shaped CPU/synthetic interface proxy；
- `REF-GDN-CORRECTED` / `CACH-A` 双臂与 20 层 typed cache；
- focused suite `11/11`、前序回归 `48/48`；
- theta0 identity、action seam gradient/JVP、causality、full/chunk orchestration、
  repeated readonly、single-CAS commit，以及 ephemeral failure injection 下的 live
  state preservation 与 post-failure recommit；
- 18 项 Phase-C source delta 与 126 项 predecessor 的零碰撞验证。

Phase-C 没有证明 vendor GDN numerical parity、exact future G3 owner/dispatcher、
transitive runtime closure、global C0--C8、C5 或任何 global gate。当前状态仍为：

```text
GATE-S0 = NOT_CLAIMED
GATE-S1 = NOT_CLAIMED
GATE-S2 = BLOCKED_NOT_CLAIMED
Global Stage 3 = NOT_AUTHORIZED
```

这些结果已经足够说明“接口和梯度路径可以工作”，但没有回答：

> 训练后，action-conditioned candidate 是否比无 action-to-video seam 的 reference
> 更好地预测由动作引起的未来变化？

因此下一步不应继续扩展 Phase-C 检查矩阵，也不应先建设完整 Global G3 admission
包；下一步应直接构造最小学习实验。

## 3. 第一验证对象

第一轮只验证一项 load-bearing hypothesis：

```text
REF-GDN-CORRECTED
        vs
CACH-A = REF-GDN-CORRECTED + action-to-video conditioning
```

唯一允许的架构差异是 action-to-video seam。以下内容不进入第一轮：

- causal softmax anchors；
- AttnRes；
- self-forcing/generated-prefix objective；
- AFCC reference、loss 或任何 AFCC state；
- rerank、best-of-N、ensemble、DAgger 或人工 recovery；
- full 2B、closed-loop success-rate 或正式 benchmark。

第一轮需要回答三个问题：

1. `Learnability`：candidate 的 action seam 能否通过训练学到非平凡信号？
2. `Action dependence`：正确 action 是否优于 shuffled/wrong action？
3. `Short dynamics`：candidate 是否在固定短 rollout 上优于 reference？

证据分为三层，禁止混称：

| 层级 | 定义 | 可支持的判定 |
|---|---|---|
| `TRANSITION_PROXY` | 简化的 pure-Torch transition，只保留接口和近似状态递推 | 只能产生 `PROXY_GO/PROXY_STOP/PROXY_INCONCLUSIVE`，不能接受或否决实际架构 |
| `REFERENCE_OPERATOR` | 与 production GDN/action/cache 使用同一数学递推和 state 语义，仅缩小 tensor shape；可以不是 fused kernel 或 public wrapper | qualification 后可以产生 single-batch `OPERATOR_GO/OPERATOR_STOP` 和 reduced screen `GO/REDUCED_ARCH_STOP` |
| `VENDOR_KERNEL` | 实际 fused CUDA/Triton numerical kernel；不自动代表 public wrapper/dispatcher/owner parity | 可替代 reference operator 做 reduced screen；完整 production parity 仍归 AH-1/Global G3 |

数值算子与集成路径是两个正交字段，run card 必须分别记录：

```text
operator_level = TRANSITION_PROXY | REFERENCE_OPERATOR | VENDOR_KERNEL
integration_path = EXPERIMENTAL_PATH | PUBLIC_PATH
```

早期允许 `EXPERIMENTAL_PATH`，因此 public dispatcher/owner 可以后移；但不能因此
降低 numerical operator 的 qualification 要求。

AV-2/AV-3/AV-4 至少使用 `REFERENCE_OPERATOR`。如果不存在可信的 unfused
reference operator，则 actual vendor kernel 必须提前到 AV-1B；不得用 proxy 结果
代替。

`correct/shuffled/no-action` 必须作用于模型真实的 action-to-video seam tensor：

- correct 与 shuffled 使用相同 timestep、noise recipe、mask、layout、padding 和尺度；
- 除该 action tensor 外，video prefix/noise/context/target bytes 全部保持相同；
- shuffled 采用预注册的 within-task、within-layout-valid-span permutation；
- 禁止额外 clean-action side channel；若早期明确使用 teacher action，只能标记
  `TEACHER_ACTION_MECHANISM_SCREEN`，不能外推为 deploy-time 效果；
- run card 必须写明 1/2/4-step 使用 teacher-prefix、action-clamped 还是
  joint-generated rollout，并把结论限制在该模式内。

## 4. 核心有效性检查与可延期检查

### 4.1 早期必须保留的核心检查

这些检查直接决定效果是否可解释，不能后移：

- reference/candidate 只有一个已知 delta，shared initialization 相同；
- action、video、target 的时间对齐正确，不使用 future clean action/video；
- candidate action seam 确实获得有限且非零梯度；
- 训练 loss、输出和梯度无 NaN/Inf；
- correct-action、shuffled-action 和 no-action 按 §3 的真实 seam 语义使用同一固定样本；
- reference/candidate 使用同一数据顺序、seed、step budget 和度量代码；
- reduced harness 保留 layout/bootstrap、future isolation、valid action exact-once、
  readonly denoise 和 paired single-commit 语义；
- 结果绑定最小 run card：代码身份、config、seed、数据切片、命令和原始指标。

### 4.2 早期不可延期的执行底线

这些不是 admission 工程，而是避免越权或不可恢复损害的最低底线：

- 未获对应授权时不使用 GPU、真实数据、checkpoint 或 optimizer；
- 不停止、修改、降优先级或争抢他人的 GPU/进程；
- 启动前做 2--5 分钟的 GPU/进程/磁盘容量快照；GPU run 还必须取得明确的独占
  reservation，并在 launch 前即时复查，发现外部进程即 fail-closed；
- 每次 run 使用新的、明确所有权的最小 immutable non-formal screen root；
- root 必须 fail-closed；成功或失败后都冻结，永不删除、修改或复用，只能新建 root；
- 不删除或修改 predecessor、formal root、screen root、receipt 或他人的产物；
- 设定 wall-time、显存和磁盘上限，资源超限记为 `HARNESS_REJECTED`；
- 明确区分 blocked/invalid、proxy、actual-operator 和 architecture verdict，采用
  §6 的状态机；
- 快速实验不得改变 global gate，不得称历史 endpoint reconstruction 或科学结果。

除了上述两组，检查默认后移。

### 4.3 正信号前默认后移

- exhaustive C0--C8 negative/failure matrix；
- exact future G3 public dispatcher/cache owner parity；
- transitive Python/native/dynamic-import/JIT runtime closure；
- 完整 source bundle、formal source manifest 和独立重建；
- DESIGN/AUTHORITY/VERIFIER/EVALUATOR/LAUNCHER 五件套；
- 超出最小 immutable screen root 的 admission schema、fsync/durable receipt、
  crash recovery；
- multi-writer fencing、power-loss recovery 和 deploy ACK；
- 完整 2B graph、真实 shape 全层 inventory 和长期资源测量；
- 多 replicate、置信区间、正式阈值 artifact 和 scientific evaluation；
- production deployment、checkpoint resume 和 controller integration。

“后移”不等于“完成”或“放行”。这些项目在正信号后、正式扩大规模或形成结论前
恢复为 blocker。

## 5. 快速验证阶梯

### AV-0：最小实验卡

目标：在写训练代码前，用一页 run card 固定第一轮问题，不建设正式 manifest。

最少字段：

- `REF-GDN-CORRECTED` 和 `CACH-A` 的唯一 delta；
- reduced topology、数据类型、batch、chunk/horizon；
- seed、固定数据切片、step 和 wall-time 上限；
- seed 0 以及条件触发的 confirmation seed 1/2、扩展 holdout 和 rollout horizon；
  这些都必须在 seed-0 结果前固定；
- optimizer/LR（只有获得训练授权后才允许填写并执行）；
- primary metric、action-shuffle metric 和快速否决阈值；
- immutable screen root、host/GPU 和授权引用。

度量字段必须在结果前同时固定：公式、分母与 epsilon、1/2/4-step 聚合权重、
shuffle seed/permutation、使用 final step 而非 best step 的策略，以及唯一一个
`review_token`。全流程中预算小幅扩展或 bug-fix rerun 二选一消费该 token，不能各用
一次。

AV-0 是几十分钟级工作，不因缺少完整 source/runtime closure 而阻塞。

### AV-1：可训练 reduced prototype

目标：新增最小可训练机制原型，并保持 Phase-C checkpoint bytes 不变。

要求：

- 只新增 additive screen module/config/test/run-card，不修改 Phase-C 18-member
  `PHASE_C_SOURCE_FILES.txt` 中的任何路径；
- 若确需改变任一固定 member，先建立独立 successor source checkpoint/revision，
  保留 v1 bytes，且不得继续声称旧 verifier 验证修改后的 canonical worktree；
- 保留同一 recurrent family、ActionDiT/action conditioner 和 action-to-video seam；
- 使用成对反事实 synthetic task：prefix/context/video noise 相同，不同 action 对应
  不同 future target；
- width、latent/spatial shape、batch 和 horizon 可缩小；
- public dispatcher、durable owner 和完整 2B wrapper 可以暂不复用；
- 保留 §4.1 的最小 temporal/commit 语义；
- 先用 synthetic single-batch 做 forward/backward/update；
- 只证明 loss 能下降、action seam 被更新且 candidate 对 action 有响应。

若只能使用 pure-Torch transition proxy，该结果只能称
`TRANSITION_PROXY`：正结果记 `PROXY_GO`，负结果记 `PROXY_STOP` 或
`PROXY_INCONCLUSIVE`，两者都不能直接接受或否决实际架构。

建议初始预算：单进程、单 seed、单 batch、最多 200 个 optimizer step；CPU 可行则
优先 CPU，否则需单独授权一张空闲 H200。不得自动扩展预算。

### AV-1B：actual-operator single-batch bridge

AV-1B 需要覆盖 actual operator、可能的 CUDA/JIT、optimizer/update 和 screen root 的
单独授权；AV-1 的授权不自动继承。

在把任何实现标为 `REFERENCE_OPERATOR` 前，必须完成轻量
`REFERENCE_OPERATOR_QUALIFICATION`：

- 固定 production formula/source SHA 到 reference decay/write/action-injection update
  的逐项映射；
- 固定 state schema、content-time、cache read/write 和 commit 顺序；
- 在固定小 tensor 上，与已固定的 vendor kernel 或独立 production oracle 对照
  forward、state update 和 backward，使用结果前固定的容差；
- 保存 qualification source SHA、输入 recipe、raw comparison 和结论。

无法完成 qualification 时只能记 `OPERATOR_UNVERIFIED`，不得产生
`OPERATOR_STOP`、`GO` 或 `REDUCED_ARCH_STOP`；可以另行授权后直接使用
`VENDOR_KERNEL`，仍无需先复用 public dispatcher。

AV-2 前必须用已 qualified 的 `REFERENCE_OPERATOR` 或 `VENDOR_KERNEL` 重复 AV-1
的 single-batch learnability、真实 seam 反事实和最小 temporal/commit 检查。只有
有效运行才能产生 `OPERATOR_GO` 或 `OPERATOR_STOP`；编译、OOM、数据或 harness
问题只能记无效运行状态。

### AV-2：tiny real-data overfit

目标：确认模型能学习真实 observation/action 对齐，而不是只拟合合成公式。

在单独获得真实数据和训练授权后：

- 必须使用已经通过 AV-1B 的 `REFERENCE_OPERATOR` 或 `VENDOR_KERNEL`；
- 固定一个很小、未选择的 episode/window 子集；
- 先确认有效 sample/mask 非空、action 有方差、shuffle 确实错配、target motion
  非退化，且 train/held-out window 不重叠；
- 记录 episode/window ID 和最小数据切片摘要；不满足时记 `DATA_INADEQUATE`；
- reference/candidate 从同一份 fresh shared initialization 开始，串行执行；不得从
  AV-1/AV-1B 的成功 state continuation；
- 最多一个 seed、一个固定 step budget；
- 报告 train/held-out loss、correct/shuffled/no-action 三组指标；
- 只保留最后一步 screen state（若 run card 授权），不选择中间 checkpoint；
- run 成功或失败都冻结整个最小 screen root，不在原 root retry。

建议初始预算：每臂不超过 1,000 step 或 60 分钟，先到者停止。

### AV-3：paired reduced-scale screen

目标：用最小成本判断 action-conditioned dynamics 是否值得继续。

- 至少使用 `REFERENCE_OPERATOR`；wrapper/cache owner 可以仍是实验路径；
- reference/candidate 从新的共同 shared initialization 开始，不继承 AV-2 overfit
  state，并使用同一数据、seed、step、optimizer 和评测窗口；
- correct/shuffled/no-action 严格采用 §3 的真实 seam 反事实语义；
- primary 为固定 1/2/4-step short-rollout video/dynamics error；
- primary 同时包含预注册的 motion/delta-sensitive 指标，避免静态背景淹没差异；
- action-shuffle delta 是必要对照；
- dream-minus-copy margin 与 action prediction 只作辅助；
- seed 0 先跑，只有达到 GO 线才补 seed 1/2；
- 不做任务、checkpoint、layer、比例或 seed 选择。

### AV-4：小规模确认

只有 AV-3 为 `GO` 才执行：

- 再补两个预先固定的 seed；
- 每个 seed 都从该 pair 的 fresh common initialization 开始；
- 扩大固定 holdout 和 rollout horizon，但仍不进入完整 2B；
- 确认正信号不是单 seed、tiny-set memorization 或 action-label shortcut；
- 形成是否值得投入 production integration 的结论。

### AH-1：正信号后的架构加固

只有 AV-4 通过后，才优先投入：

1. fused vendor GDN、正式 train/deploy wrapper 与 `REFERENCE_OPERATOR` parity；
2. future G3 dispatcher/cache owner；
3. 完整 production-path C0--C8；
4. source/runtime/native/JIT closure；
5. 正式 launcher、完整 admission-root schema 与 durable failure/recovery；
6. complete-model update-free G3；
7. registered reduced/full training 和后续 capability evaluation。

## 6. 快速判定规则

以下是 screening 默认阈值，不是科学阈值。run card 可以在结果产生前收紧或替换，
结果产生后不得修改。

判定状态按证据层级隔离：

```text
preflight/authority/environment/resource/data/implementation/numerical invalid
  -> AUTH_BLOCKED / ENV_BLOCKED / HARNESS_REJECTED / INVALID_RUN /
     DATA_INADEQUATE / IMPLEMENTATION_INVALID / NUMERICAL_INVALID /
     OPERATOR_UNVERIFIED

valid TRANSITION_PROXY
  -> PROXY_GO / PROXY_STOP / PROXY_INCONCLUSIVE

valid REFERENCE_OPERATOR or VENDOR_KERNEL single-batch
  -> OPERATOR_GO / REVIEW_ONCE / OPERATOR_STOP /
     OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED

valid AV-2 actual-operator tiny real-data screen
  -> AV2_GO / REVIEW_ONCE / REDUCED_ARCH_STOP /
     REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED

valid AV-3 actual-operator paired reduced screen
  -> GO / REVIEW_ONCE / REDUCED_ARCH_STOP /
     REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED

valid AV-4 multi-seed confirmation
  -> GO_CONFIRMED / REVIEW_ONCE / REDUCED_ARCH_STOP /
     REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED
```

任何 invalid 状态都不是架构负证据。`PROXY_STOP` 只停止 proxy 分支，实际架构仍为
`NOT_ASSESSED`。AV-1B single-batch 失败最多产生 `OPERATOR_STOP`，完整架构仍为
`NOT_ASSESSED`。只有有效 AV-2/AV-3/AV-4 actual-operator real-data screen 在消费唯一
review token 后，才能产生 `REDUCED_ARCH_STOP`；它只停止当前 reduced CACH-A 投资，
不是完整 2B 架构的科学证伪。

如果唯一 `review_token` 已在较早阶段消费并且该阶段随后通过，那么后续阶段不再
获得第二次复核。对 AV-1B，达到 stop line 记 `OPERATOR_STOP`，落在 review band
记 `OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED`，完整架构仍为 `NOT_ASSESSED`。
对 AV-2/AV-3/AV-4，达到 stop line 记 `REDUCED_ARCH_STOP`，落在 review band 记
`REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED`。这些状态均不得重跑；
invalid/blocked 仍保存对应状态，不能用“token 已耗尽”把无效运行改写成负结论。

### 6.1 AV-1/AV-2 viability

- 任一非有限 loss/output/gradient：按根因记 `NUMERICAL_INVALID` 或
  `IMPLEMENTATION_INVALID`，不得形成负结论；
- proxy 未达到 run-card-defined loss decrease/action separation：`PROXY_STOP` 或
  `PROXY_INCONCLUSIVE`；
- proxy 达线：`PROXY_GO`，只解锁 AV-1B；
- AV-1B actual operator 达到 stop line 或落在 review band 且 token 未消费：
  `REVIEW_ONCE`；token 已消费时，stop line 为 `OPERATOR_STOP`，review band 为
  `OPERATOR_INCONCLUSIVE_REVIEW_EXHAUSTED`，均不进入 AV-2；
- actual operator 达线：`OPERATOR_GO`，只解锁 AV-2；
- AV-2 数据有效且达到 run-card-defined train/held-out loss decrease、correct-vs-
  shuffled separation 和 no-action 对照阈值：`AV2_GO`，只解锁 AV-3；
- AV-2 valid result 未达到 GO line 且 token 未消费：`REVIEW_ONCE`；token 已消费
  时，stop line 为 `REDUCED_ARCH_STOP`，review band 为
  `REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED`；
- 只有 `AV2_GO` 才能进入 AV-3。

### 6.2 AV-3 seed-0 screen

默认 GO 线：

- candidate 相对 reference 的 primary short-rollout error 改善至少 5%；
- candidate 的 shuffled-action error 相对 correct-action 至少恶化 5%；
- 单步误差不比 reference 恶化超过 5%；
- 无 NaN/Inf、future-target leakage 或 action/time misalignment。

默认 REDUCED_ARCH_STOP 线：

- 在有效 actual-operator run 中消费唯一 review token 后，candidate 不优于
  reference，且 correct/shuffled action 无分离；
- candidate 明显更差，或优势只能靠选择 seed/checkpoint/window 得到；
- actual recurrent operator 上无法复现 proxy 的正信号。

上述 stop-line 条件在 token 未消费时也先记 `REVIEW_ONCE`；只有复核后或 token 已在
早期阶段消费，才记 `REDUCED_ARCH_STOP`。介于 GO/STOP 线之间同样在 token 未消费
时记 `REVIEW_ONCE`，token 已消费时记
`REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED`。唯一 token 只能用于 run card 已
固定的一次小幅预算扩展或一次 bug-fix rerun，二选一；不得自动增加候选、seed 或
训练时长。

### 6.3 AV-4 confirmation

三个固定 seed 中至少两个保持同方向，并且 aggregate 仍达到 seed-0 的预注册 GO
线，才记 `GO_CONFIRMED` 并进入 AH-1。有效结果达到预注册 stop line 时记
`REVIEW_ONCE`（token 未消费）或 `REDUCED_ARCH_STOP`（复核后/token 已消费）；
落在 review band 时记 `REVIEW_ONCE`（token 未消费）或
`REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED`（token 已消费）。该判定仍只是
“值得继续工程化”，不是 capability/scientific pass。

## 7. 效果不好时的一次有界复核

`REVIEW_ONCE` 只检查会直接伪造负结果的六项：

1. observation/action/target 时间对齐与 mask；
2. reference/candidate shared initialization、数据顺序和 step 是否相同；
3. action seam 是否真正进入 video path，梯度和参数更新是否存在；
4. loss/metric 实现、copy baseline 和 shuffled-action 对照是否正确；
5. dtype、数值范围、学习率和明显 underflow/overflow；
6. reduced proxy 是否删除了 hypothesis 所依赖的关键算子。

复核时间盒默认为半天。AV-0 发出的唯一 `review_token` 在预算扩展或 bug-fix rerun
中二选一消费；整个 AV-1--AV-4 链不能再次发放：

- 找到明确 bug：固定同一 run card 语义，产生新 code revision 后重跑一次；
- 数据信息量不足：`DATA_INADEQUATE`，不得形成架构结论；
- wiring/metric/mask 等仍无法确认正确：`IMPLEMENTATION_INVALID`，不得形成架构结论；
- proxy 有效但仍无信号：`PROXY_STOP`，实际架构保持 `NOT_ASSESSED`；
- actual operator、数据和实现均有效，且 AV-2/AV-3/AV-4 重跑达到 stop line：
  `REDUCED_ARCH_STOP`，停止当前 reduced hypothesis；落在 review band 且 token
  已消费则保留 `REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED`；
- 环境/OOM/编译/资源失败：记 `HARNESS_REJECTED`，不得冒充架构负结果。

不得因负结果启动大规模安全加固、无界调参、候选池搜索或 post-hoc 指标替换。

## 8. 最小产物与正式证据的隔离

快速 screen 每次只需保留：

- 一份 run card；
- resolved config、命令、seed、代码 HEAD 与相关 diff digest；
- 原始 loss/metric JSON 和短日志；
- §6 exact typed state、`operator_level`、`integration_path`、validity reason 和
  raw evidence；不得把 proxy/operator/invalid state 压缩成泛化 `STOP`。

screen root 必须唯一、最小、immutable，并明确标记：

```text
NON_FORMAL_ARCHITECTURE_SCREEN / IMMUTABLE / NOT_ADMISSION_EVIDENCE
```

快速 screen 的产物不得直接升级成 formal root。成功、失败或 partial root 均冻结且
不得删除、修改或复用。若信号为正，后续必须在新 authority、新 source/runtime
closure 和新 root 下做 prospective confirmation；不得称为历史 endpoint
reconstruction。

## 9. 延期工作恢复条件

| 延期项 | 何时恢复为 blocker |
|---|---|
| `REFERENCE_OPERATOR` | proxy 为正后立即在 AV-1B 恢复；AV-2 前必须通过 |
| fused vendor/public production parity | reduced actual-operator 信号确认后，在 AH-1/Global G3 前恢复 |
| exhaustive C0--C8 | 正信号确认后、complete-model G3 前 |
| source/runtime/native/JIT closure | 正信号确认后、任何 formal/admission/scientific run 前 |
| five-piece launcher 与完整 admission-root machinery | complete-model G3 或正式训练前；最小 immutable screen root 从不延期 |
| checkpoint/data 完整 provenance | 真实数据确认或正式训练前；早期只需最小 run card |
| multi-seed/statistical design | seed-0 GO 后；正式结论前必须完整 |
| failure/recovery/deploy ACK | production/deploy 路径前 |
| full 2B/resource admission | reduced-scale signal确认后 |

若早期先使用 proxy，同一数学递推的 `REFERENCE_OPERATOR` 必须在 AV-1B 恢复；
public wrapper/owner 和 fused-kernel parity 仍可留到 AH-1。

## 10. 对旧计划执行顺序的明确更新

本节中的“主计划”只指 SHA
`969d9668ffed07176844fcbd84062825f9d5f0af0325959b0068c38fe481f0dd`，
“Global 计划”只指 SHA
`93c015982b421161c137d40dee825de1f3ac5996b051a46591d9487b0eacecd8`。

| 精确旧条款 | 仅对 `NON_FORMAL_ARCHITECTURE_SCREEN` 的新状态 | 对正式阶段的状态 |
|---|---|---|
| 主计划 §0.1 items 1--10 的 admission blockers | item 1 用 minimal provenance；items 2/3/5 用 §4.1 与 AV-1B；items 4/6（AttnRes/SF）不在 CACH-A scope；item 7 deploy ACK 由 §3 offline seam 语义替代；item 8 默认不加载 checkpoint；item 9 只核对 reduced trainable inventory；item 10 用 AV-2 固定 tiny slice/data-adequacy 代替 full `DATA_AND_SCALE_DESIGN` | 对各自 component、deploy、registered/formal training 全部恢复 |
| 主计划 §0.1 item 11、§6 的 full `CANDIDATE_SPEC`/authority/root/endpoint | 用 AV-0 minimal run card、单一 delta、预先固定 seed/data slice/budget/endpoint 和单独授权代替 | 完整条款在 formal campaign 前恢复 |
| 主计划 §7.5 `G0 static` 的完整 source/runtime/input closure | 早期只要求相关代码 HEAD/diff digest、resolved config、数据切片身份、命令和最小 immutable root | Global G3/正式训练前仍须完整闭合 |
| 主计划 §7.1 “任一 C0--C8 fail 禁止训练”及 §7.5 G1--G3 | 对 AV-1--AV-4，由 §4.1 最小有效性 gate、AV-0 run card 和单独训练授权替代；未覆盖的 production checks 必须标 `NOT_ASSESSED`，不得伪称通过 | blanket gate 对 registered/formal/scientific training 继续完整有效 |
| 主计划 §8 Stage 0→4 ordering | Stage-0 完整 source closure 与 Stage-3 complete-model update-free 不再前置于 screen；分别由 AV-0 minimal provenance 和 AV-1B actual-operator bridge 取代；早期只允许非正式 reduced training | registered/formal Stage 4 仍不得绕过完整 Stage 0--3 admission |
| 主计划 §10.1 完整 artifact schema | 保留唯一 immutable/fail-closed/no-reuse root，完整 inventory、fsync、durable receipt 后移 | formal/admission/scientific run 前恢复全文 |
| 主计划 §10.2 launcher hard requirements | screen runner 只保留 exact authority check、唯一 immutable root、GPU reservation/即时进程复查、资源上限、failure trap 和 fail-closed；不先建设五件套或通用 launcher | Global G3/正式训练前恢复全文 |
| 主计划 §12 item 14 的完整 source/data/runtime identity stop | early screen 以 §4.1、§4.2 和 AV-0 minimal provenance 为满足条件；缺少这些仍立即停止 | 完整 identity 在 formal 阶段继续是 hard stop |
| Global §10、§12--§16 的 runtime/filesystem/threshold/five-piece package | 不再阻塞 reduced screen；只保留本计划的最小权限、有效性、资源和 immutable-root 底线 | 对 Global Phase D 继续完整有效 |
| Global §11 strict production-path parity | 保持未完成；actual recurrent math 在 AV-1B 使用，public wrapper/owner parity 在 AH-1 补齐 | Global G3 前必须闭合 |
| Global §19 中 graph/named-init/source/runtime/threshold 缺失的 stop | reduced graph、shared init、screen threshold 与 minimal provenance 按本文满足；complete graph/runtime 不阻塞 screen | Global Phase D 仍按原 stop rule 执行 |
| Global §19 的 dataset/checkpoint/optimizer/training stop bullet | 只有后续独立 authority 明确列出的 AV data、optimizer/update 和 final screen state 才不触发 stop；越出 run card、未授权 checkpoint、capability/scientific evaluation、deploy 仍立即停止 | Global Phase D 与 formal 阶段按原权限边界完整执行 |
| Global §20 Phase-D checklist | 不适用于 AV-1--AV-4 non-formal screen | 请求或执行 Phase D 前逐项完整有效 |
| 主计划 §16、Global 计划 §21 immediate next action | 被本文替换为：当前仅完成本计划；AV-0、AV-1、AV-1B、AV-2、AV-3、AV-4 各自执行前都必须获得覆盖其代码、operator、GPU、data、optimizer/update、diagnostic metric 的独立明确授权 | 不改变任何后续阶段的单独授权要求 |

以下内容不被 supersede：

- 不走 DAgger、人工 recovery、success-rate trick、rerank 或 best-of-N；
- treatment/control 与 AFCC 隔离；
- 所有新 run 使用唯一 immutable root，失败 fail-closed；不覆盖、删除、修改或复用
  success/failure/partial root；
- 不把 prospective run 称为历史 reconstruction；
- 不宣称已有 504-step formal 结果、正式训练或 global Stage 3；
- 不干扰他人作业，不修改冻结 predecessor/formal artifact；
- GPU、数据、checkpoint、训练、评测和 root 仍分别需要明确授权。

## 11. 下一项核心任务

当前获授权的任务到本文交付为止。下一项**需单独授权**的开发任务应是：

> **AV-0/AV-1：冻结一份 CACH-A reduced experiment card，并实现最小可训练的
> additive REF-GDN-CORRECTED/CACH-A prototype；先做 single-batch learnability 和
> correct-vs-shuffled action 检查。**

AV-1 即使只用 CPU，也包含代码修改、optimizer、parameter update 和轻量测试，必须
另行获得明确授权；本文不授权执行。该任务不应先建设完整 G3 dispatcher、runtime
closure、正式 launcher 或完整 admission-root machinery，但必须使用 §4.2 的最小
immutable screen root。若 CPU prototype 足够则先 CPU；若实际 recurrent operator
需要 CUDA，再单独申请一张空闲 H200、固定短时限的 GPU/训练授权。

AV-1 必须新增独立 screen module/config/test，不能修改 Phase-C 18-member 固定路径；
确需修改时先按 AV-1 的 successor-checkpoint 规则创建新 revision。

完成 AV-1 后：

- proxy 有学习信号：另行申请 AV-1B actual-operator bridge，不直接进入真实数据；
- AV-1B 为 `OPERATOR_GO`：再申请 AV-2 tiny real-data overfit；
- proxy 无信号：只产生 `PROXY_STOP/PROXY_INCONCLUSIVE`；
- actual operator 无信号：按唯一 review token 复核后产生 `OPERATOR_STOP` 或继续；
- 只有 AV-4 为 `GO_CONFIRMED`，才进入 AH-1 加固。
