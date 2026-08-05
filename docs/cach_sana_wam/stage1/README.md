# CACH-SANA-WAM Stage 1 实现交付

> 状态：`DRAFT / CONTRACT_IMPLEMENTED / EXECUTION_BLOCKED`
>
> 规范主机：`H200`
>
> 规范工作树：`/home/zch/workspace/sana-wam`

本目录记录
`docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md`
的 Stage 1 实现，不依赖会话记忆作为事实来源。

用户在 2026-07-31 明确授权“Stage 1 代码实现及轻量测试”。该授权只覆盖本页列出的
contract 代码和 CPU/静态轻量测试；不授权模型构建、CUDA、训练、评测、capture、
GPU reservation 或正式 run-root 创建。

## 本阶段已实现

### 1. 独立 CACH namespace 与执行边界

- `src/sana_wam/cach/authority.py`：Stage 1 capability 固定不可签发；
- `src/sana_wam/cach/config.py`：完整、严格、torch-free 的 Stage 1 config schema；
- `src/sana_wam/cach/verifier.py`：静态 config、AFCC 隔离、inventory、
  checkpoint metadata，以及 authority/source/test-report/source-bundle 内部
  pin consistency 的纯 contract verifier；caller 提供 authority SHA 时只标记
  caller pin verified，不冒充已注册 external trust anchor；
- `src/sana_wam/cach/launcher.py`：只能生成 review-only plan；
- `scripts/verify_cach_stage1.py`：只读静态报告入口；
- `Trainer`、legacy checkpoint loader、engine dispatch、policy-server builder
  与 eval wrapper 的已接线路径在数据集、模型、checkpoint glob、CUDA 或 root
  动作前 fail-closed；原样 Stage 1 YAML 还由下述 compatibility latch 提前拒绝。
  marker-free legacy train dotlist、未接线 auxiliary/AFCC probe 与未来 server
  identity handshake 仍列为 admission blocker，不能泛化宣称 repo-wide closure。

Stage 1 配置为
`configs/experiments/cach_sana_wam_stage1.yaml`。其中 source/runtime/training
坐标故意保持 `null` 或 `false`；这些值是 blocker，不是可执行默认值。
配置还保留一个明确的 `cach_stage0.compatibility_denial_latch`：它只用于让
authority-pinned、不可改写的 Stage 0 train/deploy 原始配置 guard 在导入
torch/NCCL 前拒绝 Stage 1 YAML，不表示 Stage 1 回退为 Stage 0。

### 2. Action/latent/layout contract

- `ChunkActionLayout` 固定 raw-frame、latent、action ownership、Action RoPE、
  bootstrap、partial tail、pad、proprio boundary、synthetic/production provenance
  与实例 digest；
- production builder 在 Stage 1 固定硬拒绝；外部 row-timebase verifier 与不可伪造
  trust anchor 尚未实现，不能由 caller 自报 proof 解锁；
- synthetic proof 只能通过显式 test-only API；
- episode-origin wrapper 强制 `row_start=0`，拒绝 legacy clean prefix，
  以显式 allowlist 生成 model-facing sample，完整 future `proprio_seq` 不进入模型；
- `first_frame_image[0]` 与 `video[0]` 进行 exact-content digest 绑定；per-chunk
  proprio metadata 同时绑定实际 semantic-float32 row digest；
- `num_frames=33` 明确解析为 `L=5`，而不是把 partial latent tail floor 丢弃；
- `L=8,K=3,r=8` 的 action ownership 固定为
  `[0,16)`、`[16,40)`、`[40,56)`，最后一块 action mask 为
  `16 valid + 8 pad`。

### 3. CACH-A action-to-video seam

- 公共参数名为 `action_condition`；
- 内部只映射到 vendor `use_delta_pose_additive`，不使用只影响 final layer 的
  `use_delta_actions`；
- condition width 固定为 20；
- conditioner 固定输出 `[B,K,A]` 与 boolean `[B,K]` latent mask；partial-tail
  condition pad 为 exact zero，action 输入使用固定 capacity 与 prefix mask；
- non-anchor latent 使用其 layout span 的最后一个 valid noisy command；
- latent 0 使用独立、model-owned、trainable `NO_ACTION` slot，不占 action token
  或 RoPE position；
- continuation chunk 当前只绑定 committed action prefix 的 batch/token/width/
  dtype/device 形状；prefix 的 canonical 来源、值 digest 和显式 history operator
  尚未闭合；
- vendor factory 完成全局初始化后，再逐 block 把 `delta_pose_proj` weight/bias
  恢复为 exact zero 并核验；
- Stage 1 constructor 在任何真实模型构建前固定拒绝，旧 fixed-ATC training loss
  不可继承。

当前 additive seam 位于每层 self-attention 之后。因此 block 0 当层 attention
state 不读取 action；block output、bridge 和后续 block state 可以读取 action。
本阶段不把它描述成 pre-attention conditioning。

### 4. Typed hybrid cache contract

- immutable content-time、layer registry、state manifest 与 tensor digest；
- manager 只接受与其持有的 exact `ChunkActionLayout` member 派生结果逐字段相等的
  `ContentTime`，并绑定 latent/raw/action/Action-RoPE 区间与 layout instance
  digest；
- GDN 为 recurrent full-history；
- future softmax schema 只允许 exactly one previous committed chunk；
- AttnRes 固定为 forward-local，不能进入 temporal cache；
- denoise 使用 detached private scratch，并同时检查 tensor bytes、binding 与
  opaque live-pointer identity；read view 不持有 live state；
- offline teacher-forcing contract 只允许 paired `(video, action)`、单次 `t=0`
  staging callback；每个 pair 必须使用 layout 固定的 `K` 个 latent slots 与
  `action_slot_capacity`，逐 batch 提供与 chunk 完全相同的 frame/action prefix
  masks，所有 pad slots 必须为 exact zero；frame mask tensor digest 同时进入
  source proof、staging context、paired payload digest 与 durable receipt；
- normal manager fail-closed 拒绝 `synthetic_test_only` layout，显式 synthetic
  test factory 则反向要求该 provenance，reset 不允许切换 provenance mode；
- Stage 1 只能证明 callback 的结构与事务语义，不能证明真实模型已在 `t=0`
  同时消费这对输入；
- durable receipt attestation 后才允许 CAS/live-pointer swap；
- duplicate、stale、out-of-order、reset race 和 mutation 均 fail-closed；
- self-forcing 与 deploy applied-action commit 在本阶段固定拒绝。

该模块故意没有实现 vendor `num_layers × 10 slots` codec，也没有调用模型。

### 5. 初始化、freeze、optimizer 与 checkpoint contract

- fresh CACH 禁止 `model_path`、`init_dit_from` 和初始 checkpoint；
- 只允许冻结 source-pinned VAE 与 text encoder；
- 所有 DiT/GDN/action/proprio/conditioner 参数必须进入 trainable inventory 和且仅
  一个 optimizer group；
- reference/candidate shared tensor 必须逐名比较完整 canonical metadata，包括
  shape/dtype/init digest、role、trainability、source 与 optimizer ownership；
- candidate-only operator tensor集合必须与冻结的 unique delta 完全一致；
- resume/deploy checkpoint 使用 exact key/shape/dtype/schema/revision；
- 禁止 `latest`、glob、relative/non-canonical path 和 permissive
  missing/unexpected allowlist；checkpoint basename 中的 step 必须与
  `endpoint_step` 精确相同；
- AFCC、Phase-6、action-reference 和 loss-subtraction 字段递归拒绝；兼容 schema
  中若存在 `F`，只允许 plain integer `0`。

## 轻量测试边界

本阶段允许且只允许：

1. Python AST/语法与静态 ordering 检查；
2. 标准库/metadata/layout/config 的纯 contract tests；
3. CPU tensor 上的 typed-cache contract tests；
4. `/tmp` 下 test-only durable-publisher fixture。

本阶段禁止：

- 构建 mini/full CACH 模型；
- model forward/backward、JVP 或 gradient wiring；
- CUDA/Triton；
- 真实 dataset/HDF5 扫描、VAE/text encoder/checkpoint 加载；
- 训练、评测、capture、policy server；
- 创建 `/DATA/share/sana_cach_wam_20260731/` 或任何正式实验 root。

测试命令和结果在执行后写入
`STAGE1_LIGHTWEIGHT_TEST_REPORT.json`；没有该文件或其状态不是
`passed` 时，连获授权的轻量 suite 都不得宣称通过。即使该报告为 `passed`，
也只表示上述轻量 suite 通过；开发计划的 GATE-S1 文字还包含 mini-model
contract tests，而本次授权明确禁止模型构建，所以当前 `gate_s1=not_claimed`。

## 未关闭的 admission blockers

1. **真实 row timebase/data closure**：RoboTwin timestamp/rate provenance、
   dataset/row-order manifest、split/cohort、normalization/action stats 与 seed
   尚未冻结。
2. **真实模型数值激活**：Stage 1 authority 固定拒绝 model construction；
   `flow_shift`、真实 shape/dtype、20-layer inventory 和 resource envelope 尚未核验。
3. **cache codec/integration**：typed cache 与 vendor `list[10]` 的双向 exact codec、
   slot 6 layer-type证明、逐字段 shape/dtype/device schema、训练/部署 commit 接线
   尚未实现；当前 callback attestation 也不等价于真实 paired-model consumption。
4. **partial-tail numerical behavior**：layout、reducer 与 typed-cache pair
   schema 已统一为 fixed `K`/fixed action capacity、exact prefix masks 和
   zero padding；仍未关闭的是，真实 eager/Triton GDN kernel、bridge 与 loss
   对该 mask 的数值行为，必须在 Stage 2 验证。
5. **committed-action history operator**：当前只严格检查 committed prefix 的
   shape/dtype/device；canonical 来源/value digest 与独立显式 summary operator
   尚未设计。typed cache 规定过去 action-conditioned video state应进入 GDN
   cache，但真实 codec/model consumption 未实现，不得宣称该数据流已成立。
6. **fresh pair initialization parity**：额外 conditioner 会改变普通 global RNG
   draw order；reference/candidate shared tensors 必须使用 per-name seed stream
   或构建后按 exact name/shape复制，不能只设置同一个 global seed。
7. **真实 checkpoint schema**：当前 verifier 只验证 caller-supplied metadata；
   未构建真实 2B 模型，所以真实 key allowlist/inventory 仍未生成。
8. **reference commissioning**：机器可审计的 `REF-GDN-CORRECTED` spec/config/
   authority、G0–G3 threshold 与 endpoint 均缺失。
9. **可执行五件套与入口全覆盖**：当前 launcher/verifier 仅 review-only；独立
   external trust anchor、design/authority/evaluator/launcher 闭环和 immutable
   root state machine 尚未实现。marker-free legacy train dotlist 可在
   `Trainer` 拒绝前到达 torch/NCCL，未接线 auxiliary/AFCC probe 也未形成统一
   pre-runtime guard；Stage 1 不修改冻结 AFCC source，这些路径保留为 blocker。
10. **self-forcing/deploy**：objective、stop-gradient、target identity 与
    environment-confirmed `APPLIED_ACTION_ACK` 未闭合。
11. **训练/科学结论**：没有 504-step formal 结果，没有完成正式训练，也没有任何
    capability 改善结论。

因此 Stage 1 完成只表示 contract 实现与其轻量测试可审阅，不构成 Stage 2、
训练、评测或 scientific admission。
