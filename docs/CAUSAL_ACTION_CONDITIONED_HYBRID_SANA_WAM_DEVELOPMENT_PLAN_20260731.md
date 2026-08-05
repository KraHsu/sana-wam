# Causal Action-Conditioned Hybrid SANA-WAM 开发计划（2026-07-31）

> 状态：**DRAFT v0.3 / 仅开发设计，不是实现、测试或训练授权**
>
> 目标：建立一条独立、可审计、可证伪的 prospective capability 研发线，
> 解决 SANA-WAM 在机器人操作中的 causal dynamics、action conditioning 和
> generated-prefix robustness 问题。
>
> 本文不授权安装依赖、运行测试、启动训练、评测或 capture。任何执行都必须
> 经过对应阶段的 source closure、admission 和用户单独授权。

## 0. 如何阅读本文

本文对陈述做如下分级，避免把相关工作、历史观察和新假设混为一谈：

- **FACT**：由当前仓库、冻结 receipt 或较新交接文档直接支持的事实。
- **PRIOR**：来自相关工作的设计先验；不能直接外推为 RoboTwin 结论。
- **HYPOTHESIS**：本计划要证伪的新机制假设。
- **DECISION**：本计划预注册的工程或实验选择。
- **GATE**：继续下一阶段前必须满足的条件。

证据发生冲突时，采用以下优先级：

1. `AGENTS.md`、`START_HERE_20260730.md` 中的执行与科学边界；
2. 两份 byte-identical `AGENT_HANDOFF_20260730.md` 中的冻结科学证据；
3. `docs/SANA_WAM_PROJECT_RETROSPECTIVE.md` 顶部 correction 与 Phase 10；
4. retrospective 的旧 TL;DR、旧 §2–5，仅作为历史过程；
5. `deep-research-report.md` 和本节固定的一手论文，只作为相关工作先验。

本文中的设计数字不因写入文档而成为实验授权。任何 candidate、数据、seed、训练
预算或 endpoint 的变更都必须形成新版本并重新做 source closure。

### 0.1 当前决策与 P0 阻塞摘要

本计划选择一条与 AFCC formal 完全隔离的、from-step-0 prospective capability
研发线。不是在旧 AR checkpoint 上继续 fine-tune，也不是恢复 DAgger、rerank、
best-of-N 或 approximate replay。

以下事项必须在其对应 component/campaign 进入实现前闭合；公共项先阻塞
`REF-GDN-CORRECTED`，AttnRes/self-forcing 专属项不反向阻塞较早 candidate：

1. H200 governance/source 文档闭包；
2. 用 `ChunkActionLayout` 替代 fixed-ATC，并证明 bootstrap/tail/Action RoPE；
3. 废止漂移的 `ar_observed_prefix_chunks`，固定 episode bootstrap；
4. 将 AttnRes 一手公式逐 sublayer 映射到当前 streaming block loop；
5. 固定 GDN/softmax/AttnRes 三类 state 与 paired video/action commit；
6. 定义 generated-prefix 的有效 target；未闭合前 `CACH-SF` 保持阻塞；
7. 建立 environment-confirmed applied-action acknowledgement；
8. 初训拒绝 pretrained video-DiT，并对 resume/deploy 使用精确 key allowlist；
9. 证明完整 DiT/action/router trainable inventory，只冻结 VAE/text encoder；
10. 固定 `DATA_AND_SCALE_DESIGN`；不得把约 50 demos 的低数据失败解释为架构结论；
11. 为每个 campaign 单独冻结唯一 `CANDIDATE_SPEC`、authority、root 和 endpoint。

这些是对应阶段的 admission blockers，不是待训练时“边跑边调”的事项。

## 1. Source of truth

### 1.1 仓库与源码身份

| 对象 | 身份 |
|---|---|
| H200 主仓 | `/home/zch/workspace/sana-wam` |
| 主仓固定提交 | `605f1c134b4c983ff80f8489c4bc8847036329e2` |
| Sana 子模块固定提交 | `16b9cec673e3335724ba2d8db25de7f9ed229292` |
| AFCC 研究快照固定提交 | `9586486f2a9f5172d57b325e32093a3e018d34c0` |

当前 H200 工作树不是 clean release。开始实现前必须重新记录：

- 主仓 `git status --porcelain=v1 -z`；
- Sana 子模块状态；
- 本计划涉及的每个源码文件 SHA256；
- 新 source bundle 的完整 inventory。

不得用 checkout、reset、clean、stash 或远端内容覆盖当前工作树。

### 1.2 文档证据身份

| 文档 | SHA256 | 用途 |
|---|---|---|
| `START_HERE_20260730.md` | `592d064713b1fff8b44e96f677ce019dcb94415c0b49e68b95c6ecae58573417` | 迁移入口、首轮边界和训练准入 |
| `AGENTS.md` | `effe88203589ad59c0b6178e74674490878a9ff6190cbf1e4a036924721147dd` | 当前 agent 执行规则 |
| `deep-research-report.md` | `4bd78d3c37970ed4c3fc9c05a82faef44fbeac079a6541a694fa496e9d858010` | SANA-Video、SANA-WM、SANA-Video2、Kimi K3 相关工作先验 |
| `docs/SANA_WAM_PROJECT_RETROSPECTIVE.md` | `c14a28633dddaab60aecd8b7b3d5701317d5f7f4f355a4e303920340ac3b5202` | capability 历史、seed 修正、Phase 10 负结果 |
| `docs/SANA_DREAM_ARCHITECTURE_PLAN.md` | `6c4ea1ef38752e682a2726de97f0a4258980690297be997b226004496947a424` | 既有 dream/hybrid 尝试与执行日志 |
| `docs/agent_handoff/AGENT_HANDOFF_20260730.md` | `76f251fde90cde1a2b970f6360d95f4dc51074f41579baebe130a283a4ab7920` | 最新科学边界、AFCC 结果与禁止事项 |
| `../sana-afcc-handoff/AGENT_HANDOFF_20260730.md` | `76f251fde90cde1a2b970f6360d95f4dc51074f41579baebe130a283a4ab7920` | 外置 AFCC 研究树入口；已核对与仓内 handoff byte-identical |

上述 hash 只固定本计划起草时读取的内容。后续若文档变化，必须更新本节并说明
变化如何影响计划；不能依赖聊天记忆。

### 1.3 当前关键实现身份

以下 SHA256 来自 2026-07-31 的 H200 只读检查：

| 文件 | SHA256 |
|---|---|
| `src/sana_wam/model/gdn_ar.py` | `15a75585a0a4e33b7b83c65dd5d40333c1d5ad05c1862b21412ce302dbadd08c` |
| `src/sana_wam/deploy/gdn_ar_engine.py` | `886815fc78a433f8813d159ae6acc59f03866c383c17d4a0cdb861952e4c539d` |
| `src/sana_wam/model/video_backbone/sana/pipeline_builder.py` | `905cdf95239206b2fef8ff484d8cbe4de551735af56cf970f0cad8540b35a981` |
| `third_party/Sana/diffusion/model/nets/sana_gdn_blocks.py` | `ef15578c63f9815d670dc3ef4cf2876c4f236754bfdfb0d2070da387048a495e` |
| `third_party/Sana/diffusion/model/ops/fused_gdn.py` | `863cbbb601ddebc5666179e2b209b777d5dc213c94a1450cde695a33b0a8aded` |
| `third_party/Sana/diffusion/model/nets/sana_multi_scale_video_camctrl.py` | `bfbd72dfb22a44e2843f1985049ecfbb1f887f5e5f5ed852905c4670e2029fbe` |
| `third_party/Sana/diffusion/model/nets/sana_multi_scale_video.py` | `0e6e605ae7ceea2961626db7d12fb202e414502338372679195b06e0b199dc27` |
| `configs/train_sana_wm_gdn_ar.yaml` | `9311107bb5f3593853bd13fc40116312b4dbf17f5540ba758b733a474865f406` |
| `configs/train_gdn_ar.yaml` | `9804f0368e5e3d6dcdb55601c37dc2ae9f80c4454c1bf5dea380329d6231efe5` |
| `src/sana_wam/dataloader/robotwin_dataset.py` | `a7db73364f260b693e376f16783576d383ad88c3ee9f8258135acd91e81c1445` |
| `src/sana_wam/model/video_backbone/sana/adapter.py` | `6ef2b6c21119b1357c8cab938e0351df428a71992ea0e92f25460b1b0a05d1cf` |
| `configs/deploy_gdn_ar.yaml` | `fe7341ce5d7d35c6a9b676af78c59cb4d096de2bb92ce3e393fec44396f62aec` |
| `src/sana_wam/deploy/policy.py` | `f73405f54cd750fd8798995f37fe38b33dc45345f79283ca161c9b58c06c454d` |
| `src/sana_wam/model/base.py` | `9754d27e7916c941293174afece950606d01cf5f97c2826d6310db1c5a398ebe` |
| `src/sana_wam/train/trainer.py` | `715236acec1269f1f672aafacc99bf37afcf4ca6eadca5a6ca941e85821f3496` |
| `third_party/Sana/diffusion/scheduler/self_forcing_flow_euler_sampler.py` | `8c246685cb2a7e9cf5c8954afdc7b39973876fe403a662929764788406bec023` |
| `third_party/Sana/diffusion/model/nets/sana_gdn_camctrl_blocks.py` | `6e6859c0e35a810538f2c046a285e88d06978c56df17451b62a508ec56154de7` |
| `third_party/Sana/diffusion/model/ops/fused_streaming.py` | `4775d1ea1101d7075d087428cf2f8bb2643a7de922fd241e15b65e3c4f4ea206` |

这些 SHA 不是新实验授权，只用于防止计划与实现版本漂移。

### 1.4 一手相关工作入口（实现前仍需 pin）

本计划额外复核以下一手来源，避免从名称或二手摘要猜实现：

| 来源 | 当前证据状态/用途 |
|---|---|
| [SANA-Video 2.0](https://arxiv.org/abs/2607.21553v1) | arXiv `2607.21553v1`；3:1 hybrid、`S=8`、shared-query AttnRes 和 from-scratch 先验 |
| [Attention Residuals 官方仓库](https://github.com/MoonshotAI/Attention-Residuals) | mutable 入口；Block AttnRes 的 RMSNorm、depth-softmax、completed-block/partial-block 参考伪代码 |
| [Attention Residuals](https://arxiv.org/abs/2603.15031v1) | arXiv `2603.15031v1`；Full/Block AttnRes 原始定义；它是语言模型先验，不是 RoboTwin 结论 |

SANA-Video 2.0 v1 给出了架构和实验口径，但当前主仓没有作者发布的、与本项目
20-layer streaming video loop 可直接对照的实现。实现 AttnRes 前必须把采用的公式、
sublayer boundary、normalization、初始化和 source version 写进独立 design；不得
仅凭图示或本文摘要手写“相似实现”。

`ATTNRES_DESIGN` 的 G0 还必须固定：官方仓库 commit SHA、所有使用文件 SHA256、
两篇论文 PDF SHA256 和提取公式的位置。mutable GitHub URL 本身不构成 source
closure；未完成这些 pin 时保持 P0 blocker。

### 1.5 当前 H200 文档闭包缺口

2026-07-31 首次只读核对时，H200 主仓缺少根目录的
`START_HERE_20260730.md`、`AGENTS.md` 和 `deep-research-report.md`。本轮文档交付
已将 `deep-research-report.md` 与本文同步到 H200；当前剩余缺口是前两个执行治理
文件。治理文件是否同步必须由仓库维护者单独确认。治理入口未闭合前，任何
launcher admission 均阻塞。

## 2. 当前问题定义

### 2.1 已确认事实

**FACT-1：历史绝对成功率存在 deploy seed 污染。**

- seed-fixed non-AR SANA-MoT 当前只能表述为约 `26–33%`；
- seed-fixed AR `low_noise` 当前只能表述为约 `9–10%`；
- retrospective 明确要求把全部绝对成功率视为 provisional；
- 历史 `37–44%`、AR `25/100` 或 `30%` 不得作为当前干净基线。

证据：`docs/SANA_WAM_PROJECT_RETROSPECTIVE.md:10-31,230-232`。

**FACT-2：当前存在三类结构性故障证据。**

1. linear/RoPE numerator-denominator 不一致和巨大 residual；
2. full-sequence 训练可见未来、chunk inference 不可见未来的 operator mismatch；
3. chunkwise operator 修复后，多步 rollout 仍随步数恶化，指向 off-manifold。

证据：`docs/agent_handoff/AGENT_HANDOFF_20260730.md:77-87`。

**FACT-3：AFCC/teacher-directed video 目标可能伤害 action。**

- 固定 cohort endpoint audit 支持 objective conflict；
- 真实共同 theta0 的局部 AFCC direction 同时 teacher-directed 和 action-harmful；
- S-generating component 是主要来源，Q 不是必要 harm source；
- 这些是 frozen/local 机制证据，不是 closed-loop 或 504-step path 结果。

证据：`docs/agent_handoff/AGENT_HANDOFF_20260730.md:89-110,124-131`。

**FACT-4：当前 GDN-AR 是 teacher-forced。**

`src/sana_wam/model/gdn_ar.py:185-217,437-453` 在训练时把 ground-truth clean
chunk 写入下一步 cache。部署则消费机器人真实观测和模型当前生成结果。当前没有
self-forcing distillation。

**FACT-5：当前 action 读取 video，但 action 不驱动当前 video transition。**

`src/sana_wam/model/gdn_ar.py:142-169` 先运行 video backbone，再由 action head
读取 bridge features。当前 candidate action 没有作为 video dynamics 的显式因果条件。

**FACT-6：当前存在两种不同的“hybrid”，不能混称。**

- formal AFCC 配置是 `linear_relu`，每个 block 并行叠加局部
  `window_flash`；它不是 periodic softmax anchor；
- pretrained SANA-WM GDN preset 是 20 层中的 15 GDN + 5 softmax，
  softmax 位于 `{3,7,11,15,19}`；
- from-scratch `train_gdn_ar.yaml` 的 `camctrl_layers_num=0`，实际是全 GDN。

证据：

- `src/sana_wam/model/video_backbone/sana/pipeline_builder.py:471-569`；
- `third_party/Sana/diffusion/model/nets/sana_multi_scale_video_camctrl.py:594-635,839-895`。

**FACT-7：当前没有 AttnRes。**

当前 `src/`、`configs/` 和相关 vendored Sana 模型中没有 Block AttnRes、
completed-block summaries、partial-sum reset 或 depth router。

**FACT-8：`ar_observed_prefix_chunks` 存在契约漂移。**

`src/sana_wam/model/gdn_ar.py:54-59,199-205` 声称 leading observed prefix 只
ingest、不监督；实际训练循环 `:317-435` 从 chunk 0 开始监督所有 chunk。
实现前必须先确定预期语义并建立回归测试。

**FACT-9：当前 fixed-ATC 不能精确表达 causal-VAE 时间轴。**

训练在 `src/sana_wam/model/gdn_ar.py:230-248` 使用
`total_chunks=T//K`、`atc=Ta//total_chunks`，随后按 `c*atc:(c+1)*atc`
切 action；部署在 `src/sana_wam/deploy/gdn_ar_engine.py:120-139,183-218`
固定同一个 `_action_tokens_per_chunk` 和 RoPE offset。

但 RoboTwin 数据的 video 从 raw frame 0 开始，action token 0 表示“进入 raw
frame 1 的动作”；causal VAE 的 latent 0 又是只依赖 frame 0 的 anchor。因而：

- latent 0 不消费先行动作区间；
- bootstrap chunk 和 continuation chunk 的 action 数不同；
- `Ta // floor(T/K)` 会掩盖 remainder、partial tail 和首 chunk 特例；
- 仅让 train/deploy 使用同一个近似值，不等于物理时间对齐。

证据位置：

- `src/sana_wam/dataloader/robotwin_dataset.py:447-452,1086-1104,1165-1171`；
- `src/sana_wam/model/video_backbone/sana/adapter.py:83-96`；
- `src/sana_wam/model/gdn_ar.py:230-248,325,376,423-427`；
- `src/sana_wam/deploy/gdn_ar_engine.py:120-139,183-218`。

这是 action-conditioned video、Action RoPE 和 self-forcing 共用的 P0 阻塞，不能
通过改一个常数解决。

### 2.2 已过时或不得提升为结论的表述

以下内容只能作为历史记录：

- “当前 SANA 是 37–44% 或 44%”；
- “AR 当前是 25–33%”；
- “Wan 96.8% 已确认是完全相同且无 deploy bug 的精确 comparator”；
- “linear attention/camera pretraining 已被最终证明是唯一 binding bottleneck”；
- “domain pretraining、softmax graft、DAgger 仍是未测试方向”；
- “open-loop MAE、IDM decodability 或单步 dream 足以放行 closed-loop”。

retrospective 的 Phase 10 已将主要问题更新为 covariate shift、fragile
fine-tune basin 和部署同构性；较早章节中的旧数字与旧结论不得覆盖该修正。

## 3. 从相关工作获得的先验

### 3.1 可迁移先验

**PRIOR-1：高效 recurrent/linear 主路径需要稀疏全局刷新。**

SANA-WM 使用 frame-wise GDN + periodic softmax；SANA-Video2 使用 mostly-linear
+ softmax anchors。它们提示固定大小 recurrent state 可能需要周期性精确交互，但
不能证明 `3:1` 对 RoboTwin 最优。

证据：`deep-research-report.md:46,58-60,94-97`。

**PRIOR-2：softmax anchor 的信息可能需要沿深度复用。**

SANA-Video2 和 Kimi K3 的 Block AttnRes 用 completed-block summary 在 partial sum
重置时恢复深度信息。该机制只在 anchor 本身有效后才有意义。

证据：`deep-research-report.md:62,78,101-105`。

**PRIOR-3：world-model memory 需要 decay/write，而不是永久同权累积。**

SANA-WM 的 frame-wise GDN 和 Kimi K3 的受限 decay/write gate 提供了参数化先验；
K3 的 token objective、NoPE MLA、MoE 和 serving cache 不能直接搬到视频 diffusion。

证据：`deep-research-report.md:46,72-80,103-107`。

**PRIOR-4：generated-prefix training 是处理 exposure bias 的直接方法。**

SANA-Video 和 SANA-WM 使用 self-forcing/chunk-causal training 缩小
teacher-prefix 与 model-prefix 的分布差异。

证据：`deep-research-report.md:36,54`。

**PRIOR-5：控制信号和内容监督应有清晰归属。**

SANA-WM 将 scene-static caption 与 metric pose condition 分离，避免文本泄漏 motion
监督。对机器人任务，可迁移的是“目标/内容”和“动作/动力学”分通道，而不是照搬
相机 Plücker ray。

证据：`deep-research-report.md:48,52,119`。

**PRIOR-6：完整 hybrid 应从 step 0 联合训练。**

SANA-Video2 强调 complete hybrid from-scratch training。该先验与本项目
“AR fine-tune 会离开 fragile deploy-competent basin”的历史观察相容，但仍需在
RoboTwin 上验证。

证据：

- `deep-research-report.md:64`；
- `docs/SANA_WAM_PROJECT_RETROSPECTIVE.md:296-311`。

### 3.2 明确不直接迁移

本计划不直接采用：

- Kimi K3 的 MoE、MLA/NoPE、reasoning effort、prefix cache 或 fleet scheduling；
- SANA-WM 的 camera/Plücker 参数作为机器人 action conditioning 的替代品；
- SANA-WM second-stage visual refiner；
- SANA-Video2 的 DPO/ReFL aesthetic preference 路线；
- 未经验证的固定 `3:1` 最优结论；
- explicit 3D memory 的虚构实现。相关工作只把它列为缺口。

## 4. 研发目标与非目标

### 4.1 Primary capability objective

开发一个从 step 0 联合训练的 causal world-action model，使其：

1. video transition 显式受已执行 action history 和当前 candidate action 条件化；
2. 训练和部署使用同一 chunk-causal 可见性与 cache 时间语义；
3. 对 generated-prefix drift 比 teacher-forced reference 更稳定；
4. 在 randomized/长时 closed-loop 上优于架构匹配 reference；
5. 在 clean closed-loop 上满足预注册的非劣条件；
6. 不产生固定 cohort action-harm veto。

### 4.2 非目标

本计划不以以下内容作为成功：

- 单独提高 dream PSNR、open-loop MAE、IDM decodability 或 effective rank；
- 对某个任务、seed、checkpoint、layer、anchor ratio 做事后选择；
- rerank、best-of-N、temporal ensemble 或 receding-horizon trick；
- DAgger、人工 recovery 或 on-policy expert relabel；
- approximate replay、历史 endpoint reconstruction 或 loss subtraction；
- 覆盖、删除或复用已有失败 root；
- 把 prospective candidate 称为历史 endpoint；
- 把 AFCC admission/smoke 当 capability 结果。

## 5. 新架构：CACH-SANA-WAM v0

内部简称：

```text
CACH = Causal Action-Conditioned Hybrid
```

### 5.0 Complete-hybrid initialization contract

本文的 “from scratch/from step 0” 含义固定为：

- LTX2 causal VAE 与 text encoder 是 source-pinned、冻结的外部编码器；
- video DiT 的 GDN/softmax blocks、AttnRes、action backbone、proprio encoder 和
  action conditioner 由注册 seed 随机初始化，并从 optimization step 0 联合训练；
- 初训 `init_dit_from` 必须为 `null`，不得加载 pretrained SANA-WM video-DiT；
- 每个 candidate/reference pair 对所有同名同形 shared tensors 使用相同初值，并
  保存逐 tensor digest；operator-specific tensors 按各自 design 初始化；
- resume 只允许同一 candidate revision 的 exact-schema checkpoint。

pretrained SANA-WM 是 15 GDN + 5 softmax/CamCtrl，并不与 v0 的 corrected all-GDN
reference 或 `[7,15]` candidate 同构。加载它再只训练 adapter 是另一条
pretrained-adaptation campaign，不得混入 v0。这里的 “scratch” 不表示重训 VAE
或 text encoder。

现有配置中的 `/DATA/share/SANA-WM_streaming/ltx2_causal_vae` 和
`/DATA/share/gemma-2-2b-it` 只是历史路径。G0 必须重新固定 realpath、完整
inventory/hash、LTX2 `temporal_compression=8` 与 causal-first-frame contract；
缺失时阻塞，不得静默换成 Wan VAE 或另一个 text encoder。

完整 2B random-init hybrid 也需要独立 `DATA_AND_SCALE_DESIGN`。现有配置注释明确
记录过“2B GDN DiT random-init on ~50 demos”的 undertraining 风险；这不能被
SANA-Video 2.0 的大规模 from-scratch 先验自动消除。G0 必须固定数据 source、
规模、去重/holdout、task weighting、curriculum 和可用性。迁移资料不含大规模
pretraining corpus；在数据闭包前，G4 只能是 reduced mechanism campaign，G5
full-capability training 阻塞。简单扩大 multi-task 数据也不能单独作为机制解释。

### 5.1 总体数据流

```text
scene / goal context
        |
        v
past observed video ----> causal hybrid video state ----> predicted video chunk
        ^                         |                              |
        |                         v                              |
executed action history --> action-conditioned adapters <-------+
                                  ^
                                  |
                         current noisy action candidate
                                  |
                                  v
                           predicted action chunk
```

关键点：

- video 和 action 在当前 diffusion/flow step 共同演化；
- 当前 video 只能读取过去已执行 action 和当前 noisy action candidate；
- 当前 action 可以读取 causal video features；
- 不允许读取 future clean video、future clean action 或部署时不存在的 clean duplicate；
- 所有跨 chunk memory 都必须有明确的 content-time，而不是仅靠循环 index 推定。

### 5.2 Causal visibility contract

对正在预测的 chunk `c`：

| 信息 | video 可见 | action 可见 |
|---|---:|---:|
| `< c` 的真实观测 video | 是 | 是 |
| `< c` 的已执行 action | 是 | 是 |
| `c` 的 noisy video candidate | 是 | 是 |
| `c` 的 noisy action candidate | 是 | 是 |
| `c=0` 的真实 bootstrap latent 0 | 是 | 是 |
| `c` 的其余 clean target video | 否 | 否 |
| `c` 的 clean target action | 否 | 否 |
| `> c` 的任何 video/action | 否 | 否 |

bootstrap latent 0 是 observed condition，不属于被预测的 clean target。除此特例外，
当前 chunk 的 clean video/action 一律不可见。

**GATE-C0：** 对 future video/action 做任意扰动，chunk `<=c` 的 deterministic
mini-model 输出和梯度必须保持不变。CPU reference 路径要求精确相等；GPU kernel
只有在 admission 注册并证明必要时才可使用非零容差。

这里的 future 明确定义为 chunk `>c`，以及 chunk `c` 的 clean target。chunk `c`
内部的 noisy candidate tokens 允许按注册的 chunk-bidirectional operator 共同交互，
不把同 chunk 后续 token 误判为泄漏。若要 frame-causal operator，必须另立
candidate/design，不能暗中收紧 C0。

#### 5.2.1 Episode bootstrap，不再复用漂移字段

v0 固定：

```yaml
episode_bootstrap: first_frame_pinned
observed_prefix_chunks: 0
```

即 chunk 0 在 empty cache 下固定真实 latent 0，并预测其余 latent/action；这与当前
即时控制部署语义一致。旧 `ar_observed_prefix_chunks` 在新 variant 中必须被拒绝，
不能继续“保存但不消费”。

dataset row 中现有 `num_clean_prefix_latent`、`num_clean_prefix_actions` 也不得
悄悄改变该语义：v0 schema 要求其为零或显式丢弃并记录；任何非零值若会进入
model/deploy visibility，立即 fail-closed。

若未来选择 `observed_prefix_chunks: 1`，其含义必须是先 ingest 完整 chunk 0、不对
其监督，并从 chunk 1 开始首次预测；这会引入整 chunk 启动延迟，属于新的行为和
candidate revision，不能与 `first_frame_pinned` 混称。

### 5.3 Frame-wise gated recurrent backbone

**DECISION-A1：** 复用现有 frame-wise GDN/delta-write operator 作为高效主路径，
不重新发明 recurrent kernel。

必须增加的观测：

- 每层、每 chunk 的 decay 分布；
- write/beta gate 分布；
- recurrent state norm、更新 norm 和有限性；
- past-state contribution 随 horizon 的变化；
- object/contact token 的 persistence proxy；
- cache 的 content-time、shape、dtype 和设备 identity。

任何 NaN/Inf、无界 state growth 或全部 gate 饱和都 fail-closed。

### 5.4 Periodic causal softmax anchors

**HYPOTHESIS-H1：** 少量 causal softmax anchors 能补充 recurrent state 的精细
时空 correspondence，但不会重现 dense/segmented graft 的 future shortcut。

**DECISION-A2：**

- v0 使用 20 层；
- 首次 proxy 固定使用 zero-based anchor indices `[7, 15]`；
- anchor 使用严格 chunk-causal mask；
- 不允许 noisy/clean duplicate 跨域混合；
- 不运行 layer placement 或 ratio winner sweep；
- anchor 的启用态按独立 from-scratch operator delta 处理，不伪称与被替换的 GDN
  block function-preserving；disabled selector/bypass 必须与原 block 精确一致。

`[7,15]` 是低剂量、固定 block-span 先验，不是声称最优。

本 v0 不授权在失败后自动升级到 `{3,7,11,15,19}`。若 `[7,15]` 通过因果与数值
gate、但未通过预注册的 dynamics gate，campaign 即停止。任何 5-anchor、3:1 或
其他 placement 都是新的 candidate revision，必须有新的 `CANDIDATE_SPEC`、root、
计划 SHA 和用户授权；不得与本 candidate 合并挑 winner。

### 5.5 Shared-projection Block AttnRes

**HYPOTHESIS-H2：** 当 causal softmax anchor 已产生有效刷新时，Block AttnRes 能让
该信息穿过后续 GDN 层，而不是在深度上被稀释。

**DECISION-A3：**

- v0 block span 固定为 8；
- 使用 shared projection across depth；
- source 仅包含已完成的 causal block summaries 和当前 causal partial sum；
- 不增加独立 timestep-conditioned router offset；
- disabled router/bypass 必须是 identity；enabled router 的初始化由
  `ATTNRES_DESIGN` 按一手公式固定，不预设“零初始化即 identity”；
- `CACH-R` 的唯一 reference 使用 parameter-matched inert sham；不得再增加第三成员；
- AttnRes 只能在 anchor-only gate 通过后实现/训练。

这里的 `S=8` 是 SANA-Video 2.0 的工程默认值，不是该论文 sweep 证明的唯一最优值。
实现前的 `ATTNRES_DESIGN` 还必须逐项固定：

- attention 和 FFN 是否使用各自独立 router；
- source 的 RMSNorm、depth-softmax 轴和数值 dtype；
- block boundary 前后 completed summary/partial sum 的更新顺序；
- token embedding 是否作为第一个 completed source；
- shared projection 的初始化及 no-op 构造；
- 每次 temporal `forward_chunk` 是否重建 depth state。
- 20 层在 `S=8` 下最后 4-layer partial block 的 flush、read 和 reset 顺序。

在一手公式与本项目 block loop 的逐行映射完成前，AttnRes 状态为 **P0
implementation blocker**。AttnRes depth state 只在一次 forward 内存在，绝不能
混入 temporal GDN/KV cache。

单独提高 internal rank 而未改善 dynamics/rollout proxy，不构成通过。

#### 5.5.1 Hybrid temporal cache 的准确语义

当前 streaming softmax slot 在 commit 时读取旧 K/V 后覆盖为当前 chunk，因此并非
“全历史 softmax cache”；GDN state 才是 recurrent full-history summary。v0 为减少
变量，明确采用：

```text
GDN layers: recurrent full-history state
softmax anchors: current noisy chunk + exactly one previous committed chunk
AttnRes: current forward 内的 depth state，不跨 chunk
```

不得把它描述为 full-history hybrid attention。sink token 或 bounded multi-chunk
softmax window 属于新的 cache candidate。

共同不变量：

- cache schema 固定为 `num_layers × 10 slots`，并验证 slot 6 的 layer type；
- `start_f/end_f` 为单调、绝对、与 patch/chunk 对齐的 content-time；
- read-only denoise 不得改变任何 slot；
- 每个 committed chunk 恰好更新一次；
- reset 后 temporal slots、action cursor 和 depth state 全部归零。

### 5.6 双速率 action conditioning

**HYPOTHESIS-H3：** video dynamics 显式读取 action，比“video 先独立生成、action
只读 video bridge”更适合学习 manipulation transition。

conditioning 分成两种时间尺度：

1. **chunk-rate condition**
   - 当前 proprio/机器人状态；
   - 已执行 action history 的 causal summary；
   - 结构化 task/goal identity；
   - 与 latent chunk content-time 对齐。
2. **action-token-rate condition**
   - 当前 diffusion step 的 noisy action candidate；
   - 对齐到 raw-frame/action-token 时间；
   - 通过 zero-init action-to-video adapter 注入 GDN update 和 softmax anchor。

#### 5.6.1 `ChunkActionLayout` 精确时间契约

不得继续用固定 `atc` 作为 action/video 对齐的 source of truth。令：

- `K`：一个 AR chunk 中的 latent frame 数；
- `tc`：causal VAE temporal compression；
- `vs`：dataset video stride；
- `r`：每个非 anchor latent 对应的 action steps。

row timestamp/action-rate manifest 是 source of truth。只有 verifier 证明“一 raw
video frame 对应一 action step、无 dropped/duplicated timestamp”后，才允许化简
为 `r = tc * vs`。当前 RoboTwin dataset 的 indexing 满足该代码约定，但每个实际
input source 仍需重新验证。

latent 0 是 causal-VAE anchor，不消费动作。完整 chunk
`c = [cK, (c+1)K)` 的 action-ownership interval 固定为：

```text
[max(0, (cK - 1) * r), ((c + 1)K - 1) * r)
```

在上述等频条件成立时，chunk 0 有 `(K-1)r` 个动作，后续完整 chunk 各有 `Kr`
个动作。以当前
`K=3, tc=8, vs=1` 的 SANA-WM 几何为例，首 chunk 是 16 个动作，后续完整 chunk
各 24 个；不是统一的 22、24 或 40。该例仅用于验证公式，最终数字必须从 resolved
config 和每个 row 的时间戳重算。

新增不可变的 `ChunkActionLayout`，对每个 row/chunk 产生：

- latent `[start,end)`；
- action-ownership raw-frame interval；
- VAE receptive-field metadata（若可从 pinned VAE 精确证明）；
- action `[start,end)`；
- 每个 latent bin 的 action span；
- 累积 Action RoPE offset；
- bootstrap latent 0 的 no-action slot；
- partial-tail disposition 和全覆盖证明。

共同算法/schema/VAE 时间约定固定为一个 `LAYOUT_SPEC_SHA`；每个训练 row 或部署
chunk 再产生自己的 `LAYOUT_INSTANCE_DIGEST`。train、deploy、self-forcing 和
verifier 必须使用相同 spec，并逐实例核对 digest；动态部署实例不能伪称共享同一个
静态 artifact。

上述 interval 是动作归属，不自动等于 causal VAE 完整卷积 receptive field。
禁止 floor division、静默丢 remainder、把 partial tail 当 full chunk，或让 Action
RoPE 继续依赖 `c*atc`。所有非 pad action 必须被覆盖恰好一次；所有 pad action
必须显式标记，overlap/gap 都 fail-closed。

video conditioner 需要将每个 latent bin 的 action span 映射成 `[B,K,A]`。v0
采用确定性的 end-of-bin command；对 EEF target/rot6d 不做数值均值池化。若要
learned reducer、积分轨迹或其他聚合，它是独立 candidate delta，不能混入 v0。

禁止：

- 把 future clean action 作为 video 输入；
- 把 camera Plücker ray 当作机器人 action 的同义替代；
- 用 action 文本描述泄漏逐帧轨迹；
- 只改变 action head 而仍让 video transition 与 action 无关。

#### 5.6.2 最小 action-to-video seam

主仓新增显式参数 `action_condition`，不要在公共接口中把机器人 action 命名为
camera pose。adapter 内部可以映射到 vendored `use_delta_pose_additive` seam，
因为该 seam 在每个 block 中有零初始化 projection。不得使用
`use_delta_actions`：当前实现只在全部 blocks/bridge capture 之后修改 final-layer
timestep，不能让 ActionDiT bridge 或大部分 cache state 读取 action。

`ACTION_CONDITION_DESIGN` 必须固定 `delta_pose_additive_dim`、bootstrap no-action
slot、partial-tail pad mask 和 dtype；shape 或 mask 不闭合时不得广播继续运行。

建议接口：

```python
SanaVideoBackbone.run_chunk(..., action_condition: Tensor | None)

CausalActionHybridArchitecture.forward_chunk(
    ...,
    noisy_actions: Tensor | None,
    committed_actions: Tensor | None,
    action_layout: ChunkActionLayout,
)
```

当前 vendored additive seam 位于本层 self-attention 之后：block 0 的本层
attention cache 不含 action，但 block output、bridge 和后续 block cache可以含
action。v0 必须把这一点写进 causal visibility test；若设计要求“每层 attention
前先看 action”，则需另立 pre-attention injection candidate，不能把当前 seam
描述成已满足。

### 5.7 内容、目标与动作监督分离

文本输入拆成：

- **scene/content**：物体、布局和可观测静态关系；
- **task/goal**：要完成的目标，不包含逐帧 expert trajectory；
- **motion/action**：只通过结构化 action/proprio 通道传递。

RoboTwin 的任务指令可以保留目标语义，但数据审计必须证明不存在由自动 caption
写入的 future motion trace。不得通过人工逐任务 prompt engineering 提升结果。

### 5.8 Generated-prefix / self-forcing（P0 objective blocker）

**HYPOTHESIS-H4：** 使用部署同构的 generated prefix 能降低 model-prefix exposure
gap，但不能无条件把原 expert action label 贴到模型偏离后的状态上。

以下只是必须满足的边界，不构成已完成的 objective：

- self-forcing 使用与部署完全相同的 chunk denoiser、scheduler、cache update、
  content-time 和 bootstrap contract；
- generated prefix 使用 `stop_gradient`；
- 同一训练 row 同时保留 ground-truth-prefix action path；
- action loss只在 ground-truth-prefix path 上计算；
- 不采集环境 recovery，不做 expert relabel，因此不是 DAgger；
- generated-prefix schedule 在 admission 前固定，不能根据结果调节。

关键未决点是：模型生成的 state/action 偏离后，原 GT future video 也可能成为不一致
的反事实 target。进入实现前，`SELF_FORCING_OBJECTIVE` 必须明确并可测试：

- generated-prefix target 的来源和 source SHA；
- teacher/student 是否处在相同的 state、action、noise 和 scheduler 条件；
- 是否使用 GT future video；若使用，必须证明 action/state 条件未错配；
- teacher checkpoint、detach boundary、distillation loss 和 timestep；
- generated action 与 GT video、或 GT action 与 generated video 的混配拒绝规则；
- 哪些 loss 在哪些 prefix source 上有效。

在该 design 闭合前，不得创建 `CACH-SF` 的 `CANDIDATE_SPEC`，也不得把一步
`x0` self-conditioning 称为 self-forcing distillation。

cache commit 必须满足：

1. 所有 denoise pass 都是 `save_kv_cache=False`，cache digest 不变；
2. chunk 完成后仅做一次 `t=0` commit；
3. commit 单位是配对的 `(video_x0, action_x0)`，不能混合 GT video 与 generated
   action，或相反；
4. teacher-forcing commit clean GT pair；
5. self-forcing commit detached model-generated pair；
6. 部署在 `c>=1` 时把 observed video chunk `c-1` 与真实已执行 action chunk
   `c-1` 一起 commit；
7. episode reset 同时清空 video cache、action commit cursor、Action RoPE offset
   和 AttnRes depth state。

现有 vendored `SelfForcingFlowEuler` 只能作为 `t=0` save 和 cache lifecycle 的
概念参考；它不含 ActionDiT、`ChunkActionLayout` 或训练 distillation，不能直接
当作新实现。一步 `x0` self-conditioning 也不能冒充完整 self-forcing
distillation。

部署必须核验 `conditions["action_history"]` 至少覆盖上一个完整 layout chunk，
禁止半 chunk 提前 regenerate；async/temporal ensemble 保持禁用。

仅供下一版 objective design 评估的 provisional schedule：

| 训练进度 | generated-prefix row 占比 |
|---|---:|
| `[0%, 25%)` | `0%` |
| `[25%, 50%)` | `25%` |
| `[50%, 75%)` | `50%` |
| `[75%, 100%]` | `75%` |

该 schedule 当前未获 admission，也不是 v0 已冻结选择。objective 闭合后若采用，
必须整体进入唯一 `CANDIDATE_SPEC`，不能根据结果调整；任何变更都是新 revision。

#### 5.8.1 Applied-action acknowledgement

`conditions["action_history"]` 当前保存的是 server-returned command，不能自动称为
环境实际执行动作。safety clipping、controller transform、丢帧或未执行完整 chunk
都可能令 commanded action 与 applied action 不同。

部署 commit 的科学 source of truth 必须是 environment/controller 回传的：

```text
APPLIED_ACTION_ACK {
  episode_id, chunk_id, layout_instance_digest,
  canonical_applied_tensor_or_immutable_ref,
  dtype, shape, token_order, action_values_digest,
  applied_count, controller_transform_digest,
  observation_interval_digest
}
```

只有 ack 与当前 observed video chunk、layout 和完整 action span 精确匹配，才可
commit。digest-only ack 不足以构造 `committed_actions`；engine 必须读取 canonical
applied tensor bytes，复算 dtype/shape/order/digest，并将其与 observation interval
绑定。若环境无法提供 applied values，只能把缓存标为 `commanded_action_only` 工程遥测，
不得用于“executed-action-conditioned”科学结论；默认 fail-closed。实现触点包括
`src/sana_wam/deploy/policy.py` 以及 server/environment feedback seam。

### 5.9 Loss 与 AFCC 隔离

capability lineage 不得修改或复用现有 AFCC formal 的：

- source bundle；
- authority/design；
- admission root；
- formal root；
- receipt；
- treatment/control 解释。

新 capability authority 必须单独固定 AFCC 状态。在本 capability 中：

- 本 capability 的 reference 与 candidate 都不得实例化 AFCC `F`；
- 若共享 schema 强制存在该字段，candidate/reference 均须在 `Trainer` 构造前
  设 `F=0`，并删除
  全部 AFCC/action-reference 字段；
- 新 variant 遇到 AFCC weight、reference 或 Phase-6 authority 字段应 fail-closed；
- 禁止 loss subtraction。

如果未来问题改为独立的 AFCC formal 因果对照，则回到 `AGENTS.md` 契约：
treatment 必须是原生 architectural `F=1`；control 必须在 `Trainer` 构造前
`F=0` 并删除 AFCC reference。该 formal 问题不在本 capability 计划授权范围内，
也不能复用本计划的 root、candidate 或结论。

AFCC held-out action-harm evaluator在本计划中只作 **veto**，不能作为“更
teacher-directed 就更好”的优化目标。

### 5.10 v0 明确不包含

- explicit 3D persistent memory；
- second-stage visual refiner；
- aesthetic DPO/ReFL；
- MoE；
- tokenizer/VAE 更换；
- action reranker；
- policy ensemble。

只有当 CACH-SANA-WAM 已通过 causal、generated-prefix 和 closed-loop gate，而失败
集中在遮挡、revisit 或 object permanence 时，才允许单独立项 3D memory。

## 6. Candidate ladder

本节是跨多个、需分别授权的 campaign roadmap，不是一次 campaign 中的候选池。
每个 campaign 在 G0 前只允许冻结一个 `CANDIDATE_SPEC`，字段至少包括：

- candidate/reference 的 resolved architecture 与唯一 delta；
- anchor indices、mask、cache、bootstrap 和 `ChunkActionLayout`；
- loss、self-forcing schedule、trainable parameter allowlist；
- dataset/row/seed manifest、训练步数和唯一 final endpoint；
- 唯一允许评测的 checkpoint；
- gate 阈值、任务、episode 数、CI 方法；
- source/runtime/input SHA 和 immutable root nonce。

每个 capability campaign 只能有一个 candidate 和一个 fixed reference；pair 两端
从 step 0 开始，不从上一个 campaign checkpoint fine-tune。共享参数使用相同
初始化 seed；声明为 adapter/bypass 的支路必须以预注册 no-op 初始化。anchor
replacement 和 enabled AttnRes 是显式 operator delta，使用预注册的 from-scratch
初始化，不伪造 no-op。中间 checkpoint 只用于安全遥测，不得用于科学评测或选优。

首先单独 commissioning `REF-GDN-CORRECTED`：

- LTX2 时间几何下 random-init all-GDN；
- 使用新的 bootstrap、layout、Action RoPE 和 cache/commit contract；
- video 不读取当前 action candidate；
- 完成 C0–C8 后才可称 corrected reference。

它是 reference commissioning，不是 paired campaign；当前旧 GDN-AR 因
fixed-ATC/prefix 漂移，不得称 exact-causal reference。

| Campaign | Fixed reference | Candidate 的唯一 delta | 目的 | 解锁条件 |
|---|---|---|---|---|
| `CACH-A` | fresh `REF-GDN-CORRECTED` | action-to-video conditioning | 验证 action-conditioned dynamics | reference commissioning + causal/bypass/action-label gates |
| `CACH-H` | fresh `CACH-A` | anchors `[7,15]` | 验证低剂量 global refresh | `CACH-A` G4 dynamics gate 通过 |
| `CACH-R` | fresh `CACH-H-SHAM` | 以 AttnRes router 替换 inert parameter-matched sham | 验证 routing，而非单纯参数增量 | `CACH-H` G4 anchor gate + `ATTNRES_DESIGN` |
| `CACH-SF` | fresh `CACH-R` | 注册的 generated-prefix objective/schedule | 验证 model-prefix robustness | `CACH-R` G4 + `SELF_FORCING_OBJECTIVE`；当前 P0 blocked |

`CACH-H-SHAM` 是 `CACH-R` campaign 唯一 reference 内的 inert 参数包装，不是第三
实验成员；裸 `CACH-H` 只作为前一 campaign 的冻结历史证据。

每一行相对上一行只允许表中所述单一 delta；每一行都需独立 plan SHA、authority、
root 和用户授权。当前 campaign 报告、失败 root 与 review 完成前，不得自动进入
下一行。第一个失败的 load-bearing gate 会停止后续依赖该假设的 campaign；
不得从多个失败 candidate 中挑成功率最高者。

若最终需要比较完整 `CACH-SF` package 与 `REF-GDN-CORRECTED`，那是另一个只估计
package effect、不能归因单组件的预注册 campaign，不包含在 v0 自动执行范围内。

## 7. Metrics 与 falsification gates

### 7.1 Contract metrics

| Gate | 必须证明 |
|---|---|
| `C0 causal visibility` | future perturbation 对 earlier output/JVP 无影响 |
| `C1 bypass identity` | zero-init action adapter、disabled anchor selector/router 与 reference 一致；不要求 enabled operator delta 假装 no-op |
| `C2 train/deploy equivalence` | 相同 prefix、noise、cache 下 chunk 输出闭合 |
| `C3 content-time identity` | cache slot、latent、raw frame、action 和 RoPE 使用同一 `ChunkActionLayout` |
| `C4 bootstrap semantics` | `first_frame_pinned + observed_prefix_chunks=0`；旧漂移字段被拒绝 |
| `C5 source closure` | config、code、data、checkpoint、runtime hash 完整 |
| `C6 action coverage` | chunk 0 `(K-1)r`、后续 `Kr`；非 pad action 恰好覆盖一次 |
| `C7 commit atomicity` | denoise 不写 cache；每个 completed video/action pair 只 commit 一次 |
| `C8 hybrid cache` | GDN full-history、softmax one-chunk、AttnRes forward-local 的语义与实现一致 |

任一 contract gate 失败，禁止训练。

### 7.2 Numerical metrics

- GDN decay/write/state 的 min、max、p50、p95、p99；
- state norm 与 update norm 随 chunk 的斜率；
- anchor logits/entropy 和输出 norm；
- AttnRes source 权重、block-entry rank retention；
- BF16/FP32 residual；
- NaN/Inf、overflow、underflow；
- peak CUDA、host high-water、I/O 和 step time。

数值 gate 不允许通过“忽略异常 row”或事后放宽容差。

### 7.3 Capability proxy

必须在固定、未选择的 cohort 上同时报告：

- dream minus copy-frame margin；
- 1/2/4/8-step rollout error 与 drift slope；
- teacher-prefix/model-prefix gap（仅对已闭合的 `CACH-SF` objective）；
- contact/pose correspondence；
- action open-loop error，但只作为 secondary；
- fixed-cohort AFCC/action-harm response；
- throughput 和显存。

单步 dream、open-loop MAE 或 internal rank 不能单独放行。

### 7.4 Closed-loop primary endpoint

v0 预注册只有 `CACH-SF` candidate 与 fresh `CACH-R` reference 可以进入
closed-loop。依赖链中任一 campaign 失败，v0 不启用 fallback candidate；若
`SELF_FORCING_OBJECTIVE` 未闭合，则 G5 保持阻塞。

Primary：

- randomized/长时条件下的 paired success difference；
- 预注册 episode seed manifest；
- 每任务等权；
- paired confidence interval。

Safety：

- clean condition 非劣；
- 无单任务显著崩溃被平均值掩盖；
- 无 action-harm veto；
- 无改变 action consumption、rerank、best-of-N 或 temporal ensemble。

非劣 margin、任务集、episode 数和 CI 方法必须在看到结果前由独立 design 固定。

### 7.5 Campaign gate matrix

以下 gate 是联取关系，不做加权总分。阈值必须在读取 candidate 结果前写入
`CANDIDATE_SPEC`。

| Gate | 范围 | 放行要求 | 典型否决 |
|---|---|---|---|
| `G0 static` | resolved reference/candidate | source/runtime/input SHA、唯一 delta、trainable allowlist、root/endpoint/seed 全闭合 | 外部路径未闭合；candidate 池；AFCC/reference 污染 |
| `G1 unit` | mini model/temporary root | C0–C8、selector、bypass identity、failure receipt 全通过；hard test 无 skip | fixed-ATC、future leak、cache mutation、root overwrite |
| `G2 numeric` | pair 两端依次真实 dtype forward/backward | 非有限值为 0；BF16/FP32、残差、梯度、峰值资源在预注册界内 | `1e15–1e18` 类异常、OOM、靠放宽检查运行 |
| `G3 update-free causal` | full causal vs deploy chunk；不更新参数 | full/chunk 等价、future perturbation 为零、cache/layout 闭合；zero-init adapter 用 parameter-gradient/JVP 或注册 synthetic nonzero diagnostic 证明 wiring | 任一 leak、cache mutation、operator/engine 语义不等价 |
| `G4 registered train` | 同 shared-tensor init digest/data order/optimizer/step；operator-specific init 各自冻结 | 每个 campaign 唯一 endpoint；中间 campaign 可为 reduced horizon；最终 G5 pair 必须有独立 full-horizon/data-scale authority | teacher-only 改善；多步或 action 恶化；挑 checkpoint；低数据结论越界 |
| `G5 closed-loop` | 仅 `CACH-SF`/`CACH-R` | randomized/长时 superiority CI、clean 非劣、逐任务完整 episode | fallback candidate、少跑 episode、只赢单任务/seed、结果后改口径 |

基础设施失败允许在新 root 重跑，但必须先冻结失败 root、登记故障类别与修复 SHA。
任何 code/config/threshold 变化都产生新 candidate revision，不能称同一次 retry。

## 8. 阶段化实施计划

### Stage 0：source closure 与已知契约修复

只做实现前准备：

1. 闭合 H200 的 governance/source 文档并固定新的 Git/source lineage；
2. 为当前获授权的第一个 campaign 冻结唯一 `CANDIDATE_SPEC`；
3. 新 variant 固定 `first_frame_pinned + observed_prefix_chunks=0`，拒绝旧漂移字段；
4. 实现前设计并静态证明 `ChunkActionLayout`，废止 fixed-ATC；
5. 固定 hybrid cache 的 one-chunk softmax/full-history GDN 语义；
6. 为未来 AttnRes 固定一手 source pin 要求；enabled design 在 `CACH-H` G4 后解锁；
7. 初训拒绝 pretrained DiT，固定 resume/deploy key allowlist 与完整 trainable inventory；
8. 审计 seed、部署 noise、paired cache commit 和 applied-action ack contract；
9. 建立独立 authority/design/verifier/launcher；
10. 保持 AFCC source 与 frozen roots 完全不变。

**GATE-S0：** source bundle 可从 Git/bundle 重建，当前工作树的隐式 dirty byte
不得成为唯一运行真相。公共 P0 与当前 candidate delta 的 P0 必须闭合；后续
component-specific blocker 不阻止较早 candidate，但阻止其自身实现。

### Stage 1：纯 contract 实现

只实现公共 contract 和当前已授权 candidate delta：

- action-to-video zero-init adapter；
- `ChunkActionLayout` 与累计 Action RoPE cursor；
- 当前 delta 所需 selector/mask/operator；
- hybrid cache/content-time metadata；
- 未来 component 只允许 disabled schema/bypass stub，不实现 enabled AttnRes 或
  self-forcing runner。

**GATE-S1：** 静态审计、mini-model contract tests 和 verifier 全部通过。

### Stage 2：mini-model numerical admission

仅使用小尺寸随机模型和合成 token：

- future perturbation；
- chunk equivalence；
- declared adapter/bypass identity；
- cache growth/reset；
- action/video time alignment；
- bootstrap `(K-1)r`、continuation `Kr`、partial tail 和 action 全覆盖；
- paired video/action `t=0` single commit；
- commanded/applied acknowledgement mismatch；
- failure injection 与 frozen failure receipt。

generated-prefix target/stop-gradient tests 只在 `SELF_FORCING_OBJECTIVE` 闭合后的
Stage 5 启用。该阶段不是科学结果。

### Stage 3：full-model update-free causal proxy

对当前 pair 的 complete random-init model 做真实 shape、真实 dtype、无参数更新的：

- 独立 immutable root；
- common shared-parameter initialization；
- full-sequence causal / deploy chunk-cache equivalence；
- future perturbation、cache digest 与 layout instance；
- zero-init action adapter 用 parameter-gradient/JVP wiring 检查；如需输出敏感性，
  只在非科学 diagnostic 中加载注册的 synthetic nonzero adapter；
- numerical/resource admission。

该阶段不训练、不保存“候选 checkpoint”，不产生 capability 改善结论。

### Stage 4：registered reduced from-scratch joint training

在用户逐个授权后，当前 campaign 的 reference/candidate 从共同注册初始化分别
训练。两端使用相同 data order、seed、optimizer 和唯一 step budget；只允许一个
final scientific endpoint，无 checkpoint selection。当前 campaign 的 G4、冻结
报告和独立 review 全部通过后，才可申请下一 campaign；不得自动启动。

`CACH-A/H/R` 的 4k 可以注册为 reduced mechanism endpoint，只支持对应增量 gate，
不能称 full-capability training。

历史估算显示 2B、8 GPU 约 `2.4 s/step`：

- 4k steps 每臂约 `2.7 h`、约 `21 GPUh`；
- paired 4k 约 `43 GPUh`，另加至少 20% buffer；
- 12k paired 约 `128 GPUh`，不能称为短 smoke。

这些只是容量规划估算，启动前必须重新测量，不是 runtime 承诺。

### Stage 5：generated-prefix admission

在 `CACH-R` 通过且 `SELF_FORCING_OBJECTIVE` 闭合后，先运行固定 1-row/2-row
generated-prefix admission：

- exact deploy operator；
- main-process RNG digest；
- worker-materialized row identity；
- prefix source 标记；
- action-loss mask；
- video target identity；
- cache content-time；
- distinct fresh process。

admission 必须标记 `scientific_eligible=false`。

### Stage 5.5：final full-horizon training

只有 Stage 5 通过后，`CACH-SF/CACH-R` 才能在新的 final campaign 中：

- 通过 `DATA_AND_SCALE_DESIGN`；
- 从 step 0 使用 full horizon 训练；
- 使用唯一 final checkpoint；
- 保存 full-training completion receipt。

不得从任何 4k checkpoint continuation，也不得把早期 checkpoint 挑成 final。

### Stage 6：唯一 candidate closed-loop

只有 v0 预注册的 `CACH-SF` candidate 与 fresh `CACH-R` reference 能进入：

- randomized/长时主 endpoint；
- clean 非劣 endpoint；
- 每任务完整报告；
- paired seed manifest；
- frozen launcher/config/checkpoint；
- `DATA_AND_SCALE_DESIGN` 与 full-training completion receipt；
- 无 post-hoc checkpoint 或 task selection。

其 direct reference 是 `CACH-R`，因此该 campaign 只归因注册的 generated-prefix
objective/schedule 增量。它不自动构成 `CACH-SF` 相对
`REF-GDN-CORRECTED` 的完整 package effect。

## 9. 代码触点

### 9.1 优先复用

| 文件 | 复用内容 |
|---|---|
| `src/sana_wam/model/gdn_ar.py` | chunk loop、GDN cache lifecycle、video-to-action bridge |
| `src/sana_wam/deploy/gdn_ar_engine.py` | 部署 chunk denoiser 和 cache update contract |
| `src/sana_wam/model/video_backbone/sana/pipeline_builder.py` | architecture factory 与 GDN preset |
| `third_party/Sana/diffusion/model/nets/sana_gdn_blocks.py` | frame-wise GDN、chunk-causal operator |
| `third_party/Sana/diffusion/model/ops/fused_gdn.py` | fused recurrent kernel |
| `src/sana_wam/model/cross_attn.py` | action 读取 video 的 bridge 基础 |

### 9.2 建议新建

| 建议文件 | 责任 |
|---|---|
| `src/sana_wam/model/causal_action_hybrid.py` | CACH architecture 和 action-to-video 数据流 |
| `src/sana_wam/model/action_chunk_layout.py` | causal-VAE/raw-frame/action/RoPE 精确时间映射 |
| `src/sana_wam/model/video_backbone/sana/causal_softmax_anchor.py` | fixed selector、causal anchor 和 mask |
| `src/sana_wam/model/video_backbone/sana/block_attn_res.py` | shared-projection Block AttnRes |
| `src/sana_wam/model/video_backbone/sana/hybrid_cache.py` | GDN/softmax cache type、window 和 commit verifier |
| `src/sana_wam/model/self_forcing_context.py` | exact deploy-equivalent generated-prefix paired commit |
| `src/sana_wam/deploy/causal_action_hybrid_engine.py` | action-history、layout、paired cache cursor 与 reset |
| `configs/experiments/cach_sana_wam_v0.yaml` | 单一预注册 architecture config |
| `scripts/verify_cach_sana_wam_admission.py` | source、causal、cache、row 和 artifact verifier |

优先在主仓 wrapper/adapter 中实现，避免不必要地修改 Sana 子模块。若必须修改
`third_party/Sana`：

1. 先在 Sana 独立分支提交；
2. 导出可恢复 bundle；
3. 更新主仓 gitlink；
4. 记录新 commit 和 bundle SHA；
5. 不假定 origin 包含该对象。

还需小范围接入：

- `src/sana_wam/model/__init__.py`：新 variant dispatch；
- `src/sana_wam/deploy/__init__.py`：在 GDN-AR 父类 fallback 前 dispatch 新 engine；
- `src/sana_wam/model/video_backbone/sana/adapter.py`：透传 frame-aligned
  `action_condition`；
- `src/sana_wam/deploy/policy.py` 与 server/environment feedback seam：接收并
  验证 `APPLIED_ACTION_ACK`；
- `pipeline_builder.py`：验证精确 layer type table、missing/unexpected key allowlist。

不要改 frozen `robotwin_dataset.py` 来“方便”layout；新增 wrapper/metadata
transform。若 AttnRes 必须改 vendored block loop，只增加明确 seam，并按 Sana
独立 commit/bundle/gitlink 流程交付。

### 9.3 必须新增的测试

建议测试文件：

- `tests/test_cach_anchor_schedule.py`
- `tests/test_cach_future_visibility.py`
- `tests/test_cach_block_attn_res_identity.py`
- `tests/test_cach_action_to_video_conditioning.py`
- `tests/test_cach_action_chunk_layout.py`
- `tests/test_cach_action_label_mask.py`
- `tests/test_cach_self_forcing_equivalence.py`
- `tests/test_cach_cache_content_time.py`
- `tests/test_cach_hybrid_cache_contract.py`
- `tests/test_cach_engine_action_history.py`
- `tests/test_cach_applied_action_ack.py`
- `tests/test_cach_self_forcing_target_identity.py`
- `tests/test_cach_checkpoint_key_allowlist.py`
- `tests/test_cach_trainable_parameter_allowlist.py`
- `tests/test_cach_afcc_isolation.py`
- `tests/test_gdn_ar_observed_prefix_contract.py`
- `tests/test_cach_failure_receipt.py`
- `tests/test_cach_config_schema.py`

最低覆盖：

1. fixed anchor 数量和位置；
2. future video/action leakage；
3. noisy/clean modality leakage；
4. zero-init bitwise/registered-tolerance identity；
5. train/deploy chunk equivalence；
6. cache reset、深度和 content-time；
7. generated prefix stop-gradient、target identity 和 action-loss mask
   （只在 `SELF_FORCING_OBJECTIVE` 闭合后启用）；
8. generated-prefix action loss 为零；
9. ground-truth-prefix action loss 保留；
10. chunk 0 `(K-1)r`、continuation `Kr`、partial tail、RoPE cursor；
11. pinned LTX2 真实 encode 的首块/续块/tail 与部署 obs-band 对齐；
12. `LAYOUT_SPEC_SHA` 与逐实例 digest 分离且闭合；
13. denoise cache byte不变、paired `t=0` commit 恰好一次；
14. action history/ack 不足、digest-only ack、canonical applied tensor
    dtype/shape/order 错、commanded/applied 不同、混配 commit、半 chunk
    regenerate 和 stale cursor fail-closed；
15. committed applied action 改变时，parameter-gradient/JVP wiring 改变；输出
    sensitivity 只用注册 synthetic nonzero adapter 或 G4 final endpoint 验证；
16. AttnRes 20 层、`S=8` 的 final 4-layer partial block flush/reset；
17. 初训拒绝 pretrained DiT；resume/deploy checkpoint key set 精确匹配；
18. 完整 random-init trainable inventory：DiT/GDN/anchors/AttnRes/action/
    conditioner 可训练，只有 VAE/text encoder 冻结；
19. AFCC weight/reference/config 字段 fail-closed；
20. O_EXCL/O_NOFOLLOW、immutable root 和 failure freeze。

`pipeline_builder.py` 当前 checkpoint load 使用 `strict=False` 并只 warning；
v0 初训不得调用该 permissive pretrained-DiT load。resume/deploy 只能加载同一
candidate revision，并要求 exact key set；任何 missing/unexpected key 都失败。

现有 `training.freeze: [video_backbone.dit,...]` 会递归包裹 `no_grad`，与 complete
hybrid joint training 冲突。v0 禁止冻结 `video_backbone.dit`：DiT/GDN/anchors、
AttnRes、action backbone、proprio encoder 和 action conditioner 全部
`requires_grad=True`。只冻结 source-pinned LTX2 VAE 与 text encoder，并用完整
trainable/frozen inventory 证明；不得用整体 `no_grad` 切断 trainable 支路。

## 10. Artifact 与 launcher 契约

### 10.1 独立命名空间

建议新 lineage：

```text
/DATA/share/sana_cach_wam_20260731/
```

该路径目前只是计划名称，不得在没有执行授权时创建。

结构建议：

```text
sana_cach_wam_20260731/
  <campaign_id>/
    PAIR_REQUEST.json
    CANDIDATE_SPEC.json
    SOURCE_MANIFEST.json
    REFERENCE/
    CANDIDATE/
```

`campaign_id/run_id` 必须包含 plan SHA 和唯一 nonce。每次实际执行必须使用由
launcher 排他创建的、direct、empty、`0700` root；目标已存在则在 GPU reservation
前失败。状态机固定为：

```text
RESERVED -> STATIC_PASS -> UNIT_PASS -> NUMERIC_PASS
         -> CAUSAL_PASS -> SHORT_PASS -> CAMPAIGN_COMPLETE
                                      -> CLOSED_LOOP_PASS -> CAMPAIGN_COMPLETE
ANY_NONTERMINAL -> FAILED
```

每个状态转换必须消费上一 receipt SHA，并以 exclusive/no-overwrite publish。最少
保存：

- source/runtime/input manifest、resolved config、argv/env；
- dataset/seed/row trace、`LAYOUT_SPEC_SHA` 与 instance digests；
- GPU UUID、boot ID、PID/start ticks、磁盘 snapshot；
- checkpoint raw hash、trainable parameter inventory；
- raw metrics、verifier、stdout/stderr；
- `COMPLETION` 或 `FAILURE` 以及最终 root inventory。

成功或失败后：

- 先 exclusive 写 canonical success/failure receipt 与 final inventory；
- `fsync` 并由 verifier 复核；
- 再把文件冻结为 `0400`、目录冻结为 `0500`；
- 最后完成只读 inventory/receipt SHA 复核；
- 禁止覆盖或复用。

重跑必须使用新 root，并用 `supersedes` 指向旧 root。`/tmp` 只能 staging，不能
承载唯一 authority 或证据。

### 10.2 Launcher hard requirements

launcher 必须：

- 由外部 literal SHA trust anchor 固定；
- evaluator、verifier、design、authority、launcher 五件套闭合；
- `CANDIDATE_SPEC` 与唯一 endpoint 硬绑定；
- evaluator 自身硬绑定 fixed run root；
- 在安装 failure trap 后才开始任何 root mutation；
- freeze 完成并复核后才关闭 trap；
- 使用 process group，失败时清理并验证所有 worker；
- 检查 GPU UUID/占用并持有排他 reservation；
- 检查 `/DATA` 容量与 inode；
- 检查 AFCC formal lock、root ownership 和无关 GPU process；
- 不停止或抢占他人作业；
- 对所有输出使用 exclusive creation 和 no-follow；
- 重新核对 config、data、checkpoint、source 和 runtime；
- 记录 boot ID、PID、process start ticks 和 GPU UUID。

## 11. 资源与容量边界

2026-07-31 的瞬时只读快照：

- 8 张 H200 当时无 compute process；
- 主机约 2.0 TiB RAM；
- `/` 约 2.3T free；
- `/DATA` 约 1.7T free、76% used。

该快照不构成 launch admission。每次启动前必须重查。

容量规则：

- launcher 使用字节级硬条件
  `projected_free_after >= 0.20 * total_bytes + safety_buffer_bytes`；
- `safety_buffer_bytes` 在 authority 中固定，并覆盖并发无关写入与估算误差；
- 当前容量只比 20% 保留线高约 0.3T；
- 单个 paired short-run root 的 `300GB` 只是预算上限，不是自动 admission；
  当前约 0.3T headroom 下必须缩小 projected write 或增加安全空间；
- 一次只 admit 一个 paired root；
- checkpoint 数量和保留步必须预注册；
- 不通过删除冻结证据腾空间。

规划上限（不是启动承诺）：

- G0–G1：CPU-only，约 1–2 CPU-hour，磁盘 `<5GB`；
- G2：单 H200、reference/candidate 串行，每个 paired campaign 约 1–2 小时；
- G3：每个 paired campaign 上限 8–16 GPUh，超过则重新设计 proxy；
- G4：2B、8 GPU、4k step 历史估算约 43 GPUh/paired，加 20% admission
  buffer 约 52 GPUh；12k paired 约 128 GPUh，不称为 short；
- G5：paired 100 episodes 历史估算约 24 GPUh/任务，任务数线性增长。

4k/12k 只是 reduced campaign 容量估算，不是 final random-init 2B convergence
预算。最终 full-horizon GPUh、checkpoint 容量和 walltime 必须由
`DATA_AND_SCALE_DESIGN` 与 scaling pilot 在结果前固定；当前为资源阻塞项。

已有 AFCC smoke 的单臂峰值约 131.1GB CUDA、107.5GB host RAM，只能作为容量
下界警示；不得同卡共置。完整四任务 roadmap 可能达到约 155–170 GPUh，因此
必须逐 campaign 授权，不能一次性 reserve 全链。

## 12. Stop rules

出现以下任一情况立即 fail-closed：

1. future video/action 对 earlier output 有影响；
2. train/deploy chunk 输出不闭合；
3. 声明为 zero-init adapter/disabled bypass 的路径不是 no-op；
4. cache content-time 与真实 observation/action 时间不一致；
5. fixed-ATC、action gap/overlap、partial-tail 静默丢弃或 RoPE cursor 漂移；
6. denoise 修改 cache，或一个 chunk commit 零次/多次；
7. generated-prefix 的 action 或 video target 与生成 state/action 条件不一致；
8. commanded action 被冒充 applied action，或 ack/layout/video chunk 不匹配；
9. state/gate 出现 NaN、Inf、无界增长或全面饱和；
10. candidate 只改善 rank、dream 或 open-loop，而多步 drift 不改善；
11. candidate 显示显著 action harm；
12. clean 明显退化；
13. 需要任务、seed、checkpoint、比例或层位的 post-hoc selection 才成立；
14. source、data、runtime 或 receipt identity 不完整；
15. GPU/磁盘 reservation 不满足；
16. root 被污染、复用、链接替换或部分冻结。

失败 root 保留，不能在原 root 修复后重跑。

## 13. 风险台账

| 风险 | 可能后果 | 预防/证伪 |
|---|---|---|
| action-to-video 使用 clean future action | 严重训练泄漏 | noisy candidate only + future perturbation test |
| causal softmax mask死代码/no-op | 训练利用未来 shortcut | JVP + explicit mask inventory |
| AttnRes只增加参数量 | 错误归因 depth routing | 唯一 reference 使用 inert parameter-matched sham |
| fixed-ATC忽略首 latent anchor | action/video 错位 | `ChunkActionLayout` + coverage/RoPE tests |
| softmax cache被误称全历史 | 错误解释 long-context | 明确 one-chunk window + slot verifier |
| GDN过度遗忘 | object state 丢失 | retention/object-persistence metrics |
| self-forcing复用无效 GT action/video | 反事实 target 错配 | `SELF_FORCING_OBJECTIVE` + target identity tests |
| commanded action冒充 applied action | cache 与真实动力学错位 | environment ack + controller transform digest |
| denoise/commit混用 | cache重复写或脏读 | read-only digest + paired t=0 single commit |
| from-scratch训练不稳 | 重现 fragile basin | numerical admission + fixed schedule |
| dream proxy与closed-loop脱钩 | 离线假阳性 | multi-step + final closed-loop gate |
| 3:1先验过拟合 | ratio cherry-pick | v0固定 `[7,15]`；任何变更另立 campaign |
| AFCC目标污染 capability 结论 | teacher attraction 被误当能力 | 独立 authority + action-harm veto |
| permissive checkpoint load | 静默漏权重/错权重 | 初训拒绝 pretrained DiT；resume/deploy exact keys |
| 整体 `no_grad` 冻结 DiT | complete hybrid 未联合训练 | 仅冻结 VAE/text + full trainable inventory |
| Git dirty byte成为运行依赖 | 无法重建 | source commit/bundle/manifest closure |
| frozen root占满 `/DATA` | 无法安全继续 | 20% reserve + explicit buffer + byte projection |

## 14. 明确禁止重跑/复活的路线

- epsilon、strict-positive 或 aligned-kernel 的继续调参；
- AR 全层 dense/segmented softmax graft；
- 历史 `joint_graft` 单独重跑；
- AFCC weight sweep 或 loss subtraction；
- DAgger、DART、人工 recovery；
- rerank、best-of-N、temporal ensemble；
- receding horizon；
- encoder/V-JEPA/Cosmos 替换作为单独能力修复；
- tokenizer/VAE 替换；
- 单纯扩大 multi-task 数据；
- 只凭 open-loop MAE、IDM decodability 或单步 dream 放行；
- 从多个 anchor ratio、layer placement 或 checkpoint 中挑最好结果。

## 15. 完成定义

### 15.1 文档阶段完成

- 本文与 `deep-research-report.md` 在 H200 可读；
- SHA 与 Git diff 已复核；
- 无意外修改其他文件；
- 本文明确标记未授权执行。

### 15.2 实现阶段完成

只有满足以下全部条件，才能称“实现完成”：

- architecture/config/schema 完整；
- 公共 `ChunkActionLayout`、bootstrap、hybrid cache、applied-action 和 paired
  commit 设计闭合；
- 当前获授权 candidate delta 的 design/implementation 闭合；不得用未来
  AttnRes/self-forcing 未实现阻塞较早 candidate，也不得提前宣称它们完成；
- checkpoint key 与 trainable parameter allowlist 精确闭合；
- contract tests 覆盖第 9.3 节；
- independent verifier 通过；
- source bundle 可重建；
- failure injection 能冻结失败 root；
- 尚未因此宣称 capability 改善。

### 15.3 当前增量 campaign 完成

只有满足以下全部条件，才能称“当前 candidate 的预注册增量假设得到支持”：

- 当前适用的 causal、bypass、cache 和 content-time gates 全通过；
- G4 唯一 final endpoint 完整；
- fixed cohort 多步 drift 优于 reference；
- action-harm veto 不触发；
- 无 post-hoc selection；
- 全部结果、失败与残差完整报告。

这不等于 closed-loop capability 已成立。

### 15.4 v0 closed-loop 目标完成

只有 `CACH-SF`/`CACH-R` 预注册 final campaign 额外满足以下全部条件，才能称 v0
closed-loop 假设得到支持：

- `SELF_FORCING_OBJECTIVE` 与 generated-prefix admission 通过；
- environment-confirmed applied-action ack 全覆盖；
- `DATA_AND_SCALE_DESIGN` 与 full-training completion 通过；
- randomized/长时 G5 superiority primary endpoint 通过；
- clean 非劣通过；
- 每任务完整报告且无 fallback candidate。

多个顺序 campaign 都通过，只支持各自注册的增量链；不得自动改写为完整 package
相对最早 baseline 的单一因果效应。

否则必须报告为失败、未决或仅工程进展。

## 16. 下一步（需要单独授权）

建议下一轮只做 Stage 0：

1. 将当前计划、research report 和治理入口纳入可恢复 source lineage；
2. 写 `ChunkActionLayout` design，逐 row 证明首 chunk、continuation、tail 和 RoPE；
3. 固定新 variant bootstrap，并规定旧 clean-prefix 字段 fail-closed；
4. 写 hybrid cache、applied-action ack、paired commit 和 random-init trainable design；
5. 固定 LTX2/text source，初训拒绝 pretrained video-DiT；
6. 起草 `DATA_AND_SCALE_DESIGN`，明确当前缺失的大规模训练数据/预算；
7. 先完成 `REF-GDN-CORRECTED` commissioning design；
8. 冻结 `CACH-A` 的 `CANDIDATE_SPEC`、authority 和 verifier skeleton；
9. 写测试清单/测试代码 diff，但暂不运行；
10. 提交实现前 diff 给用户确认。

AttnRes source pin/design 只在 `CACH-H` G4 通过后解锁；`SELF_FORCING_OBJECTIVE`
只在 `CACH-R` G4 通过后解锁。

训练、评测、capture 和正式 root 创建仍需后续独立授权。
