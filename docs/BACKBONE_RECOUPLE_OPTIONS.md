# SANA-WM 世界-动作模型：动作↔动态再耦合方案评审

## 1. 一句话结论

**最有前途的方向是 `2.2 Latent-Transition Inverse Dynamics`（把已搭好的 `idm` 变体做成主对齐目标）—— 它是唯一在机制上"动作目标无法从常量解出"的现行架构改动，工程量最低（复用已 materialized 的 clean state + 现有 scaffold），且有最干净的发表证据（WAM arXiv:2603.28955 同款 IDM-aux 直接给出 BC +11.8pt）；但所有方案——包括它——都受同一个上游天花板制约：SANA-WM 在该 domain 的 dream 退化到 -4dB（H2），而且 oracle clean-future video 反而让动作变差 1.2-1.3x，因此任何"强迫动作读 world latent"在落地前都必须先用 ~1 GPU-hr 的 linear-probe 验证 latent 里到底有没有动作可解的内容。**

核心判断：14 个方案里没有一个被两位评审都判为"无条件 pursue"。真正的分水岭不是"耦合机制是否新颖"——绝大多数是 LingBot-VA / Cosmos / VPP / LAPA 的移植——而是两件事：(a) 该机制能否**结构性地堵死** proprio+text 这条捷径（多数做不到，只是"把可选 side-info 换个张量"）；(b) world latent 在这个 domain 里到底有没有可用内容（H2，多数方案绕不开）。

---

## 2. 三大方向分组

### A. 现行架构改动 (current-arch)

| ID / 方案 | 根因契合 | 可行性 | 预期影响 | 关键风险 | 复核结论 | 置信度 |
|---|---|---|---|---|---|---|
| **0.0** IDM-Forced Bridge (共享 video_projs 的 aux IDM 头) | 部分。IDM 头自身抗塌缩成立，但**单向**：强制 video_projs 含内容 ≠ 强制主 denoise 头去用它。主头仍有 gate_ca + proprio/text 逃逸口 | 中。需把 clean ingest 改成 grad-bearing，AR 路径无 within-chunk ckpt → 2-chunk 全激活内存压力大 | 低-中。能改善表征/ablation 比值，但主头可能门控到 0，闭环仍 0% | 主头绕过；IDM 自身的 constant/velocity 捷径；oracle-dream 更差暗示 latent 本身无用 | **maybe** (两评审一致) | 0.7 / 0.62 |
| **0.1** Action-as-Latent-Frame (action token 写进 GDN 序列) | 评审分歧。skeptic：GDN `_reshape_to_temporal` 硬要求 N==T*S，append token 要么崩要么跑独立 recurrence → 不耦合；evidence：能动 MAE 但 WorldVLA 证明 action↔action 共注意会掉 10-50% | 低。"GDN 支持变长 token"前提被代码证伪；实为重写算子（保持 pretrained 权重可加载），远超 400-600 LOC | 低-中。GDN 层零耦合，仅 softmax 少数层耦合 | 算子 reshape 崩溃；新形态再塌缩；部署镜像分歧 | **drop**(skeptic) / **maybe**(evidence) → 取 **drop** | 0.74 / 0.7 |
| **0.2** DC-restoring bridge (RMSNorm + variance-floor 杀 mean-bias) | 否（两评审一致）。塌缩在 V 的**下游**（o-proj / gate_ca / 均匀注意力）；禁掉一个退化输入统计量无法制造输出依赖；自洽性问题：LayerNorm 重心化的是 channel 不是 token | 高（工程）/低（作为修复）。RMSNorm 会重新引入 1e9 DC 量级（当初用 LayerNorm 的原因） | 低 | 塌缩迁移到下游；loss 重新爆炸；ablation 比值靠构造升高（假阳性） | **drop**(skeptic) / **maybe**-as-falsifier(evidence) → **drop as standalone** | 0.8 / 0.72 |
| **0.3 / 2.0** GDN-state readout (tap S_kv/S_z 递归态) | 部分但方向最对。S_kv 是累积的"东西在哪"记忆，**结构上不可 mean-pool 塌缩**；但不改 loss 则 proprio 近充分，软性再塌缩 | 低。**致命**：监督 denoise pass 跑 `save_kv_cache=False` → 返回 None，无 state 可 tap；streaming state 来自 raw Triton 无 autograd → grad 路径是 kernel 集成项目，非 250-400 LOC glue；hybrid 中 ~1/4 softmax 块 slot 0/1 是 K/V token 不是 S_kv | 中（有条件）。机制正确，但 grad 路径假阳性风险高 + S_kv 可能被 texture/recon 子空间主导 | grad 检测静默 detach；低秩 readout 丢几何；-4dB state 内容本身可疑 | **maybe**(全部 4 条评审一致) | 0.7/0.66/0.62/0.62 |
| **0.4** Bridge rep-alignment + deploy-matched degradation | 部分，且**不对称**：VICReg variance 项防表征塌缩 ≠ 防信息解耦（高方差仍可纯 proprio 函数）；degradation 半边才是真杠杆但 GT 目标不变时 SGD 仍把扰动 bridge 当 nuisance 忽略 | 中。alignment 头近免费；degradation 需 v_pred 重建 self-conditioned chunk + 额外 eager forward | 低-中。对齐到 -4dB 退化 bridge = 传递贫乏 | alignment 塌到共享常量；degradation 退化成已被否的 R0；对齐目标本身退化 | **maybe**(两评审一致) | 0.7 / 0.6 |
| **1.2 / 2.2** Inverse-Dynamics readout (从 clean S_{c-1}→S_c transition 解码 action) | **是**（信息论上 transition 不可从常量解出）——但仅对**非因果 aux 分支**成立；因果部署头仍可绕过。**前提**：必须把 proprio 从 IDM 分支撤掉，否则 EEF action≈proprio 位移，捷径直接解出 | 高。clean state 已 materialized，`idm` 变体已有 scaffold；但 clean ingest 是 no_grad+detached，需小改 | 中（配 path-forcing 时）/ 低（纯 aux）。WAM 同款给 +11.8pt BC | proprio 泄漏（最可能）；aux 不传导到部署头；IDM 非因果不能做部署头；-4dB latent 内容 | **pursue**(2.2-evidence) / **maybe**(其余3评审) | 0.74/0.72/0.78/0.66 |
| **1.3** Norm-swap + GDN-state tap (0.2+0.3 组合) | 否。两个子修复都把动作留作 AdaLN-gated 的**可选** KV reader，gate_ca 逃逸口原封不动 | 高(a)/中(b)。但 evidence 指出 adapter.py 行号引用错误（文件仅 521 行），S_kv 是 detached 无法 backprop into | 低-中。诊断价值高，破 0% 概率低 | gate 再塌缩；RMSNorm 量级；GDN-state 仅 GDN-AR 有 | **maybe**(两评审一致，均强调"仅作诊断垫脚石") | 0.72 / 0.78 |
| **1.4** LAPA latent-action pretrain + future-state aux | 否。LAPA 仅是初始化，real-action finetune 可经 gate_ca→0 再塌缩；future-state aux 可被 proprio+FK 满足（预测 mean bridge = 塌缩不动点） | 中。future-state aux 低成本；两阶段 VQ pretrain 是 pipeline 重负，<200 demo 无法做 web-scale codebook | 低-中。Cosmos ablation：aux 仅值 +1.5pt，数据效率来自 shared trunk（本方案不采用） | aux 被无 shared-trunk 旁路；codebook garbage；LAPA 已知在抓取上变差 | **maybe**(两评审一致，仅推 cheap future-state aux 半边) | 0.72 / 0.7 |

### B. 相关工作移植 (related-work)

| ID / 方案 | 根因契合 | 可行性 | 预期影响 | 关键风险 | 复核结论 | 置信度 |
|---|---|---|---|---|---|---|
| **1.0** MoT joint self-attention (action token IN 共享注意力) | 是（结构上消灭 pooled bridge）——但**分歧大**。skeptic：`SanaMoTJointDriver` 硬拒 gdn kernel（仅 linear_relu），pretrained backbone 是 gdn → 不可用，"已搭好"是假；且 gate-to-zero 新塌缩 + linear-attn 本身就是低秩 pooled 读。evidence：GDN-aware `SanaMoTJointDriver` **已实现**（cumsum dual-track），仅未接线 | 低(skeptic) vs 高(evidence) — **直接代码冲突，必须现场验证** | 中-高。直击确认的 covariate-shift 根因 | 仅耦合到 observed-prefix 而非 dream；domain prior 缺失封顶 | **maybe**(skeptic) / **pursue**(evidence) | 0.7 / 0.62 |
| **1.1** Action-as-modality joint diffusion (Cosmos/UWM 统一目标) | 有条件是。weight-sharing 是真杠杆；但 independent-timestep（其卖点）**gdn_ar.py 已实现且已被本项目否为 0% 成因**（finding #5） | 中-低。repo 无 shared-weight 路径（现有 joint_self_attn 是双专家 MoT）；需新 token-injection + GDN-cache plumbing，是数周非"1-2 周" | 中。Cosmos ablation：joint 目标仅 +1.5pt 直接策略；增益来自 backbone+planning | 部署 marginalization 再解耦；忠实跟随坏 dream | **maybe**(两评审一致) | 0.62 / 0.72 |
| **1.0/2.3** Joint-Denoising MoT trunk (AR 流式版 action-in-sequence) | 是（结构最强）——但 evidence 指出这是**最吃数据多样性**的家族（robustness study + Fast-WAM：低多样性下鲁棒性崩塌），恰是 <200-demo 最差场景 | 中。`SanaARMoTJointDriver` 已存在且单测过 kernel parity，但 action_self_attn_weight rebalance 在 AR 变体被丢弃且曾致 NaN；context cross-attn 逃逸口仍在 | 中（破 0%）/低-中（达 <200-demo 泛化）。74.2% 的 LingBot 用的是 IDM 不是 joint-denoise | gate_msa→0 + 未门控 proprio/text cross-attn 再塌缩；linear_relu 低秩 blur | **maybe**(两评审一致) | 0.72 / 0.62 |

### C. 新颖方案 (novel)

| ID / 方案 | 根因契合 | 可行性 | 预期影响 | 关键风险 | 复核结论 | 置信度 |
|---|---|---|---|---|---|---|
| **2.0** GDN-State Readout Decoder | 见 A 组 0.3（合并评） | 低 | 中（有条件） | 同 0.3 | **maybe** | — |
| **2.1** Forward-Model Predictive-Coding Coupling (test-time active inference) | 否（核心论断错）。"常量动作无法解释变化的 latent transition"在本架构**为假**——GDN cache 已解释几乎全部 transition，零初始化 FiLM 边际作用≈0，∂L_pc/∂a≈0 → 写侧塌缩 | 高(LOC)/低(信号)。需 action→video write-back（当前缺）+ test-time inner loop；2x 训练成本 | 低（train-time aux）/中但高方差（test-time，**前提 dream 好**，但现在 -4dB）| oracle GT-future 已证使动作变差 → test-time corrector 前提被证伪；future-GT leakage | **maybe**(两评审一致，仅 test-time 半边新颖且须先探针) | 0.68 / 0.62 |

---

## 3. TOP-3 推荐

### 🥇 第一名：`2.2` Latent-Transition Inverse Dynamics（IDM-aux，配 path-forcing + proprio 撤离）

**理由**：这是唯一"动作目标在信息论上无法从常量/mean-pool 解出"的现行架构改动，直接反转已测得的 1.00x 塌缩。它复用已 materialized 的 clean state 和已有 `idm` scaffold，工程量最低、最快能测耦合假设。发表证据最干净且最对口——WAM (arXiv:2603.28955) 在 DreamerV2 上同款 IDM-from-latent-transition 给出 BC 59.4%→71.2%、PPO 79.8%→92.8%；Sensorimotor-WM (arXiv:2606.20104) 证明它在最复杂 3D manipulation 上 margin 最大。

**为何胜过其他**：相比 0.2/0.3/1.3（把动作留作可选 reader，gate 逃逸口原封不动），IDM 目标**结构性**堵死常量解；相比 1.0/1.1/2.3（需重写 GDN 算子或新 token-injection，数周且与 in-place cache 约束冲突），它无新 backbone graph 边、无 write-back、可用 plain MLP over state-delta 绕开 kernel 对齐。

**最便宜的首验证步（不训练，~1 GPU-hr）**：在现有 GDN-AR checkpoint 上 capture 连续 chunk 的 clean S_{c-1}/S_c，拟合三个小回归器预测驱动该 transition 的 action：**(A) 仅 GDN latent delta，(B) 仅 proprio_{c-1},proprio_c，(C) 两者**。判据：若 B 单独已解释得和 C 一样好（EEF action 极可能如此），则 IDM 目标可被 proprio 解出、必然再塌缩——**直接砍掉或强制撤 proprio**；只有当 A 显著优于 B 时才值得投入训练。这同时是 H1（解耦可修）vs H2（表征致命）的分离器。

### 🥈 第二名：`1.0 / 2.3` Joint-Denoising MoT trunk（先验证 `SanaMoTJointDriver` 是否真能跑 GDN）

**理由**：结构上最强地消灭"可选 bridge"——动作 token 进入共享 GDN 注意力，输出是共享 S_kv 的直接 readout。两位 evidence 评审都确认 repo 里 `SanaARMoTJointDriver` 已实现并通过 kernel-parity 单测，与 skeptic"GDN 不支持、需重写"的判断**直接冲突**——这个冲突本身必须先用代码解决。

**为何排第二而非第一**：(a) 两份评审对可行性给出截然相反的代码事实，存在未解的不确定性；(b) joint-denoising 是**最吃数据多样性**的家族，<200-demo 是其最弱场景（robustness study arXiv:2603.22078 + Fast-WAM arXiv:2603.16666 均报告低多样性下鲁棒性崩塌）；(c) 残留 gate_msa→0 + 未门控的 proprio/text context cross-attn 逃逸口。

**最便宜的首验证步**：先解决代码冲突——`grep` 确认 `SanaMoTJointDriver` 对 gdn kernel 是 hard-reject（sana_mot_driver.py:141-151）还是已有 cumsum dual-track GDN 路径（sana_ar_mot_driver.py）。若后者为真，在**全序列 BD 路径**（`run_joint_loop` 已工作，无需 AR-cache 工程）跑一个短训练，做"zero video keys → action 必须崩"的耦合测试 + 监控 action 行 gate_msa 是否衰减到 0。只有 BD joint 开环耦合超过 cross-attn 时才投 AR-cache 的 300-500 LOC。

### 🥉 第三名：`0.3 / 2.0` GDN-State Readout（仅在第一名探针证明 H2 可过后）

**理由**：方向最正确——S_kv/S_z 字面就是世界模型的"东西在哪"递归记忆，是唯一**结构上不可 mean-pool 塌缩**的 KV 源，且是 SANA-WM GDN 形式独有的杠杆。四份评审一致 maybe。

**为何只排第三**：grad 路径是真障碍——监督 pass 跑 `save_kv_cache=False` 返回 None，streaming state 来自 raw Triton 无 autograd grad_fn，把它接进训练循环是 kernel 集成项目而非 glue（skeptic 0.3 用源码证实）。不改 loss 则 proprio 仍近充分、软性再塌缩。

**最便宜的首验证步**：读 slot 6 `_SLOT_TYPE_FLAG` 只选 GDN 块，dump 已 cached（detached）的 S_kv，冻结 backbone 拟合 `S_kv@W_r + diag → EEF pose/action`，对照同样探 post-block x。若 pose 不可从 S_kv 解出 → 整个方向廉价毙掉（确认 H2 致命）；若可解且 chunk-to-chunk 有变化 → 才值 kernel/grad 投入。

---

## 4. 明确淘汰 (drop)

- **`0.1` Action-as-Latent-Frame**：使能前提（"GDN linear-attn 支持变长 token"）被源码证伪——`_reshape_to_temporal` 硬要求 N==T*S，append token 要么崩 reshape 要么跑独立 per-spatial recurrence 完全不耦合；真实成本是重写算子（须保持 pretrained 权重可加载且 video rollout 不变），远超声称的 LOC。`2.3` 是它的"做对版本"，应直接走 2.3。
- **`0.2` DC-restoring bridge（standalone）**：两评审一致认定误诊——塌缩在 V 的**下游**（o-proj/gate_ca/均匀注意力），禁掉一个退化输入统计量无法制造输出依赖，且 RMSNorm 会重新引入当初用 LayerNorm 压制的 1e9 量级。仅可作为更强方案的免费 falsifier 探针，不作独立修复。
- **`2.1` Forward-Model Predictive-Coding（train-time 半边）**：核心论断"常量动作无法解释变化 latent"在本架构为假——GDN cache 已解释几乎全部 transition，零初始化 FiLM ∂L_pc/∂a≈0，写侧塌缩；且 oracle GT-future 已证使动作**变差**，从根上证伪 test-time inverse-corrector 前提。仅 test-time active-inference 半边新颖，须先过两个不训练探针——但优先级低于 TOP-3。

> 注：所有 maybe 方案共享同一上游天花板——**SANA-WM 在该 domain dream 退化到 -4dB（H2），oracle clean-future 反使动作变差 1.2-1.3x**。无论选哪个耦合机制，落地前都应先跑"latent 是否含动作可解内容"的 linear-probe（即 TOP-3 各自的首验证步）。若 probe 普遍失败，则真正瓶颈是 backbone 表征/domain prior（H2），任何"强迫动作读 latent"都只是耦合到无用信号——届时应转向官方 randomized_500 多任务数据 + 改善 video objective，而非继续在动作头上做文章。