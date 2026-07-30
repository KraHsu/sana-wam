# SANA-WM 骨干能力决策报告

## 1. 一句话结论

**最有希望的路径是「先验证、后投入」的两步走：用已有的 ~1 GPU-hr IDM-decodability probe 当闸门,先在 frozen Cosmos-Predict2-2B 特征上跑一次 probe(几乎零成本),据此在两条主线中二选一——若闸门翻转,走 swap 到 Cosmos-Predict2-2B / Wan2.2-5B(thesis REFRAMED:贡献变成 GDN-AR 流式 WAM recipe,大概率能破 0%);若你必须保住 thesis,则走 C1/0.0(domain-pretrain SANA-WM,thesis KEPT,但能否达标存在真实不确定性)。** 证据现实是:所有 58-74% 的 RoboTwin leaders(LingBot-VA、Cosmos-Policy、GigaWorld、MotuBrain)和 DreamZero 用的都是 robot/physical-AI 预训练的 Wan/Cosmos 骨干,没有一个用 from-scratch 的 camera-motion 先验骨干——这是对 SANA-WM thesis 最硬的外部压力。

---

## 2. 三大方向分组

### A. Domain-Pretrain(保 thesis)

| ID | 选项 | 真修 H2? | 可行性(compute/data/eng) | 战略契合 | 预期影响 | 关键风险 | 复核结论 | 置信 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | randomized_500 video-only stage-0 | **倾向是**(唯一从根上打 H2 又最便宜) | 高 / 本地全有 / 低(`lambda_action=0`) | **KEEP**(最可辩护形态) | medium,方差大,先验不利 | 已有证据:oracle IDM gate 已失败、+1dB dream 未动闭环 0/11、video loss 卡 0.48 floor | 两评审皆 **maybe** | 0.70/0.72 |
| 0.1 | OXE/DROID/Bridge 大规模 robot-video 续训 | **是**(literature-backed,但 capacity-gated) | 低-中 / 公开需重度 curation / 中-高 | KEEP(最强形态) | bimodal:high 或 zero | RT-X capacity 闸:2B 可能欠拟合;OXE 单臂≠双臂 aloha;1-3 GPU-周 | 两评审皆 **maybe** | 0.66/0.70 |
| 0.2 | LR-corrected 全参续训(warmup+EMA) | **否**(recipe 非 representation) | 高 / 有 / 低 | KEEP | 低(作 prerequisite/falsifier 高) | **前提已被证伪**:官方 baseline 已稳定收敛却 dream 仍卡 0.48;发散是 action-only bridge 非骨干 | 两评审皆 **maybe**(仅作便宜 falsifier) | 0.78/0.72 |
| 0.3 | multi-view 唤醒 plucker 通路 | **否**(机制被自家 probe 反驳) | 低-中 / 本地 / 中(数百 LOC,最脆弱处) | KEEP(叙事最强) | 低 | static-plucker A/B 动了 -0.0%;IDM gate 连 grasp 都无边际信息;plucker 未接线 | **两评审皆 drop** | 0.72/0.78 |
| 1.2 | RoVid-X(4M)+randomized_500 续训 | 是(原则上)/否(此骨干上) | 中 / RoVid-X 开源需重 curation / 低 | **KEEP(最强)** | 中-低,高方差 | 官方 randomized_500 已跑过 in-domain video 半程→0/48;ReVidgen 只证 video 质量非 policy | maybe / **pursue(gated)** | 0.72/0.72 |

**分歧标注**:0.0/1.2 两评审在「addresses_H2」上对立(skeptic 更乐观说「攻击根因」,evidence reviewer 援引 RepWAM 说「同一个 reconstruction VAE 续更多 video 不解决」)。共识是:**都该当便宜 gate 跑,不该当 believed fix 直接投多日 compute**。

### B. Swap(换骨干,reframe/abandon thesis)

| ID | 选项 | 真修 H2? | 可行性 | 战略契合 | 预期影响 | 关键风险 | 复核结论 | 置信 |
|---|---|---|---|---|---|---|---|---|
| 1.0 | Cosmos-Predict2(.5)-2B + 保 GDN-AR recipe | **是**(根因层,最高 ceiling) | 中 / 本地+Wan VAE 已 shim / 大(AR 全重写) | **ABANDON**(reframe 成 streaming recipe) | **high**(条件于 RoboTwin 迁移) | 无公开 Cosmos RoboTwin 数;novelty 多被 Cosmos-Policy 占;AR 从零重写 | 两评审皆 **maybe**(gate on probe) | 0.72/0.72 |
| 1.1 | Wan2.2-5B + MoT joint denoising(已验证 recipe) | **是**(field 已证) | 低-中 / 本地数据正好 / 大(5B+MoT 移植) | **ABANDON**(最弱 thesis 保留) | **HIGH**(raw Wan finetune=80.6% Easy @50 demos) | 复现风险;LingBot 已有 MoT+KV+real-obs;5B 最重 | 两评审皆 **maybe**(先跑 LingBot ckpt) | 0.78/0.85 |
| 1.3 | robot-pretrained Cosmos(action-cond Bridge) | 是(但赌错 artifact) | 中 / 本地 / 中-大+embodiment remap | ABANDON(novelty 最低) | 中-高(仅配 latent-frame recipe) | 选错 checkpoint(Cosmos-Policy 用 base WFM 非 action-cond);撞 NVIDIA 已发表 | **两评审皆 drop** | 0.72/0.78 |

**分歧标注**:1.1 evidence reviewer 置信最高(0.85)且明确指出 RepWAM from-scratch 89.3 证明「joint-DiT+IDM recipe 才是杠杆,非 Wan 权重」——这弱化了「换 Wan 骨干」本身的归因,强化了「recipe 才是贡献」的 reframe。

### C. Scale(第三轴)

| ID | 选项 | 真修 H2? | 可行性 | 战略契合 | 预期影响 | 关键风险 | 复核结论 | 置信 |
|---|---|---|---|---|---|---|---|---|
| 2.0 (C1) | 不扩参,扩 SANA 见过的 data(robot-video 续训) | **是**(打在因果变量:预训练 domain) | 中 / 本地 896G ready / 低-中 | **KEEP(最强、最可发表)** | 中-高(条件于 gate) | LR 刀锋;plucker 通路休眠(<1.8% shift,非毒化);GDN linear-attn 可能架构性欠表达 contact | maybe / **pursue(gated)** | 0.62/0.70 |
| 2.1 (C2) | 保 GDN-AR machinery,distill robot 教师进 DiT | 否(多步未证 transfer 的合取) | 低-中 / 同 C1 / 中(跨 VAE distill 是杀手) | REFRAME | 低-中 | 跨 VAE(Cosmos/Wan→LTX2)几何冲突;linear-attn 学 softmax 教师有损;优化器已知发散 | **maybe**(先在 Cosmos 教师特征上跑 IDM probe) | 0.70/(evidence 未复核) |
| 2.2 (C3) | frozen robot-video 模型当 feature extractor 喂 action head | **是**(替换 rep 源,VPP/Video2Act 已证) | 中 / VPP stage-1 需 web-scale data / 中 | **ABANDON**(SANA 沦为 scaffold) | high(若 extractor 经 manipulation finetune)/ 低(frozen generic) | head 仍有 proprio 捷径会忽略新特征;VPP 增益靠 373k 轨迹 stage-1 无法复现;撞 VPP/Video2Act | skeptic **drop** / evidence **maybe** | 0.72/0.70 |

**分歧标注**:C3 是本批最大分歧。Evidence reviewer 给出**唯一 on-benchmark 的硬数**:VPP ~44.7% / Video2Act ~54.1% on RoboTwin 2.0 @100 demos/task——直接证明「video-DiT 特征→action head」recipe 在目标 scale 破 0%。但 skeptic 正确指出:(a) 这等于把贡献塌缩成「Cosmos 当骨干 + GDN-AR 头」即 swap;(b) 不加 proprio-dropout,新特征会被同一个 mean-pool head 忽略(bridge-ablation 已证)。

---

## 3. 核心权衡(显式决策)

**leaders 实际用什么骨干(这是决策的地心引力):**
- LingBot-VA(74.2% RT2.0-Plus / ~92% Easy):**Wan2.2-5B** + MoT + IDM,robot-video 预训练;关键 ablation——raw Wan2.2-5B finetune @50 demos 就有 **80.6% Easy**(SANA-WM 同 scale 是 0%)。
- Cosmos-Policy(67-98% RoboCasa/LIBERO/ALOHA @50 demos):**Cosmos-Predict2-2B**,physical-AI 预训练,action-as-latent-frame,**无架构改动**。
- GigaWorld / MOTUS / MotuBrain(90-96%):**Wan2.2-5B**。
- DreamZero(2x 泛化):**Wan2.1-14B**,video+action 联合 flow-matching,web-scale 预训练。
- RepWAM(89.3 from scratch):证明 **joint-DiT+IDM recipe + 替换 reconstruction tokenizer** 才是杠杆,非某个具体权重。

**三轴对照:**

- **Domain-pretrain SANA-WM(KEEP thesis)**:保住「SANA-WM AS backbone」这一 stated contribution,且是其最可辩护形态。但有三条收敛的反向证据——(i) oracle IDM gate 已失败;(ii) +1dB dream 未动闭环;(iii) video loss 卡 0.48 floor——指向 H2 可能是 **capacity/architecture(GDN linear-attn + LTX2 reconstruction VAE)** 而非 coverage。2B + 仅本地 sim-only 50 任务,能否够到 58-74% 的 bar 是**真实的未知**,先验不利。RepWAM 的论点尤其刺眼:同一个 reconstruction VAE 喂更多 in-domain video 不解决问题。

- **Swap backbone(REFRAME/ABANDON thesis)**:几乎确定能破 0%(raw Wan 80.6%、Cosmos-Policy 67% 都是直接证据),但放弃「SANA-WM 是骨干」的原始 thesis,贡献必须 reframe 成「GDN-AR streaming WAM recipe,backbone-agnostic」。**风险是 novelty**:Cosmos-Policy/LingBot 已发表 full-window 与 streaming recipe,你的差异化只剩 GDN-cached-AR 流式角度——这恰恰又是最大的从零重写成本。

- **Scale(扩参)**:被 Cosmos-Policy 直接证伪为必要条件——**2B 足够大,只要先验对**(67% RoboCasa @50 demos)。所以 scale 轴的正确读数是 C1:「不扩参,扩 data」,这与 domain-pretrain 合流。真正的 scale 顾虑是 OXE/RT-X 的 capacity-gating,但只在大跨 embodiment 数据下咬人。

**决策本质**:这是一道**「保 thesis 但赌它能达标」vs「弃 thesis 但大概率达标」**的题。证据天平偏向后者;但若学术/导师约束要求 SANA-WM 框架,前者唯一负责任的走法是先用便宜 gate 测「coverage vs capacity」,失败本身就是可发表的 negative result + 转向许可。

---

## 4. TOP-2 推荐(排序)

### 🥇 #1 — 先 gate,再决定 swap 方向(默认指向 Cosmos-Predict2-2B / Wan2.2-5B)

**Rationale**:这是唯一在根因层(H2 = 骨干 representation)动手、且有 on-benchmark 硬证据(raw Wan 80.6% / Cosmos-Policy 67% @50 demos / Video2Act 54% RoboTwin)的方向。预期影响 high,是把闭环从 0% 拉到 40-70% band 的最可信路径。

**Thesis 含义**:**REFRAME / ABANDON**。贡献从「SANA-WM 是骨干」变为「streaming GDN-AR World-Action recipe on a domain-pretrained WFM」。必须诚实承认这一点,并让 novelty 由 AR/streaming + KV-real-obs-feedback 的 covariate-shift 角度承担(Cosmos-Policy 是 full-window,不做 streaming——这是真实差异点)。

**单一最便宜的首验证步骤**(不做任何训练、不写 adapter):
> 下载 frozen **Cosmos-Predict2-2B-Video2World**(以及 Wan2.2 VAE),在 RoboTwin clean_50 的 teacher-forced rollout 上跑 backbone 一步,抽 per-chunk 特征,用**现成的 `/tmp/idm_decodability_probe.py`** 拟合 A(latent)/B(proprio)/C(both) 三个回归器。**判据:Cosmos 特征的边际 action R²(尤其 grasp 子集)显著高于 SANA 的 ~0.00,且 A>>B。** 翻转→swap 正当,投 3-5 GPU-周工程;若仍 B==C→连 swap 都救不了 RoboTwin,省下数周。成本 ~1 GPU-hr + 半天 glue。

### 🥈 #2 — C1 / 0.0:video-only domain-pretrain SANA-WM(保 thesis 的最强形态)

**Rationale**:唯一完整保住 stated contribution 的路径,且 evidence reviewer 给 **pursue**(0.70)。机制对(移除 proprio 梯度捷径,逼模型建模 grasp/lift/contact 转移),外部先例对(Cosmos-Policy 对 Cosmos-Predict2 做的就是这件事)。本地 896G 数据 ready、工程低(`lambda_action=0` 一个 stage)。

**Thesis 含义**:**KEEP**(最可辩护形态)。哪怕失败也是干净的 negative result——「a 2B GDN/LTX2 backbone pretrained on camera-motion objective cannot acquire RoboTwin manipulation dynamics at 25k in-domain episodes」——这本身可发表,并为 swap 提供证据许可。

**单一最便宜的首验证步骤**(不做多日 stage-0):
> 跑 **~0.5-1 GPU-day 的短 video-only stage**(randomized_500 多任务,`video_lr=1e-5` + warmup,几百到 1-2k step,plucker conditioning 置零),然后**同时跑两个现成 probe**:(1) `wm_video_probe.py`——held-out lift_pot 的 rel-MSE 是否跌破 0.48 floor 且超过 copy-frame;(2) `idm_decodability_probe.py`——bridge 是否首次给出超过 proprio 的边际 action R²(尤其 grasp)。**二者皆动→投全量 stage-0 + joint finetune(joint 阶段务必配 proprio-dropout,否则 bridge 会在 on-distribution 重新塌缩);任一不动→H2 是 capacity ceiling,转 #1 swap。**

---

## 5. 明确淘汰

- **0.2(LR-corrected 全参续训)**:作为「fix」淘汰——前提已被官方 baseline 证伪(稳定收敛却 dream 仍卡 0.48,发散来自 action-only bridge 非骨干);只保留其作为 ~6-10 GPU-hr 的 `video_lr` sweep falsifier。
- **0.3(multi-view 唤醒 plucker)**:**drop**——static-plucker A/B 动 -0.0%、IDM gate 连 grasp 都无边际信息、plucker 在 forward_long 调用点根本未接线;数百 LOC 押在自家 probe 已反驳的假设上。
- **1.3(action-conditioned Cosmos Bridge checkpoint)**:**drop**——赌错 artifact(Cosmos-Policy 用的是 base WFM + latent-frame recipe,非 action-cond 预测 checkpoint),novelty 最低且正面撞 NVIDIA 已发表方法 + NSCLv1 noncommercial。
- **2.1(C2 跨 VAE distillation)**:作为优先项淘汰——成功是「跨 VAE 几何 + linear-attn 学 softmax + 驯服已知发散优化器 + better-dream→better-action」的脆弱多步合取,每一环本项目都有反向证据;若仍想试,先在 Cosmos 教师特征上跑 IDM probe(若连 softmax 教师 rep 都不超 proprio,C2 当场死)。
- **2.2(C3 frozen extractor)skeptic 侧淘汰**:其「能 work」的诚实版本就是「Cosmos 当骨干 + joint-denoise」即 #1 swap,独立存在时塌缩成 sub-SOTA 的 VPP 复现,且不加 proprio-dropout 新特征仍被忽略——若做,直接并入 #1 的 swap-with-joint-denoise,而非作为独立 frozen-feature 路线。