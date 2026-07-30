# 从 ~44% 平台冲向 74-89%：SANA+MoT 的可执行路线裁决

## 1. 一句话结论

**保留 SANA 主干的前提下，唯一指向 leader 区间（74-89%）的路径是把"重建-only 的 VAE"换成 dynamics-aware 的语义+latent-action tokenizer（RepViTok-SANA，route 2.0）——它直接攻击两个 probe 已确认的 H2 representation ceiling（latent 携带 ~0 边际 action 信息）**；但在花掉这笔最贵的 new-tokenizer-train 之前，必须先跑 **~1 GPU-hr、零训练的语义-latent 可解码性 gate**。在 tokenizer 之前，唯一值得"现在就做"的廉价升力是 **receding-horizon + temporal ensembling（route 0.0，零代码、零重训）** 和 **multi-task scale-up（route 1.0，数据轴、与饱和的 attention 杠杆正交）**。

---

## 2. 三大方向分组（裁决表）

### A. Inference（推理侧，最便宜）

| Route | 真能破 44%? | 可行性 / 重训成本 | 预期增益 | 关键证据 | 复核结论 | 置信度 |
|---|---|---|---|---|---|---|
| **0.0 Receding-horizon + temporal ensemble**（丢弃 greedy 全 chunk 消费） | **可能**（两审一致：改变 real-obs 重锚频率，非 action-head 重参数化，decoupling 不适用） | **极高，零代码零重训**（execute_horizon/temporal_ensemble 已 wired） | +3~15pp，中等置信 | PACE(2606.00537)、FFDC-WAM(2605.06222 45→80)、ACT 时序集成 +3.3% | **pursue-now**（两审一致） | 0.55 / 0.72 |
| **0.1 Asymmetric / decoupled denoise schedule** | **基本不能**（两审一致偏否） | 高（config-only），但 flash 臂对 sync-trained ckpt 是 OOD | ~0pp | DreamZero-Flash 是**降延迟**杠杆非升力；MotuBrain 解耦"sub-percent" | skeptic: **later** / evidence: **drop**（轻微分歧，净偏 drop） | 0.78 / 0.82 |
| **0.2 / 2.2 Real-obs KV feedback（block-AR engine）** | **否**（关键事实：同款 real-obs-KV 引擎在本仓 GDN-AR 已 eval 0/48、0/11，drift 未被真实像素拉回） | 高（引擎已建，CPU 等价测试过） | ≈0pp（H2 representation ceiling 封顶） | bridge-ablation H1（shuffle/mean=1.00x）、idm-gate、wm-video-quality-probe 0/11 | **drop**（evidence 审 0.83；2.2 两审均 drop 0.82-0.85） | — |
| **0.3 Video-stream CFG（补 SANA CFG 路径）** | **否**（两审一致） | 中（实为 config-only，路线误判为缺代码；SANA CFG 其实已 wired） | ~0pp，且 RoboTwin-Hard 高 CFG 有害 | RepWAM：action 不需 CFG、高 video CFG 有害；本仓 action 读 dream 为常量 | **drop**（两审，0.82） | 0.78 / 0.82 |

**分歧标注**：0.0 两位都 pursue-now，但 skeptic 强调"单帧视觉历史 + -4dB dream 可能让增益归零"，evidence 审更乐观（引 PACE/FFDC）。0.1 唯一分歧：skeptic 留作 later（保留 sync denoise_steps 10→20→30 一臂），evidence 直接 drop。

### B. Training-data（训练/数据侧）

| Route | 真能破 44%? | 可行性 / 重训成本 | 预期增益 | 关键证据 | 复核结论 | 置信度 |
|---|---|---|---|---|---|---|
| **1.0 Multi-task scale-up（全 50 任务 + randomized_500）** | **能**（两审一致：数据分布轴，正交于饱和的 attention 杠杆；directly 打 covariate-shift） | **中，full-retrain**（本程序最大 GPU 开销，~27.5K traj × 50K steps） | +10~30pp（数据轴全未测）；但单凭数据到 leader 区低置信 | RoboTwin2.0 论文 sim2real 9→42%(+33pp)、LingBot-VA 92.9、所有 leader 都 multi-task | skeptic: **later** / evidence: **pursue-now**（净偏 pursue，先做 5-8 任务子集探针） | 0.55 / 0.70 |
| **1.2 IDM aux / executability loss** | **有条件**（仅 shared-trunk + proprio-dropout 形态；bolt-on head 会重蹈 bridge collapse） | 中（finetune + 新 head，gradient 必须流进 backbone） | 双峰：+5~15pp 或撞 H2 归零 | GigaWorld-Policy(2603.17240) RoboTwin 闭环 +7%、Seer/PIDM +13~21% | skeptic: **drop**（gate 已失败 0.78）/ evidence: **pursue-now**（gradient-flow 测试，0.62） | **强分歧** |
| **1.3 Scheduled-sampling / DART noisy-obs** | **弱**（AR 路径默认已开 noisy_cond_prob=0.5；R0 噪声曾 0/20） | 中（cheap 噪声形 finetune；on-policy rollout 形需循环内 forward） | +3~10pp（已激活，边际 headroom 未知） | DART 49→79%、LingBot Noisy History Aug；但 DreamZero-Flash 是延迟杠杆 | **later**（两审一致，0.62-0.70） | 0.62 / 0.70 |
| **1.1 Protect / up-weight lambda_video** | **否**（两审：本仓 lambda_video 已=1.0，video loss 已审计 plateau 于 0.48；video-dominant C1 闭环仍 0/48） | 高（finetune，最便宜） | ~0pp | FastWAM 是"有无 co-train"非"加权"；Robometer 均匀(1,1)最优；LTX2 latent 封顶 | skeptic: **drop**(0.85) / evidence: **later**（先免费审计 loss 曲线，0.78） | — |

**分歧标注**：1.2 是最大分歧——skeptic 认为 IDM-decodability gate 已失败、且本仓已有 `dual_system_idm` 模块、机制等于 drop；evidence 审区分"静态 frozen-rep 探针失败"vs"gradient 能否让 rep 变可解码"这个未答的问题，加 proprio-dropout 后给 pursue-now。**采纳折中：先做 evidence 审提出的 1-GPU-hr gradient-flow 测试再决定。**

### C. Architecture / Tokenizer（最贵、天花板最高）

| Route | 真能破 44%? | 可行性 / 重训成本 | 预期增益 | 关键证据 | 复核结论 | 置信度 |
|---|---|---|---|---|---|---|
| **2.0 RepViTok-SANA tokenizer retrofit** | **最可能**（两审一致：唯一改 latent substrate=被 probe 确认的真天花板，正交且上游于全部饱和子设计） | **低可行性，new-tokenizer-train**（最贵；且 latent 几何契约 t_lat 整除 + patch grid 必须匹配，否则废掉 SANA 预训练 → 很可能要 re-adapt SANA） | +4~8pp 现实估计（RepWAM 全栈从头是 +8.6/+7.1） | RepWAM(2606.13674) +8.6/+7.1；Reconstruction-or-Semantics(2605.06388) | **later**（两审一致，0.60-0.62）——**gate 通过才投钱** | 0.62 / 0.60 |
| **2.1 DreamZero joint video+action flow-matching co-train** | **分歧**（skeptic：joint denoise 已是 44% 配置的现状、非 TODO；evidence：本仓 SANA-WM 配置实为 lambda_video=0+frozen，即从未真正开过） | 中-高，full-retrain | skeptic ~0 / evidence +10~20pp（需配 multi-task） | FastWAM 去 video-cotrain −8pt；DreamZero 2x 泛化 | skeptic: **drop**(0.82) / evidence: **pursue-now**(0.60) | **强分歧** |
| **2.3 Bidirectional video↔action 耦合** | **否**（CoVAR 证明低数据下对称 cross-attn 受损至 0.32；AR>BD for 闭环；evidence 审还指出 schedule.py 这个"廉价推理 seam"在 openwam 不存在） | 低-中（声称 inference-only 错误，需先建 scheduler） | 0~小正 / 中性偏负 | MotuBrain 95.8（但靠 AR rollout 非对称耦合）、CoVAR(2512.16023)、LingBot 因果论 | skeptic: **later**(0.72) / evidence: **drop**(0.66) | — |

**分歧标注**：2.1 的分歧根源是**两位审稿看的是不同代码栈**——skeptic 读 openwam `base.py` 确认 joint denoise + lambda_video=1 已是 44% 现状；evidence 读 sana-wm 的 `train_sana_wm_gdn_*.yaml` 确认那边 lambda_video=0.0 + frozen backbone。**结论取决于"44% 平台"具体跑在哪个配置**：若是 openwam MoT（lambda_video=1），2.1 是已运行的饱和杠杆→drop；若要在 sana-wm 栈复现，2.1 是真未开过的杠杆。按本项目 ESTABLISHED 约束，破 0% 的是 openwam SANA-Video-2B MoT 配置，故 **2.1 在主战栈上偏 drop**。

---

## 3. 推荐执行顺序（LADDER，每级都 ≥100 闭环 eps）

```
LEVEL 0  零重训推理 A/B（天/GPU-hr 级，先做）
  └─ 0.0 execute_horizon=6, temporal_ensemble=false 单点 vs greedy baseline，≥100 eps
       通过(>44% p<0.05) → sweep K∈{4,8,12} × ensemble_decay∈{0.3,0.5,0.7}
       预期累积：44% → 47~55%
  └─（并行，免费）审计 44% 跑的 per-stream loss_video 曲线 + 重确认 SANA CFG 已 wired
       —— 用来证伪 1.1，决定是否丢弃 video-weight 路线

GATE A  ~1 GPU-hr 零训练语义-latent 可解码性探针（决定 2.0 投不投钱）
  └─ idm_decodability_probe.py 喂 DINOv2/SigLIP-2/V-JEPA-2 语义 latent，
     比较边际 action-R²（尤其 grasp）vs LTX2 ~+0.13 基线 vs proprio anchor
     额外跑 proprio-corrupted（OOD）变体——闭环真正失败的 regime
     通过(语义 latent 边际 action 内容 >> +0.13) → 解锁 LEVEL 3

LEVEL 1  数据轴子集探针（中等 GPU，full-retrain 前 de-risk）
  └─ 1.0 在 5-8 任务 train_tasks + variant=both 短预算训练，
       在 in-distribution 任务 ≥100 eps/任务
       不动平台 → 证伪覆盖假说，省下全 50 任务大开销
       有升 → scale 到全 50 任务（预期 → 55~70%+，但 viewpoint/init-state 维度仍弱）

LEVEL 2  gradient-flow IDM 测试（1 GPU-hr，决定 1.2）
  └─ 44% ckpt 上挂 IDM over predicted-frame latent deltas，proprio REMOVED，
       unfreeze backbone 训几百步，看 IDM action-MAE 是否跌破 proprio-only 基线
       分离 → greenlight shared-trunk + proprio-dropout finetune
       ≈proprio-only → H2 ceiling 确认，drop

LEVEL 3  Tokenizer 投资（最贵，仅 GATE A 通过后）
  └─ 2.0 RepViTok-SANA：stage-1 recon+semantic → stage-2 latent-action，
       保 t_lat=1+(vnf-1)//4 整除 + patch grid 匹配，大概率连带 re-adapt SANA DiT
       预期：+4~8pp（现实），叠加前面后有望进 leader 下沿
```

**累积预期路径**：44% →(0.0)→ ~50% →(1.0 multi-task)→ ~60-70% →(2.0 tokenizer，gate 通过)→ 趋近 74%+。注意 leader 的 89% 还叠了 14B/internet-scale 预训练，本栈 2B + RoboTwin 数据不可能纯靠 recipe 复现到顶。

**测量纪律**：永远 ≥100 闭环 eps，固定 standard seeds。备忘录已记录 n=25 摆动 ±15pp 并**伪造过一次 52%**——任何 n=25 的"突破"一律不信。

---

## 4. TOP-3 + 各自最便宜的首测

1. **Route 0.0 Receding-horizon + temporal ensemble** — 唯一两审一致 pursue-now 的零成本杠杆。
   **首测**：现有 44% MoT ckpt，`execute_horizon=6, temporal_ensemble=false` 单点 vs greedy，≥100 eps。**ensemble 先关**，隔离"高频 real-obs 重锚单独是否有效"这一 load-bearing 主张；过了再调平滑。零代码零重训。

2. **Route 2.0 RepViTok-SANA tokenizer** — 唯一指向 leader 区间、且直击已确认 H2 天花板的杠杆。
   **首测（GATE A，零训练 ~1 GPU-hr）**：复用现有 `idm_decodability_probe.py` harness，把输入换成冻结 DINOv2/SigLIP-2 语义 latent（同一批 RoboTwin clean_50 teacher-forced chunks），看边际 action-R²（尤其 grasp）是否明显 > LTX2 的 +0.13 重建-latent 天花板。**务必同时跑 proprio-corrupted 变体**。不过 gate 不建 tokenizer、不重训 DiT。

3. **Route 1.0 Multi-task scale-up** — 数据分布轴，与全部饱和的 attention 子设计正交，所有 leader 的共同底座。
   **首测**：不要直接上全 50 任务 full-retrain（本程序最大不可逆 GPU 账单）。先用同 MoT+renorm-0.5 recipe，`train_tasks=[adjust_bottle, lift_pot, +~6]`、`variant=both`、短预算，然后**专门在 adjust_bottle 上 ≥100 eps**。若被 multi-task+randomized 覆盖包围后 adjust_bottle 仍不离 44% 平台，则覆盖假说不成立，全 50 任务开销不予批准。

---

## 5. 明确淘汰（饱和 / 装饰性杠杆）

- **0.2 / 2.2 Real-obs KV feedback（独立部署修复）** — 同款 real-obs-KV 引擎在本仓 GDN-AR 已 eval **0/48、0/11**，真实像素喂回 cache 未把 drifted state 拉回；H2 representation ceiling 封顶。机制非新，是已证伪实验的重新包装。
- **0.3 Video-stream CFG** — action head 实测把 dream 读成 content-free 常量（shuffle/mean=1.00x），锐化一个被平均掉的信号增益为零；且 RepWAM 证 video CFG 在 Hard 上有害。SANA CFG 实际已 wired，路线对成本判断也错。
- **0.1 Asymmetric denoise schedule** — DreamZero-Flash/MotuBrain 证据均为**降延迟**而非升成功率；更多 action step 只把已 <0.05 的 teacher-forcing MSE 再压低，正是 decoupled 轴。（可保留 sync denoise_steps 10→20→30 单臂作低优先复确认。）
- **1.1 Up-weight lambda_video** — 本仓已 lambda_video=1.0、video loss 已审计 plateau 于 ~0.48，video-dominant C1 闭环仍 0/48；FastWAM 的 −8pt 是"有无 co-train"非"加权"，Robometer 证均匀(1,1)已最优。（免费的 loss-曲线审计仍值得做一次以正式证伪。）
- **2.3 Bidirectional video↔action 对称耦合** — CoVAR 证低数据下对称 cross-attn 掉到 0.32；三大 AR leader 均论证 AR>BD for 闭环，会放大已视觉确认的 compounding drift；其宣称的"inference-only 廉价 seam"（schedule.py/make_schedule）在 openwam **并不存在**，需先建 scheduler。
- **2.1 DreamZero joint co-train（在主战 openwam MoT 栈上）** — `base.py compute_loss` 已对 video+action 双流加噪+联合去噪、lambda_video=1.0 即 44% 现状；唯一真未试的 Beta high-noise 解耦采样器也已实现且属饱和 recipe 杠杆同族。（仅当目标是在 sana-wm lambda_video=0/frozen 栈复现时才算未开过的杠杆。）