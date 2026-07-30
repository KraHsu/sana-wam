# 为什么 Wan 在 openwam 能工作而 SANA 在 sana-wam 失败 — 决定性报告

## 1. 一句话结论

在 backbone 特征质量**已被证明等价**的前提下，works-vs-fails 的根因是**耦合拓扑 + 训练目标的组合**，而非 deploy 或数据：openwam 用**强制的、不可旁路的 per-layer 联合注意力（MoT joint self-attn）+ 可训练 backbone + joint video loss**，让 action head 在结构上**必须**消费 video dynamics；sana-wam 用**可门控、可分离的 cross-attn bridge + 冻结 backbone + `lambda_video=0`**，于是 SGD 找到了「忽略 bridge、靠 proprio 自洽」的最省力解——bridge 坍缩成 content-free 常量，闭环时无视觉反馈纠正漂移，必然 0%。这是一个 **coupling + objective** 问题，deploy 侧的差异都是次要放大器。

---

## 2. TOP differences（按嫌疑排序）

### #1 [top-suspect] 耦合拓扑：强制 MoT joint self-attn vs 可门控 cross-attn bridge

- **openwam 做 X**：每一层把 video+action 的 Q/K/V 拼接后跑**一次** SDPA（`mot_driver.py:296-312`），`a→v` mask 硬编码为 `True`（`mot_driver.py:153`）。action expert 的 self-attention **就是**那次联合注意力——`SelfAttnActionDiTBlock` 只有 `gate_msa`/`gate_mlp`，没有单独的 video 旁路门（`joint_action_dit.py:257-325`）。
- **sana-wam 做 Y**：action 先对 action token 自注意力，再走一条**独立的门控残差** `x_action = x_action + gate_ca * cross_attn(h, x_video)`（`joint_action_dit.py:274-275`），`gate_ca` 是 9 个 AdaLN 调制参数之一。
- **失效机制**：因为 action 有一条完整自洽的 action-only 自注意力 + proprio-context + FFN 通路，SGD 只要把 `gate_ca→0` 或让 `cross_attn→常量`，就能在零代价下**彻底切断 video**，仍最小化 teacher-forcing action MSE。MoT 里没有这个逃生口——不读 video 就等于丢掉自己的时序自注意力（同一个算子）。这正是 memory 记录的 bridge 坍缩（shuffle/mean = 1.00x）。
- **为何能通过「特征等价」过滤**：特征等价是**前提而非反驳**——等价的特征若 readout 从不被强制消费就毫无用处。cross-attn 从不强制 head 去读这些等价特征，MoT 强制。这正是 H1（坍缩）/ H2（表征上限）的分水岭，由拓扑决定落在哪个 regime。
- **与发现一致性**：与 bridge-collapse、covariate-shift-drift 强一致；**决定性旁证**——同一个 SANA backbone 被移植进 openwam 的强制 MoT（`SanaMoTJointDriver`），同一份 RoboTwin，从 0% → 32%（重训）→ 44%（per-modality renorm）。同特征、同数据，只改耦合就破了 0%。
- **FIX**：把可门控 bridge 换成强制耦合。GDN 的逐帧递归确实禁止拼接式 joint self-attn（`cross_attn.py:6-9`），所以最干净的路是**复用已建好的 `SanaMoTJointDriver`**（非 GDN 的 SANA video backbone）。若必须保留 GDN cross-attn，则至少：(a) **删掉 `gate_ca`**（改成 ungated，像 `context_attn` 那样），(b) 让 bridge cross-attn 成为 action 流的**主注意力**（弱化/去掉 action-only self-attn），消除 proprio-only 捷径。

### #2 [top-suspect] backbone 可训练性 + video loss：trained joint vs frozen + `lambda_video=0`

- **openwam 做 X**：Wan DiT **被训练**（`joint.yaml` 不冻结 `_pipe.dit`，注释明确「intentionally TRAINED」），`lambda_video=1.0`、`lambda_action=1.0`——producer 与 consumer 同时被梯度塑形，且耦合结构上不可旁路。
- **sana-wam 做 Y**：`freeze:[video_backbone.dit, video_backbone._pipe.vae]`（`train_sana_wm_gdn_ar.yaml:152`）、`lambda_video=0.0`（:147）、`init_dit_from` 预训练 `.pt`（:58）。只有 bridge + proprio token 在学。
- **失效机制**：冻结 producer + 零 video 梯度 ⇒ bridge 的 KV 源是固定、永不为任务适配的特征；唯一拟合 action 的途径是让 action head 包揽一切，叠加可门控 bridge 直接导致坍缩。openwam 里 video 流本身被梯度塑形成 action-predictive，**且** action 被 MoT 强制消费它——两股压力共同维持耦合存活。
- **为何能通过「特征等价」过滤**：这是**可训练性/co-adaptation**问题，与冻结特征的**质量**正交。特征离线等价 ≠ 闭环可用：一方 co-train producer+consumer，另一方冻结 producer 又让 consumer detach。
- **关键 caveat**：IDM-decodability gate 在**冻结**表征上测得 ~0 边际 action 信息，指向可能的 H2 上限——但该 gate 恰恰测的就是被本项指控的冻结表征，并未测「解冻 + joint flow-matching」的反事实。所以这是 caveat 而非反驳；先验上 #2 是必要不充分。
- **FIX**：解冻 SANA DiT 的**后 N 个 GDN block**（或对 q/k/v/o/ffn 做 LoRA，规避 8×SANA host-RAM OOM），并设 `lambda_video=0.5~1.0`，使 backbone 在 joint flow-matching 下 co-adapt。不要继续维持「全冻结 + lambda_video=0」。

### #3 [top-suspect] deploy 时间索引：纯计数器 cache index vs 真实滑动窗 obs

- **openwam 做 X**：`JointInferenceEngine.generate` **跨调用无状态**——每次用当前真实 `first_frame_image`+`proprio_state` 从头重建 chunk，无时间索引 KV cache、无 step 计数器（`engine.py:344-461`）。`execute_horizon` 与正确性无关。
- **sana-wam 做 Y**：`GDNARInferenceEngine` 持有单调 `self._step_c`，把每个 ingested obs band 标注为绝对 latent 帧 `[(c-1)*K, c*K)`（`gdn_ar_engine.py:176,237`），index 每次 generate 恒定 +K=3 **纯计数**；而 obs band 本身取自在 NOW 重锚的滑动窗（`_select_obs_band:316-358`）。
- **失效机制**：两次 generate 间内容前进 ~atc≈22 raw ≈ 2.75 latent 帧，cache index 却前进 K=3——content-time 与 index-time **逐 chunk 漂移**，ingest 了「错时」的观测，在 RoboTwin 100+ 步的 episode 上累积成闭环崩溃。`execute_horizon=24` 还是 dead config（atc≈22 < 24，buffer-empty 分支永远先触发），cadence 不受控。
- **为何能通过「特征等价」过滤**：纯 deploy/framework 判别项，与 backbone 质量无关——正是「特征等价」要求的答案所在地。
- **与发现一致性**：与 covariate-shift-drift 直接一致；错时 cache 也会让 head 学会「忽略不可靠的 cache 内容、坍缩到 proprio」，与 bridge-collapse 下游一致。
- **FIX**：采用 openwam 的无状态契约——每次 generate 从**空 cache** 重建，用 `obs_history` 重新 ingest 真实历史 chunk 0..c-1（`CrossAttnInferenceEngine` 已是此模式，移植过来）；或至少加硬断言把 `_step_c` 绑定到真实执行步数，并令 `execute_horizon == fcs*tc*video_stride == atc`、只在 chunk 边界 regenerate。

### #4 [contributing] 训练 timestep 耦合：独立采样 vs 共享 `u`

- **openwam**：video/action 各自独立 `randint`（`base.py:1427,1483`），deploy 任意配对都在分布内（`denoise_schedule.py:43-47`）。
- **sana-wam**：单一共享 `u` 映入两个 scheduler（`gdn_ar.py:350-352`，「Fix A」），action 只在**匹配噪声级**读 video。
- **机制**：独立采样让 action head 见过 (video-noise, action-noise) 全笛卡尔积，对 deploy 时退化/高噪 video bridge 鲁棒；共享 `u` 只训练对角线，强化「bridge 总是同一噪声级 → 忽略它」。
- **强 caveat（压低排名）**：独立采样**是 sana-wam 的原始行为**，那时**已经 0%**——所以 0% 早于该耦合，回退它返回的是已知失败态，必要不充分。叠加 IDM gate 的 ~0 边际信息，oracle clean dream 也无助，更削弱本项。
- **FIX**：解耦回独立采样（一行改动、与 working recipe 对齐、~零成本），但**别**当作主修复。

### #5 [contributing] cross-attn-to-frozen 的目标级组合

#1 与 #2 的交互体现：cross-attn 本身不坏（FastWAM 用它且 work），但 **cross-attn → 冻结 backbone + coupled-timestep + action-only loss** 的组合让 head 唯一梯度路径通向固定表征，且能零代价旁路。FIX：恢复一条惩罚 bridge-bypass 的梯度路径——解冻被 bridge 的 video 层（让 video loss 依赖同一特征）、加 IDM-aux 强制表征 action-decodable、或 proprio-dropout。注意 proprio-dropout 单独已 0/20，故需与解冻/IDM-aux 同时上。

### #6 [contributing] VAE 几何强制的时序 token 粒度

LTX2 8× temporal + `video_stride=1`（为补偿 tc 4→8 而减半）让每个 action-DiT 所读 latent 帧汇总 8 raw 帧（Wan 为 4），并把整个「每步推进一个 clean-prefix latent」的闭环契约重建在 LTX2 几何上、改变 deploy 重观测 cadence（≈8 vs 16）。这是**使能/contributing** 因素（更粗的 per-step 学习信号可能加速向 proprio 捷径坍缩），但无证据它是坍缩的**原因**而非共现。FIX：仅作为 A/B（试 `video_stride=2~4`，数据集会自动 snap），优先级低于 #1/#2/#3。

---

## 3. 关于上次 SANA-in-openwam 失败

prior-attempt mapper 发现：那次集成**非常完整**（registry + config + `SanaMoTJointDriver` + AR deploy engine + RoboTwin eval，89–98 测试绿），甚至有一条名为 `feat/why_sana_is_bad` 的分支专门诊断。它**用了正确的强制 MoT 耦合（#1 的修复方向）**，但栽在两个**不同于当前 sana-wam** 的问题上：

1. **欠训练**：host-RAM 在加载 8×~10.8GB SANA ckpt 时 OOM，正式 run 被限制到 2 GPU、200 步、eff. batch 4 → action loss 0.32（cosmos 0.16）。正确重训（1000 步、eff. batch 16）把 teacher-forcing action loss 0.322→0.056、闭环 0%→32%。
2. **linear-attn token dilution**：SANA 的 ReLU-kernel 线性注意力在把几千 video token 与少量 action token 拼接时，归一化分母被 video keys 主导，action 读出被稀释——用 `action_self_attn_weight` per-modality 行重归一化修复，32%→44%。

**结论**：上次「something was wrong」的根因是 **#1 的正确实现遇到了 (a) 欠训练 + (b) 线性注意力 token 稀释**，而**不是**当前 sana-wam 的 cross-attn 坍缩。证据价值极高——**换上强制 MoT 耦合就把同一 SANA 从 0% 抬到 44%**，直接坐实 #1 是最强结构杠杆。但它停在 ~44%（非 openwam ~80% 级），说明耦合拓扑是**主**判别项而非**唯一**：剩余 gap 由训练预算/数据/recipe 承担。

---

## 4. 推荐的最小行动（先测最便宜的 1–2 个）

**首选（最高期望、已有 44% 先例）— 重启 SANA-in-openwam 的 MoT 路径，带上两个已知修复：**
1. 用 `SanaMoTJointDriver`（非 GDN 的 SANA video backbone）+ `joint_self_attn` 强制耦合；
2. 带上 prior-attempt 的 **(a) per-modality 行重归一化（token-dilution fix）** 与 **(b) 充足 eff. batch + 步数**（解决 host-RAM OOM，用 LoRA/少 GPU 也要把 eff. batch 拉到 ~16、≥1000 步）。
- **测什么**：单任务 `adjust_bottle`/`lift_pot` 闭环成功率，目标**先复现 ≥44%**（证明 pipeline 可用），再调 renorm α / lambda_action 往上推。这是唯一已被证明破 0% 的配置。

**次选（保留 GDN backbone、改 sana-wam 本身，成本更低但风险更高）：**
2. 在 `gdn_ar` 上做**三处联合改动**（单独任一无效）：
   - 删掉 `gate_ca`（`joint_action_dit.py:274` 改 ungated）并让 bridge 成主注意力（#1 fallback）；
   - 解冻后 N 个 GDN block + 设 `lambda_video>0`（#2）；
   - deploy 改无状态 obs-rebuild 或加 `_step_c` 对齐断言（#3）。
- **先测一个最便宜的诊断**：在**解冻 + lambda_video>0** 的 ckpt 上**重跑 IDM-decodability gate**（~1 GPU-hr）。这是唯一能区分 H1-可修 vs H2-致命 的实验——若解冻后表征仍 ~0 边际 action 信息，则停止 SANA-GDN 方向、全力走首选 MoT 路径或 official-data baseline。

**白送的对照**：把训练 timestep 解耦回独立采样（#4，一行）、把 deploy `denoise_steps` 4→10、`execute_horizon` 对齐到 chunk 真实跨度并断言——这些与 working recipe 对齐、近零成本，可作为 A/B 基线，但**不要**期待单独破 0%。

---

## 5. 明确排除（cosmetic / 非根因）

- **action 表示（delta vs absolute）**：两边都是 absolute + min-max[-1,1]，**当前 run 完全相同**；delta 已试 0/20。保持 `delta_action:false`，别再 relitigate。
- **proprio cadence（per-chunk vs per-window）**：openwam 也给极强 proprio 锚（per-window static、per-replan 刷新），同一捷径两边都有；提议的修复就是 proprio-dropout，已 0/20。openwam 无 proprio_dropout 仍 work ⇒ 既非必要也非充分。
- **denoise 步数（4 vs 10）**：head 已忽略 bridge，video 再清晰也传不到 action；欠去噪只均匀损精度几个百分点，不会把 80% 砸到 0%。可白送 A/B，预期仍 0%。
- **temporal ensembling（ON vs OFF）**：openwam 默认 deploy 无 `policy:` 段 ⇒ `execute_horizon=None` ⇒ ensembling 实际**不激活**，是 dead flag；且它只是 jitter 平滑，救不了 0%→80%。
- **stateless vs stateful deploy（单独看）**：`CrossAttnInferenceEngine` 已是无状态自愈契约（与 openwam 同），**仍 0%** ⇒ 无状态化必要不充分，非判别项（但 #3 的「时间索引错位」是 stateful 路径的真 bug，区别对待）。
- **deploy obs-feed 形状（单帧 vs 多帧 clip）/ obs-clip cadence 重建**：真实 0% 配置里 deploy 已从 saved dataloader cfg 读到正确的 `num_frames=145/stride=1/tc=8`，「stale 4 翻倍 num_chunks」的具体 bug **不存在**；且 head 已忽略 bridge，喂它 OOD latent 也无关。
- **VAE 空间 token 数（30 vs 480）**：算术用错 patch size——GDN-AR 实际 `patch_size=(1,1,1)`，真实为 120 token/frame（4× 而非 16×）；且 openwam 同样 384×320，token 差纯来自 VAE。spatial-preserving IDM gate 在真实 120-token 下仍测得 ~0 边际信息 ⇒ 是 latent 质量（H2）而非 token 数。
- **text encoder（Gemma 2304 vs UMT5 4096）**：喂入已饱和 IDM 指标的表征，capacity gap 被特征等价证伪；两边 text_dropout=0、cfg=1。`use_first_frame_cond=true` + KV 实观测反馈已启用，观测锚与 openwam 结构对齐。
- **action↔video 对齐比 / chunk 长度**：openwam RoboTwin 实际 commit **更长**的 32 步 chunk 且 ensembling 关闭——提议的「sana chunk 更长 → 更多漂移」与配置事实相反。
- **normalization / multi-task stats / action 转换 / multiview layout**：语义 byte-identical fork（仅重命名），全部排除——数据轴的真分歧只在 VAE-tokenizer 选择及其强制的帧采样几何（见 #6）。

---

**最终判断**：把赌注押在 **#1（强制耦合）+ #2（可训练 backbone/joint video loss）** 上，路径选**重启已验证的 SANA-MoT-in-openwam（带 token-dilution renorm + 充足训练）**，因为它是唯一**实证破过 0%（→44%）** 的配置；用解冻后的 **IDM gate 重测**作为 ~1 GPU-hr 的 go/no-go，决定继续推 SANA 还是接受 H2 上限转向 official-data + video objective。