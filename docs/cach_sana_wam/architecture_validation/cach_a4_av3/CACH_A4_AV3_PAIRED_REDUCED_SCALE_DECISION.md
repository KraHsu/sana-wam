# CACH-A4 AV-3 PAIRED REDUCED-SCALE DECISION

状态：`CARD_ONLY_AV3_SOURCE_DATA_AND_EXECUTION_NOT_AUTHORIZED`

Canonical host：`H200`  
Canonical worktree：`/home/zch/workspace/sana-wam`  
日期：2026-08-05

## 1. Authority and decision

用户 exact statement 为 `继续 AV3`（UTF-8 10 bytes，SHA256
`c2e58d4f4d351b71e2e8e2dfb93c3e9741b71c9eda92011c420eff4aa8381fb1`）。上一轮交付已明确
说明该短语用于“创建并冻结 AV3 decision/run card”，因此本轮只允许排他新增并冻结本
decision 与同目录 run card，并允许只读 predecessor 引用及静态 Markdown/JSON/path/SHA
校验。

本轮不授权 source/config/runner/test、data-selection manifest 或 execution card 的创建，
不授权真实数据读取、run namespace/root、CPU/GPU 模型或测试执行、CUDA/Triton/JIT、
optimizer、parameter update、训练、checkpoint、AV-4、正式评测、admission、Global
Stage 3 或 deploy。

AV-3 的 decision 是继续验证已经得到有效 `AV2_GO` 的：

```text
CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1
```

本阶段不是架构搜索，不回到早期同名但无关的 CACH-A3 分支，也不引入 anchors、
AttnRes、self-forcing、AFCC、DAgger、rerank、best-of-N 或完整 2B。

## 2. Governing plan and eligibility

Governing plan：

- path：`docs/cach_sana_wam/global_stage3/post_phase_c_planning/ARCHITECTURE_VALIDATION_FIRST_PLAN_20260803_v1.md`
- SHA256：`06f5d09127f8bc2a945cdbdc095054f40b1fe8d9757a90eccbdf0604ea89e079`

Original AV-0 card：

- path：`docs/cach_sana_wam/architecture_validation/av1/AV0_RUN_CARD.json`
- SHA256：`c73a943047295338cd9de14e6e53199b463ea66ae92b99d4b0b3393f5ec34030`

AV-2 R2 immutable predecessor root：

```text
/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_av2_tiny_real_data_r2/06f5d09127f8/cach-a4-av2-r2-b795b1b6edba6aeb584c06db1d166bf7
```

其关键证据为：

- `RESULT.json` SHA256 `3d1ae1b81347efd258dba1d05f93f4218bb7fd7706f8b2c5a3cc3419ae2230a6`；
- `SCREEN_RESULT.json` SHA256 `3073ffa1cf2eb2f1a2d82a86d6b595f12d761f26b274a1e4b5e010a412af5a4c`；
- `DATA_EVIDENCE.json` SHA256 `6edf58dc2919b9eba6bfde7cc5565c3b8d05326076ebd906f1f6ce0c9b562892`；
- `FREEZE_RECEIPT.json` SHA256 `58d4ba15d800efc051ad4e92d3a948f2cb6f5d109c60ce683d0ac13aaf809ddc`；
- typed state `AV2_GO`、`valid_result=true`、`av3_unlocked=true`。

该结果只提供 AV-3 eligibility。AV-2 只报告了 8 train / 8 heldout tiny slice 上的
future-frame aggregate MSE；它没有预注册或报告 AV-3 所需的逐 horizon 1/2/4-step
primary，也没有形成 paired reduced-scale short-rollout 结论。

Review token 已由 immutable claim
`/DATA/share/sana_cach_wam_nonformal_screens/token_ledgers/c73a943047295338cd9de14e6e53199b463ea66ae92b99d4b0b3393f5ec34030/cadfd92f1c72ebd9cc9aa8bfad029ddc6f0e72d9f7942f34e74629ca1ae37252.claim.json`
（SHA256 `e7c355d3c4e8bfb942f8baa8a5511fb5964c2c7f2d6c975ef5aec0daa458a91f`）
消费。AV-3 不得创建、派生、重置或再次消费 token，也没有自动 bug-fix 或预算扩展
重跑额度。

## 3. Architecture invariants

AV-3 必须保持 AV-2 的 reduced actual-operator graph：

1. `operator_level=VENDOR_KERNEL`，`integration_path=EXPERIMENTAL_PATH`；
2. reference 与 candidate 使用相同的 action-blind CUDA/Triton GDN common trunk；
3. candidate 唯一 semantic delta 是 common video output 之后的 A4 action/state delta
   stream；
4. stream 读取 detached、parameter-free RMS-normalized common hidden state；
5. action/state/write/output projections 均 bias-free；
6. injected feature 为 `0.5 * (g(s,a) - g(s,-a))`，保持 exact odd；
7. decay 只读取 common state，不读取 action；
8. anchor、no-action、seam-disabled、typed-inactive 路径保持 direct exact-zero delta；
9. target 只进入 loss/metric，不进入 forward，不允许 loss subtraction；
10. reference/candidate 从一个新的共同 theta0 构造，不加载或继承 AV-2 参数；
11. 两臂 common parameter/buffer bytes 在 theta0 必须相同，candidate delta 在 theta0
    必须保持 identity；
12. expected reduced topology 固定为 batch 8、latent `[5,3,1,1]`、hidden 64、heads 2、
    depth 20、FFN 128、action dim 20、chunks `[0,3,5]`；expected trainable parameter
    counts 为 reference `752439`、candidate `766199`。

## 4. Prospective real-data slice

AV-3 不复用 AV-2 已用于判定的 episode 0--15。未来 data-binding phase 必须在任何
模型构造、CUDA import、optimizer 或 root 创建前，排他创建并冻结卡内固定路径的
data-selection manifest，并逐文件绑定 SHA256。

该 cohort 与任何模型结果无关，selector 唯一规则为：在已冻结 AV-2 episode 0--15
之后按 numeric episode ID 取连续 32 个 episode；前 16 个（16--31）作为 train，后
16 个（32--47）作为 heldout。禁止根据 motion、loss、adequacy 或模型输出重排或替换；
adequacy 只决定 `DATA_INADEQUATE`，不触发 fallback。

固定 selector：

- dataset root：`/DATA/share/RoboTwin2.0/dataset`；
- task / variant：`adjust_bottle / aloha-agilex_clean_50`；
- camera：`head_camera`；
- train episodes：16--31，共 16 个 window；
- heldout episodes：32--47，共 16 个 window；
- 每 episode 只取 raw rows `[0,33)`；禁止缺失时替换 episode/window；
- action：absolute EEF20，raw rows `[1,33)`，不 normalization；
- frames：raw indices `[0,8,16,24,32]`；
- image decoder：OpenCV native BGR，不做额外 channel conversion；
- observation projection：每帧 spatial channel mean / 255，得到 `[5,3,1,1]`；
- target：上述 absolute video 减 frame-0 anchor；target frame 0 exact zero；
- noisy video：frame-0 projection 重复 5 次；不读取 future clean video；
- proprio：absolute EEF20 raw row 0，重复到两个 chunks；
- train microbatches：episodes `[16..23]`、`[24..31]`，optimizer steps 按此顺序循环；
- heldout microbatches：episodes `[32..39]`、`[40..47]`，用于 step-0 diagnostic 与
  exact-final-step aggregate；只有 exact final step 进入 verdict；
- within-microbatch shuffle permutation：`[1,0,3,2,5,4,7,6]`。

训练与 heldout identity 必须互斥，且两者都与 AV-2 episode 0--15 互斥。每个 split
必须验证 finite、typed mask 非空、action variance `>1e-8`、shuffle mismatch MSE
`>1e-8`，并对 h1/h2/h4 分别要求 `Q[h] > 1e-8`；该 per-horizon motion 条件既要在
完整 16-window heldout aggregate 成立，也要在两个 heldout microbatch 各自成立。任一
失败为 `DATA_INADEQUATE`，禁止换数据自动重跑，且不形成架构负证据。

## 5. Rollout semantics

固定 rollout mode 为：

```text
ACTION_CLAMPED_SINGLE_CALL_CAUSAL_SEQUENCE
```

它的含义是：以一个真实 anchor observation 为唯一 video observation，传入完整的
correct/shuffled/no-action action history，执行一次 chunk-causal vendor recurrent
sequence；从同一输出读取 latent horizons 1、2、4。end-of-bin action indices 分别为
7、15、31。它不读取 teacher future video，不回灌模型 prediction，不是 teacher-prefix
rollout，也不是 joint-generated rollout。所有 AV-3 结论必须明确限定于该
anchor-conditioned、action-clamped short dynamics screen。

Reference 的三种 action mode 必须 bitwise identical。Candidate 的 correct、shuffle、
no-action 只允许真实 action seam tensor 不同；noise、anchor、context、proprio、target、
mask、timestep、layout、padding、parameter bytes 和 initial recurrent state 必须相同。
每个 mode 都从独立的相同 empty readonly state 开始，不在 mode、batch、arm 或
optimizer step 之间延续 live recurrent state。

## 6. Frozen metrics

模型以 float32 执行；所有 metric numerator/denominator 用 float64 累加。令 heldout
batch 总数 `B=16`、channels `C=3`、epsilon `1e-12`，horizon weights 固定为
`w1=0.20, w2=0.30, w4=0.50`。

对 arm `a`、mode `m`、horizon `h`：

```text
V[a,m,h] = sum_b,c (pred[a,m,b,h,c] - target[b,h,c])^2 / (B*C)
D[a,m,h] = sum_b,c ((pred[a,m,b,h,c]-pred[a,m,b,0,c])
                    -(target[b,h,c]-target[b,0,c]))^2 / (B*C)
Q[h]     = sum_b,c (target[b,h,c]-target[b,0,c])^2 / (B*C)
V124     = 0.20*V[h1] + 0.30*V[h2] + 0.50*V[h4]
D124     = 0.20*D[h1] + 0.30*D[h2] + 0.50*D[h4]
Q124     = 0.20*Q[h1] + 0.30*Q[h2] + 0.50*Q[h4]
video_nmse = V124 / max(Q124, epsilon)
delta_nmse = D124 / max(Q124, epsilon)
PRIMARY[a,m] = 0.50*video_nmse + 0.50*delta_nmse
```

该 PRIMARY 同时包含 video error 与显式 anchor-relative motion/delta error，避免静态
背景或 anchor offset 淹没 action-conditioned dynamics。还必须原样报告所有
`V/D/Q` per-horizon values、`V124/D124/Q124`、video/delta NMSE 及：

```text
candidate_gain_vs_reference = 1 - PRIMARY[candidate,correct]
                                  / max(PRIMARY[reference,correct], epsilon)
correct_vs_shuffle_improvement = 1 - PRIMARY[candidate,correct]
                                      / max(PRIMARY[candidate,shuffle], epsilon)
correct_vs_no_action_improvement = 1 - PRIMARY[candidate,correct]
                                       / max(PRIMARY[candidate,no_action], epsilon)
h1_candidate_vs_reference_ratio = V[candidate,correct,1]
                                  / max(V[reference,correct,1], epsilon)
candidate_to_reference_primary_ratio = PRIMARY[candidate,correct]
                                       / max(PRIMARY[reference,correct], epsilon)
```

No-action improvement 是必要报告的真实 seam 对照，但不额外收紧 governing plan 的
默认 GO line。Dream-minus-copy 和 action prediction 本轮不构造、不训练、不参与 verdict。

训练 action mode 固定为 `CORRECT_ONLY`：candidate 只用 correct seam action 训练，
reference 保持 action-blind；shuffle/no-action 只用于 step-0 diagnostic 和 final heldout
counterfactual evaluation。训练 objective 保持 positive masked video MSE，对 future
horizons 1--4 等权；禁止 loss subtraction。只允许 step 0 diagnostic 与 exact final
step 评测，只有 final step进入 verdict；禁止 best-step、checkpoint 或 window 选择。

## 7. Seed-0, optimizer and resource ceiling

未来 materialized execution 必须固定：

- screen seed：`202608030`；
- shared initialization seed：继承 frozen A4-R2 builder 的 `2026080331`；
- candidate-only initialization seed：继承 frozen A4 stream 的 `2026080522`；
- fresh construction 定义为从上述 frozen named seeds 重新构造全新 pair，不继承或加载
  任何 AV-2 state；theta0 manifest 在 first forward 前发布；
- arm order：reference 后 candidate，串行；
- optimizer：AdamW，betas `[0.9,0.99]`，epsilon `1e-8`，weight decay `0`；
- LR：cosine `0.003 -> 0.00003`；
- global gradient clip：`1.0`；
- batch size 8，train microbatches 固定循环，无随机 shuffle；
- exact `1000` optimizer steps per arm；
- final-step only；不得保存 model/checkpoint；
- total wall ceiling：60 minutes；
- process RSS ceiling：8 GiB；root ceiling：64 MiB；GPU allocated-memory ceiling：8 GiB；
- device：单张物理 GPU 0，UUID
  `GPU-1ec28cfb-f501-23f3-f865-275a744ca053`，launch 前必须独占且即时 idle recheck。

任何 root 创建前必须完成连续 120 秒只读资源观察：固定 GPU 的 memory/utilization/
compute-app 快照、相关进程快照，以及目标 filesystem 的 free bytes、free inodes、owner/
group 与祖先 non-symlink directory 检查；观察期末再次即时复查。出现外部 compute app、
GPU 非 idle、容量/owner/ancestor 不符时，在 root 创建前 fail-closed 为 `ENV_BLOCKED` 或
`HARNESS_REJECTED`，不得创建 root、抢占或停止他人进程。Future execution card 必须绑定
该证据的 exact schema 与 artifact path。

Seed 1/2 不属于 AV-3；即使 seed-0 为 GO，也只解锁另行授权的 AV-4 confirmation。

## 8. Typed verdict

有效性先于架构判定。至少要求：predecessor/source/data pins 一致；fresh common theta0；
两臂 data/order/step/optimizer 相同；两臂 exact 1000 steps；finite output/loss/gradient/
update/metrics；action seam gradient/update/JVP positive；future-to-prefix exact zero；
zero-origin/oddness/no-action/seam-disabled invariants；target sentinel perturbation bitwise
invariant；target SHA 不在 forward task；action/time alignment 正确；deadline/resource
ceiling 未超。

全部有效性通过后，GO line 为：

1. `candidate_gain_vs_reference >= 0.05`；
2. `correct_vs_shuffle_improvement >= 0.05`；
3. `h1_candidate_vs_reference_ratio <= 1.05`。

三项全部通过得到 `GO`，仅解锁 `AV4_ELIGIBILITY`。

有效 run 若未 GO，并且
`((candidate_gain_vs_reference <= 0 AND correct_vs_shuffle_improvement <= 0) OR
candidate_to_reference_primary_ratio >= 1.25)`，得到 `REDUCED_ARCH_STOP`。其余有效
非 GO 结果因 review token 已消费，得到
`REDUCED_ARCH_INCONCLUSIVE_REVIEW_EXHAUSTED`。任何 authority/environment/data/
implementation/numerical/resource/root failure 分别保持 blocked/invalid typed state，
不得映射为架构负结果，也不得自动重跑。

## 9. Immutable future root and additive files

未来唯一预注册 root（本轮不得创建）：

```text
/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_av3_paired_reduced_scale/06f5d09127f8/cach-a4-av3-97a25a9e1d614340697fbee523e19559
```

nonce：`97a25a9e1d614340697fbee523e19559`。

未来 data/source materialization 只允许下列 additive paths，且必须先获得独立授权：

1. `docs/cach_sana_wam/architecture_validation/cach_a4_av3/CACH_A4_AV3_DATA_SELECTION.json`
2. `src/sana_wam/model/cach_a4_av3_paired_reduced_scale.py`
3. `configs/experiments/cach_a4_av3_paired_reduced_scale.yaml`
4. `scripts/run_cach_a4_av3_paired_reduced_scale.py`
5. `tests/test_cach_a4_av3_paired_reduced_scale.py`
6. `docs/cach_sana_wam/architecture_validation/cach_a4_av3/CACH_A4_AV3_EXECUTION_RUN_CARD.json`

Execution card 必须最后创建，绑定上述 materialized source/data SHA、本 card full SHA、
exact root/nonce/GPU/env/command 及所有 predecessor evidence；未获得该 frozen execution
card 的独立执行授权前，禁止创建 namespace/root 或运行模型。

所有 predecessor source/card/root/result/claim 字节保持不变。Future root 必须
exclusive create、fail-closed、terminal freeze、不得删除、修改、复用或提升为 formal
root。

## 10. Next authorized unit

本 decision/run-card freeze 完成后的下一项工作是：在新的明确授权下，排他新增第 9 节
六个 materialized files，允许只读 data binding、CPU/static focused validation，但不应
自动执行 GPU run。之后只需用户对 frozen execution card SHA 给出简短执行授权，才可
进行一次 seed-0 single-GPU AV-3 run。
