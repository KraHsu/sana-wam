# SANA / AFCC 研究迁移交接（2026-07-30）

> 对应历史对话：`019f5e8b-a6a8-73f3-8e55-6ef9aecd50d1`。
> 本文件用于在新机器上交给新的 agent；它是研究状态与执行边界的事实记录，不是“已完成正式训练”的声明。

## 0. 当前状态（必须先读）

- 研究与实验工作已暂停；当前仅进行迁移整理和只读核验，没有启动训练或评测。
- 最近只读检查只确认：H200 上没有本项目的 `capture_prospective`、`frozen_theta0`、`rng_census` 或 `sana_wam` 进程。GPU **并非空闲**：存在两个无关的 `/home/zch/workspace/mjlab/.venv/bin/python` 计算进程；它们不属于本项目，严禁停止、修改或占用其资源。
- **不得**因为看到 launcher/evaluator 文件就直接启动正式实验：baseline 契约曾处于半重构状态（新 evaluator/verifier 配旧 launcher/authority），formal path 仍缺最终 launcher，504-step 正式 run 从未启动。
- 所有失败实验根必须保持冻结、不可复用；不要删除失败 receipt 或覆盖已有 root。
- 当前结论是机制分析与实验设计结论，不是成功率提升结果。

## 1. 环境与仓库拓扑

### 本地研究工作台（本次 Codex 工作区）

- Windows 工作区：`D:\workspace\codex\h200`
- 该目录的 `.git` 为空目录，不是有效 Git 仓库；本次本地 snapshot 使用旁置的 `.handoff_git_meta_20260730/.git` 完成提交，恢复时以导出的 bundle 为准。
- 活跃研究文件主要在：
  - `afcc_optimizer_work/`：θ0、RNG、optimizer/path、formal capture 的设计、脚本和结果；
  - `afcc_theta_work/`：旧 step-504 control-endpoint trainable-video tangent/held-out 结果；
  - `afcc_optimizer_work/THETA0_HELDOUT_RESULTS_20260728.md`：真正共同 θ0 的 formal v2 held-out 结果；
  - `afcc_gradient_work/`：固定查询 AFCC surrogate-gradient audit；
  - `afcc_objective_audit_work/`：冻结 endpoint audit；
  - `afcc_pullback_work/`、`afcc_directional_work/`、`afcc_eval_work/`：早期 pullback、directional、primary audit 材料。

### H200 Git 仓库

- SSH 主机：`H200`
- 主仓路径：`/home/zch/workspace/sana-wam`
- 迁移分支：`handoff/sana-wam-20260730`；交接快照的旧基线为 `b2a31ad377bd9ce6baae1a73d7876a6d3cda53db`，不要把该旧基线写成最终交接 commit。最终 commit ID 以外置 `FINAL_MIGRATION_MANIFEST_20260730.txt` 为准。
- 主仓交接快照相对旧基线包含 362 个路径：45 个修改和 317 个新增；它们来自审计后的 include 边界，而不是“约 700 个 untracked”全量打包。提交前的 include 路径下已无剩余 untracked 文件。
- 子模块 `third_party/Sana` 的原 detached 基线为 `c1c48d8d44a09cf7459c51830a9cc7a988431d4d`，该基线不在公开 remote branch/tag；三处修改已在本地分支 `handoff/sana-wam-20260730` 提交为 `16b9cec673e3335724ba2d8db25de7f9ed229292`，子模块工作树干净，主仓 gitlink 已暂存。
- 子模块提交包含：
  - `diffusion/model/nets/basic_modules.py`
  - `diffusion/model/nets/sana_blocks.py`
  - `diffusion/model/nets/sana_multi_scale_video.py`

## 2. 研究问题与硬约束

核心问题不是继续寻找 success-rate trick，而是从 SANA 本身解释：为什么 AFCC/teacher-directed video 目标在部署 action loss 上产生冲突，以及 optimizer、clipping、moment/RMS、BF16 projection 如何影响最终路径。

硬约束：

1. 不按任务、层、chunk、sigma 选择结果；不调权、不 best-of-N、不 rerank。
2. 不用 DAgger、人工错误恢复标注或手工 recovery 采集方案作为科学机制证据。
3. treatment 必须是原生 architectural `F=1`；control 必须在 `Trainer` 构造前设 `F=0` 并删除 AFCC reference 字段；禁止用 loss subtraction 伪造 control。
4. 历史 endpoint 缺完整 FP32 optimizer master、Adam moments、step/RNG 状态，禁止 approximate replay；新 504-step 结果只能称 prospective replicate。
5. 失败必须 fail-closed；失败 root 冻结，不在原 root 重跑或覆盖。
6. 任何“finite-path gradient drift”只能从 `D0_clip -> Dpath` 解释；不能把 raw/uniform `u_F0` 直接等同于路径方向。
7. 结果必须报告参数 chord、master rounding、BF16 projection 等残差，不能强行套理想公式或未注册 ULP 容差。

### 固定生产输入

- config：`/DATA/share/sana_phase6_principled_constraints_20260724/post_primary_afcc_v3/training_v1/config.yaml`
  - SHA256 `85c6e9e34ce8c3e1cd5b76547ce23b8de58fb1d06cb1f699bda7363df64b8a80`
  - `max_steps=504`、`batch_size=1`、`gradient_accumulation_steps=1`、`num_workers=4`、`training.seed=20260724`
- common θ0：`/home/zch/wuji-openwam-dev/sandbox/sana_ar_trajectory_consistency_init/checkpoint_step_0.safetensors`
  - SHA256 `e9549aff484da56eada21972f00fb3aa2f95f8a65397a55cf0f38c79e6254137`
  - size `11,689,906,174` bytes
- Trainer source pin：`/home/zch/workspace/sana-wam/src/sana_wam/train/trainer.py`
  - SHA256 `715236acec1269f1f672aafacc99bf37afcf4ca6eadca5a6ca941e85821f3496`
- 该 SHA 仅固定相关实验契约所读取的 Trainer **源文件内容**；它不是 Git commit、checkpoint/runtime 状态证明或启动授权。迁移后必须重新校验文件内容，不能仅凭路径或当前分支推定一致。
- Phase-6 source manifest SHA256：`46f84b151e2ea008abdd4f99f2a43a3575184eb598f78710ea45b91214f8e012`
- 最近磁盘检查：`/DATA` 约4.2TiB可用，系统根盘仅约58GiB可用，所以大产物只写 `/DATA`。GPU 空闲状态是易变事实，且交接时已有无关 `mjlab` 作业；任何资源分配都必须重新只读核验，不得干预他人进程。

## 3. 历史决策链

### 3.1 Robotwin / DAgger 阶段

- 早期目标是处理 SANA/Robotwin 的约 44% plateau。
- 尝试过 closed-loop、outcome critic、generation-zero rerank、candidate selection、不同 noise/history 等方向；这些只能改变后处理或分布，不能说明 SANA 的底层 mode formation/视频-动作机制。
- generation-zero reranker 与 common-future critic 的固定评测均为 NO-GO；“每个任务人工设计错误恢复采集”被判定没有可扩展科学价值。
- 因此停止 trick 路线，转向 SANA 原理和函数空间机制。

### SANA 原理诊断主线（AFCC 的前置背景）

以下五份报告均在 `local-afcc-handoff.bundle` 恢复出的 `analysis/` 下，主仓中的 `docs/agent_handoff/` 没有 `research/` 子目录：

1. `analysis/SANA_FIRST_PRINCIPLES_RESULTS_20260722.md`：AR 因果可见性下，ReLU feature support 可令未旋转非负分母严格为零，而 RoPE 后有符号分子非零，生产 forward 直接观测到 `1e15-1e18` 级 video residual；同时 `low_noise` 目标在推理起点 `sigma=1` 的 action 权重为零，弱 bootstrap 经 action history 放大。
2. `analysis/STRICT_POSITIVE_FEATURE_MAP_RESULTS_20260722.md`：strict-positive feature map 消除了零分母和爆炸，是当时最兼容 checkpoint 的反事实，但未统一有符号 RoPE numerator 与非负 denominator，predicted-video gate 仍未解决。
3. `analysis/ALIGNED_KERNEL_RESULTS_20260722.md`：三种一致/有界 kernel 均消除数学奇点；`post_rope`、`unrotated` 与旧 checkpoint 严重不兼容，最接近的 `absolute_rope` 仍使 predicted-video penalty 恶化，故没有启动该架构训练。
4. `analysis/FULL_ATTENTION_RETRAIN_RESULTS_20260723.md`：进一步定位完整 SANA block 的训练/缓存推理不一致——window-flash 在整段训练中双向跨帧，而推理按两帧 chunk；对称 temporal convolution 同样看到推理不可得的未来帧。
5. `analysis/CHUNKWISE_TEMPORAL_RESULTS_20260723.md`：chunkwise 执行修复 full-block 等价并改善局部 video vector field，但 10-step rollout 与 action gate 仍失败；步数越多越差，指向 flow-matching 只在 expert/noise 插值流形训练、推理反复进入 off-manifold 的 trajectory-consistency 问题。

这条主线说明已排除或降级的不是几个超参数，而是 SANA AR 的 kernel、完整 temporal operator 与有限步轨迹三层结构问题；所有相关 closed-loop gate 均为 NO-GO，不能写成成功率提升结果，也不需要逐任务人工恢复数据。

### 3.2 冻结 endpoint audit（已完成）

固定 cohort：8 tasks、24 pairs/task、2 contexts/pair、3 sigmas，共 384 contexts/1152 context-sigma rows；不做选择。

主要结果（见 `afcc_objective_audit_work/FROZEN_ENDPOINT_AUDIT_RESULTS_20260727.md`）：

- Natural Phase1 minus control action MSE `T=+0.0124475`，95% CI `[+0.0054409,+0.0191474]`。
- Teacher-S endpoint effect `E_R=+0.0119096`，CI `[+0.0051925,+0.0183727]`。
- AFCC minus control `H=+0.0147797`，CI `[+0.0067815,+0.0227584]`。
- `E_R` explains 99.62494% of teacher-gap energy；`E_A` explains 99.82702% of AFCC-harm energy。
- Exact teacher `S` leaves only 0.55% of control numerator error and 2.45% of AFCC numerator error；omitted video-z share约0.22%。
- 结论：M1 anchor/objective conflict 最有支持；query-coordinate mismatch 与 denominator omission 不是主因；endpoint-only curvature 不是沿 `d_R` 的必要解释，但 exact AFCC steepest direction 当时尚未存下。

### 3.3 固定查询 AFCC surrogate-gradient audit（已完成）

见 `afcc_gradient_work/AFCC_SURROGATE_GRADIENT_RESULTS_20260727.md`。

- 固定 query/cache 接口定义 `F(S)=normalized ||Q_control S-N_Phase1||²`，`U=-grad_S F`。
- action-loss derivative `<grad L_action,U> = +3.3721e-12`，CI `[+0.6961e-12,+7.6647e-12]`。
- AFCC derivative沿 teacher direction `d_R=-0.0799358`，沿 AFCC endpoint `d_A=-0.0753331`，均显著负（确实在下降 surrogate）。
- `cos(U,d_R)=0.000853`，但 `cos(JU,Jd_R)=0.10343`；raw cache-space steepest descent 几乎不等于最终 teacher direction。
- 结论：instantaneous interface objective conflict 已成立，但 trainable-video tangent、自然 query、optimizer dynamics、有限路径曲率仍需分解。

### 3.4 step-504 control-endpoint tangent audit（已完成，存在 base mismatch）

见 `afcc_theta_work/THETA_TANGENT_RESULTS_20260727.md` 与设计文件。

- 所有局部方向都在冻结的 step-504 `T1_E1A0` control checkpoint 求值；treatment/control 是从共同 θ0 出发的平行 504-step runs，treatment **不是**从 control endpoint continuation。
- `Delta=(theta_AFCC,504-theta0)-(theta_control,504-theta0)` 是两个 endpoint chord 之差，因此它与 control-endpoint tangent 使用不同 base point。
- 方向 norms：`u_S=0.469680`、`u_Q=0.026171`、`u_F=0.470646`、`u_V=0.075345`、`u_T=0.476647`、`Delta=4.366110`。
- `cos(u_S,u_F)=0.998453`；`cos(u_F,Delta)=0.049472`；`cos(u_F,u_V)=0.000230`；`cos(u_F,u_T)=0.987428`。
- held-out：`H_F=+0.0147999`（CI `[+0.0049453,+0.0273299]`）、`H_S=+0.0144126`、`H_Q=+0.0003873` unresolved、`H_V=-0.0000526` unresolved、`H_T=+0.0147458`、`H_Delta=+0.0123712`。
- `beta_R,F=+0.317852`（CI `[+0.283868,+0.347152]`），8/8 task point estimates positive。
- 结论只适用于 control endpoint：local AFCC direction teacher-directed 且 action-harmful，S-generating component 主导，Q 不是必要 harm source。`cos(u_F,Delta)=0.049472` 混合了 base-point mismatch，不能据此归因 AdamW、row order、curvature，也不能解释历史 path。

### 3.5 真正共同 θ0 held-out tangent formal v2（已完成）

见 `afcc_optimizer_work/THETA0_HELDOUT_RESULTS_20260728.md`；两臂都从 SHA256 `e9549aff484da56eada21972f00fb3aa2f95f8a65397a55cf0f38c79e6254137` 的真实 step-0 checkpoint 求局部方向。formal v2 source manifest/evaluator/aggregate SHA256 分别为 `fdaec98241166359d25d190e8db15bf52f6317d82528e642fd4af9eeed0508d5`、`57120fed69626472c090d31d94169dc9951c957b597733af4438791de2ab12e2`、`fd36a22bc88940f389e141bcfe4c19f792a65b9bb826bc1a4ee70fc155ee981c`。

- 参数几何：`cos(u_S0,u_F0)=0.99808050`，`cos(u_F0,Delta)=0.04229095`；所有 raw θ0 directions 与 finite endpoint contrast 近正交。
- held-out：`H_F0=+0.00848384`（drift-inflated 95% CI `[+0.00193505,+0.01645229]`）、`H_S0=+0.00814310`、`H_Q0=+0.00034074` unresolved、`H_V0=-0.00018058` unresolved、`H_T0=+0.00830100`、`H_Delta=+0.01337740`。
- `beta_R,F0=+0.27832110`（CI `[+0.25594123,+0.29812732]`），8/8 tasks positive；`H_staticS,F0=+0.01036549`，而 natural Q 与 non-S remainder 不支持 Q 是必要 harm source。
- 结论：冲突在真实训练起点已存在——raw AFCC direction 同时 teacher-directed 且 action-harmful，S-generating path 主导，并发 video direction 未显示保护性抵消。这仍是 θ0 局部 tangent，不证明 endpoint chord 由反复跟随 `u_F0` 生成，也不分解旧 AdamW/clipping/BF16 路径。

### 3.6 θ0 response-signature geometry（已完成）

见 `afcc_optimizer_work/THETA0_RESPONSE_SIGNATURE_RESULTS_20260728.md`。

- `cos_parameter(u_F0,Delta)=0.04229095`。
- 384-context action response cosine `0.71925`，centered correlation `0.63454`。
- teacher-cache `beta_R` response cosine `0.99397`；treatment-cache `beta_A` `0.99559`。
- F/S 参数 cosine `0.99808050`，F/Delta 的 output response 仍明显对齐。
- 结论：SANA parameter→response map 高度各向异性；近正交参数方向可共享 output-sensitive subspace；Euclidean parameter cosine 不是充分几何。

### 3.7 历史 replay 与 RNG 结论

- 历史 endpoint 没有完整 optimizer/master/moments/RNG；exact replay 在首个非零 update 后不可识别。失败 receipt 不得被当作结果。
- frozen RNG census v1/v2/v3 均 fail-closed，原因包括 BF16 scalar digest、root inventory 被 `nohup.log` 污染，以及真实发现 forward 消费 `torch_cpu` RNG。
- 对应失败 roots 位于
  `/DATA/share/sana_phase6_principled_constraints_20260724/post_primary_afcc_v3/`
  下的 `frozen_theta0_baseline_rng_census_v1_20260728`、`v2_20260728`、`v3_20260728`；全部只读保留，禁止复用。
- dataset direct indexing、backward 未观察到消费全局 RNG；forward 消费 `torch_cpu`。
- 因 `num_workers=4`，主进程 ledger 不能覆盖 worker RNG/prefetch；最终设计取消预采 504-row ledger 与 8-way sharding，改用每臂一个 fresh production DataLoader/iterator，保留自然 worker/prefetch 顺序。
- 可记录主进程五域 RNG digest 作为 paired observation，但不得宣称完整 RNG replay。

### 3.8 prospective path smoke（仅 admission，不是科学结论）

冻结 root：
`/DATA/share/sana_phase6_principled_constraints_20260724/post_primary_afcc_v3/prospective_paired_path_capture_smoke_v5_20260728/`

关键 SHA：paired `29636c449f5f71c5d3d5347cad247e7d099632cd7f030ab33e830dd4d100b7ca`；evaluator `9111277dba87c6f41b8a503c006d5c38d39afb4fb5aaa47672a58dee63e0b37a`；verifier `327cb2fa7efd0ab027c66c142e665422ac2d9e17b36525957f4e5198d167d5ce`；design `65da8a7fe2f69bc51bcf2ed48c8e1ff4d65cdb1906c5a0d9d75d0d18ddd78388`。

- 两行 treatment/control admission 通过；380 tensors、1,104,871,060 elements。
- `Dmaster-Dadam` 相对残差 treatment 约 5.25%、control 约 4.93%；`Dbf16-Dmaster` 约 99.98%。
- peak CUDA约131.1GB/arm，host VmHWM约107.5GB/arm；四个 CPU FP64 accumulators约35.36GB/arm。
- 结果只证明采集机制和残差必须显式保留；两行不能支持 SANA 机制 claim。

## 4. 当前 planned experiments（未运行）

### A. Frozen θ0 LR/clip baseline

目标是建立：

```text
D0_lr[j]   = -sum_t eta[t,j] * g[t,j](theta0)
D0_clip[j] = -sum_t eta[t,j] * C(g[t,j](theta0))
```

约束：`next(data_iter) -> native forward/backward -> raw BF16 accumulate -> exactly one production clip -> clipped BF16 accumulate`；无 optimizer/master/step/allreduce/normalization。只有 `D0_clip -> Dpath` 可称 finite-path gradient drift。

本地 `compute_frozen_theta0_lr_clip_baseline_v1.py` 已大体改成 single fresh process + production DataLoader + no restore/no optimizer，但其 `launch_frozen_theta0_lr_clip_baseline_v1.sh` 和 authority 曾仍是旧 8-rank/ledger 版本；必须先重新审计并生成全新 immutable roots。不要使用当前旧 launcher/authority。

暂停时本地 snapshot SHA（用于识别半重构状态，不是 authorization）：

```text
compute evaluator  44275ff9b4ac6a6227779225e7740901ff3253446f2137ec08a1a495c152d2fb
paired verifier    64548f65487571a70672e9415a96fbc9b71a9ae591113ec70a39e7667b73cfbf
design             0b2e4491ef5fe07eca51c85bd5fa67ffcd176e9b5953aca08450ef9b2c432a07
OLD launcher       27fbc081b4b0a7b58831143afc3122c4c0993ef8557ec7dd8f836fc1829e991d
OLD authority      832f66a4afde747fd5f845e12db1e06c5f1d0ea31b79e0c554eb2d4fea91926d
```

### B. Formal prospective paired path

本地 evaluator：`afcc_optimizer_work/capture_prospective_paired_path_formal_v1.py`。
本地 verifier：`afcc_optimizer_work/verify_prospective_paired_path_formal_v1.py`。
当时 SHA：evaluator `097bc6148f3b4a5716de5df78d0b8528bf20e6dcbf305632314acde897f3c1f4`；verifier `5eaad0fd9ab5b8337e9af3ab8cc732bc8486dd497b4a51b3e32e3c08d11badcf`。

formal directions：`Dpath`、`Dmom`、`Drms`、`Dadam`、`Dmaster`、`Dbf16`、`R_formula_master`、`R_master_bf16`。定义：

```text
Dpath   = -sum eta * g
Dmom    = -sum eta * mhat
Drms    = -sum eta * g/r
Dadam   = -sum eta * mhat/r
Dmaster = master504-master0
Dbf16   = bf16504-bf160
R_formula_master = Dmaster-Dadam
R_master_bf16    = Dbf16-Dmaster
```

当时尚缺最终 launcher；且独立 integration review 指出需补：固定 DataLoader/seed/完整 plan row、严格 admission schema、外部 config/checkpoint/source 终检、主进程 RNG digest paired exact。新 agent 必须先修这些，再做 admission，再做 504；不得把 prospective endpoint 叫历史 endpoint reconstruction。

### 暂停时验证状态

- formal evaluator/verifier 本地 `py_compile`、双方 `--self-test` 通过；verifier 在 H200 的 Torch+safetensors 环境自测通过。
- baseline evaluator 在半重构期间通过本地 `py_compile`/轻量 self-test，但这不验证 launcher/authority/verifier 一致性。
- H200 主仓当前大工作树没有在交接时重跑完整 `pytest`；已有历史测试日志不能证明暂停时的最终 tree。迁移提交是 work-in-progress checkpoint，不应被当作 release。

## 5. 迁移文件与提交安排

本地研究工作台作为独立 Git snapshot 提交并导出为
`local-afcc-handoff.bundle`；主仓只保存本交接入口，避免把冻结 JSON/实验
输出混入产品源码历史。独立 snapshot 包含：

- `afcc_optimizer_work/` 全部设计/脚本/结果（当前活跃工作台）；
- 其余 `afcc_theta_work/`、`afcc_gradient_work/`、`afcc_objective_audit_work/`、`afcc_pullback_work/`、`afcc_directional_work/`、`afcc_eval_work/` 中的结果摘要与设计文件；
- `analysis/remote_work/` 的 18 个历史结果、源码、配置、脚本和测试文件；两个 NO-GO 结果报告依赖这些支撑实现；
- 本文件作为唯一入口索引，并记录每个冻结 root 的远端绝对路径和 SHA。

H200 主仓提交边界：

- **include**：全部 tracked 修改，以及 `README.md benchmarks/ configs/ docs/ scripts/ src/ tests/ third_party/Sana` 下的真实代码、配置、文档、测试。
- **exclude**：`%ln` 空文件；`analysis/` 重复快照、候选树与执行 logs；`telemetry/` 运行数据；`.venv/`、`logs/`、`outputs/`、`.pytest_cache/`、`.ruff_cache/`、`__pycache__/`；任何 checkpoint/大模型权重。已筛选的本地研究材料只进入独立 AFCC bundle，不进入主仓。
- 子模块三处修改必须先在 Sana 建本地分支并提交，再提交主仓 gitlink。
- 由于新 Sana commit 不在远端，需要同时导出 `sana-submodule.bundle` 和主仓 `sana-wam.bundle`；只提交 gitlink 不足以在新机器恢复。
- 最终 commit IDs 与 bundle SHA256 只写入迁移目录外置的 `FINAL_MIGRATION_MANIFEST_20260730.txt` 和 `SHA256SUMS`；这两个最终清单不纳入被其校验的 commit/bundle，以避免自引用。仓内 `MIGRATION_MANIFEST_20260730.txt` 只记录内容边界。

## 6. 新机器恢复命令（示意，先核对 SHA）

```bash
# 先把迁移目录设为绝对路径，并按外置 manifest 核对最终 commit IDs
MIGRATION_DIR=/absolute/path/to/migration-20260730
cat "$MIGRATION_DIR/FINAL_MIGRATION_MANIFEST_20260730.txt"
(cd "$MIGRATION_DIR" && sha256sum -c SHA256SUMS)

# 主仓只从 bundle 恢复，不依赖 origin 是否含本地提交
git clone -b handoff/sana-wam-20260730 "$MIGRATION_DIR/sana-wam.bundle" sana-wam
cd sana-wam

# 先从 Sana bundle 初始化子模块；禁止先运行 submodule update 去访问 origin
git submodule init
git clone -b handoff/sana-wam-20260730 "$MIGRATION_DIR/sana-submodule.bundle" third_party/Sana
git -C third_party/Sana checkout 16b9cec673e3335724ba2d8db25de7f9ed229292
git submodule absorbgitdirs
test "$(git rev-parse HEAD:third_party/Sana)" = "$(git -C third_party/Sana rev-parse HEAD)"
git status --short --branch
git submodule status --recursive

# 独立 AFCC 研究工作台（可放在主仓旁边，不要覆盖产品仓）
cd ..
git clone -b handoff/sana-afcc-20260730 "$MIGRATION_DIR/local-afcc-handoff.bundle" sana-afcc-handoff
cd sana-afcc-handoff
cat AGENT_HANDOFF_20260730.md
```

原 Sana base 与新 commit 都不能假定存在于 NVlabs origin；即使网络可用，也必须先用 `sana-submodule.bundle` 建立对象库，再校验主仓 gitlink。三份 bundle 的最终 SHA 和两个 handoff commit ID 以外置 manifest 为准，不在本文虚构占位值。

## 7. 新 agent 第一轮任务

1. 阅读主仓 `docs/agent_handoff/AGENT_HANDOFF_20260730.md`，并在恢复出的 `sana-afcc-handoff/` 中阅读 `analysis/`、`afcc_optimizer_work/THETA0_HELDOUT_RESULTS_20260728.md` 及各 AFCC 结果摘要；不存在 `docs/agent_handoff/research/`。
2. 检查 H200/新机 GPU、磁盘、并发进程；只读验证所有冻结 roots，且不得触碰无关 `mjlab` 作业。
3. 重新审计 baseline/formal 五件套的一致 SHA、argv、root inventory、fail-closed 行为。
4. 为 baseline admission-1/2 各创建独立 immutable root；核对完整 plan row、worker 自然顺序、主进程 RNG digest、θ0 首尾 raw hash。
5. admission 全通过后，才启动 504 baseline；再独立启动 formal prospective paired path。
6. 最后实现 aggregator，按 `D0_lr -> D0_clip -> Dpath -> Dmom/Drms/Dadam -> Dmaster -> Dbf16` 比较 treatment-control contrast、geometry 和 held-out response。

## 8. 明确不可宣称的内容

- 当前没有 504-step prospective formal 结果。
- 不能把 smoke 两行结果当机制结论。
- 不能把历史 endpoint 用 approximate replay 分解成 optimizer/clipping/momentum/BF16 成因。
- 不能把 `cos(u_F0,Delta)` 很小解释为 functionally unrelated；response-signature evidence 已否定这种简单推断。
- 不能把 local tangent/teacher-forced held-out result 宣称为 closed-loop BPTT 或 deployable intervention。
