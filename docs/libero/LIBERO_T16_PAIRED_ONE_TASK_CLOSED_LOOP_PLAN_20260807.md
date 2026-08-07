# LIBERO T16：Paired One-Task Chunk-Closed-Loop Smoke

状态：**T16 已完成；最终 fresh-root execution/interface PASS，source、simulator、
paired closed-loop 和 20×accum8 update 均有效。该单任务单初态结果不是正式 benchmark。**

T16 是 T15 后的第一个行为证据门槛：在同一 fresh-init 模型进程内，先从
一个固定 LIBERO simulator state 做 pre-update rollout，再原样执行 T15 的
20×accum8 JOINT update，最后从完全相同的 simulator state 做 post-update
rollout。两臂共用独立于训练 recipe 的 paired inference noise。全程不加载、
不保存 SANA-WAM checkpoint。

本实验只验证 simulator↔observation→model→action→simulator 真实通路、
AR cache/replan 边界和 pre/post 轨迹可观测性。单任务、单初态不构成 benchmark，
不宣称 success rate、策略改善、泛化或训练 readiness。

## 1. 直接前序

直接前序唯一固定为 T15：

- source commit：`d669a2e0bc201941fd911f484df36d7afec44cd6`
- runner SHA256：
  `25ebbe148287a7ee9795f56e9253b1db238d0cf5dbea52778c7b61932ee9023a`
- immutable root：
  `/DATA/share/sana_wam_libero_nonformal_screens/t15/d669a2e0bc20/libero-t15-loss-recipe-seed-replication-accum8-balanced-joint-fixed20-9732531de18094840229d7ca0598a648`
- RESULT SHA256：
  `9e4ee9cb515e5cfafc5598eff33041232ad702b32e641b26435bcb1c643820b7`
- execution/science：
  `T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_JOINT_ARM_VALID` /
  `T15_LOSS_RECIPE_SEED_REPLICATION_ACCUM8_BALANCED_JOINT_REPLICATED`

T16 必须在任何 CUDA/model/simulator 构造前只读核对 T15 root、RESULT、
source/config/runner identity、16 ratios、selection bytes、recipe signature、计数和禁止项。

## 2. 为什么不能用原计划的 16 步

T15 的训练输入为 `actions=[1,112,7]`、`input_latents=[1,16,8,48,40]`；
`ar_frame_chunk_size=2` 将 8 个 latent frame 分成 4 个 chunk，因此每个
AR action chunk 为 `112/4 = 28` 个动作。当前 `ARInferenceEngine` 的训练对齐
deploy 契约是 greedy：生成一个 28-action chunk，执行完后才消费新观测
并推进一个 AR cache frame。

因此：

- 16-step arm 只会消费第一个 action buffer 的前 16 项，模型不会再生成；
- 把这 16 步称为“每步重规划闭环”是错误的；
- 强行设 `execute_horizon=1` 会让 cache 记住 28 个未完整执行的动作；
- 强行设 `ar_action_tokens_per_chunk=1` 则改变了训练时 28-token 语义。

T16 因此保持 28-token chunk 和 greedy cache 契约，每臂最多执行 32 个
environment step。若未提前合法终止，第 29 个 policy request 必须消费新观测并
发生第二次 model generation，这才是本实验最小的 training-aligned
chunk-closed-loop 证据。它仍不是每个 simulator step 重规划。

## 3. 确定性任务和初态选择

候选集只包含 T15 的四个固定 task。对每个 task 形成 canonical UTF-8
sorted compact JSON + LF：

```json
{"bddl_sha256":"<sha>","init_count":50,"init_file_sha256":"<sha>","suite":"<suite>","task":"<task>","task_index":<index>,"t15_result_sha256":"9e4ee9cb515e5cfafc5598eff33041232ad702b32e641b26435bcb1c643820b7"}
```

取 SHA256 字典序最小者，禁止按效果挑 task。冻结结果为：

- suite：`libero_spatial`
- LeRobot dataset task index：`5`
- LIBERO suite ordinal：`3`（不得与 dataset task index `5` 互换）
- model prompt：`pick up the black bowl on the cookie box and place it on the plate`
- BDDL/environment language：
  `pick the akita black bowl on the cookies box and place it on the plate`
- task-selection SHA256：
  `30bc48677e10f0674a5837833d4b77a3ffe48751767b28a93042e08af8e3a0d8`
- BDDL SHA256：
  `3d4ccf070c3d9883ae0676f2d888588f98c696ddad71b7694f47c379fc99ef36`
- init file SHA256：
  `0627f5f5ce3ef23be546571012be8ef603d93bcb4032bc80feb34937ba580140`

初态 payload 固定为：

```text
SANA-WAM/LIBERO/T16_INIT_V1
T15_RESULT_SHA256=9e4ee9cb515e5cfafc5598eff33041232ad702b32e641b26435bcb1c643820b7
TASK_SELECTION_SHA256=30bc48677e10f0674a5837833d4b77a3ffe48751767b28a93042e08af8e3a0d8
INIT_FILE_SHA256=0627f5f5ce3ef23be546571012be8ef603d93bcb4032bc80feb34937ba580140
```

其 SHA256 为
`92b19e8daf3b6aef7b311d5ba66a48f7cee5a99abf99b1c23d8308a2f11166fd`，
以大端整数取模 50，得到唯一 init-state index `17`。

## 4. 模型、更新和 inference 固定项

- fresh model initialization seed：`20260807`
- dataloader seed：`20260806`
- global training recipe seed：`20260827`
- 完整复用 T15 的 16 个 real-data 样本、selected-row stats、参数 allowlist、
  FP32 optimizer-master AdamW 和 `20 macro × 8 micro-backward`。
- pre/post rollout 在同一 model object 上运行；不通过 checkpoint 传递状态。
- 两臂使用不同 `episode_key`但相同 `noise_pair_key`，paired inference seed
  独立于 training recipe seed。
- inference 保持 `action_tokens_per_chunk=28`、`rolling_buffer`、
  `latent_band=auto`、`proprio=per_step`、`cache_feedback=predicted`、greedy policy、
  temporal ensemble off、async off、rerank off。
- video/action flow-matching 各 4 步；禁止 best-of-N 或 generation-zero rerank。
- pre arm 后释放 rollout cache 并恢复 training mode；post arm 前重建并重置
  rollout cache。rollout 内模型参数不得变化。

## 5. Simulator 隔离与固定身份

SANA-WAM 主环境为 Python 3.12，没有 `libero/robosuite/mujoco`；不得将旧版
simulator stack 安装进主 `.venv`。T16 通过持久 NDJSON subprocess 隔离模拟器：

- clean LIBERO checkout：
  `/home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO`
- LIBERO commit：`8f1084e3132a39270c3a13ebe37270a43ece2a01`
- simulator Python：`/home/zch/workspace/starVLA/.venv-libero/bin/python`
- interpreter SHA256：
  `7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86`
- Python `3.10.12`；robosuite `1.4.0`；MuJoCo `3.2.3`；bddl `1.0.1`；
  gym `0.25.2`；NumPy `1.24.4`；Torch `2.11.0+cu130`。

worker 只读 clean checkout 的 assets/BDDL/init state，不修改外部仓、`~/.libero`
或 simulator venv。`LIBERO_CONFIG_PATH` 必须显式指向新增且已固定 SHA 的
config，不允许 LIBERO import-time prompt 创建文件。

物理 GPU 编号契约不得做 logical-0 重映射：对选定物理 GPU `N`，simulator child
必须同时设置 `CUDA_VISIBLE_DEVICES=N`、`MUJOCO_EGL_DEVICE_ID=N` 和 renderer
device `N`。固定的 robosuite `1.4.0` 会直接按物理 EGL device list 索引，若写成
logical `0`，在非 GPU 0 上会失败或渲染到错误设备。child 的 `PYTHONPATH` 只允许
clean LIBERO checkout 与 SANA-WAM repo，不继承 caller 的额外路径。

worker 在导入 LIBERO 或构造 env 前完成 interpreter、checkout、config、BDDL、init、
Python/runtime package 版本核对，并在 `OffScreenRenderEnv` 构造前固定 NumPy seed。
关闭前复核已加载 `libero.*` module 均来自 clean checkout，且关键源文件字节未变。

worker 协议只允许 `ready/reset/step/close`；每条 request/response 有单调 seq。
`reset(pre)` 与 `reset(post)` 均执行相同 env seed `0`、相同 init index `17`
和 5 个零动作 settle step；两臂 settle 后的 MuJoCo state 与两视图 observation
SHA 必须逐字节相同，否则 fail-closed。

## 6. Observation/action 契约

- live `agentview_image` / `robot0_eye_in_hand_image` 各旋转 180°，分别映射
  `head_camera` / `left_wrist_camera`；`right_wrist_camera` 固定黑帧。
- 8D state：`eef_xyz(3) + quaternion_xyzw_to_axis_angle(3) + gripper_qpos(2)`，
  公式字节语义复用现有 LIBERO adapter，不对 quaternion 强制翻号。
- 图像、state 由现有 `PolicyServer` deploy preprocessing 路径处理，使用 trainer.dataset
  内的固定 7D action / 8D state normalizer 构造 dual normalizer。
- model action 必须是 7 个 finite 值；前 6 维不裁剪、不填充、不改解释。
- gripper 训练 target 端点为 closed `0` / open `1`，但 diffusion 连续预测不保证
  落在 `[0,1]`；任何 finite 输出都直接以 `>0.5` 映射为 LIBERO open `-1`，
  其余映射为 closed `+1`。禁止为满足端点范围而 clamp。
- 每个 environment step 仍把新观测送入 policy；但只有 action buffer 耗尽时
  才调用 model generation，必须分别记录 policy request 和 generation 计数。

## 7. Source/static closure 证据（2026-08-07）

本轮只新增以下四个 implementation 文件；其 materialized SHA256 为：

- `scripts/libero_t16_sim_worker.py`：
  `653ccc274e6dcc4f08799564c06822b28c6080c9826bfaee7af3fa5744718584`
- `scripts/smoke_libero_ar_t16_paired_one_task_closed_loop_gpu.py`：
  `b53213477f04cff37798eb7cdff9e1fbed621d57e03df30e566404a66857a178`
- `configs/benchmarks/libero/t16_simulator/config.yaml`：
  `29b385e0ec34a664c42635c3b94e817aaa711e948a83b980410fefd3be6f2ec3`
- `tests/test_smoke_libero_ar_t16_paired_one_task_closed_loop_gpu.py`：
  `643ab78cc420bf99675c890623a1d9bea08992b34f6da9cfb268c5ffa47d2aaf`

runner 内复用的 T15 update core 与 T15 runner 做 AST normalized projection，固定
SHA256 为
`357c111513e0b96e30297189f9d47bdef6a0b2486bd92b32d7e244b2f4799429`。
H200 上 `ruff format --check`、`ruff check`、`py_compile` 均通过，T16 专项静态测试
`9 passed`。这些检查不导入 LIBERO、不构造 simulator/model、不创建 root、不访问
CUDA，也不执行 update 或 rollout；因此本节不是运行证据。

### 7.1 首次 execution attempt 的 fail-closed 边界

首次 attempt 固定为 source commit
`2661f655891df6e29721cb4d796bae7e427fb466`、runner SHA256
`39c4f26811baca2ed4501286865eda2204982b5dab82a87d5c08a6e030a91a44`，
nonce `13e276b36ef98103696d87de38949885`。其 terminal root 为：

```text
/DATA/share/sana_wam_libero_nonformal_screens/t16/2661f655891d/libero-t16-paired-one-task-chunk-closed-loop-fixed32-13e276b36ef98103696d87de38949885
```

root 已冻结为 `0500`，唯一 `FAILED.json` 为 `0400`，SHA256
`f9a2ae9b0a2017b93e1dff445c579463537834de9f8d8115ba185fa356a99294`。
错误为 `ModuleNotFoundError: No module named 'benchmarks'`：runner 以
`scripts/...py` 入口执行时加入了 `scripts/src/third_party`，但没有把 repo root
加入 `sys.path`。失败发生在创建 simulator client、PRE rollout 和 T15 update 之前；
因此它不是 interface 或行为效果证据。运行中只完成了架构构造、允许的 base SANA
初始化权重加载和训练输入预处理；未加载或保存 SANA-WAM checkpoint。

根因修复只在 import 前增加 repo-root path，并加入静态回归断言；旧 root 永不复用、
不修改。修复后的 GPU 复核必须使用新 source commit、new runner SHA、fresh nonce/root，
且不得自动重跑。

第二次 attempt 固定为 source commit
`f381380034ed440c2911d5d518e59517d4066659`、runner SHA256
`b53213477f04cff37798eb7cdff9e1fbed621d57e03df30e566404a66857a178`，nonce
`e1554dd205e9ef4ffee8932686a8eb23`。其 terminal root 为：

```text
/DATA/share/sana_wam_libero_nonformal_screens/t16/f381380034ed/libero-t16-paired-one-task-chunk-closed-loop-fixed32-e1554dd205e9ef4ffee8932686a8eb23
```

root/唯一 `FAILED.json` 已分别冻结为 `0500`/`0400`，FAILED SHA256
`9ec732e81f1e5bedbc6ab530e2eeb0451b7481031013cf3f4b18d1c882b18054`。
该 attempt 已成功启动 simulator、完成 deterministic reset 和首次模型 generation，
但在第一个 env step 前因 gripper prediction `-0.8203125` 被旧 adapter 的 `[0,1]`
range gate 拒绝。训练 stats 明确固定 gripper raw targets 为 `{0,1}`，min-max 后模型
target 为 `{-1,+1}`；continuous diffusion 输出本来就不受该范围硬约束。正确 deploy
语义是对反归一化后的 finite score 直接在 `0.5` threshold 二值化，而不是把外推值
当作架构失败，也不是 clamp。此修复不改变 motion action、训练 loss、normalizer、
模型架构或 simulator controller。

修复后的 adapter / regression test / benchmark README SHA256 分别为
`58389075ba50fb7c6e2205310c9f21f677615b17e8216ee5cce606738a68a0c5`、
`39a8bac724cf37a4187564afede69fecfd493a8429b867b71c5b9852a6b6f872`、
`4395ddc2b3ff01404c64dae1d1a46714d9e61db0a9ed40316bb398065c2ea07e`。
H200 CPU contract suite 为 `43 passed`；Ruff、py_compile 和 diff check 均通过。

### 7.2 最终 T16-R2 结果

最终 source commit 为 `93e5d1cb3121e42fc021b5fb616fbe0dd94090c1`，nonce
`cb1ec6d46d85612dfa8523ecabe93e77`，terminal root 为：

```text
/DATA/share/sana_wam_libero_nonformal_screens/t16/93e5d1cb3121/libero-t16-paired-one-task-chunk-closed-loop-fixed32-cb1ec6d46d85612dfa8523ecabe93e77
```

root/唯一 `RESULT.json` 分别冻结为 `0500`/`0400`；RESULT SHA256 为
`5e2f5d61f7715f952f1762c04f5ad99891f9eda1ae5256310498c23146bcc471`。
execution/harness 均 PASS，最终 verdict 为
`T16_PAIRED_ONE_TASK_CHUNK_CLOSED_LOOP_INTERFACE_VALID`；scientific diagnostic 为
`T16_PAIRED_ONE_TASK_UPDATE_EFFECT_OBSERVED`。

核心结果：

- PRE/POST 都从逐字节相同 reset fingerprint 出发，各完成 32 policy/env steps，
  generation steps 均精确为 `[1,29]`；paired generation noise signatures 相同。
- 中间精确完成 20 macro optimizer steps、160 forward/backward accumulation，
  FP32 optimizer master 有限且 4 个 trainable roots 均更新。
- 8 个 update samples 和 8 个 same-task heldout samples 的 post/pre loss ratio
  全部严格小于 1；median ratio 分别为 `0.0517542070`、`0.0730358128`。
- 相同初态第一 action 的 PRE/POST L2 为 `6.2480444081`，轨迹发生变化；两臂
  都未 success/done，累计 reward 均为 `0`，因此不得宣称 20-step 模型已学会任务。
- raw gripper score 超出训练 `[0,1]` support 的执行步由 PRE `26/32` 降为 POST
  `4/32`，只作为校准 diagnostic；所有 score 均 finite 并按固定 threshold 执行。
- 实际 scope 为 64 simulator steps、4 model generations；未加载/保存 SANA-WAM
  checkpoint，未执行正式训练或 benchmark evaluation。

T16 的核心架构与真实闭环接口门槛至此结束。下一步不再追加 architecture-validation
细节实验，直接完成正式训练所需的最小 admission 后启动训练，并用独立 LIBERO
checkpoint 做正式 benchmark evaluation。

## 8. 预算、root 和终态

- single GPU，同时最多一个 SANA-WAM model；不完整 2B training。
- real-data update 预算与 T15 完全相同：16 prepare、192 architecture forward、
  160 backward、20 optimizer step。
- rollout 预算：pre/post 各最多 32 policy/environment step；未提前终止时
  各恰好 2 次 model generation。
- 新 namespace：`/DATA/share/sana_wam_libero_nonformal_screens/t16/{source_commit[:12]}`
- 唯一 root：
  `libero-t16-paired-one-task-chunk-closed-loop-fixed32-{nonce}`
- root fresh/exclusive/one-shot；terminal root `0500`，唯一 RESULT/FAILED `0400`；
  禁止复用、覆盖、自动重跑或 post-freeze mutation。

## 9. Verdict 与解释边界

执行 PASS 要求：

1. 所有 source/predecessor/data/simulator/runtime pin 一致；
2. pre/post 从相同 settle 后 state/observation 开始，paired inference seed 相同；
3. 两臂均无 NaN/inf、shape/IPC/simulator/cache 错误，或发生合法的 simulator
   success/done early termination；
4. 若臂运行到 32 步，model generation 必须恰为 2，第二次使用新观测；
5. rollout 期间参数不变，只在中间 T15 update 阶段变化；
6. 未执行 checkpoint load/save、formal training、benchmark aggregate、admission 或 deploy。

有效执行 verdict：

```text
T16_PAIRED_ONE_TASK_CHUNK_CLOSED_LOOP_INTERFACE_VALID
```

pre/post action trace L2、MuJoCo state/eef/object displacement、reward、success/done、终止时间
与 trajectory divergence 都是 diagnostic。允许报告 `POST_ONLY_SUCCESS`、
`PRE_ONLY_SUCCESS`、`BOTH_SUCCESS`、`NEITHER_SUCCESS`或 `TRAJECTORY_IDENTICAL/DIFFERENT`，
但它们不是 success-rate classifier，不得据单初态宣称模型改善。

## 10. 停止与后续规则

- 若 interface 无法通过，冻结 FAILED root，只修根因后申请 fresh-root 复核；
  不把行为效果差自动归因为 plumbing bug。
- 若 interface PASS，下一步才是 4 tasks × 1 deterministic init 的完整-horizon
  post-update pilot；那是新实验，不在 T16 授权中。
- 若 28-action chunk 暴露控制频率/稳定性问题，先将其视为核心架构反馈；
  不用未对齐的 horizon-1 部署 hack 掩盖。
