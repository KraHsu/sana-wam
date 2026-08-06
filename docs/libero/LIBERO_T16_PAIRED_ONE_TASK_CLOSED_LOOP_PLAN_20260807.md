# LIBERO T16：Paired One-Task Chunk-Closed-Loop Smoke

状态：**development plan；T15 停止条件已满足，T16 尚未创建运行 root，
尚未构造 simulator 或执行 GPU。**

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
- dataset task index：`5`
- task：`pick up the black bowl on the cookie box and place it on the plate`
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
- model gripper 必须在 `[0,1]`；`>0.5` 映射为 LIBERO open `-1`，
  其余映射为 closed `+1`。越界不自动 clamp，而是记录有效行为失败。
- 每个 environment step 仍把新观测送入 policy；但只有 action buffer 耗尽时
  才调用 model generation，必须分别记录 policy request 和 generation 计数。

## 7. 预算、root 和终态

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

## 8. Verdict 与解释边界

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

## 9. 停止与后续规则

- 若 interface 无法通过，冻结 FAILED root，只修根因后申请 fresh-root 复核；
  不把行为效果差自动归因为 plumbing bug。
- 若 interface PASS，下一步才是 4 tasks × 1 deterministic init 的完整-horizon
  post-update pilot；那是新实验，不在 T16 授权中。
- 若 28-action chunk 暴露控制频率/稳定性问题，先将其视为核心架构反馈；
  不用未对齐的 horizon-1 部署 hack 掩盖。
