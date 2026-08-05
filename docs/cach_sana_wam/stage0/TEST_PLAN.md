# CACH-SANA-WAM Contract Test Plan

> 状态：`DRAFT / ALL_TESTS_NOT_RUN`
>
> 本轮只提交测试设计与 Stage 0 fail-closed 测试代码 diff，不授权执行。

## 1. 阶段与授权

| 阶段 | 范围 | 本轮状态 |
|---|---|---|
| Stage 0 static | JSON/schema/source/diff/governance 静态检查 | 可做文本级检查 |
| Stage 1 unit | pure layout、schema、selector、cache transaction | 未授权运行 |
| Stage 2 mini-model | synthetic CPU/GPU numerical/causal tests | 未授权 |
| Stage 3 full-model | real shape/dtype、无参数更新 | 未授权 |
| Stage 4+ | 训练与科学 endpoint | 未授权 |

测试通过不能替代下一阶段授权；skip、xfail 或空 cohort 不能算通过 load-bearing
gate。

## 2. Stage 0 fail-closed tests

当前 diff 新增：

```text
tests/test_cach_stage0_contract.py
tests/test_cach_stage0_train_guard.py
tests/test_cach_stage0_deploy_guard.py
tests/test_cach_stage0_auxiliary_guards.py
tests/test_cach_stage0_checkpoint_config_guard.py
tests/test_cach_stage0_execution_surface_guards.py
```

它们应验证：

- source manifest 明确保留 dataset/rate/runtime blocker；
- `CACH-A` 只有一个 delta，root/endpoint 未设置；
- candidate 任意翻转 execution/scientific 状态都会失败；
- draft authority 只能 `decision=deny_execution`；
- launcher skeleton 不包含 torch/GPU/root mutation/worker spawn 执行路径。
- authority pin path 拒绝 absolute、`.`、`..`、symlink escape 与错误 role path；
- JSON duplicate key/non-finite/超大输入 fail-closed；
- denial verifier 缺 external authority SHA、治理 mirror 或 external inventory 时
  不能报告 admission valid；
- raw `cach_stage0` marker 的值无论为 mapping/null/false 都在 dotlist merge、
  torch/model/dataset import、CUDA/NCCL 和任何 root/lock/cache/temp/output 创建前
  被本轮明确列出的 train/deploy/checkpoint loader/policy server/precompute/
  smoke/stats/eval canonical surfaces 拒绝；
- deploy overlay 与 checkpoint `config.yaml` 两个 source 都覆盖；
- shell outer guard 位于 run lock、persistent root、temp config 之前。

这不是全仓 universal guard 证明。legacy checkpoint probes/audits、AFCC/Phase-6
launchers 与 direct `Trainer` construction 仍在 `SURF-001` 下；后续必须加入可
枚举所有 config/checkpoint/model execution surface 的 inventory test，新增入口
默认 fail，且不得为此修改冻结 AFCC formal source。

本轮不运行该文件。

## 3. Stage 1 pure contract tests

计划新增：

```text
tests/test_cach_action_chunk_layout.py
tests/test_gdn_ar_observed_prefix_contract.py
tests/test_cach_config_schema.py
tests/test_cach_checkpoint_key_allowlist.py
tests/test_cach_trainable_parameter_allowlist.py
tests/test_cach_afcc_isolation.py
tests/test_cach_failure_receipt.py
```

### 3.1 Layout/bootstrap

- raw action token `t` 唯一对应 destination raw frame `t+1`；
- latent 0 action span 为空；
- `K=3,tc=8,vs=1` 且 equal-rate 已证明时：
  首块 16 actions、后续完整块 24 actions；
- `L=8,K=3,r=8` tail 为 `[40,56)`；
- arbitrary `L/K/r` 下 valid action 无 gap/overlap；
- partial latent/action tail 不被 floor 丢弃；
- pad 不进入 condition、loss 或 Action RoPE；
- `LAYOUT_SPEC_SHA` 与 instance digest 分离；
- timestamp/mask/bounds/RoPE 任一篡改都会失败；
- production `row_start_raw_index` 必须为 0；普通 `start>0` sliding window以
  `NONZERO_EPISODE_ROW_START` 失败；
- legacy `growing_history=true` 虽从 0 开始但产生非零 clean prefix 时仍失败；
  Stage 1 新 episode-origin/no-clean-prefix sampler 才能通过；
- fixed-window `start>0` row 无法冒充 episode-origin row；
- chunk 0 proprio raw index 为 0；continuation 是上一 committed action span 的
  terminal observed state；
- equal-rate `K=3,r=8` 的 selected proprio indices 是 `0,16,40`；
- model-facing batch 不含完整 future `proprio_seq`，boundary 后 state 失败；
- 缺 rate receipt 时禁止设置 `equal_rate_proven`；
- 只接受 `first_frame_pinned + observed_prefix_chunks=0`；
- 旧 `ar_observed_prefix_chunks` key 即使为 0 也失败；
- 非零旧 clean-prefix 字段失败。

### 3.2 Config/source/checkpoint

- `cach_sana_wam_v0` 在显式 dispatch 实现前保持 unknown variant；
- Stage 1 dispatch 不允许 fallback 到 `gdn_autoregressive`；
- 初训 `model_path`、`init_dit_from` 必须为 null；
- 初训 checkpoint 参数出现即失败；
- resume/deploy missing/unexpected key set 必须精确为空或等于新 authority 的
  literal allowlist；
- source/runtime/data pin 漂移失败；
- CACH config 出现 AFCC weight/reference/authority 失败；
- action representation/order 精确为 registered absolute EEF 20D，
  `delta_action=false` 不能依赖 loader default；
- `F` 在 `Trainer` 构造前必须为 0 且无 reference。

### 3.3 Trainable inventory

- VAE 与 text encoder 全部 frozen；
- DiT/GDN、action backbone、proprio encoder、action conditioner 全部 trainable；
- reference/candidate shared tensors shape/name 集合一致并保存相同 init digest；
- operator-specific tensors 只按各自 design/seed 出现；
- `training.freeze` 不能包含整个 `video_backbone.dit`；
- action zero-init adapter 在参数级确为零且 identity bypass 闭合。

## 4. Stage 2 mini-model tests

计划新增：

```text
tests/test_cach_future_visibility.py
tests/test_cach_action_to_video_conditioning.py
tests/test_cach_cache_content_time.py
tests/test_cach_hybrid_cache_contract.py
tests/test_cach_applied_action_ack.py
tests/test_cach_anchor_schedule.py
tests/test_cach_block_attn_res_identity.py
tests/test_cach_self_forcing_equivalence.py
tests/test_cach_self_forcing_target_identity.py
```

### 4.1 Causal visibility

- chunk `>c` 的 video/action perturbation 不改变 chunk `<=c` output/JVP；
- chunk `c` clean target 不可见；
- chunk `c` noisy tokens 可按注册 chunk-bidirectional operator 共同交互；
- 测试不得错误升级为 frame-causal；
- clean/noisy duplicate 不跨域泄漏。

### 4.2 Cache/commit

- denoise 前后所有 temporal/depth state digest byte-identical；
- paired commit 每个 completed chunk 恰好一次；
- 任一 layer/action validation 失败时原 state 完全不变；
- GDN state full-history；
- softmax current + exactly one previous committed chunk；
- AttnRes depth state只活在一次 forward；
- content-time、layer type、shape、dtype/device 全部核对；
- reset 清空 video state、action cursor、RoPE cursor、pending transaction 和
  depth state；
- stale/double/out-of-order commit 失败。
- receipt write/fsync/read-back 任一失败时 old pointer/revision/cursor 保持不变；
- durable receipt 后 publish/verify 故障 poison episode/process，后续 reader不能
  消费 old/new state；重启只能 replay 并复核 receipt-bound state digest；
- runtime storage/data_ptr alias id 不进入 portable digest，跨进程 logical tensor
  id + tensor bytes 可重建同一 manifest SHA。

### 4.3 Applied action

- digest-only ack 失败；
- commanded values 不能进入 executed-action commit；
- inline canonical tensor 与 immutable ref 两种方式都重算 dtype/shape/order/hash；
- layout、episode、chunk、observation interval、controller transform 任一不匹配
  失败；
- partial applied span、drop、clip/transform 未声明失败；
- 环境不提供 applied values 时只能产生
  `commanded_action_only/scientific_eligible=false` telemetry。

### 4.4 Action conditioning

- zero-init adapter output identity；
- zero-init 状态用 parameter-gradient/JVP 证明 action wiring；
- 非科学 synthetic nonzero adapter 才允许做初始 output sensitivity；
- block 0 当前 self-attention cache 不得被描述为已经读取 post-attention action；
- later block/cache 是否读取 action按 seam design 精确测试；
- bootstrap no-action slot、pad mask、condition dim/dtype 不允许广播修补。

AttnRes enabled tests只在 `ATTNRES_DESIGN` source pin 后加入；generated-prefix tests
只在 `SELF_FORCING_OBJECTIVE` 闭合后加入。disabled stub 不得冒充 enabled
implementation。

## 5. Stage 3 full-model update-free admission

需要新的独立临时 root、用户授权和资源复核。pair 两端依次运行，不训练、不保存
候选 checkpoint：

- complete 2B random-init real shape/dtype forward/backward wiring；
- full-sequence causal 与 deploy chunk-cache equivalence；
- future perturbation；
- cache/layout/receipt digest；
- BF16/FP32 residual；
- finite gradient/JVP；
- peak CUDA/host memory 与 walltime。

G3 的 zero-init action adapter不要求 θ0 output 已对 action 敏感；只要求参数梯度/
JVP wiring。任何 GPU test 的 skip 都不算 admission。

## 6. Test artifact rules

每次获授权执行时：

1. 使用新 temporary 或 immutable root；
2. 记录 source/runtime/input/test-selection SHA；
3. 保存完整 stdout/stderr 与 per-test status；
4. 首个 load-bearing failure 立即停止后续依赖 gate；
5. failure receipt exclusive publish、fsync、verify 后冻结；
6. 代码/config/test 变化产生新 revision；
7. 禁止重写失败 root 或只重跑失败子集后拼接“全通过”。

## 7. 当前结论

```text
test_code_written: stage0 contract only
tests_run_this_round: 0
unit_admission: NOT_RUN
numeric_admission: NOT_RUN
full_model_admission: NOT_RUN
training_authorized: false
```
