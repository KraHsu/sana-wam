# CACH-A4 R2 DIAGNOSTIC MASK SHAPE FIX

状态：`ONE_IDLE_SINGLE_GPU_FRESH_ROOT_SYNTHETIC_EXECUTION_AUTHORIZED`

规范环境：`H200`，`/home/zch/workspace/sana-wam`，2026-08-05。

## Decision

`CACH-A4-R2-DIAGNOSTIC-MASK-SHAPE-FIX-v1` 是 frozen CACH-A4-R1 的 additive、
diagnostic-only successor。底层架构仍为
`CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1`。模型、task、target、
初始化 seeds、fresh-init AdamW、每臂 200 steps、losses、metrics、thresholds 与 verdict
mapping 均不改变。

唯一代码差异位于训练完成后的 inactive-video diagnostic。Frozen A4 bridge 将
`[B,T,1]` 的 `action_present_mask` 直接扩展到 `[B,T,C,1,1]`，因右对齐广播而失败。
R2 仅将：

```python
~correct.action_present_mask.expand_as(correct.action_delta)
```

机械改为：

```python
~correct.action_present_mask.unsqueeze(-1).unsqueeze(-1).expand_as(
    correct.action_delta
)
```

该修复只使既有 inactive-output exact-zero diagnostic 可执行，不改变 forward、backward、
参数更新或任何科学量。R2 bridge 必须能机械还原为 frozen A4 bridge，且只允许这一处 hunk。

## Frozen R1 evidence

R1 root 永久只读且不得复用：

```text
/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r1/06f5d09127f8/cach-a4-state-stream-r1-9e604d41d671ac4e705816857457e5a4
```

- `RESULT.json` SHA256 `ba24e73a316dcc92e2d759f90d9e309db9747db88f60842440bcd9e52b3875e0`
- `FREEZE_RECEIPT.json` SHA256 `383ca8321f454d62e9ec966f5db9e737532a2d2d6d1ee266b11e57972914c6f7`
- `RUN_CONTEXT.json` SHA256 `b34c8317fdce4e452be1ef209f3e3ea485d6990165b0b302053da42eba1a8a15`
- `PROGRESS_EVENTS.json` SHA256 `6be08df7a57e0871e3c278334242fb8ef4f944e26182fe9a22252abd3ebfca3e`

Frozen R1 source-six 也必须逐字保持 0444 并由 R2 runner/card 同时绑定：

- decision `aa06060c40ea2fe1abad1041dd8ebfb236619ce51cc36ac1718cca9e4d5e67f6`
- card `faddcddcb7a1a6a69f013f615f39e99e78b128b50fe7d4a6f67fe3b8a335c0a7`
- bridge `abd8a4f1d671cc7d0af735724a4034f86b85f05b5c0448dcfb2fc4ba2af31d1c`
- config `9acf0425fa8370be546686e7ee23e5c764a23a804b70dc125ee25c772e3317ae`
- runner `4c273186ebc0ff61902f25b3a38b5e59fad51a701d59e7fc0b4351bac3bdf600`
- test `b4b386d439344b0c6c041544a5c79b0d016113d4f1e030c88e77cc5f6c27f606`

R1 已证明 launcher 修复有效：双环境变量在 import 前生效，真实 vendor Triton forward /
backward kernels 编译并执行，不再出现 `triton_key` 错误。最后进度事件在 reference、
candidate-common 与 candidate-action 三个 optimizer step 后记录 `200`；common loss 从
`0.1086650491` 降至 `0.0033002612`，candidate delta loss 从 `0.2000000179` 降至
`0.0006579692`。随后才在上述 diagnostic 广播处终止。因此 R1 是
`HARNESS_REJECTED / INVALID_RUN`，不能据此判定架构效果，也不能 post-hoc 补写 metrics。

## R2 source and execution boundary

R2 只新增以下六文件：

1. `docs/cach_sana_wam/architecture_validation/cach_a4_r2/CACH_A4_R2_DIAGNOSTIC_MASK_SHAPE_FIX_DECISION.md`
2. `docs/cach_sana_wam/architecture_validation/cach_a4_r2/CACH_A4_R2_RUN_CARD.json`
3. `src/sana_wam/model/cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py`
4. `configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r2.yaml`
5. `scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py`
6. `tests/test_cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py`

R2 config 必须与 frozen R1 config 字节相同，并继续在任何 Torch/vendor import 前设置
`GDN_DISABLE_COMPILE=1` 与 `TORCHDYNAMO_DISABLE=1`；这不会禁用 bare Triton vendor
autograd kernels。

Fresh user authority 已固定为 exact UTF-8 文本
`授权 A4-R2 单GPU合成运行`（31 bytes），SHA256
`8dc3ad2e8bd1edbf8c276ff8c9979c3a923b24d026e0a9eca8dcf518311028b0`。
该 authority 仅绑定本 R2 revision 的一次 fresh-root synthetic execution，不继承 R1
authority，也不允许 automatic/second rerun。

唯一执行身份固定为：

```text
nonce = f294a23c285082ae61b1685accec416e
root = /DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r2/06f5d09127f8/cach-a4-state-stream-r2-f294a23c285082ae61b1685accec416e
physical GPU index = 0
GPU UUID = GPU-1ec28cfb-f501-23f3-f865-275a744ca053
```

规范 cwd 为 `/home/zch/workspace/sana-wam`，规范 env+argv command 为：

```text
env CUDA_VISIBLE_DEVICES=GPU-1ec28cfb-f501-23f3-f865-275a744ca053 FUSED_GDN_PRECISION=0 GDN_DISABLE_COMPILE=1 PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1 TRITON_CACHE_DIR=/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r2/06f5d09127f8/cach-a4-state-stream-r2-f294a23c285082ae61b1685accec416e/triton_cache /home/zch/workspace/sana-wam/.venv/bin/python scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r2.py --card docs/cach_sana_wam/architecture_validation/cach_a4_r2/CACH_A4_R2_RUN_CARD.json --config configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r2.yaml --root /DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r2/06f5d09127f8/cach-a4-state-stream-r2-f294a23c285082ae61b1685accec416e --nonce f294a23c285082ae61b1685accec416e --gpu-index 0 --gpu-uuid GPU-1ec28cfb-f501-23f3-f865-275a744ca053
```

进程 stdin 仍须接收 canonical UTF-8 sorted compact JSON + LF runtime authority，绑定
最终 source-six SHA、card SHA、上述 env/argv/root/nonce/GPU 和 exact execution
statement。本 authority-binding 阶段不得创建 namespace/root，不得 import/execute
CUDA/vendor model 或运行 synthetic optimizer。

后续唯一运行仍只允许 idle single-GPU、fresh-root、fresh-initialization、synthetic
non-formal screen。禁止真实数据、checkpoint load/save、完整 2B、正式训练/评测、
admission、AV2、Global Stage 3、token/claim、自动 rerun、旧 root 复用或 predecessor mutation。
