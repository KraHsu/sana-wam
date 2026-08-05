# CACH-A4 R1 LAUNCHER ENVIRONMENT FIX

状态：`ONE_IDLE_SINGLE_GPU_FRESH_ROOT_SYNTHETIC_EXECUTION_AUTHORIZED`

规范环境：`H200`，`/home/zch/workspace/sana-wam`，2026-08-05。

## Decision

`CACH-A4-R1-LAUNCHER-ENV-FIX-v1` 是 frozen CACH-A4 的 additive、launcher-only
successor。底层 architecture 仍是
`CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1`。模型公式、task、
target、初始化 seed、每臂 200-step AdamW、metrics、thresholds 和 verdict 全部不变。

唯一修复是在任何 Torch、Dynamo 或 vendor import 之前固定：

```text
GDN_DISABLE_COMPILE=1
TORCHDYNAMO_DISABLE=1
```

这两个 flag 只绕过不兼容的 `torch.compile`/Inductor wrapper。它们不禁用 vendor
GDN，也不禁用 Triton 3.5.1 的 bare `@triton.jit` forward/backward kernels 或 vendor
autograd。R1 仍是 `VENDOR_KERNEL` / `EXPERIMENTAL_PATH`，不允许 proxy、reference
operator fallback 或模型语义替换。

## Frozen A4 source pins

以下六个 predecessor 文件必须保持 0444、字节不变：

| role | path | SHA256 |
|---|---|---|
| decision | `docs/cach_sana_wam/architecture_validation/cach_a4_state_stream/CACH_A4_STATE_CONDITIONED_CAUSAL_STREAM_ARCHITECTURE_DECISION.md` | `2a2efb3bda76dd36d2d1b98c0d26703de40704bc0744098a93b9820eac0b3a92` |
| card | `docs/cach_sana_wam/architecture_validation/cach_a4_state_stream/CACH_A4_STATE_CONDITIONED_CAUSAL_STREAM_RUN_CARD.json` | `b069c5fcbfde49e1f2e7a7ae0feb47067910ed680d09b28eb8459deb22ef618d` |
| bridge | `src/sana_wam/model/cach_av1b_a4_state_conditioned_causal_odd_stream.py` | `54924361009668a5c1cd113799812cc0f3d758b83528bf0aa917efba55a5fa6a` |
| config | `configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream.yaml` | `c49e03ad22a3ed8f79160817309714a674e0c8aa9766a1beb0eb5c8a7dabf835` |
| runner | `scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream.py` | `8c106c23b579c8ab0742a9d66a99b886223c063bf876fe3f368067251dcd9cce` |
| test | `tests/test_cach_av1b_a4_state_conditioned_causal_odd_stream.py` | `892b6e8d673a180a377b74d44a6fdd2818ced8a8b27dfbd64d628569baab0977` |

R1 source may import or wrap the pinned bridge lazily, but must not copy-edit, overwrite or
mutate any predecessor file.

## Frozen failed-run evidence

The first A4 root is immutable and must never be reused:

```text
root = /DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream/06f5d09127f8/cach-a4-state-stream-ac7fb52eb22be3cce184a60a41f7e98e
nonce = ac7fb52eb22be3cce184a60a41f7e98e
RESULT.json SHA256 = 07f5246d1ad0d8b076f60fdb9f41c5ab53a3db8f9c4c9d31524f705b98a26323
FREEZE_RECEIPT.json SHA256 = 0f93621d81498ce80187c67d9ce42ae233fa560a36784e401303d102c7403620
RUN_CONTEXT.json SHA256 = d46213a56f33f02a7a9b61e0c8fcc83f687d4739c50d9d10cbde56f046b331d1
runtime authority canonical SHA256 = a153de71bdfe4fcedff2d7c752a3a907c559dc97673ffce632fe5bc609e935be
execution statement SHA256 = a5d87f6a5812063d2b9ba600ba031fd9e9613d16d8a0bf388ff595f1054620dc
user authority text SHA256 = c3631ea159561fb7d13154c592e16e019384f111b798b3380c7bcf4a4e94350f
```

该 root 的 typed verdict 是 `INVALID_RUN`，terminal state 是
`HARNESS_REJECTED`。失败原因是 Torch 2.7.1 Inductor 尝试从 Triton 3.5.1 导入已移除
的 `triton.compiler.compiler.triton_key`。错误发生在首个 theta0 vendor forward 的
compile wrapper 中、optimizer loop 之前；optimizer update 为零。因此 R1 必须 fresh
initialize，不能 continuation、resume、root reuse 或 automatic rerun。

## No scientific delta

以下契约逐项继承 frozen A4：

- config schema `cach.cach_a4.state_conditioned_causal_odd_stream.config.v1` 和 screen
  schema 不变；
- shared task/common seed `2026080331`、candidate-only seed `2026080522` 和 initializer
  revision `cach-a4-state-conditioned-causal-odd-v1` 不变；
- pinned A3 synthetic task/target bytes、counterfactual pair order、chunk boundaries、
  action/state dimensions 不变；
- common/action scopes、exact-odd recurrence、typed bypass、losses、cosine LR
  `0.003 -> 0.00003`、每臂 200 macrosteps 不变；
- validity、causal JVP、operator GO/common stability/weak-stop thresholds 与 typed verdict
  映射逐字段不变。

R1 config 必须等于 frozen A4 config 的 parsed JSON object，仅允许在 `runtime` 增加
`launcher_environment_before_torch_or_vendor_import`，其值只能是上述两个 exact flags。

## Additive source plan

R1 的完整 source-six 只允许以下路径：

1. `docs/cach_sana_wam/architecture_validation/cach_a4_r1/CACH_A4_R1_LAUNCHER_ENV_FIX_DECISION.md`
2. `docs/cach_sana_wam/architecture_validation/cach_a4_r1/CACH_A4_R1_RUN_CARD.json`
3. `src/sana_wam/model/cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py`
4. `configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r1.yaml`
5. `scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py`
6. `tests/test_cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py`

## Fresh authority and one-shot execution binding

当前用户 fresh authority 的 exact UTF-8 文本为 `继续 A4`，SHA256 为
`c3631ea159561fb7d13154c592e16e019384f111b798b3380c7bcf4a4e94350f`。该 authority
只对本 R1 revision 生效，不继承失败 A4 root 的 execution authority，也不授权第二次
或自动 rerun。

唯一一次未来 GPU execution 被排他绑定为：

```text
nonce = 9e604d41d671ac4e705816857457e5a4
root = /DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r1/06f5d09127f8/cach-a4-state-stream-r1-9e604d41d671ac4e705816857457e5a4
physical GPU index = 0
GPU UUID = GPU-1ec28cfb-f501-23f3-f865-275a744ca053
```

规范 cwd 为 `/home/zch/workspace/sana-wam`。规范 env+argv command 为：

```text
env CUDA_VISIBLE_DEVICES=GPU-1ec28cfb-f501-23f3-f865-275a744ca053 FUSED_GDN_PRECISION=0 GDN_DISABLE_COMPILE=1 PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1 TRITON_CACHE_DIR=/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r1/06f5d09127f8/cach-a4-state-stream-r1-9e604d41d671ac4e705816857457e5a4/triton_cache /home/zch/workspace/sana-wam/.venv/bin/python scripts/run_cach_av1b_a4_state_conditioned_causal_odd_stream_r1.py --card docs/cach_sana_wam/architecture_validation/cach_a4_r1/CACH_A4_R1_RUN_CARD.json --config configs/experiments/cach_av1b_a4_state_conditioned_causal_odd_stream_r1.yaml --root /DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r1/06f5d09127f8/cach-a4-state-stream-r1-9e604d41d671ac4e705816857457e5a4 --nonce 9e604d41d671ac4e705816857457e5a4 --gpu-index 0 --gpu-uuid GPU-1ec28cfb-f501-23f3-f865-275a744ca053
```

上述进程还必须从 stdin 接收 canonical UTF-8 sorted compact JSON + LF runtime
authority；authority 必须绑定最终 frozen R1 source-six SHA、card SHA、同一 env/argv、
root、nonce、GPU index/UUID 和 exact execution statement。source freeze 和执行是后续
独立步骤；本 authorization-binding 阶段仍不得创建 namespace/root、查询或使用 GPU、
import CUDA/vendor model，或运行 synthetic optimizer screen。

禁止真实数据、checkpoint load/save、完整 2B、正式训练/评测、admission、AV2、
Global Stage 3、token/claim operation、自动 rerun 和 predecessor mutation。
