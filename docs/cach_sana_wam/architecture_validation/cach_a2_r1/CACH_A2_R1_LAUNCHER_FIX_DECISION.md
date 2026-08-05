# CACH-A2 R1 LAUNCHER PREFLIGHT FIX

状态：`FROZEN_R1_LAUNCHER_FIX_SOURCE_AND_EXECUTION_AUTHORIZED`

日期：2026-08-05  
规范 SSH alias：`H200`  
规范实际节点名：`huaxiyun`  
规范工作树：`/home/zch/workspace/sana-wam`

本文件登记 additive fresh revision `CACH-A2-R1-LAUNCHER-FIX-v1`。底层
architecture 仍是 `CACH-A2-ZERO-ANCHORED-BIAS-FREE-v1`；screen schema、模型
API、模型参数、synthetic task、双臂 300-step AdamW、预算、metrics、thresholds、
verdict、无 token 和 direct-terminal root 契约均保持不变。

## 1. Authority

用户修复 authority 的逐字原文是：

> 修复

按 UTF-8、无尾随换行计算：

```text
SHA256 = 37a6939250543f0aad3e9c0c4067e40e48e4453cca242ad2ad6750c34e1575cb
```

该 authority 明确承接紧邻的 CACH-A2 execution 授权：用户已授权继续执行，且明确
要求不再重复整段授权文字。R1 不扩大能力，只修复阻止已授权 execution 到达 root
创建前的两个 launcher preflight bug。

## 2. Frozen predecessor

以下旧文件保持只读、字节不变：

| role | path | SHA256 |
|---|---|---|
| R0 decision | `docs/cach_sana_wam/architecture_validation/cach_a2/CACH_A2_ZERO_ANCHOR_ARCHITECTURE_DECISION.md` | `280e17b6e9a7433e64b20ceed587e998ab0ef1322fe7cc7467ee6a56f9589363` |
| R0 card | `docs/cach_sana_wam/architecture_validation/cach_a2/CACH_A2_ZERO_ANCHOR_RUN_CARD.json` | `9d3c3973234df8a9593b254161ca07bf72f343c4475d9985e2901153b82a9713` |
| R0 bridge | `src/sana_wam/model/cach_av1b_a2_zero_anchor_vendor_gdn_bridge.py` | `a535003429c3cc06a7dc969de28622db47f6b71974afb967cf199e4f40c95744` |
| R0 config | `configs/experiments/cach_av1b_a2_zero_anchor_vendor_gdn_bridge.yaml` | `4803b8ec5065b31c701ccda4a71b713da7345ea711038fafbd3fe259490c013c` |
| R0 runner | `scripts/run_cach_av1b_a2_zero_anchor_vendor_gdn_bridge.py` | `b203de02eeeeaf5a8e9b1899e6d70cc4e7013fb585eaad4e909e9a7414616921` |
| R0 test | `tests/test_cach_av1b_a2_zero_anchor_vendor_gdn_bridge.py` | `bbb44cfe2fe6e6dc25a5bd46863fc46b0ddfec162c2a9f83c4e220c3b7c8bdd6` |

旧 AV-1B review300 closure、frozen roots、RESULT、RAW evidence、freeze receipt 和
consumed claim 继续由 R0 card/config 的完整 pins 继承，并在 R1 card/config 中逐项保留。

## 3. Exactly two preflight fixes

### 3.1 SSH alias 与实际节点名分离

R0 错误地把规范 SSH alias `H200` 当成 `platform.node()` 的期望值。R1 冻结两个
不同字段：

```text
canonical_ssh_alias    = H200
expected_platform_node = huaxiyun
```

launcher 只能分别检查：authority/identity 使用 `H200`；运行时
`platform.node()` 只与 `huaxiyun` 比较。禁止比较 `platform.node()` 与 SSH alias。
`huaxiyun` 由 frozen predecessor `RUN_CONTEXT.json` 与当前只读观察共同支持。

### 3.2 Torch distribution local version 精确比较

R0 对 `importlib.metadata.version("torch")` 错误执行 local-suffix split，并期待
`2.7.1`。H200 安装的精确 distribution metadata 是：

```text
2.7.1+cu128
```

R1 必须逐字比较完整值 `2.7.1+cu128`。禁止 `split("+", 1)`、去除 local suffix、
接受 base-only `2.7.1`，或使用宽松 prefix/集合匹配。该修复不 import torch；实际
Torch/CUDA import 仍仅能发生在全部 preflight、authority、GPU 和 root checks 后。

## 4. No other delta

下列内容必须与 R0 保持相同：

- architecture id、bias-free zero-anchor seam、typed mask 和 direct bypass；
- reference/candidate topology、vendor GDN class、autograd kernel 与 single-call chunks；
- synthetic bytes/formulas、fresh seeds、每臂 AdamW 300 steps、总计 600 updates；
- wall 1350 seconds、GPU/RSS 16 GiB、root 512 MiB、minimum free 100 GiB；
- 所有 required metrics、validity、GO/common-mode/strong-stop thresholds；
- `OPERATOR_GO`、两个 inconclusive subtype 和 `OPERATOR_STOP` 映射；
- predecessor review token 已消费；R1 不创建、派生、重置或消费 token；
- valid result 在唯一 root 内 direct terminal 后冻结，无 provisional/claim ledger；
- 禁止真实数据、checkpoint load/save、完整 2B、正式评测、AV2 和 Global Stage 3。

任何除以上两个 preflight fixes 与 fresh revision identity/path/root 外的差异均为
`IMPLEMENTATION_INVALID`。

## 5. Fresh additive identity

R1 quartet 使用 `_r1` suffix。唯一 runtime identity：

```text
nonce = dc17cb7bdbd70955ea4b91212f0d4ac3
root  = /DATA/share/sana_cach_wam_nonformal_screens/cach_a2_r1/06f5d09127f8/cach-a2-r1-launcher-fix-dc17cb7bdbd70955ea4b91212f0d4ac3
```

旧 root 不复用。R1 root/namespace 只能由最终 execution launcher 按 exclusive-create、
fail-closed 契约创建；本静态起草阶段禁止创建任何 root、namespace 或 token ledger。

## 6. Final materialization invariant

最终 materialization/freeze 前必须回填 card、decision 与 R1 quartet 的所有完整
SHA256，并重新执行 duplicate-key JSON、AST、path、permission 与 SHA 静态审计。
最终冻结文件不得包含任何 placeholder；任一 placeholder 遗留都必须 fail-closed，
且不得创建 namespace/root、import Torch/vendor 或开始 execution。
