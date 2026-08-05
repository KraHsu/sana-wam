# CACH-A4 AV-2 TINY REAL-DATA OVERFIT DECISION

状态：`SCAFFOLD_ONLY_AV2_EXECUTION_NOT_AUTHORIZED`

规范环境：`H200`，`/home/zch/workspace/sana-wam`，2026-08-05。

## 1. Decision

下一项核心架构验证是 **AV-2 tiny real-data overfit**，不是 production-path
integration。冻结的 architecture-validation-first plan 明确规定：actual-operator
single-batch GO 只解锁 AV-2；只有 AV-4 `GO_CONFIRMED` 才解锁 AH-1、完整 C0--C8、
dispatcher/cache owner、launcher 与 production parity。

本 successor 显式接受 CACH-A4 相对原单一 CACH-A 假设的偏离，继续验证：

```text
CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1
```

A4-R2 的 valid typed verdict
`OPERATOR_GO_CAUSAL_STREAM_COMMON_STABLE` 在本 decision 中仅映射为 governing plan
的 `OPERATOR_GO` **eligibility**。该映射不自动授权 AV-2，不授权真实数据、GPU、
optimizer/update、root 或 rerun。

## 2. Frozen predecessor

A4-R2 source-six、result root 和 receipt 永久只读：

- card SHA256 `a45b55ffea796aa1edbcb46fae77654aa850965c312ed72b26797dffa4b1e6c6`
- bridge SHA256 `848d45c973de7c17995265f53d95ad92a357dad09bfabe76c8e58731b665e122`
- RESULT SHA256 `fabce5d1ac38294f79a6313c35041a723f4cb1055045ce9825a6f24d4202babd`
- FREEZE_RECEIPT SHA256 `ec70c0c409625b0d6d0d3c3e33d841e75313eb10ccea2b30f725be1fd0094a03`
- root `/DATA/share/sana_cach_wam_nonformal_screens/cach_a4_state_stream_r2/06f5d09127f8/cach-a4-state-stream-r2-f294a23c285082ae61b1685accec416e`

Phase-C 仍只是 pure-Torch production-shaped interface proxy。其 source manifest
SHA256 为 `0b88a76c5d2bf73a72a830fb58d9faac091af77e738f341f843664eaced3ffff`；
全部固定 member 保持字节不变，且不得把它称为 AV-2 actual operator 或 production
parity。

## 3. Architecture preserved in AV-2

AV-2 必须保持 A4-R2 的核心结构，不重新搜索架构：

1. REF 与 candidate 使用 action-blind、实际 CUDA/Triton vendor GDN common trunk；
2. candidate 仅增加最终 common video output 之后的 A4 delta stream；
3. stream 读取 detached、parameter-free RMS-normalized common hidden state；
4. action/state/write/output projection 全部 bias-free；
5. injected feature 为 `0.5 * (g(s,a) - g(s,-a))`，保持 exact odd；
6. decay 只读取 common state，不读取 action；
7. anchor、no-action、seam-disabled 和 typed-inactive 路径直接输出 exact-zero delta；
8. target 不得进入 forward；不得使用 loss subtraction；
9. reference/candidate 从 fresh common initialization 开始，不继承 A4-R2 参数；
10. correct/shuffled/no-action 使用同一 observation、target、mask 和数据顺序。

AV-2 仍走 `VENDOR_KERNEL / EXPERIMENTAL_PATH`。它不要求 public wrapper、durable
owner 或完整 production cache schema，也不能产生 Global Stage 3 结论。

## 4. Tiny real-data contract to bind before execution

当前 scaffold 不读取数据。后续独立 data-binding phase 必须在训练前固定并冻结：

- dataset root/manifest SHA 与数据 adapter source SHA；
- 一个 deterministic、未按模型结果挑选的 RoboTwin task/episode/window selector；
- 8 个 train windows 和 8 个 held-out windows，二者 episode/window identity 不重叠；
- 每个 window 恰好 33 个 observation/state rows 和 32 个 action transitions；
- observation-to-`[5,3,1,1]` 的 deterministic、checkpoint-free projection；
- action normalization、camera、raw indices、chunk/layout 与 proprio indices；
- 非空 typed mask、action variance、shuffle 真错配、target motion 非退化；
- selection manifest 在任何模型构造、GPU import 或 optimizer 创建前冻结。

若可用数据不能满足这些条件，结果只能是 `DATA_INADEQUATE`，不得更换 window 后继续
训练，也不得形成架构负结论。任何 VAE/text/model checkpoint 均不在 AV-2 范围内。

## 5. Fixed screen semantics

后续 execution card 的上限固定为：single seed、串行两臂、每臂最多 1,000 AdamW
steps 或 60 分钟（先到者停止）。只报告 final-step：

- train 与 held-out MSE decrease；
- candidate correct/shuffled/no-action 对照；
- candidate 相对 reference 的 held-out error；
- action seam gradient、update、JVP 和 exact-zero/odd/causal diagnostics；
- vendor forward/backward/JIT execution evidence。

初始 AV2 GO line 在结果前冻结为：所有有效性检查通过；两臂 train loss 均下降；
candidate held-out correct loss 相对自身 theta0 至少下降 5%；candidate held-out correct
相对 shuffled 与 no-action 各至少改善 5%；candidate held-out correct 不劣于 reference。
未达线时按 governing plan 和已消费 review token 映射，不允许自动 rerun。

## 6. This scaffold authority

用户本轮 exact statement `继续`（UTF-8 6 bytes，SHA256
`7c9691192f1b73408bbe4c0cb6d00db94375ca9d8fce0a0d5985e7a5178f083f`）仅按最窄范围
解释为：允许 additive decision/card/source/config/runner/test scaffold、静态校验与 CPU
synthetic focused tests。

本轮明确禁止：读取真实数据、创建 data selection 或 run root、GPU/CUDA/Triton/vendor
执行、optimizer/parameter update、训练、checkpoint load/save、完整 2B、正式评测或
admission、AV-3、Global Stage 3、deploy、token/claim 操作、旧 root 复用和任何 frozen
predecessor mutation。

完成 scaffold 后，真实 AV-2 execution 仍需一句新的明确授权，并由独立 immutable
execution card 绑定 exact data selection、source SHA、GPU UUID、root/nonce、预算与命令。
