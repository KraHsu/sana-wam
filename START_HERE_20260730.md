# SANA-WAM H200 开发入口（2026-07-30，2026-07-31 更新）

> 当前状态：CACH-SANA-WAM Stage 0 文档与准入设计中。
>
> 这不是 release，也不是测试、训练、评测或 capture 授权。

本文件的部署镜像位于
`/home/zch/workspace/sana-wam/START_HERE_20260730.md`。repository-side draft source 位于
`docs/cach_sana_wam/stage0/governance/START_HERE_20260730.md`；两者必须
byte-identical。
只有纳入 reviewed commit/bundle 后，才能称为 tracked/recoverable source。

## 工作根与固定提交

| 对象 | H200 路径 | 固定身份 |
|---|---|---|
| 主仓 | `/home/zch/workspace/sana-wam` | `605f1c134b4c983ff80f8489c4bc8847036329e2` |
| Sana | `/home/zch/workspace/sana-wam/third_party/Sana` | `16b9cec673e3335724ba2d8db25de7f9ed229292` |
| AFCC handoff | 若恢复则必须由 manifest 指定 | `9586486f2a9f5172d57b325e32093a3e018d34c0` |

Sana 提交不保证存在于公开 origin。禁止先运行 `git submodule update`，也不得
checkout/reset/clean/stash/fetch 覆盖当前工作树。

## 必读顺序

1. `AGENTS.md`
2. `docs/agent_handoff/AGENT_HANDOFF_20260730.md`
3. `deep-research-report.md`
4. `docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md`
5. `docs/cach_sana_wam/stage0/README.md`
6. 当前阶段对应的 design/spec/source manifest

起草 Stage 0 时：

- 开发计划 SHA256：
  `969d9668ffed07176844fcbd84062825f9d5f0af0325959b0068c38fe481f0dd`
- research report SHA256：
  `4bd78d3c37970ed4c3fc9c05a82faef44fbeac079a6541a694fa496e9d858010`
- 仓内 handoff SHA256：
  `76f251fde90cde1a2b970f6360d95f4dc51074f41579baebe130a283a4ab7920`

后续 SHA 变化必须在 source manifest 中解释；聊天记忆不是 source of truth。

## 当前开发顺序

```text
Stage 0 source/design closure
  -> 用户审阅实现前 diff
Stage 1 contract implementation
  -> Stage 2 mini-model admission
  -> Stage 3 full-model update-free admission
  -> 每个 Stage 4 campaign 单独授权
  -> Stage 5 generated-prefix admission
  -> Stage 5.5 final full-horizon training
  -> Stage 6 唯一 closed-loop pair
```

不得跳级，也不得因为 GPU 空闲而自动启动下一步。

## 当前已确认外部输入

- LTX2 causal VAE：
  `/DATA/share/SANA-WM_streaming/ltx2_causal_vae`
- Gemma text encoder：
  `/DATA/share/gemma-2-2b-it`
- RoboTwin：
  `/DATA/share/RoboTwin2.0/dataset`

路径存在不等于 input closure。必须消费
`docs/cach_sana_wam/stage0/SOURCE_MANIFEST.draft.json` 中的 realpath、文件
SHA/size 和 unresolved blockers。初训禁止加载
`/DATA/share/SANA-WM_streaming/sana_dit/model.pt`。

## 当前 P0

1. RoboTwin 文件没有 timestamp，action/video rate provenance 未闭合；
2. fixed-ATC、partial tail 和旧 observed-prefix 契约必须替换；
3. cache 缺 typed content-time 与事务 paired commit；
4. server/environment 缺 canonical applied-action acknowledgement；
5. permissive checkpoint load 必须替换为精确 allowlist；
6. complete random-init 2B 的 data/scale/budget 未冻结；
7. CACH 独立 launcher/verifier 尚未通过 review。

这些 blocker 清空前，任何 executable authority 都必须拒绝启动。

## 不可宣称

- 当前没有 CACH 实现完成、测试通过、正式训练或 closed-loop 结果；
- 当前没有 504-step prospective formal 结果；
- Stage 0 文档不证明新架构有效；
- prospective run 不能称作历史 endpoint reconstruction。
