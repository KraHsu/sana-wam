# CACH-SANA-WAM Stage 0 交付入口

> 状态：`DRAFT / STAGE0_ONLY / EXECUTION_BLOCKED`
>
> 适用主机：`H200`
>
> 工作树：`/home/zch/workspace/sana-wam`

本目录把
`docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md`
的 Stage 0 拆成可审计产物。它不是 Stage 1 实现、测试、训练、评测或 capture
授权。

## 本轮边界

本轮只允许：

1. 固定 H200 的治理入口、Git/source identity 和开发起点；
2. 写清 `ChunkActionLayout`、bootstrap、cache/commit、applied-action、
   初始化/checkpoint 和 data/scale 契约；
3. 建立 `REF-GDN-CORRECTED` commissioning 与 `CACH-A` 的 draft spec；
4. 写 fail-closed verifier/launcher skeleton 和测试代码 diff；
5. 做静态检查并提交实现前 diff。

本轮不允许：

- 安装或更新依赖；
- 运行测试；
- 启动任何 GPU forward/backward、训练、评测或 capture；
- 创建正式 `/DATA/share/sana_cach_wam_20260731/` run root；
- 修改或复用 AFCC formal root；
- 自动进入 Stage 1。

## 产物及其语义

| 产物 | 作用 | 当前可执行性 |
|---|---|---|
| `governance/START_HERE_20260730.md` | H200 根入口的 repository-side draft source | 只读入口 |
| `governance/AGENTS.md` | H200 agent 规则的 repository-side draft source | 只读入口 |
| `CHUNK_ACTION_LAYOUT_AND_BOOTSTRAP_DESIGN.md` | raw/video/latent/action/RoPE 精确映射 | Stage 1 前需 review |
| `HYBRID_CACHE_COMMIT_AND_APPLIED_ACTION_DESIGN.md` | typed cache、事务 commit、环境 ack | Stage 1 前需 review |
| `SOURCE_INITIALIZATION_AND_LAUNCHER_DESIGN.md` | source、初始化、checkpoint、launcher | Stage 1 前需 review |
| `DATA_AND_SCALE_DESIGN.md` | 数据闭包与 scaling pilot 约束 | 当前 blocked |
| `SOURCE_MANIFEST.draft.json` | 起草时已核验的输入 identity | draft，不是 authority |
| `CACH_A_CANDIDATE_SPEC.draft.json` | 首个 paired campaign 的冻结字段 | draft/blockers 非空 |
| `CACH_A_AUTHORITY.draft.json` | 明确拒绝执行的 authority 草案 | `decision=deny_execution` |
| `TEST_PLAN.md` | Stage 1–3 测试矩阵 | 本轮不运行 |

任何文件名中的 `draft` 都有严格含义：其 SHA 可以审阅，但不能被 launcher 当作
registered authority。只有 blockers 清空、独立 review 完成、生成无 `draft`
后缀的新 immutable artifact，并得到用户对相应阶段的单独授权，才可改变状态。
当前 verifier/launcher 只验证并执行这份 denial 语义；即使静态 pin graph 闭合，
报告仍固定 `static_graph_valid=false`、`execution_admission_valid=false`。它们
不得改名或复用于未来 executable admission；Stage 1 必须使用新 schema、新
authority 和新的 external trust-anchor review。
denial verifier 的退出码 `0` 最多表示“外部 anchor 所绑定的 draft denial bytes
彼此一致”；它的报告仍把 `static_graph_valid`、`inspection_complete` 和
`execution_admission_valid` 固定为 `false`，任何 launcher 都不得把该退出码当作
执行 admission。由于完整 Python package inventory 尚未冻结，denial verifier
不 import OmegaConf：它把 config 的 final raw SHA 同时绑定到 authenticated
contract 与 authority。exact mapping validator 和相关测试代码已提交但本轮不
执行，报告明确 `config_semantic_parser_executed=false`。

## 起始身份

- 主仓 HEAD：`605f1c134b4c983ff80f8489c4bc8847036329e2`
- Sana gitlink/worktree：`16b9cec673e3335724ba2d8db25de7f9ed229292`
- 开发计划 SHA256：
  `969d9668ffed07176844fcbd84062825f9d5f0af0325959b0068c38fe481f0dd`
- research report SHA256：
  `4bd78d3c37970ed4c3fc9c05a82faef44fbeac079a6541a694fa496e9d858010`

Stage 0 开始前 H200 已有用户工作：

- modified：`tests/test_phase6_candidate_eligibility.py`
- untracked：`%ln`、`analysis/`、`telemetry/`
- 本轮前一文档交付新增：
  `deep-research-report.md` 与主开发计划。

这些路径不属于 CACH Stage 0 修改边界，不得清理、覆盖或纳入新结论。

Stage 0 delivery 已将两份根治理镜像以 non-overwrite 方式部署为 regular 文件，
并与 repository-side draft source byte-identical；这只关闭 mirror presence，
不等于 draft 已进入 reviewed commit/bundle。

## 当前硬阻塞

1. RoboTwin HDF5 没有 timestamp；action/video 同率仍缺 provenance receipt；
2. 现有 train/deploy 使用 fixed-ATC，不能表达 causal latent 0；
3. 现有 cache 是无 schema/content-time 的可变裸 list，commit 不是事务性的；
4. policy 记录的是 command，不是 environment-confirmed applied action；这只阻塞
   deploy/G5 executed-action science，不阻塞 offline teacher-forcing G1–G4；
5. pretrained/load 路径使用 permissive key handling；
6. complete random-init 2B 的数据规模、预算和收敛依据未冻结；
7. Stage 0 denial launcher/verifier 已提交但未运行；未来 executable 五件套尚未
   实现或 independent review；
8. `REF-GDN-CORRECTED` 只有 commissioning design/blocker ledger，尚无机器可审计
   的独立 spec/config/authority，不能声称 fresh paired graph 已建立。
9. 当前 `growing_history=false` loader 仍枚举 `start>0` windows，而
   `growing_history=true` 会产生 nonzero clean prefix；episode-origin、
   no-clean-prefix sampler 与 layout-derived per-chunk proprio selector 尚未实现；
10. reserved-marker guard 已覆盖本轮列出的 canonical train/deploy/cache/smoke/
    stats/eval 入口，但仓内仍有 legacy checkpoint probes/audits、AFCC/Phase-6
    launchers 和 direct `Trainer` 构造旁路。冻结 AFCC source 本轮不修改；在独立
    surface inventory/guard review 完成前，不得声称 universal execution guard。
11. denial verifier/launcher 的 self-hash 发生在 Python 已启动并开始执行脚本
    之后，且 `git` binary 尚未 pin。它们是 externally anchored review aid，不是
    tamper-proof executable trust root；Stage 1 必须由独立 trusted runner 启动。

因此当前状态必须保持 `EXECUTION_BLOCKED`。
