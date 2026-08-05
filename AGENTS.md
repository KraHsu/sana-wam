# SANA / CACH / AFCC Agent Rules（H200）

本文件的部署镜像位于并适用于
`/home/zch/workspace/sana-wam/AGENTS.md` 及其子目录。repository-side draft source 位于
`docs/cach_sana_wam/stage0/governance/AGENTS.md`；两者必须 byte-identical。
只有纳入 reviewed commit/bundle 后，才能称为 tracked/recoverable source。

## 新 agent 首轮

新 agent 首轮必须只读：

- 先阅读 `START_HERE_20260730.md`、
  `docs/agent_handoff/AGENT_HANDOFF_20260730.md`、
  `docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md`
  和 `docs/cach_sana_wam/stage0/README.md`；
- 允许源码/文档读取、`git status/log/show/diff`、hash、磁盘/GPU/进程检查；
- 禁止编辑、安装依赖、fetch、checkout、reset、clean、stash、submodule
  update、测试、训练、评测或 capture；
- 首轮先报告 source、环境、launcher 与 admission 阻塞，等待用户确认。

不得假定一次旧会话的授权能自动延续到新阶段或新 campaign。

## 固定仓库身份

- 主仓：`605f1c134b4c983ff80f8489c4bc8847036329e2`
- Sana：`16b9cec673e3335724ba2d8db25de7f9ed229292`
- AFCC handoff：`9586486f2a9f5172d57b325e32093a3e018d34c0`

不得假定这些对象存在于 origin。禁止用远端内容覆盖 bundle 恢复对象或 gitlink。
工作树不是 clean release；所有修改前后都要记录完整 `-uall` status。

以下既有路径属于用户/历史工作，不得清理、覆盖或顺手修改：

- `tests/test_phase6_candidate_eligibility.py`
- `%ln`
- `analysis/`
- `telemetry/`
- 任何冻结 `/DATA` root、receipt 或 failure evidence。

## 当前 CACH 阶段边界

当前开发必须服从
`docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md`。

- Stage 0 只建立 source/governance/design/spec/verifier/launcher skeleton；
- Stage 0 不授权运行测试或模型执行；
- Stage 1 代码实现需要 Stage 0 diff 审阅后的单独授权；
- Stage 2/3 测试和 full-model admission 分别需要对应授权；
- 训练、评测、capture 和正式 `/DATA` root 始终需要独立 admission 与用户授权；
- campaign 之间不得自动推进。

## CACH 科学契约

- 初始 capability 线为 from-step-0 complete hybrid；只冻结 source-pinned
  LTX2 VAE 与 text encoder；
- 初训必须拒绝 pretrained video-DiT；resume/deploy 只允许同 revision exact keys；
- `REF-GDN-CORRECTED`、`CACH-A/H/R/SF` 是顺序、独立、单 delta campaign；
- bootstrap 固定为 `first_frame_pinned + observed_prefix_chunks=0`；
- fixed-ATC 禁用，train/deploy 必须共享 `ChunkActionLayout`；
- GDN temporal state 为 full-history；softmax 只允许 exactly one previous
  committed chunk；AttnRes depth state 不跨 temporal chunk；
- denoise 必须只读；video/action 只能 paired、单次、事务 commit；
- commanded action 不得称 applied action；没有 canonical applied values 时
  executed-action science fail-closed；
- `SELF_FORCING_OBJECTIVE` 未闭合前禁止创建可执行 `CACH-SF`；
- `DATA_AND_SCALE_DESIGN` 未闭合前禁止 final full-horizon training。

不得使用 DAgger、人工 recovery、rerank、best-of-N、temporal ensemble、
approximate replay、loss subtraction 或 post-hoc checkpoint/task/seed 选择。

## AFCC 隔离

CACH capability authority、root、source、receipt 与 AFCC formal 完全隔离。
CACH reference/candidate 均不得实例化 AFCC `F`。如共享 schema 强制出现该字段，
必须在 `Trainer` 构造前设 `F=0` 并删除 AFCC reference；否则 fail-closed。

若未来另做 AFCC formal，仍须满足：treatment 原生 `F=1`；control 在
`Trainer` 构造前 `F=0` 并删除 reference。不得宣称已有 504-step formal 结果。

## 修改与验证

用户确认某阶段后，只做该阶段的最小修改：

1. 先核对 H200 source SHA 和 dirty paths；
2. 用独立新文件/adapter 修根因，避开无关改动；
3. 先静态检查，再按阶段授权运行轻量测试；
4. 任何失败使用新 temporary/immutable root，保留 failure evidence；
5. 报告 diff、验证范围和仍未通过的 admission。

不得把“代码已写”“测试通过”“admission 通过”“训练完成”和“科学假设成立”
互相替代。

## 资源与安全

- 只读检查并避让所有无关进程；不得停止、修改或争抢他人 GPU 作业；
- 大产物只能写入经字节级容量 gate 确认的 `/DATA` 新 root；
- root 必须 direct、empty、排他创建；已存在即在 GPU reservation 前失败；
- 先 canonical receipt/inventory、`fsync`、verify，再冻结 `0400/0500`；
- 不删除、修改或复用失败/冻结 root；
- 不在仓库、日志或文档中写入密码、token、SSH 私钥等凭据。
