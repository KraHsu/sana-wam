# CACH-SANA-WAM Stage 0：Source、初始化、Checkpoint 与 Launcher 设计

> 状态：`DRAFT v0.1 / STAGE0_ONLY / EXECUTION_DENIED`
>
> 审计主机：`H200`
>
> 审计工作树：`/home/zch/workspace/sana-wam`
>
> 审计日期：`2026-07-31`（Asia/Shanghai）

本文落实
`docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md`
的 Stage 0 source/initialization/launcher 部分。它只定义实现前契约，不表示：

- CACH 代码已经实现；
- unit/numeric/causal admission 已通过；
- `REF-GDN-CORRECTED` 已 commissioning；
- `CACH-A` 已获训练或评测授权；
- 可以创建正式 `/DATA` run root。

本轮完成 H200 只读审计、静态设计，以及 Stage 0 denial/guard/verifier/launcher/
测试代码 diff；没有运行测试、模型 forward/backward、训练、评测或 capture。

## 1. 规范词与判定顺序

本文中的“必须”“禁止”“失败”都是 launcher/verifier 的 fail-closed 条件。
聊天消息、旧 run 的成功状态、Phase-6 ticket 或路径存在均不能覆盖这些条件。

判定顺序固定为：

1. 外部 literal trust anchor；
2. H200 governance；
3. Git/source/runtime/input closure；
4. reviewed design 与唯一 candidate spec；
5. initialization/checkpoint/trainable inventory；
6. authority；
7. verifier/evaluator/launcher identity；
8. root、容量、GPU/process admission；
9. 才允许任何会构造模型或改变 run root 的动作。

任一较早步骤失败，后续步骤不得执行。特别是，`decision=deny_execution`、文件名含
`.draft`、blocker 非空或 root 已存在，都必须在 import `torch`、GPU reservation
和正式 root mutation 前拒绝。

## 2. H200 审计身份

### 2.1 Git 与工作树

2026-07-31 只读复核得到：

| 对象 | 固定值 |
|---|---|
| 主仓 HEAD | `605f1c134b4c983ff80f8489c4bc8847036329e2` |
| 主仓 HEAD tree | `2a7a5694c6864bbee96b36d3a9c370d94a0a0129` |
| HEAD 中 `third_party/Sana` gitlink | `16b9cec673e3335724ba2d8db25de7f9ed229292` |
| H200 Sana worktree HEAD | `16b9cec673e3335724ba2d8db25de7f9ed229292` |
| Sana describe | `v2.0.0-14-g16b9cec` |
| 附加 research tree `/home/zch/workspace/sana-afcc-handoff` HEAD | `9586486f2a9f5172d57b325e32093a3e018d34c0` |
| 附加 handoff SHA256 | `76f251fde90cde1a2b970f6360d95f4dc51074f41579baebe130a283a4ab7920` |

H200 当时已有且不属于本文修改范围的用户/历史工作：

```text
 M tests/test_phase6_candidate_eligibility.py
?? %ln
?? analysis/
?? telemetry/
```

此前文档交付还新增了：

```text
?? deep-research-report.md
?? docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md
```

在 Stage 0 修改开始前，对 `src/`、`scripts/`、`configs/` 和
`third_party/Sana` 执行的定向 `git diff --name-only` 为空，因此本节下方的
pre-delivery 审计 identity 与固定 HEAD/gitlink 一致。本次 Stage 0 delivery
随后显式新增/修改 denial guard、静态 contract/verifier/launcher、draft config
和测试代码；它们必须由最终 `SOURCE_MANIFEST` 与 denial authority 的 exact SHA
闭合，不能被描述为 HEAD 原始 bytes。上述无关 dirty/untracked 路径不得被清理、
覆盖、纳入 source bundle，亦不得成为 Python import、config resolution 或运行
证据的来源。

### 2.2 Governance 缺口

H200 Stage 0 起始审计的只读结果为：

| 路径 | 状态 |
|---|---|
| `START_HERE_20260730.md` | **缺失** |
| `AGENTS.md` | **缺失** |
| `deep-research-report.md` | 存在，SHA256 `4bd78d3c37970ed4c3fc9c05a82faef44fbeac079a6541a694fa496e9d858010` |
| `docs/agent_handoff/AGENT_HANDOFF_20260730.md` | 存在，SHA256 `76f251fde90cde1a2b970f6360d95f4dc51074f41579baebe130a283a4ab7920` |

Stage 0 delivery 之后，两份 repository-side draft source 已用 non-overwrite
方式部署并只读复核：

| 根镜像 | 类型/模式 | SHA256 |
|---|---|---|
| `AGENTS.md` | regular / `0664` | `646ea39906ea3bc75af3b360187eb2c5aebac4c04c62ec5df108c0ec35e006ae` |
| `START_HERE_20260730.md` | regular / `0664` | `e85fa5be9f0d1941725cd356298493b9003ed2339d1a5cc00d5688ed8e51c061` |

它们分别与 `governance/AGENTS.md` 和
`governance/START_HERE_20260730.md` byte-identical。这个动作只关闭了根镜像
presence/bytes 子项，不把尚未进入 reviewed commit/bundle 的 draft source
升级为可恢复 lineage，也不产生 executable authority。完整闭合动作仍必须满足：

1. 两份 governance source 已 review，最终 SHA 写入 `SOURCE_MANIFEST`；
2. H200 根部署镜像是 regular、non-symlink 文件；
3. 根镜像与对应 repository-side source byte-identical；
4. launcher 从外部 authority pin 取得期望 SHA，再稳定读取并复算；
5. 缺失、symlink、非 regular、byte/SHA 不同均拒绝；
6. 不通过修改或清理上述用户 dirty paths 来“制造 clean 状态”。

当前根镜像核验已通过；`SRC-001` 和未注册 executable governance 仍保持 P0，
所有现有 authority 必须 `decision=deny_execution`。

### 2.3 文档 source

| 文档 | SHA256 |
|---|---|
| 主开发计划 | `969d9668ffed07176844fcbd84062825f9d5f0af0325959b0068c38fe481f0dd` |
| `deep-research-report.md` | `4bd78d3c37970ed4c3fc9c05a82faef44fbeac079a6541a694fa496e9d858010` |
| 仓内 handoff | `76f251fde90cde1a2b970f6360d95f4dc51074f41579baebe130a283a4ab7920` |

这些 SHA 固定本文起草依据。后续文档修改必须产生新的 source manifest/revision；
不得继续沿用本表而只在聊天中解释变化。

### 2.4 实现 source SHA

最终 Stage 0 delivery 的逐文件路径与 raw SHA 只以
`SOURCE_MANIFEST.draft.json.repository_files` 为准；denial authority 另外 pin
全部 design/guard/verifier/launcher/test bytes。本节不复制一份会在最后编辑时
变陈旧的手写 hash 表。manifest 或 authority 中任一路径缺失、hash 为 null、或与
最终 worktree bytes 不同，denial graph 自身也必须 fail-closed。

以下 SHA256 在 H200 上读取，并与本地相同路径复核一致。本文的源码行号均指向
这些 exact bytes。

| 文件 | SHA256 |
|---|---|
| `src/sana_wam/model/gdn_ar.py` | `15a75585a0a4e33b7b83c65dd5d40333c1d5ad05c1862b21412ce302dbadd08c` |
| `src/sana_wam/model/video_backbone/sana/pipeline_builder.py` | `905cdf95239206b2fef8ff484d8cbe4de551735af56cf970f0cad8540b35a981` |
| `src/sana_wam/model/video_backbone/sana/adapter.py` | `6ef2b6c21119b1357c8cab938e0351df428a71992ea0e92f25460b1b0a05d1cf` |
| `src/sana_wam/model/base.py` | `9754d27e7916c941293174afece950606d01cf5f97c2826d6310db1c5a398ebe` |
| `src/sana_wam/train/trainer.py` | `715236acec1269f1f672aafacc99bf37afcf4ca6eadca5a6ca941e85821f3496` |
| `src/sana_wam/deploy/model_loader.py` | `a886790cfa87fb5dfc8a9634cc64ca20684ae6f4a16b6801084f78cca14f4a95` |
| `configs/train_sana_wm_gdn_ar.yaml` | `9311107bb5f3593853bd13fc40116312b4dbf17f5540ba758b733a474865f406` |
| `configs/train_gdn_ar.yaml` | `9804f0368e5e3d6dcdb55601c37dc2ae9f80c4454c1bf5dea380329d6231efe5` |
| `configs/deploy_gdn_ar.yaml` | `fe7341ce5d7d35c6a9b676af78c59cb4d096de2bb92ce3e393fec44396f62aec` |
| `third_party/Sana/diffusion/model/nets/sana_gdn_blocks.py` | `ef15578c63f9815d670dc3ef4cf2876c4f236754bfdfb0d2070da387048a495e` |
| `third_party/Sana/diffusion/model/ops/fused_gdn.py` | `863cbbb601ddebc5666179e2b209b777d5dc213c94a1450cde695a33b0a8aded` |
| `third_party/Sana/diffusion/model/nets/sana_multi_scale_video_camctrl.py` | `bfbd72dfb22a44e2843f1985049ecfbb1f887f5e5f5ed852905c4670e2029fbe` |

Phase-6 launcher/source 的审计 identity 为：

| 文件 | SHA256 |
|---|---|
| `scripts/run_phase6_real_2b_smoke.py` | `fa9865509b71d28188c960ff9f5053d95f4ce808cbff4767521619205c8c826b` |
| `scripts/prepare_phase6_launch.py` | `a1eacc1a81e3912dc3ab7393888e944f766ca1c1d053ff5783f17fcc77553330` |
| `scripts/generate_phase6_launch_manifest.py` | `4526955ec9bb00e2c1534ce5fd18d4d839da72d5488c08afbe71bd80f6bc950f` |
| `scripts/validate_phase6_preflight.py` | `ff5597ee2662772fb65514152a803151bbcf5a0835894a558bf88e089e124e1b` |
| `src/sana_wam/train/phase6_arm_config.py` | `87b18b9f2c0867568c12c706284e5a665e639659991ad43615b0537266daa468` |
| `src/sana_wam/train/phase6_launch_manifest.py` | `2b2c9b594cfdcbfac33aa13471bda5ef30f0626a498cb76539f858f7b8efbde7` |
| `src/sana_wam/train/phase6_preflight.py` | `c38bd5da461ca60d5a55c1e394a68e0960ae49efc06eca49fb42d0947d5a044d` |
| `src/sana_wam/train/phase6_smoke_runtime.py` | `a292c1b1a251501f0ed73bd7297c470ee6cffb8785096de4fefd3002be683de0` |
| `src/sana_wam/train/phase6_recovery.py` | `dca996028c89edee25c151413d70c8634c1f258484f4a8b01bac864ab495cf23` |

## 3. 当前实现为何不能直接满足本计划

Stage 0 已在通用 train/deploy、checkpoint `config.yaml` loader、direct policy
server、VAE-cache precompute、AR GPU smoke、RoboTwin stats 和 repository-side
eval shell/wrapper 入口加入 raw-base `cach_stage0` reserved-marker denial；检查
发生在 CLI dotlist merge、torch/model/dataset import、GPU 选择和
root/lock/cache/temp/output 创建之前。这个 denial 只防止 draft config 被旧入口
误当成 legacy config，不能替代 Stage 1 的 `complete_random_v1` builder、exact
checkpoint schema 或 executable launcher。

### 3.1 Pretrained DiT seam 是显式存在的

`pipeline_builder.py:302-310` 定义 `init_dit_from`，并把它描述为 SANA-WM
streaming GDN checkpoint seam。`pipeline_builder.py:706-710` 还会因为
`init_dit_from != null` 改选 pretrained streaming factory/preset。

真正的加载发生在 `pipeline_builder.py:779-818`：

- `torch.load(..., weights_only=False)`；
- 从 `generator`/`state_dict` 和 `model.` prefix 提取权重；
- 丢弃 shape 不匹配的 key；
- `load_state_dict(..., strict=False)`；
- missing/unexpected 只 warning。

现有 pretrained 配置确实启用该路径：

- `configs/train_sana_wm_gdn_ar.yaml:52-65` 指向
  `/DATA/share/SANA-WM_streaming/sana_dit/model.pt`；
- 同一配置 `:137-152` 冻结 DiT，只训练 action 路径；
- `pipeline_builder.py:515-569` 表明该 preset 是 20 层中 15 GDN + 5 softmax、
  全 CamCtrl，而不是 v0 的 corrected all-GDN reference。

因此“把该字段留空即可”的软约定不够。CACH 必须在独立 static verifier 和
builder seam 双重拒绝 pretrained video-DiT。

### 3.2 当前 checkpoint handling 仍有 permissive 分支

当前训练初始 checkpoint 入口位于 `trainer.py:955-987`。虽然默认调用
`strict=True`，但 `trainer.py:989-1046` 允许 A-off checkpoint 缺少完整 adapter
prefix，并保留 fresh identity init。

更底层的 `base.py:453-507`：

- 总是先执行 `load_state_dict(..., strict=False)`；
- `allow_missing_patterns` 是 substring match；
- matched missing keys 保留默认初始化。

部署进一步在 `deploy/model_loader.py:34-44` 自动选择数值最大的
`checkpoint_step_*`，并在 `:71-76` 允许所有包含 `norm_kv` 的 missing key。
该 loader 的文件头 `:9-11` 还明确表示 architecture checkpoint 会覆盖已加载的
frozen VAE/text state。

这些兼容策略服务旧 checkpoint，不满足 CACH 的 same-revision exact-schema
resume/deploy 契约。

### 3.3 当前 freeze 语义会切断 complete-hybrid 梯度

`base.py:28-65,556-593` 表明 `freeze_modules` 不仅
`requires_grad_(False)`，还递归把整个 subtree 的每个 forward 包进
`torch.no_grad()`。因此把 `video_backbone.dit` 放入 legacy freeze list 会让其
所有后代失去梯度，不能用“稍后再打开某些 parameter”的方式恢复。

此外，legacy freeze 对 unknown dotted path 会静默跳过
（`base.py:572-576`），也不适合 fail-closed inventory。

当前 trainer 已有可利用但仍需 CACH 收紧的基础：

- model build/load 之后才配置 trainable set（`trainer.py:142-159`）；
- parameter-pattern 路径会逐 parameter 设置 `requires_grad`
  （`trainer.py:1076-1142`）；
- optimizer 分组只消费 `requires_grad=True` 参数
  （`trainer.py:1272-1306`）。

CACH 不得使用 legacy `training.freeze` 作为唯一证明；必须对最终 resolved model
生成并核验逐 tensor inventory。

### 3.4 VAE/text 当前是 frozen，但缺失会 fail-soft

已有冻结事实：

- LTX2 VAE 在 `pipeline_builder.py:990-1008` 以 FP32 load，随后
  `.eval().requires_grad_(False)`；
- text encoder 在 `pipeline_builder.py:1134-1138` 执行
  `.eval().requires_grad_(False)`。

但 `pipeline_builder.py:990-998` 在 LTX2 path 不存在时只 warning 并返回
`None`；text load 的多个失败分支也会 warning 后返回 `None`
（例如 `:1100-1108`）。CACH 的 frozen external encoder 是模型定义的一部分，
所以 source 缺失、realpath/inventory 不同或返回 `None` 必须是 hard failure。

### 3.5 当前 GDN-AR 不是 CACH-A

`gdn_ar.py:142-170` 先完成 video `run_chunk`，然后 action backbone 才读取 video
bridge；`noisy_actions` 没有进入当次 video transition。它不能证明 action
causally conditions video。

同一文件还有 Stage 0 已知不闭合项：

- `:54-59` 保存旧 `ar_observed_prefix_chunks` 语义；
- `:230-248` 用 `T//K` 与 `Ta//total_chunks` 建 fixed-ATC；
- `:317-325` 实际又从 chunk 0 开始监督，和旧 prefix 注释漂移；
- `:437-453` 把 ground-truth clean chunk teacher-forced 写入后续 cache。

因此旧 `gdn_autoregressive` config/engine 不能改名成为
`REF-GDN-CORRECTED`，旧 action-video adapter 也不能改名成为 CACH-A delta。

## 4. Phase-6 launcher 为什么禁止复用

结论是：可以借鉴安全原则，但不得直接 import、改参数或重签 Phase-6
launcher/authority/ticket 来启动 CACH。

### 4.1 它授权的是另一套科学问题

`phase6_arm_config.py:37-46` 把实验固定为五臂
`T0_E0A0/T1_E0A0/T1_E1A0/T1_E0A1/T1_E1A1`。其模型 projection
（`:407-476`）固定：

- `dual_system/autoregressive`；
- `frame_chunk_size=2`；
- `first_frame_causal`；
- pretrained `linear_relu` SANA-Video；
- `window_flash` expansion；
- Phase-6 action-memory adapter。

数据 projection（`:479-522`）固定 Wan VAE、`temporal_compression=4` 和旧任务
集合。训练 projection（`:525-594`）固定：

- 504 steps；
- `lambda_action=0`；
- 只训练选中的 video pattern，A1 时再加旧 adapter；
- adapter arm 绑定 Phase-1 action reference/checkpoint。

Phase-6 smoke 也要求加载一个既有完整 student checkpoint
（`run_phase6_real_2b_smoke.py:509-531`），并明确要求 action/proprio frozen
（`:620-640`）。这些条件与 CACH 的 LTX2、from-step-0、action/video complete
joint training 相冲突。

### 4.2 它硬绑定旧 authority 与绝对路径

`phase6_launch_manifest.py:40-100` 固定：

- Phase-6 schema/launch ID；
- `/home/zch/workspace/sana-wam`；
- 旧 `/DATA/share/sana_phase6_principled_constraints_20260724/...` source repair；
- protocol/registry/recovery/action-reference/old checkpoint roles。

`phase6_launch_manifest.py:1016-1048` 的 operation 是 `phase6_training`，环境还必须
`SANA_PHASE6_FORMAL=1`。`phase6_launch_manifest.py:1084-1177` 要求五份 raw YAML、
五个 arm、旧 run directory 和 Phase-6 ticket。把 CACH config 塞进该 schema
会使 manifest 说的 operation、source、reference 与实际执行不同。

### 4.3 “复制 ticket 后换 argv”也不成立

Phase-6 自身会在 `phase6_launch_manifest.py:1795-1812` 拒绝 config SHA、
argv、managed environment、working directory 或 output directory 变化。这是
正确的旧 authority 防篡改行为，不能被绕过后再声称得到 CACH admission。

### 4.4 可借鉴与不可继承的边界

可在新 CACH namespace 中重新实现并重新 review：

- stable regular-file read；
- `O_NOFOLLOW`/`O_EXCL`；
- canonical JSON；
- byte SHA pin；
- exact argv/env；
- worker/process identity；
- receipt `fsync`。

禁止：

- 从 `phase6_*` import authority、arm config、recovery、ticket 或 manifest；
- 引用 Phase-6 root/receipt 作为 CACH gate；
- 沿用五臂 schema、504-row action reference、student checkpoint；
- 让兼容 Phase-6 的默认值进入 CACH resolved config；
- 仅换文件名或 `operation` 字符串后复用旧 artifact DAG。

若为了减少重复代码而抽取纯 utility，必须复制到无 Phase-6 常量/side effect 的
新模块，给出新的 source SHA、单独测试与 independent review。抽取前默认不复用。

## 5. CACH 独立 source 与 trust closure

### 5.1 Source bundle

正式 CACH operation 必须消费一个 immutable `SOURCE_MANIFEST.json`。最低字段：

```text
schema_version
repository_head
repository_tree
sana_gitlink
sana_worktree_head
governance_files[]
design_files[]
code_files[]
config_files[]
runtime_files[]
external_inputs[]
excluded_dirty_paths[]
python_import_roots[]
source_bundle_path
source_bundle_sha256
```

每个 file entry 至少包含：

```text
logical_role
absolute_realpath
relative_path
size_bytes
sha256
file_type=regular
symlink=false
```

Git commit 不能代替文件 SHA；untracked CACH source 也不能仅靠工作树存在。进入
任何 executable authority 前，必须满足二者之一：

1. CACH source 已成为固定 commit/bundle 中的可重建对象；或
2. 建立完整 immutable source bundle，manifest 枚举所有 imported/config bytes。

`PYTHONPATH`、current working directory、editable install 与实际 import 的模块
realpath 必须落入 manifest allowlist。若 Python 从 `analysis/`、`telemetry/`、
`/tmp`、用户 site-package 或未登记 checkout import 同名模块，立即失败。

### 5.2 Runtime 与外部输入

source closure 之外，authority 必须独立 pin：

- Python executable realpath/hash；
- Python/package/torch/CUDA/Triton/NCCL versions；
- lockfile 或完整 installed-distribution inventory；
- GPU model/UUID/driver；
- LTX2 VAE 完整 regular-file path/hash/size inventory；
- Gemma text encoder 完整 regular-file path/hash/size inventory；
- tokenizer/config bytes；
- dataset manifest、row manifest、action stats；
- VAE temporal/spatial compression 与 causal-first-frame metadata。

历史路径
`/DATA/share/SANA-WM_streaming/ltx2_causal_vae` 和
`/DATA/share/gemma-2-2b-it` 不是 identity。路径重映射、mtime 或“目录还在”都
不能代替逐文件 SHA/size inventory。

## 6. Complete random initialization

### 6.1 初训配置硬条件

`operation_type=fresh_train` 时，raw config 与 fully resolved config 都必须显式：

```yaml
model:
  architecture:
    initialization_mode: complete_random_v1
  video_backbone:
    model_path: null
    init_dit_from: null
    vae_type: ltx2
    vae_path: <source-manifest-pinned path>
    text_encoder_name: <source-manifest-pinned path>
    freeze: false
training:
  init_checkpoint: null
  init_checkpoint_sha256: null
  freeze:
    - video_backbone.vae
    - video_backbone.text_encoder
```

这两个 freeze path 是 `SanaVideoBackbone.add_module("vae", ...)` 与
`add_module("text_encoder", ...)` 后的 exact registered path，用于保持
`requires_grad=false` 且在 `architecture.train()` 后恢复 eval；不得写
`video_backbone._pipe.vae` 或裸 `text_encoder`。该列表只是运行态 freeze/eval
机制，不能替代 build 后逐 tensor 的 frozen/trainable inventory 二次证明。

以下任一情况拒绝：

- `model_path`/`init_dit_from` key 缺失而依赖默认；
- 二者为非 null、空白变体、环境展开值或 CLI override；
- `training.init_checkpoint`、`training.init_checkpoint_sha256` 或任意 warm-start
  key 缺失或为非 null；
- config 含 Phase-6 launch/reference/AFCC authority 字段；
- builder 走到 `torch.load` pretrained DiT seam；
- build 后 metadata 不能证明 `pretrained_dit_loaded=false`；
- 使用“先 load 再随机重置”的方式冒充 from-scratch。

在代码层必须给 CACH builder 一个不可绕过的 `initialization_mode` enum。v0
fresh operation 唯一允许值为 `complete_random_v1`；legacy builder 的
`init_dit_from` 分支即使 config verifier 已检查，也要在 CACH construction seam
再次拒绝。

### 6.2 “Complete”的确切范围

`REF-GDN-CORRECTED` fresh build 中，下列已实例化参数全部从注册 seed stream
初始化，并从 optimization step 0 共同训练：

- video patch/input embedding、timestep/text conditioning projection；
- 20 层 video DiT 的 GDN self-attention、cross-attention、FFN/temporal conv、
  norm 与 output/final layer；
- action backbone、action head；
- proprio encoder/projection；
- action/video bridge 中属于共同 architecture 的 trainable 参数。

`CACH-A` 在上述集合之外只增加 action-to-video conditioner。其初始化固定为：

- 内部非 gating projection 使用注册的 operator-specific seed；
- design 指定的 output gate/projection 与 no-action/bypass 参数是 exact no-op；
- step 0 相同输入下 candidate video 输出与 reference bitwise/注册容差一致；
- conditioner 全部 trainable，不因 no-op 初始化而冻结。

REF 与 CACH-A 的同名、同形 shared tensors 必须使用相同 seed stream，并保存逐
tensor init digest。不得通过加载已训练 REF checkpoint 构造 CACH-A。两端都是
fresh build；operator-specific tensor 不伪造为 shared tensor。

### 6.3 唯一 frozen 外部模块

v0 只允许：

- source-pinned LTX2 causal VAE；
- source-pinned text encoder/tokenizer

作为 frozen external encoder。对有参数的两者必须同时证明：

```text
module exists
source inventory matches
training == false
all parameters requires_grad == false
all forward paths no_grad/inference-only
optimizer membership count == 0
parameter/state digest unchanged before/after authorized operation
```

tokenizer无参数，但其文件 inventory 与 config 仍必须 pin。scheduler、action
normalization statistics等非参数 state 另列 buffer inventory，不能被错误称为
trainable，也不能在 reference/candidate 间漂移。

任何其他 module 被 frozen 都是 failure。尤其禁止：

- `video_backbone.dit`；
- GDN/softmax/anchor/AttnRes（在对应 variant 存在时）；
- action backbone/head；
- proprio encoder；
- action conditioner。

## 7. Trainable/frozen inventory

### 7.1 Canonical inventory

模型 build 完成、optimizer 构造之前必须 exclusive publish
`MODEL_INVENTORY.json`。每个 parameter/buffer entry 至少含：

```text
fully_qualified_name
kind=parameter|buffer
architecture_role
shared_or_operator_specific
shape
dtype
numel
requires_grad
optimizer_group_or_null
init_tensor_sha256
source_external_or_random
```

顶层还要包含：

```text
candidate_revision
resolved_config_sha256
source_manifest_sha256
expected_key_schema_sha256
trainable_name_set_sha256
frozen_name_set_sha256
trainable_numel
frozen_numel
buffer_name_set_sha256
```

canonical key 顺序按 UTF-8 parameter name 排序。tensor digest 必须覆盖
name、shape、dtype 和 contiguous raw bytes，不能只 hash 数值 bytes。

### 7.2 Verifier 联取条件

verifier 必须证明：

1. expected parameter set 与 observed set 完全相等；
2. 每个 required-trainable parameter `requires_grad=True`；
3. frozen set 仅属于 VAE/text allowlist；
4. 每个 trainable parameter 恰好进入一个 optimizer group；
5. 每个 frozen parameter进入零个 optimizer group；
6. optimizer 中没有 inventory 外 parameter；
7. reference/candidate shared name/shape/dtype/init digest 一致；
8. operator-specific set 与 `CANDIDATE_SPEC.unique_delta` 完全相等；
9. 不存在 legacy `freeze_modules` 在 trainable ancestor 上留下的
   `_sana_wam_no_grad_wrapped`；
10. AFCC weight/reference/Phase-6 adapter parameter 为零个；
11. variant 未启用的 anchor/AttnRes/self-forcing parameter 为零个；
12. VAE/text state 与 external source digest 一致。

任何 glob 只允许在 design-time 生成 expected list；formal verifier 比较的是已
冻结的 exact name list，不能在运行时用一个更宽 glob 接受新参数。

## 8. Resume 与 deploy 的 exact key allowlist

### 8.1 Checkpoint schema

CACH 使用独立 `CACH_CHECKPOINT_SCHEMA.json`。每个允许 key 固定：

```text
name
role
shape
dtype
numel
required=true
```

并保存 sorted exact key set SHA。load 判定必须是集合与逐项 metadata 的联取：

```text
observed_keys == expected_keys
observed_shape[key] == expected_shape[key]
observed_dtype[key] == expected_dtype[key]
```

禁止：

- substring/wildcard missing-key exemption；
- unexpected key；
- “missing key 保留 default init”；
- 自动寻找 latest checkpoint；
- 从目录名/step number推断 endpoint；
- A-off → A-on warm start；
- 用 reference checkpoint 初始化 candidate；
- 加载后仅打印 warning 继续。

### 8.2 v0 state partition

v0 checkpoint policy 固定为：

```text
trained model parameters
+ required model buffers
- external frozen VAE/text parameters and buffers
```

VAE/text 从 `SOURCE_MANIFEST` 指定的 external source 单独构造和验证，训练
checkpoint 不得覆盖它们。因为当前 `BaseWAMArchitecture.state_dict()`/loader
默认包含并覆盖这类 state，Stage 1 必须实现 CACH-specific checkpoint
writer/loader；不能直接调用当前 permissive compatibility loader。

required buffers（例如 action normalization、position/config dependent buffer）
必须逐名列入 schema；不能用“全部 buffer”或“除了某 prefix”动态猜测。

### 8.3 Resume

`operation_type=resume` 不是 fresh initialization 的例外入口。它必须有新的独立
authority 和新 immutable root，并通过 `supersedes` 指向旧 frozen root。只允许：

- same `candidate_revision`；
- same design/source/input/resolved-config/operator schema；
- authority 指定的一个 checkpoint path 与 raw SHA；
- exact model key schema；
- exact optimizer parameter-ID/schema/master-weight state；
- exact scheduler/global-step/RNG/data-cursor/accumulation state；
- old root 已有可验证 checkpoint receipt 与 final/failure inventory。

任一完整训练状态未被保存或无法复现时，`resume_supported=false`，只能从 step 0
创建新 campaign revision；不得把 architecture-only checkpoint 称为 exact resume。

### 8.4 Deploy

deploy authority 只能指向一个已冻结、已通过相应 gate 的 endpoint：

```text
campaign_id
candidate_revision
endpoint_step
checkpoint_absolute_path
checkpoint_raw_sha256
checkpoint_schema_sha256
resolved_config_sha256
source_manifest_sha256
engine_schema_sha256
```

deploy loader 必须按 exact filename/hash 读取，禁止
`checkpoint_step_*` glob/latest 选择。build 后再次核验 model inventory、external
VAE/text inventory 和 engine/operator/layout schema；missing/unexpected/shape/dtype
任一不同即失败。

## 9. 独立 design、authority、verifier、evaluator 与 launcher

### 9.1 五件套

CACH 每个 operation 必须形成独立五件套：

| 角色 | 唯一职责 | 禁止职责 |
|---|---|---|
| `DESIGN` | 定义 architecture/operator/init/key/root/gate 契约 | 不能授权执行 |
| `AUTHORITY` | 对一个 operation/candidate/root/预算作 allow/deny 决策 | 不能计算自己的期望结果 |
| `VERIFIER` | 只读验证 source、config、inventory、state、receipt | 不能修改 run 或放宽 authority |
| `EVALUATOR` | 按预注册输入/metric 读取唯一 endpoint | 不能选 checkpoint/task/seed |
| `LAUNCHER` | 按 exact argv/env/root 启动并管理 process group | 不能生成/修改 design 或 authority |

五者都必须有 path、raw SHA256、schema version。`AUTHORITY` 必须 pin 其他四者和
`CANDIDATE_SPEC`；launcher 还必须从命令行接收 authority 的 external literal
SHA。只把期望 SHA 写在同一份可变 authority 文件旁边不构成 trust anchor。

### 9.2 Authority 最小字段

```text
schema_version
decision=allow_execution|deny_execution
operation_type
campaign_id
candidate_revision
reference_revision_or_null
unique_delta
design/source/runtime/input/spec digests
verifier/evaluator/launcher digests
exact argv/environment
exact root path and nonce
seed manifest
checkpoint policy
trainable inventory schema
endpoint
gate thresholds
projected bytes/inodes and safety buffers
GPU UUID/count/budget
blockers[]
issued_at
review receipts[]
```

规则：

- `blockers != []` 时 decision 必须 deny；
- draft artifact 永远不能被 allow authority 引用；
- commissioning authority 不能被 paired campaign 重用；
- `REF-GDN-CORRECTED` 与 `CACH-A` 使用不同 campaign ID/root/endpoint；
- CACH authority 中出现 Phase-6/AFCC formal reference、weight 或 root 即失败；
- campaign 间不得自动生成下一张 allow authority。

### 9.3 Verifier independence

verifier 的期望值只能来自 authority pin 和它自身固定源码，不得从被测 run
“学习”allowlist、阈值或 endpoint。它必须：

- 以 read-only descriptor + `O_NOFOLLOW` stable-read 输入；
- 比较 canonical bytes 与 exact keys/types；
- 重算所有文件/tensor/inventory/receipt hash；
- 核验 process start ticks、boot ID、GPU UUID；
- 核验 state transition receipt chain；
- 把 PASS/FAIL 写入新的 exclusive verifier receipt；
- 不 import training model 来推断“应该有哪些 key”。

model expected key list在 design/build review 阶段生成并冻结；formal verifier
只比较，不动态扩张。

Stage 0 denial verifier/launcher 的 bootstrap 采用更窄的顺序：先由命令行 literal
SHA stable-read 并认证 authority bytes，再从该已认证 authority 取得脚本自身和
`cach_stage0_contract.py` 的 exact pin；脚本自检且 contract bytes 匹配后，才以
已读取的同一组 bytes 编译 contract。bootstrap 不 import `sana_wam` package，
避免在 pin 校验前执行仓内 `__init__.py`。这只使 denial graph 可复核，不把 draft
升级为 executable authority；未来 executable verifier 仍需独立 schema/review。
当前 `O_NOFOLLOW` 只拒绝输入路径的末级 symlink，中间目录组件没有用 dirfd
逐级锁定；默认 canonical repository 路径之外的输入不能据此声称完整 path
resolution closure。
当前 runtime check 也只验证 Python executable bytes/version 和六个已列出的
distribution version；它不验证 driver/CUDA runtime bytes、完整 distribution
inventory 或 `package_inventory_sha256`。报告必须显式
`verified_runtime_distribution_subset=true` 且
`runtime_inventory_complete=false`，不得简称 full runtime closure。denial
verifier 即便退出 `0` 也只表示 draft denial pin graph 一致，不能作为 execution
admission。

因此 Stage 0 denial verifier 本身不 import OmegaConf 或其他第三方 YAML parser。
draft config 的最终 raw SHA 同时写入已认证 contract 常量和 authority config
pin；verifier stable-read 一次 config bytes 并比较这两个值。contract 中仍包含
exact typed mapping validator，供后续单独获准的测试执行，但本轮报告固定
`config_semantic_parser_executed=false`。raw binding 不得被夸大为已运行 semantic
config test。

外部模型目录采用 regular-file path scan-before、逐文件 `O_NOFOLLOW`
stable-read、regular-file path scan-after；它可以拒绝扫描期间常见的文件成员
增删，但不记录空目录，也不是原子 filesystem snapshot，不能证明并发修改后恢复
原状从未发生。dataset count scan 同样不是内容快照。正式 executable admission
仍须使用不可变 snapshot/read-only mount 或等价的、经独立复核的目录级冻结机制，
不能把本 Stage 0 扫描称为 atomic directory closure。

## 10. Immutable root、state 与 failure 契约

### 10.1 Namespace

计划 parent：

```text
/DATA/share/sana_cach_wam_20260731/
```

该路径当前只是 design，不授权创建。建议结构：

```text
<campaign_id>/
  PAIR_REQUEST.json
  CANDIDATE_SPEC.json
  SOURCE_MANIFEST.json
  DESIGN.json
  AUTHORITY.json
  STATE/
  REFERENCE/
  CANDIDATE/
  RECEIPTS/
  FINAL_INVENTORY.json
  COMPLETION.json | FAILURE.json
```

`REF-GDN-CORRECTED` commissioning 是 single-arm root，不能伪装成 CACH-A pair；
其 root 以 `COMMISSIONING/` 取代 `REFERENCE/CANDIDATE`。CACH-A 才是 fresh paired
root。

### 10.2 创建

launcher 必须：

1. pin parent realpath/device，拒绝 symlink parent；
2. authority 给出 literal absolute child path，不允许 glob、`..`、环境展开；
3. read-only 检查 target absent；若已存在，在 GPU reservation 前拒绝；
4. 安装 process-group/failure trap；
5. 取得 capacity 与 GPU reservation 后，以 direct exclusive `mkdir` 创建
   `0700` child；
6. 创建后复核 inode/device/owner/mode，并发布 `RESERVED` receipt；
7. 任一竞态导致 `EEXIST` 时，不触碰该 target，按 pre-mutation denial 处理。

run 内所有文件用 `O_CREAT|O_EXCL|O_NOFOLLOW` 或等价 no-overwrite publish。
regular artifact 必须 link count 1；禁止 symlink、hardlink、FIFO、socket/device
作为证据。`/tmp` 只可作非唯一 staging。

### 10.3 State machine

状态链固定为：

```text
RESERVED
  -> STATIC_PASS
  -> UNIT_PASS
  -> NUMERIC_PASS
  -> CAUSAL_PASS
  -> SHORT_PASS
       -> CAMPAIGN_COMPLETE
       -> CLOSED_LOOP_PASS    # 仅获授权且适用的最终 campaign
            -> CAMPAIGN_COMPLETE

ANY_NONTERMINAL -> FAILED
```

每个 transition 是新的 immutable receipt，必须含：

```text
from_state
to_state
previous_receipt_sha256
authority_sha256
source/config/input/inventory sha256
root inode/device
boot_id
launcher pid/start_ticks
worker pid/start_ticks/process_group
GPU UUID
timestamp
```

禁止 mutable `STATE` pointer；当前状态从 receipt chain 推导。状态跳跃、重复、
前序 SHA 不同或存在两个分叉 receipt 都失败。

### 10.4 Failure

failure trap 必须在首次 root mutation 与 worker spawn 前生效，并覆盖正常异常、
signal、launcher crash 的可恢复清理路径。失败时：

1. 向本 launcher 创建的 process group 发信号；
2. 通过 PID + start ticks 验证全部 worker 退出；
3. 不杀、停止或修改任何不属于该 process group 的进程；
4. 保存 exit/signal、最后合法 state、stdout/stderr hash、GPU/磁盘快照；
5. exclusive 写 `FAILURE.json` 和 final inventory；
6. `fsync` 文件与包含目录；
7. verifier stable-read 重算；
8. 冻结 root。

若 admission 在正式 root mutation 前失败，不创建/占用目标 run root；将 denial
写到 authority 指定的小型 append-only `ADMISSION_DENIALS` namespace。若该
namespace 自身未闭合，则只向调用者返回失败，绝不能退回一个可执行状态。

### 10.5 Freeze

success/failure 共用以下最终顺序：

1. worker 已终止且输出关闭；
2. exclusive 写 canonical `COMPLETION.json` 或 `FAILURE.json`；
3. 写 `FINAL_INVENTORY.json`；inventory 明确排除自身 bytes，但包含 terminal
   receipt SHA 和其他全部 artifact byte SHA；
4. 对所有文件与目录 `fsync`；
5. verifier 复读并确认 inventory/receipt；
6. regular files 改为 `0400`；
7. directories 自底向上改为 `0500`；
8. 再复核 mode、byte SHA、root inode/device；
9. 发布外置 freeze receipt。

冻结后不得覆盖、补文件、删除或复用。重跑使用新 nonce/root，并以
`supersedes` 指向旧 root；`supersedes` 不授予修改旧 root 的权限。

## 11. Capacity 与资源 admission

### 11.1 字节与 inode 公式

launcher 用目标 filesystem 的 `statvfs`/等价值，使用普通用户实际可用的
`f_bavail`，不能使用包含保留块的 `f_bfree`。authority 必须预先固定：

```text
projected_write_bytes
safety_buffer_bytes
projected_new_inode_count
inode_safety_floor
projection_method_sha256
```

字节硬条件使用整数计算：

```text
available_bytes - projected_write_bytes
  >= ceil(total_bytes / 5) + safety_buffer_bytes
```

inode 硬条件：

```text
available_inodes - projected_new_inode_count
  >= inode_safety_floor
```

projection 必须覆盖 reference + candidate、temporary checkpoint、optimizer/
master state、logs、failure overhead、receipt/inventory；不能只估最终
checkpoint。任一 projection/buffer 缺失、为零占位、非有限或在 authority 签发后
改变均拒绝。

容量至少在以下时点重查：

1. authority admission；
2. exclusive root creation 前；
3. worker spawn 前。

任何一次失败都不得通过删除 frozen root/receipt 解决。一次只允许 admit 一个
paired root；大产物不得写根分区。

### 11.2 H200 瞬时快照

2026-07-31 只读 `df -B1/-Pi`：

| filesystem | total bytes | available bytes | used | available inodes |
|---|---:|---:|---:|---:|
| `/` | `3776651378688` | `2462947270656` | `33%` | `232946335` |
| `/DATA` | `7619773423616` | `1803616239616` | `76%` | `233153402` |

`/DATA` 的 20% reserve 是 `1523954684724` bytes（向上取整），在尚未加入任何
safety buffer 前只剩 `279661554892` bytes 可用于 projected writes。这个快照
不是 launch authority；当前尚无 reviewed projection/safety buffer，因此
`CAP-001` 保持 blocked。

## 12. Stage 0 blocker ledger

### 12.1 公共 blocker

| ID | 状态 | 阻塞内容 | 清除证据 |
|---|---|---|---|
| `GOV-001` | Stage0 mirror PASS；executable blocked | 根 mirrors 已 regular/byte-identical；source 尚未进入 reviewed commit/bundle，authority 未注册 | reviewed source lineage + executable governance review |
| `SRC-001` | P0 blocked | CACH source 仍是工作树 draft，尚无可重建 bundle | reviewed immutable source bundle/commit + manifest |
| `SRC-002` | P0 blocked | LTX2/text/runtime/dataset 完整 inventory 未全部冻结 | `SOURCE_MANIFEST` 无 unresolved entry |
| `SRC-003` | P0 blocked | 当前 `repository_files` 只闭合 Stage0 denial review 声明集，不是未来 executable 的递归 import closure | generated transitive import/build closure + dirty-path exclusion + independent review |
| `TRUST-001` | P0 blocked | denial tool 在 Python startup/脚本执行后才自检 bytes，并通过 PATH 调用未 pin 的 `git`；不能成为独立 executable trust root | external trusted runner + pinned interpreter/git bytes + descriptor-bound execution |
| `INIT-001` | P0 blocked | builder 尚未 hard-reject pretrained DiT seam | 双层 static/build rejection + reviewed code |
| `CKPT-001` | P0 blocked | CACH exact writer/loader/key schema 尚未实现 | exact schema + no permissive branch + review |
| `INV-001` | P0 blocked | variant-resolved trainable/frozen exact list 尚未生成 | canonical expected inventory + independent verifier |
| `LCH-001` | P0 blocked | Stage0 denial skeleton 已提交但 NOT_RUN；executable 五件套尚未实现/独立 review | new executable schema + exact digests + review receipts |
| `SURF-001` | P0 blocked | reserved-marker guard 只覆盖本轮列出的 canonical surfaces；legacy checkpoint probes/audits、AFCC/Phase-6 launchers 与 direct `Trainer` construction 尚有旁路 | complete executable-surface inventory + pre-import/pre-root guards + independent review |
| `DSET-001` | P0 blocked | 当前 fixed-window loader 产生 `start>0` rows；growing-history 路径产生 clean prefix，均不满足 episode-origin/no-clean-prefix | new sampler seam + row/prefix receipts + pure contract tests |
| `PROP-001` | P0 blocked | layout-derived per-chunk current-proprio selector 尚未实现；legacy clip-level state/fixed selector 可 stale 或泄漏 future state | exact boundary indices/timestamps + future-state exclusion tests |
| `CAP-001` | P0 blocked | projected bytes/inodes 与 safety buffer 未冻结 | authority-pinned capacity projection PASS |
| `DATA-001` | P0 blocked for full G4/G5 | complete random 2B 数据/scale/budget 未闭合 | reviewed `DATA_AND_SCALE_DESIGN` + pilot authority |

### 12.2 `REF-GDN-CORRECTED`

当前只能保持 `commissioning_status=blocked`：

| ID | 阻塞内容 |
|---|---|
| `REF-001` | `first_frame_pinned + observed_prefix_chunks=0` 尚未成为唯一代码 schema；旧 prefix 字段仍存在 |
| `REF-002` | `ChunkActionLayout` 尚未替换 `T//K`、`Ta//chunks` fixed-ATC |
| `REF-003` | all-GDN LTX2 random-init operator/cache/content-time schema尚未由独立 verifier 闭合 |
| `REF-004` | train/deploy denoise-read-only、paired commit、partial-tail semantics 尚未实现 |
| `REF-005` | exact complete trainable inventory 与 shared-init digest 未生成 |
| `REF-006` | commissioning design/authority/root/endpoint 尚未冻结 |

只有公共 blocker 与 `REF-001..006` 全部清除、C0–C8 对应 admission 按阶段获单独
授权后，才能称 `REF-GDN-CORRECTED`；旧 GDN-AR checkpoint/run 不得被追认。

### 12.3 `CACH-A`

当前只能保持 `candidate_status=blocked`：

| ID | 阻塞内容 |
|---|---|
| `A-001` | `REF-GDN-CORRECTED` commissioning 尚未完成 |
| `A-002` | action-to-video seam/no-action slot/pad mask/Action RoPE exact design 尚未 review |
| `A-003` | conditioner no-op init 与 bypass identity verifier 尚未实现 |
| `A-004` | deploy/G5 的 applied-action acknowledgement 与 executed committed-action history 尚未闭合；不阻塞 offline teacher-forcing G1–G4 |
| `A-005` | fresh pair 的 shared tensor init digest 与 operator-specific set 尚未冻结 |
| `A-006` | 唯一 delta 的 `CANDIDATE_SPEC`、authority、endpoint、threshold 尚是 draft |
| `A-007` | current video-first/action-read-video path 尚未替换，不能证明 action drives video |

CACH-A 不能通过加载 `REF-GDN-CORRECTED` endpoint、复用 Phase-6 A1 adapter 或把
多个 action-conditioning 变体组成 candidate pool 来解锁。它必须是 fresh
reference/candidate pair，唯一 delta 只有 reviewed action-to-video conditioner。

## 13. Stage 1 前的实现落点

本文通过 review 后，Stage 1 的最小代码范围应是新的 CACH namespace，而不是修改
Phase-6 authority：

```text
src/sana_wam/cach/source_manifest.py
src/sana_wam/cach/initialization.py
src/sana_wam/cach/model_inventory.py
src/sana_wam/cach/checkpoint_schema.py
src/sana_wam/cach/verifier.py
src/sana_wam/cach/launcher.py
src/sana_wam/deploy/cach_model_loader.py
```

需要对 legacy seam 作最小 hardening 时，可修改 builder/trainer dispatch，但必须
保持：

- Phase-6/frozen AFCC source 与 root 不变；
- CACH variant 才启用新 fail-closed schema；
- legacy compatibility loader 不被 CACH 调用；
- 不把 source/init/launcher 修改与 layout/cache/action-conditioner 大改混成一个
  无法审计的 diff。

实现前 review 至少要确认：

1. 本文与 layout/cache/data designs 不矛盾；
2. source manifest schema 能覆盖实际 import closure；
3. fresh init 无 pretrained/checkpoint 可达路径；
4. checkpoint partition 不覆盖 VAE/text；
5. exact trainable inventory 可由 mini model 与 real model分别冻结；
6. root/receipt/failure/freeze 顺序没有 overwrite window；
7. capacity projection按 pair 总量计算；
8. REF commissioning 与 CACH-A pair 的 authority 不可互换。

## 14. 当前结论

H200 的 fixed HEAD/Sana/source bytes 已足以说明现有实现和 Phase-6 launcher 的边界，
但不足以授权 CACH 执行。当前正确状态是：

```text
source evidence: audited
design: draft
root governance mirrors: deployed and byte-verified
executable governance/source lineage: blocked
REF-GDN-CORRECTED commissioning: blocked
CACH-A candidate: blocked
tests/model execution/training/evaluation: not run
formal /DATA root creation: denied
```

下一步只能是完成 Stage 0 draft pin graph 的静态 review，并在用户另行授权后进入
Stage 1 contract 实现；不得从本文直接进入测试、模型执行、训练或评测。
