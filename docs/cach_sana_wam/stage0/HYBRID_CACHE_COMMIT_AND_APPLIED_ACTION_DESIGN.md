# CACH-SANA-WAM Hybrid Cache、Paired Commit 与 Applied-Action 设计

> 状态：`DRAFT_BLOCKED / STAGE0_DESIGN_ONLY`
>
> 适用代码快照：H200 `/home/zch/workspace/sana-wam`
>
> 主仓 HEAD：`605f1c134b4c983ff80f8489c4bc8847036329e2`
>
> Sana gitlink/worktree：`16b9cec673e3335724ba2d8db25de7f9ed229292`
>
> 本文不授权实现、测试、训练、评测、capture 或正式 launcher admission。

## 1. 目的与结论

本文把主开发计划中的 hybrid temporal cache 与 paired `t=0` commit 落成
source-discriminated Stage 0 设计，并把 `APPLIED_ACTION_ACK` 固定为 future deploy
interface/blocker。核心结论固定如下：

1. GDN attention state 是 **recurrent full-history summary**；
2. softmax anchor 只读取 **当前 noisy chunk + 恰好一个 previous committed
   chunk**，commit 时用当前 clean chunk 覆盖 previous slot；
3. AttnRes 只保存一次 DiT forward 内的 depth routing state，绝不进入 temporal
   cache；
4. denoise 一律读取 immutable snapshot，不得接触 live mutable cache；
5. video/action pair 只能通过一次 transactional paired `t=0` commit 共同进入
   content-time state，且每次 commit 必须显式声明唯一 `commit_source`；
6. `teacher_forcing_dataset_pair`、`self_forcing_generated_pair` 与
   `deploy_applied_ack` 共享事务原子性，但各自的 source proof 不能互相替代；
7. environment/controller 未返回可读取、可复算的 canonical applied tensor 时，
   只阻塞 `deploy_applied_ack` 的 executed-action-conditioned 科学 admission，
   不能反向要求 teacher-forcing dataset commit 提供环境 ACK。

当前 H200 实现尚不满足 typed read-only state 与统一 transactional commit；
`SELF_FORCING_OBJECTIVE` 尚未闭合，deploy ACK seam 也不存在，因此本文状态保持
`DRAFT_BLOCKED`。

## 2. 约束边界

本文继承
`docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md`
的以下固定边界：

- `episode_bootstrap=first_frame_pinned`；
- `observed_prefix_chunks=0`；
- video/action 的 content-time 由已核验 `ChunkActionLayout` instance 决定；
- denoise 不写 cache；
- 每个 completed video/action pair 恰好 commit 一次；
- 每次 commit 的 source discriminator 只能是
  `teacher_forcing_dataset_pair`、`self_forcing_generated_pair` 或
  `deploy_applied_ack`；
- reset 清空 temporal state、action cursor、pending transaction 和 depth state；
- 不使用 DAgger、人工 recovery、rerank 或 best-of-N；
- 本设计不改变 AFCC formal authority、root、receipt 或 treatment/control 语义。

本文不重新定义 raw-frame、latent-frame 与 action-token 的算术；这些必须由
`CHUNK_ACTION_LAYOUT_AND_BOOTSTRAP_DESIGN.md` 提供的已通过 instance 作为输入。
若 layout 尚未通过 rate/provenance gate，cache/commit verifier 必须直接拒绝，而
不是自行猜测 fixed ATC。

本 Stage 0 当前只把前两种 source 纳入 model/cache 实现设计：

- teacher-forcing 使用 dataset clean ground-truth pair；
- self-forcing 仅在独立 `SELF_FORCING_OBJECTIVE` 闭合后使用 detached generated
  pair。

`deploy_applied_ack` 在本文中只固定 future interface、proof 和 fail-closed blocker；
它不是本 Stage 0 的 deploy implementation/admission 声明。

## 3. 审计证据与 source identity

以下行号只对表中 SHA256 对应的文件字节有效。文件 SHA 或 Git identity 改变后，
本表必须重新生成，旧行号不得继续作为 admission 证据。

| Source | SHA256 | 相关行 | 已核验事实 |
|---|---|---:|---|
| `docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md` | `969d9668ffed07176844fcbd84062825f9d5f0af0325959b0068c38fe481f0dd` | 515–540, 730–743, 841–849, 1123–1140 | AttnRes forward-local、hybrid cache、ACK 和 C7/C8 gate 的上位契约 |
| `src/sana_wam/model/video_backbone/sana/adapter.py` | `6ef2b6c21119b1357c8cab938e0351df428a71992ea0e92f25460b1b0a05d1cf` | 625–708, 710–781 | runtime class rebind；cache 是每层 10-slot 裸 list；`run_chunk` 把同一 list 交给 vendored `forward_long` |
| `src/sana_wam/model/gdn_ar.py` | `15a75585a0a4e33b7b83c65dd5d40333c1d5ad05c1862b21412ce302dbadd08c` | 83–170, 185–217, 315–453 | cache 原地推进；denoise `save=False`、clean ingest `save=True`；训练用 shallow slot-list snapshot 避让后续 mutation |
| `src/sana_wam/deploy/gdn_ar_engine.py` | `886815fc78a433f8813d159ae6acc59f03866c383c17d4a0cdb861952e4c539d` | 141–156, 165–245, 278–282 | deploy 保存单一 live `_cache`；先 clean ingest，再在同一 cache 上多次 read-only denoise；reset 重新建裸 list |
| `third_party/Sana/diffusion/model/ops/fused_streaming.py` | `4775d1ea1101d7075d087428cf2f8bb2643a7de922fd241e15b65e3c4f4ea206` | 30–86, 652–747 | 10-slot layout 由数字位置隐式定义；GDN recurrent、shortconv 和 type flag 在低层原地改写 |
| `third_party/Sana/diffusion/model/nets/sana_gdn_blocks.py` | `ef15578c63f9815d670dc3ef4cf2876c4f236754bfdfb0d2070da387048a495e` | 1216–1250, 1253–1357 | cached GDN 要求 mutable cache；softmax 先读旧 K/V，commit 时覆盖为当前 K/V，再对旧+当前做 SDPA |
| `third_party/Sana/diffusion/model/nets/sana_gdn_camctrl_blocks.py` | `6e6859c0e35a810538f2c046a285e88d06978c56df17451b62a508ec56154de7` | 612–681, 744–858 | GDN/softmax camera branch 也直接复用并原地改写相同 slot list |
| `third_party/Sana/diffusion/model/nets/sana_multi_scale_video_camctrl.py` | `bfbd72dfb22a44e2843f1985049ecfbb1f887f5e5f5ed852905c4670e2029fbe` | 1566–1665, 1911–1961 | `forward_long` 把 `kv_cache[i]` 传入 block，并把 block 返回值写回 `kv_cache[i]` |
| `third_party/Sana/diffusion/model/nets/basic_modules.py` | `f5b8345d7ba35a32f612ebc2e98976d1707198bb8e24ccd95dfbfa3b4a77e2ca` | 374–415 | FFN temporal-conv 实际把 left context 写到 `kv_cache[-1]`，即 slot 9 |
| `third_party/Sana/diffusion/scheduler/self_forcing_flow_euler_sampler.py` | `8c246685cb2a7e9cf5c8954afdc7b39973876fe403a662929764788406bec023` | 156–164, 630–669, 675–843 | vendored sampler 在 denoise 后做额外 `t=0/save=True`；cache aggregation/eviction 会重写历史裸 list |
| `src/sana_wam/deploy/policy.py` | `f73405f54cd750fd8798995f37fe38b33dc45345f79283ca161c9b58c06c454d` | 88–182, 184–242, 253–284, 301–324 | action 一经 server 返回即进入 `action_history`；没有 applied ack state machine |
| `src/sana_wam/deploy/policy_server.py` | `2e89da14e2b52444f2bb59fb3d259dd742d46cacb5173176801b3c23769790f7` | 13–42, 291–352, 531–582, 843–960 | 协议只有 obs/action/reset；telemetry 的 `action` 是 server response；没有 controller ACK endpoint |
| `benchmarks/robotwin/sana_wam2robotwin_interface.py` | `667ac4019dfccb87a94304fd5dc162fff8284aacdcfe142179c9d949544c4dbd` | 521–642, 723–808 | client 收 command 后可重排/转换，再调用 `TASK_ENV.take_action`；`sent_action/post_state` 仅写 client-side telemetry |
| `benchmarks/utils/client.py` | `1c0d2d8f42128908206d80bebaa8137777b0389ebc4391a5236dbe2e55f2b583` | 102–145, 162–187 | HTTP client 只有通用 POST 和 reset helper，没有 applied-action acknowledgement |

表中 SHA 绑定当前 Stage 0 draft 字节；其中 `policy_server.py` 已加入
reserved-marker fail-closed guard，但该改动不增加 controller ACK endpoint。
表中其余事实仍只由对应 SHA 和行号支持，不能用 `git status` 或路径存在性替代
逐文件 SHA。

## 4. 当前实现的准确语义与根因

### 4.1 当前 GDN 是 full-history recurrent summary

GDN commit 把上一次 `S_kv/S_z` 作为 scan 初态，得到新的
`S_kv_new/S_z_new` 并覆盖 live slot。因而 chunk `c` 之后的 state 是
`0..c` 历史的 recurrent summary，而不是只保存 chunk `c` 的原始 token。

这一定义适用于：

- main GDN recurrent state；
- 若启用 camera GDN，camera recurrent state。

shortconv 和 FFN temporal-conv 只是各自 kernel 所需的 bounded left context，
不得混称为 full-history attention state。

### 4.2 当前 engine 中 softmax 实际是 exactly-one-previous-chunk

当前 deploy engine 只有一份 per-layer rolling list。softmax forward：

1. 读取调用开始时 slot 中的 old K/V；
2. `save_kv_cache=True` 时把当前 clean chunk K/V 覆盖进同一 slot；
3. 本次 attention 使用 old K/V 与当前 chunk K/V；
4. 下一次 commit 看到的 old K/V 因而恰好来自上一个 committed chunk。

所以本项目 v0 应把它写成：

```text
softmax_visible = previous_committed_chunk + current_candidate_chunk
```

不能把 vendored sampler 支持的 concat/window/sink 行为误投射到当前单一 rolling
engine。full-history softmax、sink token 或多 chunk window 都属于新 candidate。

### 4.3 AttnRes 当前不存在，也不属于 temporal cache

当前 pinned H200 `src/`、`configs/` 和相关 vendored model 中没有本 candidate 的
Block AttnRes 实现；主计划也将其列为 P0 implementation blocker。

未来 AttnRes 的 state 轴是 **network depth**，不是 episode/content time：

- 每次 `forward_chunk` 入口创建；
- attention/FFN sublayer 按已冻结 `ATTNRES_DESIGN` 更新；
- 一次 forward 返回或异常时销毁；
- denoise step 之间不复用；
- chunk 之间不复用；
- commit 结束不得导出到 `HybridTemporalState`；
- episode reset 后不应存在任何可清理的 persistent AttnRes tensor。

### 4.4 裸 list mutation 的根因

问题不是单个遗漏的 `.clone()`，而是 cache API 没有表达权限和时间：

1. **结构无类型。** GDN tensor、softmax K/V、camera state、shortconv、FFN
   tconv 和 type flag 共用 `[None] * 10`，调用方靠数字位置解释。
2. **同一 slot 多义。** slot 0/1 对 GDN 是 `S_kv/S_z`，对 softmax 是 K/V；
   slot 6 用 float-like flag 二次猜类型。
3. **声明与实际位置漂移。** fused layout 注释把 tconv 记为 slot 5，但
   `CachedGLUMBConvTemp` 实际读写 `[-1]`，即 slot 9。
4. **live object 直接下沉。** engine、architecture、adapter、`forward_long`、
   attention、camera branch 和 FFN 共享同一可变 list identity。
5. **read-only 只是布尔约定。** `save_kv_cache=False` 需要所有下游模块都正确
   尊重同一个 flag；类型系统和 API 均无法阻止新 operator 意外写入。
6. **content-time 在 list 外。** `start_f/end_f` 是另一个参数，cache 自身不绑定
   episode、layout、chunk interval 或 revision；错误 cache 与合法位置可被混配。
7. **shallow snapshot 只是局部补丁。** 训练用 `[list(slot) for slot in kv]`
   固定 checkpoint recompute 所见的 list bindings，但没有 schema、time identity、
   atomic paired commit 或 tensor mutation guard。
8. **aggregation 会改历史容器。** vendored self-forcing sampler 可重新拼接、
   覆盖和清空历史 chunk list，因此不能直接作为新 capability 的 authority。

因此 Stage 1 不能继续把裸 list 暴露给 engine。vendored operator 仍可通过一个窄
compatibility codec 使用 legacy list，但该 list 必须是 transaction-local scratch，
不得是 live authority。

### 4.5 当前 deploy video-only ingest 不是 `deploy_applied_ack` paired commit

当前 `GDNARInferenceEngine` 在 step `c>=1` 时把刚观察到的 video chunk 以
`save_kv_cache=True` ingest，`noisy_actions=None`；随后用同一 cache denoise 当前
video/action candidate。该路径没有：

- canonical applied action；
- video/action interval binding；
- `commit_source=deploy_applied_ack`；
- staging state；
- expected revision/CAS；
- commit receipt；
- rollback；
- duplicate/stale ACK handling。

它只能证明“deploy video clean ingest 后 cache 推进”，不能证明
`deploy_applied_ack` source 下 completed observed-video/applied-action pair 的事务
提交。这个缺口不意味着 teacher-forcing 必须等待 controller ACK；teacher source
需要的是 dataset/layout/row identity。

### 4.6 当前 controller seam 只能证明 command

当前数据流为：

```text
server response action
  -> policy.action_history（立即记录）
  -> client action_indices（可选）
  -> EEF 20D -> 16D conversion（可选）
  -> TASK_ENV.take_action(...)
  -> client-side sent_action/post_state telemetry（仅 telemetry enabled 时）
```

其中：

- `server_action` 是 server command；
- `policy_action` 可能经过 client reorder；
- `sent_action` 是传给 simulator/controller API 的参数；
- `post_state` 是随后读到的 achieved state；
- 上述任一项都没有作为 ACK 回到 server/engine；
- 即使 `sent_action` 被记录，也不能自动证明 safety clipping、controller transform、
  丢步或 actuator 接收后的 canonical applied values。

所以当前唯一合法标签是：

```text
deploy_action_evidence_mode = commanded_action_only
```

在 ACK seam 完成前，只有 `deploy_applied_ack` 的
executed-action-conditioned cache admission 保持 P0 blocked。dataset
teacher-forcing commit 的 source proof 与该 blocker 正交；self-forcing 则由
`SELF_FORCING_OBJECTIVE` 单独阻塞。

## 5. Typed content-time state

以下为 schema 语义，不要求 Stage 1 机械照抄类名；实现若改名，字段与不变量不得
减弱。

### 5.1 Content-time identity

```text
ContentTime {
  episode_id: NonEmptyString,
  episode_epoch: uint64,
  layout_spec_sha256: HexSha256,
  layout_instance_digest: HexSha256,
  chunk_id: uint64,

  latent_start: uint64,
  latent_end_exclusive: uint64,
  raw_observation_start: uint64,
  raw_observation_end_exclusive: uint64,
  action_token_start: uint64,
  action_token_end_exclusive: uint64,

  bootstrap_mode: "first_frame_pinned",
  observed_prefix_chunks: 0
}
```

约束：

- 所有 interval 使用 half-open `[start,end)`；
- `chunk_id` 只能来自 layout instance，不能由 cache depth 猜测；
- 下一个 commit 的 start 必须精确等于当前 committed boundary；
- `layout_spec_sha256` 是算法/静态参数 identity；
- `layout_instance_digest` 是这一 episode/chunk 的动态映射 identity；
- 两种 digest 不得互换；
- bootstrap latent 0 是 observed condition，不伪造 preceding action span。

### 5.2 Layer union

```text
GDNLayerTemporalState {
  kind: "gdn_full_history_v1",
  layer_index: uint32,
  through: ContentTime,
  main_s_kv: TensorRef,
  main_s_z: TensorRef,
  camera_s_kv: Optional[TensorRef],
  main_shortconv_left_context: Optional[TensorRef],
  ffn_tconv_left_context: Optional[TensorRef],
  tensor_manifest_digest: HexSha256
}

SoftmaxLayerTemporalState {
  kind: "softmax_previous_chunk_v1",
  layer_index: uint32,
  previous_chunk: ContentTime,
  main_k_post_rope: TensorRef,
  main_v: TensorRef,
  camera_k_post_ucpe: Optional[TensorRef],
  camera_v_post_ucpe: Optional[TensorRef],
  ffn_tconv_left_context: Optional[TensorRef],
  tensor_manifest_digest: HexSha256
}
```

语义：

- GDN `through.chunk_id=c` 表示 recurrent state 已吸收所有 committed chunks
  `0..c`，但不声称能够反演这些原始 token；
- softmax `previous_chunk.chunk_id=c` 表示 slot 只保存 chunk `c` 的 clean K/V；
- softmax state 不允许出现 concat 后的多 chunk token count；
- temporal-conv left context 以独立字段表达，不能借 slot 5/9 猜测；
- `TensorRef` 包含 dtype、shape、device、stable logical tensor id、runtime storage
  alias id 和 digest；
- tensor 必须 detached，live state 不得要求 autograd。

`logical_tensor_id` 由 state manifest 按 layer/field/content-time 确定性分配，进入
portable canonical digest。`runtime_storage_alias_id` 只用于同一进程内检测
data_ptr/storage alias 与意外原地 mutation，不得进入 receipt、portable manifest
或任何跨进程 SHA；重启 verifier 不依赖 allocator address。

AttnRes 不属于这个 union。若 serialized cache manifest 出现 AttnRes tensor，
verifier 必须 fail-closed。

### 5.3 Episode state

```text
HybridTemporalState {
  schema: "cach.hybrid_temporal_state.v1",
  episode_id: NonEmptyString,
  episode_epoch: uint64,
  revision: uint64,
  committed_through_chunk: Optional[uint64],
  next_chunk_id: uint64,
  action_cursor: uint64,
  layout_spec_sha256: HexSha256,
  layer_states: Tuple[GDNLayerTemporalState | SoftmaxLayerTemporalState, ...],
  committed_pair_digest: Optional[HexSha256],
  last_commit_source: Optional[
    "teacher_forcing_dataset_pair"
    | "self_forcing_generated_pair"
    | "deploy_applied_ack"
  ],
  last_source_proof_digest: Optional[HexSha256],
  state_manifest_digest: HexSha256,
  pair_evidence_mode:
    "none"
    | "dataset_ground_truth"
    | "detached_generated"
    | "applied_ack"
    | "commanded_action_only"
}
```

初态固定：

```text
revision = 0
committed_through_chunk = None
next_chunk_id = 0
action_cursor = 0
layer_states = typed empty states
committed_pair_digest = None
last_commit_source = None
last_source_proof_digest = None
pair_evidence_mode = "none"
```

`first_frame_pinned` bootstrap observation保存在当前 inference request，不作为一次
虚构的 chunk `-1` commit，也不增加 revision。

`last_commit_source` 和 `pair_evidence_mode` 必须按固定映射出现：

| `last_commit_source` | `pair_evidence_mode` | 合法 source proof |
|---|---|---|
| `teacher_forcing_dataset_pair` | `dataset_ground_truth` | dataset manifest + layout + row identity |
| `self_forcing_generated_pair` | `detached_generated` | closed objective + generated trace + detach proof |
| `deploy_applied_ack` | `applied_ack` | environment/controller ACK + observation interval |

`commanded_action_only` 不是合法 paired commit source，只能作为 deploy diagnostic
state；它不得推进 `revision`、`committed_through_chunk` 或 `action_cursor`。

### 5.4 Read view 与 legacy codec

对 vendored operator 的兼容只允许经过：

```text
TypedState -> export_scratch_legacy_cache() -> vendored forward
          -> import_and_validate_scratch() -> StagedTypedState
```

规则：

- `export_scratch_legacy_cache` 必须产生新的 container identity；
- 不允许把 `HybridTemporalState.layer_states` 中的 mutable container 直接下沉；
- denoise 的 scratch tensor 要么 clone，要么由 copy-on-write wrapper 保护；
- denoise 前后计算 typed manifest 与 scratch mutation manifest；
- `save_kv_cache=False` 时任何 slot binding、tensor bytes、shape/dtype/device 变化都
  是 verifier failure，即便最终 output 看起来合理；
- 只有 commit transaction 可以调用 `import_and_validate_scratch`；
- imported type 必须与注册 layer kind 一致，不能信任 legacy slot 6 自报类型；
- slot 6 只作为兼容一致性检查，不再是 source of truth；
- slot 9 tconv 必须显式映射，slot 5 必须按注册 schema 为 empty/reserved。

## 6. Hybrid cache 的精确行为

### 6.1 GDN full-history

对预测 chunk `c` 的 denoise：

```text
read state = GDN summary through committed chunk c-1
write state = forbidden
```

对 chunk `c` 的 paired commit：

```text
staged state = scan(source-authorized clean/detached video c,
                    source-authorized clean/detached action c,
                    previous GDN summary through c-1,
                    timestep video=0,
                    timestep action=0)
staged.through = c
```

这里的 pair 由 `commit_source` 决定：teacher 是 dataset GT pair，self-forcing 是
detached generated pair，deploy 才是 observed video + canonical applied action。
三者共享 scan/atomicity，不共享 proof。

必须验证：

- commit 前后 state finite；
- shape/dtype/device 与 layer spec 一致；
- `through` 单调增加 1；
- commit 后 state digest 变化，除非注册的 empty/no-op corner case明确允许；
- 用重新 replay `0..c` 的 CPU/reference path 对照时，在注册容差内一致；
- 不得通过保留全部 raw K/V 冒充 recurrent state。

### 6.2 Softmax exactly-one-previous

对预测 chunk `c`：

```text
query = current noisy candidate c
keys/values = clean committed chunk c-1 + current noisy candidate c
```

对 chunk `c` 的 paired commit：

```text
staged previous_kv = source-authorized t=0 K/V from chunk c
discard previous_kv from chunk c-1
```

必须验证：

- `c=0` 时没有 previous K/V；
- `c>=1` 时 previous metadata 的 chunk id 必须是 `c-1`；
- commit 后 token count 精确等于 chunk `c` 的注册 token count；
- 不得出现 chunk `0..c` concat；
- 不得静默启用 sink token；
- current source-authorized `t=0` K/V 只能在 paired commit staging 中生成，不能
  在 denoise 中写入；
- camera K/V 与 main K/V 同一 content-time、同一 transaction。

### 6.3 AttnRes forward-local

每一次 denoise forward 或 commit forward 都创建独立：

```text
AttnResForwardState {
  forward_nonce,
  sublayer_router_state,
  completed_block_summaries,
  current_partial_sum,
  block_span=8
}
```

其 lifecycle：

```text
create at forward entry
  -> update across registered depth order
  -> flush final 4-layer partial block for a 20-layer DiT
  -> consume output
  -> destroy at forward exit
```

以下行为全部禁止：

- 把 `AttnResForwardState` 放入 `HybridTemporalState`；
- 在 diffusion denoise step `i` 与 `i+1` 间复用；
- 在 chunk `c` 与 `c+1` 间复用；
- 用 episode reset 之外的隐式全局变量持有；
- 把 temporal `start_f/end_f` 加成 AttnRes router offset；
- 将 AttnRes source 误标为过去 episode content。

### 6.4 非 attention 的 temporal context

main shortconv、camera shortconv 和 FFN tconv 必须逐 operator 注册：

- 是否跨 chunk；
- 保留多少 left-context frames/tokens；
- dtype/shape；
- 对应 content-time end；
- reset 语义。

v0 不允许用“10-slot cache”概括这些差异。尤其 FFN tconv 的实际 slot 9 行为必须
由 typed codec 固定，不能延续 slot 5/9 双重解释。

## 7. Read-only denoise contract

建议 engine-facing API：

```text
read_view = cache.snapshot_for_denoise(
    expected_episode_id,
    expected_revision,
    expected_next_chunk_id,
    layout_instance_digest
)

prediction = model.denoise_chunk(
    video_candidate,
    action_candidate,
    read_view=read_view,
    save_temporal_state=False
)
```

强制条件：

1. `read_view` 绑定 revision 和 next content-time；
2. forward 前记录 live `state_manifest_digest`；
3. forward 使用 transaction-local legacy scratch；
4. forward 后再次计算 live digest，必须 byte-identical；
5. scratch 若发生写入也视为 contract failure，而不是丢弃后继续；
6. 多个 diffusion step 都读同一 committed revision；
7. exception、OOM 或取消不得改变 live revision；
8. JVP/gradient 检查使用同一 read view，不产生 cache side effect；
9. 并发请求必须持有 episode/revision lease，不能交错使用同一 state；
10. debug hook、bridge capture 和 telemetry 不得保留 mutable cache reference。

`save_kv_cache=False` 仍可传给 legacy operator，但它只是第二道 assertion，不再承担
唯一的写保护责任。

## 8. Transactional paired `t=0` commit

### 8.1 `commit_source` discriminated union

每次 transaction 必须恰好选择一种 source：

```text
commit_source =
    "teacher_forcing_dataset_pair"
  | "self_forcing_generated_pair"
  | "deploy_applied_ack"
```

| Source | video/action pair | Mandatory proof | 环境 ACK |
|---|---|---|---:|
| `teacher_forcing_dataset_pair` | 同一 dataset row/layout interval 的 clean GT pair | dataset manifest、episode/row identity、layout instance、row tensor digests | 不需要 |
| `self_forcing_generated_pair` | 已完成的 model-generated video/action pair，进入 cache 前 detach | 已闭合 `SELF_FORCING_OBJECTIVE`、generation trace、checkpoint/noise/layout identity、detach proof | 不需要 |
| `deploy_applied_ack` | environment-observed video + controller-confirmed canonical applied action | `APPLIED_ACTION_ACK`、controller transform、observation interval、layout binding | 必须 |

三种 source 只共享：

- typed content-time；
- read-only denoise；
- transaction state machine；
- staged `t=0` pair forward；
- CAS/atomic swap；
- revision/cursor 单调性；
- receipt/failure freeze。

它们不共享 source proof。尤其禁止：

- 用 environment ACK 替代 teacher dataset row identity；
- 用 dataset GT action 替代 self-forcing generated action；
- 用 generated pair 替代 deploy applied values；
- 用 server command、sent action 或 post-state替代 `deploy_applied_ack`；
- 把一个 source 的 optional metadata拼到另一个 source上，以绕过其 mandatory
  verifier。

`commit_source` 必须进入 transaction id、paired payload digest、state manifest
和 receipt；同一 `commit_id` 不得在 retry 时改变 source。

### 8.2 Source-specific proof

#### 8.2.1 `teacher_forcing_dataset_pair`

```text
TeacherForcingDatasetPairProof {
  schema: "cach.teacher_forcing_dataset_pair_proof.v1",
  dataset_manifest_sha256,
  dataset_episode_id,
  dataset_row_identity,
  dataset_row_start,
  dataset_row_end_exclusive,
  layout_spec_sha256,
  layout_instance_digest,
  video_tensor_digest,
  action_tensor_digest,
  action_mask_digest,
  row_order_manifest_sha256,
  source_proof_digest
}
```

前置条件：

- video/action 必须来自同一 frozen dataset episode、row interval 和 layout；
- pair 是 clean ground truth，不能来自当前 model output；
- action mask/pad semantics 必须闭合；
- row identity、tensor bytes 和 layout coverage 逐项复算；
- 不读取或要求 `APPLIED_ACTION_ACK`；
- 若 payload 同时声称 environment ACK 是 teacher proof，verifier 以
  `COMMIT_SOURCE_PROOF_MIXED` 拒绝。

teacher-forcing commit 可在没有 robot/controller 的离线训练环境中合法发生。

#### 8.2.2 `self_forcing_generated_pair`

```text
SelfForcingGeneratedPairProof {
  schema: "cach.self_forcing_generated_pair_proof.v1",
  self_forcing_objective_sha256,
  objective_status: "closed",
  candidate_spec_sha256,
  source_checkpoint_digest,
  generation_trace_digest,
  model_noise_schedule_digest,
  model_noise_seed,
  layout_spec_sha256,
  layout_instance_digest,
  generated_video_tensor_digest,
  generated_action_tensor_digest,
  generated_action_mask_digest,
  stop_gradient_attestation_digest,
  source_proof_digest
}
```

前置条件：

- `SELF_FORCING_OBJECTIVE` 必须先闭合并由 candidate authority pin SHA；
- video 和 action 必须来自同一次注册 generation trace、同一 chunk/layout；
- 两个 tensor 在进入 cache commit 前都必须 detach；
- committed generated prefix 不得保留通向生成过程的 autograd edge；
- target identity、generated-prefix action-loss mask 和 schedule 必须由 objective
  固定；
- dataset GT pair不能填补缺失的 generated action；
- environment ACK不能替代 generation trace 或 detach proof。

在 objective 未闭合时，即使 tensor shape 正确，也必须触发
`COMMIT_SELF_FORCING_OBJECTIVE_UNCLOSED`。

#### 8.2.3 `deploy_applied_ack`

source proof 是 §9 的完整 ACK、可读取 canonical applied tensor、controller
transform manifest、observation interval receipt 和 observed video digest。
它是唯一要求 environment/controller ACK 的 source。

本 Stage 0 当前不把该 source 纳入 model/cache implementation scope；§9–10 只固定
future seam 和 science blocker。不得因 teacher/self transaction skeleton 存在而
宣称 deploy paired commit 已实现。

### 8.3 Shared commit 前置条件

chunk `c` 只有在以下 common input 全部到齐后才能进入 `PREPARED`：

- 已通过 schema/verifier 的 `ChunkActionLayout` instance；
- 与 layout 对齐的 video/action pair；
- exactly one `commit_source`；
- 与 source discriminator严格匹配的完整 source proof；
- live state 的 expected episode、epoch、revision 和 `next_chunk_id=c`；
- 没有另一个 pending transaction。

此外必须满足 8.2 的 source-specific proof。缺 common pair 任一侧、缺 source proof
或混用 proof，都不得 commit 或 cursor advance。这里禁止的是 unpaired commit，
不是要求所有 source 都提供 environment ACK。

### 8.4 Transaction state machine

```text
IDLE
  -> PREPARING
  -> PREPARED
  -> COMMITTING
  -> COMMITTED

PREPARING | PREPARED | COMMITTING
  -> ABORTED
```

- `COMMITTED` 是唯一能改变 live state 的 terminal；
- `ABORTED` 保留 failure receipt，但 live state 和 revision 不变；
- 进程退出、reset、timeout 或任何未捕获异常都把非 terminal transaction 判为
  `ABORTED`；
- 同一 `commit_id`、相同 payload digest 的 retry 返回既有 receipt，不再次推进；
- 同一 `commit_id`、不同 payload digest 是冲突，fail-closed。

### 8.5 Staging 顺序

```text
1. validate ContentTime and expected revision
2. validate exactly one commit_source
3. dispatch only to the matching source-proof verifier
4. resolve source-authorized video/action tensor bytes:
     teacher -> clean dataset GT pair
     self_forcing -> detached generated pair
     deploy -> observed video + canonical applied tensor
5. recompute pair dtype/shape/order/mask/value digests
6. validate source-specific identity and layout binding
7. export live typed state to transaction-local scratch
8. run one paired t=0 forward:
     video_timestep = 0
     action_timestep = 0
     video = source-authorized video tensor
     actions = source-authorized action tensor
     write_target = scratch only
9. import scratch into staged typed state
10. validate GDN/softmax/tconv semantics, finiteness and content-time
11. build source-tagged paired payload digest and staged state manifest
12. acquire the exclusive commit/read-publication lock
13. CAS(expected episode_epoch, expected revision) while live pointer remains old
14. write, fsync and read-back-verify the final commit receipt
15. atomically publish the entire HybridTemporalState pointer
16. verify published state digest == receipt state_manifest_after, then unlock
```

步骤 4 的 action tensor 由 source 决定。teacher 使用 clean dataset action；
self-forcing 使用 detached generated action；deploy 使用 environment-confirmed
applied values，绝不能使用 server command。action head 若本身没有跨 chunk
cache，其 `t=0` forward 输出可以丢弃；action-to-video conditioner 和
committed-action summary 仍必须消费同一 source-authorized action tensor。

### 8.6 Atomic swap

live authority 只能有一个指针：

```text
self._state: HybridTemporalState
```

禁止按 layer 逐个写 live state。实现必须先构造完整 staged object；exclusive
commit/read-publication lock 持有期间，reader 只能看到 old pointer，不能看到
private staged state。final receipt durable 且 read-back 验证成功后才一次发布
pointer。receipt 写入、fsync 或验证失败时丢弃 staged object，old pointer/revision
保持不变并进入 `ABORTED`。发布时：

- `revision_new = revision_old + 1`；
- `committed_through_chunk = c`；
- `next_chunk_id = c + 1`；
- `action_cursor = layout.action_token_end_exclusive(c)`；
- `last_commit_source = transaction.commit_source`；
- `last_source_proof_digest = transaction.source_proof_digest`；
- `committed_pair_digest` 同时覆盖 commit source、video/action、layout 和对应
  source proof；只有 deploy source 额外覆盖 observation interval/controller ACK；
- 所有 layer content-time 一致；
- AttnRes state 不存在；
- receipt 写失败时不得发布 staged state或宣称 commit 完成；
- durable receipt 后若 pointer publish/digest verify 发生不可恢复错误，必须
  poison 当前 episode 并终止 serving/training process；不得继续读取 old/new
  任一 state。重启只能根据 durable receipt 与 source proof确定性 replay
  `state_manifest_after`，复核 digest 后恢复；无法 replay 就保持 fail-closed。

因此不存在“live revision 已推进但 receipt 缺失”的可消费窗口。若进程恰在 receipt
durable 后、pointer publish 前崩溃，recovery 把 receipt 视为唯一 commit decision，
在接收下一 chunk 前完成 deterministic replay；不能把该事务降级成 aborted 或再次
推进 revision。

### 8.7 Commit receipt

```text
PairedCommitReceipt {
  schema: "cach.paired_commit_receipt.v1",
  commit_id,
  transaction_nonce,
  episode_id,
  episode_epoch,
  revision_before,
  revision_after,
  chunk_id,
  layout_instance_digest,
  commit_source:
    "teacher_forcing_dataset_pair"
    | "self_forcing_generated_pair"
    | "deploy_applied_ack",
  source_proof_schema,
  source_proof_digest,
  pair_video_digest,
  pair_action_digest,
  pair_action_mask_digest,

  # teacher_forcing_dataset_pair only
  dataset_manifest_sha256: Optional[HexSha256],
  dataset_row_identity: Optional[String],

  # self_forcing_generated_pair only
  self_forcing_objective_sha256: Optional[HexSha256],
  generation_trace_digest: Optional[HexSha256],
  stop_gradient_attestation_digest: Optional[HexSha256],

  # deploy_applied_ack only
  applied_action_ack_digest: Optional[HexSha256],
  action_values_digest: Optional[HexSha256],
  observation_interval_digest: Optional[HexSha256],
  controller_transform_digest: Optional[HexSha256],

  state_manifest_before,
  staged_state_manifest,
  state_manifest_after,
  committed_at_monotonic_ns,
  result: "committed" | "aborted",
  failure_code: Optional[String]
}
```

每种 source 的专属字段集合必须精确匹配 discriminator：mandatory 字段缺失失败，
另两种 source 的 proof 字段被用来满足当前 source 也失败。receipt 可以保留明确
标为 non-authoritative 的诊断 metadata，但它不得进入 `source_proof_digest` 或被
verifier 当作 source closure。

wall-clock 时间不能作为 ordering source；ordering 由 episode epoch、revision、
chunk id 和 monotonic transaction sequence 决定。

## 9. `APPLIED_ACTION_ACK` v1

本节只适用于 `commit_source=deploy_applied_ack`。teacher-forcing 和 self-forcing
transaction 不消费本 schema，也不得因为缺 ACK 被拒绝；它们分别使用 8.2.1 和
8.2.2 的 source proof。

### 9.1 Transport schema

```text
APPLIED_ACTION_ACK {
  schema: "cach.applied_action_ack.v1",
  ack_id: NonEmptyString,
  episode_id: NonEmptyString,
  episode_epoch: uint64,
  command_id: NonEmptyString,
  chunk_id: uint64,
  layout_instance_digest: HexSha256,

  canonical_applied_tensor_or_immutable_ref: {
    kind: "inline_tensor" | "immutable_ref",

    # inline_tensor
    data_base64: Optional[Base64],

    # immutable_ref
    uri: Optional[RegisteredImmutableURI],
    byte_offset: Optional[uint64],
    byte_length: uint64,
    object_sha256: Optional[HexSha256]
  },

  dtype: "float32_le",
  shape: [applied_count, action_dim],
  token_order: "layout_action_token_ascending_v1",
  action_representation_id: NonEmptyString,
  action_units_id: NonEmptyString,
  action_values_digest: HexSha256,
  applied_count: uint64,
  controller_transform_digest: HexSha256,
  observation_interval_digest: HexSha256,

  controller_sequence_start: uint64,
  controller_sequence_end_exclusive: uint64,
  controller_monotonic_start_ns: uint64,
  controller_monotonic_end_ns: uint64,
  completion_status: "fully_applied",
  ack_payload_digest: HexSha256
}
```

对 `deploy_applied_ack`，主计划要求的字段全部是 mandatory；新增的 command、
epoch、representation、units 和 controller sequence 字段用于消除跨 episode、
跨 transform 和 retry 歧义。

### 9.2 Canonical tensor

v1 固定：

- little-endian IEEE float32；
- C-contiguous；
- rank 2；
- shape 为 `[applied_count, action_dim]`；
- token 按 layout action interval 递增；
- feature 顺序由 `action_representation_id` 的 immutable spec 固定；
- `action_values_digest = SHA256(raw canonical tensor bytes)`；
- NaN/Inf、负零 canonicalization 歧义或未注册单位直接拒绝；
- `applied_count` 必须等于当前 layout chunk 的 non-pad action count；
- partial tail 只能按 layout 明确的 tail instance 接受，不能靠 ACK 自报缩短。

若 controller 原生表示与训练 action representation 不同，ACK producer 必须输出
controller 已实际采用值在注册 canonical representation 中的确定性映射，并用
`controller_transform_digest` 绑定：

- transform source/config；
- safety clipping/limits；
- reorder；
- unit conversion；
- EEF/rotation conversion；
- interpolation/hold policy。

仅提供 pre-transform command 或“已调用 controller API”布尔值不合格。

### 9.3 Inline 与 immutable ref

`inline_tensor`：

- base64 解码长度必须等于 `prod(shape) * 4`；
- server 解码后复算 digest；
- JSON 数字数组不是 canonical bytes，v1 不接受。

`immutable_ref`：

- URI scheme 必须在 authority allowlist；
- 必须提供 exact byte range、object size/digest；
- verifier 读取 bytes，而不是只信 ref metadata；
- 读取后复算 tensor digest；
- ref 对象在 commit receipt 持久化前被替换或改变时拒绝；
- mutable path、symlink 跳转、缺失 object digest 或短读一律拒绝。

digest-only ACK 明确不合格，因为 digest 无法构造 action conditioner 的输入。

### 9.4 Observation interval binding

`observation_interval_digest` 必须来自 canonical interval receipt，至少覆盖：

```text
episode_id
episode_epoch
chunk_id
layout_instance_digest
raw observation [start,end)
action token [start,end)
controller sequence [start,end)
pre-observation digest
post-observation digest
camera ordering and presence mask
timestamp/rate provenance identity
```

ACK、observed video 和 layout 三者的 interval digest 必须相同。只对 post-state
求 hash、只记录一张终帧或只使用 request step 都不足以证明 interval binding。

### 9.5 ACK acceptance

deploy ACK verifier 必须按以下顺序 fail-closed：

1. schema/version；
2. episode id/epoch；
3. command id 与 pending command；
4. chunk id 与 live `next_chunk_id`；
5. layout instance digest；
6. completion status；
7. canonical bytes 可读性；
8. dtype/shape/order/representation/units；
9. action values digest；
10. applied count 和 layout coverage；
11. controller sequence contiguous；
12. controller transform digest allowlist；
13. observation interval digest；
14. ack payload digest；
15. duplicate/conflict check。

任何失败都不能退回 command 代替 applied value。

## 10. Server/controller seam

本节仅为 `deploy_applied_ack` future seam。它不属于当前 Stage 0
teacher/self-forcing commit implementation scope。

### 10.1 Server response 必须增加 command identity

未来 action response 至少需要：

```text
{
  "type": "action",
  "episode_id": "...",
  "episode_epoch": 7,
  "command_id": "...",
  "chunk_id": 3,
  "layout_instance_digest": "...",
  "action": [...],
  "action_representation_id": "...",
  "controller_transform_expected_digest": "...",
  "ack_required": true
}
```

`command_id` 必须唯一且由 server 绑定 pending state；request counter 不能替代。

### 10.2 ACK transport

可选实现：

- HTTP `POST /applied-action-ack`；
- WebSocket message `type=applied_action_ack`；
- 下一次 observation payload 内嵌 ACK。

无论 transport 如何，server-side state machine必须相同：

```text
READY_TO_COMMAND
  -> COMMAND_OUTSTANDING
  -> ACK_VALIDATED
  -> OBSERVATION_BOUND
  -> PAIRED_COMMITTED
  -> READY_TO_COMMAND
```

在 `COMMAND_OUTSTANDING`：

- 不得把 command 加入 applied history；
- 不得提交 video；
- 不得跨到下一 layout chunk；
- 缺 ACK、partial ACK 或 observation 不匹配时 fail-closed；
- reset 可终止 pending command，但之后到达的旧 ACK 必须 stale-reject。

### 10.3 Policy history 拆分

当前 `action_history` 需拆为：

```text
command_history       # 工程诊断，server-returned
applied_action_history # 科学输入，只由 accepted ACK 产生
```

`conditions["action_history"]` 在新 variant 中只能绑定
`applied_action_history`。若保留旧字段名，schema 必须同时带
`pair_evidence_mode=applied_ack`；否则 deploy launcher 拒绝。

### 10.4 Controller responsibilities

controller/environment adapter 必须：

1. 接收 command identity；
2. 完成 reorder、conversion、clipping 和执行；
3. 记录每个实际 applied token；
4. 输出 canonical tensor/ref；
5. 固定 transform digest；
6. 固定 controller sequence 和 monotonic interval；
7. 绑定产生的 observation interval；
8. 发送 ACK；
9. 保留 failure evidence。

`TASK_ENV.take_action(...)` 返回 `None` 或只读取 post-state 时，不能推断 applied
token。RoboTwin adapter 必须得到明确 environment/controller receipt，或诚实保持
`commanded_action_only`。

### 10.5 Diagnostic-only mode

为了调试网络/shape，可允许显式：

```text
pair_evidence_mode = commanded_action_only
science_admission = false
```

该模式：

- receipt 必须显著标注 diagnostic；
- 不能生成 `deploy_applied_ack` paired commit authority，也不影响合法的离线
  teacher source receipt；
- 不能进入 CACH-A/H/R/SF campaign；
- 不能宣称 executed-action conditioning；
- 不能在失败后自动升级成 `applied_ack`；
- command/applied mismatch 不能被隐去。

默认配置必须 fail-closed，不得自动回退到此模式继续正式流程。

## 11. Reset 契约

`reset(new_episode_context)` 必须原子完成：

1. 将所有非 terminal commit transaction 标记 `ABORTED_RESET`；
2. 清除 pending source proof；deploy mode 额外清除 outstanding command/ACK；
3. 增加 `episode_epoch`，即使复用同名 episode id；
4. 创建 typed empty GDN/softmax/tconv states；
5. `revision=0`；
6. `next_chunk_id=0`；
7. `action_cursor=0`；
8. 清空 source-specific pair history；deploy mode额外清空 applied/command history；
9. 清空 `last_commit_source`、`last_source_proof_digest` 和 pending observation
   interval；
10. 重建 first-frame bootstrap request state；
11. 验证没有 AttnRes persistent state；
12. 生成 reset receipt。

reset 后到达的旧 ACK、旧 observation 或旧 read view 必须因 epoch/revision 不匹配
而拒绝。相同 `episode_key` 的 idempotent reset 只有在 context、epoch intent 和
pending state完全相同且未开始 command 时才可 no-op；不能用旧 server 仅比较 noise
seed 的规则绕过 pending transaction。

## 12. Verifier 设计

当前 Stage 0 的 teacher/self implementation design 建议增加前两个只读
verifier；第三个 deploy ACK verifier 只固定 future interface，不宣称当前实现。

### 12.1 `verify_cach_hybrid_cache_contract.py`

输入：

- candidate spec；
- layer registry；
- layout spec/instance；
- initial/final typed state manifest；
- denoise mutation receipts；
- paired commit receipts；
- source proof receipts；
- reset receipt。

输出：

```text
status: PASS | FAIL
failure_codes: [...]
verified_source_sha256: {...}
verified_state_manifest: ...
```

最低检查：

- layer count/kind 与 architecture registry 完全一致；
- GDN through-time 单调且 full-history replay 闭合；
- softmax 只保留 one previous chunk；
- main/camera/tconv content-time 一致；
- slot 5/9 codec 无歧义；
- denoise live bytes不变；
- 每 chunk revision 只增加一次；
- `commit_source`、pair video/action、layout 与 source proof digest 闭合；
- teacher receipt 不要求 ACK，且只能绑定 dataset proof；
- self-forcing receipt 只在 objective closed 且 pair detached 时通过；
- deploy receipt 才要求 observation/controller/ACK 闭合；
- AttnRes 不在 serialized state；
- reset 恢复 empty schema。

### 12.2 `verify_cach_commit_source_proof.py`

输入：

- `commit_source`；
- source-specific proof；
- pair tensor manifests；
- candidate/objective/dataset authority；
- layout instance。

最低检查：

- discriminator 只能取三种冻结值；
- proof schema 与 discriminator精确匹配；
- `teacher_forcing_dataset_pair` 复算 dataset manifest、episode/row、layout、
  video/action/mask digest，不查询 environment ACK；
- `self_forcing_generated_pair` 验证 objective status/SHA、generation trace 与
  stop-gradient/detach proof；
- `deploy_applied_ack` 只路由到 12.3，不接受 dataset/generated proof替代；
- mixed、missing 或 unknown proof deterministic fail-closed。

### 12.3 `verify_cach_applied_action_ack.py`（future deploy）

输入：

- pending command receipt；
- ACK；
- immutable tensor object或 inline bytes；
- controller transform manifest；
- observation interval receipt；
- layout instance。

最低检查：

- 按 9.5 的顺序完整验证；
- 实际读取 canonical bytes；
- 输出 recomputed action digest；
- 输出 coverage map；
- 不接受 digest-only；
- 只接受 `commit_source=deploy_applied_ack`；
- 不访问模型/GPU；
- 不修改 ACK、tensor object 或 run root。

所有 verifier 都必须写 deterministic machine-readable failure code，不能只写自由
文本 warning。verifier 自身 source SHA 必须进入 launcher authority。

## 13. Failure codes

至少冻结：

| Code | 条件 |
|---|---|
| `CACHE_SCHEMA_MISMATCH` | layer union/field 与注册 schema 不一致 |
| `CACHE_LAYER_KIND_MISMATCH` | GDN/softmax operator 与 state kind 不一致 |
| `CACHE_CONTENT_TIME_MISMATCH` | cache 与 layout/episode/chunk interval 混配 |
| `CACHE_DENOISE_MUTATION` | read-only denoise 改变 live 或 scratch state |
| `CACHE_SOFTMAX_HISTORY_GT_ONE` | softmax 持有多于一个 previous chunk |
| `CACHE_GDN_NONMONOTONIC` | GDN through-time 回退/跳跃 |
| `CACHE_TCONV_SLOT_AMBIGUOUS` | slot 5/9 同时有值或 codec 不闭合 |
| `CACHE_ATTNRES_PERSISTED` | AttnRes state 出现在 temporal state |
| `COMMIT_SOURCE_UNKNOWN` | discriminator 不是三种冻结 source |
| `COMMIT_SOURCE_PROOF_MISSING` | 当前 source 缺 mandatory proof |
| `COMMIT_SOURCE_PROOF_MIXED` | 用其他 source proof 满足当前 source |
| `COMMIT_MISSING_VIDEO_PAIR` | paired transaction 缺 source-authorized video |
| `COMMIT_MISSING_ACTION_PAIR` | paired transaction 缺 source-authorized action |
| `COMMIT_TEACHER_ROW_IDENTITY_MISMATCH` | teacher pair 与 dataset row/layout 不闭合 |
| `COMMIT_SELF_FORCING_OBJECTIVE_UNCLOSED` | objective 未闭合却请求 generated commit |
| `COMMIT_GENERATED_PAIR_NOT_DETACHED` | generated video/action 仍有 autograd edge |
| `COMMIT_GENERATED_TRACE_MISMATCH` | generated pair 不来自同一注册 trace/layout |
| `COMMIT_DEPLOY_ACK_MISSING` | deploy source 缺完整 environment ACK |
| `COMMIT_STALE_REVISION` | CAS expected revision 不匹配 |
| `COMMIT_OUT_OF_ORDER` | chunk/cursor 不连续 |
| `COMMIT_DUPLICATE_CONFLICT` | 相同 commit id 对应不同 payload |
| `COMMIT_PARTIAL_WRITE` | live state只更新部分 layer/组件 |
| `ACK_SCHEMA_MISMATCH` | ACK version/字段不合法 |
| `ACK_DIGEST_ONLY` | 只有 hash，没有可读取 canonical tensor |
| `ACK_TENSOR_UNREADABLE` | inline/ref bytes 无法读取或短读 |
| `ACK_DTYPE_SHAPE_ORDER_MISMATCH` | canonical tensor metadata 不匹配 |
| `ACK_VALUES_DIGEST_MISMATCH` | bytes 复算 hash 不同 |
| `ACK_PARTIAL_ACTION_SPAN` | applied count 未覆盖 layout span |
| `ACK_CONTROLLER_TRANSFORM_MISMATCH` | transform digest 未注册/不同 |
| `ACK_OBSERVATION_INTERVAL_MISMATCH` | ACK 与 observed video interval 不同 |
| `ACK_STALE_EPISODE` | id/epoch 已 reset |
| `ACK_UNKNOWN_COMMAND` | command id 不在 pending set |
| `ACTION_EVIDENCE_COMMANDED_ONLY` | 只有 command/sent action，禁止科学 admission |

failure code 一经触发，相关 run root 必须按 launcher 契约 fail-closed/freeze；不得
覆盖失败 root 后重试。

## 14. 必须覆盖的 verifier/failure tests

本节是未来测试计划，本轮没有运行。

### 14.1 Cache semantics

1. GDN commit `0,1,2` 后 `through` 精确为 `0,1,2`；
2. GDN staged state 与从空态 replay 全 prefix 的 reference 对照；
3. softmax 预测 `c` 只能看到 committed `c-1` 与 current candidate；
4. softmax commit `c` 后不再保留 `c-1`；
5. softmax token count 大于 one chunk 时失败；
6. camera state 与 main state chunk id 不同失败；
7. slot 5 与 slot 9 同时非空失败；
8. mixed GDN state/softmax layer失败；
9. AttnRes object出现在 cache manifest失败；
10. 20-layer、`S=8` 的 final 4-layer partial AttnRes forward结束后无残留。

### 14.2 Read-only denoise

1. 单步 denoise 前后 live manifest byte-identical；
2. 完整 N-step video/action co-denoise 前后 live manifest byte-identical；
3. 恶意 legacy operator 忽略 `save=False` 写 slot 时能被检测；
4. tensor in-place write、list rebind、append/pop 均被检测；
5. denoise exception/OOM simulation 后 revision 不变；
6. 两个 read view 交错使用不改变 state；
7. stale revision read view 被拒绝；
8. gradient checkpoint recompute 读取固定 snapshot，而非后续 live state。

### 14.3 Paired commit

1. 三种 source 的完整 pair 都只产生一次 revision increment；
2. 任一 source 下 video-only、action-only 都失败；
3. paired forward 的 video/action timestep 都是 0；
4. `teacher_forcing_dataset_pair` 无 environment ACK 时凭完整 dataset proof通过；
5. teacher 缺 row identity、跨 episode取 action或用 ACK 代替 row proof时失败；
6. `self_forcing_generated_pair` 在 objective 未闭合时失败；
7. generated video/action 任一未 detach、trace/layout不同或用 GT action补齐时失败；
8. self-forcing 缺 environment ACK不会单独导致失败；
9. `deploy_applied_ack` 缺 ACK 时失败，dataset/generated proof不能替代；
10. source discriminator、proof schema、state/receipt source映射不一致时失败；
11. commit forward异常时 live state不变；
12. 第 N 层 import失败时所有前层也不得落到 live；
13. CAS 前被 reset 时 transaction abort；
14. duplicate identical retry为 no-op；
15. duplicate differing source/payload retry失败；
16. chunk `c+1` 早于 `c` 失败；
17. layout digest相同但 episode epoch不同失败；
18. deploy observation interval来自另一 chunk失败；
19. commit receipt写入/校验失败时不得宣称 success。

### 14.4 Deploy ACK

本小节仅以 `commit_source=deploy_applied_ack` 运行；不得应用到 teacher/self source。

1. inline canonical float32 bytes正常通过；
2. immutable ref读取与 digest复算正常通过；
3. digest-only失败；
4. ref 缺 object digest、发生替换、短读或 symlink escape失败；
5. float64/big-endian/rank-1/shape错误失败；
6. token order错误失败；
7. action representation或units错误失败；
8. NaN/Inf失败；
9. applied count少一、多一或把 pad 当 applied失败；
10. command与applied不同但 ACK 正确时，以 applied tensor commit并保留 mismatch
    telemetry；
11. 只提供 command时触发 `ACTION_EVIDENCE_COMMANDED_ONLY`；
12. controller transform digest错误失败；
13. observation interval digest错误失败；
14. stale/unknown command、旧 episode ACK失败；
15. controller sequence有 gap/overlap失败；
16. 相同 ack id 不同 tensor失败。

### 14.5 Deploy server/controller/reset

1. response 含唯一 command/layout identity；
2. command outstanding 时 action 不进入 applied history；
3. ACK 未到不允许 `deploy_applied_ack` paired commit，不影响合法离线 teacher
   transaction；
4. ACK 到达但 observation 未绑定不允许 commit；
5. accepted ACK 后只 append canonical applied history；
6. reset 清除 outstanding command并使旧 ACK stale；
7. WebSocket/HTTP transport得到相同 verifier结果；
8. async inference、temporal ensemble或半 chunk regenerate在本 variant fail-closed；
9. diagnostic-only receipt不能被 launcher当 science authority；
10. server telemetry分别报告 commanded/sent/applied，不合并字段。

建议未来对应测试文件：

- `tests/test_cach_hybrid_cache_contract.py`
- `tests/test_cach_cache_content_time.py`
- `tests/test_cach_cache_read_only.py`
- `tests/test_cach_paired_commit_atomicity.py`
- `tests/test_cach_commit_source_proof.py`
- `tests/test_cach_teacher_forcing_dataset_commit.py`
- `tests/test_cach_self_forcing_generated_commit.py`
- `tests/test_cach_applied_action_ack.py`
- `tests/test_cach_engine_action_history.py`
- `tests/test_cach_episode_reset.py`
- `tests/test_cach_failure_receipt.py`

## 15. Stage 1 最小实现顺序

只有本文通过 review 且用户单独授权 Stage 1 后，才按以下顺序实施：

1. 定义 content-time、layer union、state manifest 与 deterministic digest；
2. 建立 layer registry，不再让 slot 6 决定真实 operator kind；
3. 实现 typed-state/legacy-scratch codec，固定 slot 9 tconv；
4. 把 denoise 改成 immutable read view并增加 mutation detector；
5. 实现 staging/CAS/atomic-swap commit manager；
6. 实现 `teacher_forcing_dataset_pair` proof 与 clean GT paired `t=0` commit；
7. 为 `self_forcing_generated_pair` 建立 objective-gated schema；objective 未闭合时
   保持 hard fail，闭合后才实现 detached generated paired commit；
8. 把 `commit_source`/source proof写入 state manifest 和 receipt；
9. 实现 reset epoch/stale transaction rejection；
10. 编写 common、teacher 和 self-forcing failure tests；
11. 静态 review后再申请轻量测试授权。

拆分 command/applied histories、command identity、ACK transport、controller
transform/observation verifier 属于 future `deploy_applied_ack` implementation，
不在当前 Stage 0→Stage 1 最小范围内，必须另行 review 和授权。

不得先改低层 GDN kernel再补 schema；否则仍会把 mutation 和 content-time 问题藏在
裸 list 后面。

## 16. 当前 admission blockers

截至本设计对应的 H200 snapshot：

1. `CHUNK_ACTION_LAYOUT` rate/provenance 尚未闭合；
2. live cache 仍是可变 `list[10]`，没有 typed content-time；
3. read-only denoise 仍主要依赖 `save_kv_cache=False`；
4. teacher-forcing path 没有 dataset-row-identified transactional paired commit；
5. `SELF_FORCING_OBJECTIVE` 未闭合，generated pair不能获准 commit；
6. current deploy video ingest 不是 `deploy_applied_ack` paired commit；
7. server/controller 没有 `APPLIED_ACTION_ACK` transport；
8. RoboTwin adapter 只记录 `sent_action/post_state` client telemetry；
9. deploy 没有 canonical applied tensor/ref、transform digest 和 observation
   interval receipt；
10. current deploy action history 是 server-returned command；
11. AttnRes design/实现尚未解锁，且必须保持 forward-local；
12. source-aware verifier、failure receipts 和 independent review 尚未完成。

因此当前允许的唯一表述是：

```text
hybrid_cache_contract = designed_not_implemented
teacher_forcing_dataset_pair = designed_not_implemented_layout_blocked
self_forcing_generated_pair = blocked_on_self_forcing_objective
deploy_applied_ack = blocked_on_environment_ack_seam
deploy_action_evidence_mode = commanded_action_only
```

## 17. Review checklist

进入 Stage 1 前 reviewer 必须逐项确认：

- [ ] GDN 被准确描述为 full-history recurrent summary；
- [ ] softmax 被准确描述为 exactly-one-previous committed chunk；
- [ ] AttnRes 不进入 temporal cache；
- [ ] shortconv/tconv 与 attention state 分离；
- [ ] live typed state不直接下沉到 vendored operator；
- [ ] denoise mutation能由结构性机制检测；
- [ ] paired commit没有 video-only/action-only路径；
- [ ] 每个 commit恰好声明一种 `commit_source`；
- [ ] teacher commit 使用 dataset/layout/row proof且不要求 environment ACK；
- [ ] self-forcing commit受 closed objective gate约束并使用 detached generated pair；
- [ ] deploy commit 才使用 canonical applied tensor，绝不使用 command；
- [ ] 三种 source proof不能互相替代或混配；
- [ ] future deploy ACK inline/ref 都会读取 bytes并复算 digest；
- [ ] future deploy controller transform与observation interval已绑定；
- [ ] reset使用 episode epoch拒绝 stale ACK；
- [ ] diagnostic-only不会进入 science admission；
- [ ] failure code 和 receipt schema 已冻结；
- [ ] source SHA/行号已针对实现起点重新核验；
- [ ] 用户已对 Stage 1 实现和之后测试分别授权。
