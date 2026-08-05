# CACH SANA-WAM Stage 0：ChunkActionLayout 与 Episode Bootstrap 设计

状态：`DRAFT / IMPLEMENTATION-BLOCKING DESIGN`

日期：`2026-07-31`

适用对象：`Causal Action-Conditioned Hybrid SANA-WAM (CACH SANA-WAM)`

目标实现文件：`src/sana_wam/model/action_chunk_layout.py`（尚未创建）

本文只冻结时间轴、layout、bootstrap 和失败语义，不代表已实现、已测试或已获得
训练/评测授权。本文所引用代码均来自 H200：

```text
host: H200
repository: /home/zch/workspace/sana-wam
git commit: 605f1c134b4c983ff80f8489c4bc8847036329e2
```

审计时下表文件相对该提交均为 clean。

## 1. 目的与非目标

### 1.1 本设计必须解决

1. 固定 RoboTwin row 中 action token 的 `t+1` 时间语义；
2. 固定 LTX2 causal VAE 的 latent 0 anchor 语义；
3. 用逐 row、逐 chunk 的 `ChunkActionLayout` 取代 fixed
   `action_tokens_per_chunk`；
4. 明确定义首 chunk、continuation chunk、partial tail、pad 和 Action RoPE；
5. 固定 `first_frame_pinned` episode bootstrap；
6. 区分静态 `LAYOUT_SPEC_SHA` 与动态 `LAYOUT_INSTANCE_DIGEST`；
7. 对缺失 timestamp/rate 证明、旧 observed-prefix 字段和任何 coverage
   gap/overlap 执行 fail-closed。

### 1.2 本设计不宣称

- action ownership interval 等于 LTX2 完整卷积 receptive field；
- 当前 RoboTwin HDF5 已提供足以做科学时间对齐的 timestamp；
- `K=3, tc=8, vs=1` 已自动成为最终 candidate 的已验证数据契约；
- partial tail 可以继续沿用当前 `T // K` 丢弃行为；
- 已有实现、测试、训练、评测或 closed-loop 结果。

## 2. H200 源码事实与证据

### 2.1 Source ledger

| H200 路径 | 审计符号/行 | SHA256 |
|---|---|---|
| `src/sana_wam/dataloader/robotwin_dataset.py` | `RoboTwinDataset.__init__` `:321-382`；`_read_raw_actions` `:948-952`；`_video_len_to_latent` `:978-982`；`_build_sample` `:1024-1043,1066-1120,1148-1181` | `a7db73364f260b693e376f16783576d383ad88c3ee9f8258135acd91e81c1445` |
| `src/sana_wam/model/video_backbone/sana/adapter.py` | `SanaVideoBackbone.get_native_temporal_contract` `:83-96` | `6ef2b6c21119b1357c8cab938e0351df428a71992ea0e92f25460b1b0a05d1cf` |
| `src/sana_wam/model/gdn_ar.py` | `DualSystemGDNARArchitecture.__init__` `:43-63`；`compute_loss` `:185-248,317-376,423-445` | `15a75585a0a4e33b7b83c65dd5d40333c1d5ad05c1862b21412ce302dbadd08c` |
| `src/sana_wam/deploy/gdn_ar_engine.py` | `GDNARInferenceEngine.__init__` `:51-106`；`_resolve_action_tokens_per_chunk` `:120-139`；`_step_with_obs_latent` `:165-245`；`reset` `:278-282` | `886815fc78a433f8813d159ae6acc59f03866c383c17d4a0cdb861952e4c539d` |
| `configs/train_sana_wm_gdn_ar.yaml` | predecessor 的 `K/tc/vs` 和旧 prefix 配置 `:42-67,78-121` | `9311107bb5f3593853bd13fc40116312b4dbf17f5540ba758b733a474865f406` |
| `configs/train_gdn_ar.yaml` | from-scratch predecessor 与旧 prefix 配置 `:37-64,72-118` | `9804f0368e5e3d6dcdb55601c37dc2ae9f80c4454c1bf5dea380329d6231efe5` |
| `configs/deploy_gdn_ar.yaml` | fixed-ATC deploy 说明 `:26-63` | `fe7341ce5d7d35c6a9b676af78c59cb4d096de2bb92ce3e393fec44396f62aec` |

行号和 SHA 只对上述 H200 commit/文件字节成立。后续任何 source 变化都必须重新生成
ledger，不能沿用本文行号假装仍然一致。

### 2.2 Dataset raw action 是 `t+1` 语义

`RoboTwinDataset._build_sample` 明确：

- raw window 的 frame/state/action 保持 raw rate
  （`robotwin_dataset.py:1034-1038`）；
- `sampled_video = raw_frames[range(0, num_frames, video_stride)]`
  （`:1086-1089`）；
- model action tensor 是
  `action_abs = raw_actions[1:self.num_frames]`
  （`:1101-1113`）；
- action mask 对 token `t` 检查 `(t + 1) < actual_valid_len`
  （`:1115-1120`）；
- 返回字段注释再次定义：“Action token `t` is the action into raw frame
  `t+1`”（`:1164-1171`）。

因此本文采用以下唯一离散索引约定：

```text
raw frame/state index:       f = 0, 1, ..., N_raw - 1
row action-token index:      t = 0, 1, ..., N_raw - 2
dataset value:               a[t] = raw_actions[t + 1]
temporal ownership:          a[t] is the action into raw frame f=t+1
```

这里的 “into” 是 dataset 的时间标签，不额外声称 `raw_actions[t+1]` 必然是低层
controller 在整个 `(t,t+1]` 区间保持不变的实际 applied command。EEF、joint、
absolute 和 delta 参数化可以改变数值含义，但不得改变 token 的 `t+1` 时间索引。
若要做 executed-action 科学结论，仍需另行满足 `APPLIED_ACTION_ACK` 契约。

#### 2.2.1 CACH-A v0 的数值表示

CACH-A v0 明确选择 20D absolute EEF target，而不是沿用 loader 的隐式默认：

```text
action_mode = eef
delta_action = false
action_representation = absolute_eef_target_xyz_rot6d_gripper
order =
  left_xyz[0:3], left_rot6d[3:9], left_gripper[9],
  right_xyz[10:13], right_rot6d[13:19], right_gripper[19]
```

这里固定的是归一化前的语义和维度顺序。normalization stats 仍未注册，所以本文
不把任何 model-facing 数值范围视为已闭合。action、proprio、conditioner input 和
未来 `APPLIED_ACTION_ACK` 必须引用同一个 representation/order digest；不得在
train/deploy 两端各自依赖 `RoboTwinDataset(delta_action=False)` 的默认参数。

### 2.3 LTX2 latent 0 是 causal anchor

三处代码共同支持该事实：

1. `SanaVideoBackbone.get_native_temporal_contract` 对带
   `temporal_compression` 的 LTX2 shim 返回 `(8, True)`
   （`adapter.py:83-96`）；
2. dataset 的 causal divisibility 注释明确“first frame is its own latent
   token”，剩余 video frames 才按 `temporal_compression` 分组
   （`robotwin_dataset.py:442-452`）；
3. `_video_len_to_latent` 对 causal temporal 使用
   `1 + (num_video_frames - 1) // temporal_compression`
   （`:978-982`）。

所以 latent 0 只对应 episode/window 的首个 sampled video frame，不存在进入该
frame 之前的 row action token。latent 0 的 action span 必须为空；不能复制 action
0、填数值零并把它当真实 action，或把 action 0 提前一个 latent。

### 2.4 当前 fixed-ATC 是待替换的错误近似

当前训练：

- `total_chunks = max(1, T // K)`，partial latent tail 被丢弃
  （`gdn_ar.py:230-238`）；
- `atc = max(1, Ta // total_chunks)`（`:240-248`）；
- chunk `c` 使用 `actions[:, c*atc:(c+1)*atc]`（`:317-325`）；
- Action RoPE 使用 `arange(c*atc, (c+1)*atc)`（`:374-377`）。

当前部署：

- `_resolve_action_tokens_per_chunk` 用 latent floor chunk 数和
  `total_actions // num_chunks`（`gdn_ar_engine.py:120-139`）；
- 每步仍用固定 `atc` 和 `arange(c*atc,(c+1)*atc)`
  （`:183-209`）。

train/deploy 同时采用同一个近似并不能证明物理时间对齐。特别是 latent 0 不消费
action，首 chunk 与 continuation chunk 本来就不等长。

### 2.5 当前 row schema 没有 timestamp/rate 证明

H200 审计范围内：

- `RoboTwinDataset.__init__` 提供的是 `video_stride`、VAE temporal contract 等
  几何参数（`robotwin_dataset.py:321-382`）；
- `_read_raw_actions` 仅按 HDF5 positional slice 读取
  （`:948-952`）；
- `_build_sample` 返回键完整列于 `:1155-1181`，其中没有 video timestamp、
  action timestamp、capture rate、controller rate 或 dropped/duplicated sample
  指示；
- 对 `src/sana_wam/dataloader` 与 `configs` 的
  `timestamp|time_stamp|action_rate|frame_rate|fps|hz` 只读检索没有找到可供该
  row layout 使用的时间基准字段。

数组长度匹配、`video_stride` 和 positional indexing 只能说明代码约定，不能证明
真实采集没有丢帧、重复帧、异步相机或 action/observation rate drift。该缺口是本文
定义的 P0 admission blocker，详见第 8 节。

## 3. 术语与时间轴

对一个具体 row/window 定义：

| 符号 | 含义 |
|---|---|
| `N_raw` | raw observation/state buffer 长度 |
| `A_valid` | `action_mask=True` 的真实 action token 数 |
| `vs` | dataset `video_stride`，单位为 raw frame steps / sampled video frame |
| `N_video` | sampled video frame 数 |
| `tc` | causal VAE temporal compression，单位为 sampled video frames / non-anchor latent |
| `L` | 有效 latent frame 数，包含 latent 0 anchor |
| `K` | 每个 AR chunk 的最大 latent frame 数 |
| `r` | 在通过等频 verifier 后，每个 non-anchor latent 的 action token 数 |
| `c` | zero-based chunk index |
| `[s_c,e_c)` | chunk `c` 的有效 latent 半开区间 |
| `[u_c,v_c)` | chunk `c` 的 action ownership 半开区间 |

所有区间均为 zero-based、左闭右开。`action_mask=True` 表示本文语义下的 valid；
若上游张量采用 `action_is_pad=True`，wrapper 必须在 canonicalization 时显式取反，
并记录 mask convention，禁止同名异义。v0 只接受“连续 valid prefix + 连续 pad
suffix”；valid mask 中间出现 hole 或在 pad 后重新出现 valid token，说明发生了
未建模的 drop/duplicate，必须失败而不是压缩重编号。

## 4. Timestamp-first 的 canonical ownership

### 4.1 Source of truth

真实数据的 source of truth 必须是 immutable row timebase manifest，而不是
`tc*vs` 算术。manifest 至少必须给出：

- time unit 和 clock/domain identity；
- 每个 raw observation 的单调 timestamp；
- 各 camera frame timestamp 以及同步/选择规则；
- 每个 action token 的 timestamp 或明确的 effective interval；
- action timestamp 到 “into raw frame `t+1`” 的对齐规则；
- dropped、duplicated、padded、interpolated sample 的显式标志；
- raw source、manifest、normalization/action-order schema 的 digest；
- `video_stride` 采样后保留的 raw frame index。

不得根据文件名、数组等长或 nominal simulator rate 补造 timestamp。

### 4.2 General timestamp assignment

令 sampled video frame `m` 的时间为 `V[m]`。对 causal VAE：

```text
latent 0: sampled video frame 0; action span is empty
latent j>=1 ownership boundary:
    left(j)  = V[(j-1) * tc]
    right(j) = V[j * tc]
```

action token 必须先由 manifest 解析出其 destination/effective timestamp `D[t]`。
latent `j>=1` 拥有且仅拥有：

```text
{ t | left(j) < D[t] <= right(j) }
```

边界闭开约定与 dataset 的 “action into destination frame” 一致：进入右端 sampled
frame 的 action 属于当前 latent，进入左端 sampled frame 的 action已经属于前一个
latent。若 controller manifest 定义的是 command issue time 而不是 destination
time，必须先用注册的 controller latency/effective-interval transform 得到 `D[t]`；
禁止在 layout 内隐式猜测 latency。

每个 valid action 必须恰好落入一个 latent span。未落入、落入多个、timestamp
非单调或边界同值导致歧义时，layout 构造失败。

### 4.3 等频化简的 admission 条件

只有 verifier 逐 row 证明以下条件后才可使用常数化简：

1. 一个 raw observation step 对应一个 action token；
2. action token `t` 的 destination 正是 raw frame `t+1`；
3. raw observation 与 action timebase 同步且严格单调；
4. sampled video 确实按 raw index `0,vs,2vs,...` 取得；
5. 区间内无 dropped/duplicated observation 或 action；
6. row 的有效终点落在 causal-VAE temporal grid 上；
7. `tc` 来自已 pin 的实际 VAE contract，不来自名称猜测。

此时才有：

```text
r = tc * vs
```

并且对 latent `j>=1`：

```text
action_span(j) = [(j-1) * r, j * r)
action_span(0) = [0, 0)
```

`r` 是 verifier 输出，不是用户可以绕过 manifest 直接填写的“信任我”参数。

## 5. ChunkActionLayout 算法

### 5.1 Chunk latent bounds

对 `L>=1`：

```text
C   = ceil(L / K)
s_c = c * K
e_c = min((c + 1) * K, L)
```

必须使用 ceiling chunk grid；禁止 `L // K` 丢弃 partial tail。若底层 tensor kernel
要求固定 `K`，`[e_c,(c+1)K)` 只能作为显式 latent pad，并由
`latent_valid_mask=False` 屏蔽。

### 5.2 等频情况下的 chunk action bounds

在第 4.3 节全部通过时：

```text
u_c = max(0, (s_c - 1) * r)
v_c = (e_c - 1) * r
chunk_action_span(c) = [u_c, v_c)
```

该式是 chunk 内各 latent action span 的并集，不是 VAE 完整卷积 receptive field。

#### 首个完整 chunk

当 `c=0, e_0=K`：

```text
latent span = [0, K)
action span = [0, (K-1)r)
action count = (K-1)r
```

少掉的一组 `r` 不是 bug；它对应没有先行动作的 latent 0 anchor。

#### 后续完整 chunk

当 `c>0, e_c-s_c=K`：

```text
action count = K*r
```

#### Partial tail

令 tail 的有效 latent 数 `q=e_c-s_c`，其中 `1<=q<K`：

```text
if c == 0:
    action count = max(0, q-1) * r
else:
    action count = q * r
```

tail tensor 可以 pad 到固定 shape，但：

- valid action interval 不得扩展；
- 不得复制最后一个 action 作为 valid label；
- action/latent pad 必须有独立显式 mask；
- Action RoPE cursor 只按 valid action 数推进；
- row 终点若不在 latent time grid 上且含真实、非 pad action，row 必须失败，不能
  通过 floor 或把真实 remainder 改名为 pad 来“修复”。

### 5.3 `K=3, tc=8, vs=1` 的核算例

H200 predecessor `configs/train_sana_wm_gdn_ar.yaml` 当前给出：

- `K=3`（`:42-45,66-67`）；
- `tc=8`、causal（`:59-63,85-89`）；
- `vs=1`（`:90-94`）。

这只是公式核算例；在 real-data timestamp verifier 通过前，不是正式 admission
结论。若等频成立，`r=8`：

| chunk | latent interval | action interval | valid action 数 |
|---|---:|---:|---:|
| bootstrap `c=0` | `[0,3)` | `[0,16)` | 16 |
| continuation `c=1` | `[3,6)` | `[16,40)` | 24 |
| continuation `c=2` | `[6,9)` | `[40,64)` | 24 |

例如 `L=8` 时，最后一块为 2-latent tail：

| chunk | latent interval | action interval | valid action 数 |
|---|---:|---:|---:|
| `c=0` | `[0,3)` | `[0,16)` | 16 |
| `c=1` | `[3,6)` | `[16,40)` | 24 |
| tail `c=2` | `[6,8)` | `[40,56)` | 16 |

三块并集正好覆盖 `[0,56)`，与 `(L-1)r=56` 一致。任何统一 22、24、40 或
`Ta//floor(L/K)` 的结果均不符合该 anchor-aware mapping。

### 5.4 Required immutable output schema

`ChunkActionLayout` 至少输出：

```text
LayoutInstance
  layout_spec_sha
  layout_instance_digest
  source_row_digest
  episode_id_digest
  row_start_raw_index         # CACH v0 production value must be exactly 0
  timebase_manifest_digest
  timestamp_verification_receipt_digest
  K, tc, vs
  equal_rate_proven
  r                         # only present when equal_rate_proven
  valid_raw_count
  valid_video_count
  valid_latent_count
  valid_action_count
  chunks[]
    chunk_id
    latent_start
    latent_end
    latent_valid_mask
    action_start
    action_end
    action_valid_mask
    latent_action_spans[]
    proprio_raw_index
    proprio_timestamp
    proprio_source_receipt_digest
    action_rope_start
    action_rope_end
    is_bootstrap
    is_partial_tail
    anchor_no_action_slot
  coverage_receipt
  padding_receipt
```

`latent_action_spans` 对每个有效 latent 都必须存在。latent 0 的 span 固定为空，
并设置 `anchor_no_action_slot=True`。这不是一个伪造 action token。

### 5.5 Coverage invariants

构造成功前必须同时证明：

```text
I1  latent chunks are ordered, disjoint, and union to [0,L)
I2  valid action spans are ordered, disjoint, and union to [0,A_valid)
I3  latent 0 has an empty action span
I4  every latent j>=1 has exactly one registered action span
I5  every non-pad action appears exactly once
I6  every pad action is marked pad and appears in no ownership proof
I7  no chunk reads action outside its [u_c,v_c)
I8  no valid tail is silently dropped
I9  timestamp-derived and simplified bounds are identical when equal_rate_proven
I10 all indices fit the registered action/video maximum lengths
I11 valid masks are contiguous prefixes; padding is a contiguous suffix
I12 CACH v0 production rows have row_start_raw_index == 0
I13 chunk 0 proprio_raw_index == 0
I14 continuation proprio is the terminal observed state after [0, action_start)
I15 proprio timestamp never exceeds the chunk prediction boundary
I16 model input contains only the selected per-chunk states, not future proprio_seq
```

任一 invariant 未证明，构造器必须抛出带 reason code 的错误；不得返回
“best effort” layout。现有 ordinary sliding-window sampler 的 `start>0` row
不能把中途 frame 冒充 episode bootstrap；若未来需要 continuation row，必须另立
带 reviewed history/bootstrap source 的 candidate revision。

## 6. Action-token-rate condition 与 Action RoPE

### 6.1 Latent-bin condition

video conditioner 将每个 latent action span 映射为一个 `[B,K,A]` 条件。v0 规则：

- latent 0 使用显式 model-owned `NO_ACTION` slot；
- non-anchor latent 使用该 span 最后一个 valid command（end-of-bin command）；
- EEF target/rot6d、joint target和 gripper 不做数值均值池化；
- empty/pad span 不得退化为数值全零 action，因为零可能是合法 command；
- learned reducer、轨迹积分或其他聚合均属于新的 candidate revision。

`NO_ACTION` slot 不进入 action head label、不占 action token index，也不推进 RoPE
cursor。

### 6.2 Action RoPE identity

Action RoPE 使用 episode 内累计 valid action-token identity：

```text
rope_offset(c) = number of valid action tokens owned by chunks [0,c)
rope_positions(c) = [rope_offset(c), ..., rope_offset(c)+n_c-1]
```

在严格等频、无 pad 情况下：

```text
rope_offset(c) = u_c
rope_positions(c) = arange(u_c, v_c)
```

因此：

- `K=3, r=8` 时，chunk 0 positions 为 `[0,16)`；
- chunk 1 positions 为 `[16,40)`；
- 不能使用当前 `arange(c*atc,(c+1)*atc)`；
- tail 只推进实际 valid count；
- pad token 不得拥有可见 RoPE identity；
- episode reset 将 action RoPE cursor 清零；
- train、deploy、self-forcing、ACK verifier 必须使用同一个 instance 中的
  `action_rope_start/end`，不能各自重算近似值。

若 timestamp verifier 发现 action rate 可变，RoPE 仍使用严格单调的 token ordinal；
物理 timestamp/boundary 保留在 layout metadata 中。将 RoPE 改为连续物理时间属于
新的、需单独注册的 candidate delta。

## 7. Episode bootstrap

### 7.1 v0 唯一合法配置

```yaml
episode_bootstrap: first_frame_pinned
observed_prefix_chunks: 0
```

语义固定为：

1. episode reset 后 video temporal cache 为空；
2. action commit cursor 与 Action RoPE cursor 均为 0；
3. chunk 0 的 latent 0 是真实首帧 anchor，始终 clean/pinned；
4. chunk 0 的其余有效 latent 是预测目标；
5. chunk 0 的 action targets 正是 layout 的 `[0,(K-1)r)`（一般情况使用
   timestamp-derived interval）；
6. chunk 0 不需要也不得读取一个虚构的“previous full chunk”；
7. chunk 0 commit 后，chunk 1 才能读取已提交的 past。

H200 当前训练循环已经在实现层部分表现出这种 bootstrap：`gdn_ar.py:317-365`
从 empty cache 监督 chunk 0，并把 latent 0 的 sigma 设为零；部署
`gdn_ar_engine.py:165-218` 也在 step 0 pin 首 latent。但旧 config/docs 同时仍声称
`ar_observed_prefix_chunks: 1`，所以不能把现状视为契约已经闭合。

### 7.2 旧 `ar_observed_prefix_chunks` fail-closed

新 CACH variant 的 config schema 禁止出现旧 key
`model.architecture.ar_observed_prefix_chunks`，包括值为 `0` 的情况。原因是：

- 当前 `DualSystemGDNARArchitecture.__init__` 会读取该字段
  （`gdn_ar.py:43-59`）；
- docstring 声称 leading prefix “ingested but NOT supervised”
  （`:199-205`）；
- 实际 loop 从 `for c in range(valid_chunks)` 开始监督所有 chunk
  （`:317-435`）；
- predecessor configs 仍设置为 `1`
  （`train_gdn_ar.yaml:41-46`；
  `train_sana_wm_gdn_ar.yaml:42-48`）。

只允许新字段 `observed_prefix_chunks: 0` 与
`episode_bootstrap: first_frame_pinned` 成对出现。未来若要完整 ingest chunk 0、
从 chunk 1 开始预测，必须建立新的 candidate revision，不能修改本 v0 的值。

### 7.3 旧 dataset clean-prefix fail-closed

当前 dataset 会返回：

- `num_clean_prefix_latent`（`robotwin_dataset.py:1091-1103,1164`）；
- `num_clean_prefix_actions`（`:1165-1171`）；
- growing-history 模式可能采样非零 chunk-aligned clean prefix
  （`:984-1008`）。

CACH v0 的模型、layout 与 deploy 接口均禁止读取这两个旧字段作为 visibility
控制。边界 wrapper 只允许两种已注册策略：

```text
strict_zero:
    both fields must exist as integer zero (or be absent by schema);
    any non-zero value -> fail

discard_and_audit:
    migration transform records original values and source-row digest;
    outputs a new transformed-row manifest with both values fixed to zero;
    model-facing batch does not contain the legacy fields;
    transform code SHA + manifest SHA must be registered before use
```

默认策略是 `strict_zero`。`discard_and_audit` 不是 runtime 静默忽略；没有 immutable
conversion receipt 时不得使用。任一非零值进入 video/action visibility、loss mask、
proprio boundary 或 cache ingest 都立即失败。

### 7.4 每 chunk 的 causal proprio boundary

`use_proprioception=true` 不允许把一个 clip-level `proprio` 重复用于全部 chunks，
也不允许把完整 `proprio_seq` 直接交给模型后由实现自行选择。对 chunk `c` 定义
`p_c` 为 prediction boundary 上最新、已观察到的 robot-state raw index：

```text
p_0 = 0
p_c = destination_state_index(action_start(c) - 1), c > 0
```

也就是说，continuation chunk 只能读取上一 committed action union
`[0, action_start(c))` 结束后得到的状态。在已经证明 positional `t+1` 映射和
等频条件的简式中：

```text
p_c = action_start(c) = (latent_start(c) - 1) * r, c > 0
```

例如 `K=3,r=8` 时三个 chunk 的 proprio raw indices 为 `0,16,40`。timestamp-first
模式必须由 manifest 的 action destination/effective time 和 state timestamp
解析同一 boundary；没有唯一 terminal observed state、timestamp 超过 prediction
boundary、或 action/state clock identity 不一致都失败。

offline teacher forcing 可以从 source row 读取完整 state sequence 来构造这些
selected values，但 model-facing batch 只能携带 layout 已固定的
`[num_chunks,state_dim]` selected proprio、对应 mask/index/timestamp 和 receipt
digest。完整 future `proprio_seq` 不得进入 architecture。deploy 则必须由该 step
真实 observation（以及需要时的 applied-action receipt）提供同一 boundary。

当前 legacy loader 在 `growing_history=false` 时把单一 `proprio` 固定在 window
frame 0，同时仍枚举 `start>0` windows；其 `proprio_seq` selector 又依赖 fixed
`K/tc` 近似。二者都不是本契约的实现。Stage 1 必须新增 episode-origin、
no-clean-prefix sampler 与 layout-derived selector；只丢弃两个 prefix 字段不会
修复已受 `cur_raw` 影响的 payload。

## 8. P0 admission blocker：缺失可信 timestamp/timebase

截至本文审计，当前 H200 dataset row 不能证明第 4.3 节的等频条件。因此：

```text
REAL_DATA_LAYOUT_ADMISSION = BLOCKED
reason_code = MISSING_VERIFIED_ROW_TIMEBASE
```

在 blocker 关闭前：

- 可以实现纯函数和 synthetic timestamp fixtures；
- 可以静态审查 schema、digest 和公式；
- 不得对真实 RoboTwin row 生成“verified” layout；
- 不得把 `r=tc*vs` 写进正式 receipt；
- 不得启动 CACH 训练、评测或 capture；
- 不得用 predecessor 的成功解析/数组长度代替时间基准证明。

关闭 blocker 的最小证据包：

1. source 数据格式/采集器的一手 timebase 说明；
2. row-level timestamp/rate sidecar 或可复现 extraction；
3. dropped/duplicated/sync verifier；
4. 多 camera 对齐策略；
5. action effective-time 语义；
6. 抽样与全量统计 receipt；
7. source、extractor、manifest 和 receipt SHA256；
8. 人工审阅通过的异常清单，异常 row 默认拒绝。

若原始 RoboTwin 文件根本没有 timestamp，必须取得可审计的 simulator/controller
固定步进 provenance，并用 episode-level shape/sequence verifier 证明没有缺步。
“通常是同频”不满足 admission。

## 9. Digest 契约

### 9.1 `LAYOUT_SPEC_SHA`

`LAYOUT_SPEC_SHA` 标识静态算法与 candidate 时间约定。它是 canonical JSON
（UTF-8、sorted keys、无浮点 NaN、明确 integer units）的 SHA256，至少覆盖：

- schema/version；
- implementation source SHA/commit；
- dataset `t+1` convention；
- causal-VAE identity、source digest、`tc` 与 anchor rule；
- `K`、`vs` 及其 resolved-config provenance；
- timestamp boundary/clock/effective-action policy；
- equal-rate verifier version；
- bootstrap 与 observed-prefix policy；
- per-chunk proprio boundary、state timestamp 与 selector policy；
- tail/pad/mask policy；
- latent-bin reducer 与 `NO_ACTION` rule；
- Action RoPE policy；
- action dimension/order/normalization schema digest；
- fail-closed reason-code table。

只要其中任何一项变化，就必须产生新的 spec SHA；禁止更新文件内容但沿用旧 SHA。

### 9.2 `LAYOUT_INSTANCE_DIGEST`

`LAYOUT_INSTANCE_DIGEST` 标识某个具体 row/chunk layout。它以
`LAYOUT_SPEC_SHA` 为父，并至少覆盖：

- immutable episode/source-row identity；
- raw window start/end 和各 valid counts；
- concrete timestamp/timebase manifest digest；
- video/action/pad mask digest；
- 每个 chunk 的 latent/action bounds；
- 每个 latent 的 action span；
- 每个 chunk 的 selected proprio index/timestamp/source receipt；
- RoPE start/end；
- bootstrap/tail flags；
- coverage/padding receipt digest。

它不等于 action value digest，也不能替代 `APPLIED_ACTION_ACK`。若需要把 action
值绑定到训练或执行证据，另记录 canonical action tensor digest，并让 receipt 同时
引用 layout instance digest 和 value digest。

### 9.3 Static spec 与 dynamic instance 的边界

```text
one candidate revision -> one registered LAYOUT_SPEC_SHA
one concrete row/window -> one LAYOUT_INSTANCE_DIGEST
one live deploy chunk   -> one concrete instance digest
```

不同 row 即使 `K/tc/vs` 相同也不能伪称拥有同一个 instance digest。train、deploy、
self-forcing 和 verifier 可以共享 spec，但必须逐实例核对具体 digest。

## 10. Failure reason codes

Stage 1 实现至少提供以下稳定 reason code：

| Reason code | 触发条件 |
|---|---|
| `MISSING_VERIFIED_ROW_TIMEBASE` | 缺 timestamp/rate provenance 或 verifier receipt |
| `NON_MONOTONIC_TIMESTAMP` | observation/action timestamp 非严格单调 |
| `AMBIGUOUS_ACTION_EFFECTIVE_TIME` | action issue/effective/destination 语义不明 |
| `DROPPED_OR_DUPLICATED_STEP` | 检出未注册的丢步或重复 |
| `VAE_CONTRACT_MISMATCH` | runtime VAE 与 spec 的 causal/tc/source 不一致 |
| `LAYOUT_GRID_MISMATCH` | row 有效终点未落在注册 temporal grid |
| `NONZERO_EPISODE_ROW_START` | CACH v0 production row 不是从 episode raw index 0 开始 |
| `ACTION_COVERAGE_GAP` | valid action 未被任何 span 覆盖 |
| `ACTION_COVERAGE_OVERLAP` | valid action 被多个 span 覆盖 |
| `SILENT_VALID_TAIL_DROP` | valid latent/action tail 被 floor 丢弃 |
| `PAD_MARKED_VALID` | pad token 进入 condition/loss/RoPE |
| `NON_PREFIX_VALID_MASK` | valid mask 存在内部 hole 或 pad 后再次出现 valid |
| `LEGACY_AR_PREFIX_KEY_PRESENT` | 新 variant config 出现 `ar_observed_prefix_chunks` |
| `LEGACY_CLEAN_PREFIX_NONZERO` | strict mode 收到非零 clean prefix |
| `UNREGISTERED_PREFIX_DISCARD` | clean-prefix 丢弃没有 conversion receipt |
| `BOOTSTRAP_CONTRACT_MISMATCH` | 不是 first-frame-pinned / observed-prefix=0 |
| `PROPRIO_BOUNDARY_MISMATCH` | selected state 不是上一 committed interval 的 terminal observed state |
| `FUTURE_PROPRIO_VISIBLE` | model-facing input 含 boundary 之后的 proprio/state |
| `LAYOUT_SPEC_SHA_MISMATCH` | static spec 不一致 |
| `LAYOUT_INSTANCE_DIGEST_MISMATCH` | concrete bounds/masks/digest 不一致 |
| `ACTION_ROPE_CURSOR_MISMATCH` | RoPE cursor 与累计 valid action ownership 不一致 |

错误必须携带 source-row digest、spec SHA 和安全的结构化 bounds；不得在错误日志中
写入凭据或未经脱敏的大体积原始数据。

## 11. Stage 1 实现边界

Stage 1 应优先在主仓新增：

```text
src/sana_wam/model/action_chunk_layout.py
tests/test_cach_action_chunk_layout.py
tests/test_cach_config_schema.py
tests/test_gdn_ar_observed_prefix_contract.py
```

不要为 layout 方便而修改 frozen `robotwin_dataset.py`。使用 model-facing
wrapper/metadata transform 接收 row，再产出 immutable layout。Stage 1 首批实现
应保持为无 CUDA、无模型权重、无网络、无副作用的纯函数；真实数据 admission
仍由第 8 节 blocker 控制。

## 12. Stage 1 测试清单（本轮未创建、未运行）

### 12.1 Dataset indexing 与 causal anchor

- [ ] 构造递增 raw action fixture，证明输出 token `t` 等于输入 index `t+1`；
- [ ] 证明 `action_mask[t]` 使用 destination frame `t+1` 的 validity；
- [ ] 对 causal VAE 验证 `L=1+(N_video-1)//tc` 的合法 grid；
- [ ] 证明 latent 0 span 为空且 `NO_ACTION` 不产生 action label；
- [ ] 证明 latent 1 的首 action 是 token 0，不是 token `r`。

### 12.2 首块、续块与 tail

- [ ] `K=3,tc=8,vs=1`：首块 16、续块 24；
- [ ] `L=8,K=3,r=8`：三块 `[0,16),[16,40),[40,56)`；
- [ ] `L<K` 的 bootstrap-only partial chunk 使用 `(L-1)r`；
- [ ] `L=1` 产生零 action、一个 anchor latent，不伪造 token；
- [ ] `L%K=0` 不产生虚假 tail；
- [ ] `L%K!=0` 使用 ceiling chunk count，latent union 精确为 `[0,L)`；
- [ ] kernel fixed-shape padding 不改变 valid bounds；
- [ ] 重复最后 raw action 的 padded buffer 仍全部被 mask，不进入 ownership。

### 12.3 Property/coverage

- [ ] 随机合法 `L,K,r` property test：action spans 单调、不交叠、无 gap；
- [ ] 所有 non-pad action 覆盖恰好一次；
- [ ] 所有 pad action 覆盖零次；
- [ ] valid-prefix/pad-suffix 之外的 mask 触发 `NON_PREFIX_VALID_MASK`；
- [ ] 任意人为删除一个 token 触发 `ACTION_COVERAGE_GAP`；
- [ ] 任意人为重叠一个 span 触发 `ACTION_COVERAGE_OVERLAP`；
- [ ] 真实 remainder 不能通过 floor 或改 pad 静默消失；
- [ ] timestamp-derived bounds 与已证明 equal-rate 简式逐项相等。

### 12.4 Timestamp admission

- [ ] missing manifest 触发 `MISSING_VERIFIED_ROW_TIMEBASE`；
- [ ] non-monotonic timestamp 失败；
- [ ] duplicate/drop 失败；
- [ ] action issue time 未解析为 effective/destination time 时失败；
- [ ] irregular 但合法 timestamp 能产生唯一 ownership；
- [ ] camera timebase/sync identity 不一致时失败；
- [ ] nominal rate 相同但实际 timestamp 缺步时不能设置 `equal_rate_proven`；
- [ ] synthetic-only bypass 带显式 test marker，无法进入 production config。

### 12.5 Action RoPE

- [ ] 首块 RoPE 从 0 开始且只有 valid action positions；
- [ ] continuation offset 等于此前所有 valid action 数；
- [ ] `NO_ACTION` 不推进 cursor；
- [ ] pad/tail 只按 valid count 推进；
- [ ] reset 后 cursor 精确回到 0；
- [ ] `c*atc` 与 layout offset 不同时测试必须捕获；
- [ ] train/deploy 对同一 instance 产生相同 positions 和 digest。

### 12.6 Bootstrap 与 legacy fields

- [ ] 唯一接受 `first_frame_pinned + observed_prefix_chunks=0`；
- [ ] `ar_observed_prefix_chunks` key 即使为 0 也 fail；
- [ ] `strict_zero` 接受零 clean-prefix、拒绝任一非零值；
- [ ] `discard_and_audit` 缺 transform/manifest receipt 时 fail；
- [ ] 合法 migration receipt 记录原值，model-facing batch 不再暴露旧字段；
- [ ] chunk 0 empty cache、latent 0 clean、其余有效 latent 为 target；
- [ ] chunk 0 不读取 prior chunk；
- [ ] prefix 字段不能改变 proprio boundary、loss mask 或 cache ingest。
- [ ] chunk 0 proprio index 为 0，续块等于上一 committed interval 终点；
- [ ] `K=3,r=8` 的 per-chunk indices 精确为 `0,16,40`；
- [ ] model-facing batch 不含完整 future `proprio_seq`；
- [ ] state timestamp 超过 prediction boundary 触发 `FUTURE_PROPRIO_VISIBLE`；

### 12.7 Digest 与 tamper evidence

- [ ] canonical serialization 在 key order 改变时 SHA 不变；
- [ ] spec 任一时间规则/source digest 改变时 `LAYOUT_SPEC_SHA` 改变；
- [ ] row bounds/mask/timestamp 任一改变时 instance digest 改变；
- [ ] 相同 concrete input 重建得到相同 instance digest；
- [ ] spec/instance/value digest 不被混称；
- [ ] tampered bounds、mask、RoPE 或 receipt 均 fail-closed；
- [ ] error receipt 含稳定 reason code 和父 spec SHA。

## 13. Stage 0 完成条件与当前结论

本文完成的是 implementation-before-code 的设计冻结草案。进入实现 review 前仍需：

1. 用户/主 agent 审阅本文公式与 fail-closed 选择；
2. 固定 timestamp/timebase evidence acquisition 方案；
3. 将最终 `K/tc/vs`、VAE source digest 和 action-order schema 写入 candidate spec；
4. 冻结 canonical JSON schema 与 reason-code enum；
5. 确认 clean-prefix migration 是只采用 `strict_zero`，还是批准带 immutable receipt
   的 `discard_and_audit`；
6. 另行获得编写 Stage 1 代码/测试的授权。

当前 admission 结论保持：

```text
ChunkActionLayout design: DRAFTED
real-row timebase proof: MISSING
real-data layout admission: BLOCKED
implementation: NOT STARTED BY THIS DOCUMENT
tests/training/evaluation/capture: NOT RUN
```
