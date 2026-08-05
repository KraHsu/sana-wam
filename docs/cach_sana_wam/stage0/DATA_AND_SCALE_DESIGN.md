# CACH-SANA-WAM Data and Scale Design

> 状态：`DRAFT_BLOCKED`
>
> 作用域：Stage 0 数据闭包与 scaling pilot 设计。
>
> 本文不授权训练、评测、数据改写或正式 split 发布。

## 1. 已核验事实

2026-07-31 在 H200 对
`/DATA/share/RoboTwin2.0/dataset` 做了只读盘点：

| 项目 | 结果 |
|---|---:|
| task 顶层目录 | 50 |
| `aloha-agilex_clean_50/data/episode*.hdf5` | 2,500 |
| `aloha-agilex_randomized_500/data/episode*.hdf5` | 25,000 |
| dataset 全树普通文件 | 110,852 |
| `du -sh` 快照 | 896G |

这说明 H200 上不只有单任务约 50 demos；但不能因此把全部
`randomized_500` 自动当训练集。它的生成 provenance、分布角色、成功筛选、
与 clean/closed-loop evaluation 的重叠关系尚未闭合。

只读抽查
`lift_pot/aloha-agilex_clean_50/data/episode0.hdf5`：

- `observation/*/rgb`、EEF、joint action 的第一维均为 114；
- EEF 原始字段为左右 `xyz+quat` 与 gripper，代码转换为 20D；
- 文件没有 timestamp、frame id、control frequency 或 dropped-frame 字段。

因此“一 raw video row 对应一 action row”由 shape 支持，但物理时间同率和无
dropped/duplicated step 尚没有 provenance 证明。`r=tc*video_stride` 当前必须
保持 blocked。

### 1.1 CACH-A v0 action/proprio 表示

draft candidate 已显式固定：

```text
action_mode: eef
action_dim: 20
delta_action: false
representation: absolute_eef_target_xyz_rot6d_gripper
order: left xyz, left rot6d, left gripper, right xyz, right rot6d, right gripper
```

这消除了 legacy loader 默认值对 conditioner/ACK canonical values 的控制，但
normalization stats 仍为 `null`，所以数据 admission 仍 blocked。Stage 1 还必须
从 episode-origin row 构造 layout-selected per-chunk proprio；不能把 legacy
clip-level `proprio` 或整个 future `proprio_seq` 当作已闭合输入。

## 2. 数据单位与身份

最终 manifest 的最小数据单位是一个 episode，不是目录名。每个 row 必须绑定：

```text
task
variant
robot
episode_id
hdf5_realpath
hdf5_sha256
hdf5_size_bytes
instruction_realpath + sha256
scene_info_realpath + sha256
seed_value
modality names/dtypes/shapes
camera order
raw row count
action representation and order
control/video rate provenance
split role
```

mtime、目录名和 `seed.txt` 单独都不能替代 content identity。正式 campaign
前需生成 canonical episode manifest；launcher 逐文件验证，不能在数据读取后才
发现漂移。

## 3. Rate closure

只有以下全部成立，layout 才能设 `r=tc*video_stride`：

1. generator/controller 的 immutable source 或 receipt 明确 video 与 action row
   同一个 control step；
2. 每个选中 episode 的所有相机、state、EEF/joint action 长度完全一致；
3. 没有 dropped/duplicated frame，或存在显式 timestamp 可证明并处理；
4. dataset wrapper 不做隐藏重采样；
5. deploy observation/ack 使用相同时间单位。

如果无法取得第 1 项，数据只能用于工程 diagnostic，不能用于
executed-action-conditioned 科学结论。禁止用抽查一个 episode 替代全 manifest
验证。

## 4. Split、去重与权重

正式 split 发布前必须：

- 以 `(task, variant, seed, scene/object identity)` 分组，而不是随机 window
  切分；
- 同一 episode 的所有 growing-history window 只能属于一个 split；
- 用 seed、scene/object identity 和内容 digest 检查 clean/randomized 跨 variant
  重叠；
- 固定 train/validation/closed-loop task 与 episode manifest；
- closed-loop seed 不得出现在训练数据或 scaling 决策中；
- 每任务权重在结果前固定，默认任务等权而非 window 数等权；
- prompt/instruction 去除逐帧 future motion trace，只保留 scene/goal；
- normalization stats 只由 train manifest 计算并单独固定 SHA。

不得根据 candidate 结果删除困难 task、episode、长度或视角。

## 5. 两级训练数据设计

### 5.1 Reduced mechanism campaign

`CACH-A/H/R` 的 G4 只允许回答其单一机制 delta。其数据设计必须：

- 使用同一个冻结 episode/row order manifest；
- reference/candidate 共享完全相同的数据顺序；
- 有独立 validation cohort；
- 使用唯一 4k 或其他预注册 endpoint；
- 报告 task/length 分层结果；
- 明确标记 `reduced_mechanism_only=true`。

在 episode manifest、rate closure、holdout 与 row-order 生成器完成前，具体 task、
variant 和 step budget保持未决；不得边跑边选。

### 5.2 Final full-horizon campaign

`CACH-SF/CACH-R` 的 final campaign 必须从 step 0 启动，不得从 reduced
checkpoint continuation。它还需独立固定：

- 训练 corpus 与有效 episode/token 数；
- task/variant mixture；
- curriculum；
- full step/token budget；
- checkpoint 保存频率和唯一 final endpoint；
- scaling pilot 的通过阈值；
- projected GPUh、host/CUDA memory、checkpoint bytes 与 `/DATA` reserve。

当前没有证据证明 2B random-init complete hybrid 能在 `clean_50` 或任意未审计
mixture 上收敛，所以 final full-horizon training 保持 blocked。

## 6. Scaling pilot

scaling pilot 必须在看到 final candidate 结果前注册，并且不产生 closed-loop
结论。至少记录三个从 step 0 的固定预算点，用于判断：

- train/validation video flow loss 是否持续下降；
- action loss、多步 rollout drift 是否同步改善；
- per-task 方差和长 episode 是否恶化；
- GDN state/anchor/router 数值是否稳定；
- walltime、显存、host memory、checkpoint bytes 的斜率。

预算点、停止阈值和 extrapolation 方法必须在 pilot 开始前写入新 authority。
pilot checkpoint 不得被挑选或继续训练成 final endpoint。

## 7. 容量 gate

launcher 使用字节级条件：

```text
projected_free_after >= 0.20 * total_bytes + safety_buffer_bytes
```

其中可用字节必须取启动时目标文件系统的
`statvfs(path).f_bavail * statvfs(path).f_frsize`，不得使用包含仅 root 可用 block
的 `f_bfree`。同时必须做 inode gate：

```text
projected_available_inodes_after =
    statvfs(path).f_favail - projected_new_inode_count
projected_available_inodes_after >= inode_safety_floor
```

`projected_new_inode_count` 覆盖 cache shards、双臂 checkpoints、logs、receipts、
failure inventory 和临时原子写文件；`inode_safety_floor`、字节
`safety_buffer_bytes` 及所有 projection 必须在 authority 中预注册。字节 gate 与
inode gate 是联取关系，任一未知或不满足均 fail-closed。

2026-07-31 快照中 `/DATA` 约 7.0T total、1.7T free、76% used。该值易变且只比
20% 保留线多有限 headroom。任何 run 前必须重新测量，并把 projected dataset
cache、双臂 checkpoint、logs、failure inventory 和并发写入计入。

不删除 AFCC 或其他冻结 evidence 腾空间。

## 8. 当前 admission blockers

1. control/video rate provenance receipt 缺失；
2. canonical episode manifest 与逐文件 SHA 缺失；
3. clean/randomized 的生成与评测角色未确认；
4. split/去重/holdout/task weighting 未冻结；
5. normalization stats source 未冻结；
6. scaling pilot 预算和阈值未冻结；
7. final GPUh/checkpoint 容量未冻结。
8. 现有 fixed-window loader 会生成 `start>0` rows；growing-history 路径又会生成
   nonzero clean prefix，尚无 episode-origin/no-clean-prefix sampler；
9. layout-derived per-chunk proprio selector、state timestamp proof 和
   model-facing future-state exclusion尚未实现。

blockers 非空时：

```text
reduced_train_authorized = false
full_horizon_train_authorized = false
scientific_eligible = false
```
