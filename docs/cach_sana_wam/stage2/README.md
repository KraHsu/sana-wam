# CACH-SANA-WAM Stage 2 offline teacher-forcing subset

> 状态：`OFFLINE_TEST_OWNED_TEACHER_FORCING_SUBSET_PASSED /
> FULL_GATE_S2_BLOCKED / TRAINING_BLOCKED`
>
> 规范主机：`H200`
>
> 规范工作树：`/home/zch/workspace/sana-wam`

本目录只记录 Stage 2 的离线、test-owned、teacher-forcing 小尺寸随机模型合同。
它不是完整 Gate-S2 admission，也不是 capability/scientific 结果。Stage 3 未授权。

允许范围：

- 随机初始化 mini model 与合成 tensor/token；
- 单卡、短时 forward/backward/JVP 数值检查；
- CPU reference 和 GPU/Triton 对照；
- 测试本地或 `/tmp` 临时产物；
- Stage 2 contract、测试和只读证据文件。

仍禁止：

- 真实 2B 模型、VAE、text encoder、checkpoint 或 dataset/HDF5；
- optimizer step、训练、评测、model/data/formal capture、deployment 或 policy
  server；
- 正式 experiment/admission root；
- 修改 pinned `third_party/Sana` 或冻结 AFCC tree；
- 声称 504-step、formal、training 或 scientific 结果。

## 已验证的离线子集

Pinned Sana 的 cached/Triton GDN 不实现 `frame_valid_mask`。主仓在 vendor 边界
采用严格 prefix compaction：外部保持 fixed `K` 与 batch-uniform contiguous
prefix，padding 必须 exact zero，只把有效 prefix 交给 vendor，再把 video 与
bridge 恢复到 fixed `K`。该 seam 未修改 pinned Sana。

test-owned 三段合成轨迹的几何为 `L=8, K=3, r=8`：video valid lengths
`[3, 3, 2]`，action capacities `[16, 24, 24]`，valid action counts
`[16, 24, 16]`。video mini-GDN 使用 FP32 zero timestep，action mini-ActionDiT
使用 BF16 zero timestep。video/action staging 消费同一个 digest-bound synthetic
request；成功 receipt 只由测试本地 publisher 生成，失败 receipt 也只是
synthetic test-owned 证据，不是 production ledger。其独立保留副本位于
`/DATA/share/sana_cach_stage2_failure_evidence/stage2_failure_20260731_20260731T105154989919953Z`，
root mode 为 `0500`；它属于 retained read-only scoped nonformal artifact，不是
formal root。

独立 `cach.applied_action_ack.v1` verifier 已覆盖 inline 与 registered immutable
reference 的 stable read，并证明 `commanded != applied` 时返回 canonical applied
tensor。它没有接入 manager/server transport；deploy commit 仍 hard-disabled。
zero-init adapter parameter gradient 与 future-action perturbation 也在 mini/test
边界通过，但这不是 full-model JVP admission。

最终 scoped Stage 2 evidence-retention attempt 为 `75 passed, 0 failed,
0 skipped, 13 warnings in
5.63s`；CPU affected regression 为 `264 passed, 0 failed, 10 skipped,
1 deselected in 0.90s`；静态检查为 `AST_COMPILE_WHITESPACE_OK 13`。最终只读
evidence root 为
`/DATA/share/sana_cach_stage2_evidence/stage2_offline_subset_20260731_JKYEkuCw`
（mode `0500`）。详细结果、日志 SHA256 与范围见
`STAGE2_MINI_GDN_ADMISSION_20260731.md`。

第二次成功 evidence-retention attempt root
`/DATA/share/sana_cach_stage2_evidence/stage2_offline_subset_20260731_jRCAj1tM`
因其后 verifier/source pin 发生变更，已按 mode `0500` 只读保留并明确标为
superseded，不作为最终证据。

第一次 evidence-retention attempt root
`/DATA/share/sana_cach_stage2_evidence/stage2_offline_subset_20260731_pPSovwEo`
因 post-test metadata quoting failure 标记为 `CAPTURE_ABORTED`，已按 mode `0500`
只读保留，未被伪装或复用为成功 evidence。

## Full Gate-S2 阻塞项

以下内容尚未关闭，因此 full Gate-S2 必须保持 `blocked / not_claimed`：

- production architecture 与 production stager 接线；
- model-owned `NO_ACTION` 语义；
- committed-action identity 与 history summary；
- full-model gradient/JVP admission 的完整、可追溯 Gate-S2 证据；
- standalone ACK verifier 到 manager/server 的 transport、authority receipt、
  deploy commit 与 durable replay 集成；
- production paired-transaction `ABORTED` ledger 和 receipt publisher；
- source bundle 的 transitive runtime closure；
- full-model/random-shape、checkpoint/data、训练、评测与 formal root admission。

因此，此记录不授权 Stage 3、训练、评测、checkpoint、真实数据、正式 root 或
部署。standalone ACK verifier 的正向与 commanded/applied mismatch 结果不能替代
production deploy integration；deploy commit 继续 hard-disabled。
