# CACH-A4 STATE-CONDITIONED CAUSAL EXACT-ODD DELTA STREAM

状态：ARCHITECTURE_SOURCE_CPU_TEST_AND_ONE_IDLE_SINGLE_GPU_SYNTHETIC_EXECUTION_AUTHORIZED

规范环境：H200，/home/zch/workspace/sana-wam，2026-08-05。

## Authority and identity

用户 authority exact text 为“继续 A4”，UTF-8 长度 9，SHA256
c3631ea159561fb7d13154c592e16e019384f111b798b3380c7bcf4a4e94350f。

    architecture_id = CACH-A4-STATE-CONDITIONED-CAUSAL-EXACT-ODD-DELTA-STREAM-v1
    reference_arm = REF-NORMALIZED-COMMON
    candidate_arm = CACH-A4

A4 是 additive successor，不是 A3 rerun、AV2、正式训练或 admission。

## Frozen A3 evidence

A3 decision/card/config SHA256：

    ec99746eb5abbe6de29974d4fcf4e56a043fcbbf6df490264fe9fe4581b30979
    674073b39c1964bbc9c955022437390445dbb2bddfe42403b2c495f5a900be19
    8b3aa219826fedbe53e6aa33ffe7b357c7d6bf0ee4d334b116cb13a55037012f

A3 frozen root 为
/DATA/share/sana_cach_wam_nonformal_screens/cach_a3_orthogonal/06f5d09127f8/cach-a3-orthogonal-b2e55836da6fd5a51b568af613f74eef。
RESULT SHA256
81345a1c9b6e03d60dd587cb439ebdd9c2d60fc92a7120d80042c6a9e4dc78c3，
verdict OPERATOR_GO_COMMON_STABLE。RAW_EVIDENCE、RAW_METRICS、FREEZE_RECEIPT
SHA256 为 ee43f57d503a1db78bc26457efe048fbfe5c0a3be14ccaf79e6489b62363f9ef、
19ab6c95d3aed8915a887506425a2de3fd4c6d1e3f500478144586415c0cca9f、
69b72fd730b316509548e4dab198e4dd7d84565ee9f1d05a6cccd1e8a673c63c。
所有 A3 source/card/root/evidence 保持字节不变。

## Architecture

两臂共享相同 topology 的 action-blind common trunk、final parameter-free RMSNorm
(epsilon=1e-6) 和 common readout，fresh theta0 common bytes 相等。q_t 是 detached
parameter-free RMS-normalized common hidden，dim=64；raw action dim=20；delta
recurrence state d_t dim=64。

    d_(-1) = exact_zero
    u_t = 0.5*(g(q_t,a_t)-g(q_t,-a_t))                 # dim 64
    lambda_t = sigmoid(W_decay(q_t))                   # action-blind, dim 64
    d_t = lambda_t*d_(t-1) + m_t*W_write*u_t           # W_write 64->64
    r_t = W_out*d_t                                    # W_out 64->3, zero-init
    y_candidate_t = c_theta_t + typed_where(m_t,r_t,exact_zero)

W_decay 在 theta0 精确零初始化，因此 lambda_t(theta0)=0.5。state/delta 是
candidate-only scope，delta loss 不向 common 反传。对固定 q 和 history，
u(q,-a)=-u(q,a)、u(q,0)=0。future action 不影响 prefix；early action 通过 d recurrence
持续影响 future output。

单点 inactive 不调用 g 或 W_write，d_t 仍执行 action-blind decay，但 output 由 typed
where 精确 direct-bypass common。full no_action/seam_disabled 在 reducer/stream 前整体
bypass。reference 不读取 state/action/mask。

synthetic input、target 和 tensor bytes 原样复用 pinned A3 recipe，不引入 modified
state target。其冻结 target 已满足 raw_effect_t=0.5*raw_effect_(t-1)+teacher(a_t)，
因此可直接检验 causal action-delta recurrence。future-action perturbation 对 prefix 的
max-abs/max-rel leakage 必须 <=1e-6。以下三项 JVP 是独立 gate，不得混名：

- common_state_to_output_jvp_rms：扰动 detached common q/state，证明 stream 真读取 q；
- past_action_to_future_output_jvp_rms：扰动 early action，证明历史 action 影响 future；
- delta_state_to_future_output_jvp_rms：注入 d carry，证明 recurrence state 影响 future。

三项 JVP RMS 都必须 >1e-8。

## Screen and boundary

每臂 200 macrosteps；AdamW cosine lr 0.003→0.00003。只用正 common MSE 和正
half-delta MSE，独立 clip 1.0；禁止 loss subtraction、best-step、checkpoint 或
predecessor continuation。GO 沿用 A3 normalized-delta thresholds；旧 total gaps
report-only。verdict 只能是 OPERATOR_GO_CAUSAL_STREAM_COMMON_STABLE、
OPERATOR_GO_CAUSAL_STREAM_COMMON_BLOCKED、OPERATOR_STOP_CAUSAL_STREAM_WEAK、
OPERATOR_INCONCLUSIVE 或 INVALID_RUN。

A4 只有 decision/card/bridge/config/runner/test 六个 additive-only 文件；已存在即
fail-closed。本 authority 允许完成六文件 source、reduced synthetic CPU tests 和一次
idle single-GPU fresh-root synthetic run。execution envelope 必须动态绑定六个 SHA、
fresh nonce/root、physical GPU index/UUID 和 exact command。
无 token。任何 GO 不解锁/执行 AV2、正式训练/评测、真实数据、checkpoint 或 Stage 3。
