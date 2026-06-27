# SANA-WM session results: fp32 reproduction fix + ladder verdict (H2 confirmed)

Experiments ran in the openwam staging repo (`/home/zch/wuji-openwam-dev`, where the
forced-MoT-joint SANA path reproduces); this document is the record of record in
sana-wam. Eval task: RoboTwin `adjust_bottle`, standard seeds, ≥30–100 closed-loop eps.

## 1. The win — fp32 linear-attention reproduction fix (11% → 37.5%)

**Symptom:** SANA-MoT in openwam was stuck at ~11% closed-loop while the prior record
claimed ~44%.

**Root cause:** the SANA dual-track linear attention ran the whole formula in **bf16**.
bf16's ~7-bit mantissa bleeds ~3% mean-relative error per layer into the QK^T / running-state
reductions, which **compounds across the 30-layer joint MoT stack** → degraded action output.
The fix was split across two branches and no single branch had both halves
(`renorm` on `feat/why_sana_is_bad`, `fp32 upcast` on `feat/sana-self-contained-deploy`).

**Fix** (`openwam .../dual_system/sana_linear_attn.py`): in `_expanded_linear_attn`,
`_chunked_linear_attn`, `_chunked_linear_attn_checkpointed` — upcast bf16/fp16 inputs and
the cumulative state to **fp32**, do the attention math in fp32, cast back; and set
`eps 1e-15 → 1e-8` (mirrors the upstream SANA use-site). fp32/fp64 callers untouched so the
math-equivalence tests still bind. Must be active in **both training and deploy**, plus the
`action_self_attn_weight: 0.5` renorm.

**Result:** fp32 deploy + re-eval `adjust_bottle` = **12/32 ≈ 37.5%** (vs no-fp32 ~11%).
Reproduction gap closed. *Retained-architecture fix — to be ported into sana-wam code (see §4).*

## 2. Improvement ladder — every cheap/medium rung closed

Target was 44% → 74%. Each rung tested with ≥30–100 closed-loop eps.

| Rung | Test | Verdict |
|---|---|---|
| **L0** receding-horizon + temporal ensemble | execute_horizon=6 vs greedy, matched seeds | **DROPPED** — 0/16 vs greedy 3/16; it *hurt* |
| **GATE-A** semantic-latent decodability (→ L3 RepViTok) | DINOv2/SigLIP vs LTX2 latent marginal action-R² | **killed L3** — semantic latent carries no extra action info |
| **L1** multi-task scale-up (8 tasks, variant=both) | starved 4k probe, then fair 12k retrain; eval adjust_bottle | **REFUTED** — fair 12k = 25% (n=48) < single-task 37.5%; coverage *dilutes* the focal task |
| **L2** gradient-flow IDM decodability | unfrozen 2B DiT video features vs proprio-only, action-MAE | **H2 CONFIRMED** — see §3 |

## 3. L2 gradient-flow IDM gate — H2 representation ceiling confirmed

Probe (`openwam sandbox/idm_l2_probe.py`, on the fp32 ckpt, adjust_bottle clean_50, 600
windows): regress the normalized 32×20 action chunk from the DiT's predicted-frame latents
(`finalize` (B,16,3,48,40), pooled + consecutive-frame deltas = the IDM transition signal).
Three arms, identical MLP head + split:

| arm | val action-MAE (norm [-1,1]) |
|---|---|
| C proprio-only (floor) | **0.0821** |
| A frozen backbone + MLP (static) | 0.0995 |
| B **unfrozen 2B DiT + MLP** (gradient-flow, 400 steps, lr 1e-5) | **0.0935** |

Even after unfreezing the full 2B DiT and flowing gradients (B's val-MAE *did* descend
0.13→0.094, so it learned), the video/WM representation **loses to proprio-only** and barely
beats frozen. Gradient flow does **not** make the representation action-decodable beyond
proprio → **route 1.2 (shared-trunk IDM finetune) is dead**, and the H2 representation
ceiling is confirmed via the strongest (gradient-flow) form of the test.

## 4. Bottom line + retained architecture

With **2B SANA + RoboTwin data**, the WM representation is the binding constraint. coupling /
data-coverage / IDM fixes are all exhausted. The only theoretical lever left (larger /
robot-pretrained backbone) is excluded by the **KEEP-SANA** constraint. **SANA-MoT WAM
ceiling ≈ 37–44%.**

**Retained architecture = the fp32-fixed, renorm-0.5 SANA-MoT joint (`joint_self_attn`),
backbone-unfrozen, `lambda_video=1.0`.** Code port into sana-wam:
- sana-wam's `src/sana_wam/model/ar/sana_linear_attn.py` has the **same 5 functions** as the
  openwam file but still `eps=1e-15` and **no fp32 upcast** (the same bf16 bug). The fp32 patch
  (§1) applies directly here. Verify which linear-attn module the sana-wam joint-MoT path
  actually invokes before patching, and re-run sana-wam's math-equivalence tests (fp64 path
  must stay bit-equivalent).
