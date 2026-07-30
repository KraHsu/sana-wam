# SANA-WM project retrospective — all experiments, the bottleneck, the path forward

Last updated 2026-07-08. This is the consolidated record of the whole SANA-WM world-action-model
effort on RoboTwin: every experiment line, what it proved, where the bottleneck actually sits, and
what is worth doing next. Companion docs: `SESSION_FP32_AND_LADDER_VERDICT.md`,
`OPENWAM_VS_SANAWAM_DIFF.md`, `SANA_WAM_BREAK_44_PLATEAU.md`, `BACKBONE_H2_STRATEGY.md`,
`SANA_WAM_BACKBONE_HANDOFF.md`, `BACKBONE_RECOUPLE_OPTIONS.md`, `SANA_DREAM_ARCHITECTURE_PLAN.md`
(the non-AR dream-quality ladder — a separate track from the AR-graft saga in Phase 9 below).

> ## ⚠️ CORRECTION (2026-07-07): the "~37–44%" number below is INFLATED — true figure is ~26–33%
>
> A dedicated deploy-path audit (same methodology that found 4 real bugs on the AR line, §Phase 9)
> found that the non-AR SANA-MoT deploy path (`JointInferenceEngine`) hardcoded `seed=42` for EVERY
> denoising call, ever — every closed-loop episode in this line's ENTIRE history (every number in the
> TL;DR/Phase 7/8/§3/§5.A below) ran from the bit-identical starting noise tensor, while training used
> fresh i.i.d. noise every step. Fixed (default now draws real entropy, matching training; opt-in
> explicit seed preserved for debugging — see [[deploy-seed-fix-done]]/`openwam/deploy/joint_engine.py`).
> **Re-measured at standard protocol (n=100) under the fix:** the fp32-fix/`bsmntw` baseline checkpoint
> — the exact source of the historical "37.5%" figure — now scores **26/100 = 26.0%**; the `low_noise`
> checkpoint (Phase 9) scores **33/100 = 33.0%**, ABOVE the corrected baseline (flipping the earlier
> "low_noise hurts the non-AR line" verdict — see [[seed-fix-flips-flagship-number]]). **Everywhere
> below that says "~37–44%" should be read as this project's OLD, seed-bug-inflated estimate; the
> corrected figure for this exact recipe is ~26–33%.** Two things are NOT yet re-verified: (a) whether
> the specific later checkpoint behind the historical "44%" figure (the `action_self_attn_weight`
> token-dilution renorm, a further-tuned point past the plain fp32 fix) also drops when re-measured;
> (b) whether Wan's own 96.8% comparison number (§5.A) went through the same shared deploy code path
> and could be similarly affected — if Wan is unaffected while SANA was inflated, the true Wan-vs-SANA
> gap could be even LARGER than stated below, not smaller. The AR line has its own separate,
> structurally-analogous seed bug (`ar_engine.py`), not yet fixed — its 25–33% Phase-9 numbers have
> the same open question hanging over them. Treat every absolute percentage in this document as
> **directionally correct but numerically provisional** until the remaining re-verification is done.

> ## 📌 LATEST (2026-07-14): see **§Phase 10** — the AR-improvement campaign is EXHAUSTED
>
> Since the correction above: the AR seed bug WAS fixed (`ar-seed-fix-done`), putting AR `low_noise`
> at **~9–10%** and non-AR at **~26–33%** (n=100, seed-fixed). Then a full campaign to lift the AR
> line **all came back negative**: the graft family (causal + KV-cache, 0/20), and — under a
> covariate-shift reframe — DART (harmful, →1–5%), diverse-data (~5%), and DAgger (both aggressive
> and gentle fine-tunes → ~0%). A confound-controlled IDM gate showed the video rep's action grounding
> is a **model-agnostic +0.046 ceiling** (SANA = V-JEPA = Cosmos-Reason; pretraining adds zero), so
> "improve/swap the encoder" is refuted. **Net: the AR ~10% is a knife-edge fragile optimum that any
> fine-tuning destroys; fine-tuning-based AR improvement is not viable.** Open strategic fork
> (§10.6): pivot to the non-AR 26–33% line / bank the AR retrospective / try from-scratch joint
> training. See Phase 10 for the full ledger.

---

## 0. TL;DR — the two-layer answer

There are **two separate questions** that got conflated for most of the project:

1. **"Why 0%?"** → **SOLVED. It was the recipe, not SANA.** sana-wam's original architecture
   (GDN backbone + *gateable* cross-attn action bridge + frozen backbone + `lambda_video=0`) lets
   training drive the action↔video bridge to a content-free constant, so the policy is a
   proprio+text action head with the 2B world model bolted on and ignored. Switching to **forced
   MoT joint self-attention** (action and video tokens in one attention, no bypass) + training the
   backbone + `lambda_video=1` + an fp32 linear-attn fix → **0% → ~37–44%** on adjust_bottle. SANA
   is a perfectly good backbone (its features tie Wan2.2-5B on action content).

2. **"Why only ~37–44%, not the leaders' 74–89%?"** → **The remaining bottleneck is the SANA DiT
   backbone itself, NOT a backbone-agnostic tokenizer/representation ceiling.** This corrects an
   earlier over-claim. The decisive fact: in the *same* framework (openwam), on the *same* data, a
   **Wan2.2-TI2V-5B backbone reaches 96.8%** while SANA-Video-2B reaches ~37–44%. And both paths use
   a **Wan-family reconstruction VAE** (SANA's bundle ships `vae/Wan2.1_VAE.pth`; the Wan path uses
   `wan_vae`) — so **the tokenizer is essentially identical and CANNOT explain the 53-point gap.**
   RepWAM's "replace the reconstruction tokenizer" lever is a red herring for *our* setup.

   The gap is the DiT backbone, which differs on three axes, all intrinsic to SANA:
   **(1) attention** — linear-ReLU (SANA, an efficiency approximation; the source of the fp32
   error-compounding and token-dilution we fought) vs softmax/flash (Wan); **(2) pretraining domain**
   — camera-motion video (SANA) vs physical-AI/large (Wan2.2); **(3) scale** — 2B vs 5B.

   Reconciling the apparent paradox: an open-loop, single-step feature probe found Wan ≈ SANA
   (+0.13 marginal action-R² both). But that metric is open-loop; success is **closed-loop**. Both
   fit teacher-forced actions fine (~1.5 cm); the difference is the ability to **model dynamics over
   a rollout** — the *dream* quality and closed-loop covariate-shift correction. SANA's dream is
   degenerate (−4 dB vs copy-frame frozen / +1 dB unfrozen); a strong softmax robot-pretrained Wan
   dreams far better → its closed-loop correction works (DreamZero: performance ∝ video quality).

**So: the bottleneck is SANA's world-model dynamics-modeling capacity (linear-ReLU attention +
camera-domain pretraining + 2B scale), not the tokenizer and not a universal representation
ceiling.** Under the KEEP-SANA constraint this is the hard part — linear attention is *intrinsic* to
SANA (see §5).

---

## 1. Experiment ledger (chronological, by phase)

All RoboTwin closed-loop unless noted. "MAE" = normalized action MAE (open-loop, teacher-forced).

### Phase 1 — GDN cross-attn from scratch (the original architecture)
| Experiment | Config | Result | Conclusion |
|---|---|---|---|
| GDN cross-attn `lift_pot` | 2B GDN backbone from scratch, cross-attn bridge, tiny runs | 0% | Undertraining suspected; 2B from scratch on 50 demos |
| `lift_pot_v1` (fixed window) | 15k steps, 7×H200, fixed-window | 0/8…0/6 | **Not the model** — offline MAE 0.064, deploy verified correct |
| Root cause found | num_frames=49 → 48 action tokens, task is ~114 frames | — | **HORIZON MISMATCH**: model can only do first ~half; open-loop runs out, closed-loop re-gen is OOD |
| `lift_pot_grow` (growing history) | growing_history=true, 128 action tokens | (pivoted) | The principled fix for horizon; led into the covariate-shift phase |

### Phase 2 — closed-loop covariate-shift diagnosis
| Experiment | Result | Conclusion |
|---|---|---|
| Open-loop vs closed-loop gap | open-loop ~1.5 cm good; closed-loop 0% | The 0% is **covariate shift in observed inputs**, not action accuracy |
| **Delta (relative) actions** | 0/20 | Compounding is **visual/observation**, not action integration — per-step re-anchoring can't help |
| **R0 obs-prefix noise** (DART-style) | (swept σ) | Robustifying the action head to perturbed obs — insufficient alone |
| **Failure-mode diagnostic** (debug rollout) | visual proof | Arms hover at z≈0.9, never descend/grasp; drift +y, net motion > commanded = **textbook compounding drift** |

### Phase 3 — pretrained-backbone (frozen) tracks
| Experiment | Offline MAE | Closed-loop | Conclusion |
|---|---|---|---|
| Frozen linear-attn SANA (cross-attn bridge) | 0.068 (good) | 0/5 | Good fit, **open-loop drift**; not capacity |
| Pretrained **SANA-WM GDN** + LTX2 VAE swap (M0–M4, big eng. effort) | 0.20 | 0/7 | Stable closed-loop control but **imprecise fit** |
| Cross-attn bridge **on** SANA-WM GDN backbone | 0.084 | 0/7+ | GDN features ≈ linear for fitting; GDN-AR's 3× worse fit was the **chunked bridge**, not the backbone |
| **Robust negative** | {0.068, 0.08, 0.20} | **0% across every variant** | 0% not explained by backbone/bridge/fit/deploy-paradigm — systematic |

### Phase 4 — GDN-AR true autoregressive track
| Experiment | Result | Conclusion |
|---|---|---|
| GDN-AR (cached streaming, chunk-by-chunk) + many deploy fixes | — | Correct closed-loop paradigm; deploy bootstrap/proprio/cache-depth audited + fixed |
| GDN-AR unfrozen video + eff-batch-8 | 0/11 | Even the **DreamZero-aligned** config (AR + trainable WM + action couples to dream + KV real-obs feedback) doesn't move it → points at scale/data |

### Phase 5 — H1 (coupling) vs H2 (representation): the pivotal diagnostics
| Probe | Result | Conclusion |
|---|---|---|
| **Bridge-ablation** (shuffle / mean-pool / zero / oracle dream) | shuffle = mean = **1.00×**; oracle dream 1.2–1.3× *worse* | **H1 confirmed**: the cross-attn bridge collapsed to a **content-free constant**. The 2B backbone is functionally bypassed. |
| **IDM-decodability gate** (static) | bridge+proprio ≈ proprio (+0.00–0.01); even grasp | World-model rep carries **~0 marginal action info over proprio** → points at H2 |
| OOD-correction variant | both(bridge+prop) stays flat ~4 cm under proprio drift | Vision *does* carry a drift-immune ~4 cm estimate off-distribution → reopened proprio-dropout |
| **Official baseline** (randomized_500, unfrozen, lambda_video=1, test_num=100) | **0/48** | The apples-to-apples we never had — still 0% on official data + full protocol → genuinely backbone/recipe (H2) |
| **proprio-dropout + FixA** | 0/20 | Forced the action onto the bridge → it leaned more, but still got only the **content-free constant** → **H2 is the binding constraint, shown mechanistically** |
| multi-task + frozen + proprio-dropout | 0/20 | Even with bridge info present (+0.12 xyz, multi-task) and forcing, still 0% → exhausts cheap coupling fixes |

### Phase 6 — backbone / representation root-cause
| Probe / study | Result | Conclusion |
|---|---|---|
| **Camera/plucker** fed vs zeroed (A/B) | dream change −0.0% | Bad dream is **not** missing camera conditioning |
| **WM video-quality** probe | frozen dream −4 dB vs copy-frame; unfrozen +1 dB | Dream is degenerate; learnable but modest; **action is decoupled from it anyway** |
| **Domain-pretrain SANA** | dream flat / slightly worse while velocity-loss drops | Training the backbone harder does **not** lift the dream → capacity/architecture ceiling, not coverage |
| **Wan2.2-5B vs SANA features** (IDM) | **equal**: +0.131 both | SANA features are **not** weaker than the verified-working Wan → +0.13 is a metric/data ceiling, not a backbone gap; **the discriminator is recipe/deploy** |
| **RepWAM** (arXiv 2606.13674) | 89.3 from scratch; **VAE→RepViTok = +8.6** | The lever is the **tokenizer** (dynamics-aware latent), not coupling / pretrained weights → matches our H2 exactly |

### Phase 7 — the openwam smoking gun + SANA-MoT reproduction
| Finding | Result | Conclusion |
|---|---|---|
| **openwam diff** (33-agent) | prior SANA-in-openwam: **0 → 32% (retrain) → 44% (renorm)** | Same SANA, same data — only the **coupling topology** changed (gateable cross-attn → forced MoT joint). **SANA is not the problem.** |
| **SANA-MoT reproduced** | broke 0%, tracking ~33–44% | The winning config = SANA-Video + linear_relu kernel + **MoT joint self-attn** (non-AR), backbone trained, lambda_video=1 |

### Phase 8 — fp32 fix + the 44%→74% ladder (latest session)
| Rung | Result | Verdict |
|---|---|---|
| **fp32 linear-attn fix** | adjust_bottle fp32 upcast **+ `action_self_attn_weight:0.5` renorm** 11% → 37.5% | Reproduction gap closed (bf16 error compounding over 30 MoT layers). fp32 upcast alone does not reproduce 37.5% — the renorm is a co-requirement (the two halves lived on different branches pre-fix). *Retained — ported into sana-wam, commit `b2a31ad`.* |
| **L0** receding-horizon | 0/16 vs greedy 3/16 | DROPPED (hurt) |
| **GATE-A** semantic-latent decodability | no extra action info | killed **L3 RepViTok** (with the *frozen-semantic-feature* proxy) |
| **L1** multi-task (fair 12k retrain) | 25% (n=48) < 37.5% | REFUTED — coverage **dilutes** the focal task |
| **L2** gradient-flow IDM | unfrozen 2B DiT 0.094 MAE **>** proprio-only 0.082 | **H2 CONFIRMED** — gradient flow does not make the rep decodable → IDM re-coupling dead |

*Footnote: Phase 7's "tracking ~33–44%" and Phase 8's "stuck at ~11%" describe the same reproduction
lineage at different, later measurement points — `SESSION_FP32_AND_LADDER_VERDICT.md` explicitly
records that "SANA-MoT in openwam was stuck at ~11% closed-loop while the prior record claimed
~44%" before the fp32 fix. Read Phase 7's number as the earlier/looser estimate and Phase 8's
11%→37.5% as the more careful, later re-measurement and fix — not as two independent reproductions.*

### Phase 9 — AR-graft: the true-AR/streaming line breaks 0% (2026-06-29 → 2026-07-05)

*Continuation of Phase 4's GDN-AR track under its later name "AR-graft" (windowed-softmax
attention "grafts" added onto the SANA video/action blocks). Executed in the openwam staging
repo per the project's repo-workflow rule (experiments in openwam, record-of-record here).
**This is a separate, parallel architecture line from Phases 7–8's non-AR SANA-MoT line — its
results below do not revise the ~37–44% plateau (they are lower, ~25%, on a different
architecture).** Prior to this phase, the line's high-water mark was a single non-reproducing
14.3% (1/7) on a weak checkpoint, pre-dating the bug fixes below. **The phase ends with the first
confirmed non-zero, standard-protocol closed-loop result in this line's entire history** (25/100 =
25.0%, via the `low_noise` action-loss reweighting mechanism, §Result below) — reversing what had
looked, through most of this phase, like a second independent confirmation of H2.*

| Milestone | Result | Conclusion |
|---|---|---|
| **CUDA misaligned-address crash root-caused + fixed** (2026-06-29) | `compute-sanitizer --tool memcheck`: 106,090 errors, 100% "Invalid __global__ read" in the FFN's cuDNN NHWC bf16 1×1 conv (`GLUMBConvTemp`) — a channels-last **alignment** bug from AR's duplicated `[v_noisy++v_clean]` layout, not heap corruption or a bad attention kernel. Fixed with `.contiguous()` in `basic_modules.py` + reverted an interim pure-torch softmax back to SDPA. | Unblocked AR+graft training entirely (clean 8-GPU smoke; full run launched). Also surfaced a silent `NameError` that had been aborting sanitizer runs early, and a separate DeepSpeed LR-scheduler bug (8× per-step over-advance) — much of the earlier "infra friction" was these two bugs, not tooling flakiness. |
| **Deploy bootstrap-gate bug found + fixed** (2026-07-02) | 13-agent audit: the "predict from real obs, no imagined look-ahead" branch was gated on a static per-episode flag, not chunk index — every deployed chunk (not just chunk 0) took it, so the trained steady-state joint-denoise path had never been exercised in ANY closed-loop eval to date. Fixed (142/142 tests pass). Re-eval on the SAME 2 existing checkpoints (n=23 each): weak checkpoint **regressed 14.3%→0%**; single40 checkpoint stayed **0%**. | Bug was real and is now fixed, but was NOT the explanation for the offline-MAE-vs-closed-loop paradox. Making the model use its trained imagination-conditioned path removed the one flukey positive data point. AR-graft high-water mark is now treated as effectively 0%, not 14.3%. |
| **Sparsification-selector bug found (not fixed in vendored code)** (2026-07-03) | `graft_every_n` was a no-op whenever `graft_idx=None` — `_graft_for(i)` fell through to grafting ALL blocks regardless of N (verified: `graft_every_n=4` over `range(20)` returned `True` for all 20, not the intended 5). The bug lives in vendored `third_party/Sana` code and was flagged, not patched, this session (fixing it deferred as a separate decision). The run recorded as "5/20 sparsified" was actually a full 20/20 graft. The NEW action-side `joint_graft_every_n` selector (Phase 9's #1 mechanism, below) was written with correct branching from the start and is not affected. | Raw numbers from that run stand (best AR-graft dream at the time, −0.36dB vs copy; offline first-chunk MAE 0.0795) but closed-loop was still **0/20**. True sparsification remains untested. |
| **Diverse-data (`variant=both`) run** (2026-07-03) | clean_50+randomized_500 (550 eps), full 20-block graft, 12k-step budget. **NaN'd at step 10910**; salvaged step_10000. First-chunk MAE never converged (flat 0.336–0.345 across steps 2000–10000, vs. clean_50's ~0.08 at the same budget — 11× fewer gradient-batch exposures/episode). Closed-loop (n=20): **0/20**. | Undertrained, not a real test of "does diverse data help AR" — inconclusive by construction. Subsequent runs deliberately switched back to clean_50. |
| **LingBot-VA paper read in full** (arXiv 2601.21998v2, 2026-07-03) | Resolves the "<1e-3 action loss" confusion: LingBot-VA's action loss is uniform-SNR flow-matching MSE (same family as this project's) with a proven nonzero floor — no paper in that lineage (LingBot-VA, π0, GR00T N1, RDT-1B, Octo) reports a numeric converged flow-matching loss; **92% closed-loop success, not raw loss value, is the correct target.** Reveals LingBot-VA uses **joint SOFTMAX self-attention** (not linear) — echoes this project's own openwam-diff finding. Reveals a step-count comparison: LingBot's single-task RoboTwin ablation uses only 50 demos/3K steps (raw non-pretrained Wan2.2-5B still hits 80.6% there), vs. this project's AR-graft runs at 12,000–20,000 steps (4–7× more) landing at 0%. | Directly motivates the joint-graft (#1) experiment below as the single highest-precedent untried lever; independently rules out "undertrained on steps" as a confound, further supporting H2. |
| **Combined joint_graft(#1) + low_noise(#3)** (2026-07-03/04) | Two new mechanisms implemented (1358 tests pass): **#1** a genuine action-queries/video-KV softmax cross-attn graft per action block (LingBot-VA-inspired; also revealed the AR line had *never* applied any action-loss weighting, unlike every sibling architecture); **#3** min-SNR-gamma-style `low_noise` reweighting. Combined run **NaN'd 3/3 attempts** (OOM once; NaN at step 3643; NaN at step 9282/12000 despite mitigations). Salvaged step_8000: offline first-chunk MAE got **monotonically worse** through training (1.19→1.36→1.77→1.88), dream probe itself NaN'd, closed-loop **0/6** (stopped early). | Not a clean test of either mechanism — a training-stability failure, not evidence either one is harmless or helpful. Triggered isolation runs. |
| **Isolated #1-alone (`joint_graft` only), full 12000/12000 steps** (2026-07-04) | `action_loss_weighting=none` (no low_noise), clean_50, grad_accum=2. **Zero NaN through all 12000 steps** — the first "new mechanism" AR-graft run in the project's history to reach 100% of its step budget without crashing (confirms `low_noise`, not `joint_graft`, was the dominant destabilizer in the combined run). Offline first-chunk MAE still trended **worse** through training (0.51→0.48→0.73→0.72→1.06→1.39, steps 2000–12000 — milder than the combined run, same direction). Dream (step 12000): 12.47dB vs. copy 14.49dB, **Δ −2.02dB** (degenerate). Closed-loop (n=20, full run): **0/20**. | `joint_graft` alone — a genuine action↔video softmax coupling graft, directly mirroring the ONE mechanism that took the non-AR line 0%→44%, and this line's highest-precedent lever by prior expectation — tested cleanly in isolation, still lands at 0%. In isolation, this specific mechanism does not help this line. |
| **Isolated #3-alone (`low_noise` reweighting only), full 12000/12000 steps — BREAKS 0% (2026-07-04/05)** | `joint_graft=false` (no coupling change), `action_loss_weighting=low_noise` (gamma=2.0 clamp), clean_50, grad_accum=2. **Zero NaN through all 12000 steps.** Offline first-chunk MAE **converged normally**, unlike every other AR-graft run this session: 0.2127→0.1401→0.0886→0.0901→0.0818→**0.0820** (steps 2000–12000), plateauing around this project's historical "good" clean_50 baseline (~0.08). Dream (step 12000): 13.55dB vs. copy 14.49dB, Δ −0.94dB (still nominally degenerate, but the closest any AR-graft dream has come to crossing copy-frame). **Closed-loop: n=20 = 6/20 = 30.0%; confirmed at standard protocol n=100 = 25/100 = 25.0%.** | **The FIRST confirmed non-zero, non-fluke closed-loop result in the AR-graft/GDN-AR line's entire history** (10+ distinct configs since 2026-06-21, every prior one exactly 0%, including the higher-precedent `joint_graft` lever above). `low_noise` was this line's LOWEST-ranked proposal by prior expectation — the opposite of what precedent predicted. Reframes the whole phase: the AR line's persistent 0% was substantially a **training-stability/optimization problem** (unweighted flow-matching loss under-fitting the low-noise/near-clean timesteps that matter most for precise final positioning), not a genuine H2 representation ceiling. See §3.1 (revised) and §5.F for the corrected framing and next steps. |

**Summary — the AR line's closed-loop track record (chronological, exact numbers only):**

| Stage | Closed-loop result |
|---|---|
| GDN-AR v1 (depth-1, Phase 4) | 0/10 |
| AR-graft, pre-bootstrap-fix, weak ckpt | 14.3% (1/7) |
| AR-graft, pre-bootstrap-fix, single40 ckpt | 0/13 |
| AR-graft, post-bootstrap-fix, weak ckpt | 0/23 (regressed from 14.3%) |
| AR-graft, post-bootstrap-fix, single40 ckpt | 0/23 |
| AR-graft "sparsification" (actually full 20/20 graft) | 0/20 |
| AR-graft diverse-data (`variant=both`, undertrained) | 0/20 |
| AR joint-graft(#1)+low-noise(#3) combined | 0/6 (stopped early) |
| AR joint-graft(#1) alone, full 12000 steps, zero NaN | 0/20 |
| AR joint-graft(#1) + decoupled clean/noisy timesteps, zero NaN | 0/20 (offline MAE/dream dramatically better, closed-loop unchanged) |
| AR low-noise(#3) alone, buggy fixed-seed deploy | 6/20 (30.0%); confirmed 25/100 (25.0%) |
| **AR low-noise(#3) alone, SAME checkpoint, FIXED deploy (real noise diversity)** | **8/86 = 9.3% (stable trend, not chased to n=100 — see note)** |

Every variant across this line landed at exactly 0% — until `low_noise` in isolation, which broke it.
**But the magnitude of that break is smaller than first measured.** The AR deploy engine had its own
version of the fixed-seed bug found on the non-AR line (§ correction box at the top of this doc):
`ARInferenceEngine` rebuilt its noise generator from the same constant at every episode's `reset()`,
so every AR closed-loop episode in this line's entire history — including the 30.0%/25.0% numbers
above — replayed an identical noise trajectory chunk-for-chunk. Fixed (`ar-seed-fix-done`); re-run on
the exact same checkpoint: **8/86 = 9.3%**, a rock-stable trend that never approached the old
25-33% range (n=86 not 100 — the eval wrapper's own 5-hour polling cap, not the eval itself, ended the
run; the trend had been flat for 50+ episodes by then). **The qualitative finding still holds** —
`low_noise` is still the ONLY mechanism that has ever broken this line's exact-0% floor, and 9.3% is
still categorically non-zero — but the corrected magnitude (~0%→9-10%, not ~0%→25-33%) means the gap
to the non-AR line's own corrected ceiling (~26-33%, see the top-of-doc correction) is considerably
larger than post-breakthrough optimism suggested. `joint_graft`/coupling (with or without the
literature-motivated timestep fix) remains fully negative regardless of the seed bug, since both of
those results were exactly 0/20 either way.

---

### Phase 10 — AR covariate-shift campaign + encoder-gate exhaustion (2026-07-08 → 2026-07-14)

> Full method + reusable DAgger infrastructure + results table: **`docs/SANA_WAM_DAGGER_COVSHIFT_CAMPAIGN.md`**.

This phase (a) corrected the AR line's number under the seed-fix, (b) exhausted the
"improve the representation/encoder" hypothesis space with a rigorous decodability
gate, (c) reframed the true bottleneck as **covariate shift, not capacity**, and
(d) ran every covariate-shift lever on the AR line — all negative — ending with the
finding that the AR ~10% optimum is **knife-edge fragile**: any fine-tuning destroys it.

**10.0 Corrected seed-fixed baselines (n=100).** Non-AR SANA-MoT ≈ **26–33%**; AR
`low_noise` ≈ **9–10%** (`ar-seed-fix-flips-ar-number-too`). All Phase-10 evals use
the seed-fixed engine (default seed=None, ambient entropy per episode).

**10.1 AR graft family — fully exhausted.** After the plain segmented-dense graft
went negative (train sees non-causal "future" chunks), fixed the future-direction
leak (causal mask, found+fixed a dead-code bug that silently no-op'd it) AND the
past-direction asymmetry (`ARGraftTokenCache`, deploy-time cross-chunk history).
Both fixes, on the same 4k ckpt: **causal cache=0 → 0/20, cache=4 → 0/20**, dream
still −3.46 dB. The entire graft family (window/dense/segmented/causal/KV-cache) is
0–9%. (Two probe crashes here were root-caused to **disk 100% full**, not code;
freed ~700 GB.)

**10.2 The encoder/representation gate — a hard, model-agnostic +0.046 ceiling.**
Built a confound-controlled IDM decodability gate: predict the **action residual**
(action − proprio) **cross-task** (held-out tasks) from PCA-bottlenecked frozen
video features, later-half of the chunk, averaged over 4 splits. Results (grounding
= how much the video rep predicts the residual beyond a mean baseline):

| encoder | grounding |
|---|---|
| SANA (diffusion) | +0.046 |
| V-JEPA 2.1 ViT-g (JEPA self-supervision) | +0.046 |
| Cosmos-Reason2-2B (physical-reasoning VLM) | +0.046 |
| base SANA vs robot-trained SANA | +0.046 == +0.046 (training adds ZERO) |

**Three radically different pretraining objectives — and robot-training vs not —
all converge on +0.046.** So the video rep's transferable action grounding is an
**encoder-agnostic, objective-agnostic property of the RoboTwin observations**, not
of any backbone; and it is **low-rank (top ~4 PCA dirs = the whole signal),
nonlinearly encoded** (a linear head barely extracts it). Corollary: "pretrain
SANA harder", "swap to V-JEPA/Cosmos-Reason encoder", "use a JEPA objective" are
all refuted as levers — see `vjepa-equals-sana-grounding`, `cosmos-reason-equals-sana-grounding`,
`pretrain-adds-zero-grounding`, `ar-linear-vs-graft-idm-gate-red`. (Caveat: this is
an offline decodability proxy; the earlier v1 single-task gate was confounded by
proprio≈action + single-task memorization and was corrected.)

**10.3 The reframe — covariate shift, not capacity (the key pivot).** The task is
**low-information** (proprio-only offline action-MAE = **0.059**, near-solved), and
it IS learnable to 96.8% (Wan). Measured the flagship's **open-loop** action-MAE
(fed GT observations) = **0.077, flat across the chunk** — yet **closed-loop = 10%**.
Good in-distribution prediction + bad closed-loop = textbook **covariate shift**
(the policy drifts into off-demo states). Also found (code-level) the **non-AR
flagship has NO visual observation conditioning** — `use_first_frame_cond=False`, so
the video is a *blind* text+proprio→video dream, decoupled from the action; only
proprio conditions the action. I.e. **non-AR makes SANA meaningless** (≈ proprio+text
BC + a decorative dream), which is exactly why the AR/streaming line is the
SANA-native one. See `reframe-covariate-shift-not-capacity`, `nonar-flagship-blind-dream-proprio-only`.

**10.4 Every covariate-shift lever on the AR line — negative.**

| lever | closed-loop (vs AR baseline 10%) |
|---|---|
| baseline (untouched) | 10% |
| DART obs-perturb σ=0.05 / σ=0.1 | 4.8% / 1.0% (monotonically harmful) |
| diverse-data (variant=both, 550 eps; 40k NaN'd @27k, salvaged step_26000) | ~5% |
| DAgger single-round (60 expert-relabeled drift eps, aggressive fine-tune lr 1e-4) | ~0% |
| DAgger gentle fine-tune (lr 2e-5, 5k steps) | 0% at step_1000 AND step_5000 |

The DAgger pipeline itself is real and verified (on-policy capture of the baseline's
drift states → RoboTwin closed-loop expert relabels the correct action at each
reached state → co-train). The relabeling is exact (state restore max 0.0001 m).
But **the result is that any fine-tuning — even 1000 gentle steps — collapses the AR
policy to ~0%.** See `ar-dart-obs-perturb-implemented`, `ar-lownoise-diverse-data-partial-transfer`,
`dagger-poc-pipeline-built`.

**10.5 FINAL AR-campaign verdict.** The AR ~10% is a **knife-edge fragile local
optimum**: every improvement lever either fails to move it (graft family) or
*destroys* it (DART, diverse, DAgger — all fine-tune/data interventions). Training
loss stays low/converged while closed-loop collapses → the fine-tune leaves the
narrow deploy-competent basin. **Fine-tuning-based improvement of this AR line is
not viable.**

**10.6 Strategic fork (open, awaiting decision).**
1. **Pivot to the non-AR line (26–33%, 3× higher)** — accept that non-AR "wastes"
   SANA (blind dream) in exchange for the far higher measured success; do the
   engineering climb there. *(Recommended on the evidence.)*
2. **Write the AR retrospective and stop** — the AR line's levers are exhausted and
   its optimum is fragile; bank the negative result.
3. **From-scratch JOINT training of AR** (mix DAgger/on-policy data in from step 0,
   never fine-tune) — the only untested AR variant that structurally avoids the
   "fine-tuning breaks it" failure mode; higher cost, no guarantee.

---

## 2. What is firmly established

1. **SANA works as a WAM backbone.** 0→44% reproduced; SANA features tie Wan on action content.
2. **The 0% was the recipe**: gateable cross-attn bridge (collapses to a constant) + frozen
   backbone + `lambda_video=0`. Forced MoT joint coupling + trained backbone + fp32 fix → ~37–44%.
3. **The action head was decoupled from the world model** in the old architecture (bridge-ablation
   shuffle=mean=1.00×). The MoT joint coupling structurally removes that bypass.
4. **The ~37–44% plateau is the SANA backbone's dynamics-modeling capacity** — not a backbone-
   agnostic ceiling. The L2 gate showed *SANA's* unfrozen rep carries ~0 action info beyond proprio;
   but in the same framework Wan reaches 96.8% with the same Wan-family VAE, so Wan's rep evidently
   *does* support closed-loop control. The discriminator is closed-loop dream/dynamics quality, where
   SANA's linear-ReLU attention + camera-domain pretraining + 2B scale under-performs softmax Wan-5B.
   *(The L2 probe should be re-run on Wan features to confirm directly — Wan weights are not on this
   box yet.)*
5. **Closed-loop failure mode** is compounding covariate shift: ~1.5 cm open-loop precision degrades
   below grasp tolerance once the policy drives its own (OOD-rendered) rollout.
6. **Ruled out** as causes: camera/plucker conditioning, delta actions, dream quality (action is
   decoupled from it), action-head surgery (proprio-dropout / FixA / regression head), single-task
   more-data, multi-task coverage, IDM re-coupling, backbone choice (SANA vs Wan), mask/causality
   (GDN-AR streaming traced and confirmed causal — no temporal-leak bug).
7. **The AR-graft (true-autoregressive/streaming) line is architecturally distinct from the non-AR
   SANA-MoT line above, and its results do NOT revise the ~37–44% plateau (they top out lower, at
   ~25%, on a different architecture).** Phases 7–8 (non-AR, full-window joint self-attention) broke
   0% and reached ~37–44%. The AR-graft line (Phase 4's GDN-AR track, continued in Phase 9) is a
   separate cached/chunked streaming-generation architecture that landed at exactly 0% across every
   variant tried for months — including a genuine action-queries/video-KV softmax cross-attention
   graft (`joint_graft`) modeled directly on the SAME forced-joint-coupling mechanism that took the
   non-AR line from 0%→44% (tested cleanly in isolation: zero NaN, still closed-loop 0/20) — **until
   a `low_noise` action-loss reweighting fix (this line's LOWEST-precedent proposal, not the coupling
   fix) broke it to a confirmed 25/100 = 25.0%.** The corrected reading (see §3.1): the AR line's 0%
   was substantially a training-stability/optimization problem (the unweighted flow-matching loss
   under-fitting the low-noise/near-clean timesteps that matter most for precise final positioning),
   not a coupling problem and not a second, AR-specific H2 representation ceiling as it appeared to be
   for most of Phase 9. Coupling topology (`joint_graft`) remains not the lever for this line, in
   contrast to the non-AR line where it was decisive.

---

## 3. Current bottleneck (precise)

The bottleneck is **two layers**, and only the first is solved:

- **Layer 1 (SOLVED): coupling recipe.** Forced MoT joint self-attn + trained backbone +
  `lambda_video=1` + fp32 linear-attn. Gets ~37–44%.
- **Layer 2 (BINDING): the SANA DiT backbone's dynamics-modeling capacity.** Controlled evidence
  (same openwam framework, same data, same Wan-family VAE): **Wan2.2-TI2V-5B → 96.8%, SANA-Video-2B
  → ~37–44%.** Since the tokenizer is shared, the gap is the backbone. SANA's world model dreams
  manipulation dynamics poorly (−4 dB vs copy-frame), so closed-loop covariate-shift correction
  fails; Wan's strong dream supports it. Root-cause axes, all intrinsic to SANA: **linear-ReLU
  attention** (efficiency approximation of softmax), **camera-motion pretraining** (vs robot/physical-
  AI), and **2B vs 5B scale**.

Calibration on which axis dominates: Cosmos-Policy (softmax, robot-pretrained, **2B**) reaches 67% —
so 2B scale alone is *not* the cap; a good 2B prior gets ~67%. That isolates the SANA-specific
deficit to **linear-ReLU attention + camera-domain pretraining** (the 67→96.8 remainder is
scale/recipe). The earlier "reconstruction-VAE tokenizer is the ceiling" claim is **refuted** here:
both paths use a Wan VAE.

### 3.1 The AR/streaming line: 0% broken to ~25%, and it was not H2 (revised 2026-07-05)

Everything above in §3 describes the non-AR SANA-MoT line (Phases 7–8), solved at Layer 1 and
capacity-bound at Layer 2 (~37–44%). The AR-graft / true-autoregressive line (Phase 4, continued as
Phase 9) is a **separate architecture** — cached/chunked streaming generation rather than full-window
joint generation. For most of Phase 9 its 0% looked like a second, AR-specific instance of H2 (every
lever tried, including the highest-precedent coupling fix, landed at exactly 0%). **That reading did
not survive the last experiment of the phase:** `low_noise` action-loss reweighting alone — with NO
change to coupling topology — broke the line to a confirmed 25/100 = 25.0% (§ Phase 9 table). This
was this line's *lowest*-precedent proposal, tried last, after coupling (`joint_graft`, highest
precedent) had already failed in isolation.

**Revised diagnosis:** the AR line's persistent 0% was substantially a **training-stability /
optimization problem**, not a representation ceiling. The unweighted flow-matching loss
(`action_loss_weighting=none`, this line's historical default — a real, previously-undiscovered gap
vs. every sibling architecture, which defaults to the `bsmntw` weighting) under-fits the low-noise /
near-clean timesteps that matter most for precise final positioning; every other AR-graft variant this
session (`joint_graft` alone, the combined run, the diverse-data run) shows the *same* offline-MAE
symptom — flat or monotonically worsening MAE despite descending training loss — which `low_noise`
alone resolved (MAE converged normally to ~0.08, matching this project's historical "good" clean_50
baseline). What had looked like a structural, architecture-level ceiling (mirroring the non-AR line's
H2) was, for this line, a fixable training-recipe gap.

**What this does *not* settle:** `joint_graft` (coupling) remains not the lever for the AR line — it
was tested cleanly in isolation and still landed at 0/20, in contrast to the non-AR line where forced
joint coupling was decisive. The dream is also still measured degenerate at every AR-graft checkpoint,
including the 25%-achieving one (13.55dB vs. copy 14.49dB, Δ−0.94dB — the closest yet, but still below
copy-frame). So a real, if smaller, gap to the non-AR line's ~37–44% remains, and it is not yet known
whether it is capacity/architecture (H2-flavored) or a further recipe gap (more `low_noise`-style
fixes still undiscovered). Next steps in §5.F.

Two items from the diagnostic chain that produced the (revised) breakthrough, worth keeping as
context:
- The dream-architecture plan's own proven **video-only** softmax dream-quality fix (L1: 6.66dB→
  12.85dB, later ported into the non-AR MoT stack at L4/L6 with a +34% vs. 14% robustness result) has
  still never been ported into the AR/streaming line's chunked video self-attention and tested at
  long-horizon closed-loop — a different, untried mechanism from `joint_graft`/`low_noise` (see §5.F).
- Separately, `OPENWAM_VS_SANAWAM_DIFF.md`'s own #3 top-suspect finding — a content-time-vs-index-time
  drift in **sana-wam's own** `GDNARInferenceEngine` (cache index advances by a fixed per-chunk
  counter (+K=3) while the real observation window advances by a different, larger stride (~atc≈22
  raw ≈ 2.75 latent frames), so ingested observations become progressively mistimed across an
  episode) — is a candidate deploy-correctness bug specific to this repo's native GDN-AR engine,
  distinct from openwam's `ar_engine.py` (which got the separate bootstrap-gate fix in Phase 9). It is
  unclear from the audit trail whether this drift was ever fixed; still open technical debt, though
  now lower-priority given the training-recipe explanation above accounts for most of the observed 0%.

---

## 4. The KEEP-SANA constraint and what it forbids

Firm project constraint: **SANA stays the backbone.** The bottleneck (Layer 2) is now localized to
SANA's own DiT — its **linear-ReLU attention + camera-domain pretraining**. The tension: **linear
attention is intrinsic to SANA** (it *is* the efficient linear-attention video model), so "keep
SANA" and "use softmax" pull against each other. Off the table under KEEP-SANA: swapping to a
robot-pretrained softmax Wan/Cosmos backbone (what the 67–96.8% configs do) and scaling to 5B/14B +
internet-scale robot pretraining. **The tokenizer is NOT a lever** — both paths already use a Wan
VAE, so replacing it is refuted as the discriminator.

---

## 5. Future directions (ranked)

> **ACTIVE OBJECTIVE (2026-06-27): make SANA learn to dream manipulation dynamics.** The proven
> bottleneck is SANA's degenerate dream (6.66 dB vs Wan's 14.35 dB on the same recipe/data/VAE).
> Closing that gap is THE goal. **Compute is not a constraint** — expensive options (scale SANA up,
> large-scale robot-video pretraining, distill Wan's dream into SANA, hybrid softmax attention) are
> all in scope. A large parallel architecture-exploration workflow is mapping the option space; its
> ranked output lands in `docs/SANA_DREAM_ARCHITECTURE_PLAN.md`.

### A. Confirm the mechanism (cheap — do this first)
Run the **Wan-vs-SANA dream-quality head-to-head** (same data, same VAE: roll out predicted future
video, PSNR vs GT future, vs copy-frame) and re-run the **L2 decodability gate on Wan features**. If
Wan dreams well / its rep decodes actions and SANA's doesn't → the backbone-capacity diagnosis is
nailed and pinpoints which axis (attention vs pretraining) to attack.

**HEAD-TO-HEAD DONE (2026-06-27, `sandbox/dream_quality_probe.py`) — mechanism proven.** Same MoT
(`joint_self_attn`) recipe, same multi-task RoboTwin data, same Wan-VAE family, identical probe
(adjust_bottle future-frame PSNR, n=16, decoded to pixels):

| backbone (same recipe) | closed-loop | dream PSNR | copy-last-frame | Δ (dream − copy) |
|---|---|---|---|---|
| **Wan2.2-TI2V-5B** (`RoboTwin_DualSystem_JointSelfAttention`) | **96.8%** | **14.35 dB** | 12.62 dB | **+1.73 (beats copy)** |
| **SANA-Video-2B fp32** (`sana_mot_fp32_adjbottle`) | **37.5%** | **6.66 dB** | 12.58 dB | **−5.92 (degenerate)** |

The ~7.7 dB dream gap tracks the ~59 pt success gap. **Wan predicts the future (beats copy-frame);
SANA predicts a blob (far below copy-frame), despite the *same* coupling, data, and VAE family.** So
the discriminator is unambiguously the **SANA backbone's inability to dream manipulation dynamics** —
not the tokenizer, not the coupling, not the action head. (DreamZero: performance ∝ video quality.)
SANA still reaches 37.5% because its action head rides observed latents + proprio; it can't go higher
because a degenerate dream gives no closed-loop self-correction signal. *Note: SANA ships Wan2.1 VAE,
the Wan path uses Wan2.2 VAE — same family, both reconstruction VAEs; PSNR is in pixel space so the
comparison is VAE-fair.*

### B. Make SANA's world model dream manipulation dynamics — domain-pretrain (least-conclusive, keeps SANA)
The localized cause is a weak dream. The one direct attempt (domain-pretrain SANA on robot video)
ran only ~0.6 epoch and showed the dream flat — **inconclusive, not a real test**. A proper,
longer video-pretraining stage on robot manipulation video (lambda_action=0, many epochs) is the
cheapest experiment that could move the dream while keeping SANA. De-risk by tracking the
dream-quality delta (PSNR vs GT future) across the pretrain; continue only if it breaks past
copy-frame and keeps improving.

### C. Add softmax capacity to SANA without abandoning it
SANA-WM's GDN variant already interleaves **5 softmax / 15 GDN blocks** — precedent that SANA can
carry softmax blocks. Injecting/up-weighting softmax attention (especially over the temporal axis
where dynamics live) is a middle path: keep the SANA backbone + weights, add the expressivity the
dream needs. Speculative; needs a small ablation. This is the most direct attack on the "linear
attention is the deficit" hypothesis while nominally keeping SANA.

### D. Bank the durable result and write it up
The fp32-fixed SANA-MoT at ~37–44% on RoboTwin, plus the full diagnostic chain (0% was the coupling
recipe; the ~44% plateau is the SANA backbone's dynamics capacity, NOT the tokenizer — proven by
Wan 96.8% on the same VAE), is a clean, well-evidenced contribution and a publishable
negative-with-mechanism result. Code retained in sana-wam (`ar/sana_linear_attn.py`, commit
`b2a31ad`); narrative in `SESSION_FP32_AND_LADDER_VERDICT.md` + this doc.

### C. Relax a constraint (needs user decision)
If the goal shifts from "prove SANA can be a competitive WAM backbone" to "get a high RoboTwin
number", the highest-expected-value move is the one we keep excluding: a robot-pretrained Wan/Cosmos
backbone (raw Wan2.2-5B finetune is already ~80% Easy @ 50 demos). This **reframes the thesis** but
is the field's proven recipe. Only on the table if KEEP-SANA is loosened.

### E. Relax a constraint (needs user decision)
If the goal shifts from "prove SANA can be a competitive WAM backbone" to "get a high RoboTwin
number", the highest-EV move is the one excluded by KEEP-SANA: a robot-pretrained softmax Wan/Cosmos
backbone (the user already gets **96.8% with Wan**; raw Wan2.2-5B finetune ~80% Easy @ 50 demos).
Reframes the thesis but is the field's proven recipe.

### F. AR-graft (streaming) line — `low_noise` broke 0%, next steps (revised 2026-07-08)

**Confirmed positive result, magnitude revised down 2026-07-08:** `low_noise` action-loss reweighting
alone (no `joint_graft`) broke the AR-graft line's entire 0% history. Originally confirmed 25/100 =
25.0% at standard protocol; **after fixing the AR deploy engine's own fixed-seed bug (§ top-of-doc
correction box, [[ar-seed-fix-done]]), the same checkpoint re-measures at 8/86 = 9.3%** (stable
trend, [[ar-seed-fix-flips-ar-number-too]]). The qualitative finding is unchanged — `low_noise` is
still the only mechanism that has ever broken this line off exact 0% — but every number below from
before 2026-07-08 should be read as measured under the old, noise-diversity-suppressing bug; the
corrected ceiling for this line is **~9-10%, not ~25-33%.** `joint_graft` (coupling) remains negative
in isolation (0/20, unaffected by the seed bug either way) and should not be re-attempted alone.

**Ranked next steps (updated 2026-07-06 with two new results):**
1. **[PARTIALLY TESTED, 2026-07-06] Does `low_noise` transfer beyond this one checkpoint/task?**
   Re-ran on `variant=both` (550 eps) at a 2x step budget (24000 vs. 12000) — still clearly
   **undertrained** (offline MAE plateaus 0.32–0.41 vs. single-task's ~0.08; dream Δ−8.46dB, the
   worst measured this phase) since 2x is far short of the ~130K-step full-parity budget. Despite
   that, closed-loop = **3/20 = 15.0% — categorically non-zero**, unlike the pre-`low_noise`-fix
   diverse-data run at the same undertraining class, which was an exact 0/20
   ([[ar-diverse-data-undertrained-result]]). Reading: `low_noise`'s benefit is directionally
   transferable, but this run does NOT establish whether a properly-converged diverse-data run would
   match, exceed, or fall short of the single-task 25% — the step-budget question remains open
   ([[ar-lownoise-diverse-data-partial-transfer]]).
2. **[TESTED, NEGATIVE, 2026-07-08] `joint_graft` + decoupled clean/noisy-copy timesteps** (a
   literature-motivated fix, DiT4DiT/MaskWAM/PFD, for the "dense/joint attention over duplicated
   content finds a shortcut" failure mode — see [[ar-decoupled-clean-noisy-timesteps-done]]).
   Implemented, tested, independently verified clean. Result: offline metrics improved
   **dramatically** (first-chunk MAE 1.39→**0.20**; dream Δ−2.02dB→**Δ−1.45dB**, the second-best
   AR-graft dream measured this session) — but **closed-loop stayed at exactly 0/20, unchanged from
   `joint_graft`-alone.** See [[ar-decoupled-timesteps-result-negative]]. This is a stark instance of
   this project's recurring "offline metrics don't predict closed-loop success" pattern: the graft's
   training dynamics were genuinely repaired, and it still contributes nothing end-to-end.
   **Conclusion: `joint_graft`/coupling now looks close to fully exhausted as a lever for the AR
   line, even in its best-executed form. `low_noise` (pure loss-reweighting, zero coupling change)
   remains the ONLY confirmed positive lever for this line.** (Operational note: this run needed
   `batch_size=1, grad_accum=2` on a 4-GPU slice — `grad_accum=8` to "preserve the usual effective
   batch=32" caused genuine loss divergence; see [[grad-accum-instability-lesson]].)
3. **[TESTED, now CONFIRMED POSITIVE after the seed fix, 2026-07-07/08] Does `low_noise` help the
   non-AR SANA-MoT line too?** Yes — see §G item 1 below: initially measured negative (28.0% vs.
   `bsmntw`'s 37.5%) under the buggy fixed-seed deploy, but once fixed, `low_noise` (33.0%) beats
   `bsmntw` (26.0%). `low_noise` is now the best confirmed loss-weighting choice on BOTH lines.
4. **[TESTED, NEGATIVE, 2026-07-05] Port the *dream-quality* fix (L1's video-only softmax graft,
   6.66dB→12.85dB, validated at scale in the non-AR MoT stack) into the AR line.** Confirmed this is
   a config-only change (`additional_flash_attn=flash`, dropping `flash_attn_window_count`; no new
   code needed — `pipeline_builder.py` passes these keys through unchanged). Result on top of the
   confirmed-good `low_noise` recipe: **made things WORSE, not better** — dream Δ−3.98dB (vs.
   `window_flash`+`low_noise`'s −0.94dB) and closed-loop **1/20 = 5.0%** (vs. confirmed 25%).
   Plausible mechanism ([[ar-denseflash-lownoise-negative]]): `window_flash`'s temporal windows
   incidentally never cross the AR line's duplicated `[v_noisy++v_clean]` modality boundary (window
   count and clip length happen to align), so switching to dense (unwindowed) attention lets the
   graft mix noisy and clean tokens in one attention pool — a failure mode specific to the AR line's
   duplicated-sequence training scheme that the non-AR line, which has no such duplication, does not
   share. **Do not use dense `flash` on the AR line.** `window_flash`+`low_noise` remains the best
   confirmed AR-graft config (~9-10% post-seed-fix, see above).
5. Revisit `OPENWAM_VS_SANAWAM_DIFF.md`'s #3 finding (sana-wam's own `GDNARInferenceEngine`
   content-time-vs-index-time drift, §3.1) — lower priority now that the training-recipe explanation
   accounts for most of the line's historical 0%, but still open technical debt if AR investment
   continues. **Not yet attempted.**
6. **[TESTED, NEGATIVE, 2026-07-08] Segmented dense video graft** — full/dense attention independently
   within each noisy/clean segment (more capacity than `window_flash`'s tiny windows, while keeping
   the same structural no-cross-modality-leakage guarantee via the modality-windowing hardening,
   §5.G/[[ar-graft-modality-windowing-hardening-done]]). A 4000-step probe (1/3 budget, not extended)
   was negative on EVERY axis: offline MAE at matched steps WORSE than `window_flash` (0.27 vs. 0.14),
   dream worse (Δ−3.52dB vs. Δ−0.94dB), closed-loop 0/20. Root cause: the dense-segment attention has
   no causal mask, giving training full bidirectional visibility across all chunks of a copy
   (including "future" chunks) — information unavailable at deploy; the smoke test's faster training-
   loss descent was apparently exploiting this non-causal shortcut, not learning anything that
   transfers. See [[ar-segmented-dense-graft-negative]]. **Combined with items 2 and 4 above, every
   attempt to give the AR line's video graft MORE capacity than `window_flash`'s minimal windows has
   now failed — this specific lever looks exhausted.**

**Is further AR investment worth it?** More muted than the 2026-07-05/06 framing suggested, now that
both the AR line's ceiling (revised down to ~9-10%) and the non-AR line's ceiling (revised, but to
~26-33%, i.e. still clearly higher) are corrected for the seed bug. The AR line has a working,
confirmed-non-zero recipe (`low_noise`) and diverse-data transfer remains genuinely open
(§ item 1, plus the still-untested matched-budget subset ladder / curriculum-warm-start ideas in
§G), but every video-graft-capacity avenue (coupling, dense, segmented-dense) is now exhausted, and
the gap to the non-AR line is larger than it looked mid-week. Unless AR/streaming deployment (bounded-
memory, arbitrary-length generation) is itself a hard requirement independent of raw success rate, the
non-AR line is now the clearly stronger bet for further investment — the AR line is not a dead end,
but its remaining upside looks smaller and narrower (data/coverage, not architecture) than it did
before the seed-fix correction.

**Also surfaced, but out of scope for the AR-vs-non-AR question:**
`SANA_WAM_BACKBONE_HANDOFF.md`'s top-ranked surviving direction, **on-distribution DAgger** (0.42
confidence, "the only direction that survives as cause+fix" in that workflow), has never been carried
into this retrospective's ranked list. It remains available if the goal narrows from "explain the
backbone's capability ceiling" to "make one specific task work" — but per HANDOFF's own framing it is
explicitly a **task-fix, not a backbone-capability fix**, so it does not bear on the H1/H2 or
AR/non-AR questions above.

### G. 28-candidate next-directions sweep + parallelization verdict (2026-07-06/07)

A 13-agent workflow (map full project history → 5-lens ideation → 3-way adversarial critique,
`wf_5e1263b4-c02`) produced **28 candidate next-experiments; only 1 was flagged redundant**
(reopen-KEEP-SANA — correctly dropped: the empirical question is already answered in-repo by the
Wan-vs-SANA dream head-to-head, but the action itself is foreclosed by the recorded hard constraint).
Full list + all three critiques (skeptic/optimist/cost-feasibility) preserved in
[[ar-wam-next-directions-workflow]]. **Important correction:** several candidates' own cost estimates
confused wall-clock with GPU-hours — on this project's 8×H200 DDP setup, a 12k-step run is
**~70–75 GPU-hours** (not "4–9"), 24k-step ≈ 139–150 GPU-hours.

**Top-ranked (score ≥8/10, "pursue-now"):**
1. **[TESTED, NEGATIVE, 2026-07-07] Port `low_noise` to the non-AR SANA-MoT line** (A/B vs. its
   current `bsmntw` default). Implemented (`action_loss_weighting` config key added to
   `DualSystemSelfAttnArchitecture`, default `bsmntw` verified bit-identical to prior behavior),
   full retrain (12000 steps, `adjust_bottle`/`clean_50`, `action_self_attn_weight=0.5`), confirmed at
   standard protocol: **28/100 = 28.0%, BELOW the `bsmntw` baseline's 37.5%.** Dream was actually
   somewhat LESS degenerate (Δ−3.21dB vs. the historical Δ−5.92dB baseline) despite the lower
   closed-loop number — the same dream/success decoupling pattern seen elsewhere in this project.
   **Conclusion: `low_noise` does not transfer — `bsmntw` remains correct for the non-AR line,
   `low_noise` remains correct for the AR line specifically; no single loss-weighting scheme
   generalizes across both architectures.** See [[lownoise-non-ar-line-negative]].
2. **[DONE, 2026-07-07] Audit the non-AR line's deploy path** with the same methodology that found 4
   real silent train/deploy bugs on the AR line. **Found a real one:** the deploy path hardcodes
   `seed=42` for EVERY denoising call, ever (`joint_engine.py:435`, `conditions.get("seed", 42)` where
   `conditions` never actually sets `"seed"`) — every closed-loop episode across this line's entire
   history denoises from the bit-identical starting noise tensor, while training uses fresh i.i.d.
   noise every step. Plausibly confounds the existing L0 receding-horizon negative result (which
   triggers ~5× more `generate()` calls/episode, repeating whatever this fixed draw encodes more
   often). Checked and ruled out: `num_frames` persistence, proprio staleness, `execute_horizon`
   wiring, `use_first_frame_cond`, `vace_cache`. **Not yet fixed/re-verified** — see
   [[non-ar-deploy-fixed-noise-seed-bug]] for the recommended near-free fix+re-eval.
3. **Adopt an early-slope diagnostic ladder as standing policy**: screen any candidate at 4–6k steps
   (MAE-descent shape vs. the known single-task-steep / diverse-data-flat reference curves) before
   committing a full 70–150 GPU-hr run — a force-multiplier for the whole backlog below.
4. **DAgger MVP pilot on `lift_pot`** (the 0% task) — gated by a <1hr sim-only smoke test first. Zero
   code artifacts exist today despite being `SANA_WAM_BACKBONE_HANDOFF.md`'s top-ranked surviving
   direction.
5. **[DONE, 2026-07-07] Construction-guaranteed modality-first windowing** — implemented
   (`_segmented_window_graft` in `blocks_split.py`, gated on AR training's `_ar_meta`, deploy
   unaffected since it never has a boundary to guard). Verified bit-exact no-op on the confirmed 25%
   config; a regression test on a different `window_count`/T shape proves the OLD code would leak a
   clean-segment perturbation into the noisy segment's output while the new code stays invariant. This
   is infrastructure/a safety-net, not a new lever — doesn't change the current 25% number by itself.
   See [[ar-graft-modality-windowing-hardening-done]].
6. **Decouple clean/noisy-copy timesteps in `joint_graft`** — 3 independent 2026 papers (DiT4DiT,
   MaskWAM, PFD) converge on exactly this fix for the "dense attention over duplicated content finds a
   shortcut" failure mode, plausibly explaining both the dense-flash negative and `joint_graft`-alone's
   MAE-worsening as one mechanism.
7. **Non-AR→AR hybrid deployment handoff** (first feasibility pass, zero retraining) — run non-AR
   inside its trained window, hand off to AR/streaming only once exceeded; directly targets the
   *original* horizon-mismatch problem that motivated building the AR line.
8. **Disable test-time imagination at deploy** (Fast-WAM/GigaWorld-Policy-style ablation) on the
   confirmed 25% checkpoint — deploy-code-only, no retrain, independently corroborates this project's
   own repeated dream/action-decoupling finding.

Also scored `pursue-now`: tail-checkpoint model-soup on the existing 25% run's checkpoints (zero
training cost), populating the currently-empty `NO_WD_PARAM_SUFFIXES` (pattern-matches how `low_noise`
itself was an orphaned, never-wired mechanism), and a free curve-fit extrapolation on the existing
diverse-data MAE trajectory to check whether its ~0.32 plateau is a true asymptote before committing
to any larger step-budget run.

**Parallelization verdict: yes, clearly and overdue.** A large fraction of the highest-value
candidates above cost 0–30 GPU-hours and need at most 1–2 GPUs, yet this project's entire history has
run one full-8-GPU 9–20-hour job at a time. Nearly every full-retrain candidate preserves its DDP
effective-batch-size via `grad_accum` on fewer GPUs (same total GPU-hours, proportionally longer
wall-clock) — slicing costs only latency-per-job, in exchange for 2–3× aggregate throughput.
Suggested allocation: 1 GPU as a standing "fast lane" for near-zero-cost items (several need 0 GPU at
all), the remaining 6–7 split into 2–3 training slices of 2–3 GPUs each. Full allocation detail in
[[ar-wam-next-directions-workflow]].

### Lower-value / exhausted (do not re-run)
Receding-horizon, multi-task scale-up, IDM re-coupling, proprio-dropout, camera-from-action plucker,
delta actions, single-task more-data, tokenizer replacement (refuted — VAE is shared), **AR-graft's
joint action↔video softmax coupling fix, AR-line only, as a standalone lever (isolated, 0/20 —
Phase 9; may still be worth a jointly-conservative retry alongside `low_noise`, see §5.F item 2)** —
all tested, refuted, or excluded as *standalone* fixes. Note: improving the *dream* (B/C above) and
`low_noise` action-loss reweighting (Phase 9) are NOT in this list — they are confirmed or plausible
levers; what's exhausted is action-head surgery, data tricks, and (for the AR line specifically)
coupling-topology as a standalone fix.

---

## 6. One-paragraph answer to "is SANA the problem?"

Partly — and that is the corrected conclusion. The 0% was the *recipe* (coupling), now fixed →
~37–44%. But the remaining gap to the leaders **is** the SANA backbone: in the same framework, on
the same data and the same Wan-family VAE, **Wan2.2-5B reaches 96.8% while SANA-2B reaches ~44%**.
Since the tokenizer is shared, the difference is SANA's own DiT — its **linear-ReLU attention** and
**camera-domain pretraining** (with 2B-vs-5B scale a secondary factor; a softmax 2B prior like
Cosmos-Policy still reaches 67%). The mechanism is dream quality: SANA's world model dreams
manipulation dynamics poorly, so closed-loop correction fails. The earlier "tokenizer/representation
is a backbone-agnostic ceiling" claim is **withdrawn** — it was refuted by the shared VAE and by
Wan's 96.8%. **UPDATE (Phase 10):** the once-"open" KEEP-SANA levers named here — improving SANA's
*dream* via domain-pretraining, and adding softmax capacity via the graft — were both tried and are
now **exhausted/negative** (pretraining adds zero action grounding, the +0.046 gate; graft family
0–9%). Phase 10 relocates the bottleneck to **closed-loop covariate shift**, and every
covariate-shift lever on the AR line also failed (the ~10% optimum is fragile to any fine-tune). The
remaining decision is strategic (§10.6: pivot to non-AR 26–33% / bank the retrospective / from-scratch
joint training), not another SANA-side lever.
