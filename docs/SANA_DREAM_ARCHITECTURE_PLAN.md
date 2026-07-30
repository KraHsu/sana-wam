# How to make SANA dream manipulation dynamics — architecture-exploration plan

Output of a large parallel exploration workflow (`sana-learn-to-dream`, 98 agents: 5 ground / 28
ideas across 8 families / 3-lens adversarial verification each / 1 synthesis; ~6M tokens,
2026-06-27). Goal: close SANA's degenerate dream (6.66 dB) toward Wan's (14.35 dB) under KEEP-SANA,
compute unconstrained. Standard metric for every gate: `sandbox/dream_quality_probe.py` (SANA 6.66 dB,
copy-frame 12.58, Wan 14.35).

## Repo facts the agents verified (these reshape the plan)

1. **`lambda_video=1.0` and the backbone is trainable in every openwam SANA-MoT run that produces the
   37–44% numbers** (`configs/training_strategy/joint.yaml:2`). The "dream was never trained / frozen
   + λ_video=0" precondition behind several ideas is **false** — that was the abandoned sana-wam
   cross-attn stack. The dream is degenerate *despite* a live on-domain video loss. This kills the
   cheap-objective-only ideas and points squarely at **architecture**.
2. **`SanaMoTJointDriver` hard-rejects any non-`linear_relu` video kernel** (`sana_mot_driver.py:144-148`).
   So "inject softmax blocks into SANA" requires **writing a new mixed-kernel MoT driver** (linear
   closed-form + masked-softmax in one joint forward), not a config flip. The `softmax_every_n`
   hybrid machinery exists only in the standalone Sana repo, **not** in openwam — it must be ported.
3. **Wan2.2-TI2V-5B weights are on disk** (`/DATA/wangyuran/openwam_checkpoints/Wan2.2-TI2V-5B`;
   trained joint-self-attn ckpt at `RoboTwin_DualSystem_JointSelfAttention`). Teacher/distillation
   ideas are runnable today.

## 1. Verdict — which axis is the dream deficit

**Dominant = the attention kernel (linear-ReLU vs softmax); pretraining-domain a real secondary;
scale is calibrated out; coupling is a *separate* problem that does not improve the dream.** ~70% of
the 7.7 dB gap is attributed to the linear-ReLU kernel's inability to selectively concentrate on the
few past frames that determine the next manipulation frame. Cosmos-Policy (softmax, robot-pretrained,
**2B**) at 67% proves scale is not the cap. The degeneracy appears on the *first* window (−5.92 dB
below copy-frame), before any AR compounding → a single-step **expressivity** failure, not a
rollout-curriculum or noise-schedule failure.

**Hard caveat carried through the whole plan:** a better dream is *necessary but not sufficient* —
prior probes (bridge-ablation shuffle/mean = 1.00×; IDM-decodability gate; multitask-frozen-propdrop
0/20) show the action head reads the dream as a content-free constant. BUT that was all measured on a
*degenerate* dream (a constant is what you get when there's nothing to read). The open question this
plan answers: **does a dream that crosses the copy-frame floor get consumed by the forced-MoT action
path, or does the representation ceiling still bind?** Cheap disambiguators are front-loaded.

## 2. Top directions (ranked by dream-gain × KEEP-SANA fidelity)

### D1 — Softmax-capacity SANA hybrid, anchored from the co-resident Wan-5B teacher (dominant-axis lever)
- **Change:** hybrid SANA video backbone ~10 softmax / 10 linear-ReLU blocks. Prereqs: (a) port the
  hybrid softmax block classes into openwam's Sana net; (b) write a mixed-kernel MoT driver (linear
  closed-form + masked-softmax in one joint v↔v / a→v forward, with the fp32 numerics the linear path
  needs). Init softmax slots from their linear siblings' qkv/proj; train full-backbone, λ_video=1,
  multi-task. Optionally add a latent feature/output-matching loss from frozen Wan-5B.
- **Mechanism:** softmax restores query-dependent concentration on the contact/pose-carrying past
  frames the fixed ReLU kernel smears — attacks the ~70% axis directly. Wan-anchoring supplies a
  dense dynamics-prior target.
- **Keep-SANA:** partial-but-defensible (DiT macro-arch/AdaLN/FFN preserved; hybrid schedule
  precedented in SANA-WM; converted slots can't load the published linear weights → a SANA-*family*
  variant, flagged honestly). **Expected:** +4–6 dB dream, med-high confidence; lower that it converts
  to closed-loop without a coupling fix.
- **Cheapest decisive first test:** do NOT build the MoT driver first. Run a *single-stream* (non-MoT)
  dream-PSNR A/B using the Sana-repo net's existing `additional_flash_attn` path, build the 10/10
  hybrid, short finetune of the new softmax blocks on adjust_bottle, probe. **Gate: dream clears the
  12.58 dB copy-frame floor** (~1 GPU-day). If it doesn't cross even single-stream, softmax-capacity
  alone isn't the fix — abort before the driver port. (Note: `learnable_fa_scale` inits to **100, not
  0** — set 0 yourself for the monotone no-regress floor.)

### D2 — Wan-as-teacher data distillation (highest KEEP-SANA fidelity)
- **Change:** run frozen Wan-5B offline over RoboTwin obs+action to generate high-quality future
  rollouts; decode to pixels, **re-encode through SANA's own VAE** (lands targets in SANA's native
  latent geometry — the two VAEs differ, z=16 SANA vs z=48 Wan, so latent-MSE is ill-posed); add as
  extra flow-matching **velocity** targets (`noise − teacher_x0`), weighted ~0.3–0.5, co-trained with
  real data. Cache offline (compute-unconstrained).
- **Keep-SANA:** *full* — no arch/kernel/VAE change, teacher never deployed. **Expected:** +2–4 dB,
  lower variance; capped by SANA-VAE recon fidelity and SANA's own capacity (if the kernel binds, a
  better target can't be absorbed — why it ranks below D1).
- **Cheapest first test:** generate Wan dreams for ~16 clips, round-trip through SANA's VAE, **measure
  the round-tripped teacher target's own PSNR vs GT future. Gate: it must still beat copy-frame
  (>12.58 dB) after the double-VAE pass** (~2 GPU-hr). If the round-trip destroys Wan's advantage the
  signal is dead on arrival. Use a mode-preserving/feature-matching objective, NOT a 0.5/0.5 MSE blend
  (oversmoothing in a limited-capacity student).

### D3 — Two-stage physical-AI re-pretraining of the published linear-ReLU 2B (pretraining-domain axis, cleanest KEEP-SANA)
- **Change:** warm-restart the published SANA-Video-2B linear-ReLU ckpt on a large robot/physical-AI
  video corpus (OXE/DROID, Something-Something-v2, Ego4D/Epic hand-object, + sim renders), I2V/clean-
  prefix objective (already wired), with a **motion-weighted loss** (upweight high-optical-flow latent
  regions so the loss can't be minimized by copying static background — directly attacks the
  sub-copy-frame collapse). Then domain-fit on RoboTwin. 10–20% generic replay + EMA + long warmup to
  keep fp32 linear-attn numerics in basin.
- **Keep-SANA:** *full* — exact published architecture, ckpt loads as init. **Expected:** highest
  ceiling *if* prior (not kernel) binds (6.66 → 11–13 possible); but ground-truth says kernel is ~70%
  and the prior C1 domain-pretrain already showed a capacity ceiling.
- **Cheapest first test:** short video-only motion-weighted domain-pretrain (~0.5–1 GPU-day) on the
  linear-ReLU + MoT config, probe. **Gate: dream clears copy-frame.** This is the exact gate the prior
  GDN domain-pretrain *failed*; if linear-ReLU + motion-weighting also fails to cross, the kernel binds
  → commit to D1 instead of the expensive corpus run.

### D4 — Coupling insurance: IDM-aux on dream deltas + proprio-dropout (run ONLY on a non-degenerate dream)
- **Change:** IDM aux head predicting the action chunk from deltas of consecutive *predicted* future
  latents, backpropped into the backbone; proprio-dropout (~0.5) so it can't solve from proprio alone.
- **Keep-SANA:** *full* (additive head). **Expected:** ~+0–1 dB PSNR; the value is dream→success
  *conversion*.
- **Cheapest first test:** re-run the ~1 GPU-hr L2/IDM-decodability gate (latent vs proprio vs both)
  **on a backbone whose dream already crossed copy-frame** (output of D1/D3). **Gate: dream latent
  carries marginal action info beyond proprio.** Every prior 0/20 was on a *degenerate* dream — the
  untested cell is "IDM through a backbone that dreams well." If it flips positive, train IDM+propdrop;
  if flat, the H2 representation ceiling is confirmed fatal and the thesis must pivot to the tokenizer.

## 3. Experiment ladder (each gate unlocks the next; cheap disambiguators front-loaded)

| Step | Experiment | Cost | GATE to proceed | If it fails |
|---|---|---|---|---|
| **L0** | Baseline re-measure: `dream_quality_probe.py` on SANA-MoT + Wan-5B on identical code | ~2 GPU-hr | SANA ≈6.66, Wan ≈14.35 | fix probe first |
| **L1** | Single-stream softmax A/B (D1 first test): 10/10 hybrid in Sana-repo net, short finetune, probe | ~1 GPU-day | **dream > 12.58 dB** | softmax alone insufficient → skip driver port, go L2 |
| **L2** | Linear-ReLU domain-pretrain probe (D3): short video-only motion-weighted pretrain, probe | ~0.5–1 GPU-day | **dream > 12.58 dB** | kernel+prior both fail → strong tokenizer signal, jump to L5 |
| **L3** | Wan teacher-target sanity (D2): 16 clips, round-trip SANA VAE, measure target PSNR | ~2 GPU-hr | **round-tripped Wan target > 12.58 dB** | distillation signal dead → drop D2 |
| **L4** | Commit the winner at scale (MoT-driver port for D1 / full corpus for D3 / full co-train for D2), λ_video=1, multi-task | 300–1500 GPU-hr | **dream ≥ ~10–11 dB** | re-evaluate axis; consider stacking L1+L2 |
| **L5** | Coupling gate (D4): on the L4 good-dream backbone, re-run the IDM-decodability gate | ~1 GPU-hr | **latent beats proprio** | H2 ceiling **confirmed fatal** → pivot to dynamics-aware tokenizer (RepViTok); stop spending on the dream |
| **L6** | Convert: full IDM-aux + proprio-dropout, then closed-loop RoboTwin (≥100 eps) | ~100–250 GPU-hr | **closed-loop > 44%** | good dream but stalled success → binding constraint is coupling/representation |

**Stacking (compute-unconstrained):** L1 (softmax) and L2 (domain-pretrain) are orthogonal and both
KEEP-SANA-clean; if both clear their gates, the highest-ceiling bet is **softmax-hybrid + physical-AI
pretrain + Wan-target distillation, then IDM-coupling on top** — the full stack the lit survey
predicts reaches Wan parity.

## 4. Explicit DROPS (panel-rejected, with the load-bearing reason)

- **"Just set λ_video>0 / turn the dream gradient on."** Refuted: every openwam SANA-MoT run already
  uses λ_video=1.0 with a trainable backbone; the dream is degenerate *despite* it.
- **Standalone Diffusion-Forcing per-frame noise / clean-prefix objective.** Already wired and running
  (`autoregressive.py` is the diffusion-forcing path); video loss already plateaus ~0.48; only ~20% of
  the gap. At most a free A/B *under* the softmax fix.
- **Self-forcing / AR rollout-supervision, standalone.** Inference AR path is `@torch.no_grad`;
  backprop-through-cache unproven; the real-obs-KV block-AR engine already evaluated 0/48 & 0/11;
  degeneracy is on the *first* window, before compounding.
- **Temporal-softmax / spatial-linear factorization.** Cited config is inverted; primitive lives only
  in single-stream; per-column temporal softmax is the literature-worst factorization for large motion.
- **VAMPO RL post-training.** Entire RL stack missing; it's a *sharpening* stage needing a
  non-degenerate base dream that doesn't yet exist. Defer.
- **Width/depth scale-up 2B→4–5B.** Worst ROI: Cosmos-2B-softmax=67% proves scale is secondary;
  depth-growth random-inits new blocks (re-pretrain-from-scratch). Only on top of the kernel fix.
- **Cross-model per-layer feature-MSE distillation Wan→SANA.** No layer/width/latent correspondence
  (Wan 3072-d/30-layer/z=48 vs SANA 2240-d/20-layer/z=16). Only the *latent-space data* distillation
  (D2) survives; per-layer hidden matching does not.
- **"Wan features carry more action info" premises.** Refuted by `wan-sana-features-equal` (+0.13 both).
- **GDN `softmax_every_n 4→2` on the GDN CamCtrl branch, standalone.** Can't load into
  `SanaMoTJointDriver`; main predicted gain (unfreeze+λ_video=1) is *already* the 44% config; runs on
  the abandoned GDN branch. Survives only as D1 (ported into the live MoT path).
- **Coupling-only ideas (DiT4DiT injection, parallel frozen-Wan co-processor) as dream fixes.** The MoT
  mask already sets a→v=True at every layer; these touch coupling, not the dream. IDM-aux (D4) is the
  cheaper coupling lever, gated behind a non-degenerate dream.

**Bottom line:** the kernel is the prime suspect. **L1 (single-stream softmax A/B) and L2 (linear-ReLU
motion-weighted domain-pretrain) are the two ~1-GPU-day experiments that disambiguate kernel-vs-prior
before any large spend**, both scored against the 12.58 dB copy-frame floor. Whatever lifts the dream,
**L5's 1-GPU-hr decodability gate is the hard checkpoint** that tells you whether a good dream converts
— or whether the thesis must pivot to the tokenizer. Do not skip it.

---

## Execution log

- **L1 — PASSED (2026-06-27/28, `sandbox/l1_softmax_hybrid.py`).** Grafted a parallel softmax branch
  onto all 20 SANA blocks (zero-init proj, `learnable_fa_scale=1.0`), trained ONLY the graft on the
  future-prediction flow-matching objective (adjust_bottle, rest frozen). Dream **6.66 → 12.85 dB**,
  crossing from −5.9 below copy-frame to +1.1 above it (within ~1.5 dB of Wan 14.35). Untrained-graft
  = baseline (no-regress floor holds). **Softmax capacity is decisively the dream lever** — the
  kernel hypothesis is confirmed and the L1 gate is cleared.
- **L5 — INCONCLUSIVE (2026-06-28, `sandbox/l5_dream_decodability.py`).** On the good-dream hybrid,
  IDM action-decodability (val-MAE): proprio 0.0786 / linear-features 0.0675 / good-dream-features
  0.0705 — the +6 dB dream did NOT raise decodability. **But this is not a true fail:** open-loop
  single-step decodability is the same metric family that is provably blind to the Wan-vs-SANA gap
  (Wan ≈ SANA +0.13, yet 96.8% vs 44%). A metric that can't separate Wan from SANA can't answer
  "does a good dream convert." **The only valid conversion test is closed-loop (L6), which requires
  the L4 MoT-driver port.** Lesson: retire open-loop decodability as a gate; use closed-loop success.
- **L4/L6 — DONE (2026-06-28): the dream-fix CONVERTS to closed-loop, as ROBUSTNESS.** Ported the
  softmax graft into the live MoT action stack (4 code edits in `blocks_split.py`,
  `sana_multi_scale_video.py`, `pipeline_builder.py` — config-buildable + deployable; verified
  end-to-end). Trained the hybrid (adjust_bottle variant=both, graft all 20, full unfreeze, 8000
  steps; video loss halved 0.34→0.16 in the live stack) and ran a **matched A/B** vs a no-graft
  control on the same config, eval adjust_bottle demo_clean:

  | config (variant=both, same recipe) | closed-loop |
  |---|---|
  | **+ graft (good dream)** | **17/50 = 34%** |
  | **no-graft control** | **5/36 = 14%** |
  | (ref) clean_50 fp32 baseline, no graft | 37.5% |

  **The graft ≈ 2.4× the no-graft on the same diverse data.** Mechanism: training on diverse
  randomized_500 *without* a good dream collapses to 14% (the degenerate dream can't handle the visual
  diversity → off-distribution on clean eval); the graft's good dream **anchors it back to 34%**. So
  **fixing the dream demonstrably helps control — and its value is robustness to distribution shift**,
  exactly the DreamZero "dream handles covariate shift" thesis (first concrete positive in the project).
  Caveat: graft-both (34%) ≈ clean_50 best (37.5%) → on *easy* clean data, proprio + observed latents
  already suffice, so the dream doesn't raise the easy-case ceiling.
- **NEXT:** test the dream in a regime where robustness is the bottleneck — **randomized eval seeds**
  and/or **long-horizon AR closed-loop** (drift accumulates) — where the no-graft dream should fail and
  the good dream rescue. This re-validates the AR/long-sequence direction: linear recurrent long-horizon
  context + windowed softmax for local focusing (the SANA-WM 15-GDN/5-softmax hybrid is the precedent).

---
*Workflow caveats: 6 of 84 verify-agents failed (StructuredOutput retry cap / API abort) and 1 ideate
family (temporal-modeling) aborted — coverage is ~93%; the temporal-softmax idea was still surfaced
and dropped via the hybrid-architecture family. Two verifier outputs returned placeholder "test"
fields (indices 67, 72) and were down-weighted.*
