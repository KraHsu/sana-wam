# SANA-WM World-Action Model — Backbone-Capability Handoff

**Date:** 2026-06-24  ·  **Repo:** `/home/zch/workspace/sana-wam`  ·  **Branch:** `feature/sana-wm-gdn-world-model`
**Memory (read these):** `/home/zch/.claude/projects/-home-zch-workspace-sana-wam/memory/` (`MEMORY.md` is the index)

---

## 0. THE REAL QUESTION (reframed — this is what to solve)

The goal is **NOT** to make SANA-WM pass one task with task-specific tricks (DAgger, delta actions, obs-noise, more data, simpler task — all tried, see §3). The goal is:

> **Why does SANA-WM-as-backbone fail to give the ~<200-demo generalization that other video-pretrained WAM backbones achieve? What about how we use SANA-WM leaves the backbone's capability on the table?**

The user's contribution/constraint is **SANA-WM (GDN-form video world model) as the backbone**. Other WAMs reach **58–74% on RoboTwin 2.0 with ~185 demos/task** (robustness study arXiv 2603.22078; LingBot-VA 74.2%, Cosmos-Policy ~185 traj). SANA-WM at 50–530 demos = **0%**. The question is backbone leverage, not tricks.

**The #1 clue (see §4):** across every experiment, the **action head is DECOUPLED from the world-model's video prediction** — the policy predicts well open-loop *even when the dream is degenerate garbage*. That means we have NOT built a real WAM (where the action leverages the model's dynamics understanding); we have a video model with an independent proprio+text action head bolted via cross-attn. **A true WAM's power is exactly the coupling we're missing.** This is the most promising lead for "why SANA-WM doesn't generalize like other WAMs."

---

## 1. Architecture (what exists)

Two architectures, both: SANA-WM video DiT backbone (GDN/Gated-Delta-Net linear-attn + softmax hybrid, `init_dit_from=/DATA/share/SANA-WM_streaming/sana_dit/model.pt`) + LTX2 causal VAE (8× temporal, 32× spatial, 128ch) + Gemma-2-2b text encoder + a separate **ActionDiT** action head that cross-attends to the video DiT's intermediate hidden states ("bridges").

- **cross-attn** (`src/sana_wam/model/cross_attn.py`, `DualSystemCrossAttnArchitecture`): whole-clip, **bidirectional** GDN, single forward. Config `configs/train_sana_wm_gdn_cross.yaml`. Deploy `cross_attn_engine.py` (frame-0-anchored leading obs window, hard cap num_frames=121).
- **GDN-AR** (`src/sana_wam/model/gdn_ar.py`, `DualSystemGDNARArchitecture`, subclass): **true autoregressive**, cached/streaming GDN, per-chunk rolling loop (K=3 latent frames/chunk). Config `configs/train_sana_wm_gdn_ar.yaml`. Deploy `gdn_ar_engine.py` (STATEFUL recurrent KV cache, ingests real obs chunk each step). **This is the more deploy-faithful / DreamZero-aligned arch.**

Backbone has BOTH GDN linear-attn layers AND softmax layers. Dataloader `src/sana_wam/dataloader/robotwin_dataset.py` (eef 20D actions = [xyz3+rot6d6+grip1]×2; growing_history sampling; min-max normalize). Deploy = HTTP policy server (`scripts/deploy.py`) driven by RoboTwin client (`benchmarks/robotwin/`).

---

## 2. Key concepts (so the numbers make sense)

- **growing_history sampling:** each episode → ~3 windows, all anchored at frame 0, with logical length k ∈ {90, 105, full} (history_min=90, stride=15). A "sample" = "observe a clean prefix [0,P_lat), predict the rest to k". 530 episodes → 1615 windows.
- **step (DDP):** 8 GPUs × bs1 = 8 samples/optimizer-step. 1 epoch ≈ 1615/8 ≈ 202 steps. 6000 steps ≈ 30 epochs.
- **action "loss":** rectified-flow velocity MSE (`target=noise−x0`) at a random timestep — NOT a cm error. Floor is ~0.8–1.5 (target variance ≈1.3); observed ~0.1–0.2 = ~85–90% variance explained (healthy). Real action quality = demo-replay cm-MAE + closed-loop success.

---

## 3. EXPERIMENT LOG (everything tried → all 0% closed-loop on lift_pot)

| run / change | what | result |
|---|---|---|
| `outputs/sana_wm_gdn_cross_liftpot` | cross-attn, absolute actions, **frozen** WM | 0%; demo-replay open-loop **~1.5cm MAE**, gripper corr 0.97 |
| delta actions (cross-attn, single-anchor) | relative-to-current-state, normalized-space | **0/20** |
| `outputs/cross_delta_r0_s01,s03` | R0 DART obs-prefix latent noise σ=0.1/0.3 | trained ~step5800, **NOT eval'd** (ckpts exist) |
| `outputs/cross_unfreeze_vid` | cross-attn, **unfrozen** WM, lambda_video=1 | dream improved +1dB (step_3000 probe); not full-eval'd |
| `outputs/gdnar_unfreeze_vid` | **GDN-AR**, unfrozen WM, lambda_video=1, bs8 | **0/20** (step_4000 0/11) |
| `outputs/gdnar_unfreeze_vid_500` | GDN-AR, unfrozen WM, **530 demos** (10×) | step_4000 **0/6**, step_10000 **0/8** |
| GDN-AR per-chunk delta | implemented (`model.architecture.delta_action`), default off | not trained |
| adjust_bottle `mixed_1000` | simpler(?) task + clean_500+randomized_500 ~1000 | **generating now** (verdict: low-value, see §5) |

**Invariant:** cross-attn↔GDN-AR, frozen↔unfrozen WM, absolute↔delta, bs1↔bs8, 50↔530 demos — **every lever → 0%.**

---

## 4. DIAGNOSTICS / KEY FINDINGS (what we actually know)

1. **Open-loop good, closed-loop drift.** Teacher-forced ~1.5cm + gripper on cue; closed-loop the arms **hover at ~constant height and flail into off-distribution poses, never descend/grasp** (visually confirmed from `/tmp/lp_debug` head-cam rollout). **proprio tracks commanded actions to ~3cm → NOT an actuation/deploy bug; the commanded trajectory itself is wrong.** Failure = **observed-input covariate shift / compounding drift**.
2. **World-model video prediction is DEGENERATE.** Frozen SANA-WM dreams lift_pot futures **worse than copy-last-frame (−4dB PSNR, rel latent MSE >1)** — a blurry blob, not the scene (`/tmp/wm_video_probe.py`). Unfreezing+lambda_video=1 → "recognizable but blurry", **+1dB** over copy-frame at step_3000. SANA-WM **zero-shot does not understand this robot/scene domain**.
3. **⭐ ACTION IS DECOUPLED FROM THE DREAM.** The action predicts ~1.5cm open-loop *despite* the degenerate dream, and **+1dB dream (lambda_video 0→1) did NOT move closed-loop (still 0/11)**. So the action reads proprio+observed-prefix+text and **barely uses the world-model's prediction**. → We are not getting the WAM benefit. **This is the central anomaly to explain/fix.**
4. **Mask/causality RULED OUT.** Deep trace: GDN-AR streaming is causal (cache enforces cross-chunk causality; within-chunk bidirectional is symmetric train/deploy). `run_chunk` passes only the caption mask. No temporal-leak bug.
5. **A soft train/deploy asymmetry exists but is NOT the cause:** video/action diffusion timesteps sampled independently → action sees near-clean GT future-video bridge in training, degenerate at deploy. **Refuted as root cause** (identical in cross-attn which is fine open-loop; +1dB dream didn't help).

---

## 5. WORKFLOW VERDICTS (10 directions explored + adversarially verified, 2026-06-24)

Ranked (priority / adjusted-confidence / survives-refutation):
- **on-distribution DAgger — SOON, 0.42, ✅.** The only direction that survives as cause+fix. **UNLOCK:** the earlier belief "RoboTwin expert isn't a state-conditioned oracle" is **FALSE** — the scripted expert (`grasp_actor`+`move_by_displacement`, `_base_task.py:794-883,1196`) **re-plans from the current/drifted robot config**, so offline sim-DAgger relabeling IS feasible. BUT this is a task-fix, not a backbone-capability fix.
- **action-bridge-dependence — SOON (probe), refuted as cause.** Cheap ~1 GPU-hr ablation (zero/shuffle/clean-vs-degraded the bridge, measure action-MAE) would *definitively* confirm finding #4. **Do this first** — it gates the whole dream/coupling story.
- **deploy-fidelity-gaps — SOON, 0.15.** Real train/deploy asymmetry (GDN-AR engine trailing-band re-encode vs training frame-0 slice) worth fixing on principle; not the cause.
- **state-perturbation-aug — SOON, 0.22.** R0 ckpts exist & unevaluated → near-free closed-loop eval.
- **data-scale-diversity — LATER, 0.88 NOT-cause.** 530 demos w/ strong-backbone precondition still 0%; randomized widens the *visual* axis (WAM strength), not the *state* axis (WAM weakness = our drift). **Won't break 0%.**
- **simpler-task — LATER, 0.12, refuted.** adjust_bottle is actually *longer* (144/174 frames vs lift_pot 113/140); not clearly easier.
- **DROP:** timestep-bridge-leak (0.08), wm-video-quality (0.08), per-chunk-delta (0.04) — all refuted as causes.
- **recipe-alignment — FAILED (no result).** "What do the 58–74% RoboTwin WAMs do that we don't" — **strategically the most important and unanswered; re-run it.**

Workflow run/transcript: `…/subagents/workflows/wf_1003967a-5d1`; raw results `…/tasks/w4vjvrfs3.output`.

---

## 6. BACKBONE-CAPABILITY HYPOTHESES (the real to-do — for the next agent)

The reframe says: stop trick-hunting; explain/fix **why SANA-WM isn't leveraged like other WAM backbones**. Candidate hypotheses, roughly ranked:

- **H1 (strongest): the action is decoupled from the world model.** A real WAM decodes actions *from* the dynamics representation; ours bolts an independent ActionDiT via cross-attn bridges that the policy learns to mostly ignore (finding #4). Investigate: are the bridge features the right representation? Is the cross-attn conditioning structured so the action *must* use the dream (vs treating it as optional side-info)? How do LingBot-VA/Cosmos-Policy/π0 *decode* actions from the world latent (shared trunk? action-as-readout? joint denoising coupling)? **Fix likely = re-couple action↔dynamics, not a data trick.**
- **H2: domain/pretraining gap.** SANA-WM dreams this robot domain at −4dB (finding #2) → its representations may not encode robot-manipulation dynamics. What was SANA-WM pretrained on? Does the prior transfer? Compare pretraining coverage vs Cosmos/Wan2.1. Is the −4dB dream a symptom that the backbone's features are simply uninformative for the policy?
- **H3: recipe.** The 185-demo WAMs use specific finetuning recipes (which layers train, LR schedule, action-head design, proprio injection, action chunking/frequency, normalization). Enumerate concrete deltas vs ours and port the portable ones to SANA-WM. (This is the failed `recipe-alignment` direction — re-run.)
- **H4: are we using SANA-WM's *representations* or just its raw video?** Bridges = intermediate hidden states. Are those the right layers/features? Frozen-vs-unfrozen, which layers, LoRA — are we leaving capability unextracted?
- **H5: scale.** SANA-WM (~2B) vs the larger backbones others use. Is it fundamentally too weak, or under-used (H1–H4)? Settle whether the gap is "under-leveraged 2B" or "2B insufficient."

**Decisive cheap first step:** run the **bridge-ablation probe** (§5) — it directly tests H1 (does the action use the backbone at all?). If decoupled → H1 is the answer and the work is "make the action depend on / decode from the world model." That is the backbone-leverage fix the user is asking for.

---

## 7. INFRASTRUCTURE / ASSETS (so the next agent can run things)

- **Data generation FIXED & working:** RoboTwin at `/home/zch/RoboTwin`, conda env `/home/zch/miniconda3/envs/RoboTwin/bin/python`. **The blocker was a warp 1.14 vs curobo bug — fixed:** `envs/curobo/src/curobo/geom/sdf/world_mesh.py:67` `wp.torch.device_from_torch`→`wp.device_from_torch`. Added `SEED_START` env to `script/collect_data.py` for parallel gen. Recipe: `PATH=<robotwin-env>:$PATH SEED_START=N bash collect_data.sh <task> <config> <gpu>`. Two phases: seed-search (no save) → save (writes HDF5). Per-seed failures ~20-30% normal.
- **Merge script:** `/tmp/merge_gen.py <task> <robot> <variant> '<src_glob>' [extra_dir]` → merges generated dirs into `{dataset}/{task}/{robot}_{variant}/data` (symlinks hdf5, copies instructions, merges scene_info episode_N keys).
- **Data on box:** `/DATA/share/RoboTwin2.0/dataset/` — lift_pot/{clean_50(50), clean_500(530)}, adjust_bottle/{clean_50(50), mixed_1000 generating}. Only one robot (aloha-agilex). Stats auto-computed by the dataloader (DDP poll-sync).
- **Train:** `NCCL_NVLS_ENABLE=0 GDN_DISABLE_COMPILE=1 .venv/bin/torchrun --nproc_per_node=8 scripts/train.py --config <cfg> dataloader.train_tasks=[<task>] dataloader.num_frames=121 dataloader.variant=<v> training.freeze=[] training.lambda_video=1.0 training.video_lr=1e-5 training.max_steps=12000 training.output_dir=<out>`. `training.freeze=[]` unfreezes the video DiT (VAE+Gemma stay frozen). ~2.4s/step on 8 GPUs, ~3.26B trainable when unfrozen.
- **Eval (closed-loop):** `/tmp/eval_lp_gdnar.sh <ckpt_dir> <label> <srv_gpu> <sim_gpu> <test_num> <port>` (GDN-AR engine, deploy_gdn_ar.yaml). cross-attn = `/tmp/eval_lp.sh` (deploy_gdn_cross.yaml, history_len=121). Pin a checkpoint: symlink config.yaml + action_stats.npy + checkpoint_step_N.safetensors into a clean dir. Debug capture: set `debug:true` in a copy of `benchmarks/robotwin/policy_config.yml`, `POLICY_CONFIG_PATH=…`; saves per-step cams + action/state JSON.
- **Probes:** `/tmp/wm_video_probe.py <ckpt_dir> <N> cuda:0` (WM dream quality, delta-aware variant `/tmp/wm_video_probe_spread.py`); `/tmp/demo_replay_diag.py <ckpt_dir> <N> cuda:0` (open-loop teacher-forced cm-MAE — whole-clip, NOT AR-representative; an AR-aware version still needs building).
- **GPUs:** 8× H200 (143GB). Box runs **UTC** (Beijing = UTC+8).
- **Uncommitted RoboTwin fixes** (separate repo, not in sana-wam git): the warp one-liner + the `SEED_START` hook + traceback-enable in collect_data.py. Preserve them.

---

## 8. WHAT'S CURRENTLY RUNNING / STATE

- adjust_bottle `mixed_1000` data **generating** (8 procs in `/home/zch/RoboTwin/data/adjust_bottle/ab_{clean,rand}_p0..3`); a **durable cron** (`.claude/scheduled_tasks.json`) will merge+train it on completion. Workflow judges it low-value (§5) — fine to let finish or cancel.
- No training running (lift_pot stopped). GPUs free except the generation (~3GB/GPU).
- All sana-wam code changes (delta, R0, GDN-AR delta) committed: `baa52c9`, pushed.

---

## 9. POINTERS

Memory files to read: `MEMORY.md`, `wm-video-quality-probe.md` (the master diagnosis), `dreamzero-architecture.md`, `gdn-ar-delta-impl.md`, `delta-action-impl.md`, `robotwin-data-generation.md`, `sana-wm-pretrained-direction.md`, `gdn-ar-track.md`, `gdn-ar-deploy-gaps.md`.
Key code: `model/gdn_ar.py`, `model/cross_attn.py`, `model/video_backbone/sana/adapter.py` (run_chunk/forward_long/bridges), `model/action_backbone/` (the ActionDiT + how it consumes bridges — central to H1), `deploy/gdn_ar_engine.py`, `dataloader/robotwin_dataset.py`.

---

## 10. UPDATE (2026-06-24 PM) — supersedes the data/comparability bits above

**The data picture changed — this matters a lot.** We had been training on tiny *self-generated* demos with random seeds (use_seed:false) and eval'ing at only test_num=20 — i.e. **NOT the standard benchmark data or protocol**, making our 0% non-comparable to the published 58–74%. Then we found the **official RoboTwin2.0 dataset on a remote server** and are transferring the **full 896 GB / 50 tasks** into `/DATA/share/RoboTwin2.0/dataset/` (each task = official `clean_50` + `randomized_500`; matching loader path). Transfer = `scripts/transfer_robotwin.sh` (tar-over-ssh per task, resumable; remote `root@121.43.126.205:1020:/mnt/cpfs/wangyuran/RoboTwin2.0/dataset`), ~70 MB/s, ETA ~UTC 18:00 / ~00:20 Beijing. A durable cron watches/relaunches it.

**Consequences for the plan:**
- **Use the official data** (`variant=randomized_500` or `both`), NOT our self-gen (`clean_500`/`mixed_1000` — those coexist as extra variant dirs, ignore them). Local adjust_bottle generation + its cron were stopped/cancelled (redundant).
- **Eval at the FULL standard protocol**: `eval_policy.py` uses `st_seed=100000*(1+seed)`, **`test_num=100`** — use 100, not 20. The eval seeds we used were already standard; just under-ran the count.
- **Multi-task is now possible (50 tasks)** — closer to how real WAMs get <200-demo generalization (heterogeneous data); directly relevant to the backbone-capability question.
- **First clean experiment (was impossible before):** train GDN-AR (unfrozen WM) on the **official `randomized_500`** (single task) AND/OR **multi-task**, eval at **test_num=100**. THIS is the real apples-to-apples baseline. If still ~0% on standard data+protocol → the gap is genuinely backbone/recipe (pursue H1 recoupling + recipe-alignment). If it works → our prior 0% was partly the non-standard tiny data, and the lever is data/scale/multi-task.
- The **bridge-ablation probe (§5/§6 H1)** is data-independent (uses existing checkpoints) — still the cheapest first science: does the action even use the dream?

**Net reordering of "what to do next":** (0) let the official transfer finish; (1) bridge-ablation probe on an existing ckpt (gates the dream/coupling family, ~1 GPU-hr, no retrain); (2) train on official randomized_500 + multi-task, eval at test_num=100 → the real baseline; (3) decide backbone-capability (H1 recouple action↔WM + recipe-alignment) vs scale (multi-task) based on (1)+(2).

---

## 11. SESSION 2026-06-25/26 — the diagnostic chain that pinned H2, and where it leaves us

This session executed §10's plan and went further. **Net: the binding constraint is H2 = the backbone's
LATENT/representation (the LTX2 reconstruction-VAE space), proven by elimination. Not the action-head
coupling, not the camera, not the dream-quality knobs, not training the backbone harder.** Memory files
(`…/memory/`) hold the detail; this is the log.

**Experiments + findings (in order):**
1. **Bridge-ablation probe (GDN-AR)** `/tmp/bridge_ablation_gdnar.py` — AR-faithful, differential. On 2
   checkpoints: shuffle/mean-pool the bridge = **1.00× baseline** action MAE; zero = 1.16×; clean GT
   future (oracle dream) = 1.2× WORSE. → the action's cross-attn to the WM **collapsed to a content-free
   constant**; H1 (decoupling) CONFIRMED and mechanistically precise. (`bridge-ablation-h1-confirmed.md`)
2. **Re-couple options workflow** (34 agents, double-reviewed) → `docs/BACKBONE_RECOUPLE_OPTIONS.md`. TOP-3
   = IDM-aux / joint-denoise MoT / GDN-state readout; gated by a cheap latent-decodability probe.
3. **IDM-decodability GATE** `/tmp/idm_decodability_probe.py` — on SINGLE-task lift_pot, the WM rep carries
   **~0 action info beyond proprio** (oracle, randomized, even grasp R²=0.998 from proprio). Re-coupling
   can't extract info that isn't there. (`idm-decodability-gate.md`) + **OOD-correction probe**
   `/tmp/ood_correction_probe.py`: under proprio drift, vision gives a drift-immune ~4cm fallback →
   motivated proprio-dropout.
4. **Official data**: 896G/50 tasks transferred + verified (each clean_50 + randomized_500). Transfer cron
   deleted.
5. **Real baseline**: GDN-AR unfrozen on official **randomized_500 (single-task lift_pot)**, lambda_video=1,
   12000 steps, clean convergence (no divergence) → **0/48 (0.0%)**. Apples-to-apples: 0% is NOT the tiny
   non-standard data / test_num=20. (`official-baseline-run.md`)
6. **proprio-dropout + Fix A** (both IMPLEMENTED): `proprio_dropout`/`proprio_noise_std` knobs in
   `base._append_proprio_context_token` (read in `gdn_ar.__init__`, training-only); **Fix A** =
   `gdn_ar.compute_loss` now COUPLES the video↔action diffusion timestep (= DreamZero's shared-timestep
   design, [[dreamzero-architecture]]). Run on randomized_500 single-task → **0/20**. Bridge-ablation with
   proprio ABSENT: the action leans MORE on the bridge but still gets only its CONSTANT → H2 is the binding
   constraint, action-head coupling is not the lever.
7. **Camera/plucker hypothesis TESTED + REFUTED** `/tmp/dream_plucker_ab.py` — the plucker pathway is
   omitted (`plucker_emb=None`) yet heavily pretrained (plucker_proj norm ~47 × 20 blocks). For a static
   RoboTwin head cam the correct plucker = SANA-WM's own identity-pose fallback. Feeding it changes the
   dream **−0.0%**. Camera is not the cap. (`camera-plucker-tested-refuted.md`)
8. **Backbone-H2 strategy workflow** (28 agents) → `docs/BACKBONE_H2_STRATEGY.md`. Every 58–74% RoboTwin
   leader uses a **robot-pretrained Wan/Cosmos** backbone; none SANA-like. (`backbone-h2-strategy.md`)
9. **Domain-pretrain SANA-WM** (keep-thesis path; user requires SANA stays the backbone): clean_50
   multi-task (49 tasks; randomized_500 abandoned — `open_microwave` 1038-frame episodes overflow the action
   RoPE cache / NCCL-hang; use `growing_history=false` if needed). **video_lr=1e-4 DIVERGES** (video loss
   0.5→0.96, dream worse); **video_lr=3e-5 STABLE but the DREAM DOES NOT IMPROVE** (within-run AR rel-MSE
   step_500=0.537→step_2000=0.551; velocity-loss↓ but dream-flat). → training the backbone harder is NOT
   the lever (capacity/architecture ceiling). **BUT the project-changing nugget:** on MULTI-TASK data the
   bridge carries **+0.12 xyz R² action info beyond proprio** (single-task hid it because proprio was
   sufficient) — confirmed via control (step_500 ≈ step_2000, so it's a multi-task property, not from
   pretraining). (`domain-pretrain-result.md`)
10. **RepWAM verified** (arXiv 2606.13674, github.com/wdrink/RepWAM): **89.3 RoboTwin2.0 FROM SCRATCH** (no
    WAN init). The LEVER is the **tokenizer** (RepViTok = recon + semantic-alignment-to-foundation-model +
    latent-action-as-transition), +8.6/+7.1 over the WAN2.2 reconstruction VAE — NOT a separate IDM loss
    (our "+IDM" gloss was wrong; our repo `'idm'` variant is a stub). Hard constraint: **GDN cannot do MoT
    joint-attention** (frame-wise recurrence) — only the (collapsed) cross-attn bridge. RepWAM's thesis =
    our H2 exactly: the reconstruction-VAE latent, not the coupling, is the ceiling. (`repwam-verified.md`)

**CONVERGENT CONCLUSION:** action-head coupling (H1), camera, dream-quality, and harder backbone training
are all REFUTED as the lever. The ceiling is the **LTX2 reconstruction-VAE latent space** (= RepWAM's
finding). The lever is a **dynamics-aware tokenizer**. Tension with "keep SANA": the SANA DiT *architecture*
can stay, but it was pretrained on LTX2 latents — swapping the tokenizer largely invalidates that
pretraining (RepWAM's 89.3-from-scratch says the weights matter little, the tokenizer does).

**WHAT'S RUNNING (this session's tail):** multi-task **frozen-video** + proprio-dropout finetune
(`outputs/sana_mt_frozen_propdrop/…`, 601M trainable = action head only, clean_50 49-task, lambda_video=0,
proprio_dropout=0.5) — the CHEAP test of whether the multi-task bridge's +0.12 info + proprio-dropout
forcing can break 0%. Eval pending at step ~2500–3000 (test_num).

**DECISION FORK (set by that eval):**
- multi-task frozen >0% → the bridge is usable in multi-task; a **from-scratch full-data GDN-AR** (keep
  arch, `init_dit_from=null`, multi-task, joint, proprio-dropout — note LTX2/Wan are both reconstruction
  VAEs, NOT the RepViTok lever) likely lifts it further by also fixing the dream.
- still 0% → even with bridge info the coupling can't catch it → must **change the coupling** (on GDN:
  IDM/GDN-state readout; or leave GDN for MoT) and/or **replace the tokenizer** (RepViTok-style). The
  cheapest next science either way = the latent-decodability probe on a DINOv2/SigLIP semantic latent vs
  LTX2 (does a dynamics-aware latent carry the action info LTX2 lacks?).

**New code (committed-worthy):** `proprio_dropout`/`proprio_noise_std` (base.py + gdn_ar.py); Fix A
timestep-coupling (gdn_ar.compute_loss). **New probes (in /tmp):** `bridge_ablation_gdnar.py`,
`idm_decodability_probe.py`, `ood_correction_probe.py`, `dream_plucker_ab.py`, `overfit_video.py`.
**Infra gotcha:** randomized_500 multi-task needs `growing_history=false` (or higher `max_action_len`) —
long episodes (open_microwave 1038f) overflow the 1024 action-RoPE cache and trigger NCCL all-reduce hangs.
