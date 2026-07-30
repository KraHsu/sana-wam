# SANA-WM: covariate-shift campaign + on-policy DAgger PoC (2026-07-08 → 07-14)

Companion to `SANA_WAM_PROJECT_RETROSPECTIVE.md` §Phase 10. This doc records the
**method, the reusable infrastructure, and the full results** of the campaign that
tried to lift the AR (streaming) SANA-WM line above its ~10% closed-loop ceiling.
Net result: **negative** — the AR ~10% is a knife-edge fragile optimum; every lever
either failed to move it or destroyed it. All experiments ran in the **openwam**
repo (`/home/zch/wuji-openwam-dev`); this doc + conclusions live in **sana-wam**.

Numbers are seed-fixed closed-loop success on `adjust_bottle` (RoboTwin,
aloha-agilex, clean_50), n=100 unless noted. AR `low_noise` baseline =
`sana_ar_graft_AC_lownoise_only` step_12000 = **10.0%**.

---

## 1. Why: the reframe — covariate shift, not capacity

Prior phases blamed SANA's representation/generation capacity. This campaign showed
that is the wrong frame for the closed-loop gap:

- **The task is low-information.** Proprio-only offline action-MAE = **0.059**
  (near-solved without video at all).
- **It is learnable to 96.8%** by a Wan backbone (same VAE/data) — so no capacity
  wall is intrinsic to the task.
- **Open-loop vs closed-loop gap.** Feeding the flagship GROUND-TRUTH observations,
  its action-MAE is **0.077, flat across the 32-step chunk** — yet closed-loop is
  **10%**. Good in-distribution prediction + poor closed-loop = textbook
  **covariate shift**: the policy drifts into off-demo states its training never covered.
- **Encoder is not the lever (the +0.046 gate).** A confound-controlled IDM gate
  (predict the action−proprio *residual*, cross-task held-out, PCA-bottlenecked,
  later-half, 4 splits) gives the SAME transferable grounding for **SANA (diffusion)
  = V-JEPA 2.1 = Cosmos-Reason2 = +0.046**, and **robot-training adds 0** over base
  SANA. The grounding is low-rank (~4 PCA dirs) and nonlinearly encoded. So "pretrain
  harder / swap encoder / JEPA objective" are all refuted; +0.046 is an
  encoder-agnostic property of the observations.
- **Non-AR makes SANA meaningless.** The non-AR flagship has `use_first_frame_cond=False`
  → the video stream is a *blind* text+proprio→video dream, decoupled from the action;
  only proprio conditions the action (≈ proprio+text BC + a decorative dream). This is
  why the AR/streaming line — which conditions on real prior video chunks — is the
  SANA-native one, and why the campaign focused on AR.

Conclusion of the reframe: the binding constraint is **closed-loop robustness /
covariate shift**, addressable (in principle) without touching SANA's capacity.

---

## 2. The on-policy DAgger PoC — infrastructure (reusable)

The true covariate-shift fix is DAgger: roll out the learner, and at the states it
actually visits, record the **expert's** correct action, then retrain. Feasible here
because **RoboTwin's expert is a closed-loop motion planner queryable at any state**.

Pipeline (all verified end-to-end; files in openwam):

1. **Capture** — `sandbox/run_dagger_capture.sh`: deploy the baseline policy, run the
   RoboTwin eval with `DAGGER_CAPTURE_DIR` set. Hook
   `benchmarks/robotwin/dagger_capture_hook.py` (gated on the env var; strict no-op
   when unset) + one gated line in `openwam2robotwin_interface.py::eval()` dumps
   per control step: obs, policy action, and full restorable sim_state (L/R qpos,
   grippers, all object poses) to `ep<seed>.pkl`. Captured 67 baseline rollouts
   (~63 failed = the drift states DAgger wants).

2. **Relabel** — `benchmarks/robotwin/dagger_relabel.py` (run under `$ROBOTWIN_PYTHON`):
   per anchor state, `setup_demo(seed)` + `set_qpos`/`set_pose`/`scene.step()` to
   restore the reached state (restore fidelity: max 0.0001 m), detect the sub-goal
   stage (grasped predicate), roll the RoboTwin scripted expert (primitives +
   `left/right_plan_path`) to task completion, keep only `check_success()` episodes.
   189 anchors → **60 expert-success dagger episodes** written in the exact RoboTwin
   hdf5 format (reuses `merge_pkl_to_hdf5_video`), drop-in for `RoboTwinDataset`.

3. **Co-train** — `configs/dataloader/mixture_dagger.yaml` = clean_50 (50 eps) ∪
   dagger (60 eps), FROZEN base `action_stats` for identical normalization.
   `sandbox/launch_ar_AC_dagger.sh` fine-tunes from the 10% baseline.
   Framework fix: the mixture builder crashed building sub-datasets in a thread pool
   (worker-thread `print` on a closed stdout under hydra) → added
   `MIXTURE_BUILD_SERIAL=1` env → main-thread serial build
   (`openwam/dataloader/mixture.py`, backward-compatible).

This DAgger infrastructure is reusable for any future on-policy work on RoboTwin.

---

## 3. Results — every AR lever, negative

| lever | closed-loop | notes |
|---|---|---|
| AR baseline (untouched) | **10%** | `sana_ar_graft_AC_lownoise_only` |
| graft: causal segmented-dense, cache=0 | 0/20 | future+past asymmetries both fixed |
| graft: + KV-cache (cache=4) | 0/20 | dream still −3.46 dB |
| DART obs-perturb σ=0.05 | 4.8% | monotonically harmful in σ |
| DART obs-perturb σ=0.1 | 1.0% | |
| diverse-data (variant=both, 550 eps) | ~5% | 40k-from-scratch NaN'd @27k (use dream_lr 5e-4); salvaged step_26000 |
| DAgger single-round, aggressive fine-tune (lr 1e-4, 12k) | ~0% | training loss converged, closed-loop collapsed |
| DAgger gentle fine-tune (lr 2e-5, 5k) | **0% at step_1000 AND step_5000** | even 1000 gentle steps destroy it |

The DAgger *pipeline* is correct (verified restore + relabel + load + train). The
*result* is that fine-tuning the AR policy on any modified/augmented data collapses
its ~10% closed-loop competence — even a minimal, gentle fine-tune.

---

## 4. Verdict + open fork

**The AR ~10% is a knife-edge fragile local optimum.** Training loss stays
low/converged while closed-loop collapses ⇒ the fine-tune leaves the narrow
deploy-competent basin. Combined with the graft family failing to move it at all,
**fine-tuning-based improvement of this AR line is not viable.** Meanwhile the non-AR
line sits at **26–33%** (3× higher) despite "wasting" SANA (blind dream).

Open strategic fork (decision, not another SANA-side lever):
1. **Pivot to the non-AR line (26–33%)** — accept the blind-dream "waste" of SANA for
   3× the measured success; do the engineering climb there. *(Recommended on the evidence.)*
2. **Bank the AR retrospective and stop** — levers exhausted, optimum fragile.
3. **From-scratch JOINT training of AR** — mix on-policy/DAgger data in from step 0
   and never fine-tune; the only untested AR variant that structurally avoids the
   "fine-tuning breaks it" failure mode. Higher cost, no guarantee.

---

## 5. Reusable assets left in place (openwam repo)

- `benchmarks/robotwin/dagger_capture_hook.py`, `dagger_relabel.py` — DAgger capture+relabel.
- `sandbox/run_dagger_capture.sh`, `sandbox/launch_ar_AC_dagger.sh` — capture + fine-tune launchers.
- `sandbox/reeval_ar.sh` — parametrized AR deploy+eval (seed-fixed, n configurable).
- `configs/dataloader/mixture_dagger.yaml`; `MIXTURE_BUILD_SERIAL=1` env in `openwam/dataloader/mixture.py`.
- IDM/encoder gates: `sandbox/idm_ar_graft_gate.py`, `idm_crosstask_gate.py`,
  `localize_grounding.py`, `pretrain_effect_gate.py`, `vjepa_grounding_gate.py`,
  `cosmos_grounding_gate.py`; open-loop probe `sandbox/openloop_nonar_probe.py`.
