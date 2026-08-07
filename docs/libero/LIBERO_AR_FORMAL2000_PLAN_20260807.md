# LIBERO AR Formal2000 training plan — 2026-08-07

## Decision

T16 closed-loop evidence passed, so architecture validation stops here.  The
next action is one production-data training endpoint, not another architecture
screen.  Safety and reporting work is restricted to what prevents an invalid
or non-reproducible training output; additional refinements wait unless the
model architecture actually fails.

## Frozen training endpoint

- Architecture: the T16-validated LIBERO autoregressive dual-system model.
- Initialization: fresh action/proprio modules with seed `20260807`; the pinned
  SANA Video 2B, Wan VAE, and Gemma assets initialize the frozen video path.
- Population: all four repaired LIBERO training suites, excluding goal episode
  82, using selected-row stats SHA256
  `e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7`.
- Optimization: one H200, batch 1, accumulation 8, FP32 AdamW masters, 2000
  optimizer steps, action-only objective.
- Publication: no initial or intermediate saves; only
  `checkpoint_step_2000.safetensors` plus saved config and action stats.
- Recovery: intentionally non-resumable.  Any interruption freezes that root as
  failed; a rerun would require a separately authorized fresh root and fresh
  initialization.

R0 reached optimizer step 1, then exposed a real tail window for which the
unused internal video diagnostic required a target even though
`training.lambda_video=0`.  R0 is frozen failed with FAILED SHA256
`0a62b6a66ba152285d751795330d90a93558c2ca851706be0549dbd6966831c2`.
R1 makes the action-only contract explicit with
`video_on_path_loss_weight=0.0`; the existing padding regression test proves
that a bootstrap-only video tail remains valid when an action target exists.

The R1 single allowed root is
`/DATA/share/sana_wam_libero_training/formal2000/libero-ar-formal2000-r1-single-gpu-2000-ced0769ce3a32b441cda12a04929d6f9`
with nonce `ced0769ce3a32b441cda12a04929d6f9`.

## Admission and result boundary

Before CUDA construction the launcher binds the final source commit, runner and
config hashes, Sana gitlink, the frozen T16 RESULT, production stats/live data,
all base-model bytes, an idle physical GPU 7 with its UUID, and a nonexistent
exclusive root.  It rejects training-checkpoint initialization and resume
inputs.  After model construction it requires the T16 base-load signature and
the same four trainable parameter roots.  A PASS requires exactly 2000 updates,
finite final model/master/optimizer state, exactly one final checkpoint, and a
verified checkpoint/config/stats manifest before the root is frozen.

## Immediate evaluation after training

While training runs, prepare only the minimum LIBERO deployment correction:
history 113, 29 video frames, 28 action tokens, rolling observation chunks,
predicted cache feedback, four video/action denoise steps, no async/compile, and
physical EGL device 7.  When the checkpoint is frozen, run a 40-episode
task-cover pilot (four suites × ten tasks × one init).  If it shows usable
signal, expand directly to the full 800-episode benchmark; if it does not, use
the pilot evidence to decide whether the failure is training scale or the model
architecture.
