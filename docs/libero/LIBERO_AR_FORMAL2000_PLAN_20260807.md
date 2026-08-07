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

The single allowed root is
`/DATA/share/sana_wam_libero_training/formal2000/libero-ar-formal2000-single-gpu-2000-dba67fee2b53ecce092898c0a70144c4`
with nonce `dba67fee2b53ecce092898c0a70144c4`.

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
