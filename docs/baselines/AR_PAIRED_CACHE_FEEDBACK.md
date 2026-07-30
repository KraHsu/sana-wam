# AR Paired-Noise Cache-Feedback Evaluation

This protocol evaluates two action-cache feedback mechanisms without changing
the AR architecture, SANA backbone, checkpoint, denoising schedule, observation
cadence, or RoboTwin task limits.

## Arms

- Control: `configs/deploy_ar_lownoise_paired_predicted.yaml`
- Predicted-content cache refresh:
  `configs/deploy_ar_lownoise_paired_reencode_predicted.yaml`
- Treatment: `configs/deploy_ar_lownoise_paired_measured.yaml`

The files differ only in `inference.cache_feedback_mode`. All use
`episode_noise_mode: paired`, base seed `20260715`, and `history_len: 113`.
The canonical baseline config remains ambient/predicted and is not modified.

## Noise protocol

The RoboTwin wrapper captures the accepted scene seed passed to `setup_demo`.
The client sends a retry-stable reset attempt key plus a separate noise-pair key.
The attempt key includes a client-session nonce and episode index for HTTP
idempotency. The pair key contains only task, task config, and RoboTwin scene
seed. The server derives a domain-separated SHA-256 model-noise seed from the
pair key. Therefore:

- the same scene gets the same model-noise stream in both arms;
- different scenes get different streams;
- early termination in one arm cannot shift later episodes' streams;
- retrying the same named reset is an idempotent no-op;
- retrying a name with a different explicit seed is an HTTP 409 conflict.

Legacy empty reset payloads remain non-idempotent and preserve the old behavior.

## Measured feedback

In measured mode, the next generation consumes the last complete achieved
proprio/EEF trajectory, normalizes it with the checkpoint action normalizer, and
re-runs only the t=0 ActionDiT clean path for the previous action frame. It
replaces that frame's per-layer `(S, z)` cache state transactionally. It does not
sample noise, add an AR frame, or advance the AR step counter.

The strict experiment config sets `cache_feedback_fallback: error`; any missing,
short, malformed, or non-finite trajectory aborts instead of silently reverting
to predicted feedback.

## Reencoded predicted feedback

In `reencode_predicted` mode, the next generation rebuilds the previous action
frame after the real observation has replaced the predicted video ancestry. The
action content is the exact normalized output saved by the source generation;
it is never reconstructed through an unnormalize/normalize round trip. The
source generation's text context, mask, and action-proprio embedding are also
reused. This isolates cache ancestry from measured-action content.

The bootstrap commit runs before the next observation, while steady commits run
after the current real observation. Cache, RNG, pending feedback, AR step, and
counters are restored on failure. A successful t=0 commit consumes no RNG.

## Telemetry

Server telemetry contains per-step observed proprio, previous command tracking
error, returned 20D command, grippers, chunk offset, AR/cache metadata, noise
identity, and one atomically-written compressed NPZ per generated chunk. Client
telemetry separately records the exact 16D command sent to RoboTwin and a direct
post-command EEF/proprio read without rendering an extra camera observation.
Every record declares telemetry schema version 1 and its action space.

Set unique output directories for each arm. The runner does this automatically:

```bash
ROBOTWIN_TEST_NUM=30 SERVER_GPU=0 SIM_GPU=1 \
  bash scripts/run_ar_lownoise_paired_arm.sh predicted

ROBOTWIN_TEST_NUM=30 SERVER_GPU=0 SIM_GPU=1 \
  bash scripts/run_ar_lownoise_paired_arm.sh measured
```

Use the exact same RoboTwin configuration and episode count for both commands.
Run the control and treatment sequentially when they share GPU or HTTP ports.
The measured and reencode runners require `PAIR_WITH` to point at their
reference run. The runner fails unless episode start/end counts, arm modes,
effective deployment identity,
checkpoint SHA-256, ordered scene/model-noise pairs, chunk NPZ contents, feedback
commit counts, and client/server episode identities all pass validation:

```bash
PAIR_WITH=logs/<predicted-run> RUN_DIR=logs/<measured-run> \
  bash scripts/run_ar_lownoise_paired_arm.sh measured

PAIR_WITH=logs/<fresh-predicted-run> RUN_DIR=logs/<reencode-run> \
  bash scripts/run_ar_lownoise_paired_arm.sh reencode_predicted
```

Before a RoboTwin run, both server arms can be exercised on the recorded episode
with identical model noise (including bootstrap and steady feedback commits):

```bash
bash scripts/run_paired_gpu_smoke.sh
```

Set `TREATMENT_MODE=reencode_predicted` to exercise the predicted-content cache
refresh instead of measured feedback.

## Analysis rule

The old ambient `8/100` and `10/100` runs did not record per-episode model RNG
states, so they are not valid paired controls for this experiment. Compare the
new arms per scene seed with McNemar's exact test and report both marginal rates
with uncertainty. A small point-estimate difference is not evidence of an
improvement without the paired result.

## 2026-07-16 prompt-fixed results

All runs below use prompt manifest SHA-256
`8c7472a2fb5bacecc3b0f8d244b732a149ef41eaa6424e228a88114ac09f5121`
and RoboTwin's upstream 400-step limit for `adjust_bottle/demo_clean`.

| Arm | Run | Success |
| --- | --- | ---: |
| O: original predicted | `logs/paired_predicted_n30` | 8/30 (26.7%) |
| A: fresh predicted | `logs/paired_predicted_replicate_promptfixed_n30` | 9/30 (30.0%) |
| B: reencoded predicted | `logs/paired_reencode_predicted_promptfixed_n30` | 8/30 (26.7%) |
| M: measured | `logs/paired_measured_promptfixed_n30` | 5/30 (16.7%) |

The confirmatory comparison is fresh A versus B. Its paired table has 21 both
fail, one A-only success, zero B-only successes, and eight both success. The
paired risk difference B - A is -3.33 percentage points; the Newcombe paired
method-10 95% CI is [-11.35, +4.23] percentage points and exact McNemar
`p=1.0`. This detects no success-rate difference and does not establish
equivalence.

O versus A is repeatability calibration, not a second treatment comparison. It
has 90% outcome agreement and 10% discordance (Wilson 95% CI 3.5% to 25.6%).
Seven of eight original successes were retained, and success-set Jaccard is
70%. The marginal A - O difference is +3.33 percentage points (Newcombe
method-10 95% CI [-8.29, +15.07], exact McNemar `p=1.0`).

Trajectory identity is much weaker than outcome agreement. O and A diverge in
post-action proprio after the first command in 29/30 episodes. A and B do so in
30/30 episodes, before the first cache-feedback commit at request 29; their
first generated-action/chunk mismatch is request 29 in all episodes. Therefore
A versus B is a prompt/scene/model-noise-paired empirical rollout comparison,
not a deterministic counterfactual. The single A-only outcome cannot be
uniquely attributed to cache reencoding.

Both final verifier invocations report `PASS episodes=30`. B records 372 total
AR generations, 342 commits (exactly `ar_step - 1` per episode), zero fallbacks,
and byte-identical raw/normalized feedback to each previous predicted chunk.
