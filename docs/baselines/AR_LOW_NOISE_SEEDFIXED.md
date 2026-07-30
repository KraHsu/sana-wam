# AR low-noise seed-fixed reference baseline

This directory establishes one evidence-backed reference for the SANA AR line.
The machine-readable source of truth is
`docs/baselines/ar_lownoise_seedfixed.manifest.yaml`.

## Historical result

- Task: RoboTwin `adjust_bottle`, `demo_clean` / `clean_50`, aloha-agilex.
- Checkpoint: step 12000, 11,689,906,174 bytes.
- Deploy: 10 video steps followed by 10 action steps, ambient init noise.
- Result: 8 successes in 86 completed episodes, or 9.3%.
- The run requested 100 episodes but stopped at 86 when the wrapper polling cap
  expired. It must not be reported as an n=100 result.

The historical 25/100 result used a fixed per-episode noise trajectory. It is
superseded by this seed-fixed measurement.

## Native result

The sana-wam deployment completed all 100 requested RoboTwin episodes on
2026-07-14 using the exact checkpoint and ambient model noise:

- Result: 8/100, or 8.0%, over 100 unique expert-valid seeds from 100001 to
  100140.
- Execution: RoboTwin's upstream `adjust_bottle` step limit of 400; every
  repository-side step-limit override was disabled.
- Integrity: exit code 0, 8 success markers, 92 failure markers, 100 non-empty
  episode videos, and no evaluation-request, inference, CUDA, or simulator error.
- Comparison: native minus historical is -1.3 percentage points. The Wilson
  95% intervals are 4.1%-15.0% (native) and 4.8%-17.3% (historical); a two-sided
  Fisher exact test gives p=0.798. There is no detectable difference at this
  sample size, which is not the same as proving statistical equivalence.

The native server processed 37,585 successful `/predict` requests. It was bound
to `0.0.0.0` during this run and received seven unrelated scanner requests to
unknown paths, all answered with 404; every `/predict` and `/reset` came from
localhost. The canonical deploy overlay now defaults to `127.0.0.1`.

### History-length follow-up

The canonical July-compatible policy retained only 10 simulator observations,
while the training clip spans 113 raw frames and samples 29 frames at stride 4.
Consequently, a steady-state deploy clip contained 26 left-padded copies and only
three recent time points. A single-variable follow-up raised `policy.history_len`
to 113, restoring all 29 real cadence samples by simulator step 112.

An initial 30-episode screen scored 5/30 versus 2/30 for the canonical run's same
scene prefix, but this did not reproduce on a fresh server: the confirmation run's
first 30 episodes scored 2/30. The complete confirmation finished 10/100 (10.0%)
versus the canonical 8/100 (8.0%). Its Wilson 95% interval was 5.5%-17.4%; Fisher's
two-sided p=0.806 and paired McNemar exact p=0.791 show no detectable improvement.
Only two success seeds overlapped, so ambient diffusion-noise variance dominates
the two-point rate difference. The cadence mismatch is real, but correcting it
alone is not a demonstrated success-rate lever for this checkpoint.

## What is now native

The safe, independently testable parts of the reference recipe are ported into
sana-wam:

1. `ActionScheduler` supports `none`, `bsmntw`, and gamma-2 `low_noise` modes.
2. AR action loss consumes the selected per-timestep weights; its default remains
   `none` for backward compatibility.
3. Deploy engines use ambient entropy when `inference.seed` is null, while an
   explicit integer remains deterministic for debugging.
4. AR bootstrap applies only to chunk 0; later chunks use the steady-state
   imagination-conditioned branch.
5. The SANA builder merges the complete 2B preset with the recipe's incremental
   model kwargs, then constructs the learnable feature map and window-attention
   graft on all 20 blocks.
6. The split AR path applies the scaled window graft in the same residual
   location as the recorded implementation, and action-only proprio dropout
   keeps video proprio while dropping the action branch per sample in training.
7. Temporal GLUMBConv inputs are materialized as contiguous NCHW tensors before
   bf16 cuDNN convolution, matching the recorded GPU alignment guard.

## Reproduction boundary

The exact 1741-key checkpoint passes `load_from_checkpoint_dir` with
`strict=True` on CPU through the native config/build path. A 29-observation H200
smoke then covers both the chunk-0 action-only branch and the steady-state
video-then-action branch; all 29x20 physical actions are finite.

A deterministic differential runs the current external OpenWAM and sana-wam in
separate processes with the same Python environment, H200, checkpoint, recorded
observations, preprocessing, and seed. Processed images, states, prompts,
scheduler sigmas, both complete 28x20 action chunks, and all 29 returned physical
actions are bitwise identical. This is strong current-tree inference evidence,
but the source OpenWAM worktree was dirty and the checkpoint has no source
metadata, so it cannot prove parity with an unavailable July binary.

The run directory is still current-host-only rather than portable: model
construction needs the absolute `/DATA` SANA, Wan VAE, and Gemma assets recorded
in `config.yaml`.

Native training-loop parity is not claimed. The compact sana-wam trainer differs
from the July Accelerate/DeepSpeed run in its dream-parameter LR group, step and
LR-scheduler accounting, gradient-clipping config, seed semantics, and
`training_strategy` schema. The byte-identical reference config remains evidence
of the historical run, not a ready-to-launch native retraining config.

## Effective July deploy semantics

The seed-fixed script used generic `configs/deploy.yaml`, not the AR-specific
deploy YAML. Several logged/configured labels therefore need interpretation:

- Policy history was the `WAMPolicy` default of 10, not 128. Execution was
  greedy (`execute_horizon=null`); temporal ensembling defaulted on but was
  inactive.
- `schedule_type=sync` was only a generic log label. Chunk 0 denoised action
  only; later chunks completed 10 video steps and then 10 action steps.
- `inference.shift=5.0` was not consumed by the AR engine. Effective scheduler
  shifts were video 3.0 and action 5.0.

The July model also contains an intrinsic train/deploy graft mismatch. Training
uses four-frame temporal windows over the duplicated latent clip, while deploy
passes one two-frame chunk at a time; the temporal graft consequently reduces
to per-frame spatial attention at deployment. This is part of the recorded
baseline, not a missing native port.

The complete native rerun used the same effective denoise order, shifts,
10-frame policy history, greedy execution, and ambient model noise. Unlike the
older wrapper, it had no five-hour polling cap and therefore reached all 100
episodes. `benchmarks/robotwin/step_limits.yml` now contains comments only, so
future tasks use RoboTwin's own default limits unless an override is explicitly
supplied.

## Verification

Run the CPU regression gate:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider \
  tests/test_action_scheduler_weighting.py \
  tests/test_ar_compute_loss.py \
  tests/test_ar_engine.py \
  tests/test_deploy_seed_entropy.py \
  tests/test_ar_proprio_action_dropout.py \
  tests/test_sana_learnable_feature_map.py \
  tests/test_sana_window_flash_split.py \
  tests/test_sana_pipeline_preset_merge.py
```

Run the exact current-host loader check without a GPU:

```bash
.venv/bin/python - <<'PY'
from sana_wam.deploy.model_loader import load_from_checkpoint_dir

run = "/home/zch/wuji-openwam-dev/sandbox/sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25"
_, model = load_from_checkpoint_dir(
    run,
    device="cpu",
    ckpt_name="checkpoint_step_12000.safetensors",
)
assert len(model.state_dict()) == 1741
print("strict native load: ok")
PY
```

Run the two-branch H200 smoke:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_ar_lownoise_gpu.py --device cuda:0
```

Run the deterministic current-tree differential, using the same physical GPU
for the two sequential model processes:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
  /home/zch/wuji-openwam-dev/.venv/bin/python \
  scripts/diff_openwam_sanawam_gpu.py --mode external --device cuda:0
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
  /home/zch/wuji-openwam-dev/.venv/bin/python \
  scripts/diff_openwam_sanawam_gpu.py --mode native --device cuda:0
/home/zch/wuji-openwam-dev/.venv/bin/python \
  scripts/diff_openwam_sanawam_gpu.py --mode compare --require-bitwise
```

The pinned native deploy entrypoint is:

```bash
.venv/bin/python scripts/deploy.py \
  --ckpt-dir /home/zch/wuji-openwam-dev/sandbox/sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25 \
  --deploy-config configs/baselines/deploy_ar_lownoise_seedfixed.yaml \
  --device cuda:0
```

With that server healthy, the complete local RoboTwin run is launched by:

```bash
RUN_DIR="$PWD/logs/native_ar_lownoise_$(date -u +%Y%m%dT%H%M%SZ)" \
HTTP_PORT=8848 SIM_GPU=1 bash scripts/eval_ar_lownoise_seedfixed.sh
```

Verify the external checkpoint identity without copying another 11 GB:

```bash
sha256sum /home/zch/wuji-openwam-dev/sandbox/sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25/checkpoint_step_12000.safetensors
```

Expected SHA-256:
`aea6b58c9107aeeeffba561f39ffedfc3f8fd8420a4abfcab9038c8cdd4a129d`.
