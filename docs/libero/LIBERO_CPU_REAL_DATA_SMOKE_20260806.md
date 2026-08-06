# LIBERO CPU real-data smoke — 2026-08-06

## Scope

This is a non-formal data-path smoke, not training or benchmark evidence. It
used repository commit `be0933c6a83b95d2300d4cbf347451716d0b1589` on H200,
with `CUDA_VISIBLE_DEVICES=""`. No model, checkpoint, GPU, optimizer, training,
simulator, or evaluation path was constructed.

The only environment change was installing `pyarrow==23.0.1` into the project
`.venv` from `benchmarks/libero/requirements-data.txt`.

## Inputs

- Four fixed H200 LeRobot v2.1 roots from the LIBERO AR baseline template.
- Complete exclusion of
  `libero_goal_no_noops_1.0.0_lerobot:82` because its wrist video is corrupt.
- Temporary metadata-derived min-max stats:
  `333b2cb1e150b451ee1cf6833b1e628ebb50914a466f2cb1fa1808cff8e9f2d0`.
- Fixed sample: `libero_spatial_no_noops_1.0.0_lerobot`, episode 0, start 0.

## Result

The smoke passed on 2026-08-06 UTC:

| Check | Observed |
|---|---|
| Selected episode length | 110 |
| Selected training episodes | 1,692 |
| Generated windows | 68,528 |
| Excluded episodes observed | 1 |
| Alignment | `observation_t_to_action_t` |
| Action tensor | `[112, 7]`; 109 valid steps |
| Proprio sequence | `[113, 8]` |
| Video tensor input | 29 composite frames at `[384, 320, 3]`; 28 valid |
| Spatial episode-0 Parquet SHA256 | `3f875604fad478765549128759edfb33a64b69b7b82decebc9e4f38155f20a8c` |
| Primary decoded RGB frame-0 SHA256 | `0b03b94b96208846f7fdab3a5dcacd0ef8e470d3823f86101d0b7bbb84424f16` |
| Wrist decoded RGB frame-0 SHA256 | `10b9118f9a99d19fcfba4c80a53101975b136fe8453b001fa5f32f2a762429df` |

The raw frame-0 state/action anchors, normalized sample values, Parquet
episode/frame/timestamp columns, masks, and decoded frame hashes all matched.
The post-install CPU regression suite passed `122` tests.

The original temporary canonical result was written read-only at
`/tmp/sana-wam-libero-real-smoke.0MmKLZ1A/RESULT.json`, SHA256
`8ab19045874c3a2832ee1996c6a96e917cf8506a5a35776593052c393ed4c78e`.
That path is intentionally non-formal and may be cleaned by the host.
The checked-in runner then reproduced the same scientific fields at
`2026-08-06T02:24:50.701316+00:00`; its read-only `RESULT_REPRO.json` has
SHA256 `16d4013e8ecf6c2daffd2e000b11a71336e4c4b5eff2bb00ae98f9d05d902fdc`.

## Reproduction

Create a unique temporary directory, build stats with
`scripts/build_libero_stats.py`, then run:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/smoke_libero_data.py \
  --stats /path/to/action_stats.npy \
  --report /unique/path/RESULT.json
```

The runner refuses to overwrite a report and validates the fixed source hashes.

## Remaining boundary

This result admits only the native data path. Before formal training, the
project still needs split/exclusion-derived production stats, a full data-asset
manifest, an exclusive immutable training root, and separate training
admission. The immediate architecture-validation successor is a separately
authorized single-GPU, update-free real-sample model smoke.
