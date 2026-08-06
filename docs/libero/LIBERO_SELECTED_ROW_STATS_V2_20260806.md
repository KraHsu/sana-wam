# LIBERO selected-row statistics v2 — 2026-08-06

## Purpose

The existing metadata-derived artifact
`sana_wam_libero_all_minmax_stats.npy` remains smoke-only. Its suite aggregates
include the excluded Goal episode 82 and each episode's final action row, which
has no corresponding next observation under the training alignment. It also
does not bind the split, seed, exclusion set, or selected Parquet identities.

Formal LIBERO statistics therefore use a separate selected-row v2 producer and
schema. The v1 metadata builder remains byte- and behavior-compatible for
non-formal data/model smokes; v1 is rejected whenever the launcher requests
materialized training statistics.

## Shared population contract

Episode discovery, exclude-before-split selection, and six-column Parquet
identity validation now live in one shared module used by both the production
dataset and the v2 stats producer. This removes an independent selector that
could silently drift from training.

The v2 population is defined as:

- roots sorted by dataset basename;
- configured exclusions applied before the deterministic per-suite split;
- state rows exactly `[0,L)` for every selected episode;
- supervised action rows exactly `[0,L-1)`;
- no padding rows, repeated dataset copies, or overlapping-window multiplicity;
- `mean`, population `std` with a `1e-3` floor, `min`, `max`, and linear
  `q01`/`q99`, computed in FP64 and stored in FP32.

The producer also proves that the configured training windows cover the full
unique supervised action interval. Every selected Parquet is checked for state
and action shape/finite values, fixed task and episode identity, contiguous
frame indices, and exact 20 Hz timestamps.

## Population evidence

The embedded canonical population manifest binds:

- the population-defining selection projection: roots, split, `val_ratio`,
  seed, exclusions, alignment, and row policies;
- all four metadata file SHA256/size pins per suite;
- every selected episode's index, length, task, relative Parquet path, size,
  SHA256, and state/action row counts;
- excluded episode records and per-suite/global count equations.

The manifest carries its own canonical JSON SHA256. A materialized training
config must independently pin both:

```yaml
training:
  action_stats_sha256: <artifact SHA256>
  action_stats_population_sha256: <embedded population manifest SHA256>
```

Before Trainer/model/GPU construction, the formal launcher verifies the
artifact SHA, requires v2, verifies both manifest digests, pins the exact
production roots/config/counts, reproduces the selection contract from the
live config, and rebuilds both the metadata/Parquet identity manifest and all
numeric arrays from live rows for exact comparison.

That full numeric pass is performed once by the launcher. Dataset construction
then independently re-hashes and compares the live population manifest without
performing a second numeric reduction over all selected rows.

## Producer

The dedicated producer accepts one config and takes its output path only from
`dataloader.action_stats_path`. The exact one-time command below was run from
source commit `a7350bc1a7d8b5d586df0202a94a950207c7ce4f` while both source pins were
null:

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  scripts/build_libero_selected_stats.py \
  --config configs/benchmarks/libero/train_libero_ar_baseline.yaml
```

It refuses non-null source SHA pins, non-production roots/config, symlinked
inputs, an existing destination, or any population whose exact production
counts differ from 4 suites / 1,693 source episodes / 1,692 selected episodes /
1 excluded episode / 273,465 source state rows / 273,336 selected state rows /
271,644 action rows. Every gate runs before publication, so a rejected build
does not occupy the reserved path. Publication uses a temporary file,
file/directory fsync, exclusive hard-link creation, and mode `0444`.
The checked-in config is now pinned and the target exists, so repeating that
command intentionally fails closed.

## CPU synthetic verification

The synthetic fixture uses four canonical suite names. Goal episode 82 contains
only sentinel `999` values, while every selected episode's final action is a
sentinel `777`. The tests prove that neither sentinel enters the v2 statistics,
the final state does enter, root order is deterministic, repeat/window overlap
does not reweight unique rows, and bad Parquet identity, unknown exclusions,
seed drift, or byte-level Parquet drift fails closed.

The focused selected-stats/dataset/contract suite passed 24 tests. The broader
Trainer, freeze, mini-AR, and LIBERO suite passed 124 tests with 14 pre-existing
dependency warnings. Compilation, Ruff check, and Ruff formatting checks
passed with CUDA hidden.

## CPU real-data materialization

The authorized CPU-only materialization completed once on 2026-08-06:

- artifact: `/DATA/share/LIBERO/sana_wam_libero_train_all4_excl_goal82_stats_v2.npy`;
- mode/owner/size: `0444`, `zch:sharegrp`, 351,803 bytes;
- artifact SHA256:
  `e5d985903539c1767a246e63c629b214c47c395d2000dee44122b8a0672253c7`;
- population-manifest SHA256:
  `7ed9772facf261299022e55169bcdaaa49fe3a7a20e0057e08cf419b5e584146`;
- selection-contract SHA256:
  `bff79d8a8aa5e111b3cd94dc79117bdde9e3c605fe254cb968a942e5a2e305b5`;
- population: 4 suites, 1,693 source episodes, 1,692 selected episodes,
  exactly 1 excluded episode, 273,465 source state rows, 273,336 selected
  state rows, and 271,644 supervised action rows.

After the two SHA pins were written into the training config, the formal
pre-Trainer CPU preflight passed, including live manifest reconstruction and
exact recomputation of every numeric statistics array. No GPU, model, Trainer,
optimizer, training, checkpoint, simulator, or evaluation path was executed.

## Next experiment

The next architecture experiment is a separately authorized bounded LIBERO T2
multi-step learnability screen on a fixed small slice, with a fresh root and
checkpoint load/save disabled. It is not formal training or benchmark
evaluation.
