# LIBERO T9 Four-Phase Latin-Square Core Screen

Status: **complete / frozen common-position effect supported**

Date: 2026-08-06

## 1. Question and minimum intervention

T7 and T8 cover the first two cyclic phases of the same four-suite update
cycle. T9 adds only the two missing phases:

```text
arm C = [A2 Goal, A3 LIBERO-10, A0 Spatial, A1 Object] x 5
arm D = [A3 LIBERO-10, A0 Spatial, A1 Object, A2 Goal] x 5
```

Together with the frozen terminal observations from:

```text
arm A / T7 = [A0, A1, A2, A3] x 5
arm B / T8 = [A1, A2, A3, A0] x 5
```

the four arms form a complete 4 x 4 cyclic Latin square: every suite is the
newest, second-newest, second-oldest, and oldest update exactly once. The
scientific question is deliberately narrow:

> Is terminal loss degradation primarily a common function of the number of
> trailing updates, or does the response depend materially on update/target
> identity within this fixed cyclic order?

T9 does not introduce another architecture, sample, optimizer, seed, dose, or
selection rule. C and D are separate fresh-initialization arms. Neither arm is
allowed to initialize from A, B, or the other T9 arm.

## 2. Reused experimental system

C and D reuse the exact T7/T8 experimental system without substitution:

- T7 arm-A source/result/runner pins:
  `d19109a2314f8e7186571afed6b4acd816d7cfab` /
  `9f5181a30cd0d6b676f09315244f7560cd3902004f5f17ca833330e33cb44d3a` /
  `b6e764bf3d3566bf3d1fe5e2e3802dafef691f6a0162568337b49ca1a183310b`;
- T8 arm-B source/result/runner pins:
  `9cd1c490d14b3c2437e82225ee8cfdf58646837e` /
  `bfdb852e14a5bd9b1c8e776be9f4ff108899eae65d557cebe42b06f1991b0a18` /
  `9ec74d83c8443d78d2ba765dd766f82a96cccef894ae63d08cf9e2a8dbf97669`;
- update samples A0 Spatial, A1 Object, A2 Goal, and A3 LIBERO-10;
- update-free same-task probes H0, H1, H2, and H3;
- T7 manifests: update `5199c87af876c437ecec35b57da3e118a2924c44a9bb454c9846e7c0ca968aa6`,
  eligible `0b6bfe4c51e8b9e2ac3f538bac7853b5e1e871ea96dda5df6976ee27b5b25370`,
  selected `e969e76e4bcb8a3c2b0b3fffff35e0ec10c478e45cb6535b7bd91061d3a6a7ee`;
- config `configs/benchmarks/libero/train_libero_ar_baseline.yaml`, SHA256
  `5df45965d6de3a32c5154a8bb4c41d29d20113bbe8908c1a98c8c63a7bff4a45`;
- production-shaped `DualSystemARArchitecture`, frozen SANA video backbone,
  and trainable roots `action_backbone`, `proprio_encoder`,
  `proprio_video_embed`, and `proprio_action_embed`;
- fresh initialization seed `20260806` in each arm;
- recipe seed `20260826` before every forward;
- action loss weight 1.0 and video loss weight 0.0;
- one persistent FP32-master AdamW per arm, learning rate `1e-4`, betas
  `(0.9, 0.95)`, weight decay 0, and global gradient clip 1.0;
- five updates per A sample, 20 updates total, with a BF16 projection from
  the FP32 masters after every step.

Each arm must reproduce the frozen pre-action-loss vector before its first
update, under the existing T7/T8 tolerance:

```json
{
  "A0": 13.679718971252441,
  "A1": 14.696584701538086,
  "A2": 12.06648063659668,
  "A3": 20.43115234375,
  "H0": 13.379679679870605,
  "H1": 14.698094367980957,
  "H2": 12.026171684265137,
  "H3": 20.885684967041016
}
```

The full T7/T8 asset, task-text, prompt, statistics, normalization, base-model,
Sana gitlink, and loss-recipe pins remain inherited inputs to the runner. This
document does not redefine them or permit a new asset.

## 3. Exact C and D schedules

Preparation and pre-probe order in both arms is exactly:

```text
A0, A1, A2, A3, H0, H1, H2, H3
```

Arm C runs `[A2,A3,A0,A1] x 5` and inserts the following update-free probes:

| Step just completed | Own-fifth phase probes | Trailing updates before terminal |
|---:|---|---:|
| 17 | A2, H2 | 3 |
| 18 | A3, H3 | 2 |
| 19 | A0, H0 | 1 |
| 20 | A1, H1 | 0 |

Arm D runs `[A3,A0,A1,A2] x 5` and inserts:

| Step just completed | Own-fifth phase probes | Trailing updates before terminal |
|---:|---|---:|
| 17 | A3, H3 | 3 |
| 18 | A0, H0 | 2 |
| 19 | A1, H1 | 1 |
| 20 | A2, H2 | 0 |

After step 20 and its own-fifth pair, each arm takes the unified terminal block
in exact order `A0,A1,A2,A3,H0,H1,H2,H3`, with no intervening update.

Each arm therefore has exactly:

- 8 preparation calls;
- 44 forwards: 8 pre, 20 update, 8 own-fifth phase, and 8 terminal;
- 20 backwards and 20 optimizer steps;
- five update forwards per A label and zero updates from every H label.

The step-20 A/H phase pair must exactly reproduce at terminal measurement,
because no parameter-changing operation occurs between them. Phase-to-terminal
overwrite ratios for all four pairs are required diagnostics, but they are not
the combined Latin-square classifier: T7 arm A did not record post-fifth-update
phase probes.

## 4. Per-arm result boundary

C and D use one generic frozen runner with an enumerated `C` or `D` arm choice,
but execute separately with:

- a fresh model and empty optimizer state;
- a distinct never-used nonce and immutable root;
- an independently pinned source commit, runner SHA256, configuration SHA256,
  and result SHA256;
- no SANA-WAM training-checkpoint load/save and no state transfer between
  arms.

A valid C arm reports `T9_LATIN_ARM_C_VALID`; a valid D arm reports
`T9_LATIN_ARM_D_VALID`. Neither may emit a common-position, interaction, or
architecture GO verdict by itself. The arm report includes, for each suite:

```text
rA_phase, rH_phase, q_phase
rA_terminal, rH_terminal, q_terminal
overwrite_penalty = q_terminal / q_phase
trailing_updates in {0,1,2,3}
```

where each A/H ratio divides the corresponding action loss by that arm's own
pre-action loss, and `q = (rA + rH) / 2` using unrounded float64 values. It also
records the exact schedule, all losses, counts, recipe signatures, and optimizer
telemetry needed to establish that the arm followed the fixed experiment.

The arm roots are fixed to:

```text
/DATA/share/sana_wam_libero_nonformal_screens/t9/<source-commit-12>/libero-t9-arm-c-latin-fixed20-<nonce>
/DATA/share/sana_wam_libero_nonformal_screens/t9/<source-commit-12>/libero-t9-arm-d-latin-fixed20-<nonce>
```

An invalid C or D arm has no scientific status and cannot contribute to the
combined result. There is no result-dependent extension or automatic rerun.

### 4.1 Frozen C/D executions

Both arms executed exactly once from source commit
`f73a7950eded2787ad57b26523da686dc5722512` with arm-runner SHA256
`d0d4cacf35d4116db3a8c166ffcdc92578a28f5037614a960e22b4ccbcc4a84c`
on physical GPU 0 / UUID
`GPU-1ec28cfb-f501-23f3-f865-275a744ca053`. Each reproduced the frozen
starting losses and recorded exactly 8 preparations, 44 forwards, 24
measurements, 20 backwards, and 20 optimizer steps. Neither loaded or saved a
SANA-WAM training checkpoint, ran a simulator, or performed benchmark/formal
evaluation.

- C nonce `5b1425bc07c1162fe6eb0f04164f9b9e`, root
  `/DATA/share/sana_wam_libero_nonformal_screens/t9/f73a7950eded/libero-t9-arm-c-latin-fixed20-5b1425bc07c1162fe6eb0f04164f9b9e`,
  RESULT SHA256
  `213d61f4a63f42983e4a42db6db9410279cc6a898bffcf11f157c560adf39771`,
  verdict `T9_LATIN_ARM_C_VALID`;
- D nonce `741c81a12b6dda6592d9cc89b1b78765`, root
  `/DATA/share/sana_wam_libero_nonformal_screens/t9/f73a7950eded/libero-t9-arm-d-latin-fixed20-741c81a12b6dda6592d9cc89b1b78765`,
  RESULT SHA256
  `7326bff58efe3b5d07e83db96aefe539f5da36e30ef453d3404166dade38aca0`,
  verdict `T9_LATIN_ARM_D_VALID`.

The frozen per-suite diagnostics are:

| Arm | Suite | k | q phase | q terminal | terminal / phase |
|---|---|---:|---:|---:|---:|
| C | Spatial | 1 | 0.1613527581 | 0.8520549099 | 5.2806962814 |
| C | Object | 0 | 0.4574664142 | 0.4574664142 | 1.0000000000 |
| C | Goal | 3 | 2.0848917111 | 1.1307392171 | 0.5423491355 |
| C | LIBERO-10 | 2 | 0.4660701868 | 0.9862132306 | 2.1160186995 |
| D | Spatial | 2 | 3.2107157317 | 0.3584876270 | 0.1116534932 |
| D | Object | 1 | 1.2467110249 | 0.2303102918 | 0.1847343027 |
| D | Goal | 0 | 0.3174111524 | 0.3174111524 | 1.0000000000 |
| D | LIBERO-10 | 3 | 0.5941060496 | 0.4552422189 | 0.7662642371 |

These phase-overwrite values remain diagnostics only. The frozen combined
classifier below uses terminal/pre losses from all four arms.

## 5. Complete Latin-square assembly

The combined classifier runs on CPU only after T7, T8, C, and D results are
all frozen and their complete source/result pins have been supplied. Arm B is
the exact frozen T8 result pinned in section 2, with verdict
`T8_PHASE_ROTATED_MIXED_INCONCLUSIVE`; T9 does not condition inclusion on that
conclusion. C and D must respectively be `T9_LATIN_ARM_C_VALID` and
`T9_LATIN_ARM_D_VALID`.

For suite `i`, let `k` be the number of optimizer updates after its own fifth
update (`k=0` newest through `k=3` oldest). The frozen cell mapping is:

| Suite | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| A0 / Spatial | B | C | D | A |
| A1 / Object | C | D | A | B |
| A2 / Goal | D | A | B | C |
| A3 / LIBERO-10 | A | B | C | D |

For every cell, compute only from that arm's terminal and pre action losses:

```text
rA_i(k) = A_terminal_i(k) / A_pre_i(k)
rH_i(k) = H_terminal_i(k) / H_pre_i(k)
q_i(k)  = (rA_i(k) + rH_i(k)) / 2
```

All inputs must be finite and strictly positive. The aggregate must retain the
four frozen source/result identities behind every assembled row and cell.

## 6. Minimal frozen combined verdict

For each suite independently, compute:

```text
rho_i = Spearman([q_i(0), q_i(1), q_i(2), q_i(3)], [0,1,2,3])
endpoint_i = q_i(3) > q_i(0)
strong_i = rho_i >= 0.8
```

Spearman is Pearson correlation over ascending average ranks; ties receive
their average rank. A zero-variance vector produces JSON `null` and
`strong_i = false`. The endpoint comparison is strict with no epsilon.
The `0.8` threshold is inclusive; at most `1e-12` arithmetic tolerance may be
used when comparing a finite computed rho to the frozen threshold.

Then define:

```text
endpoint_count = sum(endpoint_i for all 4 suites)
strong_count   = sum(strong_i for all 4 suites)
```

The ordered classification is exactly:

1. If `endpoint_count == 4` and `strong_count >= 3`, emit
   `T9_COMMON_POSITION_EFFECT_SUPPORTED`.
2. Else if `endpoint_count == 4`, emit
   `T9_POSITION_WITH_IDENTITY_INTERACTION`.
3. Otherwise emit `T9_NO_COMMON_POSITION_EFFECT`.

This is intentionally the complete decision rule. Medians, ANOVA shares,
phase overwrite thresholds, best-arm selection, and post-hoc subgroup rules
are diagnostics at most and cannot change the verdict.

The first verdict means all four targets are worse at the oldest endpoint than
at the newest endpoint and at least three show a strong graded rank trend over
the four terminal positions. The second means the shared endpoint direction
exists, but at least two suites fail that graded trend, so the intervening
response is operationally identity-dependent within the fixed cycle. The
third means even the all-suite endpoint direction is absent. It does not prove
that no ordering effect exists.

Because all arms preserve the same directed cyclic adjacency, the interaction
verdict cannot identify a single harmful successor update or generalize to
arbitrary permutations. It identifies target/update-identity dependence only
within this four-phase cyclic design.

## 7. Required combined summary

The standard-library-only CPU harness is
`scripts/summarize_libero_ar_t9_latin_square.py`. It accepts the exact C/D
RESULT paths and SHA256 values, the aggregator's own expected source commit
and runner SHA256, plus one fresh nonce/root. The T7/T8 roots and pins, and the
completed C/D arm source commit and runner SHA256, are constants rather than
caller-selectable alternatives. Its only permitted root shape is:

```text
/DATA/share/sana_wam_libero_nonformal_screens/t9_aggregate/<source-commit-12>/libero-t9-latin-square-combined-<nonce>
```

The root is exclusive and terminal: success or failure writes one canonical
`RESULT.json`, then freezes files to `0444` and directories to `0555`. Input
roots must themselves be non-symlink, frozen, single-RESULT roots with
canonical JSON. C and D must bind the same T9 source/arm-runner/config while
using distinct roots and nonces.

The CPU aggregate records:

- the pinned T7, T8, C, and D result/source identities;
- the complete four-suite by four-position `rA`, `rH`, and `q` matrices;
- the arm-to-cell mapping above;
- each suite's `[q(0),q(1),q(2),q(3)]`, average ranks, `rho`, `endpoint`, and
  `strong` flag;
- `endpoint_count`, `strong_count`, and exactly one typed verdict;
- C/D phase overwrite diagnostics, clearly excluded from the combined rule.

No GPU, model construction, parameter update, checkpoint, additional data, or
new measurement is needed for aggregation.

### 7.1 Frozen combined result

The CPU aggregator executed once from source commit
`b7cded5fd9cf83ffabeba18a7e352b1cb4438b66`, runner SHA256
`b884ef028a48ee37afda70e730c431db48ed30f62be7f59cb2c665b6072ffc0d`,
and nonce `ab99dd8758ef03bb191d5fb48f3b9fbc`. Its immutable root is:

```text
/DATA/share/sana_wam_libero_nonformal_screens/t9_aggregate/b7cded5fd9cf/libero-t9-latin-square-combined-ab99dd8758ef03bb191d5fb48f3b9fbc
```

The canonical, single-file `RESULT.json` SHA256 is
`0cfc53b820939123de4bc2a626380495878d4d5c9a37a0be9be667173254149d`;
the root is `0555` and the result is `0444`. The aggregator strongly bound all
four input RESULT SHA256 values, recomputed every ratio from raw pre/terminal
losses, and emitted:

| Suite | q(0) | q(1) | q(2) | q(3) | average ranks | rho | q(3)>q(0) | strong |
|---|---:|---:|---:|---:|---|---:|---|---|
| Spatial | 0.645719 | 0.852055 | 0.358488 | 1.085568 | [2,3,1,4] | 0.4 | yes | no |
| Object | 0.457466 | 0.230310 | 0.761653 | 1.589239 | [2,1,3,4] | 0.8 | yes | yes |
| Goal | 0.317411 | 0.500110 | 0.598033 | 1.130739 | [1,2,3,4] | 1.0 | yes | yes |
| LIBERO-10 | 0.137864 | 0.252070 | 0.986213 | 0.455242 | [1,2,4,3] | 0.8 | yes | yes |

Therefore `endpoint_count=4`, `strong_count=3`, and the frozen typed verdict
is `T9_COMMON_POSITION_EFFECT_SUPPORTED`. The result supports a common
retention penalty as more other-suite updates follow a suite's fifth update.
It does not claim a perfectly monotonic response for every suite: Spatial is
strongly non-monotonic, and LIBERO-10 also peaks at `k=2` rather than `k=3`.

## 8. Scope

T9 is a deterministic, single-seed, one-sample-pair-per-suite, non-formal
loss-space architecture diagnostic. It is not LIBERO rollout evaluation,
success-rate evidence, formal training, checkpoint production, admission, or
deployment. Its next decision is whether a common terminal-position effect is
strong enough to motivate changing the update strategy, or whether
identity-sensitive interference requires a different core experiment.

C and D and the CPU aggregate now exist only as the immutable non-formal roots
pinned in sections 4.1 and 7.1. The verdict motivates a new update-retention
hypothesis; it does not authorize formal training or benchmark evaluation.
