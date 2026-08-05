# Phase-C production-path mini checkpoint

This directory records the additive H200 CPU/synthetic implementation
checkpoint authorized on 2026-08-02.  The executable harness is
production-shaped: it uses the repository's typed layout, SANA public
`run_chunk` adapter, actual tiny `ActionDiT`, `CACHNumericalCore`, legacy
`list[10]` codec, Stage2B read-view/staging-context types, committed-action
history, and a single-pointer ephemeral state transition.

The final CPU execution was recorded on 2026-08-03 in Asia/Shanghai
(`2026-08-02T17:18:46Z`); this does not change the authorization date.

The fixed synthetic episode deliberately separates noisy denoise inputs from
clean paired targets and uses nonzero denoise timesteps.  The four cache fields
also retain the pinned small-shape codec contracts: `main_s_kv [B,1,C,C]`,
`main_s_z [B,1,C,1]`, GDN short-conv left context `[B,3,C]`, and FFN temporal
context `[B,C,1,1]` for the 1x1 spatial mini.

It is not a production model admission.  The recurrent numerical backend is a
small pure-Torch proxy, the ordinary `build_architecture` CACH branch remains
fail-closed, and the Phase-C state owner is an additive in-memory interface.
Consequently the stricter Global Stage-3 Section 11 requirement that only
tensor sizes change and that the exact future G3 public dispatcher/cache owner
be reused is still blocked.

Artifacts:

- `PRODUCTION_PATH_MINI_C0_C8_CLOSURE_20260802.md`: implementation and C0--C8
  evidence/gap record;
- `PHASE_C_LIGHTWEIGHT_TEST_REPORT.json`: historical H200 CPU test
  attestation, with zero retained raw execution evidence;
- `PHASE_C_SOURCE_FILES.txt`: exact additive source-member order;
- `SOURCE_MANIFEST.json`: caller-pinnable byte inventory, excluding itself to
  avoid self-reference;
- `scripts/verify_cach_phase_c_production_path_mini.py`: read-only, torch-free
  source/report verifier;
- `src/sana_wam/model/cach_production_path_mini.py`: bounded harness;
- `tests/test_cach_production_path_mini.py`: hard CPU checks with no skips.

The final verifier must be called with an out-of-band manifest digest:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  scripts/verify_cach_phase_c_production_path_mini.py \
  --manifest docs/cach_sana_wam/global_stage3/phase_c_production_path_mini/SOURCE_MANIFEST.json \
  --expected-manifest-sha256 <caller-pinned-sha256>
```

The verifier does not replay tests or execute a model.  A valid result keeps
all of the following unchanged:

- global `GATE-S0=NOT_CLAIMED`;
- global `GATE-S1=NOT_CLAIMED`;
- global `GATE-S2=BLOCKED_NOT_CLAIMED`;
- Global Stage 3 and Phase D `NOT_AUTHORIZED`;
- transitive Python/native/dynamic-import runtime closure `false`;
- global C0--C8 and strict Section 11 closure `false`;
- GPU, full 2B, checkpoint, data, optimizer, training, evaluation,
  deployment, capture, formal-root and scientific authority `false`.

No source bundle, `/DATA` result root, retained test evidence, checkpoint, or
formal admission root belongs to this checkpoint.
