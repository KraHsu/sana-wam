# LIBERO AR gradient-checkpointing propagation fix — 2026-08-06

## Outcome

The AR training-loss path now preserves the runtime gradient-checkpointing
controls prepared by `Trainer`. Before this correction,
`Base.prepare_inputs()` placed `use_gradient_checkpointing` and
`use_gradient_checkpointing_offload` in the loss inputs, but
`DualSystemARArchitecture.compute_loss()` did not pass them to its model
forwards. The LIBERO baseline therefore requested full-layer checkpointing in
configuration without activating it in the AR loss path.

This is a source and CPU-test closure only. It did not run a GPU, optimizer,
parameter update, training, real-data load, checkpoint load/save, simulator, or
benchmark evaluation. It is not training admission.

## Corrected contract

`DualSystemARArchitecture.compute_loss()` now:

1. Reads both controls without removing them from `inputs`, preserving the
   Phase-6 prepared-input trace contract.
2. Requires each value to be a native Python `bool`.
3. Rejects `use_gradient_checkpointing_offload=true` when
   `use_gradient_checkpointing=false`.
4. Passes both controls to the main on-path forward and to the local-expansion
   plus forward. The expansion center and plus evaluations therefore use the
   same memory policy.
5. Forces both controls off for trajectory state-generation forwards inside
   `torch.no_grad()`, then restores the requested controls for the final
   supervised trajectory forward that participates in backward.

Omitted fields retain the previous `(false, false)` behavior.

## Verification

Focused CPU verification on H200:

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m pytest -q \
  tests/test_ar_compute_loss.py \
  tests/test_sana_local_expansion_compute_loss.py
```

Observed: `21 passed, 14 warnings`.

Broader CPU regression verification:

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m pytest -q \
  tests/test_ar_compute_loss.py \
  tests/test_sana_local_expansion_compute_loss.py \
  tests/test_sana_ar_padding_semantics.py \
  tests/test_sana_local_expansion_loss.py \
  tests/test_action_nr_compute_loss.py \
  tests/test_ar_continuous_timestep_contract.py \
  tests/test_phase6_common_row_rng.py \
  tests/test_phase6_reference_trace.py \
  tests/test_phase6_plan_metadata.py
```

Observed: `118 passed, 14 warnings`. Ruff passed for the changed source and
tests. The warnings are pre-existing dependency/deprecation warnings; no test
failed or skipped because a GPU was unavailable.

## Remaining boundaries

The intended first LIBERO baseline has trajectory loss weights disabled, so
the trajectory-specific issues below are not active in that configuration.
They remain explicit follow-up work rather than silently broadening this fix:

- The inner `mot_checkpoint_mixed_attn` guard does not also check
  `torch.is_grad_enabled()`. A future trajectory-enabled configuration should
  close that guard before relying on no-grad rollout memory behavior.
- The Phase-6 common-trace path pre-appends the proprioception context token;
  trajectory forwards do not yet receive the corresponding
  `phase6_proprio_context_prepared` signal. Phase-6 currently forbids the
  legacy trajectory-loss combination.
- The cross-attention architecture has a similar runtime-control propagation
  gap, but it is outside the current LIBERO self-attention AR path.

The successor pre-update decision is recorded in
[`LIBERO_AR_T1_AND_STATS_PREUPDATE_DECISION_20260806.md`](LIBERO_AR_T1_AND_STATS_PREUPDATE_DECISION_20260806.md).
It selects T1 for the non-formal architecture update smoke while retaining the
strict production-stats work for the formal-training boundary.

Before formal LIBERO training, the project still must:

1. Materialize and hash immutable training-split-only action/state statistics
   at the production path.
2. Run a separately authorized, fresh-root, single-GPU forward/backward/update
   smoke. Formal training and benchmark evaluation remain separate decisions.
