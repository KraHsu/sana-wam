# LIBERO AR FP32 optimizer-master precision closure — 2026-08-06

## Predecessor evidence and precision gap

The frozen one-update result at
`/tmp/sana-wam-libero-t1-one-update-fce4bd9-20260806-a2` proved that all 560
allowlisted tensors across the four action/proprioception roots received finite
nonzero gradients. It used direct BF16 AdamW parameters, however, and the
changed-value probes did not observe a first-step BF16 value change under
`proprio_encoder`.

This is consistent with a sub-ULP update: at common nonzero BF16 weight
magnitudes, an AdamW step near `1e-4` can be smaller than half of one BF16
quantization interval. Treating the projected BF16 value as the optimizer's
authoritative state would discard that residual on every step.

## Corrected optimizer contract

The existing Trainer FP32-master path is enabled for the LIBERO lineage with:

```yaml
training:
  optimizer_master_weights: true
```

The model parameters remain BF16 for forward/backward. There is exactly one
persistent, same-device FP32 master per trainable tensor. After clipping the
BF16 model gradients, Trainer converts them to FP32, updates the masters with
AdamW, and projects each master back to its paired BF16 model tensor. The next
optimizer step starts from the retained FP32 master, so sub-BF16 increments
accumulate instead of being discarded.

The LIBERO preflight now rejects any configuration without FP32 masters. The
one-update runner additionally fixes the expected partition at 560 tensors and
639,653,063 elements, checks exact model/master/optimizer-group identity, and
requires every master to receive a finite nonzero FP32 gradient and a numeric
first-step update. It reports two distinct observations:

- `master_update`: authoritative FP32 optimizer updates; all four trainable
  roots must be represented.
- `projected_bf16_update`: immediately visible BF16 changes after projection;
  an individual root may remain unchanged for one sub-ULP step.

The BF16 projection is checked exactly against `master.to(torch.bfloat16)` for
every pair. Parameter `_version` remains telemetry and is not accepted as
numeric update evidence.

## Scope boundary

This closure does not authorize a training loop, checkpoint creation, simulator
execution, or benchmark evaluation. The smoke creates no optimizer checkpoint.
Before resumable formal training, checkpoint semantics must explicitly retain
the FP32 masters and AdamW state, or require a documented fresh optimizer; that
separate issue does not block a no-save one-step architecture smoke.

## CPU verification

With CUDA hidden on H200, the focused Trainer/LIBERO contract suite passed 53
tests. The broader Trainer, freeze, mini-AR, and complete LIBERO suite passed 97
tests with 14 pre-existing dependency warnings. Python compilation and Ruff
also passed.

The new numerical regression initializes a nonzero BF16 proprioception weight
at `1.0` and applies actual AdamW updates at `1e-4`. At step 1 the FP32 master
changed while the BF16 projection remained exactly at its initial value; the
master continued to drift through step 8, and by step 32 the accumulated value
crossed a BF16 quantization boundary. Every projection exactly matched the
BF16 cast of the persistent master.

The separately authorized fresh-root single-GPU result is appended after
execution.
