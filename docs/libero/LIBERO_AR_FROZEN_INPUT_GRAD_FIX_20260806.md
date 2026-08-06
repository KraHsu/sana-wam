# LIBERO AR frozen-input-gradient root fix — 2026-08-06

## Failed first update smoke

The first authorized T1 real-sample update smoke stopped after backward and
before gradient clipping or `optimizer.step()`. Three trainable roots had
finite nonzero gradients, but `proprio_video_embed.grad` was absent.

- Root: `/tmp/sana-wam-libero-t1-one-update-009ef7cb-20260806-a1`
- `FAILED.json` SHA256:
  `731cc18781de919af27a3159108e0e43136c0c27b685a9829e7d216a7d662e1f`
- Frozen modes: root `0500`, file `0400`
- Optimizer steps: `0`
- Checkpoint load/save: no SANA-WAM checkpoint
- Training loop or benchmark evaluation: not executed

The failure root remains immutable and is not reused.

## Root cause

The exact trainable allowlist excludes the 2B `video_backbone`, so Trainer
freezes that subtree. The default freeze contract disables its parameters,
puts it in eval mode, and recursively wraps every descendant `forward` in
`torch.no_grad()`.

`proprio_video_embed` is a separate trainable root. Its output is added to the
video per-frame time embedding and then immediately enters the frozen
backbone's `t_block`. The recursive `no_grad` wrapper therefore cut the graph
at that boundary. This was a freeze/allowlist contradiction, not an AR causal
mask disconnection. The action stream reads causal video features, so the
action loss can train this seam when input autograd is preserved.

## Corrected contract

`BaseWAMArchitecture.freeze_modules()` retains its old recursive-`no_grad`
default. A new explicit `preserve_input_grad=true` mode instead:

1. keeps every selected module parameter at `requires_grad=false`;
2. keeps the complete subtree in eval mode;
3. skips recursive `no_grad`, allowing gradients only with respect to inputs;
4. rejects attempts to claim preservation after a subtree was already wrapped.

Trainer exposes this only with the exact module allowlist through:

```yaml
training:
  preserve_frozen_input_grad_modules:
    - video_backbone
```

The field rejects unknown, duplicate, dotted, or trainable roots. LIBERO pins
the list exactly to `video_backbone`; the optimizer allowlist remains the same
four action/proprioception roots. The 2B video parameters do not enter the
optimizer and `lambda_video` remains zero.

## CPU verification

On H200 with CUDA hidden, the focused freeze/Trainer/mini-AR/LIBERO suite
passed `58` tests. The broader related regression suite passed `176` tests
with 14 pre-existing dependency warnings, and Ruff passed.

The mini-AR regression covers the targeted production semantics: zero-initialized
per-chunk proprio seams, `lambda_video=0`, `lambda_action=1`, frozen/eval video
parameters, and gradient checkpointing enabled. The action loss produced a
finite nonzero `proprio_video_embed` gradient while every video parameter
remained frozen with no gradient.

This is source and CPU closure only. The separately authorized fresh-root GPU
rerun is the next step and is still non-formal architecture validation, not
training or benchmark evidence.
