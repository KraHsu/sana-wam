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

## Fresh-root GPU rerun

The separately authorized rerun passed on physical GPU 0 / UUID
`GPU-1ec28cfb-f501-23f3-f865-275a744ca053`.

- Source commit: `fce4bd90943d935123925a869efde8939cd54d46`
- Config SHA256:
  `80b6b7c7562be47953ed9972ab1c6bd56b3890811d46314375cc8132a60f9a7f`
- Runner SHA256:
  `03b36d83a3b095dd71d154e6df5d3cfb423a3f584ec8fd41adb1e29474c54f0b`
- Root: `/tmp/sana-wam-libero-t1-one-update-fce4bd9-20260806-a2`
- `RESULT.json` SHA256:
  `79e8f1201eef5f550da7287ad614933938877fd4ee54778b7377d027f7ee0086`
- Frozen modes: root `0500`, result `0400`

The full 5.840B-parameter AR architecture constructed from the pinned published
SANA base. Exactly 639,653,063 action/proprioception parameters were trainable;
all video-backbone parameters remained frozen. The fixed Spatial episode-0
sample produced a finite action loss of `13.992515563964844`. Backward produced
finite nonzero gradients for all 560 trainable tensors and all four configured
roots, with no missing gradients. The pre-clip global norm was `1272.0`; the
configured clipping operation then preceded exactly one AdamW step. All
trainable parameters and optimizer state remained finite, frozen parameter
versions were unchanged, and both optimizer groups had an observed numerical
update.

T1 reached the video timestep embedder as FP32 with fractional values, while
action timesteps remained BF16. Peak update memory was approximately 17.14 GB
allocated / 17.97 GB reserved. Model/data construction took 39.95 seconds and
the single loss-forward/backward/update section took 2.03 seconds. GPU memory
returned to zero after process exit.

## Remaining precision and evidence boundaries

Changed-value probes observed `action_backbone`, `proprio_action_embed`, and
`proprio_video_embed`, but did not observe a changed scalar under
`proprio_encoder`, despite that root having finite nonzero gradients. The run
used direct BF16 model parameters with no FP32 optimizer masters
(`optimizer_master_parameter_count=0`). This does not invalidate the wiring
closure or the two optimizer-group update assertion, but it is a precision risk
to close before formal training; the probe result must not be reported as proof
that every allowlisted root made a numerically effective first-step update.

The run used metadata-bootstrap normalization stats, one real sample, one
microbatch, positive base learning rates, and no production warmup or gradient
accumulation equivalence. It did not load or save a SANA-WAM checkpoint, enter a
training loop, run a simulator, or perform benchmark evaluation. The PASS is
non-formal architecture wiring evidence, not training admission or LIBERO
performance evidence.
