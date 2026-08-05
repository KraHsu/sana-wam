# Stage 2 offline teacher-forcing scoped-subset report

Status: **offline/test-owned teacher-forcing subset passed; full Gate-S2
blocked and not claimed; non-scientific; training blocked**

Canonical execution location: `H200:/home/zch/workspace/sana-wam`

Final evidence time: `2026-07-31T10:55:01.375271247Z`

Main HEAD: `605f1c134b4c983ff80f8489c4bc8847036329e2`

Pinned Sana HEAD: `16b9cec673e3335724ba2d8db25de7f9ed229292`

This report covers random depth-1 mini-GDN and mini-ActionDiT models with
synthetic tensors under a test-owned teacher-forcing harness. It loaded no
checkpoint or dataset, created no optimizer, ran no training, evaluation, or
model/policy capture, and created no formal experiment root. It is not a
capability or scientific result, and it does not authorize Stage 3.
The report field `capture_started=false` means no model/data/formal capture;
the evidence-retention attempts described below are audit artifact handling.

## Verified offline boundary

- Fixed external `K` uses a batch-uniform contiguous valid prefix. Exact-zero
  video, action-condition, action-token, and bridge padding is compacted before
  numerical execution and restored as exact zero afterward.
- Pinned cached/Triton GDN sees valid frames only; pinned Sana is unchanged.
- A strict bidirectional codec isolates vendor `list[10]` state from typed
  `CacheScratch` / `StagedLayerPayload` state. Executable slots are 0/1/4/9,
  slot 6 is a Python-float type flag, and unsupported/reserved layouts fail
  closed.
- The synthetic episode uses `L=8`, `K=3`, and `r=8`. Its three chunks have
  video valid lengths `[3, 3, 2]`, action capacities `[16, 24, 24]`, and valid
  action counts `[16, 24, 16]`. Action RoPE intervals are `[0,16)`, `[16,40)`,
  and `[40,56)`.
- The mini-GDN video timestep is FP32 zero. The mini-ActionDiT action timestep
  is BF16 zero. This split is deliberate and part of the tested dtype contract.
- Paired staging runs video and action paths from the same digest-bound
  synthetic request. A test-local publisher emits one receipt and one typed
  state pointer per successful chunk; this is not a production publisher.
- Synthetic failure injection checks unchanged state. Its exclusive-create,
  fsync, read-back, and mode checks produced a separately retained read-only
  scoped nonformal failure artifact; this remains test-owned evidence, not a
  production `ABORTED` ledger or formal root.
- Future-input perturbation is checked against direct cache tensors and earlier
  outputs, not only a manifest digest. The covered continuation equality is a
  codec/partial-tail test; it is not a claim of full production
  causal-versus-deploy equivalence.
- Future-action perturbation preserves prior committed outputs/state and proves
  a finite, nonzero parameter gradient through the zero-initialized adapter.
  This closes the mini zero-init wiring check, not full-model gradient/JVP.
- A standalone `cach.applied_action_ack.v1` verifier validates canonical inline
  bytes and registered immutable references through stable exact-range reads.
  It returns canonical applied values when commanded and applied differ. It has
  no HybridCacheManager/server transport or deploy commit edge, and deploy
  commit remains hard-disabled.

## Final scoped numerical results

The final combined Stage 2 command used GPU0 and
`GDN_DISABLE_COMPILE=1`:

```text
75 passed, 13 warnings in 5.63s
```

The exact seven-file selection was:

```text
tests/test_cach_stage2_prefix_compaction.py
tests/test_cach_stage2_hybrid_cache_codec.py
tests/test_cach_stage2_failure_evidence.py
tests/test_cach_stage2_mini_gdn.py
tests/test_cach_stage2_paired_mini_gdn.py
tests/test_cach_action_to_video_conditioning.py
tests/test_cach_stage2_applied_action_ack.py
```

The affected CPU regression result was:

```text
264 passed, 10 skipped, 1 deselected in 0.90s
```

The skips are CUDA-only mini tests under an empty `CUDA_VISIBLE_DEVICES`. The
sole deselection is the historical Stage-1 worktree-hash verifier because the
working bytes have intentionally advanced into Stage 2; no Stage-1 authority
or immutable bundle was rewritten.

Static verification reported:

```text
AST_COMPILE_WHITESPACE_OK 13
```

The scoped tests cover prefix compaction, exact-zero restoration, zero-init
adapter bypass and parameter gradient, strict cache-codec failures, real vendor
cache round trips, typed continuation, state immutability, reset/retry behavior,
the exact L8 three-chunk geometry, direct future-video/action separation,
finite/non-vacuous mini outputs, Action RoPE ownership, test-local failure
handling, and 35 standalone applied-action ACK verifier cases. The ACK cases
cover inline/reference stable reads, strict identity/digest/layout checks,
idempotent replay-registry behavior, and commanded/applied mismatch returning
the applied tensor. They also retain the negative rule that command evidence
cannot substitute for applied bytes.

An earlier expanded action forward failed closed because an FP32 action
timestep was supplied to a BF16 mini-ActionDiT. No receipt or state update
occurred. The input was corrected to construct action `t=0` in BF16; no
tolerance or numerical threshold was relaxed. Video `t=0` remains FP32.

## Retained scoped evidence

The final evidence root is:

```text
/DATA/share/sana_cach_stage2_evidence/stage2_offline_subset_20260731_JKYEkuCw
```

It is mode `0500`; its evidence files are mode `0400`. It is a
`retained_read_only_scoped_nonformal_artifact`, not a formal experiment or
scientific root. Its primary execution records are:

- `stage2_pytest.log` SHA256
  `51ece666183330987eede54bcc74cdc97e26262c424ff026c40d0094de53ce33`;
- `stage2_pytest.xml` SHA256
  `8d278479c275804e3f4406e8ae47a7b96993dca678f1f75cc0938cbb3fe6de27`;
- `cpu_regression.log` SHA256
  `d03ede51b69f377d987d8660fa686e44e3b32fc193a0a22b5d334c269d1f8b6d`;
- `cpu_regression.xml` SHA256
  `fea7a6db15444b2c39e39adf531a800f001664661f7727295a372ad517ee3b91`.

The root-local `SHA256SUMS` is mode `0400`, size 1190, and has SHA256
`98ae1c1d8e9549d6f0437d6896595ed8f3133b4330c447960f3a54e3d68b6904`.
It pins every other retained evidence file, not itself. Exact flat inventory
and the `SHA256SUMS` file's own digest are separate artifact-verifier
obligations.

The retained synthetic failure root is:

```text
/DATA/share/sana_cach_stage2_failure_evidence/stage2_failure_20260731_20260731T105154989919953Z
```

It is also mode `0500`. Its `FAILURE.json` is mode `0400`, with SHA256
`23c08337dd99c45ae3aaf2a314ca23afc6646b7241707c0819ff0d71758be9ed`.
This proves only the scoped failure-evidence helper and does not implement a
production ledger.

The first evidence-retention attempt root,
`/DATA/share/sana_cach_stage2_evidence/stage2_offline_subset_20260731_pPSovwEo`,
encountered a post-test metadata quoting failure. It is explicitly marked
`CAPTURE_ABORTED`, frozen mode `0500`, and retained read-only rather than reused
or represented as successful evidence.

The second evidence-retention attempt root,
`/DATA/share/sana_cach_stage2_evidence/stage2_offline_subset_20260731_jRCAj1tM`,
completed its tests but its verifier/source pin changed afterward. It is frozen
mode `0500` and explicitly retained as superseded, not final evidence.

## Why full Gate-S2 remains blocked

This scoped subset does not implement or prove:

- production architecture integration and the production paired stager;
- model-owned `NO_ACTION` in the production stager, including ownership across
  reset/tail paths;
- committed-action identity and the required history summary;
- full-model gradient/JVP Gate-S2 evidence beyond the passed mini parameter
  gradient check;
- manager/server transport and deploy commit integration for the standalone
  applied-action ACK verifier, including production authority receipts,
  observation binding, and durable replay; deploy commit remains hard-disabled;
- a production paired-transaction `ABORTED` ledger, deterministic restart after
  a durable decision, or a production receipt publisher;
- a source bundle with transitive runtime closure; the current delta source
  inventory explicitly does not establish such closure;
- full-model/random-shape admission, checkpoint/data loading, optimizer steps,
  training, evaluation, deployment, or a formal immutable run root.

Accordingly, full Gate-S2 is `blocked / not_claimed`. The standalone ACK v1
verifier, including its positive and commanded/applied mismatch cases, cannot
substitute for production deploy integration. No Stage 3, Stage 4, training,
evaluation, formal run, 504-step result, or scientific conclusion follows from
this report.
