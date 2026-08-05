# CACH Stage 2B L2B ledgered-manager implementation plan

Status: `design_closed_for_additive_implementation_and_cpu_synthetic_tests_not_admitted`

Canonical implementation host/worktree:
`H200:/home/zch/workspace/sana-wam`.

This document is the source of truth for the L2B implementation slice. It is
an additive development plan, not an execution receipt, filesystem admission,
training authorization, evaluation result, recovery authorization, or
scientific claim.

## 1. Pinned predecessors and immutable boundary

L2B is built over these caller-pinned predecessors:

- Stage 2B L0/L1 source manifest SHA256:
  `87a5f36d13e06cdd5d951cf1dd96f4e353edc50fc555c96b3269d19859126c0a`;
- Stage 2B L0/L1 deterministic bundle SHA256:
  `0feb797618cb9e02150abd52b357c03d2a4986177fc1e8ebeca626ee808389d2`;
- L2A outer source manifest SHA256:
  `5d27a778bff120ff7a27c18ebaffbfb1761b2ced3d5f2e4c46a4de8c5763b8bf`;
- L2A deterministic bundle SHA256:
  `163737a9b8c9b88bfd8d7e05064052fc2948661f270c4a9a9f4ea55e96494b7d`;
- frozen manager SHA256:
  `8ab968b77660df1092b43a52eecb4dba16316060e7f83c0ad6fd214230a8a5dd`;
- exact production paired stager SHA256:
  `c1455af51df4e0a0eff9971fa98e9272aac580084029a7990161817d246a6ef4`;
- L2A offline ledger implementation SHA256:
  `030a67332310d695395729161de7eb26999e391d469ff24b9c2d532666e77da6`.

The following predecessor files are immutable in this slice and must not be
edited:

```text
src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b.py
src/sana_wam/model/cach_paired_stager.py
src/sana_wam/cach/stage2b_offline_ledger.py
```

The L2A seven-file bundle and its outer manifest are also immutable. L2B uses
new paths and must later be captured as a zero-collision delta over the full
verified predecessor inventory.

## 2. Corrected integration finding

The L2A plan described a thin facade which wraps the stager and replaces the
receipt publisher. Read-only L2B seam audit found one omitted production
constraint: the frozen public `Stage2BHybridCacheManager.commit_paired` accepts
only the exact `CACHTeacherForcingPairedStager` type outside synthetic mode. A
normal facade proxy, wrapper, or subclassed stager is therefore rejected before
numerical staging.

The publisher hook alone also remains insufficient: it receives receipt ID,
receipt bytes, and receipt SHA256, but not the detached staged state or tensor
payloads needed by the L1 snapshot encoder.

The corrected zero-collision seam is:

1. add a private subclass in the new L2B module;
2. mirror the frozen public method's source/request/exact-stager authorization
   checks without weakening them;
3. call the pinned private `_commit_teacher` method with a local capturing
   callable;
4. have that callable invoke the already-authorized exact delegate exactly
   once, then reconstruct and L1-encode the detached staged state before
   returning the original tuple;
5. let a private publisher adapter commit the L2A terminal decision before the
   inherited manager performs its existing in-memory CAS.

This is a deliberately pinned private seam. It is technical debt, but it does
not require monkeypatching, executing the model twice, inspecting Python stack
frames, publishing after CAS, weakening the exact production-stager check, or
copying the full private commit implementation. A change to the frozen
manager/stager hashes invalidates L2B until this seam is re-audited.

## 3. Additive file scope

The initial L2B implementation is restricted to:

```text
docs/cach_sana_wam/stage2b_l2/STAGE2B_L2B_LEDGERED_MANAGER_IMPLEMENTATION_PLAN_20260731.md
src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b_ledgered.py
tests/test_cach_stage2b_ledgered_manager.py
```

Later verifier/report/source-list/manifest files may be added only after code,
focused tests, full CACH CPU regression, and independent review are green.

## 4. Public API and ownership

The public type is `Stage2BLedgeredHybridCacheManager`. Its supported initial
surface is:

```text
initialize_fresh(..., ledger: Stage2BOfflineLedger, synthetic_test_only=False)
for_synthetic_tests(..., ledger: Stage2BOfflineLedger)
state
snapshot_for_denoise(...)
finish_denoise(...)
commit_paired(...)
reset(...)
```

The exact denoise signatures delegate to the frozen manager without a public
raw-manager getter. The constructor, the seamed subclass, and the publisher
adapter remain private, and callers cannot inject an existing manager. Python
reflection can still reach underscored fields; that is outside the caller
contract and is used only by focused tests to prove ordering.

The initial factory is fresh-only. It requires:

- an empty decision chain;
- no PREPARED intent;
- no orphan operation, identity, receipt, snapshot, or tensor object beyond the
  exact genesis snapshot object set;
- manager genesis state digest equal to the ledger's pinned genesis digest;
- exact ledger/manager staging variant, layout, registry, episode, and epoch
  binding.

Any non-genesis root fails with `L2B_RECOVERY_REQUIRED`. L2B does not reopen a
non-empty root, reconstruct `_completed`, install private manager state, or
execute a model during recovery. Those are L3 responsibilities.

The current callable factory is also CPU-synthetic-only because both the pinned
L1 codec and L2A ledger reject non-synthetic layouts and non-CPU tensor
encoding. `synthetic_test_only=False` therefore fails immediately with
`L2B_CPU_SYNTHETIC_ONLY`, before manager construction or ledger mutation. The
safe default remains `False`; this slice must explicitly opt in with
`synthetic_test_only=True`, as the named synthetic-test factory does. The
production exact-stager seam is preserved and reviewed in source, but it is
not a callable production path or an admission claim in this slice.

## 5. Facade state machine

```text
READY
  -> PREPARED
  -> STAGING
  -> STAGED
  -> TERMINAL_PUBLISHING
  -> TERMINAL_EXACT
  -> CAS_CONFIRMING
  -> READY

any ambiguous/durable-boundary violation -> POISONED
```

`POISONED` is absorbing. State access, denoise access, commit, and reset all
fail closed after poison. The facade owns an `RLock`, one active transaction,
and an unforgeable per-transaction token. External commit/reset calls in any
state other than `READY` are rejected before writing another operation object
or intent.

The publisher token is disarmed inside the same facade critical section before
`_active` is cleared and `READY` becomes externally visible. Adapter arm
failure is sticky poison. This prevents a new PREPARED transaction from racing
a stale token left by the preceding transaction.

Publisher re-entry, wrong token, a second publisher call, receipt/snapshot
mismatch, ledger conflict, unreadable terminal, terminal-exact followed by
manager failure, or post-CAS mismatch poisons the facade.

## 6. Paired commit protocol

Before publishing PREPARED, the facade performs all pure request checks which
can be completed without the model:

1. normalize and authorize `CommitSource`;
2. require an exact Stage 2B teacher-forcing request;
3. preserve the production exact-stager type check;
4. validate episode/revision/layout/content-time binding against the READY
   state;
5. call pinned pure `_prepare_teacher_pair` and derive exact pair/proof/tensor
   identities;
6. compute the valid-only committed-action span digest with the frozen
   committed-action-history tensor digest;
7. build canonical `cach.stage2b.paired_commit_operation.v1` bytes;
8. publish the operation identity and the exact PREPARED intent.

The capturing callable then:

1. verifies its token and transitions PREPARED to STAGING;
2. calls the authorized delegate exactly once;
3. requires an eager exact tuple;
4. uses the same frozen `_materialize_layer_states` helper as the manager;
5. reconstructs the exact previous history or exact typed empty history;
6. appends the prepared fixed-slot teacher actions once;
7. constructs a detached `Stage2BTemporalState` with the same frozen fields and
   digests as `_commit_teacher`;
8. encodes it through the pinned L1 codec;
9. stores the snapshot under the active token and transitions to STAGED;
10. returns the original raw payload tuple unchanged.

The inherited `_commit_teacher` independently validates/materializes the tuple,
builds its own staged state and receipt, then calls the private publisher
adapter. The adapter:

1. rejects re-entry and requires the matching STAGED token;
2. validates exact receipt ID, canonical bytes, and expected SHA256;
3. requires
   `snapshot.state_manifest_digest == receipt.staged_state_manifest ==
   receipt.state_manifest_after`;
4. calls `Stage2BOfflineLedger.commit` with the detached snapshot and exact
   success receipt;
5. accepts only a stably read exact COMMITTED record bound to the active intent;
6. returns `ReceiptPublication(durable=True, readback_verified=True)` only after
   that exact terminal record exists.

Only then can the inherited manager execute its existing CAS. On return, the
facade compares the manager state, receipt, active snapshot, and terminal
decision before returning to READY.

### 6.1 Deterministic staging failure

If PREPARED exists, no publisher call began, the inherited manager did not
advance state, and staging/pre-publication validation fails, the facade
publishes one registered state-preserving decision:

```text
result = ABORTED
abort_code = STAGING_FAILED
failure_point = STAGING
snapshot_manifest_sha256 = null
state_manifest_after = state_manifest_before
```

After exact ABORTED read-back the facade returns to READY and re-raises the
original failure. The aborted transaction ID remains permanently consumed. A
new transaction ID may proceed.

### 6.2 Terminal uncertainty

Once publisher entry begins, the facade never guesses ABORTED. It reconciles
only against the exact candidate terminal SHA256:

- exact COMMITTED: continue only if the inherited manager can still complete
  and all post-CAS bindings match;
- stable ABSENT: only the L2A `abort_after_reconciled_absence` path may create a
  terminal-publication ABORTED decision; the facade still becomes POISONED
  because the inherited manager poisons on publisher exception;
- conflict, malformed, changing, or unreadable: POISONED with no ordinary
  abort attempt.

For reset, an inherited failure after PREPARED but before publisher entry is
not labeled as numerical staging. After proving the live state stayed exact,
L2B records `PRECONDITION_FAILED/AFTER_PREPARED`; any uncertainty while writing
that abort poisons the facade.

## 7. Reset protocol

Reset is accepted only in READY. The facade computes the exact empty target
state and L1 snapshot before invoking the inherited reset method, then
publishes a reset operation identity and PREPARED intent. The publisher adapter
commits the reset terminal decision during the inherited publisher callback;
the inherited manager performs its CAS only afterward.

Persistent mode rejects reset while a commit is active before writing any reset
object or intent. L2B never uses the frozen manager's ability to describe a
reset as aborting an in-memory pending commit; durable reset receipts must have
`aborted_pending_commit_id is None`.

## 8. Retry and collision rules

- Same transaction ID plus exact operation/intent/terminal bytes is
  idempotent.
- Exact COMMITTED retry in the same live process freshly reconciles the saved
  terminal SHA256 and receipt reference through L2A before returning the prior
  receipt; it must not call the stager or append ledger records.
- Any field difference for a reused transaction ID is a permanent conflict.
- An ABORTED ID cannot be restaged or committed.
- Commit/reset share the one L2A transaction namespace.
- L2B cannot use an exact terminal from a prior process because state install
  and completed-map reconstruction are L3.

## 9. Required CPU test matrix

The focused synthetic suite must prove at least:

1. PREPARED is durable before the stager is entered.
2. The exact delegate is called once and the committed ledger snapshot decodes
   byte-equivalent to detached live state, including a two-chunk sequence.
3. Terminal COMMITTED exists before the inherited manager CAS.
4. An injected exception after terminal publication but before CAS leaves the
   ledger committed, leaves the old state unadvanced, and sticky-poisons L2B.
5. Staging failure publishes exact STAGING_FAILED/STAGING ABORTED, publishes no
   snapshot, leaves state unchanged, and does not poison a new transaction.
6. An independently L1-valid wrong snapshot or receipt/snapshot mismatch is
   rejected before CAS and poisons the facade.
7. Publisher re-entry poisons even if an injected callback swallows the inner
   exception, with no second operation or intent.
8. Unknown terminal outcome covers exact, stable-absent, and conflict classes;
   no ordinary abort is attempted.
9. Reset produces the exact empty snapshot and terminal-before-CAS ordering.
10. Reset during an active commit is rejected before any reset object/intent.
11. Same-process exact retry does not stage; different reuse conflicts; an
    ABORTED ID cannot be reused.
12. Invalid source/request/layout/revision is rejected before PREPARED and does
    not change the ledger root.
13. Fresh factory rejects committed, pending, or identity-mismatched ledgers.
14. POISONED is absorbing for all exposed state/commit/reset/denoise surfaces.

After the focused suite passes, H200 must run Ruff, `py_compile`, the frozen
manager/codec/ledger regressions, and the complete lightweight CACH CPU suite
with only the already-recorded predecessor deselection.

## 10. Explicit non-authority and remaining blockers

Even a green L2B CPU suite does not authorize or establish:

- an admitted `/DATA` ledger root or real filesystem power-loss semantics;
- multi-process writer fencing, same-UID hostile mutation, or parent/mount
  admission;
- restart recovery, private manager state installation, or model-free replay;
- production dataset/checkpoint/weight access;
- CUDA/GPU, full-model execution, training, evaluation, deployment, capture,
  or a formal/scientific endpoint;
- self-forcing, deploy applied-action commits, AFCC, DAgger, rerank, best-of-N,
  or approximate replay.

The private `_commit_teacher` dependency is a tracked implementation debt, as
are the pinned L2A `_lock`, `_scan_records`, and `_genesis_state` reads and the
frozen pure preparation/materialization helpers. L3 must either pin and
continue to audit them or introduce a separately versioned, reviewed public
two-phase/state-install interface without modifying the frozen predecessor
artifact.
