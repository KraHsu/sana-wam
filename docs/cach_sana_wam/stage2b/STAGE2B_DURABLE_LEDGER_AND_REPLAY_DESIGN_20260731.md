# CACH Stage 2B durable ledger and restart-replay design

Status: `l0_l1_offline_implemented_l2_l3_blocked_not_admitted`

Canonical implementation host/worktree:
`H200:/home/zch/workspace/sana-wam`.

This document specifies the next additive Stage 2B slice. It does not
authorize training, evaluation, checkpoint/data access, deployment, formal
capture, Stage 3, or a scientific claim. It does not change any Stage 1/2
bundle, manifest, evidence root, pinned Sana tree, or AFCC tree.

Predecessor pins remain:

- main HEAD: `605f1c134b4c983ff80f8489c4bc8847036329e2`;
- Sana HEAD: `16b9cec673e3335724ba2d8db25de7f9ed229292`;
- AFCC handoff HEAD: `9586486f2a9f5172d57b325e32093a3e018d34c0`;
- Stage 2 manifest SHA256:
  `8333c74d9bc8115937f98d9fc12a60da62cab49dd8895cd403047027315052fb`;
- Stage 2 source bundle SHA256:
  `71e73b3f2b1efb125f0b57f8c47913bab47d492920310f0a53848fe7903c229c`.

## 1. Problem statement

The current Stage 2B manager stages a complete outer cache/history state,
durably publishes a COMMITTED receipt, and then swaps its in-memory pointer.
That ordering prevents an undurable state from becoming live, but it has a
crash gap:

```text
staged state ready
  -> COMMITTED receipt durable
  -> process crashes before in-memory pointer swap
```

After restart, `_state`, `_layout`, `_completed`, `_completed_resets`, and the
tensor payloads exist only in memory. Reinitializing an empty manager and
retrying the same receipt cannot recover: a correct exclusive publisher finds
the existing ID, while the manager has no exact state image to install. A
receipt manifest alone is insufficient because it binds tensor digests but
does not contain the tensor bytes.

Therefore a filesystem receipt publisher, by itself, does not close durable
commit or restart replay. The durable decision must bind a complete,
round-trippable Stage 2B state snapshot.

## 2. Non-negotiable invariants

1. One global transaction-ID namespace covers commit, reset, and abort.
2. Every transaction has exactly one terminal result: `COMMITTED` or
   `ABORTED`; the two results cannot occupy separate competing files.
3. The durable terminal decision is the linearization point. The in-memory
   pointer is a recoverable cache of that decision, not a second authority.
4. Same ID plus the exact same intent/decision is idempotent after stable
   read-back. Any differing intent, result, receipt, state, variant, episode,
   or tensor byte is a permanent conflict.
5. An `ABORTED` ID is terminal. A retry must use a new transaction ID.
6. Unknown publication outcome must be reconciled by stable lookup; it must
   never be guessed as aborted or committed.
7. Replay restores exact serialized state; it never reruns the video/action
   model, approximates history, subtracts a loss, or reconstructs from a
   caller-owned request list.
8. `CACH_A` and `REF_GDN_CORRECTED` remain distinct in intent, snapshot,
   decision, receipt, history-summary operator, and replay chain.
9. `monotonic_ns`, filename enumeration order, and filesystem mtime are
   diagnostic only and never establish replay order.
10. Every failure is fail-closed. A corrupt, ambiguous, forked, missing, or
    non-reconstructible chain grants no execution authority.

## 3. Layered implementation

### L0: strict receipt store

The first bounded implementation may provide:

- hashed receipt filenames under one pre-created dedicated directory;
- `O_NOFOLLOW | O_EXCL`, single-link regular-file checks;
- write-all, file `fsync`, mode `0400`, stable same-inode read-back;
- directory `fsync` and read-only lookup;
- strict canonical JSON/schema/embedded-ID/SHA validation.

L0 is useful plumbing, but it explicitly does **not** close ABORTED decisions,
state snapshots, the publish-before-CAS crash gap, or restart replay.

The bounded L0 implementation now exists in
`src/sana_wam/cach/stage2b_receipt_store.py`, with focused tests in
`tests/test_cach_stage2b_receipt_store.py` and a real manager receipt round
trip in `tests/test_cach_stage2b_committed_action_history.py`. On canonical
H200, the L0-inclusive Stage 2B focused suite passed `44` tests and the full
CACH CPU regression passed `286` tests with `10` skipped and the known
Stage-1-only pin test deselected. Ruff passed.

This implementation is intentionally narrower than a ledger. It accepts only
successful Stage 2B commit/reset receipts; it cannot record `PREPARED` or
`ABORTED`, persist state bytes, recover a manager, reconcile an unknown
publisher outcome, or grant execution/scientific authority. It requires one
pre-created empty root owned by the effective user and not group/world
writable, and revalidates that root pathname/inode before returning. Ancestor
directory trust/fencing and `/DATA` crash semantics remain unadmitted.

### L1: canonical Stage 2B state snapshot codec

The codec must serialize and restore all information required to reconstruct:

- exact `ChunkActionLayout` identity and manager-bound registry identity;
- the complete outer `Stage2BTemporalState` manifest and variant;
- every `LayerTemporalState`, `ContentTime`, tensor logical ID, dtype, shape,
  device identity, and exact raw tensor bytes;
- every committed-action span, interval, source proof, commit ID, tensor
  metadata, and exact valid-only action bytes;
- state, history, cache, registry, layout, and content-object digests.

Tensor data is stored as content-addressed immutable blobs. A canonical
snapshot manifest references those blobs. Decode must use an explicit,
pre-admitted device mapping, then recompute all existing tensor/history/state
digests and require exact equality. Python pickle and arbitrary code execution
formats are forbidden.

The first codec admission is CPU/synthetic only. CUDA device/stream semantics,
large-state memory pressure, and device remapping remain separate blockers.

The bounded L1 codec now exists in
`src/sana_wam/cach/stage2b_state_snapshot.py`, with focused tests in
`tests/test_cach_stage2b_state_snapshot.py`. It uses canonical JSON plus a
sorted unique table of content-addressed raw tensor blobs; accepts only exact
CPU synthetic layouts and an explicit `{"cpu": "cpu"}` device map; and
whitelists FP32, BF16, and bool, and explicitly binds little-endian raw-byte
order while rejecting other hosts/descriptors. It reconstructs the complete
outer state, inner cache/layers/`ContentTime`, and valid-only committed history, then
recomputes all existing tensor/cache/history/outer digests and revalidates the
caller-bound full layout payload, registry manifest, variant, summary
operator, and logical tensor IDs.

The L1 tests cover empty, one-chunk, continuation/partial-tail, and both
variant round trips, as well as missing, substituted, extra, duplicate, or
trailing blobs and dtype/shape/value/digest/logical-ID/layout/registry/history
corruption, including noncanonical bool storage bytes and BF16 edge bit
patterns. On canonical H200 the L1-inclusive Stage 2B suite passed `63` tests;
the full CACH CPU regression passed `305` tests with `10` skipped and
the known Stage-1-only pin test deselected; Ruff passed.

L1 remains an in-memory codec, not a filesystem object store, ledger, restart
replayer, or manager installer. It does not admit CUDA tensors, a production
20-layer snapshot, device remapping, concurrent mutation, crash recovery,
training, evaluation, deployment, or scientific use.

### L2: append-only terminal decision ledger

Before numerical staging, the ledger reserves one durable canonical intent.
The intent binds the global transaction ID, nonce, operation, variant,
episode/revision, complete source/input identity, and state-before digest.
It is `PREPARED`, not authority to advance state. A recovered intent without a
terminal decision is closed as a stable `ABORTED_PROCESS_EXIT` decision.

One canonical decision schema binds at least:

```text
schema
ledger_id
ledger_sequence
transaction_id
transaction_kind = paired_commit | reset
intent_sha256
result = COMMITTED | ABORTED
staging_variant
episode_id / episode_epoch
state_manifest_before
state_manifest_after
snapshot_manifest_sha256 | null
success_or_abort_receipt_sha256
previous_decision_sha256 | null
decision_sha256
```

`COMMITTED` requires a complete state snapshot and an advancing or resetting
state transition. `ABORTED` requires no live-state advancement and no staged
state snapshot. Abort receipts bind a stable failure code and injection/crash
point; arbitrary exception strings are diagnostic only.

The exact publication protocol is:

1. validate the live predecessor, typed variant, and global ID;
2. exclusive-publish and stable-read the complete `PREPARED` intent;
3. stage into detached transaction-local state;
4. encode content-addressed blobs and snapshot manifest;
5. stable-read and digest-verify every referenced object;
6. construct one terminal decision linked to the previous decision;
7. publish that decision with exclusive no-follow semantics on the same
   filesystem and `fsync` the decision directory;
8. stable-read the terminal decision and all referenced objects;
9. only then install the decoded/verified state as the in-memory live pointer.

The final terminal filename must never expose partially written bytes. The
implementation may prepare immutable objects under private temporary names and
then use an admitted same-filesystem no-replace publication primitive. Plain
overwrite/rename that can replace an existing decision is forbidden.

### L3: deterministic recovery

`recover_from_ledger` must start from a frozen genesis record and build one
strict chain using `ledger_sequence`, `previous_decision_sha256`, and exact
state-before/state-after equality. It must reject:

- duplicate sequence numbers or transaction IDs;
- a fork, gap, stale predecessor, or mixed ledger ID;
- a decision whose variant differs from the chain;
- an ABORTED decision that changes state;
- a COMMITTED decision without a complete valid snapshot;
- orphan/missing/mutated blobs or an unexpected filesystem entry;
- any receipt/snapshot/state digest mismatch.

Recovery loads the last committed/reset state image without executing a model.
The recovered completed-ID tombstones remain global across resets. Retrying a
known committed transaction returns its exact durable receipt; retrying an
aborted or conflicting transaction fails closed.

## 4. Crash matrix

| Crash point | Durable interpretation on restart |
|---|---|
| before PREPARED intent is durable | no transaction; predecessor state |
| after PREPARED intent, before terminal decision | publish/recover terminal `ABORTED_PROCESS_EXIT`; predecessor state |
| while writing an unreferenced content object | predecessor; orphan is diagnostic and never authoritative |
| after objects, before terminal decision | predecessor |
| during terminal decision publication | stable lookup determines absent or exact terminal record; ambiguity poisons recovery |
| after terminal decision fsync, before memory swap | recover and install the decision's exact snapshot |
| after memory swap | recover the same exact snapshot |
| after ABORTED decision | predecessor state; transaction ID remains terminally aborted |

No path may infer success merely from the presence of some blobs, or infer
abort merely because the process returned an exception.

A publisher exception with an unknown publication outcome is reconciled by
stable terminal lookup. Exact COMMITTED is recovered, exact ABORTED is
respected, absence permits the registered failure path, and an ambiguous or
unreadable result poisons the ledger. It is forbidden to append ABORTED merely
because the publishing call raised.

## 5. Concurrency and reset

The first persistent implementation is single-writer. It requires a reviewed
filesystem fencing/lock mechanism and records the writer fence identity in the
ledger. Until that mechanism is admitted, multiple manager processes sharing
one ledger root are prohibited.

The current in-memory synthetic manager permits reset during staging to test
CAS invalidation. Persistent mode initially hard-rejects reset while a paired
transaction is pending. A later design may introduce one atomic multi-effect
decision that both commits reset and terminally aborts the pending transaction;
two independent terminal publications are forbidden.

Publisher callbacks may not re-enter the manager. A durable publisher return
is followed by a second local CAS check; any impossible discrepancy poisons the
process rather than overwriting a newer state.

## 6. Required lightweight tests

L0 tests:

- exact bytes/hash/mode/inode/fsync/read-back success;
- same-ID same/different payload exclusive collision;
- malformed/noncanonical JSON, wrong schema/embedded ID/hash rejection;
- symlink/root/path traversal/hardlink/non-regular-file rejection.

L1 tests:

- exact empty, one-chunk, continuation, partial-tail, both-variant round trip;
- dtype/shape/value/layout/registry/history mutation rejection;
- content-addressed blob substitution, omission, duplication, and trailing-byte
  rejection;
- decoded state manifest equals the original exact digest.

L2/L3 tests:

- COMMITTED and ABORTED ID mutual exclusion;
- crash injection at every row in the crash matrix;
- decision-after-CAS-gap recovery without model execution;
- deterministic idempotent restart and global ID tombstones across reset;
- chain gap/fork/reorder/variant-mix/state-before mismatch rejection;
- unknown publisher outcome reconciliation;
- reset-while-pending hard rejection in persistent mode.

Tests use only temporary CPU tensors and synthetic layouts. They create no
formal root and grant no scientific or execution admission.

## 7. Production blockers after lightweight implementation

Even if all synthetic L0--L3 tests pass, the following remain blocked:

- H200 `/DATA` filesystem admission for `fsync`, no-replace publication,
  locking/fencing, hard-link/rename behavior, permissions, and crash recovery;
- canonical realpath plus parent/ancestor ownership, sticky-bit, symlink, and
  rename-fencing policy for every durable root;
- CUDA tensor synchronization, device mapping, and stream ownership policy;
- 20-layer snapshot size, latency, memory, and I/O bounds;
- complete Sana/runtime/source closure and named-init parity;
- real production layout/timebase verifier, full architecture execution, and
  complete C0--C8 evidence;
- deploy ACK transport, training, evaluation, checkpoint/data admission,
  Stage 3, or any formal/scientific claim.

No source bundle or Stage 2B immutable evidence artifact may be finalized
until the implemented scope and unresolved blockers are recorded exactly.
