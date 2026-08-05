# CACH Stage 2B L2 offline ledger implementation plan

Status: `l2a_implementation_verified_lightweight_green_ready_for_additive_source_checkpoint_not_admitted`

Canonical implementation host/worktree:
`H200:/home/zch/workspace/sana-wam`.

This records a candidate additive delta over the frozen Stage 2B L0/L1
artifact; it is not itself an immutable checkpoint or execution receipt.
It must not edit, replace, or reuse bytes in that artifact. Its caller-bound
predecessor pins are:

- Stage 2B L0/L1 source manifest SHA256:
  `87a5f36d13e06cdd5d951cf1dd96f4e353edc50fc555c96b3269d19859126c0a`;
- Stage 2B L0/L1 deterministic source bundle SHA256:
  `0feb797618cb9e02150abd52b357c03d2a4986177fc1e8ebeca626ee808389d2`;
- Stage 2 predecessor manifest SHA256:
  `8333c74d9bc8115937f98d9fc12a60da62cab49dd8895cd403047027315052fb`.

The predecessor verifier establishes only a byte-scoped additive source
artifact. It explicitly leaves runtime closure, execution replay, immutable
execution evidence, training, evaluation, deployment, capture, science, and
Stage 3 false.

Current additive implementation state on 2026-07-31:

- `stage2b_offline_ledger.py` and its dedicated test module exist only as new
  L2A files; none of the 17 pinned L0/L1 members was edited;
- the tested implementation SHA256 is
  `030a67332310d695395729161de7eb26999e391d469ff24b9c2d532666e77da6`,
  and the tested dedicated-test SHA256 is
  `0c342390bef912c2f00e9840aa9dbb79393e7ca12ae9137df6a6939746483407`;
- H200 Ruff and `py_compile` pass for both new files;
- 38 dedicated test functions expand to 82 collected cases, and the H200 L2A
  selection passes `82 passed`;
- the full H200 CACH CPU regression passes `397 passed`, with one known
  predecessor-only node deliberately deselected and 13 pre-existing dependency
  warnings;
- three concurrent read-only audits over the implementation and expanded
  fault matrix report no remaining P0 or P1 issue in the declared L2A scope;
- these are mutable development-tree results, not an immutable source
  manifest, execution receipt, formal root, admission, or scientific result.

## 1. Fixed boundary

The implementation scope recorded here is limited to new offline code,
documentation, and lightweight tests using temporary owner-only directories
and CPU synthetic tensors. This document itself grants no additional
authority, including for:

- `/DATA` ledger admission or a formal/frozen ledger root;
- dataset, checkpoint, weight, VAE, tokenizer, or HDF5 access;
- CUDA/GPU execution or full 20-layer model construction;
- optimizer construction, training, evaluation, deployment, capture, or
  Stage 3;
- mutation of the pinned Sana tree, AFCC tree, old bundles, manifests,
  receipts, or evidence roots;
- multi-process writer authority, restart authority, or a scientific claim.

The active code slice is L2A. Its record/state-machine and declared
single-process filesystem fault matrix are green, and independent final audit
is complete. A zero-collision additive source artifact remains before the L2A
checkpoint is immutable. L2B does not start merely because the live suite is
green.

## 2. Why the current publisher hook is insufficient

`Stage2BHybridCacheManager` calls its `DurableReceiptPublisher` only after
numerical staging and after constructing the staged state. The hook receives
only receipt ID, receipt bytes, and receipt SHA256. It cannot:

- publish `PREPARED` before numerical staging;
- observe or encode the detached staged state;
- publish content-addressed snapshot objects before the terminal decision;
- distinguish a known terminal decision from an unknown publication outcome;
- install a recovered state after restart.

Therefore replacing the current receipt publisher alone cannot implement L2.
The first slice adds an independent ledger. A later additive facade coordinates
that ledger with the unchanged manager. Direct support inside the original
manager would require an explicit later overlay and is not part of L2A.

## 3. L2A fixed file scope

The initial implementation is limited to new files:

```text
docs/cach_sana_wam/stage2b_l2/STAGE2B_L2_OFFLINE_LEDGER_IMPLEMENTATION_PLAN_20260731.md
src/sana_wam/cach/stage2b_offline_ledger.py
tests/test_cach_stage2b_offline_ledger.py
```

It must not edit any of the 17 Stage 2B L0/L1 bundle members. A later source
manifest must describe L2A as a zero-collision additive delta over the pinned
Stage 2B bundle.

## 4. L2A durable record model

One ledger ID, staging variant, genesis snapshot/state, layer registry, layout
spec, and sorted layout-instance inventory are immutable for a ledger root. One
global transaction-ID namespace covers `paired_commit` and `reset`. Each
transaction has one canonical `PREPARED` intent and at most one terminal
decision record. `COMMITTED` and `ABORTED` are values in the same terminal
record, never competing filenames.

`LEDGER.json` uses `cach.stage2b.offline_ledger_root.v1` and binds:

```text
admission_scope = cpu_synthetic_single_process_only
ledger_id
writer_fence_id
staging_variant
layout_spec_sha256
layout_instance_digest              # genesis layout
layout_inventory_digest             # sorted caller-pinned layout identities
layer_registry_digest
genesis_state_manifest
genesis_snapshot_manifest_sha256
```

The genesis must be a fully L1-decodable empty state. Sequence zero is valid
only when its episode ID, epoch, revision, and state digest equal that frozen
genesis. A reset may switch to a different layout instance only when that
instance was included in the immutable caller-pinned inventory at initialize
and reopen; all inventory entries must share the root layout spec.

### 4.1 PREPARED intent body

The canonical intent binds exactly:

```text
schema = cach.stage2b.ledger_intent.v1
status = PREPARED
ledger_id
ledger_sequence
transaction_id
transaction_kind = paired_commit | reset
transaction_nonce
writer_fence_id
staging_variant = cach_a | ref_gdn_corrected
episode_id_before
episode_epoch_before
revision_before
state_manifest_before
input_identity_sha256
previous_decision_sha256 | null
```

`intent_sha256` is the SHA256 of the complete canonical intent bytes and is an
external property; it is not embedded in its own preimage.

`input_identity_sha256` points to a canonical
`cach.stage2b.ledger_operation_identity.v1` object, not directly to arbitrary
JSON. That envelope binds the transaction ID/kind/nonce, variant, exact
before-state tuple, operation layout, and the SHA256 of one kind-specific
canonical operation payload. The paired payload binds content-time, source,
dataset, tensor, teacher-pair, conditioning, and committed-action-span
digests. The reset payload binds the old state details and target
episode/layout. Both the envelope and payload are content-addressed and are
cross-checked before `PREPARED` publication.

### 4.2 Terminal decision hash clarification

The older design sketch listed `decision_sha256` inside the decision being
hashed, which is self-referential. L2 v1 replaces that ambiguous shape with an
envelope:

```text
schema = cach.stage2b.ledger_decision_record.v1
decision_body = { ... }
decision_body_sha256 = SHA256(canonical(decision_body))
```

The canonical decision body contains:

```text
schema = cach.stage2b.ledger_decision_body.v1
ledger_id
ledger_sequence
transaction_id
transaction_kind
intent_sha256
result = COMMITTED | ABORTED
writer_fence_id
staging_variant
episode_id_before
episode_epoch_before
revision_before
episode_id_after
episode_epoch_after
revision_after
state_manifest_before
state_manifest_after
snapshot_manifest_sha256 | null
success_or_abort_receipt_sha256
previous_decision_sha256 | null
abort_code | null
```

The full decision-record SHA256 is an external property and is not serialized
inside that record. `previous_decision_sha256` binds the SHA256 of the complete
previous decision-record bytes, not the body hash. This defines an acyclic,
exact chain.

For `COMMITTED`, `abort_code` is null, a complete snapshot manifest is
required, and that snapshot's exact state digest equals
`state_manifest_after`. For `ABORTED`, `abort_code` is a registered stable code,
the snapshot field is null, and state/episode before and after are identical.
Arbitrary exception text is never authoritative.

### 4.3 Immutable object namespace

The L2A root uses one flat, append-only namespace with hashed logical names:

```text
LEDGER.json
intent-<sha256(transaction_id)>.json
decision-<sha256(transaction_id)>.json
object-<content_sha256>.bin
```

`object-*` stores exact L1 tensor blobs, snapshot manifest bytes, and
success/abort receipt bytes. An existing content-addressed object is
idempotent only when stable read-back proves exact bytes. Any differing bytes,
non-regular entry, hard link, wrong mode, symlink, malformed record, path-hash
mismatch, or unexpected root entry poisons the ledger.

The root is pre-created, absolute, owner-controlled, non-symlink, and mode
`0700`. Publication writes an owner-only same-directory temporary, performs
write-all/file-fsync, changes it to `0400`, fsyncs again, and atomically installs
it with Linux `renameat2(RENAME_NOREPLACE)`. It then directory-fsyncs and
stable-reads the single-link final inode. Existing exact bytes are idempotent
only after root rebind.

On single-writer constructor entry, safe unlinked `.tmp-*` remnants are
identified with `O_PATH|O_NOFOLLOW`, restricted to owner-only permission bits,
and removed before a directory fsync. Exact partial initialization containing
only the matching root identity and a subset of the expected genesis objects
may resume. These recovery rules do not provide a multi-process fence; a
second concurrent ledger instance remains prohibited. Parent/ancestor fencing
and real-filesystem power-loss admission remain explicit blockers.

## 5. L2A API and protocol

The new module exposes typed immutable intent, abort receipt, decision body,
decision record, stored-record, and publication result values plus one
`Stage2BOfflineLedger`.

The minimum store operations are:

```text
initialize(root, ledger_id, staging_variant, writer_fence_id,
           expected_layout, expected_registry, genesis_snapshot,
           additional_expected_layouts=())
reopen(root, expected ledger identity, expected_layout, expected_registry,
       expected_genesis_snapshot, additional_expected_layouts=())
publish_operation_identity(identity, operation_payload)
prepare(intent) -> exact idempotent stored intent
publish_object(payload, expected_sha256) -> exact idempotent object
publish_snapshot(encoded_snapshot) -> verified manifest SHA256
publish_abort_receipt(...stable fields...) -> receipt SHA256
commit(intent, snapshot, success_receipt_bytes, decision fields)
abort(intent, registered abort code, registered failure point)
abort_after_reconciled_absence(intent, expected terminal-record SHA256)
read_intent(transaction_id)
read_decision(transaction_id)
scan_valid_chain()
reconcile_terminal(transaction_id, expected_record_sha256)
```

Publication order is fixed:

1. validate the live predecessor and exact typed intent;
2. exclusive-publish and stable-read `PREPARED`;
3. numerical staging occurs outside L2A;
4. fully decode the candidate through L1 with an exact caller-pinned layout
   and registry, then publish and stable-read every content object and the
   snapshot manifest;
5. publish and stable-read the success or abort receipt object;
6. validate the next sequence and previous full-record digest;
7. preflight any existing terminal before publishing referenced objects, then
   exclusive-publish the single terminal decision record;
8. fsync and stable-read the record and all referenced objects;
9. only an L2B facade may allow the unchanged manager to perform its in-memory
   CAS after exact terminal `COMMITTED` read-back.

Directory enumeration never establishes ordering. `scan_valid_chain` sorts by
embedded `ledger_sequence`, then rejects gaps, duplicate sequences or IDs,
forks, stale predecessor hashes, mixed ledger IDs/variants/fences, mismatched
intent hashes, invalid state transitions, missing objects, and unexpected
entries.

Same transaction ID plus exact intent/decision bytes is idempotent. Reuse with
different kind, nonce, input, state, result, snapshot, receipt, or chain
position is a permanent conflict. An aborted ID is terminal and cannot be
retried.

Unknown terminal-publication outcome is reconciled only through stable lookup.
Exact expected terminal bytes may be returned; a different/unreadable record
poisons the ledger. `ABSENT` requires two whole-chain scans, one exact matching
pending intent, two absent target reads, and a directory fsync. A
terminal-publication abort is rejected by the ordinary abort API and may enter
only through `abort_after_reconciled_absence`. The implementation never guesses
commit or abort.

The paired transition validator requires the decoded committed-action history
to be the exact previous prefix plus one span. The new span must bind the
transaction ID, source proof, action digest, cursor interval and chunk; the
cache must bind the paired-payload digest and source proof. Reset requires
epoch `+1`, revision/cursor zero, empty history/cache coverage, no pending
transaction aborted by the legacy manager, and the caller-pinned target layout.

## 6. L2B additive facade

L2B may add, without modifying L0/L1 bundle bytes:

```text
src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b_ledgered.py
tests/test_cach_stage2b_ledgered_manager.py
```

The facade must publish `PREPARED` before calling the old manager. It wraps the
stager to reconstruct and L1-encode the exact detached staged state, and uses a
private publisher adapter to publish the terminal decision before the old
manager's CAS. The publisher must require the staged snapshot state digest to
equal the old manager receipt's `state_manifest_after`.

Staging failure publishes a registered `ABORTED` decision without a snapshot.
Persistent mode hard-rejects reset while another transaction is pending.
Publisher re-entry, an impossible second local CAS, an ambiguous terminal
outcome, or a receipt/snapshot mismatch poisons the facade.

L2B does not implement `recover_from_ledger` or install recovered private
manager state. Those are L3 and require a separately reviewed constructor or
state-install interface. L2B also must not execute the model during recovery.

## 7. Lightweight test status and remaining matrix

L2A tests use only `tmp_path`, canonical JSON, byte strings, and CPU synthetic
L1 snapshots. The current 38 test functions / 82 collected cases cover:

- exact initialize/reopen, frozen genesis, caller-pinned layout inventory,
  operation identity, PREPARED idempotence/conflict, and reset layout changes;
- real paired-manager and reset receipts/snapshots, full L1 decode, exact
  history-prefix/cache lineage, eleven operation-digest mutations, eight
  receipt-binding mutations, and three independently L1-valid lineage
  mutations;
- one terminal namespace, COMMITTED/ABORTED exclusion, direct-publication
  reconciliation-bypass rejection, terminal exact retry through fsync plus
  double-read reconciliation, and state-preserving aborts;
- contiguous sequence/predecessor validation, enumeration-independent order,
  two independently valid genesis forks, duplicate sequence, path-hash,
  fence/stale-state/body-hash, and canonical state-changing-abort rejection;
- missing/substituted referenced objects, before/after durable snapshot and
  receipt group faults, exact restart-style retry, destination EEXIST
  exact/conflict races, fragmented I/O, short reads, file/directory fsync,
  fchmod, rename, read-back, and post-rename directory-close failures;
- root replacement, symlink, hard-link, FIFO, wrong mode, unexpected entry,
  abandoned-safe temp cleanup, temp ownership collision, exact partial
  initialization, same-process lock serialization, and reconciliation across
  exact/absent/conflict, same-byte inode replacement, absent-to-present,
  malformed-target, and directory-fsync faults.

No live lightweight test creates a formal root or grants execution authority.
The remaining filesystem work is admission work, not an unexecuted claim in
this suite: multi-process fencing and temp ownership, same-UID hostile mutation,
parent/ancestor and mount checks, real `/DATA` filesystem semantics, power-loss
behavior, and resource-size bounds remain blocked.

Coverage is equivalence-class based, not a Cartesian mutation of every field
and call site. Canonical JSON and exact-key/type checks converge through the
shared parser; immutable intent, decision, and object publication converges
through `_publish_immutable_locked`; logical IDs converge through deterministic
hashed filenames; and whole-root validation converges through `_scan_records`.
The focused suite therefore tests representative paths through those shared
funnels. It does not individually parameterize every PREPARED conflict field,
mixed identity field, reset before-field, JSON whitespace/extra-member case,
or unsafe root mode. Those mechanical coverage extensions are P2; the shared
fail-closed paths were inspected and no bypass was found.

L2B tests later add exact manager ordering, detached-state equivalence,
commit/reset terminal decisions, staging aborts, re-entry, pending-reset
rejection, post-decision/pre-CAS failure, and idempotent terminal lookup.

Two explicit L2A trust/dependency notes remain:

- the ledger verifies declared paired digests against receipt, snapshot,
  history, and cache lineage, but does not reconstruct the upstream teacher
  proof or fixed-slot tensor preimages. L2B must construct the operation
  payload from the already validated manager request; this is not dataset or
  scientific admission;
- the new module deliberately calls the frozen L0 private pure parser
  `_validate_payload` because editing that pinned L0 member is prohibited.
  Its exact source remains an inherited artifact pin until a later additive
  public adapter is reviewed.

## 8. Admission blockers and stop conditions

Even if L2A/L2B CPU tests pass, all of these remain blocked:

- `/DATA` fsync/no-replace/lock/power-loss and multi-process fencing admission;
- complete parent/ancestor realpath, ownership, sticky-bit, mount, and rename
  fencing;
- L3 chain recovery plus an explicit manager state-install interface;
- CUDA synchronization/device mapping and 20-layer snapshot scale bounds;
- complete Sana/runtime/dynamic-import/native-library/Git recovery closure;
- real layout/timebase, full architecture, named-init parity, and C0--C8;
- checkpoint/data/deploy ACK admission, training, evaluation, capture, Stage 3,
  and scientific eligibility.

Implementation stops fail-closed if it would require modifying a frozen
artifact, guessing an unknown publication result, accepting multiple writers,
using noncanonical or self-referential records, replaying the model, or
expanding into any prohibited execution surface.
