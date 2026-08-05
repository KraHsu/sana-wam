# CACH Stage 2B L3 deterministic recovery implementation plan

Status: `implementation_authorized_after_plan_review_p0_0_p1_0_not_admitted`

Canonical implementation host/worktree:
`H200:/home/zch/workspace/sana-wam`.

This document is the source of truth for the next additive Stage 2B protocol
slice. In this document, **L3 means the Stage 2B ledger sublayer for
deterministic restart recovery**. It is not the project-wide Stage 3
full-model update-free causal proxy described in
`docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md`.

This plan does not grant recovery, filesystem, runtime, training, evaluation,
deployment, capture, full-model, Stage 3, or scientific authority. Independent
seam and crash-contract review completed with no open P0 or P1 plan finding;
that result authorizes only the additive CPU/synthetic implementation and
lightweight tests described here.

## 1. Pinned source lineage and immutable boundary

The caller-pinned predecessor is the completed L2B additive source checkpoint:

- L2B outer manifest:
  `docs/cach_sana_wam/stage2b_l2b/SOURCE_MANIFEST.json`,
  SHA256
  `5e577cc857de232529ea8737321643ba07a9366d2fba0c519c322da722b20487`;
- L2B verifier:
  `scripts/verify_cach_stage2b_l2b.py`,
  SHA256
  `6fb42de438057a1fb6fe9900872d1ba3158a009aaa00126123d3dd66a15f8ce3`;
- L2B deterministic bundle:
  `/DATA/share/sana_cach_source_bundles/cach_stage2b_l2b_f2380eabfe06daeb7087dc139cd67441e2164b27f91bc71ff6295b7fc8e43533_20260731.tar`,
  SHA256
  `f2380eabfe06daeb7087dc139cd67441e2164b27f91bc71ff6295b7fc8e43533`,
  size `194560`, mode `0444`, one regular link;
- resolved predecessor inventory: `112` paths with SHA256
  `9c789439b729debdf343535e785a64c31efafa4f18a3f84ecdf36226c4144445`;
- resolved inventory after L2B: `119` paths with zero predecessor
  collisions.

The direct implementation pins are:

| Path | SHA256 |
|---|---|
| `src/sana_wam/cach/stage2b_offline_ledger.py` | `030a67332310d695395729161de7eb26999e391d469ff24b9c2d532666e77da6` |
| `src/sana_wam/cach/stage2b_state_snapshot.py` | `269951d7c7e3145ffaad146eb580dc5e41d0572dccea11eadc23da2a3072805c` |
| `src/sana_wam/cach/stage2b_receipt_store.py` | `8d8f313c5d08622c74b313d32249fbb0f79284e6f1f0c6a9c36590e4f9d3b83f` |
| `src/sana_wam/cach/committed_action_history.py` | `162d7e7ee866b18cdba0f0e319b2c9b277d73d933ff5a4dd38c92ebcc762b5e9` |
| `src/sana_wam/model/cach_paired_stager.py` | `c1455af51df4e0a0eff9971fa98e9272aac580084029a7990161817d246a6ef4` |
| `src/sana_wam/model/video_backbone/sana/hybrid_cache.py` | `5b61ab6492670e8021b3603f965e9f3e8921e9d7f0ea77b49e40d92a77f78a74` |
| `src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b.py` | `8ab968b77660df1092b43a52eecb4dba16316060e7f83c0ad6fd214230a8a5dd` |
| `src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b_ledgered.py` | `716130f7fbe352cc2e8c179a33edb4cf904b869823ce6261367c499d7e35816d` |

The direct design-contract pins are:

| Path | SHA256 |
|---|---|
| `docs/cach_sana_wam/stage2b/STAGE2B_DURABLE_LEDGER_AND_REPLAY_DESIGN_20260731.md` | `f83362e69ecc27c913c8ff2d4cedee8c1e8c1609cad0786e826079f43f4f4799` |
| `docs/cach_sana_wam/stage2b_l2/STAGE2B_L2_OFFLINE_LEDGER_IMPLEMENTATION_PLAN_20260731.md` | `35ef9c283d1802c781799fcb30ace0fb69155aad9f78d963946349a14ece5286` |
| `docs/cach_sana_wam/stage2b_l2/STAGE2B_L2B_LEDGERED_MANAGER_IMPLEMENTATION_PLAN_20260731.md` | `9aa65a9aea456ff87a001024ee0c9d4ea0ed267d474091f809a17e8121692d5c` |

The L2A outer manifest remains fixed at
`5d27a778bff120ff7a27c18ebaffbfb1761b2ced3d5f2e4c46a4de8c5763b8bf`.
The Stage 2B L0/L1 manifest remains fixed at
`87a5f36d13e06cdd5d951cf1dd96f4e353edc50fc555c96b3269d19859126c0a`.

L3 must not edit any file in those checkpoints. The initial additive scope is
restricted to new paths:

```text
docs/cach_sana_wam/stage2b_l3/STAGE2B_L3_DETERMINISTIC_RECOVERY_IMPLEMENTATION_PLAN_20260731.md
src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b_recovery.py
tests/test_cach_stage2b_recovery.py
```

Verifier, report, source-list, manifest, and bundle files may be added only
after implementation, focused tests, the complete lightweight CACH regression,
and independent review are green.

## 2. Established predecessor behavior

The following are source-audited facts, not assumptions:

1. `Stage2BOfflineLedger.reopen` validates the fixed identity and genesis
   snapshot and calls `scan_valid_chain` before returning.
2. The pinned private `Stage2BOfflineLedger._scan_records` validates one
   sequence-ordered chain from genesis, exact predecessor hashes, state
   before/after transitions, operation identities and payloads, terminal
   receipts, committed snapshots, and every referenced tensor blob. It allows
   at most one terminal-free PREPARED intent.
3. `_state_after_decisions` returns the last COMMITTED snapshot, or genesis
   if no COMMITTED decision exists. ABORTED decisions preserve predecessor
   state.
4. `Stage2BOfflineLedger.abort` can close a PREPARED intent with registered
   `abort_code=PROCESS_EXIT` and
   `failure_point=AFTER_PREPARED`.
5. L2B is deliberately fresh-only. It rejects a non-genesis ledger and owns
   only same-process completed maps.
6. The frozen manager constructor creates only an empty state. It has no public
   state-install or completed-map reconstruction method.
7. Recovery can be implemented without executing the model, but only through
   additive code which continues to pin private L2A, L2B, and frozen-manager
   seams.

## 3. Scope and public surface

The first L3 implementation remains:

- CPU and synthetic layout only;
- one process, one caller-owned ledger, one live facade by contract;
- entered only after the caller guarantees that the old writer process has
  exited and that no other live `Stage2BOfflineLedger` instance or facade
  names the same root;
- model-free and optimizer-free;
- based on a caller-reopened `Stage2BOfflineLedger` whose complete identity,
  layout inventory, registry, genesis snapshot, and writer-fence identifier
  have already been supplied and verified;
- limited to deterministic reconstruction and continued lightweight synthetic
  commits/resets.

The proposed public type is:

```text
Stage2BRecoveredLedgeredHybridCacheManager
  .recover_from_ledger(
      *,
      registry: LayerRegistry,
      ledger: Stage2BOfflineLedger,
      synthetic_test_only: bool = False,
  )
  .for_synthetic_recovery(*, registry, ledger)
  .recovery_summary
```

It subclasses the pinned L2B facade so that the existing public
`state`, denoise, commit, and reset surfaces remain unchanged. Its direct
constructor is capability-guarded. It accepts neither a model nor a staging
callback during recovery and exposes no raw-manager or mutable-state getter.

`synthetic_test_only=False` fails with `L3_CPU_SYNTHETIC_ONLY` before
ledger mutation or manager construction. The safe default is therefore
non-authorizing. The named synthetic factory opts in explicitly.

The immutable in-memory `recovery_summary` contains only primitive
diagnostics:

- ledger ID and writer-fence ID;
- decision count and last decision SHA256;
- recovered state manifest, episode, epoch, revision, and layout digest;
- reconstructed committed-commit and committed-reset counts;
- the transaction ID closed as PROCESS_EXIT, or `null`;
- sorted unreferenced content-object SHA256 values;
- `model_executed=false`, `recovery_admitted=false`,
  `filesystem_admitted=false`, and `scientific_eligible=false`.

It is not an execution receipt or authority.

## 4. Recovery state machine

```text
UNOPENED
  -> LEDGER_LOCKED
  -> CHAIN_VALIDATED
  -> PENDING_CLASSIFIED
  -> [PROCESS_EXIT_ABORT_PUBLISHED]
  -> CHAIN_REVALIDATED
  -> RECEIPTS_RECONSTRUCTED
  -> DETACHED_MANAGER_BUILT
  -> STATE_INSTALLED
  -> COMPLETED_MAPS_INSTALLED
  -> FINAL_CHAIN_REVALIDATED
  -> READY

any failure before READY -> no facade escapes
```

There is no recoverable partially constructed public object. A failure may
leave only an exact immutable PROCESS_EXIT abort or content-addressed object
already published by the old process. The caller must reopen and retry from
the ledger; L3 never guesses an in-memory state.

## 5. Exact recovery algorithm

All steps from the first scan through the final scan execute while holding the
exact ledger instance's pinned `_lock`. This is only a same-process
serialization boundary and is not a multi-process fence.

### 5.1 Validate and classify

1. Require exact L2A ledger, registry, synthetic provenance, staging variant,
   registry digest, layout-spec digest, and non-empty caller-pinned layout
   inventory.
2. Call the pinned `_scan_records` and require an exact
   `(intents, decisions, ordered, pending)` result.
3. Record the stable chain count and last decision SHA256.
4. If a terminal decision exists for an intent, it is authoritative even if a
   previous process crashed before its local CAS.
5. If exactly one PREPARED intent lacks a terminal decision, close that exact
   intent with:

```text
result = ABORTED
abort_code = PROCESS_EXIT
failure_point = AFTER_PREPARED
state_manifest_after = state_manifest_before
snapshot_manifest_sha256 = null
```

6. Before publication, construct the exact PROCESS_EXIT abort receipt, decision
   body, decision record, and expected terminal-record SHA256 from the pinned
   intent. If `abort()` raises, call `reconcile_terminal()` only with that
   expected SHA256. EXACT may continue; stable ABSENT, conflict, mutation, or
   unreadable state returns no manager. A later recovery invocation must
   rescan and may retry the still-pending intent. It is forbidden to accept a
   different terminal or guess success from an exception.
7. Rescan and require no pending intent and one additional exact ABORTED
   decision when step 5 occurred.
8. Only after that closure rescan, enumerate the exact stable content-object
   set and compute authoritative reachability from the genesis snapshot
   manifest and transitive blobs, every intent operation identity and payload,
   every terminal success/abort receipt, and every COMMITTED snapshot manifest
   and its transitive blobs. Missing or mutated reachable content is fatal;
   only valid unreferenced content-addressed objects are inert diagnostics.

This rule does not overwrite an unknown terminal. A decision that completed
its no-replace publication is visible to the first valid scan. An absent
decision may be converted to PROCESS_EXIT only inside this single-writer
CPU/synthetic scope. Multi-process races remain an admission blocker.

### 5.2 Resolve the live state and layout

1. Reconstruct the current state from the validated decision chain using the
   pinned snapshot decoder; never rerun a request, stager, video model, action
   model, or denoise path.
2. Use the exact snapshot layout identity to select a layout from the
   caller-pinned ledger layout inventory.
3. Require complete equality of the decoded state manifest, episode, epoch,
   revision, cache/history manifests, layout identity, registry identity, and
   staging variant with the final authoritative chain state.
4. Clone or detach the recovered state before installation so no caller-owned
   tensor or manifest can mutate the live pointer.

### 5.3 Reconstruct exact receipts and retry identities

For each COMMITTED decision:

1. read its referenced success receipt object;
2. run the pinned L0 strict canonical parser;
3. reconstruct the exact frozen
   `Stage2BPairedCommitReceipt` or
   `Stage2BCacheResetReceipt` with explicit enum conversion;
4. require reconstructed `canonical_bytes` and receipt SHA256 to equal the
   stored bytes and decision reference;
5. read and validate the corresponding operation identity and operation
   payload.

The L2B facade completed maps are reconstructed as:

- paired commit:
  `operation_identity.operation_payload_sha256, receipt,
  terminal_decision_sha256`;
- reset: the exact pinned L2B reset retry digest derived from reset ID,
  transaction nonce, new episode ID, target layout, and staging variant,
  plus receipt and terminal decision SHA256.

The underlying frozen manager maps are reconstructed as:

- paired commit: `receipt.paired_payload_digest, receipt`;
- reset: the same exact reset retry digest and receipt.

ABORTED transaction IDs are not inserted as success. They remain permanent
global tombstones because the ledger terminal lookup rejects reuse before a
new PREPARED intent can be published.

The initial L3 contract preserves the frozen L2B retry behavior: an exact
committed retry can return its durable receipt only when it is still compatible
with the live episode/layout checks. A commit or reset from an earlier episode
remains a global tombstone but may fail as stale rather than return historical
success. L3 must not override this behavior silently. Broad historical-success
replay across resets would require a separately designed public retry API.

### 5.4 Install before publication

Construct a new private seamed manager with the recovered episode, epoch,
layout, registry, variant, and a fresh private publisher adapter. Before any
reference escapes:

1. hold the new manager's lock;
2. require its empty state, empty maps, no pending transaction, no publication
   token, and non-poisoned status;
3. install the detached recovered state;
4. install its exact state-manifest digest and current layout;
5. install the reconstructed frozen-manager completed maps;
6. require no pending transaction, no publication token, and non-poisoned
   status again;
7. construct the capability-guarded recovered L2B facade;
8. install the facade completed maps while it is still private;
9. require facade phase READY, no active transaction, publisher token
   disarmed, and no poison;
10. read the public state and require exact manifest equality with the decoded
    authoritative state.

Finally, rescan the ledger under the same ledger lock and require the exact
decision sequence, final SHA256, and exact content-object digest set observed
after pending closure. Construct the summary only from that post-closure final
stable chain and object set. Only then may the ready facade and non-authorizing
recovery summary return.

## 6. Content-object debris decision

The predecessor design used “orphan” for two incompatible cases. It required
rejection of orphan objects, but its crash matrix also required recovery to
the predecessor state when content-addressed objects were durably written
before a terminal decision.

L3 resolves the ambiguity as follows:

- an object referenced by genesis, an intent operation identity/payload, a
  terminal receipt, or a COMMITTED snapshot is **authoritative referenced
  content**; absence, mutation, filename/hash mismatch, unsafe file type,
  duplicate logical identity, or decode mismatch is fatal;
- an unexpected filename, symlink, hard link, non-regular entry, wrong hash, or
  changing object is fatal;
- a single-link regular `object-<sha256>` whose bytes exactly hash to its name
  but which is not reachable from the validated chain is **inert CAS debris**.
  It is never decoded as state, never establishes success, never changes
  ordering, and is reported by sorted digest only;
- inert CAS debris is not deleted, renamed, adopted, or used to infer an abort.
  It remains compatible with a crash after object publication and before the
  terminal decision.

This distinction is mandatory for the crash matrix. If independent review
requires rejecting every unreferenced CAS object, L3 implementation remains
blocked until a new immutable quarantine/closure protocol is designed; it
must not delete frozen evidence to make the root appear clean.

## 7. Private seams and review debt

The zero-collision implementation necessarily pins these private seams:

From L2A:

```text
Stage2BOfflineLedger._lock
Stage2BOfflineLedger._scan_records
Stage2BOfflineLedger._state_after_decisions
Stage2BOfflineLedger._load_snapshot
Stage2BOfflineLedger._layout_for_snapshot_manifest
Stage2BOfflineLedger._expected_layouts
Stage2BOfflineLedger._validate_intent_input
_parse_operation_identity
_parse_operation_payload
_parse_abort_receipt
_collect_snapshot_blob_references
```

From L2B:

```text
_CONSTRUCTION_CAPABILITY
_CaptureBridge
_LedgerPublisherAdapter
_LedgerSeamedStage2BHybridCacheManager
Stage2BLedgeredHybridCacheManager.__init__
Stage2BLedgeredHybridCacheManager._reset_retry_digest
```

From the frozen manager:

```text
_lock
_state
_state_manifest_digest
_layout
_pending
_completed
_completed_resets
_publication_token
_poisoned
```

From the frozen L0 receipt parser:

```text
stage2b_receipt_store._validate_payload
```

The implementation must assert all expected pre-install values and exact
post-install invariants. Any predecessor hash change invalidates this plan.
A later public two-phase/state-install API is preferable, but modifying a
frozen checkpoint is prohibited in this slice.

## 8. Required lightweight CPU/synthetic tests

The focused test file must cover at least:

1. exact genesis recovery, empty maps, exact state/layout, and no model call;
2. one and two committed chunks recovered byte-equivalent to the last
   snapshot, followed by one valid continued commit;
3. terminal-durable/pre-CAS crash recovery installs the terminal snapshot even
   though the old manager remained at predecessor state;
4. one PREPARED intent becomes exact
   `PROCESS_EXIT/AFTER_PREPARED` ABORTED, preserves predecessor state, consumes
   the ID, and permits a new ID;
5. recovery re-entry after an exact PROCESS_EXIT close is deterministic and
   does not append another decision;
6. PROCESS_EXIT abort publication exceptions reconcile only the precomputed
   exact terminal SHA; exact may continue while absent/conflict/unreadable
   returns no facade;
7. pending commit with inert content-addressed debris reports the sorted
   digests but never installs the debris;
8. commit-reset-commit recovery selects the final layout/state and preserves
   global commit/reset/abort tombstones;
9. exact current-episode commit/reset retry after restart returns the exact
   receipt without staging and without changing ledger bytes;
10. different reuse and commit/reset cross-kind reuse fail before ledger
   mutation;
11. strict paired/reset receipt reconstruction and byte equality;
12. missing/substituted receipt, snapshot, blob, operation identity, or
    operation payload fails before manager construction;
13. gap, fork, duplicate sequence/ID, mixed ledger/variant/fence, stale
    predecessor, state-changing abort, and unexpected entry fail closed;
14. wrong registry, layout inventory, staging variant, synthetic provenance,
    and non-CPU state fail before install;
15. injected install or final-rescan failure returns no facade;
16. recovered facade is READY, publisher disarmed, and POISONED remains
    absorbing after any later ambiguous publication;
17. recovery API has no model/stager/request argument and model-execution spies
    remain untouched.

Tests use only `tmp_path`, CPU synthetic layouts/tensors, and local immutable
objects. They create no formal `/DATA` ledger root and do not use a dataset,
checkpoint, model weight, optimizer, GPU, training loop, evaluator, deploy
transport, or capture path.

After focused tests pass, run on canonical H200:

- `py_compile` and Ruff over only the new L3 module/test first;
- the L3 focused suite;
- fixed L2A and L2B focused regressions;
- the complete lightweight `tests/test_cach_*.py` suite with only the
  already-recorded predecessor deselection.

No full model, CUDA, dataset, checkpoint, training, evaluation, or formal root
is part of this gate.

## 9. Stop conditions and remaining blockers

Implementation stops before code if independent review finds any P0/P1 in:

- pending-intent terminal classification;
- inert-debris versus authoritative-reference closure;
- exact receipt parsing or retry-digest reconstruction;
- current layout selection;
- manager state/map installation;
- terminal chain stability before returning a facade;
- any path that would require editing a frozen predecessor.

Even a green L3 CPU/synthetic suite does not establish:

- multi-process writer fencing or exclusive ownership of one ledger root;
- proof that the old writer is dead, or prevention of a second ledger object
  with an independent Python `RLock` opening the same root;
- same-UID hostile mutation resistance;
- trusted parent/ancestor ownership, sticky-bit, mount, rename, or device
  policy;
- admitted H200 `/DATA` fsync, no-replace, lock, hard-link, power-loss, or
  remount behavior;
- CUDA synchronization/device mapping or production 20-layer snapshot
  latency, memory, and I/O bounds;
- complete Sana/runtime/dynamic-import/native-library/Git recovery closure;
- real layout/timebase, named-init parity, full architecture, C0--C8, or
  project-wide Stage 3 admission;
- dataset/checkpoint/deploy ACK authority, training, evaluation, capture, or
  scientific eligibility.

The baseline/formal launchers remain separately blocked by their existing
contract and admission audit. L3 source or tests cannot authorize a 504-step
run or any prospective formal experiment.
