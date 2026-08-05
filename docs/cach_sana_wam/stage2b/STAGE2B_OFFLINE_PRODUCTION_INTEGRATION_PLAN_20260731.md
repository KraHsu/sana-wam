# CACH Stage 2B offline production-integration plan

Status: `offline_l0_l1_passed_l2_l3_replay_and_closure_blocked_non_scientific`

Canonical implementation host/worktree:
`H200:/home/zch/workspace/sana-wam`.

Stage 2B is a new, additive revision. It does not rewrite or reuse the frozen
Stage 2 evidence roots, manifest, or bundle. Its predecessor pins are:

- Stage 2 manifest SHA256:
  `8333c74d9bc8115937f98d9fc12a60da62cab49dd8895cd403047027315052fb`;
- Stage 2 source bundle SHA256:
  `71e73b3f2b1efb125f0b57f8c47913bab47d492920310f0a53848fe7903c229c`;
- final scoped evidence root:
  `/DATA/share/sana_cach_stage2_evidence/stage2_offline_subset_20260731_JKYEkuCw`.

Stage 2B is not Stage 3, a full Gate-S2 result, a training admission, a deploy
admission, or a scientific result.

Stage 2B is additive over the immutable Stage 1 source bundle. In particular,
these worktree files remain byte-identical to their Stage 1 pins:

- `src/sana_wam/model/causal_action_hybrid.py`:
  `e714ea6ade41223ed1b93e643cbd179fa63ff7ab48d568f36dc95e5eaab17560`;
- `src/sana_wam/model/video_backbone/sana/hybrid_cache.py`:
  `5b61ab6492670e8021b3603f965e9f3e8921e9d7f0ea77b49e40d92a77f78a74`.

The revision-specific implementation lives in new files:

- `src/sana_wam/cach/committed_action_history.py`;
- `src/sana_wam/cach/staging_variant.py`;
- `src/sana_wam/cach/stage2b_receipt_store.py`;
- `src/sana_wam/cach/stage2b_state_snapshot.py`;
- `src/sana_wam/model/cach_numerical_core.py`;
- `src/sana_wam/model/cach_paired_stager.py`;
- `src/sana_wam/model/video_backbone/sana/hybrid_cache_stage2b.py`.

`Stage2BTemporalState` is the single live authority pointer. It contains the
frozen typed `HybridTemporalState` as its cache component and the exact
committed-action history as its history component. The outer pointer, not the
two components independently, is CAS-published after one durable Stage 2B
receipt.

## 1. Governance drift recorded for this revision

The older development-plan snapshot says the H200 worktree lacked the two
governance entry files. That statement is no longer current. Before this
revision, H200 was rechecked and both paths were regular files:

- `START_HERE_20260730.md` SHA256
  `e85fa5be9f0d1941725cd356298493b9003ed2339d1a5cc00d5688ed8e51c061`;
- `AGENTS.md` SHA256
  `646ea39906ea3bc75af3b360187eb2c5aebac4c04c62ec5df108c0ec35e006ae`.

This record supersedes only that historical environment statement. It does
not modify the earlier frozen plan or evidence bytes.

## 2. Objective and ordering

The objective is to replace the test-owned paired mini harness with one
production-shaped offline teacher-forcing execution core while keeping all
ordinary training, checkpoint loading, deploy, server, and evaluation paths
fail-closed.

Implementation order is fixed:

1. canonical committed-action history identity and cache-summary binding;
2. production-shaped offline paired stager and model-owned `NO_ACTION`;
3. `REF-GDN-CORRECTED` identity-bypass commissioning path;
4. CACH-A action-conditioned path through the same stager;
5. production receipt/`ABORTED` ledger and deterministic restart replay;
6. production-path mini C0--C8 admission;
7. transitive runtime closure, exact inventory, and named-init parity;
8. only then may a separate Stage 3 authorization be requested.

Deploy ACK transport is deliberately not part of this slice. The standalone
ACK verifier remains available, but `CommitSource.DEPLOY_APPLIED_ACK`, CACH
deploy dispatch, checkpoint loading, policy/server wiring, and controller
transport remain hard-disabled.

## 3. Committed-action history contract

The old mini stager reconstructed history from its caller-owned list of future
requests. Stage 2B forbids that pattern. History must be owned by the same
manager that owns the live temporal-cache pointer and may advance only in the
same atomic publication as a durable paired commit.

`CommittedActionHistory` v1 binds:

- episode id and epoch;
- layout spec and instance digests;
- exact contiguous action cursor;
- ordered committed chunk ids and half-open action intervals;
- commit source and source-proof digest for every chunk;
- canonical tensor metadata and exact value digest for every valid action
  span;
- an ordered history manifest digest.

It exposes detached clones only. Padding is never appended. A continuation
view must cover exactly `[0, chunk.action_start)`. Missing, stale, reordered,
overlapping, gapped, command-only, wrong-source, wrong-dtype/device/width, or
digest-mismatched history fails closed.

The Stage 2B v1 CACH-A numerical history-summary operator is:

```text
gdn_recurrent_paired_commit_v1
```

The distinct `REF-GDN-CORRECTED` video-only recurrent summary operator is:

```text
gdn_recurrent_video_only_paired_commit_v1
```

Every committed action span is consumed once, together with its paired clean
video, by the registered end-of-latent-bin action conditioner during the sole
`t=0` staging pass. The resulting GDN recurrent temporal state is the causal
history summary. Raw past actions are not pooled, averaged, silently
reconstructed, or injected a second time into the current candidate. The live
state manifest must bind the committed-history digest so a cache state cannot
be paired with a different raw history.

For `REF-GDN-CORRECTED`, the action-to-video branch is an explicit identity
bypass and the reference must not claim an action-conditioned GDN history.
For CACH-A, every history member must have been staged through the registered
action conditioner. `CACHStagingVariant` is bound into the outer state,
denoise view, staging context, transaction digest, commit receipt, reset
receipt, and summary-operator identity. The production-shaped stager also
requires the adapter action seam to be explicitly disabled for the reference
and enabled with exact width/zero-init attestation for CACH-A. These modes
cannot share or reinterpret state or receipts.

## 4. Model-owned `NO_ACTION`

The CACH-A bootstrap condition must come from one registered, trainable model
parameter with exact-zero initialization. In this additive revision,
`CACHNumericalCore` owns that parameter and the production-shaped stager
receives that same core object; a test or caller may not substitute
`torch.zeros(...)`. The frozen Stage 1 architecture remains non-constructible
and is not rewritten to claim runtime ownership before a later admission.

The slot is valid only for bootstrap latent 0. It:

- does not occupy an action token or label;
- does not advance Action RoPE or the committed cursor;
- is not reused for partial-tail padding;
- is not reset at episode boundaries;
- must remain finite and exactly match model dtype/device/action width.

`REF-GDN-CORRECTED` retains the parameter-matched core object but its
action-to-video identity-bypass branch does not numerically read `NO_ACTION`.
The action prediction stream, exact action history identity, Action RoPE, and
paired video cache commit remain active in both variants.

Ordinary construction remains denied. Any mini integration factory/capability
must be visibly synthetic, nontraining, nondeploy, and unable to authorize a
checkpoint, dataset, optimizer, root, or full model.

## 5. Production-shaped paired staging contract

The offline stager must consume only a manager-created
`PairedStagingContext`. It may not obtain history from an out-of-band request
list. It must:

1. verify content-time, layout, source proof, conditioning, and committed
   history against the same expected revision;
2. encode the detached typed scratch into a fresh transaction-local vendor
   `list[10]` cache;
3. run video and action exactly once at `t=0`;
4. use the model-owned `NO_ACTION` only on the CACH-A bootstrap path;
5. use exact Action RoPE `[action_start, action_end)` and fixed-capacity prefix
   compaction/restoration;
6. decode the transaction-local vendor cache into typed staged payloads;
7. leave the live state, input pair, read view, and source history unchanged.

Only the manager may publish a durable receipt and atomically swap the live
state/history pair. A staging failure leaves pointer, revision, cursor, and
history unchanged.

## 6. Fail-closed boundaries for this revision

The following stay prohibited:

- real 2B/VAE/text/checkpoint/data/HDF5 access;
- optimizer construction, training, evaluation, formal capture, or formal
  experiment root creation;
- Stage 3 full-model execution;
- CACH checkpoint loader, deploy engine, policy server, or controller ACK
  transport enablement;
- self-forcing, AttnRes, causal softmax anchors, async inference, temporal
  ensemble, or half-chunk regeneration;
- changes to pinned Sana, AFCC, old bundles, or old evidence roots.

No error may fall back to a legacy fixed-ATC path, command history, approximate
replay, loss subtraction, or a test-local literal `NO_ACTION` tensor.

## 7. Admission for the first implementation slice

The initial committed-history slice is complete only when lightweight tests
prove:

- bootstrap history is empty and continuation history is exact and contiguous;
- only valid action prefixes are appended;
- source proof, layout, episode/epoch, interval, dtype, shape, order, and value
  identity are digest-bound;
- duplicate same-payload publication is idempotent and conflicting reuse is
  rejected;
- failed staging leaves both live cache and history byte-identical;
- successful durable publication advances cache revision, action cursor, and
  history in one observable step;
- reset clears history and stale views cannot be reused;
- no caller-owned future request list can supply committed history.

Passing this slice does not close full production architecture/stager
admission, ledger/replay, complete Gate-S2, or Stage 3.

## 8. Implementation checkpoint on 2026-07-31

The first offline implementation slice now has code and lightweight CPU
coverage for:

- manager-owned valid-only committed-action history and one outer atomic
  cache/history pointer;
- mutation detection including mutate-then-restore version changes;
- staging failure, durable-publication failure, retry, reset-during-stage,
  publisher re-entry, lazy iterable, receipt-ID collision, and stale-view
  failure paths;
- model-owned CACH-A `NO_ACTION`, exact reducer equivalence, fixed-prefix
  padding, absolute Action RoPE, and two-chunk prior-cache continuation;
- explicit `REF-GDN-CORRECTED` action-to-video identity bypass and strict
  variant separation across state/context/digest/receipt/reset/read view;
- non-synthetic manager rejection of arbitrary callbacks and missing
  conditioning identity;
- production adapter action-seam enablement/disablement bound to the typed
  variant;
- a strict L0 filesystem receipt store with canonical Stage 2B commit/reset
  parsing, cross-field digest/transition checks, hashed global IDs,
  exclusive-create, write-all, file/directory fsync, mode `0400`, stable
  same-inode read-back, and root-replacement detection;
- direct manager-to-filesystem commit/reset receipt round trips, plus injected
  partial writes, fsync/read-back failures, root replacement, symlink,
  hard-link, FIFO, directory-entry, collision, and malformed-schema cases;
- a CPU/synthetic-only L1 state snapshot codec using canonical JSON and
  unique sorted content-addressed raw tensor blobs, with little-endian exact
  BF16/FP32/bool bytes, layout/registry/variant/logical-device/logical-tensor
  identity, full cache `ContentTime`, and valid-only committed-history
  reconstruction;
- exact empty, one-chunk, continuation/partial-tail, and both-variant snapshot
  round trips, plus fail-closed missing/replaced/extra/duplicate/trailing blob
  and metadata/value/layout/registry/history/variant corruption cases.

On canonical H200, the final in-place check for this checkpoint was:

- Stage 2B focused CPU tests: `63 passed, 0 failed`;
- all `tests/test_cach_*.py`: `305 passed, 10 skipped, 1 deselected, 0 failed`;
- Ruff on the Stage 2B implementation/test set: passed;
- the predecessor Stage 2 verifier, including Stage 1/2 bundles and retained
  evidence: passed with manifest SHA256
  `8333c74d9bc8115937f98d9fc12a60da62cab49dd8895cd403047027315052fb`.

The one deselected test is the known Stage 1-only worktree-artifact pin check;
the current `adapter.py` is the already-frozen Stage 2 overlay and is verified
by the Stage 2 manifest. The frozen Stage 1 architecture/cache source hashes
listed above remain exact.

No dataset, checkpoint, optimizer, GPU job, training, evaluation, formal
capture, or formal root was used or created. These command results have not
yet been frozen as a new immutable Stage 2B evidence root.

The following remain hard blockers and are not claimed by this checkpoint:

- durable content-object/snapshot publication, a production
  `PREPARED`/`COMMITTED`/`ABORTED` transaction ledger, decision chaining,
  unknown-outcome reconciliation, manager recovery, and deterministic
  restart; the current L1 codec is in-memory and CPU/synthetic-only;
- trusted-parent/ancestor fencing for the L0 receipt root and H200 `/DATA`
  filesystem fault/power-loss admission; the current store validates the root
  inode/owner/mode but is not a multi-process or restart authority;
- a reconstructible Sana vendor bundle or complete dynamic-import/runtime
  closure; inherited Stage 1/2 manifests themselves state
  `transitive_runtime_closure=false`;
- full initialized 20-layer production architecture execution, named-init
  parity, real-shape/dtype full-model tests, and complete C0--C8 evidence;
- deploy ACK transport/commit, checkpoint/data admission, Stage 3, training,
  evaluation, deployment, or any scientific claim.
