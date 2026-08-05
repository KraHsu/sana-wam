# Production-shaped mini C0--C8 implementation checkpoint

Authorization/work-item date: 2026-08-02  
Final CPU verification record: 2026-08-03 Asia/Shanghai
(`2026-08-02T17:18:46Z`)  
Canonical host: `H200`  
Canonical worktree: `/home/zch/workspace/sana-wam`

## 1. Authority and result boundary

The user authorized only:

1. additive development-document and source-manifest updates;
2. implementation of a production-shaped mini harness;
3. CPU synthetic lightweight tests.

GPU use, the complete 2B graph, checkpoint or real-data access, optimizer or
trainer construction, training, evaluation, capture, formal admission roots,
and scientific claims remained forbidden.  The implementation does not add a
production/train/deploy capability to the ordinary architecture builder.

The resulting hard CPU suite passes its scoped interface checks.  This is not
the complete closure required by Section 11 of
`GLOBAL_STAGE2_CLOSURE_AND_STAGE3_UPDATE_FREE_ADMISSION_PLAN_20260802.md`.
Two strict requirements remain open:

- the CPU recurrent implementation is a pure-Torch transition proxy rather
  than the exact vendored CUDA/Triton GDN with only smaller tensor sizes;
- the additive test-named dispatcher and ephemeral owner are not the exact
  future G3 public `build_architecture` and cache-owner entrypoints.

The accurate checkpoint name is therefore **Phase-C production-shaped
interface-proxy closure**, not production-path admission and not Global Stage
3 admission.

## 2. Implemented graph

Each independently allocated arm contains:

- exact typed variants `REF-GDN-CORRECTED` and `CACH-A`;
- one shared, model-owned `cach_core.no_action_slot`, CPU FP32 and exact zero;
- one `SanaVideoBackbone.run_chunk` public adapter surface, including fixed-K
  valid-prefix compaction and restoration;
- 20 recurrent layers registered as `GDN_FULL_HISTORY`, zero softmax layers,
  camera disabled, and four typed fields per layer:
  `main_s_kv`, `main_s_z`, `main_shortconv_left_context`, and
  `ffn_tconv_left_context`;
- actual repository `ActionDiT` with 20 bridge layers and absolute Action RoPE;
- the existing `CACHNumericalCore`, `CACHChunkConditioning`,
  `scratch_to_vendor_cache` and `vendor_cache_to_staged_payloads` path;
- exact `Stage2BDenoiseReadView` and `Stage2BPairedStagingContext` types;
- existing `CommittedActionHistory` ownership and valid-prefix append logic;
- one in-memory single-pointer state owner that performs no durable or
  controller publication, reserves one pending operator callback, and performs
  one compare-and-swap after complete payload materialization.

The tensor-size recipe is deliberately small: batch 1, latent channels 3,
hidden width 4, spatial latent 1x1, fixed `K=3`, action width 20, depth 20 and
FP32 CPU.  The canonical layout preserves temporal compression 8 and has
three chunks with valid latent counts `[3, 3, 2]`, valid action counts
`[16, 24, 16]`, and action intervals `[0,16)`, `[16,40)`, `[40,56)`.

The recurrent codec shapes preserve the small-dimension form of the pinned
vendor slots: `main_s_kv=[B,1,C,C]`, `main_s_z=[B,1,C,1]`, GDN short-conv
context `[B,3,C]` for kernel size 4, and FFN temporal context `[B,C,1,1]` for
kernel size 3 at spatial size 1x1.  The arithmetic inside those states remains
the explicitly non-parity pure-Torch proxy.

The reference structurally supplies `action_condition=None` to video.  The
candidate has a separately named action embedder and 20 post-recurrence
adapter projections.  All 40 adapter weight/bias tensors are exact zero.  The
shared key intersection has byte-identical named initialization and disjoint
storage; the candidate-only inventory contains exactly the registered action
embedder and adapter tensors.

## 3. C0--C8 evidence and remaining gaps

| Contract | Phase-C scoped evidence | Status beyond this harness |
|---|---|---|
| C0 causal visibility | Clean/noisy tensors are independently generated and readonly timesteps are nonzero. Probe chunk 1 is exact invariant to its own clean target and all chunk-2 inputs; the changed clean target demonstrably changes chunk 2, while separate future-video/action perturbations demonstrably change their own future outputs. Earlier outputs, reverse gradients, None patterns and future-input JVP remain exact invariant/zero. The dispatcher seam is checked for both arms. | Complete-model and exact vendor-operator evidence remain G3-owned and unauthorized. |
| C1 bypass identity | REF and zero-init CACH video/action outputs and all 20 final cache payloads are raw-exact. Action perturbations leave both video/cache paths unchanged at theta0. All 40 zero adapter tensors receive finite nonzero reverse gradients; a combined parameter JVP is finite/nonzero. Every parameter/buffer digest and version remains unchanged and `.grad` stays `None`. | Exact complete-2B shared-input identity remains absent. |
| C2 full/chunk equivalence | The independent layer-major oracle holds a clean recurrent prefix, predicts each current noisy chunk, then advances a separate clean state. It is raw-exact against chunk-major readonly(nonzero-t noisy) then paired-commit(t=0 clean) orchestration for both arms, including every output and final typed cache/history tensor. | This proves proxy/interface equivalence only, not vendor GDN numerical equivalence or train/deploy equivalence. |
| C3 content-time identity | One object-identical layout is shared by pair, episode, dispatcher, conditioning, Action RoPE, content time, history and every layer state. Runtime captures verify all six actual video intervals, Action RoPE ranges and reducer history/current-action extents. Mask/layout/RoPE and forged live episode/epoch fail before the operator callback and leave state unchanged. | Synthetic layout provenance is explicit; production timebase verification remains unavailable. |
| C4 bootstrap semantics | Build drift and unbounded size fields are rejected before graph allocation. Inherited checkpoint/mini-pipeline constructors, device/dtype transforms, state loading and train mode are sealed. The ordinary builder is actually called and rejects CACH. `first_frame_pinned`, zero observed-prefix, one latent-0 `NO_ACTION` anchor, model ownership and REF bypass are exercised. | The ordinary production builder remains refusal-only. |
| C5 source closure | The additive manifest pins the 126-path Stage2B-L3 predecessor, all Phase-C members and registered direct dependencies; its torch-free verifier checks caller pin, bytes, hashes, sizes, zero collisions and report limitations. | Transitive Python/native/dynamic-import/runtime closure is explicitly `false`; global C5 is not closed. |
| C6 action coverage | Bootstrap, exact-K, continuation and partial-tail layout arithmetic are checked. The canonical three-chunk bootstrap/continuation/tail case executes through the runtime trace; its action ownership is exactly `range(56)` with no gap/duplicate, exact RoPE cursor, prefix masks and zero padding. The one-chunk exact-K and bootstrap-tail alternates are layout-only checks. | Real episode/rate/data closure remains deferred. |
| C7 commit atomicity | Repeated denoise does not change pointer, revision, history or cache. A completed paired request swaps cache/history exactly once. Failures are injected before vendor, after vendor, and after payload materialization/before CAS; every case leaves the live pointer unchanged. A pending reservation prevents a second callback. Duplicate ID and stale view reject, while dispatcher reset updates episode/epoch and successfully recommits chunk 0. Failure receipt is exclusive and frozen only below pytest `/tmp`. | The owner is intentionally ephemeral: no durable ledger, deploy ACK or controller commit is claimed. |
| C8 hybrid cache | Exact 20-row all-GDN table; all four fields with pinned small-shape rank/axis/window contracts traverse the actual `list[10]` codec; every commit advances all 20 `through` values; final full/chunk clean cache is raw-exact; reset empties all layers and stale views fail. | The transition arithmetic is still a proxy: vendor kernel numerical parity, complete-model resource behavior and future G3 dispatcher closure remain blocked. |

The machine report's `phase_c_*_evidence_passed` and scoped-suite status refer
only to the middle column.  They must not be interpreted as a global gate
transition.

## 4. Hard test contract

The focused suite is one fixed file with 11 collected tests and no skip,
xfail, xpass, deselection or CUDA conditional.  It is invoked with:

```bash
env CUDA_VISIBLE_DEVICES='' \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -B -m pytest \
  -p no:cacheprovider --strict-markers -x -vv \
  tests/test_cach_production_path_mini.py \
  --basetemp=/tmp/<new-exclusive-parent>/pytest
```

The suite asserts `torch.cuda.is_initialized() is False`; it constructs no
optimizer/trainer and performs no backward update.  Gradients are obtained by
`torch.autograd.grad`; JVP uses registered PyTorch functional transforms.

The immutable temporary failure root is test evidence only.  A registered
pytest finalizer restores its permissions and deletes the frozen subroot after
the assertions; the exclusive basetemp parent is then explicitly cleaned after
the run.  It is not retained as a formal, admission or scientific root.

The frozen inherited `HybridTemporalState` admits only the literal
`pair_evidence_mode="dataset_ground_truth"` for teacher forcing.  Phase C uses
that field as a **synthetic schema shim** because no real dataset is accessed.
It is not dataset/source-proof evidence, is not a production provenance claim,
and cannot contribute to global C3/C5 admission.

## 5. Source checkpoint semantics

`SOURCE_MANIFEST.json` excludes itself and is trusted only when its SHA256 is
supplied out of band to the read-only verifier.  The verifier first executes
the exact pinned Stage2B-L3 verifier, reconstructs its resolved 126-path
inventory, confirms zero collision with the Phase-C members, verifies direct
runtime-dependency pins and validates the report's explicit capability
authorization booleans, which remain `false` outside the three allowed items.

The verifier deliberately:

- imports no torch and executes no model;
- does not replay the historical test command;
- creates no archive, source bundle or root;
- claims no independently reconstructible Python/native runtime;
- treats the test report only as a byte-pinned historical attestation.

## 6. Global state after this checkpoint

The following states remain exact:

```text
global GATE-S0 = NOT_CLAIMED
global GATE-S1 = NOT_CLAIMED
global GATE-S2 = BLOCKED_NOT_CLAIMED
global Stage 3 / Phase D = NOT_AUTHORIZED
transitive runtime closure = false
formal/scientific/training/evaluation/deploy authority = false
```

The gate-ownership reconciliation proposal has not been caller-approved.
Phase D cannot be requested from this checkpoint.  A later decision must
either revise the strict Section 11 wording to admit this registered CPU
interface proxy, or separately implement and authorize the exact future G3
dispatcher/operator path.  That decision is outside the present authority.
