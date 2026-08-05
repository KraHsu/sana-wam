# Global Stage-2 closure and Stage-3 update-free admission plan

Status: `DRAFT_DESIGN_ONLY_GLOBAL_STAGE3_NOT_AUTHORIZED`

Recorded on: `2026-08-02`

Canonical host: `H200`

Canonical worktree: `/home/zch/workspace/sana-wam`

## 0. Authority boundary

This document is an additive planning artifact. It does not grant authority to:

- construct or execute the complete 2B model;
- reserve or use a GPU;
- create an admission, experiment, training, capability/scientific evaluation,
  deployment, capture, or formal ledger root;
- load or save a checkpoint, model weight, dataset, HDF5 file, VAE output, or
  text-encoder output;
- create an optimizer, update a parameter, train, run a capability/scientific
  evaluation, deploy, or capture;
- claim global `GATE-S0`, `GATE-S1`, `GATE-S2`, or Stage 3 completion;
- claim production recovery, runtime closure, filesystem admission, capability
  improvement, or scientific eligibility.

The current user instruction authorizes creation, correction, and read-only
review of this draft as one bounded work item. It does not create standing
authority for later document revisions. Static implementation, CPU tests, and
H200 full-model execution are separate phases with separate authority
decisions below. A later phase never inherits authority merely because an
earlier phase passed.

## 1. Naming boundary

Two unrelated uses of `L3`/`Stage 3` must remain distinct:

- **Stage2B L3** is the already implemented CPU/synthetic deterministic restart
  recovery sublayer in
  `docs/cach_sana_wam/stage2b_l3/`. It reconstructs an exact caller-reopened
  Stage2B ledger in one process.
- **Global Stage 3** is the project-wide, complete-model, real-shape,
  real-dtype, update-free causal proxy defined in
  `docs/CAUSAL_ACTION_CONDITIONED_HYBRID_SANA_WAM_DEVELOPMENT_PLAN_20260731.md`.

Passing Stage2B L3 does not authorize or satisfy global Stage 3.

The terms used below are:

- `S1-contract subset`: the completed pure-contract/lightweight Stage-1 work;
- `S2-mini subset`: the completed random-mini-model/synthetic Stage-2 work;
- `S2B ledger subset`: Stage2B L0/L1/L2A/L2B/L3 source and CPU tests;
- `global G3`: the future full-model update-free execution;
- `formal`: an experiment whose authority, root, inputs, endpoint, and evidence
  are registered for a scientific claim. Global G3 remains non-scientific.

## 2. Fixed predecessor trust anchors

This draft is based on the following bytes. Any mismatch creates a new plan
revision; it is not an in-place retry.

### 2.1 Repository and governing plan

| Object | SHA256 / Git object |
|---|---|
| `sana-wam` HEAD | `605f1c134b4c983ff80f8489c4bc8847036329e2` |
| `third_party/Sana` HEAD | `16b9cec673e3335724ba2d8db25de7f9ed229292` |
| `sana-afcc-handoff` HEAD | `9586486f2a9f5172d57b325e32093a3e018d34c0` |
| governing development plan | `969d9668ffed07176844fcbd84062825f9d5f0af0325959b0068c38fe481f0dd` |

The three Git objects are lineage facts, not proof that origin contains them.
No remote fetch, checkout, reset, or submodule rewrite is authorized here.

### 2.2 Stage-2 and Stage2B source chain

| Layer | Manifest | Verifier | Immutable source bundle |
|---|---|---|---|
| Stage 2 | `8333c74d9bc8115937f98d9fc12a60da62cab49dd8895cd403047027315052fb` | `1a2961770b3004e9b481a68fd92b724e2505c31253a170d8c40a8d5582bcf6ea` | `71e73b3f2b1efb125f0b57f8c47913bab47d492920310f0a53848fe7903c229c`, 327680 bytes |
| Stage2B L0/L1 | `87a5f36d13e06cdd5d951cf1dd96f4e353edc50fc555c96b3269d19859126c0a` | `a9a2a58c0dbc831f886e43a0c299a52a59683e443f05dcd646dc99a41ff5a1fe` | `0feb797618cb9e02150abd52b357c03d2a4986177fc1e8ebeca626ee808389d2`, 389120 bytes |
| Stage2B L2A | `5d27a778bff120ff7a27c18ebaffbfb1761b2ced3d5f2e4c46a4de8c5763b8bf` | `cd433d338ffb9f493f291905f0d5313c64591eafbf729bd7ae16c96fac1c9255` | `163737a9b8c9b88bfd8d7e05064052fc2948661f270c4a9a9f4ea55e96494b7d`, 276480 bytes |
| Stage2B L2B | `5e577cc857de232529ea8737321643ba07a9366d2fba0c519c322da722b20487` | `6fb42de438057a1fb6fe9900872d1ba3158a009aaa00126123d3dd66a15f8ce3` | `f2380eabfe06daeb7087dc139cd67441e2164b27f91bc71ff6295b7fc8e43533`, 194560 bytes |
| Stage2B L3 | `3c494faab4428ec8b7ba705b61c8bdd7bfe6f9f623366610ca1e9e94179f7b93` | `b058c46b01c1041677dd84a14bfc974593909daad5023b5b8bb4e432b643812f` | `1a2f984b179d6f0a719873e4eda7344b99b8093c75f4ad8528f63a373ca784c3`, 184320 bytes |

The manifest and verifier paths in the same order are:

| Layer | Manifest path | Verifier path |
|---|---|---|
| Stage 2 | `docs/cach_sana_wam/stage2/SOURCE_MANIFEST.json` | `scripts/verify_cach_stage2.py` |
| Stage2B L0/L1 | `docs/cach_sana_wam/stage2b/SOURCE_MANIFEST.json` | `scripts/verify_cach_stage2b.py` |
| Stage2B L2A | `docs/cach_sana_wam/stage2b_l2/SOURCE_MANIFEST.json` | `scripts/verify_cach_stage2b_l2a.py` |
| Stage2B L2B | `docs/cach_sana_wam/stage2b_l2b/SOURCE_MANIFEST.json` | `scripts/verify_cach_stage2b_l2b.py` |
| Stage2B L3 | `docs/cach_sana_wam/stage2b_l3/SOURCE_MANIFEST.json` | `scripts/verify_cach_stage2b_l3.py` |

Every bundle below must satisfy the verifiable predicate: regular file,
`st_nlink=1`, outer mode `0444`, and no symlink traversal:

| Layer | Exact bundle path |
|---|---|
| Stage 2 | `/DATA/share/sana_cach_source_bundles/cach_stage2_71e73b3f2b1efb125f0b57f8c47913bab47d492920310f0a53848fe7903c229c_20260731.tar` |
| Stage2B L0/L1 | `/DATA/share/sana_cach_source_bundles/cach_stage2b_0feb797618cb9e02150abd52b357c03d2a4986177fc1e8ebeca626ee808389d2_20260731.tar` |
| Stage2B L2A | `/DATA/share/sana_cach_source_bundles/cach_stage2b_l2a_163737a9b8c9b88bfd8d7e05064052fc2948661f270c4a9a9f4ea55e96494b7d_20260731.tar` |
| Stage2B L2B | `/DATA/share/sana_cach_source_bundles/cach_stage2b_l2b_f2380eabfe06daeb7087dc139cd67441e2164b27f91bc71ff6295b7fc8e43533_20260731.tar` |
| Stage2B L3 | `/DATA/share/sana_cach_source_bundles/cach_stage2b_l3_1a2f984b179d6f0a719873e4eda7344b99b8093c75f4ad8528f63a373ca784c3_20260731.tar` |

The Stage2B L3 overlay independently resolves 126 source paths. Its 119-path
predecessor inventory digest is
`ab00ec514081344e496564fd0bcbb6fc046cf12a9b19e06d9f5484c272cf67e5`.
The L3 delta has zero predecessor collisions.

All source bundles above are byte checkpoints only. Their manifests explicitly
leave `transitive_runtime_closure=false`.

## 3. Current global gate state

No global stage gate is currently claimed as passed.

| Gate | Green evidence already available | Current formal state | Blocking class |
|---|---|---|---|
| `GATE-S0` | governance, layout/bootstrap, cache/commit, source/launcher and data/scale designs; additive source bundles | `NOT_CLAIMED` | draft authority, incomplete executable/runtime closure, no external trust anchor |
| `GATE-S1` | pure contracts, typed layout/cache, action seam, guards, config and lightweight tests | `NOT_CLAIMED` | no caller-pinned global reconciliation or production-shaped mini closure; executable authority absent |
| `GATE-S2` | random mini GDN/ActionDiT, synthetic tensors, future separation, parameter-gradient wiring, paired commit, failure receipt and standalone ACK subset | `BLOCKED_NOT_CLAIMED` | production-path C0--C8, complete architecture, named init and runtime closure absent |
| Stage2B ledger subset | committed history, L0 receipt store, snapshot codec, append-only ledger, single-process manager and deterministic recovery | `SOURCE_CHECKPOINT_VERIFIED_NOT_ADMITTED` | production filesystem, multi-process fencing, CUDA, real power loss and runtime closure absent |
| global Stage 3 | no authorized execution | `NOT_AUTHORIZED` | all prerequisites and a separate user authority below |

Historical pass counts remain scoped evidence only. They must not be summed or
renamed into a global gate result.

## 4. Evidence and gap matrix for C0--C8

The governing meanings are fixed by the main plan:

| Contract | Existing scoped evidence | Required pre-G3 closure | Required only inside G3 |
|---|---|---|---|
| `C0 causal visibility` | mini future-video/action perturbation | production-path mini/static mask and dispatch proof, including reverse-mode gradient invariance | full-model output, reverse-mode parameter/input-gradient and registered JVP perturbation |
| `C1 bypass identity` | zero-init adapter and disabled bypass tests | exact reference/candidate module inventory and identity path | full-model shared-input identity check |
| `C2 train/deploy equivalence` | partial synthetic continuation checks | exact production chunk operator/cache harness | full-sequence versus chunk-cache equivalence |
| `C3 content-time identity` | `ChunkActionLayout`, cache metadata and codec tests | one production layout instance bound to every video/action/cache/RoPE field | full-model emitted layout/cache digest |
| `C4 bootstrap semantics` | `first_frame_pinned`, zero observed-prefix contracts | production builder rejects every legacy drift field | real-shape bootstrap execution |
| `C5 source closure` | layered byte bundles | executable Python/native/dynamic-import/runtime closure | runtime receipt must match the closure |
| `C6 action coverage` | bootstrap/continuation/tail synthetic cases | production-path exact-once span accounting | full-model layout receipt |
| `C7 commit atomicity` | synthetic durable protocol and recovery | production-shaped in-memory paired-state interface; denoise never writes | exactly one ephemeral cache/history advance per completed video/action pair; no durable/controller deploy commit |
| `C8 hybrid cache` | GDN/cache semantic unit coverage | exact 20-layer table and dispatch closure | full-model cache growth/equivalence/resource checks |

## 5. Gate-ownership reconciliation proposal

This section is a proposal requiring independent review. It does not change a
frozen predecessor document by itself.

The current records contain a cycle: the global plan assigns complete-model
gradient/JVP evidence to Stage 3, while the historical Stage-2 report lists
full-model gradient/JVP evidence among blockers for “full Gate-S2”. Requiring
that evidence before requesting Stage 3 would require Stage 3 to precede
itself.

The proposed unique ownership is:

1. `S1-contract subset` owns pure schema, selector, mask, layout, bypass,
   failure and fail-closed surface contracts.
2. `S2-mini subset` owns numerical forward/backward/JVP on mini random models
   and synthetic tensors, including production-path mini wiring.
3. global `G3` exclusively owns complete 2B real-shape/real-dtype, update-free
   forward/backward/JVP evidence.
4. full-model G3 evidence is therefore not a prerequisite to request G3. The
   corresponding pre-G3 blocker is replaced by production-path mini/static
   wiring plus a reviewed G3 test design.
5. Until a new caller-pinned closure artifact approves this mapping, global
   `GATE-S1` and `GATE-S2` remain not claimed.
6. Phase D entry is nevertheless strict: `GATE-S0=PASS` and `GATE-S1=PASS`.
   `GATE-S2` must either be `PASS`, or have the sole exact state
   `BLOCKED_ONLY_BY_G3_OWNED_FULL_MODEL_UPDATE_FREE_EVIDENCE`. In the latter
   state, every non-G3-owned S2 blocker must be zero and enumerated as closed.
   An arbitrary or honestly reported broader `BLOCKED` state cannot enter
   Phase D.

The closure artifact must preserve every historical failure and limitation; it
may supersede terminology, never overwrite evidence.

## 6. First global Stage-3 pair

The only proposed pair is:

- reference: `REF-GDN-CORRECTED`;
- candidate: `CACH-A`;
- purpose: action-conditioned dynamics wiring, not training capability;
- unique delta: the candidate video path reads the registered action
  conditioning seam; the reference uses the exact identity bypass;
- shared architecture: corrected 20-layer GDN video path, action path, cache,
  layout, initialization policy and synthetic inputs;
- enabled candidate-only components: zero-init action-to-video adapter only;
- disabled for both: anchors, AttnRes, self-forcing, AFCC, Phase-6 reference,
  DAgger/recovery policy, rerank, best-of-N and temporal ensemble.

This is a capability-lineage pair, not an AFCC formal pair. Neither arm may
instantiate AFCC `F`. If and only if a shared input schema syntactically
requires an `F` field, the normalization boundary must validate it as `F=0`
and remove it before any `Trainer` or model factory is constructed. It must
also delete every AFCC reference and action-reference field. The final config,
source reachability and model graph must contain no AFCC module, parameter,
buffer, reference, or dormant feature state. An AFCC formal experiment would
instead require native architectural `F=1` treatment and pre-construction
`F=0` control; that different question is outside this plan.

## 7. Initialization, checkpoint and input policy

Global G3 is update-free and prospective:

- complete video/action trainable graph is freshly random-initialized;
- all same-name, same-shape shared tensors have byte-identical initial values;
- operator-specific tensors use separately registered seeds and initializers;
- the candidate adapter is exact zero-init where required by C1;
- no pretrained video DiT is loaded;
- no resume, model checkpoint, optimizer state, VAE latent cache, dataset row,
  RoboTwin episode, HDF5 file or training input is read;
- deterministic synthetic tensors have real production shapes and registered
  value-generation seeds;
- no optimizer is constructed and no parameter is updated;
- no candidate checkpoint or “best” artifact is saved.

The first G3 revision is fixed to registered latent and text-embedding input
boundaries. VAE and text-encoder modules are not imported and their weights are
not read. Importing or executing either encoder is not a switch within this
revision; it requires a new plan revision, new source/runtime/input closure and
new authority.

## 8. Complete 20-layer graph closure

Before G3 execution authority can be requested, a machine-readable graph must
fix:

- all 20 block indices and whether each block uses GDN or softmax;
- the corrected reference table and candidate table;
- exact module class, constructor arguments, tensor names, shapes and dtypes;
- cache state fields and content-time interpretation for every layer;
- action adapter insertion point and identity-bypass path;
- Action RoPE cursor source and `ChunkActionLayout` instance digest;
- zero disabled anchors, AttnRes routers and self-forcing modules;
- exact forward/full-sequence and deploy/chunk dispatch entrypoints;
- parameter aliasing, buffer inventory and shared/candidate-only key allowlists;
- proof that no AFCC/reference parameter or source field is reachable.

The graph must be generated before reading G3 results and pinned by the pair
spec, authority and verifier.

## 9. Named initialization and inventories

The following must be produced without executing G3:

1. reference parameter/buffer inventory;
2. candidate parameter/buffer inventory;
3. exact shared-key intersection and shape/dtype agreement;
4. candidate-only zero-init adapter allowlist;
5. reference-only allowlist, expected to be empty unless justified;
6. frozen/trainable classification, even though G3 performs no update;
7. shared initialization seed and per-tensor digest algorithm;
8. operator-specific seeds and initializer names;
9. rejection rules for missing, unexpected, aliased or non-finite tensors;
10. a checkpoint-absence attestation.

Named-init parity is not satisfied by using the same global seed alone.

## 10. Source and runtime closure

The existing source chain is necessary but insufficient. A G3 closure must
pin at least:

- all repository and vendor source files reachable by the two entrypoints;
- Git recovery material or another independently reconstructible byte source;
- exact Python interpreter and environment lock;
- PyTorch, CUDA toolkit/runtime, Triton and imported Python distributions;
- GPU driver and loaded native shared libraries;
- dynamic-import results, generated kernels and environment-controlled paths;
- exact pair spec, graph, config, synthetic input recipe and thresholds;
- launcher, verifier, evaluator and evidence schemas.

The closure must distinguish “pinned worktree prerequisite” from
“independently reconstructible runtime”. Missing closure remains a hard block;
it cannot be waived by a successful forward pass.

For the first G3 revision, runtime Python/native code generation is disabled,
including compile-on-first-use paths. The exact eager/reference operator path
and its disable-compile environment must be pinned. If this is infeasible, work
stops: a new plan revision must define a separately authorized precompile
phase, freeze the generated artifacts for the exact GPU/toolchain, and rerun
the runtime verifier before G3. G3 may never discover and then bless new JIT
bytes inside the result-bearing root.

## 11. Production-path mini C0--C8 closure

Before full-model authority, an additive implementation may provide a
production-builder-shaped mini harness. It must use the same public dispatch,
layout, cache and adapter interfaces as G3 while replacing only tensor sizes.

Required checks include:

- future clean/current-target/future-chunk perturbation separation;
- exact invariance of earlier deterministic outputs and reverse-mode gradients
  under those perturbations, plus any separately registered JVP invariant;
- exact reference bypass and candidate zero-init identity;
- parameter-gradient/JVP reaches the zero-init adapter parameters and is finite;
- full versus chunk-cache equivalence on CPU reference;
- no-denoise-write behavior and exactly one ephemeral paired cache/history
  advance after each completed video/action pair;
- cache growth, reset and stale-view behavior across those ephemeral commits;
- exact bootstrap, continuation, partial-tail and action exact-once coverage;
- shared named-init digest and candidate-only inventory;
- no AFCC/anchor/AttnRes/self-forcing reachability;
- failure receipt and immutable temporary-root behavior;
- zero skipped hard checks.

These tests remain non-scientific and do not themselves authorize G3.

## 12. Filesystem and recovery decision

Before an authority is drafted, one of two policies must be selected and
reviewed:

### Policy A: admitted reusable recovery root

Requires parent/ancestor ownership, realpath and mount policy, no symlink or
hard-link ambiguity, multi-process writer fencing, old-writer death proof,
`fsync`/no-replace/rename semantics, crash/power-loss testing, CUDA snapshot
synchronization, and measured 20-layer snapshot resource bounds.

### Policy B: fresh single-writer G3 evidence root

The G3 process creates one unique non-scientific root and never resumes or
reopens it for model state. Stage2B recovery remains diagnostic and is not used
as authority. The root still needs ancestor/mount/no-replace/permission checks,
exclusive creation, failure freezing and evidence durability. This policy does
not claim production recovery admission.

The plan recommends Policy B for the first update-free G3 because G3 performs
no training or deploy commit. The recommendation is not active until the
authority explicitly selects it. Silently bypassing the decision is forbidden.

## 13. Stage-3 test matrix

Reference and candidate run serially from the registered named initialization
and the same synthetic inputs. At minimum, G3 must measure:

| Test | Required observation |
|---|---|
| graph/inventory | exact expected graph, keys, dtypes, shapes and disabled components |
| full causal | finite full-sequence output and registered digests |
| chunk causal | finite chunk/cache output and registered digests |
| equivalence | full/chunk residual within preregistered BF16/FP32 bounds |
| future perturbation | earlier output, reverse-mode parameter/input gradients and registered JVP unchanged within registered bounds |
| bypass identity | reference and zero-init candidate satisfy registered identity condition |
| wiring | candidate adapter parameter-gradient/JVP is finite and non-empty |
| cache/layout | layout instance and per-layer cache/content-time digests match; denoise writes zero state and each completed pair advances ephemeral state exactly once |
| immutability | no parameter/buffer changes outside explicitly allowed diagnostic state |
| resources | CUDA peak, host peak, wall time and optional I/O within registered limits |

If output sensitivity is structurally zero under a zero-init adapter, a
separate registered synthetic-nonzero diagnostic may be used only for wiring.
It must not replace the zero-init pair or become a capability result.

The only model-state mutation allowed inside G3 is the registered ephemeral
cache/history transition after a completed paired video/action chunk. Denoise
substeps must leave that state byte-identical, and every completed pair must
advance it exactly once. This transition is in-memory diagnostic state: it is
not a durable ledger publication, controller acknowledgement, deploy commit,
parameter update, or checkpoint save.

## 14. Threshold contract

All numeric values are deliberately unresolved in this draft. A dedicated
threshold artifact must, before any G3 output is observed, fix:

- FP32 and BF16 absolute/relative equivalence tolerances;
- exact-zero requirements for CPU reference checks;
- the deterministic scalar probe used for reverse-mode differentiation;
- the exact input tensors and parameter-key domain whose gradients are
  compared, including treatment of absent/`None` gradients;
- future-perturbation output, reverse-mode input/parameter-gradient and JVP
  tolerances;
- CPU exact-equality rules for every compared gradient tensor and digest;
- GPU BF16/FP32 absolute/relative per-tensor and registered norm tolerances for
  reverse-mode gradients, separately from output/JVP tolerances;
- minimum finite/non-empty gradient/JVP wiring conditions;
- per-layer cache residual and digest policy;
- permitted non-deterministic kernel set, preferably empty initially;
- CUDA and host memory maxima;
- wall-time and I/O maxima;
- failure-on-warning and first-failure stop behavior.

Post-result relaxation creates a new candidate revision and a new root. It may
not be called a retry.

## 15. Immutable evidence root

Any future G3 run requires a user-authorized unique root under confirmed
capacity. This root retains diagnostic admission evidence; it is not a
capability/scientific evaluation root. The authority must bind:

- random nonce and non-reusable root path;
- caller-pinned authority, pair spec, graph, source/runtime and threshold SHAs;
- exact argv, selected environment and GPU UUID;
- reference/candidate ordering and input digest;
- start/completion/failure receipts;
- stdout/stderr, structured metrics and resource telemetry digests;
- parameter-before/after digests;
- first failure and frozen partial evidence;
- final mode/ownership and no-overwrite checks.

Infrastructure failure uses a new root after the failed root is frozen. Failed,
partial, superseded and successful roots are never deleted or reused.

## 16. Five-piece execution contract

Before execution, five independently pinned roles are required:

1. `DESIGN`: this reviewed plan plus graph and threshold definitions;
2. `AUTHORITY`: exact allowed action, host, GPU, root, inputs and expiry;
3. `VERIFIER`: preflight and postflight byte/schema/gate verifier;
4. `EVALUATOR`: diagnostic update-free admission metrics only, never task,
   policy, capability, benchmark, or scientific evaluation;
5. `LAUNCHER`: imports and executes only after external trust anchors pass.

The launcher must fail before torch/model import, GPU reservation or root
creation when any caller pin, authority bit, path, environment or prerequisite
differs. The current Stage-0 refusal-only launcher cannot be relabeled as this
launcher.

The future authority schema must use distinct bits. At minimum,
`diagnostic_admission_evaluator_authorized` may become true only in Phase D,
while `capability_evaluation_authorized`, `scientific_evaluation_authorized`,
`training_authorized`, and `deployment_authorized` remain false.

## 17. Phased authorization

### Phase A: document review

Allowed in the current bounded work item: create/correct this draft, compute
text hashes, and perform read-only review. After delivery, any material plan
revision requires a new user instruction. No code/test execution is included.

Exit: independent reviewers report no open P0/P1 and the gate-ownership
proposal is explicitly accepted or revised.

### Phase B: static scaffold

Requires new user authorization. May add only schemas, default-deny authority,
pair/graph/threshold drafts, refusal-only launcher and review-only verifier.
It must not import torch or construct a model.

### Phase C: CPU lightweight closure

Requires explicit CPU-test authorization. May run production-path mini C0--C8
tests in `/tmp` only. It may not create a formal/admission root, use CUDA, load
data/checkpoints or execute the full model.

### Phase D: H200 global G3 execution

Phase D cannot be requested while `GATE-S0` or `GATE-S1` is not `PASS`, or
while `GATE-S2` has any blocker other than its exact G3-owned full-model
update-free evidence state defined in Section 5.

Requires a separate user authorization naming:

- complete 2B CUDA forward/backward/JVP;
- the exact reference/candidate pair and revision;
- serial GPU execution and GPU UUID/resource budget;
- creation and freezing of one unique non-scientific admission root;
- retained logs, receipts and resource telemetry.

Phase D permits only the specifically pinned diagnostic admission evaluator.
It still forbids optimizer steps, training, capability/scientific/benchmark or
policy evaluation, dataset/checkpoint access, checkpoint saving, deployment
and scientific claims.

## 18. Items intentionally deferred

For synthetic real-shape G3 inputs, the following are later blockers rather
than G3 prerequisites:

- RoboTwin timestamp/rate, training split, normalization and data-order closure;
- Stage-4 `DATA_AND_SCALE_DESIGN` execution budget;
- deploy applied-action ACK transport and controller commit;
- enabled anchors, AttnRes and self-forcing;
- checkpoint load/resume policy beyond explicit rejection.

They become mandatory before their corresponding Stage 4/5/deploy work. Their
deferral must not be misreported as completion.

## 19. Stop conditions

Work stops fail-closed on any of the following:

- a predecessor pin or canonical-host byte differs;
- any global gate is described as passed without a new closure artifact;
- global Stage 3 is confused with Stage2B L3;
- reference/candidate differ in more than the registered action seam;
- either capability arm contains AFCC/reference state;
- full-model evidence is again made a prerequisite to enter its own stage;
- graph, named-init, source/runtime or threshold closure is missing;
- a hard C0--C8 check is skipped, softened or made result-dependent;
- a root path is reused, pre-exists, or lacks exclusive creation/failure freeze;
- unrelated GPU work would be disturbed or resources are insufficient;
- a dataset/checkpoint/optimizer/training/capability-evaluation/deploy path is
  reached, or a diagnostic evaluator exceeds its pinned admission metrics;
- any P0/P1 remains open in the applicable phase review.

## 20. Admission checklist before requesting Phase D

- [ ] gate-ownership reconciliation approved;
- [ ] global `GATE-S0=PASS` and `GATE-S1=PASS` in a caller-pinned closure;
- [ ] global `GATE-S2=PASS`, or its sole exact state is
      `BLOCKED_ONLY_BY_G3_OWNED_FULL_MODEL_UPDATE_FREE_EVIDENCE` with zero
      other blockers;
- [ ] pair spec and unique delta caller-pinned;
- [ ] exact 20-layer graph and disabled components pinned;
- [ ] shared named-init and key inventories pinned;
- [ ] production-path mini C0--C8 has zero hard failures/skips;
- [ ] source/runtime/dynamic-import/native closure verified;
- [ ] filesystem Policy A or B explicitly selected and admitted for its scope;
- [ ] thresholds fixed before results;
- [ ] five-piece paths and SHAs closed;
- [ ] H200 GPU/RAM/disk/inode/process preflight current;
- [ ] unique immutable root path and failure policy registered;
- [ ] independent review has zero open P0/P1;
- [ ] user separately authorizes Phase D exact actions.

## 21. Immediate next action

The next action is independent review of this design-only draft. If it passes,
request authorization for **Phase B static scaffold only**. Do not combine that
request with CPU tests or Phase D execution.

No global Stage-3 model execution should begin from this document alone.
