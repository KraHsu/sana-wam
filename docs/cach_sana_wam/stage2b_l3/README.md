# CACH Stage 2B L3 additive source checkpoint

This directory describes a seven-file, zero-collision L3 source delta over
the fixed Stage 2B L2B source checkpoint. It authenticates only source bytes
and the reported live H200 CPU/synthetic checks. It is not retained execution
evidence, a formal ledger root, production recovery admission, filesystem or
runtime admission, training, evaluation, deployment, capture, or a scientific
result.

The supported L3 scope is deliberately narrow: deterministic reconstruction
from one caller-reopened, exact L2A ledger after the caller guarantees that
the old writer has exited. It closes at most one pending PREPARED intent with
an exact `PROCESS_EXIT/AFTER_PREPARED` terminal, accepts an exceptional
publication only after exact-SHA reconciliation, reconstructs current state,
layout, strict typed receipts, and both completed-map layers without executing
a model, and reports inert content-addressed debris only after closure. It
performs a final chain and content-object rescan before returning the private
state-installed CPU/synthetic facade.

The outer manifest is deliberately excluded from the source inventory and tar
to avoid self-reference:

```text
docs/cach_sana_wam/stage2b_l3/SOURCE_MANIFEST.json
```

The exact included inventory is `STAGE2B_L3_SOURCE_FILES.txt`. Its seven paths
must not collide with the 119 resolved predecessor paths authenticated by the
fixed L2B manifest, verifier, and deterministic source bundle.

## Verification

Verification requires the manifest SHA256 as an out-of-band trust anchor and
must run in the canonical H200 worktree:

```bash
.venv/bin/python scripts/verify_cach_stage2b_l3.py \
  --manifest docs/cach_sana_wam/stage2b_l3/SOURCE_MANIFEST.json \
  --expected-manifest-sha256 <64-lowercase-hex-digest>
```

The verifier must authenticate the fixed L2B predecessor manifest, verifier,
and deterministic source bundle; rerun its byte verifier without replaying
tests; reconstruct the resolved 119-path predecessor inventory; check the
seven L3 files and their zero-collision overlay; validate the lightweight
report and direct pins; and byte-compare the deterministic POSIX USTAR archive.

The source tar is an uncompressed, digest-named archive under
`/DATA/share/sana_cach_source_bundles`. Members are regular files in
source-list order with mode `0644`, uid/gid `0`, empty owner/group names, no
links or PAX headers, and a fixed checkpoint mtime. The outer tar must be one
regular link with mode `0444`.

## Recorded checks

The live H200 checks recorded in
`STAGE2B_L3_LIGHTWEIGHT_TEST_REPORT.json` are:

- `py_compile` and Ruff over the L3 recovery module and focused test file;
- 31 focused CPU/synthetic deterministic-recovery tests;
- the fixed L2A and L2B focused regressions: 106 passed;
- the complete lightweight CACH CPU suite: 452 passed, 1 known predecessor
  deselection, and 13 existing warnings;
- independent final code, test, and plan audits with zero open P0 and P1
  findings.

These results were observed before the source manifest and deterministic tar
existed. They are not retained immutable execution evidence.

## Boundary

No log, JUnit XML, SHA256SUMS execution root, dataset, checkpoint, model
weight, optimizer, GPU execution, formal `/DATA` ledger root, or real
power-loss replay is retained by this checkpoint. Proof that the old writer
is dead, multi-process writer fencing, hostile fd-bound mutation resistance,
CUDA/device mapping, runtime and dynamic-import closure, and admitted real
`/DATA` fsync/no-replace/remount behavior remain unestablished. Every
production recovery, filesystem, runtime, training, evaluation, deployment,
capture, and scientific authority remains false.
