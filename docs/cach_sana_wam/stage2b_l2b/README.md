# CACH Stage 2B L2B additive source checkpoint

This directory describes a seven-file, zero-collision L2B source delta over
the fixed Stage 2B L0/L1 and L2A source checkpoints. It authenticates only
source bytes and the reported live H200 CPU/synthetic checks. It is not
retained execution evidence, a formal ledger root, filesystem admission,
restart recovery, L3 completion, training, evaluation, deployment, capture,
or a scientific result.

The supported L2B scope is deliberately narrow: a fresh-only,
single-process, CPU/synthetic ledgered-manager facade. It preserves PREPARED
before staging, terminal-before-CAS publication, exact same-process retry,
sticky poison, and fail-closed reset coordination. It does not reopen a
non-genesis root or install recovered private manager state.

The outer manifest is deliberately excluded from the source inventory and tar
to avoid self-reference:

```text
docs/cach_sana_wam/stage2b_l2b/SOURCE_MANIFEST.json
```

The exact included inventory is `STAGE2B_L2B_SOURCE_FILES.txt`. Its seven
paths must not collide with the 112 resolved predecessor paths authenticated
by the fixed Stage 2B L0/L1 and L2A manifests, verifiers, and deterministic
source bundles.

## Verification

Verification requires the manifest SHA256 as an out-of-band trust anchor and
must run in the canonical H200 worktree:

```bash
.venv/bin/python scripts/verify_cach_stage2b_l2b.py \
  --manifest docs/cach_sana_wam/stage2b_l2b/SOURCE_MANIFEST.json \
  --expected-manifest-sha256 <64-lowercase-hex-digest>
```

The verifier must authenticate the fixed Stage 2B L0/L1 and L2A predecessor
manifests, verifiers, and deterministic source bundles; rerun their byte
verifiers without replaying tests; reconstruct the resolved 112-path
predecessor inventory; check the seven L2B files and their zero-collision
overlay; validate the lightweight report and direct inherited pins; and
byte-compare the deterministic POSIX USTAR archive.

The source tar is an uncompressed, digest-named archive under
`/DATA/share/sana_cach_source_bundles`. Members are regular files in
source-list order with mode `0644`, uid/gid `0`, empty owner/group names, no
links or PAX headers, and a fixed checkpoint mtime. The outer tar must be one
regular link with mode `0444`.

## Recorded checks

The live H200 checks recorded in
`STAGE2B_L2B_LIGHTWEIGHT_TEST_REPORT.json` are:

- `py_compile` and Ruff over the L2B module and focused test file;
- 24 focused CPU/synthetic tests;
- the complete lightweight CACH CPU suite: 421 passed, 1 known predecessor
  deselection, and 13 existing warnings;
- three independent final read-only audits with zero open P0 and P1 findings.

These results were observed before the source manifest and deterministic tar
existed. They are not retained immutable execution evidence.

## Boundary

No log, JUnit XML, SHA256SUMS execution root, dataset, checkpoint, model
weight, optimizer, GPU execution, formal `/DATA` ledger root, or restart
replay is retained by this checkpoint. Runtime and dynamic-import closure,
multi-process writer fencing, parent/mount checks, real filesystem power-loss
behavior, restart recovery, and private manager state installation remain
unestablished. Every training, evaluation, deployment, capture, filesystem,
recovery, and scientific authority remains false.
