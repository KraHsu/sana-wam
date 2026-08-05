# CACH Stage 2B L2A additive source checkpoint

This directory describes a seven-file, zero-collision L2A source delta over
the fixed Stage 2B L0/L1 artifact. It authenticates only source bytes and the
reported live H200 CPU/synthetic checks. It is not retained execution evidence,
a formal ledger root, filesystem admission, recovery authority, L2B/L3
completion, training, evaluation, deployment, capture, or a scientific result.

The outer manifest is deliberately excluded from the source inventory and tar
to avoid self-reference:

```text
docs/cach_sana_wam/stage2b_l2/SOURCE_MANIFEST.json
```

The exact included inventory is
`STAGE2B_L2A_SOURCE_FILES.txt`. None of its seven paths collides with the 105
resolved predecessor paths authenticated by the fixed Stage 2B manifest and
verifier.

## Verification

Verification requires the manifest SHA256 as an out-of-band trust anchor and
must run in the canonical H200 worktree:

```bash
.venv/bin/python scripts/verify_cach_stage2b_l2a.py \
  --manifest docs/cach_sana_wam/stage2b_l2/SOURCE_MANIFEST.json \
  --expected-manifest-sha256 <64-lowercase-hex-digest>
```

The verifier authenticates the predecessor Stage 2B manifest, verifier, and
deterministic source bundle; reruns the fixed predecessor byte verifier without
replaying tests; reconstructs the resolved 105-path predecessor inventory;
checks the seven source files and their zero-collision overlay; validates the
live lightweight report and direct inherited pins; and byte-compares the
deterministic POSIX USTAR archive.

The source tar is an uncompressed, digest-named archive under
`/DATA/share/sana_cach_source_bundles`. Members are regular files in source-list
order with mode `0644`, uid/gid `0`, empty owner/group names, no links or PAX
headers, and fixed mtime `1785456000` (`2026-07-31T00:00:00Z`). The outer tar is
one regular link with mode `0444`.

## Boundary

The report records mutable live test results from the moment before this source
manifest and tar existed. No log, JUnit XML, SHA256SUMS root, dataset,
checkpoint, model weight, optimizer, GPU execution, formal `/DATA` ledger root,
or restart replay is retained by this checkpoint. Runtime and dynamic-import
closure remain false. Multi-process fencing, parent/mount checks, real
filesystem power-loss behavior, L2B manager coordination, and L3 state install
remain blocked.
