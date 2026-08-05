# CACH Stage 2B additive source artifact

Stage 2B is an additive, synthetic-CPU-only integration slice over the fixed
Stage 0, Stage 1, and Stage 2 source archives.  It does not revise an older
archive, manifest, or evidence root, and it does not grant Gate-S2, Stage 3,
training, evaluation, deployment, capture, or scientific authority.

The exact Stage 2B delta is the sorted 17-path inventory in
`STAGE2B_SOURCE_FILES.txt`.  The outer manifest is deliberately not in that
inventory, because including it would create a self-reference:

```text
docs/cach_sana_wam/stage2b/SOURCE_MANIFEST.json
```

At the time recorded by the lightweight report, no Stage 2B source manifest,
tar archive, or `/DATA` root had been created.  The outer manifest and
digest-named source tar were created later; their identity is determined only
by the caller-pinned outer manifest and verifier.  There is no retained
immutable Stage 2B execution evidence: the report records live H200 reruns
only and has no retained log, JUnit XML, or SHA256SUMS trust anchor.

## Verification contract

The scoped additive artifact is checked with an independently supplied
manifest digest:

```bash
python scripts/verify_cach_stage2b.py \
  --manifest docs/cach_sana_wam/stage2b/SOURCE_MANIFEST.json \
  --expected-manifest-sha256 <64-lowercase-hex-digest>
```

The expected manifest digest is mandatory.  The verifier authenticates the
17 source bytes, the source-list ordering, the exact lightweight report, all
Stage 0/1/2 archive and manifest pins, the declared overlay collision table,
the test harness and inherited import pins, and a deterministic regular-file
Stage 2B tar.  It never runs tests or replays the reported execution.

The overlay order is fixed:

```text
Stage 0 -> Stage 1 -> Stage 2 -> Stage 2B
```

Stage 2B must have no path collision with any predecessor.  Historical
predecessor collisions are explicitly allowlisted by the verifier; later
members win only for those paths.

The manifest schema is `cach.stage2b.source_manifest.v1`.  Its exact top-level
members are:

```text
base_lineage
canonical_host
canonical_worktree
created_at
execution_harness_pins
files
git_recovery_bundles
inherited_source_pins
overlay
retained_stage2b_evidence
schema
scope
stage2b_source_bundle
status
```

`git_recovery_bundles` contains the three keys `sana_wam`,
`sana_submodule`, and `local_afcc_handoff`.  They are `null` in this v1
checkpoint because no corresponding `.bundle` exists on canonical H200.
Their absence does not invalidate the byte-authenticated additive delta, but
it prevents a transitive or independently reconstructible runtime claim.

The Stage 2B tar must be an uncompressed deterministic POSIX USTAR archive named
`cach_stage2b_<sha256>_20260731.tar` under
`/DATA/share/sana_cach_source_bundles`, with outer mode `0444`.  It contains
exactly the 17 regular members in source-list order, normalized to mode
`0644`, uid/gid `0`, empty uname/gname/linkname, no PAX headers, and mtime
`1785456000` (`2026-07-31T00:00:00Z`).  Snapshot raw descriptors bind
`byte_order=little`; a big-endian descriptor is rejected fail closed, and the
focused scope covers canonical bool round trips plus BF16 edge bit patterns.

## Scope boundary

The four `tests/test_cach_stage2b_*.py` files are the focused Stage 2B suite.
The wider `tests/test_cach_*.py` run is diagnostic regression evidence: it
depends on predecessor source and includes one known Stage-1-only deselection.
Neither result is retained immutable evidence.

`transitive_runtime_closure` remains exactly `false`.  The delta archives do
not contain the three handoff git bundles, a Sana vendor recovery archive, a
wheelhouse/container/native-library closure, or a complete dynamic-import
inventory.  `pyproject.toml`, `uv.lock`, `tests/conftest.py`, package
initializers, selected inherited source, and selected vendor files are pinned
as worktree prerequisites, not claimed as a full runtime closure.
