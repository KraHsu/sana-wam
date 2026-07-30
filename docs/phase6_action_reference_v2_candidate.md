# ActionReference v2 candidate

Local-only candidate for a Phase1 T0/E0/A0 paired action-error table, fixed-row
live verification, common-input tracing, and the torch-free Phase-6 preflight
state machine. No GPU forward, checkpoint load, training, or remote mutation was
performed while preparing this directory.

## State machine

The preflight purposes are strictly separated:

1. `reference_precompute` authorizes the 504-row forward-only build and a later
   fresh-process live re-forward at steps 1, 253, and 504. Its report is written
   once with no-overwrite semantics. The spot process stable-reads the same
   request/report by SHA and reproduces validation without rewriting the report.
2. `smoke` consumes the real table, live-spot artifact, reference authorization
   records, full five-arm projection manifest, source/runtime closure, and exact
   real-2B smoke authorization. It cannot create an optimizer or start formal
   training.
3. `training` additionally consumes the actual smoke request/report and strict
   `real_2b_smoke_gate`. It binds one canonical arm projection sidecar and the
   exact run identity. Trainer stable-reads an already issued report by SHA; it
   never attempts to overwrite it.

Registered reports, source manifests, and runtime-support manifests use an
exclusive hard-link publish. Existing destinations fail closed.

## Precompute config

`configs/phase1_action_reference_precompute_overlay.yaml` is an overlay, not a
standalone model config. Merge it into the frozen materialized `T0_E0A0` config,
resolve the result, and remove every action-reference input field. The resulting
file must retain the common Phase-6 dataset/plan paths and set the Phase1
checkpoint as `init_checkpoint`.

After all source changes are frozen:

1. Generate the v2 source manifest, including `src`, `scripts`,
   `third_party/Sana`, `pyproject.toml`, `uv.lock`, `.python-version`, Git
   provenance, Python ABI, and required distribution versions.
2. Put the full source-manifest SHA in
   `training.phase6_code_source_manifest_sha256`. Do not use the dataset-only
   preprocessing digest `6bdc3bcc...`.
3. Compute the final precompute YAML SHA and pin that exact file as the
   `reference_precompute` request `input_config`.
4. Generate the runtime-support manifest for the actual SANA/Gemma config files,
   then publish the request and its report exclusively before the first GPU
   query.
5. Run `build`, record the table/build-manifest SHA values, then run `spot` in a
   fresh process using `--preflight-report-sha256`. Spot never issues a second
   authorization report.

Both subcommands reject an existing output, a path outside the authorized output
root, the wrong world size, a config/report/source mismatch, T1/E1/A1, caches,
workers, training/autograd/checkpointing, or noncanonical provenance before the
corresponding expensive operation.

## Trace contract

- `phase6_common_input_trace_required` is enabled for every formal arm. It hashes
  logical raw tensor bits plus dtype/shape/stride/layout/device metadata. A1
  compares this trace with its paired Phase1 table row before returning loss.
- `phase6_t0_reference_forward_trace_required` is enabled only by Phase1 build
  and spot. It requires T0/E0/A0 and eval mode. Formal training never copies or
  hashes model predictions for this trace.
- Post-proprio context is prepared once and reused by the actual forward, so
  tracing cannot consume a second dropout/noise draw or hash a context different
  from the model input.

The common trace still copies several prepared video tensors to CPU for hashing.
Its exact cost has intentionally not been GPU-benchmarked in this candidate.
Before formal launch, record per-step copied bytes and wall time in the real-2B
smoke gate; treat excessive trace overhead as a launch blocker rather than
silently weakening the equality contract.

## Integration ownership

The run-integrity/metrics component is the sole owner of a formal run directory.
When merging, remove the candidate Trainer's provisional
`_create_phase6_output_directory`/thin integrity-manifest writer and let the
run-integrity writer create the directory, snapshot the training preflight
report, record all 504 common traces, and publish completion artifacts.

The arm materializer, smoke runner, launcher ticket, and run-integrity writer are
separate candidates. Their torch-free validators are imported lazily by
`phase6_preflight.py`; all components must be merged before a formal launch.
