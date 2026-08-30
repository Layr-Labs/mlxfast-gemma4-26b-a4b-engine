# Gemma 4 Single-Prompt Exact MTP Verifier and Performance Chart Design

Date: 2026-08-30  
Status: approved after B1 applicability audit
Worktree: `mlxfast-gemma4-mtp-depth3-tip`  
Baseline revision: `dfb5d48257de703e235f54f8cb6bf45b914fe6df`

## Problem and premise

Gemma 4 26B A4B single-prompt generation is currently faster without MTP. The
unchanged autoregressive control measures about 96.8 decode tok/s, while the
first real fixed-depth-1 MTP smoke measures 73.3831 decode tok/s despite 90.91%
draft acceptance and 1.882 committed tokens per round. The same smoke measures
5,135.6874 prefill tok/s for the 1,024-token Python prompt.

The acceptance result makes draft quality unlikely to be the primary cause of
this measured regression. The current
`.serialTarget` verifier performs one target forward per candidate token, so
verification repeats the expensive target path and erases the benefit of
committing multiple tokens per round. A stock rectangular target forward is not
an acceptable shortcut: on the existing hidden-prompt seal it changes tokens on
7 of 8 cases because the quantized matrix reductions for `[B, 1]` and `[B, L]`
are not numerically equivalent.

An exact fixed-width verifier implementation already exists as three clean
commits based on the same upstream tip: `3128eb5`, `5eea619`, and `764e9eb`.
Those commits are useful arithmetic scaffolding, but a plan-time applicability
audit found that every verifier kernel test uses `[8, C, ...]` tensors and the
model route is explicitly gated by `tokens.dim(0) == 8`. The authoritative
single-prompt receipt says `batch_size: 1`, and CBv2 does not pad that request to
eight rows. The clean verifier therefore cannot be enabled unchanged: it would
never engage for the requested workload.

The corrected design extends the clean fixed-width machinery with a physical
B1/C2-C4 route whose projection reductions are certified against independent
B1/L1 calls. The dirty verifier worktree remains evidence only and is not a
source of code.

## Approaches considered

### A. Exact B1/C2-C4 weight-sharing verifier — selected

Generalize the clean fixed-width projection kernels so one weight traversal
serves two through four speculative positions while each position retains the
ordinary B1/L1 reduction order. Serialize attention positions exactly as the
current rectangular cache contract requires. This directly attacks repeated
weight reads and is the only approach with a credible multi-token throughput
gain while preserving token identity.

### B. Lazy serial target graph

Keep independent B1/L1 target calls but defer their evaluation into one graph.
This has a smaller patch, but an earlier experiment showed that mutable KV can
observe later graph state unless every column is evaluated immediately. Fixing
that ownership hazard restores much of the synchronization cost, and the model
still rereads weights for every column. It remains a diagnostic, not the chosen
production path.

### C. Stock rectangular forward

Evaluate `[1, C]` with stock kernels. This is simplest and likely fast, but it
already changes output tokens on 7 of 8 hidden prompts. It is rejected because
the user did not authorize approximate verification.

## Scope

This project will:

1. Integrate the three clean exact-verifier commits onto the current clean
   fixed-depth Gemma branch as the reviewed fixed-width arithmetic base.
2. Add and certify physical B1 verifier projections for C2, C3, and C4 against
   independent ordinary B1/L1 projection calls.
3. Route single-prompt MTP verification widths C2, C3, and C4 through the
   installed B1 exact verifier after construction-time topology, quantization,
   storage, dtype, and artifact validation.
4. Preserve serial width-1 arithmetic and token output exactly for every tested
   prompt and depth.
5. Profile drafter, target verification, head, synchronization, and round
   overhead outside the measured hot path.
6. Tune fixed MTP depths 1, 2, and 3 using a 16K prefix, a 1,024-token Python
   coding suffix, 1,024 decoded tokens, batch 1, concurrency 1, and three timed
   repetitions per depth.
7. Benchmark the winning exact configuration and the unchanged autoregressive
   control at 0, 64K, and 128K prefix, always followed by the same 1,024-token
   Python coding suffix and 1,024 decoded tokens, with three timed repetitions
   per cell.
8. Produce a deterministic, publication-quality single-prompt performance chart
   from the final JSON receipt, with prefill and decode throughput shown as
   separate metrics.

## Non-goals

- Batch sizes or concurrency greater than one.
- Promoting or benchmarking the existing physical-B8 verifier route.
- Approximate verification, tolerance-based token equivalence, or accepting a
  different generated stream for speed.
- A hot-path fallback from the exact verifier to stock or serial execution.
- Claiming the requested fivefold decode improvement without measurement.
- Promoting an MTP configuration that does not beat the unchanged AR control.
- Reusing the dirty `mlxfast-gemma4-mtp-tip-qmm` worktree or transplanting the
  older divergent batch-8 verifier branch.

## Architecture and data flow

### Construction boundary

Model loading validates the fixed Gemma 4 target topology, artifact width,
quantization metadata, tensor storage, and supported verifier widths once. If
all invariants hold, it installs typed B1/C2, B1/C3, and B1/C4 verifier contexts
and a fixed route table. The installed context includes the physical batch and
column width, so a B8 context cannot satisfy a B1 request. If any invariant
fails, construction fails before warmup with a specific error; an experimental
route is never partially installed.

### Decode round

For fixed draft depth `d`:

1. The drafter proposes `d` tokens using the existing cache-correct MTP path.
2. The round forms the target verification window of width `d + 1`.
3. The construction-installed route directly selects B1/C2, B1/C3, or B1/C4.
4. The exact verifier evaluates that width with fixed-shape kernels whose
   arithmetic matches independent serial `[1, 1]` target calls.
5. The existing acceptance rule commits the matching prefix plus the target
   correction token and rolls caches back to the committed length.

No model metadata checks, environment reads, engagement counters, exception
fallbacks, or eligibility branches are added per layer, token, or round.

### Explicit phase routing

Autoregressive decode remains an explicit serial route. Exact fixed-width MTP is
an explicit construction-installed route. Unsupported widths fail before the
benchmark; they do not silently use stock rectangular or serial verification.

### Benchmark and chart pipeline

The benchmark runner emits one authoritative JSON receipt containing source and
artifact hashes, workload token counts, run order, individual samples,
arithmetic means, min/max values, output digests, acceptance statistics, peak
memory, and service/lock lifecycle evidence.

The chart generator consumes only a receipt that passes schema and completeness
validation. It emits an SVG as the canonical artifact and a PNG rendering when
an available renderer can reproduce it. The chart contains two aligned panels:

- Prefill throughput versus extra prefix length: 0, 64K, and 128K.
- Decode throughput over the same contexts: unchanged AR versus the winning
  exact MTP depth.

Each point/bar shows the three-run arithmetic mean with min-to-max whiskers. The
subtitle states `Gemma 4 26B A4B`, `single prompt`, `batch 1`, `concurrency 1`,
`1K Python input + 1K decode`, and `mean of 3`. A compact callout reports the
16K depth sweep and winning acceptance/committed-tokens-per-round statistics.
Prefill and decode never share a scale or become one aggregate TPS number.

## Interfaces and contracts

### Exact verifier contract

- The promoted physical batch is exactly one; supported logical widths are
  exactly 2, 3, and 4.
- For each width, target logits, chosen tokens, accepted count, committed token
  sequence, and final cache lengths match the serial verifier.
- Every fixed-width projection is bit-exact to concatenating `C` independent
  ordinary `[1, 1, ...]` projection results; comparison against the existing
  B8 oracle is insufficient.
- The verifier head uses the artifact-declared width; tests may not substitute a
  smaller convenient vocabulary or hidden width for route certification.
- Route installation is immutable after model construction.

### Benchmark contract

- Model: the pinned local Gemma 4 26B A4B weights and assistant-head artifact.
- Source: a clean committed revision derived from `origin/main@8fbf2f3`.
- Workload: identical 1,024-token Gemma-tokenized Python coding suffix in every
  cell; extra prefix lengths 0, 65,536, and 131,072 tokens.
- Decode: exactly 1,024 committed output tokens per timed sample.
- Execution: width 1, batch 1, concurrency 1.
- Statistics: one discarded primer per configuration, then three timed samples;
  report each sample and the arithmetic mean.
- Metrics: prefill tok/s and decode tok/s are separately timed and reported.
- Correctness: all three repetitions within a configuration must have the
  expected output digest; control/candidate parity is required where their
  algorithms promise identical output.
- Lifecycle: the parent holds `/tmp/mtplx-gpu-exclusive.lock` before any
  MLX/Metal execution, unloads the exact service only while holding it, runs the
  owned child under the guarded lifecycle, restores the exact service, verifies
  health and idle state, and releases the lock last.

### Promotion gates

1. Focused B1/C2-C4 kernel and route tests pass against independent B1/L1
   projection calls.
2. Multi-prompt real-model token/digest and cache parity against serial
   verification pass at depths 1, 2, and 3.
3. The unchanged AR control remains within its repeatability band.
4. At 16K + 1K input, the winning exact MTP depth has a higher three-run mean
   decode tok/s than the matched AR control.
5. The final 0/64K/128K matrix is complete, digest-valid, and free of dirty
   timing or lifecycle boundaries.

The approximately fivefold decode uplift requested by the user is a stretch
target, not a correctness waiver. The measured result is reported honestly even
if the gain is smaller.

## Testing strategy

Implementation follows test-driven development:

1. Add failing construction and route tests before enabling any verifier route.
2. Add failing physical-B1 fixed-width projection parity tests for C2, C3, and
   C4 before integrating the corresponding kernel implementation.
3. Add failing full-round tests for logits, selected tokens, accepted tokens,
   committed sequence, and cache lengths before changing the engine strategy.
4. Run existing real-engine single and batch token-losslessness tests to ensure
   the width-1 work does not regress shared MTP behavior.
5. Use a small bounded-shape GPU self-check before the full model to establish
   compile and allocation headroom.
6. Profile the matched 16K + 1K workload before and after each single
   optimization; retain only attributable improvements.
7. Unit-test receipt validation and chart generation with synthetic fixtures,
   including rejection of missing samples, mixed revisions, mixed workloads,
   non-finite values, and combined/ambiguous TPS fields.

## Error handling

- Artifact, topology, quantization, storage, or dtype mismatch: fail once during
  construction with the violated invariant.
- Unsupported draft depth/verification width: reject the request before model
  execution.
- Token, logit, acceptance, or cache parity failure: disable promotion of the
  candidate and preserve the serial route as the explicit control; do not add a
  runtime fallback.
- Incomplete or mixed benchmark samples: mark the receipt invalid and do not
  chart it.
- GPU child failure or interruption: terminate only the owned child, allow the
  lock-owning guard to restore the service, verify service health and lock
  release, then diagnose before retrying.

## Failure-mode check

### 1. Component parity passes but full-model decode still diverges — critical

Fixed-width QMM kernels can be exact in isolation while attention scheduling,
RoPE position handling, or speculative cache ownership diverges in a complete
round. The design therefore requires full-round and multi-prompt real-model
token/digest/cache parity at C2/C3/C4 before route installation can be promoted.

### 2. Exact rectangular verification remains slower than AR — critical to the performance goal

The drafter, dispatches, synchronization, or output head may dominate after the
serial target forwards are removed. The design profiles these boundaries and
uses an unchanged 16K + 1K AR bracket as the first performance gate. A candidate
that remains at or below the AR mean is not called a success and is not used as
the chart headline. The final chart still shows that best exact MTP result as an
honest comparison, with AR labeled as the winner.

### 3. A physical-B8 verifier is mistaken for the single-prompt route — critical

The clean verifier commits compile and install on the real artifact but are
gated to `tokens.dim(0) == 8`, so tests that inspect only installation would
produce a false engagement claim. The revised route key includes both physical
batch and columns, full-round tests assert B1/C2-C4 engagement from the model
boundary, and the receipt must report `batch_size: 1` and positive rectangular
verification rounds.

### 4. Construction accepts an artifact with incompatible fixed-width geometry — critical

A test-only or adjacent model shape could compile yet violate the real Gemma
artifact's width or quantized layout. The verifier is bound only after validating
the real artifact-declared topology and head width, and the route has no enabled
fallback. Mismatch fails before warmup.

## Rollout

1. Cherry-pick the three clean verifier commits into the isolated clean-tip
   worktree and resolve only conflicts caused by the four already-committed MTP
   depth/cache fixes.
2. Add physical-B1 route keys and prove B1/C2-C4 projection parity against
   independent B1/L1 calls.
3. Wire only the B1 exact verification route for this workload and prove
   full-round token, logit, acceptance, and cache parity.
4. Profile and tune one change at a time at 16K + 1K.
5. Run the full guarded 0/64K/128K three-repeat matrix.
6. Generate the chart from the final receipt and visually inspect the SVG/PNG
   for clipped labels, misleading scales, and data transcription errors.
7. Keep the branch local unless the user separately authorizes remote delivery.
