# Gemma 4 Fidelity-Gated DFlash 200 tok/s Design

**Status:** Approved for implementation on 2026-08-31.

## Supersession

This design supersedes the performance and fidelity conclusions in
`2026-08-31-gemma4-dflash-300-tps-design.md`. The C96 result used by that
design measured a real fast execution path, but it did not preserve the
current serial token trajectory and is not an admissible speedup.

## Goal and Acceptance Contract

Reach at least 200 decode tok/s for Gemma 4 on the exact single-prompt
benchmark while preserving the benchmark contract:

- physical batch 1 and concurrency 1;
- the pinned 1,024-token Python prompt and 128-token decode;
- prefill and decode measured and reported separately;
- one warmup followed by three retained measured samples;
- every retained decode sample, and their arithmetic mean, at least
  200 tok/s;
- at most 12 positional token mismatches out of 128 versus a fresh serial
  control from the same worker build;
- requested DFlash depth and maximum physical verifier width reported
  separately;
- construction-bound routes with no hot-path metadata checks, environment
  reads, proof counters, exception fallback, or silent stock fallback.

Exact token identity remains the preferred outcome, but it is not required.
Up to 12 mismatches are permitted to accommodate target tie-break drift. A
candidate with 13 or more mismatches is rejected regardless of throughput.

## Current Evidence

Fresh measurements on the current worker establish:

| Route | Decode tok/s | Fidelity result | Status |
|---|---:|---:|---|
| Serial B1 | 113.15 | 0/128 mismatches | Correct control |
| Ordinary DFlash D14/C15 | 95.35 | Diverges from the current serial chain | Reject |
| Target-only periodic C96 | 354.61 mean | 108/128 mismatches | Reject |

The C96 timing is useful only as evidence that wide target rectangles can be
fast. Its token chain diverges at generated position 4, so it cannot support a
fidelity-preserving performance claim.

The current serial output develops a period-18 suffix. A generic shortest-
suffix detector first proves that recurrence after 38 generated tokens. This
is an optimization opportunity, not a correctness assumption: every proposed
token remains target-verified.

The existing `CBv2Gemma4MTPVerifierRoute.production` is the only current route
that deliberately preserves serial decode arithmetic. It is construction-
bound and certified for physical B1 with C2 through C4 only. The ordinary
DFlash rectangular target forward explicitly does not promise serial-identical
arithmetic. Therefore, simply changing the current C96 proposal width cannot
repair fidelity.

## Selected Design

### 1. Use the certified verifier for the initial DFlash phase

The experimental single-stream lane installs an exact C4 target verifier using
the same projection strategies, serialized causal attention, cache ownership,
and fixed-shape binding used by the existing Gemma 4 MTP verifier route.

The drafter may continue to create the initial C4 proposals, but those
proposals are committed only through the certified target verifier and the
existing DFlash accept walk. The initial gate is a full 128-token comparison
against the fresh serial control. No wide work begins until C4 is within the
12-token fidelity budget.

### 2. Learn recurrence only from committed target tokens

The session owns a generic suffix-period detector. It examines only committed
target tokens and contains no prompt digest, hard-coded token IDs, or
benchmark-specific token constants.

The detector considers periods 2 through 32. A period is installed only after
the shortest suffix contains two consecutive complete copies of that period.
The learned cycle is then repeated to form target-only proposals.

The period-18 observation explains the named workload, but period 18 is not
encoded as a special case.

### 3. Extend the serial-equivalent verifier one width at a time

Add fixed physical widths in this order:

1. C8;
2. C16;
3. C32.

Each width extends the existing construction-time Gemma 4 verifier contract:

- physical B1 only;
- the same per-projection route selection as the certified C2–C4 path;
- serialized causal attention unless a separately proven implementation is
  bit-equivalent or within the token-fidelity budget;
- fixed-shape compiled functions installed before measured generation;
- cache offsets, accepted-prefix commit, rejected-suffix rewind, and bonus
  token handling unchanged.

Only one new width is implemented and evaluated at a time. The next width is
not added until the current width passes its construction tests, real-model
self-check, 128-token fidelity gate, and memory gate. The retained route is the
smallest width that satisfies the 200 tok/s contract.

If C32 remains below 200 tok/s, the result is a measured architectural limit,
not permission to reuse the divergent ordinary C96 verifier. A further width
requires a new measured design decision.

### 4. Use explicit runtime phases

The experimental lane has two runtime phases:

```text
exactDFlashC4 -> periodicExactWide
```

`exactDFlashC4` runs the installed C4 proposal and verification function. The
period detector observes the committed output after each round.

`periodicExactWide` materializes the learned cycle at the retained fixed width
and invokes the prebound target verifier directly. It does not run the DFlash
draft projection.

If a wide round rejects any proposed token, the round completes through the
normal target-verified accept walk and cache transaction. The next round makes
an explicit phase transition back to `exactDFlashC4`, clears the learned
period, and relearns from later committed target tokens. This is not a
try-custom-then-stock fallback within a measured round.

The final shortened emission still calls the installed fixed-width verifier;
the existing emission limit and cache rewind discard uncommitted tail columns.
No dynamic-width compilation occurs in the loop.

## Construction and Ownership

### `CBv2Gemma4MTPVerifierRoute`

Owns the certified width set and the immutable projection, attention, gate/up,
QKV, and tied-head strategies for each width. Unsupported shapes return no
route during construction and cannot enter execution.

### `Gemma4TextModel`

Installs the fixed B1 verifier contexts atomically. Installation validates the
model topology, storage layout, dtype, kernel compatibility, and cache shape
once. It exposes prebound fixed-width forwards only after all required layer
bindings succeed.

### `Gemma4DFlashCompiledVerifierBank`

Owns the required C4 and one retained experimental wide forward, stable cache
arrays, and compile/evaluation roots. It invokes the installed forward
directly and has no enabled-path eligibility branch.

### `Gemma4DFlashProposalPhasePolicy`

Is a pure session value that owns committed-token history, generic recurrence
detection, and phase transitions. It does not own model validation or choose
among stock and optimized kernels.

### `RuntimeWorkerDFlashFreeRunSession`

Owns the accept-walk loop and reports requested depth and actual maximum
physical verifier width. It removes the current period-2-specific C96 route
from the fidelity-gated lane.

## Correctness Gates

Every width must pass all of these gates in order:

1. CPU-only route and policy tests.
2. Fixed-shape construction tests proving that all required projection and
   attention bindings exist before execution.
3. Real-model per-width verifier self-check under the exclusive GPU guard.
4. One exact 1K/128 scout against a fresh serial control from the same worker
   build.
5. Positional mismatch count no greater than 12/128.
6. Exactly 128 committed tokens, valid stop handling, and correct cache
   frontier after rejected and shortened-tail rounds.
7. Only after correctness: guarded performance measurement.

The receipt stores the serial and candidate token arrays, positional mismatch
count, both digests, worker hash, requested depth, maximum physical width,
acceptance lengths, prefill wall time/TPS, and decode wall time/TPS. A digest
match alone is sufficient for zero mismatches, but a digest mismatch must be
resolved by the positional comparison rather than treated as an automatic
failure.

## Performance Measurement

For each fidelity-admissible width:

1. run one warmup and one measured scout;
2. compare with a fresh unchanged serial control from the same worker;
3. if the scout is plausibly capable of 200 tok/s, run one warmup plus three
   retained samples;
4. retain the lowest decode wall-time configuration whose three samples and
   mean are each at least 200 tok/s;
5. continue reporting prefill separately; prefill TPS cannot satisfy the
   decode gate.

The final proof is invalid if it lacks token arrays, mismatch count, physical
width, worker identity, or any of the three sample wall times.

## GPU and Memory Safety

All model loading, compilation, self-checking, profiling, and benchmarking run
under the canonical parent-held `/tmp/mtplx-gpu-exclusive.lock` workflow.
The guard acquires the lock before unloading production, owns the GPU child,
restores the exact production service, verifies health and idle state, and
releases the workflow only afterward.

Before each new full-model width:

1. statically account for fixed cache, activation, compile graph, and
   projection scratch shapes;
2. establish a hard peak bound with smaller shapes or allocation accounting;
3. check current wired memory and leave safe headroom for the compile peak;
4. run the guarded full-model self-check only after the bound is established;
5. record MLX peak memory and system pressure before considering the next
   width.

A free GPU lock is not treated as proof that loading is memory-safe.

## TDD and Verification Strategy

Behavior changes are introduced with failing tests first:

- route rejects C8/C16/C32 before each width is installed;
- each installed width selects an explicit projection and attention strategy;
- generic recurrence detection chooses the shortest proven suffix in the
  2...32 range;
- fewer than two complete copies do not install a period;
- wide rejection completes correctly, then demotes the following round and
  clears recurrence state;
- a shortened final block keeps the fixed physical verifier width while
  committing only the requested tokens;
- the fidelity receipt calculates positional mismatches from captured token
  arrays;
- the enabled lane contains no ordinary-DFlash fallback or C96 special case.

After focused tests, run the complete relevant Swift test suite. The final
claim additionally requires fresh guarded real-model output; unit tests cannot
substitute for the 1K/128 fidelity and throughput gates.

## Failure-Mode Review

### Wider serial-equivalent geometry changes target arithmetic

**Severity:** Critical.

Reduction order, quantized projection geometry, or attention execution may
change argmaxes at larger widths. Each width is rejected if the complete
1K/128 chain exceeds 12 mismatches. No compatibility fallback masks the
failure.

### Serialized attention erases the batching gain

**Severity:** Critical for the 200 tok/s goal, not for correctness.

This is measured at each width. A correct but slower route is evidence, not a
win. Projection batching can still provide a gain while attention is
serialized, but the 200 tok/s claim remains open until all three retained
samples pass.

### False or transient recurrence

**Severity:** Low for correctness, material for performance.

The target verifier prevents unverified commits. Any rejection demotes the
next round to exact C4 and clears the learned period, limiting repeated bad
proposals without an in-round fallback.

### Benchmark overfitting

**Severity:** Critical for a general product claim.

The policy is generic and learns only relationships among committed tokens.
The resulting performance claim is nevertheless limited to the named 1K/128
workload until separately tested on other prompts and context lengths.

### Tail or rollback corrupts cache ownership

**Severity:** Critical.

The existing accept/rewind arithmetic is retained and locked with focused
tests. A cache-frontier failure blocks real-model benchmarking.

## Alternatives Rejected

- **Keep ordinary C96 and permit drift:** rejected because 108/128 mismatches
  exceed the accepted budget by 96 positions.
- **Start ordinary DFlash at C15/C16:** rejected because the current route is
  slower than serial and already diverges before any periodic acceleration.
- **Optimize serial decode alone:** rejected for this objective because the
  measured 113.15 tok/s baseline is too far below 200 without a demonstrated
  five-times-per-token kernel gain.
- **Claim the historical D11-D15 zero-mismatch table:** rejected because the
  current worker does not reproduce that result; current evidence is
  authoritative.
- **Hard-code the observed 18-token cycle:** rejected as prompt-specific and
  unnecessary; the generic suffix detector discovers it from verified output.

## Non-goals

- Batched or concurrent DFlash.
- MTP changes or MTP weight requantization.
- Prefill optimization.
- 64K or 128K context benchmarks.
- Ordinary block-shaped DFlash as a fidelity-preserving verifier.
- A general-serving performance claim beyond the exact 1K/128 workload.
