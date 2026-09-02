# Gemma 4 DFlash 300 tok/s Design

**Status:** Approved for implementation.

## Goal

Raise the exact Gemma 4 single-prompt DFlash benchmark from the verified
201.43 decode tok/s mean to at least 300 decode tok/s while preserving:

- physical batch 1 and concurrency 1;
- the exact 1,024-token Python prompt and 128-token decode workload;
- the retained output-token digest
  `704585a3a78c96932198c35e98e9c9e9018e16951e5ccb40a9bd7f3b4941c37d`;
- separate prefill and decode timing;
- one warmup followed by three measured samples;
- construction-bound verifier routes with no hot-path eligibility checks,
  metadata validation, environment reads, proof counters, or silent fallback.

The performance gate is strict: the mean and every retained measured sample
must be at least 300 decode tok/s. A faster result with a different token digest
is a failed candidate.

## Measured Baseline and Bottleneck

The retained D15/C16 period-two route measures 201.43 decode tok/s and 0.63546 s
decode wall time over three samples. It executes 29 target-verification rounds.

The first 35 generated tokens consume 23 C4 DFlash rounds before the existing
period-two phase can be proven. Their repeated token structure is:

```text
A, B, unknown, A, B, unknown, ...
```

For the exact Python continuation, `A, B` decode to newline and `import`. An
offline replay over the captured target-verified token stream shows that a
generic learned `A, B, unknown` phase removes exactly five early rounds. That
change alone projects to roughly 240 tok/s and therefore cannot satisfy the
goal by itself.

The period-two tail currently needs six C16 target passes. After reducing the
early round count, the remaining opportunity is a measured wider target-only
verification rectangle for the already-proven period-two cycle.

## Considered Approaches

### 1. Learned structural proposals plus wider period-two verification

This is the selected design. It attacks both measured sources of wall time,
keeps every proposal target-verified, and reuses the existing compiled verifier
bank and cache transaction.

### 2. Wider period-two verification only

This is too narrow. The 23-round early phase already consumes approximately the
entire 0.4267 s decode budget implied by 300 tok/s.

### 3. Top-k tree verification

This is rejected for this iteration. Captured drafter diagnostics place the
correct rejection token in the top eight at only 16 of 40 rejection points;
several important early module tokens are absent. Tree attention would add a
new target-mask and cache-ownership geometry without evidence that it can clear
the gate.

## Architecture

### Construction-time route

Only the requested D15 experimental lane installs this design. Construction
binds:

1. the ordinary C4 DFlash proposal function;
2. a C4 structural-proposal verifier function;
3. fixed-shape period-two verifier functions for the candidate widths selected
   for the current experiment;
4. the existing exact cache transaction and target layer IDs.

Invalid or missing bindings fail once before measured generation. D1-D14 keep
their existing loop and do not inspect the new phase state.

### Runtime phase state

The D15 session owns one explicit phase enum:

```text
dflashC4 -> structuralC4 -> dflashC4 -> period2Wide
```

Transitions depend only on committed target tokens and acceptance results,
which genuinely vary at runtime. No model/configuration metadata is rechecked.

#### `dflashC4`

Run the existing C4 DFlash round. Retain the last six committed target tokens.
When they match `A, B, X, A, B, Y`, install `(A, B)` and transition to
`structuralC4` for the next round.

The existing four-token period-two detector continues to observe committed
target tokens. When it proves `A, B, A, C`, it installs the latest pair and
transitions irreversibly to `period2Wide`.

#### `structuralC4`

Submit the fixed proposal `[A, B, A]` through the existing precomputed-proposal
round. The third token is only a sentinel: when the sequence is
`A, B, unknown`, the verifier accepts the first two proposals and emits the
target's unknown token as the bonus column, committing three tokens in one
round.

If fewer than the first two proposals are accepted, transition explicitly to
`dflashC4` for the next round. This is a phase transition after a fully
target-verified result, not a try-custom-then-fallback branch inside a round.
The route may be reinstalled only after a new six-token structural proof.

#### `period2Wide`

Generate the already-proven two-token cycle on the CPU, materialize one fixed
proposal rectangle, and invoke the construction-bound target verifier directly.
The DFlash draft model and hidden projection are not executed in this phase.

The phase is irreversible because every committed pair updates the cycle and
the target still verifies every proposed column. A rejection remains correct:
the ordinary accept walk emits the target mismatch and rewinds rejected cache
columns exactly as today.

### Width selection

Width is selected by guarded measurement, not intuition:

1. retain C16 as the unchanged control;
2. add and measure C32;
3. if C32 does not clear 300 tok/s, add and measure C64;
4. if C64 does not clear the gate and measured memory headroom remains safe,
   add and measure C96;
5. retain the lowest-decode-wall candidate that passes parity and memory gates.

Only one width is added and measured at a time. A slower width is removed before
testing the next width so attribution remains clear. The final installed bank
contains only widths used by the retained D15 route plus the unchanged standard
routes.

The benchmark receipt must report the actual maximum physical verifier width;
the D15 request depth alone is not sufficient evidence once a target-only phase
uses a wider rectangle.

## Interfaces and Ownership

- `Gemma4DFlashProposalPhasePolicy` is a pure value type that owns the six-token
  structural detector, the four-token period-two detector, and phase
  transitions.
- `RuntimeWorkerDFlashFreeRunSession` dispatches once per round using the
  installed phase enum. Each phase calls one prebound round function directly.
- `Gemma4DFlashCompiledVerifierBank` owns fixed-shape compiled target forwards.
  It installs only certified widths during construction.
- `runDFlashGreedyProposalRound` remains the single accept-walk and cache
  transaction for structural and period-two proposals. Its arithmetic is not
  forked again.
- The benchmark harness records requested depth and actual physical verifier
  width separately.

## Correctness and Error Handling

- All structural and period-two proposals are target-verified before commit.
- Cache begin/commit/rewind arithmetic is unchanged.
- A structural mismatch changes only the next round's explicit phase.
- Missing compiled widths, incompatible cache geometry, or impossible proposal
  sizes fail before measured generation.
- Stop-token handling and the exact `committed_total == 128` assembly contract
  remain unchanged.
- No enabled phase catches verifier errors or silently calls another route.

## Memory Safety

The current C16 worker peak is approximately 24.5 GB on a 128 GB machine. A
wider width is not assumed safe merely because the C16 peak is known.

Before each wider full-model run:

1. statically account for the additional fixed cache and activation shapes;
2. compile and self-check the new width under the exclusive GPU guard;
3. inspect measured MLX peak memory and system pressure;
4. proceed to the next width only with hard headroom;
5. restore and verify the exact production service before releasing the guard.

## Testing Strategy

### Pure policy tests

- D15 recognizes `A, B, X, A, B, Y` and installs structural C4.
- D1-D14 never install the structural or wider phases.
- Structural proposals are exactly `[A, B, A]`.
- Two accepted structural prefix tokens retain the structural phase.
- An earlier mismatch transitions to DFlash for the next round.
- A later four-token period-two proof supersedes structural state and installs
  the wide phase.
- Period-two proposals track the newest verified pair.

Every behavior test is introduced red, observed failing for the intended
reason, then made green with the minimum implementation.

### Source and construction tests

- Standard DFlash loops contain no structural phase checks.
- D15 binds every required verifier width during construction.
- Proposal phases call the existing explicit precomputed-proposal round.
- No `try?`, eligibility fallback, environment read, or hot-path proof counter
  appears in the installed D15 loop.
- The benchmark receipt records actual physical width.

### Guarded real-model gates

For each candidate width:

1. one warmup plus one scout sample;
2. exact prompt digest and 128 committed tokens;
3. exact retained token digest;
4. safe peak-memory receipt;
5. if promising, one warmup plus three measured samples;
6. fresh unchanged C16 control using the same worker hash;
7. production service identity, health, idle state, and GPU-lock release.

The final result must have all three measured decode samples and their mean at
or above 300 tok/s. Prefill TPS is reported separately and is not included in
the decode claim.

## Failure-Mode Review

### False structural match

**Severity:** Minor for correctness, potentially material for performance.

The target verifier prevents incorrect commits. The first mismatch transitions
the next round back to C4 DFlash, bounding the performance loss. The route is
not claimed as a universal improvement for other prompts.

### Wider verifier becomes slower or exceeds safe memory

**Severity:** Critical if installed without measurement.

Widths are added one at a time, guarded, and retained only after parity,
wall-time, and hard memory gates. A failed width is removed before the next
candidate.

### Benchmark-specific overfitting

**Severity:** Minor for the named objective, critical for a general product
claim.

The detector learns token relationships at runtime and contains no hard-coded
token IDs or prompt digest. Nevertheless, the performance claim is limited to
the exact 1K/128 benchmark until additional prompts and contexts are measured.

## Non-goals

- Batched or concurrent DFlash.
- MTP changes.
- Prefill optimization.
- 64K or 128K context validation.
- Top-k tree attention.
- A general 300 tok/s product claim across arbitrary prompts.
- Changes to sampling semantics or target arithmetic.

## Rollout

All work remains local in the existing isolated worktree. No commit, push, PR,
or production-server configuration change occurs without a separate user
request. Rejected benchmark artifacts remain as evidence; only the fastest
digest-preserving candidate is retained in the code path.
