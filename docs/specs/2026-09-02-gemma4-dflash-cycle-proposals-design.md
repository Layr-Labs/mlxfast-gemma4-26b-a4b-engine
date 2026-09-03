# Gemma 4 DFlash on main: target-verified cycle proposals

Status: Proposal. Implemented and measured on branch `perf/gemma4-dflash-cycle-proposals` (not merged); blocked by the open issues in docs/benchmarks/gemma4-b1-humaneval-2026-09-02.md.
Supersedes the exact-verifier study (PR #2097, which stays as the record of that road).
Acceptance is HumanEval parity, not bit-exactness (David, 2026-09-02).

Base: `origin/main` of Layr-Labs/mlxfast-gemma4-26b-a4b-engine at 8ae4c54, 59 promoted
submissions. No exact or hand-bound verifier kernels are added. The stock rectangular target
verify (`DFlashTargetModel.forwardGreedyTokensForDFlash`) verifies every proposal, so main's
promoted kernels -- fused gate|up storage, QKFUSE, run-sum norms, fused top-8, v27 GEMV -- are
used unchanged and every future promoted kernel composes for free.

## The idea in one paragraph

A DFlash round is `[bonus, d1 ... dk] -> target verify -> accept walk`. Nothing in that round
cares where `d1 ... dk` came from: the target's own greedy argmax is what gets committed at
every emitted position, and a wrong proposal costs only the rejected tail of one block. So when
the committed output has fallen into a repeating cycle -- the degenerate loop a decoder can
enter and never leave -- the cheapest possible drafter is the cycle itself. Continuing it costs
no forward at all, and the same target forward accepts or rejects it exactly as it would a
drafter block. This is not loop suppression: the policy never changes a committed token, only
what gets proposed.

## What changed, where

### 1. `Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashGreedyRound.swift` (+79 / -18)

`runDFlashGreedyRound` gained one optional argument:

```swift
proposal: [Int]? = nil
```

When it is `nil` the round is byte-for-byte the ported one. When it is non-nil:

* `draftTokens = MLXArray(proposal.map(Int32.init))[.newAxis, .ellipsis]` (shape `[1, blockSize - 1]`);
* `drafter.draftBlock` is NOT called -- there is exactly one call site in the file and it lives
  in the `else` arm, pinned by a test;
* a proposal whose length disagrees with `blockSize - 1` is refused rather than silently
  verified at a different width, because every downstream counter is derived from `blockSize`;
* the verify input, the rectangular target forward, the accept walk, the emit clamp and the KV
  rollback are the same lines on the same rectangle;
* `drafted` / `accepted` are reported the same way, so `accepted_total / drafted_total` remains
  the acceptance rate of the rectangle that was actually run.

### 2. `Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashCycleProposalPolicy.swift` (+199, new)

A pure struct: no MLX, no environment reads, no clocks, no counters in the hot path. It is a
decision function over the committed token stream, which is why its rules are testable
exhaustively on a host with no Metal.

```swift
public init(ordinaryBlockSize: Int, wideBlockSize: Int)

public enum Action: Equatable, Sendable {
    case drafter(blockSize: Int)   // run draftBlock at this width
    case cycle(tokens: [Int])      // verify these instead; block size is tokens.count + 1
}

public mutating func nextAction(remaining: Int) -> Action
public mutating func record(roundTokens: [Int], proposed: Int, accepted: Int, terminal: Bool)

public var installedCycle: [Int]?        // read-only, tests and evidence
public var installedCycleOffset: Int?
```

`nextAction` narrows BOTH arms to what the run can still commit: a drafter block is
`min(ordinaryBlockSize, remaining + 1)`, a cycle proposal is `min(wideBlockSize, remaining + 1) - 1`
tokens. A round always emits at least the verified bonus column, so neither arm ever verifies a
column the run cannot commit.

Detection and judgement, unchanged in semantics from the branch's
`Gemma4DFlashRecurrencePolicy` (worktree `mlxfast-gemma4-mtp-depth3-tip`, commit 2b7e552):

* candidate periods 2 through 32; the SHORTEST non-constant period with one full copy plus three
  confirming committed tokens wins;
* period 1 is never installed -- a constant run would have the policy propose a block of
  identical tokens on the evidence of a single distinct value;
* three confirmations rather than two is what keeps a period-3 coincidence inside a period-18
  stanza from being installed as a period-3 cycle;
* a fully accepted wide round advances the cycle offset by the round's committed count (the
  proposed tokens plus the target's own bonus token on top);
* a wide round with `accepted < proposed` demotes the next round to the drafter and DISCARDS the
  detection history, so re-arming needs a fresh full period plus three confirmations;
* a terminal round -- one clipped by the run's own token budget, or ended by a stop token -- is
  never read as a rejection, because `accepted < proposed` there says nothing about the cycle.

Detection reads only tokens the target COMMITTED inside the current request. The policy is
constructed per session and dies with it, so nothing is cached across requests (track rule).

### 3. `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift` (+123)

The session consults the policy once per round and passes `proposal:` and the block size
through. Existing stop-token, evidence and accounting behaviour is unchanged, with one ordering
requirement made explicit: stop tokens are resolved BEFORE the policy trains on a round, so a
token the verifier produced past the stop can never become evidence.

Widths: ordinary block is the session's requested `depth + 1` (plumbing unchanged --
`free_decode_begin` passes the resolved `spec.dflash.depth`). The wide block is 16, which is
`MLXFastConstants.experimentalDFlashMaxBlockSize` and the drafter checkpoint's own trained
`block_size`, so a cycle round verifies a rectangle the target already runs. The wide block
never narrows below an ordinary round.

### 4. `DFlashDraftModel.submissionDraftDepth`

Left at 1. It is only the fallback for non-wire callers; the benchmark and benchd always pass an
explicit depth through `RuntimeWorkerSpecRegistry.resolveDFlashDepth`, and that resolved value is
both echoed and executed.

### 5. Tests: `Tests/MLXFastTests/DFlashCycleProposalPolicyTests.swift` (+296, 13 tests)

CPU-only, unconditional (not gated behind `MLXFAST_RUN_MLX_RUNTIME_TESTS=1`). Covers: period-18
promotion only at the 23rd committed token, the period-3 two-token false positive rejected at
three confirmations, period 1 and periods above 32 refused, a repeated tail (`[1,2,3,3]`) not
mistaken for a constant, shortest-period selection with evidence spanning rounds, both arms'
tail narrowing, cycle-offset walk across consecutive fully accepted wide rounds, demotion on a
partial wide round, the re-arm rule, the terminal round, and an empty round changing nothing.

The round and session seams are pinned against SOURCE TEXT rather than executed:
`runDFlashGreedyRound` takes a concrete `DFlashDraftModel`, not a protocol, so no spy can be
substituted, and constructing a real one allocates MLX arrays -- which is why every existing
DFlash forward test is skipped on the CPU suite. The two facts worth pinning are that a proposal
round cannot reach `draftBlock`, and that the session repays the drafter-context debt; both fail
SILENTLY if reverted (correct tokens, quietly collapsed acceptance), so a compile-time signal
beats nothing.

## The drafter cache: the one subtle part

`draftBlock` is the call that writes drafter context KV, and it advances that cache by
`targetHidden.dim(1)` -- the number of tokens the PREVIOUS round committed -- never by
`blockSize`. Look at `DFlashAttention.callAsFunction`: it caches keys/values of the projected
CONTEXT and merely concatenates the proposal's own keys for that one forward. That is why the
trim inside the round is a no-op in steady state: after `draftBlock`, `draftCache.offset` already
equals `promptTokenCount + generatedTokenCount - 1`.

A round that skips `draftBlock` therefore leaves the draft cache SHORT by that round's committed
tokens. The trim cannot repair it (trimming only removes context) and correctly does nothing --
the delta is negative.

The session repays the debt exactly rather than approximately. It holds each skipped round's
returned `targetHidden` -- which is the accepted prefix's hidden, so its length is exactly that
round's committed token count -- in `pendingDraftContext`, and hands the concatenation to the
next drafter round as one context. The drafter then caches the same vectors at the same RoPE
positions a run of ordinary rounds would have cached (`contextKeys` is roped at `cache.offset`,
which is where those tokens actually live), and the bonus column stays consistent because
`bonus` is the last committed token either way. The invariant
`draftCache.offset == promptTokenCount + generatedTokenCount - 1` is restored exactly on the next
drafter round, and the trim stays a no-op.

No replay through the drafter is needed, which is the simplification over the branch's
`replayWideDrafterAndDemote` / `gemma4DFlashReplayPlan` machinery: the context the skipped rounds
would have cached already exists as their committed hidden, so it is carried rather than
recomputed.

On a run where the policy never fires, `pendingDraftContext` is always one element and the
argument passed is main's own `targetHidden`, unchanged. Cost while a cycle runs: about 33 KB per
token held (6 taps x 2816, bf16); each slice is `asyncEval`'d when appended so the unevaluated
verify graph behind it is not kept alive across the loop.

## Why this is the integrated version

* Composes with every future promoted submission: no kernel duplication, no per-frontier
  re-derivation. The only contract used is the existing round function.
* Every emitted token is still target-verified through main's own forward. Divergence from
  serial is limited to near-tie argmax drift of the rectangular forward -- what the track's 10%
  budget exists for, and what David accepted subject to HumanEval.
* Small: ~280 lines of engine, ~300 lines of tests, ~120 lines of session wiring.

## Two things measurement should account for

1. **The diagnostics echo (FIXED).** `Gemma4RuntimeWorker.swift` used to echo
   `source: "DFlashDraftModel.draftBlock(blockSize: depth + 1)"` and `cap: depth + 1` into the
   free-run diagnostics sidecar (`configSource` / `rectangularCap`), which understated the
   widest rectangle this route verifies once a cycle is installed. It now names both draft
   sources and caps at `Swift.max(depth + 1, RuntimeWorkerDFlashFreeRunSession.cycleProposalBlockSize)`,
   read from the session's own constant so the echo and the policy cannot drift. Pure
   string/int change in the echo; nothing on the hot path.
2. **The rotating-cache rollback is already the expensive path on the 1K prompt.** Gemma 4's
   sliding layers use `RotatingKVCache(maxSize: sliding_window = 1024, keep: 0)`, and
   `isTrimmable` is `offset < maxSize`. With a 1024-token prompt the target cache is at the
   boundary from prefill on, so `makeDefaultDFlashCacheRollbackState` takes the copy-snapshot arm
   every round and a rejection replays the accepted prefix through a second forward. This is
   PRE-EXISTING on main and not caused by this change -- but a 16-wide cycle round rejects more
   tokens at once than a `depth + 1` round does, so it changes the price of a miss and belongs in
   the cost model when reading G2.

Also worth stating for anyone preparing a submission from this branch: `DFlashGreedyRound.swift`,
the new policy file, and `Sources/MLXFastHarness/` are all OUTSIDE `benchmark.json.editablePaths`
(which carries `MLXSpeculative/DFlashDraftModel.swift` but not the rest of that directory). This
is an engine/research branch as it stands, not a submittable surface.

## Gates (in order; each needs the GPU lock through the guard)

Numbers are placeholders until each gate runs.

* **G0 -- main baseline, batch 1, 1K Python prompt.** Serial tok/s + digest; stock DFlash
  D1/D3/D15 tok/s and positional mismatches vs main's serial.
  Serial: _TBD_ tok/s, digest _TBD_. D1 _TBD_ / D3 _TBD_ / D15 _TBD_ tok/s, mismatches _TBD_.
* **G1 -- quality reference.** HumanEval-164 pass@1 of main serial via the worker driver
  (EvalPlus 0.3.1, greedy, 768 max tokens). ~15 min GPU. Result: _TBD_.
* **G2 -- the measurement.** 1K-prompt tok/s for serial vs D\<ordinary\> vs D\<ordinary\>+cycle;
  mismatches vs main serial; token-level agreement rate on the HumanEval generations; and the
  cycle-round share (`cycleRoundsRun / roundsRun`). Result: _TBD_.
* **G3 -- acceptance.** HumanEval-164 pass@1 of the DFlash+cycle route. Pass = within 2 problems
  of G1 with no systematic failure class (per-problem diff reviewed). This is David's stated
  acceptance bar. Result: _TBD_.
* **G4 -- ship.** Push `perf/gemma4-dflash-cycle-proposals`, open a PR to main, mark ready for
  review. Only on David's word; nothing is pushed from the implementation session.

### Driving the 1K-prompt benchmark

`/private/tmp/gemma_dflash_width4_mean3.py` reads `physical_verifier_width` off the
`free_decode_run` reply. **main does not emit that field** -- it exists only on the depth3-tip
branch -- so the driver reads `-1` and must be run with `GEMMA_DFLASH_PHYSICAL_WIDTH=-1`, plus
`GEMMA_DFLASH_WORKER` pointing at this branch's `.build-worker/release/mlxfast-runtime-worker`.
Two caveats: the driver's `WORKTREE` is hardcoded to `mlxfast-gemma4-mtp-depth3-tip`, so
`source_head` in the report will name that branch rather than this one (the `dflash-head` it
points at is the same pinned drafter, so the run itself is fine); and there is no single honest
physical width any more -- ordinary rounds run `depth + 1`, cycle rounds run 16 -- so a real
width field would have to be reported per round, not per run.

## Expected outcome, stated up front

* On the 1K Python prompt (period-18 loop), the cycle route should approach the branch's
  203 tok/s if main's rectangular verify at width 16 is not slower than the branch's exact C16.
* On HumanEval-style output (no loops) the policy is inert, so expect DFlash to land at roughly
  serial or below. If G2 shows the ordinary DFlash route below serial, the shipped default
  becomes "cycle proposals only" with the neural drafter disabled -- which is then simply
  target-verified loop acceleration.
* darkbloom runs its own loop detection on Gemma 4 requests, so production value is bounded.
  This is worth landing as the composable engine hook plus the measured data.
