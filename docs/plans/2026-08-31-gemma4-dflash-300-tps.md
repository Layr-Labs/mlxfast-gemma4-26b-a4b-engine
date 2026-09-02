# Gemma 4 DFlash 300 tok/s Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the exact physical-B1, concurrency-1, 1,024-token Python prompt plus 128-token Gemma 4 DFlash decode sustain at least 300 decode tok/s in every one of three retained samples and in their mean without changing its output-token digest.

**Architecture:** Replace the D15-only period-two policy with one construction-installed proposal-phase policy that runs ordinary C4 DFlash, a target-verified learned structural C4 phase, and an irreversible target-only period-two phase. Compile one explicitly selected wide period-two target route in addition to the unchanged C2-C16 route bank, surface the physical width on the worker response, and tune C32, then C64, then C96 one at a time under the exclusive GPU guard. D1-D14 continue through the existing standard loop and never inspect this policy.

**Tech Stack:** Swift 6, Swift Testing, MLX Swift compiled graphs, Python 3 benchmark driver, macOS launchd GPU/service guard.

**Assumptions:** Assumes the retained 128-token target chain keeps the measured structural and period-two patterns — the candidate will not clear 300 tok/s if that chain changes. Assumes a C32, C64, or C96 target rectangle compiles within measured safe headroom — the plan will not run an unbounded full-model shape. Assumes target verification preserves the retained digest — any digest change rejects the candidate regardless of speed. Assumes the current dirty feature worktree is the intended workspace — this plan will not reset, overwrite, commit, push, or promote unrelated changes.

---

## File Structure

- Modify `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`: own the D15 phase machine, selected physical width, direct phase dispatch, and construction-bound compiled forwards.
- Modify `Sources/MLXFastHarness/Gemma4RuntimeWorker.swift`: emit the actual physical verifier width used by a DFlash session as an additive response field.
- Modify `Tests/MLXFastTests/Gemma4DFlashForwardTests.swift`: pure red/green phase-policy tests.
- Modify `Tests/MLXFastTests/Gemma4DFlashCBv2TargetCacheTests.swift`: construction and hot-path source-contract tests.
- Modify `/private/tmp/gemma_dflash_width4_mean3.py`: copy the worker-reported physical width into the benchmark receipt and reject a missing/mismatched echo.
- Generate `.benchmark-artifacts/gemma4-swift-dflash-reference-ab/*.json`: guarded scout, control, and final mean-three evidence.

## Baseline and Acceptance Contract

- Retained candidate baseline: `period2-c16-final-mean3-width16-mean3.json`, 201.42845459095565 decode tok/s mean, 0.6354615553 s mean decode wall, 29 rounds.
- Required token digest: `704585a3a78c96932198c35e98e9c9e9018e16951e5ccb40a9bd7f3b4941c37d`.
- Required workload: physical batch 1, concurrency 1, 1,024 prompt tokens, 128 committed decode tokens, one discarded warmup, three retained samples.
- Final gate: each retained `decode_tps` and their mean is at least 300; each retained token digest matches; prefill and decode rates remain separate; the exact production service is healthy before lock release.

### Task 1: Introduce the D15 proposal-phase policy test-first

**Files:**
- Modify: `Tests/MLXFastTests/Gemma4DFlashForwardTests.swift`
- Modify: `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`

**Security flag:** `none`

**Does NOT cover:** D1-D14, batched execution, MTP, sampling, stop-token semantics, target arithmetic, or selecting the final wide physical width.

- [ ] **Step 1: Replace the period-two-only tests with failing phase-policy tests**

Add these tests beside the existing width-policy tests:

```swift
@Test func d15ProposalPolicyLearnsStructuralC4FromTwoVerifiedFrames() {
    var policy = Gemma4DFlashProposalPhasePolicy(
        requestedDepth: 15, period2VerifierWidth: 32)
    #expect(policy.phase == .dflashC4)

    policy.record(committedTokens: [10, 11, 90], accepted: 0)
    policy.record(committedTokens: [10, 11, 91], accepted: 0)

    #expect(policy.phase == .structuralC4(first: 10, second: 11))
    #expect(policy.blockSize(baseDepth: 3) == 4)
    #expect(policy.makeDraftTokens(count: 3) == [10, 11, 10])
}

@Test func structuralC4StaysInstalledAfterAcceptingItsTwoTokenPrefix() {
    var policy = Gemma4DFlashProposalPhasePolicy(
        requestedDepth: 15, period2VerifierWidth: 32)
    policy.record(committedTokens: [10, 11, 90, 10, 11, 91], accepted: 0)
    policy.record(committedTokens: [10, 11, 92], accepted: 2)

    #expect(policy.phase == .structuralC4(first: 10, second: 11))
    #expect(policy.makeDraftTokens(count: 2) == [10, 11])
}

@Test func structuralC4MismatchReturnsToDFlashForTheNextRound() {
    var policy = Gemma4DFlashProposalPhasePolicy(
        requestedDepth: 15, period2VerifierWidth: 32)
    policy.record(committedTokens: [10, 11, 90, 10, 11, 91], accepted: 0)
    policy.record(committedTokens: [77], accepted: 0)

    #expect(policy.phase == .dflashC4)
    #expect(policy.makeDraftTokens(count: 3) == nil)
}

@Test func period2ProofSupersedesStructuralAndTracksLatestPair() {
    var policy = Gemma4DFlashProposalPhasePolicy(
        requestedDepth: 15, period2VerifierWidth: 32)
    policy.record(committedTokens: [10, 11, 90, 10, 11, 91], accepted: 0)
    policy.record(committedTokens: [41, 7, 41, 9], accepted: 0)

    #expect(policy.phase == .period2Wide(first: 41, second: 9))
    #expect(policy.blockSize(baseDepth: 3) == 32)
    #expect(policy.makeDraftTokens(count: 6) == [41, 9, 41, 9, 41, 9])

    policy.record(committedTokens: [52, 13], accepted: 6)
    #expect(policy.phase == .period2Wide(first: 52, second: 13))
    #expect(policy.makeDraftTokens(count: 4) == [52, 13, 52, 13])
}

@Test func proposalPhasesNeverInstallOutsideD15() {
    for depth in 1 ... 14 {
        var policy = Gemma4DFlashProposalPhasePolicy(
            requestedDepth: depth, period2VerifierWidth: 32)
        policy.record(committedTokens: [10, 11, 90, 10, 11, 91], accepted: 0)
        policy.record(committedTokens: [41, 7, 41, 9], accepted: 0)
        #expect(policy.phase == .dflashC4)
        #expect(policy.makeDraftTokens(count: 3) == nil)
    }
}
```

- [ ] **Step 2: Run the focused tests and observe the intended red state**

Run:

```bash
swift test --filter Gemma4DFlashForwardTests
```

Expected: compilation fails because `Gemma4DFlashProposalPhasePolicy` and `Gemma4DFlashProposalPhase` do not exist. No MLX runtime opt-in is set, so this command must not load or execute a Metal model.

- [ ] **Step 3: Implement the pure phase policy**

In `Gemma4DFlashFreeRunSession.swift`, replace `Gemma4DFlashPeriod2ProposalPolicy` with:

```swift
enum Gemma4DFlashProposalPhase: Equatable {
    case dflashC4
    case structuralC4(first: Int, second: Int)
    case period2Wide(first: Int, second: Int)
}

struct Gemma4DFlashProposalPhasePolicy {
    let requestedDepth: Int
    let period2VerifierWidth: Int
    private(set) var phase: Gemma4DFlashProposalPhase = .dflashC4
    private var recentTokens: [Int] = []

    func blockSize(baseDepth: Int) -> Int {
        switch phase {
        case .dflashC4, .structuralC4:
            return baseDepth + 1
        case .period2Wide:
            return period2VerifierWidth
        }
    }

    func makeDraftTokens(count: Int) -> [Int]? {
        guard count > 0 else { return nil }
        switch phase {
        case .dflashC4:
            return nil
        case .structuralC4(let first, let second):
            return Array([first, second, first].prefix(count))
        case .period2Wide(let first, let second):
            return (0 ..< count).map { $0.isMultiple(of: 2) ? first : second }
        }
    }

    mutating func record(committedTokens: [Int], accepted: Int) {
        guard !committedTokens.isEmpty else { return }
        let priorPhase = phase
        recentTokens.append(contentsOf: committedTokens)
        if recentTokens.count > 6 {
            recentTokens.removeFirst(recentTokens.count - 6)
        }
        guard requestedDepth == 15 else { return }

        if case .period2Wide = priorPhase {
            let pair = Array(recentTokens.suffix(2))
            phase = .period2Wide(first: pair[0], second: pair[1])
            return
        }

        let tail4 = Array(recentTokens.suffix(4))
        if tail4.count == 4, tail4[0] == tail4[2] {
            phase = .period2Wide(first: tail4[2], second: tail4[3])
            return
        }

        if case .structuralC4 = priorPhase {
            if accepted >= 2 { return }
            phase = .dflashC4
            return
        }

        guard recentTokens.count == 6,
            recentTokens[0] == recentTokens[3],
            recentTokens[1] == recentTokens[4]
        else { return }
        phase = .structuralC4(first: recentTokens[3], second: recentTokens[4])
    }
}
```

- [ ] **Step 4: Run the focused tests and verify green**

Run:

```bash
swift test --filter Gemma4DFlashForwardTests
```

Expected: all selected tests pass; runtime-gated MLX tests remain skipped because `MLXFAST_RUN_MLX_RUNTIME_TESTS` is unset.

### Task 2: Install direct D15 phase dispatch and one certified wide verifier

**Files:**
- Modify: `Tests/MLXFastTests/Gemma4DFlashCBv2TargetCacheTests.swift`
- Modify: `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`

**Security flag:** `none`

**Does NOT cover:** Dynamic metadata eligibility, runtime environment selection, silent fallback, top-k trees, widths other than the single source constant, or changes to the existing accept walk and cache transaction.

- [ ] **Step 1: Add failing construction and source-contract tests**

Add:

```swift
@Test func d15PhaseLoopBindsOneWideVerifierAndUsesExplicitPhaseDispatch() throws {
    let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(
        contentsOf: repositoryRoot.appendingPathComponent(
            "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
        encoding: .utf8)

    #expect(source.contains("static let d15Period2VerifierWidth = 32"))
    #expect(source.contains("requiredVerifierColumns:"))
    #expect(source.contains("case .dflashC4:"))
    #expect(source.contains("case .structuralC4:"))
    #expect(source.contains("case .period2Wide:"))
    #expect(source.contains("runDFlashGreedyProposalRound("))
    #expect(!source.contains("try? runDFlashGreedyProposalRound"))
}

@Test func standardLoopDoesNotInspectD15ProposalPhases() throws {
    let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(
        contentsOf: repositoryRoot.appendingPathComponent(
            "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
        encoding: .utf8)
    let standardStart = try #require(source.range(of: "func runStandardRounds("))
    let d15Start = try #require(source.range(
        of: "func runD15ProposalRounds(",
        range: standardStart.upperBound ..< source.endIndex))
    let standard = source[standardStart.lowerBound ..< d15Start.lowerBound]

    #expect(!standard.contains("proposalPhasePolicy"))
    #expect(!standard.contains("Gemma4DFlashProposalPhase"))
    #expect(!standard.contains("runDFlashGreedyProposalRound"))
}
```

- [ ] **Step 2: Run the focused tests and observe red**

Run:

```bash
swift test --filter Gemma4DFlashCBv2TargetCacheTests
```

Expected: the new source assertions fail because the D15 loop and selected C32 binding are absent.

- [ ] **Step 3: Bind the selected width at construction**

In `RuntimeWorkerDFlashFreeRunSession`, add:

```swift
static let d15Period2VerifierWidth = 32
private(set) var maximumPhysicalVerifierWidth: Int
private var proposalPhasePolicy: Gemma4DFlashProposalPhasePolicy
```

Initialize the policy and width once from `depth`; do not read an environment variable:

```swift
let wideWidth = depth == 15 ? Self.d15Period2VerifierWidth : depth + 1
self.maximumPhysicalVerifierWidth = wideWidth
self.proposalPhasePolicy = Gemma4DFlashProposalPhasePolicy(
    requestedDepth: depth,
    period2VerifierWidth: Self.d15Period2VerifierWidth)
```

Change `Gemma4DFlashCompiledVerifierBank` construction to receive `requiredVerifierColumns: Set<Int>`. Use `Set(2 ... MLXFastConstants.experimentalDFlashMaxBlockSize)` for unchanged standard routes and insert the selected D15 width only when `depth == 15`. Store the exact set in the bank, include it in `supports(...)`, and bind exactly its sorted members:

```swift
for columns in requiredVerifierColumns.sorted() {
    forwards[columns] = try target.bindCompiledDFlashGreedyForward(
        cache: cache,
        targetLayerIds: targetLayerIds,
        columns: columns)
}
```

- [ ] **Step 4: Replace the D15 optional-proposal branch with explicit phase dispatch**

Rename `runPeriod2Rounds` to `runD15ProposalRounds`. At each round snapshot `proposalPhasePolicy.phase`, compute the fixed route width, narrow only for the final remaining tokens, and dispatch directly:

```swift
let phase = proposalPhasePolicy.phase
let routeBlockSize = proposalPhasePolicy.blockSize(
    baseDepth: widthPolicy.currentDepth)
let roundBlockSize = Swift.min(routeBlockSize, remaining + 1)

let round: DFlashGreedyRoundResult
switch phase {
case .dflashC4:
    round = try runDFlashGreedyRound(
        target: target,
        drafter: drafter,
        targetCache: &targetCache,
        draftCache: draftCache,
        bonus: bonus,
        projectedContext: draftContext,
        promptTokenCount: promptTokenCount,
        generatedTokenCount: generatedTokenCount,
        blockSize: roundBlockSize,
        maxEmitCount: remaining,
        targetCacheTransaction: targetCacheTransaction)
case .structuralC4, .period2Wide:
    let proposal = proposalPhasePolicy.makeDraftTokens(
        count: roundBlockSize - 1)!
    round = try runDFlashGreedyProposalRound(
        target: target,
        targetCache: &targetCache,
        bonus: bonus,
        draftTokens: MLXArray(proposal.map(Int32.init))[.newAxis, .ellipsis],
        targetLayerIds: drafter.config.targetLayerIds,
        blockSize: roundBlockSize,
        maxEmitCount: remaining,
        targetCacheTransaction: targetCacheTransaction)
}
```

Project target hidden only for `.dflashC4`, then update the phase with the target-verified result:

```swift
if case .dflashC4 = phase {
    draftContext = try drafter.projectTargetHidden(round.targetHidden)
}
proposalPhasePolicy.record(
    committedTokens: round.tokens,
    accepted: round.accepted)
```

Preserve the existing builder, stop-token, `generatedTokenCount`, drafted/accepted, rollback, and final-tail behavior unchanged.

- [ ] **Step 5: Run focused and full non-Metal verification**

Run:

```bash
swift test --filter Gemma4DFlashForwardTests
swift test --filter Gemma4DFlashCBv2TargetCacheTests
swift test
```

Expected: focused suites pass. The full suite may still expose the already-known unrelated wire-fixture digest mismatch; if it does, record the exact failing test and confirm no new failure before continuing.

### Task 3: Make the benchmark receipt prove the physical width

**Files:**
- Modify: `Tests/MLXFastTests/Gemma4DFlashCBv2TargetCacheTests.swift`
- Modify: `Sources/MLXFastHarness/Gemma4RuntimeWorker.swift`
- Modify: `/private/tmp/gemma_dflash_width4_mean3.py`

**Security flag:** `none`

**Does NOT cover:** Changing the effective requested draft depth, claiming a worker width from the Python environment, or adding hot-path counters. This is a once-per-run response echo from construction-owned session state.

- [ ] **Step 1: Add a failing additive-wire source test**

Add source assertions that `RuntimeWorkerResponse` owns `physicalVerifierWidth`, maps it to `physical_verifier_width`, and the DFlash `free_decode_run` response receives a construction-owned `dflashPhysicalVerifierWidth` captured before the session is consumed. Run:

```bash
swift test --filter Gemma4DFlashCBv2TargetCacheTests
```

Expected: the assertions fail because the response field is absent.

- [ ] **Step 2: Add the construction-owned response field**

Add `let physicalVerifierWidth: Int?`, an initializer argument/default, assignment, and coding key to `RuntimeWorkerResponse`. Immediately before `runFreeDecode`, capture:

```swift
let dflashPhysicalVerifierWidth =
    route == .dflash
    ? state.dflashFreeRunSession?.maximumPhysicalVerifierWidth
    : nil
```

Pass `physicalVerifierWidth: dflashPhysicalVerifierWidth` to the successful response. Do not add type checks or fallback inside the decode round loop.

- [ ] **Step 3: Make the Python receipt reject an unproven width**

In `/private/tmp/gemma_dflash_width4_mean3.py`, replace `PHYSICAL_WIDTH = DEPTH + 1` with a required candidate expectation:

```python
EXPECTED_PHYSICAL_WIDTH = int(
    os.environ.get("GEMMA_DFLASH_PHYSICAL_WIDTH", str(DEPTH + 1))
)
```

After each `free_decode_run`, require and retain the worker echo:

```python
physical_width = int(run.get("physical_verifier_width", -1))
if MODE == "dflash" and physical_width != EXPECTED_PHYSICAL_WIDTH:
    raise RuntimeError(
        f"physical verifier width {physical_width} != expected "
        f"{EXPECTED_PHYSICAL_WIDTH}"
    )
row["physical_verifier_width"] = physical_width
```

Set report-level `physical_verify_width` from the unanimous sample echoes after all samples, not from `DEPTH`.

- [ ] **Step 4: Re-run non-Metal wire and source tests**

Run:

```bash
swift test --filter Gemma4DFlashCBv2TargetCacheTests
swift test --filter EmitWireFixtureTests
git diff --check
```

Expected: source tests pass; update the additive wire fixture only if this repository intentionally seals all optional response keys. `git diff --check` emits no output.

### Task 4: Tune one wide rectangle at a time and prove the 300 tok/s gate

**Files:**
- Modify iteratively: `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`
- Generate: `.benchmark-artifacts/gemma4-swift-dflash-reference-ab/structural-c32-scout-width32-mean3.json`
- Generate conditionally: corresponding C64 and C96 scout receipts
- Generate: a fresh unchanged C16 control receipt from the retained worker source
- Generate: final warmup-plus-three receipt for the retained candidate

**Security flag:** `none`

**Does NOT cover:** 64K/128K prefill, concurrency greater than one, batch greater than one, MTP, general-prompt claims, committing, pushing, or changing the production service configuration.

- [ ] **Step 1: Build the C32 worker without loading Metal**

Run:

```bash
swift build -c release --product mlxfast-runtime-worker --build-path .build-worker
shasum -a 256 .build-worker/release/mlxfast-runtime-worker
```

Expected: release build succeeds and the worker SHA is recorded for every paired receipt.

- [ ] **Step 2: Statically bound C32 before a full-model run**

Inspect the fixed-cache dimensions and C32 verification intermediates, compare their byte counts against the known C16 approximately-24.5-GB peak and 128-GB host, and write the bound into the scout receipt directory. Do not run the full model if the calculated peak is not hard-bounded with safe headroom.

- [ ] **Step 3: Run one guarded C32 scout**

Run only through the canonical guard:

```bash
GEMMA_DFLASH_LABEL=structural-c32-scout \
GEMMA_DFLASH_DEPTH=15 \
GEMMA_DFLASH_PHYSICAL_WIDTH=32 \
GEMMA_DFLASH_SAMPLE_COUNT=2 \
/opt/homebrew/bin/python3 \
  ~/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist ~/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --timeout-seconds 1800 \
  --child-timeout-seconds 1200 \
  -- /opt/homebrew/bin/python3 /private/tmp/gemma_dflash_width4_mean3.py
```

Expected: guard attestation present, prompt length 1,024, one warmup plus one retained sample, 128 committed tokens, required digest, physical width 32, safe MLX peak, exact production service restored healthy and idle, and lock released. Record prefill TPS separately.

- [ ] **Step 4: Retain or reject C32 from measured evidence**

If the retained scout is slower than the current 201.43 tok/s baseline or changes the digest, restore the previous constant immediately. If it improves but remains below 300, profile round count and decode wall, then change only `d15Period2VerifierWidth` from 32 to 64. Do not keep both wide candidates installed.

- [ ] **Step 5: Repeat the static bound, build, and guarded scout for C64 only if C32 misses**

Use label `structural-c64-scout`, `GEMMA_DFLASH_PHYSICAL_WIDTH=64`, and the same exact workload. Retain C64 only if it is faster than C32 with the same digest and safe peak. If C64 remains below 300, repeat once for C96 after a new hard memory bound; do not attempt a fourth width without new profiling evidence.

- [ ] **Step 6: Diagnose after each miss before widening again**

Compare, using the receipt and worker diagnostics:

```text
decode_s, decode_tps, rounds, acceptance_lengths,
drafted_total, accepted_total, committed_total,
physical_verifier_width, mlx peak, worker SHA, token digest
```

The next width is justified only if the remaining wall time is still dominated by multiple fully accepted period-two target passes. If early C4 rounds dominate after the structural change, stop widening and profile that phase instead.

- [ ] **Step 7: Run a fresh unchanged C16 control with the same source and worker hash**

Temporarily set `d15Period2VerifierWidth` to 16, rebuild, and run one warmup plus three measured samples under the guard. Then restore the winning width and rebuild. The control must retain the same target digest and workload; its purpose is attribution, not promotion.

- [ ] **Step 8: Run the final exact warmup plus three measurement**

With the retained winning width and release worker, run the canonical guard with `GEMMA_DFLASH_SAMPLE_COUNT=4`. Expected final evidence:

```text
samples.count == 3
all(sample.committed == 128)
all(sample.token_digest == required_digest)
all(sample.physical_verifier_width == retained_width)
all(sample.decode_tps >= 300.0)
mean.decode_tps >= 300.0
```

Report sample-by-sample prefill TPS, decode TPS, decode seconds, round trajectories, physical width, worker SHA, token digest, and peak memory. Do not average prefill and decode together.

- [ ] **Step 9: Run the completion audit**

Run fresh:

```bash
swift test --filter Gemma4DFlashForwardTests
swift test --filter Gemma4DFlashCBv2TargetCacheTests
swift test
git diff --check
rg -n "TODO|FIXME|placeholder|NotImplementedError" \
  Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift \
  Sources/MLXFastHarness/Gemma4RuntimeWorker.swift
```

Then inspect the final JSON directly and verify the exact service identity, health, idle state, and lock release from the guard receipt. Completion is unproven unless all three final samples clear 300 tok/s with the required digest and physical-B1/concurrency-1 workload.

## Execution Choice

Use **Inline Execution**. The four tasks share one dirty worktree, one phase-policy implementation, one compiled worker, and one exclusive GPU/service lifecycle; sequential state and one-variable-at-a-time attribution matter more than parallelism. No subagents are required or authorized.
