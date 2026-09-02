# Gemma 4 Fidelity-Gated DFlash 200 tok/s Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the exact physical-B1, concurrency-1, 1,024-token Python prompt plus 128-token Gemma 4 DFlash decode sustain at least 200 decode tok/s in every one of three retained samples and in their mean, with no more than 12 positional mismatches against a fresh serial control.

**Architecture:** Keep the D15 fidelity lane on its construction-owned CBv2 cache so the existing serialized rectangular-attention controller actually participates; the current compiled-cache copy remains exclusive to unchanged standard DFlash lanes. Install fixed B1 verifier contexts for C4/C8/C16/C32, using the proven C2-C4 kernels where available and construction-bound stock quantized modules at wider shapes. A generic committed-token suffix detector switches from exact C4 DFlash to one target-only fixed-width recurrence phase, and a rejection demotes the next round back to exact C4.

**Tech Stack:** Swift 6, Swift Testing, MLX Swift, Python 3 `unittest`, JSON benchmark receipts, macOS launchd, and the canonical exclusive GPU guard.

**Assumptions:** Assumes the current 113.15 tok/s serial chain is reproducible from the rebuilt worker — the plan will not use an older serial digest as its control. Assumes enabling the already-certified CBv2 serialized-attention controller on the D15 target cache brings C4 within 12/128 mismatches — recurrence and wide work stop if it does not. Assumes token-local stock quantized modules at C8/C16/C32 plus serialized attention retain enough batching benefit to reach 200 tok/s — the plan will not reuse the divergent ordinary C96 block verifier if they do not. Assumes the measured C96 peak of 24,474,883,100 MLX bytes bounds the smaller C32 position rectangle once the unchanged cache allocation is accounted for — a full-model run will not start if current wired memory removes the required headroom. Assumes the dirty feature worktree is authoritative — the plan will not reset, overwrite, commit, push, or promote unrelated changes.

---

## File Structure

- Create `tools/gemma4_dflash_fidelity_gate.py`: compare captured serial and candidate token arrays positionally and enforce the throughput/fidelity contract outside the measured path.
- Create `tools/tests/test_gemma4_dflash_fidelity_gate.py`: CPU-only contract tests for the receipt gate.
- Modify `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`: retain the CBv2 transaction for D15, install serialized attention once after prefill, own the generic recurrence policy, and select one fixed physical width.
- Modify `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/Gemma4MTPVerifierRoute.swift`: admit only the fixed B1 verifier widths and distinguish proven custom projections from prebound stock quantized projections.
- Modify `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4PrefillGlueV1.swift`: bind its row-generic exact glue kernels for the fixed wider verifier widths.
- Modify `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift`: atomically install C8/C16/C32 contexts with stock projection closures and the existing C2-C4 contexts with proven kernels.
- Modify `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4TextDFlash.swift`: require an installed verifier context for every fidelity-lane width and expose a direct construction-bound forward over the retained CBv2 cache.
- Modify `Tests/MLXFastTests/Gemma4DFlashForwardTests.swift`: pure recurrence, rejection, tail, and fixed-control tests.
- Modify `Tests/MLXFastTests/Gemma4MTPVerifierRouteTests.swift`: fixed-width route and construction-bound projection-strategy tests.
- Modify `Tests/MLXFastTests/Gemma4DFlashCBv2TargetCacheTests.swift`: cache-controller, explicit-lane, no-fallback, and response-receipt source contracts.
- Generate `.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-*.json`: fresh serial, C4 gate, staged wide scouts, and final warmup-plus-three evidence.

## Authoritative Acceptance Contract

- Physical batch: 1.
- Concurrency: 1.
- Prompt: the pinned 1,024-token Python prompt from `/private/tmp/gemma-b1-e203306-smallest-proof-20260830T1548/receipt.json`.
- Decode: exactly 128 committed tokens after the seed.
- Fidelity: no more than 12 positional mismatches against a fresh serial sample produced by the same worker SHA.
- Timing: prefill and decode remain separate.
- Final performance: three retained samples; every `decode_tps >= 200.0`; arithmetic mean `decode_tps >= 200.0`.
- Evidence: token arrays, token digests, positional mismatch counts, requested depth, physical verifier width, acceptance lengths, worker SHA, wall seconds, TPS, memory peak, and guard/service lifecycle.

### Task 1: Add a persistent external fidelity and performance gate

**Files:**
- Create: `tools/gemma4_dflash_fidelity_gate.py`
- Create: `tools/tests/test_gemma4_dflash_fidelity_gate.py`

**Security flag:** `none`

**Does NOT cover:** Token generation, prompt selection, model loading, GPU execution, semantic scoring, or accepting more than 12 positional mismatches. It validates already-captured receipts only.

- [x] **Step 1: Write failing receipt-gate tests**

Create `tools/tests/test_gemma4_dflash_fidelity_gate.py`:

```python
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "gemma4_dflash_fidelity_gate.py"
SPEC = importlib.util.spec_from_file_location("fidelity_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class FidelityGateTests(unittest.TestCase):
    def test_positional_mismatches_counts_values_and_length(self) -> None:
        self.assertEqual(gate.positional_mismatches([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(gate.positional_mismatches([1, 2], [1, 2, 3]), 1)

    def test_candidate_requires_all_three_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "three retained samples"):
            gate.validate_candidate(
                serial_tokens=list(range(128)),
                samples=[{"tokens": list(range(128)), "decode_tps": 250.0}],
                max_mismatches=12,
                min_decode_tps=200.0,
            )

    def test_candidate_rejects_thirteenth_mismatch(self) -> None:
        serial = list(range(128))
        candidate = serial.copy()
        candidate[:13] = [value + 1000 for value in candidate[:13]]
        samples = [
            {"tokens": candidate, "decode_tps": 250.0},
            {"tokens": candidate, "decode_tps": 251.0},
            {"tokens": candidate, "decode_tps": 252.0},
        ]
        with self.assertRaisesRegex(ValueError, "13 positional mismatches"):
            gate.validate_candidate(serial, samples, 12, 200.0)

    def test_candidate_rejects_one_slow_sample_even_when_mean_passes(self) -> None:
        serial = list(range(128))
        samples = [
            {"tokens": serial, "decode_tps": 199.0},
            {"tokens": serial, "decode_tps": 250.0},
            {"tokens": serial, "decode_tps": 250.0},
        ]
        with self.assertRaisesRegex(ValueError, "sample 0 decode_tps"):
            gate.validate_candidate(serial, samples, 12, 200.0)

    def test_candidate_returns_per_sample_mismatches_and_mean(self) -> None:
        serial = list(range(128))
        drifted = serial.copy()
        drifted[7] = 999
        result = gate.validate_candidate(
            serial,
            [
                {"tokens": drifted, "decode_tps": 201.0},
                {"tokens": serial, "decode_tps": 202.0},
                {"tokens": serial, "decode_tps": 203.0},
            ],
            12,
            200.0,
        )
        self.assertEqual(result["positional_mismatches"], [1, 0, 0])
        self.assertEqual(result["mean_decode_tps"], 202.0)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the test and verify the intended failure**

Run:

```bash
/opt/homebrew/bin/python3 -m unittest tools/tests/test_gemma4_dflash_fidelity_gate.py
```

Expected: import fails because `tools/gemma4_dflash_fidelity_gate.py` does not exist.

- [x] **Step 3: Implement the receipt gate**

Create `tools/gemma4_dflash_fidelity_gate.py`:

```python
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def positional_mismatches(reference: list[int], candidate: list[int]) -> int:
    shared = sum(left != right for left, right in zip(reference, candidate))
    return shared + abs(len(reference) - len(candidate))


def validate_candidate(
    serial_tokens: list[int],
    samples: list[dict[str, Any]],
    max_mismatches: int,
    min_decode_tps: float,
    required_samples: int = 3,
) -> dict[str, Any]:
    if len(serial_tokens) != 128:
        raise ValueError(f"serial token count {len(serial_tokens)} != 128")
    if len(samples) != required_samples:
        if required_samples == 3:
            raise ValueError(
                f"expected three retained samples, found {len(samples)}")
        raise ValueError(
            f"expected {required_samples} retained samples, found {len(samples)}")
    mismatches: list[int] = []
    rates: list[float] = []
    for index, sample in enumerate(samples):
        tokens = [int(token) for token in sample.get("tokens", [])]
        if len(tokens) != 128:
            raise ValueError(f"sample {index} token count {len(tokens)} != 128")
        mismatch_count = positional_mismatches(serial_tokens, tokens)
        if mismatch_count > max_mismatches:
            raise ValueError(
                f"sample {index} has {mismatch_count} positional mismatches; "
                f"maximum is {max_mismatches}"
            )
        rate = float(sample["decode_tps"])
        if rate < min_decode_tps:
            raise ValueError(
                f"sample {index} decode_tps {rate} < {min_decode_tps}"
            )
        mismatches.append(mismatch_count)
        rates.append(rate)
    mean_rate = statistics.fmean(rates)
    if mean_rate < min_decode_tps:
        raise ValueError(f"mean decode_tps {mean_rate} < {min_decode_tps}")
    return {
        "passed": True,
        "positional_mismatches": mismatches,
        "decode_tps": rates,
        "mean_decode_tps": mean_rate,
    }


def first_tokens(receipt: dict[str, Any]) -> list[int]:
    samples = receipt.get("samples", [])
    if not samples:
        raise ValueError("serial receipt has no retained samples")
    return [int(token) for token in samples[0].get("tokens", [])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-mismatches", type=int, default=12)
    parser.add_argument("--min-decode-tps", type=float, default=200.0)
    parser.add_argument("--required-samples", type=int, default=3)
    args = parser.parse_args()
    serial = json.loads(args.serial.read_text())
    candidate = json.loads(args.candidate.read_text())
    if serial.get("worker_sha256") != candidate.get("worker_sha256"):
        raise ValueError("serial and candidate worker SHA values differ")
    result = validate_candidate(
        first_tokens(serial),
        candidate.get("samples", []),
        args.max_mismatches,
        args.min_decode_tps,
        args.required_samples,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run the tests and verify green**

Run:

```bash
/opt/homebrew/bin/python3 -m unittest tools/tests/test_gemma4_dflash_fidelity_gate.py
```

Expected: five tests pass without importing MLX or touching Metal.

### Task 2: Keep the D15 fidelity lane on certified CBv2 attention

**Files:**
- Modify: `Tests/MLXFastTests/Gemma4DFlashCBv2TargetCacheTests.swift`
- Modify: `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`

**Security flag:** `none`

**Does NOT cover:** D1-D14 behavior, a wide verifier, recurrence proposals, the standard compiled verifier bank, or an in-round fallback. D15 uses one concrete CBv2 transaction selected at construction.

- [ ] **Step 1: Replace the stale ordinary-attention assertion with failing certified-lane tests**

In `Gemma4DFlashCBv2TargetCacheTests.swift`, replace the assertion that the session lacks `setCertifiedMTPRectangularVerification(true)` and add:

```swift
@Test func d15RetainsCBv2CacheAndInstallsSerializedAttentionOnce() throws {
    let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(
        contentsOf: repositoryRoot.appendingPathComponent(
            "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
        encoding: .utf8)

    #expect(source.contains("certifiedRectangularVerification: depth == 15"))
    #expect(source.contains("bank.setCertifiedMTPRectangularVerification(true)"))
    #expect(source.contains("runD15ProposalRounds(\n                targetN: targetN,\n                targetCacheTransaction: prefillTargetCacheTransaction)"))
    #expect(!source.contains("try? runD15ProposalRounds"))
}

@Test func standardDFlashStillUsesThePersistentCompiledBank() throws {
    let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(
        contentsOf: repositoryRoot.appendingPathComponent(
            "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
        encoding: .utf8)
    let d15Branch = try #require(source.range(of: "if depth == 15 {"))
    let compiledBranch = try #require(source.range(
        of: "let targetCacheTransaction: Gemma4DFlashCompiledVerifierBank",
        range: d15Branch.upperBound..<source.endIndex))
    #expect(d15Branch.lowerBound < compiledBranch.lowerBound)
    #expect(source.contains("return try runStandardRounds("))
}
```

- [ ] **Step 2: Run the focused test and verify red**

Run:

```bash
swift test --filter Gemma4DFlashCBv2TargetCacheTests
```

Expected: the new source assertions fail because D15 still copies its cache into `CompilableKVCache` and the CBv2 bank never installs certified serialized attention.

- [ ] **Step 3: Make certification a construction property of the CBv2 transaction**

Add this stored property to `Gemma4DFlashCBv2TargetCache`:

```swift
private let certifiedRectangularVerification: Bool
```

Append this defaulted parameter to the existing initializer signature:

```swift
certifiedRectangularVerification: Bool = false
```

Immediately after `let bank = CBv2LayerCacheBank(caches: unboundCaches)`, insert:

```swift
if certifiedRectangularVerification,
    !bank.supportsCertifiedMTPRectangularVerification
{
    backend.release(rowState)
    throw MLXFastError.invalidInput(
        "runtime worker dflash target cache cannot install certified "
            + "rectangular attention for every target layer")
}
```

Assign `self.certifiedRectangularVerification = certifiedRectangularVerification` with the initializer's other stored-property assignments.

After the existing prefill barrier, install the immutable decode mode once:

```swift
func finishPrefill(evaluating outputs: [MLXArray]) {
    eval(outputs + evaluationRoots)
    if certifiedRectangularVerification {
        bank.setCertifiedMTPRectangularVerification(true)
    }
}
```

This branch runs once outside measured generation. `forwardGreedy` continues to call the already-bound target directly and contains no eligibility decision.

- [ ] **Step 4: Retain the D15 CBv2 transaction instead of constructing the compiled-cache copy**

Construct the prefill transaction with:

```swift
let prefillCacheTransaction = try Gemma4DFlashCBv2TargetCache(
    target: target,
    promptTokenCount: seedTokens.count,
    maxLength: seedTokens.count
        + MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens + 1,
    certifiedRectangularVerification: depth == 15)
```

At the start of `run(targetN:)`, before the compiled-bank branch, install the already-prefilled D15 cache and call a concrete D15 loop:

```swift
if depth == 15 {
    self.targetCache = prefillTargetCacheTransaction.cache
    self.prefillTargetCacheTransaction = nil
    return try runD15ProposalRounds(
        targetN: targetN,
        targetCacheTransaction: prefillTargetCacheTransaction)
}
```

Change the D15 loop signature to the concrete transaction:

```swift
private func runD15ProposalRounds(
    targetN: Int,
    targetCacheTransaction: Gemma4DFlashCBv2TargetCache
) throws -> RuntimeWorkerFreeRunResult
```

Leave the existing `Gemma4DFlashCompiledVerifierBank` construction and reuse path below this branch unchanged for D1-D14.

Remove the now-unreachable D15 width insertion from the standard compiled-bank requirements:

```swift
let requiredVerifierColumns = Set(
    2 ... MLXFastConstants.experimentalDFlashMaxBlockSize)
```

The previous `if depth == 15 { requiredVerifierColumns.insert(...) }` block is deleted because D15 has already returned through its concrete CBv2 transaction.

- [ ] **Step 5: Run focused and full CPU-only verification**

Run:

```bash
swift test --filter Gemma4DFlashCBv2TargetCacheTests
swift test --filter Gemma4DFlashForwardTests
swift test
git diff --check
```

Expected: focused suites pass; full-suite output has no new failure. If the known dirty-tree `emitEngineWireFixture` digest pin remains the only failure, record both expected and observed digests and continue without rewriting unrelated fixture state.

- [ ] **Step 6: Build and run the exact C4 fidelity scout under the canonical guard**

Keep the existing policy type but set its selected width to C4 for this gate:

```swift
static let d15Period2VerifierWidth = 4
```

Build without loading a model:

```bash
swift build -c release --product mlxfast-runtime-worker --build-path .build-worker
shasum -a 256 .build-worker/release/mlxfast-runtime-worker
```

Run a fresh serial sample with token capture, then a D15/C4 sample, each through the canonical parent-held guard:

```bash
GEMMA_DFLASH_LABEL=fidelity-current-serial-c4-worker \
GEMMA_BENCH_MODE=serial \
GEMMA_DFLASH_PHYSICAL_WIDTH=1 \
GEMMA_DFLASH_SAMPLE_COUNT=2 \
GEMMA_DFLASH_CAPTURE_TOKENS=1 \
/opt/homebrew/bin/python3 \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 --timeout-seconds 1800 \
  --child-timeout-seconds 1200 -- \
  /opt/homebrew/bin/python3 /private/tmp/gemma_dflash_width4_mean3.py

GEMMA_DFLASH_LABEL=fidelity-c4-scout \
GEMMA_BENCH_MODE=dflash \
GEMMA_DFLASH_DEPTH=15 \
GEMMA_DFLASH_PHYSICAL_WIDTH=4 \
GEMMA_DFLASH_SAMPLE_COUNT=2 \
GEMMA_DFLASH_CAPTURE_TOKENS=1 \
/opt/homebrew/bin/python3 \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 --timeout-seconds 1800 \
  --child-timeout-seconds 1200 -- \
  /opt/homebrew/bin/python3 /private/tmp/gemma_dflash_width4_mean3.py
```

Expected: both guards restore `mtplx-flash-next-optimized-speed`, verify port 8080 healthy and idle, release the lock, and write captured token arrays. Compare the one retained C4 sample directly against serial with a Python invocation of `positional_mismatches`; continue only when the count is no greater than 12. Do not apply the three-sample 200 tok/s gate at C4.

Run the one-sample fidelity check explicitly:

```bash
gemma_c4_serial=/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-current-serial-c4-worker-width1-mean3.json
gemma_c4_candidate=/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-c4-scout-width4-mean3.json
/opt/homebrew/bin/python3 tools/gemma4_dflash_fidelity_gate.py \
  --serial "$gemma_c4_serial" \
  --candidate "$gemma_c4_candidate" \
  --max-mismatches 12 \
  --min-decode-tps 0 \
  --required-samples 1
```

Expected: `passed` is true and the sole positional mismatch count is no greater than 12.

### Task 3: Replace the period-2 special case with generic committed-token recurrence

**Files:**
- Modify: `Tests/MLXFastTests/Gemma4DFlashForwardTests.swift`
- Modify: `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`

**Security flag:** `none`

**Does NOT cover:** Prompt digests, hard-coded token IDs, period 1, periods above 32, ordinary DFlash controls, or accepting a wide rejection without demoting the following round.

- [ ] **Step 1: Replace the structural/period-2 tests with failing generic recurrence tests**

Add:

```swift
@Test func recurrenceRequiresTwoCompleteCopiesAndChoosesShortestPeriod() {
    var policy = Gemma4DFlashRecurrencePolicy(verifierWidth: 16)
    policy.record(committedTokens: [1, 2, 3, 1, 2], accepted: 0, proposed: 3)
    #expect(policy.phase == .exactDFlashC4)
    policy.record(committedTokens: [3], accepted: 0, proposed: 3)
    #expect(policy.phase == .periodicExactWide(cycle: [1, 2, 3]))
    #expect(policy.makeDraftTokens(count: 8) == [1, 2, 3, 1, 2, 3, 1, 2])
}

@Test func recurrenceFindsTheMeasuredPeriod18WithoutEncodingIt() {
    let cycle = Array(100..<118)
    var policy = Gemma4DFlashRecurrencePolicy(verifierWidth: 32)
    policy.record(
        committedTokens: cycle + cycle,
        accepted: 0,
        proposed: 3)
    #expect(policy.phase == .periodicExactWide(cycle: cycle))
}

@Test func recurrenceRejectsPeriodOneAndPeriodsAboveThirtyTwo() {
    var policy = Gemma4DFlashRecurrencePolicy(verifierWidth: 16)
    policy.record(
        committedTokens: Array(repeating: 7, count: 64),
        accepted: 0,
        proposed: 3)
    #expect(policy.phase == .exactDFlashC4)
    let long = Array(0..<33)
    policy.record(committedTokens: long + long, accepted: 0, proposed: 3)
    #expect(policy.phase == .exactDFlashC4)
}

@Test func wideRejectionDemotesNextRoundAndClearsTheCycle() {
    var policy = Gemma4DFlashRecurrencePolicy(verifierWidth: 16)
    policy.record(
        committedTokens: [4, 5, 6, 4, 5, 6],
        accepted: 0,
        proposed: 3)
    #expect(policy.phase == .periodicExactWide(cycle: [4, 5, 6]))
    policy.record(committedTokens: [4, 99], accepted: 1, proposed: 15)
    #expect(policy.phase == .exactDFlashC4)
    #expect(policy.makeDraftTokens(count: 3) == nil)
}

@Test func wideTailKeepsItsConstructionBoundPhysicalWidth() {
    var policy = Gemma4DFlashRecurrencePolicy(verifierWidth: 16)
    policy.record(
        committedTokens: [4, 5, 6, 4, 5, 6],
        accepted: 0,
        proposed: 3)
    #expect(policy.verifierBlockSize(remaining: 1) == 16)
    #expect(policy.verifierBlockSize(remaining: 127) == 16)
}
```

- [ ] **Step 2: Run the focused test and verify red**

Run:

```bash
swift test --filter Gemma4DFlashForwardTests
```

Expected: compilation fails because `Gemma4DFlashRecurrencePolicy` and the new phase cases do not exist.

- [ ] **Step 3: Implement the pure recurrence state**

Replace `Gemma4DFlashProposalPhase` and `Gemma4DFlashProposalPhasePolicy` with:

```swift
enum Gemma4DFlashProposalPhase: Equatable {
    case exactDFlashC4
    case periodicExactWide(cycle: [Int])
}

struct Gemma4DFlashRecurrencePolicy {
    let verifierWidth: Int
    private(set) var phase: Gemma4DFlashProposalPhase = .exactDFlashC4
    private var recentTokens: [Int] = []
    private let maximumPeriod = 32

    func verifierBlockSize(remaining: Int) -> Int {
        switch phase {
        case .exactDFlashC4:
            return Swift.min(4, remaining + 1)
        case .periodicExactWide:
            return verifierWidth
        }
    }

    func makeDraftTokens(count: Int) -> [Int]? {
        guard count > 0 else { return nil }
        guard case .periodicExactWide(let cycle) = phase else { return nil }
        return (0..<count).map { cycle[$0 % cycle.count] }
    }

    mutating func record(
        committedTokens: [Int], accepted: Int, proposed: Int
    ) {
        guard !committedTokens.isEmpty else { return }
        if case .periodicExactWide = phase, accepted < proposed {
            phase = .exactDFlashC4
            recentTokens = Array(committedTokens.suffix(maximumPeriod * 2))
            return
        }
        recentTokens.append(contentsOf: committedTokens)
        if recentTokens.count > maximumPeriod * 2 {
            recentTokens.removeFirst(recentTokens.count - maximumPeriod * 2)
        }
        guard case .exactDFlashC4 = phase else { return }
        let upper = Swift.min(maximumPeriod, recentTokens.count / 2)
        guard upper >= 2 else { return }
        for period in 2...upper {
            let tail = Array(recentTokens.suffix(period * 2))
            if Array(tail[..<period]) == Array(tail[period...]) {
                phase = .periodicExactWide(
                    cycle: Array(tail.suffix(period)))
                return
            }
        }
    }
}
```

- [ ] **Step 4: Wire the two explicit phases into the D15 loop**

Rename the session's stored policy and width declaration at the same edit boundary:

```swift
static let d15FidelityVerifierWidth = 4
private var proposalPhasePolicy: Gemma4DFlashRecurrencePolicy
```

The old `d15Period2VerifierWidth` declaration is removed rather than retained as an alias.

Initialize once:

```swift
self.proposalPhasePolicy = Gemma4DFlashRecurrencePolicy(
    verifierWidth: Self.d15FidelityVerifierWidth)
self.maximumPhysicalVerifierWidth = Self.d15FidelityVerifierWidth
```

Dispatch one preselected route per round:

```swift
let phase = proposalPhasePolicy.phase
let roundBlockSize = proposalPhasePolicy.verifierBlockSize(
    remaining: remaining)
let round: DFlashGreedyRoundResult
switch phase {
case .exactDFlashC4:
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
case .periodicExactWide:
    let proposal = proposalPhasePolicy.makeDraftTokens(
        count: roundBlockSize - 1)!
    round = try runDFlashGreedyProposalRound(
        target: target,
        targetCache: &targetCache,
        bonus: bonus,
        draftTokens: MLXArray(proposal.map(Int32.init))[
            .newAxis, .ellipsis],
        targetLayerIds: drafter.config.targetLayerIds,
        blockSize: roundBlockSize,
        maxEmitCount: remaining,
        targetCacheTransaction: targetCacheTransaction)
}
```

Update the drafter context only after `.exactDFlashC4`, and record with the actual proposal count:

```swift
if case .exactDFlashC4 = phase {
    draftContext = try drafter.projectTargetHidden(round.targetHidden)
}
let proposed = roundBlockSize - 1
proposalPhasePolicy.record(
    committedTokens: round.tokens,
    accepted: round.accepted,
    proposed: proposed)
```

- [ ] **Step 5: Run pure and source-contract tests**

Run:

```bash
swift test --filter Gemma4DFlashForwardTests
swift test --filter Gemma4DFlashCBv2TargetCacheTests
git diff --check
```

Expected: both suites pass; no period-2, structural-C4, or C96 special case remains in the D15 lane.

### Task 4: Install fixed wider verifier contexts with explicit stock projection bindings

**Files:**
- Modify: `Tests/MLXFastTests/Gemma4MTPVerifierRouteTests.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/Gemma4MTPVerifierRoute.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4PrefillGlueV1.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4TextDFlash.swift`

**Security flag:** `none`

**Does NOT cover:** Arbitrary widths, C64/C96, dynamic shape compilation, ordinary DFlash target forwards, changing C2-C4 kernel arithmetic, or falling back from a missing wide binding during decode.

- [ ] **Step 1: Add failing fixed-width route tests**

Add to `Gemma4MTPVerifierRouteTests.swift`:

```swift
@Test func fidelityWidthsUseSerializedAttentionAndExplicitProjectionStrategies() {
    let route = CBv2Gemma4MTPVerifierRoute.production
    for columns in [2, 3, 4] {
        #expect(route.supports(.init(batch: 1, columns: columns)))
        #expect(route.strategy(for: .qkv, columns: columns) != .stockModule)
        #expect(route.attentionStrategy(columns: columns) == .serializedDecode)
    }
    for columns in [8, 16, 32] {
        #expect(route.supports(.init(batch: 1, columns: columns)))
        for projection in [
            CBv2Gemma4MTPVerifierProjection.qkv,
            .attentionOutput, .denseGateUp, .denseDown,
            .expert, .router, .tiedHead,
        ] {
            #expect(route.strategy(for: projection, columns: columns) == .stockModule)
        }
        #expect(route.attentionStrategy(columns: columns) == .serializedDecode)
    }
    for columns in [1, 5, 6, 7, 9, 15, 31, 33, 64, 96] {
        #expect(!route.supports(.init(batch: 1, columns: columns)))
        #expect(route.attentionStrategy(columns: columns) == nil)
    }
    #expect(!route.supports(.init(batch: 2, columns: 8)))
}

@Test func installationPublishesEveryFidelityWidthAtomically() throws {
    let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(
        contentsOf: repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift"),
        encoding: .utf8)
    #expect(source.contains("let shapes = [2, 3, 4, 8, 16, 32].map"))
    #expect(source.contains("case .stockModule:"))
    #expect(source.contains("installedMTPVerifierContexts = contexts"))
}
```

- [ ] **Step 2: Run the route tests and verify red**

Run:

```bash
swift test --filter Gemma4MTPVerifierRouteTests
```

Expected: compilation fails because `.stockModule` does not exist and C8/C16/C32 are rejected.

- [ ] **Step 3: Make the fixed construction route explicit**

In `Gemma4MTPVerifierRoute.swift`, add the strategy case and fixed set:

```swift
public enum CBv2Gemma4MTPVerifierProjectionStrategy: Sendable, Equatable {
    case combined
    case independentB8
    case stockModule
}

public struct CBv2Gemma4MTPVerifierRoute: Sendable {
    public static let certifiedColumns: Set<Int> = [2, 3, 4, 8, 16, 32]
}
```

At the start of `strategy(for:columns:)`, route wider fixed shapes directly:

```swift
guard Self.certifiedColumns.contains(columns) else { return nil }
if columns > 4 { return .stockModule }
```

Change `attentionStrategy(columns:)` and `supports(_:)` to use `certifiedColumns`. Keep `candidateAttentionStrategy`, `gateUpRows`, and `qkvGeometry` restricted to C2-C4 because their custom kernels have not been certified at wider shapes. Change `Gemma4B1MTPFullAttentionGeometry.visibleKeyLengths` to accept the fixed set rather than a numeric range.

- [ ] **Step 4: Extend only the row-generic verifier glue**

In `Gemma4PrefillGlueV1.bindVerifier`, replace the C2-C4 range check with:

```swift
guard CBv2Gemma4MTPVerifierRoute.certifiedColumns.contains(columns),
    epsIn == eps,
    weights.allSatisfy({
        $0.dtype == .bfloat16 && $0.ndim == 1 && $0.dim(0) == axis
    })
else { return nil }
```

The existing output shape `[1, columns, 2816]` and `rows = columns` remain unchanged; no new Metal kernel text or reduction geometry is introduced.

- [ ] **Step 5: Bind stock modules once for C8/C16/C32**

In each private `bindMTPVerifier(columns:)` helper in `Gemma4Text.swift`, read the immutable route strategy before returning a closure. Use the existing proven binder for C2-C4 and the already-validated quantized module for `.stockModule`.

For attention Q/K/V:

```swift
let strategy = CBv2Gemma4MTPVerifierRoute.production.strategy(
    for: .qkv, columns: columns)!

func bindQKV(
    _ layer: Linear?, component: String
) throws -> (MLXArray) -> MLXArray {
    let value = try quantized(layer, component: component)
    switch strategy {
    case .stockModule:
        return { input in value(input) }
    case .combined, .independentB8:
        guard let bound = CBv2AttentionQKVMMA8V1.bindB1Verifier(
            columns: columns, weight: value.weight, scales: value.scales,
            biases: value.biases, groupSize: value.groupSize,
            bits: value.bits, mode: value.mode)
        else {
            throw Gemma4MTPVerifierInstallationError.incompatibleModel(component)
        }
        return bound
    }
}
```

Apply the same switch to dense gate/up/down and router: `.stockModule` returns `{ input in value(input) }`; C2-C4 retain their current binders.

For the experts, add a `.stockModule` closure that reproduces the ordinary branch without rechecking model metadata:

```swift
case .stockModule:
    let switchGLU = switchGLU
    let fuseWeightedUnsort = fuseWeightedUnsort
    return { x, topKIndices, topKWeights in
        let (batch, sequence, hidden) = (x.dim(0), x.dim(1), x.dim(2))
        let topK = topKIndices.dim(-1)
        let y = switchGLU.callAndWeightedReduce(
            x.reshaped(batch * sequence, hidden),
            topKIndices.reshaped(batch * sequence, topK),
            weights: topKWeights.reshaped(batch * sequence, topK),
            fuseSortedReduction: fuseWeightedUnsort,
            isProductionPrefill: false)
        return y.reshaped(batch, sequence, hidden)
    }
```

For the tied head inside `installCBv2MTPVerifier()`:

```swift
let head: (MLXArray) -> MLXArray
switch route.strategy(for: .tiedHead, columns: columns)! {
case .stockModule:
    head = { hidden in embedding.asLinear(hidden) }
case .combined, .independentB8:
    guard let bound = Gemma4MMAQuantizedGEMV.bindB1Verifier(
        columns: columns, inDim: config.hiddenSize,
        outDim: config.vocabSize, w: embedding.weight,
        scales: embedding.scales, biases: embedding.biases,
        groupSize: embedding.groupSize, bits: embedding.bits,
        mode: embedding.mode)
    else {
        throw Gemma4MTPVerifierInstallationError.incompatibleModel(
            "tied language-model head")
    }
    head = bound
}
```

Change the installed shape list to:

```swift
let shapes = [2, 3, 4, 8, 16, 32].map {
    CBv2Gemma4MTPVerifierShape(batch: 1, columns: $0)
}
```

The local `contexts` dictionary remains unpublished until every layer and head binding succeeds.

- [ ] **Step 6: Require a context in the fidelity direct forward**

Add to `Gemma4TextDFlash.swift`:

```swift
public func bindCertifiedDFlashGreedyForward(
    cache: [KVCache],
    targetLayerIds: [Int],
    columns: Int
) throws -> Gemma4DFlashGreedyForwardBinding {
    try DFlashTargetValidation.validateTargetLayerIds(
        targetLayerIds, layerCount: configuration.numHiddenLayers)
    guard let verifier = cbv2MTPVerifierContext(batch: 1, columns: columns)
    else {
        preconditionFailure(
            "certified DFlash verifier has no installed B1/C\(columns) route")
    }
    precondition(
        cache.count == configuration.numHiddenLayers
            && cache.allSatisfy { $0 is any CBv2AttendingLayerCache },
        "certified DFlash verifier requires one CBv2 cache per target layer")
    let hiddenStateCount = targetLayerIds.count
    return Gemma4DFlashGreedyForwardBinding(
        columns: columns,
        hiddenStateCount: hiddenStateCount,
        body: { [self] input in
            CBv2OrderOnlyLogits.withGreedyOrderOnly {
                let forward = model.callCapturingValidatedDFlashHiddenStates(
                    input,
                    cache: cache,
                    targetLayerIds: targetLayerIds,
                    forceArrayMask: false,
                    verifier: verifier)
                let tokens = applyLMHead(
                    forward.postNorm, verifier: verifier
                ).argMax(axis: -1)
                return [tokens] + forward.hiddenStates
            }
        })
}
```

Have `Gemma4DFlashCBv2TargetCache` receive `requiredVerifierColumns` and `targetLayerIds` at construction. Add these parameters to its existing initializer signature and insert the binding loop immediately after the existing `let cache: [KVCache] = ...` conversion succeeds:

```swift
private let forwards: [Int: Gemma4DFlashGreedyForwardBinding]

var forwards: [Int: Gemma4DFlashGreedyForwardBinding] = [:]
for columns in requiredVerifierColumns.sorted() {
    forwards[columns] = try target.bindCertifiedDFlashGreedyForward(
        cache: cache,
        targetLayerIds: targetLayerIds,
        columns: columns)
}
self.forwards = forwards
```

The complete added initializer parameters have defaults so existing CPU/runtime cache tests remain source-compatible:

```swift
requiredVerifierColumns: Set<Int> = [],
targetLayerIds: [Int] = []
```

Update the session construction call to pass the construction-owned D15 set explicitly:

```swift
requiredVerifierColumns:
    depth == 15 ? Set([4, Self.d15FidelityVerifierWidth]) : [],
targetLayerIds: drafter.config.targetLayerIds
```

`forwardGreedy` performs only the genuine runtime column lookup and invokes the prebound closure:

```swift
let columns = verifyInput.dim(1)
guard let forward = forwards[columns] else {
    preconditionFailure(
        "certified DFlash cache has no installed C\(columns) forward")
}
return forward(verifyInput)
```

Missing bindings fail during construction; the measured D15 path does not call an ordinary target forward. Standard sessions never call this transaction's `forwardGreedy` because Task 2 routes them to `Gemma4DFlashCompiledVerifierBank`.

- [ ] **Step 7: Run construction and source-contract verification**

Run:

```bash
swift test --filter Gemma4MTPVerifierRouteTests
swift test --filter Gemma4DFlashCBv2TargetCacheTests
swift test --filter Gemma4DFlashForwardTests
swift test
git diff --check
```

Expected: fixed-width route tests pass, C2-C4 tests remain unchanged, D15 source tests prove no ordinary or compiled fallback, and full-suite output introduces no new failure.

### Task 5: Evaluate C8, C16, and C32 in order and retain the first 200 tok/s winner

**Files:**
- Modify iteratively: `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`
- Generate: `.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-current-serial-*.json`
- Generate: `.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-c8-scout-*.json`
- Generate conditionally: corresponding C16 and C32 scout receipts
- Generate: final retained warmup-plus-three receipt

**Security flag:** `none`

**Does NOT cover:** C64/C96, 64K/128K context, batch greater than one, concurrency greater than one, MTP, prefill optimization, commits, pushes, or production configuration changes.

- [ ] **Step 1: Establish memory headroom before each full-model width**

Use the measured same-model C96 receipt as the empirical upper bound:

```bash
jq '{peak:.final_diagnostics.mlx_peak_memory_bytes,active:.final_diagnostics.mlx_active_memory_bytes,cache:.final_diagnostics.mlx_cache_memory_bytes,ram:.final_diagnostics.peak_ram_gb}' \
  /Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/structural-c96-final-width96-mean3.json
memory_pressure
vm_stat
```

Expected bound: C96 peak is exactly 24,474,883,100 MLX bytes. The new largest width is C32, uses the same target/checkpoint/cache length, retains one CBv2 cache stack instead of the prior CBv2-plus-compiled transition, and has no position-local tensor wider than the previously completed C96 run. Do not run if current wired plus 24,474,883,100 bytes and the production restoration reserve do not fit with safe headroom.

- [ ] **Step 2: Build and run one guarded C8 scout**

Set only:

```swift
static let d15FidelityVerifierWidth = 8
```

Build and record the worker SHA:

```bash
swift build -c release --product mlxfast-runtime-worker --build-path .build-worker
shasum -a 256 .build-worker/release/mlxfast-runtime-worker
```

Run one warmup plus one retained C8 sample with `GEMMA_DFLASH_CAPTURE_TOKENS=1`, label `fidelity-c8-scout`, depth 15, and expected physical width 8 through the canonical guard command used in Task 2. Run a fresh one-sample serial receipt from the same worker SHA. Expected: no more than 12 mismatches, 128 committed tokens, safe peak, production restored, and lock released.

Apply the one-sample gate:

```bash
gemma_c8_serial=/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-c8-serial-width1-mean3.json
gemma_c8_candidate=/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-c8-scout-width8-mean3.json
/opt/homebrew/bin/python3 tools/gemma4_dflash_fidelity_gate.py \
  --serial "$gemma_c8_serial" --candidate "$gemma_c8_candidate" \
  --max-mismatches 12 --min-decode-tps 0 --required-samples 1
```

- [ ] **Step 3: Retain C8 or advance from evidence**

If C8 is within the fidelity budget and its scout is at least 200 decode tok/s, skip to Step 6 with retained width 8. If it is within budget but below 200, inspect `decode_s`, rounds, acceptance lengths, drafted/accepted totals, and peak memory, then continue to C16. If it exceeds 12 mismatches, stop widening and return to the first mismatching layer/operation diagnostic; C16 cannot repair an already-invalid C8 arithmetic route.

- [ ] **Step 4: Build and run C16 only when C8 is correct but below 200**

Change only the constant to 16, rebuild, run a fresh serial sample from that worker, and run one warmup plus one retained candidate using label `fidelity-c16-scout` and expected physical width 16. Apply the same 12/128 fidelity, 128-token, memory, service, and lock gates. Retain C16 if its scout reaches 200; otherwise continue only when it remains within the fidelity budget.

Apply the one-sample gate:

```bash
gemma_c16_serial=/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-c16-serial-width1-mean3.json
gemma_c16_candidate=/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-c16-scout-width16-mean3.json
/opt/homebrew/bin/python3 tools/gemma4_dflash_fidelity_gate.py \
  --serial "$gemma_c16_serial" --candidate "$gemma_c16_candidate" \
  --max-mismatches 12 --min-decode-tps 0 --required-samples 1
```

- [ ] **Step 5: Build and run C32 only when C16 is correct but below 200**

Change only the constant to 32, rebuild, run a fresh serial sample from that worker, and run one warmup plus one retained candidate using label `fidelity-c32-scout` and expected physical width 32. Apply the same gates. If correct C32 remains below 200, record it as a measured miss; do not substitute ordinary C96 or claim completion.

Apply the one-sample gate:

```bash
gemma_c32_serial=/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-c32-serial-width1-mean3.json
gemma_c32_candidate=/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-c32-scout-width32-mean3.json
/opt/homebrew/bin/python3 tools/gemma4_dflash_fidelity_gate.py \
  --serial "$gemma_c32_serial" --candidate "$gemma_c32_candidate" \
  --max-mismatches 12 --min-decode-tps 0 --required-samples 1
```

- [ ] **Step 6: Run the retained width as one warmup plus three measured samples**

With the retained source constant and release worker, assign the measured winner to a task-specific shell variable and run:

```bash
gemma_retained_width=8  # use 16 or 32 when that measured width won
GEMMA_DFLASH_LABEL=fidelity-final-200 \
GEMMA_BENCH_MODE=dflash \
GEMMA_DFLASH_DEPTH=15 \
GEMMA_DFLASH_PHYSICAL_WIDTH="$gemma_retained_width" \
GEMMA_DFLASH_SAMPLE_COUNT=4 \
GEMMA_DFLASH_CAPTURE_TOKENS=1 \
/opt/homebrew/bin/python3 \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 --timeout-seconds 1800 \
  --child-timeout-seconds 1200 -- \
  /opt/homebrew/bin/python3 /private/tmp/gemma_dflash_width4_mean3.py
```

The assignment is changed to 16 or 32 only when that width won its preceding scout; it is not committed to a script or read by the hot path.

Run a fresh serial warmup plus one retained token-capture receipt from the identical worker SHA, then apply:

```bash
gemma_serial_receipt=/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-final-serial-width1-mean3.json
gemma_candidate_receipt=/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/gemma4-swift-dflash-reference-ab/fidelity-final-200-width${gemma_retained_width}-mean3.json
/opt/homebrew/bin/python3 tools/gemma4_dflash_fidelity_gate.py \
  --serial "$gemma_serial_receipt" \
  --candidate "$gemma_candidate_receipt" \
  --max-mismatches 12 \
  --min-decode-tps 200
```

The two paths are first checked against the `out` values printed by the just-completed guard children. Expected JSON: `passed: true`, three mismatch counts each no greater than 12, three decode rates each at least 200, and mean at least 200.

### Task 6: Perform the completion audit and preserve only admissible claims

**Files:**
- Modify if evidence requires correction: `docs/benchmarks/gemma4-dflash-depth-sweep-2026-08-30.md`
- Generate: final benchmark receipt and guard log from Task 5

**Security flag:** `none`

**Does NOT cover:** Publishing, committing, pushing, pull requests, generalizing the result beyond the named workload, or rewriting raw failed receipts.

- [ ] **Step 1: Run fresh software verification**

Run:

```bash
/opt/homebrew/bin/python3 -m unittest tools/tests/test_gemma4_dflash_fidelity_gate.py
swift test --filter Gemma4DFlashForwardTests
swift test --filter Gemma4MTPVerifierRouteTests
swift test --filter Gemma4DFlashCBv2TargetCacheTests
swift test
git diff --check
rg -n "period2Wide|structuralC4|d15Period2VerifierWidth|= 96|try\? .*DFlash|eligible.*stock|fallback" \
  Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift \
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4TextDFlash.swift
```

Expected: focused suites and Python tests pass; no stale invalid D15 phase or C96 special case remains; full-suite output has no new failure; diff check is clean.

- [ ] **Step 2: Audit every final receipt field**

Use `jq` to prove:

```text
prompt_tokens == 1024
mode == "dflash"
draft_depth == 15
physical_verify_width in [8, 16, 32]
samples.length == 3
all samples committed == 128
all samples contain exactly 128 captured tokens
all samples decode_tps >= 200.0
mean.decode_tps >= 200.0
serial worker_sha256 == candidate worker_sha256
all positional mismatch counts <= 12
```

Also record each sample's prefill seconds/TPS and decode seconds/TPS separately. Do not combine them into one throughput number.

- [ ] **Step 3: Audit the GPU/service lifecycle**

Inspect the final guard log and current state. Required evidence:

```text
the guard acquired /tmp/mtplx-gpu-exclusive.lock before unload
only the owned benchmark child ran under that lock
mtplx-flash-next-optimized-speed was restored on port 8080
health check passed
active requests == 0
foreground requests == 0
background warmup completed
no candidate worker remains
the exclusive lock is free
memory pressure is healthy
```

If any item is missing, completion remains unproven and the guard/service check is rerun before reporting.

- [ ] **Step 4: Correct the benchmark narrative without erasing failed evidence**

Append a dated correction to `docs/benchmarks/gemma4-dflash-depth-sweep-2026-08-30.md` that distinguishes:

```text
113.15 tok/s current serial control: fidelity-valid
354.61 tok/s ordinary C96 timing: 108/128 mismatches, rejected
retained fidelity-gated result: exact three sample rates, mean, mismatch counts,
physical width, worker SHA, and receipt path
```

Do not delete or rewrite the raw C96 receipt. If no C8/C16/C32 configuration satisfies every gate, document the measured correct ceiling and leave the 200 tok/s goal open.

## Execution Choice

Use **Subagent-Driven Development**. There are six tasks, and the implementation skill requires a fresh worker plus two-stage review per task. The tasks remain sequential because they share one dirty worktree and the GPU width ladder depends on the preceding correctness result; no two agents may run MLX/Metal concurrently.
