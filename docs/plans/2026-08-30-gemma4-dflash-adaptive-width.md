# Gemma 4 DFlash Adaptive Width Implementation Plan

> **For agentic workers:** Execute inline because the implementation and exact
> benchmark share one stateful worktree and GPU lane. Do not commit or push
> unless the user requests it.

**Goal:** Clear 150 decode tok/s on the exact 1K/128 average-of-three DFlash workload without changing the retained token digest.

**Architecture:** Add a pure session-owned width policy. A requested D7 session starts at D3, promotes after two fully accepted C4 rounds, and demotes after a partial full-width C8 round. The existing round function, acceptance arithmetic, verifier routing, and cache rollback remain unchanged.

**Tech Stack:** Swift 6, Swift Testing, MLX Swift, guarded Python benchmark harness.

**Assumptions:** The measured late-prefix full acceptance recurs in one continuous session — the exact benchmark is the arbiter. Assumes the D7 drafter ceiling is installed — it will not activate for any requested depth other than D7. Assumes the retained C4 token chain remains the fidelity reference — a different digest is a failed candidate even if faster.

---

## File structure

- `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`: pure width-policy state and its single call site in the DFlash loop.
- `Tests/MLXFastTests/Gemma4DFlashForwardTests.swift`: deterministic policy transition tests.
- `.benchmark-artifacts/gemma4-swift-dflash-reference-ab/`: ignored benchmark receipts proving the exact result.

### Task 1: Add the policy with a red-green cycle

**Files:**
- Modify: `Tests/MLXFastTests/Gemma4DFlashForwardTests.swift`
- Modify: `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`

**Security flag:** none

**Does NOT cover:** Requested depths other than D7, batched DFlash, prefill, sampling, verifier arithmetic, and cache rollback remain unchanged.

- [ ] **Step 1: Write failing policy tests**

```swift
@Test func d7WidthPolicyPromotesAfterTwoFullC4Rounds() {
    var policy = Gemma4DFlashWidthPolicy(requestedDepth: 7)
    #expect(policy.currentDepth == 3)
    policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 4)
    #expect(policy.currentDepth == 3)
    policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 4)
    #expect(policy.currentDepth == 7)
}

@Test func d7WidthPolicyResetsAndDemotesWithoutTreatingTailAsFailure() {
    var policy = Gemma4DFlashWidthPolicy(requestedDepth: 7)
    policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 4)
    policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 2)
    policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 4)
    #expect(policy.currentDepth == 3)
    policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 4)
    #expect(policy.currentDepth == 7)
    policy.record(roundBlockSize: 7, maxEmitCount: 6, committed: 6)
    #expect(policy.currentDepth == 7)
    policy.record(roundBlockSize: 8, maxEmitCount: 7, committed: 7)
    #expect(policy.currentDepth == 7)
    policy.record(roundBlockSize: 8, maxEmitCount: 8, committed: 3)
    #expect(policy.currentDepth == 3)
}

@Test func nonD7WidthPoliciesRemainFixed() {
    for depth in [1, 2, 3, 4, 5, 6, 8, 11] {
        var policy = Gemma4DFlashWidthPolicy(requestedDepth: depth)
        policy.record(
            roundBlockSize: depth + 1,
            maxEmitCount: depth + 1,
            committed: depth + 1)
        policy.record(
            roundBlockSize: depth + 1,
            maxEmitCount: depth + 1,
            committed: 1)
        #expect(policy.currentDepth == depth)
    }
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
swift test --filter Gemma4DFlashForwardTests
```

Expected: compilation fails because `Gemma4DFlashWidthPolicy` does not exist.

- [ ] **Step 3: Implement the pure policy and wire it once**

```swift
struct Gemma4DFlashWidthPolicy {
    let requestedDepth: Int
    private(set) var currentDepth: Int
    private var consecutiveFullC4Rounds = 0

    init(requestedDepth: Int) {
        self.requestedDepth = requestedDepth
        currentDepth = requestedDepth == 7 ? 3 : requestedDepth
    }

    mutating func record(
        roundBlockSize: Int, maxEmitCount: Int, committed: Int
    ) {
        let fullWidth = currentDepth + 1
        guard requestedDepth == 7,
            roundBlockSize == fullWidth,
            maxEmitCount >= fullWidth
        else { return }
        let full = committed == roundBlockSize
        if currentDepth == 3 {
            consecutiveFullC4Rounds = full ? consecutiveFullC4Rounds + 1 : 0
            if consecutiveFullC4Rounds == 2 {
                currentDepth = 7
                consecutiveFullC4Rounds = 0
            }
        } else if !full {
            currentDepth = 3
            consecutiveFullC4Rounds = 0
        }
    }
}
```

Construct the policy in `RuntimeWorkerDFlashFreeRunSession.init`, calculate each
round from `widthPolicy.currentDepth + 1`, and call
`widthPolicy.record(roundBlockSize: roundBlockSize, maxEmitCount: remaining,
committed: round.tokens.count)`
after the round succeeds.

- [ ] **Step 4: Verify GREEN and relevant regressions**

Run:

```bash
swift test --filter Gemma4DFlashForwardTests
swift test --filter RuntimeWorker
git diff --check
```

Expected: all selected tests pass and `git diff --check` is silent.

### Task 2: Build and prove the exact performance result

**Files:**
- Generated: `.benchmark-artifacts/gemma4-swift-dflash-reference-ab/adaptive-c4-c8-exact-width8-mean3.json`

**Security flag:** none

**Does NOT cover:** Other prompts, output lengths, context lengths, batched execution, or requested depths. The named objective is the exact 1K/128 single-prompt workload.

- [ ] **Step 1: Build the release worker**

Run:

```bash
swift build -c release --force-resolved-versions --scratch-path .build-worker --product mlxfast-runtime-worker
shasum -a 256 .build-worker/release/mlxfast-runtime-worker
```

Expected: release build succeeds and prints the candidate worker SHA-256.

- [ ] **Step 2: Run the guarded exact benchmark**

Run from the workspace root:

```bash
/opt/homebrew/bin/python3 bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 --timeout-seconds 1800 \
  --child-timeout-seconds 1200 -- \
  /usr/bin/env GEMMA_DFLASH_DEPTH=7 \
  GEMMA_DFLASH_LABEL=adaptive-c4-c8-exact \
  /opt/homebrew/bin/python3 /private/tmp/gemma_dflash_width4_mean3.py
```

Expected: one warmup plus three measured samples complete under the lock.

- [ ] **Step 3: Audit the receipt and postflight**

Run:

```bash
jq '{mean,samples:[.samples[]|{decode_tps,token_digest,committed,acceptance_lengths}]}' \
  .benchmark-artifacts/gemma4-swift-dflash-reference-ab/adaptive-c4-c8-exact-width8-mean3.json
curl -fsS http://127.0.0.1:8080/health
lsof /tmp/mtplx-gpu-exclusive.lock
```

Expected: mean decode TPS is greater than 150; all three digests equal
`704585a3a78c96932198c35e98e9c9e9018e16951e5ccb40a9bd7f3b4941c37d`;
each sample commits 128 tokens and includes acceptance lengths greater than
four; health names `mtplx-flash-next-optimized-speed`; `lsof` prints no owner.

If the exact mean is not greater than 150, retain the receipt as a rejected
candidate and return to the measured round-cost profile. Do not weaken the
target or report the reconstructed-prefix probe as completion.
