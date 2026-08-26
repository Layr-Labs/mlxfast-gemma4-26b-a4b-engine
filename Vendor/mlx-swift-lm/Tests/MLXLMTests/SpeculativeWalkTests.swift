// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import Testing

@Suite("Gemma4SpeculativeWalk.single")
struct SpeculativeWalkSingleTests {

    @Test func emptyDraftReturnsJustMain() {
        let (accepted, emitted) = Gemma4SpeculativeWalk.single(
            draft: [], main: [7]
        )
        #expect(accepted == 0)
        #expect(emitted == [7])
    }

    @Test func allDraftsMatch() {
        let (accepted, emitted) = Gemma4SpeculativeWalk.single(
            draft: [1, 2, 3], main: [1, 2, 3, 4]
        )
        #expect(accepted == 3)
        #expect(emitted == [1, 2, 3, 4])
    }

    @Test func firstMismatchAtZero() {
        let (accepted, emitted) = Gemma4SpeculativeWalk.single(
            draft: [9, 2, 3], main: [1, 2, 3, 4]
        )
        #expect(accepted == 0)
        #expect(emitted == [1])
    }

    @Test func firstMismatchInMiddle() {
        let (accepted, emitted) = Gemma4SpeculativeWalk.single(
            draft: [1, 2, 99], main: [1, 2, 3, 4]
        )
        #expect(accepted == 2)
        #expect(emitted == [1, 2, 3])
    }

    @Test func emittedCountEqualsAcceptedPlusOne() {
        // Invariant: emitted.count == accepted + 1 for any non-trivial input.
        let draft = [1, 2, 3, 4, 5]
        let main = [1, 2, 99, 4, 5, 6]
        let (accepted, emitted) = Gemma4SpeculativeWalk.single(draft: draft, main: main)
        #expect(emitted.count == accepted + 1)
    }
}

@Suite("Gemma4SpeculativeWalk.batched")
struct SpeculativeWalkBatchedTests {

    @Test func perRowEquivalenceToSingle() {
        // B=2, k=3. Row 0 fully accepts; row 1 mismatches at position 1.
        let draft: [[Int]] = [
            [1, 2, 3],
            [4, 99, 6],
        ]
        let main: [[Int]] = [
            [1, 2, 3, 42],
            [4, 5, 6, 7],
        ]
        let budgets = [Int.max, Int.max]

        let (batchedAccepted, batchedNew) =
            Gemma4SpeculativeWalk.batched(draft: draft, main: main, budgets: budgets)

        // Expected per-row via single(...)
        var expectedAccepted: [Int] = []
        var expectedNew: [[Int]] = []
        for i in 0 ..< draft.count {
            let (a, n) = Gemma4SpeculativeWalk.single(draft: draft[i], main: main[i])
            expectedAccepted.append(a)
            expectedNew.append(n)
        }
        #expect(batchedAccepted == expectedAccepted)
        #expect(batchedNew == expectedNew)
    }

    @Test func budgetTruncates() {
        // Row accepts 3 and would emit 4 tokens, but budget = 2 truncates.
        let draft: [[Int]] = [[1, 2, 3]]
        let main: [[Int]] = [[1, 2, 3, 42]]
        let (accepted, emitted) =
            Gemma4SpeculativeWalk.batched(draft: draft, main: main, budgets: [2])
        // Emission is capped to 2 tokens; accepted is also capped so
        // the invariant emitted.count == accepted + 1 still holds.
        #expect(emitted[0].count == 2)
        #expect(accepted[0] == 1)
    }
}
