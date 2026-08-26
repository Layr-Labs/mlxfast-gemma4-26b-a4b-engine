// Copyright © 2026 Eigen Labs.
//
// MTP speculation hooks in SchedulerV2: pure bookkeeping tests (no MLX
// arrays, no weights, no engine thread). The engine loop sets
// `speculationPlanner` to plan 1+k tokens for decode-ready rows, marks 1+k
// pending samples, and at finalize confirms e via `recordSampled`, drops the
// rest via `discardPendingSamples`, and rejects the executed suffix via
// `rollbackComputed`. The in-flight invariant: with P = tokens.count before
// the round, plan+mark leaves effectiveTokenCount = P+1+k and
// numComputedTokens = P+k, so remainingTokens stays 1 (decode-ready); the
// finalize sequence lands the row back on the standard decode invariant.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Scripted capacity oracle that records every call and can be told to
/// throw for specific `additionalTokens` values (or everything).
private final class SpecScriptedCapacity: CBv2StepCapacity {
    var failTokenCounts: Set<Int> = []
    var failAll = false
    private(set) var reserveCalls: [(id: CBv2RequestID, tokens: Int)] = []
    private(set) var unreserveCalls: [(id: CBv2RequestID, tokens: Int)] = []
    private(set) var releaseAllCalls: [CBv2RequestID] = []
    private(set) var reserved: [CBv2RequestID: Int] = [:]

    func reserve(id: CBv2RequestID, additionalTokens: Int) throws {
        reserveCalls.append((id: id, tokens: additionalTokens))
        if failAll || failTokenCounts.contains(additionalTokens) {
            throw CBv2KVError.capacityExhausted(needed: additionalTokens, available: 0)
        }
        reserved[id, default: 0] += additionalTokens
    }

    func unreserve(id: CBv2RequestID, tokens: Int) {
        unreserveCalls.append((id: id, tokens: tokens))
        let value = max(0, (reserved[id] ?? 0) - tokens)
        if value == 0 { reserved.removeValue(forKey: id) } else { reserved[id] = value }
    }

    func releaseAll(id: CBv2RequestID) {
        releaseAllCalls.append(id)
        reserved.removeValue(forKey: id)
    }

    func hasHeadroom(additionalTokens: Int) -> Bool { true }
}

final class CBv2SchedulerSpeculationTests: XCTestCase {

    private func makeScheduler(
        maxConcurrent: Int = 4, budget: Int = 2048, chunk: Int = 512, maxWaiting: Int = 64,
        capacity: CBv2StepCapacity? = nil
    ) -> SchedulerV2 {
        SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent, maxBatchedTokensPerStep: budget,
                prefillChunkSize: chunk, maxWaiting: maxWaiting),
            capacity: capacity)
    }

    /// Enqueue + prefill + confirm the first sample: the row is decode-ready.
    @discardableResult
    private func decodeReadyRow(
        _ scheduler: SchedulerV2, prompt: [Int], maxTokens: Int = 32
    ) throws -> CBv2Request {
        let request = CBv2SchedFixtures.request(prompt: prompt, maxTokens: maxTokens)
        try scheduler.enqueue(request)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
        XCTAssertEqual(scheduler.record(for: request.id)?.isDecodeReady, true)
        return request
    }

    // MARK: Planner eligibility

    func testPlannerAppliesOnlyToDecodeReadyRows() throws {
        let scheduler = makeScheduler(budget: 100, chunk: 50)
        var plannerCalls: [CBv2RequestID] = []
        scheduler.speculationPlanner = { rec in
            plannerCalls.append(rec.id)
            return 3
        }

        let decode = try decodeReadyRow(scheduler, prompt: Array(0 ..< 5), maxTokens: 10)
        XCTAssertTrue(plannerCalls.isEmpty, "waiting admission never consults the planner")

        let prefill = CBv2SchedFixtures.request(prompt: Array(0 ..< 80), maxTokens: 10)
        try scheduler.enqueue(prefill)

        // Decode row gets 1+k; the prefilling row gets its chunk.
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.id), [decode.id, prefill.id])
        XCTAssertEqual(plan.assignments.map(\.numTokens), [4, 50])
        XCTAssertEqual(plannerCalls, [decode.id])
        scheduler.markPendingSamples(counts: [(id: decode.id, count: 4)])

        // Mid-prefill row keeps chunking; the planner still only sees decode.
        let plan2 = scheduler.plan()
        XCTAssertEqual(plan2.assignments.map(\.id), [decode.id, prefill.id])
        XCTAssertEqual(plan2.assignments.map(\.numTokens), [4, 30])
        XCTAssertEqual(plannerCalls, [decode.id, decode.id])
    }

    func testPlannerReturningZeroOrNegativeYieldsPlainDecode() throws {
        let scheduler = makeScheduler()
        let request = try decodeReadyRow(scheduler, prompt: [1, 2, 3])

        scheduler.speculationPlanner = { _ in 0 }
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [1])
        scheduler.markPendingSamples(counts: [(id: request.id, count: 1)])
        scheduler.recordSampled(id: request.id, token: 9)

        scheduler.speculationPlanner = { _ in -2 }
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [1])
    }

    func testPausedAndCancelledRowsNeverConsultPlanner() throws {
        let scheduler = makeScheduler()
        var plannerCalls: [CBv2RequestID] = []
        let paused = try decodeReadyRow(scheduler, prompt: [1], maxTokens: 10)
        let cancelled = try decodeReadyRow(scheduler, prompt: [2], maxTokens: 10)

        scheduler.speculationPlanner = { rec in
            plannerCalls.append(rec.id)
            return 3
        }
        scheduler.pause(paused.id)
        scheduler.requestCancel(cancelled.id)

        let plan = scheduler.plan()
        XCTAssertTrue(plan.assignments.isEmpty)
        XCTAssertTrue(plannerCalls.isEmpty)
    }

    func testDeferredBlockRowNeverConsultsPlanner() throws {
        let scheduler = makeScheduler(budget: 10, chunk: 10)
        let decode = try decodeReadyRow(scheduler, prompt: [1, 2], maxTokens: 10)

        // Vision request whose leading block (9 tokens) fits a FULL budget
        // but not what the decode row leaves over this step.
        let vision = CBv2Request(
            id: CBv2RequestID(990_001), promptTokens: Array(repeating: 7, count: 12),
            maxTokens: 4,
            multimodal: CBv2MultimodalInput(
                spans: [CBv2ImageSpan(tokenOffset: 0, length: 9)], embeddings: { [] }))
        try scheduler.enqueue(vision)

        var plannerCalls: [CBv2RequestID] = []
        scheduler.speculationPlanner = { rec in
            plannerCalls.append(rec.id)
            return 3
        }

        // Step 1: decode gets 1+3 = 4; the block (9) cannot fit the
        // remaining 6 ⇒ deferred, not admitted.
        let plan1 = scheduler.plan()
        XCTAssertEqual(plan1.assignments.map(\.id), [decode.id])
        XCTAssertEqual(plan1.assignments.map(\.numTokens), [4])
        XCTAssertEqual(scheduler.waitingCount, 1)
        scheduler.markPendingSamples(counts: [(id: decode.id, count: 4)])

        // Step 2: the deferred block row claims the fresh budget first; the
        // planner is never consulted for it.
        let plan2 = scheduler.plan()
        XCTAssertEqual(plan2.assignments.map(\.id), [vision.id])
        XCTAssertEqual(plan2.assignments.map(\.numTokens), [10])
        XCTAssertEqual(plannerCalls, [decode.id])
    }

    // MARK: Budget clamp

    func testBudgetClampFallsBackToPlainDecode() throws {
        let scheduler = makeScheduler(budget: 2, chunk: 2)
        let request = try decodeReadyRow(scheduler, prompt: [1, 2], maxTokens: 8)

        scheduler.speculationPlanner = { _ in 3 }  // 1+3 = 4 > budget 2 ⇒ 1
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.numTokens), [1])
        XCTAssertEqual(plan.speculationFallbacks[request.id], .tokenBudget)
        XCTAssertEqual(scheduler.record(for: request.id)?.numComputedTokens, 3)
    }

    func testSpeculativeAssignmentDebitsBudget() throws {
        let scheduler = makeScheduler(budget: 10, chunk: 512)
        let decode = try decodeReadyRow(scheduler, prompt: [1, 2], maxTokens: 8)
        let waiting = CBv2SchedFixtures.request(
            prompt: Array(repeating: 5, count: 20), maxTokens: 4)
        try scheduler.enqueue(waiting)

        scheduler.speculationPlanner = { _ in 3 }
        // Decode takes 1+3 = 4 of 10; the waiting prompt gets the 6 left.
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.id), [decode.id, waiting.id])
        XCTAssertEqual(plan.assignments.map(\.numTokens), [4, 6])
    }

    // MARK: Capacity retry & backstop

    func testCapacityRetryShrinksToPlainDecodeWithoutPreemption() throws {
        let capacity = SpecScriptedCapacity()
        capacity.failTokenCounts = [4]
        let scheduler = makeScheduler(capacity: capacity)
        let request = try decodeReadyRow(scheduler, prompt: Array(0 ..< 10), maxTokens: 8)

        scheduler.speculationPlanner = { _ in 3 }
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.numTokens), [1])
        XCTAssertTrue(plan.preemptions.isEmpty, "speculative slack must never preempt")
        XCTAssertEqual(plan.speculationFallbacks[request.id], .kvHeadroom)
        XCTAssertEqual(capacity.reserveCalls.map(\.tokens), [10, 4, 1])
        XCTAssertTrue(capacity.releaseAllCalls.isEmpty)
        XCTAssertEqual(scheduler.record(for: request.id)?.numComputedTokens, 11)
        XCTAssertEqual(scheduler.record(for: request.id)?.status, .running)
    }

    func testCapacityHardFailMatchesPlainDecodePreemption() throws {
        func runScenario(planner: ((CBv2ScheduledRequest) -> Int)?) throws
            -> (plan: CBv2StepPlan, scheduler: SchedulerV2, capacity: SpecScriptedCapacity, id: CBv2RequestID)
        {
            let capacity = SpecScriptedCapacity()
            let scheduler = makeScheduler(capacity: capacity)
            let request = try decodeReadyRow(scheduler, prompt: Array(0 ..< 10), maxTokens: 8)
            capacity.failAll = true
            scheduler.speculationPlanner = planner
            return (scheduler.plan(), scheduler, capacity, request.id)
        }

        let spec = try runScenario(planner: { _ in 3 })
        let plain = try runScenario(planner: nil)

        // Same backstop either way: self-preempt (only row), stop, requeue.
        XCTAssertEqual(spec.plan.preemptions, [spec.id])
        XCTAssertEqual(plain.plan.preemptions, [plain.id])
        XCTAssertTrue(spec.plan.assignments.isEmpty)
        XCTAssertTrue(plain.plan.assignments.isEmpty)
        XCTAssertEqual(spec.scheduler.record(for: spec.id)?.status, .preempted)
        XCTAssertEqual(plain.scheduler.record(for: plain.id)?.status, .preempted)
        XCTAssertEqual(spec.scheduler.waiting.first?.id, spec.id)
        XCTAssertEqual(plain.scheduler.waiting.first?.id, plain.id)
        XCTAssertEqual(spec.capacity.releaseAllCalls, [spec.id])
        XCTAssertEqual(plain.capacity.releaseAllCalls, [plain.id])
        // The speculative path tried 1+k, then plain 1, before preempting.
        XCTAssertEqual(Array(spec.capacity.reserveCalls.map(\.tokens).suffix(2)), [4, 1])
    }

    // MARK: In-flight invariant & finalize

    func testInFlightInvariantAndFinalizeSequences() throws {
        let k = 3
        // e confirmed tokens: 1 (no drafts accepted), 2 (one accepted),
        // k+1 (all accepted).
        for confirmed in [1, 2, k + 1] {
            let capacity = SpecScriptedCapacity()
            let scheduler = makeScheduler(capacity: capacity)
            let request = try decodeReadyRow(scheduler, prompt: Array(0 ..< 10))
            let rec = scheduler.record(for: request.id)!
            let base = rec.tokens.count  // P = 11

            scheduler.speculationPlanner = { _ in k }
            let plan = scheduler.plan()
            XCTAssertEqual(plan.assignments.map(\.numTokens), [1 + k])
            scheduler.markPendingSamples(counts: [(id: request.id, count: 1 + k)])

            // In flight: the row still reads decode-ready (remaining == 1).
            XCTAssertEqual(rec.effectiveTokenCount, base + 1 + k)
            XCTAssertEqual(rec.numComputedTokens, base + k)
            XCTAssertEqual(rec.remainingTokens, 1)
            XCTAssertTrue(rec.isDecodeReady)

            // Finalize: confirm e, discard the rest, roll back the rejected
            // executed suffix.
            for i in 0 ..< confirmed { scheduler.recordSampled(id: request.id, token: 100 + i) }
            let rejected = (1 + k) - confirmed
            scheduler.discardPendingSamples(id: request.id, count: rejected)
            scheduler.rollbackComputed(id: request.id, tokens: rejected)

            // Standard decode invariant restored.
            XCTAssertEqual(rec.tokens.count, base + confirmed)
            XCTAssertEqual(rec.numComputedTokens, base + confirmed - 1)
            XCTAssertEqual(rec.pendingSamples, 0)
            XCTAssertTrue(rec.isDecodeReady)
            // Capacity saw the rejection: 10 (prefill) + 1+k − rejected.
            XCTAssertEqual(capacity.reserved[request.id], 10 + confirmed)
            if rejected > 0 {
                XCTAssertEqual(capacity.unreserveCalls.last?.tokens, rejected)
                XCTAssertEqual(capacity.unreserveCalls.last?.id, request.id)
            }
        }
    }

    // MARK: Rollback of an unexecuted speculative plan

    func testRollbackRestoresSpeculativeAssignment() throws {
        let capacity = SpecScriptedCapacity()
        let scheduler = makeScheduler(capacity: capacity)
        let request = try decodeReadyRow(scheduler, prompt: Array(0 ..< 10))
        let rec = scheduler.record(for: request.id)!

        scheduler.speculationPlanner = { _ in 3 }
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.numTokens), [4])
        XCTAssertEqual(rec.numComputedTokens, 14)
        XCTAssertEqual(capacity.reserved[request.id], 14)

        scheduler.rollback(plan)
        XCTAssertEqual(rec.numComputedTokens, 10)
        XCTAssertEqual(capacity.reserved[request.id], 10)
        XCTAssertEqual(capacity.unreserveCalls.last?.tokens, 4)

        // Re-planning reproduces the same speculative assignment.
        let replan = scheduler.plan()
        XCTAssertEqual(replan.assignments.map(\.numTokens), [4])
    }

    // MARK: Preemption with pending speculative samples

    func testPreemptedRowWithPendingSamplesBlocksUntilCorrected() throws {
        let capacity = SpecScriptedCapacity()
        let scheduler = makeScheduler(capacity: capacity)
        let request = try decodeReadyRow(scheduler, prompt: Array(0 ..< 10))

        scheduler.speculationPlanner = { _ in 3 }
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.numTokens), [4])
        scheduler.markPendingSamples(counts: [(id: request.id, count: 4)])

        // Demoted mid-flight (capacity requeue = preempted-style restart).
        XCTAssertTrue(scheduler.requeueOnCapacity(request.id))
        let rec = scheduler.record(for: request.id)!
        XCTAssertEqual(rec.pendingSamples, 4)
        XCTAssertEqual(rec.numComputedTokens, 0)

        // Waiting admission must skip it while samples are unconfirmed.
        XCTAssertTrue(scheduler.plan().assignments.isEmpty)
        XCTAssertEqual(scheduler.waiting.first?.id, request.id)

        // Finalize correction: 2 confirmed, 2 discarded; computed already 0.
        scheduler.recordSampled(id: request.id, token: 100)
        scheduler.recordSampled(id: request.id, token: 101)
        scheduler.discardPendingSamples(id: request.id, count: 2)
        scheduler.rollbackComputed(id: request.id, tokens: 2)
        XCTAssertEqual(rec.pendingSamples, 0)
        XCTAssertEqual(rec.numComputedTokens, 0)
        XCTAssertEqual(rec.tokens.count, 13)

        // Now it re-admits with a full re-prefill of the kept tokens.
        let readmit = scheduler.plan()
        XCTAssertEqual(readmit.assignments.map(\.id), [request.id])
        XCTAssertEqual(readmit.assignments.map(\.numTokens), [13])
    }

    // MARK: rollbackComputed edge cases

    func testRollbackComputedUnknownIDIsNoOp() throws {
        let capacity = SpecScriptedCapacity()
        let scheduler = makeScheduler(capacity: capacity)
        scheduler.rollbackComputed(id: CBv2RequestID(424_242), tokens: 3)
        XCTAssertTrue(capacity.unreserveCalls.isEmpty)

        // Finished mid-flight: finish + engine releaseAll cover the
        // reservation; the late rollbackComputed is a documented no-op.
        let request = try decodeReadyRow(scheduler, prompt: [1, 2])
        scheduler.finish(id: request.id, reason: .cancelled)
        capacity.releaseAll(id: request.id)
        scheduler.rollbackComputed(id: request.id, tokens: 2)
        XCTAssertTrue(capacity.unreserveCalls.isEmpty)
        XCTAssertNil(capacity.reserved[request.id])
    }
}
