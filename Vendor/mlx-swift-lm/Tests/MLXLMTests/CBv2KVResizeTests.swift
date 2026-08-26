// CBv2KVResizeTests.swift — runtime-resizable KV capacity through EngineV2.
//
// Multi-model co-residency (v0.7.5): the provider re-slices KV grants when a
// second model loads (shrink residents first, grant the newcomer the rest)
// and grows survivors back on unload. The engine-side contract under test:
//
//  (i)   `EngineV2.updateKVBytesCapacity` fans out to BOTH ledgers — the
//        `AdmissionV2` soft ledger and the KV backend's hard ledger — and
//        `capacity()` reflects the new ceiling immediately, even while the
//        engine is idle (no step ever published a snapshot).
//  (ii)  Shrink: an in-flight stream keeps decoding untouched (its
//        reservations are never evicted) while NEW submissions are refused
//        against the new ceiling; once the pool drains, admission resumes
//        under the NEW ceiling.
//  (iii) Grow: a submission refused at the old ceiling admits immediately
//        after the ceiling rises, and completes.
//
// TinyTestModel + the real contiguous backend; compiled decode is disabled
// so the admission arithmetic carries no external padding reserve. No model
// downloads; weights are seeded random.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2KVResizeTests: XCTestCase {

    private struct Stack {
        let engine: EngineV2
        let backend: CBv2ContiguousKVBackend
        let admission: AdmissionV2
    }

    private func makeStack(bytesCapacity: Int, maxWaiting: Int = 8) -> Stack {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: bytesCapacity))
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 7),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: maxWaiting))
        return Stack(engine: engine, backend: backend, admission: engine.admissionForTesting)
    }

    private func greedyRequest(id: UInt64, prompt: [Int], maxTokens: Int) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0), maxTokens: maxTokens)
    }

    // MARK: (i) Fan-out — both ledgers + the capacity snapshot

    func testUpdateReachesAdmissionBackendAndCapacitySnapshot() async throws {
        let initial = 1 << 20
        let stack = makeStack(bytesCapacity: initial)
        XCTAssertEqual(stack.admission.bytesCapacity, initial)
        XCTAssertEqual(stack.backend.bytesCapacity, initial)
        XCTAssertEqual(stack.engine.capacity().kvBytesCapacity, initial)

        // Grow, on an IDLE engine: no step has run, so the gauge update must
        // come from the resize itself, not a step snapshot — and it must
        // refresh the BACKEND capacity too (the contiguous backend really
        // resized): min(kvBytesCapacity, kvBytesBackendCapacity) consumers
        // (provider heartbeats) must not under-advertise off a stale
        // backend value until the next step publish.
        let grown = 1 << 22
        stack.engine.updateKVBytesCapacity(grown)
        XCTAssertEqual(stack.admission.bytesCapacity, grown, "admission ledger must resize")
        XCTAssertEqual(stack.backend.bytesCapacity, grown, "backend ledger must resize")
        XCTAssertEqual(
            stack.engine.capacity().kvBytesCapacity, grown,
            "capacity() must reflect the resize while idle")
        XCTAssertEqual(
            stack.engine.capacity().kvBytesBackendCapacity, grown,
            "idle resize must refresh the snapshot's backend capacity")

        // Shrink.
        let shrunk = 1 << 16
        stack.engine.updateKVBytesCapacity(shrunk)
        XCTAssertEqual(stack.admission.bytesCapacity, shrunk)
        XCTAssertEqual(stack.backend.bytesCapacity, shrunk)
        XCTAssertEqual(stack.engine.capacity().kvBytesCapacity, shrunk)
        XCTAssertEqual(stack.engine.capacity().kvBytesBackendCapacity, shrunk)

        // Degenerate input clamps to zero on both ledgers.
        stack.engine.updateKVBytesCapacity(-1)
        XCTAssertEqual(stack.admission.bytesCapacity, 0)
        XCTAssertEqual(stack.backend.bytesCapacity, 0)
        await stack.engine.shutdown()
    }

    // MARK: (ii) Shrink under load — in-flight untouched, new submits refused

    func testShrinkUnderLoadKeepsInFlightAndRefusesNewSubmits() async throws {
        let stack = makeStack(bytesCapacity: 1 << 28)
        let prompt = makePromptTokens(length: 12, seed: 41)
        let maxTokens = 24

        // Submit A, then shrink while it streams. The new ceiling leaves A's
        // FULL worst case admissible (4x margin over the estimate, so its
        // per-step reserves can never hit the wall — no preemption), but is
        // far too small for B's worst case.
        let streamA = try stack.engine.submit(
            greedyRequest(id: 1, prompt: prompt, maxTokens: maxTokens))
        let worstA = stack.admission.estimatedBytes(forTokens: prompt.count + maxTokens)
        stack.engine.updateKVBytesCapacity(worstA * 4)

        // B could never fit the shrunken ceiling — refused at submit, while
        // A is still live.
        XCTAssertThrowsError(
            try stack.engine.submit(greedyRequest(id: 2, prompt: prompt, maxTokens: 100_000))
        ) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
        }

        // A is unaffected: full-length finish.
        let collectedA = await cbv2SchedCollect(streamA)
        XCTAssertEqual(collectedA.finishReason, .length, "in-flight request must survive shrink")
        XCTAssertEqual(collectedA.tokens.count, maxTokens)

        // After the pool drains, a request sized for the NEW ceiling admits
        // and completes.
        let collectedC = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 3, prompt: prompt, maxTokens: 8)))
        XCTAssertEqual(collectedC.finishReason, .length)
        XCTAssertEqual(collectedC.tokens.count, 8)
        await stack.engine.shutdown()
    }

    // MARK: (iii) Grow — refused at the old ceiling, admitted after

    func testGrowAdmitsPreviouslyRefusedRequest() async throws {
        // Ceiling too small for the request's worst case (canEverFit fails):
        // 12 + 128 = 140 tokens × 128 B/token on the full layer alone ≈ 18 KB
        // against a 4 KB budget.
        let stack = makeStack(bytesCapacity: 1 << 12)
        let prompt = makePromptTokens(length: 12, seed: 42)
        let request = greedyRequest(id: 1, prompt: prompt, maxTokens: 128)
        XCTAssertThrowsError(try stack.engine.submit(request)) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
        }

        // Grow: the SAME request admits immediately and completes.
        stack.engine.updateKVBytesCapacity(1 << 24)
        let collected = await cbv2SchedCollect(try stack.engine.submit(request))
        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertEqual(collected.tokens.count, 128)
        await stack.engine.shutdown()
    }
}
