// CBv2TeacherForcedScoringTests.swift
//
// Seam 3 of the Gate G2 parity harness: `CBv2Engine.teacherForcedTop1`.
//
// WHY THE SEAM EXISTS. Comparing two engine arms by FREE RUNNING is valid
// only up to their first disagreement. Past it each arm is decoding its own
// context, so position i in arm A and position i in arm B are answers to
// different questions — the harness can report a first-flip index but must
// refuse to divide anything by anything. Teacher forcing removes the
// coupling: every position is scored against `prompt + continuation[0..<i]`,
// the same context in both arms, so agreement is a rate over a fixed
// denominator and the bar can come from a control arm instead of a number
// somebody picked.
//
// WHAT IS ACTUALLY AT RISK. A witness that returned `continuation` unchanged,
// or scored the positions offline against the raw model, or quietly free-ran
// past the first flip, would satisfy a naive "does it return N ids" test and
// report a rate that measures nothing. Three of these tests are built to fail
// on exactly those implementations:
//
//   * `forcedContinuationIsScoredAgainstTheForcedContext` pins the result
//     against an INDEPENDENT teacher-forced reference built from the raw
//     backend + cache bank + model — and separately proves the answer is
//     neither the input echoed back nor what free running would have said.
//   * the execution counters (`teacherForcedScoringActivity`) are asserted as
//     exact identities, so a witness that scored outside the engine returns
//     the right ids and still fails: `decodeForwardsExecuted` stays flat.
//   * `failClosedDefaultRefusesRatherThanReturningOne` pins the protocol
//     default at a throw, because `[]` divides to a vacuous 100%.
//
// All fixtures are `TinyTestModel` (seeded random weights, real attention,
// real KV) over the real contiguous and paged backends. No downloads.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 teacher-forced top-1 scoring", .serialized)
struct CBv2TeacherForcedScoringTests {

    // MARK: - Fixture

    /// headDim 64 for BOTH arms: the paged Metal kernel supports
    /// {64, 128, 256, 512}, and a cross-backend comparison is only meaningful
    /// when the two arms carry byte-identical weights.
    private static let weightSeed: UInt64 = 0xC0FFEE
    private static let headDim = 64
    /// Prompt spans several chunks AND crosses the sliding window (16), so
    /// the windowed layer is exercised from the first scored position.
    private static let promptLength = 37
    private static let chunkSize = 16
    private static let stepBudget = 256
    private static let vocab = TinyTestModelConfig().vocabSize  // 128

    private enum BackendKind { case contiguous, paged }

    private func makeModel() -> TinyTestModel {
        TinyTestModel.make(seed: Self.weightSeed, headDim: Self.headDim)
    }

    private func makeBackend(_ kind: BackendKind, _ kinds: [CBv2LayerKind]) throws -> CBv2KVBackend
    {
        switch kind {
        case .contiguous:
            return CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28))
        case .paged:
            return try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    capacityBytes: 64 << 20,
                    maxPrefillChunk: Self.chunkSize,
                    nominalMaxSequenceLength: 512))
        }
    }

    private func makeBank(_ backend: CBv2KVBackend, _ kinds: [CBv2LayerKind]) -> CBv2LayerCacheBank
    {
        if let paged = backend as? PagedKVBackend {
            return CBv2LayerCacheBank(caches: paged.makeLayerCaches())
        }
        return CBv2LayerCacheBank(layerKinds: kinds)
    }

    /// A solo-request engine: greedy sampler (so a free run is argmax, the
    /// same decision rule the scoring seam applies), prefix cache off (an
    /// adoption would change the prompt's chunking and defeat the fixed-point
    /// comparison).
    private func makeEngine(_ kind: BackendKind, model: TinyTestModel) throws -> EngineV2 {
        let backend = try makeBackend(kind, model.layerKinds)
        return EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: backend,
            cacheProvider: makeBank(backend, model.layerKinds),
            sampler: CBv2GreedySampler(),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4,
                maxBatchedTokensPerStep: Self.stepBudget,
                prefillChunkSize: Self.chunkSize,
                maxWaiting: 16,
                enablePrefixCache: false))
    }

    /// Free-run the engine greedily and return the ids it chose for itself.
    private func freeRun(
        _ engine: EngineV2, prompt: [Int], maxTokens: Int, id: UInt64
    ) async throws -> [Int] {
        let collected = await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(id), promptTokens: prompt,
                    sampling: CBv2SamplingParams(temperature: 0), maxTokens: maxTokens)))
        return collected.tokens
    }

    /// INDEPENDENT teacher-forced reference: raw backend + cache bank + model,
    /// no `EngineV2` and no `EngineLoopV2` anywhere in the call graph. Same
    /// shapes and same order as the seam (chunked prompt, then one `[1, 1]`
    /// forward per forced token), so agreement with it is bit-exact rather
    /// than approximate.
    private func referenceTeacherForcedTop1(
        model: TinyTestModel, backend: CBv2KVBackend, bank: CBv2LayerCacheBank,
        prompt: [Int], continuation: [Int]
    ) throws -> [Int] {
        let state = try backend.makeSequenceState(
            layerKinds: model.layerKinds,
            promptLength: prompt.count,
            maxLength: prompt.count + continuation.count)
        defer {
            bank.releaseBoundRows()
            backend.release(state)
        }
        var top1: [Int] = []
        var index = 0
        while index < prompt.count {
            let count = min(Self.chunkSize, prompt.count - index)
            let caches = bank.layerCaches(rowStates: [state])
            let logits = model.forward(
                tokens: MLXArray(prompt[index ..< index + count].map(Int32.init))
                    .reshaped(1, count),
                caches: caches)
            if index + count == prompt.count {
                top1.append(Int(argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self)[0]))
            }
            index += count
        }
        for forced in continuation.dropLast() {
            let caches = bank.layerCaches(rowStates: [state])
            let logits = model.forward(
                tokens: MLXArray([Int32(forced)]).reshaped(1, 1), caches: caches)
            top1.append(Int(argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self)[0]))
        }
        return top1
    }

    /// Chunks the prompt costs at `min(prefillChunkSize, budget)`.
    private var expectedPrefillChunks: Int {
        let chunk = min(Self.chunkSize, Self.stepBudget)
        return (Self.promptLength + chunk - 1) / chunk
    }

    private func agreement(_ lhs: [Int], _ rhs: [Int]) -> Double {
        precondition(lhs.count == rhs.count && !lhs.isEmpty)
        var matches = 0
        for (l, r) in zip(lhs, rhs) where l == r { matches += 1 }
        return Double(matches) / Double(lhs.count)
    }

    // MARK: - 1. A self-produced continuation is a fixed point

    /// Forcing the engine's OWN greedy continuation must return it unchanged.
    /// The forced context and the free-running context coincide at every
    /// position, so scoring can only reproduce it — unless the scoring path
    /// prefills differently, offsets positions differently, or feeds the
    /// wrong token, all of which show up as a mismatch here.
    @Test func forcingTheEnginesOwnGreedyContinuationIsAFixedPoint() async throws {
        let model = makeModel()
        let engine = try makeEngine(.contiguous, model: model)
        let prompt = makePromptTokens(length: Self.promptLength, seed: 0x7EA_C4E)
        let steps = 12

        let greedy = try await freeRun(engine, prompt: prompt, maxTokens: steps, id: 1)
        #expect(greedy.count == steps)

        let scored = try engine.teacherForcedTop1(promptTokens: prompt, continuation: greedy)
        #expect(scored == greedy)

        await engine.shutdown()
    }

    // MARK: - 2. The forcing is REAL

    /// Force a continuation the engine would never have produced and every
    /// position must be scored against THAT context.
    ///
    /// Four independent claims, each of which kills a different fake witness:
    ///   (a) position 0 still matches free running — it sees only the prompt,
    ///       so a scoring path that mangled the prompt fails here alone;
    ///   (b) the result is NOT the forced continuation echoed back;
    ///   (c) the result DIVERGES from what free running said, so the forced
    ///       tokens genuinely entered the context rather than being ignored;
    ///   (d) the result equals an independent teacher-forced reference
    ///       computed straight off the backend, bit for bit.
    @Test func forcedContinuationIsScoredAgainstTheForcedContext() async throws {
        let model = makeModel()
        let engine = try makeEngine(.contiguous, model: model)
        let prompt = makePromptTokens(length: Self.promptLength, seed: 0x5C0_5ED)
        let steps = 12

        let greedy = try await freeRun(engine, prompt: prompt, maxTokens: steps, id: 1)
        #expect(greedy.count == steps)

        // Differs from the model's own path at EVERY position, and 17 is
        // coprime with the vocab so no element accidentally coincides.
        let forced = greedy.map { ($0 + 17) % Self.vocab }
        #expect(zip(greedy, forced).allSatisfy { $0 != $1 })

        let before = engine.teacherForcedScoringActivity()
        let scored = try engine.teacherForcedTop1(promptTokens: prompt, continuation: forced)
        let after = engine.teacherForcedScoringActivity()

        // (a) Position 0 is conditioned on the prompt alone.
        #expect(scored.count == forced.count)
        #expect(scored[0] == greedy[0])

        // (b) Not an echo of the input.
        #expect(scored != forced)

        // (c) The forced context really displaced the model's own path.
        #expect(scored != greedy)

        // (d) Bit-exact against a reference that never touches EngineV2.
        let referenceBackend = try makeBackend(.contiguous, model.layerKinds)
        let reference = try referenceTeacherForcedTop1(
            model: model,
            backend: referenceBackend,
            bank: makeBank(referenceBackend, model.layerKinds),
            prompt: prompt, continuation: forced)
        #expect(scored == reference)

        // EXECUTION evidence, not a capability claim: a witness that scored
        // these positions offline would satisfy (a)-(d) and leave these flat.
        #expect(
            after.decodeForwardsExecuted - before.decodeForwardsExecuted == forced.count - 1,
            "one [1, 1] engine forward per forced token, except the last")
        #expect(
            after.prefillChunksExecuted - before.prefillChunksExecuted == expectedPrefillChunks,
            "the prompt must travel the engine's own chunked prefill")
        #expect(after.didExecute)

        await engine.shutdown()
    }

    // MARK: - 3. Determinism

    /// No sampler, no temperature, no seed: repeated calls must be identical,
    /// including when another scoring call is interleaved between them (a
    /// scoring path that leaked KV or cache-binding state across calls would
    /// drift on the third).
    @Test func repeatedCallsReturnIdenticalIds() async throws {
        let model = makeModel()
        let engine = try makeEngine(.contiguous, model: model)
        let promptA = makePromptTokens(length: Self.promptLength, seed: 0xDE7_E4)
        let promptB = makePromptTokens(length: 23, seed: 0xDE7_E5)
        let forcedA = (0 ..< 10).map { (7 * $0 + 3) % Self.vocab }
        let forcedB = (0 ..< 6).map { (11 * $0 + 5) % Self.vocab }

        let first = try engine.teacherForcedTop1(promptTokens: promptA, continuation: forcedA)
        let second = try engine.teacherForcedTop1(promptTokens: promptA, continuation: forcedA)
        _ = try engine.teacherForcedTop1(promptTokens: promptB, continuation: forcedB)
        let third = try engine.teacherForcedTop1(promptTokens: promptA, continuation: forcedA)

        #expect(first.count == forcedA.count)
        #expect(second == first)
        #expect(third == first, "an interleaved scoring call must not perturb the next one")

        // The control arm the harness derives its bar from: an arm scored
        // against itself is 100% by construction, not by luck.
        #expect(agreement(first, third) == 1.0)

        await engine.shutdown()
    }

    // MARK: - 4. Fail-closed default

    /// An engine with no scoring path must REFUSE. Returning `[]` would give
    /// the harness `0/0`, which it would round to a passing 100%; echoing the
    /// continuation would report perfect agreement between two arms that were
    /// never run. Both are worse than an error.
    @Test func failClosedDefaultRefusesRatherThanReturningOne() throws {
        let engine = CBv2NonScoringEngine()

        #expect(throws: CBv2TeacherForcingError.unsupported(engine: "CBv2NonScoringEngine")) {
            _ = try engine.teacherForcedTop1(promptTokens: [1, 2, 3], continuation: [4, 5])
        }
        // And the evidence surface fails closed with it: no capability, no
        // execution, nothing a harness could read as a measurement.
        #expect(engine.teacherForcedScoringActivity() == .none)
        #expect(engine.teacherForcedScoringActivity().didExecute == false)
    }

    // MARK: - 5. The other fail-closed refusals

    @Test func emptyPromptOrEmptyContinuationRefuses() async throws {
        let model = makeModel()
        let engine = try makeEngine(.contiguous, model: model)
        let prompt = makePromptTokens(length: 8, seed: 0xE_3179)

        #expect(throws: CBv2TeacherForcingError.nothingToScore(promptTokens: 8, continuation: 0)) {
            _ = try engine.teacherForcedTop1(promptTokens: prompt, continuation: [])
        }
        #expect(throws: CBv2TeacherForcingError.nothingToScore(promptTokens: 0, continuation: 3)) {
            _ = try engine.teacherForcedTop1(promptTokens: [], continuation: [1, 2, 3])
        }
        #expect(engine.teacherForcedScoringActivity().didExecute == false)

        await engine.shutdown()
    }

    @Test func aShutDownEngineRefusesToScore() async throws {
        let model = makeModel()
        let engine = try makeEngine(.contiguous, model: model)
        let prompt = makePromptTokens(length: 8, seed: 0x5_47D0)
        await engine.shutdown()

        #expect(throws: CBv2TeacherForcingError.engineNotRunning) {
            _ = try engine.teacherForcedTop1(promptTokens: prompt, continuation: [1, 2, 3])
        }
    }

    /// A contended pool seats the scoring row on different pages, and storage
    /// order is precisely the drift the harness is measuring — so a busy
    /// engine refuses rather than returning a number that is not comparable
    /// with the one the other arm took while idle.
    @Test func aBusyEngineRefusesToScore() async throws {
        // Mock stack: the scripted model's `forwardDelay` holds the engine
        // inside a step, so "busy" is a state the test creates rather than
        // races for.
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 8, maxWaiting: 8))
        harness.model.forwardDelay = 0.02
        let request = CBv2SchedFixtures.request(
            prompt: makePromptTokens(length: 16, seed: 0xB115_C0DE, vocabSize: 64), maxTokens: 256)
        let stream = try harness.engine.submit(request)
        let consumer = Task { _ = await cbv2SchedCollect(stream, timeoutSeconds: 30) }
        defer {
            consumer.cancel()
        }

        #expect(
            await cbv2SchedWait(timeoutSeconds: 10) {
                harness.engine.capacity().activeRequests > 0
            })

        #expect(throws: (any Error).self) {
            _ = try harness.engine.teacherForcedTop1(promptTokens: [1, 2, 3], continuation: [4, 5])
        }
        do {
            _ = try harness.engine.teacherForcedTop1(promptTokens: [1, 2, 3], continuation: [4, 5])
            Issue.record("a busy engine must refuse")
        } catch let error as CBv2TeacherForcingError {
            guard case .engineBusy(let scheduled) = error else {
                Issue.record("expected .engineBusy, got \(error)")
                return
            }
            #expect(scheduled > 0)
        }
        #expect(harness.engine.teacherForcedScoringActivity().didExecute == false)

        harness.engine.cancel(request.id)
        await harness.engine.shutdown()
    }

    // MARK: - 6. Both backends, one forced context

    /// The seam's whole purpose, exercised on the pair it exists for.
    ///
    /// Under free running the two arms can be compared only up to their first
    /// flip. Under teacher forcing both score the SAME forced context, so the
    /// denominator is `continuation.count` in both arms whatever either would
    /// have preferred — and the bar comes from the control arm (an arm scored
    /// against itself, exactly 1.0) rather than a chosen constant.
    @Test func bothBackendsScoreEveryPositionAgainstTheSameForcedContext() async throws {
        let model = makeModel()
        let contiguous = try makeEngine(.contiguous, model: model)
        let paged = try makeEngine(.paged, model: model)
        let prompt = makePromptTokens(length: Self.promptLength, seed: 0xBEEF_0001)
        let steps = 12

        // Each arm's own greedy continuation is a fixed point of its own
        // scoring — neither arm is echoing, on either storage layout.
        let greedyContiguous = try await freeRun(
            contiguous, prompt: prompt, maxTokens: steps, id: 1)
        let greedyPaged = try await freeRun(paged, prompt: prompt, maxTokens: steps, id: 2)
        #expect(
            try contiguous.teacherForcedTop1(
                promptTokens: prompt, continuation: greedyContiguous) == greedyContiguous)
        #expect(
            try paged.teacherForcedTop1(
                promptTokens: prompt, continuation: greedyPaged) == greedyPaged)

        // One forced context, both arms.
        let forced = greedyContiguous.map { ($0 + 17) % Self.vocab }
        let scoredContiguous = try contiguous.teacherForcedTop1(
            promptTokens: prompt, continuation: forced)
        let scoredPaged = try paged.teacherForcedTop1(promptTokens: prompt, continuation: forced)

        // EVERY position is comparable — this is what free running cannot do.
        #expect(scoredContiguous.count == forced.count)
        #expect(scoredPaged.count == forced.count)

        // Both arms must actually be FORCED, or the cross-arm rate below is
        // two free runs wearing a rate's clothes. Neither arm may echo the
        // input, and neither may quietly reproduce its own preferred path.
        #expect(scoredContiguous != forced)
        #expect(scoredPaged != forced)
        #expect(scoredContiguous != greedyContiguous)
        #expect(scoredPaged != greedyPaged)

        // Control arm: 1.0 by construction, which is what makes the candidate
        // rate a measurement instead of a number next to a chosen threshold.
        let control = try contiguous.teacherForcedTop1(
            promptTokens: prompt, continuation: forced)
        #expect(agreement(scoredContiguous, control) == 1.0)

        // Candidate rate, computed over the full denominator. Its VALUE is a
        // property of the two kernels on this hardware, so the assertion is
        // that it is well defined at every position, not that it hits a
        // number: a candidate below control is exactly the finding the gate
        // exists to surface, and pinning a constant here would hide it.
        let candidate = agreement(scoredPaged, scoredContiguous)
        #expect(candidate >= 0.0 && candidate <= 1.0)

        // Execution evidence on the PAGED arm too: the paged pages were
        // really walked, not simulated off the contiguous arm. Exactly two
        // scoring calls ran on this engine — the fixed-point check and the
        // forced one — each spending one forward per position but the last.
        let pagedActivity = paged.teacherForcedScoringActivity()
        #expect(pagedActivity.didExecute)
        #expect(
            pagedActivity.decodeForwardsExecuted
                == (greedyPaged.count - 1) + (forced.count - 1))
        #expect(pagedActivity.prefillChunksExecuted == 2 * expectedPrefillChunks)

        await contiguous.shutdown()
        await paged.shutdown()
    }
}

// MARK: - Fail-closed default fixture

/// The smallest legal `CBv2Engine`: it implements the four members that have
/// no default and inherits everything else, so it is exactly the shape of an
/// engine that never opted into teacher-forced scoring.
private final class CBv2NonScoringEngine: CBv2Engine, @unchecked Sendable {
    func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> {
        AsyncStream { $0.finish() }
    }
    func cancel(_ id: CBv2RequestID) {}
    func capacity() -> CBv2CapacitySnapshot {
        CBv2CapacitySnapshot(
            activeRequests: 0, waitingRequests: 0, kvBytesInUse: 0, kvBytesCapacity: 0,
            activeTokens: 0)
    }
    func shutdown() async {}
}
