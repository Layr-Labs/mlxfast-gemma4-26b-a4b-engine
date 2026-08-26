// CBv2EndToEndTests.swift — integration: the REAL engine assembly, end to end.
//
// Constructs the production EngineV2 from the real subsystems — TinyTestModel
// (a real 2-layer transformer conforming to CBv2SteppableModel; layer 0 full
// attention, layer 1 sliding-window(16)), a real KV backend
// (CBv2ContiguousKVBackend or PagedKVBackend), CBv2LayerCacheBank,
// CBv2DefaultSampler (LogitsPipelineV2 + SamplerV2), CBv2TextDetokenizerFactory
// (DetokenizerV2 + StopHoldback), and PrefixCacheV2 — and asserts:
//
//  (i)   3 concurrent streaming requests with mixed prompt lengths finish
//        with correct text and TOKEN-EXACT batch invariance vs solo runs;
//  (ii)  a stop string completed mid-stream truncates the text exactly at
//        the match start (holdback: no text at/past the match ever emitted);
//  (iii) cancel mid-decode frees the slot promptly and batchmates are
//        unaffected (still token-exact vs solo);
//  (iv)  a second submit of the same prompt hits the prefix cache
//        (usage.prefixCacheHitTokens > 0) with IDENTICAL greedy output.
//        The adopted full-attention KV is bit-identical (donated arrays),
//        and the windowed layer is last, so the trailing-window replay
//        recomputes the same values — but the replay's SDPA calls run at
//        different shapes than the cold run's chunks, and cross-shape SDPA
//        determinism is an empirical property of the kernels on this
//        hardware, not a guarantee. Greedy argmax over these logit gaps
//        makes the token-exact assertion stable in practice;
//  (v)   the same suite runs green with the paged backend (fp16 pages,
//        custom Metal decode kernel) substituted for the contiguous one.
//
// No model downloads; weights are seeded random.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - Test tokenizer

/// Deterministic id → text mapping ("<id>") so expected text is a pure
/// function of the token stream; distinct ids can never be substrings of
/// each other's rendering ("<12>" vs "<123>").
private struct CBv2E2ETokenizer: Tokenizer {
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "<\($0)>" }.joined()
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
}

private func renderedText(_ tokens: [Int]) -> String {
    tokens.map { "<\($0)>" }.joined()
}

// MARK: - Suite

final class CBv2EndToEndTests: XCTestCase {

    private enum BackendKind {
        case contiguous
        case paged
    }

    private struct Stack {
        let engine: EngineV2
        let model: TinyTestModel
        let prefixCache: PrefixCacheV2
    }

    /// Window of the fixture model (both head-dim variants).
    private let window = TinyTestModelConfig().windowSize  // 16
    /// Small hash blocks so short prompts produce multi-block hits.
    private let blockSize = 8

    /// One model per backend kind per test (seeded ⇒ identical weights for
    /// every stack built from it, so solo and batched runs share weights).
    private func makeModel(_ kind: BackendKind) -> TinyTestModel {
        switch kind {
        case .contiguous: return TinyTestModel.make(seed: 0xC0FFEE)
        // The paged Metal kernel supports headDim ∈ {64,128,256,512}.
        case .paged: return TinyTestModel.make(seed: 0xC0FFEE, headDim: 64)
        }
    }

    private func makeStack(
        _ kind: BackendKind,
        model: TinyTestModel,
        enablePrefixCache: Bool = true,
        prefillChunkSize: Int = 16
    ) throws -> Stack {
        let backend: CBv2KVBackend
        let bank: CBv2LayerCacheBank
        var materializeOnDonate = false
        switch kind {
        case .contiguous:
            backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28))
            bank = CBv2LayerCacheBank(layerKinds: model.layerKinds)
        case .paged:
            let paged: PagedKVBackend
            do {
                paged = try PagedKVBackend(
                    layerKinds: model.layerKinds,
                    config: PagedKVPoolConfig(
                        capacityBytes: 64 << 20,
                        maxPrefillChunk: 64,
                        nominalMaxSequenceLength: 512))
            } catch let error as CBv2KVError {
                throw XCTSkip("paged backend unavailable on this hardware: \(error)")
            }
            backend = paged
            bank = CBv2LayerCacheBank(caches: paged.makeLayerCaches())
            // Donated views reference the live fp16 slabs; pages are
            // recycled after release, so donations must materialize.
            materializeOnDonate = true
        }
        let prefixCache = PrefixCacheV2(
            config: .init(
                blockSize: blockSize, promptContractID: "cbv2-e2e",
                materializeOnDonate: materializeOnDonate))
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: backend,
            cacheProvider: bank,
            sampler: CBv2DefaultSampler(fallbackSeed: 11),
            detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: CBv2E2ETokenizer()),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: prefillChunkSize, maxWaiting: 16,
                enablePrefixCache: enablePrefixCache),
            prefixCache: prefixCache)
        return Stack(engine: engine, model: model, prefixCache: prefixCache)
    }

    private func greedyRequest(
        id: UInt64, prompt: [Int], maxTokens: Int, stopStrings: [String] = []
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0),
            maxTokens: maxTokens, stopStrings: stopStrings)
    }

    /// Run one greedy request to completion on a FRESH engine (the solo
    /// baseline) and return what it produced.
    private func soloRun(
        _ kind: BackendKind, model: TinyTestModel, prompt: [Int], maxTokens: Int
    ) async throws -> CBv2SchedCollected {
        let stack = try makeStack(kind, model: model, enablePrefixCache: false)
        let collected = await cbv2SchedCollect(
            try stack.engine.submit(
                greedyRequest(id: 900, prompt: prompt, maxTokens: maxTokens)))
        await stack.engine.shutdown()
        return collected
    }

    // MARK: (i) Concurrent streaming — text correctness + batch invariance

    private func runConcurrentStreamingInvariance(_ kind: BackendKind) async throws {
        let model = makeModel(kind)
        // Mixed prompt lengths: shorter than the window, spanning it, and
        // several prefill chunks long.
        let prompts = [
            makePromptTokens(length: 5, seed: 11),
            makePromptTokens(length: 24, seed: 22),
            makePromptTokens(length: 41, seed: 33),
        ]
        let maxTokens = [32, 24, 40]

        var solos: [CBv2SchedCollected] = []
        for (prompt, budget) in zip(prompts, maxTokens) {
            solos.append(
                try await soloRun(kind, model: model, prompt: prompt, maxTokens: budget))
        }

        let stack = try makeStack(kind, model: model, enablePrefixCache: false)
        var streams: [AsyncStream<CBv2Event>] = []
        for (i, prompt) in prompts.enumerated() {
            streams.append(
                try stack.engine.submit(
                    greedyRequest(id: UInt64(i + 1), prompt: prompt, maxTokens: maxTokens[i])))
        }
        var batched: [CBv2SchedCollected] = []
        for stream in streams {
            batched.append(await cbv2SchedCollect(stream))
        }
        await stack.engine.shutdown()

        for i in prompts.indices {
            XCTAssertEqual(batched[i].finishReason, .length, "request \(i)")
            XCTAssertEqual(batched[i].tokens.count, maxTokens[i], "request \(i)")
            XCTAssertEqual(
                batched[i].tokens, solos[i].tokens,
                "request \(i) diverged from its solo run under batching")
            XCTAssertEqual(
                batched[i].text, renderedText(batched[i].tokens),
                "request \(i): streamed text must be the exact rendering of its tokens")
            XCTAssertEqual(batched[i].usage?.promptTokens, prompts[i].count, "request \(i)")
            XCTAssertEqual(batched[i].usage?.completionTokens, maxTokens[i], "request \(i)")
        }
        XCTAssertGreaterThan(
            stack.engine.capacity().stepsExecuted, 0, "step counter must be published")
    }

    func testConcurrentStreamingInvariance_Contiguous() async throws {
        try await runConcurrentStreamingInvariance(.contiguous)
    }

    func testConcurrentStreamingInvariance_Paged() async throws {
        try await runConcurrentStreamingInvariance(.paged)
    }

    // MARK: (ii) Stop string mid-stream

    private func runStopStringTruncation(_ kind: BackendKind) async throws {
        let model = makeModel(kind)
        let prompt = makePromptTokens(length: 12, seed: 44)
        let budget = 48

        // Baseline (no stop) discovers the greedy continuation, from which
        // a mid-stream stop string is derived.
        let baseline = try await soloRun(kind, model: model, prompt: prompt, maxTokens: budget)
        XCTAssertEqual(baseline.tokens.count, budget)
        let stopToken = baseline.tokens[budget / 2]
        let stopString = renderedText([stopToken])
        let firstHit = baseline.tokens.firstIndex(of: stopToken)!
        let expectedText = renderedText(Array(baseline.tokens[..<firstHit]))

        let stack = try makeStack(kind, model: model, enablePrefixCache: false)
        let collected = await cbv2SchedCollect(
            try stack.engine.submit(
                greedyRequest(
                    id: 1, prompt: prompt, maxTokens: budget, stopStrings: [stopString])))
        await stack.engine.shutdown()

        XCTAssertEqual(collected.finishReason, .stop)
        XCTAssertEqual(
            collected.text, expectedText,
            "stop-string truncation must cut EXACTLY at the match start")
        XCTAssertFalse(
            collected.text.contains(stopString),
            "no text at or past the stop match may ever be emitted")
    }

    func testStopStringTruncation_Contiguous() async throws {
        try await runStopStringTruncation(.contiguous)
    }

    func testStopStringTruncation_Paged() async throws {
        try await runStopStringTruncation(.paged)
    }

    // MARK: (ii-b) Stop TOKEN text suppression

    /// Regression: the stop token's rendering used to be emitted in the
    /// final delta before the `.stop` finish. OpenAI behavior excludes it:
    /// `text` must end exactly before the stop token, while the raw delta
    /// `tokens` still carry its id and usage still counts it (see the
    /// CBv2Event.delta contract doc).
    private func runStopTokenTextSuppression(_ kind: BackendKind) async throws {
        let model = makeModel(kind)
        let prompt = makePromptTokens(length: 12, seed: 88)
        let budget = 48

        // Baseline greedy run discovers a token to stop on mid-stream.
        let baseline = try await soloRun(kind, model: model, prompt: prompt, maxTokens: budget)
        XCTAssertEqual(baseline.tokens.count, budget)
        let stopToken = baseline.tokens[budget / 2]
        let firstHit = baseline.tokens.firstIndex(of: stopToken)!

        let stack = try makeStack(kind, model: model, enablePrefixCache: false)
        var request = greedyRequest(id: 1, prompt: prompt, maxTokens: budget)
        request.stopTokens = [stopToken]
        let collected = await cbv2SchedCollect(try stack.engine.submit(request))
        await stack.engine.shutdown()

        XCTAssertEqual(collected.finishReason, .stop)
        // Raw ids include the stop token; text must NOT.
        XCTAssertEqual(collected.tokens, Array(baseline.tokens[...firstHit]))
        XCTAssertEqual(
            collected.text, renderedText(Array(baseline.tokens[..<firstHit])),
            "the stop token's rendering must never be emitted")
        XCTAssertFalse(collected.text.contains(renderedText([stopToken])))
        // Usage still counts the sampled stop token.
        XCTAssertEqual(collected.usage?.completionTokens, firstHit + 1)
    }

    func testStopTokenTextSuppression_Contiguous() async throws {
        try await runStopTokenTextSuppression(.contiguous)
    }

    func testStopTokenTextSuppression_Paged() async throws {
        try await runStopTokenTextSuppression(.paged)
    }

    // MARK: (iii) Cancel mid-decode

    private func runCancelMidDecode(_ kind: BackendKind) async throws {
        let model = makeModel(kind)
        let victimPrompt = makePromptTokens(length: 9, seed: 55)
        let survivorPrompt = makePromptTokens(length: 21, seed: 66)
        let survivorBudget = 40

        let survivorSolo = try await soloRun(
            kind, model: model, prompt: survivorPrompt, maxTokens: survivorBudget)

        let stack = try makeStack(kind, model: model, enablePrefixCache: false)
        let victimID = CBv2RequestID(1)
        let victimStream = try stack.engine.submit(
            greedyRequest(id: 1, prompt: victimPrompt, maxTokens: 400))
        let survivorStream = try stack.engine.submit(
            greedyRequest(id: 2, prompt: survivorPrompt, maxTokens: survivorBudget))

        // Consume a few victim deltas mid-decode, then cancel.
        var victimTokens: [Int] = []
        var victimReason: CBv2FinishReason?
        for await event in victimStream {
            switch event {
            case .delta(_, let tokens, _):
                victimTokens.append(contentsOf: tokens)
                if victimTokens.count == 5 {
                    stack.engine.cancel(victimID)
                }
            case .finished(let reason, _):
                victimReason = reason
            }
        }
        XCTAssertEqual(victimReason, .cancelled)
        XCTAssertLessThan(victimTokens.count, 400, "cancel must stop generation promptly")

        // The batchmate is unaffected: finishes with its solo output.
        let survivor = await cbv2SchedCollect(survivorStream)
        XCTAssertEqual(survivor.finishReason, .length)
        XCTAssertEqual(
            survivor.tokens, survivorSolo.tokens,
            "cancelling a batchmate must not perturb other requests")

        // The victim's slot is freed: capacity drains to zero active.
        let drained = await cbv2SchedWait {
            stack.engine.capacity().activeRequests == 0
        }
        XCTAssertTrue(drained, "cancelled + finished requests must free their slots")
        await stack.engine.shutdown()
    }

    func testCancelMidDecode_Contiguous() async throws {
        try await runCancelMidDecode(.contiguous)
    }

    func testCancelMidDecode_Paged() async throws {
        try await runCancelMidDecode(.paged)
    }

    // MARK: (iv) Prefix-cache hit round trip

    private func runPrefixCacheRoundTrip(_ kind: BackendKind) async throws {
        let model = makeModel(kind)
        // 41-token prompt, blockSize 8: lookup is capped at prompt-1 = 40
        // tokens = 5 whole blocks; window 16 ⇒ recompute 16 ⇒ 24 adopted.
        let prompt = makePromptTokens(length: 41, seed: 77)
        let budget = 8
        let expectedHit = 5 * blockSize - window  // 24

        let stack = try makeStack(kind, model: model, enablePrefixCache: true)

        let first = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 1, prompt: prompt, maxTokens: budget)))
        XCTAssertEqual(first.finishReason, .length)
        XCTAssertEqual(first.usage?.prefixCacheHitTokens, 0, "first run is a miss")

        // Donation is asynchronous (off the engine queue) — wait for it.
        let donated = await cbv2SchedWait {
            stack.prefixCache.stats().entryCount >= 1
        }
        XCTAssertTrue(donated, "finished request must donate its prefix")

        let second = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 2, prompt: prompt, maxTokens: budget)))
        await stack.engine.shutdown()

        XCTAssertEqual(second.finishReason, .length)
        let hit = second.usage?.prefixCacheHitTokens ?? 0
        XCTAssertGreaterThan(hit, 0, "second submit of the same prompt must hit")
        XCTAssertEqual(hit, expectedHit, "hit = whole-block match minus windowed recompute")
        XCTAssertEqual(
            second.tokens, first.tokens,
            "prefix-cache adoption must be token-exact vs the cold run")
        XCTAssertEqual(second.text, first.text)
        XCTAssertGreaterThan(stack.prefixCache.stats().hits, 0)
    }

    func testPrefixCacheRoundTrip_Contiguous() async throws {
        try await runPrefixCacheRoundTrip(.contiguous)
    }

    func testPrefixCacheRoundTrip_Paged() async throws {
        try await runPrefixCacheRoundTrip(.paged)
    }

    // MARK: (iv-c) Frozen-full hybrid replay

    /// End-to-end production-engine regression for
    /// [full, sliding(16), sliding(16), full]. M=72, R=32, C=40: sliding rows
    /// rebuild from C while both owning full rows retain exact K/V through M.
    func testStackedSlidingFrozenFullReplayIsTokenExact() async throws {
        let model = TinyTestModel.make(seed: 0xD00D_F00D, stackedSlidingFull: true)
        XCTAssertEqual(model.layerKinds.count, 4, "shape must be [full, sliding, sliding, full]")
        let prompt = makePromptTokens(length: 73, seed: 0x5EED)
        let budget = 16

        let stack = try makeStack(
            .contiguous,
            model: model,
            enablePrefixCache: true,
            prefillChunkSize: 64)

        let first = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 1, prompt: prompt, maxTokens: budget)))
        XCTAssertEqual(first.finishReason, .length)
        XCTAssertEqual(first.usage?.prefixCacheHitTokens, 0, "first run is a miss")

        // Donation is asynchronous (off the engine queue) — wait for it.
        let donated = await cbv2SchedWait { stack.prefixCache.stats().entryCount >= 1 }
        XCTAssertTrue(donated, "finished request must donate its prefix")

        let second = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 2, prompt: prompt, maxTokens: budget)))
        await stack.engine.shutdown()

        XCTAssertEqual(second.finishReason, .length)
        XCTAssertEqual(second.usage?.prefixCacheMatchedTokens, 72)
        XCTAssertEqual(second.usage?.prefixCachePrefillTokensSaved, 40)
        XCTAssertEqual(second.usage?.prefixCacheStrategy, .frozenFullReplay)
        XCTAssertEqual(second.usage?.prefixCacheReplayTokens, 32)
        XCTAssertEqual(second.usage?.prefixCacheBoundarySplits, 1)
        XCTAssertEqual(
            second.tokens, first.tokens,
            "frozen-full hybrid replay must remain target-token exact")
        XCTAssertEqual(second.text, first.text)
    }

    /// WS-4.1. The paged twin of `testStackedSlidingFrozenFullReplayIsToken
    /// Exact` above, on the same layout and prompt: paged used to fail cold on
    /// an interleaved hybrid (a since-deleted "requires a dual cursor"
    /// capability refusal) and now serves
    /// it end to end through the production engine, donation queue and
    /// PrefixCacheV2 — a real hit, token-identical to the cold run.
    ///
    /// The numbers are now IDENTICAL to the contiguous twin's, and that is
    /// what this test is for. They used to differ by exactly one window:
    /// `PagedLayerCache.prefillKV` attended `gather ++ chunk` with the chunk
    /// half freshly projected, where `CBv2FrozenReplayFullSequenceKV` hands
    /// back the cached keys, so a frozen paged row needed an extra window of
    /// replay — replayBound 48, saving 72 - 48 = 24, against contiguous's 32
    /// and 40. `prefillKVWritingChunk` reads the cached diagonal out of the
    /// frozen pages now, so both arms run M = 72, R = 32, C = 40 and save 40:
    ///   contiguous  replayBound 2 x 16 = 32, saves 72 - 32 = 40
    ///   paged       replayBound 2 x 16 = 32, saves 72 - 32 = 40
    /// This is the end-to-end statement of paged/contiguous parity on prefix
    /// reuse. If these numbers ever diverge from the twin above again, a
    /// paged-specific replay term has come back.
    ///
    /// The zero-replay form (every sliding row restored EXACTLY at M from a
    /// `CBv2PagedWindowSnapshot`) is covered at the backend level by
    /// `CBv2PrefixReusePagedFrozenFullTests`. It cannot be exercised from here
    /// because nothing in the engine donates a window: `PrefixCacheV2
    /// .isCacheable` nils every sliding layer, and the payload comes from the
    /// provider's per-block window sidecar.
    func testStackedSlidingPagedBackendServesFrozenFullReplay() async throws {
        let model = TinyTestModel.make(
            seed: 0xD00D_F00D,
            headDim: 64,
            stackedSlidingFull: true)
        let prompt = makePromptTokens(length: 73, seed: 0x5EED)
        let stack = try makeStack(.paged, model: model, enablePrefixCache: true)
        XCTAssertTrue(stack.engine.prefixReuseCapability.isSupported)
        XCTAssertEqual(stack.engine.prefixReuseCapability.strategy, .frozenFullReplay)
        XCTAssertEqual(stack.engine.prefixReuseCapability.conservativeReplayBoundTokens, 32)

        let first = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 1, prompt: prompt, maxTokens: 8)))
        XCTAssertEqual(first.usage?.prefixCacheHitTokens, 0, "first run is a miss")
        let donated = await cbv2SchedWait { stack.prefixCache.stats().entryCount >= 1 }
        XCTAssertTrue(donated, "a supported capability must arm donation")
        let second = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 2, prompt: prompt, maxTokens: 8)))
        await stack.engine.shutdown()

        XCTAssertEqual(second.usage?.prefixCacheMatchedTokens, 72)
        XCTAssertEqual(second.usage?.prefixCacheStrategy, .frozenFullReplay)
        XCTAssertEqual(second.usage?.prefixCacheReplayTokens, 32)
        XCTAssertEqual(second.usage?.prefixCachePrefillTokensSaved, 40)
        XCTAssertEqual(
            second.tokens, first.tokens,
            "paged frozen-full replay must be target-token exact")
        XCTAssertEqual(second.text, first.text)
    }

    // MARK: (iv-b) Per-request cache salt isolation (TB-007)

    /// Requests carrying different `cacheSalt`s must never share cached KV
    /// (in either direction, nil included), while the same salt round-trips
    /// with a hit — end to end through the real engine, donation queue, and
    /// PrefixCacheV2.
    func testPrefixCacheSaltIsolation_Contiguous() async throws {
        let model = makeModel(.contiguous)
        let prompt = makePromptTokens(length: 41, seed: 99)
        let budget = 8
        let expectedHit = 5 * blockSize - window  // 24, as in the round trip

        let stack = try makeStack(.contiguous, model: model, enablePrefixCache: true)
        func salted(_ id: UInt64, _ salt: String?) -> CBv2Request {
            var request = greedyRequest(id: id, prompt: prompt, maxTokens: budget)
            request.cacheSalt = salt
            return request
        }

        let first = await cbv2SchedCollect(try stack.engine.submit(salted(1, "tenantA")))
        XCTAssertEqual(first.finishReason, .length)
        XCTAssertEqual(first.usage?.prefixCacheHitTokens, 0, "first run is a miss")
        let donated = await cbv2SchedWait { stack.prefixCache.stats().entryCount >= 1 }
        XCTAssertTrue(donated, "finished request must donate under its salt")

        // Different salt and nil salt: no cross-hit, greedy output unchanged.
        let otherTenant = await cbv2SchedCollect(try stack.engine.submit(salted(2, "tenantB")))
        XCTAssertEqual(
            otherTenant.usage?.prefixCacheHitTokens, 0,
            "a different salt must never adopt tenantA's KV")
        XCTAssertEqual(otherTenant.tokens, first.tokens)

        let unsalted = await cbv2SchedCollect(try stack.engine.submit(salted(3, nil)))
        XCTAssertEqual(
            unsalted.usage?.prefixCacheHitTokens, 0,
            "a nil-salt request must never adopt salted KV")
        XCTAssertEqual(unsalted.tokens, first.tokens)

        // Same salt: hits, token-exact.
        let second = await cbv2SchedCollect(try stack.engine.submit(salted(4, "tenantA")))
        await stack.engine.shutdown()
        XCTAssertEqual(second.usage?.prefixCacheHitTokens, expectedHit, "same salt must hit")
        XCTAssertEqual(second.tokens, first.tokens)
    }

    // MARK: (v) Backend / prefix-cache pairing guard

    /// `EngineV2.init` preconditions on `prefixCachePairingViolation`: a
    /// backend whose donated snapshots reference recyclable storage (paged
    /// slabs) must never feed a prefix cache with `materializeOnDonate:
    /// false`. Asserted through the internal validator (a precondition is
    /// not catchable in-process); construction with a safe pairing is also
    /// exercised for real.
    func testPagedBackendRejectsNonMaterializingPrefixCache() throws {
        let model = makeModel(.paged)
        let paged: PagedKVBackend
        do {
            paged = try PagedKVBackend(
                layerKinds: model.layerKinds,
                config: PagedKVPoolConfig(
                    capacityBytes: 64 << 20, maxPrefillChunk: 64,
                    nominalMaxSequenceLength: 512))
        } catch let error as CBv2KVError {
            throw XCTSkip("paged backend unavailable on this hardware: \(error)")
        }
        let contiguous = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 24))
        XCTAssertTrue(paged.requiresMaterializedSnapshots)
        XCTAssertFalse(contiguous.requiresMaterializedSnapshots)

        let unsafeCache = PrefixCacheV2(
            config: .init(blockSize: blockSize, materializeOnDonate: false))
        let safeCache = PrefixCacheV2(config: .init(blockSize: blockSize))  // default: true

        XCTAssertNotNil(
            EngineV2.prefixCachePairingViolation(backend: paged, prefixCache: unsafeCache),
            "paged + materializeOnDonate:false must be rejected at construction")
        XCTAssertNil(
            EngineV2.prefixCachePairingViolation(backend: paged, prefixCache: safeCache))
        XCTAssertNil(EngineV2.prefixCachePairingViolation(backend: paged, prefixCache: nil))
        XCTAssertNil(
            EngineV2.prefixCachePairingViolation(backend: contiguous, prefixCache: unsafeCache),
            "contiguous snapshot views own their buffers via ARC — opt-out stays legal")

        // A safe pairing constructs and shuts down cleanly.
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: paged.makeLayerCaches()),
            sampler: CBv2DefaultSampler(fallbackSeed: 11),
            detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: CBv2E2ETokenizer()),
            schedulerConfig: CBv2SchedulerConfig(enablePrefixCache: true),
            prefixCache: safeCache)
        let drained = expectation(description: "shutdown")
        Task {
            await engine.shutdown()
            drained.fulfill()
        }
        wait(for: [drained], timeout: 10)
    }

    // MARK: (vi) Terminal pool exhaustion — canonical capacity finish

    /// Two same-step-admissible requests race for a pool sized to hold one:
    /// both pass `AdmissionV2.canEverFit` (each fits alone; the ledger
    /// reserves chunk-wise, never the worst case), the winner's
    /// `makeSequenceState` charges its worst-case pages atomically, and the
    /// loser bounces `requeueOnCapacity` until `maxCapacityRequeues` trips.
    /// The terminal finish must carry
    /// `CBv2KVError.capacityExhaustedFinishPrefix` — bridges key their
    /// retryable capacity error (429-class, never a 5xx) off that prefix.
    func testPagedPoolExhaustionFinishesWithCanonicalCapacityError() async throws {
        let model = makeModel(.paged)
        // One shared page group (both layers are (kvHeads 2, headDim 64)).
        // Per-request worst case at maxLength 308: ceil(308/16)=20 full
        // pages + ring ceil((16+64)/16)+1=6 pages = 26 pages × 8 KiB
        // ≈ 208 KiB. 256 KiB holds one request, not two; the byte ledger
        // (~162 KiB estimate) admits each individually.
        let paged: PagedKVBackend
        do {
            paged = try PagedKVBackend(
                layerKinds: model.layerKinds,
                config: PagedKVPoolConfig(
                    capacityBytes: 256 << 10,
                    maxPrefillChunk: 64,
                    nominalMaxSequenceLength: 512))
        } catch let error as CBv2KVError {
            throw XCTSkip("paged backend unavailable on this hardware: \(error)")
        }
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: paged.makeLayerCaches()),
            sampler: CBv2DefaultSampler(fallbackSeed: 11),
            detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: CBv2E2ETokenizer()),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16,
                enablePrefixCache: false),
            prefixCache: nil)

        // Winner: decode long enough (300 steps) to outlive the loser's
        // 64 requeue attempts (~1 per step). Wait for its first delta so
        // its worst-case pages are charged before the loser submits.
        let winnerStream = try engine.submit(
            greedyRequest(id: 7001, prompt: makePromptTokens(length: 8, seed: 71), maxTokens: 300))
        var winnerIter = winnerStream.makeAsyncIterator()
        var winnerFinish: CBv2FinishReason?
        var winnerStarted = false
        while let event = await winnerIter.next() {
            if case .delta = event {
                winnerStarted = true
                break
            }
            if case .finished(let reason, _) = event {
                winnerFinish = reason
                break
            }
        }
        XCTAssertTrue(winnerStarted, "winner must decode, got \(String(describing: winnerFinish))")

        // Pool-truth surfaces through the engine's capacity snapshot: the
        // winner's atomic worst-case page charge is RESERVED (not yet all
        // in-use — pages materialize lazily as tokens are written), and
        // the backend's physical capacity rides next to the admission
        // ceiling (they diverge on paged re-slices).
        let snapshot = engine.capacity()
        XCTAssertEqual(snapshot.kvBytesReserved, paged.bytesReserved)
        XCTAssertGreaterThan(snapshot.kvBytesReserved, 0)
        XCTAssertEqual(snapshot.kvBytesBackendCapacity, paged.bytesCapacity)
        XCTAssertLessThanOrEqual(
            snapshot.kvBytesReserved, snapshot.kvBytesBackendCapacity)

        let loser = await cbv2SchedCollect(
            try engine.submit(
                greedyRequest(
                    id: 7002, prompt: makePromptTokens(length: 8, seed: 72), maxTokens: 300)),
            timeoutSeconds: 120)
        guard case .error(let message) = loser.finishReason else {
            await engine.shutdown()
            return XCTFail(
                "expected terminal capacity error, got \(String(describing: loser.finishReason))")
        }
        XCTAssertTrue(
            message.hasPrefix(CBv2KVError.capacityExhaustedFinishPrefix),
            "terminal capacity finish must carry the canonical prefix, got: \(message)")

        // The winner is unaffected by the loser's rejection: it finishes
        // by length with its pages intact.
        while let event = await winnerIter.next() {
            if case .finished(let reason, _) = event {
                winnerFinish = reason
                break
            }
        }
        XCTAssertEqual(winnerFinish, .length)
        await engine.shutdown()
    }

    // MARK: (vii) Zero re-slice survives step publishes (paged)

    /// A ledger re-slice to ZERO on the paged backend must hold through
    /// step publishes: the INSTALLED admission ledger is authoritative
    /// even at 0, while the unchanged physical pool rides
    /// `kvBytesBackendCapacity`. The gauge must never overwrite a zeroed
    /// ceiling with pool truth and re-advertise capacity that submit-time
    /// admission rejects.
    func testPagedZeroResliceSurvivesStepPublish() async throws {
        let model = makeModel(.paged)
        let stack = try makeStack(.paged, model: model, enablePrefixCache: false)
        let requestID: UInt64 = 8801
        let stream = try stack.engine.submit(
            greedyRequest(
                id: requestID, prompt: makePromptTokens(length: 8, seed: 81), maxTokens: 120))
        var iterator = stream.makeAsyncIterator()
        var started = false
        while let event = await iterator.next() {
            if case .delta = event {
                started = true
                break
            }
            if case .finished = event { break }
        }
        XCTAssertTrue(started, "the in-flight request must be decoding (steps publishing)")

        stack.engine.updateKVBytesCapacity(0)
        // The in-flight request keeps decoding (reservations untouched),
        // so steps keep publishing — the settled snapshot must carry the
        // ledger's zero next to the pool's physical truth.
        let settled = await cbv2SchedWait {
            let snapshot = stack.engine.capacity()
            return snapshot.kvBytesCapacity == 0 && snapshot.kvBytesBackendCapacity > 0
        }
        XCTAssertTrue(settled, "a zero re-slice must not be overwritten by pool truth")

        stack.engine.cancel(CBv2RequestID(requestID))
        while let event = await iterator.next() {
            if case .finished = event { break }
        }
        await stack.engine.shutdown()
    }
}
