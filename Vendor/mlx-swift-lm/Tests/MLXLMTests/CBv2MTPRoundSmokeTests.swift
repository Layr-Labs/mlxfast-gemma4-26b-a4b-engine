// CBv2MTPRoundSmokeTests.swift
//
// MTP round-driver smoke tests through the REAL engine: EngineV2 over a
// tiny random-init Gemma-4 target + assistant drafter (the same weight-free
// fixtures as CBv2MTPModelSeamTests), contiguous KV backend,
// CBv2LayerCacheBank, CBv2DefaultSampler. Compiled decode is disabled so
// both legs run the eager paths the MTP round reuses.
//
// The acceptance invariant is GREEDY LOSSLESSNESS: an MTP-on engine emits
// token-exactly what an MTP-off engine emits, for every scenario —
// including sliding-window wrap during rounds (window 16 << generation
// length), mixed batches (verify rows + prefill neighbors), mid-round
// stop-token and maxTokens truncation, and batched [2, 1+k] verify rounds.
// Plus liveness: a cancelled MTP request must not leak pendingSamples
// (a leak blocks the ENTIRE waiting-admission loop, so a follow-up request
// would never complete).

import Foundation
import MLX
@testable import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXLLM

@Suite("CBv2MTPRoundSmoke", .serialized)
struct CBv2MTPRoundSmokeTests {

    private let vocabSize = 256
    private let hiddenSize = 64
    /// Small window so decode + verify rounds wrap the sliding ring early.
    private let slidingWindow = 16

    // MARK: - Fixtures (mirrors CBv2MTPModelSeamTests)

    /// 6-layer target, last 2 KV-shared; capture layers full=2, sliding=3.
    private func targetConfig(
        tieWordEmbeddings: Bool = true
    ) throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 6,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "full_attention", "sliding_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": \(slidingWindow),
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": \(tieWordEmbeddings),
                "vocab_size": \(vocabSize),
                "vocab_size_per_layer_input": \(vocabSize),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// 2-layer fully-KV-shared drafter matching the target's hidden/vocab.
    private func drafterConfig() throws -> Gemma4AssistantConfiguration {
        let json = """
            {
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": \(hiddenSize),
                "use_ordered_embeddings": false,
                "num_centroids": 16,
                "centroid_intermediate_top_k": 4,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "head_dim": 32,
                    "global_head_dim": 32,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 2,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "sliding_window": \(slidingWindow),
                    "final_logit_softcapping": null,
                    "tie_word_embeddings": true,
                    "vocab_size": \(vocabSize),
                    "vocab_size_per_layer_input": \(vocabSize),
                    "rms_norm_eps": 1e-6,
                    "hidden_size_per_layer_input": 0,
                    "use_double_wide_mlp": false
                }
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(json.utf8))
    }

    private struct Fixture {
        let target: Gemma4TextModel
        let drafter: Gemma4AssistantDraftModel
    }

    private func makeFixture(
        seed: UInt64 = 0x5EED, deterministicTarget: Bool = false
    ) throws -> Fixture {
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(
            try targetConfig(tieWordEmbeddings: !deterministicTarget))
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        if deterministicTarget { stabilizeCBv2MTPGreedyCycleTarget(target) }
        eval(target, drafter)
        return Fixture(target: target, drafter: drafter)
    }

    /// The real engine over the fixture. `mtp: false` builds the identical
    /// engine WITHOUT a drafter (the MTP-off baseline). Compiled decode is
    /// off: MTP never touches it, and the parity legs should compare the
    /// same eager paths.
    private func makeEngine(
        _ fixture: Fixture, mtp: Bool, maxDraftTokens: Int = 2, maxSpeculativeBatch: Int = 2,
        maxConcurrent: Int = 4,
        verificationMode: CBv2MTPVerificationMode = .serialTarget
    ) throws -> EngineV2 {
        let kinds = fixture.target.cbv2LayerKinds
        let mtpDrafter: Gemma4CBv2MTPDrafter? =
            mtp
            ? try Gemma4CBv2MTPDrafter(drafter: fixture.drafter, target: fixture.target)
            : nil
        let mtpConfig = CBv2MTPConfig(
            enabled: mtp, maxDraftTokens: maxDraftTokens,
            maxSpeculativeBatch: maxSpeculativeBatch,
            fixedDraftTokens: maxDraftTokens,
            verificationMode: verificationMode,
            maxAutomaticRectangularTokens: 8)
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(fixture.target),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            mtpDrafter: mtpDrafter,
            mtpConfig: mtpConfig)
    }

    private func greedyRequest(
        id: UInt64, prompt: [Int], maxTokens: Int, stopTokens: Set<Int> = [],
        stopStrings: [String] = [], temperature: Float = 0
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: temperature),
            maxTokens: maxTokens, stopTokens: stopTokens,
            stopStrings: stopStrings)
    }

    private func run(
        _ engine: EngineV2, _ request: CBv2Request
    ) async throws -> CBv2SchedCollected {
        await cbv2SchedCollect(try engine.submit(request))
    }

    // MARK: - (1) Solo greedy parity through window wraps + metrics sanity

    @Test func soloGreedyTokenExactWithWindowWrap() async throws {
        let fixture = try makeFixture()
        // Prompt 24 > window 16 and 40 generated tokens: the sliding ring
        // wraps repeatedly THROUGH verify rounds (staged-write edge).
        let prompt = makePromptTokens(length: 24, seed: 11, vocabSize: vocabSize)

        let off = try makeEngine(fixture, mtp: false)
        let baseline = try await run(off, greedyRequest(id: 1, prompt: prompt, maxTokens: 40))
        await off.shutdown()
        #expect(baseline.finishReason == .length)
        #expect(baseline.tokens.count == 40)
        #expect(off.mtpMetricsSnapshot() == nil, "MTP-off engine must report no MTP state")

        let on = try makeEngine(fixture, mtp: true, verificationMode: .automatic)
        let speculative = try await run(on, greedyRequest(id: 1, prompt: prompt, maxTokens: 40))
        let metrics = try #require(on.mtpMetricsSnapshot())
        await on.shutdown()

        #expect(speculative.finishReason == .length)
        #expect(
            speculative.tokens == baseline.tokens,
            "MTP-on output diverged: on=\(speculative.tokens) off=\(baseline.tokens)")

        // The round loop really ran: one seed step to establish the carry,
        // then rounds emitting 1..1+k tokens each.
        #expect(metrics.seedSteps >= 1)
        #expect(metrics.rounds >= 1)
        // Terminal-depth clamping may run the final rounds at k=1 even though
        // the configured ceiling is k=2.
        #expect(metrics.draftedTokens >= metrics.rounds)
        #expect(metrics.draftedTokens <= metrics.rounds * 2)
        #expect(metrics.controllerFallbacks["tail_depth", default: 0] > 0)
        #expect(metrics.emittedTokens >= metrics.rounds)
        #expect(metrics.emittedTokens <= metrics.rounds * 3)
        #expect(metrics.acceptedTokens <= metrics.draftedTokens)
        // Per-position acceptance is monotonically non-increasing.
        if metrics.perPositionAccepted.count == 2 {
            #expect(metrics.perPositionAccepted[0] >= metrics.perPositionAccepted[1])
        }
        // Every generated token comes from the prefill bonus (1), seed
        // steps (1 each), round emissions, or plain decode steps (near the
        // length cap / after a carry loss) — never more than the total.
        #expect(1 + metrics.seedSteps + metrics.emittedTokens <= 40)
    }

    // MARK: - (2) Mixed batch: verify rows + a prefilling neighbor

    @Test func mixedBatchWithPrefillNeighborStaysTokenExact() async throws {
        let fixture = try makeFixture()
        let promptA = makePromptTokens(length: 20, seed: 21, vocabSize: vocabSize)
        // Long prompt = several [1, 16] prefill chunks riding beside A's
        // seed/verify steps.
        let promptB = makePromptTokens(length: 44, seed: 22, vocabSize: vocabSize)

        var baselines: [CBv2SchedCollected] = []
        for (index, (prompt, maxTokens)) in [(promptA, 28), (promptB, 20)].enumerated() {
            let off = try makeEngine(fixture, mtp: false)
            baselines.append(
                try await run(
                    off, greedyRequest(id: UInt64(index + 1), prompt: prompt, maxTokens: maxTokens))
            )
            await off.shutdown()
        }

        let on = try makeEngine(fixture, mtp: true)
        async let a = run(on, greedyRequest(id: 1, prompt: promptA, maxTokens: 28))
        // Give A a head start so it is decoding (seeding/rounds) while B
        // prefills its chunks — the mixed-plan shape the driver must handle.
        try await Task.sleep(nanoseconds: 100_000_000)
        async let b = run(on, greedyRequest(id: 2, prompt: promptB, maxTokens: 20))
        let (collectedA, collectedB) = try await (a, b)
        await on.shutdown()

        #expect(collectedA.tokens == baselines[0].tokens, "row A diverged in the mixed batch")
        #expect(collectedB.tokens == baselines[1].tokens, "row B diverged in the mixed batch")
    }

    // MARK: - (3) Two verify rows in one rectangular round

    @Test func twoSpeculatingRowsStayTokenExact() async throws {
        let fixture = try makeFixture(deterministicTarget: true)
        let promptA = makePromptTokens(length: 12, seed: 31, vocabSize: vocabSize)
        let promptB = makePromptTokens(length: 18, seed: 32, vocabSize: vocabSize)

        var baselines: [CBv2SchedCollected] = []
        for (index, prompt) in [promptA, promptB].enumerated() {
            let off = try makeEngine(fixture, mtp: false)
            baselines.append(
                try await run(off, greedyRequest(id: UInt64(index + 1), prompt: prompt, maxTokens: 24)))
            await off.shutdown()
        }
        let expectedA = cbv2MTPExpectedGreedyCycle(
            after: promptA.last!, count: 24, vocabularySize: vocabSize)
        let expectedB = cbv2MTPExpectedGreedyCycle(
            after: promptB.last!, count: 24, vocabularySize: vocabSize)
        #expect(baselines[0].tokens == expectedA)
        #expect(baselines[1].tokens == expectedB)

        // Both rows greedy and running (batch gate 2): exercise the universal
        // serial target verifier and the explicit rectangular optimization.
        for mode in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
            let on = try makeEngine(fixture, mtp: true, verificationMode: mode)
            async let a = run(on, greedyRequest(id: 1, prompt: promptA, maxTokens: 24))
            async let b = run(on, greedyRequest(id: 2, prompt: promptB, maxTokens: 24))
            let (collectedA, collectedB) = try await (a, b)
            let metrics = try #require(on.mtpMetricsSnapshot())
            await on.shutdown()

            #expect(
                collectedA.tokens == baselines[0].tokens,
                "row A diverged in \(mode.rawValue) mode")
            #expect(
                collectedB.tokens == baselines[1].tokens,
                "row B diverged in \(mode.rawValue) mode")
            #expect(metrics.rounds >= 1)
        }
    }

    // MARK: - (4) Mid-round stop-token truncation

    @Test func stopTokenMidRoundTruncatesExactly() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 20, seed: 41, vocabSize: vocabSize)

        // Baseline without stops, to pick a token the stream really emits
        // somewhere mid-generation.
        let probe = try makeEngine(fixture, mtp: false)
        let unstopped = try await run(probe, greedyRequest(id: 9, prompt: prompt, maxTokens: 32))
        await probe.shutdown()
        try #require(unstopped.tokens.count == 32)
        // A token from the middle of the stream: rounds emit up to 3
        // tokens, so an odd index regularly lands mid-round.
        let stopToken = unstopped.tokens[17]
        let stops: Set<Int> = [stopToken]

        let off = try makeEngine(fixture, mtp: false)
        let baseline = try await run(
            off, greedyRequest(id: 1, prompt: prompt, maxTokens: 32, stopTokens: stops))
        await off.shutdown()
        #expect(baseline.finishReason == .stop)

        let on = try makeEngine(fixture, mtp: true)
        let speculative = try await run(
            on, greedyRequest(id: 1, prompt: prompt, maxTokens: 32, stopTokens: stops))
        await on.shutdown()

        #expect(speculative.finishReason == .stop)
        #expect(
            speculative.tokens == baseline.tokens,
            "stop-token truncation diverged: on=\(speculative.tokens) off=\(baseline.tokens)")
    }

    // MARK: - (5) maxTokens truncation lands exactly

    @Test func maxTokensTruncationStaysExact() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 20, seed: 51, vocabSize: vocabSize)
        // Odd cap: with k=2 the last round would overshoot without the
        // mid-scan maxTokens truncation + planner max_tokens clamp.
        for maxTokens in [3, 7] {
            let off = try makeEngine(fixture, mtp: false)
            let baseline = try await run(
                off, greedyRequest(id: 1, prompt: prompt, maxTokens: maxTokens))
            await off.shutdown()

            let on = try makeEngine(fixture, mtp: true)
            let speculative = try await run(
                on, greedyRequest(id: 1, prompt: prompt, maxTokens: maxTokens))
            await on.shutdown()

            #expect(baseline.finishReason == .length)
            #expect(speculative.finishReason == .length)
            #expect(speculative.tokens.count == maxTokens)
            #expect(
                speculative.tokens == baseline.tokens,
                "maxTokens=\(maxTokens) diverged: on=\(speculative.tokens) off=\(baseline.tokens)")
        }
    }

    // MARK: - (6) Non-greedy rows never speculate

    @Test func temperatureRowsNeverSpeculate() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 16, seed: 61, vocabSize: vocabSize)

        let on = try makeEngine(fixture, mtp: true)
        let collected = try await run(
            on, greedyRequest(id: 1, prompt: prompt, maxTokens: 12, temperature: 0.7))
        let metrics = try #require(on.mtpMetricsSnapshot())
        await on.shutdown()

        #expect(collected.finishReason == .length)
        #expect(collected.tokens.count == 12)
        #expect(metrics.rounds == 0, "temperature>0 row must never draft")
        #expect(metrics.seedSteps == 0, "temperature>0 row must never seed")
    }

    @Test func stopStringRowsFailOpenToPlainDecode() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 16, seed: 62, vocabSize: vocabSize)
        let engine = try makeEngine(fixture, mtp: true)
        let collected = try await run(
            engine,
            greedyRequest(
                id: 1, prompt: prompt, maxTokens: 8,
                stopStrings: ["never-matched-by-null-detokenizer"]))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(collected.finishReason == .length)
        #expect(metrics.rounds == 0)
        #expect(metrics.seedSteps == 0)
    }

    // MARK: - (7) Cancel mid-generation leaks nothing (admission liveness)

    @Test func cancelMidRoundDoesNotBlockAdmission() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 20, seed: 71, vocabSize: vocabSize)

        let on = try makeEngine(fixture, mtp: true, verificationMode: .automatic)
        let victim = greedyRequest(id: 1, prompt: prompt, maxTokens: 512)
        let stream = try on.submit(victim)
        // Let it get well into MTP rounds, then cancel mid-flight.
        var seen = 0
        var finish: CBv2FinishReason?
        for await event in stream {
            switch event {
            case .delta(_, let tokens, _):
                seen += tokens.count
                if seen >= 8 { on.cancel(victim.id) }
            case .finished(let reason, _):
                finish = reason
            }
        }
        #expect(finish == .cancelled)

        // A leaked pendingSamples would wedge the waiting-admission loop:
        // this follow-up request must run to completion.
        let follower = try await run(on, greedyRequest(id: 2, prompt: prompt, maxTokens: 8))
        await on.shutdown()
        #expect(follower.finishReason == .length)
        #expect(follower.tokens.count == 8)
    }

    @Test func requestIDReuseAndShutdownClearMTPState() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 20, seed: 72, vocabSize: vocabSize)
        let engine = try makeEngine(fixture, mtp: true)
        let reusedID = CBv2RequestID(44)

        let first = try engine.submit(
            CBv2Request(
                id: reusedID, promptTokens: prompt,
                sampling: .init(temperature: 0), maxTokens: 256))
        var iterator = first.makeAsyncIterator()
        var seen = 0
        while let event = await iterator.next() {
            switch event {
            case .delta(_, let tokens, _):
                seen += tokens.count
                if seen >= 8 { engine.cancel(reusedID) }
            case .finished(let reason, _):
                #expect(reason == .cancelled)
                break
            }
            if seen >= 8, engine.capacity().activeRequests == 0 { break }
        }

        let reused = try await run(
            engine,
            CBv2Request(
                id: reusedID, promptTokens: prompt,
                sampling: .init(temperature: 0), maxTokens: 8))
        #expect(reused.finishReason == .length)
        await engine.shutdown()
        #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
    }

    @Test func fixedDepthZeroProbesOnceThenKeepsNormalChaining() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 16, seed: 73, vocabSize: vocabSize)
        let engine = try makeEngine(fixture, mtp: true, maxDraftTokens: 0)
        let result = try await run(
            engine, greedyRequest(id: 1, prompt: prompt, maxTokens: 20))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        let baseline = try #require(
            metrics.costInputs.first {
                $0.decodeRowBucket == 1 && $0.depth == 0
            })
        await engine.shutdown()

        #expect(result.finishReason == .length)
        #expect(engine.chainedStepCount > 0)
        #expect(baseline.samples == 1)
        #expect(metrics.rounds == 0)
    }
}
