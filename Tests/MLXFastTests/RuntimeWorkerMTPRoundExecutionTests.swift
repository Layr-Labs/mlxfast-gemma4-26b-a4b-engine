import Foundation
import MLX
import MLXFastCore
@testable import MLXLLM
import MLXLMCommon
import MLXRandom
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Real round-execution coverage for the 2026-08-23 MTP arm increment: drives
// the ACTUAL vendored CBv2 engine (`EngineV2` + `CBv2MTPRoundDriver`) over a
// tiny random-init Gemma 4 target + assistant drafter — the same weight-free
// fixture shape `Vendor/mlx-swift-lm/Tests/MLXLMTests/CBv2MTPRoundSmokeTests
// .swift` uses to prove the vendored driver itself, mined here to prove THIS
// repository's harness-level wiring (`RuntimeWorkerMTPSession`,
// `RuntimeWorkerCohortSession.runMTP`, `assembleMTPCohortFreeRun`) actually
// drives it end to end rather than merely compiling against it.
//
// Gated behind `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` (this repo's convention for
// any test that allocates real MLXArrays / needs `mlx.metallib` — see
// RuntimeWorkerSupportTests.swift) plus `tools/build-mlx-metallib.sh
// --all-build-roots` once per checkout, exactly as every other MLX-runtime
// test here requires. No real weights, no network, no box.

@Suite("RuntimeWorkerMTPRoundExecution", .serialized)
struct RuntimeWorkerMTPRoundExecutionTests {

    private let vocabSize = 64
    private let hiddenSize = 32
    /// Small window so decode + verify rounds wrap the sliding ring early,
    /// exactly like the vendored smoke test's own rationale.
    private let slidingWindow = 12

    @Test
    func cachedDrafterRoPEPreservesSingletonQueryAxis() {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        let hidden = MLXArray((0 ..< 128).map(Float.init)).reshaped([2, 1, 64])
        let table = DrafterRoPETable(
            cos: MLXArray.ones([2, 16], dtype: .float32),
            sin: MLXArray.zeros([2, 16], dtype: .float32),
            dims: 32,
            startPosition: 11,
            windowAhead: 2,
            base: 10_000)

        let rotated = Gemma4CBv2MTPDrafter.applyCachedDrafterRoPE(
            hidden: hidden,
            table: table,
            positionOffset: .batch(MLXArray([Int32(11), Int32(12)])))
        eval(rotated)

        #expect(rotated.shape == [2, 1, 64])
        #expect(allClose(rotated, hidden, rtol: 0, atol: 0).item(Bool.self))
    }

    // MARK: - Fixtures (mirrors CBv2MTPRoundSmokeTests, scaled down further)

    private func targetConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 6,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "full_attention", "sliding_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": \(slidingWindow),
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true,
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

    private func drafterConfig() throws -> Gemma4AssistantConfiguration {
        let json = """
            {
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": \(hiddenSize),
                "use_ordered_embeddings": false,
                "num_centroids": 8,
                "centroid_intermediate_top_k": 4,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 16,
                    "num_hidden_layers": 2,
                    "intermediate_size": 32,
                    "num_attention_heads": 2,
                    "head_dim": 16,
                    "global_head_dim": 16,
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

    private func makeFixture(seed: UInt64) throws -> Fixture {
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(try targetConfig())
        let drafter = try Gemma4AssistantDraftModel(config: try drafterConfig())
        eval(target, drafter)
        return Fixture(target: target, drafter: drafter)
    }

    /// Deterministic pseudo-random prompt, no MLX involved — a plain LCG so
    /// the prompt is reproducible without depending on `MLXRandom`'s own
    /// stream (which the fixture above already consumes).
    private func promptTokens(length: Int, seed: Int) -> [Int] {
        var value = UInt64(bitPattern: Int64(seed)) &+ 0x9E3779B97F4A7C15
        return (0 ..< length).map { _ in
            value = value &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int(value % UInt64(vocabSize))
        }
    }

    private func mtpConfig(maxDraftTokens: Int, maxSpeculativeBatch: Int) -> CBv2MTPConfig {
        CBv2MTPConfig(
            enabled: true, maxDraftTokens: maxDraftTokens,
            maxSpeculativeBatch: maxSpeculativeBatch,
            verificationMode: .automatic,
            // Generous rectangular cap so this fixture's small batch/depth
            // combinations never fall back to the serial oracle for a reason
            // unrelated to what these tests are checking.
            maxAutomaticRectangularTokens: 64)
    }

    // MARK: - (1) Single-stream (v1.1): real rounds, losslessness, the triple

    @Test
    func singleStreamMTPSessionMatchesPlainDecodeAndSatisfiesTheTriple() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        let fixture = try makeFixture(seed: 0x5EED_0001)
        let seed = promptTokens(length: 14, seed: 11)
        let n = 24
        let maxTokens = n + 8

        // Baseline: the SAME fixture through a plain (no drafter) engine —
        // the losslessness reference. `RuntimeWorkerMTPSession.run` always
        // requires an active MTP metrics snapshot (fail-closed — see test
        // (3) below), so a plain baseline uses the width-1 COHORT session's
        // `runSerial` instead, exactly the tool this repo already has for a
        // drafter-free width-1 leg (`serialCohortAssemblyAtWidthOneMatches
        // TheV11SerialBuilder` establishes the same B=1-equals-v1.1
        // equivalence at the pure-assembly layer).
        let offEngine = try Gemma4Runtime.makeCohortEngine(
            model: fixture.target, batchSize: 1, seedTokenCount: seed.count,
            maxTokensPerStream: maxTokens)
        let offSession = try RuntimeWorkerCohortSession(
            engine: offEngine, seedTokensByStream: [seed], maxTokensPerStream: maxTokens)
        let offResult = try offSession.runSerial(targetN: n)
        let baseline = [offSession.seedTokenByStream[0]] + offResult.tokensByStream[0]

        // MTP: the SAME target + drafter instances, bound into a fresh
        // MTP-configured engine (mirrors the vendored smoke test's own
        // sequential on/off engine construction over one fixture).
        let mtpDrafter = try Gemma4CBv2MTPDrafter(drafter: fixture.drafter, target: fixture.target)
        let onEngine = try Gemma4Runtime.makeCohortEngine(
            model: fixture.target, batchSize: 1, seedTokenCount: seed.count,
            maxTokensPerStream: maxTokens,
            mtpDrafter: mtpDrafter,
            mtpConfig: mtpConfig(maxDraftTokens: 3, maxSpeculativeBatch: 2))
        try Gemma4Runtime.requireMTPActive(onEngine)
        let onSession = try RuntimeWorkerMTPSession(
            engine: onEngine, seedTokens: seed, maxTokens: maxTokens, stopTokens: [])
        let onResult = try onSession.run(targetN: n)
        let speculative = [onSession.seedToken] + onResult.tokens

        // GREEDY LOSSLESSNESS: MTP-on output is token-exact vs MTP-off,
        // exactly the invariant the vendored round driver is built to
        // preserve, now proven through THIS harness's own wiring.
        #expect(
            speculative == baseline,
            "mtp-on diverged from serial: on=\(speculative) off=\(baseline)")

        // The §2.6-style consistency triple, on REAL engine output.
        #expect(onResult.tokens.count == n)
        #expect(onResult.committedTotal == n)
        #expect(onResult.acceptanceLengths.reduce(0, +) == n)
        #expect(onResult.completedWork == onResult.rounds + 1)
        #expect(onResult.draftedTotal >= onResult.acceptedTotal)
        #expect(onResult.rounds >= 1)
        // Real drafting actually happened this window — not a silent
        // target-only fallback wearing an mtp label.
        #expect(onResult.draftedTotal > 0)
        #expect(onResult.acceptanceLengths.allSatisfy { $0 >= 1 })
    }

    // MARK: - (2) Batched cohort (v1.2): real rounds, the quadruple

    @Test
    func cohortMTPRunMatchesPlainDecodeAndSatisfiesTheQuadruple() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        let fixture = try makeFixture(seed: 0x5EED_0002)
        let batchSize = 2
        let seeds = (0 ..< batchSize).map { promptTokens(length: 10, seed: 21 + $0) }
        let n = 16
        let maxTokens = n + 8

        // Baseline: the plain serial cohort at the same width — reuses the
        // production `assembleSerialCohortFreeRun` path unchanged.
        let offEngine = try Gemma4Runtime.makeCohortEngine(
            model: fixture.target, batchSize: batchSize, seedTokenCount: seeds[0].count,
            maxTokensPerStream: maxTokens)
        let offSession = try RuntimeWorkerCohortSession(
            engine: offEngine, seedTokensByStream: seeds, maxTokensPerStream: maxTokens)
        let offResult = try offSession.runSerial(targetN: n)
        let baseline = zip(offSession.seedTokenByStream, offResult.tokensByStream)
            .map { [$0] + $1 }

        let mtpDrafter = try Gemma4CBv2MTPDrafter(drafter: fixture.drafter, target: fixture.target)
        let onEngine = try Gemma4Runtime.makeCohortEngine(
            model: fixture.target, batchSize: batchSize, seedTokenCount: seeds[0].count,
            maxTokensPerStream: maxTokens,
            mtpDrafter: mtpDrafter,
            mtpConfig: mtpConfig(maxDraftTokens: 3, maxSpeculativeBatch: batchSize))
        try Gemma4Runtime.requireMTPActive(onEngine)
        let onSession = try RuntimeWorkerCohortSession(
            engine: onEngine, seedTokensByStream: seeds, maxTokensPerStream: maxTokens)
        let onResult = try onSession.runMTP(targetN: n)
        let speculative = zip(onSession.seedTokenByStream, onResult.tokensByStream)
            .map { [$0] + $1 }

        #expect(
            speculative == baseline,
            "mtp-on cohort diverged from serial cohort: on=\(speculative) off=\(baseline)")

        // The consistency QUADRUPLE, on REAL engine output.
        #expect(onResult.tokensByStream.count == batchSize)
        #expect(onResult.tokensByStream.allSatisfy { $0.count == n })
        #expect(onResult.committedTotal == batchSize * n)
        #expect(onResult.acceptanceLengths.reduce(0, +) == n)
        #expect(onResult.completedWork == onResult.rounds + 1)
        #expect(onResult.draftedTotal >= onResult.acceptedTotal)
        #expect(onResult.draftedTotal > 0)
        #expect(onResult.rounds >= 1)
        #expect(onResult.activeStreamsByRound == Array(repeating: batchSize, count: onResult.rounds))
        #expect(onResult.naturalAcceptedByStream.count == batchSize)
        for stream in onResult.naturalAcceptedByStream {
            #expect(stream.count == onResult.rounds)
            for (walked, committed) in zip(stream, onResult.acceptanceLengths) {
                // The benchd-side invariant this repo's own docs cite:
                // `walked >= committed` must hold even for the documented
                // floor value (equality) this harness reports.
                #expect(walked >= committed)
            }
        }
    }

    // MARK: - (3) A resolved-mtp engine that cannot activate refuses loudly

    /// Forwards every `CBv2StepSampler` call to a real `CBv2GreedySampler`
    /// (byte-identical sampling behavior) through a DIFFERENT dynamic type —
    /// `EngineV2.init`'s `samplerSupportsMTP` gate is a TYPE check
    /// (`sampler is CBv2DefaultSampler || sampler is CBv2GreedySampler`), so
    /// this proves the "custom sampler ⇒ MTP inactive" path without needing
    /// a behaviorally different (and therefore non-lossless) sampler.
    private final class PassthroughSampler: CBv2StepSampler {
        private let inner = CBv2GreedySampler()
        var supportsTokenConstraints: Bool { inner.supportsTokenConstraints }
        func sample(
            logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
            stepIndex: Int, pendingSampledTokens: MLXArray?,
            rowContext: () -> [CBv2SamplerRow]
        ) -> MLXArray {
            inner.sample(
                logits: logits, params: params, requestIDs: requestIDs,
                stepIndex: stepIndex, pendingSampledTokens: pendingSampledTokens,
                rowContext: rowContext)
        }
        func takeStepLogprobs() -> CBv2StepLogprobs? { inner.takeStepLogprobs() }
        func requestDidFinish(_ id: CBv2RequestID) { inner.requestDidFinish(id) }
        func confirmSampledTokens(_ tokens: [Int], requestIDs: [CBv2RequestID]) {
            inner.confirmSampledTokens(tokens, requestIDs: requestIDs)
        }
        func tokenConstraintFailure(for id: CBv2RequestID) -> String? {
            inner.tokenConstraintFailure(for: id)
        }
    }

    private func makeEngine(
        fixture: Fixture, sampler: any CBv2StepSampler, mtpDrafter: (any CBv2MTPDrafter)?
    ) throws -> EngineV2 {
        let layerKinds = fixture.target.cbv2LayerKinds
        let caches = try fixture.target.newCacheV2 { index, kind in
            CBv2LayerCache(layerIndex: index, kind: kind)
        }
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(fixture.target),
            layerKinds: layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 24)),
            cacheProvider: CBv2LayerCacheBank(caches: caches),
            sampler: sampler,
            mtpDrafter: mtpDrafter,
            mtpConfig: mtpConfig(maxDraftTokens: 2, maxSpeculativeBatch: 2))
    }

    @Test
    func requireMTPActiveRefusesWhenTheEngineCannotBindMTP() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        let fixture = try makeFixture(seed: 0x5EED_0003)
        let mtpDrafter = try Gemma4CBv2MTPDrafter(drafter: fixture.drafter, target: fixture.target)

        // Positive control: CBv2GreedySampler IS one of the two
        // argmax-equivalent samplers, so this engine activates MTP.
        let activeEngine = try makeEngine(
            fixture: fixture, sampler: CBv2GreedySampler(), mtpDrafter: mtpDrafter)
        #expect(activeEngine.mtpInactiveReason == nil)
        #expect(throws: Never.self) { try Gemma4Runtime.requireMTPActive(activeEngine) }
        await activeEngine.shutdown()

        // Negative control: a same-behavior-but-different-TYPE sampler
        // leaves MTP inactive at the vendored engine's own gate — and this
        // harness's `requireMTPActive` must refuse loudly rather than
        // silently proceed under an mtp label (the same rule the former
        // "round execution not yet wired" refusal enforced).
        let inactiveEngine = try makeEngine(
            fixture: fixture, sampler: PassthroughSampler(), mtpDrafter: mtpDrafter)
        #expect(inactiveEngine.mtpInactiveReason != nil)
        #expect(throws: MLXFastError.self) {
            try Gemma4Runtime.requireMTPActive(inactiveEngine)
        }
        await inactiveEngine.shutdown()
    }
}
