import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXRandom
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Machinery test for the operator-only `width-probe` diagnostic verb
// (Gemma4RuntimeWidthProbe.swift): drives the probe CORE over a weight-free
// MoE fixture and pins the report shapes, the per-layer/router/logits
// capture plumbing, and the teacher-forced window indexing. On laptop
// silicon at fixture scale every family is expected bit-identical (the
// width flip is box physics — the probe's whole purpose is to run ON the
// box); what this test guards is that the probe MACHINERY measures the
// right tensors at the right positions, so the box report can be trusted.

@Suite("WidthProbeMachinery", .serialized)
struct WidthProbeMachineryTests {

    private let vocabSize = 64
    private let hiddenSize = 32
    private let slidingWindow = 12

    private func moeTargetConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 6,
                "intermediate_size": 64,
                "enable_moe_block": true,
                "num_experts": 8,
                "top_k_experts": 2,
                "moe_intermediate_size": 32,
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

    private func promptTokens(length: Int, seed: Int) -> [Int] {
        var value = UInt64(bitPattern: Int64(seed)) &+ 0x9E3779B97F4A7C15
        return (0 ..< length).map { _ in
            value = value &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int(value % UInt64(vocabSize))
        }
    }

    @Test
    func probeCoreProducesAlignedFamiliesOverAnMoEFixture() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        MLXRandom.seed(0xF00D_0001)
        let model = Gemma4TextModel(try moeTargetConfig())
        eval(model)
        let seed = promptTokens(length: 14, seed: 11)

        // Build the fixture's own greedy chain (the tape stand-in), width-1.
        var chain: [Int] = []
        do {
            let cache = model.newCache(parameters: nil)
            var logits = model(
                MLXArray(seed.map(Int32.init)).reshaped([1, seed.count]), cache: cache)
            for _ in 0 ..< 12 {
                let token = logits[0..., -1, 0...].argMax(axis: -1).item(Int.self)
                chain.append(token)
                logits = model(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)
            }
        }
        let tape = WidthProbeTape(seedTokens: seed, chain: chain)

        let result = Gemma4Runtime.runWidthProbeCore(
            model: model, tape: tape, steps: 8, windowWidths: [2, 3], batchWidth: 2,
            logitTopK: 8, relEnvelope: 0.05)
        let families = result.families

        // Three families: b1-width2, b1-width3, batch2-width1.
        #expect(families.count == 3)
        let names = families.compactMap { $0["family"] as? String }
        #expect(names == ["b1-width2", "b1-width3", "batch2-width1"])
        for family in families {
            let positions = try #require(family["positions"] as? [[String: Any]])
            #expect(positions.count == 8)
            // Steps must be aligned 0..7 in order.
            #expect(positions.compactMap { $0["step"] as? Int } == Array(0 ..< 8))
            for position in positions {
                // The capture plumbing produced a real per-position record.
                #expect(position["logit_argmax_ref"] as? Int != nil)
                #expect((position["logit_argmax_ref"] as? Int ?? -1) >= 0)
                #expect(position["logits_bit_equal"] as? Bool != nil)
                // Teacher forcing: the reference argmaxes must reproduce
                // the tape chain (position p's forward consumed chain[p]
                // and predicts chain[p+1]).
                let step = position["step"] as? Int ?? -1
                if step + 1 < tape.chain.count {
                    #expect(
                        position["logit_argmax_ref"] as? Int == tape.chain[step + 1],
                        "reference argmax at step \(step) must reproduce the chain")
                }
                // Phase-1 additive fields: the top-N ranked readout arity
                // matches the requested topk (8, capped at vocab), the depth
                // is at least 1 (top-1 is always within its own envelope), the
                // relative gap at rank 0 is exactly 0, and the provenance flag
                // is the pinned post-softcap choice.
                let relativeGaps = try #require(
                    position["ranked_relative_gaps_ref"] as? [Double])
                #expect(relativeGaps.count == min(8, vocabSize))
                #expect(relativeGaps.first == 0.0)
                let depth = try #require(position["within_envelope_depth"] as? Int)
                #expect(depth >= 1)
                #expect(depth <= relativeGaps.count)
                #expect(position["logit_provenance"] as? String == "post_softcap")
                #expect((position["ranked_tokens_ref"] as? [Int])?.count == min(8, vocabSize))
            }
            // Per-family within-envelope rollup.
            #expect(family["max_within_envelope_depth"] as? Int != nil)
            #expect(family["within_envelope_depth_histogram"] as? [String: Int] != nil)
            // Laptop expectation (documented, not physics-load-bearing):
            // fixture-scale forwards are width-stable here, so the summary
            // machinery should report bit-identity; a false here on a dev
            // Mac is itself a finding worth reporting, not a broken probe.
            #expect(family["bit_identical_everywhere"] as? Bool != nil)
            #expect(family["logit_argmax_flip_steps"] as? [Int] != nil)
            #expect(family["router_selection_flip_steps"] as? [Int] != nil)
        }

        // Per-run reference envelope: the canonical single-copy reference-side
        // characterization, aligned to the same 8 steps, carrying the pinned
        // provenance/topk/rel_envelope and the run-level depth rollup.
        // `referenceEnvelope` is already `[String: Any]` on `WidthProbeResult`
        // (Gemma4RuntimeWidthProbe.swift:442), so a conditional cast here always
        // succeeds and asserts nothing. What is worth asserting is that the
        // envelope was POPULATED, which the cast never checked.
        let envelope = result.referenceEnvelope
        #expect(!envelope.isEmpty, "the probe must publish a reference envelope")
        #expect(envelope["logit_provenance"] as? String == "post_softcap")
        #expect(envelope["topk"] as? Int == 8)
        #expect(envelope["rel_envelope"] as? Double == 0.05)
        let envelopePositions = try #require(envelope["positions"] as? [[String: Any]])
        #expect(envelopePositions.count == 8)
        #expect(envelopePositions.compactMap { $0["step"] as? Int } == Array(0 ..< 8))
        #expect(envelope["max_within_envelope_depth"] as? Int != nil)
        #expect(envelope["within_envelope_depth_histogram"] as? [String: Int] != nil)

        // The MoE fixture must actually exercise the router capture: rerun
        // one captured forward directly and observe events.
        let cache = model.newCache(parameters: nil)
        let prefill = model(
            MLXArray(seed.map(Int32.init)).reshaped([1, seed.count]), cache: cache)
        eval(prefill)
        let capture = capturedForward(
            model: model, tokens: [tape.chain[0]], batch: 1, cache: cache)
        #expect(capture.routerEvents.count == 6, "one router event per MoE layer")
        #expect(capture.layerKV.count == 6, "one K/V capture per layer")
        #expect(capture.logits != nil)
    }

    // GPU-free unit tests over the pure-Swift ranking core (no model, no MLX
    // device): topk readout arity, the relative-ratio computation, the
    // within-envelope-depth counting, and the provenance flag — the Phase-1
    // arithmetic the box run's answer hinges on.

    @Test
    func rankedReadoutArityAndOrderingOverAKnownVector() throws {
        // Descending logits with clear separations. top1 = 10.
        let logits: [Float] = [10, 9.8, 9.5, 5, 4, 1, 0, -3]
        let characterization = rankedReferenceCharacterization(
            logits: logits, logitTopK: 5, relEnvelope: 0.05)
        // Arity is exactly the requested K (<= vocab).
        #expect(characterization.tokens.count == 5)
        #expect(characterization.logits.count == 5)
        #expect(characterization.logitGaps.count == 5)
        #expect(characterization.relativeGaps.count == 5)
        // Rank order: token ids are the descending-logit argsort; here the
        // vector is already descending so the ids are 0,1,2,3,4.
        #expect(characterization.tokens == [0, 1, 2, 3, 4])
        // Logits round-trip through Float32, so compare within tolerance.
        let expectedLogits: [Double] = [10, 9.8, 9.5, 5, 4]
        for (got, want) in zip(characterization.logits, expectedLogits) {
            #expect(abs(got - want) < 1e-5)
        }
        // topk larger than the vector clamps to the vector length.
        let clamped = rankedReferenceCharacterization(
            logits: logits, logitTopK: 100, relEnvelope: 0.05)
        #expect(clamped.tokens.count == logits.count)
    }

    @Test
    func relativeRatioIsGapOverMaxOneAbsTop1() throws {
        // top1 = 20, so denominator = max(1, |20|) = 20. Gaps: 0, 1, 4, 20.
        let logits: [Float] = [20, 19, 16, 0]
        let characterization = rankedReferenceCharacterization(
            logits: logits, logitTopK: 4, relEnvelope: 1.0)
        let expectedGaps: [Double] = [0, 1, 4, 20]
        for (got, want) in zip(characterization.logitGaps, expectedGaps) {
            #expect(abs(got - want) < 1e-5)
        }
        // relative = gap / 20.
        let expected = [0.0, 0.05, 0.2, 1.0]
        for (got, want) in zip(characterization.relativeGaps, expected) {
            #expect(abs(got - want) < 1e-6)
        }
        // The max(1, ...) floor engages when |top1| < 1: denominator = 1.
        let small: [Float] = [0.5, 0.1]
        let smallCharacterization = rankedReferenceCharacterization(
            logits: small, logitTopK: 2, relEnvelope: 1.0)
        #expect(abs(smallCharacterization.relativeGaps[1] - 0.4) < 1e-6)
    }

    @Test
    func withinEnvelopeDepthCountsRanksUnderThreshold() throws {
        // top1 = 100, denominator = 100. Gaps 0,1,3,50 -> relative 0,0.01,0.03,0.5.
        let logits: [Float] = [100, 99, 97, 50]
        // Envelope 0.05 covers ranks {0,1,2} (0, 0.01, 0.03) but not rank 3 (0.5).
        let depth5 = rankedReferenceCharacterization(
            logits: logits, logitTopK: 4, relEnvelope: 0.05)
        #expect(depth5.withinEnvelopeDepth == 3)
        // A tighter envelope (0.02) covers only ranks {0,1}.
        let depth2 = rankedReferenceCharacterization(
            logits: logits, logitTopK: 4, relEnvelope: 0.02)
        #expect(depth2.withinEnvelopeDepth == 2)
        // top-1 is always within its own envelope: depth >= 1 even at 0.
        let depth1 = rankedReferenceCharacterization(
            logits: logits, logitTopK: 4, relEnvelope: 0.0)
        #expect(depth1.withinEnvelopeDepth == 1)
    }

    @Test
    func provenanceFlagAndDepthHistogramAreStable() throws {
        // The provenance constant is the pinned post-softcap choice.
        #expect(widthProbeLogitProvenance == "post_softcap")
        // The histogram tallies depths by decimal-string key.
        let histogram = withinEnvelopeDepthHistogram([1, 1, 3, 3, 3, 2])
        #expect(histogram == ["1": 2, "2": 1, "3": 3])
        #expect(withinEnvelopeDepthHistogram([]).isEmpty)
    }
}
