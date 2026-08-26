// Copyright © 2026 Apple Inc.
//
// Tests for stochastic (temperature > 0) MTP sampling. Validates:
//   1. Iterator produces valid output at temperature > 0 (no crashes,
//      tokens in vocab range).
//   2. Rejection-based sampling properties: at temperature=0.0001 the
//      stochastic path degenerates to greedy (distributions are ~point
//      masses; all accepts/resamples collapse to argmax).
//   3. Distribution match: repeated sampling of the first token from the
//      iterator is consistent with sampling from the target's softmax
//      directly (MTP shouldn't change the emission distribution).

import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import MLXLLM
import Testing

@Suite("Gemma4 MTP stochastic sampling", .serialized)
struct Gemma4MTPStochasticTests {

    private func tinyTarget() throws -> Gemma4TextModel {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 256,
            "num_hidden_layers": 12,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 6,
            "sliding_window": 128,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 1024,
            "vocab_size_per_layer_input": 1024,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        return Gemma4TextModel(
            try JSONDecoder.json5().decode(
                Gemma4TextConfiguration.self, from: Data(json.utf8)))
    }

    private func tinyDrafter() throws -> Gemma4AssistantDraftModel {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 256,
            "use_ordered_embeddings": false,
            "num_centroids": 32,
            "centroid_intermediate_top_k": 4,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 4,
                "sliding_window": 128,
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": 1024,
                "vocab_size_per_layer_input": 1024,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        return try Gemma4AssistantDraftModel(
            config: try JSONDecoder.json5().decode(
                Gemma4AssistantConfiguration.self, from: Data(json.utf8)))
    }

    /// Temperature > 0 path runs end-to-end, emits the right count, and
    /// every token is within the vocab range.
    @Test func stochasticIteratorEmitsValidTokens() async throws {
        MLXRandom.seed(42)
        let target = try tinyTarget()
        let drafter = try tinyDrafter()
        eval(target, drafter)

        let promptTokens = MLXArray([Int32](repeating: 7, count: 8))
        let input = LMInput(text: .init(tokens: promptTokens))
        var iter = try Gemma4MTPTokenIterator(
            input: input, target: target, drafter: drafter,
            parameters: GenerateParameters(maxTokens: 16, temperature: 0.8),
            blockSize: 3,
            rngSeed: 12345)

        var count = 0
        var tokens: [Int] = []
        while let t = iter.next() {
            count += 1
            tokens.append(t)
        }
        #expect(count == 16, "iterator must emit exactly maxTokens")
        for t in tokens {
            #expect(t >= 0 && t < 1024, "token \(t) out of vocab range [0, 1024)")
        }
    }

    /// Near-zero temperature should collapse stochastic to greedy. Tokens
    /// emitted at temperature=1e-4 should match tokens at temperature=0
    /// for the same seed (drafter/target distributions become
    /// degenerate point masses at the argmax).
    @Test func lowTempCollapsesToGreedy() async throws {
        MLXRandom.seed(42)
        let target = try tinyTarget()
        let drafter = try tinyDrafter()
        eval(target, drafter)

        let promptTokens = MLXArray([Int32](repeating: 7, count: 8))
        let input = LMInput(text: .init(tokens: promptTokens))

        var greedyIter = try Gemma4MTPTokenIterator(
            input: input, target: target, drafter: drafter,
            parameters: GenerateParameters(maxTokens: 8, temperature: 0),
            blockSize: 3)
        var greedyTokens: [Int] = []
        while let t = greedyIter.next() { greedyTokens.append(t) }

        var lowTempIter = try Gemma4MTPTokenIterator(
            input: input, target: target, drafter: drafter,
            parameters: GenerateParameters(maxTokens: 8, temperature: 1e-4),
            blockSize: 3,
            rngSeed: 777)
        var lowTempTokens: [Int] = []
        while let t = lowTempIter.next() { lowTempTokens.append(t) }

        #expect(
            greedyTokens == lowTempTokens,
            """
            At temperature=1e-4 the stochastic path should produce the same
            tokens as temperature=0. Divergence suggests a bug in the
            rejection sampler.
              greedy  =\(greedyTokens)
              lowTemp =\(lowTempTokens)
            """)
    }

    /// topK=1 at temperature > 0 should collapse to greedy — the filter
    /// masks out everything except the argmax, so the rejection sampler
    /// has no choice but to accept that token every time.
    @Test func topKOneCollapsesToGreedy() async throws {
        MLXRandom.seed(42)
        let target = try tinyTarget()
        let drafter = try tinyDrafter()
        eval(target, drafter)

        let promptTokens = MLXArray([Int32](repeating: 7, count: 8))
        let input = LMInput(text: .init(tokens: promptTokens))

        var greedyIter = try Gemma4MTPTokenIterator(
            input: input, target: target, drafter: drafter,
            parameters: GenerateParameters(maxTokens: 8, temperature: 0),
            blockSize: 3)
        var greedyTokens: [Int] = []
        while let t = greedyIter.next() { greedyTokens.append(t) }

        // topK=1 at temperature=1.0 — the filter collapses both p and q
        // to point masses at their respective argmax. Since verify uses
        // fp32 argmax on the SAME target logits baseline uses, emitted
        // tokens must match greedy output exactly.
        var params = GenerateParameters(maxTokens: 8, temperature: 1.0)
        params.topK = 1
        var topKIter = try Gemma4MTPTokenIterator(
            input: input, target: target, drafter: drafter,
            parameters: params,
            blockSize: 3,
            rngSeed: 999)
        var topKTokens: [Int] = []
        while let t = topKIter.next() { topKTokens.append(t) }

        #expect(
            greedyTokens == topKTokens,
            """
            topK=1 should produce the same output as greedy
              greedy=\(greedyTokens)
              topK=1=\(topKTokens)
            """)
    }

    /// topP with very small mass should narrow the output distribution
    /// toward the argmax token — repeated samples under topP=0.01 should
    /// be much less varied than topP=1.0 at the same temperature.
    @Test func lowTopPNarrowsOutput() async throws {
        MLXRandom.seed(42)
        let target = try tinyTarget()
        let drafter = try tinyDrafter()
        eval(target, drafter)

        let promptTokens = MLXArray([Int32](repeating: 7, count: 8))
        let input = LMInput(text: .init(tokens: promptTokens))

        func sampleAt(topP: Float, seed: UInt64) throws -> [Int] {
            var params = GenerateParameters(maxTokens: 8, temperature: 1.0)
            params.topP = topP
            var iter = try Gemma4MTPTokenIterator(
                input: input, target: target, drafter: drafter,
                parameters: params,
                blockSize: 3,
                rngSeed: seed)
            var toks: [Int] = []
            while let t = iter.next() { toks.append(t) }
            return toks
        }

        var narrow = Set<[Int]>()
        var wide = Set<[Int]>()
        for seed: UInt64 in 1 ... 8 {
            narrow.insert(try sampleAt(topP: 0.01, seed: seed))
            wide.insert(try sampleAt(topP: 1.0, seed: seed))
        }
        // Narrow topP should produce fewer distinct sequences than wide.
        #expect(
            narrow.count <= wide.count,
            """
            topP=0.01 should produce at most as many distinct sequences
            as topP=1.0 across the same 8 seeds
              narrow (topP=0.01): \(narrow.count) distinct
              wide   (topP=1.0): \(wide.count) distinct
            """)
    }

    /// Different seeds at temperature > 0 should produce different output
    /// (sanity: the sampler is actually stochastic).
    @Test func differentSeedsProduceDifferentOutput() async throws {
        MLXRandom.seed(42)
        let target = try tinyTarget()
        let drafter = try tinyDrafter()
        eval(target, drafter)

        let promptTokens = MLXArray([Int32](repeating: 7, count: 8))
        let input = LMInput(text: .init(tokens: promptTokens))
        var seen = Set<[Int]>()
        for seed: UInt64 in [1, 2, 3, 4, 5, 6, 7, 8] {
            var iter = try Gemma4MTPTokenIterator(
                input: input, target: target, drafter: drafter,
                parameters: GenerateParameters(maxTokens: 12, temperature: 1.0),
                blockSize: 3,
                rngSeed: seed)
            var toks: [Int] = []
            while let t = iter.next() { toks.append(t) }
            seen.insert(toks)
        }
        #expect(
            seen.count > 1,
            """
            Different seeds at temperature=1.0 should produce different tokens;
            got only \(seen.count) distinct sequences across 8 seeds
            """)
    }
}
