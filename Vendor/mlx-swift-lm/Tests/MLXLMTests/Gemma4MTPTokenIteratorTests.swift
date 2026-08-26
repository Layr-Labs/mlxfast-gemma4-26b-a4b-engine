// Copyright © 2026 Apple Inc.
//
// Tests for `Gemma4MTPTokenIterator`. Validates:
//   1. Iterator conforms to `TokenIteratorProtocol` and emits the
//      expected token count.
//   2. Output matches `runGemma4MTPRounds` (the AsyncStream entry point)
//      for the same config — the iterator is a drop-in equivalent.

import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import MLXLLM
import Testing

@Suite("Gemma4MTPTokenIterator", .serialized)
struct Gemma4MTPTokenIteratorTests {

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

    @Test func iteratorEmitsExpectedTokenCount() async throws {
        MLXRandom.seed(42)
        let target = try tinyTarget()
        let drafter = try tinyDrafter()
        eval(target, drafter)

        let promptTokens = MLXArray([Int32](repeating: 7, count: 8))
        let input = LMInput(text: .init(tokens: promptTokens))
        var iter = try Gemma4MTPTokenIterator(
            input: input, target: target, drafter: drafter,
            parameters: GenerateParameters(maxTokens: 16, temperature: 0),
            blockSize: 3)

        var count = 0
        while let _ = iter.next() { count += 1 }
        #expect(count == 16, "iterator must emit exactly maxTokens")
        #expect(iter.tokenCount == 16)
        #expect(iter.promptPrefillTime > 0)
    }

    @Test func iteratorMatchesAsyncStreamPath() async throws {
        MLXRandom.seed(42)
        let target1 = try tinyTarget()
        let drafter1 = try tinyDrafter()
        eval(target1, drafter1)

        // MLXRandom.seed is process-global, while Swift Testing runs suites
        // concurrently. Re-seeding and rebuilding is therefore not a safe
        // way to clone a fixture: another suite can consume/reset the global
        // stream between the two constructions. Build distinct module
        // identities, then copy the already-realized immutable parameters.
        let target2 = try tinyTarget()
        let drafter2 = try tinyDrafter()
        target2.update(parameters: target1.parameters())
        drafter2.update(parameters: drafter1.parameters())
        eval(target2, drafter2)

        #expect(ObjectIdentifier(target1) != ObjectIdentifier(target2))
        #expect(ObjectIdentifier(drafter1) != ObjectIdentifier(drafter2))

        let promptTokens = MLXArray([Int32](repeating: 7, count: 8))
        let iteratorCache = target1.newCache(parameters: nil)
        let streamCache = target2.newCache(parameters: nil)
        #expect(iteratorCache.map(\.offset).allSatisfy { $0 == 0 })
        #expect(streamCache.map(\.offset).allSatisfy { $0 == 0 })

        // Iterator path. Temperature zero makes both the initial bonus and
        // every MTP round greedy, so neither path consumes RNG state.
        let input = LMInput(text: .init(tokens: promptTokens))
        var iter = try Gemma4MTPTokenIterator(
            input: input, target: target1, drafter: drafter1,
            cache: iteratorCache,
            parameters: GenerateParameters(maxTokens: 12, temperature: 0),
            blockSize: 3,
            rngSeed: 42)
        var iteratorTokens: [Int] = []
        while let t = iter.next() { iteratorTokens.append(t) }
        #expect(
            streamCache.map(\.offset).allSatisfy { $0 == 0 },
            "running the iterator must not advance the stream cache")


        // AsyncStream path, starting from an independent reset cache.
        let prefillOut = target2.forwardForMTP(
            promptTokens[.newAxis, .ellipsis], cache: streamCache)
        let firstBonus = Int(prefillOut.logits[0..., -1, 0...]
                                 .asType(.float32).argMax(axis: -1).item(Int32.self))
        let firstHidden = prefillOut.lastHidden[
            0..., -1 ..< prefillOut.lastHidden.dim(1), 0...]
        let stream = try runGemma4MTPRounds(
            target: target2, drafter: drafter2,
            targetCache: streamCache,
            firstBonus: firstBonus, firstHidden: firstHidden,
            firstSharedKV: prefillOut.capturedSharedKV,
            maxTokens: 12, blockSize: 3)
        var streamTokens: [Int] = []
        for await gen in stream {
            if case .chunk(let s) = gen, let t = Int(s) {
                streamTokens.append(t)
            }
        }

        #expect(iteratorTokens.count == 12)
        #expect(streamTokens.count == 12)
        #expect(iteratorCache.count == streamCache.count)
        for (layer, caches) in zip(iteratorCache, streamCache).enumerated() {
            let (iteratorLayer, streamLayer) = caches
            #expect(
                iteratorLayer.offset == streamLayer.offset,
                "cache offset differs at layer \(layer)")
            #expect(
                iteratorLayer.state.count == streamLayer.state.count,
                "cache state arity differs at layer \(layer)")
            for (stateIndex, states) in zip(
                iteratorLayer.state, streamLayer.state
            ).enumerated() {
                let (iteratorState, streamState) = states
                #expect(
                    allClose(iteratorState, streamState, rtol: 0, atol: 0).item(Bool.self),
                    "cache state differs at layer \(layer), state \(stateIndex)")
            }
        }
        #expect(
            iteratorTokens == streamTokens,
            """
            TokenIterator output differs from AsyncStream output — they
            must be identical for cloned model weights, reset caches, the
            same prompt, and deterministic greedy sampling.
              iterator=\(iteratorTokens)
              stream  =\(streamTokens)
            """)
    }
}
