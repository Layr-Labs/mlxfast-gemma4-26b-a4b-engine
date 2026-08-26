// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import Testing

@Suite("Gemma4TextModelInner shared-KV capture hook")
struct Gemma4CaptureHookTests {

    private func smallTargetConfig() throws -> Gemma4TextConfiguration {
        // 10-layer config with the last 5 kv-shared. First 5 are pattern
        // [sliding, sliding, sliding, sliding, full] — matching the real
        // model's slidingWindowPattern = 5. After derivation, layer_types
        // should be [sliding, sliding, sliding, sliding, full,
        //            sliding, sliding, sliding, sliding, full].
        // Last non-shared full-attn = layer 4; last non-shared sliding = 3.
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 10,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 5,
            "sliding_window": 16,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: data)
    }

    @Test func innerCaptureDirectlyPopulatesBothSlots() throws {
        let config = try smallTargetConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([Int32(1), 2, 3, 4])[.newAxis, .ellipsis]
        let cache = model.newCache(parameters: nil)
        let capture = Gemma4SharedKVCapture()
        _ = model._testCallInner(tokens, cache: cache) { idx, kv in
            if idx == 4 {
                capture.fullAttention = kv
            } else if idx == 3 {
                capture.slidingAttention = kv
            }
        }
        #expect(capture.fullAttention != nil)
        #expect(capture.slidingAttention != nil)
        // Expected shapes for this config: nKVHeads = 1, head_dim = 32,
        // global_head_dim = 32, T = 4.
        #expect(capture.fullAttention?.0.dim(2) == 4)
        #expect(capture.slidingAttention?.0.dim(2) == 4)
    }

    @Test func defaultCaptureIsNil() throws {
        // Regression: confirm existing callers unaffected when no capture is passed.
        let config = try smallTargetConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
        let out = model(tokens, cache: model.newCache(parameters: nil))
        #expect(out.dim(-1) == model.vocabularySize)
    }
}
