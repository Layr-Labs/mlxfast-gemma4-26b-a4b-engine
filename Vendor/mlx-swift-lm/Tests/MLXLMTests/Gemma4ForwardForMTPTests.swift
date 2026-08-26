// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Gemma4TextModel.forwardForMTP")
struct Gemma4ForwardForMTPTests {

    private func smallConfig() throws -> Gemma4TextConfiguration {
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

    @Test func shapesMatchExpectations() throws {
        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([Int32(1), 2, 3, 4])[.newAxis, .ellipsis]  // [1, 4]
        let cache = model.newCache(parameters: nil)
        let out = model.forwardForMTP(tokens, cache: cache)
        #expect(out.logits.shape == [1, 4, 32])       // [B, L, vocab]
        #expect(out.lastHidden.shape == [1, 4, 64])   // [B, L, hidden]
        // Shared-KV shapes: [B, nKVHeads, T, headDim]
        #expect(out.capturedSharedKV.fullAttention.0.dim(2) == 4)
        #expect(out.capturedSharedKV.slidingAttention.0.dim(2) == 4)
    }

    @Test func logitsRespectSoftcap() throws {
        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([Int32(1), 2, 3, 4])[.newAxis, .ellipsis]
        let cache = model.newCache(parameters: nil)
        let out = model.forwardForMTP(tokens, cache: cache)
        // Softcap 30.0 ⇒ abs(logits) < 30.0 everywhere.
        let absMax = MLX.abs(out.logits).max().item(Float.self)
        #expect(absMax <= 30.0)
    }

    @Test func logitsMatchPlainForward() throws {
        // forwardForMTP.logits must equal callAsFunction output on the
        // same tokens + same fresh cache (byte-equality expected since
        // the head is the factored-out applyLMHead helper in both paths).
        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
        let cache = model.newCache(parameters: nil)
        let out = model.forwardForMTP(tokens, cache: cache)
        let cache2 = model.newCache(parameters: nil)
        let reference = model(tokens, cache: cache2)
        let close = allClose(out.logits, reference, rtol: 1e-5, atol: 1e-5)
        #expect(close.item(Bool.self))
    }
}
