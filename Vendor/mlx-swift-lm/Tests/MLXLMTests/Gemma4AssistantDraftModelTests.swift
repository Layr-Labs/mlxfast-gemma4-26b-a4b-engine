// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
import Testing

@Suite("Gemma4AssistantDraftModel skeleton")
struct Gemma4AssistantDraftModelTests {

    /// Build a minimal drafter config (4 layers, all kv-shared, tied embeds,
    /// no centroid head).
    private func drafterConfig(
        backbone: Int = 64,
        vocab: Int = 64,
        useOrderedEmbeddings: Bool = false,
        attentionKeqV: Bool = false,
        numGlobalKVHeads: Int? = nil
    ) throws -> Gemma4AssistantConfiguration {
        let kvHeadsField = numGlobalKVHeads.map { "\"num_global_key_value_heads\": \($0)," } ?? ""
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": \(backbone),
            "use_ordered_embeddings": \(useOrderedEmbeddings),
            "num_centroids": 8,
            "centroid_intermediate_top_k": 2,
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
                \(kvHeadsField)
                "sliding_window": 64,
                "attention_k_eq_v": \(attentionKeqV),
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": \(vocab),
                "vocab_size_per_layer_input": \(vocab),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
    }

    /// Build a matching small target config (same hidden_size = backbone,
    /// same vocab).
    private func targetConfig(
        hiddenSize: Int = 64,
        vocab: Int = 64,
        attentionKeqV: Bool = false,
        numGlobalKVHeads: Int? = nil
    ) throws -> Gemma4TextConfiguration {
        let kvHeadsField = numGlobalKVHeads.map { "\"num_global_key_value_heads\": \($0)," } ?? ""
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": \(hiddenSize),
            "num_hidden_layers": 10,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            \(kvHeadsField)
            "num_kv_shared_layers": 5,
            "sliding_window": 64,
            "sliding_window_pattern": 5,
            "attention_k_eq_v": \(attentionKeqV),
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": \(vocab),
            "vocab_size_per_layer_input": \(vocab),
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: data)
    }

    @Test func drafterInitializes() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        #expect(drafter.config.backboneHiddenSize == 64)
        #expect(drafter.maskedEmbedder == nil)     // useOrderedEmbeddings: false
    }

    @Test func drafterWithCentroidHead() throws {
        let cfg = try drafterConfig(useOrderedEmbeddings: true)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        #expect(drafter.maskedEmbedder != nil)
    }

    @Test func bindSucceedsOnMatchedTarget() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let tCfg = try targetConfig()
        let target = Gemma4TextModel(tCfg)
        eval(drafter, target)
        try drafter.bind(target: target)
    }

    @Test func bindIsIdempotentOnSameTarget() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig())
        eval(drafter, target)
        try drafter.bind(target: target)
        try drafter.bind(target: target)   // second call is a no-op
    }

    @Test func rebindToDifferentTargetThrows() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let t1 = Gemma4TextModel(try targetConfig())
        let t2 = Gemma4TextModel(try targetConfig())
        eval(drafter, t1, t2)
        try drafter.bind(target: t1)
        #expect(throws: Gemma4MTPError.rebindForbidden) {
            try drafter.bind(target: t2)
        }
    }

    @Test func hiddenSizeMismatchThrows() throws {
        // Drafter backbone = 64, target hidden = 128 → mismatch.
        let cfg = try drafterConfig(backbone: 64)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(hiddenSize: 128))
        eval(drafter, target)
        #expect(throws: (any Error).self) {
            try drafter.bind(target: target)
        }
    }

    @Test func vocabMismatchThrows() throws {
        let cfg = try drafterConfig(vocab: 64)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(vocab: 128))
        eval(drafter, target)
        #expect(throws: (any Error).self) {
            try drafter.bind(target: target)
        }
    }

    @Test func kEqVCompatCheckPasses() throws {
        // Both drafter and target have attention_k_eq_v = true + matching
        // num_global_key_value_heads.
        let cfg = try drafterConfig(
            attentionKeqV: true, numGlobalKVHeads: 2)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(
            attentionKeqV: true, numGlobalKVHeads: 2))
        eval(drafter, target)
        try drafter.bind(target: target)
    }

    @Test func kEqVMismatchThrows() throws {
        // Drafter has k_eq_v=true, target has k_eq_v=false → throw.
        let cfg = try drafterConfig(
            attentionKeqV: true, numGlobalKVHeads: 2)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(
            attentionKeqV: false))
        eval(drafter, target)
        #expect(throws: (any Error).self) {
            try drafter.bind(target: target)
        }
    }

    @Test func bindRejectsMismatchedEffectiveKVHeadsWhenKeqVDisabled() throws {
        // Global KV heads are honored independent of k_eq_v: a drafter whose
        // full layers use 1 KV head cannot verify a target using 2.
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        let target = Gemma4TextModel(
            try targetConfig(attentionKeqV: false, numGlobalKVHeads: 2))
        eval(drafter, target)

        do {
            try drafter.bind(target: target)
            Issue.record("binding unexpectedly succeeded")
        } catch let Gemma4MTPError.incompatibleDrafter(field, _, _) {
            #expect(field == "fullAttention.effectiveKVHeads")
        } catch {
            Issue.record("threw unexpected error \(error)")
        }
    }

    @Test func bindAcceptsMatchedEffectiveKVHeadsWhenKeqVDisabled() throws {
        // Matched effective full-layer KV head counts bind cleanly even with
        // k_eq_v disabled on both sides.
        let drafter = try Gemma4AssistantDraftModel(
            config: drafterConfig(attentionKeqV: false, numGlobalKVHeads: 2))
        let target = Gemma4TextModel(
            try targetConfig(attentionKeqV: false, numGlobalKVHeads: 2))
        eval(drafter, target)

        try drafter.bind(target: target)
    }

    @Test func slidingWindowMismatchReportsExactGeometry() throws {
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        var targetConfiguration = try targetConfig()
        targetConfiguration.slidingWindow = 128
        let target = Gemma4TextModel(targetConfiguration)
        eval(drafter, target)

        #expect(
            throws: Gemma4MTPError.incompatibleDrafter(
                field: "slidingAttention.slidingWindow", drafter: "64", target: "128")
        ) {
            try drafter.bind(target: target)
        }
    }

    @Test func captureGeometryMismatchesThrowBeforeDrafting() throws {
        typealias DrafterMutation = (inout Gemma4AssistantConfiguration) -> Void
        typealias TargetMutation = (inout Gemma4TextConfiguration) -> Void
        let noDrafterChange: DrafterMutation = { _ in }
        let noTargetChange: TargetMutation = { _ in }
        let cases: [(String, String, DrafterMutation, TargetMutation)] = [
            (
                "sliding head dimension", "slidingAttention.headDimension",
                noDrafterChange, { $0.headDim = 16 }),
            (
                "sliding effective KV heads", "slidingAttention.effectiveKVHeads",
                noDrafterChange, { $0.numKeyValueHeads = 2 }),
            (
                "full head dimension", "fullAttention.headDimension",
                noDrafterChange, { $0.globalHeadDim = 16 }),
            (
                "full K equals V", "fullAttention.attentionKeqV",
                noDrafterChange, {
                    $0.attentionKeqV = true
                    $0.numGlobalKeyValueHeads = 1
                }),
            (
                "global KV heads", "fullAttention.effectiveKVHeads",
                {
                    $0.textConfig.attentionKeqV = true
                    $0.textConfig.numGlobalKeyValueHeads = 1
                },
                {
                    $0.attentionKeqV = true
                    $0.numGlobalKeyValueHeads = 2
                }),
            (
                "full effective KV heads", "fullAttention.effectiveKVHeads",
                {
                    $0.textConfig.layerTypes = Array(
                        repeating: "full_attention",
                        count: $0.textConfig.numHiddenLayers)
                },
                { $0.numKeyValueHeads = 2 }),
            (
                "missing sliding capture", "captureLayer.sliding_attention",
                noDrafterChange, {
                    $0.layerTypes =
                        Array(repeating: "full_attention", count: 5)
                        + Array(repeating: "sliding_attention", count: 5)
                }),
            (
                "missing full capture", "captureLayer.full_attention",
                noDrafterChange, {
                    $0.layerTypes =
                        Array(repeating: "sliding_attention", count: 5)
                        + Array(repeating: "full_attention", count: 5)
                }),
            (
                "matched geometry", "", noDrafterChange, noTargetChange),
        ]

        for (name, expectedField, mutateDrafter, mutateTarget) in cases {
            var drafterConfiguration = try drafterConfig()
            var targetConfiguration = try targetConfig()
            mutateDrafter(&drafterConfiguration)
            mutateTarget(&targetConfiguration)
            let drafter = try Gemma4AssistantDraftModel(config: drafterConfiguration)
            let target = Gemma4TextModel(targetConfiguration)
            eval(drafter, target)

            if expectedField.isEmpty {
                try drafter.bind(target: target)
                continue
            }
            do {
                try drafter.bind(target: target)
                Issue.record("\(name) unexpectedly bound")
            } catch let Gemma4MTPError.incompatibleDrafter(field, _, _) {
                #expect(field == expectedField, "\(name) reported \(field)")
            } catch {
                Issue.record("\(name) threw unexpected error \(error)")
            }
        }
    }
}

@Suite("Gemma4AssistantDraftModel forward")
struct Gemma4AssistantDraftModelForwardTests {

    private func drafterConfig(
        useOrderedEmbeddings: Bool = false
    ) throws -> Gemma4AssistantConfiguration {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 64,
            "use_ordered_embeddings": \(useOrderedEmbeddings),
            "num_centroids": 8,
            "centroid_intermediate_top_k": 2,
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
                "sliding_window": 64,
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
    }

    /// Build dummy shared-KV (K/V for each layer-type). Shapes:
    ///   fullAttention: [B, num_key_value_heads, T, global_head_dim]
    ///   slidingAttention: [B, num_key_value_heads, T, head_dim]
    /// Here: B=1, kvHeads=1, T=8, both head_dim = 32.
    private func makeSharedKV(kvLen: Int = 8) -> Gemma4SharedKV {
        let B = 1, H = 1, D = 32
        func kv() -> (MLXArray, MLXArray) {
            let k = MLXArray.ones([B, H, kvLen, D], dtype: .float32) * 0.1
            let v = MLXArray.ones([B, H, kvLen, D], dtype: .float32) * 0.2
            return (k, v)
        }
        return Gemma4SharedKV(fullAttention: kv(), slidingAttention: kv())
    }

    @Test func forwardProducesExpectedShapes() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        let inputsEmbeds = MLXArray.zeros([1, 1, 2 * 64], dtype: .float32)
        let sharedKV = makeSharedKV()
        let (lastHidden, logits) = drafter(
            inputsEmbeds: inputsEmbeds,
            sharedKV: sharedKV,
            positionOffset: .scalar(8)
        )
        #expect(lastHidden.shape == [1, 1, 64])   // [B, 1, backbone]
        #expect(logits.shape == [1, 1, 64])       // [B, 1, vocab]
    }

    @Test func forwardWithCentroidHeadHasCorrectShapes() throws {
        let cfg = try drafterConfig(useOrderedEmbeddings: true)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        let inputsEmbeds = MLXArray.zeros([1, 1, 2 * 64], dtype: .float32)
        let sharedKV = makeSharedKV()
        let (lastHidden, logits) = drafter(
            inputsEmbeds: inputsEmbeds,
            sharedKV: sharedKV,
            positionOffset: .scalar(8)
        )
        #expect(lastHidden.shape == [1, 1, 64])
        #expect(logits.shape == [1, 1, 64])
    }

    @Test func forwardLogitsAreFinite() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        let inputsEmbeds = MLXArray.zeros([1, 1, 2 * 64], dtype: .float32)
        let sharedKV = makeSharedKV()
        let (_, logits) = drafter(
            inputsEmbeds: inputsEmbeds,
            sharedKV: sharedKV,
            positionOffset: .scalar(8)
        )
        // Softcap NOT applied on drafter path, so logits can exceed ±30.
        // This test just verifies they're not NaN / ±inf on zero input.
        let absMax = MLX.abs(logits).max().item(Float.self)
        #expect(absMax.isFinite)
    }
}

@Suite("Gemma4AssistantDraftModel sanitize")
struct Gemma4AssistantDraftModelSanitizeTests {

    private func drafterConfig(
        tieWordEmbeddings: Bool = true
    ) throws -> Gemma4AssistantConfiguration {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 64,
            "use_ordered_embeddings": true,
            "num_centroids": 8,
            "centroid_intermediate_top_k": 2,
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
                "sliding_window": 64,
                "tie_word_embeddings": \(tieWordEmbeddings),
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
    }

    @Test func castsTokenOrderingToInt32() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let int64Ordering = MLXArray([Int64(0), 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 10, 11, 12, 13, 14, 15,
                                      16, 17, 18, 19, 20, 21, 22, 23,
                                      24, 25, 26, 27, 28, 29, 30, 31,
                                      32, 33, 34, 35, 36, 37, 38, 39,
                                      40, 41, 42, 43, 44, 45, 46, 47,
                                      48, 49, 50, 51, 52, 53, 54, 55,
                                      56, 57, 58, 59, 60, 61, 62, 63])
        var weights: [String: MLXArray] = [
            "masked_embedding.token_ordering": int64Ordering,
        ]
        weights = try drafter.sanitize(weights: weights)
        let cast = weights["masked_embedding.token_ordering"]
        #expect(cast != nil)
        #expect(cast!.dtype == .int32)
    }

    @Test func dropsLmHeadWhenTied() throws {
        let cfg = try drafterConfig(tieWordEmbeddings: true)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let lmHead = MLXArray.zeros([64, 64], dtype: .float32)
        let preProj = MLXArray.zeros([64, 128], dtype: .float32)
        var weights: [String: MLXArray] = [
            "lm_head.weight": lmHead,
            "pre_projection.weight": preProj,
        ]
        weights = try drafter.sanitize(weights: weights)
        #expect(weights["lm_head.weight"] == nil)         // dropped
        #expect(weights["pre_projection.weight"] != nil)  // kept
    }

    @Test func keepsLmHeadWhenUntied() throws {
        let cfg = try drafterConfig(tieWordEmbeddings: false)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let lmHead = MLXArray.zeros([64, 64], dtype: .float32)
        var weights: [String: MLXArray] = [
            "lm_head.weight": lmHead,
        ]
        weights = try drafter.sanitize(weights: weights)
        #expect(weights["lm_head.weight"] != nil)
    }

    @Test func failsLoudOnUnexpectedKProj() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let kProj = MLXArray.zeros([64, 64], dtype: .float32)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.k_proj.weight": kProj,
        ]
        #expect(throws: (any Error).self) {
            _ = try drafter.sanitize(weights: weights)
        }
    }

    @Test func failsLoudOnUnexpectedVProj() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let vProj = MLXArray.zeros([64, 64], dtype: .float32)
        let weights: [String: MLXArray] = [
            "model.layers.2.self_attn.v_proj.weight": vProj,
        ]
        #expect(throws: (any Error).self) {
            _ = try drafter.sanitize(weights: weights)
        }
    }

    @Test func preservesOtherWeights() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let qProj = MLXArray.zeros([64, 64], dtype: .float32)
        let preProj = MLXArray.zeros([64, 128], dtype: .float32)
        let postProj = MLXArray.zeros([64, 64], dtype: .float32)
        let embed = MLXArray.zeros([64, 64], dtype: .float32)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_proj.weight": qProj,
            "pre_projection.weight": preProj,
            "post_projection.weight": postProj,
            "model.embed_tokens.weight": embed,
        ]
        let out = try drafter.sanitize(weights: weights)
        #expect(out.count == 4)  // all four kept
    }
}

@Suite("Gemma4AssistantDraftModel loading")
struct Gemma4AssistantDraftModelLoadingTests {

    /// Integration test gated on env var. Loads a pre-downloaded drafter
    /// directory from disk using the local-path variant of `load`.
    @Test(.enabled(if: ProcessInfo.processInfo.environment[
        "MLX_SWIFT_LM_INTEGRATION_DATA_DIR"] != nil))
    func loadsFromDirectory() async throws {
        guard let dirPath = ProcessInfo.processInfo.environment[
            "MLX_SWIFT_LM_INTEGRATION_DATA_DIR"] else { return }
        // Expect a sub-directory e.g. "gemma-4-E4B-it-assistant-bf16" under
        // the integration data dir; caller is responsible for pre-downloading.
        let drafterDir = URL(fileURLWithPath: dirPath)
            .appending(path: "gemma-4-E4B-it-assistant-bf16")
        let drafter = try await Gemma4AssistantDraftModel.load(from: drafterDir)
        // A basic shape check: the first layer's q_proj weight should exist.
        // We don't assert exact values here; the parity tests do that.
        let params = drafter.parameters()
        // Flatten to check a known key exists.
        let flat = params.flattened()
        let keys = Set(flat.map(\.0))
        #expect(keys.contains("model.layers.0.self_attn.q_proj.weight"))
    }
}
