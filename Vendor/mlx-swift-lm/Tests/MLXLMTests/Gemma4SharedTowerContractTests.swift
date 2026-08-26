import Foundation
import MLX
import MLXFast

@testable import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXVLM

/// Contract tests pinning the deleted inline VLM text tower's semantics onto
/// the canonical `Gemma4TextModel` and configuration types. Each test guards a
/// behavior whose loss was found during review of the shared-tower cutover:
/// fp16 attention promotion, rank-1 token acceptance, full-layer global KV
/// head counts independent of k_eq_v, and nested quantization round-tripping.
@Suite("Gemma 4 shared-tower contracts", .serialized)
struct Gemma4SharedTowerContractTests {

    private func tinyConfigJSON(extraFields: [String] = []) -> String {
        var fields = [
            "\"model_type\": \"gemma4_text\"",
            "\"hidden_size\": 32",
            "\"num_hidden_layers\": 2",
            "\"intermediate_size\": 64",
            "\"num_attention_heads\": 2",
            "\"head_dim\": 16",
            "\"global_head_dim\": 16",
            "\"num_key_value_heads\": 2",
            "\"num_kv_shared_layers\": 0",
            "\"sliding_window\": 16",
            "\"final_logit_softcapping\": 30.0",
            "\"hidden_size_per_layer_input\": 0",
            "\"use_double_wide_mlp\": false",
            "\"tie_word_embeddings\": true",
            "\"vocab_size\": 64",
            "\"vocab_size_per_layer_input\": 64",
            "\"rms_norm_eps\": 1e-6",
        ]
        if !extraFields.contains(where: { $0.hasPrefix("\"layer_types\"") }) {
            fields.append("\"layer_types\": [\"sliding_attention\", \"full_attention\"]")
        }
        fields.append(contentsOf: extraFields)
        return "{ \(fields.joined(separator: ", ")) }"
    }

    private func tinyConfig(extraFields: [String] = []) throws -> Gemma4TextConfiguration {
        let json = tinyConfigJSON(extraFields: extraFields)
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func tinyModel(_ config: Gemma4TextConfiguration) -> Gemma4TextModel {
        Gemma4TextModel(config)
    }

    private func param(
        _ params: [(String, MLXArray)], _ key: String
    ) -> MLXArray? {
        params.first { $0.0 == key }?.1
    }

    // MARK: F2 — rank-1 token acceptance

    /// The deleted inline twin accepted [N] token ids on cache-reuse turns and
    /// expanded them to [1, N]. `Gemma4TextModel` must keep that contract at
    /// both public entry points instead of trapping on `dim(1)`.
    @Test("rank-1 token ids produce the batched result, not a trap")
    func rankOneTokenAcceptance() throws {
        let config = try tinyConfig()
        let model = tinyModel(config)
        let ids = MLXArray([Int32(3), 5, 7, 11])
        let singleton = model(ids, cache: nil as [KVCache]?)
        let batched = model(ids.expandedDimensions(axis: 0), cache: nil as [KVCache]?)
        eval(singleton, batched)
        #expect(singleton.shape == batched.shape)
        #expect(singleton.dtype == batched.dtype)
        let equal = allClose(singleton, batched, rtol: 0, atol: 0)
        eval(equal)
        #expect(equal.item(Bool.self))
    }

    /// The second normalized entry point: the MTP verify surface
    /// (`callCapturingPreNorm`) must accept [N] ids identically.
    @Test("rank-1 token ids through the preNorm capture entry point")
    func rankOneTokenAcceptancePreNorm() throws {
        let config = try tinyConfig()
        let model = tinyModel(config)
        let ids = MLXArray([Int32(3), 5, 7, 11])
        let singleton = model.model.callCapturingPreNorm(ids)
        let batched = model.model.callCapturingPreNorm(ids.expandedDimensions(axis: 0))
        eval(singleton.postNorm, batched.postNorm)
        #expect(singleton.postNorm.shape == batched.postNorm.shape)
        #expect(allClose(singleton.postNorm, batched.postNorm, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test("rank-1 multimodal tuple normalizes tokens, embeddings, and mask")
    func rankOneMultimodalNormalization() throws {
        let config = try tinyConfig(extraFields: [
            "\"use_bidirectional_attention\": \"vision\"",
        ])
        let model = tinyModel(config)
        let ids = MLXArray([Int32(3), 5, 7, 11])
        let batchedIDs = ids.expandedDimensions(axis: 0)
        let embedding = model.scaledInputEmbeddings(ids)
        let batchedEmbedding = embedding.expandedDimensions(axis: 0)
        let mask = MLXArray([false, true, true, false])

        let rankOne = model(
            ids, inputEmbedding: embedding, cache: nil, imageTokenMask: mask)
        let batched = model(
            batchedIDs, inputEmbedding: batchedEmbedding, cache: nil,
            imageTokenMask: mask.expandedDimensions(axis: 0))
        eval(rankOne, batched)
        #expect(rankOne.shape == batched.shape)
        #expect(allClose(rankOne, batched, rtol: 0, atol: 0).item(Bool.self))
    }

    // MARK: F3 — global KV heads independent of k_eq_v

    /// A full-attention layer with `num_global_key_value_heads` different from
    /// `num_key_value_heads` uses the global count even when `attention_k_eq_v`
    /// is false — k_eq_v only elides `v_proj`. The pre-fix tower resolved the
    /// global count only under k_eq_v, mis-sizing projections and caches for
    /// such checkpoints.
    /// Asserted through the public module tree: projection widths encode the
    /// head-count rule (`rows = nKvHeads * effectiveHeadDim`), and `kvHeads`
    /// feeds cache sizing directly. Layer index 1 is the declared
    /// full-attention layer; the fixture uses head_dim == global_head_dim ==
    /// 16, so the distinguishing row counts are 32 (2 heads) vs 16 (1 head).
    @Test("full layer honors global KV head count without k_eq_v")
    func fullLayerGlobalKVHeadsWithoutKEqV() throws {
        let config = try tinyConfig(extraFields: [
            "\"attention_k_eq_v\": false",
            "\"num_global_key_value_heads\": 1",
            "\"num_key_value_heads\": 2",
        ])
        let model = tinyModel(config)
        #expect(model.kvHeads == [2, 1])

        let params = model.parameters().flattened()
        // Sliding layer 0: 2 KV heads × headDim 16 = 32 rows.
        #expect(param(params, "model.layers.0.self_attn.k_proj.weight")?.shape == [32, 32])
        // Full layer 1: global KV heads (1) × globalHeadDim 16 = 16 rows, and
        // k_eq_v=false keeps a real v_proj of the same width.
        #expect(param(params, "model.layers.1.self_attn.k_proj.weight")?.shape == [16, 32])
        #expect(param(params, "model.layers.1.self_attn.v_proj.weight")?.shape == [16, 32])
    }

    @Test("full layer still honors global KV head count under k_eq_v")
    func fullLayerGlobalKVHeadsWithKEqV() throws {
        let config = try tinyConfig(extraFields: [
            "\"attention_k_eq_v\": true",
            "\"num_global_key_value_heads\": 1",
            "\"num_key_value_heads\": 2",
        ])
        let model = tinyModel(config)
        #expect(model.kvHeads == [2, 1])
        let params = model.parameters().flattened()
        // k_eq_v=true: the global count still rules, and v_proj is elided.
        #expect(param(params, "model.layers.1.self_attn.k_proj.weight")?.shape == [16, 32])
        #expect(param(params, "model.layers.1.self_attn.v_proj.weight") == nil)
    }

    /// The CBv2 layer-kind derivation (KV storage geometry) must agree with
    /// the model, layer for layer — `CBv2LayerCache` preconditions on it.
    @Test("CBv2 layer kinds agree with model head counts across k_eq_v")
    func layerKindsAgreeWithModel() throws {
        for keqV in [false, true] {
            let config = try tinyConfig(extraFields: [
                "\"attention_k_eq_v\": \(keqV)",
                "\"num_global_key_value_heads\": 1",
                "\"num_key_value_heads\": 2",
            ])
            let model = tinyModel(config)
            let kinds = config.cbv2LayerKinds
            #expect(kinds.map(\.kvHeads) == model.kvHeads)
            #expect(kinds[1].kvHeads == 1)
        }
    }

    /// Short explicit `layer_types` lists normalize at decode (padding
    /// sliding) instead of trapping at model construction; an empty list
    /// falls back to all-sliding.
    @Test("short and empty layer_types lists are normalized")
    func shortLayerTypesNormalized() throws {
        let short = try tinyConfig(extraFields: [
            "\"layer_types\": [\"full_attention\"]",
        ])
        #expect(short.layerTypes == ["full_attention", "sliding_attention"])
        _ = tinyModel(short)  // must not trap

        let empty = try tinyConfig(extraFields: ["\"layer_types\": []"])
        #expect(empty.layerTypes == ["sliding_attention", "sliding_attention"])
        _ = tinyModel(empty)
    }

    @Test("oversized layer_types is truncated to the declared layer count")
    func oversizedLayerTypesNormalized() throws {
        let config = try tinyConfig(extraFields: [
            "\"layer_types\": [\"full_attention\", \"sliding_attention\", \"full_attention\"]",
        ])
        #expect(config.layerTypes == ["full_attention", "sliding_attention"])
        #expect(config.cbv2LayerKinds.count == config.numHiddenLayers)
        _ = tinyModel(config)
    }

    @Test("all-mode rectangular masks preserve cached columns")
    func rectangularAllModeMaskSymmetrization() throws {
        let base = MLXArray([
            false, true, true, false,
            false, false, true, true,
        ]).reshaped(1, 1, 2, 4)
        let mode = gemma4TextSymmetrizeMask(.array(base))
        guard case .array(let result) = mode else {
            Issue.record("all-mode mask was not materialized")
            return
        }
        let expected = MLXArray([
            false, true, true, true,
            false, false, true, true,
        ]).reshaped(1, 1, 2, 4)
        eval(result, expected)
        #expect(result.shape == [1, 1, 2, 4])
        #expect(allClose(result.asType(.int32), expected.asType(.int32)).item(Bool.self))
    }

    @Test("all-mode Gemma rejects chunked CBv2 prefill")
    func allModeCBv2PrefillRejection() throws {
        let config = try tinyConfig(extraFields: [
            "\"use_bidirectional_attention\": \"all\"",
        ])
        let model = tinyModel(config)
        let kinds = config.cbv2LayerKinds
        #expect(kinds.map(\.isBidirectional) == [true, true])
        #expect(throws: Gemma4TextModel.CBv2CompatibilityError.fullyBidirectionalAttentionUnsupported) {
            try model.newCacheV2 { index, kind in
                CBv2LayerCache(layerIndex: index, kind: kind)
            }
        }
    }

    @Test("all-mode legacy prepare keeps the prompt in one forward")
    func allModeLegacyPrefillIsNotChunked() throws {
        let tokens = MLXArray([Int32(1), 2, 3, 4])
        let input = LMInput(tokens: tokens)
        let allMode = tinyModel(try tinyConfig(extraFields: [
            "\"use_bidirectional_attention\": \"all\"",
        ]))
        let ordinary = tinyModel(try tinyConfig())

        guard case .tokens(let allTokens) = try allMode.prepare(
            input, cache: allMode.newCache(parameters: nil), windowSize: 2)
        else {
            Issue.record("all-mode prepare unexpectedly returned logits")
            return
        }
        guard case .tokens(let ordinaryTokens) = try ordinary.prepare(
            input, cache: ordinary.newCache(parameters: nil), windowSize: 2)
        else {
            Issue.record("ordinary prepare unexpectedly returned logits")
            return
        }

        #expect(allTokens.tokens.size == 4)
        #expect(ordinaryTokens.tokens.size == 2)
    }

    // MARK: Full-layer RoPE construction (disclosed divergence from the
    // deleted inline tower)

    /// The canonical full-layer rope is `ProportionalRoPE` over the FULL head
    /// dim with `/dims` denominators and +inf-padded pass-through pairs —
    /// matching the HF `modeling_rope_utils._compute_proportional_rope_parameters`
    /// and mlx-lm `ProportionalRoPE` references. The deleted inline VLM tower
    /// instead built a default rope over the truncated 128 dims (with
    /// `/rotatedDims` frequencies and NeoX pairing) — that tower was the
    /// anomaly, and the cutover corrects it. This pin locks the canonical
    /// construction so any future convergence back to the truncated scheme is
    /// an explicit, reviewed change.
    @Test("full-layer rope uses proportional construction over the full head dim")
    func fullLayerRoPEConstruction() {
        let rope = ProportionalRoPE(
            dims: 512,
            traditional: false,
            base: 1_000_000,
            scalingConfig: ["partial_rotary_factor": .float(0.25)])
        let freqs = try! #require(rope._freqs)
        eval(freqs)
        // 64 real frequencies + 192 infinities = dims/2 entries over the FULL
        // head dim (a truncated-rope scheme would stop at 64 entries).
        #expect(freqs.shape == [256])
        for i in [0, 1, 31, 63] as [Int] {
            let expected = pow(1_000_000 as Float, Float(2 * i) / 512)
            #expect(abs(freqs[i].item(Float.self) - expected) / expected < 1e-5)
        }
        for i in [64, 128, 255] as [Int] {
            #expect(freqs[i].item(Float.self) == .infinity)
        }
    }

    // MARK: F1 — fp16 attention promotion

    /// fp16 activations are promoted to fp32 around SDPA (vmlx #52). Scaling
    /// `q_norm`/`k_norm` GAINS (post-normalization) by 600 reaches the score
    /// computation un-normalized: |q·k| ≈ 600²×16×|x·y| ≫ 65504, so any build
    /// WITHOUT the promotion produces non-finite logits (negative-control
    /// property). The promoted tower must stay finite and softcap-bounded.
    /// Forging the projections instead would be vacuous — RMSNorm divides
    /// projection scale out before attention.
    @Test("fp16 forward with fp16-overflowing Q×K scores stays finite")
    func fp16AttentionPromotion() throws {
        let config = try tinyConfig()
        let model = tinyModel(config)

        var fp16: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            var value = value.asType(.float16)
            if key.hasSuffix(".self_attn.q_norm.weight") || key.hasSuffix(".self_attn.k_norm.weight") {
                value = (value.asType(.float32) * 600).asType(.float16)
            }
            fp16[key] = value
        }
        model.update(parameters: ModuleParameters.unflattened(fp16))

        let out = model(MLXArray([Int32(3), 5, 7, 11]), cache: nil as [KVCache]?)
        eval(out)
        let magnitude = abs(out).max().item(Float.self)
        #expect(magnitude.isFinite)
        // Soft-capped logits are bounded by the configured cap.
        #expect(magnitude <= 30.0)
    }

    @Test("fp16 CBv2 forward keeps overflowing attention scores finite")
    func fp16CBv2AttentionPromotion() throws {
        let config = try tinyConfig()
        let model = tinyModel(config)
        var fp16: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            var value = value.asType(.float16)
            if key.hasSuffix(".self_attn.q_norm.weight") || key.hasSuffix(".self_attn.k_norm.weight") {
                value = (value.asType(.float32) * 600).asType(.float16)
            }
            fp16[key] = value
        }
        model.update(parameters: ModuleParameters.unflattened(fp16))

        let layerKinds = model.cbv2LayerKinds
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20))
        let state = try backend.makeSequenceState(
            layerKinds: layerKinds, promptLength: 4, maxLength: 8)
        defer { backend.release(state) }
        let caches: [CBv2LayerCache] = layerKinds.enumerated().map {
            CBv2LayerCache(layerIndex: $0.offset, kind: $0.element)
        }
        for index in layerKinds.indices where layerKinds[index].sharesKVWithLayer == nil {
            caches[index].setRows([state[index]!])
        }

        let out = model(
            MLXArray([Int32(3), 5, 7, 11]).reshaped(1, 4),
            cache: caches.map { $0 as KVCache })
        eval(out)
        let magnitude = abs(out).max().item(Float.self)
        #expect(magnitude.isFinite)
        #expect(magnitude <= 30.0)
    }

    // MARK: F4 — nested quantization round trip

    private func vlmJSON(textQuantization: String?, rootQuantization: String?) -> String {
        let textQ = textQuantization.map { ", \"quantization\": \($0)" } ?? ""
        let rootQ = rootQuantization.map { ", \"quantization\": \($0)" } ?? ""
        return """
            {
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "head_dim": 16,
                    "global_head_dim": 16,
                    "num_key_value_heads": 2,
                    "vocab_size": 64
                    \(textQ)
                },
                "vision_config": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "image_size": 8,
                    "patch_size": 4
                }
                \(rootQ)
            }
            """
    }

    @Test("nested text_config quantization survives decode-encode-decode")
    func nestedQuantizationRoundTrip() throws {
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()

        let first = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: Data(vlmJSON(
                textQuantization: "{\"bits\": 4, \"group_size\": 64}",
                rootQuantization: nil).utf8))
        #expect(first.quantization == nil)
        #expect(first.textConfig.quantizationBits == 4)
        #expect(first.textConfig.quantizationGroupSize == 64)
        #expect(first.textConfig.quantizationMode == .affine)
        #expect(gemma4SupportsSafeExpertQMMQuantization(first.textConfig))

        let encoded = try encoder.encode(first)
        let second = try decoder.decode(MLXVLM.Gemma4Configuration.self, from: encoded)
        #expect(second.textConfig.quantizationBits == 4)
        #expect(second.textConfig.quantizationGroupSize == 64)
        #expect(second.textConfig.quantizationMode == .affine)
        #expect(gemma4SupportsSafeExpertQMMQuantization(second.textConfig))

        // A third cycle must be idempotent (no drift, no duplication).
        let third = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: try encoder.encode(second))
        #expect(third.textConfig.quantizationBits == 4)
        #expect(third.textConfig.quantizationGroupSize == 64)
        #expect(third.textConfig.quantizationMode == .affine)
    }

    @Test("mxfp4 mode survives round-trip and remains unsafe for expert R1")
    func mxfp4QuantizationRoundTrip() throws {
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        let first = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: Data(vlmJSON(
                textQuantization:
                    "{\"bits\": 4, \"group_size\": 64, \"mode\": \"mxfp4\"}",
                rootQuantization: nil).utf8))
        #expect(first.textConfig.quantizationMode == .mxfp4)
        #expect(!gemma4SupportsSafeExpertQMMQuantization(first.textConfig))

        let second = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: try encoder.encode(first))
        #expect(second.textConfig.quantizationBits == 4)
        #expect(second.textConfig.quantizationGroupSize == 64)
        #expect(second.textConfig.quantizationMode == .mxfp4)
        #expect(!gemma4SupportsSafeExpertQMMQuantization(second.textConfig))
    }

    @Test("root quantization keeps precedence and round-trips")
    func rootQuantizationRoundTrip() throws {
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        let first = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: Data(vlmJSON(
                textQuantization: "{\"bits\": 8, \"group_size\": 128}",
                rootQuantization: "{\"quant_method\": \"affine\", \"bits\": 4, \"group_size\": 64}").utf8))
        // Root metadata overlays nested text config at decode.
        #expect(first.textConfig.quantizationBits == 4)
        #expect(first.textConfig.quantizationGroupSize == 64)
        #expect(first.textConfig.quantizationMode == .affine)
        let second = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: try encoder.encode(first))
        #expect(second.textConfig.quantizationBits == 4)
        #expect(second.textConfig.quantizationGroupSize == 64)
        #expect(second.textConfig.quantizationMode == .affine)
    }

    @Test("root per-layer quantization survives VLM round-trip")
    func rootPerLayerQuantizationRoundTrip() throws {
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        let path = "model.layers.0.experts.switch_glu.gate_proj"
        let first = try decoder.decode(
            MLXVLM.Gemma4Configuration.self,
            from: Data(vlmJSON(
                textQuantization: nil,
                rootQuantization:
                    "{\"bits\": 4, \"group_size\": 64, \"\(path)\": false}").utf8))
        #expect(first.textConfig.hasExpertQuantizationOverrides)

        let second = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: try encoder.encode(first))
        #expect(second.textConfig.quantizationBits == 4)
        #expect(second.textConfig.quantizationGroupSize == 64)
        #expect(second.textConfig.hasExpertQuantizationOverrides)
        #expect(!gemma4SupportsSafeExpertQMMQuantization(second.textConfig))
    }
}
