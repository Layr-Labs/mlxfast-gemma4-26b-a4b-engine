//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - vMLX decode hot-path helpers (ported from osaurus/main Gemma4Text)
private let gemma4CompiledDecodeSupported: Bool = {
    if let raw = ProcessInfo.processInfo.environment["MLX_COMPILED_DECODE"] {
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }
    return true
}()

// MARK: - CBv2 B=8 decode graph-submission ladder

@inline(__always)
internal func resolveGemma4DecodeAsyncEvalLadderEnabled(_ raw: String?) -> Bool {
    guard let raw else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}

private let gemma4DecodeAsyncEvalLadderEnabled =
    resolveGemma4DecodeAsyncEvalLadderEnabled(
        ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_DECODE_ASYNC_EVAL_LADDER"])

@inline(__always)
internal func gemma4ShouldSubmitDecodeAsyncEvalLadder(
    enabled: Bool,
    schedulePrefill: Bool,
    isCBv2: Bool,
    batchSize: Int,
    inputLength: Int,
    layerIndex: Int
) -> Bool {
    guard enabled, isCBv2, !schedulePrefill, batchSize == 8, inputLength == 1
    else { return false }

    if let set = gemma4DecodeAsyncEvalLadderSet {
        return set.contains(layerIndex)
    }
    switch layerIndex {
    case 0, 1, 2, 3:
        return true
    default:
        return false
    }
}

private let gemma4DecodeAsyncEvalLadderSet: Set<Int>? = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_DECODE_LADDER_SET"]
    else { return nil }
    return Set(raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(
        in: .whitespaces)) })
}()

// MARK: - CBv2 prompt-path knobs (prefill only; decode never reads these)

@inline(__always)
private func gemma4TruthyFlag(_ raw: String?) -> Bool {
    guard let raw else { return false }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

@inline(__always)
internal func resolveGemma4PrefillChunkEvalLayers(_ raw: String?) -> Int {
    guard let raw, let value = Int(raw) else { return 18 }
    return max(0, value)
}

private let gemma4PrefillChunkEvalLayers = resolveGemma4PrefillChunkEvalLayers(
    ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_PREFILL_CHUNK_EVAL"])

private let gemma4LongPrefillChunkEvalLayers: Int = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_CHUNK_EVAL_LONG"],
        let value = Int(raw), value >= 0
    else { return 3 }
    return value
}()

private let gemma4BlockedQueryPrefillThreshold = 128

@inline(__always)
private func gemma4EffectivePrefillChunkEvalLayers(
    configured: Int, inputLength: Int
) -> Int {
    guard configured == 18,
        gemma4LongPrefillChunkEvalLayers > 0,
        gemma4LongPrefillChunkEvalLayers < configured,
        inputLength > gemma4BlockedQueryPrefillThreshold
    else { return configured }
    return gemma4LongPrefillChunkEvalLayers
}

@inline(__always)
internal func gemma4ShouldSubmitPrefillChunkEval(
    schedulePrefill: Bool,
    isCBv2: Bool,
    inputLength: Int,
    layerNumber: Int,
    interval: Int
) -> Bool {
    schedulePrefill && isCBv2 && interval > 0 && inputLength > 1
        && layerNumber.isMultiple(of: interval)
}

private let gemma4PrefillTailRows: Int = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_TAIL_ROWS"],
        let value = Int(raw)
    else { return 1 }
    return max(0, value)
}()

func gemma4FusedWeightedUnsortFlag(_ raw: String?) -> Bool {
    guard let raw else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}

func gemma4AllowsWeightedExpertUnsort(schedulePrefill: Bool) -> Bool {
    schedulePrefill
}

func gemma4SupportsProductionExpertTopology(_ config: Gemma4TextConfiguration) -> Bool {
    config.enableMoeBlock
        && config.hiddenSize == 2816
        && config.numHiddenLayers == 30
        && config.numExperts == 128
        && config.topKExperts == 8
        && config.moeIntermediateSize == 704
        && config.useBidirectionalAttention == "vision"
}

func gemma4HasExpertQuantizationOverrides(
    _ quantization: BaseConfiguration.PerLayerQuantization?
) -> Bool {
    quantization?.perLayerQuantization.keys.contains { path in
        path.split(separator: ".").contains("experts")
    } ?? false
}

func gemma4SupportsSafeExpertQMMQuantization(_ config: Gemma4TextConfiguration) -> Bool {
    config.quantizationBits == 4 && config.quantizationGroupSize == 64
        && config.quantizationMode == .affine
        && !config.hasExpertQuantizationOverrides
}

func gemma4SupportsCoupledExpertOptimizations(_ config: Gemma4TextConfiguration) -> Bool {
    gemma4SupportsProductionExpertTopology(config)
        && gemma4SupportsSafeExpertQMMQuantization(config)
}

internal let gemma4FusedWeightedUnsortRequested = gemma4FusedWeightedUnsortFlag(
    ProcessInfo.processInfo.environment["MLX_GEMMA4_FUSED_WEIGHTED_UNSORT"])

func gemma4ShouldFuseWeightedUnsort(
    _ config: Gemma4TextConfiguration,
    requested: Bool = gemma4FusedWeightedUnsortRequested
) -> Bool {
    requested && gemma4SupportsCoupledExpertOptimizations(config)
}

private let gemma4PrefillTailMinChunk: Int = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_TAIL_MIN_CHUNK"],
        let value = Int(raw)
    else { return 128 }
    return max(2, value)
}()

private let gemma4PrefillLastQueryEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_LAST_QUERY"]
    else { return true }
    return gemma4TruthyFlag(raw)
}()

func gemma4SupportsLastQueryPrefill(_ config: Gemma4TextConfiguration) -> Bool {
    config.layerTypes.count == config.numHiddenLayers
        && config.layerTypes.last == "full_attention"
        && !config.layerUsesSharedKV(layerIdx: config.numHiddenLayers - 1)
}

func gemma4UseLastQueryPrefill(
    _ config: Gemma4TextConfiguration,
    layerIdx: Int,
    batchSize: Int,
    sequenceLength: Int,
    outputTailRows: Int?,
    hasCapableCache: Bool,
    enabled: Bool = gemma4PrefillLastQueryEnabled
) -> Bool {
    enabled
        && hasCapableCache
        && outputTailRows == 1
        && layerIdx == config.numHiddenLayers - 1
        && batchSize > 0
        && sequenceLength > 1
        && gemma4SupportsLastQueryPrefill(config)
}

private let gemma4SafeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    return gemma4CompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

private let gemma4SafeGeluProduct: @Sendable (
    MLXArray, MLXArray
) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        let activated = 0.5 * gate
            * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))
        return activated * up
    }
    return gemma4CompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

private let gemma4SafeGeluProductShaped: @Sendable (
    MLXArray, MLXArray
) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        let activated = 0.5 * gate
            * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))
        return activated * up
    }
    return gemma4CompiledDecodeSupported ? compile(body) : body
}()

@inline(__always)
func geluFusionClaimsPinnedDecode(_ gate: MLXArray, _ up: MLXArray) -> Bool {
    guard gemma4ShapedGeluFuseEnabled,
        gate.dtype == .bfloat16, up.dtype == .bfloat16,
        gate.shape == up.shape
    else { return false }
    let s = gate.shape
    if s.count == 3, s[1] == 1, s[0] == 8 || s[0] == 64 { return true }
    if s.count == 2, s[0] == 64 { return true }
    if CBv2MTPWideVerifyContext.active {
        let columns = CBv2MTPWideVerifyContext.columns
        if s.count == 3, s[1] == 1, s[0] == 8 * columns || s[0] == 64 * columns { return true }
        if s.count == 2, s[0] == 64 * columns { return true }
    }
    return false
}

let gemma4ShapedGeluFuseEnabled: Bool =
    ProcessInfo.processInfo.environment["DARKBLOOM_GELU_SHAPED_FUSE"] != "0"

private let gemma4GeluPrefillFuseEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GELU_SHAPED_FUSE_PREFILL"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4GeluPrefillShapes = ShapedGeluPrefillShapes(
    cap: shapedGeluPrefillShapeCap)

@inline(__always)
func geluFusionClaimsPrefill(_ gate: MLXArray, _ up: MLXArray) -> Bool {
    guard gemma4ShapedGeluFuseEnabled, gemma4GeluPrefillFuseEnabled,
        gate.dtype == .bfloat16, up.dtype == .bfloat16,
        gate.shape == up.shape,
        gate.size >= shapedGeluPrefillMinElements,
        gemma4GeluPrefillShapes.admits(gate.shape)
    else { return false }
    CBv2EngageMark.once("gelu-shaped-prefill-dense")
    return true
}

@inline(__always)
func gemma4GeluProduct(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    if geluFusionClaimsPinnedDecode(gate, up) || geluFusionClaimsPrefill(gate, up) {
        return gemma4SafeGeluProductShaped(gate, up)
    }
    return gemma4SafeGeluProduct(gate, up)
}

private let gemma4CompiledLogitSoftcap: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (x: MLXArray, cap: MLXArray) -> MLXArray in
        tanh(x / cap) * cap
    }
    return gemma4CompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

// MARK: - Configuration

struct Gemma4WeightQuantizationMetadata: Codable, Sendable {
    var bits: Int?
    var groupSize: Int?
    var mode: QuantizationMode?

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
    }
}

private struct Gemma4WeightQuantizationConfiguration: Encodable {
    let fallback: Gemma4WeightQuantizationMetadata
    let overrides: [String: BaseConfiguration.QuantizationOption]

    private struct DynamicKey: CodingKey {
        let stringValue: String
        let intValue: Int? = nil
        init(stringValue: String) { self.stringValue = stringValue }
        init(intValue: Int) { self.stringValue = "\(intValue)" }
    }

    func encode(to encoder: Encoder) throws {
        try fallback.encode(to: encoder)
        var container = encoder.container(keyedBy: DynamicKey.self)
        for (path, option) in overrides {
            switch option {
            case .skip:
                try container.encode(false, forKey: DynamicKey(stringValue: path))
            case .quantize(let quantization):
                try container.encode(quantization, forKey: DynamicKey(stringValue: path))
            }
        }
    }
}

public enum Gemma4TextConfigurationDefaults: Sendable, Equatable {
    case languageModel
    case visionLanguageModel
}

public struct Gemma4TextConfiguration: Codable, Sendable {
    public internal(set) var modelType: String = "gemma4_text"
    public internal(set) var hiddenSize: Int = 1536
    public internal(set) var numHiddenLayers: Int = 35
    public internal(set) var intermediateSize: Int = 6144
    public internal(set) var numAttentionHeads: Int = 8
    public internal(set) var headDim: Int = 256
    public internal(set) var globalHeadDim: Int = 512
    public internal(set) var globalPartialRotaryFactor: Float = 0.25
    public internal(set) var rmsNormEps: Float = 1e-6
    public internal(set) var vocabSize: Int = 262144
    public internal(set) var vocabSizePerLayerInput: Int = 262144
    public internal(set) var numKeyValueHeads: Int = 1
    public internal(set) var numGlobalKeyValueHeads: Int?
    public var numKvSharedLayers: Int = 20
    public internal(set) var hiddenSizePerLayerInput: Int = 256
    public internal(set) var slidingWindow: Int = 512
    public internal(set) var slidingWindowPattern: Int = 5
    public internal(set) var maxPositionEmbeddings: Int = 131072
    public internal(set) var attentionKeqV: Bool = false
    public internal(set) var finalLogitSoftcapping: Float = 30.0
    public internal(set) var useDoubleWideMlp: Bool = true
    public internal(set) var layerTypes: [String] = []
    public internal(set) var tieWordEmbeddings: Bool = true
    public internal(set) var quantizationBits: Int?
    public internal(set) var quantizationGroupSize: Int?
    public internal(set) var quantizationMode: QuantizationMode = .affine
    public internal(set) var perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    public var hasExpertQuantizationOverrides: Bool {
        gemma4HasExpertQuantizationOverrides(perLayerQuantization)
    }

    public internal(set) var enableMoeBlock: Bool = false
    public internal(set) var numExperts: Int?
    public internal(set) var topKExperts: Int?
    public internal(set) var moeIntermediateSize: Int?

    public internal(set) var ropeParameters: [String: [String: StringOrNumber]]?

    public internal(set) var useBidirectionalAttention: String?

    public internal(set) var slidingRopeTheta: Float = 10000.0
    public internal(set) var fullRopeTheta: Float = 1_000_000.0
    public internal(set) var fullPartialRotaryFactor: Float = 1.0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case globalPartialRotaryFactor = "global_partial_rotary_factor"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKeqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case layerTypes = "layer_types"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case useBidirectionalAttention = "use_bidirectional_attention"
    }

    enum VLMCompatibilityCodingKeys: String, CodingKey {
        case attentionBias = "attention_bias"
        case ropeTraditional = "rope_traditional"
    }

    enum QuantizationCodingKeys: String, CodingKey {
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(globalHeadDim, forKey: .globalHeadDim)
        try c.encode(globalPartialRotaryFactor, forKey: .globalPartialRotaryFactor)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(vocabSizePerLayerInput, forKey: .vocabSizePerLayerInput)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encodeIfPresent(numGlobalKeyValueHeads, forKey: .numGlobalKeyValueHeads)
        try c.encode(numKvSharedLayers, forKey: .numKvSharedLayers)
        try c.encode(hiddenSizePerLayerInput, forKey: .hiddenSizePerLayerInput)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(slidingWindowPattern, forKey: .slidingWindowPattern)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(attentionKeqV, forKey: .attentionKeqV)
        try c.encode(finalLogitSoftcapping, forKey: .finalLogitSoftcapping)
        try c.encode(useDoubleWideMlp, forKey: .useDoubleWideMlp)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encodeIfPresent(ropeParameters, forKey: .ropeParameters)
        try c.encode(enableMoeBlock, forKey: .enableMoeBlock)
        try c.encodeIfPresent(numExperts, forKey: .numExperts)
        try c.encodeIfPresent(topKExperts, forKey: .topKExperts)
        try c.encodeIfPresent(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encodeIfPresent(useBidirectionalAttention, forKey: .useBidirectionalAttention)

        if quantizationBits != nil || quantizationGroupSize != nil {
            var qc = encoder.container(keyedBy: QuantizationCodingKeys.self)
            let metadata = Gemma4WeightQuantizationMetadata(
                bits: quantizationBits, groupSize: quantizationGroupSize,
                mode: quantizationMode)
            if let perLayerQuantization {
                try qc.encode(
                    Gemma4WeightQuantizationConfiguration(
                        fallback: metadata,
                        overrides: perLayerQuantization.perLayerQuantization),
                    forKey: .quantization)
            } else {
                try qc.encode(metadata, forKey: .quantization)
            }
        }
    }

    public init(from decoder: Decoder) throws {
        try self.init(from: decoder, defaults: .languageModel)
    }

    public init(
        from decoder: Decoder,
        defaults: Gemma4TextConfigurationDefaults
    ) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let quantizationContainer = try decoder.container(keyedBy: QuantizationCodingKeys.self)
        let baseConfiguration = try? BaseConfiguration(from: decoder)
        let perLayerQuantization = baseConfiguration?.perLayerQuantization
        let compatibilityContainer = try decoder.container(
            keyedBy: VLMCompatibilityCodingKeys.self)
        let isVLM = defaults == .visionLanguageModel
        if isVLM,
            try compatibilityContainer.decodeIfPresent(
                Bool.self, forKey: .attentionBias) == true
        {
            throw DecodingError.dataCorruptedError(
                forKey: .attentionBias,
                in: compatibilityContainer,
                debugDescription:
                    "Gemma4 VLM attention_bias=true is unsupported by the canonical text tower.")
        }
        if isVLM,
            try compatibilityContainer.decodeIfPresent(
                Bool.self, forKey: .ropeTraditional) == true
        {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeTraditional,
                in: compatibilityContainer,
                debugDescription:
                    "Gemma4 VLM rope_traditional=true is unsupported by the canonical text tower.")
        }
        self.fullPartialRotaryFactor = isVLM ? 0.25 : 1.0

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        self.hiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? (isVLM ? 2816 : 1536)
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? (isVLM ? 30 : 35)
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? (isVLM ? 2112 : 6144)
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
            ?? (isVLM ? 16 : 8)
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        self.globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        self.globalPartialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .globalPartialRotaryFactor) ?? 0.25
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? (isVLM ? 8 : 1)
        self.numGlobalKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)

        let decodedHiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput)
            ?? (isVLM ? 0 : 256)
        var decodedVocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput)
            ?? (isVLM ? 0 : 262144)
        if isVLM {
            if decodedHiddenSizePerLayerInput == 0 {
                decodedVocabSizePerLayerInput = 0
            } else if decodedVocabSizePerLayerInput == 0 {
                throw DecodingError.dataCorruptedError(
                    forKey: .hiddenSizePerLayerInput,
                    in: container,
                    debugDescription:
                        "Gemma4 VLM PLE config requires positive vocab_size_per_layer_input when hidden_size_per_layer_input is positive.")
            }
        }
        self.hiddenSizePerLayerInput = decodedHiddenSizePerLayerInput
        self.vocabSizePerLayerInput = decodedVocabSizePerLayerInput
        self.numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers)
            ?? (isVLM ? 0 : 20)
        self.slidingWindow =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
            ?? (isVLM ? 1024 : 512)
        self.slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.attentionKeqV =
            try container.decodeIfPresent(Bool.self, forKey: .attentionKeqV) ?? false
        self.finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
            ?? (isVLM ? 0 : 30.0)
        self.useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp)
            ?? !isVLM
        if let decoded = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            if decoded.isEmpty {
                self.layerTypes = Array(
                    repeating: "sliding_attention", count: numHiddenLayers)
            } else if decoded.count < numHiddenLayers {
                self.layerTypes =
                    decoded
                    + Array(
                        repeating: "sliding_attention",
                        count: numHiddenLayers - decoded.count)
            } else {
                self.layerTypes = Array(decoded.prefix(numHiddenLayers))
            }
        } else if isVLM {
            self.layerTypes = Array(
                repeating: "sliding_attention", count: numHiddenLayers)
        } else {
            var pattern = [String]()
            for i in 0 ..< slidingWindowPattern {
                pattern.append(
                    i == slidingWindowPattern - 1 ? "full_attention" : "sliding_attention")
            }
            var types = [String]()
            while types.count < numHiddenLayers {
                types.append(contentsOf: pattern)
            }
            self.layerTypes = Array(types.prefix(numHiddenLayers))
        }
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        let quantization =
            try quantizationContainer.decodeIfPresent(
                Gemma4WeightQuantizationMetadata.self, forKey: .quantization)
            ?? quantizationContainer.decodeIfPresent(
                Gemma4WeightQuantizationMetadata.self, forKey: .quantizationConfig)
        self.quantizationBits = quantization?.bits
        self.quantizationGroupSize = quantization?.groupSize
        self.quantizationMode =
            perLayerQuantization?.quantization?.mode ?? quantization?.mode ?? .affine
        self.perLayerQuantization = perLayerQuantization
        self.ropeParameters =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)

        self.enableMoeBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        self.topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        self.useBidirectionalAttention =
            try container.decodeIfPresent(String.self, forKey: .useBidirectionalAttention)

        if let ropeParams = ropeParameters {
            if let sliding = ropeParams["sliding_attention"] {
                self.slidingRopeTheta = sliding["rope_theta"]?.asFloat() ?? 10000.0
            }
            if let full = ropeParams["full_attention"] {
                self.fullRopeTheta = full["rope_theta"]?.asFloat() ?? 1_000_000.0
                self.fullPartialRotaryFactor =
                    full["partial_rotary_factor"]?.asFloat()
                    ?? (isVLM ? 0.25 : 1.0)
            }
        }
    }
}

extension Gemma4TextConfiguration {
    public mutating func mergeQuantization(
        _ quantization: BaseConfiguration.Quantization?
    ) {
        guard let quantization else { return }
        quantizationBits = quantization.bits
        quantizationGroupSize = quantization.groupSize
        quantizationMode = quantization.mode
        if var effective = perLayerQuantization {
            effective.quantization = quantization
            perLayerQuantization = effective
        }
    }

    public mutating func mergeQuantization(
        _ quantization: BaseConfiguration.PerLayerQuantization?
    ) {
        guard let quantization else { return }
        if let fallback = quantization.quantization {
            quantizationBits = fallback.bits
            quantizationGroupSize = fallback.groupSize
            quantizationMode = fallback.mode
        }
        perLayerQuantization = quantization
    }
}

extension Gemma4TextConfiguration {

    public func layerUsesSharedKV(layerIdx: Int, forceSharedKV: Bool = false) -> Bool {
        if forceSharedKV { return true }
        guard numKvSharedLayers > 0 else { return false }
        let firstShared = numHiddenLayers - numKvSharedLayers
        return layerIdx >= firstShared
    }
}

// MARK: - Helper Modules

private class RMSNormNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

private struct Gemma4QKVRopeParameters {
    let log2Base: MLXArray
    let frequencies: MLXArray
    let usesFrequencies: Bool
}

private let gemma4QKVNormRopeEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_QKV_NORM_ROPE"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4QKVNormKernel = MLXFast.metalKernel(
        name: "gemma4_b8_qkv_rms_norm_rope_v2_vec1",
    inputNames: [
        "q", "k", "v", "q_weight", "k_weight",
        "position_offsets", "rope_log2_base", "rope_freqs",
    ],
    outputNames: ["q_out", "k_out", "v_out"],
    source: """
        typedef vec<T, 4> T4;
        constexpr uint reads = 4;
        const uint row = threadgroup_position_in_grid.x;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;

        const bool is_query = row < Q_ROWS;
        const bool is_key = row >= Q_ROWS && row < Q_ROWS + K_ROWS;
        const bool weighted = is_query || is_key;
        const device T* input = q;
        const device T* weight = q_weight;
        device T* output_row = q_out;
        uint local_row = row;
        if (!KEY_VALUE_SHARED && row >= Q_ROWS + K_ROWS) {
            input = v;
            output_row = v_out;
            local_row = row - Q_ROWS - K_ROWS;
        } else if (is_key) {
            input = k;
            weight = k_weight;
            output_row = k_out;
            local_row = row - Q_ROWS;
        }

        input += local_row * D + lid * reads;
        output_row += local_row * D;
        device T* output = output_row + lid * reads;
        weight += lid * reads;
        // Keep the pointer inside the V allocation for Q rows even though
        // those rows never dereference it. K rows advance to their matching
        // V row only in the compile-time shared-input variant.
        device T* shared_value_output = v_out;
        if (KEY_VALUE_SHARED && is_key) {
            shared_value_output += local_row * D + lid * reads;
        }

        const T4 vin = *reinterpret_cast<const device T4*>(input);
        float sum = 0.0f;
        for (uint i = 0; i < reads; ++i) {
            const float value = float(vin[i]);
            sum += value * value;
        }
        sum = simd_sum(sum);

        threadgroup float partials[32];
        threadgroup float inverse_rms;
        threadgroup T rounded[D];
        if (simd_group == 0) partials[lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) partials[simd_group] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            sum = simd_sum(partials[lane]);
            if (lane == 0) {
                inverse_rms = metal::precise::rsqrt(sum / float(D) + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (weighted) {
            const T4 wv = *reinterpret_cast<const device T4*>(weight);
            if (APPLY_ROPE) {
                for (uint i = 0; i < reads; ++i) {
                    const uint element = lid * reads + i;
                    const T normalized = T(float(vin[i]) * inverse_rms);
                    // Reproduce the separate norm kernel's BF16 output-store
                    // boundary before any RoPE arithmetic reads the value.
                    rounded[element] = T(wv[i] * normalized);
                }
            } else {
                T4 outv;
                for (uint i = 0; i < reads; ++i) {
                    const T normalized = T(float(vin[i]) * inverse_rms);
                    outv[i] = wv[i] * normalized;
                }
                *reinterpret_cast<device T4*>(output) = outv;
            }
            // Gemma's full-attention K-eq-V layers feed the same raw key
            // projection to K RMSNorm and V RMSNormNoScale. The reduction
            // above is therefore identical for both outputs; keep each
            // output's established final expression, but write V while the
            // exact normalizer and input value are live.
            if (KEY_VALUE_SHARED && is_key) {
                T4 sharedv;
                for (uint i = 0; i < reads; ++i) {
                    const T normalized = T(float(vin[i]) * inverse_rms);
                    sharedv[i] = T(1) * normalized;
                }
                *reinterpret_cast<device T4*>(shared_value_output) = sharedv;
            }
        } else {
            T4 outv;
            for (uint i = 0; i < reads; ++i) {
                const T normalized = T(float(vin[i]) * inverse_rms);
                outv[i] = T(1) * normalized;
            }
            *reinterpret_cast<device T4*>(output) = outv;
        }
        if (APPLY_ROPE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (APPLY_ROPE && weighted && lid * reads < D / 2) {
            const uint heads = is_query ? Q_HEADS : K_HEADS;
            const uint batch = local_row / heads;
            const float L = static_cast<float>(position_offsets[batch]);
            for (uint i = 0; i < reads; ++i) {
                const uint pair = lid * reads + i;
                const float d = static_cast<float>(pair) / static_cast<float>(D / 2);
                const float inv_freq = USE_FREQS
                    ? 1.0f / rope_freqs[pair]
                    : metal::exp2(-d * rope_log2_base[0]);
                const float theta = L * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const float x1 = static_cast<float>(rounded[pair]);
                const float x2 = static_cast<float>(rounded[pair + D / 2]);
                const float rx1 = x1 * costheta - x2 * sintheta;
                const float rx2 = x1 * sintheta + x2 * costheta;
                output_row[pair] = static_cast<T>(rx1);
                output_row[pair + D / 2] = static_cast<T>(rx2);
            }
        }
    """,
    ensureRowContiguous: true
)

private let gemma4QKVNormPrefillKernel = MLXFast.metalKernel(
    name: "gemma4_qkv_rms_norm_head_major_v2",
    inputNames: [
        "q", "k", "q_weight", "k_weight",
        "position_offsets", "rope_freqs",
    ],
    outputNames: ["q_out", "k_out", "v_out"],
    source: """
        constexpr uint reads = 4;
        constexpr uint row_threads = D / reads;
        const uint tid = thread_position_in_threadgroup.x;
        const uint slot = tid / row_threads;
        const uint lid = tid - slot * row_threads;
        const uint row = threadgroup_position_in_grid.x * RPT + slot;
        const uint lane = thread_index_in_simdgroup;
        const uint row_simd = lid / 32;

        threadgroup float partials[RPT][32];
        threadgroup float inv_rms[RPT];
        threadgroup T rounded[RPT][D];
        threadgroup uint row_position[RPT];

        const device T* input = q;
        const device T* weight = q_weight;
        device T* output = q_out;
        // Held inside the V allocation on Q rows, which never dereference it.
        device T* value_output = v_out;
        bool is_key = false;

        if (row < TOTAL_ROWS) {
            if (row < Q_ROWS) {
                const uint b = row / (LQ * HQ);
                const uint rem = row - b * (LQ * HQ);
                const uint l = rem / HQ;
                const uint h = rem - l * HQ;
                row_position[slot] = l;
                input = q + (size_t)row * D;
                output = q_out + (((size_t)b * HQ + h) * LQ + l) * D;
            } else {
                is_key = true;
                const uint krow = row - Q_ROWS;
                const uint b = krow / (LK * HK);
                const uint rem = krow - b * (LK * HK);
                const uint l = rem / HK;
                const uint h = rem - l * HK;
                row_position[slot] = l;
                const size_t off = (((size_t)b * HK + h) * LK + l) * D;
                input = k + (size_t)krow * D;
                weight = k_weight;
                output = k_out + off;
                value_output += off;
            }
        }

        input += lid * reads;
        device T* output_row = output;
        output += lid * reads;
        weight += lid * reads;
        value_output += lid * reads;

        float sum = 0.0f;
        if (row < TOTAL_ROWS) {
            for (uint i = 0; i < reads; ++i) {
                const float value = float(input[i]);
                sum += value * value;
            }
        }
        sum = simd_sum(sum);

        // Slots 2..31 stay exactly zero, so the 32-lane combine returns the
        // two simdgroup partials' sum whatever order the tree adds them in.
        if (row_simd == 0) partials[slot][lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) partials[slot][row_simd] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row_simd == 0) {
            sum = simd_sum(partials[slot][lane]);
            if (lane == 0) {
                inv_rms[slot] = metal::precise::rsqrt(sum / float(D) + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row >= TOTAL_ROWS) return;
        const float inverse_rms = inv_rms[slot];
        for (uint i = 0; i < reads; ++i) {
            const T normalized = T(float(input[i]) * inverse_rms);
            if (APPLY_ROPE) {
                // Stage the weighted norm AS T first — the BF16 memory
                // boundary the separate norm kernel's output store performed
                // before stock RoPE read it.
                rounded[slot][lid * reads + i] = T(weight[i] * normalized);
            } else {
                output[i] = weight[i] * normalized;
            }
            // K rows also carry V: same raw input, same normalizer, and
            // `RMSNormNoScale`'s own final expression.
            if (is_key) {
                value_output[i] = T(1) * normalized;
            }
        }
        if (APPLY_ROPE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (APPLY_ROPE && lid * reads < D / 2) {
            const uint b = row < Q_ROWS
                ? row / (LQ * HQ)
                : (row - Q_ROWS) / (LK * HK);
            const float L =
                static_cast<float>(row_position[slot] + position_offsets[b]);
            for (uint i = 0; i < reads; ++i) {
                const uint pair = lid * reads + i;
                const float inv_freq = 1.0f / rope_freqs[pair];
                const float theta = L * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const float x1 = static_cast<float>(rounded[slot][pair]);
                const float x2 = static_cast<float>(rounded[slot][pair + D / 2]);
                const float rx1 = x1 * costheta - x2 * sintheta;
                const float rx2 = x1 * sintheta + x2 * costheta;
                output_row[pair] = static_cast<T>(rx1);
                output_row[pair + D / 2] = static_cast<T>(rx2);
            }
        }
    """,
    ensureRowContiguous: true
)

private let gemma4QKVNormPrefillEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_QKV_NORM_PREFILL"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private func gemma4FusedQKVNormHeadMajor(
    q: MLXArray,
    k: MLXArray,
    qWeight: MLXArray,
    kWeight: MLXArray,
    eps: Float,
    keyValueShared: Bool, positionOffsets: MLXArray,
    ropeParameters: Gemma4QKVRopeParameters, applyRope: Bool
) -> (q: MLXArray, k: MLXArray, v: MLXArray, appliedRope: Bool)? {
    guard gemma4QKVNormPrefillEnabled, keyValueShared, eps == 1.0e-6,
        positionOffsets.dtype == .int32,
        positionOffsets.size == q.dim(0),
        ropeParameters.frequencies.dtype == .float32,
        q.dtype == .bfloat16, k.dtype == .bfloat16,
        qWeight.dtype == .bfloat16, kWeight.dtype == .bfloat16,
        q.ndim == 4, k.ndim == 4,
        q.dim(0) == k.dim(0), q.dim(0) >= 1,
        q.dim(1) >= 1, k.dim(1) >= 1,
        q.dim(0) * max(q.dim(1), k.dim(1)) >= 1024,
        q.dim(2) == 16, q.dim(3) == k.dim(3),
        (q.dim(3) == 256 && k.dim(2) == 8) || (q.dim(3) == 512 && k.dim(2) == 2),
        qWeight.shape == [q.dim(3)], kWeight.shape == [q.dim(3)]
    else { return nil }

    let (batch, lq, hq, dimension) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let (lk, hk) = (k.dim(1), k.dim(2))
    let qRows = batch * lq * hq
    let rows = qRows + batch * lk * hk
    let rowThreads = dimension / 4
    let rowsPerGroup = 512 / rowThreads
    let groups = (rows + rowsPerGroup - 1) / rowsPerGroup
    let fusedRope = gemma4QKVNormRopeEnabled && applyRope
        && ropeParameters.usesFrequencies
        && ropeParameters.frequencies.size == q.dim(3) / 2
    let outputs = gemma4QKVNormPrefillKernel(
        [q, k, qWeight, kWeight, positionOffsets, ropeParameters.frequencies],
        template: [
            ("T", q.dtype), ("D", dimension), ("Q_ROWS", qRows),
            ("TOTAL_ROWS", rows), ("RPT", rowsPerGroup),
            ("LQ", lq), ("HQ", hq), ("LK", lk), ("HK", hk),
            ("APPLY_ROPE", fusedRope),
        ],
        grid: (groups * rowsPerGroup * rowThreads, 1, 1),
        threadGroup: (rowsPerGroup * rowThreads, 1, 1),
        outputShapes: [
            [batch, hq, lq, dimension], [batch, hk, lk, dimension],
            [batch, hk, lk, dimension],
        ],
        outputDTypes: [q.dtype, q.dtype, q.dtype]
    )
    if fusedRope { CBv2EngageMark.once("qkv-norm-rope-prefill") }
    return (outputs[0], outputs[1], outputs[2], fusedRope)
}

private let gemma4QKVNormPrefillSlidingKernel = MLXFast.metalKernel(
    name: "gemma4_qkv_rms_norm_head_major_sliding_v1",
    inputNames: [
        "q", "k", "v", "q_weight", "k_weight",
        "position_offsets", "rope_log2_base",
    ],
    outputNames: ["q_out", "k_out", "v_out"],
    source: """
        constexpr uint reads = 4;
        constexpr uint row_threads = D / reads;
        const uint tid = thread_position_in_threadgroup.x;
        const uint slot = tid / row_threads;
        const uint lid = tid - slot * row_threads;
        const uint row = threadgroup_position_in_grid.x * RPT + slot;
        const uint lane = thread_index_in_simdgroup;
        const uint row_simd = lid / 32;

        threadgroup float partials[RPT][32];
        threadgroup float inv_rms[RPT];
        threadgroup T rounded[RPT][D];
        threadgroup uint row_position[RPT];

        // Clean per-bank input row pointers: flat [B, L, H, D] rows.
        const device T* input = q;
        const device T* weight = q_weight;
        device T* output = q_out;
        uint local_row = row;
        bool weighted = true;
        if (row >= Q_ROWS + K_ROWS) {
            input = v;
            output = v_out;
            local_row = row - Q_ROWS - K_ROWS;
            weighted = false;
        } else if (row >= Q_ROWS) {
            input = k;
            weight = k_weight;
            output = k_out;
            local_row = row - Q_ROWS;
        }

        if (row < TOTAL_ROWS) {
            // Flat input rows -> head-major [B, H, L, D] output slots; each
            // bank carries its own head count and length.
            const uint h_count = row < Q_ROWS ? HQ : HK;
            const uint l_count = row < Q_ROWS ? LQ : LK;
            const uint b = local_row / (l_count * h_count);
            const uint rem = local_row - b * (l_count * h_count);
            const uint l = rem / h_count;
            const uint h = rem - l * h_count;
            row_position[slot] = l;
            output += (((size_t)b * h_count + h) * l_count + l) * D;
        }

        if (row < Q_ROWS) {
            input = q + (size_t)row * D + lid * reads;
        } else if (row < Q_ROWS + K_ROWS) {
            input = k + (size_t)local_row * D + lid * reads;
        } else {
            input = v + (size_t)local_row * D + lid * reads;
        }
        device T* output_row = output;
        output += lid * reads;
        weight += lid * reads;

        float sum = 0.0f;
        if (row < TOTAL_ROWS) {
            for (uint i = 0; i < reads; ++i) {
                const float value = float(input[i]);
                sum += value * value;
            }
        }
        sum = simd_sum(sum);

        if (row_simd == 0) partials[slot][lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) partials[slot][row_simd] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row_simd == 0) {
            sum = simd_sum(partials[slot][lane]);
            if (lane == 0) {
                inv_rms[slot] = metal::precise::rsqrt(sum / float(D) + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row >= TOTAL_ROWS) return;
        const float inverse_rms = inv_rms[slot];
        for (uint i = 0; i < reads; ++i) {
            const T normalized = T(float(input[i]) * inverse_rms);
            if (APPLY_ROPE && weighted) {
                // The BF16 memory boundary the separate norm kernel's
                // output store performed before stock RoPE read it.
                rounded[slot][lid * reads + i] = T(weight[i] * normalized);
            } else {
                output[i] = weighted ? weight[i] * normalized : T(1) * normalized;
            }
        }
        if (APPLY_ROPE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (APPLY_ROPE && weighted && lid * reads < D / 2) {
            const uint h_count = row < Q_ROWS ? HQ : HK;
            const uint l_count = row < Q_ROWS ? LQ : LK;
            const uint b = local_row / (l_count * h_count);
            const float L =
                static_cast<float>(row_position[slot] + position_offsets[b]);
            for (uint i = 0; i < reads; ++i) {
                const uint pair = lid * reads + i;
                const float d = static_cast<float>(pair) / static_cast<float>(D / 2);
                const float inv_freq = metal::exp2(-d * rope_log2_base[0]);
                const float theta = L * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const float x1 = static_cast<float>(rounded[slot][pair]);
                const float x2 = static_cast<float>(rounded[slot][pair + D / 2]);
                const float rx1 = x1 * costheta - x2 * sintheta;
                const float rx2 = x1 * sintheta + x2 * costheta;
                output_row[pair] = static_cast<T>(rx1);
                output_row[pair + D / 2] = static_cast<T>(rx2);
            }
        }
    """,
    ensureRowContiguous: true
)

private func gemma4FusedQKVNormHeadMajorSliding(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    qWeight: MLXArray,
    kWeight: MLXArray,
    eps: Float,
    positionOffsets: MLXArray,
    ropeParameters: Gemma4QKVRopeParameters, applyRope: Bool
) -> (q: MLXArray, k: MLXArray, v: MLXArray, appliedRope: Bool)? {
    guard gemma4QKVNormPrefillEnabled, eps == 1.0e-6,
        positionOffsets.dtype == .int32,
        positionOffsets.size == q.dim(0),
        !ropeParameters.usesFrequencies,
        ropeParameters.log2Base.dtype == .float32,
        ropeParameters.log2Base.size == 1,
        q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16,
        qWeight.dtype == .bfloat16, kWeight.dtype == .bfloat16,
        q.ndim == 4, k.ndim == 4, v.ndim == 4,
        q.dim(0) == k.dim(0), q.dim(0) >= 1,
        q.dim(1) >= 1, k.dim(1) >= 1, v.shape == k.shape,
        q.dim(0) * max(q.dim(1), k.dim(1)) >= 1024,
        q.dim(2) == 16, q.dim(3) == 256, k.dim(2) == 8,
        qWeight.shape == [q.dim(3)], kWeight.shape == [q.dim(3)]
    else { return nil }

    let (batch, lq, hq, dimension) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let lk = k.dim(1)
    let hk = k.dim(2)
    let qRows = batch * lq * hq
    let kRows = batch * lk * hk
    let rows = qRows + 2 * kRows
    let rowThreads = dimension / 4
    let rowsPerGroup = 512 / rowThreads
    let groups = (rows + rowsPerGroup - 1) / rowsPerGroup
    let fusedRope = gemma4QKVNormRopeEnabled && applyRope
    let outputs = gemma4QKVNormPrefillSlidingKernel(
        [q, k, v, qWeight, kWeight, positionOffsets, ropeParameters.log2Base],
        template: [
            ("T", q.dtype), ("D", dimension), ("Q_ROWS", qRows),
            ("K_ROWS", kRows), ("TOTAL_ROWS", rows), ("RPT", rowsPerGroup),
            ("LQ", lq), ("HQ", hq), ("LK", lk), ("HK", hk),
            ("APPLY_ROPE", fusedRope),
        ],
        grid: (groups * rowsPerGroup * rowThreads, 1, 1),
        threadGroup: (rowsPerGroup * rowThreads, 1, 1),
        outputShapes: [
            [batch, hq, lq, dimension], [batch, hk, lk, dimension],
            [batch, hk, lk, dimension],
        ],
        outputDTypes: [q.dtype, q.dtype, q.dtype]
    )
    if fusedRope { CBv2EngageMark.once("qkv-norm-rope-prefill-sliding") }
    return (outputs[0], outputs[1], outputs[2], fusedRope)
}

private func gemma4FusedQKVNorm(
    q: MLXArray, k: MLXArray, v: MLXArray,
    qWeight: MLXArray, kWeight: MLXArray, eps: Float,
    keyValueShared: Bool, positionOffsets: MLXArray,
    ropeParameters: Gemma4QKVRopeParameters, applyRope: Bool
) -> (q: MLXArray, k: MLXArray, v: MLXArray, appliedRope: Bool)? {
    let qkvRows = q.ndim == 4 ? q.dim(0) : 0
    guard eps == 1.0e-6,
        q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16,
        qWeight.dtype == .bfloat16, kWeight.dtype == .bfloat16,
        positionOffsets.dtype == .int32, positionOffsets.shape == [qkvRows],
        ropeParameters.log2Base.dtype == .float32, ropeParameters.log2Base.size == 1,
        ropeParameters.frequencies.dtype == .float32,
        q.ndim == 4, k.ndim == 4, v.ndim == 4,
        qkvRows == 8
            || (CBv2MTPWideVerifyContext.active && qkvRows % 8 == 0 && qkvRows > 0),
        q.dim(1) == 1, q.dim(2) == 16,
        k.dim(0) == qkvRows, k.dim(1) == 1, v.shape == k.shape,
        q.dim(3) == k.dim(3),
        (q.dim(3) == 256 && k.dim(2) == 8) || (q.dim(3) == 512 && k.dim(2) == 2),
        qWeight.shape == [q.dim(3)], kWeight.shape == [q.dim(3)],
        !keyValueShared || v.shape == k.shape,
        !ropeParameters.usesFrequencies
            || ropeParameters.frequencies.size == q.dim(3) / 2
    else { return nil }

    let dimension = q.dim(3)
    let qRows = qkvRows * 16
    let kRows = qkvRows * k.dim(2)
    let threads = dimension / 4
    let fusedRope = gemma4QKVNormRopeEnabled && applyRope
    let normRows = qRows + kRows + (keyValueShared ? 0 : kRows)
    let outputs = gemma4QKVNormKernel(
        [q, k, v, qWeight, kWeight, positionOffsets,
         ropeParameters.log2Base, ropeParameters.frequencies],
        template: [
            ("T", q.dtype), ("D", dimension), ("Q_ROWS", qRows), ("K_ROWS", kRows),
            ("Q_HEADS", 16), ("K_HEADS", k.dim(2)),
            ("KEY_VALUE_SHARED", keyValueShared), ("APPLY_ROPE", fusedRope),
            ("USE_FREQS", ropeParameters.usesFrequencies),
        ],
        grid: (normRows * threads, 1, 1), threadGroup: (threads, 1, 1),
        outputShapes: fusedRope
            ? [[qkvRows, 16, 1, dimension], [qkvRows, k.dim(2), 1, dimension], v.shape]
            : [q.shape, k.shape, v.shape],
        outputDTypes: [q.dtype, k.dtype, v.dtype]
    )
    if fusedRope { CBv2EngageMark.once("qkv-norm-rope") }
    return (outputs[0], outputs[1], outputs[2], fusedRope)
}

private class ScaledLinear: Module {
    let weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.weight = MLXArray.zeros([outFeatures, inFeatures])
        self.scalar = scalar
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T) * scalar
    }
}

@inline(__always)
internal func gemma4CapturePositionOffset(from cache: KVCache?) -> Gemma4.PositionOffset {
    if let compilableRot = cache as? CompilableRotatingKVCache {
        .graphArray(compilableRot.offsetArray + 0)
    } else if let compilable = cache as? CompilableKVCache {
        .graphArray(compilable.offsetArray + 0)
    } else if let batchCache = cache as? BatchPositionedKVCache {
        .batch(batchCache.batchOffset + 0)
    } else {
        .scalar(cache?.offset ?? 0)
    }
}

@inline(__always)
internal func gemma4ApplyRotaryPosition<R: RoPELayer>(
    _ rope: R,
    to x: MLXArray,
    offset: Gemma4.PositionOffset
) -> MLXArray {
    switch offset {
    case .scalar(let value):
        rope(x, offset: value)
    case .batch(let values):
        rope(x, offset: values)
    case .graphArray(let offsetArray):
        rope(x, offset: offsetArray)
    }
}

private func gemma4AttentionFallback(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let (B, nQHeads, L, D) = (
        queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)
    )
    let nKVHeads = keys.dim(1)
    let repeats = nQHeads / nKVHeads

    var q = queries * scale
    var k = keys
    var v = values
    if repeats > 1 {
        q = q.reshaped([B, nKVHeads, repeats, L, D])
        k = expandedDimensions(k, axis: 2)
        v = expandedDimensions(v, axis: 2)
    }

    var scores = matmul(q, k.swappedAxes(-1, -2))

    func applyMask(_ maskArray: MLXArray) {
        var mask = maskArray
        if scores.ndim == 5 && mask.ndim == 4 && mask.dim(0) == scores.dim(0) {
            mask = expandedDimensions(mask, axis: 2)
        }
        if mask.dtype == .bool {
            scores = MLX.where(
                mask, scores, MLXArray(-Float.infinity, dtype: scores.dtype))
        } else {
            scores = scores + mask
        }
    }

    switch mask {
    case .none:
        break
    case .causal:
        let qL = scores.dim(-2)
        let kL = scores.dim(-1)
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1),
            expandedDimensions(kIndices, axis: -2))
        applyMask(causalMask)
    case .array(let maskArray):
        applyMask(maskArray)
    case .arrays(let maskArrays):
        if let maskArray = maskArrays.first {
            applyMask(maskArray)
        }
    }

    var probs = softmax(scores.asType(.float32), axis: -1, precise: true)
    probs = MLX.where(probs .!= probs, MLXArray(Float(0)), probs)
    scores = probs.asType(scores.dtype)
    var output = matmul(scores, v)
    if repeats > 1 {
        output = output.reshaped([B, nQHeads, L, values.dim(3)])
    }
    return output
}

// MARK: - Attention

private class Gemma4Attention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let effectiveHeadDim: Int
    let nHeads: Int
    let nKvHeads: Int
    let useKeqV: Bool
    let usesSharedKV: Bool
    let scale: Float
    let qkvRopeParameters: Gemma4QKVRopeParameters

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    @ModuleInfo(key: "v_norm") var vNorm: RMSNormNoScale?

    @ModuleInfo var rope: RoPELayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int, forceSharedKV: Bool = false) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.usesSharedKV = config.layerUsesSharedKV(
            layerIdx: layerIdx, forceSharedKV: forceSharedKV)

        self.effectiveHeadDim =
            isSliding ? config.headDim : config.globalHeadDim

        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads

        self.useKeqV = config.attentionKeqV && !isSliding
        if !isSliding, let globalKvHeads = config.numGlobalKeyValueHeads {
            self.nKvHeads = globalKvHeads
        } else {
            self.nKvHeads = config.numKeyValueHeads
        }

        self.scale = 1.0

        self._qProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        if !usesSharedKV {
            self._kProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
            if !useKeqV {
                self._vProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
            }
            self._kNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
            self._vNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)
        }
        self._oProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)

        if isSliding {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.slidingRopeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
            self.qkvRopeParameters = Gemma4QKVRopeParameters(
                log2Base: MLXArray([log2f(config.slidingRopeTheta)]),
                frequencies: MLXArray([Float.infinity]), usesFrequencies: false)
        } else {
            let fullRope = initializeRope(
                dims: effectiveHeadDim, base: config.fullRopeTheta, traditional: false,
                scalingConfig: [
                    "type": .string("proportional"),
                    "partial_rotary_factor": .float(config.fullPartialRotaryFactor),
                ],
                maxPositionEmbeddings: nil)
            guard let proportional = fullRope as? ProportionalRoPE,
                let frequencies = proportional.frequencyTable
            else {
                preconditionFailure("Gemma4 full-attention RoPE requires a frequency table")
            }
            self.rope = proportional
            self.qkvRopeParameters = Gemma4QKVRopeParameters(
                log2Base: MLXArray([Float.zero]), frequencies: frequencies,
                usesFrequencies: true)
        }

        super.init()
    }

    @inline(__always)
    private func tierProjection(
        _ layer: Linear, _ x: MLXArray, rsTable: MLXArray? = nil
    ) -> MLXArray {
        guard let quantized = layer as? QuantizedLinear,
            quantized.bias == nil,
            let projected = CBv2AttentionQKVMMA8V1.matmul(
                x: x,
                weight: quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode,
                rsTable: rsTable)
        else { return layer(x) }
        return projected
    }

    @inline(__always)
    private func fusedQKProjection(
        _ x: MLXArray, rsTable: MLXArray? = nil
    ) -> (MLXArray, MLXArray)? {
        guard let q = qProj as? QuantizedLinear, q.bias == nil,
            let kProj, let k = kProj as? QuantizedLinear, k.bias == nil,
            q.groupSize == k.groupSize, q.bits == k.bits, q.mode == k.mode
        else { return nil }
        return CBv2AttentionQKVMMA8V1.fusedQKMatmul(
            x: x,
            qWeight: q.weight, qScales: q.scales, qBiases: q.biases,
            kWeight: k.weight, kScales: k.scales, kBiases: k.biases,
            groupSize: q.groupSize, bits: q.bits, mode: q.mode,
            cacheKey: ObjectIdentifier(q),
            rsTable: rsTable)
    }

    @inline(__always)
    private func fusedQKVProjection(
        _ x: MLXArray, rsTable: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard let q = qProj as? QuantizedLinear, q.bias == nil,
            let kProj, let k = kProj as? QuantizedLinear, k.bias == nil,
            let vProj, let v = vProj as? QuantizedLinear, v.bias == nil,
            q.groupSize == k.groupSize, q.groupSize == v.groupSize,
            q.bits == k.bits, q.bits == v.bits,
            q.mode == k.mode, q.mode == v.mode
        else { return nil }
        return CBv2AttentionQKVMMA8V1.fusedQKVMatmul(
            x: x,
            qWeight: q.weight, qScales: q.scales, qBiases: q.biases,
            kWeight: k.weight, kScales: k.scales, kBiases: k.biases,
            vWeight: v.weight, vScales: v.scales, vBiases: v.biases,
            groupSize: q.groupSize, bits: q.bits, mode: q.mode,
            cacheKey: ObjectIdentifier(q),
            rsTable: rsTable)
    }

    @inline(__always)
    private func outputProjection(_ x: MLXArray) -> MLXArray {
        guard let quantized = oProj as? QuantizedLinear,
            quantized.bias == nil,
            let projected = CBv2AttentionOQMVV1.matmul(
                x: x,
                weight: quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode,
                rsTable: CBv2AttentionOQMVV1.runsumTable(for: x))
        else { return oProj(x) }
        return projected
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        positionOffset: Gemma4.PositionOffset? = nil,
        v2SharedSource: (any CBv2AttendingLayerCache)? = nil,
        outputStart: Int = 0,
        useLastQueryPrefill: Bool = false,
        wideColumns: Int = 1,
        qkvRunsumTable: MLXArray? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4.PositionOffset) {
        if let layerCacheV2 = cache as? (any CBv2AttendingLayerCache) {
            return forwardV2(
                x, layerCache: layerCacheV2, source: v2SharedSource,
                sharedKV: sharedKV, positionOffset: positionOffset,
                outputStart: outputStart, useLastQueryPrefill: useLastQueryPrefill,
                wideColumns: wideColumns, carriedQKVRunsumTable: qkvRunsumTable)
        }
        precondition(
            outputStart == 0 && !useLastQueryPrefill,
            "Gemma4: prompt output narrowing is a CBv2-only path")

        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
        queries = qNorm(queries)

        let keys: MLXArray
        let values: MLXArray
        let activePositionOffset = positionOffset ?? gemma4CapturePositionOffset(from: cache)

        if let (sharedK, sharedV) = sharedKV {
            keys = sharedK
            values = sharedV
        } else {
            guard let kProj, let kNorm, let vNorm else {
                preconditionFailure("Gemma4 shared-KV layers require sharedKV input")
            }

            let kRaw = kProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
            var k = kNorm(kRaw)
            k = k.transposed(0, 2, 1, 3)
            k = gemma4ApplyRotaryPosition(rope, to: k, offset: activePositionOffset)

            var v: MLXArray
            if let vProj {
                v = vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
            } else {
                v = kRaw
            }
            v = vNorm(v)
            v = v.transposed(0, 2, 1, 3)

            if let cache {
                let (updatedK, updatedV) = cache.update(keys: k, values: v)
                keys = updatedK
                values = updatedV
            } else {
                keys = k
                values = v
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: activePositionOffset)

        var adjustedMask = mask
        if case .array(let maskArray) = mask {
            let keysSeqLen = keys.dim(2)
            if maskArray.dim(-1) != keysSeqLen {
                adjustedMask = .array(maskArray[.ellipsis, 0 ..< keysSeqLen])
            }
        }

        let hasCachedPrefix: Bool
        switch activePositionOffset {
        case .scalar(let offset):
            hasCachedPrefix = offset > 0
        case .batch:
            hasCachedPrefix = true
        case .graphArray:
            hasCachedPrefix = true
        }

        let attentionInputDType = queries.dtype
        var attentionQueries = queries
        var attentionKeys = keys
        var attentionValues = values
        if attentionInputDType == .float16 {
            attentionQueries = attentionQueries.asType(.float32)
            attentionKeys = attentionKeys.asType(.float32)
            attentionValues = attentionValues.asType(.float32)
        }

        let attentionRaw: MLXArray
        if L > 1 && hasCachedPrefix {
            attentionRaw = gemma4AttentionFallback(
                queries: attentionQueries,
                keys: attentionKeys,
                values: attentionValues,
                scale: scale,
                mask: adjustedMask ?? .none)
        } else if sharedKV != nil,
            let mirrored = Gemma4DrafterMirrorAttention.attend(
                queries: attentionQueries, isSliding: isSliding)
        {
            attentionRaw = mirrored
        } else {
            attentionRaw = MLXFast.scaledDotProductAttention(
                queries: attentionQueries,
                keys: attentionKeys,
                values: attentionValues,
                scale: scale,
                mask: adjustedMask ?? .none
            )
        }
        let attention =
            attentionInputDType == .float16
            ? attentionRaw.asType(.float16) : attentionRaw

        let output = attention
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return (outputProjection(output), (keys, values), activePositionOffset)
    }

    private func forwardV2(
        _ x: MLXArray,
        layerCache: any CBv2AttendingLayerCache,
        source: (any CBv2AttendingLayerCache)?,
        sharedKV: (MLXArray, MLXArray)?,
        positionOffset: Gemma4.PositionOffset?,
        outputStart: Int = 0,
        useLastQueryPrefill: Bool = false,
        wideColumns: Int = 1,
        carriedQKVRunsumTable: MLXArray? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4.PositionOffset) {
        let (B, L) = (x.dim(0), x.dim(1))
        precondition(
            outputStart >= 0 && outputStart < L,
            "Gemma4: output narrowing start \(outputStart) outside chunk length \(L)")
        precondition(
            wideColumns == 1 || (L == 1 && B % wideColumns == 0 && !useLastQueryPrefill),
            "Gemma4: wide verify rows must be single-token and a multiple of the column count")

        let lastQueryCache: (any CBv2LastQueryPrefillLayerCache)? =
            useLastQueryPrefill
            ? layerCache as? (any CBv2LastQueryPrefillLayerCache) : nil
        if useLastQueryPrefill {
            precondition(
                lastQueryCache != nil,
                "Gemma4 last-query prefill requires a capable layer cache")
            precondition(
                B > 0 && L > 1 && outputStart == L - 1 && !isSliding && !usesSharedKV,
                "Gemma4 last-query prefill requires a final-row, non-shared, full-attention layer")
        }

        let queryInput = lastQueryCache == nil ? x : x[0..., outputStart..., 0...]
        let queryLength = queryInput.dim(1)

        let qkvRunsumTable = CBv2AttentionQKVMMA8V1.rsPrepassEnabled
            ? (carriedQKVRunsumTable ?? CBv2AttentionQKVMMA8V1.runsumTable(for: x))
            : nil

        let fusedQKV: (MLXArray, MLXArray, MLXArray)? =
            (lastQueryCache == nil && !usesSharedKV && vProj != nil)
            ? fusedQKVProjection(x, rsTable: qkvRunsumTable) : nil
        let fusedQK: (MLXArray, MLXArray)? =
            (lastQueryCache == nil && !usesSharedKV && vProj == nil)
            ? fusedQKProjection(x, rsTable: qkvRunsumTable) : nil
        let queryRaw = (
            fusedQKV?.0 ?? fusedQK?.0
                ?? tierProjection(qProj, queryInput, rsTable: qkvRunsumTable)
        ).reshaped(B, queryLength, nHeads, effectiveHeadDim)

        if usesSharedKV {
            guard let source, let positionOffset, let sharedKV else {
                preconditionFailure(
                    """
                    Gemma4 CBv2 shared-KV layer \(layerIdx) requires the source \
                    layer cache, its captured position offsets, and its per-step \
                    K/V (threaded by Gemma4TextModelInner)
                    """)
            }
            var queries = qNorm(queryRaw).transposed(0, 2, 1, 3)
            queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: positionOffset)
            let outputDType = queries.dtype
            let attentionQueries =
                outputDType == .float16 ? queries.asType(.float32) : queries
            let attention = layerCache.attendBorrowing(
                source: source, queries: attentionQueries, scale: scale, sinks: nil)
            var output = attention.transposed(0, 2, 1, 3).reshaped(B, L, -1)
            if outputStart > 0 {
                output = output[0..., outputStart..., 0...]
            }
            if output.dtype != outputDType {
                output = output.asType(outputDType)
            }
            return (outputProjection(output), sharedKV, positionOffset)
        }

        guard let kProj, let kNorm, let vNorm else {
            preconditionFailure("Gemma4 non-shared layers require K/V projection modules")
        }

        let capturedOffsets: MLXArray
        let captured: Gemma4.PositionOffset
        if let positionOffset {
            guard case .batch(let offsets) = positionOffset else {
                preconditionFailure("Gemma4 CBv2 position offsets must be a per-row batch")
            }
            capturedOffsets = offsets
            captured = positionOffset
        } else {
            capturedOffsets = layerCache.positionOffsets + 0
            captured = .batch(capturedOffsets)
        }

        let queryPositionOffset: Gemma4.PositionOffset =
            lastQueryCache == nil
            ? captured
            : .batch(capturedOffsets + Int32(outputStart))
        let kRaw = (
            fusedQKV?.1 ?? fusedQK?.1
                ?? tierProjection(kProj, x, rsTable: qkvRunsumTable)
        ).reshaped(B, L, nKvHeads, effectiveHeadDim)
        let vRaw: MLXArray
        if let fusedV = fusedQKV?.2 {
            vRaw = fusedV.reshaped(B, L, nKvHeads, effectiveHeadDim)
        } else if let vProj {
            vRaw = tierProjection(vProj, x, rsTable: qkvRunsumTable)
                .reshaped(B, L, nKvHeads, effectiveHeadDim)
        } else {
            vRaw = kRaw
        }

        var queries: MLXArray
        var k: MLXArray
        var v: MLXArray
        var appliedRope = false
        if let normalized = gemma4FusedQKVNorm(
            q: queryRaw, k: kRaw, v: vRaw,
            qWeight: qNorm.weight, kWeight: kNorm.weight, eps: config.rmsNormEps,
            keyValueShared: vProj == nil, positionOffsets: capturedOffsets,
            ropeParameters: qkvRopeParameters, applyRope: lastQueryCache == nil)
        {
            appliedRope = normalized.appliedRope
            queries = appliedRope ? normalized.q : normalized.q.transposed(0, 2, 1, 3)
            k = appliedRope ? normalized.k : normalized.k.transposed(0, 2, 1, 3)
            v = normalized.v.transposed(0, 2, 1, 3)
        } else if let headMajor = gemma4FusedQKVNormHeadMajor(
            q: queryRaw, k: kRaw,
            qWeight: qNorm.weight, kWeight: kNorm.weight, eps: config.rmsNormEps,
            keyValueShared: vProj == nil, positionOffsets: capturedOffsets,
            ropeParameters: qkvRopeParameters, applyRope: lastQueryCache == nil)
        {
            (queries, k, v) = (headMajor.q, headMajor.k, headMajor.v)
            appliedRope = headMajor.appliedRope
        } else if let sliding = gemma4FusedQKVNormHeadMajorSliding(
            q: queryRaw, k: kRaw, v: vRaw,
            qWeight: qNorm.weight, kWeight: kNorm.weight, eps: config.rmsNormEps,
            positionOffsets: capturedOffsets,
            ropeParameters: qkvRopeParameters, applyRope: lastQueryCache == nil)
        {
            (queries, k, v) = (sliding.q, sliding.k, sliding.v)
            appliedRope = sliding.appliedRope
        } else {
            queries = qNorm(queryRaw).transposed(0, 2, 1, 3)
            k = kNorm(kRaw).transposed(0, 2, 1, 3)
            v = vNorm(vRaw).transposed(0, 2, 1, 3)
        }

        if !appliedRope {
            queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: queryPositionOffset)
            k = gemma4ApplyRotaryPosition(rope, to: k, offset: captured)
        }

        let outputDType = queries.dtype
        let attentionQueries =
            outputDType == .float16 ? queries.asType(.float32) : queries
        let attention: MLXArray
        if let lastQueryCache {
            attention = lastQueryCache.updateAndAttendLastQuery(
                queries: attentionQueries, keys: k, values: v, scale: scale, sinks: nil)
        } else if wideColumns > 1,
            let direct = (layerCache as? CBv2LayerCache)?.updateAndAttendColumns(
                queries: attentionQueries, keys: k, values: v,
                columns: wideColumns, scale: scale, sinks: nil)
        {
            attention = direct
        } else if wideColumns > 1 {
            let cohort = B / wideColumns
            func fold(_ t: MLXArray) -> MLXArray {
                t.reshaped([cohort, wideColumns, t.dim(1), t.dim(3)]).transposed(0, 2, 1, 3)
            }
            let folded = layerCache.updateAndAttend(
                queries: fold(attentionQueries), keys: fold(k), values: fold(v),
                scale: scale, sinks: nil)
            attention = folded.transposed(0, 2, 1, 3).reshaped([B, folded.dim(1), 1, folded.dim(3)])
        } else {
            attention = layerCache.updateAndAttend(
                queries: attentionQueries, keys: k, values: v, scale: scale, sinks: nil)
        }

        var output = attention.transposed(0, 2, 1, 3).reshaped(B, queryLength, -1)
        if lastQueryCache == nil && outputStart > 0 {
            output = output[0..., outputStart..., 0...]
        }
        if output.dtype != outputDType {
            output = output.asType(outputDType)
        }
        return (outputProjection(output), (k, v), captured)
    }
}

// MARK: - MoE (26B-A4B)

public enum Gemma4RouterProbe {
    nonisolated(unsafe) public static var recorder:
        ((_ expertScores: MLXArray, _ topKIndices: MLXArray) -> Void)?
}

enum Gemma4RouteStats {
    static let enabled: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_ROUTE_STATS"] == "1"

    static func record(layer: Int, _ topKIndices: MLXArray) {
        let rows = topKIndices.size / topKIndices.dim(-1)
        guard rows <= 64 else { return }
        let keys = topKIndices.flattened().asType(.uint32).asArray(UInt32.self).sorted()
        var runs = [Int](repeating: 0, count: 5)
        var unique = 0
        var i = 0
        while i < keys.count {
            var j = i + 1
            while j < keys.count, keys[j] == keys[i] { j += 1 }
            runs[min(j - i, 5) - 1] += 1
            unique += 1
            i = j
        }
        let hist = (0 ..< 5).map { "\($0 == 4 ? "5+" : "\($0 + 1)")=\(runs[$0])" }
            .joined(separator: " ")
        let wide = CBv2MTPWideVerifyContext.active ? 1 : 0
        FileHandle.standardError.write(
            Data(
                "[route-stats] layer=\(layer) rows=\(rows) asg=\(keys.count) uniq=\(unique) runs \(hist) wide=\(wide)\n"
                    .utf8))
    }
}

private enum Gemma4FusedRouterTop8 {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_FUSED_ROUTER_TOP8"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    private static let rows = 8
    private static let experts = 128
    private static let selected = 8

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_fused_router_top8_e128_k8_bf16_v1",
        inputNames: ["scores", "pes"],
        outputNames: ["inds", "wts"],
        source: """
            constexpr int SIMD_SIZE = 32;
            constexpr int N_READS = 4;
            constexpr int KTH = E - K;

            const int row = int(threadgroup_position_in_grid.x);
            const int lid = int(thread_position_in_threadgroup.x);

            threadgroup float vals[E];
            threadgroup float topv[K];
            threadgroup uint topi[K];
            threadgroup float local_max[SIMD_SIZE];
            threadgroup float local_normalizer[SIMD_SIZE];

            const device T* srow = scores + row * E;
            for (int i = lid; i < E; i += SIMD_SIZE) {
                vals[i] = float(srow[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Stable-argsort position by predecessor counting under sort.h's
            // LessThan comparator (NaN orders after every non-NaN; ties keep
            // the original index order because the merge sort is stable).
            // Position == #{i : less(v_i, v_e)} + #{i < e : neither less} —
            // a permutation, so the writes below never collide.
            for (int j = 0; j < E / SIMD_SIZE; ++j) {
                const int e = lid + j * SIMD_SIZE;
                const float v = vals[e];
                const bool v_nan = isnan(v);
                int rank = 0;
                for (int i = 0; i < E; ++i) {
                    const float u = vals[i];
                    const bool u_nan = isnan(u);
                    bool u_less_v;
                    bool v_less_u;
                    if (u_nan || v_nan) {
                        u_less_v = !u_nan && v_nan;
                        v_less_u = !v_nan && u_nan;
                    } else {
                        u_less_v = u < v;
                        v_less_u = v < u;
                    }
                    if (u_less_v || (!v_less_u && i < e)) {
                        ++rank;
                    }
                }
                if (rank >= KTH) {
                    const int p = rank - KTH;
                    inds[row * K + p] = uint(e);
                    topi[p] = uint(e);
                    topv[p] = v;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // softmax_single_row<T, float, N_READS=4> transcription
            // (softmax.h) at axis_size = K on one 32-thread simdgroup, with
            // the stock bf16 per-expert-scale multiply fused into the write.
            const int simd_lane_id = int(thread_index_in_simdgroup);
            const int simd_group_id = int(simdgroup_index_in_threadgroup);

            float ld[N_READS];
            const int base = lid * N_READS;
            if (base + N_READS <= K) {
                for (int i = 0; i < N_READS; i++) {
                    ld[i] = topv[base + i];
                }
            } else {
                for (int i = 0; i < N_READS; i++) {
                    ld[i] = ((base + i) < K) ? topv[base + i] : Limits<float>::min;
                }
            }
            if (simd_group_id == 0) {
                local_max[simd_lane_id] = Limits<float>::min;
                local_normalizer[simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float maxval = Limits<float>::finite_min;
            for (int i = 0; i < N_READS; i++) {
                maxval = (maxval < ld[i]) ? ld[i] : maxval;
            }
            maxval = simd_max(maxval);
            if (simd_lane_id == 0) {
                local_max[simd_group_id] = maxval;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                maxval = simd_max(local_max[simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_max[0] = maxval;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maxval = local_max[0];

            float normalizer = 0;
            for (int i = 0; i < N_READS; i++) {
                float exp_x = fast::exp(ld[i] - maxval);
                ld[i] = exp_x;
                normalizer += exp_x;
            }
            normalizer = simd_sum(normalizer);
            if (simd_lane_id == 0) {
                local_normalizer[simd_group_id] = normalizer;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                normalizer = simd_sum(local_normalizer[simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_normalizer[0] = normalizer;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            normalizer = 1 / local_normalizer[0];

            if (base + N_READS <= K) {
                for (int i = 0; i < N_READS; i++) {
                    const T w = T(ld[i] * normalizer);
                    wts[row * K + base + i] = w * pes[topi[base + i]];
                }
            } else {
                for (int i = 0; i < N_READS; i++) {
                    if ((base + i) < K) {
                        const T w = T(ld[i] * normalizer);
                        wts[row * K + base + i] = w * pes[topi[base + i]];
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )

    static func apply(
        expertScores: MLXArray, perExpertScale: MLXArray, topK: Int
    ) -> (indices: MLXArray, weights: MLXArray)? {
        guard enabled,
            topK == selected,
            expertScores.ndim == 3,
            expertScores.dim(0) == rows
                || (CBv2MTPWideVerifyContext.active && expertScores.dim(0) % rows == 0
                    && expertScores.dim(0) > 0),
            expertScores.dim(1) == 1,
            expertScores.dim(2) == experts,
            expertScores.dtype == .bfloat16,
            perExpertScale.ndim == 1,
            perExpertScale.dim(0) == experts,
            perExpertScale.dtype == .bfloat16
        else { return nil }
        CBv2EngageMark.once("router-top8")
        let rows = expertScores.dim(0)

        let outputs = kernel(
            [expertScores, perExpertScale],
            template: [
                ("T", expertScores.dtype),
                ("E", experts),
                ("K", selected),
            ],
            grid: (rows * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[rows, 1, selected], [rows, 1, selected]],
            outputDTypes: [.uint32, .bfloat16]
        )
        return (outputs[0], outputs[1])
    }
}

private enum Gemma4RouterFinalistsV1 {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_ROUTER_FINALISTS32"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_router_finalists32_stable_bf16_v1",
        inputNames: ["scores"],
        outputNames: ["indices"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const uint expert = group * 32u + lane;
            // Pack the unchanged BF16 bits and the original expert index.
            // This is a payload, NOT an unsigned floating-point ordinal:
            // comparisons below retain native BF16 LessThan semantics.
            uint item = (uint(bfloat16_to_uint16(scores[row * 128u + expert])) << 7)
                | expert;
            threadgroup uint finalists[32];

            for (uint width = 2u; width <= 32u; width <<= 1) {
                for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                    const uint other = simd_shuffle_xor(item, ushort(stride));
                    const bool otherBefore = gemma4_finalists_before(other, item);
                    const bool takeMinimum = ((lane & width) == 0u)
                        == ((lane & stride) == 0u);
                    if (takeMinimum ? otherBefore : !otherBefore) item = other;
                }
            }

            if (lane >= 24u) {
                finalists[group * 8u + lane - 24u] = item;
            }
            // All four complete SIMD groups participate in this barrier.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (group == 0u) {
                item = finalists[lane];
                for (uint width = 2u; width <= 32u; width <<= 1) {
                    for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                        const uint other = simd_shuffle_xor(item, ushort(stride));
                        const bool otherBefore = gemma4_finalists_before(other, item);
                        const bool takeMinimum = ((lane & width) == 0u)
                            == ((lane & stride) == 0u);
                        if (takeMinimum ? otherBefore : !otherBefore) item = other;
                    }
                }
                if (lane >= 24u) indices[row * 8u + lane - 24u] = item & 127u;
            }
        """,
        header: """
            inline bool gemma4_finalists_before(uint a, uint b) {
                const bfloat16_t av = uint16_to_bfloat16(uint16_t(a >> 7));
                const bfloat16_t bv = uint16_to_bfloat16(uint16_t(b >> 7));
                const bool an = metal::isnan(av);
                const bool bn = metal::isnan(bv);
                bool ab;
                bool ba;
                if (an | bn) {
                    ab = (!an) & bn;
                    ba = (!bn) & an;
                } else {
                    ab = av < bv;
                    ba = bv < av;
                }
                return ab || (!ba && (a & 127u) < (b & 127u));
            }
        """,
        ensureRowContiguous: true
    )

    static func apply(_ scores: MLXArray, topK: Int, kth: Int) -> MLXArray? {
        guard enabled, topK == 8, kth == 120,
            scores.ndim == 3,
            scores.dim(0) == 8
                || (CBv2MTPWideVerifyContext.active && scores.dim(0) % 8 == 0
                    && scores.dim(0) > 0),
            scores.dim(1) == 1, scores.dim(2) == 128,
            scores.dtype == .bfloat16
        else { return nil }
        let rows = scores.dim(0)
        return kernel(
            [scores],
            grid: (rows * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[rows, 1, 8]],
            outputDTypes: [.uint32]
        )[0]
    }

    static let prefillEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_ROUTER_FINALISTS32_PREFILL"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static func applyPrefill(_ scores: MLXArray, topK: Int, kth: Int) -> MLXArray? {
        guard prefillEnabled, topK == 8, kth == 120,
            scores.ndim == 3, scores.dim(0) == 8,
            scores.dim(1) > 1, scores.dim(2) == 128,
            scores.dtype == .bfloat16
        else { return nil }
        let b = scores.dim(0)
        let l = scores.dim(1)
        let rows = b * l
        CBv2EngageMark.once("router-finalists32-prefill")
        return kernel(
            [scores],
            grid: (rows * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[b, l, 8]],
            outputDTypes: [.uint32]
        )[0]
    }
}

private enum Gemma4RouteGlueFoldV1 {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_GLUE_FOLD"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    struct Fold {
        let indices: MLXArray
        let weights: MLXArray
        let table: SwitchRouteTable
    }

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_route_monolithic_top8_e128_k8_bf16_v2",
        inputNames: ["scores", "pes"],
        outputNames: ["indices", "weights", "row_order", "sorted_keys", "inverse_order"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint row = tid / 128u;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint group = sg % 4u;
            const uint expert = group * 32u + lane;
            // Phase 1 -- the incumbent finalists32 selection, verbatim, with
            // the per-row threadgroup slices offset by `row`. Pack the
            // unchanged BF16 bits and the original expert index; comparisons
            // retain native BF16 LessThan semantics.
            uint item = (uint(bfloat16_to_uint16(scores[row * 128u + expert])) << 7)
                | expert;
            threadgroup uint finalists[256];
            threadgroup uint sel[64];

            #pragma clang loop unroll(full)
            for (uint width = 2u; width <= 32u; width <<= 1) {
                #pragma clang loop unroll(full)
                for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                    const uint other = simd_shuffle_xor(item, ushort(stride));
                    const bool otherBefore = gemma4_finalists_before(other, item);
                    const bool takeMinimum = ((lane & width) == 0u)
                        == ((lane & stride) == 0u);
                    if (takeMinimum ? otherBefore : !otherBefore) item = other;
                }
            }

            if (lane >= 24u) {
                finalists[row * 32u + group * 8u + lane - 24u] = item;
            }
            // All thirty-two complete SIMD groups participate in this barrier.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Group 0 reduces 32 finalists to top 8 and evaluates in-register softmax + scaling
            if (group == 0u) {
                item = finalists[row * 32u + lane];
                #pragma clang loop unroll(full)
                for (uint width = 2u; width <= 32u; width <<= 1) {
                    #pragma clang loop unroll(full)
                    for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                        const uint other = simd_shuffle_xor(item, ushort(stride));
                        const bool otherBefore = gemma4_finalists_before(other, item);
                        const bool takeMinimum = ((lane & width) == 0u)
                            == ((lane & stride) == 0u);
                        if (takeMinimum ? otherBefore : !otherBefore) item = other;
                    }
                }

                float score = (lane >= 24u) ? float(uint16_to_bfloat16(uint16_t(item >> 7))) : -1e38f;
                float max_score = score;
                max_score = metal::max(max_score, simd_shuffle_xor(max_score, 4));
                max_score = metal::max(max_score, simd_shuffle_xor(max_score, 2));
                max_score = metal::max(max_score, simd_shuffle_xor(max_score, 1));

                float exp_score = (lane >= 24u) ? metal::precise::exp(score - max_score) : 0.0f;
                float sum_exp = exp_score;
                sum_exp += simd_shuffle_xor(sum_exp, 4);
                sum_exp += simd_shuffle_xor(sum_exp, 2);
                sum_exp += simd_shuffle_xor(sum_exp, 1);

                if (lane >= 24u) {
                    const uint selected = item & 127u;
                    const uint out_idx = row * 8u + (lane - 24u);
                    indices[out_idx] = selected;
                    sel[out_idx] = selected;

                    float weight = (sum_exp > 0.0f) ? (exp_score / sum_exp) : 0.0f;
                    float scale = float(pes[selected]);
                    weights[out_idx] = bfloat16_t(weight * scale);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 2 -- the incumbent simd-rank scatter, verbatim, over the
            // staged 64 keys. Threads 0..63 are exactly the two complete
            // SIMD groups the standalone kernel launched; `assignment` and
            // `lane` reproduce its coordinates.
            if (tid < 64u) {
                const uint assignment = tid;
                const uint key = sel[assignment];
                const uint key_low = sel[lane];
                const uint key_high = sel[32u + lane];
                uint rank = 0;
                #pragma clang loop unroll(full)
                for (uint source = 0; source < 32; ++source) {
                    const uint other_low = simd_broadcast(key_low, ushort(source));
                    rank += (other_low < key)
                        || (other_low == key && source < assignment);
                    const uint other_high = simd_broadcast(key_high, ushort(source));
                    const uint high_assignment = 32u + source;
                    rank += (other_high < key)
                        || (other_high == key && high_assignment < assignment);
                }
                row_order[rank] = assignment / 8;
                sorted_keys[rank] = key;
                inverse_order[assignment] = rank;
            }
        """,
        header: """
            inline bool gemma4_finalists_before(uint a, uint b) {
                const bfloat16_t av = uint16_to_bfloat16(uint16_t(a >> 7));
                const bfloat16_t bv = uint16_to_bfloat16(uint16_t(b >> 7));
                const bool an = metal::isnan(av);
                const bool bn = metal::isnan(bv);
                bool ab;
                bool ba;
                if (an | bn) {
                    ab = (!an) & bn;
                    ba = (!bn) & an;
                } else {
                    ab = av < bv;
                    ba = bv < av;
                }
                return ab || (!ba && (a & 127u) < (b & 127u));
            }
        """,
        ensureRowContiguous: true
    )

    static func apply(
        _ scores: MLXArray, perExpertScale: MLXArray, topK: Int, kth: Int
    ) -> Fold? {
        guard enabled, Gemma4RouterFinalistsV1.enabled,
            topK == 8, kth == 120,
            scores.ndim == 3, scores.dim(0) == 8,
            scores.dim(1) == 1, scores.dim(2) == 128,
            scores.dtype == .bfloat16,
            perExpertScale.ndim == 1, perExpertScale.dim(0) == 128,
            perExpertScale.dtype == .bfloat16
        else { return nil }
        CBv2EngageMark.once("glue-fold")
        let outs = kernel(
            [scores, perExpertScale],
            grid: (1024, 1, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[8, 1, 8], [8, 1, 8], [64], [64], [64]],
            outputDTypes: [.uint32, .bfloat16, .uint32, .uint32, .uint32]
        )
        return Fold(
            indices: outs[0],
            weights: outs[1],
            table: SwitchRouteTable(
                rowOrder: outs[2],
                sortedKeys: outs[3],
                inverseOrder: outs[4]))
    }
}

public enum Gemma4WideRouteFold {
    public typealias Route = (indices: MLXArray, weights: MLXArray)

    public static func apply(
        scores: MLXArray, perExpertScale: MLXArray, topK: Int = 8, kth: Int = 120
    ) -> Route? {
        guard scores.ndim == 3, scores.dim(1) == 1,
            scores.dim(0) > 8, scores.dim(0) % 8 == 0
        else { return nil }
        var indices: [MLXArray] = []
        var weights: [MLXArray] = []
        for start in stride(from: 0, to: scores.dim(0), by: 8) {
            guard
                let tile = applyEightRows(
                    scores: scores[start ..< (start + 8)], perExpertScale: perExpertScale,
                    topK: topK, kth: kth)
            else { return nil }
            indices.append(tile.indices)
            weights.append(tile.weights)
        }
        return (concatenated(indices, axis: 0), concatenated(weights, axis: 0))
    }

    public static func applyEightRows(
        scores: MLXArray, perExpertScale: MLXArray, topK: Int = 8, kth: Int = 120
    ) -> Route? {
        guard
            let fold = Gemma4RouteGlueFoldV1.apply(
                scores, perExpertScale: perExpertScale, topK: topK, kth: kth)
        else { return nil }
        return (fold.indices, fold.weights)
    }
}

public final class Gemma4GlueChainBox {
    var pending: (source: MLXArray, normed: MLXArray, qkvRunsumTable: MLXArray?)?
    public init() {}
}

private enum Gemma4FusedLayerGlue {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_FUSED_LAYER_GLUE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let pairedRmsEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_DECODE_PAIRED_RMS"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let qkvRunsumCarryEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_QKV_XSUM_CARRY"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let rows = 8
    private static let axis = 2816
    private static let eps: Float = 1e-6
    private static let nReads = 4
    private static let tgThreads = 704  // 2816 / 4, exactly rms_single_row's shape

    private static func rmsReduce(_ src: String, into slot: String) -> String {
        """
            {
                float acc = 0;
                for (int i = 0; i < 4; i++) {
                    float xi = (float)\(src)[base + i];
                    acc += xi * xi;
                }
                acc = simd_sum(acc);
                if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    acc = simd_sum(
                        simd_lane_id < 22 ? local_sums[simd_lane_id] : 0.0f);
                    if (simd_lane_id == 0) {
                        \(slot) = metal::precise::rsqrt(acc / 2816.0f + 1e-06f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        """
    }

    private static let pairedRmsSource = """
        float av[4];
        float bv[4];
        threadgroup float local_sums_b[32];
        {
            float acc_a = 0;
            float acc_b = 0;
            for (int i = 0; i < 4; i++) {
                av[i] = (float)a[base + i];
                bv[i] = (float)b[base + i];
                acc_a += av[i] * av[i];
                acc_b += bv[i] * bv[i];
            }
            acc_a = simd_sum(acc_a);
            acc_b = simd_sum(acc_b);
            if (simd_lane_id == 0) {
                local_sums[simd_group_id] = acc_a;
                local_sums_b[simd_group_id] = acc_b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc_a = simd_sum(
                    simd_lane_id < 22 ? local_sums[simd_lane_id] : 0.0f);
                acc_b = simd_sum(
                    simd_lane_id < 22 ? local_sums_b[simd_lane_id] : 0.0f);
                if (simd_lane_id == 0) {
                    local_inv[0] = metal::precise::rsqrt(acc_a / 2816.0f + 1e-06f);
                    local_inv[1] = metal::precise::rsqrt(acc_b / 2816.0f + 1e-06f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """

    private static func pairedRmsTailSource(_ source: String) -> String {
        var result = source
        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }
        replaceOnce(rmsReduce("a", into: "local_inv[0]"), with: pairedRmsSource)
        replaceOnce(rmsReduce("b", into: "local_inv[1]"), with: "")
        replaceOnce(
            "const T h1 = w1[wbase + i] * static_cast<T>((float)a[base + i] * inv1);",
            with: "const T h1 = w1[wbase + i] * static_cast<T>(av[i] * inv1);")
        replaceOnce(
            "const T h2 = w2[wbase + i] * static_cast<T>((float)b[base + i] * inv2);",
            with: "const T h2 = w2[wbase + i] * static_cast<T>(bv[i] * inv2);")
        return result
    }

    private static let normResidualKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_norm_residual_2816_bf16_v1_nb1",
        inputNames: ["x", "res", "w"],
        outputNames: ["out"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[1];
            threadgroup float local_sums[32];
            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("x", into: "local_inv[0]"))
            const float inv = local_inv[0];
            for (int i = 0; i < 4; i++) {
                // The stock chain rounds the norm's output to T in memory
                // before the residual add reads it; reproduce both roundings.
                const T normed = static_cast<T>(
                    w[wbase + i] * static_cast<T>((float)x[base + i] * inv));
                out[base + i] = res[base + i] + normed;
            }
        """,
        ensureRowContiguous: true
    )

    private static let attentionBranchPrefixKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "gemma4_glue_attention_branch_prefix_2816_bf16_v1_nb1",
            inputNames: ["attn", "res", "wa", "wd", "we", "wr"],
            outputNames: ["out", "dense", "expert", "router", "xSums"],
            source: """
                const uint row = threadgroup_position_in_grid.x;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                threadgroup float local_inv[1];
                threadgroup float local_sums[32];
                const uint base = row * 2816 + lid * 4;
                const uint wbase = lid * 4;
            \(rmsReduce("attn", into: "local_inv[0]"))
                const float attn_inv = local_inv[0];
                T outv[4];
                for (int i = 0; i < 4; i++) {
                    const T normed = static_cast<T>(
                        wa[wbase + i]
                            * static_cast<T>(
                                (float)attn[base + i] * attn_inv));
                    outv[i] = res[base + i] + normed;
                    out[base + i] = outv[i];
                }
            \(rmsReduce("outv", into: "local_inv[0]").replacingOccurrences(
                of: "(float)outv[base + i]", with: "(float)outv[i]"))
                const float branch_inv = local_inv[0];
                float xsum = 0.0f;
                for (int i = 0; i < 4; i++) {
                    const T nx =
                        static_cast<T>((float)outv[i] * branch_inv);
                    const T densev = wd[wbase + i] * nx;
                    dense[base + i] = densev;
                    expert[base + i] = we[wbase + i] * nx;
                    router[base + i] = wr[wbase + i] * nx;
                    xsum += densev;
                }
                xSums[lid * 8 + row] = xsum;
            """,
            ensureRowContiguous: true
        )

    private static let dualPreNormKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_dual_prenorm_xsum_2816_bf16_v2_nb1",
        inputNames: ["x", "w1", "w2"],
        outputNames: ["out1", "out2", "xSums"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[1];
            threadgroup float local_sums[32];
            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("x", into: "local_inv[0]"))
            const float inv = local_inv[0];
            float xsum = 0.0f;
            for (int i = 0; i < 4; i++) {
                const T nx = static_cast<T>((float)x[base + i] * inv);
                const T dense = w1[wbase + i] * nx;
                out1[base + i] = dense;
                out2[base + i] = w2[wbase + i] * nx;
                xsum += dense;
            }
            // `lid == k_block * 32 + lane`, exactly the standalone DMLP
            // xsum table's first two coordinates. Row remains unit stride.
            xSums[lid * 8 + row] = xsum;
        """,
        ensureRowContiguous: true
    )

    private static let tailSource = """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[2];
            threadgroup float local_sums[32];
            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("a", into: "local_inv[0]"))
        \(rmsReduce("b", into: "local_inv[1]"))
            const float inv1 = local_inv[0];
            const float inv2 = local_inv[1];
            T sv[4];
            for (int i = 0; i < 4; i++) {
                const T h1 = w1[wbase + i] * static_cast<T>((float)a[base + i] * inv1);
                const T h2 = w2[wbase + i] * static_cast<T>((float)b[base + i] * inv2);
                sv[i] = h1 + h2;
            }
        \(rmsReduce("sv", into: "local_inv[0]").replacingOccurrences(
            of: "(float)sv[base + i]", with: "(float)sv[i]"))
            const float inv3 = local_inv[0];
            const T scalar = s[0];
            for (int i = 0; i < 4; i++) {
                // Same double rounding as the stock norm-then-add pair, then
                // the layer-scalar multiply with its own stock rounding: the
                // residual sum rounds to T in a register exactly where the
                // stock graph stored it to memory, and the T*T product rounds
                // once on the store exactly like the stock multiply kernel.
                const T normed = static_cast<T>(
                    w3[wbase + i] * static_cast<T>((float)sv[i] * inv3));
                const T summed = res[base + i] + normed;
                out[base + i] = summed * scalar;
            }
        """

    private static let tailKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_tail_2816_bf16_v2_nb1",
        inputNames: ["a", "b", "res", "w1", "w2", "w3", "s"],
        outputNames: ["out"],
        source: tailSource,
        ensureRowContiguous: true
    )

    private static let pairedTailKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_tail_paired_rms_2816_bf16_v1_nb1",
        inputNames: ["a", "b", "res", "w1", "w2", "w3", "s"],
        outputNames: ["out"],
        source: pairedRmsTailSource(tailSource),
        ensureRowContiguous: true
    )

    private static func glueRows(_ x: MLXArray) -> Int? {
        guard x.ndim == 3, x.dim(1) == 1, x.dim(2) == axis else { return nil }
        if x.dim(0) == rows { return rows }
        if CBv2MTPWideVerifyContext.active, x.dim(0) % rows == 0, x.dim(0) > 0 {
            return x.dim(0)
        }
        return nil
    }

    private static func admits(_ x: MLXArray, weight: MLXArray, eps: Float) -> Bool {
        enabled
            && eps == Self.eps
            && glueRows(x) != nil
            && x.dtype == .bfloat16
            && weight.ndim == 1 && weight.dim(0) == axis
            && weight.dtype == .bfloat16
    }

    static func normResidual(
        x: MLXArray, residual: MLXArray, weight: MLXArray, eps: Float
    ) -> MLXArray? {
        guard admits(x, weight: weight, eps: eps),
            residual.shape == x.shape, residual.dtype == .bfloat16
        else { return nil }
        CBv2EngageMark.once("glue-norm-residual")
        let rows = glueRows(x)!
        return normResidualKernel(
            [x, residual, weight],
            template: [("T", x.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[rows, 1, axis]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    struct AttentionBranchPrefix {
        let out: MLXArray
        let denseNorm: MLXArray
        let expertNorm: MLXArray
        let routerNorm: MLXArray
        let denseSums: CBv2DenseMLPQMVV1.ActivationSums
    }

    static func attentionBranchPrefix(
        attn: MLXArray,
        residual: MLXArray,
        postAttentionWeight: MLXArray,
        denseWeight: MLXArray,
        expertWeight: MLXArray,
        routerWeight: MLXArray,
        eps: Float
    ) -> AttentionBranchPrefix? {
        guard CBv2DenseMLPQMVV1.enabled,
            CBv2DenseMLPQMVV1.activationSumsEnabled,
            admits(attn, weight: postAttentionWeight, eps: eps),
            residual.shape == attn.shape, residual.dtype == .bfloat16,
            denseWeight.ndim == 1, denseWeight.dim(0) == axis,
            denseWeight.dtype == .bfloat16,
            expertWeight.ndim == 1, expertWeight.dim(0) == axis,
            expertWeight.dtype == .bfloat16,
            routerWeight.ndim == 1, routerWeight.dim(0) == axis,
            routerWeight.dtype == .bfloat16
        else { return nil }
        let rows = glueRows(attn)!
        let outs = attentionBranchPrefixKernel(
            [
                attn, residual, postAttentionWeight, denseWeight,
                expertWeight, routerWeight,
            ],
            template: [("T", attn.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [
                [rows, 1, axis],
                [rows, 1, axis],
                [rows, 1, axis],
                [rows, 1, axis],
                [(axis / 128) * 32 * rows],
            ],
            outputDTypes: [
                .bfloat16, .bfloat16, .bfloat16, .bfloat16, .float32,
            ]
        )
        guard let denseSums = CBv2DenseMLPQMVV1.activationSums(
            produced: outs[4], for: outs[1])
        else { return nil }
        CBv2EngageMark.once("attention-branch-prefix")
        return AttentionBranchPrefix(
            out: outs[0],
            denseNorm: outs[1],
            expertNorm: outs[2],
            routerNorm: outs[3],
            denseSums: denseSums)
    }

    static func dualPreNorm(
        x: MLXArray, w1: MLXArray, w2: MLXArray, eps: Float
    ) -> (MLXArray, MLXArray, CBv2DenseMLPQMVV1.ActivationSums?)? {
        guard admits(x, weight: w1, eps: eps),
            w2.ndim == 1, w2.dim(0) == axis, w2.dtype == .bfloat16
        else { return nil }
        CBv2EngageMark.once("glue-dual-prenorm")
        let rows = glueRows(x)!
        let outs = dualPreNormKernel(
            [x, w1, w2],
            template: [("T", x.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [
                [rows, 1, axis],
                [rows, 1, axis],
                [(axis / 128) * 32 * rows],
            ],
            outputDTypes: [.bfloat16, .bfloat16, .float32]
        )
        let sums = CBv2DenseMLPQMVV1.activationSums(
            produced: outs[2], for: outs[0])
        return (outs[0], outs[1], sums)
    }

    private static let tailChainSource = """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[2];
            threadgroup float local_sums[32];
            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("a", into: "local_inv[0]"))
        \(rmsReduce("b", into: "local_inv[1]"))
            const float inv1 = local_inv[0];
            const float inv2 = local_inv[1];
            T sv[4];
            for (int i = 0; i < 4; i++) {
                const T h1 = w1[wbase + i] * static_cast<T>((float)a[base + i] * inv1);
                const T h2 = w2[wbase + i] * static_cast<T>((float)b[base + i] * inv2);
                sv[i] = h1 + h2;
            }
        \(rmsReduce("sv", into: "local_inv[0]").replacingOccurrences(
            of: "(float)sv[base + i]", with: "(float)sv[i]"))
            const float inv3 = local_inv[0];
            const T scalar = s[0];
            T outv[4];
            for (int i = 0; i < 4; i++) {
                const T normed3 = static_cast<T>(
                    w3[wbase + i] * static_cast<T>((float)sv[i] * inv3));
                const T summed = res[base + i] + normed3;
                outv[i] = summed * scalar;
                out[base + i] = outv[i];
            }
        \(rmsReduce("outv", into: "local_inv[0]").replacingOccurrences(
            of: "(float)outv[base + i]", with: "(float)outv[i]"))
            const float inv4 = local_inv[0];
            for (int i = 0; i < 4; i++) {
                normed[base + i] =
                    wn[wbase + i] * static_cast<T>((float)outv[i] * inv4);
            }
        """

    private static let tailChainKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_tail_chain_2816_bf16_v1_nb1",
        inputNames: ["a", "b", "res", "w1", "w2", "w3", "s", "wn"],
        outputNames: ["out", "normed"],
        source: tailChainSource,
        ensureRowContiguous: true
    )

    private static let pairedTailChainKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_tail_chain_paired_rms_2816_bf16_v1_nb1",
        inputNames: ["a", "b", "res", "w1", "w2", "w3", "s", "wn"],
        outputNames: ["out", "normed"],
        source: pairedRmsTailSource(tailChainSource),
        ensureRowContiguous: true
    )

    private static let deferredExpertValuesSource = """
            T expertv[4];
            const uint assignment_base = row * 8u;
            uint sorted_rows[8];
            T routed_weights[8];
            for (uint slot = 0u; slot < 8u; ++slot) {
                const uint assignment = assignment_base + slot;
                sorted_rows[slot] = (uint)inverse[assignment];
                routed_weights[slot] = route_weights[assignment];
            }
            for (int i = 0; i < 4; ++i) {
                T accumulator = static_cast<T>(0.0f);
                for (uint slot = 0u; slot < 8u; ++slot) {
                    const T weighted = static_cast<T>(
                        (float)sorted[sorted_rows[slot] * 2816u + wbase + (uint)i]
                        * (float)routed_weights[slot]);
                    accumulator = accumulator + weighted;
                }
                expertv[i] = accumulator;
            }
    """

    private static let deferredTailKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_deferred_expert_tail_2816_bf16_v1_nb1_vec1",
        inputNames: [
            "a", "sorted", "inverse", "route_weights", "res",
            "w1", "w2", "w3", "s",
        ],
        outputNames: ["out"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[2];
            threadgroup float local_sums[32];
            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("a", into: "local_inv[0]"))
        \(deferredExpertValuesSource)
        \(rmsReduce("expertv", into: "local_inv[1]").replacingOccurrences(
            of: "(float)expertv[base + i]", with: "(float)expertv[i]"))
            const float inv1 = local_inv[0];
            const float inv2 = local_inv[1];
            T sv[4];
            for (int i = 0; i < 4; i++) {
                const T h1 = w1[wbase + i]
                    * static_cast<T>((float)a[base + i] * inv1);
                const T h2 = w2[wbase + i]
                    * static_cast<T>((float)expertv[i] * inv2);
                sv[i] = h1 + h2;
            }
        \(rmsReduce("sv", into: "local_inv[0]").replacingOccurrences(
            of: "(float)sv[base + i]", with: "(float)sv[i]"))
            const float inv3 = local_inv[0];
            const T scalar = s[0];
            for (int i = 0; i < 4; i++) {
                const T normed3 = static_cast<T>(
                    w3[wbase + i] * static_cast<T>((float)sv[i] * inv3));
                const T summed = res[base + i] + normed3;
                out[base + i] = summed * scalar;
            }
        """,
        ensureRowContiguous: true
    )

    private static let deferredTailChainSource = """
                const uint row = threadgroup_position_in_grid.x;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                threadgroup float local_inv[2];
                threadgroup float local_sums[32];
                const uint base = row * 2816 + lid * 4;
                const uint wbase = lid * 4;
            \(rmsReduce("a", into: "local_inv[0]"))
            \(deferredExpertValuesSource)
            \(rmsReduce("expertv", into: "local_inv[1]").replacingOccurrences(
                of: "(float)expertv[base + i]", with: "(float)expertv[i]"))
                const float inv1 = local_inv[0];
                const float inv2 = local_inv[1];
                T sv[4];
                for (int i = 0; i < 4; i++) {
                    const T h1 = w1[wbase + i]
                        * static_cast<T>((float)a[base + i] * inv1);
                    const T h2 = w2[wbase + i]
                        * static_cast<T>((float)expertv[i] * inv2);
                    sv[i] = h1 + h2;
                }
            \(rmsReduce("sv", into: "local_inv[0]").replacingOccurrences(
                of: "(float)sv[base + i]", with: "(float)sv[i]"))
                const float inv3 = local_inv[0];
                const T scalar = s[0];
                T outv[4];
                for (int i = 0; i < 4; i++) {
                    const T normed3 = static_cast<T>(
                        w3[wbase + i]
                            * static_cast<T>((float)sv[i] * inv3));
                    const T summed = res[base + i] + normed3;
                    outv[i] = summed * scalar;
                    out[base + i] = outv[i];
                }
            \(rmsReduce("outv", into: "local_inv[0]").replacingOccurrences(
                of: "(float)outv[base + i]", with: "(float)outv[i]"))
                const float inv4 = local_inv[0];
                for (int i = 0; i < 4; i++) {
                    normed[base + i] =
                        wn[wbase + i]
                            * static_cast<T>((float)outv[i] * inv4);
                }
            """

    private static let deferredTailChainKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "gemma4_glue_deferred_expert_tail_chain_2816_bf16_v1_nb1_vec1",
            inputNames: [
                "a", "sorted", "inverse", "route_weights", "res",
                "w1", "w2", "w3", "s", "wn",
            ],
            outputNames: ["out", "normed"],
            source: deferredTailChainSource,
            ensureRowContiguous: true
        )

    private static let deferredTailChainQKVRunsumSource: String = {
        let old = """
                const float inv4 = local_inv[0];
                for (int i = 0; i < 4; i++) {
                    normed[base + i] =
                        wn[wbase + i]
                            * static_cast<T>((float)outv[i] * inv4);
                }
            """
        let new = """
                const float inv4 = local_inv[0];
                T normedv[4];
                for (int i = 0; i < 4; i++) {
                    normedv[i] =
                        wn[wbase + i]
                            * static_cast<T>((float)outv[i] * inv4);
                    normed[base + i] = normedv[i];
                }

                // Exact QKV prepass order. Adjacent tail threads hold the
                // two four-BF16 expressions consumed by one runsum4 lane;
                // masks 2/4/8 then reproduce that prepass lane's 1/2/4
                // butterfly over the eight runsum4 values in this g64.
                float qkv_sum = 0.0f;
                qkv_sum += normedv[0] + normedv[1] + normedv[2] + normedv[3];
                qkv_sum += simd_shuffle_xor(qkv_sum, 1u);
                qkv_sum += simd_shuffle_xor(qkv_sum, 2u);
                qkv_sum += simd_shuffle_xor(qkv_sum, 4u);
                qkv_sum += simd_shuffle_xor(qkv_sum, 8u);
                if ((lid & 15u) == 0u) {
                    qkv_sums[row * 44u + lid / 16u] = qkv_sum;
                }
            """
        precondition(deferredTailChainSource.components(separatedBy: old).count == 2)
        return deferredTailChainSource.replacingOccurrences(of: old, with: new)
    }()

    private static let deferredTailChainQKVRunsumKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "gemma4_glue_deferred_expert_tail_chain_qkv_xsum_2816_bf16_v1",
            inputNames: [
                "a", "sorted", "inverse", "route_weights", "res",
                "w1", "w2", "w3", "s", "wn",
            ],
            outputNames: ["out", "normed", "qkv_sums"],
            source: deferredTailChainQKVRunsumSource,
            ensureRowContiguous: true
        )

    private static func admitsDeferred(
        _ expertRows: DeferredWeightedExpertRows, rows: Int
    ) -> Bool {
        expertRows.sortedOutputs.dtype == .bfloat16
            && expertRows.sortedOutputs.shape == [rows * 8, axis]
            && expertRows.inverseOrder.dtype == .uint32
            && expertRows.inverseOrder.ndim == 1
            && expertRows.inverseOrder.size == rows * 8
            && expertRows.weights.dtype == .bfloat16
            && expertRows.weights.shape == [rows, 8]
    }

    static func tailChainedDeferred(
        mlpOut: MLXArray,
        expertRows: DeferredWeightedExpertRows,
        residual: MLXArray,
        w1: MLXArray,
        w2: MLXArray,
        w3: MLXArray,
        layerScalar: MLXArray,
        nextInputNormWeight: MLXArray,
        eps: Float,
        carryQKVRunsums: Bool? = nil
    ) -> (out: MLXArray, normedNext: MLXArray, qkvRunsumTable: MLXArray?)? {
        guard admits(mlpOut, weight: w1, eps: eps),
            let rows = glueRows(mlpOut),
            admitsDeferred(expertRows, rows: rows),
            residual.shape == mlpOut.shape,
            residual.dtype == .bfloat16,
            w2.ndim == 1, w2.dim(0) == axis, w2.dtype == .bfloat16,
            w3.ndim == 1, w3.dim(0) == axis, w3.dtype == .bfloat16,
            layerScalar.size == 1, layerScalar.dtype == .bfloat16,
            nextInputNormWeight.ndim == 1,
            nextInputNormWeight.dim(0) == axis,
            nextInputNormWeight.dtype == .bfloat16
        else { return nil }
        CBv2EngageMark.once("glue-deferred-expert-tail-chain")
        let carryRunsums = rows == Self.rows
            && CBv2AttentionQKVMMA8V1.rsPrepassEnabled
            && (carryQKVRunsums ?? qkvRunsumCarryEnabled)
        if carryRunsums {
            CBv2EngageMark.once("glue-deferred-tail-qkv-xsum-carry")
        }
        let outs = (carryRunsums
            ? deferredTailChainQKVRunsumKernel : deferredTailChainKernel)(
            [
                mlpOut, expertRows.sortedOutputs, expertRows.inverseOrder,
                expertRows.weights, residual, w1, w2, w3, layerScalar,
                nextInputNormWeight,
            ],
            template: [("T", mlpOut.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: carryRunsums
                ? [[rows, 1, axis], [rows, 1, axis], [rows, axis / 64]]
                : [[rows, 1, axis], [rows, 1, axis]],
            outputDTypes: carryRunsums
                ? [.bfloat16, .bfloat16, .float32]
                : [.bfloat16, .bfloat16]
        )
        return (outs[0], outs[1], carryRunsums ? outs[2] : nil)
    }

    static func tailDeferred(
        mlpOut: MLXArray,
        expertRows: DeferredWeightedExpertRows,
        residual: MLXArray,
        w1: MLXArray,
        w2: MLXArray,
        w3: MLXArray,
        layerScalar: MLXArray,
        eps: Float
    ) -> MLXArray? {
        guard admits(mlpOut, weight: w1, eps: eps),
            let rows = glueRows(mlpOut),
            admitsDeferred(expertRows, rows: rows),
            residual.shape == mlpOut.shape,
            residual.dtype == .bfloat16,
            w2.ndim == 1, w2.dim(0) == axis, w2.dtype == .bfloat16,
            w3.ndim == 1, w3.dim(0) == axis, w3.dtype == .bfloat16,
            layerScalar.size == 1, layerScalar.dtype == .bfloat16
        else { return nil }
        CBv2EngageMark.once("glue-deferred-expert-tail")
        return deferredTailKernel(
            [
                mlpOut, expertRows.sortedOutputs, expertRows.inverseOrder,
                expertRows.weights, residual, w1, w2, w3, layerScalar,
            ],
            template: [("T", mlpOut.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[rows, 1, axis]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    static func tailChained(
        mlpOut: MLXArray, expertOut: MLXArray, residual: MLXArray,
        w1: MLXArray, w2: MLXArray, w3: MLXArray, layerScalar: MLXArray,
        nextInputNormWeight: MLXArray, eps: Float
    ) -> (out: MLXArray, normedNext: MLXArray)? {
        guard admits(mlpOut, weight: w1, eps: eps),
            expertOut.shape == mlpOut.shape, expertOut.dtype == .bfloat16,
            residual.shape == mlpOut.shape, residual.dtype == .bfloat16,
            w2.ndim == 1, w2.dim(0) == axis, w2.dtype == .bfloat16,
            w3.ndim == 1, w3.dim(0) == axis, w3.dtype == .bfloat16,
            layerScalar.size == 1, layerScalar.dtype == .bfloat16,
            nextInputNormWeight.ndim == 1, nextInputNormWeight.dim(0) == axis,
            nextInputNormWeight.dtype == .bfloat16
        else { return nil }
        CBv2EngageMark.once("glue-tail-chain")
        let rows = glueRows(mlpOut)!
        let selected = pairedRmsEnabled ? pairedTailChainKernel : tailChainKernel
        let outs = selected(
            [mlpOut, expertOut, residual, w1, w2, w3, layerScalar,
             nextInputNormWeight],
            template: [("T", mlpOut.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[rows, 1, axis], [rows, 1, axis]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outs[0], outs[1])
    }

    static func tail(
        mlpOut: MLXArray, expertOut: MLXArray, residual: MLXArray,
        w1: MLXArray, w2: MLXArray, w3: MLXArray, layerScalar: MLXArray,
        eps: Float
    ) -> MLXArray? {
        guard admits(mlpOut, weight: w1, eps: eps),
            expertOut.shape == mlpOut.shape, expertOut.dtype == .bfloat16,
            residual.shape == mlpOut.shape, residual.dtype == .bfloat16,
            w2.ndim == 1, w2.dim(0) == axis, w2.dtype == .bfloat16,
            w3.ndim == 1, w3.dim(0) == axis, w3.dtype == .bfloat16,
            layerScalar.size == 1, layerScalar.dtype == .bfloat16
        else { return nil }
        CBv2EngageMark.once("glue-tail")
        let rows = glueRows(mlpOut)!
        let selected = pairedRmsEnabled ? pairedTailKernel : tailKernel
        return selected(
            [mlpOut, expertOut, residual, w1, w2, w3, layerScalar],
            template: [("T", mlpOut.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[rows, 1, axis]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}

enum Gemma4TailQKVRunsumExactnessFixture {
    static func apply(
        mlpOut: MLXArray,
        expertRows: DeferredWeightedExpertRows,
        residual: MLXArray,
        w1: MLXArray,
        w2: MLXArray,
        w3: MLXArray,
        layerScalar: MLXArray,
        nextInputNormWeight: MLXArray,
        carryRunsums: Bool
    ) -> (out: MLXArray, normedNext: MLXArray, qkvRunsumTable: MLXArray?)? {
        Gemma4FusedLayerGlue.tailChainedDeferred(
            mlpOut: mlpOut,
            expertRows: expertRows,
            residual: residual,
            w1: w1,
            w2: w2,
            w3: w3,
            layerScalar: layerScalar,
            nextInputNormWeight: nextInputNormWeight,
            eps: 1e-6,
            carryQKVRunsums: carryRunsums)
    }
}

private class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    let topK: Int
    let eps: Float
    let rootSize: Float
    let kth: Int
    private var cachedEffectiveScale: MLXArray?
    var routeStatsLayer = -1

    fileprivate func noteRoute(_ expertScores: MLXArray, _ topKIndices: MLXArray) {
        Gemma4RouterProbe.recorder?(expertScores, topKIndices)
        if Gemma4RouteStats.enabled {
            Gemma4RouteStats.record(layer: routeStatsLayer, topKIndices)
        }
    }

    init(_ config: Gemma4TextConfiguration) {
        precondition(
            config.numExperts != nil && config.topKExperts != nil,
            "Gemma4Router requires num_experts and top_k_experts in the config"
        )
        let numExperts = config.numExperts ?? 0
        self.topK = config.topKExperts ?? 0
        self.eps = config.rmsNormEps
        self.rootSize = pow(Float(config.hiddenSize), -0.5)
        self.kth = numExperts - self.topK

        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (topKIndices: MLXArray, topKWeights: MLXArray) {
        let effScale: MLXArray
        if let cached = cachedEffectiveScale {
            effScale = cached
        } else {
            let eff = scale * rootSize
            cachedEffectiveScale = eff
            effScale = eff
        }
        let normed = MLXFast.rmsNorm(x, weight: effScale, eps: eps)
        let expertScores = projectScores(normed)

        if let fused = Gemma4FusedRouterTop8.apply(
            expertScores: expertScores, perExpertScale: perExpertScale, topK: topK)
        {
            noteRoute(expertScores, fused.indices)
            return (fused.indices, fused.weights)
        }

        var topKIndices: MLXArray
        if let selected = Gemma4RouterFinalistsV1.applyPrefill(
            expertScores, topK: topK, kth: kth)
        {
            topKIndices = selected
        } else {
            topKIndices = MLX.argPartition(expertScores, kth: kth, axis: -1)
            topKIndices = topKIndices[.ellipsis, kth...]
        }

        var topKWeights = MLX.takeAlong(expertScores, topKIndices, axis: -1)
        topKWeights = MLX.softmax(topKWeights, axis: -1, precise: true)
        topKWeights = topKWeights * perExpertScale[topKIndices]

        noteRoute(expertScores, topKIndices)

        return (topKIndices, topKWeights)
    }

    // MARK: ZIP-ROUTER-001 stages

    fileprivate var zipAdmits: Bool { !Gemma4FusedRouterTop8.enabled }

    fileprivate func zipEffectiveScale() -> MLXArray {
        if let cached = cachedEffectiveScale { return cached }
        let eff = scale * rootSize
        cachedEffectiveScale = eff
        return eff
    }

    fileprivate func zipNorm(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: zipEffectiveScale(), eps: eps)
    }

    fileprivate func zipScores(_ normed: MLXArray) -> MLXArray {
        projectScores(normed)
    }

    fileprivate func projectScores(_ normed: MLXArray) -> MLXArray {
        if normed.ndim == 3, normed.dim(1) == 1, normed.dim(0) > 8,
            let tiled = CBv2MTPWideVerifyContext.rowTiles(normed, tile: 8, { proj($0) })
        {
            return tiled
        }
        return proj(normed)
    }

    fileprivate func zipPartition(_ expertScores: MLXArray) -> MLXArray {
        if let selected = Gemma4RouterFinalistsV1.apply(
            expertScores, topK: topK, kth: kth)
        {
            return selected
        }
        return MLX.argPartition(expertScores, kth: kth, axis: -1)
    }

    fileprivate func zipSelected(_ partition: MLXArray) -> MLXArray {
        if topK == 8, kth == 120, partition.ndim == 3,
            partition.dim(0) % 8 == 0, partition.dim(1) == 1,
            partition.dim(2) == 8, partition.dtype == .uint32
        {
            return partition
        }
        return partition[.ellipsis, kth...]
    }

    fileprivate func zipWeights(
        expertScores: MLXArray, topKIndices: MLXArray
    ) -> MLXArray {
        var topKWeights = MLX.takeAlong(expertScores, topKIndices, axis: -1)
        topKWeights = MLX.softmax(topKWeights, axis: -1, precise: true)
        return topKWeights * perExpertScale[topKIndices]
    }
}

private class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU
    let fuseWeightedUnsort: Bool

    struct Output {
        let output: MLXArray
        let unsortCarrier: WeightedExpertUnsortCarrier?
    }

    init(
        _ config: Gemma4TextConfiguration,
        fuseWeightedUnsort: Bool = false
    ) {
        let numExperts = config.numExperts ?? 1
        let moeIntermediate = config.moeIntermediateSize ?? config.intermediateSize
        self.fuseWeightedUnsort = fuseWeightedUnsort

        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeIntermediate,
            numExperts: numExperts,
            activation: { gemma4SafeGeluApproximate($0) },
            bias: false,
            weightedReductionProfile: .gemma4ProductionGeGLU
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        topKIndices: MLXArray,
        topKWeights: MLXArray,
        isExpertPrefill: Bool,
        sortedPlane: SwitchSortedPlaneProducer? = nil
    ) -> Output {
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = topKIndices.dim(-1)
        let result = switchGLU.callAndWeightedReduceWithUnsortCarrier(
            x.reshaped(B * S, H),
            topKIndices.reshaped(B * S, K),
            weights: topKWeights.reshaped(B * S, K),
            fuseSortedReduction: fuseWeightedUnsort,
            isProductionPrefill: isExpertPrefill,
            sortedPlane: sortedPlane)
        return Output(
            output: result.output.reshaped(B, S, H),
            unsortCarrier: result.carrier)
    }

    func deferredWeightedRows(
        _ x: MLXArray,
        topKIndices: MLXArray,
        topKWeights: MLXArray,
        isExpertPrefill: Bool,
        routeTable: SwitchRouteTable? = nil
    ) -> DeferredWeightedExpertRows? {
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = topKIndices.dim(-1)
        return switchGLU.callAndDeferWeightedReduce(
            x.reshaped(B * S, H),
            topKIndices.reshaped(B * S, K),
            weights: topKWeights.reshaped(B * S, K),
            fuseSortedReduction: fuseWeightedUnsort,
            isProductionPrefill: isExpertPrefill,
            routeTable: routeTable)
    }
}

// MARK: - MLP

private class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let isKvSharedLayer = config.layerUsesSharedKV(layerIdx: layerIdx)
        let useDoubleWide = config.useDoubleWideMlp && isKvSharedLayer
        let intermediateSize = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)

        super.init()
    }

    @inline(__always)
    private func denseProjection(
        _ layer: Linear,
        _ x: MLXArray,
        activationSums: CBv2DenseMLPQMVV1.ActivationSums? = nil
    ) -> MLXArray {
        guard let quantized = layer as? QuantizedLinear,
            quantized.bias == nil,
            let tight = CBv2DenseMLPQMVV1.matmul(
                x: x,
                weight: quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode,
                activationSums: activationSums)
        else { return layer(x) }
        return tight
    }

    func callAsFunction(
        _ x: MLXArray,
        activationSums producerSums: CBv2DenseMLPQMVV1.ActivationSums? = nil
    ) -> MLXArray {
        let activationSums = producerSums ?? CBv2DenseMLPQMVV1.activationSums(for: x)
        return denseProjection(
            downProj,
            gemma4GeluProduct(
                denseProjection(gateProj, x, activationSums: activationSums),
                denseProjection(upProj, x, activationSums: activationSums)))
    }

    // MARK: ZIP-ROUTER-001 stages

    fileprivate func zipActivationSums(
        _ x: MLXArray
    ) -> CBv2DenseMLPQMVV1.ActivationSums? {
        CBv2DenseMLPQMVV1.activationSums(for: x)
    }

    fileprivate func zipGate(
        _ x: MLXArray, _ activationSums: CBv2DenseMLPQMVV1.ActivationSums?
    ) -> MLXArray {
        denseProjection(gateProj, x, activationSums: activationSums)
    }

    fileprivate func zipUp(
        _ x: MLXArray, _ activationSums: CBv2DenseMLPQMVV1.ActivationSums?
    ) -> MLXArray {
        denseProjection(upProj, x, activationSums: activationSums)
    }

    fileprivate func zipDown(_ activated: MLXArray) -> MLXArray {
        denseProjection(downProj, activated)
    }
}

private enum Gemma4ZipRouterV1 {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_ZIP_ROUTER"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let plan: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_ZIP_ROUTER_PLAN"], let v = Int(raw)
        else { return 1 }
        return v
    }()

    struct Zipped {
        let denseOut: MLXArray
        let expertNorm: MLXArray
        let topKIndices: MLXArray
        let topKWeights: MLXArray
        let routeTable: SwitchRouteTable?
    }

    static func makeAttentionBranchPrefix(
        router: Gemma4Router,
        attn: MLXArray,
        residual: MLXArray,
        postAttentionWeight: MLXArray,
        denseWeight: MLXArray,
        expertWeight: MLXArray,
        eps: Float
    ) -> Gemma4FusedLayerGlue.AttentionBranchPrefix? {
        guard enabled, router.zipAdmits else { return nil }
        return Gemma4FusedLayerGlue.attentionBranchPrefix(
            attn: attn,
            residual: residual,
            postAttentionWeight: postAttentionWeight,
            denseWeight: denseWeight,
            expertWeight: expertWeight,
            routerWeight: router.zipEffectiveScale(),
            eps: eps)
    }

    static func run(
        router: Gemma4Router,
        mlp: Gemma4MLP,
        out: MLXArray,
        w1: MLXArray,
        w2: MLXArray,
        eps: Float,
        prefix: Gemma4FusedLayerGlue.AttentionBranchPrefix? = nil
    ) -> Zipped? {
        guard enabled, router.zipAdmits,
            out.ndim == 3, out.dim(1) == 1,
            out.dim(0) == 8
                || (CBv2MTPWideVerifyContext.active && out.dim(0) % 8 == 0 && out.dim(0) > 0),
            out.dtype == .bfloat16
        else { return nil }

        let n1: MLXArray
        let n2: MLXArray
        let producerSums: CBv2DenseMLPQMVV1.ActivationSums?
        let carriedRouterNorm: MLXArray?
        if let prefix, prefix.out === out {
            (n1, n2, producerSums, carriedRouterNorm) = (
                prefix.denseNorm,
                 prefix.expertNorm,
                 prefix.denseSums,
                 prefix.routerNorm)
        } else if let (d1, d2, dSums) = Gemma4FusedLayerGlue.dualPreNorm(
            x: out, w1: w1, w2: w2, eps: eps)
        {
            (n1, n2, producerSums, carriedRouterNorm) = (d1, d2, dSums, nil)
        } else {
            return nil
        }

        guard let sums = producerSums ?? mlp.zipActivationSums(n1) else { return nil }
        let normed = carriedRouterNorm ?? router.zipNorm(out)

        let expertScores = router.zipScores(
            MLX.depends(input: normed, dependencies: [sums.dependencyHandle]))
        let denseIn = MLX.depends(input: n1, dependencies: [normed])
        let gate = mlp.zipGate(denseIn, sums)
        let up = mlp.zipUp(denseIn, sums)

        let held = MLX.depends(inputs: [gate, up], dependencies: [expertScores])
        let activated = gemma4GeluProduct(held[0], held[1])

        let topKIndices: MLXArray
        let topKWeights: MLXArray
        let denseOut: MLXArray
        var routeTable: SwitchRouteTable? = nil
        if plan == 2 {
            let partition = router.zipPartition(
                MLX.depends(input: expertScores, dependencies: [gate, up]))
            topKIndices = router.zipSelected(
                MLX.depends(input: partition, dependencies: [activated]))
            denseOut = mlp.zipDown(
                MLX.depends(input: activated, dependencies: [partition]))
            topKWeights = router.zipWeights(
                expertScores: expertScores, topKIndices: topKIndices)
        } else {
            denseOut = mlp.zipDown(activated)
            let foldScores = MLX.depends(input: expertScores, dependencies: [denseOut])
            if let fold = Gemma4RouteGlueFoldV1.apply(
                foldScores,
                perExpertScale: router.perExpertScale,
                topK: router.topK, kth: router.kth)
            {
                topKIndices = fold.indices
                topKWeights = fold.weights
                routeTable = fold.table
            } else if let wide = Gemma4WideRouteFold.apply(
                scores: foldScores, perExpertScale: router.perExpertScale,
                topK: router.topK, kth: router.kth)
            {
                topKIndices = wide.indices
                topKWeights = wide.weights
            } else {
                let partition = router.zipPartition(foldScores)
                topKIndices = router.zipSelected(partition)
                topKWeights = router.zipWeights(
                    expertScores: expertScores, topKIndices: topKIndices)
            }
        }

        router.noteRoute(expertScores, topKIndices)

        let expertNorm =
            plan >= 1
            ? MLX.depends(input: n2, dependencies: [denseOut]) : n2

        CBv2EngageMark.once("zip-router")
        return Zipped(
            denseOut: denseOut,
            expertNorm: expertNorm,
            topKIndices: topKIndices,
            topKWeights: topKWeights,
            routeTable: routeTable)
    }
}

// MARK: - Decoder Layer

public class Gemma4DecoderLayer: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "self_attn") fileprivate var selfAttn: Gemma4Attention
    @ModuleInfo fileprivate var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    @ModuleInfo(key: "router") fileprivate var router: Gemma4Router?
    @ModuleInfo(key: "experts") fileprivate var experts: Gemma4Experts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm?

    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    let isMoE: Bool

    public init(
        _ config: Gemma4TextConfiguration, layerIdx: Int, forceSharedKV: Bool = false,
        fuseWeightedUnsort: Bool = false
    ) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        self.isMoE = config.enableMoeBlock

        self._selfAttn.wrappedValue = Gemma4Attention(
            config, layerIdx: layerIdx, forceSharedKV: forceSharedKV)
        self._mlp.wrappedValue = Gemma4MLP(config, layerIdx: layerIdx)

        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.enableMoeBlock {
            let router = Gemma4Router(config)
            router.routeStatsLayer = layerIdx
            self._router.wrappedValue = router
            self._experts.wrappedValue = Gemma4Experts(
                config,
                fuseWeightedUnsort: fuseWeightedUnsort)
            self._postFeedforwardLayernorm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        if hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1], dtype: .float16)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        positionOffset: Gemma4.PositionOffset? = nil,
        v2SharedSource: (any CBv2AttendingLayerCache)? = nil,
        outputTailRows: Int? = nil,
        useLastQueryPrefill: Bool = false,
        isExpertPrefill: Bool = false,
        glueChain: Gemma4GlueChainBox? = nil,
        nextInputLayernormWeight: MLXArray? = nil,
        enableAttentionBranchPrefix: Bool = false,
        wideColumns: Int = 1
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4.PositionOffset) {
        let outputStart: Int
        if let outputTailRows {
            precondition(outputTailRows > 0, "Gemma4: output tail must retain at least one row")
            precondition(
                (cache as? (any CBv2AttendingLayerCache)) != nil,
                "Gemma4: output-tail narrowing is only valid for CBv2 attention")
            outputStart = max(0, x.dim(1) - outputTailRows)
        } else {
            outputStart = 0
        }
        if useLastQueryPrefill {
            precondition(
                outputTailRows == 1 && outputStart == x.dim(1) - 1,
                "Gemma4: last-query prefill retains exactly one output row")
        }

        let residual = outputStart > 0 ? x[0..., outputStart..., 0...] : x
        let activePerLayerInput: MLXArray?
        if let perLayerInput, outputStart > 0 {
            activePerLayerInput = perLayerInput[0..., outputStart..., 0...]
        } else {
            activePerLayerInput = perLayerInput
        }

        let h: MLXArray
        let carriedQKVRunsumTable: MLXArray?
        if let chain = glueChain, let pending = chain.pending,
            pending.source === x
        {
            chain.pending = nil
            h = pending.normed
            carriedQKVRunsumTable = pending.qkvRunsumTable
        } else {
            glueChain?.pending = nil
            h = inputLayernorm(x)
            carriedQKVRunsumTable = nil
        }
        let (attnOut, kvPair, attnPositionOffset) = selfAttn(
            h, mask: mask, cache: cache, sharedKV: sharedKV, positionOffset: positionOffset,
            v2SharedSource: v2SharedSource, outputStart: outputStart,
            useLastQueryPrefill: useLastQueryPrefill, wideColumns: wideColumns,
            qkvRunsumTable: carriedQKVRunsumTable)
        let attentionBranchPrefix: Gemma4FusedLayerGlue.AttentionBranchPrefix? = {
            guard enableAttentionBranchPrefix,
                isMoE, let router, let preFeedforwardLayernorm2
            else {
                return nil
            }
            return Gemma4ZipRouterV1.makeAttentionBranchPrefix(
                router: router,
                attn: attnOut,
                residual: residual,
                postAttentionWeight: postAttentionLayernorm.weight,
                denseWeight: preFeedforwardLayernorm.weight,
                expertWeight: preFeedforwardLayernorm2.weight,
                eps: config.rmsNormEps)
        }()
        var out: MLXArray
        if let attentionBranchPrefix {
            out = attentionBranchPrefix.out
        } else if let fusedOut = Gemma4FusedLayerGlue.normResidual(
            x: attnOut, residual: residual,
            weight: postAttentionLayernorm.weight, eps: config.rmsNormEps)
        {
            out = fusedOut
        } else if let fusedOut = Gemma4PrefillGlueV1.normResidual(
            x: attnOut,
            weight: postAttentionLayernorm.weight,
            residual: residual,
            eps: config.rmsNormEps)
        {
            out = fusedOut
        } else {
            let postAttn = postAttentionLayernorm(attnOut)
            out = residual + postAttn
        }

        let residual2 = out
        var tailApplied = false
        var scalarFolded = false

        if isMoE,
            let router,
            let experts,
            let postFeedforwardLayernorm1,
            let preFeedforwardLayernorm2,
            let postFeedforwardLayernorm2
        {
            let h1Raw: MLXArray
            let expertBranch: (
                raw: MLXArray?,
                deferred: DeferredWeightedExpertRows?,
                unsortCarrier: WeightedExpertUnsortCarrier?
            )
            let canFoldScalar =
                perLayerInputGate == nil || activePerLayerInput == nil
            func projectExpertBranch(
                _ input: MLXArray,
                indices: MLXArray,
                weights: MLXArray,
                sortedPlane: SwitchSortedPlaneProducer? = nil,
                routeTable: SwitchRouteTable? = nil
            ) -> (
                raw: MLXArray?,
                deferred: DeferredWeightedExpertRows?,
                unsortCarrier: WeightedExpertUnsortCarrier?
            ) {
                if canFoldScalar,
                    let deferred = experts.deferredWeightedRows(
                        input,
                        topKIndices: indices,
                        topKWeights: weights,
                        isExpertPrefill: isExpertPrefill,
                        routeTable: routeTable)
                {
                    return (nil, deferred, nil)
                }
                let result = experts(
                    input,
                    topKIndices: indices,
                    topKWeights: weights,
                    isExpertPrefill: isExpertPrefill,
                    sortedPlane: sortedPlane)
                return (result.output, nil, result.unsortCarrier)
            }

            if let zipped = Gemma4ZipRouterV1.run(
                router: router,
                mlp: mlp,
                out: out,
                w1: preFeedforwardLayernorm.weight,
                w2: preFeedforwardLayernorm2.weight,
                eps: config.rmsNormEps,
                prefix: attentionBranchPrefix)
            {
                h1Raw = zipped.denseOut
                expertBranch = projectExpertBranch(
                    zipped.expertNorm,
                    indices: zipped.topKIndices,
                    weights: zipped.topKWeights,
                    routeTable: zipped.routeTable)
            } else {
                let (topKIndices, topKWeights) = router(out)

                if let (n1, n2, denseSums) = Gemma4FusedLayerGlue.dualPreNorm(
                    x: out,
                    w1: preFeedforwardLayernorm.weight,
                    w2: preFeedforwardLayernorm2.weight,
                    eps: config.rmsNormEps)
                {
                    h1Raw = mlp(n1, activationSums: denseSums)
                    expertBranch = projectExpertBranch(
                        n2,
                        indices: topKIndices,
                        weights: topKWeights)
                } else if isExpertPrefill,
                    Gemma4PrefillGlueV1.prenormGatherEnabled,
                    let n1 = Gemma4PrefillGlueV1.preNorm(
                        x: out,
                        weight: preFeedforwardLayernorm.weight,
                        eps: config.rmsNormEps),
                    let n2 = Gemma4PrefillGlueV1.preNorm(
                        x: out,
                        weight: preFeedforwardLayernorm2.weight,
                        eps: config.rmsNormEps)
                {
                    let expertNormWeight = preFeedforwardLayernorm2.weight
                    let expertTopK = topKIndices.dim(-1)
                    let normEps = config.rmsNormEps
                    h1Raw = mlp(n1)
                    expertBranch = projectExpertBranch(
                        n2,
                        indices: topKIndices,
                        weights: topKWeights,
                        sortedPlane: { inverseOrder in
                            Gemma4PrefillGlueV1.preNormScatter(
                                x: out,
                                weight: expertNormWeight,
                                inverseOrder: inverseOrder,
                                topK: expertTopK,
                                eps: normEps)
                        })
                } else if let (n1, n2) = Gemma4PrefillGlueV1.dualPreNorm(
                    x: out,
                    w1: preFeedforwardLayernorm.weight,
                    w2: preFeedforwardLayernorm2.weight,
                    eps: config.rmsNormEps)
                {
                    h1Raw = mlp(n1)
                    expertBranch = projectExpertBranch(
                        n2,
                        indices: topKIndices,
                        weights: topKWeights)
                } else {
                    h1Raw = mlp(preFeedforwardLayernorm(out))
                    expertBranch = projectExpertBranch(
                        preFeedforwardLayernorm2(out),
                        indices: topKIndices,
                        weights: topKWeights)
                }
            }
            if canFoldScalar, let deferred = expertBranch.deferred,
                let chain = glueChain,
                let nextWeight = nextInputLayernormWeight,
                let chained = Gemma4FusedLayerGlue.tailChainedDeferred(
                    mlpOut: h1Raw, expertRows: deferred, residual: residual2,
                    w1: postFeedforwardLayernorm1.weight,
                    w2: postFeedforwardLayernorm2.weight,
                    w3: postFeedforwardLayernorm.weight,
                    layerScalar: layerScalar,
                    nextInputNormWeight: nextWeight,
                    eps: config.rmsNormEps)
            {
                out = chained.out
                chain.pending = (
                    source: chained.out, normed: chained.normedNext,
                    qkvRunsumTable: chained.qkvRunsumTable)
                tailApplied = true
                scalarFolded = true
            } else if canFoldScalar, let deferred = expertBranch.deferred,
                let fusedTail = Gemma4FusedLayerGlue.tailDeferred(
                    mlpOut: h1Raw, expertRows: deferred, residual: residual2,
                    w1: postFeedforwardLayernorm1.weight,
                    w2: postFeedforwardLayernorm2.weight,
                    w3: postFeedforwardLayernorm.weight,
                    layerScalar: layerScalar,
                    eps: config.rmsNormEps)
            {
                out = fusedTail
                tailApplied = true
                scalarFolded = true
            } else {
                let h2Raw: MLXArray
                if let raw = expertBranch.raw {
                    h2Raw = raw
                } else if let deferred = expertBranch.deferred {
                    h2Raw = resolveDeferredWeightedExpertRows(deferred)
                } else {
                    preconditionFailure("Gemma4 expert branch produced no output")
                }

                if canFoldScalar, let chain = glueChain,
                    let nextWeight = nextInputLayernormWeight,
                    let chained = Gemma4FusedLayerGlue.tailChained(
                        mlpOut: h1Raw, expertOut: h2Raw, residual: residual2,
                        w1: postFeedforwardLayernorm1.weight,
                        w2: postFeedforwardLayernorm2.weight,
                        w3: postFeedforwardLayernorm.weight,
                        layerScalar: layerScalar,
                        nextInputNormWeight: nextWeight,
                        eps: config.rmsNormEps)
                {
                    out = chained.out
                    chain.pending = (
                        source: chained.out, normed: chained.normedNext,
                        qkvRunsumTable: nil)
                    tailApplied = true
                    scalarFolded = true
                } else if canFoldScalar,
                    let fusedTail = Gemma4FusedLayerGlue.tail(
                        mlpOut: h1Raw, expertOut: h2Raw, residual: residual2,
                        w1: postFeedforwardLayernorm1.weight,
                        w2: postFeedforwardLayernorm2.weight,
                        w3: postFeedforwardLayernorm.weight,
                        layerScalar: layerScalar,
                        eps: config.rmsNormEps)
                {
                    out = fusedTail
                    tailApplied = true
                    scalarFolded = true
                } else if canFoldScalar, let chain = glueChain,
                    let nextWeight = nextInputLayernormWeight,
                    let expert = expertBranch.unsortCarrier,
                    let chained = Gemma4PrefillGlueV1.branchTailChainedUnsort(
                        h1: h1Raw,
                        expert: expert,
                        w1: postFeedforwardLayernorm1.weight,
                        w2: postFeedforwardLayernorm2.weight,
                        w3: postFeedforwardLayernorm.weight,
                        residual2: residual2,
                        layerScalar: layerScalar,
                        nextInputNormWeight: nextWeight,
                        eps: config.rmsNormEps)
                {
                    out = chained.out
                    chain.pending = (
                        source: chained.out, normed: chained.normedNext,
                        qkvRunsumTable: nil)
                    tailApplied = true
                    scalarFolded = true
                } else if canFoldScalar, let chain = glueChain,
                    let nextWeight = nextInputLayernormWeight,
                    let chained = Gemma4PrefillGlueV1.branchTailChained(
                        h1: h1Raw,
                        h2: h2Raw,
                        w1: postFeedforwardLayernorm1.weight,
                        w2: postFeedforwardLayernorm2.weight,
                        w3: postFeedforwardLayernorm.weight,
                        residual2: residual2,
                        layerScalar: layerScalar,
                        nextInputNormWeight: nextWeight,
                        eps: config.rmsNormEps)
                {
                    out = chained.out
                    chain.pending = (
                        source: chained.out, normed: chained.normedNext,
                        qkvRunsumTable: nil)
                    tailApplied = true
                    scalarFolded = true
                } else if let fusedTail = Gemma4PrefillGlueV1.branchTail(
                    h1: h1Raw,
                    h2: h2Raw,
                    w1: postFeedforwardLayernorm1.weight,
                    w2: postFeedforwardLayernorm2.weight,
                    w3: postFeedforwardLayernorm.weight,
                    residual2: residual2,
                    eps: config.rmsNormEps)
                {
                    out = fusedTail
                    tailApplied = true
                } else {
                    let h1 = postFeedforwardLayernorm1(h1Raw)
                    let h2 = postFeedforwardLayernorm2(h2Raw)
                    out = h1 + h2
                }
            }
        } else {
            out = preFeedforwardLayernorm(out)
            out = mlp(out)
        }

        if !tailApplied {
            out = postFeedforwardLayernorm(out)
            out = residual2 + out
        }

        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let perLayerInput = activePerLayerInput
        {
            let residual3 = out
            var g = gemma4GeluProduct(gate(out), perLayerInput)
            g = proj(g)
            if let fusedPLE = Gemma4PrefillGlueV1.normResidual(
                x: g, weight: norm.weight, residual: residual3,
                eps: config.rmsNormEps)
            {
                out = fusedPLE
            } else {
                g = norm(g)
                out = residual3 + g
            }
        }

        if !scalarFolded {
            out = out * layerScalar
        }

        return (out, kvPair, attnPositionOffset)
    }
}

// MARK: - EMB-001: fused scaled input embedding

enum Gemma4FusedScaledEmbedding {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_SCALED_EMBEDDING"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let decodeEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_SCALED_EMBEDDING_DECODE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let groupSize = 64
    private static let bits = 4
    private static let codesPerWord = 8
    private static let wordsPerGroup = 8

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_fused_scaled_embedding_affine4_g64_v1",
        inputNames: ["tokens", "w", "scales", "biases", "embed_scale"],
        outputNames: ["out"],
        source: """
            const uint col = thread_position_in_grid.x;
            const uint row = thread_position_in_grid.y;
            // The launch is exactly one thread per packed word of one token
            // row, so the grid carries the row geometry with no shape buffer.
            const uint words_per_row = threads_per_grid.x;
            const uint groups_per_row = words_per_row >> 3;

            // Stock `weight[x]` gathers through `offset_neg_idx`: a negative
            // id wraps by the axis size. Positive out-of-range ids are
            // undefined in the stock gather too and are not redefined here.
            const int raw_token = tokens[row];
            const int vocab = w_shape[0];
            const size_t t = size_t(raw_token < 0 ? raw_token + vocab : raw_token);

            const uint packed = w[t * size_t(words_per_row) + size_t(col)];
            const size_t gindex = t * size_t(groups_per_row) + size_t(col >> 3);

            T scale = scales[gindex];
            T bias = biases[gindex];
            T es = embed_scale;

            device T* o = out
                + (size_t(row) * size_t(words_per_row) + size_t(col)) * 8;

            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; i++) {
                uint8_t d = (packed >> (4 * i)) & 0x0f;
                // Boundary 1 — identical to `affine_dequantize`'s store.
                const T dequantized = scale * d + bias;
                // Boundary 2 — identical to the stock `* embedScale` multiply.
                o[i] = dequantized * es;
            }
            """,
        ensureRowContiguous: true
    )

    static func apply(
        tokens: MLXArray, embedding: Embedding, embedScale: Float, hiddenSize: Int
    ) -> MLXArray? {
        guard enabled,
            tokens.ndim == 2,
            tokens.dtype == .int32,
            tokens.dim(1) > 1 || decodeEnabled,
            let quantized = embedding as? QuantizedEmbedding,
            quantized.mode == .affine,
            quantized.bits == bits,
            quantized.groupSize == groupSize,
            let biases = quantized.biases
        else { return nil }

        let weight = quantized.weight
        let scales = quantized.scales
        guard weight.dtype == .uint32,
            weight.ndim == 2,
            scales.dtype == .bfloat16,
            biases.dtype == .bfloat16,
            scales.ndim == 2,
            biases.shape == scales.shape,
            scales.dim(0) == weight.dim(0),
            weight.dim(1) == hiddenSize / codesPerWord,
            weight.dim(1) % wordsPerGroup == 0,
            scales.dim(1) == hiddenSize / groupSize,
            hiddenSize % groupSize == 0
        else { return nil }

        let batch = tokens.dim(0)
        let length = tokens.dim(1)
        let wordsPerRow = weight.dim(1)

        CBv2EngageMark.once(length > 1 ? "scaled-embedding" : "scaled-embedding-decode")
        return kernel(
            [tokens, weight, scales, biases, embedScale.asMLXArray(dtype: .bfloat16)],
            template: [("T", DType.bfloat16)],
            grid: (wordsPerRow, batch * length, 1),
            threadGroup: (32, 8, 1),
            outputShapes: [[batch, length, hiddenSize]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}

// MARK: - Text Model

private enum Gemma4FinalNormMMAHeadSumsV1 {
    private static let rows = 8
    private static let axis = 2816
    private static let groupSize = 64
    private static let valuesPerThread = 4
    private static let threadgroupSize = axis / valuesPerThread
    private static let eps: Float = 1e-6

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_final_rmsnorm_mma_xsum_2816_bf16_v1",
        inputNames: ["x", "w"],
        outputNames: ["out", "xSums"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[1];
            threadgroup float local_sums[32];
            threadgroup float quad_sums[704];

            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;

            // Exact `rms_single_row<T, 4>` reduction for axis 2816.
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                const float xi = x[base + i];
                acc += xi * xi;
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) {
                local_sums[simd_lane_id] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) {
                local_sums[simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_inv[0] =
                        metal::precise::rsqrt(acc / 2816.0f + 1e-06f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            T outv[4];
            for (int i = 0; i < 4; ++i) {
                // Preserve the stock RMSNorm's BF16 boundary exactly.
                outv[i] = w[wbase + i]
                    * static_cast<T>((float)x[base + i] * local_inv[0]);
                out[base + i] = outv[i];
            }

            // This four-value expression is exactly one addend of the head's
            // stock xsum loop, evaluated at activation dtype then widened.
            quad_sums[lid] = outv[0] + outv[1] + outv[2] + outv[3];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // One leader serially reproduces the head prepass's sixteen
            // ascending `s += four BF16 values` statements for this 64-wide
            // group. No SIMD reassociation is introduced.
            if ((lid % 16) == 0) {
                float s = 0.0f;
                for (uint c = 0; c < 8; ++c) {
                    const uint q = lid + c * 2;
                    s += quad_sums[q];
                    s += quad_sums[q + 1];
                }
                xSums[row * 44 + lid / 16] = s;
            }
            """,
        ensureRowContiguous: true
    )

    static func apply(
        _ x: MLXArray, weight: MLXArray, eps: Float
    ) -> (postNorm: MLXArray, sums: Gemma4MMAQuantizedGEMV.ActivationSums)? {
        guard Gemma4MMAQuantizedGEMV.consumesActivationSums,
            eps == Self.eps,
            x.dtype == .bfloat16,
            x.ndim == 3,
            x.dim(0) == rows,
            x.dim(1) == 1,
            x.dim(2) == axis,
            x.size == x.dim(0) * axis,
            weight.dtype == .bfloat16,
            weight.ndim == 1,
            weight.dim(0) == axis
        else { return nil }
        let rows = x.dim(0)

        let outputs = kernel(
            [x, weight],
            template: [("T", x.dtype)],
            grid: (rows * threadgroupSize, 1, 1),
            threadGroup: (threadgroupSize, 1, 1),
            outputShapes: [[rows, 1, axis], [rows * (axis / groupSize)]],
            outputDTypes: [.bfloat16, .float32]
        )
        guard let sums = Gemma4MMAQuantizedGEMV.activationSums(
            produced: outputs[1], for: outputs[0])
        else { return nil }
        CBv2EngageMark.once("final-norm-mma-xsum")
        return (outputs[0], sums)
    }
}

public class Gemma4TextModelInner: Module {
    let config: Gemma4TextConfiguration
    let embedScale: Float
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [Gemma4DecoderLayer]
    @ModuleInfo public var norm: RMSNorm

    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") fileprivate var perLayerModelProjection: ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    let previousKvs: [Int]
    let firstKvSharedLayerIdx: Int

    let lastFullAttentionNonSharedIdx: Int
    let lastSlidingAttentionNonSharedIdx: Int

    public init(
        _ config: Gemma4TextConfiguration, forceSharedKV: Bool = false,
        fuseWeightedUnsort: Bool = false
    ) {
        self.config = config
        self.embedScale = Float(config.hiddenSize).squareRoot()
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4DecoderLayer(
                config, layerIdx: $0, forceSharedKV: forceSharedKV,
                fuseWeightedUnsort: fuseWeightedUnsort)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * config.hiddenSizePerLayerInput)
            self._perLayerModelProjection.wrappedValue = ScaledLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.numHiddenLayers * config.hiddenSizePerLayerInput,
                scalar: pow(Float(config.hiddenSize), -0.5))
            self._perLayerProjectionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        self.firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        var kvMap = Array(0 ..< config.numHiddenLayers)
        if config.numKvSharedLayers > 0 {
            var lastByType = [String: Int]()
            for i in 0 ..< firstKvSharedLayerIdx {
                lastByType[config.layerTypes[i]] = i
            }
            for j in firstKvSharedLayerIdx ..< config.numHiddenLayers {
                if let prev = lastByType[config.layerTypes[j]] {
                    kvMap[j] = prev
                }
            }
        }
        self.previousKvs = kvMap

        let firstShared = self.firstKvSharedLayerIdx
        var lastFull = -1
        var lastSliding = -1
        for i in 0 ..< firstShared {
            if config.layerTypes[i] == "full_attention" { lastFull = i }
            if config.layerTypes[i] == "sliding_attention" { lastSliding = i }
        }
        self.lastFullAttentionNonSharedIdx = lastFull
        self.lastSlidingAttentionNonSharedIdx = lastSliding

        super.init()
    }
    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        captureHook: ((Int, (MLXArray, MLXArray)) -> Void)? = nil,
        inputEmbedding: MLXArray? = nil,
        imageTokenMask: MLXArray? = nil
    ) -> MLXArray {
        let rankOneInputs = inputs.ndim == 1
        let inputs = rankOneInputs ? inputs.expandedDimensions(axis: 0) : inputs
        let inputEmbedding =
            rankOneInputs && inputEmbedding?.ndim == 2
            ? inputEmbedding?.expandedDimensions(axis: 0) : inputEmbedding
        let imageTokenMask =
            rankOneInputs && imageTokenMask?.ndim == 1
            ? imageTokenMask?.expandedDimensions(axis: 0) : imageTokenMask
        return forwardTrunk(
            inputs, cache: cache, captureHook: captureHook, capturePreNorm: false,
            inputEmbedding: inputEmbedding, imageTokenMask: imageTokenMask
        ).postNorm
    }

    fileprivate func callWithMMAHeadSums(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil
    ) -> (
        postNorm: MLXArray,
        activationSums: Gemma4MMAQuantizedGEMV.ActivationSums?
    ) {
        let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        let result = forwardTrunk(
            inputs, cache: cache, captureHook: nil, capturePreNorm: false,
            emitMMAHeadSums: true)
        return (result.postNorm, result.mmaHeadSums)
    }

    fileprivate func cbv2Prefill(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        inputEmbedding: MLXArray?
    ) -> MLXArray {
        forwardTrunk(
            inputs, cache: cache, captureHook: nil, capturePreNorm: false,
            inputEmbedding: inputEmbedding, schedulePrefill: true
        ).postNorm
    }

    public func callCapturingPreNorm(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        captureHook: ((Int, (MLXArray, MLXArray)) -> Void)? = nil
    ) -> (postNorm: MLXArray, preNorm: MLXArray) {
        let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        let r = forwardTrunk(
            inputs, cache: cache, captureHook: captureHook, capturePreNorm: true)
        return (r.postNorm, r.preNorm!)
    }

    fileprivate func callCapturingPreNormWithMMAHeadSums(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> (
        postNorm: MLXArray, preNorm: MLXArray,
        mmaHeadSums: Gemma4MMAQuantizedGEMV.ActivationSums?
    ) {
        let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        let r = forwardTrunk(
            inputs, cache: cache, captureHook: nil, capturePreNorm: true,
            emitMMAHeadSums: true)
        return (r.postNorm, r.preNorm!, r.mmaHeadSums)
    }

    func callCapturingDFlashHiddenStates(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        targetLayerIds: [Int],
        forceArrayMask: Bool = false
    ) throws -> (postNorm: MLXArray, hiddenStates: [MLXArray]) {
        try DFlashTargetValidation.validateTargetLayerIds(
            targetLayerIds, layerCount: layers.count)
        let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        let hiddenCapture = Gemma4DFlashHiddenCapture(
            layerIds: targetLayerIds, layerCount: layers.count)
        let r = forwardTrunk(
            inputs,
            cache: cache,
            captureHook: nil,
            capturePreNorm: false,
            dFlashHiddenCapture: hiddenCapture,
            forceArrayMask: forceArrayMask)
        return (r.postNorm, hiddenCapture.orderedHiddenStates())
    }

    private func forwardTrunk(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureHook: ((Int, (MLXArray, MLXArray)) -> Void)?,
        capturePreNorm: Bool,
        inputEmbedding: MLXArray? = nil,
        imageTokenMask: MLXArray? = nil,
        schedulePrefill: Bool = false,
        dFlashHiddenCapture: Gemma4DFlashHiddenCapture? = nil,
        forceArrayMask requestedArrayMask: Bool = false,
        emitMMAHeadSums: Bool = false
    ) -> (
        postNorm: MLXArray,
        preNorm: MLXArray?,
        mmaHeadSums: Gemma4MMAQuantizedGEMV.ActivationSums?
    ) {
        let rectangleBatchSize = inputs.dim(0)
        let rectangleLength = inputs.dim(1)
        let wideColumns: Int = {
            guard CBv2MTPWideVerifyContext.active, !schedulePrefill, rectangleLength > 1,
                rectangleLength == CBv2MTPWideVerifyContext.columns,
                inputEmbedding == nil, imageTokenMask == nil, dFlashHiddenCapture == nil
            else { return 1 }
            return rectangleLength
        }()
        let inputBatchSize = wideColumns > 1 ? rectangleBatchSize * wideColumns : rectangleBatchSize
        let inputLength = wideColumns > 1 ? 1 : rectangleLength

        var h: MLXArray
        if let inputEmbedding {
            h = inputEmbedding.ndim == 2 ? inputEmbedding.expandedDimensions(axis: 0) : inputEmbedding
        } else {
            if let fused = Gemma4FusedScaledEmbedding.apply(
                tokens: inputs, embedding: embedTokens, embedScale: embedScale,
                hiddenSize: config.hiddenSize)
            {
                h = fused
            } else {
                h = embedTokens(inputs) * embedScale
            }
        }
        if wideColumns > 1 {
            h = h.reshaped([inputBatchSize, 1, config.hiddenSize])
            CBv2EngageMark.once("mtp-wide-verify-trunk")
        }

        var perLayerInputs: [MLXArray?]
        if hiddenSizePerLayerInput > 0,
            let embedPerLayer = embedTokensPerLayer,
            let modelProj = perLayerModelProjection,
            let projNorm = perLayerProjectionNorm
        {
            let tokenPLE =
                embedPerLayer(inputs)
                * Float(config.hiddenSizePerLayerInput).squareRoot()

            let reshapedTokenPLE = tokenPLE.reshaped(
                tokenPLE.dim(0), tokenPLE.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)

            let modelPLE = modelProj(h).reshaped(
                h.dim(0), h.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)
            let normedModelPLE = projNorm(modelPLE)

            let perLayerInputScale = pow(Float(2.0), -0.5)
            let combined = (normedModelPLE + reshapedTokenPLE) * perLayerInputScale

            perLayerInputs = (0 ..< config.numHiddenLayers).map { i in
                combined[.ellipsis, i, 0...]
            }
        } else {
            perLayerInputs = Array(repeating: nil, count: config.numHiddenLayers)
        }

        var fullCache: [KVCache?]
        if let cache {
            fullCache = cache.map { Optional($0) }
            while fullCache.count < config.numHiddenLayers {
                fullCache.append(nil)
            }
        } else {
            fullCache = Array(repeating: nil, count: config.numHiddenLayers)
        }

        let isCBv2 = fullCache.contains { ($0 as? (any CBv2AttendingLayerCache)) != nil }
        let unifiedCBv2PositionOffset: Gemma4.PositionOffset? = {
            guard isCBv2 else { return nil }
            for case let entry? in fullCache {
                if let offsets = (entry as? CBv2LayerCache)?.unifiedPositionOffsets {
                    if wideColumns > 1 {
                        let columnSteps = MLXArray((0 ..< Int32(wideColumns)).map { $0 })
                            .reshaped([1, wideColumns])
                        let expanded = (offsets.reshaped([rectangleBatchSize, 1]) + columnSteps)
                            .reshaped([inputBatchSize])
                        return .batch(expanded)
                    }
                    return .batch(offsets + 0)
                }
            }
            return nil
        }()

        var maskByType = [String: MLXFast.ScaledDotProductAttentionMaskMode]()
        if !isCBv2 {
            let useBidirectionalVision =
                imageTokenMask != nil && config.useBidirectionalAttention == "vision"
                && h.dim(1) > 1
            let useBidirectionalAll =
                config.useBidirectionalAttention == "all" && h.dim(1) > 1
            let forceArrayMask =
                useBidirectionalVision || useBidirectionalAll || requestedArrayMask
            for (i, layer) in layers.enumerated() {
                let lt = layer.layerType
                if maskByType[lt] == nil {
                    var mask: MLXFast.ScaledDotProductAttentionMaskMode
                    if lt == "sliding_attention" {
                        mask = createAttentionMask(
                            h: h, cache: fullCache[i], windowSize: config.slidingWindow,
                            returnArray: forceArrayMask)
                    } else {
                        mask = createAttentionMask(
                            h: h, cache: fullCache[i], windowSize: nil,
                            returnArray: forceArrayMask)
                    }
                    if useBidirectionalVision, let imageTokenMask {
                        mask = gemma4TextOverlayBidirectionalVision(
                            mask, isVision: imageTokenMask)
                    } else if useBidirectionalAll {
                        mask = gemma4TextSymmetrizeMask(mask)
                    }
                    maskByType[lt] = mask
                }
            }
        }

        var intermediates = [(kv: (MLXArray, MLXArray)?, positionOffset: Gemma4.PositionOffset?)](
            repeating: (nil, nil), count: config.numHiddenLayers)

        let glueChain = Gemma4GlueChainBox()
        for (idx, layer) in layers.enumerated() {
            let prevIdx = previousKvs[idx]
            let sharedKV = intermediates[prevIdx].kv
            let sharedPositionOffset = intermediates[prevIdx].positionOffset

            let v2SharedSource: (any CBv2AttendingLayerCache)? =
                isCBv2 && prevIdx != idx
                ? fullCache[prevIdx] as? (any CBv2AttendingLayerCache) : nil

            let mask = maskByType[layer.layerType]
            let isFinalPromptLayer =
                schedulePrefill && isCBv2 && idx == layers.count - 1
                && h.dim(0) > 0 && h.dim(1) >= gemma4PrefillTailMinChunk
            let outputTailRows: Int? =
                isFinalPromptLayer && gemma4PrefillTailRows > 0
                ? min(gemma4PrefillTailRows, h.dim(1)) : nil
            let useLastQueryPrefill = gemma4UseLastQueryPrefill(
                config,
                layerIdx: idx,
                batchSize: h.dim(0),
                sequenceLength: h.dim(1),
                outputTailRows: outputTailRows,
                hasCapableCache: fullCache[idx] is any CBv2LastQueryPrefillLayerCache)
            let (out, kvPair, positionOffset) = layer(
                h,
                mask: mask,
                cache: fullCache[idx],
                perLayerInput: perLayerInputs[idx],
                sharedKV: sharedKV,
                positionOffset: unifiedCBv2PositionOffset ?? sharedPositionOffset,
                v2SharedSource: v2SharedSource,
                outputTailRows: outputTailRows,
                useLastQueryPrefill: useLastQueryPrefill,
                isExpertPrefill: gemma4AllowsWeightedExpertUnsort(
                    schedulePrefill: schedulePrefill),
                glueChain: glueChain,
                nextInputLayernormWeight: idx + 1 < layers.count
                    ? layers[idx + 1].inputLayernorm.weight : nil,
                enableAttentionBranchPrefix:
                    isCBv2 && !schedulePrefill
                    && (inputBatchSize == 8 || (wideColumns > 1 && inputBatchSize % 8 == 0))
                    && inputLength == 1
                    && (!capturePreNorm || gemma4HiddenPromotedTrunkEnabled)
                    && dFlashHiddenCapture == nil,
                wideColumns: wideColumns
            )
            h = out
            intermediates[idx] = (kvPair, positionOffset)
            captureHook?(idx, kvPair)
            dFlashHiddenCapture?.capture(h, layer: idx)

            if gemma4ShouldSubmitDecodeAsyncEvalLadder(
                enabled: gemma4DecodeAsyncEvalLadderEnabled,
                schedulePrefill: schedulePrefill,
                isCBv2: isCBv2,
                batchSize: inputBatchSize,
                inputLength: inputLength,
                layerIndex: idx)
            {
                asyncEval(h)
                CBv2EngageMark.once("gemma4-b8-decode-async-ladder")
                CBv2StepProfiler.recordEvent(
                    "v2.gemma4.decode.async_eval_ladder")
            }

            let layerNumber = idx + 1
            if gemma4ShouldSubmitPrefillChunkEval(
                schedulePrefill: schedulePrefill,
                isCBv2: isCBv2,
                inputLength: inputLength,
                layerNumber: layerNumber,
                interval: gemma4EffectivePrefillChunkEvalLayers(
                    configured: gemma4PrefillChunkEvalLayers,
                    inputLength: inputLength))
            {
                asyncEval(h)
                CBv2StepProfiler.recordEvent("v2.gemma4.prefill.chunk_eval")
            }
        }

        let postNorm: MLXArray
        let mmaHeadSums: Gemma4MMAQuantizedGEMV.ActivationSums?
        if emitMMAHeadSums,
            let produced = Gemma4FinalNormMMAHeadSumsV1.apply(
                h, weight: norm.weight, eps: norm.eps)
        {
            postNorm = produced.postNorm
            mmaHeadSums = produced.sums
        } else {
            postNorm = norm(h)
            mmaHeadSums = nil
        }
        return (postNorm, capturePreNorm ? h : nil, mmaHeadSums)
    }
}

// MARK: - Bidirectional vision attention overlay (mirror of the VLM twin)

private func gemma4TextVisionBlockIds(_ isVision: MLXArray) -> MLXArray {
    let length = isVision.dim(1)
    let leading = MLXArray.zeros([isVision.dim(0), 1], dtype: .bool)
    let prev = concatenated([leading, isVision[0..., ..<(length - 1)]], axis: 1)
    let starts = logicalAnd(isVision, logicalNot(prev))
    let groupIds = cumsum(starts.asType(.int32), axis: 1) - 1
    return MLX.where(isVision, groupIds, MLXArray(Int32(-1)))
}

private func gemma4TextBidirectionalVisionMask(
    _ baseMask: MLXArray, isVision: MLXArray
) -> MLXArray {
    let blockIds = gemma4TextVisionBlockIds(isVision)
    let qBlocks = expandedDimensions(blockIds, axis: -1)  // [B, L, 1]
    let kBlocks = expandedDimensions(blockIds, axis: -2)  // [B, 1, L]
    var sameBlock = logicalAnd(qBlocks .!= MLXArray(Int32(-1)), qBlocks .== kBlocks)  // [B, L, L]
    let L = isVision.dim(1)
    let keyColumns = baseMask.dim(-1)
    if keyColumns > L {
        let pad = MLXArray.zeros([sameBlock.dim(0), L, keyColumns - L], dtype: .bool)
        sameBlock = concatenated([pad, sameBlock], axis: -1)  // [B, L, offset+L]
    }
    return logicalOr(baseMask, expandedDimensions(sameBlock, axis: 1))  // -> [B, 1, L, offset+L]
}

private func gemma4TextOverlayBidirectionalVision(
    _ mode: MLXFast.ScaledDotProductAttentionMaskMode, isVision: MLXArray
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mode {
    case .array(let maskArray):
        return .array(gemma4TextBidirectionalVisionMask(maskArray, isVision: isVision))
    default:
        return mode
    }
}

func gemma4TextSymmetrizeMask(
    _ mode: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mode {
    case .array(let maskArray):
        let queryCount = maskArray.dim(-2)
        let keyCount = maskArray.dim(-1)
        guard keyCount >= queryCount else { return mode }
        let prefixCount = keyCount - queryCount
        let current = maskArray[.ellipsis, prefixCount...]
        let symmetricCurrent = logicalOr(current, current.swappedAxes(-1, -2))
        guard prefixCount > 0 else { return .array(symmetricCurrent) }
        return .array(concatenated(
            [maskArray[.ellipsis, ..<prefixCount], symmetricCurrent], axis: -1))
    default:
        return mode
    }
}

// MARK: - Public Model

public class Gemma4TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let config: Gemma4TextConfiguration
    let model: Gemma4TextModelInner
    let fuseWeightedUnsort: Bool

    public var configuration: Gemma4TextConfiguration { config }

    public var weightedExpertUnsortRequested: Bool { gemma4FusedWeightedUnsortRequested }
    public var weightedExpertUnsortEffective: Bool { fuseWeightedUnsort }

    public var expertQMMGeometryEligible: Bool {
        gemma4SupportsCoupledExpertOptimizations(config)
    }

    public var decoderLayers: [Module] { model.layers }

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Gemma4TextConfiguration) {
        let fuseWeightedUnsort = gemma4ShouldFuseWeightedUnsort(config)
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { idx in
            let layerType = idx < config.layerTypes.count ? config.layerTypes[idx] : "sliding_attention"
            return layerType == "full_attention"
                ? (config.numGlobalKeyValueHeads ?? config.numKeyValueHeads)
                : config.numKeyValueHeads
        }
        self.fuseWeightedUnsort = fuseWeightedUnsort
        self.model = Gemma4TextModelInner(
            config,
            fuseWeightedUnsort: fuseWeightedUnsort)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard config.useBidirectionalAttention != "all" else {
            return .tokens(input.text)
        }

        let prefillStepSize = windowSize ?? 512
        var remaining = input.text
        while remaining.tokens.size > prefillStepSize {
            let chunk = remaining[.newAxis, ..<prefillStepSize]
            _ = self(chunk, cache: cache.isEmpty ? nil : cache, state: nil)
            eval(cache)
            remaining = remaining[prefillStepSize...]
        }
        return .tokens(remaining)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        if lmHead == nil,
            inputs.ndim == 2,
            inputs.dim(0) == 8,
            inputs.dim(1) == 1,
            let quantized = model.embedTokens as? QuantizedEmbedding,
            quantized.mode == .affine,
            quantized.groupSize == 64,
            quantized.bits == 4,
            Gemma4MMAQuantizedGEMV.consumesActivationSums
        {
            let produced = model.callWithMMAHeadSums(inputs, cache: cache)
            return applyLMHead(
                produced.postNorm, activationSums: produced.activationSums)
        }
        let hidden = model(inputs, cache: cache)
        return applyLMHead(hidden)
    }

    public func callAsFunction(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        imageTokenMask: MLXArray? = nil
    ) -> MLXArray {
        applyLMHead(
            model(
                inputs, cache: cache, inputEmbedding: inputEmbedding,
                imageTokenMask: imageTokenMask))
    }

    @inline(__always)
    private func tiedLMHeadMMA(
        _ hidden: MLXArray,
        activationSums: Gemma4MMAQuantizedGEMV.ActivationSums? = nil
    ) -> MLXArray? {
        guard lmHead == nil,
            let quantized = model.embedTokens as? QuantizedEmbedding,
            quantized.mode == .affine,
            let mma = Gemma4MMAQuantizedGEMV.apply(
                x: hidden,
                w: quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                activationSums: activationSums)
        else { return nil }
        return mma.reshaped(Array(hidden.shape.dropLast()) + [mma.dim(-1)])
    }

    private func tiedLMHeadTightGrid(_ hidden: MLXArray) -> MLXArray? {
        guard lmHead == nil,
            let quantized = model.embedTokens as? QuantizedEmbedding,
            quantized.groupSize == 64,
            quantized.bits == 4
        else { return nil }
        return CBv2TiedLMHeadQMVV1.matmul(
            x: hidden,
            weight: quantized.weight,
            scales: quantized.scales,
            biases: quantized.biases,
            inDim: config.hiddenSize,
            outDim: config.vocabSize)
    }

    func applyLMHead(
        _ hidden: MLXArray,
        activationSums: Gemma4MMAQuantizedGEMV.ActivationSums? = nil
    ) -> MLXArray {
        var out: MLXArray
        if let lmHead {
            out = lmHead(hidden)
        } else if let mma = tiedLMHeadMMA(
            hidden, activationSums: activationSums)
        {
            out = mma
        } else if let tight = tiedLMHeadTightGrid(hidden) {
            out = tight
        } else {
            out = model.embedTokens.asLinear(hidden)
        }
        if config.finalLogitSoftcapping > 0, !CBv2OrderOnlyLogits.engaged {
            out = gemma4CompiledLogitSoftcap(
                out, MLXArray(config.finalLogitSoftcapping))
        }
        return out
    }

    func applyRawLMHead(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        if let tight = tiedLMHeadTightGrid(hidden) {
            return tight
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func embedTokensForDrafter(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens) * Float(config.hiddenSize).squareRoot()
    }

    public func widthProbeForward(
        _ inputs: MLXArray,
        cache: [KVCache],
        captureHook: @escaping (Int, (MLXArray, MLXArray)) -> Void
    ) -> MLXArray {
        applyLMHead(model(inputs, cache: cache, captureHook: captureHook))
    }

    internal func _testCallInner(
        _ inputs: MLXArray,
        cache: [KVCache],
        captureHook: ((Int, (MLXArray, MLXArray)) -> Void)? = nil
    ) -> MLXArray {
        model(inputs, cache: cache, captureHook: captureHook)
    }

    private func extractLayerIdx(from key: String) -> Int? {
        guard let layersRange = key.range(of: "layers.") else { return nil }
        let after = key[layersRange.upperBound...]
        let end = after.firstIndex(of: ".") ?? after.endIndex
        return Int(after[..<end])
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (k, v) in weights {
            if k.contains("self_attn.rotary_emb")
                || k.contains("input_max")
                || k.contains("input_min")
                || k.contains("output_max")
                || k.contains("output_min")
            {
                continue
            }

            if let layerIdx = extractLayerIdx(from: k),
                config.layerUsesSharedKV(layerIdx: layerIdx),
                k.contains(".self_attn.k_proj.")
                    || k.contains(".self_attn.v_proj.")
                    || k.contains(".self_attn.k_norm.")
                    || k.contains(".self_attn.v_norm.")
            {
                continue
            }

            if k.hasSuffix(".experts.gate_up_proj") {
                let base = String(k.dropLast(".gate_up_proj".count))
                let parts = MLX.split(v, parts: 2, axis: -2)
                sanitized["\(base).switch_glu.gate_proj.weight"] = parts[0]
                sanitized["\(base).switch_glu.up_proj.weight"] = parts[1]
                continue
            }

            if k.hasSuffix(".experts.down_proj") {
                let base = String(k.dropLast(".down_proj".count))
                sanitized["\(base).switch_glu.down_proj.weight"] = v
                continue
            }

            sanitized[k] = v
        }
        fuseExpertGateUpStorage(&sanitized)
        return sanitized
    }

    private func fuseExpertGateUpStorage(_ sanitized: inout [String: MLXArray]) {
        guard switchGateUpFusePrefillEnabled else { return }
        let gateWeightSuffix = ".experts.switch_glu.gate_proj.weight"
        for key in sanitized.keys where key.hasSuffix(gateWeightSuffix) {
            let base = String(key.dropLast(gateWeightSuffix.count))
            guard let layerIdx = extractLayerIdx(from: key),
                layerIdx < model.layers.count,
                let experts = model.layers[layerIdx].experts,
                let gateWeight = sanitized["\(base).experts.switch_glu.gate_proj.weight"],
                let gateScales = sanitized["\(base).experts.switch_glu.gate_proj.scales"],
                let gateBiases = sanitized["\(base).experts.switch_glu.gate_proj.biases"],
                let upWeight = sanitized["\(base).experts.switch_glu.up_proj.weight"],
                let upScales = sanitized["\(base).experts.switch_glu.up_proj.scales"],
                let upBiases = sanitized["\(base).experts.switch_glu.up_proj.biases"],
                let storage = SwitchGateUpFusedStorage(
                    gateWeight: gateWeight, gateScales: gateScales, gateBiases: gateBiases,
                    upWeight: upWeight, upScales: upScales, upBiases: upBiases)
            else { continue }
            sanitized["\(base).experts.switch_glu.gate_proj.weight"] = storage.gateWeight
            sanitized["\(base).experts.switch_glu.gate_proj.scales"] = storage.gateScales
            sanitized["\(base).experts.switch_glu.gate_proj.biases"] = storage.gateBiases
            sanitized["\(base).experts.switch_glu.up_proj.weight"] = storage.upWeight
            sanitized["\(base).experts.switch_glu.up_proj.scales"] = storage.upScales
            sanitized["\(base).experts.switch_glu.up_proj.biases"] = storage.upBiases
            experts.switchGLU.bindFusedGateUpStorage(storage)
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let firstKvShared = config.numHiddenLayers - config.numKvSharedLayers

        var caches = [any KVCache]()
        for i in 0 ..< firstKvShared {
            if config.layerTypes[i] == "full_attention" {
                if let maxKVSize = parameters?.maxKVSize {
                    caches.append(RotatingKVCache(maxSize: maxKVSize, keep: 4))
                } else {
                    caches.append(StandardKVCache())
                }
            } else {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }
}

// MARK: - LoRA

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.map { $0.selfAttn }
    }
}

// MARK: - ContinuousBatchingV2

extension Gemma4TextConfiguration {
    public var cbv2LayerKinds: [CBv2LayerKind] {
        CBv2LayerKindDerivation.gemma4LayerKinds(
            layerTypes: layerTypes,
            slidingWindow: slidingWindow,
            numKvSharedLayers: numKvSharedLayers,
            headDim: headDim,
            globalHeadDim: globalHeadDim,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            numGlobalKeyValueHeads: numGlobalKeyValueHeads,
            isBidirectional: useBidirectionalAttention == "all"
        )
    }
}

extension Gemma4TextModel {
    public var cbv2LayerKinds: [CBv2LayerKind] {
        config.cbv2LayerKinds
    }

    public var cbv2PrefillChunkEvalInterval: Int {
        gemma4PrefillChunkEvalLayers
    }

    public enum CBv2CompatibilityError: Error, Equatable, CustomStringConvertible {
        case fullyBidirectionalAttentionUnsupported

        public var description: String {
            switch self {
            case .fullyBidirectionalAttentionUnsupported:
                return "Gemma4 CBv2 does not support use_bidirectional_attention=all because split prefill cannot preserve whole-prompt visibility"
            }
        }
    }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) throws -> [any CBv2AttendingLayerCache] {
        guard config.useBidirectionalAttention != "all" else {
            throw CBv2CompatibilityError.fullyBidirectionalAttentionUnsupported
        }
        return try cbv2LayerKinds.enumerated().map { index, kind in
            try makeLayerCache(index, kind)
        }
    }
}

// MARK: - ContinuousBatchingV2 prompt-only output narrowing

extension Gemma4TextModel: CBv2LanguageModelPrefillForwardable {

    public var cbv2SupportsPackedPrefill: Bool { true }
    public var cbv2SupportsPackedMultimodalPrefill: Bool { true }

    public func cbv2Prefill(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray?,
        cache: [KVCache]?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        let hidden = model.cbv2Prefill(
            inputs, cache: cache, inputEmbedding: inputEmbedding)
        switch requirement {
        case .evaluationOnly:
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            return applyLMHead(hidden[0..., -1, 0...])
        }
    }
}

extension Gemma4TextModel: CBv2LanguageModelDecodeOutputCoversCacheMutations {}

// MARK: - ContinuousBatchingV2 multimodal (vision prefill)

extension Gemma4TextModel: CBv2EmbeddingForwardable {

    public var supportsVisionSpanPrefill: Bool {
        config.useBidirectionalAttention == "vision"
    }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        model.embedTokens(inputs) * model.embedScale
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        applyLMHead(model(inputs, cache: cache, inputEmbedding: inputEmbedding))
    }
}

// MARK: - ContinuousBatchingV2 MTP (speculative decoding)

private let gemma4HiddenPromotedTrunkEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_HIDDEN_PROMOTED_TRUNK"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

extension Gemma4TextModel: CBv2MTPForwardable {

    public var cbv2MTPCaptureLayers: CBv2MTPCaptureLayers? {
        let full = model.lastFullAttentionNonSharedIdx
        let sliding = model.lastSlidingAttentionNonSharedIdx
        guard full >= 0, sliding >= 0 else { return nil }
        return CBv2MTPCaptureLayers(full: full, sliding: sliding)
    }

    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache]
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        guard gemma4HiddenPromotedTrunkEnabled else {
            let (postNorm, preNorm) = model.callCapturingPreNorm(tokens, cache: caches)
            return (applyLMHead(postNorm), preNorm)
        }
        let r = model.callCapturingPreNormWithMMAHeadSums(tokens, cache: caches)
        return (applyLMHead(r.postNorm, activationSums: r.mmaHeadSums), r.preNorm)
    }
}
