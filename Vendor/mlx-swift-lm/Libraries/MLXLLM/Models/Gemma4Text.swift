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
//
// File-private, self-contained compiled fusions. They do NOT depend on the
// SwitchLayers / HardwareInfo lane so this file builds stand-alone; when that
// lane lands public `safeGeluApproximate` / `MLXHardwareInfo` these can collapse
// to the shared symbols (identical math + same env knob) with no behavior change.
// `compile(shapeless: true)` is gated by `MLX_COMPILED_DECODE` (default on),
// mirroring `MLXHardwareInfo.isCompiledDecodeSupported`, so M1/M2 + macOS Tahoe
// (MLX #3329) can opt out without a code change. Matches the ungated
// `compiledSiluProduct` / `weightedExpertSum` convention already in this tree.
private let gemma4CompiledDecodeSupported: Bool = {
    if let raw = ProcessInfo.processInfo.environment["MLX_COMPILED_DECODE"] {
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }
    return true
}()

// MARK: - CBv2 B=8 decode graph-submission ladder

/// Earlier graph submission is ON by default for the one scored decode
/// geometry below. `DARKBLOOM_GEMMA4_DECODE_ASYNC_EVAL_LADDER=0` (also
/// `false`/`no`/`off`) is the attribution and emergency kill switch.
@inline(__always)
internal func resolveGemma4DecodeAsyncEvalLadderEnabled(_ raw: String?) -> Bool {
    guard let raw else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}

private let gemma4DecodeAsyncEvalLadderEnabled =
    resolveGemma4DecodeAsyncEvalLadderEnabled(
        ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_DECODE_ASYNC_EVAL_LADDER"])

/// Pure, fail-closed policy for the Gemma 4 decode submission ladder.
///
/// Layer indices name boundaries AFTER a complete decoder layer. In
/// particular, the MoE layer has already recombined its dense and sparse
/// branches before a selected boundary is submitted, so both branches retain
/// their natural concurrency inside the same graph frontier.
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

    switch layerIndex {
    case 0, 1, 5, 11, 17, 23, 27:
        return true
    default:
        return false
    }
}

// MARK: - CBv2 prompt-path knobs (prefill only; decode never reads these)

@inline(__always)
private func gemma4TruthyFlag(_ raw: String?) -> Bool {
    guard let raw else { return false }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

/// Submit intermediate Gemma 4 prefill graphs while Swift continues to build
/// later layers. This changes only when already-built work is queued; the
/// operations and results are unchanged. Single-token decode is excluded.
///
/// The 18-layer default leaves twelve layers of the 30-layer 26B model for
/// useful CPU/GPU overlap. `DARKBLOOM_GEMMA4_PREFILL_CHUNK_EVAL=0` restores
/// one final submission; another positive value tunes the layer interval.
@inline(__always)
internal func resolveGemma4PrefillChunkEvalLayers(_ raw: String?) -> Int {
    guard let raw, let value = Int(raw) else { return 18 }
    return max(0, value)
}

private let gemma4PrefillChunkEvalLayers = resolveGemma4PrefillChunkEvalLayers(
    ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_PREFILL_CHUNK_EVAL"])

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

/// CBv2 consumes only the final prompt position, so the LAST decoder layer
/// can keep full attention and every K/V write while retaining just this
/// many trailing rows for `o_proj`, the residual, the feed-forward/MoE
/// branches, PLE, and the final norm. One row is what the frontier needs,
/// and it puts that work on the same small-M expert path as B=1 decode.
///
/// `DARKBLOOM_GEMMA4_PREFILL_TAIL_ROWS=0` restores the full final layer
/// (the kill switch); a larger value is for comparing kernel geometries.
private let gemma4PrefillTailRows: Int = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_TAIL_ROWS"],
        let value = Int(raw)
    else { return 1 }
    return max(0, value)
}()

/// Parse the independent direct expert-reduction control, which is ON by
/// default: the coupled weighted-unsort + safe-R1 pair is what was measured
/// faster, and the ranked box sets no environment.
///
/// R1 is selected by MLX itself for the sorted expert QMM whenever the
/// checkpoint satisfies the selector's contract, so on the production
/// checkpoint the weighted half was the only part still left on the table.
/// `MLX_GEMMA4_FUSED_WEIGHTED_UNSORT=0` (or `false`/`no`/`off`) is the kill
/// switch back to scatter-unsort + `weightedExpertSum`.
/// `gemma4SupportsCoupledExpertOptimizations` still has the final say, so a
/// checkpoint that categorically cannot run safe R1 never reaches the
/// weighted-only state the comment below warns about.
func gemma4FusedWeightedUnsortFlag(_ raw: String?) -> Bool {
    guard let raw else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}

/// The retained weighted/R1 pair is measured only on scheduled CBv2 prefill.
/// Direct model forwards keep the established reduction path.
func gemma4AllowsWeightedExpertUnsort(schedulePrefill: Bool) -> Bool {
    schedulePrefill
}

/// The exact production expert topology. Near matches retain the established
/// unsort + weighted sum and the generic gather-QMM route.
func gemma4SupportsProductionExpertTopology(_ config: Gemma4TextConfiguration) -> Bool {
    config.enableMoeBlock
        && config.hiddenSize == 2816
        && config.numHiddenLayers == 30
        && config.numExperts == 128
        && config.topKExperts == 8
        && config.moeIntermediateSize == 704
        && config.useBidirectionalAttention == "vision"
}

/// The safe Gemma 4 expert-QMM selector (`classify_gemma4_expert_qmm`) rejects
/// anything that is not affine 4-bit at group size 64 with
/// `fallback_quantization`, before it ever looks at topology. The remaining
/// selector conditions (dtypes, contiguity, assignment count, AOT metallib,
/// NAX precedence) are dispatch-time facts MLX reports separately.
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

/// Direct weighted unsort and the safe expert-QMM (R1) kernel are one measured
/// unit. Weighted unsort on its own is materially slower than the retained
/// baseline, so it must never engage on a checkpoint where safe R1
/// categorically cannot — which is any checkpoint outside the exact production
/// topology *and* the selector's 4-bit / group-size-64 quantization contract.
/// Both features gate on this single predicate, so reported eligibility
/// matches real dispatch and no weighted-only state is reachable.
func gemma4SupportsCoupledExpertOptimizations(_ config: Gemma4TextConfiguration) -> Bool {
    gemma4SupportsProductionExpertTopology(config)
        && gemma4SupportsSafeExpertQMMQuantization(config)
}

internal let gemma4FusedWeightedUnsortRequested = gemma4FusedWeightedUnsortFlag(
    ProcessInfo.processInfo.environment["MLX_GEMMA4_FUSED_WEIGHTED_UNSORT"])

/// Pure policy seam for the weighted-unsort resolution, so the coupling with
/// safe R1 is unit-testable without building a production-sized model. The
/// request is the only thing separating this from `expertQMMGeometryEligible`.
func gemma4ShouldFuseWeightedUnsort(
    _ config: Gemma4TextConfiguration,
    requested: Bool = gemma4FusedWeightedUnsortRequested
) -> Bool {
    requested && gemma4SupportsCoupledExpertOptimizations(config)
}


/// Chunks shorter than this keep the unnarrowed final layer: the saving
/// scales with the discarded row count, and tiny chunks are dominated by
/// fixed overhead. Overridable so tests can exercise the narrow path on
/// small fixtures.
private let gemma4PrefillTailMinChunk: Int = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_TAIL_MIN_CHUNK"],
        let value = Int(raw)
    else { return 128 }
    return max(2, value)
}()

/// Final-layer last-query prefill: project and cache the whole chunk's K/V
/// but compute Q and attention for the frontier row alone. Requires the tail
/// narrowing above (exactly one retained row) and a cache that can commit
/// full K/V for a single query. Default ON with
/// `DARKBLOOM_GEMMA4_PREFILL_LAST_QUERY=0` as the kill switch.
private let gemma4PrefillLastQueryEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_LAST_QUERY"]
    else { return true }
    return gemma4TruthyFlag(raw)
}()

/// The final layer must own a full-attention, non-shared cache for
/// last-query prefill to be equivalent. Sliding windows give each query a
/// different visible span, and a KV-shared final layer writes nothing.
func gemma4SupportsLastQueryPrefill(_ config: Gemma4TextConfiguration) -> Bool {
    config.layerTypes.count == config.numHiddenLayers
        && config.layerTypes.last == "full_attention"
        && !config.layerUsesSharedKV(layerIdx: config.numHiddenLayers - 1)
}

/// Pure policy seam for the final layer's prompt specialization, so the
/// decision is unit-testable without building a model. Cache capability is
/// supplied by the caller: only the contiguous CBv2 cache exposes the
/// atomic full-K/V + last-query operation.
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

/// Approximate (tanh) GELU written with `x * x * x` instead of the Power
/// primitive (`x ** 3`) so it is safe under `compile(shapeless: true)` — the
/// Power primitive returns zero results on the Tahoe Metal JIT (MLX #3329).
/// Numerically identical to `MLXNN.geluApproximate` (vMLX `safeGeluApproximate`).
private let gemma4SafeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    return gemma4CompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Safe approximate GELU and its following dense-MLP product in one compiled
/// graph. Operation order matches `gemma4SafeGeluApproximate(gate) * up`.
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

/// Final-logit softcap (`tanh(x / cap) * cap`) fused into one Metal dispatch
/// (vMLX `compiledLogitSoftcap`). The untyped (float32) cap keeps the softcap
/// math — and the logits handed to the sampler — full precision.
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

/// Default profile used while decoding Gemma4 text configuration. Direct
/// language-model checkpoints and nested VLM checkpoints historically shipped
/// different omission semantics; selecting the profile at the decoder boundary
/// keeps one implementation without silently changing VLM topology.
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
    /// Any explicit expert-path quantization entry makes the coupled
    /// weighted-unsort/R1 optimization fail closed. The runtime quantizer
    /// resolves these entries per module, so global bits/group size alone is
    /// not proof that every expert projection reaches safe R1.
    public var hasExpertQuantizationOverrides: Bool {
        gemma4HasExpertQuantizationOverrides(perLayerQuantization)
    }

    // MoE (only set on the 26B-A4B variant; 2B/4B/31B are dense)
    public internal(set) var enableMoeBlock: Bool = false
    public internal(set) var numExperts: Int?
    public internal(set) var topKExperts: Int?
    public internal(set) var moeIntermediateSize: Int?

    // RoPE parameters (nested dict with full_attention/sliding_attention sub-configs)
    public internal(set) var ropeParameters: [String: [String: StringOrNumber]]?

    // "vision" enables blockwise bidirectional attention within image/video
    // soft-token spans. "all" makes the full prefill bidirectional (bounded by
    // the configured window on sliding layers). nil/other remains causal.
    public internal(set) var useBidirectionalAttention: String?

    // Derived properties
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

    /// The synthesized encoder silently dropped the effective quantization
    /// metadata (it has no `CodingKeys` case), so a
    /// decode→encode→decode round trip lost the nested quantization contract
    /// and a later strict load of a quantized checkpoint skipped quantization
    /// outright. Encode explicitly: every keyed property plus the nested
    /// `quantization` block in exactly the shape the decoder first looks for.
    /// The derived rope thetas/partial factor re-derive from `ropeParameters`
    /// on decode, so they are intentionally not keyed.
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
            // Preserve the former nested VLM DTO's PLE contract: zero hidden
            // width disables both tensors, while positive hidden width requires
            // an explicitly positive vocabulary width.
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
                // The deleted VLM tower interpreted an explicit empty list as
                // all sliding-attention layers (non-VLM checkpoints inherit
                // the same robust fallback rather than trapping later).
                self.layerTypes = Array(
                    repeating: "sliding_attention", count: numHiddenLayers)
            } else if decoded.count < numHiddenLayers {
                // The deleted towers fell back to sliding attention for
                // out-of-range layer indices; normalize short explicit lists
                // by padding instead of trapping at model construction.
                self.layerTypes =
                    decoded
                    + Array(
                        repeating: "sliding_attention",
                        count: numHiddenLayers - decoded.count)
            } else {
                self.layerTypes = Array(decoded.prefix(numHiddenLayers))
            }
        } else if isVLM {
            // The same VLM fallback applies when the key is absent.
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
    /// Overlay checkpoint-level quantization metadata on a decoded text
    /// configuration. VLM checkpoints commonly keep this metadata beside
    /// `text_config`; an absent overlay preserves any nested metadata.
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

    /// Overlay the effective root mixed-precision map used by the model
    /// loader. Expert-path entries make the coupled optimization fail closed,
    /// even when the root default remains nominally 4-bit/group-64.
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

    /// Predicate for whether a layer uses shared K/V (consuming it from an
    /// earlier layer rather than projecting its own).
    ///
    /// A layer is shared when either:
    /// - `forceSharedKV` is true (drafter / assistant models where every layer
    ///   borrows K/V from the target), or
    /// - the config declares `numKvSharedLayers > 0` AND this layer's index
    ///   falls within the trailing shared block.
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

private let gemma4QKVNormKernel = MLXFast.metalKernel(
    name: "gemma4_b8_qkv_rms_norm_v1",
    inputNames: ["q", "k", "v", "q_weight", "k_weight"],
    outputNames: ["q_out", "k_out", "v_out"],
    source: """
        constexpr uint reads = 4;
        const uint row = threadgroup_position_in_grid.x;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;

        const device T* input = q;
        const device T* weight = q_weight;
        device T* output = q_out;
        uint local_row = row;
        bool weighted = true;
        if (!KEY_VALUE_SHARED && row >= Q_ROWS + K_ROWS) {
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

        input += local_row * D + lid * reads;
        output += local_row * D + lid * reads;
        weight += lid * reads;
        // Keep the pointer inside the V allocation for Q rows even though
        // those rows never dereference it. K rows advance to their matching
        // V row only in the compile-time shared-input variant.
        device T* shared_value_output = v_out;
        if (KEY_VALUE_SHARED && row >= Q_ROWS) {
            shared_value_output += local_row * D + lid * reads;
        }

        float sum = 0.0f;
        for (uint i = 0; i < reads; ++i) {
            const float value = float(input[i]);
            sum += value * value;
        }
        sum = simd_sum(sum);

        threadgroup float partials[32];
        threadgroup float inverse_rms;
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

        for (uint i = 0; i < reads; ++i) {
            const T normalized = T(float(input[i]) * inverse_rms);
            output[i] = weighted ? weight[i] * normalized : T(1) * normalized;
            // Gemma's full-attention K-eq-V layers feed the same raw key
            // projection to K RMSNorm and V RMSNormNoScale. The reduction
            // above is therefore identical for both outputs; keep each
            // output's established final expression, but write V while the
            // exact normalizer and input value are live.
            if (KEY_VALUE_SHARED && row >= Q_ROWS) {
                shared_value_output[i] = T(1) * normalized;
            }
        }
    """,
    ensureRowContiguous: true
)

private func gemma4FusedQKVNorm(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    qWeight: MLXArray,
    kWeight: MLXArray,
    eps: Float,
    keyValueShared: Bool
) -> (MLXArray, MLXArray, MLXArray)? {
    guard eps == 1.0e-6,
        q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16,
        qWeight.dtype == .bfloat16, kWeight.dtype == .bfloat16,
        q.ndim == 4, k.ndim == 4, v.ndim == 4,
        q.dim(0) == 8, q.dim(1) == 1, q.dim(2) == 16,
        k.dim(0) == 8, k.dim(1) == 1, v.shape == k.shape,
        q.dim(3) == k.dim(3),
        (q.dim(3) == 256 && k.dim(2) == 8) || (q.dim(3) == 512 && k.dim(2) == 2),
        qWeight.shape == [q.dim(3)], kWeight.shape == [q.dim(3)],
        !keyValueShared || v.shape == k.shape
    else { return nil }

    let dimension = q.dim(3)
    let qRows = 8 * 16
    let kRows = 8 * k.dim(2)
    let threads = dimension / 4
    // In the exact K-eq-V case V reads kRaw, so one row reduction produces
    // both the weighted K and no-scale V outputs. Keep the ordinary three
    // banks for every non-shared projection and for all guard failures.
    let normRows = qRows + kRows + (keyValueShared ? 0 : kRows)
    let outputs = gemma4QKVNormKernel(
        [q, k, v, qWeight, kWeight],
        template: [
            ("T", q.dtype), ("D", dimension), ("Q_ROWS", qRows), ("K_ROWS", kRows),
            ("KEY_VALUE_SHARED", keyValueShared),
        ],
        grid: (normRows * threads, 1, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [q.shape, k.shape, v.shape],
        outputDTypes: [q.dtype, k.dtype, v.dtype]
    )
    return (outputs[0], outputs[1], outputs[2])
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
        // Snapshot: `+ 0` creates a graph-safe copy so cache.update()
        // advancing offsetArray doesn't shift the query RoPE position.
        .graphArray(compilableRot.offsetArray + 0)
    } else if let compilable = cache as? CompilableKVCache {
        // Snapshot: `+ 0` creates a graph-safe copy so cache.update()
        // advancing offsetArray doesn't shift the query RoPE position.
        .graphArray(compilable.offsetArray + 0)
    } else if let batchCache = cache as? BatchPositionedKVCache {
        // Snapshot the per-sequence offsets before cache.update(...) advances them.
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
    // A fully-masked query row (every key masked -> all -inf) softmaxes to NaN.
    // For left-padded batches these are the padding query positions, whose
    // outputs are discarded — but `0 * NaN = NaN` in the value matmul below
    // would propagate NaN into the hidden state, and a later layer's real
    // queries (which mask padding keys to weight 0) then hit `0 * NaN` again
    // and corrupt EVERY row of the batch. Map NaN -> 0 so a fully-masked query
    // contributes nothing. This matches `MLXFast.scaledDotProductAttention`,
    // which this manual fallback replaces for the batched (ragged) path.
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

        // Full attention uses globalHeadDim, sliding uses headDim
        self.effectiveHeadDim =
            isSliding ? config.headDim : config.globalHeadDim

        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads

        // K-eq-V for full attention layers
        self.useKeqV = config.attentionKeqV && !isSliding
        // Full layers honor `num_global_key_value_heads` whenever it is
        // present, independent of `attention_k_eq_v`; k_eq_v only elides the
        // v_proj. This restores the deleted inline VLM tower's rule — a full
        // layer with global heads different from the sliding count and
        // k_eq_v=false still allocates its K/V projections for the global
        // count, matching such checkpoints' weights.
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

        // RoPE: sliding uses default, full uses proportional with partial rotation
        if isSliding {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.slidingRopeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.fullRopeTheta, traditional: false,
                scalingConfig: [
                    "type": .string("proportional"),
                    "partial_rotary_factor": .float(config.fullPartialRotaryFactor),
                ],
                maxPositionEmbeddings: nil)
        }

        super.init()
    }

    /// Route only exact production CBv2 Q/K projections through the shared
    /// activation-sum QMV. The ordinary layer call remains the fail-closed
    /// path for prefill, assistants, representation drift, and V/O matrices.
    @inline(__always)
    private func qkProjection(
        _ layer: Linear,
        _ x: MLXArray,
        activationSums: CBv2AttentionQKQMVV1.ActivationSums?
    ) -> MLXArray {
        guard let quantized = layer as? QuantizedLinear,
            quantized.bias == nil,
            let projected = CBv2AttentionQKQMVV1.matmul(
                x: x,
                weight: quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode,
                activationSums: activationSums)
        else { return layer(x) }
        return projected
    }

    /// Exact B8/L1 attention output projection. Sliding/full K widths select
    /// the tight affine4 fast-QMV replica; every other path keeps the layer.
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
                mode: quantized.mode)
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
        useLastQueryPrefill: Bool = false
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4.PositionOffset) {
        // ContinuousBatchingV2: the layer cache owns both the KV update and
        // the attention computation (no masks, no padding — see
        // CBv2Contracts.swift). Entirely separate branch; the legacy paths
        // below are untouched.
        if let layerCacheV2 = cache as? (any CBv2AttendingLayerCache) {
            return forwardV2(
                x, layerCache: layerCacheV2, source: v2SharedSource,
                sharedKV: sharedKV, positionOffset: positionOffset,
                outputStart: outputStart, useLastQueryPrefill: useLastQueryPrefill)
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
            // KV-shared layers use pre-computed KV from an earlier layer
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

            // K-eq-V (`attention_k_eq_v: true` on Gemma 4 26B/31B):
            // values reuses the raw key projection (pre-norm), then goes
            // through its own `vNorm` and transpose to land in the same
            // `[B, n_kv_heads, L, D]` layout as keys.
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

        // Adjust mask if cache size differs from mask size
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
            // CompilableKVCache: can't read Int offset without readback.
            // During compiled decode L==1, so L>1 && hasCachedPrefix is
            // false anyway. Setting true is safe for the prefill path.
            hasCachedPrefix = true
        }

        // vmlx #52 text-path: Gemma 4 attention scores can exceed the fp16
        // range (±65504) on long contexts, and the fused/composed SDPA shapes
        // would materialize non-finite intermediates. Promote Q/K/V to
        // float32 for the attention math when the activation dtype is fp16,
        // then cast back so `oProj` sees its own dtype. Mirrors the deleted
        // inline VLM twin; bf16 activations (production) skip the cast.
        // The CBv2 path applies the same promotion to queries below; its cache
        // keeps K/V in their storage dtype and widens the attention views.
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

    /// ContinuousBatchingV2 attention path. The `CBv2AttendingLayerCache`
    /// owns the KV update AND the attention computation, so this method only
    /// projects/normalizes/ropes Q (and K/V for non-shared layers) and
    /// dispatches. The model never builds masks and never pads — decode is
    /// rectangular `[B, 1]`, prefill is per-request `[1, chunk]`.
    ///
    /// Invariant 1 (report 10 §4): RoPE offsets are per-row absolutes,
    /// snapshotted BEFORE `updateAndAttend` advances the rows, and KV-shared
    /// layers reuse the SOURCE layer's captured snapshot byte-identically.
    private func forwardV2(
        _ x: MLXArray,
        layerCache: any CBv2AttendingLayerCache,
        source: (any CBv2AttendingLayerCache)?,
        sharedKV: (MLXArray, MLXArray)?,
        positionOffset: Gemma4.PositionOffset?,
        outputStart: Int = 0,
        useLastQueryPrefill: Bool = false
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4.PositionOffset) {
        let (B, L) = (x.dim(0), x.dim(1))
        precondition(
            outputStart >= 0 && outputStart < L,
            "Gemma4: output narrowing start \(outputStart) outside chunk length \(L)")

        // Last-query prefill projects Q for the frontier row only. Every
        // other path keeps the full query rectangle and narrows (if at all)
        // AFTER attention.
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

        // One exact affine activation-sum table feeds both Q and K. Forced
        // shared-KV assistants have no K projection and stay on their stock Q
        // path; target CBv2 decode has a non-shared K on every layer.
        let qkActivationSums = usesSharedKV
            ? nil : CBv2AttentionQKQMVV1.activationSums(for: x)
        let queryRaw = qkProjection(
            qProj, queryInput, activationSums: qkActivationSums
        ).reshaped(B, queryLength, nHeads, effectiveHeadDim)

        if usesSharedKV {
            // KV-shared layer: projects queries only and borrows (K, V) from
            // the source layer's cache at attention time. The RoPE offsets
            // MUST be the source layer's pre-update snapshot (threaded by the
            // trunk) — reading `source.positionOffsets` here would observe
            // positions already advanced by the source's update this step.
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

        // A unified contiguous bank supplies one graph-safe pre-step snapshot
        // for every layer. Standalone and paged caches retain the established
        // per-layer capture (`+ 0` = graph-safe copy, same convention as
        // gemma4CapturePositionOffset). KV-shared consumers of this layer
        // reuse this exact snapshot via the returned PositionOffset.
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

        // The frontier query sits `outputStart` positions past the chunk's
        // first token, so last-query prefill must shift its RoPE position.
        // K/V keep the unshifted capture: they cover the whole chunk.
        let queryPositionOffset: Gemma4.PositionOffset =
            lastQueryCache == nil
            ? captured
            : .batch(capturedOffsets + Int32(outputStart))
        let kRaw = qkProjection(
            kProj, x, activationSums: qkActivationSums
        ).reshaped(B, L, nKvHeads, effectiveHeadDim)
        let vRaw: MLXArray
        if let vProj {
            vRaw = vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
        } else {
            vRaw = kRaw
        }

        let normalized = gemma4FusedQKVNorm(
            q: queryRaw, k: kRaw, v: vRaw,
            qWeight: qNorm.weight, kWeight: kNorm.weight, eps: config.rmsNormEps,
            keyValueShared: vProj == nil)
        var queries = normalized?.0 ?? qNorm(queryRaw)
        var k = normalized?.1 ?? kNorm(kRaw)
        var v = normalized?.2 ?? vNorm(vRaw)

        queries = queries.transposed(0, 2, 1, 3)
        queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: queryPositionOffset)
        k = k.transposed(0, 2, 1, 3)
        k = gemma4ApplyRotaryPosition(rope, to: k, offset: captured)

        v = v.transposed(0, 2, 1, 3)

        let outputDType = queries.dtype
        let attentionQueries =
            outputDType == .float16 ? queries.asType(.float32) : queries
        let attention: MLXArray
        if let lastQueryCache {
            attention = lastQueryCache.updateAndAttendLastQuery(
                queries: attentionQueries, keys: k, values: v, scale: scale, sinks: nil)
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

/// Width-probe observability sink (exactness round three, 2026-08-25).
///
/// Armed ONLY by the operator-driven `width-probe` diagnostic verb so it can
/// record every MoE router's expert scores and top-K selection per forward;
/// nil in production (one optional check per MoE layer per forward — no
/// tensor work, no graph change when disarmed). The recorder receives the
/// PRE-selection expert scores `[.., E]` and the selected `topKIndices`
/// `[.., K]`, in layer execution order — the width-divergence localization
/// needs exactly this seam to decide whether a forward-width numeric flip
/// first enters the network at a router selection (unfixable-by-kernel
/// design) or only at the final logits (width-stable head candidate).
public enum Gemma4RouterProbe {
    nonisolated(unsafe) public static var recorder:
        ((_ expertScores: MLXArray, _ topKIndices: MLXArray) -> Void)?
}

/// ROUTE-001: one-dispatch, byte-identical replacement of the decode router's
/// selection chain — `argPartition(kth: E-8)` → slice → `takeAlong` →
/// `softmax(precise)` over 8 → `perExpertScale` gather + multiply — for the
/// exact B=8 decode geometry (`expertScores` [8, 1, 128] bf16). Five sort /
/// gather / softmax / gather / multiply dispatches per MoE layer per step
/// (plus the contiguous copy the strided index slice forces downstream)
/// collapse into one 8-threadgroup kernel.
///
/// Exactness (counting-predecessors lemma): `ArgPartition::eval_gpu` on Metal
/// is `gpu_merge_sort(argsort=true)` — a FULL stable merge sort (sort.cpp) —
/// so the sliced `[kth...]` output is the stable ascending argsort tail. Under
/// sort.h's `LessThan` comparator (NaN ordered after every non-NaN, ties kept
/// in original index order by stability) each element's stable-sort position
/// equals its predecessor count, which the kernel evaluates directly; the
/// selected values then run a verbatim transcription of
/// `softmax_single_row<bfloat16_t, float, N_READS=4>` (softmax.h — same lane
/// layout, same `Limits<float>::min` padding, same `fast::exp`, same
/// `simd_max`/`simd_sum` reduction order on one 32-thread simdgroup) and the
/// stock bf16 `Multiply` expression against the gathered per-expert scale.
/// Bit-exact parity vs the stock op chain verified on uniform / tied /
/// ulp-near-tie / ±inf / NaN / realistic rows (indices and uint16-viewed
/// weights).
///
/// Fail-closed: any other row count, sequence length, expert count, top-K, or
/// dtype takes the established chain (cohort prefill at [8, 1024, ·] never
/// matches; the narrowed final-layer prompt tail at [8, 1, ·] does, and is
/// bit-identical there too). Kill switch:
/// `DARKBLOOM_GEMMA4_FUSED_ROUTER_TOP8=0`.
private enum Gemma4FusedRouterTop8 {
    /// DEFAULT OFF (`DARKBLOOM_GEMMA4_FUSED_ROUTER_TOP8=1` enables): the
    /// fused chain is bit-exact (113/113 adversarial parity) but measured
    /// +~0.1 ms/round inside the +0.27 ms consolidation cost of three
    /// counterbalanced local B=8 probe pairs — dispatch deletion does not
    /// pay while the concurrent encoder overlaps these small kernels.
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
            expertScores.dim(0) == rows,
            expertScores.dim(1) == 1,
            expertScores.dim(2) == experts,
            expertScores.dtype == .bfloat16,
            perExpertScale.ndim == 1,
            perExpertScale.dim(0) == experts,
            perExpertScale.dtype == .bfloat16
        else { return nil }
        CBv2EngageMark.once("router-top8")

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

/// GLUE-003: one-per-forward chain box. Layer L's fused tail deposits the
/// (output, next-layer-input-norm) pair; layer L+1 consumes the norm instead
/// of re-reading and re-normalizing the same tensor — guarded by pointer
/// identity on the source array, so any intervening transformation falls back
/// to the stock `inputLayernorm(x)`.
public final class Gemma4GlueChainBox {
    var pending: (source: MLXArray, normed: MLXArray)?
    public init() {}
}

/// MMA-064': triage gate for the GLUE-003 cross-layer input-norm chain.
///
/// DEFAULT ON, so an unset environment reproduces the established tree
/// bit-for-bit: `Gemma4GlueChainBox()` is constructed exactly as before and
/// every downstream branch sees a non-nil chain. `DARKBLOOM_GEMMA4_GLUE_CHAIN`
/// set to `0`/`false`/`no`/`off` passes `nil` instead, which fails the
/// `let chain = glueChain` binding at the chained-tail call site and drops the
/// forward onto the GLUE-002 parent-only fused tail (`Gemma4FusedLayerGlue.tail`,
/// layer scalar still folded) plus the stock `inputLayernorm` — the incumbent
/// path this tree carried before MMA-041 introduced the chain.
///
/// WHY THIS EXISTS. GLUE-003 arrived in `d2dc948` (MMA-041) and has never been
/// acquitted on the ranked box; our last accepted tree (`ab673cda`, 1.65400)
/// did not carry it. It is also the mechanism `josusanmartin` identified as the
/// shared surface across three official acceptance failures (`0fc156c3`
/// accepted_pairs=1, `b51f5fdb` accepted_pairs=0) and demoted behind an opt-in
/// flag of their own. Until this commit it had no kill switch at all, so it
/// could be neither A/B'd locally nor disabled during triage. One binary can
/// now price the chain (candidate legs `=0`, control legs unset) and a ranked
/// run that returns low `accepted_pairs` has a one-variable retry.
///
/// Read once into a `let`; the only process state is this environment read.
private let gemma4GlueChainEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_GLUE_CHAIN"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// GLUE-001: fused decode-plane layer glue. Three single-dispatch kernels
/// replace the strictly SERIAL RMSNorm/add chains between the layer's matmuls
/// at the exact ranked decode geometry ([8, 1, 2816] bfloat16):
///
///   1. `dualPreNorm` — `preFeedforwardLayernorm(out)` and
///      `preFeedforwardLayernorm2(out)` norm the SAME tensor; one reduction
///      feeds both weight applications (2 kernels -> 1).
///   2. `tail` — `postFFLN1(h1) + postFFLN2(h2)` -> `postFFLN` -> `+ residual`
///      (5 kernels -> 1).
///   3. `normResidual` — `residual + postAttentionLayernorm(attnOut)`
///      (2 kernels -> 1).
///
/// Unlike the (default-off) fused router above, every op fused here sits on
/// the layer's DEPENDENT chain — none of them can hide under the concurrent
/// encoder's overlap with the expert branch — so dispatch deletion shortens
/// the critical path rather than deleting already-hidden work.
///
/// Numerics reproduce the stock kernels verbatim: the rms reduction is the
/// exact `rms_single_row` tree at 704 threads x N_READS=4 (float square
/// accumulation in thread-read order, simd_sum, 32-slot cross-simd combine,
/// `metal::precise::rsqrt(acc/2816 + 1e-6)`), the output cast order is the
/// stock `w * static_cast<T>(x * inv)`, and the adds are single bfloat adds
/// exactly as the binary kernel performs them.
private enum Gemma4FusedLayerGlue {
    /// Kill switch: DARKBLOOM_GEMMA4_FUSED_LAYER_GLUE=0 restores the stock
    /// per-op chain. Default ON.
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_FUSED_LAYER_GLUE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let rows = 8
    private static let axis = 2816
    private static let eps: Float = 1e-6
    private static let nReads = 4
    private static let tgThreads = 704  // 2816 / 4, exactly rms_single_row's shape

    /// Shared reduction preamble: the exact rms_single_row tree at 704x4.
    /// `PREFIX` names the array to reduce; `SLOT` the shared slot written.
    private static func rmsReduce(_ src: String, into slot: String) -> String {
        """
            {
                float acc = 0;
                for (int i = 0; i < 4; i++) {
                    float xi = (float)\(src)[base + i];
                    acc += xi * xi;
                }
                acc = simd_sum(acc);
                if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    acc = simd_sum(local_sums[simd_lane_id]);
                    if (simd_lane_id == 0) {
                        \(slot) = metal::precise::rsqrt(acc / 2816.0f + 1e-06f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        """
    }

    private static let normResidualKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_norm_residual_2816_bf16_v1",
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

    private static let dualPreNormKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_dual_prenorm_2816_bf16_v1",
        inputNames: ["x", "w1", "w2"],
        outputNames: ["out1", "out2"],
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
                const T nx = static_cast<T>((float)x[base + i] * inv);
                out1[base + i] = w1[wbase + i] * nx;
                out2[base + i] = w2[wbase + i] * nx;
            }
        """,
        ensureRowContiguous: true
    )

    private static let tailKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_tail_2816_bf16_v2",
        inputNames: ["a", "b", "res", "w1", "w2", "w3", "s"],
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
        """,
        ensureRowContiguous: true
    )

    private static func admits(_ x: MLXArray, weight: MLXArray, eps: Float) -> Bool {
        enabled
            && eps == Self.eps
            && x.ndim == 3
            && x.dim(0) == rows && x.dim(1) == 1 && x.dim(2) == axis
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
        return normResidualKernel(
            [x, residual, weight],
            template: [("T", x.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[rows, 1, axis]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    static func dualPreNorm(
        x: MLXArray, w1: MLXArray, w2: MLXArray, eps: Float
    ) -> (MLXArray, MLXArray)? {
        guard admits(x, weight: w1, eps: eps),
            w2.ndim == 1, w2.dim(0) == axis, w2.dtype == .bfloat16
        else { return nil }
        CBv2EngageMark.once("glue-dual-prenorm")
        let outs = dualPreNormKernel(
            [x, w1, w2],
            template: [("T", x.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[rows, 1, axis], [rows, 1, axis]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outs[0], outs[1])
    }

    /// GLUE-003: tail variant that ALSO emits the NEXT layer's input norm.
    /// The threadgroup already holds the finished output row in registers, so
    /// the next layer's `inputLayernorm(out)` costs one more in-kernel
    /// reduction instead of a standalone serial dispatch plus a full re-read
    /// of the row. The normed output replicates the stock rms sequence over
    /// the exact stored bf16 output values.
    private static let tailChainKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_glue_tail_chain_2816_bf16_v1",
        inputNames: ["a", "b", "res", "w1", "w2", "w3", "s", "wn"],
        outputNames: ["out", "normed"],
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
        """,
        ensureRowContiguous: true
    )

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
        let outs = tailChainKernel(
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
        return tailKernel(
            [mlpOut, expertOut, residual, w1, w2, w3, layerScalar],
            template: [("T", mlpOut.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[rows, 1, axis]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}

/// Expert router. Norms `x` with a learnable scale, projects to expert
/// scores, and returns top-K (indices, weights) where weights are
/// softmax-normalized and scaled by a per-expert scalar.
private class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    let topK: Int
    let eps: Float
    let rootSize: Float
    let kth: Int
    private var cachedEffectiveScale: MLXArray?

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
        let expertScores = proj(normed)

        // ROUTE-001: single-dispatch byte-identical replacement of the chain
        // below for the B=8 decode geometry. Every other geometry, dtype, or
        // the kill switch falls through to the established chain.
        if let fused = Gemma4FusedRouterTop8.apply(
            expertScores: expertScores, perExpertScale: perExpertScale, topK: topK)
        {
            Gemma4RouterProbe.recorder?(expertScores, fused.indices)
            return (fused.indices, fused.weights)
        }

        var topKIndices = MLX.argPartition(expertScores, kth: kth, axis: -1)
        topKIndices = topKIndices[.ellipsis, kth...]

        var topKWeights = MLX.takeAlong(expertScores, topKIndices, axis: -1)
        topKWeights = MLX.softmax(topKWeights, axis: -1, precise: true)
        topKWeights = topKWeights * perExpertScale[topKIndices]

        // Diagnostic-only observability (nil in production; see
        // `Gemma4RouterProbe`). Recording the PRE-selection scores and the
        // selection itself, never altering either.
        Gemma4RouterProbe.recorder?(expertScores, topKIndices)

        return (topKIndices, topKWeights)
    }
}

/// Sparse MoE feed-forward block. Wraps `SwitchGLU` with GeGLU activation.
private class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU
    let fuseWeightedUnsort: Bool

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
        isExpertPrefill: Bool
    ) -> MLXArray {
        // Flatten [B, S, H] and always enter SwitchGLU's combined API. It
        // selects direct sorted reduction only for the exact production
        // contract; every other case performs the established unsort + sum.
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = topKIndices.dim(-1)
        let y = switchGLU.callAndWeightedReduce(
            x.reshaped(B * S, H),
            topKIndices.reshaped(B * S, K),
            weights: topKWeights.reshaped(B * S, K),
            fuseSortedReduction: fuseWeightedUnsort,
            // Ordinary/direct VLM and CBv2 prompt entry points may engage.
            // Rectangular MTP verification explicitly passes false.
            isProductionPrefill: isExpertPrefill)
        return y.reshaped(B, S, H)
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

    /// DMLP-001: route only the pinned batch-eight/decode-one affine-8 dense
    /// MLP geometries through the exact quad-stream kernel's tight grid.
    /// Everything else, including prefill and any strided input, keeps the
    /// original layer call.
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // DMLP-002: one exact activation-sum prepass feeds both fallback
        // projections. If either projection is not the pinned affine8 cell,
        // the candidate arrays remain unevaluated and stock takes over.
        let activationSums = CBv2DenseMLPQMVV1.activationSums(for: x)
        return denseProjection(
            downProj,
            gemma4SafeGeluProduct(
                denseProjection(gateProj, x, activationSums: activationSums),
                denseProjection(upProj, x, activationSums: activationSums)))
    }
}

// MARK: - Decoder Layer

/// Gemma 4 decoder layer. Combines `Gemma4Attention` with an MLP (or MoE)
/// block, the per-layer-input (PLE) path, and residual / layer-scalar
/// plumbing. Consumed by `Gemma4TextModelInner` and by the Gemma 4 MTP
/// drafter's trunk in `Gemma4MTP`; not intended as a user-facing
/// composable layer.
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

    // MoE-only modules (26B-A4B); nil on dense variants.
    @ModuleInfo(key: "router") fileprivate var router: Gemma4Router?
    @ModuleInfo(key: "experts") fileprivate var experts: Gemma4Experts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm?

    // Per-layer input (PLE) gating
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    // Per-layer scalar
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
            self._router.wrappedValue = Gemma4Router(config)
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
        nextInputLayernormWeight: MLXArray? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4.PositionOffset) {
        // Prompt-path narrowing (CBv2 only): attention and every K/V write
        // still cover the full chunk; only the token-local work AFTER
        // attention is restricted to the trailing rows CBv2 actually reads.
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

        // GLUE-003 consumption: the previous layer's fused tail already
        // produced this layer's input norm. Pointer identity on the source
        // guarantees the normed tensor was computed from exactly this input.
        let h: MLXArray
        if let chain = glueChain, let pending = chain.pending,
            pending.source === x
        {
            chain.pending = nil
            h = pending.normed
        } else {
            glueChain?.pending = nil
            h = inputLayernorm(x)
        }
        let (attnOut, kvPair, attnPositionOffset) = selfAttn(
            h, mask: mask, cache: cache, sharedKV: sharedKV, positionOffset: positionOffset,
            v2SharedSource: v2SharedSource, outputStart: outputStart,
            useLastQueryPrefill: useLastQueryPrefill)
        var out: MLXArray
        if let fusedOut = Gemma4FusedLayerGlue.normResidual(
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
        // GLUE-001 fuses the whole post-branch tail (postFFLN1 + postFFLN2 +
        // sum + postFFLN + residual) into one dispatch; when it engages, the
        // common tail below must not run again.
        var tailApplied = false
        // Decode GLUE-002 also folds the terminal layer-scalar multiply. The
        // prefill tail deliberately stops before it, preserving the stock
        // materialization boundary and applying the scalar below.
        var scalarFolded = false

        if isMoE,
            let router,
            let experts,
            let postFeedforwardLayernorm1,
            let preFeedforwardLayernorm2,
            let postFeedforwardLayernorm2
        {
            // Dense + sparse branches in parallel, summed into one residual.
            let (topKIndices, topKWeights) = router(out)

            let h1Raw: MLXArray
            let h2Raw: MLXArray
            if let (n1, n2) = Gemma4FusedLayerGlue.dualPreNorm(
                x: out,
                w1: preFeedforwardLayernorm.weight,
                w2: preFeedforwardLayernorm2.weight,
                eps: config.rmsNormEps)
            {
                h1Raw = mlp(n1)
                h2Raw = experts(
                    n2,
                    topKIndices: topKIndices,
                    topKWeights: topKWeights,
                    isExpertPrefill: isExpertPrefill)
            } else if let (n1, n2) = Gemma4PrefillGlueV1.dualPreNorm(
                x: out,
                w1: preFeedforwardLayernorm.weight,
                w2: preFeedforwardLayernorm2.weight,
                eps: config.rmsNormEps)
            {
                h1Raw = mlp(n1)
                h2Raw = experts(
                    n2,
                    topKIndices: topKIndices,
                    topKWeights: topKWeights,
                    isExpertPrefill: isExpertPrefill)
            } else {
                h1Raw = mlp(preFeedforwardLayernorm(out))
                h2Raw = experts(
                    preFeedforwardLayernorm2(out),
                    topKIndices: topKIndices,
                    topKWeights: topKWeights,
                    isExpertPrefill: isExpertPrefill)
            }

            // The scalar fold is only valid when nothing sits between the
            // tail and the layer-scalar multiply (PLE absent on this model).
            let canFoldScalar =
                perLayerInputGate == nil || activePerLayerInput == nil
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
                chain.pending = (source: chained.out, normed: chained.normedNext)
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
        } else {
            out = preFeedforwardLayernorm(out)
            out = mlp(out)
        }

        if !tailApplied {
            out = postFeedforwardLayernorm(out)
            out = residual2 + out
        }

        // PLE gating
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let perLayerInput = activePerLayerInput
        {
            let residual3 = out
            var g = gate(out)
            g = gemma4SafeGeluApproximate(g)
            g = g * perLayerInput
            g = proj(g)
            g = norm(g)
            out = residual3 + g
        }

        if !scalarFolded {
            out = out * layerScalar
        }

        return (out, kvPair, attnPositionOffset)
    }
}

// MARK: - Text Model

/// Inner Gemma 4 trunk: embeddings + per-layer-input (PLE) + 35 decoder
/// layers + final norm. Public so the Gemma 4 MTP drafter in
/// `Gemma4MTP` can build its own 4-layer kv-shared trunk; not
/// intended as a user-facing model — use `Gemma4TextModel` for
/// standalone inference.
public class Gemma4TextModelInner: Module {
    let config: Gemma4TextConfiguration
    let embedScale: Float
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [Gemma4DecoderLayer]
    @ModuleInfo public var norm: RMSNorm

    // Per-layer embeddings (PLE)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") fileprivate var perLayerModelProjection: ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    // KV sharing mapping: for each layer, which earlier layer provides KVs
    let previousKvs: [Int]
    let firstKvSharedLayerIdx: Int

    /// Index of the last non-shared full-attention layer (-1 if none).
    /// Used by the shared-KV capture hook for the MTP drafter.
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

        // PLE
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

        // Build KV-sharing map
        self.firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        var kvMap = Array(0 ..< config.numHiddenLayers)
        if config.numKvSharedLayers > 0 {
            // Find the last non-shared layer of each type
            var lastByType = [String: Int]()
            for i in 0 ..< firstKvSharedLayerIdx {
                lastByType[config.layerTypes[i]] = i
            }
            // Shared layers reference the last non-shared layer of the same type
            for j in firstKvSharedLayerIdx ..< config.numHiddenLayers {
                if let prev = lastByType[config.layerTypes[j]] {
                    kvMap[j] = prev
                }
            }
        }
        self.previousKvs = kvMap

        // Capture indices for MTP drafter: the last layer of each type that
        // still has its own K/V (not shared from an earlier layer).
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
        // Callers may hand rank-1 token ids ([N] on cache-reuse turns, e.g.
        // the deprecated TokenIterator API) — the deleted inline VLM twin
        // normalized the whole multimodal tuple before any dimension read.
        // Expand tokens, supplied embeddings, and the visual mask together
        // so they continue to agree on [B, L].
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

    /// CBv2 prompt-forward entry point. Keeping the scheduled-prefill
    /// specializations behind their own entry point means legacy forwards,
    /// compiled [B, 1] decode, and MTP verification can never reach them.
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

    /// Variant that ALSO returns the pre-norm last-layer hidden state.
    /// The MTP drafter's `pre_projection` was trained against the pre-norm
    /// hidden (HF captures `hidden_states` at the decoder-layer boundary,
    /// BEFORE `model.norm`); the LM head consumes the post-norm hidden.
    /// The non-MTP path goes through `callAsFunction`.
    public func callCapturingPreNorm(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        captureHook: ((Int, (MLXArray, MLXArray)) -> Void)? = nil
    ) -> (postNorm: MLXArray, preNorm: MLXArray) {
        // Same rank-1 defense as `callAsFunction`: token ids may arrive as
        // [N] on cache-reuse turns; forwardTrunk assumes [B, L].
        let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        let r = forwardTrunk(
            inputs, cache: cache, captureHook: captureHook, capturePreNorm: true)
        return (r.postNorm, r.preNorm!)
    }

    /// DFlash target-hidden capture (2026-08-25, gemma4-dflash-real-loader
    /// lane). The z-lab DFlash drafter conditions on the CONCATENATION of the
    /// target's post-layer hidden states at `dflash_config.target_layer_ids`,
    /// so the capture has to happen inside the one trunk pass the verify
    /// forward already runs — a second forward would both double the target
    /// cost and (worse) advance the KV cache a second time.
    ///
    /// This is the only DFlash-shaped change to the trunk: an optional
    /// observer over the layer outputs the trunk already computed, plus the
    /// `forceArrayMask` pass-through below. Nothing about the non-DFlash
    /// numerics moves.
    func callCapturingDFlashHiddenStates(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        targetLayerIds: [Int],
        forceArrayMask: Bool = false
    ) throws -> (postNorm: MLXArray, hiddenStates: [MLXArray]) {
        try DFlashTargetValidation.validateTargetLayerIds(
            targetLayerIds, layerCount: layers.count)
        // Same rank-1 defense as `callAsFunction`.
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
        forceArrayMask requestedArrayMask: Bool = false
    ) -> (postNorm: MLXArray, preNorm: MLXArray?) {
        // Shape queries cross the Swift/C boundary. Cache the two immutable
        // input dimensions once rather than paying for them at every ladder
        // policy check while the host is building the decode graph.
        let inputBatchSize = inputs.dim(0)
        let inputLength = inputs.dim(1)

        // Vision prefill (mirrors the inline VLM twin `TextModel.callAsFunction`):
        // `inputEmbedding` — the scaled text embeddings with image soft-token
        // embeddings spliced at placeholder positions — replaces the trunk's
        // own lookup; token ids still feed the per-layer embeddings (PLE)
        // below. nil keeps the text path byte-identical.
        var h: MLXArray
        if let inputEmbedding {
            h = inputEmbedding.ndim == 2 ? inputEmbedding.expandedDimensions(axis: 0) : inputEmbedding
        } else {
            h = embedTokens(inputs) * embedScale
        }

        // Compute per-layer inputs (PLE)
        var perLayerInputs: [MLXArray?]
        if hiddenSizePerLayerInput > 0,
            let embedPerLayer = embedTokensPerLayer,
            let modelProj = perLayerModelProjection,
            let projNorm = perLayerProjectionNorm
        {
            // Token-based PLE
            let tokenPLE =
                embedPerLayer(inputs)
                * Float(config.hiddenSizePerLayerInput).squareRoot()

            // [B, L, numLayers * hiddenSizePerLayerInput] -> [B, L, numLayers, hiddenSizePerLayerInput]
            let reshapedTokenPLE = tokenPLE.reshaped(
                tokenPLE.dim(0), tokenPLE.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)

            // Model projection PLE
            let modelPLE = modelProj(h).reshaped(
                h.dim(0), h.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)
            let normedModelPLE = projNorm(modelPLE)

            // Combine: (model_proj + token_embed) * 2^{-0.5}
            let perLayerInputScale = pow(Float(2.0), -0.5)
            let combined = (normedModelPLE + reshapedTokenPLE) * perLayerInputScale

            perLayerInputs = (0 ..< config.numHiddenLayers).map { i in
                combined[.ellipsis, i, 0...]
            }
        } else {
            perLayerInputs = Array(repeating: nil, count: config.numHiddenLayers)
        }

        // Extend cache array for shared layers (which get nil caches)
        var fullCache: [KVCache?]
        if let cache {
            fullCache = cache.map { Optional($0) }
            while fullCache.count < config.numHiddenLayers {
                fullCache.append(nil)
            }
        } else {
            fullCache = Array(repeating: nil, count: config.numHiddenLayers)
        }

        // ContinuousBatchingV2 detection: v2 layer caches own attention AND
        // masking, so the trunk builds no masks at all on that path (there is
        // no padding and no shared frontier to mask). In v2 mode every layer
        // (including KV-shared ones) has a cache object.
        let isCBv2 = fullCache.contains { ($0 as? (any CBv2AttendingLayerCache)) != nil }
        // All-contiguous banks expose one position chain. Snapshot it before
        // the first layer advances the chain, then reuse that same lazy array
        // for every Q/K RoPE call in this forward.
        let unifiedCBv2PositionOffset: Gemma4.PositionOffset? = {
            guard isCBv2 else { return nil }
            for case let entry? in fullCache {
                if let offsets = (entry as? CBv2LayerCache)?.unifiedPositionOffsets {
                    return .batch(offsets + 0)
                }
            }
            return nil
        }()

        // Build masks: one per attention type (legacy path only). "vision"
        // overlays bidirectional access within visual spans. "all" preserves
        // Gemma4's fully bidirectional prefill by symmetrizing both global and
        // sliding causal masks. Either mode needs a materialized array; ordinary
        // text and single-token decode retain the symbolic causal fast path.
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

        // Forward through layers, tracking intermediate KV pairs for sharing
        var intermediates = [(kv: (MLXArray, MLXArray)?, positionOffset: Gemma4.PositionOffset?)](
            repeating: (nil, nil), count: config.numHiddenLayers)

        // GLUE-003: one chain box per forward; layer L's fused tail hands
        // layer L+1 its input norm through it. MMA-064': nil when
        // DARKBLOOM_GEMMA4_GLUE_CHAIN is off, which drops every layer onto the
        // GLUE-002 parent-only tail. Default ON — unset env is the old code.
        let glueChain = gemma4GlueChainEnabled ? Gemma4GlueChainBox() : nil
        for (idx, layer) in layers.enumerated() {
            let prevIdx = previousKvs[idx]
            let sharedKV = intermediates[prevIdx].kv
            let sharedPositionOffset = intermediates[prevIdx].positionOffset

            // CBv2: KV-shared layers attend by borrowing the SOURCE layer's
            // cache object (attendBorrowing) instead of consuming raw K/V
            // tensors. Thread the source cache alongside the source's
            // captured (pre-update) position offsets.
            let v2SharedSource: (any CBv2AttendingLayerCache)? =
                isCBv2 && prevIdx != idx
                ? fullCache[prevIdx] as? (any CBv2AttendingLayerCache) : nil

            let mask = maskByType[layer.layerType]
            // Prompt-path specializations, final layer only. Every earlier
            // layer runs the full chunk unchanged because later positions'
            // K/V depend on it.
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
                // The retained pair is a CBv2 production-prefill optimization.
                // Ordinary direct forwards keep the established reduction;
                // enabling it there regressed the raw-prefill control without
                // affecting the serving path selected by the benchmark.
                isExpertPrefill: gemma4AllowsWeightedExpertUnsort(
                    schedulePrefill: schedulePrefill),
                glueChain: glueChain,
                nextInputLayernormWeight: idx + 1 < layers.count
                    ? layers[idx + 1].inputLayernorm.weight : nil
            )
            h = out
            intermediates[idx] = (kvPair, positionOffset)
            captureHook?(idx, kvPair)
            dFlashHiddenCapture?.capture(h, layer: idx)

            // `layer` returns the recombined dense+sparse result. Submitting
            // only here starts the completed prefix early without serializing
            // those independent per-layer branches or changing any math.
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
                interval: gemma4PrefillChunkEvalLayers)
            {
                asyncEval(h)
                CBv2StepProfiler.recordEvent("v2.gemma4.prefill.chunk_eval")
            }
        }

        let postNorm = norm(h)
        return (postNorm, capturePreNorm ? h : nil)
    }
}

// MARK: - Bidirectional vision attention overlay (mirror of the VLM twin)

/// Per-token block id for vision spans: each contiguous run of vision tokens
/// shares an id, non-vision tokens get -1. Exact mirror of
/// `gemma4VisionBlockIds` in Libraries/MLXVLM/Models/Gemma4.swift (Python
/// `_block_sequence_ids_for_mask`).
private func gemma4TextVisionBlockIds(_ isVision: MLXArray) -> MLXArray {
    let length = isVision.dim(1)
    let leading = MLXArray.zeros([isVision.dim(0), 1], dtype: .bool)
    let prev = concatenated([leading, isVision[0..., ..<(length - 1)]], axis: 1)
    let starts = logicalAnd(isVision, logicalNot(prev))
    let groupIds = cumsum(starts.asType(.int32), axis: 1) - 1
    return MLX.where(isVision, groupIds, MLXArray(Int32(-1)))
}

/// Overlay blockwise bidirectional attention for vision-token spans onto a
/// boolean causal mask (true = attend): tokens in the same image block
/// attend each other in BOTH directions. Exact mirror of
/// `gemma4BidirectionalVisionMask` (Python
/// `_apply_blockwise_bidirectional_overlay`).
private func gemma4TextBidirectionalVisionMask(
    _ baseMask: MLXArray, isVision: MLXArray
) -> MLXArray {
    let blockIds = gemma4TextVisionBlockIds(isVision)
    let qBlocks = expandedDimensions(blockIds, axis: -1)  // [B, L, 1]
    let kBlocks = expandedDimensions(blockIds, axis: -2)  // [B, 1, L]
    var sameBlock = logicalAnd(qBlocks .!= MLXArray(Int32(-1)), qBlocks .== kBlocks)  // [B, L, L]
    // Cached (chunked) prefill: `baseMask` covers ALL key columns
    // (`offset + L`) while `sameBlock` only describes the current window's
    // L columns. Left-pad with `false` so the overlay lands on the LAST L
    // key columns — cached keys stay causal. Callers must never split an
    // image block across the cache boundary (the CBv2 scheduler snaps
    // chunks to block edges; whole-prompt prefill has offset 0), or the
    // overlay could not see the cached half of the block (PR#63 review).
    let L = isVision.dim(1)
    let keyColumns = baseMask.dim(-1)
    if keyColumns > L {
        let pad = MLXArray.zeros([sameBlock.dim(0), L, keyColumns - L], dtype: .bool)
        sameBlock = concatenated([pad, sameBlock], axis: -1)  // [B, L, offset+L]
    }
    return logicalOr(baseMask, expandedDimensions(sameBlock, axis: 1))  // -> [B, 1, L, offset+L]
}

/// If `mode` carries a boolean array mask, overlay the vision bidirectional
/// attention; pass other modes (`.causal`, `.none`) through unchanged.
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

/// Symmetrize the materialized causal/windowed mask for
/// `use_bidirectional_attention == "all"`. Global layers become fully
/// bidirectional; sliding layers remain bounded by their symmetric window.
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
        // Cached columns already describe the exact visible prefix for every
        // current query. Only the trailing current-query square has a valid
        // transpose; keep the rectangular prefix unchanged.
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

    /// Read-only accessor for the underlying text configuration. Needed by
    /// `Gemma4AssistantDraftModel` for its bind-time compatibility checks.
    public var configuration: Gemma4TextConfiguration { config }

    /// Process request and resolved immutable eligibility for production
    /// benchmark provenance. A truthy request stays ineffective unless the
    /// checkpoint is the exact supported Gemma 4 geometry *and* carries the
    /// safe expert-QMM quantization contract, because weighted unsort is only
    /// a win as half of the coupled weighted + safe-R1 pair.
    public var weightedExpertUnsortRequested: Bool { gemma4FusedWeightedUnsortRequested }
    public var weightedExpertUnsortEffective: Bool { fuseWeightedUnsort }

    /// Whether this checkpoint satisfies everything the safe Gemma 4
    /// expert-QMM selector can decide from configuration: the exact expert
    /// topology and the 4-bit / group-size-64 quantization contract. The
    /// runtime feature request, AOT capability, and NAX precedence are
    /// reported separately by MLX. Identical to the predicate gating weighted
    /// unsort, so the pair can never report or run half-applied.
    public var expertQMMGeometryEligible: Bool {
        gemma4SupportsCoupledExpertOptimizations(config)
    }

    /// Canonical decoder-layer roots. Wrappers whose existing LoRA adapter
    /// keys are decoder-relative use these roots without owning another tower.
    public var decoderLayers: [Module] { model.layers }

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Gemma4TextConfiguration) {
        let fuseWeightedUnsort = gemma4ShouldFuseWeightedUnsort(config)
        self.config = config
        self.vocabularySize = config.vocabSize
        // Per-layer KV head counts must agree with `Gemma4Attention.init`:
        // full layers use `num_global_key_value_heads` when present (whether
        // or not k_eq_v is enabled), sliding layers the sliding count.
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
        // Fully bidirectional prompt states require whole-prompt visibility.
        // Returning the complete prompt lets TokenIterator evaluate it once.
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
        let hidden = model(inputs, cache: cache)
        return applyLMHead(hidden)
    }

    /// Vision forward (mirror of the VLM wrapper's
    /// `languageModel(tokens, inputEmbedding:cache:imageTokenMask:)` call):
    /// `inputEmbedding` replaces the trunk's own embedding lookup (spliced
    /// image soft tokens; token ids still feed the PLE side inputs), and
    /// `imageTokenMask` ([B, L] bool) enables the blockwise bidirectional
    /// overlay on the LEGACY mask path (v2 layer caches own their masks and
    /// ignore it). Both nil ⇒ byte-identical to `callAsFunction(_:cache:)`.
    public func callAsFunction(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        imageTokenMask: MLXArray? = nil
    ) -> MLXArray {
        applyLMHead(
            model(
                inputs, cache: cache, inputEmbedding: inputEmbedding,
                imageTokenMask: imageTokenMask))
    }

    /// MMA-003: serve all eight cohort rows from one matrix-unit pass over the
    /// tied affine-4 vocabulary plane. The implementation fails closed for
    /// every non-production geometry, allowing the promoted tight-grid QMV
    /// below to remain the exact fallback.
    @inline(__always)
    private func tiedLMHeadMMA(_ hidden: MLXArray) -> MLXArray? {
        guard lmHead == nil,
            let quantized = model.embedTokens as? QuantizedEmbedding,
            quantized.mode == .affine,
            let mma = Gemma4MMAQuantizedGEMV.apply(
                x: hidden,
                w: quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                groupSize: quantized.groupSize,
                bits: quantized.bits)
        else { return nil }
        return mma.reshaped(Array(hidden.shape.dropLast()) + [mma.dim(-1)])
    }

    /// Apply the LM head (tied embedding or explicit `lm_head`) plus the
    /// configured final-logit softcap. Pure function of the post-norm hidden.
    /// LMH-001: tight-grid dispatch for the tied lm_head ordinary QMV.
    ///
    /// The vendored host launches ordinary QMV with an x grid extent of M = 8
    /// (`backend/metal/quantized.cpp`), while the promoted large-N tier claims
    /// four cohort rows per threadgroup and returns from the rest. On the tied
    /// head that is 262144 threadgroups of which 196608 exist only to hit that
    /// early return. `CBv2TiedLMHeadQMVV1` runs the same computation from a
    /// kernel whose own x extent is two, so only the groups that were already
    /// doing the work are launched. Returns `nil` unless every pin holds, and
    /// the caller then keeps the stock path.
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

    func applyLMHead(_ hidden: MLXArray) -> MLXArray {
        var out: MLXArray
        if let lmHead {
            out = lmHead(hidden)
        } else if let mma = tiedLMHeadMMA(hidden) {
            out = mma
        } else if let tight = tiedLMHeadTightGrid(hidden) {
            out = tight
        } else {
            out = model.embedTokens.asLinear(hidden)
        }
        // The VLM omission profile uses zero to represent the former optional
        // softcap's nil/disabled state.
        if config.finalLogitSoftcapping > 0 {
            out = gemma4CompiledLogitSoftcap(
                out, MLXArray(config.finalLogitSoftcapping))
        }
        return out
    }

    /// The LM head WITHOUT the configured final-logit softcap.
    ///
    /// The DFlash drafter borrows the target's LM head but applies its OWN
    /// `final_logit_softcapping` (from the DRAFTER's config.json) to the
    /// result — see `DFlashDraftModel.callAsFunction`. Handing it
    /// `applyLMHead` would softcap twice, with the target's constant. Only
    /// `logitsForDFlashHidden` calls this; the target's own logits keep
    /// going through `applyLMHead`.
    func applyRawLMHead(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        if let tight = tiedLMHeadTightGrid(hidden) {
            return tight
        }
        return model.embedTokens.asLinear(hidden)
    }

    /// Compute the scaled input embedding for `tokens`, matching what the
    /// inner trunk does in its first step (`embedTokens(inputs) * embedScale`).
    /// Used by `Gemma4AssistantDraftModel` as the "target embedding" input
    /// when building its drafter-step input `[target_embed(last_token), last_hidden]`.
    public func embedTokensForDrafter(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens) * Float(config.hiddenSize).squareRoot()
    }

    /// Width-probe diagnostic forward (exactness round three): full logits
    /// plus the per-layer K/V capture hook — the layer-by-layer seam the
    /// operator-only `width-probe` verb bit-compares across forward widths.
    /// Identical compute to the plain forward (`applyLMHead` over the same
    /// trunk); the hook only observes the per-layer K/V pairs the trunk
    /// already produced.
    public func widthProbeForward(
        _ inputs: MLXArray,
        cache: [KVCache],
        captureHook: @escaping (Int, (MLXArray, MLXArray)) -> Void
    ) -> MLXArray {
        applyLMHead(model(inputs, cache: cache, captureHook: captureHook))
    }

    /// Internal helper for Gemma4CaptureHookTests. Not part of the public API.
    internal func _testCallInner(
        _ inputs: MLXArray,
        cache: [KVCache],
        captureHook: ((Int, (MLXArray, MLXArray)) -> Void)? = nil
    ) -> MLXArray {
        model(inputs, cache: cache, captureHook: captureHook)
    }

    /// Parse the layer index out of a weight key like
    /// `"model.layers.15.self_attn.k_proj.weight"`. Returns nil if the key
    /// doesn't match the expected `...layers.<N>...` pattern.
    private func extractLayerIdx(from key: String) -> Int? {
        guard let layersRange = key.range(of: "layers.") else { return nil }
        let after = key[layersRange.upperBound...]
        let end = after.firstIndex(of: ".") ?? after.endIndex
        return Int(after[..<end])
    }


    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (k, v) in weights {
            // Skip vision/audio/rotary/quantization-range weights.
            if k.contains("self_attn.rotary_emb")
                || k.contains("input_max")
                || k.contains("input_min")
                || k.contains("output_max")
                || k.contains("output_min")
            {
                continue
            }

            // Skip k_proj/v_proj/k_norm/v_norm weights for layers that
            // borrow K/V from an earlier non-shared layer (num_kv_shared_layers
            // tail). Our `Gemma4Attention.init` doesn't allocate these modules
            // for shared-KV layers, so the checkpoint's copies would fail the
            // strict `update(parameters:verify:.all)` check.
            if let layerIdx = extractLayerIdx(from: k),
                config.layerUsesSharedKV(layerIdx: layerIdx),
                k.contains(".self_attn.k_proj.")
                    || k.contains(".self_attn.v_proj.")
                    || k.contains(".self_attn.k_norm.")
                    || k.contains(".self_attn.v_norm.")
            {
                continue
            }

            // Some 26B-A4B checkpoints ship one raw expert `gate_up_proj`
            // tensor plus `down_proj`. The ordinary SwitchGLU topology owns
            // split projections, so normalize the packed tensor here.
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
        return sanitized
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
    /// Per-layer attention structure for the CBv2 engine, derived purely
    /// from this configuration (invariant 11: model structure is data).
    /// Matches `Gemma4Attention.init` / `Gemma4TextModelInner.previousKvs`
    /// layer for layer.
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
    /// Per-layer CBv2 attention structure for this model (one entry per
    /// hidden layer, including the trailing KV-shared block).
    public var cbv2LayerKinds: [CBv2LayerKind] {
        config.cbv2LayerKinds
    }

    /// Effective layer interval for scheduled CBv2 prompt submissions.
    /// Zero means the optimization is disabled and the trunk has only its
    /// caller's final graph submission.
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

    /// Build the per-layer CBv2 attending caches for this model: one
    /// `CBv2AttendingLayerCache` per hidden layer (KV-shared layers get a
    /// cache object too — it owns no storage and serves `attendBorrowing`).
    ///
    /// The concrete layer-cache classes are owned by the CBv2 core runtime;
    /// `makeLayerCache` is the injection point (typically wrapping a
    /// `CBv2KVBackend`). This model file codes purely against the contract.
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

/// CBv2 consumes only the final prompt position, so the public
/// `LanguageModel` forward contract stays unchanged while the engine's
/// prompt path skips the vocabulary projection for discarded positions:
/// intermediate chunks project nothing, and the frontier chunk projects one
/// hidden row. Attention, multimodal span masks, positions, and every K/V
/// write still cover the full chunk.
extension Gemma4TextModel: CBv2LanguageModelPrefillForwardable {

    /// The Gemma trunk is shape-generic over `[B, L]`, and the CBv2
    /// attention dispatch handles a rectangular `B > 1, L > 1` prompt batch
    /// by attending each row against its OWN KV (the same per-row path a
    /// `[1, chunk]` call takes), so a packed row is bit-identical to running
    /// alone. The engine still requires the cache provider to vouch for row
    /// independence before it packs anything.
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
            // Small handle whose graph depends on the whole trunk — forcing
            // it commits every layer's K/V write for this chunk.
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            return applyLMHead(hidden[0..., -1, 0...])
        }
    }
}

// MARK: - ContinuousBatchingV2 multimodal (vision prefill)

/// The CBv2 engine's embedding-spliced prefill surface
/// (`CBv2SteppableLanguageModelAdapter` forwards through this). The v2
/// attention branch is reached exactly as for token forwards — the layer
/// caches detected in `cache` own attention AND masking (the engine binds
/// the span-mask context on them) — only the embedding source differs.
/// Positions, KV sharing, and dual RoPE are untouched.
extension Gemma4TextModel: CBv2EmbeddingForwardable {

    /// Only configs whose weights were trained with the bidirectional
    /// image-span attention may serve CBv2 vision spans — the same gate the
    /// legacy `imageTokenMask` path applies. Text-only Gemma4 configs
    /// (nil / non-`"vision"`) reject multimodal requests at submit instead
    /// of silently serving logits under masks the weights never saw
    /// (PR#63 review).
    public var supportsVisionSpanPrefill: Bool {
        config.useBidirectionalAttention == "vision"
    }

    /// `embed(tokens) * embedScale` — exactly the trunk's pre-layer-0 hidden
    /// state, the tensor the engine splices image embeddings into (the
    /// VLM wrapper's `prepare` computes the same product before
    /// `maskedScatter`).
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

/// The CBv2 engine's MTP verify surface (`CBv2MTPForwardable`): the plain
/// forward plus the PRE-norm last-decoder-layer hidden the Gemma-4 drafter
/// chains from, and the layer indices the engine snapshots for the drafter's
/// frozen KV. The logits side is numerically identical to
/// `callAsFunction(_:cache:)` — same trunk, same LM head, same softcap.
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
        let (postNorm, preNorm) = model.callCapturingPreNorm(tokens, cache: caches)
        return (applyLMHead(postNorm), preNorm)
    }
}
