//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Yukon executable-equivalent frontier sample: delordemm1 / e8f / de1.

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

    if let set = gemma4DecodeAsyncEvalLadderSet {
        return set.contains(layerIndex)
    }
    // Only the two EARLY boundaries pay. Submitting after layers 0 and 1
    // starts GPU work while the host is still building the remaining 28
    // layers; by layer 5 the device already has queued work, so the middle
    // cadence {5, 11, 17, 23, 27} adds no overlap and only fragments the
    // command buffer. Measured on an M1 Ultra at the ranked B=8 geometry,
    // paired and interleaved, tokens identical in every arm:
    //
    //     {} (no boundaries)          +0.17%   <- overlap genuinely lost
    //     {0,1}                       -0.52%   (64-step)  -0.51% (128-step)
    //     {0,1,11,23}                 -0.53%
    //     {0,1,5,11,17,23,27}          baseline (previous default)
    //     {0,1,5,11,17,23,27,29}      +0.13%
    //     A/A control                 -0.09%   <- the noise floor
    //
    // The empty-set row is the control that matters: this is not "fewer is
    // always better", it is "the early pair carries all of the overlap".
    switch layerIndex {
    case 0, 1, 2, 3:
        return true
    default:
        return false
    }
}

/// LOCAL EXPERIMENT ONLY. `DARKBLOOM_GEMMA4_DECODE_LADDER_SET` overrides the
/// shipped boundary list with a comma-separated set of layer indices, so the
/// geometry can be swept on one binary instead of one rebuild per candidate.
/// The ranked runner sets no environment, so an unset variable keeps the
/// shipped switch above verbatim and this is inert in a submission. An empty
/// value means "no boundaries", which is distinct from unset.
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

/// Submission cadence used when a prompt pass is long enough to run on the
/// blocked-query prefill path. The documented 18-layer serving default, the
/// zero kill switch and every explicit non-default tuning value are preserved;
/// only a pass already in that regime takes the narrower cadence.
///
/// `0` disables the specialization (the cadence falls back to the configured
/// value), which is the kill switch.
private let gemma4LongPrefillChunkEvalLayers: Int = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_CHUNK_EVAL_LONG"],
        let value = Int(raw), value >= 0
    else { return 3 }
    return value
}()

/// The query-row block width above which a prompt pass is blocked by
/// `CBv2AttentionV1` (`queryBlockSize`). A pass at or below it is not blocked
/// and keeps the configured cadence.
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

/// GELU-FUSE: the SAME body, compiled WITHOUT `shapeless`.
///
/// Shapeless tracing runs under a dynamic-broadcast regime, so every binary op
/// in the body contributes broadcast nodes that a shape-specialised trace omits
/// on equal shapes. The extra nodes push this expression past MLX's compile
/// fusion depth limit, and the closure is emitted as TWO Metal kernels — the
/// cube term and the rest — with a materialised intermediate between them. The
/// shape-specialised trace fits under the limit and emits one.
///
/// Admission is deliberately narrow (`geluFusionClaimsPinnedDecode`): a
/// shape-specialised compile adds a compiler-cache entry per distinct input
/// shape and that lookup is a linear scan, so admitting prefill — whose
/// sequence length changes per prompt — would grow the scan on the decode hot
/// path. Everything not pinned falls open to the shapeless closure above.
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

/// The pinned decode signatures of the two GELU-product sites: dense/PLE rows
/// `[8, 1, N]` and routed-expert rows `[64, 1, N]` / `[64, N]`, both operands
/// bfloat16 with identical shapes. Anything else keeps the shapeless path.
@inline(__always)
func geluFusionClaimsPinnedDecode(_ gate: MLXArray, _ up: MLXArray) -> Bool {
    guard gemma4ShapedGeluFuseEnabled,
        gate.dtype == .bfloat16, up.dtype == .bfloat16,
        gate.shape == up.shape
    else { return false }
    let s = gate.shape
    if s.count == 3, s[1] == 1, s[0] == 8 || s[0] == 64 { return true }
    if s.count == 2, s[0] == 64 { return true }
    return false
}

/// Kill switch for the shape-specialised fusion.
let gemma4ShapedGeluFuseEnabled: Bool =
    ProcessInfo.processInfo.environment["DARKBLOOM_GELU_SHAPED_FUSE"] != "0"

/// GELU-FUSE-PREFILL: the same one-kernel trace for the prefill rectangles,
/// under a hard cap on how many distinct shapes may ever be admitted.
///
/// The comment above states why prefill was excluded: a shape-specialised
/// compile costs one compiler-cache entry per distinct shape, the lookup is a
/// linear scan, and per-prompt sequence lengths would grow it without bound.
/// That is an argument about the entry *count*, so the cap answers it — at most
/// ``shapedGeluPrefillShapeCap`` rectangles are admitted for the process
/// lifetime and everything after falls open to the shapeless closure. The
/// decode signatures are matched first and never reach the set.
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

/// Route the pinned decode signatures and the capped prefill rectangles to the
/// one-kernel trace.
@inline(__always)
func gemma4GeluProduct(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    if geluFusionClaimsPinnedDecode(gate, up) {
        return gemma4SafeGeluProductShaped(gate, up)
    }
    // PROMPT-GLUE (pg1): prompt-width rectangles take the vector kernel; the
    // decode signatures were matched above and never reach it.
    if let product = Gemma4PromptGlueV1.geluProduct(gate: gate, up: up) {
        if Gemma4PromptGlueV1.xcheck {
            Gemma4PromptGlueV1.exhaustiveCheck(stock: gemma4SafeGeluProductShaped)
            Gemma4PromptGlueV1.report(
                product, reference: gemma4SafeGeluProductShaped(gate, up), site: "dense")
        }
        return product
    }
    if geluFusionClaimsPrefill(gate, up) {
        return gemma4SafeGeluProductShaped(gate, up)
    }
    return gemma4SafeGeluProduct(gate, up)
}

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

/// QKVNORM-PREFILL-001: the prefill twin of `gemma4_b8_qkv_rms_norm_v1`.
///
/// Same per-row arithmetic, three differences in the plumbing.
///
/// Rows are addressed from the shapes rather than the decode cohort's fixed
/// geometry, so `[B, chunk, H, D]` is admitted. V rides the K reduction (this
/// checkpoint is `attention_k_eq_v`, so `vRaw === kRaw`), which removes a
/// whole re-read of the key projection. And each row is written straight into
/// its `[B, H, L, D]` slot, so the `transposed(0, 2, 1, 3)` attention wants
/// costs nothing downstream.
///
/// The load-bearing detail is `RPT`. A 256-wide row is 64 threads at
/// `RMS_N_READS = 4`, and one 64-thread threadgroup per row leaves the
/// prefill plane far off saturation — that shape, not the dispatch count, is
/// why the stock three-norm chain is slow here. `RPT` rows share one 512-wide
/// threadgroup, and each row keeps its own 64 threads and its own two
/// simdgroups, so the reduction tree is the stock one row for row.
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

/// `(qNorm(q), kNorm(k), vNorm(k))` already in `[B, H, L, D]`. Returns `nil`
/// off the plane, including for every non-`k_eq_v` projection.
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
        // Below ~1024 rectangle tokens the wide threadgroup stops paying for
        // itself and the stock three-kernel chain is faster; measured, not
        // assumed. Decode also leaves through here, back to its own kernel.
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

/// QKV-NORM-ROPE-SLIDING: the sliding-layer prefill arm — three separate
/// banks (q, k, v; no K-eq-V on sliding layers) with the base-route RoPE
/// (`metal::exp2(-d * log2(theta))`) folded after the same explicit BF16
/// staging boundary. Structure extends the head-major twin; rotation is a
/// line-for-line transcription of rope.metal's base path.
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

/// PROMPT-GLUE (pg1) PACK-IN-NORMROPE: `gemma4_qkv_rms_norm_head_major_sliding_v1`
/// with the prompt q4 KV mirror pack folded in. The norm and RoPE text is
/// the incumbent's statement for statement (the early return is a predicate
/// so every thread reaches every barrier); each K row's rotated values and
/// each V row's normalized values are additionally staged, as the very T
/// words stored to `k_out`/`v_out`, in `final_vals`, and after one barrier
/// the row's first simdgroup runs `cbv2_kvq4g64_pack_pair_chunk_batch_d256_v1`'s
/// per-lane body over them: lane `l` owns values `8l..8l+7` (the pack
/// kernel's `per_lane` mapping), the serial min/max over those eight in the
/// same order, the same `simd_shuffle_xor` 1/2/4 combine across the same
/// eight lanes, `half` scale/bias, `rint` codes, and the same word layout at
/// the same `(plane, head, token) * 36` offset of the same per-row mirror
/// allocation. The mirror is therefore the pack kernel's output byte for
/// byte, computed without re-reading the 67 MB of K/V it packs.
private let gemma4QKVNormPrefillSlidingPackKernel = MLXFast.metalKernel(
    name: "gemma4_qkv_rms_norm_head_major_sliding_pack_pg1",
    inputNames: [
        "q", "k", "v", "q_weight", "k_weight",
        "position_offsets", "rope_log2_base",
    ],
    outputNames: ["q_out", "k_out", "v_out", "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"],
    source: """
        constexpr uint reads = 4;
        constexpr uint row_threads = D / reads;
        constexpr uint mirror_row_words = D / 8 + D / 64;
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
        threadgroup T final_vals[RPT][D];

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

        const bool valid = row < TOTAL_ROWS;
        uint mirror_b = 0;
        uint mirror_h = 0;
        uint mirror_l = 0;
        if (valid) {
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
            mirror_b = b;
            mirror_h = h;
            mirror_l = l;
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
        if (valid) {
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

        if (valid) {
            const float inverse_rms = inv_rms[slot];
            for (uint i = 0; i < reads; ++i) {
                const T normalized = T(float(input[i]) * inverse_rms);
                if (APPLY_ROPE && weighted) {
                    // The BF16 memory boundary the separate norm kernel's
                    // output store performed before stock RoPE read it.
                    rounded[slot][lid * reads + i] = T(weight[i] * normalized);
                } else {
                    const T stored = weighted ? weight[i] * normalized : T(1) * normalized;
                    output[i] = stored;
                    final_vals[slot][lid * reads + i] = stored;
                }
            }
        }
        if (APPLY_ROPE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (APPLY_ROPE && valid && weighted && lid * reads < D / 2) {
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
                const T o1 = static_cast<T>(rx1);
                const T o2 = static_cast<T>(rx2);
                output_row[pair] = o1;
                output_row[pair + D / 2] = o2;
                final_vals[slot][pair] = o1;
                final_vals[slot][pair + D / 2] = o2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The pack kernel's body over this row's stored K (rotated) or V
        // values: one 32-lane simdgroup per (plane, head, token) row.
        if (valid && row >= Q_ROWS && lid < 32) {
            const uint plane = weighted ? 0u : 1u;
            device uint32_t* mout;
            switch (mirror_b) {
                case 0: mout = m0; break;
                case 1: mout = m1; break;
                case 2: mout = m2; break;
                case 3: mout = m3; break;
                case 4: mout = m4; break;
                case 5: mout = m5; break;
                case 6: mout = m6; break;
                case 7: mout = m7; break;
                default: return;
            }
            mout += (plane * (HK * LK) + mirror_h * LK + mirror_l) * mirror_row_words;
            const threadgroup T* src = final_vals[slot];

            float vmin = 3.402823466e+38F;
            float vmax = -3.402823466e+38F;
            for (int i = 0; i < 8; ++i) {
                const float value = float(src[lane * 8 + i]);
                vmin = min(vmin, value);
                vmax = max(vmax, value);
            }
            for (uint m = 1; m < 8; m <<= 1) {
                vmin = min(vmin, simd_shuffle_xor(vmin, m));
                vmax = max(vmax, simd_shuffle_xor(vmax, m));
            }

            const half hs = half(max((vmax - vmin) / 15.0f, 1e-6f));
            const half hb = half(vmin);
            const float s = float(hs);
            const float b = float(hb);

            uint32_t word = 0u;
            for (int i = 0; i < 8; ++i) {
                const float qv = metal::rint((float(src[lane * 8 + i]) - b) / s);
                word |= uint32_t(clamp(qv, 0.0f, 15.0f)) << (4 * i);
            }
            mout[lane] = word;
            if (lane % 8 == 0) {
                mout[D / 8 + lane / 8] =
                    uint32_t(as_type<ushort>(hs)) | (uint32_t(as_type<ushort>(hb)) << 16);
            }
        }
    """,
    ensureRowContiguous: true
)

/// PROMPT-GLUE2 (pg2): `gemma4_qkv_rms_norm_head_major_sliding_pack_pg1` in
/// vector-access form. The norm, RoPE and pack arithmetic is the pg1 text
/// statement for statement -- the same float products in the same order,
/// the same `T` roundings at the same points, the same `simd_sum` tree over
/// the same per-lane operands, the same pack lane mapping and min/max
/// order. What changes is how the words move:
/// - each thread's four input words are one 8-byte load held in registers
///   for the reduction and the normalize (pg1 loads them twice, scalar);
/// - the four output words of a thread are one 8-byte store, for the
///   normalized banks and for each half of a rotated row;
/// - the row's cross-simdgroup combine is run by every simdgroup of the
///   row over the same 32 operands the first simdgroup combined (the
///   partials in lanes below the simdgroup count, the zero-fill's 0.0f in
///   a register elsewhere), so each holds the bit-identical inverse rms
///   without the zero-fill and publish-back barriers;
/// - one 8-byte-aligned staging array per row replaces `rounded` and
///   `final_vals`: it holds the rounded words before the rotation and the
///   rotated words after it (each pair is read and rewritten by the one
///   thread that owns it), and the pack reads the very words stored to
///   the output, as before. Q rows, which are never packed, stage only
///   for the rotation.
/// Every thread reaches every barrier (all row work is predicated), so the
/// row count need not divide the rows per threadgroup.
private let gemma4QKVNormPrefillSlidingPackKernelPg2 = MLXFast.metalKernel(
    name: "gemma4_qkv_rms_norm_head_major_sliding_pack_pg2",
    inputNames: [
        "q", "k", "v", "q_weight", "k_weight",
        "position_offsets", "rope_log2_base",
    ],
    outputNames: ["q_out", "k_out", "v_out", "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"],
    source: """
        constexpr uint reads = 4;
        constexpr uint row_threads = D / reads;
        constexpr uint row_simds = row_threads / 32;
        constexpr uint mirror_row_words = D / 8 + D / 64;
        typedef vec<T, 4> T4;
        const uint tid = thread_position_in_threadgroup.x;
        const uint slot = tid / row_threads;
        const uint lid = tid - slot * row_threads;
        const uint row = threadgroup_position_in_grid.x * RPT + slot;
        const uint lane = thread_index_in_simdgroup;
        const uint row_simd = lid / 32;

        threadgroup float partials[RPT][32];
        // PROMPT-GLUE2 (pg2): one 8-byte-aligned staging array per row serves
        // the RoPE pair exchange and the pack. It holds, as the very T words
        // stored to the output, the rounded values before the rotation and the
        // rotated values after it; each pair is read and rewritten by the one
        // thread that owns it.
        threadgroup T4 staged4[RPT][D / 4];
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

        const bool valid = row < TOTAL_ROWS;
        uint mirror_b = 0;
        uint mirror_h = 0;
        uint mirror_l = 0;
        if (valid) {
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
            mirror_b = b;
            mirror_h = h;
            mirror_l = l;
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
        threadgroup T* stg = reinterpret_cast<threadgroup T*>(&staged4[slot][0]);

        // The thread's four input words as one 8-byte load, kept in registers
        // for the reduction and the normalize (the incumbent re-reads them).
        T4 in4 = T4(0);
        if (valid) {
            in4 = *reinterpret_cast<const device T4*>(input);
        }
        float sum = 0.0f;
        if (valid) {
            for (uint i = 0; i < reads; ++i) {
                const float value = float(in4[i]);
                sum += value * value;
            }
        }
        sum = simd_sum(sum);

        if (lane == 0) partials[slot][row_simd] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Every simdgroup of the row combines the same 32 operands the
        // incumbent's first simdgroup combined -- the partials in lanes below
        // row_simds, the zero-fill's 0.0f in a register elsewhere -- so each
        // holds the bit-identical inverse rms with no publish-back barrier.
        sum = simd_sum(lane < row_simds ? partials[slot][lane] : 0.0f);
        const float inverse_rms = metal::precise::rsqrt(sum / float(D) + 1.0e-6f);

        if (valid) {
            if (APPLY_ROPE && weighted) {
                const T4 w4 = *reinterpret_cast<const device T4*>(weight);
                T4 r4;
                for (uint i = 0; i < reads; ++i) {
                    const T normalized = T(float(in4[i]) * inverse_rms);
                    // The BF16 memory boundary the separate norm kernel's
                    // output store performed before stock RoPE read it.
                    r4[i] = T(w4[i] * normalized);
                }
                staged4[slot][lid] = r4;
            } else {
                T4 s4;
                if (weighted) {
                    const T4 w4 = *reinterpret_cast<const device T4*>(weight);
                    for (uint i = 0; i < reads; ++i) {
                        const T normalized = T(float(in4[i]) * inverse_rms);
                        s4[i] = w4[i] * normalized;
                    }
                } else {
                    for (uint i = 0; i < reads; ++i) {
                        const T normalized = T(float(in4[i]) * inverse_rms);
                        s4[i] = T(1) * normalized;
                    }
                }
                *reinterpret_cast<device T4*>(output) = s4;
                if (row >= Q_ROWS) {
                    staged4[slot][lid] = s4;
                }
            }
        }
        if (APPLY_ROPE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (APPLY_ROPE && valid && weighted && lid * reads < D / 2) {
            const uint h_count = row < Q_ROWS ? HQ : HK;
            const uint l_count = row < Q_ROWS ? LQ : LK;
            const uint b = local_row / (l_count * h_count);
            const float L =
                static_cast<float>(row_position[slot] + position_offsets[b]);
            T4 o1;
            T4 o2;
            for (uint i = 0; i < reads; ++i) {
                const uint pair = lid * reads + i;
                const float d = static_cast<float>(pair) / static_cast<float>(D / 2);
                const float inv_freq = metal::exp2(-d * rope_log2_base[0]);
                const float theta = L * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const float x1 = static_cast<float>(stg[pair]);
                const float x2 = static_cast<float>(stg[pair + D / 2]);
                const float rx1 = x1 * costheta - x2 * sintheta;
                const float rx2 = x1 * sintheta + x2 * costheta;
                o1[i] = static_cast<T>(rx1);
                o2[i] = static_cast<T>(rx2);
            }
            *reinterpret_cast<device T4*>(output_row + lid * reads) = o1;
            *reinterpret_cast<device T4*>(output_row + lid * reads + D / 2) = o2;
            if (row >= Q_ROWS) {
                // K rows: the rotated words replace the pairs this thread alone
                // read, for the pack below.
                staged4[slot][lid] = o1;
                staged4[slot][lid + D / 8] = o2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The pack kernel's body over this row's stored K (rotated) or V
        // values: one 32-lane simdgroup per (plane, head, token) row.
        if (valid && row >= Q_ROWS && lid < 32) {
            const uint plane = weighted ? 0u : 1u;
            device uint32_t* mout;
            switch (mirror_b) {
                case 0: mout = m0; break;
                case 1: mout = m1; break;
                case 2: mout = m2; break;
                case 3: mout = m3; break;
                case 4: mout = m4; break;
                case 5: mout = m5; break;
                case 6: mout = m6; break;
                case 7: mout = m7; break;
                default: return;
            }
            mout += (plane * (HK * LK) + mirror_h * LK + mirror_l) * mirror_row_words;
            const threadgroup T* src = stg;

            float vmin = 3.402823466e+38F;
            float vmax = -3.402823466e+38F;
            for (int i = 0; i < 8; ++i) {
                const float value = float(src[lane * 8 + i]);
                vmin = min(vmin, value);
                vmax = max(vmax, value);
            }
            for (uint m = 1; m < 8; m <<= 1) {
                vmin = min(vmin, simd_shuffle_xor(vmin, m));
                vmax = max(vmax, simd_shuffle_xor(vmax, m));
            }

            const half hs = half(max((vmax - vmin) / 15.0f, 1e-6f));
            const half hb = half(vmin);
            const float s = float(hs);
            const float b = float(hb);

            uint32_t word = 0u;
            for (int i = 0; i < 8; ++i) {
                const float qv = metal::rint((float(src[lane * 8 + i]) - b) / s);
                word |= uint32_t(clamp(qv, 0.0f, 15.0f)) << (4 * i);
            }
            mout[lane] = word;
            if (lane % 8 == 0) {
                mout[D / 8 + lane / 8] =
                    uint32_t(as_type<ushort>(hs)) | (uint32_t(as_type<ushort>(hb)) << 16);
            }
        }
    """,
    ensureRowContiguous: true
)

/// Rows per threadgroup of the pg2 twin (384 threads at D = 256; any value
/// is exact).
private let gemma4QKVNormPrefillSlidingPackRowsPerGroupPg2 = 6

/// The sliding twin of `gemma4FusedQKVNormHeadMajor`: three banks, base-route
/// RoPE, same guards and same fallback discipline. Returns `nil` off the
/// plane (non-sliding geometry, small rectangles, guard failures) and the
/// caller keeps the stock three-norm chain plus separate RoPE.
/// PROMPT-GLUE (pg1): `packMirrors` asks for the pack-folded twin; when it
/// engages, `mirrors` carries one `[2, HK, LK, 36]` uint32 plane per batch
/// row (the q4 mirror the prompt commit would otherwise pack from `k`/`v`).
private func gemma4FusedQKVNormHeadMajorSliding(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    qWeight: MLXArray,
    kWeight: MLXArray,
    eps: Float,
    positionOffsets: MLXArray,
    ropeParameters: Gemma4QKVRopeParameters, applyRope: Bool,
    packMirrors: Bool = false
) -> (q: MLXArray, k: MLXArray, v: MLXArray, appliedRope: Bool, mirrors: [MLXArray]?)? {
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
    let template: [(String, any KernelTemplateArg)] = [
        ("T", q.dtype), ("D", dimension), ("Q_ROWS", qRows),
        ("K_ROWS", kRows), ("TOTAL_ROWS", rows), ("RPT", rowsPerGroup),
        ("LQ", lq), ("HQ", hq), ("LK", lk), ("HK", hk),
        ("APPLY_ROPE", fusedRope),
    ]
    // PROMPT-GLUE (pg1): the pack fold needs the rotated K values in the
    // kernel (fusedRope) and one mirror output per batch row (<= 8).
    if packMirrors, fusedRope, Gemma4PromptGlueV1.packEnabled,
        batch <= 8, rows % rowsPerGroup == 0
    {
        let words = dimension / 8 + dimension / 64
        let outputShapes: [[Int]] = [
            [batch, hq, lq, dimension], [batch, hk, lk, dimension],
            [batch, hk, lk, dimension],
        ] + (0 ..< 8).map { $0 < batch ? [2, hk, lk, words] : [1] }
        let outputDTypes = [q.dtype, q.dtype, q.dtype]
            + Array(repeating: DType.uint32, count: 8)
        // PROMPT-GLUE2 (pg2): the prompt plane takes the vector-access twin
        // at its own rows per threadgroup; pg1 stays for everything else.
        if Gemma4PromptGlue2V1.enabled, batch * max(lq, lk) >= Gemma4PromptGlue2V1.minRows {
            let rowsPerGroup2 = gemma4QKVNormPrefillSlidingPackRowsPerGroupPg2
            let groups2 = (rows + rowsPerGroup2 - 1) / rowsPerGroup2
            let template2: [(String, any KernelTemplateArg)] = [
                ("T", q.dtype), ("D", dimension), ("Q_ROWS", qRows),
                ("K_ROWS", kRows), ("TOTAL_ROWS", rows), ("RPT", rowsPerGroup2),
                ("LQ", lq), ("HQ", hq), ("LK", lk), ("HK", hk),
                ("APPLY_ROPE", fusedRope),
            ]
            let outputs = gemma4QKVNormPrefillSlidingPackKernelPg2(
                [q, k, v, qWeight, kWeight, positionOffsets, ropeParameters.log2Base],
                template: template2,
                grid: (groups2 * rowsPerGroup2 * rowThreads, 1, 1),
                threadGroup: (rowsPerGroup2 * rowThreads, 1, 1),
                outputShapes: outputShapes,
                outputDTypes: outputDTypes
            )
            if Gemma4PromptGlue2V1.xcheck {
                let reference = gemma4QKVNormPrefillSlidingPackKernel(
                    [q, k, v, qWeight, kWeight, positionOffsets, ropeParameters.log2Base],
                    template: template,
                    grid: (groups * rowsPerGroup * rowThreads, 1, 1),
                    threadGroup: (rowsPerGroup * rowThreads, 1, 1),
                    outputShapes: outputShapes,
                    outputDTypes: outputDTypes
                )
                for index in 0 ..< (3 + batch) {
                    Gemma4PromptGlue2V1.report(
                        outputs[index], reference: reference[index],
                        site: "qkv-norm-rope-pack output \(index)")
                }
            }
            CBv2EngageMark.once("qkv-norm-rope-prefill-sliding")
            Gemma4PromptGlue2V1.mark()
            return (outputs[0], outputs[1], outputs[2], true, Array(outputs[3 ..< (3 + batch)]))
        }
        let outputs = gemma4QKVNormPrefillSlidingPackKernel(
            [q, k, v, qWeight, kWeight, positionOffsets, ropeParameters.log2Base],
            template: template,
            grid: (groups * rowsPerGroup * rowThreads, 1, 1),
            threadGroup: (rowsPerGroup * rowThreads, 1, 1),
            outputShapes: outputShapes,
            outputDTypes: outputDTypes
        )
        CBv2EngageMark.once("qkv-norm-rope-prefill-sliding")
        return (outputs[0], outputs[1], outputs[2], true, Array(outputs[3 ..< (3 + batch)]))
    }
    let outputs = gemma4QKVNormPrefillSlidingKernel(
        [q, k, v, qWeight, kWeight, positionOffsets, ropeParameters.log2Base],
        template: template,
        grid: (groups * rowsPerGroup * rowThreads, 1, 1),
        threadGroup: (rowsPerGroup * rowThreads, 1, 1),
        outputShapes: [
            [batch, hq, lq, dimension], [batch, hk, lk, dimension],
            [batch, hk, lk, dimension],
        ],
        outputDTypes: [q.dtype, q.dtype, q.dtype]
    )
    if fusedRope { CBv2EngageMark.once("qkv-norm-rope-prefill-sliding") }
    return (outputs[0], outputs[1], outputs[2], fusedRope, nil)
}

private func gemma4FusedQKVNorm(
    q: MLXArray, k: MLXArray, v: MLXArray,
    qWeight: MLXArray, kWeight: MLXArray, eps: Float,
    keyValueShared: Bool, positionOffsets: MLXArray,
    ropeParameters: Gemma4QKVRopeParameters, applyRope: Bool
) -> (q: MLXArray, k: MLXArray, v: MLXArray, appliedRope: Bool)? {
    guard eps == 1.0e-6,
        q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16,
        qWeight.dtype == .bfloat16, kWeight.dtype == .bfloat16,
        positionOffsets.dtype == .int32, positionOffsets.shape == [8],
        ropeParameters.log2Base.dtype == .float32, ropeParameters.log2Base.size == 1,
        ropeParameters.frequencies.dtype == .float32,
        q.ndim == 4, k.ndim == 4, v.ndim == 4,
        q.dim(0) == 8, q.dim(1) == 1, q.dim(2) == 16,
        k.dim(0) == 8, k.dim(1) == 1, v.shape == k.shape,
        q.dim(3) == k.dim(3),
        (q.dim(3) == 256 && k.dim(2) == 8) || (q.dim(3) == 512 && k.dim(2) == 2),
        qWeight.shape == [q.dim(3)], kWeight.shape == [q.dim(3)],
        !keyValueShared || v.shape == k.shape,
        !ropeParameters.usesFrequencies
            || ropeParameters.frequencies.size == q.dim(3) / 2
    else { return nil }

    let dimension = q.dim(3)
    let qRows = 8 * 16
    let kRows = 8 * k.dim(2)
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
            ? [[8, 16, 1, dimension], [8, k.dim(2), 1, dimension], v.shape]
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

// MARK: - PREFILL-DEQ-GEMM-001: prompt-plane dense projections on the bf16 GEMM road

/// PREFILL-DEQ-GEMM-001. On the scored prompt plane (`[8, 1024]` packed rows,
/// M = 8192 activation rows) every dense projection of a layer -- the q and
/// k/v projections and `o_proj` in attention, gate/up/down in the dense MLP --
/// reaches MLX's `QuantizedMatmul`, which at this M selects the quantized
/// tile kernel (`affine_qmm_t`, or `affine_qmm_t_nax` on a device with the
/// matrix accelerator). That kernel dequantizes each 64x64 weight block once
/// PER M-TILE of the output: at M = 8192 with 64-row tiles every weight plane
/// of the layer is dequantized 128 times over, each time into threadgroup
/// memory between two barriers, ahead of one 64x64 accumulation step.
///
/// This road dequantizes each weight plane ONCE per layer per prompt pass
/// (`dequantized`, a single bandwidth pass over the packed plane into a
/// transient bf16 plane) and hands that plane to MLX's dense GEMM
/// (`steel_gemm_fused`; `steel_gemm_fused_nax` on the accelerator: a wider
/// tile, a deeper K step, both operands loaded straight from device memory,
/// no dequantization and no threadgroup staging inside the K loop). The
/// weight bytes the GEMM streams are four times wider, but the dequantization
/// work drops from once per M-tile to once, and the K loop loses the
/// per-step dequantize-store-barrier sequence entirely.
///
/// ## Exactness
///
/// Every output element is the same fp32 accumulation of the same bf16
/// products, stored once to bf16:
///
///  1. The dequantized value. `affine_dequantize`
///     (`mlx-generated/metal/quantized.h`) is instantiated at
///     `T = result_type(scales, biases)` = bf16 and computes
///     `out[i] = scale * d + bias` with `uint8_t d` the extracted code. The
///     quantized tile kernel's cooperative loader (`QuantizedBlockLoader ->
///     dequantize<T, N, bits>`) computes `s[0] * (w & 0x0f) + bias` for the
///     low nibble and `s[1] * (w & 0xf0) + bias` with `s[1] = scale / 16` for
///     the high one; `scale / 16` is an exact power-of-two rescale in bf16
///     and `(scale / 16) * (16 * d)` is the same fp32 product as
///     `scale * d`, so both roads produce the identical bf16 word for every
///     code, 4-bit and 8-bit alike. The NAX loader (`quantized_nax.h`) calls
///     the same `dequantize` template.
///  2. The accumulation. This is a CLAIM ABOUT KERNEL INTERNALS, not a
///     theorem, and it is what the cross-check below exists to decide. Both
///     kernels are `mlx::steel` block GEMMs over the same operands: they
///     walk K in ascending order and accumulate every K sub-step into one
///     fp32 tile with the same matrix-unit instruction (`BlockMMA` over
///     8-wide K fragments off the accelerator, `tile_matmad_nax` over
///     16-wide fragments on it), and both store the plain `T(acc)` rounding.
///     If the fragment K width and the ascending order match, the block
///     width BK only changes how many fragment steps are issued between two
///     threadgroup loads, not the association of the sum, and every output
///     element is bit-identical. If they do NOT match, the sum is
///     re-associated and the road is inexact -- a value change inside the
///     board's per-stream token tolerance, not a defect, but it must be
///     reported as such rather than assumed away.
///
///     The non-NAX pair (`affine_qmm_t` vs `steel_gemm_fused_nt`) is
///     directly decidable on this development part: `..._XCHECK=1` runs both
///     dispatches on identical operands and counts differing bf16 words, and
///     the prefill token plane / KV digest gate decides it end to end. The
///     NAX pair (`affine_qmm_t_nax` vs `steel_gemm_fused_nax`) is not
///     executable off the accelerator and is carried on the argument above.
///
/// Admission: affine mode, group size 64, 4- or 8-bit, bf16 activations and
/// scales, no bias, and at least `minRows` activation rows. Decode (`[8, 1]`),
/// the MTP verify rectangles and the one-row frontier tail of the final
/// prompt layer never reach the row floor and keep the incumbent dispatch,
/// as does every other geometry. Kill switch:
/// `DARKBLOOM_GEMMA4_PREFILL_DEQ_GEMM=0` (also `false`/`no`/`off`) restores
/// the quantized dispatch byte for byte. Engage mark: `prefill-deq-gemm`,
/// fired at the site that builds the dequantize + GEMM graph.
private enum Gemma4PrefillDeqGEMMV1 {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_DEQ_GEMM"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Cache for dequantized transposed weight planes across prompt passes.
    /// `DARKBLOOM_GEMMA4_PREFILL_DEQ_CACHE=0` restores dynamic dequantization on every call.
    static let cacheEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_DEQ_CACHE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Activation-row floor. The scored prompt plane carries 8192 rows; the
    /// default admits any plane of at least 1024 rows (one full-length
    /// prompt row) so the ranked geometry and a solo prompt take the same
    /// road, while every decode and verify rectangle (8..32 rows) does not.
    static let minRows: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_DEQ_GEMM_MIN_ROWS"],
            let value = Int(raw), value > 0
        else { return 1024 }
        return value
    }()

    private static let planeLock = NSLock()
    nonisolated(unsafe) private static var cachedTransposedPlanes: [ObjectIdentifier: MLXArray] = [:]

    @inline(__always)
    private static func plane(for quantized: QuantizedLinear, biases: MLXArray) -> MLXArray {
        guard cacheEnabled else {
            return dequantized(
                quantized.weight, scales: quantized.scales, biases: biases,
                groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode
            ).transposed()
        }
        let key = ObjectIdentifier(quantized)
        planeLock.lock()
        let existing = cachedTransposedPlanes[key]
        planeLock.unlock()
        if let existing { return existing }
        // DEQ-PLANE-LOCK-001: the plane is built and evaluated with no lock
        // held (an evaluation under the table lock stalls any other prompt
        // pass that reaches the table meanwhile); the table is then taken
        // only to insert. A pass that built the same plane concurrently
        // keeps the first entry — the same values from the same operation.
        let p = dequantized(
            quantized.weight, scales: quantized.scales, biases: biases,
            groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode
        ).transposed()
        eval(p)
        planeLock.lock()
        if let raced = cachedTransposedPlanes[key] {
            planeLock.unlock()
            return raced
        }
        cachedTransposedPlanes[key] = p
        planeLock.unlock()
        return p
    }

    @inline(__always)
    static func apply(_ layer: Linear, _ x: MLXArray) -> MLXArray? {
        guard enabled,
            let quantized = layer as? QuantizedLinear,
            quantized.bias == nil,
            quantized.mode == .affine,
            quantized.groupSize == 64,
            quantized.bits == 4 || quantized.bits == 8,
            x.dtype == .bfloat16, x.ndim >= 2,
            quantized.scales.dtype == .bfloat16,
            let biases = quantized.biases, biases.dtype == .bfloat16
        else { return nil }
        let inputDims = x.dim(-1)
        guard inputDims > 0, x.size / inputDims >= minRows else { return nil }
        let weight = quantized.weight
        guard weight.ndim == 2, weight.dtype == .uint32,
            weight.dim(1) * (32 / quantized.bits) == inputDims
        else { return nil }
        CBv2EngageMark.once("prefill-deq-gemm")
        let transPlane = plane(for: quantized, biases: biases)
        let product = MLX.matmul(x, transPlane)
        if xcheck {
            // Local diagnostics only (never on the ranked path): evaluate the
            // incumbent quantized dispatch beside this road on the identical
            // operands and count differing bf16 words. An exact road reports
            // zero on every call; anything else is a defect in this file.
            let incumbent = layer(x)
            let differing = MLX.sum(
                product.view(dtype: .uint16) .!= incumbent.view(dtype: .uint16),
                stream: .default)
            eval(product, incumbent, differing)
            FileHandle.standardError.write(
                Data(
                    ("[xcheck] prefill-deq-gemm rows \(x.size / inputDims) "
                        + "K \(inputDims) N \(weight.dim(0)) bits \(quantized.bits) "
                        + "differing \(differing.item(Int32.self))\n").utf8))
        }
        return product
    }

    /// `DARKBLOOM_GEMMA4_PREFILL_DEQ_GEMM_XCHECK=1`: bitwise cross-check of
    /// every admitted projection against the incumbent dispatch (diagnostic;
    /// forces evaluation, so it is never set on a timed run).
    static let xcheck: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_PREFILL_DEQ_GEMM_XCHECK"] == "1"
}

// MARK: - Attention

/// QKFUSE-SLIDING. Default ON: sliding attention layers (vProj != nil) take
/// the fused Q|K dispatch alongside the K-eq-V global layers, while V keeps
/// its separate tierProjection. The fused kernels admit the sliding widths
/// (qWidth 4096, kWidth 2048) and compute each output column from that
/// column's own plane row, so the Q and K halves are bit-identical to the
/// separate q_proj/k_proj dispatches. The per-layer cached concatenated
/// planes ([6144, 352] uint32 + [6144, 44] bf16 scales + same-size biases)
/// add 9,732,096 bytes (~9.3 MiB) of resident memory per sliding layer,
/// 243,302,400 bytes (~232 MiB) across the 25 sliding layers.
/// `DARKBLOOM_GEMMA4_QKFUSE_SLIDING=0` (also false/no/off) restores the
/// vProj == nil gate.
private let gemma4QKFuseSlidingEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_QKFUSE_SLIDING"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

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

        // RoPE: sliding uses the base route; full attention reuses the exact
        // proportional frequency table (including +inf pass-through pairs).
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

    /// Exact B8/L1 Q/K/V projection: the tight-grid host for the promoted
    /// matrix-unit tier (same kernel text, grid.x = 1). Any guard failure
    /// keeps the quantized module, which reaches the tier through MLX.
    /// MMA-RS-001: `rsTable` is the layer input's precomputed affine run-sum
    /// table; nil keeps the incumbent in-kernel reductions.
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
        else { return Gemma4PrefillDeqGEMMV1.apply(layer, x) ?? layer(x) }
        return projected
    }

    /// QKFUSE-001. Q and K read the same activation at decode, so their
    /// planes concatenate into one dispatch. Nil whenever the shapes, the
    /// quantization parameters or the arm's switch say otherwise.
    /// MMA-RS-001: `rsTable` is the layer input's shared run-sum table; the
    /// fused concatenated-N dispatch reads the same per-row, per-64-group
    /// entries the separate Q and K dispatches would; nil keeps the
    /// incumbent fused dispatch.
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

    /// Exact B8/L1 attention output projection. Sliding/full K widths select
    /// the tight affine4 fast-QMV replica; every other path keeps the layer.
    /// MMA-RS-001: the projection input's run-sum table is computed here (the
    /// o_proj plane consumes it alone); nil keeps the incumbent dispatch.
    /// ORSFOLD-001: `carriedRunsum` is the table the resident attention kernel
    /// emitted for this exact activation; nil, or any table that misses the
    /// shape contract, falls through to the standalone prepass.
    /// ORS-D512: a carried `[8, 256]` pair table (the D=512 dispatch-3
    /// epilogue at 32-column tiles) takes the `_rsp2` o_proj body; a carried
    /// `[8, 128]` table (64-column tiles) takes the established `_rsp` body.
    @inline(__always)
    private func outputProjection(
        _ x: MLXArray, carriedRunsum: MLXArray? = nil
    ) -> MLXArray {
        let carriedPairs = CBv2AttentionOQMVV1.acceptRunsumPairTable(
            carriedRunsum, for: x)
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
                rsTable: carriedPairs != nil
                    ? nil
                    : (CBv2AttentionOQMVV1.acceptRunsumTable(
                        carriedRunsum, for: x)
                        ?? CBv2AttentionOQMVV1.runsumTable(for: x)),
                rsPairTable: carriedPairs)
        else { return Gemma4PrefillDeqGEMMV1.apply(oProj, x) ?? oProj(x) }
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
        carriedRunsum: MLXArray? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4.PositionOffset) {
        // ContinuousBatchingV2: the layer cache owns both the KV update and
        // the attention computation (no masks, no padding — see
        // CBv2Contracts.swift). Entirely separate branch; the legacy paths
        // below are untouched.
        if let layerCacheV2 = cache as? (any CBv2AttendingLayerCache) {
            return forwardV2(
                x, layerCache: layerCacheV2, source: v2SharedSource,
                sharedKV: sharedKV, positionOffset: positionOffset,
                outputStart: outputStart, useLastQueryPrefill: useLastQueryPrefill,
                carriedRunsum: carriedRunsum)
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
        useLastQueryPrefill: Bool = false,
        carriedRunsum: MLXArray? = nil
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

        // MMA-RS-001: one affine run-sum table for the layer input serves
        // every Q/K/V projection that consumes x (at decode queryInput is x
        // itself; the prefill paths leave the table nil because the guard
        // requires the exact L=1 decode shape, and the incumbent dispatch
        // runs). The table kernel is lazy: it executes only when a projection
        // actually consumes it.
        let qkvRunsumTable = CBv2AttentionQKVMMA8V1.runsumTable(
            for: x, carried: carriedRunsum)

        // Keep Q/K/V on the promoted matrix-unit tier's arithmetic. At the
        // exact B=8/L=1 decode shapes the tight-grid host re-dispatches the
        // tier's own kernel text with grid.x = 1 (the frozen MLX host launches
        // 8 x-groups and the tier returns from 7 of them); every other shape
        // keeps the module's affine_qmv road. Routing Q or K through the older
        // custom helper would silently bypass the winning kernel.
        // QKFUSE-001: `queryInput === x` unless last-query prefill narrowed it,
        // which is the only case where Q and K cannot share a dispatch.
        // QKFUSE-SLIDING: sliding layers (vProj != nil) take the fused Q|K
        // dispatch too; V keeps its separate tierProjection below, and the
        // K-eq-V structure is untouched (keyValueShared stays vProj == nil).
        // `DARKBLOOM_GEMMA4_QKFUSE_SLIDING=0` restores the vProj == nil gate.
        // The relaxation is only reachable here; fusedQKProjection's own
        // admission still requires the exact B=8/L=1 decode shape, so
        // prefill, last-query, shared-KV and other batch widths keep their
        // incumbent dispatches.
        // MMA-RS-001: the fused Q|K dispatch consumes the shared run-sum
        // table — the table is per activation row and per 64-group of K,
        // independent of N, so the concatenated-N dispatch reads the same
        // entries the separate Q and K dispatches would.
        let fusedQK: (MLXArray, MLXArray)? =
            (lastQueryCache == nil && !usesSharedKV
                && (vProj == nil || gemma4QKFuseSlidingEnabled))
            ? fusedQKProjection(x, rsTable: qkvRunsumTable) : nil
        let queryRaw = (
            fusedQK?.0 ?? tierProjection(qProj, queryInput, rsTable: qkvRunsumTable)
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
        let kRaw = (
            fusedQK?.1 ?? tierProjection(kProj, x, rsTable: qkvRunsumTable)
        ).reshaped(B, L, nKvHeads, effectiveHeadDim)
        let vRaw: MLXArray
        if let vProj {
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
            // Written head-major, so the three transposes are already applied.
            (queries, k, v) = (headMajor.q, headMajor.k, headMajor.v)
            appliedRope = headMajor.appliedRope
        } else if let sliding = gemma4FusedQKVNormHeadMajorSliding(
            q: queryRaw, k: kRaw, v: vRaw,
            qWeight: qNorm.weight, kWeight: kNorm.weight, eps: config.rmsNormEps,
            positionOffsets: capturedOffsets,
            ropeParameters: qkvRopeParameters, applyRope: lastQueryCache == nil,
            packMirrors: lastQueryCache == nil)
        {
            // Sliding twin: also written head-major, with base-route RoPE.
            (queries, k, v) = (sliding.q, sliding.k, sliding.v)
            appliedRope = sliding.appliedRope
            // PROMPT-GLUE (pg1): the mirrors packed beside these exact K/V
            // arrays; the prompt commit takes them by identity.
            if let mirrors = sliding.mirrors {
                Gemma4PromptGlueV1.registerPackedMirrors(keys: k, values: v, mirrors: mirrors)
            }
        } else {
            queries = qNorm(queryRaw).transposed(0, 2, 1, 3)
            k = kNorm(kRaw).transposed(0, 2, 1, 3)
            v = vNorm(vRaw).transposed(0, 2, 1, 3)
        }

        if !appliedRope {
            queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: queryPositionOffset)
            k = gemma4ApplyRotaryPosition(rope, to: k, offset: captured)
        }
        // F4: let the resident decode attention kernel take the raw
        // projections and normalize/rotate them once itself; a miss falls
        // back to the arrays above.
        if isSliding {
            _ = CBv2RaggedTwoPassDecodeAttentionV1.registerResidentNormRope(
                normalizedQueries: queries, normalizedKeys: k, normalizedValues: v,
                rawQueries: queryRaw, rawKeys: kRaw, rawValues: vRaw,
                qWeight: qNorm.weight, kWeight: kNorm.weight,
                positionOffsets: capturedOffsets,
                ropeLog2Base: qkvRopeParameters.log2Base,
                eps: config.rmsNormEps, appliedRope: appliedRope)
        } else if vProj == nil, qkvRopeParameters.usesFrequencies {
            // NORMROPE-D512: the full layers' store dispatch takes the raw
            // k-eq-v projections and normalizes/rotates them once itself; a
            // miss falls back to the arrays above.
            _ = CBv2RaggedComposedD512DecodeAttentionV1.registerFullNormRope(
                normalizedQueries: queries, normalizedKeys: k, normalizedValues: v,
                rawQueries: queryRaw, rawKeys: kRaw, rawValues: vRaw,
                qWeight: qNorm.weight, kWeight: kNorm.weight,
                positionOffsets: capturedOffsets,
                ropeFrequencies: qkvRopeParameters.frequencies,
                eps: config.rmsNormEps, appliedRope: appliedRope)
        }

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

        let residentProducts =
            CBv2RaggedTwoPassDecodeAttentionV1.takeResidentProducts(for: attention)
        var output = attention.transposed(0, 2, 1, 3).reshaped(B, queryLength, -1)
        if lastQueryCache == nil && outputStart > 0 {
            output = output[0..., outputStart..., 0...]
        }
        if output.dtype != outputDType {
            output = output.asType(outputDType)
        }
        return (
            outputProjection(output, carriedRunsum: residentProducts?.runsumTable),
            (residentProducts?.normalizedKeys ?? k,
             residentProducts?.normalizedValues ?? v),
            captured)
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

/// ROUTER-FINALISTS-017: selection only, within the existing ZIP stage.
/// Keep the stable ascending argsort tail: each 32-entry subset retains its
/// largest eight, then one SIMD group sorts the 32 survivors. A discarded
/// element already has eight successors in its own subset. The total order
/// is sort.h LessThan plus original expert index, including NaNs and zeros.
/// No score arithmetic, weight fusion, or expert-assignment sort is changed.
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
            scores.ndim == 3, scores.dim(0) == 8,
            scores.dim(1) == 1, scores.dim(2) == 128,
            scores.dtype == .bfloat16
        else { return nil }
        return kernel(
            [scores],
            grid: (8 * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[8, 1, 8]],
            outputDTypes: [.uint32]
        )[0]
    }

    /// PREFILL-W1 mechanism 2: admits the identical finalists32 selection
    /// kernel above to prompt-rectangle shapes `[8, L, 128]` with `L > 1`,
    /// which `apply` above never reaches (it is pinned to the decode cell's
    /// `L == 1`). The kernel body is untouched: each threadgroup already
    /// processes exactly one flattened row of 128 scores addressed by
    /// `threadgroup_position_in_grid.x`, so a `[B, L, E]` row-major buffer
    /// flattens to `B*L` such rows the same way `[B, 1, E]` flattens to `B`
    /// of them, and the `[B, L, 8]` output flattens identically on the
    /// output side. Nothing about the selection arithmetic depends on how
    /// the row count was produced.
    ///
    /// Fail-closed: any shape, dtype, topK, or kth outside the pin falls
    /// through to the caller's stock `argPartition` chain. Independent kill
    /// switch `DARKBLOOM_GEMMA4_ROUTER_FINALISTS32_PREFILL=0` so the prefill
    /// admission can be disabled without touching the decode path's own
    /// switch above. Engage mark: `router-finalists32-prefill`.
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

/// ROUTER-TAIL-PREFILL: fold the prefill router's four dependent stock
/// weight ops — `takeAlong(expertScores, topKIndices, -1)` →
/// `softmax(..., precise: true)` → `perExpertScale[topKIndices]` gather
/// → bf16 `multiply` (four dispatches x 30 MoE layers per prefill, plus
/// a re-read of the whole `[8, L, 128]` score plane) — into the
/// finalists32 selection kernel that already holds the eight winners.
/// The selection network is the incumbent
/// `gemma4_router_finalists32_stable_bf16_v1` verbatim; the weight tail
/// is the decode cell's adversarially verified `Gemma4FusedRouterTop8`
/// tail — the bit-exact transcription of
/// `softmax_single_row<bfloat16_t, float, N_READS=4>` (softmax.h) plus
/// the stock bf16 per-expert-scale multiply — ported onto the packed
/// winners this kernel already holds in its group-0 lanes 24-31. The
/// O(E^2)-per-row selection of `Gemma4FusedRouterTop8` itself stays
/// decode-only; this kernel keeps the finalists32 selection network.
///
/// Exactness, op by op against the stock chain:
/// 1. `takeAlong` is a pure gather of the bf16 scores at the selected
///    indices (no arithmetic). The winners ride the selection network as
///    unchanged BF16 bits inside the packed payload, so
///    `float(uint16_to_bfloat16(bits))` is the bit-identical float32 of
///    the bit-identical bf16 score, staged in the same ascending-rank
///    order `takeAlong` emits (position p = rank - kth, lane 24 + p).
/// 2. `softmax(precise: true)` on bf16 dispatches the single kernel
///    `block_softmax_precise_bfloat16` =
///    `softmax_single_row<bfloat16_t, float, N_READS=4>`: float32
///    accumulation throughout, with exactly one bf16 rounding, at the
///    output write `T(ld[i] * normalizer)`. The stock axis-8 launch is
///    ONE 32-thread simdgroup per row; this kernel's group 0 is exactly
///    that simdgroup (its lanes are threadgroup positions 0-31), so the
///    transcribed lane layout (lanes 0-1 hold the 8 values, 4 reads
///    each), the `Limits<float>::min` (-inf) padding, `fast::exp`, the
///    `simd_max`/`simd_sum` reduction order and the `1 / normalizer`
///    division all execute on the same lanes with the same values.
///    Every write to the shared max/normalizer slots is gated to group
///    0 — in the stock one-simdgroup launch only slot 0 is ever
///    written — so slots 1-31 keep their -inf / 0 init values exactly
///    as the stock launch leaves them, and the cross-simdgroup
///    `simd_max(local_max[lane])` / `simd_sum(local_normalizer[lane])`
///    combines identical operands. Groups 1-3 only meet the barriers.
/// 3. The per-expert-scale gather is a pure bf16 copy; `pes[topi[p]]`
///    loads the identical element of the identical vector.
/// 4. The stock `Multiply` on bfloat16 is MSL `x * y` (binary_ops.h),
///    so `w * pes[topi[p]]` is the same expression on the same types
///    with the same single bf16 rounding as the stock binary kernel.
///
/// Fail-closed: any shape, dtype, topK, or kth outside the pinned
/// prefill geometry, a disabled finalists stage (either of its two kill
/// switches), or the kill switch below selects the unchanged
/// selection-only kernel plus the stock four-op chain. Kill switch:
/// `DARKBLOOM_GEMMA4_ROUTER_WEIGHTS32_PREFILL=0` (off = the exact
/// incumbent chain). Engage mark: `router-weights32-prefill`.
private enum Gemma4RouterFinalistsWeightsV1 {
    /// Default ON; `DARKBLOOM_GEMMA4_ROUTER_WEIGHTS32_PREFILL` with
    /// `0/false/no/off` restores the stock four-op weight chain.
    static let prefillEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_ROUTER_WEIGHTS32_PREFILL"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_router_finalists32_weights_bf16_v1",
        inputNames: ["scores", "pes"],
        outputNames: ["indices", "weights"],
        source: """
            constexpr int SIMD_SIZE = 32;
            constexpr int N_READS = 4;
            constexpr int K = 8;

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
            threadgroup float topv[K];
            threadgroup uint topi[K];
            threadgroup float local_max[SIMD_SIZE];
            threadgroup float local_normalizer[SIMD_SIZE];

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
                if (lane >= 24u) {
                    indices[row * 8u + lane - 24u] = item & 127u;
                    // Stage the winners for the weight tail in the stock
                    // chain's ascending-rank (takeAlong) order: the
                    // unchanged BF16 score bits and the expert id.
                    topv[lane - 24u] = float(uint16_to_bfloat16(uint16_t(item >> 7)));
                    topi[lane - 24u] = item & 127u;
                }
            }
            // All four complete SIMD groups participate in this barrier.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // softmax_single_row<T, float, N_READS=4> transcription
            // (softmax.h) at axis_size = K on group 0 — the stock axis-8
            // launch's single 32-thread simdgroup — with the stock bf16
            // per-expert-scale multiply fused into the write (the
            // verified decode tail of Gemma4FusedRouterTop8). Groups 1-3
            // only meet the barriers; every shared-slot write is gated to
            // group 0 so slots 1-31 keep their init values exactly as the
            // stock one-simdgroup launch leaves them.
            float ld[N_READS];
            const int base = int(lane) * N_READS;
            if (group == 0u) {
                if (base + N_READS <= K) {
                    for (int i = 0; i < N_READS; i++) {
                        ld[i] = topv[base + i];
                    }
                } else {
                    for (int i = 0; i < N_READS; i++) {
                        ld[i] = ((base + i) < K) ? topv[base + i] : Limits<float>::min;
                    }
                }
                local_max[lane] = Limits<float>::min;
                local_normalizer[lane] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (group == 0u) {
                float maxval = Limits<float>::finite_min;
                for (int i = 0; i < N_READS; i++) {
                    maxval = (maxval < ld[i]) ? ld[i] : maxval;
                }
                maxval = simd_max(maxval);
                if (lane == 0u) {
                    local_max[0] = maxval;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                float maxval = simd_max(local_max[lane]);
                if (lane == 0u) {
                    local_max[0] = maxval;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                const float maxval = local_max[0];
                float normalizer = 0;
                for (int i = 0; i < N_READS; i++) {
                    float exp_x = fast::exp(ld[i] - maxval);
                    ld[i] = exp_x;
                    normalizer += exp_x;
                }
                normalizer = simd_sum(normalizer);
                if (lane == 0u) {
                    local_normalizer[0] = normalizer;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                float normalizer = simd_sum(local_normalizer[lane]);
                if (lane == 0u) {
                    local_normalizer[0] = normalizer;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                const float normalizer = 1 / local_normalizer[0];
                if (base + N_READS <= K) {
                    for (int i = 0; i < N_READS; i++) {
                        const T w = T(ld[i] * normalizer);
                        weights[row * 8u + uint(base + i)] = w * pes[topi[base + i]];
                    }
                } else {
                    for (int i = 0; i < N_READS; i++) {
                        if ((base + i) < K) {
                            const T w = T(ld[i] * normalizer);
                            weights[row * 8u + uint(base + i)]
                                = w * pes[topi[base + i]];
                        }
                    }
                }
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

    static func applyPrefill(
        _ scores: MLXArray, perExpertScale: MLXArray, topK: Int, kth: Int
    ) -> (indices: MLXArray, weights: MLXArray)? {
        guard prefillEnabled, Gemma4RouterFinalistsV1.enabled,
            Gemma4RouterFinalistsV1.prefillEnabled,
            topK == 8, kth == 120,
            scores.ndim == 3, scores.dim(0) == 8,
            scores.dim(1) > 1, scores.dim(2) == 128,
            scores.dtype == .bfloat16,
            perExpertScale.ndim == 1, perExpertScale.dim(0) == 128,
            perExpertScale.dtype == .bfloat16
        else { return nil }
        let b = scores.dim(0)
        let l = scores.dim(1)
        let rows = b * l
        CBv2EngageMark.once("router-weights32-prefill")
        let outs = kernel(
            [scores, perExpertScale],
            template: [("T", scores.dtype)],
            grid: (rows * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[b, l, 8], [b, l, 8]],
            outputDTypes: [.uint32, .bfloat16]
        )
        return (outs[0], outs[1])
    }
}

/// GLUE-FOLD (G2): merge the two ADJACENT single-threadgroup route glue
/// stages of the pinned B=8 decode cell into one dispatch. On the incumbent
/// chain the router scores feed `gemma4_router_finalists32_stable_bf16_v1`
/// (top-8 selection, one 128-thread group per row) and its `[8, 8]` output
/// then feeds the strictly-serial `mlx_lm_route_simd_rank_scatter_m8_u32_n64`
/// launch inside `SwitchGLU.projectExperts` (the sorted route table). Both are
/// launch-drain stages on the layer's DEPENDENT chain: the expert gathers
/// cannot start until the rank scatter has drained. This kernel runs the
/// identical selection network and the identical rank scatter back to back in
/// ONE 1024-thread threadgroup (8 rows x 128 threads; the 64 selected keys
/// are staged through threadgroup memory instead of a device round trip), so
/// one whole dispatch and its barrier stage leave the dependent chain in every
/// MoE layer of every decode step (x30/step).
///
/// Exactness by construction: every operation in both phases is integer or
/// raw-bit work -- the comparator reads the unchanged BF16 score bits as a
/// packed word exactly like the incumbent finalists kernel, the selection
/// bitonic and the rank loop are copied verbatim (only the threadgroup-memory
/// indexing gains a `row` offset), and no floating-point value is produced,
/// re-associated or re-rounded anywhere. There is no accumulation and no
/// reduction-tree change to reason about: outputs are bit-identical to the
/// incumbent pair of kernels for every input. The staged `sel` array holds
/// exactly the values the incumbent chain would have written to the `[8, 8]`
/// indices buffer, and phase 2 reads them at the same flattened positions.
///
/// Fail-closed: any geometry other than the pinned decode cell, a disabled
/// finalists stage, plan != 1, or the kill switch selects the incumbent
/// two-dispatch chain unchanged. `DARKBLOOM_GEMMA4_GLUE_FOLD=0` is the kill
/// switch (off = exact incumbent chain). Engage mark: `glue-fold`.
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

/// GLUE-003: one-per-forward chain box. Layer L's fused tail deposits the
/// (output, next-layer-input-norm) pair; layer L+1 consumes the norm instead
/// of re-reading and re-normalizing the same tensor — guarded by pointer
/// identity on the source array, so any intervening transformation falls back
/// to the stock `inputLayernorm(x)`.
public final class Gemma4GlueChainBox {
    var pending: (source: MLXArray, normed: MLXArray, rs: MLXArray?)?
    public init() {}
}

/// GLUE-001: fused B8/L1 layer glue. Three single-dispatch kernels
/// replace the strictly SERIAL RMSNorm/add chains between the layer's matmuls
/// at the exact ranked decode geometry ([8, 1, 2816] bfloat16). These
/// shape-based gates also admit eligible final-prefill tails narrowed to
/// the same geometry:
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

    /// Pair only the two independent branch reductions in the existing
    /// [8, 1, 2816] tails. The shape gate includes decode and eligible final
    /// prefill tails narrowed to L=1. Disabling this selects their unchanged
    /// original kernels.
    private static let pairedRmsEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_DECODE_PAIRED_RMS"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// DENSE-XSUM-ELIDE: With MMA-GATEUP-DEFAULT-001 promoted, the dense MLP gate/up
    /// projection runs the matrix-unit body and never consumes the DMLP-002
    /// activation-sum table. This elides the dead 5,632-element float32 buffer
    /// allocation and its writes from the attention branch prefix kernel, removes
    /// the unconsumed dependency edge in Gemma4ZipRouterV1, and skips the
    /// standalone activationSumKernel dispatch.
    /// `DARKBLOOM_GEMMA4_DENSE_XSUM_ELIDE=0` restores the unconsumed table and its edge.
    static let denseXSumElideEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_DENSE_XSUM_ELIDE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// NORM-TB1: halve the threadgroup barriers inside the four glue
    /// kernels that actually dispatch on the ranked B=8/L=1 decode cell.
    ///
    /// Each `rmsReduce` tree currently costs TWO threadgroup barriers: one to
    /// publish the 22 per-simdgroup partials, and a second one whose only job
    /// is to broadcast the single resulting normalizer back out of threadgroup
    /// memory. This switch selects the emission that keeps the first and
    /// deletes the second by having every simdgroup recompute the identical
    /// cross-simd `simd_sum` itself (see `rmsReduce` below for why that is
    /// bit-identical, not merely equal to rounding).
    ///
    /// It removes 2 of 4 barriers in the attention-branch prefix, 4 of 8 in
    /// the deferred expert tail chain, 3 of 6 in the deferred expert tail, and
    /// 1 of 2 in the layer-zero input norm. It changes NO dispatch, NO barrier
    /// stage, and NO reduction tree.
    ///
    /// `DARKBLOOM_GEMMA4_NORM_TG_BARRIER_HALVE=0` restores the incumbent
    /// kernels: every helper below returns the incumbent text and the
    /// incumbent name, so the off state is the incumbent emission verbatim.
    private static let tgBarrierHalveEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_NORM_TG_BARRIER_HALVE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// NORM-TB1 scratch plane for a kernel's i-th reduction, or nil (the
    /// incumbent two-barrier emission) when the switch is off. Consecutive
    /// reductions alternate; see `rmsReduce` for the ordering argument.
    private static func tbPlane(_ i: Int) -> String? {
        guard tgBarrierHalveEnabled else { return nil }
        return i % 2 == 0 ? "local_sums" : "local_sums_tb"
    }

    /// NORM-TB1: `local_inv` moves from threadgroup memory to a per-thread
    /// register array, because every thread now holds the normalizer it
    /// computed instead of reading back the one lane 0 stored. Every
    /// downstream `local_inv[k]` read is unchanged and now reads that thread's
    /// own copy, which is the same float in every thread.
    private static func tbInvDecl(_ n: Int) -> String {
        (tgBarrierHalveEnabled ? "float" : "threadgroup float")
            + " local_inv[\(n)];"
    }

    /// NORM-TB1 alternate scratch plane, declared only by the kernels that
    /// run more than one reduction.
    private static let tbSecondPlane: String =
        tgBarrierHalveEnabled
        ? "\n            threadgroup float local_sums_tb[32];" : ""

    /// NORM-TB1 rekey. `CustomKernel::eval_gpu` caches compiled pipelines by
    /// name (backend/metal/custom_kernel.cpp:56-70) and calls
    /// `clear_library(name_)` on a name/source mismatch, so the two emissions
    /// must never share a name.
    private static let tbSuffix: String = tgBarrierHalveEnabled ? "_tb1" : ""

    private static let rows = 8
    private static let axis = 2816
    private static let eps: Float = 1e-6
    private static let nReads = 4
    private static let tgThreads = 704  // 2816 / 4, exactly rms_single_row's shape

    /// Shared reduction preamble: the exact rms_single_row tree at 704x4.
    /// `PREFIX` names the array to reduce; `SLOT` the shared slot written.
    ///
    /// NORM-TB1 (`plane != nil`) selects the one-barrier emission. The
    /// REDUCTION TREE IS UNCHANGED: the same four squares accumulated in
    /// thread-read order, the same `simd_sum`, the same 22-slot cross-simd
    /// `simd_sum`, the same `metal::precise::rsqrt(acc / 2816 + 1e-6)`, in
    /// that order. Only the BROADCAST differs. The incumbent has simdgroup 0
    /// run the cross-simd combine, lane 0 store the normalizer to threadgroup
    /// memory, and a second threadgroup barrier so the other 21 simdgroups can
    /// read it back. This form has every simdgroup run that same combine over
    /// the same published `plane[0..21]` and keep the result in a register.
    ///
    /// That is bit-identical, not just numerically close: `simd_sum` returns
    /// one value to every lane of the simdgroup and is a deterministic
    /// function of its input vector, and every simdgroup runs it after the
    /// publish barrier and therefore over byte-identical `plane[0..21]`
    /// (slots 22..31 are never read -- the `< 22` predicate substitutes an
    /// exact `0.0f`). The float that reaches the norm body is the float the
    /// broadcast used to deliver. Nothing is reassociated, which matters
    /// because `setFastMathEnabled(false)` (backend/metal/device.cpp:631)
    /// means the compiler will not reassociate on our behalf either.
    ///
    /// Ordering: dropping the second barrier means reduction k's READERS of
    /// `plane` are no longer ordered against reduction k+1's WRITERS of it, so
    /// callers alternate two planes. Reduction k+2 may reuse reduction k's
    /// plane, because every thread's read in reduction k precedes its arrival
    /// at reduction k+1's publish barrier, and every thread's write in
    /// reduction k+2 follows its departure from that same barrier.
    private static func rmsReduce(
        _ src: String, into slot: String, plane: String? = nil
    ) -> String {
        if let plane {
            return """
                {
                    float acc = 0;
                    for (int i = 0; i < 4; i++) {
                        float xi = (float)\(src)[base + i];
                        acc += xi * xi;
                    }
                    acc = simd_sum(acc);
                    if (simd_lane_id == 0) \(plane)[simd_group_id] = acc;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    acc = simd_sum(
                        simd_lane_id < 22 ? \(plane)[simd_lane_id] : 0.0f);
                    \(slot) = metal::precise::rsqrt(acc / 2816.0f + 1e-06f);
                }
            """
        }
        return """
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

    /// Starting with four already-rounded BF16 values per thread, reproduce
    /// `mma8_runsum4` and its xor(1,2,4) tree. Adjacent tail threads own the
    /// two four-value halves, so masks 1,2,4,8 over sixteen lanes produce the
    /// same group-64 FP32 table entry.
    private static func qkvRunsumEpilogue(_ values: String) -> String {
        """
            float qkv_sum = 0.0f;
            qkv_sum += \(values)[0] + \(values)[1] + \(values)[2] + \(values)[3];
            qkv_sum += simd_shuffle_xor(qkv_sum, 1u);
            qkv_sum += simd_shuffle_xor(qkv_sum, 2u);
            qkv_sum += simd_shuffle_xor(qkv_sum, 4u);
            qkv_sum += simd_shuffle_xor(qkv_sum, 8u);
            if ((lid & 15u) == 0u) {
                qkv_rs[row * 44u + lid / 16u] = qkv_sum;
            }
        """
    }

    /// Same independent trees as prefill's glue_inv_rms2: four ordered
    /// squares per input, the original SIMD and cross-SIMD sums, then precise
    /// rsqrt. Share three barriers instead of running two three-barrier
    /// reductions. Keep the widened BF16 values for the following norm step.
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

    /// Derive both variants from the complete incumbent tail sources. Only
    /// the first two reductions and their later device reloads are replaced;
    /// every BF16 norm/product/add and the remaining reductions stay intact.
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

    /// RS0: layer zero has no predecessor tail. Fuse its input RMSNorm with
    /// the exact QKV run-sum producer so it also avoids the standalone table
    /// dispatch. This body is the public parity-tested F1 layer-zero kernel.
    private static let inputNormRunsumKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "gemma4_glue_input_rmsnorm_qkv_runsum_2816_bf16_v1"
                + tbSuffix,
            inputNames: ["x", "w"],
            outputNames: ["normed", "qkv_rs"],
            source: """
                const uint row = threadgroup_position_in_grid.x;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                \(tbInvDecl(1))
                threadgroup float local_sums[32];
                const uint base = row * 2816 + lid * 4;
                const uint wbase = lid * 4;
            \(rmsReduce("x", into: "local_inv[0]", plane: tbPlane(0)))
                const float inv = local_inv[0];
                T normedv[4];
                for (int i = 0; i < 4; ++i) {
                    normedv[i] = w[wbase + i]
                        * static_cast<T>((float)x[base + i] * inv);
                    normed[base + i] = normedv[i];
                }
            \(qkvRunsumEpilogue("normedv"))
            """,
            ensureRowContiguous: true)

    /// PREFIX-001: join the two serial normalization producers at the
    /// attention/feed-forward boundary. The first reduction reproduces
    /// `residual + postAttentionLayernorm(attnOut)` and stores that BF16
    /// boundary in `outv`. The second reduction consumes those exact rounded
    /// values in registers and emits the dense, expert and router pre-norms
    /// plus the promoted dense activation-sum table.
// T28 second sample marker (r2): content identical to ranked 32f5d19f apart from this comment.
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

    private static let attentionBranchPrefixKernelV2: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "gemma4_glue_attention_branch_prefix_2816_bf16_v2_nb1"
                + tbSuffix,
            inputNames: ["attn", "res", "wa", "wd", "we", "wr"],
            outputNames: ["out", "dense", "expert", "router"],
            source: """
                const uint row = threadgroup_position_in_grid.x;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                \(tbInvDecl(1))
                threadgroup float local_sums[32];\(tbSecondPlane)
                const uint base = row * 2816 + lid * 4;
                const uint wbase = lid * 4;
            \(rmsReduce("attn", into: "local_inv[0]", plane: tbPlane(0)))
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
            \(rmsReduce("outv", into: "local_inv[0]", plane: tbPlane(1))
                .replacingOccurrences(
                    of: "(float)outv[base + i]", with: "(float)outv[i]"))
                const float branch_inv = local_inv[0];
                for (int i = 0; i < 4; i++) {
                    const T nx =
                        static_cast<T>((float)outv[i] * branch_inv);
                    dense[base + i] = wd[wbase + i] * nx;
                    expert[base + i] = we[wbase + i] * nx;
                    router[base + i] = wr[wbase + i] * nx;
                }
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

    /// RS0 layer-input producer. Outside the exact B8/L1/K2816 decode cell,
    /// return nil and preserve the incumbent input norm plus table dispatch.
    static func inputNormWithQKVRunsum(
        x: MLXArray, weight: MLXArray, eps: Float
    ) -> (normed: MLXArray, qkvRunsumTable: MLXArray)? {
        guard admits(x, weight: weight, eps: eps) else { return nil }
        let outs = inputNormRunsumKernel(
            [x, weight],
            template: [("T", x.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[rows, 1, axis], [rows, axis / 64]],
            outputDTypes: [.bfloat16, .float32]
        )
        guard let table = CBv2AttentionQKVMMA8V1.runsumTable(
            produced: outs[1], for: outs[0])
        else { return nil }
        CBv2EngageMark.once("qkv-runsum-input-norm-fold")
        return (outs[0], table)
    }

    struct AttentionBranchPrefix {
        let out: MLXArray
        let denseNorm: MLXArray
        let expertNorm: MLXArray
        let routerNorm: MLXArray
        let denseSums: CBv2DenseMLPQMVV1.ActivationSums?
    }

    /// PREFIX-001. The returned `out` is still materialized because the layer
    /// tail consumes it as its residual. Only its otherwise-serial reread and
    /// the second dispatch disappear.
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
            (CBv2DenseMLPQMVV1.activationSumsEnabled || denseXSumElideEnabled),
            admits(attn, weight: postAttentionWeight, eps: eps),
            residual.shape == attn.shape, residual.dtype == .bfloat16,
            denseWeight.ndim == 1, denseWeight.dim(0) == axis,
            denseWeight.dtype == .bfloat16,
            expertWeight.ndim == 1, expertWeight.dim(0) == axis,
            expertWeight.dtype == .bfloat16,
            routerWeight.ndim == 1, routerWeight.dim(0) == axis,
            routerWeight.dtype == .bfloat16
        else { return nil }
        if denseXSumElideEnabled {
            CBv2EngageMark.once("dense-xsum-elide")
            let outs = attentionBranchPrefixKernelV2(
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
                ],
                outputDTypes: [
                    .bfloat16, .bfloat16, .bfloat16, .bfloat16,
                ]
            )
            CBv2EngageMark.once("attention-branch-prefix")
            return AttentionBranchPrefix(
                out: outs[0],
                denseNorm: outs[1],
                expertNorm: outs[2],
                routerNorm: outs[3],
                denseSums: nil)
        }
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

    /// GLUE-003: tail variant that ALSO emits the NEXT layer's input norm.
    /// The threadgroup already holds the finished output row in registers, so
    /// the next layer's `inputLayernorm(out)` costs one more in-kernel
    /// reduction instead of a standalone serial dispatch plus a full re-read
    /// of the row. The normed output replicates the stock rms sequence over
    /// the exact stored bf16 output values.
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

    /// Build the exact legacy `weightedExpertUnsort` value for the four
    /// features owned by this tail thread. The value remains in registers and
    /// feeds the expert RMS directly, deleting only the reduced `[8, 2816]`
    /// materialization and its standalone dispatch.
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
        name: "gemma4_glue_deferred_expert_tail_2816_bf16_v1_nb1_vec1"
            + tbSuffix,
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
            \(tbInvDecl(2))
            threadgroup float local_sums[32];\(tbSecondPlane)
            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("a", into: "local_inv[0]", plane: tbPlane(0)))
        \(deferredExpertValuesSource)
        \(rmsReduce("expertv", into: "local_inv[1]", plane: tbPlane(1))
            .replacingOccurrences(
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
        \(rmsReduce("sv", into: "local_inv[0]", plane: tbPlane(2))
            .replacingOccurrences(
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

    private static let deferredTailChainKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name:
                "gemma4_glue_deferred_expert_tail_chain_2816_bf16_v1_nb1_vec1_rs1"
                + tbSuffix,
            inputNames: [
                "a", "sorted", "inverse", "route_weights", "res",
                "w1", "w2", "w3", "s", "wn",
            ],
            outputNames: ["out", "normed", "rs"],
            source: """
                const uint row = threadgroup_position_in_grid.x;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                \(tbInvDecl(2))
                threadgroup float local_sums[32];\(tbSecondPlane)
                const uint base = row * 2816 + lid * 4;
                const uint wbase = lid * 4;
            \(rmsReduce("a", into: "local_inv[0]", plane: tbPlane(0)))
            \(deferredExpertValuesSource)
            \(rmsReduce("expertv", into: "local_inv[1]", plane: tbPlane(1))
                .replacingOccurrences(
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
            \(rmsReduce("sv", into: "local_inv[0]", plane: tbPlane(2))
                .replacingOccurrences(
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
            \(rmsReduce("outv", into: "local_inv[0]", plane: tbPlane(3))
                .replacingOccurrences(
                    of: "(float)outv[base + i]", with: "(float)outv[i]"))
                const float inv4 = local_inv[0];
                T nq[4];
                for (int i = 0; i < 4; i++) {
                    nq[i] = wn[wbase + i]
                        * static_cast<T>((float)outv[i] * inv4);
                    normed[base + i] = nq[i];
                }
                // RS-CHAIN: the next layer's affine run-sum table entry for
                // this thread's 64-wide group, in the table kernel's order:
                // each octet is float(quadA) + float(quadB) with each quad
                // summed left to right in T (lanes 2f and 2f+1 hold the two
                // quads of octet f), then the same balanced xor tree over
                // the eight octets (lane bits 1..3 here are the table
                // kernel's fm bits 0..2). Sixteen lanes never straddle a
                // simdgroup (704 = 22 x 32).
                const T nquad = ((nq[0] + nq[1]) + nq[2]) + nq[3];
                float rsv = 0.0f + (float)nquad;
                rsv += simd_shuffle_xor(rsv, 1u);
                rsv += simd_shuffle_xor(rsv, 2u);
                rsv += simd_shuffle_xor(rsv, 4u);
                rsv += simd_shuffle_xor(rsv, 8u);
                if ((lid & 15u) == 0u) {
                    rs[row * 44 + (lid >> 4)] = rsv;
                }
            """,
            ensureRowContiguous: true
        )

    private static func admitsDeferred(
        _ expertRows: DeferredWeightedExpertRows
    ) -> Bool {
        expertRows.sortedOutputs.dtype == .bfloat16
            && expertRows.sortedOutputs.shape == [64, axis]
            && expertRows.inverseOrder.dtype == .uint32
            && expertRows.inverseOrder.ndim == 1
            && expertRows.inverseOrder.size == 64
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
        eps: Float
    ) -> (out: MLXArray, normedNext: MLXArray, rs: MLXArray)? {
        guard admits(mlpOut, weight: w1, eps: eps),
            admitsDeferred(expertRows),
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
        let outs = deferredTailChainKernel(
            [
                mlpOut, expertRows.sortedOutputs, expertRows.inverseOrder,
                expertRows.weights, residual, w1, w2, w3, layerScalar,
                nextInputNormWeight,
            ],
            template: [("T", mlpOut.dtype)],
            grid: (rows * tgThreads, 1, 1),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[rows, 1, axis], [rows, 1, axis], [rows, axis / 64]],
            outputDTypes: [.bfloat16, .bfloat16, .float32]
        )
        return (outs[0], outs[1], outs[2])
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
            admitsDeferred(expertRows),
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
        routeScores(zipScores(zipNorm(x)))
    }

    /// PREFILL-PREFIX twin: route from the router norm the branch-prefix
    /// kernel already produced over the post-attention row. The projection
    /// consumes the identical normed array `zipNorm` would have returned, so
    /// the shared tail sees bit-identical scores.
    fileprivate func routePrefilledNorm(
        _ normed: MLXArray
    ) -> (topKIndices: MLXArray, topKWeights: MLXArray) {
        routeScores(zipScores(normed))
    }

    /// The selection tail both entries share: the fused top-8 on the pinned
    /// B=8 decode cell, the finalists kernel on prompt rectangles,
    /// argPartition otherwise, then the takeAlong -> precise softmax ->
    /// per-expert-scale weight chain.
    private func routeScores(
        _ expertScores: MLXArray
    ) -> (topKIndices: MLXArray, topKWeights: MLXArray) {
        // ROUTE-001: single-dispatch byte-identical replacement of the chain
        // below for the B=8 decode geometry. Every other geometry, dtype, or
        // the kill switch falls through to the established chain.
        if let fused = Gemma4FusedRouterTop8.apply(
            expertScores: expertScores, perExpertScale: perExpertScale, topK: topK)
        {
            Gemma4RouterProbe.recorder?(expertScores, fused.indices)
            return (fused.indices, fused.weights)
        }

        // ROUTER-TAIL-PREFILL: the finalists32 selection with the four
        // dependent weight ops (takeAlong -> softmax(precise) ->
        // per-expert-scale gather -> multiply) folded into the same
        // dispatch as a second output, bit-exactly (see
        // Gemma4RouterFinalistsWeightsV1). When it engages it subsumes
        // the selection-only admission below; any gate, shape, dtype,
        // topK, or kth outside the pin falls through to the unchanged
        // stock chains.
        if let tail = Gemma4RouterFinalistsWeightsV1.applyPrefill(
            expertScores, perExpertScale: perExpertScale, topK: topK, kth: kth)
        {
            Gemma4RouterProbe.recorder?(expertScores, tail.indices)
            return (tail.indices, tail.weights)
        }

        // PREFILL-W1 mechanism 2: the finalists32 selection kernel, already
        // bit-identical to the stock `argPartition` + slice pair on the
        // decode cell, admitted here to prompt-rectangle shapes as well.
        // Selection only -- the weight tail below (takeAlong, precise
        // softmax, per-expert scale) stays the identical stock chain on
        // whichever indices were produced.
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

        // Diagnostic-only observability (nil in production; see
        // `Gemma4RouterProbe`). Recording the PRE-selection scores and the
        // selection itself, never altering either.
        Gemma4RouterProbe.recorder?(expertScores, topKIndices)

        return (topKIndices, topKWeights)
    }

    // MARK: ZIP-ROUTER-001 stages
    //
    // Stages retain the independent dense-MLP interleave. FINALISTS-017 can
    // replace only partition+slice with a compact stable tail; the norm,
    // projection, score gather, softmax and scale operations remain stock.

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
        // ROUTER-PREFILL-DEQ-CACHE: at prompt width the router projection
        // reuses the same dequantize-once transposed plane cache the dense
        // projections use (its own admission floor keeps decode rows on the
        // incumbent quantized dispatch). Unquantized or off-contract routers
        // fall through untouched.
        Gemma4PrefillDeqGEMMV1.apply(proj, normed) ?? proj(normed)
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
        // Only the private admitted producer above returns the compact tail.
        // A depends node wrapped around it by ZIP plan2 keeps that shape and
        // remains the returned value, preserving the activated dependency.
        if topK == 8, kth == 120, partition.ndim == 3,
            partition.dim(0) == 8, partition.dim(1) == 1,
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

/// Sparse MoE feed-forward block. Wraps `SwitchGLU` with GeGLU activation.
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
        // Flatten [B, S, H] and always enter SwitchGLU's combined API. It
        // selects direct sorted reduction only for the exact production
        // contract; every other case performs the established unsort + sum.
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = topKIndices.dim(-1)
        let result = switchGLU.callAndWeightedReduceWithUnsortCarrier(
            x.reshaped(B * S, H),
            topKIndices.reshaped(B * S, K),
            weights: topKWeights.reshaped(B * S, K),
            fuseSortedReduction: fuseWeightedUnsort,
            // Ordinary/direct VLM and CBv2 prompt entry points may engage.
            // Rectangular MTP verification explicitly passes false.
            isProductionPrefill: isExpertPrefill,
            // PRENORM-GATHER: the prefill producer of the sorted plane.
            sortedPlane: sortedPlane)
        return Output(
            output: result.output.reshaped(B, S, H),
            unsortCarrier: result.carrier)
    }

    /// Decode-only producer for the fused layer-tail consumer. The promoted
    /// expert projection remains unchanged; only the final inverse-permutation
    /// and weighted reduction are left lazy for the tail kernel.
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

/// Primary joined storage for the dense MLP's two affine-8 input projections.
/// The bound gate/up parameters are zero-copy row slices, while the pinned B8
/// decode path can submit the full 4,224-column plane in one QMV dispatch.
private final class Gemma4DenseGateUpStorage {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let gateWeight: MLXArray
    let gateScales: MLXArray
    let gateBiases: MLXArray
    let upWeight: MLXArray
    let upScales: MLXArray
    let upBiases: MLXArray

    init?(
        gateWeight: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upWeight: MLXArray, upScales: MLXArray, upBiases: MLXArray
    ) {
        let n = 2112
        guard gateWeight.shape == [n, 704], upWeight.shape == [n, 704],
            gateScales.shape == [n, 44], upScales.shape == [n, 44],
            gateBiases.shape == [n, 44], upBiases.shape == [n, 44],
            gateWeight.dtype == .uint32, upWeight.dtype == .uint32,
            gateScales.dtype == .bfloat16, upScales.dtype == .bfloat16,
            gateBiases.dtype == .bfloat16, upBiases.dtype == .bfloat16
        else { return nil }

        weight = concatenated([gateWeight, upWeight], axis: 0)
        scales = concatenated([gateScales, upScales], axis: 0)
        biases = concatenated([gateBiases, upBiases], axis: 0)
        self.gateWeight = weight[..<n]
        self.gateScales = scales[..<n]
        self.gateBiases = biases[..<n]
        self.upWeight = weight[n...]
        self.upScales = scales[n...]
        self.upBiases = biases[n...]
    }
}

private let gemma4DenseGateUpJoinEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_DENSE_GATEUP_JOIN"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// DENSE-GEGLU-EPILOGUE. Fold the dense MLP's GeGLU into the store epilogue
/// of the single prefill GEMM over the paired gate|up dequantized plane.
///
/// Mechanism. On the prefill road the dense gate/up projections are two
/// `Gemma4PrefillDeqGEMMV1` dispatches over separately cached transposed bf16
/// planes, followed by the shaped GeGLU product. This arm builds ONE
/// 16-gate/16-up interleaved `[2816, 4224]` plane (a pure permutation of the
/// same dequantized bytes; no weight is re-quantized or re-represented), runs
/// one `MLX.matmul`, and the specialized kernel epilogue in
/// `steel_gemm_fused[_nax]` closes GeGLU from the paired MMA fragments and
/// stores the compact `[rows, 2112]` plane in the physical prefix of the
/// ordinary `[rows, 4224]` allocation. The second GEMM dispatch, the standalone
/// GeGLU dispatch, and one full read of the activation plane all disappear.
///
/// Exactness. The paired plane is a transposed view of a contiguous
/// `[4224, 2816]` row-major matrix, exactly like the split planes, so the
/// dispatch stays the same transpose_b steel GEMM with the same tile
/// parameters (they depend on dtype/transposes, never on N; 4224 keeps every
/// alignment predicate 2112 satisfies), and every output column's K-chain is
/// bit-identical. The epilogue rounds each accumulator to bfloat16 at the same
/// boundary as the two split stores, then reproduces the compiled GeGLU tape
/// (`gemma4_dense_geglu_compiled_tape`, a dedicated copy of the promoted
/// expert-path function) before storing the compact plane.
///
/// Admission mirrors the kernel-side uniform predicate exactly: nt bf16
/// matmul, per-slice `M >= 512`, `N == 4224`, `K == 2816`, on the deq-GEMM
/// road only (`x.size / 2816 >= Gemma4PrefillDeqGEMMV1.minRows`). Decode,
/// verify, MTP rectangles and sub-floor chunks keep the incumbent paths.
///
/// Kill switch: `DARKBLOOM_GEMMA4_DENSE_GEGLU_EPILOGUE=0` (also false/no/off)
/// never builds the paired plane, so the kernel predicate can never observe
/// its geometry and the incumbent path runs exactly as before.
/// Engage mark: `dense-geglu-epilogue-prefill`.
/// Selects the form in which the paired gate|up prefill plane is handed to the
/// dense GeGLU close: the out-source form, or the plain form.
///
/// `DARKBLOOM_GEMMA4_GEGLU_RESTRICT=0` (or `false`/`no`/`off`) selects the plain
/// form and the incumbent kernel text.
internal let gemma4GeGLURestrictEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_GEGLU_RESTRICT"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4DenseGeGLUEpilogueEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_DENSE_GEGLU_EPILOGUE"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    private var fusedGateUpStorage: Gemma4DenseGateUpStorage?

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let isKvSharedLayer = config.layerUsesSharedKV(layerIdx: layerIdx)
        let useDoubleWide = config.useDoubleWideMlp && isKvSharedLayer
        let intermediateSize = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)

        super.init()
    }

    fileprivate func bindFusedGateUpStorage(_ storage: Gemma4DenseGateUpStorage) {
        fusedGateUpStorage = storage
    }

    // MARK: DENSE-GEGLU-EPILOGUE (prefill)

    private static let pairedPlaneLock = NSLock()
    nonisolated(unsafe) private static var cachedPairedGateUpPlanes:
        [ObjectIdentifier: MLXArray] = [:]

    /// Build the `[2816, 4224]` bf16 plane whose columns interleave one
    /// 16-column gate block with the matching 16-column up block, as a
    /// transposed view of a contiguous `[4224, 2816]` row-major matrix so the
    /// matmul keeps the split planes' transpose_b dispatch. A pure
    /// permutation of the same `dequantized` bytes the split planes hold; no
    /// weight is re-quantized or re-represented.
    private static func buildPairedGateUpPlane(
        gate: QuantizedLinear, gateBiases: MLXArray,
        up: QuantizedLinear, upBiases: MLXArray
    ) -> MLXArray? {
        guard gate.weight.shape == [2112, 704], up.weight.shape == [2112, 704],
            gate.scales.shape == [2112, 44], up.scales.shape == [2112, 44],
            gateBiases.shape == [2112, 44], upBiases.shape == [2112, 44]
        else { return nil }
        let gateDQ = dequantized(
            gate.weight, scales: gate.scales, biases: gateBiases,
            groupSize: gate.groupSize, bits: gate.bits, mode: gate.mode)
        let upDQ = dequantized(
            up.weight, scales: up.scales, biases: upBiases,
            groupSize: up.groupSize, bits: up.bits, mode: up.mode)
        let paired = MLX.stacked(
            [
                gateDQ.reshaped(132, 16, 2816),
                upDQ.reshaped(132, 16, 2816),
            ], axis: 1
        ).reshaped(4224, 2816).transposed()
        eval(paired)
        return paired
    }

    /// Per-layer cache under the same DEQ-PLANE-LOCK-001 discipline as the
    /// split planes; `DARKBLOOM_GEMMA4_PREFILL_DEQ_CACHE=0` restores a
    /// dynamic build on every call.
    private static func pairedGateUpPlane(
        gate: QuantizedLinear, gateBiases: MLXArray,
        up: QuantizedLinear, upBiases: MLXArray
    ) -> MLXArray? {
        guard Gemma4PrefillDeqGEMMV1.cacheEnabled else {
            return buildPairedGateUpPlane(
                gate: gate, gateBiases: gateBiases, up: up, upBiases: upBiases)
        }
        let key = ObjectIdentifier(gate)
        pairedPlaneLock.lock()
        let existing = cachedPairedGateUpPlanes[key]
        pairedPlaneLock.unlock()
        if let existing { return existing }
        guard let paired = buildPairedGateUpPlane(
            gate: gate, gateBiases: gateBiases, up: up, upBiases: upBiases)
        else { return nil }
        pairedPlaneLock.lock()
        if let raced = cachedPairedGateUpPlanes[key] {
            pairedPlaneLock.unlock()
            return raced
        }
        cachedPairedGateUpPlanes[key] = paired
        pairedPlaneLock.unlock()
        return paired
    }

    /// One prefill GEMM over the paired gate|up plane whose kernel epilogue
    /// closes GeGLU and stores the compact `[.., 2112]` activated plane in the
    /// physical prefix of the ordinary `[.., 4224]` output allocation. The
    /// Swift admission mirrors the kernel-side uniform predicate exactly;
    /// a nil keeps the incumbent split road untouched.
    fileprivate func zipPrefillGateUpGeGLU(_ x: MLXArray) -> MLXArray? {
        guard gemma4DenseGeGLUEpilogueEnabled,
            Gemma4PrefillDeqGEMMV1.enabled,
            let gate = gateProj as? QuantizedLinear,
            let up = upProj as? QuantizedLinear,
            gate.bias == nil, up.bias == nil,
            gate.groupSize == 64, up.groupSize == gate.groupSize,
            gate.bits == 8, up.bits == gate.bits,
            gate.mode == .affine, up.mode == gate.mode,
            gate.weight.dtype == .uint32, up.weight.dtype == .uint32,
            gate.weight.shape == [2112, 704], up.weight.shape == [2112, 704],
            gate.scales.dtype == .bfloat16, up.scales.dtype == .bfloat16,
            gate.scales.shape == [2112, 44], up.scales.shape == [2112, 44],
            let gateBiases = gate.biases, gateBiases.dtype == .bfloat16,
            gateBiases.shape == [2112, 44],
            let upBiases = up.biases, upBiases.dtype == .bfloat16,
            upBiases.shape == [2112, 44],
            x.dtype == .bfloat16, x.ndim >= 2,
            x.dim(-1) == 2816, x.dim(-2) >= 512,
            x.size / 2816 >= Gemma4PrefillDeqGEMMV1.minRows,
            let plane = Self.pairedGateUpPlane(
                gate: gate, gateBiases: gateBiases, up: up, upBiases: upBiases)
        else { return nil }
        let joined: MLXArray
        if gemma4GeGLURestrictEnabled {
            let zero = MLXArray.zeros([1, 1], dtype: .bfloat16)
            joined = MLX.addMM(zero, x, plane, alpha: 1.0, beta: 0.0)
            CBv2EngageMark.once("geglu-restrict")
        } else {
            joined = MLX.matmul(x, plane)
        }
        CBv2EngageMark.once("dense-geglu-epilogue-prefill")
        // The specialized store writes the compact plane densely into the
        // first physical half of the ordinary N=4224 allocation. Preserve
        // that physical row stride when exposing the logical N=2112 result;
        // slicing the last axis would retain the allocation's 4224 stride.
        var activatedShape = x.shape
        activatedShape[activatedShape.count - 1] = 2112
        return joined.flattened()[..<(joined.size / 2)].reshaped(activatedShape)
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
        else { return Gemma4PrefillDeqGEMMV1.apply(layer, x) ?? layer(x) }
        return tight
    }

    func callAsFunction(
        _ x: MLXArray,
        activationSums producerSums: CBv2DenseMLPQMVV1.ActivationSums? = nil
    ) -> MLXArray {
        // DENSE-GEGLU-EPILOGUE: the exact prefill geometry closes GeGLU inside
        // the single paired GEMM; every other rectangle falls through.
        if let activated = zipPrefillGateUpGeGLU(x) {
            return denseProjection(downProj, activated)
        }
        // DMLP-002: one exact activation-sum prepass feeds both fallback
        // projections. If either projection is not the pinned affine8 cell,
        // the candidate arrays remain unevaluated and stock takes over.
        let activationSums = producerSums ?? CBv2DenseMLPQMVV1.activationSums(for: x)
        return denseProjection(
            downProj,
            gemma4GeluProduct(
                denseProjection(gateProj, x, activationSums: activationSums),
                denseProjection(upProj, x, activationSums: activationSums)))
    }

    // MARK: ZIP-ROUTER-001 stages
    //
    // `callAsFunction` above, split at its four dependent stages (activation
    // table, gate, up, down) in the same left-to-right order Swift already
    // evaluates them in, so `Gemma4ZipRouterV1` can emit each stage next to
    // the router stage that should run beside it.

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

    /// One decode QMV over the joined gate|up plane. The returned halves are
    /// views with the exact shapes and bytes produced by the two split calls.
    fileprivate func zipGateUp(
        _ x: MLXArray, _ activationSums: CBv2DenseMLPQMVV1.ActivationSums?
    ) -> (gate: MLXArray, up: MLXArray)? {
        guard gemma4DenseGateUpJoinEnabled,
            let storage = fusedGateUpStorage,
            let gate = gateProj as? QuantizedLinear,
            let up = upProj as? QuantizedLinear,
            gate.bias == nil, up.bias == nil,
            gate.groupSize == 64, up.groupSize == gate.groupSize,
            gate.bits == 8, up.bits == gate.bits,
            gate.mode == .affine, up.mode == gate.mode,
            let joined = CBv2DenseMLPQMVV1.matmul(
                x: x,
                weight: storage.weight,
                scales: storage.scales,
                biases: storage.biases,
                groupSize: gate.groupSize,
                bits: gate.bits,
                mode: gate.mode,
                activationSums: activationSums)
        else { return nil }
        CBv2EngageMark.once("dense-gateup-join")
        return (
            joined[.ellipsis, ..<2112],
            joined[.ellipsis, 2112...]
        )
    }

    fileprivate func zipDown(_ activated: MLXArray) -> MLXArray {
        denseProjection(downProj, activated)
    }
}

/// ZIP-ROUTER-001 -- interleave the MoE layer's router chain with the
/// independent dense-MLP chain in the ENCODE ORDER, changing no arithmetic.
///
/// Mechanism. MLX's Metal command encoder is concurrent: `dispatch_threadgroups`
/// calls `maybeInsertBarrier`, which emits a global
/// `memoryBarrier(BarrierScopeBuffers)` only when the dispatch being encoded
/// reads a buffer written since the last barrier, or writes one read since it
/// (`backend/metal/device.cpp`, `set_input_array` / `register_output_array` /
/// `maybeInsertBarrier`). Consecutive independent dispatches therefore share a
/// barrier stage and run concurrently on the GPU.
///
/// The MoE layer holds two chains that are independent of each other: the
/// router chain (rms norm -> 128-row affine QMV -> argPartition -> selected
/// slice) is a latency chain of small dispatches -- the QMV walks 22 dependent
/// K blocks over a 0.38 MB plane -- while the dense-MLP chain (activation-sum
/// table -> gate/up -> GeLU product -> down) is bandwidth work on 6.32 MB
/// planes. In the stock tape they never overlap: MLX evaluates the graph in
/// reverse breadth-first order from the layer tail (`transforms.cpp`,
/// `eval_impl`), and the dense branch sits behind the expert branch, so the
/// router chain runs with nothing underneath it and the dense chain runs after
/// the expert kernels.
///
/// This zips them. `MLX.depends(input:dependencies:)` (`Ops.swift`) builds a
/// `Depends` primitive whose `eval` is `copy_shared_buffer` -- no dispatch, no
/// copy, the output IS the input's buffer with the input's shape, strides and
/// flags (`backend/common/common.cpp`). Naming it as an ordering edge in both
/// directions -- router stage k+1 depends on dense stage k, dense stage k+1
/// depends on router stage k -- makes the two chains strictly alternate in any
/// topological order, so the encoder pairs them into barrier stages whose cost
/// is the max of the pair instead of the sum.
///
/// Exactness. Every kernel receives the identical operand buffer it receives
/// today: `Depends` aliases, and because its output shares the input's
/// `array::Data` the shared buffer is never donatable, so no downstream
/// primitive can write through the alias. No expression is re-associated, no
/// dispatch is added or removed by the ordering edges themselves. The optional
/// FINALISTS-017 stage separately replaces partition+slice with the same
/// ordered eight indices; its score/weight tail stays stock. Kill switch
/// `DARKBLOOM_GEMMA4_ZIP_ROUTER=0` restores the stock
/// call; every geometry outside the pinned B=8 decode cell fails closed onto
/// it because the dense activation table (`CBv2DenseMLPQMVV1.activationSums`)
/// returns nil there.
// ZIP+MASK current-crown retry marker (newjordan r2); executable path retains the twice-sealed T25 schedule.
private enum Gemma4ZipRouterV1 {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_ZIP_ROUTER"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Tape plan, a bisection knob only -- the ranked run sets no environment
    /// and always takes the default. `1` (default) is the shipped pairing:
    /// (router QMV | dense gate+up), (dense GeLU), (argPartition | dense
    /// down), (selected slice), with the expert branch fenced behind the
    /// dense chain so the down projection stays inside the zip. `2` slides
    /// the argPartition one stage earlier, pairing it with the 2 us GeLU
    /// product instead of the 25 us down projection (measured 0.02 ms/step
    /// worse and noisier). `0` drops the expert fence, which lets the tape
    /// float the dense down back behind the expert kernels.
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
        /// GLUE-FOLD: the route table emitted beside the top-8 selection, or
        /// nil when the incumbent two-dispatch chain produced the indices.
        let routeTable: SwitchRouteTable?
    }

    /// PREFIX-001 admission lives beside the ZIP admission so the eager
    /// custom producer is never built unless this exact consumer will use all
    /// of its outputs.
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
        // Pure shape predicate first: no graph node exists until every pin
        // below holds, so prefill, MTP rectangles and any other cohort walk
        // away from here without having built anything.
        guard enabled, router.zipAdmits,
            out.ndim == 3, out.dim(0) == 8, out.dim(1) == 1,
            out.dtype == .bfloat16
        else { return nil }

        // Stage 0, shared: the two pre-norms plus the exact dense activation
        // table. A nil here means this is not the fused-glue cell and no node
        // was built.
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

        // Stage 1: router norm. The table is normally producer-emitted in
        // stage 0; the standalone producer survives only as the fail-closed
        // fallback for a disabled or mismatched carrier. A nil leaves the
        // dual pre-norm arrays unreferenced, so MLX never evaluates them and
        // the caller's stock path rebuilds the identical pair.
        let normed = carriedRouterNorm ?? router.zipNorm(out)

        // Stage 2: router QMV | dense gate + up.
        let expertScores: MLXArray
        let gate: MLXArray
        let up: MLXArray
        if Gemma4FusedLayerGlue.denseXSumElideEnabled {
            expertScores = router.zipScores(normed)
            let denseIn = MLX.depends(input: n1, dependencies: [normed])
            if let joined = mlp.zipGateUp(denseIn, nil) {
                (gate, up) = joined
            } else {
                gate = mlp.zipGate(denseIn, nil)
                up = mlp.zipUp(denseIn, nil)
            }
        } else {
            guard let sums = producerSums ?? mlp.zipActivationSums(n1) else { return nil }
            expertScores = router.zipScores(
                MLX.depends(input: normed, dependencies: [sums.dependencyHandle]))
            let denseIn = MLX.depends(input: n1, dependencies: [normed])
            if let joined = mlp.zipGateUp(denseIn, sums) {
                (gate, up) = joined
            } else {
                gate = mlp.zipGate(denseIn, sums)
                up = mlp.zipUp(denseIn, sums)
            }
        }

        // Stage 3: the dense GeLU product, which the router has no partner
        // for -- the argPartition is deliberately NOT paired with it.
        let held = MLX.depends(inputs: [gate, up], dependencies: [expertScores])
        let activated = gemma4GeluProduct(held[0], held[1])

        // Stage 4: router argPartition | dense down projection. The sort is
        // 8 us and the down projection 25 us, so this is the pairing that
        // pays; the selected slice then trails alone at 1.7 us.
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
            // GLUE-FOLD: one dispatch emits the top-8 selection, in-register
            // float32 softmax + per-expert scaling, AND the sorted route table,
            // so the standalone rank-scatter launch and the weight tail
            // (takeAlong -> softmax -> perExpertScale) leave the dependent chain.
            // The fold consumes the identical fenced scores node the incumbent
            // partition consumed, keeping the tape's ordering edges unchanged.
            // Fail-closed onto the incumbent finalists + slice + weight tail.
            if let fold = Gemma4RouteGlueFoldV1.apply(
                MLX.depends(input: expertScores, dependencies: [denseOut]),
                perExpertScale: router.perExpertScale,
                topK: router.topK, kth: router.kth)
            {
                topKIndices = fold.indices
                topKWeights = fold.weights
                routeTable = fold.table
            } else {
                let partition = router.zipPartition(
                    MLX.depends(input: expertScores, dependencies: [denseOut]))
                topKIndices = router.zipSelected(partition)
                topKWeights = router.zipWeights(
                    expertScores: expertScores, topKIndices: topKIndices)
            }
        }

        Gemma4RouterProbe.recorder?(expertScores, topKIndices)

        // Every expert dispatch already sits behind `topKIndices`; fencing the
        // expert branch's activation behind the dense chain's last stage as
        // well keeps the down projection inside the zip instead of letting the
        // reverse-BFS tape float it past the expert kernels.
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
        nextInputLayernormWeight: MLXArray? = nil,
        enableAttentionBranchPrefix: Bool = false
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
        var carriedRunsum: MLXArray? = nil
        if let chain = glueChain, let pending = chain.pending,
            pending.source === x
        {
            chain.pending = nil
            h = pending.normed
            carriedRunsum = pending.rs
        } else if cache is any CBv2AttendingLayerCache,
            let produced = Gemma4FusedLayerGlue.inputNormWithQKVRunsum(
                x: x, weight: inputLayernorm.weight, eps: config.rmsNormEps)
        {
            glueChain?.pending = nil
            h = produced.normed
            carriedRunsum = produced.qkvRunsumTable
        } else {
            glueChain?.pending = nil
            h = inputLayernorm(x)
        }
        let (attnOut, kvPair, attnPositionOffset) = selfAttn(
            h, mask: mask, cache: cache, sharedKV: sharedKV, positionOffset: positionOffset,
            v2SharedSource: v2SharedSource, outputStart: outputStart,
            useLastQueryPrefill: useLastQueryPrefill, carriedRunsum: carriedRunsum)
        // PREFIX-001: only build the joined producer when the ZIP consumer is
        // guaranteed to accept it. A nil leaves the established attention
        // residual and branch pre-norm paths untouched.
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
        // PREFILL-PREFIX twin: when it engages, the router and the dense
        // pre-norm consumers below read the norms it already produced
        // instead of re-reducing the same post-attention row.
        var prefillBranchPrefix: Gemma4PrefillGlueV1.AttentionBranchPrefix? = nil
        if let attentionBranchPrefix {
            out = attentionBranchPrefix.out
        } else if let fusedOut = Gemma4FusedLayerGlue.normResidual(
            x: attnOut, residual: residual,
            weight: postAttentionLayernorm.weight, eps: config.rmsNormEps)
        {
            out = fusedOut
        } else if isMoE, let router, let preFeedforwardLayernorm2,
            let prefix = Gemma4PrefillGlueV1.attentionBranchPrefix(
                attn: attnOut,
                residual: residual,
                wPostAttn: postAttentionLayernorm.weight,
                wDense: preFeedforwardLayernorm.weight,
                wRouter: router.zipEffectiveScale(),
                eps: config.rmsNormEps)
        {
            prefillBranchPrefix = prefix
            out = prefix.out
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
            let h1Raw: MLXArray
            let expertBranch: (
                raw: MLXArray?,
                deferred: DeferredWeightedExpertRows?,
                unsortCarrier: WeightedExpertUnsortCarrier?
            )
            // The deferred carrier has a consumer only when the decode tail
            // may also fold the layer scalar. PLE geometries select the
            // complete established expert reduction immediately.
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

            // ZIP-ROUTER-001: emit the router chain and the dense chain
            // interleaved so the encoder pairs them into shared barrier
            // stages. Returns nil for every geometry but the pinned B=8
            // decode cell, and under the kill switch.
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
                // PREFILL-PREFIX: the branch-prefix kernel already produced
                // the router norm over this exact `out`; only the projection
                // and the shared selection tail run.
                let topKIndices: MLXArray
                let topKWeights: MLXArray
                if let prefix = prefillBranchPrefix {
                    let routed = router.routePrefilledNorm(prefix.routerNorm)
                    topKIndices = routed.topKIndices
                    topKWeights = routed.topKWeights
                } else {
                    let routed = router(out)
                    topKIndices = routed.topKIndices
                    topKWeights = routed.topKWeights
                }

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
                    // PREFILL-PREFIX: the dense pre-norm arrived with `out`.
                    let n1 = prefillBranchPrefix?.denseNorm
                        ?? Gemma4PrefillGlueV1.preNorm(
                            x: out,
                            weight: preFeedforwardLayernorm.weight,
                            eps: config.rmsNormEps),
                    let n2 = Gemma4PrefillGlueV1.preNorm(
                        x: out,
                        weight: preFeedforwardLayernorm2.weight,
                        eps: config.rmsNormEps)
                {
                    // PRENORM-GATHER: the expert pre-norm is written straight
                    // into expert-sorted order by the producer below, from
                    // the residual and the sort's inverse order, so the
                    // un-sorted expert norm and the standalone gather of it
                    // leave the prefill plane. `n2` is that un-sorted norm as
                    // a lazy fallback: it is dispatched only if SwitchGLU
                    // declines the producer, and never otherwise.
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
                chain.pending = (source: chained.out, normed: chained.normedNext, rs: chained.rs)
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
                    chain.pending = (source: chained.out, normed: chained.normedNext, rs: nil)
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
                    chain.pending = (source: chained.out, normed: chained.normedNext, rs: nil)
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
                    chain.pending = (source: chained.out, normed: chained.normedNext, rs: nil)
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

        // PLE gating
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let perLayerInput = activePerLayerInput
        {
            let residual3 = out
            // Same compiled graph the dense MLP site already uses, and it
            // rounds in the same places as the two-dispatch form: `compile`
            // keeps each node at its own dtype, so the activated intermediate
            // is materialised bf16 either way.
            var g = gemma4GeluProduct(gate(out), perLayerInput)
            g = proj(g)
            // Same `residual + rmsNorm(x, w)` at 2816 and the same eps as the
            // post-attention site, so the prefill fusion applies unchanged.
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

/// EMB-001 (revived). The trunk entry evaluates
///
///     h = embedTokens(inputs) * embedScale
///
/// which on this checkpoint is FIVE dependent GPU operations over a
/// `QuantizedEmbedding`: three row gathers (packed uint32 weight, bf16 scales,
/// bf16 biases), the generic `affine_dequantize` kernel, and finally a separate
/// full-width elementwise multiply by `sqrt(hiddenSize)`. On the scored prefill
/// rectangle `[8, 1024]` that materialises a 11.5 MB gathered-weight
/// intermediate, a 45 MB dequantized intermediate, and then reads and rewrites
/// that 45 MB once more just to apply one scalar.
///
/// This enum collapses the whole chain into one kernel: each thread owns one
/// packed uint32 word of one vocabulary row, reads the single (scale, bias)
/// pair its 8 codes share, and writes 8 already-scaled bf16 features. Nothing
/// is gathered into a temporary; the only traffic is the packed row in and the
/// finished hidden state out.
///
/// ## Bit-exactness
///
/// The kernel is a transcription of the two rounding boundaries the stock
/// chain has, in the same order:
///
///  1. `affine_dequantize` (`mlx-generated/metal/quantized.h`) is instantiated
///     at `T = out.dtype()`, and `ops.cpp` infers that dtype as
///     `result_type(scales, biases)` — bf16 here. Its body is literally
///     `out[i] = scale * d + bias;` with `uint8_t d = (val >> (bits*i)) & 0x0f`
///     and `gindex = oindex / group_size`. The same expression, the same
///     operand types, and the same store-to-`T` rounding are reproduced below.
///  2. `MLXArray * Float` (`MLXArray+Ops.swift`) forwards through
///     `ScalarOrArray.asMLXArray(dtype: lhs.dtype)`, so the Float scale is
///     ROUNDED TO bf16 BEFORE the multiply — `sqrt(2816)` becomes exactly
///     `53.0`. The multiplier below is built by calling that very same
///     `asMLXArray(dtype:)`, so the constant cannot drift from the stock one.
///
/// The affine expression is therefore never reassociated with the embedding
/// scale and the intermediate is never promoted to float: `dequantized` is a
/// named `T` value, which pins the first rounding exactly where stock puts it.
/// Negative token ids are wrapped by the vocabulary size, matching MLX's
/// `offset_neg_idx` gather semantics; out-of-range positive ids are undefined
/// in both paths.
///
/// ## Gating (PLE-GLUE-028 lesson)
///
/// The kernel is geometry-agnostic: `words_per_row` and `row` are read off
/// the launched grid, not off a compile-time or host-passed `L`, so nothing
/// in its body assumes a prefill-sized rectangle. It was nonetheless
/// admitted for the PREFILL rectangle only (`L > 1`) at first, on the
/// PLE-GLUE-028 lesson that at `[B, 1]` the whole chain produces only 22,528
/// values and a custom-kernel launch is not reliably cheaper than the five
/// small dispatches it replaces — the same trap `Gemma4FusedRouterTop8` fell
/// into.
///
/// The decode cell (`L == 1`) is admitted as well below, behind its own
/// independent switch, since the argument above is a dispatch-count
/// tradeoff, not a correctness one: the same two-boundary exactness argument
/// applies unchanged regardless of `L`.
///
/// Kill switch: `DARKBLOOM_GEMMA4_SCALED_EMBEDDING=0` (also `false`/`no`/`off`)
/// restores the stock expression on the same binary for every geometry.
/// `DARKBLOOM_GEMMA4_SCALED_EMBEDDING_DECODE=0` restores the stock expression
/// for the decode cell only, leaving the prefill admission untouched.
///
/// Internal rather than file-private only so the local full-vocabulary parity
/// test can drive this exact kernel instead of a transcription of it.
enum Gemma4FusedScaledEmbedding {
    /// DEFAULT ON for the prefill rectangle.
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_SCALED_EMBEDDING"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// DEFAULT ON for the `[B, 1]` decode cell, independent of `enabled`
    /// above so either admission can be disabled without touching the
    /// other.
    static let decodeEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_SCALED_EMBEDDING_DECODE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// This checkpoint's embedding quantization. Anything else fails closed.
    private static let groupSize = 64
    private static let bits = 4
    /// 32 / 4: affine-4 codes packed per uint32 word.
    private static let codesPerWord = 8
    /// 64 / 8: packed words covered by one (scale, bias) pair.
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

    /// Returns the scaled hidden state, or `nil` when any pin fails — the
    /// caller then evaluates the pre-existing expression unchanged.
    static func apply(
        tokens: MLXArray, embedding: Embedding, embedScale: Float, hiddenSize: Int
    ) -> MLXArray? {
        guard enabled,
            tokens.ndim == 2,
            tokens.dtype == .int32,
            // Prefill rectangle admits unconditionally; the [B, 1] decode
            // cell admits behind its own independent switch.
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
            // `asMLXArray(dtype:)` is the exact conversion the stock
            // `MLXArray * Float` overload performs on the scalar.
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

/// FINAL-NORM-XSum. The ranked tied head consumes one exact affine activation
/// sum for each `[row, 64-wide hidden group]`. The stock path materializes the
/// final BF16 RMSNorm output and then launches a second kernel that rereads all
/// 22,528 values to build 352 sums. This producer writes the same norm output
/// and publishes the same sums while those BF16 values are still resident.
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
            x.size == rows * axis,
            weight.dtype == .bfloat16,
            weight.ndim == 1,
            weight.dim(0) == axis
        else { return nil }

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

    /// Ordinary target forward with an optional producer-side affine-sum
    /// carrier for the ranked tied head. Every non-B8x1 geometry retains the
    /// established final RMSNorm and returns a nil carrier.
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
        forceArrayMask requestedArrayMask: Bool = false,
        emitMMAHeadSums: Bool = false
    ) -> (
        postNorm: MLXArray,
        preNorm: MLXArray?,
        mmaHeadSums: Gemma4MMAQuantizedGEMV.ActivationSums?
    ) {
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
            if let fused = Gemma4FusedScaledEmbedding.apply(
                tokens: inputs, embedding: embedTokens, embedScale: embedScale,
                hiddenSize: config.hiddenSize)
            {
                h = fused
            } else {
                h = embedTokens(inputs) * embedScale
            }
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
        // layer L+1 its input norm through it.
        let glueChain = Gemma4GlueChainBox()
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
                    ? layers[idx + 1].inputLayernorm.weight : nil,
                enableAttentionBranchPrefix:
                    isCBv2 && !schedulePrefill
                    && inputBatchSize == 8 && inputLength == 1
                    && !capturePreNorm && dFlashHiddenCapture == nil
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
        // The VLM omission profile uses zero to represent the former optional
        // softcap's nil/disabled state.
        //
        // SOFTCAP-SKIP: `tanh(x / cap) * cap` is strictly increasing, so it
        // cannot reorder the vocabulary axis. When the engine has declared
        // that this step's logits are consumed for their order alone (every
        // row greedy, no logprobs, bias or penalties), the emitted token is
        // identical with or without it and the dispatch is pure overhead —
        // one transcendental pass over the whole vocabulary plus the float32
        // widening the untyped cap forces on the tensor the sampler reads.
        if config.finalLogitSoftcapping > 0, !CBv2OrderOnlyLogits.engaged {
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
        fuseDenseGateUpStorage(&sanitized)
        fuseExpertGateUpStorage(&sanitized)
        return sanitized
    }

    /// Make each dense layer's joined gate|up plane its primary storage.
    /// Split module parameters remain zero-copy row views, while the exact B8
    /// decode path can issue one 4,224-column QMV instead of two 2,112-column
    /// launches. Any shape or dtype mismatch leaves that layer untouched.
    private func fuseDenseGateUpStorage(_ sanitized: inout [String: MLXArray]) {
        guard gemma4DenseGateUpJoinEnabled else { return }
        let gateWeightSuffix = ".mlp.gate_proj.weight"
        for key in sanitized.keys where key.hasSuffix(gateWeightSuffix) {
            let base = String(key.dropLast(gateWeightSuffix.count))
            guard let layerIdx = extractLayerIdx(from: key),
                layerIdx < model.layers.count,
                let gateWeight = sanitized["\(base).mlp.gate_proj.weight"],
                let gateScales = sanitized["\(base).mlp.gate_proj.scales"],
                let gateBiases = sanitized["\(base).mlp.gate_proj.biases"],
                let upWeight = sanitized["\(base).mlp.up_proj.weight"],
                let upScales = sanitized["\(base).mlp.up_proj.scales"],
                let upBiases = sanitized["\(base).mlp.up_proj.biases"],
                let storage = Gemma4DenseGateUpStorage(
                    gateWeight: gateWeight, gateScales: gateScales,
                    gateBiases: gateBiases, upWeight: upWeight,
                    upScales: upScales, upBiases: upBiases)
            else { continue }
            sanitized["\(base).mlp.gate_proj.weight"] = storage.gateWeight
            sanitized["\(base).mlp.gate_proj.scales"] = storage.gateScales
            sanitized["\(base).mlp.gate_proj.biases"] = storage.gateBiases
            sanitized["\(base).mlp.up_proj.weight"] = storage.upWeight
            sanitized["\(base).mlp.up_proj.scales"] = storage.upScales
            sanitized["\(base).mlp.up_proj.biases"] = storage.upBiases
            model.layers[layerIdx].mlp.bindFusedGateUpStorage(storage)
        }
    }

    /// GATEUP-FUSE-PREFILL: make the concatenated gate|up right-hand side the
    /// primary storage of every routed-expert layer at load. The layer's
    /// `gate_proj` / `up_proj` weight, scales and biases become zero-copy row
    /// slices of that storage (see ``SwitchGateUpFusedStorage``), so the bound
    /// split parameters read the identical bytes with no second copy, and the
    /// sorted prefill plane dispatches one gather over the whole storage.
    /// Layers or checkpoints outside the exact production geometry, and the
    /// arm's off-state, leave the loaded split arrays untouched.
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

/// Every storage-owning CBv2 attention result is consumed by the sequential
/// Gemma trunk and final LM head, so ordinary decode logits transitively root
/// that forward's K/V mutations. Cache-layout gates remain in the adapter.
extension Gemma4TextModel: CBv2LanguageModelDecodeOutputCoversCacheMutations {}

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

// Ranked resample marker 2: this archive is a further ranked sample of the tree carried
// by the preceding ranked submission of this content apart from any rotation item declared in its note.

// Ranked resample marker 2: this archive is a further ranked sample of the tree carried
// by the preceding ranked submission of this content apart from any rotation item declared in its note.

// Ranked resample marker 2: this archive is a further ranked sample of the tree carried
// by the preceding ranked submission of this content apart from any rotation item declared in its note.

// MARK: - LGH-001 --- logitsless greedy head

/// Cross-check every fused token against the logits the stock chain would have
/// produced. Costs a host sync per step, so it is a diagnostic, never a mode
/// the benchmark runs in.
private let gemma4LogitslessHeadVerify: Bool = gemma4TruthyFlag(
    ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_LOGITSLESS_HEAD_VERIFY"])

/// Off only on an explicit off value, so the fold is the default road and the
/// switch restores the stock final norm plus the standalone sum prepass.
private let gemma4DecodeHeadNormXSumFoldEnabled: Bool = {
    guard
        let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_DECODE_HEAD_NORM_XSUM_FOLD"]
    else { return true }
    switch raw.trimmingCharacters(in: .whitespaces).lowercased() {
    case "0", "false", "no", "off": return false
    default: return true
    }
}()

/// The tied head can answer the chained decode step with token ids alone.
///
/// The values the fused kernel compares are the bf16 the MMA head would have
/// stored, and the final softcap `tanh(x / c) * c` is strictly increasing for
/// every `c >= 0`, so the fused top-1 is the stock argmax including its
/// first-index-wins tie rule. See `Gemma4MMAQuantizedGEMV.applyArgmax`.
extension Gemma4TextModel: CBv2ArgmaxDecodeForwardable {

    public func cbv2AdmitsArgmaxDecode(_ tokens: MLXArray) -> Bool {
        guard tokens.ndim == 2, tokens.dim(1) == 1 else { return false }
        guard config.finalLogitSoftcapping >= 0 else { return false }
        guard lmHead == nil,
            let quantized = model.embedTokens as? QuantizedEmbedding,
            quantized.mode == .affine
        else { return false }
        return Gemma4MMAQuantizedGEMV.admitsArgmax(
            x: [tokens.dim(0), 1, config.hiddenSize],
            xDType: .bfloat16,
            w: quantized.weight,
            scales: quantized.scales,
            biases: quantized.biases,
            groupSize: quantized.groupSize,
            bits: quantized.bits)
    }

    public func cbv2DecodeArgmax(_ tokens: MLXArray, caches: [KVCache]) -> MLXArray {
        // The greedy road's serial tail is final RMSNorm, the head's affine
        // activation-sum prepass, the fused head+argmax, then the reduce. The
        // tree already carries a producer that emits the first two together
        // and the logits entry point already takes it at this exact geometry;
        // only this path was still paying for both dispatches.
        let hidden: MLXArray
        let carriedSums: Gemma4MMAQuantizedGEMV.ActivationSums?
        if gemma4DecodeHeadNormXSumFoldEnabled,
            lmHead == nil,
            tokens.ndim == 2,
            tokens.dim(0) == 8,
            tokens.dim(1) == 1,
            let quantized = model.embedTokens as? QuantizedEmbedding,
            quantized.mode == .affine,
            quantized.groupSize == 64,
            quantized.bits == 4,
            Gemma4MMAQuantizedGEMV.consumesActivationSums
        {
            let produced = model.callWithMMAHeadSums(tokens, cache: caches)
            hidden = produced.postNorm
            carriedSums = produced.activationSums
        } else {
            hidden = model(tokens, cache: caches)
            carriedSums = nil
        }
        let rows = tokens.dim(0)
        guard lmHead == nil,
            let quantized = model.embedTokens as? QuantizedEmbedding,
            quantized.mode == .affine,
            let fused = Gemma4MMAQuantizedGEMV.applyArgmax(
                x: hidden,
                w: quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                activationSums: carriedSums)
        else {
            return applyLMHead(hidden).argMax(axis: -1).asType(.int32).reshaped([rows])
        }
        CBv2EngageMark.once("logitsless-greedy-head")
        if gemma4LogitslessHeadVerify {
            let stock = applyLMHead(hidden).argMax(axis: -1).asType(.int32).reshaped([rows])
            let disagreements = sum(notEqual(fused, stock)).item(Int.self)
            // The one place the fused comparison could diverge is a softcap
            // that maps two DISTINCT stored bf16 logits onto one float; that
            // needs |logit| in the hundreds, so the observed peak is the
            // margin. Reported alongside every verified step.
            let raw = Gemma4MMAQuantizedGEMV.apply(
                x: hidden, w: quantized.weight, scales: quantized.scales,
                biases: quantized.biases, groupSize: quantized.groupSize,
                bits: quantized.bits)!
            let peak = max(abs(raw.asType(.float32))).item(Float.self)
            let report =
                "[lgh] verify mismatch=\(disagreements) max_abs_logit=\(peak)"
                + (disagreements == 0
                    ? "\n"
                    : " fused=\(fused.asArray(Int32.self)) "
                        + "stock=\(stock.asArray(Int32.self))\n")
            FileHandle.standardError.write(Data(report.utf8))
        }
        return fused
    }
}

// Ranked resample marker 3: this archive is a further ranked sample of the tree carried
// by the preceding ranked submission of this content apart from any rotation item declared in its note.

// Ranked resample marker 36: this archive is a further ranked sample of the tree carried
// by the preceding ranked submission of this content apart from any rotation item declared in its note.
