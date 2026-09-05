import CoreFoundation
import Foundation
import MLXFastCore

/// The layer-type names below are taken from the model's configuration
/// document exactly as written; nothing here recomputes them.
/// Attention kind of one decoder layer, as spelled in `layer_types`.
public enum Gemma4A4BLayerType: String, Equatable, Sendable {
    case sliding = "sliding_attention"
    case full = "full_attention"
}

/// Per-layer-type RoPE spec. Gemma 4 has NO single top-level RoPE block: the
/// two attention kinds carry independent parameters, and the full-attention
/// entry additionally carries a partial rotary factor the sliding one omits.
/// Modelling this as one spec with an optional partial factor (the shape the
/// Qwen tower used) would silently accept a config that specified the wrong
/// theta for one of the two kinds.
public struct Gemma4A4BRopeSpec: Equatable, Sendable {
    public let theta: Double
    public let type: String
    /// Present on `full_attention` only; `nil` on `sliding_attention`.
    public let partialRotaryFactor: Double?

    public init(theta: Double, type: String, partialRotaryFactor: Double?) {
        self.theta = theta
        self.type = type
        self.partialRotaryFactor = partialRotaryFactor
    }
}

/// One resolved quantization setting.
public struct Gemma4A4BQuantizationSpec: Equatable, Sendable {
    public let groupSize: Int
    public let bits: Int

    public init(groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
    }
}

/// The checkpoint's quantization contract: a fallback spec plus the per-tensor
/// overrides that promote specific paths to a different width.
///
/// WHY THIS TYPE EXISTS, and it is the single most important thing in this
/// file. The pinned checkpoint's `quantization` block is NOT the three-key
/// object every previous target in this repository carried. It is affine
/// 4-bit / group-64 PLUS 120 per-tensor overrides promoting four projection
/// families to 8 bits on every one of the 30 layers.
///
/// The Qwen-era contract (`Qwen35Config.qwenQuantization`) read exactly
/// `{group_size, bits, mode}` and ignored every other key. Against this
/// checkpoint that parse SUCCEEDS and silently discards all 120 overrides,
/// after which the runtime quantizes 120 tensors at 4 bits that were written
/// at 8. Nothing downstream catches it: the tensor names are right, the shapes
/// are right, and only the numerics are wrong. That is why the overrides are
/// modelled as data here rather than validated away, and why
/// `Gemma4A4BRuntimeWeightCache` resolves the width PER PATH instead of
/// passing one triple to `quantize(model:)`.
public struct Gemma4A4BQuantization: Equatable, Sendable {
    /// Applies to any path not named in `overrides`.
    public let fallback: Gemma4A4BQuantizationSpec
    /// Quantization mode; affine for this checkpoint. Overrides do not carry a
    /// mode of their own, so this applies to fallback and overrides alike.
    public let mode: String
    /// Tensor path -> width. Keys are checkpoint paths as they appear in the
    /// source config, e.g. `language_model.model.layers.0.mlp.gate_proj`.
    public let overrides: [String: Gemma4A4BQuantizationSpec]

    public init(
        fallback: Gemma4A4BQuantizationSpec,
        mode: String,
        overrides: [String: Gemma4A4BQuantizationSpec]
    ) {
        self.fallback = fallback
        self.mode = mode
        self.overrides = overrides
    }

    /// Resolve the width for one checkpoint path.
    public func spec(forPath path: String) -> Gemma4A4BQuantizationSpec {
        overrides[path] ?? fallback
    }
}

/// Frozen text-tower contract for
/// `mlx-community/gemma-4-26B-A4B-it-qat-4bit@0e3cbab38ce568cf6e23543010d08d03b731910c`.
///
/// The transformed `weights/config.json` is the source checkpoint's
/// `text_config` flattened to the top level plus the checkpoint-wide
/// `quantization` block, so this type parses the flattened schema. The
/// `Gemma4A4B` prefix is deliberate: the bare `Gemma4*` names belong to the
/// vendored reference implementation and must not be shadowed, and the
/// transform still carries a separate legacy `.gemma4` family for the archived
/// dense 31B multimodal layout.
public struct Gemma4A4BConfig: Equatable, Sendable {
    public let modelType: String
    public let vocabSize: Int
    public let vocabSizePerLayerInput: Int
    public let hiddenSize: Int
    public let hiddenSizePerLayerInput: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numGlobalKeyValueHeads: Int
    public let numKvSharedLayers: Int
    public let headDim: Int
    public let globalHeadDim: Int
    public let slidingWindow: Int
    public let layerTypes: [Gemma4A4BLayerType]
    public let rmsNormEps: Double
    public let hiddenActivation: String
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    public let attentionDropout: Double
    public let attentionKeqV: Bool
    public let finalLogitSoftcapping: Double
    public let tieWordEmbeddings: Bool
    public let enableMoeBlock: Bool
    public let numExperts: Int
    public let topKExperts: Int
    public let moeIntermediateSize: Int
    public let useDoubleWideMlp: Bool
    public let useBidirectionalAttention: String
    public let dtype: String
    public let useCache: Bool
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let padTokenId: Int
    public let initializerRange: Double
    public let slidingRope: Gemma4A4BRopeSpec
    public let fullRope: Gemma4A4BRopeSpec
    public let quantization: Gemma4A4BQuantization

    /// The pinned layer schedule: 30 layers on a six-layer repeat, the LAST
    /// layer of each group global. Indices 5, 11, 17, 23, 29.
    public static let expectedLayerTypes: [Gemma4A4BLayerType] =
        (0..<MLXFastConstants.numHiddenLayers).map {
            $0 % MLXFastConstants.fullAttentionInterval
                == MLXFastConstants.fullAttentionInterval - 1 ? .full : .sliding
        }

    /// Every key the transformed config must carry, non-null. Read off the
    /// pinned revision's own `text_config`, which has exactly these 36 and no
    /// null values.
    ///
    /// SINGLE-SOURCED from `MLXFastCore.Gemma4A4BConfigKeys.required` (rather
    /// than restated here) so this loader and the trusted runtime-worker
    /// pinned-configuration gate -- which cannot import this module, see that
    /// gate's own doc comment -- cannot silently diverge on which keys this
    /// target's config carries.
    static let requiredKeys: Set<String> = Gemma4A4BConfigKeys.required

    /// Keys that must be ABSENT or null. `moe_router_logit_softcapping` is the
    /// Gemma-family knob that would change router numerics if it ever appeared;
    /// `qkv_bias` and `query_pre_attn_scalar` likewise change attention.
    /// Carrying the check forward from the Laguna/Qwen contract shape: a key
    /// that silently appears is exactly as dangerous as one that disappears.
    /// Single-sourced from `MLXFastCore.Gemma4A4BConfigKeys.forbidden`, same
    /// reason as `requiredKeys` above.
    static let forbiddenKeys: [String] = Gemma4A4BConfigKeys.forbidden

    public init(
        modelType: String, vocabSize: Int, vocabSizePerLayerInput: Int,
        hiddenSize: Int, hiddenSizePerLayerInput: Int, intermediateSize: Int,
        numHiddenLayers: Int, numAttentionHeads: Int, numKeyValueHeads: Int,
        numGlobalKeyValueHeads: Int, numKvSharedLayers: Int, headDim: Int,
        globalHeadDim: Int, slidingWindow: Int,
        layerTypes: [Gemma4A4BLayerType], rmsNormEps: Double,
        hiddenActivation: String, maxPositionEmbeddings: Int,
        attentionBias: Bool, attentionDropout: Double, attentionKeqV: Bool,
        finalLogitSoftcapping: Double, tieWordEmbeddings: Bool,
        enableMoeBlock: Bool, numExperts: Int, topKExperts: Int,
        moeIntermediateSize: Int, useDoubleWideMlp: Bool,
        useBidirectionalAttention: String, dtype: String, useCache: Bool,
        bosTokenId: Int, eosTokenId: Int, padTokenId: Int,
        initializerRange: Double, slidingRope: Gemma4A4BRopeSpec,
        fullRope: Gemma4A4BRopeSpec, quantization: Gemma4A4BQuantization
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.vocabSizePerLayerInput = vocabSizePerLayerInput
        self.hiddenSize = hiddenSize
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.numGlobalKeyValueHeads = numGlobalKeyValueHeads
        self.numKvSharedLayers = numKvSharedLayers
        self.headDim = headDim
        self.globalHeadDim = globalHeadDim
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes
        self.rmsNormEps = rmsNormEps
        self.hiddenActivation = hiddenActivation
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.attentionKeqV = attentionKeqV
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.tieWordEmbeddings = tieWordEmbeddings
        self.enableMoeBlock = enableMoeBlock
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.moeIntermediateSize = moeIntermediateSize
        self.useDoubleWideMlp = useDoubleWideMlp
        self.useBidirectionalAttention = useBidirectionalAttention
        self.dtype = dtype
        self.useCache = useCache
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.padTokenId = padTokenId
        self.initializerRange = initializerRange
        self.slidingRope = slidingRope
        self.fullRope = fullRope
        self.quantization = quantization
    }

    public static func load(from weightsPath: String) throws -> Gemma4A4BConfig {
        let path = URL(fileURLWithPath: weightsPath)
            .appendingPathComponent("config.json")
        try requireFile(path.path, description: "transformed weights config")

        return try load(data: try Data(contentsOf: path))
    }

    /// Data-based entry point, single-sourced with `load(from:)` above.
    ///
    /// The runtime worker's pinned-configuration gate
    /// (`validateRuntimeWorkerPinnedConfigurationData` in both
    /// `Sources/MLXFastHarness/Gemma4RuntimeWorker.swift` and its
    /// `Sources/MLXFastTrustedHarness` twin) calls this directly instead of
    /// re-encoding a second, hand-maintained key/value list: the exact
    /// key-set check, the frozen invariant check, and the structural sanity
    /// check below are the ONLY copy of this target's config contract, so the
    /// worker gate and this loader cannot silently drift apart the way the
    /// pre-port Qwen-shaped gate drifted from the Gemma 4 artifact it was
    /// actually being handed.
    public static func load(data: Data) throws -> Gemma4A4BConfig {
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("config.json must be a JSON object")
        }

        try validateKeyPresence(in: root)

        let ropeObject = try gemmaObject("rope_parameters", in: root)
        let quantization = try gemmaQuantization(in: root)

        // Reject an unsafe layer count before parsing the layer array.
        let numHiddenLayers = try gemmaInt("num_hidden_layers", in: root)
        guard numHiddenLayers == MLXFastConstants.numHiddenLayers else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B config invariant check failed: "
                    + "num_hidden_layers=\(numHiddenLayers) expected "
                    + "\(MLXFastConstants.numHiddenLayers)"
            )
        }

        let config = Gemma4A4BConfig(
            modelType: try gemmaString("model_type", in: root),
            vocabSize: try gemmaInt("vocab_size", in: root),
            vocabSizePerLayerInput: try gemmaInt(
                "vocab_size_per_layer_input", in: root),
            hiddenSize: try gemmaInt("hidden_size", in: root),
            hiddenSizePerLayerInput: try gemmaInt(
                "hidden_size_per_layer_input", in: root),
            intermediateSize: try gemmaInt("intermediate_size", in: root),
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: try gemmaInt("num_attention_heads", in: root),
            numKeyValueHeads: try gemmaInt("num_key_value_heads", in: root),
            numGlobalKeyValueHeads: try gemmaInt(
                "num_global_key_value_heads", in: root),
            numKvSharedLayers: try gemmaInt("num_kv_shared_layers", in: root),
            headDim: try gemmaInt("head_dim", in: root),
            globalHeadDim: try gemmaInt("global_head_dim", in: root),
            slidingWindow: try gemmaInt("sliding_window", in: root),
            layerTypes: try gemmaLayerTypes("layer_types", in: root),
            rmsNormEps: try gemmaDouble("rms_norm_eps", in: root),
            hiddenActivation: try gemmaString("hidden_activation", in: root),
            maxPositionEmbeddings: try gemmaInt(
                "max_position_embeddings", in: root),
            attentionBias: try gemmaBool("attention_bias", in: root),
            attentionDropout: try gemmaDouble("attention_dropout", in: root),
            attentionKeqV: try gemmaBool("attention_k_eq_v", in: root),
            finalLogitSoftcapping: try gemmaDouble(
                "final_logit_softcapping", in: root),
            tieWordEmbeddings: try gemmaBool("tie_word_embeddings", in: root),
            enableMoeBlock: try gemmaBool("enable_moe_block", in: root),
            numExperts: try gemmaInt("num_experts", in: root),
            topKExperts: try gemmaInt("top_k_experts", in: root),
            moeIntermediateSize: try gemmaInt("moe_intermediate_size", in: root),
            useDoubleWideMlp: try gemmaBool("use_double_wide_mlp", in: root),
            useBidirectionalAttention: try gemmaString(
                "use_bidirectional_attention", in: root),
            dtype: try gemmaString("dtype", in: root),
            useCache: try gemmaBool("use_cache", in: root),
            bosTokenId: try gemmaInt("bos_token_id", in: root),
            eosTokenId: try gemmaInt("eos_token_id", in: root),
            padTokenId: try gemmaInt("pad_token_id", in: root),
            initializerRange: try gemmaDouble("initializer_range", in: root),
            slidingRope: try gemmaRope("sliding_attention", in: ropeObject),
            fullRope: try gemmaRope("full_attention", in: ropeObject),
            quantization: quantization
        )
        try config.validateFrozenInvariants()
        try config.validateStructuralValues()
        return config
    }

    /// Exact key-set discipline on the flattened config: every required key
    /// present and non-null, no forbidden key present, and no UNKNOWN key.
    ///
    /// The unknown-key rejection is the half that matters most on this target.
    /// The vendored `Gemma4TextConfiguration` decodes with
    /// `decodeIfPresent` for most fields, so a config carrying an unexpected
    /// knob loads silently with that knob active in the vendored model and
    /// invisible to this contract. Rejecting unknowns makes a checkpoint
    /// respin that adds a field a loud failure instead of a numerics change.
    static func validateKeyPresence(in root: [String: Any]) throws {
        var errors: [String] = []

        for key in requiredKeys.sorted() {
            guard let value = root[key] else {
                errors.append("missing required key \(key)")
                continue
            }
            if value is NSNull {
                errors.append("required key \(key) must not be null")
            }
        }
        for key in forbiddenKeys where root[key] != nil && !(root[key] is NSNull) {
            errors.append("forbidden key \(key) is present and non-null")
        }
        let known = requiredKeys.union(forbiddenKeys).union(["quantization"])
        for key in root.keys.sorted() where !known.contains(key) {
            errors.append("unexpected key \(key)")
        }

        guard errors.isEmpty else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B config key-set check failed: "
                    + errors.joined(separator: ", ")
            )
        }
    }

    public func validateFrozenInvariants() throws {
        var errors: [String] = []
        func expect<T: Equatable>(_ name: String, _ actual: T, _ expected: T) {
            if actual != expected {
                errors.append(
                    "\(name)=\(String(describing: actual)) "
                        + "expected \(String(describing: expected))"
                )
            }
        }

        expect("model_type", modelType, "gemma4_text")
        expect("vocab_size", vocabSize, MLXFastConstants.vocabSize)
        expect("vocab_size_per_layer_input", vocabSizePerLayerInput, 262_144)
        expect("hidden_size", hiddenSize, MLXFastConstants.hiddenSize)
        expect("hidden_size_per_layer_input", hiddenSizePerLayerInput, 0)
        expect(
            "intermediate_size", intermediateSize,
            MLXFastConstants.intermediateSize)
        expect(
            "num_hidden_layers", numHiddenLayers,
            MLXFastConstants.numHiddenLayers)
        expect(
            "num_attention_heads", numAttentionHeads,
            MLXFastConstants.attentionHeads)
        expect(
            "num_key_value_heads", numKeyValueHeads,
            MLXFastConstants.numKeyValueHeads)
        expect(
            "num_global_key_value_heads", numGlobalKeyValueHeads,
            MLXFastConstants.numGlobalKeyValueHeads)
        // 0 on this checkpoint: no layer borrows KV from an earlier one. The
        // vendored `sanitize` drops k/v/k_norm/v_norm for shared-KV layers, so
        // a non-zero value here would change which tensors the loader expects.
        expect("num_kv_shared_layers", numKvSharedLayers, 0)
        expect("head_dim", headDim, MLXFastConstants.headDim)
        expect("global_head_dim", globalHeadDim, MLXFastConstants.globalHeadDim)
        expect("sliding_window", slidingWindow, MLXFastConstants.slidingWindow)
        expect("rms_norm_eps", rmsNormEps, 1e-6)
        expect("hidden_activation", hiddenActivation, "gelu_pytorch_tanh")
        expect("max_position_embeddings", maxPositionEmbeddings, 262_144)
        expect("attention_bias", attentionBias, false)
        expect("attention_dropout", attentionDropout, 0)
        // TRUE, and it is a tensor-inventory fact as much as a numerics one:
        // the five full-attention layers ship NO v_proj at all.
        expect("attention_k_eq_v", attentionKeqV, true)
        expect(
            "final_logit_softcapping", finalLogitSoftcapping,
            MLXFastConstants.finalLogitSoftcapping)
        expect(
            "tie_word_embeddings", tieWordEmbeddings,
            MLXFastConstants.tieWordEmbeddings)
        expect("enable_moe_block", enableMoeBlock, true)
        expect("num_experts", numExperts, MLXFastConstants.numExperts)
        expect("top_k_experts", topKExperts, MLXFastConstants.numExpertsPerToken)
        expect(
            "moe_intermediate_size", moeIntermediateSize,
            MLXFastConstants.moeIntermediateSize)
        expect("use_double_wide_mlp", useDoubleWideMlp, false)
        expect("use_bidirectional_attention", useBidirectionalAttention, "vision")
        expect("dtype", dtype, "bfloat16")
        expect("use_cache", useCache, true)
        expect("bos_token_id", bosTokenId, 2)
        expect("eos_token_id", eosTokenId, 1)
        expect("pad_token_id", padTokenId, 0)
        expect("sliding_attention.rope_theta", slidingRope.theta, 10_000)
        expect("sliding_attention.rope_type", slidingRope.type, "default")
        expect(
            "sliding_attention.partial_rotary_factor",
            slidingRope.partialRotaryFactor, nil)
        expect("full_attention.rope_theta", fullRope.theta, 1_000_000)
        expect("full_attention.rope_type", fullRope.type, "proportional")
        expect(
            "full_attention.partial_rotary_factor",
            fullRope.partialRotaryFactor, 0.25)
        expect("quantization.group_size", quantization.fallback.groupSize, 64)
        expect("quantization.bits", quantization.fallback.bits, 4)
        expect("quantization.mode", quantization.mode, "affine")

        if layerTypes != Self.expectedLayerTypes {
            if layerTypes.count != Self.expectedLayerTypes.count {
                errors.append(
                    "layer_types count=\(layerTypes.count) "
                        + "expected \(Self.expectedLayerTypes.count)"
                )
            } else if let mismatch = zip(layerTypes, Self.expectedLayerTypes)
                .enumerated()
                .first(where: { $0.element.0 != $0.element.1 })
            {
                errors.append(
                    "layer_types[\(mismatch.offset)]="
                        + "\(mismatch.element.0.rawValue) expected "
                        + "\(mismatch.element.1.rawValue)"
                )
            }
        }

        errors.append(contentsOf: quantizationOverrideErrors())

        guard errors.isEmpty else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B config invariant check failed: "
                    + errors.joined(separator: ", ")
            )
        }
    }

    /// The override table is pinned by CONSTRUCTION, not by count.
    ///
    /// A count check would pass on 120 overrides naming the wrong paths. This
    /// builds the exact expected key set — four projection families on every
    /// layer — and requires set equality plus a uniform 8-bit/group-64 width,
    /// so both a missing promotion and an unexpected extra one are caught.
    func quantizationOverrideErrors() -> [String] {
        var expected: Set<String> = []
        for layer in 0..<numHiddenLayers {
            for family in Gemma4A4BWeightNames.quantizationOverrideFamilies {
                expected.insert(Gemma4A4BWeightNames.layer(layer, family))
            }
        }

        var errors: [String] = []
        let actual = Set(quantization.overrides.keys)
        let missing = expected.subtracting(actual).sorted()
        let unexpected = actual.subtracting(expected).sorted()
        if !missing.isEmpty {
            errors.append(
                "quantization is missing \(missing.count) expected per-tensor "
                    + "override(s), first: \(missing[0])"
            )
        }
        if !unexpected.isEmpty {
            errors.append(
                "quantization carries \(unexpected.count) unexpected "
                    + "per-tensor override(s), first: \(unexpected[0])"
            )
        }
        for path in expected.intersection(actual).sorted() {
            guard let spec = quantization.overrides[path] else { continue }
            if spec.bits != 8 || spec.groupSize != 64 {
                errors.append(
                    "quantization override \(path) is bits=\(spec.bits) "
                        + "group_size=\(spec.groupSize), expected bits=8 "
                        + "group_size=64"
                )
                break
            }
        }
        return errors
    }

    public func validateStructuralValues() throws {
        guard vocabSize > 0, hiddenSize > 0, intermediateSize > 0,
              numHiddenLayers > 0, moeIntermediateSize > 0
        else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B vocabulary and model dimensions must be positive"
            )
        }
        guard numAttentionHeads > 0, numKeyValueHeads > 0,
              numAttentionHeads.isMultiple(of: numKeyValueHeads),
              numGlobalKeyValueHeads > 0,
              numAttentionHeads.isMultiple(of: numGlobalKeyValueHeads),
              headDim > 0, headDim.isMultiple(of: 2),
              globalHeadDim > 0, globalHeadDim.isMultiple(of: 2)
        else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B attention heads and head dims are structurally invalid"
            )
        }
        guard numExperts > 0, topKExperts > 0, topKExperts <= numExperts else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B MoE routing is structurally invalid"
            )
        }
        guard numKvSharedLayers >= 0, numKvSharedLayers <= numHiddenLayers,
              layerTypes.count == numHiddenLayers
        else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B layer_types must cover every decoder layer"
            )
        }
        guard slidingWindow > 0, maxPositionEmbeddings > 0,
              rmsNormEps.isFinite, rmsNormEps > 0,
              attentionDropout.isFinite, attentionDropout >= 0,
              attentionDropout < 1,
              finalLogitSoftcapping.isFinite, finalLogitSoftcapping > 0
        else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B window, normalization, or dropout values are invalid"
            )
        }
        for (name, rope) in [("sliding", slidingRope), ("full", fullRope)] {
            guard rope.theta.isFinite, rope.theta > 0 else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B \(name) RoPE theta is invalid"
                )
            }
            if let partial = rope.partialRotaryFactor {
                let headWidth = name == "full" ? globalHeadDim : headDim
                guard partial.isFinite, partial > 0, partial <= 1,
                      let rotary = Int(exactly: Double(headWidth) * partial),
                      rotary >= 2, rotary.isMultiple(of: 2)
                else {
                    throw MLXFastError.invalidInput(
                        "Gemma 4 26B A4B \(name) partial rotary dimensions are invalid"
                    )
                }
            }
        }
        // Divisibility is checked against EVERY width the checkpoint actually
        // uses, not just the fallback: an 8-bit override on a path whose width
        // is not a multiple of its group size is as broken as a bad fallback.
        var widths = Set([quantization.fallback.groupSize])
        for spec in quantization.overrides.values {
            widths.insert(spec.groupSize)
        }
        for groupSize in widths.sorted() {
            guard groupSize > 0,
                  hiddenSize.isMultiple(of: groupSize),
                  intermediateSize.isMultiple(of: groupSize),
                  moeIntermediateSize.isMultiple(of: groupSize)
            else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B dimensions are not divisible by "
                        + "quantization group size \(groupSize)"
                )
            }
        }
        var bitWidths = Set([quantization.fallback.bits])
        for spec in quantization.overrides.values {
            bitWidths.insert(spec.bits)
        }
        guard bitWidths.allSatisfy({ [2, 4, 8].contains($0) }) else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B quantization bit width is invalid"
            )
        }
    }
}

private func gemmaRequiredValue(
    _ key: String, in root: [String: Any]
) throws -> Any {
    guard let value = root[key] else {
        throw MLXFastError.invalidInput("config field \(key) is required")
    }
    guard !(value is NSNull) else {
        throw MLXFastError.invalidInput("config field \(key) must not be null")
    }
    return value
}

private func gemmaObject(
    _ key: String, in root: [String: Any]
) throws -> [String: Any] {
    let value = try gemmaRequiredValue(key, in: root)
    guard let result = value as? [String: Any] else {
        throw MLXFastError.invalidInput("config field \(key) must be a JSON object")
    }
    return result
}

/// Parse one per-attention-kind RoPE entry with EXACT key-set equality.
///
/// The two entries have different key sets by design — `full_attention`
/// carries `partial_rotary_factor` and `sliding_attention` does not — so the
/// allowed set is computed per kind rather than shared. An entry that grows a
/// key, or that gains a partial factor on the sliding side, is rejected here
/// rather than silently ignored.
private func gemmaRope(
    _ key: String, in ropeParameters: [String: Any]
) throws -> Gemma4A4BRopeSpec {
    let entry = try gemmaObject(key, in: ropeParameters)
    let theta = try gemmaDouble("rope_theta", in: entry)
    let type = try gemmaString("rope_type", in: entry)

    let hasPartial = entry.keys.contains("partial_rotary_factor")
    let allowed: Set<String> = hasPartial
        ? ["rope_theta", "rope_type", "partial_rotary_factor"]
        : ["rope_theta", "rope_type"]
    guard Set(entry.keys) == allowed else {
        throw MLXFastError.invalidInput(
            "config rope_parameters.\(key) key set is "
                + "\(Set(entry.keys).sorted()), expected \(allowed.sorted())"
        )
    }
    return Gemma4A4BRopeSpec(
        theta: theta,
        type: type,
        partialRotaryFactor: hasPartial
            ? try gemmaDouble("partial_rotary_factor", in: entry) : nil
    )
}

/// Parse the mixed-precision quantization block.
///
/// Shape: three scalar keys (`group_size`, `bits`, `mode`) that form the
/// fallback, plus any number of tensor-path keys mapping to a `{group_size,
/// bits}` object. Anything else in the block is rejected: a scalar key that is
/// not one of the three, or an override object carrying an unknown key, both
/// mean the checkpoint is describing something this contract does not model.
///
/// `quantization_config` is deliberately NOT read here. The transform emits a
/// single canonical `quantization` block and removes the duplicate, so a
/// transformed config carrying both would have failed the key-set check above
/// before reaching this function.
private func gemmaQuantization(
    in root: [String: Any]
) throws -> Gemma4A4BQuantization {
    let block = try gemmaObject("quantization", in: root)

    let scalarKeys: Set<String> = ["group_size", "bits", "mode"]
    for key in scalarKeys {
        guard block[key] != nil else {
            throw MLXFastError.invalidInput(
                "config quantization is missing required key \(key)"
            )
        }
    }

    var overrides: [String: Gemma4A4BQuantizationSpec] = [:]
    for key in block.keys.sorted() where !scalarKeys.contains(key) {
        guard let entry = block[key] as? [String: Any] else {
            throw MLXFastError.invalidInput(
                "config quantization key \(key) is neither a known scalar key "
                    + "nor a per-tensor override object"
            )
        }
        guard Set(entry.keys) == ["group_size", "bits"] else {
            throw MLXFastError.invalidInput(
                "config quantization override \(key) key set is "
                    + "\(Set(entry.keys).sorted()), expected [bits, group_size]"
            )
        }
        overrides[key] = Gemma4A4BQuantizationSpec(
            groupSize: try gemmaInt("group_size", in: entry),
            bits: try gemmaInt("bits", in: entry)
        )
    }

    return Gemma4A4BQuantization(
        fallback: Gemma4A4BQuantizationSpec(
            groupSize: try gemmaInt("group_size", in: block),
            bits: try gemmaInt("bits", in: block)
        ),
        mode: try gemmaString("mode", in: block),
        overrides: overrides
    )
}

private func gemmaInt(_ key: String, in root: [String: Any]) throws -> Int {
    try gemmaParseInt(gemmaRequiredValue(key, in: root), field: key)
}

private func gemmaDouble(_ key: String, in root: [String: Any]) throws -> Double {
    let value = try gemmaRequiredValue(key, in: root)
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID()
    else {
        throw MLXFastError.invalidInput(
            "config field \(key) must be a finite number")
    }
    let result = number.doubleValue
    guard result.isFinite else {
        throw MLXFastError.invalidInput(
            "config field \(key) must be a finite number")
    }
    return result
}

private func gemmaBool(_ key: String, in root: [String: Any]) throws -> Bool {
    let value = try gemmaRequiredValue(key, in: root)
    guard let number = value as? NSNumber,
          CFGetTypeID(number) == CFBooleanGetTypeID()
    else {
        throw MLXFastError.invalidInput("config field \(key) must be a boolean")
    }
    return number.boolValue
}

private func gemmaString(_ key: String, in root: [String: Any]) throws -> String {
    let value = try gemmaRequiredValue(key, in: root)
    guard let result = value as? String else {
        throw MLXFastError.invalidInput("config field \(key) must be a string")
    }
    return result
}

private func gemmaLayerTypes(
    _ key: String, in root: [String: Any]
) throws -> [Gemma4A4BLayerType] {
    let value = try gemmaRequiredValue(key, in: root)
    guard let values = value as? [String] else {
        throw MLXFastError.invalidInput(
            "config field \(key) must be a string array")
    }
    return try values.map {
        guard let layerType = Gemma4A4BLayerType(rawValue: $0) else {
            throw MLXFastError.invalidInput(
                "config field \(key) contains unsupported layer type \($0)"
            )
        }
        return layerType
    }
}

private func gemmaParseInt(_ value: Any, field: String) throws -> Int {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID(),
          !CFNumberIsFloatType(number),
          let integer = Int(number.stringValue)
    else {
        throw MLXFastError.invalidInput(
            "config field \(field) must be a finite integer in Int range"
        )
    }
    return integer
}
