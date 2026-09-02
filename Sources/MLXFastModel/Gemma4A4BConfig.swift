import CoreFoundation
import Foundation
import MLXFastCore

public enum Gemma4A4BLayerType: String, Equatable, Sendable {
    case sliding = "sliding_attention"
    case full = "full_attention"
}

public struct Gemma4A4BRopeSpec: Equatable, Sendable {
    public let theta: Double
    public let type: String
    public let partialRotaryFactor: Double?

    public init(theta: Double, type: String, partialRotaryFactor: Double?) {
        self.theta = theta
        self.type = type
        self.partialRotaryFactor = partialRotaryFactor
    }
}

public struct Gemma4A4BQuantizationSpec: Equatable, Sendable {
    public let groupSize: Int
    public let bits: Int

    public init(groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
    }
}

public struct Gemma4A4BQuantization: Equatable, Sendable {
    public let fallback: Gemma4A4BQuantizationSpec
    public let mode: String
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

    public func spec(forPath path: String) -> Gemma4A4BQuantizationSpec {
        overrides[path] ?? fallback
    }
}

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

    public static let expectedLayerTypes: [Gemma4A4BLayerType] =
        (0..<MLXFastConstants.numHiddenLayers).map {
            $0 % MLXFastConstants.fullAttentionInterval
                == MLXFastConstants.fullAttentionInterval - 1 ? .full : .sliding
        }

    static let requiredKeys: Set<String> = Gemma4A4BConfigKeys.required

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

    public static func load(data: Data) throws -> Gemma4A4BConfig {
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("config.json must be a JSON object")
        }

        try validateKeyPresence(in: root)

        let ropeObject = try gemmaObject("rope_parameters", in: root)
        let quantization = try gemmaQuantization(in: root)

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
        expect("num_kv_shared_layers", numKvSharedLayers, 0)
        expect("head_dim", headDim, MLXFastConstants.headDim)
        expect("global_head_dim", globalHeadDim, MLXFastConstants.globalHeadDim)
        expect("sliding_window", slidingWindow, MLXFastConstants.slidingWindow)
        expect("rms_norm_eps", rmsNormEps, 1e-6)
        expect("hidden_activation", hiddenActivation, "gelu_pytorch_tanh")
        expect("max_position_embeddings", maxPositionEmbeddings, 262_144)
        expect("attention_bias", attentionBias, false)
        expect("attention_dropout", attentionDropout, 0)
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
