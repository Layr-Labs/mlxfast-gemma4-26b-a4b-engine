import Foundation
import MLX
import MLXFastCore

public enum Gemma4A4BWeightNames {
    public static let modelPrefix = "language_model.model"

    public static let embedTokens = "\(modelPrefix).embed_tokens"
    public static let finalNorm = "\(modelPrefix).norm.weight"

    public static func layer(_ index: Int, _ suffix: String) -> String {
        "\(modelPrefix).layers.\(index).\(suffix)"
    }

    public static let quantizationOverrideFamilies =
        Gemma4A4BConfigKeys.quantizationOverrideFamilies

    public static let layerNormSuffixes = [
        "input_layernorm", "post_attention_layernorm",
        "pre_feedforward_layernorm", "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm", "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
    ]

    public static let quantizedComponents = ["weight", "scales", "biases"]
}

public struct Gemma4A4BWeightLoader {
    public static let requiredTensorCount = 1_339
    public static let requiredTopLevelTensorCount = 4
    public static let requiredSlidingLayerTensorCount = 45
    public static let requiredGlobalLayerTensorCount = 42

    public let denseStore: DenseTensorStore
    private let bridge: MLXArrayTensorBridge

    public init(
        weightsPath: String,
        bridge: MLXArrayTensorBridge = MLXArrayTensorBridge()
    ) throws {
        self.denseStore = try DenseTensorStore(weightsPath: weightsPath)
        self.bridge = bridge
    }

    public init(
        denseStore: DenseTensorStore,
        bridge: MLXArrayTensorBridge = MLXArrayTensorBridge()
    ) {
        self.denseStore = denseStore
        self.bridge = bridge
    }

    public func materializedTensor(
        named name: String,
        expectedShape: [Int]? = nil
    ) throws -> MaterializedTensor {
        let tensor = try denseStore.materializedTensor(named: name)
        try validateShape(
            tensor.shape, expectedShape: expectedShape, tensorName: name)
        return tensor
    }

    public func denseArray(
        named name: String,
        expectedShape: [Int]? = nil
    ) throws -> MLXArray {
        try bridge.makeArray(
            from: materializedTensor(named: name, expectedShape: expectedShape))
    }

    // MARK: - Geometry helpers

    public static func headDim(
        layer index: Int, config: Gemma4A4BConfig
    ) -> Int {
        config.layerTypes[index] == .full
            ? config.globalHeadDim : config.headDim
    }

    public static func kvHeads(
        layer index: Int, config: Gemma4A4BConfig
    ) -> Int {
        config.layerTypes[index] == .full
            ? config.numGlobalKeyValueHeads : config.numKeyValueHeads
    }

    static func packedColumns(inFeatures: Int, bits: Int) -> Int {
        inFeatures * bits / 32
    }

    static func groupColumns(inFeatures: Int, groupSize: Int) -> Int {
        inFeatures / groupSize
    }

    // MARK: - Inventory

    public static func requiredTensorNames(
        config: Gemma4A4BConfig
    ) throws -> Set<String> {
        var names: Set<String> = [Gemma4A4BWeightNames.finalNorm]
        for component in Gemma4A4BWeightNames.quantizedComponents {
            names.insert("\(Gemma4A4BWeightNames.embedTokens).\(component)")
        }

        for index in 0..<config.numHiddenLayers {
            for norm in Gemma4A4BWeightNames.layerNormSuffixes {
                names.insert(
                    Gemma4A4BWeightNames.layer(index, "\(norm).weight"))
            }
            names.insert(Gemma4A4BWeightNames.layer(index, "layer_scalar"))
            names.insert(Gemma4A4BWeightNames.layer(index, "router.scale"))
            names.insert(
                Gemma4A4BWeightNames.layer(index, "router.per_expert_scale"))
            names.insert(
                Gemma4A4BWeightNames.layer(index, "self_attn.q_norm.weight"))
            names.insert(
                Gemma4A4BWeightNames.layer(index, "self_attn.k_norm.weight"))

            var quantizedStems = [
                "router.proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                "experts.switch_glu.gate_proj",
                "experts.switch_glu.up_proj",
                "experts.switch_glu.down_proj",
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj",
            ]
            if config.layerTypes[index] == .sliding {
                quantizedStems.append("self_attn.v_proj")
            }
            for stem in quantizedStems {
                for component in Gemma4A4BWeightNames.quantizedComponents {
                    names.insert(
                        Gemma4A4BWeightNames.layer(index, "\(stem).\(component)")
                    )
                }
            }
        }
        return names
    }

    public func validateRequiredMetadata(config: Gemma4A4BConfig) throws {
        try config.validateFrozenInvariants()
        try config.validateStructuralValues()

        let actualNames = Set(denseStore.tensorNames)
        guard !actualNames.contains(where: { $0.hasPrefix("vision_tower.") })
                && !actualNames.contains(where: { $0.hasPrefix("embed_vision.") })
        else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B transformed weights still carry vision "
                    + "tensors; the transform must drop every vision_tower.* "
                    + "and embed_vision.* tensor"
            )
        }
        guard !actualNames.contains(where: { $0.hasPrefix("lm_head.") }) else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B checkpoint has tied embeddings and must not "
                    + "carry an lm_head tensor"
            )
        }

        let expectedNames = try Self.requiredTensorNames(config: config)
        guard actualNames == expectedNames else {
            let missing = expectedNames.subtracting(actualNames).sorted()
            let unexpected = actualNames.subtracting(expectedNames).sorted()
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B tensor inventory mismatch: expected "
                    + "\(expectedNames.count), found \(actualNames.count); "
                    + "missing=\(Self.summarize(missing)); "
                    + "unexpected=\(Self.summarize(unexpected))"
            )
        }

        try validateQuantizedTensor(
            stem: Gemma4A4BWeightNames.embedTokens,
            leadingShape: [config.vocabSize],
            inFeatures: config.hiddenSize,
            config: config
        )
        try validateDenseTensorMetadata(
            named: Gemma4A4BWeightNames.finalNorm,
            expectedShape: [config.hiddenSize]
        )

        for index in 0..<config.numHiddenLayers {
            try validateLayerMetadata(index: index, config: config)
        }
    }

    private func validateLayerMetadata(
        index: Int, config: Gemma4A4BConfig
    ) throws {
        let headDim = Self.headDim(layer: index, config: config)
        let kvHeads = Self.kvHeads(layer: index, config: config)
        let queryWidth = config.numAttentionHeads * headDim
        let kvWidth = kvHeads * headDim

        for norm in Gemma4A4BWeightNames.layerNormSuffixes {
            try validateDenseTensorMetadata(
                named: Gemma4A4BWeightNames.layer(index, "\(norm).weight"),
                expectedShape: [config.hiddenSize]
            )
        }
        try validateDenseTensorMetadata(
            named: Gemma4A4BWeightNames.layer(index, "layer_scalar"),
            expectedShape: [1]
        )
        try validateDenseTensorMetadata(
            named: Gemma4A4BWeightNames.layer(index, "router.scale"),
            expectedShape: [config.hiddenSize]
        )
        try validateDenseTensorMetadata(
            named: Gemma4A4BWeightNames.layer(index, "router.per_expert_scale"),
            expectedShape: [config.numExperts]
        )
        for norm in ["q_norm", "k_norm"] {
            try validateDenseTensorMetadata(
                named: Gemma4A4BWeightNames.layer(
                    index, "self_attn.\(norm).weight"),
                expectedShape: [headDim]
            )
        }

        try validateQuantizedTensor(
            stem: Gemma4A4BWeightNames.layer(index, "self_attn.q_proj"),
            leadingShape: [queryWidth],
            inFeatures: config.hiddenSize, config: config)
        try validateQuantizedTensor(
            stem: Gemma4A4BWeightNames.layer(index, "self_attn.k_proj"),
            leadingShape: [kvWidth],
            inFeatures: config.hiddenSize, config: config)
        if config.layerTypes[index] == .sliding {
            try validateQuantizedTensor(
                stem: Gemma4A4BWeightNames.layer(index, "self_attn.v_proj"),
                leadingShape: [kvWidth],
                inFeatures: config.hiddenSize, config: config)
        }
        try validateQuantizedTensor(
            stem: Gemma4A4BWeightNames.layer(index, "self_attn.o_proj"),
            leadingShape: [config.hiddenSize],
            inFeatures: queryWidth, config: config)

        try validateQuantizedTensor(
            stem: Gemma4A4BWeightNames.layer(index, "router.proj"),
            leadingShape: [config.numExperts],
            inFeatures: config.hiddenSize, config: config)

        for stem in ["mlp.gate_proj", "mlp.up_proj"] {
            try validateQuantizedTensor(
                stem: Gemma4A4BWeightNames.layer(index, stem),
                leadingShape: [config.intermediateSize],
                inFeatures: config.hiddenSize, config: config)
        }
        try validateQuantizedTensor(
            stem: Gemma4A4BWeightNames.layer(index, "mlp.down_proj"),
            leadingShape: [config.hiddenSize],
            inFeatures: config.intermediateSize, config: config)

        for stem in [
            "experts.switch_glu.gate_proj", "experts.switch_glu.up_proj",
        ] {
            try validateQuantizedTensor(
                stem: Gemma4A4BWeightNames.layer(index, stem),
                leadingShape: [config.numExperts, config.moeIntermediateSize],
                inFeatures: config.hiddenSize, config: config)
        }
        try validateQuantizedTensor(
            stem: Gemma4A4BWeightNames.layer(
                index, "experts.switch_glu.down_proj"),
            leadingShape: [config.numExperts, config.hiddenSize],
            inFeatures: config.moeIntermediateSize, config: config)
    }

    private func validateQuantizedTensor(
        stem: String,
        leadingShape: [Int],
        inFeatures: Int,
        config: Gemma4A4BConfig
    ) throws {
        let spec = config.quantization.spec(forPath: stem)
        guard inFeatures.isMultiple(of: spec.groupSize) else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B tensor \(stem) has in_features \(inFeatures) "
                    + "which is not a multiple of its group size \(spec.groupSize)"
            )
        }
        try validateDenseTensorMetadata(
            named: "\(stem).weight",
            expectedShape: leadingShape
                + [Self.packedColumns(inFeatures: inFeatures, bits: spec.bits)]
        )
        let groupShape = leadingShape
            + [Self.groupColumns(inFeatures: inFeatures, groupSize: spec.groupSize)]
        for component in ["scales", "biases"] {
            try validateDenseTensorMetadata(
                named: "\(stem).\(component)", expectedShape: groupShape)
        }
    }

    private func validateDenseTensorMetadata(
        named name: String, expectedShape: [Int]
    ) throws {
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B transformed weights are missing tensor \(name)"
            )
        }
        guard record.shape == expectedShape else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B tensor \(name) shape \(record.shape) does not "
                    + "match expected \(expectedShape)"
            )
        }
    }

    private func validateShape(
        _ shape: [Int], expectedShape: [Int]?, tensorName: String
    ) throws {
        guard let expectedShape, shape != expectedShape else { return }
        throw MLXFastError.invalidInput(
            "Gemma 4 26B A4B tensor \(tensorName) shape \(shape) does not "
                + "match expected \(expectedShape)"
        )
    }

    private static func summarize(_ names: [String]) -> String {
        guard !names.isEmpty else { return "none" }
        let head = names.prefix(3).joined(separator: ", ")
        return names.count <= 3 ? head : "\(head), ... (\(names.count) total)"
    }
}
