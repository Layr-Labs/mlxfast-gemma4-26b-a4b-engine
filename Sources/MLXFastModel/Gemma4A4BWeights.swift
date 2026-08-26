import Foundation
import MLX
import MLXFastCore

/// Tensor names for the text-only Gemma 4 26B A4B checkpoint.
///
/// The transform passes the source `language_model.*` names through unchanged,
/// so these are both the checkpoint names and the names in the transformed
/// `weights/` tree.
public enum Gemma4A4BWeightNames {
    public static let modelPrefix = "language_model.model"

    public static let embedTokens = "\(modelPrefix).embed_tokens"
    public static let finalNorm = "\(modelPrefix).norm.weight"

    /// There is NO `lm_head`. `tie_word_embeddings` is true on this
    /// checkpoint, so the output projection IS the embedding and the index
    /// carries no `lm_head.*` entry at all.

    public static func layer(_ index: Int, _ suffix: String) -> String {
        "\(modelPrefix).layers.\(index).\(suffix)"
    }

    /// The four projection families the checkpoint promotes to 8 bits, on
    /// every layer. Single-sourced from
    /// `MLXFastCore.Gemma4A4BConfigKeys.quantizationOverrideFamilies` (which
    /// the trusted runtime-worker gate also reads, see that type's doc
    /// comment) so `Gemma4A4BConfig`'s override check, this inventory, and
    /// the trusted gate all agree on the same four families.
    public static let quantizationOverrideFamilies =
        Gemma4A4BConfigKeys.quantizationOverrideFamilies

    /// Per-layer norms, all `[hidden_size]`. Gemma 4 carries seven, including
    /// the two `_1`/`_2` post-feedforward variants and the `_2` pre-feedforward
    /// one that the Gemma 3 layout does not have.
    public static let layerNormSuffixes = [
        "input_layernorm", "post_attention_layernorm",
        "pre_feedforward_layernorm", "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm", "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
    ]

    /// Quantized-tensor component suffixes, in checkpoint order.
    public static let quantizedComponents = ["weight", "scales", "biases"]
}

/// Dense tensor access and startup metadata validation for the Gemma 4 26B A4B
/// text tower.
///
/// The scored loader uses `denseStore` to load complete safetensors shards into
/// MLX; the primitive accessors remain useful for metadata and bridge tests.
public struct Gemma4A4BWeightLoader {
    /// Read off the pinned revision's own `model.safetensors.index.json`:
    /// 1,697 tensors total, of which 1,339 carry the `language_model.` prefix
    /// and 358 are vision (`vision_tower.` 355, `embed_vision.` 3) and are
    /// dropped by the transform.
    public static let requiredTensorCount = 1_339
    /// 4 top-level: `embed_tokens.{weight,scales,biases}` + `norm.weight`.
    public static let requiredTopLevelTensorCount = 4
    /// A sliding layer carries 45 tensors; a global layer carries 42 — it
    /// ships no `v_proj.{weight,scales,biases}` because `attention_k_eq_v` is
    /// true and values reuse the keys. 25 x 45 + 5 x 42 + 4 = 1,339.
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

    /// Query-head width of one layer. Global layers use `global_head_dim`.
    public static func headDim(
        layer index: Int, config: Gemma4A4BConfig
    ) -> Int {
        config.layerTypes[index] == .full
            ? config.globalHeadDim : config.headDim
    }

    /// KV-head count of one layer. Global layers use
    /// `num_global_key_value_heads`.
    public static func kvHeads(
        layer index: Int, config: Gemma4A4BConfig
    ) -> Int {
        config.layerTypes[index] == .full
            ? config.numGlobalKeyValueHeads : config.numKeyValueHeads
    }

    /// Packed width of a quantized weight's last axis: `in * bits / 32`,
    /// because affine-quantized weights are stored as `UInt32` words.
    static func packedColumns(inFeatures: Int, bits: Int) -> Int {
        inFeatures * bits / 32
    }

    /// Width of a `scales`/`biases` last axis: one entry per group.
    static func groupColumns(inFeatures: Int, groupSize: Int) -> Int {
        inFeatures / groupSize
    }

    // MARK: - Inventory

    /// The complete expected tensor-name set, derived from the config rather
    /// than hardcoded, so a geometry change surfaces as a name mismatch here
    /// instead of a shape error deep in the loader.
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
            // `attention_k_eq_v`: only the sliding layers ship a v_proj. A
            // global layer that carries one is as much a mismatch as a sliding
            // layer that omits one, so this is expressed as presence/absence
            // rather than as a tolerated extra.
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

    /// Metadata-only startup validation: names, dtypes and shapes, with no
    /// tensor bytes read. Everything here is derivable from the safetensors
    /// headers, which is what makes it runnable before the model loads.
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
        // Q/K norms are per-HEAD, so they follow the layer's own head width.
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

        // Experts are SwitchGLU-stacked: a leading experts axis, never split
        // per expert.
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

    /// Validate one affine-quantized tensor triple against the width the
    /// config resolves for ITS OWN path.
    ///
    /// This is where the mixed-precision contract becomes a shape check: an
    /// 8-bit path packs `in * 8 / 32` columns and a 4-bit path packs
    /// `in * 4 / 32`, so a promotion that the config dropped shows up here as
    /// a factor-of-two shape mismatch rather than as silent numerics.
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
