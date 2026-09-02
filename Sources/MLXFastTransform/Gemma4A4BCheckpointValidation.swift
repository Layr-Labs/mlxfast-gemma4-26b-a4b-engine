import CoreFoundation
import Foundation
import MLXFastCore

struct Gemma4A4BTransformTensorQuantization: Equatable {
    let groupSize: Int
    let bits: Int
}

struct Gemma4A4BTransformQuantizationSpec: Equatable {
    let groupSize: Int
    let bits: Int
    let mode: String
    let overrides: [String: Gemma4A4BTransformTensorQuantization]

    var fallback: Gemma4A4BTransformTensorQuantization {
        Gemma4A4BTransformTensorQuantization(groupSize: groupSize, bits: bits)
    }

    func spec(forPath path: String) -> Gemma4A4BTransformTensorQuantization {
        overrides[path] ?? fallback
    }
}

enum Gemma4A4BCheckpointValidation {
    struct ExpectedTensorMetadata: Equatable {
        let dtype: String
        let shape: [Int]
    }

    enum PinnedGeometry {
        static let vocabSize = 262_144
        static let hiddenSize = 2_816
        static let intermediateSize = 2_112
        static let moeIntermediateSize = 704
        static let expertCount = 128
        static let layerCount = 30
        static let fullAttentionInterval = 6
        static let attentionHeads = 16
        static let keyValueHeads = 8
        static let globalKeyValueHeads = 2
        static let headDim = 256
        static let globalHeadDim = 512
        static let quantizationGroupSize = 64
        static let quantizationBits = 4
        static let quantizationOverrideBits = 8
        static let quantizationMode = "affine"
        static let quantizationOverrideFamilies = [
            "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "router.proj",
        ]

        static func isFullAttention(layer index: Int) -> Bool {
            index % fullAttentionInterval == fullAttentionInterval - 1
        }
    }

    static let expectedTensorCount = 1_339
    static let expectedSlidingLayerTensorCount = 45
    static let expectedFullAttentionLayerTensorCount = 42
    static let expectedTopLevelTensorCount = 4
    static let expectedQuantizationOverrideCount = 120

    static let textTowerPrefix = "language_model."

    private static let modelPrefix = "language_model.model"
    private static let layerPrefix = "language_model.model.layers."

    private static let layerNormSuffixes = [
        "input_layernorm", "post_attention_layernorm",
        "pre_feedforward_layernorm", "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm", "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
    ]

    static func quantizationSpec(
        fromConfigRoot root: [String: Any]
    ) throws -> Gemma4A4BTransformQuantizationSpec {
        func parseBlock(
            _ key: String
        ) throws -> Gemma4A4BTransformQuantizationSpec? {
            guard let value = root[key], !(value is NSNull) else {
                return nil
            }
            guard let block = value as? [String: Any] else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B config \(key) must be an object"
                )
            }
            let scalarKeys: Set<String> = ["group_size", "bits", "mode"]
            guard block["group_size"] != nil, block["bits"] != nil,
                  block["mode"] != nil
            else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B config \(key) must explicitly define "
                        + "group_size, bits, and mode"
                )
            }
            let groupSize = try intField("group_size", in: block)
            let bits = try intField("bits", in: block)
            let mode = try stringField("mode", in: block)
            guard mode == PinnedGeometry.quantizationMode,
                  groupSize == PinnedGeometry.quantizationGroupSize,
                  bits == PinnedGeometry.quantizationBits
            else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B fallback quantization must be affine "
                        + "4-bit group_size 64"
                )
            }

            var overrides: [String: Gemma4A4BTransformTensorQuantization] = [:]
            for name in block.keys.sorted() where !scalarKeys.contains(name) {
                guard let entry = block[name] as? [String: Any] else {
                    throw MLXFastError.invalidInput(
                        "Gemma 4 26B A4B config \(key) key \(name) is neither "
                            + "a known scalar key nor a per-tensor override "
                            + "object"
                    )
                }
                guard Set(entry.keys) == ["group_size", "bits"] else {
                    throw MLXFastError.invalidInput(
                        "Gemma 4 26B A4B config \(key) override \(name) key "
                            + "set is \(Set(entry.keys).sorted()), expected "
                            + "[bits, group_size]"
                    )
                }
                overrides[name] = Gemma4A4BTransformTensorQuantization(
                    groupSize: try intField("group_size", in: entry),
                    bits: try intField("bits", in: entry)
                )
            }
            try validateOverrideTable(overrides, blockKey: key)

            return Gemma4A4BTransformQuantizationSpec(
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                overrides: overrides
            )
        }

        let quantization = try parseBlock("quantization")
        let quantizationConfig = try parseBlock("quantization_config")
        if let quantization, let quantizationConfig {
            guard quantization == quantizationConfig else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B config quantization and "
                        + "quantization_config must match exactly"
                )
            }
            return quantization
        }
        guard let spec = quantization ?? quantizationConfig else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B config is missing both quantization and "
                    + "quantization_config"
            )
        }
        return spec
    }

    private static func validateOverrideTable(
        _ overrides: [String: Gemma4A4BTransformTensorQuantization],
        blockKey: String
    ) throws {
        var expected: Set<String> = []
        for layerIndex in 0..<PinnedGeometry.layerCount {
            for family in PinnedGeometry.quantizationOverrideFamilies {
                expected.insert("\(layerPrefix)\(layerIndex).\(family)")
            }
        }

        let actual = Set(overrides.keys)
        let missing = expected.subtracting(actual).sorted()
        let unexpected = actual.subtracting(expected).sorted()
        guard missing.isEmpty, unexpected.isEmpty else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B config \(blockKey) must carry exactly the "
                    + "\(expectedQuantizationOverrideCount) per-tensor "
                    + "overrides of the pinned checkpoint "
                    + "(missing: \(missing.prefix(4).joined(separator: ", ")); "
                    + "unexpected: "
                    + "\(unexpected.prefix(4).joined(separator: ", ")))"
            )
        }
        for path in expected.sorted() {
            guard let spec = overrides[path] else { continue }
            guard spec.bits == PinnedGeometry.quantizationOverrideBits,
                  spec.groupSize == PinnedGeometry.quantizationGroupSize
            else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B config \(blockKey) override \(path) is "
                        + "bits=\(spec.bits) group_size=\(spec.groupSize), "
                        + "expected bits="
                        + "\(PinnedGeometry.quantizationOverrideBits) "
                        + "group_size=\(PinnedGeometry.quantizationGroupSize)"
                )
            }
        }
    }

    static func expectedTensorInventory() -> [String: ExpectedTensorMetadata] {
        let hidden = PinnedGeometry.hiddenSize
        let intermediate = PinnedGeometry.intermediateSize
        let moeIntermediate = PinnedGeometry.moeIntermediateSize
        let experts = PinnedGeometry.expertCount
        let vocab = PinnedGeometry.vocabSize
        let groupSize = PinnedGeometry.quantizationGroupSize
        let fallbackBits = PinnedGeometry.quantizationBits
        let overrideBits = PinnedGeometry.quantizationOverrideBits

        var inventory: [String: ExpectedTensorMetadata] = [:]
        func add(_ name: String, _ dtype: TensorDType, _ shape: [Int]) {
            precondition(
                inventory[name] == nil,
                "duplicate expected Gemma 4 26B A4B tensor \(name)"
            )
            inventory[name] = ExpectedTensorMetadata(
                dtype: dtype.rawValue, shape: shape)
        }
        func addAffine(
            _ stem: String, leading: [Int], inFeatures: Int, bits: Int
        ) {
            precondition(
                inFeatures.isMultiple(of: groupSize)
                    && (inFeatures * bits).isMultiple(of: 32),
                "Gemma 4 26B A4B tensor \(stem) is not packable at \(bits) bits"
            )
            add(
                "\(stem).weight", .u32, leading + [inFeatures * bits / 32])
            add("\(stem).scales", .bf16, leading + [inFeatures / groupSize])
            add("\(stem).biases", .bf16, leading + [inFeatures / groupSize])
        }

        addAffine(
            "\(modelPrefix).embed_tokens",
            leading: [vocab],
            inFeatures: hidden,
            bits: fallbackBits
        )
        add("\(modelPrefix).norm.weight", .bf16, [hidden])

        for layerIndex in 0..<PinnedGeometry.layerCount {
            let prefix = "\(layerPrefix)\(layerIndex)"
            let isFull = PinnedGeometry.isFullAttention(layer: layerIndex)
            let headDim = isFull
                ? PinnedGeometry.globalHeadDim : PinnedGeometry.headDim
            let kvHeads = isFull
                ? PinnedGeometry.globalKeyValueHeads
                : PinnedGeometry.keyValueHeads
            let queryWidth = PinnedGeometry.attentionHeads * headDim
            let kvWidth = kvHeads * headDim

            for norm in layerNormSuffixes {
                add("\(prefix).\(norm).weight", .bf16, [hidden])
            }
            add("\(prefix).layer_scalar", .bf16, [1])
            add("\(prefix).router.scale", .bf16, [hidden])
            add("\(prefix).router.per_expert_scale", .bf16, [experts])
            add("\(prefix).self_attn.q_norm.weight", .bf16, [headDim])
            add("\(prefix).self_attn.k_norm.weight", .bf16, [headDim])

            addAffine(
                "\(prefix).self_attn.q_proj",
                leading: [queryWidth], inFeatures: hidden, bits: fallbackBits)
            addAffine(
                "\(prefix).self_attn.k_proj",
                leading: [kvWidth], inFeatures: hidden, bits: fallbackBits)
            if !isFull {
                addAffine(
                    "\(prefix).self_attn.v_proj",
                    leading: [kvWidth], inFeatures: hidden,
                    bits: fallbackBits)
            }
            addAffine(
                "\(prefix).self_attn.o_proj",
                leading: [hidden], inFeatures: queryWidth, bits: fallbackBits)

            addAffine(
                "\(prefix).router.proj",
                leading: [experts], inFeatures: hidden, bits: overrideBits)
            addAffine(
                "\(prefix).mlp.gate_proj",
                leading: [intermediate], inFeatures: hidden, bits: overrideBits)
            addAffine(
                "\(prefix).mlp.up_proj",
                leading: [intermediate], inFeatures: hidden, bits: overrideBits)
            addAffine(
                "\(prefix).mlp.down_proj",
                leading: [hidden], inFeatures: intermediate,
                bits: overrideBits)

            addAffine(
                "\(prefix).experts.switch_glu.gate_proj",
                leading: [experts, moeIntermediate], inFeatures: hidden,
                bits: fallbackBits)
            addAffine(
                "\(prefix).experts.switch_glu.up_proj",
                leading: [experts, moeIntermediate], inFeatures: hidden,
                bits: fallbackBits)
            addAffine(
                "\(prefix).experts.switch_glu.down_proj",
                leading: [experts, hidden], inFeatures: moeIntermediate,
                bits: fallbackBits)
        }

        precondition(
            inventory.count == expectedTensorCount,
            "Gemma 4 26B A4B inventory must contain \(expectedTensorCount) "
                + "tensors"
        )
        return inventory
    }

    static func validateSelectedTensors(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        quantization: Gemma4A4BTransformQuantizationSpec
    ) throws {
        let forbiddenSuffixes = [
            ".weight_packed",
            ".input_global_scale",
            ".weight_global_scale",
            ".k_scale",
            ".v_scale",
        ]
        if let forbiddenName = selectedKeys.sorted().first(where: { name in
            forbiddenSuffixes.contains { suffix in name.hasSuffix(suffix) }
        }) {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B MLX transform rejects "
                    + "compressed-tensors/global-scale and FP8 KV-scale tensor "
                    + "\(forbiddenName)"
            )
        }
        if let headName = selectedKeys.sorted().first(where: { name in
            name.split(separator: ".").contains("lm_head")
        }) {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B has tied embeddings and must not select an "
                    + "output-head tensor \(headName)"
            )
        }
        if let mtpName = selectedKeys.sorted().first(where: { name in
            name.split(separator: ".").contains("mtp")
        }) {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B backbone transform must not select MTP tensor "
                    + "\(mtpName); the MTP head is a separately pinned artifact"
            )
        }

        for name in selectedKeys.sorted() where name.hasSuffix(".weight") {
            let stem = String(name.dropLast(".weight".count))
            let scalesName = "\(stem).scales"
            let biasesName = "\(stem).biases"
            guard selectedKeys.contains(scalesName)
                    || selectedKeys.contains(biasesName)
            else {
                continue
            }
            guard selectedKeys.contains(scalesName),
                  selectedKeys.contains(biasesName)
            else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B affine projection \(stem) must ship both "
                        + ".scales and .biases"
                )
            }
            let weightInfo = try tensorInfo(
                named: name, index: index, headers: headers)
            let scalesInfo = try tensorInfo(
                named: scalesName, index: index, headers: headers)
            let biasesInfo = try tensorInfo(
                named: biasesName, index: index, headers: headers)
            guard weightInfo.dtype == TensorDType.u32.rawValue,
                  scalesInfo.dtype == TensorDType.bf16.rawValue,
                  biasesInfo.dtype == TensorDType.bf16.rawValue,
                  weightInfo.shape.count >= 2,
                  scalesInfo.shape == biasesInfo.shape,
                  scalesInfo.shape.count == weightInfo.shape.count,
                  weightInfo.shape.dropLast() == scalesInfo.shape.dropLast(),
                  weightInfo.shape.allSatisfy({ $0 > 0 }),
                  scalesInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B affine projection \(stem) has "
                        + "incompatible weight, scale, or bias metadata"
                )
            }

            let pathSpec = quantization.spec(forPath: stem)
            guard let packedWidth = weightInfo.shape.last,
                  let groupCount = scalesInfo.shape.last
            else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B affine projection \(stem) has no packed "
                        + "axis"
                )
            }
            let (inputFeatures, inputOverflow) =
                groupCount.multipliedReportingOverflow(by: pathSpec.groupSize)
            let (packedBits, packedOverflow) =
                packedWidth.multipliedReportingOverflow(by: 32)
            guard !inputOverflow, !packedOverflow else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B projection \(stem) packed width overflows "
                        + "Int"
                )
            }
            let (expectedPackedBits, expectedOverflow) =
                inputFeatures.multipliedReportingOverflow(by: pathSpec.bits)
            guard !expectedOverflow, packedBits == expectedPackedBits else {
                throw MLXFastError.invalidInput(
                    "quantized Gemma 4 26B A4B projection \(stem) stored width "
                        + "\(packedWidth) does not match config quantization "
                        + "group_size \(pathSpec.groupSize) bits "
                        + "\(pathSpec.bits) for input dimension "
                        + "\(inputFeatures)"
                )
            }
        }

        try validateExactPublicInventory(
            selectedKeys: selectedKeys,
            index: index,
            headers: headers
        )
    }

    static func validateExactPublicInventory(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws {
        let expected = expectedTensorInventory()
        let expectedNames = Set(expected.keys)
        let headerNameList = headers.values.flatMap { $0.tensors.keys }
        let headerNames = Set(headerNameList)

        guard selectedKeys == expectedNames,
              expectedNames.isSubset(of: Set(index.weightMap.keys)),
              expectedNames.isSubset(of: headerNames),
              headerNameList.count == headerNames.count
        else {
            let missing = expectedNames.subtracting(selectedKeys).sorted()
            let extra = selectedKeys.subtracting(expectedNames).sorted()
            let unindexed = expectedNames
                .subtracting(Set(index.weightMap.keys)).sorted()
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B checkpoint tensor inventory must match the "
                    + "exact public \(expectedTensorCount)-tensor contract "
                    + "(missing: \(missing.prefix(8).joined(separator: ", ")); "
                    + "extra: \(extra.prefix(8).joined(separator: ", ")); "
                    + "unindexed/duplicate header tensors: "
                    + "\(unindexed.prefix(8).joined(separator: ", ")))"
            )
        }

        for name in expected.keys.sorted() {
            guard let expectedMetadata = expected[name] else {
                preconditionFailure(
                    "missing expected Gemma 4 26B A4B metadata for \(name)")
            }
            let actual = try tensorInfo(
                named: name, index: index, headers: headers)
            guard actual.dtype == expectedMetadata.dtype,
                  actual.shape == expectedMetadata.shape
            else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B tensor \(name) dtype/shape "
                        + "\(actual.dtype) \(actual.shape) does not match "
                        + "exact public metadata \(expectedMetadata.dtype) "
                        + "\(expectedMetadata.shape)"
                )
            }
        }
    }

    private static func tensorInfo(
        named name: String,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws -> SafetensorInfo {
        guard let shardName = index.weightMap[name],
              let info = headers[shardName]?.tensors[name]
        else {
            throw MLXFastError.invalidInput(
                "missing validated tensor metadata for \(name)")
        }
        return info
    }

    private static func intField(
        _ key: String, in object: [String: Any]
    ) throws -> Int {
        guard let number = object[key] as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number),
              let integer = Int(number.stringValue)
        else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B quantization field \(key) must be a finite "
                    + "integer in Int range"
            )
        }
        return integer
    }

    private static func stringField(
        _ key: String, in object: [String: Any]
    ) throws -> String {
        guard let string = object[key] as? String else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B quantization field \(key) must be a string"
            )
        }
        return string
    }
}
