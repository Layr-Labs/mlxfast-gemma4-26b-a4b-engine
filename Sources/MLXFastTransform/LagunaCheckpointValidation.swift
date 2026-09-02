import CoreFoundation
import Foundation
import MLXFastCore

struct LagunaTransformQuantizationSpec: Equatable {
    struct Override: Equatable {
        let groupSize: Int
        let bits: Int
    }

    let groupSize: Int
    let bits: Int
    let mode: String
    let overrides: [String: Override]

    func expected(forTensorStem stem: String) -> (groupSize: Int, bits: Int) {
        if let override = overrides[stem] {
            return (override.groupSize, override.bits)
        }
        return (groupSize, bits)
    }
}

enum LagunaCheckpointValidation {
    struct ExpectedTensorMetadata: Equatable {
        let dtype: String
        let shape: [Int]
    }

    struct RecognizedQuantizedStem: Equatable {
        enum Kind: Equatable {
            case matrix
            case stackedExperts
        }

        let kind: Kind
        let layerIndex: Int?
    }

    private static let layerPrefix = "model.layers."

    static func recognizedQuantizedStem(forWeightName name: String) -> RecognizedQuantizedStem? {
        guard name.hasSuffix(".weight") else {
            return nil
        }
        guard name.hasPrefix(layerPrefix) else {
            return nil
        }
        let remainder = name.dropFirst(layerPrefix.count)
        guard let separator = remainder.firstIndex(of: "."),
              let layerIndex = Int(remainder[..<separator])
        else {
            return nil
        }
        switch String(remainder[separator...]) {
        case ".mlp.shared_expert.gate_proj.weight",
             ".mlp.shared_expert.up_proj.weight",
             ".mlp.shared_expert.down_proj.weight":
            return RecognizedQuantizedStem(kind: .matrix, layerIndex: layerIndex)
        case ".mlp.switch_mlp.gate_proj.weight",
             ".mlp.switch_mlp.up_proj.weight",
             ".mlp.switch_mlp.down_proj.weight":
            return RecognizedQuantizedStem(kind: .stackedExperts, layerIndex: layerIndex)
        default:
            return nil
        }
    }

    static func quantizationSpec(
        fromConfigRoot root: [String: Any]
    ) throws -> LagunaTransformQuantizationSpec {
        func requiredBlock(_ key: String) throws -> LagunaTransformQuantizationSpec {
            guard let value = root[key] else {
                throw MLXFastError.invalidInput("Laguna config is missing required \(key)")
            }
            guard let block = value as? [String: Any] else {
                throw MLXFastError.invalidInput("Laguna config \(key) must be an object")
            }

            let allowedKeys: Set<String> = ["group_size", "bits", "mode"]
            let unexpectedKeys = Set(block.keys).subtracting(allowedKeys)
            guard unexpectedKeys.isEmpty else {
                throw MLXFastError.invalidInput(
                    "Laguna config \(key) contains unsupported fields: \(unexpectedKeys.sorted().joined(separator: ", "))"
                )
            }
            guard block["group_size"] != nil, block["bits"] != nil, block["mode"] != nil else {
                throw MLXFastError.invalidInput(
                    "Laguna config \(key) must explicitly define group_size, bits, and mode"
                )
            }

            let groupSize = try intField("group_size", in: block, defaultValue: 0)
            let bits = try intField("bits", in: block, defaultValue: 0)
            let mode = try stringField("mode", in: block, defaultValue: "")
            guard mode == "nvfp4",
                  groupSize == 16,
                  bits == 4
            else {
                throw MLXFastError.invalidInput(
                    "Poolside Laguna quantization must be NVFP4 4-bit group_size 16"
                )
            }
            return LagunaTransformQuantizationSpec(
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                overrides: [:]
            )
        }

        let quantization = try requiredBlock("quantization")
        let quantizationConfig = try requiredBlock("quantization_config")
        guard quantization == quantizationConfig else {
            throw MLXFastError.invalidInput(
                "Laguna config quantization and quantization_config must match exactly"
            )
        }
        return quantization
    }

    static func validateSelectedTensors(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        quantization: LagunaTransformQuantizationSpec
    ) throws {
        let forbiddenSuffixes = [
            ".weight_packed",
            ".input_global_scale",
            ".weight_global_scale",
            ".k_scale",
            ".v_scale",
            ".biases",
        ]
        if let forbiddenName = selectedKeys.sorted().first(where: { name in
            forbiddenSuffixes.contains { suffix in name.hasSuffix(suffix) }
        }) {
            throw MLXFastError.invalidInput(
                "Poolside Laguna MLX transform rejects compressed-tensors/global-scale, "
                    + "FP8 KV-scale, and affine-bias tensor \(forbiddenName)"
            )
        }

        var routerExpertsByLayer: [Int: Int] = [:]
        var stackedExpertsByLayer: [Int: Int] = [:]

        for name in selectedKeys.sorted() {
            guard let recognized = recognizedQuantizedStem(forWeightName: name) else {
                continue
            }
            let stem = String(name.dropLast(".weight".count))
            let scalesName = "\(stem).scales"
            let biasesName = "\(stem).biases"
            let hasScales = selectedKeys.contains(scalesName)
            let hasBiases = selectedKeys.contains(biasesName)
            let weightInfo = try tensorInfo(named: name, index: index, headers: headers)
            guard weightInfo.dtype == TensorDType.u32.rawValue,
                  hasScales,
                  !hasBiases
            else {
                throw MLXFastError.invalidInput(
                    "Poolside NVFP4 expert \(name) must use U32 packed weights, U8 scales, and no affine biases"
                )
            }
            let scalesInfo = try tensorInfo(named: scalesName, index: index, headers: headers)
            let expectedRank = recognized.kind == .stackedExperts ? 3 : 2
            guard scalesInfo.dtype == TensorDType.u8.rawValue,
                  weightInfo.shape.count == expectedRank,
                  scalesInfo.shape.count == expectedRank,
                  weightInfo.shape.dropLast() == scalesInfo.shape.dropLast(),
                  weightInfo.shape.allSatisfy({ $0 > 0 }),
                  scalesInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "Poolside NVFP4 expert \(stem) has incompatible weight or scale metadata"
                )
            }

            let expected = quantization.expected(forTensorStem: stem)
            let packedWidth = weightInfo.shape[expectedRank - 1]
            let groupCount = scalesInfo.shape[expectedRank - 1]
            let (inputFeatures, inputOverflow) = groupCount.multipliedReportingOverflow(
                by: expected.groupSize
            )
            guard !inputOverflow else {
                throw MLXFastError.invalidInput(
                    "Laguna projection \(stem) input dimension overflows Int"
                )
            }
            let (packedBits, packedOverflow) = packedWidth.multipliedReportingOverflow(by: 32)
            let (expectedPackedBits, expectedOverflow) = inputFeatures.multipliedReportingOverflow(
                by: expected.bits
            )
            guard !packedOverflow, !expectedOverflow else {
                throw MLXFastError.invalidInput(
                    "Laguna projection \(stem) packed width overflows Int"
                )
            }
            guard packedBits == expectedPackedBits else {
                throw MLXFastError.invalidInput(
                    "quantized Laguna projection \(stem) stored width \(packedWidth) does not "
                        + "match config quantization group_size \(expected.groupSize) "
                        + "bits \(expected.bits) for input dimension \(inputFeatures)"
                )
            }

            switch recognized.kind {
            case .matrix:
                break
            case .stackedExperts:
                if let layerIndex = recognized.layerIndex {
                    let experts = weightInfo.shape[0]
                    if let existing = stackedExpertsByLayer[layerIndex], existing != experts {
                        throw MLXFastError.invalidInput(
                            "Laguna layer \(layerIndex) stacked expert tensors disagree on the "
                                + "expert count (\(existing) vs \(experts))"
                        )
                    }
                    stackedExpertsByLayer[layerIndex] = experts
                }
            }
        }

        for name in selectedKeys.sorted() where name.hasSuffix(".mlp.gate.weight") {
            guard let layerIndex = layerIndex(from: name) else {
                throw MLXFastError.invalidInput("invalid Poolside Laguna router tensor name \(name)")
            }
            let weightInfo = try tensorInfo(named: name, index: index, headers: headers)
            guard weightInfo.dtype == TensorDType.bf16.rawValue,
                  weightInfo.shape.count == 2,
                  weightInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "Poolside Laguna router \(name) must be a non-empty BF16 matrix"
                )
            }
            try validateRouterCorrectionBias(
                routerName: name,
                routerRows: weightInfo.shape[0],
                selectedKeys: selectedKeys,
                index: index,
                headers: headers
            )
            routerExpertsByLayer[layerIndex] = weightInfo.shape[0]
        }

        for (layerIndex, experts) in stackedExpertsByLayer.sorted(by: { $0.key < $1.key }) {
            guard let routerExperts = routerExpertsByLayer[layerIndex] else {
                throw MLXFastError.invalidInput(
                    "Laguna layer \(layerIndex) has NVFP4 expert stacks but no BF16 router"
                )
            }
            if routerExperts != experts {
                throw MLXFastError.invalidInput(
                    "Laguna layer \(layerIndex) router rows (\(routerExperts)) do not match the "
                        + "stacked expert count (\(experts))"
                )
            }
        }

        try validateExactPublicInventory(
            selectedKeys: selectedKeys,
            index: index,
            headers: headers
        )
    }

    static func expectedTensorInventory() -> [String: ExpectedTensorMetadata] {
        let hiddenSize = 2_048
        let denseIntermediateSize = 8_192
        let layerCount = 40
        let headDimension = 128
        let keyValueHeads = 8
        let expertCount = 256
        let expertIntermediateSize = 512
        let groupSize = 16
        let bits = 4

        var inventory: [String: ExpectedTensorMetadata] = [:]
        func add(_ name: String, _ dtype: TensorDType, _ shape: [Int]) {
            precondition(inventory[name] == nil, "duplicate expected Laguna tensor \(name)")
            inventory[name] = ExpectedTensorMetadata(dtype: dtype.rawValue, shape: shape)
        }
        func addNVFP4(_ stem: String, leadingShape: [Int], inputFeatures: Int) {
            add(
                "\(stem).weight",
                .u32,
                leadingShape + [inputFeatures * bits / 32]
            )
            add(
                "\(stem).scales",
                .u8,
                leadingShape + [inputFeatures / groupSize]
            )
        }

        add("model.embed_tokens.weight", .bf16, [100_352, hiddenSize])
        add("model.norm.weight", .bf16, [hiddenSize])
        add("lm_head.weight", .bf16, [100_352, hiddenSize])

        for layerIndex in 0..<layerCount {
            let prefix = "model.layers.\(layerIndex)"
            let queryHeads = layerIndex.isMultiple(of: 4) ? 48 : 64
            let queryWidth = queryHeads * headDimension
            let keyValueWidth = keyValueHeads * headDimension

            add("\(prefix).input_layernorm.weight", .bf16, [hiddenSize])
            add("\(prefix).post_attention_layernorm.weight", .bf16, [hiddenSize])
            add("\(prefix).self_attn.q_proj.weight", .bf16, [queryWidth, hiddenSize])
            add("\(prefix).self_attn.k_proj.weight", .bf16, [keyValueWidth, hiddenSize])
            add("\(prefix).self_attn.v_proj.weight", .bf16, [keyValueWidth, hiddenSize])
            add("\(prefix).self_attn.o_proj.weight", .bf16, [hiddenSize, queryWidth])
            add("\(prefix).self_attn.g_proj.weight", .bf16, [queryHeads, hiddenSize])
            add("\(prefix).self_attn.q_norm.weight", .bf16, [headDimension])
            add("\(prefix).self_attn.k_norm.weight", .bf16, [headDimension])

            if layerIndex == 0 {
                add(
                    "\(prefix).mlp.gate_proj.weight",
                    .bf16,
                    [denseIntermediateSize, hiddenSize]
                )
                add(
                    "\(prefix).mlp.up_proj.weight",
                    .bf16,
                    [denseIntermediateSize, hiddenSize]
                )
                add(
                    "\(prefix).mlp.down_proj.weight",
                    .bf16,
                    [hiddenSize, denseIntermediateSize]
                )
                continue
            }

            add("\(prefix).mlp.gate.weight", .bf16, [expertCount, hiddenSize])
            add(
                "\(prefix).mlp.gate.e_score_correction_bias",
                .f32,
                [expertCount]
            )
            for projection in ["gate_proj", "up_proj"] {
                addNVFP4(
                    "\(prefix).mlp.switch_mlp.\(projection)",
                    leadingShape: [expertCount, expertIntermediateSize],
                    inputFeatures: hiddenSize
                )
                addNVFP4(
                    "\(prefix).mlp.shared_expert.\(projection)",
                    leadingShape: [expertIntermediateSize],
                    inputFeatures: hiddenSize
                )
            }
            addNVFP4(
                "\(prefix).mlp.switch_mlp.down_proj",
                leadingShape: [expertCount, hiddenSize],
                inputFeatures: expertIntermediateSize
            )
            addNVFP4(
                "\(prefix).mlp.shared_expert.down_proj",
                leadingShape: [hiddenSize],
                inputFeatures: expertIntermediateSize
            )
        }

        precondition(inventory.count == 912, "Poolside Laguna inventory must contain 912 tensors")
        return inventory
    }

    static func validateExactPublicInventory(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws {
        let expected = expectedTensorInventory()
        let expectedNames = Set(expected.keys)
        let indexedNames = Set(index.weightMap.keys)
        let headerNameList = headers.values.flatMap { $0.tensors.keys }
        let headerNames = Set(headerNameList)

        guard selectedKeys == expectedNames,
              indexedNames == expectedNames,
              headerNames == expectedNames,
              headerNameList.count == expectedNames.count
        else {
            let missing = expectedNames.subtracting(indexedNames).sorted()
            let extra = indexedNames.subtracting(expectedNames).sorted()
            let unindexed = headerNames.subtracting(indexedNames).sorted()
            throw MLXFastError.invalidInput(
                "Poolside Laguna checkpoint tensor inventory must match the exact public "
                    + "912-tensor contract (missing: \(missing.prefix(8).joined(separator: ", ")); "
                    + "extra: \(extra.prefix(8).joined(separator: ", ")); "
                    + "unindexed/duplicate header tensors: "
                    + "\(unindexed.prefix(8).joined(separator: ", ")))"
            )
        }

        for name in expected.keys.sorted() {
            guard let expectedMetadata = expected[name] else {
                preconditionFailure("missing expected Laguna metadata for \(name)")
            }
            let actual = try tensorInfo(named: name, index: index, headers: headers)
            guard actual.dtype == expectedMetadata.dtype,
                  actual.shape == expectedMetadata.shape
            else {
                throw MLXFastError.invalidInput(
                    "Poolside Laguna tensor \(name) dtype/shape \(actual.dtype) \(actual.shape) "
                        + "does not match exact public metadata "
                        + "\(expectedMetadata.dtype) \(expectedMetadata.shape)"
                )
            }
        }
    }

    private static func validateRouterCorrectionBias(
        routerName: String,
        routerRows: Int,
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws {
        let routerStem = String(routerName.dropLast(".weight".count))
        let biasName = routerStem + ".e_score_correction_bias"
        guard selectedKeys.contains(biasName) else {
            throw MLXFastError.invalidInput(
                "Laguna router \(routerStem) is missing its e_score_correction_bias tensor"
            )
        }
        let biasInfo = try tensorInfo(named: biasName, index: index, headers: headers)
        guard biasInfo.dtype == TensorDType.f32.rawValue,
              biasInfo.shape == [routerRows]
        else {
            throw MLXFastError.invalidInput(
                "Laguna router correction bias \(biasName) must be F32 with one entry per expert"
            )
        }
    }

    private static func layerIndex(from name: String) -> Int? {
        guard name.hasPrefix(layerPrefix) else { return nil }
        let remainder = name.dropFirst(layerPrefix.count)
        guard let separator = remainder.firstIndex(of: ".") else { return nil }
        return Int(remainder[..<separator])
    }

    private static func tensorInfo(
        named name: String,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws -> SafetensorInfo {
        guard let shardName = index.weightMap[name],
              let info = headers[shardName]?.tensors[name]
        else {
            throw MLXFastError.invalidInput("missing validated tensor metadata for \(name)")
        }
        return info
    }

    private static func validateGroupAndBits(
        groupSize: Int,
        bits: Int,
        context: String
    ) throws {
        guard groupSize > 0, [2, 4, 8].contains(bits) else {
            throw MLXFastError.invalidInput(
                "Laguna \(context) must use a positive group size and bits in {2, 4, 8}"
            )
        }
    }

    private static func intField(
        _ key: String,
        in object: [String: Any],
        defaultValue: Int
    ) throws -> Int {
        guard let value = object[key], !(value is NSNull) else {
            return defaultValue
        }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number),
              let integer = Int(number.stringValue)
        else {
            throw MLXFastError.invalidInput(
                "Laguna quantization field \(key) must be a finite integer in Int range"
            )
        }
        return integer
    }

    private static func stringField(
        _ key: String,
        in object: [String: Any],
        defaultValue: String
    ) throws -> String {
        guard let value = object[key], !(value is NSNull) else {
            return defaultValue
        }
        guard let string = value as? String else {
            throw MLXFastError.invalidInput(
                "Laguna quantization field \(key) must be a string"
            )
        }
        return string
    }
}
