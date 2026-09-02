import CoreFoundation
import Foundation
import MLXFastCore

struct Qwen35TransformQuantizationSpec: Equatable {
    let groupSize: Int
    let bits: Int
    let mode: String
}

enum Qwen35CheckpointValidation {
    struct ExpectedTensorMetadata: Equatable {
        let dtype: String
        let shape: [Int]
    }

    enum PinnedGeometry {
        static let vocabSize = 248_320
        static let hiddenSize = 5_120
        static let intermediateSize = 17_408
        static let layerCount = 64
        static let fullAttentionInterval = 4
        static let attentionHeads = 24
        static let keyValueHeads = 4
        static let headDim = 256
        static let linearValueHeads = 48
        static let linearKeyHeads = 16
        static let linearValueHeadDim = 128
        static let linearKeyHeadDim = 128
        static let linearConvKernelDim = 4
        static let quantizationGroupSize = 64
        static let quantizationBits = 4
        static let quantizationMode = "affine"
    }

    static let expectedTensorCount = 1_847

    static let textTowerPrefix = "language_model."

    private static let layerPrefix = "language_model.model.layers."

    static func quantizationSpec(
        fromConfigRoot root: [String: Any]
    ) throws -> Qwen35TransformQuantizationSpec {
        func parseBlock(_ key: String) throws -> Qwen35TransformQuantizationSpec? {
            guard let value = root[key], !(value is NSNull) else {
                return nil
            }
            guard let block = value as? [String: Any] else {
                throw MLXFastError.invalidInput("Qwen3.6 config \(key) must be an object")
            }
            let allowedKeys: Set<String> = ["group_size", "bits", "mode"]
            let unexpectedKeys = Set(block.keys).subtracting(allowedKeys)
            guard unexpectedKeys.isEmpty else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 config \(key) contains unsupported fields: "
                        + unexpectedKeys.sorted().joined(separator: ", ")
                )
            }
            guard block["group_size"] != nil, block["bits"] != nil, block["mode"] != nil else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 config \(key) must explicitly define group_size, bits, and mode"
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
                    "Qwen3.6 quantization must be affine 4-bit group_size 64"
                )
            }
            return Qwen35TransformQuantizationSpec(
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
        }

        let quantization = try parseBlock("quantization")
        let quantizationConfig = try parseBlock("quantization_config")
        if let quantization, let quantizationConfig {
            guard quantization == quantizationConfig else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 config quantization and quantization_config must match exactly"
                )
            }
            return quantization
        }
        guard let spec = quantization ?? quantizationConfig else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 config is missing both quantization and quantization_config"
            )
        }
        return spec
    }

    static func expectedTensorInventory() -> [String: ExpectedTensorMetadata] {
        let hidden = PinnedGeometry.hiddenSize
        let intermediate = PinnedGeometry.intermediateSize
        let vocab = PinnedGeometry.vocabSize
        let bits = PinnedGeometry.quantizationBits
        let groupSize = PinnedGeometry.quantizationGroupSize

        var inventory: [String: ExpectedTensorMetadata] = [:]
        func add(_ name: String, _ dtype: TensorDType, _ shape: [Int]) {
            precondition(inventory[name] == nil, "duplicate expected Qwen3.6 tensor \(name)")
            inventory[name] = ExpectedTensorMetadata(dtype: dtype.rawValue, shape: shape)
        }
        func addAffine(_ stem: String, outFeatures: Int, inFeatures: Int) {
            add("\(stem).weight", .u32, [outFeatures, inFeatures * bits / 32])
            add("\(stem).scales", .bf16, [outFeatures, inFeatures / groupSize])
            add("\(stem).biases", .bf16, [outFeatures, inFeatures / groupSize])
        }

        addAffine(
            "language_model.model.embed_tokens",
            outFeatures: vocab,
            inFeatures: hidden
        )
        add("language_model.model.norm.weight", .bf16, [hidden])
        addAffine("language_model.lm_head", outFeatures: vocab, inFeatures: hidden)

        let linearKeySize = PinnedGeometry.linearKeyHeads * PinnedGeometry.linearKeyHeadDim
        let linearValueSize = PinnedGeometry.linearValueHeads * PinnedGeometry.linearValueHeadDim
        let linearConvSize = linearKeySize * 2 + linearValueSize
        let fullQuerySize = PinnedGeometry.attentionHeads * PinnedGeometry.headDim * 2
        let fullKeyValueSize = PinnedGeometry.keyValueHeads * PinnedGeometry.headDim
        let fullOutputSize = PinnedGeometry.attentionHeads * PinnedGeometry.headDim

        for layerIndex in 0..<PinnedGeometry.layerCount {
            let prefix = "\(layerPrefix)\(layerIndex)"
            add("\(prefix).input_layernorm.weight", .bf16, [hidden])
            add("\(prefix).post_attention_layernorm.weight", .bf16, [hidden])

            addAffine("\(prefix).mlp.gate_proj", outFeatures: intermediate, inFeatures: hidden)
            addAffine("\(prefix).mlp.up_proj", outFeatures: intermediate, inFeatures: hidden)
            addAffine("\(prefix).mlp.down_proj", outFeatures: hidden, inFeatures: intermediate)

            if layerIndex % PinnedGeometry.fullAttentionInterval
                == PinnedGeometry.fullAttentionInterval - 1
            {
                add("\(prefix).self_attn.q_norm.weight", .bf16, [PinnedGeometry.headDim])
                add("\(prefix).self_attn.k_norm.weight", .bf16, [PinnedGeometry.headDim])
                addAffine(
                    "\(prefix).self_attn.q_proj",
                    outFeatures: fullQuerySize,
                    inFeatures: hidden
                )
                addAffine(
                    "\(prefix).self_attn.k_proj",
                    outFeatures: fullKeyValueSize,
                    inFeatures: hidden
                )
                addAffine(
                    "\(prefix).self_attn.v_proj",
                    outFeatures: fullKeyValueSize,
                    inFeatures: hidden
                )
                addAffine(
                    "\(prefix).self_attn.o_proj",
                    outFeatures: hidden,
                    inFeatures: fullOutputSize
                )
                continue
            }

            add(
                "\(prefix).linear_attn.conv1d.weight",
                .bf16,
                [linearConvSize, PinnedGeometry.linearConvKernelDim, 1]
            )
            add("\(prefix).linear_attn.A_log", .bf16, [PinnedGeometry.linearValueHeads])
            add("\(prefix).linear_attn.dt_bias", .bf16, [PinnedGeometry.linearValueHeads])
            add("\(prefix).linear_attn.norm.weight", .bf16, [PinnedGeometry.linearValueHeadDim])
            addAffine(
                "\(prefix).linear_attn.in_proj_qkv",
                outFeatures: linearConvSize,
                inFeatures: hidden
            )
            addAffine(
                "\(prefix).linear_attn.in_proj_z",
                outFeatures: linearValueSize,
                inFeatures: hidden
            )
            addAffine(
                "\(prefix).linear_attn.in_proj_b",
                outFeatures: PinnedGeometry.linearValueHeads,
                inFeatures: hidden
            )
            addAffine(
                "\(prefix).linear_attn.in_proj_a",
                outFeatures: PinnedGeometry.linearValueHeads,
                inFeatures: hidden
            )
            addAffine(
                "\(prefix).linear_attn.out_proj",
                outFeatures: hidden,
                inFeatures: linearValueSize
            )
        }

        precondition(
            inventory.count == expectedTensorCount,
            "Qwen3.6 inventory must contain \(expectedTensorCount) tensors"
        )
        return inventory
    }

    static func validateSelectedTensors(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        quantization: Qwen35TransformQuantizationSpec
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
                "Qwen3.6 MLX transform rejects compressed-tensors/global-scale and "
                    + "FP8 KV-scale tensor \(forbiddenName)"
            )
        }
        if let mtpName = selectedKeys.sorted().first(where: { name in
            name.split(separator: ".").contains("mtp")
        }) {
            throw MLXFastError.invalidInput(
                "Qwen3.6 backbone transform must not select MTP tensor \(mtpName); the MTP "
                    + "head is a separately pinned artifact"
            )
        }

        for name in selectedKeys.sorted() where name.hasSuffix(".weight") {
            let stem = String(name.dropLast(".weight".count))
            let scalesName = "\(stem).scales"
            let biasesName = "\(stem).biases"
            guard selectedKeys.contains(scalesName) || selectedKeys.contains(biasesName) else {
                continue
            }
            guard selectedKeys.contains(scalesName), selectedKeys.contains(biasesName) else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 affine projection \(stem) must ship both .scales and .biases"
                )
            }
            let weightInfo = try tensorInfo(named: name, index: index, headers: headers)
            let scalesInfo = try tensorInfo(named: scalesName, index: index, headers: headers)
            let biasesInfo = try tensorInfo(named: biasesName, index: index, headers: headers)
            guard weightInfo.dtype == TensorDType.u32.rawValue,
                  scalesInfo.dtype == TensorDType.bf16.rawValue,
                  biasesInfo.dtype == TensorDType.bf16.rawValue,
                  weightInfo.shape.count == 2,
                  scalesInfo.shape == biasesInfo.shape,
                  scalesInfo.shape.count == 2,
                  weightInfo.shape[0] == scalesInfo.shape[0],
                  weightInfo.shape.allSatisfy({ $0 > 0 }),
                  scalesInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 affine projection \(stem) has incompatible weight, scale, or "
                        + "bias metadata"
                )
            }

            let packedWidth = weightInfo.shape[1]
            let groupCount = scalesInfo.shape[1]
            let (inputFeatures, inputOverflow) = groupCount.multipliedReportingOverflow(
                by: quantization.groupSize
            )
            let (packedBits, packedOverflow) = packedWidth.multipliedReportingOverflow(by: 32)
            guard !inputOverflow, !packedOverflow else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 projection \(stem) packed width overflows Int"
                )
            }
            let (expectedPackedBits, expectedOverflow) = inputFeatures.multipliedReportingOverflow(
                by: quantization.bits
            )
            guard !expectedOverflow, packedBits == expectedPackedBits else {
                throw MLXFastError.invalidInput(
                    "quantized Qwen3.6 projection \(stem) stored width \(packedWidth) does not "
                        + "match config quantization group_size \(quantization.groupSize) "
                        + "bits \(quantization.bits) for input dimension \(inputFeatures)"
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
            let unindexed = expectedNames.subtracting(Set(index.weightMap.keys)).sorted()
            throw MLXFastError.invalidInput(
                "Qwen3.6 checkpoint tensor inventory must match the exact public "
                    + "\(expectedTensorCount)-tensor contract "
                    + "(missing: \(missing.prefix(8).joined(separator: ", ")); "
                    + "extra: \(extra.prefix(8).joined(separator: ", ")); "
                    + "unindexed/duplicate header tensors: "
                    + "\(unindexed.prefix(8).joined(separator: ", ")))"
            )
        }

        for name in expected.keys.sorted() {
            guard let expectedMetadata = expected[name] else {
                preconditionFailure("missing expected Qwen3.6 metadata for \(name)")
            }
            let actual = try tensorInfo(named: name, index: index, headers: headers)
            guard actual.dtype == expectedMetadata.dtype,
                  actual.shape == expectedMetadata.shape
            else {
                throw MLXFastError.invalidInput(
                    "Qwen3.6 tensor \(name) dtype/shape \(actual.dtype) \(actual.shape) "
                        + "does not match exact public metadata "
                        + "\(expectedMetadata.dtype) \(expectedMetadata.shape)"
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
            throw MLXFastError.invalidInput("missing validated tensor metadata for \(name)")
        }
        return info
    }

    private static func intField(_ key: String, in object: [String: Any]) throws -> Int {
        guard let number = object[key] as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number),
              let integer = Int(number.stringValue)
        else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 quantization field \(key) must be a finite integer in Int range"
            )
        }
        return integer
    }

    private static func stringField(_ key: String, in object: [String: Any]) throws -> String {
        guard let string = object[key] as? String else {
            throw MLXFastError.invalidInput(
                "Qwen3.6 quantization field \(key) must be a string"
            )
        }
        return string
    }
}
