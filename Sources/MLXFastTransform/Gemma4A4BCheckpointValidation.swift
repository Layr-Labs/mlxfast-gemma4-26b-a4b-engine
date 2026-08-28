import CoreFoundation
import Foundation
import MLXFastCore

/// One resolved affine width, as it appears either in the checkpoint's
/// fallback scalars or in one of its per-tensor override objects.
struct Gemma4A4BTransformTensorQuantization: Equatable {
    let groupSize: Int
    let bits: Int
}

/// Quantization expectations parsed from the pinned Gemma 4 26B A4B
/// checkpoint's `quantization` / `quantization_config` blocks.
///
/// UNLIKE EVERY OTHER TARGET THIS TRANSFORM CARRIES, the block is not a
/// three-key object. It is affine 4-bit / group-64 PLUS 120 per-tensor
/// overrides promoting four projection families to 8 bits on all 30 layers.
/// The Qwen-era spec type modelled exactly `{group_size, bits, mode}` and
/// REJECTED anything else; ported verbatim it would refuse this checkpoint
/// outright, and relaxed the obvious way (ignore unknown keys) it would accept
/// it while discarding every promotion. So the override table is modelled as
/// data and `spec(forPath:)` is the only way widths are read.
struct Gemma4A4BTransformQuantizationSpec: Equatable {
    /// Applies to any path not named in `overrides`.
    let groupSize: Int
    let bits: Int
    /// Affine for this checkpoint. Overrides carry no mode of their own, so
    /// this applies to fallback and overrides alike.
    let mode: String
    /// Checkpoint tensor path -> width, e.g.
    /// `language_model.model.layers.0.mlp.gate_proj`.
    let overrides: [String: Gemma4A4BTransformTensorQuantization]

    var fallback: Gemma4A4BTransformTensorQuantization {
        Gemma4A4BTransformTensorQuantization(groupSize: groupSize, bits: bits)
    }

    /// Resolve the width for one checkpoint tensor stem.
    func spec(forPath path: String) -> Gemma4A4BTransformTensorQuantization {
        overrides[path] ?? fallback
    }
}

/// Transform-side structural validation of the Gemma 4 26B A4B text-tower
/// tensor set.
///
/// The source checkpoint (`mlx-community/gemma-4-26B-A4B-it-qat-4bit`) is
/// already MLX affine-quantized, so the transform passes tensors through
/// unchanged. This pass fails fast -- before the multi-GB copy -- when the set
/// it would copy cannot satisfy the runtime loader
/// (`Gemma4A4BWeightLoader`):
///
/// - the selected `language_model.*` namespace must be EXACTLY the public
///   1,339-tensor inventory, tensor for tensor, at the exact dtype and shape;
/// - every quantized projection stored as packed U32 codes must ship BF16
///   `.scales` AND BF16 `.biases` (affine needs both; the Laguna NVFP4
///   contract forbids `.biases`, which is why the two validators cannot share
///   one rule) with matching leading dimensions;
/// - each packed width must match the group size and bit width the emitted
///   config.json declares FOR THAT PATH -- 4-bit group-64 by default, 8-bit
///   group-64 on the 120 promoted tensors. A validator that used one global
///   width here would accept a 4-bit-packed `mlp.gate_proj` on a checkpoint
///   that declares it at 8 bits, which is precisely the silent-numerics
///   failure the override table exists to prevent;
/// - compressed-tensors aliases, global-scale tensors, and FP8 KV scales are
///   rejected outright;
/// - no `lm_head.*` tensor may appear. `tie_word_embeddings` is true, so the
///   output projection IS the embedding; a checkpoint that ships an untied
///   head is a different artifact;
/// - no `mtp.*` tensor may appear. The Gemma 4 MTP head is a separately pinned
///   artifact and the pinned backbone revision contains none.
///
/// Deliberately independent of shard placement: this pass pins the complete
/// tensor namespace, dtype, and shape, while the public inventory fixture pins
/// placement and header digests.
enum Gemma4A4BCheckpointValidation {
    struct ExpectedTensorMetadata: Equatable {
        let dtype: String
        let shape: [Int]
    }

    /// Frozen geometry of the pinned `gemma4_text` tower. Kept as literals
    /// rather than read from the config under validation: a validator that
    /// derives its expectations from the artifact it is checking cannot detect
    /// a changed artifact. (`MLXFastConstants` mirrors these; they are NOT
    /// referenced from there for the same reason the Qwen and Laguna
    /// validators keep their own copies -- the constants block is what a
    /// repin edits, and this block is what catches a repin nobody meant.)
    enum PinnedGeometry {
        static let vocabSize = 262_144
        static let hiddenSize = 2_816
        /// The DENSE per-layer MLP width. Every layer carries both this dense
        /// MLP and a 128-expert MoE block at `moeIntermediateSize`.
        static let intermediateSize = 2_112
        static let moeIntermediateSize = 704
        static let expertCount = 128
        static let layerCount = 30
        /// The tower repeats every 6 layers and the LAST layer of each group
        /// (index % 6 == 5, i.e. 5/11/17/23/29) is full attention; the other
        /// 25 are sliding-window attention.
        static let fullAttentionInterval = 6
        /// Query heads, uniform across both layer types.
        static let attentionHeads = 16
        static let keyValueHeads = 8
        static let globalKeyValueHeads = 2
        static let headDim = 256
        static let globalHeadDim = 512
        static let quantizationGroupSize = 64
        static let quantizationBits = 4
        /// Width of the 120 promoted tensors.
        static let quantizationOverrideBits = 8
        static let quantizationMode = "affine"
        /// Per-layer projection families promoted to 8 bits, on every layer.
        static let quantizationOverrideFamilies = [
            "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "router.proj",
        ]

        static func isFullAttention(layer index: Int) -> Bool {
            index % fullAttentionInterval == fullAttentionInterval - 1
        }
    }

    /// Total tensors in the transformed text-tower artifact: 4 top level plus
    /// 25 sliding layers of 45 and 5 full-attention layers of 42.
    ///
    /// The two per-layer counts differ by exactly one quantized triple:
    /// `attention_k_eq_v` is true, so a full-attention layer ships NO
    /// `self_attn.v_proj.*` and its values reuse the keys.
    static let expectedTensorCount = 1_339
    static let expectedSlidingLayerTensorCount = 45
    static let expectedFullAttentionLayerTensorCount = 42
    static let expectedTopLevelTensorCount = 4
    /// 4 families x 30 layers.
    static let expectedQuantizationOverrideCount = 120

    static let textTowerPrefix = "language_model."

    private static let modelPrefix = "language_model.model"
    private static let layerPrefix = "language_model.model.layers."

    /// Per-layer norms, all `[hidden_size]`. Gemma 4 carries seven, including
    /// the two `_1`/`_2` post-feedforward variants and the `_2`
    /// pre-feedforward one the Gemma 3 layout does not have.
    private static let layerNormSuffixes = [
        "input_layernorm", "post_attention_layernorm",
        "pre_feedforward_layernorm", "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm", "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
    ]

    /// Parses the checkpoint's quantization block(s).
    ///
    /// The pinned artifact publishes the SAME spec twice, as `quantization`
    /// and `quantization_config`; `SwiftTransform.makeRuntimeConfigData`
    /// accepts either form and requires them to agree when both are present,
    /// so this validator applies exactly that policy rather than a stricter
    /// one -- a transform that emits a config must not reject the checkpoint
    /// that config came from.
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

            // Every non-scalar key is a per-tensor override. Rejecting an
            // unknown scalar or a malformed override object matters more here
            // than on any previous target: this is the block whose contents
            // decide the packed width of 120 tensors, and a key this parse
            // skipped is a width the runtime would get wrong.
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

    /// The override table is pinned by CONSTRUCTION, not by count: a count
    /// check passes on 120 overrides naming the wrong paths.
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

    /// Exact metadata contract of the transformed text tower.
    ///
    /// DTYPES are the MLX affine-conversion convention rather than a reading
    /// of the pinned headers: packed codes `U32`, their scale/bias companions
    /// and every unquantized parameter `BF16` (the checkpoint's own `dtype` is
    /// `bfloat16`). Shapes and names ARE read off the pinned artifact. If the
    /// box session finds a different dtype on one of the small unquantized
    /// tensors, that is a repin of this table, not a relaxation of the check.
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
        /// One affine-quantized projection: packed U32 codes plus the BF16
        /// scale and bias companions the affine scheme requires. `leading` is
        /// everything before the contracted axis, so the stacked SwitchGLU
        /// expert tensors pass their experts axis through it.
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

        // Top level: the tied embedding and the final norm. There is NO
        // `lm_head.*` -- `tie_word_embeddings` is true on this checkpoint.
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
            // Global layers run wider heads and fewer of them.
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
            // Q/K norms are per-HEAD, so they follow the layer's own head
            // width: [256] sliding, [512] global.
            add("\(prefix).self_attn.q_norm.weight", .bf16, [headDim])
            add("\(prefix).self_attn.k_norm.weight", .bf16, [headDim])

            addAffine(
                "\(prefix).self_attn.q_proj",
                leading: [queryWidth], inFeatures: hidden, bits: fallbackBits)
            addAffine(
                "\(prefix).self_attn.k_proj",
                leading: [kvWidth], inFeatures: hidden, bits: fallbackBits)
            if !isFull {
                // Sliding layers only: `attention_k_eq_v` makes the five
                // full-attention layers reuse keys as values.
                addAffine(
                    "\(prefix).self_attn.v_proj",
                    leading: [kvWidth], inFeatures: hidden,
                    bits: fallbackBits)
            }
            addAffine(
                "\(prefix).self_attn.o_proj",
                leading: [hidden], inFeatures: queryWidth, bits: fallbackBits)

            // The four promoted families, at 8 bits.
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

            // Experts are SwitchGLU-STACKED: a leading experts axis, never
            // split per expert. They stay at the 4-bit fallback.
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
        // Affine quantization legitimately ships `.biases`, so unlike the
        // Laguna NVFP4 contract that suffix is NOT forbidden here.
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
            // Rank is >= 2 rather than == 2: the stacked SwitchGLU expert
            // tensors are rank 3 (experts x out x packed-in).
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

            // THE PER-PATH LOOKUP, and it is the point of this whole file. A
            // single global width would accept a 4-bit-packed tensor the
            // config declares at 8 bits and vice versa -- right names, right
            // leading dimensions, wrong numerics.
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
        // The public checkpoint also carries the `vision_tower.*` and
        // `embed_vision.*` namespaces (358 tensors), which the transform never
        // selects, so -- unlike the Laguna validator -- the index and header
        // name sets are supersets rather than equal. Constrain them where it
        // matters: every selected name must be indexed exactly once and
        // present in exactly one header.
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
