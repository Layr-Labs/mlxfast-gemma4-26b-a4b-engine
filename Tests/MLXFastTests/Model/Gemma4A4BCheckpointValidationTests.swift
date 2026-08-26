import Foundation
import MLXFastCore
@testable import MLXFastTransform
import Testing

// Transform-side conformance for the Gemma 4 26B A4B family: family detection
// against the legacy dense Gemma layout it collides with, the mixed-precision
// quantization contract, and the exact 1,339-tensor inventory.
//
// EVERYTHING HERE IS SYNTHETIC and derived from the validator's own pinned
// geometry rather than from a checkpoint capture. There is no
// `fixtures/gemma4_26b_a4b_*` yet -- those are box-generated
// (docs/gemma4-port-notes.md section 6.1) -- so these tests pin the validator's
// INTERNAL consistency and its rejection behaviour, not the artifact. The
// artifact-facing half (does the real checkpoint's header set equal this
// inventory?) is the box's, and it is the reason the dtype column below is
// still a convention rather than a reading.

private typealias Geometry = Gemma4A4BCheckpointValidation.PinnedGeometry

// MARK: - Family detection

/// The A4B target and the legacy dense 31B share BOTH the top-level
/// `model_type` "gemma4" and a nested `text_config`, so nothing about the
/// model-type string can separate them. This pins the architecture
/// discriminator and, more importantly, its ORDER: the legacy branch is an
/// unconditional `return .gemma4` for any config carrying a `text_config`, so a
/// discriminator placed after it would never run.
@Test
func gemma4A4BIsDetectedBeforeTheLegacyDenseGemmaFallthrough() throws {
    let a4b: [String: Any] = [
        "model_type": "gemma4",
        "text_config": [
            "model_type": "gemma4_text",
            "enable_moe_block": true,
            "num_experts": 128,
        ],
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: a4b) == .gemma4A4B)

    // The legacy dense family declares neither key and still routes to .gemma4.
    let dense: [String: Any] = [
        "model_type": "gemma4",
        "text_config": ["model_type": "gemma4_text"],
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: dense) == .gemma4)
}

/// Both halves of the discriminator are required. Either alone would misroute:
/// the flag alone catches any future MoE Gemma whose inventory this validator
/// does not describe, and the count alone catches a config that declares the
/// field with the block switched off.
@Test
func gemma4A4BDiscriminatorNeedsBothTheMoEFlagAndTheExpertCount() throws {
    let flagOnly: [String: Any] = [
        "text_config": ["enable_moe_block": true, "num_experts": 64]
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: flagOnly) == .gemma4)

    let countOnly: [String: Any] = [
        "text_config": ["enable_moe_block": false, "num_experts": 128]
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: countOnly) == .gemma4)

    // Wrong JSON kind on either key is not a near-miss to be coerced: a string
    // "true" or "128" leaves the config on the legacy path rather than
    // selecting a family whose validator would then reject it with a confusing
    // inventory error.
    let stringly: [String: Any] = [
        "text_config": ["enable_moe_block": "true", "num_experts": "128"]
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: stringly) == .gemma4)
}

/// Qwen still wins over both Gemma families: its `text_config` could otherwise
/// be probed for MoE keys it does not carry, but the model-type prefix is
/// checked first and this pins that ordering survives the new branch.
@Test
func qwenStillRoutesAheadOfEitherGemmaFamily() throws {
    #expect(
        try SwiftTransform.detectModelFamily(
            sourceConfigRoot: ["text_config": ["model_type": "qwen3_5_text"]]
        ) == .qwen35
    )
}

// MARK: - The mixed-precision quantization block

/// Build the pinned `quantization` block: three fallback scalars plus the 120
/// per-tensor 8-bit promotions.
private func pinnedQuantizationBlock() -> [String: Any] {
    var block: [String: Any] = [
        "group_size": Geometry.quantizationGroupSize,
        "bits": Geometry.quantizationBits,
        "mode": Geometry.quantizationMode,
    ]
    for layer in 0..<Geometry.layerCount {
        for family in Geometry.quantizationOverrideFamilies {
            block["language_model.model.layers.\(layer).\(family)"] = [
                "group_size": Geometry.quantizationGroupSize,
                "bits": Geometry.quantizationOverrideBits,
            ]
        }
    }
    return block
}

private func pinnedSourceConfig(
    textConfig: [String: Any]? = nil,
    quantization: [String: Any]? = nil,
    quantizationConfig: [String: Any]? = nil
) -> [String: Any] {
    let block = quantization ?? pinnedQuantizationBlock()
    return [
        "model_type": "gemma4",
        "text_config": textConfig ?? [
            "model_type": "gemma4_text",
            "enable_moe_block": true,
            "num_experts": Geometry.expertCount,
            "num_hidden_layers": Geometry.layerCount,
        ],
        "quantization": block,
        "quantization_config": quantizationConfig ?? block,
    ]
}

@Test
func quantizationSpecParsesTheFallbackAndAllOneHundredTwentyOverrides() throws {
    let spec = try Gemma4A4BCheckpointValidation.quantizationSpec(
        fromConfigRoot: pinnedSourceConfig())
    #expect(spec.groupSize == 64)
    #expect(spec.bits == 4)
    #expect(spec.mode == "affine")
    #expect(
        spec.overrides.count
            == Gemma4A4BCheckpointValidation.expectedQuantizationOverrideCount)
    // Resolution is PER PATH, which is the whole reason the table is data.
    #expect(
        spec.spec(forPath: "language_model.model.layers.0.mlp.gate_proj").bits == 8)
    #expect(
        spec.spec(forPath: "language_model.model.layers.29.router.proj").bits == 8)
    #expect(
        spec.spec(forPath: "language_model.model.layers.0.self_attn.q_proj").bits
            == 4)
    #expect(spec.spec(forPath: "language_model.model.embed_tokens").bits == 4)
}

@Test
func quantizationSpecRequiresTheDuplicateBlocksToAgree() throws {
    var divergent = pinnedQuantizationBlock()
    divergent["language_model.model.layers.7.mlp.up_proj"] = [
        "group_size": 64, "bits": 4,
    ]
    #expect(throws: (any Error).self) {
        _ = try Gemma4A4BCheckpointValidation.quantizationSpec(
            fromConfigRoot: pinnedSourceConfig(quantizationConfig: divergent))
    }
    // One block alone is accepted -- the transform must not demand a duplicate
    // the checkpoint may legitimately not publish.
    var single = pinnedSourceConfig()
    single.removeValue(forKey: "quantization_config")
    #expect(throws: Never.self) {
        _ = try Gemma4A4BCheckpointValidation.quantizationSpec(
            fromConfigRoot: single)
    }
}

/// The override table is pinned by CONSTRUCTION. A count check alone would pass
/// on 120 overrides naming the wrong paths, so each of these mutations keeps
/// the count valid (or valid-looking) and changes what the table says.
@Test
func quantizationSpecRejectsEveryShapeOfOverrideTableDrift() throws {
    func rejected(_ mutate: (inout [String: Any]) -> Void) -> Bool {
        var block = pinnedQuantizationBlock()
        mutate(&block)
        do {
            _ = try Gemma4A4BCheckpointValidation.quantizationSpec(
                fromConfigRoot: pinnedSourceConfig(
                    quantization: block, quantizationConfig: block))
            return false
        } catch {
            return true
        }
    }

    // A promotion silently dropped -- the failure that produces right names,
    // right shapes and wrong numerics.
    #expect(rejected { $0.removeValue(forKey: "language_model.model.layers.3.mlp.up_proj") })
    // Swapped for a path that is not promoted on this checkpoint: count intact.
    #expect(
        rejected {
            $0.removeValue(forKey: "language_model.model.layers.3.mlp.up_proj")
            $0["language_model.model.layers.3.self_attn.q_proj"] = [
                "group_size": 64, "bits": 8,
            ]
        })
    // Right path, wrong width.
    #expect(
        rejected {
            $0["language_model.model.layers.3.mlp.up_proj"] = [
                "group_size": 64, "bits": 4,
            ]
        })
    // Right path, wrong group size.
    #expect(
        rejected {
            $0["language_model.model.layers.3.mlp.up_proj"] = [
                "group_size": 32, "bits": 8,
            ]
        })
    // An override object carrying an unknown key is not a tolerated extra.
    #expect(
        rejected {
            $0["language_model.model.layers.3.mlp.up_proj"] = [
                "group_size": 64, "bits": 8, "mode": "affine",
            ]
        })
    // A non-object value under a tensor path.
    #expect(rejected { $0["language_model.model.layers.3.mlp.up_proj"] = 8 })
    // A changed fallback: this checkpoint is affine 4-bit group-64.
    #expect(rejected { $0["bits"] = 8 })
    #expect(rejected { $0["mode"] = "nvfp4" })
}

// MARK: - The emitted runtime config

/// THE REGRESSION THIS EXISTS FOR. The `.qwen35` branch of
/// `makeRuntimeConfigData` REBUILDS the emitted `quantization` from a parsed
/// `{group_size, bits, mode}` triple, which is lossless only because the Qwen
/// block has exactly three keys. Copying that shape here would emit a config
/// declaring uniform 4-bit for the 120 tensors the shards were written at 8 --
/// right names, right shapes, wrong numerics, and nothing downstream notices.
@Test
func emittedRuntimeConfigPreservesAllOneHundredTwentyOverridesVerbatim() throws {
    let data = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: pinnedSourceConfig(), family: .gemma4A4B)
    let emitted = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any])

    // The text tower is flattened to the top level.
    #expect(emitted["text_config"] == nil)
    #expect(emitted["enable_moe_block"] as? Bool == true)
    #expect(emitted["num_experts"] as? Int == Geometry.expertCount)
    // The duplicate is removed once the two were verified to agree.
    #expect(emitted["quantization_config"] == nil)
    // And nothing else is invented.
    #expect(emitted["vision_config"] == nil)

    let quantization = try #require(emitted["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == 64)
    #expect(quantization["bits"] as? Int == 4)
    #expect(quantization["mode"] as? String == "affine")
    let overrideKeys = quantization.keys.filter {
        !["group_size", "bits", "mode"].contains($0)
    }
    #expect(
        overrideKeys.count
            == Gemma4A4BCheckpointValidation.expectedQuantizationOverrideCount)
    for layer in 0..<Geometry.layerCount {
        for family in Geometry.quantizationOverrideFamilies {
            let entry = try #require(
                quantization["language_model.model.layers.\(layer).\(family)"]
                    as? [String: Any],
                Comment(rawValue: "layer \(layer) \(family)")
            )
            #expect(entry["bits"] as? Int == 8)
            #expect(entry["group_size"] as? Int == 64)
        }
    }
}

/// The legacy `.gemma4` family emits the projection and tied-head sidecars;
/// `.gemma4A4B` must not, and the transform's sidecar switch is where that is
/// decided. This pins the family's membership of the emit-nothing branch by
/// exercising the one observable consequence available without a checkpoint:
/// the runtime config is the flattened tower and the quantization block, and
/// nothing else.
@Test
func emittedRuntimeConfigCarriesOnlyTheTowerAndTheQuantizationBlock() throws {
    let data = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: pinnedSourceConfig(), family: .gemma4A4B)
    let emitted = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any])
    let textKeys: Set<String> = [
        "model_type", "enable_moe_block", "num_experts", "num_hidden_layers",
    ]
    #expect(Set(emitted.keys) == textKeys.union(["quantization"]))
}

@Test
func textTowerSelectionKeepsOnlyTheLanguageModelPrefix() {
    #expect(
        SwiftTransform.isSelectedTextTowerKey(
            "language_model.model.layers.0.mlp.gate_proj.weight",
            family: .gemma4A4B))
    for dropped in [
        "vision_tower.blocks.0.attn.qkv.weight",
        "embed_vision.weight",
        "multi_modal_projector.linear.weight",
    ] {
        #expect(!SwiftTransform.isSelectedTextTowerKey(dropped, family: .gemma4A4B))
    }
}

// MARK: - The exact 1,339-tensor inventory

@Test
func expectedInventoryMatchesThePinnedTensorCounts() {
    let inventory = Gemma4A4BCheckpointValidation.expectedTensorInventory()
    #expect(inventory.count == 1_339)
    #expect(
        inventory.count == Gemma4A4BCheckpointValidation.expectedTensorCount)

    // 4 top level: the tied embedding's triple plus the final norm.
    let topLevel = inventory.keys.filter { !$0.contains(".layers.") }
    #expect(
        topLevel.count
            == Gemma4A4BCheckpointValidation.expectedTopLevelTensorCount)
    #expect(
        Set(topLevel) == [
            "language_model.model.embed_tokens.weight",
            "language_model.model.embed_tokens.scales",
            "language_model.model.embed_tokens.biases",
            "language_model.model.norm.weight",
        ])

    // 25 sliding layers of 45 and 5 global layers of 42; the difference is
    // exactly one quantized triple, the v_proj a global layer does not ship.
    var slidingLayers = 0
    var globalLayers = 0
    for layer in 0..<Geometry.layerCount {
        let prefix = "language_model.model.layers.\(layer)."
        let count = inventory.keys.filter { $0.hasPrefix(prefix) }.count
        if Geometry.isFullAttention(layer: layer) {
            globalLayers += 1
            #expect(
                count
                    == Gemma4A4BCheckpointValidation
                    .expectedFullAttentionLayerTensorCount,
                "layer \(layer)")
        } else {
            slidingLayers += 1
            #expect(
                count
                    == Gemma4A4BCheckpointValidation
                    .expectedSlidingLayerTensorCount,
                "layer \(layer)")
        }
    }
    #expect(globalLayers == 5)
    #expect(slidingLayers == 25)
    #expect(
        Gemma4A4BCheckpointValidation.expectedTopLevelTensorCount
            + slidingLayers
            * Gemma4A4BCheckpointValidation.expectedSlidingLayerTensorCount
            + globalLayers
            * Gemma4A4BCheckpointValidation
            .expectedFullAttentionLayerTensorCount == 1_339)
}

/// `attention_k_eq_v` is a TENSOR-INVENTORY fact as much as a numerics one, and
/// `tie_word_embeddings` removes a whole family. Both are asserted as
/// presence/absence rather than as tolerated extras.
@Test
func expectedInventoryEncodesTiedEmbeddingsAndTheMissingGlobalValueProjection() {
    let inventory = Gemma4A4BCheckpointValidation.expectedTensorInventory()
    #expect(!inventory.keys.contains { $0.contains("lm_head") })
    for layer in 0..<Geometry.layerCount {
        let vProj = "language_model.model.layers.\(layer).self_attn.v_proj.weight"
        #expect(
            (inventory[vProj] == nil) == Geometry.isFullAttention(layer: layer),
            "layer \(layer)")
    }
}

@Test
func expectedInventoryPacksEachPathAtItsOwnDeclaredWidth() {
    let inventory = Gemma4A4BCheckpointValidation.expectedTensorInventory()
    func shape(_ name: String) -> [Int] { inventory[name]?.shape ?? [] }

    // 4-bit fallback: hidden 2816 packs to 2816 * 4 / 32 = 352 U32 columns,
    // with 2816 / 64 = 44 group columns.
    #expect(
        shape("language_model.model.embed_tokens.weight") == [262_144, 352])
    #expect(
        shape("language_model.model.embed_tokens.scales") == [262_144, 44])
    // 8-bit promotion: the SAME input width packs to 2816 * 8 / 32 = 704, and
    // the group column count is unchanged. A dropped promotion shows up here as
    // a factor-of-two shape mismatch rather than as silent numerics.
    #expect(
        shape("language_model.model.layers.0.mlp.gate_proj.weight")
            == [2_112, 704])
    #expect(
        shape("language_model.model.layers.0.mlp.gate_proj.scales")
            == [2_112, 44])
    #expect(
        shape("language_model.model.layers.0.mlp.down_proj.weight")
            == [2_816, 528])
    #expect(
        shape("language_model.model.layers.0.router.proj.weight") == [128, 704])

    // Global layers run wider heads and fewer of them: q is 16 * 512, k is
    // 2 * 512, and the per-head norms follow the layer's own head width.
    #expect(
        shape("language_model.model.layers.5.self_attn.q_proj.weight")
            == [8_192, 352])
    #expect(
        shape("language_model.model.layers.5.self_attn.k_proj.weight")
            == [1_024, 352])
    #expect(shape("language_model.model.layers.5.self_attn.q_norm.weight") == [512])
    #expect(shape("language_model.model.layers.0.self_attn.q_norm.weight") == [256])
    #expect(
        shape("language_model.model.layers.0.self_attn.k_proj.weight")
            == [2_048, 352])

    // Experts are SwitchGLU-STACKED: a leading experts axis, never split.
    #expect(
        shape("language_model.model.layers.0.experts.switch_glu.gate_proj.weight")
            == [128, 704, 352])
    #expect(
        shape("language_model.model.layers.0.experts.switch_glu.down_proj.weight")
            == [128, 2_816, 88])

    // The unquantized per-layer scalars and norms.
    #expect(shape("language_model.model.layers.0.layer_scalar") == [1])
    #expect(shape("language_model.model.layers.0.router.scale") == [2_816])
    #expect(shape("language_model.model.layers.0.router.per_expert_scale") == [128])
    #expect(
        shape("language_model.model.layers.0.post_feedforward_layernorm_2.weight")
            == [2_816])
}

// MARK: - validateSelectedTensors against a synthetic exact checkpoint

private func gemmaValidationMetadata(
    mutate: (inout [String: Gemma4A4BCheckpointValidation.ExpectedTensorMetadata])
        -> Void = { _ in }
) -> (
    selectedKeys: Set<String>, index: CheckpointIndex,
    headers: [String: SafetensorsHeader]
) {
    var inventory = Gemma4A4BCheckpointValidation.expectedTensorInventory()
    mutate(&inventory)

    let shardName = "model-00001-of-00001.safetensors"
    var weightMap: [String: String] = [:]
    var tensors: [String: SafetensorInfo] = [:]
    for (name, metadata) in inventory {
        weightMap[name] = shardName
        tensors[name] = SafetensorInfo(
            name: name,
            dtype: metadata.dtype,
            shape: metadata.shape,
            dataStart: 0,
            dataEnd: 1
        )
    }
    return (
        Set(inventory.keys),
        CheckpointIndex(raw: ["weight_map": weightMap], weightMap: weightMap),
        [
            shardName: SafetensorsHeader(
                headerLength: 8, metadata: ["format": "pt"], tensors: tensors)
        ]
    )
}

@Test
func transformAcceptsTheExactPublicTextTower() throws {
    let metadata = gemmaValidationMetadata()
    #expect(metadata.selectedKeys.count == 1_339)
    try Gemma4A4BCheckpointValidation.validateSelectedTensors(
        selectedKeys: metadata.selectedKeys,
        index: metadata.index,
        headers: metadata.headers,
        quantization: try Gemma4A4BCheckpointValidation.quantizationSpec(
            fromConfigRoot: pinnedSourceConfig())
    )
}

@Test
func transformRejectsInventoryAndPackingDrift() throws {
    let quantization = try Gemma4A4BCheckpointValidation.quantizationSpec(
        fromConfigRoot: pinnedSourceConfig())

    func rejected(
        _ mutate: (
            inout [String: Gemma4A4BCheckpointValidation.ExpectedTensorMetadata]
        ) -> Void
    ) -> Bool {
        let metadata = gemmaValidationMetadata(mutate: mutate)
        do {
            try Gemma4A4BCheckpointValidation.validateSelectedTensors(
                selectedKeys: metadata.selectedKeys,
                index: metadata.index,
                headers: metadata.headers,
                quantization: quantization
            )
            return false
        } catch {
            return true
        }
    }

    let quantized = Gemma4A4BCheckpointValidation.ExpectedTensorMetadata.self
    // A promotion the shards honoured but the config dropped: the 8-bit tensor
    // stored at 4-bit width. This is the packing check earning its place.
    #expect(
        rejected {
            $0["language_model.model.layers.0.mlp.gate_proj.weight"] = quantized
                .init(dtype: "U32", shape: [2_112, 352])
        })
    // A global layer that ships a v_proj after all.
    #expect(
        rejected {
            for component in ["weight", "scales", "biases"] {
                $0["language_model.model.layers.5.self_attn.v_proj.\(component)"] =
                    quantized.init(
                        dtype: component == "weight" ? "U32" : "BF16",
                        shape: component == "weight"
                            ? [1_024, 352] : [1_024, 44])
            }
        })
    // A sliding layer that drops the one it must ship.
    #expect(
        rejected {
            $0.removeValue(
                forKey: "language_model.model.layers.0.self_attn.v_proj.weight")
        })
    // An untied output head.
    #expect(
        rejected {
            $0["language_model.lm_head.weight"] = quantized.init(
                dtype: "U32", shape: [262_144, 352])
        })
    // An MTP tensor smuggled into the backbone.
    #expect(
        rejected {
            $0["language_model.model.mtp.layers.0.norm.weight"] = quantized
                .init(dtype: "BF16", shape: [2_816])
        })
    // Compressed-tensors and FP8 KV-scale aliases.
    #expect(
        rejected {
            $0["language_model.model.layers.0.self_attn.k_proj.k_scale"] =
                quantized.init(dtype: "BF16", shape: [1])
        })
    // A quantized projection missing its bias companion (affine needs both).
    #expect(
        rejected {
            $0.removeValue(
                forKey: "language_model.model.layers.0.self_attn.q_proj.biases")
        })
    // Wrong dtype on a packed weight.
    #expect(
        rejected {
            $0["language_model.model.layers.0.self_attn.q_proj.weight"] =
                quantized.init(dtype: "BF16", shape: [4_096, 352])
        })
    // Wrong shape on an unquantized norm.
    #expect(
        rejected {
            $0["language_model.model.layers.0.self_attn.q_norm.weight"] =
                quantized.init(dtype: "BF16", shape: [512])
        })
    // A stacked expert tensor flattened into rank 2.
    #expect(
        rejected {
            $0["language_model.model.layers.0.experts.switch_glu.gate_proj.weight"] =
                quantized.init(dtype: "U32", shape: [90_112, 352])
        })
}
