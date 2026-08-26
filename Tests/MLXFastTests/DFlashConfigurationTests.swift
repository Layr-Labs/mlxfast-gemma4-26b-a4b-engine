import Foundation
import MLXLMCommon
import MLXSpeculative
import Testing

// DFlash drafter CONFIG decoding — ported 2026-08-25 from
// Layr-Labs/mlx-swift-lm `origin/dflash-framework-updates`
// (Tests/MLXLMTests/DFlashConfigurationTests.swift @ d41c3003) into this
// engine's test target.
//
// Adaptations, all mechanical:
//   * `Bundle.module` fixture loading → a `#filePath`-relative read, the
//     convention this test target already uses (`HarnessHashRootSetTests`
//     .repoRoot) so no SwiftPM test-resource declaration is needed.
//   * The upstream GPT-OSS-120B and Qwen-3.5-27B fixtures are not vendored
//     here (they describe drafters for target families this engine does not
//     carry); their two decode tests are dropped and the directory-load pair
//     is re-pointed at the Gemma 4 fixture that IS vendored.
//
// This is what makes the loader real: a z-lab DFlash `config.json` is
// exactly this shape, and it is NOT decodable by `Gemma4AssistantConfiguration`
// — which is why the #38 alias loader could never have bound one.
@Suite("DFlashConfiguration decoding")
struct DFlashConfigurationTests {

    /// `Tests/MLXFastTests/Resources/`, resolved from THIS source file so the
    /// location is known independent of the process CWD.
    private static let resourcesDirectory: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Resources", isDirectory: true)

    private func loadFixture(named name: String) throws -> Data {
        try Data(
            contentsOf: Self.resourcesDirectory
                .appendingPathComponent("\(name).json"))
    }

    private func validJSON(
        layerTypes: String = #""full_attention", "full_attention""#,
        slidingWindow: String = "",
        blockSize: Int = 4,
        numTargetLayers: Int = 3,
        targetLayerIds: String = "0, 1",
        maskTokenId: Int = 4
    ) -> String {
        """
        {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 128,
            "block_size": \(blockSize),
            "num_target_layers": \(numTargetLayers),
            "layer_types": [\(layerTypes)],
            \(slidingWindow)
            "tie_word_embeddings": true,
            "dflash_config": {
                "target_layer_ids": [\(targetLayerIds)],
                "mask_token_id": \(maskTokenId)
            }
        }
        """
    }

    @Test func decodesSupportedConfig() throws {
        let config = try JSONDecoder.json5().decode(
            DFlashConfiguration.self, from: Data(validJSON().utf8))
        #expect(config.architectures == ["DFlashDraftModel"])
        #expect(config.modelType == "qwen3")
        #expect(config.hiddenSize == 16)
        #expect(config.layerTypes == [.fullAttention, .fullAttention])
        #expect(config.numTargetLayers == 3)
        #expect(config.targetLayerIds == [0, 1])
        #expect(config.maskTokenId == 4)
        #expect(config.targetHiddenSize == 32)
        #expect(config.ignoredConfigKeys.isEmpty)
    }

    @Test func decodesSlidingAttentionWhenWindowPresent() throws {
        let config = try JSONDecoder.json5().decode(
            DFlashConfiguration.self,
            from: Data(
                validJSON(
                    layerTypes: #""sliding_attention", "full_attention""#,
                    slidingWindow: #""sliding_window": 64,"#
                ).utf8))
        #expect(config.layerTypes == [.slidingAttention, .fullAttention])
        #expect(config.slidingWindow == 64)
    }

    @Test func recommendsCheckpointBlockForSlidingDrafts() throws {
        let full = try JSONDecoder.json5().decode(
            DFlashConfiguration.self,
            from: Data(validJSON(blockSize: 16).utf8))
        let sliding = try JSONDecoder.json5().decode(
            DFlashConfiguration.self,
            from: Data(
                validJSON(
                    layerTypes: #""sliding_attention", "full_attention""#,
                    slidingWindow: #""sliding_window": 64,"#,
                    blockSize: 16
                ).utf8))

        #expect(full.recommendedBlockSize == 16)
        #expect(sliding.recommendedBlockSize == 16)
    }

    /// The REAL z-lab `gemma-4-26B-A4B-it-DFlash` config shape. Two facts the
    /// arm depends on are pinned here rather than assumed:
    ///
    ///   * `num_target_layers` is 30 and every tap id is `< 30`, so
    ///     `DFlashDraftModel.bind`'s `numTargetLayers == target.dFlashLayerCount`
    ///     check passes against a 30-layer A4B target and
    ///     `validateTargetLayerIds` finds every tap in range.
    ///   * `head_dim` is 128 — the DRAFTER's own attention head dim, which is
    ///     never compared against the target's. It sizes the drafter's q/k/v
    ///     projections, its q/k RMSNorms, and its RoPE
    ///     (`initializeRope(dims: config.headDim, …)`), all of which live
    ///     entirely inside the drafter tower. `validateCompatibility` checks
    ///     hidden size, vocab size and layer count only.
    @Test func decodesRealGemma426BA4BDFlashShape() throws {
        let json = """
        {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 2816,
            "num_hidden_layers": 5,
            "intermediate_size": 5632,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 262144,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 262144,
            "block_size": 16,
            "num_target_layers": 30,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention"
            ],
            "sliding_window": 2048,
            "tie_word_embeddings": false,
            "dflash_config": {
                "target_layer_ids": [1, 6, 11, 17, 22, 27],
                "mask_token_id": 4
            }
        }
        """
        let config = try JSONDecoder.json5().decode(
            DFlashConfiguration.self, from: Data(json.utf8))
        #expect(config.blockSize == 16)
        #expect(config.recommendedBlockSize == 16)
        #expect(config.headDim == 128)
        #expect(config.numTargetLayers == 30)
        #expect(config.targetLayerIds == [1, 6, 11, 17, 22, 27])
        #expect(config.targetLayerIds.allSatisfy { $0 < config.numTargetLayers })
        // Six taps at hidden 2816: the width the drafter's `fc` projection
        // expects, and what `DFlashTargetForward` concatenates to.
        #expect(config.targetHiddenSize == 16896)
    }

    @Test func rejectsSlidingAttentionWithoutWindow() throws {
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                DFlashConfiguration.self,
                from: Data(
                    validJSON(layerTypes: #""sliding_attention", "full_attention""#).utf8))
        }
    }

    @Test func rejectsInvalidLayerTypeCount() throws {
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                DFlashConfiguration.self,
                from: Data(validJSON(layerTypes: #""full_attention""#).utf8))
        }
    }

    @Test func rejectsEmptyTargetLayerIds() throws {
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                DFlashConfiguration.self,
                from: Data(validJSON(targetLayerIds: "").utf8))
        }
    }

    @Test func rejectsDuplicateTargetLayerIds() throws {
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                DFlashConfiguration.self,
                from: Data(validJSON(targetLayerIds: "0, 0").utf8))
        }
    }

    @Test func rejectsOutOfRangeTargetLayerIds() throws {
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                DFlashConfiguration.self,
                from: Data(validJSON(numTargetLayers: 3, targetLayerIds: "0, 3").utf8))
        }
    }

    @Test func rejectsBlockSizeBelowTwo() throws {
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                DFlashConfiguration.self,
                from: Data(validJSON(blockSize: 1).utf8))
        }
    }

    @Test func rejectsMaskTokenOutsideVocabulary() throws {
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                DFlashConfiguration.self,
                from: Data(validJSON(maskTokenId: 32).utf8))
        }
    }

    @Test func recordsUnknownTopLevelAndDFlashConfigKeys() throws {
        let json = """
        {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "block_size": 4,
            "num_target_layers": 3,
            "layer_types": ["full_attention", "full_attention"],
            "unknown_top_level": true,
            "dflash_config": {
                "target_layer_ids": [0, 1],
                "mask_token_id": 4,
                "unknown_nested": "kept for diagnostics"
            }
        }
        """
        let config = try JSONDecoder.json5().decode(
            DFlashConfiguration.self, from: Data(json.utf8))
        #expect(config.ignoredConfigKeys == ["unknown_top_level"])
        #expect(config.dflashConfig.ignoredConfigKeys == ["unknown_nested"])
    }

    @Test func decodesGemma4GatedSchemaFixture() throws {
        let config = try JSONDecoder.json5().decode(
            DFlashConfiguration.self,
            from: try loadFixture(named: "dflash-gemma4-gated-schema-config"))
        #expect(config.blockSize == 16)
        #expect(config.numTargetLayers == 35)
        #expect(config.targetLayerIds == [1, 9, 17, 25, 33])
        #expect(config.vocabularySize == 262144)
        #expect(config.ignoredConfigKeys.contains("fixture_note"))
    }

    @Test func loadConfigurationFromDirectory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configURL = directory.appendingPathComponent("config.json")
        try loadFixture(named: "dflash-gemma4-gated-schema-config").write(to: configURL)

        let config = try DFlashDraftModel.loadConfiguration(from: directory)

        #expect(config.targetLayerIds == [1, 9, 17, 25, 33])
    }

    @Test func loadRejectsDirectoryWithoutSafetensors() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configURL = directory.appendingPathComponent("config.json")
        try loadFixture(named: "dflash-gemma4-gated-schema-config").write(to: configURL)

        await #expect(throws: DFlashError.noSafetensorsFound(directory.path)) {
            _ = try await DFlashDraftModel.load(from: directory)
        }
    }
}
