// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
@testable import MLXLLM
import Testing

@Suite("Gemma4AssistantConfiguration decoding")
struct Gemma4AssistantConfigurationTests {

    private func tinyConfig(
        modelType: String = "gemma4_assistant",
        backbone: String = "64",
        hidden: String = "64",
        vocab: String = "64",
        layers: String = "2",
        fullPartialRotaryFactor: String = "1.0",
        quantization: String = ""
    ) -> Data {
        Data(
            """
            {
                "model_type": "\(modelType)",
                "backbone_hidden_size": \(backbone),
                "use_ordered_embeddings": false,
                "num_centroids": 8,
                "centroid_intermediate_top_k": 2,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": \(hidden),
                    "num_hidden_layers": \(layers),
                    "intermediate_size": 128,
                    "num_attention_heads": 2,
                    "head_dim": 32,
                    "global_head_dim": 32,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": \(layers),
                    "layer_types": ["sliding_attention", "full_attention"],
                    "sliding_window": 64,
                    "sliding_window_pattern": 2,
                    "tie_word_embeddings": true,
                    "vocab_size": \(vocab),
                    "vocab_size_per_layer_input": 0,
                    "hidden_size_per_layer_input": 0,
                    "rope_parameters": {
                        "full_attention": {
                            "rope_theta": 1000000.0,
                            "partial_rotary_factor": \(fullPartialRotaryFactor)
                        }
                    },
                    "rms_norm_eps": 1e-6,
                    "use_double_wide_mlp": false
                }
                \(quantization)
            }
            """.utf8)
    }

    private func loadError(for data: Data) async throws -> Gemma4MTPError? {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "gemma4-mtp-hostile-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try data.write(to: directory.appending(path: "config.json"))
        do {
            _ = try await Gemma4AssistantDraftModel.load(from: directory)
            return nil
        } catch let error as Gemma4MTPError {
            return error
        }
    }

    private func loadFixture(named name: String) throws -> Data {
        let url = Bundle.module.url(forResource: name, withExtension: "json")
        #expect(url != nil, "fixture \(name).json not found in test bundle")
        return try Data(contentsOf: url!)
    }

    @Test func decodesE4BDrafterConfig() throws {
        let data = try loadFixture(named: "gemma4-E4B-assistant-config")
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        #expect(config.backboneHiddenSize == 2560)
        #expect(config.useOrderedEmbeddings == true)
        #expect(config.numCentroids == 2048)
        #expect(config.centroidIntermediateTopK == 32)
        #expect(config.textConfig.tieWordEmbeddings == true)
        #expect(config.textConfig.numHiddenLayers == 4)
        #expect(config.textConfig.numKvSharedLayers == 4)
        #expect(config.textConfig.attentionKeqV == false)
    }

    @Test func decodes26BA4BDrafterConfig() throws {
        let data = try loadFixture(named: "gemma4-26B-A4B-assistant-config")
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        #expect(config.backboneHiddenSize == 2816)
        #expect(config.useOrderedEmbeddings == false)
        #expect(config.numCentroids == 2048)
        #expect(config.textConfig.tieWordEmbeddings == true)
        #expect(config.textConfig.numHiddenLayers == 4)
        #expect(config.textConfig.numKvSharedLayers == 4)
        #expect(config.textConfig.attentionKeqV == true)
        #expect(config.textConfig.numGlobalKeyValueHeads == 2)
    }

    @Test func validatesOfficialLikeFixturesAndCaptureGeometry() throws {
        for fixture in ["gemma4-E4B-assistant-config", "gemma4-26B-A4B-assistant-config"] {
            let document = try Gemma4AssistantConfigurationDocument.decode(
                loadFixture(named: fixture))
            let assistant = document.config
            let text = assistant.textConfig
            let globalHeads = text.numGlobalKeyValueHeads.map {
                "\"num_global_key_value_heads\": \($0),"
            } ?? ""
            let targetJSON = """
                {
                    "model_type": "gemma4_text",
                    "hidden_size": \(assistant.backboneHiddenSize),
                    "num_hidden_layers": 2,
                    "intermediate_size": 128,
                    "num_attention_heads": \(text.numAttentionHeads),
                    "head_dim": \(text.headDim),
                    "global_head_dim": \(text.globalHeadDim),
                    "num_key_value_heads": \(text.numKeyValueHeads),
                    \(globalHeads)
                    "num_kv_shared_layers": 0,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "sliding_window": \(text.slidingWindow),
                    "attention_k_eq_v": \(text.attentionKeqV),
                    "vocab_size": \(text.vocabSize),
                    "vocab_size_per_layer_input": 0,
                    "hidden_size_per_layer_input": 0,
                    "rms_norm_eps": 1e-6,
                    "use_double_wide_mlp": false
                }
                """
            let target = try JSONDecoder.json5().decode(
                Gemma4TextConfiguration.self, from: Data(targetJSON.utf8))
            try Gemma4MTPCompatibilityValidator.validate(
                drafter: assistant, target: target)
        }
    }

    @Test func hostileConfigsFailWithStableErrorsBeforeAllocation() async throws {
        let overCap = Data(
            repeating: 0x20,
            count: Gemma4AssistantConfigurationValidator.maximumConfigBytes + 1)
        let cases: [(String, Data, String)] = [
            ("huge vocab", tinyConfig(vocab: "524289"), "textConfig.vocabSize"),
            ("negative layers", tinyConfig(layers: "-1"), "textConfig.numHiddenLayers"),
            ("wrong model type", tinyConfig(modelType: "gemma4_text"), "modelType"),
            ("over-cap file", overCap, "config.json"),
            ("overflowing doubled backbone", tinyConfig(backbone: "9223372036854775807"), "backboneHiddenSize"),
            (
                "huge finite rotary factor",
                tinyConfig(fullPartialRotaryFactor: "1e30"),
                "textConfig.fullPartialRotaryFactor"),
            ("wrong scalar type", tinyConfig(hidden: "\"64\""), "config.json"),
            (
                "invalid quantization",
                tinyConfig(quantization: ", \"quantization\": {\"group_size\": -1, \"bits\": 4}"),
                "quantization.groupSize"),
        ]

        for (name, data, expectedField) in cases {
            let error = try await loadError(for: data)
            guard case .invalidConfiguration(let field, _)? = error else {
                Issue.record("\(name) did not throw Gemma4MTPError.invalidConfiguration")
                continue
            }
            #expect(field == expectedField, "\(name) reported \(field)")
        }
    }

    @Test func clampsNumKvSharedLayersWhenMissing() throws {
        // Synthetic config: text_config omits num_kv_shared_layers; the
        // post-init should set it to num_hidden_layers = 4.
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 256,
            "use_ordered_embeddings": true,
            "num_centroids": 16,
            "centroid_intermediate_top_k": 4,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "sliding_window": 64,
                "tie_word_embeddings": true,
                "vocab_size": 1024,
                "vocab_size_per_layer_input": 1024,
                "rms_norm_eps": 1e-6
            }
        }
        """
        let data = Data(json.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        #expect(config.textConfig.numKvSharedLayers == 4)
    }

    @Test func clampsNumKvSharedLayersWhenZero() throws {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 256,
            "use_ordered_embeddings": false,
            "num_centroids": 16,
            "centroid_intermediate_top_k": 4,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "sliding_window": 64,
                "tie_word_embeddings": true,
                "vocab_size": 1024,
                "vocab_size_per_layer_input": 1024,
                "rms_norm_eps": 1e-6
            }
        }
        """
        let data = Data(json.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        #expect(config.textConfig.numKvSharedLayers == 4)
    }
}
