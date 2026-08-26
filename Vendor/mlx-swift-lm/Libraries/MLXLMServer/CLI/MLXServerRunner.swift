// Copyright © 2026 Eigen Labs Inc.

import Foundation
import Hummingbird
import MLXLMCommon

public enum MLXServer {
    public static func run(configuration: MLXServerConfiguration) async throws {
        var primaryModelConfiguration = modelConfiguration(
            for: configuration.model,
            revision: configuration.revision
        )
        // Pin the tool-call format ONCE, at load time. An explicit
        // --tool-call-parser wins here; otherwise the field stays nil so
        // LLMModelFactory's load-time inference (model_type + config.json
        // secondary detection, e.g. Llama 3 via vocab_size/rope_scaling)
        // fills it. Either way the LOADED configuration is the single source
        // of truth the container engine's streaming tool parser reads;
        // per-request overrides are validated against it, never applied by
        // mutating shared model state (that raced across concurrent
        // requests — see MLXModelContainerEngine).
        if let parser = configuration.toolCallParser {
            primaryModelConfiguration.toolCallFormat = try ServerToolParser.resolve(
                requested: parser,
                modelType: configuration.modelType
            )
        }

        // v0.7.5 one-engine: the legacy continuous-batching adapter
        // (`MLXBatchedEngineServerEngine`) died with the v1 engine. The CLI
        // serves through the single-request container engine; batched
        // serving lives in the Darkbloom provider (ContinuousBatchingV2).
        let model = try await MLXServerModelLoader.load(
            configuration: primaryModelConfiguration
        )
        let engine: any MLXServerEngine = MLXModelContainerEngine(
            modelID: configuration.model,
            model: model,
            modelType: configuration.modelType,
            defaultToolCallParser: configuration.toolCallParser
        )

        let embeddingEngine: MLXEmbedderContainerEngine?
        if let embeddingModelID = configuration.embeddingModel {
            let embeddingModel = try await MLXServerEmbedderLoader.load(
                configuration: modelConfiguration(for: embeddingModelID)
            )
            embeddingEngine = MLXEmbedderContainerEngine(
                modelID: embeddingModelID,
                model: embeddingModel
            )
        } else {
            embeddingEngine = nil
        }
        let service = MLXOpenAIService(
            engine: engine,
            embeddingEngine: embeddingEngine,
            defaultReasoningParser: configuration.reasoningParser
        )
        let app = MLXServerApplication.buildApplication(
            service: service,
            host: configuration.host,
            port: configuration.port
        )

        try await app.runService()
    }

    static func modelConfiguration(for idOrPath: String, revision: String = "main") -> ModelConfiguration {
        let expandedPath: String
        if idOrPath == "~" {
            expandedPath = FileManager.default.homeDirectoryForCurrentUser.path
        } else if idOrPath.hasPrefix("~/") {
            expandedPath = FileManager.default.homeDirectoryForCurrentUser
                .appending(path: String(idOrPath.dropFirst(2)))
                .path
        } else {
            expandedPath = idOrPath
        }

        let url = URL(fileURLWithPath: expandedPath)
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: expandedPath, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            return .init(directory: url)
        }

        return .init(id: idOrPath, revision: revision)
    }
}
