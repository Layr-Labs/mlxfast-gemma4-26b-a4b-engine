// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

struct Gemma4AssistantValidatedGeometry {
    let preProjectionInputSize: Int
}

struct Gemma4AssistantTextConfigurationPreflight: Decodable {
    let numHiddenLayers: Int
    let slidingWindowPattern: Int

    enum CodingKeys: String, CodingKey {
        case numHiddenLayers = "num_hidden_layers"
        case slidingWindowPattern = "sliding_window_pattern"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 35
        self.slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
    }
}

enum Gemma4AssistantConfigurationValidator {
    static let supportedModelType = "gemma4_assistant"
    static let supportedTextModelType = "gemma4_text"
    static let maximumConfigBytes = 1 << 20

    private enum Limit {
        static let backboneHiddenSize = 8_192
        static let hiddenSize = 4_096
        static let intermediateSize = 65_536
        static let vocabSize = 524_288
        static let hiddenLayers = 16
        static let attentionHeads = 128
        static let keyValueHeads = 128
        static let headDimension = 2_048
        static let centroids = 16_384
        static let centroidTopK = 1_024
        static let hiddenSizePerLayerInput = 4_096
        static let slidingWindow = 1_048_576
        static let positionEmbeddings = 4_194_304
        static let tensorElements = 536_870_912
        static let totalRepeatedElements = 1_610_612_736
    }

    static func validatePreflight(_ config: Gemma4AssistantTextConfigurationPreflight) throws {
        try positive(
            config.numHiddenLayers,
            field: "textConfig.numHiddenLayers",
            maximum: Limit.hiddenLayers)
        try positive(
            config.slidingWindowPattern,
            field: "textConfig.slidingWindowPattern",
            maximum: Limit.hiddenLayers)
    }

    @discardableResult
    static func validate(
        _ config: Gemma4AssistantConfiguration,
        quantization: BaseConfiguration.PerLayerQuantization? = nil
    ) throws -> Gemma4AssistantValidatedGeometry {
        guard config.modelType == supportedModelType else {
            throw invalid(
                "modelType",
                "expected \(supportedModelType), got \(config.modelType)")
        }

        try positive(
            config.backboneHiddenSize,
            field: "backboneHiddenSize",
            maximum: Limit.backboneHiddenSize,
            checkMultiplier: 2)
        let preProjectionInputSize = try product(
            config.backboneHiddenSize,
            2,
            field: "2*backboneHiddenSize",
            maximum: Limit.backboneHiddenSize * 2)
        try positive(
            config.numCentroids,
            field: "numCentroids",
            maximum: Limit.centroids)
        try positive(
            config.centroidIntermediateTopK,
            field: "centroidIntermediateTopK",
            maximum: Limit.centroidTopK)
        guard config.centroidIntermediateTopK <= config.numCentroids else {
            throw invalid(
                "centroidIntermediateTopK",
                "must not exceed numCentroids")
        }
        guard (2 ... 16).contains(config.blockSize) else {
            throw invalid("blockSize", "must be between 2 and 16")
        }

        let text = config.textConfig
        guard text.modelType == supportedTextModelType else {
            throw invalid(
                "textConfig.modelType",
                "expected \(supportedTextModelType), got \(text.modelType)")
        }
        try positive(text.hiddenSize, field: "textConfig.hiddenSize", maximum: Limit.hiddenSize)
        try positive(
            text.intermediateSize,
            field: "textConfig.intermediateSize",
            maximum: Limit.intermediateSize,
            checkMultiplier: text.useDoubleWideMlp ? 2 : 1)
        try positive(
            text.vocabSize,
            field: "textConfig.vocabSize",
            maximum: Limit.vocabSize)
        try nonNegative(
            text.vocabSizePerLayerInput,
            field: "textConfig.vocabSizePerLayerInput",
            maximum: Limit.vocabSize)
        try positive(
            text.numHiddenLayers,
            field: "textConfig.numHiddenLayers",
            maximum: Limit.hiddenLayers)
        try positive(
            text.numAttentionHeads,
            field: "textConfig.numAttentionHeads",
            maximum: Limit.attentionHeads)
        try positive(
            text.numKeyValueHeads,
            field: "textConfig.numKeyValueHeads",
            maximum: Limit.keyValueHeads)
        if let globalHeads = text.numGlobalKeyValueHeads {
            try positive(
                globalHeads,
                field: "textConfig.numGlobalKeyValueHeads",
                maximum: Limit.keyValueHeads)
        }
        try positive(text.headDim, field: "textConfig.headDim", maximum: Limit.headDimension)
        try positive(
            text.globalHeadDim,
            field: "textConfig.globalHeadDim",
            maximum: Limit.headDimension)
        try nonNegative(
            text.hiddenSizePerLayerInput,
            field: "textConfig.hiddenSizePerLayerInput",
            maximum: Limit.hiddenSizePerLayerInput)
        try positive(
            text.slidingWindow,
            field: "textConfig.slidingWindow",
            maximum: Limit.slidingWindow)
        try positive(
            text.slidingWindowPattern,
            field: "textConfig.slidingWindowPattern",
            maximum: Limit.hiddenLayers)
        try positive(
            text.maxPositionEmbeddings,
            field: "textConfig.maxPositionEmbeddings",
            maximum: Limit.positionEmbeddings)

        guard text.numKvSharedLayers == text.numHiddenLayers else {
            throw invalid(
                "textConfig.numKvSharedLayers",
                "assistant layers must all consume shared KV")
        }
        guard text.layerTypes.count == text.numHiddenLayers else {
            throw invalid(
                "textConfig.layerTypes",
                "count must equal numHiddenLayers")
        }
        for (index, layerType) in text.layerTypes.enumerated() {
            guard layerType == "sliding_attention" || layerType == "full_attention" else {
                throw invalid(
                    "textConfig.layerTypes[\(index)]",
                    "unsupported attention type \(layerType)")
            }
        }

        try positiveFinite(text.rmsNormEps, field: "textConfig.rmsNormEps")
        try rotaryFactor(
            text.globalPartialRotaryFactor,
            dimensions: text.headDim,
            field: "textConfig.globalPartialRotaryFactor")
        try positiveFinite(text.slidingRopeTheta, field: "textConfig.slidingRopeTheta")
        try positiveFinite(text.fullRopeTheta, field: "textConfig.fullRopeTheta")
        try rotaryFactor(
            text.fullPartialRotaryFactor,
            dimensions: text.globalHeadDim,
            field: "textConfig.fullPartialRotaryFactor")

        if text.enableMoeBlock {
            throw invalid(
                "textConfig.enableMoeBlock",
                "Gemma 4 assistant trunks must be dense")
        }

        let hasSliding = text.layerTypes.contains("sliding_attention")
        let hasFull = text.layerTypes.contains("full_attention")
        if hasSliding {
            try divides(
                text.numKeyValueHeads,
                into: text.numAttentionHeads,
                field: "textConfig.numKeyValueHeads")
        }
        if hasFull {
            if text.attentionKeqV && text.numGlobalKeyValueHeads == nil {
                throw invalid(
                    "textConfig.numGlobalKeyValueHeads",
                    "is required when full attention uses K=V")
            }
            let fullKVHeads = text.numGlobalKeyValueHeads ?? text.numKeyValueHeads
            try divides(
                fullKVHeads,
                into: text.numAttentionHeads,
                field: "textConfig.fullAttentionKeyValueHeads")
        }

        let effectiveIntermediate = try product(
            text.intermediateSize,
            text.useDoubleWideMlp ? 2 : 1,
            field: "textConfig.effectiveIntermediateSize",
            maximum: Limit.intermediateSize * 2)
        let embeddingElements = try product(
            text.vocabSize,
            text.hiddenSize,
            field: "textConfig.vocabSize*hiddenSize",
            maximum: Limit.tensorElements)
        _ = embeddingElements
        let mlpElements = try product(
            text.hiddenSize,
            effectiveIntermediate,
            field: "textConfig.hiddenSize*effectiveIntermediateSize",
            maximum: Limit.tensorElements)
        let repeatedMLPElements = try product(
            mlpElements,
            text.numHiddenLayers,
            field: "textConfig.mlpElements*numHiddenLayers",
            maximum: Limit.totalRepeatedElements)
        _ = try product(
            repeatedMLPElements,
            3,
            field: "textConfig.totalMLPProjectionElements",
            maximum: Limit.totalRepeatedElements)
        _ = try product(
            config.backboneHiddenSize,
            text.hiddenSize,
            field: "backboneHiddenSize*textConfig.hiddenSize",
            maximum: Limit.tensorElements)

        if hasSliding {
            try validateAttentionProducts(
                heads: text.numAttentionHeads,
                kvHeads: text.numKeyValueHeads,
                dimension: text.headDim,
                field: "textConfig.slidingAttention")
        }
        if hasFull {
            let fullKVHeads = text.numGlobalKeyValueHeads ?? text.numKeyValueHeads
            try validateAttentionProducts(
                heads: text.numAttentionHeads,
                kvHeads: fullKVHeads,
                dimension: text.globalHeadDim,
                field: "textConfig.fullAttention")
        }

        if text.hiddenSizePerLayerInput > 0 {
            guard text.vocabSizePerLayerInput > 0 else {
                throw invalid(
                    "textConfig.vocabSizePerLayerInput",
                    "must be positive when per-layer input is enabled")
            }
            let perLayerWidth = try product(
                text.numHiddenLayers,
                text.hiddenSizePerLayerInput,
                field: "textConfig.numHiddenLayers*hiddenSizePerLayerInput",
                maximum: Limit.hiddenLayers * Limit.hiddenSizePerLayerInput)
            _ = try product(
                text.vocabSizePerLayerInput,
                perLayerWidth,
                field: "textConfig.perLayerEmbeddingElements",
                maximum: Limit.tensorElements)
            _ = try product(
                text.hiddenSize,
                perLayerWidth,
                field: "textConfig.perLayerProjectionElements",
                maximum: Limit.tensorElements)
        }

        if config.useOrderedEmbeddings {
            guard text.vocabSize % config.numCentroids == 0 else {
                throw invalid(
                    "numCentroids",
                    "must divide textConfig.vocabSize")
            }
            let vocabPerCentroid = text.vocabSize / config.numCentroids
            _ = try product(
                config.centroidIntermediateTopK,
                vocabPerCentroid,
                field: "centroidIntermediateTopK*vocabSizePerCentroid",
                maximum: Limit.vocabSize)
            _ = try product(
                text.hiddenSize,
                config.numCentroids,
                field: "textConfig.hiddenSize*numCentroids",
                maximum: Limit.tensorElements)
        }

        try validateNestedQuantization(text)
        try validateQuantization(quantization)

        return Gemma4AssistantValidatedGeometry(
            preProjectionInputSize: preProjectionInputSize)
    }

    private static func validateAttentionProducts(
        heads: Int,
        kvHeads: Int,
        dimension: Int,
        field: String
    ) throws {
        _ = try product(
            heads,
            dimension,
            field: "\(field).queryElements",
            maximum: Limit.tensorElements)
        _ = try product(
            kvHeads,
            dimension,
            field: "\(field).keyValueElements",
            maximum: Limit.tensorElements)
    }

    private static func validateNestedQuantization(
        _ text: Gemma4TextConfiguration
    ) throws {
        switch (text.quantizationBits, text.quantizationGroupSize) {
        case (nil, nil):
            return
        case (.some(let bits), .some(let groupSize)):
            if let violation = QuantizationGeometry.violation(
                in: .init(groupSize: groupSize, bits: bits))
            {
                throw invalid(
                    "textConfig.quantization\(violation.fieldSuffix)", violation.reason)
            }
        default:
            throw invalid(
                "textConfig.quantization",
                "bits and group_size must either both be present or both be absent")
        }
    }

    private static func validateQuantization(
        _ quantization: BaseConfiguration.PerLayerQuantization?
    ) throws {
        guard let quantization else { return }
        if let violation = QuantizationGeometry.violation(in: quantization) {
            throw invalid("quantization\(violation.fieldSuffix)", violation.reason)
        }
    }

    private static func positive(
        _ value: Int,
        field: String,
        maximum: Int,
        checkMultiplier: Int? = nil
    ) throws {
        guard value > 0 else {
            throw invalid(field, "must be positive")
        }
        if let checkMultiplier {
            let result = value.multipliedReportingOverflow(by: checkMultiplier)
            guard !result.overflow else {
                throw invalid(field, "overflows Int when multiplied by \(checkMultiplier)")
            }
        }
        guard value <= maximum else {
            throw invalid(field, "exceeds supported maximum \(maximum)")
        }
    }

    private static func nonNegative(
        _ value: Int,
        field: String,
        maximum: Int
    ) throws {
        guard value >= 0 else {
            throw invalid(field, "must not be negative")
        }
        guard value <= maximum else {
            throw invalid(field, "exceeds supported maximum \(maximum)")
        }
    }

    private static func positiveFinite(_ value: Float, field: String) throws {
        guard value.isFinite && value > 0 else {
            throw invalid(field, "must be finite and positive")
        }
    }

    private static func rotaryFactor(
        _ value: Float, dimensions: Int, field: String
    ) throws {
        guard value.isFinite, value > 0, value <= 1 else {
            throw invalid(field, "must be finite and in (0, 1]")
        }
        let halfAngles = value * Float(dimensions) / 2
        guard halfAngles.isFinite,
            halfAngles >= 1,
            halfAngles <= Float(Int.max)
        else {
            throw invalid(field, "produces an invalid rotary dimension")
        }
        let angles = Int(halfAngles)
        let rotated = angles.multipliedReportingOverflow(by: 2)
        guard !rotated.overflow,
            rotated.partialValue > 0,
            rotated.partialValue <= dimensions,
            rotated.partialValue.isMultiple(of: 2)
        else {
            throw invalid(field, "produces an unsupported rotary dimension")
        }
    }

    private static func divides(_ divisor: Int, into dividend: Int, field: String) throws {
        guard dividend % divisor == 0 else {
            throw invalid(field, "must divide numAttentionHeads")
        }
    }

    private static func product(
        _ lhs: Int,
        _ rhs: Int,
        field: String,
        maximum: Int
    ) throws -> Int {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        guard !result.overflow else {
            throw invalid(field, "overflows Int")
        }
        guard result.partialValue >= 0 && result.partialValue <= maximum else {
            throw invalid(field, "exceeds supported maximum \(maximum)")
        }
        return result.partialValue
    }

    private static func invalid(_ field: String, _ reason: String) -> Gemma4MTPError {
        .invalidConfiguration(field: field, reason: reason)
    }
}

struct Gemma4AssistantConfigurationDocument {
    let config: Gemma4AssistantConfiguration
    let baseConfiguration: BaseConfiguration

    static func read(from url: URL) throws -> Self {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(
            upToCount: Gemma4AssistantConfigurationValidator.maximumConfigBytes + 1) ?? Data()
        guard data.count <= Gemma4AssistantConfigurationValidator.maximumConfigBytes else {
            throw Gemma4MTPError.invalidConfiguration(
                field: "config.json",
                reason:
                    "exceeds \(Gemma4AssistantConfigurationValidator.maximumConfigBytes)-byte limit")
        }
        return try decode(data)
    }

    static func decode(_ data: Data) throws -> Self {
        do {
            let decoder = JSONDecoder.json5()
            let config = try decoder.decode(Gemma4AssistantConfiguration.self, from: data)
            let baseConfiguration = try JSONDecoder.json5().decode(
                BaseConfiguration.self, from: data)
            try Gemma4AssistantConfigurationValidator.validate(
                config,
                quantization: baseConfiguration.perLayerQuantization)
            return Self(config: config, baseConfiguration: baseConfiguration)
        } catch let error as Gemma4MTPError {
            throw error
        } catch {
            throw Gemma4MTPError.invalidConfiguration(
                field: "config.json",
                reason: "could not be decoded")
        }
    }
}

enum Gemma4MTPCompatibilityValidator {
    private enum AttentionType: String, CaseIterable {
        case sliding = "sliding_attention"
        case full = "full_attention"
    }

    static func validate(
        drafter: Gemma4AssistantConfiguration,
        target: Gemma4TextConfiguration
    ) throws {
        try equal(
            drafter.backboneHiddenSize,
            target.hiddenSize,
            field: "backboneHiddenSize")
        try equal(
            drafter.textConfig.vocabSize,
            target.vocabSize,
            field: "vocabSize")

        let drafterText = drafter.textConfig
        for type in AttentionType.allCases
        where drafterText.layerTypes.contains(type.rawValue) {
            guard captureLayer(of: type, in: target) != nil else {
                throw mismatch(
                    field: "captureLayer.\(type.rawValue)",
                    drafter: "required",
                    target: "missing from non-shared target layers")
            }

            switch type {
            case .sliding:
                try equal(
                    drafterText.slidingWindow,
                    target.slidingWindow,
                    field: "slidingAttention.slidingWindow")
                try equal(
                    drafterText.headDim,
                    target.headDim,
                    field: "slidingAttention.headDimension")
                try equal(
                    drafterText.numKeyValueHeads,
                    target.numKeyValueHeads,
                    field: "slidingAttention.effectiveKVHeads")

            case .full:
                try equal(
                    drafterText.globalHeadDim,
                    target.globalHeadDim,
                    field: "fullAttention.headDimension")
                try equal(
                    drafterText.attentionKeqV,
                    target.attentionKeqV,
                    field: "fullAttention.attentionKeqV")
                try equal(
                    effectiveFullKVHeads(drafterText),
                    effectiveFullKVHeads(target),
                    field: "fullAttention.effectiveKVHeads")
            }
        }
    }

    private static func captureLayer(
        of type: AttentionType,
        in config: Gemma4TextConfiguration
    ) -> Int? {
        let layerCount = min(max(0, config.numHiddenLayers), config.layerTypes.count)
        let sharedCount = min(max(0, config.numKvSharedLayers), layerCount)
        let nonSharedCount = layerCount - sharedCount
        return (0 ..< nonSharedCount).reversed().first {
            config.layerTypes[$0] == type.rawValue
        }
    }

    private static func effectiveFullKVHeads(_ config: Gemma4TextConfiguration) -> Int {
        config.numGlobalKeyValueHeads ?? config.numKeyValueHeads
    }

    private static func equal<T: Equatable & CustomStringConvertible>(
        _ drafter: T,
        _ target: T,
        field: String
    ) throws {
        guard drafter == target else {
            throw mismatch(
                field: field,
                drafter: drafter.description,
                target: target.description)
        }
    }

    private static func equalOptional<T: Equatable & CustomStringConvertible>(
        _ drafter: T?,
        _ target: T?,
        field: String
    ) throws {
        guard drafter == target else {
            throw mismatch(
                field: field,
                drafter: drafter?.description ?? "nil",
                target: target?.description ?? "nil")
        }
    }

    private static func mismatch(
        field: String,
        drafter: String,
        target: String
    ) -> Gemma4MTPError {
        .incompatibleDrafter(field: field, drafter: drafter, target: target)
    }
}
