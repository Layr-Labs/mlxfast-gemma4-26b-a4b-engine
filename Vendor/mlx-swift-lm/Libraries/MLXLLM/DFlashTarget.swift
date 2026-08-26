// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Per-thread target-side DFlash options used while constructing verify graphs.
///
/// The batched DFlash path intentionally disables some single-row Gemma4
/// verify fusions because a live request can be processed as a single-row
/// subgroup even while it still shares cache state with other requests.
public enum DFlashTargetRuntimeOptions {
    private static let disabledSmallRowVerifyFusionsKey =
        "mlxswiftlm.dflash.disableSmallRowVerifyFusions"
    private static let enabledSmallRowVerifyFusionsKey =
        "mlxswiftlm.dflash.enableSmallRowVerifyFusions"

    public static var smallRowVerifyFusionsDisabled: Bool {
        Thread.current.threadDictionary[disabledSmallRowVerifyFusionsKey] as? Bool ?? false
    }

    public static var smallRowVerifyFusionsEnabled: Bool {
        Thread.current.threadDictionary[enabledSmallRowVerifyFusionsKey] as? Bool ?? false
    }

    public static func withSmallRowVerifyFusionsEnabled<T>(
        _ body: () throws -> T
    ) rethrows -> T {
        let dictionary = Thread.current.threadDictionary
        let previous = dictionary[enabledSmallRowVerifyFusionsKey]
        dictionary[enabledSmallRowVerifyFusionsKey] = true
        defer {
            if let previous {
                dictionary[enabledSmallRowVerifyFusionsKey] = previous
            } else {
                dictionary.removeObject(forKey: enabledSmallRowVerifyFusionsKey)
            }
        }
        return try body()
    }

    public static func withSmallRowVerifyFusionsDisabled<T>(
        _ body: () throws -> T
    ) rethrows -> T {
        let dictionary = Thread.current.threadDictionary
        let previous = dictionary[disabledSmallRowVerifyFusionsKey]
        dictionary[disabledSmallRowVerifyFusionsKey] = true
        defer {
            if let previous {
                dictionary[disabledSmallRowVerifyFusionsKey] = previous
            } else {
                dictionary.removeObject(forKey: disabledSmallRowVerifyFusionsKey)
            }
        }
        return try body()
    }
}

/// Target-model output used by DFlash speculative decoding.
///
/// `targetHidden` is the concatenation of the selected post-layer hidden
/// states in the exact order requested by the DFlash draft configuration.
public struct DFlashTargetForward: @unchecked Sendable {
    public let logits: MLXArray
    public let hiddenStates: [MLXArray]
    public let targetHidden: MLXArray

    public init(logits: MLXArray, hiddenStates: [MLXArray]) {
        self.logits = logits
        self.hiddenStates = hiddenStates
        self.targetHidden = hiddenStates.count == 1
            ? hiddenStates[0]
            : concatenated(hiddenStates, axis: -1)
    }
}

/// Greedy target-model output used by DFlash when only top-1 tokens are needed.
public struct DFlashGreedyTargetForward: @unchecked Sendable {
    public let tokens: MLXArray
    public let targetHidden: MLXArray
    public let verifyTimings: DFlashTargetVerifyTimings?

    public init(
        tokens: MLXArray,
        targetHidden: MLXArray,
        verifyTimings: DFlashTargetVerifyTimings? = nil
    ) {
        self.tokens = tokens
        self.targetHidden = targetHidden
        self.verifyTimings = verifyTimings
    }
}

/// Diagnostic target-side timing split for DFlash greedy verification.
///
/// These timings intentionally force additional eval barriers and should only
/// be used for profiling, not for normal throughput measurements.
public struct DFlashTargetVerifyTimings: Sendable {
    public let trunkSeconds: Double
    public let hiddenConcatSeconds: Double
    public let lmHeadSeconds: Double
    public let softcapArgmaxSeconds: Double
    public let trunkEmbeddingSeconds: Double
    public let trunkPLESeconds: Double
    public let trunkMaskSeconds: Double
    public let trunkAttentionSeconds: Double
    public let trunkDenseMLPSeconds: Double
    public let trunkRouterSeconds: Double
    public let trunkExpertsSeconds: Double
    public let trunkPLEGateSeconds: Double
    public let trunkFinalNormSeconds: Double

    public init(
        trunkSeconds: Double,
        hiddenConcatSeconds: Double,
        lmHeadSeconds: Double,
        softcapArgmaxSeconds: Double,
        trunkEmbeddingSeconds: Double = 0,
        trunkPLESeconds: Double = 0,
        trunkMaskSeconds: Double = 0,
        trunkAttentionSeconds: Double = 0,
        trunkDenseMLPSeconds: Double = 0,
        trunkRouterSeconds: Double = 0,
        trunkExpertsSeconds: Double = 0,
        trunkPLEGateSeconds: Double = 0,
        trunkFinalNormSeconds: Double = 0
    ) {
        self.trunkSeconds = trunkSeconds
        self.hiddenConcatSeconds = hiddenConcatSeconds
        self.lmHeadSeconds = lmHeadSeconds
        self.softcapArgmaxSeconds = softcapArgmaxSeconds
        self.trunkEmbeddingSeconds = trunkEmbeddingSeconds
        self.trunkPLESeconds = trunkPLESeconds
        self.trunkMaskSeconds = trunkMaskSeconds
        self.trunkAttentionSeconds = trunkAttentionSeconds
        self.trunkDenseMLPSeconds = trunkDenseMLPSeconds
        self.trunkRouterSeconds = trunkRouterSeconds
        self.trunkExpertsSeconds = trunkExpertsSeconds
        self.trunkPLEGateSeconds = trunkPLEGateSeconds
        self.trunkFinalNormSeconds = trunkFinalNormSeconds
    }
}

public enum DFlashTargetError: LocalizedError, Sendable, Equatable {
    case emptyTargetLayerIds
    case duplicateTargetLayerIds([Int])
    case targetLayerOutOfRange(layerId: Int, layerCount: Int)
    case untrimmableCache

    public var errorDescription: String? {
        switch self {
        case .emptyTargetLayerIds:
            return "DFlash target hidden capture requires at least one target layer id."
        case .duplicateTargetLayerIds(let ids):
            return "DFlash target layer ids must be unique; got \(ids)."
        case .targetLayerOutOfRange(let layerId, let layerCount):
            return
                "DFlash target layer id \(layerId) is outside the valid range 0..<\(layerCount)."
        case .untrimmableCache:
            return "DFlash target cache could not be rolled back after speculative rejection."
        }
    }
}

/// Opaque rollback checkpoint captured before the target verifies a DFlash
/// draft block. Target implementations can return their own state type for
/// optimized rollback without changing the generation loop.
public protocol DFlashTargetRollbackState {}

public struct DFlashCopiedTargetRollbackState: DFlashTargetRollbackState {
    public let cache: [KVCache]

    public init(cache: [KVCache]) {
        self.cache = cache
    }
}

/// Minimal target surface a DFlash drafter needs from a loaded target model.
///
/// Keep this in MLXLLM rather than MLXSpeculative so model implementations
/// can conform without reversing the package dependency direction.
public protocol DFlashTargetModel: LLMModel {
    var dFlashVocabularySize: Int { get }
    var dFlashHiddenSize: Int { get }
    var dFlashLayerCount: Int { get }

    func forwardForDFlash(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        targetLayerIds: [Int]
    ) throws -> DFlashTargetForward

    func forwardGreedyTokensForDFlash(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        targetLayerIds: [Int]
    ) throws -> DFlashGreedyTargetForward

    func embedTokensForDFlash(_ tokens: MLXArray) -> MLXArray
    /// Raw target LM-head projection for drafter hidden states. The DFlash
    /// drafter applies any config-level final-logit transform itself.
    func logitsForDFlashHidden(_ hidden: MLXArray) -> MLXArray
}

/// Optional target hook for optimized DFlash cache rollback.
///
/// Targets with trimmable caches can rely on the default rollback helpers.
/// Hybrid targets can conform to this protocol to avoid baking their cache
/// details into the DFlash generation loop.
public protocol DFlashTargetCacheRollbackProvider: DFlashTargetModel {
    func makeDFlashCacheRollbackState(cache: [KVCache]) -> (any DFlashTargetRollbackState)?

    func rollbackDFlashCache(
        _ cache: inout [KVCache],
        state: (any DFlashTargetRollbackState)?,
        verifyInput: MLXArray,
        acceptedTokenCount: Int,
        rejectedTokenCount: Int,
        targetLayerIds: [Int],
        verifiedTargetHidden: MLXArray
    ) throws -> MLXArray
}

/// Optional target hook that can expose a diagnostic timing split for greedy
/// verification. Implementations may insert eval barriers to make the split
/// meaningful, so callers should request this only in profiling modes.
public protocol DFlashTargetDiagnosticForwardProvider: DFlashTargetModel {
    func forwardGreedyTokensForDFlash(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        targetLayerIds: [Int],
        collectVerifyTimings: Bool
    ) throws -> DFlashGreedyTargetForward
}

extension DFlashTargetModel {
    public func forwardGreedyTokensForDFlash(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        targetLayerIds: [Int]
    ) throws -> DFlashGreedyTargetForward {
        let forward = try forwardForDFlash(
            inputs,
            cache: cache,
            targetLayerIds: targetLayerIds
        )
        return DFlashGreedyTargetForward(
            tokens: forward.logits.argMax(axis: -1),
            targetHidden: forward.targetHidden
        )
    }

    public func makeDefaultDFlashCacheRollbackState(
        cache: [KVCache]
    ) -> (any DFlashTargetRollbackState)? {
        canTrimPromptCache(cache) ? nil : DFlashCopiedTargetRollbackState(
            cache: cache.map { $0.copy() })
    }

    public func rollbackDFlashCacheUsingDefault(
        _ cache: inout [KVCache],
        state: (any DFlashTargetRollbackState)?,
        verifyInput: MLXArray,
        acceptedTokenCount: Int,
        rejectedTokenCount: Int,
        targetLayerIds: [Int],
        verifiedTargetHidden: MLXArray
    ) throws -> MLXArray {
        let acceptedHidden = verifiedTargetHidden[0..., 0 ..< acceptedTokenCount + 1, 0...]
        guard rejectedTokenCount > 0 else {
            return acceptedHidden
        }

        if canTrimPromptCache(cache) {
            let trimmed = trimPromptCache(cache, numTokens: rejectedTokenCount)
            guard trimmed == rejectedTokenCount else {
                throw DFlashTargetError.untrimmableCache
            }
            return acceptedHidden
        }

        guard let copiedState = state as? DFlashCopiedTargetRollbackState else {
            throw DFlashTargetError.untrimmableCache
        }

        cache = copiedState.cache
        let acceptedPrefix = verifyInput[0..., 0 ..< acceptedTokenCount + 1]
        let replay = try forwardForDFlash(
            acceptedPrefix,
            cache: cache,
            targetLayerIds: targetLayerIds
        )
        eval(replay.targetHidden)
        return replay.targetHidden
    }
}

public enum DFlashTargetValidation {
    public static func validateTargetLayerIds(_ ids: [Int], layerCount: Int) throws {
        guard !ids.isEmpty else {
            throw DFlashTargetError.emptyTargetLayerIds
        }

        guard Set(ids).count == ids.count else {
            throw DFlashTargetError.duplicateTargetLayerIds(ids)
        }

        for id in ids where id < 0 || id >= layerCount {
            throw DFlashTargetError.targetLayerOutOfRange(
                layerId: id, layerCount: layerCount)
        }
    }
}
