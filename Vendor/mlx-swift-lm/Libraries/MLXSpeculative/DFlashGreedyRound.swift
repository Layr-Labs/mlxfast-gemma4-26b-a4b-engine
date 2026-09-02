// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// PORT NOTE (engine, 2026-08-25). Ported verbatim from
// Layr-Labs/mlx-swift-lm `origin/dflash-framework-updates` @ d41c3003 EXCEPT
// for the `phaseAccumulator: DFlashPhaseAccumulator?` parameter and every
// `dflashTimingStart` / `dflashRecord` call site, which are dropped here.
// That accumulator is defined in `DFlashBenchmark.swift`, a bench-only file
// this engine does not vendor: carrying the parameter would mean vendoring a
// benchmark type with no caller, and the engine measures through benchd, not
// through this round's internal phase split. The accept/reject arithmetic,
// the draft-cache trim, the verify shape and the rollback are unchanged.

/// PUBLIC in this vendored copy (the fork keeps it `internal`). The engine's
/// runtime worker drives DFlash rounds itself rather than through
/// `DFlashTokenIterator`, because the benchd free-run wire surface needs the
/// PER-ROUND committed width and draft counters — `acceptance_lengths`,
/// `drafted_total`, `accepted_total` — which an `IteratorProtocol` that
/// yields one token at a time cannot report.
public struct DFlashGreedyRoundResult {
    public let accepted: Int
    public let tokens: [Int]
    public let bonus: Int
    public let targetHidden: MLXArray
}

private let dFlashCPUAcceptWalk: Bool = {
    switch ProcessInfo.processInfo.environment["MLX_DFLASH_CPU_ACCEPT_WALK"]?.lowercased() {
    case "0", "false", "no", "off":
        return false
    default:
        return true
    }
}()

/// Construction-bound target-cache lifecycle for one DFlash verify round.
///
/// The ordinary public overload below installs the legacy target-model
/// rollback implementation. Runtimes whose target cache has a stronger
/// transactional contract can call the generic overload directly, avoiding
/// an eligibility/fallback branch inside their enabled verify path.
public protocol DFlashTargetCacheRoundTransaction: AnyObject {
    func beginRound()

    /// Execute the construction-bound target verifier for this transaction.
    /// The ordinary transaction binds the model's direct forward; specialized
    /// runtimes can bind a compiled fixed-state forward without placing an
    /// eligibility-or-fallback decision in the measured round.
    func forwardGreedy(
        target: any DFlashTargetModel,
        verifyInput: MLXArray,
        cache: [KVCache],
        targetLayerIds: [Int]
    ) throws -> DFlashGreedyTargetForward

    /// Evaluation roots not already guaranteed by `targetTokens`. CBv2 uses
    /// these to collapse its on-device position chain and explicit ring-write
    /// fences at the same barrier that feeds the accept walk.
    func evaluationRoots(cache: [KVCache]) -> [MLXArray]

    func finishRound(
        target: any DFlashTargetModel,
        cache: inout [KVCache],
        verifyInput: MLXArray,
        acceptedTokenCount: Int,
        rejectedTokenCount: Int,
        targetLayerIds: [Int],
        verifiedTargetHidden: MLXArray
    ) throws -> MLXArray
}

private final class LegacyDFlashTargetCacheRoundTransaction:
    DFlashTargetCacheRoundTransaction
{
    private let rollbackProvider: (any DFlashTargetCacheRollbackProvider)?
    private let rollbackState: (any DFlashTargetRollbackState)?

    init(target: any DFlashTargetModel, cache: [KVCache]) {
        let provider = target as? any DFlashTargetCacheRollbackProvider
        self.rollbackProvider = provider
        self.rollbackState =
            provider?.makeDFlashCacheRollbackState(cache: cache)
            ?? target.makeDefaultDFlashCacheRollbackState(cache: cache)
    }

    func beginRound() {}

    func forwardGreedy(
        target: any DFlashTargetModel,
        verifyInput: MLXArray,
        cache: [KVCache],
        targetLayerIds: [Int]
    ) throws -> DFlashGreedyTargetForward {
        try target.forwardGreedyTokensForDFlash(
            verifyInput, cache: cache, targetLayerIds: targetLayerIds)
    }

    func evaluationRoots(cache: [KVCache]) -> [MLXArray] { [] }

    func finishRound(
        target: any DFlashTargetModel,
        cache: inout [KVCache],
        verifyInput: MLXArray,
        acceptedTokenCount: Int,
        rejectedTokenCount: Int,
        targetLayerIds: [Int],
        verifiedTargetHidden: MLXArray
    ) throws -> MLXArray {
        if let rollbackProvider {
            return try rollbackProvider.rollbackDFlashCache(
                &cache,
                state: rollbackState,
                verifyInput: verifyInput,
                acceptedTokenCount: acceptedTokenCount,
                rejectedTokenCount: rejectedTokenCount,
                targetLayerIds: targetLayerIds,
                verifiedTargetHidden: verifiedTargetHidden)
        }
        return try target.rollbackDFlashCacheUsingDefault(
            &cache,
            state: rollbackState,
            verifyInput: verifyInput,
            acceptedTokenCount: acceptedTokenCount,
            rejectedTokenCount: rejectedTokenCount,
            targetLayerIds: targetLayerIds,
            verifiedTargetHidden: verifiedTargetHidden)
    }
}

public func runDFlashGreedyRound(
    target: any DFlashTargetModel,
    drafter: DFlashDraftModel,
    targetCache: inout [KVCache],
    draftCache: [KVCache],
    bonus: Int,
    projectedContext: MLXArray,
    promptTokenCount: Int,
    generatedTokenCount: Int,
    blockSize: Int,
    maxEmitCount: Int
) throws -> DFlashGreedyRoundResult {
    let transaction = LegacyDFlashTargetCacheRoundTransaction(
        target: target, cache: targetCache)
    return try runDFlashGreedyRound(
        target: target,
        drafter: drafter,
        targetCache: &targetCache,
        draftCache: draftCache,
        bonus: bonus,
        projectedContext: projectedContext,
        promptTokenCount: promptTokenCount,
        generatedTokenCount: generatedTokenCount,
        blockSize: blockSize,
        maxEmitCount: maxEmitCount,
        targetCacheTransaction: transaction)
}

/// Direct transactional DFlash round. The caller installs one concrete
/// transaction at session construction, so this path contains no
/// eligible-or-legacy fallback decision.
public func runDFlashGreedyRound<Transaction: DFlashTargetCacheRoundTransaction>(
    target: any DFlashTargetModel,
    drafter: DFlashDraftModel,
    targetCache: inout [KVCache],
    draftCache: [KVCache],
    bonus: Int,
    projectedContext: MLXArray,
    promptTokenCount: Int,
    generatedTokenCount: Int,
    blockSize: Int,
    maxEmitCount: Int,
    targetCacheTransaction: Transaction
) throws -> DFlashGreedyRoundResult {
    guard blockSize >= 2 else {
        throw DFlashError.invalidBlockSize(blockSize)
    }

    let draftTokens = try drafter.draftBlock(
        bonus: bonus,
        projectedContext: projectedContext,
        cache: draftCache,
        blockSize: blockSize
    )
    asyncEval(draftTokens)

    let committedDraftOffset = Swift.max(0, promptTokenCount + generatedTokenCount - 1)
    if let draftOffset = draftCache.first?.offset {
        let extraDraftContext = draftOffset - committedDraftOffset
        if extraDraftContext > 0 {
            let trimmed = trimPromptCache(draftCache, numTokens: extraDraftContext)
            if trimmed != extraDraftContext {
                throw DFlashError.untrimmableCache
            }
        }
    }

    let bonusColumn = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
    let verifyInput = concatenated([bonusColumn, draftTokens], axis: 1)
    targetCacheTransaction.beginRound()
    let verifyOut: DFlashGreedyTargetForward =
        try DFlashTargetRuntimeOptions.withSmallRowVerifyFusionsEnabled {
            try targetCacheTransaction.forwardGreedy(
                target: target,
                verifyInput: verifyInput,
                cache: targetCache,
                targetLayerIds: drafter.config.targetLayerIds)
        }
    let targetTokens = verifyOut.tokens
    let verifiedTokenCount = targetTokens.dim(1)
    let draftTokenIds = draftTokens.squeezed(axis: 0)
    let targetTokenIds = targetTokens.squeezed(axis: 0)
    let proposedCount = Swift.max(0, blockSize - 1)
    let comparableCount = Swift.min(proposedCount, verifiedTokenCount)
    let cacheEvaluationRoots = targetCacheTransaction.evaluationRoots(cache: targetCache)
    let walkedAccepted: Int
    let emitted: [Int]
    let accepted: Int
    if dFlashCPUAcceptWalk {
        eval([targetTokens, draftTokens] + cacheEvaluationRoots)
        let targetReadCount = Swift.min(
            verifiedTokenCount,
            Swift.max(comparableCount, Swift.min(maxEmitCount, comparableCount + 1)))
        let targetIds =
            targetReadCount > 0
            ? targetTokenIds[0 ..< targetReadCount].asArray(Int32.self)
            : []
        let draftIds =
            comparableCount > 0
            ? draftTokenIds[0 ..< comparableCount].asArray(Int32.self)
            : []

        var prefix = 0
        while prefix < comparableCount, draftIds[prefix] == targetIds[prefix] {
            prefix += 1
        }
        walkedAccepted = prefix
        let walkedTokenCount = walkedAccepted + 1
        let emittedCount = Swift.min(maxEmitCount, walkedTokenCount, verifiedTokenCount)
        emitted = targetIds.prefix(emittedCount).map { Int($0) }
        accepted =
            emittedCount < walkedTokenCount
            ? Swift.max(0, emittedCount - 1)
            : walkedAccepted
    } else {
        let acceptedArray: MLXArray?
        if comparableCount == 0 {
            acceptedArray = nil
        } else {
            let targetPrefix = targetTokenIds[0 ..< comparableCount]
            let draftPrefix = draftTokenIds[0 ..< comparableCount]
            let matches = (draftPrefix .== targetPrefix).asType(.int32)
            let prefixMatches = matches.cumprod(axis: 0)
            acceptedArray = prefixMatches.sum()
        }
        if let acceptedArray {
            eval([targetTokens, draftTokens, acceptedArray] + cacheEvaluationRoots)
        } else {
            eval([targetTokens, draftTokens] + cacheEvaluationRoots)
        }

        walkedAccepted = acceptedArray.map { Int($0.item(Int32.self)) } ?? 0
        let walkedTokenCount = walkedAccepted + 1
        let emittedCount = Swift.min(maxEmitCount, walkedTokenCount, verifiedTokenCount)
        emitted =
            targetTokenIds[0 ..< emittedCount]
            .asArray(Int32.self)
            .map { Int($0) }
        accepted =
            emittedCount < walkedTokenCount
            ? Swift.max(0, emittedCount - 1)
            : walkedAccepted
    }

    let trim = Swift.max(0, verifiedTokenCount - accepted - 1)
    let verifiedTargetHidden: MLXArray
    if let hiddenStates = verifyOut.hiddenStates {
        let acceptedHiddenStates = hiddenStates.map {
            $0[0..., 0 ..< accepted + 1, 0...]
        }
        verifiedTargetHidden = acceptedHiddenStates.count == 1
            ? acceptedHiddenStates[0]
            : concatenated(acceptedHiddenStates, axis: -1)
    } else {
        verifiedTargetHidden = verifyOut.targetHidden
    }
    let nextTargetHidden: MLXArray
    nextTargetHidden = try targetCacheTransaction.finishRound(
        target: target,
        cache: &targetCache,
        verifyInput: verifyInput,
        acceptedTokenCount: accepted,
        rejectedTokenCount: trim,
        targetLayerIds: drafter.config.targetLayerIds,
        verifiedTargetHidden: verifiedTargetHidden)

    return DFlashGreedyRoundResult(
        accepted: accepted,
        tokens: emitted,
        bonus: emitted.last ?? bonus,
        targetHidden: nextTargetHidden
    )
}

/// Verify a construction-produced proposal block without executing the
/// DFlash draft model. The target accept walk and cache transaction are the
/// same arithmetic as `runDFlashGreedyRound`; only the proposal source is
/// already bound by the caller's explicit runtime phase route.
public func runDFlashGreedyProposalRound<
    Transaction: DFlashTargetCacheRoundTransaction
>(
    target: any DFlashTargetModel,
    targetCache: inout [KVCache],
    bonus: Int,
    draftTokens: MLXArray,
    targetLayerIds: [Int],
    blockSize: Int,
    maxEmitCount: Int,
    targetCacheTransaction: Transaction
) throws -> DFlashGreedyRoundResult {
    guard blockSize >= 2 else {
        throw DFlashError.invalidBlockSize(blockSize)
    }

    let bonusColumn = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
    let verifyInput = concatenated([bonusColumn, draftTokens], axis: 1)
    targetCacheTransaction.beginRound()
    let verifyOut: DFlashGreedyTargetForward =
        try DFlashTargetRuntimeOptions.withSmallRowVerifyFusionsEnabled {
            try targetCacheTransaction.forwardGreedy(
                target: target,
                verifyInput: verifyInput,
                cache: targetCache,
                targetLayerIds: targetLayerIds)
        }
    let targetTokens = verifyOut.tokens
    let verifiedTokenCount = targetTokens.dim(1)
    let draftTokenIds = draftTokens.squeezed(axis: 0)
    let targetTokenIds = targetTokens.squeezed(axis: 0)
    let proposedCount = Swift.max(0, blockSize - 1)
    let comparableCount = Swift.min(proposedCount, verifiedTokenCount)
    let cacheEvaluationRoots = targetCacheTransaction.evaluationRoots(cache: targetCache)
    let walkedAccepted: Int
    let emitted: [Int]
    let accepted: Int
    if dFlashCPUAcceptWalk {
        eval([targetTokens, draftTokens] + cacheEvaluationRoots)
        let targetReadCount = Swift.min(
            verifiedTokenCount,
            Swift.max(comparableCount, Swift.min(maxEmitCount, comparableCount + 1)))
        let targetIds =
            targetReadCount > 0
            ? targetTokenIds[0 ..< targetReadCount].asArray(Int32.self)
            : []
        let draftIds =
            comparableCount > 0
            ? draftTokenIds[0 ..< comparableCount].asArray(Int32.self)
            : []

        var prefix = 0
        while prefix < comparableCount, draftIds[prefix] == targetIds[prefix] {
            prefix += 1
        }
        walkedAccepted = prefix
        let walkedTokenCount = walkedAccepted + 1
        let emittedCount = Swift.min(maxEmitCount, walkedTokenCount, verifiedTokenCount)
        emitted = targetIds.prefix(emittedCount).map { Int($0) }
        accepted =
            emittedCount < walkedTokenCount
            ? Swift.max(0, emittedCount - 1)
            : walkedAccepted
    } else {
        let acceptedArray: MLXArray?
        if comparableCount == 0 {
            acceptedArray = nil
        } else {
            let targetPrefix = targetTokenIds[0 ..< comparableCount]
            let draftPrefix = draftTokenIds[0 ..< comparableCount]
            let matches = (draftPrefix .== targetPrefix).asType(.int32)
            let prefixMatches = matches.cumprod(axis: 0)
            acceptedArray = prefixMatches.sum()
        }
        if let acceptedArray {
            eval([targetTokens, draftTokens, acceptedArray] + cacheEvaluationRoots)
        } else {
            eval([targetTokens, draftTokens] + cacheEvaluationRoots)
        }

        walkedAccepted = acceptedArray.map { Int($0.item(Int32.self)) } ?? 0
        let walkedTokenCount = walkedAccepted + 1
        let emittedCount = Swift.min(maxEmitCount, walkedTokenCount, verifiedTokenCount)
        emitted =
            targetTokenIds[0 ..< emittedCount]
            .asArray(Int32.self)
            .map { Int($0) }
        accepted =
            emittedCount < walkedTokenCount
            ? Swift.max(0, emittedCount - 1)
            : walkedAccepted
    }

    let trim = Swift.max(0, verifiedTokenCount - accepted - 1)
    let verifiedTargetHidden: MLXArray
    if let hiddenStates = verifyOut.hiddenStates {
        let acceptedHiddenStates = hiddenStates.map {
            $0[0..., 0 ..< accepted + 1, 0...]
        }
        verifiedTargetHidden = acceptedHiddenStates.count == 1
            ? acceptedHiddenStates[0]
            : concatenated(acceptedHiddenStates, axis: -1)
    } else {
        verifiedTargetHidden = verifyOut.targetHidden
    }
    let nextTargetHidden = try targetCacheTransaction.finishRound(
        target: target,
        cache: &targetCache,
        verifyInput: verifyInput,
        acceptedTokenCount: accepted,
        rejectedTokenCount: trim,
        targetLayerIds: targetLayerIds,
        verifiedTargetHidden: verifiedTargetHidden)

    return DFlashGreedyRoundResult(
        accepted: accepted,
        tokens: emitted,
        bonus: emitted.last ?? bonus,
        targetHidden: nextTargetHidden)
}
